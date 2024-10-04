import Mathlib

namespace parabola_properties_l73_73952

-- Define the parabola function as y = x^2 + px + q
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p * x + q

-- Prove the properties of parabolas for varying p and q.
theorem parabola_properties (p q p' q' : ℝ) :
  (∀ x : ℝ, parabola p q x = x^2 + p * x + q) ∧
  (∀ x : ℝ, parabola p' q' x = x^2 + p' * x + q') →
  (∀ x : ℝ, ( ∃ k h : ℝ, parabola p q x = (x + h)^2 + k ) ∧ 
               ( ∃ k' h' : ℝ, parabola p' q' x = (x + h')^2 + k' ) ) ∧
  (∀ x : ℝ, h = -p / 2 ∧ k = q - p^2 / 4 ) ∧
  (∀ x : ℝ, h' = -p' / 2 ∧ k' = q' - p'^2 / 4 ) ∧
  (∀ x : ℝ, (h, k) ≠ (h', k') → parabola p q x ≠ parabola p' q' x) ∧
  (∀ x : ℝ, h = h' ∧ k = k' → parabola p q x = parabola p' q' x) :=
by
  sorry

end parabola_properties_l73_73952


namespace original_price_of_wand_l73_73134

theorem original_price_of_wand (x : ℝ) (h : x / 8 = 12) : x = 96 :=
by
  sorry

end original_price_of_wand_l73_73134


namespace factor_theorem_solution_l73_73351

theorem factor_theorem_solution (t : ℝ) :
  (∃ p q : ℝ, 10 * p * q = 10 * t * t + 21 * t - 10 ∧ (x - q) = (x - t)) →
  t = 2 / 5 ∨ t = -5 / 2 := by
  sorry

end factor_theorem_solution_l73_73351


namespace sqrt_inequality_l73_73160

theorem sqrt_inequality : (Real.sqrt 6 + Real.sqrt 7) > (2 * Real.sqrt 2 + Real.sqrt 5) :=
by {
  sorry
}

end sqrt_inequality_l73_73160


namespace lettuce_price_1_l73_73475

theorem lettuce_price_1 (customers_per_month : ℕ) (lettuce_per_customer : ℕ) (tomatoes_per_customer : ℕ) 
(price_per_tomato : ℝ) (total_sales : ℝ)
  (h_customers : customers_per_month = 500)
  (h_lettuce_per_customer : lettuce_per_customer = 2)
  (h_tomatoes_per_customer : tomatoes_per_customer = 4)
  (h_price_per_tomato : price_per_tomato = 0.5)
  (h_total_sales : total_sales = 2000) :
  let heads_of_lettuce_sold := customers_per_month * lettuce_per_customer
  let tomato_sales := customers_per_month * tomatoes_per_customer * price_per_tomato
  let lettuce_sales := total_sales - tomato_sales
  let price_per_lettuce := lettuce_sales / heads_of_lettuce_sold
  price_per_lettuce = 1 := by
{
  sorry
}

end lettuce_price_1_l73_73475


namespace square_perimeter_l73_73809

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l73_73809


namespace manu_wins_probability_l73_73819

def prob_manu_wins : ℚ :=
  let a := (1/2) ^ 5
  let r := (1/2) ^ 4
  a / (1 - r)

theorem manu_wins_probability : prob_manu_wins = 1 / 30 :=
  by
  -- here we would have the proof steps
  sorry

end manu_wins_probability_l73_73819


namespace find_c_l73_73732

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem find_c (a b c S : ℝ) (C : ℝ) 
  (ha : a = 3) 
  (hC : C = 120) 
  (hS : S = 15 * Real.sqrt 3 / 4) 
  (hab : a * b = 15)
  (hc2 : c^2 = a^2 + b^2 - 2 * a * b * cos_deg C) :
  c = 7 :=
by 
  sorry

end find_c_l73_73732


namespace volume_of_prism_l73_73172

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 100) (h2 : z = 10) (h3 : x * z = 50) (h4 : y * z = 40):
  x * y * z = 200 :=
by
  sorry

end volume_of_prism_l73_73172


namespace largest_initial_number_l73_73404

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l73_73404


namespace lcm_9_12_15_l73_73975

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l73_73975


namespace non_zero_real_positive_integer_l73_73357

theorem non_zero_real_positive_integer (x : ℝ) (h : x ≠ 0) : 
  (∃ k : ℤ, k > 0 ∧ (x - |x-1|) / x = k) ↔ x = 1 := 
sorry

end non_zero_real_positive_integer_l73_73357


namespace range_of_a_l73_73702

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x >= 2 then (a - 1 / 2) * x 
  else a^x - 4

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (1 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l73_73702


namespace waiting_time_boarding_l73_73416

noncomputable def time_taken_uber_to_house : ℕ := 10
noncomputable def time_taken_uber_to_airport : ℕ := 5 * time_taken_uber_to_house
noncomputable def time_taken_bag_check : ℕ := 15
noncomputable def time_taken_security : ℕ := 3 * time_taken_bag_check
noncomputable def total_process_time : ℕ := 180
noncomputable def remaining_time : ℕ := total_process_time - (time_taken_uber_to_house + time_taken_uber_to_airport + time_taken_bag_check + time_taken_security)
noncomputable def time_before_takeoff (B : ℕ) := 2 * B

theorem waiting_time_boarding : ∃ B : ℕ, B + time_before_takeoff B = remaining_time ∧ B = 20 := 
by 
  sorry

end waiting_time_boarding_l73_73416


namespace zero_one_law_for_gaussian_systems_l73_73743

def is_gaussian_random_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, is_gaussian (X n)

def linear_subspace_of_R_infty (L : Set (ℕ → ℝ)) : Prop :=
  ∀ f1 f2 ∈ L, ∀ (a b : ℝ), (λ n, a * f1 n + b * f2 n) ∈ L ∧ (λ n, f1 n - f2 n) ∈ L

theorem zero_one_law_for_gaussian_systems {X : ℕ → ℝ} {L : Set (ℕ → ℝ)} 
  (hX : is_gaussian_random_sequence X) 
  (hL : linear_subspace_of_R_infty L) : 
  (Prob (X ∈ L) = 0 ∨ Prob (X ∈ L) = 1) 
  ∧ (Prob (λ ω, ∃ n, |X ω n| < ∞) = 0 ∨ Prob (λ ω, ∃ n, |X ω n| < ∞) = 1) := 
sorry

end zero_one_law_for_gaussian_systems_l73_73743


namespace sin_cos_identity_l73_73870

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l73_73870


namespace unique_root_exists_maximum_value_lnx_l73_73230

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x

theorem unique_root_exists (k : ℝ) :
  ∃ a, a = 1 ∧ (∃ x ∈ Set.Ioo k (k+1), f x = g x) :=
sorry

theorem maximum_value_lnx (p q : ℝ) :
  (∃ x, (x = min p q) ∧ Real.log x = ( 4 / Real.exp 2 )) :=
sorry

end unique_root_exists_maximum_value_lnx_l73_73230


namespace problem_D_l73_73386

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b c : V}

def is_parallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

theorem problem_D (h₁ : is_parallel a b) (h₂ : is_parallel b c) (h₃ : b ≠ 0) : is_parallel a c :=
sorry

end problem_D_l73_73386


namespace boys_in_school_l73_73984

theorem boys_in_school (x : ℕ) (boys girls : ℕ) (h1 : boys = 5 * x) 
  (h2 : girls = 13 * x) (h3 : girls - boys = 128) : boys = 80 :=
by
  sorry

end boys_in_school_l73_73984


namespace lcm_9_12_15_l73_73955

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l73_73955


namespace find_p_q_r_l73_73850

def f (x : ℝ) : ℝ := x^2 + 2*x + 2
def g (x p q r : ℝ) : ℝ := x^3 + 2*x^2 + 6*p*x + 4*q*x + r

noncomputable def roots_sum_f := -2
noncomputable def roots_product_f := 2

theorem find_p_q_r (p q r : ℝ) (h1 : ∀ x, f x = 0 → g x p q r = 0) :
  (p + q) * r = 0 :=
sorry

end find_p_q_r_l73_73850


namespace total_pokemon_cards_l73_73020

-- Definitions based on conditions
def jenny_cards : ℕ := 6
def orlando_cards : ℕ := jenny_cards + 2
def richard_cards : ℕ := 3 * orlando_cards

-- The theorem stating the total number of cards
theorem total_pokemon_cards : jenny_cards + orlando_cards + richard_cards = 38 :=
by
  sorry

end total_pokemon_cards_l73_73020


namespace kylie_earrings_l73_73028

def number_of_necklaces_monday := 10
def number_of_necklaces_tuesday := 2
def number_of_bracelets_wednesday := 5
def beads_per_necklace := 20
def beads_per_bracelet := 10
def beads_per_earring := 5
def total_beads := 325

theorem kylie_earrings : 
    (total_beads - ((number_of_necklaces_monday + number_of_necklaces_tuesday) * beads_per_necklace + number_of_bracelets_wednesday * beads_per_bracelet)) / beads_per_earring = 7 :=
by
    sorry

end kylie_earrings_l73_73028


namespace minimum_BC_length_l73_73628

theorem minimum_BC_length (AB AC DC BD BC : ℕ)
  (h₁ : AB = 5) (h₂ : AC = 12) (h₃ : DC = 8) (h₄ : BD = 20) (h₅ : BC > 12) : BC = 13 :=
by
  sorry

end minimum_BC_length_l73_73628


namespace second_ball_probability_l73_73121

-- Definitions and conditions
def red_balls := 3
def white_balls := 2
def black_balls := 5
def total_balls := red_balls + white_balls + black_balls

def first_ball_white_condition : Prop := (white_balls / total_balls) = (2 / 10)
def second_ball_red_given_first_white (first_ball_white : Prop) : Prop :=
  (first_ball_white → (red_balls / (total_balls - 1)) = (1 / 3))

-- Mathematical equivalence proof problem statement in Lean
theorem second_ball_probability : 
  first_ball_white_condition ∧ second_ball_red_given_first_white first_ball_white_condition :=
by
  sorry

end second_ball_probability_l73_73121


namespace compute_c_plus_d_l73_73282

theorem compute_c_plus_d (c d : ℝ) 
  (h1 : c^3 - 18 * c^2 + 25 * c - 75 = 0) 
  (h2 : 9 * d^3 - 72 * d^2 - 345 * d + 3060 = 0) : 
  c + d = 10 := 
sorry

end compute_c_plus_d_l73_73282


namespace solve_system_l73_73716

theorem solve_system (x y : ℝ) (h1 : 4 * x - y = 2) (h2 : 3 * x - 2 * y = -1) : x - y = -1 := 
by
  sorry

end solve_system_l73_73716


namespace baker_price_l73_73501

theorem baker_price
  (P : ℝ)
  (h1 : 8 * P = 320)
  (h2 : 10 * (0.80 * P) = 320)
  : P = 40 := sorry

end baker_price_l73_73501


namespace downstream_distance_correct_l73_73630

-- Definitions based on the conditions
def still_water_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- The effective speed downstream is the sum of the still water speed and the stream speed
def effective_speed_downstream : ℝ := still_water_speed + stream_speed

-- The distance covered downstream is the product of effective speed and travel time
def downstream_distance : ℝ := effective_speed_downstream * travel_time

-- The theorem to be proven
theorem downstream_distance_correct : downstream_distance = 81 := by
  sorry

end downstream_distance_correct_l73_73630


namespace monthly_rate_is_24_l73_73645

noncomputable def weekly_rate : ℝ := 10
noncomputable def weeks_per_year : ℕ := 52
noncomputable def months_per_year : ℕ := 12
noncomputable def yearly_savings : ℝ := 232

theorem monthly_rate_is_24 (M : ℝ) (h : weeks_per_year * weekly_rate - months_per_year * M = yearly_savings) : 
  M = 24 :=
by
  sorry

end monthly_rate_is_24_l73_73645


namespace evaluate_expression_l73_73350

theorem evaluate_expression :
  let a := 3^1005
  let b := 4^1006
  (a + b)^2 - (a - b)^2 = 160 * 10^1004 :=
by
  sorry

end evaluate_expression_l73_73350


namespace lcm_9_12_15_l73_73974

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l73_73974


namespace volume_of_prism_l73_73812

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 10) (hwh : w * h = 15) (hlh : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 :=
by
  sorry

end volume_of_prism_l73_73812


namespace two_students_one_common_material_l73_73454

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l73_73454


namespace f_has_two_zeros_l73_73546

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_has_two_zeros (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := sorry

end f_has_two_zeros_l73_73546


namespace square_perimeter_l73_73804

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l73_73804


namespace jakes_present_weight_l73_73264

theorem jakes_present_weight (J S : ℕ) (h1 : J - 32 = 2 * S) (h2 : J + S = 212) : J = 152 :=
by
  sorry

end jakes_present_weight_l73_73264


namespace max_n_arithmetic_seq_sum_neg_l73_73362

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + ((n - 1) * d)

-- Define the terms of the sequence
def a₃ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 3
def a₆ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 6
def a₇ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 7

-- Condition: a₆ is the geometric mean of a₃ and a₇
def geometric_mean_condition (a₁ : ℤ) : Prop :=
  (a₃ a₁) * (a₇ a₁) = (a₆ a₁) * (a₆ a₁)

-- Sum of the first n terms of the arithmetic sequence
def S_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

-- The goal: the maximum value of n for which S_n < 0
theorem max_n_arithmetic_seq_sum_neg : 
  ∃ n : ℕ, ∀ k : ℕ, geometric_mean_condition (-13) →  S_n (-13) 2 k < 0 → n ≤ 13 := 
sorry

end max_n_arithmetic_seq_sum_neg_l73_73362


namespace value_of_m_l73_73559

theorem value_of_m (m : ℝ) : (3 = 2 * m + 1) → m = 1 :=
by
  intro h
  -- skipped proof due to requirement
  sorry

end value_of_m_l73_73559


namespace dispatch_plans_l73_73944

theorem dispatch_plans (students : Finset ℕ) (h : students.card = 6) :
  ∃ (plans : Finset (Finset ℕ)), plans.card = 180 :=
by
  sorry

end dispatch_plans_l73_73944


namespace temperature_drop_change_l73_73560

theorem temperature_drop_change (T : ℝ) (h1 : T + 2 = T + 2) :
  (T - 4) - T = -4 :=
by
  sorry

end temperature_drop_change_l73_73560


namespace vikki_worked_42_hours_l73_73071

-- Defining the conditions
def hourly_pay_rate : ℝ := 10
def tax_deduction : ℝ := 0.20 * hourly_pay_rate
def insurance_deduction : ℝ := 0.05 * hourly_pay_rate
def union_dues : ℝ := 5
def take_home_pay : ℝ := 310

-- Equation derived from the given conditions
def total_hours_worked (h : ℝ) : Prop :=
  hourly_pay_rate * h - (tax_deduction * h + insurance_deduction * h + union_dues) = take_home_pay

-- Prove that Vikki worked for 42 hours given the conditions
theorem vikki_worked_42_hours : total_hours_worked 42 := by
  sorry

end vikki_worked_42_hours_l73_73071


namespace n_m_odd_implies_sum_odd_l73_73379

theorem n_m_odd_implies_sum_odd {n m : ℤ} (h : Odd (n^2 + m^2)) : Odd (n + m) :=
by
  sorry

end n_m_odd_implies_sum_odd_l73_73379


namespace taxes_are_135_l73_73882

def gross_pay : ℕ := 450
def net_pay : ℕ := 315
def taxes_paid (G N: ℕ) : ℕ := G - N

theorem taxes_are_135 : taxes_paid gross_pay net_pay = 135 := by
  sorry

end taxes_are_135_l73_73882


namespace y_is_never_perfect_square_l73_73319

theorem y_is_never_perfect_square (x : ℕ) : ¬ ∃ k : ℕ, k^2 = x^4 + 2*x^3 + 2*x^2 + 2*x + 1 :=
sorry

end y_is_never_perfect_square_l73_73319


namespace intersections_vary_with_A_l73_73224

theorem intersections_vary_with_A (A : ℝ) (hA : A > 0) :
  ∃ x y : ℝ, (y = A * x^2) ∧ (y^2 + 2 = x^2 + 6 * y) ∧ (y = 2 * x - 1) :=
sorry

end intersections_vary_with_A_l73_73224


namespace find_other_number_l73_73495

open BigOperators

noncomputable def other_number (n : ℕ) : Prop := n = 12

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 8 n = 24) (h_hcf : Nat.gcd 8 n = 4) : other_number n := 
by
  sorry

end find_other_number_l73_73495


namespace ming_estimate_less_l73_73779

theorem ming_estimate_less (x y δ : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : δ > 0) : 
  (x + δ) - (y + 2 * δ) < x - y :=
by 
  sorry

end ming_estimate_less_l73_73779


namespace parallel_slope_l73_73479

theorem parallel_slope {x1 y1 x2 y2 : ℝ} (h : x1 = 3 ∧ y1 = -2 ∧ x2 = 1 ∧ y2 = 5) :
    let slope := (y2 - y1) / (x2 - x1)
    slope = -7 / 2 := 
by 
    sorry

end parallel_slope_l73_73479


namespace max_abs_ax_plus_b_l73_73256

theorem max_abs_ax_plus_b (a b c : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2 :=
by
  sorry

end max_abs_ax_plus_b_l73_73256


namespace pairs_equality_l73_73817

-- Define all the pairs as given in the problem.
def pairA_1 : ℤ := - (2^7)
def pairA_2 : ℤ := (-2)^7
def pairB_1 : ℤ := - (3^2)
def pairB_2 : ℤ := (-3)^2
def pairC_1 : ℤ := -3 * (2^3)
def pairC_2 : ℤ := - (3^2) * 2
def pairD_1 : ℤ := -((-3)^2)
def pairD_2 : ℤ := -((-2)^3)

-- The problem statement.
theorem pairs_equality :
  pairA_1 = pairA_2 ∧ ¬ (pairB_1 = pairB_2) ∧ ¬ (pairC_1 = pairC_2) ∧ ¬ (pairD_1 = pairD_2) := by
  sorry

end pairs_equality_l73_73817


namespace set_A_correct_l73_73285

-- Definition of the sets and conditions
def A : Set ℤ := {-3, 0, 2, 6}
def B : Set ℤ := {-1, 3, 5, 8}

theorem set_A_correct : 
  (∃ a1 a2 a3 a4 : ℤ, A = {a1, a2, a3, a4} ∧ 
  {a1 + a2 + a3, a1 + a2 + a4, a1 + a3 + a4, a2 + a3 + a4} = B) → 
  A = {-3, 0, 2, 6} :=
by 
  sorry

end set_A_correct_l73_73285


namespace Jamie_earns_10_per_hour_l73_73572

noncomputable def JamieHourlyRate (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_hours := days_per_week * hours_per_day * weeks
  total_earnings / total_hours

theorem Jamie_earns_10_per_hour :
  JamieHourlyRate 2 3 6 360 = 10 := by
  sorry

end Jamie_earns_10_per_hour_l73_73572


namespace line_equation_parametric_to_implicit_l73_73765

theorem line_equation_parametric_to_implicit (t : ℝ) :
  ∀ x y : ℝ, (x = 3 * t + 6 ∧ y = 5 * t - 7) → y = (5 / 3) * x - 17 :=
by
  intros x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end line_equation_parametric_to_implicit_l73_73765


namespace two_students_choose_materials_l73_73467

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l73_73467


namespace scalar_mult_l73_73365

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem scalar_mult (a : α) (h : a ≠ 0) : (-4) • (3 • a) = -12 • a :=
  sorry

end scalar_mult_l73_73365


namespace total_detergent_used_l73_73754

-- Define the parameters of the problem
def total_pounds_of_clothes : ℝ := 9
def pounds_of_cotton : ℝ := 4
def pounds_of_woolen : ℝ := 5
def detergent_per_pound_cotton : ℝ := 2
def detergent_per_pound_woolen : ℝ := 1.5

-- Main theorem statement
theorem total_detergent_used : 
  (pounds_of_cotton * detergent_per_pound_cotton) + (pounds_of_woolen * detergent_per_pound_woolen) = 15.5 :=
by
  sorry

end total_detergent_used_l73_73754


namespace mandy_bike_time_l73_73583

-- Definitions of the ratios and time spent on yoga
def ratio_gym_bike : ℕ × ℕ := (2, 3)
def ratio_yoga_exercise : ℕ × ℕ := (2, 3)
def time_yoga : ℕ := 20

-- Theorem stating that Mandy will spend 18 minutes riding her bike
theorem mandy_bike_time (r_gb : ℕ × ℕ) (r_ye : ℕ × ℕ) (t_y : ℕ) 
  (h_rgb : r_gb = (2, 3)) (h_rye : r_ye = (2, 3)) (h_ty : t_y = 20) : 
  let t_e := (r_ye.snd * t_y) / r_ye.fst
  let t_part := t_e / (r_gb.fst + r_gb.snd)
  t_part * r_gb.snd = 18 := sorry

end mandy_bike_time_l73_73583


namespace sin_cos_product_l73_73868

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l73_73868


namespace negate_exists_real_l73_73594

theorem negate_exists_real (h : ¬ ∃ x : ℝ, x^2 - 2 ≤ 0) : ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negate_exists_real_l73_73594


namespace find_largest_beta_l73_73893

theorem find_largest_beta (α : ℝ) (r : ℕ → ℝ) (C : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < 1)
  (h3 : ∀ n, ∀ m ≠ n, dist (r n) (r m) ≥ (r n) ^ α)
  (h4 : ∀ n, r n ≤ r (n + 1)) 
  (h5 : ∀ n, r n ≥ C * n ^ (1 / (2 * (1 - α)))) :
  ∀ β, (∃ C > 0, ∀ n, r n ≥ C * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end find_largest_beta_l73_73893


namespace penalty_kicks_calculation_l73_73295

def totalPlayers := 24
def goalkeepers := 4
def nonGoalkeeperShootsAgainstOneGoalkeeper := totalPlayers - 1
def totalPenaltyKicks := goalkeepers * nonGoalkeeperShootsAgainstOneGoalkeeper

theorem penalty_kicks_calculation : totalPenaltyKicks = 92 := by
  sorry

end penalty_kicks_calculation_l73_73295


namespace solution_set_to_coeff_properties_l73_73855

theorem solution_set_to_coeff_properties 
  (a b c : ℝ) 
  (h : ∀ x, (2 < x ∧ x < 3) → ax^2 + bx + c > 0) 
  : 
  (a < 0) 
  ∧ (b * c < 0) 
  ∧ (b + c = a) :=
sorry

end solution_set_to_coeff_properties_l73_73855


namespace problem_statement_l73_73898

-- Defining the real numbers and the hypothesis
variables {a b c x y z : ℝ}
variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 31 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

-- State the theorem
theorem problem_statement : 
  (a / (a - 17) + b / (b - 31) + c / (c - 53) = 1) :=
by
  sorry

end problem_statement_l73_73898


namespace smallest_number_of_students_l73_73644

theorem smallest_number_of_students 
    (ratio_9th_10th : Nat := 3 / 2)
    (ratio_9th_11th : Nat := 5 / 4)
    (ratio_9th_12th : Nat := 7 / 6) :
  ∃ N9 N10 N11 N12 : Nat, 
  N9 / N10 = 3 / 2 ∧ N9 / N11 = 5 / 4 ∧ N9 / N12 = 7 / 6 ∧ N9 + N10 + N11 + N12 = 349 :=
by {
  sorry
}

#print axioms smallest_number_of_students

end smallest_number_of_students_l73_73644


namespace Cinderella_solves_l73_73074

/--
There are three bags labeled as "Poppy", "Millet", and "Mixture". Each label is incorrect.
By inspecting one grain from the bag labeled as "Mixture", Cinderella can determine the exact contents of all three bags.
-/
theorem Cinderella_solves (bag_contents : String → String) (examined_grain : String) :
  (bag_contents "Mixture" = "Poppy" ∨ bag_contents "Mixture" = "Millet") →
  (∀ l, bag_contents l ≠ l) →
  (examined_grain = "Poppy" ∨ examined_grain = "Millet") →
  examined_grain = bag_contents "Mixture" →
  ∃ poppy_bag millet_bag mixture_bag : String,
    poppy_bag ≠ "Poppy" ∧ millet_bag ≠ "Millet" ∧ mixture_bag ≠ "Mixture" ∧
    bag_contents poppy_bag = "Poppy" ∧
    bag_contents millet_bag = "Millet" ∧
    bag_contents mixture_bag = "Mixture" :=
sorry

end Cinderella_solves_l73_73074


namespace math_quiz_l73_73145

theorem math_quiz (x : ℕ) : 
  (∃ x ≥ 14, (∃ y : ℕ, 16 = x + y + 1) → (6 * x - 2 * y ≥ 75)) → 
  x ≥ 14 :=
by
  sorry

end math_quiz_l73_73145


namespace ellipse_equation_and_min_area_of_triangle_l73_73544

theorem ellipse_equation_and_min_area_of_triangle
  (a b : ℝ)
  (h1 : a = sqrt 3 * b)
  (h2 : (x : ℝ) → Set (x : ℝ, y : ℝ) := { p | x^2 / a^2 + p.snd^2 / b^2 = 1 })
  (hdirectrix : x = -3 * sqrt 2 / 2 = -a^2 / c)
  (l1 l2 : ℝ → ℝ)
  (h3 : l1 = (λ x, sqrt 3 / 3 * x))
  (h4 : l2 = (λ x, -sqrt 3 / 3 * x))
  (A B : ℝ × ℝ)
  (h5 : ∃ x1, A = (x1, sqrt 3 / 3 * x1))
  (h6 : ∃ x2, B = (x2, -sqrt 3 / 3 * x2))
  (P : ℝ × ℝ)
  (h7 : ∃ λ : ℝ, λ > 0 ∧ (P = (1 / (1 + λ) • A + λ / (1 + λ) • B))
  (h8 : P ∈ { p | p.fst^2 / a^2 - p.snd^2 / b^2 = 1 }) :

  -- Part 1: The equation of the ellipse
  (a^2 = 3 ∧ b^2 = 1) ∧

  -- Part 2: The minimum value of the area of ΔOAB
  (∀ (λ : ℝ), λ > 0 → (1 / 2 * (abs (λ) + 1 / abs (λ) + 2) * sqrt 3 ≥ 2 * sqrt 3) :=
sorry

end ellipse_equation_and_min_area_of_triangle_l73_73544


namespace complex_solution_l73_73268

theorem complex_solution (a : ℝ) (h : (a + complex.I) * (1 - a * complex.I) = 2) : a = 1 :=
by 
  sorry

end complex_solution_l73_73268


namespace tom_saves_money_l73_73187

-- Defining the cost of a normal doctor's visit
def normal_doctor_cost : ℕ := 200

-- Defining the discount percentage for the discount clinic
def discount_percentage : ℕ := 70

-- Defining the cost reduction based on the discount percentage
def discount_amount (cost percentage : ℕ) : ℕ := (percentage * cost) / 100

-- Defining the cost of a visit to the discount clinic
def discount_clinic_cost (normal_cost discount_amount : ℕ ) : ℕ := normal_cost - discount_amount

-- Defining the number of visits to the discount clinic
def discount_clinic_visits : ℕ := 2

-- Defining the total cost for the discount clinic visits
def total_discount_clinic_cost (visit_cost visits : ℕ) : ℕ := visits * visit_cost

-- The final cost savings calculation
def cost_savings (normal_cost total_discount_cost : ℕ) : ℕ := normal_cost - total_discount_cost

-- Proving the amount Tom saves by going to the discount clinic
theorem tom_saves_money : cost_savings normal_doctor_cost (total_discount_clinic_cost (discount_clinic_cost normal_doctor_cost (discount_amount normal_doctor_cost discount_percentage)) discount_clinic_visits) = 80 :=
by
  sorry

end tom_saves_money_l73_73187


namespace evaluate_M_l73_73834

noncomputable def M : ℝ := 
  (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

theorem evaluate_M : M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 :=
by
  sorry

end evaluate_M_l73_73834


namespace binomial_probability_l73_73647

theorem binomial_probability (n : ℕ) (p : ℝ) (h1 : (n * p = 300)) (h2 : (n * p * (1 - p) = 200)) :
    p = 1 / 3 :=
by
  sorry

end binomial_probability_l73_73647


namespace solve_abc_values_l73_73001

theorem solve_abc_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 8/3) :
  abc = 1 ∨ abc = 37/3 :=
sorry

end solve_abc_values_l73_73001


namespace no_nondegenerate_triangle_l73_73498

def distinct_positive_integers (a b c : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem no_nondegenerate_triangle (a b c : ℕ)
  (h_distinct : distinct_positive_integers a b c)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1)
  (h1 : a ∣ (b - c) ^ 2)
  (h2 : b ∣ (c - a) ^ 2)
  (h3 : c ∣ (a - b) ^ 2) :
  ¬nondegenerate_triangle a b c :=
sorry

end no_nondegenerate_triangle_l73_73498


namespace amount_subtracted_correct_l73_73503

noncomputable def find_subtracted_amount (N : ℝ) (A : ℝ) : Prop :=
  0.40 * N - A = 23

theorem amount_subtracted_correct :
  find_subtracted_amount 85 11 :=
by
  sorry

end amount_subtracted_correct_l73_73503


namespace shuai_shuai_total_words_l73_73040

-- Conditions
def words (a : ℕ) (n : ℕ) : ℕ := a + n

-- Total words memorized in 7 days
def total_memorized (a : ℕ) : ℕ := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) + (words a 4) + (words a 5) + (words a 6)

-- Condition: Sum of words memorized in the first 4 days equals sum of words in the last 3 days
def condition (a : ℕ) : Prop := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) = (words a 4) + (words a 5) + (words a 6)

-- Theorem: If condition is satisfied, then the total number of words memorized is 84.
theorem shuai_shuai_total_words : 
  ∀ a : ℕ, condition a → total_memorized a = 84 :=
by
  intro a h
  sorry

end shuai_shuai_total_words_l73_73040


namespace sequence_b_10_eq_110_l73_73341

theorem sequence_b_10_eq_110 :
  (∃ (b : ℕ → ℕ), b 1 = 2 ∧ (∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) ∧ b 10 = 110) :=
sorry

end sequence_b_10_eq_110_l73_73341


namespace area_of_tangency_triangle_l73_73601

theorem area_of_tangency_triangle 
  (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : r3 = 4) 
  (mutually_tangent : ∀ {c1 c2 c3 : ℝ}, c1 + c2 = r1 + r2 ∧ c2 + c3 = r2 + r3 ∧ c1 + c3 = r1 + r3 ) :
  ∃ area : ℝ, area = 3 * (Real.sqrt 6) / 2 :=
by
  sorry

end area_of_tangency_triangle_l73_73601


namespace find_x_l73_73847

theorem find_x (x : ℕ) : 3 * 2^x + 5 * 2^x = 2048 → x = 8 := by
  sorry

end find_x_l73_73847


namespace bake_sale_cookies_l73_73220

theorem bake_sale_cookies (raisin_cookies : ℕ) (oatmeal_cookies : ℕ) 
  (h1 : raisin_cookies = 42) 
  (h2 : raisin_cookies / oatmeal_cookies = 6) :
  raisin_cookies + oatmeal_cookies = 49 :=
sorry

end bake_sale_cookies_l73_73220


namespace min_value_fraction_l73_73133

theorem min_value_fraction (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) : 
  ∃ a, (∀ x y, (x - 1 ≥ 0) ∧ (x - y + 1 ≤ 0) ∧ (x + y - 4 ≤ 0) → (x / (y + 1)) ≥ a) ∧ 
      (a = 1 / 4) :=
sorry

end min_value_fraction_l73_73133


namespace number_is_16_l73_73316

theorem number_is_16 (n : ℝ) (h : (1/2) * n + 5 = 13) : n = 16 :=
sorry

end number_is_16_l73_73316


namespace eval_ceil_sqrt_sum_l73_73348

theorem eval_ceil_sqrt_sum :
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by sorry
  have h3 : 18 < Real.sqrt 333 ∧ Real.sqrt 333 < 19 := by sorry
  sorry

end eval_ceil_sqrt_sum_l73_73348


namespace find_x_l73_73284

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - q.1, p.2 + q.2)

theorem find_x (x y : ℤ) :
  star (3, 3) (0, 0) = star (x, y) (3, 2) → x = 6 :=
by
  intro h
  sorry

end find_x_l73_73284


namespace min_value_frac_sum_l73_73360

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_frac_sum_l73_73360


namespace quadratic_no_real_roots_min_k_l73_73103

theorem quadratic_no_real_roots_min_k :
  ∀ (k : ℤ), 
    (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) ↔ 
    (k ≥ 3) := 
by 
  sorry

end quadratic_no_real_roots_min_k_l73_73103


namespace polar_to_rectangular_l73_73826

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 6) (h₂ : θ = Real.pi / 2) :
  (r * Real.cos θ, r * Real.sin θ) = (0, 6) :=
by
  sorry

end polar_to_rectangular_l73_73826


namespace total_visitors_three_days_l73_73815

def V_Rachel := 92
def V_prev_day := 419
def V_day_before_prev := 103

theorem total_visitors_three_days : V_Rachel + V_prev_day + V_day_before_prev = 614 := 
by sorry

end total_visitors_three_days_l73_73815


namespace number_of_ways_to_choose_materials_l73_73471

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l73_73471


namespace problem_part1_problem_part2_l73_73389

theorem problem_part1 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  2 * ((1 / x) + (1 / y) + (1 / z)) ≤ (1 / p) + (1 / q) + (1 / r) :=
sorry

theorem problem_part2 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  x * y + y * z + z * x ≥ 2 * (p * x + q * y + r * z) :=
sorry

end problem_part1_problem_part2_l73_73389


namespace right_triangle_area_l73_73813

theorem right_triangle_area
    (h : ∀ {a b c : ℕ}, a^2 + b^2 = c^2 → c = 13 → a = 5 ∨ b = 5)
    (hypotenuse : ℕ)
    (leg : ℕ)
    (hypotenuse_eq : hypotenuse = 13)
    (leg_eq : leg = 5) : ∃ (area: ℕ), area = 30 :=
by
  -- The proof will go here.
  sorry

end right_triangle_area_l73_73813


namespace remaining_days_temperature_l73_73914

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end remaining_days_temperature_l73_73914


namespace conditions_necessary_sufficient_l73_73542

variables (p q r s : Prop)

theorem conditions_necessary_sufficient :
  ((p → r) ∧ (¬ (r → p)) ∧ (q → r) ∧ (s → r) ∧ (q → s)) →
  ((s ↔ q) ∧ ((p → q) ∧ ¬ (q → p)) ∧ ((¬ p → ¬ s) ∧ ¬ (¬ s → ¬ p))) := by
  sorry

end conditions_necessary_sufficient_l73_73542


namespace negation_of_proposition_l73_73374

theorem negation_of_proposition (p : Real → Prop) : 
  (∀ x : Real, p x) → ¬(∀ x : Real, x ≥ 1) ↔ (∃ x : Real, x < 1) := 
by sorry

end negation_of_proposition_l73_73374


namespace minimum_additional_squares_to_symmetry_l73_73894

-- Define the type for coordinates in the grid
structure Coord where
  x : Nat
  y : Nat

-- Define the conditions
def initial_shaded_squares : List Coord := [
  ⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩, ⟨1, 4⟩
]

def grid_size : Coord := ⟨6, 5⟩

def vertical_line_of_symmetry : Nat := 3 -- between columns 3 and 4
def horizontal_line_of_symmetry : Nat := 2 -- between rows 2 and 3

-- Define reflection across lines of symmetry
def reflect_vertical (c : Coord) : Coord :=
  ⟨2 * vertical_line_of_symmetry - c.x, c.y⟩

def reflect_horizontal (c : Coord) : Coord :=
  ⟨c.x, 2 * horizontal_line_of_symmetry - c.y⟩

def reflect_both (c : Coord) : Coord :=
  reflect_vertical (reflect_horizontal c)

-- Define the theorem
theorem minimum_additional_squares_to_symmetry :
  ∃ (additional_squares : Nat), additional_squares = 5 := 
sorry

end minimum_additional_squares_to_symmetry_l73_73894


namespace find_smallest_n_l73_73234

-- Define costs and relationships
def cost_red (r : ℕ) : ℕ := 10 * r
def cost_green (g : ℕ) : ℕ := 18 * g
def cost_blue (b : ℕ) : ℕ := 20 * b
def cost_purple (n : ℕ) : ℕ := 24 * n

-- Define the mathematical problem
theorem find_smallest_n (r g b : ℕ) :
  ∃ n : ℕ, 24 * n = Nat.lcm (cost_red r) (Nat.lcm (cost_green g) (cost_blue b)) ∧ n = 15 :=
by
  sorry

end find_smallest_n_l73_73234


namespace difference_xy_l73_73720

theorem difference_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := sorry

end difference_xy_l73_73720


namespace arithmetic_sequence_m_value_l73_73361

theorem arithmetic_sequence_m_value 
  (a : ℕ → ℝ) (d : ℝ) (h₁ : d ≠ 0) 
  (h₂ : a 3 + a 6 + a 10 + a 13 = 32) 
  (m : ℕ) (h₃ : a m = 8) : 
  m = 8 :=
sorry

end arithmetic_sequence_m_value_l73_73361


namespace min_boat_trips_l73_73067
-- Import Mathlib to include necessary libraries

-- Define the problem using noncomputable theory if necessary
theorem min_boat_trips (students boat_capacity : ℕ) (h1 : students = 37) (h2 : boat_capacity = 5) : ∃ x : ℕ, x ≥ 9 :=
by
  -- Here we need to prove the assumption and goal, hence adding sorry
  sorry

end min_boat_trips_l73_73067


namespace two_students_choose_materials_l73_73468

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l73_73468


namespace percent_pension_participation_l73_73828

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

end percent_pension_participation_l73_73828


namespace find_a_l73_73550

theorem find_a (α β : ℝ) (h1 : α + β = 10) (h2 : α * β = 20) : (1 / α + 1 / β) = 1 / 2 :=
sorry

end find_a_l73_73550


namespace quadrilateral_equality_l73_73569

-- Variables definitions for points and necessary properties
variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assumptions based on given conditions
variables (AB : ℝ) (AD : ℝ) (BC : ℝ) (DC : ℝ) (beta : ℝ)
variables {angleB : ℝ} {angleD : ℝ}

-- Given conditions
axiom AB_eq_AD : AB = AD
axiom angleB_eq_angleD : angleB = angleD

-- The statement to be proven
theorem quadrilateral_equality (h1 : AB = AD) (h2 : angleB = angleD) : BC = DC :=
by
  sorry

end quadrilateral_equality_l73_73569


namespace ages_of_residents_l73_73797

theorem ages_of_residents (a b c : ℕ)
  (h1 : a * b * c = 1296)
  (h2 : a + b + c = 91)
  (h3 : ∀ x y z : ℕ, x * y * z = 1296 → x + y + z = 91 → (x < 80 ∧ y < 80 ∧ z < 80) → (x = 1 ∧ y = 18 ∧ z = 72)) :
  (a = 1 ∧ b = 18 ∧ c = 72 ∨ a = 1 ∧ b = 72 ∧ c = 18 ∨ a = 18 ∧ b = 1 ∧ c = 72 ∨ a = 18 ∧ b = 72 ∧ c = 1 ∨ a = 72 ∧ b = 1 ∧ c = 18 ∨ a = 72 ∧ b = 18 ∧ c = 1) :=
by
  sorry

end ages_of_residents_l73_73797


namespace polynomial_value_l73_73476

noncomputable def polynomial_spec (p : ℝ) : Prop :=
  p^3 - 5 * p + 1 = 0

theorem polynomial_value (p : ℝ) (h : polynomial_spec p) : 
  p^4 - 3 * p^3 - 5 * p^2 + 16 * p + 2015 = 2018 := 
by
  sorry

end polynomial_value_l73_73476


namespace solve_quadratic_inequality_l73_73166

theorem solve_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, (a * x ^ 2 - (2 * a + 1) * x + 2 > 0 ↔
    if a = 0 then
      x < 2
    else if a > 0 then
      if a >= 1 / 2 then
        x < 1 / a ∨ x > 2
      else
        x < 2 ∨ x > 1 / a
    else
      x > 1 / a ∧ x < 2)) :=
sorry

end solve_quadratic_inequality_l73_73166


namespace probability_none_A_B_C_l73_73330

-- Define the probabilities as given conditions
def P_A : ℝ := 0.25
def P_B : ℝ := 0.40
def P_C : ℝ := 0.35
def P_AB : ℝ := 0.20
def P_AC : ℝ := 0.15
def P_BC : ℝ := 0.25
def P_ABC : ℝ := 0.10

-- Prove that the probability that none of the events A, B, C occur simultaneously is 0.50
theorem probability_none_A_B_C : 1 - (P_A + P_B + P_C - P_AB - P_AC - P_BC + P_ABC) = 0.50 :=
by
  sorry

end probability_none_A_B_C_l73_73330


namespace probability_of_symmetric_line_l73_73884

noncomputable def probability_symmetric_line :=
  let total_points := 121
  let remaining_points := 119
  let symmetric_points := 40
  symmetric_points / remaining_points

theorem probability_of_symmetric_line :
  probability_symmetric_line = 40 / 119 := 
by sorry

end probability_of_symmetric_line_l73_73884


namespace calculation_proof_l73_73656

theorem calculation_proof : (96 / 6) * 3 / 2 = 24 := by
  sorry

end calculation_proof_l73_73656


namespace solve_system_of_equations_l73_73587

theorem solve_system_of_equations : 
  ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 5 * x + 2 * y = 8 ∧ x = 20 / 23 ∧ y = 42 / 23 :=
by
  sorry

end solve_system_of_equations_l73_73587


namespace total_pears_l73_73153

def jason_pears : Nat := 46
def keith_pears : Nat := 47
def mike_pears : Nat := 12

theorem total_pears : jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end total_pears_l73_73153


namespace number_of_B_students_l73_73146

/- Define the assumptions of the problem -/
variable (x : ℝ)  -- the number of students who earn a B

/- Express the number of students getting each grade in terms of x -/
def number_of_A (x : ℝ) := 0.6 * x
def number_of_C (x : ℝ) := 1.3 * x
def number_of_D (x : ℝ) := 0.8 * x
def total_students (x : ℝ) := number_of_A x + x + number_of_C x + number_of_D x

/- Prove that x = 14 for the total number of students being 50 -/
theorem number_of_B_students : total_students x = 50 → x = 14 :=
by 
  sorry

end number_of_B_students_l73_73146


namespace moles_of_C6H5CH3_formed_l73_73838

-- Stoichiometry of the reaction
def balanced_reaction (C6H6 CH4 C6H5CH3 H2 : ℝ) : Prop :=
  C6H6 + CH4 = C6H5CH3 + H2

-- Given conditions
def reaction_conditions (initial_CH4 : ℝ) (initial_C6H6 final_C6H5CH3 final_H2 : ℝ) : Prop :=
  balanced_reaction initial_C6H6 initial_CH4 final_C6H5CH3 final_H2 ∧ initial_CH4 = 3 ∧ final_H2 = 3

-- Theorem to prove
theorem moles_of_C6H5CH3_formed (initial_CH4 final_C6H5CH3 : ℝ) : reaction_conditions initial_CH4 3 final_C6H5CH3 3 → final_C6H5CH3 = 3 :=
by
  intros h
  sorry

end moles_of_C6H5CH3_formed_l73_73838


namespace find_train_speed_l73_73625

variable (L V : ℝ)

-- Conditions
def condition1 := V = L / 10
def condition2 := V = (L + 600) / 30

-- Theorem statement
theorem find_train_speed (h1 : condition1 L V) (h2 : condition2 L V) : V = 30 :=
by
  sorry

end find_train_speed_l73_73625


namespace total_tickets_sold_l73_73990

theorem total_tickets_sold (n : ℕ) 
  (h1 : n * n = 1681) : 
  2 * n = 82 :=
by
  sorry

end total_tickets_sold_l73_73990


namespace infinite_n_exist_l73_73676

def S (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem infinite_n_exist (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ᶠ n in at_top, S n ≡ n [MOD p] :=
sorry

end infinite_n_exist_l73_73676


namespace joe_fruit_probability_l73_73574

theorem joe_fruit_probability :
  let prob_same := (1 / 4) ^ 3
  let total_prob_same := 4 * prob_same
  let prob_diff := 1 - total_prob_same
  prob_diff = 15 / 16 :=
by
  sorry

end joe_fruit_probability_l73_73574


namespace at_least_one_greater_than_one_l73_73739

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

end at_least_one_greater_than_one_l73_73739


namespace shopkeeper_sold_articles_l73_73094

theorem shopkeeper_sold_articles (C : ℝ) (N : ℕ) 
  (h1 : (35 * C = N * C + (1/6) * (N * C))) : 
  N = 30 :=
by
  sorry

end shopkeeper_sold_articles_l73_73094


namespace voting_problem_l73_73385

theorem voting_problem (x y x' y' : ℕ) (m : ℕ) (h1 : x + y = 500) (h2 : y > x)
    (h3 : y - x = m) (h4 : x' = (10 * y) / 9) (h5 : x' + y' = 500)
    (h6 : x' - y' = 3 * m) :
    x' - x = 59 := 
sorry

end voting_problem_l73_73385


namespace bus_people_difference_l73_73941

theorem bus_people_difference 
  (initial : ℕ) (got_off : ℕ) (got_on : ℕ) (current : ℕ) 
  (h_initial : initial = 35)
  (h_got_off : got_off = 18)
  (h_got_on : got_on = 15)
  (h_current : current = initial - got_off + got_on) :
  initial - current = 3 := by
  sorry

end bus_people_difference_l73_73941


namespace count_multiples_of_7_not_14_300_l73_73549

open Finset

def count_multiples_of_7_not_14 (n : ℕ) : ℕ :=
  (Icc 1 n).filter (λ k, k % 7 = 0 ∧ k % 14 ≠ 0).card

theorem count_multiples_of_7_not_14_300 : count_multiples_of_7_not_14 300 = 21 :=
  by sorry

end count_multiples_of_7_not_14_300_l73_73549


namespace abscissa_of_A_is_5_l73_73729

theorem abscissa_of_A_is_5
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A.1 = A.2 ∧ A.1 > 0)
  (hB : B = (5, 0))
  (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hC : C = ((A.1 + 5) / 2, A.2 / 2))
  (hD : D = (5 / 2, 5 / 2))
  (dot_product_eq : (B.1 - A.1, B.2 - A.2) • (D.1 - C.1, D.2 - C.2) = 0) :
  A.1 = 5 :=
sorry

end abscissa_of_A_is_5_l73_73729


namespace cost_per_bag_l73_73943

theorem cost_per_bag
  (friends : ℕ)
  (payment_per_friend : ℕ)
  (total_bags : ℕ)
  (total_cost : ℕ)
  (h1 : friends = 3)
  (h2 : payment_per_friend = 5)
  (h3 : total_bags = 5)
  (h4 : total_cost = friends * payment_per_friend) :
  total_cost / total_bags = 3 :=
by {
  sorry
}

end cost_per_bag_l73_73943


namespace find_factor_l73_73641

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end find_factor_l73_73641


namespace solution_set_of_inequality_l73_73852

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_2 : f 2 = 1 / 2
axiom f_prime_lt_exp : ∀ x : ℝ, deriv f x < Real.exp x

theorem solution_set_of_inequality :
  {x : ℝ | f x < Real.exp x - 1 / 2} = {x : ℝ | 0 < x} :=
by
  sorry

end solution_set_of_inequality_l73_73852


namespace intersection_x_value_l73_73309

theorem intersection_x_value :
  (∃ x y : ℝ, y = 5 * x - 20 ∧ y = 110 - 3 * x ∧ x = 16.25) := sorry

end intersection_x_value_l73_73309


namespace find_a_b_f_inequality_l73_73254

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

-- a == 1 and b == 1 from the given conditions
theorem find_a_b (e : ℝ) (h_e : e = Real.exp 1) (b : ℝ) (a : ℝ) 
  (h_tangent : ∀ x, f x a = (e - 2) * x + b → a = 1 ∧ b = 1) : a = 1 ∧ b = 1 :=
sorry

-- prove f(x) > x^2 + 4x - 14 for x >= 0
theorem f_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ x : ℝ, 0 ≤ x → f x 1 > x^2 + 4 * x - 14 :=
sorry

end find_a_b_f_inequality_l73_73254


namespace find_b_l73_73178

noncomputable def func (x a b : ℝ) := (1 / 12) * x^2 + a * x + b

theorem  find_b (a b : ℝ) (x1 x2 : ℝ):
    (func x1 a b = 0) →
    (func x2 a b = 0) →
    (b = (x1 * x2) / 12) →
    ((3 - x1) = (x2 - 3)) →
    (b = -6) :=
by
    sorry

end find_b_l73_73178


namespace equilateral_triangle_l73_73905

theorem equilateral_triangle
  (A B C : Type)
  (angle_A : ℝ)
  (side_BC : ℝ)
  (perimeter : ℝ)
  (h1 : angle_A = 60)
  (h2 : side_BC = 1/3 * perimeter)
  (side_AB : ℝ)
  (side_AC : ℝ)
  (h3 : perimeter = side_BC + side_AB + side_AC) :
  (side_AB = side_BC) ∧ (side_AC = side_BC) :=
by
  sorry

end equilateral_triangle_l73_73905


namespace find_A_when_B_is_largest_l73_73940

theorem find_A_when_B_is_largest :
  ∃ A : ℕ, ∃ B : ℕ, A = 17 * 25 + B ∧ B < 17 ∧ B = 16 ∧ A = 441 :=
by
  sorry

end find_A_when_B_is_largest_l73_73940


namespace inequality_always_true_l73_73179

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end inequality_always_true_l73_73179


namespace intersection_of_M_and_N_l73_73411

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | x < 1}
def expected_intersection : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = expected_intersection :=
sorry

end intersection_of_M_and_N_l73_73411


namespace Bobby_candy_l73_73100

theorem Bobby_candy (initial_candy remaining_candy1 remaining_candy2 : ℕ)
  (H1 : initial_candy = 21)
  (H2 : remaining_candy1 = initial_candy - 5)
  (H3 : remaining_candy2 = remaining_candy1 - 9):
  remaining_candy2 = 7 :=
by
  sorry

end Bobby_candy_l73_73100


namespace tg_plus_ctg_l73_73835

theorem tg_plus_ctg (x : ℝ) (h : 1 / Real.cos x - 1 / Real.sin x = Real.sqrt 15) :
  Real.tan x + (1 / Real.tan x) = -3 ∨ Real.tan x + (1 / Real.tan x) = 5 :=
sorry

end tg_plus_ctg_l73_73835


namespace number_of_zeros_l73_73924

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 3

theorem number_of_zeros (b : ℝ) : 
  ∃ x₁ x₂ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ x₁ ≠ x₂ := by
  sorry

end number_of_zeros_l73_73924


namespace Q_subset_P_l73_73738

def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | x^2 < 4 }

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l73_73738


namespace pricePerRedStamp_l73_73902

namespace StampCollection

-- Definitions for the conditions
def totalRedStamps : ℕ := 20
def soldRedStamps : ℕ := 20
def totalBlueStamps : ℕ := 80
def soldBlueStamps : ℕ := 80
def pricePerBlueStamp : ℝ := 0.8
def totalYellowStamps : ℕ := 7
def pricePerYellowStamp : ℝ := 2
def totalTargetEarnings : ℝ := 100

-- Derived definitions from conditions
def earningsFromBlueStamps : ℝ := soldBlueStamps * pricePerBlueStamp
def earningsFromYellowStamps : ℝ := totalYellowStamps * pricePerYellowStamp
def earningsRequiredFromRedStamps : ℝ := totalTargetEarnings - (earningsFromBlueStamps + earningsFromYellowStamps)

-- The statement asserting the main proof obligation
theorem pricePerRedStamp :
  (earningsRequiredFromRedStamps / soldRedStamps) = 1.10 :=
sorry

end StampCollection

end pricePerRedStamp_l73_73902


namespace adam_age_l73_73070

theorem adam_age (x : ℤ) :
  (∃ m : ℤ, x - 2 = m^2) ∧ (∃ n : ℤ, x + 2 = n^3) → x = 6 :=
by
  sorry

end adam_age_l73_73070


namespace square_perimeter_l73_73808

theorem square_perimeter (area : ℝ) (h : area = 625) :
  ∃ p : ℝ, p = 4 * real.sqrt area ∧ p = 100 :=
by
  sorry

end square_perimeter_l73_73808


namespace range_of_a_l73_73257

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Lean statement for the problem
theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : -1 < a ∧ a ≤ 1 := 
by
  -- Proof is skipped
  sorry

end range_of_a_l73_73257


namespace selected_40th_is_795_l73_73291

-- Definitions of constants based on the problem conditions
def total_participants : ℕ := 1000
def selections : ℕ := 50
def equal_spacing : ℕ := total_participants / selections
def first_selected_number : ℕ := 15
def nth_selected_number (n : ℕ) : ℕ := (n - 1) * equal_spacing + first_selected_number

-- The theorem to prove the 40th selected number is 795
theorem selected_40th_is_795 : nth_selected_number 40 = 795 := 
by 
  -- Skipping the detailed proof
  sorry

end selected_40th_is_795_l73_73291


namespace number_of_circumcenter_quadrilaterals_l73_73786

-- Definitions for each type of quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

def is_square (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_kite (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def has_circumcenter (q : Quadrilateral) : Prop := sorry

-- List of quadrilaterals
def square : Quadrilateral := sorry
def rectangle : Quadrilateral := sorry
def rhombus : Quadrilateral := sorry
def kite : Quadrilateral := sorry
def trapezoid : Quadrilateral := sorry

-- Proof that the number of quadrilaterals with a point equidistant from all vertices is 2
theorem number_of_circumcenter_quadrilaterals :
  (has_circumcenter square) ∧
  (has_circumcenter rectangle) ∧
  ¬ (has_circumcenter rhombus) ∧
  ¬ (has_circumcenter kite) ∧
  ¬ (has_circumcenter trapezoid) →
  2 = 2 :=
by
  sorry

end number_of_circumcenter_quadrilaterals_l73_73786


namespace methane_production_proof_l73_73353

noncomputable def methane_production
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : Prop :=
  methane_formed = 3

theorem methane_production_proof 
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : methane_production C H methane_formed h_formula h_initial_conditions h_reaction :=
by {
  sorry
}

end methane_production_proof_l73_73353


namespace students_material_selection_l73_73466

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l73_73466


namespace sin_cos_value_l73_73872

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l73_73872


namespace smallest_value_expression_l73_73782

theorem smallest_value_expression (n : ℕ) (hn : n > 0) : (n = 8) ↔ ((n / 2) + (32 / n) = 8) := by
  sorry

end smallest_value_expression_l73_73782


namespace div_mul_fraction_eq_neg_81_over_4_l73_73658

theorem div_mul_fraction_eq_neg_81_over_4 : 
  -4 / (4 / 9) * (9 / 4) = - (81 / 4) := 
by
  sorry

end div_mul_fraction_eq_neg_81_over_4_l73_73658


namespace transformed_sum_l73_73649

open BigOperators -- Open namespace to use big operators like summation

theorem transformed_sum (n : ℕ) (x : Fin n → ℝ) (s : ℝ) 
  (h_sum : ∑ i, x i = s) : 
  ∑ i, ((3 * (x i + 10)) - 10) = 3 * s + 20 * n :=
by
  sorry

end transformed_sum_l73_73649


namespace two_students_choose_materials_l73_73469

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l73_73469


namespace distinct_terms_in_expansion_l73_73712

theorem distinct_terms_in_expansion:
  (∀ (x y z u v w: ℝ), (x + y + z) * (u + v + w + x + y) = 0 → false) →
  3 * 5 = 15 := by sorry

end distinct_terms_in_expansion_l73_73712


namespace find_abc_digits_l73_73771

theorem find_abc_digits (N : ℕ) (abcd : ℕ) (a b c d : ℕ) (hN : N % 10000 = abcd) (hNsq : N^2 % 10000 = abcd)
  (ha_ne_zero : a ≠ 0) (hb_ne_six : b ≠ 6) (hc_ne_six : c ≠ 6) : (a * 100 + b * 10 + c) = 106 :=
by
  -- The proof is omitted.
  sorry

end find_abc_digits_l73_73771


namespace total_surface_area_correct_l73_73315

-- Defining the dimensions of the rectangular solid
def length := 10
def width := 9
def depth := 6

-- Definition of the total surface area of a rectangular solid
def surface_area (l w d : ℕ) := 2 * (l * w + w * d + l * d)

-- Proposition that the total surface area for the given dimensions is 408 square meters
theorem total_surface_area_correct : surface_area length width depth = 408 := 
by
  sorry

end total_surface_area_correct_l73_73315


namespace find_m_n_l73_73617

theorem find_m_n : ∃ (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 :=
by {
  sorry
}

end find_m_n_l73_73617


namespace probability_A_not_lose_l73_73795

theorem probability_A_not_lose (p_win p_draw : ℝ) (h_win : p_win = 0.3) (h_draw : p_draw = 0.5) :
  (p_win + p_draw = 0.8) :=
by
  rw [h_win, h_draw]
  norm_num

end probability_A_not_lose_l73_73795


namespace inequality_holds_for_all_x_l73_73842

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m * x - 1) / (2 * x^2 - 2 * x + 3) < 1) ↔ -6 < m ∧ m < 2 := 
sorry -- Proof to be provided

end inequality_holds_for_all_x_l73_73842


namespace shift_right_graph_l73_73186

theorem shift_right_graph (x : ℝ) :
  (3 : ℝ)^(x+1) = (3 : ℝ)^((x+1) - 1) :=
by 
  -- Here we prove that shifting the graph of y = 3^(x+1) to right by 1 unit 
  -- gives the graph of y = 3^x
  sorry

end shift_right_graph_l73_73186


namespace shared_bill_approx_16_99_l73_73307

noncomputable def calculate_shared_bill (total_bill : ℝ) (num_people : ℕ) (tip_rate : ℝ) : ℝ :=
  let tip := total_bill * tip_rate
  let total_with_tip := total_bill + tip
  total_with_tip / num_people

theorem shared_bill_approx_16_99 :
  calculate_shared_bill 139 9 0.10 ≈ 16.99 :=
sorry

end shared_bill_approx_16_99_l73_73307


namespace width_of_jesses_room_l73_73024

theorem width_of_jesses_room (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) (total_area : ℝ) (width : ℝ) :
  length = 2 → tile_area = 4 → num_tiles = 6 → total_area = (num_tiles * tile_area : ℝ) → (length * width) = total_area → width = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end width_of_jesses_room_l73_73024


namespace additional_people_needed_l73_73841

theorem additional_people_needed (h₁ : ∀ p h : ℕ, (p * h = 40)) (h₂ : 5 * 8 = 40) : 7 - 5 = 2 :=
by
  sorry

end additional_people_needed_l73_73841


namespace pencil_weight_l73_73698

theorem pencil_weight (total_weight : ℝ) (empty_case_weight : ℝ) (num_pencils : ℕ)
  (h1 : total_weight = 11.14) 
  (h2 : empty_case_weight = 0.5) 
  (h3 : num_pencils = 14) :
  (total_weight - empty_case_weight) / num_pencils = 0.76 := by
  sorry

end pencil_weight_l73_73698


namespace remaining_days_temperature_l73_73917

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end remaining_days_temperature_l73_73917


namespace intersection_points_form_line_slope_l73_73356

theorem intersection_points_form_line_slope (s : ℝ) :
  ∃ (m : ℝ), m = 1/18 ∧ ∀ (x y : ℝ),
    (3 * x + y = 5 * s + 6) ∧ (2 * x - 3 * y = 3 * s - 5) →
    ∃ k : ℝ, (y = m * x + k) :=
by
  sorry

end intersection_points_form_line_slope_l73_73356


namespace students_material_selection_l73_73463

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l73_73463


namespace lcm_9_12_15_l73_73973

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l73_73973


namespace expected_adjacent_black_pairs_proof_l73_73994

-- Define the modified deck conditions.
def modified_deck (n : ℕ) := n = 60
def black_cards (b : ℕ) := b = 30
def red_cards (r : ℕ) := r = 30

-- Define the expected value of pairs of adjacent black cards.
def expected_adjacent_black_pairs (n b : ℕ) : ℚ :=
  b * (b - 1) / (n - 1)

theorem expected_adjacent_black_pairs_proof :
  modified_deck 60 →
  black_cards 30 →
  red_cards 30 →
  expected_adjacent_black_pairs 60 30 = 870 / 59 :=
by intros; sorry

end expected_adjacent_black_pairs_proof_l73_73994


namespace ratio_of_ages_l73_73044

theorem ratio_of_ages (D R : ℕ) (h1 : D = 3) (h2 : R + 22 = 26) : R / D = 4 / 3 := by
  sorry

end ratio_of_ages_l73_73044


namespace aerith_is_correct_l73_73332

theorem aerith_is_correct :
  ∀ x : ℝ, x = 1.4 → (x ^ (x ^ x)) < 2 → ∃ y : ℝ, y = x ^ (x ^ x) :=
by
  sorry

end aerith_is_correct_l73_73332


namespace willie_exchange_rate_l73_73620

theorem willie_exchange_rate :
  let euros := 70
  let normal_exchange_rate := 1 / 5 -- euros per dollar
  let airport_exchange_rate := 5 / 7
  let dollars := euros * normal_exchange_rate * airport_exchange_rate
  dollars = 10 := by
  sorry

end willie_exchange_rate_l73_73620


namespace x_squared_minus_y_squared_l73_73007

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end x_squared_minus_y_squared_l73_73007


namespace topsoil_cost_l73_73945

theorem topsoil_cost (cost_per_cubic_foot : ℕ) (cubic_yard_to_cubic_foot : ℕ) (volume_in_cubic_yards : ℕ) :
  cost_per_cubic_foot = 8 →
  cubic_yard_to_cubic_foot = 27 →
  volume_in_cubic_yards = 3 →
  volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 648 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l73_73945


namespace case_a_case_b_case_c_l73_73539

-- Definitions of game manageable
inductive Player
| First
| Second

def sum_of_dimensions (m n : Nat) : Nat := m + n

def is_winning_position (m n : Nat) : Player :=
  if sum_of_dimensions m n % 2 = 1 then Player.First else Player.Second

-- Theorem statements for the given grid sizes
theorem case_a : is_winning_position 9 10 = Player.First := 
  sorry

theorem case_b : is_winning_position 10 12 = Player.Second := 
  sorry

theorem case_c : is_winning_position 9 11 = Player.Second := 
  sorry

end case_a_case_b_case_c_l73_73539


namespace correct_statement_l73_73554

-- We assume the existence of lines and planes with certain properties.
variables {Line : Type} {Plane : Type}
variables {m n : Line} {alpha beta gamma : Plane}

-- Definitions for perpendicular and parallel relations
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- The theorem we aim to prove given the conditions
theorem correct_statement :
  line_perpendicular_to_plane m beta ∧ line_parallel_to_plane m alpha → perpendicular alpha beta :=
by sorry

end correct_statement_l73_73554


namespace dogs_legs_l73_73387

theorem dogs_legs (num_dogs : ℕ) (legs_per_dog : ℕ) (h1 : num_dogs = 109) (h2 : legs_per_dog = 4) : num_dogs * legs_per_dog = 436 :=
by {
  -- The proof is omitted as it's indicated that it should contain "sorry"
  sorry
}

end dogs_legs_l73_73387


namespace roots_of_equation_l73_73929

theorem roots_of_equation (x : ℝ) : ((x - 5) ^ 2 = 2 * (x - 5)) ↔ (x = 5 ∨ x = 7) := by
sorry

end roots_of_equation_l73_73929


namespace average_weight_l73_73428

theorem average_weight :
  ∀ (A B C : ℝ),
    (A + B = 84) → 
    (B + C = 86) → 
    (B = 35) → 
    (A + B + C) / 3 = 45 :=
by
  intros A B C hab hbc hb
  -- proof omitted
  sorry

end average_weight_l73_73428


namespace second_year_selection_l73_73384

noncomputable def students_from_first_year : ℕ := 30
noncomputable def students_from_second_year : ℕ := 40
noncomputable def selected_from_first_year : ℕ := 6
noncomputable def selected_from_second_year : ℕ := (selected_from_first_year * students_from_second_year) / students_from_first_year

theorem second_year_selection :
  students_from_second_year = 40 ∧ students_from_first_year = 30 ∧ selected_from_first_year = 6 →
  selected_from_second_year = 8 :=
by
  intros h
  sorry

end second_year_selection_l73_73384


namespace evaluate_expression_at_minus_half_l73_73759

noncomputable def complex_expression (x : ℚ) : ℚ :=
  (x - 3)^2 + (x + 3) * (x - 3) - 2 * x * (x - 2) + 1

theorem evaluate_expression_at_minus_half :
  complex_expression (-1 / 2) = 2 :=
by
  sorry

end evaluate_expression_at_minus_half_l73_73759


namespace fraction_meaningful_iff_l73_73068

theorem fraction_meaningful_iff (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 :=
by 
  sorry

end fraction_meaningful_iff_l73_73068


namespace second_box_capacity_l73_73631

-- Given conditions
def height1 := 4 -- height of the first box in cm
def width1 := 2 -- width of the first box in cm
def length1 := 6 -- length of the first box in cm
def clay_capacity1 := 48 -- weight capacity of the first box in grams

def height2 := 3 * height1 -- height of the second box in cm
def width2 := 2 * width1 -- width of the second box in cm
def length2 := length1 -- length of the second box in cm

-- Hypothesis: weight capacity increases quadratically with height
def quadratic_relationship (h1 h2 : ℕ) (capacity1 : ℕ) : ℕ :=
  (h2 / h1) * (h2 / h1) * capacity1

-- The proof problem
theorem second_box_capacity :
  quadratic_relationship height1 height2 clay_capacity1 = 432 :=
by
  -- proof omitted
  sorry

end second_box_capacity_l73_73631


namespace proof_problem_l73_73888

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 1

-- Define the circle C with center (h, k) and radius r
def circle_eq (h k r : ℝ) (x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Define condition of line that intersects the circle C at points A and B
def line_eq (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Condition: OA ⊥ OB
def perpendicular_cond (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem stating the proof problem
theorem proof_problem :
  (∃ (h k r : ℝ),
    circle_eq h k r 3 1 ∧
    circle_eq h k r 5 0 ∧
    circle_eq h k r 1 0 ∧
    h = 3 ∧ k = 1 ∧ r = 3) ∧
    (∃ (a : ℝ),
      (∀ (x1 y1 x2 y2 : ℝ),
        line_eq a x1 y1 ∧
        circle_eq 3 1 3 x1 y1 ∧
        line_eq a x2 y2 ∧
        circle_eq 3 1 3 x2 y2 → 
        perpendicular_cond x1 y1 x2 y2) →
      a = -1) :=
by
  sorry

end proof_problem_l73_73888


namespace square_perimeter_l73_73807

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l73_73807


namespace find_x_l73_73741

theorem find_x (x : ℚ) (h1 : 8 * x^2 + 9 * x - 2 = 0) (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1 / 8 :=
by sorry

end find_x_l73_73741


namespace sequence_geometric_and_general_term_sum_of_sequence_l73_73252

theorem sequence_geometric_and_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, S k = 2 * a k - k) : 
  (a 0 = 1) ∧ 
  (∀ k : ℕ, a (k + 1) = 2 * a k + 1) ∧ 
  (∀ k : ℕ, a k = 2^k - 1) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, a k = 2^k - 1)
  (h2 : ∀ k : ℕ, b k = 1 / a (k+1) + 1 / (a k * a (k+1))) :
  T n = 1 - 1 / (2^(n+1) - 1) :=
sorry

end sequence_geometric_and_general_term_sum_of_sequence_l73_73252


namespace total_turtles_in_lake_l73_73445

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l73_73445


namespace angle_double_of_supplementary_l73_73073

theorem angle_double_of_supplementary (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 2 * (180 - x)) : x = 120 :=
sorry

end angle_double_of_supplementary_l73_73073


namespace incidence_bounds_l73_73422

noncomputable def incidence_upper_bound (n : ℕ) : Prop :=
  ∃ c : ℝ, ∀ (n_points n_lines : ℕ), n_points = n ∧ n_lines = n → I(n_points, n_lines) ≤ c * (n : ℝ) ^ (4/3)

noncomputable def incidence_lower_bound (n : ℕ) : Prop :=
  ∃ c' : ℝ, ∀ (n_points n_lines : ℕ), n_points = n ∧ n_lines = n → I(n_points, n_lines) ≥ c' * (n : ℝ) ^ (4/3)

theorem incidence_bounds (n : ℕ) :
  incidence_upper_bound n ∧ incidence_lower_bound n := by
  sorry

end incidence_bounds_l73_73422


namespace reeya_fifth_score_l73_73161

theorem reeya_fifth_score
  (s1 s2 s3 s4 avg: ℝ)
  (h1: s1 = 65)
  (h2: s2 = 67)
  (h3: s3 = 76)
  (h4: s4 = 82)
  (h_avg: avg = 75) :
  ∃ s5, s1 + s2 + s3 + s4 + s5 = 5 * avg ∧ s5 = 85 :=
by
  use 85
  sorry

end reeya_fifth_score_l73_73161


namespace least_common_multiple_9_12_15_l73_73966

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l73_73966


namespace erica_pie_percentage_l73_73830

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end erica_pie_percentage_l73_73830


namespace incorrect_statements_l73_73270

variable {Ω : Type*} [probability_space Ω]

def A1 : event Ω := sorry
def A2 : event Ω := sorry
def A3 : event Ω := sorry

axiom P_A1 : P A1 = 0.2
axiom P_A2 : P A2 = 0.3
axiom P_A3 : P A3 = 0.5
axiom P_non_neg {e : event Ω} : 0 ≤ P e 

theorem incorrect_statements :
  (¬(mutually_exclusive (A1 ∪ A2) A3 ∧ (A1 ∪ A2) ∪ A3 = compl (A1 ∪ A2))) ∧
  (¬((A1 ∪ A2) ∪ A3 = Ω)) ∧
  (¬(P (A2 ∪ A3) = 0.8)) ∧
  (P (A1 ∪ A2) ≤ 0.5) :=
by
  sorry

end incorrect_statements_l73_73270


namespace lcm_9_12_15_l73_73977

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l73_73977


namespace zachary_needs_more_money_l73_73242

def cost_in_usd_football (euro_to_usd : ℝ) (football_cost_eur : ℝ) : ℝ :=
  football_cost_eur * euro_to_usd

def cost_in_usd_shorts (gbp_to_usd : ℝ) (shorts_cost_gbp : ℝ) (pairs : ℕ) : ℝ :=
  shorts_cost_gbp * pairs * gbp_to_usd

def cost_in_usd_shoes (shoes_cost_usd : ℝ) : ℝ :=
  shoes_cost_usd

def cost_in_usd_socks (jpy_to_usd : ℝ) (socks_cost_jpy : ℝ) (pairs : ℕ) : ℝ :=
  socks_cost_jpy * pairs * jpy_to_usd

def cost_in_usd_water_bottle (krw_to_usd : ℝ) (water_bottle_cost_krw : ℝ) : ℝ :=
  water_bottle_cost_krw * krw_to_usd

def total_cost_before_discount (cost_football_usd cost_shorts_usd cost_shoes_usd
                                cost_socks_usd cost_water_bottle_usd : ℝ) : ℝ :=
  cost_football_usd + cost_shorts_usd + cost_shoes_usd + cost_socks_usd + cost_water_bottle_usd

def discounted_total_cost (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost * (1 - discount)

def additional_money_needed (discounted_total_cost current_money : ℝ) : ℝ :=
  discounted_total_cost - current_money

theorem zachary_needs_more_money (euro_to_usd : ℝ) (gbp_to_usd : ℝ) (jpy_to_usd : ℝ) (krw_to_usd : ℝ)
  (football_cost_eur : ℝ) (shorts_cost_gbp : ℝ) (pairs_shorts : ℕ) (shoes_cost_usd : ℝ)
  (socks_cost_jpy : ℝ) (pairs_socks : ℕ) (water_bottle_cost_krw : ℝ) (current_money_usd : ℝ)
  (discount : ℝ) : additional_money_needed 
      (discounted_total_cost
          (total_cost_before_discount
            (cost_in_usd_football euro_to_usd football_cost_eur)
            (cost_in_usd_shorts gbp_to_usd shorts_cost_gbp pairs_shorts)
            (cost_in_usd_shoes shoes_cost_usd)
            (cost_in_usd_socks jpy_to_usd socks_cost_jpy pairs_socks)
            (cost_in_usd_water_bottle krw_to_usd water_bottle_cost_krw)) 
          discount) 
      current_money_usd = 7.127214 := 
sorry

end zachary_needs_more_money_l73_73242


namespace least_common_multiple_9_12_15_l73_73964

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l73_73964


namespace find_vector_AM_l73_73756

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
          (A B C M : V)
          (a b : V) -- vectors corresponding to AB and AC

-- Conditions
axiom H1 : M = (2 / 7) • B + (5 / 7) • C
axiom H2 : B - A = a
axiom H3 : C - A = b

-- Proof that the result vector AM is (2/7)a + (5/7)b
theorem find_vector_AM : A - M = (2 / 7) • a + (5 / 7) • b := sorry

end find_vector_AM_l73_73756


namespace exists_cubic_polynomial_with_cubed_roots_l73_73131

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

-- Statement that we need to prove
theorem exists_cubic_polynomial_with_cubed_roots :
  ∃ (b c d : ℝ), ∀ (x : ℝ),
  (f x = 0) → (x^3 = y → x^3^3 + b * x^3^2 + c * x^3 + d = 0) :=
sorry

end exists_cubic_polynomial_with_cubed_roots_l73_73131


namespace find_factor_l73_73640

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end find_factor_l73_73640


namespace root_in_interval_l73_73696

theorem root_in_interval (a b c : ℝ) (h_a : a ≠ 0)
    (h_table : ∀ x y, (x = 1.2 ∧ y = -1.16) ∨ (x = 1.3 ∧ y = -0.71) ∨ (x = 1.4 ∧ y = -0.24) ∨ (x = 1.5 ∧ y = 0.25) ∨ (x = 1.6 ∧ y = 0.76) → y = a * x^2 + b * x + c ) :
  ∃ x₁, 1.4 < x₁ ∧ x₁ < 1.5 ∧ a * x₁^2 + b * x₁ + c = 0 :=
by sorry

end root_in_interval_l73_73696


namespace smaug_silver_coins_l73_73760

theorem smaug_silver_coins :
  ∀ (num_gold num_copper num_silver : ℕ)
  (value_per_silver value_per_gold conversion_factor value_total : ℕ),
  num_gold = 100 →
  num_copper = 33 →
  value_per_silver = 8 →
  value_per_gold = 3 →
  conversion_factor = value_per_gold * value_per_silver →
  value_total = 2913 →
  (num_gold * conversion_factor + num_silver * value_per_silver + num_copper = value_total) →
  num_silver = 60 :=
by
  intros num_gold num_copper num_silver value_per_silver value_per_gold conversion_factor value_total
  intros h1 h2 h3 h4 h5 h6 h_eq
  sorry

end smaug_silver_coins_l73_73760


namespace total_cans_to_collect_l73_73286

def cans_for_project (marthas_cans : ℕ) (additional_cans_needed : ℕ) (total_cans_needed : ℕ) : Prop :=
  ∃ diegos_cans : ℕ, diegos_cans = (marthas_cans / 2) + 10 ∧ 
  total_cans_needed = marthas_cans + diegos_cans + additional_cans_needed

theorem total_cans_to_collect : 
  cans_for_project 90 5 150 :=
by
  -- Insert proof here in actual usage
  sorry

end total_cans_to_collect_l73_73286


namespace bees_flew_in_l73_73310

theorem bees_flew_in (initial_bees additional_bees total_bees : ℕ) 
  (h1 : initial_bees = 16) (h2 : total_bees = 25) 
  (h3 : initial_bees + additional_bees = total_bees) : additional_bees = 9 :=
by sorry

end bees_flew_in_l73_73310


namespace Calvin_mistake_correct_l73_73222

theorem Calvin_mistake_correct (a : ℕ) : 37 + 31 * a = 37 * 31 + a → a = 37 :=
sorry

end Calvin_mistake_correct_l73_73222


namespace apples_minimum_count_l73_73600

theorem apples_minimum_count :
  ∃ n : ℕ, n ≡ 2 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 2 [MOD 5] ∧ n = 62 := by
sorry

end apples_minimum_count_l73_73600


namespace professional_tax_correct_l73_73045

-- Define the total income and professional deductions
def total_income : ℝ := 50000
def professional_deductions : ℝ := 35000

-- Define the tax rates
def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_exp : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

-- Define the expected tax amount
def expected_tax_professional_income : ℝ := 2000

-- Define a function to calculate the professional income tax for self-employed individuals
def calculate_professional_income_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

-- Define the main theorem to assert the correctness of the tax calculation
theorem professional_tax_correct :
  calculate_professional_income_tax total_income tax_rate_professional_income = expected_tax_professional_income :=
by
  sorry

end professional_tax_correct_l73_73045


namespace range_of_a_l73_73344

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1/2)^x = 3 * a + 2 ∧ x < 0) ↔ (a > -1 / 3) :=
by
  sorry

end range_of_a_l73_73344


namespace Kate_has_223_pennies_l73_73897

-- Definition of the conditions
variables (J K : ℕ)
variable (h1 : J = 388)
variable (h2 : J = K + 165)

-- Prove the question equals the answer
theorem Kate_has_223_pennies : K = 223 :=
by
  sorry

end Kate_has_223_pennies_l73_73897


namespace no_two_digit_multiples_of_3_5_7_l73_73861

theorem no_two_digit_multiples_of_3_5_7 : ∀ n : ℕ, 10 ≤ n ∧ n < 100 → ¬ (3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := 
by
  intro n
  intro h
  intro h_div
  sorry

end no_two_digit_multiples_of_3_5_7_l73_73861


namespace product_of_points_l73_73383

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def Chris_rolls : List ℕ := [5, 2, 1, 6]
def Dana_rolls : List ℕ := [6, 2, 3, 3]

def Chris_points : ℕ := (Chris_rolls.map f).sum
def Dana_points : ℕ := (Dana_rolls.map f).sum

theorem product_of_points : Chris_points * Dana_points = 297 := by
  sorry

end product_of_points_l73_73383


namespace total_expenditure_correct_l73_73885

def length : ℝ := 50
def width : ℝ := 30
def cost_per_square_meter : ℝ := 100

def area (L W : ℝ) : ℝ := L * W
def total_expenditure (A C : ℝ) : ℝ := A * C

theorem total_expenditure_correct :
  total_expenditure (area length width) cost_per_square_meter = 150000 := by
  sorry

end total_expenditure_correct_l73_73885


namespace formation_enthalpy_benzene_l73_73497

/-- Define the enthalpy changes based on given conditions --/
def ΔH_acetylene : ℝ := 226.7 -- kJ/mol for C₂H₂
def ΔH_benzene_formation : ℝ := 631.1 -- kJ for reactions forming C₆H₆
def ΔH_benzene_phase_change : ℝ := -33.9 -- kJ for phase change of C₆H₆

/-- Define the enthalpy change of formation for benzene --/
def ΔH_formation_benzene : ℝ := 3 * ΔH_acetylene + ΔH_benzene_formation + ΔH_benzene_phase_change

/-- Theorem stating the heat change in the reaction equals the calculated value --/
theorem formation_enthalpy_benzene :
  ΔH_formation_benzene = -82.9 :=
by
  sorry

end formation_enthalpy_benzene_l73_73497


namespace sum_of_first_five_terms_l73_73062

theorem sum_of_first_five_terms : 
  ∀ (S : ℕ → ℕ) (a : ℕ → ℕ), 
    (a 1 = 1) ∧ 
    (∀ n ≥ 2, S n = S (n - 1) + n + 2) → 
    S 5 = 23 :=
by
  sorry

end sum_of_first_five_terms_l73_73062


namespace monotonic_f_deriv_nonneg_l73_73875

theorem monotonic_f_deriv_nonneg (k : ℝ) :
  (∀ x : ℝ, (1 / 2) < x → k - 1 / x ≥ 0) ↔ k ≥ 2 :=
by sorry

end monotonic_f_deriv_nonneg_l73_73875


namespace smallest_c_for_defined_expression_l73_73345

theorem smallest_c_for_defined_expression :
  ∃ (c : ℤ), (∀ x : ℝ, x^2 + (c : ℝ) * x + 15 ≠ 0) ∧
             (∀ k : ℤ, (∀ x : ℝ, x^2 + (k : ℝ) * x + 15 ≠ 0) → c ≤ k) ∧
             c = -7 :=
by 
  sorry

end smallest_c_for_defined_expression_l73_73345


namespace simplify_expr1_simplify_expr2_simplify_expr3_l73_73521

-- For the first expression
theorem simplify_expr1 (a b : ℝ) : 2 * a - 3 * b + a - 5 * b = 3 * a - 8 * b :=
by
  sorry

-- For the second expression
theorem simplify_expr2 (a : ℝ) : (a^2 - 6 * a) - 3 * (a^2 - 2 * a + 1) + 3 = -2 * a^2 :=
by
  sorry

-- For the third expression
theorem simplify_expr3 (x y : ℝ) : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l73_73521


namespace angle_B_solution_l73_73722

variables (A B C a b c : ℝ)
hypothesis (h : a * Real.cos B - b * Real.cos A = (1 / 2) * c)
noncomputable def prove_angle_B : Prop :=
  B = Real.pi / 6

theorem angle_B_solution (h : a * Real.cos B - b * Real.cos A = (1 / 2) * c) : prove_angle_B A B C a b c :=
sorry

end angle_B_solution_l73_73722


namespace x_squared_minus_y_squared_l73_73010

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end x_squared_minus_y_squared_l73_73010


namespace certain_time_in_seconds_l73_73793

theorem certain_time_in_seconds
  (ratio : ℕ) (minutes : ℕ) (time_in_minutes : ℕ) (seconds_in_a_minute : ℕ)
  (h_ratio : ratio = 8)
  (h_minutes : minutes = 4)
  (h_time : time_in_minutes = minutes)
  (h_conversion : seconds_in_a_minute = 60) :
  time_in_minutes * seconds_in_a_minute = 240 :=
by
  sorry

end certain_time_in_seconds_l73_73793


namespace derivative_of_f_l73_73553

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

theorem derivative_of_f (x : ℝ) : deriv f x = -2 * Real.exp (-x) * Real.sin x :=
by sorry

end derivative_of_f_l73_73553


namespace ceil_sums_l73_73349

theorem ceil_sums (h1: 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2)
                  (h2: 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6)
                  (h3: 18 < Real.sqrt 333 ∧ Real.sqrt 333 < 19):
                  Real.ceil (Real.sqrt 3) + Real.ceil (Real.sqrt 33) + Real.ceil (Real.sqrt 333) = 27 := 
by 
  sorry

end ceil_sums_l73_73349


namespace division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l73_73622

def f (n : ℕ) (a : ℚ) : ℚ := a ^ (2 - n)

theorem division_powers_5_half : f 5 (1/2) = 8 := by
  -- skip the proof
  sorry

theorem division_powers_6_3 : f 6 3 = 1/81 := by
  -- skip the proof
  sorry

theorem division_powers_formula (n : ℕ) (a : ℚ) (h : n > 0) : f n a = a^(2 - n) := by
  -- skip the proof
  sorry

theorem division_powers_combination : f 5 (1/3) * f 4 3 * f 5 (1/2) + f 5 (-1/4) / f 6 (-1/2) = 20 := by
  -- skip the proof
  sorry

end division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l73_73622


namespace reciprocal_roots_l73_73193

theorem reciprocal_roots (a b : ℝ) (h : a ≠ 0) :
  ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + a = 0) ∧ (a * x2^2 + b * x2 + a = 0) → x1 = 1 / x2 ∧ x2 = 1 / x1 :=
by
  intros x1 x2 hroots
  have hsum : x1 + x2 = -b / a := by sorry
  have hprod : x1 * x2 = 1 := by sorry
  sorry

end reciprocal_roots_l73_73193


namespace total_games_l73_73213

theorem total_games (teams : ℕ) (games_per_pair : ℕ) (h_teams : teams = 12) (h_games_per_pair : games_per_pair = 4) : 
  (teams * (teams - 1) / 2) * games_per_pair = 264 :=
by
  sorry

end total_games_l73_73213


namespace diameter_of_large_circle_l73_73293

-- Given conditions
def small_radius : ℝ := 3
def num_small_circles : ℕ := 6

-- Problem statement: Prove the diameter of the large circle
theorem diameter_of_large_circle (r : ℝ) (n : ℕ) (h_radius : r = small_radius) (h_num : n = num_small_circles) :
  ∃ (R : ℝ), R = 9 * 2 := 
sorry

end diameter_of_large_circle_l73_73293


namespace awards_distribution_l73_73041

theorem awards_distribution :
  let num_awards := 6
  let num_students := 3 
  let min_awards_per_student := 2
  (num_awards = 6 ∧ num_students = 3 ∧ min_awards_per_student = 2) →
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end awards_distribution_l73_73041


namespace f_bounded_l73_73052

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (3 * x) = 3 * f x - 4 * (f x) ^ 3

axiom f_continuous_at_zero : ContinuousAt f 0

theorem f_bounded : ∀ x : ℝ, |f x| ≤ 1 :=
by
  sorry

end f_bounded_l73_73052


namespace relationship_between_a_b_c_l73_73246

noncomputable def a : ℝ := 81 ^ 31
noncomputable def b : ℝ := 27 ^ 41
noncomputable def c : ℝ := 9 ^ 61

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_between_a_b_c_l73_73246


namespace math_problem_l73_73126

variable (f g : ℝ → ℝ)
variable (a b x : ℝ)
variable (h_has_derivative_f : ∀ x, Differentiable ℝ f)
variable (h_has_derivative_g : ∀ x, Differentiable ℝ g)
variable (h_deriv_ineq : ∀ x, deriv f x > deriv g x)
variable (h_interval : x ∈ Ioo a b)

theorem math_problem :
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) :=
sorry

end math_problem_l73_73126


namespace smallest_tax_amount_is_professional_income_tax_l73_73048

def total_income : ℝ := 50000.00
def professional_deductions : ℝ := 35000.00

def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_expenditure : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

def ndfl_tax : ℝ := (total_income - professional_deductions) * tax_rate_ndfl
def simplified_tax_income : ℝ := total_income * tax_rate_simplified_income
def simplified_tax_income_minus_expenditure : ℝ := (total_income - professional_deductions) * tax_rate_simplified_income_minus_expenditure
def professional_income_tax : ℝ := total_income * tax_rate_professional_income

theorem smallest_tax_amount_is_professional_income_tax : 
  min (min ndfl_tax (min simplified_tax_income simplified_tax_income_minus_expenditure)) professional_income_tax = professional_income_tax := 
sorry

end smallest_tax_amount_is_professional_income_tax_l73_73048


namespace A_square_or_cube_neg_identity_l73_73899

open Matrix

theorem A_square_or_cube_neg_identity (A : Matrix (Fin 2) (Fin 2) ℚ)
  (n : ℕ) (hn_nonzero : n ≠ 0) (hA_pow_n : A ^ n = -(1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end A_square_or_cube_neg_identity_l73_73899


namespace parabola_equation_l73_73637

theorem parabola_equation {p : ℝ} (hp : 0 < p)
  (h_cond : ∃ A B : ℝ × ℝ, (A.1^2 = 2 * A.2 * p) ∧ (B.1^2 = 2 * B.2 * p) ∧ (A.2 = A.1 - p / 2) ∧ (B.2 = B.1 - p / 2) ∧ (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4))
  : y^2 = 2 * x := sorry

end parabola_equation_l73_73637


namespace l_shape_area_l73_73518

theorem l_shape_area (P : ℝ) (L : ℝ) (x : ℝ)
  (hP : P = 52) 
  (hL : L = 16) 
  (h_x : L + (L - x) + 2 * (16 - x) = P)
  (h_split : 2 * (16 - x) * x = 120) :
  2 * ((16 - x) * x) = 120 :=
by
  -- This is the proof problem statement
  sorry

end l_shape_area_l73_73518


namespace last_two_digits_of_quotient_l73_73922

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem last_two_digits_of_quotient :
  let a : ℤ := 10 ^ 93
  let b : ℤ := 10 ^ 31 + 3
  let x : ℤ := greatest_integer_not_exceeding (a / b : ℝ)
  (x % 100) = 8 :=
by
  sorry

end last_two_digits_of_quotient_l73_73922


namespace volume_of_displaced_water_square_of_displaced_water_volume_l73_73993

-- Definitions for the conditions
def cube_side_length : ℝ := 10
def displaced_water_volume : ℝ := cube_side_length ^ 3
def displaced_water_volume_squared : ℝ := displaced_water_volume ^ 2

-- The Lean theorem statements proving the equivalence
theorem volume_of_displaced_water : displaced_water_volume = 1000 := by
  sorry

theorem square_of_displaced_water_volume : displaced_water_volume_squared = 1000000 := by
  sorry

end volume_of_displaced_water_square_of_displaced_water_volume_l73_73993


namespace find_third_angle_l73_73767

variable (A B C : ℝ)

theorem find_third_angle
  (hA : A = 32)
  (hB : B = 3 * A)
  (hC : C = 2 * A - 12) :
  C = 52 := by
  sorry

end find_third_angle_l73_73767


namespace jennie_rental_cost_is_306_l73_73590

-- Definitions for the given conditions
def weekly_rate_mid_size : ℕ := 190
def daily_rate_mid_size_upto10 : ℕ := 25
def total_rental_days : ℕ := 13
def coupon_discount : ℝ := 0.10

-- Define the cost calculation
def rental_cost (days : ℕ) : ℕ :=
  let weeks := days / 7
  let extra_days := days % 7
  let cost_weeks := weeks * weekly_rate_mid_size
  let cost_extra := extra_days * daily_rate_mid_size_upto10
  cost_weeks + cost_extra

def discount (total : ℝ) (rate : ℝ) : ℝ := total * rate

def final_amount (initial_amount : ℝ) (discount_amount : ℝ) : ℝ := initial_amount - discount_amount

-- Main theorem to prove the final payment amount
theorem jennie_rental_cost_is_306 : 
  final_amount (rental_cost total_rental_days) (discount (rental_cost total_rental_days) coupon_discount) = 306 := 
by
  sorry

end jennie_rental_cost_is_306_l73_73590


namespace product_of_integers_l73_73180

theorem product_of_integers (x y : ℤ) (h1 : Int.gcd x y = 5) (h2 : Int.lcm x y = 60) : x * y = 300 :=
by
  sorry

end product_of_integers_l73_73180


namespace cost_small_and_large_puzzle_l73_73800

-- Define the cost of a large puzzle L and the cost equation for large and small puzzles
def cost_large_puzzle : ℤ := 15

def cost_equation (S : ℤ) : Prop := cost_large_puzzle + 3 * S = 39

-- Theorem to prove the total cost of a small puzzle and a large puzzle together
theorem cost_small_and_large_puzzle : ∃ S : ℤ, cost_equation S ∧ (S + cost_large_puzzle = 23) :=
by
  sorry

end cost_small_and_large_puzzle_l73_73800


namespace percentage_less_than_l73_73510

namespace PercentProblem

noncomputable def A (C : ℝ) : ℝ := 0.65 * C
noncomputable def B (C : ℝ) : ℝ := 0.8923076923076923 * A C

theorem percentage_less_than (C : ℝ) (hC : C ≠ 0) : (C - B C) / C = 0.42 :=
by
  sorry

end PercentProblem

end percentage_less_than_l73_73510


namespace find_g1_gneg1_l73_73876

variables {f g : ℝ → ℝ}

theorem find_g1_gneg1 (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
                      (h2 : f (-2) = f 1 ∧ f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end find_g1_gneg1_l73_73876


namespace determine_a_l73_73259

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end determine_a_l73_73259


namespace find_p_l73_73859

-- Assume the parametric equations and conditions specified in the problem.
noncomputable def parabola_eqns (p t : ℝ) (M E F : ℝ × ℝ) :=
  ∃ m : ℝ,
    (M = (6, m)) ∧
    (E = (-p / 2, m)) ∧
    (F = (p / 2, 0)) ∧
    (m^2 = 6 * p) ∧
    (|E.1 - F.1|^2 + |E.2 - F.2|^2 = |F.1 - M.1|^2 + |F.2 - M.2|^2) ∧
    (|F.1 - M.1|^2 + |F.2 - M.2|^2 = (F.1 + p / 2)^2 + (F.2 - m)^2)

theorem find_p {p t : ℝ} {M E F : ℝ × ℝ} (h : parabola_eqns p t M E F) : p = 4 :=
by
  sorry

end find_p_l73_73859


namespace x_minus_y_value_l73_73555

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) : x - y = 1 ∨ x - y = 5 := by
  sorry

end x_minus_y_value_l73_73555


namespace probability_single_trial_l73_73141

open Real

theorem probability_single_trial :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (1 - p)^4 = 16 / 81 ∧ p = 1 / 3 :=
by
  -- The proof steps have been skipped.
  sorry

end probability_single_trial_l73_73141


namespace volume_OABC_is_l73_73188

noncomputable def volume_tetrahedron_ABC (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) : ℝ :=
  1 / 6 * a * b * c

theorem volume_OABC_is (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) :
  volume_tetrahedron_ABC a b c hx hy hz = (5 / 6) * Real.sqrt 30.375 :=
by
  sorry

end volume_OABC_is_l73_73188


namespace area_of_shaded_region_l73_73727

def parallelogram_exists (EFGH : Type) : Prop :=
  ∃ (E F G H : EFGH) (EJ JH EH : ℝ) (height : ℝ), EJ + JH = EH ∧ EH = 12 ∧ JH = 8 ∧ height = 10

theorem area_of_shaded_region {EFGH : Type} (h : parallelogram_exists EFGH) : 
  ∃ (area_shaded : ℝ), area_shaded = 100 := 
by
  sorry

end area_of_shaded_region_l73_73727


namespace isosceles_triangle_angles_l73_73733

noncomputable 
def is_triangle_ABC_isosceles (A B C : ℝ) (alpha beta : ℝ) (AB AC : ℝ) 
  (h1 : AB = AC) (h2 : alpha = 2 * beta) : Prop :=
  180 - 3 * beta = C ∧ C / 2 = 90 - 1.5 * beta

theorem isosceles_triangle_angles (A B C C1 C2 : ℝ) (alpha beta : ℝ) (AB AC : ℝ)
  (h1 : AB = AC) (h2 : alpha = 2 * beta) :
  (180 - 3 * beta) / 2 = 90 - 1.5 * beta :=
by sorry

end isosceles_triangle_angles_l73_73733


namespace cups_of_rice_morning_l73_73163

variable (cupsMorning : Nat) -- Number of cups of rice Robbie eats in the morning
variable (cupsAfternoon : Nat := 2) -- Cups of rice in the afternoon
variable (cupsEvening : Nat := 5) -- Cups of rice in the evening
variable (fatPerCup : Nat := 10) -- Fat in grams per cup of rice
variable (weeklyFatIntake : Nat := 700) -- Total fat in grams per week

theorem cups_of_rice_morning :
  ((cupsMorning + cupsAfternoon + cupsEvening) * fatPerCup) = (weeklyFatIntake / 7) → cupsMorning = 3 :=
  by
    sorry

end cups_of_rice_morning_l73_73163


namespace largest_initial_number_l73_73392

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l73_73392


namespace general_term_a_l73_73597

noncomputable def S (n : ℕ) : ℤ := 3^n - 2

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2 * 3^(n - 1)

theorem general_term_a (n : ℕ) (hn : n > 0) : a n = if n = 1 then 1 else 2 * 3^(n - 1) := by
  -- Proof goes here
  sorry

end general_term_a_l73_73597


namespace lcm_of_9_12_15_is_180_l73_73961

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l73_73961


namespace temperature_rise_result_l73_73306

def initial_temperature : ℤ := -2
def rise : ℤ := 3

theorem temperature_rise_result : initial_temperature + rise = 1 := 
by 
  sorry

end temperature_rise_result_l73_73306


namespace probability_reach_2C_l73_73507

noncomputable def f (x C : ℝ) : ℝ :=
  x / (2 * C)

theorem probability_reach_2C (x C : ℝ) (hC : 0 < C) (hx : 0 < x ∧ x < 2 * C) :
  f x C = x / (2 * C) := 
by
  sorry

end probability_reach_2C_l73_73507


namespace find_a_given_solution_set_l73_73304

theorem find_a_given_solution_set :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 ↔ x^2 + a * x + 6 ≤ 0) → a = -5 :=
by
  sorry

end find_a_given_solution_set_l73_73304


namespace compute_expression_l73_73338

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by
  sorry

end compute_expression_l73_73338


namespace fractional_part_of_blue_square_four_changes_l73_73096

theorem fractional_part_of_blue_square_four_changes 
  (initial_area : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ (a : ℝ), f a = (8 / 9) * a) :
  (f^[4]) initial_area / initial_area = 4096 / 6561 :=
by
  sorry

end fractional_part_of_blue_square_four_changes_l73_73096


namespace math_problem_l73_73200

theorem math_problem :
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := 
by
  sorry

end math_problem_l73_73200


namespace max_slope_of_circle_l73_73303

theorem max_slope_of_circle (x y : ℝ) 
  (h : x^2 + y^2 - 6 * x - 6 * y + 12 = 0) : 
  ∃ k : ℝ, k = 3 + 2 * Real.sqrt 2 ∧ ∀ k' : ℝ, (x = 0 → k' = 0) ∧ (x ≠ 0 → y = k' * x → k' ≤ k) :=
sorry

end max_slope_of_circle_l73_73303


namespace circumscribed_circle_radius_of_rectangle_l73_73183

theorem circumscribed_circle_radius_of_rectangle 
  (a b : ℝ) 
  (h1: a = 1) 
  (angle_between_diagonals : ℝ) 
  (h2: angle_between_diagonals = 60) : 
  ∃ R, R = 1 :=
by 
  sorry

end circumscribed_circle_radius_of_rectangle_l73_73183


namespace no_real_solutions_for_m_l73_73015

theorem no_real_solutions_for_m (m : ℝ) :
  ∃! m, (4 * m + 2) ^ 2 - 4 * m = 0 → false :=
by 
  sorry

end no_real_solutions_for_m_l73_73015


namespace reciprocal_of_neg_two_l73_73058

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end reciprocal_of_neg_two_l73_73058


namespace ratio_of_sum_l73_73552

theorem ratio_of_sum (a b c : ℚ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := 
by
  sorry

end ratio_of_sum_l73_73552


namespace tangerines_times_persimmons_l73_73311

-- Definitions from the problem conditions
def apples : ℕ := 24
def tangerines : ℕ := 6 * apples
def persimmons : ℕ := 8

-- Statement to be proved
theorem tangerines_times_persimmons :
  tangerines / persimmons = 18 := by
  sorry

end tangerines_times_persimmons_l73_73311


namespace initialNumberMembers_l73_73080

-- Define the initial number of members in the group
def initialMembers (n : ℕ) : Prop :=
  let W := n * 48 -- Initial total weight
  let newWeight := W + 78 + 93 -- New total weight after two members join
  let newAverageWeight := (n + 2) * 51 -- New total weight based on the new average weight
  newWeight = newAverageWeight -- The condition that the new total weights are equal

-- Theorem stating that the initial number of members is 23
theorem initialNumberMembers : initialMembers 23 :=
by
  -- Placeholder for proof steps
  sorry

end initialNumberMembers_l73_73080


namespace value_of_t_plus_k_l73_73127

noncomputable def f (x t : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

theorem value_of_t_plus_k (k t : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x, f x t = 2 * x - 1)
  (h3 : ∃ x₁ x₂, f x₁ t = 2 * x₁ - 1 ∧ f x₂ t = 2 * x₂ - 1) :
  t + k = 7 :=
sorry

end value_of_t_plus_k_l73_73127


namespace turtles_in_lake_l73_73442

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l73_73442


namespace study_group_books_l73_73502

theorem study_group_books (x n : ℕ) (h1 : n = 5 * x - 2) (h2 : n = 4 * x + 3) : x = 5 ∧ n = 23 := by
  sorry

end study_group_books_l73_73502


namespace sequences_properties_l73_73704

-- Definitions for properties P and P'
def is_property_P (seq : List ℕ) : Prop := sorry
def is_property_P' (seq : List ℕ) : Prop := sorry

-- Define sequences
def sequence1 := [1, 2, 3, 1]
def sequence2 := [1, 234, 5]  -- Extend as needed

-- Conditions
def bn_is_permutation_of_an (a b : List ℕ) : Prop := sorry -- Placeholder for permutation check

-- Main Statement 
theorem sequences_properties :
  is_property_P sequence1 ∧
  is_property_P' sequence2 := 
by
  sorry

-- Additional theorem to check permutation if needed
-- theorem permutation_check :
--  bn_is_permutation_of_an sequence1 sequence2 :=
-- by
--  sorry

end sequences_properties_l73_73704


namespace quadratic_inequality_solution_set_l73_73140

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a < 0)
  (h2 : -1 + 2 = b / a) (h3 : -1 * 2 = c / a) :
  (b = a) ∧ (c = -2 * a) :=
by
  sorry

end quadratic_inequality_solution_set_l73_73140


namespace total_amount_spent_l73_73538

variables (D B : ℝ)

-- Conditions
def condition1 : Prop := B = 1.5 * D
def condition2 : Prop := D = B - 15

-- Question: Prove that the total amount they spent together is 75.00
theorem total_amount_spent (h1 : condition1 D B) (h2 : condition2 D B) : B + D = 75 :=
sorry

end total_amount_spent_l73_73538


namespace game_goal_impossible_l73_73606

-- Definition for initial setup
def initial_tokens : ℕ := 2013
def initial_piles : ℕ := 1

-- Definition for the invariant
def invariant (tokens piles : ℕ) : ℕ := tokens + piles

-- Initial value of the invariant constant
def initial_invariant : ℕ :=
  invariant initial_tokens initial_piles

-- Goal is to check if the final configuration is possible
theorem game_goal_impossible (n : ℕ) :
  (invariant (3 * n) n = initial_invariant) → false :=
by
  -- The invariant states 4n = initial_invariant which is 2014.
  -- Thus, we need to check if 2014 / 4 results in an integer.
  have invariant_expr : 4 * n = 2014 := by sorry
  have n_is_integer : 2014 % 4 = 0 := by sorry
  sorry

end game_goal_impossible_l73_73606


namespace triangle_construction_l73_73541

-- Define the problem statement in Lean
theorem triangle_construction (a b c : ℝ) :
  correct_sequence = [3, 1, 4, 2] :=
sorry

end triangle_construction_l73_73541


namespace point_A_coordinates_l73_73177

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem point_A_coordinates (h1 : a > 0) (h2 : a ≠ 1) (hf : ∀ x, f x = a^(x - 1)) :
  f 1 = 1 :=
by
  sorry

end point_A_coordinates_l73_73177


namespace min_triangle_perimeter_proof_l73_73415

noncomputable def min_triangle_perimeter (l m n : ℕ) : ℕ :=
  if l > m ∧ m > n ∧ (3^l % 10000 = 3^m % 10000) ∧ (3^m % 10000 = 3^n % 10000) then
    l + m + n
  else
    0

theorem min_triangle_perimeter_proof : ∃ (l m n : ℕ), l > m ∧ m > n ∧ 
  (3^l % 10000 = 3^m % 10000) ∧
  (3^m % 10000 = 3^n % 10000) ∧ min_triangle_perimeter l m n = 3003 :=
  sorry

end min_triangle_perimeter_proof_l73_73415


namespace max_min_product_l73_73205

theorem max_min_product (A B : ℕ) (h : A + B = 100) : 
  (∃ (maxProd : ℕ), maxProd = 2500 ∧ (∀ (A B : ℕ), A + B = 100 → A * B ≤ maxProd)) ∧
  (∃ (minProd : ℕ), minProd = 0 ∧ (∀ (A B : ℕ), A + B = 100 → minProd ≤ A * B)) :=
by 
  -- Proof omitted
  sorry

end max_min_product_l73_73205


namespace two_students_one_common_material_l73_73452

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l73_73452


namespace dot_product_parallel_a_b_l73_73860

noncomputable def a : ℝ × ℝ := (-1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Definition of parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2)

-- Given conditions and result to prove
theorem dot_product_parallel_a_b : ∀ (x : ℝ), parallel a (b x) → x = -2 → (a.1 * (b x).1 + a.2 * (b x).2) = -4 := 
by
  intros x h_parallel h_x
  subst h_x
  sorry

end dot_product_parallel_a_b_l73_73860


namespace two_students_choose_materials_l73_73470

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l73_73470


namespace lucille_paint_cans_needed_l73_73901

theorem lucille_paint_cans_needed :
  let wall1_area := 3 * 2
  let wall2_area := 3 * 2
  let wall3_area := 5 * 2
  let wall4_area := 4 * 2
  let total_area := wall1_area + wall2_area + wall3_area + wall4_area
  let coverage_per_can := 2
  let cans_needed := total_area / coverage_per_can
  cans_needed = 15 := 
by 
  sorry

end lucille_paint_cans_needed_l73_73901


namespace radian_measure_15_degrees_l73_73314

theorem radian_measure_15_degrees : (15 * (Real.pi / 180)) = (Real.pi / 12) :=
by
  sorry

end radian_measure_15_degrees_l73_73314


namespace angle_sum_impossible_l73_73551

theorem angle_sum_impossible (A1 A2 A3 : ℝ) (h : A1 + A2 + A3 = 180) :
  ¬ ((A1 > 90 ∧ A2 > 90 ∧ A3 < 90) ∨ (A1 > 90 ∧ A3 > 90 ∧ A2 < 90) ∨ (A2 > 90 ∧ A3 > 90 ∧ A1 < 90)) :=
sorry

end angle_sum_impossible_l73_73551


namespace largest_initial_number_l73_73391

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l73_73391


namespace friends_prove_l73_73269

theorem friends_prove (a b c d : ℕ) (h1 : 3^a * 7^b = 3^c * 7^d) (h2 : 3^a * 7^b = 21) :
  (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by {
  sorry
}

end friends_prove_l73_73269


namespace num_ways_choose_materials_l73_73458

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l73_73458


namespace probability_of_3_rainy_days_l73_73880

noncomputable def rainy_probability : ℝ :=
  let n := 4
  let k := 3
  let p := 0.5
  binomial_pdf p n k

theorem probability_of_3_rainy_days : rainy_probability = 0.25 := by
  sorry

end probability_of_3_rainy_days_l73_73880


namespace smallest_four_digit_divisible_by_6_l73_73480

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l73_73480


namespace shingles_needed_l73_73921

structure Dimensions where
  length : ℝ
  width : ℝ

def area (d : Dimensions) : ℝ :=
  d.length * d.width

def houseDimensions : Dimensions := { length := 20.5, width := 10 }
def porchDimensions : Dimensions := { length := 6, width := 4.5 }

def totalArea (d1 d2 : Dimensions) : ℝ :=
  area d1 + area d2

theorem shingles_needed :
  totalArea houseDimensions porchDimensions = 232 :=
by
  simp [totalArea, area, houseDimensions, porchDimensions]
  norm_num
  sorry

end shingles_needed_l73_73921


namespace nth_monomial_correct_l73_73814

-- Definitions of the sequence of monomials

def coeff (n : ℕ) : ℕ := 3 * n + 2
def exponent (n : ℕ) : ℕ := n

def nth_monomial (n : ℕ) (a : ℕ) : ℕ := (coeff n) * (a ^ (exponent n))

-- Theorem statement
theorem nth_monomial_correct (n : ℕ) (a : ℕ) : nth_monomial n a = (3 * n + 2) * (a ^ n) :=
by
  sorry

end nth_monomial_correct_l73_73814


namespace difference_area_octagon_shaded_l73_73099

-- Definitions based on the given conditions
def radius : ℝ := 10
def pi_value : ℝ := 3.14

-- Lean statement for the given proof problem
theorem difference_area_octagon_shaded :
  ∃ S_octagon S_shaded, 
    10^2 * pi_value = 314 ∧
    (20 / 2^0.5)^2 = 200 ∧
    S_octagon = 200 - 114 ∧ -- transposed to reverse engineering step
    S_shaded = 28 ∧ -- needs refinement
    S_octagon - S_shaded = 86 :=
sorry

end difference_area_octagon_shaded_l73_73099


namespace smallest_D_for_inequality_l73_73840

theorem smallest_D_for_inequality :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧ 
           D = -Real.sqrt (72 / 11) :=
by
  sorry

end smallest_D_for_inequality_l73_73840


namespace find_transform_l73_73308

structure Vector3D (α : Type) := (x y z : α)

def T (u : Vector3D ℝ) : Vector3D ℝ := sorry

axiom linearity (a b : ℝ) (u v : Vector3D ℝ) : T (Vector3D.mk (a * u.x + b * v.x) (a * u.y + b * v.y) (a * u.z + b * v.z)) = 
                      Vector3D.mk (a * (T u).x + b * (T v).x) (a * (T u).y + b * (T v).y) (a * (T u).z + b * (T v).z)

axiom cross_product (u v : Vector3D ℝ) : T (Vector3D.mk (u.y * v.z - u.z * v.y) (u.z * v.x - u.x * v.z) (u.x * v.y - u.y * v.x)) = 
                    (Vector3D.mk ((T u).y * (T v).z - (T u).z * (T v).y) ((T u).z * (T v).x - (T u).x * (T v).z) ((T u).x * (T v).y - (T u).y * (T v).x))

axiom transform1 : T (Vector3D.mk 3 3 7) = Vector3D.mk 2 (-4) 5
axiom transform2 : T (Vector3D.mk (-2) 5 4) = Vector3D.mk 6 1 0

theorem find_transform : T (Vector3D.mk 5 15 11) = Vector3D.mk a b c := sorry

end find_transform_l73_73308


namespace milton_zoology_books_l73_73753

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l73_73753


namespace sum_of_two_numbers_l73_73936

theorem sum_of_two_numbers (a b S : ℤ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 :=
by
  sorry

end sum_of_two_numbers_l73_73936


namespace blocks_left_l73_73987

/-- Problem: Randy has 78 blocks. He uses 19 blocks to build a tower. Prove that he has 59 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (remaining_blocks : ℕ) : initial_blocks = 78 → used_blocks = 19 → remaining_blocks = initial_blocks - used_blocks → remaining_blocks = 59 :=
by
  sorry

end blocks_left_l73_73987


namespace math_problem_l73_73845

variables {x y : ℝ}

theorem math_problem (h1 : x + y = 6) (h2 : x * y = 5) :
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y) ^ 2 = 16) ∧ (x ^ 2 + y ^ 2 = 26) :=
by
  sorry

end math_problem_l73_73845


namespace smallest_four_digit_number_divisible_by_6_l73_73484

theorem smallest_four_digit_number_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m % 6 = 0) → n ≤ m :=
begin
  use 1002,
  split,
  { exact nat.le_succ 999,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.le_succ 1001,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  { intros m h1,
    exact le_of_lt_iff.2 (by linarith) }
end

end smallest_four_digit_number_divisible_by_6_l73_73484


namespace exists_x_eq_28_l73_73012

theorem exists_x_eq_28 : ∃ x : Int, 45 - (x - (37 - (15 - 16))) = 55 ↔ x = 28 := 
by
  sorry

end exists_x_eq_28_l73_73012


namespace transylvanian_human_truth_transylvanian_vampire_lie_l73_73589

-- Definitions of predicates for human and vampire behavior
def is_human (A : Type) : Prop := ∀ (X : Prop), (A → X) → X
def is_vampire (A : Type) : Prop := ∀ (X : Prop), (A → X) → ¬X

-- Lean definitions for the problem
theorem transylvanian_human_truth (A : Type) (X : Prop) (h_human : is_human A) (h_says_true : A → X) :
  X :=
by sorry

theorem transylvanian_vampire_lie (A : Type) (X : Prop) (h_vampire : is_vampire A) (h_says_true : A → X) :
  ¬X :=
by sorry

end transylvanian_human_truth_transylvanian_vampire_lie_l73_73589


namespace largest_initial_number_l73_73403

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l73_73403


namespace u2008_is_5898_l73_73740

-- Define the sequence as given in the problem.
def u (n : ℕ) : ℕ := sorry  -- The nth term of the sequence defined in the problem.

-- The main theorem stating u_{2008} = 5898.
theorem u2008_is_5898 : u 2008 = 5898 := sorry

end u2008_is_5898_l73_73740


namespace lcm_9_12_15_l73_73954

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l73_73954


namespace right_square_pyramid_height_l73_73692

theorem right_square_pyramid_height :
  ∀ (h x : ℝ),
    let topBaseSide := 3
    let bottomBaseSide := 6
    let lateralArea := 4 * (1/2) * (topBaseSide + bottomBaseSide) * x
    let baseAreasSum := topBaseSide^2 + bottomBaseSide^2
    lateralArea = baseAreasSum →
    x = 5/2 →
    h = 2 :=
by
  intros h x topBaseSide bottomBaseSide lateralArea baseAreasSum lateralEq baseEq
  sorry

end right_square_pyramid_height_l73_73692


namespace turtles_in_lake_l73_73441

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l73_73441


namespace find_value_of_square_sums_l73_73695

variable (x y z : ℝ)

-- Define the conditions
def weighted_arithmetic_mean := (2 * x + 2 * y + 3 * z) / 8 = 9
def weighted_geometric_mean := Real.rpow (x^2 * y^2 * z^3) (1 / 7) = 6
def weighted_harmonic_mean := 7 / ((2 / x) + (2 / y) + (3 / z)) = 4

-- State the theorem to be proved
theorem find_value_of_square_sums
  (h1 : weighted_arithmetic_mean x y z)
  (h2 : weighted_geometric_mean x y z)
  (h3 : weighted_harmonic_mean x y z) :
  x^2 + y^2 + z^2 = 351 :=
by sorry

end find_value_of_square_sums_l73_73695


namespace radius_of_inscribed_circle_l73_73566

theorem radius_of_inscribed_circle (a b x : ℝ) (hx : 0 < x) 
  (h_side_length : a > 20) 
  (h_TM : a = x + 8) 
  (h_OM : b = x + 9) 
  (h_Pythagorean : (a - 8)^2 + (b - 9)^2 = x^2) :
  x = 29 :=
by
  -- Assume all conditions and continue to the proof part.
  sorry

end radius_of_inscribed_circle_l73_73566


namespace conditional_probability_problem_l73_73843

-- Definitions of the problem:
variable {Ω : Type*}
variable choiceSpace : set (Ω × Ω) := { (a, b) | a ≠ b ∧ a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} }
variable eventA : set (Ω × Ω) := { (a, b) | (a + b).even }
variable eventB : set (Ω × Ω) := { (a, b) | a % 2 = 0 ∧ b % 2 = 0 }

-- Lean statement of the proof problem:
theorem conditional_probability_problem : 
  probability_space.cond_prob choiceSpace eventA eventB = 1/4 := 
sorry

end conditional_probability_problem_l73_73843


namespace measure_angle_E_l73_73581

-- Definitions based on conditions
variables {p q : Type} {A B E : ℝ}

noncomputable def measure_A (A B : ℝ) : ℝ := A
noncomputable def measure_B (A B : ℝ) : ℝ := 9 * A
noncomputable def parallel_lines (p q : Type) : Prop := true

-- Condition: measure of angle A is 1/9 of the measure of angle B
axiom angle_condition : A = (1 / 9) * B

-- Condition: p is parallel to q
axiom parallel_condition : parallel_lines p q

-- Prove that the measure of angle E is 18 degrees
theorem measure_angle_E (y : ℝ) (h1 : A = y) (h2 : B = 9 * y) : E = 18 :=
by
  sorry

end measure_angle_E_l73_73581


namespace find_s_l73_73674

theorem find_s (s : ℝ) (t : ℝ) (h1 : t = 4) (h2 : t = 12 * s^2 + 2 * s) : s = 0.5 ∨ s = -2 / 3 :=
by
  sorry

end find_s_l73_73674


namespace tangent_line_b_value_l73_73853

theorem tangent_line_b_value (b : ℝ) : 
  (∃ pt : ℝ × ℝ, (pt.1)^2 + (pt.2)^2 = 25 ∧ pt.1 - pt.2 + b = 0)
  ↔ b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by
  sorry

end tangent_line_b_value_l73_73853


namespace milton_books_l73_73749

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l73_73749


namespace unit_vector_norm_equal_l73_73849

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2D Euclidean space for simplicity

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem unit_vector_norm_equal {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : is_unit_vector a) (hb : is_unit_vector b) : ‖a‖ = ‖b‖ :=
by 
  sorry

end unit_vector_norm_equal_l73_73849


namespace problem_part1_problem_part2_l73_73715

open Real

theorem problem_part1 (α : ℝ) (h : (sin (π - α) * cos (2 * π - α)) / (tan (π - α) * sin (π / 2 + α) * cos (π / 2 - α)) = 1 / 2) :
  (cos α - 2 * sin α) / (3 * cos α + sin α) = 5 := sorry

theorem problem_part2 (α : ℝ) (h : tan α = -2) :
  1 - 2 * sin α * cos α + cos α ^ 2 = 2 / 5 := sorry

end problem_part1_problem_part2_l73_73715


namespace last_number_is_four_l73_73904

theorem last_number_is_four (a b c d e last_number : ℕ) (h_counts : a = 6 ∧ b = 12 ∧ c = 1 ∧ d = 12 ∧ e = 7)
    (h_mean : (a + b + c + d + e + last_number) / 6 = 7) : last_number = 4 := 
sorry

end last_number_is_four_l73_73904


namespace shortest_distance_D_to_V_l73_73432

-- Define distances
def distance_A_to_G : ℕ := 12
def distance_G_to_B : ℕ := 10
def distance_A_to_B : ℕ := 8
def distance_D_to_G : ℕ := 15
def distance_V_to_G : ℕ := 17

-- Prove the shortest distance from Dasha to Vasya
theorem shortest_distance_D_to_V : 
  let dD_to_V := distance_D_to_G + distance_V_to_G
  let dAlt := dD_to_V + distance_A_to_B - distance_A_to_G - distance_G_to_B
  (dAlt < dD_to_V) -> dAlt = 18 :=
by
  sorry

end shortest_distance_D_to_V_l73_73432


namespace find_positive_Y_for_nine_triangle_l73_73677

def triangle_relation (X Y : ℝ) : ℝ := X^2 + 3 * Y^2

theorem find_positive_Y_for_nine_triangle (Y : ℝ) : (9^2 + 3 * Y^2 = 360) → Y = Real.sqrt 93 := 
by
  sorry

end find_positive_Y_for_nine_triangle_l73_73677


namespace power_of_product_l73_73768

variable (a b : ℝ) (m : ℕ)
theorem power_of_product (h : 0 < m) : (a * b)^m = a^m * b^m :=
sorry

end power_of_product_l73_73768


namespace comprehensive_score_l73_73376

theorem comprehensive_score :
  let w_c := 0.4
  let w_u := 0.6
  let s_c := 80
  let s_u := 90
  s_c * w_c + s_u * w_u = 86 :=
by
  sorry

end comprehensive_score_l73_73376


namespace jane_savings_l73_73799

noncomputable def cost_promotion_A (price: ℝ) : ℝ :=
  price + (price / 2)

noncomputable def cost_promotion_B (price: ℝ) : ℝ :=
  price + (price - (price * 0.25))

theorem jane_savings (price : ℝ) (h_price_pos : 0 < price) : 
  cost_promotion_B price - cost_promotion_A price = 12.5 :=
by
  let price := 50
  unfold cost_promotion_A
  unfold cost_promotion_B
  norm_num
  sorry

end jane_savings_l73_73799


namespace find_t_l73_73709

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l73_73709


namespace problem_a_l73_73204

def part_a : Prop :=
  ∃ (tokens : Finset (Fin 4 × Fin 4)), 
    tokens.card = 7 ∧ 
    (∀ (rows : Finset (Fin 4)) (cols : Finset (Fin 4)), rows.card = 2 → cols.card = 2 → 
      ∃ (token : (Fin 4 × Fin 4)), token ∈ tokens ∧ token.1 ∉ rows ∧ token.2 ∉ cols)

theorem problem_a : part_a :=
  sorry

end problem_a_l73_73204


namespace fifteenth_number_in_base_5_l73_73271

theorem fifteenth_number_in_base_5 :
  ∃ n : ℕ, n = 15 ∧ (n : ℕ) = 3 * 5^1 + 0 * 5^0 :=
by
  sorry

end fifteenth_number_in_base_5_l73_73271


namespace area_of_rhombus_l73_73139

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 10) : 
  1 / 2 * d1 * d2 = 30 :=
by 
  rw [h1, h2]
  norm_num

end area_of_rhombus_l73_73139


namespace linear_equation_solution_l73_73136

theorem linear_equation_solution (a b : ℤ) (x y : ℤ) (h1 : x = 2) (h2 : y = -1) (h3 : a * x + b * y = -1) : 
  1 + 2 * a - b = 0 :=
by
  sorry

end linear_equation_solution_l73_73136


namespace num_true_statements_is_two_l73_73168

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem num_true_statements_is_two :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 2 :=
by
  sorry

end num_true_statements_is_two_l73_73168


namespace find_a_of_complex_eq_l73_73267

theorem find_a_of_complex_eq (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * (⟨1, -a⟩ : ℂ) = 2) : a = 1 :=
by
  sorry

end find_a_of_complex_eq_l73_73267


namespace distance_to_focus_F2_l73_73176

noncomputable def ellipse_foci_distance
  (x y : ℝ)
  (a b : ℝ) 
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1) 
  (a2 : a^2 = 9) 
  (b2 : b^2 = 2) 
  (F1 P : ℝ) 
  (h_P_on_ellipse : F1 = 3) 
  (h_PF1 : F1 = 4) 
: ℝ :=
  2

-- theorem to prove the problem statement
theorem distance_to_focus_F2
  (x y : ℝ)
  (a b : ℝ)
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1)
  (a2 : a^2 = 9)
  (b2 : b^2 = 2)
  (F1 P : ℝ)
  (h_P_on_ellipse : F1 = 3)
  (h_PF1 : F1 = 4)
: F2 = 2 :=
by
  sorry

end distance_to_focus_F2_l73_73176


namespace jason_total_payment_l73_73573

def total_cost (shorts jacket shoes socks tshirts : ℝ) : ℝ :=
  shorts + jacket + shoes + socks + tshirts

def discount_amount (total : ℝ) (discount_rate : ℝ) : ℝ :=
  total * discount_rate

def total_after_discount (total discount : ℝ) : ℝ :=
  total - discount

def sales_tax_amount (total : ℝ) (tax_rate : ℝ) : ℝ :=
  total * tax_rate

def final_amount (total after_discount tax : ℝ) : ℝ :=
  after_discount + tax

theorem jason_total_payment :
  let shorts := 14.28
  let jacket := 4.74
  let shoes := 25.95
  let socks := 6.80
  let tshirts := 18.36
  let discount_rate := 0.15
  let tax_rate := 0.07
  let total := total_cost shorts jacket shoes socks tshirts
  let discount := discount_amount total discount_rate
  let after_discount := total_after_discount total discount
  let tax := sales_tax_amount after_discount tax_rate
  let final := final_amount total after_discount tax
  final = 63.78 :=
by
  sorry

end jason_total_payment_l73_73573


namespace common_ratio_l73_73575

theorem common_ratio (a_3 S_3 : ℝ) (q : ℝ) 
  (h1 : a_3 = 3 / 2) 
  (h2 : S_3 = 9 / 2)
  (h3 : S_3 = (1 + q + q^2) * a_3 / q^2) :
  q = 1 ∨ q = -1 / 2 := 
by 
  sorry

end common_ratio_l73_73575


namespace number_of_ways_l73_73462

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l73_73462


namespace product_of_repeating_decimal_l73_73670

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l73_73670


namespace num_points_on_ellipse_l73_73181

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 9 = 1

def line (x y : ℝ) : Prop :=
  x / 3 + y / 4 = 1

theorem num_points_on_ellipse (A B : ℝ × ℝ) (A_intersects : ellipse A.1 A.2 ∧ line A.1 A.2)
                             (B_intersects : ellipse B.1 B.2 ∧ line B.1 B.2) :
  ∃ (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (area_eq : area_of_triangle P A B = 3), ∃! (P : ℝ × ℝ), (hP ∧ area_eq) :=
sorry

end num_points_on_ellipse_l73_73181


namespace triplet_zero_solution_l73_73109

theorem triplet_zero_solution (x y z : ℝ) 
  (h1 : x^3 + y = z^2) 
  (h2 : y^3 + z = x^2) 
  (h3 : z^3 + x = y^2) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end triplet_zero_solution_l73_73109


namespace part_I_part_II_l73_73129

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem part_I (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  ∃ x ∈ Set.Ioo m (m + 1), ∀ y ∈ Set.Ioo m (m + 1), f y ≤ f x := sorry

theorem part_II (x : ℝ) (h : 1 < x) :
  (x + 1) * (x + Real.exp (-x)) * f x > 2 * (1 + 1 / Real.exp 1) := sorry

end part_I_part_II_l73_73129


namespace square_perimeter_l73_73810

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l73_73810


namespace hyperbola_transformation_l73_73312

def equation_transform (x y : ℝ) : Prop :=
  y = (1 - 3 * x) / (2 * x - 1)

def coordinate_shift (x y X Y : ℝ) : Prop :=
  X = x - 0.5 ∧ Y = y + 1.5

theorem hyperbola_transformation (x y X Y : ℝ) :
  equation_transform x y →
  coordinate_shift x y X Y →
  (X * Y = -0.25) :=
by
  sorry

end hyperbola_transformation_l73_73312


namespace fg_neg_one_eq_neg_eight_l73_73576

def f (x : ℤ) : ℤ := x - 4
def g (x : ℤ) : ℤ := x^2 + 2*x - 3

theorem fg_neg_one_eq_neg_eight : f (g (-1)) = -8 := by
  sorry

end fg_neg_one_eq_neg_eight_l73_73576


namespace erica_pie_fraction_as_percentage_l73_73832

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end erica_pie_fraction_as_percentage_l73_73832


namespace milton_books_l73_73747

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l73_73747


namespace solution_set_of_inequality_l73_73438

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x^2 + 2 < x} = {x : ℝ | x < -2 / 3} ∪ {x : ℝ | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l73_73438


namespace monotonic_intervals_harmonic_log_inequality_l73_73857

noncomputable theory
open Real

-- Define the function f(x) as given
def f (x : ℝ) (a : ℝ) : ℝ := log x - (a*x - 1) / x

-- Define the statement for monotonic intervals
theorem monotonic_intervals (a : ℝ) :
  (∀ x > 1, 0 < (f x a)) ∧ (∀ x, 0 < x < 1 → 0 > (f x a)) :=
sorry

-- Define the harmonic function H(n) and its bounds
def H (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), (1 : ℝ) / (k + 1)

-- Prove the inequality for the harmonic sum and natural logarithm
theorem harmonic_log_inequality (n : ℕ) (h : 0 < n) :
  H n - 1 < log (n + 1) ∧ log (n + 1) < 1 + H n :=
sorry

end monotonic_intervals_harmonic_log_inequality_l73_73857


namespace smallest_n_for_candy_l73_73232

theorem smallest_n_for_candy (r g b n : ℕ) (h1 : 10 * r = 18 * g) (h2 : 18 * g = 20 * b) (h3 : 20 * b = 24 * n) : n = 15 :=
by
  sorry

end smallest_n_for_candy_l73_73232


namespace reading_proof_l73_73430

noncomputable def reading (arrow_pos : ℝ) : ℝ :=
  if arrow_pos > 9.75 ∧ arrow_pos < 10.0 then 9.95 else 0

theorem reading_proof
  (arrow_pos : ℝ)
  (h0 : 9.75 < arrow_pos)
  (h1 : arrow_pos < 10.0)
  (possible_readings : List ℝ)
  (h2 : possible_readings = [9.80, 9.90, 9.95, 10.0, 9.85]) :
  reading arrow_pos = 9.95 := by
  -- Proof would go here
  sorry

end reading_proof_l73_73430


namespace negation_proposition_l73_73054

theorem negation_proposition (x : ℝ) : ¬(∀ x, x > 0 → x^2 > 0) ↔ ∃ x, x > 0 ∧ x^2 ≤ 0 :=
by
  sorry

end negation_proposition_l73_73054


namespace largest_number_of_minerals_per_shelf_l73_73582

theorem largest_number_of_minerals_per_shelf (d : ℕ) :
  d ∣ 924 ∧ d ∣ 1386 ∧ d ∣ 462 ↔ d = 462 :=
by
  sorry

end largest_number_of_minerals_per_shelf_l73_73582


namespace proper_divisors_increased_by_one_l73_73218

theorem proper_divisors_increased_by_one
  (n : ℕ)
  (hn1 : 2 ≤ n)
  (exists_m : ∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ d + 1 ≠ m)
  : n = 4 ∨ n = 8 :=
  sorry

end proper_divisors_increased_by_one_l73_73218


namespace smallest_same_terminal_1000_l73_73932

def has_same_terminal_side (theta phi : ℝ) : Prop :=
  ∃ n : ℤ, theta = phi + n * 360

theorem smallest_same_terminal_1000 : ∀ θ : ℝ,
  θ ≥ 0 → θ < 360 → has_same_terminal_side θ 1000 → θ = 280 :=
by
  sorry

end smallest_same_terminal_1000_l73_73932


namespace valid_parameterizations_l73_73301

open Real

def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2 * p.1 - 7

def valid_parametrization (p d : ℝ × ℝ) : Prop :=
  lies_on_line p ∧ is_scalar_multiple d (1, 2)

theorem valid_parameterizations :
  valid_parametrization (4, 1) (-2, -4) ∧ 
  ¬ valid_parametrization (12, 17) (5, 10) ∧ 
  valid_parametrization (3.5, 0) (1, 2) ∧ 
  valid_parametrization (-2, -11) (0.5, 1) ∧ 
  valid_parametrization (0, -7) (10, 20) :=
by {
  sorry
}

end valid_parameterizations_l73_73301


namespace grade_assignment_count_l73_73216

open BigOperators
open Fin

-- Define the set of students and the set of grades.
def students : Fin 10 := Fin 10
def grades : Fin 4 := Fin 4

-- Define the problem statement as a theorem in Lean 4.
theorem grade_assignment_count :
  (Fin 4) ^ (Fin 10) = 1048576 :=
by
  sorry

end grade_assignment_count_l73_73216


namespace initial_elephants_count_l73_73781

def exodus_rate : ℕ := 2880
def exodus_time : ℕ := 4
def entrance_rate : ℕ := 1500
def entrance_time : ℕ := 7
def final_elephants : ℕ := 28980

theorem initial_elephants_count :
  final_elephants - (exodus_rate * exodus_time) + (entrance_rate * entrance_time) = 27960 := by
  sorry

end initial_elephants_count_l73_73781


namespace inequality_ab_ab2_a_l73_73844

theorem inequality_ab_ab2_a (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_ab_ab2_a_l73_73844


namespace find_t_l73_73703

variables (t : ℝ)

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, t)
def a_plus_b : ℝ × ℝ := (2, 1 + t)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_t (t : ℝ) :
  are_parallel (3, t) (2, 1 + t) ↔ t = -3 :=
sorry

end find_t_l73_73703


namespace problem_sum_K_l73_73900

-- Define K(x) as the sum of totient functions
def K (x : ℕ) : ℕ := ∑ i in Finset.range x, Nat.totient i

-- Statement of the problem in lean
theorem problem_sum_K :
  (K 100) + (K (100 / 2)) + (K (100 / 3)) + (K (100 / 4)) + (K (100 / 5)) +
  (K (100 / 6)) + (K (100 / 7)) + (K (100 / 8)) + (K (100 / 9)) + (K (100 / 10)) +
  (K (100 / 11)) + (K (100 / 12)) + (K (100 / 13)) + (K (100 / 14)) + (K (100 / 15)) +
  (K (100 / 16)) + (K (100 / 17)) + (K (100 / 18)) + (K (100 / 19)) + (K (100 / 20)) +
  (K (100 / 21)) + (K (100 / 22)) + (K (100 / 23)) + (K (100 / 24)) + (K (100 / 25)) +
  (K (100 / 26)) + (K (100 / 27)) + (K (100 / 28)) + (K (100 / 29)) + (K (100 / 30)) +
  (K (100 / 31)) + (K (100 / 32)) + (K (100 / 33)) + (K (100 / 34)) + (K (100 / 35)) +
  (K (100 / 36)) + (K (100 / 37)) + (K (100 / 38)) + (K (100 / 39)) + (K (100 / 40)) +
  (K (100 / 41)) + (K (100 / 42)) + (K (100 / 43)) + (K (100 / 44)) + (K (100 / 45)) +
  (K (100 / 46)) + (K (100 / 47)) + (K (100 / 48)) + (K (100 / 49)) + (K (100 / 50)) +
  (K (100 / 51)) + (K (100 / 52)) + (K (100 / 53)) + (K (100 / 54)) + (K (100 / 55)) +
  (K (100 / 56)) + (K (100 / 57)) + (K (100 / 58)) + (K (100 / 59)) + (K (100 / 60)) +
  (K (100 / 61)) + (K (100 / 62)) + (K (100 / 63)) + (K (100 / 64)) + (K (100 / 65)) +
  (K (100 / 66)) + (K (100 / 67)) + (K (100 / 68)) + (K (100 / 69)) + (K (100 / 70)) +
  (K (100 / 71)) + (K (100 / 72)) + (K (100 / 73)) + (K (100 / 74)) + (K (100 / 75)) +
  (K (100 / 76)) + (K (100 / 77)) + (K (100 / 78)) + (K (100 / 79)) + (K (100 / 80)) +
  (K (100 / 81)) + (K (100 / 82)) + (K (100 / 83)) + (K (100 / 84)) + (K (100 / 85)) +
  (K (100 / 86)) + (K (100 / 87)) + (K (100 / 88)) + (K (100 / 89)) + (K (100 / 90)) +
  (K (100 / 91)) + (K (100 / 92)) + (K (100 / 93)) + (K (100 / 94)) + (K (100 / 95)) +
  (K (100 / 96)) + (K (100 / 97)) + (K (100 / 98)) + (K (100 / 99)) + (K 1) = 9801 := sorry

end problem_sum_K_l73_73900


namespace person_speed_l73_73326

theorem person_speed (distance_m : ℝ) (time_min : ℝ) (h₁ : distance_m = 800) (h₂ : time_min = 5) : 
  let distance_km := distance_m / 1000
  let time_hr := time_min / 60
  distance_km / time_hr = 9.6 := 
by
  sorry

end person_speed_l73_73326


namespace negation_of_universal_l73_73923

theorem negation_of_universal:
  ¬(∀ x : ℝ, (0 < x ∧ x < (π / 2)) → x > Real.sin x) ↔
  ∃ x : ℝ, (0 < x ∧ x < (π / 2)) ∧ x ≤ Real.sin x := by
  sorry

end negation_of_universal_l73_73923


namespace sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l73_73734

theorem sum_of_squares_multiple_of_five :
  ( (-1)^2 + 0^2 + 1^2 + 2^2 + 3^2 ) % 5 = 0 :=
by
  sorry

theorem sum_of_consecutive_squares_multiple_of_five 
  (n : ℤ) :
  ((n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l73_73734


namespace find_integer_solutions_l73_73108

theorem find_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + 1 / (x: ℝ)) * (1 + 1 / (y: ℝ)) * (1 + 1 / (z: ℝ)) = 2 ↔ (x = 2 ∧ y = 4 ∧ z = 15) ∨ (x = 2 ∧ y = 5 ∧ z = 9) ∨ (x = 2 ∧ y = 6 ∧ z = 7) ∨ (x = 3 ∧ y = 3 ∧ z = 8) ∨ (x = 3 ∧ y = 4 ∧ z = 5) := sorry

end find_integer_solutions_l73_73108


namespace find_positive_integer_solutions_l73_73228

theorem find_positive_integer_solutions :
  ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ (1 / (a : ℚ)) - (1 / (b : ℚ)) = 1 / 37 ∧ (a, b) = (38, 1332) :=
by
  sorry

end find_positive_integer_solutions_l73_73228


namespace min_trips_calculation_l73_73602

noncomputable def min_trips (total_weight : ℝ) (truck_capacity : ℝ) : ℕ :=
  ⌈total_weight / truck_capacity⌉₊

theorem min_trips_calculation : min_trips 18.5 3.9 = 5 :=
by
  -- Proof goes here
  sorry

end min_trips_calculation_l73_73602


namespace sum_ab_eq_five_l73_73928

theorem sum_ab_eq_five (a b : ℕ) (h : (∃ (ab : ℕ), ab = a * 10 + b ∧ 3 / 13 = ab / 100)) : a + b = 5 :=
sorry

end sum_ab_eq_five_l73_73928


namespace smallest_positive_value_l73_73873

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℝ), k = 2 ∧ k = (↑(a - b) / ↑(a + b) + ↑(a + b) / ↑(a - b)) :=
sorry

end smallest_positive_value_l73_73873


namespace fraction_paint_remaining_l73_73648

theorem fraction_paint_remaining 
  (original_paint : ℝ)
  (h_original : original_paint = 2) 
  (used_first_day : ℝ)
  (h_used_first_day : used_first_day = (1 / 4) * original_paint) 
  (remaining_after_first : ℝ)
  (h_remaining_first : remaining_after_first = original_paint - used_first_day) 
  (used_second_day : ℝ)
  (h_used_second_day : used_second_day = (1 / 3) * remaining_after_first) 
  (remaining_after_second : ℝ)
  (h_remaining_second : remaining_after_second = remaining_after_first - used_second_day) : 
  remaining_after_second / original_paint = 1 / 2 :=
by
  -- Proof goes here.
  sorry

end fraction_paint_remaining_l73_73648


namespace min_value_of_parabola_in_interval_l73_73545

theorem min_value_of_parabola_in_interval :
  ∀ x : ℝ, -10 ≤ x ∧ x ≤ 0 → (x^2 + 12 * x + 35) ≥ -1 := by
  sorry

end min_value_of_parabola_in_interval_l73_73545


namespace cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l73_73260

-- Definitions for geometric objects
def cube : Type := sorry
def regular_tetrahedron : Type := sorry

-- Definitions for axes of symmetry
def axes_of_symmetry (shape : Type) : Nat := sorry

-- Theorem statements
theorem cube_axes_of_symmetry : axes_of_symmetry cube = 13 := 
by 
  sorry

theorem regular_tetrahedron_axes_of_symmetry : axes_of_symmetry regular_tetrahedron = 7 :=
by 
  sorry

end cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l73_73260


namespace avg_mpg_sum_l73_73564

def first_car_gallons : ℕ := 25
def second_car_gallons : ℕ := 35
def total_miles : ℕ := 2275
def first_car_mpg : ℕ := 40

noncomputable def sum_of_avg_mpg_of_two_cars : ℝ := 76.43

theorem avg_mpg_sum :
  let first_car_miles := (first_car_gallons * first_car_mpg : ℕ)
  let second_car_miles := total_miles - first_car_miles
  let second_car_mpg := (second_car_miles : ℝ) / second_car_gallons
  let sum_avg_mpg := (first_car_mpg : ℝ) + second_car_mpg
  sum_avg_mpg = sum_of_avg_mpg_of_two_cars :=
by
  sorry

end avg_mpg_sum_l73_73564


namespace Jake_weight_196_l73_73378

def Jake_and_Sister : Prop :=
  ∃ (J S : ℕ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196)

theorem Jake_weight_196 : Jake_and_Sister :=
by
  sorry

end Jake_weight_196_l73_73378


namespace nine_digit_palindrome_count_l73_73548

-- Defining the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

-- Defining the proposition of the number of 9-digit palindromes
def num_9_digit_palindromes (digs : Multiset ℕ) : ℕ := 36

-- The proof statement
theorem nine_digit_palindrome_count : num_9_digit_palindromes digits = 36 := 
sorry

end nine_digit_palindrome_count_l73_73548


namespace pentagon_vertex_assignment_l73_73418

theorem pentagon_vertex_assignment :
  ∃ (x_A x_B x_C x_D x_E : ℝ),
    x_A + x_B = 1 ∧
    x_B + x_C = 2 ∧
    x_C + x_D = 3 ∧
    x_D + x_E = 4 ∧
    x_E + x_A = 5 ∧
    (x_A, x_B, x_C, x_D, x_E) = (1.5, -0.5, 2.5, 0.5, 3.5) := by
  sorry

end pentagon_vertex_assignment_l73_73418


namespace least_n_factorial_6930_l73_73478

theorem least_n_factorial_6930 (n : ℕ) (h : n! % 6930 = 0) : n ≥ 11 := by
  sorry

end least_n_factorial_6930_l73_73478


namespace largest_initial_number_l73_73395

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l73_73395


namespace perpendicular_vectors_parallel_vectors_l73_73705

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end perpendicular_vectors_parallel_vectors_l73_73705


namespace brokerage_percentage_calculation_l73_73607

theorem brokerage_percentage_calculation
  (face_value : ℝ)
  (discount_percentage : ℝ)
  (cost_price : ℝ)
  (h_face_value : face_value = 100)
  (h_discount_percentage : discount_percentage = 6)
  (h_cost_price : cost_price = 94.2) :
  ((cost_price - (face_value - (discount_percentage / 100 * face_value))) / cost_price * 100) = 0.2124 := 
by
  sorry

end brokerage_percentage_calculation_l73_73607


namespace find_functions_l73_73683

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem find_functions (f : ℝ × ℝ → ℝ) :
  (is_non_decreasing (λ x => f (0, x))) →
  (∀ x y, f (x, y) = f (y, x)) →
  (∀ x y z, (f (x, y) - f (y, z)) * (f (y, z) - f (z, x)) * (f (z, x) - f (x, y)) = 0) →
  (∀ x y a, f (x + a, y + a) = f (x, y) + a) →
  (∃ a : ℝ, (∀ x y, f (x, y) = a + min x y) ∨ (∀ x y, f (x, y) = a + max x y)) :=
  by sorry

end find_functions_l73_73683


namespace solve_for_x_l73_73426

theorem solve_for_x (x : ℝ) (d : ℝ) (h1 : x > 0) (h2 : x^2 = 4 + d) (h3 : 25 = x^2 + d) : x = Real.sqrt 14.5 := 
by 
  sorry

end solve_for_x_l73_73426


namespace factorize_l73_73236

theorem factorize (a : ℝ) : 5*a^3 - 125*a = 5*a*(a + 5)*(a - 5) :=
sorry

end factorize_l73_73236


namespace parabola_x_intercept_y_intercept_point_l73_73731

theorem parabola_x_intercept_y_intercept_point (a b w : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 4) 
  (h3 : ∀ x : ℝ, x = 0 → w = 8): 
  ∃ (w : ℝ), w = 8 := 
by
  sorry

end parabola_x_intercept_y_intercept_point_l73_73731


namespace smallest_m_4_and_n_229_l73_73758

def satisfies_condition (m n : ℕ) : Prop :=
  19 * m + 8 * n = 1908

def is_smallest_m (m n : ℕ) : Prop :=
  ∀ m' n', satisfies_condition m' n' → m' > 0 → n' > 0 → m ≤ m'

theorem smallest_m_4_and_n_229 : ∃ (m n : ℕ), satisfies_condition m n ∧ is_smallest_m m n ∧ m = 4 ∧ n = 229 :=
by
  sorry

end smallest_m_4_and_n_229_l73_73758


namespace roots_quadratic_equation_l73_73540

theorem roots_quadratic_equation (x1 x2 : ℝ) (h1 : x1^2 - x1 - 1 = 0) (h2 : x2^2 - x2 - 1 = 0) :
  (x2 / x1) + (x1 / x2) = -3 :=
by
  sorry

end roots_quadratic_equation_l73_73540


namespace sin_cos_identity_l73_73866

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l73_73866


namespace negation_proposition_l73_73769

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) :=
by
  sorry

end negation_proposition_l73_73769


namespace ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l73_73419

theorem ab_cd_ge_ac_bd_squared (a b c d : ℝ) : ((a^2 + b^2) * (c^2 + d^2)) ≥ (a * c + b * d)^2 := 
by sorry

theorem eq_condition_ad_eq_bc (a b c d : ℝ) (h : a * d = b * c) : ((a^2 + b^2) * (c^2 + d^2)) = (a * c + b * d)^2 := 
by sorry

end ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l73_73419


namespace find_t_l73_73708

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l73_73708


namespace triangle_inequality_l73_73433

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c R : ℝ) : ℝ := a * b * c / (4 * R)
noncomputable def inradius_area (a b c r : ℝ) : ℝ := semiperimeter a b c * r

theorem triangle_inequality (a b c R r : ℝ) (h₁ : a ≤ 1) (h₂ : b ≤ 1) (h₃ : c ≤ 1)
  (h₄ : area a b c R = semiperimeter a b c * r) : 
  semiperimeter a b c * (1 - 2 * R * r) ≥ 1 :=
by 
  -- Proof goes here
  sorry

end triangle_inequality_l73_73433


namespace product_of_repeating_decimal_l73_73665

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l73_73665


namespace sin_cos_identity_l73_73865

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l73_73865


namespace effective_annual_rate_correct_l73_73985

noncomputable def nominal_annual_interest_rate : ℝ := 0.10
noncomputable def compounding_periods_per_year : ℕ := 2
noncomputable def effective_annual_rate : ℝ := (1 + nominal_annual_interest_rate / compounding_periods_per_year) ^ compounding_periods_per_year - 1

theorem effective_annual_rate_correct :
  effective_annual_rate = 0.1025 :=
by
  sorry

end effective_annual_rate_correct_l73_73985


namespace frequency_of_sixth_group_l73_73693

theorem frequency_of_sixth_group :
  ∀ (total_data_points : ℕ)
    (freq1 freq2 freq3 freq4 : ℕ)
    (freq5_ratio : ℝ),
    total_data_points = 40 →
    freq1 = 10 →
    freq2 = 5 →
    freq3 = 7 →
    freq4 = 6 →
    freq5_ratio = 0.10 →
    (total_data_points - (freq1 + freq2 + freq3 + freq4) - (total_data_points * freq5_ratio)) = 8 :=
by
  sorry

end frequency_of_sixth_group_l73_73693


namespace no_integer_roots_l73_73380
open Polynomial

theorem no_integer_roots (p : Polynomial ℤ) (c1 c2 c3 : ℤ) (h1 : p.eval c1 = 1) (h2 : p.eval c2 = 1) (h3 : p.eval c3 = 1) (h_distinct : c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3) : ¬ ∃ a : ℤ, p.eval a = 0 :=
by
  sorry

end no_integer_roots_l73_73380


namespace possible_to_form_square_l73_73325

noncomputable def shape : Type := sorry
noncomputable def is_square (s : shape) : Prop := sorry
noncomputable def divide_into_parts (s : shape) (n : ℕ) : Prop := sorry
noncomputable def all_triangles (s : shape) : Prop := sorry

theorem possible_to_form_square (s : shape) :
  (∃ (parts : ℕ), parts ≤ 4 ∧ divide_into_parts s parts ∧ is_square s) ∧
  (∃ (parts : ℕ), parts ≤ 5 ∧ divide_into_parts s parts ∧ all_triangles s ∧ is_square s) :=
sorry

end possible_to_form_square_l73_73325


namespace find_t_l73_73707

-- Define vectors a and b
def a := (3 : ℝ, 4 : ℝ)
def b := (1 : ℝ, 0 : ℝ)

-- Define the vector c as a function of t
def c (t : ℝ) := (a.1 + t * b.1, a.2 + t * b.2)

-- Statement of the theorem to be proven
theorem find_t (t : ℝ) :
  (a.1 * (a.1 + t * b.1) + a.2 * (a.2 + t * b.2)) = (b.1 * (a.1 + t * b.1) + b.2 * (a.2 + t * b.2)) →
  t = 5 :=
by
  sorry

end find_t_l73_73707


namespace distinct_shell_arrangements_l73_73408

def number_of_distinct_arrangements : Nat := 14.factorial / 14

theorem distinct_shell_arrangements : number_of_distinct_arrangements = 6227020800 := by
  have h1 : 14.factorial = 87178291200 := by norm_num
  have h2 : 87178291200 / 14 = 6227020800 := by norm_num
  rw [number_of_distinct_arrangements, h1, h2]
  norm_num

end distinct_shell_arrangements_l73_73408


namespace largest_initial_number_l73_73401

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l73_73401


namespace butternut_wood_figurines_l73_73653

theorem butternut_wood_figurines (B : ℕ) (basswood_blocks : ℕ) (aspen_blocks : ℕ) (butternut_blocks : ℕ) 
  (basswood_figurines_per_block : ℕ) (aspen_figurines_per_block : ℕ) (total_figurines : ℕ) 
  (h_basswood_blocks : basswood_blocks = 15)
  (h_aspen_blocks : aspen_blocks = 20)
  (h_butternut_blocks : butternut_blocks = 20)
  (h_basswood_figurines_per_block : basswood_figurines_per_block = 3)
  (h_aspen_figurines_per_block : aspen_figurines_per_block = 2 * basswood_figurines_per_block)
  (h_total_figurines : total_figurines = 245) :
  B = 4 :=
by
  -- Definitions based on the given conditions
  let basswood_figurines := basswood_blocks * basswood_figurines_per_block
  let aspen_figurines := aspen_blocks * aspen_figurines_per_block
  let figurines_from_butternut := total_figurines - basswood_figurines - aspen_figurines
  -- Calculate the number of figurines per block of butternut wood
  let butternut_figurines_per_block := figurines_from_butternut / butternut_blocks
  -- The objective is to prove that the number of figurines per block of butternut wood is 4
  exact sorry

end butternut_wood_figurines_l73_73653


namespace calculate_present_worth_l73_73627

variable (BG : ℝ) (r : ℝ) (t : ℝ)

theorem calculate_present_worth (hBG : BG = 24) (hr : r = 0.10) (ht : t = 2) : 
  ∃ PW : ℝ, PW = 120 := 
by
  sorry

end calculate_present_worth_l73_73627


namespace range_of_a_l73_73874

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 5^x = a + 3) → a > -3 :=
by
  sorry

end range_of_a_l73_73874


namespace lcm_of_9_12_15_is_180_l73_73962

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l73_73962


namespace seating_arrangements_l73_73726

theorem seating_arrangements {n k : ℕ} (h1 : n = 8) (h2 : k = 6) :
  ∃ c : ℕ, c = (n - 1) * Nat.factorial k ∧ c = 20160 :=
by
  sorry

end seating_arrangements_l73_73726


namespace runner_time_difference_l73_73092

theorem runner_time_difference 
  (v : ℝ)  -- runner's initial speed (miles per hour)
  (H1 : 0 < v)  -- speed is positive
  (d : ℝ)  -- total distance
  (H2 : d = 40)  -- total distance condition
  (t2 : ℝ)  -- time taken for the second half
  (H3 : t2 = 10)  -- second half time condition
  (H4 : v ≠ 0)  -- initial speed cannot be zero
  (H5: 20 = 10 * (v / 2))  -- equation derived from the second half conditions
  : (t2 - (20 / v)) = 5 := 
by
  sorry

end runner_time_difference_l73_73092


namespace speed_ratio_l73_73982

theorem speed_ratio (L v_a v_b : ℝ) (h1 : v_a = c * v_b) (h2 : (L / v_a) = (0.8 * L / v_b)) :
  v_a / v_b = 5 / 4 :=
by
  sorry

end speed_ratio_l73_73982


namespace milton_books_l73_73748

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l73_73748


namespace sum_geometric_sequence_l73_73934

variable {α : Type*} [LinearOrderedField α]

theorem sum_geometric_sequence {S : ℕ → α} {n : ℕ} (h1 : S n = 3) (h2 : S (3 * n) = 21) :
    S (2 * n) = 9 := 
sorry

end sum_geometric_sequence_l73_73934


namespace tysons_speed_in_ocean_l73_73605

theorem tysons_speed_in_ocean
  (speed_lake : ℕ) (half_races_lake : ℕ) (total_races : ℕ) (race_distance : ℕ) (total_time : ℕ)
  (speed_lake_val : speed_lake = 3)
  (half_races_lake_val : half_races_lake = 5)
  (total_races_val : total_races = 10)
  (race_distance_val : race_distance = 3)
  (total_time_val : total_time = 11) :
  ∃ (speed_ocean : ℚ), speed_ocean = 2.5 := 
by
  sorry

end tysons_speed_in_ocean_l73_73605


namespace problem_statement_l73_73578

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a + 1 / a) ^ 2 + (b + 1 / b) ^ 2 ≥ 25 / 2 := 
by
  sorry

end problem_statement_l73_73578


namespace num_photos_to_include_l73_73996

-- Define the conditions
def num_preselected_photos : ℕ := 7
def total_choices : ℕ := 56

-- Define the statement to prove
theorem num_photos_to_include : total_choices / num_preselected_photos = 8 :=
by sorry

end num_photos_to_include_l73_73996


namespace solution_set_inequality_l73_73536

noncomputable def solution_set (x : ℝ) : Prop :=
  (2 * x - 1) / (x + 2) > 1

theorem solution_set_inequality :
  { x : ℝ | solution_set x } = { x : ℝ | x < -2 ∨ x > 3 } := by
  sorry

end solution_set_inequality_l73_73536


namespace interior_angle_of_regular_pentagon_is_108_l73_73608

-- Define the sum of angles in a triangle
def sum_of_triangle_angles : ℕ := 180

-- Define the number of triangles in a convex pentagon
def num_of_triangles_in_pentagon : ℕ := 3

-- Define the total number of interior angles in a pentagon
def num_of_angles_in_pentagon : ℕ := 5

-- Define the total sum of the interior angles of a pentagon
def sum_of_pentagon_interior_angles : ℕ := num_of_triangles_in_pentagon * sum_of_triangle_angles

-- Define the degree measure of an interior angle of a regular pentagon
def interior_angle_of_regular_pentagon : ℕ := sum_of_pentagon_interior_angles / num_of_angles_in_pentagon

theorem interior_angle_of_regular_pentagon_is_108 :
  interior_angle_of_regular_pentagon = 108 :=
by
  -- Proof will be filled in here
  sorry

end interior_angle_of_regular_pentagon_is_108_l73_73608


namespace combination_addition_l73_73824

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_addition :
  combination 13 11 + 3 = 81 :=
by
  sorry

end combination_addition_l73_73824


namespace oranges_per_group_l73_73776

theorem oranges_per_group (total_oranges groups : ℕ) (h1 : total_oranges = 384) (h2 : groups = 16) :
  total_oranges / groups = 24 := by
  sorry

end oranges_per_group_l73_73776


namespace n_is_perfect_square_l73_73895

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k ^ 2

theorem n_is_perfect_square (a b c d : ℤ) (h : a + b + c + d = 0) : 
  is_perfect_square ((ab - cd) * (bc - ad) * (ca - bd)) := 
  sorry

end n_is_perfect_square_l73_73895


namespace initial_roses_l73_73777

theorem initial_roses (R : ℕ) (h : R + 16 = 23) : R = 7 :=
sorry

end initial_roses_l73_73777


namespace symmetric_origin_coordinates_l73_73730

-- Given the coordinates (m, n) of point P
variables (m n : ℝ)
-- Define point P
def P := (m, n)

-- Define point P' which is symmetric to P with respect to the origin O
def P'_symmetric_origin : ℝ × ℝ := (-m, -n)

-- Prove that the coordinates of P' are (-m, -n)
theorem symmetric_origin_coordinates :
  P'_symmetric_origin m n = (-m, -n) :=
by
  -- Proof content goes here but we're skipping it with sorry
  sorry

end symmetric_origin_coordinates_l73_73730


namespace star_value_l73_73689

def star (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem star_value : star 3 2 = 125 :=
by
  sorry

end star_value_l73_73689


namespace shortest_distance_l73_73798

-- The initial position of the cowboy.
def initial_position : ℝ × ℝ := (-2, -6)

-- The position of the cabin relative to the cowboy's initial position.
def cabin_position : ℝ × ℝ := (10, -15)

-- The equation of the stream flowing due northeast.
def stream_equation : ℝ → ℝ := id  -- y = x

-- Function to calculate the distance between two points (x1, y1) and (x2, y2).
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Calculate the reflection point of C over y = x.
def reflection_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Main proof statement: shortest distance the cowboy can travel.
theorem shortest_distance : distance initial_position (reflection_point initial_position) +
                            distance (reflection_point initial_position) cabin_position = 8 +
                            Real.sqrt 545 :=
by
  sorry

end shortest_distance_l73_73798


namespace acute_triangle_l73_73195

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ 0 < A ∧ 0 < B ∧ 0 < C

def each_angle_less_than_sum_of_others (A B C : ℝ) : Prop :=
  A < B + C ∧ B < A + C ∧ C < A + B

theorem acute_triangle (A B C : ℝ) 
  (h1 : is_triangle A B C) 
  (h2 : each_angle_less_than_sum_of_others A B C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := 
sorry

end acute_triangle_l73_73195


namespace product_of_repeating_decimal_l73_73668

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l73_73668


namespace smallest_b_factor_2020_l73_73111

theorem smallest_b_factor_2020 :
  ∃ b : ℕ, b > 0 ∧
  (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ b = r + s) ∧
  (∀ c : ℕ, c > 0 → (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ c = r + s) → b ≤ c) ∧
  b = 121 :=
sorry

end smallest_b_factor_2020_l73_73111


namespace smallest_multiple_17_mod_71_l73_73611

theorem smallest_multiple_17_mod_71 (a : ℕ) : 
  (17 * a ≡ 3 [MOD 71]) → 17 * a = 1139 :=
by
  intro h
  sorry

end smallest_multiple_17_mod_71_l73_73611


namespace perimeter_of_polygon_is_15_l73_73055

-- Definitions for the problem conditions
def side_length_of_square : ℕ := 5
def fraction_of_square_occupied (n : ℕ) : ℚ := 3 / 4

-- Problem statement: Prove that the perimeter of the polygon is 15 units
theorem perimeter_of_polygon_is_15 :
  4 * side_length_of_square * (fraction_of_square_occupied side_length_of_square) = 15 := 
by
  sorry

end perimeter_of_polygon_is_15_l73_73055


namespace sample_size_proof_l73_73633

-- Define the quantities produced by each workshop
def units_A : ℕ := 120
def units_B : ℕ := 80
def units_C : ℕ := 60

-- Define the number of units sampled from Workshop C
def samples_C : ℕ := 3

-- Calculate the total sample size n
def total_sample_size : ℕ :=
  let sampling_fraction := samples_C / units_C
  let samples_A := sampling_fraction * units_A
  let samples_B := sampling_fraction * units_B
  samples_A + samples_B + samples_C

-- The theorem we want to prove
theorem sample_size_proof : total_sample_size = 13 :=
by sorry

end sample_size_proof_l73_73633


namespace interior_angle_solution_l73_73189

noncomputable def interior_angle_of_inscribed_triangle (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) : ℝ :=
  (1 / 2) * (x + 80)

theorem interior_angle_solution (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) :
  interior_angle_of_inscribed_triangle x h = 64 :=
sorry

end interior_angle_solution_l73_73189


namespace negation_proposition_l73_73925

theorem negation_proposition :
  ∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n :=
sorry

end negation_proposition_l73_73925


namespace perpendicular_bisects_third_side_l73_73596

open EuclideanGeometry

variable {ABC : Triangle}
variable {A B C : Point}
variable {D E M : Point}

-- Conditions
variable (hAD : altitude A BC)
variable (hBE : altitude B AC)
variable (hD : foot_of_altitude A BC D)
variable (hE : foot_of_altitude B AC E)
variable (hM : midpoint D E M)

-- Question: Prove that the perpendicular drawn from M bisects the side AB
theorem perpendicular_bisects_third_side :
  ∀ {A B C : Point} {D E M : Point},
  altitude A BC →
  altitude B AC →
  midpoint D E M →
  is_midpoint_of_perpendicular_intersection M A B :=
by
  sorry

end perpendicular_bisects_third_side_l73_73596


namespace reciprocal_of_neg_two_l73_73059

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end reciprocal_of_neg_two_l73_73059


namespace largest_initial_number_l73_73393

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l73_73393


namespace problem_statement_l73_73579

theorem problem_statement (g : ℝ → ℝ) :
  (∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - x + 2) →
  (∃ m t : ℝ, m = 1 ∧ t = 3 ∧ m * t = 3) :=
sorry

end problem_statement_l73_73579


namespace dan_money_left_l73_73339

theorem dan_money_left (initial_amount spent_amount remaining_amount : ℤ) (h1 : initial_amount = 300) (h2 : spent_amount = 100) : remaining_amount = 200 :=
by 
  sorry

end dan_money_left_l73_73339


namespace maximize_x3y4_l73_73412

noncomputable def maximize_expr (x y : ℝ) : ℝ :=
x^3 * y^4

theorem maximize_x3y4 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 50 ∧ maximize_expr x y = maximize_expr 30 20 :=
by
  sorry

end maximize_x3y4_l73_73412


namespace proof_problem_l73_73262

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end proof_problem_l73_73262


namespace factor_by_resultant_is_three_l73_73643

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end factor_by_resultant_is_three_l73_73643


namespace variable_v_value_l73_73263

theorem variable_v_value (w x v : ℝ) (h1 : 2 / w + 2 / x = 2 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) :
  v = 0.25 :=
sorry

end variable_v_value_l73_73263


namespace problem1_problem2_l73_73659

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end problem1_problem2_l73_73659


namespace sum_of_exponents_l73_73562

-- Given product of integers from 1 to 15
def y := Nat.factorial 15

-- Prime exponent variables in the factorization of y
variables (i j k m n p q : ℕ)

-- Conditions
axiom h1 : y = 2^i * 3^j * 5^k * 7^m * 11^n * 13^p * 17^q 

-- Prove that the sum of the exponents equals 24
theorem sum_of_exponents :
  i + j + k + m + n + p + q = 24 := 
sorry

end sum_of_exponents_l73_73562


namespace max_total_profit_max_avg_annual_profit_l73_73652

noncomputable def total_profit (x : ℕ) : ℝ := - (x : ℝ)^2 + 18 * x - 36
noncomputable def avg_annual_profit (x : ℕ) : ℝ := (total_profit x) / x

theorem max_total_profit : ∃ x : ℕ, total_profit x = 45 ∧ x = 9 :=
  by sorry

theorem max_avg_annual_profit : ∃ x : ℕ, avg_annual_profit x = 6 ∧ x = 6 :=
  by sorry

end max_total_profit_max_avg_annual_profit_l73_73652


namespace interval_of_decrease_l73_73251

theorem interval_of_decrease (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x) :
  ∀ x0 : ℝ, ∀ x1 : ℝ, x0 ≥ 3 → x0 ≤ x1 → f (x1 - 3) ≤ f (x0 - 3) := sorry

end interval_of_decrease_l73_73251


namespace largest_initial_number_l73_73394

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l73_73394


namespace percentage_increase_in_side_of_square_l73_73931

theorem percentage_increase_in_side_of_square (p : ℝ) : 
  (1 + p / 100) ^ 2 = 1.3225 → 
  p = 15 :=
by
  sorry

end percentage_increase_in_side_of_square_l73_73931


namespace expected_number_of_own_hats_l73_73794

-- Define the number of people
def num_people : ℕ := 2015

-- Define the expectation based on the problem description
noncomputable def expected_hats (n : ℕ) : ℝ := 1

-- The main theorem representing the problem statement
theorem expected_number_of_own_hats : expected_hats num_people = 1 := sorry

end expected_number_of_own_hats_l73_73794


namespace percentage_of_alcohol_in_original_solution_l73_73082

noncomputable def alcohol_percentage_in_original_solution (P: ℝ) (V_original: ℝ) (V_water: ℝ) (percentage_new: ℝ): ℝ :=
  (P * V_original) / (V_original + V_water) * 100

theorem percentage_of_alcohol_in_original_solution : 
  ∀ (P: ℝ) (V_original : ℝ) (V_water : ℝ) (percentage_new : ℝ), 
  V_original = 3 → 
  V_water = 1 → 
  percentage_new = 24.75 →
  alcohol_percentage_in_original_solution P V_original V_water percentage_new = 33 := 
by
  sorry

end percentage_of_alcohol_in_original_solution_l73_73082


namespace Katie_has_more_games_than_friends_l73_73027

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem Katie_has_more_games_than_friends :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end Katie_has_more_games_than_friends_l73_73027


namespace total_theme_parks_l73_73277

theorem total_theme_parks 
  (J V M N : ℕ) 
  (hJ : J = 35)
  (hV : V = J + 40)
  (hM : M = J + 60)
  (hN : N = 2 * M) 
  : J + V + M + N = 395 :=
sorry

end total_theme_parks_l73_73277


namespace Willie_dollars_exchange_l73_73621

theorem Willie_dollars_exchange:
  ∀ (euros : ℝ) (official_rate : ℝ) (airport_rate : ℝ),
  euros = 70 →
  official_rate = 5 →
  airport_rate = 5 / 7 →
  euros / official_rate * airport_rate = 10 :=
by
  intros euros official_rate airport_rate
  intros h_euros h_official_rate h_airport_rate
  rw [h_euros, h_official_rate, h_airport_rate]
  norm_num
  sorry

end Willie_dollars_exchange_l73_73621


namespace net_pay_rate_is_26_dollars_per_hour_l73_73084

-- Defining the conditions
noncomputable def total_distance (time_hours : ℝ) (speed_mph : ℝ) : ℝ :=
  time_hours * speed_mph

noncomputable def adjusted_fuel_efficiency (original_efficiency : ℝ) (decrease_percentage : ℝ) : ℝ :=
  original_efficiency * (1 - decrease_percentage)

noncomputable def gasoline_used (distance : ℝ) (efficiency : ℝ) : ℝ :=
  distance / efficiency

noncomputable def earnings (rate_per_mile : ℝ) (distance : ℝ) : ℝ :=
  rate_per_mile * distance

noncomputable def updated_gasoline_price (original_price : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_price * (1 + increase_percentage)

noncomputable def total_cost_gasoline (gasoline_price : ℝ) (gasoline_used : ℝ) : ℝ :=
  gasoline_price * gasoline_used

noncomputable def net_earnings (earnings : ℝ) (cost : ℝ) : ℝ :=
  earnings - cost

noncomputable def net_rate_of_pay (net_earnings : ℝ) (time_hours : ℝ) : ℝ :=
  net_earnings / time_hours

-- Given constants
def time_hours : ℝ := 3
def speed_mph : ℝ := 50
def original_efficiency : ℝ := 30
def decrease_percentage : ℝ := 0.10
def rate_per_mile : ℝ := 0.60
def original_gasoline_price : ℝ := 2.00
def increase_percentage : ℝ := 0.20

-- Proof problem statement
theorem net_pay_rate_is_26_dollars_per_hour :
  net_rate_of_pay 
    (net_earnings
       (earnings rate_per_mile (total_distance time_hours speed_mph))
       (total_cost_gasoline
          (updated_gasoline_price original_gasoline_price increase_percentage)
          (gasoline_used
            (total_distance time_hours speed_mph)
            (adjusted_fuel_efficiency original_efficiency decrease_percentage))))
    time_hours = 26 := 
  sorry

end net_pay_rate_is_26_dollars_per_hour_l73_73084


namespace find_functions_l73_73532

theorem find_functions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2) :
  (∀ x, f x = (x^2) / 2002) ∨ (∀ x, f x = (x^2) / 2002 + 2 * x + 2002) :=
sorry

end find_functions_l73_73532


namespace value_preserving_interval_of_g_l73_73851

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  x + m - Real.log x

theorem value_preserving_interval_of_g
  (m : ℝ)
  (h_increasing : ∀ x, x ∈ Set.Ici 2 → 1 - 1 / x > 0)
  (h_range : ∀ y, y ∈ Set.Ici 2): 
  (2 + m - Real.log 2 = 2) → 
  m = Real.log 2 :=
by 
  sorry

end value_preserving_interval_of_g_l73_73851


namespace milton_zoology_books_l73_73751

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l73_73751


namespace bus_stops_bound_l73_73318

-- Definitions based on conditions
variables (n x : ℕ)

-- Condition 1: Any bus stop is serviced by at most 3 bus lines
def at_most_three_bus_lines (bus_stops : ℕ) : Prop :=
  ∀ (stop : ℕ), stop < bus_stops → stop ≤ 3

-- Condition 2: Any bus line has at least two stops
def at_least_two_stops (bus_lines : ℕ) : Prop :=
  ∀ (line : ℕ), line < bus_lines → line ≥ 2

-- Condition 3: For any two specific bus lines, there is a third line such that passengers can transfer
def transfer_line_exists (bus_lines : ℕ) : Prop :=
  ∀ (line1 line2 : ℕ), line1 < bus_lines ∧ line2 < bus_lines →
  ∃ (line3 : ℕ), line3 < bus_lines

-- Theorem statement: The number of bus stops is at least 5/6 (n-5)
theorem bus_stops_bound (h1 : at_most_three_bus_lines x) (h2 : at_least_two_stops n)
  (h3 : transfer_line_exists n) : x ≥ (5 * (n - 5)) / 6 :=
sorry

end bus_stops_bound_l73_73318


namespace anton_food_cost_l73_73410

def food_cost_julie : ℝ := 10
def food_cost_letitia : ℝ := 20
def tip_per_person : ℝ := 4
def num_people : ℕ := 3
def tip_percentage : ℝ := 0.20

theorem anton_food_cost (A : ℝ) :
  tip_percentage * (food_cost_julie + food_cost_letitia + A) = tip_per_person * num_people →
  A = 30 :=
by
  intro h
  sorry

end anton_food_cost_l73_73410


namespace hexahedron_volume_l73_73275

open Real

noncomputable def volume_of_hexahedron (AB A1B1 AA1 : ℝ) : ℝ :=
  let S_base := (3 * sqrt 3 / 2) * AB^2
  let S_top := (3 * sqrt 3 / 2) * A1B1^2
  let h := AA1
  (1 / 3) * h * (S_base + sqrt (S_base * S_top) + S_top)

theorem hexahedron_volume : volume_of_hexahedron 2 3 (sqrt 10) = 57 * sqrt 3 / 2 := by
  sorry

end hexahedron_volume_l73_73275


namespace unique_pair_prime_m_positive_l73_73281

theorem unique_pair_prime_m_positive (p m : ℕ) (hp : Nat.Prime p) (hm : 0 < m) :
  p * (p + m) + p = (m + 1) ^ 3 → (p = 2 ∧ m = 1) :=
by
  sorry

end unique_pair_prime_m_positive_l73_73281


namespace sin_cos_product_l73_73867

-- Define the problem's main claim
theorem sin_cos_product (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 :=
by
  have h1 : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := Real.sin_square_add_cos_square x
  sorry

end sin_cos_product_l73_73867


namespace valid_arrangement_after_removal_l73_73201

theorem valid_arrangement_after_removal (n : ℕ) (m : ℕ → ℕ) :
  (∀ i j, i ≠ j → m i ≠ m j → ¬ (i < n ∧ j < n))
  → (∀ i, i < n → m i ≥ m (i + 1))
  → ∃ (m' : ℕ → ℕ), (∀ i, i < n.pred → m' i = m (i + 1) - 1 ∨ m' i = m (i + 1))
    ∧ (∀ i, m' i ≥ m' (i + 1))
    ∧ (∀ i j, i ≠ j → i < n.pred → j < n.pred → ¬ (m' i = m' j ∧ m' i = m (i + 1))) := sorry

end valid_arrangement_after_removal_l73_73201


namespace two_students_one_common_material_l73_73453

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l73_73453


namespace choose_rectangles_l73_73651

theorem choose_rectangles (n : ℕ) (hn : n ≥ 2) :
  ∃ (chosen_rectangles : Finset (ℕ × ℕ)), 
    (chosen_rectangles.card = 2 * n ∧
     ∀ (r1 r2 : ℕ × ℕ), r1 ∈ chosen_rectangles → r2 ∈ chosen_rectangles →
      (r1.fst ≤ r2.fst ∧ r1.snd ≤ r2.snd) ∨ 
      (r2.fst ≤ r1.fst ∧ r2.snd ≤ r1.snd) ∨ 
      (r1.fst ≤ r2.snd ∧ r1.snd ≤ r2.fst) ∨ 
      (r2.fst ≤ r1.snd ∧ r2.snd <= r1.fst)) :=
sorry

end choose_rectangles_l73_73651


namespace find_value_given_conditions_l73_73372

def equation_result (x y k : ℕ) : Prop := x ^ y + y ^ x = k

theorem find_value_given_conditions (y : ℕ) (k : ℕ) : 
  equation_result 2407 y k := 
by 
  sorry

end find_value_given_conditions_l73_73372


namespace lcm_9_12_15_l73_73970

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l73_73970


namespace students_in_class_l73_73440

theorem students_in_class (S : ℕ) 
  (h1 : chess_students = S / 3)
  (h2 : tournament_students = chess_students / 2)
  (h3 : tournament_students = 4) : 
  S = 24 :=
by
  sorry

end students_in_class_l73_73440


namespace number_of_ways_l73_73461

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l73_73461


namespace ratio_15_to_1_l73_73487

theorem ratio_15_to_1 (x : ℕ) (h : 15 / 1 = x / 10) : x = 150 := 
by sorry

end ratio_15_to_1_l73_73487


namespace total_Pokemon_cards_l73_73023

def j : Nat := 6
def o : Nat := j + 2
def r : Nat := 3 * o
def t : Nat := j + o + r

theorem total_Pokemon_cards : t = 38 := by 
  sorry

end total_Pokemon_cards_l73_73023


namespace triangle_PQR_min_perimeter_l73_73946

theorem triangle_PQR_min_perimeter (PQ PR QR : ℕ) (QJ : ℕ) 
  (hPQ_PR : PQ = PR) (hQJ_10 : QJ = 10) (h_pos_QR : 0 < QR) :
  QR * 2 + PQ * 2 = 96 :=
  sorry

end triangle_PQR_min_perimeter_l73_73946


namespace average_of_first_12_results_l73_73913

theorem average_of_first_12_results
  (average_25_results : ℝ)
  (average_last_12_results : ℝ)
  (result_13th : ℝ)
  (total_results : ℕ)
  (num_first_12 : ℕ)
  (num_last_12 : ℕ)
  (total_sum : ℝ)
  (sum_first_12 : ℝ)
  (sum_last_12 : ℝ)
  (A : ℝ)
  (h1 : average_25_results = 24)
  (h2 : average_last_12_results = 17)
  (h3 : result_13th = 228)
  (h4 : total_results = 25)
  (h5 : num_first_12 = 12)
  (h6 : num_last_12 = 12)
  (h7 : total_sum = average_25_results * total_results)
  (h8 : sum_last_12 = average_last_12_results * num_last_12)
  (h9 : total_sum = sum_first_12 + result_13th + sum_last_12)
  (h10 : sum_first_12 = A * num_first_12) :
  A = 14 :=
by
  sorry

end average_of_first_12_results_l73_73913


namespace num_ways_choose_materials_l73_73455

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l73_73455


namespace xyz_sum_l73_73265

theorem xyz_sum (x y z : ℕ) (h1 : xyz = 240) (h2 : xy + z = 46) (h3 : x + yz = 64) : x + y + z = 20 :=
sorry

end xyz_sum_l73_73265


namespace girls_attending_event_l73_73235

theorem girls_attending_event (total_students girls_attending boys_attending : ℕ) 
    (h1 : total_students = 1500) 
    (h2 : girls_attending = 3 / 5 * girls) 
    (h3 : boys_attending = 2 / 3 * (total_students - girls)) 
    (h4 : girls_attending + boys_attending = 900) : 
    girls_attending = 900 := 
by 
    sorry

end girls_attending_event_l73_73235


namespace problem1_problem2_l73_73846

-- Definitions of the polynomials A and B
def A (x y : ℝ) := x^2 + x * y + 3 * y
def B (x y : ℝ) := x^2 - x * y

-- Problem 1 Statement: 
theorem problem1 (x y : ℝ) (h : (x - 2)^2 + |y + 5| = 0) : 2 * (A x y) - (B x y) = -56 := by
  sorry

-- Problem 2 Statement:
theorem problem2 (x : ℝ) (h : ∀ y, 2 * (A x y) - (B x y) = 0) : x = -2 := by
  sorry

end problem1_problem2_l73_73846


namespace orange_pyramid_total_l73_73634

theorem orange_pyramid_total :
  let base_length := 7
  let base_width := 9
  -- layer 1 -> dimensions (7, 9)
  -- layer 2 -> dimensions (6, 8)
  -- layer 3 -> dimensions (5, 6)
  -- layer 4 -> dimensions (4, 5)
  -- layer 5 -> dimensions (3, 3)
  -- layer 6 -> dimensions (2, 2)
  -- layer 7 -> dimensions (1, 1)
  (base_length * base_width) + ((base_length - 1) * (base_width - 1))
  + ((base_length - 2) * (base_width - 3)) + ((base_length - 3) * (base_width - 4))
  + ((base_length - 4) * (base_width - 6)) + ((base_length - 5) * (base_width - 7))
  + ((base_length - 6) * (base_width - 8)) = 175 := sorry

end orange_pyramid_total_l73_73634


namespace calvin_weeks_buying_chips_l73_73661

variable (daily_spending : ℝ := 0.50)
variable (days_per_week : ℝ := 5)
variable (total_spending : ℝ := 10)
variable (spending_per_week := daily_spending * days_per_week)

theorem calvin_weeks_buying_chips :
  total_spending / spending_per_week = 4 := by
  sorry

end calvin_weeks_buying_chips_l73_73661


namespace triangle_properties_l73_73930

-- Define the sides of the triangle
def side1 : ℕ := 8
def side2 : ℕ := 15
def hypotenuse : ℕ := 17

-- Using the Pythagorean theorem to assert it is a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Calculate the area of the right triangle
def triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Calculate the perimeter of the triangle
def triangle_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_properties :
  let a := side1
  let b := side2
  let c := hypotenuse
  is_right_triangle a b c →
  triangle_area a b = 60 ∧ triangle_perimeter a b c = 40 := by
  intros h
  sorry

end triangle_properties_l73_73930


namespace problem1_problem2_l73_73202

-- Problem (1)
variables {p q : ℝ}

theorem problem1 (hpq : p^3 + q^3 = 2) : p + q ≤ 2 := sorry

-- Problem (2)
variables {a b : ℝ}

theorem problem2 (hab : |a| + |b| < 1) : ∀ x : ℝ, (x^2 + a * x + b = 0) → |x| < 1 := sorry

end problem1_problem2_l73_73202


namespace proof_problem_l73_73005

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end proof_problem_l73_73005


namespace yogurt_strawberry_probability_l73_73527

theorem yogurt_strawberry_probability :
  let p_first_days := 1/2
      p_last_days := 3/4 
      case_1a_prob := (Real.binom 3 2) * (Real.binom 3 2) * (p_first_days^2) * (p_last_days^2)
      case_1b_prob := (Real.binom 3 3) * (Real.binom 3 1) * (p_first_days^3) * (p_last_days)
      total_prob_per_comb := case_1a_prob + case_1b_prob
      num_ways := Real.binom 6 4 
      final_prob := num_ways * total_prob_per_comb
  in final_prob = 1485 / 64 := by
  sorry

end yogurt_strawberry_probability_l73_73527


namespace part_a_l73_73488

theorem part_a (α β : ℝ) (h₁ : α = 1.0000000004) (h₂ : β = 1.00000000002) (h₃ : α > β) :
  2.00000000002 / (β * β + 2.00000000002) > 2.00000000004 / α := 
sorry

end part_a_l73_73488


namespace carbonate_weight_l73_73686

namespace MolecularWeight

def molecular_weight_Al2_CO3_3 : ℝ := 234
def molecular_weight_Al : ℝ := 26.98
def num_Al_atoms : ℕ := 2

theorem carbonate_weight :
  molecular_weight_Al2_CO3_3 - (num_Al_atoms * molecular_weight_Al) = 180.04 :=
sorry

end MolecularWeight

end carbonate_weight_l73_73686


namespace total_coffee_cost_l73_73039

def vacation_days : ℕ := 40
def daily_coffee : ℕ := 3
def pods_per_box : ℕ := 30
def box_cost : ℕ := 8

theorem total_coffee_cost : vacation_days * daily_coffee / pods_per_box * box_cost = 32 := by
  -- proof goes here
  sorry

end total_coffee_cost_l73_73039


namespace g_25_eq_zero_l73_73592

noncomputable def g : ℝ → ℝ := sorry

axiom g_def (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x^2 * g y - y^2 * g x = g (x^2 / y^2)

theorem g_25_eq_zero : g 25 = 0 := by
  sorry

end g_25_eq_zero_l73_73592


namespace matrix_A_squared_unique_l73_73744

open Matrix

variables {R : Type*} [CommRing R] {A : Matrix (Fin 2) (Fin 2) R}

def matrix_A_condition (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop := A ^ 4 = 0

theorem matrix_A_squared_unique (A : Matrix (Fin 2) (Fin 2) ℝ) (h : matrix_A_condition A) : A ^ 2 = 0 :=
sorry

end matrix_A_squared_unique_l73_73744


namespace total_shingles_needed_l73_73920

-- Defining the dimensions of the house and the porch
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_length : ℝ := 6
def porch_width : ℝ := 4.5

-- The goal is to prove that the total area of the shingles needed is 232 square feet
theorem total_shingles_needed :
  (house_length * house_width) + (porch_length * porch_width) = 232 := by
  sorry

end total_shingles_needed_l73_73920


namespace remaining_days_temperature_l73_73915

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end remaining_days_temperature_l73_73915


namespace cost_to_feed_turtles_l73_73912

-- Define the conditions
def ounces_per_half_pound : ℝ := 1 
def total_weight_turtles : ℝ := 30
def food_per_half_pound : ℝ := 0.5
def ounces_per_jar : ℝ := 15
def cost_per_jar : ℝ := 2

-- Define the statement to prove
theorem cost_to_feed_turtles : (total_weight_turtles / food_per_half_pound) / ounces_per_jar * cost_per_jar = 8 := by
  sorry

end cost_to_feed_turtles_l73_73912


namespace largest_initial_number_l73_73398

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l73_73398


namespace highest_score_runs_l73_73496

theorem highest_score_runs 
  (avg : ℕ) (innings : ℕ) (total_runs : ℕ) (H L : ℕ)
  (diff_HL : ℕ) (excl_avg : ℕ) (excl_innings : ℕ) (excl_total_runs : ℕ) :
  avg = 60 → innings = 46 → total_runs = avg * innings →
  diff_HL = 180 → excl_avg = 58 → excl_innings = 44 → 
  excl_total_runs = excl_avg * excl_innings →
  H - L = diff_HL →
  total_runs = excl_total_runs + H + L →
  H = 194 :=
by
  intros h_avg h_innings h_total_runs h_diff_HL h_excl_avg h_excl_innings h_excl_total_runs h_H_minus_L h_total_eq
  sorry

end highest_score_runs_l73_73496


namespace coefficient_x3_in_expansion_l73_73891

theorem coefficient_x3_in_expansion :
  let coeff (n k : ℕ) := (Nat.choose n k) * (-1)^k in
  coeff 5 3 + coeff 6 3 + coeff 7 3 + coeff 8 3 = -121 :=
by
  sorry

end coefficient_x3_in_expansion_l73_73891


namespace math_problem_l73_73159

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) →
  (1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥  3 / 2)

theorem math_problem (a b c : ℝ) :
  proof_problem a b c :=
by
  sorry

end math_problem_l73_73159


namespace number_of_ways_to_choose_materials_l73_73472

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l73_73472


namespace find_point_B_find_line_BC_l73_73152

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the equation of the median on side AB
def median_AB (x y : ℝ) : Prop := x + 3 * y = 6

-- Define the equation of the internal angle bisector of ∠ABC
def bisector_BC (x y : ℝ) : Prop := x - y = -1

-- Prove the coordinates of point B
theorem find_point_B :
  (a b : ℝ) →
  (median_AB ((a + 2) / 2) ((b - 1) / 2)) →
  (a - b = -1) →
  a = 5 / 2 ∧ b = 7 / 2 :=
sorry

-- Define the line equation BC
def line_BC (x y : ℝ) : Prop := x - 9 * y + 29 = 0

-- Prove the equation of the line containing side BC
theorem find_line_BC :
  (x0 y0 : ℝ) →
  bisector_BC x0 y0 →
  (x0, y0) = (-2, 3) →
  line_BC x0 y0 :=
sorry

end find_point_B_find_line_BC_l73_73152


namespace num_students_even_l73_73167

theorem num_students_even (n : ℕ) (f : Fin n → ℤ → ℤ) :
  (∀ i, f i (-1) ∨ f i 1) →
  n % 2 = 0 :=
by
  sorry

end num_students_even_l73_73167


namespace commercials_count_l73_73417

-- Given conditions as definitions
def total_airing_time : ℤ := 90         -- 1.5 hours in minutes
def commercial_time : ℤ := 10           -- each commercial lasts 10 minutes
def show_time : ℤ := 60                 -- TV show (without commercials) lasts 60 minutes

-- Statement: Prove that the number of commercials is 3
theorem commercials_count :
  (total_airing_time - show_time) / commercial_time = 3 :=
sorry

end commercials_count_l73_73417


namespace simple_interest_rate_l73_73823

/-- Prove that given Principal (P) = 750, Amount (A) = 900, and Time (T) = 5 years,
    the rate (R) such that the Simple Interest formula holds is 4 percent. -/
theorem simple_interest_rate :
  ∀ (P A T : ℕ) (R : ℕ),
    P = 750 → 
    A = 900 → 
    T = 5 → 
    A = P + (P * R * T / 100) →
    R = 4 :=
by
  intros P A T R hP hA hT h_si
  sorry

end simple_interest_rate_l73_73823


namespace find_a3_l73_73124

variable (a_n : ℕ → ℤ) (a1 a4 a5 : ℤ)
variable (d : ℤ := -2)

-- Conditions
axiom h1 : ∀ n : ℕ, a_n (n + 1) = a_n n + d
axiom h2 : a4 = a1 + 3 * d
axiom h3 : a5 = a1 + 4 * d
axiom h4 : a4 * a4 = a1 * a5

-- Question to prove
theorem find_a3 : (a_n 3) = 5 := by
  sorry

end find_a3_l73_73124


namespace quadratic_inequality_solution_set_l73_73773

theorem quadratic_inequality_solution_set (p q : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) →
  p = 5 ∧ q = -6 ∧
  (∀ x : ℝ, - (1 : ℝ) / 2 < x ∧ x < - (1 : ℝ) / 3 → 6 * x^2 + 5 * x + 1 < 0) :=
by
  sorry

end quadratic_inequality_solution_set_l73_73773


namespace least_common_multiple_9_12_15_l73_73967

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l73_73967


namespace investment_ratio_same_period_l73_73983

-- Define the profits of A and B
def profit_A : ℕ := 60000
def profit_B : ℕ := 6000

-- Define their investment ratio given the same time period
theorem investment_ratio_same_period : profit_A / profit_B = 10 :=
by
  -- Proof skipped 
  sorry

end investment_ratio_same_period_l73_73983


namespace value_three_in_range_of_g_l73_73629

theorem value_three_in_range_of_g (a : ℝ) : ∀ (a : ℝ), ∃ (x : ℝ), x^2 + a * x + 1 = 3 :=
by
  sorry

end value_three_in_range_of_g_l73_73629


namespace probability_two_different_colors_l73_73143

-- Conditions
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def total_chips := blue_chips + red_chips + yellow_chips -- 15

-- Proof problem: Prove that the probability of drawing two chips of different colors is 148/225
theorem probability_two_different_colors : 
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips) = 
  148 / 225 := by
  sorry

end probability_two_different_colors_l73_73143


namespace value_of_x_minus_y_l73_73011

theorem value_of_x_minus_y (x y : ℝ) (h1 : abs x = 4) (h2 : abs y = 7) (h3 : x + y > 0) :
  x - y = -3 ∨ x - y = -11 :=
sorry

end value_of_x_minus_y_l73_73011


namespace largest_initial_number_l73_73399

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l73_73399


namespace zeros_of_f_l73_73938

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 2

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by {
  sorry
}

end zeros_of_f_l73_73938


namespace find_cost_price_l73_73491

variable (CP : ℝ)

def SP1 : ℝ := 0.80 * CP
def SP2 : ℝ := 1.06 * CP

axiom cond1 : SP2 - SP1 = 520

theorem find_cost_price : CP = 2000 :=
by
  sorry

end find_cost_price_l73_73491


namespace min_value_a_plus_b_l73_73547

theorem min_value_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (2 / a) + (2 / b) = 1) :
  a + b >= 8 :=
sorry

end min_value_a_plus_b_l73_73547


namespace number_of_ways_to_choose_reading_materials_l73_73450

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l73_73450


namespace twentyfive_percent_in_usd_l73_73237

variable (X : ℝ)
variable (Y : ℝ) (hY : Y > 0)

theorem twentyfive_percent_in_usd : 0.25 * X * Y = (0.25 : ℝ) * X * Y := by
  sorry

end twentyfive_percent_in_usd_l73_73237


namespace maryann_free_time_l73_73287

theorem maryann_free_time
    (x : ℕ)
    (expensive_time : ℕ := 8)
    (friends : ℕ := 3)
    (total_time : ℕ := 42)
    (lockpicking_time : 3 * (x + expensive_time) = total_time) : 
    x = 6 :=
by
  sorry

end maryann_free_time_l73_73287


namespace rectangle_area_expression_l73_73927

theorem rectangle_area_expression {d x : ℝ} (h : d^2 = 29 * x^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = (10 / 29) :=
by {
 sorry
}

end rectangle_area_expression_l73_73927


namespace proof_correctness_l73_73639

-- Define the new operation
def new_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Definitions for the conclusions
def conclusion_1 : Prop := new_op 1 (-2) = -8
def conclusion_2 : Prop := ∀ a b : ℝ, new_op a b = new_op b a
def conclusion_3 : Prop := ∀ a b : ℝ, new_op a b = 0 → a = 0
def conclusion_4 : Prop := ∀ a b : ℝ, a + b = 0 → (new_op a a + new_op b b = 8 * a^2)

-- Specify the correct conclusions
def correct_conclusions : Prop := conclusion_1 ∧ conclusion_2 ∧ ¬conclusion_3 ∧ conclusion_4

-- State the theorem
theorem proof_correctness : correct_conclusions := by
  sorry

end proof_correctness_l73_73639


namespace inscribed_circle_radius_range_l73_73593

noncomputable def r_range (AD DB : ℝ) (angle_A : ℝ) : Set ℝ :=
  { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 }

theorem inscribed_circle_radius_range (AD DB : ℝ) (angle_A : ℝ) (h1 : AD = 2 * Real.sqrt 3) 
    (h2 : DB = Real.sqrt 3) (h3 : angle_A > 60) : 
    r_range AD DB angle_A = { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 } :=
by
  sorry

end inscribed_circle_radius_range_l73_73593


namespace last_bead_is_black_l73_73989

-- Definition of the repeating pattern
def pattern := [1, 2, 3, 1, 2]  -- 1: black, 2: white, 3: gray (one full cycle)

-- Given constants
def total_beads : Nat := 91
def pattern_length : Nat := List.length pattern  -- This should be 9

-- Proof statement: The last bead is black
theorem last_bead_is_black : pattern[(total_beads % pattern_length) - 1] = 1 :=
by
  -- The following steps would be the proof which is not required
  sorry

end last_bead_is_black_l73_73989


namespace probability_same_color_l73_73142

-- Define the total combinations function
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- The given values from the problem
def whiteBalls := 2
def blackBalls := 3
def totalBalls := whiteBalls + blackBalls
def drawnBalls := 2

-- Calculate combinations
def comb_white_2 := comb whiteBalls drawnBalls
def comb_black_2 := comb blackBalls drawnBalls
def comb_total_2 := comb totalBalls drawnBalls

-- The correct answer given in the solution
def correct_probability := 2 / 5

-- Statement for the proof in Lean
theorem probability_same_color : (comb_white_2 + comb_black_2) / comb_total_2 = correct_probability := by
  sorry

end probability_same_color_l73_73142


namespace mina_numbers_l73_73157

theorem mina_numbers (a b : ℤ) (h1 : 3 * a + 4 * b = 140) (h2 : a = 20 ∨ b = 20) : a = 20 ∧ b = 20 :=
by
  sorry

end mina_numbers_l73_73157


namespace integer_b_if_integer_a_l73_73595

theorem integer_b_if_integer_a (a b : ℤ) (h : 2 * a + a^2 = 2 * b + b^2) : (∃ a' : ℤ, a = a') → ∃ b' : ℤ, b = b' :=
by
-- proof will be filled in here
sorry

end integer_b_if_integer_a_l73_73595


namespace common_term_sequence_7n_l73_73299

theorem common_term_sequence_7n (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (7 / 9) * (10^n - 1) :=
by
  sorry

end common_term_sequence_7n_l73_73299


namespace no_real_roots_of_quadratic_l73_73184

theorem no_real_roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = 1 ∧ c = 1) :
  (b^2 - 4 * a * c < 0) → ¬∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end no_real_roots_of_quadratic_l73_73184


namespace series_sum_l73_73673

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 2) / ((6 * n - 5)^2 * (6 * n + 1)^2)

theorem series_sum :
  (∑' n : ℕ, series_term (n + 1)) = 1 / 6 :=
by
  sorry

end series_sum_l73_73673


namespace time_to_walk_2_miles_l73_73261

/-- I walked 2 miles in a certain amount of time. -/
def walked_distance : ℝ := 2

/-- If I maintained this pace for 8 hours, I would walk 16 miles. -/
def pace_condition (pace : ℝ) : Prop :=
  pace * 8 = 16

/-- Prove that it took me 1 hour to walk 2 miles. -/
theorem time_to_walk_2_miles (t : ℝ) (pace : ℝ) (h1 : walked_distance = pace * t) (h2 : pace_condition pace) :
  t = 1 :=
sorry

end time_to_walk_2_miles_l73_73261


namespace joshua_needs_more_cents_l73_73896

-- Definitions of inputs
def cost_of_pen_dollars : ℕ := 6
def joshua_money_dollars : ℕ := 5
def borrowed_cents : ℕ := 68

-- Convert dollar amounts to cents
def dollar_to_cents (d : ℕ) : ℕ := d * 100

def cost_of_pen_cents := dollar_to_cents cost_of_pen_dollars
def joshua_money_cents := dollar_to_cents joshua_money_dollars

-- Total amount Joshua has in cents
def total_cents := joshua_money_cents + borrowed_cents

-- Calculation of the required amount
def needed_cents := cost_of_pen_cents - total_cents

theorem joshua_needs_more_cents : needed_cents = 32 := by 
  sorry

end joshua_needs_more_cents_l73_73896


namespace number_of_ways_to_choose_materials_l73_73474

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l73_73474


namespace number_of_ways_to_choose_reading_materials_l73_73447

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l73_73447


namespace tree_planting_activity_l73_73635

theorem tree_planting_activity (x y : ℕ) 
  (h1 : y = 2 * x + 15)
  (h2 : x = y / 3 + 6) : 
  y = 81 ∧ x = 33 := 
by sorry

end tree_planting_activity_l73_73635


namespace isosceles_triangle_area_l73_73780

theorem isosceles_triangle_area :
  ∀ (P Q R S : ℝ) (h1 : dist P Q = 26) (h2 : dist P R = 26) (h3 : dist Q R = 50),
  ∃ (area : ℝ), area = 25 * Real.sqrt 51 :=
by
  sorry

end isosceles_triangle_area_l73_73780


namespace compare_neg_fractions_l73_73337

theorem compare_neg_fractions : (-5 / 4) < (-4 / 5) := sorry

end compare_neg_fractions_l73_73337


namespace sum_of_roots_l73_73784

theorem sum_of_roots (a b c : ℚ) (h_eq : 6 * a^3 + 7 * a^2 - 12 * a = 0) (h_eq_b : 6 * b^3 + 7 * b^2 - 12 * b = 0) (h_eq_c : 6 * c^3 + 7 * c^2 - 12 * c = 0) : 
  a + b + c = -7/6 := 
by
  -- Insert proof steps here
  sorry

end sum_of_roots_l73_73784


namespace xiaoming_mirrored_time_l73_73655

-- Define the condition: actual time is 7:10 AM.
def actual_time : (ℕ × ℕ) := (7, 10)

-- Define a function to compute the mirrored time given an actual time.
def mirror_time (h m : ℕ) : (ℕ × ℕ) :=
  let mirrored_minute := if m = 0 then 0 else 60 - m
  let mirrored_hour := if m = 0 then if h = 12 then 12 else (12 - h) % 12
                        else if h = 12 then 11 else (11 - h) % 12
  (mirrored_hour, mirrored_minute)

-- Our goal is to verify that the mirrored time of 7:10 is 4:50.
theorem xiaoming_mirrored_time : mirror_time 7 10 = (4, 50) :=
by
  -- Proof will verify that mirror_time (7, 10) evaluates to (4, 50).
  sorry

end xiaoming_mirrored_time_l73_73655


namespace diana_debt_l73_73229

noncomputable def calculate_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem diana_debt :
  calculate_amount 75 0.07 12 1 ≈ 80.38 :=
by
  sorry

end diana_debt_l73_73229


namespace sum_of_consecutive_integers_l73_73436

theorem sum_of_consecutive_integers (n : ℕ) (h : n*(n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end sum_of_consecutive_integers_l73_73436


namespace time_to_travel_from_B_to_A_without_paddles_l73_73118

-- Variables definition 
variables (v v_r S : ℝ)
-- Assume conditions
def condition_1 (t₁ t₂ : ℝ) (v v_r S : ℝ) := t₁ = 3 * t₂
def t₁ (S v v_r : ℝ) := S / (v + v_r)
def t₂ (S v v_r : ℝ) := S / (v - v_r)

theorem time_to_travel_from_B_to_A_without_paddles
  (v v_r S : ℝ)
  (h1 : v = 2 * v_r)
  (h2 : t₁ S v v_r = 3 * t₂ S v v_r) :
  let t_no_paddle := S / v_r in
  t_no_paddle = 3 * t₂ S v v_r :=
sorry

end time_to_travel_from_B_to_A_without_paddles_l73_73118


namespace normal_line_at_point_l73_73791

noncomputable def curve (x : ℝ) : ℝ := (4 * x - x ^ 2) / 4

theorem normal_line_at_point (x0 : ℝ) (h : x0 = 2) :
  ∃ (L : ℝ → ℝ), ∀ (x : ℝ), L x = (2 : ℝ) :=
by
  sorry

end normal_line_at_point_l73_73791


namespace total_action_figures_l73_73331

theorem total_action_figures (figures_per_shelf : ℕ) (number_of_shelves : ℕ) (h1 : figures_per_shelf = 10) (h2 : number_of_shelves = 8) : figures_per_shelf * number_of_shelves = 80 := by
  sorry

end total_action_figures_l73_73331


namespace express_h_l73_73128

variable (a b S h : ℝ)
variable (h_formula : S = 1/2 * (a + b) * h)
variable (h_nonzero : a + b ≠ 0)

theorem express_h : h = 2 * S / (a + b) := 
by 
  sorry

end express_h_l73_73128


namespace at_least_3_speaking_l73_73881

noncomputable def probability_speaking : ℚ := 1 / 3

theorem at_least_3_speaking (n : ℕ) (speaking : ℕ → Prop) : 
  n = 6 →
  (∀ k, speaking k ↔ k ≤ n ∧ k ≥ 0) →
  (∀ k, ProbabilitySpace.Probability (fun _ => speaking k) = probability_speaking) →
  ProbabilitySpace.Probability (fun _ => (∑ k in finset.range 7, k * (ProbabilitySpace.Probability (fun _ => speaking k))) ≥ 3) = 233 / 729 :=
by sorry

end at_least_3_speaking_l73_73881


namespace professional_tax_correct_l73_73046

-- Define the total income and professional deductions
def total_income : ℝ := 50000
def professional_deductions : ℝ := 35000

-- Define the tax rates
def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_exp : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

-- Define the expected tax amount
def expected_tax_professional_income : ℝ := 2000

-- Define a function to calculate the professional income tax for self-employed individuals
def calculate_professional_income_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

-- Define the main theorem to assert the correctness of the tax calculation
theorem professional_tax_correct :
  calculate_professional_income_tax total_income tax_rate_professional_income = expected_tax_professional_income :=
by
  sorry

end professional_tax_correct_l73_73046


namespace compute_k_plus_m_l73_73155

theorem compute_k_plus_m :
  ∃ k m : ℝ, 
    (∀ (x y z : ℝ), x^3 - 9 * x^2 + k * x - m = 0 -> x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9 ∧ 
    (x = 1 ∨ y = 1 ∨ z = 1) ∧ (x = 3 ∨ y = 3 ∨ z = 3) ∧ (x = 5 ∨ y = 5 ∨ z = 5)) →
    k + m = 38 :=
by
  sorry

end compute_k_plus_m_l73_73155


namespace sin_600_eq_neg_sqrt3_div_2_l73_73066

theorem sin_600_eq_neg_sqrt3_div_2 :
  Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_600_eq_neg_sqrt3_div_2_l73_73066


namespace max_branch_diameter_l73_73778

theorem max_branch_diameter (d : ℝ) (w : ℝ) (angle : ℝ) (H: w = 1 ∧ angle = 90) :
  d ≤ 2 * Real.sqrt 2 + 2 := 
sorry

end max_branch_diameter_l73_73778


namespace sum_of_solutions_correct_l73_73613

noncomputable def sum_of_solutions : ℕ :=
  { 
    val := (List.range' 1 30 / 2).sum,
    property := sorry
  }

theorem sum_of_solutions_correct :
  (∀ x : ℕ, (x > 0 ∧ x ≤ 30 ∧ (17 * (5 * x - 3)).mod 10 = 34.mod 10) →
    List.mem x ((List.range' 1 30).filter (λ n, n % 2 = 1)) →
    ((List.range' 1 30 / 2).sum = 225)) :=
by {
  intro x h1 h2,
  sorry
}

end sum_of_solutions_correct_l73_73613


namespace firefighter_remaining_money_correct_l73_73506

noncomputable def firefighter_weekly_earnings : ℕ := 30 * 48
noncomputable def firefighter_monthly_earnings : ℕ := firefighter_weekly_earnings * 4
noncomputable def firefighter_rent_expense : ℕ := firefighter_monthly_earnings / 3
noncomputable def firefighter_food_expense : ℕ := 500
noncomputable def firefighter_tax_expense : ℕ := 1000
noncomputable def firefighter_total_expenses : ℕ := firefighter_rent_expense + firefighter_food_expense + firefighter_tax_expense
noncomputable def firefighter_remaining_money : ℕ := firefighter_monthly_earnings - firefighter_total_expenses

theorem firefighter_remaining_money_correct :
  firefighter_remaining_money = 2340 :=
by 
  rfl

end firefighter_remaining_money_correct_l73_73506


namespace boiling_temperature_l73_73278

-- Definitions according to conditions
def initial_temperature : ℕ := 41

def temperature_increase_per_minute : ℕ := 3

def pasta_cooking_time : ℕ := 12

def mixing_and_salad_time : ℕ := pasta_cooking_time / 3

def total_evening_time : ℕ := 73

-- Conditions and the problem statement in Lean
theorem boiling_temperature :
  initial_temperature + (total_evening_time - (pasta_cooking_time + mixing_and_salad_time)) * temperature_increase_per_minute = 212 :=
by
  -- Here would be the proof, skipped with sorry
  sorry

end boiling_temperature_l73_73278


namespace rise_in_water_level_l73_73632

noncomputable def edge : ℝ := 15.0
noncomputable def base_length : ℝ := 20.0
noncomputable def base_width : ℝ := 15.0
noncomputable def volume_cube : ℝ := edge ^ 3
noncomputable def base_area : ℝ := base_length * base_width

theorem rise_in_water_level :
  (volume_cube / base_area) = 11.25 :=
by
  sorry

end rise_in_water_level_l73_73632


namespace price_per_package_l73_73333

theorem price_per_package (P : ℝ) (hp1 : 10 * P + 50 * (4 / 5 * P) = 1096) :
  P = 21.92 :=
by 
  sorry

end price_per_package_l73_73333


namespace final_price_correct_l73_73025

def original_price : Float := 100
def store_discount_rate : Float := 0.20
def promo_discount_rate : Float := 0.10
def sales_tax_rate : Float := 0.05
def handling_fee : Float := 5

def final_price (original_price : Float) 
                (store_discount_rate : Float) 
                (promo_discount_rate : Float) 
                (sales_tax_rate : Float) 
                (handling_fee : Float) 
                : Float :=
  let price_after_store_discount := original_price * (1 - store_discount_rate)
  let price_after_promo := price_after_store_discount * (1 - promo_discount_rate)
  let price_after_tax := price_after_promo * (1 + sales_tax_rate)
  let total_price := price_after_tax + handling_fee
  total_price

theorem final_price_correct : final_price original_price store_discount_rate promo_discount_rate sales_tax_rate handling_fee = 80.60 :=
by
  simp only [
    original_price,
    store_discount_rate,
    promo_discount_rate,
    sales_tax_rate,
    handling_fee
  ]
  norm_num
  sorry

end final_price_correct_l73_73025


namespace largest_initial_number_l73_73390

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l73_73390


namespace regular_polygon_sides_l73_73087

theorem regular_polygon_sides (h : ∀ n : ℕ, n ≥ 3 → (total_internal_angle_sum / n) = 150) :
    n = 12 := by
  sorry

end regular_polygon_sides_l73_73087


namespace calculate_fraction_l73_73537

-- Define the fractions we are working with
def fraction1 : ℚ := 3 / 4
def fraction2 : ℚ := 15 / 5
def one_half : ℚ := 1 / 2

-- Define the main calculation
def main_fraction (f1 f2 one_half : ℚ) : ℚ := f1 * f2 - one_half

-- State the theorem
theorem calculate_fraction : main_fraction fraction1 fraction2 one_half = (7 / 4) := by
  sorry

end calculate_fraction_l73_73537


namespace total_value_correct_l73_73500

noncomputable def total_value (num_coins : ℕ) : ℕ :=
  let value_one_rupee := num_coins * 1
  let value_fifty_paise := (num_coins * 50) / 100
  let value_twentyfive_paise := (num_coins * 25) / 100
  value_one_rupee + value_fifty_paise + value_twentyfive_paise

theorem total_value_correct :
  let num_coins := 40
  total_value num_coins = 70 := by
  sorry

end total_value_correct_l73_73500


namespace x_is_36_percent_of_z_l73_73561

variable (x y z : ℝ)

theorem x_is_36_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.30 * z) : x = 0.36 * z :=
by
  sorry

end x_is_36_percent_of_z_l73_73561


namespace time_for_A_l73_73217

noncomputable def work_days (A B C D E : ℝ) : Prop :=
  (1/A + 1/B + 1/C + 1/D = 1/8) ∧
  (1/B + 1/C + 1/D + 1/E = 1/6) ∧
  (1/A + 1/E = 1/12)

theorem time_for_A (A B C D E : ℝ) (h : work_days A B C D E) : A = 48 :=
  by
    sorry

end time_for_A_l73_73217


namespace decreasing_function_l73_73816

-- Define the functions
noncomputable def fA (x : ℝ) : ℝ := 3^x
noncomputable def fB (x : ℝ) : ℝ := Real.logb 0.5 x
noncomputable def fC (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fD (x : ℝ) : ℝ := 1/x

-- Define the domains
def domainA : Set ℝ := Set.univ
def domainB : Set ℝ := {x | x > 0}
def domainC : Set ℝ := {x | x ≥ 0}
def domainD : Set ℝ := {x | x < 0} ∪ {x | x > 0}

-- Prove that fB is the only decreasing function in its domain
theorem decreasing_function:
  (∀ x y, x ∈ domainA → y ∈ domainA → x < y → fA x > fA y) = false ∧
  (∀ x y, x ∈ domainB → y ∈ domainB → x < y → fB x > fB y) ∧
  (∀ x y, x ∈ domainC → y ∈ domainC → x < y → fC x > fC y) = false ∧
  (∀ x y, x ∈ domainD → y ∈ domainD → x < y → fD x > fD y) = false :=
  sorry

end decreasing_function_l73_73816


namespace approximate_reading_l73_73431

-- Define the given conditions
def arrow_location_between (a b : ℝ) : Prop := a < 42.3 ∧ 42.6 < b

-- Statement of the proof problem
theorem approximate_reading (a b : ℝ) (ha : arrow_location_between a b) :
  a = 42.3 :=
sorry

end approximate_reading_l73_73431


namespace expectation_equal_sets_l73_73029

noncomputable section

variables {X : Type*} [MeasureSpace X] [ProbabilitySpace X] (X_rv : X → ℝ) [IsNonNeg X_rv] [IsNonDegenerate X_rv]

theorem expectation_equal_sets (a b c d : ℝ) (h_sum : a + b = c + d) :
  ( 𝔼[X_rv ^ a] * 𝔼[X_rv ^ b] = 𝔼[X_rv ^ c] * 𝔼[X_rv ^ d] ) → ({a, b} = {c, d}) :=
by
  sorry

end expectation_equal_sets_l73_73029


namespace radius_of_curvature_at_final_point_l73_73214

-- Definitions used in the conditions:
def v0 : ℝ := 10  -- initial velocity
def theta : ℝ := real.pi / 3  -- 60 degrees in radians
def g : ℝ := 10  -- acceleration due to gravity

-- The proof statement:
theorem radius_of_curvature_at_final_point 
  (v0 : ℝ)
  (theta : ℝ)
  (g : ℝ) :
  v0 = 10 → theta = real.pi / 3 → g = 10 → 
  (let vf := v0 in
   let ac := g * real.cos theta in
   let R := (vf^2) / ac in
   R = 20) :=
begin
  -- We should provide no proof steps here, just the proof statement
  sorry
end

end radius_of_curvature_at_final_point_l73_73214


namespace ratio_sum_2_or_4_l73_73283

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end ratio_sum_2_or_4_l73_73283


namespace new_person_weight_increase_avg_l73_73918

theorem new_person_weight_increase_avg
  (W : ℝ) -- total weight of the original 20 people
  (new_person_weight : ℝ) -- weight of the new person
  (h1 : (W - 80 + new_person_weight) = W + 20 * 15) -- condition given in the problem
  : new_person_weight = 380 := 
sorry

end new_person_weight_increase_avg_l73_73918


namespace fg_of_3_eq_29_l73_73043

def f (x : ℝ) : ℝ := 2 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l73_73043


namespace proof_problem_l73_73003

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end proof_problem_l73_73003


namespace tank_full_time_l73_73789

def tank_capacity : ℕ := 900
def fill_rate_A : ℕ := 40
def fill_rate_B : ℕ := 30
def drain_rate_C : ℕ := 20
def cycle_time : ℕ := 3
def net_fill_per_cycle : ℕ := fill_rate_A + fill_rate_B - drain_rate_C

theorem tank_full_time :
  (tank_capacity / net_fill_per_cycle) * cycle_time = 54 :=
by
  sorry

end tank_full_time_l73_73789


namespace concentric_but_different_radius_l73_73710

noncomputable def circleF (x y : ℝ) : ℝ :=
  x^2 + y^2 - 1

def pointP (x : ℝ) : ℝ × ℝ :=
  (x, x)

def circleEquation (x y : ℝ) : Prop :=
  circleF x y = 0

def circleEquation' (x y : ℝ) : Prop :=
  circleF x y - circleF x y = 0

theorem concentric_but_different_radius (x : ℝ) (hP : circleF x x ≠ 0) (hCenter : x ≠ 0):
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧
    ∀ x y, (circleEquation x y ↔ x^2 + y^2 = 1) ∧ 
           (circleEquation' x y ↔ x^2 + y^2 = 2) :=
by
  sorry

end concentric_but_different_radius_l73_73710


namespace ratio_of_weights_l73_73191

noncomputable def tyler_weight (sam_weight : ℝ) : ℝ := sam_weight + 25
noncomputable def ratio_of_peter_to_tyler (peter_weight tyler_weight : ℝ) : ℝ := peter_weight / tyler_weight

theorem ratio_of_weights (sam_weight : ℝ) (peter_weight : ℝ) (h_sam : sam_weight = 105) (h_peter : peter_weight = 65) :
  ratio_of_peter_to_tyler peter_weight (tyler_weight sam_weight) = 0.5 := by
  -- We use the conditions to derive the information
  sorry

end ratio_of_weights_l73_73191


namespace regular_pentagons_similar_l73_73076

-- Define a regular pentagon
structure RegularPentagon :=
  (side_length : ℝ)
  (internal_angle : ℝ)
  (angle_eq : internal_angle = 108)
  (side_positive : side_length > 0)

-- The theorem stating that two regular pentagons are always similar
theorem regular_pentagons_similar (P Q : RegularPentagon) : 
  ∀ P Q : RegularPentagon, P.internal_angle = Q.internal_angle ∧ P.side_length * Q.side_length ≠ 0 := 
sorry

end regular_pentagons_similar_l73_73076


namespace sin_cos_identity_l73_73869

theorem sin_cos_identity (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l73_73869


namespace first_person_job_completion_time_l73_73604

noncomputable def job_completion_time :=
  let A := 1 - (1/5)
  let C := 1/8
  let combined_rate := A + C
  have h1 : combined_rate = 0.325 := by
    sorry
  have h2 : A ≠ 0 := by
    sorry
  (1 / A : ℝ)
  
theorem first_person_job_completion_time :
  job_completion_time = 1.25 :=
by
  sorry

end first_person_job_completion_time_l73_73604


namespace estate_area_is_correct_l73_73512

noncomputable def actual_area_of_estate (length_in_inches : ℕ) (width_in_inches : ℕ) (scale : ℕ) : ℕ :=
  let actual_length := length_in_inches * scale
  let actual_width := width_in_inches * scale
  actual_length * actual_width

theorem estate_area_is_correct :
  actual_area_of_estate 9 6 350 = 6615000 := by
  -- Here, we would provide the proof steps, but for this exercise, we use sorry.
  sorry

end estate_area_is_correct_l73_73512


namespace tile_difference_l73_73571

theorem tile_difference :
  let initial_blue_tiles := 20
  let initial_green_tiles := 15
  let first_border_tiles := 18
  let second_border_tiles := 18
  let total_green_tiles := initial_green_tiles + first_border_tiles + second_border_tiles
  let total_blue_tiles := initial_blue_tiles
  total_green_tiles - total_blue_tiles = 31 := 
by
  sorry

end tile_difference_l73_73571


namespace ordered_pairs_count_l73_73377

theorem ordered_pairs_count : 
  ∃ n : ℕ, n = 6 ∧ ∀ A B : ℕ, (0 < A ∧ 0 < B) → (A * B = 32 ↔ A = 1 ∧ B = 32 ∨ A = 32 ∧ B = 1 ∨ A = 2 ∧ B = 16 ∨ A = 16 ∧ B = 2 ∨ A = 4 ∧ B = 8 ∨ A = 8 ∧ B = 4) := 
sorry

end ordered_pairs_count_l73_73377


namespace subtract_real_numbers_l73_73519

theorem subtract_real_numbers : 3.56 - 1.89 = 1.67 :=
by
  sorry

end subtract_real_numbers_l73_73519


namespace inequality_solution_set_l73_73825

theorem inequality_solution_set {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_inc : ∀ {x y : ℝ}, 0 < x → x < y → f x ≤ f y)
  (h_value : f 1 = 0) :
  {x | (f x - f (-x)) / x ≤ 0} = {x | -1 ≤ x ∧ x < 0} ∪ {x | 0 < x ∧ x ≤ 1} :=
by
  sorry


end inequality_solution_set_l73_73825


namespace lcm_of_9_12_15_is_180_l73_73959

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l73_73959


namespace lcm_9_12_15_l73_73971

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l73_73971


namespace smallest_x_value_l73_73863

theorem smallest_x_value : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (3 : ℚ) / 4 = y / (250 + x) ∧ x = 2 := by
  sorry

end smallest_x_value_l73_73863


namespace num_ways_choose_materials_l73_73456

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l73_73456


namespace no_real_solution_l73_73584

theorem no_real_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : 1 / a + 1 / b = 1 / (a + b)) : False :=
by
  sorry

end no_real_solution_l73_73584


namespace min_a_plus_b_l73_73030

theorem min_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : a + b >= 4 :=
sorry

end min_a_plus_b_l73_73030


namespace complex_number_solution_l73_73701

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (h_i : z * i = 2 + i) : z = 1 - 2 * i := by
  sorry

end complex_number_solution_l73_73701


namespace find_t_l73_73706

-- Define vectors a and b
def a := (3 : ℝ, 4 : ℝ)
def b := (1 : ℝ, 0 : ℝ)

-- Define the vector c as a function of t
def c (t : ℝ) := (a.1 + t * b.1, a.2 + t * b.2)

-- Statement of the theorem to be proven
theorem find_t (t : ℝ) :
  (a.1 * (a.1 + t * b.1) + a.2 * (a.2 + t * b.2)) = (b.1 * (a.1 + t * b.1) + b.2 * (a.2 + t * b.2)) →
  t = 5 :=
by
  sorry

end find_t_l73_73706


namespace Tim_carrots_count_l73_73382

theorem Tim_carrots_count (initial_potatoes new_potatoes initial_carrots final_potatoes final_carrots : ℕ) 
  (h_ratio : 3 * final_potatoes = 4 * final_carrots)
  (h_initial_potatoes : initial_potatoes = 32)
  (h_new_potatoes : new_potatoes = 28)
  (h_final_potatoes : final_potatoes = initial_potatoes + new_potatoes)
  (h_initial_ratio : 3 * 32 = 4 * initial_carrots) : 
  final_carrots = 45 :=
by {
  sorry
}

end Tim_carrots_count_l73_73382


namespace expand_and_simplify_l73_73682

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 9) = 2*x^2 + 24*x + 54 :=
by
  sorry

end expand_and_simplify_l73_73682


namespace slower_pump_time_l73_73790

def pool_problem (R : ℝ) :=
  (∀ t : ℝ, (2.5 * R * t = 1) → (t = 5))
  ∧ (∀ R1 R2 : ℝ, (R1 = 1.5 * R) → (R1 + R = 2.5 * R))
  ∧ (∀ t : ℝ, (R * t = 1) → (t = 12.5))

theorem slower_pump_time (R : ℝ) : pool_problem R :=
by
  -- Assume that the combined rates take 5 hours to fill the pool
  sorry

end slower_pump_time_l73_73790


namespace area_of_intersection_l73_73130

def setA : set (ℝ × ℝ) := { p | let (x, y) := p in (y - x) * (y - 1/x) ≥ 0 }
def setB : set (ℝ × ℝ) := { p | let (x, y) := p in (x - 1)^2 + (y - 1)^2 ≤ 1 }

theorem area_of_intersection : measure_theory.measure_space.volume (setA ∩ setB) = (real.pi / 2) :=
sorry

end area_of_intersection_l73_73130


namespace lacy_correct_percentage_is_80_l73_73724

-- Define the total number of problems
def total_problems (x : ℕ) : ℕ := 5 * x + 10

-- Define the number of problems Lacy missed
def problems_missed (x : ℕ) : ℕ := x + 2

-- Define the number of problems Lacy answered correctly
def problems_answered (x : ℕ) : ℕ := total_problems x - problems_missed x

-- Define the fraction of problems Lacy answered correctly
def fraction_answered_correctly (x : ℕ) : ℚ :=
  (problems_answered x : ℚ) / (total_problems x : ℚ)

-- The main theorem to prove the percentage of problems correctly answered is 80%
theorem lacy_correct_percentage_is_80 (x : ℕ) : 
  fraction_answered_correctly x = 4 / 5 := 
by 
  sorry

end lacy_correct_percentage_is_80_l73_73724


namespace quadratic_has_two_distinct_real_roots_l73_73014

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  (∃ a b c : ℝ, a = k - 2 ∧ b = -2 ∧ c = 1 / 2 ∧ a ≠ 0 ∧ b ^ 2 - 4 * a * c > 0) ↔ (k < 4 ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l73_73014


namespace find_value_of_x_squared_plus_inverse_squared_l73_73258

theorem find_value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + (1/x) = 2) : x^2 + (1/x^2) = 2 :=
sorry

end find_value_of_x_squared_plus_inverse_squared_l73_73258


namespace racers_in_final_segment_l73_73662

def initial_racers := 200

def racers_after_segment_1 (initial: ℕ) := initial - 10
def racers_after_segment_2 (after_segment_1: ℕ) := after_segment_1 - after_segment_1 / 3
def racers_after_segment_3 (after_segment_2: ℕ) := after_segment_2 - after_segment_2 / 4
def racers_after_segment_4 (after_segment_3: ℕ) := after_segment_3 - after_segment_3 / 3
def racers_after_segment_5 (after_segment_4: ℕ) := after_segment_4 - after_segment_4 / 2
def racers_after_segment_6 (after_segment_5: ℕ) := after_segment_5 - (3 * after_segment_5 / 4)

theorem racers_in_final_segment : racers_after_segment_6 (racers_after_segment_5 (racers_after_segment_4 (racers_after_segment_3 (racers_after_segment_2 (racers_after_segment_1 initial_racers))))) = 8 :=
  by
  sorry

end racers_in_final_segment_l73_73662


namespace total_number_of_turtles_l73_73444

variable {T : Type} -- Define a variable for the type of turtles

-- Define the conditions as hypotheses
variable (total_turtles : ℕ)
variable (female_percentage : ℚ) (male_percentage : ℚ)
variable (striped_male_prop : ℚ)
variable (baby_striped_males : ℕ) (adult_striped_males_prop : ℚ)
variable (striped_male_percentage : ℚ)
variable (striped_males : ℕ)
variable (male_turtles : ℕ)

-- Condition definitions
def female_percentage_def := female_percentage = 60 / 100
def male_percentage_def := male_percentage = 1 - female_percentage
def striped_male_prop_def := striped_male_prop = 1 / 4
def adult_striped_males_prop_def := adult_striped_males_prop = 60 / 100
def baby_and_adult_striped_males_prop_def := (1 - adult_striped_males_prop) = 40 / 100
def striped_males_def := striped_males = baby_striped_males / (1 - adult_striped_males_prop)
def male_turtles_def := male_turtles = striped_males / striped_male_prop
def male_turtles_percentage_def := male_turtles = total_turtles * (1 - female_percentage)

-- The proof statement to show the total number of turtles is 100
theorem total_number_of_turtles (h_female : female_percentage_def)
                                (h_male : male_percentage_def)
                                (h_striped_male_prop : striped_male_prop_def)
                                (h_adult_striped_males_prop : adult_striped_males_prop_def)
                                (h_baby_and_adult_striped_males_prop : baby_and_adult_striped_males_prop_def)
                                (h_striped_males : striped_males_def)
                                (h_male_turtles : male_turtles_def)
                                (h_male_turtles_percentage : male_turtles_percentage_def):
  total_turtles = 100 := 
by sorry

end total_number_of_turtles_l73_73444


namespace average_student_headcount_proof_l73_73221

def average_student_headcount : ℕ := (11600 + 11800 + 12000 + 11400) / 4

theorem average_student_headcount_proof :
  average_student_headcount = 11700 :=
by
  -- calculation here
  sorry

end average_student_headcount_proof_l73_73221


namespace original_number_l73_73144

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def permutations_sum (a b c : ℕ) : ℕ :=
  let abc := 100 * a + 10 * b + c
  let acb := 100 * a + 10 * c + b
  let bac := 100 * b + 10 * a + c
  let bca := 100 * b + 10 * c + a
  let cab := 100 * c + 10 * a + b
  let cba := 100 * c + 10 * b + a
  abc + acb + bac + bca + cab + cba

theorem original_number (abc : ℕ) (a b c : ℕ) :
  is_three_digit abc →
  abc = 100 * a + 10 * b + c →
  permutations_sum a b c = 3194 →
  abc = 358 :=
by
  sorry

end original_number_l73_73144


namespace find_side_b_l73_73151

theorem find_side_b
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2 / 3) :
  b = Real.sqrt 6 :=
by
  sorry

end find_side_b_l73_73151


namespace incorrect_statement_D_l73_73197

/-
Define the conditions for the problem:
- A prism intersected by a plane.
- The intersection of a sphere and a plane when the plane is less than the radius.
- The intersection of a plane parallel to the base of a circular cone.
- The geometric solid formed by rotating a right triangle around one of its sides.
- The incorrectness of statement D.
-/

noncomputable def intersect_prism_with_plane (prism : Type) (plane : Type) : Prop := sorry

noncomputable def sphere_intersection (sphere_radius : ℝ) (distance_to_plane : ℝ) : Type := sorry

noncomputable def cone_intersection (cone : Type) (plane : Type) : Type := sorry

noncomputable def rotation_result (triangle : Type) (side : Type) : Type := sorry

theorem incorrect_statement_D :
  ¬(rotation_result RightTriangle Side = Cone) :=
sorry

end incorrect_statement_D_l73_73197


namespace total_number_of_participants_l73_73018

theorem total_number_of_participants (boys_achieving_distance : ℤ) (frequency : ℝ) (h1 : boys_achieving_distance = 8) (h2 : frequency = 0.4) : 
  (boys_achieving_distance : ℝ) / frequency = 20 := 
by 
  sorry

end total_number_of_participants_l73_73018


namespace point_in_second_quadrant_l73_73889

def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant : is_in_second_quadrant (-2) 3 :=
by
  sorry

end point_in_second_quadrant_l73_73889


namespace square_of_1017_l73_73102

theorem square_of_1017 : 1017^2 = 1034289 :=
by
  sorry

end square_of_1017_l73_73102


namespace find_a_l73_73354

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) (h2 : x₁ = -2 * a) (h3 : x₂ = 4 * a) (h4 : x₂ - x₁ = 15) : a = 5 / 2 :=
by 
  sorry

end find_a_l73_73354


namespace persimmons_picked_l73_73735

theorem persimmons_picked : 
  ∀ (J H : ℕ), (4 * J = H - 3) → (H = 35) → (J = 8) := 
by
  intros J H hJ hH
  sorry

end persimmons_picked_l73_73735


namespace product_of_repeating_decimal_l73_73669

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l73_73669


namespace tree_growth_per_year_l73_73069

-- Defining the initial height and age.
def initial_height : ℕ := 5
def initial_age : ℕ := 1

-- Defining the height and age after a certain number of years.
def height_at_7_years : ℕ := 23
def age_at_7_years : ℕ := 7

-- Calculating the total growth and number of years.
def total_height_growth : ℕ := height_at_7_years - initial_height
def years_of_growth : ℕ := age_at_7_years - initial_age

-- Stating the theorem to be proven.
theorem tree_growth_per_year : total_height_growth / years_of_growth = 3 :=
by
  sorry

end tree_growth_per_year_l73_73069


namespace sum_of_a_for_unique_solution_l73_73113

theorem sum_of_a_for_unique_solution (a : ℝ) (h : (a + 12)^2 - 384 = 0) : 
  let a1 := -12 + 16 * Real.sqrt 6
  let a2 := -12 - 16 * Real.sqrt 6
  a1 + a2 = -24 := 
by
  sorry

end sum_of_a_for_unique_solution_l73_73113


namespace sum_of_solutions_eq_225_l73_73612

theorem sum_of_solutions_eq_225 :
  ∑ x in Finset.filter (λ x, 0 < x ∧ x ≤ 30 ∧ (17 * (5 * x - 3)) % 10 = 4) (Finset.range 31), x = 225 :=
by sorry

end sum_of_solutions_eq_225_l73_73612


namespace total_number_of_turtles_l73_73443

variable {T : Type} -- Define a variable for the type of turtles

-- Define the conditions as hypotheses
variable (total_turtles : ℕ)
variable (female_percentage : ℚ) (male_percentage : ℚ)
variable (striped_male_prop : ℚ)
variable (baby_striped_males : ℕ) (adult_striped_males_prop : ℚ)
variable (striped_male_percentage : ℚ)
variable (striped_males : ℕ)
variable (male_turtles : ℕ)

-- Condition definitions
def female_percentage_def := female_percentage = 60 / 100
def male_percentage_def := male_percentage = 1 - female_percentage
def striped_male_prop_def := striped_male_prop = 1 / 4
def adult_striped_males_prop_def := adult_striped_males_prop = 60 / 100
def baby_and_adult_striped_males_prop_def := (1 - adult_striped_males_prop) = 40 / 100
def striped_males_def := striped_males = baby_striped_males / (1 - adult_striped_males_prop)
def male_turtles_def := male_turtles = striped_males / striped_male_prop
def male_turtles_percentage_def := male_turtles = total_turtles * (1 - female_percentage)

-- The proof statement to show the total number of turtles is 100
theorem total_number_of_turtles (h_female : female_percentage_def)
                                (h_male : male_percentage_def)
                                (h_striped_male_prop : striped_male_prop_def)
                                (h_adult_striped_males_prop : adult_striped_males_prop_def)
                                (h_baby_and_adult_striped_males_prop : baby_and_adult_striped_males_prop_def)
                                (h_striped_males : striped_males_def)
                                (h_male_turtles : male_turtles_def)
                                (h_male_turtles_percentage : male_turtles_percentage_def):
  total_turtles = 100 := 
by sorry

end total_number_of_turtles_l73_73443


namespace lcm_9_12_15_l73_73976

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l73_73976


namespace cone_height_l73_73992

theorem cone_height (V : ℝ) (h r : ℝ) (π : ℝ) (h_eq_r : h = r) (volume_eq : V = 12288 * π) (V_def : V = (1/3) * π * r^3) : h = 36 := 
by
  sorry

end cone_height_l73_73992


namespace parallelogram_properties_l73_73829

noncomputable def perimeter (x y : ℤ) : ℝ :=
  2 * (5 + Real.sqrt ((x - 7) ^ 2 + (y - 3) ^ 2))

noncomputable def area (x y : ℤ) : ℝ :=
  5 * abs (y - 3)

theorem parallelogram_properties (x y : ℤ) (hx : x = 7) (hy : y = 7) :
  (perimeter x y + area x y) = 38 :=
by
  simp [perimeter, area, hx, hy]
  sorry

end parallelogram_properties_l73_73829


namespace tanya_addition_problem_l73_73405

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l73_73405


namespace line_through_midpoint_of_ellipse_l73_73364

theorem line_through_midpoint_of_ellipse:
  (∀ x y : ℝ, (x - 4)^2 + (y - 2)^2 = (1/36) * ((9 * 4) + 36 * (1 / 4)) → (1 + 2 * (y - 2) / (x - 4) = 0)) →
  (x - 8) + 2 * (y - 4) = 0 :=
by
  sorry

end line_through_midpoint_of_ellipse_l73_73364


namespace milton_books_l73_73750

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l73_73750


namespace smallest_four_digit_divisible_by_6_l73_73481

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l73_73481


namespace distance_focus_parabola_to_line_l73_73175

theorem distance_focus_parabola_to_line :
  let focus : ℝ × ℝ := (1, 0)
  let distance (p : ℝ × ℝ) (A B C : ℝ) : ℝ := |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)
  distance focus 1 (-Real.sqrt 3) 0 = 1 / 2 :=
by
  sorry

end distance_focus_parabola_to_line_l73_73175


namespace solve_sqrt_eq_l73_73837

theorem solve_sqrt_eq (z : ℚ) (h : Real.sqrt (5 - 4 * z) = 10) : z = -95 / 4 := by
  sorry

end solve_sqrt_eq_l73_73837


namespace decrease_is_75_86_percent_l73_73892

noncomputable def decrease_percent (x y z : ℝ) : ℝ :=
  let x' := 0.8 * x
  let y' := 0.75 * y
  let z' := 0.9 * z
  let original_value := x^2 * y^3 * z
  let new_value := (x')^2 * (y')^3 * z'
  let decrease_value := original_value - new_value
  decrease_value / original_value

theorem decrease_is_75_86_percent (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  decrease_percent x y z = 0.7586 :=
sorry

end decrease_is_75_86_percent_l73_73892


namespace simplify_evaluate_expr_l73_73908

noncomputable def expr (x : ℝ) : ℝ := 
  ( ( (x^2 - 3) / (x + 2) - x + 2 ) / ( (x^2 - 4) / (x^2 + 4*x + 4) ) )

theorem simplify_evaluate_expr : 
  expr (Real.sqrt 2 + 1) = Real.sqrt 2 + 1 := by
  sorry

end simplify_evaluate_expr_l73_73908


namespace area_shaded_region_in_hexagon_l73_73327

theorem area_shaded_region_in_hexagon (s : ℝ) (r : ℝ) (h_s : s = 4) (h_r : r = 2) :
  let area_hexagon := ((3 * Real.sqrt 3) / 2) * s^2
  let area_semicircle := (π * r^2) / 2
  let total_area_semicircles := 8 * area_semicircle
  let area_shaded_region := area_hexagon - total_area_semicircles
  area_shaded_region = 24 * Real.sqrt 3 - 16 * π :=
by {
  sorry
}

end area_shaded_region_in_hexagon_l73_73327


namespace point_on_y_axis_m_value_l73_73570

theorem point_on_y_axis_m_value (m : ℝ) (h : 6 - 2 * m = 0) : m = 3 := by
  sorry

end point_on_y_axis_m_value_l73_73570


namespace necessary_and_sufficient_condition_l73_73122

-- Sum of the first n terms of the sequence
noncomputable def S_n (n : ℕ) (c : ℤ) : ℤ := (n + 1) * (n + 1) + c

-- The nth term of the sequence
noncomputable def a_n (n : ℕ) (c : ℤ) : ℤ := S_n n c - (S_n (n - 1) c)

-- Define the sequence being arithmetic
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) - a n = d

theorem necessary_and_sufficient_condition (c : ℤ) :
  (∀ n ≥ 1, a_n n c - a_n (n-1) c = 2) ↔ (c = -1) :=
by
  sorry

end necessary_and_sufficient_condition_l73_73122


namespace set_intersection_eq_l73_73032

theorem set_intersection_eq (M N : Set ℝ) (hM : M = { x : ℝ | 0 < x ∧ x < 1 }) (hN : N = { x : ℝ | -2 < x ∧ x < 2 }) :
  M ∩ N = M :=
sorry

end set_intersection_eq_l73_73032


namespace circumscribed_radius_eq_eight_sec_half_angle_l73_73093

noncomputable def sector_radius {φ : ℝ} (hφ : φ > π / 2 ∧ φ < π) : ℝ :=
  8 * Real.sec (φ / 2)

theorem circumscribed_radius_eq_eight_sec_half_angle (φ : ℝ) (hφ : φ > π / 2 ∧ φ < π) :
  sector_radius hφ = 8 * Real.sec (φ / 2) :=
sorry

end circumscribed_radius_eq_eight_sec_half_angle_l73_73093


namespace sequence_sum_l73_73615

theorem sequence_sum (a b : ℤ) (h1 : ∃ d, d = 5 ∧ (∀ n : ℕ, (3 + n * d) = a ∨ (3 + (n-1) * d) = b ∨ (3 + (n-2) * d) = 33)) : 
  a + b = 51 :=
by
  sorry

end sequence_sum_l73_73615


namespace t_shirt_cost_l73_73991

theorem t_shirt_cost
  (marked_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (selling_price : ℝ)
  (cost : ℝ)
  (h1 : marked_price = 240)
  (h2 : discount_rate = 0.20)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = 0.8 * marked_price)
  (h5 : selling_price = cost + profit_rate * cost)
  : cost = 160 := 
sorry

end t_shirt_cost_l73_73991


namespace sequence_expression_l73_73132

noncomputable def a_n (n : ℕ) : ℤ :=
if n = 1 then -1 else 1 - 2^n

def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
2 * a_n n + n

theorem sequence_expression :
  ∀ n : ℕ, n > 0 → (a_n n = 1 - 2^n) :=
by
  intro n hn
  sorry

end sequence_expression_l73_73132


namespace cricket_team_right_handed_count_l73_73289

theorem cricket_team_right_handed_count 
  (total throwers non_throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h_total : total = 70)
  (h_throwers : throwers = 37)
  (h_non_throwers : non_throwers = total - throwers)
  (h_left_handed_non_throwers : left_handed_non_throwers = non_throwers / 3)
  (h_right_handed_non_throwers : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h_all_throwers_right_handed : ∀ (t : ℕ), t = throwers → t = right_handed_non_throwers + (total - throwers) - (non_throwers / 3)) :
  right_handed_non_throwers + throwers = 59 := 
by 
  sorry

end cricket_team_right_handed_count_l73_73289


namespace cost_of_chicken_l73_73035

theorem cost_of_chicken (cost_beef_per_pound : ℝ) (quantity_beef : ℝ) (cost_oil : ℝ) (total_grocery_cost : ℝ) (contribution_each : ℝ) :
  cost_beef_per_pound = 4 →
  quantity_beef = 3 →
  cost_oil = 1 →
  total_grocery_cost = 16 →
  contribution_each = 1 →
  ∃ (cost_chicken : ℝ), cost_chicken = 3 :=
by
  intros h1 h2 h3 h4 h5
  -- This line is required to help Lean handle any math operations
  have h6 := h1
  have h7 := h2
  have h8 := h3
  have h9 := h4
  have h10 := h5
  sorry

end cost_of_chicken_l73_73035


namespace amount_of_tin_in_new_mixture_l73_73638

def tin_in_alloy_A (weight_A : ℚ) : ℚ := (3/4) * weight_A
def tin_in_alloy_B (weight_B : ℚ) : ℚ := (3/8) * weight_B
def tin_in_alloy_C (weight_C : ℚ) : ℚ := (1/5) * weight_C

theorem amount_of_tin_in_new_mixture :
  tin_in_alloy_A 170 + tin_in_alloy_B 250 + tin_in_alloy_C 120 = 245.25 :=
by
  -- Proof goes here
  sorry

end amount_of_tin_in_new_mixture_l73_73638


namespace number_of_valid_permutations_l73_73742

noncomputable def is_permutation_of_set {α : Type*} [DecidableEq α] (s : Finset α) (l : List α) : Prop :=
l.perm s.to_list

theorem number_of_valid_permutations :
  {l : List ℕ // is_permutation_of_set {1, 2, 3, 4} l ∧ 
  (Nat.abs (l.nthLe 0 (by simp [Nat.lt_succ_self])) - 1) +
  (Nat.abs (l.nthLe 1 (by simp [Nat.lt_succ_self])) - 2) +
  (Nat.abs (l.nthLe 2 (by simp [Nat.lt_succ_self])) - 3) +
  (Nat.abs (l.nthLe 3 (by simp [Nat.lt_succ_self])) - 4) = 6 } = 9 := 
sorry

end number_of_valid_permutations_l73_73742


namespace solve_for_x_l73_73165
-- Lean 4 Statement

theorem solve_for_x (x : ℝ) (h : 2^(3 * x) = Real.sqrt 32) : x = 5 / 6 := 
sorry

end solve_for_x_l73_73165


namespace hardcover_books_count_l73_73347

theorem hardcover_books_count (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 30 * h + 15 * p = 270) : h = 6 :=
by
  sorry

end hardcover_books_count_l73_73347


namespace find_greater_number_l73_73063

theorem find_greater_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x > y) : x = 25 := 
sorry

end find_greater_number_l73_73063


namespace geometric_sequence_sum_l73_73061

theorem geometric_sequence_sum (S : ℕ → ℝ) (a₄_to_a₁₂_sum : ℝ):
  (S 3 = 2) → (S 6 = 6) → a₄_to_a₁₂_sum = (S 12 - S 3)  :=
by
  sorry

end geometric_sequence_sum_l73_73061


namespace smallest_uv_non_factor_of_48_l73_73948

theorem smallest_uv_non_factor_of_48 :
  ∃ (u v : ℕ) (hu : u ∣ 48) (hv : v ∣ 48), u ≠ v ∧ ¬ (u * v ∣ 48) ∧ u * v = 18 :=
sorry

end smallest_uv_non_factor_of_48_l73_73948


namespace find_x_l73_73937

noncomputable def inv_cubicroot (y x : ℝ) : ℝ := y * x^(1/3)

theorem find_x (x y : ℝ) (h1 : ∃ k, inv_cubicroot 2 8 = k) (h2 : y = 8) : x = 1 / 8 :=
by
  sorry

end find_x_l73_73937


namespace production_days_l73_73494

theorem production_days (n : ℕ) (h1 : (40 * n + 90) / (n + 1) = 45) : n = 9 :=
by
  sorry

end production_days_l73_73494


namespace smallest_tax_amount_is_professional_income_tax_l73_73047

def total_income : ℝ := 50000.00
def professional_deductions : ℝ := 35000.00

def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_expenditure : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

def ndfl_tax : ℝ := (total_income - professional_deductions) * tax_rate_ndfl
def simplified_tax_income : ℝ := total_income * tax_rate_simplified_income
def simplified_tax_income_minus_expenditure : ℝ := (total_income - professional_deductions) * tax_rate_simplified_income_minus_expenditure
def professional_income_tax : ℝ := total_income * tax_rate_professional_income

theorem smallest_tax_amount_is_professional_income_tax : 
  min (min ndfl_tax (min simplified_tax_income simplified_tax_income_minus_expenditure)) professional_income_tax = professional_income_tax := 
sorry

end smallest_tax_amount_is_professional_income_tax_l73_73047


namespace arielle_age_l73_73775

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end arielle_age_l73_73775


namespace groups_of_four_on_plane_l73_73821

-- Define the points in the tetrahedron
inductive Point
| vertex : Point
| midpoint : Point

noncomputable def points : List Point :=
  [Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.midpoint,
   Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.vertex]

-- Condition: all 10 points are either vertices or midpoints of the edges of a tetrahedron 
def points_condition : ∀ p ∈ points, p = Point.vertex ∨ p = Point.midpoint := sorry

-- Function to count unique groups of four points lying on the same plane
noncomputable def count_groups : ℕ :=
  33  -- Given as the correct answer in the problem

-- Proof problem stating the count of groups
theorem groups_of_four_on_plane : count_groups = 33 :=
by 
  sorry -- Proof omitted

end groups_of_four_on_plane_l73_73821


namespace largest_initial_number_l73_73396

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l73_73396


namespace perfect_square_divisible_by_12_l73_73164

theorem perfect_square_divisible_by_12 (k : ℤ) : 12 ∣ (k^2 * (k^2 - 1)) :=
by sorry

end perfect_square_divisible_by_12_l73_73164


namespace problem_statement_l73_73420

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end problem_statement_l73_73420


namespace lcm_of_9_12_15_is_180_l73_73963

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l73_73963


namespace poly_at_2_eq_0_l73_73520

def poly (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

theorem poly_at_2_eq_0 : poly 2 = 0 := by
  sorry

end poly_at_2_eq_0_l73_73520


namespace problem_statement_l73_73013

theorem problem_statement (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_eq : x + y + z = 1/x + 1/y + 1/z) : 
  x + y + z ≥ Real.sqrt ((x * y + 1) / 2) + Real.sqrt ((y * z + 1) / 2) + Real.sqrt ((z * x + 1) / 2) :=
by
  sorry

end problem_statement_l73_73013


namespace product_of_repeating_decimal_l73_73671

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l73_73671


namespace number_of_ways_l73_73459

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l73_73459


namespace megan_markers_final_count_l73_73288

theorem megan_markers_final_count :
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  final_markers = 582 :=
by
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  have h : final_markers = 582 := sorry
  exact h

end megan_markers_final_count_l73_73288


namespace train_crossing_time_l73_73514

/--
A train requires 8 seconds to pass a pole while it requires some seconds to cross a stationary train which is 400 meters long. 
The speed of the train is 144 km/h. Prove that it takes 18 seconds for the train to cross the stationary train.
-/
theorem train_crossing_time
  (train_speed_kmh : ℕ)
  (time_to_pass_pole : ℕ)
  (length_stationary_train : ℕ)
  (speed_mps : ℕ)
  (length_moving_train : ℕ)
  (total_length : ℕ)
  (crossing_time : ℕ) :
  train_speed_kmh = 144 →
  time_to_pass_pole = 8 →
  length_stationary_train = 400 →
  speed_mps = (train_speed_kmh * 1000) / 3600 →
  length_moving_train = speed_mps * time_to_pass_pole →
  total_length = length_moving_train + length_stationary_train →
  crossing_time = total_length / speed_mps →
  crossing_time = 18 :=
by
  intros;
  sorry

end train_crossing_time_l73_73514


namespace total_litter_weight_l73_73243

-- Definitions of the conditions
def gina_bags : ℕ := 2
def neighborhood_multiplier : ℕ := 82
def bag_weight : ℕ := 4

-- Representing the total calculation
def neighborhood_bags : ℕ := neighborhood_multiplier * gina_bags
def total_bags : ℕ := neighborhood_bags + gina_bags

def total_weight : ℕ := total_bags * bag_weight

-- Statement of the problem
theorem total_litter_weight : total_weight = 664 :=
by
  sorry

end total_litter_weight_l73_73243


namespace sin_range_l73_73926

theorem sin_range (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2)) : 
  Set.range (fun x => Real.sin x) = Set.Icc (1/2 : ℝ) 1 :=
sorry

end sin_range_l73_73926


namespace measure_of_angle_A_l73_73610

-- Defining the measures of angles
def angle_B : ℝ := 50
def angle_C : ℝ := 40
def angle_D : ℝ := 30

-- Prove that measure of angle A is 120 degrees given the conditions
theorem measure_of_angle_A (B C D : ℝ) (hB : B = angle_B) (hC : C = angle_C) (hD : D = angle_D) : B + C + D + 60 = 180 -> 180 - (B + C + D + 60) = 120 :=
by sorry

end measure_of_angle_A_l73_73610


namespace smallest_four_digit_divisible_by_six_l73_73483

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l73_73483


namespace sum_of_solutions_l73_73614

theorem sum_of_solutions : 
  let solutions := {x | 0 < x ∧ x ≤ 30 ∧ 17 * (5 * x - 3) % 10 = 34 % 10}
  in (∑ x in solutions, x) = 225 := by
  sorry

end sum_of_solutions_l73_73614


namespace total_pokemon_cards_l73_73021

-- Definitions based on conditions
def jenny_cards : ℕ := 6
def orlando_cards : ℕ := jenny_cards + 2
def richard_cards : ℕ := 3 * orlando_cards

-- The theorem stating the total number of cards
theorem total_pokemon_cards : jenny_cards + orlando_cards + richard_cards = 38 :=
by
  sorry

end total_pokemon_cards_l73_73021


namespace FerrisWheelCostIsSix_l73_73820

structure AmusementPark where
  roller_coaster_cost : ℕ
  log_ride_cost : ℕ
  initial_tickets : ℕ
  additional_tickets_needed : ℕ

def ferris_wheel_cost (a : AmusementPark) : ℕ :=
  let total_needed := a.initial_tickets + a.additional_tickets_needed
  let total_ride_cost := a.roller_coaster_cost + a.log_ride_cost
  total_needed - total_ride_cost

theorem FerrisWheelCostIsSix (a : AmusementPark) 
  (h₁ : a.roller_coaster_cost = 5)
  (h₂ : a.log_ride_cost = 7)
  (h₃ : a.initial_tickets = 2)
  (h₄ : a.additional_tickets_needed = 16) :
  ferris_wheel_cost a = 6 :=
by
  -- proof omitted
  sorry

end FerrisWheelCostIsSix_l73_73820


namespace five_letter_words_with_one_consonant_l73_73135

theorem five_letter_words_with_one_consonant :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E']
  let consonants := ['B', 'C', 'D', 'F']
  let total_words := (letters.length : ℕ)^5
  let vowel_only_words := (vowels.length : ℕ)^5
  total_words - vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_one_consonant_l73_73135


namespace square_rectangle_area_ratio_l73_73650

theorem square_rectangle_area_ratio (l1 l2 : ℕ) (h1 : l1 = 32) (h2 : l2 = 64) (p : ℕ) (s : ℕ) 
  (h3 : p = 256) (h4 : s = p / 4)  :
  (s * s) / (l1 * l2) = 2 := 
by
  sorry

end square_rectangle_area_ratio_l73_73650


namespace geometric_sum_common_ratio_l73_73933

theorem geometric_sum_common_ratio (a₁ a₂ : ℕ) (q : ℕ) (S₃ : ℕ)
  (h1 : S₃ = a₁ + 3 * a₂)
  (h2: S₃ = a₁ * (1 + q + q^2)) :
  q = 2 :=
by
  sorry

end geometric_sum_common_ratio_l73_73933


namespace sum_fifth_to_seventh_terms_arith_seq_l73_73919

theorem sum_fifth_to_seventh_terms_arith_seq (a d : ℤ)
  (h1 : a + 7 * d = 16) (h2 : a + 8 * d = 22) (h3 : a + 9 * d = 28) :
  (a + 4 * d) + (a + 5 * d) + (a + 6 * d) = 12 :=
by
  sorry

end sum_fifth_to_seventh_terms_arith_seq_l73_73919


namespace cars_minus_trucks_l73_73680

theorem cars_minus_trucks (total : ℕ) (trucks : ℕ) (h_total : total = 69) (h_trucks : trucks = 21) :
  (total - trucks) - trucks = 27 :=
by
  sorry

end cars_minus_trucks_l73_73680


namespace point_coordinates_l73_73755

namespace CoordinateProof

structure Point where
  x : ℝ
  y : ℝ

def isSecondQuadrant (P : Point) : Prop := P.x < 0 ∧ P.y > 0
def distToXAxis (P : Point) : ℝ := |P.y|
def distToYAxis (P : Point) : ℝ := |P.x|

theorem point_coordinates (P : Point) (h1 : isSecondQuadrant P) (h2 : distToXAxis P = 3) (h3 : distToYAxis P = 7) : P = ⟨-7, 3⟩ :=
by
  sorry

end CoordinateProof

end point_coordinates_l73_73755


namespace third_student_number_l73_73988

theorem third_student_number (A B C D : ℕ) 
  (h1 : A + B + C + D = 531) 
  (h2 : A + B = C + D + 31) 
  (h3 : C = D + 22) : 
  C = 136 := 
by
  sorry

end third_student_number_l73_73988


namespace number_of_m_gons_proof_l73_73580

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end number_of_m_gons_proof_l73_73580


namespace anne_cleaning_time_l73_73493

theorem anne_cleaning_time (B A : ℝ) 
  (h₁ : 4 * (B + A) = 1) 
  (h₂ : 3 * (B + 2 * A) = 1) : 
  1 / A = 12 :=
sorry

end anne_cleaning_time_l73_73493


namespace max_books_borrowed_l73_73788

theorem max_books_borrowed (total_students : ℕ) (students_no_books : ℕ) (students_1_book : ℕ)
  (students_2_books : ℕ) (avg_books_per_student : ℕ) (remaining_students_borrowed_at_least_3 :
  ∀ (s : ℕ), s ≥ 3) :
  total_students = 25 →
  students_no_books = 3 →
  students_1_book = 11 →
  students_2_books = 6 →
  avg_books_per_student = 2 →
  ∃ (max_books : ℕ), max_books = 15 :=
  by
  sorry

end max_books_borrowed_l73_73788


namespace max_perimeter_of_polygons_l73_73038

theorem max_perimeter_of_polygons 
  (t s : ℕ) 
  (hts : t + s = 7) 
  (hsum_angles : 60 * t + 90 * s = 360) 
  (max_squares : s ≤ 4) 
  (side_length : ℕ := 2) 
  (tri_perimeter : ℕ := 3 * side_length) 
  (square_perimeter : ℕ := 4 * side_length) :
  2 * (t * tri_perimeter + s * square_perimeter) = 68 := 
sorry

end max_perimeter_of_polygons_l73_73038


namespace total_Pokemon_cards_l73_73022

def j : Nat := 6
def o : Nat := j + 2
def r : Nat := 3 * o
def t : Nat := j + o + r

theorem total_Pokemon_cards : t = 38 := by 
  sorry

end total_Pokemon_cards_l73_73022


namespace girls_points_l73_73017

theorem girls_points (g b : ℕ) (total_points : ℕ) (points_g : ℕ) (points_b : ℕ) :
  b = 9 * g ∧
  total_points = 10 * g * (10 * g - 1) ∧
  points_g = 2 * g * (10 * g - 1) ∧
  points_b = 4 * points_g ∧
  total_points = points_g + points_b
  → points_g = 18 := 
by
  sorry

end girls_points_l73_73017


namespace canoe_no_paddle_time_l73_73117

-- All conditions needed for the problem
variables {S v v_r : ℝ}
variables (time_pa time_pb : ℝ)

-- Condition that time taken from A to B is 3 times the time taken from B to A
def condition1 : Prop := time_pa = 3 * time_pb

-- Define time taken from A to B (downstream) and B to A (upstream)
def time_pa_def : time_pa = S / (v + v_r) := sorry
def time_pb_def : time_pb = S / (v - v_r) := sorry

-- Main theorem stating the problem to prove
theorem canoe_no_paddle_time :
  condition1 →
  ∃ (t_no_paddle : ℝ), t_no_paddle = 3 * time_pb :=
begin
  intro h1,
  sorry
end

end canoe_no_paddle_time_l73_73117


namespace smallest_y_value_l73_73486

-- Define the original equation
def original_eq (y : ℝ) := 3 * y^2 + 36 * y - 90 = y * (y + 18)

-- Define the problem statement
theorem smallest_y_value : ∃ (y : ℝ), original_eq y ∧ y = -15 :=
by
  sorry

end smallest_y_value_l73_73486


namespace square_perimeter_l73_73811

noncomputable def side_length_of_square_with_area (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def perimeter_of_square_with_side (side : ℝ) : ℝ :=
  4 * side

theorem square_perimeter {area : ℝ} (h_area : area = 625) :
  perimeter_of_square_with_side (side_length_of_square_with_area area) = 100 :=
by
  have h_side_length : side_length_of_square_with_area area = 25 := by
    rw [side_length_of_square_with_area, real.sqrt, h_area]
    norm_num
  rw [perimeter_of_square_with_side, h_side_length]
  norm_num
  sorry

end square_perimeter_l73_73811


namespace sum_of_largest_odd_divisors_l73_73154

def largestOddDivisor (n : ℕ) : ℕ :=
  n / (2^ (n.totient 2))  -- equivalent to continually dividing by 2 until odd

theorem sum_of_largest_odd_divisors :
  (∑ k in (finset.range (220) \ finset.range (111)), largestOddDivisor k) = 12045 := by
  sorry

end sum_of_largest_odd_divisors_l73_73154


namespace inequality_proof_l73_73120

-- Let x and y be real numbers such that x > y
variables {x y : ℝ} (hx : x > y)

-- We need to prove -2x < -2y
theorem inequality_proof (hx : x > y) : -2 * x < -2 * y :=
sorry

end inequality_proof_l73_73120


namespace solve_equation_l73_73425

-- Defining the original equation as a Lean function
def equation (x : ℝ) : Prop :=
  (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2))

theorem solve_equation :
  ∃ x : ℝ, equation x ∧ x = -13 / 2 :=
by
  -- Equation specification and transformations
  sorry

end solve_equation_l73_73425


namespace max_members_club_l73_73089

open Finset

theorem max_members_club (A B C : Finset ℕ) 
  (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11) 
  (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
  (A ∪ B ∪ C).card ≥ 22 :=
  sorry

end max_members_club_l73_73089


namespace max_members_club_l73_73088

open Finset

theorem max_members_club (A B C : Finset ℕ) 
  (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11) 
  (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
  (A ∪ B ∪ C).card ≥ 22 :=
  sorry

end max_members_club_l73_73088


namespace solve_congruence_l73_73239

theorem solve_congruence (n : ℤ) (h1 : 6 ∣ (n - 4)) (h2 : 10 ∣ (n - 8)) : n ≡ -2 [MOD 30] :=
sorry

end solve_congruence_l73_73239


namespace find_m_l73_73363

-- Definitions of the given vectors a, b, and c
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c (m : ℝ) : ℝ × ℝ := (m, 3)

-- Definition of vector addition and subtraction
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Parallel vectors condition: the ratio of their components must be equal
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- The main theorem stating the desired result
theorem find_m (m : ℝ) :
  parallel (vec_add (vec_a m) (vec_c m)) (vec_sub (vec_a m) vec_b) ↔ 
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 :=
by
  sorry

end find_m_l73_73363


namespace minimum_value_of_quadratic_function_l73_73053

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2

theorem minimum_value_of_quadratic_function :
  ∃ m : ℝ, (∀ x : ℝ, quadratic_function x ≥ m) ∧ (∀ ε > 0, ∃ x : ℝ, quadratic_function x < m + ε) ∧ m = 2 :=
by
  sorry

end minimum_value_of_quadratic_function_l73_73053


namespace least_value_b_l73_73148

-- Defining the conditions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

variables (a b c : ℕ)

-- Conditions
axiom angle_sum : a + b + c = 180
axiom primes : is_prime a ∧ is_prime b ∧ is_prime c
axiom order : a > b ∧ b > c

-- The statement to be proved
theorem least_value_b (h : a + b + c = 180) (hp : is_prime a ∧ is_prime b ∧ is_prime c) (ho : a > b ∧ b > c) : b = 5 :=
sorry

end least_value_b_l73_73148


namespace baseball_card_value_decrease_l73_73490

theorem baseball_card_value_decrease (initial_value : ℝ) :
  (1 - 0.70 * 0.90) * 100 = 37 := 
by sorry

end baseball_card_value_decrease_l73_73490


namespace product_of_repeating_decimal_l73_73664

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l73_73664


namespace pair_students_l73_73563

theorem pair_students :
  (∃ (S5 S4 S3 : Finset ℕ), S5.card = 6 ∧ S4.card = 7 ∧ S3.card = 1 ∧
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 ∈ S5 ∧ p.2 ∈ S4) ∨ (p.1 ∈ S4 ∧ p.2 ∈ S3)) ∧ 
    pairs.card = 7) ∧ 
  ∃ (ways : ℕ), ways = 7 * 720 * 1) :=
exists.intro (Finset.range 6)
(exists.intro (Finset.range 7)
(exists.intro (Finset.singleton 3)
(and.intro (Finset.card_range 6)
(and.intro (Finset.card_range 7)
(and.intro (Finset.card_singleton 3)
(exists.intro 
  ((Finset.range 7).product (Finset.range 6) ∪ (Finset.singleton 3).product (Finset.range 1))
(and.intro 
  (λ p h,
  ((Finset.mem_product.1 (Finset.mem_union.1 h)).2.1 →
    or.inl ⟨Finset.range 6, Finset.range 7⟩) 
  // Similarly for the other combination
  ) ∧ sorry) ∧ sorry)))) sorry.

end pair_students_l73_73563


namespace milton_books_l73_73746

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l73_73746


namespace factorization_solution_l73_73766

def factorization_problem : Prop :=
  ∃ (a b c : ℤ), (∀ (x : ℤ), x^2 + 17 * x + 70 = (x + a) * (x + b)) ∧ 
                 (∀ (x : ℤ), x^2 - 18 * x + 80 = (x - b) * (x - c)) ∧ 
                 (a + b + c = 28)

theorem factorization_solution : factorization_problem :=
sorry

end factorization_solution_l73_73766


namespace sin_cos_value_l73_73871

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l73_73871


namespace hyperbola_equation_l73_73858

theorem hyperbola_equation {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)
    (hfocal : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5)
    (hslope : b / a = 1 / 8) :
    (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  -- Goals and conditions to handle proof
  sorry

end hyperbola_equation_l73_73858


namespace triangle_external_angle_properties_l73_73887

theorem triangle_external_angle_properties (A B C : ℝ) (hA : 0 < A ∧ A < 180) (hB : 0 < B ∧ B < 180) (hC : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) :
  (∃ E1 E2 E3, E1 + E2 + E3 = 360 ∧ E1 > 90 ∧ E2 > 90 ∧ E3 <= 90) :=
by
  sorry

end triangle_external_angle_properties_l73_73887


namespace tire_price_l73_73206

-- Definitions based on given conditions
def tire_cost (T : ℝ) (n : ℕ) : Prop :=
  n * T + 56 = 224

-- The equivalence we want to prove
theorem tire_price (T : ℝ) (n : ℕ) (h : tire_cost T n) : n * T = 168 :=
by
  sorry

end tire_price_l73_73206


namespace travel_time_tripled_l73_73116

variable {S v v_r : ℝ}

-- Conditions of the problem
def condition1 (t1 t2 : ℝ) : Prop :=
  t1 = 3 * t2

def condition2 (t1 t2 : ℝ) : Prop :=
  t1 = S / (v + v_r) ∧ t2 = S / (v - v_r)

def stationary_solution : Prop :=
  v = 2 * v_r

-- Conclusion: Time taken to travel from B to A without paddles is 3 times longer than usual
theorem travel_time_tripled (t_no_paddle t2 : ℝ) (h1 : condition1 t_no_paddle t2) (h2 : condition2 t_no_paddle t2) (h3 : stationary_solution) :
  t_no_paddle = 3 * t2 :=
sorry

end travel_time_tripled_l73_73116


namespace red_car_initial_distance_ahead_l73_73313

theorem red_car_initial_distance_ahead 
    (Speed_red Speed_black : ℕ) (Time : ℝ)
    (H1 : Speed_red = 10)
    (H2 : Speed_black = 50)
    (H3 : Time = 0.5) :
    let Distance_black := Speed_black * Time
    let Distance_red := Speed_red * Time
    Distance_black - Distance_red = 20 := 
by
  let Distance_black := Speed_black * Time
  let Distance_red := Speed_red * Time
  sorry

end red_car_initial_distance_ahead_l73_73313


namespace water_speed_l73_73646

theorem water_speed (swim_speed : ℝ) (time : ℝ) (distance : ℝ) (v : ℝ) 
  (h1: swim_speed = 10) (h2: time = 2) (h3: distance = 12) 
  (h4: distance = (swim_speed - v) * time) : 
  v = 4 :=
by
  sorry

end water_speed_l73_73646


namespace quadratic_function_analysis_l73_73373

theorem quadratic_function_analysis (a b c : ℝ) :
  (a - b + c = -1) →
  (c = 2) →
  (4 * a + 2 * b + c = 2) →
  (16 * a + 4 * b + c = -6) →
  (¬ ∃ x > 3, a * x^2 + b * x + c = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end quadratic_function_analysis_l73_73373


namespace lcm_9_12_15_l73_73956

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l73_73956


namespace solve_for_A_l73_73714

def clubsuit (A B : ℤ) : ℤ := 3 * A + 2 * B + 7

theorem solve_for_A (A : ℤ) : (clubsuit A 6 = 70) -> (A = 17) :=
by
  sorry

end solve_for_A_l73_73714


namespace lcm_9_12_15_l73_73978

-- Defining the numbers
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Defining the function to find the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Proving that the LCM of 9, 12, and 15 is 180
theorem lcm_9_12_15 : lcm a (lcm b c) = 180 := by
  -- Placeholder for the proof
  sorry

end lcm_9_12_15_l73_73978


namespace evaluate_expression_l73_73107

theorem evaluate_expression : 
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = (137 / 52) :=
by
  -- We need to evaluate from the innermost part to the outermost,
  -- as noted in the problem statement and solution steps.
  sorry

end evaluate_expression_l73_73107


namespace product_lcm_gcd_eq_product_original_numbers_l73_73657

theorem product_lcm_gcd_eq_product_original_numbers :
  let a := 12
  let b := 18
  (Int.gcd a b) * (Int.lcm a b) = a * b :=
by
  sorry

end product_lcm_gcd_eq_product_original_numbers_l73_73657


namespace valentine_cards_l73_73290

theorem valentine_cards (x y : ℕ) (h : x * y = x + y + 18) : x * y = 40 :=
by
  sorry

end valentine_cards_l73_73290


namespace shop_owner_percentage_profit_l73_73212

theorem shop_owner_percentage_profit :
  let cost_price_per_kg := 100
  let buy_cheat_percent := 18.5 / 100
  let sell_cheat_percent := 22.3 / 100
  let amount_bought := 1 / (1 + buy_cheat_percent)
  let amount_sold := 1 - sell_cheat_percent
  let effective_cost_price := cost_price_per_kg * amount_sold / amount_bought
  let selling_price := cost_price_per_kg
  let profit := selling_price - effective_cost_price
  let percentage_profit := (profit / effective_cost_price) * 100
  percentage_profit = 52.52 :=
by
  sorry

end shop_owner_percentage_profit_l73_73212


namespace prob_A_hits_B_misses_prob_equal_hits_after_two_shots_l73_73169

open ProbabilityTheory

-- Definitions for conditions
def prob_A : ℚ := 3 / 4
def prob_B : ℚ := 4 / 5

-- Theorem statement for Part (I)
theorem prob_A_hits_B_misses :
  prob_A * (1 - prob_B) = 3 / 20 :=
by
  sorry

-- Theorem statement for Part (II)
theorem prob_equal_hits_after_two_shots :
  (prob_A^2 * prob_B^2 + 2 * (prob_A * (1 - prob_A)) * (prob_B * (1 - prob_B)) + (1 - prob_A)^2 * (1 - prob_B)^2) =
  193 / 400 :=
by
  sorry

end prob_A_hits_B_misses_prob_equal_hits_after_two_shots_l73_73169


namespace lcm_9_12_15_l73_73969

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l73_73969


namespace racket_price_l73_73279

theorem racket_price (cost_sneakers : ℕ) (cost_outfit : ℕ) (total_spent : ℕ) 
  (h_sneakers : cost_sneakers = 200) 
  (h_outfit : cost_outfit = 250) 
  (h_total : total_spent = 750) : 
  (total_spent - cost_sneakers - cost_outfit) = 300 :=
sorry

end racket_price_l73_73279


namespace least_possible_value_of_z_l73_73699

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end least_possible_value_of_z_l73_73699


namespace exp_add_l73_73292

theorem exp_add (a : ℝ) (x₁ x₂ : ℝ) : a^(x₁ + x₂) = a^x₁ * a^x₂ :=
sorry

end exp_add_l73_73292


namespace meaningful_expression_range_l73_73864

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = (1 / (Real.sqrt (x - 2)))) ↔ (x > 2) := 
sorry

end meaningful_expression_range_l73_73864


namespace inequality_proof_l73_73907

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
    (((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2) ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l73_73907


namespace trig_relationship_l73_73690

noncomputable def a := Real.cos 1
noncomputable def b := Real.cos 2
noncomputable def c := Real.sin 2

theorem trig_relationship : c > a ∧ a > b := by
  sorry

end trig_relationship_l73_73690


namespace cricket_matches_total_l73_73297

theorem cricket_matches_total
  (n : ℕ)
  (avg_all : ℝ)
  (avg_first4 : ℝ)
  (avg_last3 : ℝ)
  (h_avg_all : avg_all = 56)
  (h_avg_first4 : avg_first4 = 46)
  (h_avg_last3 : avg_last3 = 69.33333333333333)
  (h_total_runs : n * avg_all = 4 * avg_first4 + 3 * avg_last3) :
  n = 7 :=
by
  sorry

end cricket_matches_total_l73_73297


namespace problem1_problem2_l73_73660

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end problem1_problem2_l73_73660


namespace polygon_sides_l73_73997

theorem polygon_sides (interior_angle: ℝ) (sum_exterior_angles: ℝ) (n: ℕ) (h: interior_angle = 108) (h1: sum_exterior_angles = 360): n = 5 :=
by 
  sorry

end polygon_sides_l73_73997


namespace sufficient_but_not_necessary_l73_73119

noncomputable def problem_statement (a : ℝ) : Prop :=
(a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2)

theorem sufficient_but_not_necessary (a : ℝ) : problem_statement a := 
sorry

end sufficient_but_not_necessary_l73_73119


namespace fencing_required_l73_73623

theorem fencing_required (L W : ℕ) (area : ℕ) (hL : L = 20) (hA : area = 120) (hW : area = L * W) :
  2 * W + L = 32 :=
by
  -- Steps and proof logic to be provided here
  sorry

end fencing_required_l73_73623


namespace total_turtles_in_lake_l73_73446

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l73_73446


namespace number_of_ways_l73_73460

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l73_73460


namespace small_triangle_perimeter_l73_73903

theorem small_triangle_perimeter (P : ℕ) (P₁ : ℕ) (P₂ : ℕ) (P₃ : ℕ)
  (h₁ : P = 11) (h₂ : P₁ = 5) (h₃ : P₂ = 7) (h₄ : P₃ = 9) :
  (P₁ + P₂ + P₃) - P = 10 :=
by
  sorry

end small_triangle_perimeter_l73_73903


namespace no_adjacent_stand_up_probability_l73_73170

noncomputable def coin_flip_prob_adjacent_people_stand_up : ℚ :=
  123 / 1024

theorem no_adjacent_stand_up_probability :
  let num_people := 10
  let total_outcomes := 2^num_people
  (123 : ℚ) / total_outcomes = coin_flip_prob_adjacent_people_stand_up :=
by
  sorry

end no_adjacent_stand_up_probability_l73_73170


namespace dan_present_age_l73_73104

-- Let x be Dan's present age
variable (x : ℤ)

-- Condition: Dan's age after 18 years will be 8 times his age 3 years ago
def condition (x : ℤ) : Prop :=
  x + 18 = 8 * (x - 3)

-- The goal is to prove that Dan's present age is 6
theorem dan_present_age (x : ℤ) (h : condition x) : x = 6 :=
by
  sorry

end dan_present_age_l73_73104


namespace study_days_needed_l73_73517

theorem study_days_needed :
  let math_chapters := 4
  let math_worksheets := 7
  let physics_chapters := 5
  let physics_worksheets := 9
  let chemistry_chapters := 6
  let chemistry_worksheets := 8

  let math_chapter_hours := 2.5
  let math_worksheet_hours := 1.5
  let physics_chapter_hours := 3.0
  let physics_worksheet_hours := 2.0
  let chemistry_chapter_hours := 3.5
  let chemistry_worksheet_hours := 1.75

  let daily_study_hours := 7.0
  let breaks_first_3_hours := 3 * 10 / 60.0
  let breaks_next_3_hours := 3 * 15 / 60.0
  let breaks_final_hour := 1 * 20 / 60.0
  let snack_breaks := 2 * 20 / 60.0
  let lunch_break := 45 / 60.0

  let break_time_per_day := breaks_first_3_hours + breaks_next_3_hours + breaks_final_hour + snack_breaks + lunch_break
  let effective_study_time_per_day := daily_study_hours - break_time_per_day

  let total_math_hours := (math_chapters * math_chapter_hours) + (math_worksheets * math_worksheet_hours)
  let total_physics_hours := (physics_chapters * physics_chapter_hours) + (physics_worksheets * physics_worksheet_hours)
  let total_chemistry_hours := (chemistry_chapters * chemistry_chapter_hours) + (chemistry_worksheets * chemistry_worksheet_hours)

  let total_study_hours := total_math_hours + total_physics_hours + total_chemistry_hours
  let total_study_days := total_study_hours / effective_study_time_per_day
  
  total_study_days.ceil = 23 := by sorry

end study_days_needed_l73_73517


namespace reciprocal_of_neg_two_l73_73057

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end reciprocal_of_neg_two_l73_73057


namespace sin_double_angle_second_quadrant_l73_73125

theorem sin_double_angle_second_quadrant (α : ℝ) (h1 : Real.cos α = -3/5) (h2 : α ∈ Set.Ioo (π / 2) π) :
    Real.sin (2 * α) = -24 / 25 := by
  sorry

end sin_double_angle_second_quadrant_l73_73125


namespace sally_initial_orange_balloons_l73_73586

variable (initial_orange_balloons : ℕ)  -- The initial number of orange balloons Sally had
variable (lost_orange_balloons : ℕ := 2)  -- The number of orange balloons Sally lost
variable (current_orange_balloons : ℕ := 7)  -- The number of orange balloons Sally currently has

theorem sally_initial_orange_balloons : 
  current_orange_balloons + lost_orange_balloons = initial_orange_balloons := 
by
  sorry

end sally_initial_orange_balloons_l73_73586


namespace jackson_weekly_mileage_increase_l73_73736

theorem jackson_weekly_mileage_increase :
  ∃ (weeks : ℕ), weeks = (7 - 3) / 1 := by
  sorry

end jackson_weekly_mileage_increase_l73_73736


namespace proof_problem_l73_73004

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end proof_problem_l73_73004


namespace speed_of_current_l73_73626
  
  theorem speed_of_current (v c : ℝ)
    (h1 : 64 = (v + c) * 8)
    (h2 : 24 = (v - c) * 8) :
    c = 2.5 :=
  by {
    sorry
  }
  
end speed_of_current_l73_73626


namespace right_triangle_ineq_l73_73725

variable (a b c : ℝ)
variable (h : c^2 = a^2 + b^2)

theorem right_triangle_ineq (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 :=
by
  sorry

end right_triangle_ineq_l73_73725


namespace initial_amount_l73_73208

theorem initial_amount 
  (M : ℝ)
  (h1 : M * (3 / 5) * (2 / 3) * (3 / 4) * (4 / 7) = 700) : 
  M = 24500 / 6 :=
by sorry

end initial_amount_l73_73208


namespace simplify_144_over_1296_times_36_l73_73424

theorem simplify_144_over_1296_times_36 :
  (144 / 1296) * 36 = 4 :=
by
  sorry

end simplify_144_over_1296_times_36_l73_73424


namespace sum_xyz_l73_73138

variables (x y z : ℤ)

theorem sum_xyz (h1 : y = 3 * x) (h2 : z = 3 * y - x) : x + y + z = 12 * x :=
by 
  -- skip the proof
  sorry

end sum_xyz_l73_73138


namespace gcd_lcm_sum_eq_l73_73783

-- Define the two numbers
def a : ℕ := 72
def b : ℕ := 8712

-- Define the GCD and LCM functions.
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Define the sum of the GCD and LCM.
def sum_gcd_lcm : ℕ := gcd_ab + lcm_ab

-- The theorem we want to prove
theorem gcd_lcm_sum_eq : sum_gcd_lcm = 26160 := by
  -- Details of the proof would go here
  sorry

end gcd_lcm_sum_eq_l73_73783


namespace sixth_group_points_l73_73694

-- Definitions of conditions
def total_data_points : ℕ := 40

def group1_points : ℕ := 10
def group2_points : ℕ := 5
def group3_points : ℕ := 7
def group4_points : ℕ := 6
def group5_frequency : ℝ := 0.10

def group5_points : ℕ := (group5_frequency * total_data_points).toInt

-- Theorem: The number of data points in the sixth group
theorem sixth_group_points :
  group1_points + group2_points + group3_points + group4_points + group5_points + x = total_data_points →
  x = 8 :=
by
  sorry

end sixth_group_points_l73_73694


namespace debate_schedule_ways_l73_73340

-- Definitions based on the problem conditions
def east_debaters : Fin 4 := 4
def west_debaters : Fin 4 := 4
def total_debates := east_debaters.val * west_debaters.val
def debates_per_session := 3
def sessions := 5
def rest_debates := total_debates - sessions * debates_per_session

-- Claim that the number of scheduling ways is the given number
theorem debate_schedule_ways : (Nat.factorial total_debates) / ((Nat.factorial debates_per_session) ^ sessions * Nat.factorial rest_debates) = 20922789888000 :=
by
  -- Proof is skipped with sorry
  sorry

end debate_schedule_ways_l73_73340


namespace vector_addition_example_l73_73663

theorem vector_addition_example : 
  let v1 := (⟨-5, 3⟩ : ℝ × ℝ)
  let v2 := (⟨7, -6⟩ : ℝ × ℝ)
  v1 + v2 = (⟨2, -3⟩ : ℝ × ℝ) := 
by {
  sorry
}

end vector_addition_example_l73_73663


namespace cube_sum_identity_l73_73616

theorem cube_sum_identity (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by 
 sorry

end cube_sum_identity_l73_73616


namespace intersection_points_count_l73_73678

theorem intersection_points_count:
  let line1 := { p : ℝ × ℝ | ∃ x y : ℝ, 4 * y - 3 * x = 2 ∧ (p.1 = x ∧ p.2 = y) }
  let line2 := { p : ℝ × ℝ | ∃ x y : ℝ, x + 3 * y = 3 ∧ (p.1 = x ∧ p.2 = y) }
  let line3 := { p : ℝ × ℝ | ∃ x y : ℝ, 6 * x - 8 * y = 6 ∧ (p.1 = x ∧ p.2 = y) }
  ∃! p1 p2 : ℝ × ℝ, p1 ∈ line1 ∧ p1 ∈ line2 ∧ p2 ∈ line2 ∧ p2 ∈ line3 :=
by
  sorry

end intersection_points_count_l73_73678


namespace fraction_equals_one_l73_73081

/-- Given the fraction (12-11+10-9+8-7+6-5+4-3+2-1) / (1-2+3-4+5-6+7-8+9-10+11),
    prove that its value is equal to 1. -/
theorem fraction_equals_one :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end fraction_equals_one_l73_73081


namespace quadratic_root_sqrt_2010_2009_l73_73836

theorem quadratic_root_sqrt_2010_2009 :
  (∃ (a b : ℤ), a = 0 ∧ b = -(2010 + 2 * Real.sqrt 2009) ∧
  ∀ (x : ℝ), x^2 + (a : ℝ) * x + (b : ℝ) = 0 → x = Real.sqrt (2010 + 2 * Real.sqrt 2009) ∨ x = -Real.sqrt (2010 + 2 * Real.sqrt 2009)) :=
sorry

end quadratic_root_sqrt_2010_2009_l73_73836


namespace segment_area_formula_l73_73016
noncomputable def area_of_segment (r a : ℝ) : ℝ :=
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2)

theorem segment_area_formula (r a : ℝ) : area_of_segment r a =
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2) :=
sorry

end segment_area_formula_l73_73016


namespace sufficient_condition_l73_73112

variable (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, -1 ≤ x → x ≤ 2 → x^2 - a ≥ 0) : a ≤ -1 := 
sorry

end sufficient_condition_l73_73112


namespace max_product_l73_73207

def geometric_sequence (a1 q : ℝ) (n : ℕ) :=
  a1 * q ^ (n - 1)

def product_of_terms (a1 q : ℝ) (n : ℕ) :=
  (List.range n).foldr (λ i acc => acc * geometric_sequence a1 q (i + 1)) 1

theorem max_product (n : ℕ) (a1 q : ℝ) (h₁ : a1 = 1536) (h₂ : q = -1/2) :
  n = 11 ↔ ∀ m : ℕ, m ≤ 11 → product_of_terms a1 q m ≤ product_of_terms a1 q 11 :=
by
  sorry

end max_product_l73_73207


namespace find_sum_12_terms_of_sequence_l73_73886

variable {a : ℕ → ℕ}

def geometric_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

def is_periodic_sequence (a : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n : ℕ, a n = a (n + period)

noncomputable def given_sequence : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => (given_sequence n * given_sequence (n + 1) / 4) -- This should ensure periodic sequence of period 3 given a common product of 8 and simplifying the product equation.

theorem find_sum_12_terms_of_sequence :
  geometric_sequence given_sequence 8 ∧ given_sequence 0 = 1 ∧ given_sequence 1 = 2 →
  (Finset.range 12).sum given_sequence = 28 :=
by
  sorry

end find_sum_12_terms_of_sequence_l73_73886


namespace otherWorkStations_accommodate_students_l73_73504

def numTotalStudents := 38
def numStations := 16
def numWorkStationsForTwo := 10
def capacityWorkStationsForTwo := 2

theorem otherWorkStations_accommodate_students : 
  (numTotalStudents - numWorkStationsForTwo * capacityWorkStationsForTwo) = 18 := 
by
  sorry

end otherWorkStations_accommodate_students_l73_73504


namespace possible_values_x_l73_73499

variable (a b x : ℕ)

theorem possible_values_x (h1 : a + b = 20)
                          (h2 : a * x + b * 3 = 109) :
    x = 10 ∨ x = 52 :=
sorry

end possible_values_x_l73_73499


namespace amoebas_after_ten_days_l73_73818

def amoeba_split_fun (n : Nat) : Nat := 3^n

theorem amoebas_after_ten_days : amoeba_split_fun 10 = 59049 := by
  have h : 3 ^ 10 = 59049 := by norm_num
  exact h

end amoebas_after_ten_days_l73_73818


namespace original_percentage_of_acid_l73_73509

theorem original_percentage_of_acid 
  (a w : ℝ) 
  (h1 : a + w = 6) 
  (h2 : a / (a + w + 2) = 15 / 100) 
  (h3 : (a + 2) / (a + w + 4) = 25 / 100) :
  (a / 6) * 100 = 20 :=
  sorry

end original_percentage_of_acid_l73_73509


namespace find_a3_l73_73266

theorem find_a3 (a0 a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, x^4 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4) →
  a3 = -8 :=
by
  sorry

end find_a3_l73_73266


namespace arithmetic_sequence_tenth_term_l73_73051

theorem arithmetic_sequence_tenth_term (a d : ℤ) (h₁ : a + 3 * d = 23) (h₂ : a + 8 * d = 38) : a + 9 * d = 41 := by
  sorry

end arithmetic_sequence_tenth_term_l73_73051


namespace square_side_length_false_l73_73302

theorem square_side_length_false (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 8) (h2 : side_length = 4) :
  ¬(4 * side_length = perimeter) :=
by
  sorry

end square_side_length_false_l73_73302


namespace tanya_addition_problem_l73_73407

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l73_73407


namespace find_the_number_l73_73556

theorem find_the_number (n : ℤ) 
    (h : 45 - (28 - (n - (15 - 18))) = 57) :
    n = 37 := 
sorry

end find_the_number_l73_73556


namespace find_a_l73_73543

theorem find_a (a : ℝ) (k_l : ℝ) (h1 : k_l = -1)
  (h2 : a ≠ 3) 
  (h3 : (2 - (-1)) / (3 - a) * k_l = -1) : a = 6 :=
by
  sorry

end find_a_l73_73543


namespace cube_increasing_on_reals_l73_73245

theorem cube_increasing_on_reals (a b : ℝ) (h : a < b) : a^3 < b^3 :=
sorry

end cube_increasing_on_reals_l73_73245


namespace unique_primes_solution_l73_73352

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_primes_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  p^3 - q^5 = (p + q)^2 ↔ (p = 7 ∧ q = 3) :=
by
  sorry

end unique_primes_solution_l73_73352


namespace Kyle_monthly_income_l73_73280

theorem Kyle_monthly_income :
  let rent := 1250
  let utilities := 150
  let retirement_savings := 400
  let groceries_eatingout := 300
  let insurance := 200
  let miscellaneous := 200
  let car_payment := 350
  let gas_maintenance := 350
  rent + utilities + retirement_savings + groceries_eatingout + insurance + miscellaneous + car_payment + gas_maintenance = 3200 :=
by
  -- Informal proof was provided in the solution.
  sorry

end Kyle_monthly_income_l73_73280


namespace problem_1_l73_73203

theorem problem_1 (a : ℝ) : (1 + a * x) * (1 + x) ^ 5 = 1 + 5 * x + 5 * i * x^2 → a = -1 := sorry

end problem_1_l73_73203


namespace three_digit_addition_l73_73065

theorem three_digit_addition (a b : ℕ) (h₁ : 307 = 300 + a * 10 + 7) (h₂ : 416 + 10 * (a * 1) + 7 = 700 + b * 10 + 3) (h₃ : (7 + b + 3) % 3 = 0) : a + b = 2 :=
by
  -- mock proof, since solution steps are not considered
  sorry

end three_digit_addition_l73_73065


namespace canonical_line_eq_l73_73785

-- Define the system of linear equations
def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x - 3 * y - 2 * z + 6 = 0 ∧ x - 3 * y + z + 3 = 0)

-- Define the canonical equation of the line
def canonical_equation (x y z : ℝ) : Prop :=
  (x + 3) / 9 = y / 4 ∧ (x + 3) / 9 = z / 3 ∧ y / 4 = z / 3

-- The theorem to prove equivalence
theorem canonical_line_eq : 
  ∀ (x y z : ℝ), system_of_equations x y z → canonical_equation x y z :=
by
  intros x y z H
  sorry

end canonical_line_eq_l73_73785


namespace points_opposite_side_of_line_l73_73879

theorem points_opposite_side_of_line :
  (∀ a : ℝ, ((2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0) ↔ -1 < a ∧ a < 1) :=
by sorry

end points_opposite_side_of_line_l73_73879


namespace average_grade_of_female_students_is_92_l73_73764

noncomputable def female_average_grade 
  (overall_avg : ℝ) (male_avg : ℝ) (num_males : ℕ) (num_females : ℕ) : ℝ :=
  let total_students := num_males + num_females
  let total_score := total_students * overall_avg
  let male_total_score := num_males * male_avg
  let female_total_score := total_score - male_total_score
  female_total_score / num_females

theorem average_grade_of_female_students_is_92 :
  female_average_grade 90 83 8 28 = 92 := 
by
  -- Proof steps to be completed
  sorry

end average_grade_of_female_students_is_92_l73_73764


namespace unreachable_y_l73_73346

noncomputable def y_function (x : ℝ) : ℝ := (2 - 3 * x) / (5 * x - 1)

theorem unreachable_y : ¬ ∃ x : ℝ, y_function x = -3 / 5 ∧ x ≠ 1 / 5 :=
by {
  sorry
}

end unreachable_y_l73_73346


namespace distance_between_planes_is_zero_l73_73238

def plane1 (x y z : ℝ) : Prop := x - 2 * y + 2 * z = 9
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y + 4 * z = 18

theorem distance_between_planes_is_zero :
  (∀ x y z : ℝ, plane1 x y z ↔ plane2 x y z) → 0 = 0 :=
by
  sorry

end distance_between_planes_is_zero_l73_73238


namespace line_equation_l73_73429

theorem line_equation (t : ℝ) : 
  ∃ m b, (∀ x y : ℝ, (x, y) = (3 * t + 6, 5 * t - 7) → y = m * x + b) ∧
  m = 5 / 3 ∧ b = -17 :=
by
  use 5 / 3, -17
  sorry

end line_equation_l73_73429


namespace number_of_students_l73_73317

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N) (h2 : (T - 250) / (N - 5) = 90) : N = 20 :=
sorry

end number_of_students_l73_73317


namespace distance_between_bus_stops_l73_73174

theorem distance_between_bus_stops (d : ℕ) (unit : String) 
  (h: d = 3000 ∧ unit = "meters") : unit = "C" := 
by 
  sorry

end distance_between_bus_stops_l73_73174


namespace triple_integral_value_l73_73522

theorem triple_integral_value :
  (∫ x in (-1 : ℝ)..1, ∫ y in (x^2 : ℝ)..1, ∫ z in (0 : ℝ)..y, (4 + z) ) = (16 / 3 : ℝ) :=
by
  sorry

end triple_integral_value_l73_73522


namespace sum_of_digits_B_equals_4_l73_73413

theorem sum_of_digits_B_equals_4 (A B : ℕ) (N : ℕ) (hN : N = 4444 ^ 4444)
    (hA : A = (N.digits 10).sum) (hB : B = (A.digits 10).sum) :
    (B.digits 10).sum = 4 := by
  sorry

end sum_of_digits_B_equals_4_l73_73413


namespace total_hours_worked_l73_73320

def hours_per_day : ℕ := 8 -- Frank worked 8 hours on each day
def number_of_days : ℕ := 4 -- First 4 days of the week

theorem total_hours_worked : hours_per_day * number_of_days = 32 := by
  sorry

end total_hours_worked_l73_73320


namespace x_plus_y_value_l73_73244

theorem x_plus_y_value (x y : ℕ) (h1 : 2^x = 8^(y + 1)) (h2 : 9^y = 3^(x - 9)) : x + y = 27 :=
by
  sorry

end x_plus_y_value_l73_73244


namespace arithmetic_geometric_seq_sum_5_l73_73248

-- Define the arithmetic-geometric sequence a_n
def a (n : ℕ) : ℤ := sorry

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℤ := sorry

-- Condition: a_1 = 1
axiom a1 : a 1 = 1

-- Condition: a_{n+2} + a_{n+1} - 2 * a_{n} = 0 for all n ∈ ℕ_+
axiom recurrence (n : ℕ) : a (n + 2) + a (n + 1) - 2 * a n = 0

-- Prove that S_5 = 11
theorem arithmetic_geometric_seq_sum_5 : S 5 = 11 := 
by
  sorry

end arithmetic_geometric_seq_sum_5_l73_73248


namespace find_number_l73_73717

-- Define the main condition and theorem.
theorem find_number (x : ℤ) : 45 - (x - (37 - (15 - 19))) = 58 ↔ x = 28 :=
by
  sorry  -- placeholder for the proof

end find_number_l73_73717


namespace area_of_quadrilateral_ABCD_l73_73890

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem area_of_quadrilateral_ABCD :
  let AB := 15 * sqrt 2
  let BE := 15 * sqrt 2
  let BC := 7.5 * sqrt 2
  let CE := 7.5 * sqrt 6
  let CD := 7.5 * sqrt 2
  let DE := 7.5 * sqrt 6
  (1/2 * AB * BE) + (1/2 * BC * CE) + (1/2 * CD * DE) = 225 + 112.5 * sqrt 12 :=
by
  sorry

end area_of_quadrilateral_ABCD_l73_73890


namespace arc_length_polar_l73_73101

theorem arc_length_polar :
  (∫ φ in (0:Real)..(π/3), 
   sqrt ((5 * exp (5 * φ / 12))^2 + (deriv (λ φ, 5 * exp (5 * φ / 12)) φ)^2)) 
  = 13 * (exp (5 * π / 36) - 1) :=
by
  sorry

end arc_length_polar_l73_73101


namespace find_smallest_n_l73_73233

-- Define costs and relationships
def cost_red (r : ℕ) : ℕ := 10 * r
def cost_green (g : ℕ) : ℕ := 18 * g
def cost_blue (b : ℕ) : ℕ := 20 * b
def cost_purple (n : ℕ) : ℕ := 24 * n

-- Define the mathematical problem
theorem find_smallest_n (r g b : ℕ) :
  ∃ n : ℕ, 24 * n = Nat.lcm (cost_red r) (Nat.lcm (cost_green g) (cost_blue b)) ∧ n = 15 :=
by
  sorry

end find_smallest_n_l73_73233


namespace factor_by_resultant_is_three_l73_73642

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end factor_by_resultant_is_three_l73_73642


namespace nat_pairs_solution_l73_73226

theorem nat_pairs_solution (x y : ℕ) :
  2^(2*x+1) + 2^x + 1 = y^2 → (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by
  sorry

end nat_pairs_solution_l73_73226


namespace sum_geom_seq_l73_73371

theorem sum_geom_seq (S : ℕ → ℝ) (a_n : ℕ → ℝ) (h1 : S 4 ≠ 0) 
  (h2 : S 8 / S 4 = 4) 
  (h3 : ∀ n : ℕ, S n = a_n 0 * (1 - (a_n 1 / a_n 0)^n) / (1 - a_n 1 / a_n 0)) :
  S 12 / S 4 = 13 :=
sorry

end sum_geom_seq_l73_73371


namespace B_finishes_work_in_4_days_l73_73198

-- Define the work rates of A and B
def work_rate_A : ℚ := 1 / 5
def work_rate_B : ℚ := 1 / 10

-- Combined work rate when A and B work together
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Work done by A and B in 2 days
def work_done_in_2_days : ℚ := 2 * combined_work_rate

-- Remaining work after 2 days
def remaining_work : ℚ := 1 - work_done_in_2_days

-- Time B needs to finish the remaining work
def time_for_B_to_finish_remaining_work : ℚ := remaining_work / work_rate_B

theorem B_finishes_work_in_4_days : time_for_B_to_finish_remaining_work = 4 := by
  sorry

end B_finishes_work_in_4_days_l73_73198


namespace total_number_of_books_ways_to_select_books_l73_73050

def first_layer_books : ℕ := 6
def second_layer_books : ℕ := 5
def third_layer_books : ℕ := 4

theorem total_number_of_books : first_layer_books + second_layer_books + third_layer_books = 15 := by
  sorry

theorem ways_to_select_books : first_layer_books * second_layer_books * third_layer_books = 120 := by
  sorry

end total_number_of_books_ways_to_select_books_l73_73050


namespace remainder_when_dividing_polynomial_by_x_minus_3_l73_73194

noncomputable def P (x : ℤ) : ℤ := 
  2 * x^8 - 3 * x^7 + 4 * x^6 - x^4 + 6 * x^3 - 5 * x^2 + 18 * x - 20

theorem remainder_when_dividing_polynomial_by_x_minus_3 :
  P 3 = 17547 :=
by
  sorry

end remainder_when_dividing_polynomial_by_x_minus_3_l73_73194


namespace remainder_sum_mod_13_l73_73688

theorem remainder_sum_mod_13 : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 :=
by
  sorry

end remainder_sum_mod_13_l73_73688


namespace conic_eccentricity_l73_73719

theorem conic_eccentricity (m : ℝ) (h : 0 < -m) (h2 : (Real.sqrt (1 + (-1 / m))) = 2) : m = -1/3 := 
by
  -- Proof can be added here
  sorry

end conic_eccentricity_l73_73719


namespace increasing_function_a_range_l73_73253

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4 * a * x else (2 * a + 3) * x - 4 * a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end increasing_function_a_range_l73_73253


namespace oldest_bride_age_l73_73599

theorem oldest_bride_age (B G : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) :
  B = 102 :=
by
  sorry

end oldest_bride_age_l73_73599


namespace trajectory_is_eight_rays_l73_73598

open Real

def trajectory_of_point (x y : ℝ) : Prop :=
  abs (abs x - abs y) = 2

theorem trajectory_is_eight_rays :
  ∃ (x y : ℝ), trajectory_of_point x y :=
sorry

end trajectory_is_eight_rays_l73_73598


namespace solution_set_quadratic_l73_73772

-- Define the quadratic equation as a function
def quadratic_eq (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- The theorem to prove
theorem solution_set_quadratic :
  {x : ℝ | quadratic_eq x = 0} = {1, 2} :=
by
  sorry

end solution_set_quadratic_l73_73772


namespace proof_equivalent_problem_l73_73423

-- Definition of conditions
def cost_condition_1 (x y : ℚ) : Prop := 500 * x + 40 * y = 1250
def cost_condition_2 (x y : ℚ) : Prop := 1000 * x + 20 * y = 1000
def budget_condition (a b : ℕ) (total_masks : ℕ) (budget : ℕ) : Prop := 2 * a + (total_masks - a) / 2 + 25 * b = budget

-- Main theorem
theorem proof_equivalent_problem : 
  ∃ (x y : ℚ) (a b : ℕ), 
    cost_condition_1 x y ∧
    cost_condition_2 x y ∧
    (x = 1 / 2) ∧ 
    (y = 25) ∧
    (budget_condition a b 200 400) ∧
    ((a = 150 ∧ b = 3) ∨
     (a = 100 ∧ b = 6) ∨
     (a = 50 ∧ b = 9)) :=
by {
  sorry -- The proof steps are not required
}

end proof_equivalent_problem_l73_73423


namespace ace_then_king_same_suit_probability_l73_73603

theorem ace_then_king_same_suit_probability : 
  let deck_size := 52
  let ace_count := 4
  let king_count := 4
  let same_suit_aces := 1
  let same_suit_kings := 1
  P((draw1 = ace) ∧ (draw2 = king) | (same_suit)) = 1 / 663 :=
by
  sorry

end ace_then_king_same_suit_probability_l73_73603


namespace cos_squared_value_l73_73697

theorem cos_squared_value (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) : 
  Real.cos (π / 3 - x) ^ 2 = 1 / 16 := 
sorry

end cos_squared_value_l73_73697


namespace find_parabola_equation_l73_73110

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
    (y = a * x ^ 2 + b * x + c) ∧ 
    (y = (x - 3) ^ 2 - 2) ∧
    (a * (4 - 3) ^ 2 - 2 = 2)

theorem find_parabola_equation :
  ∃ (a b c : ℝ), parabola_equation a b c ∧ a = 4 ∧ b = -24 ∧ c = 34 :=
sorry

end find_parabola_equation_l73_73110


namespace rectangle_width_is_pi_l73_73511

theorem rectangle_width_is_pi (w : ℝ) (h1 : real_w ≠ 0)
    (h2 : ∀ w, ∃ length, length = 2 * w)
    (h3 : ∀ w, 2 * (length + w) = 6 * w)
    (h4 : 2 * (2 * w + w) = 6 * π) : 
    w = π :=
by {
  sorry -- The proof would go here.
}

end rectangle_width_is_pi_l73_73511


namespace systematic_sampling_methods_l73_73098

-- Definitions for sampling methods ①, ②, ④
def sampling_method_1 : Prop :=
  ∀ (l : ℕ), (l ≤ 15 ∧ l + 5 ≤ 15 ∧ l + 10 ≤ 15 ∨
              l ≤ 15 ∧ l + 5 ≤ 20 ∧ l + 10 ≤ 20) → True

def sampling_method_2 : Prop :=
  ∀ (t : ℕ), (t % 5 = 0) → True

def sampling_method_3 : Prop :=
  ∀ (n : ℕ), (n > 0) → True

def sampling_method_4 : Prop :=
  ∀ (row : ℕ) (seat : ℕ), (seat = 12) → True

-- Equivalence Proof Statement
theorem systematic_sampling_methods :
  sampling_method_1 ∧ sampling_method_2 ∧ sampling_method_4 :=
by sorry

end systematic_sampling_methods_l73_73098


namespace razorback_tshirt_profit_l73_73763

theorem razorback_tshirt_profit :
  let profit_per_tshirt := 9
  let cost_per_tshirt := 4
  let num_tshirts_sold := 245
  let discount := 0.2
  let selling_price := profit_per_tshirt + cost_per_tshirt
  let discount_amount := discount * selling_price
  let discounted_price := selling_price - discount_amount
  let total_revenue := discounted_price * num_tshirts_sold
  let total_production_cost := cost_per_tshirt * num_tshirts_sold
  let total_profit := total_revenue - total_production_cost
  total_profit = 1568 :=
by
  sorry

end razorback_tshirt_profit_l73_73763


namespace solution_set_of_inequality_l73_73305

theorem solution_set_of_inequality (x : ℝ) : 
  (|x - 1| + |x - 2| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by sorry

end solution_set_of_inequality_l73_73305


namespace smallest_four_digit_divisible_by_six_l73_73482

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l73_73482


namespace determine_constants_l73_73827

theorem determine_constants
  (C D : ℝ)
  (h1 : 3 * C + D = 7)
  (h2 : 4 * C - 2 * D = -15) :
  C = -0.1 ∧ D = 7.3 :=
by
  sorry

end determine_constants_l73_73827


namespace remaining_days_temperature_l73_73916

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end remaining_days_temperature_l73_73916


namespace common_tangent_slope_l73_73877

theorem common_tangent_slope (a m : ℝ) : 
  ((∃ a, ∃ m, l = (2 * a) ∧ l = (3 * m^2) ∧ a^2 = 2 * m^3) → (l = 0 ∨ l = 64 / 27)) := 
sorry

end common_tangent_slope_l73_73877


namespace at_least_one_less_than_two_l73_73691

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
by 
  sorry

end at_least_one_less_than_two_l73_73691


namespace sequence_length_l73_73227

theorem sequence_length :
  ∀ (n : ℕ), 
    (2 + 4 * (n - 1) = 2010) → n = 503 :=
by
    intro n
    intro h
    sorry

end sequence_length_l73_73227


namespace greatest_prime_factor_of_221_l73_73953

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greatest_prime_factor (n : ℕ) (p : ℕ) : Prop := 
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → q ≤ p

theorem greatest_prime_factor_of_221 : greatest_prime_factor 221 17 := by
  sorry

end greatest_prime_factor_of_221_l73_73953


namespace commute_time_absolute_difference_l73_73995

theorem commute_time_absolute_difference (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : (x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_absolute_difference_l73_73995


namespace angle_in_third_quadrant_l73_73083

open Real

/--
Given that 2013° can be represented as 213° + 5 * 360° and that 213° is a third quadrant angle,
we can deduce that 2013° is also a third quadrant angle.
-/
theorem angle_in_third_quadrant (h1 : 2013 = 213 + 5 * 360) (h2 : 180 < 213 ∧ 213 < 270) : 
  (540 < 2013 % 360 ∧ 2013 % 360 < 270) :=
sorry

end angle_in_third_quadrant_l73_73083


namespace first_solution_carbonation_l73_73999

-- Definitions of given conditions in the problem
variable (C : ℝ) -- Percentage of carbonated water in the first solution
variable (L : ℝ) -- Percentage of lemonade in the first solution

-- The second solution is 55% carbonated water and 45% lemonade
def second_solution_carbonated : ℝ := 55
def second_solution_lemonade : ℝ := 45

-- The mixture is 65% carbonated water and 40% of the volume is the first solution
def mixture_carbonated : ℝ := 65
def first_solution_contribution : ℝ := 0.40
def second_solution_contribution : ℝ := 0.60

-- The relationship between the solution components
def equation := first_solution_contribution * C + second_solution_contribution * second_solution_carbonated = mixture_carbonated

-- The statement to prove: C = 80
theorem first_solution_carbonation :
  equation C →
  C = 80 :=
sorry

end first_solution_carbonation_l73_73999


namespace ratio_area_III_IV_l73_73162

theorem ratio_area_III_IV 
  (perimeter_I : ℤ)
  (perimeter_II : ℤ)
  (perimeter_IV : ℤ)
  (side_III_is_three_times_side_I : ℤ)
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 20)
  (h3 : perimeter_IV = 32)
  (h4 : side_III_is_three_times_side_I = 3 * (perimeter_I / 4)) :
  (3 * (perimeter_I / 4))^2 / (perimeter_IV / 4)^2 = 9 / 4 :=
by
  sorry

end ratio_area_III_IV_l73_73162


namespace problem_sum_congruent_mod_11_l73_73681

theorem problem_sum_congruent_mod_11 : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := 
by
  -- Proof needed here
  sorry

end problem_sum_congruent_mod_11_l73_73681


namespace expected_winnings_is_0_25_l73_73801

def prob_heads : ℚ := 3 / 8
def prob_tails : ℚ := 1 / 4
def prob_edge  : ℚ := 1 / 8
def prob_disappear : ℚ := 1 / 4

def winnings_heads : ℚ := 2
def winnings_tails : ℚ := 5
def winnings_edge  : ℚ := -2
def winnings_disappear : ℚ := -6

def expected_winnings : ℚ := 
  prob_heads * winnings_heads +
  prob_tails * winnings_tails +
  prob_edge  * winnings_edge +
  prob_disappear * winnings_disappear

theorem expected_winnings_is_0_25 : expected_winnings = 0.25 := by
  sorry

end expected_winnings_is_0_25_l73_73801


namespace other_solution_quadratic_l73_73366

theorem other_solution_quadratic (h : (49 : ℚ) * (5 / 7)^2 - 88 * (5 / 7) + 40 = 0) : 
  ∃ x : ℚ, x ≠ 5 / 7 ∧ (49 * x^2 - 88 * x + 40 = 0) ∧ x = 8 / 7 :=
by
  sorry

end other_solution_quadratic_l73_73366


namespace least_common_multiple_9_12_15_l73_73968

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l73_73968


namespace odd_multiple_of_9_implies_multiple_of_3_l73_73792

-- Define an odd number that is a multiple of 9
def odd_multiple_of_nine (m : ℤ) : Prop := 9 * m % 2 = 1

-- Define multiples of 3 and 9
def multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k
def multiple_of_nine (n : ℤ) : Prop := ∃ k : ℤ, n = 9 * k

-- The main statement
theorem odd_multiple_of_9_implies_multiple_of_3 (n : ℤ) 
  (h1 : ∀ n, multiple_of_nine n → multiple_of_three n)
  (h2 : odd_multiple_of_nine n ∧ multiple_of_nine n) : 
  multiple_of_three n :=
by sorry

end odd_multiple_of_9_implies_multiple_of_3_l73_73792


namespace tire_circumference_l73_73787

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) 
  (h1 : rpm = 400) 
  (h2 : speed_kmh = 144) 
  (h3 : (speed_kmh * 1000 / 60) = (rpm * C)) : 
  C = 6 :=
by
  sorry

end tire_circumference_l73_73787


namespace drew_got_wrong_19_l73_73147

theorem drew_got_wrong_19 :
  ∃ (D_wrong C_wrong : ℕ), 
    (20 + D_wrong = 52) ∧
    (14 + C_wrong = 52) ∧
    (C_wrong = 2 * D_wrong) ∧
    D_wrong = 19 :=
by
  sorry

end drew_got_wrong_19_l73_73147


namespace maximize_probability_sum_is_15_l73_73192

def initial_list : List ℤ := [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]

def valid_pairs (lst : List ℤ) : List (ℤ × ℤ) :=
  (lst.product lst).filter (λ ⟨x, y⟩ => x < y ∧ x + y = 15)

def remove_one_element (lst : List ℤ) (x : ℤ) : List ℤ :=
  lst.erase x

theorem maximize_probability_sum_is_15 :
  (List.length (valid_pairs (remove_one_element initial_list 8))
   = List.maximum (List.map (λ x => List.length (valid_pairs (remove_one_element initial_list x))) initial_list)) :=
sorry

end maximize_probability_sum_is_15_l73_73192


namespace stone_10th_image_l73_73219

-- Definition of the recursive sequence
def stones (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 1 => stones n + 3 * (n + 1) + 1

-- The statement we need to prove
theorem stone_10th_image : stones 9 = 145 := 
  sorry

end stone_10th_image_l73_73219


namespace complement_of_union_eq_l73_73033

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define the subset A
def A : Set ℤ := {-1, 0, 1}

-- Define the subset B
def B : Set ℤ := {0, 1, 2, 3}

-- Define the union of A and B
def A_union_B : Set ℤ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℤ := U \ A_union_B

-- State the theorem to be proved
theorem complement_of_union_eq {U A B : Set ℤ} :
  U = {-1, 0, 1, 2, 3, 4} →
  A = {-1, 0, 1} →
  B = {0, 1, 2, 3} →
  complement_U_A_union_B = {4} :=
by
  intros hU hA hB
  sorry

end complement_of_union_eq_l73_73033


namespace no_finite_set_A_exists_l73_73342

theorem no_finite_set_A_exists (A : Set ℕ) (h : Finite A ∧ ∀ a ∈ A, 2 * a ∈ A ∨ a / 3 ∈ A) : False :=
sorry

end no_finite_set_A_exists_l73_73342


namespace milton_zoology_books_l73_73752

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l73_73752


namespace geom_seq_min_sum_l73_73355

theorem geom_seq_min_sum {a : ℕ → ℝ} (a_pos : ∀ n, 0 < a n) (r : ℝ) 
  (r_pos : 0 < r) (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_eq : a 3 + a 2 = a 1 + a 0 + 8) : 
  ∃ a_6 a_5, a_6 + a_5 = 32 ∧ 
    a_6 = a 1 * r ^ 5 ∧ a_5 = a 1 * r ^ 4 := 
begin 
  sorry 
end

end geom_seq_min_sum_l73_73355


namespace M_lt_N_l73_73718

/-- M is the coefficient of x^4 y^2 in the expansion of (x^2 + x + 2y)^5 -/
def M : ℕ := 120

/-- N is the sum of the coefficients in the expansion of (3/x - x)^7 -/
def N : ℕ := 128

/-- The relationship between M and N -/
theorem M_lt_N : M < N := by 
  dsimp [M, N]
  sorry

end M_lt_N_l73_73718


namespace candy_bar_cost_l73_73675

theorem candy_bar_cost :
  ∃ C : ℕ, (C + 1 = 3) → (C = 2) :=
by
  use 2
  intros h
  linarith

end candy_bar_cost_l73_73675


namespace find_k_inv_h_8_l73_73761

variable (h k : ℝ → ℝ)

-- Conditions
axiom h_inv_k_x (x : ℝ) : h⁻¹ (k x) = 3 * x - 4
axiom h_3x_minus_4 (x : ℝ) : k x = h (3 * x - 4)

-- The statement we want to prove
theorem find_k_inv_h_8 : k⁻¹ (h 8) = 8 := 
  sorry

end find_k_inv_h_8_l73_73761


namespace min_AB_CD_value_l73_73878

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def AB_CD (AC BD CB : vector) : ℝ :=
  let AB := (CB.1 + AC.1, CB.2 + AC.2)
  let CD := (CB.1 + BD.1, CB.2 + BD.2)
  dot_product AB CD

theorem min_AB_CD_value : ∀ (AC BD : vector), AC = (1, 2) → BD = (-2, 2) → 
  ∃ CB : vector, AB_CD AC BD CB = -9 / 4 :=
by
  intros AC BD hAC hBD
  sorry

end min_AB_CD_value_l73_73878


namespace decimal_subtraction_l73_73979

theorem decimal_subtraction (a b : ℝ) (h1 : a = 3.79) (h2 : b = 2.15) : a - b = 1.64 := by
  rw [h1, h2]
  -- This follows from the correct calculation rule
  sorry

end decimal_subtraction_l73_73979


namespace two_numbers_ratio_l73_73762

theorem two_numbers_ratio (A B : ℕ) (h_lcm : Nat.lcm A B = 30) (h_sum : A + B = 25) :
  ∃ x y : ℕ, x = 2 ∧ y = 3 ∧ A / B = x / y := 
sorry

end two_numbers_ratio_l73_73762


namespace milton_books_l73_73745

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l73_73745


namespace number_of_ways_to_choose_materials_l73_73473

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l73_73473


namespace square_perimeter_l73_73803

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l73_73803


namespace quadratic_inequality_solution_range_l73_73114

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, a*x^2 + 2*a*x - 4 < 0) ↔ -4 < a ∧ a < 0 := 
by
  sorry

end quadratic_inequality_solution_range_l73_73114


namespace product_of_repeating_decimal_l73_73667

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l73_73667


namespace area_above_line_of_circle_l73_73072

-- Define the circle equation
def circle_eq (x y : ℝ) := (x - 10)^2 + (y - 5)^2 = 50

-- Define the line equation
def line_eq (x y : ℝ) := y = x - 6

-- The area to determine
def area_above_line (R : ℝ) := 25 * R

-- Proof statement
theorem area_above_line_of_circle : area_above_line Real.pi = 25 * Real.pi :=
by
  -- mark the proof as sorry to skip the proof
  sorry

end area_above_line_of_circle_l73_73072


namespace perpendicular_line_slopes_l73_73370

theorem perpendicular_line_slopes (α₁ : ℝ) (hα₁ : α₁ = 30) (l₁ : ℝ) (k₁ : ℝ) (k₂ : ℝ) (α₂ : ℝ)
  (h₁ : k₁ = Real.tan (α₁ * Real.pi / 180))
  (h₂ : k₂ = - 1 / k₁)
  (h₃ : k₂ = - Real.sqrt 3)
  (h₄ : 0 < α₂ ∧ α₂ < 180)
  : k₂ = - Real.sqrt 3 ∧ α₂ = 120 := sorry

end perpendicular_line_slopes_l73_73370


namespace number_of_ways_to_choose_reading_materials_l73_73448

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l73_73448


namespace square_perimeter_l73_73806

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l73_73806


namespace river_flow_rate_l73_73328

-- Define the conditions
def depth : ℝ := 8
def width : ℝ := 25
def volume_per_min : ℝ := 26666.666666666668

-- The main theorem proving the rate at which the river is flowing
theorem river_flow_rate : (volume_per_min / (depth * width)) = 133.33333333333334 := by
  -- Express the area of the river's cross-section
  let area := depth * width
  -- Define the velocity based on the given volume and calculated area
  let velocity := volume_per_min / area
  -- Simplify and derive the result
  show velocity = 133.33333333333334
  sorry

end river_flow_rate_l73_73328


namespace geometric_sequence_ratio_28_l73_73149

noncomputable def geometric_sequence_sum_ratio (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :=
  S 6 / S 3 = 28

theorem geometric_sequence_ratio_28 (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_GS : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h_increasing : ∀ n m, n < m → a1 * q^n < a1 * q^m) 
  (h_mean : 2 * 6 * a1 * q^6 = a1 * q^7 + a1 * q^8) : 
  geometric_sequence_sum_ratio a1 q S := 
by {
  -- Proof should be completed here
  sorry
}

end geometric_sequence_ratio_28_l73_73149


namespace sum_of_coefficients_of_expansion_l73_73368

-- Define a predicate for a term being constant
def is_constant_term (n : ℕ) (term : ℚ) : Prop := 
  term = 0

-- Define the sum of coefficients computation
noncomputable def sum_of_coefficients (n : ℕ) : ℚ := 
  (1 - 3)^n

-- The main statement of the problem in Lean
theorem sum_of_coefficients_of_expansion {n : ℕ} 
  (h : is_constant_term n (2 * n - 10)) : 
  sum_of_coefficients 5 = -32 := 
sorry

end sum_of_coefficients_of_expansion_l73_73368


namespace solve_equation_l73_73274

theorem solve_equation (x : ℝ) (hx : x ≠ 0) 
  (h : 1 / 4 + 8 / x = 13 / x + 1 / 8) : 
  x = 40 :=
sorry

end solve_equation_l73_73274


namespace rectangle_circle_area_ratio_l73_73770

noncomputable def area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) : ℝ :=
  (2 * w^2) / (Real.pi * r^2)

theorem rectangle_circle_area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) :
  area_ratio w r h = 18 / (Real.pi * Real.pi) :=
by
  sorry

end rectangle_circle_area_ratio_l73_73770


namespace meal_cost_l73_73298

theorem meal_cost :
  ∃ (s c p : ℝ),
  (5 * s + 8 * c + 2 * p = 5.40) ∧
  (3 * s + 11 * c + 2 * p = 4.95) ∧
  (s + c + p = 1.55) :=
sorry

end meal_cost_l73_73298


namespace greatest_positive_integer_x_l73_73526

theorem greatest_positive_integer_x : ∃ (x : ℕ), (x > 0) ∧ (∀ y : ℕ, y > 0 → (y^3 < 20 * y → y ≤ 4)) ∧ (x^3 < 20 * x) ∧ ∀ z : ℕ, (z > 0) → (z^3 < 20 * z → x ≥ z)  :=
sorry

end greatest_positive_integer_x_l73_73526


namespace derivative_of_curve_tangent_line_at_one_l73_73856

-- Definition of the curve
def curve (x : ℝ) : ℝ := x^3 + 5 * x^2 + 3 * x

-- Part 1: Prove the derivative of the curve
theorem derivative_of_curve (x : ℝ) :
  deriv curve x = 3 * x^2 + 10 * x + 3 :=
sorry

-- Part 2: Prove the equation of the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a = 16 ∧ b = -1 ∧ c = -7 ∧
  ∀ (x y : ℝ), curve 1 = 9 → y - 9 = 16 * (x - 1) → a * x + b * y + c = 0 :=
sorry

end derivative_of_curve_tangent_line_at_one_l73_73856


namespace part_a_part_b_l73_73577

-- Definition based on conditions
def S (n k : ℕ) : ℕ :=
  -- Placeholder: Actual definition would count the coefficients
  -- of (x+1)^n that are not divisible by k.
  sorry

-- Part (a) proof statement
theorem part_a : S 2012 3 = 324 :=
by sorry

-- Part (b) proof statement
theorem part_b : 2012 ∣ S (2012^2011) 2011 :=
by sorry

end part_a_part_b_l73_73577


namespace subset_singleton_natural_l73_73654

/-
  Problem Statement:
  Prove that the set {2} is a subset of the set of natural numbers.
-/

open Set

theorem subset_singleton_natural :
  {2} ⊆ (Set.univ : Set ℕ) :=
by
  sorry

end subset_singleton_natural_l73_73654


namespace unit_vectors_equal_magnitude_l73_73848

variable {ℝ : Type*}
variable [normed_group ℝ] [normed_space ℝ ℝ]

theorem unit_vectors_equal_magnitude
    (a b : ℝ)
    (unit_a : ‖a‖ = 1)
    (unit_b : ‖b‖ = 1) :
    ‖a‖ = ‖b‖ := 
sorry

end unit_vectors_equal_magnitude_l73_73848


namespace twenty_five_billion_scientific_notation_l73_73728

theorem twenty_five_billion_scientific_notation :
  (25 * 10^9 : ℝ) = 2.5 * 10^10 := 
by simp only [←mul_assoc, ←@pow_add ℝ, pow_one, two_mul];
   norm_num

end twenty_five_billion_scientific_notation_l73_73728


namespace less_than_reciprocal_l73_73980

theorem less_than_reciprocal (a b c d e : ℝ) (ha : a = -3) (hb : b = -1/2) (hc : c = 0.5) (hd : d = 1) (he : e = 3) :
  (a < 1 / a) ∧ (c < 1 / c) ∧ ¬(b < 1 / b) ∧ ¬(d < 1 / d) ∧ ¬(e < 1 / e) :=
by
  sorry

end less_than_reciprocal_l73_73980


namespace hypotenuse_unique_l73_73524

theorem hypotenuse_unique (a b : ℝ) (h: ∃ x : ℝ, x^2 = a^2 + b^2 ∧ x > 0) : 
  ∃! c : ℝ, c^2 = a^2 + b^2 :=
sorry

end hypotenuse_unique_l73_73524


namespace gumball_problem_l73_73085

/--
A gumball machine contains 10 red, 6 white, 8 blue, and 9 green gumballs.
The least number of gumballs a person must buy to be sure of getting four gumballs of the same color is 13.
-/
theorem gumball_problem
  (red white blue green : ℕ)
  (h_red : red = 10)
  (h_white : white = 6)
  (h_blue : blue = 8)
  (h_green : green = 9) :
  ∃ n, n = 13 ∧ (∀ gumballs : ℕ, gumballs ≥ 13 → (∃ color_count : ℕ, color_count ≥ 4 ∧ (color_count = red ∨ color_count = white ∨ color_count = blue ∨ color_count = green))) :=
sorry

end gumball_problem_l73_73085


namespace remainder_ab_mod_n_l73_73156

theorem remainder_ab_mod_n (n : ℕ) (a c : ℤ) (h1 : a * c ≡ 1 [ZMOD n]) (h2 : b = a * c) :
    (a * b % n) = (a % n) :=
  by
  sorry

end remainder_ab_mod_n_l73_73156


namespace max_marks_are_700_l73_73329

/-- 
A student has to obtain 33% of the total marks to pass.
The student got 175 marks and failed by 56 marks.
Prove that the maximum marks are 700.
-/
theorem max_marks_are_700 (M : ℝ) (h1 : 0.33 * M = 175 + 56) : M = 700 :=
sorry

end max_marks_are_700_l73_73329


namespace general_term_formula_l73_73935

def sequence_sum (n : ℕ) : ℕ := 3 * n^2 - 2 * n

def general_term (n : ℕ) : ℕ := if n = 0 then 0 else 6 * n - 5

theorem general_term_formula (n : ℕ) (h : n > 0) :
  general_term n = sequence_sum n - sequence_sum (n - 1) := by
  sorry

end general_term_formula_l73_73935


namespace least_possible_value_of_z_l73_73700

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end least_possible_value_of_z_l73_73700


namespace smallest_n_for_candy_l73_73231

theorem smallest_n_for_candy (r g b n : ℕ) (h1 : 10 * r = 18 * g) (h2 : 18 * g = 20 * b) (h3 : 20 * b = 24 * n) : n = 15 :=
by
  sorry

end smallest_n_for_candy_l73_73231


namespace cone_sphere_ratio_l73_73210

/-- A right circular cone and a sphere have bases with the same radius r. 
If the volume of the cone is one-third that of the sphere, find the ratio of 
the altitude of the cone to the radius of its base. -/
theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) 
    (volume_cone : ℝ) (volume_sphere : ℝ)
    (cone_volume_formula : volume_cone = (1 / 3) * π * r^2 * h) 
    (sphere_volume_formula : volume_sphere = (4 / 3) * π * r^3) 
    (volume_relation : volume_cone = (1 / 3) * volume_sphere) : 
    h / r = 4 / 3 :=
by
    sorry

end cone_sphere_ratio_l73_73210


namespace gears_together_again_l73_73037

theorem gears_together_again (r₁ r₂ : ℕ) (h₁ : r₁ = 3) (h₂ : r₂ = 5) : 
  (∃ t : ℕ, t = Nat.lcm r₁ r₂ / r₁ ∨ t = Nat.lcm r₁ r₂ / r₂) → 5 = Nat.lcm r₁ r₂ / min r₁ r₂ := 
by
  sorry

end gears_together_again_l73_73037


namespace no_real_solution_l73_73679

theorem no_real_solution (n : ℝ) : (∀ x : ℝ, (x+6)*(x-3) = n + 4*x → false) ↔ n < -73/4 := by
  sorry

end no_real_solution_l73_73679


namespace arithmetic_sequence_problem_l73_73910

-- Define the arithmetic sequence and related sum functions
def a_n (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

def S (a1 d : ℤ) (n : ℕ) : ℤ :=
  (a1 + a_n a1 d n) * n / 2

-- Problem statement: proving a_5 = -1 given the conditions
theorem arithmetic_sequence_problem :
  (∃ (a1 d : ℕ), S a1 d 2 = S a1 d 6 ∧ a_n a1 d 4 = 1) → a_n a1 d 5 = -1 :=
by
  -- Assume the statement and then skip the proof
  sorry

end arithmetic_sequence_problem_l73_73910


namespace ratio_black_white_extended_pattern_l73_73529

def originalBlackTiles : ℕ := 8
def originalWhiteTiles : ℕ := 17
def originalSquareSide : ℕ := 5
def extendedSquareSide : ℕ := 7
def newBlackTiles : ℕ := (extendedSquareSide * extendedSquareSide) - (originalSquareSide * originalSquareSide)
def totalBlackTiles : ℕ := originalBlackTiles + newBlackTiles
def totalWhiteTiles : ℕ := originalWhiteTiles

theorem ratio_black_white_extended_pattern : totalBlackTiles / totalWhiteTiles = 32 / 17 := sorry

end ratio_black_white_extended_pattern_l73_73529


namespace lines_coinicide_l73_73421

open Real

theorem lines_coinicide (k m n : ℝ) :
  (∃ (x y : ℝ), y = k * x + m ∧ y = m * x + n ∧ y = n * x + k) →
  k = m ∧ m = n :=
by
  sorry

end lines_coinicide_l73_73421


namespace two_students_one_common_material_l73_73451

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l73_73451


namespace total_area_of_house_is_2300_l73_73636

-- Definitions based on the conditions in the problem
def area_living_room_dining_room_kitchen : ℕ := 1000
def area_master_bedroom_suite : ℕ := 1040
def area_guest_bedroom : ℕ := area_master_bedroom_suite / 4

-- Theorem to state the total area of the house
theorem total_area_of_house_is_2300 :
  area_living_room_dining_room_kitchen + area_master_bedroom_suite + area_guest_bedroom = 2300 :=
by
  sorry

end total_area_of_house_is_2300_l73_73636


namespace solve_inequality_l73_73684

theorem solve_inequality : 
  {x : ℝ | (1 / (x^2 + 1)) > (4 / x) + (21 / 10)} = {x : ℝ | -2 < x ∧ x < 0} :=
by
  sorry

end solve_inequality_l73_73684


namespace largest_initial_number_l73_73400

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l73_73400


namespace find_coordinates_of_B_find_equation_of_BC_l73_73388

-- Problem 1: Prove that the coordinates of B are (10, 5)
theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0) :
  B = (10, 5) :=
sorry

-- Problem 2: Prove that the equation of line BC is 2x + 9y - 65 = 0
theorem find_equation_of_BC (A B C : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0)
  (coordinates_B : B = (10, 5)) :
  ∃ k : ℝ, ∀ P : ℝ × ℝ, (P.1 - C.1) / (P.2 - C.2) = k → 2 * P.1 + 9 * P.2 - 65 = 0 :=
sorry

end find_coordinates_of_B_find_equation_of_BC_l73_73388


namespace math_problem_l73_73477

theorem math_problem
  (a b c d : ℚ)
  (h₁ : a = 1 / 3)
  (h₂ : b = 1 / 6)
  (h₃ : c = 1 / 9)
  (h₄ : d = 1 / 18) :
  9 * (a + b + c + d)⁻¹ = 27 / 2 := 
sorry

end math_problem_l73_73477


namespace participants_in_sports_activities_l73_73185

theorem participants_in_sports_activities:
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = 3 ∧
  let a := 10 * x + 6
  let b := 10 * y + 6
  let c := 10 * z + 6
  a + b + c = 48 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a = 6 ∧ b = 16 ∧ c = 26 ∨ a = 6 ∧ b = 26 ∧ c = 16 ∨ a = 16 ∧ b = 6 ∧ c = 26 ∨ a = 16 ∧ b = 26 ∧ c = 6 ∨ a = 26 ∧ b = 6 ∧ c = 16 ∨ a = 26 ∧ b = 16 ∧ c = 6)
  :=
by {
  sorry
}

end participants_in_sports_activities_l73_73185


namespace sqrt3_times_3_minus_sqrt3_bound_l73_73528

theorem sqrt3_times_3_minus_sqrt3_bound : 2 < (Real.sqrt 3) * (3 - (Real.sqrt 3)) ∧ (Real.sqrt 3) * (3 - (Real.sqrt 3)) < 3 := 
by 
  sorry

end sqrt3_times_3_minus_sqrt3_bound_l73_73528


namespace g_at_5_l73_73300

variable (g : ℝ → ℝ)

-- Define the condition on g
def functional_condition : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1

-- The statement proven should be g(5) = 8 given functional_condition
theorem g_at_5 (h : functional_condition g) : g 5 = 8 := by
  sorry

end g_at_5_l73_73300


namespace smallest_y_for_perfect_cube_l73_73086

theorem smallest_y_for_perfect_cube (x y : ℕ) (x_def : x = 11 * 36 * 54) : 
  (∃ y : ℕ, y > 0 ∧ ∀ (n : ℕ), (x * y = n^3 ↔ y = 363)) := 
by 
  sorry

end smallest_y_for_perfect_cube_l73_73086


namespace students_material_selection_l73_73464

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l73_73464


namespace evaluate_fraction_expression_l73_73105

theorem evaluate_fraction_expression :
  ( (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) ) = 2 / 5 :=
by
  sorry

end evaluate_fraction_expression_l73_73105


namespace students_material_selection_l73_73465

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l73_73465


namespace trig_expr_eval_sin_minus_cos_l73_73321

-- Problem 1: Evaluation of trigonometric expression
theorem trig_expr_eval : 
    (Real.sin (-π / 2) + 3 * Real.cos 0 - 2 * Real.tan (3 * π / 4) - 4 * Real.cos (5 * π / 3)) = 2 :=
by 
    sorry

-- Problem 2: Given tangent value and angle constraints, find sine minus cosine
theorem sin_minus_cos {θ : ℝ} 
    (h1 : Real.tan θ = 4 / 3)
    (h2 : 0 < θ)
    (h3 : θ < π / 2) : 
    (Real.sin θ - Real.cos θ) = 1 / 5 :=
by 
    sorry

end trig_expr_eval_sin_minus_cos_l73_73321


namespace product_of_repeating_decimal_l73_73672

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l73_73672


namespace slope_of_line_l73_73241

theorem slope_of_line (x y : ℝ) (h : 6 * x + 7 * y - 3 = 0) : - (6 / 7) = -6 / 7 := 
by
  sorry

end slope_of_line_l73_73241


namespace number_greater_than_neg_one_by_two_l73_73435

/-- Theorem: The number that is greater than -1 by 2 is 1. -/
theorem number_greater_than_neg_one_by_two : -1 + 2 = 1 :=
by
  sorry

end number_greater_than_neg_one_by_two_l73_73435


namespace lcm_9_12_15_l73_73957

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l73_73957


namespace problem1_problem2_l73_73414

noncomputable def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

noncomputable def g' (x : ℝ) : ℝ := 15 * x^4 - 16 * x^3 + 6 * x^2 - 56 * x + 15

theorem problem1 : g 6 = 17568 := 
by {
  sorry
}

theorem problem2 : g' 6 = 15879 := 
by {
  sorry
}

end problem1_problem2_l73_73414


namespace percentage_employed_females_is_16_l73_73883

/- 
  In Town X, the population is divided into three age groups: 18-34, 35-54, and 55+.
  For each age group, the percentage of the employed population is 64%, and the percentage of employed males is 48%.
  We need to prove that the percentage of employed females in each age group is 16%.
-/

theorem percentage_employed_females_is_16
  (percentage_employed_population : ℝ)
  (percentage_employed_males : ℝ)
  (h1 : percentage_employed_population = 0.64)
  (h2 : percentage_employed_males = 0.48) :
  percentage_employed_population - percentage_employed_males = 0.16 :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end percentage_employed_females_is_16_l73_73883


namespace inequality_1_inequality_2_inequality_3_l73_73909

variable (x : ℝ)

theorem inequality_1 (h : 2 * x^2 - 3 * x + 1 ≥ 0) : x ≤ 1 / 2 ∨ x ≥ 1 := 
  sorry

theorem inequality_2 (h : x^2 - 2 * x - 3 < 0) : -1 < x ∧ x < 3 := 
  sorry

theorem inequality_3 (h : -3 * x^2 + 5 * x - 2 > 0) : 2 / 3 < x ∧ x < 1 := 
  sorry

end inequality_1_inequality_2_inequality_3_l73_73909


namespace x_squared_minus_y_squared_l73_73008

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end x_squared_minus_y_squared_l73_73008


namespace max_club_members_l73_73090

open Set

variable {U : Type} (A B C : Set U)

theorem max_club_members (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11)
    (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
    (A ∪ B ∪ C).card ≤ 22 :=
by {
  -- The proof will go here, but for now we skip it.
  sorry
}

end max_club_members_l73_73090


namespace arithmetic_sequence_general_formula_l73_73249

noncomputable def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_formula {a : ℕ → ℤ} (h_seq : arithmetic_seq a) 
  (h_a1 : a 1 = 6) (h_a3a5 : a 3 + a 5 = 0) : 
  ∀ n, a n = 8 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l73_73249


namespace range_of_a_outside_circle_l73_73854

  variable (a : ℝ)

  def point_outside_circle (a : ℝ) : Prop :=
    let x := a
    let y := 2
    let distance_sqr := (x - a) ^ 2 + (y - 3 / 2) ^ 2
    let r_sqr := 1 / 4
    distance_sqr > r_sqr

  theorem range_of_a_outside_circle {a : ℝ} (h : point_outside_circle a) :
      2 < a ∧ a < 9 / 4 := sorry
  
end range_of_a_outside_circle_l73_73854


namespace train_is_late_l73_73097

theorem train_is_late (S : ℝ) (T : ℝ) (T' : ℝ) (h1 : T = 2) (h2 : T' = T * 5 / 4) :
  (T' - T) * 60 = 30 :=
by
  sorry

end train_is_late_l73_73097


namespace g_h_value_l73_73002

def g (x : ℕ) : ℕ := 3 * x^2 + 2
def h (x : ℕ) : ℕ := 5 * x^3 - 2

theorem g_h_value : g (h 2) = 4334 := by
  sorry

end g_h_value_l73_73002


namespace division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l73_73530

theorem division_to_fraction : (7 / 9) = 7 / 9 := by
  sorry

theorem fraction_to_division : 12 / 7 = 12 / 7 := by
  sorry

theorem mixed_to_improper_fraction : (3 + 5 / 8) = 29 / 8 := by
  sorry

theorem whole_to_fraction : 6 = 66 / 11 := by
  sorry

end division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l73_73530


namespace area_of_shaded_triangle_l73_73273

-- Definitions of the conditions
def AC := 4
def BC := 3
def BD := 10
def CD := BD - BC

-- Statement of the proof problem
theorem area_of_shaded_triangle :
  (1 / 2 * CD * AC = 14) := by
  sorry

end area_of_shaded_triangle_l73_73273


namespace star_shell_arrangements_l73_73409

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem star_shell_arrangements : nat :=
  let total_arrangements := factorial 14
  let symmetries := 14
  total_arrangements / symmetries = 6227020800

end star_shell_arrangements_l73_73409


namespace trig_expression_value_l73_73359

theorem trig_expression_value (α : Real) (h : Real.tan (3 * Real.pi + α) = 3) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) /
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 :=
by
  sorry

end trig_expression_value_l73_73359


namespace largest_initial_number_l73_73397

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l73_73397


namespace possible_values_of_a_l73_73255

noncomputable def f (x a : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 2 * a * x + 2 else x + 9 / x - 3 * a

theorem possible_values_of_a (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ 1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end possible_values_of_a_l73_73255


namespace unique_solution_condition_l73_73250

-- Define p and q as real numbers
variables (p q : ℝ)

-- The Lean statement to prove a unique solution when q ≠ 4
theorem unique_solution_condition : (∀ x : ℝ, (4 * x - 7 + p = q * x + 2) ↔ (q ≠ 4)) :=
by
  sorry

end unique_solution_condition_l73_73250


namespace lcm_9_12_15_l73_73972

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l73_73972


namespace distinct_terms_in_expansion_l73_73713

theorem distinct_terms_in_expansion :
  let P1 := (x + y + z)
  let P2 := (u + v + w + x + y)
  ∃ n : ℕ, n = 14 ∧ 
    ∀ a b, 
      (a ∈ {x, y, z} ∧ b ∈ {u, v, w, x, y}) → 
      (a * b ∈ expansion_of P1 P2)
:= sorry

end distinct_terms_in_expansion_l73_73713


namespace points_on_single_circle_l73_73123

theorem points_on_single_circle (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ i j : Fin n, ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p, f p ≠ p) ∧ f (points i) = points j ∧ 
        (∀ k : Fin n, ∃ p, points k = f p)) :
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ i : Fin n, dist (points i) O = r := sorry

end points_on_single_circle_l73_73123


namespace son_l73_73981

variable (S M : ℤ)

-- Conditions
def condition1 : Prop := M = S + 24
def condition2 : Prop := M + 2 = 2 * (S + 2)

theorem son's_age : condition1 S M ∧ condition2 S M → S = 22 :=
by
  sorry

end son_l73_73981


namespace max_intersection_distance_l73_73272

theorem max_intersection_distance :
  let C1_x (α : ℝ) := 2 + 2 * Real.cos α
  let C1_y (α : ℝ) := 2 * Real.sin α
  let C2_x (β : ℝ) := 2 * Real.cos β
  let C2_y (β : ℝ) := 2 + 2 * Real.sin β
  let l1 (α : ℝ) := α
  let l2 (α : ℝ) := α - Real.pi / 6
  (0 < Real.pi / 2) →
  let OP (α : ℝ) := 4 * Real.cos α
  let OQ (α : ℝ) := 4 * Real.sin (α - Real.pi / 6)
  let pq_prod (α : ℝ) := OP α * OQ α
  ∀α, 0 < α ∧ α < Real.pi / 2 → pq_prod α ≤ 4 := by
  sorry

end max_intersection_distance_l73_73272


namespace expected_number_of_digits_on_fair_icosahedral_die_l73_73505

noncomputable def expected_digits_fair_icosahedral_die : ℚ :=
  let prob_one_digit := (9 : ℚ) / 20
  let prob_two_digits := (11 : ℚ) / 20
  (prob_one_digit * 1) + (prob_two_digits * 2)

theorem expected_number_of_digits_on_fair_icosahedral_die : expected_digits_fair_icosahedral_die = 1.55 := by
  sorry

end expected_number_of_digits_on_fair_icosahedral_die_l73_73505


namespace average_of_P_and_R_l73_73173

theorem average_of_P_and_R (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : P = 3000)
  : (P + R) / 2 = 6200 := by
  sorry

end average_of_P_and_R_l73_73173


namespace distance_equal_x_value_l73_73049

theorem distance_equal_x_value :
  (∀ P Q R : ℝ × ℝ × ℝ, P = (x, 2, 1) ∧ Q = (1, 1, 2) ∧ R = (2, 1, 1) →
  dist P Q = dist P R →
  x = 1) :=
by
  -- Define the points P, Q, R
  let P := (x, 2, 1)
  let Q := (1, 1, 2)
  let R := (2, 1, 1)

  -- Given the condition
  intro h
  sorry

end distance_equal_x_value_l73_73049


namespace hypotenuse_length_l73_73911

noncomputable def side_lengths_to_hypotenuse (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_length 
  (AB BC : ℝ) 
  (h1 : Real.sqrt (AB * BC) = 8) 
  (h2 : (1 / 2) * AB * BC = 48) :
  side_lengths_to_hypotenuse AB BC = 4 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_length_l73_73911


namespace ashley_percentage_secured_l73_73568

noncomputable def marks_secured : ℕ := 332
noncomputable def max_marks : ℕ := 400
noncomputable def percentage_secured : ℕ := (marks_secured * 100) / max_marks

theorem ashley_percentage_secured 
    (h₁ : marks_secured = 332)
    (h₂ : max_marks = 400) :
    percentage_secured = 83 := by
  -- Proof goes here
  sorry

end ashley_percentage_secured_l73_73568


namespace power_function_value_l73_73558

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h₁ : f x = x ^ α) (h₂ : f (1 / 2) = 4) : f 8 = 1 / 64 := by
  sorry

end power_function_value_l73_73558


namespace find_other_endpoint_l73_73434

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ)
  (h_midpoint_x : x_m = (x_1 + x_2) / 2)
  (h_midpoint_y : y_m = (y_1 + y_2) / 2)
  (h_given_midpoint : x_m = 3 ∧ y_m = 0)
  (h_given_endpoint1 : x_1 = 7 ∧ y_1 = -4) :
  x_2 = -1 ∧ y_2 = 4 :=
sorry

end find_other_endpoint_l73_73434


namespace lcm_of_9_12_15_is_180_l73_73960

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l73_73960


namespace executed_is_9_l73_73986

-- Define the conditions based on given problem
variables (x K I : ℕ)

-- Condition 1: Number of killed
def number_killed (x : ℕ) : ℕ := 2 * x + 4

-- Condition 2: Number of injured
def number_injured (x : ℕ) : ℕ := (16 * x) / 3 + 8

-- Condition 3: Total of killed, injured, and executed is less than 98
def total_less_than_98 (x : ℕ) (k : ℕ) (i : ℕ) : Prop := k + i + x < 98

-- Condition 4: Relation between killed and executed
def killed_relation (x : ℕ) (k : ℕ) : Prop := k - 4 = 2 * x

-- The final theorem statement to prove
theorem executed_is_9 : ∃ x, number_killed x = 2 * x + 4 ∧
                       number_injured x = (16 * x) / 3 + 8 ∧
                       total_less_than_98 x (number_killed x) (number_injured x) ∧
                       killed_relation x (number_killed x) ∧
                       x = 9 :=
by
  sorry

end executed_is_9_l73_73986


namespace mary_cut_10_roses_l73_73942

-- Define the initial and final number of roses
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses cut as the difference between final and initial
def roses_cut : ℕ :=
  final_roses - initial_roses

-- Theorem stating the number of roses cut by Mary
theorem mary_cut_10_roses : roses_cut = 10 := by
  sorry

end mary_cut_10_roses_l73_73942


namespace early_time_l73_73796

noncomputable def speed1 : ℝ := 5 -- km/hr
noncomputable def timeLate : ℝ := 5 / 60 -- convert minutes to hours
noncomputable def speed2 : ℝ := 10 -- km/hr
noncomputable def distance : ℝ := 2.5 -- km

theorem early_time (speed1 speed2 distance : ℝ) (timeLate : ℝ) :
  (distance / speed1 - timeLate) * 60 - (distance / speed2) * 60 = 10 :=
by
  sorry

end early_time_l73_73796


namespace lcm_9_12_15_l73_73958

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l73_73958


namespace no_six_consecutive_nat_num_sum_eq_2015_l73_73276

theorem no_six_consecutive_nat_num_sum_eq_2015 :
  ∀ (a b c d e f : ℕ),
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e + 1 = f →
  a * b * c + d * e * f ≠ 2015 :=
by
  intros a b c d e f h
  sorry

end no_six_consecutive_nat_num_sum_eq_2015_l73_73276


namespace evaluate_expression_l73_73106

theorem evaluate_expression : 
  (Int.ceil ((Int.floor ((15 / 8 : Rat) ^ 2) : Rat) - (19 / 5 : Rat) : Rat) : Int) = 0 :=
sorry

end evaluate_expression_l73_73106


namespace least_common_multiple_9_12_15_l73_73965

def prime_factorizations (n : ℕ) : list (ℕ × ℕ) -- This is just a placeholder to suggest the existence of a function
| 9 := [(3, 2)]
| 12 := [(2, 2), (3, 1)]
| 15 := [(3, 1), (5, 1)]
| _ := []

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b -- This computes the least common multiple of two numbers

def LCM_three (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem least_common_multiple_9_12_15 :
  LCM_three 9 12 15 = 180 := by
  sorry

end least_common_multiple_9_12_15_l73_73965


namespace garden_length_l73_73199

theorem garden_length (w l : ℝ) (h1: l = 2 * w) (h2 : 2 * l + 2 * w = 180) : l = 60 := 
by
  sorry

end garden_length_l73_73199


namespace kelly_baking_powder_l73_73489

variable (current_supply : ℝ) (additional_supply : ℝ)

theorem kelly_baking_powder (h1 : current_supply = 0.3)
                            (h2 : additional_supply = 0.1) :
                            current_supply + additional_supply = 0.4 := 
by
  sorry

end kelly_baking_powder_l73_73489


namespace square_perimeter_l73_73805

theorem square_perimeter (area : ℝ) (h : area = 625) : 
  let s := Real.sqrt area in
  (4 * s) = 100 :=
by
  let s := Real.sqrt area
  have hs : s = 25 := by sorry
  calc
    (4 * s) = 4 * 25 : by rw hs
          ... = 100   : by norm_num

end square_perimeter_l73_73805


namespace percentage_fertilizer_in_second_solution_l73_73036

theorem percentage_fertilizer_in_second_solution 
    (v1 v2 v3 : ℝ) 
    (p1 p2 p3 : ℝ) 
    (h1 : v1 = 20) 
    (h2 : v2 + v1 = 42) 
    (h3 : p1 = 74 / 100) 
    (h4 : p2 = 63 / 100) 
    (h5 : v3 = (63 * 42 - 74 * 20) / 22) 
    : p3 = (53 / 100) :=
by
  sorry

end percentage_fertilizer_in_second_solution_l73_73036


namespace inequality_solution_1_inequality_system_solution_2_l73_73042

theorem inequality_solution_1 (x : ℝ) : 
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 := 
sorry

theorem inequality_system_solution_2 (x : ℝ) : 
  (-2 * x ≤ -3) ∧ (x / 2 < 2) ↔ (3 / 2 ≤ x) ∧ (x < 4) :=
sorry

end inequality_solution_1_inequality_system_solution_2_l73_73042


namespace budget_equality_year_l73_73079

theorem budget_equality_year :
  let budget_q_1990 := 540000
  let budget_v_1990 := 780000
  let annual_increase_q := 30000
  let annual_decrease_v := 10000

  let budget_q (n : ℕ) := budget_q_1990 + n * annual_increase_q
  let budget_v (n : ℕ) := budget_v_1990 - n * annual_decrease_v

  (∃ n : ℕ, budget_q n = budget_v n ∧ 1990 + n = 1996) :=
by
  sorry

end budget_equality_year_l73_73079


namespace number_of_girls_l73_73211

theorem number_of_girls (B G : ℕ) 
  (h1 : B = G + 124) 
  (h2 : B + G = 1250) : G = 563 :=
by
  sorry

end number_of_girls_l73_73211


namespace john_spent_l73_73737

/-- John bought 9.25 meters of cloth at a cost price of $44 per meter.
    Prove that the total amount John spent on the cloth is $407. -/
theorem john_spent :
  let length_of_cloth := 9.25
  let cost_per_meter := 44
  let total_cost := length_of_cloth * cost_per_meter
  total_cost = 407 := by
  sorry

end john_spent_l73_73737


namespace largest_initial_number_l73_73402

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l73_73402


namespace square_perimeter_l73_73802

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l73_73802


namespace xyz_eq_7cubed_l73_73557

theorem xyz_eq_7cubed (x y z : ℤ) (h1 : x^2 * y * z^3 = 7^4) (h2 : x * y^2 = 7^5) : x * y * z = 7^3 := 
by 
  sorry

end xyz_eq_7cubed_l73_73557


namespace proof_problem_l73_73006

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end proof_problem_l73_73006


namespace perimeter_of_square_l73_73095

-- Definitions based on problem conditions
def is_square_divided_into_four_congruent_rectangles (s : ℝ) (rect_perimeter : ℝ) : Prop :=
  rect_perimeter = 30 ∧ s > 0

-- Statement of the theorem to be proved
theorem perimeter_of_square (s : ℝ) (rect_perimeter : ℝ) (h : is_square_divided_into_four_congruent_rectangles s rect_perimeter) :
  4 * s = 48 :=
by sorry

end perimeter_of_square_l73_73095


namespace number_of_planting_methods_l73_73513

noncomputable def num_planting_methods : ℕ :=
  -- Six different types of crops
  let crops := ['A', 'B', 'C', 'D', 'E', 'F']
  -- Six trial fields arranged in a row, numbered 1 through 6
  -- Condition: Crop A cannot be planted in the first two fields
  -- Condition: Crop B must not be adjacent to crop A
  -- Answer: 240 different planting methods
  240

theorem number_of_planting_methods :
  num_planting_methods = 240 :=
  by
    -- Proof omitted
    sorry

end number_of_planting_methods_l73_73513


namespace distance_from_A_to_directrix_l73_73367

open Real

noncomputable def distance_from_point_to_directrix (p : ℝ) : ℝ :=
  1 + p / 2

theorem distance_from_A_to_directrix : 
  ∃ (p : ℝ), (sqrt 5)^2 = 2 * p ∧ distance_from_point_to_directrix p = 9 / 4 :=
by 
  sorry

end distance_from_A_to_directrix_l73_73367


namespace isosceles_triangle_area_48_l73_73171

noncomputable def isosceles_triangle_area (b h s : ℝ) : ℝ :=
  (1 / 2) * (2 * b) * h

theorem isosceles_triangle_area_48 :
  ∀ (b s : ℝ),
  b ^ 2 + 8 ^ 2 = s ^ 2 ∧ s + b = 16 →
  isosceles_triangle_area b 8 s = 48 :=
by
  intros b s h
  unfold isosceles_triangle_area
  sorry

end isosceles_triangle_area_48_l73_73171


namespace arielle_age_l73_73774

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end arielle_age_l73_73774


namespace equation_has_solution_implies_a_ge_2_l73_73533

theorem equation_has_solution_implies_a_ge_2 (a : ℝ) :
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) → a ≥ 2 :=
by
  sorry

end equation_has_solution_implies_a_ge_2_l73_73533


namespace no_2007_in_display_can_2008_appear_in_display_l73_73323

-- Definitions of the operations as functions on the display number.
def button1 (n : ℕ) : ℕ := 1
def button2 (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n
def button3 (n : ℕ) : ℕ := if n >= 3 then n - 3 else n
def button4 (n : ℕ) : ℕ := 4 * n

-- Initial condition
def initial_display : ℕ := 0

-- Define can_appear as a recursive function to determine if a number can appear on the display.
def can_appear (target : ℕ) : Prop :=
  ∃ n : ℕ, n = target ∧ (∃ f : (ℕ → ℕ) → ℕ, f initial_display = target)

-- Prove the statements:
theorem no_2007_in_display : ¬ can_appear 2007 :=
  sorry

theorem can_2008_appear_in_display : can_appear 2008 :=
  sorry

end no_2007_in_display_can_2008_appear_in_display_l73_73323


namespace erica_pie_percentage_l73_73831

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end erica_pie_percentage_l73_73831


namespace max_club_members_l73_73091

open Set

variable {U : Type} (A B C : Set U)

theorem max_club_members (hA : A.card = 8) (hB : B.card = 7) (hC : C.card = 11)
    (hAB : (A ∩ B).card ≥ 2) (hBC : (B ∩ C).card ≥ 3) (hAC : (A ∩ C).card ≥ 4) :
    (A ∪ B ∪ C).card ≤ 22 :=
by {
  -- The proof will go here, but for now we skip it.
  sorry
}

end max_club_members_l73_73091


namespace intersection_high_probability_l73_73947

-- Definitions
def parabola (a b : ℤ) (x : ℝ) : ℝ := x^2 + (a : ℝ) * x + (b : ℝ)
def line (c d : ℤ) (x : ℝ) : ℝ := (c : ℝ) * x + (d : ℝ)

def quadratic_discriminant (a b c d : ℤ) : ℝ := (a - c : ℝ)^2 - 4 * (b - d : ℝ)

-- Event E: The quadratic equation has at least one real root (i.e., its discriminant is non-negative)
def E (a b c d : ℤ) : Prop := quadratic_discriminant a b c d ≥ 0

-- Probability measure on the set of integers from 1 to 6
noncomputable def uniform_measure : measure ℤ := sorry  -- We assume a uniform probability measure on the given range

-- Mathematical statement
theorem intersection_high_probability : 
  (probability (set_of (λ (abcd : ℤ × ℤ × ℤ × ℤ), E abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2))) > 5/6 :=
sorry

end intersection_high_probability_l73_73947


namespace solve_for_x_l73_73322

theorem solve_for_x : ∀ x : ℕ, x + 1315 + 9211 - 1569 = 11901 → x = 2944 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l73_73322


namespace transform_equation_to_polynomial_l73_73196

variable (x y : ℝ)

theorem transform_equation_to_polynomial (h : (x^2 + 2) / (x + 1) = y) :
    (x^2 + 2) / (x + 1) + (5 * (x + 1)) / (x^2 + 2) = 6 → y^2 - 6 * y + 5 = 0 :=
by
  intro h_eq
  sorry

end transform_equation_to_polynomial_l73_73196


namespace chocolate_game_winner_l73_73949

theorem chocolate_game_winner (m n : ℕ) (h_m : m = 6) (h_n : n = 8) :
  (∃ k : ℕ, (48 - 1) - 2 * k = 0) ↔ true :=
by
  sorry

end chocolate_game_winner_l73_73949


namespace number_of_ways_to_choose_reading_materials_l73_73449

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l73_73449


namespace largest_7_10_triple_l73_73225

theorem largest_7_10_triple :
  ∃ M : ℕ, (3 * M = Nat.ofDigits 10 (Nat.digits 7 M))
  ∧ (∀ N : ℕ, (3 * N = Nat.ofDigits 10 (Nat.digits 7 N)) → N ≤ M)
  ∧ M = 335 :=
sorry

end largest_7_10_triple_l73_73225


namespace sum_of_ages_l73_73026

theorem sum_of_ages (juliet_age maggie_age ralph_age nicky_age : ℕ)
  (h1 : juliet_age = 10)
  (h2 : juliet_age = maggie_age + 3)
  (h3 : ralph_age = juliet_age + 2)
  (h4 : nicky_age = ralph_age / 2) :
  maggie_age + ralph_age + nicky_age = 25 :=
by
  sorry

end sum_of_ages_l73_73026


namespace S_5_value_l73_73369

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom a2a4 (h : geometric_sequence a) : a 1 * a 3 = 16
axiom S3 : S 3 = 7

theorem S_5_value 
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = a 0 * (1 - (a 1)^(n)) / (1 - a 1)) :
  S 5 = 31 :=
sorry

end S_5_value_l73_73369


namespace find_greater_number_l73_73064

theorem find_greater_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x > y) : x = 25 := 
sorry

end find_greater_number_l73_73064


namespace every_positive_integer_sum_of_distinct_powers_of_3_4_7_l73_73158

theorem every_positive_integer_sum_of_distinct_powers_of_3_4_7 :
  ∀ n : ℕ, n > 0 →
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  ∃ (i j k : ℕ), n = 3^i + 4^j + 7^k :=
by
  sorry

end every_positive_integer_sum_of_distinct_powers_of_3_4_7_l73_73158


namespace largest_multiple_of_8_smaller_than_neg_80_l73_73609

theorem largest_multiple_of_8_smaller_than_neg_80 :
  ∃ n : ℤ, (8 ∣ n) ∧ n < -80 ∧ ∀ m : ℤ, (8 ∣ m ∧ m < -80 → m ≤ n) :=
sorry

end largest_multiple_of_8_smaller_than_neg_80_l73_73609


namespace maple_tree_total_l73_73939

-- Conditions
def initial_maple_trees : ℕ := 53
def trees_planted_today : ℕ := 11

-- Theorem to prove the result
theorem maple_tree_total : initial_maple_trees + trees_planted_today = 64 := by
  sorry

end maple_tree_total_l73_73939


namespace lunch_break_duration_l73_73906

-- Definitions based on the conditions
variables (p h1 h2 L : ℝ)
-- Monday equation
def monday_eq : Prop := (9 - L/60) * (p + h1 + h2) = 0.55
-- Tuesday equation
def tuesday_eq : Prop := (7 - L/60) * (p + h2) = 0.35
-- Wednesday equation
def wednesday_eq : Prop := (5 - L/60) * (p + h1 + h2) = 0.25
-- Thursday equation
def thursday_eq : Prop := (4 - L/60) * p = 0.15

-- Combine all conditions
def all_conditions : Prop :=
  monday_eq p h1 h2 L ∧ tuesday_eq p h2 L ∧ wednesday_eq p h1 h2 L ∧ thursday_eq p L

-- Proof that the lunch break duration is 60 minutes
theorem lunch_break_duration : all_conditions p h1 h2 L → L = 60 :=
by
  sorry

end lunch_break_duration_l73_73906


namespace profit_percentage_is_correct_l73_73624

-- Define the conditions
variables (market_price_per_pen : ℝ) (discount_percentage : ℝ) (total_pens_bought : ℝ) (cost_pens_market_price : ℝ)
variables (cost_price_per_pen : ℝ) (selling_price_per_pen : ℝ) (profit_per_pen : ℝ) (profit_percent : ℝ)

-- Conditions
def condition_1 : market_price_per_pen = 1 := by sorry
def condition_2 : discount_percentage = 0.01 := by sorry
def condition_3 : total_pens_bought = 80 := by sorry
def condition_4 : cost_pens_market_price = 36 := by sorry

-- Definitions based on conditions
def cost_price_per_pen_def : cost_price_per_pen = cost_pens_market_price / total_pens_bought := by sorry
def selling_price_per_pen_def : selling_price_per_pen = market_price_per_pen * (1 - discount_percentage) := by sorry
def profit_per_pen_def : profit_per_pen = selling_price_per_pen - cost_price_per_pen := by sorry
def profit_percent_def : profit_percent = (profit_per_pen / cost_price_per_pen) * 100 := by sorry

-- The statement to prove
theorem profit_percentage_is_correct : profit_percent = 120 :=
by
  have h1 : cost_price_per_pen = 36 / 80 := by sorry
  have h2 : selling_price_per_pen = 1 * (1 - 0.01) := by sorry
  have h3 : profit_per_pen = 0.99 - 0.45 := by sorry
  have h4 : profit_percent = (0.54 / 0.45) * 100 := by sorry
  sorry

end profit_percentage_is_correct_l73_73624


namespace fraction_conversion_l73_73075

theorem fraction_conversion :
  let A := 4.5
  let B := 0.8
  let C := 80.0
  let D := 0.08
  let E := 0.45
  (4 / 5) = B :=
by
  sorry

end fraction_conversion_l73_73075


namespace tanya_addition_problem_l73_73406

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l73_73406


namespace range_of_expression_l73_73150

theorem range_of_expression (x : ℝ) (h1 : 1 - 3 * x ≥ 0) (h2 : 2 * x ≠ 0) : x ≤ 1 / 3 ∧ x ≠ 0 := by
  sorry

end range_of_expression_l73_73150


namespace division_correct_l73_73531

-- Definitions based on conditions
def expr1 : ℕ := 12 + 15 * 3
def expr2 : ℚ := 180 / expr1

-- Theorem statement using the question and correct answer
theorem division_correct : expr2 = 180 / 57 := by
  sorry

end division_correct_l73_73531


namespace total_short_trees_after_planting_l73_73439

def current_short_oak_trees := 3
def current_short_pine_trees := 4
def current_short_maple_trees := 5
def new_short_oak_trees := 9
def new_short_pine_trees := 6
def new_short_maple_trees := 4

theorem total_short_trees_after_planting :
  current_short_oak_trees + current_short_pine_trees + current_short_maple_trees +
  new_short_oak_trees + new_short_pine_trees + new_short_maple_trees = 31 := by
  sorry

end total_short_trees_after_planting_l73_73439


namespace possible_values_of_n_are_1_prime_or_prime_squared_l73_73567

/-- A function that determines if an n x n grid with n marked squares satisfies the condition
    that every rectangle of exactly n grid squares contains at least one marked square. -/
def satisfies_conditions (n : ℕ) (marked_squares : List (ℕ × ℕ)) : Prop :=
  n.succ.succ ≤ marked_squares.length ∧ ∀ (a b : ℕ), a * b = n → ∃ x y, (x, y) ∈ marked_squares ∧ x < n ∧ y < n

/-- The main theorem stating the possible values of n. -/
theorem possible_values_of_n_are_1_prime_or_prime_squared :
  ∀ (n : ℕ), (∃ p : ℕ, Prime p ∧ (n = 1 ∨ n = p ∨ n = p^2)) ↔ satisfies_conditions n marked_squares :=
by
  sorry

end possible_values_of_n_are_1_prime_or_prime_squared_l73_73567


namespace triangle_angle_A_eq_pi_div_3_triangle_area_l73_73381

variable (A B C a b c : ℝ)
variable (S : ℝ)

-- First part: Proving A = π / 3
theorem triangle_angle_A_eq_pi_div_3 (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                                      (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : A > 0) (h6 : A < Real.pi) :
  A = Real.pi / 3 :=
sorry

-- Second part: Finding the area of the triangle
theorem triangle_area (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                      (h2 : b + c = Real.sqrt 10) (h3 : a = 2) (h4 : A = Real.pi / 3) :
  S = Real.sqrt 3 / 2 :=
sorry

end triangle_angle_A_eq_pi_div_3_triangle_area_l73_73381


namespace perpendicular_lines_a_equals_one_l73_73375

theorem perpendicular_lines_a_equals_one
  (a : ℝ)
  (l1 : ∀ x y : ℝ, x - 2 * y + 1 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + a * y - 1 = 0)
  (perpendicular : ∀ x y : ℝ, (x - 2 * y + 1 = 0) ∧ (2 * x + a * y - 1 = 0) → 
    (-(1 / -2) * -(2 / a)) = -1) :
  a = 1 :=
by
  sorry

end perpendicular_lines_a_equals_one_l73_73375


namespace x_squared_minus_y_squared_l73_73009

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end x_squared_minus_y_squared_l73_73009


namespace tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l73_73492

structure Tetrahedron :=
  (faces : Nat := 4)
  (vertices : Nat := 4)
  (valence : Nat := 3)
  (face_shape : String := "triangular")

structure Cube :=
  (faces : Nat := 6)
  (vertices : Nat := 8)
  (valence : Nat := 3)
  (face_shape : String := "square")

structure Octahedron :=
  (faces : Nat := 8)
  (vertices : Nat := 6)
  (valence : Nat := 4)
  (face_shape : String := "triangular")

structure Dodecahedron :=
  (faces : Nat := 12)
  (vertices : Nat := 20)
  (valence : Nat := 3)
  (face_shape : String := "pentagonal")

structure Icosahedron :=
  (faces : Nat := 20)
  (vertices : Nat := 12)
  (valence : Nat := 5)
  (face_shape : String := "triangular")

theorem tetrahedron_is_self_dual:
  Tetrahedron := by
  sorry

theorem cube_is_dual_to_octahedron:
  Cube × Octahedron := by
  sorry

theorem dodecahedron_is_dual_to_icosahedron:
  Dodecahedron × Icosahedron := by
  sorry

end tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l73_73492


namespace smallest_four_digit_number_divisible_by_6_l73_73485

theorem smallest_four_digit_number_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m % 6 = 0) → n ≤ m :=
begin
  use 1002,
  split,
  { exact nat.le_succ 999,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.le_succ 1001,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  { intros m h1,
    exact le_of_lt_iff.2 (by linarith) }
end

end smallest_four_digit_number_divisible_by_6_l73_73485


namespace central_angle_measure_l73_73427

-- Constants representing the arc length and the area of the sector.
def arc_length : ℝ := 5
def sector_area : ℝ := 5

-- Variables representing the central angle in radians and the radius.
variable (α r : ℝ)

-- Conditions given in the problem.
axiom arc_length_eq : arc_length = α * r
axiom sector_area_eq : sector_area = 1 / 2 * α * r^2

-- The goal to prove that the radian measure of the central angle α is 5 / 2.
theorem central_angle_measure : α = 5 / 2 := by sorry

end central_angle_measure_l73_73427


namespace first_system_solution_second_system_solution_l73_73294

theorem first_system_solution (x y : ℝ) (h₁ : 3 * x - y = 8) (h₂ : 3 * x - 5 * y = -20) : 
  x = 5 ∧ y = 7 := 
by
  sorry

theorem second_system_solution (x y : ℝ) (h₁ : x / 3 - y / 2 = -1) (h₂ : 3 * x - 2 * y = 1) : 
  x = 3 ∧ y = 4 := 
by
  sorry

end first_system_solution_second_system_solution_l73_73294


namespace remainder_sum_of_numbers_l73_73687

theorem remainder_sum_of_numbers :
  ((123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7) = 5 :=
by
  sorry

end remainder_sum_of_numbers_l73_73687


namespace english_alphabet_is_set_l73_73618

-- Conditions definition: Elements of a set must have the properties of definiteness, distinctness, and unorderedness.
def is_definite (A : Type) : Prop := ∀ (a b : A), a = b ∨ a ≠ b
def is_distinct (A : Type) : Prop := ∀ (a b : A), a ≠ b → (a ≠ b)
def is_unordered (A : Type) : Prop := true  -- For simplicity, we assume unorderedness holds for any set

-- Property that verifies if the 26 letters of the English alphabet can form a set
def english_alphabet_set : Prop :=
  is_definite Char ∧ is_distinct Char ∧ is_unordered Char

theorem english_alphabet_is_set : english_alphabet_set :=
  sorry

end english_alphabet_is_set_l73_73618


namespace sequence_general_term_l73_73019

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l73_73019


namespace a_2017_value_l73_73247

variable (a S : ℕ → ℤ)

-- Given conditions
axiom a1 : a 1 = 1
axiom a_nonzero : ∀ n, a n ≠ 0
axiom a_S_relation : ∀ n, a n * a (n + 1) = 2 * S n - 1

theorem a_2017_value : a 2017 = 2017 := by
  sorry

end a_2017_value_l73_73247


namespace greatest_t_value_l73_73685

theorem greatest_t_value :
  ∃ t_max : ℝ, (∀ t : ℝ, ((t ≠  8) ∧ (t ≠ -7) → (t^2 - t - 90) / (t - 8) = 6 / (t + 7) → t ≤ t_max)) ∧ t_max = -1 :=
sorry

end greatest_t_value_l73_73685


namespace find_sum_of_abc_l73_73591

theorem find_sum_of_abc
  (h : ∀ x : ℝ, sin x ^ 2 + sin (2 * x) ^ 2 + sin (3 * x) ^ 2 + sin (4 * x) ^ 2 = 2) :
  let a := 1
  let b := 2
  let c := 5
  (cos a x * cos b x * cos c x = 0) ∧ (a + b + c = 8) :=
by
  sorry

end find_sum_of_abc_l73_73591


namespace primes_dividing_sequence_l73_73951

def a_n (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def is_prime (p : ℕ) := Nat.Prime p

theorem primes_dividing_sequence :
  {p : ℕ | is_prime p ∧ p ≤ 19 ∧ ∃ n ≥ 1, p ∣ a_n n} = {3, 7, 13, 17} :=
by
  sorry

end primes_dividing_sequence_l73_73951


namespace right_triangle_area_l73_73757

-- Define the initial lengths and the area calculation function.
def area_right_triangle (base height : ℕ) : ℕ :=
  (1 / 2) * base * height

theorem right_triangle_area
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a = 18)
  (h2 : b = 24)
  (h3 : c = 30)  -- Derived from the solution steps
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_right_triangle a b = 216 :=
sorry

end right_triangle_area_l73_73757


namespace sum_of_areas_of_squares_l73_73334

def is_right_angle (a b c : ℝ) : Prop := (a^2 + b^2 = c^2)

def isSquare (side : ℝ) : Prop := (side > 0)

def area_of_square (side : ℝ) : ℝ := side^2

theorem sum_of_areas_of_squares 
  (P Q R S X Y : ℝ) 
  (h1 : is_right_angle P Q R)
  (h2 : PR = 15)
  (h3 : isSquare PR)
  (h4 : isSquare PQ) :
  area_of_square PR + area_of_square PQ = 450 := 
sorry


end sum_of_areas_of_squares_l73_73334


namespace percentage_of_75_eq_percent_of_450_l73_73324

theorem percentage_of_75_eq_percent_of_450 (x : ℝ) (h : (x / 100) * 75 = 0.025 * 450) : x = 15 := 
sorry

end percentage_of_75_eq_percent_of_450_l73_73324


namespace transformation_matrix_is_correct_l73_73336

open Real
open Matrix

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let θ := (60 : ℝ) * (π / 180)
  let scale := 2 : ℝ
  let rotation := λ θ : ℝ, Matrix.of ![
    ![cos θ, -sin θ],
    ![sin θ, cos θ]
  ]
  scale • (rotation θ)

-- Expected matrix result for 60-degree anticlockwise rotation and scaling by 2
noncomputable def expected_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![1, -sqrt 3],
    ![sqrt 3, 1]
  ]

theorem transformation_matrix_is_correct : transformation_matrix = expected_matrix :=
  sorry

end transformation_matrix_is_correct_l73_73336


namespace erica_pie_fraction_as_percentage_l73_73833

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end erica_pie_fraction_as_percentage_l73_73833


namespace value_of_expression_l73_73137

theorem value_of_expression (a b : ℝ) (h1 : ∃ x : ℝ, x^2 + 3 * x - 5 = 0)
  (h2 : ∃ y : ℝ, y^2 + 3 * y - 5 = 0)
  (h3 : a ≠ b)
  (h4 : ∀ r : ℝ, r^2 + 3 * r - 5 = 0 → r = a ∨ r = b) : a^2 + 3 * a * b + a - 2 * b = -4 :=
by
  sorry

end value_of_expression_l73_73137


namespace slope_of_tangent_at_1_0_l73_73060

noncomputable def f (x : ℝ) : ℝ :=
2 * x^2 - 2 * x

def derivative_f (x : ℝ) : ℝ :=
4 * x - 2

theorem slope_of_tangent_at_1_0 : derivative_f 1 = 2 :=
by
  sorry

end slope_of_tangent_at_1_0_l73_73060


namespace borrowed_sheets_l73_73711

-- Defining the page sum function
def sum_pages (n : ℕ) : ℕ := n * (n + 1)

-- Formulating the main theorem statement
theorem borrowed_sheets (b c : ℕ) (H : c + b ≤ 30) (H_avg : (sum_pages b + sum_pages (30 - b - c) - sum_pages (b + c)) * 2 = 25 * (60 - 2 * c)) :
  c = 10 :=
sorry

end borrowed_sheets_l73_73711


namespace evaluate_composite_l73_73031

def f (x : ℕ) : ℕ := 2 * x + 5
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_composite : f (g (f 3)) = 79 := by
  sorry

end evaluate_composite_l73_73031


namespace dice_sum_probability_l73_73358

theorem dice_sum_probability
  (a b c d : ℕ)
  (cond1 : 1 ≤ a ∧ a ≤ 6)
  (cond2 : 1 ≤ b ∧ b ≤ 6)
  (cond3 : 1 ≤ c ∧ c ≤ 6)
  (cond4 : 1 ≤ d ∧ d ≤ 6)
  (sum_cond : a + b + c + d = 5) :
  (∃ p, p = 1 / 324) :=
sorry

end dice_sum_probability_l73_73358


namespace best_k_k_l73_73534

theorem best_k_k' (v w x y z : ℝ) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  1 < (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) ∧ 
  (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) < 4 :=
sorry

end best_k_k_l73_73534


namespace households_using_both_brands_l73_73508

def total : ℕ := 260
def neither : ℕ := 80
def onlyA : ℕ := 60
def onlyB (both : ℕ) : ℕ := 3 * both

theorem households_using_both_brands (both : ℕ) : 80 + 60 + both + onlyB both = 260 → both = 30 :=
by
  intro h
  sorry

end households_using_both_brands_l73_73508


namespace find_x_l73_73182

-- Define the mean of three numbers
def mean_three (a b c : ℕ) : ℚ := (a + b + c) / 3

-- Define the mean of two numbers
def mean_two (x y : ℕ) : ℚ := (x + y) / 2

-- Main theorem: value of x that satisfies the given condition
theorem find_x : 
  (mean_three 6 9 18) = (mean_two x 15) → x = 7 :=
by
  sorry

end find_x_l73_73182


namespace cyclist_speed_north_l73_73190

theorem cyclist_speed_north (v : ℝ) :
  (∀ d t : ℝ, d = 50 ∧ t = 1 ∧ 40 * t + v * t = d) → v = 10 :=
by
  sorry

end cyclist_speed_north_l73_73190


namespace truth_values_of_p_and_q_l73_73862

variable {p q : Prop}

theorem truth_values_of_p_and_q (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end truth_values_of_p_and_q_l73_73862


namespace compute_abs_a_plus_b_plus_c_l73_73588

variable (a b c : ℝ)

theorem compute_abs_a_plus_b_plus_c (h1 : a^2 - b * c = 14)
                                   (h2 : b^2 - c * a = 14)
                                   (h3 : c^2 - a * b = -3) :
                                   |a + b + c| = 5 :=
sorry

end compute_abs_a_plus_b_plus_c_l73_73588


namespace one_third_times_seven_times_nine_l73_73335

theorem one_third_times_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_times_seven_times_nine_l73_73335


namespace area_between_curves_l73_73296

open Real Integral

theorem area_between_curves : 
  (∫ x in (0 : ℝ)..(1 : ℝ), (sqrt x - x)) = (1 / 6) :=
by
  sorry

end area_between_curves_l73_73296


namespace student_score_is_64_l73_73215

-- Define the total number of questions and correct responses.
def total_questions : ℕ := 100
def correct_responses : ℕ := 88

-- Function to calculate the score based on the grading rule.
def calculate_score (total : ℕ) (correct : ℕ) : ℕ :=
  correct - 2 * (total - correct)

-- The theorem that states the score for the given conditions.
theorem student_score_is_64 :
  calculate_score total_questions correct_responses = 64 :=
by
  sorry

end student_score_is_64_l73_73215


namespace value_of_x_plus_y_l73_73000

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) : x + y = 4/3 :=
sorry

end value_of_x_plus_y_l73_73000


namespace range_of_m_l73_73240

theorem range_of_m (m: ℝ) : (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → x^2 - x + 1 > 2*x + m) → m < -1 :=
by
  intro h
  sorry

end range_of_m_l73_73240


namespace travel_time_without_paddles_l73_73115

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l73_73115


namespace rectangle_new_area_l73_73209

theorem rectangle_new_area
  (L W : ℝ) (h1 : L * W = 600) :
  let L' := 0.8 * L
  let W' := 1.3 * W
  (L' * W' = 624) :=
by
  -- Let L' = 0.8 * L
  -- Let W' = 1.3 * W
  -- Proof goes here
  sorry

end rectangle_new_area_l73_73209


namespace fill_blanks_l73_73585

/-
Given the following conditions:
1. 20 * (x1 - 8) = 20
2. x2 / 2 + 17 = 20
3. 3 * x3 - 4 = 20
4. (x4 + 8) / 12 = y4
5. 4 * x5 = 20
6. 20 * (x6 - y6) = 100

Prove that:
1. x1 = 9
2. x2 = 6
3. x3 = 8
4. x4 = 4 and y4 = 1
5. x5 = 5
6. x6 = 7 and y6 = 2
-/
theorem fill_blanks (x1 x2 x3 x4 y4 x5 x6 y6 : ℕ) :
  20 * (x1 - 8) = 20 →
  x2 / 2 + 17 = 20 →
  3 * x3 - 4 = 20 →
  (x4 + 8) / 12 = y4 →
  4 * x5 = 20 →
  20 * (x6 - y6) = 100 →
  x1 = 9 ∧
  x2 = 6 ∧
  x3 = 8 ∧
  x4 = 4 ∧
  y4 = 1 ∧
  x5 = 5 ∧
  x6 = 7 ∧
  y6 = 2 :=
by
  sorry

end fill_blanks_l73_73585


namespace two_regular_pentagons_similar_l73_73077

def is_regular_pentagon (P : Type) [polygon P] : Prop := sorry

theorem two_regular_pentagons_similar (P1 P2 : Type) [polygon P1] [polygon P2] 
  (h1 : is_regular_pentagon P1) (h2 : is_regular_pentagon P2) : 
  similar P1 P2 :=
sorry

end two_regular_pentagons_similar_l73_73077


namespace fraction_irreducible_gcd_2_power_l73_73078

-- Proof problem (a)
theorem fraction_irreducible (n : ℕ) : gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

-- Proof problem (b)
theorem gcd_2_power (n m : ℕ) : gcd (2^100 - 1) (2^120 - 1) = 2^20 - 1 :=
sorry

end fraction_irreducible_gcd_2_power_l73_73078


namespace sum_of_integers_c_with_four_solutions_l73_73223

noncomputable def g (x : ℝ) : ℝ :=
  ((x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 120) - 2

theorem sum_of_integers_c_with_four_solutions :
  (∃ (c : ℤ), ∀ x : ℝ, -4.5 ≤ x ∧ x ≤ 4.5 → g x = c ↔ c = -2) → c = -2 :=
by
  sorry

end sum_of_integers_c_with_four_solutions_l73_73223


namespace lily_pad_half_lake_l73_73565

theorem lily_pad_half_lake
  (P : ℕ → ℝ) -- Define a function P(n) which represents the size of the patch on day n.
  (h1 : ∀ n, P n = P (n - 1) * 2) -- Every day, the patch doubles in size.
  (h2 : P 58 = 1) -- It takes 58 days for the patch to cover the entire lake (normalized to 1).
  : P 57 = 1 / 2 :=
by
  sorry

end lily_pad_half_lake_l73_73565


namespace derivative_of_f_l73_73535

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + (1 / 3) * ((3 * x - 1) / (3 * x^2 - 2 * x + 1))

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
by intros; sorry

end derivative_of_f_l73_73535


namespace find_shortest_side_of_triangle_l73_73721

def Triangle (A B C : Type) := true -- Dummy definition for a triangle

structure Segments :=
(BD DE EC : ℝ)

def angle_ratios (AD AE : ℝ) (r1 r2 : ℕ) := true -- Dummy definition for angle ratios

def triangle_conditions (ABC : Type) (s : Segments) (r1 r2 : ℕ)
  (h1 : angle_ratios AD AE r1 r2)
  (h2 : s.BD = 4)
  (h3 : s.DE = 2)
  (h4 : s.EC = 5) : Prop := True

noncomputable def shortestSide (ABC : Type) (s : Segments) (r1 r2 : ℕ) : ℝ := 
  if true then sorry else 0 -- Placeholder for the shortest side length function

theorem find_shortest_side_of_triangle (ABC : Type) (s : Segments)
  (h1 : angle_ratios AD AE 2 3) (h2 : angle_ratios AE AD 1 1)
  (h3 : s.BD = 4) (h4 : s.DE = 2) (h5 : s.EC = 5) :
  shortestSide ABC s 2 3 = 30 / 11 :=
sorry

end find_shortest_side_of_triangle_l73_73721


namespace final_price_correct_l73_73523

def cost_price : ℝ := 20
def profit_percentage : ℝ := 0.30
def sale_discount_percentage : ℝ := 0.50
def local_tax_percentage : ℝ := 0.10
def packaging_fee : ℝ := 2

def selling_price_before_discount : ℝ := cost_price * (1 + profit_percentage)
def sale_discount : ℝ := sale_discount_percentage * selling_price_before_discount
def price_after_discount : ℝ := selling_price_before_discount - sale_discount
def tax : ℝ := local_tax_percentage * price_after_discount
def price_with_tax : ℝ := price_after_discount + tax
def final_price : ℝ := price_with_tax + packaging_fee

theorem final_price_correct : final_price = 16.30 :=
by
  sorry

end final_price_correct_l73_73523


namespace fraction_of_income_from_tips_l73_73515

theorem fraction_of_income_from_tips (S T : ℚ) (h : T = (11/4) * S) : (T / (S + T)) = (11/15) :=
by sorry

end fraction_of_income_from_tips_l73_73515


namespace david_tips_l73_73525

noncomputable def avg_tips_resort (tips_other_months : ℝ) (months : ℕ) := tips_other_months / months

theorem david_tips 
  (tips_march_to_july_september : ℝ)
  (tips_august_resort : ℝ)
  (total_tips_delivery_driver : ℝ)
  (total_tips_resort : ℝ)
  (total_tips : ℝ)
  (fraction_august : ℝ)
  (avg_tips := avg_tips_resort tips_march_to_july_september 6):
  tips_august_resort = 4 * avg_tips →
  total_tips_delivery_driver = 2 * avg_tips →
  total_tips_resort = tips_march_to_july_september + tips_august_resort →
  total_tips = total_tips_resort + total_tips_delivery_driver →
  fraction_august = tips_august_resort / total_tips →
  fraction_august = 1 / 2 :=
by
  sorry

end david_tips_l73_73525


namespace probability_exactly_three_cured_l73_73516

 noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.fact n / (Nat.fact k * Nat.fact (n - k))

theorem probability_exactly_three_cured (p : ℝ) (n k : ℕ) (p_cure : p = 0.9) (n_pigs : n = 5) (k_cured : k = 3) :
  (combination n k) * p^k * (1 - p)^(n - k) = (combination 5 3) * 0.9^3 * 0.1^2 := 
begin
  -- proof steps go here
  sorry
end

end probability_exactly_three_cured_l73_73516


namespace correct_statements_eq_l73_73619

-- Definitions used in the Lean 4 statement should only directly appear in the conditions
variable {a b c : ℝ} 

-- Use the condition directly
theorem correct_statements_eq (h : a / c = b / c) (hc : c ≠ 0) : a = b := 
by
  -- This is where the proof would go
  sorry

end correct_statements_eq_l73_73619


namespace reciprocal_of_neg_two_l73_73056

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end reciprocal_of_neg_two_l73_73056


namespace product_of_repeating_decimal_l73_73666

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l73_73666


namespace area_of_region_enclosed_by_graph_l73_73839

noncomputable def area_of_enclosed_region : ℝ :=
  let x1 := 41.67
  let x2 := 62.5
  let y1 := 8.33
  let y2 := -8.33
  0.5 * (x2 - x1) * (y1 - y2)

theorem area_of_region_enclosed_by_graph :
  area_of_enclosed_region = 173.28 :=
sorry

end area_of_region_enclosed_by_graph_l73_73839


namespace expected_number_of_games_l73_73950
noncomputable def probability_of_A_winning (g : ℕ) : ℚ := 2 / 3
noncomputable def probability_of_B_winning (g : ℕ) : ℚ := 1 / 3
noncomputable def expected_games: ℚ := 266 / 81

theorem expected_number_of_games 
  (match_ends : ∀ g : ℕ, (∃ p1 p2 : ℕ, (p1 = g ∧ p2 = 0) ∨ (p1 = 0 ∧ p2 = g))) 
  (independent_outcomes : ∀ g1 g2 : ℕ, g1 ≠ g2 → probability_of_A_winning g1 * probability_of_A_winning g2 = (2 / 3) * (2 / 3) ∧ probability_of_B_winning g1 * probability_of_B_winning g2 = (1 / 3) * (1 / 3)) :
  (expected_games = 266 / 81) := 
sorry

end expected_number_of_games_l73_73950


namespace log_eq_condition_pq_l73_73822

theorem log_eq_condition_pq :
  ∀ (p q : ℝ), p > 0 → q > 0 → (Real.log p + Real.log q = Real.log (2 * p + q)) → p = 3 ∧ q = 3 :=
by
  intros p q hp hq hlog
  sorry

end log_eq_condition_pq_l73_73822


namespace rate_of_descent_correct_l73_73998

def depth := 3500 -- in feet
def time := 100 -- in minutes

def rate_of_descent : ℕ := depth / time

theorem rate_of_descent_correct : rate_of_descent = 35 := by
  -- We intentionally skip the proof part as per the requirement
  sorry

end rate_of_descent_correct_l73_73998


namespace systematic_sampling_number_l73_73723

theorem systematic_sampling_number {n m s a b c d : ℕ} (h_n : n = 60) (h_m : m = 4) 
  (h_s : s = 3) (h_a : a = 33) (h_b : b = 48) 
  (h_gcd_1 : ∃ k, s + k * (n / m) = a) (h_gcd_2 : ∃ k, a + k * (n / m) = b) :
  ∃ k, s + k * (n / m) = d → d = 18 := by
  sorry

end systematic_sampling_number_l73_73723


namespace number_of_ordered_pairs_l73_73343

theorem number_of_ordered_pairs (p q : ℂ) (h1 : p^4 * q^3 = 1) (h2 : p^8 * q = 1) : (∃ n : ℕ, n = 40) :=
sorry

end number_of_ordered_pairs_l73_73343


namespace set_A_enumeration_l73_73437

-- Define the conditions of the problem.
def A : Set ℕ := { x | ∃ (n : ℕ), 6 = n * (6 - x) }

-- State the theorem to be proved.
theorem set_A_enumeration : A = {0, 2, 3, 4, 5} :=
by
  sorry

end set_A_enumeration_l73_73437


namespace num_ways_choose_materials_l73_73457

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l73_73457


namespace cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l73_73034

-- Definitions related to the problem statements
def good_tetrahedron (V S : ℝ) := V = S

def good_parallelepiped (V' S1 S2 S3 : ℝ) := V' = 2 * (S1 + S2 + S3)

-- Theorem statement
theorem cannot_inscribe_good_tetrahedron_in_good_parallelepiped
  (V V' S : ℝ) (S1 S2 S3 : ℝ) (h1 h2 h3 : ℝ)
  (HT : good_tetrahedron V S)
  (HP : good_parallelepiped V' S1 S2 S3)
  (Hheights : S1 ≥ S2 ∧ S2 ≥ S3) :
  ¬ (V = S ∧ V' = 2 * (S1 + S2 + S3) ∧ h1 > 6 * S1 ∧ h2 > 6 * S2 ∧ h3 > 6 * S3) := 
sorry

end cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l73_73034
