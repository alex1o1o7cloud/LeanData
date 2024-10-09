import Mathlib

namespace ratio_lcm_gcf_240_360_l357_35759

theorem ratio_lcm_gcf_240_360 : Nat.lcm 240 360 / Nat.gcd 240 360 = 60 :=
by
  sorry

end ratio_lcm_gcf_240_360_l357_35759


namespace air_conditioning_price_november_l357_35727

noncomputable def price_in_november : ℝ :=
  let january_price := 470
  let february_price := january_price * (1 - 0.12)
  let march_price := february_price * (1 + 0.08)
  let april_price := march_price * (1 - 0.10)
  let june_price := april_price * (1 + 0.05)
  let august_price := june_price * (1 - 0.07)
  let october_price := august_price * (1 + 0.06)
  october_price * (1 - 0.15)

theorem air_conditioning_price_november : price_in_november = 353.71 := by
  sorry

end air_conditioning_price_november_l357_35727


namespace hcf_of_three_numbers_l357_35700

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : Nat.lcm a (Nat.lcm b c) = 45600) (h2 : a * b * c = 109183500000) :
  Nat.gcd a (Nat.gcd b c) = 2393750 := by
  sorry

end hcf_of_three_numbers_l357_35700


namespace range_of_a_l357_35794

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a*x^2 - 3*x + 2 = 0) ∧ 
  (∀ x y : ℝ, a*x^2 - 3*x + 2 = 0 ∧ a*y^2 - 3*y + 2 = 0 → x = y) 
  ↔ (a = 0 ∨ a = 9 / 8) := by sorry

end range_of_a_l357_35794


namespace pictures_left_l357_35790

def initial_zoo_pics : ℕ := 49
def initial_museum_pics : ℕ := 8
def deleted_pics : ℕ := 38

theorem pictures_left (total_pics : ℕ) :
  total_pics = initial_zoo_pics + initial_museum_pics →
  total_pics - deleted_pics = 19 :=
by
  intro h1
  rw [h1]
  sorry

end pictures_left_l357_35790


namespace carla_total_time_l357_35768

def total_time_spent (knife_time : ℕ) (peeling_time_multiplier : ℕ) : ℕ :=
  knife_time + peeling_time_multiplier * knife_time

theorem carla_total_time :
  total_time_spent 10 3 = 40 :=
by
  sorry

end carla_total_time_l357_35768


namespace solution_set_of_inequality_system_l357_35744

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 2 ≤ 3 ∧ 1 + x > -2) ↔ (-3 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_of_inequality_system_l357_35744


namespace weights_difference_l357_35713

-- Definitions based on conditions
def A : ℕ := 36
def ratio_part : ℕ := A / 4
def B : ℕ := 5 * ratio_part
def C : ℕ := 6 * ratio_part

-- Theorem to prove
theorem weights_difference :
  (A + C) - B = 45 := by
  sorry

end weights_difference_l357_35713


namespace decreasing_condition_l357_35765

variable (m : ℝ)

def quadratic_fn (x : ℝ) : ℝ := x^2 + m * x + 1

theorem decreasing_condition (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → (deriv (quadratic_fn m) x ≤ 0)) :
    m ≤ -10 := 
by
  -- Proof omitted
  sorry

end decreasing_condition_l357_35765


namespace quadratic_unique_solution_l357_35701

theorem quadratic_unique_solution (k : ℝ) (x : ℝ) :
  (16 ^ 2 - 4 * 2 * k * 4 = 0) → (k = 8 ∧ x = -1 / 2) :=
by
  sorry

end quadratic_unique_solution_l357_35701


namespace roses_carnations_price_comparison_l357_35757

variables (x y : ℝ)

theorem roses_carnations_price_comparison
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y :=
sorry

end roses_carnations_price_comparison_l357_35757


namespace apples_given_by_Susan_l357_35791

theorem apples_given_by_Susan (x y final_apples : ℕ) (h1 : y = 9) (h2 : final_apples = 17) (h3: final_apples = y + x) : x = 8 := by
  sorry

end apples_given_by_Susan_l357_35791


namespace determinant_of_roots_l357_35767

noncomputable def determinant_expr (a b c d s p q r : ℝ) : ℝ :=
  by sorry

theorem determinant_of_roots (a b c d s p q r : ℝ)
    (h1 : a + b + c + d = -s)
    (h2 : abcd = r)
    (h3 : abc + abd + acd + bcd = -q)
    (h4 : ab + ac + bc = p) :
    determinant_expr a b c d s p q r = r - q + pq + p :=
  by sorry

end determinant_of_roots_l357_35767


namespace inequality_solution_sets_l357_35772

variable (a x : ℝ)

theorem inequality_solution_sets:
    ({x | 12 * x^2 - a * x > a^2} =
        if a > 0 then {x | x < -a/4} ∪ {x | x > a/3}
        else if a = 0 then {x | x ≠ 0}
        else {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end inequality_solution_sets_l357_35772


namespace rectangle_area_l357_35711

-- Define the given dimensions
def length : ℝ := 1.5
def width : ℝ := 0.75
def expected_area : ℝ := 1.125

-- State the problem
theorem rectangle_area (l w : ℝ) (h_l : l = length) (h_w : w = width) : l * w = expected_area :=
by sorry

end rectangle_area_l357_35711


namespace binomial_square_correct_k_l357_35761

theorem binomial_square_correct_k (k : ℚ) : (∃ t u : ℚ, k = t^2 ∧ 28 = 2 * t * u ∧ 9 = u^2) → k = 196 / 9 :=
by
  sorry

end binomial_square_correct_k_l357_35761


namespace Merry_sold_470_apples_l357_35725

-- Define the conditions
def boxes_on_Saturday : Nat := 50
def boxes_on_Sunday : Nat := 25
def apples_per_box : Nat := 10
def boxes_left : Nat := 3

-- Define the question as the number of apples sold
theorem Merry_sold_470_apples :
  (boxes_on_Saturday - boxes_on_Sunday) * apples_per_box +
  (boxes_on_Sunday - boxes_left) * apples_per_box = 470 := by
  sorry

end Merry_sold_470_apples_l357_35725


namespace records_given_l357_35736

theorem records_given (X : ℕ) (started_with : ℕ) (bought : ℕ) (days_per_record : ℕ) (total_days : ℕ)
  (h1 : started_with = 8) (h2 : bought = 30) (h3 : days_per_record = 2) (h4 : total_days = 100) :
  X = 12 := by
  sorry

end records_given_l357_35736


namespace capsules_per_bottle_l357_35739

-- Translating conditions into Lean definitions
def days := 180
def daily_serving_size := 2
def total_bottles := 6
def total_capsules_required := days * daily_serving_size

-- The statement to prove
theorem capsules_per_bottle : total_capsules_required / total_bottles = 60 :=
by
  sorry

end capsules_per_bottle_l357_35739


namespace average_exp_Feb_to_Jul_l357_35733

theorem average_exp_Feb_to_Jul (x y z : ℝ) 
    (h1 : 1200 + x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) = 6 * 4200) 
    (h2 : 0 ≤ x) 
    (h3 : 0 ≤ z) : 
    (x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) + 1500) / 6 = 4250 :=
by
    sorry

end average_exp_Feb_to_Jul_l357_35733


namespace smallest_value_of_Q_l357_35773

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 7*x^2 - 2*x + 10

theorem smallest_value_of_Q :
  min (Q 1) (min (10 : ℝ) (min (4 : ℝ) (min (1 - 4 + 7 - 2 + 10 : ℝ) (2.5 : ℝ)))) = 2.5 :=
by sorry

end smallest_value_of_Q_l357_35773


namespace probability_team_A_3_points_probability_team_A_1_point_probability_combined_l357_35732

namespace TeamProbabilities

noncomputable def P_team_A_3_points : ℚ :=
  (1 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_A_1_point : ℚ :=
  (1 / 3) * (2 / 3) * (2 / 3) + (2 / 3) * (1 / 3) * (2 / 3) + (2 / 3) * (2 / 3) * (1 / 3)

noncomputable def P_team_A_2_points : ℚ :=
  (1 / 3) * (1 / 3) * (2 / 3) + (1 / 3) * (2 / 3) * (1 / 3) + (2 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_B_1_point : ℚ :=
  (1 / 2) * (2 / 3) * (3 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (1 / 2) * (2 / 3) * (1 / 4) + (1 / 2) * (2 / 3) * (1 / 4) +
  (1 / 2) * (1 / 3) * (1 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (2 / 3) * (2 / 3) * (1 / 4) + (2 / 3) * (1 / 3) * (1 / 4)

noncomputable def combined_probability : ℚ :=
  P_team_A_2_points * P_team_B_1_point

theorem probability_team_A_3_points :
  P_team_A_3_points = 1 / 27 := by
  sorry

theorem probability_team_A_1_point :
  P_team_A_1_point = 4 / 9 := by
  sorry

theorem probability_combined :
  combined_probability = 11 / 108 := by
  sorry

end TeamProbabilities

end probability_team_A_3_points_probability_team_A_1_point_probability_combined_l357_35732


namespace investment_amount_l357_35729

noncomputable def total_investment (A T : ℝ) : Prop :=
  (0.095 * T = 0.09 * A + 2750) ∧ (T = A + 25000)

theorem investment_amount :
  ∃ T, ∀ A, total_investment A T ∧ T = 100000 :=
by
  sorry

end investment_amount_l357_35729


namespace seokgi_jumped_furthest_l357_35743

noncomputable def yooseung_jump : ℝ := 15 / 8
def shinyoung_jump : ℝ := 2
noncomputable def seokgi_jump : ℝ := 17 / 8

theorem seokgi_jumped_furthest :
  yooseung_jump < seokgi_jump ∧ shinyoung_jump < seokgi_jump :=
by
  sorry

end seokgi_jumped_furthest_l357_35743


namespace fraction_power_four_l357_35728

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end fraction_power_four_l357_35728


namespace max_a_for_three_solutions_l357_35778

-- Define the equation as a Lean function
def equation (x a : ℝ) : ℝ :=
  (|x-2| + 2 * a)^2 - 3 * (|x-2| + 2 * a) + 4 * a * (3 - 4 * a)

-- Statement of the proof problem
theorem max_a_for_three_solutions :
  (∃ (a : ℝ), (∀ x : ℝ, equation x a = 0) ∧
  (∀ (b : ℝ), (∀ x : ℝ, equation x b = 0) → b ≤ 0.5)) :=
sorry

end max_a_for_three_solutions_l357_35778


namespace friends_truth_l357_35740

-- Definitions for the truth values of the friends
def F₁_truth (a x₁ x₂ x₃ : Prop) : Prop := a ↔ ¬ (x₁ ∨ x₂ ∨ x₃)
def F₂_truth (b x₁ x₂ x₃ : Prop) : Prop := b ↔ (x₂ ∧ ¬ x₁ ∧ ¬ x₃)
def F₃_truth (c x₁ x₂ x₃ : Prop) : Prop := c ↔ x₃

-- Main theorem statement
theorem friends_truth (a b c x₁ x₂ x₃ : Prop) 
  (H₁ : F₁_truth a x₁ x₂ x₃) 
  (H₂ : F₂_truth b x₁ x₂ x₃) 
  (H₃ : F₃_truth c x₁ x₂ x₃)
  (H₄ : a ∨ b ∨ c) 
  (H₅ : ¬ (a ∧ b ∧ c)) : a ∧ ¬b ∧ ¬c ∨ ¬a ∧ b ∧ ¬c ∨ ¬a ∧ ¬b ∧ c :=
sorry

end friends_truth_l357_35740


namespace katie_five_dollar_bills_l357_35714

theorem katie_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end katie_five_dollar_bills_l357_35714


namespace number_of_hydrogen_atoms_l357_35750

/-- 
A compound has a certain number of Hydrogen, 1 Chromium, and 4 Oxygen atoms. 
The molecular weight of the compound is 118. How many Hydrogen atoms are in the compound?
-/
theorem number_of_hydrogen_atoms
  (H Cr O : ℕ)
  (mw_H : ℕ := 1)
  (mw_Cr : ℕ := 52)
  (mw_O : ℕ := 16)
  (H_weight : ℕ := H * mw_H)
  (Cr_weight : ℕ := 1 * mw_Cr)
  (O_weight : ℕ := 4 * mw_O)
  (total_weight : ℕ := 118)
  (weight_without_H : ℕ := Cr_weight + O_weight) 
  (H_weight_calculated : ℕ := total_weight - weight_without_H) :
  H = 2 :=
  by
    sorry

end number_of_hydrogen_atoms_l357_35750


namespace second_pipe_fill_time_l357_35712

theorem second_pipe_fill_time
  (rate1: ℝ) (rate_outlet: ℝ) (combined_time: ℝ)
  (h1: rate1 = 1 / 18)
  (h2: rate_outlet = 1 / 45)
  (h_combined: combined_time = 0.05):
  ∃ (x: ℝ), (1 / x) = 60 :=
by
  sorry

end second_pipe_fill_time_l357_35712


namespace solve_equation_l357_35766

theorem solve_equation : ∀ x y : ℤ, x^2 + y^2 = 3 * x * y → x = 0 ∧ y = 0 := by
  intros x y h
  sorry

end solve_equation_l357_35766


namespace integer_solution_for_x_l357_35793

theorem integer_solution_for_x (x : ℤ) : 
  (∃ y z : ℤ, x = 7 * y + 3 ∧ x = 5 * z + 2) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by
  sorry

end integer_solution_for_x_l357_35793


namespace fraction_of_network_advertisers_l357_35735

theorem fraction_of_network_advertisers 
  (total_advertisers : ℕ := 20) 
  (percentage_from_uni_a : ℝ := 0.75)
  (advertisers_from_uni_a := total_advertisers * percentage_from_uni_a) :
  (advertisers_from_uni_a / total_advertisers) = (3 / 4) :=
by
  sorry

end fraction_of_network_advertisers_l357_35735


namespace shortest_side_of_right_triangle_l357_35745

theorem shortest_side_of_right_triangle
  (a b c : ℝ)
  (h : a = 5) (k : b = 13) (rightangled : a^2 + c^2 = b^2) : c = 12 := 
sorry

end shortest_side_of_right_triangle_l357_35745


namespace train_speed_kmh_l357_35782

-- Definitions based on the conditions
variables (L V : ℝ)
variable (h1 : L = 10 * V)
variable (h2 : L + 600 = 30 * V)

-- The proof statement, no solution steps, just the conclusion
theorem train_speed_kmh : (V * 3.6) = 108 :=
by
  sorry

end train_speed_kmh_l357_35782


namespace find_c_l357_35771

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x : ℝ, g_inv (g x c) = x) -> c = 3 :=
by 
  intro h
  sorry

end find_c_l357_35771


namespace total_cost_first_3_years_l357_35786

def monthly_fee : ℕ := 12
def down_payment : ℕ := 50
def years : ℕ := 3

theorem total_cost_first_3_years :
  (years * 12 * monthly_fee + down_payment) = 482 :=
by
  sorry

end total_cost_first_3_years_l357_35786


namespace gwen_points_per_bag_l357_35749

theorem gwen_points_per_bag : 
  ∀ (total_bags recycled_bags total_points_per_bag points_per_bag : ℕ),
  total_bags = 4 → 
  recycled_bags = total_bags - 2 →
  total_points_per_bag = 16 →
  points_per_bag = (total_points_per_bag / total_bags) →
  points_per_bag = 4 :=
by
  intros
  sorry

end gwen_points_per_bag_l357_35749


namespace range_of_m_l357_35723

theorem range_of_m (x m : ℝ)
  (h1 : (x + 2) / (10 - x) ≥ 0)
  (h2 : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (h3 : m < 0)
  (h4 : ∀ (x : ℝ), (x + 2) / (10 - x) ≥ 0 → (x^2 - 2 * x + 1 - m^2 ≤ 0)) :
  -3 ≤ m ∧ m < 0 :=
sorry

end range_of_m_l357_35723


namespace find_a_l357_35796

noncomputable def g (x : ℝ) := 5 * x - 7

theorem find_a (a : ℝ) (h : g a = 0) : a = 7 / 5 :=
sorry

end find_a_l357_35796


namespace numbers_lcm_sum_l357_35752

theorem numbers_lcm_sum :
  ∃ A : List ℕ, A.length = 100 ∧
    (A.count 1 = 89 ∧ A.count 2 = 8 ∧ [4, 5, 6] ⊆ A) ∧
    A.sum = A.foldr lcm 1 :=
by
  sorry

end numbers_lcm_sum_l357_35752


namespace damage_conversion_l357_35741

def usd_to_cad_conversion_rate : ℝ := 1.25
def damage_in_usd : ℝ := 60000000
def damage_in_cad : ℝ := 75000000

theorem damage_conversion :
  damage_in_usd * usd_to_cad_conversion_rate = damage_in_cad :=
sorry

end damage_conversion_l357_35741


namespace ratio_arithmetic_sequence_triangle_l357_35704

theorem ratio_arithmetic_sequence_triangle (a b c : ℝ) 
  (h_triangle : a^2 + b^2 = c^2)
  (h_arith_seq : ∃ d, b = a + d ∧ c = a + 2 * d) :
  a / b = 3 / 4 ∧ b / c = 4 / 5 :=
by
  sorry

end ratio_arithmetic_sequence_triangle_l357_35704


namespace inverse_f_neg_3_l357_35779

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_neg_3 : (∃ x : ℝ, f x = -3) ∧ (f 4 = -3) :=
by
  sorry

end inverse_f_neg_3_l357_35779


namespace quad_roots_expression_l357_35758

theorem quad_roots_expression (x1 x2 : ℝ) (h1 : x1 * x1 + 2019 * x1 + 1 = 0) (h2 : x2 * x2 + 2019 * x2 + 1 = 0) :
  x1 * x2 - x1 - x2 = 2020 :=
sorry

end quad_roots_expression_l357_35758


namespace area_of_dodecagon_l357_35775

theorem area_of_dodecagon (r : ℝ) : 
  ∃ A : ℝ, (∃ n : ℕ, n = 12) ∧ (A = 3 * r^2) := 
by
  sorry

end area_of_dodecagon_l357_35775


namespace area_AKM_less_than_area_ABC_l357_35784

-- Define the rectangle ABCD
structure Rectangle :=
(A B C D : ℝ) -- Four vertices of the rectangle
(side_AB : ℝ) (side_BC : ℝ) (side_CD : ℝ) (side_DA : ℝ)

-- Define the arbitrary points K and M on sides BC and CD respectively
variables (B C D K M : ℝ)

-- Define the area of triangle function and area of rectangle function
def area_triangle (A B C : ℝ) : ℝ := sorry -- Assuming a function calculating area of triangle given 3 vertices
def area_rectangle (A B C D : ℝ) : ℝ := sorry -- Assuming a function calculating area of rectangle given 4 vertices

-- Assuming the conditions given in the problem statement
variables (A : ℝ) (rect : Rectangle)

-- Prove that the area of triangle AKM is less than the area of triangle ABC
theorem area_AKM_less_than_area_ABC : 
  ∀ (K M : ℝ), K ∈ [B,C] → M ∈ [C,D] →
    area_triangle A K M < area_triangle A B C := sorry

end area_AKM_less_than_area_ABC_l357_35784


namespace polynomial_expansion_l357_35753

theorem polynomial_expansion :
  (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
  35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := 
by
  sorry

end polynomial_expansion_l357_35753


namespace part1_part2_l357_35788

noncomputable def f (a x : ℝ) := a * x^2 - (a + 1) * x + 1

theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ (-3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2) :=
sorry

theorem part2 (a : ℝ) (h1 : a ≠ 0) (x : ℝ) :
  (f a x < 0) ↔
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
     (a = 1 ∧ false) ∨
     (a > 1 ∧ 1 / a < x ∧ x < 1) ∨
     (a < 0 ∧ (x < 1 / a ∨ x > 1))) :=
sorry

end part1_part2_l357_35788


namespace inequality_proof_l357_35719

-- Defining the conditions
variable (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (cond : 1 / a + 1 / b = 1)

-- Defining the theorem to be proved
theorem inequality_proof (n : ℕ) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by
  sorry

end inequality_proof_l357_35719


namespace simple_fraction_pow_l357_35722

theorem simple_fraction_pow : (66666^4 / 22222^4) = 81 := by
  sorry

end simple_fraction_pow_l357_35722


namespace radius_moon_scientific_notation_l357_35731

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end radius_moon_scientific_notation_l357_35731


namespace linear_inequality_solution_l357_35708

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end linear_inequality_solution_l357_35708


namespace infinite_hexagons_exist_l357_35795

theorem infinite_hexagons_exist :
  ∃ (a1 a2 a3 a4 a5 a6 : ℤ), 
  (a1 + a2 + a3 + a4 + a5 + a6 = 20) ∧
  (a1 ≤ a2) ∧ (a1 + a2 ≤ a3) ∧ (a2 + a3 ≤ a4) ∧
  (a3 + a4 ≤ a5) ∧ (a4 + a5 ≤ a6) ∧ (a1 + a2 + a3 + a4 + a5 > a6) :=
sorry

end infinite_hexagons_exist_l357_35795


namespace find_function_l357_35706

theorem find_function (f : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, f n < f (n + 1)) →
  (∀ n : ℕ, f (f n) = n + 2 * k) →
  ∀ n : ℕ, f n = n + k := 
by
  intro h1 h2
  sorry

end find_function_l357_35706


namespace find_pairs_l357_35737

def is_prime (p : ℕ) : Prop := (p ≥ 2) ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_pairs (a p : ℕ) (h_pos_a : a > 0) (h_prime_p : is_prime p) :
  (∀ m n : ℕ, 0 < m → 0 < n → (a ^ (2 ^ n) % p ^ n = a ^ (2 ^ m) % p ^ m ∧ a ^ (2 ^ n) % p ^ n ≠ 0))
  ↔ (∃ k : ℕ, a = 2 * k + 1 ∧ p = 2) :=
sorry

end find_pairs_l357_35737


namespace rook_reaches_upper_right_in_expected_70_minutes_l357_35748

section RookMoves

noncomputable def E : ℝ := 70

-- Definition of expected number of minutes considering the row and column moves.
-- This is a direct translation from the problem's correct answer.
def rook_expected_minutes_to_upper_right (E_0 E_1 : ℝ) : Prop :=
  E_0 = (70 : ℝ) ∧ E_1 = (70 : ℝ)

theorem rook_reaches_upper_right_in_expected_70_minutes : E = 70 := sorry

end RookMoves

end rook_reaches_upper_right_in_expected_70_minutes_l357_35748


namespace find_angle_EHG_l357_35742

noncomputable def angle_EHG (angle_EFG : ℝ) (angle_GHE : ℝ) : ℝ := angle_GHE - angle_EFG
 
theorem find_angle_EHG : 
  ∀ (EF GH : Prop) (angle_EFG angle_GHE : ℝ), (EF ∧ GH) → 
    EF ∧ GH ∧ angle_EFG = 50 ∧ angle_GHE = 80 → angle_EHG angle_EFG angle_GHE = 30 := 
by 
  intros EF GH angle_EFG angle_GHE h1 h2
  sorry

end find_angle_EHG_l357_35742


namespace decimal_to_binary_thirteen_l357_35799

theorem decimal_to_binary_thirteen : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_thirteen_l357_35799


namespace correct_statement_l357_35710

def synthetic_method_is_direct : Prop := -- define the synthetic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

def analytic_method_is_direct : Prop := -- define the analytic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

theorem correct_statement : synthetic_method_is_direct ∧ analytic_method_is_direct → 
                             "Synthetic method and analytic method are direct proof methods" = "A" :=
by
  intros h
  cases h
  -- This is where you would provide the proof steps. We skip this with sorry.
  sorry

end correct_statement_l357_35710


namespace sum_of_constants_l357_35718

variable (a b c : ℝ)

theorem sum_of_constants (h :  2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) :
  a + b + c = 9 := 
sorry

end sum_of_constants_l357_35718


namespace alix_has_15_more_chocolates_than_nick_l357_35798

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l357_35798


namespace product_of_m_and_u_l357_35777

noncomputable def g : ℝ → ℝ := sorry

axiom g_conditions : (∀ x y : ℝ, g (x^2 - y^2) = (x - y) * ((g x) ^ 3 + (g y) ^ 3)) ∧ (g 1 = 1)

def m : ℕ := sorry
def u : ℝ := sorry

theorem product_of_m_and_u : m * u = 3 :=
by 
  -- all conditions about 'g' are assumed as axioms and not directly included in the proof steps
  exact sorry

end product_of_m_and_u_l357_35777


namespace polynomial_root_cubic_sum_l357_35754

theorem polynomial_root_cubic_sum
  (a b c : ℝ)
  (h : ∀ x : ℝ, (Polynomial.eval x (3 * Polynomial.X^3 + 5 * Polynomial.X^2 - 150 * Polynomial.X + 7) = 0)
    → x = a ∨ x = b ∨ x = c) :
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 :=
  sorry

end polynomial_root_cubic_sum_l357_35754


namespace farmer_apples_l357_35756

theorem farmer_apples : 127 - 39 = 88 := by
  -- Skipping proof details
  sorry

end farmer_apples_l357_35756


namespace tom_seashells_l357_35715

theorem tom_seashells (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) : total_seashells = 35 := 
by 
  sorry

end tom_seashells_l357_35715


namespace unit_circle_sector_arc_length_l357_35774

theorem unit_circle_sector_arc_length (r S l : ℝ) (h1 : r = 1) (h2 : S = 1) (h3 : S = 1 / 2 * l * r) : l = 2 :=
by
  sorry

end unit_circle_sector_arc_length_l357_35774


namespace mowing_time_l357_35760

/-- 
Rena uses a mower to trim her "L"-shaped lawn which consists of two rectangular sections 
sharing one $50$-foot side. One section is $120$-foot by $50$-foot and the other is $70$-foot by 
$50$-foot. The mower has a swath width of $35$ inches with overlaps by $5$ inches. 
Rena walks at the rate of $4000$ feet per hour. 
Prove that it takes 0.95 hours for Rena to mow the entire lawn.
-/
theorem mowing_time 
  (length1 length2 width mower_swath overlap : ℝ) 
  (Rena_speed : ℝ) (effective_swath : ℝ) (total_area total_strips total_distance : ℝ)
  (h1 : length1 = 120)
  (h2 : length2 = 70)
  (h3 : width = 50)
  (h4 : mower_swath = 35 / 12)
  (h5 : overlap = 5 / 12)
  (h6 : effective_swath = mower_swath - overlap)
  (h7 : Rena_speed = 4000)
  (h8 : total_area = length1 * width + length2 * width)
  (h9 : total_strips = (length1 + length2) / effective_swath)
  (h10 : total_distance = total_strips * width) : 
  (total_distance / Rena_speed = 0.95) :=
by sorry

end mowing_time_l357_35760


namespace simplify_expression_l357_35702

theorem simplify_expression :
  ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := 
by
  sorry

end simplify_expression_l357_35702


namespace max_time_for_taxiing_is_15_l357_35724

-- Declare the function representing the distance traveled by the plane with respect to time
def distance (t : ℝ) : ℝ := 60 * t - 2 * t ^ 2

-- The main theorem stating the maximum time s the plane uses for taxiing
theorem max_time_for_taxiing_is_15 : ∃ s, ∀ t, distance t ≤ distance s ∧ s = 15 :=
by
  sorry

end max_time_for_taxiing_is_15_l357_35724


namespace sequence_fifth_term_l357_35705

theorem sequence_fifth_term (a b c : ℕ) :
  (a = (2 + b) / 3) →
  (b = (a + 34) / 3) →
  (34 = (b + c) / 3) →
  c = 89 :=
by
  intros ha hb hc
  sorry

end sequence_fifth_term_l357_35705


namespace math_problem_l357_35770

variable (x y : ℝ)

theorem math_problem (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0) (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) :
  x * y - 12 * x + 15 * y = 0 :=
  sorry

end math_problem_l357_35770


namespace compute_a_l357_35730

theorem compute_a (a : ℝ) (h : 2.68 * 0.74 = a) : a = 1.9832 :=
by
  -- Here skip the proof steps
  sorry

end compute_a_l357_35730


namespace evaluate_expression_is_sixth_l357_35721

noncomputable def evaluate_expression := (1 / Real.log 3000^4 / Real.log 8) + (4 / Real.log 3000^4 / Real.log 9)

theorem evaluate_expression_is_sixth:
  evaluate_expression = 1 / 6 :=
  by
  sorry

end evaluate_expression_is_sixth_l357_35721


namespace midpoint_distance_from_school_l357_35747

def distance_school_kindergarten_km := 1
def distance_school_kindergarten_m := 700
def distance_kindergarten_house_m := 900

theorem midpoint_distance_from_school : 
  (1000 * distance_school_kindergarten_km + distance_school_kindergarten_m + distance_kindergarten_house_m) / 2 = 1300 := 
by
  sorry

end midpoint_distance_from_school_l357_35747


namespace cody_spent_tickets_l357_35751

theorem cody_spent_tickets (initial_tickets lost_tickets remaining_tickets : ℝ) (h1 : initial_tickets = 49.0) (h2 : lost_tickets = 6.0) (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by
  sorry

end cody_spent_tickets_l357_35751


namespace growth_operation_two_operations_growth_operation_four_operations_l357_35789

noncomputable def growth_operation_perimeter (initial_side_length : ℕ) (growth_operations : ℕ) := 
  initial_side_length * 3 * (4/3 : ℚ)^(growth_operations + 1)

theorem growth_operation_two_operations :
  growth_operation_perimeter 9 2 = 48 := by sorry

theorem growth_operation_four_operations :
  growth_operation_perimeter 9 4 = 256 / 3 := by sorry

end growth_operation_two_operations_growth_operation_four_operations_l357_35789


namespace distinct_sums_is_98_l357_35787

def arithmetic_sequence_distinct_sums (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :=
  (∀ n : ℕ, S n = (n * (2 * a_n 0 + (n - 1) * d)) / 2) ∧
  S 5 = 0 ∧
  d ≠ 0 →
  (∃ distinct_count : ℕ, distinct_count = 98 ∧
   ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 ∧ S i = S j → i = j)

theorem distinct_sums_is_98 (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence_distinct_sums a_n S d) :
  ∃ distinct_count : ℕ, distinct_count = 98 :=
sorry

end distinct_sums_is_98_l357_35787


namespace survey_total_people_l357_35762

theorem survey_total_people (number_represented : ℕ) (percentage : ℝ) (h : number_represented = percentage * 200) : 
  (number_represented : ℝ) = 200 := 
by 
 sorry

end survey_total_people_l357_35762


namespace train_travel_section_marked_l357_35755

-- Definition of the metro structure with the necessary conditions.
structure Metro (Station : Type) :=
  (lines : List (Station × Station))
  (travel_time : Station → Station → ℕ)
  (terminal_turnaround : Station → Station)
  (transfer_station : Station → Station)

variable {Station : Type}

/-- The function that defines the bipolar coloring of the metro stations. -/
def station_color (s : Station) : ℕ := sorry  -- Placeholder for actual coloring function.

theorem train_travel_section_marked 
  (metro : Metro Station)
  (initial_station : Station)
  (end_station : Station)
  (travel_time : ℕ)
  (marked_section : Station × Station)
  (h_start : initial_station = marked_section.fst)
  (h_end : end_station = marked_section.snd)
  (h_travel_time : travel_time = 2016)
  (h_condition : ∀ s1 s2, (s1, s2) ∈ metro.lines → metro.travel_time s1 s2 = 1 ∧ 
                metro.terminal_turnaround s1 ≠ s1 ∧ metro.transfer_station s1 ≠ s2) :
  ∃ (time : ℕ), time = 2016 ∧ ∃ s1 s2, (s1, s2) = marked_section :=
sorry

end train_travel_section_marked_l357_35755


namespace smallest_of_three_consecutive_even_numbers_l357_35769

def sum_of_three_consecutive_even_numbers (n : ℕ) : Prop :=
  n + (n + 2) + (n + 4) = 162

theorem smallest_of_three_consecutive_even_numbers (n : ℕ) (h : sum_of_three_consecutive_even_numbers n) : n = 52 :=
by
  sorry

end smallest_of_three_consecutive_even_numbers_l357_35769


namespace breadth_of_rectangular_plot_l357_35763

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : ∃ l : ℝ, l = 3 * b) (h2 : b * 3 * b = 675) : b = 15 :=
by
  sorry

end breadth_of_rectangular_plot_l357_35763


namespace range_of_a_l357_35726

variable {R : Type} [LinearOrderedField R]

def is_even (f : R → R) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_non_neg (f : R → R) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem range_of_a 
  (f : R → R) 
  (even_f : is_even f)
  (mono_f : is_monotone_increasing_on_non_neg f)
  (ineq : ∀ a, f (a + 1) ≤ f 4) : 
  ∀ a, -5 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l357_35726


namespace sum_invested_7000_l357_35780

-- Define the conditions
def interest_15 (P : ℝ) : ℝ := P * 0.15 * 2
def interest_12 (P : ℝ) : ℝ := P * 0.12 * 2

-- Main statement to prove
theorem sum_invested_7000 (P : ℝ) (h : interest_15 P - interest_12 P = 420) : P = 7000 := by
  sorry

end sum_invested_7000_l357_35780


namespace find_asymptote_slope_l357_35781

theorem find_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 0) → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end find_asymptote_slope_l357_35781


namespace ratio_equality_l357_35797

theorem ratio_equality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h8 : (x + 1) / y = (y + 1) / (x + z)) :
  (x + 1) / y = 1 :=
by
  sorry

end ratio_equality_l357_35797


namespace polygon_sides_l357_35703

theorem polygon_sides (n : ℕ) : (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end polygon_sides_l357_35703


namespace charlie_pennies_l357_35783

variable (a c : ℕ)

theorem charlie_pennies (h1 : c + 1 = 4 * (a - 1)) (h2 : c - 1 = 3 * (a + 1)) : c = 31 := 
by
  sorry

end charlie_pennies_l357_35783


namespace gift_exchange_equation_l357_35738

theorem gift_exchange_equation (x : ℕ) (h : x * (x - 1) = 40) : 
  x * (x - 1) = 40 :=
by
  exact h

end gift_exchange_equation_l357_35738


namespace henry_games_total_l357_35792

theorem henry_games_total
    (wins : ℕ)
    (losses : ℕ)
    (draws : ℕ)
    (hw : wins = 2)
    (hl : losses = 2)
    (hd : draws = 10) :
  wins + losses + draws = 14 :=
by
  -- The proof is omitted.
  sorry

end henry_games_total_l357_35792


namespace mean_of_set_l357_35709

theorem mean_of_set (x y : ℝ) 
  (h : (28 + x + 50 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 :=
by
  -- we would now proceed to prove this according to lean's proof tactics.
  sorry

end mean_of_set_l357_35709


namespace black_greater_than_gray_by_103_l357_35776

def a := 12
def b := 9
def c := 7
def d := 3

def area (side: ℕ) := side * side

def black_area_sum : ℕ := area a + area c
def gray_area_sum : ℕ := area b + area d

theorem black_greater_than_gray_by_103 :
  black_area_sum - gray_area_sum = 103 := by
  sorry

end black_greater_than_gray_by_103_l357_35776


namespace min_value_ab_min_value_a_plus_2b_l357_35720
open Nat

theorem min_value_ab (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 8 ≤ a * b :=
by
  sorry

theorem min_value_a_plus_2b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 9 ≤ a + 2 * b :=
by
  sorry

end min_value_ab_min_value_a_plus_2b_l357_35720


namespace triangle_existence_l357_35707

theorem triangle_existence 
  (h_a h_b m_a : ℝ) :
  (m_a ≥ h_a) → 
  ((h_a > 1/2 * h_b ∧ m_a > h_a → true ∨ false) ∧ 
  (m_a = h_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b < m_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b = m_a → false ∨ true) ∧ 
  (1/2 * h_b > m_a → false)) :=
by
  intro
  sorry

end triangle_existence_l357_35707


namespace side_of_beef_weight_after_processing_l357_35734

theorem side_of_beef_weight_after_processing (initial_weight : ℝ) (lost_percentage : ℝ) (final_weight : ℝ) 
  (h1 : initial_weight = 400) 
  (h2 : lost_percentage = 0.4) 
  (h3 : final_weight = initial_weight * (1 - lost_percentage)) : 
  final_weight = 240 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end side_of_beef_weight_after_processing_l357_35734


namespace complex_fraction_simplify_l357_35716

variable (i : ℂ)
variable (h : i^2 = -1)

theorem complex_fraction_simplify :
  (1 - i) / ((1 + i) ^ 2) = -1/2 - i/2 :=
by
  sorry

end complex_fraction_simplify_l357_35716


namespace Chloe_second_round_points_l357_35746

-- Conditions
def firstRoundPoints : ℕ := 40
def lastRoundPointsLost : ℕ := 4
def totalPoints : ℕ := 86
def secondRoundPoints : ℕ := 50

-- Statement to prove: Chloe scored 50 points in the second round
theorem Chloe_second_round_points :
  firstRoundPoints + secondRoundPoints - lastRoundPointsLost = totalPoints :=
by {
  -- Proof (not required, skipping with sorry)
  sorry
}

end Chloe_second_round_points_l357_35746


namespace repeating_decimal_sum_l357_35717

noncomputable def a : ℚ := 0.66666667 -- Repeating decimal 0.666... corresponds to 2/3
noncomputable def b : ℚ := 0.22222223 -- Repeating decimal 0.222... corresponds to 2/9
noncomputable def c : ℚ := 0.44444445 -- Repeating decimal 0.444... corresponds to 4/9
noncomputable def d : ℚ := 0.99999999 -- Repeating decimal 0.999... corresponds to 1

theorem repeating_decimal_sum : a + b - c + d = 13 / 9 := by
  sorry

end repeating_decimal_sum_l357_35717


namespace abc_sum_is_12_l357_35785

theorem abc_sum_is_12
  (a b c : ℕ)
  (h : 28 * a + 30 * b + 31 * c = 365) :
  a + b + c = 12 :=
by
  sorry

end abc_sum_is_12_l357_35785


namespace total_number_of_sheep_l357_35764

theorem total_number_of_sheep (a₁ a₂ a₃ a₄ a₅ a₆ a₇ d : ℤ)
    (h1 : a₂ = a₁ + d)
    (h2 : a₃ = a₁ + 2 * d)
    (h3 : a₄ = a₁ + 3 * d)
    (h4 : a₅ = a₁ + 4 * d)
    (h5 : a₆ = a₁ + 5 * d)
    (h6 : a₇ = a₁ + 6 * d)
    (h_sum : a₁ + a₂ + a₃ = 33)
    (h_seven: 2 * a₂ + 9 = a₇) :
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 133 := sorry

end total_number_of_sheep_l357_35764
