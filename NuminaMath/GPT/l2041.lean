import Mathlib

namespace range_of_m_l2041_204150

noncomputable def proposition_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 + m * x + 1 ≥ 0

noncomputable def proposition_q (m : ℝ) : Prop :=
∀ x : ℝ, (8 * x + 4 * (m - 1)) ≥ 0

def conditions (m : ℝ) : Prop :=
(proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m)

theorem range_of_m (m : ℝ) : 
  conditions m → ( -2 ≤ m ∧ m < 1 ) ∨ m > 2 :=
by
  intros h
  sorry

end range_of_m_l2041_204150


namespace mandy_total_cost_after_discount_l2041_204177

-- Define the conditions
def packs_black_shirts : ℕ := 6
def packs_yellow_shirts : ℕ := 8
def packs_green_socks : ℕ := 5

def items_per_pack_black_shirts : ℕ := 7
def items_per_pack_yellow_shirts : ℕ := 4
def items_per_pack_green_socks : ℕ := 5

def cost_per_pack_black_shirts : ℕ := 25
def cost_per_pack_yellow_shirts : ℕ := 15
def cost_per_pack_green_socks : ℕ := 10

def discount_rate : ℚ := 0.10

-- Calculate the total number of each type of item
def total_black_shirts : ℕ := packs_black_shirts * items_per_pack_black_shirts
def total_yellow_shirts : ℕ := packs_yellow_shirts * items_per_pack_yellow_shirts
def total_green_socks : ℕ := packs_green_socks * items_per_pack_green_socks

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ :=
  (packs_black_shirts * cost_per_pack_black_shirts) +
  (packs_yellow_shirts * cost_per_pack_yellow_shirts) +
  (packs_green_socks * cost_per_pack_green_socks)

-- Calculate the total cost after discount
def discount_amount : ℚ := discount_rate * total_cost_before_discount
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount

-- Problem to prove: Total cost after discount is $288
theorem mandy_total_cost_after_discount : total_cost_after_discount = 288 := by
  sorry

end mandy_total_cost_after_discount_l2041_204177


namespace part2_l2041_204197

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem part2 (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f 1 x1 = f 1 x2) : x1 + x2 > 2 := by
  have f_x1 := h2
  sorry

end part2_l2041_204197


namespace sum_of_roots_of_quadratic_l2041_204151

theorem sum_of_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (Polynomial.eval x1 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) ∧ 
                 (Polynomial.eval x2 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) -> 
                 x1 + x2 = 3 := 
by
  intro x1 x2
  intro H
  sorry

end sum_of_roots_of_quadratic_l2041_204151


namespace original_number_is_two_thirds_l2041_204118

theorem original_number_is_two_thirds (x : ℚ) (h : 1 + (1 / x) = 5 / 2) : x = 2 / 3 :=
by
  sorry

end original_number_is_two_thirds_l2041_204118


namespace solve_inequality_l2041_204105

theorem solve_inequality (x : ℝ) : 1 + 2 * (x - 1) ≤ 3 → x ≤ 2 :=
by
  sorry

end solve_inequality_l2041_204105


namespace angelina_speed_l2041_204162

theorem angelina_speed (v : ℝ) (h1 : 840 / v - 40 = 240 / v) :
  2 * v = 30 :=
by
  sorry

end angelina_speed_l2041_204162


namespace problem_1_problem_2_l2041_204135

-- Definitions according to the conditions
def f (x a : ℝ) := |2 * x + a| + |x - 2|

-- The first part of the problem: Proof when a = -4, solve f(x) >= 6
theorem problem_1 (x : ℝ) : 
  f x (-4) ≥ 6 ↔ x ≤ 0 ∨ x ≥ 4 := by
  sorry

-- The second part of the problem: Prove the range of a for inequality f(x) >= 3a^2 - |2 - x|
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 * a^2 - |2 - x|) ↔ (-1 ≤ a ∧ a ≤ 4 / 3) := by
  sorry

end problem_1_problem_2_l2041_204135


namespace peter_walks_more_time_l2041_204199

-- Define the total distance Peter has to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def distance_walked : ℝ := 1.0

-- Define Peter's walking pace in minutes per mile
def walking_pace : ℝ := 20.0

-- Prove that Peter has to walk 30 more minutes to reach the grocery store
theorem peter_walks_more_time : walking_pace * (total_distance - distance_walked) = 30 :=
by
  sorry

end peter_walks_more_time_l2041_204199


namespace amount_spent_on_drink_l2041_204112

-- Definitions based on conditions provided
def initialAmount : ℝ := 9
def remainingAmount : ℝ := 6
def additionalSpending : ℝ := 1.25

-- Theorem to prove the amount spent on the drink
theorem amount_spent_on_drink : 
  initialAmount - remainingAmount - additionalSpending = 1.75 := 
by 
  sorry

end amount_spent_on_drink_l2041_204112


namespace range_of_k_l2041_204189

noncomputable def f (x k : ℝ) : ℝ := 2^x + 3*x - k

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x < 2 ∧ f x k = 0) ↔ 5 ≤ k ∧ k < 10 :=
by sorry

end range_of_k_l2041_204189


namespace distinct_x_intercepts_l2041_204120

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, (x - 5) * (x ^ 2 + 3 * x + 2) = 0) ∧ s.card = 3 :=
by {
  sorry
}

end distinct_x_intercepts_l2041_204120


namespace cosine_of_half_pi_minus_double_alpha_l2041_204103

theorem cosine_of_half_pi_minus_double_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (π / 2 - 2 * α) = 4 / 5 :=
sorry

end cosine_of_half_pi_minus_double_alpha_l2041_204103


namespace find_m_and_e_l2041_204138

theorem find_m_and_e (m e : ℕ) (hm : 0 < m) (he : e < 10) 
(h1 : 4 * m^2 + m + e = 346) 
(h2 : 4 * m^2 + m + 6 = 442 + 7 * e) : 
  m + e = 22 := by
  sorry

end find_m_and_e_l2041_204138


namespace trajectory_of_M_l2041_204131

theorem trajectory_of_M
  (x y : ℝ)
  (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
sorry

end trajectory_of_M_l2041_204131


namespace preimage_of_3_1_is_2_half_l2041_204187

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2 * p.2, p.1 - 2 * p.2)

theorem preimage_of_3_1_is_2_half :
  (∃ x y : ℝ, f (x, y) = (3, 1) ∧ (x = 2 ∧ y = 1/2)) :=
by
  sorry

end preimage_of_3_1_is_2_half_l2041_204187


namespace parallel_sufficient_not_necessary_l2041_204180

def line := Type
def parallel (l1 l2 : line) : Prop := sorry
def in_plane (l : line) : Prop := sorry

theorem parallel_sufficient_not_necessary (a β : line) :
  (parallel a β → ∃ γ, in_plane γ ∧ parallel a γ) ∧
  ¬( (∃ γ, in_plane γ ∧ parallel a γ) → parallel a β ) :=
by sorry

end parallel_sufficient_not_necessary_l2041_204180


namespace power_of_thousand_l2041_204155

-- Define the notion of googol
def googol := 10^100

-- Prove that 1000^100 is equal to googol^3
theorem power_of_thousand : (1000 ^ 100) = googol^3 := by
  -- proof step to be filled here
  sorry

end power_of_thousand_l2041_204155


namespace negation_exists_l2041_204134

theorem negation_exists:
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by
  sorry

end negation_exists_l2041_204134


namespace inverse_sum_is_minus_two_l2041_204139

variable (f : ℝ → ℝ)
variable (h_injective : Function.Injective f)
variable (h_surjective : Function.Surjective f)
variable (h_eq : ∀ x : ℝ, f (x + 1) + f (-x - 3) = 2)

theorem inverse_sum_is_minus_two (x : ℝ) : f⁻¹ (2009 - x) + f⁻¹ (x - 2007) = -2 := 
  sorry

end inverse_sum_is_minus_two_l2041_204139


namespace range_of_a_l2041_204183

noncomputable def quadratic_inequality_solution_set (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x - 4 < 0

theorem range_of_a :
  {a : ℝ | quadratic_inequality_solution_set a} = {a | -16 < a ∧ a ≤ 0} := 
sorry

end range_of_a_l2041_204183


namespace total_meals_per_week_l2041_204104

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_l2041_204104


namespace gcd_442872_312750_l2041_204141

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 :=
by
  sorry

end gcd_442872_312750_l2041_204141


namespace sum_greater_than_two_l2041_204125

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end sum_greater_than_two_l2041_204125


namespace solve_derivative_equation_l2041_204184

theorem solve_derivative_equation :
  (∃ n : ℤ, ∀ x,
    x = 2 * n * Real.pi ∨
    x = 2 * n * Real.pi - 2 * Real.arctan (3 / 5)) :=
by
  sorry

end solve_derivative_equation_l2041_204184


namespace birds_total_distance_l2041_204169

-- Define the speeds of the birds
def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30

-- Define the flying time for each bird
def flying_time : ℕ := 2

-- Calculate the total distance flown by all birds
def total_distance_flown : ℕ := (eagle_speed * flying_time) +
                                 (falcon_speed * flying_time) +
                                 (pelican_speed * flying_time) +
                                 (hummingbird_speed * flying_time)

-- The goal is to prove that the total distance flown by all birds is 248 miles
theorem birds_total_distance : total_distance_flown = 248 := by
  -- Proof here
  sorry

end birds_total_distance_l2041_204169


namespace remainder_p_q_add_42_l2041_204109

def p (k : ℤ) : ℤ := 98 * k + 84
def q (m : ℤ) : ℤ := 126 * m + 117

theorem remainder_p_q_add_42 (k m : ℤ) : 
  (p k + q m) % 42 = 33 := by
  sorry

end remainder_p_q_add_42_l2041_204109


namespace unique_solution_for_a_l2041_204190

def system_has_unique_solution (a : ℝ) (x y : ℝ) : Prop :=
(x^2 + y^2 + 2 * x ≤ 1) ∧ (x - y + a = 0)

theorem unique_solution_for_a (a x y : ℝ) :
  (system_has_unique_solution 3 x y ∨ system_has_unique_solution (-1) x y)
  ∧ (((a = 3) → (x, y) = (-2, 1)) ∨ ((a = -1) → (x, y) = (0, -1))) :=
sorry

end unique_solution_for_a_l2041_204190


namespace max_value_of_quadratic_l2041_204178

theorem max_value_of_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), -3 * y^2 + 18 * y - 5 ≤ -3 * x^2 + 18 * x - 5 ∧ -3 * x^2 + 18 * x - 5 = 22 :=
sorry

end max_value_of_quadratic_l2041_204178


namespace stratified_sampling_groupD_l2041_204179

-- Definitions for the conditions
def totalDistrictCount : ℕ := 38
def groupADistrictCount : ℕ := 4
def groupBDistrictCount : ℕ := 10
def groupCDistrictCount : ℕ := 16
def groupDDistrictCount : ℕ := 8
def numberOfCitiesToSelect : ℕ := 9

-- Define stratified sampling calculation with a floor function or rounding
noncomputable def numberSelectedFromGroupD : ℕ := (groupDDistrictCount * numberOfCitiesToSelect) / totalDistrictCount

-- The theorem to prove 
theorem stratified_sampling_groupD : numberSelectedFromGroupD = 2 := by
  sorry -- This is where the proof would go

end stratified_sampling_groupD_l2041_204179


namespace convert_10203_base4_to_base10_l2041_204164

def base4_to_base10 (n : ℕ) (d₀ d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  d₄ * 4^4 + d₃ * 4^3 + d₂ * 4^2 + d₁ * 4^1 + d₀ * 4^0

theorem convert_10203_base4_to_base10 :
  base4_to_base10 10203 3 0 2 0 1 = 291 :=
by
  -- proof goes here
  sorry

end convert_10203_base4_to_base10_l2041_204164


namespace supplement_comp_greater_l2041_204195

theorem supplement_comp_greater {α β : ℝ} (h : α + β = 90) : 180 - α = β + 90 :=
by
  sorry

end supplement_comp_greater_l2041_204195


namespace range_of_a_l2041_204100

theorem range_of_a (a : ℝ) (h : a > 0) : (∀ x : ℝ, x > 0 → 9 * x + a^2 / x ≥ a^2 + 8) → 2 ≤ a ∧ a ≤ 4 :=
by
  intros h1
  sorry

end range_of_a_l2041_204100


namespace problem_I_solution_set_problem_II_range_a_l2041_204130

-- Problem (I)
-- Given f(x) = |x-1|, g(x) = 2|x+1|, and a=1, prove that the inequality f(x) - g(x) > 1 has the solution set (-1, -1/3)
theorem problem_I_solution_set (x: ℝ) : abs (x - 1) - 2 * abs (x + 1) > 1 ↔ -1 < x ∧ x < -1 / 3 := 
by sorry

-- Problem (II)
-- Given f(x) = |x-1|, g(x) = 2|x+a|, prove that if 2f(x) + g(x) ≤ (a + 1)^2 has a solution for x,
-- then a ∈ (-∞, -3] ∪ [1, ∞)
theorem problem_II_range_a (a x: ℝ) (h : ∃ x, 2 * abs (x - 1) + 2 * abs (x + a) ≤ (a + 1) ^ 2) : 
  a ≤ -3 ∨ a ≥ 1 := 
by sorry

end problem_I_solution_set_problem_II_range_a_l2041_204130


namespace father_l2041_204111

-- Define the variables
variables (F S : ℕ)

-- Define the conditions
def condition1 : Prop := F = 4 * S
def condition2 : Prop := F + 20 = 2 * (S + 20)
def condition3 : Prop := S = 10

-- Statement of the problem
theorem father's_age (h1 : condition1 F S) (h2 : condition2 F S) (h3 : condition3 S) : F = 40 :=
by sorry

end father_l2041_204111


namespace four_digit_number_divisibility_l2041_204163

theorem four_digit_number_divisibility : ∃ x : ℕ, 
  (let n := 1000 + x * 100 + 50 + x; 
   ∃ k₁ k₂ : ℤ, (n = 36 * k₁) ∧ ((10 * 5 + x) = 4 * k₂) ∧ ((2 * x + 6) % 9 = 0)) :=
sorry

end four_digit_number_divisibility_l2041_204163


namespace intersection_is_correct_l2041_204168

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_is_correct : A ∩ B = {2, 4, 8} :=
by
  sorry

end intersection_is_correct_l2041_204168


namespace intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l2041_204143

noncomputable def h (a x : ℝ) : ℝ := a * x^3 - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def f (a x : ℝ) : ℝ := h a x + 3 * x * g x
noncomputable def F (a x : ℝ) : ℝ := (a - (1/3)) * x^3 + (1/2) * x^2 * g a - h a x - 1

theorem intervals_of_monotonicity (a : ℝ) (ha : f a 1 = -1) :
  ((a = 0) → (∀ x : ℝ, (0 < x ∧ x < Real.exp (-1) → f 0 x < f 0 x + 3 * x * g x)) ∧
    (Real.exp (-1) < x ∧ 0 < x → f 0 x + 3 * x * g x > f 0 x)) := sorry

theorem m_in_terms_of_x0 (a x0 m : ℝ) (ha : a > Real.exp (10 / 3))
  (tangent_line : ∀ y, y - ( -(1 / 3) * x0^3 + (1 / 2) * x0^2 * g a) = 
    (-(x0^2) + x0 * g a) * (x - x0)) :
  m = (2 / 3) * x0^3 - (1 + (1 / 2) * g a) * x0^2 + x0 * g a := sorry

theorem at_least_two_tangents (a m : ℝ) (ha : a > Real.exp (10 / 3))
  (at_least_two : ∃ x0 y, x0 ≠ y ∧ F a x0 = m ∧ F a y = m) :
  m = 4 / 3 := sorry

end intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l2041_204143


namespace problem_I_problem_II_l2041_204152

noncomputable def f (a x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a * Real.log x
noncomputable def g (a x : ℝ) : ℝ := (1 - a) * x
noncomputable def h (x : ℝ) : ℝ := (x^2 - 2 * x) / (x - Real.log x)

theorem problem_I (a : ℝ) (ha : a > 1 / 2) :
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → deriv (f a) x > 0) ∧
  (∀ x : ℝ, 1 / 2 < x ∧ x < a → deriv (f a) x < 0) ∧
  (∀ x : ℝ, a < x → deriv (f a) x > 0) :=
sorry

theorem problem_II (a : ℝ) :
  (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ f a x₀ ≥ g a x₀) ↔ a ≤ (Real.exp 1 * (Real.exp 1 - 2)) / (Real.exp 1 - 1) :=
sorry

end problem_I_problem_II_l2041_204152


namespace divide_angle_into_parts_l2041_204176

-- Definitions based on the conditions
def given_angle : ℝ := 19

/-- 
Theorem: An angle of 19 degrees can be divided into 19 equal parts using a compass and a ruler,
and each part will measure 1 degree.
-/
theorem divide_angle_into_parts (angle : ℝ) (n : ℕ) (h1 : angle = given_angle) (h2 : n = 19) : angle / n = 1 :=
by
  -- Proof to be filled out
  sorry

end divide_angle_into_parts_l2041_204176


namespace proof_solution_l2041_204194

def proof_problem : Prop :=
  ∀ (s c p d : ℝ), 
  4 * s + 8 * c + p + 2 * d = 5.00 → 
  5 * s + 11 * c + p + 3 * d = 6.50 → 
  s + c + p + d = 1.50

theorem proof_solution : proof_problem :=
  sorry

end proof_solution_l2041_204194


namespace geometric_sum_l2041_204113

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 6) (h3 : a 3 = -18) :
  a 1 + a 2 + a 3 + a 4 = 40 :=
sorry

end geometric_sum_l2041_204113


namespace domain_condition_implies_m_range_range_condition_implies_m_range_l2041_204165

noncomputable def f (x m : ℝ) : ℝ := Real.log (x^2 - 2 * m * x + m + 2)

def condition1 (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 2 * m * x + m + 2 > 0)

def condition2 (m : ℝ) : Prop :=
  ∃ y : ℝ, (∀ x : ℝ, y = Real.log (x^2 - 2 * m * x + m + 2))

theorem domain_condition_implies_m_range (m : ℝ) :
  condition1 m → -1 < m ∧ m < 2 :=
sorry

theorem range_condition_implies_m_range (m : ℝ) :
  condition2 m → (m ≤ -1 ∨ m ≥ 2) :=
sorry

end domain_condition_implies_m_range_range_condition_implies_m_range_l2041_204165


namespace total_volume_correct_l2041_204119

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2812

-- Define the target volume
def total_volume_of_water : ℕ := 11248

-- The theorem to be proved
theorem total_volume_correct : volume_of_hemisphere * number_of_hemispheres = total_volume_of_water :=
by
  sorry

end total_volume_correct_l2041_204119


namespace gingerbread_percentage_red_hats_l2041_204173

def total_gingerbread_men (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ) : ℕ :=
  n_red_hats + n_blue_boots - n_both

def percentage_with_red_hats (n_red_hats : ℕ) (total : ℕ) : ℕ :=
  (n_red_hats * 100) / total

theorem gingerbread_percentage_red_hats 
  (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ)
  (h_red_hats : n_red_hats = 6)
  (h_blue_boots : n_blue_boots = 9)
  (h_both : n_both = 3) : 
  percentage_with_red_hats n_red_hats (total_gingerbread_men n_red_hats n_blue_boots n_both) = 50 := by
  sorry

end gingerbread_percentage_red_hats_l2041_204173


namespace problem_one_problem_two_l2041_204186

-- Problem 1
theorem problem_one : -9 + 5 * (-6) - 18 / (-3) = -33 :=
by
  sorry

-- Problem 2
theorem problem_two : ((-3/4) - (5/8) + (9/12)) * (-24) + (-8) / (2/3) = -6 :=
by
  sorry

end problem_one_problem_two_l2041_204186


namespace miniature_tower_height_l2041_204153

-- Definitions of conditions
def actual_tower_height := 60
def actual_dome_volume := 200000 -- in liters
def miniature_dome_volume := 0.4 -- in liters

-- Goal: Prove the height of the miniature tower
theorem miniature_tower_height
  (actual_tower_height: ℝ)
  (actual_dome_volume: ℝ)
  (miniature_dome_volume: ℝ) : 
  actual_tower_height = 60 ∧ actual_dome_volume = 200000 ∧ miniature_dome_volume = 0.4 →
  (actual_tower_height / ( (actual_dome_volume / miniature_dome_volume)^(1/3) )) = 1.2 :=
by
  sorry

end miniature_tower_height_l2041_204153


namespace tan_sub_pi_div_four_eq_neg_seven_f_range_l2041_204129

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Proof for the first part
theorem tan_sub_pi_div_four_eq_neg_seven (x : ℝ) (h : 3 / 4 * Real.cos x + Real.sin x = 0) :
  Real.tan (x - Real.pi / 4) = -7 := sorry

noncomputable def f (x : ℝ) : ℝ := 
  2 * ((a x).fst + (b x).fst) * (b x).fst + 2 * ((a x).snd + (b x).snd) * (b x).snd

-- Proof for the second part
theorem f_range (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  1 / 2 < f x ∧ f x < 3 / 2 + Real.sqrt 2 := sorry

end tan_sub_pi_div_four_eq_neg_seven_f_range_l2041_204129


namespace sparrows_on_fence_l2041_204167

-- Define the number of sparrows initially on the fence
def initial_sparrows : ℕ := 2

-- Define the number of sparrows that joined later
def additional_sparrows : ℕ := 4

-- Define the number of sparrows that flew away
def sparrows_flew_away : ℕ := 3

-- Define the final number of sparrows on the fence
def final_sparrows : ℕ := initial_sparrows + additional_sparrows - sparrows_flew_away

-- Prove that the final number of sparrows on the fence is 3
theorem sparrows_on_fence : final_sparrows = 3 := by
  sorry

end sparrows_on_fence_l2041_204167


namespace systematic_sampling_40th_number_l2041_204147

open Nat

theorem systematic_sampling_40th_number (N n : ℕ) (sample_size_eq : n = 50) (total_students_eq : N = 1000) (k_def : k = N / n) (first_number : ℕ) (first_number_eq : first_number = 15) : 
  first_number + k * 39 = 795 := by
  sorry

end systematic_sampling_40th_number_l2041_204147


namespace routeY_is_quicker_l2041_204144

noncomputable def timeRouteX : ℝ := 
  8 / 40 

noncomputable def timeRouteY1 : ℝ := 
  6.5 / 50 

noncomputable def timeRouteY2 : ℝ := 
  0.5 / 10

noncomputable def timeRouteY : ℝ := 
  timeRouteY1 + timeRouteY2  

noncomputable def timeDifference : ℝ := 
  (timeRouteX - timeRouteY) * 60 

theorem routeY_is_quicker : 
  timeDifference = 1.2 :=
by
  sorry

end routeY_is_quicker_l2041_204144


namespace total_cost_one_each_l2041_204185

theorem total_cost_one_each (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 6.3)
  (h2 : 4 * x + 10 * y + z = 8.4) :
  x + y + z = 2.1 :=
  sorry

end total_cost_one_each_l2041_204185


namespace find_all_functions_l2041_204101

theorem find_all_functions 
  (f : ℤ → ℝ)
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a t : ℝ, a > 0 ∧ (∀ n : ℤ, f n = a * (n + t)) :=
sorry

end find_all_functions_l2041_204101


namespace workers_in_first_group_l2041_204188

-- Define the first condition: Some workers collect 48 kg of cotton in 4 days
def cotton_collected_by_W_workers_in_4_days (W : ℕ) : ℕ := 48

-- Define the second condition: 9 workers collect 72 kg of cotton in 2 days
def cotton_collected_by_9_workers_in_2_days : ℕ := 72

-- Define the rate of cotton collected per worker per day for both scenarios
def rate_per_worker_first_group (W : ℕ) : ℕ :=
cotton_collected_by_W_workers_in_4_days W / (W * 4)

def rate_per_worker_second_group : ℕ :=
cotton_collected_by_9_workers_in_2_days / (9 * 2)

-- Given the rates are the same for both groups, prove W = 3
theorem workers_in_first_group (W : ℕ) (h : rate_per_worker_first_group W = rate_per_worker_second_group) : W = 3 :=
sorry

end workers_in_first_group_l2041_204188


namespace smallest_n_for_sum_or_difference_divisible_l2041_204158

theorem smallest_n_for_sum_or_difference_divisible (n : ℕ) :
  (∃ n : ℕ, ∀ (S : Finset ℤ), S.card = n → (∃ (x y : ℤ) (h₁ : x ≠ y), ((x + y) % 1991 = 0) ∨ ((x - y) % 1991 = 0))) ↔ n = 997 :=
sorry

end smallest_n_for_sum_or_difference_divisible_l2041_204158


namespace union_A_B_l2041_204128

def A : Set ℝ := {x | ∃ y : ℝ, y = Real.log x}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : (A ∪ B) = Set.univ :=
by
  sorry

end union_A_B_l2041_204128


namespace lassis_from_mangoes_l2041_204132

theorem lassis_from_mangoes (L M : ℕ) (h : 2 * L = 11 * M) : 12 * L = 66 :=
by sorry

end lassis_from_mangoes_l2041_204132


namespace empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l2041_204171

noncomputable def A (a : ℝ) : Set ℝ := { x | a*x^2 - 3*x + 2 = 0 }

theorem empty_set_a_gt_nine_over_eight (a : ℝ) : A a = ∅ ↔ a > 9 / 8 :=
by
  sorry

theorem singleton_set_a_values (a : ℝ) : (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) :=
by
  sorry

theorem at_most_one_element_set_a_range (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) →
  (A a = ∅ ∨ ∃ x, A a = {x}) ↔ (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l2041_204171


namespace scientific_notation_to_decimal_l2041_204160

theorem scientific_notation_to_decimal :
  5.2 * 10^(-5) = 0.000052 :=
sorry

end scientific_notation_to_decimal_l2041_204160


namespace equation_conditions_l2041_204148

theorem equation_conditions (m n : ℤ) (h1 : m ≠ 1) (h2 : n = 1) :
  ∃ x : ℤ, (m - 1) * x = 3 ↔ m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end equation_conditions_l2041_204148


namespace total_length_segments_l2041_204191

noncomputable def segment_length (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment : ℕ) :=
  let total_length := rect_horizontal_1 + rect_horizontal_2 + rect_vertical
  total_length - 8 + left_segment

theorem total_length_segments
  (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment total_left : ℕ)
  (h1 : rect_horizontal_1 = 10)
  (h2 : rect_horizontal_2 = 3)
  (h3 : rect_vertical = 12)
  (h4 : left_segment = 8)
  (h5 : total_left = 19)
  : segment_length rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment = total_left :=
sorry

end total_length_segments_l2041_204191


namespace factor_quadratic_expression_l2041_204181

theorem factor_quadratic_expression (a b : ℤ) (h: 25 * -198 = -4950 ∧ a + b = -195 ∧ a * b = -4950) : a + 2 * b = -420 :=
sorry

end factor_quadratic_expression_l2041_204181


namespace remaining_soup_feeds_20_adults_l2041_204149

theorem remaining_soup_feeds_20_adults (cans_of_soup : ℕ) (feed_4_adults : ℕ) (feed_7_children : ℕ) (initial_cans : ℕ) (children_fed : ℕ)
    (h1 : feed_4_adults = 4)
    (h2 : feed_7_children = 7)
    (h3 : initial_cans = 8)
    (h4 : children_fed = 21) : 
    (initial_cans - (children_fed / feed_7_children)) * feed_4_adults = 20 :=
by
  sorry

end remaining_soup_feeds_20_adults_l2041_204149


namespace number_of_possible_ordered_pairs_l2041_204116

theorem number_of_possible_ordered_pairs (n : ℕ) (f m : ℕ) 
  (cond1 : n = 6) 
  (cond2 : f ≥ 0) 
  (cond3 : m ≥ 0) 
  (cond4 : f + m ≤ 12) 
  : ∃ s : Finset (ℕ × ℕ), s.card = 6 := 
by 
  sorry

end number_of_possible_ordered_pairs_l2041_204116


namespace find_real_number_a_l2041_204115

theorem find_real_number_a (a : ℝ) (h : (a^2 - 3*a + 2 = 0)) (h' : (a - 2) ≠ 0) : a = 1 :=
sorry

end find_real_number_a_l2041_204115


namespace hour_division_convenience_dozen_division_convenience_l2041_204124

theorem hour_division_convenience :
  ∃ (a b c d e f g h i j : ℕ), 
  60 = 2 * a ∧
  60 = 3 * b ∧
  60 = 4 * c ∧
  60 = 5 * d ∧
  60 = 6 * e ∧
  60 = 10 * f ∧
  60 = 12 * g ∧
  60 = 15 * h ∧
  60 = 20 * i ∧
  60 = 30 * j := by
  -- to be filled with a proof later
  sorry

theorem dozen_division_convenience :
  ∃ (a b c d : ℕ),
  12 = 2 * a ∧
  12 = 3 * b ∧
  12 = 4 * c ∧
  12 = 6 * d := by
  -- to be filled with a proof later
  sorry

end hour_division_convenience_dozen_division_convenience_l2041_204124


namespace solve_for_x_l2041_204170

theorem solve_for_x (x : ℤ) : (16 : ℝ) ^ (3 * x - 5) = ((1 : ℝ) / 4) ^ (2 * x + 6) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l2041_204170


namespace right_triangle_legs_l2041_204117

theorem right_triangle_legs (a b c : ℝ) 
  (h : ℝ) 
  (h_h : h = 12) 
  (h_perimeter : a + b + c = 60) 
  (h1 : a^2 + b^2 = c^2) 
  (h_altitude : h = a * b / c) :
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l2041_204117


namespace houses_with_garage_l2041_204114

theorem houses_with_garage (P GP N : ℕ) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) 
    (total_houses : P + GP - GP + N = 65) : 
    P + 65 - P - GP + GP - N = 50 :=
by
  sorry

end houses_with_garage_l2041_204114


namespace kevin_stone_count_l2041_204193

theorem kevin_stone_count :
  ∃ (N : ℕ), (∀ (n k : ℕ), 2007 = 9 * n + 11 * k → N = 20) := 
sorry

end kevin_stone_count_l2041_204193


namespace tetrahedron_volume_le_one_l2041_204146

open Real

noncomputable def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := A
  let (x1, y1, z1) := B
  let (x2, y2, z2) := C
  let (x3, y3, z3) := D
  abs ((x1 - x0) * ((y2 - y0) * (z3 - z0) - (y3 - y0) * (z2 - z0)) -
       (x2 - x0) * ((y1 - y0) * (z3 - z0) - (y3 - y0) * (z1 - z0)) +
       (x3 - x0) * ((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0))) / 6

theorem tetrahedron_volume_le_one (A B C D : ℝ × ℝ × ℝ)
  (h1 : dist A B ≤ 2) (h2 : dist A C ≤ 2) (h3 : dist A D ≤ 2)
  (h4 : dist B C ≤ 2) (h5 : dist B D ≤ 2) (h6 : dist C D ≤ 2) :
  volume_tetrahedron A B C D ≤ 1 := by
  sorry

end tetrahedron_volume_le_one_l2041_204146


namespace bananas_to_pears_l2041_204102

theorem bananas_to_pears:
  (∀ b a o p : ℕ, 
    6 * b = 4 * a → 
    5 * a = 3 * o → 
    4 * o = 7 * p → 
    36 * b = 28 * p) :=
by
  intros b a o p h1 h2 h3
  -- We need to prove 36 * b = 28 * p under the given conditions
  sorry

end bananas_to_pears_l2041_204102


namespace question_mark_value_l2041_204126

theorem question_mark_value :
  ∀ (x : ℕ), ( ( (5568: ℝ) / (x: ℝ) )^(1/3: ℝ) + ( (72: ℝ) * (2: ℝ) )^(1/2: ℝ) = (256: ℝ)^(1/2: ℝ) ) → x = 87 :=
by
  intro x
  intro h
  sorry

end question_mark_value_l2041_204126


namespace copies_made_in_half_hour_l2041_204121

theorem copies_made_in_half_hour
  (rate1 rate2 : ℕ)  -- rates of the two copy machines
  (time : ℕ)         -- time considered
  (h_rate1 : rate1 = 40)  -- the first machine's rate
  (h_rate2 : rate2 = 55)  -- the second machine's rate
  (h_time : time = 30)    -- time in minutes
  : (rate1 * time + rate2 * time = 2850) := 
sorry

end copies_made_in_half_hour_l2041_204121


namespace general_term_formula_l2041_204133

noncomputable def a (n : ℕ) : ℝ := 1 / (Real.sqrt n)

theorem general_term_formula :
  ∀ (n : ℕ), a n = 1 / Real.sqrt n :=
by
  intros
  rfl

end general_term_formula_l2041_204133


namespace final_bicycle_price_is_225_l2041_204137

noncomputable def final_selling_price (cp_A : ℝ) (profit_A : ℝ) (profit_B : ℝ) : ℝ :=
  let sp_B := cp_A * (1 + profit_A / 100)
  let sp_C := sp_B * (1 + profit_B / 100)
  sp_C

theorem final_bicycle_price_is_225 :
  final_selling_price 114.94 35 45 = 224.99505 :=
by
  sorry

end final_bicycle_price_is_225_l2041_204137


namespace price_of_feed_corn_l2041_204157

theorem price_of_feed_corn :
  ∀ (num_sheep : ℕ) (num_cows : ℕ) (grass_per_cow : ℕ) (grass_per_sheep : ℕ)
    (feed_corn_duration_cow : ℕ) (feed_corn_duration_sheep : ℕ)
    (total_grass : ℕ) (total_expenditure : ℕ) (months_in_year : ℕ),
  num_sheep = 8 →
  num_cows = 5 →
  grass_per_cow = 2 →
  grass_per_sheep = 1 →
  feed_corn_duration_cow = 1 →
  feed_corn_duration_sheep = 2 →
  total_grass = 144 →
  total_expenditure = 360 →
  months_in_year = 12 →
  ((total_expenditure : ℝ) / (((num_cows * feed_corn_duration_cow * 4) + (num_sheep * (4 / feed_corn_duration_sheep))) : ℝ)) = 10 :=
by
  intros
  sorry

end price_of_feed_corn_l2041_204157


namespace zoey_holidays_in_a_year_l2041_204172

-- Given conditions as definitions
def holidays_per_month : ℕ := 2
def months_in_a_year : ℕ := 12

-- Definition of the total holidays in a year
def total_holidays_in_year : ℕ := holidays_per_month * months_in_a_year

-- Proof statement
theorem zoey_holidays_in_a_year : total_holidays_in_year = 24 := 
by
  sorry

end zoey_holidays_in_a_year_l2041_204172


namespace four_digit_perfect_square_l2041_204107

theorem four_digit_perfect_square : 
  ∃ (N : ℕ), (1000 ≤ N ∧ N ≤ 9999) ∧ (∃ (a b : ℕ), a = N / 1000 ∧ b = (N % 100) / 10 ∧ a = N / 100 - (N / 100 % 10) ∧ b = (N % 100 / 10) - N % 10) ∧ (∃ (n : ℕ), N = n * n) →
  N = 7744 := 
sorry

end four_digit_perfect_square_l2041_204107


namespace students_per_minibus_calculation_l2041_204122

-- Define the conditions
variables (vans minibusses total_students students_per_van : ℕ)
variables (students_per_minibus : ℕ)

-- Define the given conditions based on the problem
axiom six_vans : vans = 6
axiom four_minibusses : minibusses = 4
axiom ten_students_per_van : students_per_van = 10
axiom total_students_are_156 : total_students = 156

-- Define the problem statement in Lean
theorem students_per_minibus_calculation
  (h1 : vans = 6)
  (h2 : minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : total_students = 156) :
  students_per_minibus = 24 :=
sorry

end students_per_minibus_calculation_l2041_204122


namespace faye_initial_books_l2041_204182

theorem faye_initial_books (X : ℕ) (h : (X - 3) + 48 = 79) : X = 34 :=
sorry

end faye_initial_books_l2041_204182


namespace tax_on_clothing_l2041_204166

variable (T : ℝ)
variable (c : ℝ := 0.45 * T)
variable (f : ℝ := 0.45 * T)
variable (o : ℝ := 0.10 * T)
variable (x : ℝ)
variable (t_c : ℝ := x / 100 * c)
variable (t_f : ℝ := 0)
variable (t_o : ℝ := 0.10 * o)
variable (t : ℝ := 0.0325 * T)

theorem tax_on_clothing :
  t_c + t_o = t → x = 5 :=
by
  sorry

end tax_on_clothing_l2041_204166


namespace solve_system_of_equations_l2041_204198

theorem solve_system_of_equations :
  ∃ (x y z : ℝ),
    (x^2 + y^2 + 8 * x - 6 * y = -20) ∧
    (x^2 + z^2 + 8 * x + 4 * z = -10) ∧
    (y^2 + z^2 - 6 * y + 4 * z = 0) ∧
    ((x = -3 ∧ y = 1 ∧ z = 1) ∨
     (x = -3 ∧ y = 1 ∧ z = -5) ∨
     (x = -3 ∧ y = 5 ∧ z = 1) ∨
     (x = -3 ∧ y = 5 ∧ z = -5) ∨
     (x = -5 ∧ y = 1 ∧ z = 1) ∨
     (x = -5 ∧ y = 1 ∧ z = -5) ∨
     (x = -5 ∧ y = 5 ∧ z = 1) ∨
     (x = -5 ∧ y = 5 ∧ z = -5)) :=
sorry

end solve_system_of_equations_l2041_204198


namespace sale_price_of_sarees_after_discounts_l2041_204175

theorem sale_price_of_sarees_after_discounts :
  let original_price := 400.0
  let discount_1 := 0.15
  let discount_2 := 0.08
  let discount_3 := 0.07
  let discount_4 := 0.10
  let price_after_first_discount := original_price * (1 - discount_1)
  let price_after_second_discount := price_after_first_discount * (1 - discount_2)
  let price_after_third_discount := price_after_second_discount * (1 - discount_3)
  let final_price := price_after_third_discount * (1 - discount_4)
  final_price = 261.81 := by
    -- Sorry is used to skip the proof
    sorry

end sale_price_of_sarees_after_discounts_l2041_204175


namespace first_stack_height_l2041_204174

theorem first_stack_height (x : ℕ) (h1 : x + (x + 2) + (x - 3) + (x + 2) = 21) : x = 5 :=
by
  sorry

end first_stack_height_l2041_204174


namespace a_c3_b3_equiv_zero_l2041_204140

-- Definitions based on conditions
def cubic_eq_has_geom_progression_roots (a b c : ℝ) :=
  ∃ d q : ℝ, d ≠ 0 ∧ q ≠ 0 ∧ d + d * q + d * q^2 = -a ∧
    d^2 * q * (1 + q + q^2) = b ∧
    d^3 * q^3 = -c

-- Main theorem to prove
theorem a_c3_b3_equiv_zero (a b c : ℝ) :
  cubic_eq_has_geom_progression_roots a b c → a^3 * c - b^3 = 0 :=
by
  sorry

end a_c3_b3_equiv_zero_l2041_204140


namespace Sandy_fingernails_reach_world_record_in_20_years_l2041_204156

-- Definitions for the conditions of the problem
def world_record_len : ℝ := 26
def current_len : ℝ := 2
def growth_rate : ℝ := 0.1

-- Proof goal
theorem Sandy_fingernails_reach_world_record_in_20_years :
  (world_record_len - current_len) / growth_rate / 12 = 20 :=
by
  sorry

end Sandy_fingernails_reach_world_record_in_20_years_l2041_204156


namespace fraction_of_clerical_staff_is_one_third_l2041_204159

-- Defining the conditions
variables (employees clerical_f clerical employees_reduced employees_remaining : ℝ)

def company_conditions (employees clerical_f clerical employees_reduced employees_remaining : ℝ) : Prop :=
  employees = 3600 ∧
  clerical = 3600 * clerical_f ∧
  employees_reduced = clerical * (2 / 3) ∧
  employees_remaining = employees - clerical * (1 / 3) ∧
  employees_reduced = 0.25 * employees_remaining

-- The statement to prove the fraction of clerical employees given the conditions
theorem fraction_of_clerical_staff_is_one_third
  (hc : company_conditions employees clerical_f clerical employees_reduced employees_remaining) :
  clerical_f = 1 / 3 :=
sorry

end fraction_of_clerical_staff_is_one_third_l2041_204159


namespace total_working_days_l2041_204196

variables (x a b c : ℕ)

-- Given conditions
axiom bus_morning : b + c = 6
axiom bus_afternoon : a + c = 18
axiom train_commute : a + b = 14

-- Proposition to prove
theorem total_working_days : x = a + b + c → x = 19 :=
by
  -- Placeholder for Lean's automatic proof generation
  sorry

end total_working_days_l2041_204196


namespace joe_cut_kids_hair_l2041_204161

theorem joe_cut_kids_hair
  (time_women minutes_women count_women : ℕ)
  (time_men minutes_men count_men : ℕ)
  (time_kid minutes_kid : ℕ)
  (total_minutes: ℕ) : 
  minutes_women = 50 → 
  minutes_men = 15 →
  minutes_kid = 25 →
  count_women = 3 →
  count_men = 2 →
  total_minutes = 255 →
  (count_women * minutes_women + count_men * minutes_men + time_kid * minutes_kid) = total_minutes →
  time_kid = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof is not provided, hence stating sorry.
  sorry

end joe_cut_kids_hair_l2041_204161


namespace contrapositive_of_real_roots_l2041_204192

theorem contrapositive_of_real_roots (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0 :=
sorry

end contrapositive_of_real_roots_l2041_204192


namespace rectangle_similarity_l2041_204154

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def is_congruent (A B : Rectangle) : Prop :=
  A.length = B.length ∧ A.width = B.width

def is_similar (A B : Rectangle) : Prop :=
  A.length / A.width = B.length / B.width

theorem rectangle_similarity (A B : Rectangle)
  (h1 : ∀ P, is_congruent P A → ∃ Q, is_similar Q B)
  : ∀ P, is_congruent P B → ∃ Q, is_similar Q A :=
by sorry

end rectangle_similarity_l2041_204154


namespace division_of_product_l2041_204123

theorem division_of_product :
  (1.6 * 0.5) / 1 = 0.8 :=
sorry

end division_of_product_l2041_204123


namespace votes_for_veggies_l2041_204106

theorem votes_for_veggies (T M V : ℕ) (hT : T = 672) (hM : M = 335) (hV : V = T - M) : V = 337 := 
by
  rw [hT, hM] at hV
  simp at hV
  exact hV

end votes_for_veggies_l2041_204106


namespace possible_values_of_x_l2041_204110

theorem possible_values_of_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) 
    (h1 : x + 1 / z = 15) (h2 : z + 1 / x = 9 / 20) :
    x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by
  sorry

end possible_values_of_x_l2041_204110


namespace paul_and_lisa_total_dollars_l2041_204145

def total_dollars_of_paul_and_lisa (paul_dol : ℚ) (lisa_dol : ℚ) : ℚ :=
  paul_dol + lisa_dol

theorem paul_and_lisa_total_dollars (paul_dol := (5 / 6 : ℚ)) (lisa_dol := (2 / 5 : ℚ)) :
  total_dollars_of_paul_and_lisa paul_dol lisa_dol = (123 / 100 : ℚ) :=
by
  sorry

end paul_and_lisa_total_dollars_l2041_204145


namespace find_number_l2041_204136

-- Assume the necessary definitions and conditions
variable (x : ℝ)

-- Sixty-five percent of the number is 21 less than four-fifths of the number
def condition := 0.65 * x = 0.8 * x - 21

-- Final proof goal: We need to prove that the number x is 140
theorem find_number (h : condition x) : x = 140 := by
  sorry

end find_number_l2041_204136


namespace max_area_of_triangle_l2041_204127

theorem max_area_of_triangle :
  ∀ (O O' : EuclideanSpace ℝ (Fin 2)) (M : EuclideanSpace ℝ (Fin 2)),
  dist O O' = 2014 →
  dist O M = 1 ∨ dist O' M = 1 →
  ∃ (A : ℝ), A = 1007 :=
by
  intros O O' M h₁ h₂
  sorry

end max_area_of_triangle_l2041_204127


namespace color_of_217th_marble_l2041_204108

-- Definitions of conditions
def total_marbles := 240
def pattern_length := 15
def red_marbles := 6
def blue_marbles := 5
def green_marbles := 4
def position := 217

-- Lean 4 statement
theorem color_of_217th_marble :
  (position % pattern_length ≤ red_marbles) :=
by sorry

end color_of_217th_marble_l2041_204108


namespace solve_for_x_l2041_204142

theorem solve_for_x (x : ℝ) : 9 * x^2 - 4 = 0 → (x = 2/3 ∨ x = -2/3) :=
by
  sorry

end solve_for_x_l2041_204142
