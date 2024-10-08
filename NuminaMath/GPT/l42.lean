import Mathlib

namespace air_quality_conditional_prob_l42_42256

theorem air_quality_conditional_prob :
  let p1 := 0.8
  let p2 := 0.68
  let p := p2 / p1
  p = 0.85 :=
by
  sorry

end air_quality_conditional_prob_l42_42256


namespace part_I_part_II_l42_42918

noncomputable def f (x : ℝ) := (Real.sin x) * (Real.cos x) + (Real.sin x)^2

-- Part I: Prove that f(π / 4) = 1
theorem part_I : f (Real.pi / 4) = 1 := sorry

-- Part II: Prove that the maximum value of f(x) for x ∈ [0, π / 2] is (√2 + 1) / 2
theorem part_II : ∃ x ∈ Set.Icc 0 (Real.pi / 2), (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧ f x = (Real.sqrt 2 + 1) / 2 := sorry

end part_I_part_II_l42_42918


namespace smallest_possible_AC_l42_42805

theorem smallest_possible_AC 
    (AB AC CD : ℤ) 
    (BD_squared : ℕ) 
    (h_isosceles : AB = AC)
    (h_point_D : ∃ D : ℤ, D = CD)
    (h_perpendicular : BD_squared = 85) 
    (h_integers : ∃ x y : ℤ, AC = x ∧ CD = y) 
    : AC = 11 :=
by
  sorry

end smallest_possible_AC_l42_42805


namespace initial_leaves_l42_42193

theorem initial_leaves (l_0 : ℕ) (blown_away : ℕ) (leaves_left : ℕ) (h1 : blown_away = 244) (h2 : leaves_left = 112) (h3 : l_0 - blown_away = leaves_left) : l_0 = 356 :=
by
  sorry

end initial_leaves_l42_42193


namespace schur_theorem_l42_42733

theorem schur_theorem {n : ℕ} (P : Fin n → Set ℕ) (h_partition : ∀ x : ℕ, ∃ i : Fin n, x ∈ P i) :
  ∃ (i : Fin n) (x y : ℕ), x ∈ P i ∧ y ∈ P i ∧ x + y ∈ P i :=
sorry

end schur_theorem_l42_42733


namespace apples_in_each_box_l42_42161

variable (A : ℕ)
variable (ApplesSaturday : ℕ := 50 * A)
variable (ApplesSunday : ℕ := 25 * A)
variable (ApplesLeft : ℕ := 3 * A)
variable (ApplesSold : ℕ := 720)

theorem apples_in_each_box :
  (ApplesSaturday + ApplesSunday - ApplesSold = ApplesLeft) → A = 10 :=
by
  sorry

end apples_in_each_box_l42_42161


namespace shaded_region_equality_l42_42266

-- Define the necessary context and variables
variable {r : ℝ} -- radius of the circle
variable {θ : ℝ} -- angle measured in degrees

-- Define the relevant trigonometric functions
noncomputable def tan_degrees (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)
noncomputable def tan_half_degrees (x : ℝ) : ℝ := Real.tan ((x / 2) * Real.pi / 180)

-- State the theorem we need to prove given the conditions
theorem shaded_region_equality (hθ1 : θ / 2 = 90 - θ) :
  tan_degrees θ + (tan_degrees θ)^2 * tan_half_degrees θ = (θ * Real.pi) / 180 - (θ^2 * Real.pi) / 360 :=
  sorry

end shaded_region_equality_l42_42266


namespace minimize_xy_l42_42222

theorem minimize_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_eq : 7 * x + 4 * y = 200) : (x * y = 172) :=
sorry

end minimize_xy_l42_42222


namespace inscribed_sphere_tetrahedron_volume_l42_42598

theorem inscribed_sphere_tetrahedron_volume
  (R : ℝ) (S1 S2 S3 S4 : ℝ) :
  ∃ V : ℝ, V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_tetrahedron_volume_l42_42598


namespace regular_price_of_ticket_l42_42688

theorem regular_price_of_ticket (P : Real) (discount_paid : Real) (discount_rate : Real) (paid : Real)
  (h_discount_rate : discount_rate = 0.40)
  (h_paid : paid = 9)
  (h_discount_paid : discount_paid = P * (1 - discount_rate))
  (h_paid_eq_discount_paid : paid = discount_paid) :
  P = 15 := 
by
  sorry

end regular_price_of_ticket_l42_42688


namespace recurring_decimal_addition_l42_42980

noncomputable def recurring_decimal_sum : ℚ :=
  (23 / 99) + (14 / 999) + (6 / 9999)

theorem recurring_decimal_addition :
  recurring_decimal_sum = 2469 / 9999 :=
sorry

end recurring_decimal_addition_l42_42980


namespace sequence_term_4th_l42_42593

theorem sequence_term_4th (a_n : ℕ → ℝ) (h : ∀ n, a_n n = 2 / (n^2 + n)) :
  ∃ n, a_n n = 1 / 10 ∧ n = 4 :=
by
  sorry

end sequence_term_4th_l42_42593


namespace range_of_a_l42_42778

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + |x - 2|
  else x^2 - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l42_42778


namespace least_number_to_add_l42_42893

theorem least_number_to_add (n d : ℕ) (h : n = 1024) (h_d : d = 25) :
  ∃ x : ℕ, (n + x) % d = 0 ∧ x = 1 :=
by sorry

end least_number_to_add_l42_42893


namespace hyperbola_asymptotes_angle_l42_42229

noncomputable def angle_between_asymptotes 
  (a b : ℝ) (e : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) : ℝ :=
  2 * Real.arctan (b / a)

theorem hyperbola_asymptotes_angle (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) 
  (b_eq : b = Real.sqrt (e^2 * a^2 - a^2)) : 
  angle_between_asymptotes a b e h1 h2 h3 = π / 3 := 
by
  -- proof omitted
  sorry
  
end hyperbola_asymptotes_angle_l42_42229


namespace find_w_l42_42811

noncomputable def roots_cubic_eq (x : ℝ) : ℝ := x^3 + 2 * x^2 + 5 * x - 8

def p : ℝ := sorry -- one root of x^3 + 2x^2 + 5x - 8 = 0
def q : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0
def r : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0

theorem find_w 
  (h1 : roots_cubic_eq p = 0)
  (h2 : roots_cubic_eq q = 0)
  (h3 : roots_cubic_eq r = 0)
  (h4 : p + q + r = -2): 
  ∃ w : ℝ, w = 18 := 
sorry

end find_w_l42_42811


namespace min_value_a_b_c_l42_42900

def A_n (a : ℕ) (n : ℕ) : ℕ := a * ((10^n - 1) / 9)
def B_n (b : ℕ) (n : ℕ) : ℕ := b * ((10^n - 1) / 9)
def C_n (c : ℕ) (n : ℕ) : ℕ := c * ((10^(2*n) - 1) / 9)

theorem min_value_a_b_c (a b c : ℕ) (Ha : 0 < a ∧ a < 10) (Hb : 0 < b ∧ b < 10) (Hc : 0 < c ∧ c < 10) :
  (∃ n1 n2 : ℕ, (n1 ≠ n2) ∧ (C_n c n1 - A_n a n1 = B_n b n1 ^ 2) ∧ (C_n c n2 - A_n a n2 = B_n b n2 ^ 2)) →
  a + b + c = 5 :=
by
  sorry

end min_value_a_b_c_l42_42900


namespace hexagon_angle_Q_l42_42349

theorem hexagon_angle_Q
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : a1 = 134) 
  (h2 : a2 = 98) 
  (h3 : a3 = 120) 
  (h4 : a4 = 110) 
  (h5 : a5 = 96) 
  (sum_hexagon_angles : a1 + a2 + a3 + a4 + a5 + Q = 720) : 
  Q = 162 := by {
  sorry
}

end hexagon_angle_Q_l42_42349


namespace smallest_four_digit_remainder_l42_42063

theorem smallest_four_digit_remainder :
  ∃ N : ℕ, (N % 6 = 5) ∧ (1000 ≤ N ∧ N ≤ 9999) ∧ (∀ M : ℕ, (M % 6 = 5) ∧ (1000 ≤ M ∧ M ≤ 9999) → N ≤ M) ∧ N = 1001 :=
by
  sorry

end smallest_four_digit_remainder_l42_42063


namespace inequality_div_half_l42_42723

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end inequality_div_half_l42_42723


namespace julio_twice_james_in_years_l42_42080

noncomputable def years_until_julio_twice_james := 
  let x := 14
  (36 + x = 2 * (11 + x))

theorem julio_twice_james_in_years : 
  years_until_julio_twice_james := 
  by 
  sorry

end julio_twice_james_in_years_l42_42080


namespace factor_expression_l42_42664

theorem factor_expression :
  (12 * x ^ 6 + 40 * x ^ 4 - 6) - (2 * x ^ 6 - 6 * x ^ 4 - 6) = 2 * x ^ 4 * (5 * x ^ 2 + 23) :=
by sorry

end factor_expression_l42_42664


namespace church_members_l42_42651

theorem church_members (M A C : ℕ) (h1 : A = 4/10 * M)
  (h2 : C = 6/10 * M) (h3 : C = A + 24) : M = 120 := 
  sorry

end church_members_l42_42651


namespace mod_exp_equivalence_l42_42364

theorem mod_exp_equivalence :
  (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end mod_exp_equivalence_l42_42364


namespace Danny_bottle_caps_l42_42538

theorem Danny_bottle_caps (r w c : ℕ) (h1 : r = 11) (h2 : c = r + 1) : c = 12 := by
  sorry

end Danny_bottle_caps_l42_42538


namespace value_of_expression_l42_42246

theorem value_of_expression (x y z : ℤ) (h1 : x = -3) (h2 : y = 5) (h3 : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 :=
by
  -- proof goes here
  sorry

end value_of_expression_l42_42246


namespace find_a_values_l42_42373

noncomputable def function_a_max_value (a : ℝ) : ℝ :=
  a^2 + 2 * a - 9

theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : function_a_max_value a = 6) : 
    a = 3 ∨ a = 1/3 :=
  sorry

end find_a_values_l42_42373


namespace Ben_cards_left_l42_42267

def BenInitialBasketballCards : ℕ := 4 * 10
def BenInitialBaseballCards : ℕ := 5 * 8
def BenTotalInitialCards : ℕ := BenInitialBasketballCards + BenInitialBaseballCards
def BenGivenCards : ℕ := 58
def BenRemainingCards : ℕ := BenTotalInitialCards - BenGivenCards

theorem Ben_cards_left : BenRemainingCards = 22 :=
by 
  -- The proof will be placed here.
  sorry

end Ben_cards_left_l42_42267


namespace vacuum_upstairs_more_than_twice_downstairs_l42_42200

theorem vacuum_upstairs_more_than_twice_downstairs 
  (x y : ℕ) 
  (h1 : 27 = 2 * x + y) 
  (h2 : x + 27 = 38) : 
  y = 5 :=
by 
  sorry

end vacuum_upstairs_more_than_twice_downstairs_l42_42200


namespace find_N_l42_42074

def f (N : ℕ) : ℕ :=
  if N % 2 = 0 then 5 * N else 3 * N + 2

theorem find_N (N : ℕ) :
  f (f (f (f (f N)))) = 542 ↔ N = 112500 := by
  sorry

end find_N_l42_42074


namespace maximum_sine_sum_l42_42323

open Real

theorem maximum_sine_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : x ≤ π / 2) (hz : 0 ≤ y) (hw : y ≤ π / 2) (hv : 0 ≤ z) (hu : z ≤ π / 2) :
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x y z : ℝ, 0 ≤ x → x ≤ π / 2 → 0 ≤ y → y ≤ π / 2 → 0 ≤ z → z ≤ π / 2 → 
  sin (x - y) + sin (y - z) + sin (z - x) ≤ M :=
by
  sorry

end maximum_sine_sum_l42_42323


namespace complement_of_A_in_U_l42_42209

noncomputable def U : Set ℝ := {x | (x - 2) / x ≤ 1}

noncomputable def A : Set ℝ := {x | 2 - x ≤ 1}

theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x < 1} :=
by
  sorry

end complement_of_A_in_U_l42_42209


namespace tangent_line_min_slope_equation_l42_42284

def curve (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

theorem tangent_line_min_slope_equation :
  ∃ (k : ℝ) (b : ℝ), (∀ x y, y = curve x → y = k * x + b)
  ∧ (k = 3)
  ∧ (b = -2)
  ∧ (3 * x - y - 2 = 0) :=
by
  sorry

end tangent_line_min_slope_equation_l42_42284


namespace necessary_but_not_sufficient_l42_42110

-- Define the function f(x)
def f (a x : ℝ) := |a - 3 * x|

-- Define the condition for the function to be monotonically increasing on [1, +∞)
def is_monotonically_increasing_on_interval (a : ℝ) : Prop :=
  ∀ (x y : ℝ), 1 ≤ x → x ≤ y → (f a x ≤ f a y)

-- Define the condition that a must be 3
def condition_a_eq_3 (a : ℝ) : Prop := (a = 3)

-- Prove that condition_a_eq_3 is a necessary but not sufficient condition
theorem necessary_but_not_sufficient (a : ℝ) :
  (is_monotonically_increasing_on_interval a) →
  condition_a_eq_3 a ↔ (∀ (b : ℝ), b ≠ a → is_monotonically_increasing_on_interval b → false) := 
sorry

end necessary_but_not_sufficient_l42_42110


namespace part1_part2_l42_42472

open Complex

theorem part1 {m : ℝ} : m + (m^2 + 2) * I = 0 -> m = 0 :=
by sorry

theorem part2 {m : ℝ} (h : (m + I)^2 - 2 * (m + I) + 2 = 0) :
    (let z1 := m + I
     let z2 := 2 + m * I
     im ((z2 / z1) : ℂ) = -1 / 2) :=
by sorry

end part1_part2_l42_42472


namespace solve_inequality_l42_42413

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 2) * x + 2 < 0

-- Prove the solution sets for different values of a
theorem solve_inequality :
  ∀ (a : ℝ),
    (a = -1 → {x : ℝ | inequality a x} = {x | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x | x < 2 / a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x | 1 < x ∧ x < 2 / a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x | 2 / a < x ∧ x < 1}) :=
by sorry

end solve_inequality_l42_42413


namespace probability_face_then_number_l42_42684

theorem probability_face_then_number :
  let total_cards := 52
  let total_ways_to_draw_two := total_cards * (total_cards - 1)
  let face_cards := 3 * 4
  let number_cards := 9 * 4
  let probability := (face_cards * number_cards) / total_ways_to_draw_two
  probability = 8 / 49 :=
by
  sorry

end probability_face_then_number_l42_42684


namespace negation_of_universal_proposition_l42_42963

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1 / 4 > 0)) = ∃ x : ℝ, x^2 - x + 1 / 4 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l42_42963


namespace shepherds_sheep_l42_42086

theorem shepherds_sheep (x y : ℕ) 
  (h1 : x - 4 = y + 4) 
  (h2 : x + 4 = 3 * (y - 4)) : 
  x = 20 ∧ y = 12 := 
by 
  sorry

end shepherds_sheep_l42_42086


namespace vika_made_84_dollars_l42_42262

-- Define the amount of money Saheed, Kayla, and Vika made
variable (S K V : ℕ)

-- Given conditions
def condition1 : Prop := S = 4 * K
def condition2 : Prop := K = V - 30
def condition3 : Prop := S = 216

-- Statement to prove
theorem vika_made_84_dollars (S K V : ℕ) (h1 : condition1 S K) (h2 : condition2 K V) (h3 : condition3 S) : 
  V = 84 :=
by sorry

end vika_made_84_dollars_l42_42262


namespace find_eccentricity_l42_42427

variables {a b x_N x_M : ℝ}
variable {e : ℝ}

-- Conditions
def line_passes_through_N (x_N : ℝ) (x_M : ℝ) : Prop :=
x_N ≠ 0 ∧ x_N = 4 * x_M

def hyperbola (x y a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def midpoint_x_M (x_M : ℝ) : Prop :=
∃ (x1 x2 y1 y2 : ℝ), (x1 + x2) / 2 = x_M

-- Proof Problem
theorem find_eccentricity
  (hN : line_passes_through_N x_N x_M)
  (hC : hyperbola x_N 0 a b)
  (hM : midpoint_x_M x_M) :
  e = 2 :=
sorry

end find_eccentricity_l42_42427


namespace bags_on_wednesday_l42_42987

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l42_42987


namespace ten_person_round_robin_l42_42634

def number_of_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem ten_person_round_robin : number_of_matches 10 = 45 :=
by
  -- Proof steps would go here, but are omitted for this task
  sorry

end ten_person_round_robin_l42_42634


namespace other_number_is_twelve_l42_42716

variable (x certain_number : ℕ)
variable (h1: certain_number = 60)
variable (h2: certain_number = 5 * x)

theorem other_number_is_twelve :
  x = 12 :=
by
  sorry

end other_number_is_twelve_l42_42716


namespace necessary_and_sufficient_condition_l42_42756

theorem necessary_and_sufficient_condition (x : ℝ) :
  (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := sorry

end necessary_and_sufficient_condition_l42_42756


namespace max_value_of_function_l42_42052

open Real 

theorem max_value_of_function : ∀ x : ℝ, 
  cos (2 * x) + 6 * cos (π / 2 - x) ≤ 5 ∧ 
  ∃ x' : ℝ, cos (2 * x') + 6 * cos (π / 2 - x') = 5 :=
by 
  sorry

end max_value_of_function_l42_42052


namespace find_smaller_number_l42_42424

theorem find_smaller_number (a b : ℤ) (h₁ : a + b = 8) (h₂ : a - b = 4) : b = 2 :=
by
  sorry

end find_smaller_number_l42_42424


namespace largest_n_l42_42648

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_n (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) ∧
  100 ≤ x * y * (10 * y + x) ∧ x * y * (10 * y + x) < 1000

theorem largest_n : ∃ x y : ℕ, valid_n x y ∧ x * y * (10 * y + x) = 777 := by
  sorry

end largest_n_l42_42648


namespace complement_intersect_l42_42416

def U : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def A : Set ℤ := {x | x^2 - 1 ≤ 0}
def B : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def C : Set ℤ := {x | x ∉ A ∧ x ∈ U} -- complement of A in U

theorem complement_intersect (U A B : Set ℤ) :
  (C ∩ B) = {2, 3} :=
by
  sorry

end complement_intersect_l42_42416


namespace project_completion_time_l42_42171

theorem project_completion_time :
  let A_work_rate := (1 / 30) * (2 / 3)
  let B_work_rate := (1 / 60) * (3 / 4)
  let C_work_rate := (1 / 40) * (5 / 6)
  let combined_work_rate_per_12_days := 12 * (A_work_rate + B_work_rate + C_work_rate)
  let remaining_work_after_12_days := 1 - (2 / 3)
  let additional_work_rates_over_5_days := 
        5 * A_work_rate + 
        5 * B_work_rate + 
        5 * C_work_rate
  let remaining_work_after_5_days := remaining_work_after_12_days - additional_work_rates_over_5_days
  let B_additional_time := remaining_work_after_5_days / B_work_rate
  12 + 5 + B_additional_time = 17.5 :=
sorry

end project_completion_time_l42_42171


namespace value_of_f_ln6_l42_42435

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + Real.exp x else -(x + Real.exp (-x))

theorem value_of_f_ln6 : (f (Real.log 6)) = Real.log 6 - (1/6) :=
by
  sorry

end value_of_f_ln6_l42_42435


namespace cone_lateral_area_l42_42826

/--
Given that the radius of the base of a cone is 3 cm and the slant height is 6 cm,
prove that the lateral area of this cone is 18π cm².
-/
theorem cone_lateral_area {r l : ℝ} (h_radius : r = 3) (h_slant_height : l = 6) :
  (π * r * l) = 18 * π :=
by
  have h1 : r = 3 := h_radius
  have h2 : l = 6 := h_slant_height
  rw [h1, h2]
  norm_num
  sorry

end cone_lateral_area_l42_42826


namespace smallest_fraction_l42_42450

theorem smallest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 3) (h2 : f2 = 3 / 4) (h3 : f3 = 5 / 6) 
  (h4 : f4 = 5 / 8) (h5 : f5 = 11 / 12) : f4 = 5 / 8 ∧ f4 < f1 ∧ f4 < f2 ∧ f4 < f3 ∧ f4 < f5 := 
by 
  sorry

end smallest_fraction_l42_42450


namespace remaining_people_statement_l42_42446

-- Definitions of conditions
def number_of_people : Nat := 10
def number_of_knights (K : Nat) : Prop := K ≤ number_of_people
def number_of_liars (L : Nat) : Prop := L ≤ number_of_people
def statement (s : String) : Prop := s = "There are more liars" ∨ s = "There are equal numbers"

-- Main theorem
theorem remaining_people_statement (K L : Nat) (h_total : K + L = number_of_people) 
  (h_knights_behavior : ∀ k, k < K → statement "There are equal numbers") 
  (h_liars_behavior : ∀ l, l < L → statement "There are more liars") :
  K = 5 → L = 5 → ∀ i, i < number_of_people → (i < 5 → statement "There are more liars") ∧ (i >= 5 → statement "There are equal numbers") := 
by
  sorry

end remaining_people_statement_l42_42446


namespace speed_of_X_l42_42764

theorem speed_of_X (t1 t2 Vx : ℝ) (h1 : t2 - t1 = 3) 
  (h2 : 3 * Vx + Vx * t1 = 60 * t1 + 30)
  (h3 : 3 * Vx + Vx * t2 + 30 = 60 * t2) : Vx = 60 :=
by sorry

end speed_of_X_l42_42764


namespace average_run_per_day_l42_42090

theorem average_run_per_day (n6 n7 n8 : ℕ) 
  (h1 : 3 * n7 = n6) 
  (h2 : 3 * n8 = n7) 
  (h3 : n6 * 20 + n7 * 18 + n8 * 16 = 250 * n8) : 
  (n6 * 20 + n7 * 18 + n8 * 16) / (n6 + n7 + n8) = 250 / 13 :=
by sorry

end average_run_per_day_l42_42090


namespace bridge_max_weight_l42_42285

variables (M K Mi B : ℝ)

-- Given conditions
def kelly_weight : K = 34 := sorry
def kelly_megan_relation : K = 0.85 * M := sorry
def mike_megan_relation : Mi = M + 5 := sorry
def total_excess : K + M + Mi = B + 19 := sorry

-- Proof goal: The maximum weight the bridge can hold is 100 kg.
theorem bridge_max_weight : B = 100 :=
by
  sorry

end bridge_max_weight_l42_42285


namespace medium_pizza_promotion_price_l42_42788

-- Define the conditions
def regular_price_medium_pizza : ℝ := 18
def total_savings : ℝ := 39
def number_of_medium_pizzas : ℝ := 3

-- Define the goal
theorem medium_pizza_promotion_price : 
  ∃ P : ℝ, 3 * regular_price_medium_pizza - 3 * P = total_savings ∧ P = 5 := 
by
  sorry

end medium_pizza_promotion_price_l42_42788


namespace braids_each_dancer_l42_42582

-- Define the conditions
def num_dancers := 8
def time_per_braid := 30 -- seconds per braid
def total_time := 20 * 60 -- convert 20 minutes into seconds

-- Define the total number of braids Jill makes
def total_braids := total_time / time_per_braid

-- Define the number of braids per dancer
def braids_per_dancer := total_braids / num_dancers

-- Theorem: Prove that each dancer has 5 braids
theorem braids_each_dancer : braids_per_dancer = 5 := 
by sorry

end braids_each_dancer_l42_42582


namespace carousel_seat_count_l42_42028

theorem carousel_seat_count
  (total_seats : ℕ)
  (colors : ℕ → Prop)
  (num_yellow num_blue num_red : ℕ)
  (num_colors : ∀ n, colors n → n = num_yellow ∨ n = num_blue ∨ n = num_red)
  (opposite_blue_red_7_3 : ∀ n, n = 7 ↔ n + 50 = 3)
  (opposite_yellow_red_7_23 : ∀ n, n = 7 ↔ n + 50 = 23)
  (total := 100)
 :
 (num_yellow = 34 ∧ num_blue = 20 ∧ num_red = 46) :=
by
  sorry

end carousel_seat_count_l42_42028


namespace fraction_of_larger_part_l42_42964

theorem fraction_of_larger_part (x y : ℝ) (f : ℝ) (h1 : x = 50) (h2 : x + y = 66) (h3 : f * x = 0.625 * y + 10) : f = 0.4 :=
by
  sorry

end fraction_of_larger_part_l42_42964


namespace smaller_integer_of_two_digits_l42_42303

theorem smaller_integer_of_two_digits (a b : ℕ) (ha : 10 ≤ a ∧ a ≤ 99) (hb: 10 ≤ b ∧ b ≤ 99) (h_diff : a ≠ b)
  (h_eq : (a + b) / 2 = a + b / 100) : a = 49 ∨ b = 49 := 
by
  sorry

end smaller_integer_of_two_digits_l42_42303


namespace empty_to_occupied_ratio_of_spheres_in_cylinder_package_l42_42376

theorem empty_to_occupied_ratio_of_spheres_in_cylinder_package
  (R : ℝ) 
  (volume_sphere : ℝ)
  (volume_cylinder : ℝ)
  (sphere_occupies_fraction : ∀ R : ℝ, volume_sphere = (2 / 3) * volume_cylinder) 
  (num_spheres : ℕ) 
  (h_num_spheres : num_spheres = 5) :
  (num_spheres : ℝ) * volume_sphere = (5 * (2 / 3) * π * R^3) → 
  volume_sphere = (4 / 3) * π * R^3 → 
  volume_cylinder = 2 * π * R^3 → 
  (volume_cylinder - volume_sphere) / volume_sphere = 1 / 2 := by 
  sorry

end empty_to_occupied_ratio_of_spheres_in_cylinder_package_l42_42376


namespace find_a_l42_42753

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Theorem to be proved
theorem find_a (a : ℝ) 
  (h1 : A a ∪ B a = {1, 3, a}) : a = 0 ∨ a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l42_42753


namespace find_AG_l42_42867

-- Defining constants and variables
variables (DE EC AD BC FB AG : ℚ)
variables (BC_def : BC = (1 / 3) * AD)
variables (FB_def : FB = (2 / 3) * AD)
variables (DE_val : DE = 8)
variables (EC_val : EC = 6)
variables (sum_AD : BC + FB = AD)

-- The theorem statement
theorem find_AG : AG = 56 / 9 :=
by
  -- Placeholder for the proof
  sorry

end find_AG_l42_42867


namespace num_natural_numbers_divisible_by_7_l42_42268

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l42_42268


namespace arithmetic_sequence_problem_l42_42774

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence

theorem arithmetic_sequence_problem
  (a_4 : ℝ) (a_9 : ℝ)
  (h_a4 : a_4 = 5)
  (h_a9 : a_9 = 17)
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) = a_n n + (a_n 2 - a_n 1)) :
  a_n 14 = 29 :=
by
  -- the proof will utilize the property of arithmetic sequence and substitutions
  sorry

end arithmetic_sequence_problem_l42_42774


namespace find_C_and_D_l42_42556

noncomputable def C : ℚ := 51 / 10
noncomputable def D : ℚ := 29 / 10

theorem find_C_and_D (x : ℚ) (h1 : x^2 - 4*x - 21 = (x - 7)*(x + 3))
  (h2 : (8*x - 5) / ((x - 7)*(x + 3)) = C / (x - 7) + D / (x + 3)) :
  C = 51 / 10 ∧ D = 29 / 10 :=
by
  sorry

end find_C_and_D_l42_42556


namespace percent_of_rectangle_area_inside_square_l42_42325

theorem percent_of_rectangle_area_inside_square
  (s : ℝ)  -- Let the side length of the square be \( s \).
  (width : ℝ) (length: ℝ)
  (h1 : width = 3 * s)  -- The width of the rectangle is \( 3s \).
  (h2 : length = 2 * width) :  -- The length of the rectangle is \( 2 * width \).
  (s^2 / (length * width)) * 100 = 5.56 :=
by
  sorry

end percent_of_rectangle_area_inside_square_l42_42325


namespace cube_div_identity_l42_42808

theorem cube_div_identity (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) :
  (a^3 + b^3) / (a^2 - a * b + b^2) = 9 := by
  sorry

end cube_div_identity_l42_42808


namespace profit_percentage_each_portion_l42_42374

theorem profit_percentage_each_portion (P : ℝ) (total_apples : ℝ) 
  (portion1_percentage : ℝ) (portion2_percentage : ℝ) (total_profit_percentage : ℝ) :
  total_apples = 280 →
  portion1_percentage = 0.4 →
  portion2_percentage = 0.6 →
  total_profit_percentage = 0.3 →
  portion1_percentage * P + portion2_percentage * P = total_profit_percentage →
  P = 0.3 :=
by
  intros
  sorry

end profit_percentage_each_portion_l42_42374


namespace arrangements_TOOTH_l42_42794
-- Import necessary libraries

-- Define the problem conditions
def word_length : Nat := 5
def count_T : Nat := 2
def count_O : Nat := 2

-- State the problem as a theorem
theorem arrangements_TOOTH : 
  (word_length.factorial / (count_T.factorial * count_O.factorial)) = 30 := by
  sorry

end arrangements_TOOTH_l42_42794


namespace translation_vector_condition_l42_42405

theorem translation_vector_condition (m n : ℝ) :
  (∀ x : ℝ, 2 * (x - m) + n = 2 * x + 5) → n = 2 * m + 5 :=
by
  intro h
  -- proof can be filled here
  sorry

end translation_vector_condition_l42_42405


namespace curvature_formula_l42_42464

noncomputable def curvature_squared (x y : ℝ → ℝ) (t : ℝ) :=
  let x' := (deriv x t)
  let y' := (deriv y t)
  let x'' := (deriv (deriv x) t)
  let y'' := (deriv (deriv y) t)
  (x'' * y' - y'' * x')^2 / (x'^2 + y'^2)^3

theorem curvature_formula (x y : ℝ → ℝ) (t : ℝ) :
  let k_sq := curvature_squared x y t
  k_sq = ((deriv (deriv x) t * deriv y t - deriv (deriv y) t * deriv x t)^2 /
         ((deriv x t)^2 + (deriv y t)^2)^3) := 
by 
  sorry

end curvature_formula_l42_42464


namespace gcd_84_294_315_l42_42087

def gcd_3_integers : ℕ := Nat.gcd (Nat.gcd 84 294) 315

theorem gcd_84_294_315 : gcd_3_integers = 21 :=
by
  sorry

end gcd_84_294_315_l42_42087


namespace percentage_deposited_l42_42101

theorem percentage_deposited (amount_deposited income : ℝ) 
  (h1 : amount_deposited = 2500) (h2 : income = 10000) : 
  (amount_deposited / income) * 100 = 25 :=
by
  have amount_deposited_val : amount_deposited = 2500 := h1
  have income_val : income = 10000 := h2
  sorry

end percentage_deposited_l42_42101


namespace time_taken_y_alone_l42_42308

-- Define the work done in terms of rates
def work_done (Rx Ry Rz : ℝ) (W : ℝ) :=
  Rx = W / 8 ∧ (Ry + Rz) = W / 6 ∧ (Rx + Rz) = W / 4

-- Prove that the time taken by y alone is 24 hours
theorem time_taken_y_alone (Rx Ry Rz W : ℝ) (h : work_done Rx Ry Rz W) :
  (1 / Ry) = 24 :=
by
  sorry

end time_taken_y_alone_l42_42308


namespace computer_multiplications_l42_42497

def rate : ℕ := 15000
def time : ℕ := 2 * 3600
def expected_multiplications : ℕ := 108000000

theorem computer_multiplications : rate * time = expected_multiplications := by
  sorry

end computer_multiplications_l42_42497


namespace clock_minutes_to_correct_time_l42_42012

def slow_clock_time_ratio : ℚ := 14 / 15

noncomputable def slow_clock_to_correct_time (slow_clock_time : ℚ) : ℚ :=
  slow_clock_time / slow_clock_time_ratio

theorem clock_minutes_to_correct_time :
  slow_clock_to_correct_time 14 = 15 :=
by
  sorry

end clock_minutes_to_correct_time_l42_42012


namespace runs_by_running_percentage_l42_42829

def total_runs := 125
def boundaries := 5
def boundary_runs := boundaries * 4
def sixes := 5
def sixes_runs := sixes * 6
def runs_by_running := total_runs - (boundary_runs + sixes_runs)
def percentage_runs_by_running := (runs_by_running : ℚ) / total_runs * 100

theorem runs_by_running_percentage :
  percentage_runs_by_running = 60 := by sorry

end runs_by_running_percentage_l42_42829


namespace largest_prime_factor_1001_l42_42777

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l42_42777


namespace milk_production_group_B_l42_42461

theorem milk_production_group_B (a b c d e : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_pos_d : d > 0) (h_pos_e : e > 0) :
  ((1.2 * b * d * e) / (a * c)) = 1.2 * (b / (a * c)) * d * e := 
by
  sorry

end milk_production_group_B_l42_42461


namespace one_number_greater_than_one_l42_42592

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1)
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  ((1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ 1 < b ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ 1 < c)) 
  ∧ (¬ ((1 < a ∧ 1 < b) ∨ (1 < b ∧ 1 < c) ∨ (1 < a ∧ 1 < c))) :=
sorry

end one_number_greater_than_one_l42_42592


namespace ROI_difference_is_correct_l42_42050

noncomputable def compound_interest (P : ℝ) (rates : List ℝ) : ℝ :=
rates.foldl (λ acc rate => acc * (1 + rate)) P

noncomputable def Emma_investment := compound_interest 300 [0.15, 0.12, 0.18]

noncomputable def Briana_investment := compound_interest 500 [0.10, 0.08, 0.14]

noncomputable def ROI_difference := Briana_investment - Emma_investment

theorem ROI_difference_is_correct : ROI_difference = 220.808 := 
sorry

end ROI_difference_is_correct_l42_42050


namespace candy_count_correct_l42_42901

-- Define initial count of candy
def initial_candy : ℕ := 47

-- Define number of pieces of candy eaten
def eaten_candy : ℕ := 25

-- Define number of pieces of candy received
def received_candy : ℕ := 40

-- The final count of candy is what we are proving
theorem candy_count_correct : initial_candy - eaten_candy + received_candy = 62 :=
by
  sorry

end candy_count_correct_l42_42901


namespace find_quadratic_function_l42_42254

theorem find_quadratic_function (a h k x y : ℝ) (vertex_y : ℝ) (intersect_y : ℝ)
    (hv : h = 1 ∧ k = 2)
    (hi : x = 0 ∧ y = 3) :
    (∀ x, y = a * (x - h) ^ 2 + k) → vertex_y = h ∧ intersect_y = k →
    y = x^2 - 2 * x + 3 :=
by
  sorry

end find_quadratic_function_l42_42254


namespace chemist_mixing_solution_l42_42471

theorem chemist_mixing_solution (x : ℝ) : 0.30 * x = 0.20 * (x + 1) → x = 2 :=
by
  intro h
  sorry

end chemist_mixing_solution_l42_42471


namespace max_groups_l42_42984

def eggs : ℕ := 20
def marbles : ℕ := 6
def eggs_per_group : ℕ := 5
def marbles_per_group : ℕ := 2

def groups_of_eggs := eggs / eggs_per_group
def groups_of_marbles := marbles / marbles_per_group

theorem max_groups (h1 : eggs = 20) (h2 : marbles = 6) 
                    (h3 : eggs_per_group = 5) (h4 : marbles_per_group = 2) : 
                    min (groups_of_eggs) (groups_of_marbles) = 3 :=
by
  sorry

end max_groups_l42_42984


namespace error_percent_in_area_l42_42795

theorem error_percent_in_area 
    (L W : ℝ) 
    (measured_length : ℝ := 1.09 * L) 
    (measured_width : ℝ := 0.92 * W) 
    (correct_area : ℝ := L * W) 
    (incorrect_area : ℝ := measured_length * measured_width) :
    100 * (incorrect_area - correct_area) / correct_area = 0.28 :=
by
  sorry

end error_percent_in_area_l42_42795


namespace fourth_root_squared_cubed_l42_42192

theorem fourth_root_squared_cubed (x : ℝ) (h : (x^(1/4))^2^3 = 1296) : x = 256 :=
sorry

end fourth_root_squared_cubed_l42_42192


namespace prob_selecting_green_ball_l42_42159

-- Definition of the number of red and green balls in each container
def containerI_red := 10
def containerI_green := 5
def containerII_red := 3
def containerII_green := 5
def containerIII_red := 2
def containerIII_green := 6
def containerIV_red := 4
def containerIV_green := 4

-- Total number of balls in each container
def total_balls_I := containerI_red + containerI_green
def total_balls_II := containerII_red + containerII_green
def total_balls_III := containerIII_red + containerIII_green
def total_balls_IV := containerIV_red + containerIV_green

-- Probability of selecting a green ball from each container
def prob_green_I := containerI_green / total_balls_I
def prob_green_II := containerII_green / total_balls_II
def prob_green_III := containerIII_green / total_balls_III
def prob_green_IV := containerIV_green / total_balls_IV

-- Probability of selecting any one container
def prob_select_container := (1:ℚ) / 4

-- Combined probability for a green ball from each container
def combined_prob_I := prob_select_container * prob_green_I 
def combined_prob_II := prob_select_container * prob_green_II 
def combined_prob_III := prob_select_container * prob_green_III 
def combined_prob_IV := prob_select_container * prob_green_IV 

-- Total probability of selecting a green ball
def total_prob_green := combined_prob_I + combined_prob_II + combined_prob_III + combined_prob_IV 

-- Theorem to prove
theorem prob_selecting_green_ball : total_prob_green = 53 / 96 :=
by sorry

end prob_selecting_green_ball_l42_42159


namespace max_product_l42_42233

theorem max_product (a b : ℕ) (h1: a + b = 100) 
    (h2: a % 3 = 2) (h3: b % 7 = 5) : a * b ≤ 2491 := by
  sorry

end max_product_l42_42233


namespace evaluate_64_pow_3_div_2_l42_42576

theorem evaluate_64_pow_3_div_2 : (64 : ℝ)^(3/2) = 512 := by
  -- given 64 = 2^6
  have h : (64 : ℝ) = 2^6 := by norm_num
  -- use this substitution and properties of exponents
  rw [h, ←pow_mul]
  norm_num
  sorry -- completing the proof, not needed based on the guidelines

end evaluate_64_pow_3_div_2_l42_42576


namespace Jason_spent_on_jacket_l42_42059

/-
Given:
- Amount_spent_on_shorts: ℝ := 14.28
- Total_spent_on_clothing: ℝ := 19.02

Prove:
- Amount_spent_on_jacket = 4.74
-/
def Amount_spent_on_shorts : ℝ := 14.28
def Total_spent_on_clothing : ℝ := 19.02

-- We need to prove:
def Amount_spent_on_jacket : ℝ := Total_spent_on_clothing - Amount_spent_on_shorts 

theorem Jason_spent_on_jacket : Amount_spent_on_jacket = 4.74 := by
  sorry

end Jason_spent_on_jacket_l42_42059


namespace correct_statements_B_and_C_l42_42107

-- Given real numbers a, b, c satisfying the conditions
variables (a b c : ℝ)
variables (h1 : a > b)
variables (h2 : b > c)
variables (h3 : a + b + c = 0)

theorem correct_statements_B_and_C : (a - c > 2 * b) ∧ (a ^ 2 > b ^ 2) :=
by
  sorry

end correct_statements_B_and_C_l42_42107


namespace new_mean_correct_l42_42540

-- Define the original condition data
def initial_mean : ℝ := 42
def total_numbers : ℕ := 60
def discard1 : ℝ := 50
def discard2 : ℝ := 60
def increment : ℝ := 2

-- A function representing the new arithmetic mean
noncomputable def new_arithmetic_mean : ℝ :=
  let initial_sum := initial_mean * total_numbers
  let sum_after_discard := initial_sum - (discard1 + discard2)
  let sum_after_increment := sum_after_discard + (increment * (total_numbers - 2))
  sum_after_increment / (total_numbers - 2)

-- The theorem statement
theorem new_mean_correct : new_arithmetic_mean = 43.55 :=
by 
  sorry

end new_mean_correct_l42_42540


namespace unique_intersection_l42_42343

def line1 (x y : ℝ) : Prop := 3 * x - 2 * y - 9 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = -1

theorem unique_intersection : ∃! p : ℝ × ℝ, 
                             (line1 p.1 p.2) ∧ 
                             (line2 p.1 p.2) ∧ 
                             (line3 p.1) ∧ 
                             (line4 p.2) ∧ 
                             p = (3, -1) :=
by
  sorry

end unique_intersection_l42_42343


namespace raised_bed_height_l42_42610

theorem raised_bed_height : 
  ∀ (total_planks : ℕ) (num_beds : ℕ) (planks_per_bed : ℕ) (height : ℚ),
  total_planks = 50 →
  num_beds = 10 →
  planks_per_bed = 4 * height →
  (total_planks = num_beds * planks_per_bed) →
  height = 5 / 4 :=
by
  intros total_planks num_beds planks_per_bed H
  intros h1 h2 h3 h4
  sorry

end raised_bed_height_l42_42610


namespace frank_won_skee_ball_tickets_l42_42451

noncomputable def tickets_whack_a_mole : ℕ := 33
noncomputable def candies_bought : ℕ := 7
noncomputable def tickets_per_candy : ℕ := 6
noncomputable def total_tickets_spent : ℕ := candies_bought * tickets_per_candy
noncomputable def tickets_skee_ball : ℕ := total_tickets_spent - tickets_whack_a_mole

theorem frank_won_skee_ball_tickets : tickets_skee_ball = 9 :=
  by
  sorry

end frank_won_skee_ball_tickets_l42_42451


namespace Rachel_age_when_father_is_60_l42_42331

-- Given conditions
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Proof problem statement
theorem Rachel_age_when_father_is_60 : Rachel_age + (60 - Father_age) = 25 :=
by sorry

end Rachel_age_when_father_is_60_l42_42331


namespace can_cut_one_more_square_l42_42675

theorem can_cut_one_more_square (G : Finset (Fin 29 × Fin 29)) (hG : G.card = 99) :
  (∃ S : Finset (Fin 29 × Fin 29), S.card = 4 ∧ (S ⊆ G) ∧ (∀ s1 s2 : Fin 29 × Fin 29, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → (|s1.1 - s2.1| > 2 ∨ |s1.2 - s2.2| > 2))) :=
sorry

end can_cut_one_more_square_l42_42675


namespace intersects_x_axis_vertex_coordinates_l42_42780

-- Definition of the quadratic function and conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - a * x - 2 * a^2

-- Condition: a ≠ 0
axiom a_nonzero (a : ℝ) : a ≠ 0

-- Statement for the first part of the problem
theorem intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x₁ x₂ : ℝ, quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0 ∧ x₁ * x₂ < 0 :=
by 
  sorry

-- Statement for the second part of the problem
theorem vertex_coordinates (a : ℝ) (h : a ≠ 0) (hy_intercept : quadratic_function a 0 = -2) :
  ∃ x_vertex : ℝ, quadratic_function a x_vertex = (if a = 1 then (1/2)^2 - 9/4 else (1/2)^2 - 9/4) :=
by 
  sorry


end intersects_x_axis_vertex_coordinates_l42_42780


namespace partA_l42_42463

theorem partA (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, a * x ^ 2 + b * x + c = k ^ 4) : a = 0 ∧ b = 0 := 
sorry

end partA_l42_42463


namespace soccer_lineup_count_l42_42021

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem soccer_lineup_count : 
  let total_players := 18
  let goalies := 1
  let defenders := 6
  let forwards := 4
  18 * choose 17 6 * choose 11 4 = 73457760 :=
by
  sorry

end soccer_lineup_count_l42_42021


namespace symmetric_point_origin_l42_42828

theorem symmetric_point_origin (m : ℤ) : 
  (symmetry_condition : (3, m - 2) = (-(-3), -5)) → m = -3 :=
by
  sorry

end symmetric_point_origin_l42_42828


namespace parabola_and_hyperbola_tangent_l42_42255

theorem parabola_and_hyperbola_tangent (m : ℝ) :
  (∀ (x y : ℝ), (y = x^2 + 6) → (y^2 - m * x^2 = 6) → (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6)) :=
sorry

end parabola_and_hyperbola_tangent_l42_42255


namespace bug_traverses_36_tiles_l42_42009

-- Define the dimensions of the rectangle and the bug's problem setup
def width : ℕ := 12
def length : ℕ := 25

-- Define the function to calculate the number of tiles traversed by the bug
def tiles_traversed (w l : ℕ) : ℕ :=
  w + l - Nat.gcd w l

-- Prove the number of tiles traversed by the bug is 36
theorem bug_traverses_36_tiles : tiles_traversed width length = 36 :=
by
  -- This part will be proven; currently, we add sorry
  sorry

end bug_traverses_36_tiles_l42_42009


namespace largest_three_digit_multiple_of_12_and_sum_of_digits_24_l42_42679

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end largest_three_digit_multiple_of_12_and_sum_of_digits_24_l42_42679


namespace sequence_x_value_l42_42206

theorem sequence_x_value (p q r x : ℕ) 
  (h1 : 13 = 5 + p + q) 
  (h2 : r = p + q + 13) 
  (h3 : x = 13 + r + 40) : 
  x = 74 := 
by 
  sorry

end sequence_x_value_l42_42206


namespace find_m_root_zero_l42_42299

theorem find_m_root_zero (m : ℝ) : (m - 1) * 0 ^ 2 + 0 + m ^ 2 - 1 = 0 → m = -1 :=
by
  intro h
  sorry

end find_m_root_zero_l42_42299


namespace fraction_identity_l42_42722

theorem fraction_identity (a b : ℝ) (h : a / b = 3 / 4) : a / (a + b) = 3 / 7 := 
by
  sorry

end fraction_identity_l42_42722


namespace measure_six_liters_l42_42003

-- Given conditions as constants
def container_capacity : ℕ := 40
def ten_liter_bucket_capacity : ℕ := 10
def nine_liter_jug_capacity : ℕ := 9
def five_liter_jug_capacity : ℕ := 5

-- Goal: Measure out exactly 6 liters of milk using the above containers
theorem measure_six_liters (container : ℕ) (ten_bucket : ℕ) (nine_jug : ℕ) (five_jug : ℕ) :
  container = 40 →
  ten_bucket ≤ 10 →
  nine_jug ≤ 9 →
  five_jug ≤ 5 →
  ∃ (sequence_of_steps : ℕ → ℕ) (final_ten_bucket : ℕ),
    final_ten_bucket = 6 ∧ final_ten_bucket ≤ ten_bucket :=
by
  intro hcontainer hten_bucket hnine_jug hfive_jug
  sorry

end measure_six_liters_l42_42003


namespace isosceles_triangle_perimeter_l42_42783

/-
Problem:
Given an isosceles triangle with side lengths 5 and 6, prove that the perimeter of the triangle is either 16 or 17.
-/

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5 ∨ a = 6) (h₂ : b = 5 ∨ b = 6) (h₃ : a ≠ b) : 
  (a + a + b = 16 ∨ a + a + b = 17) ∧ (b + b + a = 16 ∨ b + b + a = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l42_42783


namespace sum_primes_reversed_l42_42123

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def valid_primes : List ℕ := [31, 37, 71, 73]

theorem sum_primes_reversed :
  (∀ p ∈ valid_primes, 20 < p ∧ p < 80 ∧ is_prime p ∧ is_prime (reverse_digits p)) ∧
  (valid_primes.sum = 212) :=
by
  sorry

end sum_primes_reversed_l42_42123


namespace general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l42_42112

def seq_a : ℕ → ℕ 
| 0 => 0  -- a_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2^(n+1)

def seq_b : ℕ → ℕ 
| 0 => 0  -- b_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2*(n+1) -1

def sum_S (n : ℕ) : ℕ := (seq_a (n+1) * 2) - 2

def sum_T : ℕ → ℕ 
| 0 => 0  -- T_0 is not defined in natural numbers, put it as zero for base case too
| (n+1) => (n+1)^2

def sum_D : ℕ → ℕ
| 0 => 0
| (n+1) => (seq_a (n+1) * seq_b (n+1)) + sum_D n

theorem general_term_a_n (n : ℕ) : seq_a n = 2^n := sorry

theorem general_term_b_n (n : ℕ) : seq_b n = 2*n - 1 := sorry

theorem sum_of_first_n_terms_D_n (n : ℕ) : sum_D n = (2*n - 3)*2^(n+1) + 6 := sorry

end general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l42_42112


namespace carlos_laundry_l42_42564

theorem carlos_laundry (n : ℕ) 
  (h1 : 45 * n + 75 = 165) : n = 2 :=
by
  sorry

end carlos_laundry_l42_42564


namespace simple_interest_rate_l42_42330

theorem simple_interest_rate (P A T : ℝ) (H1 : P = 1750) (H2 : A = 2000) (H3 : T = 4) :
  ∃ R : ℝ, R = 3.57 ∧ A = P * (1 + (R * T) / 100) :=
by
  sorry

end simple_interest_rate_l42_42330


namespace milk_production_l42_42238

variables (x α y z w β v : ℝ)

theorem milk_production :
  (w * v * β * y) / (α^2 * x * z) = β * y * w * v / (α^2 * x * z) := 
by
  sorry

end milk_production_l42_42238


namespace minimum_AP_BP_l42_42637

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 3)
def parabola (P : ℝ × ℝ) : Prop := P.2 * P.2 = 8 * P.1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

theorem minimum_AP_BP : 
  ∀ (P : ℝ × ℝ), parabola P → distance A P + distance B P ≥ 3 * Real.sqrt 10 :=
by 
  intros P hP
  sorry

end minimum_AP_BP_l42_42637


namespace total_carrots_l42_42882

theorem total_carrots (carrots_sandy carrots_mary : ℕ) (h1 : carrots_sandy = 8) (h2 : carrots_mary = 6) :
  carrots_sandy + carrots_mary = 14 :=
by
  sorry

end total_carrots_l42_42882


namespace sufficient_but_not_necessary_not_necessary_l42_42611

variable (x y : ℝ)

theorem sufficient_but_not_necessary (h1: x ≥ 2) (h2: y ≥ 2): x^2 + y^2 ≥ 4 :=
by
  sorry

theorem not_necessary (hx4 : x^2 + y^2 ≥ 4) : ¬ (x ≥ 2 ∧ y ≥ 2) → ∃ x y, (x^2 + y^2 ≥ 4) ∧ (¬ (x ≥ 2) ∨ ¬ (y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l42_42611


namespace total_drawing_sheets_l42_42635

-- Definitions based on the conditions given
def brown_sheets := 28
def yellow_sheets := 27

-- The statement we need to prove
theorem total_drawing_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end total_drawing_sheets_l42_42635


namespace inserting_eights_is_composite_l42_42197

theorem inserting_eights_is_composite (n : ℕ) : ¬ Nat.Prime (2000 * 10^n + 8 * ((10^n - 1) / 9) + 21) := 
by sorry

end inserting_eights_is_composite_l42_42197


namespace smallest_positive_perfect_cube_l42_42856

theorem smallest_positive_perfect_cube (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ m : ℕ, m = (a * b * c^2)^3 ∧ (a^2 * b^3 * c^5 ∣ m)
:=
sorry

end smallest_positive_perfect_cube_l42_42856


namespace card_drawing_ways_l42_42975

theorem card_drawing_ways :
  (30 * 20 = 600) :=
by
  sorry

end card_drawing_ways_l42_42975


namespace upgrade_days_to_sun_l42_42290

/-- 
  Determine the minimum number of additional active days required for 
  a user currently at level 2 moons and 1 star to upgrade to 1 sun.
-/
theorem upgrade_days_to_sun (level_new_star : ℕ) (level_new_moon : ℕ) (active_days_initial : ℕ) : 
  active_days_initial =  9 * (9 + 4) → 
  level_new_star = 1 → 
  level_new_moon = 2 → 
  ∃ (days_required : ℕ), 
    (days_required + active_days_initial = 16 * (16 + 4)) ∧ (days_required = 203) :=
by
  sorry

end upgrade_days_to_sun_l42_42290


namespace least_number_to_add_to_4499_is_1_l42_42734

theorem least_number_to_add_to_4499_is_1 (x : ℕ) : (4499 + x) % 9 = 0 → x = 1 := sorry

end least_number_to_add_to_4499_is_1_l42_42734


namespace solve_z_l42_42905

noncomputable def complex_equation (z : ℂ) := (1 + 3 * Complex.I) * z = Complex.I - 3

theorem solve_z (z : ℂ) (h : complex_equation z) : z = Complex.I :=
by
  sorry

end solve_z_l42_42905


namespace find_rates_l42_42799

theorem find_rates
  (d b p t_p t_b t_w: ℕ)
  (rp rb rw: ℚ)
  (h1: d = b + 10)
  (h2: b = 3 * p)
  (h3: p = 50)
  (h4: t_p = 4)
  (h5: t_b = 2)
  (h6: t_w = 5)
  (h7: rp = p / t_p)
  (h8: rb = b / t_b)
  (h9: rw = d / t_w):
  rp = 12.5 ∧ rb = 75 ∧ rw = 32 := by
  sorry

end find_rates_l42_42799


namespace simplify_expression_l42_42717

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end simplify_expression_l42_42717


namespace SamaraSpentOnDetailing_l42_42194

def costSamara (D : ℝ) : ℝ := 25 + 467 + D
def costAlberto : ℝ := 2457
def difference : ℝ := 1886

theorem SamaraSpentOnDetailing : 
  ∃ (D : ℝ), costAlberto = costSamara D + difference ∧ D = 79 := 
sorry

end SamaraSpentOnDetailing_l42_42194


namespace range_of_m_l42_42890

/-- Given the conditions:
- \( \left|1 - \frac{x - 2}{3}\right| \leq 2 \)
- \( x^2 - 2x + 1 - m^2 \leq 0 \) where \( m > 0 \)
- \( \neg \left( \left|1 - \frac{x - 2}{3}\right| \leq 2 \right) \) is a necessary but not sufficient condition for \( x^2 - 2x + 1 - m^2 \leq 0 \)

Prove that the range of \( m \) is \( m \geq 10 \).
-/
theorem range_of_m (m : ℝ) (x : ℝ)
  (h1 : ∀ x, ¬(abs (1 - (x - 2) / 3) ≤ 2) → x < -1 ∨ x > 11)
  (h2 : ∀ x, ∀ m > 0, x^2 - 2 * x + 1 - m^2 ≤ 0)
  : m ≥ 10 :=
sorry

end range_of_m_l42_42890


namespace number_of_possible_lengths_of_diagonal_l42_42957

theorem number_of_possible_lengths_of_diagonal :
  ∃ n : ℕ, n = 13 ∧
  (∀ y : ℕ, (5 ≤ y ∧ y ≤ 17) ↔ (y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨
   y = 10 ∨ y = 11 ∨ y = 12 ∨ y = 13 ∨ y = 14 ∨ y = 15 ∨ y = 16 ∨ y = 17)) :=
by
  exists 13
  sorry

end number_of_possible_lengths_of_diagonal_l42_42957


namespace hundredths_digit_of_power_l42_42359

theorem hundredths_digit_of_power (n : ℕ) (h : n % 20 = 14) : 
  (8 ^ n % 1000) / 100 = 1 :=
by sorry

lemma test_power_hundredths_digit : (8 ^ 1234 % 1000) / 100 = 1 :=
hundredths_digit_of_power 1234 (by norm_num)

end hundredths_digit_of_power_l42_42359


namespace arithmetic_seq_first_term_l42_42916

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (n : ℕ) (a : ℚ)
  (h₁ : ∀ n, S n = n * (2 * a + (n - 1) * 5) / 2)
  (h₂ : ∀ n, S (3 * n) / S n = 9) :
  a = 5 / 2 :=
by
  sorry

end arithmetic_seq_first_term_l42_42916


namespace trig_identity_l42_42650

open Real

theorem trig_identity :
  3.4173 * sin (2 * pi / 17) + sin (4 * pi / 17) - sin (6 * pi / 17) - (1/2) * sin (8 * pi / 17) =
  8 * (sin (2 * pi / 17))^3 * (cos (pi / 17))^2 :=
by sorry

end trig_identity_l42_42650


namespace total_surface_area_of_resulting_structure_l42_42986

-- Definitions for the conditions
def bigCube := 12 * 12 * 12
def smallCube := 2 * 2 * 2
def totalSmallCubes := 64
def removedCubes := 7
def remainingCubes := totalSmallCubes - removedCubes
def surfaceAreaPerSmallCube := 24
def extraExposedSurfaceArea := 6
def effectiveSurfaceAreaPerSmallCube := surfaceAreaPerSmallCube + extraExposedSurfaceArea

-- Definition and the main statement of the proof problem.
def totalSurfaceArea := remainingCubes * effectiveSurfaceAreaPerSmallCube

theorem total_surface_area_of_resulting_structure : totalSurfaceArea = 1710 :=
by
  sorry

end total_surface_area_of_resulting_structure_l42_42986


namespace number_of_buses_l42_42929

theorem number_of_buses (total_supervisors : ℕ) (supervisors_per_bus : ℕ) (h1 : total_supervisors = 21) (h2 : supervisors_per_bus = 3) : total_supervisors / supervisors_per_bus = 7 :=
by
  sorry

end number_of_buses_l42_42929


namespace geometric_sequence_eighth_term_l42_42775

variable (a r : ℕ)
variable (h1 : a = 3)
variable (h2 : a * r^6 = 2187)
variable (h3 : a = 3)

theorem geometric_sequence_eighth_term (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) (h3 : a = 3) :
  a * r^7 = 6561 := by
  sorry

end geometric_sequence_eighth_term_l42_42775


namespace number_of_carbons_l42_42316

-- Definitions of given conditions
def molecular_weight (total_c total_h total_o c_weight h_weight o_weight : ℕ) := 
    total_c * c_weight + total_h * h_weight + total_o * o_weight

-- Given values
def num_hydrogen_atoms : ℕ := 8
def num_oxygen_atoms : ℕ := 2
def molecular_wt : ℕ := 88
def atomic_weight_c : ℕ := 12
def atomic_weight_h : ℕ := 1
def atomic_weight_o : ℕ := 16

-- The theorem to be proved
theorem number_of_carbons (num_carbons : ℕ) 
    (H_hydrogen : num_hydrogen_atoms = 8)
    (H_oxygen : num_oxygen_atoms = 2)
    (H_molecular_weight : molecular_wt = 88)
    (H_atomic_weight_c : atomic_weight_c = 12)
    (H_atomic_weight_h : atomic_weight_h = 1)
    (H_atomic_weight_o : atomic_weight_o = 16) :
    molecular_weight num_carbons num_hydrogen_atoms num_oxygen_atoms atomic_weight_c atomic_weight_h atomic_weight_o = molecular_wt → 
    num_carbons = 4 :=
by
  intros h
  sorry 

end number_of_carbons_l42_42316


namespace B_finish_work_in_10_days_l42_42236

variable (W : ℝ) -- amount of work
variable (x : ℝ) -- number of days B can finish the work alone

theorem B_finish_work_in_10_days (h1 : ∀ A_rate, A_rate = W / 4)
                                (h2 : ∀ B_rate, B_rate = W / x)
                                (h3 : ∀ Work_done_together Remaining_work,
                                      Work_done_together = 2 * (W / 4 + W / x) ∧
                                      Remaining_work = W - Work_done_together ∧
                                      Remaining_work = (W / x) * 3.0000000000000004) :
  x = 10 :=
by
  sorry

end B_finish_work_in_10_days_l42_42236


namespace tan_eq_243_deg_l42_42329

theorem tan_eq_243_deg (n : ℤ) : -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (243 * Real.pi / 180) ↔ n = 63 :=
by sorry

end tan_eq_243_deg_l42_42329


namespace loss_is_selling_price_of_16_pencils_l42_42122

theorem loss_is_selling_price_of_16_pencils
  (S : ℝ) -- Assume the selling price of one pencil is S
  (C : ℝ) -- Assume the cost price of one pencil is C
  (h₁ : 80 * C = 1.2 * 80 * S) -- The cost of 80 pencils is 1.2 times the selling price of 80 pencils
  : (80 * C - 80 * S) = 16 * S := -- The loss for selling 80 pencils equals the selling price of 16 pencils
  sorry

end loss_is_selling_price_of_16_pencils_l42_42122


namespace halfway_between_one_eighth_and_one_third_l42_42939

theorem halfway_between_one_eighth_and_one_third : (1/8 + 1/3) / 2 = 11/48 :=
by
  sorry

end halfway_between_one_eighth_and_one_third_l42_42939


namespace calc_a_minus_3b_l42_42353

noncomputable def a : ℂ := 5 - 3 * Complex.I
noncomputable def b : ℂ := 2 + 3 * Complex.I

theorem calc_a_minus_3b : a - 3 * b = -1 - 12 * Complex.I := by
  sorry

end calc_a_minus_3b_l42_42353


namespace C_investment_is_20000_l42_42989

-- Definitions of investments and profits
def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def total_profit : ℕ := 86400
def C_share_of_profit : ℕ := 36000

-- The proof problem statement
theorem C_investment_is_20000 (X : ℕ) (hA : A_investment = 12000) (hB : B_investment = 16000)
  (h_total_profit : total_profit = 86400) (h_C_share_of_profit : C_share_of_profit = 36000) :
  X = 20000 :=
sorry

end C_investment_is_20000_l42_42989


namespace area_of_quadrilateral_APQC_l42_42670

-- Define the geometric entities and conditions
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_eq: c = Real.sqrt (a ^ 2 + b ^ 2))

-- Triangles PAQ and PQC are right triangles with given sides
def PAQ := RightTriangle 9 12 (Real.sqrt (9^2 + 12^2))
def PQC := RightTriangle 12 9 (Real.sqrt (15^2 - 12^2))

-- Prove that the area of quadrilateral APQC is 108 square units
theorem area_of_quadrilateral_APQC :
  let area_PAQ := 1/2 * 9 * 12
  let area_PQC := 1/2 * 12 * 9
  area_PAQ + area_PQC = 108 :=
by
  sorry

end area_of_quadrilateral_APQC_l42_42670


namespace no_rational_satisfies_l42_42506

theorem no_rational_satisfies (a b c d : ℚ) : ¬ ((a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 = 1 + Real.sqrt 3) :=
sorry

end no_rational_satisfies_l42_42506


namespace randy_total_trees_l42_42393

theorem randy_total_trees (mango_trees : ℕ) (coconut_trees : ℕ) 
  (h1 : mango_trees = 60) 
  (h2 : coconut_trees = (mango_trees / 2) - 5) : 
  mango_trees + coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l42_42393


namespace fraction_simplification_l42_42297

theorem fraction_simplification :
  (3 / 7 + 5 / 8 + 2 / 9) / (5 / 12 + 1 / 4) = 643 / 336 :=
by
  sorry

end fraction_simplification_l42_42297


namespace Petya_wrong_example_l42_42411

def a := 8
def b := 128

theorem Petya_wrong_example : (a^7 ∣ b^3) ∧ ¬ (a^2 ∣ b) :=
by {
  -- Prove the divisibility conditions and the counterexample
  sorry
}

end Petya_wrong_example_l42_42411


namespace students_on_right_side_l42_42836

-- Define the total number of students and the number of students on the left side
def total_students : ℕ := 63
def left_students : ℕ := 36

-- Define the number of students on the right side using subtraction
def right_students (total_students left_students : ℕ) : ℕ := total_students - left_students

-- Theorem: Prove that the number of students on the right side is 27
theorem students_on_right_side : right_students total_students left_students = 27 := by
  sorry

end students_on_right_side_l42_42836


namespace hundredth_odd_integer_l42_42965

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l42_42965


namespace abs_inequality_range_l42_42342

theorem abs_inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 6| > a) ↔ a < 5 :=
by
  sorry

end abs_inequality_range_l42_42342


namespace value_of_J_l42_42232

theorem value_of_J (J : ℕ) : 32^4 * 4^4 = 2^J → J = 28 :=
by
  intro h
  sorry

end value_of_J_l42_42232


namespace neither_long_furred_nor_brown_dogs_is_8_l42_42469

def total_dogs : ℕ := 45
def long_furred_dogs : ℕ := 29
def brown_dogs : ℕ := 17
def long_furred_and_brown_dogs : ℕ := 9

def neither_long_furred_nor_brown_dogs : ℕ :=
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_and_brown_dogs)

theorem neither_long_furred_nor_brown_dogs_is_8 :
  neither_long_furred_nor_brown_dogs = 8 := 
by 
  -- Here we can use substitution and calculation steps used in the solution
  sorry

end neither_long_furred_nor_brown_dogs_is_8_l42_42469


namespace find_a_l42_42455

-- Given conditions
variables (x y z a : ℤ)

def conditions : Prop :=
  (x - 10) * (y - a) * (z - 2) = 1000 ∧
  ∃ (x y z : ℤ), x + y + z = 7

theorem find_a (x y z : ℤ) (h : conditions x y z 1) : a = 1 := 
  by
    sorry

end find_a_l42_42455


namespace fish_upstream_speed_l42_42398

def Vs : ℝ := 45
def Vdownstream : ℝ := 55

def Vupstream (Vs Vw : ℝ) : ℝ := Vs - Vw
def Vstream (Vs Vdownstream : ℝ) : ℝ := Vdownstream - Vs

theorem fish_upstream_speed :
  Vupstream Vs (Vstream Vs Vdownstream) = 35 := by
  sorry

end fish_upstream_speed_l42_42398


namespace calc_nabla_example_l42_42755

-- Define the custom operation ∇
def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- State the proof problem
theorem calc_nabla_example : op_nabla (op_nabla 2 3) (op_nabla 4 5) = 49 / 56 := by
  sorry

end calc_nabla_example_l42_42755


namespace recurrence_solution_proof_l42_42099

noncomputable def recurrence_relation (a : ℕ → ℚ) : Prop :=
  (∀ n ≥ 2, a n = 5 * a (n - 1) - 6 * a (n - 2) + n + 2) ∧
  a 0 = 27 / 4 ∧
  a 1 = 49 / 4

noncomputable def solution (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = 3 * 2^n + 3^n + n / 2 + 11 / 4

theorem recurrence_solution_proof : ∃ a : ℕ → ℚ, recurrence_relation a ∧ solution a :=
by { sorry }

end recurrence_solution_proof_l42_42099


namespace discount_difference_is_24_l42_42212

-- Definitions based on conditions
def smartphone_price : ℝ := 800
def single_discount_rate : ℝ := 0.25
def first_successive_discount_rate : ℝ := 0.20
def second_successive_discount_rate : ℝ := 0.10

-- Definitions of discounted prices
def single_discount_price (p : ℝ) (d1 : ℝ) : ℝ := p * (1 - d1)
def successive_discount_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ := 
  let intermediate_price := p * (1 - d1) 
  intermediate_price * (1 - d2)

-- Calculate the difference between the two final prices
def price_difference (p : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (single_discount_price p d1) - (successive_discount_price p d2 d3)

theorem discount_difference_is_24 :
  price_difference smartphone_price single_discount_rate first_successive_discount_rate second_successive_discount_rate = 24 := 
sorry

end discount_difference_is_24_l42_42212


namespace sum_of_two_numbers_l42_42352

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = x + 4) (h2 : y = 30) : x + y = 56 :=
by
  -- Asserts the conditions and goal statement
  sorry

end sum_of_two_numbers_l42_42352


namespace ascorbic_acid_weight_l42_42531

def molecular_weight (formula : String) : ℝ :=
  if formula = "C6H8O6" then 176.12 else 0

theorem ascorbic_acid_weight : molecular_weight "C6H8O6" = 176.12 :=
by {
  sorry
}

end ascorbic_acid_weight_l42_42531


namespace total_boys_in_class_l42_42862

theorem total_boys_in_class (n : ℕ) (h_circle : ∀ i, 1 ≤ i ∧ i ≤ n -> i ≤ n) 
  (h_opposite : ∀ j k, j = 7 ∧ k = 27 ∧ j < k -> (k - j = n / 2)) : 
  n = 40 :=
sorry

end total_boys_in_class_l42_42862


namespace value_of_f_sum_l42_42496

variable (a b c m : ℝ)

def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f_sum :
  f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end value_of_f_sum_l42_42496


namespace monomial_degree_and_coefficient_l42_42406

theorem monomial_degree_and_coefficient (a b : ℤ) (h1 : -a = 7) (h2 : 1 + b = 4) : a + b = -4 :=
by
  sorry

end monomial_degree_and_coefficient_l42_42406


namespace min_value_expression_l42_42154

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) (hxy : x + y = 6) : 
  ( (x - 1)^2 / (y - 2) + ( (y - 1)^2 / (x - 2) ) ) >= 8 :=
by 
  sorry

end min_value_expression_l42_42154


namespace a1_geq_2_pow_k_l42_42389

-- Definitions of the problem conditions in Lean 4
def conditions (a : ℕ → ℕ) (n k : ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i < 2 * n) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → ¬(a i ∣ a j)) ∧
  (3^k < 2 * n ∧ 2 * n < 3^(k+1))

-- The main theorem to be proven
theorem a1_geq_2_pow_k (a : ℕ → ℕ) (n k : ℕ) (h : conditions a n k) : 
  a 1 ≥ 2^k :=
sorry

end a1_geq_2_pow_k_l42_42389


namespace max_wx_plus_xy_plus_yz_plus_wz_l42_42218

theorem max_wx_plus_xy_plus_yz_plus_wz (w x y z : ℝ) (h_nonneg : 0 ≤ w ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : w + x + y + z = 200) :
  wx + xy + yz + wz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_plus_wz_l42_42218


namespace scientific_notation_l42_42014

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end scientific_notation_l42_42014


namespace distinct_real_solutions_l42_42372

theorem distinct_real_solutions
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x₁ x₂ x₃ x₄ : ℝ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - d) +
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - e) +
    (x₁ - a) * (x₁ - b) * (x₁ - d) * (x₁ - e) +
    (x₁ - a) * (x₁ - c) * (x₁ - d) * (x₁ - e) +
    (x₁ - b) * (x₁ - c) * (x₁ - d) * (x₁ - e) = 0 ∧
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - d) +
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - e) +
    (x₂ - a) * (x₂ - b) * (x₂ - d) * (x₂ - e) +
    (x₂ - a) * (x₂ - c) * (x₂ - d) * (x₂ - e) +
    (x₂ - b) * (x₂ - c) * (x₂ - d) * (x₂ - e) = 0 ∧
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - d) +
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - e) +
    (x₃ - a) * (x₃ - b) * (x₃ - d) * (x₃ - e) +
    (x₃ - a) * (x₃ - c) * (x₃ - d) * (x₃ - e) +
    (x₃ - b) * (x₃ - c) * (x₃ - d) * (x₃ - e) = 0 ∧
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - d) +
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - e) +
    (x₄ - a) * (x₄ - b) * (x₄ - d) * (x₄ - e) +
    (x₄ - a) * (x₄ - c) * (x₄ - d) * (x₄ - e) +
    (x₄ - b) * (x₄ - c) * (x₄ - d) * (x₄ - e) = 0 :=
  sorry

end distinct_real_solutions_l42_42372


namespace find_line_equation_l42_42899

theorem find_line_equation : 
  ∃ c : ℝ, (∀ x y : ℝ, 2*x + 4*y + c = 0 ↔ x + 2*y - 8 = 0) ∧ (2*2 + 4*3 + c = 0) :=
sorry

end find_line_equation_l42_42899


namespace arrangement_count_l42_42870

-- Definitions from the conditions
def people : Nat := 5
def valid_positions_for_A : Finset Nat := Finset.range 5 \ {0, 4}

-- The theorem that states the question equals the correct answer given the conditions
theorem arrangement_count (A_positions : Finset Nat := valid_positions_for_A) : 
  ∃ (total_arrangements : Nat), total_arrangements = 72 :=
by
  -- Placeholder for the proof
  sorry

end arrangement_count_l42_42870


namespace car_race_probability_l42_42468

theorem car_race_probability :
  let pX := 1/8
  let pY := 1/12
  let pZ := 1/6
  pX + pY + pZ = 3/8 :=
by
  sorry

end car_race_probability_l42_42468


namespace geometric_progression_condition_l42_42445

variables (a b c : ℝ) (k n p : ℕ)

theorem geometric_progression_condition :
  (a / b) ^ (k - p) = (a / c) ^ (k - n) :=
sorry

end geometric_progression_condition_l42_42445


namespace point_B_value_l42_42655

theorem point_B_value (A : ℝ) (B : ℝ) (hA : A = -5) (hB : B = -1 ∨ B = -9) :
  ∃ B : ℝ, (B = A + 4 ∨ B = A - 4) :=
by sorry

end point_B_value_l42_42655


namespace rob_total_cards_l42_42852

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end rob_total_cards_l42_42852


namespace nonneg_real_inequality_l42_42195

theorem nonneg_real_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := 
by
  sorry

end nonneg_real_inequality_l42_42195


namespace rational_roots_of_polynomial_l42_42840

theorem rational_roots_of_polynomial :
  { x : ℚ | (x + 1) * (x - (2 / 3)) * (x^2 - 2) = 0 } = {-1, 2 / 3} :=
by
  sorry

end rational_roots_of_polynomial_l42_42840


namespace mia_has_largest_final_value_l42_42871

def daniel_final : ℕ := (12 * 2 - 3 + 5)
def mia_final : ℕ := ((15 - 2) * 2 + 3)
def carlos_final : ℕ := (13 * 2 - 4 + 6)

theorem mia_has_largest_final_value : mia_final > daniel_final ∧ mia_final > carlos_final := by
  sorry

end mia_has_largest_final_value_l42_42871


namespace harry_travel_ratio_l42_42831

theorem harry_travel_ratio
  (bus_initial_time : ℕ)
  (bus_rest_time : ℕ)
  (total_travel_time : ℕ)
  (walking_time : ℕ := total_travel_time - (bus_initial_time + bus_rest_time))
  (bus_total_time : ℕ := bus_initial_time + bus_rest_time)
  (ratio : ℚ := walking_time / bus_total_time)
  (h1 : bus_initial_time = 15)
  (h2 : bus_rest_time = 25)
  (h3 : total_travel_time = 60)
  : ratio = (1 / 2) := 
sorry

end harry_travel_ratio_l42_42831


namespace honey_harvested_correct_l42_42082

def honey_harvested_last_year : ℕ := 2479
def honey_increase_this_year : ℕ := 6085
def honey_harvested_this_year : ℕ := 8564

theorem honey_harvested_correct :
  honey_harvested_last_year + honey_increase_this_year = honey_harvested_this_year :=
sorry

end honey_harvested_correct_l42_42082


namespace milk_leftover_l42_42785

variable {v : ℕ} -- 'v' is the number of sets of milkshakes in the 2:1 ratio.
variables {milk vanilla_chocolate : ℕ} -- spoon amounts per milkshake types
variables {total_milk total_vanilla_ice_cream total_chocolate_ice_cream : ℕ} -- total amount constraints
variables {milk_left : ℕ} -- amount of milk left after

-- Definitions based on the conditions
def milk_per_vanilla := 4
def milk_per_chocolate := 5
def ice_vanilla_per_milkshake := 12
def ice_chocolate_per_milkshake := 10
def initial_milk := 72
def initial_vanilla_ice_cream := 96
def initial_chocolate_ice_cream := 96

-- Constraints
def max_milkshakes := 16
def milk_needed (v : ℕ) := (4 * 2 * v) + (5 * v)
def vanilla_needed (v : ℕ) := 12 * 2 * v
def chocolate_needed (v : ℕ) := 10 * v 

-- Inequalities
lemma milk_constraint (v : ℕ) : milk_needed v ≤ initial_milk := sorry

lemma vanilla_constraint (v : ℕ) : vanilla_needed v ≤ initial_vanilla_ice_cream := sorry

lemma chocolate_constraint (v : ℕ) : chocolate_needed v ≤ initial_chocolate_ice_cream := sorry

lemma total_milkshakes_constraint (v : ℕ) : 3 * v ≤ max_milkshakes := sorry

-- Conclusion
theorem milk_leftover : milk_left = initial_milk - milk_needed 5 := sorry

end milk_leftover_l42_42785


namespace max_fraction_l42_42864

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  1 + y / x ≤ -2 :=
sorry

end max_fraction_l42_42864


namespace length_increase_percentage_l42_42312

theorem length_increase_percentage 
  (L B : ℝ)
  (x : ℝ)
  (h1 : B' = B * 0.8)
  (h2 : L' = L * (1 + x / 100))
  (h3 : A = L * B)
  (h4 : A' = L' * B')
  (h5 : A' = A * 1.04) 
  : x = 30 :=
sorry

end length_increase_percentage_l42_42312


namespace three_digit_identical_divisible_by_37_l42_42338

theorem three_digit_identical_divisible_by_37 (A : ℕ) (h : A ≤ 9) : 37 ∣ (111 * A) :=
sorry

end three_digit_identical_divisible_by_37_l42_42338


namespace no_intersection_l42_42762

-- Definitions of the sets M1 and M2 based on parameters A, B, C and integer x
def M1 (A B : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = x^2 + A * x + B}
def M2 (C : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = 2 * x^2 + 2 * x + C}

-- The statement of the theorem
theorem no_intersection (A B : ℤ) : ∃ C : ℤ, M1 A B ∩ M2 C = ∅ :=
sorry

end no_intersection_l42_42762


namespace decreasing_interval_of_even_function_l42_42767

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k*x^2 + (k - 1)*x + 2

theorem decreasing_interval_of_even_function (k : ℝ) (h : ∀ x : ℝ, f k x = f k (-x)) :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, (x < 0 → f k x > f k (-x)) := 
sorry

end decreasing_interval_of_even_function_l42_42767


namespace total_apples_l42_42075

variable (A : ℕ)
variables (too_small not_ripe perfect : ℕ)

-- Conditions
axiom small_fraction : too_small = A / 6
axiom ripe_fraction  : not_ripe = A / 3
axiom remaining_fraction : perfect = A / 2
axiom perfect_count : perfect = 15

theorem total_apples : A = 30 :=
sorry

end total_apples_l42_42075


namespace socks_thrown_away_l42_42620

theorem socks_thrown_away 
  (initial_socks new_socks current_socks : ℕ) 
  (h1 : initial_socks = 11) 
  (h2 : new_socks = 26) 
  (h3 : current_socks = 33) : 
  initial_socks + new_socks - current_socks = 4 :=
by {
  sorry
}

end socks_thrown_away_l42_42620


namespace units_digit_of_ksq_plus_2k_l42_42241

def k := 2023^3 - 3^2023

theorem units_digit_of_ksq_plus_2k : (k^2 + 2^k) % 10 = 1 := 
  sorry

end units_digit_of_ksq_plus_2k_l42_42241


namespace distinct_x_intercepts_l42_42565

-- Given conditions
def polynomial (x : ℝ) : ℝ := (x - 4) * (x^2 + 4 * x + 13)

-- Statement of the problem as a Lean theorem
theorem distinct_x_intercepts : 
  (∃ (x : ℝ), polynomial x = 0 ∧ 
    ∀ (y : ℝ), y ≠ x → polynomial y = 0 → False) :=
  sorry

end distinct_x_intercepts_l42_42565


namespace union_complement_A_B_eq_U_l42_42955

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5, 7}
def A : Set ℕ := {4, 7}
def B : Set ℕ := {1, 3, 4, 7}

-- Define the complement of A with respect to U (C_U A)
def C_U_A : Set ℕ := U \ A
-- Define the complement of B with respect to U (C_U B)
def C_U_B : Set ℕ := U \ B

-- The theorem to prove
theorem union_complement_A_B_eq_U : (C_U_A ∪ B) = U := by
  sorry

end union_complement_A_B_eq_U_l42_42955


namespace students_both_l42_42710

noncomputable def students_total : ℕ := 32
noncomputable def students_go : ℕ := 18
noncomputable def students_chess : ℕ := 23

theorem students_both : students_go + students_chess - students_total = 9 := by
  sorry

end students_both_l42_42710


namespace probability_even_product_l42_42549

-- Define spinner A and spinner C
def SpinnerA : List ℕ := [1, 2, 3, 4]
def SpinnerC : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define even and odd number sets for Spinner A and Spinner C
def evenNumbersA : List ℕ := [2, 4]
def oddNumbersA : List ℕ := [1, 3]

def evenNumbersC : List ℕ := [2, 4, 6]
def oddNumbersC : List ℕ := [1, 3, 5]

-- Define a function to check if a product is even
def isEven (n : ℕ) : Bool := n % 2 == 0

-- Probability calculation
def evenProductProbability : ℚ :=
  let totalOutcomes := (SpinnerA.length * SpinnerC.length)
  let evenA_outcomes := (evenNumbersA.length * SpinnerC.length)
  let oddA_evenC_outcomes := (oddNumbersA.length * evenNumbersC.length)
  (evenA_outcomes + oddA_evenC_outcomes) / totalOutcomes

theorem probability_even_product :
  evenProductProbability = 3 / 4 :=
by
  sorry

end probability_even_product_l42_42549


namespace motorcyclist_cross_time_l42_42131

/-- Definitions and conditions -/
def speed_X := 2 -- Rounds per hour
def speed_Y := 4 -- Rounds per hour

/-- Proof statement -/
theorem motorcyclist_cross_time : (1 / (speed_X + speed_Y) * 60 = 10) :=
by
  sorry

end motorcyclist_cross_time_l42_42131


namespace find_m_l42_42210

theorem find_m (m : ℝ) (a b : ℝ × ℝ)
  (ha : a = (3, m)) (hb : b = (1, -2))
  (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) :
  m = -1 :=
by {
  sorry
}

end find_m_l42_42210


namespace cookies_fit_in_box_l42_42280

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end cookies_fit_in_box_l42_42280


namespace nat_pairs_solution_l42_42147

theorem nat_pairs_solution (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) :=
by
  sorry

end nat_pairs_solution_l42_42147


namespace percentage_seniors_with_cars_is_40_l42_42071

noncomputable def percentage_of_seniors_with_cars 
  (total_students: ℕ) (seniors: ℕ) (lower_grades: ℕ) (percent_cars_all: ℚ) (percent_cars_lower_grades: ℚ) : ℚ :=
  let total_with_cars := percent_cars_all * total_students
  let lower_grades_with_cars := percent_cars_lower_grades * lower_grades
  let seniors_with_cars := total_with_cars - lower_grades_with_cars
  (seniors_with_cars / seniors) * 100

theorem percentage_seniors_with_cars_is_40
  : percentage_of_seniors_with_cars 1800 300 1500 0.15 0.10 = 40 := 
by
  -- Proof is omitted
  sorry

end percentage_seniors_with_cars_is_40_l42_42071


namespace find_z_coordinate_of_point_on_line_l42_42889

theorem find_z_coordinate_of_point_on_line (x1 y1 z1 x2 y2 z2 x_target : ℝ) 
(h1 : x1 = 1) (h2 : y1 = 3) (h3 : z1 = 2) 
(h4 : x2 = 4) (h5 : y2 = 4) (h6 : z2 = -1)
(h_target : x_target = 7) : 
∃ z_target : ℝ, z_target = -4 := 
by {
  sorry
}

end find_z_coordinate_of_point_on_line_l42_42889


namespace quadratic_rational_root_contradiction_l42_42271

def int_coefficients (a b c : ℤ) : Prop := true  -- Placeholder for the condition that coefficients are integers

def is_rational_root (a b c p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ p.gcd q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0  -- p/q is a rational root in simplest form

def ear_even (b c : ℤ) : Prop :=
  b % 2 = 0 ∨ c % 2 = 0

def assume_odd (a b c : ℤ) : Prop :=
  a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem quadratic_rational_root_contradiction (a b c p q : ℤ)
  (h1 : int_coefficients a b c)
  (h2 : a ≠ 0)
  (h3 : is_rational_root a b c p q)
  (h4 : ear_even b c) :
  assume_odd a b c :=
sorry

end quadratic_rational_root_contradiction_l42_42271


namespace smallest_positive_period_intervals_of_monotonicity_max_min_values_l42_42743

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

-- Prove the smallest positive period
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x := sorry

-- Prove the intervals of monotonicity
theorem intervals_of_monotonicity (k : ℤ) : 
  ∀ x y, (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) → 
         (k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ k * Real.pi + Real.pi / 6) → 
         (x < y → f x < f y) ∨ (y < x → f y < f x) := sorry

-- Prove the maximum and minimum values on [0, π/2]
theorem max_min_values : ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧ 
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ max_val ∧ f x ≥ min_val := sorry

end smallest_positive_period_intervals_of_monotonicity_max_min_values_l42_42743


namespace evaluate_expression_l42_42742

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end evaluate_expression_l42_42742


namespace parabola_equation_l42_42294

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the standard equation form of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the right vertex of the hyperbola
def right_vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

-- State the final proof problem
theorem parabola_equation :
  hyperbola 4 0 →
  parabola 8 x y →
  y^2 = 16 * x :=
by
  -- Skip the proof for now
  sorry

end parabola_equation_l42_42294


namespace inequality_abc_l42_42361

theorem inequality_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by {
  sorry
}

end inequality_abc_l42_42361


namespace f_of_2_l42_42973

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end f_of_2_l42_42973


namespace total_selling_price_is_18000_l42_42397

def cost_price_per_meter : ℕ := 50
def loss_per_meter : ℕ := 5
def meters_sold : ℕ := 400

def selling_price_per_meter := cost_price_per_meter - loss_per_meter

def total_selling_price := selling_price_per_meter * meters_sold

theorem total_selling_price_is_18000 :
  total_selling_price = 18000 :=
sorry

end total_selling_price_is_18000_l42_42397


namespace fraction_calculation_l42_42513

theorem fraction_calculation : 
  (1 / 4 + 1 / 6 - 1 / 2) / (-1 / 24) = 2 := 
by 
  sorry

end fraction_calculation_l42_42513


namespace female_with_advanced_degrees_l42_42822

theorem female_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_employees_with_advanced_degrees : ℕ)
  (total_employees_with_college_degree_only : ℕ)
  (total_males_with_college_degree_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : total_employees_with_advanced_degrees = 90)
  (h4 : total_employees_with_college_degree_only = 90)
  (h5 : total_males_with_college_degree_only = 35) :
  ∃ (female_with_advanced_degrees : ℕ), female_with_advanced_degrees = 55 :=
by
  -- the proof goes here
  sorry

end female_with_advanced_degrees_l42_42822


namespace MrsBrownCarrotYield_l42_42124

theorem MrsBrownCarrotYield :
  let pacesLength := 25
  let pacesWidth := 30
  let strideLength := 2.5
  let yieldPerSquareFoot := 0.5
  let lengthInFeet := pacesLength * strideLength
  let widthInFeet := pacesWidth * strideLength
  let area := lengthInFeet * widthInFeet
  let yield := area * yieldPerSquareFoot
  yield = 2343.75 :=
by
  sorry

end MrsBrownCarrotYield_l42_42124


namespace initial_salt_percentage_is_10_l42_42801

-- Declarations for terminology
def initial_volume : ℕ := 72
def added_water : ℕ := 18
def final_volume : ℕ := initial_volume + added_water
def final_salt_percentage : ℝ := 0.08

-- Amount of salt in the initial solution
def initial_salt_amount (P : ℝ) := initial_volume * P

-- Amount of salt in the final solution
def final_salt_amount : ℝ := final_volume * final_salt_percentage

-- Proof that the initial percentage of salt was 10%
theorem initial_salt_percentage_is_10 :
  ∃ P : ℝ, initial_salt_amount P = final_salt_amount ∧ P = 0.1 :=
by
  sorry

end initial_salt_percentage_is_10_l42_42801


namespace value_of_polynomial_at_2_l42_42138

def f (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3 * x^3 - 2 * x^2 - 2500 * x + 434

theorem value_of_polynomial_at_2 : f 2 = -3390 := by
  -- proof would go here
  sorry

end value_of_polynomial_at_2_l42_42138


namespace point_slope_form_of_perpendicular_line_l42_42881

theorem point_slope_form_of_perpendicular_line :
  ∀ (l1 l2 : ℝ → ℝ) (P : ℝ × ℝ),
    (l2 x = x + 1) →
    (P = (2, 1)) →
    (∀ x, l2 x = -1 * l1 x) →
    (∀ x, l1 x = -x + 3) :=
by
  intros l1 l2 P h1 h2 h3
  sorry

end point_slope_form_of_perpendicular_line_l42_42881


namespace quadratic_complete_square_l42_42979

theorem quadratic_complete_square (x : ℝ) (m t : ℝ) :
  (4 * x^2 - 16 * x - 448 = 0) → ((x + m) ^ 2 = t) → (t = 116) :=
by
  sorry

end quadratic_complete_square_l42_42979


namespace range_f_in_interval_l42_42982

-- Define the function f and the interval
def f (x : ℝ) (f_deriv_neg1 : ℝ) := x^3 + 2 * x * f_deriv_neg1
def interval := Set.Icc (-2 : ℝ) (3 : ℝ)

-- State the theorem
theorem range_f_in_interval :
  ∃ (f_deriv_neg1 : ℝ),
  (∀ x ∈ interval, f x f_deriv_neg1 ∈ Set.Icc (-4 * Real.sqrt 2) 9) :=
sorry

end range_f_in_interval_l42_42982


namespace average_score_in_5_matches_l42_42983

theorem average_score_in_5_matches 
  (avg1 avg2 : ℕ)
  (total_matches1 total_matches2 : ℕ)
  (h1 : avg1 = 27) 
  (h2 : avg2 = 32)
  (h3 : total_matches1 = 2) 
  (h4 : total_matches2 = 3) 
  : 
  (avg1 * total_matches1 + avg2 * total_matches2) / (total_matches1 + total_matches2) = 30 :=
by 
  sorry

end average_score_in_5_matches_l42_42983


namespace files_deleted_l42_42876

-- Definitions based on the conditions
def initial_files : ℕ := 93
def files_per_folder : ℕ := 8
def num_folders : ℕ := 9

-- The proof problem
theorem files_deleted : initial_files - (files_per_folder * num_folders) = 21 :=
by
  sorry

end files_deleted_l42_42876


namespace solve_a_plus_b_l42_42906

theorem solve_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : 143 * a + 500 * b = 2001) : a + b = 9 :=
by
  -- Add proof here
  sorry

end solve_a_plus_b_l42_42906


namespace third_smallest_four_digit_in_pascal_l42_42272

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l42_42272


namespace inequality_holds_for_m_l42_42954

theorem inequality_holds_for_m (m : ℝ) :
  (-2 : ℝ) ≤ m ∧ m ≤ (3 : ℝ) ↔ ∀ x : ℝ, x < -1 →
    (m - m^2) * (4 : ℝ)^x + (2 : ℝ)^x + 1 > 0 :=
by sorry

end inequality_holds_for_m_l42_42954


namespace find_r_l42_42535

theorem find_r (a b m p r : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b = 4)
  (h4 : ∀ x : ℚ, x^2 - m * x + 4 = (x - a) * (x - b)) :
  (a - 1 / b) * (b - 1 / a) = 9 / 4 := by
  sorry

end find_r_l42_42535


namespace sofia_running_time_l42_42865

theorem sofia_running_time :
  let distance_first_section := 100 -- meters
  let speed_first_section := 5 -- meters per second
  let distance_second_section := 300 -- meters
  let speed_second_section := 4 -- meters per second
  let num_laps := 6
  let time_first_section := distance_first_section / speed_first_section -- in seconds
  let time_second_section := distance_second_section / speed_second_section -- in seconds
  let time_per_lap := time_first_section + time_second_section -- in seconds
  let total_time_seconds := num_laps * time_per_lap -- in seconds
  let total_time_minutes := total_time_seconds / 60 -- integer division for minutes
  let remaining_seconds := total_time_seconds % 60 -- modulo for remaining seconds
  total_time_minutes = 9 ∧ remaining_seconds = 30 := 
  by
  sorry

end sofia_running_time_l42_42865


namespace geometric_sequence_sum_l42_42621

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h2 : a 3 + a 5 = 6) :
  a 5 + a 7 + a 9 = 28 :=
  sorry

end geometric_sequence_sum_l42_42621


namespace cost_prices_l42_42666

theorem cost_prices (C_t C_c C_b : ℝ)
  (h1 : 2 * C_t = 1000)
  (h2 : 1.75 * C_c = 1750)
  (h3 : 0.75 * C_b = 1500) :
  C_t = 500 ∧ C_c = 1000 ∧ C_b = 2000 :=
by
  sorry

end cost_prices_l42_42666


namespace defective_pens_count_l42_42818

theorem defective_pens_count (total_pens : ℕ) (prob_not_defective : ℚ) (D : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : prob_not_defective = 0.5357142857142857) : 
  D = 2 := 
by
  sorry

end defective_pens_count_l42_42818


namespace perfect_square_polynomial_l42_42555

theorem perfect_square_polynomial (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = m - 10 * x + x^2) → m = 25 :=
sorry

end perfect_square_polynomial_l42_42555


namespace angle_between_sum_is_pi_over_6_l42_42357

open Real EuclideanSpace

noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_u := sqrt (u.1^2 + u.2^2)
  let norm_v := sqrt (v.1^2 + v.2^2)
  arccos (dot_product / (norm_u * norm_v))

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (1/2 * cos (π / 3), 1/2 * sin (π / 3))

theorem angle_between_sum_is_pi_over_6 :
  angle_between_vectors (a.1 + 2 * b.1, a.2 + 2 * b.2) b = π / 6 :=
by
  sorry

end angle_between_sum_is_pi_over_6_l42_42357


namespace total_distance_traveled_l42_42239

theorem total_distance_traveled 
  (Vm : ℝ) (Vr : ℝ) (T_total : ℝ) (D : ℝ) 
  (H_Vm : Vm = 6) 
  (H_Vr : Vr = 1.2) 
  (H_T_total : T_total = 1) 
  (H_time_eq : D / (Vm - Vr) + D / (Vm + Vr) = T_total) 
  : 2 * D = 5.76 := 
by sorry

end total_distance_traveled_l42_42239


namespace two_p_plus_q_l42_42642

variable {p q : ℚ}

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by
  sorry

end two_p_plus_q_l42_42642


namespace purely_periodic_fraction_period_length_divisible_l42_42031

noncomputable def purely_periodic_fraction (p q n : ℕ) : Prop :=
  ∃ (r : ℕ), 10 ^ n - 1 = r * q ∧ (∃ (k : ℕ), q * (10 ^ (n * k)) ∣ p)

theorem purely_periodic_fraction_period_length_divisible
  (p q n : ℕ) (hq : ¬ (2 ∣ q) ∧ ¬ (5 ∣ q)) (hpq : p < q) (hn : 10 ^ n - 1 ∣ q) :
  purely_periodic_fraction p q n :=
by
  sorry

end purely_periodic_fraction_period_length_divisible_l42_42031


namespace twelfth_term_l42_42498

-- Definitions based on the given conditions
def a_3_condition (a d : ℚ) : Prop := a + 2 * d = 10
def a_6_condition (a d : ℚ) : Prop := a + 5 * d = 20

-- The main theorem stating that the twelfth term is 40
theorem twelfth_term (a d : ℚ) (h1 : a_3_condition a d) (h2 : a_6_condition a d) :
  a + 11 * d = 40 :=
sorry

end twelfth_term_l42_42498


namespace solve_for_x_l42_42754

theorem solve_for_x (x : ℝ) 
  (h : 6 * x + 12 * x = 558 - 9 * (x - 4)) : 
  x = 22 := 
sorry

end solve_for_x_l42_42754


namespace minimize_g_function_l42_42952

noncomputable def g (x : ℝ) : ℝ := (9 * x^2 + 18 * x + 29) / (8 * (2 + x))

theorem minimize_g_function : ∀ x : ℝ, x ≥ -1 → g x = 29 / 8 :=
sorry

end minimize_g_function_l42_42952


namespace total_crosswalk_lines_l42_42645

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end total_crosswalk_lines_l42_42645


namespace proof_solution_l42_42550

noncomputable def proof_problem : Prop :=
  ∀ (x y z : ℝ), 3 * x - 4 * y - 2 * z = 0 ∧ x - 2 * y - 8 * z = 0 ∧ z ≠ 0 → 
  (x^2 + 3 * x * y) / (y^2 + z^2) = 329 / 61

theorem proof_solution : proof_problem :=
by
  intros x y z h
  sorry

end proof_solution_l42_42550


namespace quadratic_solution_l42_42921

theorem quadratic_solution (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
sorry

end quadratic_solution_l42_42921


namespace vertex_farthest_from_origin_l42_42526

theorem vertex_farthest_from_origin (center : ℝ × ℝ) (area : ℝ) (top_side_horizontal : Prop) (dilation_center : ℝ × ℝ) (scale_factor : ℝ) :
  center = (10, -5) ∧ area = 16 ∧ top_side_horizontal ∧ dilation_center = (0, 0) ∧ scale_factor = 3 →
  ∃ (vertex_farthest : ℝ × ℝ), vertex_farthest = (36, -21) :=
by
  sorry

end vertex_farthest_from_origin_l42_42526


namespace largest_integer_solution_l42_42168

theorem largest_integer_solution (x : ℤ) : 
  x < (92 / 21 : ℝ) → ∀ y : ℤ, y < (92 / 21 : ℝ) → y ≤ x :=
by
  sorry

end largest_integer_solution_l42_42168


namespace dessert_menu_count_l42_42226

def Dessert : Type := {d : String // d = "cake" ∨ d = "pie" ∨ d = "ice cream" ∨ d = "pudding"}

def valid_menu (menu : Fin 7 → Dessert) : Prop :=
  (menu 0).1 ≠ (menu 1).1 ∧
  menu 1 = ⟨"ice cream", Or.inr (Or.inr (Or.inl rfl))⟩ ∧
  (menu 1).1 ≠ (menu 2).1 ∧
  (menu 2).1 ≠ (menu 3).1 ∧
  (menu 3).1 ≠ (menu 4).1 ∧
  (menu 4).1 ≠ (menu 5).1 ∧
  menu 5 = ⟨"cake", Or.inl rfl⟩ ∧
  (menu 5).1 ≠ (menu 6).1

def total_valid_menus : Nat :=
  4 * 1 * 3 * 3 * 3 * 1 * 3

theorem dessert_menu_count : ∃ (count : Nat), count = 324 ∧ count = total_valid_menus :=
  sorry

end dessert_menu_count_l42_42226


namespace radius_of_tangent_sphere_l42_42602

theorem radius_of_tangent_sphere (r1 r2 : ℝ) (h : r1 = 12 ∧ r2 = 3) :
  ∃ r : ℝ, (r = 6) :=
by
  sorry

end radius_of_tangent_sphere_l42_42602


namespace find_base_b_l42_42944

theorem find_base_b (b : ℕ) (h : (3 * b + 4) ^ 2 = b ^ 3 + 2 * b ^ 2 + 9 * b + 6) : b = 10 :=
sorry

end find_base_b_l42_42944


namespace parabola_vertex_l42_42823

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ y : ℝ, y = 2 * (x - 5)^2 + 3) → (5, 3) = (5, 3) :=
by
  intros x y_eq
  sorry

end parabola_vertex_l42_42823


namespace total_games_in_season_l42_42173

-- Definitions based on the conditions
def num_teams := 16
def teams_per_division := 8
def num_divisions := num_teams / teams_per_division

-- Each team plays every other team in its division twice
def games_within_division_per_team := (teams_per_division - 1) * 2

-- Each team plays every team in the other division once
def games_across_divisions_per_team := teams_per_division

-- Total games per team
def games_per_team := games_within_division_per_team + games_across_divisions_per_team

-- Total preliminary games for all teams (each game is counted twice)
def preliminary_total_games := games_per_team * num_teams

-- Since each game is counted twice, the final number of games
def total_games := preliminary_total_games / 2

theorem total_games_in_season : total_games = 176 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end total_games_in_season_l42_42173


namespace student_age_is_17_in_1960_l42_42054

noncomputable def student's_age_in_1960 (x y : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) : ℕ := 
  let birth_year : ℕ := 1900 + 10 * x + y
  let age_in_1960 : ℕ := 1960 - birth_year
  age_in_1960

theorem student_age_is_17_in_1960 :
  ∃ x y : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ (1960 - (1900 + 10 * x + y) = 1 + 9 + x + y) ∧ (1960 - (1900 + 10 * x + y) = 17) :=
by {
  sorry -- Proof goes here
}

end student_age_is_17_in_1960_l42_42054


namespace geometric_seq_a7_l42_42713

theorem geometric_seq_a7 (a : ℕ → ℝ) (r : ℝ) (h1 : a 3 = 16) (h2 : a 5 = 4) (h_geom : ∀ n, a (n + 1) = a n * r) : a 7 = 1 := by
  sorry

end geometric_seq_a7_l42_42713


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l42_42652

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l42_42652


namespace problem_min_value_l42_42976

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : ℝ :=
  1 / x^2 + 1 / y^2 + 1 / (x * y)

theorem problem_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  min_value x y hx hy hxy = 3 := 
sorry

end problem_min_value_l42_42976


namespace range_of_a_l42_42216

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → (2 * x + 2 * (a - 1)) ≤ 0) → a ≤ -3 :=
by
  sorry

end range_of_a_l42_42216


namespace negation_of_universal_proposition_l42_42421

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)

theorem negation_of_universal_proposition :
  (∀ x1 x2 : R, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : R, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end negation_of_universal_proposition_l42_42421


namespace volleyball_team_selection_l42_42953

theorem volleyball_team_selection (total_players starting_players : ℕ) (libero : ℕ) : 
  total_players = 12 → 
  starting_players = 6 → 
  libero = 1 →
  (∃ (ways : ℕ), ways = 5544) :=
by
  intros h1 h2 h3
  sorry

end volleyball_team_selection_l42_42953


namespace cyclic_sum_inequality_l42_42390

open Real

theorem cyclic_sum_inequality
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / c + b^2 / a + c^2 / b) + (b^2 / c + c^2 / a + a^2 / b) + (c^2 / a + a^2 / b + b^2 / c) + 
  7 * (a + b + c) 
  ≥ ((a + b + c)^3) / (a * b + b * c + c * a) + (2 * (a * b + b * c + c * a)^2) / (a * b * c) := 
sorry

end cyclic_sum_inequality_l42_42390


namespace hemisphere_surface_area_l42_42730

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end hemisphere_surface_area_l42_42730


namespace negation_of_existence_l42_42277

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
sorry

end negation_of_existence_l42_42277


namespace exists_rectangle_in_inscribed_right_triangle_l42_42686

theorem exists_rectangle_in_inscribed_right_triangle :
  ∃ (L W : ℝ), 
    (45^2 / (1 + (5/2)^2) = L * L) ∧
    (2 * L = 45) ∧
    (2 * W = 45) ∧
    ((L = 25 ∧ W = 10) ∨ (L = 18.75 ∧ W = 7.5)) :=
by sorry

end exists_rectangle_in_inscribed_right_triangle_l42_42686


namespace inequality_b_c_a_l42_42661

-- Define the values of a, b, and c
def a := 8^53
def b := 16^41
def c := 64^27

-- State the theorem to prove the inequality b > c > a
theorem inequality_b_c_a : b > c ∧ c > a := by
  sorry

end inequality_b_c_a_l42_42661


namespace domain_of_sqrt_fn_l42_42076

theorem domain_of_sqrt_fn : {x : ℝ | -2 ≤ x ∧ x ≤ 2} = {x : ℝ | 4 - x^2 ≥ 0} := 
by sorry

end domain_of_sqrt_fn_l42_42076


namespace least_n_condition_l42_42522

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end least_n_condition_l42_42522


namespace largest_arithmetic_seq_3digit_l42_42761

theorem largest_arithmetic_seq_3digit : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (∃ a b c : ℕ, n = 100*a + 10*b + c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a = 9 ∧ ∃ d, b = a - d ∧ c = a - 2*d) ∧ n = 963 :=
by sorry

end largest_arithmetic_seq_3digit_l42_42761


namespace total_biscuits_needed_l42_42041

-- Definitions
def number_of_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem statement
theorem total_biscuits_needed : number_of_dogs * biscuits_per_dog = 6 :=
by sorry

end total_biscuits_needed_l42_42041


namespace second_pipe_filling_time_l42_42094

theorem second_pipe_filling_time :
  ∃ T : ℝ, (1/20 + 1/T) * 2/3 * 16 = 1 ∧ T = 160/7 :=
by
  use 160 / 7
  sorry

end second_pipe_filling_time_l42_42094


namespace inscribed_cube_volume_l42_42188

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end inscribed_cube_volume_l42_42188


namespace find_number_l42_42067

theorem find_number (x : ℝ) (h : (x - 8 - 12) / 5 = 7) : x = 55 :=
sorry

end find_number_l42_42067


namespace trigonometric_identity_l42_42672

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : α + β = π / 3)  -- Note: 60 degrees is π/3 radians
  (tan_add : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) 
  (tan_60 : Real.tan (π / 3) = Real.sqrt 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 :=
sorry

end trigonometric_identity_l42_42672


namespace find_initial_terms_l42_42279

theorem find_initial_terms (a : ℕ → ℕ) (h : ∀ n, a (n + 3) = a (n + 2) * (a (n + 1) + 2 * a n))
  (a6 : a 6 = 2288) : a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 2 :=
by
  sorry

end find_initial_terms_l42_42279


namespace sum_base_49_l42_42431

-- Definitions of base b numbers and their base 10 conversion
def num_14_in_base (b : ℕ) : ℕ := b + 4
def num_17_in_base (b : ℕ) : ℕ := b + 7
def num_18_in_base (b : ℕ) : ℕ := b + 8
def num_6274_in_base (b : ℕ) : ℕ := 6 * b^3 + 2 * b^2 + 7 * b + 4

-- The question: Compute 14 + 17 + 18 in base b
def sum_in_base (b : ℕ) : ℕ := 14 + 17 + 18

-- The main statement to prove
theorem sum_base_49 (b : ℕ) (h : (num_14_in_base b) * (num_17_in_base b) * (num_18_in_base b) = num_6274_in_base (b)) :
  sum_in_base b = 49 :=
by sorry

end sum_base_49_l42_42431


namespace probability_of_same_color_l42_42478

-- Defining the given conditions
def green_balls := 6
def red_balls := 4
def total_balls := green_balls + red_balls

def probability_same_color : ℚ :=
  let prob_green := (green_balls / total_balls) * (green_balls / total_balls)
  let prob_red := (red_balls / total_balls) * (red_balls / total_balls)
  prob_green + prob_red

-- Statement of the problem rewritten in Lean 4
theorem probability_of_same_color :
  probability_same_color = 13 / 25 :=
by
  sorry

end probability_of_same_color_l42_42478


namespace pencils_calculation_l42_42156

variable (C B D : ℕ)

theorem pencils_calculation : 
  (C = B + 5) ∧
  (B = 2 * D - 3) ∧
  (C = 20) →
  D = 9 :=
by sorry

end pencils_calculation_l42_42156


namespace last_matching_date_2008_l42_42108

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The last date in 2008 when the sum of the first four digits equals the sum of the last four digits is 25 December 2008. -/
theorem last_matching_date_2008 :
  ∃ d m y, d = 25 ∧ m = 12 ∧ y = 2008 ∧
            sum_of_digits 2512 = sum_of_digits 2008 :=
by {
  sorry
}

end last_matching_date_2008_l42_42108


namespace contrapositive_of_inequality_l42_42608

variable {a b c : ℝ}

theorem contrapositive_of_inequality (h : a + c ≤ b + c) : a ≤ b :=
sorry

end contrapositive_of_inequality_l42_42608


namespace ratio_proof_l42_42366

theorem ratio_proof (x y z s : ℝ) (h1 : x < y) (h2 : y < z)
    (h3 : (x : ℝ) / y = y / z) (h4 : x + y + z = s) (h5 : x + y = z) :
    (x / y = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end ratio_proof_l42_42366


namespace right_to_left_evaluation_l42_42420

variable (a b c d : ℝ)

theorem right_to_left_evaluation :
  a / b - c + d = a / (b - c - d) :=
sorry

end right_to_left_evaluation_l42_42420


namespace sum_of_midpoint_coordinates_l42_42605

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = 16) (h3 : x2 = -2) (h4 : y2 = -8) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 7 := by
  sorry

end sum_of_midpoint_coordinates_l42_42605


namespace evaluate_expression_l42_42737

theorem evaluate_expression (x : Int) (h : x = -2023) : abs (abs (abs x - x) + abs x) + x = 4046 :=
by
  rw [h]
  sorry

end evaluate_expression_l42_42737


namespace integer_values_of_a_l42_42668

-- Define the polynomial P(x)
def P (a x : ℤ) : ℤ := x^3 + a * x^2 + 3 * x + 7

-- Define the main theorem
theorem integer_values_of_a (a x : ℤ) (hx : P a x = 0) (hx_is_int : x = 1 ∨ x = -1 ∨ x = 7 ∨ x = -7) :
  a = -11 ∨ a = -3 :=
by
  sorry

end integer_values_of_a_l42_42668


namespace circumcircle_eqn_l42_42335

def point := ℝ × ℝ

def A : point := (-1, 5)
def B : point := (5, 5)
def C : point := (6, -2)

def circ_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D * x + E * y + F = 0

theorem circumcircle_eqn :
  ∃ D E F : ℝ, (∀ (p : point), p ∈ [A, B, C] → circ_eq D E F p.1 p.2) ∧
              circ_eq (-4) (-2) (-20) = circ_eq D E F := by
  sorry

end circumcircle_eqn_l42_42335


namespace fraction_traditionalists_l42_42937

theorem fraction_traditionalists {P T : ℕ} (h1 : ∀ (i : ℕ), i < 5 → T = P / 15) (h2 : T = P / 15) :
  (5 * T : ℚ) / (P + 5 * T : ℚ) = 1 / 4 :=
by
  sorry

end fraction_traditionalists_l42_42937


namespace quadratic_positive_intervals_l42_42491

-- Problem setup
def quadratic (x : ℝ) : ℝ := x^2 - x - 6

-- Define the roots of the quadratic function
def is_root (a b : ℝ) (f : ℝ → ℝ) := f a = 0 ∧ f b = 0

-- Proving the intervals where the quadratic function is greater than 0
theorem quadratic_positive_intervals :
  is_root (-2) 3 quadratic →
  { x : ℝ | quadratic x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 3 } :=
by
  sorry

end quadratic_positive_intervals_l42_42491


namespace ratio_of_speeds_l42_42317

theorem ratio_of_speeds (v1 v2 : ℝ) (h1 : v1 > v2) (h2 : 8 = (v1 + v2) * 2) (h3 : 8 = (v1 - v2) * 4) : v1 / v2 = 3 :=
by
  sorry

end ratio_of_speeds_l42_42317


namespace johns_grandpa_money_l42_42115

theorem johns_grandpa_money :
  ∃ G : ℝ, (G + 3 * G = 120) ∧ (G = 30) := 
by
  sorry

end johns_grandpa_money_l42_42115


namespace golden_section_AC_length_l42_42996

namespace GoldenSection

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def AC_length (AB : ℝ) : ℝ :=
  let φ := golden_ratio
  AB / φ

theorem golden_section_AC_length (AB : ℝ) (C_gold : Prop) (hAB : AB = 2) (A_gt_B : AC_length AB > AB - AC_length AB) :
  AC_length AB = Real.sqrt 5 - 1 :=
  sorry

end GoldenSection

end golden_section_AC_length_l42_42996


namespace surface_area_of_modified_structure_l42_42572

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

end surface_area_of_modified_structure_l42_42572


namespace five_digit_numbers_l42_42462

def divisible_by_4_and_9 (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 9 = 0)

def is_candidate (n : ℕ) : Prop :=
  ∃ a b, n = 10000 * a + 1000 + 200 + 30 + b ∧ a < 10 ∧ b < 10

theorem five_digit_numbers :
  ∀ (n : ℕ), is_candidate n → divisible_by_4_and_9 n → n = 11232 ∨ n = 61236 :=
by
  sorry

end five_digit_numbers_l42_42462


namespace minimum_distinct_numbers_l42_42560

theorem minimum_distinct_numbers (a : ℕ → ℕ) (h_pos : ∀ i, 1 ≤ i → a i > 0)
  (h_distinct_ratios : ∀ i j : ℕ, 1 ≤ i ∧ i < 2006 ∧ 1 ≤ j ∧ j < 2006 ∧ i ≠ j → a i / a (i + 1) ≠ a j / a (j + 1)) :
  ∃ (n : ℕ), n = 46 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2006 ∧ 1 ≤ j ∧ j ≤ i ∧ (a i = a j → i = j) :=
sorry

end minimum_distinct_numbers_l42_42560


namespace evaluate_expression_l42_42489

theorem evaluate_expression : 3 + 2 * (8 - 3) = 13 := by
  sorry

end evaluate_expression_l42_42489


namespace sum_of_fractions_eq_one_l42_42191

variable {a b c d : ℝ} (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)
          (h_equiv : (a * d + b * c) / (b * d) = (a * c) / (b * d))

theorem sum_of_fractions_eq_one : b / a + d / c = 1 :=
by sorry

end sum_of_fractions_eq_one_l42_42191


namespace express_y_in_terms_of_x_l42_42184

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end express_y_in_terms_of_x_l42_42184


namespace expression_equality_l42_42174

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end expression_equality_l42_42174


namespace larry_spent_on_lunch_l42_42908

noncomputable def starting_amount : ℕ := 22
noncomputable def ending_amount : ℕ := 15
noncomputable def amount_given_to_brother : ℕ := 2

theorem larry_spent_on_lunch : 
  (starting_amount - (ending_amount + amount_given_to_brother)) = 5 :=
by
  -- The conditions and the proof structure would be elaborated here
  sorry

end larry_spent_on_lunch_l42_42908


namespace number_of_hens_l42_42295

theorem number_of_hens (H C : ℕ) 
  (h1 : H + C = 60) 
  (h2 : 2 * H + 4 * C = 200) : H = 20 :=
sorry

end number_of_hens_l42_42295


namespace prime_quadratic_root_range_l42_42718

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end prime_quadratic_root_range_l42_42718


namespace verify_statements_l42_42938

noncomputable def f (x : ℝ) : ℝ := 10 ^ x

theorem verify_statements (x1 x2 : ℝ) (h : x1 ≠ x2) :
  (f (x1 + x2) = f x1 * f x2) ∧
  (f x1 - f x2) / (x1 - x2) > 0 :=
by
  sorry

end verify_statements_l42_42938


namespace range_of_m_l42_42318

variable (x y m : ℝ)

theorem range_of_m (h1 : Real.sin x = m * (Real.sin y)^3)
                   (h2 : Real.cos x = m * (Real.cos y)^3) :
                   1 ≤ m ∧ m ≤ Real.sqrt 2 :=
by
  sorry

end range_of_m_l42_42318


namespace average_of_three_l42_42575

-- Definitions of Conditions
variables (A B C : ℝ)
variables (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132)

-- The proof problem stating the goal
theorem average_of_three (A B C : ℝ) 
    (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132) : 
    (A + B + C) / 3 = 67 := 
sorry

end average_of_three_l42_42575


namespace value_of_three_inch_cube_l42_42079

theorem value_of_three_inch_cube (value_two_inch: ℝ) (volume_two_inch: ℝ) (volume_three_inch: ℝ) (cost_two_inch: ℝ):
  value_two_inch = cost_two_inch * ((volume_three_inch / volume_two_inch): ℝ) := 
by
  have volume_two_inch := 2^3 -- Volume of two-inch cube
  have volume_three_inch := 3^3 -- Volume of three-inch cube
  let volume_ratio := (volume_three_inch / volume_two_inch: ℝ)
  have := cost_two_inch * volume_ratio
  norm_num
  sorry

end value_of_three_inch_cube_l42_42079


namespace smallest_solution_fraction_eq_l42_42415

theorem smallest_solution_fraction_eq (x : ℝ) (h : x ≠ 3) :
    3 * x / (x - 3) + (3 * x^2 - 27) / x = 16 ↔ x = (2 - Real.sqrt 31) / 3 := 
sorry

end smallest_solution_fraction_eq_l42_42415


namespace octagon_perimeter_correct_l42_42182

def octagon_perimeter (n : ℕ) (side_length : ℝ) : ℝ :=
  n * side_length

theorem octagon_perimeter_correct :
  octagon_perimeter 8 3 = 24 :=
by
  sorry

end octagon_perimeter_correct_l42_42182


namespace game_no_loser_l42_42773

theorem game_no_loser (x : ℕ) (h_start : x = 2017) :
  ∀ y, (y = x ∨ ∀ n, (n = 2 * y ∨ n = y - 1000) → (n > 1000 ∧ n < 4000)) →
       (y > 1000 ∧ y < 4000) :=
sorry

end game_no_loser_l42_42773


namespace value_of_b_l42_42561

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 :=
sorry

end value_of_b_l42_42561


namespace boys_and_girls_equal_l42_42263

theorem boys_and_girls_equal (m d M D : ℕ) (hm : m > 0) (hd : d > 0) (h1 : (M / m) ≠ (D / d)) (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d := 
sorry

end boys_and_girls_equal_l42_42263


namespace base5_product_is_correct_l42_42728

-- Definitions for the problem context
def base5_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 528 then [4, 1, 0, 0, 3] else []

-- Theorem to prove the base-5 multiplication result
theorem base5_product_is_correct :
  base10_to_base5 (base5_to_base10 2 3 1 * base5_to_base10 1 3 0) = [4, 1, 0, 0, 3] :=
by
  sorry

end base5_product_is_correct_l42_42728


namespace hallway_width_equals_four_l42_42888

-- Define the conditions: dimensions of the areas and total installed area.
def centralAreaLength : ℝ := 10
def centralAreaWidth : ℝ := 10
def centralArea : ℝ := centralAreaLength * centralAreaWidth

def totalInstalledArea : ℝ := 124
def hallwayLength : ℝ := 6

-- Total area minus central area's area yields hallway's area
def hallwayArea : ℝ := totalInstalledArea - centralArea

-- Statement to prove: the width of the hallway given its area and length.
theorem hallway_width_equals_four :
  (hallwayArea / hallwayLength) = 4 := 
by
  sorry

end hallway_width_equals_four_l42_42888


namespace magician_earnings_at_least_l42_42736

def magician_starting_decks := 15
def magician_remaining_decks := 3
def decks_sold := magician_starting_decks - magician_remaining_decks
def standard_price_per_deck := 3
def discount := 1
def discounted_price_per_deck := standard_price_per_deck - discount
def min_earnings := decks_sold * discounted_price_per_deck

theorem magician_earnings_at_least :
  min_earnings ≥ 24 :=
by sorry

end magician_earnings_at_least_l42_42736


namespace min_value_f_l42_42158

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1)^2 + (Real.exp (-x) - 1)^2

theorem min_value_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 :=
sorry

end min_value_f_l42_42158


namespace reassemble_black_rectangles_into_1x2_rectangle_l42_42542

theorem reassemble_black_rectangles_into_1x2_rectangle
  (x y : ℝ)
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < y ∧ y < 2)
  (black_white_equal : 2*x*y - 2*x - 2*y + 2 = 0) :
  (x = 1 ∨ y = 1) →
  ∃ (z : ℝ), z = 1 :=
by
  sorry

end reassemble_black_rectangles_into_1x2_rectangle_l42_42542


namespace polygon_sides_sum_720_l42_42274

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l42_42274


namespace simultaneous_equations_solution_exists_l42_42243

theorem simultaneous_equations_solution_exists (m : ℝ) :
  ∃ x y : ℝ, y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 :=
by
  sorry

end simultaneous_equations_solution_exists_l42_42243


namespace quadratic_nonnegative_quadratic_inv_nonnegative_l42_42760

-- Problem Definitions and Proof Statements

variables {R : Type*} [LinearOrderedField R]

def f (a b c x : R) : R := a * x^2 + 2 * b * x + c

theorem quadratic_nonnegative {a b c : R} (ha : a ≠ 0) (h : ∀ x : R, f a b c x ≥ 0) : 
  a ≥ 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 :=
sorry

theorem quadratic_inv_nonnegative {a b c : R} (ha : a ≥ 0) (hc : c ≥ 0) (hac : a * c - b^2 ≥ 0) :
  ∀ x : R, f a b c x ≥ 0 :=
sorry

end quadratic_nonnegative_quadratic_inv_nonnegative_l42_42760


namespace find_angle_CBO_l42_42687

theorem find_angle_CBO :
  ∀ (BAO CAO CBO ABO ACO BCO AOC : ℝ), 
  BAO = CAO → 
  CBO = ABO → 
  ACO = BCO → 
  AOC = 110 →
  CBO = 20 :=
by
  intros BAO CAO CBO ABO ACO BCO AOC hBAO_CAOC hCBO_ABO hACO_BCO hAOC
  sorry

end find_angle_CBO_l42_42687


namespace cost_of_shirt_l42_42449

theorem cost_of_shirt (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) : S = 15 :=
by
  sorry

end cost_of_shirt_l42_42449


namespace avg_calculation_l42_42960

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem avg_calculation :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end avg_calculation_l42_42960


namespace sum_of_largest_and_smallest_four_digit_numbers_is_11990_l42_42720

theorem sum_of_largest_and_smallest_four_digit_numbers_is_11990 (A B C D : ℕ) 
    (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D)
    (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
    (h_eq : 1001 * A + 110 * B + 110 * C + 1001 * D = 11990) :
    (min (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 1999) ∧
    (max (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 9991) :=
by
  sorry

end sum_of_largest_and_smallest_four_digit_numbers_is_11990_l42_42720


namespace validate_operation_l42_42770

theorem validate_operation (x y m a b : ℕ) :
  (2 * x - x ≠ 2) →
  (2 * m + 3 * m ≠ 5 * m^2) →
  (5 * xy - 4 * xy = xy) →
  (2 * a + 3 * b ≠ 5 * a * b) →
  (5 * xy - 4 * xy = xy) :=
by
  intros hA hB hC hD
  exact hC

end validate_operation_l42_42770


namespace rattlesnakes_count_l42_42026

theorem rattlesnakes_count (total_snakes : ℕ) (boa_constrictors pythons rattlesnakes : ℕ)
  (h1 : total_snakes = 200)
  (h2 : boa_constrictors = 40)
  (h3 : pythons = 3 * boa_constrictors)
  (h4 : total_snakes = boa_constrictors + pythons + rattlesnakes) :
  rattlesnakes = 40 :=
by
  sorry

end rattlesnakes_count_l42_42026


namespace total_number_of_students_l42_42943

theorem total_number_of_students (T G : ℕ) (h1 : 50 + G = T) (h2 : G = 50 * T / 100) : T = 100 :=
  sorry

end total_number_of_students_l42_42943


namespace competition_arrangements_l42_42772

noncomputable def count_arrangements (students : Fin 4) (events : Fin 3) : Nat :=
  -- The actual counting function is not implemented
  sorry

theorem competition_arrangements (students : Fin 4) (events : Fin 3) :
  let count := count_arrangements students events
  (∃ (A B C D : Fin 4), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧ 
    (A ≠ 0) ∧ 
    count = 24) := sorry

end competition_arrangements_l42_42772


namespace min_value_a_l42_42046

theorem min_value_a (a : ℕ) :
  (6 * (a + 1)) / (a^2 + 8 * a + 6) ≤ 1 / 100 ↔ a ≥ 594 := sorry

end min_value_a_l42_42046


namespace time_difference_l42_42766

-- Definitions for the problem conditions
def Zoe_speed : ℕ := 9 -- Zoe's speed in minutes per mile
def Henry_speed : ℕ := 7 -- Henry's speed in minutes per mile
def Race_length : ℕ := 12 -- Race length in miles

-- Theorem to prove the time difference
theorem time_difference : (Race_length * Zoe_speed) - (Race_length * Henry_speed) = 24 :=
by
  sorry

end time_difference_l42_42766


namespace clock_hands_overlap_l42_42015

theorem clock_hands_overlap:
  ∃ x y: ℚ,
  -- Conditions
  (60 * 10 + x = 60 * 11 * 54 + 6 / 11) ∧
  (y - (5 / 60) * y = 60) ∧
  (65 * 5 / 11 = y) := sorry

end clock_hands_overlap_l42_42015


namespace painted_cube_probability_l42_42155

-- Define the conditions
def cube_size : Nat := 5
def total_unit_cubes : Nat := cube_size ^ 3
def corner_cubes_with_three_faces : Nat := 1
def edges_with_two_faces : Nat := 3 * (cube_size - 2) -- 3 edges, each (5 - 2) = 3
def faces_with_one_face : Nat := 2 * (cube_size * cube_size - corner_cubes_with_three_faces - edges_with_two_faces)
def no_painted_faces_cubes : Nat := total_unit_cubes - corner_cubes_with_three_faces - faces_with_one_face

-- Compute the probability
def probability := (corner_cubes_with_three_faces * no_painted_faces_cubes) / (total_unit_cubes * (total_unit_cubes - 1) / 2)

-- The theorem statement
theorem painted_cube_probability :
  probability = (2 : ℚ) / 155 := 
by {
  sorry
}

end painted_cube_probability_l42_42155


namespace total_rainfall_over_3_days_l42_42304

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end total_rainfall_over_3_days_l42_42304


namespace remainder_3001_3005_mod_23_l42_42967

theorem remainder_3001_3005_mod_23 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 :=
by {
  sorry
}

end remainder_3001_3005_mod_23_l42_42967


namespace find_speeds_l42_42439

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l42_42439


namespace lcm_of_two_numbers_l42_42927

-- Define the given conditions: Two numbers a and b, their HCF, and their product.
variables (a b : ℕ)
def hcf : ℕ := 55
def product := 82500

-- Define the concept of HCF and LCM, using the provided relationship in the problem
def gcd_ab := hcf
def lcm_ab := (product / gcd_ab)

-- State the main theorem to prove: The LCM of the two numbers is 1500
theorem lcm_of_two_numbers : lcm_ab = 1500 := by
  -- This is the place where the actual proof steps would go
  sorry

end lcm_of_two_numbers_l42_42927


namespace cost_of_camel_l42_42444

variables (C H O E G Z L : ℕ)

theorem cost_of_camel :
  (10 * C = 24 * H) →
  (16 * H = 4 * O) →
  (6 * O = 4 * E) →
  (3 * E = 5 * G) →
  (8 * G = 12 * Z) →
  (20 * Z = 7 * L) →
  (10 * E = 120000) →
  C = 4800 :=
by
  sorry

end cost_of_camel_l42_42444


namespace baseball_game_earnings_l42_42744

theorem baseball_game_earnings
  (S : ℝ) (W : ℝ)
  (h1 : S = 2662.50)
  (h2 : W + S = 5182.50) :
  S - W = 142.50 :=
by
  sorry

end baseball_game_earnings_l42_42744


namespace prob_both_even_correct_l42_42426

-- Define the dice and verify their properties
def die1 := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def die2 := {n : ℕ // n ≥ 1 ∧ n ≤ 7}

-- Define the sets of even numbers for both dice
def even_die1 (n : die1) : Prop := n.1 % 2 = 0
def even_die2 (n : die2) : Prop := n.1 % 2 = 0

-- Define the probabilities of rolling an even number on each die
def prob_even_die1 := 3 / 6
def prob_even_die2 := 3 / 7

-- Calculate the combined probability
def prob_both_even := prob_even_die1 * prob_even_die2

-- The theorem stating the probability of both dice rolling even is 3/14
theorem prob_both_even_correct : prob_both_even = 3 / 14 :=
by
  -- Proof is omitted
  sorry

end prob_both_even_correct_l42_42426


namespace students_helped_on_fourth_day_l42_42639

theorem students_helped_on_fourth_day (total_books : ℕ) (books_per_student : ℕ)
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ)
  (H1 : total_books = 120) (H2 : books_per_student = 5)
  (H3 : day1_students = 4) (H4 : day2_students = 5) (H5 : day3_students = 6) :
  (total_books - (day1_students * books_per_student + day2_students * books_per_student + day3_students * books_per_student)) / books_per_student = 9 :=
by
  sorry

end students_helped_on_fourth_day_l42_42639


namespace distance_between_cities_l42_42487

variable (D : ℝ) -- D is the distance between City A and City B
variable (time_AB : ℝ) -- Time from City A to City B
variable (time_BA : ℝ) -- Time from City B to City A
variable (saved_time : ℝ) -- Time saved per trip
variable (avg_speed : ℝ) -- Average speed for the round trip with saved time

theorem distance_between_cities :
  time_AB = 6 → time_BA = 4.5 → saved_time = 0.5 → avg_speed = 90 →
  D = 427.5 :=
by
  sorry

end distance_between_cities_l42_42487


namespace students_absent_percentage_l42_42105

theorem students_absent_percentage (total_students present_students : ℕ) (h_total : total_students = 50) (h_present : present_students = 45) :
  (total_students - present_students) * 100 / total_students = 10 := 
by
  sorry

end students_absent_percentage_l42_42105


namespace pencils_left_l42_42078

theorem pencils_left (anna_pencils : ℕ) (harry_pencils : ℕ)
  (h_anna : anna_pencils = 50) (h_harry : harry_pencils = 2 * anna_pencils)
  (lost_pencils : ℕ) (h_lost : lost_pencils = 19) :
  harry_pencils - lost_pencils = 81 :=
by
  sorry

end pencils_left_l42_42078


namespace cider_apples_production_l42_42454

def apples_total : Real := 8.0
def baking_fraction : Real := 0.30
def cider_fraction : Real := 0.60

def apples_remaining : Real := apples_total * (1 - baking_fraction)
def apples_for_cider : Real := apples_remaining * cider_fraction

theorem cider_apples_production : 
    apples_for_cider = 3.4 := 
by
  sorry

end cider_apples_production_l42_42454


namespace tracy_initial_candies_l42_42567

theorem tracy_initial_candies 
  (x : ℕ)
  (h1 : 4 ∣ x)
  (h2 : 5 ≤ ((x / 2) - 24))
  (h3 : ((x / 2) - 24) ≤ 9) 
  : x = 68 :=
sorry

end tracy_initial_candies_l42_42567


namespace inequality_solution_l42_42097

theorem inequality_solution (x : ℝ) : (-3 * x^2 - 9 * x - 6 ≥ -12) ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

end inequality_solution_l42_42097


namespace carolyn_initial_marbles_l42_42083

theorem carolyn_initial_marbles (x : ℕ) (h1 : x - 42 = 5) : x = 47 :=
by
  sorry

end carolyn_initial_marbles_l42_42083


namespace find_xyz_l42_42588

theorem find_xyz (x y z : ℝ) (h1 : x * (y + z) = 195) (h2 : y * (z + x) = 204) (h3 : z * (x + y) = 213) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * y * z = 1029 := by
  sorry

end find_xyz_l42_42588


namespace value_of_t_for_x_equals_y_l42_42507

theorem value_of_t_for_x_equals_y (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : 
    t = 1 / 2 → x = y :=
by 
  intro ht
  rw [ht] at h1 h2
  sorry

end value_of_t_for_x_equals_y_l42_42507


namespace square_triangle_ratios_l42_42467

theorem square_triangle_ratios (s t : ℝ) 
  (P_s := 4 * s) 
  (R_s := s * Real.sqrt 2 / 2)
  (P_t := 3 * t) 
  (R_t := t * Real.sqrt 3 / 3) 
  (h : s = t) : 
  (P_s / P_t = 4 / 3) ∧ (R_s / R_t = Real.sqrt 6 / 2) := 
by
  sorry

end square_triangle_ratios_l42_42467


namespace remainder_of_3n_mod_9_l42_42523

theorem remainder_of_3n_mod_9 (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 :=
by
  sorry

end remainder_of_3n_mod_9_l42_42523


namespace quadratic_common_root_l42_42235

theorem quadratic_common_root (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0)
  (h2 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + b * x + c = 0)
  (h3 : ∃ x, x^2 + b * x + c = 0 ∧ x^2 + c * x + a = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratic_common_root_l42_42235


namespace problem_product_xyzw_l42_42204

theorem problem_product_xyzw
    (x y z w : ℝ)
    (h1 : x + 1 / y = 1)
    (h2 : y + 1 / z + w = 1)
    (h3 : w = 2) :
    xyzw = -2 * y^2 + 2 * y :=
by
    sorry

end problem_product_xyzw_l42_42204


namespace randy_trip_length_l42_42443

theorem randy_trip_length (x : ℝ) (h : x / 2 + 30 + x / 4 = x) : x = 120 :=
by
  sorry

end randy_trip_length_l42_42443


namespace max_sum_arith_seq_l42_42132

theorem max_sum_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = 8 + (n - 1) * d) →
  d ≠ 0 →
  a 1 = 8 →
  a 5 ^ 2 = a 1 * a 7 →
  S n = n * a 1 + (n * (n - 1) * d) / 2 →
  ∃ n : ℕ, S n = 36 :=
by
  intros
  sorry

end max_sum_arith_seq_l42_42132


namespace total_copies_produced_l42_42802

theorem total_copies_produced
  (rate_A : ℕ)
  (rate_B : ℕ)
  (rate_C : ℕ)
  (time_A : ℕ)
  (time_B : ℕ)
  (time_C : ℕ)
  (total_time : ℕ)
  (ha : rate_A = 10)
  (hb : rate_B = 10)
  (hc : rate_C = 10)
  (hA_time : time_A = 15)
  (hB_time : time_B = 20)
  (hC_time : time_C = 25)
  (h_total_time : total_time = 30) :
  rate_A * time_A + rate_B * time_B + rate_C * time_C = 600 :=
by 
  -- Machine A: 10 copies per minute * 15 minutes = 150 copies
  -- Machine B: 10 copies per minute * 20 minutes = 200 copies
  -- Machine C: 10 copies per minute * 25 minutes = 250 copies
  -- Hence, the total number of copies = 150 + 200 + 250 = 600
  sorry

end total_copies_produced_l42_42802


namespace euler_polyhedron_problem_l42_42399

theorem euler_polyhedron_problem : 
  ( ∀ (V E F T S : ℕ), F = 42 → (T = 2 ∧ S = 3) → V - E + F = 2 → 100 * S + 10 * T + V = 337 ) := 
by sorry

end euler_polyhedron_problem_l42_42399


namespace geom_seq_decreasing_l42_42060

theorem geom_seq_decreasing :
  (∀ n : ℕ, (4 : ℝ) * 3^(1 - (n + 1) : ℤ) < (4 : ℝ) * 3^(1 - n : ℤ)) :=
sorry

end geom_seq_decreasing_l42_42060


namespace solutionToEquations_solutionToInequalities_l42_42824

-- Part 1: Solve the system of equations
def solveEquations (x y : ℝ) : Prop :=
2 * x - y = 3 ∧ x + y = 6

theorem solutionToEquations (x y : ℝ) (h : solveEquations x y) : 
x = 3 ∧ y = 3 :=
sorry

-- Part 2: Solve the system of inequalities
def solveInequalities (x : ℝ) : Prop :=
3 * x > x - 4 ∧ (4 + x) / 3 > x + 2

theorem solutionToInequalities (x : ℝ) (h : solveInequalities x) : 
-2 < x ∧ x < -1 :=
sorry

end solutionToEquations_solutionToInequalities_l42_42824


namespace find_overlap_length_l42_42945

-- Definitions of the given conditions
def total_length_of_segments := 98 -- cm
def edge_to_edge_distance := 83 -- cm
def number_of_overlaps := 6

-- Theorem stating the value of x in centimeters
theorem find_overlap_length (x : ℝ) 
  (h1 : total_length_of_segments = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : number_of_overlaps = 6) 
  (h4 : total_length_of_segments = edge_to_edge_distance + number_of_overlaps * x) : 
  x = 2.5 :=
  sorry

end find_overlap_length_l42_42945


namespace average_speed_is_65_l42_42217

-- Definitions based on the problem's conditions
def speed_first_hour : ℝ := 100 -- 100 km in the first hour
def speed_second_hour : ℝ := 30 -- 30 km in the second hour
def total_distance : ℝ := speed_first_hour + speed_second_hour -- total distance
def total_time : ℝ := 2 -- total time in hours (1 hour + 1 hour)

-- Problem: prove that the average speed is 65 km/h
theorem average_speed_is_65 : (total_distance / total_time) = 65 := by
  sorry

end average_speed_is_65_l42_42217


namespace solve_equation_l42_42328

def f (x : ℝ) := |3 * x - 2|

theorem solve_equation 
  (x : ℝ) 
  (a : ℝ)
  (hx1 : x ≠ 3)
  (hx2 : x ≠ 0) :
  (3 * x - 2) ^ 2 = (x + a) ^ 2 ↔
  (a = -4 * x + 2) ∨ (a = 2 * x - 2) := by
  sorry

end solve_equation_l42_42328


namespace problem_ineq_l42_42165

theorem problem_ineq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := 
by 
  sorry

end problem_ineq_l42_42165


namespace b_plus_c_eq_neg3_l42_42135

theorem b_plus_c_eq_neg3 (b c : ℝ)
  (h1 : ∀ x : ℝ, x^2 + b * x + c > 0 ↔ (x < -1 ∨ x > 2)) :
  b + c = -3 :=
sorry

end b_plus_c_eq_neg3_l42_42135


namespace lucy_total_cost_for_lamp_and_table_l42_42423

noncomputable def original_price_lamp : ℝ := 200 / 1.2

noncomputable def table_price : ℝ := 2 * original_price_lamp

noncomputable def total_cost_paid (lamp_cost discounted_price table_price: ℝ) :=
  lamp_cost + table_price

theorem lucy_total_cost_for_lamp_and_table :
  total_cost_paid 20 (original_price_lamp * 0.6) table_price = 353.34 :=
by
  let lamp_original_price := original_price_lamp
  have h1 : original_price_lamp * (0.6 * (1 / 5)) = 20 := by sorry
  have h2 : table_price = 2 * original_price_lamp := by sorry
  have h3 : total_cost_paid 20 (original_price_lamp * 0.6) table_price = 20 + table_price := by sorry
  have h4 : table_price = 2 * (200 / 1.2) := by sorry
  have h5 : 20 + table_price = 353.34 := by sorry
  exact h5

end lucy_total_cost_for_lamp_and_table_l42_42423


namespace originally_anticipated_profit_margin_l42_42629

theorem originally_anticipated_profit_margin (decrease_percent increase_percent : ℝ) (original_price current_price : ℝ) (selling_price : ℝ) :
  decrease_percent = 6.4 → 
  increase_percent = 8 → 
  original_price = 1 → 
  current_price = original_price - original_price * decrease_percent / 100 → 
  selling_price = original_price * (1 + x / 100) → 
  selling_price = current_price * (1 + (x + increase_percent) / 100) →
  x = 117 :=
by
  intros h_dec_perc h_inc_perc h_org_price h_cur_price h_selling_price_orig h_selling_price_cur
  sorry

end originally_anticipated_profit_margin_l42_42629


namespace partition_diff_l42_42933

theorem partition_diff {A : Type} (S : Finset ℕ) (S_card : S.card = 67)
  (P : Finset (Finset ℕ)) (P_card : P.card = 4) :
  ∃ (U : Finset ℕ) (hU : U ∈ P), ∃ (a b c : ℕ) (ha : a ∈ U) (hb : b ∈ U) (hc : c ∈ U),
  a = b - c ∧ (1 ≤ a ∧ a ≤ 67) :=
by sorry

end partition_diff_l42_42933


namespace find_a_if_perpendicular_l42_42106

def m (a : ℝ) : ℝ × ℝ := (3, a - 1)
def n (a : ℝ) : ℝ × ℝ := (a, -2)

theorem find_a_if_perpendicular (a : ℝ) (h : (m a).fst * (n a).fst + (m a).snd * (n a).snd = 0) : a = -2 :=
by sorry

end find_a_if_perpendicular_l42_42106


namespace max_bag_weight_l42_42757

-- Let's define the conditions first
def green_beans_weight := 4
def milk_weight := 6
def carrots_weight := 2 * green_beans_weight
def additional_capacity := 2

-- The total weight of groceries
def total_groceries_weight := green_beans_weight + milk_weight + carrots_weight

-- The maximum weight the bag can hold is the total weight of groceries plus the additional capacity
theorem max_bag_weight : (total_groceries_weight + additional_capacity) = 20 := by
  sorry

end max_bag_weight_l42_42757


namespace sphere_volume_l42_42457

theorem sphere_volume (C : ℝ) (h : C = 30) : 
  ∃ (V : ℝ), V = 4500 / (π^2) :=
by sorry

end sphere_volume_l42_42457


namespace find_functional_f_l42_42786

-- Define the problem domain and functions
variable (f : ℕ → ℕ)
variable (ℕ_star : Set ℕ) -- ℕ_star is {1,2,3,...}

-- Conditions
axiom f_increasing (h1 : ℕ) (h2 : ℕ) (h1_lt_h2 : h1 < h2) : f h1 < f h2
axiom f_functional (x : ℕ) (y : ℕ) : f (y * f x) = x^2 * f (x * y)

-- The proof problem
theorem find_functional_f : (∀ x ∈ ℕ_star, f x = x^2) :=
sorry

end find_functional_f_l42_42786


namespace cone_cylinder_volume_ratio_l42_42198

theorem cone_cylinder_volume_ratio (h r : ℝ) (hc_pos : h > 0) (r_pos : r > 0) :
  let V_cylinder := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * (3 / 4 * h)
  (V_cone / V_cylinder) = 1 / 4 := 
by 
  sorry

end cone_cylinder_volume_ratio_l42_42198


namespace maximum_sum_of_composites_l42_42863

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def pairwise_coprime (A B C : ℕ) : Prop :=
  Nat.gcd A B = 1 ∧ Nat.gcd A C = 1 ∧ Nat.gcd B C = 1

theorem maximum_sum_of_composites (A B C : ℕ)
  (hA : is_composite A) (hB : is_composite B) (hC : is_composite C)
  (h_pairwise : pairwise_coprime A B C)
  (h_prod_eq : A * B * C = 11011 * 28) :
  A + B + C = 1626 := 
sorry

end maximum_sum_of_composites_l42_42863


namespace min_formula_l42_42521

theorem min_formula (a b : ℝ) : 
  min a b = (a + b - Real.sqrt ((a - b) ^ 2)) / 2 :=
by
  sorry

end min_formula_l42_42521


namespace solution_to_system_l42_42333

theorem solution_to_system (x y z : ℝ) (h1 : x^2 + y^2 = 6 * z) (h2 : y^2 + z^2 = 6 * x) (h3 : z^2 + x^2 = 6 * y) :
  (x = 3) ∧ (y = 3) ∧ (z = 3) :=
sorry

end solution_to_system_l42_42333


namespace units_digit_8th_group_l42_42884

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_8th_group (t k : ℕ) (ht : t = 7) (hk : k = 8) : 
  units_digit (t + k) = 5 := 
by
  -- Proof step will go here.
  sorry

end units_digit_8th_group_l42_42884


namespace rachel_removed_bottle_caps_l42_42259

def original_bottle_caps : ℕ := 87
def remaining_bottle_caps : ℕ := 40

theorem rachel_removed_bottle_caps :
  original_bottle_caps - remaining_bottle_caps = 47 := by
  sorry

end rachel_removed_bottle_caps_l42_42259


namespace value_of_s_l42_42153

-- Conditions: (m - 8) is a factor of m^2 - sm - 24

theorem value_of_s (s : ℤ) (m : ℤ) (h : (m - 8) ∣ (m^2 - s*m - 24)) : s = 5 :=
by
  sorry

end value_of_s_l42_42153


namespace sqrt_sum_simplification_l42_42618

theorem sqrt_sum_simplification :
  (Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by
    sorry

end sqrt_sum_simplification_l42_42618


namespace min_PA_squared_plus_PB_squared_l42_42377

-- Let points A, B, and the circle be defined as given in the problem.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

def PA_squared (P : Point) : ℝ :=
  (P.x - A.x)^2 + (P.y - A.y)^2

def PB_squared (P : Point) : ℝ :=
  (P.x - B.x)^2 + (P.y - B.y)^2

def F (P : Point) : ℝ := PA_squared P + PB_squared P

theorem min_PA_squared_plus_PB_squared : ∃ P : Point, on_circle P ∧ F P = 26 := sorry

end min_PA_squared_plus_PB_squared_l42_42377


namespace sum_of_roots_l42_42599

theorem sum_of_roots (x : ℝ) (h : x + 49 / x = 14) : x + x = 14 :=
sorry

end sum_of_roots_l42_42599


namespace total_number_of_students_l42_42671

theorem total_number_of_students (b h p s : ℕ) 
  (h1 : b = 30)
  (h2 : b = 2 * h)
  (h3 : p = h + 5)
  (h4 : s = 3 * p) :
  b + h + p + s = 125 :=
by sorry

end total_number_of_students_l42_42671


namespace largest_divisor_of_n_squared_divisible_by_72_l42_42201

theorem largest_divisor_of_n_squared_divisible_by_72
    (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : 12 ∣ n :=
by {
    sorry
}

end largest_divisor_of_n_squared_divisible_by_72_l42_42201


namespace seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l42_42141

-- Problem 1
theorem seq_inv_an_is_arithmetic (a : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) :
  ∃ d, ∀ n, n ≥ 2 → (1 / a n) = 2 + (n - 1) * d :=
sorry

-- Problem 2
theorem seq_fn_over_an_has_minimum (a f : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) (h3 : ∀ n, f n = (9 / 10) ^ n) :
  ∃ m, ∀ n, n ≠ m → f n / a n ≥ f m / a m :=
sorry

end seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l42_42141


namespace number_of_true_propositions_l42_42493

noncomputable def proposition1 : Prop := ∀ (x : ℝ), x^2 - 3 * x + 2 > 0
noncomputable def proposition2 : Prop := ∃ (x : ℚ), x^2 = 2
noncomputable def proposition3 : Prop := ∃ (x : ℝ), x^2 - 1 = 0
noncomputable def proposition4 : Prop := ∀ (x : ℝ), 4 * x^2 > 2 * x - 1 + 3 * x^2

theorem number_of_true_propositions : (¬ proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ ¬ proposition4) → 1 = 1 :=
by
  intros
  sorry

end number_of_true_propositions_l42_42493


namespace westward_measurement_l42_42966

def east_mov (d : ℕ) : ℤ := - (d : ℤ)

def west_mov (d : ℕ) : ℤ := d

theorem westward_measurement :
  east_mov 50 = -50 →
  west_mov 60 = 60 :=
by
  intro h
  exact rfl

end westward_measurement_l42_42966


namespace max_parts_divided_by_three_planes_l42_42819

theorem max_parts_divided_by_three_planes (parts_0_plane parts_1_plane parts_2_planes parts_3_planes: ℕ)
  (h0 : parts_0_plane = 1)
  (h1 : parts_1_plane = 2)
  (h2 : parts_2_planes = 4)
  (h3 : parts_3_planes = 8) :
  parts_3_planes = 8 :=
by
  sorry

end max_parts_divided_by_three_planes_l42_42819


namespace determinant_of_A_l42_42530

section
  open Matrix

  -- Define the given matrix
  def A : Matrix (Fin 3) (Fin 3) ℤ :=
    ![ ![0, 2, -4], ![6, -1, 3], ![2, -3, 5] ]

  -- State the theorem for the determinant
  theorem determinant_of_A : det A = 16 :=
  sorry
end

end determinant_of_A_l42_42530


namespace find_product_of_abc_l42_42258

theorem find_product_of_abc :
  ∃ (a b c m : ℝ), 
    a + b + c = 195 ∧
    m = 8 * a ∧
    m = b - 10 ∧
    m = c + 10 ∧
    a * b * c = 95922 := by
  sorry

end find_product_of_abc_l42_42258


namespace subtraction_example_l42_42832

theorem subtraction_example : 34.256 - 12.932 - 1.324 = 20.000 := 
by
  sorry

end subtraction_example_l42_42832


namespace smaller_number_is_neg_five_l42_42554

theorem smaller_number_is_neg_five (x y : ℤ) (h1 : x + y = 30) (h2 : x - y = 40) : y = -5 :=
by
  sorry

end smaller_number_is_neg_five_l42_42554


namespace xyz_expr_min_max_l42_42860

open Real

theorem xyz_expr_min_max (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 1) :
  ∃ m M : ℝ, m = 0 ∧ M = 1/4 ∧
    (∀ x y z : ℝ, x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
      xy + yz + zx - 3 * xyz ≥ m ∧ xy + yz + zx - 3 * xyz ≤ M) :=
sorry

end xyz_expr_min_max_l42_42860


namespace center_of_circle_l42_42584

theorem center_of_circle : ∃ c : ℝ × ℝ, (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ ((x - c.1)^2 + (y + c.2)^2 = 2))) ∧ (c = (1, -2)) :=
by
  -- Proof is omitted
  sorry

end center_of_circle_l42_42584


namespace solve_S20_minus_2S10_l42_42804

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a n > 0) ∧
  (∀ n : ℕ, n ≥ 2 → S n = (n / (n - 1 : ℝ)) * (a n ^ 2 - a 1 ^ 2))

theorem solve_S20_minus_2S10 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    arithmetic_sequence a S →
    S 20 - 2 * S 10 = 50 :=
by
  intros
  sorry

end solve_S20_minus_2S10_l42_42804


namespace num_rectangles_in_5x5_grid_l42_42759

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l42_42759


namespace susan_avg_speed_l42_42429

variable (d1 d2 : ℕ) (s1 s2 : ℕ)

def time (d s : ℕ) : ℚ := d / s

theorem susan_avg_speed 
  (h1 : d1 = 40) 
  (h2 : s1 = 30) 
  (h3 : d2 = 40) 
  (h4 : s2 = 15) : 
  (d1 + d2) / (time d1 s1 + time d2 s2) = 20 := 
by 
  -- Sorry to skip the proof.
  sorry

end susan_avg_speed_l42_42429


namespace child_running_speed_l42_42667

theorem child_running_speed
  (c s t : ℝ)
  (h1 : (74 - s) * 3 = 165)
  (h2 : (74 + s) * t = 372) :
  c = 74 :=
by sorry

end child_running_speed_l42_42667


namespace find_fraction_l42_42685

theorem find_fraction
  (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℚ)
  (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : b₁ = 6) (h₄ : b₂ = 5)
  (h₅ : c₁ = 1) (h₆ : c₂ = 7)
  (h : (a₁ / a₂) / (b₁ / b₂) = (c₁ / c₂) / (x / y)) :
  (x / y) = 2 / 5 := 
by
  sorry

end find_fraction_l42_42685


namespace initial_amount_liquid_A_l42_42261

theorem initial_amount_liquid_A (A B : ℝ) (h1 : A / B = 4)
    (h2 : (A / (B + 40)) = 2 / 3) : A = 32 := by
  sorry

end initial_amount_liquid_A_l42_42261


namespace tournament_cycle_exists_l42_42287

theorem tournament_cycle_exists :
  ∃ (A B C : Fin 12), 
  (∃ M : Fin 12 → Fin 12 → Bool, 
    (∀ p : Fin 12, ∃ q : Fin 12, q ≠ p ∧ M p q) ∧
    M A B = true ∧ M B C = true ∧ M C A = true) :=
sorry

end tournament_cycle_exists_l42_42287


namespace alex_loan_comparison_l42_42962

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem alex_loan_comparison :
  let P : ℝ := 15000
  let r1 : ℝ := 0.08
  let r2 : ℝ := 0.10
  let n : ℕ := 12
  let t1_10 : ℝ := 10
  let t1_5 : ℝ := 5
  let t2 : ℝ := 15
  let owed_after_10 := compound_interest P r1 n t1_10
  let payment_after_10 := owed_after_10 / 2
  let remaining_after_10 := owed_after_10 / 2
  let owed_after_15 := compound_interest remaining_after_10 r1 n t1_5
  let total_payment_option1 := payment_after_10 + owed_after_15
  let total_payment_option2 := simple_interest P r2 t2
  total_payment_option1 - total_payment_option2 = 4163 :=
by
  sorry

end alex_loan_comparison_l42_42962


namespace total_profit_calculation_l42_42709

-- Definitions based on conditions
def initial_investment_A := 5000
def initial_investment_B := 8000
def initial_investment_C := 9000
def initial_investment_D := 7000

def investment_A_after_4_months := initial_investment_A + 2000
def investment_B_after_4_months := initial_investment_B - 1000

def investment_C_after_6_months := initial_investment_C + 3000
def investment_D_after_6_months := initial_investment_D + 5000

def profit_A_percentage := 20
def profit_B_percentage := 30
def profit_C_percentage := 25
def profit_D_percentage := 25

def profit_C := 60000

-- Total profit is what we need to determine
def total_profit := 240000

-- The proof statement
theorem total_profit_calculation :
  total_profit = (profit_C * 100) / profit_C_percentage := 
by 
  sorry

end total_profit_calculation_l42_42709


namespace range_of_a_l42_42310

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (1 - x - a) < 1) → -1/2 < a ∧ a < 3/2 :=
by
  sorry

end range_of_a_l42_42310


namespace correct_calculation_l42_42646

variable (a b : ℝ)

theorem correct_calculation : (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l42_42646


namespace prime_numbers_in_list_l42_42910

noncomputable def list_numbers : ℕ → ℕ
| 0       => 43
| (n + 1) => 43 * ((10 ^ (2 * n + 2) - 1) / 99) 

theorem prime_numbers_in_list : ∃ n:ℕ, (∀ m, (m > n) → ¬ Prime (list_numbers m)) ∧ Prime (list_numbers 0) := 
by
  sorry

end prime_numbers_in_list_l42_42910


namespace negation_of_p_l42_42257

variable (p : ∀ x : ℝ, x^2 + x - 6 ≤ 0)

theorem negation_of_p : (∃ x : ℝ, x^2 + x - 6 > 0) :=
sorry

end negation_of_p_l42_42257


namespace inequality_abc_l42_42495

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (b * c / a) + (a * c / b) + (a * b / c) ≥ a + b + c := 
  sorry

end inequality_abc_l42_42495


namespace exponent_equality_l42_42358

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l42_42358


namespace gcd_of_18_and_30_l42_42481

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l42_42481


namespace excircle_inequality_l42_42947

variables {a b c : ℝ} -- The sides of the triangle

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2 -- Definition of semiperimeter

noncomputable def excircle_distance (p a : ℝ) : ℝ := p - a -- Distance from vertices to tangency points

theorem excircle_inequality (a b c : ℝ) (p : ℝ) 
    (h1 : p = semiperimeter a b c) : 
    (excircle_distance p a) + (excircle_distance p b) > p := 
by
    -- Placeholder for proof
    sorry

end excircle_inequality_l42_42947


namespace range_zero_of_roots_l42_42113

theorem range_zero_of_roots (x y z w : ℝ) (h1 : x + y + z + w = 0) 
                            (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
  sorry

end range_zero_of_roots_l42_42113


namespace math_problem_l42_42370
open Real

noncomputable def problem_statement : Prop :=
  let a := 99
  let b := 3
  let c := 20
  let area := (99 * sqrt 3) / 20
  a + b + c = 122 ∧ 
  ∃ (AB: ℝ) (QR: ℝ), AB = 14 ∧ QR = 3 * sqrt 3 ∧ area = (1 / 2) * QR * (QR / (2 * (sqrt 3 / 2))) * (sqrt 3 / 2)

theorem math_problem : problem_statement := by
  sorry

end math_problem_l42_42370


namespace num_values_divisible_by_120_l42_42311

theorem num_values_divisible_by_120 (n : ℕ) (h_seq : ∀ n, ∃ k, n = k * (k + 1)) :
  ∃ k, k = 8 := sorry

end num_values_divisible_by_120_l42_42311


namespace more_pens_than_pencils_l42_42475

-- Define the number of pencils (P) and pens (Pe)
def num_pencils : ℕ := 15 * 80

-- Define the number of pens (Pe) is more than twice the number of pencils (P)
def num_pens (Pe : ℕ) : Prop := Pe > 2 * num_pencils

-- State the total cost equation in terms of pens and pencils
def total_cost_eq (Pe : ℕ) : Prop := (5 * Pe + 4 * num_pencils = 18300)

-- Prove that the number of more pens than pencils is 1500
theorem more_pens_than_pencils (Pe : ℕ) (h1 : num_pens Pe) (h2 : total_cost_eq Pe) : (Pe - num_pencils = 1500) :=
by
  sorry

end more_pens_than_pencils_l42_42475


namespace women_doubles_tournament_handshakes_l42_42442

theorem women_doubles_tournament_handshakes :
  ∀ (teams : List (List Prop)), List.length teams = 4 → (∀ t ∈ teams, List.length t = 2) →
  (∃ (handshakes : ℕ), handshakes = 24) :=
by
  intro teams h1 h2
  -- Assume teams are disjoint and participants shake hands meeting problem conditions
  -- The lean proof will follow the logical structure used for the mathematical solution
  -- We'll now formalize the conditions and the handshake calculation
  sorry

end women_doubles_tournament_handshakes_l42_42442


namespace f_neg1_plus_f_2_l42_42375

def f (x : Int) : Int :=
  if x = -3 then -1
  else if x = -2 then -5
  else if x = -1 then -2
  else if x = 0 then 0
  else if x = 1 then 2
  else if x = 2 then 1
  else if x = 3 then 4
  else 0  -- This handles x values not explicitly in the table, although technically unnecessary.

theorem f_neg1_plus_f_2 : f (-1) + f (2) = -1 := by
  sorry

end f_neg1_plus_f_2_l42_42375


namespace lucas_journey_distance_l42_42245

noncomputable def distance (D : ℝ) : ℝ :=
  let usual_speed := D / 150
  let distance_before_traffic := 2 * D / 5
  let speed_after_traffic := usual_speed - 1 / 2
  let time_before_traffic := distance_before_traffic / usual_speed
  let time_after_traffic := (3 * D / 5) / speed_after_traffic
  time_before_traffic + time_after_traffic

theorem lucas_journey_distance : ∃ D : ℝ, distance D = 255 ∧ D = 48.75 :=
sorry

end lucas_journey_distance_l42_42245


namespace cost_price_percentage_l42_42654

-- Define the condition that the profit percent is 11.11111111111111%
def profit_percent (CP SP: ℝ) : Prop :=
  ((SP - CP) / CP) * 100 = 11.11111111111111

-- Prove that under this condition, the cost price (CP) is 90% of the selling price (SP).
theorem cost_price_percentage (CP SP : ℝ) (h: profit_percent CP SP) : (CP / SP) * 100 = 90 :=
sorry

end cost_price_percentage_l42_42654


namespace mt_product_l42_42974

noncomputable def g (x : ℝ) : ℝ := sorry

theorem mt_product
  (hg : ∀ (x y : ℝ), g (g x + y) = g x + g (g y + g (-x)) - x) : 
  ∃ m t : ℝ, m = 1 ∧ t = -5 ∧ m * t = -5 := 
by
  sorry

end mt_product_l42_42974


namespace cos_five_pi_over_four_l42_42950

theorem cos_five_pi_over_four : Real.cos (5 * Real.pi / 4) = -1 / Real.sqrt 2 := 
by
  sorry

end cos_five_pi_over_four_l42_42950


namespace complex_power_identity_l42_42917

theorem complex_power_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end complex_power_identity_l42_42917


namespace percentage_of_games_not_won_is_40_l42_42845

def ratio_games_won_to_lost (games_won games_lost : ℕ) : Prop := 
  games_won / gcd games_won games_lost = 3 ∧ games_lost / gcd games_won games_lost = 2

def total_games (games_won games_lost ties : ℕ) : ℕ :=
  games_won + games_lost + ties

def percentage_games_not_won (games_won games_lost ties : ℕ) : ℕ :=
  ((games_lost + ties) * 100) / (games_won + games_lost + ties)

theorem percentage_of_games_not_won_is_40
  (games_won games_lost ties : ℕ)
  (h_ratio : ratio_games_won_to_lost games_won games_lost)
  (h_ties : ties = 5)
  (h_no_other_games : games_won + games_lost + ties = total_games games_won games_lost ties) :
  percentage_games_not_won games_won games_lost ties = 40 := 
sorry

end percentage_of_games_not_won_is_40_l42_42845


namespace find_common_ratio_limit_SN_over_TN_l42_42693

noncomputable def S (q : ℚ) (n : ℕ) : ℚ := (1 - q^n) / (1 - q)
noncomputable def T (q : ℚ) (n : ℕ) : ℚ := (1 - q^(2 * n)) / (1 - q^2)

theorem find_common_ratio
  (S3 : S q 3 = 3)
  (S6 : S q 6 = -21) :
  q = -2 :=
sorry

theorem limit_SN_over_TN
  (q_pos : 0 < q)
  (Tn_def : ∀ n, T q n = 1) :
  (q > 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 0| < ε) ∧
  (0 < q ∧ q < 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - (1 + q)| < ε) ∧
  (q = 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 1| < ε) :=
sorry

end find_common_ratio_limit_SN_over_TN_l42_42693


namespace positive_difference_l42_42998

theorem positive_difference (y : ℤ) (h : (46 + y) / 2 = 52) : |y - 46| = 12 := by
  sorry

end positive_difference_l42_42998


namespace find_alpha_l42_42520

theorem find_alpha
  (α : Real)
  (h1 : α > 0)
  (h2 : α < π)
  (h3 : 1 / Real.sin α + 1 / Real.cos α = 2) :
  α = π + 1 / 2 * Real.arcsin ((1 - Real.sqrt 5) / 2) :=
sorry

end find_alpha_l42_42520


namespace cut_scene_length_l42_42789

theorem cut_scene_length
  (original_length final_length : ℕ)
  (h_original : original_length = 60)
  (h_final : final_length = 54) :
  original_length - final_length = 6 :=
by 
  sorry

end cut_scene_length_l42_42789


namespace max_value_T_n_l42_42769

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q^n

noncomputable def sum_of_first_n_terms (a₁ q : ℝ) (n : ℕ) :=
  a₁ * (1 - q^(n + 1)) / (1 - q)

noncomputable def T_n (a₁ q : ℝ) (n : ℕ) :=
  (9 * sum_of_first_n_terms a₁ q n - sum_of_first_n_terms a₁ q (2 * n)) /
  geometric_sequence a₁ q (n + 1)

theorem max_value_T_n
  (a₁ : ℝ) (n : ℕ) (h : n > 0) (q : ℝ) (hq : q = 2) :
  ∃ n₀ : ℕ, T_n a₁ q n₀ = 3 := sorry

end max_value_T_n_l42_42769


namespace scrooge_share_l42_42902

def leftover_pie : ℚ := 8 / 9

def share_each (x : ℚ) : Prop :=
  2 * x + 3 * x = leftover_pie

theorem scrooge_share (x : ℚ):
  share_each x → (2 * x = 16 / 45) := by
  sorry

end scrooge_share_l42_42902


namespace train_length_l42_42244

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (length_m : ℝ)
  (h1 : speed_kmph = 120) 
  (h2 : time_sec = 6)
  (h3 : speed_ms = 33.33)
  (h4 : length_m = 200) : 
  speed_kmph * 1000 / 3600 * time_sec = length_m :=
by
  sorry

end train_length_l42_42244


namespace range_of_k_tan_alpha_l42_42023

noncomputable def f (x k : Real) : Real := Real.sin x + k

theorem range_of_k (k : Real) : 
  (∃ x : Real, f x k = 1) ↔ (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem tan_alpha (α k : Real) (h : α ∈ Set.Ioo (0 : Real) Real.pi) (hf : f α k = 1 / 3 + k) : 
  Real.tan α = Real.sqrt 2 / 4 :=
sorry

end range_of_k_tan_alpha_l42_42023


namespace percentage_reduction_price_increase_for_profit_price_increase_max_profit_l42_42128

-- Define the conditions
def original_price : ℝ := 50
def final_price : ℝ := 32
def daily_sales : ℝ := 500
def profit_per_kg : ℝ := 10
def sales_decrease_per_yuan : ℝ := 20
def required_daily_profit : ℝ := 6000
def max_possible_profit : ℝ := 6125

-- Proving the percentage reduction each time
theorem percentage_reduction (a : ℝ) :
  (original_price * (1 - a) ^ 2 = final_price) → (a = 0.2) :=
sorry

-- Proving the price increase per kilogram to ensure a daily profit of 6000 yuan
theorem price_increase_for_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = required_daily_profit) → (x = 5) :=
sorry

-- Proving the price increase per kilogram to maximize daily profit
theorem price_increase_max_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = max_possible_profit) → (x = 7.5) :=
sorry

end percentage_reduction_price_increase_for_profit_price_increase_max_profit_l42_42128


namespace two_digit_numbers_condition_l42_42187

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end two_digit_numbers_condition_l42_42187


namespace intersection_P_Q_l42_42632

def P : Set ℝ := {x | Real.log x / Real.log 2 < -1}
def Q : Set ℝ := {x | abs x < 1}

theorem intersection_P_Q : P ∩ Q = {x | 0 < x ∧ x < 1 / 2} := by
  sorry

end intersection_P_Q_l42_42632


namespace sequence_6th_term_sequence_1994th_term_l42_42214

def sequence_term (n : Nat) : Nat := n * (n + 1)

theorem sequence_6th_term:
  sequence_term 6 = 42 :=
by
  -- proof initially skipped
  sorry

theorem sequence_1994th_term:
  sequence_term 1994 = 3978030 :=
by
  -- proof initially skipped
  sorry

end sequence_6th_term_sequence_1994th_term_l42_42214


namespace sum_mod_12_l42_42792

def remainder_sum_mod :=
  let nums := [10331, 10333, 10335, 10337, 10339, 10341, 10343]
  let sum_nums := nums.sum
  sum_nums % 12 = 7

theorem sum_mod_12 : remainder_sum_mod :=
by
  sorry

end sum_mod_12_l42_42792


namespace min_value_expression_l42_42548

theorem min_value_expression {x y z w : ℝ} 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) 
  (hw : 0 ≤ w ∧ w ≤ 1) : 
  ∃ m, m = 2 ∧ ∀ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) →
  m ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) :=
by
  sorry

end min_value_expression_l42_42548


namespace non_shaded_area_l42_42458

theorem non_shaded_area (r : ℝ) (A : ℝ) (shaded : ℝ) (non_shaded : ℝ) :
  (r = 5) ∧ (A = 4 * (π * r^2)) ∧ (shaded = 8 * (1 / 4 * π * r^2 - (1 / 2 * r * r))) ∧
  (non_shaded = A - shaded) → 
  non_shaded = 50 * π + 100 :=
by
  intro h
  obtain ⟨r_eq_5, A_eq, shaded_eq, non_shaded_eq⟩ := h
  rw [r_eq_5] at *
  sorry

end non_shaded_area_l42_42458


namespace exterior_angle_regular_polygon_l42_42070

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l42_42070


namespace investment_double_l42_42120

theorem investment_double (A : ℝ) (r t : ℝ) (hA : 0 < A) (hr : 0 < r) :
  2 * A ≤ A * (1 + r)^t ↔ t ≥ (Real.log 2) / (Real.log (1 + r)) := 
by
  sorry

end investment_double_l42_42120


namespace probability_of_yellow_or_green_l42_42603

def bag : List (String × Nat) := [("yellow", 4), ("green", 3), ("red", 2), ("blue", 1)]

def total_marbles (bag : List (String × Nat)) : Nat := bag.foldr (fun (_, n) acc => n + acc) 0

def favorable_outcomes (bag : List (String × Nat)) : Nat :=
  (bag.filter (fun (color, _) => color = "yellow" ∨ color = "green")).foldr (fun (_, n) acc => n + acc) 0

theorem probability_of_yellow_or_green :
  (favorable_outcomes bag : ℚ) / (total_marbles bag : ℚ) = 7 / 10 := by
  sorry

end probability_of_yellow_or_green_l42_42603


namespace triangle_side_relation_l42_42690

theorem triangle_side_relation (a b c : ℝ) (α β γ : ℝ)
  (h1 : 3 * α + 2 * β = 180)
  (h2 : α + β + γ = 180) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_side_relation_l42_42690


namespace smaller_circle_circumference_l42_42926

noncomputable def circumference_of_smaller_circle :=
  let π := Real.pi
  let R := 352 / (2 * π)
  let area_difference := 4313.735577562732
  let R_squared_minus_r_squared := area_difference / π
  let r_squared := R ^ 2 - R_squared_minus_r_squared
  let r := Real.sqrt r_squared
  2 * π * r

theorem smaller_circle_circumference : 
  let circumference_larger := 352
  let area_difference := 4313.735577562732
  circumference_of_smaller_circle = 263.8934 := sorry

end smaller_circle_circumference_l42_42926


namespace arc_length_is_correct_l42_42253

-- Define the radius and central angle as given
def radius := 16
def central_angle := 2

-- Define the arc length calculation
def arc_length (r : ℕ) (α : ℕ) := α * r

-- The theorem stating the mathematically equivalent proof problem
theorem arc_length_is_correct : arc_length radius central_angle = 32 :=
by sorry

end arc_length_is_correct_l42_42253


namespace factor_in_form_of_2x_l42_42676

theorem factor_in_form_of_2x (w : ℕ) (hw : w = 144) : ∃ x : ℕ, 936 * w = 2^x * P → x = 4 :=
by
  sorry

end factor_in_form_of_2x_l42_42676


namespace surface_area_of_cone_l42_42167

-- Definitions based solely on conditions
def central_angle (θ : ℝ) := θ = (2 * Real.pi) / 3
def slant_height (l : ℝ) := l = 2
def radius_cone (r : ℝ) := ∃ (θ l : ℝ), central_angle θ ∧ slant_height l ∧ θ * l = 2 * Real.pi * r
def lateral_surface_area (A₁ : ℝ) (r l : ℝ) := A₁ = Real.pi * r * l
def base_area (A₂ : ℝ) (r : ℝ) := A₂ = Real.pi * r^2
def total_surface_area (A A₁ A₂ : ℝ) := A = A₁ + A₂

-- The theorem proving the total surface area is as specified
theorem surface_area_of_cone :
  ∃ (r l A₁ A₂ A : ℝ), central_angle ((2 * Real.pi) / 3) ∧ slant_height 2 ∧ radius_cone r ∧
  lateral_surface_area A₁ r 2 ∧ base_area A₂ r ∧ total_surface_area A A₁ A₂ ∧ A = (16 * Real.pi) / 9 := sorry

end surface_area_of_cone_l42_42167


namespace fraction_power_calc_l42_42887

theorem fraction_power_calc : 
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
sorry

end fraction_power_calc_l42_42887


namespace find_other_root_l42_42456

variables {a b c : ℝ}

theorem find_other_root
  (h_eq : ∀ x : ℝ, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0)
  (root1 : a * (b - c) * 1^2 + b * (c - a) * 1 + c * (a - b) = 0) :
  ∃ k : ℝ, k = c * (a - b) / (a * (b - c)) ∧
           a * (b - c) * k^2 + b * (c - a) * k + c * (a - b) = 0 := 
sorry

end find_other_root_l42_42456


namespace tile_covering_problem_l42_42691

theorem tile_covering_problem :
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12  -- converting feet to inches
  let region_width := 3 * 12   -- converting feet to inches
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area = 144 := 
by 
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12
  let region_width := 3 * 12
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  sorry

end tile_covering_problem_l42_42691


namespace ratio_of_members_l42_42617

theorem ratio_of_members (r p : ℕ) (h1 : 5 * r + 12 * p = 8 * (r + p)) : (r / p : ℚ) = 4 / 3 := by
  sorry -- This is a placeholder for the actual proof.

end ratio_of_members_l42_42617


namespace smallest_value_of_a_squared_plus_b_l42_42505

theorem smallest_value_of_a_squared_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1) :
    a^2 + b = 2 / (3 * Real.sqrt 3) :=
by
  sorry

end smallest_value_of_a_squared_plus_b_l42_42505


namespace silver_cost_l42_42044

theorem silver_cost (S : ℝ) : 
  (1.5 * S) + (3 * 50 * S) = 3030 → S = 20 :=
by
  intro h
  sorry

end silver_cost_l42_42044


namespace problem1_problem2_problem3_l42_42698

noncomputable def f (b a : ℝ) (x : ℝ) := (b - 2^x) / (2^x + a) 

-- (1) Prove values of a and b
theorem problem1 (a b : ℝ) : 
  (f b a 0 = 0) ∧ (f b a (-1) = -f b a 1) → (a = 1 ∧ b = 1) :=
sorry

-- (2) Prove f is decreasing function
theorem problem2 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f b a x₁ - f b a x₂ > 0 :=
sorry

-- (3) Find range of k such that inequality always holds
theorem problem3 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) (k : ℝ) : 
  (∀ t : ℝ, f b a (t^2 - 2*t) + f b a (2*t^2 - k) < 0) → k < -(1/3) :=
sorry

end problem1_problem2_problem3_l42_42698


namespace right_triangle_hypotenuse_product_square_l42_42981

theorem right_triangle_hypotenuse_product_square (A₁ A₂ : ℝ) (a₁ b₁ a₂ b₂ : ℝ) 
(h₁ : a₁ * b₁ / 2 = A₁) (h₂ : a₂ * b₂ / 2 = A₂) 
(h₃ : A₁ = 2) (h₄ : A₂ = 3) 
(h₅ : a₁ = a₂) (h₆ : b₂ = 2 * b₁) : 
(a₁ ^ 2 + b₁ ^ 2) * (a₂ ^ 2 + b₂ ^ 2) = 325 := 
by sorry

end right_triangle_hypotenuse_product_square_l42_42981


namespace distance_between_lines_correct_l42_42336

noncomputable def distance_between_parallel_lines 
  (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_correct :
  distance_between_parallel_lines 4 2 (-2) 1 = 3 * Real.sqrt 5 / 10 :=
by
  -- Proof steps would go here
  sorry

end distance_between_lines_correct_l42_42336


namespace general_term_formula_l42_42001

theorem general_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 3^n - 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n - 1)) →
  a 1 = 2 →
  ∀ n, a n = 2 * 3^(n - 1) :=
by
    intros hS ha h1 n
    sorry

end general_term_formula_l42_42001


namespace find_ab_l42_42248

theorem find_ab (a b : ℕ) (h : (Real.sqrt 30 - Real.sqrt 18) * (3 * Real.sqrt a + Real.sqrt b) = 12) : a = 2 ∧ b = 30 :=
sorry

end find_ab_l42_42248


namespace projectile_reaches_45_feet_first_time_l42_42057

theorem projectile_reaches_45_feet_first_time :
  ∃ t : ℝ, (-20 * t^2 + 90 * t = 45) ∧ abs (t - 0.9) < 0.1 := sorry

end projectile_reaches_45_feet_first_time_l42_42057


namespace cistern_leak_time_l42_42186

theorem cistern_leak_time (R : ℝ) (L : ℝ) (eff_R : ℝ) : 
  (R = 1/5) → 
  (eff_R = 1/6) → 
  (eff_R = R - L) → 
  (1 / L = 30) :=
by
  intros hR heffR heffRate
  sorry

end cistern_leak_time_l42_42186


namespace sum_is_zero_l42_42440

variable (a b c x y : ℝ)

theorem sum_is_zero (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
(h4 : a^3 + a * x + y = 0)
(h5 : b^3 + b * x + y = 0)
(h6 : c^3 + c * x + y = 0) : a + b + c = 0 :=
sorry

end sum_is_zero_l42_42440


namespace car_A_overtakes_car_B_l42_42571

theorem car_A_overtakes_car_B (z : ℕ) :
  let y := (5 * z) / 4
  let x := (13 * z) / 10
  10 * y / (x - y) = 250 := 
by
  sorry

end car_A_overtakes_car_B_l42_42571


namespace value_of_a_plus_b_l42_42298

theorem value_of_a_plus_b (a b : ℕ) (h1 : Real.sqrt 44 = 2 * Real.sqrt a) (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l42_42298


namespace cubic_conversion_l42_42004

theorem cubic_conversion (h : 1 = 100) : 1 = 1000000 :=
by
  sorry

end cubic_conversion_l42_42004


namespace cubic_roots_and_k_value_l42_42459

theorem cubic_roots_and_k_value (k r₃ : ℝ) :
  (∃ r₃, 3 - 2 + r₃ = -5 ∧ 3 * (-2) * r₃ = -12 ∧ k = 3 * (-2) + (-2) * r₃ + r₃ * 3) →
  (k = -12 ∧ r₃ = -6) :=
by
  sorry

end cubic_roots_and_k_value_l42_42459


namespace isabella_most_efficient_jumper_l42_42653

noncomputable def weight_ricciana : ℝ := 120
noncomputable def jump_ricciana : ℝ := 4

noncomputable def weight_margarita : ℝ := 110
noncomputable def jump_margarita : ℝ := 2 * jump_ricciana - 1

noncomputable def weight_isabella : ℝ := 100
noncomputable def jump_isabella : ℝ := jump_ricciana + 3

noncomputable def ratio_ricciana : ℝ := weight_ricciana / jump_ricciana
noncomputable def ratio_margarita : ℝ := weight_margarita / jump_margarita
noncomputable def ratio_isabella : ℝ := weight_isabella / jump_isabella

theorem isabella_most_efficient_jumper :
  ratio_isabella < ratio_margarita ∧ ratio_isabella < ratio_ricciana :=
by
  sorry

end isabella_most_efficient_jumper_l42_42653


namespace geometric_sequence_sum_l42_42752

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : ∀ n, a n > 0)
  (h1 : a 1 + a 3 = 3)
  (h2 : a 4 + a 6 = 6):
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 :=
sorry

end geometric_sequence_sum_l42_42752


namespace students_drawn_from_A_l42_42048

-- Define the conditions as variables (number of students in each school)
def studentsA := 3600
def studentsB := 5400
def studentsC := 1800
def sampleSize := 90

-- Define the total number of students
def totalStudents := studentsA + studentsB + studentsC

-- Define the proportion of students in School A
def proportionA := studentsA / totalStudents

-- Define the number of students to be drawn from School A using stratified sampling
def drawnFromA := sampleSize * proportionA

-- The theorem to prove
theorem students_drawn_from_A : drawnFromA = 30 :=
by
  sorry

end students_drawn_from_A_l42_42048


namespace min_people_same_score_l42_42511

theorem min_people_same_score (participants : ℕ) (nA nB : ℕ) (pointsA pointsB : ℕ) (scores : Finset ℕ) :
  participants = 400 →
  nA = 8 →
  nB = 6 →
  pointsA = 4 →
  pointsB = 7 →
  scores.card = (nA + 1) * (nB + 1) - 6 →
  participants / scores.card < 8 :=
by
  intros h_participants h_nA h_nB h_pointsA h_pointsB h_scores_card
  sorry

end min_people_same_score_l42_42511


namespace minimum_blue_chips_l42_42276

theorem minimum_blue_chips (w r b : ℕ) : 
  (b ≥ w / 3) ∧ (b ≤ r / 4) ∧ (w + b ≥ 75) → b ≥ 19 :=
by sorry

end minimum_blue_chips_l42_42276


namespace john_weekly_allowance_l42_42042

noncomputable def weekly_allowance (A : ℝ) :=
  (3/5) * A + (1/3) * ((2/5) * A) + 0.60 <= A

theorem john_weekly_allowance : ∃ A : ℝ, (3/5) * A + (1/3) * ((2/5) * A) + 0.60 = A := by
  let A := 2.25
  sorry

end john_weekly_allowance_l42_42042


namespace device_identification_l42_42341

def sum_of_device_numbers (numbers : List ℕ) : ℕ :=
  numbers.foldr (· + ·) 0

def is_standard_device (d : List ℕ) : Prop :=
  (d = [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ (sum_of_device_numbers d = 45)

theorem device_identification (d : List ℕ) : 
  (sum_of_device_numbers d = 45) → is_standard_device d :=
by
  sorry

end device_identification_l42_42341


namespace chris_dana_shared_rest_days_l42_42394

/-- Chris's and Dana's working schedules -/
structure work_schedule where
  work_days : ℕ
  rest_days : ℕ

/-- Define Chris's and Dana's schedules -/
def Chris_schedule : work_schedule := { work_days := 5, rest_days := 2 }
def Dana_schedule : work_schedule := { work_days := 6, rest_days := 1 }

/-- Number of days to consider -/
def total_days : ℕ := 1200

/-- Combinatorial function to calculate the number of coinciding rest-days -/
noncomputable def coinciding_rest_days (schedule1 schedule2 : work_schedule) (days : ℕ) : ℕ :=
  (days / (Nat.lcm (schedule1.work_days + schedule1.rest_days) (schedule2.work_days + schedule2.rest_days)))

/-- The proof problem statement -/
theorem chris_dana_shared_rest_days : 
coinciding_rest_days Chris_schedule Dana_schedule total_days = 171 :=
by sorry

end chris_dana_shared_rest_days_l42_42394


namespace slices_remaining_is_correct_l42_42663

def slices_per_pizza : ℕ := 8
def pizzas_ordered : ℕ := 2
def slices_eaten : ℕ := 7
def total_slices : ℕ := slices_per_pizza * pizzas_ordered
def slices_remaining : ℕ := total_slices - slices_eaten

theorem slices_remaining_is_correct : slices_remaining = 9 := by
  sorry

end slices_remaining_is_correct_l42_42663


namespace total_afternoon_evening_emails_l42_42875

-- Definitions based on conditions
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

-- Statement to be proven
theorem total_afternoon_evening_emails : afternoon_emails + evening_emails = 13 :=
by 
  sorry

end total_afternoon_evening_emails_l42_42875


namespace greatest_product_of_digits_l42_42291

theorem greatest_product_of_digits :
  ∀ a b : ℕ, (10 * a + b) % 35 = 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  ∃ ab_max : ℕ, ab_max = a * b ∧ ab_max = 15 :=
by
  sorry

end greatest_product_of_digits_l42_42291


namespace roots_quadratic_square_diff_10_l42_42088

-- Definition and theorem statement in Lean 4
theorem roots_quadratic_square_diff_10 :
  ∀ x1 x2 : ℝ, (2 * x1^2 + 4 * x1 - 3 = 0) ∧ (2 * x2^2 + 4 * x2 - 3 = 0) →
  (x1 - x2)^2 = 10 :=
by
  sorry

end roots_quadratic_square_diff_10_l42_42088


namespace square_problem_solution_l42_42134

theorem square_problem_solution
  (x : ℝ)
  (h1 : ∃ s1 : ℝ, s1^2 = x^2 + 12*x + 36)
  (h2 : ∃ s2 : ℝ, s2^2 = 4*x^2 - 12*x + 9)
  (h3 : 4 * (s1 + s2) = 64) :
  x = 13 / 3 :=
by
  sorry

end square_problem_solution_l42_42134


namespace distinct_ordered_pairs_l42_42347

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end distinct_ordered_pairs_l42_42347


namespace sym_diff_A_B_l42_42585

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x < 0}

theorem sym_diff_A_B :
  sym_diff A B = {x | x < -1} ∪ {x | 0 ≤ x ∧ x < 1} := by
  sorry

end sym_diff_A_B_l42_42585


namespace tangents_quadrilateral_cyclic_l42_42985

variables {A B C D K L O1 O2 : Point}
variable (r : ℝ)
variable (AB_cut_circles : ∀ {A B : Point} {O1 O2 : Point}, is_intersect AB O1 O2)
variable (parallel_AB_O1O2 : is_parallel AB O1O2)
variable (tangents_formed_quadrilateral : is_quadrilateral C D K L)
variable (quadrilateral_contains_circles : contains C D K L O1 O2)

theorem tangents_quadrilateral_cyclic
  (h1: AB_cut_circles)
  (h2: parallel_AB_O1O2) 
  (h3: tangents_formed_quadrilateral)
  (h4: quadrilateral_contains_circles)
  : ∃ O : Circle, is_inscribed O C D K L :=
sorry

end tangents_quadrilateral_cyclic_l42_42985


namespace congruence_problem_l42_42121

theorem congruence_problem {x : ℤ} (h : 4 * x + 5 ≡ 3 [ZMOD 20]) : 3 * x + 8 ≡ 2 [ZMOD 10] :=
sorry

end congruence_problem_l42_42121


namespace average_value_eq_l42_42997

variable (x : ℝ)

theorem average_value_eq :
  ( -4 * x + 0 + 4 * x + 12 * x + 20 * x ) / 5 = 6.4 * x :=
by
  sorry

end average_value_eq_l42_42997


namespace tangent_line_y_intercept_l42_42990

def circle1Center: ℝ × ℝ := (3, 0)
def circle1Radius: ℝ := 3
def circle2Center: ℝ × ℝ := (7, 0)
def circle2Radius: ℝ := 2

theorem tangent_line_y_intercept
    (tangent_line: ℝ × ℝ -> ℝ) 
    (P : tangent_line (circle1Center.1, circle1Center.2 + circle1Radius) = 0) -- Tangent condition for Circle 1
    (Q : tangent_line (circle2Center.1, circle2Center.2 + circle2Radius) = 0) -- Tangent condition for Circle 2
    :
    tangent_line (0, 4.5) = 0 := 
sorry

end tangent_line_y_intercept_l42_42990


namespace rectangle_circle_area_ratio_l42_42781

theorem rectangle_circle_area_ratio (w r: ℝ) (h1: 3 * w = π * r) (h2: 2 * w = l) :
  (l * w) / (π * r ^ 2) = 2 * π / 9 :=
by 
  sorry

end rectangle_circle_area_ratio_l42_42781


namespace S_63_value_l42_42265

noncomputable def b (n : ℕ) : ℚ := (3 + (-1)^(n-1))/2

noncomputable def a : ℕ → ℚ
| 0       => 0
| 1       => 2
| (n+2)   => if (n % 2 = 0) then - (a (n+1))/2 else 2 - 2*(a (n+1))

noncomputable def S : ℕ → ℚ
| 0       => 0
| (n+1)   => S n + a (n+1)

theorem S_63_value : S 63 = 464 := by
  sorry

end S_63_value_l42_42265


namespace complement_intersection_l42_42547

-- Conditions
def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

-- Theorem statement (proof not included)
theorem complement_intersection :
  let C_UA : Set Int := U \ A
  (C_UA ∩ B) = {1, 2} := 
by
  sorry

end complement_intersection_l42_42547


namespace g_inv_undefined_at_1_l42_42935

noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x - 5)

noncomputable def g_inv (x : ℝ) : ℝ := (5 * x - 3) / (x - 1)

theorem g_inv_undefined_at_1 : ∀ x : ℝ, (g_inv x) = g_inv 1 → x = 1 :=
by
  intro x h
  sorry

end g_inv_undefined_at_1_l42_42935


namespace largest_circle_radius_l42_42125

theorem largest_circle_radius (a b c : ℝ) (h : a > b ∧ b > c) :
  ∃ radius : ℝ, radius = b :=
by
  sorry

end largest_circle_radius_l42_42125


namespace two_point_distribution_p_value_l42_42148

noncomputable def X : Type := ℕ -- discrete random variable (two-point)
def p (E_X2 : ℝ): ℝ := E_X2 -- p == E(X)

theorem two_point_distribution_p_value (var_X : ℝ) (E_X : ℝ) (E_X2 : ℝ) 
    (h1 : var_X = 2 / 9) 
    (h2 : E_X = p E_X2) 
    (h3 : E_X2 = E_X): 
    E_X = 1 / 3 ∨ E_X = 2 / 3 :=
by
  sorry

end two_point_distribution_p_value_l42_42148


namespace bouquet_carnations_l42_42176

def proportion_carnations (P : ℚ) (R : ℚ) (PC : ℚ) (RC : ℚ) : ℚ := PC + RC

theorem bouquet_carnations :
  let P := (7 / 10 : ℚ)
  let R := (3 / 10 : ℚ)
  let PC := (1 / 2) * P
  let RC := (2 / 3) * R
  let C := proportion_carnations P R PC RC
  (C * 100) = 55 :=
by
  sorry

end bouquet_carnations_l42_42176


namespace desired_average_sale_l42_42269

theorem desired_average_sale
  (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7991) :
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 7000 :=
by
  sorry

end desired_average_sale_l42_42269


namespace quadratic_has_two_distinct_real_roots_l42_42392

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := -(k + 3)
  let c := 2 * k + 1
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l42_42392


namespace Jill_age_l42_42894

variable (J R : ℕ) -- representing Jill's current age and Roger's current age

theorem Jill_age :
  (R = 2 * J + 5) →
  (R - J = 25) →
  J = 20 :=
by
  intros h1 h2
  sorry

end Jill_age_l42_42894


namespace right_triangle_median_square_l42_42381

theorem right_triangle_median_square (a b c k_a k_b : ℝ) :
  c = Real.sqrt (a^2 + b^2) → -- c is the hypotenuse
  k_a = Real.sqrt ((2 * b^2 + 2 * (a^2 + b^2) - a^2) / 4) → -- k_a is the median to side a
  k_b = Real.sqrt ((2 * a^2 + 2 * (a^2 + b^2) - b^2) / 4) → -- k_b is the median to side b
  c^2 = (4 / 5) * (k_a^2 + k_b^2) :=
by
  intros h_c h_ka h_kb
  sorry

end right_triangle_median_square_l42_42381


namespace ratio_of_weight_l42_42293

theorem ratio_of_weight (B : ℝ) : 
    (2 * (4 + B) = 16) → ((B = 4) ∧ (4 + B) / 2 = 4) := by
  intro h
  have h₁ : B = 4 := by
    linarith
  have h₂ : (4 + B) / 2 = 4 := by
    rw [h₁]
    norm_num
  exact ⟨h₁, h₂⟩

end ratio_of_weight_l42_42293


namespace additional_boys_went_down_slide_l42_42848

theorem additional_boys_went_down_slide (initial_boys total_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) : additional_boys = 13 :=
by {
    -- Proof body will be here
    sorry
}

end additional_boys_went_down_slide_l42_42848


namespace width_of_second_square_is_seven_l42_42040

-- The conditions translated into Lean definitions
def first_square : ℕ × ℕ := (8, 5)
def third_square : ℕ × ℕ := (5, 5)
def flag_dimensions : ℕ × ℕ := (15, 9)

-- The area calculation functions
def area (dim : ℕ × ℕ) : ℕ := dim.fst * dim.snd

-- Given areas for the first and third square
def area_first_square : ℕ := area first_square
def area_third_square : ℕ := area third_square

-- Desired flag area
def flag_area : ℕ := area flag_dimensions

-- Total area of first and third squares
def total_area_first_and_third : ℕ := area_first_square + area_third_square

-- Required area for the second square
def area_needed_second_square : ℕ := flag_area - total_area_first_and_third

-- Given length of the second square
def second_square_length : ℕ := 10

-- Solve for the width of the second square
def second_square_width : ℕ := area_needed_second_square / second_square_length

-- The proof goal
theorem width_of_second_square_is_seven : second_square_width = 7 := by
  sorry

end width_of_second_square_is_seven_l42_42040


namespace not_perfect_square_7_301_l42_42250

theorem not_perfect_square_7_301 :
  ¬ ∃ x : ℝ, x^2 = 7^301 := sorry

end not_perfect_square_7_301_l42_42250


namespace num_books_second_shop_l42_42150

-- Define the conditions
def num_books_first_shop : ℕ := 32
def cost_first_shop : ℕ := 1500
def cost_second_shop : ℕ := 340
def avg_price_per_book : ℕ := 20

-- Define the proof statement
theorem num_books_second_shop : 
  (num_books_first_shop + (cost_second_shop + cost_first_shop) / avg_price_per_book) - num_books_first_shop = 60 := by
  sorry

end num_books_second_shop_l42_42150


namespace inequality_solution_l42_42203

theorem inequality_solution (a : ℝ) (x : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 1 < a) 
  (y₁ : ℝ := a^(2 * x + 1)) 
  (y₂ : ℝ := a^(-3 * x)) :
  y₁ > y₂ → x > - (1 / 5) :=
by
  sorry

end inequality_solution_l42_42203


namespace pythagorean_triple_example_l42_42302

noncomputable def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  is_pythagorean_triple 5 12 13 :=
by
  sorry

end pythagorean_triple_example_l42_42302


namespace part_a_part_b_l42_42731

-- Define the cost variables for chocolates, popsicles, and lollipops
variables (C P L : ℕ)

-- Given conditions
axiom cost_relation1 : 3 * C = 2 * P
axiom cost_relation2 : 2 * L = 5 * C

-- Part (a): Prove that Mário can buy 5 popsicles with the money for 3 lollipops
theorem part_a : 
  (3 * L) / P = 5 :=
by sorry

-- Part (b): Prove that Mário can buy 11 chocolates with the money for 3 chocolates, 2 popsicles, and 2 lollipops combined
theorem part_b : 
  (3 * C + 2 * P + 2 * L) / C = 11 :=
by sorry

end part_a_part_b_l42_42731


namespace problem_statement_l42_42483

-- Define the operation * based on the given mathematical definition
def op (a b : ℕ) : ℤ := a * (a - b)

-- The core theorem to prove the expression in the problem
theorem problem_statement : op 2 3 + op (6 - 2) 4 = -2 :=
by
  -- This is where the proof would go, but it's omitted with sorry.
  sorry

end problem_statement_l42_42483


namespace magic_grid_product_l42_42053

theorem magic_grid_product (p q r s t x : ℕ) 
  (h1: p * 6 * 3 = q * r * s)
  (h2: p * q * t = 6 * r * 2)
  (h3: p * r * x = 6 * 2 * t)
  (h4: q * 2 * 3 = r * s * x)
  (h5: t * 2 * x = 6 * s * 3)
  (h6: 6 * q * 3 = r * s * t)
  (h7: p * r * s = 6 * 2 * q)
  : x = 36 := 
by
  sorry

end magic_grid_product_l42_42053


namespace carly_shipping_cost_l42_42596

noncomputable def total_shipping_cost (flat_fee cost_per_pound weight : ℝ) : ℝ :=
flat_fee + cost_per_pound * weight

theorem carly_shipping_cost : 
  total_shipping_cost 5 0.80 5 = 9 :=
by 
  unfold total_shipping_cost
  norm_num

end carly_shipping_cost_l42_42596


namespace earnings_per_puppy_l42_42100

def daily_pay : ℝ := 40
def total_earnings : ℝ := 76
def num_puppies : ℕ := 16

theorem earnings_per_puppy : (total_earnings - daily_pay) / num_puppies = 2.25 := by
  sorry

end earnings_per_puppy_l42_42100


namespace quadratic_has_two_distinct_real_roots_l42_42892

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l42_42892


namespace math_proof_problem_l42_42010

open Set

noncomputable def alpha : ℝ := (3 - Real.sqrt 5) / 2

theorem math_proof_problem (α_pos : 0 < α) (α_lt_delta : α < alpha) :
  ∃ n p : ℕ, p > α * 2^n ∧ ∃ S T : Finset (Fin n) → Finset (Fin n), (∀ i j, (S i) ∩ (T j) ≠ ∅) :=
  sorry

end math_proof_problem_l42_42010


namespace tan_half_angle_inequality_l42_42932

theorem tan_half_angle_inequality (a b c : ℝ) (α β : ℝ)
  (h : a + b < 3 * c)
  (h_tan_identity : Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c)) :
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by
  sorry

end tan_half_angle_inequality_l42_42932


namespace scientific_notation_of_74850000_l42_42740

theorem scientific_notation_of_74850000 : 74850000 = 7.485 * 10^7 :=
  by
  sorry

end scientific_notation_of_74850000_l42_42740


namespace base_of_second_fraction_l42_42763

theorem base_of_second_fraction (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 :=
by
  sorry

end base_of_second_fraction_l42_42763


namespace total_tweets_is_correct_l42_42914

-- Define the conditions of Polly's tweeting behavior and durations
def happy_tweets := 18
def hungry_tweets := 4
def mirror_tweets := 45
def duration := 20

-- Define the total tweets calculation
def total_tweets := duration * happy_tweets + duration * hungry_tweets + duration * mirror_tweets

-- Prove that the total number of tweets is 1340
theorem total_tweets_is_correct : total_tweets = 1340 := by
  sorry

end total_tweets_is_correct_l42_42914


namespace probability_of_rolling_four_threes_l42_42430
open BigOperators

def probability_four_threes (n : ℕ) (k : ℕ) (p : ℚ) (q : ℚ) : ℚ := 
  (n.choose k) * (p ^ k) * (q ^ (n - k))

theorem probability_of_rolling_four_threes : 
  probability_four_threes 5 4 (1 / 10) (9 / 10) = 9 / 20000 := 
by 
  sorry

end probability_of_rolling_four_threes_l42_42430


namespace trisha_take_home_pay_l42_42499

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l42_42499


namespace arithmetic_mean_of_1_and_4_l42_42143

theorem arithmetic_mean_of_1_and_4 : 
  (1 + 4) / 2 = 5 / 2 := by
  sorry

end arithmetic_mean_of_1_and_4_l42_42143


namespace roots_depend_on_k_l42_42748

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem roots_depend_on_k (k : ℝ) :
  let a := 1
  let b := -3
  let c := 2 - k
  discriminant a b c = 1 + 4 * k :=
by
  sorry

end roots_depend_on_k_l42_42748


namespace sum_of_angles_l42_42595

theorem sum_of_angles : 
    ∀ (angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC : ℝ),
    angle1 + angle3 + angle5 = 180 ∧
    angle2 + angle4 + angle6 = 180 ∧
    angleA + angleB + angleC = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 :=
by
  intro angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC
  intro h
  sorry

end sum_of_angles_l42_42595


namespace machine_C_time_l42_42144

theorem machine_C_time (T_c : ℝ) : 
  (1/4) + (1/3) + (1/T_c) = (3/4) → T_c = 6 := 
by 
  sorry

end machine_C_time_l42_42144


namespace melissa_remaining_bananas_l42_42151

theorem melissa_remaining_bananas :
  let initial_bananas := 88
  let shared_bananas := 4
  initial_bananas - shared_bananas = 84 :=
by
  sorry

end melissa_remaining_bananas_l42_42151


namespace inequality_solution_l42_42877

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end inequality_solution_l42_42877


namespace problem1_problem2_l42_42758

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end problem1_problem2_l42_42758


namespace machine_c_more_bottles_l42_42020

theorem machine_c_more_bottles (A B C : ℕ) 
  (hA : A = 12)
  (hB : B = A - 2)
  (h_total : 10 * A + 10 * B + 10 * C = 370) :
  C - B = 5 :=
by
  sorry

end machine_c_more_bottles_l42_42020


namespace additional_charge_per_international_letter_l42_42616

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end additional_charge_per_international_letter_l42_42616


namespace stadium_length_in_feet_l42_42286

-- Assume the length of the stadium is 80 yards
def stadium_length_yards := 80

-- Assume the conversion factor is 3 feet per yard
def conversion_factor := 3

-- The length in feet is the product of the length in yards and the conversion factor
def length_in_feet := stadium_length_yards * conversion_factor

-- We want to prove that this length in feet is 240 feet
theorem stadium_length_in_feet : length_in_feet = 240 := by
  -- Definitions and conditions are directly restated here; the proof is sketched as 'sorry'
  sorry

end stadium_length_in_feet_l42_42286


namespace Rajesh_work_completion_time_l42_42838

-- Definitions based on conditions in a)
def Mahesh_rate := 1 / 60 -- Mahesh's rate of work (work per day)
def Mahesh_work := 20 * Mahesh_rate -- Work completed by Mahesh in 20 days
def Rajesh_time_to_complete_remaining_work := 30 -- Rajesh time to complete remaining work (days)
def Remaining_work := 1 - Mahesh_work -- Remaining work after Mahesh's contribution

-- Statement that needs to be proved
theorem Rajesh_work_completion_time :
  (Rajesh_time_to_complete_remaining_work : ℝ) * (1 / Remaining_work) = 45 :=
sorry

end Rajesh_work_completion_time_l42_42838


namespace simplify_fractions_l42_42098

theorem simplify_fractions :
  (30 / 45) * (75 / 128) * (256 / 150) = 1 / 6 := 
by
  sorry

end simplify_fractions_l42_42098


namespace min_soldiers_in_square_formations_l42_42512

theorem min_soldiers_in_square_formations : ∃ (a : ℕ), 
  ∃ (k : ℕ), 
    (a = k^2 ∧ 
    11 * a + 1 = (m : ℕ) ^ 2) ∧ 
    (∀ (b : ℕ), 
      (∃ (j : ℕ), b = j^2 ∧ 11 * b + 1 = (n : ℕ) ^ 2) → a ≤ b) ∧ 
    a = 9 := 
sorry

end min_soldiers_in_square_formations_l42_42512


namespace find_g2_l42_42796

-- Given conditions:
variables (g : ℝ → ℝ) 
axiom cond1 : ∀ (x y : ℝ), x * g y = 2 * y * g x
axiom cond2 : g 10 = 5

-- Proof to show g(2) = 2
theorem find_g2 : g 2 = 2 := 
by
  -- Skipping the actual proof
  sorry

end find_g2_l42_42796


namespace range_of_y_div_x_l42_42934

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + y^2 + 4*x + 3 = 0) :
  - (Real.sqrt 3) / 3 <= y / x ∧ y / x <= (Real.sqrt 3) / 3 :=
sorry

end range_of_y_div_x_l42_42934


namespace rancher_loss_l42_42018

-- Define the necessary conditions
def initial_head_of_cattle := 340
def original_total_price := 204000
def cattle_died := 172
def price_reduction_per_head := 150

-- Define the original and new prices per head
def original_price_per_head := original_total_price / initial_head_of_cattle
def new_price_per_head := original_price_per_head - price_reduction_per_head

-- Define the number of remaining cattle
def remaining_cattle := initial_head_of_cattle - cattle_died

-- Define the total amount at the new price
def total_amount_new_price := new_price_per_head * remaining_cattle

-- Define the loss
def loss := original_total_price - total_amount_new_price

-- Prove that the loss is $128,400
theorem rancher_loss : loss = 128400 := by
  sorry

end rancher_loss_l42_42018


namespace candidate_percentage_l42_42252

variables (P candidate_votes rival_votes total_votes : ℝ)

-- Conditions
def candidate_lost_by_2460 (candidate_votes rival_votes : ℝ) : Prop :=
  rival_votes = candidate_votes + 2460

def total_votes_cast (candidate_votes rival_votes total_votes : ℝ) : Prop :=
  candidate_votes + rival_votes = total_votes

-- Proof problem
theorem candidate_percentage (h1 : candidate_lost_by_2460 candidate_votes rival_votes)
                             (h2 : total_votes_cast candidate_votes rival_votes 8200) :
  P = 35 :=
sorry

end candidate_percentage_l42_42252


namespace negation_of_exisential_inequality_l42_42045

open Classical

theorem negation_of_exisential_inequality :
  ¬ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 := 
by 
sorry

end negation_of_exisential_inequality_l42_42045


namespace sixth_graders_l42_42624

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end sixth_graders_l42_42624


namespace complex_ratio_real_l42_42879

theorem complex_ratio_real (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ z : ℂ, z = a + b * Complex.I ∧ (z * (1 - 2 * Complex.I)).im = 0) :
  a / b = 1 / 2 :=
sorry

end complex_ratio_real_l42_42879


namespace men_in_first_group_l42_42275

theorem men_in_first_group (M : ℕ) (h1 : M * 35 = 7 * 50) : M = 10 := by
  sorry

end men_in_first_group_l42_42275


namespace average_salary_l42_42587

theorem average_salary (a b c d e : ℕ) (h1 : a = 8000) (h2 : b = 5000) (h3 : c = 16000) (h4 : d = 7000) (h5 : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by
  sorry

end average_salary_l42_42587


namespace total_value_of_horse_and_saddle_l42_42332

def saddle_value : ℝ := 12.5
def horse_value : ℝ := 7 * saddle_value

theorem total_value_of_horse_and_saddle : horse_value + saddle_value = 100 := by
  sorry

end total_value_of_horse_and_saddle_l42_42332


namespace tan_frac_a_pi_six_eq_sqrt_three_l42_42841

theorem tan_frac_a_pi_six_eq_sqrt_three (a : ℝ) (h : (a, 9) ∈ { p : ℝ × ℝ | p.2 = 3 ^ p.1 }) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := 
by
  sorry

end tan_frac_a_pi_six_eq_sqrt_three_l42_42841


namespace part_one_part_two_l42_42350

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 - (a + 1/a) * x + 1

theorem part_one (x : ℝ) : f x (1/2) ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by
  sorry

theorem part_two (x a : ℝ) (h : a > 0) : 
  ((a < 1) → (f x a ≤ 0 ↔ (a ≤ x ∧ x ≤ 1/a))) ∧
  ((a > 1) → (f x a ≤ 0 ↔ (1/a ≤ x ∧ x ≤ a))) ∧
  ((a = 1) → (f x a ≤ 0 ↔ (x = 1))) :=
by
  sorry

end part_one_part_two_l42_42350


namespace complex_division_result_l42_42360

theorem complex_division_result :
  let z := (⟨0, 1⟩ - ⟨2, 0⟩) / (⟨1, 0⟩ + ⟨0, 1⟩ : ℂ)
  let a := z.re
  let b := z.im
  a + b = 1 :=
by
  sorry

end complex_division_result_l42_42360


namespace min_output_to_avoid_losses_l42_42396

theorem min_output_to_avoid_losses (x : ℝ) (y : ℝ) (h : y = 0.1 * x - 150) : y ≥ 0 → x ≥ 1500 :=
sorry

end min_output_to_avoid_losses_l42_42396


namespace book_arrangement_l42_42842

theorem book_arrangement :
  let total_books := 6
  let identical_books := 3
  let unique_arrangements := Nat.factorial total_books / Nat.factorial identical_books
  unique_arrangements = 120 := by
  sorry

end book_arrangement_l42_42842


namespace problem1_problem2_l42_42170

theorem problem1 (a b c : ℝ) (h1 : a = 5.42) (h2 : b = 3.75) (h3 : c = 0.58) :
  a - (b - c) = 2.25 :=
by sorry

theorem problem2 (d e f g h : ℝ) (h4 : d = 4 / 5) (h5 : e = 7.7) (h6 : f = 0.8) (h7 : g = 3.3) (h8 : h = 1) :
  d * e + f * g - d = 8 :=
by sorry

end problem1_problem2_l42_42170


namespace workout_days_l42_42127

theorem workout_days (n : ℕ) (squats : ℕ → ℕ) 
  (h1 : squats 1 = 30)
  (h2 : ∀ k, squats (k + 1) = squats k + 5)
  (h3 : squats 4 = 45) :
  n = 4 :=
sorry

end workout_days_l42_42127


namespace train_length_is_330_meters_l42_42072

noncomputable def train_speed : ℝ := 60 -- in km/hr
noncomputable def man_speed : ℝ := 6    -- in km/hr
noncomputable def time : ℝ := 17.998560115190788  -- in seconds

noncomputable def relative_speed_km_per_hr : ℝ := train_speed + man_speed
noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def relative_speed_m_per_s : ℝ := 
  relative_speed_km_per_hr * conversion_factor

theorem train_length_is_330_meters : 
  (relative_speed_m_per_s * time) = 330 := 
sorry

end train_length_is_330_meters_l42_42072


namespace probability_both_groups_stop_same_round_l42_42674

noncomputable def probability_same_round : ℚ :=
  let probability_fair_coin_stop (n : ℕ) : ℚ := (1/2)^n
  let probability_biased_coin_stop (n : ℕ) : ℚ := (2/3)^(n-1) * (1/3)
  let probability_fair_coin_group_stop (n : ℕ) : ℚ := (probability_fair_coin_stop n)^3
  let probability_biased_coin_group_stop (n : ℕ) : ℚ := (probability_biased_coin_stop n)^3
  let combined_round_probability (n : ℕ) : ℚ := 
    probability_fair_coin_group_stop n * probability_biased_coin_group_stop n
  let total_probability : ℚ := ∑' n, combined_round_probability n
  total_probability

theorem probability_both_groups_stop_same_round :
  probability_same_round = 1 / 702 := by sorry

end probability_both_groups_stop_same_round_l42_42674


namespace morning_snowfall_l42_42660

theorem morning_snowfall (total_snowfall afternoon_snowfall morning_snowfall : ℝ) 
  (h1 : total_snowfall = 0.625) 
  (h2 : afternoon_snowfall = 0.5) 
  (h3 : total_snowfall = morning_snowfall + afternoon_snowfall) : 
  morning_snowfall = 0.125 :=
by
  sorry

end morning_snowfall_l42_42660


namespace pizza_problem_l42_42219

theorem pizza_problem
  (pizza_slices : ℕ)
  (total_pizzas : ℕ)
  (total_people : ℕ)
  (pepperoni_only_friend : ℕ)
  (remaining_pepperoni : ℕ)
  (equal_distribution : Prop)
  (h_cond1 : pizza_slices = 16)
  (h_cond2 : total_pizzas = 2)
  (h_cond3 : total_people = 4)
  (h_cond4 : pepperoni_only_friend = 1)
  (h_cond5 : remaining_pepperoni = 1)
  (h_cond6 : equal_distribution ∧ (pepperoni_only_friend ≤ total_people)) :
  ∃ cheese_slices_left : ℕ, cheese_slices_left = 7 := by
  sorry

end pizza_problem_l42_42219


namespace complex_solution_l42_42868

open Complex

theorem complex_solution (z : ℂ) (h : z + Complex.abs z = 1 + Complex.I) : z = Complex.I := 
by
  sorry

end complex_solution_l42_42868


namespace find_ordered_pair_l42_42924

theorem find_ordered_pair (s l : ℝ) :
  (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) →
  (s = -19 ∧ l = -7 / 2) :=
by
  intro h
  have : (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) := h
  sorry

end find_ordered_pair_l42_42924


namespace infinitely_many_primes_satisfying_condition_l42_42607

theorem infinitely_many_primes_satisfying_condition :
  ∀ k : Nat, ∃ p : Nat, Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014) := 
sorry

end infinitely_many_primes_satisfying_condition_l42_42607


namespace elmo_to_laura_books_ratio_l42_42502

-- Definitions of the conditions given in the problem
def ElmoBooks : ℕ := 24
def StuBooks : ℕ := 4
def LauraBooks : ℕ := 2 * StuBooks

-- Ratio calculation and proof of the ratio being 3:1
theorem elmo_to_laura_books_ratio : (ElmoBooks : ℚ) / (LauraBooks : ℚ) = 3 / 1 := by
  sorry

end elmo_to_laura_books_ratio_l42_42502


namespace rainfall_difference_l42_42172

-- Define the conditions
def day1_rainfall := 26
def day2_rainfall := 34
def average_rainfall := 140
def less_rainfall := 58

-- Calculate the total rainfall this year in the first three days
def total_rainfall_this_year := average_rainfall - less_rainfall

-- Calculate the total rainfall in the first two days
def total_first_two_days := day1_rainfall + day2_rainfall

-- Calculate the rainfall on the third day
def day3_rainfall := total_rainfall_this_year - total_first_two_days

-- The proof problem
theorem rainfall_difference : day2_rainfall - day3_rainfall = 12 := 
by
  sorry

end rainfall_difference_l42_42172


namespace chess_tournament_games_l42_42644

def stage1_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage2_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage3_games : ℕ := 4

def total_games (stage1 stage2 stage3 : ℕ) : ℕ := stage1 + stage2 + stage3

theorem chess_tournament_games : total_games (stage1_games 20) (stage2_games 10) stage3_games = 474 :=
by
  unfold stage1_games
  unfold stage2_games
  unfold total_games
  simp
  sorry

end chess_tournament_games_l42_42644


namespace sqrt_expression_eq_two_l42_42447

theorem sqrt_expression_eq_two : 
  (Real.sqrt 3) * (Real.sqrt 3 - 1 / (Real.sqrt 3)) = 2 := 
  sorry

end sqrt_expression_eq_two_l42_42447


namespace beta_cannot_be_determined_l42_42816

variables (α β : ℝ)
def consecutive_interior_angles (α β : ℝ) : Prop := -- define what it means for angles to be consecutive interior angles
  α + β = 180  -- this is true for interior angles, for illustrative purposes.

theorem beta_cannot_be_determined
  (h1 : consecutive_interior_angles α β)
  (h2 : α = 55) :
  ¬(∃ β, β = α) :=
by
  sorry

end beta_cannot_be_determined_l42_42816


namespace b_plus_c_neg_seven_l42_42545

theorem b_plus_c_neg_seven {A B : Set ℝ} (hA : A = {x : ℝ | x > 3 ∨ x < -1}) (hB : B = {x : ℝ | -1 ≤ x ∧ x ≤ 4})
  (h_union : A ∪ B = Set.univ) (h_inter : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 4}) :
  ∃ b c : ℝ, (∀ x, x^2 + b * x + c ≤ 0 ↔ x ∈ B) ∧ b + c = -7 :=
by
  sorry

end b_plus_c_neg_seven_l42_42545


namespace high_school_students_total_l42_42178

theorem high_school_students_total
    (students_taking_music : ℕ)
    (students_taking_art : ℕ)
    (students_taking_both_music_and_art : ℕ)
    (students_taking_neither : ℕ)
    (h1 : students_taking_music = 50)
    (h2 : students_taking_art = 20)
    (h3 : students_taking_both_music_and_art = 10)
    (h4 : students_taking_neither = 440) :
    students_taking_music - students_taking_both_music_and_art + students_taking_art - students_taking_both_music_and_art + students_taking_both_music_and_art + students_taking_neither = 500 :=
by
  sorry

end high_school_students_total_l42_42178


namespace common_difference_of_arithmetic_sequence_l42_42708

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ) -- define the arithmetic sequence
  (h_arith : ∀ n : ℕ, a n = a 0 + n * 4) -- condition of arithmetic sequence
  (h_a5 : a 4 = 8) -- given a_5 = 8
  (h_a9 : a 8 = 24) -- given a_9 = 24
  : 4 = 4 := -- statement to be proven
by
  sorry

end common_difference_of_arithmetic_sequence_l42_42708


namespace find_parabola_focus_l42_42583

theorem find_parabola_focus : 
  ∀ (x y : ℝ), (y = 2 * x ^ 2 + 4 * x - 1) → (∃ p q : ℝ, p = -1 ∧ q = -(23:ℝ) / 8 ∧ (y = 2 * x ^ 2 + 4 * x - 1) → (x, y) = (p, q)) :=
by
  sorry

end find_parabola_focus_l42_42583


namespace smallest_a_l42_42179

theorem smallest_a (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) : a = 17 :=
sorry

end smallest_a_l42_42179


namespace draw_3_odd_balls_from_15_is_336_l42_42830

-- Define the problem setting as given in the conditions
def odd_balls : Finset ℕ := {1, 3, 5, 7, 9, 11, 13, 15}

-- Define the function that calculates the number of ways to draw 3 balls
noncomputable def draw_3_odd_balls (S : Finset ℕ) : ℕ :=
  S.card * (S.card - 1) * (S.card - 2)

-- Prove that the drawing of 3 balls results in 336 ways
theorem draw_3_odd_balls_from_15_is_336 : draw_3_odd_balls odd_balls = 336 := by
  sorry

end draw_3_odd_balls_from_15_is_336_l42_42830


namespace fraction_of_capital_subscribed_l42_42428

theorem fraction_of_capital_subscribed (T : ℝ) (x : ℝ) :
  let B_capital := (1 / 4) * T
  let C_capital := (1 / 5) * T
  let Total_profit := 2445
  let A_profit := 815
  A_profit / Total_profit = x → x = 1 / 3 :=
by
  sorry

end fraction_of_capital_subscribed_l42_42428


namespace problem_solution_l42_42062

theorem problem_solution :
  (3012 - 2933)^2 / 196 = 32 := sorry

end problem_solution_l42_42062


namespace smallest_n_for_sum_is_24_l42_42705

theorem smallest_n_for_sum_is_24 :
  ∃ (n : ℕ), (0 < n) ∧ 
    (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
    ∀ (m : ℕ), ((0 < m) ∧ 
                (∃ (k' : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = k') → n ≤ m) := sorry

end smallest_n_for_sum_is_24_l42_42705


namespace counts_of_arson_l42_42140

-- Define variables A (arson), B (burglary), L (petty larceny)
variables (A B L : ℕ)

-- Conditions given in the problem
def burglary_charges : Prop := B = 2
def petty_larceny_charges_relation : Prop := L = 6 * B
def total_sentence_calculation : Prop := 36 * A + 18 * B + 6 * L = 216

-- Prove that given these conditions, the counts of arson (A) is 3
theorem counts_of_arson (h1 : burglary_charges B)
                        (h2 : petty_larceny_charges_relation B L)
                        (h3 : total_sentence_calculation A B L) :
                        A = 3 :=
sorry

end counts_of_arson_l42_42140


namespace trajectory_of_Q_l42_42379

/-- Let P(m, n) be a point moving on the circle x^2 + y^2 = 2.
     The trajectory of the point Q(m+n, 2mn) is y = x^2 - 2. -/
theorem trajectory_of_Q (m n : ℝ) (hyp : m^2 + n^2 = 2) : 
  ∃ x y : ℝ, x = m + n ∧ y = 2 * m * n ∧ y = x^2 - 2 :=
by
  sorry

end trajectory_of_Q_l42_42379


namespace percentage_of_students_liking_chess_l42_42319

theorem percentage_of_students_liking_chess (total_students : ℕ) (basketball_percentage : ℝ) (soccer_percentage : ℝ) 
(identified_chess_or_basketball : ℕ) (students_liking_basketball : ℕ) : 
total_students = 250 ∧ basketball_percentage = 0.40 ∧ soccer_percentage = 0.28 ∧ identified_chess_or_basketball = 125 ∧ 
students_liking_basketball = 100 → ∃ C : ℝ, C = 0.10 :=
by
  sorry

end percentage_of_students_liking_chess_l42_42319


namespace yellow_balls_count_l42_42292

-- Definition of problem conditions
def initial_red_balls : ℕ := 16
def initial_blue_balls : ℕ := 2 * initial_red_balls
def red_balls_lost : ℕ := 6
def green_balls_given_away : ℕ := 7  -- This is not used in the calculations
def yellow_balls_bought : ℕ := 3 * red_balls_lost
def final_total_balls : ℕ := 74

-- Defining the total balls after all transactions
def remaining_red_balls : ℕ := initial_red_balls - red_balls_lost
def total_accounted_balls : ℕ := remaining_red_balls + initial_blue_balls + yellow_balls_bought

-- Lean statement to prove
theorem yellow_balls_count : yellow_balls_bought = 18 :=
by
  sorry

end yellow_balls_count_l42_42292


namespace excursion_min_parents_l42_42568

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end excursion_min_parents_l42_42568


namespace simplify_expression_l42_42630

def a : ℚ := (3 / 4) * 60
def b : ℚ := (8 / 5) * 60
def c : ℚ := 63

theorem simplify_expression : a - b + c = 12 := by
  sorry

end simplify_expression_l42_42630


namespace equal_distribution_l42_42725

theorem equal_distribution (k : ℤ) : ∃ n : ℤ, n = 81 + 95 * k ∧ ∃ b : ℤ, (19 + 6 * n) = 95 * b :=
by
  -- to be proved
  sorry

end equal_distribution_l42_42725


namespace johnny_red_pencils_l42_42638

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l42_42638


namespace meet_at_centroid_l42_42414

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end meet_at_centroid_l42_42414


namespace xiao_wang_conjecture_incorrect_l42_42994

theorem xiao_wang_conjecture_incorrect : ∃ n : ℕ, n > 0 ∧ (n^2 - 8 * n + 7 > 0) := by
  sorry

end xiao_wang_conjecture_incorrect_l42_42994


namespace area_equivalence_l42_42647

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Point := sorry
noncomputable def arc_midpoint (A B C : Point) : Point := sorry
noncomputable def is_concyclic (P Q R S : Point) : Prop := sorry
noncomputable def area_of_quad (A B C D : Point) : ℝ := sorry
noncomputable def area_of_pent (A B C D E : Point) : ℝ := sorry

theorem area_equivalence (A B C I X Y M : Point)
  (h1 : I = incenter A B C)
  (h2 : X = angle_bisector B A C)
  (h3 : Y = angle_bisector C A B)
  (h4 : M = arc_midpoint A B C)
  (h5 : is_concyclic M X I Y) :
  area_of_quad M B I C = area_of_pent B X I Y C := 
sorry

end area_equivalence_l42_42647


namespace point_P_distance_to_y_axis_l42_42922

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- The distance from point P to the y-axis
def distance_to_y_axis (pt : ℝ × ℝ) : ℝ :=
  abs pt.1

-- Statement to prove
theorem point_P_distance_to_y_axis :
  distance_to_y_axis point_P = 2 :=
by
  sorry

end point_P_distance_to_y_axis_l42_42922


namespace largest_common_term_l42_42220

/-- The arithmetic progression sequence1 --/
def sequence1 (n : ℕ) : ℤ := 4 + 5 * n

/-- The arithmetic progression sequence2 --/
def sequence2 (n : ℕ) : ℤ := 5 + 8 * n

/-- The common term condition for sequence1 --/
def common_term_condition1 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence1 n

/-- The common term condition for sequence2 --/
def common_term_condition2 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence2 n

/-- The largest common term less than 1000 --/
def is_largest_common_term (a : ℤ) : Prop :=
  common_term_condition1 a ∧ common_term_condition2 a ∧ a < 1000 ∧
  ∀ b : ℤ, common_term_condition1 b ∧ common_term_condition2 b ∧ b < 1000 → b ≤ a

/-- Lean theorem statement --/
theorem largest_common_term :
  ∃ a : ℤ, is_largest_common_term a ∧ a = 989 :=
sorry

end largest_common_term_l42_42220


namespace percent_difference_l42_42466

theorem percent_difference (a b : ℝ) : 
  a = 67.5 * 250 / 100 → 
  b = 52.3 * 180 / 100 → 
  (a - b) = 74.61 :=
by
  intros ha hb
  rw [ha, hb]
  -- omitted proof
  sorry

end percent_difference_l42_42466


namespace denomination_other_currency_notes_l42_42750

noncomputable def denomination_proof : Prop :=
  ∃ D x y : ℕ, 
  (x + y = 85) ∧
  (100 * x + D * y = 5000) ∧
  (D * y = 3500) ∧
  (D = 50)

theorem denomination_other_currency_notes :
  denomination_proof :=
sorry

end denomination_other_currency_notes_l42_42750


namespace class_mean_l42_42085

theorem class_mean
  (num_students_1 : ℕ)
  (num_students_2 : ℕ)
  (total_students : ℕ)
  (mean_score_1 : ℚ)
  (mean_score_2 : ℚ)
  (new_mean_score : ℚ)
  (h1 : num_students_1 + num_students_2 = total_students)
  (h2 : total_students = 30)
  (h3 : num_students_1 = 24)
  (h4 : mean_score_1 = 80)
  (h5 : num_students_2 = 6)
  (h6 : mean_score_2 = 85) :
  new_mean_score = 81 :=
by
  sorry

end class_mean_l42_42085


namespace sum_of_purchases_l42_42324

variable (J : ℕ) (K : ℕ)

theorem sum_of_purchases :
  J = 230 →
  2 * J = K + 90 →
  J + K = 600 :=
by
  intros hJ hEq
  rw [hJ] at hEq
  sorry

end sum_of_purchases_l42_42324


namespace area_of_parallelogram_l42_42880

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 :=
by
  rw [h_base, h_height]
  norm_num

end area_of_parallelogram_l42_42880


namespace part1_part2_1_part2_2_l42_42417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x > 0 → (2 * a * x - Real.log x - 1) ≥ 0) ↔ a ≥ 0.5 := 
sorry

theorem part2_1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 = x1 ∧ f a x2 = x2) :
  0 < a ∧ a < 1 := 
sorry

theorem part2_2 (a x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f a x1 = x1) (h3 : f a x2 = x2) (h4 : x2 ≥ 3 * x1) :
  x1 * x2 ≥ 9 / Real.exp 2 := 
sorry

end part1_part2_1_part2_2_l42_42417


namespace line_intercepts_l42_42580

theorem line_intercepts :
  (exists a b : ℝ, (forall x y : ℝ, x - 2*y - 2 = 0 ↔ (x = 2 ∨ y = -1)) ∧ a = 2 ∧ b = -1) :=
by
  sorry

end line_intercepts_l42_42580


namespace count_multiples_4_or_5_not_20_l42_42095

-- We define the necessary ranges and conditions
def is_multiple_of (n k : ℕ) := n % k = 0

def count_multiples (n k : ℕ) := (n / k)

def not_multiple_of (n k : ℕ) := ¬ is_multiple_of n k

def count_multiples_excluding (n k l : ℕ) :=
  count_multiples n k + count_multiples n l - count_multiples n (Nat.lcm k l)

theorem count_multiples_4_or_5_not_20 : count_multiples_excluding 3010 4 5 = 1204 := 
by
  sorry

end count_multiples_4_or_5_not_20_l42_42095


namespace percentage_difference_l42_42157

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.70 * x) (h2 : z = 1.50 * y) :
   x / z = 39.22 / 100 :=
by
  sorry

end percentage_difference_l42_42157


namespace arithmetic_problem_l42_42089

theorem arithmetic_problem : 1357 + 3571 + 5713 - 7135 = 3506 :=
by
  sorry

end arithmetic_problem_l42_42089


namespace total_selling_price_correct_l42_42093

def cost_price_1 := 750
def cost_price_2 := 1200
def cost_price_3 := 500

def loss_percent_1 := 10
def loss_percent_2 := 15
def loss_percent_3 := 5

noncomputable def selling_price_1 := cost_price_1 - ((loss_percent_1 / 100) * cost_price_1)
noncomputable def selling_price_2 := cost_price_2 - ((loss_percent_2 / 100) * cost_price_2)
noncomputable def selling_price_3 := cost_price_3 - ((loss_percent_3 / 100) * cost_price_3)

noncomputable def total_selling_price := selling_price_1 + selling_price_2 + selling_price_3

theorem total_selling_price_correct : total_selling_price = 2170 := by
  sorry

end total_selling_price_correct_l42_42093


namespace QED_mul_eq_neg_25I_l42_42946

namespace ComplexMultiplication

open Complex

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := -Complex.I
def D : ℂ := 3 - 4 * Complex.I

theorem QED_mul_eq_neg_25I : Q * E * D = -25 * Complex.I :=
by
  sorry

end ComplexMultiplication

end QED_mul_eq_neg_25I_l42_42946


namespace handshake_problem_l42_42177

theorem handshake_problem (x : ℕ) (hx : (x * (x - 1)) / 2 = 55) : x = 11 := 
sorry

end handshake_problem_l42_42177


namespace find_x_l42_42574

theorem find_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) : 
  x = (36 * Real.sqrt 5)^(4/11) := 
sorry

end find_x_l42_42574


namespace milo_skateboarding_speed_l42_42711

theorem milo_skateboarding_speed (cory_speed milo_skateboarding_speed : ℝ) 
  (h1 : cory_speed = 12) 
  (h2 : cory_speed = 2 * milo_skateboarding_speed) : 
  milo_skateboarding_speed = 6 :=
by sorry

end milo_skateboarding_speed_l42_42711


namespace expression_D_is_odd_l42_42008

namespace ProofProblem

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem expression_D_is_odd :
  is_odd (3 + 5 + 1) :=
by
  sorry

end ProofProblem

end expression_D_is_odd_l42_42008


namespace polynomial_coefficients_even_or_odd_l42_42307

-- Define the problem conditions as Lean definitions
variables {P Q : Polynomial ℤ}

-- Theorem: Given the conditions, prove the required statement
theorem polynomial_coefficients_even_or_odd
  (hP : ∀ n : ℕ, P.coeff n % 2 = 0)
  (hQ : ∀ n : ℕ, Q.coeff n % 2 = 0)
  (hProd : ¬ ∀ n : ℕ, (P * Q).coeff n % 4 = 0) :
  (∀ n : ℕ, P.coeff n % 2 = 0 ∧ ∃ k : ℕ, Q.coeff k % 2 ≠ 0) ∨
  (∀ n : ℕ, Q.coeff n % 2 = 0 ∧ ∃ k: ℕ, P.coeff k % 2 ≠ 0) :=
sorry

end polynomial_coefficients_even_or_odd_l42_42307


namespace john_ate_10_chips_l42_42528

variable (c p : ℕ)

/-- Given the total calories from potato chips and the calories increment of cheezits,
prove the number of potato chips John ate. -/
theorem john_ate_10_chips (h₀ : p * c = 60)
  (h₁ : ∃ c_cheezit, (c_cheezit = (4 / 3 : ℝ) * c))
  (h₂ : ∀ c_cheezit, p * c + 6 * c_cheezit = 108) :
  p = 10 :=
by {
  sorry
}

end john_ate_10_chips_l42_42528


namespace circle_diameter_l42_42356

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ d : ℝ, d = 16 :=
by
  sorry

end circle_diameter_l42_42356


namespace running_race_l42_42348

-- Define participants
inductive Participant : Type
| Anna
| Bella
| Csilla
| Dora

open Participant

-- Define positions
@[ext] structure Position :=
(first : Participant)
(last : Participant)

-- Conditions:
def conditions (p : Participant) (q : Participant) (r : Participant) (s : Participant)
  (pa : Position) : Prop :=
  (pa.first = r) ∧ -- Csilla was first
  (pa.first ≠ q) ∧ -- Bella was not first
  (pa.first ≠ p) ∧ (pa.last ≠ p) ∧ -- Anna was not first or last
  (pa.last = s) -- Dóra's statement about being last

-- Definition of the liar
def liar (p : Participant) : Prop :=
  p = Dora

-- Proof problem
theorem running_race : ∃ (pa : Position), liar Dora ∧ (pa.first = Csilla) :=
  sorry

end running_race_l42_42348


namespace sum_geometric_seq_eq_l42_42055

-- Defining the parameters of the geometric sequence
def a : ℚ := 1 / 5
def r : ℚ := 2 / 5
def n : ℕ := 8

-- Required to prove the sum of the first eight terms equals the given fraction
theorem sum_geometric_seq_eq :
  (a * (1 - r^n) / (1 - r)) = (390369 / 1171875) :=
by
  -- Proof to be completed
  sorry

end sum_geometric_seq_eq_l42_42055


namespace stable_table_configurations_l42_42948

noncomputable def numberOfStableConfigurations (n : ℕ) : ℕ :=
  1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)

theorem stable_table_configurations (n : ℕ) (hn : 0 < n) :
  numberOfStableConfigurations n = 
    (1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)) :=
by
  sorry

end stable_table_configurations_l42_42948


namespace mary_cut_roses_l42_42930

-- Definitions from conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- The theorem to prove
theorem mary_cut_roses : (final_roses - initial_roses) = 10 :=
by
  sorry

end mary_cut_roses_l42_42930


namespace expression_value_l42_42202

theorem expression_value (a : ℝ) (h_nonzero : a ≠ 0) (h_ne_two : a ≠ 2) (h_ne_neg_two : a ≠ -2) (h_ne_neg_one : a ≠ -1) (h_eq_one : a = 1) :
  1 - (((a-2)/a) / ((a^2-4)/(a^2+a))) = 1 / 3 :=
by
  sorry

end expression_value_l42_42202


namespace min_x2_y2_z2_l42_42913

theorem min_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 22 :=
by
  sorry

end min_x2_y2_z2_l42_42913


namespace recurrence_relation_solution_l42_42569

theorem recurrence_relation_solution (a : ℕ → ℕ) 
  (h_rec : ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2))
  (h0 : a 0 = 3)
  (h1 : a 1 = 5) :
  ∀ n, a n = 3^n + 2 :=
by
  sorry

end recurrence_relation_solution_l42_42569


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l42_42114

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l42_42114


namespace max_y_difference_intersection_l42_42683

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference_intersection :
  let x1 := 1
  let y1 := g x1
  let x2 := -1
  let y2 := g x2
  y1 - y2 = 2 :=
by
  sorry

end max_y_difference_intersection_l42_42683


namespace purely_imaginary_z_implies_m_zero_l42_42283

theorem purely_imaginary_z_implies_m_zero (m : ℝ) :
  m * (m + 1) = 0 → m ≠ -1 := by sorry

end purely_imaginary_z_implies_m_zero_l42_42283


namespace part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l42_42029

noncomputable def A : Set ℝ := { x : ℝ | 3 < x ∧ x < 10 }
noncomputable def B : Set ℝ := { x : ℝ | x^2 - 9 * x + 14 < 0 }
noncomputable def C (m : ℝ) : Set ℝ := { x : ℝ | 5 - m < x ∧ x < 2 * m }

theorem part_I_A_inter_B : A ∩ B = { x : ℝ | 3 < x ∧ x < 7 } :=
sorry

theorem part_I_complement_A_union_B :
  (Aᶜ) ∪ B = { x : ℝ | x < 7 ∨ x ≥ 10 } :=
sorry

theorem part_II_range_of_m :
  {m : ℝ | C m ⊆ A ∩ B} = {m : ℝ | m ≤ 2} :=
sorry

end part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l42_42029


namespace largest_perfect_square_factor_1760_l42_42199

theorem largest_perfect_square_factor_1760 :
  ∃ n, (∃ k, n = k^2) ∧ n ∣ 1760 ∧ ∀ m, (∃ j, m = j^2) ∧ m ∣ 1760 → m ≤ n := by
  sorry

end largest_perfect_square_factor_1760_l42_42199


namespace christel_gave_andrena_l42_42005

theorem christel_gave_andrena (d m c a: ℕ) (h1: d = 20 - 2) (h2: c = 24) 
  (h3: a = c + 2) (h4: a = d + 3) : (24 - c = 5) :=
by { sorry }

end christel_gave_andrena_l42_42005


namespace students_taking_history_l42_42133

-- Defining the conditions
def num_students (total_students history_students statistics_students both_students : ℕ) : Prop :=
  total_students = 89 ∧
  statistics_students = 32 ∧
  (history_students + statistics_students - both_students) = 59 ∧
  (history_students - both_students) = 27

-- The theorem stating that given the conditions, the number of students taking history is 54
theorem students_taking_history :
  ∃ history_students, ∃ statistics_students, ∃ both_students, 
  num_students 89 history_students statistics_students both_students ∧ history_students = 54 :=
by
  sorry

end students_taking_history_l42_42133


namespace jack_final_apples_l42_42039

-- Jack's transactions and initial count as conditions
def initial_count : ℕ := 150
def sold_to_jill : ℕ := initial_count * 30 / 100
def remaining_after_jill : ℕ := initial_count - sold_to_jill
def sold_to_june : ℕ := remaining_after_jill * 20 / 100
def remaining_after_june : ℕ := remaining_after_jill - sold_to_june
def donated_to_charity : ℕ := 5
def final_count : ℕ := remaining_after_june - donated_to_charity

-- Proof statement
theorem jack_final_apples : final_count = 79 := by
  sorry

end jack_final_apples_l42_42039


namespace tangent_line_of_ellipse_l42_42839

noncomputable def ellipse_tangent_line (a b x0 y0 x y : ℝ) : Prop :=
  x0 * x / a^2 + y0 * y / b^2 = 1

theorem tangent_line_of_ellipse
  (a b x0 y0 : ℝ)
  (h_ellipse : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_b : a > b) :
  ellipse_tangent_line a b x0 y0 x y :=
sorry

end tangent_line_of_ellipse_l42_42839


namespace max_value_is_zero_l42_42609

noncomputable def max_value (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : ℝ :=
  x^2 - y^2

theorem max_value_is_zero (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : max_value x y h = 0 :=
sorry

end max_value_is_zero_l42_42609


namespace contrapositive_mul_non_zero_l42_42533

variables (a b : ℝ)

theorem contrapositive_mul_non_zero (h : a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :
  (a = 0 ∨ b = 0) → a * b = 0 :=
by
  sorry

end contrapositive_mul_non_zero_l42_42533


namespace brother_raking_time_l42_42282

theorem brother_raking_time (x : ℝ) (hx : x > 0)
  (h_combined : (1 / 30) + (1 / x) = 1 / 18) : x = 45 :=
by
  sorry

end brother_raking_time_l42_42282


namespace tv_selection_l42_42384

theorem tv_selection (A B : ℕ) (hA : A = 4) (hB : B = 5) : 
  ∃ n, n = 3 ∧ (∃ k, k = 70 ∧ 
    (n = 1 ∧ k = A * (B * (B - 1) / 2) + A * (A - 1) / 2 * B)) :=
sorry

end tv_selection_l42_42384


namespace min_value_l42_42883

theorem min_value (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (h_sum : x1 + x2 = 1) :
  ∃ m, (∀ x1 x2, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = 1 → (3 * x1 / x2 + 1 / (x1 * x2)) ≥ m) ∧ m = 6 :=
by
  sorry

end min_value_l42_42883


namespace student_weekly_allowance_l42_42719

theorem student_weekly_allowance (A : ℝ) 
  (h1 : ∃ spent_arcade, spent_arcade = (3 / 5) * A)
  (h2 : ∃ spent_toy, spent_toy = (1 / 3) * ((2 / 5) * A))
  (h3 : ∃ spent_candy, spent_candy = 0.60)
  (h4 : ∃ remaining_after_toy, remaining_after_toy = ((6 / 15) * A - (2 / 15) * A))
  (h5 : remaining_after_toy = 0.60) : 
  A = 2.25 := by
  sorry

end student_weekly_allowance_l42_42719


namespace tens_digit_of_11_pow_12_pow_13_l42_42473

theorem tens_digit_of_11_pow_12_pow_13 :
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  tens_digit = 2 :=
by 
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  show tens_digit = 2
  sorry

end tens_digit_of_11_pow_12_pow_13_l42_42473


namespace common_root_values_max_n_and_a_range_l42_42339

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+1) * x - 4 * (a+5)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x + 5

-- Part 1
theorem common_root_values (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ g a x = 0) → a = -9/16 ∨ a = -6 ∨ a = -4 ∨ a = 0 :=
sorry

-- Part 2
theorem max_n_and_a_range (a : ℝ) (m n : ℕ) (x0 : ℝ) :
  (m < n ∧ (m : ℝ) < x0 ∧ x0 < (n : ℝ) ∧ f a x0 < 0 ∧ g a x0 < 0) →
  n = 4 ∧ -1 ≤ a ∧ a ≤ -2/9 :=
sorry

end common_root_values_max_n_and_a_range_l42_42339


namespace base_6_to_10_conversion_l42_42700

theorem base_6_to_10_conversion : 
  ∀ (n : ℕ), n = 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0 → n = 1295 :=
by
  intro n h
  sorry

end base_6_to_10_conversion_l42_42700


namespace total_notes_l42_42036

theorem total_notes (total_amount : ℤ) (num_50_notes : ℤ) (value_50 : ℤ) (value_500 : ℤ) (total_notes : ℤ) :
  total_amount = num_50_notes * value_50 + (total_notes - num_50_notes) * value_500 → 
  total_amount = 10350 → num_50_notes = 77 → value_50 = 50 → value_500 = 500 → total_notes = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_notes_l42_42036


namespace count_integers_with_same_remainder_l42_42532

theorem count_integers_with_same_remainder (n : ℤ) : 
  (150 < n ∧ n < 250) ∧ 
  (∃ r : ℤ, 0 ≤ r ∧ r ≤ 6 ∧ ∃ a b : ℤ, n = 7 * a + r ∧ n = 9 * b + r) ↔ n = 7 :=
sorry

end count_integers_with_same_remainder_l42_42532


namespace min_value_x_plus_2y_l42_42712

theorem min_value_x_plus_2y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 * y - x * y = 0) : x + 2 * y = 8 := 
by
  sorry

end min_value_x_plus_2y_l42_42712


namespace least_n_value_l42_42956

open Nat

theorem least_n_value (n : ℕ) (h : 1 / (n * (n + 1)) < 1 / 15) : n = 4 :=
sorry

end least_n_value_l42_42956


namespace lattice_points_on_hyperbola_l42_42092

open Real

theorem lattice_points_on_hyperbola : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ S ↔ x^2 - y^2 = 65)) ∧ S.card = 8 :=
by
  sorry

end lattice_points_on_hyperbola_l42_42092


namespace length_of_second_train_l42_42205

def first_train_length : ℝ := 290
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def cross_time : ℝ := 9

noncomputable def first_train_speed_mps := (first_train_speed_kmph * 1000) / 3600
noncomputable def second_train_speed_mps := (second_train_speed_kmph * 1000) / 3600
noncomputable def relative_speed := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance_covered := relative_speed * cross_time
noncomputable def second_train_length := total_distance_covered - first_train_length

theorem length_of_second_train : second_train_length = 209.95 := by
  sorry

end length_of_second_train_l42_42205


namespace inequality_nonneg_ab_l42_42732

theorem inequality_nonneg_ab (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 :=
by
  sorry

end inequality_nonneg_ab_l42_42732


namespace fred_gave_cards_l42_42703

theorem fred_gave_cards (initial_cards : ℕ) (torn_cards : ℕ) 
  (bought_cards : ℕ) (total_cards : ℕ) (fred_cards : ℕ) : 
  initial_cards = 18 → torn_cards = 8 → bought_cards = 40 → total_cards = 84 →
  fred_cards = total_cards - (initial_cards - torn_cards + bought_cards) →
  fred_cards = 34 :=
by
  intros h_initial h_torn h_bought h_total h_fred
  sorry

end fred_gave_cards_l42_42703


namespace standard_parabola_with_symmetry_axis_eq_1_l42_42812

-- Define the condition that the axis of symmetry is x = 1
def axis_of_symmetry_x_eq_one (x : ℝ) : Prop :=
  x = 1

-- Define the standard equation of the parabola y^2 = -4x
def standard_parabola_eq (y x : ℝ) : Prop :=
  y^2 = -4 * x

-- Theorem: Prove that given the axis of symmetry of the parabola is x = 1,
-- the standard equation of the parabola is y^2 = -4x.
theorem standard_parabola_with_symmetry_axis_eq_1 : ∀ (x y : ℝ),
  axis_of_symmetry_x_eq_one x → standard_parabola_eq y x :=
by
  intros
  sorry

end standard_parabola_with_symmetry_axis_eq_1_l42_42812


namespace area_of_triangle_bounded_by_line_and_axes_l42_42296

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l42_42296


namespace average_of_four_variables_l42_42234

theorem average_of_four_variables (x y z w : ℝ) (h : (5 / 2) * (x + y + z + w) = 25) :
  (x + y + z + w) / 4 = 2.5 :=
sorry

end average_of_four_variables_l42_42234


namespace gobblean_total_words_l42_42476

-- Define the Gobblean alphabet and its properties.
def gobblean_letters := 6
def max_word_length := 4

-- Function to calculate number of permutations without repetition for a given length.
def num_words (length : ℕ) : ℕ :=
  if length = 1 then 6
  else if length = 2 then 6 * 5
  else if length = 3 then 6 * 5 * 4
  else if length = 4 then 6 * 5 * 4 * 3
  else 0

-- Main theorem stating the total number of possible words.
theorem gobblean_total_words : 
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4) = 516 :=
by
  -- Proof is not required
  sorry

end gobblean_total_words_l42_42476


namespace find_constants_l42_42970

open BigOperators

theorem find_constants (a b c : ℕ) :
  (∀ n : ℕ, n > 0 → (∑ k in Finset.range n, k.succ * (k.succ + 1) ^ 2) = (n * (n + 1) * (a * n^2 + b * n + c)) / 12) →
  (a = 3 ∧ b = 11 ∧ c = 10) :=
by
  sorry

end find_constants_l42_42970


namespace minimum_inlets_needed_l42_42486

noncomputable def waterInflow (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := x * a - b

theorem minimum_inlets_needed (a b : ℝ) (ha : a = b)
  (h1 : (4 * a - b) * 5 = (2 * a - b) * 15)
  (h2 : (a * 9 - b) * 2 ≥ 1) : 
  ∃ n : ℕ, 2 * (a * n - b) ≥ (4 * a - b) * 5 := 
by 
  sorry

end minimum_inlets_needed_l42_42486


namespace original_equation_proof_l42_42806

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

end original_equation_proof_l42_42806


namespace perpendicular_lines_l42_42682

theorem perpendicular_lines (a : ℝ) :
  (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0 ↔ (a = 1 ∨ a = -1) := 
sorry

end perpendicular_lines_l42_42682


namespace compute_expr_l42_42817

open Real

-- Define the polynomial and its roots.
def polynomial (x : ℝ) := 3 * x^2 - 5 * x - 2

-- Given conditions: p and q are roots of the polynomial.
def is_root (p q : ℝ) : Prop := 
  polynomial p = 0 ∧ polynomial q = 0

-- The main theorem.
theorem compute_expr (p q : ℝ) (h : is_root p q) : 
  ∃ k : ℝ, k = p - q ∧ (p ≠ q) → (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) :=
sorry

end compute_expr_l42_42817


namespace square_possible_n12_square_possible_n15_l42_42508

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l42_42508


namespace harmonic_point_P_3_m_harmonic_point_hyperbola_l42_42061

-- Part (1)
theorem harmonic_point_P_3_m (t : ℝ) (m : ℝ) (P : ℝ × ℝ → Prop)
  (h₁ : P ⟨ 3, m ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  m = -7 :=
by sorry

-- Part (2)
theorem harmonic_point_hyperbola (k : ℝ) (P : ℝ × ℝ → Prop)
  (h_hb : ∀ x, -3 < x ∧ x < -1 → P ⟨ x, k / x ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  3 < k ∧ k < 4 :=
by sorry

end harmonic_point_P_3_m_harmonic_point_hyperbola_l42_42061


namespace matrix_cubic_l42_42849

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l42_42849


namespace ferris_wheel_small_seat_capacity_l42_42776

def num_small_seats : Nat := 2
def capacity_per_small_seat : Nat := 14

theorem ferris_wheel_small_seat_capacity : num_small_seats * capacity_per_small_seat = 28 := by
  sorry

end ferris_wheel_small_seat_capacity_l42_42776


namespace positive_solutions_l42_42037

theorem positive_solutions (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔
  x = 1 ∨ x = 3 :=
by
  sorry

end positive_solutions_l42_42037


namespace find_n_l42_42797

theorem find_n : ∃ n : ℤ, 100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180) ∧ n = 317 := 
by
  sorry

end find_n_l42_42797


namespace find_a_l42_42696

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2 : ℝ) * a * x^3 - (3 / 2 : ℝ) * x^2 + (3 / 2 : ℝ) * a^2 * x

theorem find_a (a : ℝ) (h_max : ∀ x : ℝ, f a x ≤ f a 1) : a = -2 :=
sorry

end find_a_l42_42696


namespace sum_of_remainders_l42_42749

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  sorry

end sum_of_remainders_l42_42749


namespace total_cost_backpacks_l42_42925

theorem total_cost_backpacks:
  let original_price := 20.00
  let discount := 0.20
  let monogram_cost := 12.00
  let coupon := 5.00
  let state_tax : List Real := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discounted_price := original_price * (1 - discount)
  let pre_tax_cost := discounted_price + monogram_cost
  let final_costs := state_tax.map (λ tax_rate => pre_tax_cost * (1 + tax_rate))
  let total_cost_before_coupon := final_costs.sum
  total_cost_before_coupon - coupon = 143.61 := by
    sorry

end total_cost_backpacks_l42_42925


namespace f_7_5_l42_42380

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_7_5 : f 7.5 = -0.5 := by
  sorry

end f_7_5_l42_42380


namespace stratified_sample_over_30_l42_42034

-- Define the total number of employees and conditions
def total_employees : ℕ := 49
def employees_over_30 : ℕ := 14
def employees_30_or_younger : ℕ := 35
def sample_size : ℕ := 7

-- State the proportion and the final required count
def proportion_over_30 (total : ℕ) (over_30 : ℕ) : ℚ := (over_30 : ℚ) / (total : ℚ)
def required_count (proportion : ℚ) (sample : ℕ) : ℚ := proportion * (sample : ℚ)

theorem stratified_sample_over_30 :
  required_count (proportion_over_30 total_employees employees_over_30) sample_size = 2 := 
by sorry

end stratified_sample_over_30_l42_42034


namespace tracy_initial_candies_l42_42643

theorem tracy_initial_candies (x y : ℕ) (h₁ : x = 108) (h₂ : 2 ≤ y ∧ y ≤ 6) : 
  let remaining_after_eating := (3 / 4) * x 
  let remaining_after_giving := (2 / 3) * remaining_after_eating
  let remaining_after_mom := remaining_after_giving - 40
  remaining_after_mom - y = 10 :=
by 
  sorry

end tracy_initial_candies_l42_42643


namespace cookies_in_each_bag_l42_42281

-- Definitions based on the conditions
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41
def baggies : ℕ := 6

-- Assertion of the correct answer
theorem cookies_in_each_bag : 
  (chocolate_chip_cookies + oatmeal_cookies) / baggies = 9 := by
  sorry

end cookies_in_each_bag_l42_42281


namespace final_point_P_after_transformations_l42_42968

noncomputable def point := (ℝ × ℝ)

def rotate_90_clockwise (p : point) : point :=
  (-p.2, p.1)

def reflect_across_x (p : point) : point :=
  (p.1, -p.2)

def P : point := (3, -5)

def Q : point := (5, -2)

def R : point := (5, -5)

theorem final_point_P_after_transformations : reflect_across_x (rotate_90_clockwise P) = (-5, 3) :=
by 
  sorry

end final_point_P_after_transformations_l42_42968


namespace final_elephants_count_l42_42362

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end final_elephants_count_l42_42362


namespace number_of_diagonals_excluding_dividing_diagonals_l42_42656

theorem number_of_diagonals_excluding_dividing_diagonals (n : ℕ) (h1 : n = 150) :
  let totalDiagonals := n * (n - 3) / 2
  let dividingDiagonals := n / 2
  totalDiagonals - dividingDiagonals = 10950 :=
by
  sorry

end number_of_diagonals_excluding_dividing_diagonals_l42_42656


namespace range_of_a_div_b_l42_42641

theorem range_of_a_div_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : 2 < b ∧ b < 8) : 
  1 / 8 < a / b ∧ a / b < 2 :=
sorry

end range_of_a_div_b_l42_42641


namespace total_points_of_players_l42_42033

variables (Samanta Mark Eric Daisy Jake : ℕ)
variables (h1 : Samanta = Mark + 8)
variables (h2 : Mark = 3 / 2 * Eric)
variables (h3 : Eric = 6)
variables (h4 : Daisy = 3 / 4 * (Samanta + Mark + Eric))
variables (h5 : Jake = Samanta - Eric)
 
theorem total_points_of_players :
  Samanta + Mark + Eric + Daisy + Jake = 67 :=
sorry

end total_points_of_players_l42_42033


namespace arithmetic_geometric_sequence_min_sum_l42_42301

theorem arithmetic_geometric_sequence_min_sum :
  ∃ (A B C D : ℕ), 
    (C - B = B - A) ∧ 
    (C * 4 = B * 7) ∧ 
    (D * 4 = C * 7) ∧ 
    (16 ∣ B) ∧ 
    (A + B + C + D = 97) :=
by sorry

end arithmetic_geometric_sequence_min_sum_l42_42301


namespace functional_equation_solution_l42_42460

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y * f (x + y) + f x) = 4 * x + 2 * y * f (x + y)) →
  (∀ x : ℝ, f x = 2 * x) :=
sorry

end functional_equation_solution_l42_42460


namespace number_of_outfits_l42_42872

theorem number_of_outfits : (5 * 4 * 6 * 3) = 360 := by
  sorry

end number_of_outfits_l42_42872


namespace determinant_matrix_zero_l42_42314

theorem determinant_matrix_zero (θ φ : ℝ) : 
  Matrix.det ![
    ![0, Real.cos θ, -Real.sin θ],
    ![-Real.cos θ, 0, Real.cos φ],
    ![Real.sin θ, -Real.cos φ, 0]
  ] = 0 := by sorry

end determinant_matrix_zero_l42_42314


namespace g_g_g_25_l42_42073

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 18

theorem g_g_g_25 :
  g (g (g 25)) = 22 :=
by
  sorry

end g_g_g_25_l42_42073


namespace steve_marbles_after_trans_l42_42066

def initial_marbles (S T L H : ℕ) : Prop :=
  S = 2 * T ∧
  L = S - 5 ∧
  H = T + 3

def transactions (S T L H : ℕ) (new_S new_T new_L new_H : ℕ) : Prop :=
  new_S = S - 10 ∧
  new_L = L - 4 ∧
  new_T = T + 4 ∧
  new_H = H - 6

theorem steve_marbles_after_trans (S T L H new_S new_T new_L new_H : ℕ) :
  initial_marbles S T L H →
  transactions S T L H new_S new_T new_L new_H →
  new_S = 6 →
  new_T = 12 :=
by
  sorry

end steve_marbles_after_trans_l42_42066


namespace lies_on_new_ellipse_lies_on_new_hyperbola_l42_42315

variable (x y c d a : ℝ)

def new_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Definition for new ellipse.
def is_new_ellipse (E : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  new_distance E F1 + new_distance E F2 = 2 * a

-- Definition for new hyperbola.
def is_new_hyperbola (H : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  |new_distance H F1 - new_distance H F2| = 2 * a

-- The point E lies on the new ellipse.
theorem lies_on_new_ellipse
  (E F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_ellipse E F1 F2 a :=
by sorry

-- The point H lies on the new hyperbola.
theorem lies_on_new_hyperbola
  (H F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_hyperbola H F1 F2 a :=
by sorry

end lies_on_new_ellipse_lies_on_new_hyperbola_l42_42315


namespace scientific_notation_of_0_0000003_l42_42784

theorem scientific_notation_of_0_0000003 :
  0.0000003 = 3 * 10^(-7) :=
sorry

end scientific_notation_of_0_0000003_l42_42784


namespace average_minutes_run_per_day_l42_42589

-- Define the given averages for each grade
def sixth_grade_avg : ℕ := 10
def seventh_grade_avg : ℕ := 18
def eighth_grade_avg : ℕ := 12

-- Define the ratios of the number of students in each grade
def num_sixth_eq_three_times_num_seventh (num_seventh : ℕ) : ℕ := 3 * num_seventh
def num_eighth_eq_half_num_seventh (num_seventh : ℕ) : ℕ := num_seventh / 2

-- Average number of minutes run per day by all students
theorem average_minutes_run_per_day (num_seventh : ℕ) :
  (sixth_grade_avg * num_sixth_eq_three_times_num_seventh num_seventh +
   seventh_grade_avg * num_seventh +
   eighth_grade_avg * num_eighth_eq_half_num_seventh num_seventh) / 
  (num_sixth_eq_three_times_num_seventh num_seventh + 
   num_seventh + 
   num_eighth_eq_half_num_seventh num_seventh) = 12 := 
sorry

end average_minutes_run_per_day_l42_42589


namespace pencil_case_costs_l42_42858

variable {x y : ℝ}

theorem pencil_case_costs :
  (2 * x + 3 * y = 108) ∧ (5 * x = 6 * y) → 
  (x = 24) ∧ (y = 20) :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  sorry

end pencil_case_costs_l42_42858


namespace back_parking_lot_filled_fraction_l42_42907

theorem back_parking_lot_filled_fraction
    (front_spaces : ℕ) (back_spaces : ℕ) (cars_parked : ℕ) (spaces_available : ℕ)
    (h1 : front_spaces = 52)
    (h2 : back_spaces = 38)
    (h3 : cars_parked = 39)
    (h4 : spaces_available = 32) :
    (back_spaces - (front_spaces + back_spaces - cars_parked - spaces_available)) / back_spaces = 1 / 2 :=
by
  sorry

end back_parking_lot_filled_fraction_l42_42907


namespace range_of_a_l42_42896

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) ↔ a < 1004 := 
sorry

end range_of_a_l42_42896


namespace complementary_angles_ratio_l42_42694

theorem complementary_angles_ratio (x : ℝ) (hx : 5 * x = 90) : abs (4 * x - x) = 54 :=
by
  have h₁ : x = 18 := by 
    linarith [hx]
  rw [h₁]
  norm_num

end complementary_angles_ratio_l42_42694


namespace find_a_integer_condition_l42_42345

theorem find_a_integer_condition (a : ℚ) :
  (∀ n : ℕ, (a * (n * (n+2) * (n+3) * (n+4)) : ℚ).den = 1) ↔ ∃ k : ℤ, a = k / 6 := 
sorry

end find_a_integer_condition_l42_42345


namespace passengers_on_ship_l42_42104

theorem passengers_on_ship (P : ℕ)
  (h1 : P / 12 + P / 4 + P / 9 + P / 6 + 42 = P) :
  P = 108 := 
by sorry

end passengers_on_ship_l42_42104


namespace fuel_consumption_per_100_km_l42_42058

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end fuel_consumption_per_100_km_l42_42058


namespace initial_amount_is_800_l42_42346

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end initial_amount_is_800_l42_42346


namespace part1_l42_42211

theorem part1 (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  (a^2 + a * b + b^2) / (a + b) - (a^2 - a * b + b^2) / (a - b) + (2 * b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := 
sorry

end part1_l42_42211


namespace solution_set_of_inequality_l42_42928

variable {R : Type*} [LinearOrder R] [OrderedAddCommGroup R]

def odd_function (f : R → R) := ∀ x, f (-x) = -f x

def monotonic_increasing_on (f : R → R) (s : Set R) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : odd_function f)
  (h_mono_inc : monotonic_increasing_on f (Set.Ioi 0))
  (h_f_neg1 : f (-1) = 2) : 
  {x : ℝ | 0 < x ∧ f (x-1) + 2 ≤ 0 } = Set.Ioc 1 2 :=
by
  sorry

end solution_set_of_inequality_l42_42928


namespace perpendicular_lines_k_value_l42_42408

theorem perpendicular_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_k_value_l42_42408


namespace solve_for_multiplier_l42_42597

namespace SashaSoup
  
-- Variables representing the amounts of salt
variables (x y : ℝ)

-- Condition provided: amount of salt added today
def initial_salt := 2 * x
def additional_salt_today := 0.5 * y

-- Given relationship
axiom salt_relationship : x = 0.5 * y

-- The multiplier k to achieve the required amount of salt
def required_multiplier : ℝ := 1.5

-- Lean theorem statement
theorem solve_for_multiplier :
  (2 * x) * required_multiplier = x + y :=
by
  -- Mathematical proof goes here but since asked to skip proof we use sorry
  sorry

end SashaSoup

end solve_for_multiplier_l42_42597


namespace sqrt_sum_of_fractions_as_fraction_l42_42452

theorem sqrt_sum_of_fractions_as_fraction :
  (Real.sqrt ((36 / 49) + (16 / 9) + (1 / 16))) = (45 / 28) :=
by
  sorry

end sqrt_sum_of_fractions_as_fraction_l42_42452


namespace purchase_price_of_radio_l42_42035

theorem purchase_price_of_radio 
  (selling_price : ℚ) (loss_percentage : ℚ) (purchase_price : ℚ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 0.05):
  purchase_price = 490 :=
by 
  sorry

end purchase_price_of_radio_l42_42035


namespace sufficient_but_not_necessary_condition_l42_42747

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (∃ m0 : ℝ, m0 > 0 ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m0 x1 ≤ f m0 x2)) ∧ 
  ¬ (∀ m : ℝ, (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m x1 ≤ f m x2) → m > 0) :=
by sorry

end sufficient_but_not_necessary_condition_l42_42747


namespace find_multiple_of_games_l42_42891

-- declaring the number of video games each person has
def Tory_videos := 6
def Theresa_videos := 11
def Julia_videos := Tory_videos / 3

-- declaring the multiple we need to find
def multiple_of_games := Theresa_videos - Julia_videos * 5

-- Theorem stating the problem
theorem find_multiple_of_games : ∃ m : ℕ, Julia_videos * m + 5 = Theresa_videos :=
by
  sorry

end find_multiple_of_games_l42_42891


namespace find_sum_l42_42227

theorem find_sum 
  (R : ℝ) -- Original interest rate
  (P : ℝ) -- Principal amount
  (h: (P * (R + 3) * 3 / 100) = ((P * R * 3 / 100) + 81)): 
  P = 900 :=
sorry

end find_sum_l42_42227


namespace rectangular_field_area_l42_42546

theorem rectangular_field_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) :
  w * l = 243 :=
by
  -- Proof goes here
  sorry

end rectangular_field_area_l42_42546


namespace sets_equal_l42_42221

theorem sets_equal (M N : Set ℝ) (hM : M = { x | x^2 = 1 }) (hN : N = { a | ∀ x ∈ M, a * x = 1 }) : M = N :=
sorry

end sets_equal_l42_42221


namespace monotonically_increasing_condition_l42_42410

theorem monotonically_increasing_condition 
  (a b c d : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ 3 * a * x ^ 2 + 2 * b * x + c) ↔ (b^2 - 3 * a * c ≤ 0) :=
by {
  sorry
}

end monotonically_increasing_condition_l42_42410


namespace pine_sample_count_l42_42425

variable (total_saplings : ℕ)
variable (pine_saplings : ℕ)
variable (sample_size : ℕ)

theorem pine_sample_count (h1 : total_saplings = 30000) (h2 : pine_saplings = 4000) (h3 : sample_size = 150) :
  pine_saplings * sample_size / total_saplings = 20 := 
sorry

end pine_sample_count_l42_42425


namespace fabric_problem_l42_42129

theorem fabric_problem
  (x y : ℝ)
  (h1 : y > 0)
  (cost_second_piece := x)
  (cost_first_piece := x + 126)
  (cost_per_meter_first := (x + 126) / y)
  (cost_per_meter_second := x / y)
  (h2 : 4 * cost_per_meter_first - 3 * cost_per_meter_second = 135)
  (h3 : 3 * cost_per_meter_first + 4 * cost_per_meter_second = 382.5) :
  y = 5.6 ∧ cost_per_meter_first = 67.5 ∧ cost_per_meter_second = 45 :=
sorry

end fabric_problem_l42_42129


namespace monotonicity_f_on_interval_l42_42959

def f (x : ℝ) : ℝ := |x + 2|

theorem monotonicity_f_on_interval :
  ∀ x1 x2 : ℝ, x1 < x2 → x1 < -4 → x2 < -4 → f x1 ≥ f x2 :=
by
  sorry

end monotonicity_f_on_interval_l42_42959


namespace not_proportional_l42_42821

theorem not_proportional (x y : ℕ) :
  (∀ k : ℝ, y ≠ 3 * x - 7 ∧ y ≠ (13 - 4 * x) / 3) → 
  ((y = 3 * x - 7 ∨ y = (13 - 4 * x) / 3) → ¬(∃ k : ℝ, (y = k * x) ∨ (y = k / x))) := sorry

end not_proportional_l42_42821


namespace equivalent_statements_l42_42518

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalent_statements_l42_42518


namespace minimum_value_of_f_l42_42988

-- Define the function
def f (a b x : ℝ) := x^2 + (a + 2) * x + b

-- Condition that ensures the graph is symmetric about x = 1
def symmetric_about_x1 (a : ℝ) : Prop := a + 2 = -2

-- Minimum value of the function f(x) in terms of the constant c
theorem minimum_value_of_f (a b : ℝ) (h : symmetric_about_x1 a) : ∃ c : ℝ, ∀ x : ℝ, f a b x ≥ c :=
by sorry

end minimum_value_of_f_l42_42988


namespace volume_of_right_prism_correct_l42_42920

variables {α β l : ℝ}

noncomputable def volume_of_right_prism (α β l : ℝ) : ℝ :=
  (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α))

theorem volume_of_right_prism_correct
  (α β l : ℝ)
  (α_gt0 : 0 < α) (α_lt90 : α < Real.pi / 2)
  (l_pos : 0 < l)
  : volume_of_right_prism α β l = (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α)) :=
sorry

end volume_of_right_prism_correct_l42_42920


namespace part_I_part_II_l42_42395

open Real

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem part_I (x : ℝ) : f x > 0 ↔ (x > 1 ∨ x < -5) := 
sorry

theorem part_II (m : ℝ) : (∀ x : ℝ, f x + 3 * abs (x - 4) > m) ↔ (m < 9) :=
sorry

end part_I_part_II_l42_42395


namespace algebra_books_needed_l42_42313

theorem algebra_books_needed (A' H' S' M' E' : ℕ) (x y : ℝ) (z : ℝ)
  (h1 : y > x)
  (h2 : A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ S' ≠ M' ∧ S' ≠ E' ∧ M' ≠ E')
  (h3 : A' * x + H' * y = z)
  (h4 : S' * x + M' * y = z)
  (h5 : E' * x = 2 * z) :
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by
  sorry

end algebra_books_needed_l42_42313


namespace min_value_of_linear_combination_of_variables_l42_42354

-- Define the conditions that x and y are positive numbers and satisfy the equation x + 3y = 5xy
def conditions (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y

-- State the theorem that the minimum value of 3x + 4y given the conditions is 5
theorem min_value_of_linear_combination_of_variables (x y : ℝ) (h: conditions x y) : 3 * x + 4 * y ≥ 5 :=
by 
  sorry

end min_value_of_linear_combination_of_variables_l42_42354


namespace abs_inequality_m_eq_neg4_l42_42480

theorem abs_inequality_m_eq_neg4 (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ (m = -4) :=
by
  sorry

end abs_inequality_m_eq_neg4_l42_42480


namespace yearly_savings_l42_42590

-- Define the various constants given in the problem
def weeks_in_year : ℕ := 52
def months_in_year : ℕ := 12
def non_peak_weeks : ℕ := 16
def peak_weeks : ℕ := weeks_in_year - non_peak_weeks
def non_peak_months : ℕ := 4
def peak_months : ℕ := months_in_year - non_peak_months

-- Rates
def weekly_cost_non_peak_large : ℕ := 10
def weekly_cost_peak_large : ℕ := 12
def monthly_cost_non_peak_large : ℕ := 42
def monthly_cost_peak_large : ℕ := 48

-- Additional surcharge
def holiday_weeks : ℕ := 6
def holiday_surcharge : ℕ := 2

-- Compute the yearly costs
def yearly_weekly_cost : ℕ :=
  (non_peak_weeks * weekly_cost_non_peak_large) +
  (peak_weeks * weekly_cost_peak_large) +
  (holiday_weeks * (holiday_surcharge + weekly_cost_peak_large))

def yearly_monthly_cost : ℕ :=
  (non_peak_months * monthly_cost_non_peak_large) +
  (peak_months * monthly_cost_peak_large)

theorem yearly_savings : yearly_weekly_cost - yearly_monthly_cost = 124 := by
  sorry

end yearly_savings_l42_42590


namespace inequality_proof_l42_42378

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) : 
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) :=
sorry

end inequality_proof_l42_42378


namespace solve_eq_l42_42803

theorem solve_eq (x : ℝ) : (x - 2)^2 = 9 * x^2 ↔ x = -1 ∨ x = 1 / 2 := by
  sorry

end solve_eq_l42_42803


namespace cos_240_eq_neg_half_l42_42025

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l42_42025


namespace domain_of_f_l42_42273

-- The domain of the function is the set of all x such that the function is defined.
theorem domain_of_f:
  {x : ℝ | x > 3 ∧ x ≠ 4} = (Set.Ioo 3 4 ∪ Set.Ioi 4) := 
sorry

end domain_of_f_l42_42273


namespace symmetric_origin_a_minus_b_l42_42000

noncomputable def A (a : ℝ) := (a, -2)
noncomputable def B (b : ℝ) := (4, b)
def symmetric (p q : ℝ × ℝ) : Prop := (q.1 = -p.1) ∧ (q.2 = -p.2)

theorem symmetric_origin_a_minus_b (a b : ℝ) (hA : A a = (-4, -2)) (hB : B b = (4, 2)) :
  a - b = -6 := by
  sorry

end symmetric_origin_a_minus_b_l42_42000


namespace problem_statement_l42_42109

def prop_p (x : ℝ) : Prop := x^2 >= x
def prop_q : Prop := ∃ x : ℝ, x^2 >= x

theorem problem_statement : (∀ x : ℝ, prop_p x) = false ∧ prop_q = true :=
by 
  sorry

end problem_statement_l42_42109


namespace probability_participation_on_both_days_l42_42915

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end probability_participation_on_both_days_l42_42915


namespace impossible_d_values_count_l42_42673

def triangle_rectangle_difference (d : ℕ) : Prop :=
  ∃ (l w : ℕ),
  l = 2 * w ∧
  6 * w > 0 ∧
  (6 * w + 2 * d) - 6 * w = 1236 ∧
  d > 0

theorem impossible_d_values_count : ∀ d : ℕ, d ≠ 618 → ¬triangle_rectangle_difference d :=
by
  sorry

end impossible_d_values_count_l42_42673


namespace girls_from_clay_is_30_l42_42800

-- Definitions for the given conditions
def total_students : ℕ := 150
def total_boys : ℕ := 90
def total_girls : ℕ := 60
def students_jonas : ℕ := 50
def students_clay : ℕ := 70
def students_hart : ℕ := 30
def boys_jonas : ℕ := 25

-- Theorem to prove that the number of girls from Clay Middle School is 30
theorem girls_from_clay_is_30 
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : students_jonas = 50)
  (h5 : students_clay = 70)
  (h6 : students_hart = 30)
  (h7 : boys_jonas = 25) : 
  ∃ girls_clay : ℕ, girls_clay = 30 :=
by 
  sorry

end girls_from_clay_is_30_l42_42800


namespace reciprocal_fraction_addition_l42_42941

theorem reciprocal_fraction_addition (a b c : ℝ) (h : a ≠ b) :
  (a + c) / (b + c) = b / a ↔ c = - (a + b) := 
by
  sorry

end reciprocal_fraction_addition_l42_42941


namespace part_I_equality_condition_part_II_l42_42160

-- Lean statement for Part (I)
theorem part_I (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) : 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ 5 :=
sorry

theorem equality_condition (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  (2 * Real.sqrt x + Real.sqrt (5 - x) = 5) ↔ (x = 4) :=
sorry

-- Lean statement for Part (II)
theorem part_II (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 5) → 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ |m - 2|) →
  (m ≥ 7 ∨ m ≤ -3) :=
sorry

end part_I_equality_condition_part_II_l42_42160


namespace ratio_largest_middle_l42_42270

-- Definitions based on given conditions
def A : ℕ := 24  -- smallest number
def B : ℕ := 40  -- middle number
def C : ℕ := 56  -- largest number

theorem ratio_largest_middle (h1 : C = 56) (h2 : A = C - 32) (h3 : A = 24) (h4 : B = 40) :
  C / B = 7 / 5 := by
  sorry

end ratio_largest_middle_l42_42270


namespace smallest_possible_value_of_AP_plus_BP_l42_42181

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem smallest_possible_value_of_AP_plus_BP :
  let A := (1, 0)
  let B := (-3, 4)
  ∃ P : ℝ × ℝ, (P.2 ^ 2 = 4 * P.1) ∧
  (distance A P + distance B P = 12) :=
by
  -- proof steps would go here
  sorry

end smallest_possible_value_of_AP_plus_BP_l42_42181


namespace steve_distance_l42_42230

theorem steve_distance (D : ℝ) (S : ℝ) 
  (h1 : 2 * S = 10)
  (h2 : (D / S) + (D / (2 * S)) = 6) :
  D = 20 :=
by
  sorry

end steve_distance_l42_42230


namespace lines_intersect_l42_42534

-- Define the coefficients of the lines
def A1 : ℝ := 3
def B1 : ℝ := -2
def C1 : ℝ := 5

def A2 : ℝ := 1
def B2 : ℝ := 3
def C2 : ℝ := 10

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := A1 * x + B1 * y + C1 = 0
def line2 (x y : ℝ) : Prop := A2 * x + B2 * y + C2 = 0

-- Mathematical problem to prove
theorem lines_intersect : ∃ (x y : ℝ), line1 x y ∧ line2 x y :=
by
  sorry

end lines_intersect_l42_42534


namespace find_two_digit_number_l42_42453

theorem find_two_digit_number (n : ℕ) (h1 : n % 9 = 7) (h2 : n % 7 = 5) (h3 : n % 3 = 1) (h4 : 10 ≤ n) (h5 : n < 100) : n = 61 := 
by
  sorry

end find_two_digit_number_l42_42453


namespace number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l42_42264

theorem number_of_apples (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (apples_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    apples_mult = 5 → 
    (apples_mult * peaches_fraction * oranges_fraction * total_fruit) = 35 :=
by
  intros h1 h2 h3
  sorry

theorem ratio_of_mixed_fruits (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (mixed_fruits_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    mixed_fruits_mult = 2 → 
    (mixed_fruits_mult * peaches_fraction * oranges_fraction * total_fruit) / total_fruit = 1/4 :=
by
  intros h1 h2 h3
  sorry

theorem total_weight_of_oranges (total_fruit : ℕ) (oranges_fraction : ℚ) (orange_weight : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    orange_weight = 200 → 
    (orange_weight * oranges_fraction * total_fruit) = 2800 :=
by
  intros h1 h2
  sorry

end number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l42_42264


namespace heartsuit_fraction_l42_42477

-- Define the operation heartsuit
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Define the proof statement
theorem heartsuit_fraction :
  (heartsuit 2 4) / (heartsuit 4 2) = 2 :=
by
  -- We use 'sorry' to skip the actual proof steps
  sorry

end heartsuit_fraction_l42_42477


namespace pyramid_four_triangular_faces_area_l42_42224

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l42_42224


namespace volume_of_blue_tetrahedron_in_cube_l42_42553

theorem volume_of_blue_tetrahedron_in_cube (side_length : ℝ) (h : side_length = 8) :
  let cube_volume := side_length^3
  let tetrahedra_volume := 4 * (1/3 * (1/2 * side_length * side_length) * side_length)
  cube_volume - tetrahedra_volume = 512/3 :=
by
  sorry

end volume_of_blue_tetrahedron_in_cube_l42_42553


namespace expression_simplifies_to_one_l42_42096

theorem expression_simplifies_to_one :
  ( (105^2 - 8^2) / (80^2 - 13^2) ) * ( (80 - 13) * (80 + 13) / ( (105 - 8) * (105 + 8) ) ) = 1 :=
by
  sorry

end expression_simplifies_to_one_l42_42096


namespace coin_overlap_black_region_cd_sum_l42_42579

noncomputable def black_region_probability : ℝ := 
  let square_side := 10
  let triangle_leg := 3
  let diamond_side := 3 * Real.sqrt 2
  let coin_diameter := 2
  let coin_radius := coin_diameter / 2
  let reduced_square_side := square_side - coin_diameter
  let reduced_square_area := reduced_square_side * reduced_square_side
  let triangle_area := 4 * ((triangle_leg * triangle_leg) / 2)
  let extra_triangle_area := 4 * (Real.pi / 4 + 3)
  let diamond_area := (diamond_side * diamond_side) / 2
  let extra_diamond_area := Real.pi + 12 * Real.sqrt 2
  let total_black_area := triangle_area + extra_triangle_area + diamond_area + extra_diamond_area

  total_black_area / reduced_square_area

theorem coin_overlap_black_region: 
  black_region_probability = (1 / 64) * (30 + 12 * Real.sqrt 2 + Real.pi) := 
sorry

theorem cd_sum: 
  let c := 30
  let d := 12
  c + d = 42 := 
by
  trivial

end coin_overlap_black_region_cd_sum_l42_42579


namespace discs_contain_equal_minutes_l42_42213

theorem discs_contain_equal_minutes (total_time discs_capacity : ℕ) 
  (h1 : total_time = 520) (h2 : discs_capacity = 65) :
  ∃ discs_needed : ℕ, discs_needed = total_time / discs_capacity ∧ 
  ∀ (k : ℕ), k = total_time / discs_needed → k = 65 :=
by
  sorry

end discs_contain_equal_minutes_l42_42213


namespace smallest_a_l42_42701

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_a (a : ℕ) (h1 : 5880 = 2^3 * 3^1 * 5^1 * 7^2)
                    (h2 : ∀ b : ℕ, b < a → ¬ is_perfect_square (5880 * b))
                    : a = 15 :=
by
  sorry

end smallest_a_l42_42701


namespace true_proposition_l42_42604

noncomputable def prop_p (x : ℝ) : Prop := x > 0 → x^2 - 2*x + 1 > 0

noncomputable def prop_q (x₀ : ℝ) : Prop := x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

theorem true_proposition : ¬ (∀ x > 0, x^2 - 2*x + 1 > 0) ∧ (∃ x₀ > 0, x₀^2 - 2*x₀ + 1 ≤ 0) :=
by
  sorry

end true_proposition_l42_42604


namespace total_students_like_sports_l42_42689

def Total_students := 30

def B : ℕ := 12
def C : ℕ := 10
def S : ℕ := 8
def BC : ℕ := 4
def BS : ℕ := 3
def CS : ℕ := 2
def BCS : ℕ := 1

theorem total_students_like_sports : 
  (B + C + S - (BC + BS + CS) + BCS = 22) := by
  sorry

end total_students_like_sports_l42_42689


namespace average_salary_all_workers_l42_42249

/-- The total number of workers in the workshop is 15 -/
def total_number_of_workers : ℕ := 15

/-- The number of technicians is 5 -/
def number_of_technicians : ℕ := 5

/-- The number of other workers is given by the total number minus technicians -/
def number_of_other_workers : ℕ := total_number_of_workers - number_of_technicians

/-- The average salary per head of the technicians is Rs. 800 -/
def average_salary_per_technician : ℕ := 800

/-- The average salary per head of the other workers is Rs. 650 -/
def average_salary_per_other_worker : ℕ := 650

/-- The total salary for all the workers -/
def total_salary : ℕ := (number_of_technicians * average_salary_per_technician) + (number_of_other_workers * average_salary_per_other_worker)

/-- The average salary per head of all the workers in the workshop is Rs. 700 -/
theorem average_salary_all_workers :
  total_salary / total_number_of_workers = 700 := by
  sorry

end average_salary_all_workers_l42_42249


namespace MN_length_correct_l42_42658

open Real

noncomputable def MN_segment_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  sqrt (a * b)

theorem MN_length_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (MN : ℝ), MN = MN_segment_length a b h1 h2 :=
by
  use sqrt (a * b)
  exact rfl

end MN_length_correct_l42_42658


namespace max_possible_value_of_a_l42_42145

theorem max_possible_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) : 
  a ≤ 8924 :=
by {
  sorry
}

end max_possible_value_of_a_l42_42145


namespace miranda_pillows_l42_42340

theorem miranda_pillows (feathers_per_pound : ℕ) (total_feathers : ℕ) (pillows : ℕ)
  (h1 : feathers_per_pound = 300) (h2 : total_feathers = 3600) (h3 : pillows = 6) :
  (total_feathers / feathers_per_pound) / pillows = 2 := by
  sorry

end miranda_pillows_l42_42340


namespace sum_of_elements_in_T_l42_42065

noncomputable def digit_sum : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 504
noncomputable def repeating_sum : ℕ := digit_sum * 1111
noncomputable def sum_T : ℚ := repeating_sum / 9999

theorem sum_of_elements_in_T : sum_T = 2523 := by
  sorry

end sum_of_elements_in_T_l42_42065


namespace tan_sum_sin_cos_conditions_l42_42573

theorem tan_sum_sin_cos_conditions {x y : ℝ} 
  (h1 : Real.sin x + Real.sin y = 1 / 2) 
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = -Real.sqrt 3 := 
sorry

end tan_sum_sin_cos_conditions_l42_42573


namespace median_of_roller_coaster_times_l42_42854

theorem median_of_roller_coaster_times:
  let data := [80, 85, 90, 125, 130, 135, 140, 145, 195, 195, 210, 215, 240, 245, 300, 305, 315, 320, 325, 330, 300]
  ∃ median_time, median_time = 210 ∧
    (∀ t ∈ data, t ≤ median_time ↔ index_of_median = 11) :=
by
  sorry

end median_of_roller_coaster_times_l42_42854


namespace total_votes_l42_42537

theorem total_votes (votes_veggies : ℕ) (votes_meat : ℕ) (H1 : votes_veggies = 337) (H2 : votes_meat = 335) : votes_veggies + votes_meat = 672 :=
by
  sorry

end total_votes_l42_42537


namespace Q_investment_time_l42_42404

theorem Q_investment_time  
  (P Q x t : ℝ)
  (h_ratio_investments : P = 7 * x ∧ Q = 5 * x)
  (h_ratio_profits : (7 * x * 10) / (5 * x * t) = 7 / 10) :
  t = 20 :=
by {
  sorry
}

end Q_investment_time_l42_42404


namespace larger_of_two_numbers_l42_42247

theorem larger_of_two_numbers (A B : ℕ) (HCF : ℕ) (factor1 factor2 : ℕ) (h_hcf : HCF = 23) (h_factor1 : factor1 = 13) (h_factor2 : factor2 = 14)
(hA : A = HCF * factor1) (hB : B = HCF * factor2) :
  max A B = 322 :=
by
  sorry

end larger_of_two_numbers_l42_42247


namespace jason_additional_manager_months_l42_42702

def additional_manager_months (bartender_years manager_years total_exp_months : ℕ) : ℕ :=
  let bartender_months := bartender_years * 12
  let manager_months := manager_years * 12
  total_exp_months - (bartender_months + manager_months)

theorem jason_additional_manager_months : 
  additional_manager_months 9 3 150 = 6 := 
by 
  sorry

end jason_additional_manager_months_l42_42702


namespace area_of_sector_l42_42407

-- Given conditions
def central_angle : ℝ := 2
def perimeter : ℝ := 8

-- Define variables and expressions
variable (r l : ℝ)

-- Equations based on the conditions
def eq1 := l + 2 * r = perimeter
def eq2 := l = central_angle * r

-- Assertion of the correct answer
theorem area_of_sector : ∃ r l : ℝ, eq1 r l ∧ eq2 r l ∧ (1 / 2 * l * r = 4) := by
  sorry

end area_of_sector_l42_42407


namespace find_a_l42_42207

def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem find_a : (f_prime a (-1) = 3) → a = 3 :=
by
  sorry

end find_a_l42_42207


namespace total_charge_3_hours_l42_42084

-- Define the charges for the first hour (F) and additional hours (A)
variable (F A : ℝ)

-- Given conditions
axiom charge_relation : F = A + 20
axiom total_charge_5_hours : F + 4 * A = 300

-- The theorem stating the total charge for 3 hours of therapy
theorem total_charge_3_hours : 
  (F + 2 * A) = 188 :=
by
  -- Insert the proof here
  sorry

end total_charge_3_hours_l42_42084


namespace arithmetic_sequence_general_term_l42_42869

theorem arithmetic_sequence_general_term
  (d : ℕ) (a : ℕ → ℕ)
  (ha4 : a 4 = 14)
  (hd : d = 3) :
  ∃ a₁, ∀ n, a n = a₁ + (n - 1) * d := by
  sorry

end arithmetic_sequence_general_term_l42_42869


namespace number_of_solutions_l42_42536

theorem number_of_solutions :
  ∃ n : ℕ,  (1 + ⌊(102 * n : ℚ) / 103⌋ = ⌈(101 * n : ℚ) / 102⌉) ↔ (n < 10506) := 
sorry

end number_of_solutions_l42_42536


namespace fill_time_calculation_l42_42183

-- Definitions based on conditions
def pool_volume : ℝ := 24000
def number_of_hoses : ℕ := 6
def water_per_hose_per_minute : ℝ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement translating the mathematically equivalent proof problem
theorem fill_time_calculation :
  pool_volume / (number_of_hoses * water_per_hose_per_minute * minutes_per_hour) = 22 :=
by
  sorry

end fill_time_calculation_l42_42183


namespace tangent_line_through_origin_eq_ex_l42_42846

theorem tangent_line_through_origin_eq_ex :
  ∃ (k : ℝ), (∀ x : ℝ, y = e^x) ∧ (∃ x₀ : ℝ, y - e^x₀ = e^x₀ * (x - x₀)) ∧ 
  (y = k * x) :=
sorry

end tangent_line_through_origin_eq_ex_l42_42846


namespace estimated_percentage_negative_attitude_l42_42793

-- Define the conditions
def total_parents := 2500
def sample_size := 400
def negative_attitude := 360

-- Prove the estimated percentage of parents with a negative attitude is 90%
theorem estimated_percentage_negative_attitude : 
  (negative_attitude: ℝ) / (sample_size: ℝ) * 100 = 90 := by
  sorry

end estimated_percentage_negative_attitude_l42_42793


namespace sandy_savings_last_year_l42_42228

theorem sandy_savings_last_year (S : ℝ) (P : ℝ) 
(h1 : P / 100 * S = x)
(h2 : 1.10 * S = y)
(h3 : 0.10 * y = 0.11 * S)
(h4 : 0.11 * S = 1.8333333333333331 * x) :
P = 6 := by
  -- proof goes here
  sorry

end sandy_savings_last_year_l42_42228


namespace circles_intersect_l42_42509

-- Definitions of the circles
def circle_O1 := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
def circle_O2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 9}

-- Proving the relationship between the circles
theorem circles_intersect : ∀ (p : ℝ × ℝ),
  p ∈ circle_O1 ∧ p ∈ circle_O2 :=
sorry

end circles_intersect_l42_42509


namespace value_to_subtract_l42_42787

theorem value_to_subtract (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 34) / 10 = 2) : x = 5 :=
by 
  sorry

end value_to_subtract_l42_42787


namespace B2F_base16_to_base10_l42_42369

theorem B2F_base16_to_base10 :
  let d2 := 11
  let d1 := 2
  let d0 := 15
  d2 * 16^2 + d1 * 16^1 + d0 * 16^0 = 2863 :=
by
  let d2 := 11
  let d1 := 2
  let d0 := 15
  sorry

end B2F_base16_to_base10_l42_42369


namespace solve_complex_eq_l42_42448

theorem solve_complex_eq (z : ℂ) (h : z^2 = -100 - 64 * I) : z = 3.06 - 10.46 * I ∨ z = -3.06 + 10.46 * I :=
by
  sorry

end solve_complex_eq_l42_42448


namespace john_bought_3_reels_l42_42517

theorem john_bought_3_reels (reel_length section_length : ℕ) (n_sections : ℕ)
  (h1 : reel_length = 100) (h2 : section_length = 10) (h3 : n_sections = 30) :
  n_sections * section_length / reel_length = 3 :=
by
  sorry

end john_bought_3_reels_l42_42517


namespace peter_remaining_money_l42_42961

def initial_amount : Float := 500.0 
def sales_tax : Float := 0.05
def discount : Float := 0.10

def calculate_cost_with_tax (price_per_kilo: Float) (quantity: Float) (tax_rate: Float) : Float :=
  quantity * price_per_kilo * (1 + tax_rate)

def calculate_cost_with_discount (price_per_kilo: Float) (quantity: Float) (discount_rate: Float) : Float :=
  quantity * price_per_kilo * (1 - discount_rate)

def total_first_trip : Float :=
  calculate_cost_with_tax 2.0 6 sales_tax +
  calculate_cost_with_tax 3.0 9 sales_tax +
  calculate_cost_with_tax 4.0 5 sales_tax +
  calculate_cost_with_tax 5.0 3 sales_tax +
  calculate_cost_with_tax 3.50 2 sales_tax +
  calculate_cost_with_tax 4.25 7 sales_tax +
  calculate_cost_with_tax 6.0 4 sales_tax +
  calculate_cost_with_tax 5.50 8 sales_tax

def total_second_trip : Float :=
  calculate_cost_with_discount 1.50 2 discount +
  calculate_cost_with_discount 2.75 5 discount

def remaining_money (initial: Float) (first_trip: Float) (second_trip: Float) : Float :=
  initial - first_trip - second_trip

theorem peter_remaining_money : remaining_money initial_amount total_first_trip total_second_trip = 297.24 := 
  by
    -- Proof omitted
    sorry

end peter_remaining_money_l42_42961


namespace lines_parallel_l42_42640

def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_parallel : ∀ x1 x2 : ℝ, l1 x1 = l2 x2 → false := 
by
  intros x1 x2 h
  rw [l1, l2] at h
  sorry

end lines_parallel_l42_42640


namespace complement_U_M_correct_l42_42729

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 4 * x + 3 = 0}
def complement_U_M : Set ℕ := U \ M

theorem complement_U_M_correct : complement_U_M = {2, 4} :=
by
  -- Proof will be provided here
  sorry

end complement_U_M_correct_l42_42729


namespace waiter_tables_l42_42322

theorem waiter_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  initial_customers = 62 → 
  customers_left = 17 → 
  people_per_table = 9 → 
  remaining_customers = initial_customers - customers_left →
  tables = remaining_customers / people_per_table →
  tables = 5 :=
by
  intros hinitial hleft hpeople hremaining htables
  rw [hinitial, hleft, hpeople] at *
  simp at *
  sorry

end waiter_tables_l42_42322


namespace cheryl_more_points_l42_42479

-- Define the number of each type of eggs each child found
def kevin_small_eggs : Nat := 5
def kevin_large_eggs : Nat := 3

def bonnie_small_eggs : Nat := 13
def bonnie_medium_eggs : Nat := 7
def bonnie_large_eggs : Nat := 2

def george_small_eggs : Nat := 9
def george_medium_eggs : Nat := 6
def george_large_eggs : Nat := 1

def cheryl_small_eggs : Nat := 56
def cheryl_medium_eggs : Nat := 30
def cheryl_large_eggs : Nat := 15

-- Define the points for each type of egg
def small_egg_points : Nat := 1
def medium_egg_points : Nat := 3
def large_egg_points : Nat := 5

-- Calculate the total points for each child
def kevin_points : Nat := kevin_small_eggs * small_egg_points + kevin_large_eggs * large_egg_points
def bonnie_points : Nat := bonnie_small_eggs * small_egg_points + bonnie_medium_eggs * medium_egg_points + bonnie_large_eggs * large_egg_points
def george_points : Nat := george_small_eggs * small_egg_points + george_medium_eggs * medium_egg_points + george_large_eggs * large_egg_points
def cheryl_points : Nat := cheryl_small_eggs * small_egg_points + cheryl_medium_eggs * medium_egg_points + cheryl_large_eggs * large_egg_points

-- Statement of the proof problem
theorem cheryl_more_points : cheryl_points - (kevin_points + bonnie_points + george_points) = 125 :=
by
  -- Here would go the proof steps
  sorry

end cheryl_more_points_l42_42479


namespace simplify_expression_l42_42825

theorem simplify_expression :
  64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 := 
by
  sorry

end simplify_expression_l42_42825


namespace distance_traveled_on_second_day_l42_42622

theorem distance_traveled_on_second_day 
  (a₁ : ℝ) 
  (h_sum : a₁ + a₁ / 2 + a₁ / 4 + a₁ / 8 + a₁ / 16 + a₁ / 32 = 189) 
  : a₁ / 2 = 48 :=
by
  sorry

end distance_traveled_on_second_day_l42_42622


namespace employee_payments_l42_42385

noncomputable def amount_paid_to_Y : ℝ := 934 / 3
noncomputable def amount_paid_to_X : ℝ := 1.20 * amount_paid_to_Y
noncomputable def amount_paid_to_Z : ℝ := 0.80 * amount_paid_to_Y

theorem employee_payments :
  amount_paid_to_X + amount_paid_to_Y + amount_paid_to_Z = 934 :=
by
  sorry

end employee_payments_l42_42385


namespace fraction_subtraction_l42_42470

theorem fraction_subtraction : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 :=
  sorry

end fraction_subtraction_l42_42470


namespace simplify_expression_l42_42878
theorem simplify_expression (c : ℝ) : 
    (3 * c + 6 - 6 * c) / 3 = -c + 2 := 
by 
    sorry

end simplify_expression_l42_42878


namespace smallest_number_with_2020_divisors_l42_42807

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l42_42807


namespace c_minus_a_value_l42_42337

theorem c_minus_a_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 :=
by 
  sorry

end c_minus_a_value_l42_42337


namespace sin_60_eq_sqrt3_div_2_l42_42851

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l42_42851


namespace people_attend_both_reunions_l42_42032

theorem people_attend_both_reunions (N D H x : ℕ) 
  (hN : N = 50)
  (hD : D = 50)
  (hH : H = 60)
  (h_total : N = D + H - x) : 
  x = 60 :=
by
  sorry

end people_attend_both_reunions_l42_42032


namespace find_power_y_l42_42779

theorem find_power_y 
  (y : ℕ) 
  (h : (12 : ℝ)^y * (6 : ℝ)^3 / (432 : ℝ) = 72) : 
  y = 2 :=
by
  sorry

end find_power_y_l42_42779


namespace three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l42_42189

theorem three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four :
  (3.242 * 12) / 100 = 0.38904 :=
by 
  sorry

end three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l42_42189


namespace cost_of_dog_l42_42223

-- Given conditions
def dollars_misha_has : ℕ := 34
def dollars_misha_needs_earn : ℕ := 13

-- Formal statement of the mathematic proof
theorem cost_of_dog : dollars_misha_has + dollars_misha_needs_earn = 47 := by
  sorry

end cost_of_dog_l42_42223


namespace net_progress_l42_42798

def lost_yards : Int := 5
def gained_yards : Int := 7

theorem net_progress : gained_yards - lost_yards = 2 := 
by
  sorry

end net_progress_l42_42798


namespace length_of_first_train_is_270_l42_42163

/-- 
Given:
1. Speed of the first train = 120 kmph
2. Speed of the second train = 80 kmph
3. Time to cross each other = 9 seconds
4. Length of the second train = 230.04 meters
  
Prove that the length of the first train is 270 meters.
-/
theorem length_of_first_train_is_270
  (speed_first_train : ℝ := 120)
  (speed_second_train : ℝ := 80)
  (time_to_cross : ℝ := 9)
  (length_second_train : ℝ := 230.04)
  (conversion_factor : ℝ := 1000/3600) :
  (length_second_train + (speed_first_train + speed_second_train) * conversion_factor * time_to_cross - length_second_train) = 270 :=
by
  sorry

end length_of_first_train_is_270_l42_42163


namespace train_length_l42_42552

def relative_speed (v_fast v_slow : ℕ) : ℚ :=
  v_fast - v_slow

def convert_speed (speed : ℚ) : ℚ :=
  (speed * 1000) / 3600

def covered_distance (speed : ℚ) (time_seconds : ℚ) : ℚ :=
  speed * time_seconds

theorem train_length (L : ℚ) (v_fast v_slow : ℕ) (time_seconds : ℚ)
    (hf : v_fast = 42) (hs : v_slow = 36) (ht : time_seconds = 36)
    (hc : relative_speed v_fast v_slow * 1000 / 3600 * time_seconds = 2 * L) :
    L = 30 := by
  sorry

end train_length_l42_42552


namespace find_number_l42_42909

-- Define the condition
def condition : Prop := ∃ x : ℝ, x / 0.02 = 50

-- State the theorem to prove
theorem find_number (x : ℝ) (h : x / 0.02 = 50) : x = 1 :=
sorry

end find_number_l42_42909


namespace paintable_fence_l42_42895

theorem paintable_fence :
  ∃ h t u : ℕ,  h > 1 ∧ t > 1 ∧ u > 1 ∧ 
  (∀ n, 4 + (n * h) ≠ 5 + (m * (2 * t))) ∧
  (∀ n, 4 + (n * h) ≠ 6 + (l * (3 * u))) ∧ 
  (∀ m l, 5 + (m * (2 * t)) ≠ 6 + (l * (3 * u))) ∧
  (100 * h + 20 * t + 2 * u = 390) :=
by 
  sorry

end paintable_fence_l42_42895


namespace distinct_ordered_pairs_proof_l42_42992

def num_distinct_ordered_pairs_satisfying_reciprocal_sum : ℕ :=
  List.length [
    (7, 42), (8, 24), (9, 18), (10, 15), 
    (12, 12), (15, 10), (18, 9), (24, 8), 
    (42, 7)
  ]

theorem distinct_ordered_pairs_proof : num_distinct_ordered_pairs_satisfying_reciprocal_sum = 9 := by
  sorry

end distinct_ordered_pairs_proof_l42_42992


namespace trig_identity_solutions_l42_42636

open Real

theorem trig_identity_solutions (x : ℝ) (k n : ℤ) :
  (4 * sin x * cos (π / 2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3 * π / 2 - x) * cos (π + x) = 1) ↔ 
  (∃ k : ℤ, x = arctan (1 / 3) + π * k) ∨ (∃ n : ℤ, x = π / 4 + π * n) := 
sorry

end trig_identity_solutions_l42_42636


namespace cost_of_4_stamps_l42_42738

theorem cost_of_4_stamps (cost_per_stamp : ℕ) (h : cost_per_stamp = 34) : 4 * cost_per_stamp = 136 :=
by
  sorry

end cost_of_4_stamps_l42_42738


namespace quadratic_real_roots_leq_l42_42164

theorem quadratic_real_roots_leq (m : ℝ) :
  ∃ x : ℝ, x^2 - 3 * x + 2 * m = 0 → m ≤ 9 / 8 :=
by
  sorry

end quadratic_real_roots_leq_l42_42164


namespace no_triangle_satisfies_sine_eq_l42_42625

theorem no_triangle_satisfies_sine_eq (A B C : ℝ) (a b c : ℝ) 
  (hA: 0 < A) (hB: 0 < B) (hC: 0 < C) 
  (hA_ineq: A < π) (hB_ineq: B < π) (hC_ineq: C < π) 
  (h_sum: A + B + C = π) 
  (sin_eq: Real.sin A + Real.sin B = Real.sin C)
  (h_tri_ineq: a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_sines: a = 2 * (1) * Real.sin A ∧ b = 2 * (1) * Real.sin B ∧ c = 2 * (1) * Real.sin C) :
  False :=
sorry

end no_triangle_satisfies_sine_eq_l42_42625


namespace simplify_fraction_l42_42482

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end simplify_fraction_l42_42482


namespace window_ratio_area_l42_42242

/-- Given a rectangle with semicircles at either end, if the ratio of AD to AB is 3:2,
    and AB is 30 inches, then the ratio of the area of the rectangle to the combined 
    area of the semicircles is 6 : π. -/
theorem window_ratio_area (AD AB r : ℝ) (h1 : AB = 30) (h2 : AD / AB = 3 / 2) (h3 : r = AB / 2) :
    (AD * AB) / (π * r^2) = 6 / π :=
by
  sorry

end window_ratio_area_l42_42242


namespace range_of_m_l42_42126

-- Define the first circle
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y + 1 = 0

-- Define the second circle
def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - m = 0

-- Lean statement for the proof problem
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by sorry

end range_of_m_l42_42126


namespace find_capacity_of_second_vessel_l42_42175

noncomputable def capacity_of_second_vessel (x : ℝ) : Prop :=
  let alcohol_from_first_vessel := 0.25 * 2
  let alcohol_from_second_vessel := 0.40 * x
  let total_liquid := 2 + x
  let total_alcohol := alcohol_from_first_vessel + alcohol_from_second_vessel
  let new_concentration := (total_alcohol / 10) * 100
  2 + x = 8 ∧ new_concentration = 29

open scoped Real

theorem find_capacity_of_second_vessel : ∃ x : ℝ, capacity_of_second_vessel x ∧ x = 6 :=
by
  sorry

end find_capacity_of_second_vessel_l42_42175


namespace volume_not_determined_l42_42068

noncomputable def tetrahedron_volume_not_unique 
  (area1 area2 area3 : ℝ) (circumradius : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∃ a' b' c', 
      (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧ 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)))

theorem volume_not_determined 
  (area1 area2 area3 circumradius: ℝ) 
  (h: tetrahedron_volume_not_unique area1 area2 area3 circumradius) : 
  ¬ ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∀ a' b' c', 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)) → 
      (a = a' ∧ b = b' ∧ c = c')) := 
by sorry

end volume_not_determined_l42_42068


namespace courtyard_length_l42_42260

theorem courtyard_length 
  (stone_area : ℕ) 
  (stones_total : ℕ) 
  (width : ℕ)
  (total_area : ℕ) 
  (L : ℕ) 
  (h1 : stone_area = 4)
  (h2 : stones_total = 135)
  (h3 : width = 18)
  (h4 : total_area = stones_total * stone_area)
  (h5 : total_area = L * width) :
  L = 30 :=
by
  -- Proof steps would go here
  sorry

end courtyard_length_l42_42260


namespace unique_bijective_function_l42_42081

noncomputable def find_bijective_function {n : ℕ}
  (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ)
  (f : Fin n → ℝ) : Prop :=
∀ i : Fin n, f i = x i

theorem unique_bijective_function (n : ℕ) (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ) (f : Fin n → ℝ)
  (hf_bij : Function.Bijective f)
  (h_abs_diff : ∀ i, |f i - x i| = 0) : find_bijective_function hn hodd x f :=
by
  sorry

end unique_bijective_function_l42_42081


namespace subset_implies_range_of_a_l42_42047

theorem subset_implies_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → x > a) → a < -2 :=
by
  intro h
  sorry

end subset_implies_range_of_a_l42_42047


namespace minimum_kinds_of_candies_l42_42501

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l42_42501


namespace find_functions_l42_42365

-- Define the function f and its properties.
variable {f : ℝ → ℝ}

-- Define the condition given in the problem as a hypothesis.
def condition (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f x + f y) = y + f x ^ 2

-- State the theorem we want to prove.
theorem find_functions (hf : condition f) : (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
  sorry

end find_functions_l42_42365


namespace find_abc_pairs_l42_42581

theorem find_abc_pairs :
  ∀ (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧ (a-1)*(b-1)*(c-1) ∣ a*b*c - 1 → 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  -- Proof omitted
  sorry

end find_abc_pairs_l42_42581


namespace complex_magnitude_add_reciprocals_l42_42874

open Complex

theorem complex_magnitude_add_reciprocals
  (z w : ℂ)
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hz_plus_w : Complex.abs (z + w) = 6) :
  Complex.abs (1 / z + 1 / w) = 3 / 4 := by
  sorry

end complex_magnitude_add_reciprocals_l42_42874


namespace find_number_l42_42142

theorem find_number (x : ℝ) (h : 97 * x - 89 * x = 4926) : x = 615.75 :=
by
  sorry

end find_number_l42_42142


namespace probability_all_calls_same_probability_two_calls_for_A_l42_42525

theorem probability_all_calls_same (pA pB pC : ℚ) (hA : pA = 1/6) (hB : pB = 1/3) (hC : pC = 1/2) :
  (pA^3 + pB^3 + pC^3) = 1/6 :=
by
  sorry

theorem probability_two_calls_for_A (pA : ℚ) (hA : pA = 1/6) :
  (3 * (pA^2) * (5/6)) = 5/72 :=
by
  sorry

end probability_all_calls_same_probability_two_calls_for_A_l42_42525


namespace serves_probability_l42_42149

variable (p : ℝ) (hpos : 0 < p) (hneq0 : p ≠ 0)

def ExpectedServes (p : ℝ) : ℝ :=
  p + 2 * p * (1 - p) + 3 * (1 - p) ^ 2

theorem serves_probability (h : ExpectedServes p > 1.75) : 0 < p ∧ p < 1 / 2 :=
  sorry

end serves_probability_l42_42149


namespace combined_average_speed_l42_42418

-- Definitions based on conditions
def distance_A : ℕ := 250
def time_A : ℕ := 4

def distance_B : ℕ := 480
def time_B : ℕ := 6

def distance_C : ℕ := 390
def time_C : ℕ := 5

def total_distance : ℕ := distance_A + distance_B + distance_C
def total_time : ℕ := time_A + time_B + time_C

-- Prove combined average speed
theorem combined_average_speed : (total_distance : ℚ) / (total_time : ℚ) = 74.67 :=
  by
    sorry

end combined_average_speed_l42_42418


namespace train_b_speed_l42_42237

/-- Two trains, A and B, start simultaneously from two stations 480 kilometers apart and meet after 2.5 hours. 
Train A travels at a speed of 102 kilometers per hour. What is the speed of train B in kilometers per hour? -/
theorem train_b_speed (d t : ℝ) (speedA speedB : ℝ) (h1 : d = 480) (h2 : t = 2.5) (h3 : speedA = 102)
  (h4 : speedA * t + speedB * t = d) : speedB = 90 := 
by
  sorry

end train_b_speed_l42_42237


namespace max_consecutive_integers_sum_48_l42_42077

-- Define the sum of consecutive integers
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Define the main theorem
theorem max_consecutive_integers_sum_48 : 
  ∃ N a : ℤ, sum_consecutive_integers a N = 48 ∧ (∀ N' : ℤ, ((N' * (2 * a + N' - 1)) / 2 = 48) → N' ≤ N) :=
sorry

end max_consecutive_integers_sum_48_l42_42077


namespace part1_part2_l42_42741

-- Define A and B according to given expressions
def A (a b : ℚ) : ℚ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) : ℚ := -a^2 + a * b - 1

-- Prove the first statement
theorem part1 (a b : ℚ) : 4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 :=
by sorry

-- Prove the second statement
theorem part2 (F : ℚ) (b : ℚ) : (∀ a, A a b + 2 * B a b = F) → b = 2 / 5 :=
by sorry

end part1_part2_l42_42741


namespace greatest_two_digit_with_product_9_l42_42659

theorem greatest_two_digit_with_product_9 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ a b : ℕ, n = 10 * a + b ∧ a * b = 9) ∧ (∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ (∃ c d : ℕ, m = 10 * c + d ∧ c * d = 9) → m ≤ 91) :=
by
  sorry

end greatest_two_digit_with_product_9_l42_42659


namespace find_x_l42_42657

variables (t x : ℕ)

theorem find_x (h1 : 0 < t) (h2 : t = 4) (h3 : ((9 / 10 : ℚ) * (t * x : ℚ)) - 6 = 48) : x = 15 :=
by
  sorry

end find_x_l42_42657


namespace min_value_eval_l42_42835

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100)

theorem min_value_eval (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x = y → min_value_expr x y = -2500 :=
by
  intros hxy
  -- Insert proof steps here
  sorry

end min_value_eval_l42_42835


namespace jeans_cost_l42_42117

-- Definitions based on conditions
def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def total_cost : ℕ := 51
def n_shirts : ℕ := 3
def n_hats : ℕ := 4
def n_jeans : ℕ := 2

-- The goal is to prove that the cost of one pair of jeans (J) is 10
theorem jeans_cost (J : ℕ) (h : n_shirts * shirt_cost + n_jeans * J + n_hats * hat_cost = total_cost) : J = 10 :=
  sorry

end jeans_cost_l42_42117


namespace gcd_of_sum_of_cubes_and_increment_l42_42940

theorem gcd_of_sum_of_cubes_and_increment {n : ℕ} (h : n > 3) : Nat.gcd (n^3 + 27) (n + 4) = 1 :=
by sorry

end gcd_of_sum_of_cubes_and_increment_l42_42940


namespace solve_equation1_solve_equation2_l42_42326

noncomputable def solutions_equation1 : Set ℝ := { x | x^2 - 2 * x - 8 = 0 }
noncomputable def solutions_equation2 : Set ℝ := { x | x^2 - 2 * x - 5 = 0 }

theorem solve_equation1 :
  solutions_equation1 = {4, -2} := 
by
  sorry

theorem solve_equation2 :
  solutions_equation2 = {1 + Real.sqrt 6, 1 - Real.sqrt 6} :=
by
  sorry

end solve_equation1_solve_equation2_l42_42326


namespace linear_regression_neg_corr_l42_42492

-- Given variables x and y with certain properties
variables (x y : ℝ)

-- Conditions provided in the problem
def neg_corr (x y : ℝ) : Prop := ∀ a b : ℝ, (a < b → x * y < 0)
def sample_mean_x := (2 : ℝ)
def sample_mean_y := (1.5 : ℝ)

-- Statement to prove the linear regression equation
theorem linear_regression_neg_corr (h1 : neg_corr x y)
    (hx : sample_mean_x = 2)
    (hy : sample_mean_y = 1.5) : 
    ∃ b0 b1 : ℝ, b0 = 5.5 ∧ b1 = -2 ∧ y = b0 + b1 * x :=
sorry

end linear_regression_neg_corr_l42_42492


namespace total_cards_l42_42707

theorem total_cards (Brenda_card Janet_card Mara_card Michelle_card : ℕ)
  (h1 : Janet_card = Brenda_card + 9)
  (h2 : Mara_card = 7 * Janet_card / 4)
  (h3 : Michelle_card = 4 * Mara_card / 5)
  (h4 : Mara_card = 210 - 60) :
  Janet_card + Brenda_card + Mara_card + Michelle_card = 432 :=
by
  sorry

end total_cards_l42_42707


namespace remainder_when_x_squared_divided_by_20_l42_42519

theorem remainder_when_x_squared_divided_by_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] :=
sorry

end remainder_when_x_squared_divided_by_20_l42_42519


namespace primes_infinite_l42_42558

theorem primes_infinite : ∀ (S : Set ℕ), (∀ p, p ∈ S → Nat.Prime p) → (∃ a, a ∉ S ∧ Nat.Prime a) :=
by
  sorry

end primes_infinite_l42_42558


namespace total_expenditure_l42_42051

-- Define the conditions
def cost_per_acre : ℕ := 20
def acres_bought : ℕ := 30
def house_cost : ℕ := 120000
def cost_per_cow : ℕ := 1000
def cows_bought : ℕ := 20
def cost_per_chicken : ℕ := 5
def chickens_bought : ℕ := 100
def hourly_installation_cost : ℕ := 100
def installation_hours : ℕ := 6
def solar_equipment_cost : ℕ := 6000

-- Define the total cost breakdown
def land_cost : ℕ := cost_per_acre * acres_bought
def cows_cost : ℕ := cost_per_cow * cows_bought
def chickens_cost : ℕ := cost_per_chicken * chickens_bought
def solar_installation_cost : ℕ := (hourly_installation_cost * installation_hours) + solar_equipment_cost

-- Define the total cost
def total_cost : ℕ :=
  land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost

-- The theorem statement
theorem total_expenditure : total_cost = 147700 :=
by
  -- Proof steps would go here
  sorry

end total_expenditure_l42_42051


namespace time_to_run_above_tree_l42_42049

-- Defining the given conditions
def tiger_length : ℕ := 5
def tree_trunk_length : ℕ := 20
def time_to_pass_grass : ℕ := 1

-- Defining the speed of the tiger
def tiger_speed : ℕ := tiger_length / time_to_pass_grass

-- Defining the total distance the tiger needs to run
def total_distance : ℕ := tree_trunk_length + tiger_length

-- The theorem stating the time it takes for the tiger to run above the fallen tree trunk
theorem time_to_run_above_tree :
  (total_distance / tiger_speed) = 5 :=
by
  -- Trying to fit the solution steps as formal Lean statements
  sorry

end time_to_run_above_tree_l42_42049


namespace pages_printed_l42_42139

theorem pages_printed (P : ℕ) 
  (H1 : P % 7 = 0)
  (H2 : P % 3 = 0)
  (H3 : P - (P / 7 + P / 3 - P / 21) = 24) : 
  P = 42 :=
sorry

end pages_printed_l42_42139


namespace spanish_peanuts_l42_42225

variable (x : ℝ)

theorem spanish_peanuts :
  (10 * 3.50 + x * 3.00 = (10 + x) * 3.40) → x = 2.5 :=
by
  intro h
  sorry

end spanish_peanuts_l42_42225


namespace vector_sum_correct_l42_42633

def vec1 : Fin 3 → ℤ := ![-7, 3, 5]
def vec2 : Fin 3 → ℤ := ![4, -1, -6]
def vec3 : Fin 3 → ℤ := ![1, 8, 2]
def expectedSum : Fin 3 → ℤ := ![-2, 10, 1]

theorem vector_sum_correct :
  (fun i => vec1 i + vec2 i + vec3 i) = expectedSum := 
by
  sorry

end vector_sum_correct_l42_42633


namespace teresa_jogged_distance_l42_42136

-- Define the conditions as Lean constants.
def teresa_speed : ℕ := 5 -- Speed in kilometers per hour
def teresa_time : ℕ := 5 -- Time in hours

-- Define the distance formula.
def teresa_distance (speed time : ℕ) : ℕ := speed * time

-- State the theorem.
theorem teresa_jogged_distance : teresa_distance teresa_speed teresa_time = 25 := by
  -- Proof is skipped using 'sorry'.
  sorry

end teresa_jogged_distance_l42_42136


namespace work_problem_l42_42436

theorem work_problem (W : ℕ) (h1: ∀ w, w = W → (24 * w + 1 = 73)) : W = 3 :=
by {
  -- Insert proof here
  sorry
}

end work_problem_l42_42436


namespace am_gm_inequality_l42_42726

theorem am_gm_inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := 
by
  sorry

end am_gm_inequality_l42_42726


namespace sum_of_other_endpoint_l42_42570

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l42_42570


namespace denominator_of_second_fraction_l42_42069

theorem denominator_of_second_fraction :
  let a := 2007
  let b := 2999
  let c := 8001
  let d := 2001
  let e := 3999
  let sum := 3.0035428163476343
  let first_fraction := (2007 : ℝ) / 2999
  let third_fraction := (2001 : ℝ) / 3999
  ∃ x : ℤ, (first_fraction + (8001 : ℝ) / x + third_fraction) = 3.0035428163476343 ∧ x = 4362 := 
by
  sorry

end denominator_of_second_fraction_l42_42069


namespace min_value_of_reciprocal_sum_l42_42768

theorem min_value_of_reciprocal_sum {a b : ℝ} (h : a > 0 ∧ b > 0)
  (h_circle1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (h_circle2 : ∀ x y : ℝ, (x - 2)^2 + (y - 2)^2 = 4)
  (h_common_chord : a + b = 2) :
  (1 / a + 9 / b = 8) := 
sorry

end min_value_of_reciprocal_sum_l42_42768


namespace faster_speed_l42_42628

variable (v : ℝ)
variable (distance fasterDistance speed time : ℝ)
variable (h_distance : distance = 24)
variable (h_speed : speed = 4)
variable (h_fasterDistance : fasterDistance = distance + 6)
variable (h_time : time = distance / speed)

theorem faster_speed (h : 6 = fasterDistance / v) : v = 5 :=
by
  sorry

end faster_speed_l42_42628


namespace DE_value_l42_42524

theorem DE_value {AG GF FC HJ DE : ℝ} (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : DE = 2 * Real.sqrt 22 :=
sorry

end DE_value_l42_42524


namespace number_of_strawberry_cakes_l42_42563

def number_of_chocolate_cakes := 3
def price_of_chocolate_cake := 12
def price_of_strawberry_cake := 22
def total_payment := 168

theorem number_of_strawberry_cakes (S : ℕ) : 
    number_of_chocolate_cakes * price_of_chocolate_cake + S * price_of_strawberry_cake = total_payment → 
    S = 6 :=
by
  sorry

end number_of_strawberry_cakes_l42_42563


namespace phi_value_l42_42993

theorem phi_value (phi : ℝ) (h : 0 < phi ∧ phi < π) 
  (hf : ∀ x : ℝ, 3 * Real.sin (2 * abs x - π / 3 + phi) = 3 * Real.sin (2 * x - π / 3 + phi)) 
  : φ = 5 * π / 6 :=
by 
  sorry

end phi_value_l42_42993


namespace sum_of_repeating_decimals_l42_42850

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l42_42850


namespace radius_of_smaller_molds_l42_42665

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  (64 * hemisphere_volume (1/2)) = hemisphere_volume 2 :=
by
  sorry

end radius_of_smaller_molds_l42_42665


namespace value_of_c_l42_42649

theorem value_of_c (c : ℝ) : (∃ x : ℝ, x^2 + c * x - 36 = 0 ∧ x = -9) → c = 5 :=
by
  sorry

end value_of_c_l42_42649


namespace Jenna_total_cost_l42_42724

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end Jenna_total_cost_l42_42724


namespace center_of_circle_l42_42810

theorem center_of_circle (ρ θ : ℝ) (h : ρ = 2 * Real.cos (θ - π / 4)) : (ρ, θ) = (1, π / 4) :=
sorry

end center_of_circle_l42_42810


namespace train_speed_in_kmph_l42_42102

noncomputable def motorbike_speed : ℝ := 64
noncomputable def overtaking_time : ℝ := 40
noncomputable def train_length_meters : ℝ := 400.032

theorem train_speed_in_kmph :
  let train_length_km := train_length_meters / 1000
  let overtaking_time_hours := overtaking_time / 3600
  let relative_speed := train_length_km / overtaking_time_hours
  let train_speed := motorbike_speed + relative_speed
  train_speed = 100.00288 := by
  sorry

end train_speed_in_kmph_l42_42102


namespace polynomial_factors_l42_42169

theorem polynomial_factors (t q : ℤ) (h1 : 81 - 3 * t + q = 0) (h2 : -3 + t + q = 0) : |3 * t - 2 * q| = 99 :=
sorry

end polynomial_factors_l42_42169


namespace sqrt_expression_l42_42931

open Real

theorem sqrt_expression :
  3 * sqrt 12 / (3 * sqrt (1 / 3)) - 2 * sqrt 3 = 6 - 2 * sqrt 3 :=
by
  sorry

end sqrt_expression_l42_42931


namespace factorial_division_l42_42007

theorem factorial_division (n : ℕ) (h : n = 9) : n.factorial / (n - 1).factorial = 9 :=
by 
  rw [h]
  sorry

end factorial_division_l42_42007


namespace women_more_than_men_l42_42419

def men (W : ℕ) : ℕ := (5 * W) / 11

theorem women_more_than_men (M W : ℕ) (h1 : M + W = 16) (h2 : M = (5 * W) / 11) : W - M = 6 :=
by
  sorry

end women_more_than_men_l42_42419


namespace no_solution_range_of_a_l42_42949

theorem no_solution_range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) → a ≤ 8 :=
by
  sorry

end no_solution_range_of_a_l42_42949


namespace transformed_cube_edges_l42_42516

-- Let's define the problem statement
theorem transformed_cube_edges : 
  let original_edges := 12 
  let new_edges_per_edge := 2 
  let additional_edges_per_pyramid := 1 
  let total_edges := original_edges + (original_edges * new_edges_per_edge) + (original_edges * additional_edges_per_pyramid) 
  total_edges = 48 :=
by sorry

end transformed_cube_edges_l42_42516


namespace ratio_of_b_to_a_l42_42813

theorem ratio_of_b_to_a (a b c : ℕ) (x y : ℕ) 
  (h1 : a > 0) 
  (h2 : x = 100 * a + 10 * b + c)
  (h3 : y = 100 * 9 + 10 * 9 + 9 - 241) 
  (h4 : x = y) :
  b = 5 → a = 7 → (b / a : ℚ) = 5 / 7 := 
by
  intros
  subst_vars
  sorry

end ratio_of_b_to_a_l42_42813


namespace jake_snake_length_l42_42706

theorem jake_snake_length (j p : ℕ) (h1 : j = p + 12) (h2 : j + p = 70) : j = 41 := by
  sorry

end jake_snake_length_l42_42706


namespace purchase_price_mobile_l42_42919

-- Definitions of the given conditions
def purchase_price_refrigerator : ℝ := 15000
def loss_percent_refrigerator : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10
def overall_profit : ℝ := 50

-- Defining the statement to prove
theorem purchase_price_mobile (P : ℝ)
  (h1 : purchase_price_refrigerator = 15000)
  (h2 : loss_percent_refrigerator = 0.05)
  (h3 : profit_percent_mobile = 0.10)
  (h4 : overall_profit = 50) :
  (15000 * (1 - 0.05) + P * (1 + 0.10)) - (15000 + P) = 50 → P = 8000 :=
by {
  -- Proof is omitted
  sorry
}

end purchase_price_mobile_l42_42919


namespace circle_radius_and_circumference_l42_42843

theorem circle_radius_and_circumference (A : ℝ) (hA : A = 64 * Real.pi) :
  ∃ r C : ℝ, r = 8 ∧ C = 2 * Real.pi * r :=
by
  -- statement ensures that with given area A, you can find r and C satisfying the conditions.
  sorry

end circle_radius_and_circumference_l42_42843


namespace monotonic_increasing_iff_monotonic_decreasing_on_interval_l42_42745

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 - a * x - 1

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ a ≤ 0 :=
by 
  sorry

theorem monotonic_decreasing_on_interval (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → ∀ y : ℝ, -1 < y ∧ y < 1 → x < y → f y a < f x a) ↔ 3 ≤ a :=
by 
  sorry

end monotonic_increasing_iff_monotonic_decreasing_on_interval_l42_42745


namespace find_x_for_opposite_directions_l42_42951

-- Define the vectors and the opposite direction condition
def vector_a (x : ℝ) : ℝ × ℝ := (1, -x)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -16)

-- Define the condition that vectors are in opposite directions
def opp_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (-k) • b

-- The main theorem statement
theorem find_x_for_opposite_directions : ∃ x : ℝ, opp_directions (vector_a x) (vector_b x) ∧ x = -5 := 
sorry

end find_x_for_opposite_directions_l42_42951


namespace jessica_journey_total_distance_l42_42586

theorem jessica_journey_total_distance
  (y : ℝ)
  (h1 : y = (y / 4) + 25 + (y / 4)) :
  y = 50 :=
by
  sorry

end jessica_journey_total_distance_l42_42586


namespace circles_intersect_if_and_only_if_l42_42631

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6 * x - 8 * y - 11 = 0) ↔ (1 < m ∧ m < 121) :=
by
  sorry

end circles_intersect_if_and_only_if_l42_42631


namespace minimal_coach_handshakes_l42_42833

theorem minimal_coach_handshakes (n k1 k2 : ℕ) (h1 : k1 < n) (h2 : k2 < n)
  (hn : (n * (n - 1)) / 2 + k1 + k2 = 300) : k1 + k2 = 0 := by
  sorry

end minimal_coach_handshakes_l42_42833


namespace quadratic_expression_value_l42_42699

theorem quadratic_expression_value :
  ∀ x1 x2 : ℝ, (x1^2 - 4 * x1 - 2020 = 0) ∧ (x2^2 - 4 * x2 - 2020 = 0) →
  (x1^2 - 2 * x1 + 2 * x2 = 2028) :=
by
  intros x1 x2 h
  sorry

end quadratic_expression_value_l42_42699


namespace area_of_triangle_BCD_l42_42409

-- Define the points A, B, C, D
variables {A B C D : Type} 

-- Define the lengths of segments AC and CD
variables (AC CD : ℝ)
-- Define the area of triangle ABC
variables (area_ABC : ℝ)

-- Define height h
variables (h : ℝ)

-- Initial conditions
axiom length_AC : AC = 9
axiom length_CD : CD = 39
axiom area_ABC_is_36 : area_ABC = 36
axiom height_is_8 : h = (2 * area_ABC) / AC

-- Define the area of triangle BCD
def area_BCD (CD h : ℝ) : ℝ := 0.5 * CD * h

-- The theorem that we want to prove
theorem area_of_triangle_BCD : area_BCD 39 8 = 156 :=
by
  sorry

end area_of_triangle_BCD_l42_42409


namespace find_x_l42_42739

-- conditions
variable (k : ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- proportional relationship
def proportional_relationship (k x y z : ℝ) : Prop := 
  x = (k * y^2) / z

-- initial conditions
def initial_conditions (k : ℝ) : Prop := 
  proportional_relationship k 6 1 3

-- prove x = 24 when y = 2 and z = 3 under given conditions
theorem find_x (k : ℝ) (h : initial_conditions k) : 
  proportional_relationship k 24 2 3 :=
sorry

end find_x_l42_42739


namespace find_direction_vector_l42_42490

def line_parametrization (v d : ℝ × ℝ) (t x y : ℝ) : ℝ × ℝ :=
  (v.fst + t * d.fst, v.snd + t * d.snd)

theorem find_direction_vector : 
  ∀ d: ℝ × ℝ, ∀ t: ℝ,
    ∀ (v : ℝ × ℝ) (x y : ℝ), 
    v = (-3, -1) → 
    y = (2 * x + 3) / 5 →
    x + 3 ≤ 0 →
    dist (line_parametrization v d t x y) (-3, -1) = t →
    d = (5/2, 1) :=
by
  intros d t v x y hv hy hcond hdist
  sorry

end find_direction_vector_l42_42490


namespace find_4digit_number_l42_42999

theorem find_4digit_number (a b c d n n' : ℕ) :
  n = 1000 * a + 100 * b + 10 * c + d →
  n' = 1000 * d + 100 * c + 10 * b + a →
  n = n' - 7182 →
  n = 1909 :=
by
  intros h1 h2 h3
  sorry

end find_4digit_number_l42_42999


namespace meals_distinct_pairs_l42_42697

theorem meals_distinct_pairs :
  let entrees := 4
  let drinks := 3
  let desserts := 3
  let total_meals := entrees * drinks * desserts
  total_meals * (total_meals - 1) = 1260 :=
by 
  sorry

end meals_distinct_pairs_l42_42697


namespace total_history_and_maths_l42_42064

-- Defining the conditions
def total_students : ℕ := 25
def fraction_like_maths : ℚ := 2 / 5
def fraction_like_science : ℚ := 1 / 3

-- Theorem statement
theorem total_history_and_maths : (total_students * fraction_like_maths + (total_students * (1 - fraction_like_maths) * (1 - fraction_like_science))) = 20 := by
  sorry

end total_history_and_maths_l42_42064


namespace cos_alpha_plus_pi_over_4_l42_42388

theorem cos_alpha_plus_pi_over_4
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = 3 / 5)
  (h4 : Real.sin (β - π / 4) = 5 / 13) : 
  Real.cos (α + π / 4) = 56 / 65 :=
by
  sorry 

end cos_alpha_plus_pi_over_4_l42_42388


namespace solve_for_y_l42_42911

theorem solve_for_y (y : ℤ) : 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) → y = -37 :=
by
  intro h
  sorry

end solve_for_y_l42_42911


namespace Malcom_has_more_cards_l42_42401

-- Define the number of cards Brandon has
def Brandon_cards : ℕ := 20

-- Define the number of cards Malcom has initially, to be found
def Malcom_initial_cards (n : ℕ) := n

-- Define the given condition: Malcom has 14 cards left after giving away half of his cards
def Malcom_half_condition (n : ℕ) := n / 2 = 14

-- Prove that Malcom had 8 more cards than Brandon initially
theorem Malcom_has_more_cards (n : ℕ) (h : Malcom_half_condition n) :
  Malcom_initial_cards n - Brandon_cards = 8 :=
by
  sorry

end Malcom_has_more_cards_l42_42401


namespace will_buy_toys_l42_42024

theorem will_buy_toys : 
  ∀ (initialMoney spentMoney toyCost : ℕ), 
  initialMoney = 83 → spentMoney = 47 → toyCost = 4 → 
  (initialMoney - spentMoney) / toyCost = 9 :=
by
  intros initialMoney spentMoney toyCost hInit hSpent hCost
  sorry

end will_buy_toys_l42_42024


namespace sum_of_consecutive_evens_l42_42615

/-- 
  Prove that the sum of five consecutive even integers 
  starting from 2n, with a common difference of 2, is 10n + 20.
-/
theorem sum_of_consecutive_evens (n : ℕ) :
  (2 * n) + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 10 * n + 20 := 
by
  sorry

end sum_of_consecutive_evens_l42_42615


namespace sequence_problem_l42_42017

theorem sequence_problem (S : ℕ → ℚ) (a : ℕ → ℚ) (h : ∀ n, S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ 
  (∀ n : ℕ, n > 0 → a n = (2^n - 1) / 2^(n-1)) :=
by
  sorry

end sequence_problem_l42_42017


namespace increase_in_average_l42_42782

theorem increase_in_average {a1 a2 a3 a4 : ℕ} 
                            (h1 : a1 = 92) 
                            (h2 : a2 = 89) 
                            (h3 : a3 = 91) 
                            (h4 : a4 = 93) : 
    ((a1 + a2 + a3 + a4 : ℚ) / 4) - ((a1 + a2 + a3 : ℚ) / 3) = 0.58 := 
by
  sorry

end increase_in_average_l42_42782


namespace sets_are_equal_l42_42791

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l42_42791


namespace sum_n_k_of_binomial_coefficient_ratio_l42_42727

theorem sum_n_k_of_binomial_coefficient_ratio :
  ∃ (n k : ℕ), (n = (7 * k + 5) / 2) ∧ (2 * (n - k) = 5 * (k + 1)) ∧ 
    ((k % 2 = 1) ∧ (n + k = 7 ∨ n + k = 16)) ∧ (23 = 7 + 16) :=
by
  sorry

end sum_n_k_of_binomial_coefficient_ratio_l42_42727


namespace mary_rental_hours_l42_42714

-- Definitions of the given conditions
def fixed_fee : ℝ := 17
def hourly_rate : ℝ := 7
def total_paid : ℝ := 80

-- Goal: Prove that the number of hours Mary paid for is 9
theorem mary_rental_hours : (total_paid - fixed_fee) / hourly_rate = 9 := 
by
  sorry

end mary_rental_hours_l42_42714


namespace double_root_values_l42_42912

theorem double_root_values (c : ℝ) :
  (∃ a : ℝ, (a^5 - 5 * a + c = 0) ∧ (5 * a^4 - 5 = 0)) ↔ (c = 4 ∨ c = -4) :=
by
  sorry

end double_root_values_l42_42912


namespace total_students_in_class_l42_42027

theorem total_students_in_class (F G B N T : ℕ)
  (hF : F = 41)
  (hG : G = 22)
  (hB : B = 9)
  (hN : N = 15)
  (hT : T = (F + G - B) + N) :
  T = 69 :=
by
  -- This is a theorem statement, proof is intentionally omitted.
  sorry

end total_students_in_class_l42_42027


namespace find_a_l42_42474

theorem find_a (a : ℝ) (h : -1 ^ 2 + 2 * -1 + a = 0) : a = 1 :=
sorry

end find_a_l42_42474


namespace carrie_expected_strawberries_l42_42309

noncomputable def calculate_strawberries (base height : ℝ) (plants_per_sq_ft strawberries_per_plant : ℝ) : ℝ :=
  let area := (1/2) * base * height
  let total_plants := plants_per_sq_ft * area
  total_plants * strawberries_per_plant

theorem carrie_expected_strawberries : calculate_strawberries 10 12 5 8 = 2400 :=
by
  /-
  Given: base = 10, height = 12, plants_per_sq_ft = 5, strawberries_per_plant = 8
  - calculate the area of the right triangle garden
  - calculate the total number of plants
  - calculate the total number of strawberries
  -/
  sorry

end carrie_expected_strawberries_l42_42309


namespace jack_initial_checked_plates_l42_42305

-- Define Jack's initial and resultant plate counts
variable (C : Nat)
variable (initial_flower_plates : Nat := 4)
variable (broken_flower_plates : Nat := 1)
variable (polka_dotted_plates := 2 * C)
variable (total_plates : Nat := 27)

-- Statement of the problem
theorem jack_initial_checked_plates (h_eq : 3 + C + 2 * C = total_plates) : C = 8 :=
by
  sorry

end jack_initial_checked_plates_l42_42305


namespace tangent_line_at_x_5_l42_42485

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_x_5 :
  (∀ x, f x = -x + 8 → f 5 + deriv f 5 = 2) := sorry

end tangent_line_at_x_5_l42_42485


namespace xiaomin_house_position_l42_42422

-- Define the initial position of the school at the origin
def school_pos : ℝ × ℝ := (0, 0)

-- Define the movement east and south from the school's position
def xiaomin_house_pos (east_distance south_distance : ℝ) : ℝ × ℝ :=
  (school_pos.1 + east_distance, school_pos.2 - south_distance)

-- The given conditions
def east_distance := 200
def south_distance := 150

-- The theorem stating Xiaomin's house position
theorem xiaomin_house_position :
  xiaomin_house_pos east_distance south_distance = (200, -150) :=
by
  -- Skipping the proof steps
  sorry

end xiaomin_house_position_l42_42422


namespace sum_of_integers_is_106_l42_42859

theorem sum_of_integers_is_106 (n m : ℕ) 
  (h1: n * (n + 1) = 1320) 
  (h2: m * (m + 1) * (m + 2) = 1320) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 106 :=
  sorry

end sum_of_integers_is_106_l42_42859


namespace remainder_g_x12_div_g_x_l42_42820

-- Define the polynomial g
noncomputable def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Proving the remainder when g(x^12) is divided by g(x) is 6
theorem remainder_g_x12_div_g_x : 
  (g (x^12) % g x) = 6 :=
sorry

end remainder_g_x12_div_g_x_l42_42820


namespace find_n_l42_42019

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) = 9) : n = 2 :=
by
  sorry

end find_n_l42_42019


namespace fraction_comparison_and_differences_l42_42166

theorem fraction_comparison_and_differences :
  (1/3 < 0.5) ∧ (0.5 < 3/5) ∧ 
  (0.5 - 1/3 = 1/6) ∧ 
  (3/5 - 0.5 = 1/10) :=
by
  sorry

end fraction_comparison_and_differences_l42_42166


namespace price_of_pastries_is_5_l42_42111

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l42_42111


namespace line_passes_fixed_point_l42_42215

open Real

theorem line_passes_fixed_point
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)
  (M N : ℝ × ℝ)
  (hM : M.1^2 / a^2 + M.2^2 / b^2 = 1)
  (hN : N.1^2 / a^2 + N.2^2 / b^2 = 1)
  (hMAhNA : (M.1 + a) * (N.1 + a) + M.2 * N.2 = 0):
  ∃ (P : ℝ × ℝ), P = (a * (b^2 - a^2) / (a^2 + b^2), 0) ∧ (N.2 - M.2) * (P.1 - M.1) = (P.2 - M.2) * (N.1 - M.1) :=
sorry

end line_passes_fixed_point_l42_42215


namespace range_of_x_for_obtuse_angle_l42_42433

def vectors_are_obtuse (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product < 0

theorem range_of_x_for_obtuse_angle :
  ∀ (x : ℝ), vectors_are_obtuse (1, 3) (x, -1) ↔ (x < -1/3 ∨ (-1/3 < x ∧ x < 3)) :=
by
  sorry

end range_of_x_for_obtuse_angle_l42_42433


namespace parallel_line_slope_l42_42403

theorem parallel_line_slope (x y : ℝ) :
  ∃ m b : ℝ, (3 * x - 6 * y = 21) → ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 21) → m = 1 / 2 :=
by
  sorry

end parallel_line_slope_l42_42403


namespace f_of_2014_l42_42231

theorem f_of_2014 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 4) = -f x + 2 * Real.sqrt 2)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  : f 2014 = Real.sqrt 2 :=
sorry

end f_of_2014_l42_42231


namespace C_pow_50_l42_42746

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_pow_50 :
  C ^ 50 = !![-299, -100; 800, 249] := by
  sorry

end C_pow_50_l42_42746


namespace set_of_integers_between_10_and_16_l42_42904

theorem set_of_integers_between_10_and_16 :
  {x : ℤ | 10 < x ∧ x < 16} = {11, 12, 13, 14, 15} :=
by
  sorry

end set_of_integers_between_10_and_16_l42_42904


namespace largest_five_digit_integer_congruent_to_16_mod_25_l42_42619

theorem largest_five_digit_integer_congruent_to_16_mod_25 :
  ∃ x : ℤ, x % 25 = 16 ∧ x < 100000 ∧ ∀ y : ℤ, y % 25 = 16 → y < 100000 → y ≤ x :=
by
  sorry

end largest_five_digit_integer_congruent_to_16_mod_25_l42_42619


namespace unique_solution_for_digits_l42_42861

theorem unique_solution_for_digits :
  ∃ (A B C D E : ℕ),
  (A < B ∧ B < C ∧ C < D ∧ D < E) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
   C ≠ D ∧ C ≠ E ∧
   D ≠ E) ∧
  (10 * A + B) * C = 10 * D + E ∧
  (A = 1 ∧ B = 3 ∧ C = 6 ∧ D = 7 ∧ E = 8) :=
sorry

end unique_solution_for_digits_l42_42861


namespace roots_sum_product_l42_42678

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end roots_sum_product_l42_42678


namespace typists_initial_group_l42_42903

theorem typists_initial_group
  (T : ℕ) 
  (h1 : 0 < T) 
  (h2 : T * (240 / 40 * 20) = 2400) : T = 10 :=
by
  sorry

end typists_initial_group_l42_42903


namespace card_area_after_shortening_l42_42771

theorem card_area_after_shortening 
  (length : ℕ) (width : ℕ) (area_after_shortening : ℕ) 
  (h_initial : length = 8) (h_initial_width : width = 3)
  (h_area_shortened_by_2 : area_after_shortening = 15) :
  (length - 2) * width = 8 :=
by
  -- Original dimensions
  let original_length := 8
  let original_width := 3
  -- Area after shortening one side by 2 inches
  let area_after_shortening_width := (original_length) * (original_width - 2)
  let area_after_shortening_length := (original_length - 2) * (original_width)
  sorry

end card_area_after_shortening_l42_42771


namespace num_ways_128_as_sum_of_four_positive_perfect_squares_l42_42578

noncomputable def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, 0 < m ∧ m * m = n

noncomputable def four_positive_perfect_squares_sum (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    is_positive_perfect_square a ∧
    is_positive_perfect_square b ∧
    is_positive_perfect_square c ∧
    is_positive_perfect_square d ∧
    a + b + c + d = n

theorem num_ways_128_as_sum_of_four_positive_perfect_squares :
  (∃! (a b c d : ℕ), four_positive_perfect_squares_sum 128) :=
sorry

end num_ways_128_as_sum_of_four_positive_perfect_squares_l42_42578


namespace tan_double_angle_l42_42152

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4 / 3 :=
by
  sorry

end tan_double_angle_l42_42152


namespace person_Y_share_l42_42103

theorem person_Y_share (total_amount : ℝ) (r1 r2 r3 r4 r5 : ℝ) (ratio_Y : ℝ) 
  (h1 : total_amount = 1390) 
  (h2 : r1 = 13) 
  (h3 : r2 = 17)
  (h4 : r3 = 23) 
  (h5 : r4 = 29) 
  (h6 : r5 = 37) 
  (h7 : ratio_Y = 29): 
  (total_amount / (r1 + r2 + r3 + r4 + r5) * ratio_Y) = 338.72 :=
by
  sorry

end person_Y_share_l42_42103


namespace part1_part2_l42_42043

noncomputable def f (a x : ℝ) := a * Real.log x - x / 2

theorem part1 (a : ℝ) : (∀ x, f a x = a * Real.log x - x / 2) → (∃ x, x = 2 ∧ deriv (f a) x = 0) → a = 1 :=
by sorry

theorem part2 (k : ℝ) : (∀ x, x > 1 → f 1 x + k / x < 0) → k ≤ 1 / 2 :=
by sorry

end part1_part2_l42_42043


namespace candy_problem_l42_42510

theorem candy_problem
  (G : Nat := 7) -- Gwen got 7 pounds of candy
  (C : Nat := 17) -- Combined weight of candy
  (F : Nat) -- Pounds of candy Frank got
  (h : F + G = C) -- Condition: Combined weight
  : F = 10 := 
by
  sorry

end candy_problem_l42_42510


namespace solve_for_a4b4_l42_42669

theorem solve_for_a4b4 (
    a1 a2 a3 a4 b1 b2 b3 b4 : ℝ
) (h1 : a1 * b1 + a2 * b3 = 1) 
  (h2 : a1 * b2 + a2 * b4 = 0) 
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end solve_for_a4b4_l42_42669


namespace value_at_7_6_l42_42344

noncomputable def f : ℝ → ℝ := sorry

lemma periodic_f (x : ℝ) : f (x + 4) = f x := sorry

lemma f_on_interval (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) : f x = x := sorry

theorem value_at_7_6 : f 7.6 = -0.4 :=
by
  have p := periodic_f 7.6
  have q := periodic_f 3.6
  have r := f_on_interval (-0.4)
  sorry

end value_at_7_6_l42_42344


namespace problem1_problem2_l42_42527

theorem problem1 : 6 + (-8) - (-5) = 3 := sorry

theorem problem2 : 18 / (-3) + (-2) * (-4) = 2 := sorry

end problem1_problem2_l42_42527


namespace triangle_side_lengths_relation_l42_42559

-- Given a triangle ABC with side lengths a, b, c
variables (a b c R d : ℝ)
-- Given orthocenter H and circumcenter O, and the radius of the circumcircle is R,
-- and distance between O and H is d.
-- Prove that a² + b² + c² = 9R² - d²

theorem triangle_side_lengths_relation (a b c R d : ℝ) (H O : Type) (orthocenter : H) (circumcenter : O)
  (radius_circumcircle : O → ℝ)
  (distance_OH : O → H → ℝ) :
  a^2 + b^2 + c^2 = 9 * R^2 - d^2 :=
sorry

end triangle_side_lengths_relation_l42_42559


namespace descent_time_on_moving_escalator_standing_l42_42190

theorem descent_time_on_moving_escalator_standing (l v_mont v_ek t : ℝ)
  (H1 : l / v_mont = 42)
  (H2 : l / (v_mont + v_ek) = 24)
  : t = 56 := by
  sorry

end descent_time_on_moving_escalator_standing_l42_42190


namespace compare_decimal_to_fraction_l42_42692

theorem compare_decimal_to_fraction : (0.650 - (1 / 8) = 0.525) :=
by
  /- We need to prove that 0.650 - 1/8 = 0.525 -/
  sorry

end compare_decimal_to_fraction_l42_42692


namespace lara_bag_total_chips_l42_42627

theorem lara_bag_total_chips (C : ℕ)
  (h1 : ∃ (b : ℕ), b = C / 6)
  (h2 : 34 + 16 + C / 6 = C) :
  C = 60 := by
  sorry

end lara_bag_total_chips_l42_42627


namespace Sam_weight_l42_42320

theorem Sam_weight :
  ∃ (sam_weight : ℕ), (∀ (tyler_weight : ℕ), (∀ (peter_weight : ℕ), peter_weight = 65 → tyler_weight = 2 * peter_weight → tyler_weight = sam_weight + 25 → sam_weight = 105)) :=
by {
    sorry
}

end Sam_weight_l42_42320


namespace maria_remaining_money_l42_42130

theorem maria_remaining_money (initial_amount ticket_cost : ℕ) (h_initial : initial_amount = 760) (h_ticket : ticket_cost = 300) :
  let hotel_cost := ticket_cost / 2
  let total_spent := ticket_cost + hotel_cost
  let remaining := initial_amount - total_spent
  remaining = 310 :=
by
  intros
  sorry

end maria_remaining_money_l42_42130


namespace opposite_neg_half_l42_42515

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l42_42515


namespace avg_children_in_families_with_children_l42_42251

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l42_42251


namespace greatest_length_of_pieces_l42_42300

theorem greatest_length_of_pieces (a b c : ℕ) (ha : a = 48) (hb : b = 60) (hc : c = 72) :
  Nat.gcd (Nat.gcd a b) c = 12 := by
  sorry

end greatest_length_of_pieces_l42_42300


namespace find_a_l42_42371

-- Define the real numbers x, y, and a
variables (x y a : ℝ)

-- Define the conditions as premises
axiom cond1 : x + 3 * y + 5 ≥ 0
axiom cond2 : x + y - 1 ≤ 0
axiom cond3 : x + a ≥ 0

-- Define z as x + 2y and state its minimum value is -4
def z : ℝ := x + 2 * y
axiom min_z : z = -4

-- The theorem to prove the value of a given the above conditions
theorem find_a : a = 2 :=
sorry

end find_a_l42_42371


namespace Sue_button_count_l42_42529

variable (K S : ℕ)

theorem Sue_button_count (H1 : 64 = 5 * K + 4) (H2 : S = K / 2) : S = 6 := 
by
sorry

end Sue_button_count_l42_42529


namespace odd_function_decreasing_l42_42056

theorem odd_function_decreasing (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, x < y → y < 0 → f x > f y) :
  ∀ x y, 0 < x → x < y → f y < f x :=
by
  sorry

end odd_function_decreasing_l42_42056


namespace recipe_calls_for_nine_cups_of_flour_l42_42815

def cups_of_flour (x : ℕ) := 
  ∃ cups_added_sugar : ℕ, 
    cups_added_sugar = (6 - 4) ∧ 
    x = cups_added_sugar + 7

theorem recipe_calls_for_nine_cups_of_flour : cups_of_flour 9 :=
by
  sorry

end recipe_calls_for_nine_cups_of_flour_l42_42815


namespace additional_ice_cubes_made_l42_42681

def original_ice_cubes : ℕ := 2
def total_ice_cubes : ℕ := 9

theorem additional_ice_cubes_made :
  (total_ice_cubes - original_ice_cubes) = 7 :=
by
  sorry

end additional_ice_cubes_made_l42_42681


namespace gcd_seven_eight_fact_l42_42119

-- Definitions based on the problem conditions
def seven_fact : ℕ := 1 * 2 * 3 * 4 * 5 * 6 * 7
def eight_fact : ℕ := 8 * seven_fact

-- Statement of the theorem
theorem gcd_seven_eight_fact : Nat.gcd seven_fact eight_fact = seven_fact := by
  sorry

end gcd_seven_eight_fact_l42_42119


namespace continuous_function_fixed_point_l42_42438

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_comp : ∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ (f^[n] x = 1))

theorem continuous_function_fixed_point : f 1 = 1 := 
by
  sorry

end continuous_function_fixed_point_l42_42438


namespace divisibility_by_9_l42_42387

theorem divisibility_by_9 (x y z : ℕ) (h1 : 9 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (100 * x + 10 * y + z) % 9 = 0 ↔ (x + y + z) % 9 = 0 := by
  sorry

end divisibility_by_9_l42_42387


namespace constant_term_in_expansion_l42_42942

noncomputable def P (x : ℕ) : ℕ := x^4 + 2 * x + 7
noncomputable def Q (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 + 10

theorem constant_term_in_expansion :
  (P 0) * (Q 0) = 70 := 
sorry

end constant_term_in_expansion_l42_42942


namespace solution_set_of_f_double_exp_inequality_l42_42030

theorem solution_set_of_f_double_exp_inequality (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ 0 < f x) :
  {x : ℝ | f (2^x) < 0} = {x : ℝ | x > 0} :=
sorry

end solution_set_of_f_double_exp_inequality_l42_42030


namespace perfect_square_of_expression_l42_42038

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem perfect_square_of_expression : 
  (∃ k : ℕ, (factorial 19 * 2 = k ∧ (factorial 20 * factorial 19) / 5 = k * k)) := sorry

end perfect_square_of_expression_l42_42038


namespace geometry_problem_l42_42240

theorem geometry_problem
  (A B C D E : Type*)
  (BAC ABC ACB ADE ADC AEB DEB CDE : ℝ)
  (h₁ : ABC = 72)
  (h₂ : ACB = 90)
  (h₃ : CDE = 36)
  (h₄ : ADC = 180)
  (h₅ : AEB = 180) :
  DEB = 162 :=
sorry

end geometry_problem_l42_42240


namespace product_of_numbers_l42_42016

theorem product_of_numbers :
  ∃ (a b c : ℚ), a + b + c = 30 ∧
                 a = 2 * (b + c) ∧
                 b = 5 * c ∧
                 a + c = 22 ∧
                 a * b * c = 2500 / 9 :=
by
  sorry

end product_of_numbers_l42_42016


namespace new_person_weight_l42_42382

theorem new_person_weight 
    (W : ℝ) -- total weight of original 8 people
    (x : ℝ) -- weight of the new person
    (increase_by : ℝ) -- average weight increases by 2.5 kg
    (replaced_weight : ℝ) -- weight of the replaced person (55 kg)
    (h1 : increase_by = 2.5)
    (h2 : replaced_weight = 55)
    (h3 : x = replaced_weight + (8 * increase_by)) : x = 75 := 
by
  sorry

end new_person_weight_l42_42382


namespace three_digit_divisible_by_8_l42_42677

theorem three_digit_divisible_by_8 : ∃ n : ℕ, n / 100 = 5 ∧ n % 10 = 3 ∧ n % 8 = 0 :=
by
  use 533
  sorry

end three_digit_divisible_by_8_l42_42677


namespace problem_statement_l42_42289

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (x : ℝ) : ℝ := (2 ^ x) / 2 - 2 / (2 ^ x) - x + 1

theorem problem_statement (a : ℝ) (x₁ x₂ : ℝ) (h₀ : x₁ < x₂)
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) : g x₁ + g x₂ > 0 :=
sorry

end problem_statement_l42_42289


namespace intersection_x_value_l42_42386

/-- Prove that the x-value at the point of intersection of the lines
    y = 5x - 28 and 3x + y = 120 is 18.5 -/
theorem intersection_x_value :
  ∃ x y : ℝ, (y = 5 * x - 28) ∧ (3 * x + y = 120) ∧ (x = 18.5) :=
by
  sorry

end intersection_x_value_l42_42386


namespace branches_sum_one_main_stem_l42_42503

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_sum_one_main_stem_l42_42503


namespace intersection_of_sets_l42_42991

def set_M := { y : ℝ | y ≥ 0 }
def set_N := { y : ℝ | ∃ x : ℝ, y = -x^2 + 1 }

theorem intersection_of_sets : set_M ∩ set_N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_sets_l42_42991


namespace percentage_problem_l42_42814

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end percentage_problem_l42_42814


namespace expected_value_decisive_games_l42_42539

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end expected_value_decisive_games_l42_42539


namespace price_difference_l42_42118

noncomputable def original_price (discounted_price : ℝ) : ℝ :=
  discounted_price / 0.85

noncomputable def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.25

theorem price_difference (discounted_price : ℝ) (h : discounted_price = 71.4) : 
  (final_price discounted_price) - (original_price discounted_price) = 5.25 := 
by
  sorry

end price_difference_l42_42118


namespace number_of_elephants_l42_42715

theorem number_of_elephants (giraffes penguins total_animals elephants : ℕ)
  (h1 : giraffes = 5)
  (h2 : penguins = 2 * giraffes)
  (h3 : penguins = total_animals / 5)
  (h4 : elephants = total_animals * 4 / 100) :
  elephants = 2 := by
  -- The proof is omitted
  sorry

end number_of_elephants_l42_42715


namespace remainder_698_div_D_l42_42995

-- Defining the conditions
variables (D k1 k2 k3 R : ℤ)

-- Given conditions
axiom condition1 : 242 = k1 * D + 4
axiom condition2 : 940 = k3 * D + 7
axiom condition3 : 698 = k2 * D + R

-- The theorem to prove the remainder 
theorem remainder_698_div_D : R = 3 :=
by
  -- Here you would provide the logical deduction steps
  sorry

end remainder_698_div_D_l42_42995


namespace system_of_equations_solutions_l42_42116

theorem system_of_equations_solutions (x y a b : ℝ) 
  (h1 : 2 * x + y = b) 
  (h2 : x - b * y = a) 
  (hx : x = 1)
  (hy : y = 0) : a - b = -1 :=
by 
  sorry

end system_of_equations_solutions_l42_42116


namespace find_common_difference_l42_42971

def is_arithmetic_sequence (a : (ℕ → ℝ)) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_arithmetic_sequence_with_sum (a : (ℕ → ℝ)) (S : (ℕ → ℝ)) (d : ℝ) : Prop :=
  S 0 = a 0 ∧
  ∀ n, S (n + 1) = S n + a (n + 1) ∧
        ∀ n, (S (n + 1) / a (n + 1) - S n / a n) = d

theorem find_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence_with_sum a S d →
  (d = 1 ∨ d = 1 / 2) :=
sorry

end find_common_difference_l42_42971


namespace painting_time_equation_l42_42886

theorem painting_time_equation (t : ℝ) :
  let Doug_rate := (1 : ℝ) / 5
  let Dave_rate := (1 : ℝ) / 7
  let combined_rate := Doug_rate + Dave_rate
  (combined_rate * (t - 1) = 1) :=
sorry

end painting_time_equation_l42_42886


namespace gcd_multiples_l42_42855

theorem gcd_multiples (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : Nat.gcd p q = 15) : Nat.gcd (8 * p) (18 * q) = 30 :=
by sorry

end gcd_multiples_l42_42855


namespace arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l42_42873

open Nat

variable (a : ℕ → ℝ)
variable (c : ℕ → ℝ)
variable (k b : ℝ)

-- Condition 1: sequence definition
def sequence_condition := ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + n + 1

-- Condition 2: initial value
def initial_value := a 1 = -1

-- Condition 3: c_n definition
def geometric_sequence_condition := ∀ n : ℕ, 0 < n → c (n + 1) / c n = 2

-- Problem 1: Arithmetic sequence parameters
theorem arith_sequence_parameters (h1 : sequence_condition a) (h2 : initial_value a) : a 1 = -3 ∧ 2 * (a 1 + 2) - a 1 - 7 = -1 :=
by sorry

-- Problem 2: Cannot be a geometric sequence
theorem not_geo_sequence (h1 : sequence_condition a) (h2 : initial_value a) : ¬ (∃ q, ∀ n : ℕ, 0 < n → a n * q = a (n + 1)) :=
by sorry

-- Problem 3: c_n is a geometric sequence and general term for a_n
theorem geo_sequence_and_gen_term (h1 : sequence_condition a) (h2 : initial_value a) 
    (h3 : ∀ n : ℕ, 0 < n → c n = a n + k * n + b)
    (hk : k = 1) (hb : b = 2) : sequence_condition a ∧ initial_value a :=
by sorry

end arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l42_42873


namespace find_a_find_distance_l42_42600

-- Problem 1: Given conditions to find 'a'
theorem find_a (a : ℝ) :
  (∃ θ ρ, ρ = 2 * Real.cos θ ∧ 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0) →
  (a = 2 ∨ a = -8) :=
sorry

-- Problem 2: Given point and line, find the distance
theorem find_distance : 
  ∃ (d : ℝ), d = Real.sqrt 3 + 5/2 ∧
  (∃ θ ρ, θ = 11 * Real.pi / 6 ∧ ρ = 2 ∧ 
   (ρ = Real.sqrt (3 * (Real.sin θ - Real.pi / 6)^2 + (ρ * Real.cos (θ - Real.pi / 6))^2) 
   → ρ * Real.sin (θ - Real.pi / 6) = 1)) :=
sorry

end find_a_find_distance_l42_42600


namespace expression_eval_l42_42735

theorem expression_eval :
    (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
    (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * 5040 = 
    (5^128 - 4^128) * 5040 := by
  sorry

end expression_eval_l42_42735


namespace algebraic_expression_value_l42_42434

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -4) :
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 23 :=
by
  sorry

end algebraic_expression_value_l42_42434


namespace find_other_number_l42_42288

theorem find_other_number
  (n m lcm gcf : ℕ)
  (h_n : n = 40)
  (h_lcm : lcm = 56)
  (h_gcf : gcf = 10)
  (h_lcm_gcf : lcm * gcf = n * m) : m = 14 :=
by
  sorry

end find_other_number_l42_42288


namespace line_passes_through_vertex_of_parabola_l42_42544

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end line_passes_through_vertex_of_parabola_l42_42544


namespace area_rectangle_around_right_triangle_l42_42923

theorem area_rectangle_around_right_triangle (AB BC : ℕ) (hAB : AB = 5) (hBC : BC = 6) :
    let ADE_area := AB * BC
    ADE_area = 30 := by
  sorry

end area_rectangle_around_right_triangle_l42_42923


namespace simplify_expression_l42_42432

theorem simplify_expression (x y : ℤ) (h1 : x = 1) (h2 : y = -2) :
  2 * x ^ 2 - (3 * (-5 / 3 * x ^ 2 + 2 / 3 * x * y) - (x * y - 3 * x ^ 2)) + 2 * x * y = 2 :=
by {
  sorry
}

end simplify_expression_l42_42432


namespace probability_all_red_or_all_white_l42_42936

theorem probability_all_red_or_all_white :
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 6
  let total_marbles := red_marbles + white_marbles + blue_marbles
  let probability_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let probability_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  (probability_red + probability_white) = (14 / 455) :=
by
  sorry

end probability_all_red_or_all_white_l42_42936


namespace polar_to_rectangular_coordinates_l42_42391

theorem polar_to_rectangular_coordinates :
  let r := 2
  let θ := Real.pi / 3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (1, Real.sqrt 3) :=
by
  sorry

end polar_to_rectangular_coordinates_l42_42391


namespace distance_from_Bangalore_l42_42885

noncomputable def calculate_distance (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) : ℕ :=
  let total_travel_minutes := (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute) - halt_minutes
  let total_travel_hours := total_travel_minutes / 60
  speed * total_travel_hours

theorem distance_from_Bangalore (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) :
  speed = 87 ∧ start_hour = 9 ∧ start_minute = 0 ∧ end_hour = 13 ∧ end_minute = 45 ∧ halt_minutes = 45 →
  calculate_distance speed start_hour start_minute end_hour end_minute halt_minutes = 348 := by
  sorry

end distance_from_Bangalore_l42_42885


namespace not_in_sequence_l42_42500

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence property
def sequence_property (a b : ℕ) : Prop :=
  b = a + sum_of_digits a

-- Main theorem
theorem not_in_sequence (n : ℕ) (h : n = 793210041) : 
  ¬ (∃ a : ℕ, sequence_property a n) :=
by
  sorry

end not_in_sequence_l42_42500


namespace cost_price_of_computer_table_l42_42208

theorem cost_price_of_computer_table (C SP : ℝ) (h1 : SP = 1.25 * C) (h2 : SP = 8340) :
  C = 6672 :=
by
  sorry

end cost_price_of_computer_table_l42_42208


namespace quilt_width_l42_42541

-- Definitions according to the conditions
def quilt_length : ℕ := 16
def patch_area : ℕ := 4
def first_10_patches_cost : ℕ := 100
def total_cost : ℕ := 450
def remaining_budget : ℕ := total_cost - first_10_patches_cost
def cost_per_additional_patch : ℕ := 5
def num_additional_patches : ℕ := remaining_budget / cost_per_additional_patch
def total_patches : ℕ := 10 + num_additional_patches
def total_area : ℕ := total_patches * patch_area

-- Theorem statement
theorem quilt_width :
  (total_area / quilt_length) = 20 :=
by
  sorry

end quilt_width_l42_42541


namespace max_pages_copied_l42_42402

-- Definitions based on conditions
def cents_per_page := 7 / 4
def budget_cents := 1500

-- The theorem to prove
theorem max_pages_copied (c : ℝ) (budget : ℝ) (h₁ : c = cents_per_page) (h₂ : budget = budget_cents) : 
  ⌊(budget / c)⌋ = 857 :=
sorry

end max_pages_copied_l42_42402


namespace hexagon_chord_length_valid_l42_42809

def hexagon_inscribed_chord_length : ℚ := 48 / 49

theorem hexagon_chord_length_valid : 
    ∃ (p q : ℕ), gcd p q = 1 ∧ hexagon_inscribed_chord_length = p / q ∧ p + q = 529 :=
sorry

end hexagon_chord_length_valid_l42_42809


namespace inequality_solution_set_l42_42334

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1 / 2} :=
by
  sorry

end inequality_solution_set_l42_42334


namespace james_fish_tanks_l42_42790

theorem james_fish_tanks (n t1 t2 t3 : ℕ) (h1 : t1 = 20) (h2 : t2 = 2 * t1) (h3 : t3 = 2 * t1) (h4 : t1 + t2 + t3 = 100) : n = 3 :=
sorry

end james_fish_tanks_l42_42790


namespace find_current_l42_42494

noncomputable def V : ℂ := 2 + 3 * Complex.I
noncomputable def Z : ℂ := 2 - 2 * Complex.I

theorem find_current : (V / Z) = (-1 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I := by
  sorry

end find_current_l42_42494


namespace reading_time_per_disc_l42_42751

theorem reading_time_per_disc (total_minutes : ℕ) (disc_capacity : ℕ) (d : ℕ) (reading_per_disc : ℕ) :
  total_minutes = 528 ∧ disc_capacity = 45 ∧ d = 12 ∧ total_minutes = d * reading_per_disc → reading_per_disc = 44 :=
by
  sorry

end reading_time_per_disc_l42_42751


namespace computer_price_increase_l42_42958

theorem computer_price_increase
  (P : ℝ)
  (h1 : 1.30 * P = 351) :
  (P + 1.30 * P) / P = 2.3 := by
  sorry

end computer_price_increase_l42_42958


namespace susan_more_cats_than_bob_after_transfer_l42_42543

-- Definitions and conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def cats_transferred : ℕ := 4

-- Question statement translated to Lean
theorem susan_more_cats_than_bob_after_transfer :
  (susan_initial_cats - cats_transferred) - bob_initial_cats = 14 :=
by
  sorry

end susan_more_cats_than_bob_after_transfer_l42_42543


namespace min_value_at_x_eq_2_l42_42278

theorem min_value_at_x_eq_2 (x : ℝ) (h : x > 1) : 
  x + 1/(x-1) = 3 ↔ x = 2 :=
by sorry

end min_value_at_x_eq_2_l42_42278


namespace find_other_number_l42_42834

theorem find_other_number (x : ℕ) (h : x + 42 = 96) : x = 54 :=
by {
  sorry
}

end find_other_number_l42_42834


namespace book_price_percentage_change_l42_42504

theorem book_price_percentage_change (P : ℝ) (x : ℝ) (h : P * (1 - (x / 100) ^ 2) = 0.90 * P) : x = 32 := by
sorry

end book_price_percentage_change_l42_42504


namespace problem_solution_l42_42355

theorem problem_solution
  (a b : ℝ)
  (h_eqn : ∃ (a b : ℝ), 3 * a * a + 9 * a - 21 = 0 ∧ 3 * b * b + 9 * b - 21 = 0 )
  (h_vieta_sum : a + b = -3)
  (h_vieta_prod : a * b = -7) :
  (2 * a - 5) * (3 * b - 4) = 47 := 
by
  sorry

end problem_solution_l42_42355


namespace probability_A_seven_rolls_l42_42367

noncomputable def probability_A_after_n_rolls (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1/3 * (1 - (-1/2)^(n-1))

theorem probability_A_seven_rolls : probability_A_after_n_rolls 7 = 21 / 64 :=
by sorry

end probability_A_seven_rolls_l42_42367


namespace intersection_of_A_and_B_l42_42594

def A : Set ℝ := { x | x^2 - 5 * x - 6 ≤ 0 }

def B : Set ℝ := { x | x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -1 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l42_42594


namespace intersection_P_Q_l42_42321

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l42_42321


namespace alcohol_solution_mixing_l42_42601

theorem alcohol_solution_mixing :
  ∀ (V_i C_i C_f C_a x : ℝ),
    V_i = 6 →
    C_i = 0.40 →
    C_f = 0.50 →
    C_a = 0.90 →
    x = 1.5 →
  0.50 * (V_i + x) = (C_i * V_i) + C_a * x →
  C_f * (V_i + x) = (C_i * V_i) + (C_a * x) := 
by
  intros V_i C_i C_f C_a x Vi_eq Ci_eq Cf_eq Ca_eq x_eq h
  sorry

end alcohol_solution_mixing_l42_42601


namespace min_c_value_l42_42566

theorem min_c_value (c : ℝ) : (-c^2 + 9 * c - 14 >= 0) → (c >= 2) :=
by {
  sorry
}

end min_c_value_l42_42566


namespace base3_to_base10_conversion_l42_42465

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end base3_to_base10_conversion_l42_42465


namespace max_distance_between_sparkling_points_l42_42022

theorem max_distance_between_sparkling_points (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 = 1) (h₂ : a₂^2 + b₂^2 = 1) :
  ∃ d, d = 2 ∧ ∀ (x y : ℝ), x = a₂ - a₁ ∧ y = b₂ - b₁ → (x ^ 2 + y ^ 2 = d ^ 2) :=
by
  sorry

end max_distance_between_sparkling_points_l42_42022


namespace whale_crossing_time_l42_42853

theorem whale_crossing_time
  (speed_fast : ℝ)
  (speed_slow : ℝ)
  (length_slow : ℝ)
  (h_fast : speed_fast = 18)
  (h_slow : speed_slow = 15)
  (h_length : length_slow = 45) :
  (length_slow / (speed_fast - speed_slow) = 15) :=
by
  sorry

end whale_crossing_time_l42_42853


namespace find_k_point_verification_l42_42006

-- Definition of the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Condition that the point (2, 7) lies on the graph of the linear function
def passes_through (k : ℝ) : Prop := linear_function k 2 = 7

-- The actual proof task to verify the value of k
theorem find_k : ∃ k : ℝ, passes_through k ∧ k = 2 :=
by
  sorry

-- The condition that the point (-2, 1) is not on the graph with k = 2
def point_not_on_graph : Prop := ¬ (linear_function 2 (-2) = 1)

-- The actual proof task to verify the point (-2, 1) is not on the graph of y = 2x + 3
theorem point_verification : point_not_on_graph :=
by
  sorry

end find_k_point_verification_l42_42006


namespace roland_thread_length_l42_42368

noncomputable def length_initial : ℝ := 12
noncomputable def length_two_thirds : ℝ := (2 / 3) * length_initial
noncomputable def length_increased : ℝ := length_initial + length_two_thirds
noncomputable def length_half_increased : ℝ := (1 / 2) * length_increased
noncomputable def length_total : ℝ := length_increased + length_half_increased
noncomputable def length_inches : ℝ := length_total / 2.54

theorem roland_thread_length : length_inches = 11.811 :=
by sorry

end roland_thread_length_l42_42368


namespace lines_intersection_l42_42969

theorem lines_intersection :
  ∃ (x y : ℝ), 
    (x - y = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end lines_intersection_l42_42969


namespace mathematically_equivalent_proof_l42_42514

noncomputable def proof_problem (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ a^x = 2 ∧ a^y = 3 → a^(x - 2 * y) = 2 / 9

theorem mathematically_equivalent_proof (a : ℝ) (x y : ℝ) :
  proof_problem a x y :=
by
  sorry  -- Proof steps will go here

end mathematically_equivalent_proof_l42_42514


namespace tan_beta_minus_2alpha_l42_42551

open Real

-- Given definitions
def condition1 (α : ℝ) : Prop :=
  (sin α * cos α) / (1 - cos (2 * α)) = 1 / 4

def condition2 (α β : ℝ) : Prop :=
  tan (α - β) = 2

-- Proof problem statement
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : condition1 α) (h2 : condition2 α β) :
  tan (β - 2 * α) = 4 / 3 :=
sorry

end tan_beta_minus_2alpha_l42_42551


namespace correct_total_annual_salary_expression_l42_42306

def initial_workers : ℕ := 8
def initial_salary : ℝ := 1.0 -- in ten thousand yuan
def new_workers : ℕ := 3
def new_worker_initial_salary : ℝ := 0.8 -- in ten thousand yuan
def salary_increase_rate : ℝ := 1.2 -- 20% increase each year

def total_annual_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * salary_increase_rate^n + (new_workers * new_worker_initial_salary)

theorem correct_total_annual_salary_expression (n : ℕ) :
  total_annual_salary n = (3 * n + 5) * 1.2^n + 2.4 := 
by
  sorry

end correct_total_annual_salary_expression_l42_42306


namespace correct_statement_B_l42_42844

/-- Define the diameter of a sphere -/
def diameter (d : ℝ) (s : Set (ℝ × ℝ × ℝ)) : Prop :=
∃ x y : ℝ × ℝ × ℝ, x ∈ s ∧ y ∈ s ∧ dist x y = d ∧ ∀ z ∈ s, dist x y ≥ dist x z ∧ dist x y ≥ dist z y

/-- Define that a line segment connects two points on the sphere's surface and passes through the center -/
def connects_diameter (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ) : Prop :=
dist center x = radius ∧ dist center y = radius ∧ (x + y) / 2 = center

/-- A sphere is the set of all points at a fixed distance from the center -/
def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ × ℝ) :=
{x | dist center x = radius}

theorem correct_statement_B (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ):
  (∀ (s : Set (ℝ × ℝ × ℝ)), sphere center radius = s → diameter (2 * radius) s)
  → connects_diameter center radius x y
  → (∃ d : ℝ, diameter d (sphere center radius)) := 
by
  intros
  sorry

end correct_statement_B_l42_42844


namespace infinite_pairs_exists_l42_42847

noncomputable def exists_infinite_pairs : Prop :=
  ∃ (a b : ℕ), (a + b ∣ a * b + 1) ∧ (a - b ∣ a * b - 1) ∧ b > 1 ∧ a > b * Real.sqrt 3 - 1

theorem infinite_pairs_exists : ∃ (count : ℕ) (a b : ℕ), ∀ n < count, exists_infinite_pairs :=
sorry

end infinite_pairs_exists_l42_42847


namespace equalSumSeqDefinition_l42_42857

def isEqualSumSeq (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n - 1) + s n = s (n + 1)

theorem equalSumSeqDefinition (s : ℕ → ℝ) :
  isEqualSumSeq s ↔ 
  ∀ n : ℕ, n > 0 → s n = s (n - 1) + s (n + 1) :=
by
  sorry

end equalSumSeqDefinition_l42_42857


namespace min_a1_a7_l42_42137

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem min_a1_a7 (a : ℕ → ℝ) (h : geom_seq a)
  (h1 : a 3 * a 5 = 64) :
  ∃ m, m = (a 1 + a 7) ∧ m = 16 :=
by
  sorry

end min_a1_a7_l42_42137


namespace perimeter_eq_20_l42_42721

-- Define the lengths of the sides
def horizontal_sides := [2, 3]
def vertical_sides := [2, 3, 3, 2]

-- Define the perimeter calculation
def perimeter := horizontal_sides.sum + vertical_sides.sum

theorem perimeter_eq_20 : perimeter = 20 :=
by
  -- We assert that the calculations do hold
  sorry

end perimeter_eq_20_l42_42721


namespace range_of_p_l42_42484

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 1023 :=
by
  sorry

end range_of_p_l42_42484


namespace erased_number_is_six_l42_42614

theorem erased_number_is_six (n x : ℕ) (h1 : (n * (n + 1)) / 2 - x = 45 * (n - 1) / 4):
  x = 6 :=
by
  sorry

end erased_number_is_six_l42_42614


namespace number_of_ways_to_draw_balls_l42_42977

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l42_42977


namespace prove_u_div_p_l42_42680

theorem prove_u_div_p (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) : 
  u / p = 15 / 8 := 
by 
  sorry

end prove_u_div_p_l42_42680


namespace minimum_shots_required_l42_42898

noncomputable def minimum_shots_to_sink_boat : ℕ := 4000

-- Definitions for the problem conditions.
structure Boat :=
(square_side : ℕ)
(base1 : ℕ)
(base2 : ℕ)
(rotatable : Bool)

def boat : Boat := { square_side := 1, base1 := 1, base2 := 3, rotatable := true }

def grid_size : ℕ := 100

def shot_covers_triangular_half : Prop := sorry -- Assumption: Define this appropriately

-- Problem statement in Lean 4
theorem minimum_shots_required (boat_within_grid : Bool) : 
  Boat → grid_size = 100 → boat_within_grid → minimum_shots_to_sink_boat = 4000 :=
by
  -- Here you would do the full proof which we assume is "sorry" for now
  sorry

end minimum_shots_required_l42_42898


namespace sum_first_100_sum_51_to_100_l42_42146

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem sum_first_100 : sum_natural_numbers 100 = 5050 :=
  sorry

theorem sum_51_to_100 : sum_natural_numbers 100 - sum_natural_numbers 50 = 3775 :=
  sorry

end sum_first_100_sum_51_to_100_l42_42146


namespace center_of_circle_is_at_10_3_neg5_l42_42383

noncomputable def center_of_tangent_circle (x y : ℝ) : Prop :=
  (6 * x - 5 * y = 50 ∨ 6 * x - 5 * y = -20) ∧ (3 * x + 2 * y = 0)

theorem center_of_circle_is_at_10_3_neg5 :
  ∃ x y : ℝ, center_of_tangent_circle x y ∧ x = 10 / 3 ∧ y = -5 :=
by
  sorry

end center_of_circle_is_at_10_3_neg5_l42_42383


namespace initial_amount_l42_42765

theorem initial_amount (P : ℝ) (h1 : ∀ x : ℝ, x * (9 / 8) * (9 / 8) = 81000) : P = 64000 :=
sorry

end initial_amount_l42_42765


namespace carB_speed_l42_42412

variable (distance : ℝ) (time : ℝ) (ratio : ℝ) (speedB : ℝ)

theorem carB_speed (h1 : distance = 240) (h2 : time = 1.5) (h3 : ratio = 3 / 5) 
(h4 : (speedB + ratio * speedB) * time = distance) : speedB = 100 := 
by 
  sorry

end carB_speed_l42_42412


namespace find_initial_amount_l42_42351

noncomputable def initial_amount (diff : ℝ) : ℝ :=
  diff / (1.4641 - 1.44)

theorem find_initial_amount
  (diff : ℝ)
  (h : diff = 964.0000000000146) :
  initial_amount diff = 40000 :=
by
  -- the steps to prove this can be added here later
  sorry

end find_initial_amount_l42_42351


namespace ratio_sprite_to_coke_l42_42363

theorem ratio_sprite_to_coke (total_drink : ℕ) (coke_ounces : ℕ) (mountain_dew_parts : ℕ)
  (parts_coke : ℕ) (parts_mountain_dew : ℕ) (total_parts : ℕ) :
  total_drink = 18 →
  coke_ounces = 6 →
  parts_coke = 2 →
  parts_mountain_dew = 3 →
  total_parts = parts_coke + parts_mountain_dew + ((total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / (coke_ounces / parts_coke)) →
  (total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / coke_ounces = 1 / 2 :=
by sorry

end ratio_sprite_to_coke_l42_42363


namespace largest_expr_l42_42011

noncomputable def A : ℝ := 2 * 1005 ^ 1006
noncomputable def B : ℝ := 1005 ^ 1006
noncomputable def C : ℝ := 1004 * 1005 ^ 1005
noncomputable def D : ℝ := 2 * 1005 ^ 1005
noncomputable def E : ℝ := 1005 ^ 1005
noncomputable def F : ℝ := 1005 ^ 1004

theorem largest_expr : A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by
  sorry

end largest_expr_l42_42011


namespace hyperbola_equation_l42_42623

noncomputable def hyperbola_eqn : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (b = (1/2) * a) ∧ (a^2 + b^2 = 25) ∧ 
    (∀ x y, (x^2 / (a^2)) - (y^2 / (b^2)) = 1 ↔ (x^2 / 20) - (y^2 / 5) = 1)

theorem hyperbola_equation : hyperbola_eqn := 
  sorry

end hyperbola_equation_l42_42623


namespace find_value_of_a_l42_42591

theorem find_value_of_a (a : ℤ) (h : ∀ x : ℚ,  x^6 - 33 * x + 20 = (x^2 - x + a) * (x^4 + b * x^3 + c * x^2 + d * x + e)) :
  a = 4 := 
by 
  sorry

end find_value_of_a_l42_42591


namespace number_of_ordered_pairs_l42_42866

theorem number_of_ordered_pairs (h : ∀ (m n : ℕ), 0 < m → 0 < n → 6/m + 3/n = 1 → true) : 
∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x : ℕ × ℕ), x ∈ s → 0 < x.1 ∧ 0 < x.2 ∧ 6 / ↑x.1 + 3 / ↑x.2 = 1 :=
by
-- Sorry, skipping the proof
  sorry

end number_of_ordered_pairs_l42_42866


namespace Petya_workout_duration_l42_42162

theorem Petya_workout_duration :
  ∃ x : ℕ, (x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 135) ∧
            (x + 7 > x) ∧
            (x + 14 > x + 7) ∧
            (x + 21 > x + 14) ∧
            (x + 28 > x + 21) ∧
            x = 13 :=
by sorry

end Petya_workout_duration_l42_42162


namespace Ursula_hours_per_day_l42_42562

theorem Ursula_hours_per_day (hourly_wage : ℝ) (days_per_month : ℕ) (annual_salary : ℝ) (months_per_year : ℕ) :
  hourly_wage = 8.5 →
  days_per_month = 20 →
  annual_salary = 16320 →
  months_per_year = 12 →
  (annual_salary / months_per_year / days_per_month / hourly_wage) = 8 :=
by
  intros
  sorry

end Ursula_hours_per_day_l42_42562


namespace marie_lost_erasers_l42_42002

def initialErasers : ℕ := 95
def finalErasers : ℕ := 53

theorem marie_lost_erasers : initialErasers - finalErasers = 42 := by
  sorry

end marie_lost_erasers_l42_42002


namespace tangent_line_eqn_l42_42577

theorem tangent_line_eqn :
  ∃ k : ℝ, 
  x^2 + y^2 - 4*x + 3 = 0 → 
  (∃ x y : ℝ, (x-2)^2 + y^2 = 1 ∧ x > 2 ∧ y < 0 ∧ y = k*x) → 
  k = - (Real.sqrt 3) / 3 := 
by
  sorry

end tangent_line_eqn_l42_42577


namespace slope_of_line_l42_42662

theorem slope_of_line : 
  let A := Real.sin (Real.pi / 6)
  let B := Real.cos (5 * Real.pi / 6)
  (- A / B) = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_l42_42662


namespace train_passing_platform_time_l42_42897

theorem train_passing_platform_time :
  (500 : ℝ) / (50 : ℝ) > 0 →
  (500 : ℝ) + (500 : ℝ) / ((500 : ℝ) / (50 : ℝ)) = 100 := by
  sorry

end train_passing_platform_time_l42_42897


namespace average_temperature_l42_42704

theorem average_temperature (T_tue T_wed T_thu : ℝ) 
  (h1 : (42 + T_tue + T_wed + T_thu) / 4 = 48)
  (T_fri : ℝ := 34) :
  ((T_tue + T_wed + T_thu + T_fri) / 4 = 46) :=
by
  sorry

end average_temperature_l42_42704


namespace jellybean_problem_l42_42606

theorem jellybean_problem:
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by
  intros black green orange h_black h_green h_total
  sorry

end jellybean_problem_l42_42606


namespace circle_center_and_radius_l42_42488

theorem circle_center_and_radius (x y : ℝ) : 
  (x^2 + y^2 - 6 * x = 0) → ((x - 3)^2 + (y - 0)^2 = 9) :=
by
  intro h
  -- The proof is left as an exercise.
  sorry

end circle_center_and_radius_l42_42488


namespace sequence_a_n_l42_42612

theorem sequence_a_n (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = 3 + 2^n) →
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) ↔ 
  (∀ n : ℕ, a n = if n = 1 then 5 else 2^(n-1)) :=
by
  sorry

end sequence_a_n_l42_42612


namespace parking_space_unpainted_side_l42_42180

theorem parking_space_unpainted_side 
  (L W : ℝ) 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 125) : 
  L = 8.90 := 
by 
  sorry

end parking_space_unpainted_side_l42_42180


namespace mikes_ride_is_46_miles_l42_42613

-- Define the conditions and the question in Lean 4
variable (M : ℕ)

-- Mike's cost formula
def mikes_cost (M : ℕ) : ℚ := 2.50 + 0.25 * M

-- Annie's total cost
def annies_miles : ℕ := 26
def annies_cost : ℚ := 2.50 + 5.00 + 0.25 * annies_miles

-- The proof statement
theorem mikes_ride_is_46_miles (h : mikes_cost M = annies_cost) : M = 46 :=
by sorry

end mikes_ride_is_46_miles_l42_42613


namespace cos_B_value_l42_42437

-- Define the sides of the triangle
def AB : ℝ := 8
def AC : ℝ := 10
def right_angle_at_A : Prop := true

-- Define the cosine function within the context of the given triangle
noncomputable def cos_B : ℝ := AB / AC

-- The proof statement asserting the condition
theorem cos_B_value : cos_B = 4 / 5 :=
by
  -- Given conditions
  have h1 : AB = 8 := rfl
  have h2 : AC = 10 := rfl
  -- Direct computation
  sorry

end cos_B_value_l42_42437


namespace y_increases_as_x_increases_l42_42185

-- Define the linear function y = (m^2 + 2)x
def linear_function (m x : ℝ) : ℝ := (m^2 + 2) * x

-- Prove that y increases as x increases
theorem y_increases_as_x_increases (m x1 x2 : ℝ) (h : x1 < x2) : linear_function m x1 < linear_function m x2 :=
by
  -- because m^2 + 2 is always positive, the function is strictly increasing
  have hm : 0 < m^2 + 2 := by linarith [pow_two_nonneg m]
  have hx : (m^2 + 2) * x1 < (m^2 + 2) * x2 := by exact (mul_lt_mul_left hm).mpr h
  exact hx

end y_increases_as_x_increases_l42_42185


namespace ratio_distance_traveled_by_foot_l42_42091

theorem ratio_distance_traveled_by_foot (D F B C : ℕ) (hD : D = 40) 
(hB : B = D / 2) (hC : C = 10) (hF : F = D - (B + C)) : F / D = 1 / 4 := 
by sorry

end ratio_distance_traveled_by_foot_l42_42091


namespace find_PB_l42_42972

noncomputable def PA : ℝ := 5
noncomputable def PT (AB : ℝ) : ℝ := 2 * (AB - PA) + 1
noncomputable def PB (AB : ℝ) : ℝ := PA + AB

theorem find_PB (AB : ℝ) (AB_condition : AB = PB AB - PA) :
  PB AB = (81 + Real.sqrt 5117) / 8 :=
by
  sorry

end find_PB_l42_42972


namespace find_speed_way_home_l42_42196

theorem find_speed_way_home
  (speed_to_mother : ℝ)
  (average_speed : ℝ)
  (speed_to_mother_val : speed_to_mother = 130)
  (average_speed_val : average_speed = 109) :
  ∃ v : ℝ, v = 109 * 130 / 151 := by
  sorry

end find_speed_way_home_l42_42196


namespace find_r_l42_42626

-- Define the basic conditions based on the given problem.
def pr (r : ℕ) := 360 / 6
def p := pr 4 / 4
def cr (c r : ℕ) := 6 * c * r

-- Prove that r = 4 given the conditions.
theorem find_r (r : ℕ) : r = 4 :=
by
  sorry

end find_r_l42_42626


namespace obtuse_triangle_has_exactly_one_obtuse_angle_l42_42827

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- Definition of an obtuse angle
def is_obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

-- The theorem statement
theorem obtuse_triangle_has_exactly_one_obtuse_angle {A B C : ℝ} 
  (h1 : is_obtuse_triangle A B C) : 
  (is_obtuse_angle A ∨ is_obtuse_angle B ∨ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle B) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle B ∧ is_obtuse_angle C) :=
sorry

end obtuse_triangle_has_exactly_one_obtuse_angle_l42_42827


namespace total_handshakes_l42_42327

-- Definitions and conditions
def num_dwarves := 25
def num_elves := 18

def handshakes_among_dwarves : ℕ := num_dwarves * (num_dwarves - 1) / 2
def handshakes_between_dwarves_and_elves : ℕ := num_elves * num_dwarves

-- Total number of handshakes
theorem total_handshakes : handshakes_among_dwarves + handshakes_between_dwarves_and_elves = 750 := by 
  sorry

end total_handshakes_l42_42327


namespace middle_digit_base_5_reversed_in_base_8_l42_42441

theorem middle_digit_base_5_reversed_in_base_8 (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 4) (h₂ : 0 ≤ b ∧ b ≤ 4) 
  (h₃ : 0 ≤ c ∧ c ≤ 4) (h₄ : 25 * a + 5 * b + c = 64 * c + 8 * b + a) : b = 3 := 
by 
  sorry

end middle_digit_base_5_reversed_in_base_8_l42_42441


namespace Diane_bakes_160_gingerbreads_l42_42837

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end Diane_bakes_160_gingerbreads_l42_42837


namespace new_mean_rent_l42_42557

theorem new_mean_rent (avg_rent : ℕ) (num_friends : ℕ) (rent_increase_pct : ℕ) (initial_rent : ℕ) :
  avg_rent = 800 →
  num_friends = 4 →
  rent_increase_pct = 25 →
  initial_rent = 800 →
  (avg_rent * num_friends + initial_rent * rent_increase_pct / 100) / num_friends = 850 :=
by
  intros h_avg h_num h_pct h_init
  sorry

end new_mean_rent_l42_42557


namespace sum_of_first_17_terms_arithmetic_sequence_l42_42400

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem sum_of_first_17_terms_arithmetic_sequence
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 3 + a 9 + a 15 = 9) :
  sum_of_first_n_terms a 17 = 51 :=
sorry

end sum_of_first_17_terms_arithmetic_sequence_l42_42400


namespace structure_cube_count_l42_42013

theorem structure_cube_count :
  let middle_layer := 16
  let other_layers := 4 * 24
  middle_layer + other_layers = 112 :=
by
  let middle_layer := 16
  let other_layers := 4 * 24
  have h : middle_layer + other_layers = 112 := by
    sorry
  exact h

end structure_cube_count_l42_42013


namespace cos_five_pi_over_three_l42_42695

theorem cos_five_pi_over_three : Real.cos (5 * Real.pi / 3) = 1 / 2 := 
by 
  sorry

end cos_five_pi_over_three_l42_42695


namespace average_speed_third_hour_l42_42978

theorem average_speed_third_hour
  (total_distance : ℝ)
  (total_time : ℝ)
  (speed_first_hour : ℝ)
  (speed_second_hour : ℝ)
  (speed_third_hour : ℝ) :
  total_distance = 150 →
  total_time = 3 →
  speed_first_hour = 45 →
  speed_second_hour = 55 →
  (speed_first_hour + speed_second_hour + speed_third_hour) / total_time = 50 →
  speed_third_hour = 50 :=
sorry

end average_speed_third_hour_l42_42978
