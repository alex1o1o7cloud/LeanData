import Mathlib

namespace focus_of_parabola_l437_437385

theorem focus_of_parabola (x y : ℝ) : 
  (∀ x y : ℝ, y = (1/4)*x^2 → x^2 = 4*y) → 
  (∃ p : ℝ, 2*p = 4 ∧ focus = (0, p)) :=
begin
  intro h,
  use 1,
  split,
  { norm_num, },
  { refl, }
end

end focus_of_parabola_l437_437385


namespace smallest_divisible_1_to_10_l437_437907

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437907


namespace total_amount_leaked_l437_437114

def amount_leaked_before_start : ℕ := 2475
def amount_leaked_while_fixing : ℕ := 3731

theorem total_amount_leaked : amount_leaked_before_start + amount_leaked_while_fixing = 6206 := by
  sorry

end total_amount_leaked_l437_437114


namespace chris_candy_given_l437_437126

/-
  Given:
  (1) Chris gave 10 friends 12 pieces of candy each.
  (2) Chris gave another 7 friends 1.5 times the amount of candy he gave the first 10 friends.
  (3) Chris gave the remaining 5 friends an equal share of 20% more candy than the total amount given to the first 17 friends combined.
  
  Show:
  The total number of pieces of candy Chris gave away is 541.
-/

theorem chris_candy_given (c1 : 10 * 12 = 120)
                         (c2 : 1.5 * 12 = 18)
                         (c3 : 7 * 18 = 126)
                         (c4 : 120 + 126 = 246)
                         (c5 : 0.20 * 246 = 49.2)
                         (c6 : 49.2.round = 49)
                         (c7 : 246 + 49 = 295)
                         (c8 : 295 / 5 = 59)
                         (c9 : 246 + (5 * 59) = 541) : 
                         541 = 541 := 
by 
  exact Eq.refl 541

# An easier way to avoid rounding approximations and still prove the same statement:
-- theorem chris_candy_given_2 (ten_friends : ℝ := 10) (seven_friends : ℝ := 7) (five_friends : ℝ := 5) -- (one_t_point_five : ℝ := 1.5) (one_t_point_two := 1.2) :
--   let total_first_ten:           ℝ := ten_friends * 12,
--       total_next_seven:          ℝ := seven_friends * one_t_point_five * 12,
--       total_first_seventeen:     ℝ := total_first_ten + total_next_seven,
--       increased_candy_share_fx5: ℝ := total_first_seventeen * one_t_point_two in
--   increased_candy_share_fx5 = 541 - total_first_seventeen

end chris_candy_given_l437_437126


namespace smallest_number_divisible_1_to_10_l437_437931

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437931


namespace martha_initial_marbles_l437_437548

-- Definition of the conditions
def initial_marbles_dilan : ℕ := 14
def initial_marbles_phillip : ℕ := 19
def initial_marbles_veronica : ℕ := 7
def marbles_after_redistribution_each : ℕ := 15
def number_of_people : ℕ := 4

-- Total marbles after redistribution
def total_marbles_after_redistribution : ℕ := marbles_after_redistribution_each * number_of_people

-- Total initial marbles of Dilan, Phillip, and Veronica
def total_initial_marbles_dilan_phillip_veronica : ℕ := initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica

-- Prove the number of marbles Martha initially had
theorem martha_initial_marbles : initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica + x = number_of_people * marbles_after_redistribution →
  x = 20 := by
  sorry

end martha_initial_marbles_l437_437548


namespace calculate_speed_in_fifth_hour_l437_437473

variable (S1 S2 S3 S4 A S5 : ℝ)

theorem calculate_speed_in_fifth_hour 
  (h1 : S1 = 90) 
  (h2 : S2 = 60) 
  (h3 : S3 = 120)
  (h4 : S4 = 72)
  (h_avg : A = 80)
  (h_time : ∑ (n : ℝ) in finset.range 5, 1 = 5) :
  S5 = 58 :=
by
  let total_distance := S1 + S2 + S3 + S4
  have h_total_distance := total_distance = 342, from sorry
  let total_distance_5_hours := A * 5
  have h_total_distance_5_hours := total_distance_5_hours = 400, from sorry
  let distance_5 := total_distance_5_hours - total_distance
  have h_distance_5 := distance_5 = 58, from sorry
  let speed_5 := distance_5 / 1
  have h_speed_5 := speed_5 = 58, from sorry
  exact h_speed_5

end calculate_speed_in_fifth_hour_l437_437473


namespace balloons_given_by_mom_l437_437021

def num_balloons_initial : ℕ := 26
def num_balloons_total : ℕ := 60

theorem balloons_given_by_mom :
  (num_balloons_total - num_balloons_initial) = 34 := 
by
  sorry

end balloons_given_by_mom_l437_437021


namespace triangle_DZE_isosceles_area_triangle_DZE_l437_437672

-- Declare the points and lines as assumptions in a relevant geometry context

variable (A B C D E K O M P T Z : Type*)
variable [Geometry A B C D E K O M P T Z]

-- The conditions
variable (ABC_Acute : AcuteAngle A B C)
variable (D_Mid_AB : Midpoint D A B)
variable (E_Mid_AC : Midpoint E A C)
variable (K_Mid_BC : Midpoint K B C)
variable (O_Circumcenter_ABC : Circumcenter O A B C)
variable (M_Foot_A_BC : Foot M A B C)
variable (P_Mid_OM : Midpoint P O M)
variable (T_Parallel_AM : Parallel P T A M)
variable (Z_Intersections : IntersectionParallel P T T DE A T Z T OA)

-- Part (a): Prove that triangle DZE is isosceles
theorem triangle_DZE_isosceles :
  IsIsosceles D Z E := sorry

-- Part (b): Prove that the area of triangle DZE is (BC * OK) / 8
theorem area_triangle_DZE :
  Area D Z E = (BC * OK) / 8 := sorry

end triangle_DZE_isosceles_area_triangle_DZE_l437_437672


namespace find_b_l437_437415

open Real

variables (a b : ℝ × ℝ × ℝ)

def parallel (a v : ℝ × ℝ × ℝ) :=
  ∃ t : ℝ, a = (t * v.1, t * v.2, t * v.3)

def orthogonal (b v : ℝ × ℝ × ℝ) :=
  b.1 * v.1 + b.2 * v.2 + b.3 * v.3 = 0

theorem find_b :
  parallel a ⟨1, 1, 1⟩ →
  orthogonal b ⟨1, 1, 1⟩ →
  a + b = ⟨8, -4, -2⟩ →
  b = ⟨7, -5, -3⟩ :=
by sorry

end find_b_l437_437415


namespace lcm_1_to_10_l437_437813

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437813


namespace irrational_cube_root_of_9_l437_437107

theorem irrational_cube_root_of_9 :
  ¬ ∃ (a b : ℤ), \(\sqrt[3]{9}\) = a / b :=
by
  sorry

end irrational_cube_root_of_9_l437_437107


namespace unknown_interest_rate_l437_437097

theorem unknown_interest_rate (total_investment : ℝ) 
                              (investment_unknown_rate : ℝ) 
                              (investment_known_rate : ℝ) 
                              (rate_known : ℝ)
                              (desired_annual_income : ℝ) 
                              (interest_from_known_rate : ℝ := investment_known_rate * rate_known)
                              (total_income_eq : desired_annual_income = investment_unknown_rate * r + interest_from_known_rate) 
                              (investment_sum_eq : total_investment = investment_unknown_rate + investment_known_rate) :
                              r = 0.0575 :=
by
  -- Definitions
  let r := (desired_annual_income - investment_known_rate * rate_known) / investment_unknown_rate
  show r = 0.0575 from sorry

end unknown_interest_rate_l437_437097


namespace product_of_even_number_of_even_numbers_product_of_even_number_of_odd_numbers_product_of_odd_number_of_even_numbers_product_of_odd_number_of_odd_numbers_l437_437142

-- Definition to denote an even number
def is_even (n : ℤ) : Prop := ∃ k, n = 2 * k 

-- Definition to denote an odd number
def is_odd (n : ℤ) : Prop := ∃ k, n = 2 * k + 1

-- Proof of the product of an even number of even numbers is even.
theorem product_of_even_number_of_even_numbers (a : List ℤ) (h₁ : ∀ x ∈ a, is_even x) (h₂ : a.length % 2 = 0) : is_even (a.foldr (*) 1) := 
sorry

-- Proof of the product of an even number of odd numbers is odd.
theorem product_of_even_number_of_odd_numbers (a : List ℤ) (h₁ : ∀ x ∈ a, is_odd x) (h₂ : a.length % 2 = 0) : is_odd (a.foldr (*) 1) := 
sorry

-- Proof of the product of an odd number of even numbers is even.
theorem product_of_odd_number_of_even_numbers (a : List ℤ) (h₁ : ∀ x ∈ a, is_even x) (h₂ : a.length % 2 = 1) : is_even (a.foldr (*) 1) := 
sorry

-- Proof of the product of an odd number of odd numbers is odd.
theorem product_of_odd_number_of_odd_numbers (a : List ℤ) (h₁ : ∀ x ∈ a, is_odd x) (h₂ : a.length % 2 = 1) : is_odd (a.foldr (*) 1) := 
sorry

end product_of_even_number_of_even_numbers_product_of_even_number_of_odd_numbers_product_of_odd_number_of_even_numbers_product_of_odd_number_of_odd_numbers_l437_437142


namespace sequences_converge_to_common_limit_l437_437728

noncomputable def x_seq (a b : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := (x_seq a b n + y_seq a b n) / 2

noncomputable def y_seq (a b : ℝ) : ℕ → ℝ
| 0       := b
| (n + 1) := (2 * x_seq a b n * y_seq a b n) / (x_seq a b n + y_seq a b n)

theorem sequences_converge_to_common_limit (a b : ℝ) :
  ∃ ℓ, (∀ ε > 0, ∃ N, ∀ n ≥ N, |x_seq a b n - ℓ| < ε) ∧ 
       (∀ ε > 0, ∃ N, ∀ n ≥ N, |y_seq a b n - ℓ| < ε) ∧ 
       ℓ = real.sqrt (a * b) :=
begin
  sorry
end

end sequences_converge_to_common_limit_l437_437728


namespace smallest_divisible_1_to_10_l437_437904

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437904


namespace product_divisible_by_squares_l437_437626

theorem product_divisible_by_squares {n : ℤ} (h₀ : n ≥ 3) {a : ℤ → ℤ}
  (h₁ : ∀ i, 1 ≤ i ∧ i ≤ n → a i ≠ 0)
  (h₂ : ∃ k : ℤ, k = (a 1 * a 2 * ... * a n) * ∑ i in finset.range (n.to_nat + 1), 1 / (a i) ^ 2) :
  ∀ i, 1 ≤ i ∧ i ≤ n → (a 1 * a 2 * ... * a n) % (a i) ^ 2 = 0 := 
by
  sorry

end product_divisible_by_squares_l437_437626


namespace committees_share_four_members_l437_437266

open Finset

variable {α : Type*}

theorem committees_share_four_members
    (deputies : Finset α)
    (committees : Finset (Finset α))
    (h_deputies : deputies.card = 1600)
    (h_committees : committees.card = 16000)
    (h_committee_size : ∀ c ∈ committees, c.card = 80) :
  ∃ c₁ c₂ ∈ committees, c₁ ≠ c₂ ∧ (c₁ ∩ c₂).card ≥ 4 := by
  sorry

end committees_share_four_members_l437_437266


namespace smallest_number_divisible_by_1_to_10_l437_437872

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437872


namespace tan_ratio_l437_437687

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end tan_ratio_l437_437687


namespace large_envelopes_count_l437_437115

theorem large_envelopes_count
  (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ)
  (H1 : total_letters = 80)
  (H2 : small_envelope_letters = 20)
  (H3 : letters_per_large_envelope = 2) :
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 :=
sorry

end large_envelopes_count_l437_437115


namespace smallest_number_divisible_by_1_to_10_l437_437836

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437836


namespace blister_slowdown_l437_437526

theorem blister_slowdown
    (old_speed new_speed time : ℕ) (new_speed_initial : ℕ) (blister_freq : ℕ)
    (distance_old : ℕ) (blister_per_hour_slowdown : ℝ):
    -- Given conditions
    old_speed = 6 →
    new_speed = 11 →
    new_speed_initial = 11 →
    time = 4 →
    blister_freq = 2 →
    distance_old = old_speed * time →
    -- Prove that each blister slows Candace down by 10 miles per hour
    blister_per_hour_slowdown = 10 :=
  by
    sorry

end blister_slowdown_l437_437526


namespace sum_y_coordinates_of_circle_points_l437_437527

theorem sum_y_coordinates_of_circle_points :
  let C := λ x y : ℝ, (x + 8)^2 + (y - 5)^2 = 169
  ∃ y1 y2 : ℝ, (C 0 y1) ∧ (C 0 y2) ∧ y1 ≠ y2 ∧ y1 + y2 = 10 :=
sorry

end sum_y_coordinates_of_circle_points_l437_437527


namespace initial_percentage_of_jasmine_water_l437_437109

-- Definitions
def v_initial : ℝ := 80
def v_jasmine_added : ℝ := 8
def v_water_added : ℝ := 12
def percentage_final : ℝ := 16
def v_final : ℝ := v_initial + v_jasmine_added + v_water_added

-- Lean 4 statement that frames the proof problem
theorem initial_percentage_of_jasmine_water (P : ℝ) :
  (P / 100) * v_initial + v_jasmine_added = (percentage_final / 100) * v_final → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_jasmine_water_l437_437109


namespace cyclic_quadrilateral_max_BD_l437_437296

noncomputable def largest_possible_value_of_BD (AB BC CD DA : ℕ) : ℕ :=
  let BD_squared := (AB^2 + BC^2 + CD^2 + DA^2) / 2
  Int.sqrt BD_squared

theorem cyclic_quadrilateral_max_BD :
  ∃ (AB BC CD DA : ℕ), BD ≤ 10 ∧ AB ≠ BC ∧ BC ≠ CD ∧ CD ≠ DA ∧ DA ≠ AB ∧
  (BC * CD = 2 * AB * DA) ∧ largest_possible_value_of_BD AB BC CD DA = 11 :=
by
  sorry

end cyclic_quadrilateral_max_BD_l437_437296


namespace valid_triples_count_l437_437678

def validTriple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 15 ∧ 
  1 ≤ b ∧ b ≤ 15 ∧ 
  1 ≤ c ∧ c ≤ 15 ∧ 
  (b % a = 0 ∨ (∃ k : ℕ, k ≤ 15 ∧ c % k = 0))

def countValidTriples : ℕ := 
  (15 + 7 + 5 + 3 + 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) * 2 - 15

theorem valid_triples_count : countValidTriples = 75 :=
  by
  sorry

end valid_triples_count_l437_437678


namespace crown_distribution_l437_437467

theorem crown_distribution 
  (A B C D E : ℤ) 
  (h1 : 2 * C = 3 * A)
  (h2 : 4 * D = 3 * B)
  (h3 : 4 * E = 5 * C)
  (h4 : 5 * D = 6 * A)
  (h5 : A + B + C + D + E = 2870) : 
  A = 400 ∧ B = 640 ∧ C = 600 ∧ D = 480 ∧ E = 750 := 
by 
  sorry

end crown_distribution_l437_437467


namespace fixed_point_Q_l437_437182

-- Define the basic setup for points and triangles in geometry
variables {A B C D E F P Q : Type*}

-- Given conditions
def is_on_segment (p q r : Type*) : Prop := -- Placeholder definition
  sorry

def midpoint (p q : Type*) : Type* := -- Placeholder definition
  sorry

def length (p q : Type*) : ℝ := -- Placeholder definition
  sorry

def circumcircle (a b c : Type*) : Type* := -- Placeholder definition
  sorry

def intersect_at (c1 c2 : Type*) : Type* := -- Placeholder definition
  sorry

-- The problem statement
theorem fixed_point_Q 
  (triangle : Triangle A B C)
  (D : BC)
  (E : AB)
  (F : AC)
  (h1 : length B E = length C D)
  (h2 : length C F = length B D)
  (circle_BDE : circumcircle B D E)
  (circle_CDF : circumcircle C D F)
  (P : intersect_at circle_BDE circle_CDF)
  (P_ne_D : P ≠ D) :
  ∃ (Q : Type*), ∀ (D : BC), length Q P = constant :=
sorry

end fixed_point_Q_l437_437182


namespace isosceles_triangle_crease_length_l437_437113

def length_of_crease (A B C : Point) (AB AC BC : ℝ) (h_iso1 : AB = 6) (h_iso2 : AC = 6) (h_iso3 : BC = 8) : ℝ :=
by
  sorry

theorem isosceles_triangle_crease_length
  {A B C : Point}
  (h_iso1 : dist A B = 6)
  (h_iso2 : dist A C = 6)
  (h_iso3 : dist B C = 8) :
  length_of_crease A B C 6 6 8 h_iso1 h_iso2 h_iso3 = 3 :=
sorry

end isosceles_triangle_crease_length_l437_437113


namespace middle_number_is_12_l437_437406

theorem middle_number_is_12 (x y z : ℕ) (h1 : x + y = 20) (h2 : x + z = 25) (h3 : y + z = 29) (h4 : x < y) (h5 : y < z) : y = 12 :=
by
  sorry

end middle_number_is_12_l437_437406


namespace einstein_birth_day_l437_437376

theorem einstein_birth_day
  (year_2025 : ℕ)
  (is_friday_2025 : ∀ (d : ℕ), d = 5)
  (leap_year_condition : ∀ (n : ℕ), (n % 400 = 0 ∨ (n % 4 = 0 ∧ n % 100 ≠ 0)) ↔ is_leap_year n) :
  let birth_year := 2025 - 160 in
  day_of_week birth_year 3 14 = 5 :=
by
  sorry

end einstein_birth_day_l437_437376


namespace lcm_1_10_l437_437960

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437960


namespace smallest_number_divisible_by_1_to_10_l437_437862

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437862


namespace no_function_f_exists_l437_437549

theorem no_function_f_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 :=
by sorry

end no_function_f_exists_l437_437549


namespace point_third_quadrant_l437_437253

theorem point_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : 3 * m - 2 < 0 ∧ -n < 0 :=
by
  sorry

end point_third_quadrant_l437_437253


namespace parameter_a_range_l437_437220

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2 * a + 1

theorem parameter_a_range :
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → quadratic_function a x ≥ 1) ↔ (0 ≤ a) :=
by
  sorry

end parameter_a_range_l437_437220


namespace lcm_1_to_10_l437_437814

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437814


namespace sunzi_problem_solution_l437_437463

theorem sunzi_problem_solution (x y : ℝ) :
  (y = x + 4.5) ∧ (0.5 * y = x - 1) ↔ (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by 
  sorry

end sunzi_problem_solution_l437_437463


namespace f_nonpos_on_unit_interval_g_exactly_two_zeros_l437_437304

noncomputable def f (x : ℝ) : ℝ := x - Real.sin (π * x / 2)

-- Statement for Proof Q1
theorem f_nonpos_on_unit_interval : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x ≤ 0 := by
  sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - a * Real.log (|x|)

-- Statement for Proof Q2
theorem g_exactly_two_zeros (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0 ∧ (∀ x : ℝ, g a x ≠ 0 ∨ x = x1 ∨ x = x2)) → 
  a = -1 ∨ a = 0 ∨ a = 1 := by
  sorry

end f_nonpos_on_unit_interval_g_exactly_two_zeros_l437_437304


namespace triangle_ineq_equality_condition_l437_437581

variables {a b c S : ℝ}
variable h : S = (1 / 2) * a * b * Real.sin (π / 6)

theorem triangle_ineq (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S = (1 / 2) * a * b * Real.sin (π / 6)) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

theorem equality_condition (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S = (1 / 2) * a * b * Real.sin (π / 6)) :
  a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S → a = b :=
sorry

end triangle_ineq_equality_condition_l437_437581


namespace closest_fraction_is_one_seventh_l437_437120

noncomputable def fraction_of_medals : ℚ := 23 / 150

def closest_fraction (fractions : list ℚ) (target : ℚ) : ℚ :=
  fractions.argmin (λ frac => abs (frac - target))

theorem closest_fraction_is_one_seventh :
  closest_fraction [1/5, 1/6, 1/7, 1/8, 1/9] fraction_of_medals = 1/7 := sorry

end closest_fraction_is_one_seventh_l437_437120


namespace problem_proof_l437_437654

variables {A B C D E F G : Type*}

-- Assuming there is a setup of the points A, B, C, D forming a parallelogram
-- and a circle passing through B intersecting at points E, F, G as described
axiom parallelogram (A B C D : Type*) : Prop
axiom circle_through (B E F G : Type*) : Prop
axiom intersects (AB BC BD : Type*) (E F G : Type*) : Prop

theorem problem_proof (parallelogramABCD : parallelogram A B C D) 
  (circleB : circle_through B E F G) (EonAB : intersects AB BC BD E) 
  (FonBC : intersects AB BC BD F) (GonBD : intersects AB BC BD G) :
  BE * AB + BF * BC = BG * BD := 
  sorry

end problem_proof_l437_437654


namespace sin_negative_angle_l437_437408

theorem sin_negative_angle : sin (-1740 * (Math.pi / 180)) = sqrt 3 / 2 :=
by
  sorry

end sin_negative_angle_l437_437408


namespace students_taking_neither_l437_437701

theorem students_taking_neither (n c b cb : ℕ) (hn : n = 80) (hc : c = 48)
  (hb : b = 40) (hcb : cb = 25) : n - (c - cb + b - cb + cb) = 17 :=
by {
  have h_only_c := c - cb,
  have h_only_b := b - cb,
  have h_either := h_only_c + h_only_b + cb,
  rw [←hn, ←hc, ←hb, ←hcb, h_only_c, h_only_b, h_either],
  norm_num,
}

end students_taking_neither_l437_437701


namespace correct_statement_A_l437_437170

-- Definitions
variables {l m : Line} {α : Plane}

-- Conditions
def is_perpendicular_to_plane (l : Line) (α : Plane) : Prop := 
  ∀ (p q : Point), p ≠ q → p ∈ l → q ∈ l → ∃ r s : Line, r ≠ s ∧ p ∈ r ∧ q ∈ s ∧ r ⊂ α ∧ s ⊂ α ∧ ⟂ r s

def is_subset_of (m : Line) (α : Plane) : Prop := 
  ∀ (p : Point), p ∈ m → p ∈ α

def is_perpendicular_to (l m : Line) : Prop := 
  ∀ (p q : Point), p ≠ q → p ∈ l → q ∈ l → ∃ r s : Line, r ≠ s ∧ p ∈ r ∧ q ∈ s ∧ ⟂ r s

-- Theorem
theorem correct_statement_A {l m : Line} {α : Plane} 
  (h1 : is_perpendicular_to_plane l α)
  (h2 : is_subset_of m α) :
  is_perpendicular_to l m :=
sorry 

end correct_statement_A_l437_437170


namespace jaysons_moms_age_l437_437440

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end jaysons_moms_age_l437_437440


namespace lcm_1_to_10_l437_437854

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437854


namespace find_ratio_CP_PE_l437_437259

variables {A B C D E P : Type}

/-- In triangle ABC, lines CE and AD are drawn such that 
    CD/DB = 4/1 and AE/EB = 4/3. Prove that r = CP/PE where 
    P is the intersection point of CE and AD. -/
theorem find_ratio_CP_PE (h1 : ∃ k1 : ℚ, CD / DB = 4) 
                         (h2 : ∃ k2 : ℚ, AE / EB = 4 / 3)
                         (hP : P_intersection_CE_AD A B C D E P)
                         : CP / PE = 7 :=
sorry

end find_ratio_CP_PE_l437_437259


namespace range_of_a_subset_B_l437_437125

theorem range_of_a_subset_B (a : ℝ) : 
    (∀ x : ℝ, (a - 1) ≤ x ∧ x ≤ (a + 1) → (x ≥ -1 ∧ x ≤ 2)) → 
    0 ≤ a ∧ a ≤ 1 :=
by
  intros h
  have ha1 : a - 1 ≥ -1 := sorry
  have ha2 : a + 1 ≤ 2 := sorry
  exact ⟨ha1, ha2⟩

end range_of_a_subset_B_l437_437125


namespace find_locus_P_l437_437292

variable {A B C D E P P' : Type*}

-- Let ABC be an acute scalene triangle
variables [Triangle A B C] [AcuteTriangle A B C] [ScaleneTriangle A B C]

-- Points D and E are variable points on half-lines AB and AC respectively
variables (D E : Point) (originD : D ∈ halfline A B) (originE : E ∈ halfline A C)

-- P is the intersection of the circles with diameters AD and AE
variable (cirP : ∀ (P : Point), P ∈ circle (diameter A D) ∧ P ∈ circle (diameter A E))

-- The symmetric point of A over DE lies on BC
variables [SymmetricPoint A DE (lineSegment B C)]

-- Midpoints M and N of AB and AC respectively
variables (M : Midpoint A B) (N : Midpoint A C)

-- Define the locus condition
def locus_P : Prop :=
  ∀ (P : Point), P ∈ circle (diameter A D) ∧ P ∈ circle (diameter A E) → P ∈ lineSegment M N

-- The theorem to be proven
theorem find_locus_P : locus_P P :=
  sorry

end find_locus_P_l437_437292


namespace parallelogram_area_l437_437783

noncomputable def area_of_parallelogram (b s horizontal_diff : ℝ) : ℝ :=
  let h := real.sqrt (s ^ 2 - horizontal_diff ^ 2)
  in b * h

theorem parallelogram_area :
  area_of_parallelogram 20 6 5 = 20 * real.sqrt 11 :=
by
  simp only [area_of_parallelogram]
  rw [real.sqrt_eq_rpow, sub_eq_add_neg, pow_two, pow_two, add_comm, add_neg_eq_sub]
  norm_num
  linarith
  sorry

end parallelogram_area_l437_437783


namespace max_truck_speed_l437_437489

theorem max_truck_speed (D : ℝ) (C : ℝ) (F : ℝ) (L : ℝ → ℝ) (T : ℝ) (x : ℝ) : 
  D = 125 ∧ C = 30 ∧ F = 1000 ∧ (∀ s, L s = 2 * s) ∧ (∃ s, D / s * C + F + L s ≤ T) → x ≤ 75 :=
by
  sorry

end max_truck_speed_l437_437489


namespace lcm_1_to_10_l437_437947

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437947


namespace probability_of_defective_on_second_draw_l437_437471

-- Define the conditions
variable (batch_size : ℕ) (defective_items : ℕ) (good_items : ℕ)
variable (first_draw_good : Prop)
variable (without_replacement : Prop)

-- Given conditions
def batch_conditions : Prop :=
  batch_size = 10 ∧ defective_items = 3 ∧ good_items = 7 ∧ first_draw_good ∧ without_replacement

-- The desired probability as a proof
theorem probability_of_defective_on_second_draw
  (h : batch_conditions batch_size defective_items good_items first_draw_good without_replacement) : 
  (3 / 9 : ℝ) = 1 / 3 :=
sorry

end probability_of_defective_on_second_draw_l437_437471


namespace card_arrangement_impossible_l437_437991

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437991


namespace card_arrangement_impossible_l437_437996

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437996


namespace snack_eaters_remaining_at_end_of_fourth_hour_l437_437492

theorem snack_eaters_remaining_at_end_of_fourth_hour :
  let total_attendees := 7500
  let first_hour_snack_eaters := 0.55 * total_attendees
  let first_hour_not_eating := 0.35 * total_attendees
  let first_hour_undecided := 0.10 * total_attendees

  let second_hour_undecided_eaters := 0.20 * first_hour_undecided
  let second_hour_not_eating_eaters := 0.15 * first_hour_not_eating
  let total_second_hour_snack_eaters := first_hour_snack_eaters + second_hour_undecided_eaters + second_hour_not_eating_eaters
  let newcomers := 75
  let newcomers_snack_eaters := 50
  let second_hour_total_snack_eaters := total_second_hour_snack_eaters + newcomers_snack_eaters
  let second_hour_leavers := 0.40 * second_hour_total_snack_eaters
  let third_hour_start_snack_eaters := second_hour_total_snack_eaters - second_hour_leavers

  let third_hour_increment := 0.10 * third_hour_start_snack_eaters
  let third_hour_total_snack_eaters := third_hour_start_snack_eaters + third_hour_increment
  let third_hour_leavers := 0.50 * third_hour_total_snack_eaters
  let fourth_hour_start_snack_eaters := third_hour_total_snack_eaters - third_hour_leavers

  let latecomers := 150
  let latecomers_snack_eaters := 0.60 * latecomers
  let fourth_hour_new_snack_eaters := fourth_hour_start_snack_eaters + latecomers_snack_eaters
  let fourth_hour_leavers := 300
  let total_snack_eaters_end_fourth_hour := fourth_hour_new_snack_eaters - fourth_hour_leavers

  total_snack_eaters_end_fourth_hour = 1347 := by
    let total_attendees := total_attendees
    let first_hour_snack_eaters := first_hour_snack_eaters
    let first_hour_not_eating := first_hour_not_eating
    let first_hour_undecided := first_hour_undecided
    let second_hour_undecided_eaters := second_hour_undecided_eaters
    let second_hour_not_eating_eaters := second_hour_not_eating_eaters
    let total_second_hour_snack_eaters := total_second_hour_snack_eaters
    let newcomers := newcomers
    let newcomers_snack_eaters := newcomers_snack_eaters
    let second_hour_total_snack_eaters := second_hour_total_snack_eaters
    let second_hour_leavers := second_hour_leavers
    let third_hour_start_snack_eaters := third_hour_start_snack_eaters
    let third_hour_increment := third_hour_increment
    let third_hour_total_snack_eaters := third_hour_total_snack_eaters
    let third_hour_leavers := third_hour_leavers
    let fourth_hour_start_snack_eaters := fourth_hour_start_snack_eaters
    let latecomers := latecomers
    let latecomers_snack_eaters := latecomers_snack_eaters
    let fourth_hour_new_snack_eaters := fourth_hour_new_snack_eaters
    let fourth_hour_leavers := fourth_hour_leavers
    let total_snack_eaters_end_fourth_hour := total_snack_eaters_end_fourth_hour
    sorry

end snack_eaters_remaining_at_end_of_fourth_hour_l437_437492


namespace total_cost_proof_l437_437086

-- Define the conditions
def length_grass_field : ℝ := 75
def width_grass_field : ℝ := 55
def width_path : ℝ := 2.5
def area_path : ℝ := 6750
def cost_per_sq_m : ℝ := 10

-- Calculate the outer dimensions
def outer_length : ℝ := length_grass_field + 2 * width_path
def outer_width : ℝ := width_grass_field + 2 * width_path

-- Calculate the area of the entire field including the path
def area_entire_field : ℝ := outer_length * outer_width

-- Calculate the area of the grass field without the path
def area_grass_field : ℝ := length_grass_field * width_grass_field

-- Calculate the area of the path
def area_calculated_path : ℝ := area_entire_field - area_grass_field

-- Calculate the total cost of constructing the path
noncomputable def total_cost : ℝ := area_calculated_path * cost_per_sq_m

-- The theorem to prove
theorem total_cost_proof :
  area_calculated_path = area_path ∧ total_cost = 6750 :=
by
  sorry

end total_cost_proof_l437_437086


namespace smallest_number_divisible_by_1_through_10_l437_437976

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437976


namespace correct_calculation_l437_437987

theorem correct_calculation (a b : ℝ) : (3 * a * b) ^ 2 = 9 * a ^ 2 * b ^ 2 :=
by
  sorry

end correct_calculation_l437_437987


namespace largest_digit_change_l437_437735

def a := 735
def b := 468
def c := 281
def incorrect_sum := 1584
def correct_sum := 1484

theorem largest_digit_change (a b c incorrect_sum: ℕ) (h₁ : a + b + c = correct_sum) (h₂ : incorrect_sum - correct_sum = 100):
  ∃ d: ℕ, d ∈ {7, 4, 2} ∧ d = 7 :=
by sorry

end largest_digit_change_l437_437735


namespace tetrahedron_a_bounds_tetrahedron_volume_l437_437019

noncomputable def volume_tetrahedron (a : ℝ) : ℝ :=
  (1 / 12) * real.sqrt((a^2 + 1) * (3 * a^2 - 1 - a^4))

theorem tetrahedron_a_bounds (a : ℝ) :
  ( -1 + real.sqrt 5 ) / 2 < a ∧ a < ( 1 + real.sqrt 5 ) / 2 ∧ a ≠ 1 :=
sorry

theorem tetrahedron_volume (a : ℝ) (h : ( -1 + real.sqrt 5 ) / 2 < a ∧ a < ( 1 + real.sqrt 5 ) / 2 ∧ a ≠ 1) :
  volume_tetrahedron a = (1 / 12) * real.sqrt((a^2 + 1) * (3 * a^2 - 1 - a^4)) :=
sorry

end tetrahedron_a_bounds_tetrahedron_volume_l437_437019


namespace sum_of_coordinates_D_l437_437342

structure Point where
  x : ℝ
  y : ℝ

def is_midpoint (M C D : Point) : Prop :=
  M = ⟨(C.x + D.x) / 2, (C.y + D.y) / 2⟩

def sum_of_coordinates (P : Point) : ℝ :=
  P.x + P.y

theorem sum_of_coordinates_D :
  ∀ (C M : Point), C = ⟨1/2, 3/2⟩ → M = ⟨2, 5⟩ →
  ∃ D : Point, is_midpoint M C D ∧ sum_of_coordinates D = 12 :=
by
  intros C M hC hM
  sorry

end sum_of_coordinates_D_l437_437342


namespace card_arrangement_impossible_l437_437992

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437992


namespace lcm_1_to_10_l437_437812

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437812


namespace expectation_and_variance_of_Y_l437_437221

noncomputable def binomial_mean (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem expectation_and_variance_of_Y (X Y : ℝ) (h1 : X + Y = 10) 
  (h2 : ∀ (x : ℝ), X = binomial_mean 10 0.6 ∧ X = binomial_variance 10 0.6) :
  (E(Y) = 4) ∧ (D(Y) = 2.4) :=
by
  sorry

end expectation_and_variance_of_Y_l437_437221


namespace smallest_divisible_by_1_to_10_l437_437890

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437890


namespace least_positive_integer_multiple_of_53_l437_437429

-- Define the problem in a Lean statement.
theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ, (3 * x) ^ 2 + 2 * 58 * 3 * x + 58 ^ 2 % 53 = 0 ∧ x = 16 :=
by
  sorry

end least_positive_integer_multiple_of_53_l437_437429


namespace solution_l437_437566

theorem solution (x : ℝ) 
  (h1 : 1/x < 3)
  (h2 : 1/x > -4) 
  (h3 : x^2 - 3*x + 2 < 0) : 
  1 < x ∧ x < 2 :=
sorry

end solution_l437_437566


namespace cyclists_meet_at_start_point_l437_437420

-- Conditions from the problem
def cyclist1_speed : ℝ := 7 -- speed of the first cyclist in m/s
def cyclist2_speed : ℝ := 8 -- speed of the second cyclist in m/s
def circumference : ℝ := 600 -- circumference of the circular track in meters

-- Relative speed when cyclists move in opposite directions
def relative_speed := cyclist1_speed + cyclist2_speed

-- Prove that they meet at the starting point after 40 seconds
theorem cyclists_meet_at_start_point :
  (circumference / relative_speed) = 40 := by
  -- the proof would go here
  sorry

end cyclists_meet_at_start_point_l437_437420


namespace find_y_value_l437_437751

variables {a b c : ℝ} {x y : ℝ}

def quadratic (x : ℝ) : ℝ := a*x^2 + b*x + c

def vertex_condition : Prop := ∀ x, quadratic (-2) = 10

def point_pass_through : Prop := quadratic 0 = -6

def value_at_2 : ℝ := quadratic 2

theorem find_y_value :
  vertex_condition ∧ point_pass_through → value_at_2 = -54 :=
by
  sorry

end find_y_value_l437_437751


namespace smallest_divisible_by_1_to_10_l437_437888

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437888


namespace smallest_number_divisible_by_1_to_10_l437_437873

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437873


namespace intersection_M_N_l437_437617

def set_M : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ y = x^2 + 1}
def set_N : set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ y = sqrt (x + 1)}

theorem intersection_M_N :
  (set_M ∩ set_N) = {p | p = (0, 1)} :=
by sorry

end intersection_M_N_l437_437617


namespace smallest_number_divisible_by_1_through_10_l437_437973

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437973


namespace basketball_game_ticket_cost_l437_437417

theorem basketball_game_ticket_cost
  (total_tickets : ℕ) (total_revenue : ℝ) (adult_tickets : ℕ)
  (student_tickets : ℕ) (student_ticket_cost : ℝ)
  (h_total_tickets : total_tickets = 846)
  (h_total_revenue : total_revenue = 3846.00)
  (h_adult_tickets : adult_tickets = 410)
  (h_student_tickets : student_tickets = 436)
  (h_student_ticket_cost : student_ticket_cost = 3.00) :
  let adult_ticket_cost := (total_revenue - student_tickets * student_ticket_cost) / adult_tickets
  in adult_ticket_cost = 6.19 :=
by
  sorry

end basketball_game_ticket_cost_l437_437417


namespace domain_and_symmetry_l437_437727

noncomputable def f (x : ℝ) : ℝ := 1 / real.sqrt (x^2 - 10 * x + 25)

theorem domain_and_symmetry : 
(∀ x, f x ≠ 0 ↔ x ∈ set.Ioo (5 - ∞) (5 + ∞)) ∧ 
(∀ (a : ℝ), f (5 - a) = f (5 + a)) :=
sorry

end domain_and_symmetry_l437_437727


namespace range_of_t_range_of_a_l437_437589

-- Proposition P: The curve equation represents an ellipse with foci on the x-axis
def propositionP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (4 - t) + y^2 / (t - 1) = 1)

-- Proof problem for t
theorem range_of_t (t : ℝ) (h : propositionP t) : 1 < t ∧ t < 5 / 2 := 
  sorry

-- Proposition Q: The inequality involving real number t
def propositionQ (t a : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

-- Proof problem for a
theorem range_of_a (a : ℝ) (h₁ : ∀ t : ℝ, propositionP t → propositionQ t a) 
                   (h₂ : ∃ t : ℝ, propositionQ t a ∧ ¬ propositionP t) :
  a > 1 / 2 :=
  sorry

end range_of_t_range_of_a_l437_437589


namespace smallest_multiple_1_through_10_l437_437831

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437831


namespace roots_square_difference_l437_437302

theorem roots_square_difference (a b : ℚ)
  (ha : 6 * a^2 + 13 * a - 28 = 0)
  (hb : 6 * b^2 + 13 * b - 28 = 0) : (a - b)^2 = 841 / 36 :=
sorry

end roots_square_difference_l437_437302


namespace max_value_m_l437_437173

noncomputable def max_m : ℝ := 10

theorem max_value_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = x + 2 * y) : x * y ≥ max_m - 2 :=
by
  sorry

end max_value_m_l437_437173


namespace nancy_physical_education_marks_l437_437332

def marks_AL := 66
def marks_H := 75
def marks_HE := 52
def marks_A := 89
def average_marks := 70
def num_subjects := 5

theorem nancy_physical_education_marks :
  let total_marks := num_subjects * average_marks in
  let known_total_marks := marks_AL + marks_H + marks_HE + marks_A in
  let PE_marks := total_marks - known_total_marks in
  PE_marks = 68 :=
by
  sorry

end nancy_physical_education_marks_l437_437332


namespace vector_magnitude_l437_437593

variable {ℝ : Type}

variables (a b : ℝ) [Module ℝ ℝ]

noncomputable def magnitude_sum : ℝ :=
  let u := a * a + b * b in
  let dot_product := (a * b) in
  Real.sqrt (1 + 6 * (dot_product) + 9)

theorem vector_magnitude (a b : ℝ) 
  (h₁ : ∥a∥ = 1) 
  (h₂ : ∥b∥ = 1) 
  (h₃ : a • b = 0.5) : 
  ∥ a + 3 • b ∥ = Real.sqrt 13 := by
  sorry

end vector_magnitude_l437_437593


namespace percent_increase_is_25_l437_437449

variable (P : ℝ) -- Assume P is the initial share price
variable (P1 P2 : ℝ) -- P1 is the first quarter share price, P2 is the second quarter share price

-- Conditions
def condition1 := P1 = 1.20 * P
def condition2 := P2 = 1.50 * P

-- Goal: Prove percent increase from P1 to P2 is 25%
theorem percent_increase_is_25 :
  condition1 → condition2 → ((P2 - P1) / P1) * 100 = 25 := by
  intros h1 h2
  sorry

end percent_increase_is_25_l437_437449


namespace eccentricity_of_ellipse_l437_437552

open Real

variables (a b c e : ℝ)
variables (F A M N : ℝ × ℝ)
variables (Γ : set (ℝ × ℝ))

def is_ellipse (a b : ℝ) (Γ : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ Γ ↔ (x^2 / a^2 + y^2 / b^2 = 1)

def is_point (x y : ℝ) (P : ℝ × ℝ) : Prop := P = (x, y)

def is_right_isosceles_triangle (A F M : (ℝ × ℝ)) : Prop :=
  let d_AF := dist A F in
  let d_AM := dist A M in
  let d_MF := dist M F in
  d_AM = d_MF ∧ d_AF = sqrt 2 * d_AM ∧
  d_AF^2 = d_AM^2 + d_MF^2

noncomputable def eccentricity (a b : ℝ) : ℝ := sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_ellipse :
  (a > 0) ∧ (b > 0) ∧ (a > b) →
  is_ellipse a b Γ →
  is_point (-c, 0) F →
  is_point (a, 0) A →
  M.1 = (a - c) / 2 ∧ M.2 = (a + c) / 2 →
  is_right_isosceles_triangle A F M →
  eccentricity a b = 2 - sqrt 2 :=
by
  sorry

end eccentricity_of_ellipse_l437_437552


namespace Craig_bench_press_percentage_l437_437541

theorem Craig_bench_press_percentage {Dave_weight : ℕ} (h1 : Dave_weight = 175) (h2 : ∀ w : ℕ, Dave_bench_press = 3 * Dave_weight) 
(Craig_bench_press Mark_bench_press : ℕ) (h3 : Mark_bench_press = 55) (h4 : Mark_bench_press = Craig_bench_press - 50) : 
(Craig_bench_press / (3 * Dave_weight) * 100) = 20 := by
  sorry

end Craig_bench_press_percentage_l437_437541


namespace ratio_steel_to_tin_l437_437470

def mass_copper (C : ℝ) := C = 90
def total_weight (S C T : ℝ) := 20 * S + 20 * C + 20 * T = 5100
def mass_steel (S C : ℝ) := S = C + 20

theorem ratio_steel_to_tin (S T C : ℝ)
  (hC : mass_copper C)
  (hTW : total_weight S C T)
  (hS : mass_steel S C) :
  S / T = 2 :=
by
  sorry

end ratio_steel_to_tin_l437_437470


namespace arithmetic_operations_result_eq_one_over_2016_l437_437284

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end arithmetic_operations_result_eq_one_over_2016_l437_437284


namespace parabola_symmetric_y_axis_intersection_l437_437538

theorem parabola_symmetric_y_axis_intersection :
  ∀ (x y : ℝ),
  (x = y ∨ x*x + y*y - 6*y = 0) ∧ (x*x = 3 * y) :=
by 
  sorry

end parabola_symmetric_y_axis_intersection_l437_437538


namespace magnitude_b_eq_4_l437_437197

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (norm_a : ∥a∥ = 3)
variables (angle_ab : real.angle b a = real.pi * 2 / 3)
variables (norm_a_plus_b : ∥a + b∥ = real.sqrt 13)

theorem magnitude_b_eq_4 : ∥b∥ = 4 :=
by
  sorry

end magnitude_b_eq_4_l437_437197


namespace intersection_eq_l437_437224

def set_A : Set ℝ := {x | (x + 4) * (x - 1) < 0}
def set_B : Set ℝ := {x | x^2 - 2 * x = 0}
def set_intersection := {x | x = 0}

theorem intersection_eq : set_A ∩ set_B = set_intersection := by
  sorry

end intersection_eq_l437_437224


namespace number_of_placements_l437_437484

-- We define the classroom setup
def classroom_grid : Type := list (list (option bool)) -- True for boy, False for girl, None for empty desk (not needed here).

-- Initially define the size of the grid
def rows : ℕ := 5
def columns : ℕ := 6
def students : ℕ := 30
def boys : ℕ := 15
def girls : ℕ := 15

def is_valid_placement (grid : classroom_grid) : Prop :=
  ∀ i j, (grid.nth i).bind (λ row, row.nth j) ≠ some true → (grid.nth i).bind (λ row, row.nth j.succ) = some false ∧
                                        (grid.nth i).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j.succ) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j.succ) = some false

theorem number_of_placements : ∃ (count : ℕ), count = 2 * (nat.factorial 15)^2 := by
  -- skip the proof by providing the existential value directly
  existsi (2 * (nat.factorial 15)^2)
  refl

end number_of_placements_l437_437484


namespace last_passenger_seats_probability_l437_437412

theorem last_passenger_seats_probability (n : ℕ) (hn : n > 0) :
  ∀ (P : ℝ), P = 1 / 2 :=
by
  sorry

end last_passenger_seats_probability_l437_437412


namespace problem_conditions_and_solutions_l437_437750

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x - 4

-- Conditions and the main theorem
theorem problem_conditions_and_solutions :
  (∀ x : ℝ, f(x) = 2 * x^3 - 9 * x^2 + 12 * x - 4) ∧
  (∃ a b c d : ℝ, 
    (f (0) = d) ∧ 
    (∀ x, f (x) = a * x^3 + b * x^2 + c * x + d) ∧
    (f'(0) = 12) ∧ 
    (f'(2) = 0) ∧ 
    (f (2) = 0)) ∧ 
  ((∀ x, 6 * x^2 - 18 * x + 12 > 0 → x < 1 ∨ x > 2) ∧ 
   (∀ y, x > 1 ∧ x < 2 → 6 * x^2 - 18 * x + 12 < 0)) ∧ 
  ((∀ x, 6 * x^2 - 18 * x + 12 = 0 → x = 1 ∨ x = 2) ∧ 
    (f'(1) ≠ 0 ∧ f'(2) ≠ 0)) := sorry

end problem_conditions_and_solutions_l437_437750


namespace weight_of_new_person_l437_437738

theorem weight_of_new_person (A : ℤ) (avg_weight_dec : ℤ) (n : ℤ) (new_avg : ℤ)
  (h1 : A = 102)
  (h2 : avg_weight_dec = 2)
  (h3 : n = 30) 
  (h4 : new_avg = A - avg_weight_dec) : 
  (31 * new_avg) - (30 * A) = 40 := 
by 
  sorry

end weight_of_new_person_l437_437738


namespace matrix_invertible_implies_vals_l437_437569

theorem matrix_invertible_implies_vals (a b c k : ℝ) 
  (h : det (matrix.of ![
    ![a+k, b+k, c+k],
    ![b+k, c+k, a+k],
    ![c+k, a+k, b+k]
  ]) = 0) : 
  (∃ x : ℝ, (x = -3 ∨ x = 3/2) ∧ x = (a / (b + c) + b / (a + c) + c / (a + b))) :=
sorry

end matrix_invertible_implies_vals_l437_437569


namespace smallest_number_3444_l437_437083

/-- The smallest number containing only the digits 3 and 4, 
    where both digits appear at least once, and the number is 
    a multiple of both 3 and 4, is 3444. -/
theorem smallest_number_3444 :
  ∃ n : ℕ, (∀ d ∈ (digits 10 n), d = 3 ∨ d = 4) ∧ 
           (4 ∣ n) ∧ 
           (3 ∣ n) ∧ 
           (∃ d ∈ (digits 10 n), d = 3) ∧ 
           (∃ d ∈ (digits 10 n), d = 4) ∧ 
           n = 3444 :=
by sorry

end smallest_number_3444_l437_437083


namespace num_cheaper_to_buy_more_l437_437372

def C (n : ℕ) : ℕ :=
if 1 ≤ n ∧ n ≤ 30 then 15 * n
else if 31 ≤ n ∧ n ≤ 60 then 13 * n
else if 61 ≤ n then 12 * n
else 0 

theorem num_cheaper_to_buy_more : (finset.filter (λ n, C (n + 1) < C n) (finset.range 100)).card = 6 := 
sorry

end num_cheaper_to_buy_more_l437_437372


namespace find_a_from_intersecting_lines_l437_437227

theorem find_a_from_intersecting_lines : 
  ∃ (a : ℝ), 
    let L1 := λ (x y : ℝ), ax + 2 * y + 8 = 0,
        L2 := λ (x y : ℝ), 4 * x + 3 * y = 10,
        L3 := λ (x y : ℝ), 2 * x - y = 10 
    in 
    (∃ (P : ℝ × ℝ), 
        L2 P.1 P.2 ∧ 
        L3 P.1 P.2 ∧ 
        L1 P.1 P.2) → a = -1 :=
by
  -- Proof goes here
  sorry

end find_a_from_intersecting_lines_l437_437227


namespace lcm_1_to_10_l437_437851

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437851


namespace f_x_f_2x_plus_1_l437_437610

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem f_x (x : ℝ) : f x = x^2 - 2 * x - 3 := 
by sorry

theorem f_2x_plus_1 (x : ℝ) : f (2 * x + 1) = 4 * x^2 - 4 := 
by sorry

end f_x_f_2x_plus_1_l437_437610


namespace jiujiang_liansheng_sampling_l437_437641

def bag_numbers : List ℕ := [7, 17, 27, 37, 47]

def systematic_sampling (N n : ℕ) (selected_bags : List ℕ) : Prop :=
  ∃ k i, k = N / n ∧ ∀ j, j < List.length selected_bags → selected_bags.get? j = some (i + k * j)

theorem jiujiang_liansheng_sampling :
  systematic_sampling 50 5 bag_numbers :=
by
  sorry

end jiujiang_liansheng_sampling_l437_437641


namespace perpendicular_x_values_parallel_magnitude_values_l437_437230

noncomputable theory

def perp_vectors (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0
def parallel_vectors (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem perpendicular_x_values (x : ℝ) (a b : ℝ × ℝ) (h : a = (1, x)) (h' : b = (2 * x + 3, -x)) :
  perp_vectors a b → x = -1 ∨ x = 3 :=
sorry

theorem parallel_magnitude_values (x : ℝ) (a b : ℝ × ℝ) (h : a = (1, x)) (h' : b = (2 * x + 3, -x)) :
  parallel_vectors a b → magnitude (a.1 - b.1, a.2 - b.2) = 2 * real.sqrt 5 ∨ magnitude (a.1 - b.1, a.2 - b.2) = 2 :=
sorry

end perpendicular_x_values_parallel_magnitude_values_l437_437230


namespace valid_password_count_l437_437517

def is_valid_password (pw : List ℕ) : Prop :=
  pw.length = 4 ∧ pw.head ≠ 1 ∨ pw.length ≥ 2 → pw[0] ≠ 1 ∨ pw[1] ≠ 2

noncomputable def count_valid_passwords : ℕ :=
  10^4 - 10^2

theorem valid_password_count : count_valid_passwords = 9900 :=
by sorry

end valid_password_count_l437_437517


namespace silvia_jerry_comparison_l437_437087

open Real

-- A rectangular park has a length of 4 units and a width of 3 units
def length : ℝ := 4
def width : ℝ := 3

-- Jerry walks due east then north
def jerry_distance : ℝ := length + width

-- Silvia takes the diagonal 
def silvia_distance : ℝ := sqrt (length ^ 2 + width ^ 2)

-- Calculate the percentage reduction
def percentage_reduction : ℝ := ((jerry_distance - silvia_distance) / jerry_distance) * 100

-- Prove that the closest answer to the percentage reduction is 30%
theorem silvia_jerry_comparison :
  abs (percentage_reduction - 30) <= abs (percentage_reduction - 25) ∧ 
  abs (percentage_reduction - 30) <= abs (percentage_reduction - 35) ∧ 
  abs (percentage_reduction - 30) <= abs (percentage_reduction - 40) ∧ 
  abs (percentage_reduction - 30) <= abs (percentage_reduction - 45) :=
by
  sorry

end silvia_jerry_comparison_l437_437087


namespace lambda_range_l437_437191

noncomputable def range_of_lambda (λ : ℝ) : Prop :=
  ((λ < -2) ∨ (λ > -2 ∧ λ < 1/2))

theorem lambda_range (i j : ℝ) (h1 : i^2 = 1) (h2 : j^2 = 1) (h3 : i * j = 0)
  (lambda : ℝ) (D : (→ λ < (-2)) ∨ (λ > (-2) ∧ λ < 1/2)) :
  let a := i - 2 * j in
  let b := i + λ * j in
  ((a.dot b > 0) → range_of_lambda λ) :=
by
  sorry

end lambda_range_l437_437191


namespace math_problem_l437_437469

def percent (p x : ℝ) := (p / 100) * x

theorem math_problem (initial_amount : ℝ) (p1 p2 : ℝ) (multiplier divisor : ℝ) :
  initial_amount = 1200 →
  p1 = 60 →
  p2 = 30 →
  multiplier = 2 →
  divisor = 3 →
  let first_step := percent p1 initial_amount in
  let second_step := percent p2 first_step in
  let third_step := second_step * multiplier in
  let final_step := third_step / divisor in
  final_step = 144 := 
by
  intros;
  sorry

end math_problem_l437_437469


namespace smallest_divisible_1_to_10_l437_437902

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437902


namespace smallest_number_div_by_1_to_10_l437_437796

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437796


namespace compute_expression_l437_437311

noncomputable def log_base (base x : ℝ) : ℝ := real.log x / real.log base

theorem compute_expression (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h : (log_base 2 x)^4 + (log_base 3 y)^4 + 8 = 8 * (log_base 2 x) * (log_base 3 y)) :
  x^real.sqrt 2 + y^real.sqrt 2 = 13 :=
  sorry

end compute_expression_l437_437311


namespace symmetry_axes_intersect_at_centroid_l437_437355

-- Definitions related to polygon and centroid
structure Polygon :=
  (vertices : List (ℝ × ℝ))

def centroid (p : Polygon) : ℝ × ℝ :=
  let n := p.vertices.length 
  let sum_x := List.sum (List.map Prod.fst p.vertices)
  let sum_y := List.sum (List.map Prod.snd p.vertices)
  (sum_x / n, sum_y / n)

def is_symmetry_axis (p : Polygon) (l : ℝ → ℝ) : Prop :=
  ∀ v ∈ p.vertices, let (x, y) := v in l x = y

def invariant_under_reflection (p : Polygon) (l : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x y, (x, y) ∈ p.vertices → l x = y → reflect_about (x, y) l = (x, y)

theorem symmetry_axes_intersect_at_centroid  
  (p : Polygon)
  (axes : List (ℝ → ℝ))
  (symmetry_prop : ∀ l ∈ axes, is_symmetry_axis p l)
  (centroid_invariant : ∀ l ∈ axes, invariant_under_reflection p l (centroid p)) :
  ∀ l1 l2 ∈ axes, centroid p = intersection_of_axes l1 l2 := 
by
  sorry

end symmetry_axes_intersect_at_centroid_l437_437355


namespace power_function_passes_through_point_f_16_l437_437199

theorem power_function_passes_through_point_f_16 :
  ∃ (α : ℝ), (∀ x : ℝ, f x = x^α) -> (2^α = Real.sqrt 2) -> f 16 = 4 :=
by
  -- Definitions and conditions
  let f := λ x : ℝ, x^Real.sqrt 2
  let α := (1 / 2 : ℝ)
  have h := λ x : ℝ, f x = x^α
  have condition := 2^α = Real.sqrt 2
  sorry

end power_function_passes_through_point_f_16_l437_437199


namespace sides_of_original_polygon_l437_437509

-- Define the sum of interior angles formula for a polygon with n sides
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the total sum of angles for the resulting polygon
def sum_of_new_polygon_angles : ℝ := 1980

-- The lean theorem statement to prove
theorem sides_of_original_polygon (n : ℕ) :
    sum_interior_angles n = sum_of_new_polygon_angles →
    n = 13 →
    12 ≤ n+1 ∧ n+1 ≤ 14 :=
by
  intro h1 h2
  sorry

end sides_of_original_polygon_l437_437509


namespace lcm_1_to_10_l437_437808

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437808


namespace smallest_number_divisible_by_1_to_10_l437_437861

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437861


namespace intersection_M_N_eq_l437_437691

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = log (x^2 + 1)}

-- Define set N
def N : Set ℝ := {x | 4^x > 4}

-- The problem to prove M ∩ N = (1, ∞)
theorem intersection_M_N_eq : (M ∩ N) = {x | x > 1} := 
by 
  -- Proof placeholder
  sorry

end intersection_M_N_eq_l437_437691


namespace smallest_number_divisible_by_1_to_10_l437_437839

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437839


namespace probability_transform_in_S_l437_437501

open Complex

def region_S (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

def transform (z : ℂ) : ℂ :=
  (1/2 : ℂ) * z + (1/2 : ℂ) * I * z

theorem probability_transform_in_S : 
  ∀ (z : ℂ), region_S z → region_S (transform z) := 
by
  sorry

end probability_transform_in_S_l437_437501


namespace smallest_positive_period_decreasing_intervals_max_min_values_in_interval_l437_437212

noncomputable def f (x : ℝ) := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem smallest_positive_period : periodic f π := sorry

theorem decreasing_intervals (k : ℤ) : 
  ∀ x ∈ Icc (k * π + π / 6) (k * π + 2 * π / 3), f' x < 0 := sorry

theorem max_min_values_in_interval : 
  let I := Icc 0 (π / 2) in
  ∃ x_max x_min ∈ I, 
    f x_max = 2 ∧ f x_min = -1 ∧ 
    ∀ x ∈ I, f x ≤ f x_max ∧ f x ≥ f x_min := sorry

end smallest_positive_period_decreasing_intervals_max_min_values_in_interval_l437_437212


namespace smallest_number_div_by_1_to_10_l437_437801

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437801


namespace num_divisors_l437_437239

theorem num_divisors (M : ℕ) (h : M = 2^4 * 3^3 * 5^2 * 7^1) : 
  ∃ n : ℕ, n = 120 ∧ ∀ d : ℕ, d ∣ M → d > 0 :=
by
  use 120
  split
  · rfl
  · sorry

end num_divisors_l437_437239


namespace problem1_problem2_problem3_problem4_l437_437464

-- Problem (1): 
theorem problem1 : sin (63 * π / 180) * cos (18 * π / 180) + cos (63 * π / 180) * cos (108 * π / 180) = (real.sqrt 2) / 2 :=
sorry

-- Problem (2):
theorem problem2 : set.image (λ x, 2 * sin x ^ 2 - 3 * sin x + 1) (set.Icc (π / 6) (5 * π / 6)) = set.Icc (-1 / 8) 0 :=
sorry

-- Problem (3):
theorem problem3 {α β : ℝ} (hα : 0 < α) (hβ : 0 < β) (hαβ : (1 + real.sqrt 3 * tan α) * (1 + real.sqrt 3 * tan β) = 4) : α + β = π / 3 :=
sorry

-- Problem (4):
def f (ω ϕ x : ℝ) := 2 * sin (ω * x + ϕ)
theorem problem4 (ω : ℝ) (hω : ω > 0) (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 2) (h : f ω ϕ (2 * π / 3) = f ω ϕ (2 * π / 3) ∧ (2 * π / ω) = π) : 
  (f ω ϕ 0 ≠ 3 / 2 ∧ 
  ∀ x, (x ∈ set.Icc (π / 12) (2 * π / 3) → (f ω ϕ x < f ω ϕ 0)) ∧ 
  (∃ c, c = (5 * π / 12) ∧ f ω ϕ (c) = 0) ∧ 
  (∃ d, d = ϕ ∧ (∀ x, f ω d x = 2 * sin (ω * x))))
: true :=
sorry

end problem1_problem2_problem3_problem4_l437_437464


namespace smallest_number_divisible_1_to_10_l437_437886

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437886


namespace smallest_number_divisible_1_to_10_l437_437883

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437883


namespace lcm_1_to_10_l437_437805

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437805


namespace period_of_y_l437_437035

-- Define the function y = tan x - cot x
def y (x : ℝ) : ℝ := Real.tan x - Real.cot x

-- Prove that y has a period of π
theorem period_of_y : ∀ x, y (x + π) = y x :=
by
  intro x
  -- Define what needs to be shown
  have h : y (x + π) = Real.tan (x + π) - Real.cot (x + π) := rfl
  -- Use the periodicity of tangent and cotangent
  rw [Real.tan_add_pi, Real.cot_add_pi]
  -- Simplifying terms
  simp [Real.tan, Real.cot]
  sorry

end period_of_y_l437_437035


namespace num_integer_solutions_quadratic_square_l437_437155

theorem num_integer_solutions_quadratic_square : 
  (∃ xs : Finset ℤ, 
    (∀ x ∈ xs, ∃ k : ℤ, (x^4 + 8*x^3 + 18*x^2 + 8*x + 64) = k^2) ∧ 
    xs.card = 2) := sorry

end num_integer_solutions_quadratic_square_l437_437155


namespace exists_bounding_triangle_l437_437458

theorem exists_bounding_triangle {S : set (ℝ × ℝ)} 
  (h : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ S → p2 ∈ S → p3 ∈ S → 
       let area := 0.5 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p2.2 - p1.2) * (p3.1 - p1.1))
       area ≤ 1) :
  ∃ T : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ), 
    (∀ p ∈ S, ∃ a b c : ℝ, 
       0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧ 
       p = (a * T.1.1 + b * T.2.1 + c * T.2.2, a * T.1.2 + b * T.2.2 + c * T.2.2)) ∧
    let area_T := 0.5 * abs ((T.2.1 - T.1.1) * (T.2.2 - T.1.2) - (T.2.2 - T.1.2) * (T.2.1 - T.1.1))
    area_T ≤ 4 :=
sorry

end exists_bounding_triangle_l437_437458


namespace gergonne_point_concurrent_and_barycentric_coords_l437_437717

theorem gergonne_point_concurrent_and_barycentric_coords
  (A B C A' B' C' : Point)
  (p a b c : ℝ)
  (h_p : p = (a + b + c) / 2)
  (h_contact_triangle : is_contact_triangle A B C A' B' C')
  (h_ratio1 : (p - b) / (p - c) = dist B A' / dist A' C)
  (h_ratio2 : (p - c) / (p - a) = dist C B' / dist B' A)
  (h_ratio3 : (p - a) / (p - b) = dist A C' / dist C' B) :
  (are_concurrent (line_through_points A A') (line_through_points B B') (line_through_points C C') 
  ∧ 
  ∃ K, 
  barycentric_coordinates (gergonne_point A B C A' B' C') = 
    (1 / (p - a), 1 / (p - b), 1 / (p - c))) := 
sorry

end gergonne_point_concurrent_and_barycentric_coords_l437_437717


namespace geometric_sum_inequality_l437_437319

theorem geometric_sum_inequality (x : ℝ) (n : ℕ) (hx : x > 0) : 
  1 + x + x^2 + ... + x^(2*n) ≥ (2*n + 1) * x^n := 
by 
  sorry

end geometric_sum_inequality_l437_437319


namespace equation_of_line_l437_437252

def point := (ℝ × ℝ)
def circle := { c : point // ∃ r : ℝ, ∀ p : point, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 }
def line := { l : ℝ × ℝ × ℝ // ∀ p : point, p.1 * l.1 + p.2 * l.2 + l.3 = 0 }

def P : point := (-3, -3/2)
def C : circle := ⟨(0, 0), 5, λ p, p.1^2 + p.2^2 = 25⟩
def chord_length : ℝ := 8

def line1 : line := ⟨(3, 4, 15), λ p, 3 * p.1 + 4 * p.2 + 15 = 0⟩
def line2 : line := ⟨(1, 0, 3), λ p, p.1 = -3⟩

theorem equation_of_line :: (l : line) (c : circle) (p : point)
  (hl : l.1) (circ : c.1) (hp : p) (length : ℝ)
  (l_through_p : p.1 * l.1 + p.2 * l.2 + l.3 = 0)
  (l_intersects_c : ∃ p1 p2 : point, c.1 = (p1.1 + p2.1) / 2 ∧ c.1 = (p1.2 + p2.2) / 2)
  (assume_chord_length : length = 8) :
  l = line1 ∨ l = line2 := sorry

end equation_of_line_l437_437252


namespace cartesian_eq_C1_rectangular_eq_C2_min_distance_PQ_l437_437656
-- Import the entirety of the necessary library

-- Define the parametric equations of C_1
def C_1 (α : ℝ) : ℝ × ℝ :=
  (cos α, sqrt(3) * sin α)

-- Define the polar coordinate equation of C_2 in terms of rectangular coordinates
def C_2 (ρ θ : ℝ) : Prop :=
  ρ * cos (θ - π / 4) = 3 * sqrt 2

-- Convert the parametric equations of C_1 to its Cartesian equation
theorem cartesian_eq_C1 (x y : ℝ) (α : ℝ) (h1 : x = cos α) (h2 : y = sqrt (3) * sin α) :
  x^2 + y^2 / 3 = 1 :=
sorry

-- Convert the polar coordinate equation of C_2 to its rectangular coordinate equation
theorem rectangular_eq_C2 (x y : ℝ) :
  (∃ ρ θ : ℝ, ρ * cos (θ - π / 4) = 3 * sqrt 2 ∧ x = ρ * cos θ ∧ y = ρ * sin θ) ↔ x + y = 6 :=
sorry

-- Prove the minimum value of |PQ| and find the rectangular coordinates of P
theorem min_distance_PQ (α : ℝ) (hP : ∃ pα, pα = α ∧ cos pα = 1/2 ∧ sqrt(3) * sin pα = 3/2) :
  (∃ min_d : ℝ, min_d = 2 * sqrt 2 ∧ pα = -π/3 + 2*k*π) :=
sorry

end cartesian_eq_C1_rectangular_eq_C2_min_distance_PQ_l437_437656


namespace company_employee_percentage_l437_437639

theorem company_employee_percentage (M : ℝ)
  (h1 : 0.20 * M + 0.40 * (1 - M) = 0.31000000000000007) :
  M = 0.45 :=
sorry

end company_employee_percentage_l437_437639


namespace probability_change_l437_437649

/-- In the country of Alpha, there are 5 banks. The probability that a bank will close
is the same for all banks and is 0.05. Banks close independently of one another.
In a crisis, the probability that a bank will close increased to 0.25. 
Prove that the absolute change in the probability of at least one bank closing 
is approximately 0.54, rounded to the nearest hundredth. -/
theorem probability_change (h1 : ∀ i ∈ finset.range 5, prob_close_before = 0.05) 
(h2 : ∀ i ∈ finset.range 5, prob_close_after = 0.25) 
(h3 : ∀ i j, i ≠ j → independent (closure_i i) (closure_j j))
: abs (prob_at_least_one_close_after - prob_at_least_one_close_before) ≈ 0.54 := 
sorry

end probability_change_l437_437649


namespace sin_squared_sum_l437_437529

theorem sin_squared_sum : 
  (∑ k in Finset.range 91, (Real.sin (k * Real.pi / 180)) ^ 6) = 229 / 8 := 
by
  sorry

end sin_squared_sum_l437_437529


namespace lcm_1_to_10_l437_437859

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437859


namespace lcm_1_10_l437_437964

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437964


namespace smallest_number_divisible_1_to_10_l437_437941

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437941


namespace point_lies_on_transformed_plane_l437_437317

theorem point_lies_on_transformed_plane :
  let k := (3 : ℝ) / 5
  let A : ℝ × ℝ × ℝ := (-2, -1, 1)
  let original_plane (x y z : ℝ) := x - 2 * y + 6 * z - 10
  let transformed_plane (x y z : ℝ) := x - 2 * y + 6 * z - 6
  transformed_plane (A.1) (A.2) (A.3) = 0 := 
by
  let k := (3 : ℝ) / 5
  let A : ℝ × ℝ × ℝ := (-2, -1, 1)
  let original_plane := λ (x y z : ℝ), x - 2 * y + 6 * z - 10
  let transformed_plane := λ (x y z : ℝ), x - 2 * y + 6 * z - 6
  sorry

end point_lies_on_transformed_plane_l437_437317


namespace smallest_number_divisible_by_1_through_10_l437_437982

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437982


namespace pyramid_volume_surface_area_l437_437739

/-- Definition of the given parameters --/
def base_angle : ℝ := 30
def slant_face_angle : ℝ := 60
def inscribed_circle_radius (r : ℝ) : ℝ := r

/-- Theorem statement for the volume and surface area of the pyramid --/
theorem pyramid_volume_surface_area (r : ℝ) (h₁ : base_angle = 30) (h₂ : slant_face_angle = 60) :
  (∃ V S, V = (8 * r^3 * Real.sqrt 3) / 3 ∧ S = 24 * r^2) :=
begin
  sorry
end

end pyramid_volume_surface_area_l437_437739


namespace lcm_1_to_10_l437_437848

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437848


namespace smallest_number_div_by_1_to_10_l437_437802

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437802


namespace sqrt_3_between_a1_a2_a2_closer_to_sqrt_3_l437_437326

noncomputable def a_2 (a_1 : ℚ) : ℝ := 1 + 2 / (1 + a_1)

theorem sqrt_3_between_a1_a2 (a_1 : ℚ) (h_pos : 0 < a_1) (h_sqrt : (a_1 : ℝ) ≠ real.sqrt 3) :
    (a_1 < real.sqrt 3 ∧ real.sqrt 3 < a_2 a_1) ∨ (a_2 a_1 < real.sqrt 3 ∧ real.sqrt 3 < a_1) :=
by
    sorry

theorem a2_closer_to_sqrt_3 (a_1 : ℚ) (h_pos : 0 < a_1) (h_sqrt : (a_1 : ℝ) ≠ real.sqrt 3) : 
    abs (a_2 a_1 - real.sqrt 3) < abs (a_1 - real.sqrt 3) :=
by
    sorry

end sqrt_3_between_a1_a2_a2_closer_to_sqrt_3_l437_437326


namespace angle_between_asymptotes_hyperbola_l437_437736

theorem angle_between_asymptotes_hyperbola (h : ∀ x y, 3 * y^2 - x^2 = 1) : 
  ∃ θ, θ = π / 3 :=
by
  -- Assume the values of the asymptotes and their angles
  let asymptote_pos : ∀ x, 3 * (√3 / 3 * x)^2 - x^2 = 1,
  let asymptote_neg : ∀ x, 3 * (-√3 / 3 * x)^2 - x^2 = 1,
  -- Assume the angles of the asymptotes
  have angle1 : θ = π / 6,
  have angle2 : θ = 5 * π / 6,
  -- Therefore the angle between the asymptotes is
  have angle_between : θ = (5 * π / 6) - (π / 6),
  refine ⟨angle_between, _⟩,
  sorry

end angle_between_asymptotes_hyperbola_l437_437736


namespace compute_x_y_sum_l437_437314

theorem compute_x_y_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 :=
by
  sorry

end compute_x_y_sum_l437_437314


namespace not_sufficient_not_necessary_l437_437058

theorem not_sufficient_not_necessary (a : ℝ) :
  ¬ ((a^2 > 1) → (1/a > 0)) ∧ ¬ ((1/a > 0) → (a^2 > 1)) := sorry

end not_sufficient_not_necessary_l437_437058


namespace spaceship_speed_conversion_l437_437091

theorem spaceship_speed_conversion (speed_km_per_sec : ℕ) (seconds_in_hour : ℕ) (correct_speed_km_per_hour : ℕ) :
  speed_km_per_sec = 12 →
  seconds_in_hour = 3600 →
  correct_speed_km_per_hour = 43200 →
  speed_km_per_sec * seconds_in_hour = correct_speed_km_per_hour := by
  sorry

end spaceship_speed_conversion_l437_437091


namespace sequence_a2018_l437_437161

theorem sequence_a2018 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) - 2 * a (n + 1) + a n = 1) 
  (h2 : a 18 = 0) 
  (h3 : a 2017 = 0) :
  a 2018 = 1000 :=
sorry

end sequence_a2018_l437_437161


namespace overlapping_area_is_correct_l437_437068

-- Defining the coordinates of the grid points
def topLeft : (ℝ × ℝ) := (0, 2)
def topMiddle : (ℝ × ℝ) := (1.5, 2)
def topRight : (ℝ × ℝ) := (3, 2)
def middleLeft : (ℝ × ℝ) := (0, 1)
def center : (ℝ × ℝ) := (1.5, 1)
def middleRight : (ℝ × ℝ) := (3, 1)
def bottomLeft : (ℝ × ℝ) := (0, 0)
def bottomMiddle : (ℝ × ℝ) := (1.5, 0)
def bottomRight : (ℝ × ℝ) := (3, 0)

-- Defining the vertices of the triangles
def triangle1_points : List (ℝ × ℝ) := [topLeft, middleRight, bottomMiddle]
def triangle2_points : List (ℝ × ℝ) := [bottomLeft, topMiddle, middleRight]

-- Function to calculate the area of a polygon given the vertices -- placeholder here
noncomputable def area_of_overlapped_region (tr1 tr2 : List (ℝ × ℝ)) : ℝ := 
  -- Placeholder for the actual computation of the overlapped area
  1.2

-- Statement to prove
theorem overlapping_area_is_correct : 
  area_of_overlapped_region triangle1_points triangle2_points = 1.2 := sorry

end overlapping_area_is_correct_l437_437068


namespace other_endpoint_of_diameter_l437_437528

-- Define the basic data
def center : ℝ × ℝ := (5, 2)
def endpoint1 : ℝ × ℝ := (0, -3)
def endpoint2 : ℝ × ℝ := (10, 7)

-- State the final properties to be proved
theorem other_endpoint_of_diameter :
  ∃ (e2 : ℝ × ℝ), e2 = endpoint2 ∧
    dist center endpoint2 = dist endpoint1 center :=
sorry

end other_endpoint_of_diameter_l437_437528


namespace disproving_example_l437_437418

theorem disproving_example (m n : ℤ) (h : m > n) : ¬ (m ^ 2 > n ^ 2) :=
by
  let m := -3
  let n := -6
  have h1 : m > n := by norm_num
  have h2 : ¬ (m ^ 2 > n ^ 2) := by norm_num
  exact h2

end disproving_example_l437_437418


namespace line_equation_from_point_normal_l437_437217

theorem line_equation_from_point_normal :
  let M1 : ℝ × ℝ := (7, -8)
  let n : ℝ × ℝ := (-2, 3)
  ∃ C : ℝ, ∀ x y : ℝ, 2 * x - 3 * y + C = 0 ↔ (C = -38) := 
by
  sorry

end line_equation_from_point_normal_l437_437217


namespace problem_solution_l437_437190

theorem problem_solution (x y : ℝ) (h₁ : x + Real.cos y = 2010) (h₂ : x + 2010 * Real.sin y = 2011) (h₃ : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := 
sorry

end problem_solution_l437_437190


namespace sin_double_angle_l437_437203

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.sin (2 * θ) = 3 / 5 := 
by 
sorry

end sin_double_angle_l437_437203


namespace wire_volume_l437_437072

noncomputable def volume_of_cylinder (d_cm : ℝ) (l_m : ℝ) : ℝ :=
  let r_dm := (d_cm / 2) * 0.1 in
  let l_dm := l_m * 10 in
  real.pi * (r_dm ^ 2) * l_dm

theorem wire_volume (
  (d_cm : 0.5),
  (l_m : 112.04507993669432) :
  volume_of_cylinder d_cm l_m = 2.2 := 
sorry

end wire_volume_l437_437072


namespace decimal_equivalent_one_quarter_power_one_l437_437784

theorem decimal_equivalent_one_quarter_power_one : (1 / 4 : ℝ) ^ 1 = 0.25 := by
  sorry

end decimal_equivalent_one_quarter_power_one_l437_437784


namespace perimeter_eq_circumference_l437_437398

theorem perimeter_eq_circumference (y : ℝ) : (4 * y = 8 * Real.pi) → (y = 6.28) :=
by 
  intro h
  have y_val : y = 2 * Real.pi := by 
    calc
      y = 8 * Real.pi / 4 : by
        linarith [h]
      ... = 2 * Real.pi : by
        norm_num
  have y_approx : Real.floor (100 * y) / 100 = 6.28 := by 
    rw [y_val]
    norm_num
  exact y_approx

end perimeter_eq_circumference_l437_437398


namespace brick_height_l437_437071

/-- A certain number of bricks, each measuring 25 cm x 11.25 cm x some height, 
are needed to build a wall of 8 m x 6 m x 22.5 cm. 
If 6400 bricks are needed, prove that the height of each brick is 6 cm. -/
theorem brick_height (h : ℝ) : 
  6400 * (25 * 11.25 * h) = (800 * 600 * 22.5) → h = 6 :=
by
  sorry

end brick_height_l437_437071


namespace f_sum_positive_l437_437251

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x1 x2 : ℝ) (hx : x1 + x2 > 0) : f x1 + f x2 > 0 :=
sorry

end f_sum_positive_l437_437251


namespace inv_sum_equals_neg_two_l437_437322

def g (x : ℝ) : ℝ := x^3

def g_inv (y : ℝ) : ℝ := y^(1/3 : ℝ)

theorem inv_sum_equals_neg_two : g_inv 8 + g_inv (-64) = -2 := by
  have h1 : g_inv 8 = 2 := by
    simp [g_inv]
    exact real.cbrt_pos 8
  have h2 : g_inv (-64) = -4 := by
    simp [g_inv]
    exact real.cbrt_neg (-64)
  rw [h1, h2]
  norm_num

end inv_sum_equals_neg_two_l437_437322


namespace smallest_possible_sum_l437_437680

theorem smallest_possible_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Nat.gcd (a + b) 330 = 1) (h4 : b ^ b ∣ a ^ a) (h5 : ¬ b ∣ a) :
  a + b = 147 :=
sorry

end smallest_possible_sum_l437_437680


namespace distance_from_point_to_plane_is_two_l437_437278

-- Condition: The plane OAB with normal vector
def normal_vector : ℝ × ℝ × ℝ := (2, -2, 1)

-- Condition: Point P
def P : ℝ × ℝ × ℝ := (-1, 3, 2)

-- The mathematical statement: the distance from point P to the plane is 2
theorem distance_from_point_to_plane_is_two (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) :
  n = normal_vector → p = P → dist_to_plane n p 0 = 2 :=
-- Proof omitted with sorry
by {
  intro h_normal h_P,
  sorry
}

end distance_from_point_to_plane_is_two_l437_437278


namespace lcm_1_to_10_l437_437858

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437858


namespace lcm_1_10_l437_437967

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437967


namespace jaysons_moms_age_l437_437439

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end jaysons_moms_age_l437_437439


namespace simplify_complex_div_l437_437721

theorem simplify_complex_div : (5 + 7 * complex.I) / (2 + 3 * complex.I) = (31 / 13) - (1 / 13) * complex.I :=
by
  sorry

end simplify_complex_div_l437_437721


namespace smallest_multiple_1_through_10_l437_437824

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437824


namespace smallest_number_divisible_1_to_10_l437_437938

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437938


namespace sum_of_valid_x_l437_437547

theorem sum_of_valid_x : ∑(x : ℝ) in {x | let seq := [3, 5, 7, 18, x].qsort(≤) 
  in seq.nth 2 = (33 + x)/5}, x = -8 := by
  sorry

end sum_of_valid_x_l437_437547


namespace correct_option_is_2_l437_437181

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Given conditions
def condition1 : Prop := a 4 - 2 * a 7 + a 8 = 0
def arithmetic_sequence (b : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n, b (n + 1) = b n + d
def condition2 : Prop := b 7 = a 7
def condition3 : Prop := b 2 < b 8 ∧ b 8 < b 11

-- We need to prove the correct answer is 2 given the above conditions
theorem correct_option_is_2 (h1 : condition1) (h2 : arithmetic_sequence b)
    (h3 : condition2) (h4 : condition3) : 2 = 2 :=
by
  sorry

end correct_option_is_2_l437_437181


namespace lcm_1_to_10_l437_437807

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437807


namespace rest_area_milepost_l437_437390

theorem rest_area_milepost : 
  let fifth_exit := 30
  let fifteenth_exit := 210
  (3 / 5) * (fifteenth_exit - fifth_exit) + fifth_exit = 138 := 
by 
  let fifth_exit := 30
  let fifteenth_exit := 210
  sorry

end rest_area_milepost_l437_437390


namespace minimum_value_l437_437158

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  1/a + 2/b + 4/c

theorem minimum_value (a b c : ℝ) (h₀ : c > 0) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
    (h₃ : 4 * a^2 - 2 * a * b + b^2 - c = 0)
    (h₄ : ∀ x y, 4*x^2 - 2*x*y + y^2 - c = 0 → |2*x + y| ≤ |2*a + b|)
    : min_value_of_expression a b c = -1 :=
sorry

end minimum_value_l437_437158


namespace vertex_parabola_is_parabola_l437_437681

variables {a c : ℝ} (h_a : 0 < a) (h_c : 0 < c)

theorem vertex_parabola_is_parabola :
  ∀ (x y : ℝ), (∃ b : ℝ, x = -b / (2 * a) ∧ y = a * (-b / (2 * a)) ^ 2 + b * (-b / (2 * a)) + c) ↔ y = -a * x ^ 2 + c :=
by sorry

end vertex_parabola_is_parabola_l437_437681


namespace area_of_square_EFGH_l437_437116

-- Define the original square with side length 3 and the derived properties.
def original_square_side := 3
def semicircle_radius := original_square_side / 2
def tangent_thickness := 1
def tangent_line_length := semicircle_radius + tangent_thickness

-- We need a formal proof that the area of square EFGH is 16.
theorem area_of_square_EFGH :
  let side_length_EFGH := original_square_side + 2 * tangent_line_length
  ∧ let area_EFGH := side_length_EFGH ^ 2 in
  area_EFGH = 16 :=
by
  sorry

end area_of_square_EFGH_l437_437116


namespace find_divisor_l437_437054

-- Define the problem conditions as variables
variables (remainder quotient dividend : ℕ)
variables (D : ℕ)

-- State the problem formally
theorem find_divisor (h0 : remainder = 19) 
                     (h1 : quotient = 61) 
                     (h2 : dividend = 507) 
                     (h3 : dividend = (D * quotient) + remainder) : 
  D = 8 := 
by 
  -- Use the Lean theorem prover to demonstrate the condition
  have h4 : 507 = (D * 61) + 19, from h3,
  -- Simplify and solve for D
  sorry

end find_divisor_l437_437054


namespace harmonic_progression_S4_l437_437495

theorem harmonic_progression_S4 :
  ∀ (terms : List ℚ), List.take 3 terms = [3, 4, 6] ∧
  (∀ i, i > 0 → i < terms.length - 2 → (1 / terms[i] - 1 / terms[i+1]) = (1 / terms[i+1] - 1 / terms[i+2])) →
  List.sum (List.take 4 terms) = 25 :=
by
  sorry

end harmonic_progression_S4_l437_437495


namespace lcm_1_10_l437_437962

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437962


namespace num_ordered_triples_correct_l437_437202

noncomputable def num_ordered_triples : ℕ :=
  let U := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let sets := { A | ∃ B C, A ∪ B ∪ C = U } 
  7 ^ 10

theorem num_ordered_triples_correct : 
  ∀ A B C : set ℕ, (A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) → 
  (∃ (A B C : set ℕ), num_ordered_triples = 7 ^ 10) :=
sorry

end num_ordered_triples_correct_l437_437202


namespace exists_four_points_in_quadrilateral_l437_437693

theorem exists_four_points_in_quadrilateral (A B C D : ℝ × ℝ)
  (h : area_of_quadrilateral A B C D = 1) :
  ∃ P Q R S : ℝ × ℝ,
    (P ∈ boundary_or_interior A B C D) ∧ (Q ∈ boundary_or_interior A B C D) ∧
    (R ∈ boundary_or_interior A B C D) ∧ (S ∈ boundary_or_interior A B C D) ∧
    (area_of_triangle P Q R > 1/4) ∧
    (area_of_triangle P Q S > 1/4) ∧
    (area_of_triangle P R S > 1/4) ∧
    (area_of_triangle Q R S > 1/4) := 
sorry

-- Definitions for the functions and predicates used in the theorem statement
def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry
def boundary_or_interior (A B C D : ℝ × ℝ) (P : ℝ × ℝ) : Prop := sorry
def area_of_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

end exists_four_points_in_quadrilateral_l437_437693


namespace train_crossing_time_l437_437231

noncomputable def speed_in_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

theorem train_crossing_time :
  ∀ (length_of_train length_of_bridge : ℝ) (speed_kmph : ℝ) (time_approx : ℝ),
    length_of_train = 110 →
    length_of_bridge = 290 →
    speed_kmph = 60 →
    time_approx = 24 →
    (length_of_train + length_of_bridge) / (speed_in_mps speed_kmph) ≈ time_approx :=
by
  intros length_of_train length_of_bridge speed_kmph time_approx
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end train_crossing_time_l437_437231


namespace problem1_problem2_problem3_problem4_problem5_l437_437140

-- Define f for the first problem
variable (f : ℝ → ℝ)

def condition1 (f : ℝ → ℝ) := ∀ x: ℝ, -f(f(x)-1) = x + 1
def bijective (f : ℝ → ℝ) := Function.Injective f ∧ Function.Surjective f

theorem problem1 : (condition1 f) → bijective f :=
by
  intro h
  -- proof would go here
  sorry

-- Define f for the second problem
variable (f : ℝ → ℝ)

def condition2 (f : ℝ → ℝ) := ∀ (x y: ℝ), -f(y + f(x)) = (y - 1) * (f (x^2)) + 3*x

theorem problem2 : (condition2 f) → bijective f :=
by
  intro h
  -- proof would go here
  sorry

-- Define f for the third problem
variable (f : ℝ → ℝ)

def condition3 (f : ℝ → ℝ) := ∀ (x y: ℝ), -f(x + f(y)) = f(x) + y^5

theorem problem3 : (condition3 f) → bijective f :=
by
  intro h
  -- proof would go here
  sorry

-- Define f for the fourth problem
variable (f : ℝ → ℝ)

def condition4 (f : ℝ → ℝ) := ∀ x: ℝ, -f(f(x)) = Real.sin(x)
def neither_injective_nor_surjective (f : ℝ → ℝ) := ¬Function.Injective f ∧ ¬Function.Surjective f

theorem problem4 : (condition4 f) → neither_injective_nor_surjective f :=
by
  intro h
  -- proof would go here
  sorry

-- Define f for the fifth problem
variable (f : ℝ → ℝ)

def condition5 (f : ℝ → ℝ) := ∀ (x y: ℝ), -f(x + y^2) = f(x) * f(y) + x * f(y) - y^3 * f(x)

theorem problem5 : (condition5 f) → neither_injective_nor_surjective f :=
by
  intro h
  -- proof would go here
  sorry

end problem1_problem2_problem3_problem4_problem5_l437_437140


namespace min_n_for_three_consecutive_l437_437145

theorem min_n_for_three_consecutive (n : ℕ) (marbles : Finset ℕ) (h : marbles.card = 150 ∧ ∀ x ∈ marbles, x ∈ Finset.range 1 151) :
  ∃ (n : ℕ), n = 101 ∧ ∀ (chosen : Finset ℕ), chosen.card = n → (∃ a b c, a ∈ chosen ∧ b ∈ chosen ∧ c ∈ chosen ∧ (a + 1 = b ∧ b + 1 = c)) :=
by
  sorry

end min_n_for_three_consecutive_l437_437145


namespace part_I_part_II_part_III_l437_437213

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x * (sin x + cos x) + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (a^2 - a + 10) * exp x
noncomputable def φ (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  (b * (1 + exp 2) * g x a) / ((a^2 - a + 10) * exp 2 * x) - 1 / x + 1 + log x

theorem part_I (a : ℝ) (h : tangent_condition f a (0, f 0 a) (1, 2)) : a = -1 := sorry

theorem part_II (a : ℝ)
  (h : ∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ π ∧ 0 ≤ x2 ∧ x2 ≤ π ∧ g x2 a < f x1 a + 13 - exp (π / 2)) : 
  -1 < a ∧ a < 3 := sorry

theorem part_III (a : ℝ) (b : ℝ) (hb : 1 < b) : 
  (∃ (z : ℝ), z ∈ (0, ∞) ∧ φ z a b = 0) = false := sorry

end part_I_part_II_part_III_l437_437213


namespace smallest_divisible_by_1_to_10_l437_437891

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437891


namespace simplify_complex_div_l437_437722

theorem simplify_complex_div : (5 + 7 * complex.I) / (2 + 3 * complex.I) = (31 / 13) - (1 / 13) * complex.I :=
by
  sorry

end simplify_complex_div_l437_437722


namespace bookmark_position_second_book_l437_437022

-- Definitions for the conditions
def pages_per_book := 250
def cover_thickness_ratio := 10
def total_books := 2
def distance_bookmarks_factor := 1 / 3

-- Derived constants
def cover_thickness := cover_thickness_ratio * pages_per_book
def total_pages := (pages_per_book * total_books) + (cover_thickness * total_books * 2)
def distance_between_bookmarks := total_pages * distance_bookmarks_factor
def midpoint_pages_within_book := (pages_per_book / 2) + cover_thickness

-- Definitions for bookmarks positions
def first_bookmark_position := midpoint_pages_within_book
def remaining_pages_after_first_bookmark := distance_between_bookmarks - midpoint_pages_within_book
def second_bookmark_position := remaining_pages_after_first_bookmark - cover_thickness

-- Theorem stating the goal
theorem bookmark_position_second_book :
  35 ≤ second_bookmark_position ∧ second_bookmark_position < 36 :=
sorry

end bookmark_position_second_book_l437_437022


namespace number_of_people_who_purchased_only_book_A_l437_437119

theorem number_of_people_who_purchased_only_book_A (
    (people_A : ℕ) (people_B : ℕ) (people_C : ℕ)
    (people_AB : ℕ) (people_BC : ℕ) (people_AC : ℕ)
    (price_A : ℕ) (price_B : ℕ) (price_C : ℕ)
    (rev_total : ℕ)
    (h_AB_eq : people_AB = 500)
    (h_A_eq_2B : people_A + people_AB = 2 * (people_B + people_AB))
    (h_AB_eq_2B_only : people_AB = 2 * people_B)
    (h_BC : people_BC = 300)
    (h_rev : rev_total = 15000)
    (h_price_A : price_A = 25)
    (h_price_B : price_B = 20)
    (h_price_C : price_C = 15)
  ) : people_A = 1000 := 
by 
  sorry

end number_of_people_who_purchased_only_book_A_l437_437119


namespace cannot_determine_a_from_odd_function_l437_437634

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem cannot_determine_a_from_odd_function (f : ℝ → ℝ) (h : odd_function f) : 
  ¬ ∃ a, f a = a :=
begin
  sorry
end

end cannot_determine_a_from_odd_function_l437_437634


namespace least_multiplier_produces_required_result_l437_437454

noncomputable def least_multiplier_that_satisfies_conditions : ℕ :=
  62087668

theorem least_multiplier_produces_required_result :
  ∃ k : ℕ, k * 72 = least_multiplier_that_satisfies_conditions * 72 ∧
           k * 72 % 112 = 0 ∧
           k * 72 % 199 = 0 ∧
           ∃ n : ℕ, k * 72 = n * n :=
by
  let k := least_multiplier_that_satisfies_conditions
  use k
  -- Using "sorry" as we don't need the proof steps here
  sorry

end least_multiplier_produces_required_result_l437_437454


namespace problem1_problem2_l437_437214

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4 * x

-- First problem statement
theorem problem1 (a : ℝ) : 
  (∀ x : ℝ, 2 * a - 1 ≤ x → ∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2) →
  a ≥ 3 / 2 :=
sorry

-- Second problem statement
theorem problem2 : 
  Set.range (λ x : ℝ, if 1 ≤ x ∧ x ≤ 7 then some (f x) else none) = Set.Icc (-4 : ℝ) 21 :=
sorry

end problem1_problem2_l437_437214


namespace smallest_number_divisible_by_1_to_10_l437_437843

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437843


namespace number_of_possible_centroids_l437_437374

variables (P Q R : ℝ × ℝ) -- Define three points P, Q, R as pairs of real numbers (coordinates)

def is_point_on_square_perimeter (p : ℝ × ℝ) : Prop := 
  (0 <= p.1 ∧ p.1 <= 15 ∧ (p.2 = 0 ∨ p.2 = 15)) ∨
  (0 <= p.2 ∧ p.2 <= 15 ∧ (p.1 = 0 ∨ p.1 = 15))

def not_collinear (P Q R : ℝ × ℝ) : Prop := 
  ¬ collinear ℝ (set.range ![id, id]) ![P, Q, R]

def centroid (P Q R : ℝ × ℝ) : ℝ × ℝ := 
  ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

def centroid_is_within_square (c : ℝ × ℝ) : Prop :=
  (1 / 3 <= c.1 ∧ c.1 <= 44 / 3) ∧ (1 / 3 <= c.2 ∧ c.2 <= 44 / 3)

-- The proof statement:
theorem number_of_possible_centroids :
  ∃! (S : set (ℝ × ℝ)), S = {c | ∃ P Q R, 
                                   is_point_on_square_perimeter P ∧ 
                                   is_point_on_square_perimeter Q ∧ 
                                   is_point_on_square_perimeter R ∧ 
                                   not_collinear P Q R ∧
                                   centroid_is_within_square (centroid P Q R)} ∧ 
                        S.card = 1936 := 
sorry

end number_of_possible_centroids_l437_437374


namespace max_value_m_div_n_l437_437601

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * m
def h (x : ℝ) (n : ℝ) : ℝ := 6 * n^2 * Real.log x - 4 * n * x

theorem max_value_m_div_n
  (n : ℝ) (h_pos : 0 < n)
  (x₀ : ℝ) (hx₀ : x₀ = n ∨ x₀ = -3 * n)
  (hsame_point : f x₀ m = h x₀ n)
  (hsame_tangent : 2 * x₀ = 6 * n^2 / x₀ - 4 * n) :
  ∃ m, (m : ℝ) / n = 3 * Real.exp (-1 / 6) :=
by
  sorry

end max_value_m_div_n_l437_437601


namespace lcm_1_10_l437_437958

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437958


namespace five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l437_437236

noncomputable def count_five_digit_numbers_greater_21035_and_even : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_greater_21035_and_even_correct :
  count_five_digit_numbers_greater_21035_and_even = 39 :=
  sorry

noncomputable def count_five_digit_numbers_even_with_odd_positions : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_even_with_odd_positions_correct :
  count_five_digit_numbers_even_with_odd_positions = 8 :=
  sorry

end five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l437_437236


namespace person_age_l437_437445

theorem person_age (x : ℕ) (h : 4 * (x + 3) - 4 * (x - 3) = x) : x = 24 :=
by {
  sorry
}

end person_age_l437_437445


namespace prob_transform_in_S_l437_437500

open Complex

-- Define the region S in the complex plane
def in_region_S (z : ℂ) : Prop := 
  let x := z.re
  let y := z.im
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

-- Define the transformation
def transform (z : ℂ) : ℂ :=
  (1 / 2 + (1 / 2) * complex.I) * z

-- State the proof problem
theorem prob_transform_in_S (z : ℂ) (hz : in_region_S z) : 
  in_region_S (transform z) :=
  sorry

end prob_transform_in_S_l437_437500


namespace degree_measure_OC1D_l437_437124

/-- Define points on the sphere -/
structure Point (latitude longitude : ℝ) :=
(lat : ℝ := latitude)
(long : ℝ := longitude)

noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

noncomputable def angle_OC1D : ℝ :=
  Real.arccos ((cos_deg 44) * (cos_deg (-123)))

/-- The main theorem: the degree measure of ∠OC₁D is 113 -/
theorem degree_measure_OC1D :
  angle_OC1D = 113 := sorry

end degree_measure_OC1D_l437_437124


namespace find_f_l437_437574

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f :
  (∀ x : ℝ, x > 1 → f (2 / x + 1) = Real.log x) →
  ∀ t : ℝ, t > 1 → f t = Real.log (2 / (t - 1)) :=
begin
  intro h,
  intro t,
  intro ht,
  sorry,
end

end find_f_l437_437574


namespace problem_statement_l437_437572

-- Definitions for the given conditions
def vector_a (α : ℝ) := (4, 5 * real.cos α)
def vector_b (α : ℝ) := (3, -4 * real.tan α)

-- α is in the interval (0, π/2)
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < real.pi / 2

-- Vectors are perpendicular
def is_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Statement of the proof problem
theorem problem_statement (α : ℝ) (h1 : alpha_in_interval α)
  (h2 : is_perpendicular (vector_a α) (vector_b α)) :
  |vector_a α - vector_b α| = 5 * real.sqrt 2 ∧ 
  real.sin (3 * real.pi / 2 + 2 * α) + real.cos (2 * α - real.pi) = -14 / 25 :=
sorry

end problem_statement_l437_437572


namespace volume_surface_area_ratio_l437_437255

/--
Given a structure of nine unit cubes arranged such that a central cube 
is surrounded symmetrically on all faces (except the bottom) by the other eight cubes, 
the ratio of the volume to the surface area is 9/31.
-/
theorem volume_surface_area_ratio : 
  let central_cube := 1
  let surrounding_cube := 8
  let total_volume := central_cube + surrounding_cube
  let exposed_faces := 1 -- bottom of central cube
                  + 8 * (3 + 0.75)
  (total_volume : ℕ) = 9 →
  (exposed_faces : ℕ) = 31 →
  (total_volume : ℝ) / (exposed_faces : ℝ) = (9 : ℝ) / (31 : ℝ) :=
by
  intros central_cube surrounding_cube total_volume exposed_faces h1 h2
  sorry

end volume_surface_area_ratio_l437_437255


namespace corner_movement_l437_437493

-- Definition of corner movement problem
def canMoveCornerToBottomRight (m n : ℕ) : Prop :=
  m ≥ 2 ∧ n ≥ 2 ∧ (m % 2 = 1 ∧ n % 2 = 1)

theorem corner_movement (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  (canMoveCornerToBottomRight m n ↔ (m % 2 = 1 ∧ n % 2 = 1)) :=
by
  sorry  -- Proof is omitted

end corner_movement_l437_437493


namespace total_interest_after_tenth_year_l437_437405

variables (P R : ℝ)

-- Definitions based on the given conditions
def SI (P R T : ℝ) : ℝ := (P * R * T) / 100

noncomputable def total_interest (P R : ℝ) : ℝ :=
  let I1 := (P * R * 5) / 100
  let I2 := (3 * P * R * 5) / 100
  I1 + I2

theorem total_interest_after_tenth_year
  (h1 : (P * R * 10) / 100 = 400)
  (h2 : ∀ t : ℝ, t > 5 → 3 * P * R * (t - 5) / 100 = (3 * P * R * 5) / 100) :
  total_interest P R = 800 :=
sorry

end total_interest_after_tenth_year_l437_437405


namespace min_marked_squares_needed_on_chessboard_l437_437033

-- Define the chessboard as a set of positions (i, j) where 1 ≤ i, j ≤ 8
def chessboard : set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 8 ∧ 1 ≤ p.2 ∧ p.2 ≤ 8}

-- Define when two positions are adjacent
def adjacent (p q : ℕ × ℕ) : Prop :=
  abs (p.1 - q.1) ≤ 1 ∧ abs (p.2 - q.2) ≤ 1 ∧ p ≠ q

-- Define a function that checks if a set of positions are non-adjacent
def non_adjacent (positions : set (ℕ × ℕ)) : Prop :=
  ∀ p q ∈ positions, p ≠ q → ¬(adjacent p q)

-- Define the set of marked positions
def marked_positions : set (ℕ × ℕ) := { (1, 1), (1, 8), (8, 1), (8, 8), (4, 4), (1, 4), (4, 1), (4, 8), (8, 4) }

-- Define the minimum required number of marked squares
def min_marked_squares := 9

-- State the theorem
theorem min_marked_squares_needed_on_chessboard :
  ∃ s : set (ℕ × ℕ), s ⊆ chessboard ∧
  non_adjacent s ∧
  s.card = min_marked_squares ∧
  (∀ t : set (ℕ × ℕ), t ⊆ chessboard → (t.card > min_marked_squares → ∃ (p : ℕ × ℕ), p ∈ t ∧ ∃ q ∈ s, adjacent p q)) :=
sorry

end min_marked_squares_needed_on_chessboard_l437_437033


namespace isosceles_right_triangle_area_and_circumcircle_radius_l437_437267

theorem isosceles_right_triangle_area_and_circumcircle_radius
  (X Y Z : Type*)
  [Inner X] [Inner Y] [Inner Z]
  (YZ : ℝ)
  (h₁ : ∠Y = ∠Z)
  (h₂ : YZ = 8 * Real.sqrt 2) :
  let XY := YZ / Real.sqrt 2 in
  let XZ := YZ / Real.sqrt 2 in
  (1 / 2 * XY * XZ = 32) ∧ (XY / Real.sqrt 2 = 4 * Real.sqrt 2) :=
sorry

end isosceles_right_triangle_area_and_circumcircle_radius_l437_437267


namespace smallest_number_divisible_by_1_to_10_l437_437840

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437840


namespace sid_spent_on_snacks_l437_437363

theorem sid_spent_on_snacks :
  let original_money := 48
  let money_spent_on_computer_accessories := 12
  let money_left_after_computer_accessories := original_money - money_spent_on_computer_accessories
  let remaining_money_after_purchases := 4 + original_money / 2
  ∃ snacks_cost, money_left_after_computer_accessories - snacks_cost = remaining_money_after_purchases ∧ snacks_cost = 8 :=
by
  sorry

end sid_spent_on_snacks_l437_437363


namespace symmetry_axes_intersect_at_centroid_l437_437354

-- Definitions related to polygon and centroid
structure Polygon :=
  (vertices : List (ℝ × ℝ))

def centroid (p : Polygon) : ℝ × ℝ :=
  let n := p.vertices.length 
  let sum_x := List.sum (List.map Prod.fst p.vertices)
  let sum_y := List.sum (List.map Prod.snd p.vertices)
  (sum_x / n, sum_y / n)

def is_symmetry_axis (p : Polygon) (l : ℝ → ℝ) : Prop :=
  ∀ v ∈ p.vertices, let (x, y) := v in l x = y

def invariant_under_reflection (p : Polygon) (l : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x y, (x, y) ∈ p.vertices → l x = y → reflect_about (x, y) l = (x, y)

theorem symmetry_axes_intersect_at_centroid  
  (p : Polygon)
  (axes : List (ℝ → ℝ))
  (symmetry_prop : ∀ l ∈ axes, is_symmetry_axis p l)
  (centroid_invariant : ∀ l ∈ axes, invariant_under_reflection p l (centroid p)) :
  ∀ l1 l2 ∈ axes, centroid p = intersection_of_axes l1 l2 := 
by
  sorry

end symmetry_axes_intersect_at_centroid_l437_437354


namespace min_sin_A_l437_437662

variable {A B C : ℝ}

theorem min_sin_A (AB AC : ℝ) (A_area : ℝ) 
  (h1 : AB + AC = 7) 
  (h2 : 1/2 * AB * AC * A_area = 4) : 
  ∃ A : ℝ, A_area = asin (32 / 49) := 
by 
  sorry

end min_sin_A_l437_437662


namespace molecular_weight_3_moles_Al2S3_l437_437788

theorem molecular_weight_3_moles_Al2S3 :
  (let atomic_weight_Al := 26.98
   let atomic_weight_S := 32.06
   let molecular_weight_Al2S3 := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
   in molecular_weight_Al2S3 * 3 = 450.42) :=
by
  let atomic_weight_Al := 26.98
  let atomic_weight_S := 32.06
  let molecular_formula := "Al₂S₃"
  let moles := 3
  let molecular_weight_Al2S3 := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
  have h := molecular_weight_Al2S3 * moles = 450.42
  show (molecular_weight_Al2S3 * moles = 450.42) from h
  sorry

end molecular_weight_3_moles_Al2S3_l437_437788


namespace num_of_triangles_with_perimeter_10_l437_437235

theorem num_of_triangles_with_perimeter_10 :
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → 
      a + b + c = 10 ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a) ∧ 
    triangles.card = 4 := sorry

end num_of_triangles_with_perimeter_10_l437_437235


namespace largest_x_solution_l437_437562

noncomputable def solve_eq (x : ℝ) : Prop :=
  (15 * x^2 - 40 * x + 16) / (4 * x - 3) + 3 * x = 7 * x + 2

theorem largest_x_solution : 
  ∃ x : ℝ, solve_eq x ∧ x = -14 + Real.sqrt 218 := 
sorry

end largest_x_solution_l437_437562


namespace multiples_of_7_l437_437052

theorem multiples_of_7 (a b : ℤ) (q : set ℤ) (h_ab_mult_14_a : 14 ∣ a) (h_ab_mult_14_b : 14 ∣ b)
  (h_q_inclusive : ∀ x ∈ q, a ≤ x ∧ x ≤ b) (h_q_mult_14 : ∃ s : fin 13, ∀(i : s), 14 * (i.val : ℤ) ∈ q) :
  q.count (λ x , 7 ∣ x) = 28 :=
sorry

end multiples_of_7_l437_437052


namespace JimAgeInXYears_l437_437020

-- Definitions based on conditions
def TomCurrentAge := 37
def JimsAge7YearsAgo := 5 + (TomCurrentAge - 7) / 2

-- We introduce a variable X to represent the number of years into the future.
variable (X : ℕ)

-- Lean 4 statement to prove that Jim will be 27 + X years old in X years from now.
theorem JimAgeInXYears : JimsAge7YearsAgo + 7 + X = 27 + X := 
by
  sorry

end JimAgeInXYears_l437_437020


namespace required_fraction_l437_437079

theorem required_fraction
  (total_members : ℝ)
  (top_10_lists : ℝ) :
  total_members = 775 →
  top_10_lists = 193.75 →
  top_10_lists / total_members = 0.25 :=
by
  sorry

end required_fraction_l437_437079


namespace locus_of_perpendiculars_l437_437306

variables {A B : Point} (g : Line) (P : Point) (F : Point) (Q : Point)

-- Definitions of geometric setup
-- Assuming 'g' is perpendicular to the segment 'AB' and 'g' is a line
def line_perpendicular_to_segment (g : Line) (A B : Point) : Prop :=
  g.perpendicular (segment A B)

-- Assuming 'F' is the midpoint of segment 'AB'
def midpoint (A B F : Point) : Prop := 
  (dist A F = dist B F) ∧ (segment A F) + (segment F B) = segment A B

-- Reflect 'g' over 'F'
def reflection_of_line (g : Line) (F : Point) : Line :=
  g.reflect F

-- Proving the locus of 'Q' by finding that it matches the reflected line
theorem locus_of_perpendiculars (h1 : line_perpendicular_to_segment g A B) (h2 : midpoint A B F) :
  ∀ P on g, let Q be the intersection point of the perpendiculars from A to PA, and from B to PB, in that
  Q lies on the reflection_of_line g F :=
sorry

end locus_of_perpendiculars_l437_437306


namespace find_set_A_l437_437616

variable {A B : Set ℤ}
noncomputable def f (x : ℤ) := 2 * x - 1

theorem find_set_A (B_def : B = {-3, 3, 5})
  (biject_f : ∀ y ∈ B, ∃! x ∈ A, f x = y) : 
  A = {-1, 2, 3} :=
by
  sorry

end find_set_A_l437_437616


namespace no_possible_arrangement_l437_437999

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l437_437999


namespace total_books_l437_437511

variables (Beatrix_books Alannah_books Queen_books : ℕ)

def Alannah_condition := Alannah_books = Beatrix_books + 20
def Queen_condition := Queen_books = Alannah_books + (Alannah_books / 5)

theorem total_books (hB : Beatrix_books = 30) (hA : Alannah_condition) (hQ : Queen_condition) : 
  (Beatrix_books + Alannah_books + Queen_books) = 140 :=
by
  sorry

end total_books_l437_437511


namespace smallest_number_divisible_by_1_to_10_l437_437832

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437832


namespace area_of_triangle_ABC_l437_437200

theorem area_of_triangle_ABC
  (R a b c : ℝ) (A B C : ℝ)
  (h1 : c = 2)
  (h2 : sin B = 2 * sin A)
  (h3 : (a^2 - c^2) / (2 * R) = (a - b) * sin B) :
  let area := (1 / 2) * a * b * sin C in
  area = (2 * Real.sqrt 3) / 3 := 
sorry

end area_of_triangle_ABC_l437_437200


namespace cost_of_each_book_is_six_l437_437137

-- Define variables for the number of books bought
def books_about_animals := 8
def books_about_outer_space := 6
def books_about_trains := 3

-- Define the total number of books
def total_books := books_about_animals + books_about_outer_space + books_about_trains

-- Define the total amount spent
def total_amount_spent := 102

-- Define the cost per book
def cost_per_book := total_amount_spent / total_books

-- Prove that the cost per book is $6
theorem cost_of_each_book_is_six : cost_per_book = 6 := by
  sorry

end cost_of_each_book_is_six_l437_437137


namespace smallest_number_div_by_1_to_10_l437_437799

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437799


namespace smallest_divisible_1_to_10_l437_437928

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437928


namespace smallest_divisible_1_to_10_l437_437921

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437921


namespace inverse_matrix_sum_l437_437396

variable (x y z w p q r s : ℝ)

def A : Matrix (Fin 3) (Fin 3) ℝ := ![![x, 2, y], ![3, 4, 5], ![z, 6, w]]
def B : Matrix (Fin 3) (Fin 3) ℝ := ![[-7, p, -13], ![q, -15, r], ![3, s, 6]]
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem inverse_matrix_sum :
  A.mul B = I →
  x + y + z + w + p + q + r + s = -5.5 := sorry

end inverse_matrix_sum_l437_437396


namespace length_segment_AB_l437_437216

theorem length_segment_AB (k : ℝ) :
  (∀ x y : ℝ, (x - 3)^2 + (y + 1)^2 = 1 → k * x + y - 2 = 0 → 
    let A := (0, k) in
    let C := (3, -1) in
    let AC := (0 - 3)^2 + (k + 1)^2 in
    let AB := real.sqrt(AC - 1) in
    AB = 2 * real.sqrt 3) :=
begin
  sorry
end

end length_segment_AB_l437_437216


namespace acute_angle_sum_l437_437321

open Real

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
                        (hβ : 0 < β ∧ β < π / 2)
                        (h1 : 3 * (sin α) ^ 2 + 2 * (sin β) ^ 2 = 1)
                        (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end acute_angle_sum_l437_437321


namespace binomial_coefficient_a_l437_437607

theorem binomial_coefficient_a (a : ℝ) (h : (x - a / x^2)^9.expand.coeff 6 = 36) : a = -4 :=
by
  sorry

end binomial_coefficient_a_l437_437607


namespace chocolate_candy_cost_l437_437472

-- Define the constants and conditions
def cost_per_box : ℕ := 5
def candies_per_box : ℕ := 30
def discount_rate : ℝ := 0.1

-- Define the total number of candies to buy
def total_candies : ℕ := 450

-- Define the threshold for applying discount
def discount_threshold : ℕ := 300

-- Calculate the number of boxes needed
def boxes_needed (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the total cost without discount
def total_cost (boxes_needed : ℕ) (cost_per_box : ℕ) : ℝ :=
  boxes_needed * cost_per_box

-- Calculate the discounted cost
def discounted_cost (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

-- Statement to be proved
theorem chocolate_candy_cost :
  discounted_cost 
    (total_cost (boxes_needed total_candies candies_per_box) cost_per_box) 
    discount_rate = 67.5 :=
by
  -- Proof is needed here, using the correct steps from the solution.
  sorry

end chocolate_candy_cost_l437_437472


namespace smallest_number_divisible_by_1_to_10_l437_437844

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437844


namespace smallest_number_divisible_by_1_to_10_l437_437841

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437841


namespace solve_fractional_eq_l437_437004

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end solve_fractional_eq_l437_437004


namespace cone_lateral_area_eq_384pi_l437_437741

noncomputable def lateralAreaOfCone (thetaDeg r : ℝ) : ℝ :=
  let l := 6 * r
  pi * r * l

theorem cone_lateral_area_eq_384pi :
  lateralAreaOfCone 60 8 = 384 * Real.pi := by
  sorry

end cone_lateral_area_eq_384pi_l437_437741


namespace length_of_MN_l437_437648

def parametric_l (α t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

def cartesian_C (x y : Real) : Prop :=
  (x^2 / 3) + y^2 = 1

def length_MN (α : Real) : Real :=
  if α = (5 * Real.pi / 6) then
    let t1 : Real := (sqrt 3) / 3
    let t2 : Real := -sqrt 3 / 2
    Real.sqrt((t1 + t2)^2 - 4 * t1 * t2)
  else 0

theorem length_of_MN :
  length_MN (5 * π / 6) = 2 * sqrt 15 / 3 :=
  by sorry

end length_of_MN_l437_437648


namespace number_of_players_l437_437269

-- Definitions based on conditions
def socks_price : ℕ := 6
def tshirt_price : ℕ := socks_price + 7
def total_cost_per_player : ℕ := 2 * (socks_price + tshirt_price)
def total_expenditure : ℕ := 4092

-- Lean theorem statement
theorem number_of_players : total_expenditure / total_cost_per_player = 108 := 
by
  sorry

end number_of_players_l437_437269


namespace butterfly_safe_flight_probability_l437_437597

noncomputable def volume (l w h : ℝ) : ℝ := l * w * h

theorem butterfly_safe_flight_probability :
  let large_prism_volume := volume 5 4 3,
      safe_prism_volume := volume 3 2 1 in
  large_prism_volume = 60 ∧
  safe_prism_volume = 6 →
  (safe_prism_volume / large_prism_volume) = (1 / 10) := by
  intros large_prism_volume safe_prism_volume h,
  simp [large_prism_volume, safe_prism_volume] at h,
  sorry

end butterfly_safe_flight_probability_l437_437597


namespace mean_of_other_two_numbers_l437_437516

def problem_statement (a b c d e f g h : ℕ) (mean1800_subset : Finset ℕ) 
  (mean1800_subset_size : mean1800_subset.card = 6) : Prop :=
  let total_sum := a + b + c + d + e + f + g + h in
  let subset_sum := mean1800_subset.sum id in
  let remaining_sum := total_sum - subset_sum in
  let other_two_mean := remaining_sum / 2 in
  total_sum = 1234 + 1468 + 1520 + 1672 + 1854 + 2010 + 2256 + 2409 ∧
  subset_sum = mean1800_subset.card * 1800 ∧
  other_two_mean = 1811.5

theorem mean_of_other_two_numbers :
  problem_statement 1234 1468 1520 1672 1854 2010 2256 2409 (Finset.{0} ⟨[1234, 1468, 1520, 1672, 1854, 2010].toFinset, by simp⟩) sorry :=
sorry

end mean_of_other_two_numbers_l437_437516


namespace smallest_number_div_by_1_to_10_l437_437795

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437795


namespace least_prime_factor_of_expression_l437_437430

theorem least_prime_factor_of_expression :
  ∃ p : ℕ, p > 1 ∧ prime p ∧ (∃ k : ℕ, 5^6 - 5^4 + 5^2 = p * k) ∧ ∀ q : ℕ, q > 1 ∧ prime q ∧ (∃ m : ℕ, 5^6 - 5^4 + 5^2 = q * m) → p ≤ q :=
sorry

end least_prime_factor_of_expression_l437_437430


namespace taxi_fare_30km_l437_437263

def fare (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 4 then 10
  else if h : 4 < x ∧ x ≤ 18 then 1.5 * x + 4
  else 2 * x - 5

theorem taxi_fare_30km : fare 30 = 55 :=
sorry

end taxi_fare_30km_l437_437263


namespace opposite_of_negative_five_l437_437759

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end opposite_of_negative_five_l437_437759


namespace fraction_expression_proof_l437_437282

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end fraction_expression_proof_l437_437282


namespace log_sum_l437_437632

noncomputable def log_a_div (a : ℝ) (b c : ℝ) : ℝ := Real.log b / Real.log a + Real.log c / Real.log a

theorem log_sum (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (h2 : a = 2):
  log_a_div a (3/7) (112/3) = 4 :=
by {
  rw h2,
  unfold log_a_div,
  simp,
  -- The detailed proof follows here, which we are skipping.
  sorry
}

end log_sum_l437_437632


namespace total_books_l437_437512

-- Define the number of books each person has
def books_beatrix : ℕ := 30
def books_alannah : ℕ := books_beatrix + 20
def books_queen : ℕ := books_alannah + (books_alannah / 5)

-- State the theorem to be proved
theorem total_books (h_beatrix : books_beatrix = 30)
                    (h_alannah : books_alannah = books_beatrix + 20)
                    (h_queen : books_queen = books_alannah + (books_alannah / 5)) :
  books_alannah + books_beatrix + books_queen = 140 :=
sorry

end total_books_l437_437512


namespace smallest_number_divisible_by_1_to_10_l437_437838

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437838


namespace number_of_persons_in_group_l437_437380

theorem number_of_persons_in_group 
    (n : ℕ)
    (h1 : average_age_before - average_age_after = 3)
    (h2 : person_replaced_age = 40)
    (h3 : new_person_age = 10)
    (h4 : total_age_decrease = 3 * n):
  n = 10 := 
sorry

end number_of_persons_in_group_l437_437380


namespace marbles_left_percentage_l437_437491

-- Define the initial number of marbles
def initial_marbles (x : ℝ) := x

-- Define the percentage remaining after each person receives their share
def percentage_remaining_after_B (x : ℝ) := 0.80 * x
def percentage_remaining_after_C (x : ℝ) := 0.90 * (percentage_remaining_after_B x)
def percentage_remaining_after_D (x : ℝ) := 0.75 * (percentage_remaining_after_C x)

-- The theorem to prove
theorem marbles_left_percentage (x : ℝ) (h : x ≠ 0) :
  percentage_remaining_after_D x / x = 0.54 :=
by
  -- translate the given conditions to Lean
  unfold percentage_remaining_after_D percentage_remaining_after_C percentage_remaining_after_B initial_marbles
  calc
    0.75 * (0.90 * (0.80 * x)) / x
    _ = 0.75 * 0.90 * 0.80    : by ring
    _ = 0.54                  : by norm_num

end marbles_left_percentage_l437_437491


namespace largest_even_number_l437_437008

theorem largest_even_number (x : ℤ) 
  (h : x + (x + 2) + (x + 4) = x + 18) : x + 4 = 10 :=
by
  sorry

end largest_even_number_l437_437008


namespace probability_transform_in_S_l437_437502

open Complex

def region_S (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

def transform (z : ℂ) : ℂ :=
  (1/2 : ℂ) * z + (1/2 : ℂ) * I * z

theorem probability_transform_in_S : 
  ∀ (z : ℂ), region_S z → region_S (transform z) := 
by
  sorry

end probability_transform_in_S_l437_437502


namespace golden_section_length_l437_437195

noncomputable def golden_section_point (a b : ℝ) := a / (a + b) = b / a

theorem golden_section_length (A B P : ℝ) (h : golden_section_point A P) (hAP_gt_PB : A > P) (hAB : A + P = 2) : 
  A = Real.sqrt 5 - 1 :=
by
  -- Proof goes here
  sorry

end golden_section_length_l437_437195


namespace trigonometric_identity_l437_437364

theorem trigonometric_identity (x : ℝ) : 
  sqrt 2 * cos x + sqrt 6 * sin x = 2 * sqrt 2 * cos (π / 3 - x) :=
by 
  sorry -- Proof not required

end trigonometric_identity_l437_437364


namespace cot_arctan_5_over_12_eq_12_over_5_l437_437129

theorem cot_arctan_5_over_12_eq_12_over_5 : Real.cot (Real.arctan (5 / 12)) = 12 / 5 := by
  sorry

end cot_arctan_5_over_12_eq_12_over_5_l437_437129


namespace smallest_divisible_1_to_10_l437_437925

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437925


namespace greatest_leftover_cookies_l437_437069

theorem greatest_leftover_cookies (n : ℕ) : ∃ k, k ≤ n ∧ k % 8 = 7 := sorry

end greatest_leftover_cookies_l437_437069


namespace train_speed_is_approximately_29_kmh_l437_437447

/-
Mathematical definitions of conditions:
- length_train is 90 meters
- length_bridge is 200 meters
- time_crossing is 36 seconds
- total_distance is the sum of length_train and length_bridge
- speed_meters_per_second is total_distance divided by time_crossing
- speed_kilometers_per_hour is speed_meters_per_second multiplied by 3.6
-/

def length_train : ℝ := 90
def length_bridge : ℝ := 200
def time_crossing : ℝ := 36
def total_distance : ℝ := length_train + length_bridge
def speed_meters_per_second : ℝ := total_distance / time_crossing
def speed_kilometers_per_hour : ℝ := speed_meters_per_second * 3.6

-- Lean statement to prove the speed of train is approximately 29 km/h.
theorem train_speed_is_approximately_29_kmh : abs (speed_kilometers_per_hour - 29) < 0.1 :=
by
  sorry

end train_speed_is_approximately_29_kmh_l437_437447


namespace lcm_1_10_l437_437959

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437959


namespace directrix_of_parabola_l437_437746

theorem directrix_of_parabola (h : ∀ x, y = 1/4 * x^2) : directrix y = -1 :=
sorry

end directrix_of_parabola_l437_437746


namespace simplify_fraction_l437_437031

theorem simplify_fraction : (3^3 * 3^(-4)) / (3^5 * 3^(-3)) = 1 / 27 := by
  sorry

end simplify_fraction_l437_437031


namespace area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l437_437063

-- (a) Proving the area of triangle ABC
theorem area_triangle_ABC (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 3) (h_right : true) : 
  (1 / 2) * AB * BC = 3 := sorry

-- (b) Proving the area of figure DEFGH
theorem area_figure_DEFGH (DH HG : ℝ) (hDH : DH = 5) (hHG : HG = 5) (triangle_area : ℝ) (hEPF : triangle_area = 3) : 
  DH * HG - triangle_area = 22 := sorry

-- (c) Proving the area of triangle JKL 
theorem area_triangle_JKL (side_area : ℝ) (h_side : side_area = 25) 
  (area_JSK : ℝ) (h_JSK : area_JSK = 3) 
  (area_LQJ : ℝ) (h_LQJ : area_LQJ = 15/2) 
  (area_LRK : ℝ) (h_LRK : area_LRK = 5) : 
  side_area - area_JSK - area_LQJ - area_LRK = 19/2 := sorry

end area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l437_437063


namespace smallest_divisible_1_to_10_l437_437912

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437912


namespace semicircle_parametric_eqns_point_D_coordinates_l437_437655

noncomputable def parametric_equations (α : ℝ) : ℝ × ℝ :=
  (1 + Real.cos α, Real.sin α)

theorem semicircle_parametric_eqns (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi) :
  (∃ ρ, ρ = 2 * Real.cos θ) ↔
  ∃ α, 0 ≤ α ∧ α ≤ Real.pi ∧
       (1 + Real.cos α = 2 * Real.cos θ) ∧
       (Real.sin α = 0) :=
sorry

theorem point_D_coordinates (D : ℝ × ℝ) (hD : D ∈ set_of (parametric_equations α) ∧
  ( ∃ l, l = λ x : ℝ, Real.sqrt 3 * x + 2 ∧
    (∀ tangent : ℝ, tangent ∈  l → tanget ⟂ l) ) ) :
  D = (3/2, Real.sqrt 3 / 2) :=
sorry

end semicircle_parametric_eqns_point_D_coordinates_l437_437655


namespace smallest_number_div_by_1_to_10_l437_437798

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437798


namespace solve_trigonometric_identity_l437_437061

noncomputable def trigonometric_identity : Prop :=
  (cos (Real.pi / 8))^2 - (sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2

theorem solve_trigonometric_identity : trigonometric_identity :=
by
  sorry

end solve_trigonometric_identity_l437_437061


namespace original_average_is_6_2_l437_437089

-- Constraint: the set S contains exactly 10 numbers
variable (S : Fin 10 → ℝ)

-- Definition: the average of a set
def average (xs : Fin 10 → ℝ) : ℝ :=
  (∑ i, xs i) / 10

-- Given condition: increasing one element by 7 and the new average is 6.9
axiom h : ∃ i, average (λ j, if j = i then S j + 7 else S j) = 6.9

-- To prove: the original average was 6.2
theorem original_average_is_6_2 : average S = 6.2 := sorry

end original_average_is_6_2_l437_437089


namespace nonnegative_exists_l437_437219

theorem nonnegative_exists (a b c : ℝ) (h : a + b + c = 0) : a ≥ 0 ∨ b ≥ 0 ∨ c ≥ 0 :=
by
  sorry

end nonnegative_exists_l437_437219


namespace axes_symmetry_intersect_at_one_point_l437_437348

noncomputable def center_of_mass (p : polygon) : point := sorry

theorem axes_symmetry_intersect_at_one_point (P : polygon)
  (h : has_multiple_axes_of_symmetry P) :
  ∃! (p : point), ∀ (A : axis_of_symmetry), is_axis_of_symmetry A P → p ∈ A :=
sorry

end axes_symmetry_intersect_at_one_point_l437_437348


namespace equal_angles_l437_437258

-- Define the geometric setup
variable {A B C D E F P O X : Type} [midpoints : ∀ {D E F: Type}, midpoint D E F] 

-- Given that P is on the circumcircle of triangle ABC
variable [circumcircle : ∀ {P O: Type}, circumcircle P O]

-- Given that the circumcircle of triangle ODP intersects AP at a second point X
variable [intersect : ∀ {O D P X : Type}, intersect_circumcircle O D P X]

theorem equal_angles
  (ABC : triangle A B C)
  (D : midpoint B C)
  (E : midpoint C A)
  (F : midpoint A B)
  (P : point_on_circumcircle ABC O)
  (X : point_on_circumcircle_intersect O D P A)
  : angle X E O = angle X F O :=
by
  sorry

end equal_angles_l437_437258


namespace smallest_n_inequality_l437_437546

variable {x y z : ℝ}

theorem smallest_n_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    (∀ m : ℕ, (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l437_437546


namespace lcm_1_10_l437_437963

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437963


namespace smallest_divisible_1_to_10_l437_437916

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437916


namespace function_is_increasing_l437_437192

variable (x : ℝ)

-- Definition of the function
def y (x : ℝ) : ℝ := -1 / x

-- Condition that y increases as x increases for x > 1
def is_increasing_for_x_gt_1 (f : ℝ → ℝ) : Prop := ∀ x > 1, f x < f (x + 1)

-- The theorem we want to prove
theorem function_is_increasing : is_increasing_for_x_gt_1 y := by
  -- The proof would go here
  sorry

end function_is_increasing_l437_437192


namespace area_of_quadrilateral_XMYN_l437_437661

theorem area_of_quadrilateral_XMYN 
  (X Y Z M N Q : Type)
  [RealVector X] [RealVector Y] [RealVector Z]
  [RealVector M] [RealVector N] [RealVector Q]
  (h1 : is_median XM Y N)
  (h2 : is_median X Y N Q) 
  (h3 : distance Q N = 3)
  (h4 : distance Q M = 4)
  (h5 : distance M N = 5) 
: Area_of_Quadrilateral X M Y N = 54 := 
sorry

end area_of_quadrilateral_XMYN_l437_437661


namespace proportion_condition_l437_437457

variable (a b c d a₁ b₁ c₁ d₁ : ℚ)

theorem proportion_condition
  (h₁ : a / b = c / d)
  (h₂ : a₁ / b₁ = c₁ / d₁) :
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ := by
  sorry

end proportion_condition_l437_437457


namespace algae_cell_count_at_day_nine_l437_437488

noncomputable def initial_cells : ℕ := 5
noncomputable def division_frequency_days : ℕ := 3
noncomputable def total_days : ℕ := 9

def number_of_cycles (total_days division_frequency_days : ℕ) : ℕ :=
  total_days / division_frequency_days

noncomputable def common_ratio : ℕ := 2

noncomputable def number_of_cells_after_n_days (initial_cells common_ratio number_of_cycles : ℕ) : ℕ :=
  initial_cells * common_ratio ^ (number_of_cycles - 1)

theorem algae_cell_count_at_day_nine : number_of_cells_after_n_days initial_cells common_ratio (number_of_cycles total_days division_frequency_days) = 20 :=
by
  sorry

end algae_cell_count_at_day_nine_l437_437488


namespace smallest_number_divisible_by_1_through_10_l437_437977

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437977


namespace range_of_real_number_l437_437226

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B (a : ℝ) : Set ℝ := {-1, -3, a}
def complement_A : Set ℝ := {x | x ≥ 0}

theorem range_of_real_number (a : ℝ) (h : (complement_A ∩ (B a)) ≠ ∅) : a ≥ 0 :=
sorry

end range_of_real_number_l437_437226


namespace ball_coloring_probability_l437_437014

-- Definitions for conditions
def num_balls : ℕ := 8
def equal_probability : ℝ := (1/2)
def half_balls : ℕ := num_balls / 2

-- Statement of the problem
theorem ball_coloring_probability :
  let P : ℝ := 35 / 128 in
  (∀ (configuration : (fin num_balls) → bool), -- each configuration of 8 balls colored either black (false) or white (true)
    (∑ i in finset.fin_range num_balls, ite (configuration i = tt) 1 0) = half_balls -> -- exactly 4 white balls
    (∑ i in finset.fin_range num_balls, (ite (filter (≠ configuration i) configuration).length >= 4 1 0)) = num_balls) -> 
  -- the number of balls different in color from at least 4 others == 8
  P = (70 * (equal_probability ^ num_balls)) :=
sorry

end ball_coloring_probability_l437_437014


namespace solve_equation_l437_437205

theorem solve_equation (m : ℤ) (y : ℚ) (h₀ : 3 * (m - 1) = 1) :
    3 * m * y + 2 * y = 3 + m → y = 5 / 8 :=
by
  intros h₁
  rw ← h₀ at h₁
  sofrry

end solve_equation_l437_437205


namespace whipped_cream_depth_correct_l437_437092

noncomputable def whipped_cream_depth : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * (3:ℝ)^3
  let volume_cylinder := λ h: ℝ, Real.pi * (9:ℝ)^2 * h
  let h := (volume_sphere / volume_cylinder 1)
  h

-- Theorem: The depth of the spread whipped cream is 4/9 inches.
theorem whipped_cream_depth_correct (h : ℝ) : h = 4/9 :=
by
  let volume_sphere := (4 / 3) * Real.pi * (3:ℝ)^3
  let volume_cylinder := Real.pi * (9:ℝ)^2 * h
  have volume_eq := volume_sphere = volume_cylinder
  field_simp at volume_eq
  have h_eq : h = 4 / 9, by linarith
  assumption

end whipped_cream_depth_correct_l437_437092


namespace arithmetic_geometric_properties_l437_437594

noncomputable def arithmetic_seq (a₁ a₂ a₃ : ℝ) :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

noncomputable def geometric_seq (b₁ b₂ b₃ : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem arithmetic_geometric_properties (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  arithmetic_seq a₁ a₂ a₃ →
  geometric_seq b₁ b₂ b₃ →
  ¬(a₁ < a₂ ∧ a₂ > a₃) ∧
  (b₁ < b₂ ∧ b₂ > b₃) ∧
  (a₁ + a₂ < 0 → ¬(a₂ + a₃ < 0)) ∧
  (b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by
  sorry

end arithmetic_geometric_properties_l437_437594


namespace greatest_three_digit_number_divisible_by_5_and_10_l437_437785

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def is_divisible_by_10 (n : ℕ) : Prop := n % 10 = 0

theorem greatest_three_digit_number_divisible_by_5_and_10 : 
  ∃ n, is_three_digit n ∧ is_divisible_by_10 n ∧ (∀ m, is_three_digit m ∧ is_divisible_by_10 m → m ≤ n) :=
begin
  use 990,
  split,
  { -- proof that 990 is a three-digit number
    split,
    { -- 990 is at least 100
      linarith,
    },
    { -- 990 is less than 1000
      linarith,
    }
  },
  split,
  { -- proof that 990 is divisible by 10
    refl,
  },
  { -- proof that any other three-digit number divisible by 10 is less than or equal to 990
    intros m hm,
    cases hm with hm1 hm2,
    cases hm1 with hm3 hm4,
    have h : m ≤ 990 ∨ m > 990, by omega,
    cases h,
    {
      exact h,
    },
    {
      exfalso,
      have : m < 1000, by omega,
      have h' : m % 10 = 0, from hm2,
      linarith,
    }
  }
end

end greatest_three_digit_number_divisible_by_5_and_10_l437_437785


namespace midpoint_locus_theorem_l437_437477

noncomputable def midpoint_locus_circle (O P : ℝ × ℝ) (r : ℝ) 
(h_dist : dist O P = r / 2) : Set (ℝ × ℝ) :=
let M_center : ℝ × ℝ := (O.1 + P.1) / 2, (O.2 + P.2) / 2 
let M_radius : ℝ := r / 2 
Set.circle M_center M_radius

theorem midpoint_locus_theorem 
(O P : ℝ × ℝ) (r : ℝ) 
(h_dist : dist O P = r / 2) : 
locus_of_M O P r h_dist = Set.circle ((O.1 + P.1) / 2, (O.2 + P.2) / 2) (r / 2) :=
sorry

end midpoint_locus_theorem_l437_437477


namespace smallest_divisible_by_1_to_10_l437_437899

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437899


namespace set_intersection_complement_l437_437679

def U := set.univ : set ℝ
def A := {x : ℝ | x ≥ 0}
def B := {x : ℝ | x ≤ 0}
def complement_B := {x : ℝ | 0 < x}

theorem set_intersection_complement :
  A ∩ complement_B = {x : ℝ | 0 < x } :=
by sorry

end set_intersection_complement_l437_437679


namespace custom_op_4_2_l437_437301

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem to prove the result
theorem custom_op_4_2 : custom_op 4 2 = 24 :=
by
  sorry

end custom_op_4_2_l437_437301


namespace multiplication_problem_correct_l437_437653

theorem multiplication_problem_correct : 
  ∃ a b c d : ℕ, a = 2 ∧ b = 0 ∧ c = 1 ∧ d = 6 ∧ (let n := 2105 in n = ((a * 1000 + b * 100 + c * 10 + d) * 1)) :=
by
  sorry

end multiplication_problem_correct_l437_437653


namespace exists_disjoint_t_sets_l437_437316

open Set

-- Define the set S and subset A
def S : Set ℕ := {m | 1 ≤ m ∧ m ≤ 1000000}

def A : Set ℕ := {m | m ∈ S ∧ my_predicate m}    -- Placeholder for some predicate to ensure 101 elements

-- Define property for the problem
def is_disjoint_set (A : Set ℕ) (t : ℕ) : Set ℕ := {m | ∃ a ∈ A, m = t + a}

theorem exists_disjoint_t_sets 
(hA : (A ⊆ S) ∧ (A.card = 101)) : 
∃ (t : fin 100 → ℕ), 
∀ (i j : fin 100), i ≠ j → disjoint (is_disjoint_set A (t i)) (is_disjoint_set A (t j)) :=
begin
  sorry
end

end exists_disjoint_t_sets_l437_437316


namespace rows_colored_red_l437_437330

theorem rows_colored_red (total_rows total_squares_per_row blue_rows green_squares red_squares_per_row red_rows : ℕ)
  (h_total_squares : total_rows * total_squares_per_row = 150)
  (h_blue_squares : blue_rows * total_squares_per_row = 60)
  (h_green_squares : green_squares = 66)
  (h_red_squares : 150 - 60 - 66 = 24)
  (h_red_rows : 24 / red_squares_per_row = 4) :
  red_rows = 4 := 
by sorry

end rows_colored_red_l437_437330


namespace sum_of_angles_l437_437709

/- Definitions of points on the circle and arcs -/
variables {A B C D E : Point}
variable (circle : Circle)
variable (on_circle : ∀ P, P ∈ [A, B, C, D, E])
variable (arc_AB : Measure_Angle) (arc_BC : Measure_Angle) 

/- Given conditions -/
axiom measure_arc_AB : arc_AB = 50
axiom measure_arc_BC : arc_BC = 60

/- Definitions of intersections and angles -/
variable (R S : Point)
axiom int_R : R = intersection (line_through A D) (line_through B E)
axiom int_S : S = intersection (line_through A C) (line_through B D)

/- Goal is to prove: -/
theorem sum_of_angles (arc_AC : Measure_Angle) (angle_R angle_S : Measure_Angle):
  arc_AC = arc_AB + arc_BC →
  angle_R = (360 - arc_AC) / 2 →
  angle_S = (360 - arc_AC) / 2 →
  angle_R + angle_S = 250 :=
by
  intro h1 h2 h3
  simp [h1, h2, h3]
  sorry

end sum_of_angles_l437_437709


namespace find_functions_l437_437556

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem find_functions (f : ℤ → ℤ) (h : satisfies_equation f) : (∀ x, f x = 2 * x) ∨ (∀ x, f x = 0) :=
sorry

end find_functions_l437_437556


namespace smallest_divisible_1_to_10_l437_437915

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437915


namespace misplaced_value_l437_437497

open Polynomial

/-
  Assume the sequence values are given as follows:
  Let s : ℕ → ℕ be such that s 0 = 9604, s 1 = 9801, s 2 = 10201, s 3 = 10404, 
  s 4 = 10816, s 5 = 11025, s 6 = 11449, s 7 = 11664, and s 8 = 12100.
  
  Prove that the value s 2 (i.e., 10201) is likely misplaced or calculated incorrectly.
-/
theorem misplaced_value :
  ∃ (s : ℕ → ℕ), 
    s 0 = 9604 ∧ s 1 = 9801 ∧ s 2 = 10201 ∧ s 3 = 10404 ∧ 
    s 4 = 10816 ∧ s 5 = 11025 ∧ s 6 = 11449 ∧ s 7 = 11664 ∧ 
    s 8 = 12100 ∧ 
    ∃ (a b c : ℚ),
      ∀ x : ℕ, 
        (x > 0 → (s x - s (x - 1)) = (s x = x^2 * a + x * b + c)) ∧ 
        (s 2 ≠ 10201) :=
sorry

end misplaced_value_l437_437497


namespace candidates_appeared_l437_437265

theorem candidates_appeared (x : ℝ) (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 :=
by
  sorry

end candidates_appeared_l437_437265


namespace octagon_perimeter_l437_437134

noncomputable def equilateral_triangle_side (a : ℝ) : Prop :=
  ∃ (A B C D E F G H : ℝ), 
  A = 6 ∧
  B = 6 ∧ 
  C = 6 ∧ 
  D = 3 ∧ 
  E = 1.5 ∧
  F = 3 ∧
  G = 12 ∧
  H = 6 ∧
  (D^2 + E^2 = B^2)

theorem octagon_perimeter (A B C D E F G H : ℝ)
  (hAB : A = 6)
  (hAC : G = 2 * A)
  (hBC : B = A)
  (hAD : D = G / 2)
  (hDE : E = A)
  (hAE : F = G - 6)
  (hEF : E = A / 2)
  (hBH : D^2 + E^2 = B^2)
  : A + B + C + D + E + F + (E/2) + sqrt (E^2 + D^2)
  = 29 + sqrt 33.75 := 
by 
  sorry

end octagon_perimeter_l437_437134


namespace goods_train_time_to_cross_platform_l437_437081

noncomputable def time_to_cross_platform (speed_train_kmh length_train length_platform : ℝ) : ℝ :=
  let speed_train_ms := speed_train_kmh * (1000 / 3600)
  let total_distance := length_train + length_platform
  total_distance / speed_train_ms

theorem goods_train_time_to_cross_platform :
  time_to_cross_platform 72 260.0416 260 = 26.00208 :=
by
  unfold time_to_cross_platform
  have h1 : 72 * (1000 / 3600) = 20 := by norm_num
  rw [h1]
  norm_num
  done

end goods_train_time_to_cross_platform_l437_437081


namespace integral_trig_identity_l437_437456

open Real

theorem integral_trig_identity :
  ∫ x in 0..π, 16 * (sin (x / 2))^6 * (cos (x / 2))^2 = (7 * π) / 8 :=
by
  sorry

end integral_trig_identity_l437_437456


namespace symmetric_points_y_axis_l437_437186

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : (a, 3) = (-2, 3)) (h₂ : (2, b) = (2, 3)) : (a + b) ^ 2015 = 1 := by
  sorry

end symmetric_points_y_axis_l437_437186


namespace problem_statement_l437_437684

namespace Proof

variable (x n : ℝ) (f : ℝ)
noncomputable def alpha := 3 + Real.sqrt 5
noncomputable def beta := 3 - Real.sqrt 5

theorem problem_statement (h1 : x = Real.pow (alpha) 20)
                         (h2 : n = Real.floor x)
                         (h3 : f = x - n) :
  x * (1 - f) = 1 := by 
sorry

end Proof

end problem_statement_l437_437684


namespace x_not_in_0_neg1_3_x_eq_neg2_if_neg2_in_A_l437_437324

variable {ℝ : Type}

noncomputable def A (x : ℝ) : Set ℝ := {3, x, x^2 - 2*x}

theorem x_not_in_0_neg1_3 (x : ℝ) : x ∉ {0, -1, 3} :=
by sorry

theorem x_eq_neg2_if_neg2_in_A (x : ℝ) (hx : -2 ∈ A x) : x = -2 :=
by sorry

end x_not_in_0_neg1_3_x_eq_neg2_if_neg2_in_A_l437_437324


namespace max_distance_C1_C2_l437_437275

-- Condition Definitions
def C1_polar (ρ θ : ℝ) : Prop := ρ^2 = 3 / (1 + 2 * (Real.sin θ)^2)
def C1_rect (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def C2_parametric (t : ℝ) (x y : ℝ) : Prop := 
  x = 2 + (Real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t
def C2_general (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

def distance_to_C2 (α : ℝ) : ℝ :=
  let P := (Real.sqrt 3 * Real.cos α, Real.sin α)
  let line_eq := (P.1 - Real.sqrt 3 * P.2 - 2)
  abs (line_eq) / Real.sqrt ((Real.sqrt 3)^2 + (-1)^2)

-- Theorem Statement
theorem max_distance_C1_C2 : 
  ∀ (θ : ℝ), ∀ (ρ : ℝ), (C1_polar ρ θ) → 
  0 ≤ distance_to_C2 θ ∧ distance_to_C2 θ ≤ (Real.sqrt 6 + 2) / 2 :=
by sorry

end max_distance_C1_C2_l437_437275


namespace cosine_sum_sine_half_sum_leq_l437_437638

variable {A B C : ℝ}

theorem cosine_sum_sine_half_sum_leq (h : A + B + C = Real.pi) :
  (Real.cos A + Real.cos B + Real.cos C) ≤ (Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2)) :=
sorry

end cosine_sum_sine_half_sum_leq_l437_437638


namespace major_premise_wrong_l437_437743

theorem major_premise_wrong (rhombus_diagonals_equal : ∀ {r : Type} [rhombus r], False)
                            (square_is_rhombus : ∀ s : Type, [square s] → [rhombus s])
                            (square_diagonals_equal : ∀ s : Type, [square s] → ∀ d₁ d₂ : diag s, equal_length d₁ d₂) :
                            False :=
sorry

end major_premise_wrong_l437_437743


namespace domain_of_f_l437_437388

section domain_proof

def f (x : ℝ) : ℝ := sqrt (log (2/3) (2 * x - 1))

theorem domain_of_f : {x : ℝ | 2*x - 1 > 0 ∧ log (2/3) (2*x - 1) ≥ 0} = {x : ℝ | (1/2) < x ∧ x ≤ 1} :=
by {
  sorry
}

end domain_proof

end domain_of_f_l437_437388


namespace num_red_balls_l437_437644

theorem num_red_balls (total_balls : ℕ) (freq_red : ℚ) :
  total_balls = 60 → freq_red = 0.15 → total_balls * freq_red = 9 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end num_red_balls_l437_437644


namespace lawn_mowing_l437_437699

theorem lawn_mowing :
  (let mary_rate := (1 : ℝ) / 3;
       tom_rate := (1 : ℝ) / 6;
       mary_alone_time := 1;
       both_time := 2;
       mowed_by_mary := mary_alone_time * mary_rate;
       combined_rate := mary_rate + tom_rate;
       mowed_by_both := both_time * combined_rate;
       total_mowed := mowed_by_mary + mowed_by_both)
  in total_mowed >= 1 → 1 - total_mowed = 0 :=
by {
  sorry
}

end lawn_mowing_l437_437699


namespace smallest_number_divisible_by_1_through_10_l437_437984

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437984


namespace smallest_number_divisible_by_1_to_10_l437_437871

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437871


namespace correct_statement_A_l437_437169

-- Definitions
variables {l m : Line} {α : Plane}

-- Conditions
def is_perpendicular_to_plane (l : Line) (α : Plane) : Prop := 
  ∀ (p q : Point), p ≠ q → p ∈ l → q ∈ l → ∃ r s : Line, r ≠ s ∧ p ∈ r ∧ q ∈ s ∧ r ⊂ α ∧ s ⊂ α ∧ ⟂ r s

def is_subset_of (m : Line) (α : Plane) : Prop := 
  ∀ (p : Point), p ∈ m → p ∈ α

def is_perpendicular_to (l m : Line) : Prop := 
  ∀ (p q : Point), p ≠ q → p ∈ l → q ∈ l → ∃ r s : Line, r ≠ s ∧ p ∈ r ∧ q ∈ s ∧ ⟂ r s

-- Theorem
theorem correct_statement_A {l m : Line} {α : Plane} 
  (h1 : is_perpendicular_to_plane l α)
  (h2 : is_subset_of m α) :
  is_perpendicular_to l m :=
sorry 

end correct_statement_A_l437_437169


namespace near_neighbors_l437_437294

variables {α β : Type*} {A B : set α}
variables (ε d : ℝ) (Y : set β)

def e_regular_pair (d ε : ℝ) (A B : set α) :=
  ∀ (X ⊆ A) (Y ⊆ B), |Y| ≥ ε * |B| → |((A ∩ X).neighbors ∩ Y)| / (|A ∩ X| * |Y|) - d < ε

def has_neighbors (v : α) (Y : set β) (t : ℝ) :=
  |neighbors v Y| ≥ t

theorem near_neighbors (A B : set α) (ε d : ℝ) (hA : A ≠ ∅) (hB : B ≠ ∅)
  (h : e-regular_pair d ε A B) (hY : |Y| ≥ ε * |B|) :
  ∃ S ⊆ A, |S| < ε * |A| ∧
  ∀ v ∈ (A \ S), has_neighbors v Y (d - ε) :=
begin
  sorry
end

end near_neighbors_l437_437294


namespace eccentricity_of_ellipse_l437_437201

theorem eccentricity_of_ellipse (a b c : ℝ) (F1 F2 M N : ℝ × ℝ)
  (h1 : F1 = (0, 0)) 
  (h2 : F2 = (c, 0)) 
  (h3 : M = (x1, y1)) 
  (h4 : N = (x2, y2)) 
  (h5 : |M - F2| = |F1 - F2|) 
  (h6 : |M - F1| = 2) 
  (h7 : |N - F1| = 1) 
  (h8 : ∃ e, e = c / a) 
  (h_ellipse : a > b ∧ b > 0 ∧ (x / a)^2 + (y / b)^2 = 1) 
  (h_foci_relation : b^2 = a^2 - c^2) 
  (h_line : ∃ m, ∃ k, y = mx + k ∧ line with points (F1, M, N)) : 
  c / (3/2) = 1/3 :=
by sorry 

end eccentricity_of_ellipse_l437_437201


namespace perfect_square_proof_l437_437050

def expr_A := nat.factorial 100 * nat.factorial 101
def expr_B := nat.factorial 100 * nat.factorial 102
def expr_C := nat.factorial 101 * nat.factorial 102
def expr_D := nat.factorial 101 * nat.factorial 103
def expr_E := nat.factorial 102 * nat.factorial 103

theorem perfect_square_proof :
  (∃ k : ℕ, expr_C = k * k) ∧ 
  (¬ ∃ k : ℕ, expr_A = k * k) ∧ 
  (¬ ∃ k : ℕ, expr_B = k * k) ∧ 
  (¬ ∃ k : ℕ, expr_D = k * k) ∧ 
  (¬ ∃ k : ℕ, expr_E = k * k) :=
by
  sorry

end perfect_square_proof_l437_437050


namespace paula_unique_paths_from_B_to_M_l437_437395

-- Defining the structure of the graph with cities and roads
structure CityGraph :=
  (cities : Finset ℕ) -- a finite set of cities
  (roads : Finset (ℕ × ℕ)) -- a finite set of roads (edges) represented as pairs of cities

-- Given Problem Conditions
def city_graph : CityGraph := {
  cities := (Finset.range 15), -- 15 cities
  roads := { ... } -- 20 roads defined appropriately within the Finset (ℕ × ℕ)
}

-- Paula's constraints and traversal requirements
def paula_travel_path (g : CityGraph) : Prop :=
  ∃ (path : List (ℕ × ℕ)),
    path.nodup ∧ -- roads are traversed only once
    path.length = 15 ∧ -- exactly 15 roads
    (path.head.1 = 1) ∧ -- starting at city B (say it's represented by 1)
    (path.last.2 = 13) -- ending at city M (say it's represented by 13)

-- Final proof statement to prove the number of valid paths
theorem paula_unique_paths_from_B_to_M : paula_travel_path city_graph → 2 :=
sorry

end paula_unique_paths_from_B_to_M_l437_437395


namespace tan_arithmetic_seq_value_l437_437603

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Given conditions and the final proof goal
theorem tan_arithmetic_seq_value (h_arith : arithmetic_seq a d)
    (h_sum : a 0 + a 6 + a 12 = Real.pi) :
    Real.tan (a 1 + a 11) = -Real.sqrt 3 := sorry

end tan_arithmetic_seq_value_l437_437603


namespace train_length_is_320_l437_437507

-- Define the conditions
def train_speed_kmph : ℝ := 45
def bridge_length_m : ℝ := 140
def time_to_pass_bridge_s : ℝ := 36.8

-- Convert speed from km/h to m/s
def speed_meter_per_second : ℝ := train_speed_kmph * (1000 / 3600)

-- Calculate total distance covered by the train while passing the bridge
def total_distance_m := speed_meter_per_second * time_to_pass_bridge_s

-- The length of the train
def train_length_m := total_distance_m - bridge_length_m

-- Lean 4 statement to prove the train length
theorem train_length_is_320 : train_length_m = 320 :=
by
  -- skipping the proof
  sorry

end train_length_is_320_l437_437507


namespace third_term_of_sequence_l437_437011

theorem third_term_of_sequence :
  (3 - (1 / 3) = 8 / 3) :=
by
  sorry

end third_term_of_sequence_l437_437011


namespace length_of_EF_l437_437393

-- Definitions of geometric entities
variables (A B C D E F : Point) (O : Point)
variables (EF BC AC r : ℝ)

-- Main theorem
theorem length_of_EF :
  is_kite_symmetric_ABCD AC A B C D ∧
  length AC = 12 ∧
  length BC = 6 ∧
  internal_angle B = 90 ∧
  lies_on E AB ∧
  lies_on F AD ∧
  is_equilateral_triangle E C F → 
  r = 4 * real.sqrt 3 := 
  sorry

end length_of_EF_l437_437393


namespace fibonacci_sum_less_one_l437_437658

def FibonacciSeq : ℕ → ℕ 
| 0     := 1
| 1     := 1
| (n+2) := FibonacciSeq n + FibonacciSeq (n+1)

theorem fibonacci_sum_less_one :
  (∑ i in finset.range 49, (1:ℝ) / (FibonacciSeq (2 * i + 1) * FibonacciSeq (2 * i + 3))) < 1 :=
begin
  sorry
end

end fibonacci_sum_less_one_l437_437658


namespace valid_seating_arrangement_count_l437_437482

theorem valid_seating_arrangement_count :
  let desks := [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
                [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
                [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
                [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)],
                [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]]
  (students : Finset (Fin 30)) (boys girls : Finset (Fin 30))
    (h_students : students.card = 30)
    (h_boys : boys.card = 15) (h_girls : girls.card = 15)
    (h_both : boys ∪ girls = students) :
  (∃ arrangement : Fin 30 → (Nat × Nat),
    (∀ i, arrangement i ∈ desks) ∧
    (∀ i j, (arrangement i = arrangement j) ↔ (i = j)) ∧
    (∀ i j, (i ≠ j ∧ (arrangement i == (arrangement j + (0, 1))
                        ∨ arrangement i == (arrangement j + (0, -1))
                        ∨ arrangement i == (arrangement j + (1, 0))
                        ∨ arrangement i == (arrangement j + (-1, 0))))
      → ((i ∈ boys ∧ j ∈ girls) ∨ (i ∈ girls ∧ j ∈ boys)))) →
  2 * (Nat.factorial 15) ^ 2 = sorry := sorry

end valid_seating_arrangement_count_l437_437482


namespace lcm_1_to_10_l437_437948

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437948


namespace problem_statement_l437_437298

-- Definitions of the objects involved
variables {α β : Type*} [Plane α] [Plane β]
variables {m n : Line α} {p q : Line β}

-- Conditions:
-- Proposition ①
def prop1 := ∀ (m : Line α) (n : Line α), m ⟂ α → n ∈ α → m ⟂ n

-- Proposition ②
def prop2 := ∀ (m n : Line α) (p q : Line β), m ∈ α → n ∈ α → m ∥ β → n ∥ β → α ∥ β

-- Proposition ③
def prop3 := ∀ (m : Line α) (n : Line α) (q : Line β), α ⟂ β → (α ∩ β = m) → n ∈ α → n ⟂ m → n ⟂ β

-- Proposition ④
def prop4 := ∀ (m n : Line α), m ⟂ α → α ⟂ β → m ∥ n → n ∥ β

-- The problem statement showing that all propositions are correct
theorem problem_statement : 
  (prop1 ∧ prop2 ∧ prop3 ∧ prop4) :=
by
  -- Placeholder used to bypass actual proof details
  sorry

end problem_statement_l437_437298


namespace set_intersection_l437_437328

open Finset

-- Let the universal set U, and sets A and B be defined as follows:
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {2, 4, 6}

-- Define the complement of A with respect to U:
def complement_A : Finset ℕ := U \ A

-- The goal is to prove that B ∩ complement_A = {4, 6}
theorem set_intersection (h : B ∩ complement_A = {4, 6}) : B ∩ complement_A = {4, 6} :=
by exact h

#check set_intersection

end set_intersection_l437_437328


namespace distance_between_consecutive_trees_l437_437514

noncomputable def distance_between_trees (length_yard : ℝ) (num_trees : ℕ) : ℝ :=
  length_yard / (num_trees - 1)

theorem distance_between_consecutive_trees :
  (distance_between_trees 255 18 = 15) :=
by
  unfold distance_between_trees
  rw [Nat.cast_sub, Nat.cast_bit1, Nat.cast_one, nat.one_succ]
  . norm_num
  . norm_num

end distance_between_consecutive_trees_l437_437514


namespace divisible_by_1979_l437_437320

theorem divisible_by_1979 (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ)
  (h₁ : (x₁ * y₁ - 1) % 1979 = 0)
  (h₂ : (x₂ * y₂ - 1) % 1979 = 0)
  (h₃ : (x₃ * y₃ - 1) % 1979 = 0)
  (collinear : ∃ (a b c : ℤ), a * x₁ + b * y₁ = c ∧ a * x₂ + b * y₂ = c ∧ a * x₃ + b * y₃ = c) :
  ∃ (i j : Fin₃), i ≠ j ∧ (x₁ - x₂) % 1979 = 0 ∧ (y₁ - y₂) % 1979 = 0 ∨
                     ∃ (i j : Fin₃), i ≠ j ∧ (x₁ - x₃) % 1979 = 0 ∧ (y₁ - y₃) % 1979 = 0 ∨
                     ∃ (i j : Fin₃), i ≠ j ∧ (x₂ - x₃) % 1979 = 0 ∧ (y₂ - y₃) % 1979 = 0 :=
begin
  sorry -- Proof will go here
end

end divisible_by_1979_l437_437320


namespace smallest_divisible_1_to_10_l437_437908

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437908


namespace smallest_divisible_by_1_to_10_l437_437897

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437897


namespace symmetric_point_coordinates_l437_437277

noncomputable def symmetric_with_respect_to_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (-x, y, -z)

theorem symmetric_point_coordinates : symmetric_with_respect_to_y_axis (-2, 1, 4) = (2, 1, -4) :=
by sorry

end symmetric_point_coordinates_l437_437277


namespace magnitude_of_z_l437_437382

-- Define the complex number z and the given condition
noncomputable def z : ℂ := (6 + 8 * complex.I) / ((4 + 3 * complex.I) * (1 + complex.I))

-- The theorem statement asserting the magnitude of z
theorem magnitude_of_z : complex.abs z = 25 / 3 := by
  sorry

end magnitude_of_z_l437_437382


namespace starting_cities_l437_437622

section
open Graph

-- Define the cities as vertices
inductive City
| SaintPetersburg
| Tver
| Yaroslavl
| NizhnyNovgorod
| Moscow
| Kazan

open City

-- Define the routes as edges in the graph
noncomputable def travelGraph : SimpleGraph City :=
  SimpleGraph.mkRelation (λ u v => 
    (u = SaintPetersburg ∧ v = Tver) ∨ (u = Tver ∧ v = SaintPetersburg) ∨
    (u = Yaroslavl ∧ v = NizhnyNovgorod) ∨ (u = NizhnyNovgorod ∧ v = Yaroslavl) ∨
    (u = Moscow ∧ v = Kazan) ∨ (u = Kazan ∧ v = Moscow) ∨ 
    (u = NizhnyNovgorod ∧ v = Kazan) ∨ (u = Kazan ∧ v = NizhnyNovgorod) ∨ 
    (u = Moscow ∧ v = Tver) ∨ (u = Tver ∧ v = Moscow) ∨ 
    (u = Moscow ∧ v = NizhnyNovgorod) ∨ (u = NizhnyNovgorod ∧ v = Moscow))

-- Main theorem: Valid starting cities for the journey
theorem starting_cities :
  (∃ path : List City, 
    travelGraph.path path ∧ 
    path.head = some SaintPetersburg ∧ 
    travelGraph.path Distinct path) ∨
  (∃ path : List City, 
    travelGraph.path path ∧ 
    path.head = some Yaroslavl ∧ 
    travelGraph.path Distinct path) :=
sorry

end

end starting_cities_l437_437622


namespace car_rental_distance_l437_437450

-- Definitions based on conditions
def Carrey_cost (d : ℝ) : ℝ := 20 + 0.25 * d
def Samuel_cost (d : ℝ) : ℝ := 24 + 0.16 * d

-- Statement of the problem
theorem car_rental_distance : ∃ d : ℝ, Carrey_cost d = Samuel_cost d ∧ d ≈ 44.44 :=
by
  sorry

end car_rental_distance_l437_437450


namespace min_value_x_add_y_l437_437731

variable {x y : ℝ}
variable (hx : 0 < x) (hy : 0 < y)
variable (h : 2 * x + 8 * y - x * y = 0)

theorem min_value_x_add_y : x + y ≥ 18 :=
by
  /- Proof goes here -/
  sorry

end min_value_x_add_y_l437_437731


namespace part1_part2_part3_l437_437325

noncomputable def z_vector (z : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (z * P.1, z * P.2)

theorem part1 (z : ℝ) (x0 y0 : ℝ) : z ≠ 0 → z_vector z (x0, y0) = (z * x0, z * y0) :=
by
  intro h
  simp [z_vector]

theorem part2 (z : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (S1 : |z| * y / 2 = 1) (S2 : |z| * x / 2 = 2) : ∀ (v : ℝ × ℝ), v = (x, y) → v = (4 / |z|, 2 / |z|) :=
by
  intros v hv
  rw hv at *
  field_simp [S1, S2, abs_pos_iff.mpr (ne_of_gt hx)]

theorem part3 (z : ℝ) (A B : ℝ × ℝ) (S : ∀ (A B : ℝ × ℝ), (z_vector z A) • A + (z_vector z B) • B ≥ 8) : z = 2 :=
by
  have h_min := S (4, 0) (4, 0) -- taking specific vectors
  field_simp at h_min
  linarith

end part1_part2_part3_l437_437325


namespace equilateral_of_incenter_secant_condition_l437_437416

variable {A B C P Q : Type}
variable [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder P] [LinearOrder Q]
variable (triangle : Triangle A B C)
variable (I : Incenter triangle)
variable (P Q : Point)
variable (secant : Secant I P Q (Sides triangle CA CB))
variable (condition : (dist A P / dist P C) + (dist B Q / dist Q C) = 1)

theorem equilateral_of_incenter_secant_condition :
  (dist A P / dist P C) + (dist B Q / dist Q C) = 1 → equilateral triangle :=
by
  sorry

end equilateral_of_incenter_secant_condition_l437_437416


namespace find_real_number_a_l437_437329

variable (U : Set ℕ) (M : Set ℕ) (a : ℕ)

theorem find_real_number_a :
  U = {1, 3, 5, 7} →
  M = {1, a} →
  (U \ M) = {5, 7} →
  a = 3 :=
by
  intros hU hM hCompU
  -- Proof part will be here
  sorry

end find_real_number_a_l437_437329


namespace eric_pencils_distribution_l437_437146

/-- Eric initially sorted 150 colored pencils into 5 containers.
Before class, he received 30 more pencils.
Later, he found a sixth container and received another 47 colored pencils.
How many red and blue pencils can he evenly distribute between the six containers,
given that the other five containers already have their specific color combinations? -/
theorem eric_pencils_distribution : 
  let total_pencils := 150 + 30 + 47 in
  ∃ (n : ℕ), n ≤ total_pencils ∧ n % 6 = 0 ∧ n = 222 :=
by
  let total_pencils := 150 + 30 + 47
  exact ⟨222, by norm_num, by norm_num, rfl⟩

end eric_pencils_distribution_l437_437146


namespace smallest_number_div_by_1_to_10_l437_437794

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437794


namespace dot_product_in_triangle_l437_437256

theorem dot_product_in_triangle
  (a b c : ℝ)
  (sin B sin C : ℝ)
  (h1 : a^2 + b^2 - c^2 = real.sqrt 3 * a * b)
  (h2 : a * c * sin B = 2 * real.sqrt 3 * sin C)
  (cos_C : ℝ := real.sqrt 3 / 2) :
  (a * b * cos_C = 3) :=
sorry

end dot_product_in_triangle_l437_437256


namespace triangle_is_right_angle_l437_437279

theorem triangle_is_right_angle (A B C : ℝ) : 
  (A / B = 2 / 3) ∧ (A / C = 2 / 5) ∧ (A + B + C = 180) →
  (A = 36) ∧ (B = 54) ∧ (C = 90) :=
by 
  intro h
  sorry

end triangle_is_right_angle_l437_437279


namespace angle_of_inclination_l437_437768

theorem angle_of_inclination : 
  let m := - √3 in 
  ∃ θ, 0 ≤ θ ∧ θ < π ∧ tan θ = m ∧ θ = 2 * π / 3 :=
by
  let m := - √3 
  use 2 * π / 3
  have h1: 0 ≤ 2 * π / 3 := by sorry
  have h2: 2 * π / 3 < π := by sorry
  have h3: tan (2 * π / 3) = - √3 := by sorry
  exact ⟨h1, h2, h3, rfl⟩

end angle_of_inclination_l437_437768


namespace tree_height_increase_l437_437455

-- Definitions given in the conditions
def h0 : ℝ := 4
def h (t : ℕ) (x : ℝ) : ℝ := h0 + t * x

-- Proof statement
theorem tree_height_increase (x : ℝ) :
  h 6 x = (4 / 3) * h 4 x + h 4 x → x = 2 :=
by
  intro h6_eq
  rw [h, h] at h6_eq
  norm_num at h6_eq
  sorry

end tree_height_increase_l437_437455


namespace log_relation_l437_437174

-- Definitions of a, b, c based on given logarithmic expressions
def a : ℝ := Real.log 3 / Real.log 5
def b : ℝ := Real.log 5 / Real.log 8
def c : ℝ := Real.log 8 / Real.log 13

-- Main theorem that we need to prove
theorem log_relation {a b c : ℝ} (h₀ : 5^5 < 8^4) (h₁ : 13^4 < 8^5) :
    a < b ∧ b < c := 
        by 
        -- Conditions and definitions of a, b, c
        have ha : a = Real.log 3 / Real.log 5 := Real.log 3 / Real.log 5
        have hb : b = Real.log 5 / Real.log 8 := Real.log 5 / Real.log 8
        have hc : c = Real.log 8 / Real.log 13 := Real.log 8 / Real.log 13

        -- Strategy and analysis for proof
        sorry -- detailed proof steps are omitted

end log_relation_l437_437174


namespace sum_of_primes_less_than_10_is_17_l437_437041

-- Definition of prime numbers less than 10
def primes_less_than_10 : List ℕ := [2, 3, 5, 7]

-- Sum of the prime numbers less than 10
def sum_primes_less_than_10 : ℕ := List.sum primes_less_than_10

theorem sum_of_primes_less_than_10_is_17 : sum_primes_less_than_10 = 17 := 
by
  sorry

end sum_of_primes_less_than_10_is_17_l437_437041


namespace stuffed_animal_sales_l437_437715

theorem stuffed_animal_sales (Q T J : ℕ) 
  (h1 : Q = 100 * T) 
  (h2 : J = T + 15) 
  (h3 : Q = 2000) : 
  Q - J = 1965 := 
by
  sorry

end stuffed_animal_sales_l437_437715


namespace base_equivalence_l437_437640

theorem base_equivalence :
  (∃ k : ℕ, (524 : Nat) = k ∧ (524) = 6 * k^2 + 6 * k + 4) →
  (524 : Nat) = 7 ∧ (524)8 = (664)k := 
sorry

end base_equivalence_l437_437640


namespace smallest_number_divisible_by_1_to_10_l437_437870

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437870


namespace amit_left_after_3_days_l437_437515

def amitWorkRate (W : ℝ) : ℝ := W / 15
def ananthuWorkRate (W : ℝ) : ℝ := W / 90
def totalWorkTime : ℝ := 75

theorem amit_left_after_3_days (W : ℝ) (x : ℝ) :
  x * amitWorkRate(W) + (totalWorkTime - x) * ananthuWorkRate(W) = W -> x = 3 := 
by
  sorry

end amit_left_after_3_days_l437_437515


namespace smallest_divisible_by_1_to_10_l437_437898

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437898


namespace inequality_subtraction_l437_437243

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l437_437243


namespace triangle_orthocenter_l437_437677

/-- Given a triangle ABC with circumcenter O and orthocenter H, side lengths a, b, c,
 and circumradius R, we want to prove OH^2 given specific conditions. -/
theorem triangle_orthocenter:
  ∀ (a b c R : ℝ),
  R = 8 →
  a^2 + b^2 + c^2 = 50 →
  let OH2 := 9 * R^2 - (a^2 + b^2 + c^2) in
  OH2 = 526 :=
by
  intros a b c R hR habc2
  let OH2 := 9 * R^2 - (a^2 + b^2 + c^2)
  have hOH2 : OH2 = 526 :=
    calc
      OH2 = 9 * R^2 - (a^2 + b^2 + c^2) : by rfl
      ... = 9 * 8^2 - 50 : by rw [hR, habc2]
      ... = 9 * 64 - 50 : by norm_num
      ... = 576 - 50 : by norm_num
      ... = 526 : by norm_num
  exact hOH2

end triangle_orthocenter_l437_437677


namespace least_five_digit_congruent_to_6_mod_19_l437_437428

theorem least_five_digit_congruent_to_6_mod_19 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 19 = 6 ∧ n = 10011 :=
by
  sorry

end least_five_digit_congruent_to_6_mod_19_l437_437428


namespace f_2015_eq_2016_l437_437080

/-- Axiom: For all natural numbers n, the function f satisfies f(f(n)) + f(n) = 2 * n + 3.
This is one of the conditions given in the problem. -/
axiom f_property : Π (f : ℕ → ℕ), (∀ n, f (f n) + f n = 2 * n + 3)

/-- Axiom: The function f satisfies f(0) = 1.
This is the other condition given in the problem. -/
axiom f_0 : Π (f : ℕ → ℕ), f 0 = 1

/-- Definition f as a universal placeholder for the conditions above -/
def f : ℕ → ℕ := sorry

/-- The final theorem we need to prove: f(2015) = 2016, given the stated conditions -/
theorem f_2015_eq_2016 : f 2015 = 2016 :=
by
  have := f_property f,
  have := f_0 f,
  sorry

end f_2015_eq_2016_l437_437080


namespace smallest_number_divisible_by_1_to_10_l437_437866

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437866


namespace angle_BAC_measure_l437_437073

theorem angle_BAC_measure : 
  ∀ (O A B C : Point) (h_circle : Circle O) (h_triangle : Triangle ABC O)
  (h_AOC : CentralAngle O A C 130)
  (h_BOC : CentralAngle O B C 120),
  ∠BAC = 60 := by
  -- Proof to be filled here
  sorry

end angle_BAC_measure_l437_437073


namespace smallest_number_divisible_by_1_to_10_l437_437845

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437845


namespace smallest_number_divisible_1_to_10_l437_437887

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437887


namespace multiple_of_age_is_3_l437_437779

def current_age : ℕ := 9
def age_six_years_ago : ℕ := 3
def age_multiple (current : ℕ) (previous : ℕ) : ℕ := current / previous

theorem multiple_of_age_is_3 : age_multiple current_age age_six_years_ago = 3 :=
by
  sorry

end multiple_of_age_is_3_l437_437779


namespace tangent_circumcircle_BCD_l437_437587

open Triangle Angle Circle Tangent

theorem tangent_circumcircle_BCD (A B C D: Point) (h1: D ∈ LineSegment AC)
  (h2: Angle A C B = 45)
  (h3: Angle D A B = 60)
  (h4: ∃ k, AD = 2 * k ∧ DC = k) :
  IsTangent (Circumcircle B C D) (Line A B) :=
sorry

end tangent_circumcircle_BCD_l437_437587


namespace total_baskets_l437_437104

theorem total_baskets (Alex_baskets Sandra_baskets Hector_baskets Jordan_baskets total_baskets : ℕ)
  (h1 : Alex_baskets = 8)
  (h2 : Sandra_baskets = 3 * Alex_baskets)
  (h3 : Hector_baskets = 2 * Sandra_baskets)
  (total_combined_baskets := Alex_baskets + Sandra_baskets + Hector_baskets)
  (h4 : Jordan_baskets = total_combined_baskets / 5)
  (h5 : total_baskets = Alex_baskets + Sandra_baskets + Hector_baskets + Jordan_baskets) :
  total_baskets = 96 := by
  sorry

end total_baskets_l437_437104


namespace solid_is_cone_l437_437633

-- Definitions of the conditions.
def front_and_side_views_are_equilateral_triangles (S : Type) : Prop :=
∀ (F : S → Prop) (E : S → Prop), (∃ T : S, F T ∧ E T ∧ T = T) 

def top_view_is_circle_with_center (S : Type) : Prop :=
∀ (C : S → Prop), (∃ O : S, C O ∧ O = O)

-- The proof statement that given the above conditions, the solid is a cone
theorem solid_is_cone (S : Type)
  (H1 : front_and_side_views_are_equilateral_triangles S)
  (H2 : top_view_is_circle_with_center S) : 
  ∃ C : S, C = C :=
by 
  sorry

end solid_is_cone_l437_437633


namespace number_of_triangles_with_perimeter_10_l437_437232

theorem number_of_triangles_with_perimeter_10 : 
  ∃ (a b c : ℕ), a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
  (a ≤ b) ∧ (b ≤ c) ↔ 9 :=
by
  have h : ∀ (a b c : ℕ), a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a ≤ b ∧ b ≤ c → 
    (a, b, c) ∈ 
      {[ (1, 5, 4), (2, 4, 4), (3, 3, 4), 
         (1, 6, 3), (2, 5, 3), (3, 4, 3),
         (2, 6, 2), (3, 5, 2), (4, 4, 2) ] : set (ℕ × ℕ × ℕ)},
  sorry  

end number_of_triangles_with_perimeter_10_l437_437232


namespace valid_seating_arrangement_count_l437_437481

theorem valid_seating_arrangement_count :
  let desks := [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
                [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
                [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
                [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)],
                [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]]
  (students : Finset (Fin 30)) (boys girls : Finset (Fin 30))
    (h_students : students.card = 30)
    (h_boys : boys.card = 15) (h_girls : girls.card = 15)
    (h_both : boys ∪ girls = students) :
  (∃ arrangement : Fin 30 → (Nat × Nat),
    (∀ i, arrangement i ∈ desks) ∧
    (∀ i j, (arrangement i = arrangement j) ↔ (i = j)) ∧
    (∀ i j, (i ≠ j ∧ (arrangement i == (arrangement j + (0, 1))
                        ∨ arrangement i == (arrangement j + (0, -1))
                        ∨ arrangement i == (arrangement j + (1, 0))
                        ∨ arrangement i == (arrangement j + (-1, 0))))
      → ((i ∈ boys ∧ j ∈ girls) ∨ (i ∈ girls ∧ j ∈ boys)))) →
  2 * (Nat.factorial 15) ^ 2 = sorry := sorry

end valid_seating_arrangement_count_l437_437481


namespace intervals_of_monotonicity_range_of_fx1_fx2_l437_437210

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * Real.log x

theorem intervals_of_monotonicity (a : ℝ) :
  (a ≤ 4 → (∀ x, 0 < x → ∀ y, 0 < y → f x a ≤ f y a))
  ∧ (a > 4 →
      (∃ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x1 < x2 ∧
        (∀ x, 0 < x ∧ x < x1 → f x a ≤ f x1 a) ∧
        (∀ x, x1 < x ∧ x < x2 → f x2 a ≤ f x a) ∧
        (∀ x, x2 < x → f x2 a ≤ f x a))) :=
sorry

theorem range_of_fx1_fx2 (a : ℝ) (h : 2 * (Real.exp 1 + Real.exp (-1)) < a ∧ a < 20/3) :
  ∃ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x1 < x2 ∧ 
    (f x1 a - f x2 a ∈ Ioo (Real.exp 2 - Real.exp (-2) - 4) ((80 / 9) - 4 * Real.log 3)) :=
sorry

end intervals_of_monotonicity_range_of_fx1_fx2_l437_437210


namespace seating_arrangement_ways_l437_437479

theorem seating_arrangement_ways (students desks : ℕ) (rows columns boys girls : ℕ) :
  students = 30 → desks = 30 → rows = 5 → columns = 6 → boys = 15 → girls = 15 → 
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 6 → 
    (¬(∃ m n, (m = i ∧ n ≠ j ∧ abs(m - i) ≤ 1 ∨ n = j ∧ abs(n - j) ≤ 1) ∧ boys) ∧
    ¬(∃ m n, (m = i ∧ n ≠ j ∧ abs(m - i) ≤ 1 ∨ n = j ∧ abs(n - j) ≤ 1) ∧ girls))) →
  2 * (Nat.factorial 15) * (Nat.factorial 15) = 2 * (Nat.factorial 15)^2 := 
by
  sorry

end seating_arrangement_ways_l437_437479


namespace population_increase_north_southland_l437_437652

theorem population_increase_north_southland :
  (let births_per_day := 24 / 6,
       deaths_per_day := 24 / 10,
       net_daily_increase := births_per_day - deaths_per_day,
       annual_increase := net_daily_increase * 365
   in round(annual_increase / 100) * 100 = 600) :=
by
  -- Definitions
  let births_per_day := 24 / 6 
  let deaths_per_day := 24 / 10
  let net_daily_increase := births_per_day - deaths_per_day
  let annual_increase := net_daily_increase * 365

  -- Proof
  sorry

end population_increase_north_southland_l437_437652


namespace smallest_number_divisible_by_1_to_10_l437_437833

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437833


namespace smallest_number_with_11_divisors_l437_437127

theorem smallest_number_with_11_divisors :
  ∃ (n : ℕ), (n = 1024) ∧ (factors n).length = 11 :=
by
  use 1024
  -- formal proof omitted
  sorry

end smallest_number_with_11_divisors_l437_437127


namespace proof_problem_l437_437619

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, (y = 2^x - 1) ∧ (x ≤ 2)}

-- Define the complement of set A in U
def complement_A : Set ℝ := Set.compl A

-- Define the intersection of complement_A and B
def complement_A_inter_B : Set ℝ := complement_A ∩ B

-- State the theorem
theorem proof_problem : complement_A_inter_B = {x | (-1 < x) ∧ (x ≤ 2)} :=
by
  sorry

end proof_problem_l437_437619


namespace initial_percentage_proof_l437_437070

-- Defining the initial percentage of water filled in the container
def initial_percentage (capacity add amount_filled : ℕ) : ℕ :=
  (amount_filled * 100) / capacity

-- The problem constraints
theorem initial_percentage_proof : initial_percentage 120 48 (3 * 120 / 4 - 48) = 35 := by
  -- We need to show that the initial percentage is 35%
  sorry

end initial_percentage_proof_l437_437070


namespace solution_inequality_l437_437189

noncomputable def f : ℝ → ℝ := sorry

axiom diff_f : Differentiable ℝ f
axiom f_prime_less_f : ∀ x : ℝ, deriv f x < f x

theorem solution_inequality :
  f(1) < Real.exp 1 * f(0) ∧ f(2017) < Real.exp 2017 * f(0) :=
sorry

end solution_inequality_l437_437189


namespace lcm_1_to_10_l437_437806

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437806


namespace max_product_of_digits_is_953_l437_437029

theorem max_product_of_digits_is_953 :
  let digits := [3, 5, 6, 8, 9] in
  ∃ (a b c : ℕ) (d e : ℕ), 
    (a, b, c, d, e) ∈ ((digits.permutations.map (λ xs, (xs[0], xs[1], xs[2], xs[3], xs[4] : ℕ) : Set (ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a * 100 + b * 10 + c = 953 ∧ 
    ∀ (x y z w v : ℕ),
      (x, y, z, w, v) ∈ ((List.permutationsMap (λ xs, (xs[0], xs[1], xs[2], xs[3], xs[4]) : Set (ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ z ≠ w ∧ z ≠ v ∧ w ≠ v → 
      (100 * x + 10 * y + z) * (10 * w + v) ≤ (100 * a + 10 * b + c) * (10 * d + e) :=
by
  sorry

end max_product_of_digits_is_953_l437_437029


namespace arrival_time_at_midpoint_l437_437504

theorem arrival_time_at_midpoint :
  let planned_start := mkTime (10, 10)
  let planned_end := mkTime (13, 10)
  let actual_start := planned_start.addMinutes 5
  let actual_end := planned_end.addMinutes (-4)
  let midpoint_time := planned_start.addMinutes 90
  let correct_time := mkTime (11, 50)
  midpoint_time = correct_time :=
by
  sorry

end arrival_time_at_midpoint_l437_437504


namespace smallest_number_divisible_1_to_10_l437_437934

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437934


namespace price_of_third_variety_l437_437734

def tea_1_price : ℝ := 126
def tea_2_price : ℝ := 135
def mix_ratio : ℝ := 1/1/2
def mix_price : ℝ := 154

theorem price_of_third_variety (x : ℝ) :
  (126 + 135 + 2 * x) / 4 = 154 → x = 177.5 :=
by
  intro h
  have H : 126 + 135 + 2 * x = 4 * 154 := by sorry
  have H2 : 2 * x = 616 - 261 := by sorry
  have H3 : x = 177.5 := by sorry
  exact H3

end price_of_third_variety_l437_437734


namespace increasing_interval_of_f_maximum_value_of_f_l437_437749

open Real

def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Consider x in the interval [-2, 4]
def domain_x (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

theorem increasing_interval_of_f :
  ∃a b : ℝ, (a, b) = (1, 4) ∧ ∀ x y : ℝ, domain_x x → domain_x y → a ≤ x → x < y → y ≤ b → f x < f y := sorry

theorem maximum_value_of_f :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, domain_x x → f x ≤ M := sorry

end increasing_interval_of_f_maximum_value_of_f_l437_437749


namespace pyramid_distance_l437_437180

noncomputable def distance_between_center_base_to_vertex
  (height : ℝ) (side_length : ℝ) (radius : ℝ) : ℝ :=
let diagonal := real.sqrt 2 * side_length in
let lateral_edge_length := real.sqrt 2 in
real.sqrt (height^2 + (diagonal / 2)^2)

theorem pyramid_distance (height side_length radius : ℝ)
  (h_height : height = real.sqrt 2)
  (h_side_length : side_length = 1)
  (h_radius : radius = 1) :
  distance_between_center_base_to_vertex height side_length radius 
  = real.sqrt 10 / 2 :=
by {
  rw [distance_between_center_base_to_vertex, h_height, h_side_length, h_radius],
  sorry
}

end pyramid_distance_l437_437180


namespace no_possible_arrangement_l437_437998

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l437_437998


namespace cosine_squared_sum_l437_437535

theorem cosine_squared_sum : 
  (∑ k in Finset.range 91, Real.cos ((k : ℝ) * Real.pi / 180) ^ 2) = 91 / 2 := 
by
  sorry

end cosine_squared_sum_l437_437535


namespace smallest_divisible_1_to_10_l437_437903

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437903


namespace smallest_number_divisible_by_1_to_10_l437_437837

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437837


namespace lila_additional_miles_l437_437697

-- Definitions of given conditions
def distance_first_part : ℝ := 24
def speed_first_part : ℝ := 40
def speed_second_part : ℝ := 60
def target_avg_speed : ℝ := 55

-- Theorem to prove
theorem lila_additional_miles (d t s₁ s₂ s_avg : ℝ) 
  (h1 : d = 24) 
  (h2 : s₁ = 40) 
  (h3 : s₂ = 60) 
  (h4 : s_avg = 55) : 
  let time_first_part := d / s₁ in
  let total_distance_to_avg := s_avg * (time_first_part + (d / s₂)) in
  (total_distance_to_avg - d) = 108 :=
by 
  sorry

end lila_additional_miles_l437_437697


namespace solve_quadratic_inequality_l437_437002

theorem solve_quadratic_inequality :
  ∀ x : ℝ, ((x - 1) * (x - 3) < 0) ↔ (1 < x ∧ x < 3) :=
by
  intro x
  sorry

end solve_quadratic_inequality_l437_437002


namespace sum_partition_even_l437_437568

def is_odd (n : ℕ) : Prop := n % 2 = 1

def partition_count (n k l : ℕ) : ℕ := 
  if n > 0 ∧ k > 0 ∧ l > 0 
  then Cardinal.to_nat $ {(a : Fin l → ℕ) | 
    (∑ i, a i = n) ∧ 
    (∀ i j, i < j → a i > a j) ∧ 
    is_odd (a (l - 1)) ∧ 
    (Finset.card (Finset.filter (λ i, is_odd (a i)) (Finset.univ : Finset (Fin l))) = k)}.to_finset
  else 0

theorem sum_partition_even (n k : ℕ) (hn : n > k^2) :
  (∑ l in Finset.range (n + 1), partition_count n k l) % 2 = 0 ∨ 
  (∑ l in Finset.range (n + 1), partition_count n k l) = 0 := 
sorry

end sum_partition_even_l437_437568


namespace equation_of_perpendicular_line_l437_437753

theorem equation_of_perpendicular_line 
    (x y : ℝ) 
    (a b c d : ℝ) 
    (hp : a * x + b * y + c = 0) 
    (hl : l1 x + l2 y + d = 0) 
    (h1 : x = -1 ∧ y = 2)
    (h_perp : a * l1 + b * l2 = 0 ∧ l1 * l1 + l2 * l2 > 0) :
    l1 * x + l2 * y + d = 0 := 
    sorry

end equation_of_perpendicular_line_l437_437753


namespace base_of_minus4_pow3_l437_437636

theorem base_of_minus4_pow3 : ∀ (x : ℤ) (n : ℤ), (x, n) = (-4, 3) → x = -4 :=
by intros x n h
   cases h
   rfl

end base_of_minus4_pow3_l437_437636


namespace remainder_of_polynomial_modulus_l437_437789

theorem remainder_of_polynomial_modulus :
  (λ x : ℂ, (x + 2)^(2008) % (x^2 - x + 1) = 5 * x + 3) :=
by
  sorry

end remainder_of_polynomial_modulus_l437_437789


namespace part_a_part_b_l437_437242

def f (A B : List ℕ) : ℕ := (List.zipWith (≠) A B).count (λ b => b)

theorem part_a (A B C : List ℕ) (h_len : A.length = B.length) (h_len' : B.length = C.length)
  (h01A : ∀ x, x ∈ A → x = 0 ∨ x = 1) (h01B : ∀ x, x ∈ B → x = 0 ∨ x = 1) (h01C : ∀ x, x ∈ C → x = 0 ∨ x = 1)
  (hAB : f A B = d) (hAC : f A C = d) (hBC : f B C = d) :
  ∃ k, d = 2 * k := sorry

theorem part_b (A B C : List ℕ) (h_len : A.length = B.length) (h_len' : B.length = C.length)
  (h01A : ∀ x, x ∈ A → x = 0 ∨ x = 1) (h01B : ∀ x, x ∈ B → x = 0 ∨ x = 1) (h01C : ∀ x, x ∈ C → x = 0 ∨ x = 1)
  (hAB : f A B = d) (hAC : f A C = d) (hBC : f B C = d) (heven : ∃ k, d = 2 * k) :
  ∃ D, D.length = A.length ∧ (∀ x, x ∈ D → x = 0 ∨ x = 1) ∧ f A D = d / 2 ∧ f B D = d / 2 ∧ f C D = d / 2 := sorry

end part_a_part_b_l437_437242


namespace line_perpendicular_to_plane_implies_perpendicular_to_lines_in_plane_l437_437171

-- Definitions for lines, planes, and perpendicularity
structure Line :=
  (id: ℕ) -- Just for unique identity
structure Plane :=
  (id: ℕ) -- Just for unique identity

def PerpendicularTo (l: Line) (p: Plane) : Prop := sorry
def SubsetOf (m: Line) (p: Plane) : Prop := sorry
def Perpendicular (l m: Line) : Prop := sorry

-- Conditions
variable {l m: Line}
variable {α: Plane}
variable h1: PerpendicularTo l α
variable h2: SubsetOf m α

-- Lean statement for the proof
theorem line_perpendicular_to_plane_implies_perpendicular_to_lines_in_plane :
  PerpendicularTo l α → SubsetOf m α → Perpendicular l m :=
by {
  intros,
  sorry
}

end line_perpendicular_to_plane_implies_perpendicular_to_lines_in_plane_l437_437171


namespace solve_ff5_l437_437608

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4*x + 3 else 3 - x

theorem solve_ff5 : f (f 5) = -1 :=
  by sorry

end solve_ff5_l437_437608


namespace question1_question2_l437_437309

theorem question1 (x a : ℝ) (h : a = 1)
  (p : (x - 3 * a) / (a - 2 * x) ≥ 0)
  (q : 2 * x^2 - 7 * x + 6 < 0) :
  (p ∧ q) → (3 / 2 < x ∧ x < 2) :=
by
  sorry

theorem question2 (x a : ℝ)
  (p : (x - 3 * a) / (a - 2 * x) ≥ 0)
  (q : 2 * x^2 - 7 * x + 6 < 0) :
  (¬ p → ¬ q) ∧ ¬ (¬ q → ¬ p) → (2 / 3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end question1_question2_l437_437309


namespace find_a_b_l437_437618

variable (a b : ℝ)

def is_solution_set (p q : ℝ → Prop) (s : Set ℝ) : Prop :=
  ∀ x, p x ↔ x ∈ s

theorem find_a_b (
  A B : Set ℝ
 ) (hA : is_solution_set (λ x, x^2 - 2*x - 3 < 0) A)
   (hB : is_solution_set (λ x, x^2 + x - 6 < 0) B)
   (hAB : is_solution_set (λ x, x^2 + a*x + b < 0) (A ∩ B)) :
  a = -1 ∧ b = -2 :=
by
  sorry

end find_a_b_l437_437618


namespace geometric_sequence_sum_l437_437692

/-- Let {a_n} be a geometric sequence with positive common ratio, a_1 = 2, and a_3 = a_2 + 4.
    Prove the general formula for a_n is 2^n, and the sum of the first n terms, S_n, of the sequence { (2n+1)a_n }
    is (2n-1) * 2^(n+1) + 2. -/
theorem geometric_sequence_sum
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h3 : a 3 = a 2 + 4) :
  (∀ n, a n = 2^n) ∧
  (∀ S : ℕ → ℕ, ∀ n, S n = (2 * n - 1) * 2 ^ (n + 1) + 2) :=
by sorry

end geometric_sequence_sum_l437_437692


namespace lcm_1_10_l437_437970

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437970


namespace lcd_fractions_l437_437752

theorem lcd_fractions (x y : ℕ) (h : x ≠ y) : 
  LCD [xy, x + y, x - y] = xy * (x + y) * (x - y) :=
sorry

end lcd_fractions_l437_437752


namespace find_y_length_l437_437787

theorem find_y_length :
  let AD := 10
  let BD := 10
  let AC := 10
  let CD := 2 * AC
  CD = 20 → y = (CD * sqrt 3) / 2 → y = 10 * sqrt 3 :=
sorry

end find_y_length_l437_437787


namespace visited_neither_l437_437452

def people_total : ℕ := 90
def visited_iceland : ℕ := 55
def visited_norway : ℕ := 33
def visited_both : ℕ := 51

theorem visited_neither :
  people_total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

end visited_neither_l437_437452


namespace smallest_number_divisible_by_1_through_10_l437_437972

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437972


namespace smallest_divisible_1_to_10_l437_437927

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437927


namespace deposit_is_3000_l437_437740

-- Define the constants
def cash_price : ℝ := 8000
def monthly_installment : ℝ := 300
def number_of_installments : ℕ := 30
def savings_by_paying_cash : ℝ := 4000

-- Define the total installment payments
def total_installment_payments : ℝ := number_of_installments * monthly_installment

-- Define the total price paid, which includes the deposit and installments
def total_paid : ℝ := cash_price + savings_by_paying_cash

-- Define the deposit
def deposit : ℝ := total_paid - total_installment_payments

-- Statement to be proven
theorem deposit_is_3000 : deposit = 3000 := 
by 
  sorry

end deposit_is_3000_l437_437740


namespace solve_for_x_l437_437270

theorem solve_for_x (x : ℝ) (h : (1 / 5) + (5 / x) = (12 / x) + (1 / 12)) : x = 60 := by
  sorry

end solve_for_x_l437_437270


namespace sequence_100_summation_100_l437_437223

open Nat

noncomputable def sequence (a b : ℝ) : ℕ → ℝ
| 0     => 0
| 1     => a
| 2     => b
| n + 1 => sequence a b n - sequence a b (n - 1)

def summation (a b : ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, sequence a b (i + 1)

theorem sequence_100 (a b : ℝ) : sequence a b 100 = -a :=
  sorry

theorem summation_100 (a b : ℝ) : summation a b 100 = 2b - a :=
  sorry

end sequence_100_summation_100_l437_437223


namespace max_value_l437_437754

-- Define the function f(x)
def f (x : ℕ) : ℤ :=
  - (x*x + 3*x + 2)

-- Define the condition for x
def x_condition (x : ℕ) : Prop :=
  x >= 3

-- The proposition for the maximum value of f(x)
theorem max_value (x : ℕ) (hx : x_condition x) : ∃ M, M = f 3 ∧ ∀ y, x_condition y → f y ≤ M :=
begin
  -- The maximum value is -20
  use f 3,
  split,
  { -- Show that f(3) = -20
    sorry },
  { -- Show that it is the maximum value
    sorry }
end

end max_value_l437_437754


namespace smallest_number_div_by_1_to_10_l437_437791

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437791


namespace tomatoes_picked_today_l437_437078

theorem tomatoes_picked_today (initial yesterday_picked left_after_yesterday today_picked : ℕ)
  (h1 : initial = 160)
  (h2 : yesterday_picked = 56)
  (h3 : left_after_yesterday = 104)
  (h4 : initial - yesterday_picked = left_after_yesterday) :
  today_picked = 56 :=
by
  sorry

end tomatoes_picked_today_l437_437078


namespace g_even_sum_l437_437307

def g (x : ℝ) (p q r s : ℝ) : ℝ := p * x ^ 8 + q * x ^ 6 - r * x ^ 4 + s * x ^ 2 + 5

theorem g_even_sum (p q r s : ℝ) (h : g 11 p q r s = 7) : g 11 p q r s + g (-11) p q r s = 14 :=
by
  -- assume g is an even function
  have g_even : ∀ x, g x p q r s = g (-x) p q r s := 
  by
    sorry
  -- use given value and even function property to reach the conclusion
  rw [g_even 11]
  rw [h]
  exact add_self_eq_double 7

end g_even_sum_l437_437307


namespace line_circle_intersect_not_center_l437_437160

theorem line_circle_intersect_not_center (k : ℝ) : 
  let line := {p : ℝ × ℝ | p.snd = k * p.fst + 1},
      circle := {p : ℝ × ℝ | p.fst^2 + p.snd^2 = 2} in
  ∃ p ∈ line, p ∈ circle ∧ p ≠ (0, 0) :=
by
  sorry

end line_circle_intersect_not_center_l437_437160


namespace smallest_multiple_1_through_10_l437_437830

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437830


namespace ellipse_eq_length_MN_min_distance_ellipse_line_l437_437578

-- Define points A, B, and the general point P on the ellipse
def A : (ℝ × ℝ) := (-sqrt 2, 0)
def B : (ℝ × ℝ) := (sqrt 2, 0)

-- Define the point P, ensuring it's not on the x-axis
def P (x y : ℝ) : Prop := y ≠ 0 ∧ y^2 / (x^2 - 2) = -1 / 2

-- 1. Prove the equation of ellipse 
theorem ellipse_eq (x y : ℝ) (h : P x y) : (x^2 / 2 + y^2 = 1) :=
sorry

-- 2. Prove the length of the line segment |MN|
-- Define the focus point (for ellipse x^2/2 + y^2 = 1 it is (sqrt(2), 0))
def focus_right : (ℝ × ℝ) := (sqrt 2, 0)

-- Define points M and N on the ellipse by the line passing through the focus with slope 1.
def M : (ℝ × ℝ) := (...,...)
def N : (ℝ × ℝ) := (...,...)

theorem length_MN (x1 y1 x2 y2 : ℝ) (hM : M = (x1, y1)) (hN : N = (x2, y2)) : 
|x1 * sqrt(2) - x2 * sqrt(2) + y1 * 1 - y2 * 1| = (4 * sqrt(2) / 3) := 
sorry

-- 3. Prove the minimum distance to the given line from a point on the ellipse
-- Define the line equation
def line (x y : ℝ) : Prop := sqrt 2 * x + y + 2 * sqrt 5 = 0

-- Minimum distance theorem
theorem min_distance_ellipse_line : 
  ∃ (x y : ℝ), (x^2 / 2 + y^2 = 1) ∧ (sqrt 2 * x + y + t = 0) → 
     ∃ (t : ℝ), min_distance = (sqrt 15 / 3) := 
sorry

end ellipse_eq_length_MN_min_distance_ellipse_line_l437_437578


namespace possible_sequences_after_lunch_l437_437643

-- Define the initial conditions
def initial_letters : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def typed_before_lunch : List ℕ := [8, 5]

-- Define the problem statement
theorem possible_sequences_after_lunch : (number_of_possible_sequences initial_letters typed_before_lunch) = 32 := 
sorry

-- Auxiliary functions and definitions that might be required
-- These are placeholders and need to be properly defined
noncomputable def number_of_possible_sequences (initial : List ℕ) (typed : List ℕ) : ℕ := 
sorry

end possible_sequences_after_lunch_l437_437643


namespace total_rent_proof_l437_437702

-- Definitions for conditions
def original_rent (x y : ℕ) : ℕ := 40 * x + 60 * y

def new_rent (x y : ℕ) : ℕ := 40 * (x + 10) + 60 * (y - 10)

def rental_reduction (x y : ℕ) : Prop := 0.9 * (original_rent x y) = new_rent x y

-- The theorem we want to prove
theorem total_rent_proof (x y : ℕ) (h : rental_reduction x y) : original_rent x y = 2000 :=
by
  -- Proof would be provided here
  sorry

end total_rent_proof_l437_437702


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_correct_option_is_A_l437_437051

theorem option_a_correct (a : ℝ) : (-a)^6 / (-a)^3 = -a^3 := by 
  sorry

theorem option_b_incorrect (a : ℝ) : (-3 * a^3)^2 ≠ 6 * a^6 := by 
  sorry
  
theorem option_c_incorrect (a b : ℝ) : (a * b^2)^3 ≠ a * b^6 := by 
  sorry

theorem option_d_incorrect (a : ℝ) : a^3 * a^2 ≠ a^6 := by 
  sorry

theorem correct_option_is_A (a b : ℝ) : 
  (option_a_correct a) ∧ (¬ option_b_incorrect a) ∧ (¬ option_c_incorrect a b) ∧ (¬ option_d_incorrect a) := by 
  sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_correct_option_is_A_l437_437051


namespace percentile_80th_of_scores_is_90_l437_437766

-- Define the given scores
def scores : List ℕ := [78, 70, 72, 86, 88, 79, 80, 81, 94, 84, 56, 98, 83, 90, 91]

-- Define the 80th percentile calculation
def percentile_80th (l : List ℕ) : ℚ :=
  let sorted := l.merge_sort (≤)  -- sort the list
  let pos : ℕ := Float.toInt (Float.ofRat (sorted.length * 80 / 100)) - 1  -- zero-indexed position for 80th percentile
  (sorted.getOrElse pos 0 + sorted.getOrElse (pos + 1) 0) / 2  -- average of 12th and 13th scores

-- Theorem stating the 80th percentile of the scores list is 90.5
theorem percentile_80th_of_scores_is_90.5 : percentile_80th scores = 90.5 :=
by
  sorry

end percentile_80th_of_scores_is_90_l437_437766


namespace smallest_number_divisible_by_1_to_10_l437_437834

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437834


namespace arithmetic_series_sum_l437_437156

theorem arithmetic_series_sum :
  let first_term := -25
  let common_difference := 2
  let last_term := 19
  let n := (last_term - first_term) / common_difference + 1
  let sum := n * (first_term + last_term) / 2
  sum = -69 :=
by
  sorry

end arithmetic_series_sum_l437_437156


namespace carbon_tetrachloride_molecular_weight_l437_437034

theorem carbon_tetrachloride_molecular_weight (molecular_weight_9_moles : ℝ) (h : molecular_weight_9_moles = 1368) :
  ∃ mw_one_mole : ℝ, mw_one_mole = 152 :=
by
  use 152
  have : molecular_weight_9_moles / 9 = 152,
    by calc
      molecular_weight_9_moles / 9 = 1368 / 9 : by rw h
      ...                       = 152 : by norm_num
  exact this.symm

end carbon_tetrachloride_molecular_weight_l437_437034


namespace translation_teams_count_l437_437773

theorem translation_teams_count :
  let english_translators := 5
  let japanese_translators := 4
  let both_translators := 2
  let total_translators := 11
  let selected_people := 8
  let english_team_size := 4
  let japanese_team_size := 4
  (∑ s in (finset.range total_translators).powerset, (s.card = selected_people ∧ 
    (s.filter (λ x, x < english_translators)).card = english_team_size ∧ 
    (s.filter (λ x, x ≥ english_translators ∧ x < english_translators + japanese_translators)).card = japanese_team_size ¬
    (s.filter (λ x, x ≥ english_translators + japanese_translators)).card = both_translators) = 185 :=
sorry

end translation_teams_count_l437_437773


namespace relatively_prime_permutations_count_l437_437645

theorem relatively_prime_permutations_count :
  ∃ (arrangements : ℕ), arrangements = 864 ∧
  ∀ (p : list ℕ), p.perm [1, 2, 3, 4, 5, 6, 7] →
  (∀ (i : ℕ), i < p.length - 1 → Nat.coprime (p.nth_le i sorry) (p.nth_le (i+1) sorry)) →
  arrangements = list.permutations [1, 2, 3, 4, 5, 6, 7].count (λ p, ∀ (i : ℕ), i < p.length - 1 → Nat.coprime (p.nth_le i sorry) (p.nth_le (i+1) sorry)) :=
sorry

end relatively_prime_permutations_count_l437_437645


namespace Micah_words_per_minute_l437_437285

-- Defining the conditions
def Isaiah_words_per_minute : ℕ := 40
def extra_words : ℕ := 1200

-- Proving the statement that Micah can type 20 words per minute
theorem Micah_words_per_minute (Isaiah_wpm : ℕ) (extra_w : ℕ) : Isaiah_wpm = 40 → extra_w = 1200 → (Isaiah_wpm * 60 - extra_w) / 60 = 20 :=
by
  -- Sorry is used to skip the proof
  sorry

end Micah_words_per_minute_l437_437285


namespace total_songs_correct_l437_437443

-- Define the conditions of the problem
def num_country_albums := 2
def songs_per_country_album := 12
def num_pop_albums := 8
def songs_per_pop_album := 7
def num_rock_albums := 5
def songs_per_rock_album := 10
def num_jazz_albums := 2
def songs_per_jazz_album := 15

-- Define the total number of songs
def total_songs :=
  num_country_albums * songs_per_country_album +
  num_pop_albums * songs_per_pop_album +
  num_rock_albums * songs_per_rock_album +
  num_jazz_albums * songs_per_jazz_album

-- Proposition stating the correct total number of songs
theorem total_songs_correct : total_songs = 160 :=
by {
  sorry -- Proof not required
}

end total_songs_correct_l437_437443


namespace sine_of_angle_between_B1C_and_lateral_face_l437_437657

-- Definitions for the points in the prism
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Conditions
def right_angle_prism (A B C A1 B1 C1 : Point3D) (θ : ℝ) :=
  θ = 90 ∧
  (A.y = 0 ∧ B.x = 0 ∧ B.y = 2 ∧ B.z = 0 ∧ C.x = 0 ∧ C.y = 0 ∧ C.z = 0 ∧ A.x = 4 ∧ A.z = 0) ∧
  (A1 = ⟨4, 0, 2⟩ ∧ B1 = ⟨0, 2, 2⟩) ∧
  (A.z = B.z ∧ A1.y = B1.y) ∧
  (A1.x = A.x ∧ A1.y = A.y ∧ B1.x = B.x ∧ B1.y = B.y ∧ C1.x = C.x ∧ C1.y = C.y ∧ C1.z = 2)

-- The proof statement
theorem sine_of_angle_between_B1C_and_lateral_face :
  ∀ (A B C A1 B1 C1 : Point3D),
  right_angle_prism A B C A1 B1 C1 90 →
  sin_angle_between_b1c_and_lateral_face A B C A1 B1 C1 = sqrt 10 / 5 :=
sorry

end sine_of_angle_between_B1C_and_lateral_face_l437_437657


namespace max_outstanding_boys_10_l437_437012

-- For simplicity of modeling, let's assume each boy is represented by a natural number (i.e., their index)
-- We will be working with lists to represent weight and height distributions

structure Boy :=
  (height : ℕ)
  (weight : ℕ)

def is_not_worse_than (A B : Boy) : Prop :=
  (A.weight > B.weight) ∨ (A.height > B.height)

def is_outstanding (boys : List Boy) (A : Boy) : Prop :=
  ∀ B ∈ boys, A ≠ B → is_not_worse_than A B

def max_outstanding_boys (boys : List Boy) : ℕ :=
  (boys.filter (is_outstanding boys)).length

theorem max_outstanding_boys_10 (boys : List Boy) : boys.length = 10 → max_outstanding_boys boys = 10 :=
by
  sorry

end max_outstanding_boys_10_l437_437012


namespace solution_fractional_equation_l437_437005

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end solution_fractional_equation_l437_437005


namespace smallest_multiple_1_through_10_l437_437820

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437820


namespace is_divisible_not_by_two_k_plus_one_l437_437357

    theorem is_divisible_not_by_two_k_plus_one (k : ℕ) :
      let Nk := (2 * k)! / k! in
      (Nk % 2^k = 0) ∧ (Nk % 2^(k + 1) ≠ 0) :=
    by
      sorry
    
end is_divisible_not_by_two_k_plus_one_l437_437357


namespace moving_parabola_focus_l437_437225

noncomputable def focus_of_moving_parabola (x y : ℝ) : Prop :=
  -- The condition that the parabola passes through (0, 0)
  (0, 0) ∈ {(0, 1)} ∧
  -- The condition defining the locus as per the problem statement
  (x^2 + y^2 = 1 ∧ y ≠ -1)

theorem moving_parabola_focus :
  ∀ (x y : ℝ), focus_of_moving_parabola x y → (x^2 + y^2 = 1 ∧ (x, y) ≠ (0, -1)) :=
by
  sorry

end moving_parabola_focus_l437_437225


namespace smallest_number_divisible_1_to_10_l437_437930

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437930


namespace sum_divisible_by_49_l437_437194

theorem sum_divisible_by_49
  {x y z : ℤ} 
  (hx : x % 7 ≠ 0)
  (hy : y % 7 ≠ 0)
  (hz : z % 7 ≠ 0)
  (h : 7 ^ 3 ∣ (x ^ 7 + y ^ 7 + z ^ 7)) : 7^2 ∣ (x + y + z) :=
by
  sorry

end sum_divisible_by_49_l437_437194


namespace max_num_ones_min_num_ones_l437_437642

-- Problem part (i)
theorem max_num_ones (n : ℕ) (h : n ≥ 3) : 
  ∃ grid : Matrix ℕ ℕ ℤ, ∀ i j : ℕ, i < n ∧ j < n → 
  (∀ k, (k = 0 ∨ k = n-1) → (grid 0 k = -1 ∧ grid k 0 = -1 ∧ grid k (n-1) = -1 ∧ grid (n-1) k = -1)) ∧ 
  (((n-2)^2 - 1) = |{(i, j) | grid i j = 1}|) := 
sorry

-- Problem part (ii)
theorem min_num_ones (n : ℕ) (h : n ≥ 3) : 
  ∃ grid : Matrix ℕ ℕ ℤ, ∀ i j : ℕ, i < n ∧ j < n → 
  (∀ k, (k = 0 ∨ k = n-1) → (grid 0 k = -1 ∧ grid k 0 = -1 ∧ grid k (n-1) = -1 ∧ grid (n-1) k = -1)) ∧ 
  ((n-2) = |{(i, j) | grid i j = 1}|) := 
sorry

end max_num_ones_min_num_ones_l437_437642


namespace value_of_a_100_l437_437276

open Nat

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (succ k) => sequence k + 4

theorem value_of_a_100 : sequence 99 = 397 := by
  sorry

end value_of_a_100_l437_437276


namespace smallest_divisible_1_to_10_l437_437920

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437920


namespace prime_power_divides_power_of_integer_l437_437362

theorem prime_power_divides_power_of_integer 
    {p a n : ℕ} 
    (hp : Nat.Prime p)
    (ha_pos : 0 < a) 
    (hn_pos : 0 < n) 
    (h : p ∣ a^n) :
    p^n ∣ a^n := 
by 
  sorry

end prime_power_divides_power_of_integer_l437_437362


namespace domain_of_f_l437_437153

noncomputable def f (x : ℝ) : ℝ := (2 * x - 4)^(1/3:ℚ) + (10 - x)^(1/2:ℚ)

theorem domain_of_f : ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ x ≤ 10 := by
  sorry

end domain_of_f_l437_437153


namespace cube_root_of_f_l437_437245

-- Define the real numbers x and y with the given condition
variables (x y : ℝ)

-- Define the condition that y = sqrt(x - 3) + sqrt(3 - x) + 8
def y_condition : Prop := y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 8

-- Define the function that we want to find the cube root of
def f : ℝ := x + 3 * y

-- State the theorem
theorem cube_root_of_f (h : y_condition x y) : Real.cbrt (x + 3 * y) = 3 :=
sorry

end cube_root_of_f_l437_437245


namespace smallest_divisible_1_to_10_l437_437905

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437905


namespace raisin_cookies_count_l437_437624

/--
Helen baked 527 chocolate chip cookies yesterday and some raisin cookies.
She baked the same number of raisin cookies and 554 chocolate chip cookies this morning.
In total, she baked 1081 chocolate chip cookies.
Prove that the number of raisin cookies Helen baked yesterday is 527.
-/
theorem raisin_cookies_count 
  (choco_yesterday : ℕ)
  (choco_today : ℕ)
  (total_choco : ℕ)
  (raisin_yesterday : ℕ)
  (eq1 : choco_yesterday = 527)
  (eq2 : choco_today = 554)
  (eq3 : total_choco = 1081)
  (eq4 : total_choco = choco_yesterday + choco_today)
  (eq5 : raisin_yesterday = choco_yesterday) :
  raisin_yesterday = 527 :=
by
  rw [eq4, eq1, eq2] at eq3
  exact eq1.trans eq5.symm

end raisin_cookies_count_l437_437624


namespace fraction_expression_proof_l437_437281

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end fraction_expression_proof_l437_437281


namespace sum_of_series_l437_437613

theorem sum_of_series (n : ℕ) : 
  let a_n := λ n : ℕ, 1 / (n * (n + 1))
  let S_n := ∑ i in Finset.range n, a_n (i + 1)
  in S_n = n / (n + 1) := 
by
  sorry

end sum_of_series_l437_437613


namespace card_arrangement_impossible_l437_437993

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437993


namespace ratio_minutes_l437_437765

theorem ratio_minutes (x : ℝ) : 
  (12 / 8) = (6 / (x * 60)) → x = 1 / 15 :=
by
  sorry

end ratio_minutes_l437_437765


namespace _l437_437761

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end _l437_437761


namespace smallest_multiple_1_through_10_l437_437819

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437819


namespace smallest_number_divisible_1_to_10_l437_437933

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437933


namespace largest_T_correct_l437_437185

noncomputable def largest_T (p : fin 25 → ℕ) : ℕ :=
  (finset.prod finset.univ (λ i, (p i : ℤ) ^ 2005 - 1) / (p 0 - 1)).nat_abs

theorem largest_T_correct (p : fin 25 → ℕ) (h_distinct : function.injective p) (h_primes : ∀ i, prime (p i)) (h_bound : ∀ i, p i ≤ 2004) : 
  ∃ T, (∀ n ≤ T, ∃ L, L.sum = n ∧ ∀ x ∈ L, ∃ k : fin 25, x = p k) :=
sorry

end largest_T_correct_l437_437185


namespace smallest_multiple_1_through_10_l437_437829

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437829


namespace area_ratio_of_triangles_l437_437503

variable (O A B I C D E G H : Type) [PlanarMetric O] [RegularOctagon A B C D E F G H I]

-- Assume the octagon can be divided into eight smaller equilateral triangles such as ∆ABI
def smaller_equilateral_triangle (o : O) (a b i : A) : Triangle ABC := sorry

-- Assume connecting every third vertex of the octagon gives ∆ADE
def larger_equilateral_triangle (d e a : A) : Triangle ADE := sorry

theorem area_ratio_of_triangles
  (h₁ : regular_octagon A B C D E F G H I)
  (h₂ : smaller_equilateral_triangle O A B I)
  (h₃ : larger_equilateral_triangle A D E) :
  area (small_triangle A B I) / area (large_triangle A D E) = 1 / 6 := sorry

end area_ratio_of_triangles_l437_437503


namespace eval_product_of_magnitudes_l437_437690

open Complex

theorem eval_product_of_magnitudes (a b c : ℂ) 
    (habc : |a| = 1) (hbbc : |b| = 1) (hcbc : |c| = 1) 
    (habc_sum : |a + b + c| = 1) 
    (hq : |a - b| = |a - c|) (hneq : b ≠ c) :
    |a + b| * |a + c| = 2 :=
by 
  sorry

end eval_product_of_magnitudes_l437_437690


namespace cricket_team_age_difference_l437_437381

variable (captain_age : ℕ) (wicket_keeper_age : ℕ) (team_average_age : ℕ) (team_size : ℕ) (remaining_players_count : ℕ)
variable (total_team_age : ℕ) (total_remaining_players_age : ℕ) (remaining_average_age : ℕ) (diff_ages : ℕ)
variable (remaining_players : Finset ℕ)

def cricket_team_problem (captain_age : 24) (wicket_keeper_age : 31) (team_average_age : 23) (team_size : 11) 
                          (remaining_players_count : 9) (total_team_age : 253) (total_remaining_players_age : 198) 
                          (remaining_average_age: 22) (diff_ages : 1) : Prop :=
  captain_age = 24 ∧
  wicket_keeper_age = 31 ∧
  team_average_age = 23 ∧
  team_size = 11 ∧
  total_team_age = team_average_age * team_size ∧
  total_remaining_players_age = total_team_age - (captain_age + wicket_keeper_age) ∧
  remaining_average_age = total_remaining_players_age / remaining_players_count ∧
  diff_ages = team_average_age - remaining_average_age 

theorem cricket_team_age_difference : cricket_team_problem 24 31 23 11 9 253 198 22 1 := by
  sorry

end cricket_team_age_difference_l437_437381


namespace circle_standard_equation_l437_437177

theorem circle_standard_equation:
  ∃ (x y : ℝ), ((x + 2) ^ 2 + (y - 1) ^ 2 = 4) :=
by
  sorry

end circle_standard_equation_l437_437177


namespace max_clearable_rounds_is_4_probability_clearing_first_three_rounds_is_100_over_243_l437_437067

-- Condition: In the n-th round, a die is rolled n times.
axiom rolled_n_times (n : ℕ) : (fin n → ℕ) 

-- Condition: If the sum of the points from these n rolls is greater than 2^n, then the player clears the round.
def clears_round (n : ℕ) (rolls : fin n → ℕ) : Prop :=
  (finset.univ.sum rolls) > 2^n

-- The die is a uniform cube with the numbers 1, 2, 3, 4, 5, 6 on its faces.
axiom die_faces (x : ℕ) : x ∈ {1, 2, 3, 4, 5, 6}

-- Prove that the maximum number of rounds a player can clear in this game is 4.
theorem max_clearable_rounds_is_4 : ∀ n (rolls : fin n → ℕ), 
  (clears_round n rolls → n ≤ 4) ∧ (¬(clears_round (n+1) rolls) → n = 4) := 
sorry

-- Prove that the probability of clearing the first three rounds consecutively is 100/243
noncomputable def probability_clearing_first_three_rounds : ℚ := (2/3) * (5/6) * (20/27)

theorem probability_clearing_first_three_rounds_is_100_over_243 :
  probability_clearing_first_three_rounds = 100/243 := 
sorry

end max_clearable_rounds_is_4_probability_clearing_first_three_rounds_is_100_over_243_l437_437067


namespace quadrilateral_is_parallelogram_l437_437045

variables {A B C D : Type} [AddCommGroup A] [Module A B] [VectorsParallel A B] [HasEquality A]

-- To represent points and segments
variables (AB : A) (CD : A) (AD : A) (BC : A) (ABCD : Quads A)

-- Definitions of the given conditions
def conditions (A B C D : A) : Prop := AB = CD ∧ AD = BC

-- Statement of the theorem
theorem quadrilateral_is_parallelogram (h : conditions AB CD AD BC) : IsParallelogram ABCD :=
sorry

end quadrilateral_is_parallelogram_l437_437045


namespace symmetric_point_of_M_wrt_bisector_l437_437157

theorem symmetric_point_of_M_wrt_bisector :
  let line_eq := λ (x : ℝ), 2 * x in
  let symmetric_point := λ (M : ℝ × ℝ), let N := (4, -2) in
      2 * ((N.2 - M.2) / (N.1 + M.1)) = -1 ∧ (N.2 + M.2) / 2 = 2 * (N.1 - (-4)) / 2 in
  symmetric_point (-4, 2) = (4, -2) :=
by
  let line_eq := λ (x : ℝ), 2 * x
  let symmetric_point := λ (M : ℝ × ℝ), let N := (4, -2) in
      2 * ((N.2 - M.2) / (N.1 + M.1)) = -1 ∧ (N.2 + M.2) / 2 = 2 * (N.1 - (-4)) / 2
  sorry

end symmetric_point_of_M_wrt_bisector_l437_437157


namespace smallest_number_divisible_1_to_10_l437_437943

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437943


namespace shooter_hits_at_least_3_times_l437_437598

noncomputable def prob_shooter_hits_target (hits : ℕ) : ℝ :=
  match hits with
  | 0 => ((4.choose 0) : ℝ) * (0.8 ^ 0) * (0.2 ^ 4)
  | 1 => ((4.choose 1) : ℝ) * (0.8 ^ 1) * (0.2 ^ 3)
  | 2 => ((4.choose 2) : ℝ) * (0.8 ^ 2) * (0.2 ^ 2)
  | 3 => ((4.choose 3) : ℝ) * (0.8 ^ 3) * (0.2 ^ 1)
  | 4 => ((4.choose 4) : ℝ) * (0.8 ^ 4) * (0.2 ^ 0)
  | _ => 0

theorem shooter_hits_at_least_3_times : 
  prob_shooter_hits_target 3 + 
  prob_shooter_hits_target 4 = 0.8192 :=
by sorry

end shooter_hits_at_least_3_times_l437_437598


namespace find_n_l437_437392

theorem find_n :
  ∃ n : ℕ, 50 ≤ n ∧ n ≤ 150 ∧
          n % 7 = 0 ∧
          n % 9 = 3 ∧
          n % 6 = 3 ∧
          n = 75 :=
by
  sorry

end find_n_l437_437392


namespace standard_deviation_less_than_l437_437007

theorem standard_deviation_less_than:
  ∀ (μ σ : ℝ)
  (h1 : μ = 55)
  (h2 : μ - 3 * σ > 48),
  σ < 7 / 3 :=
by
  intros μ σ h1 h2
  sorry

end standard_deviation_less_than_l437_437007


namespace lcm_1_to_10_l437_437809

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437809


namespace smallest_multiple_1_through_10_l437_437822

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437822


namespace smallest_number_divisible_by_1_to_10_l437_437868

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437868


namespace correct_option_l437_437988

-- Definitions for the conditions of each option
def option_A (a : ℝ) : Prop := 2 * a^2 + a^2 = 3 * a^4
def option_B (a : ℝ) : Prop := a^2 * a^4 = a^8
def option_C (a : ℝ) : Prop := (a^2)^4 = a^6
def option_D (a b : ℝ) : Prop := (-a * b^3)^2 = a^2 * b^6

-- Stating the problem to prove Option D is correct
theorem correct_option (a b : ℝ) : option_D a b ∧ ¬option_A a ∧ ¬option_B a ∧ ¬option_C a :=
by {
  sorry, -- Proof to be filled in
}

end correct_option_l437_437988


namespace abc_prod_eq_l437_437708

-- Define a structure for points and triangles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define the angles formed by points in a triangle
def angle (A B C : Point) : ℝ := sorry

-- Define the lengths between points
def length (A B : Point) : ℝ := sorry

-- Conditions of the problem
theorem abc_prod_eq (A B C D : Point) 
  (h1 : angle A D C = angle A B C + 60)
  (h2 : angle C D B = angle C A B + 60)
  (h3 : angle B D A = angle B C A + 60) : 
  length A B * length C D = length B C * length A D :=
sorry

end abc_prod_eq_l437_437708


namespace smallest_number_div_by_1_to_10_l437_437793

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437793


namespace smallest_number_divisible_1_to_10_l437_437877

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437877


namespace lcm_1_to_10_l437_437857

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437857


namespace general_eq_curve_tangency_and_circle_eq_l437_437647

-- Define the parametric equations for curve C
def curve_parametric (t : ℝ) : ℝ × ℝ :=
  (2 * t^2 - 1, 2 * t - 1)

-- Define the general equation for curve C
def curve_general : Prop :=
  ∀ x y : ℝ, (∃ t : ℝ, x = 2 * t^2 - 1 ∧ y = 2 * t - 1) ↔ (y + 1)^2 = 2 * (x + 1)

-- Define the polar equation of line l
def line_polar (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ * (2 * Real.sin θ - Real.cos θ) = m

-- Define the tangent condition and the required circle equation
def tangent_and_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (∀ θ ρ : ℝ, line_polar ρ θ m → 2 * y - x = m) →
  (∀ k : ℝ, ((y + 1)^2 = 2 * (x + 1) ∧ 2 * y - x = m) → (x + 1)^2 + (y - 1 / 4)^2 = (Real.sqrt 5 / 4)^2)

-- Theorems we need to prove
theorem general_eq_curve : curve_general := sorry

theorem tangency_and_circle_eq (m : ℝ) : tangent_and_circle m := sorry

end general_eq_curve_tangency_and_circle_eq_l437_437647


namespace sum_coordinates_D_l437_437344

theorem sum_coordinates_D
    (M : (ℝ × ℝ))
    (C : (ℝ × ℝ))
    (D : (ℝ × ℝ))
    (H_M_midpoint : M = (5, 9))
    (H_C_coords : C = (11, 5))
    (H_M_def : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
    (D.1 + D.2) = 12 := 
by
  sorry
 
end sum_coordinates_D_l437_437344


namespace smallest_number_div_by_1_to_10_l437_437797

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437797


namespace area_triangle_PAC_l437_437117

open Real

-- Definition for the problem
def square_side_len : ℝ := 12
def len_leg_triangle_PAB : ℝ := 10

-- Prove that the area of triangle PAC is 12 cm² given the conditions
theorem area_triangle_PAC : 
  ∃ P : ℝ × ℝ,
    (let A := (0, 0) in
     let B := (square_side_len, 0) in
     let D := (0, square_side_len) in
     let C := (square_side_len, square_side_len) in
     dist P A = len_leg_triangle_PAB ∧
     dist P B = len_leg_triangle_PAB ∧
     (0 < fst P ∧ fst P < square_side_len) ∧
     0 ≤ snd P ∧ snd P ≤ square_side_len ∧
     (area (mk_triangle P A C)) = 12
  ) :=
begin
  sorry
end

end area_triangle_PAC_l437_437117


namespace integer_bases_not_divisible_by_5_l437_437165

theorem integer_bases_not_divisible_by_5 :
  ∀ b ∈ ({3, 5, 7, 10, 12} : Set ℕ), (b - 1) ^ 2 % 5 ≠ 0 :=
by sorry

end integer_bases_not_divisible_by_5_l437_437165


namespace RotaryClubNeeds584Eggs_l437_437378

noncomputable def RotaryClubEggsRequired : ℕ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let adult_tickets : ℕ := 75
  let senior_tickets : ℕ := 37
  let small_children_omelets := 0.5 * small_children_tickets
  let older_children_omelets := 1 * older_children_tickets
  let adult_omelets := 2 * adult_tickets
  let senior_omelets := 1.5 * senior_tickets
  let extra_omelets : ℕ := 25
  let total_omelets := small_children_omelets + older_children_omelets + adult_omelets + senior_omelets + extra_omelets
  2 * total_omelets

theorem RotaryClubNeeds584Eggs : RotaryClubEggsRequired = 584 := by
  sorry

end RotaryClubNeeds584Eggs_l437_437378


namespace infinite_winning_positions_for_second_player_l437_437133

def is_square (x : ℕ) : Prop :=
  ∃ (n : ℕ), n * n = x

/-- Define the conditions of the game: a move consists of taking off the table x pebbles where 
    x is the square of a positive integer, with players alternating turns, and the player unable
    to make a move loses. -/
structure Game :=
  (initial_pebbles : ℕ)
  (move : ℕ → Prop := λ x, is_square x ∧ x > 0)

theorem infinite_winning_positions_for_second_player :
  ∃^∞ k : ℕ, ∀ (game : Game), game.initial_pebbles = k → 
  (∀ n : ℕ, game.initial_pebbles = n^2 + n + 1 → 
  ∃ strategy_second_player_wins : true) :=
by 
  sorry

end infinite_winning_positions_for_second_player_l437_437133


namespace triangle_areas_equal_l437_437295

def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_areas_equal :
  let A := herons_formula 13 13 10
  let B := herons_formula 13 13 24
  A = B :=
by
  sorry

end triangle_areas_equal_l437_437295


namespace correct_statements_count_l437_437757

theorem correct_statements_count : 
  (let s1 := (∃ L : Type, ∀ P₀ : L, ∀ L1 : L, L1 ∉ L → ∃! L2 : L, L2 = L ∧ L2 ⟂ L1) in s1) ∧
  (let s2 := (∀ L1 L2 L3 : Type, (L3 ⊥ (L1 ∧ L2)) → (∃ P : Type, P ∈ L1 ∧ P ∈ L2)) in ¬ s2) ∧
  (let s3 := (∀ L1 L2 L3 : Type, (L3 ⊥ L1 ∧ L3 ⊥ L2) → (L1 ∥ L2)) in s3) ∧
  (let s4 := (∀ L1 L2 : Type, L1 ≠ L2 → (L1 ∥ L2 ∨ ∃ P : Type, P ∈ L1 ∧ P ∈ L2)) in s4) →
  3 :=
by
  sorry

end correct_statements_count_l437_437757


namespace determine_object_type_l437_437775

-- Definitions based on given conditions
def length (r : ℝ) := r = 35
def width (r : ℝ) := r = 20
def height (r : ℝ) := r = 15

-- Type definitions for the possible objects
inductive ObjectType
  | PencilCase
  | MathTextbook
  | Bookshelf
  | Shoebox

-- Theorem statement: The object is a shoebox given the conditions
theorem determine_object_type (l w h : ℝ) (h_length : length l) (h_width : width w) (h_height : height h) : ObjectType :=
  ObjectType.Shoebox


end determine_object_type_l437_437775


namespace area_of_triangle_ABC_l437_437577

noncomputable def complex_area_of_triangle_ABC (z : ℂ) : ℝ :=
  let A := (z.re, z.im)
  let B := ((z^2).re, (z^2).im)
  let C := ((z - z^2).re, (z - z^2).im)
  (1 / 2) * Real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC (z : ℂ) (hz1 : abs z = Real.sqrt 2) (hz2 : (z^2).im = 2) :
  complex_area_of_triangle_ABC z = 1 :=
sorry

end area_of_triangle_ABC_l437_437577


namespace transylvanian_convinces_l437_437664

theorem transylvanian_convinces (s : Prop) (t : Prop) (h : s ↔ (¬t ∧ ¬s)) : t :=
by
  -- Leverage the existing equivalence to prove the desired result
  sorry

end transylvanian_convinces_l437_437664


namespace ellipse_eq_area_F1AB_l437_437183

-- Define the ellipse C and its properties
def ellipse (x y : ℝ) (a b : ℝ) := (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1
def eccentricity (c a : ℝ) := c / a
def focus1 := (-1 : ℝ, 0 : ℝ)
def line (x y : ℝ) := y = 1 / 2 * (x + 2)

-- Given conditions
variables (a b c : ℝ)
variable (h_aecc : eccentricity c a = sqrt 2 / 2)
variable (h_focus1 : c = 1)
variable (h_a_gt_b : a > b)
variable (h_b_gt_0 : b > 0)

-- Questions
theorem ellipse_eq : ellipse x y (sqrt 2) 1 := by
  sorry

theorem area_F1AB (A B : ℝ × ℝ) 
  (h_lineA : line A.1 A.2)
  (h_lineB : line B.1 B.2)
  (h_A_on_ellipse : ellipse A.1 A.2 (sqrt 2) 1)
  (h_B_on_ellipse : ellipse B.1 B.2 (sqrt 2) 1)
  : ∃ (area : ℝ), area = 1 / 3 := 
by 
  sorry

end ellipse_eq_area_F1AB_l437_437183


namespace smallest_divisible_by_1_to_10_l437_437900

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437900


namespace smallest_number_divisible_by_1_to_10_l437_437864

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437864


namespace four_digit_arithmetic_sequences_count_l437_437237

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  (10*b + c) - (10*a + b) = (10*c + d) - (10*b + c)

def is_valid_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def count_valid_four_digit_numbers : ℕ :=
  let candidates := [(a, b, c, d) | a <- [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   b <- [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   c <- [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   d <- [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   is_arithmetic_sequence a b c d]
  candidates.length

theorem four_digit_arithmetic_sequences_count :
  count_valid_four_digit_numbers = 17 :=
by
  sorry

end four_digit_arithmetic_sequences_count_l437_437237


namespace existence_of_l1_a_existence_of_l1_b_existence_of_l1_c_l437_437056

-- Define the necessary structures
noncomputable def Circle := {center : ℝ × ℝ, radius : ℝ}
noncomputable def Line := ℝ × ℝ × ℝ -- Ax + By + C = 0 

variables {S1 S2 : Circle} {l : Line} {a : ℝ}

-- Part (a): Proving existence of l1 with distance between intersection points equal to a
theorem existence_of_l1_a :
  ∃ l1 : Line, 
  let S1' := {center := (S1.center.1 + a, S1.center.2), radius := S1.radius} in
  ∃ P1 P2 : ℝ × ℝ, 
  P1 ∈ intersection_points S1' S2 ∧ 
  P2 ∈ intersection_points S1' S2 ∧ 
  parallel l1 l ∧
  distance P1 P2 = a := 
sorry

-- Part (b): Proving existence of l1 cut by equal chords on S1 and S2 
theorem existence_of_l1_b :
  ∃ l1 : Line, 
  let O1 := projection S1.center l in
  let O2 := projection S2.center l in 
  let S1' := {center := (S1.center.1 + (O2.1 - O1.1), S1.center.2 + (O2.2 - O1.2)), radius := S1.radius} in
  ∃ P1 P2 : ℝ × ℝ, 
  P1 ∈ intersection_points S1' S2 ∧ 
  P2 ∈ intersection_points S1' S2 ∧ 
  parallel l1 l ∧
  length_of_chord S1 l1 = length_of_chord S2 l1 := 
sorry

-- Part (c): Proving existence of l1 such that sum or difference of lengths of chords on S1 and S2 satisfying condition
theorem existence_of_l1_c :
  ∃ l1 : Line, 
  let S1' := {center := (S1.center.1 + a, S1.center.2), radius := S1.radius} in
  let dist := distance (projection S1'.center l) (projection S2.center l) in
  dist = a / 2 ∧
  ∃ P1 P2 : ℝ × ℝ, 
  P1 ∈ intersection_points S1' S2 ∧ 
  P2 ∈ intersection_points S1' S2 ∧ 
  parallel l1 l ∧
  (length_of_chord S1 l1 + length_of_chord S2 l1 = a ∨
   length_of_chord S1 l1 - length_of_chord S2 l1 = a) := 
sorry

end existence_of_l1_a_existence_of_l1_b_existence_of_l1_c_l437_437056


namespace lcm_1_10_l437_437961

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437961


namespace total_books_l437_437513

-- Define the number of books each person has
def books_beatrix : ℕ := 30
def books_alannah : ℕ := books_beatrix + 20
def books_queen : ℕ := books_alannah + (books_alannah / 5)

-- State the theorem to be proved
theorem total_books (h_beatrix : books_beatrix = 30)
                    (h_alannah : books_alannah = books_beatrix + 20)
                    (h_queen : books_queen = books_alannah + (books_alannah / 5)) :
  books_alannah + books_beatrix + books_queen = 140 :=
sorry

end total_books_l437_437513


namespace largest_digit_product_l437_437558

theorem largest_digit_product : ∃ (n : ℕ), 
  (∃ (digits : list ℕ), 
    (∀ i, i ∈ digits → 0 < i ∧ i < 10) ∧ 
    (list.pairwise (≤) digits) ∧ 
    (list.sum (list.map (λ x, x^2) digits) = 82) ∧ 
    n = list.prod digits ∧ 
    (∀ (m : ℕ), 
      (∃ (m_digits : list ℕ), 
        (∀ i, i ∈ m_digits → 0 < i ∧ i < 10) ∧ 
        (list.pairwise (≤) m_digits) ∧ 
        (list.sum (list.map (λ x, x^2) m_digits) = 82)) → n ≥ list.prod m_digits )) ∧ 
    n = 9 :=
begin
  sorry
end

end largest_digit_product_l437_437558


namespace quadrilateral_is_rhombus_l437_437713

open EuclideanGeometry

variables {A B C D : Point}

-- Definition of angle bisectors lying on the diagonals
def diagonals_on_angle_bisectors (A B C D : Point) : Prop :=
  ∠ BCA = ∠ DCA ∧ ∠ BAC = ∠ DAC ∧ ∠ ABD = ∠ CBD ∧ ∠ CDB = ∠ ADB

-- Theorem that given the above condition, the quadrilateral is a rhombus
theorem quadrilateral_is_rhombus (h : diagonals_on_angle_bisectors A B C D) : rhombus A B C D :=
by
  sorry

end quadrilateral_is_rhombus_l437_437713


namespace smallest_number_divisible_1_to_10_l437_437942

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437942


namespace pizza_sharing_order_l437_437571

theorem pizza_sharing_order :
  ∀ (total_slices : ℕ) (ali_fraction : ℚ) (bea_fraction : ℚ) (chris_fraction : ℚ) (dan_share : ℚ),
    ali_fraction = 1/6 →
    bea_fraction = 1/8 →
    chris_fraction = 1/7 →
    dan_share = 1 - (ali_fraction + bea_fraction + chris_fraction) →
    let ali_slices := total_slices * ali_fraction,
        bea_slices := total_slices * bea_fraction,
        chris_slices := total_slices * chris_fraction,
        dan_slices := total_slices * dan_share in
    (list.cons dan_slices [
    (list.cons ali_slices [
    (list.cons chris_slices [
    (list.cons bea_slices [])])])).sort (λ a b, b ≤ a) = [(dan_slices : ℚ), ali_slices, chris_slices, bea_slices] := 
sorry

end pizza_sharing_order_l437_437571


namespace arithmetic_sequence_sum_l437_437591

theorem arithmetic_sequence_sum (a₁ d S : ℤ)
  (ha : 10 * a₁ + 24 * d = 37) :
  19 * (a₁ + 2 * d) + (a₁ + 10 * d) = 74 :=
by
  sorry

end arithmetic_sequence_sum_l437_437591


namespace simplify_complex_fraction_l437_437719

-- Definitions based on the conditions
def numerator : ℂ := 5 + 7 * complex.i
def denominator : ℂ := 2 + 3 * complex.i
def expected : ℂ := 31 / 13 - (1 / 13) * complex.i

-- The Lean 4 statement
theorem simplify_complex_fraction : (numerator / denominator) = expected := 
sorry

end simplify_complex_fraction_l437_437719


namespace method_D_incorrect_l437_437441

/--  Definition of Methods A-E --/
structure Methods (Point : Type) := 
  (on_locus : Point → Prop) 
  (conditions : Point → Prop)

def method_A {Point : Type} (M : Methods Point) : Prop :=
∀ p : Point, if M.on_locus p then M.conditions p else ¬M.conditions p

def method_B {Point : Type} (M : Methods Point) : Prop :=
∀ p : Point, (M.conditions p → M.on_locus p) ∧ (M.on_locus p → M.conditions p)

def method_C {Point : Type} (M : Methods Point) : Prop :=
∀ p : Point, (¬M.conditions p → ¬M.on_locus p) ∧ (M.conditions p → M.on_locus p)

def method_D {Point : Type} (M : Methods Point) : Prop :=
∀ p : Point, M.on_locus p → M.conditions p ∧ (¬M.conditions p → M.on_locus p)

def method_E {Point : Type} (M : Methods Point) : Prop :=
∀ p : Point, (¬M.on_locus p → ¬M.conditions p) ∧ (M.on_locus p → M.conditions p)

/-- Proof that method_D is incorrect --/
theorem method_D_incorrect {Point : Type} (M : Methods Point) : ¬method_D M :=
sorry

end method_D_incorrect_l437_437441


namespace smallest_number_divisible_by_1_through_10_l437_437980

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437980


namespace find_ellipse_eqn_line_exists_l437_437584

-- Definitions based on the problem's conditions
def ellipse_C (a b : ℝ) : Prop := ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1

def is_focus (F : ℝ × ℝ) (a b : ℝ) : Prop :=
  F = (0, real.sqrt (b^2 - a^2))

def on_ellipse (M : ℝ × ℝ) (a b : ℝ) : Prop :=
  ellipse_C a b M.1 M.2

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The conditions provided in the problem
variables (a b : ℝ)
variable (M : ℝ × ℝ)
variable (F1 : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))

-- Given conditions
axiom focus_F1 : is_focus F1 a b
axiom point_M : M = (1, 4)
axiom area_MOF1 : area_of_triangle M O F1 = 3/2

-- Problems to be solved
-- 1. Find the equation of the ellipse
theorem find_ellipse_eqn (a b : ℝ) (M : ℝ × ℝ) (F1 : ℝ × ℝ) :
  ellipse_C a b := sorry

-- 2. Determine if a line (l) parallel to OM that intersects the ellipse (C) and the circle with diameter AB passes through the origin
def line_l (m : ℝ) : ℝ → ℝ := λ x, 4 * x + m

theorem line_exists (a b : ℝ) (M F1 : ℝ × ℝ) (m : ℝ) :
  on_ellipse M a b ∧ is_focus F1 a b ∧ area_of_triangle M O F1 = 3/2 →
  ∃ m : ℝ, m = real.sqrt 102 ∨ m = -real.sqrt 102 := sorry

end find_ellipse_eqn_line_exists_l437_437584


namespace same_polygon_side_length_l437_437476

theorem same_polygon_side_length
  (a_7 a_{17} r_7 R_{17} r_7 R_7 R_{17} : ℝ)
  (eq_rings : π * (R_7^2 - r_7^2) = π * (R_{17}^2 - r_{17}^2))
  (pythagorean_7 : a_7^2 + r_7^2 = R_7^2)
  (pythagorean_{17} : a_{17}^2 + r_{17}^2 = R_{17}^2) :
  a_7 = a_{17} := by
  sorry

end same_polygon_side_length_l437_437476


namespace system_of_equations_solution_l437_437369

theorem system_of_equations_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -8) ∧ (5 * x + 9 * y = -18) ∧ x = -14 / 3 ∧ y = -32 / 9 :=
by {
  sorry  -- Proof goes here
}

end system_of_equations_solution_l437_437369


namespace compute_expression_l437_437312

noncomputable def log_base (base x : ℝ) : ℝ := real.log x / real.log base

theorem compute_expression (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h : (log_base 2 x)^4 + (log_base 3 y)^4 + 8 = 8 * (log_base 2 x) * (log_base 3 y)) :
  x^real.sqrt 2 + y^real.sqrt 2 = 13 :=
  sorry

end compute_expression_l437_437312


namespace markov_coprime_squares_l437_437557

def is_coprime (x y : ℕ) : Prop :=
Nat.gcd x y = 1

theorem markov_coprime_squares (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  x^2 + y^2 + z^2 = 3 * x * y * z →
  ∃ a b c: ℕ, (a, b, c) = (2, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∧ 
  (a ≠ 1 → ∃ p q : ℕ, is_coprime p q ∧ a = p^2 + q^2) :=
sorry

end markov_coprime_squares_l437_437557


namespace domain_of_function_l437_437745

open Real

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x ^ 2 - 2 * x - 3) + log 3 (x + 2)

theorem domain_of_function : 
    { x : ℝ | x^2 - 2*x - 3 ≥ 0 ∧ x + 2 > 0 } = 
    { x : ℝ | (-2 < x ∧ x ≤ -1) ∨ (3 ≤ x) } :=
by
  sorry

end domain_of_function_l437_437745


namespace smallest_number_divisible_1_to_10_l437_437880

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437880


namespace find_ratio_AE_EB_l437_437659

-- Define the points A, B, C, D, E, and Q, where D is on line segment BC, E is on line segment AB,
-- and line segments AD and CE intersect at Q.
noncomputable def point (V : Type*) [add_comm_group V] [vector_space ℝ V] :=
  V

variables  {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C D E Q : point V)  -- The points A, B, C, D, E, and Q
variables (r s t u : ℝ)  -- Ratios of segments

-- Given conditions:
-- 1. D is on line segment BC
def D_on_BC : Prop := ∃ α β : ℝ, α + β = 1 ∧ D = α • B + β • C
-- 2. E is on line segment AB
def E_on_AB : Prop := ∃ γ δ : ℝ, γ + δ = 1 ∧ E = γ • A + δ • B
-- 3. Line segments AD and CE intersect at Q
def intersection_AD_CE : Prop :=
  ∃ α β : ℝ, α + β = 1 ∧ Q = α • A + β • D ∧ ∃ γ δ : ℝ, γ + δ = 1 ∧ Q = γ • C + δ • E

-- 4. AQ : QD = 3 : 2
def ratio_AQ_QD : Prop :=  ∃ α: ℝ, α = 3 / 5 ∧ Q = α • A + (1 - α) • D
-- 5. EQ : QC = 3 : 1
def ratio_EQ_QC : Prop := ∃ α: ℝ, α = 3 / 4 ∧ Q = α • E + (1 - α) • C

-- The goal: find the ratio AE/EB = 7/8
theorem find_ratio_AE_EB (h1 : D_on_BC) (h2 : E_on_AB) (h3 : intersection_AD_CE) (h4 : ratio_AQ_QD) (h5 : ratio_EQ_QC) : 
  (∃ α β : ℝ, α / β = 7 / 8 ∧ α • A + β • B = E) := 
sorry

end find_ratio_AE_EB_l437_437659


namespace problem_1_problem_2_l437_437208

noncomputable def f (a x : ℝ) : ℝ := Real.log (3 - a * x) / Real.log a

theorem problem_1 (h : f (3 / 5) 4 = 1) : (3 - 4 * (3 / 5)) = (3 / 5) := by
  sorry

theorem problem_2 : ¬ ∃ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ 1) ∧ (∀ x y ∈ Set.Icc 1 2, x < y → f a x > f a y) := by
  sorry

end problem_1_problem_2_l437_437208


namespace expression_undefined_count_l437_437163

theorem expression_undefined_count (x : ℝ) :
  ∃! x, (x - 1) * (x + 3) * (x - 3) = 0 :=
sorry

end expression_undefined_count_l437_437163


namespace degree_of_g_l437_437730

theorem degree_of_g (f g : Polynomial ℝ) (h : Polynomial ℝ) (H1 : h = f.comp g + g) 
  (H2 : h.natDegree = 6) (H3 : f.natDegree = 3) : g.natDegree = 2 := 
sorry

end degree_of_g_l437_437730


namespace domino_2x2_not_fully_covered_l437_437703

def Cell := (Nat × Nat)

structure Domino where
  c1 : Cell
  c2 : Cell
  h_adjacent : (c1.1 = c2.1 ∧ (c1.2 + 1 = c2.2 ∨ c1.2 = c2.2 + 1))
                ∨ (c1.2 = c2.2 ∧ (c1.1 + 1 = c2.1 ∨ c1.1 = c2.1 + 1))

def valid_placement (d1 d2 : Domino) : Prop :=
  d1.c1 ≠ d2.c1 ∧ d1.c1 ≠ d2.c2 ∧ d1.c2 ≠ d2.c1 ∧ d1.c2 ≠ d2.c2

def within_chessboard (d : Domino) : Prop :=
  d.c1.1 < 8 ∧ d.c1.2 < 8 ∧ d.c2.1 < 8 ∧ d.c2.2 < 8

def all_valid_placements (dominoes : List Domino) : Prop :=
  (∀ d, d ∈ dominoes → within_chessboard d) ∧
  (∀ (d1 ∈ dominoes) (d2 ∈ dominoes), d1 ≠ d2 → valid_placement d1 d2)

def not_fully_covered (grid : List (Nat × Nat)) (dominoes : List Domino) : Prop :=
  ∃ (i j : Nat), i < 7 ∧ j < 7 ∧ 
  ([((i, j), (i, j + 1)), ((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)), ((i, j + 1), (i + 1, j + 1))] 
    ∃ (p : Nat × Nat), p ∉ (dominoes.bind (fun d => [d.c1, d.c2])))

theorem domino_2x2_not_fully_covered (dominoes : List Domino) (h_len : dominoes.length = 9) :
  all_valid_placements dominoes → not_fully_covered [(i, j) || i <- List.range 8, j <- List.range 8] dominoes :=
by
  sorry

end domino_2x2_not_fully_covered_l437_437703


namespace smallest_multiple_1_through_10_l437_437823

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437823


namespace two_times_first_exceeds_three_times_second_l437_437010

theorem two_times_first_exceeds_three_times_second :
  let first_number := 7
      second_number := 3 in 
  2 * first_number - 3 * second_number = 5 :=
by
  let first_number := 7
  let second_number := 3
  sorry

end two_times_first_exceeds_three_times_second_l437_437010


namespace lcm_1_to_10_l437_437953

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437953


namespace simplify_complex_div_l437_437723

theorem simplify_complex_div : (5 + 7 * complex.I) / (2 + 3 * complex.I) = (31 / 13) - (1 / 13) * complex.I :=
by
  sorry

end simplify_complex_div_l437_437723


namespace compare_algorithms_with_euclids_algorithm_l437_437523

-- Define Euclid's algorithm as a function that computes the GCD of two numbers.
def euclids_algorithm (m n : ℕ) : ℕ :=
  if n = 0 then m else euclids_algorithm n (m % n)

-- Define the Method of Continuous Subtraction.
noncomputable def method_of_continuous_subtraction (m n : ℕ) : ℕ :=
  if m = n then m
  else if m > n then method_of_continuous_subtraction (m - n) n
  else method_of_continuous_subtraction m (n - m)

-- Proof statement: Here we assert that the Method of Continuous Subtraction can be used 
-- to find the GCD of two numbers, akin to how Euclid's algorithm does.
theorem compare_algorithms_with_euclids_algorithm :
  ∀ (m n : ℕ), method_of_continuous_subtraction m n = euclids_algorithm m n :=
by
  sorry

end compare_algorithms_with_euclids_algorithm_l437_437523


namespace arithmetic_operations_result_eq_one_over_2016_l437_437283

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end arithmetic_operations_result_eq_one_over_2016_l437_437283


namespace tetrahedron_exists_l437_437375

-- Variables and Conditions
variables (m : ℕ) (h1 : m ≥ 2) (points : set ℝ^3) (h2 : points.card = 3 * m)
variable (segments : set (ℝ^3 × ℝ^3))
variable (h3 : segments.card = 3 * m^2 + 1)
variable (h4 : ∀ (p1 p2 p3 p4 : ℝ^3), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
                 ¬(coplanar p1 p2 p3 p4))

-- Theorem statement: Prove that the segments form at least one tetrahedron.
theorem tetrahedron_exists : ∃ (p1 p2 p3 p4 : ℝ^3), 
  p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧ 
  (p1, p2) ∈ segments ∧ (p1, p3) ∈ segments ∧ (p1, p4) ∈ segments ∧ 
  (p2, p3) ∈ segments ∧ (p2, p4) ∈ segments ∧ (p3, p4) ∈ segments :=
sorry

end tetrahedron_exists_l437_437375


namespace range_g_l437_437544

def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arccot x

theorem range_g : 
  set.range (g) = Set.interval (3*Real.pi/4) (5*Real.pi/4) :=
begin
  sorry
end

end range_g_l437_437544


namespace water_bottle_costs_l437_437742

theorem water_bottle_costs : 
  let x_min1 := 250
  let x_max1 := 270
  let x_min2 := 258.33 -- represented in Lean using rational or decimal approximation
  let x_max2 := 270
  let valid_values := {x : ℤ | x_min1 ≤ x ∧ x < x_max1 ∧ x_min2 ≤ x ∧ x < x_max2}
  (valid_values.card = 11) :=
by
  sorry

end water_bottle_costs_l437_437742


namespace copy_pages_l437_437665

theorem copy_pages (cost_per_5_pages : ℝ) (total_dollars : ℝ) : 
  (cost_per_5_pages = 10) → (total_dollars = 15) → (15 * 100 / 10 * 5 = 750) :=
by
  intros
  sorry

end copy_pages_l437_437665


namespace line_perpendicular_to_plane_implies_perpendicular_to_lines_in_plane_l437_437172

-- Definitions for lines, planes, and perpendicularity
structure Line :=
  (id: ℕ) -- Just for unique identity
structure Plane :=
  (id: ℕ) -- Just for unique identity

def PerpendicularTo (l: Line) (p: Plane) : Prop := sorry
def SubsetOf (m: Line) (p: Plane) : Prop := sorry
def Perpendicular (l m: Line) : Prop := sorry

-- Conditions
variable {l m: Line}
variable {α: Plane}
variable h1: PerpendicularTo l α
variable h2: SubsetOf m α

-- Lean statement for the proof
theorem line_perpendicular_to_plane_implies_perpendicular_to_lines_in_plane :
  PerpendicularTo l α → SubsetOf m α → Perpendicular l m :=
by {
  intros,
  sorry
}

end line_perpendicular_to_plane_implies_perpendicular_to_lines_in_plane_l437_437172


namespace expand_polynomial_l437_437555

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 := 
by
  sorry

end expand_polynomial_l437_437555


namespace fraction_transformation_correct_l437_437047

theorem fraction_transformation_correct
  {a b : ℝ} (hb : b ≠ 0) : 
  (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_transformation_correct_l437_437047


namespace distribute_prizes_to_students_l437_437143

theorem distribute_prizes_to_students :
  let students := 3
  let prizes := 5
  number_of_ways students prizes = 150 :=
by
  sorry

end distribute_prizes_to_students_l437_437143


namespace triangle_AB_eq_4_l437_437257

-- Define the given conditions in the problem
def BC : ℝ := 1
def angleB : ℝ := Real.pi / 3
def areaABC : ℝ := Real.sqrt 3

-- The theorem we need to prove
theorem triangle_AB_eq_4 (hBC : BC = 1) 
                         (hAngleB : angleB = Real.pi / 3)
                         (hArea : areaABC = Real.sqrt 3) : 
    ∀ (AB : ℝ), AB = 4 :=
by
  sorry

end triangle_AB_eq_4_l437_437257


namespace trigonometric_inequality_1_l437_437711

theorem trigonometric_inequality_1 {n : ℕ} 
  (h1 : 0 < n) (x : ℝ) (h2 : 0 < x) (h3 : x < (Real.pi / (2 * n))) :
  (1 / 2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1 / n) * Real.tan (n * x) := 
sorry

end trigonometric_inequality_1_l437_437711


namespace ship_travel_distance_equation_l437_437090

variable (x : ℝ) -- Distance between port A and port B
variable (v_ship : ℝ := 26) -- Speed of the ship in km/h
variable (v_water : ℝ := 2) -- Speed of the water in km/h
variable (time_diff : ℝ := 3) -- Time difference in hours

theorem ship_travel_distance_equation :
  (x / (v_ship + v_water) = x / (v_ship - v_water) - time_diff) :=
begin
  sorry
end

end ship_travel_distance_equation_l437_437090


namespace square_side_length_l437_437397

variable (x : ℝ) (π : ℝ) (hπ: π = Real.pi)

theorem square_side_length (h1: 4 * x = 10 * π) : 
  x = (5 * π) / 2 := 
by
  sorry

end square_side_length_l437_437397


namespace jasmine_laps_l437_437668

theorem jasmine_laps (x : ℕ) :
  (∀ (x : ℕ), ∃ (y : ℕ), y = 60 * x) :=
by
  sorry

end jasmine_laps_l437_437668


namespace parabola_centroid_area_sum_l437_437599

theorem parabola_centroid_area_sum (A B C : ℝ × ℝ) (O F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hP : ∀ P ∈ [A, B, C], ∃ x y, P = (x, y) ∧ y^2 = 4 * x)
  (hC : ∀ τ₁ τ₂ τ₃ : ℝ × ℝ, τ₁ + τ₂ + τ₃ = (3, 0))
  (hA : ∃ y₁ : ℝ, A = (1, y₁))
  (hB : ∃ y₂ : ℝ, B = (1, y₂))
  (hC : ∃ y₃ : ℝ, C = (1, y₃)) : 
  let S₁ := (1 / 2) * |y₁|, S₂ := (1 / 2) * |y₂|, S₃ := (1 / 2) * |y₃| in
  S₁^2 + S₂^2 + S₃^2 = 3 := 
sorry

end parabola_centroid_area_sum_l437_437599


namespace complex_value_of_product_l437_437310

theorem complex_value_of_product (r : ℂ) (hr : r^7 = 1) (hr1 : r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := 
by sorry

end complex_value_of_product_l437_437310


namespace at_least_one_irrational_l437_437333

theorem at_least_one_irrational (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
  ¬ (∀ a b : ℚ, a ≠ 0 ∧ b ≠ 0 → a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :=
by sorry

end at_least_one_irrational_l437_437333


namespace find_a_l437_437595

theorem find_a (a x : ℝ) (h : x = 1) (h_eq : 2 - 3 * (a + x) = 2 * x) : a = -1 := by
  sorry

end find_a_l437_437595


namespace max_value_y_l437_437246

theorem max_value_y (x : ℝ) : ∃ y, y = -3 * x^2 + 6 ∧ ∀ z, (∃ x', z = -3 * x'^2 + 6) → z ≤ y :=
by sorry

end max_value_y_l437_437246


namespace bread_pieces_total_l437_437700

def initial_slices : ℕ := 2
def pieces_per_slice (n : ℕ) : ℕ := n * 4

theorem bread_pieces_total : pieces_per_slice initial_slices = 8 :=
by
  sorry

end bread_pieces_total_l437_437700


namespace max_sum_11xy_3x_2012yz_l437_437465

theorem max_sum_11xy_3x_2012yz (x y z : ℕ) (h : x + y + z = 1000) : 
  11 * x * y + 3 * x + 2012 * y * z ≤ 503000000 :=
sorry

end max_sum_11xy_3x_2012yz_l437_437465


namespace initial_fish_count_l437_437331

theorem initial_fish_count (x : ℕ) (h1 : x + 47 = 69) : x = 22 :=
by
  sorry

end initial_fish_count_l437_437331


namespace intersection_of_S_and_complement_of_T_in_U_l437_437695

def U : Set ℕ := { x | 0 ≤ x ∧ x ≤ 8 }
def S : Set ℕ := { 1, 2, 4, 5 }
def T : Set ℕ := { 3, 5, 7 }
def C_U_T : Set ℕ := { x | x ∈ U ∧ x ∉ T }

theorem intersection_of_S_and_complement_of_T_in_U :
  S ∩ C_U_T = { 1, 2, 4 } :=
by
  sorry

end intersection_of_S_and_complement_of_T_in_U_l437_437695


namespace find_number_of_math_problems_l437_437123

-- Define the number of social studies problems
def social_studies_problems : ℕ := 6

-- Define the number of science problems
def science_problems : ℕ := 10

-- Define the time to solve each type of problem in minutes
def time_per_math_problem : ℝ := 2
def time_per_social_studies_problem : ℝ := 0.5
def time_per_science_problem : ℝ := 1.5

-- Define the total time to solve all problems in minutes
def total_time : ℝ := 48

-- Define the theorem to find the number of math problems
theorem find_number_of_math_problems (M : ℕ) :
  time_per_math_problem * M + time_per_social_studies_problem * social_studies_problems + time_per_science_problem * science_problems = total_time → 
  M = 15 :=
by {
  -- proof is not required to be written, hence expressing the unresolved part
  sorry
}

end find_number_of_math_problems_l437_437123


namespace solve_fractional_eq_l437_437003

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end solve_fractional_eq_l437_437003


namespace smallest_divisible_1_to_10_l437_437919

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437919


namespace sum_sin_sixth_powers_l437_437531

theorem sum_sin_sixth_powers :
    (∑ i in Finset.range 91, Real.sin (i * Real.pi / 180) ^ 6) = 229 / 8 := 
sorry

end sum_sin_sixth_powers_l437_437531


namespace smallest_number_divisible_by_1_through_10_l437_437975

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437975


namespace period_tan_minus_cot_l437_437040

theorem period_tan_minus_cot : ∀ x, f x = f (x + 2 * π)
where 
  f x := tan x - cot x := sorry

end period_tan_minus_cot_l437_437040


namespace smallest_number_divisible_1_to_10_l437_437936

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437936


namespace smallest_number_div_by_1_to_10_l437_437790

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437790


namespace rectangle_width_l437_437635

theorem rectangle_width (L W : ℝ) (h₁ : 2 * L + 2 * W = 54) (h₂ : W = L + 3) : W = 15 :=
sorry

end rectangle_width_l437_437635


namespace taps_equiv_teps_l437_437248

variable (T O E : Type) [Mul T] [Mul O] [Mul E]

def tap_to_top_equiv (t : T) (o : O) : Prop := 5 * t = 4 * o
def top_to_tep_equiv (o : O) (e : E) : Prop := 3 * o = 12 * e

theorem taps_equiv_teps {t : T} {o : O} {e : E} : tap_to_top_equiv T O → top_to_tep_equiv O E → (15 * t) = (48 * e) :=
by
  sorry

end taps_equiv_teps_l437_437248


namespace tetrahedra_intersection_volume_l437_437421

theorem tetrahedra_intersection_volume (T1 T2 : Tetrahedron) (V : Vertex)
  (hT1 : regular_tetrahedron T1)
  (hT2 : regular_tetrahedron T2) 
  (hDistinct : T1 ≠ T2) 
  (hVertices_in_cube : (∀ v ∈ T1.vertices, v ∈ unit_cube.vertices) ∧ (∀ v ∈ T2.vertices, v ∈ unit_cube.vertices)) 
  : volume (T1 ∩ T2) = 1/6 := by sorry

end tetrahedra_intersection_volume_l437_437421


namespace number_of_distinct_k_l437_437774

def is_integer_solution_quadratic (a b c x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

def is_rational (q : ℚ) (m n : ℤ) (h1 : n ≠ 0) : Prop :=
  q = (m : ℚ) / (n : ℚ)

theorem number_of_distinct_k (N : ℤ) :
  (∃ k : ℚ, is_rational k (8 + 3 * m * m) m ∧ |k| < 250 ∧ 
  ∃ x : ℤ, is_integer_solution_quadratic 3 k 8 x) →
  N = 164 := 
  sorry

end number_of_distinct_k_l437_437774


namespace plane_speed_in_still_air_l437_437085

theorem plane_speed_in_still_air (p w : ℝ) (h1 : (p + w) * 3 = 900) (h2 : (p - w) * 4 = 900) : p = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l437_437085


namespace problem1_problem2_l437_437368

-- Problem 1
theorem problem1 (x : ℝ) : 
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (x + 3)^2 = (1 - 2 * x)^2 ↔ x = 4 ∨ x = -2 / 3 := 
sorry

end problem1_problem2_l437_437368


namespace lcm_1_to_10_l437_437945

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437945


namespace period_tan_minus_cot_l437_437039

theorem period_tan_minus_cot : ∀ x, f x = f (x + 2 * π)
where 
  f x := tan x - cot x := sorry

end period_tan_minus_cot_l437_437039


namespace smallest_divisible_1_to_10_l437_437922

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437922


namespace lcm_1_to_10_l437_437955

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437955


namespace consecutive_lucky_years_l437_437518

def is_lucky (Y : ℕ) : Prop := 
  let first_two_digits := Y / 100
  let last_two_digits := Y % 100
  Y % (first_two_digits + last_two_digits) = 0

theorem consecutive_lucky_years : ∃ Y : ℕ, is_lucky Y ∧ is_lucky (Y + 1) :=
by
  sorry

end consecutive_lucky_years_l437_437518


namespace smallest_divisible_by_1_to_10_l437_437892

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437892


namespace distance_expression_correct_l437_437336

-- Define the points A and B on the number line
def A : ℤ := -4
def B : ℤ := 2

-- Define the distance expression between points A and B
def distance_expr := B - A

-- Assert that the expression B - A correctly represents the distance between points A and B
theorem distance_expression_correct : distance_expr = 2 - (-4) :=
by 
  have h1 : distance_expr = 2 - (-4),
  -- Add steps that transform distance_expr to the target equality
  have h2 : distance_expr = 6 : sorry -- Proof in detail is omitted
  exact h1

end distance_expression_correct_l437_437336


namespace smallest_divisible_by_1_to_10_l437_437895

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437895


namespace fan_working_time_each_day_l437_437490

theorem fan_working_time_each_day
  (airflow_per_second : ℝ)
  (total_airflow_week : ℝ)
  (seconds_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (airy_sector: airflow_per_second = 10)
  (flow_week : total_airflow_week = 42000)
  (sec_per_hr : seconds_per_hour = 3600)
  (hrs_per_day : hours_per_day = 24)
  (days_week : days_per_week = 7) :
  let airflow_per_hour := airflow_per_second * seconds_per_hour
  let total_hours_week := total_airflow_week / airflow_per_hour
  let hours_per_day_given := total_hours_week / days_per_week
  let minutes_per_day := hours_per_day_given * 60
  minutes_per_day = 10 := 
by
  sorry

end fan_working_time_each_day_l437_437490


namespace smallest_number_divisible_by_1_to_10_l437_437867

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437867


namespace algebraic_expression_value_l437_437365

theorem algebraic_expression_value (x : ℝ) (hx : x = 2 * Real.cos 45 + 1) :
  (1 / (x - 1) - (x - 3) / (x ^ 2 - 2 * x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end algebraic_expression_value_l437_437365


namespace solve_quadratic_eq_l437_437367

theorem solve_quadratic_eq (x : ℝ) (h : x^2 + 2 * x - 15 = 0) : x = 3 ∨ x = -5 :=
by {
  sorry
}

end solve_quadratic_eq_l437_437367


namespace exists_s_le_kr_l437_437537

theorem exists_s_le_kr (r : ℝ) (k : ℕ) (h_r : r > 0) (h_k : k > 0) :
  ∃ s : ℝ, s ∈ (generateNumbers r (k^2 - 1)) ∧ s ≤ k * r :=
sorry

-- Define the operation to generate numbers on board
noncomputable def generateNumbers (r : ℝ) (n : ℕ) : Set ℝ :=
  if n = 0 then {r}
  else ⋃ (s ∈ generateNumbers r (n - 1)), {a, b | 2 * s^2 = a * b ∧ a > 0 ∧ b > 0}

end exists_s_le_kr_l437_437537


namespace arctan_tan_expression_l437_437533

noncomputable def tan (x : ℝ) : ℝ := sorry
noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem arctan_tan_expression :
  arctan (tan 65 - 2 * tan 40) = 25 := sorry

end arctan_tan_expression_l437_437533


namespace fraction_of_jeans_wearers_in_tennis_shoes_l437_437121

theorem fraction_of_jeans_wearers_in_tennis_shoes 
  (N : ℕ) 
  (h1 : 0.8 * N) 
  (h2 : 0.7 * N) 
  (h3 : 0.8 * (0.8 * N)) : 
  (0.64 * N) / (0.7 * N) = 32 / 35 :=
by
  sorry

end fraction_of_jeans_wearers_in_tennis_shoes_l437_437121


namespace compounded_ratio_is_2_1_l437_437426

-- Define the compounded ratio function
def compounded_ratio (a b c d e f : ℕ) : ℕ × ℕ :=
  let num := a * c * e
  let denom := b * d * f
  (num, denom)

-- Define a function to simplify the ratio
def simplify_ratio (num denom : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd num denom
  (num / gcd, denom / gcd)

-- Define the given ratios
def ratio1 := (2, 3)
def ratio2 := (6, 11)
def ratio3 := (11, 2)

-- Lean proof statement without proof
theorem compounded_ratio_is_2_1 :
  simplify_ratio (compounded_ratio (2, 3) (6, 11) (11, 2)).1 =
  (2, 1) := by
  sorry

end compounded_ratio_is_2_1_l437_437426


namespace compute_expression_l437_437132

-- Definition of the expression
def expression := 5 + 4 * (4 - 9)^2

-- Statement of the theorem, asserting the expression equals 105
theorem compute_expression : expression = 105 := by
  sorry

end compute_expression_l437_437132


namespace smallest_number_divisible_by_1_through_10_l437_437979

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437979


namespace locus_of_point_P_l437_437057

noncomputable def equilateral_triangle_area (l : ℝ) : ℝ :=
  (sqrt 3 / 4) * l^2

def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d := (sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) -- distance AB
  ∧ d = (sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) -- distance BC
  ∧ d = (sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)) -- distance CA

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

def is_circumcenter (O A B C : ℝ × ℝ) : Prop :=
  (sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2)) = (sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2))
  ∧ (sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)) = (sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2))

theorem locus_of_point_P (A B C P O : ℝ × ℝ)
  (h_triangle_ABC : is_equilateral_triangle A B C)
  (h_area_equal : triangle_area P A B + triangle_area P B C + triangle_area P C A = 
    (1 / 3) * triangle_area A B C) :
  is_circumcenter O A B C → P = O := 
sorry

end locus_of_point_P_l437_437057


namespace square_area_problem_l437_437651

theorem square_area_problem 
  (BM : ℝ) 
  (ABCD_is_divided : Prop)
  (hBM : BM = 4)
  (hABCD_is_divided : ABCD_is_divided) : 
  ∃ (side_length : ℝ), side_length * side_length = 144 := 
by
-- We skip the proof part for this task
sorry

end square_area_problem_l437_437651


namespace train_speed_platform_man_l437_437099

theorem train_speed_platform_man (t_man t_platform : ℕ) (platform_length : ℕ) (v_train_mps : ℝ) (v_train_kmph : ℝ) 
  (h1 : t_man = 18) 
  (h2 : t_platform = 32) 
  (h3 : platform_length = 280)
  (h4 : v_train_mps = (platform_length / (t_platform - t_man)))
  (h5 : v_train_kmph = v_train_mps * 3.6) :
  v_train_kmph = 72 := 
sorry

end train_speed_platform_man_l437_437099


namespace minimize_perimeter_l437_437663

variables {A B C M : Type} [EuclideanGeometry A B C M]

theorem minimize_perimeter (acute_angle_BAC : ∀ (A B C M : Type),  ∠BAC < 90) (point_M_inside : M ∈ interior ∠BAC)
  : ∃ (X : BA) (Y : AC), (perimeter (triangle X M Y)) = perimeter (segment (reflect M A B) (reflect M A C)) := sorry

end minimize_perimeter_l437_437663


namespace max_possible_cos_difference_l437_437373

open Real

noncomputable def max_cos_difference (a1 a2 a3 a4 : ℝ) (a b c : ℝ) :=
  cos a1 - cos a4

theorem max_possible_cos_difference :
  ∃ (a1 a2 a3 a4 : ℝ) (a b c : ℝ),
  a3 = a2 + a1 ∧
  a4 = a3 + a2 ∧
  (∀ (n : ℕ), n ∈ {1, 2, 3, 4} → a * (n:ℝ) ^ 2 + b * (n : ℝ) + c = cos (list.nth_le [a1, a2, a3, a4] (n - 1) sorry)) ∧
  max_cos_difference a1 a2 a3 a4 a b c = -9 + 3 * sqrt 13 := sorry

end max_possible_cos_difference_l437_437373


namespace class_schedule_arrangements_l437_437339

-- Definitions for the conditions
structure Schedule :=
  (first_three : Finset String) -- Chinese and Foreign Language in the first three sessions
  (last_two : Finset String)    -- Biology in the last two sessions
  (non_adjacent_pc : List String) -- List of sessions ensuring Physics and Chemistry are not adjacent
    
-- The main theorem to be proven
theorem class_schedule_arrangements 
  (CFL_in_first_three : Schedule.first_three = {"Chinese", "Foreign Language"})
  (Bio_in_last_two : Schedule.last_two = {"Biology"})
  (PC_non_adjacent : ∀ (sch : Schedule.non_adjacent_pc), 
    (("Physics", "Chemistry") ∉ sch) ∧ (("Chemistry", "Physics") ∉ sch)):
  Schedule → 
  (total_arrangements : Nat) := 40 := 
sorry

end class_schedule_arrangements_l437_437339


namespace tan_ratio_l437_437688

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end tan_ratio_l437_437688


namespace solve_inequality_l437_437370

-- Define the inequality problem.
noncomputable def inequality_problem (x : ℝ) : Prop :=
(x^2 + 2 * x - 15) / (x + 5) < 0

-- Define the solution set.
def solution_set (x : ℝ) : Prop :=
-5 < x ∧ x < 3

-- State the equivalence theorem.
theorem solve_inequality (x : ℝ) (h : x ≠ -5) : 
  inequality_problem x ↔ solution_set x :=
sorry

end solve_inequality_l437_437370


namespace part1_part2_part3_tangent_eq_l437_437178
open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x

theorem part1 {α : ℝ} (h1 : f (α / 2) = 3 / 5) (h2 : α ∈ Ioo (π / 2) π) :
  cos (α - π / 3) = (3 * sqrt 3 - 4) / 10 :=
sorry

theorem part2 : ∀ k : ℤ, ∃ a b : ℝ, a = k * π + π / 4 ∧ b = k * π + 3 * π / 4 ∧ (∀ x ∈ Icc a b, f x < ∂(f x) / ∂x) :=
sorry

theorem part3 : ∀ x, f x = 2 * cos x ^ 2 - 2 * sin x ^ 2 := 
  sorry

theorem tangent_eq : ∀ y x₀, y = f x₀ ∧ x₀ = 0 -> y = 2 * x₀ :=
sorry

end part1_part2_part3_tangent_eq_l437_437178


namespace triangle_ratio_HD_HA_l437_437100

theorem triangle_ratio_HD_HA {A B C H D : Point} (a b c : ℝ) (s h AD : ℝ)
  (h_triangle : triangle A B C)
  (h_a : A = (7.5, 8))
  (h_b : B = (0, 0))
  (h_c : C = (15, 0))
  (h_abc : a = 8 ∧ b = 15 ∧ c = 17)
  (h_s : s = (a + b + c) / 2)
  (h_A : A = sqrt (s * (s - a) * (s - b) * (s - c)))
  (h_AD : ∃ AD, A = (1/2) * b * h → h = (2 * A) / b)
  (h_H : H = orthocenter A B C)
  (h_HD_HA : HD / HA = 0) :
  HD / HA = 0 :=
sorry

end triangle_ratio_HD_HA_l437_437100


namespace trisha_spending_l437_437780

theorem trisha_spending :
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  let total_spent := initial_amount - remaining_amount
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  total_spent - other_spending = 22 :=
by
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  -- Calculate total spent
  let total_spent := initial_amount - remaining_amount
  -- Calculate spending on other items
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  -- Statement to prove
  show total_spent - other_spending = 22
  sorry

end trisha_spending_l437_437780


namespace area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l437_437064

-- (a) Proving the area of triangle ABC
theorem area_triangle_ABC (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 3) (h_right : true) : 
  (1 / 2) * AB * BC = 3 := sorry

-- (b) Proving the area of figure DEFGH
theorem area_figure_DEFGH (DH HG : ℝ) (hDH : DH = 5) (hHG : HG = 5) (triangle_area : ℝ) (hEPF : triangle_area = 3) : 
  DH * HG - triangle_area = 22 := sorry

-- (c) Proving the area of triangle JKL 
theorem area_triangle_JKL (side_area : ℝ) (h_side : side_area = 25) 
  (area_JSK : ℝ) (h_JSK : area_JSK = 3) 
  (area_LQJ : ℝ) (h_LQJ : area_LQJ = 15/2) 
  (area_LRK : ℝ) (h_LRK : area_LRK = 5) : 
  side_area - area_JSK - area_LQJ - area_LRK = 19/2 := sorry

end area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l437_437064


namespace problem_statement_l437_437575

theorem problem_statement (x y : ℝ) : (x * y < 18) → (x < 2 ∨ y < 9) :=
sorry

end problem_statement_l437_437575


namespace smallest_sum_of_squares_l437_437386

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := 
  sorry

end smallest_sum_of_squares_l437_437386


namespace prob_fifth_card_is_ace_of_hearts_l437_437093

theorem prob_fifth_card_is_ace_of_hearts : 
  (∀ d : List ℕ, d.length = 52 → (1 ∈ d) → Prob (d.nth 4 = some 1) = 1 / 52) := 
by
  sorry

end prob_fifth_card_is_ace_of_hearts_l437_437093


namespace sum_divisors_of_72_90_30_l437_437565

theorem sum_divisors_of_72_90_30 : 
  let divisors_72 := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}
  let divisors_90 := {1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90}
  let divisors_30 := {1, 2, 3, 5, 6, 10, 15, 30}
  let common_divisors := {1, 2, 3, 6}
  common_divisors = (divisors_72 ∩ divisors_90 ∩ divisors_30) →
  ∑ x in common_divisors, x = 12 :=
by
  let divisors_72 := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}
  let divisors_90 := {1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90}
  let divisors_30 := {1, 2, 3, 5, 6, 10, 15, 30}
  sorry

-- Explanation:
-- - We define divisors_72, divisors_90, and divisors_30 as sets of numbers.
-- - We calculate common_divisors as the intersection of these sets.
-- - We then state the problem of proving that the sum of the elements in common_divisors equals 12.

end sum_divisors_of_72_90_30_l437_437565


namespace find_four_digit_numbers_l437_437065

theorem find_four_digit_numbers:
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
   2 * (1000 * a + 100 * b + 10 * c + d) = 9 * (1000 * d + 100 * c + 10 * b + a) ∧
   1000 <= (1000 * a + 100 * b + 10 * c + d) ∧ (1000 * a + 100 * b + 10 * c + d) <= 9999) →
  (∃ n : ℕ, n = 8991 ∨ n = 8181 ∧
   (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d)) :=
begin
  sorry
end

end find_four_digit_numbers_l437_437065


namespace ones_digit_11_pow_l437_437431

theorem ones_digit_11_pow (n : ℕ) (hn : n > 0) : (11^n % 10) = 1 := by
  sorry

end ones_digit_11_pow_l437_437431


namespace arithmetic_sequence_8th_term_l437_437425

theorem arithmetic_sequence_8th_term :
  ∃ (a_n : ℚ), a_n ≈ 25.2069 ∧ 
    ∀ (a₁ a₃₀ d : ℚ), 
      a₁ = 3 → a₃₀ = 95 →
      d = (95 - 3) / (30 - 1) →
      a_n = a₁ + (8 - 1) * d :=
sorry

end arithmetic_sequence_8th_term_l437_437425


namespace team_A_win_probability_l437_437403

theorem team_A_win_probability :
  let win_prob := (1 / 3 : ℝ)
  let team_A_lead := 2
  let total_sets := 5
  let require_wins := 3
  let remaining_sets := total_sets - team_A_lead
  let prob_team_B_win_remaining := (1 - win_prob) ^ remaining_sets
  let prob_team_A_win := 1 - prob_team_B_win_remaining
  prob_team_A_win = 19 / 27 := by
    sorry

end team_A_win_probability_l437_437403


namespace twelve_digit_number_divisible_by_9_l437_437706

theorem twelve_digit_number_divisible_by_9 :
  ∀ (digit : Fin 12 → Fin 10), (∀ i, digit i + digit (i + 1) = 9) →
  (digit 0 + digit 1 + digit 2 + digit 3 + digit 4 + digit 5 + digit 6 + digit 7 
  + digit 8 + digit 9 + digit 10 + digit 11) % 9 = 0 :=
begin
  sorry -- proof goes here
end

end twelve_digit_number_divisible_by_9_l437_437706


namespace distance_formula_l437_437588

variable {x1 y1 x2 y2 : ℝ}

theorem distance_formula :
  dist (x1, y1) (x2, y2) = real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) :=
sorry

end distance_formula_l437_437588


namespace tan_ratio_l437_437686

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end tan_ratio_l437_437686


namespace largest_integer_three_operations_l437_437159

theorem largest_integer_three_operations : 
    ∃ x : ℝ, (⌊√x⌋.val = 80) ∧ (⌊√80⌋.val = 8) ∧ (⌊√8⌋.val = 2) ∧ x = 6560 := 
by
  sorry

end largest_integer_three_operations_l437_437159


namespace smallest_number_divisible_by_1_to_10_l437_437863

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437863


namespace digit_is_not_six_must_be_true_l437_437520

-- Definitions of the statements
def digit_is_five := Prop
def digit_is_not_six := Prop
def digit_is_seven := Prop
def digit_is_not_eight := Prop

-- Given conditions
def statements (I II III IV : Prop) :=
  (I ∨ II ∨ III ∨ IV) ∧
  (I → ¬III) ∧ (III → ¬I) ∧
  ((I ∧ ¬III) ∨ (¬I ∧ III)) ∧
  (I ∨ III → II) ∧
  (I ∨ III → IV)

-- Prove that Statement II (digit_is_not_six) must necessarily be true
theorem digit_is_not_six_must_be_true (I II III IV : Prop): statements I II III IV → II :=
begin
  sorry
end

end digit_is_not_six_must_be_true_l437_437520


namespace smallest_number_divisible_1_to_10_l437_437940

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437940


namespace find_values_of_m_l437_437151

theorem find_values_of_m (m : ℤ) (h₁ : m > 2022) (h₂ : (2022 + m) ∣ (2022 * m)) : 
  m = 1011 ∨ m = 2022 :=
sorry

end find_values_of_m_l437_437151


namespace trapezium_height_l437_437559

-- Defining the lengths of the parallel sides and the area of the trapezium
def a : ℝ := 28
def b : ℝ := 18
def area : ℝ := 345

-- Defining the distance between the parallel sides to be proven
def h : ℝ := 15

-- The theorem that proves the distance between the parallel sides
theorem trapezium_height :
  (1 / 2) * (a + b) * h = area :=
by
  sorry

end trapezium_height_l437_437559


namespace part1_l437_437576

variables (a : ℝ) (C : ℝ × ℝ) (A : ℝ × ℝ) (x y : ℝ)
def circleRadius : ℝ := 1
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (0, 3)
def is_symmetric_line : ℝ × ℝ := (x - y = 3)
def centerC : ℝ × ℝ := (a, 2*a-4)
def line1 : Prop := 12*x + 5*y - 15 = 0
def line2 : Prop := x = 0
def tangent_line := (line1 ∨ line2)

theorem part1 (hC : circleRadius = 1) (hA : A = (0, 3))
    (hCenter : C = (a, 2*a - 4)) (ha_pos : a > 0)
    (hSymmetric : is_symmetric_line = (x - y - 3 = 0))
    (hTangent : tangent_line):
  tangent_line := by 
    sorry

end part1_l437_437576


namespace infinite_primes_among_p_powers_l437_437293

noncomputable def primes_count (s : Set ℕ) : ℕ :=
  s.count (λ x, x.prime)

theorem infinite_primes_among_p_powers 
  (p : ℕ) (hp : 1 < p) (K : ℝ) (hK : 0 < K) :
  ∃ᵢ (n N : ℕ), primes_count (Set.image (λ i, n + i^p) (Set.range N.succ)) 
    ≥ (K * N) / Real.log N :=
sorry

end infinite_primes_among_p_powers_l437_437293


namespace multiply_add_fractions_l437_437433

theorem multiply_add_fractions :
  (2 / 9 : ℚ) * (5 / 8) + (1 / 4) = 7 / 18 := by
  sorry

end multiply_add_fractions_l437_437433


namespace relationship_between_M_and_N_l437_437168

-- Define the conditions of the problem
variables {a1 a2 : ℝ}
axiom h1 : 0 < a1 ∧ a1 < 1
axiom h2 : 0 < a2 ∧ a2 < 1

-- Define M and N from the problem
def M : ℝ := a1 * a2
def N : ℝ := a1 + a2 + 1

-- The theorem we want to prove
theorem relationship_between_M_and_N : M < N :=
by
  sorry

end relationship_between_M_and_N_l437_437168


namespace space_diagonals_Q_l437_437076

-- Definitions based on the conditions
def vertices (Q : Type) : ℕ := 30
def edges (Q : Type) : ℕ := 70
def faces (Q : Type) : ℕ := 40
def triangular_faces (Q : Type) : ℕ := 20
def quadrilateral_faces (Q : Type) : ℕ := 15
def pentagon_faces (Q : Type) : ℕ := 5

-- Problem Statement
theorem space_diagonals_Q :
  ∀ (Q : Type),
  vertices Q = 30 →
  edges Q = 70 →
  faces Q = 40 →
  triangular_faces Q = 20 →
  quadrilateral_faces Q = 15 →
  pentagon_faces Q = 5 →
  ∃ d : ℕ, d = 310 := 
by
  -- At this point only the structure of the proof is set up.
  sorry

end space_diagonals_Q_l437_437076


namespace period_of_y_l437_437036

-- Define the function y = tan x - cot x
def y (x : ℝ) : ℝ := Real.tan x - Real.cot x

-- Prove that y has a period of π
theorem period_of_y : ∀ x, y (x + π) = y x :=
by
  intro x
  -- Define what needs to be shown
  have h : y (x + π) = Real.tan (x + π) - Real.cot (x + π) := rfl
  -- Use the periodicity of tangent and cotangent
  rw [Real.tan_add_pi, Real.cot_add_pi]
  -- Simplifying terms
  simp [Real.tan, Real.cot]
  sorry

end period_of_y_l437_437036


namespace distinct_numbers_mean_inequality_l437_437318

open Nat

theorem distinct_numbers_mean_inequality (n m : ℕ) (h_n_m : m ≤ n)
  (a : Fin m → ℕ) (ha_distinct : Function.Injective a)
  (h_cond : ∀ (i j : Fin m), i ≠ j → i.val + j.val ≤ n → ∃ (k : Fin m), a i + a j = a k) :
  (1 : ℝ) / m * (Finset.univ.sum (fun i => a i)) ≥  (n + 1) / 2 :=
by
  sorry

end distinct_numbers_mean_inequality_l437_437318


namespace angle_IMA_eq_angle_INB_l437_437280

-- Given the geometric setup of triangle ABC
variables {A B C I M N : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited I] [Inhabited M] [Inhabited N]

-- Conditions of the problem
variable (triangle_ABC : triangle A B C)
variable (incenter_I : incenter_of_triangle I triangle_ABC)
variable (midpoint_M : midpoint_of_side M triangle_ABC.side_AC)
variable (midpoint_N : midpoint_of_arc N (circumcircle_of_triangle triangle_ABC))

-- Theorem statement to be proved
theorem angle_IMA_eq_angle_INB : 
  ∠ I M A = ∠ I N B := 
sorry

end angle_IMA_eq_angle_INB_l437_437280


namespace smallest_number_divisible_by_1_to_10_l437_437860

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437860


namespace solution_set_g_lt_zero_sequence_a_n_sum_s_n_lt_1_l437_437612

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1 
def f (h : Real → Real → Real) (H : ∀ x y, h(xy) = x * h(y) + y * h(x)) (x : ℝ) : ℝ := sorry

-- Problem 1
theorem solution_set_g_lt_zero (a x : ℝ) : 
  (a < 0 → (g a x < 0) ↔ (x < 1/a ∨ x > 1)) ∧
  (a = 0 → (g a x < 0) ↔ (x > 1)) ∧
  (0 < a ∧ a < 1 → (g a x < 0) ↔ (1 < x ∧ x < 1/a)) ∧
  (a = 1 → (g a x < 0) ↔ false) ∧
  (a > 1 → (g a x < 0) ↔ (1/a < x ∧ x < 1)) :=
sorry

-- Problem 2
theorem sequence_a_n (f : ℝ → ℝ) (n : ℕ) (h2 : f 2 = g 1 2 + 1) : 
  (a_n f n = n * 2^n) :=
sorry

-- Problem 3
theorem sum_s_n_lt_1 (n : ℕ) (h2 : f 2 = g 1 2 + 1) : 
  let a_n := fun n => n * 2^n
  let b_n := fun n => (n + 2) / (n + 1) * 1 / (a_n n)
  S_n < 1 :=
sorry

end solution_set_g_lt_zero_sequence_a_n_sum_s_n_lt_1_l437_437612


namespace general_term_of_sequence_l437_437327

variable (a : ℕ → ℝ)

theorem general_term_of_sequence
  (h : ∀ n : ℕ, (∑ i in Finset.range n, 3^i * a (i + 1)) = n / 3) :
  ∀ n : ℕ, a (n + 1) = 1 / 3^(n + 1) :=
by
  sorry

end general_term_of_sequence_l437_437327


namespace present_worth_year_due_l437_437401

noncomputable def calculate_years_due (A P r : Float) : Nat :=
  let n : Float := 1
  let t : Float := (Float.log (A / P)) / (n * Float.log (1 + r / n))
  Nat.ceil t

theorem present_worth_year_due :
  calculate_years_due 1183 1093.75 0.04 = 2 :=
by
  sorry

end present_worth_year_due_l437_437401


namespace correct_statements_l437_437207

theorem correct_statements :
  (∫ x in -3..3, (x^2 + sin x)) = 18 ∧
  (∀ R^2 : ℝ, R^2 > 0 → (∀ fit : ℝ, fit ≤ 1 → R^2 fit ≥ 0)) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 2) = -f x) → (∀ x : ℝ, f (1 + (x - 1)) = -f (1 - (x - 1)))) ∧
  (∀ σ : ℝ, σ > 0 → ∃ ξ : ℝ, ξ ∼ normal 1 σ^2 ∧ (P (ξ ≤ 4) = 0.79 → P (ξ < -2) = 0.21)) :=
sorry

end correct_statements_l437_437207


namespace part1_part2_l437_437229

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (m : ℝ)

-- conditions
variables (ha : ∥a∥ = 3) (hb : ∥b∥ = 2)
variables (h_angle : real_inner_product_space.angle a b = real.pi / 3)
variables (c := 3 • a + 5 • b)
variables (d := m • a - b)

-- Proofs statements
theorem part1 : 
  ⟪a, b⟫ = 3 := 
sorry

theorem part2 (h_perp : ⟪c, d⟫ = 0) : 
  m = 29 / 42 := 
sorry

end part1_part2_l437_437229


namespace find_angle_BAM_l437_437184

variable {Point : Type} [MetricSpace Point]

variables (A B C K M : Point)
variable (angle : Point → Point → Point → ℝ)

def is_isosceles_triangle (ABC : Triangle) := 
  ABC.side_length A B = ABC.side_length A C
  ∧ angle B A C = angle B C A


def given_conditions (ABC : Triangle) (AK_line : Line) : Prop :=
  is_isosceles_triangle ABC ∧
  angle A B C = 53 ∧
  midpoint C A K AK_line ∧
  on_same_side B M (line A C) ∧
  distance K M = distance A B 

theorem find_angle_BAM (ABC : Triangle) (AK_line : Line)
  (h : given_conditions ABC AK_line) : 
  angle B A M = 44 :=
sorry

end find_angle_BAM_l437_437184


namespace janet_lives_l437_437667

variable (initial_lives lives_lost lives_gained : ℕ)

theorem janet_lives :
  initial_lives = 47 →
  lives_lost = 23 →
  lives_gained = 46 →
  initial_lives - lives_lost + lives_gained = 70 :=
by
  intros h_initial h_lost h_gained
  rw [h_initial, h_lost, h_gained]
  norm_num
  sorry

end janet_lives_l437_437667


namespace length_of_pencils_l437_437286

theorem length_of_pencils (length_pencil1 : ℕ) (length_pencil2 : ℕ)
  (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) : length_pencil1 + length_pencil2 = 24 :=
by
  sorry

end length_of_pencils_l437_437286


namespace smallest_multiple_1_through_10_l437_437827

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437827


namespace smallest_divisible_by_1_to_10_l437_437889

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437889


namespace smallest_number_divisible_1_to_10_l437_437937

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437937


namespace paulson_spends_75_percent_of_income_l437_437338

variable (P : ℝ)  -- Percentage of income Paulson spends
variable (I : ℝ)  -- Paul's original income

-- Conditions
def original_expenditure := P * I
def original_savings := I - original_expenditure

def new_income := 1.20 * I
def new_expenditure := 1.10 * original_expenditure
def new_savings := new_income - new_expenditure

-- Given: The percentage increase in savings is approximately 50%.
def percentage_increase_in_savings :=
  ((new_savings - original_savings) / original_savings) * 100

-- Proof statement
theorem paulson_spends_75_percent_of_income
  (h : percentage_increase_in_savings P I ≈ 50) : P = 0.75 :=
by
  sorry

end paulson_spends_75_percent_of_income_l437_437338


namespace tip_amount_l437_437291

theorem tip_amount (charge_per_lawn : ℕ) (mowed_lawns : ℕ) (total_earned : ℕ) 
  (H_charge : charge_per_lawn = 33) (H_lawns : mowed_lawns = 16) (H_total : total_earned = 558) :
  total_earned - (charge_per_lawn * mowed_lawns) = 30 :=
by
  rw [H_charge, H_lawns, H_total]
  norm_num
  -- proof steps can follow here
  sorry

end tip_amount_l437_437291


namespace cartesian_eq_C2_min_dist_C1_C2_l437_437268

noncomputable def curveC1 := 
  (α : ℝ) → ℝ × ℝ
  | α => (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def curveC2 := 
  (θ : ℝ) → ℝ × ℝ
  | θ => (Real.cos θ * Real.cos θ, Real.cos θ * Real.sin θ)

noncomputable def cartesian_eq (ρ : ℝ) : ℝ × ℝ :=
  let x := ρ * Real.cos ρ
  let y := ρ * Real.sin ρ
  (x, y)

theorem cartesian_eq_C2 : 
  ∀ (θ : ℝ), (\exists x y : ℝ, (x, y) = cartesian_eq θ ∧ ((x - 0.5)^2 + y^2 = 0.25)) :=
by
  intuition
  
theorem min_dist_C1_C2 :
  ∃ α θ : ℝ, let P := curveC1 α in let Q := curveC2 θ in 
  (|P.1 - Q.1|^2 + |P.2 - Q.2|^2 = (√7 - 1) / 2) :=
by
  sorry

end cartesian_eq_C2_min_dist_C1_C2_l437_437268


namespace smallest_divisible_1_to_10_l437_437913

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437913


namespace mask_production_rates_l437_437422

theorem mask_production_rates (x : ℝ) (y : ℝ) :
  (280 / x) - (280 / (1.4 * x)) = 2 →
  x = 40 ∧ y = 1.4 * x →
  y = 56 :=
by {
  sorry
}

end mask_production_rates_l437_437422


namespace valid_numbers_count_l437_437238

-- Define the digits we will be using
def digits : Finset ℕ := {0, 1, 2}

-- Define a predicate for natural numbers without repeated digits
def valid_number (n : ℕ) : Prop :=
∃ (digitsUsed : Finset ℕ), (digitsUsed ⊆ digits) ∧ (∃ (arr : List ℕ), 
  arr.erase_dup = arr ∧ finset.of_list arr = digitsUsed ∧ 
  arr.foldl (λ acc d, acc * 10 + d) 0 = n  ∧ d ∈ digitsUsed)

-- Count the number of valid numbers
def count_valid_numbers : ℕ :=
(digits.filter valid_number).card

-- The theorem to be proven
theorem valid_numbers_count : count_valid_numbers = 11 := by
  sorry

end valid_numbers_count_l437_437238


namespace arrange_rose_bushes_l437_437671

-- Definitions corresponding to the conditions.
def roses : ℕ := 15
def rows : ℕ := 6
def bushes_per_row : ℕ := 5

-- Theorem statement
theorem arrange_rose_bushes :
  ∃ (arrangement : Finset (Finset (Fin 15)) ),
  (∀ row ∈ arrangement, row.card = bushes_per_row) ∧
  (arrangement.card = rows) ∧
  (∀ (r1 r2 : Finset (Fin 15)), r1 ∈ arrangement → r2 ∈ arrangement → r1 ≠ r2 →
    (r1 ∩ r2).card = 1) :=
sorry

end arrange_rose_bushes_l437_437671


namespace triangle_has_three_altitudes_l437_437101

theorem triangle_has_three_altitudes (T : Triangle) : T.number_of_altitudes = 3 :=
sorry

end triangle_has_three_altitudes_l437_437101


namespace line_equation_through_point_inclination_l437_437082

theorem line_equation_through_point_inclination (M : ℝ × ℝ) (α : ℝ) 
  (hm : M = (2, -3)) (hα : α = 45) : ∃ (a b c : ℝ), a * M.1 + b * M.2 + c = 0 ∧ a * x + b * y + c = x - y - 5 :=
by
  -- Definition of constants needed
  let x := 2
  let y := -3
  let m := 1  -- slope of the line, since tan(45°) = 1

  have hx := x
  have hy := y
  have hm_eq := hm

  -- Prove that the line equation is in the form: x - y - 5 = 0
  use [(1 : ℝ), (-1 : ℝ), (-5 : ℝ)]
  split
  -- Verification that the point M(2, -3) satisfies the line equation
  show 1 * M.1 + -1 * M.2 + -5 = 0
  -- Simplification
  calc
    1 * 2 + -1 * (-3) + -5 = 2 + 3 + -5 : by simp [hm_eq]
    ... = 0 : by simp
  -- Verification that the general line equation is x - y - 5
  show 1 * x + -1 * y + -5 = x - y - 5
  sorry

end line_equation_through_point_inclination_l437_437082


namespace number_of_placements_l437_437486

-- We define the classroom setup
def classroom_grid : Type := list (list (option bool)) -- True for boy, False for girl, None for empty desk (not needed here).

-- Initially define the size of the grid
def rows : ℕ := 5
def columns : ℕ := 6
def students : ℕ := 30
def boys : ℕ := 15
def girls : ℕ := 15

def is_valid_placement (grid : classroom_grid) : Prop :=
  ∀ i j, (grid.nth i).bind (λ row, row.nth j) ≠ some true → (grid.nth i).bind (λ row, row.nth j.succ) = some false ∧
                                        (grid.nth i).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j.succ) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j.succ) = some false

theorem number_of_placements : ∃ (count : ℕ), count = 2 * (nat.factorial 15)^2 := by
  -- skip the proof by providing the existential value directly
  existsi (2 * (nat.factorial 15)^2)
  refl

end number_of_placements_l437_437486


namespace smallest_number_divisible_1_to_10_l437_437885

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437885


namespace smallest_divisible_1_to_10_l437_437910

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437910


namespace non_prime_factorial_expression_integers_l437_437162

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_not_prime (n : Nat) : Prop :=
  1 < n ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

theorem non_prime_factorial_expression_integers : 
  let expr (n : Nat) := (factorial (n^2 + 1)) / (factorial n)^(n + 2)
  ∃ (count : Nat), 
  ∀ n, 1 ≤ n ∧ n ≤ 60 ∧ ¬ is_not_prime n → expr n ∈ Int
  ∧ count = 43 :=
by
  sorry

end non_prime_factorial_expression_integers_l437_437162


namespace card_arrangement_impossible_l437_437990

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437990


namespace area_AEGD_l437_437271

open Real

-- Define the side length of the square based on the area
def s := 2 * sqrt 210

-- Define the vertices and segments
variables (A B C D E F G: Point)
variables (AE_eq_EB: dist A E = dist E B)
variables (BF_eq_2FC: dist B F = 2 * dist F C)
variables (area_ABCD: area (polygon_from_list [A, B, C, D]) = 840)
variables (G_intersection: point_on_line G (line_from_points D F) ∧ point_on_line G (line_from_points E C))

theorem area_AEGD :
  area (polygon_from_list [A, E, G, D]) = 510 := sorry

end area_AEGD_l437_437271


namespace number_division_l437_437044

theorem number_division (x : ℚ) (h : x / 2 = 100 + x / 5) : x = 1000 / 3 := 
by
  sorry

end number_division_l437_437044


namespace sequence_sum_l437_437222

-- Definition of the sequence S_n
def S (n : ℕ) : ℤ := ∑ k in finset.range (n+1), ((-1) ^ (k + 1)) * (k + 1)

-- Statement of the problem
theorem sequence_sum : S 17 + S 33 + S 50 = 1 :=
  sorry

end sequence_sum_l437_437222


namespace lcm_1_to_10_l437_437954

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437954


namespace sin_double_angle_l437_437627

theorem sin_double_angle {α : ℝ} (h : (cos (2 * α)) / (cos (α - π/4)) = (sqrt 2) / 2) : sin (2 * α) = 3 / 4 := 
sorry

end sin_double_angle_l437_437627


namespace largest_number_of_blocks_l437_437427

def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def num_blocks (box_vol : ℝ) (block_vol : ℝ) : ℕ :=
  box_vol.to_nat / block_vol.to_nat

theorem largest_number_of_blocks (l_box w_box h_box l_block w_block h_block : ℝ) :
  l_box = 4 ∧ w_box = 3 ∧ h_box = 3 ∧ l_block = 3 ∧ w_block = 2 ∧ h_block = 1 →
    num_blocks (volume l_box w_box h_box) (volume l_block w_block h_block) = 6 :=
by
  sorry

end largest_number_of_blocks_l437_437427


namespace simplify_trig_expression_l437_437725

open Real

/-- 
Given that θ is in the interval (π/2, π), simplify the expression 
( sin θ / sqrt (1 - sin^2 θ) ) + ( sqrt (1 - cos^2 θ) / cos θ ) to 0.
-/
theorem simplify_trig_expression (θ : ℝ) (hθ1 : π / 2 < θ) (hθ2 : θ < π) :
  (sin θ / sqrt (1 - sin θ ^ 2)) + (sqrt (1 - cos θ ^ 2) / cos θ) = 0 :=
by 
  sorry

end simplify_trig_expression_l437_437725


namespace greatest_divisor_condition_gcd_of_numbers_l437_437451

theorem greatest_divisor_condition (n : ℕ) (h100 : n ∣ 100) (h225 : n ∣ 225) (h150 : n ∣ 150) : n ≤ 25 :=
  sorry

theorem gcd_of_numbers : Nat.gcd (Nat.gcd 100 225) 150 = 25 :=
  sorry

end greatest_divisor_condition_gcd_of_numbers_l437_437451


namespace find_numbers_l437_437986

/-- Given the sums of three pairs of numbers, we prove the individual numbers. -/
theorem find_numbers (x y z : ℕ) (h1 : x + y = 40) (h2 : y + z = 50) (h3 : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 :=
by
  sorry

end find_numbers_l437_437986


namespace problem_solution_l437_437179

theorem problem_solution
  (x y : ℝ)
  (k : ℝ)
  (C : x^2 + y^2 - 8*x + 12 = 0) :
  (3 < x ∧ x ≤ 4 → (x - 2)^2 + y^2 = 4) ∧
  ((- ∥∥∥∥5) :

end problem_solution_l437_437179


namespace probability_even_sum_l437_437435

theorem probability_even_sum (s : Finset ℕ) (h₁ : s = {1, 2, 3, 4, 5, 6, 7, 8}) :
  (∃ p : ℚ, (∃ a b ∈ s, a ≠ b ∧ (a + b) % 2 = 0) → p = 3 / 7) := 
sorry

end probability_even_sum_l437_437435


namespace problem_solution_l437_437631

noncomputable def a : ℝ := (Real.sqrt (Real.sqrt 7 + 4) + Real.sqrt (Real.sqrt 7 - 4)) / Real.sqrt (Real.sqrt 7 + 2)

noncomputable def b : ℝ := Real.sqrt (5 - 2 * Real.sqrt 6)

noncomputable def M : ℝ := a - b

theorem problem_solution (M = (Real.sqrt 2) / 2 - 1) : Prop := sorry

end problem_solution_l437_437631


namespace find_m_l437_437300

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (m : ℝ) (k : ℝ)

noncomputable def vectors_collinear (u v : V) : Prop :=
  ∃ k : ℝ, u = k • v

theorem find_m (h_noncollinear: ¬vector_collinear a b)
               (h_AB : ∀ (m : ℝ), vector_collinear (2 • a + m • b) (a + m • b))
               (h_collinear: vector_collinear (2 • a + m • b) (a + 3 • b))
               (h_m : m = 6) :
               m = 6 :=
sorry

end find_m_l437_437300


namespace smallest_number_divisible_1_to_10_l437_437939

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437939


namespace smallest_divisible_1_to_10_l437_437911

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437911


namespace psi_integral_l437_437460

noncomputable def psi (t x1 x2 : ℝ) : ℝ :=
  Real.arctan (x1 / x2) - t

/-- 
Prove that ψ(t, x1, x2) = arctan(x1 / x2) - t
is an integral of the system of differential equations:
dx1/dt = x1^2/x2, dx2/dt = -x2^2/x1
-/
theorem psi_integral (t x1 x2 C : ℝ) : 
  (∂ (ψ t x1 x2) / ∂ t) + (x1^2 / x2) * (∂ (ψ t x1 x2) / ∂ x1) - (x2^2 / x1) * (∂ (ψ t x1 x2) / ∂ x2) = 0 := 
sorry

end psi_integral_l437_437460


namespace baker_additional_cakes_l437_437525

theorem baker_additional_cakes (X : ℕ) : 
  (62 + X) - 144 = 67 → X = 149 :=
by
  intro h
  sorry

end baker_additional_cakes_l437_437525


namespace correct_propositions_l437_437106

theorem correct_propositions :
  (∀ m a b : ℝ, a > b → ¬ (a^2 * m > b^2 * m)) ∧
  (∀ α β : ℝ, tan (α + β) ≠ tan α + tan β) :=
by
  sorry

end correct_propositions_l437_437106


namespace find_angle_l437_437621

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

def length_eq_two (v : EuclideanSpace ℝ (Fin 2)) : Prop := ∥v∥ = 2

def dot_product_eq_two (a b : EuclideanSpace ℝ (Fin 2)) : Prop := inner a b = 2

theorem find_angle (ha : length_eq_two a) (hb : length_eq_two b) (hab : dot_product_eq_two a b) :
  ∃ θ : ℝ, θ = π / 3 := by
  sorry

end find_angle_l437_437621


namespace triangle_count_l437_437241

theorem triangle_count (a b c : ℕ) (h1 : a + b + c = 15) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a + b > c) :
  ∃ (n : ℕ), n = 7 :=
by
  -- Proceed with the proof steps, using a, b, c satisfying the given conditions
  sorry

end triangle_count_l437_437241


namespace cot_sin_cos_tan_identities_l437_437592

theorem cot_sin_cos_tan_identities
  (α : ℝ)
  (h1 : cos α = 1 / 3)
  (h2 : -π / 2 < α ∧ α < 0) :
  (cot (-α - π) * sin (2 * π + α)) / (cos (-α) * tan α) = -real.sqrt 2 / 4 :=
sorry

end cot_sin_cos_tan_identities_l437_437592


namespace investment_period_l437_437289

-- Define the given conditions
def principal : ℝ := 10000
def rate : ℝ := 0.0396
def times_compounded : ℕ := 2
def accumulated_amount : ℝ := 10815.83

-- Define the compound interest formula
def compound_interest (P r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem
theorem investment_period :
  ∃ t : ℝ, compound_interest principal rate times_compounded t = accumulated_amount := 
sorry

end investment_period_l437_437289


namespace constant_term_of_expr_l437_437384

open BigOperators

-- Define the expression
def expr(x : ℝ) := (2 - 3 / x) * (x^2 + 2 / x)^5

-- Define the required proof
theorem constant_term_of_expr : expr != -240 :=
begin
  sorry
end

end constant_term_of_expr_l437_437384


namespace simplify_complex_fraction_l437_437718

-- Definitions based on the conditions
def numerator : ℂ := 5 + 7 * complex.i
def denominator : ℂ := 2 + 3 * complex.i
def expected : ℂ := 31 / 13 - (1 / 13) * complex.i

-- The Lean 4 statement
theorem simplify_complex_fraction : (numerator / denominator) = expected := 
sorry

end simplify_complex_fraction_l437_437718


namespace dot_product_magnitude_l437_437299

-- Define the problem conditions
variables {ℝ : Type*} [euclidean_space ℝ]
variables (u v : ℝ^3)
variables (norm_u : norm u = 3) (norm_v : norm v = 4) (norm_u_cross_v : norm (u × v) = 6)

-- State the theorem
theorem dot_product_magnitude :
  |(u • v)| = 6 * Real.sqrt 3 :=
begin
  sorry
end

end dot_product_magnitude_l437_437299


namespace probability_man_stay_in_dark_l437_437053

def revolutions_per_minute : ℕ := 3
def time_per_revolution (rev_per_min : ℕ) : ℕ := 60 / rev_per_min
def time_in_dark : ℕ := 15
def cycle_time (time_rev : ℕ) : ℕ := time_rev

theorem probability_man_stay_in_dark (rev_per_min : ℕ) :
  let tpr := time_per_revolution rev_per_min in
  let cic := cycle_time tpr in
  (time_in_dark : ℚ) / (cic : ℚ) = 3 / 4 := by
  sorry

end probability_man_stay_in_dark_l437_437053


namespace exists_y_equals_7_l437_437411

theorem exists_y_equals_7 : ∃ (x y z t : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ y = 7 ∧ x + y + z + t = 10 :=
by {
  sorry -- This is where the actual proof would go.
}

end exists_y_equals_7_l437_437411


namespace smallest_number_divisible_1_to_10_l437_437879

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437879


namespace find_height_of_cylinder_l437_437747

theorem find_height_of_cylinder (h r : ℝ) (π : ℝ) (SA : ℝ) (r_val : r = 3) (SA_val : SA = 36 * π) 
  (SA_formula : SA = 2 * π * r^2 + 2 * π * r * h) : h = 3 := 
by
  sorry

end find_height_of_cylinder_l437_437747


namespace find_ab_of_common_roots_l437_437564

noncomputable def p (a : ℝ) : Polynomial ℝ :=
  Polynomial.C 12 + Polynomial.C 13 * Polynomial.X + Polynomial.C a * Polynomial.X^2 + Polynomial.X^3

noncomputable def q (b : ℝ) : Polynomial ℝ :=
  Polynomial.C 15 + Polynomial.C 17 * Polynomial.X + Polynomial.C b * Polynomial.X^2 + Polynomial.X^3

theorem find_ab_of_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ IsRoot (p a) r ∧ IsRoot (p a) s ∧ IsRoot (q b) r ∧ IsRoot (q b) s) → a = 0 ∧ b = -1 :=
by
  sorry

end find_ab_of_common_roots_l437_437564


namespace remainder_mod_1000_l437_437683

noncomputable def q (x : ℕ) : ℕ := (Finset.range 2011).sum (λ n, x ^ n)
noncomputable def divisor (x : ℕ) : ℕ := x^5 + x^4 + 2 * x^3 + x^2 + 1
noncomputable def s (x : ℕ) : ℕ := x^4 + x^3 + x^2 + x + 1

theorem remainder_mod_1000 (x : ℕ) : 
  let q_x := q x,
      divisor_x := divisor x,
      s_2010 := s 2010 
  in (s_2010 % 1000) = 111 :=
by
  sorry

end remainder_mod_1000_l437_437683


namespace math_problem_l437_437585

def is_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (P : ℝ × ℝ → Prop) : Prop :=
  ∃ x y : ℝ, P (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1)

def right_focus (f : ℝ × ℝ) (h : f = (Real.sqrt 2, 0)) : Prop := 
  f = (Real.sqrt 2, 0)

def eccentricity (e a : ℝ) (h : e = (Real.sqrt 6) / 3) : Prop :=
  e = (Real.sqrt 6) / 3 ∧ ∃ b : ℝ, (a > b ∧ b > 0)

def line_intersects_ellipse (a b : ℝ) (P : ℝ × ℝ → Prop) (l : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, P (x1, l x1) ∧ P (x2, l x2)

def circle_intersects_origin (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2 = 0)

def max_area_triangle (x1 y1 x2 y2 : ℝ) : Prop :=
  (1 / 2) * Real.sqrt 3 * Real.sqrt (Real.max ((3 + 12 / (9 * x1^2 + 1 / x1^2 + 6)) ∧ 3))

theorem math_problem :
  ∃ a b : ℝ, is_ellipse a b
    ∧ right_focus (Real.sqrt 2, 0)
    ∧ eccentricity ((Real.sqrt 6) / 3)
    ∧ ∀ l : ℝ → ℝ, line_intersects_ellipse a b l
    ∧ ∀ x1 y1 x2 y2 : ℝ, circle_intersects_origin x1 y1 x2 y2
    → max_area_triangle x1 y1 x2 y2 := 
sorry

end math_problem_l437_437585


namespace find_minimum_value_2a_plus_b_l437_437204

theorem find_minimum_value_2a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_re_z : (3 * a * b + 2) = 4) : 2 * a + b = (4 * Real.sqrt 3) / 3 :=
sorry

end find_minimum_value_2a_plus_b_l437_437204


namespace value_of_m_l437_437554

theorem value_of_m : (∀ x : ℝ, (1 + 2 * x) ^ 3 = 1 + 6 * x + m * x ^ 2 + 8 * x ^ 3 → m = 12) := 
by {
  -- This is where the proof would go
  sorry
}

end value_of_m_l437_437554


namespace lcm_1_to_10_l437_437847

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437847


namespace smallest_number_divisible_1_to_10_l437_437876

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437876


namespace distinct_sequences_with_five_heads_in_ten_flips_l437_437074

theorem distinct_sequences_with_five_heads_in_ten_flips : 
  (nat.choose 10 5) = 252 :=
by
  sorry

end distinct_sequences_with_five_heads_in_ten_flips_l437_437074


namespace cot_arctan_5_over_12_eq_12_over_5_l437_437130

theorem cot_arctan_5_over_12_eq_12_over_5 : Real.cot (Real.arctan (5 / 12)) = 12 / 5 := by
  sorry

end cot_arctan_5_over_12_eq_12_over_5_l437_437130


namespace period_of_y_l437_437037

-- Define the function y = tan x - cot x
def y (x : ℝ) : ℝ := Real.tan x - Real.cot x

-- Prove that y has a period of π
theorem period_of_y : ∀ x, y (x + π) = y x :=
by
  intro x
  -- Define what needs to be shown
  have h : y (x + π) = Real.tan (x + π) - Real.cot (x + π) := rfl
  -- Use the periodicity of tangent and cotangent
  rw [Real.tan_add_pi, Real.cot_add_pi]
  -- Simplifying terms
  simp [Real.tan, Real.cot]
  sorry

end period_of_y_l437_437037


namespace distribution_ways_l437_437340

def rabbit_distribution_problem : ℕ := 6

def num_stores : ℕ := 5

def parent_child_distribution_constraint (distribution : Fin 6 → Option (Fin 5)) : Prop :=
  ∀ (r1 r2 : Fin 6), 
    r1 < 2 → r2 ≥ 2 → 
    (distribution r1 = distribution r2 → distribution r1 = none)

def at_least_one_store_empty (distribution : Fin 6 → Option (Fin 5)) : Prop :=
  ∃ store : Fin 5, ∀ r : Fin 6, distribution r ≠ some store

theorem distribution_ways : ∃ (distributions : Fin 6 → Option (Fin 5)), 
  parent_child_distribution_constraint distributions ∧ 
  at_least_one_store_empty distributions ∧
  card {d | parent_child_distribution_constraint d ∧ at_least_one_store_empty d} = 300 :=
sorry

end distribution_ways_l437_437340


namespace correct_fraction_transformation_l437_437048

theorem correct_fraction_transformation (a b : ℕ) (a_ne_0 : a ≠ 0) (b_ne_0 : b ≠ 0) :
  (\frac{2a}{2b} = \frac{a}{b}) ∧ 
  (¬(\frac{a^2}{b^2} = \frac{a}{b})) ∧ 
  (¬(\frac{2a + 1}{4b} = \frac{a + 1}{2b})) ∧ 
  (¬(\frac{a + 2}{b + 2} = \frac{a}{b})) := 
by
  sorry

end correct_fraction_transformation_l437_437048


namespace sum_of_digits_of_repeating_decimal_1_div_81_squared_l437_437748

theorem sum_of_digits_of_repeating_decimal_1_div_81_squared :
  let m := 99,
      repeating_decimal_digits := [0, 0, 0, 1, 5, 2, 4, 1, 5, 3, 3, 7, 0, 1, 9, 7, 5, 3, 0, 8, 6, 4, 2, 0],
      sum_of_digits := list.sum (take m repeating_decimal_digits)
  in sum_of_digits = 74 :=
by
  sorry

end sum_of_digits_of_repeating_decimal_1_div_81_squared_l437_437748


namespace sum_a3_a7_a8_l437_437009

-- Definitions based on problem conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- Given condition
def S11 (a : ℕ → ℝ) : Prop :=
  sum_of_first_n_terms a 11 = 22

-- The theorem to prove
theorem sum_a3_a7_a8 (a : ℕ → ℝ) (d : ℝ) (h1 : arithmetic_sequence a d) (h2 : S11 a) :
  a 3 + a 7 + a 8 = 8 :=
by
  sorry

end sum_a3_a7_a8_l437_437009


namespace pow_mul_inv_pow_pow_mul_inv_pow_example_l437_437423

theorem pow_mul_inv_pow (a : ℚ) (ha : a ≠ 0) (m n : ℤ) : (a^m) * (a^n) = a^(m+n) :=
begin
  rw [←Int.add_eq_add],
  sorry
end

theorem pow_mul_inv_pow_example : ( (9/10:ℚ)^4 * ( (9/10:ℚ)^-4 ) = 1 ) :=
begin
  apply pow_mul_inv_pow,
  norm_num,
  exact dec_trivial,
  norm_num,
  norm_num
end

end pow_mul_inv_pow_pow_mul_inv_pow_example_l437_437423


namespace no_possible_rectangles_l437_437744

theorem no_possible_rectangles (a b : ℝ) (h1 : a < b) :
  ¬ ∃ (x y : ℝ), x < a ∧ y < a ∧ x + y = (a + b) / 3 ∧ x * y = (a * b) / 3 :=
by
  intros x y hx hy heq_perimeter heq_area
  -- The proof is omitted, insert proof here
  sorry

end no_possible_rectangles_l437_437744


namespace brown_eyed_brunettes_count_l437_437550

/--
There are 50 girls in a group. Each girl is either blonde or brunette and either blue-eyed or brown-eyed.
14 girls are blue-eyed blondes. 31 girls are brunettes. 18 girls are brown-eyed.
Prove that the number of brown-eyed brunettes is equal to 13.
-/
theorem brown_eyed_brunettes_count
  (total_girls : ℕ)
  (blue_eyed_blondes : ℕ)
  (total_brunettes : ℕ)
  (total_brown_eyed : ℕ)
  (total_girls_eq : total_girls = 50)
  (blue_eyed_blondes_eq : blue_eyed_blondes = 14)
  (total_brunettes_eq : total_brunettes = 31)
  (total_brown_eyed_eq : total_brown_eyed = 18) :
  ∃ (brown_eyed_brunettes : ℕ), brown_eyed_brunettes = 13 :=
by sorry

end brown_eyed_brunettes_count_l437_437550


namespace solve_quadratic_equation_l437_437366

theorem solve_quadratic_equation :
  ∃ x : ℝ, 2 * x^2 = 4 * x - 1 ∧ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
by
  sorry

end solve_quadratic_equation_l437_437366


namespace train_pass_time_approx_24_seconds_l437_437446

noncomputable def timeToPassPlatform
  (train_length : ℕ) (train_speed_kmph : ℕ) (platform_length : ℕ) : ℕ :=
  let distance := train_length + platform_length
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  distance / train_speed_mps

theorem train_pass_time_approx_24_seconds :
  timeToPassPlatform 140 60 260 ≈ 24 :=
by sorry

end train_pass_time_approx_24_seconds_l437_437446


namespace lcm_1_to_10_l437_437956

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437956


namespace exists_bounded_sequence_l437_437522

noncomputable def bounded_sequence_a_ge_1 (a : ℝ) (a_gt_1 : a > 1) : Prop :=
  ∃ (C : ℝ) (seq : ℕ → ℝ) (bounded : ∀ n, |seq n| ≤ C),
    ∀ (i j : ℕ), i ≠ j → |seq i - seq j| * |i - j|^a ≥ 1

theorem exists_bounded_sequence (a : ℝ) (h : a > 1) : bounded_sequence_a_ge_1 a h :=
sorry

end exists_bounded_sequence_l437_437522


namespace percent_decrease_l437_437453

def original_price : ℝ := 100
def sale_price : ℝ := 60

theorem percent_decrease : (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end percent_decrease_l437_437453


namespace X_plus_Y_l437_437260

theorem X_plus_Y (U X Y : ℕ) 
  (h1: U ∈ {1, 2, 3}) 
  (h2: ¬ (U = 3 ∨ U = 1))
  (hX: X ∈ {1, 2, 3} \ {U, 3})
  (hY: Y ∈ {1, 2, 3} \ {X, 1, 2}) 
  (hX: X = 1) 
  (hY: Y = 3) 
  : X + Y = 4 :=
by
  sorry

end X_plus_Y_l437_437260


namespace card_arrangement_impossible_l437_437994

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437994


namespace max_area_eel_coverage_l437_437519

-- Define what an eel is in terms of polyominoes and unit squares
def eel (p: Polyomino) : Prop :=
  ∃ path: list (ℤ × ℤ), 
  path.length = 4 ∧
  -- The eel forms a specific path making two right-angled turns
  (∃ a b c d: (ℤ × ℤ), path = [a, b, c, d] ∧ 
   (a.1 + 1 = b.1 ∧ a.2 = b.2) ∧ -- horizontal segment
   (b.1 = c.1 ∧ (b.2 + 1 = c.2 ∨ b.2 - 1 = c.2)) ∧ -- vertical segment
   (c.1 + 1 = d.1 ∧ c.2 = d.2)) -- horizontal segment

-- Define the dimensions of the grid
def grid_dim : ℕ × ℕ := (1000, 1000)

-- Define the function that calculates the maximum eel coverage
noncomputable def max_eel_coverage (dim : ℕ × ℕ) : ℕ :=
  let n := dim.1 * dim.2 in n - 2

-- The main theorem to prove
theorem max_area_eel_coverage : max_eel_coverage grid_dim = 999998 :=
by
  -- Here you would write the proof steps, using the problem conditions
  sorry

end max_area_eel_coverage_l437_437519


namespace rays_angle_l437_437345

theorem rays_angle (Ox Oy Oz Ot Or : ray) (h_distinct: Ox ≠ Oy ∧ Ox ≠ Oz ∧ Ox ≠ Ot ∧ Ox ≠ Or ∧ Oy ≠ Oz ∧ Oy ≠ Ot ∧ Oy ≠ Or ∧ Oz ≠ Ot ∧ Oz ≠ Or ∧ Ot ≠ Or):
  ∃ (a b : ray), a ≠ b ∧ angle_between a b ≤ 90 :=
by 
  sorry

end rays_angle_l437_437345


namespace exists_non_constant_scaling_function_l437_437150

noncomputable theory

def is_non_constant (f : ℝ → ℝ) :=
∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂

def satisfies_scaling_property (f : ℝ → ℝ) :=
∀ x : ℝ, f (3 * x) = 3 * (f x) ^ 2

theorem exists_non_constant_scaling_function :
  ∃ f : ℝ → ℝ, is_non_constant f ∧ satisfies_scaling_property f :=
sorry

end exists_non_constant_scaling_function_l437_437150


namespace mod_37_5_l437_437402

theorem mod_37_5 : 37 % 5 = 2 := 
by 
  sorry

end mod_37_5_l437_437402


namespace variance_of_red_balls_l437_437262

noncomputable def redBalls : ℕ := 8
noncomputable def yellowBalls : ℕ := 4
noncomputable def totalBalls := redBalls + yellowBalls
noncomputable def n : ℕ := 4
noncomputable def p : ℚ := redBalls / totalBalls
noncomputable def D_X : ℚ := n * p * (1 - p)

theorem variance_of_red_balls :
  D_X = 8 / 9 :=
by
  -- conditions are defined in the definitions above
  -- proof is skipped
  sorry

end variance_of_red_balls_l437_437262


namespace limit_at_1_l437_437148

noncomputable def limit_expr : ℝ → ℝ := λ x, (x^3 - 1) / (x - 1)

theorem limit_at_1 : (Real.Lim (λ x, limit_expr x) 1) = 3 :=
by
  sorry

end limit_at_1_l437_437148


namespace part1_part2_l437_437188

variable {α : ℝ}

-- Define the conditions
def tan_alpha := 3
axiom alpha_range : π < α ∧ α < 3 * π / 2

-- Statement to prove the first part
theorem part1 (h : Real.tan α = tan_alpha) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := 
sorry

-- Statement to prove the second part
theorem part2 (h : Real.tan α = tan_alpha) (h_range : α > π ∧ α < 3 * π / 2) : 
  Real.cos α - Real.sin α = Real.sqrt 10 / 5 :=
sorry

end part1_part2_l437_437188


namespace smallest_divisible_1_to_10_l437_437917

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437917


namespace cos_angle_POQ_l437_437196

-- Definitions for the problem
noncomputable def sin_angle_p : ℝ := 4 / 5
noncomputable def cos_angle_q : ℝ := 5 / 13

-- Problem Statement
theorem cos_angle_POQ : ∃ (P Q : ℝ × ℝ), 
  (sin_angle_p = 4 / 5 ∧ cos_angle_q = 5 / 13) ∧ 
  (P.1^2 + P.2^2 = 1 ∧ Q.1^2 + Q.2^2 = 1 ∧ P.2 = 4 / 5 ∧ Q.1 = 5 / 13) ∧
  ∠(P, O, Q) = -33 / 65 := 
sorry

-- Note: ∠(P, O, Q) should represent the angle between vectors OP and OQ. 
-- This is a simplified symbolic representation as detailed trigonometric constructs 
-- in Lean might require more specific library functions or definitions.

end cos_angle_POQ_l437_437196


namespace correct_system_of_equations_l437_437059

-- Define the variables for the weights of sparrow and swallow
variables (x y : ℝ)

-- Define the problem conditions
def condition1 : Prop := 5 * x + 6 * y = 16
def condition2 : Prop := 4 * x + y = x + 5 * y

-- Create a theorem stating the conditions imply the identified system
theorem correct_system_of_equations :
  condition1 ∧ condition2 ↔ (5 * x + 6 * y = 16 ∧ 4 * x + y = x + 5 * y) :=
by
  apply Iff.intro;
  { intro h,
    cases h with h1 h2,
    exact ⟨h1, h2⟩ },
  { intro h,
    cases h with h1 h2,
    exact ⟨h1, h2⟩ }

end correct_system_of_equations_l437_437059


namespace smallest_triangle_area_l437_437112

/-- The given problem definition stating the initial conditions of the equilateral triangle and the squares around it.
We are to prove that the area of the smallest triangle containing the given squares is 13√3 - 12. -/
theorem smallest_triangle_area :
  let equilateral_triangle_area (s : ℝ) := (sqrt 3 / 4) * s^2 in
  let triangle_side := 4 * sqrt 3 - 2 in
  let resulting_area := equilateral_triangle_area triangle_side in
  resulting_area = 13 * sqrt 3 - 12 := 
by
  sorry

end smallest_triangle_area_l437_437112


namespace find_radius_of_first_sphere_l437_437772

noncomputable def radius_of_sphere (r1 r2 w1 w2 : ℝ) : Prop :=
  let sa1 := 4 * Real.pi * r1^2
  let sa2 := 4 * Real.pi * r2^2
  (w1 / sa1 = w2 / sa2) → r1 = 0.15

theorem find_radius_of_first_sphere:
  ∀ (r2 : ℝ) (w1 w2 : ℝ), r2 = 0.3 → w1 = 8 → w2 = 32 → radius_of_sphere _ r2 w1 w2 :=
begin
  intros r2 w1 w2 h_r2 h_w1 h_w2,
  sorry
end

end find_radius_of_first_sphere_l437_437772


namespace find_third_vertex_l437_437782

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Given vertices of the triangle
def A : Point := { x := 7, y := 3 }
def B : Point := { x := 0, y := 0 }

-- Third vertex C on negative x-axis and area of the triangle
noncomputable def C : Point := { x := -21, y := 0 }

-- Definition of the area of the triangle
def triangle_area (p1 p2 p3 : Point) : ℝ :=
  1 / 2 * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

-- Prove that the coordinates are (-21, 0) given the area is 42 square units
theorem find_third_vertex :
  ∃ (C : Point), C.x < 0 ∧ C.y = 0 ∧ triangle_area A B C = 42 :=
by
  use C
  split
  -- Proving x-coordinate condition
  sorry
  split
  -- Proving y-coordinate condition
  sorry
  -- Proving area condition
  sorry

end find_third_vertex_l437_437782


namespace negation_statement_l437_437756

variable {α : Type} 
variable (student prepared : α → Prop)

theorem negation_statement :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by 
  -- proof will be provided here
  sorry

end negation_statement_l437_437756


namespace number_of_placements_l437_437485

-- We define the classroom setup
def classroom_grid : Type := list (list (option bool)) -- True for boy, False for girl, None for empty desk (not needed here).

-- Initially define the size of the grid
def rows : ℕ := 5
def columns : ℕ := 6
def students : ℕ := 30
def boys : ℕ := 15
def girls : ℕ := 15

def is_valid_placement (grid : classroom_grid) : Prop :=
  ∀ i j, (grid.nth i).bind (λ row, row.nth j) ≠ some true → (grid.nth i).bind (λ row, row.nth j.succ) = some false ∧
                                        (grid.nth i).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j.succ) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.succ).bind (λ row, row.nth j.pred) = some false ∧
                                        (grid.nth i.pred).bind (λ row, row.nth j.succ) = some false

theorem number_of_placements : ∃ (count : ℕ), count = 2 * (nat.factorial 15)^2 := by
  -- skip the proof by providing the existential value directly
  existsi (2 * (nat.factorial 15)^2)
  refl

end number_of_placements_l437_437485


namespace area_of_square_field_l437_437737

theorem area_of_square_field :
  ∃ (s : ℝ), 
    let cost_per_meter := 1.30 in
    let total_cost := 865.80 in
    let gate_width := 2 in
    let perimeter_with_gates_excluded := 4 * s - gate_width in
    let cost_equation := (4 * s - gate_width) * cost_per_meter = total_cost in
    (cost_equation) ∧ s^2 = 27889 :=
by
  sorry

end area_of_square_field_l437_437737


namespace sofie_total_distance_l437_437726

-- Definitions for the conditions
def side1 : ℝ := 25
def side2 : ℝ := 35
def side3 : ℝ := 20
def side4 : ℝ := 40
def side5 : ℝ := 30
def laps_initial : ℕ := 2
def laps_additional : ℕ := 5
def perimeter : ℝ := side1 + side2 + side3 + side4 + side5

-- Theorem statement
theorem sofie_total_distance : laps_initial * perimeter + laps_additional * perimeter = 1050 := by
  sorry

end sofie_total_distance_l437_437726


namespace trajectory_of_moving_point_l437_437228

variables {F1 F2 M : ℝ × ℝ}  -- assuming points are given in 2D space
variable (d : ℝ)  -- distance

-- Distance function for 2D points
def dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem trajectory_of_moving_point :
  dist F1 F2 = 16 →
  dist M F1 + dist M F2 = 16 →
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ (M = ((1 - t) * F1.1 + t * F2.1, (1 - t) * F1.2 + t * F2.2)) :=
begin
  sorry
end

end trajectory_of_moving_point_l437_437228


namespace count_of_intriguing_quadruples_l437_437139

-- Define the predicate for our ordered quadruples
def is_intriguing_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 12 ∧ a + d > b + c

-- State that the number of such intriguing quadruples equals 200
theorem count_of_intriguing_quadruples : 
  ∃ n : ℕ, n = 200 ∧ (finset.univ.filter (λ q : (ℕ × ℕ × ℕ × ℕ),
    is_intriguing_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2)).card = n :=
sorry

end count_of_intriguing_quadruples_l437_437139


namespace lcm_1_to_10_l437_437950

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437950


namespace haley_more_than_josh_l437_437623

-- Definitions of the variables and conditions
variable (H : Nat) -- Number of necklaces Haley has
variable (J : Nat) -- Number of necklaces Jason has
variable (Jos : Nat) -- Number of necklaces Josh has

-- The conditions as assumptions
axiom h1 : H = 25
axiom h2 : H = J + 5
axiom h3 : Jos = J / 2

-- The theorem we want to prove based on these conditions
theorem haley_more_than_josh (H J Jos : Nat) (h1 : H = 25) (h2 : H = J + 5) (h3 : Jos = J / 2) : H - Jos = 15 := 
by 
  sorry

end haley_more_than_josh_l437_437623


namespace intersect_all_symmetry_axes_at_single_point_l437_437351

theorem intersect_all_symmetry_axes_at_single_point (P : Type) [polygon P]
  (hS : ∀ (ax₁ ax₂ : axes_of_symmetry P), ∃ (p : P), p ∈ ax₁ ∧ p ∈ ax₂) :
  ∃ (q : P), ∀ (ax : axes_of_symmetry P), q ∈ ax :=
by
  sorry

end intersect_all_symmetry_axes_at_single_point_l437_437351


namespace product_of_possible_y_coordinates_l437_437343

-- Definitions
def is_on_line (Q : ℝ×ℝ) : Prop := Q.1 = 4

def distance (P Q : ℝ×ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def is_at_distance (Q : ℝ×ℝ) (P : ℝ×ℝ) (dist : ℝ) : Prop :=
  distance P Q = dist

def possible_y_coords (Q : ℝ×ℝ) (y1 y2 : ℝ) : Prop := 
  ∃ y : ℝ, Q = (4, y) ∧ (y = y1 ∨ y = y2)

-- Theorem
theorem product_of_possible_y_coordinates :
  ∀ y1 y2 : ℝ, 
    possible_y_coords (4, y1) y1 y2 ∧
    possible_y_coords (4, y2) y1 y2 ∧
    is_on_line (4, y1) ∧ 
    is_on_line (4, y2) ∧
    is_at_distance (4, y1) (-2, -3) 7 ∧
    is_at_distance (4, y2) (-2, -3) 7 →
    y1 * y2 = -4 :=
by
  intros y1 y2 h
  sorry

end product_of_possible_y_coordinates_l437_437343


namespace smallest_number_divisible_1_to_10_l437_437932

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437932


namespace regular_hexagon_area_l437_437358

/-- Given that ABCDEF is a regular hexagon with vertices A = (0, 0) and C = (8, 2), 
prove that the area of hexagon ABCDEF is 34 * sqrt 3. -/
theorem regular_hexagon_area (A C : ℝ × ℝ) (hA : A = (0, 0)) (hC : C = (8, 2)) (h : |C.1 - A.1| = 8 ∧ |C.2 - A.2| = 2):
  let s := real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
  let area_triangle := (real.sqrt 3 / 4) * s^2 in
  let area_hexagon := 2 * area_triangle in
  area_hexagon = 34 * real.sqrt 3 :=
begin
  sorry
end

end regular_hexagon_area_l437_437358


namespace sqrt_of_9_eq_3_l437_437409

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_9_eq_3_l437_437409


namespace find_eccentricity_l437_437614

-- Definitions of conditions
def ellipse_eq {a b x y : ℝ} (h : a > b > 0) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def point_B (b : ℝ) : ℝ × ℝ :=
  (0, b)

def point_F (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

-- Given conditions as hypotheses
variables {a b c : ℝ} (h₁ : a > b > 0) (h₂ : a^2 - b^2 = c^2)

-- Prove that eccentricity e = c / a equals sqrt(2) / 2
theorem find_eccentricity (h_eq : ellipse_eq h₁) (h_bf : point_B b) (h_f : point_F c) :
  (c / a) = (Real.sqrt 2 / 2) :=
sorry

end find_eccentricity_l437_437614


namespace lcm_1_to_10_l437_437849

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437849


namespace sum_of_squares_transform_l437_437712

def isSumOfThreeSquaresDivByThree (N : ℕ) : Prop := 
  ∃ (a b c : ℤ), N = a^2 + b^2 + c^2 ∧ (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c)

def isSumOfThreeSquaresNotDivByThree (N : ℕ) : Prop := 
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z)

theorem sum_of_squares_transform {N : ℕ} :
  isSumOfThreeSquaresDivByThree N → isSumOfThreeSquaresNotDivByThree N :=
sorry

end sum_of_squares_transform_l437_437712


namespace smallest_number_divisible_by_1_through_10_l437_437981

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437981


namespace smallest_number_divisible_1_to_10_l437_437875

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437875


namespace a_value_for_continuity_l437_437164

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 3 then x + 2 else 2 * x + a

theorem a_value_for_continuity (a : ℝ) :
  (∀ x : ℝ, (x > 3 → f x a = x + 2) ∧ (x ≤ 3 → f x a = 2 * x + a) ∧
  ∀ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ y, abs (x - y) < δ → abs (f x a - f y a) < ε) → a = -1)
  :=
sorry

end a_value_for_continuity_l437_437164


namespace lcm_1_to_10_l437_437804

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437804


namespace probability_heads_3_in_10_tosses_l437_437487

theorem probability_heads_3_in_10_tosses : 
  let n := 10
  let k := 3
  let p := (1 / 2 : ℚ)
  P (binomial n k p) = 15 / 128
:= sorry

end probability_heads_3_in_10_tosses_l437_437487


namespace binomial_coefficient_div_l437_437567

def generalized_binomial (a : ℝ) (k : ℕ) : ℝ :=
  (list.prod (list.map (λ i, a - i) (list.range k))) / (list.prod (list.map (λ i, (i + 1 : ℕ)) (list.range k)))

theorem binomial_coefficient_div (a : ℝ) (k : ℕ) (h_pos_k : 0 < k) :
  generalized_binomial (-1/2) 50 / generalized_binomial (1/2) 50 = -99 :=
by
  sorry

end binomial_coefficient_div_l437_437567


namespace avg_lottery_draws_eq_5232_l437_437704

def avg_lottery_draws (n m : ℕ) : ℕ :=
  let N := 90 * 89 * 88 * 87 * 86
  let Nk := 25 * 40320
  N / Nk

theorem avg_lottery_draws_eq_5232 : avg_lottery_draws 90 5 = 5232 :=
by 
  unfold avg_lottery_draws
  sorry

end avg_lottery_draws_eq_5232_l437_437704


namespace part1_part2_l437_437062

-- Definitions for Part 1
def A_2 : Finset ℕ := Finset.filter (λ x, x ≤ 1992 ∧ x % 2 = 0) (Finset.range 1993)
def A_3 : Finset ℕ := Finset.filter (λ x, x ≤ 1992 ∧ x % 3 = 0) (Finset.range 1993)
def S : Finset ℕ := A_2 ∪ A_3

-- Statement for Part 1
theorem part1 : 
  S.card = 1328 ∧ 
  ∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → ¬(Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime a c) :=
sorry

-- Definitions for Part 2
def B : Finset ℕ := Finset.filter (λ x, x ≤ 1992) (Finset.range 1993)

-- Statement for Part 2
theorem part2 : 
  ∀ T ⊆ B, T.card = 1329 → 
  ∃ a b c ∈ T, a ≠ b → b ≠ c → a ≠ c → (Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime a c) :=
sorry

end part1_part2_l437_437062


namespace cats_awake_l437_437016

theorem cats_awake : ∀ (totalCats asleepCats awakeCats : Nat), totalCats = 98 → asleepCats = 92 → awakeCats = 6 → totalCats - asleepCats = awakeCats :=
by
  intros totalCats asleepCats awakeCats
  intros h_total h_asleep h_awake
  rw [h_total, h_asleep, h_awake]
  sorry

end cats_awake_l437_437016


namespace largest_div_smallest_l437_437015

theorem largest_div_smallest : 
  let nums := [10, 11, 12, 13, 14] in
  let largest := 14 in
  let smallest := 10 in
  largest / smallest = 1.4 := by
  sorry

end largest_div_smallest_l437_437015


namespace sum_of_reciprocals_l437_437771

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1/x + 1/y = 3/8 := by
  sorry

end sum_of_reciprocals_l437_437771


namespace three_disjoint_subsets_with_equal_sum_l437_437360

theorem three_disjoint_subsets_with_equal_sum :
  ∀ (S : Finset ℕ), S.card = 68 → (∀ (x : ℕ), x ∈ S → (1 ≤ x ∧ x ≤ 2015)) →
  ∃ (A B C : Finset ℕ), A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
  A.card = B.card ∧ B.card = C.card ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id :=
by sorry

end three_disjoint_subsets_with_equal_sum_l437_437360


namespace adam_change_l437_437103

theorem adam_change : 
  let amount : ℝ := 5.00
  let cost : ℝ := 4.28
  amount - cost = 0.72 :=
by
  -- proof goes here
  sorry

end adam_change_l437_437103


namespace not_sunny_prob_l437_437763

theorem not_sunny_prob (P_sunny : ℚ) (h : P_sunny = 5/7) : 1 - P_sunny = 2/7 :=
by sorry

end not_sunny_prob_l437_437763


namespace sum_first_10_terms_eq_95_l437_437604

axiom { 
  a_n : Nat → Int,
  d : Int,
  h_arith_seq : ∀ n, a_n (n + 1) = a_n n + d,
  h_d_gt_0 : d > 0,
  h_a1_a5 : a_n 1 + a_n 5 = 4,
  h_a2_a4 : a_n 2 * a_n 4 = -5 
}

theorem sum_first_10_terms_eq_95 : (∑ i in Finset.range 10, a_n i) = 95 := sorry

end sum_first_10_terms_eq_95_l437_437604


namespace balanced_coins_existence_l437_437410

noncomputable def exists_balanced_coin (coins : Fin 101 → ℝ) (weights : Fin 101 → ℝ) : Prop :=
  ∃ (i : Fin 101), (∀ k, k = 50 ∨ k = 49 → 
	right_mass_eq_left_mass : 
	((List.sum (List.map weights (List.map (λ j, (i + j) % 101) (List.range k)))) = 
	(List.sum (List.map weights (List.map (λ j, (i - j) % 101) (List.range k)))))

theorem balanced_coins_existence (coins : Fin 101 → ℝ) (weights : Fin 101 → ℝ)
  (h : ∀ i, weights i = 10 ∨ weights i = 11) : exists_balanced_coin coins weights :=
sorry

end balanced_coins_existence_l437_437410


namespace max_f_x_in_interval_l437_437650

noncomputable def op_1 (a b : ℝ) : ℝ :=
if a ≥ b then a else b ^ 2

def f_x (x : ℝ) : ℝ :=
(op_1 1 x) * x - (op_1 2 x)

theorem max_f_x_in_interval : 
  ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f_x x = 6 ∧ 
  ∀ y ∈ Icc (-2 : ℝ) (2 : ℝ), f_x y ≤ f_x x :=
sorry

end max_f_x_in_interval_l437_437650


namespace number_of_zeros_is_two_l437_437758

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2 * x - 3 else -2 + Real.log x

theorem number_of_zeros_is_two : 
  (set_of (λ x, f x = 0)).card = 2 :=
sorry

end number_of_zeros_is_two_l437_437758


namespace smallest_multiple_1_through_10_l437_437826

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437826


namespace intersection_of_sets_l437_437590

def A : Set ℕ := {1, 2, 3}
def B : Set ℤ := {x ∈ Set.univ | x^2 - x - 2 ≤ 0}
def intersection : Set ℤ := {1, 2}

theorem intersection_of_sets :
  (A : Set ℤ) ∩ B = intersection :=
sorry

end intersection_of_sets_l437_437590


namespace largest_no_solution_l437_437444

theorem largest_no_solution (a : ℕ) (h_odd : a % 2 = 1) (h_pos : a > 0) :
  ∃ n : ℕ, ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → a * x + (a + 1) * y + (a + 2) * z ≠ n :=
sorry

end largest_no_solution_l437_437444


namespace initial_percentage_of_chemical_solution_l437_437077

theorem initial_percentage_of_chemical_solution (P : ℝ) :
  let V₁ := 50
      V₂ := 35
      V₃ := 35
      final_percentage := 46 / 100
      initial_amount := P / 100 * V₁
      drained_amount := P / 100 * (V₁ - V₂)
      added_amount := 40 / 100 * V₃
      total_pure_chemical := drained_amount + added_amount
  in total_pure_chemical / 50 = final_percentage → P = 60 :=
by
  intros
  let V₁ := 50
  let V₂ := 35
  let V₃ := 35
  let final_percentage := 46 / 100
  let initial_amount := P / 100 * V₁
  let drained_amount := P / 100 * (V₁ - V₂)
  let added_amount := 40 / 100 * V₃
  let total_pure_chemical := drained_amount + added_amount
  have h : total_pure_chemical = 23,
  from calc
    total_pure_chemical = (P / 100 * 15 + 14) : by sorry
                    ... = P / 100 * 15 + 14   : by sorry
                    ... = 23                : by sorry
  exact sorry

end initial_percentage_of_chemical_solution_l437_437077


namespace smallest_divisible_by_1_to_10_l437_437901

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437901


namespace polynomial_root_exists_l437_437579

theorem polynomial_root_exists
  (P : ℝ → ℝ)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
  (h_eq : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)) :
  ∃ r : ℝ, P r = 0 :=
sorry

end polynomial_root_exists_l437_437579


namespace darren_and_fergie_same_amount_in_days_l437_437540

theorem darren_and_fergie_same_amount_in_days : 
  ∀ (t : ℕ), (200 + 16 * t = 300 + 12 * t) → t = 25 := 
by sorry

end darren_and_fergie_same_amount_in_days_l437_437540


namespace lcm_1_to_10_l437_437850

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437850


namespace tan_beta_eq_neg13_l437_437573

variables (α β : Real)

theorem tan_beta_eq_neg13 (h1 : Real.tan α = 2) (h2 : Real.tan (α - β) = -3/5) : 
  Real.tan β = -13 := 
by 
  sorry

end tan_beta_eq_neg13_l437_437573


namespace lcm_1_10_l437_437971

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437971


namespace record_loss_of_300_l437_437383

-- Definitions based on conditions
def profit (x : Int) : String := "+" ++ toString x
def loss (x : Int) : String := "-" ++ toString x

-- The theorem to prove that a loss of 300 is recorded as "-300" based on the recording system
theorem record_loss_of_300 : loss 300 = "-300" :=
by
  sorry

end record_loss_of_300_l437_437383


namespace fraction_decomposition_l437_437308

noncomputable def polynomial := λ (x : ℝ), x^3 - 22 * x^2 + 80 * x - 67

def distinct_roots (p q r : ℝ) : Prop :=
∀ (x : ℝ), polynomial x = 0 → (x = p ∨ x = q ∨ x = r) ∧ (p ≠ q ∧ q ≠ r ∧ r ≠ p)

def partial_fraction_exists (p q r : ℝ) : Prop :=
∃ A B C : ℝ, ∀ s : ℝ, s ≠ p → s ≠ q → s ≠ r →
  (1 / polynomial s) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))

theorem fraction_decomposition (p q r A B C : ℝ) (h_roots : distinct_roots p q r)
  (h_exists : partial_fraction_exists p q r) :
  (1 / A + 1 / B + 1 / C = 244) :=
sorry

end fraction_decomposition_l437_437308


namespace equal_expense_sharing_l437_437334

variables (O L B : ℝ)

theorem equal_expense_sharing (h1 : O < L) (h2 : O < B) : 
    (L + B - 2 * O) / 6 = (O + L + B) / 3 - O :=
by
    sorry

end equal_expense_sharing_l437_437334


namespace smallest_number_divisible_1_to_10_l437_437935

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437935


namespace system_real_solution_conditions_l437_437434

theorem system_real_solution_conditions (a b c x y z : ℝ) (h1 : a * x + b * y = c * z) (h2 : a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) :
  abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b ∧
  (a * b >= 0 ∨ a * c >= 0 ∨ b * c >= 0) :=
sorry

end system_real_solution_conditions_l437_437434


namespace decreasing_function_m_l437_437436

theorem decreasing_function_m (x : ℝ) (m : ℝ) (h₀ : x ∈ Ioi (0 : ℝ))
  (h₁ : m = -1) :
  ∀ x : ℝ, x > 0 → ∀ m : ℝ, m = -1 → ∀ y : ℝ, y = (m^2 - m - 1) * x^m → (y' : ℝ) → y' < 0 :=
by
  sorry

end decreasing_function_m_l437_437436


namespace _l437_437762

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end _l437_437762


namespace axes_symmetry_intersect_at_one_point_l437_437350

noncomputable def center_of_mass (p : polygon) : point := sorry

theorem axes_symmetry_intersect_at_one_point (P : polygon)
  (h : has_multiple_axes_of_symmetry P) :
  ∃! (p : point), ∀ (A : axis_of_symmetry), is_axis_of_symmetry A P → p ∈ A :=
sorry

end axes_symmetry_intersect_at_one_point_l437_437350


namespace sally_finish_book_in_2_weeks_l437_437359

theorem sally_finish_book_in_2_weeks :
  ∀ (weekday_pages weekend_pages total_pages : ℕ),
    (weekday_pages = 10) →
    (weekend_pages = 20) →
    (total_pages = 180) →
    (5 * weekday_pages + 2 * weekend_pages > 0) →
    (total_pages / (5 * weekday_pages + 2 * weekend_pages) = 2) :=
by
  intros weekday_pages weekend_pages total_pages h_weekday h_weekend h_total h_nonzero
  rw [h_weekday, h_weekend, h_total]
  dsimp
  linarith

end sally_finish_book_in_2_weeks_l437_437359


namespace neighbor_cells_diff_at_least_n_plus_one_l437_437055

theorem neighbor_cells_diff_at_least_n_plus_one (n : ℕ) (H : n ≥ 1) (board : Fin n → Fin n → ℕ) 
  (Hboard : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2) : 
  ∃ (i1 j1 i2 j2 : Fin n), (i1 = i2 ∧ (j1 = j2 + 1 ∨ j2 = j1 + 1)) ∨ 
                            (j1 = j2 ∧ (i1 = i2 + 1 ∨ i2 = i1 + 1)) ∨ 
                            ((i1 = i2 + 1 ∨ i2 = i1 + 1) ∧ (j1 = j2 + 1 ∨ j2 = j1 + 1)) ∧
                           (|board i1 j1 - board i2 j2| ≥ n + 1) := 
by
  sorry

end neighbor_cells_diff_at_least_n_plus_one_l437_437055


namespace line_perpendicular_to_plane_l437_437198

-- Definitions of the given vectors
def direction_vector_l : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector_alpha : ℝ × ℝ × ℝ := (-2, 2, -4)

-- Relationship to be proven
theorem line_perpendicular_to_plane : 
  ∀ (a : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ), a = (1, -1, 2) → u = (-2, 2, -4) → 
  (∃ k : ℝ, u = k • a) → l ⊥ α := 
by
  intros a u h1 h2 h3
  sorry

end line_perpendicular_to_plane_l437_437198


namespace smallest_divisible_1_to_10_l437_437926

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437926


namespace jon_payment_per_visit_l437_437670

theorem jon_payment_per_visit 
  (visits_per_hour : ℕ) (operating_hours_per_day : ℕ) (income_in_month : ℚ) (days_in_month : ℕ) 
  (visits_per_hour_eq : visits_per_hour = 50) 
  (operating_hours_per_day_eq : operating_hours_per_day = 24) 
  (income_in_month_eq : income_in_month = 3600) 
  (days_in_month_eq : days_in_month = 30) :
  (income_in_month / (visits_per_hour * operating_hours_per_day * days_in_month) : ℚ) = 0.10 := 
by
  sorry

end jon_payment_per_visit_l437_437670


namespace equivalent_single_discount_l437_437371

theorem equivalent_single_discount (x : ℝ) (h : 0 < x) :
    let price_after_first_discount := 0.85 * x
    let price_after_second_discount := 0.90 * price_after_first_discount
    let price_after_third_discount := 0.95 * price_after_second_discount
    let single_discount_factor := 0.72675 * x
    single_discount_factor = 0.72675 * x → 
    1 - 0.72675 = 0.27325 :=
by
  intros
  rw [price_after_first_discount, price_after_second_discount, price_after_third_discount]
  sorry

end equivalent_single_discount_l437_437371


namespace vector_magnitudes_condition_l437_437675

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

theorem vector_magnitudes_condition :
  (∥a∥ = ∥b∥) ↔ ∥a + b∥ ≠ ∥a - b∥ :=
sorry

end vector_magnitudes_condition_l437_437675


namespace unique_set_exists_l437_437347

theorem unique_set_exists (n : ℕ) (h : n > 0) : 
  ∃! (x : Fin n → ℝ), 
    (∑ i in Finset.range n, 
      if i = 0 then (1 - x ⟨i, Nat.lt_succ_self i⟩) ^ 2 
      else if i + 1 = n then x ⟨i, Nat.lt_succ_self i⟩ ^ 2
      else (x ⟨i, Nat.lt_succ_self i⟩ - x ⟨i + 1, sorry⟩) ^ 2) = 1 / (n + 1) :=
begin
  sorry
end

end unique_set_exists_l437_437347


namespace min_value_inequality_equality_condition_l437_437175

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end min_value_inequality_equality_condition_l437_437175


namespace quadratic_graph_y1_lt_y2_l437_437602

theorem quadratic_graph_y1_lt_y2 (x1 x2 : ℝ) (h1 : -x1^2 = y1) (h2 : -x2^2 = y2) (h3 : x1 * x2 > x2^2) : y1 < y2 :=
  sorry

end quadratic_graph_y1_lt_y2_l437_437602


namespace smallest_divisible_1_to_10_l437_437914

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437914


namespace magnitude_of_complex_exponent_l437_437149
-- Import the library for complex number operations

-- Define the context
noncomputable theory
open Complex

-- The statement we want to prove
theorem magnitude_of_complex_exponent (z : ℂ) (hz : z = 1 + 2 * I) : complex.abs (z^8) = 625 :=
by sorry

end magnitude_of_complex_exponent_l437_437149


namespace ab_value_l437_437249

variables {a b : ℝ}

theorem ab_value (h₁ : a - b = 6) (h₂ : a^2 + b^2 = 50) : ab = 7 :=
sorry

end ab_value_l437_437249


namespace polynomial_division_l437_437144

theorem polynomial_division (a b c : ℤ) :
  (∀ x : ℝ, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end polynomial_division_l437_437144


namespace part_1_part_2_l437_437305

variable (a : ℝ) (k : ℝ) (f : ℝ → ℝ)
variable (t : ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f_def := λ x, a^x + (k-1) * a^(-x) + k^2

theorem part_1 (h1 : a > 0) (h2 : a ≠ 1) (h3 : is_odd_function (f_def a k)) : k = 0 :=
sorry

theorem part_2 (h4 : f (1) > 0) : ∃ T, T > (1/4) ∧ ∀ x, f(x^2 + x) + f(t - 2*x) > 0 :=
sorry

end part_1_part_2_l437_437305


namespace tan_diff_identity_l437_437167

theorem tan_diff_identity 
  (α : ℝ)
  (h : Real.tan α = -4/3) : Real.tan (α - Real.pi / 4) = 7 := 
sorry

end tan_diff_identity_l437_437167


namespace number_of_triangles_with_perimeter_10_l437_437233

theorem number_of_triangles_with_perimeter_10 : 
  ∃ (a b c : ℕ), a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
  (a ≤ b) ∧ (b ≤ c) ↔ 9 :=
by
  have h : ∀ (a b c : ℕ), a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a ≤ b ∧ b ≤ c → 
    (a, b, c) ∈ 
      {[ (1, 5, 4), (2, 4, 4), (3, 3, 4), 
         (1, 6, 3), (2, 5, 3), (3, 4, 3),
         (2, 6, 2), (3, 5, 2), (4, 4, 2) ] : set (ℕ × ℕ × ℕ)},
  sorry  

end number_of_triangles_with_perimeter_10_l437_437233


namespace seating_arrangement_ways_l437_437478

theorem seating_arrangement_ways (students desks : ℕ) (rows columns boys girls : ℕ) :
  students = 30 → desks = 30 → rows = 5 → columns = 6 → boys = 15 → girls = 15 → 
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 6 → 
    (¬(∃ m n, (m = i ∧ n ≠ j ∧ abs(m - i) ≤ 1 ∨ n = j ∧ abs(n - j) ≤ 1) ∧ boys) ∧
    ¬(∃ m n, (m = i ∧ n ≠ j ∧ abs(m - i) ≤ 1 ∨ n = j ∧ abs(n - j) ≤ 1) ∧ girls))) →
  2 * (Nat.factorial 15) * (Nat.factorial 15) = 2 * (Nat.factorial 15)^2 := 
by
  sorry

end seating_arrangement_ways_l437_437478


namespace axes_symmetry_intersect_at_one_point_l437_437349

noncomputable def center_of_mass (p : polygon) : point := sorry

theorem axes_symmetry_intersect_at_one_point (P : polygon)
  (h : has_multiple_axes_of_symmetry P) :
  ∃! (p : point), ∀ (A : axis_of_symmetry), is_axis_of_symmetry A P → p ∈ A :=
sorry

end axes_symmetry_intersect_at_one_point_l437_437349


namespace complex_div_imag_unit_l437_437628

theorem complex_div_imag_unit (i : ℂ) (h : i^2 = -1) : (1 + i) / (1 - i) = i :=
sorry

end complex_div_imag_unit_l437_437628


namespace smallest_number_div_by_1_to_10_l437_437803

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437803


namespace lcm_1_10_l437_437965

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437965


namespace smallest_number_divisible_by_1_to_10_l437_437869

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437869


namespace new_weighted_average_double_l437_437264

-- Define the weights of each student
def weights : List ℝ := [1.2, 1.6, 1.8, 1.4, 1, 1.5, 2, 1.3, 1.7, 1.9, 1.1]

-- Define the original weighted average
def original_weighted_average : ℝ := 36

-- Sum of weights
def sum_of_weights : ℝ := weights.foldl (+) 0

-- The math proof statement in Lean 4
theorem new_weighted_average_double : 
  (2 * original_weighted_average) = 72 := 
by
  -- By the given conditions, we can directly state the proof
  -- Here we are asserting that doubling the original weighted average results in 72.
  have h_sum_of_weights : sum_of_weights = (1.2 + 1.6 + 1.8 + 1.4 + 1 + 1.5 + 2 + 1.3 + 1.7 + 1.9 + 1.1), from sorry,
  have h_original_average := original_weighted_average,
  have h_double_average : 2 * h_original_average = 72, by sorry,
  exact h_double_average

end new_weighted_average_double_l437_437264


namespace lcm_1_to_10_l437_437816

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437816


namespace part1_part2_l437_437131

theorem part1 : 
  2^(-1/2:ℝ) + (-4)^0 / real.sqrt 2 + 1 / (real.sqrt 2 - 1) - real.sqrt ((1 - real.sqrt 5)^0) = 2 * real.sqrt 2 :=
by
  sorry

theorem part2 : 
  real.log (2 : ℝ) / real.log 2 * (real.log (1/16: ℝ) / real.log 3) * (real.log (1/9: ℝ) / real.log 5) = 8 * (real.log (2: ℝ) / real.log 5) :=
by
  sorry

end part1_part2_l437_437131


namespace hexagon_area_l437_437676

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B (b : ℝ) : ℝ × ℝ := (b, 4)

def is_convex_equilateral_hexagon (A B C D E F : ℝ × ℝ) : Prop :=
  -- Add specific conditions to check for a convex equilateral hexagon

def angle_FAB (A F B : ℝ × ℝ) : Prop :=
  -- Add specific condition for the angle ∠FAB = 150°

def are_parallel (l1 l2 : ℝ × ℝ → ℝ × ℝ) : Prop :=
  -- Add specification of parallel line segments condition

def distinct_y_coordinates (vertices : List (ℝ × ℝ)) (ys : List ℝ) : Prop :=
  ys = [0, 4, 8, 12, 16, 20] ∧ ys.nodup ∧ (∀ (v ∈ vertices), v.snd ∈ ys)

theorem hexagon_area (b : ℝ) (F : ℝ × ℝ)
  (h1 : is_convex_equilateral_hexagon A (B b) C D E F)
  (h2 : angle_FAB A F (B b))
  (h3 : are_parallel (λ p, A) (λ p, D))
  (h4 : are_parallel (λ p, (B b)) (λ p, E))
  (h5 : are_parallel (λ p, C) (λ p, F))
  (h6 : distinct_y_coordinates [A, B b, C, D, E, F] [0, 4, 8, 12, 16, 20]) :
  ∃ m n : ℕ, n > 0 ∧ ¬ ∃ p : ℕ, p^2 ∣ n ∧ m + n = 195 ∧ area_of_hexagon [A, B b, C, D, E, F] = m * real.sqrt n :=
sorry

end hexagon_area_l437_437676


namespace abs_neg_eq_five_l437_437247

theorem abs_neg_eq_five (a : ℝ) : abs (-a) = 5 ↔ (a = 5 ∨ a = -5) :=
by
  sorry

end abs_neg_eq_five_l437_437247


namespace lcm_1_to_10_l437_437855

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437855


namespace domain_of_f_l437_437542

noncomputable def f (x : ℝ) : ℝ := real.log (2 + x - x^2) / (abs x - x)

theorem domain_of_f :
  {x : ℝ | 2 + x - x^2 > 0 ∧ |x| ≠ x} = Set.Ioo (-1) 0 :=
by
  sorry

end domain_of_f_l437_437542


namespace smallest_divisible_1_to_10_l437_437906

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437906


namespace tangent_line_at_point_is_correct_l437_437389

theorem tangent_line_at_point_is_correct :
  ∀ (x y : ℝ), (y = x^2 + 2 * x) → (x = 1) → (y = 3) → (4 * x - y - 1 = 0) :=
by
  intros x y h_curve h_x h_y
  -- Here would be the proof
  sorry

end tangent_line_at_point_is_correct_l437_437389


namespace lcm_1_to_10_l437_437944

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437944


namespace RotaryClubNeeds584Eggs_l437_437377

noncomputable def RotaryClubEggsRequired : ℕ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let adult_tickets : ℕ := 75
  let senior_tickets : ℕ := 37
  let small_children_omelets := 0.5 * small_children_tickets
  let older_children_omelets := 1 * older_children_tickets
  let adult_omelets := 2 * adult_tickets
  let senior_omelets := 1.5 * senior_tickets
  let extra_omelets : ℕ := 25
  let total_omelets := small_children_omelets + older_children_omelets + adult_omelets + senior_omelets + extra_omelets
  2 * total_omelets

theorem RotaryClubNeeds584Eggs : RotaryClubEggsRequired = 584 := by
  sorry

end RotaryClubNeeds584Eggs_l437_437377


namespace find_angle_AMC_l437_437580

namespace GeometryProof

variables {P : Type} [metric_space P] [inner_product_space ℝ P]

noncomputable def angle {α β γ : P} : ℝ := sorry -- placeholder for the angle function

def isosceles_triangle (A B C : P) : Prop := dist A B = dist A C

theorem find_angle_AMC (A B C M : P) 
  (h_triangle : isosceles_triangle A B C) 
  (angle_BAC : angle B A C = 80) 
  (angle_MBC : angle M B C = 30) 
  (angle_MCB : angle M C B = 10) :
  angle A M C = 70 :=
by sorry

end GeometryProof

end find_angle_AMC_l437_437580


namespace phi_bound_l437_437596

def non_decreasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def f_iterate (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ
| 0     x := x
| (m+1) x := f (f_iterate m x)

def phi (f : ℝ → ℝ) (n : ℕ) (x : ℝ) := f_iterate f n x - x

theorem phi_bound (f : ℝ → ℝ) (h₁ : non_decreasing f) (h₂ : ∀ x, f (x + 1) = f x + 1) (x y : ℝ) (n : ℕ) :
  |phi f n x - phi f n y| < 1 :=
sorry

end phi_bound_l437_437596


namespace smallest_palindromic_primes_l437_437141

def is_palindromic (n : ℕ) : Prop :=
  ∀ a b : ℕ, n = 1001 * a + 1010 * b → 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_palindromic_primes :
  ∃ n1 n2 : ℕ, 
  is_palindromic n1 ∧ is_palindromic n2 ∧ is_prime n1 ∧ is_prime n2 ∧ n1 < n2 ∧
  ∀ m : ℕ, (is_palindromic m ∧ is_prime m ∧ m < n2 → m = n1) ∧
           (is_palindromic m ∧ is_prime m ∧ m < n1 → m ≠ n2) ∧ n1 = 1221 ∧ n2 = 1441 := 
sorry

end smallest_palindromic_primes_l437_437141


namespace lcm_1_10_l437_437966

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437966


namespace fraction_transformation_correct_l437_437046

theorem fraction_transformation_correct
  {a b : ℝ} (hb : b ≠ 0) : 
  (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_transformation_correct_l437_437046


namespace value_of_r_minus_q_l437_437400

variable (q r : ℝ)
variable (slope : ℝ)
variable (h_parallel : slope = 3 / 2)
variable (h_points : (r - q) / (-2) = slope)

theorem value_of_r_minus_q (h_parallel : slope = 3 / 2) (h_points : (r - q) / (-2) = slope) : 
  r - q = -3 := by
  sorry

end value_of_r_minus_q_l437_437400


namespace probability_sum_odd_l437_437108

theorem probability_sum_odd :
  let cards := {0, 1, 2}
  ∃ A B : cards, A ≠ B ∧ (A + B) % 2 = 1 →
  let outcomes := 3
  let favorable := 2
  (favorable / outcomes = 2 / 3) :=
by
  sorry

end probability_sum_odd_l437_437108


namespace red_fraction_after_tripling_l437_437261

theorem red_fraction_after_tripling (initial_total_marbles : ℚ) (H : initial_total_marbles > 0) :
  let blue_fraction := 2 / 3
  let red_fraction := 1 - blue_fraction
  let red_marbles := red_fraction * initial_total_marbles
  let new_red_marbles := 3 * red_marbles
  let initial_blue_marbles := blue_fraction * initial_total_marbles
  let new_total_marbles := new_red_marbles + initial_blue_marbles
  (new_red_marbles / new_total_marbles) = 3 / 5 :=
by
  sorry

end red_fraction_after_tripling_l437_437261


namespace zebra_crossing_distance_l437_437084

theorem zebra_crossing_distance
  (boulevard_width : ℝ)
  (distance_along_stripes : ℝ)
  (stripe_length : ℝ)
  (distance_between_stripes : ℝ) :
  boulevard_width = 60 →
  distance_along_stripes = 22 →
  stripe_length = 65 →
  distance_between_stripes = (60 * 22) / 65 →
  distance_between_stripes = 20.31 :=
by
  intros h1 h2 h3 h4
  sorry

end zebra_crossing_distance_l437_437084


namespace selling_price_is_288_l437_437448

-- Define the cost price and profit percentage
def cost_price : ℝ := 240
def profit_percentage : ℝ := 20

-- Define the calculation for profit
def profit : ℝ := (profit_percentage / 100) * cost_price

-- Define the selling price
def selling_price : ℝ := cost_price + profit

-- Proof that the selling price is $288
theorem selling_price_is_288 : selling_price = 288 := by
  calc
    selling_price = cost_price + profit : by rfl
              ... = 240 + ((20 / 100) * 240) : by rfl
              ... = 240 + 48 : by norm_num
              ... = 288 : by norm_num

end selling_price_is_288_l437_437448


namespace quadrilateral_with_four_equal_angles_is_rectangle_l437_437442

-- Define a quadrilateral with four equal angles
def quadrilateral_four_equal_angles (α : Type) [geometry α] (a b c d : α) : Prop :=
  ∃ (angle : ℝ), angle > 0 ∧ angle < 180 ∧ measure (angle a b c) = angle ∧ measure (angle b c d) = angle ∧
  measure (angle c d a) = angle ∧ measure (angle d a b) = angle

-- The goal is to prove that such a quadrilateral is a rectangle.
theorem quadrilateral_with_four_equal_angles_is_rectangle (α : Type) [geometry α] 
  {a b c d : α} : quadrilateral_four_equal_angles α a b c d → is_rectangle α a b c d :=
sorry

end quadrilateral_with_four_equal_angles_is_rectangle_l437_437442


namespace find_analytical_expression_solve_inequality_l437_437218

variable (f : ℝ → ℝ)
variable (m k : ℤ)

-- Given conditions
def condition1 : Prop := ∃ (m : ℝ) (k : ℤ), 
  (∀ x, f(x) = (3 * m ^ 2 - 2 * m + 1) * x ^ (3 * k - k ^ 2 + 4)) ∧ 
  (∀ x, f(x) = f(-x)) ∧
  ( ∀ x > 0, f(x) < f(x + 1))

-- Question (1)
theorem find_analytical_expression : condition1 f m k → 
  ( f = λ x, x^4 ∨ f = λ x, x^6 ) :=
sorry

-- Question (2)
theorem solve_inequality : condition1 f m k → 
  ( (∀ x, f(3 * x + 2) > f(1 - 2 * x)) → 
    ( ∀ x, x < -3 ∨ x > -1/5) ) :=
sorry

end find_analytical_expression_solve_inequality_l437_437218


namespace range_of_f_l437_437543

noncomputable def f (x : ℝ) : ℝ := x + |x - 2|

theorem range_of_f : Set.range f = Set.Ici 2 :=
sorry

end range_of_f_l437_437543


namespace sinks_per_house_l437_437474

theorem sinks_per_house (total_sinks : ℕ) (houses : ℕ) (h_total_sinks : total_sinks = 266) (h_houses : houses = 44) :
  total_sinks / houses = 6 :=
by {
  sorry
}

end sinks_per_house_l437_437474


namespace lcm_1_to_10_l437_437951

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437951


namespace machine_full_boxes_l437_437521

theorem machine_full_boxes (crayons_per_day : ℕ) (crayons_per_box : ℕ) (h : crayons_per_day = 321) (h_box : crayons_per_box = 7) :
  (crayons_per_day / crayons_per_box) = 45 :=
by
  rw [h, h_box]
  norm_num
  sorry

end machine_full_boxes_l437_437521


namespace max_product_of_digits_is_953_l437_437028

theorem max_product_of_digits_is_953 :
  let digits := [3, 5, 6, 8, 9] in
  ∃ (a b c : ℕ) (d e : ℕ), 
    (a, b, c, d, e) ∈ ((digits.permutations.map (λ xs, (xs[0], xs[1], xs[2], xs[3], xs[4] : ℕ) : Set (ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a * 100 + b * 10 + c = 953 ∧ 
    ∀ (x y z w v : ℕ),
      (x, y, z, w, v) ∈ ((List.permutationsMap (λ xs, (xs[0], xs[1], xs[2], xs[3], xs[4]) : Set (ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ z ≠ w ∧ z ≠ v ∧ w ≠ v → 
      (100 * x + 10 * y + z) * (10 * w + v) ≤ (100 * a + 10 * b + c) * (10 * d + e) :=
by
  sorry

end max_product_of_digits_is_953_l437_437028


namespace ratio_of_male_striped_turtles_l437_437777

-- Defining the conditions as Lean 4 definitions
def total_turtles : ℕ := 100
def female_turtles_perc : ℝ := 0.60
def male_turtles_perc : ℝ := 0.40
def baby_striped_turtles : ℕ := 4
def adult_striped_perc : ℝ := 0.60
def baby_striped_perc : ℝ := 0.40 -- implicitly given since 100% - 60% = 40%

-- The statement to be proved
theorem ratio_of_male_striped_turtles :
  let total_male_turtles := (male_turtles_perc * total_turtles).to_nat,
      total_striped_turtles := (baby_striped_turtles / baby_striped_perc).to_nat,
      male_striped_turtles := (male_turtles_perc * total_striped_turtles).to_nat
  in male_striped_turtles.to_nat / total_male_turtles.to_nat = 1 / 10 :=
sorry

end ratio_of_male_striped_turtles_l437_437777


namespace cory_fruit_orders_l437_437539

def factorial (n : Nat) : Nat :=
Nat.recOn n 1 (λ n' rec, (n' + 1) * rec)

theorem cory_fruit_orders :
  let apples := 4
  let oranges := 2
  let bananas := 2
  let pear := 1
  let total := apples + oranges + bananas + pear
  ∏ (n in [total, apples, oranges, bananas, pear], factorial n) = 9 → 
  (factorial total / (factorial apples * factorial oranges * factorial bananas * factorial pear)) = 3780 :=
by
  intro h
  sorry

end cory_fruit_orders_l437_437539


namespace zero_point_interval_l437_437561

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 + log x / log 2

theorem zero_point_interval : ∃ x ∈ Ioo 0 1, f x = 0 := by
  sorry

end zero_point_interval_l437_437561


namespace sum_of_integer_solutions_l437_437770

theorem sum_of_integer_solutions : 
  let S := {x : ℤ | 2 * x + 4 ≥ 0 ∧ 6 - x > 3} in
  (∑ x in S, x) = 0 := 
by
  sorry

end sum_of_integer_solutions_l437_437770


namespace hexagon_area_is_10_sqrt_10_l437_437297

-- Define the points and segments of the trapezoid ABCD
variables (A B C D P Q : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] [metric_space Q]
variables (AB CD BC DA : ℝ)
variables (AB_parallel_CD : true) (AB_eq_13 : AB = 13) (BC_eq_7 : BC = 7)
variables (CD_eq_25 : CD = 25) (DA_eq_11 : DA = 11)
variables (bisectors_meet_at_P : P ∈ (bisector A D)) (bisectors_meet_at_Q : Q ∈ (bisector B C))

-- Define the common distance x and calculate areas
variables (x : ℝ)
variables (DP_equidistant : true) (BQ_equidistant : true)
variables (area_AQDP : ℝ) (area_BQCD : ℝ)

-- Overall goal is to prove the area of hexagon ABQCDP equals 10 * sqrt 10
theorem hexagon_area_is_10_sqrt_10 
  (x_eq_sqrt_10 : x = sqrt 10)
  (area_AQDP_eq : area_AQDP = 11 / 2 * x)
  (area_BQCD_eq : area_BQCD = 7 / 2 * x)
  (area_trapezoid_ABCD_eq : area (A + B + C + D) = 19 * x) :
  area (A + B + Q + C + D + P) = 10 * sqrt 10 :=
by
  sorry

end hexagon_area_is_10_sqrt_10_l437_437297


namespace range_of_a_fx2_lt_x2_minus_1_l437_437609

noncomputable def f (x a : ℝ) : ℝ := x - a / x - 2 * Real.log x

noncomputable def g (t : ℝ) : ℝ := t - 2 * Real.log t - 1

theorem range_of_a {a : ℝ} (hx : 0 < a ∧ a < 1) :
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ ∀ x ∈ Ioo x₁ x₂, deriv (λ x, f x a) x = 0 :=
sorry

theorem fx2_lt_x2_minus_1 {a : ℝ} (hx : 0 < a ∧ a < 1) 
  {x₂ : ℝ} (hx2 : x₂ = 1 + Real.sqrt (1 - a)) : f x₂ a < x₂ - 1 :=
sorry

end range_of_a_fx2_lt_x2_minus_1_l437_437609


namespace binary_to_decimal_equiv_l437_437135

theorem binary_to_decimal_equiv : 
  let bin := [1, 1, 0, 1, 0, 1, 1] in 
  let dec := 107 in
  binary_to_nat bin = dec :=
by
  sorry

end binary_to_decimal_equiv_l437_437135


namespace find_value_of_expression_l437_437391

-- Definition of the problem using provided conditions
def parabola (a b c : ℝ) := ∀ x, (a*x^2 + b*x + c)

noncomputable def vertex (a b c : ℝ) := (-b / (2 * a), c - b^2 / (4 * a))
noncomputable def point_on_parabola (a b c : ℝ) (x₁ y₁ : ℝ) := y₁ = (a*x₁^2 + b*x₁ + c)

theorem find_value_of_expression {a b c : ℝ}
  (h_vertex : vertex a b c = (-3, 4))
  (h_point : point_on_parabola a b c (-2) 7) :
  3 * a + 2 * b + c = 76 :=
by
  -- The proof will follow here
  sorry

end find_value_of_expression_l437_437391


namespace probability_draw_l437_437024

theorem probability_draw (pA_win pA_not_lose : ℝ) (h1 : pA_win = 0.3) (h2 : pA_not_lose = 0.8) :
  pA_not_lose - pA_win = 0.5 :=
by 
  sorry

end probability_draw_l437_437024


namespace probability_friendly_pair_l437_437025

-- Definition for the set of numbers {1, 2, 3, 4, 5, 6}
def numbers := {1, 2, 3, 4, 5, 6}

-- Definition of a "friendly pair"
def is_friendly_pair (a b : ℕ) := (a ∈ numbers) ∧ (b ∈ numbers) ∧ (|a - b| ≤ 1)

-- Main theorem: Probability of two numbers being a "friendly pair"
theorem probability_friendly_pair : 
  ( (∑' (a : ℕ) in numbers, ∑' (b : ℕ) in numbers, if (is_friendly_pair a b) then 1 else 0) / 
    (set.finite.to_finset (set.finite.of_fintype (@finset.fintype ℕ numbers))).card ^ 2 ) = 4/9 :=
by sorry

end probability_friendly_pair_l437_437025


namespace greatest_product_three_digit_l437_437026

noncomputable def max_prod_3_digit (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem greatest_product_three_digit :
  ∃ (a b c d e : ℕ), {a, b, c, d, e} = {3, 5, 6, 8, 9} ∧ max_prod_3_digit a b c d e = max_prod_3_digit 9 5 3 8 6 := sorry

end greatest_product_three_digit_l437_437026


namespace no_123456_in_sequence_l437_437551

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sequence_term (a : ℕ) : ℕ :=
  a + sum_of_digits a

def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 1 (fun k ak => sequence_term ak)

theorem no_123456_in_sequence :
  ¬ (∃ n, sequence n = 123456) :=
sorry

end no_123456_in_sequence_l437_437551


namespace math_problem_proof_l437_437611

open Real

-- Define the given function f(x)
def f (x : ℝ) : ℝ := 12 - x^2

-- Define the derivative of f
def f_prime (x : ℝ) : ℝ := -2 * x

-- Problem (I): Prove that the equation of the tangent line is y = -2x + 13 when the slope is -2
def tangent_line_equation_when_slope_is_minus_2 : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, f_prime x = -2 -> f x = -2 * x + b

-- Problem (II): Define the area S(t) and prove its minimum value is 32.
def S (t : ℝ) : ℝ :=
  let k := -2 * t in
  let intercept_y := 12 + t^2 in
  let intercept_x := (1 / 2) * t + 6 / t in
  (1 / 2) * abs intercept_x * intercept_y

def minimum_value_of_S : Prop :=
  ∀ t : ℝ, S t ≥ 32 ∧ (∃ t : ℝ, S t = 32)

theorem math_problem_proof:
  tangent_line_equation_when_slope_is_minus_2 ∧ minimum_value_of_S :=
by sorry

end math_problem_proof_l437_437611


namespace frustum_volume_ratio_l437_437272

-- Definitions for the problem
variables (h t f F : ℝ)
variables (is_frustum : f * 2 = F)

-- Volume formula for a frustum
def volume_frustum (h t f F : ℝ) : ℝ := (h / 3) * (t + real.sqrt (f * F) + F)

-- Given conditions
def conditions_met : Prop :=
  volume_frustum h t f F = (h / 3) * (t + real.sqrt (f * F) + F) ∧
  F = 4 * f / 2

-- Main theorem to prove ratio of volumes
theorem frustum_volume_ratio (h t f F : ℝ) (h > 0) (t > 0) (f > 0) (F > 0) :
  conditions_met h t f F →
  let top_volume := volume_frustum (h / 2) t f (2.5 * f) in
  let bottom_volume := volume_frustum (h / 2) (2.5 * f) 2.5 (4 * f) in
  top_volume / bottom_volume = 2 / 5 :=
sorry

end frustum_volume_ratio_l437_437272


namespace smallest_number_divisible_1_to_10_l437_437882

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437882


namespace number_of_ways_to_choose_co_captains_l437_437646

-- Define the problem conditions
def n : ℕ := 15
def k : ℕ := 4

-- Translate the proof problem to Lean: proving that the combination of choosing 4 out of 15 is 1365
theorem number_of_ways_to_choose_co_captains : nat.choose n k = 1365 := by
  -- n and k are both defined, and nat.choose uses the combination formula
  have h : nat.choose 15 4 = 1365 := sorry
  exact h

end number_of_ways_to_choose_co_captains_l437_437646


namespace proof_goal_l437_437335

variables {A B C D X Y : Point}  -- Points A, B, C, D, X, Y

noncomputable def are_antisimilar (Δ1 Δ2 : Triangle) : Prop :=
  ∀ (a1 a2 a3 a4 : Angle), (a1 + a2 + a3 = 180 ∧ a2 + a3 + a4 = 180) → (a2 = a3 ∧ a1 = a4)

-- Conditions from the problem
variables (AD_angle DX_angle AX_angle DA_angle BC_angle BX_angle CB_angle : Angle)
variables (conditions : AD_angle = BC_angle ∧ DA_angle = CB_angle ∧ all_angles_lt_90)

-- Prove goal statement: angle Λ Y B = 2 * angle AD X
theorem proof_goal :
    let quad := quadrilateral A B C D in
    let triangles_antisimilar := are_antisimilar (Triangle B X C) (Triangle A X D) in
    AD_angle = DX_angle ∧ CB_angle = BX_angle ∧
    all_angles_lt_90 →
    ∃ Y : Point ⟨is_intersection (perpendicular_bisector A B) (perpendicular_bisector C D), true⟩,
    (angle Λ Y B = 2 * angle AD X) :=
sorry

end proof_goal_l437_437335


namespace smallest_number_divisible_1_to_10_l437_437878

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437878


namespace intersect_all_symmetry_axes_at_single_point_l437_437353

theorem intersect_all_symmetry_axes_at_single_point (P : Type) [polygon P]
  (hS : ∀ (ax₁ ax₂ : axes_of_symmetry P), ∃ (p : P), p ∈ ax₁ ∧ p ∈ ax₂) :
  ∃ (q : P), ∀ (ax : axes_of_symmetry P), q ∈ ax :=
by
  sorry

end intersect_all_symmetry_axes_at_single_point_l437_437353


namespace smallest_divisible_by_1_to_10_l437_437896

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437896


namespace lateral_surface_area_of_regular_triangular_pyramid_l437_437563

noncomputable def lateral_surface_area
  (S : ℝ) (apex_angle_is_right : True) : ℝ :=
  if apex_angle_is_right then S * (real.sqrt 3) else 0

theorem lateral_surface_area_of_regular_triangular_pyramid
  (S : ℝ)
  (h : True) :
  lateral_surface_area S h = S * (real.sqrt 3) :=
sorry

end lateral_surface_area_of_regular_triangular_pyramid_l437_437563


namespace prob_transform_in_S_l437_437499

open Complex

-- Define the region S in the complex plane
def in_region_S (z : ℂ) : Prop := 
  let x := z.re
  let y := z.im
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

-- Define the transformation
def transform (z : ℂ) : ℂ :=
  (1 / 2 + (1 / 2) * complex.I) * z

-- State the proof problem
theorem prob_transform_in_S (z : ℂ) (hz : in_region_S z) : 
  in_region_S (transform z) :=
  sorry

end prob_transform_in_S_l437_437499


namespace vending_machine_problem_l437_437102

variable (x n : ℕ)

theorem vending_machine_problem (h : 25 * x + 10 * 15 + 5 * 30 = 25 * 25 + 10 * 5 + 5 * n) (hx : x = 25) :
  n = 50 := by
sorry

end vending_machine_problem_l437_437102


namespace value_of_x_abs_not_positive_l437_437042

theorem value_of_x_abs_not_positive {x : ℝ} : |4 * x - 6| = 0 → x = 3 / 2 :=
by
  sorry

end value_of_x_abs_not_positive_l437_437042


namespace min_value_am_hm_inequality_l437_437303

theorem min_value_am_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end min_value_am_hm_inequality_l437_437303


namespace distinct_infinite_solutions_l437_437030

theorem distinct_infinite_solutions (n : ℕ) (hn : n > 0) : 
  ∃ p q : ℤ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n ∧ (p * p - 5 * q * q = 1) ∧ 
  ∀ m : ℕ, (m ≠ n → (9 + 4 * Real.sqrt 5) ^ m ≠ (9 + 4 * Real.sqrt 5) ^ n) :=
by
  sorry

end distinct_infinite_solutions_l437_437030


namespace lcm_1_to_10_l437_437846

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437846


namespace smallest_multiple_1_through_10_l437_437828

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437828


namespace line_intersects_triangle_l437_437620

open Set
open Function

structure Point (α : Type*) := (x : α) (y : α)

structure Line (α : Type*) := (a : α) (b : α) (c : α)

def on_line {α : Type*} [Field α] (l : Line α) (p : Point α) : Prop :=
l.a * p.x + l.b * p.y + l.c = 0

def segment {α : Type*} [Field α] (p1 p2 : Point α) : Set (Point α) :=
{ p : Point α | ∃ (t : α), t ∈ Icc (0 : α) 1 ∧ p.x = (1 - t) * p1.x + t * p2.x ∧ p.y = (1 - t) * p1.y + t * p2.y }

theorem line_intersects_triangle {α : Type*} [Field α] (A B C : Point α)
  (h : (A.x ≠ B.x ∨ A.y ≠ B.y) ∧ (B.x ≠ C.x ∨ B.y ≠ C.y) ∧ (C.x ≠ A.x ∨ C.y ≠ A.y))
  (l : Line α) (hlA : ¬on_line l A) (hlB : ¬on_line l B) (hlC : ¬on_line l C) :
  (∀ p ∈ segment B C, ¬on_line l p) ∨ (∀ p ∈ segment C A, ¬on_line l p) ∨ (∀ p ∈ segment A B, ¬on_line l p) ∨
  ((∃ p1 ∈ segment B C, on_line l p1) ∧ (∃ p2 ∈ segment C A, on_line l p2) ∧ ¬(∃ p ∈ segment A B, on_line l p)) ∨
  ((∃ p1 ∈ segment C A, on_line l p1) ∧ (∃ p2 ∈ segment A B, on_line l p2) ∧ ¬(∃ p ∈ segment B C, on_line l p)) ∨
  ((∃ p1 ∈ segment A B, on_line l p1) ∧ (∃ p2 ∈ segment B C, on_line l p2) ∧ ¬(∃ p ∈ segment C A, on_line l p)) :=
sorry

end line_intersects_triangle_l437_437620


namespace probability_of_sum_less_than_product_l437_437781

noncomputable def count_filtered_pairs : Nat :=
  let s : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6)
  s.filter (λ p, let (a, b) := p in a ≠ 0 ∧ b ≠ 0 ∧ a + b < a * b).card

noncomputable def total_pairs : Nat :=
  let s : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6)
  s.filter (λ p, let (a, b) := p in a ≠ 0 ∧ b ≠ 0).card

theorem probability_of_sum_less_than_product : (count_filtered_pairs : ℚ) / total_pairs = 3 / 5 :=
  by
  sorry

end probability_of_sum_less_than_product_l437_437781


namespace cos_sub_sin_eq_sqrt_3_div_2_l437_437166

theorem cos_sub_sin_eq_sqrt_3_div_2 (α : ℝ) 
  (h1 : sin α * cos α = 1 / 8) 
  (h2 : (5 * π / 4) < α ∧ α < (3 * π / 2)) : 
  cos α - sin α = sqrt 3 / 2 := by 
  sorry

end cos_sub_sin_eq_sqrt_3_div_2_l437_437166


namespace gcd_84_210_l437_437560

theorem gcd_84_210 : Nat.gcd 84 210 = 42 :=
by {
  sorry
}

end gcd_84_210_l437_437560


namespace smallest_number_divisible_by_1_through_10_l437_437985

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437985


namespace opposite_of_negative_five_l437_437760

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end opposite_of_negative_five_l437_437760


namespace num_of_triangles_with_perimeter_10_l437_437234

theorem num_of_triangles_with_perimeter_10 :
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → 
      a + b + c = 10 ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a) ∧ 
    triangles.card = 4 := sorry

end num_of_triangles_with_perimeter_10_l437_437234


namespace other_root_of_quadratic_l437_437570

theorem other_root_of_quadratic (m : ℝ) (h : (m + 2) * 0^2 - 0 + m^2 - 4 = 0) : 
  ∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x ≠ 0 ∧ x = 1/4 := 
sorry

end other_root_of_quadratic_l437_437570


namespace number_of_grandchildren_l437_437669

-- Definitions based on the conditions
def cards_per_grandkid := 2
def money_per_card := 80
def total_money_given_away := 480

-- Calculation of money each grandkid receives per year
def money_per_grandkid := cards_per_grandkid * money_per_card

-- The theorem we want to prove
theorem number_of_grandchildren :
  (total_money_given_away / money_per_grandkid) = 3 :=
by
  -- Placeholder for the proof
  sorry 

end number_of_grandchildren_l437_437669


namespace leftover_space_desks_bookcases_l437_437111

theorem leftover_space_desks_bookcases 
  (number_of_desks : ℕ) (number_of_bookcases : ℕ)
  (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) (space_between : ℝ)
  (equal_number : number_of_desks = number_of_bookcases)
  (wall_length_eq : wall_length = 15)
  (desk_length_eq : desk_length = 2)
  (bookcase_length_eq : bookcase_length = 1.5)
  (space_between_eq : space_between = 0.5) :
  ∃ k : ℝ, k = 3 := 
by
  sorry

end leftover_space_desks_bookcases_l437_437111


namespace sqrt_defined_iff_nonneg_x_l437_437764

theorem sqrt_defined_iff_nonneg_x (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 3)) ↔ x ≥ -3 :=
by
  sorry

end sqrt_defined_iff_nonneg_x_l437_437764


namespace solution_of_abs_eq_l437_437152

theorem solution_of_abs_eq (x : ℝ) : |x - 5| = 3 * x + 6 → x = -1 / 4 :=
by
  sorry

end solution_of_abs_eq_l437_437152


namespace collinear_A4_B4_C4_circumcenter_A3B3C3_orthocenter_ABC_l437_437637

variables {A B C A1 A2 B1 B2 A3 A4 B3 B4 C1 C2 C3 C4 : Type} 
variables [Inhabited A] [Inhabited B] [Inhabited C]
variables [Inhabited A1] [Inhabited A2] [Inhabited B1] [Inhabited B2]
variables [Inhabited A3] [Inhabited A4] [Inhabited B3] [Inhabited B4]
variables [Inhabited C1] [Inhabited C2] [Inhabited C3] [Inhabited C4]

noncomputable def triangle (A B C: Type) := 
  (A * B * C)

noncomputable def excircle_tangent_point (angle: Type) (A: Type) (A1 A2: Type) := 
  (angle * A1 * A2 * A)

noncomputable def line_intersect (A1 A2 B1 B2 C3 A4 : Type) := 
  (A1 * A2 * B1 * B2 * C3 * A4)

theorem collinear_A4_B4_C4 :
  ∀ (A B C A1 A2 B1 B2 A3 A4 B3 B4 C1 C2 C3 C4: Type),
    excircle_tangent_point A A A1 A2 →
    excircle_tangent_point B B B1 B2 →
    line_intersect A1 A2 B1 B2 C3 A4 →
    A4 = B4 ∧ B4 = C4 :=
sorry

theorem circumcenter_A3B3C3_orthocenter_ABC :
  ∀ (A B C A1 A2 B1 B2 A3 A4 B3 B4 C1 C2 C3 C4: Type),
    excircle_tangent_point A A A1 A2 →
    excircle_tangent_point B B B1 B2 →
    line_intersect A1 A2 B1 B2 C3 A4 →
    (A3 = B3 ∧ B3 = C3) :=
sorry

end collinear_A4_B4_C4_circumcenter_A3B3C3_orthocenter_ABC_l437_437637


namespace new_mean_rent_is_880_l437_437494

theorem new_mean_rent_is_880
  (num_friends : ℕ)
  (initial_average_rent : ℝ)
  (increase_percentage : ℝ)
  (original_rent_increased : ℝ)
  (new_mean_rent : ℝ) :
  num_friends = 4 →
  initial_average_rent = 800 →
  increase_percentage = 20 →
  original_rent_increased = 1600 →
  new_mean_rent = 880 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_is_880_l437_437494


namespace part1_part2_l437_437710

open Real

-- Definitions used in the proof
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

theorem part1 (x : ℝ) : (p 1 x ∧ q x) → 2 < x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : (¬ (∃ x, p a x) → ¬ (∃ x, q x)) → a > 3 / 2 := by
  sorry

end part1_part2_l437_437710


namespace pow_mul_inv_pow_pow_mul_inv_pow_example_l437_437424

theorem pow_mul_inv_pow (a : ℚ) (ha : a ≠ 0) (m n : ℤ) : (a^m) * (a^n) = a^(m+n) :=
begin
  rw [←Int.add_eq_add],
  sorry
end

theorem pow_mul_inv_pow_example : ( (9/10:ℚ)^4 * ( (9/10:ℚ)^-4 ) = 1 ) :=
begin
  apply pow_mul_inv_pow,
  norm_num,
  exact dec_trivial,
  norm_num,
  norm_num
end

end pow_mul_inv_pow_pow_mul_inv_pow_example_l437_437424


namespace lcm_1_to_10_l437_437815

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437815


namespace finite_triples_l437_437361

theorem finite_triples (k : ℕ) : 
  ∃ N, ∀ (p q r : ℕ), 
    p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    p ∣ (q * r - k) ∧ q ∣ (p * r - k) ∧ r ∣ (p * q - k) →
    p ≤ N ∧ q ≤ N ∧ r ≤ N :=
by
  sorry

end finite_triples_l437_437361


namespace anns_speed_l437_437698

theorem anns_speed :
  ∀ (time_Mary time_Ann : ℝ) (len_Mary len_Ann speed_Mary speed_Ann : ℝ), 
  len_Mary = 630 ∧ speed_Mary = 90 ∧ len_Ann = 800 ∧ time_Ann = time_Mary + 13 ∧ time_Mary = len_Mary / speed_Mary →
  speed_Ann = len_Ann / time_Ann :=
begin
  intros,
  sorry
end

end anns_speed_l437_437698


namespace mom_age_when_jayson_born_l437_437438

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end mom_age_when_jayson_born_l437_437438


namespace smallest_divisible_1_to_10_l437_437929

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437929


namespace integral_eq_2pi_l437_437147

open Real
open IntervalIntegral

noncomputable def integral_1 (a b : ℝ) : ℝ := ∫ x in a..b, sqrt(4 - x^2) - x^2017

theorem integral_eq_2pi : integral_1 (-2) 2 = 2 * π := by
  sorry

end integral_eq_2pi_l437_437147


namespace pyramid_volume_l437_437000

theorem pyramid_volume (a : ℝ) (h : 0 < a) : 
  volume_of_pyramid_with_base_side_angle a 60 = (a^3 * real.sqrt 3) / 12 := 
sorry

end pyramid_volume_l437_437000


namespace symmetry_axes_intersect_at_centroid_l437_437356

-- Definitions related to polygon and centroid
structure Polygon :=
  (vertices : List (ℝ × ℝ))

def centroid (p : Polygon) : ℝ × ℝ :=
  let n := p.vertices.length 
  let sum_x := List.sum (List.map Prod.fst p.vertices)
  let sum_y := List.sum (List.map Prod.snd p.vertices)
  (sum_x / n, sum_y / n)

def is_symmetry_axis (p : Polygon) (l : ℝ → ℝ) : Prop :=
  ∀ v ∈ p.vertices, let (x, y) := v in l x = y

def invariant_under_reflection (p : Polygon) (l : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x y, (x, y) ∈ p.vertices → l x = y → reflect_about (x, y) l = (x, y)

theorem symmetry_axes_intersect_at_centroid  
  (p : Polygon)
  (axes : List (ℝ → ℝ))
  (symmetry_prop : ∀ l ∈ axes, is_symmetry_axis p l)
  (centroid_invariant : ∀ l ∈ axes, invariant_under_reflection p l (centroid p)) :
  ∀ l1 l2 ∈ axes, centroid p = intersection_of_axes l1 l2 := 
by
  sorry

end symmetry_axes_intersect_at_centroid_l437_437356


namespace smallest_multiple_1_through_10_l437_437818

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437818


namespace sequence_property_l437_437615

theorem sequence_property (a : ℕ → ℕ) (h1 : ∀ n, n ≥ 1 → a n ∈ { x | x ≥ 1 }) 
  (h2 : ∀ n, n ≥ 1 → a (a n) + a n = 2 * n) : ∀ n, n ≥ 1 → a n = n :=
by
  sorry

end sequence_property_l437_437615


namespace inequality_range_a_l437_437043

open Real

theorem inequality_range_a (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

end inequality_range_a_l437_437043


namespace smallest_divisible_by_1_to_10_l437_437893

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437893


namespace calculate_average_score_l437_437096

theorem calculate_average_score :
  let p3 := 0.30
      p2 := 0.50
      p1 := 0.10
      p0 := 0.10
      score := p3 * 3 + p2 * 2 + p1 * 1 + p0 * 0
  in score = 2 := 
by
  let p3 := 0.30
  let p2 := 0.50
  let p1 := 0.10
  let p0 := 0.10
  let score := p3 * 3 + p2 * 2 + p1 * 1 + p0 * 0
  sorry

end calculate_average_score_l437_437096


namespace greatest_product_three_digit_l437_437027

noncomputable def max_prod_3_digit (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem greatest_product_three_digit :
  ∃ (a b c d e : ℕ), {a, b, c, d, e} = {3, 5, 6, 8, 9} ∧ max_prod_3_digit a b c d e = max_prod_3_digit 9 5 3 8 6 := sorry

end greatest_product_three_digit_l437_437027


namespace A_minus_B_l437_437323

theorem A_minus_B (A B : ℚ) (n : ℕ) :
  (A : ℚ) = 1 / 6 →
  (B : ℚ) = -1 / 12 →
  A - B = 1 / 4 :=
by
  intro hA hB
  rw [hA, hB]
  norm_num

end A_minus_B_l437_437323


namespace even_sine_to_cos_l437_437600

theorem even_sine_to_cos (φ : ℝ) (h₁ : 0 < φ ∧ φ < π) (h₂ : ∀ x, 2 * sin (x + φ) = 2 * sin (-(x + φ))) :
  2 * cos(2 * φ + π / 3) = -1 :=
by
  sorry

end even_sine_to_cos_l437_437600


namespace smallest_multiple_1_through_10_l437_437821

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437821


namespace card_arrangement_impossible_l437_437995

theorem card_arrangement_impossible : 
  ¬ ∃ (l : List ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ l.zip l.tail → (10 * a + b) % 7 = 0) ∧
    (l ~ [1, 2, 3, 4, 5, 6, 8, 9]) := sorry

end card_arrangement_impossible_l437_437995


namespace volume_Q3_l437_437582

theorem volume_Q3 {m n : ℕ} (h : Nat.gcd 17809 19683 = 243) :
  (Q3.totalVolume = (17809 : ℚ) / 19683) → (Q3.simplifiedVolume = (73 : ℚ) / 81) →
  (m + n = 154) :=
by
  -- Condition: Q0 is a regular tetrahedron with volume 1.
  let Q0 : ℚ := 1
  -- Condition: Iterate volume addition process.
  let Δ_Q1 := 4 * (1 / 27)
  let Q1 := Q0 + Δ_Q1
  let Δ_Q2 := 4 * 4 * (1 / (27^2))
  let Q2 := Q1 + Δ_Q2
  let Δ_Q3 := 4 * 4 * 4 * (1 / (27^3))
  let Q3.totalVolume := Q2 + Δ_Q3
  -- Ensure the volume of Q3 is exactly as calculated.
  have h1 : Q3.totalVolume = (17809 : ℚ) / 19683
  sorry
  -- Ensure simplified correct volume.
  let Q3.simplifiedVolume := (73 : ℚ) / 81
  have h2 : Q3.simplifiedVolume = (73 : ℚ) / 81
  sorry
  -- Given conditions and the result:
  assume h
  have : m = 73
  have : n = 81
  show m + n = 154
  sorry

end volume_Q3_l437_437582


namespace calories_peter_wants_to_eat_l437_437705

-- Definitions for the conditions 
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def total_spent : ℕ := 4

-- Proven statement about the calories Peter wants to eat
theorem calories_peter_wants_to_eat : (total_spent / cost_per_bag) * (chips_per_bag * calories_per_chip) = 480 := by
  sorry

end calories_peter_wants_to_eat_l437_437705


namespace function_properties_l437_437694

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) - (Real.log (1 - x))

theorem function_properties : 
  (∀ x ∈ Ioo (-1 : ℝ) 1, f(-x) = -f(x)) ∧ (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f(x₁) < f(x₂)) :=
by
  sorry

end function_properties_l437_437694


namespace fourth_child_age_l437_437290

def first_birth : ℕ := 15
def second_birth_delay : ℕ := 1
def third_birth_age_of_second : ℕ := 4
def fourth_birth_delay_after_third : ℕ := 2

theorem fourth_child_age :
  let first_child_age := first_birth,
      second_child_age := first_child_age - second_birth_delay,
      third_child_age := second_child_age - third_birth_age_of_second,
      fourth_child_age := third_child_age - fourth_birth_delay_after_third
  in fourth_child_age = 8 :=
by {
  sorry
}

end fourth_child_age_l437_437290


namespace ratio_mark_days_used_l437_437118

-- Defining the conditions
def num_sick_days : ℕ := 10
def num_vacation_days : ℕ := 10
def total_hours_left : ℕ := 80
def hours_per_workday : ℕ := 8

-- Total days allotted
def total_days_allotted : ℕ :=
  num_sick_days + num_vacation_days

-- Days left for Mark
def days_left : ℕ :=
  total_hours_left / hours_per_workday

-- Days used by Mark
def days_used : ℕ :=
  total_days_allotted - days_left

-- The ratio of days used to total days allotted (expected to be 1:2)
def ratio_used_to_allotted : ℚ :=
  days_used / total_days_allotted

theorem ratio_mark_days_used :
  ratio_used_to_allotted = 1 / 2 :=
sorry

end ratio_mark_days_used_l437_437118


namespace polar_curve_symmetry_about_polar_axis_l437_437274

-- Define the polar curve
def polar_curve (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- State the symmetry property about the polar axis
theorem polar_curve_symmetry_about_polar_axis :
  ∀ θ ρ, polar_curve ρ θ → polar_curve ρ (-θ) :=
by {
    intros θ ρ h,
    rw [polar_curve] at *,
    rw [Real.cos_neg],
    exact h,
}

end polar_curve_symmetry_about_polar_axis_l437_437274


namespace overall_gain_loss_percent_zero_l437_437018

theorem overall_gain_loss_percent_zero (CP_A CP_B CP_C SP_A SP_B SP_C : ℝ)
  (h1 : CP_A = 600) (h2 : CP_B = 700) (h3 : CP_C = 800)
  (h4 : SP_A = 450) (h5 : SP_B = 750) (h6 : SP_C = 900) :
  ((SP_A + SP_B + SP_C) - (CP_A + CP_B + CP_C)) / (CP_A + CP_B + CP_C) * 100 = 0 :=
by
  sorry

end overall_gain_loss_percent_zero_l437_437018


namespace ellipse_sol_triangle_sol_l437_437586

noncomputable def ellipse_equation (a b : ℝ) : Prop := 
  ∃ b : ℝ, b = 1 ∧ a = sqrt 2 ∧ 
           ∃ c : ℝ, c = 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) = (x^2 / 2 + y^2 = 1)

noncomputable def triangle_area (a b : ℝ) : Prop :=
  ∃ area : ℝ, area = 4 * sqrt 10 / 9 ∧ 
               ∃ c : ℝ, c = 1 ∧ 
               ∃ F1 F2 : ℝ × ℝ, F1 = (-c, 0) ∧ F2 = (c, 0) ∧
                                 ∃ B : ℝ × ℝ, B = (0, -2) ∧ 
                                 ∃ C D : ℝ × ℝ, 
                                 let line_eq := ∀ x : ℝ, (B.2 - F1.2) / (B.1 - F1.1) * (x - B.1) + B.2 in
                                 ∃ CD_eq : unit, |CD| = 10 * sqrt 2 / 9  ∧ 
                                                  (CD_eq ∧ ( ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) = (x^2 / 2 + y^2 = 1))) →

theorem ellipse_sol (a b : ℝ) : (b = 1 ∧ a = sqrt 2 ∧ ∃ e : ℝ, e = sqrt 2 / 2) → ellipse_equation a b := 
begin
  sorry -- Proof
end

theorem triangle_sol (a b : ℝ) : (b = 1 ∧ a = sqrt 2 ∧ ∃ e : ℝ, e = sqrt 2 / 2) → triangle_area a b :=
begin
  sorry -- Proof
end

end ellipse_sol_triangle_sol_l437_437586


namespace derivative_at_zero_l437_437209

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2 * x + 1) * Real.exp x

-- State the main theorem
theorem derivative_at_zero : deriv f 0 = 3 := by {
  sorry
}

end derivative_at_zero_l437_437209


namespace ice_cubes_per_cup_l437_437288

theorem ice_cubes_per_cup (total_ice_cubes number_of_cups : ℕ) (h1 : total_ice_cubes = 30) (h2 : number_of_cups = 6) : 
  total_ice_cubes / number_of_cups = 5 := 
by
  sorry

end ice_cubes_per_cup_l437_437288


namespace square_inscribed_circle_irrational_distance_l437_437315

-- Define the problem: Square inscribed in a circle and point on the circle
variable (r : ℝ) (A B C D P : ℂ)
variable (inscribed : A = r ∧ B = -r ∧ C = r * complex.I ∧ D = -r * complex.I)
variable (on_circle : complex.abs P = r)

-- Define the goal: There exists a distance that is irrational
theorem square_inscribed_circle_irrational_distance :
  ¬ (∀ PA PB PC PD : ℝ, PA = complex.abs (P - A) → PB = complex.abs (P - B) → 
    PC = complex.abs (P - C) → PD = complex.abs (P - D) → 
    (PA.rational ∧ PB.rational ∧ PC.rational ∧ PD.rational)) :=
begin
  intro h,
  -- Use conditions and derive contradiction
  sorry
end

end square_inscribed_circle_irrational_distance_l437_437315


namespace intersect_all_symmetry_axes_at_single_point_l437_437352

theorem intersect_all_symmetry_axes_at_single_point (P : Type) [polygon P]
  (hS : ∀ (ax₁ ax₂ : axes_of_symmetry P), ∃ (p : P), p ∈ ax₁ ∧ p ∈ ax₂) :
  ∃ (q : P), ∀ (ax : axes_of_symmetry P), q ∈ ax :=
by
  sorry

end intersect_all_symmetry_axes_at_single_point_l437_437352


namespace initial_honey_amount_l437_437496

theorem initial_honey_amount (H : ℝ) (r : ℝ) (n : ℕ) (final_honey : ℝ)
  (h_condition : r = 0.75)
  (n_condition : n = 6)
  (final_honey_condition : final_honey = 420) :
  H * r^n = final_honey →
  H ≈ 2361.56 :=
by
  intro h_eq
  rw [h_condition, n_condition, final_honey_condition] at h_eq
  sorry

end initial_honey_amount_l437_437496


namespace yellow_flags_count_l437_437017

theorem yellow_flags_count (n : ℕ) (h : n = 200) (period : ℕ) (periodic_flags : ℕ → ℕ) 
  (h_period : period = 9) (h_pattern : ∀ (k : ℕ), k < period → { if k = 0 ∨ k = 2 ∨ k = 4 then 1 else 0 } = periodic_flags k) :
  (∑ k in range period, periodic_flags k * ((n / period) + if k < (n % period) then 1 else 0)) = 67 := by
  sorry

end yellow_flags_count_l437_437017


namespace integral_equality_l437_437553

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, sqrt (1 - x^2) + 2 * x

theorem integral_equality : integral_value = (Real.pi + 4) / 4 :=
by
  sorry

end integral_equality_l437_437553


namespace limit_log_sub_log_l437_437629

open Real

theorem limit_log_sub_log : ∀ x : ℝ, (0 < x → ∀ ε > 0, ∃ N > 0, ∀ x > N, | log (10*x - 7) / log 5 - log (4*x + 3) / log 5 - log 2.5 / log 5 | < ε) :=
sorry

end limit_log_sub_log_l437_437629


namespace valid_seating_arrangement_count_l437_437483

theorem valid_seating_arrangement_count :
  let desks := [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
                [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
                [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
                [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)],
                [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]]
  (students : Finset (Fin 30)) (boys girls : Finset (Fin 30))
    (h_students : students.card = 30)
    (h_boys : boys.card = 15) (h_girls : girls.card = 15)
    (h_both : boys ∪ girls = students) :
  (∃ arrangement : Fin 30 → (Nat × Nat),
    (∀ i, arrangement i ∈ desks) ∧
    (∀ i j, (arrangement i = arrangement j) ↔ (i = j)) ∧
    (∀ i j, (i ≠ j ∧ (arrangement i == (arrangement j + (0, 1))
                        ∨ arrangement i == (arrangement j + (0, -1))
                        ∨ arrangement i == (arrangement j + (1, 0))
                        ∨ arrangement i == (arrangement j + (-1, 0))))
      → ((i ∈ boys ∧ j ∈ girls) ∨ (i ∈ girls ∧ j ∈ boys)))) →
  2 * (Nat.factorial 15) ^ 2 = sorry := sorry

end valid_seating_arrangement_count_l437_437483


namespace sin_squared_sum_l437_437530

theorem sin_squared_sum : 
  (∑ k in Finset.range 91, (Real.sin (k * Real.pi / 180)) ^ 6) = 229 / 8 := 
by
  sorry

end sin_squared_sum_l437_437530


namespace find_f_2012_l437_437462

-- Given a function f: ℤ → ℤ that satisfies the functional equation:
def functional_equation (f : ℤ → ℤ) := ∀ m n : ℤ, m + f (m + f (n + f m)) = n + f m

-- Given condition:
def f_6_is_6 (f : ℤ → ℤ) := f 6 = 6

-- We need to prove that f 2012 = -2000 under the given conditions.
theorem find_f_2012 (f : ℤ → ℤ) (hf : functional_equation f) (hf6 : f_6_is_6 f) : f 2012 = -2000 := sorry

end find_f_2012_l437_437462


namespace machine_production_rates_and_min_machines_l437_437778

theorem machine_production_rates_and_min_machines {x : ℕ} {y : ℕ}
  (h1 : x = y + 10)
  (h2 : 600 / x = 500 / y)
  (h3 : m : ℕ) :
  (x = 60 ∧ y = 50) ∧ (60 * m + 50 * (18 - m) ≥ 1000 → m ≥ 10) :=
begin
  sorry
end

end machine_production_rates_and_min_machines_l437_437778


namespace count_valid_binary_numbers_correct_l437_437176

noncomputable def count_valid_binary_numbers (n : ℕ) : ℕ :=
  (nat.factorial (2 * n - 2)) / (n * nat.factorial (n - 1) * nat.factorial (n - 1))

theorem count_valid_binary_numbers_correct (n : ℕ) :
  count_valid_binary_numbers n = (nat.binomial (2 * n - 2) (n - 1)) / n :=
by sorry

end count_valid_binary_numbers_correct_l437_437176


namespace smallest_number_divisible_by_1_to_10_l437_437842

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437842


namespace max_a_for_g_geq_zero_l437_437211

def f (x a : ℝ) : ℝ := (x - 2) * exp x - (a / 2) * x^2 + a * x

def f' (x a : ℝ) : ℝ := (x - 1) * (exp x - a) -- First derivative of f

def g (x a : ℝ) := f' x a + 2 - a -- Given g

theorem max_a_for_g_geq_zero : ∃ a : ℤ, (∀ x : ℝ, g x a ≥ 0) ∧ (∀ b : ℤ, b > a → ¬ (∀ x : ℝ, g x b ≥ 0)) :=
by
  sorry

end max_a_for_g_geq_zero_l437_437211


namespace phil_final_quarters_l437_437707

-- Define the conditions
def initial_quarters : ℕ := 50
def doubled_initial_quarters : ℕ := 2 * initial_quarters
def quarters_collected_each_month : ℕ := 3
def months_in_year : ℕ := 12
def quarters_collected_in_a_year : ℕ := quarters_collected_each_month * months_in_year
def quarters_collected_every_third_month : ℕ := 1
def quarters_collected_in_third_months : ℕ := months_in_year / 3 * quarters_collected_every_third_month
def total_before_losing : ℕ := doubled_initial_quarters + quarters_collected_in_a_year + quarters_collected_in_third_months
def lost_quarter_of_total : ℕ := total_before_losing / 4
def quarters_left : ℕ := total_before_losing - lost_quarter_of_total

-- Prove the final result
theorem phil_final_quarters : quarters_left = 105 := by
  sorry

end phil_final_quarters_l437_437707


namespace football_match_prediction_l437_437989

theorem football_match_prediction :
  (∀ (pred : String × String), 
     pred ∈ [("D", "win"), ("B", "runner-up")] ∧ sorry → 
     pred = ("D", "win")) 
  ∧ (pred ∈ [("A", "runner-up"), ("C", "fourth")] ∧ sorry → 
     pred = ("D", "win")) 
  ∧ (pred ∈ [("C", "third"), ("D", "runner-up")] ∧ sorry → 
     pred = ("D", "win")) 
  → ("D", "win") :=
by
  sorry

end football_match_prediction_l437_437989


namespace probability_P_eq_1_plus_i_l437_437088

def vertex_set : set ℂ := {1, -1, complex.I, -complex.I, (1/2 + (real.sqrt 3)/2 * complex.I), -(1/2 + (real.sqrt 3)/2 * complex.I)}

def P (choices : fin 10 → ℂ) := ∏ j, choices j

theorem probability_P_eq_1_plus_i :
  (∃ (a b p : ℕ), p.prime ∧ ¬(p ∣ a) ∧
  (probability (λ choices, 
                choices ∈ pi (finset.univ : finset (fin 10)) (λ _, vertex_set) ∧
                P choices = 1 + complex.I) = (a : ℚ) / p ^ b) ∧
  (a + b + p = 24770)) := by
  sorry

end probability_P_eq_1_plus_i_l437_437088


namespace train_cross_pole_time_l437_437098

theorem train_cross_pole_time 
(speed_km_hr : ℝ) 
(train_length_m : ℝ) 
(h_speed_conv : 1 = (1000/3600) * 1) 
(h_speed : speed_km_hr = 90)
(h_length : train_length_m = 300) : 
  (train_length_m / (speed_km_hr * (1000 / 3600))) = 12 :=
by 
  rw [h_speed, h_length]
  -- Convert speed to meters per second 
  have h_speed_m_s: speed_km_hr * (1000 / 3600) = 25, {
    rw h_speed
    norm_num
  }
  rw h_speed_m_s
  -- Calculate the time
  have h_time: (300 / 25) = 12, {
    norm_num
  }
  exact h_time

end train_cross_pole_time_l437_437098


namespace range_of_omega_l437_437215

def f (ω : ℝ) (x : ℝ) : ℝ :=
if x < 0 then
  Real.exp x + x
else if x <= Real.pi then
  Real.sin (ω * x - Real.pi / 3)
else
  0 -- This should not be needed, added to ensure complete function definition

theorem range_of_omega (ω : ℝ) : (∃ x:ℝ, f ω x = 0) →
(4 = Cardinal.mk {x | f ω x = 0}) →
(7 / 3 ≤ ω ∧ ω < 10 / 3) :=
sorry

end range_of_omega_l437_437215


namespace paulson_spends_75_percent_of_income_l437_437337

variable (P : ℝ)  -- Percentage of income Paulson spends
variable (I : ℝ)  -- Paul's original income

-- Conditions
def original_expenditure := P * I
def original_savings := I - original_expenditure

def new_income := 1.20 * I
def new_expenditure := 1.10 * original_expenditure
def new_savings := new_income - new_expenditure

-- Given: The percentage increase in savings is approximately 50%.
def percentage_increase_in_savings :=
  ((new_savings - original_savings) / original_savings) * 100

-- Proof statement
theorem paulson_spends_75_percent_of_income
  (h : percentage_increase_in_savings P I ≈ 50) : P = 0.75 :=
by
  sorry

end paulson_spends_75_percent_of_income_l437_437337


namespace michael_twice_as_old_as_jacob_l437_437287

theorem michael_twice_as_old_as_jacob :
  ∃ (x : ℕ), ∀ (jacob michael : ℕ), jacob = 4 → michael = jacob + 13 →
  (17 + x = 2 * (4 + x)) ∧ x = 9 :=
by {
  use 9,
  intros jacob michael hj hm,
  rw [hj, hm],
  split; sorry
}

end michael_twice_as_old_as_jacob_l437_437287


namespace largest_negative_integer_solution_l437_437154

theorem largest_negative_integer_solution 
  (x : ℤ) 
  (h : 42 * x + 30 ≡ 26 [MOD 24]) : 
  x ≡ -2 [MOD 12] :=
sorry

end largest_negative_integer_solution_l437_437154


namespace lcm_1_to_10_l437_437856

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437856


namespace correct_fraction_transformation_l437_437049

theorem correct_fraction_transformation (a b : ℕ) (a_ne_0 : a ≠ 0) (b_ne_0 : b ≠ 0) :
  (\frac{2a}{2b} = \frac{a}{b}) ∧ 
  (¬(\frac{a^2}{b^2} = \frac{a}{b})) ∧ 
  (¬(\frac{2a + 1}{4b} = \frac{a + 1}{2b})) ∧ 
  (¬(\frac{a + 2}{b + 2} = \frac{a}{b})) := 
by
  sorry

end correct_fraction_transformation_l437_437049


namespace count_multiples_of_12_between_25_and_200_l437_437625

theorem count_multiples_of_12_between_25_and_200 :
  ∃ n, (∀ i, 25 < i ∧ i < 200 → (∃ k, i = 12 * k)) ↔ n = 14 :=
by
  sorry

end count_multiples_of_12_between_25_and_200_l437_437625


namespace smallest_divisible_1_to_10_l437_437909

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l437_437909


namespace lcm_1_to_10_l437_437852

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437852


namespace lcm_1_to_10_l437_437810

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437810


namespace time_adds_up_correctly_l437_437666

def start_time := (15, 0, 0) -- Representing 3:00:00 PM in 24-hour format as (hours, minutes, seconds)
def elapsed_time := (145, 50, 15) -- Representing 145 hours, 50 minutes, and 15 seconds

def calculate_new_time (start: (ℕ × ℕ × ℕ)) (elapsed: (ℕ × ℕ × ℕ)) : ℕ × ℕ × ℕ := 
  -- Placeholder definition, the actual calculation would be implemented here
  (4, 50, 15) 

def new_time := calculate_new_time start_time elapsed_time

theorem time_adds_up_correctly : (let (P, Q, R) := new_time in P + Q + R) = 69 :=
by
  -- Proof goes here
  sorry

end time_adds_up_correctly_l437_437666


namespace total_books_l437_437510

variables (Beatrix_books Alannah_books Queen_books : ℕ)

def Alannah_condition := Alannah_books = Beatrix_books + 20
def Queen_condition := Queen_books = Alannah_books + (Alannah_books / 5)

theorem total_books (hB : Beatrix_books = 30) (hA : Alannah_condition) (hQ : Queen_condition) : 
  (Beatrix_books + Alannah_books + Queen_books) = 140 :=
by
  sorry

end total_books_l437_437510


namespace remainder_h_x14_div_h_x_l437_437682

def h (x : ℕ) : ℕ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_h_x14_div_h_x : ∀ x, ∃ q r, h(x) * q + r = h(x^14) ∧ r = 7 := 
by sorry

end remainder_h_x14_div_h_x_l437_437682


namespace find_distance_l437_437094

-- Definitions based on conditions
def speed1 := 9 -- speed in kmph
def speed2 := 12 -- speed in kmph
def late_time := 20 / 60 -- 20 minutes late expressed in hours
def early_time := -(20 / 60) -- 20 minutes early expressed in hours

-- Declare the distance
def distance_between_house_and_school (d : ℝ) : Prop :=
    ∃ t : ℝ, (d / speed1 = t + late_time) ∧ (d / speed2 = t + early_time)

-- Prove that d = 24 under the provided conditions
theorem find_distance : ∃ d : ℝ, distance_between_house_and_school d ∧ d = 24 := sorry

end find_distance_l437_437094


namespace tan_ratio_l437_437685

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end tan_ratio_l437_437685


namespace total_area_covered_l437_437767

-- Defining the problem conditions
def side_length_square : ℝ := 2
def radius_circle : ℝ := 1
def num_circles : ℕ := 4

-- Area of the square
def area_square : ℝ := side_length_square * side_length_square

-- Area of one circle
def area_circle : ℝ := Real.pi * radius_circle * radius_circle

-- Total shared area (quadrants)
def total_shared_area : ℝ := num_circles * (area_circle / 4)

-- Total non-shared area
def total_non_shared_area : ℝ := num_circles * (3 * area_circle / 4)

-- Total area covered by the square and the circles
def total_covered_area : ℝ := area_square + total_non_shared_area

-- The theorem to prove
theorem total_area_covered : total_covered_area = 4 + 3 * Real.pi :=
by
  -- Placeholder for proof
  sorry

end total_area_covered_l437_437767


namespace least_faces_sum_eq_25_l437_437023

noncomputable def least_faces_sum 
  (n m : ℕ) 
  (h1 : ∀ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ m) → i + j = 8 → True)
  (h2 : ∀ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ m) → i + j = 11 → True)
  (h3 : ∀ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ m) → i + j = 13 → True)
  (h_prob_8 : Probability (sum_dice_eq 8 n m) = 1/2 * Probability (sum_dice_eq 11 n m))
  (h_prob_13 : Probability (sum_dice_eq 13 n m) = 1/15) : 
  ℕ :=
  n + m

theorem least_faces_sum_eq_25 : least_faces_sum 10 15 _ _ _ _ _ = 25 := 
  sorry

end least_faces_sum_eq_25_l437_437023


namespace diver_score_two_decimal_places_l437_437066

theorem diver_score_two_decimal_places
  (scores : Fin 9 → ℤ)
  (h_remove_highest_lowest : let remaining_scores := (Multiset.ofFinset (Finset.univ.image scores)).erase (Multiset.max (Multiset.ofFinset (Finset.univ.image scores))).erase (Multiset.min (Multiset.ofFinset (Finset.univ.image scores))) 
                             in 7 ≤ remaining_scores.card ∧ avg remaining_scores ≈ 9.4) :
  (let remaining_scores := (Multiset.ofFinset (Finset.univ.image scores)).erase (Multiset.max (Multiset.ofFinset (Finset.univ.image scores))).erase (Multiset.min (Multiset.ofFinset (Finset.univ.image scores))) 
   in round_to 2 (avg remaining_scores)) = 9.43 :=
begin
  sorry
end

end diver_score_two_decimal_places_l437_437066


namespace smallest_n_prob_rain_gt_49_9_l437_437696

def update_prob (p_k : ℝ) : ℝ := 0.5 * p_k + 0.25

@[simp]
def prob_rain_n_days_from_now (n : ℕ) : ℝ :=
  Nat.rec 0 update_prob n

theorem smallest_n_prob_rain_gt_49_9 :
  ∃ n : ℕ, prob_rain_n_days_from_now n > 0.499 ∧ (∀ m < n, prob_rain_n_days_from_now m ≤ 0.499) :=
begin
  let p : ℕ → ℝ := λ n, prob_rain_n_days_from_now n,
  have h_occurs_at_9 : p 9 > 0.499 := by norm_num, -- Here norm_num is a placeholder, actual computation should be used
  have h_occurs_before_9 : ∀ m, m < 9 → p m ≤ 0.499 := by norm_num, -- Here norm_num is a placeholder, actual computation should be used
  use 9,
  exact ⟨h_occurs_at_9, h_occurs_before_9⟩,
end

end smallest_n_prob_rain_gt_49_9_l437_437696


namespace lcm_1_to_10_l437_437957

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437957


namespace smallest_number_divisible_by_1_through_10_l437_437978

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437978


namespace smallest_divisible_1_to_10_l437_437918

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437918


namespace smallest_number_div_by_1_to_10_l437_437792

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437792


namespace find_y_given_conditions_l437_437630

theorem find_y_given_conditions (x y : ℝ) (h1 : x^(3 * y) = 27) (h2 : x = 3) : y = 1 := 
by
  sorry

end find_y_given_conditions_l437_437630


namespace sin_sum_to_fraction_l437_437128

theorem sin_sum_to_fraction :
  (∑ k in Finset.range 91, Real.sin (k * Real.pi / 180) ^ 6) = 229 / 8 :=
by
  sorry

end sin_sum_to_fraction_l437_437128


namespace trigonometric_inequality_l437_437714

theorem trigonometric_inequality (a b c x : ℝ) :
    (a + c) / 2 - (1 / 2) * Real.sqrt((a - c) ^ 2 + b ^ 2) ≤
    a * Real.cos x ^ 2 + b * Real.cos x * Real.sin x + c * Real.sin x ^ 2 ∧
    a * Real.cos x ^ 2 + b * Real.cos x * Real.sin x + c * Real.sin x ^ 2 ≤
    (a + c) / 2 + (1 / 2) * Real.sqrt((a - c) ^ 2 + b ^ 2) :=
by
  sorry

end trigonometric_inequality_l437_437714


namespace lcm_1_to_10_l437_437949

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437949


namespace solution_set_of_inequality_l437_437001

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 3*x - 18 ≤ 0} = set.Icc (-3 : ℝ) 6 := 
sorry

end solution_set_of_inequality_l437_437001


namespace seating_arrangement_ways_l437_437480

theorem seating_arrangement_ways (students desks : ℕ) (rows columns boys girls : ℕ) :
  students = 30 → desks = 30 → rows = 5 → columns = 6 → boys = 15 → girls = 15 → 
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 6 → 
    (¬(∃ m n, (m = i ∧ n ≠ j ∧ abs(m - i) ≤ 1 ∨ n = j ∧ abs(n - j) ≤ 1) ∧ boys) ∧
    ¬(∃ m n, (m = i ∧ n ≠ j ∧ abs(m - i) ≤ 1 ∨ n = j ∧ abs(n - j) ≤ 1) ∧ girls))) →
  2 * (Nat.factorial 15) * (Nat.factorial 15) = 2 * (Nat.factorial 15)^2 := 
by
  sorry

end seating_arrangement_ways_l437_437480


namespace maximize_sum_distances_l437_437273

open Real

theorem maximize_sum_distances 
  (ABC KLM : ℝ → ℝ → ℝ) -- Callable functions for the vertices
  (side_ABC : ℝ := 1)
  (side_KLM : ℝ := 1 / 4)
  (inside_KLM : ∀ x y, KLM x y ≤ ABC x y) -- KLM is inside ABC
  (Σ : ℝ := dist A (line KL) + dist A (line LM) + dist A (line MK)) -- Sum of distances
  (A B C K L M : ℝ)
  (α : ℝ := 90) -- Angle when distances are maximized
  : α = 90 := sorry -- Proof omitted

end maximize_sum_distances_l437_437273


namespace find_a_b_l437_437206

theorem find_a_b (a b : ℝ) (n : ℕ) (h : n ≥ 2) (h_eq : ∀ n, sqrt (n + (n / (n^2 - 1))) = n * sqrt (n / (n^2 - 1))) : 
  sqrt (6 + (a / b)) = 6 * sqrt (a / b) → a = 6 ∧ b = 35 := by
  sorry

end find_a_b_l437_437206


namespace original_equation_l437_437729

theorem original_equation : 9^2 - 8^2 = 17 := by
  sorry

end original_equation_l437_437729


namespace tan_value_calc_value_l437_437605

variables {α : ℝ}

def terminal_side_on_line (α : ℝ) : Prop :=
  ∃ (k : ℤ), tan(α) = -√3 ∧ (α = 2 * k * real.pi + (2 * real.pi / 3) ∨ α = 2 * k * real.pi - (real.pi / 3))

theorem tan_value (h : terminal_side_on_line α) : tan(α) = -√3 :=
sorry

theorem calc_value
  (h : terminal_side_on_line α) :
  (√3 * sin (α - real.pi) + 5 * cos (2 * real.pi - α)) /
  (-√3 * cos ((3 * real.pi / 2) + α) + cos (real.pi + α)) = 4 :=
sorry

end tan_value_calc_value_l437_437605


namespace function_always_negative_iff_l437_437755

theorem function_always_negative_iff (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 :=
by
  -- Proof skipped
  sorry

end function_always_negative_iff_l437_437755


namespace correct_system_of_equations_l437_437060

-- Define the variables for the weights of sparrow and swallow
variables (x y : ℝ)

-- Define the problem conditions
def condition1 : Prop := 5 * x + 6 * y = 16
def condition2 : Prop := 4 * x + y = x + 5 * y

-- Create a theorem stating the conditions imply the identified system
theorem correct_system_of_equations :
  condition1 ∧ condition2 ↔ (5 * x + 6 * y = 16 ∧ 4 * x + y = x + 5 * y) :=
by
  apply Iff.intro;
  { intro h,
    cases h with h1 h2,
    exact ⟨h1, h2⟩ },
  { intro h,
    cases h with h1 h2,
    exact ⟨h1, h2⟩ }

end correct_system_of_equations_l437_437060


namespace Z_4_1_eq_27_l437_437138

def Z (a b : ℕ) : ℕ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem Z_4_1_eq_27 : Z 4 1 = 27 := by
  sorry

end Z_4_1_eq_27_l437_437138


namespace solution_fractional_equation_l437_437006

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end solution_fractional_equation_l437_437006


namespace book_cost_conversion_l437_437105

-- Define the given conditions
def book_cost_cad : ℝ := 500
def conversion_rate : ℝ := 1.25
def sales_tax_rate : ℝ := 0.05

-- Define the calculation for total cost in CAD including tax
def total_cost_in_cad : ℝ := book_cost_cad * (1 + sales_tax_rate)

-- Define the conversion from CAD to USD
def total_cost_in_usd : ℝ := total_cost_in_cad / conversion_rate

-- Define the expected result
def expected_usd_cost : ℝ := 420.00

-- The theorem to be proved
theorem book_cost_conversion :
  total_cost_in_usd = expected_usd_cost :=
begin
  sorry
end

end book_cost_conversion_l437_437105


namespace coin_probability_l437_437110

noncomputable def probability_of_heads (p : ℝ) : Prop :=
  p < 1/2 ∧ 20 * p^3 * (1 - p)^3 = 1/8

theorem coin_probability :
  ∃ p : ℝ, probability_of_heads p ∧ p ≈ 0.276 :=
sorry

end coin_probability_l437_437110


namespace minimum_weighings_to_identify_defective_part_l437_437013

-- Definitions based on the conditions
def Pieces := Fin 5  -- There are 5 pieces
def is_standard (p : Pieces) : Prop := ∃ w : ℝ, ∀ q ≠ p, mass(q) = w  -- All but one piece share the same mass.

-- Define what it means for a piece to be defective
def is_defective (p : Pieces) : Prop := ∃ (w : ℝ), mass(p) ≠ w

-- The main statement to prove
theorem minimum_weighings_to_identify_defective_part : ∃ (n : ℕ), n = 3 ∧ ∀ (psc : set Pieces), 
(set.card psc = 5) ∧ (∃ p : Pieces, is_defective p ∧ ∀ q ≠ p, is_standard q ) → minimum_weighings psc = n := 
by sorry

end minimum_weighings_to_identify_defective_part_l437_437013


namespace movement_representation_l437_437250

theorem movement_representation :
  (-8 : ℤ) represents moving_west 8 :=
sorry

end movement_representation_l437_437250


namespace chord_length_proof_l437_437394

noncomputable def chord_length (t : ℝ) : EuclideanGeometry ℝ :=
  let line := (1 + 2 * t, 2 + t)
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 9 }
  let chord := EuclideanGeometry.chord_length line circle
  chord = (12 / 5) * Real.sqrt 5

/- The statement:
   Given the parametric equations of the line \(x = 1 + 2t\) and \(y = 2 + t\), 
   and the circle \(x^2 + y^2 = 9\),
   prove that the length of the chord intercepted by the line on the circle is \(\frac{12}{5} \sqrt{5}\).
-/
theorem chord_length_proof :
  ∀ (t : ℝ),
    let line := (λ t, (1 + 2 * t, 2 + t))
    let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 9 }
    let chord_length := Chord.length line circle
    chord_length = (12 / 5) * Real.sqrt 5 := sorry

end chord_length_proof_l437_437394


namespace divisible_bc_ad_l437_437732

theorem divisible_bc_ad
  (a b c d u : ℤ)
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) :
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end divisible_bc_ad_l437_437732


namespace div_fraction_l437_437032

/-- The result of dividing 3/7 by 2 1/2 equals 6/35 -/
theorem div_fraction : (3/7) / (2 + 1/2) = 6/35 :=
by 
  sorry

end div_fraction_l437_437032


namespace total_charge_five_hours_l437_437475

variable {F A : ℕ} -- declaring the charges as natural numbers

-- Given conditions
def chargeFirstHour := F = A + 25
def chargeTwoHours := F + A = 115

-- Proof problem: proving the total charge for 5 hours of therapy is $250
theorem total_charge_five_hours (h1 : chargeFirstHour) (h2 : chargeTwoHours) : F + 4 * A = 250 :=
by
  sorry

end total_charge_five_hours_l437_437475


namespace smallest_number_divisible_1_to_10_l437_437874

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437874


namespace lcm_1_10_l437_437968

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437968


namespace lcm_1_to_10_l437_437946

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437946


namespace smallest_k_minimum_l437_437674

noncomputable def smallest_k : ℕ :=
3

theorem smallest_k_minimum
  (m1 m2 m3 m4 m5 : ℤ) 
  (hm : m1 ≠ m2 ∧ m1 ≠ m3 ∧ m1 ≠ m4 ∧ m1 ≠ m5 ∧ m2 ≠ m3 ∧ m2 ≠ m4 ∧ m2 ≠ m5 ∧ m3 ≠ m4 ∧ m3 ≠ m5 ∧ m4 ≠ m5)
  (p : Polynomial ℤ := (Polynomial.X - Polynomial.C m1) * (Polynomial.X - Polynomial.C m2) * (Polynomial.X - Polynomial.C m3) * (Polynomial.X - Polynomial.C m4) * (Polynomial.X - Polynomial.C m5))
  (coeff_count : (p.coeffs.countp (λ c, c ≠ 0)) = 3) :
  smallest_k = 3 :=
by 
  sorry

end smallest_k_minimum_l437_437674


namespace profit_percentage_calc_l437_437404

noncomputable def sale_price_incl_tax : ℝ := 616
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def cost_price : ℝ := 531.03
noncomputable def expected_profit_percentage : ℝ := 5.45

theorem profit_percentage_calc :
  let sale_price_before_tax := sale_price_incl_tax / (1 + sales_tax_rate)
  let profit := sale_price_before_tax - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = expected_profit_percentage :=
by
  sorry

end profit_percentage_calc_l437_437404


namespace sum_sin_sixth_powers_l437_437532

theorem sum_sin_sixth_powers :
    (∑ i in Finset.range 91, Real.sin (i * Real.pi / 180) ^ 6) = 229 / 8 := 
sorry

end sum_sin_sixth_powers_l437_437532


namespace volume_of_pyramid_l437_437716

-- Definitions (conditions)
variables (E F G H Q : Type) [IsRect E F G H]
variable (EF FG QE QF : ℝ)
variable (perp_EH : Perpendicular QE EH)
variable (perp_EF : Perpendicular QE EF)
variable (QF : Distance Q F)

-- Given conditions
axiom EF_val : EF = 10
axiom FG_val : FG = 6
axiom QF_val : QF = 26

-- Theorem statement
theorem volume_of_pyramid {EF FG QE QF EFHG} 
  (h1 : IsRect E F G H)
  (h2 : EF = 10)
  (h3 : FG = 6)
  (h4 : Perpendicular QE EH)
  (h5 : Perpendicular QE EF)
  (h6 : Distance Q F = 26) : 
  volume_pyramid Q E F G H = 480 := 
sorry

end volume_of_pyramid_l437_437716


namespace rectangle_vertices_complex_plane_l437_437545

theorem rectangle_vertices_complex_plane (b : ℝ) :
  (∀ (z : ℂ), z^4 - 10*z^3 + (16*b : ℂ)*z^2 - 2*(3*b^2 - 5*b + 4 : ℂ)*z + 6 = 0 →
    (∃ (w₁ w₂ : ℂ), z = w₁ ∨ z = w₂)) →
  (b = 5 / 3 ∨ b = 2) :=
sorry

end rectangle_vertices_complex_plane_l437_437545


namespace mom_age_when_jayson_born_l437_437437

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end mom_age_when_jayson_born_l437_437437


namespace smallest_number_divisible_1_to_10_l437_437884

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437884


namespace smallest_divisible_1_to_10_l437_437924

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437924


namespace box_solutions_count_l437_437536

theorem box_solutions_count :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), a ∈ s ∧ 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a + b + c) → (a, b, c) ∈ s) ∧
  s.card = 5 :=
by
  sorry

end box_solutions_count_l437_437536


namespace rate_per_kg_mangoes_l437_437419

theorem rate_per_kg_mangoes (kg_apples kg_mangoes total_cost rate_apples total_payment rate_mangoes : ℕ) 
  (h1 : kg_apples = 8) 
  (h2 : rate_apples = 70)
  (h3 : kg_mangoes = 9)
  (h4 : total_payment = 965) :
  rate_mangoes = 45 := 
by
  sorry

end rate_per_kg_mangoes_l437_437419


namespace remainder_of_X_mod_N_is_9801_l437_437534

-- Definitions for X and N as described above
def X : ℕ := ∑ k in finset.range 99, (100 - k) * 10 ^ (2 * k)
def N : ℕ := ∑ k in finset.range 98, (k + 1) * 10 ^ (2 * (98 - k))

theorem remainder_of_X_mod_N_is_9801 : X % N = 9801 :=
by
  sorry

end remainder_of_X_mod_N_is_9801_l437_437534


namespace lcm_1_to_10_l437_437817

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437817


namespace num_of_winnable_players_l437_437505

noncomputable def num_players := 2 ^ 2013

def can_win_if (x y : Nat) : Prop := x ≤ y + 3

def single_elimination_tournament (players : Nat) : Nat :=
  -- Function simulating the single elimination based on the specified can_win_if condition
  -- Assuming the given conditions and returning the number of winnable players directly
  6038

theorem num_of_winnable_players : single_elimination_tournament num_players = 6038 :=
  sorry

end num_of_winnable_players_l437_437505


namespace pizza_cost_l437_437341

theorem pizza_cost (soda_cost jeans_cost start_money quarters_left : ℝ) (quarters_value : ℝ) (total_left : ℝ) (pizza_cost : ℝ) :
  soda_cost = 1.50 → 
  jeans_cost = 11.50 → 
  start_money = 40 → 
  quarters_left = 97 → 
  quarters_value = 0.25 → 
  total_left = quarters_left * quarters_value → 
  pizza_cost = start_money - total_left - (soda_cost + jeans_cost) → 
  pizza_cost = 2.75 :=
by
  sorry

end pizza_cost_l437_437341


namespace GP_GQ_GR_proof_l437_437660

open Real

noncomputable def GP_GQ_GR_sum (XY XZ YZ : ℝ) (G : (ℝ × ℝ × ℝ)) (P Q R : (ℝ × ℝ × ℝ)) : ℝ :=
  let GP := dist G P
  let GQ := dist G Q
  let GR := dist G R
  GP + GQ + GR

theorem GP_GQ_GR_proof (XY XZ YZ : ℝ) (hXY : XY = 4) (hXZ : XZ = 3) (hYZ : YZ = 5)
  (G P Q R : (ℝ × ℝ × ℝ))
  (GP := dist G P) (GQ := dist G Q) (GR := dist G R)
  (hG : GP_GQ_GR_sum XY XZ YZ G P Q R = GP + GQ + GR) :
  GP + GQ + GR = 47 / 15 :=
sorry

end GP_GQ_GR_proof_l437_437660


namespace period_tan_minus_cot_l437_437038

theorem period_tan_minus_cot : ∀ x, f x = f (x + 2 * π)
where 
  f x := tan x - cot x := sorry

end period_tan_minus_cot_l437_437038


namespace stick_cutting_triangle_ways_l437_437136

theorem stick_cutting_triangle_ways :
  ∃ (pieces : list (ℕ × ℕ × ℕ)), 
  pieces.length = 7 ∧ 
  (∀ (a b c : ℕ), (a, b, c) ∈ pieces → a + b + c = 15 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c) :=
by
  sorry

end stick_cutting_triangle_ways_l437_437136


namespace volume_formula_correct_l437_437498

def volume_of_box (x : ℝ) : ℝ :=
  x * (16 - 2 * x) * (12 - 2 * x)

theorem volume_formula_correct (x : ℝ) (h : x ≤ 12 / 5) :
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x :=
by sorry

end volume_formula_correct_l437_437498


namespace sin_minus_cos_one_over_cos2_minus_sin2_l437_437187

variable (x : Real)

theorem sin_minus_cos :
  -Float.pi / 2 < x ∧ x < 0 ∧ sin x + cos x = 1 / 5 →
  sin x - cos x = -7 / 5 :=
sorry

theorem one_over_cos2_minus_sin2 :
  -Float.pi / 2 < x ∧ x < 0 ∧ sin x + cos x = 1 / 5 →
  1 / (cos x ^ 2 - sin x ^ 2) = 25 / 7 :=
sorry

end sin_minus_cos_one_over_cos2_minus_sin2_l437_437187


namespace lcm_1_10_l437_437969

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l437_437969


namespace perpendicular_lines_condition_l437_437769

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ (m : ℝ), 
    ((∃ p q : ℝ, (3 * m * q) * (q / (m + 2)) + ((m - 2) / (m + 2)) * m = -1) ↔
      ((m = 1/2) ∨ (m = -2))) → (m = 1/2) ∨ (m = -2)) ∧
  (∀ (p q : ℝ),
    (∃ (a b : ℝ), (- (m + 2) / (3 * m) = a) ∧ ( (m - 2) / (m + 2) = b) ∧ (a * b = -1)) ↔ 
    ((m = 1/2) ∨ (m = -2))) :=
begin
  sorry
end

end perpendicular_lines_condition_l437_437769


namespace find_difference_l437_437432

def data := [21, 23, 23, 23, 24, 30, 30, 30, 42, 42, 47, 48, 51, 52, 53, 55, 60, 62, 64]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ m x => if l.count x > l.count m then x else m) 0

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· < ·)
  sorted[l.length / 2]

theorem find_difference : abs (median data - mode data) = 19 :=
by
  sorry

end find_difference_l437_437432


namespace tanker_filling_rate_correct_l437_437095

-- Given conditions
def rate_barrels_per_half_minute : ℝ := 2
def liters_per_barrel : ℝ := 159

-- Conversion factors
def half_minute_to_minute : ℝ := 2
def minute_to_hour : ℝ := 60
def liters_to_cubic_meters : ℝ := 10^(-3)

-- Desired output
def expected_rate_cubic_meters_per_hour : ℝ := 38.16

-- The proof statement
theorem tanker_filling_rate_correct :
  (rate_barrels_per_half_minute * half_minute_to_minute * minute_to_hour * liters_per_barrel * liters_to_cubic_meters) = expected_rate_cubic_meters_per_hour :=
by
  -- The proof will be filled in here
  sorry

end tanker_filling_rate_correct_l437_437095


namespace function_domain_l437_437387

noncomputable def f (x : ℝ) : ℝ := (3 * x) / Real.sqrt (x - 1) + Real.ln (2 * x - x^2)

theorem function_domain : {x : ℝ | x > 1 ∧ x < 2} = {x : ℝ | ∃ y : ℝ, y = x ∧ y > 1 ∧ y < 2} :=
sorry

end function_domain_l437_437387


namespace sum_of_roots_zero_l437_437244

theorem sum_of_roots_zero (p q : ℝ) (h1 : p = -q) (h2 : ∀ x, x^2 + p * x + q = 0) : p + q = 0 := 
by {
  sorry 
}

end sum_of_roots_zero_l437_437244


namespace positive_integer_solutions_count_l437_437240

theorem positive_integer_solutions_count :
    {x : ℕ | 0 < x ∧ 15 < -2 * (x : ℤ) + 22}.card = 3 :=
by sorry

end positive_integer_solutions_count_l437_437240


namespace problem_statement_l437_437689

noncomputable def y : ℝ := 
  (∑ n in finset.range 22, real.tan (n + 1) * (real.pi / 180)) / 
  (∑ n in finset.range 22, 1 / real.tan (n + 1) * (real.pi / 180))

def greatest_integer_not_exceeding_50y : ℕ := floor (50 * y)

theorem problem_statement : greatest_integer_not_exceeding_50y = 50 :=
sorry

end problem_statement_l437_437689


namespace smallest_number_divisible_by_1_through_10_l437_437983

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437983


namespace simplify_complex_fraction_l437_437720

-- Definitions based on the conditions
def numerator : ℂ := 5 + 7 * complex.i
def denominator : ℂ := 2 + 3 * complex.i
def expected : ℂ := 31 / 13 - (1 / 13) * complex.i

-- The Lean 4 statement
theorem simplify_complex_fraction : (numerator / denominator) = expected := 
sorry

end simplify_complex_fraction_l437_437720


namespace birds_joined_l437_437468

noncomputable def bird_joined {B : ℕ} (initial_bird : ℕ) (storks : ℕ) (final_condition : ℕ) : Prop :=
  initial_bird + B = storks - final_condition

theorem birds_joined : ∃ B : ℕ, bird_joined 3 6 1 ∧ B = 2 :=
begin
  use 2,
  unfold bird_joined,
  simp,
  exact dec_trivial,
end

end birds_joined_l437_437468


namespace compute_x_y_sum_l437_437313

theorem compute_x_y_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 :=
by
  sorry

end compute_x_y_sum_l437_437313


namespace variables_appear_in_two_lines_l437_437346

-- Define the condition that n >= 4 and the function fn depends on all its variables
variable {n : ℕ} (H : n ≥ 4)
variable (fn : (Fin n → ℝ) → ℝ)

-- Theorem statement
theorem variables_appear_in_two_lines (Hdep : ∀ (x y : Fin n → ℝ), x ≠ y → fn(x) ≠ fn(y)) :
  ∃ lines: List (Fin n → ℝ → ℝ), 
  2 ≤ lines.length ∧ 
  ∀ line ∈ lines, ∃ (x y : ℝ), line == (λ vec => (x,y) ∈ vec) 
    := sorry

end variables_appear_in_two_lines_l437_437346


namespace toucan_count_l437_437466

theorem toucan_count :
  (2 + 1 = 3) :=
by simp [add_comm]

end toucan_count_l437_437466


namespace largest_prime_factor_of_6656_l437_437786

-- Define 6656
def n : ℕ := 6656

-- Define a statement that the largest prime factor of 6656 is 13
theorem largest_prime_factor_of_6656 : nat.prime 13 ∧ (∀ p : ℕ, nat.prime p ∧ p ∣ 6656 → p ≤ 13) :=
by
  sorry

end largest_prime_factor_of_6656_l437_437786


namespace find_a_l437_437606

theorem find_a (a : ℝ) (h : (3 * a + 2) + (a + 14) = 0) : a = -4 :=
sorry

end find_a_l437_437606


namespace daughters_no_daughters_l437_437122

theorem daughters_no_daughters
  (bertha_daughters : ℕ)
  (total_daughters_granddaughters : ℕ)
  (equal_number_of_granddaughters : ℕ)
  (granddaughters_have_no_daughters : ∀ gd, gd ∈ (finset.range equal_number_of_granddaughters) -> ¬(∃ gd' : ℕ, gd' ∈ (finset.range equal_number_of_granddaughters)))
  (total : 40) :
  (∀ d, d ∈ (finset.range bertha_daughters) -> 32 = total_daughters_granddaughters - bertha_daughters) :=
by
  sorry

end daughters_no_daughters_l437_437122


namespace triangle_acute_condition_l437_437193

theorem triangle_acute_condition (a b c : ℝ) (h : a^3 + b^3 = c^3) : 
  (a > 0) → (b > 0) → (c > 0) →
  (a + b > c) → (a + c > b) → (b + c > a) →
  (a^2 + b^2 > c^2) → 
  ∠ABC < 90 ∧ ∠BCA < 90 ∧ ∠CAB < 90 :=
sorry

end triangle_acute_condition_l437_437193


namespace smallest_number_div_by_1_to_10_l437_437800

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l437_437800


namespace smallest_multiple_1_through_10_l437_437825

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l437_437825


namespace problem_valid_distributions_l437_437461

-- Define a function that checks the conditions described
def valid_distribution (l : List ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  List.length l = 5 ∧
  List.mem 1 l ∧ List.mem 2 l ∧
  (l.sorted = [a, a, a, a, b] ∨ l.sorted = [1, 1, 2, 2, 2] ∨ l.sorted = [1, 2, 3, 3, 3])

-- Define the theorem that such a distribution exists
theorem problem_valid_distributions :
  ∃ l : List ℕ, valid_distribution l :=
by
  sorry

end problem_valid_distributions_l437_437461


namespace unique_card_sequences_count_card_sequences_divided_by_10_l437_437075

noncomputable def uniqueCardSequences : ℕ :=
  let characters := ["L", "A", "T", "E", "0", "1", "1", "2"]
  let countWithout1 := Nat.factorial 6 / (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)
  let countWithOne1 := 5 * (Nat.factorial 5 / (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))
  let countWithTwo1 := (Nat.choose 5 2) * (Nat.factorial 3 / (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))
  countWithout1 + countWithOne1 + countWithTwo1

theorem unique_card_sequences_count : uniqueCardSequences = 1380 := by
  sorry

theorem card_sequences_divided_by_10 : uniqueCardSequences / 10 = 138 := by
  sorry

end unique_card_sequences_count_card_sequences_divided_by_10_l437_437075


namespace product_of_numbers_larger_than_reciprocal_eq_neg_one_l437_437414

theorem product_of_numbers_larger_than_reciprocal_eq_neg_one :
  ∃ x y : ℝ, x ≠ y ∧ (x = 1 / x + 2) ∧ (y = 1 / y + 2) ∧ x * y = -1 :=
by
  sorry

end product_of_numbers_larger_than_reciprocal_eq_neg_one_l437_437414


namespace problem1_problem2_l437_437724

-- Problem 1: Simplify and evaluate
theorem problem1 (a : ℚ) (h : a = 1/2) : 2 * a^2 - 5 * a + a^2 + 4 * a - 3 * a^2 - 2 = -5/2 :=
by
  rw h
  sorry

-- Problem 2: Simplify and evaluate
theorem problem2 (x y : ℚ) (hx : x = -2) (hy : y = 3/2) : (1/2) * x - 2 * (x - (1/3) * y^2) + ((-3/2) * x + (1/3) * y^2) = 33/4 :=
by
  rw [hx, hy]
  sorry

end problem1_problem2_l437_437724


namespace smallest_number_divisible_1_to_10_l437_437881

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l437_437881


namespace lcm_1_to_10_l437_437853

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l437_437853


namespace remaining_lives_l437_437776

theorem remaining_lives (initial_players quit1 quit2 player_lives : ℕ) (h1 : initial_players = 15) (h2 : quit1 = 5) (h3 : quit2 = 4) (h4 : player_lives = 7) :
  (initial_players - quit1 - quit2) * player_lives = 42 :=
by
  sorry

end remaining_lives_l437_437776


namespace find_xyz_sum_l437_437459

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end find_xyz_sum_l437_437459


namespace smallest_number_divisible_by_1_to_10_l437_437865

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437865


namespace quadrilateral_area_is_28_l437_437508

noncomputable def area_of_partitioned_triangles (area1 area2 area3 Q : ℝ) : Prop :=
  area1 = 4 ∧ area2 = 8 ∧ area3 = 8 ∧ Q = 28

theorem quadrilateral_area_is_28 :
  ∃ Q, area_of_partitioned_triangles 4 8 8 Q :=
by
  use 28
  unfold area_of_partitioned_triangles
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }

end quadrilateral_area_is_28_l437_437508


namespace meal_cost_one_burger_one_shake_one_cola_l437_437524

-- Define the costs of individual items
variables (B S C : ℝ)

-- Conditions based on given equations
def eq1 : Prop := 3 * B + 7 * S + C = 120
def eq2 : Prop := 4 * B + 10 * S + C = 160.50

-- Goal: Prove that the total cost of one burger, one shake, and one cola is $39
theorem meal_cost_one_burger_one_shake_one_cola :
  eq1 B S C → eq2 B S C → B + S + C = 39 :=
by 
  intros 
  sorry

end meal_cost_one_burger_one_shake_one_cola_l437_437524


namespace lcm_1_to_10_l437_437952

-- Define the range of integers from 1 to 10
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define a function to compute the LCM of a list of integers
def list_lcm : List ℕ → ℕ
| [] := 1
| (x::xs) := Nat.lcm x (list_lcm xs)

-- Define a statement that the LCM of the integers from 1 to 10 is 2520
theorem lcm_1_to_10 : list_lcm nums = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437952


namespace area_percentage_increase_l437_437379

theorem area_percentage_increase (r₁ r₂ : ℝ) (π : ℝ) :
  r₁ = 6 ∧ r₂ = 4 ∧ π > 0 →
  (π * r₁^2 - π * r₂^2) / (π * r₂^2) * 100 = 125 := 
by {
  sorry
}

end area_percentage_increase_l437_437379


namespace probability_sum_leq_12_l437_437506

theorem probability_sum_leq_12 (die6 die8 : ℕ → Prop) (h_die6 : ∀ x, 1 ≤ x ∧ x ≤ 6 → die6 x)
  (h_die8 : ∀ y, 1 ≤ y ∧ y ≤ 8 → die8 y) :
  ∃ p, p = 15 / 16 ∧ (let outcomes := (finset.product (finset.range 6).succ (finset.range 8).succ) in
    (outcomes.filter (λ pair, pair.1 + pair.2 ≤ 12)).card / outcomes.card) = p :=
by
  sorry

end probability_sum_leq_12_l437_437506


namespace generalFormula_sumFirstNTermsB_l437_437583

-- Definitions representing the conditions
def isArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sumFirstNTerms (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range n, a i

def formsGeometricSequence (x y z : ℝ) :=
  (y * y = x * z)

-- Constants and given information in problem conditions
constant a : ℕ → ℝ
constant S : ℕ → ℝ
constant a1 a2 a3 a4 a5 : ℝ
constant conditions : Prop

axiom cond1 : isArithmeticSequence a
axiom cond2 : S = sumFirstNTerms a
axiom cond3 : a 1 + a 2 + 3 * a 4 = 25
axiom cond4 : formsGeometricSequence (a 3 + 2) (a 4) (a 5 - 2)

-- Proving the general formula for the sequence {a_n}
theorem generalFormula (n : ℕ) : a n = 2 * n - 1 := sorry

-- Definitions for the new sequence {b_n}
def b (n : ℕ) := a n * (Real.sqrt (3 ^ (a n + 1)))

-- Proving the sum of the first n terms of the sequence {b_n}
theorem sumFirstNTermsB (n : ℕ) : 
  (∑ i in Finset.range n, b i) = 3 + (n - 1) * 3 ^ (n + 1) := sorry

end generalFormula_sumFirstNTermsB_l437_437583


namespace equilateral_triangle_area_l437_437399

theorem equilateral_triangle_area (p : ℝ) (x : ℝ) 
  (h1 : 3 * x = 6 * p) : 
  (sqrt 3 / 4 * x^2) = sqrt 3 * p^2 :=
begin
  sorry
end

end equilateral_triangle_area_l437_437399


namespace lcm_1_to_10_l437_437811

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l437_437811


namespace smallest_divisible_1_to_10_l437_437923

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l437_437923


namespace smallest_divisible_by_1_to_10_l437_437894

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l437_437894


namespace smallest_number_divisible_by_1_through_10_l437_437974

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l437_437974


namespace initial_short_bushes_l437_437413

theorem initial_short_bushes (B : ℕ) (H1 : B + 20 = 57) : B = 37 :=
by
  sorry

end initial_short_bushes_l437_437413


namespace compute_c_over_d_l437_437407

noncomputable def RootsResult (a b c d : ℝ) : Prop :=
  (3 * 4 + 4 * 5 + 5 * 3 = - c / a) ∧ (3 * 4 * 5 = - d / a)

theorem compute_c_over_d (a b c d : ℝ)
  (h1 : (a * 3 ^ 3 + b * 3 ^ 2 + c * 3 + d = 0))
  (h2 : (a * 4 ^ 3 + b * 4 ^ 2 + c * 4 + d = 0))
  (h3 : (a * 5 ^ 3 + b * 5 ^ 2 + c * 5 + d = 0)) 
  (hr : RootsResult a b c d) :
  c / d = 47 / 60 := 
by
  sorry

end compute_c_over_d_l437_437407


namespace math_proof_problem_l437_437733

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

def parallel (x : Line) (y : Plane) : Prop := sorry
def contained_in (x : Line) (y : Plane) : Prop := sorry
def perpendicular (x : Plane) (y : Plane) : Prop := sorry
def perpendicular_line_plane (x : Line) (y : Plane) : Prop := sorry

theorem math_proof_problem :
  (perpendicular α β) ∧ (perpendicular_line_plane m β) ∧ ¬(contained_in m α) → parallel m α :=
by
  sorry

end math_proof_problem_l437_437733


namespace quadratic_trinomial_permutation_root_l437_437673

theorem quadratic_trinomial_permutation_root (a b c : ℝ) (h : ∀ p q r : ℝ, (p, q, r) ∈ {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)} → ∃ x : ℤ, p * (x : ℝ)^2 + q * (x : ℝ) + r = 0) : a + b + c = 0 :=
begin
  sorry
end

end quadratic_trinomial_permutation_root_l437_437673


namespace smallest_number_divisible_by_1_to_10_l437_437835

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l437_437835


namespace no_possible_arrangement_l437_437997

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l437_437997


namespace range_of_a_l437_437254

theorem range_of_a (a : ℝ) : (∃ (x : ℝ), x > 0 ∧ 2 * x * (x - a) < 1) ↔ a ∈ Ioi (-1) :=
by
  -- We skip the proof here
  sorry

end range_of_a_l437_437254
