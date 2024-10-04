import Mathlib

namespace number_of_grandchildren_l237_237049

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

end number_of_grandchildren_l237_237049


namespace prince_wish_fulfilled_l237_237520

theorem prince_wish_fulfilled
  (k : ℕ)
  (k_gt_1 : 1 < k)
  (k_lt_13 : k < 13)
  (city : Fin 13 → Fin k) 
  (initial_goblets : Fin k → Fin 13)
  (is_gold : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧ city i = city j ∧ is_gold i = true ∧ is_gold j = true := 
sorry

end prince_wish_fulfilled_l237_237520


namespace find_ordered_pair_l237_237898

theorem find_ordered_pair (s h : ℝ) :
  (∀ (u : ℝ), ∃ (x y : ℝ), x = s + 3 * u ∧ y = -3 + h * u ∧ y = 4 * x + 2) →
  (s, h) = (-5 / 4, 12) :=
by
  sorry

end find_ordered_pair_l237_237898


namespace triangle_ABC_is_right_l237_237690

structure Point (α : Type) :=
  (x : α)
  (y : α)

def dist_sq {α : Type} [Field α] (p1 p2 : Point α) : α :=
  (p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2

def is_right_triangle (α : Type) [Field α] (A B C : Point α) : Prop :=
  let AB_sq := dist_sq A B;
  let BC_sq := dist_sq B C;
  let CA_sq := dist_sq C A in
  (AB_sq = BC_sq + CA_sq ∨ BC_sq = AB_sq + CA_sq ∨ CA_sq = AB_sq + BC_sq)

theorem triangle_ABC_is_right :
  is_right_triangle ℝ ⟨5, -2⟩ ⟨1, 5⟩ ⟨-1, 2⟩ :=
by
  -- We need to show that this triangle is a right triangle by the distances formula
  sorry

end triangle_ABC_is_right_l237_237690


namespace necessary_but_not_sufficient_condition_l237_237743

theorem necessary_but_not_sufficient_condition (p q : ℝ → Prop)
    (h₁ : ∀ x k, p x ↔ x ≥ k) 
    (h₂ : ∀ x, q x ↔ 3 / (x + 1) < 1) 
    (h₃ : ∃ k : ℝ, ∀ x, p x → q x ∧ ¬ (q x → p x)) :
  ∃ k, k > 2 :=
by
  sorry

end necessary_but_not_sufficient_condition_l237_237743


namespace no_four_digit_numbers_divisible_by_11_l237_237759

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) 
(h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) (h₇ : 0 ≤ d) (h₈ : d ≤ 9) 
(h₉ : a + b + c + d = 10) (h₁₀ : a + c = b + d) : 
0 = 0 :=
sorry

end no_four_digit_numbers_divisible_by_11_l237_237759


namespace no_valid_solutions_l237_237816

theorem no_valid_solutions (a b : ℝ) (h1 : ∀ x, (a * x + b) ^ 2 = 4 * x^2 + 4 * x + 4) : false :=
  by
  sorry

end no_valid_solutions_l237_237816


namespace cevian_sum_equals_two_l237_237427

-- Definitions based on conditions
variables {A B C D E F O : Type*}
variables (AD BE CF : ℝ) (R : ℝ)
variables (circumcenter_O : O = circumcenter A B C)
variables (intersect_AD_O : AD = abs ((line A D).proj O))
variables (intersect_BE_O : BE = abs ((line B E).proj O))
variables (intersect_CF_O : CF = abs ((line C F).proj O))

-- Prove the main statement
theorem cevian_sum_equals_two (h : circumcenter_O ∧ intersect_AD_O ∧ intersect_BE_O ∧ intersect_CF_O) :
  1 / AD + 1 / BE + 1 / CF = 2 / R :=
sorry

end cevian_sum_equals_two_l237_237427


namespace M_inter_N_eq_2_4_l237_237241

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l237_237241


namespace part_I_part_II_l237_237300

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part_I (x : ℝ) : (∃ a : ℝ, a = 1 ∧ f x a < 1) ↔ x < (1/2) :=
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 6) ↔ (a = 5 ∨ a = -7) :=
sorry

end part_I_part_II_l237_237300


namespace theresa_more_than_thrice_julia_l237_237367

-- Define the problem parameters
variable (tory julia theresa : ℕ)

def tory_videogames : ℕ := 6
def theresa_videogames : ℕ := 11

-- Define the relationships between the numbers of video games
def julia_relationship := julia = tory / 3
def theresa_compared_to_julia := theresa = theresa_videogames
def tory_value := tory = tory_videogames

theorem theresa_more_than_thrice_julia (h1 : julia_relationship tory julia) 
                                       (h2 : tory_value tory)
                                       (h3 : theresa_compared_to_julia theresa) :
  theresa - 3 * julia = 5 :=
by 
  -- Here comes the proof (not required for the task)
  sorry

end theresa_more_than_thrice_julia_l237_237367


namespace circumradius_of_right_triangle_l237_237187

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) : 
  ∃ R : ℝ, R = 8.5 :=
by
  sorry

end circumradius_of_right_triangle_l237_237187


namespace natalia_crates_l237_237733

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l237_237733


namespace number_of_solutions_l237_237612

theorem number_of_solutions :
  ∃ S : Finset (ℤ × ℤ), 
  (∀ (m n : ℤ), (m, n) ∈ S ↔ m^4 + 8 * n^2 + 425 = n^4 + 42 * m^2) ∧ 
  S.card = 16 :=
by { sorry }

end number_of_solutions_l237_237612


namespace toll_for_18_wheel_truck_l237_237364

-- Define the conditions
def wheels_per_axle : Nat := 2
def total_wheels : Nat := 18
def toll_formula (x : Nat) : ℝ := 1.5 + 0.5 * (x - 2)

-- Calculate number of axles from the number of wheels
def number_of_axles := total_wheels / wheels_per_axle

-- Target statement: The toll for the given truck
theorem toll_for_18_wheel_truck : toll_formula number_of_axles = 5.0 := by
  sorry

end toll_for_18_wheel_truck_l237_237364


namespace smallest_multiple_1_through_10_l237_237140

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l237_237140


namespace solve_for_x_l237_237629

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := by
  sorry

end solve_for_x_l237_237629


namespace base_of_minus4_pow3_l237_237040

theorem base_of_minus4_pow3 : ∀ (x : ℤ) (n : ℤ), (x, n) = (-4, 3) → x = -4 :=
by intros x n h
   cases h
   rfl

end base_of_minus4_pow3_l237_237040


namespace triangle_incircle_excircle_ratio_l237_237915

theorem triangle_incircle_excircle_ratio
  {A B C M : Point}
  {b l x y : ℝ}
  (r1 r2 r rho1 rho2 rho : ℝ)
  (hM : M ∈ LineSegment A B)
  (hACM : r1 = inradius_🎄 (Triangle.mk A C M))
  (hBCM : r2 = inradius_🎄 (Triangle.mk B C M))
  (hACM_ex : rho1 = exradius_🎄 (Triangle.mk A C M) M)
  (hBCM_ex : rho2 = exradius_🎄 (Triangle.mk B C M) M)
  (hABC_in : r = inradius_🎄 (Triangle.mk A B C))
  (hABC_ex : rho = exradius_🎄 (Triangle.mk A B C) (LineSegment A B)) :
  (r1 / rho1) * (r2 / rho2) = r / rho :=
by
  sorry

end triangle_incircle_excircle_ratio_l237_237915


namespace probability_of_rolling_two_exactly_four_times_in_five_rolls_l237_237624

theorem probability_of_rolling_two_exactly_four_times_in_five_rolls :
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n-k)
  probability = (25 / 7776) :=
by
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n - k)
  have h : probability = (25 / 7776) := sorry
  exact h

end probability_of_rolling_two_exactly_four_times_in_five_rolls_l237_237624


namespace intersection_correct_l237_237298

open Set

def M : Set ℤ := {-1, 3, 5}
def N : Set ℤ := {-1, 0, 1, 2, 3}
def MN_intersection : Set ℤ := {-1, 3}

theorem intersection_correct : M ∩ N = MN_intersection := by
  sorry

end intersection_correct_l237_237298


namespace multiple_of_sandy_age_l237_237969

theorem multiple_of_sandy_age
    (k_age : ℕ)
    (e : ℕ) 
    (s_current_age : ℕ) 
    (h1: k_age = 10) 
    (h2: e = 340) 
    (h3: s_current_age + 2 = 3 * (k_age + 2)) :
  e / s_current_age = 10 :=
by
  sorry

end multiple_of_sandy_age_l237_237969


namespace sum_of_squares_is_149_l237_237513

-- Define the integers and their sum and product
def integers_sum (b : ℤ) : ℤ := (b - 1) + b + (b + 1)
def integers_product (b : ℤ) : ℤ := (b - 1) * b * (b + 1)

-- Define the condition given in the problem
def condition (b : ℤ) : Prop :=
  integers_product b = 12 * integers_sum b + b^2

-- Define the sum of squares of three consecutive integers
def sum_of_squares (b : ℤ) : ℤ :=
  (b - 1)^2 + b^2 + (b + 1)^2

-- The main statement to be proved
theorem sum_of_squares_is_149 (b : ℤ) (h : condition b) : sum_of_squares b = 149 :=
by
  sorry

end sum_of_squares_is_149_l237_237513


namespace trisha_spending_l237_237092

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

end trisha_spending_l237_237092


namespace trigonometric_inequality_1_l237_237067

theorem trigonometric_inequality_1 {n : ℕ} 
  (h1 : 0 < n) (x : ℝ) (h2 : 0 < x) (h3 : x < (Real.pi / (2 * n))) :
  (1 / 2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1 / n) * Real.tan (n * x) := 
sorry

end trigonometric_inequality_1_l237_237067


namespace fraction_food_l237_237707

-- Define the salary S and remaining amount H
def S : ℕ := 170000
def H : ℕ := 17000

-- Define fractions of the salary spent on house rent and clothes
def fraction_rent : ℚ := 1 / 10
def fraction_clothes : ℚ := 3 / 5

-- Define the fraction F to be proven
def F : ℚ := 1 / 5

-- Define the remaining fraction of the salary
def remaining_fraction : ℚ := H / S

theorem fraction_food :
  ∀ S H : ℕ,
  S = 170000 →
  H = 17000 →
  F = 1 / 5 →
  F + (fraction_rent + fraction_clothes) + remaining_fraction = 1 :=
by
  intros S H hS hH hF
  sorry

end fraction_food_l237_237707


namespace intersection_M_N_l237_237261

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237261


namespace perfectCubesCount_l237_237311

theorem perfectCubesCount (a b : Nat) (h₁ : 50 < a ∧ a ^ 3 > 50) (h₂ : b ^ 3 < 2000 ∧ b < 2000) :
  let n := b - a + 1
  n = 9 := by
  sorry

end perfectCubesCount_l237_237311


namespace lcm_1_10_l237_237135

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l237_237135


namespace fraction_to_decimal_l237_237998

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237998


namespace solve_trig_eq_l237_237362

-- Define the equation
def equation (x : ℝ) : Prop := 3 * Real.sin x = 1 + Real.cos (2 * x)

-- Define the solution set
def solution_set (x : ℝ) : Prop := ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)

-- The proof problem statement
theorem solve_trig_eq {x : ℝ} : equation x ↔ solution_set x := sorry

end solve_trig_eq_l237_237362


namespace problem_statement_l237_237019

theorem problem_statement (x y : ℝ) : (x * y < 18) → (x < 2 ∨ y < 9) :=
sorry

end problem_statement_l237_237019


namespace fraction_of_dehydrated_men_did_not_finish_l237_237519

theorem fraction_of_dehydrated_men_did_not_finish (total_men : ℕ)
  (tripped_fraction : ℚ) (dehydrated_fraction : ℚ) (finished_men : ℕ) 
  (tripped_men : ℕ) (remaining_men : ℕ) (dehydrated_men : ℕ) (did_not_finish_men : ℕ) 
  (dehydrated_did_not_finish_men : ℕ) :
  total_men = 80 → 
  tripped_fraction = 1/4 → 
  dehydrated_fraction = 2/3 → 
  finished_men = 52 → 
  tripped_men = tripped_fraction * total_men → 
  remaining_men = total_men - tripped_men → 
  dehydrated_men = dehydrated_fraction * remaining_men → 
  did_not_finish_men = total_men - finished_men → 
  dehydrated_did_not_finish_men = did_not_finish_men - tripped_men → 
  dehydrated_did_not_finish_men / dehydrated_men = 1/5 := 
by {
  intros,
  sorry
}

end fraction_of_dehydrated_men_did_not_finish_l237_237519


namespace together_work_days_l237_237700

theorem together_work_days (A B C : ℕ) (nine_days : A = 9) (eighteen_days : B = 18) (twelve_days : C = 12) :
  (1 / A + 1 / B + 1 / C) = 1 / 4 :=
by
  sorry

end together_work_days_l237_237700


namespace minimum_value_S_l237_237591

noncomputable def S (x a : ℝ) : ℝ := (x - a)^2 + (Real.log x - a)^2

theorem minimum_value_S : ∃ x a : ℝ, x > 0 ∧ (S x a = 1 / 2) := by
  sorry

end minimum_value_S_l237_237591


namespace seat_to_right_proof_l237_237379

def Xiaofang_seat : ℕ × ℕ := (3, 5)

def seat_to_right (seat : ℕ × ℕ) : ℕ × ℕ :=
  (seat.1 + 1, seat.2)

theorem seat_to_right_proof : seat_to_right Xiaofang_seat = (4, 5) := by
  unfold Xiaofang_seat
  unfold seat_to_right
  sorry

end seat_to_right_proof_l237_237379


namespace change_proof_l237_237338

-- Definitions of the given conditions
def lee_money : ℕ := 10
def friend_money : ℕ := 8
def chicken_wings_cost : ℕ := 6
def chicken_salad_cost : ℕ := 4
def soda_cost : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Statement of the theorem
theorem change_proof : 
  let total_money : ℕ := lee_money + friend_money,
      meal_cost_before_tax : ℕ := chicken_wings_cost + chicken_salad_cost + num_sodas * soda_cost,
      total_meal_cost : ℕ := meal_cost_before_tax + tax
  in total_money - total_meal_cost = 3 := 
by
  -- We skip the proof, as it's not required per instructions
  sorry

end change_proof_l237_237338


namespace find_p_q_of_divisible_polynomial_l237_237617

theorem find_p_q_of_divisible_polynomial :
  ∃ p q : ℤ, (p, q) = (-7, -12) ∧
    (∀ x : ℤ, (x^5 - x^4 + x^3 - p*x^2 + q*x + 4 = 0) → (x = -2 ∨ x = 1)) :=
by
  sorry

end find_p_q_of_divisible_polynomial_l237_237617


namespace cosine_sum_sine_half_sum_leq_l237_237041

variable {A B C : ℝ}

theorem cosine_sum_sine_half_sum_leq (h : A + B + C = Real.pi) :
  (Real.cos A + Real.cos B + Real.cos C) ≤ (Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2)) :=
sorry

end cosine_sum_sine_half_sum_leq_l237_237041


namespace greatest_of_six_consecutive_mixed_numbers_l237_237687

theorem greatest_of_six_consecutive_mixed_numbers (A : ℚ) :
  let B := A + 1
  let C := A + 2
  let D := A + 3
  let E := A + 4
  let F := A + 5
  (A + B + C + D + E + F = 75.5) →
  F = 15 + 1/12 :=
by {
  sorry
}

end greatest_of_six_consecutive_mixed_numbers_l237_237687


namespace intersection_M_N_l237_237271

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237271


namespace intersection_points_count_l237_237202

open Real

theorem intersection_points_count :
  (∃ (x y : ℝ), ((x - ⌊x⌋)^2 + y^2 = x - ⌊x⌋) ∧ (y = 1/3 * x + 1)) →
  (∃ (n : ℕ), n = 8) :=
by
  -- proof goes here
  sorry

end intersection_points_count_l237_237202


namespace quadrilateral_area_sum_l237_237881

theorem quadrilateral_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a^2 * b = 36) : a + b = 4 := 
sorry

end quadrilateral_area_sum_l237_237881


namespace numbers_not_crossed_out_l237_237416

/-- Total numbers between 1 and 90 after crossing out multiples of 3 and 5 is 48. -/
theorem numbers_not_crossed_out : 
  let n := 90 
  let multiples_of_3 := n / 3 
  let multiples_of_5 := n / 5 
  let multiples_of_15 := n / 15 
  let crossed_out := multiples_of_3 + multiples_of_5 - multiples_of_15
  n - crossed_out = 48 :=
by {
  sorry
}

end numbers_not_crossed_out_l237_237416


namespace intersection_M_N_l237_237268

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237268


namespace classroom_chairs_count_l237_237634

theorem classroom_chairs_count :
  ∃ (blue_chairs green_chairs white_chairs total_chairs : ℕ),
    blue_chairs = 10 ∧ 
    green_chairs = 3 * blue_chairs ∧ 
    white_chairs = (green_chairs + blue_chairs) - 13 ∧ 
    total_chairs = blue_chairs + green_chairs + white_chairs ∧ 
    total_chairs = 67 :=
by
  use 10, 30, 27, 67
  split; try refl -- instantiate the variables with the respective values and satisfy the conditions
  split; try reflexivity
  split; try reflexivity
  split; try reflexivity
  trivial   -- this proves that the final sum equals 67

end classroom_chairs_count_l237_237634


namespace smallest_possible_sum_l237_237053

theorem smallest_possible_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Nat.gcd (a + b) 330 = 1) (h4 : b ^ b ∣ a ^ a) (h5 : ¬ b ∣ a) :
  a + b = 147 :=
sorry

end smallest_possible_sum_l237_237053


namespace intersection_of_S_and_complement_of_T_in_U_l237_237060

def U : Set ℕ := { x | 0 ≤ x ∧ x ≤ 8 }
def S : Set ℕ := { 1, 2, 4, 5 }
def T : Set ℕ := { 3, 5, 7 }
def C_U_T : Set ℕ := { x | x ∈ U ∧ x ∉ T }

theorem intersection_of_S_and_complement_of_T_in_U :
  S ∩ C_U_T = { 1, 2, 4 } :=
by
  sorry

end intersection_of_S_and_complement_of_T_in_U_l237_237060


namespace smallest_number_divisible_by_1_through_10_l237_237159

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l237_237159


namespace evaluate_expression_l237_237581

variable {R : Type} [CommRing R]

theorem evaluate_expression (x y z w : R) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w :=
by
  sorry

end evaluate_expression_l237_237581


namespace num_pos_int_x_l237_237424

theorem num_pos_int_x (x : ℕ) : 
  (30 < x^2 + 5 * x + 10) ∧ (x^2 + 5 * x + 10 < 60) ↔ x = 3 ∨ x = 4 ∨ x = 5 := 
sorry

end num_pos_int_x_l237_237424


namespace intersection_M_N_l237_237262

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237262


namespace vector_subtraction_correct_l237_237924

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_correct : (a - b) = (5, -3) :=
by 
  have h1 : a = (2, 1) := by rfl
  have h2 : b = (-3, 4) := by rfl
  sorry

end vector_subtraction_correct_l237_237924


namespace remainder_of_55_power_55_plus_55_div_56_l237_237530

theorem remainder_of_55_power_55_plus_55_div_56 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  -- to be filled with the proof
  sorry

end remainder_of_55_power_55_plus_55_div_56_l237_237530


namespace Hezekiah_age_l237_237968

variable (H : ℕ)
variable (R : ℕ) -- Ryanne's age

-- Defining the conditions
def condition1 : Prop := R = H + 7
def condition2 : Prop := H + R = 15

-- The main theorem we want to prove
theorem Hezekiah_age : condition1 H R → condition2 H R → H = 4 :=
by  -- proof will be here
  sorry

end Hezekiah_age_l237_237968


namespace problem1_problem2_problem3_problem4_l237_237200

theorem problem1 : 0.175 / 0.25 / 4 = 0.175 := by
  sorry

theorem problem2 : 1.4 * 99 + 1.4 = 140 := by 
  sorry

theorem problem3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by
  sorry

theorem problem4 : (3.2 + 0.16) / 0.8 = 4.2 := by
  sorry

end problem1_problem2_problem3_problem4_l237_237200


namespace total_wet_surface_area_is_correct_l237_237180

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end total_wet_surface_area_is_correct_l237_237180


namespace sum_of_series_l237_237571

noncomputable def infinite_series_sum : ℚ :=
∑' n : ℕ, (3 * (n + 1) - 2) / (((n + 1) : ℚ) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_of_series : infinite_series_sum = 11 / 24 := by
  sorry

end sum_of_series_l237_237571


namespace find_f_2011_l237_237745

theorem find_f_2011 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f (x + 1) * f (x - 1) = 1) 
  (h3 : ∀ x, f x > 0) : 
  f 2011 = 1 := 
sorry

end find_f_2011_l237_237745


namespace concentration_after_5_days_l237_237844

noncomputable def ozverin_concentration_after_iterations 
    (initial_volume : ℝ) (initial_concentration : ℝ)
    (drunk_volume : ℝ) (iterations : ℕ) : ℝ :=
initial_concentration * (1 - drunk_volume / initial_volume)^iterations

theorem concentration_after_5_days : 
  ozverin_concentration_after_iterations 0.5 0.4 0.05 5 = 0.236 :=
by
  sorry

end concentration_after_5_days_l237_237844


namespace voice_of_china_signup_ways_l237_237779

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end voice_of_china_signup_ways_l237_237779


namespace A_is_sufficient_but_not_necessary_for_D_l237_237505

variable {A B C D : Prop}

-- Defining the conditions
axiom h1 : A → B
axiom h2 : B ↔ C
axiom h3 : C → D

-- Statement to be proven
theorem A_is_sufficient_but_not_necessary_for_D : (A → D) ∧ ¬(D → A) :=
  by
  sorry

end A_is_sufficient_but_not_necessary_for_D_l237_237505


namespace find_m_values_l237_237432

-- Given function
def f (m x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

-- Theorem statement
theorem find_m_values (m : ℝ) :
  (∃ x y, f m x = 0 ∧ f m y = 0 ∧ (x = 0 ∨ y = 0)) →
  (m = 1 ∨ m = -(5/4)) :=
by sorry

end find_m_values_l237_237432


namespace intersection_M_N_l237_237288

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237288


namespace sufficient_not_necessary_condition_l237_237575

noncomputable section

def is_hyperbola_point (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

def foci_distance_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  |(P.1 - F1.1)^2 + (P.2 - F1.2)^2 - (P.1 - F2.1)^2 + (P.2 - F2.2)^2| = 6

theorem sufficient_not_necessary_condition 
  (x y F1_1 F1_2 F2_1 F2_2 : ℝ) (P : ℝ × ℝ)
  (P_hyp: is_hyperbola_point x y)
  (cond : foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2)) :
  ∃ x y, is_hyperbola_point x y ∧ foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2) :=
  sorry

end sufficient_not_necessary_condition_l237_237575


namespace intersection_of_M_and_N_l237_237228

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l237_237228


namespace intersection_M_N_l237_237277

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237277


namespace total_savings_l237_237394

-- Definition to specify the denomination of each bill
def bill_value : ℕ := 100

-- Condition: Number of $100 bills Michelle has
def num_bills : ℕ := 8

-- The theorem to prove the total savings amount
theorem total_savings : num_bills * bill_value = 800 :=
by
  sorry

end total_savings_l237_237394


namespace sum_of_b_values_l237_237987

theorem sum_of_b_values (b : ℤ) (hb : ∃ k, b^2 - 12*b = k^2) : 
  let possible_b_values : List ℤ := 
    [ (9 + 4) / 2 + 6, (6 + 6) / 2 + 6, 
      (12 + 3) / 2 + 6, (18 + 2) / 2 + 6, 
      (36 + 1) / 2 + 6 ] 
  in possible_b_values.sum = 80 :=
by sorry

end sum_of_b_values_l237_237987


namespace prism_faces_l237_237880

-- Define basic properties and conditions
def is_prism_with_L_gon_base (edges : ℕ) (L : ℕ) :=
  edges = 3 * L

noncomputable def number_of_faces (L : ℕ) : ℕ :=
  L + 2  -- L lateral faces and 2 bases

-- Main statement
theorem prism_faces (edges : ℕ) (L : ℕ) (h : edges = 3 * L) : number_of_faces L = 8 :=
by
  -- Instantiate the given condition
  have hL : L = 6 := sorry
  -- Prove the number of faces calculation
  have h_faces : number_of_faces L = number_of_faces 6 := by rw hL
  have h_faces_calc : number_of_faces 6 = 8 := sorry
  exact (h_faces.trans h_faces_calc)

end prism_faces_l237_237880


namespace fraction_to_decimal_l237_237994

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237994


namespace simplify_and_evaluate_l237_237493

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l237_237493


namespace smallest_number_div_by_1_to_10_l237_237161

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l237_237161


namespace sphere_volume_l237_237754

noncomputable def volume_of_sphere {x y z : ℝ} (v : ℝ × ℝ × ℝ) 
  (h : (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) = (4 * v.1 - 16 * v.2 + 32 * v.3)) : ℝ :=
  (4 / 3) * Real.pi * 18^3

theorem sphere_volume {v : ℝ × ℝ × ℝ}
  (h : (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) = (4 * v.1 - 16 * v.2 + 32 * v.3)) :
  volume_of_sphere v h = 7776 * Real.pi :=
sorry

end sphere_volume_l237_237754


namespace solve_quadratic1_solve_quadratic2_l237_237811

-- Equation 1
theorem solve_quadratic1 (x : ℝ) :
  (x = 4 + 3 * Real.sqrt 2 ∨ x = 4 - 3 * Real.sqrt 2) ↔ x ^ 2 - 8 * x - 2 = 0 := by
  sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) :
  (x = 3 / 2 ∨ x = -1) ↔ 2 * x ^ 2 - x - 3 = 0 := by
  sorry

end solve_quadratic1_solve_quadratic2_l237_237811


namespace mr_wang_returns_to_start_elevator_electricity_consumption_l237_237964

-- Definition for the first part of the problem
def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]

theorem mr_wang_returns_to_start : List.sum floor_movements = 0 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

-- Definitions for the second part of the problem
def height_per_floor : Int := 3
def electricity_per_meter : Float := 0.2

-- Calculation of electricity consumption (distance * electricity_per_meter per floor)
def total_distance_traveled : Int := 
  (floor_movements.map Int.natAbs).sum * height_per_floor

theorem elevator_electricity_consumption : 
  (Float.ofInt total_distance_traveled) * electricity_per_meter = 33.6 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

end mr_wang_returns_to_start_elevator_electricity_consumption_l237_237964


namespace intersection_M_N_l237_237280

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237280


namespace primes_divisibility_l237_237640

theorem primes_divisibility
  (p1 p2 p3 p4 q1 q2 q3 q4 : ℕ)
  (hp1_lt_p2 : p1 < p2) (hp2_lt_p3 : p2 < p3) (hp3_lt_p4 : p3 < p4)
  (hq1_lt_q2 : q1 < q2) (hq2_lt_q3 : q2 < q3) (hq3_lt_q4 : q3 < q4)
  (hp4_minus_p1 : p4 - p1 = 8) (hq4_minus_q1 : q4 - q1 = 8)
  (hp1_gt_5 : 5 < p1) (hq1_gt_5 : 5 < q1) :
  30 ∣ (p1 - q1) :=
sorry

end primes_divisibility_l237_237640


namespace xy_z_eq_inv_sqrt2_l237_237453

noncomputable def f (t : ℝ) : ℝ := (Real.sqrt 2) * t + 1 / ((Real.sqrt 2) * t)

theorem xy_z_eq_inv_sqrt2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (Real.sqrt 2) * x + 1 / ((Real.sqrt 2) * x) 
      + (Real.sqrt 2) * y + 1 / ((Real.sqrt 2) * y) 
      + (Real.sqrt 2) * z + 1 / ((Real.sqrt 2) * z) 
      = 6 - 2 * (Real.sqrt (2 * x)) * abs (y - z) 
            - (Real.sqrt (2 * y)) * (x - z) ^ 2 
            - (Real.sqrt (2 * z)) * (Real.sqrt (abs (x - y)))) :
  x = y ∧ y = z ∧ z = 1 / (Real.sqrt 2) :=
sorry

end xy_z_eq_inv_sqrt2_l237_237453


namespace jims_speed_l237_237401

variable (x : ℝ)

theorem jims_speed (bob_speed : ℝ) (bob_head_start : ℝ) (time : ℝ) (bob_distance : ℝ) :
  bob_speed = 6 →
  bob_head_start = 1 →
  time = 1 / 3 →
  bob_distance = bob_speed * time →
  (x * time = bob_distance + bob_head_start) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jims_speed_l237_237401


namespace fraction_product_l237_237566

theorem fraction_product : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end fraction_product_l237_237566


namespace Tommy_Ratio_Nickels_to_Dimes_l237_237988

def TommyCoinsProblem :=
  ∃ (P D N Q : ℕ), 
    (D = P + 10) ∧ 
    (Q = 4) ∧ 
    (P = 10 * Q) ∧ 
    (N = 100) ∧ 
    (N / D = 2)

theorem Tommy_Ratio_Nickels_to_Dimes : TommyCoinsProblem := by
  sorry

end Tommy_Ratio_Nickels_to_Dimes_l237_237988


namespace prism_faces_l237_237872

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end prism_faces_l237_237872


namespace find_a_l237_237297

noncomputable def A : Set ℝ := {x | x^2 - x - 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def is_solution (a : ℝ) : Prop := ∀ b, b ∈ B a → b ∈ A

theorem find_a (a : ℝ) : (B a ⊆ A) → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  intro h
  sorry

end find_a_l237_237297


namespace solve_equation_integers_l237_237356

theorem solve_equation_integers :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (1 + 1 / (x : ℚ)) * (1 + 1 / (y : ℚ)) * (1 + 1 / (z : ℚ)) = 2 ∧
  (x = 2 ∧ y = 4 ∧ z = 15 ∨
   x = 2 ∧ y = 5 ∧ z = 9 ∨
   x = 2 ∧ y = 6 ∧ z = 7 ∨
   x = 3 ∧ y = 4 ∧ z = 5 ∨
   x = 3 ∧ y = 3 ∧ z = 8 ∨
   x = 2 ∧ y = 15 ∧ z = 4 ∨
   x = 2 ∧ y = 9 ∧ z = 5 ∨
   x = 2 ∧ y = 7 ∧ z = 6 ∨
   x = 3 ∧ y = 5 ∧ z = 4 ∨
   x = 3 ∧ y = 8 ∧ z = 3) ∧
  (y = 2 ∧ x = 4 ∧ z = 15 ∨
   y = 2 ∧ x = 5 ∧ z = 9 ∨
   y = 2 ∧ x = 6 ∧ z = 7 ∨
   y = 3 ∧ x = 4 ∧ z = 5 ∨
   y = 3 ∧ x = 3 ∧ z = 8 ∨
   y = 15 ∧ x = 4 ∧ z = 2 ∨
   y = 9 ∧ x = 5 ∧ z = 2 ∨
   y = 7 ∧ x = 6 ∧ z = 2 ∨
   y = 5 ∧ x = 4 ∧ z = 3 ∨
   y = 8 ∧ x = 3 ∧ z = 3) ∧
  (z = 2 ∧ x = 4 ∧ y = 15 ∨
   z = 2 ∧ x = 5 ∧ y = 9 ∨
   z = 2 ∧ x = 6 ∧ y = 7 ∨
   z = 3 ∧ x = 4 ∧ y = 5 ∨
   z = 3 ∧ x = 3 ∧ y = 8 ∨
   z = 15 ∧ x = 4 ∧ y = 2 ∨
   z = 9 ∧ x = 5 ∧ y = 2 ∨
   z = 7 ∧ x = 6 ∧ y = 2 ∨
   z = 5 ∧ x = 4 ∧ y = 3 ∨
   z = 8 ∧ x = 3 ∧ y = 3)
:= sorry

end solve_equation_integers_l237_237356


namespace solve_for_k_l237_237388

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end solve_for_k_l237_237388


namespace problem1_problem2_problem3_l237_237405

theorem problem1 : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 :=
by
  sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 6) ^ 2 - (Real.sqrt 5 + Real.sqrt 6) ^ 2 = -4 * Real.sqrt 30 :=
by
  sorry

theorem problem3 : (2 * Real.sqrt (3 / 2) - Real.sqrt (1 / 2)) * (1 / 2 * Real.sqrt 8 + Real.sqrt (2 / 3)) = (5 / 3) * Real.sqrt 3 + 1 :=
by
  sorry

end problem1_problem2_problem3_l237_237405


namespace complex_modulus_proof_l237_237580

noncomputable def complex_modulus_example : ℝ := 
  Complex.abs ⟨3/4, -3⟩

theorem complex_modulus_proof : complex_modulus_example = Real.sqrt 153 / 4 := 
by 
  unfold complex_modulus_example
  sorry

end complex_modulus_proof_l237_237580


namespace conic_curve_eccentricity_l237_237748

theorem conic_curve_eccentricity (m : ℝ) 
    (h1 : ∃ k, k ≠ 0 ∧ 1 * k = m ∧ m * k = 4)
    (h2 : m = -2) : ∃ e : ℝ, e = Real.sqrt 3 :=
by
  sorry

end conic_curve_eccentricity_l237_237748


namespace decomposition_of_5_to_4_eq_125_l237_237738

theorem decomposition_of_5_to_4_eq_125 :
  (∃ a b c : ℕ, (5^4 = a + b + c) ∧ 
                (a = 121) ∧ 
                (b = 123) ∧ 
                (c = 125)) := by 
sorry

end decomposition_of_5_to_4_eq_125_l237_237738


namespace tan_ratio_l237_237055

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end tan_ratio_l237_237055


namespace jamie_paid_0_more_than_alex_l237_237559

/-- Conditions:
     1. Alex and Jamie shared a pizza cut into 10 equally-sized slices.
     2. Alex wanted a plain pizza.
     3. Jamie wanted a special spicy topping on one-third of the pizza.
     4. The cost of a plain pizza was $10.
     5. The spicy topping on one-third of the pizza cost an additional $3.
     6. Jamie ate all the slices with the spicy topping and two extra plain slices.
     7. Alex ate the remaining plain slices.
     8. They each paid for what they ate.
    
     Question: How many more dollars did Jamie pay than Alex?
     Answer: 0
-/
theorem jamie_paid_0_more_than_alex :
  let total_slices := 10
  let cost_plain := 10
  let cost_spicy := 3
  let total_cost := cost_plain + cost_spicy
  let cost_per_slice := total_cost / total_slices
  let jamie_slices := 5
  let alex_slices := total_slices - jamie_slices
  let jamie_cost := jamie_slices * cost_per_slice
  let alex_cost := alex_slices * cost_per_slice
  jamie_cost - alex_cost = 0 :=
by
  sorry

end jamie_paid_0_more_than_alex_l237_237559


namespace Taehyung_walked_distance_l237_237574

variable (step_distance : ℝ) (steps_per_set : ℕ) (num_sets : ℕ)
variable (h1 : step_distance = 0.45)
variable (h2 : steps_per_set = 90)
variable (h3 : num_sets = 13)

theorem Taehyung_walked_distance :
  (steps_per_set * step_distance) * num_sets = 526.5 :=
by 
  rw [h1, h2, h3]
  sorry

end Taehyung_walked_distance_l237_237574


namespace intersection_M_N_l237_237257

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237257


namespace determine_x_l237_237901

theorem determine_x 
  (w : ℤ) (hw : w = 90)
  (z : ℤ) (hz : z = 4 * w + 40)
  (y : ℤ) (hy : y = 3 * z + 15)
  (x : ℤ) (hx : x = 2 * y + 6) :
  x = 2436 := 
by
  sorry

end determine_x_l237_237901


namespace perry_more_games_than_phil_l237_237350

theorem perry_more_games_than_phil (dana_wins charlie_wins perry_wins : ℕ) :
  perry_wins = dana_wins + 5 →
  charlie_wins = dana_wins - 2 →
  charlie_wins + 3 = 12 →
  perry_wins - 12 = 4 :=
by
  sorry

end perry_more_games_than_phil_l237_237350


namespace smallest_lcm_of_4digit_integers_with_gcd_5_l237_237628

theorem smallest_lcm_of_4digit_integers_with_gcd_5 :
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 1000 ≤ b ∧ b < 10000 ∧ gcd a b = 5 ∧ lcm a b = 201000 :=
by
  sorry

end smallest_lcm_of_4digit_integers_with_gcd_5_l237_237628


namespace sign_up_ways_l237_237777

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end sign_up_ways_l237_237777


namespace markov_coprime_squares_l237_237012

def is_coprime (x y : ℕ) : Prop :=
Nat.gcd x y = 1

theorem markov_coprime_squares (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  x^2 + y^2 + z^2 = 3 * x * y * z →
  ∃ a b c: ℕ, (a, b, c) = (2, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∧ 
  (a ≠ 1 → ∃ p q : ℕ, is_coprime p q ∧ a = p^2 + q^2) :=
sorry

end markov_coprime_squares_l237_237012


namespace probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l237_237679

-- Conditions
def red_ball_probability := 1 / 2
def yellow_ball_probability := 1 / 2
def num_draws := 3

-- Define the events and their probabilities
def prob_three_red : ℚ := red_ball_probability ^ num_draws
def prob_three_same : ℚ := 2 * (red_ball_probability ^ num_draws)
def prob_not_all_same : ℚ := 1 - prob_three_same / 2

-- Lean statements
theorem probability_three_red_balls : prob_three_red = 1 / 8 :=
by
  sorry

theorem probability_three_same_color_balls : prob_three_same = 1 / 4 :=
by
  sorry

theorem probability_not_all_same_color_balls : prob_not_all_same = 3 / 4 :=
by
  sorry

end probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l237_237679


namespace knights_gold_goblets_l237_237521

theorem knights_gold_goblets (k : ℕ) (k_gt_1 : 1 < k) (k_lt_13 : k < 13)
  (goblets : Fin 13 → Bool) (gold_goblets : (Fin 13 → Bool) → ℕ) 
  (cities : Fin 13 → Fin k) :
  (∃ (i j : Fin 13), i ≠ j ∧ cities i = cities j ∧ goblets i ∧ goblets j) :=
begin
  sorry
end

end knights_gold_goblets_l237_237521


namespace gcd_sum_and_lcm_eq_gcd_l237_237805

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end gcd_sum_and_lcm_eq_gcd_l237_237805


namespace bob_height_in_inches_l237_237539

theorem bob_height_in_inches (tree_height shadow_tree bob_shadow : ℝ)
  (h1 : tree_height = 50)
  (h2 : shadow_tree = 25)
  (h3 : bob_shadow = 6) :
  (12 * (tree_height / shadow_tree) * bob_shadow) = 144 :=
by sorry

end bob_height_in_inches_l237_237539


namespace intersection_M_N_l237_237260

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237260


namespace roll_two_twos_in_five_l237_237618

def probability_of_exactly_two_twos (n k : Nat) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem roll_two_twos_in_five :
  probability_of_exactly_two_twos 5 2 (1 / 8) = 3430 / 32768 := by
  sorry

end roll_two_twos_in_five_l237_237618


namespace remove_brackets_l237_237966

-- Define the variables a, b, and c
variables (a b c : ℝ)

-- State the theorem
theorem remove_brackets (a b c : ℝ) : a - (b - c) = a - b + c := 
sorry

end remove_brackets_l237_237966


namespace avg_lottery_draws_eq_5232_l237_237062

def avg_lottery_draws (n m : ℕ) : ℕ :=
  let N := 90 * 89 * 88 * 87 * 86
  let Nk := 25 * 40320
  N / Nk

theorem avg_lottery_draws_eq_5232 : avg_lottery_draws 90 5 = 5232 :=
by 
  unfold avg_lottery_draws
  sorry

end avg_lottery_draws_eq_5232_l237_237062


namespace smallest_number_divisible_1_to_10_l237_237177

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237177


namespace quadratic_root_in_l237_237601

variable (a b c m : ℝ)

theorem quadratic_root_in (ha : a > 0) (hm : m > 0) 
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 := 
by
  sorry

end quadratic_root_in_l237_237601


namespace compute_expression_l237_237573

theorem compute_expression :
  25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := 
sorry

end compute_expression_l237_237573


namespace prime_divisor_form_l237_237460


open Int

theorem prime_divisor_form (a b : ℤ) (h : IsCoprime a b) : 
  ∀ p : ℕ, Prime p → p ∣ (a ^ 2 + 2 * b ^ 2) → ∃ x y : ℤ, (p : ℤ) = x ^ 2 + 2 * y ^ 2 :=
sorry

end prime_divisor_form_l237_237460


namespace time_gaps_l237_237326

theorem time_gaps (dist_a dist_b dist_c : ℕ) (time_a time_b time_c : ℕ) :
  dist_a = 130 →
  dist_b = 130 →
  dist_c = 130 →
  time_a = 36 →
  time_b = 45 →
  time_c = 42 →
  (time_b - time_a = 9) ∧ (time_c - time_a = 6) ∧ (time_b - time_c = 3) := by
  intros hdist_a hdist_b hdist_c htime_a htime_b htime_c
  sorry

end time_gaps_l237_237326


namespace green_more_than_red_l237_237386

def red_peaches : ℕ := 7
def green_peaches : ℕ := 8

theorem green_more_than_red : green_peaches - red_peaches = 1 := by
  sorry

end green_more_than_red_l237_237386


namespace Albert_more_rocks_than_Joshua_l237_237336

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_rocks_than_Joshua_l237_237336


namespace prism_faces_l237_237861

theorem prism_faces (edges : ℕ) (h_edges : edges = 18) : ∃ faces : ℕ, faces = 8 :=
by
  -- Define L as the number of lateral faces
  let L := edges / 3
  have h_L : L = 6, from calc
    L = 18 / 3 : by rw [h_edges]
    ... = 6 : by norm_num
  -- Define the total number of faces
  let total_faces := L + 2
  have h_faces : total_faces = 8, from calc
    total_faces = 6 + 2 : by rw [h_L]
    ... = 8 : by norm_num
  use total_faces
  exact h_faces

end prism_faces_l237_237861


namespace find_linear_function_l237_237630

theorem find_linear_function (f : ℝ → ℝ) (hf_inc : ∀ x y, x < y → f x < f y)
  (hf_lin : ∃ a b, a > 0 ∧ ∀ x, f x = a * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 :=
by
  sorry

end find_linear_function_l237_237630


namespace smallest_multiple_1_through_10_l237_237138

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l237_237138


namespace range_of_x_l237_237670

def y_function (x : ℝ) : ℝ := x

def y_translated (x : ℝ) : ℝ := x + 2

theorem range_of_x {x : ℝ} (h : y_translated x > 0) : x > -2 := 
by {
  sorry
}

end range_of_x_l237_237670


namespace martha_initial_marbles_l237_237006

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

end martha_initial_marbles_l237_237006


namespace right_triangle_area_l237_237193

theorem right_triangle_area :
  ∃ (a b c : ℕ), (c^2 = a^2 + b^2) ∧ (2 * b^2 - 23 * b + 11 = 0) ∧ (a * b / 2 = 330) :=
sorry

end right_triangle_area_l237_237193


namespace simplify_fraction_l237_237437

variables {x y : ℝ}

theorem simplify_fraction (h : x / y = 2 / 5) : (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 :=
by
  sorry

end simplify_fraction_l237_237437


namespace change_calculation_l237_237340

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l237_237340


namespace sum_last_two_digits_l237_237377

theorem sum_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) :
  (a ^ 30 + b ^ 30) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_l237_237377


namespace brokerage_percentage_l237_237885

theorem brokerage_percentage
  (f : ℝ) (d : ℝ) (c : ℝ) 
  (hf : f = 100)
  (hd : d = 0.08)
  (hc : c = 92.2)
  (h_disc_price : f - f * d = 92) :
  (c - (f - f * d)) / f * 100 = 0.2 := 
by
  sorry

end brokerage_percentage_l237_237885


namespace counterexample_exists_l237_237205

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end counterexample_exists_l237_237205


namespace find_x_plus_y_l237_237762

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end find_x_plus_y_l237_237762


namespace lcm_1_to_10_l237_237169

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l237_237169


namespace student_community_arrangements_l237_237537

theorem student_community_arrangements 
  (students : Finset ℕ)
  (communities : Finset ℕ)
  (h_students : students.card = 4)
  (h_communities : communities.card = 3)
  (student_to_community : ∀ s ∈ students, ∃ c ∈ communities, true)
  (at_least_one_student : ∀ c ∈ communities, ∃ s ∈ students, true) :
  ∃ arrangements : ℕ, arrangements = 36 :=
by 
  use 36 
  sorry

end student_community_arrangements_l237_237537


namespace count_multiples_of_12_between_25_and_200_l237_237035

theorem count_multiples_of_12_between_25_and_200 :
  ∃ n, (∀ i, 25 < i ∧ i < 200 → (∃ k, i = 12 * k)) ↔ n = 14 :=
by
  sorry

end count_multiples_of_12_between_25_and_200_l237_237035


namespace crates_needed_l237_237731

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l237_237731


namespace books_arrangement_l237_237315

-- All conditions provided in Lean as necessary definitions
def num_arrangements (math_books english_books science_books : ℕ) : ℕ :=
  if math_books = 4 ∧ english_books = 6 ∧ science_books = 2 then
    let arrangements_groups := 2 * 3  -- Number of valid group placements
    let arrangements_math := Nat.factorial math_books
    let arrangements_english := Nat.factorial english_books
    let arrangements_science := Nat.factorial science_books
    arrangements_groups * arrangements_math * arrangements_english * arrangements_science
  else
    0

theorem books_arrangement : num_arrangements 4 6 2 = 207360 :=
by
  sorry

end books_arrangement_l237_237315


namespace person_savings_l237_237823

theorem person_savings (income expenditure savings : ℝ) 
  (h1 : income = 18000)
  (h2 : income / expenditure = 5 / 4)
  (h3 : savings = income - expenditure) : 
  savings = 3600 := 
sorry

end person_savings_l237_237823


namespace add_base_6_l237_237403

theorem add_base_6 (a b c : ℕ) (h₀ : a = 3 * 6^3 + 4 * 6^2 + 2 * 6 + 1)
                    (h₁ : b = 4 * 6^3 + 5 * 6^2 + 2 * 6 + 5)
                    (h₂ : c = 1 * 6^4 + 2 * 6^3 + 3 * 6^2 + 5 * 6 + 0) : 
  a + b = c :=
by  
  sorry

end add_base_6_l237_237403


namespace total_shaded_cubes_l237_237511

/-
The large cube consists of 27 smaller cubes, each face is a 3x3 grid.
Opposite faces are shaded in an identical manner, with each face having 5 shaded smaller cubes.
-/

theorem total_shaded_cubes (number_of_smaller_cubes : ℕ)
  (face_shade_pattern : ∀ (face : ℕ), ℕ)
  (opposite_face_same_shade : ∀ (face1 face2 : ℕ), face1 = face2 → face_shade_pattern face1 = face_shade_pattern face2)
  (faces_possible : ∀ (face : ℕ), face < 6)
  (each_face_shaded_squares : ∀ (face : ℕ), face_shade_pattern face = 5)
  : ∃ (n : ℕ), n = 20 :=
by
  sorry

end total_shaded_cubes_l237_237511


namespace katie_total_marbles_l237_237947

def pink_marbles := 13
def orange_marbles := pink_marbles - 9
def purple_marbles := 4 * orange_marbles
def blue_marbles := 2 * purple_marbles
def total_marbles := pink_marbles + orange_marbles + purple_marbles + blue_marbles

theorem katie_total_marbles : total_marbles = 65 := 
by
  -- The proof is omitted here.
  sorry

end katie_total_marbles_l237_237947


namespace original_price_of_stamp_l237_237703

theorem original_price_of_stamp (original_price : ℕ) (h : original_price * (1 / 5 : ℚ) = 6) : original_price = 30 :=
by
  sorry

end original_price_of_stamp_l237_237703


namespace find_a_plus_b_l237_237922

theorem find_a_plus_b :
  let A := {x : ℝ | -1 < x ∧ x < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  let S := {x : ℝ | -1 < x ∧ x < 2}
  ∃ (a b : ℝ), (∀ x, S x ↔ (x^2 + a * x + b < 0)) ∧ a + b = -3 :=
by
  sorry

end find_a_plus_b_l237_237922


namespace compare_abc_l237_237739

/-- Define the constants a, b, and c as given in the problem -/
noncomputable def a : ℝ := -5 / 4 * Real.log (4 / 5)
noncomputable def b : ℝ := Real.exp (1 / 4) / 4
noncomputable def c : ℝ := 1 / 3

/-- The theorem to be proved: a < b < c -/
theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l237_237739


namespace total_height_geometric_solid_l237_237199

-- Definitions corresponding to conditions
def radius_cylinder1 : ℝ := 1
def radius_cylinder2 : ℝ := 3
def height_water_surface_figure2 : ℝ := 20
def height_water_surface_figure3 : ℝ := 28

-- The total height of the geometric solid is 29 cm
theorem total_height_geometric_solid :
  ∃ height_total : ℝ,
    (height_water_surface_figure2 + height_total - height_water_surface_figure3) = 29 :=
sorry

end total_height_geometric_solid_l237_237199


namespace inequality_proof_l237_237953

noncomputable def a : ℝ := (1 / 2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
noncomputable def b : ℝ := (2 * Real.tan (13 * Real.pi / 180)) / (1 - (Real.tan (13 * Real.pi / 180))^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem inequality_proof : a < c ∧ c < b := by
  sorry

end inequality_proof_l237_237953


namespace pentagon_perimeter_l237_237217

-- Problem statement: Given an irregular pentagon with specified side lengths,
-- prove that its perimeter is equal to 52.9 cm.

theorem pentagon_perimeter 
  (a b c d e : ℝ)
  (h1 : a = 5.2)
  (h2 : b = 10.3)
  (h3 : c = 15.8)
  (h4 : d = 8.7)
  (h5 : e = 12.9) 
  : a + b + c + d + e = 52.9 := 
by
  sorry

end pentagon_perimeter_l237_237217


namespace difference_SP_l237_237397

-- Definitions for amounts
variables (P Q R S : ℕ)

-- Conditions given in the problem
def total_amount := P + Q + R + S = 1000
def P_condition := P = 2 * Q
def S_condition := S = 4 * R
def Q_R_equal := Q = R

-- Statement of the problem that needs to be proven
theorem difference_SP (P Q R S : ℕ) (h1 : total_amount P Q R S) 
  (h2 : P_condition P Q) (h3 : S_condition S R) (h4 : Q_R_equal Q R) : 
  S - P = 250 :=
by 
  sorry

end difference_SP_l237_237397


namespace arrangement_of_students_in_communities_l237_237536

theorem arrangement_of_students_in_communities :
  ∃ arr : ℕ, arr = 36 ∧ 4_students_in_3_communities arr :=
by
  -- Definitions and conditions
  let number_of_students := 4
  let number_of_communities := 3
  let each_student_only_goes_to_one_community : Prop := ∀ s ∈ students, ∃ c ∈ communities, s goes to c
  let each_community_must_have_at_least_one_student : Prop := ∀ c ∈ communities, ∃ s ∈ students, c has s
  -- Using these conditions to prove the total number of arrangements
  let total_number_of_arrangements := 36
  
  -- The statement to prove
  have h : ∀ arr, number_of_arrangements arr = total_number_of_arrangements, from by sorry
  exact ⟨total_number_of_arrangements, h total_number_of_arrangements⟩

end arrangement_of_students_in_communities_l237_237536


namespace problem_1_2_a_problem_1_2_b_l237_237382

theorem problem_1_2_a (x : ℝ) : x * (1 - x) ≤ 1 / 4 := sorry

theorem problem_1_2_b (x a : ℝ) : x * (a - x) ≤ a^2 / 4 := sorry

end problem_1_2_a_problem_1_2_b_l237_237382


namespace dog_food_vs_cat_food_l237_237391

-- Define the quantities of dog food and cat food
def dog_food : ℕ := 600
def cat_food : ℕ := 327

-- Define the problem as a statement asserting the required difference
theorem dog_food_vs_cat_food : dog_food - cat_food = 273 := by
  sorry

end dog_food_vs_cat_food_l237_237391


namespace intersection_eq_l237_237251

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l237_237251


namespace noelle_speed_l237_237801

theorem noelle_speed (v d : ℝ) (h1 : d > 0) (h2 : v > 0) 
  (h3 : (2 * d) / ((d / v) + (d / 15)) = 5) : v = 3 := 
sorry

end noelle_speed_l237_237801


namespace compute_expression_l237_237572

theorem compute_expression :
  25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := 
sorry

end compute_expression_l237_237572


namespace increasing_function_range_l237_237749

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 - 2 * x + Real.log x

theorem increasing_function_range (m : ℝ) : (∀ x > 0, m * x + (1 / x) - 2 ≥ 0) ↔ m ≥ 1 := 
by 
  sorry

end increasing_function_range_l237_237749


namespace matt_days_alone_l237_237960

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem matt_days_alone (M P : ℝ) (h1 : work_rate M + work_rate P = work_rate 20) 
  (h2 : 1 - 12 * (work_rate M + work_rate P) = 2 / 5) 
  (h3 : 10 * work_rate M = 2 / 5) : M = 25 :=
by
  sorry

end matt_days_alone_l237_237960


namespace extreme_value_h_at_a_zero_range_of_a_l237_237750

noncomputable def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x / (a * x + 1)

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (Real.exp (-x)) * (g x a)

-- Statement for the first proof problem
theorem extreme_value_h_at_a_zero :
  ∀ x : ℝ, h x 0 ≤ 1 / Real.exp 1 :=
sorry

-- Statement for the second proof problem
theorem range_of_a:
  ∀ x : ℝ, (0 ≤ x → x ≤ 1 / 2) → (f x ≤ g x x) :=
sorry

end extreme_value_h_at_a_zero_range_of_a_l237_237750


namespace max_composite_numbers_l237_237472
open Nat

def is_composite (n : ℕ) : Prop := 1 < n ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

def has_gcd_of_one (l : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ l → b ∈ l → a ≠ b → gcd a b = 1

def valid_composite_numbers (n : ℕ) : Prop :=
  ∀ m ∈ (List.range n).filter is_composite, m < 1500 →

-- Main theorem
theorem max_composite_numbers :
  ∃ l : List ℕ, l.length = 12 ∧ valid_composite_numbers l ∧ has_gcd_of_one l :=
sorry

end max_composite_numbers_l237_237472


namespace intersection_M_N_l237_237278

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237278


namespace wood_cost_l237_237946

theorem wood_cost (C : ℝ) (h1 : 20 * 15 = 300) (h2 : 300 - C = 200) : C = 100 :=
by
  -- The proof is to be filled here, but it is currently skipped with 'sorry'.
  sorry

end wood_cost_l237_237946


namespace three_digit_number_base_10_l237_237544

theorem three_digit_number_base_10 (A B C : ℕ) (x : ℕ)
  (h1 : x = 100 * A + 10 * B + 6)
  (h2 : x = 82 * C + 36)
  (hA : 1 ≤ A ∧ A ≤ 9)
  (hB : 0 ≤ B ∧ B ≤ 9)
  (hC : 0 ≤ C ∧ C ≤ 8) :
  x = 446 := by
  sorry

end three_digit_number_base_10_l237_237544


namespace even_sum_probability_l237_237676

-- Conditions
def prob_even_first_wheel : ℚ := 1 / 4
def prob_odd_first_wheel : ℚ := 3 / 4
def prob_even_second_wheel : ℚ := 2 / 3
def prob_odd_second_wheel : ℚ := 1 / 3

-- Statement: Theorem that the probability of the sum being even is 5/12
theorem even_sum_probability : 
  (prob_even_first_wheel * prob_even_second_wheel) + 
  (prob_odd_first_wheel * prob_odd_second_wheel) = 5 / 12 :=
by
  -- Proof steps would go here
  sorry

end even_sum_probability_l237_237676


namespace conic_sections_of_equation_l237_237577

theorem conic_sections_of_equation :
  (∀ x y : ℝ, y^6 - 6 * x^6 = 3 * y^2 - 8 → y^2 = 6 * x^2 ∨ y^2 = -6 * x^2 + 2) :=
sorry

end conic_sections_of_equation_l237_237577


namespace fraction_to_decimal_l237_237997

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237997


namespace find_a_l237_237605

variable (f g : ℝ → ℝ) (a : ℝ)

-- Conditions
axiom h1 : ∀ x, f x = a^x * g x
axiom h2 : ∀ x, g x ≠ 0
axiom h3 : ∀ x, f x * (deriv g x) > (deriv f x) * g x

-- Question and target proof
theorem find_a (h4 : (f 1) / (g 1) + (f (-1)) / (g (-1)) = 5 / 2) : a = 1 / 2 :=
by sorry

end find_a_l237_237605


namespace geometric_sequence_17th_term_l237_237668

variable {α : Type*} [Field α]

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

theorem geometric_sequence_17th_term :
  ∀ (a r : α),
    a * r ^ 4 = 9 →  -- Fifth term condition
    a * r ^ 12 = 1152 →  -- Thirteenth term condition
    a * r ^ 16 = 36864 :=  -- Seventeenth term conclusion
by
  intros a r h5 h13
  sorry

end geometric_sequence_17th_term_l237_237668


namespace prism_faces_l237_237854

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end prism_faces_l237_237854


namespace square_area_problem_l237_237044

theorem square_area_problem 
  (BM : ℝ) 
  (ABCD_is_divided : Prop)
  (hBM : BM = 4)
  (hABCD_is_divided : ABCD_is_divided) : 
  ∃ (side_length : ℝ), side_length * side_length = 144 := 
by
-- We skip the proof part for this task
sorry

end square_area_problem_l237_237044


namespace sum_of_positive_integers_for_quadratic_l237_237667

theorem sum_of_positive_integers_for_quadratic :
  (∑ k in {k : ℕ | (∃ α β : ℤ, α * β = 18 ∧ α + β = k)}.to_finset) = 39 :=
by
  sorry

end sum_of_positive_integers_for_quadratic_l237_237667


namespace gcd_sum_and_lcm_eq_gcd_l237_237804

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end gcd_sum_and_lcm_eq_gcd_l237_237804


namespace expand_product_l237_237212

theorem expand_product : (2 : ℝ) * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 :=
by
  sorry

end expand_product_l237_237212


namespace g_is_even_l237_237721

noncomputable def g (x : ℝ) := 2 ^ (x ^ 2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  sorry

end g_is_even_l237_237721


namespace intersection_M_N_l237_237292

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l237_237292


namespace expression_value_l237_237837

noncomputable def expr := (1.90 * (1 / (1 - (3: ℝ)^(1/4)))) + (1 / (1 + (3: ℝ)^(1/4))) + (2 / (1 + (3: ℝ)^(1/2)))

theorem expression_value : expr = -2 := 
by
  sorry

end expression_value_l237_237837


namespace dividend_percentage_shares_l237_237191

theorem dividend_percentage_shares :
  ∀ (purchase_price market_value : ℝ) (interest_rate : ℝ),
  purchase_price = 56 →
  market_value = 42 →
  interest_rate = 0.12 →
  ( (interest_rate * purchase_price) / market_value * 100 = 16) :=
by
  intros purchase_price market_value interest_rate h1 h2 h3
  rw [h1, h2, h3]
  -- Calculations were done in solution
  sorry

end dividend_percentage_shares_l237_237191


namespace log_sum_eq_two_l237_237698

theorem log_sum_eq_two:
  ∀ (a b : ℝ), (a = 2) → (b = 50) → (log 10 a + log 10 b = 2) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end log_sum_eq_two_l237_237698


namespace intersection_M_N_l237_237258

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237258


namespace visual_range_percent_increase_l237_237380

-- Define the original and new visual ranges
def original_range : ℝ := 90
def new_range : ℝ := 150

-- Define the desired percent increase as a real number
def desired_percent_increase : ℝ := 66.67

-- The theorem to prove that the visual range is increased by the desired percentage
theorem visual_range_percent_increase :
  ((new_range - original_range) / original_range) * 100 = desired_percent_increase := 
sorry

end visual_range_percent_increase_l237_237380


namespace original_equation_l237_237072

theorem original_equation : 9^2 - 8^2 = 17 := by
  sorry

end original_equation_l237_237072


namespace calculate_expression_l237_237717

theorem calculate_expression : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end calculate_expression_l237_237717


namespace intersection_M_N_l237_237295

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l237_237295


namespace marcus_savings_l237_237462

theorem marcus_savings
  (running_shoes_price : ℝ)
  (running_shoes_discount : ℝ)
  (cashback : ℝ)
  (running_shoes_tax_rate : ℝ)
  (athletic_socks_price : ℝ)
  (athletic_socks_tax_rate : ℝ)
  (bogo : ℝ)
  (performance_tshirt_price : ℝ)
  (performance_tshirt_discount : ℝ)
  (performance_tshirt_tax_rate : ℝ)
  (total_budget : ℝ)
  (running_shoes_final_price : ℝ)
  (athletic_socks_final_price : ℝ)
  (performance_tshirt_final_price : ℝ) :
  running_shoes_price = 120 →
  running_shoes_discount = 30 / 100 →
  cashback = 10 →
  running_shoes_tax_rate = 8 / 100 →
  athletic_socks_price = 25 →
  athletic_socks_tax_rate = 6 / 100 →
  bogo = 2 →
  performance_tshirt_price = 55 →
  performance_tshirt_discount = 10 / 100 →
  performance_tshirt_tax_rate = 7 / 100 →
  total_budget = 250 →
  running_shoes_final_price = (running_shoes_price * (1 - running_shoes_discount) - cashback) * (1 + running_shoes_tax_rate) →
  athletic_socks_final_price = (athletic_socks_price * bogo) * (1 + athletic_socks_tax_rate) / bogo →
  performance_tshirt_final_price = (performance_tshirt_price * (1 - performance_tshirt_discount)) * (1 + performance_tshirt_tax_rate) →
  total_budget - (running_shoes_final_price + athletic_socks_final_price + performance_tshirt_final_price) = 103.86 :=
sorry

end marcus_savings_l237_237462


namespace smallest_n_inequality_l237_237005

variable {x y z : ℝ}

theorem smallest_n_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    (∀ m : ℕ, (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l237_237005


namespace prism_faces_l237_237875

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end prism_faces_l237_237875


namespace L_shape_perimeter_correct_l237_237512

-- Define the dimensions of the rectangles
def rect_height : ℕ := 3
def rect_width : ℕ := 4

-- Define the combined shape and perimeter calculation
def L_shape_perimeter (h w : ℕ) : ℕ := (2 * w) + (2 * h)

theorem L_shape_perimeter_correct : 
  L_shape_perimeter rect_height rect_width = 14 := 
  sorry

end L_shape_perimeter_correct_l237_237512


namespace simplify_and_evaluate_l237_237499

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l237_237499


namespace power_of_binomials_l237_237402

theorem power_of_binomials :
  (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 :=
by
  -- This is a placeholder for the actual proof steps.
  -- We use 'sorry' to indicate that the proof is omitted here.
  sorry

end power_of_binomials_l237_237402


namespace bread_pieces_total_l237_237061

def initial_slices : ℕ := 2
def pieces_per_slice (n : ℕ) : ℕ := n * 4

theorem bread_pieces_total : pieces_per_slice initial_slices = 8 :=
by
  sorry

end bread_pieces_total_l237_237061


namespace sum_of_first_33_terms_arith_seq_l237_237423

noncomputable def sum_arith_prog (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_33_terms_arith_seq :
  ∃ (a_1 d : ℝ), (4 * a_1 + 64 * d = 28) → (sum_arith_prog a_1 d 33 = 231) :=
by
  sorry

end sum_of_first_33_terms_arith_seq_l237_237423


namespace distinct_orders_scoops_l237_237662

-- Conditions
def total_scoops : ℕ := 4
def chocolate_scoops : ℕ := 2
def vanilla_scoops : ℕ := 1
def strawberry_scoops : ℕ := 1

-- Problem statement
theorem distinct_orders_scoops :
  (Nat.factorial total_scoops) / ((Nat.factorial chocolate_scoops) * (Nat.factorial vanilla_scoops) * (Nat.factorial strawberry_scoops)) = 12 := by
  sorry

end distinct_orders_scoops_l237_237662


namespace number_of_men_in_first_group_l237_237444

/-
Given the initial conditions:
1. Some men can color a 48 m long cloth in 2 days.
2. 6 men can color a 36 m long cloth in 1 day.

We need to prove that the number of men in the first group is equal to 9.
-/

theorem number_of_men_in_first_group (M : ℕ)
    (h1 : ∃ (x : ℕ), x * 48 = M * 2)
    (h2 : 6 * 36 = 36 * 1) :
    M = 9 :=
by
sorry

end number_of_men_in_first_group_l237_237444


namespace sum_first_six_terms_l237_237950

variable {S : ℕ → ℝ}

theorem sum_first_six_terms (h2 : S 2 = 4) (h4 : S 4 = 6) : S 6 = 7 := 
  sorry

end sum_first_six_terms_l237_237950


namespace find_integer_n_l237_237760

theorem find_integer_n (n : ℤ) (h : (⌊n^2 / 4⌋ - (⌊n / 2⌋)^2) = 3) : n = 7 :=
sorry

end find_integer_n_l237_237760


namespace abs_sin_diff_le_abs_sin_sub_l237_237477

theorem abs_sin_diff_le_abs_sin_sub (A B : ℝ) (hA : 0 ≤ A) (hA' : A ≤ π) (hB : 0 ≤ B) (hB' : B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| :=
by
  -- Proof would go here
  sorry

end abs_sin_diff_le_abs_sin_sub_l237_237477


namespace gcd_7920_14553_l237_237215

theorem gcd_7920_14553 : Int.gcd 7920 14553 = 11 := by
  sorry

end gcd_7920_14553_l237_237215


namespace three_pow_255_mod_7_l237_237991

theorem three_pow_255_mod_7 : 3^255 % 7 = 6 :=
by 
  have h1 : 3^1 % 7 = 3 := by norm_num
  have h2 : 3^2 % 7 = 2 := by norm_num
  have h3 : 3^3 % 7 = 6 := by norm_num
  have h4 : 3^4 % 7 = 4 := by norm_num
  have h5 : 3^5 % 7 = 5 := by norm_num
  have h6 : 3^6 % 7 = 1 := by norm_num
  sorry

end three_pow_255_mod_7_l237_237991


namespace problem_solution_l237_237589

noncomputable def problem_statement : Prop :=
  ∀ (α β : ℝ), 
    (0 < α ∧ α < Real.pi / 2) →
    (0 < β ∧ β < Real.pi / 2) →
    (Real.sin α = 4 / 5) →
    (Real.cos (α + β) = 5 / 13) →
    (Real.cos β = 63 / 65 ∧ (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4)
    
theorem problem_solution : problem_statement :=
by
  sorry

end problem_solution_l237_237589


namespace ratio_c_d_l237_237325

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -4 / 5 :=
by
  sorry

end ratio_c_d_l237_237325


namespace seqAN_81_eq_640_l237_237597

-- Definitions and hypotheses
def seqAN (n : ℕ) : ℝ := sorry   -- A sequence a_n to be defined properly.

def sumSN (n : ℕ) : ℝ := sorry  -- The sum of the first n terms of a_n.

axiom condition_positivity : ∀ n : ℕ, 0 < seqAN n
axiom condition_a1 : seqAN 1 = 1
axiom condition_sum (n : ℕ) (h : 2 ≤ n) : 
  sumSN n * Real.sqrt (sumSN (n-1)) - sumSN (n-1) * Real.sqrt (sumSN n) = 
  2 * Real.sqrt (sumSN n * sumSN (n-1))

-- Proof problem: 
theorem seqAN_81_eq_640 : seqAN 81 = 640 := by sorry

end seqAN_81_eq_640_l237_237597


namespace students_to_communities_l237_237535

/-- There are 4 students and 3 communities. Each student only goes to one community, 
and each community must have at least 1 student. The total number of permutations where
these conditions are satisfied is 36. -/
theorem students_to_communities : 
  let students : ℕ := 4 in
  let communities : ℕ := 3 in
  (students > 0) ∧ (communities > 0) ∧ (students ≥ communities) ∧ (students ≤ communities * 2) →
  (number_of_arrangements students communities = 36) :=
by
  sorry

/-- The number of different arrangements function is defined here -/
noncomputable def number_of_arrangements : ℕ → ℕ → ℕ
| 4, 3 => 36 -- From the given problem, we know this is 36
| _, _ => 0 -- This is a simplification for this specific problem

end students_to_communities_l237_237535


namespace intersection_eq_l237_237755

def set_M : Set ℝ := { x : ℝ | (x + 3) * (x - 2) < 0 }
def set_N : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq : set_M ∩ set_N = { x : ℝ | 1 ≤ x ∧ x < 2 } := by
  sorry

end intersection_eq_l237_237755


namespace watermelons_left_l237_237809

theorem watermelons_left (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (h1 : initial = 4) (h2 : eaten = 3) : remaining = 1 :=
by
  sorry

end watermelons_left_l237_237809


namespace expand_polynomial_l237_237010

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 := 
by
  sorry

end expand_polynomial_l237_237010


namespace simplified_expression_value_l237_237497

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l237_237497


namespace single_jalapeno_strips_l237_237507

-- Definitions based on conditions
def strips_per_sandwich : ℕ := 4
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8
def total_jalapeno_peppers_used : ℕ := 48
def minutes_per_hour : ℕ := 60

-- Calculate intermediate steps
def total_minutes : ℕ := hours_per_day * minutes_per_hour
def total_sandwiches_served : ℕ := total_minutes / minutes_per_sandwich
def total_strips_needed : ℕ := total_sandwiches_served * strips_per_sandwich

theorem single_jalapeno_strips :
  total_strips_needed / total_jalapeno_peppers_used = 8 := 
by
  sorry

end single_jalapeno_strips_l237_237507


namespace complex_expression_simplified_l237_237840

theorem complex_expression_simplified :
  let z1 := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z2 := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z3 := 1 / (8 * Complex.I^3)
  z1 + z2 + z3 = -1.6 + 0.125 * Complex.I := 
by
  sorry

end complex_expression_simplified_l237_237840


namespace interest_rate_second_type_l237_237198

variable (totalInvestment : ℝ) (interestFirstTypeRate : ℝ) (investmentSecondType : ℝ) (totalInterestRate : ℝ) 
variable [Nontrivial ℝ]

theorem interest_rate_second_type :
    totalInvestment = 100000 ∧
    interestFirstTypeRate = 0.09 ∧
    investmentSecondType = 29999.999999999993 ∧
    totalInterestRate = 9 + 3 / 5 →
    (9.6 * totalInvestment - (interestFirstTypeRate * (totalInvestment - investmentSecondType))) / investmentSecondType = 0.11 :=
by
  sorry

end interest_rate_second_type_l237_237198


namespace cube_surface_area_150_of_volume_125_l237_237517

def volume (s : ℝ) : ℝ := s^3

def surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_surface_area_150_of_volume_125 :
  ∀ (s : ℝ), volume s = 125 → surface_area s = 150 :=
by 
  intros s hs
  sorry

end cube_surface_area_150_of_volume_125_l237_237517


namespace smallest_number_divisible_by_1_to_10_l237_237094

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237094


namespace vityas_miscalculation_l237_237684

/-- Vitya's miscalculated percentages problem -/
theorem vityas_miscalculation :
  ∀ (N : ℕ)
  (acute obtuse nonexistent right depends_geometry : ℕ)
  (H_acute : acute = 5)
  (H_obtuse : obtuse = 5)
  (H_nonexistent : nonexistent = 5)
  (H_right : right = 50)
  (H_total : acute + obtuse + nonexistent + right + depends_geometry = 100),
  depends_geometry = 110 :=
by
  intros
  sorry

end vityas_miscalculation_l237_237684


namespace student_community_arrangements_l237_237534

theorem student_community_arrangements :
  ∃ (students : Fin 4 -> Fin 3), ∀ c : Fin 3, ∃! s : Finset (Fin 4), ∃ (student_assignment : Fin 4 → Fin 3), 
  (∀ s ∈ Finset.univ, student_assignment s ∈ Finset.univ) ∧ 
  (∀ c ∈ Finset.univ, 1 ≤ (Finset.count (λ s, student_assignment s = c) Finset.univ)) ∧ 
  set.univ.card = 4 ∧ 
  ∀ d, d ∈ Finset.univ → Finset.count (λ s, student_assignment s = c) Finset.univ ∈ {1, 2} ∧ 
  Finset.card {Community | (student_assignment.to_finset : Finset (Fin 3)).card = 3} = 1 ∧ 
  (∏ (c : Fin 3), choose 4 2 * 6 + choose 3 1 * choose 4 2 * 2 = 36) :=
sorry

end student_community_arrangements_l237_237534


namespace intersection_of_M_and_N_l237_237256

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l237_237256


namespace num_pos_divisors_30_l237_237313

theorem num_pos_divisors_30 : ∃ n : ℕ, n = 8 ∧ (∀ m : ℕ, m ∣ 30 ↔ m ∈ {1, 2, 3, 5, 6, 10, 15, 30})
 :=
begin
  sorry
end

end num_pos_divisors_30_l237_237313


namespace mr_wang_returns_to_first_floor_electricity_consumed_l237_237962

def floor_changes : List Int := [+6, -3, +10, -8, +12, -7, -10]

-- Total change in floors
def total_floor_change (changes : List Int) : Int :=
  changes.foldl (+) 0

-- Electricity consumption calculation
def total_distance_traveled (height_per_floor : Int) (changes : List Int) : Int :=
  height_per_floor * changes.foldl (λ acc x => acc + abs x) 0

def electricity_consumption (height_per_floor : Int) (consumption_rate : Float) (changes : List Int) : Float :=
  Float.ofInt (total_distance_traveled height_per_floor changes) * consumption_rate

theorem mr_wang_returns_to_first_floor : total_floor_change floor_changes = 0 :=
  by
    sorry

theorem electricity_consumed : electricity_consumption 3 0.2 floor_changes = 33.6 :=
  by
    sorry

end mr_wang_returns_to_first_floor_electricity_consumed_l237_237962


namespace Mary_work_hours_l237_237959

variable (H : ℕ)
variable (weekly_earnings hourly_wage : ℕ)
variable (hours_Tuesday hours_Thursday : ℕ)

def weekly_hours (H : ℕ) : ℕ := 3 * H + hours_Tuesday + hours_Thursday

theorem Mary_work_hours:
  weekly_earnings = 11 * weekly_hours H → hours_Tuesday = 5 →
  hours_Thursday = 5 → weekly_earnings = 407 →
  hourly_wage = 11 → H = 9 :=
by
  intros earnings_eq tues_hours thurs_hours total_earn wage
  sorry

end Mary_work_hours_l237_237959


namespace star_5_3_eq_31_l237_237615

def star (a b : ℤ) : ℤ := a^2 + a * b - b^2

theorem star_5_3_eq_31 : star 5 3 = 31 :=
by
  sorry

end star_5_3_eq_31_l237_237615


namespace probability_of_four_twos_in_five_rolls_l237_237621

theorem probability_of_four_twos_in_five_rolls :
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  total_probability = 3125 / 7776 :=
by
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  show total_probability = 3125 / 7776
  sorry

end probability_of_four_twos_in_five_rolls_l237_237621


namespace counterexample_exists_l237_237206

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end counterexample_exists_l237_237206


namespace necessary_and_sufficient_condition_l237_237384

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 1) ↔ ∀ x : ℝ, (x^2 - 2*x + a > 0) :=
by 
  sorry

end necessary_and_sufficient_condition_l237_237384


namespace smallest_number_divisible_by_1_to_10_l237_237125

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237125


namespace employee_saves_l237_237841

-- Given conditions
def cost_price : ℝ := 500
def markup_percentage : ℝ := 0.15
def employee_discount_percentage : ℝ := 0.15

-- Definitions
def final_retail_price : ℝ := cost_price * (1 + markup_percentage)
def employee_discount_amount : ℝ := final_retail_price * employee_discount_percentage

-- Assertion
theorem employee_saves :
  employee_discount_amount = 86.25 := by
  sorry

end employee_saves_l237_237841


namespace train_distance_in_2_hours_l237_237706

theorem train_distance_in_2_hours :
  (∀ (t : ℕ), t = 90 → (1 / ↑t) * 7200 = 80) :=
by
  sorry

end train_distance_in_2_hours_l237_237706


namespace probability_roll_2_four_times_in_five_rolls_l237_237625

theorem probability_roll_2_four_times_in_five_rolls :
  (∃ (prob_roll_2 : ℚ) (prob_not_roll_2 : ℚ), 
   prob_roll_2 = 1/6 ∧ prob_not_roll_2 = 5/6 ∧ 
   (5 * prob_roll_2^4 * prob_not_roll_2 = 5/72)) :=
sorry

end probability_roll_2_four_times_in_five_rolls_l237_237625


namespace instantaneous_velocity_at_2_l237_237508

def displacement (t : ℝ) : ℝ := 100 * t - 5 * t^2

noncomputable def instantaneous_velocity_at (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  (deriv s) t

theorem instantaneous_velocity_at_2 : instantaneous_velocity_at displacement 2 = 80 :=
by
  sorry

end instantaneous_velocity_at_2_l237_237508


namespace geometric_sequence_sum_l237_237059

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

end geometric_sequence_sum_l237_237059


namespace number_of_blue_stamps_l237_237649

theorem number_of_blue_stamps (
    red_stamps : ℕ := 20
) (
    yellow_stamps : ℕ := 7
) (
    price_per_red_stamp : ℝ := 1.1
) (
    price_per_blue_stamp : ℝ := 0.8
) (
    total_earnings : ℝ := 100
) (
    price_per_yellow_stamp : ℝ := 2
) : red_stamps = 20 ∧ yellow_stamps = 7 ∧ price_per_red_stamp = 1.1 ∧ price_per_blue_stamp = 0.8 ∧ total_earnings = 100 ∧ price_per_yellow_stamp = 2 → ∃ (blue_stamps : ℕ), blue_stamps = 80 :=
by
  sorry

end number_of_blue_stamps_l237_237649


namespace smallest_number_divisible_by_1_to_10_l237_237099

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237099


namespace smallest_divisible_1_to_10_l237_237112

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l237_237112


namespace sin_2x_equals_neg_61_div_72_l237_237783

variable (x y : Real)
variable (h1 : Real.sin y = (3 / 2) * Real.sin x + (2 / 3) * Real.cos x)
variable (h2 : Real.cos y = (2 / 3) * Real.sin x + (3 / 2) * Real.cos x)

theorem sin_2x_equals_neg_61_div_72 : Real.sin (2 * x) = -61 / 72 :=
by
  -- Proof goes here
  sorry

end sin_2x_equals_neg_61_div_72_l237_237783


namespace quartic_two_real_roots_l237_237744

theorem quartic_two_real_roots
  (a b c d e : ℝ)
  (h : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + e - d = 0)
  (ha : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^4 + b * x1^3 + c * x1^2 + d * x1 + e = 0) ∧ (a * x2^4 + b * x2^3 + c * x2^2 + d * x2 + e = 0) := 
  sorry

end quartic_two_real_roots_l237_237744


namespace problem1_problem2_problem3_problem4_problem5_problem6_l237_237895

-- Proof for 238 + 45 × 5 = 463
theorem problem1 : 238 + 45 * 5 = 463 := by
  sorry

-- Proof for 65 × 4 - 128 = 132
theorem problem2 : 65 * 4 - 128 = 132 := by
  sorry

-- Proof for 900 - 108 × 4 = 468
theorem problem3 : 900 - 108 * 4 = 468 := by
  sorry

-- Proof for 369 + (512 - 215) = 666
theorem problem4 : 369 + (512 - 215) = 666 := by
  sorry

-- Proof for 758 - 58 × 9 = 236
theorem problem5 : 758 - 58 * 9 = 236 := by
  sorry

-- Proof for 105 × (81 ÷ 9 - 3) = 630
theorem problem6 : 105 * (81 / 9 - 3) = 630 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l237_237895


namespace starting_cities_l237_237033

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

end starting_cities_l237_237033


namespace math_proof_problem_l237_237076

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

end math_proof_problem_l237_237076


namespace intersection_complement_l237_237756

def M : Set ℝ := { x | x^2 - x - 6 ≥ 0 }
def N : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def neg_R (A : Set ℝ) : Set ℝ := { x | x ∉ A }

theorem intersection_complement (N : Set ℝ) (M : Set ℝ) :
  N ∩ (neg_R M) = { x | -2 < x ∧ x ≤ 1 } := 
by {
  -- Proof goes here
  sorry
}

end intersection_complement_l237_237756


namespace series_sum_eq_4_over_9_l237_237903

noncomputable def sum_series : ℝ := ∑' (k : ℕ), (k+1) / 4^(k+1)

theorem series_sum_eq_4_over_9 : sum_series = 4 / 9 := 
sorry

end series_sum_eq_4_over_9_l237_237903


namespace unique_solution_p_eq_neg8_l237_237955

theorem unique_solution_p_eq_neg8 (p : ℝ) (h : ∀ y : ℝ, 2 * y^2 - 8 * y - p = 0 → ∃! y : ℝ, 2 * y^2 - 8 * y - p = 0) : p = -8 :=
sorry

end unique_solution_p_eq_neg8_l237_237955


namespace cyclic_sums_sine_cosine_l237_237344

theorem cyclic_sums_sine_cosine (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) = 
  2 * (Real.sin α + Real.sin β + Real.sin γ) * 
      (Real.cos α + Real.cos β + Real.cos γ) - 
  2 * (Real.sin α + Real.sin β + Real.sin γ) := 
  sorry

end cyclic_sums_sine_cosine_l237_237344


namespace smallest_number_divisible_by_1_to_10_l237_237127

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237127


namespace lcm_1_to_10_l237_237170

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l237_237170


namespace simplify_expression_l237_237487

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l237_237487


namespace minimize_transport_cost_l237_237977

noncomputable def total_cost (v : ℝ) (a : ℝ) : ℝ :=
  if v > 0 ∧ v ≤ 80 then
    1000 * (v / 4 + a / v)
  else
    0

theorem minimize_transport_cost :
  ∀ v a : ℝ, a = 400 → (0 < v ∧ v ≤ 80) → total_cost v a = 20000 → v = 40 :=
by
  intros v a ha h_dom h_cost
  sorry

end minimize_transport_cost_l237_237977


namespace probability_A_selected_l237_237478

def n : ℕ := 5
def k : ℕ := 2

def total_ways : ℕ := Nat.choose n k  -- C(n, k)

def favorable_ways : ℕ := Nat.choose (n - 1) (k - 1)  -- C(n-1, k-1)

theorem probability_A_selected : (favorable_ways : ℚ) / (total_ways : ℚ) = 2 / 5 :=
by
  sorry

end probability_A_selected_l237_237478


namespace M_inter_N_eq_2_4_l237_237240

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l237_237240


namespace molecular_weight_calculation_l237_237376

-- Define the condition given in the problem
def molecular_weight_of_4_moles := 488 -- molecular weight of 4 moles in g/mol

-- Define the number of moles
def number_of_moles := 4

-- Define the expected molecular weight of 1 mole
def expected_molecular_weight_of_1_mole := 122 -- molecular weight of 1 mole in g/mol

-- Theorem statement
theorem molecular_weight_calculation : 
  molecular_weight_of_4_moles / number_of_moles = expected_molecular_weight_of_1_mole := 
by
  sorry

end molecular_weight_calculation_l237_237376


namespace probability_of_four_twos_in_five_rolls_l237_237620

theorem probability_of_four_twos_in_five_rolls :
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  total_probability = 3125 / 7776 :=
by
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  show total_probability = 3125 / 7776
  sorry

end probability_of_four_twos_in_five_rolls_l237_237620


namespace smallest_number_div_by_1_to_10_l237_237164

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l237_237164


namespace simplify_and_evaluate_l237_237490

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l237_237490


namespace b_n_geometric_a_n_formula_T_n_sum_less_than_2_l237_237917

section problem

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {C_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- Given conditions
axiom seq_a (n : ℕ) : a_n 1 = 1
axiom recurrence (n : ℕ) : 2 * a_n (n + 1) - a_n n = (n - 2) / (n * (n + 1) * (n + 2))
axiom seq_b (n : ℕ) : b_n n = a_n n - 1 / (n * (n + 1))

-- Required proofs
theorem b_n_geometric : ∀ n : ℕ, b_n n = (1 / 2) ^ n := sorry
theorem a_n_formula : ∀ n : ℕ, a_n n = (1 / 2) ^ n + 1 / (n * (n + 1)) := sorry
theorem T_n_sum_less_than_2 : ∀ n : ℕ, T_n n < 2 := sorry

end problem

end b_n_geometric_a_n_formula_T_n_sum_less_than_2_l237_237917


namespace product_of_fractions_l237_237565

theorem product_of_fractions : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end product_of_fractions_l237_237565


namespace total_solutions_l237_237570

-- Definitions and conditions
def tetrahedron_solutions := 1
def cube_solutions := 1
def octahedron_solutions := 3
def dodecahedron_solutions := 2
def icosahedron_solutions := 3

-- Main theorem statement
theorem total_solutions : 
  tetrahedron_solutions + cube_solutions + octahedron_solutions + dodecahedron_solutions + icosahedron_solutions = 10 := by
  sorry

end total_solutions_l237_237570


namespace rectangle_width_l237_237039

theorem rectangle_width (L W : ℝ) (h₁ : 2 * L + 2 * W = 54) (h₂ : W = L + 3) : W = 15 :=
sorry

end rectangle_width_l237_237039


namespace calories_peter_wants_to_eat_l237_237063

-- Definitions for the conditions 
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def total_spent : ℕ := 4

-- Proven statement about the calories Peter wants to eat
theorem calories_peter_wants_to_eat : (total_spent / cost_per_bag) * (chips_per_bag * calories_per_chip) = 480 := by
  sorry

end calories_peter_wants_to_eat_l237_237063


namespace numbers_starting_with_6_div_by_25_no_numbers_divisible_by_35_after_first_digit_removed_l237_237586

-- Definitions based on conditions
def starts_with_six (x : ℕ) : Prop :=
  ∃ n y, x = 6 * 10^n + y

def is_divisible_by_25 (y : ℕ) : Prop :=
  y % 25 = 0

def is_divisible_by_35 (y : ℕ) : Prop :=
  y % 35 = 0

-- Main theorem statements
theorem numbers_starting_with_6_div_by_25:
  ∀ x, starts_with_six x → ∃ k, x = 625 * 10^k :=
by
  sorry

theorem no_numbers_divisible_by_35_after_first_digit_removed:
  ∀ a x, a ≠ 0 → 
  ∃ n, x = a * 10^n + y →
  ¬(is_divisible_by_35 y) :=
by
  sorry

end numbers_starting_with_6_div_by_25_no_numbers_divisible_by_35_after_first_digit_removed_l237_237586


namespace cylinder_volume_ratio_l237_237184

theorem cylinder_volume_ratio :
  let h_A := 8
  let C_A := 5
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * (r_A^2) * h_A
  let h_B := 5
  let C_B := 8
  let r_B := C_B / (2 * Real.pi)
  let V_B := Real.pi * (r_B^2) * h_B
  V_B / V_A = (8 / 5) :=
by
  sorry

end cylinder_volume_ratio_l237_237184


namespace correct_expression_l237_237914

theorem correct_expression (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : a / b = 4 / 3 := by
  sorry

end correct_expression_l237_237914


namespace intersection_of_M_and_N_l237_237252

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l237_237252


namespace prism_faces_l237_237853

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end prism_faces_l237_237853


namespace card_probability_l237_237522

theorem card_probability :
  let deck_size := 52
  let hearts_count := 13
  let first_card_prob := 1 / deck_size
  let second_card_prob := 1 / (deck_size - 1)
  let third_card_prob := hearts_count / (deck_size - 2)
  let total_prob := first_card_prob * second_card_prob * third_card_prob
  total_prob = 13 / 132600 :=
by
  sorry

end card_probability_l237_237522


namespace bundles_burned_in_afternoon_l237_237553

theorem bundles_burned_in_afternoon 
  (morning_burn : ℕ)
  (start_bundles : ℕ)
  (end_bundles : ℕ)
  (h_morning_burn : morning_burn = 4)
  (h_start : start_bundles = 10)
  (h_end : end_bundles = 3)
  : (start_bundles - morning_burn - end_bundles) = 3 := 
by 
  sorry

end bundles_burned_in_afternoon_l237_237553


namespace smallest_number_divisible_1_to_10_l237_237174

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237174


namespace max_collisions_l237_237795

-- Define the problem
theorem max_collisions (n : ℕ) (hn : n > 0) : 
  ∃ C : ℕ, C = (n * (n - 1)) / 2 := 
sorry

end max_collisions_l237_237795


namespace find_t_l237_237614

variable (s t : ℚ) -- Using the rational numbers since the correct answer involves a fraction

theorem find_t (h1 : 8 * s + 7 * t = 145) (h2 : s = t + 3) : t = 121 / 15 :=
by 
  sorry

end find_t_l237_237614


namespace simplify_trig_expression_l237_237070

open Real

/-- 
Given that θ is in the interval (π/2, π), simplify the expression 
( sin θ / sqrt (1 - sin^2 θ) ) + ( sqrt (1 - cos^2 θ) / cos θ ) to 0.
-/
theorem simplify_trig_expression (θ : ℝ) (hθ1 : π / 2 < θ) (hθ2 : θ < π) :
  (sin θ / sqrt (1 - sin θ ^ 2)) + (sqrt (1 - cos θ ^ 2) / cos θ) = 0 :=
by 
  sorry

end simplify_trig_expression_l237_237070


namespace uv_divisible_by_3_l237_237659

theorem uv_divisible_by_3
  {u v : ℤ}
  (h : 9 ∣ (u^2 + u * v + v^2)) :
  3 ∣ u ∧ 3 ∣ v :=
sorry

end uv_divisible_by_3_l237_237659


namespace scientific_notation_3080000_l237_237582

theorem scientific_notation_3080000 : (3080000 : ℝ) = 3.08 * 10^6 := 
by
  sorry

end scientific_notation_3080000_l237_237582


namespace Jake_weight_correct_l237_237961

def Mildred_weight : ℕ := 59
def Carol_weight : ℕ := Mildred_weight + 9
def Jake_weight : ℕ := 2 * Carol_weight

theorem Jake_weight_correct : Jake_weight = 136 := by
  sorry

end Jake_weight_correct_l237_237961


namespace smallest_divisible_1_to_10_l237_237111

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l237_237111


namespace train_length_l237_237888

/-- A train crosses a tree in 120 seconds. It takes 230 seconds to pass a platform 1100 meters long.
    How long is the train? -/
theorem train_length (L : ℝ) (V : ℝ)
    (h1 : V = L / 120)
    (h2 : V = (L + 1100) / 230) :
    L = 1200 :=
by
  sorry

end train_length_l237_237888


namespace b_95_mod_49_l237_237457

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 7^n + 9^n

-- Goal: Prove that the remainder when b 95 is divided by 49 is 28
theorem b_95_mod_49 : b 95 % 49 = 28 := 
by
  sorry

end b_95_mod_49_l237_237457


namespace river_depth_mid_June_l237_237775

theorem river_depth_mid_June (D : ℝ) : 
    (∀ (mid_May mid_June mid_July : ℝ),
    mid_May = 5 →
    mid_June = mid_May + D →
    mid_July = 3 * mid_June →
    mid_July = 45) →
    D = 10 :=
by
    sorry

end river_depth_mid_June_l237_237775


namespace good_quadruple_inequality_l237_237475

theorem good_quadruple_inequality {p a b c : ℕ} (hp : Nat.Prime p) (hodd : p % 2 = 1) 
(habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
(hab : (a * b + 1) % p = 0) (hbc : (b * c + 1) % p = 0) (hca : (c * a + 1) % p = 0) :
  p + 2 ≤ (a + b + c) / 3 := 
by
  sorry

end good_quadruple_inequality_l237_237475


namespace sum_odd_divisors_300_l237_237414

theorem sum_odd_divisors_300 : 
  ∑ d in (Nat.divisors 300).filter Nat.Odd, d = 124 := 
sorry

end sum_odd_divisors_300_l237_237414


namespace complement_M_l237_237447

section ComplementSet

variable (x : ℝ)

def M : Set ℝ := {x | 1 / x < 1}

theorem complement_M : {x | 0 ≤ x ∧ x ≤ 1} = Mᶜ := sorry

end ComplementSet

end complement_M_l237_237447


namespace simplify_expression_l237_237485

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l237_237485


namespace num_ways_to_queue_ABC_l237_237678

-- Definitions for the problem
def num_people : ℕ := 5
def fixed_order_positions : ℕ := 3

-- Lean statement to prove the problem
theorem num_ways_to_queue_ABC (h : num_people = 5) (h_fop : fixed_order_positions = 3) : 
  (Nat.factorial num_people / Nat.factorial (num_people - fixed_order_positions)) * 1 = 20 := 
by
  sorry

end num_ways_to_queue_ABC_l237_237678


namespace geometric_sequence_S6_l237_237951

-- Definitions for the sum of terms in a geometric sequence.
noncomputable def S : ℕ → ℝ
| 2 := 4
| 4 := 6
| _ := sorry

-- Theorem statement for the given problem.
theorem geometric_sequence_S6 : S 6 = 7 :=
by
  -- Statements reflecting the given conditions.
  have h1 : S 2 = 4 := rfl
  have h2 : S 4 = 6 := rfl
  sorry  -- The proof will be filled in, but is not required for this task.

end geometric_sequence_S6_l237_237951


namespace max_composite_numbers_l237_237471

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end max_composite_numbers_l237_237471


namespace crates_needed_l237_237729

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l237_237729


namespace profit_percentage_l237_237714

theorem profit_percentage (cost_price selling_price profit_percentage : ℚ) 
  (h_cost_price : cost_price = 240) 
  (h_selling_price : selling_price = 288) 
  (h_profit_percentage : profit_percentage = 20) : 
  profit_percentage = ((selling_price - cost_price) / cost_price) * 100 := 
by 
  sorry

end profit_percentage_l237_237714


namespace simplify_and_evaluate_l237_237501

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l237_237501


namespace smallest_divisible_1_to_10_l237_237110

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l237_237110


namespace remainder_not_power_of_4_l237_237203

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end remainder_not_power_of_4_l237_237203


namespace verify_segment_lengths_l237_237673

noncomputable def segment_lengths_proof : Prop :=
  let a := 2
  let b := 3
  let alpha := Real.arccos (5 / 16)
  let segment1 := 4 / 3
  let segment2 := 2 / 3
  let segment3 := 2
  let segment4 := 1
  ∀ (s1 s2 s3 s4 : ℝ), 
    (s1 = segment1 ∧ s2 = segment2 ∧ s3 = segment3 ∧ s4 = segment4) ↔
    -- Parallelogram sides and angle constraints
    (s1 + s2 = a ∧ s3 + s4 = b ∧ 
     -- Mutually perpendicular lines divide into equal areas
     (s1 * s3 * Real.sin alpha / 2 = s2 * s4 * Real.sin alpha / 2) )

-- Placeholder for proof
theorem verify_segment_lengths : segment_lengths_proof :=
  sorry

end verify_segment_lengths_l237_237673


namespace problem_l237_237761

variable {x y : ℝ}

theorem problem (h : x < y) : 3 - x > 3 - y :=
sorry

end problem_l237_237761


namespace change_proof_l237_237339

-- Definitions of the given conditions
def lee_money : ℕ := 10
def friend_money : ℕ := 8
def chicken_wings_cost : ℕ := 6
def chicken_salad_cost : ℕ := 4
def soda_cost : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Statement of the theorem
theorem change_proof : 
  let total_money : ℕ := lee_money + friend_money,
      meal_cost_before_tax : ℕ := chicken_wings_cost + chicken_salad_cost + num_sodas * soda_cost,
      total_meal_cost : ℕ := meal_cost_before_tax + tax
  in total_money - total_meal_cost = 3 := 
by
  -- We skip the proof, as it's not required per instructions
  sorry

end change_proof_l237_237339


namespace rectangular_field_area_l237_237883

theorem rectangular_field_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) :
  w * l = 243 :=
by
  -- Proof goes here
  sorry

end rectangular_field_area_l237_237883


namespace frustumViews_l237_237669

-- Define the notion of a frustum
structure Frustum where
  -- You may add necessary geometric properties of a frustum if needed
  
-- Define a function to describe the view of the frustum
def frontView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def sideView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def topView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type

-- Define the properties of the views
def isCongruentIsoscelesTrapezoid (fig : Type) : Prop := sorry -- Define property for congruent isosceles trapezoid
def isTwoConcentricCircles (fig : Type) : Prop := sorry -- Define property for two concentric circles

-- State the theorem based on the given problem
theorem frustumViews (f : Frustum) :
  isCongruentIsoscelesTrapezoid (frontView f) ∧ 
  isCongruentIsoscelesTrapezoid (sideView f) ∧ 
  isTwoConcentricCircles (topView f) := 
sorry

end frustumViews_l237_237669


namespace find_two_digit_number_l237_237393

theorem find_two_digit_number (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : n % 2 = 0)
  (h3 : (n + 1) % 3 = 0)
  (h4 : (n + 2) % 4 = 0)
  (h5 : (n + 3) % 5 = 0) : n = 62 :=
by
  sorry

end find_two_digit_number_l237_237393


namespace blueberry_pies_correct_l237_237724

def total_pies := 36
def apple_pie_ratio := 3
def blueberry_pie_ratio := 4
def cherry_pie_ratio := 5

-- Total parts in the ratio
def total_ratio_parts := apple_pie_ratio + blueberry_pie_ratio + cherry_pie_ratio

-- Number of pies per part
noncomputable def pies_per_part := total_pies / total_ratio_parts

-- Number of blueberry pies
noncomputable def blueberry_pies := blueberry_pie_ratio * pies_per_part

theorem blueberry_pies_correct : blueberry_pies = 12 := 
by
  sorry

end blueberry_pies_correct_l237_237724


namespace intersection_of_M_and_N_l237_237233

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l237_237233


namespace smallest_number_divisible_by_1_through_10_l237_237154

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l237_237154


namespace max_composite_numbers_l237_237467

theorem max_composite_numbers (S : Finset ℕ) (h1 : ∀ n ∈ S, n < 1500) (h2 : ∀ m n ∈ S, m ≠ n → Nat.gcd m n = 1) : S.card ≤ 12 := sorry

end max_composite_numbers_l237_237467


namespace smallest_multiple_1_through_10_l237_237139

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l237_237139


namespace stuffed_animal_sales_l237_237069

theorem stuffed_animal_sales (Q T J : ℕ) 
  (h1 : Q = 100 * T) 
  (h2 : J = T + 15) 
  (h3 : Q = 2000) : 
  Q - J = 1965 := 
by
  sorry

end stuffed_animal_sales_l237_237069


namespace arrangements_with_AB_together_l237_237671

theorem arrangements_with_AB_together (n : ℕ) (A B: ℕ) (students: Finset ℕ) (h₁ : students.card = 6) (h₂ : A ∈ students) (h₃ : B ∈ students):
  ∃! (count : ℕ), count = 240 :=
by
  sorry

end arrangements_with_AB_together_l237_237671


namespace opposite_of_negative_five_l237_237083

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end opposite_of_negative_five_l237_237083


namespace handshakes_correct_l237_237562

-- Definitions based on conditions
def num_gremlins : ℕ := 25
def num_imps : ℕ := 20
def num_imps_shaking_hands_among_themselves : ℕ := num_imps / 2
def comb (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the total handshakes
def total_handshakes : ℕ :=
  (comb num_gremlins 2) + -- Handshakes among gremlins
  (comb num_imps_shaking_hands_among_themselves 2) + -- Handshakes among half the imps
  (num_gremlins * num_imps) -- Handshakes between all gremlins and all imps

-- The theorem to be proved
theorem handshakes_correct : total_handshakes = 845 := by
  sorry

end handshakes_correct_l237_237562


namespace rowing_distance_l237_237181

theorem rowing_distance :
  let row_speed := 4 -- kmph
  let river_speed := 2 -- kmph
  let total_time := 1.5 -- hours
  ∃ d, 
    let downstream_speed := row_speed + river_speed
    let upstream_speed := row_speed - river_speed
    let downstream_time := d / downstream_speed
    let upstream_time := d / upstream_speed
    downstream_time + upstream_time = total_time ∧ d = 2.25 :=
by
  sorry

end rowing_distance_l237_237181


namespace cos_4theta_l237_237320

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (4 * θ) = 17 / 81 := 
by 
  sorry

end cos_4theta_l237_237320


namespace prob_sum_7_9_11_correct_l237_237509

def die1 : List ℕ := [1, 2, 3, 3, 4, 4]
def die2 : List ℕ := [2, 2, 5, 6, 7, 8]

def prob_sum_7_9_11 : ℚ := 
  (1/6 * 1/6 + 1/6 * 1/6) + 2/6 * 3/6

theorem prob_sum_7_9_11_correct :
  prob_sum_7_9_11 = 4 / 9 := 
by
  sorry

end prob_sum_7_9_11_correct_l237_237509


namespace find_b_l237_237681

theorem find_b (a b c : ℕ) (h1 : 2 * b = a + c) (h2 : b^2 = c * (a + 1)) (h3 : b^2 = a * (c + 2)) : b = 12 :=
by 
  sorry

end find_b_l237_237681


namespace certain_number_l237_237933

theorem certain_number (a n b : ℕ) (h1 : a = 30) (h2 : a * n = b^2) (h3 : ∀ m : ℕ, (m * n = b^2 → a ≤ m)) :
  n = 30 :=
by
  sorry

end certain_number_l237_237933


namespace line_equations_satisfy_conditions_l237_237680

-- Definitions and conditions:
def intersects_at_distance (k m b : ℝ) : Prop :=
  |(k^2 + 7*k + 12) - (m*k + b)| = 8

def passes_through_point (m b : ℝ) : Prop :=
  7 = 2*m + b

def line_equation_valid (m b : ℝ) : Prop :=
  b ≠ 0

-- Main theorem:
theorem line_equations_satisfy_conditions :
  (line_equation_valid 1 5 ∧ passes_through_point 1 5 ∧ 
  ∃ k, intersects_at_distance k 1 5) ∨
  (line_equation_valid 5 (-3) ∧ passes_through_point 5 (-3) ∧ 
  ∃ k, intersects_at_distance k 5 (-3)) :=
by
  sorry

end line_equations_satisfy_conditions_l237_237680


namespace percent_decrease_in_cost_l237_237770

theorem percent_decrease_in_cost (cost_1990 cost_2010 : ℕ) (h1 : cost_1990 = 35) (h2 : cost_2010 = 5) : 
  ((cost_1990 - cost_2010) * 100 / cost_1990 : ℚ) = 86 := 
by
  sorry

end percent_decrease_in_cost_l237_237770


namespace pictures_hung_in_new_galleries_l237_237197

noncomputable def total_pencils_used : ℕ := 218
noncomputable def pencils_per_picture : ℕ := 5
noncomputable def pencils_per_exhibition : ℕ := 3

noncomputable def pictures_initial : ℕ := 9
noncomputable def galleries_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]
noncomputable def total_exhibitions : ℕ := 1 + galleries_requests.length

theorem pictures_hung_in_new_galleries :
  let total_pencils_for_signing := total_exhibitions * pencils_per_exhibition
  let total_pencils_for_drawing := total_pencils_used - total_pencils_for_signing
  let total_pictures_drawn := total_pencils_for_drawing / pencils_per_picture
  let pictures_in_new_galleries := total_pictures_drawn - pictures_initial
  pictures_in_new_galleries = 29 :=
by
  sorry

end pictures_hung_in_new_galleries_l237_237197


namespace inequality_proof_l237_237656

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_proof_l237_237656


namespace sum_of_reciprocals_l237_237089

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1/x + 1/y = 3/8 := by
  sorry

end sum_of_reciprocals_l237_237089


namespace phil_final_quarters_l237_237064

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

end phil_final_quarters_l237_237064


namespace product_of_digits_of_N_l237_237710

theorem product_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2485) : 
  (N.digits 10).prod = 0 :=
sorry

end product_of_digits_of_N_l237_237710


namespace original_salary_l237_237531

theorem original_salary (S : ℝ) (h : 1.10 * S * 0.95 = 3135) : S = 3000 := 
by 
  sorry

end original_salary_l237_237531


namespace number_of_divisors_30_l237_237312

theorem number_of_divisors_30 : 
  ∃ (d : ℕ), d = 2 * 2 * 2 ∧ d = 8 :=
  by sorry

end number_of_divisors_30_l237_237312


namespace inequality_solution_set_nonempty_l237_237431

-- Define the statement
theorem inequality_solution_set_nonempty (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) ↔ m > 2 :=
by
  sorry

end inequality_solution_set_nonempty_l237_237431


namespace point_coordinates_with_respect_to_origin_l237_237975

theorem point_coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  sorry

end point_coordinates_with_respect_to_origin_l237_237975


namespace perpendicular_vectors_l237_237610

open scoped BigOperators

noncomputable def i : ℝ × ℝ := (1, 0)
noncomputable def j : ℝ × ℝ := (0, 1)
noncomputable def u : ℝ × ℝ := (1, 3)
noncomputable def v : ℝ × ℝ := (3, -1)

theorem perpendicular_vectors :
  (u.1 * v.1 + u.2 * v.2) = 0 :=
by
  have hi : i = (1, 0) := rfl
  have hj : j = (0, 1) := rfl
  have hu : u = (1, 3) := rfl
  have hv : v = (3, -1) := rfl
  -- using the dot product definition for perpendicularity
  sorry

end perpendicular_vectors_l237_237610


namespace statue_selling_price_l237_237784

/-- Problem conditions -/
def original_cost : ℤ := 550
def profit_percentage : ℝ := 0.20

/-- Proof problem statement -/
theorem statue_selling_price : original_cost + profit_percentage * original_cost = 660 := by
  sorry

end statue_selling_price_l237_237784


namespace program_output_eq_l237_237404

theorem program_output_eq : ∀ (n : ℤ), n^2 + 3 * n - (2 * n^2 - n) = -n^2 + 4 * n := by
  intro n
  sorry

end program_output_eq_l237_237404


namespace intersection_M_N_l237_237279

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237279


namespace ruby_height_is_192_l237_237442

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end ruby_height_is_192_l237_237442


namespace original_price_of_candy_box_is_8_l237_237890

-- Define the given conditions
def candy_box_price_after_increase : ℝ := 10
def candy_box_increase_rate : ℝ := 1.25

-- Define the original price of the candy box
noncomputable def original_candy_box_price : ℝ := candy_box_price_after_increase / candy_box_increase_rate

-- The theorem to prove
theorem original_price_of_candy_box_is_8 :
  original_candy_box_price = 8 := by
  sorry

end original_price_of_candy_box_is_8_l237_237890


namespace fraction_to_decimal_l237_237999

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237999


namespace smallest_divisible_by_1_to_10_l237_237145

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l237_237145


namespace simplify_fraction_l237_237353

theorem simplify_fraction :
  (1 / (1 + Real.sqrt 3) * 1 / (1 - Real.sqrt 5)) = 
  (1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15)) :=
by
  sorry

end simplify_fraction_l237_237353


namespace Helga_articles_written_this_week_l237_237926

def articles_per_30_minutes : ℕ := 5
def work_hours_per_day : ℕ := 4
def work_days_per_week : ℕ := 5
def extra_hours_thursday : ℕ := 2
def extra_hours_friday : ℕ := 3

def articles_per_hour : ℕ := articles_per_30_minutes * 2
def regular_daily_articles : ℕ := articles_per_hour * work_hours_per_day
def regular_weekly_articles : ℕ := regular_daily_articles * work_days_per_week
def extra_thursday_articles : ℕ := articles_per_hour * extra_hours_thursday
def extra_friday_articles : ℕ := articles_per_hour * extra_hours_friday
def extra_weekly_articles : ℕ := extra_thursday_articles + extra_friday_articles
def total_weekly_articles : ℕ := regular_weekly_articles + extra_weekly_articles

theorem Helga_articles_written_this_week : total_weekly_articles = 250 := by
  sorry

end Helga_articles_written_this_week_l237_237926


namespace exists_abc_l237_237599

theorem exists_abc (n k : ℕ) (hn : n > 20) (hk : k > 1) (hdiv : k^2 ∣ n) : 
  ∃ (a b c : ℕ), n = a * b + b * c + c * a :=
by
  sorry

end exists_abc_l237_237599


namespace wood_burned_afternoon_l237_237551

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon_l237_237551


namespace latest_leave_time_correct_l237_237711

-- Define the conditions
def flight_time := 20 -- 8:00 pm in 24-hour format
def check_in_early := 2 -- 2 hours early
def drive_time := 45 -- 45 minutes
def park_time := 15 -- 15 minutes

-- Define the target time to be at the airport
def at_airport_time := flight_time - check_in_early -- 18:00 or 6:00 pm

-- Total travel time required (minutes)
def total_travel_time := drive_time + park_time -- 60 minutes

-- Convert total travel time to hours
def travel_time_in_hours : ℕ := total_travel_time / 60

-- Define the latest time to leave the house
def latest_leave_time := at_airport_time - travel_time_in_hours

-- Theorem to state the equivalence of the latest time they can leave their house
theorem latest_leave_time_correct : latest_leave_time = 17 :=
    by
    sorry

end latest_leave_time_correct_l237_237711


namespace geometric_sum_S6_l237_237952

noncomputable def S (n : ℕ) : ℝ := sorry  -- Assume S is defined as the sum of the first n terms of a geometric sequence

theorem geometric_sum_S6 :
  (S 2 = 4) ∧ (S 4 = 6) → S 6 = 7 :=
by
  intros h
  cases h with hS2 hS4
  sorry -- Complete the proof accordingly

end geometric_sum_S6_l237_237952


namespace deposit_is_3000_l237_237078

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

end deposit_is_3000_l237_237078


namespace find_pos_ints_a_b_c_p_l237_237904

theorem find_pos_ints_a_b_c_p (a b c p : ℕ) (hp : Nat.Prime p) : 
  73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 ↔
  (p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4) :=
by
  sorry

end find_pos_ints_a_b_c_p_l237_237904


namespace lcm_1_to_10_l237_237100

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l237_237100


namespace prism_faces_l237_237869

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end prism_faces_l237_237869


namespace proposition_p_and_not_q_is_true_l237_237740

-- Define proposition p
def p : Prop := ∀ x > 0, Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : Real, a > b → a^2 > b^2

-- State the theorem to be proven in Lean
theorem proposition_p_and_not_q_is_true : p ∧ ¬q :=
by
  -- Sorry placeholder for the proof
  sorry

end proposition_p_and_not_q_is_true_l237_237740


namespace radius_of_inscribed_circle_l237_237835

noncomputable def inscribed_circle_radius (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_of_inscribed_circle :
  inscribed_circle_radius 6 8 10 = 2 :=
by
  sorry

end radius_of_inscribed_circle_l237_237835


namespace sum_of_squares_transform_l237_237068

def isSumOfThreeSquaresDivByThree (N : ℕ) : Prop := 
  ∃ (a b c : ℤ), N = a^2 + b^2 + c^2 ∧ (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c)

def isSumOfThreeSquaresNotDivByThree (N : ℕ) : Prop := 
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z)

theorem sum_of_squares_transform {N : ℕ} :
  isSumOfThreeSquaresDivByThree N → isSumOfThreeSquaresNotDivByThree N :=
sorry

end sum_of_squares_transform_l237_237068


namespace prove_y_minus_x_l237_237516

theorem prove_y_minus_x (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 7 / 8) : y - x = 100 / 3 := 
by
  sorry

end prove_y_minus_x_l237_237516


namespace inequality_proof_l237_237657

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_proof_l237_237657


namespace determinant_matrix_equivalence_l237_237604

variable {R : Type} [CommRing R]

theorem determinant_matrix_equivalence
  (x y z w : R)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end determinant_matrix_equivalence_l237_237604


namespace find_x_l237_237543

theorem find_x (x : ℕ) (h : 220030 = (x + 445) * (2 * (x - 445)) + 30) : x = 555 := 
sorry

end find_x_l237_237543


namespace prism_faces_l237_237873

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end prism_faces_l237_237873


namespace largest_reciprocal_l237_237993

theorem largest_reciprocal: 
  let A := -(1 / 4)
  let B := 2 / 7
  let C := -2
  let D := 3
  let E := -(3 / 2)
  let reciprocal (x : ℚ) := 1 / x
  reciprocal B > reciprocal A ∧
  reciprocal B > reciprocal C ∧
  reciprocal B > reciprocal D ∧
  reciprocal B > reciprocal E :=
by
  sorry

end largest_reciprocal_l237_237993


namespace value_of_a_plus_b_l237_237429

theorem value_of_a_plus_b (a b : ℝ) (h : |a - 2| = -(b + 5)^2) : a + b = -3 :=
sorry

end value_of_a_plus_b_l237_237429


namespace range_of_t_range_of_a_l237_237022

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

end range_of_t_range_of_a_l237_237022


namespace simplify_and_evaluate_l237_237491

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l237_237491


namespace probability_roll_2_four_times_in_five_rolls_l237_237626

theorem probability_roll_2_four_times_in_five_rolls :
  (∃ (prob_roll_2 : ℚ) (prob_not_roll_2 : ℚ), 
   prob_roll_2 = 1/6 ∧ prob_not_roll_2 = 5/6 ∧ 
   (5 * prob_roll_2^4 * prob_not_roll_2 = 5/72)) :=
sorry

end probability_roll_2_four_times_in_five_rolls_l237_237626


namespace father_son_fish_problem_l237_237183

variables {F S x : ℕ}

theorem father_son_fish_problem (h1 : F - x = S + x) (h2 : F + x = 2 * (S - x)) : 
  (F - S) / S = 2 / 5 :=
by sorry

end father_son_fish_problem_l237_237183


namespace inequality_a_b_c_l237_237655

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_a_b_c_l237_237655


namespace union_of_sets_l237_237956

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Define the set representing the union's result
def C : Set ℝ := { x | -1 < x ∧ x < 4 }

-- The theorem statement
theorem union_of_sets : ∀ x : ℝ, (x ∈ (A ∪ B) ↔ x ∈ C) :=
by
  sorry

end union_of_sets_l237_237956


namespace positive_divisors_multiple_of_5_l237_237309

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end positive_divisors_multiple_of_5_l237_237309


namespace total_goals_scored_l237_237911

theorem total_goals_scored (g1 t1 g2 t2 : ℕ)
  (h1 : g1 = 2)
  (h2 : g1 = t1 - 3)
  (h3 : t2 = 6)
  (h4 : g2 = t2 - 2) :
  g1 + t1 + g2 + t2 = 17 :=
by
  sorry

end total_goals_scored_l237_237911


namespace intersection_of_M_and_N_l237_237232

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l237_237232


namespace lucas_pay_per_window_l237_237647

-- Conditions
def num_floors : Nat := 3
def windows_per_floor : Nat := 3
def days_to_finish : Nat := 6
def penalty_rate : Nat := 3
def penalty_amount : Nat := 1
def final_payment : Nat := 16

-- Theorem statement
theorem lucas_pay_per_window :
  let total_windows := num_floors * windows_per_floor
  let total_penalty := penalty_amount * (days_to_finish / penalty_rate)
  let original_payment := final_payment + total_penalty
  let payment_per_window := original_payment / total_windows
  payment_per_window = 2 :=
by
  sorry

end lucas_pay_per_window_l237_237647


namespace first_part_lent_years_l237_237886

theorem first_part_lent_years (x n : ℕ) (total_sum second_sum : ℕ) (rate1 rate2 years2 : ℝ) :
  total_sum = 2743 →
  second_sum = 1688 →
  rate1 = 3 →
  rate2 = 5 →
  years2 = 3 →
  (x = total_sum - second_sum) →
  (x * n * rate1 / 100 = second_sum * rate2 * years2 / 100) →
  n = 8 :=
by
  sorry

end first_part_lent_years_l237_237886


namespace parametric_circle_section_l237_237594

theorem parametric_circle_section (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x = 4 - Real.cos θ ∧ y = 1 - Real.sin θ) ∧ (4 - x)^2 + (1 - y)^2 = 1 :=
sorry

end parametric_circle_section_l237_237594


namespace average_rate_of_change_l237_237820

noncomputable def f (x : ℝ) : ℝ :=
  -2 * x^2 + 1

theorem average_rate_of_change : 
  ((f 1 - f 0) / (1 - 0)) = -2 :=
by
  sorry

end average_rate_of_change_l237_237820


namespace hyperbola_asymptotes_l237_237608

theorem hyperbola_asymptotes (p : ℝ) (h : (p / 2, 0) ∈ {q : ℝ × ℝ | q.1 ^ 2 / 8 - q.2 ^ 2 / p = 1}) :
  (y = x) ∨ (y = -x) :=
by
  sorry

end hyperbola_asymptotes_l237_237608


namespace f_at_neg_one_l237_237616

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_at_neg_one : f (-1) = 0 := by
  sorry

end f_at_neg_one_l237_237616


namespace group4_exceeds_group2_group4_exceeds_group3_l237_237884

-- Define conditions
def score_group1 : Int := 100
def score_group2 : Int := 150
def score_group3 : Int := -400
def score_group4 : Int := 350
def score_group5 : Int := -100

-- Theorem 1: Proving Group 4 exceeded Group 2 by 200 points
theorem group4_exceeds_group2 :
  score_group4 - score_group2 = 200 := by
  sorry

-- Theorem 2: Proving Group 4 exceeded Group 3 by 750 points
theorem group4_exceeds_group3 :
  score_group4 - score_group3 = 750 := by
  sorry

end group4_exceeds_group2_group4_exceeds_group3_l237_237884


namespace base9_square_multiple_of_3_ab4c_l237_237935

theorem base9_square_multiple_of_3_ab4c (a b c : ℕ) (N : ℕ) (h1 : a ≠ 0)
  (h2 : N = a * 9^3 + b * 9^2 + 4 * 9 + c)
  (h3 : ∃ k : ℕ, N = k^2)
  (h4 : N % 3 = 0) :
  c = 0 :=
sorry

end base9_square_multiple_of_3_ab4c_l237_237935


namespace complex_div_imag_unit_l237_237036

theorem complex_div_imag_unit (i : ℂ) (h : i^2 = -1) : (1 + i) / (1 - i) = i :=
sorry

end complex_div_imag_unit_l237_237036


namespace dasha_rectangle_l237_237412

theorem dasha_rectangle:
  ∃ (a b c : ℤ), a * (2 * b + 2 * c - a) = 43 ∧ a = 1 ∧ b + c = 22 :=
by
  sorry

end dasha_rectangle_l237_237412


namespace card_draws_with_conditions_l237_237366

-- Definitions based on the conditions in the problem
structure Card :=
(color : Fin 3) -- Three colors: red (0), yellow (1), green (2)
(letter : Fin 5) -- Letters A, B, C, D, E

-- Helper function to count the number of ways to draw cards under given conditions
def valid_draws : Finset (Finset Card) :=
  { s : Finset Card | s.cardinality = 4 ∧ (∀ c : Fin 3, ∃ card ∈ s, card.color = c)
    ∧ function.injective (λ card, card.letter) }.to_finset

-- Main theorem statement
theorem card_draws_with_conditions :
  Fintype.card valid_draws = 360 :=
sorry

end card_draws_with_conditions_l237_237366


namespace GP_GQ_GR_proof_l237_237045

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

end GP_GQ_GR_proof_l237_237045


namespace prism_faces_l237_237877

-- Define basic properties and conditions
def is_prism_with_L_gon_base (edges : ℕ) (L : ℕ) :=
  edges = 3 * L

noncomputable def number_of_faces (L : ℕ) : ℕ :=
  L + 2  -- L lateral faces and 2 bases

-- Main statement
theorem prism_faces (edges : ℕ) (L : ℕ) (h : edges = 3 * L) : number_of_faces L = 8 :=
by
  -- Instantiate the given condition
  have hL : L = 6 := sorry
  -- Prove the number of faces calculation
  have h_faces : number_of_faces L = number_of_faces 6 := by rw hL
  have h_faces_calc : number_of_faces 6 = 8 := sorry
  exact (h_faces.trans h_faces_calc)

end prism_faces_l237_237877


namespace intersection_A_B_at_3_range_of_a_l237_237642

open Set

-- Definitions from the condition
def A (x : ℝ) : Prop := abs x ≥ 2
def B (x a : ℝ) : Prop := (x - 2 * a) * (x + 3) < 0

-- Part (Ⅰ)
theorem intersection_A_B_at_3 :
  let a := 3
  let A := {x : ℝ | abs x ≥ 2}
  let B := {x : ℝ | (x - 6) * (x + 3) < 0}
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (2 ≤ x ∧ x < 6)} :=
by
  sorry

-- Part (Ⅱ)
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, A x ∨ B x a) → a ≥ 1 :=
by
  sorry

end intersection_A_B_at_3_range_of_a_l237_237642


namespace max_composite_numbers_l237_237469
open Nat

theorem max_composite_numbers : 
  ∃ X : Finset Nat, 
  (∀ x ∈ X, x < 1500 ∧ ¬Prime x) ∧ 
  (∀ x y ∈ X, x ≠ y → gcd x y = 1) ∧ 
  X.card = 12 := 
sorry

end max_composite_numbers_l237_237469


namespace Albert_more_than_Joshua_l237_237334

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_than_Joshua_l237_237334


namespace ratio_speed_car_speed_bike_l237_237974

def speed_of_tractor := 575 / 23
def speed_of_bike := 2 * speed_of_tractor
def speed_of_car := 540 / 6
def ratio := speed_of_car / speed_of_bike

theorem ratio_speed_car_speed_bike : ratio = 9 / 5 := by
  sorry

end ratio_speed_car_speed_bike_l237_237974


namespace smallest_number_divisible_1_to_10_l237_237175

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237175


namespace bundles_burned_in_afternoon_l237_237555

theorem bundles_burned_in_afternoon 
  (morning_burn : ℕ)
  (start_bundles : ℕ)
  (end_bundles : ℕ)
  (h_morning_burn : morning_burn = 4)
  (h_start : start_bundles = 10)
  (h_end : end_bundles = 3)
  : (start_bundles - morning_burn - end_bundles) = 3 := 
by 
  sorry

end bundles_burned_in_afternoon_l237_237555


namespace reflection_line_sum_l237_237510

-- Prove that the sum of m and b is 10 given the reflection conditions

theorem reflection_line_sum
    (m b : ℚ)
    (H : ∀ (x y : ℚ), (2, 2) = (x, y) → (8, 6) = (2 * (5 - (3 / 2) * (2 - x)), 2 + m * (y - 2)) ∧ y = m * x + b) :
  m + b = 10 :=
sorry

end reflection_line_sum_l237_237510


namespace total_time_to_climb_seven_flights_l237_237409

-- Define the conditions
def first_flight_time : ℕ := 15
def difference_between_flights : ℕ := 10
def num_of_flights : ℕ := 7

-- Define the sum of an arithmetic series function
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the theorem
theorem total_time_to_climb_seven_flights :
  arithmetic_series_sum first_flight_time difference_between_flights num_of_flights = 315 :=
by
  sorry

end total_time_to_climb_seven_flights_l237_237409


namespace arrange_letters_l237_237613

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrange_letters : factorial 7 / (factorial 3 * factorial 2 * factorial 2) = 210 := 
by
  sorry

end arrange_letters_l237_237613


namespace value_of_x_and_z_l237_237595

theorem value_of_x_and_z (x y z : ℤ) (h1 : x / y = 7 / 3) (h2 : y = 21) (h3 : z = 3 * y) : x = 49 ∧ z = 63 :=
by
  sorry

end value_of_x_and_z_l237_237595


namespace total_fish_l237_237843

theorem total_fish (fish_Lilly fish_Rosy : ℕ) (hL : fish_Lilly = 10) (hR : fish_Rosy = 8) : fish_Lilly + fish_Rosy = 18 := 
by 
  sorry

end total_fish_l237_237843


namespace flower_combinations_l237_237188

theorem flower_combinations (t l : ℕ) (h : 4 * t + 3 * l = 60) : 
  ∃ (t_values : Finset ℕ), (∀ x ∈ t_values, 0 ≤ x ∧ x ≤ 15 ∧ x % 3 = 0) ∧
  t_values.card = 6 :=
sorry

end flower_combinations_l237_237188


namespace intersection_of_M_and_N_l237_237253

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l237_237253


namespace impossible_to_form_triangle_l237_237976

theorem impossible_to_form_triangle 
  (a b c : ℝ)
  (h1 : a = 9) 
  (h2 : b = 4) 
  (h3 : c = 3) 
  : ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  rw [h1, h2, h3]
  simp
  sorry

end impossible_to_form_triangle_l237_237976


namespace find_x_plus_y_l237_237765

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end find_x_plus_y_l237_237765


namespace commission_percentage_l237_237705

theorem commission_percentage 
  (cost_price : ℝ) (profit_percentage : ℝ) (observed_price : ℝ) (C : ℝ) 
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.10)
  (h3 : observed_price = 19.8) 
  (h4 : 1 + C / 100 = 19.8 / (cost_price * (1 + profit_percentage)))
  : C = 20 := 
by
  sorry

end commission_percentage_l237_237705


namespace parabola_symmetric_y_axis_intersection_l237_237000

theorem parabola_symmetric_y_axis_intersection :
  ∀ (x y : ℝ),
  (x = y ∨ x*x + y*y - 6*y = 0) ∧ (x*x = 3 * y) :=
by 
  sorry

end parabola_symmetric_y_axis_intersection_l237_237000


namespace Albert_more_than_Joshua_l237_237333

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_than_Joshua_l237_237333


namespace cos_neg_60_equals_half_l237_237722

  theorem cos_neg_60_equals_half : Real.cos (-60 * Real.pi / 180) = 1 / 2 :=
  by
    sorry
  
end cos_neg_60_equals_half_l237_237722


namespace train_length_l237_237195

open Real

theorem train_length 
  (v : ℝ) -- speed of the train in km/hr
  (t : ℝ) -- time in seconds
  (d : ℝ) -- length of the bridge in meters
  (h_v : v = 36) -- condition 1
  (h_t : t = 50) -- condition 2
  (h_d : d = 140) -- condition 3
  : (v * 1000 / 3600) * t = 360 + 140 := 
sorry

end train_length_l237_237195


namespace lcm_1_to_10_l237_237104

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l237_237104


namespace darren_and_fergie_same_amount_in_days_l237_237001

theorem darren_and_fergie_same_amount_in_days : 
  ∀ (t : ℕ), (200 + 16 * t = 300 + 12 * t) → t = 25 := 
by sorry

end darren_and_fergie_same_amount_in_days_l237_237001


namespace total_money_spent_l237_237209

/-- Erika, Elizabeth, Emma, and Elsa went shopping on Wednesday.
Emma spent $58.
Erika spent $20 more than Emma.
Elsa spent twice as much as Emma.
Elizabeth spent four times as much as Elsa.
Erika received a 10% discount on what she initially spent.
Elizabeth had to pay a 6% tax on her purchases.
Prove that the total amount of money they spent is $736.04.
-/
theorem total_money_spent :
  let emma_spent := 58
  let erika_initial_spent := emma_spent + 20
  let erika_discount := 0.10 * erika_initial_spent
  let erika_final_spent := erika_initial_spent - erika_discount
  let elsa_spent := 2 * emma_spent
  let elizabeth_initial_spent := 4 * elsa_spent
  let elizabeth_tax := 0.06 * elizabeth_initial_spent
  let elizabeth_final_spent := elizabeth_initial_spent + elizabeth_tax
  let total_spent := emma_spent + erika_final_spent + elsa_spent + elizabeth_final_spent
  total_spent = 736.04 := by
  sorry

end total_money_spent_l237_237209


namespace quadratic_function_properties_l237_237846

theorem quadratic_function_properties
    (f : ℝ → ℝ)
    (h_vertex : ∀ x, f x = -(x - 2)^2 + 1)
    (h_point : f (-1) = -8) :
  (∀ x, f x = -(x - 2)^2 + 1) ∧
  (f 1 = 0) ∧ (f 3 = 0) ∧ (f 0 = 1) :=
  by
    sorry

end quadratic_function_properties_l237_237846


namespace natalia_crates_l237_237732

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l237_237732


namespace paco_manu_product_lt_40_l237_237965

noncomputable def probability_product_less_than_40 : ℚ :=
  let paco_numbers := {n // 1 ≤ n ∧ n ≤ 5}
  let manu_numbers := {n // 1 ≤ n ∧ n ≤ 15}
  let valid_pairs := (paco_numbers.product manu_numbers).filter (λ p, p.1 * p.2 < 40)
  (valid_pairs.card : ℚ) / (paco_numbers.card * manu_numbers.card)

theorem paco_manu_product_lt_40 : probability_product_less_than_40 = 59/75 :=
by
  -- The proof will be here, using the calculations and steps from the solution to prove the result.
  sorry

end paco_manu_product_lt_40_l237_237965


namespace multiple_of_age_is_3_l237_237091

def current_age : ℕ := 9
def age_six_years_ago : ℕ := 3
def age_multiple (current : ℕ) (previous : ℕ) : ℕ := current / previous

theorem multiple_of_age_is_3 : age_multiple current_age age_six_years_ago = 3 :=
by
  sorry

end multiple_of_age_is_3_l237_237091


namespace bhanu_house_rent_expenditure_l237_237695

variable (Income house_rent_expenditure petrol_expenditure remaining_income : ℝ)
variable (h1 : petrol_expenditure = (30 / 100) * Income)
variable (h2 : remaining_income = Income - petrol_expenditure)
variable (h3 : house_rent_expenditure = (20 / 100) * remaining_income)
variable (h4 : petrol_expenditure = 300)

theorem bhanu_house_rent_expenditure :
  house_rent_expenditure = 140 :=
by sorry

end bhanu_house_rent_expenditure_l237_237695


namespace boards_cannot_be_covered_by_dominos_l237_237603

-- Definitions of the boards
def board_6x4 := (6 : ℕ) * (4 : ℕ)
def board_5x5 := (5 : ℕ) * (5 : ℕ)
def board_L_shaped := (5 : ℕ) * (5 : ℕ) - (2 : ℕ) * (2 : ℕ)
def board_3x7 := (3 : ℕ) * (7 : ℕ)
def board_plus_shaped := (3 : ℕ) * (3 : ℕ) + (1 : ℕ) * (3 : ℕ)

-- Definition to check if a board can't be covered by dominoes
def cannot_be_covered_by_dominos (n : ℕ) : Prop := n % 2 = 1

-- Theorem stating which specific boards cannot be covered by dominoes
theorem boards_cannot_be_covered_by_dominos :
  cannot_be_covered_by_dominos board_5x5 ∧
  cannot_be_covered_by_dominos board_L_shaped ∧
  cannot_be_covered_by_dominos board_3x7 :=
by
  -- Proof here
  sorry

end boards_cannot_be_covered_by_dominos_l237_237603


namespace intersection_M_N_l237_237263

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237263


namespace problem_arithmetic_sequence_l237_237223

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) := a + d * (n - 1)

theorem problem_arithmetic_sequence (a d : ℝ) (h₁ : d < 0) (h₂ : (arithmetic_sequence a d 1)^2 = (arithmetic_sequence a d 9)^2):
  (arithmetic_sequence a d 5) = 0 :=
by
  -- This is where the proof would go
  sorry

end problem_arithmetic_sequence_l237_237223


namespace wood_burned_in_afternoon_l237_237547

theorem wood_burned_in_afternoon 
  (burned_morning : ℕ) 
  (start_bundles : ℕ) 
  (end_bundles : ℕ) 
  (burned_afternoon : ℕ) 
  (h1 : burned_morning = 4) 
  (h2 : start_bundles = 10) 
  (h3 : end_bundles = 3) 
  (h4 : burned_morning + burned_afternoon = start_bundles - end_bundles) :
  burned_afternoon = 3 := 
sorry

end wood_burned_in_afternoon_l237_237547


namespace calculate_total_cost_l237_237561

theorem calculate_total_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let num_sandwiches := 6
  let num_sodas := 5
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 39 := by
  sorry

end calculate_total_cost_l237_237561


namespace reciprocal_of_neg4_is_neg_one_fourth_l237_237982

theorem reciprocal_of_neg4_is_neg_one_fourth (x : ℝ) (h : x * -4 = 1) : x = -1/4 := 
by 
  sorry

end reciprocal_of_neg4_is_neg_one_fourth_l237_237982


namespace area_square_EFGH_l237_237776

theorem area_square_EFGH (AB BE : ℝ) (h : BE = 2) (h2 : AB = 10) :
  ∃ s : ℝ, (s = 8 * Real.sqrt 6 - 2) ∧ s^2 = (8 * Real.sqrt 6 - 2)^2 := by
  sorry

end area_square_EFGH_l237_237776


namespace assignment_ways_l237_237907

-- Definitions
def graduates := 5
def companies := 3

-- Statement to be proven
theorem assignment_ways :
  ∃ (ways : ℕ), ways = 150 :=
sorry

end assignment_ways_l237_237907


namespace simplified_expression_value_l237_237498

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l237_237498


namespace intersection_M_N_l237_237242

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237242


namespace lcm_1_to_10_l237_237151

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

end lcm_1_to_10_l237_237151


namespace toothpicks_grid_total_l237_237830

theorem toothpicks_grid_total (L W : ℕ) (hL : L = 60) (hW : W = 32) : 
  (L + 1) * W + (W + 1) * L = 3932 := 
by 
  sorry

end toothpicks_grid_total_l237_237830


namespace intersection_M_N_l237_237274

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237274


namespace quadratic_graph_y1_lt_y2_l237_237027

theorem quadratic_graph_y1_lt_y2 (x1 x2 : ℝ) (h1 : -x1^2 = y1) (h2 : -x2^2 = y2) (h3 : x1 * x2 > x2^2) : y1 < y2 :=
  sorry

end quadratic_graph_y1_lt_y2_l237_237027


namespace ruby_height_is_192_l237_237441

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end ruby_height_is_192_l237_237441


namespace largest_x_solution_l237_237015

noncomputable def solve_eq (x : ℝ) : Prop :=
  (15 * x^2 - 40 * x + 16) / (4 * x - 3) + 3 * x = 7 * x + 2

theorem largest_x_solution : 
  ∃ x : ℝ, solve_eq x ∧ x = -14 + Real.sqrt 218 := 
sorry

end largest_x_solution_l237_237015


namespace sets_of_bleachers_l237_237348

def totalFans : ℕ := 2436
def fansPerSet : ℕ := 812

theorem sets_of_bleachers (n : ℕ) (h : totalFans = n * fansPerSet) : n = 3 :=
by {
    sorry
}

end sets_of_bleachers_l237_237348


namespace degree_of_g_l237_237073

theorem degree_of_g (f g : Polynomial ℝ) (h : Polynomial ℝ) (H1 : h = f.comp g + g) 
  (H2 : h.natDegree = 6) (H3 : f.natDegree = 3) : g.natDegree = 2 := 
sorry

end degree_of_g_l237_237073


namespace smallest_number_div_by_1_to_10_l237_237165

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l237_237165


namespace orthographic_projection_area_l237_237226

theorem orthographic_projection_area (s : ℝ) (h : s = 1) : 
  let S := (Real.sqrt 3) / 4 
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  S' = (Real.sqrt 6) / 16 :=
by
  let S := (Real.sqrt 3) / 4
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  sorry

end orthographic_projection_area_l237_237226


namespace dollar_function_twice_l237_237576

noncomputable def f (N : ℝ) : ℝ := 0.4 * N + 2

theorem dollar_function_twice (N : ℝ) (h : N = 30) : (f ∘ f) N = 5 := 
by
  sorry

end dollar_function_twice_l237_237576


namespace opposite_meaning_for_option_C_l237_237396

def opposite_meaning (a b : Int) : Bool :=
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem opposite_meaning_for_option_C :
  (opposite_meaning 300 (-500)) ∧ 
  ¬ (opposite_meaning 5 (-5)) ∧ 
  ¬ (opposite_meaning 180 90) ∧ 
  ¬ (opposite_meaning 1 (-1)) :=
by
  unfold opposite_meaning
  sorry

end opposite_meaning_for_option_C_l237_237396


namespace solution_l237_237016

theorem solution (x : ℝ) 
  (h1 : 1/x < 3)
  (h2 : 1/x > -4) 
  (h3 : x^2 - 3*x + 2 < 0) : 
  1 < x ∧ x < 2 :=
sorry

end solution_l237_237016


namespace weeds_in_rice_l237_237329

-- Define the conditions
def total_weight_of_rice := 1536
def sample_size := 224
def weeds_in_sample := 28

-- State the main proof
theorem weeds_in_rice (total_rice : ℕ) (sample_size : ℕ) (weeds_sample : ℕ) 
  (H1 : total_rice = total_weight_of_rice) (H2 : sample_size = sample_size) (H3 : weeds_sample = weeds_in_sample) :
  total_rice * weeds_sample / sample_size = 192 := 
by
  -- Evidence of calculations and external assumptions, translated initial assumptions into mathematical format
  sorry

end weeds_in_rice_l237_237329


namespace find_original_workers_and_time_l237_237556

-- Definitions based on the identified conditions
def original_workers (x : ℕ) (y : ℕ) : Prop :=
  (x - 2) * (y + 4) = x * y ∧
  (x + 3) * (y - 2) > x * y ∧
  (x + 4) * (y - 3) > x * y

-- Problem statement to prove
theorem find_original_workers_and_time (x y : ℕ) :
  original_workers x y → x = 6 ∧ y = 8 :=
by
  sorry

end find_original_workers_and_time_l237_237556


namespace seating_arrangements_l237_237894

/-- 
Given seven seats in a row, with four people sitting such that exactly two adjacent seats are empty,
prove that the number of different seating arrangements is 480.
-/
theorem seating_arrangements (seats people : ℕ) (adj_empty : ℕ) : 
  seats = 7 → people = 4 → adj_empty = 2 → 
  (∃ count : ℕ, count = 480) :=
by
  sorry

end seating_arrangements_l237_237894


namespace M_inter_N_eq_2_4_l237_237238

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l237_237238


namespace dot_product_bounds_l237_237430

theorem dot_product_bounds
  (A : ℝ × ℝ)
  (hA : A.1 ^ 2 + (A.2 - 1) ^ 2 = 1) :
  -2 ≤ A.1 * 2 ∧ A.1 * 2 ≤ 2 := 
sorry

end dot_product_bounds_l237_237430


namespace smallest_divisible_1_to_10_l237_237115

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l237_237115


namespace cost_of_camel_l237_237699

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

end cost_of_camel_l237_237699


namespace fill_cistern_l237_237696

theorem fill_cistern (p_rate q_rate : ℝ) (total_time first_pipe_time : ℝ) (remaining_fraction : ℝ): 
  p_rate = 1/12 → q_rate = 1/15 → total_time = 2 → remaining_fraction = 7/10 → 
  (remaining_fraction / q_rate) = 10.5 :=
by
  sorry

end fill_cistern_l237_237696


namespace gcd_84_210_l237_237014

theorem gcd_84_210 : Nat.gcd 84 210 = 42 :=
by {
  sorry
}

end gcd_84_210_l237_237014


namespace citizen_income_l237_237411

theorem citizen_income (total_tax : ℝ) (income : ℝ) :
  total_tax = 15000 →
  (income ≤ 20000 → total_tax = income * 0.10) ∧
  (20000 < income ∧ income ≤ 50000 → total_tax = (20000 * 0.10) + ((income - 20000) * 0.15)) ∧
  (50000 < income ∧ income ≤ 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + ((income - 50000) * 0.20)) ∧
  (income > 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + (40000 * 0.20) + ((income - 90000) * 0.25)) →
  income = 92000 :=
by
  sorry

end citizen_income_l237_237411


namespace derivatives_at_zero_l237_237585

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem derivatives_at_zero :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv (deriv f) 0 = -4 ∧
  deriv (deriv (deriv f)) 0 = 0 ∧
  deriv (deriv (deriv (deriv f))) 0 = 16 :=
by
  sorry

end derivatives_at_zero_l237_237585


namespace smallest_divisor_of_2880_that_results_in_perfect_square_l237_237692

theorem smallest_divisor_of_2880_that_results_in_perfect_square : 
  ∃ (n : ℕ), (n ∣ 2880) ∧ (∃ m : ℕ, 2880 / n = m * m) ∧ (∀ k : ℕ, (k ∣ 2880) ∧ (∃ m' : ℕ, 2880 / k = m' * m') → n ≤ k) ∧ n = 10 :=
sorry

end smallest_divisor_of_2880_that_results_in_perfect_square_l237_237692


namespace intersection_eq_l237_237250

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l237_237250


namespace intersection_M_N_l237_237244

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237244


namespace inequality_on_abc_l237_237986

variable (a b c : ℝ)

theorem inequality_on_abc (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) :=
by
  sorry

end inequality_on_abc_l237_237986


namespace find_functions_l237_237011

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem find_functions (f : ℤ → ℤ) (h : satisfies_equation f) : (∀ x, f x = 2 * x) ∨ (∀ x, f x = 0) :=
sorry

end find_functions_l237_237011


namespace remaining_lives_l237_237090

theorem remaining_lives (initial_players quit1 quit2 player_lives : ℕ) (h1 : initial_players = 15) (h2 : quit1 = 5) (h3 : quit2 = 4) (h4 : player_lives = 7) :
  (initial_players - quit1 - quit2) * player_lives = 42 :=
by
  sorry

end remaining_lives_l237_237090


namespace parabola_vertex_l237_237363

theorem parabola_vertex (c d : ℝ) (h : ∀ (x : ℝ), (-x^2 + c * x + d ≤ 0) ↔ (x ≤ -5 ∨ x ≥ 3)) :
  (∃ a b : ℝ, a = 4 ∧ b = 1 ∧ (-x^2 + c * x + d = -x^2 + 8 * x - 15)) :=
by
  sorry

end parabola_vertex_l237_237363


namespace driver_net_rate_of_pay_is_30_33_l237_237542

noncomputable def driver_net_rate_of_pay : ℝ :=
  let hours := 3
  let speed_mph := 65
  let miles_per_gallon := 30
  let pay_per_mile := 0.55
  let cost_per_gallon := 2.50
  let total_distance := speed_mph * hours
  let gallons_used := total_distance / miles_per_gallon
  let gross_earnings := total_distance * pay_per_mile
  let fuel_cost := gallons_used * cost_per_gallon
  let net_earnings := gross_earnings - fuel_cost
  let net_rate_per_hour := net_earnings / hours
  net_rate_per_hour

theorem driver_net_rate_of_pay_is_30_33 :
  driver_net_rate_of_pay = 30.33 :=
by
  sorry

end driver_net_rate_of_pay_is_30_33_l237_237542


namespace area_increase_by_16_percent_l237_237361

theorem area_increase_by_16_percent (L B : ℝ) :
  ((1.45 * L) * (0.80 * B)) / (L * B) = 1.16 :=
by
  sorry

end area_increase_by_16_percent_l237_237361


namespace least_odd_prime_factor_of_2023_8_plus_1_l237_237420

-- Define the example integers and an assumption for modular arithmetic
def n : ℕ := 2023
def p : ℕ := 97

-- Conditions and the theorem statement
theorem least_odd_prime_factor_of_2023_8_plus_1 :
  n ^ 8 ≡ -1 [MOD p] →
  ∀ q, prime q → q ∣ (n ^ 8 + 1) → q ≥ p :=
by
  sorry

end least_odd_prime_factor_of_2023_8_plus_1_l237_237420


namespace find_constants_C_D_l237_237220

theorem find_constants_C_D
  (C : ℚ) (D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → (5 * x - 3) / (x^2 - 5 * x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32 / 9 ∧ D = 13 / 9 :=
by
  sorry

end find_constants_C_D_l237_237220


namespace smallest_integer_among_three_l237_237524

theorem smallest_integer_among_three 
  (x y z : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hxy : y - x ≤ 6)
  (hxz : z - x ≤ 6) 
  (hprod : x * y * z = 2808) : 
  x = 12 := 
sorry

end smallest_integer_among_three_l237_237524


namespace intersection_M_N_l237_237282

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l237_237282


namespace intersection_M_N_l237_237294

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l237_237294


namespace probability_odd_sum_of_6_balls_drawn_l237_237540

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_odd_sum_of_6_balls_drawn :
  let n := 11
  let k := 6
  let total_ways := binom n k
  let odd_count := 6
  let even_count := 5
  let cases := 
    (binom odd_count 1 * binom even_count (k - 1)) +
    (binom odd_count 3 * binom even_count (k - 3)) +
    (binom odd_count 5 * binom even_count (k - 5))
  let favorable_outcomes := cases
  let probability := favorable_outcomes / total_ways
  probability = 118 / 231 := 
by {
  sorry
}

end probability_odd_sum_of_6_balls_drawn_l237_237540


namespace find_value_of_expression_l237_237746

variable (a b c : ℝ)

def parabola_symmetry (a b c : ℝ) :=
  (36 * a + 6 * b + c = 2) ∧ 
  (25 * a + 5 * b + c = 6) ∧ 
  (49 * a + 7 * b + c = -4)

theorem find_value_of_expression :
  (∃ a b c : ℝ, parabola_symmetry a b c) →
  3 * a + 3 * c + b = -8 :=  sorry

end find_value_of_expression_l237_237746


namespace sum_xyz_zero_l237_237299

theorem sum_xyz_zero 
  (x y z : ℝ)
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : y = 6 * z) : 
  x + y + z = 0 := by
  sorry

end sum_xyz_zero_l237_237299


namespace baker_cakes_remaining_l237_237399

theorem baker_cakes_remaining (initial_cakes: ℕ) (fraction_sold: ℚ) (sold_cakes: ℕ) (cakes_remaining: ℕ) :
  initial_cakes = 149 ∧ fraction_sold = 2/5 ∧ sold_cakes = 59 ∧ cakes_remaining = initial_cakes - sold_cakes → cakes_remaining = 90 :=
by
  sorry

end baker_cakes_remaining_l237_237399


namespace least_m_plus_n_l237_237954

theorem least_m_plus_n (m n : ℕ) (h1 : Nat.gcd (m + n) 231 = 1) 
                                  (h2 : m^m ∣ n^n) 
                                  (h3 : ¬ m ∣ n)
                                  : m + n = 75 :=
sorry

end least_m_plus_n_l237_237954


namespace sum_of_roots_quadratic_eq_l237_237645

variable (h : ℝ)
def quadratic_eq_roots (x : ℝ) : Prop := 6 * x^2 - 5 * h * x - 4 * h = 0

theorem sum_of_roots_quadratic_eq (x1 x2 : ℝ) (h : ℝ) 
  (h_roots : quadratic_eq_roots h x1 ∧ quadratic_eq_roots h x2) 
  (h_distinct : x1 ≠ x2) :
  x1 + x2 = 5 * h / 6 := by
sorry

end sum_of_roots_quadratic_eq_l237_237645


namespace given_tan_alpha_eq_3_then_expression_eq_8_7_l237_237592

theorem given_tan_alpha_eq_3_then_expression_eq_8_7 (α : ℝ) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := 
by
  sorry

end given_tan_alpha_eq_3_then_expression_eq_8_7_l237_237592


namespace smallest_number_divisible_by_1_to_10_l237_237124

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237124


namespace smallest_number_divisible_by_1_through_10_l237_237155

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l237_237155


namespace smallest_number_divisible_by_1_to_10_l237_237129

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237129


namespace Craig_bench_press_percentage_l237_237002

theorem Craig_bench_press_percentage {Dave_weight : ℕ} (h1 : Dave_weight = 175) (h2 : ∀ w : ℕ, Dave_bench_press = 3 * Dave_weight) 
(Craig_bench_press Mark_bench_press : ℕ) (h3 : Mark_bench_press = 55) (h4 : Mark_bench_press = Craig_bench_press - 50) : 
(Craig_bench_press / (3 * Dave_weight) * 100) = 20 := by
  sorry

end Craig_bench_press_percentage_l237_237002


namespace precision_tens_place_l237_237735

-- Given
def given_number : ℝ := 4.028 * (10 ^ 5)

-- Prove that the precision of the given_number is to the tens place.
theorem precision_tens_place : true := by
  -- Proof goes here
  sorry

end precision_tens_place_l237_237735


namespace part_a_part_b_l237_237686
open Set

def fantastic (n : ℕ) : Prop :=
  ∃ a b : ℚ, a > 0 ∧ b > 0 ∧ n = a + 1 / a + b + 1 / b

theorem part_a : ∃ᶠ p in at_top, Prime p ∧ ∀ k, ¬ fantastic (k * p) := 
  sorry

theorem part_b : ∃ᶠ p in at_top, Prime p ∧ ∃ k, fantastic (k * p) :=
  sorry

end part_a_part_b_l237_237686


namespace tan_ratio_l237_237058

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end tan_ratio_l237_237058


namespace partial_fraction_decomposition_l237_237736

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 - 24 * Polynomial.X^2 + 143 * Polynomial.X - 210

theorem partial_fraction_decomposition (A B C p q r : ℝ) (h1 : Polynomial.roots polynomial = {p, q, r}) 
  (h2 : ∀ s : ℝ, 1 / (s^3 - 24 * s^2 + 143 * s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 243 :=
by
  sorry

end partial_fraction_decomposition_l237_237736


namespace value_of_m_l237_237009

theorem value_of_m : (∀ x : ℝ, (1 + 2 * x) ^ 3 = 1 + 6 * x + m * x ^ 2 + 8 * x ^ 3 → m = 12) := 
by {
  -- This is where the proof would go
  sorry
}

end value_of_m_l237_237009


namespace probability_of_four_twos_in_five_rolls_l237_237619

theorem probability_of_four_twos_in_five_rolls :
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  total_probability = 3125 / 7776 :=
by
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  show total_probability = 3125 / 7776
  sorry

end probability_of_four_twos_in_five_rolls_l237_237619


namespace solve_for_x_l237_237665

theorem solve_for_x (x : ℚ) : (x = 70 / (8 - 3 / 4)) → (x = 280 / 29) :=
by
  intro h
  -- Proof to be provided here
  sorry

end solve_for_x_l237_237665


namespace probability_roll_2_four_times_in_five_rolls_l237_237627

theorem probability_roll_2_four_times_in_five_rolls :
  (∃ (prob_roll_2 : ℚ) (prob_not_roll_2 : ℚ), 
   prob_roll_2 = 1/6 ∧ prob_not_roll_2 = 5/6 ∧ 
   (5 * prob_roll_2^4 * prob_not_roll_2 = 5/72)) :=
sorry

end probability_roll_2_four_times_in_five_rolls_l237_237627


namespace find_two_numbers_l237_237372

noncomputable def quadratic_roots (a b : ℝ) : Prop :=
  a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2

theorem find_two_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : 2 * (a * b) / (a + b) = 5 / 2) :
  quadratic_roots a b :=
by
  sorry

end find_two_numbers_l237_237372


namespace intersection_of_prime_and_even_is_two_l237_237788

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 2 * k

theorem intersection_of_prime_and_even_is_two :
  {n : ℕ | is_prime n} ∩ {n : ℕ | is_even n} = {2} :=
by
  sorry

end intersection_of_prime_and_even_is_two_l237_237788


namespace vector_division_by_three_l237_237317

def OA : ℝ × ℝ := (2, 8)
def OB : ℝ × ℝ := (-7, 2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def scalar_mult (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)

theorem vector_division_by_three :
  scalar_mult (1 / 3) (vector_sub OB OA) = (-3, -2) :=
sorry

end vector_division_by_three_l237_237317


namespace kevin_total_cost_l237_237948

theorem kevin_total_cost :
  let muffin_cost := 0.75
  let juice_cost := 1.45
  let total_muffins := 3
  let cost_muffins := total_muffins * muffin_cost
  let total_cost := cost_muffins + juice_cost
  total_cost = 3.70 :=
by
  sorry

end kevin_total_cost_l237_237948


namespace factor_determines_d_l237_237767

theorem factor_determines_d (d : ℚ) :
  (∀ x : ℚ, x - 4 ∣ d * x^3 - 8 * x^2 + 5 * d * x - 12) → d = 5 / 3 := by
  sorry

end factor_determines_d_l237_237767


namespace solid_is_cone_l237_237038

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

end solid_is_cone_l237_237038


namespace smallest_number_divisible_1_to_10_l237_237122

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237122


namespace cost_to_fill_sandbox_l237_237332

-- Definitions for conditions
def side_length : ℝ := 3
def volume_per_bag : ℝ := 3
def cost_per_bag : ℝ := 4

-- Theorem statement
theorem cost_to_fill_sandbox : (side_length ^ 3 / volume_per_bag * cost_per_bag) = 36 := by
  sorry

end cost_to_fill_sandbox_l237_237332


namespace bob_has_17_pennies_l237_237319

-- Definitions based on the problem conditions
variable (a b : ℕ)
def condition1 : Prop := b + 1 = 4 * (a - 1)
def condition2 : Prop := b - 2 = 2 * (a + 2)

-- The main statement to be proven
theorem bob_has_17_pennies (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 17 :=
by
  sorry

end bob_has_17_pennies_l237_237319


namespace negative_square_inequality_l237_237222

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l237_237222


namespace height_of_windows_l237_237821

theorem height_of_windows
  (L W H d_l d_w w_w : ℕ)
  (C T : ℕ)
  (hl : L = 25)
  (hw : W = 15)
  (hh : H = 12)
  (hdl : d_l = 6)
  (hdw : d_w = 3)
  (hww : w_w = 3)
  (hc : C = 3)
  (ht : T = 2718):
  ∃ h : ℕ, 960 - (18 + 9 * h) = 906 ∧ 
  (T = C * (960 - (18 + 9 * h))) ∧
  (960 = 2 * (L * H) + 2 * (W * H)) ∧ 
  (18 = d_l * d_w) ∧ 
  (9 * h = 3 * (h * w_w)) := 
sorry

end height_of_windows_l237_237821


namespace abs_neg_six_l237_237972

theorem abs_neg_six : |(-6)| = 6 := by
  sorry

end abs_neg_six_l237_237972


namespace find_sum_of_digits_in_base_l237_237506

theorem find_sum_of_digits_in_base (d A B : ℕ) (hd : d > 8) (hA : A < d) (hB : B < d) (h : (A * d + B) + (A * d + A) - (B * d + A) = 1 * d^2 + 8 * d + 0) : A + B = 10 :=
sorry

end find_sum_of_digits_in_base_l237_237506


namespace reflection_correct_l237_237526

/-- Definition of reflection across the line y = -x -/
def reflection_across_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- Given points C and D, and their images C' and D' respectively, under reflection,
    prove the transformation is correct. -/
theorem reflection_correct :
  (reflection_across_y_eq_neg_x (-3, 2) = (3, -2)) ∧ (reflection_across_y_eq_neg_x (-2, 5) = (2, -5)) :=
  by
    sorry

end reflection_correct_l237_237526


namespace intersection_eq_l237_237248

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l237_237248


namespace range_of_a_l237_237741

variable (a : ℝ)

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) : 1 < a :=
by {
  -- Proof will go here.
  sorry
}

end range_of_a_l237_237741


namespace sum_arithmetic_sequence_l237_237324

-- Define the arithmetic sequence condition and sum of given terms
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n d : ℕ, a n = a 1 + (n - 1) * d

def given_sum_condition (a : ℕ → ℕ) : Prop :=
  a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_arithmetic_sequence (a : ℕ → ℕ) (h_arith_seq : arithmetic_sequence a) 
  (h_sum_cond : given_sum_condition a) : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry  -- Proof of the theorem

end sum_arithmetic_sequence_l237_237324


namespace prism_faces_l237_237871

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end prism_faces_l237_237871


namespace largest_angle_in_right_isosceles_triangle_l237_237682

theorem largest_angle_in_right_isosceles_triangle (X Y Z : Type) 
  (angle_X : ℝ) (angle_Y : ℝ) (angle_Z : ℝ) 
  (h1 : angle_X = 45) 
  (h2 : angle_Y = 90)
  (h3 : angle_Y + angle_X + angle_Z = 180) 
  (h4 : angle_X = angle_Z) : angle_Y = 90 := by 
  sorry

end largest_angle_in_right_isosceles_triangle_l237_237682


namespace quotient_remainder_l237_237218

theorem quotient_remainder (x y : ℕ) (hx : 0 ≤ x) (hy : 0 < y) : 
  ∃ q r : ℕ, q ≥ 0 ∧ 0 ≤ r ∧ r < y ∧ x = q * y + r := by
  sorry

end quotient_remainder_l237_237218


namespace abc_prod_eq_l237_237065

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

end abc_prod_eq_l237_237065


namespace mia_study_time_l237_237799

theorem mia_study_time 
  (T : ℕ)
  (watching_tv_exercise_social_media : T = 1440 ∧ 
    ∃ study_time : ℚ, 
    (study_time = (1 / 4) * 
      (((27 / 40) * T - (9 / 80) * T) / 
        (T * 1 / 40 - (1 / 5) * T - (1 / 8) * T))
    )) :
  T = 1440 → study_time = 202.5 := 
by
  sorry

end mia_study_time_l237_237799


namespace max_composite_numbers_with_gcd_one_l237_237470

theorem max_composite_numbers_with_gcd_one : 
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, Nat.isComposite x) ∧ 
    (∀ x ∈ S, x < 1500) ∧ 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → Nat.gcd x y = 1) ∧
    S.card = 12 :=
sorry

end max_composite_numbers_with_gcd_one_l237_237470


namespace polynomial_root_exists_l237_237020

theorem polynomial_root_exists
  (P : ℝ → ℝ)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
  (h_eq : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)) :
  ∃ r : ℝ, P r = 0 :=
sorry

end polynomial_root_exists_l237_237020


namespace smallest_number_divisible_by_1_through_10_l237_237156

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l237_237156


namespace prism_faces_l237_237862

theorem prism_faces (edges : ℕ) (h_edges : edges = 18) : ∃ faces : ℕ, faces = 8 :=
by
  -- Define L as the number of lateral faces
  let L := edges / 3
  have h_L : L = 6, from calc
    L = 18 / 3 : by rw [h_edges]
    ... = 6 : by norm_num
  -- Define the total number of faces
  let total_faces := L + 2
  have h_faces : total_faces = 8, from calc
    total_faces = 6 + 2 : by rw [h_L]
    ... = 8 : by norm_num
  use total_faces
  exact h_faces

end prism_faces_l237_237862


namespace wood_burned_in_afternoon_l237_237549

theorem wood_burned_in_afternoon 
  (burned_morning : ℕ) 
  (start_bundles : ℕ) 
  (end_bundles : ℕ) 
  (burned_afternoon : ℕ) 
  (h1 : burned_morning = 4) 
  (h2 : start_bundles = 10) 
  (h3 : end_bundles = 3) 
  (h4 : burned_morning + burned_afternoon = start_bundles - end_bundles) :
  burned_afternoon = 3 := 
sorry

end wood_burned_in_afternoon_l237_237549


namespace pumpkins_at_other_orchard_l237_237398

-- Defining the initial conditions
def sunshine_pumpkins : ℕ := 54
def other_orchard_pumpkins : ℕ := 14

-- Equation provided in the problem
def condition_equation (P : ℕ) : Prop := 54 = 3 * P + 12

-- Proving the main statement using the conditions
theorem pumpkins_at_other_orchard : condition_equation other_orchard_pumpkins :=
by
  unfold condition_equation
  sorry -- To be completed with the proof

end pumpkins_at_other_orchard_l237_237398


namespace lcm_1_to_10_l237_237171

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l237_237171


namespace find_a_l237_237357

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom cond1 : a^2 / b = 5
axiom cond2 : b^2 / c = 3
axiom cond3 : c^2 / a = 7

theorem find_a : a = 15 := sorry

end find_a_l237_237357


namespace find_integer_x_l237_237381

theorem find_integer_x (x : ℕ) (pos_x : 0 < x) (ineq : x + 1000 > 1000 * x) : x = 1 :=
sorry

end find_integer_x_l237_237381


namespace parabola_opens_upward_l237_237942

structure QuadraticFunction :=
  (a b c : ℝ)

def quadratic_y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

def points : List (ℝ × ℝ) :=
  [(-1, 10), (0, 5), (1, 2), (2, 1), (3, 2)]

theorem parabola_opens_upward (f : QuadraticFunction)
  (h_values : ∀ (x : ℝ), (x, quadratic_y f x) ∈ points) :
  f.a > 0 :=
sorry

end parabola_opens_upward_l237_237942


namespace serving_guests_possible_iff_even_l237_237417

theorem serving_guests_possible_iff_even (n : ℕ) : 
  (∀ seats : Finset ℕ, ∀ p : ℕ → ℕ, (∀ i : ℕ, i < n → p i ∈ seats) → 
    (∀ i j : ℕ, i < j → p i ≠ p j) → (n % 2 = 0)) = (n % 2 = 0) :=
by sorry

end serving_guests_possible_iff_even_l237_237417


namespace sequence_property_l237_237031

theorem sequence_property (a : ℕ → ℕ) (h1 : ∀ n, n ≥ 1 → a n ∈ { x | x ≥ 1 }) 
  (h2 : ∀ n, n ≥ 1 → a (a n) + a n = 2 * n) : ∀ n, n ≥ 1 → a n = n :=
by
  sorry

end sequence_property_l237_237031


namespace alice_bob_numbers_sum_l237_237583

-- Fifty slips of paper numbered 1 to 50 are placed in a hat.
-- Alice and Bob each draw one number from the hat without replacement, keeping their numbers hidden from each other.
-- Alice cannot tell who has the larger number.
-- Bob knows who has the larger number.
-- Bob's number is composite.
-- If Bob's number is multiplied by 50 and Alice's number is added, the result is a perfect square.
-- Prove that the sum of Alice's and Bob's numbers is 29.

theorem alice_bob_numbers_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 50) (hB : 1 ≤ B ∧ B ≤ 50) 
  (hAB_distinct : A ≠ B) (hA_unknown : ¬(A = 1 ∨ A = 50))
  (hB_composite : ∃ d > 1, d < B ∧ B % d = 0) (h_perfect_square : ∃ k, 50 * B + A = k ^ 2) :
  A + B = 29 := by
  sorry

end alice_bob_numbers_sum_l237_237583


namespace smallest_number_divisible_by_1_to_10_l237_237098

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237098


namespace smallest_divisible_1_to_10_l237_237117

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l237_237117


namespace circle_area_difference_l237_237928

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 30) (h2 : r2 = 7.5) : 
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end circle_area_difference_l237_237928


namespace total_cost_price_proof_l237_237849

variable (C O B : ℝ)
variable (paid_computer_table paid_office_chair paid_bookshelf : ℝ)
variable (markup_computer_table markup_office_chair markup_bookshelf : ℝ)

noncomputable def total_cost_price {paid_computer_table paid_office_chair paid_bookshelf : ℝ} 
                                    {markup_computer_table markup_office_chair markup_bookshelf : ℝ}
                                    (C O B : ℝ) : ℝ :=
  C + O + B

theorem total_cost_price_proof 
  (h1 : paid_computer_table = C + markup_computer_table * C)
  (h2 : paid_office_chair = O + markup_office_chair * O)
  (h3 : paid_bookshelf = B + markup_bookshelf * B)
  (h_paid_computer_table : paid_computer_table = 8340)
  (h_paid_office_chair : paid_office_chair = 4675)
  (h_paid_bookshelf : paid_bookshelf = 3600)
  (h_markup_computer_table : markup_computer_table = 0.25)
  (h_markup_office_chair : markup_office_chair = 0.30)
  (h_markup_bookshelf : markup_bookshelf = 0.20) :
  total_cost_price (C) (O) (B) = 13268.15 := 
by
  sorry

end total_cost_price_proof_l237_237849


namespace positive_divisors_of_5400_multiple_of_5_l237_237308

-- Declare the necessary variables and conditions
theorem positive_divisors_of_5400_multiple_of_5 :
  let n := 5400
  let factorization := [(2, 2), (3, 3), (5, 2)]
  ∀ (a b c: ℕ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2 →
    (a*b*c).count(n) = 24 := 
sorry

end positive_divisors_of_5400_multiple_of_5_l237_237308


namespace other_root_of_quadratic_l237_237017

theorem other_root_of_quadratic (m : ℝ) (h : (m + 2) * 0^2 - 0 + m^2 - 4 = 0) : 
  ∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x ≠ 0 ∧ x = 1/4 := 
sorry

end other_root_of_quadratic_l237_237017


namespace double_luckiness_l237_237221

variable (oats marshmallows : ℕ)
variable (initial_luckiness doubled_luckiness : ℚ)

def luckiness (marshmallows total_pieces : ℕ) : ℚ :=
  marshmallows / total_pieces

theorem double_luckiness (h_oats : oats = 90) (h_marshmallows : marshmallows = 9)
  (h_initial : initial_luckiness = luckiness marshmallows (oats + marshmallows))
  (h_doubled : doubled_luckiness = 2 * initial_luckiness) :
  ∃ x : ℕ, doubled_luckiness = luckiness (marshmallows + x) (oats + marshmallows + x) :=
  sorry

#check double_luckiness

end double_luckiness_l237_237221


namespace prob_same_gender_eq_two_fifths_l237_237702

-- Define the number of male and female students
def num_male_students : ℕ := 3
def num_female_students : ℕ := 2

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability calculation
def probability_same_gender := (num_male_students * (num_male_students - 1) / 2 + num_female_students * (num_female_students - 1) / 2) / (total_students * (total_students - 1) / 2)

theorem prob_same_gender_eq_two_fifths :
  probability_same_gender = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end prob_same_gender_eq_two_fifths_l237_237702


namespace ruby_height_l237_237440

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end ruby_height_l237_237440


namespace cuboid_height_l237_237448

-- Define the necessary constants
def width : ℕ := 30
def length : ℕ := 22
def sum_edges : ℕ := 224

-- Theorem stating the height of the cuboid
theorem cuboid_height (h : ℕ) : 4 * length + 4 * width + 4 * h = sum_edges → h = 4 := by
  sorry

end cuboid_height_l237_237448


namespace prism_faces_l237_237870

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end prism_faces_l237_237870


namespace smallest_divisible_by_1_to_10_l237_237142

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l237_237142


namespace min_value_f_l237_237920

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * Real.arcsin x + 3

theorem min_value_f (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) (hmax : ∃ x, f a b x = 10) : ∃ y, f a b y = -4 := by
  sorry

end min_value_f_l237_237920


namespace smallest_number_divisible_1_to_10_l237_237119

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237119


namespace simplify_and_evaluate_l237_237492

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l237_237492


namespace intersection_M_N_l237_237291

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237291


namespace func_passes_through_1_2_l237_237822

-- Given conditions
variable (a : ℝ) (x : ℝ) (y : ℝ)
variable (h1 : 0 < a) (h2 : a ≠ 1)

-- Definition of the function
noncomputable def func (x : ℝ) : ℝ := a^(x-1) + 1

-- Proof statement
theorem func_passes_through_1_2 : func a 1 = 2 :=
by
  -- proof goes here
  sorry

end func_passes_through_1_2_l237_237822


namespace prism_faces_l237_237868

-- Define variables for number of edges E, lateral faces L, and total number of faces F
variables (E : ℕ := 18) (L : ℕ) (F : ℕ)

-- The relationship between edges and lateral faces for a prism
lemma prism_lateral_faces (hE : E = 18) : 3 * L = E :=
sorry

-- Calculate the number of lateral faces
lemma solve_lateral_faces (hE : E = 18) : L = 6 :=
begin
  have h : 3 * L = E := prism_lateral_faces hE,
  rw hE at h,
  exact (nat.mul_left_inj 3 (nat.zero_lt_succ 2)).mp h,
end

-- Prove the total number of faces
theorem prism_faces (hE : E = 18) : F = 8 :=
begin
  have hL : L = 6 := solve_lateral_faces hE,
  exact calc
    F = L + 2 : by rw [hL]
       ... = 6 + 2 : rfl
       ... = 8 : rfl,
end

end prism_faces_l237_237868


namespace max_volume_prism_l237_237328

theorem max_volume_prism (a b h : ℝ) (V : ℝ) 
  (h1 : a * h + b * h + a * b = 32) : 
  V = a * b * h → V ≤ 128 * Real.sqrt 3 / 3 := 
by
  sorry

end max_volume_prism_l237_237328


namespace crates_needed_l237_237730

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end crates_needed_l237_237730


namespace lcm_1_to_10_l237_237105

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l237_237105


namespace negation_statement_l237_237082

variable {α : Type} 
variable (student prepared : α → Prop)

theorem negation_statement :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by 
  -- proof will be provided here
  sorry

end negation_statement_l237_237082


namespace matrix_non_invertible_value_l237_237918

-- Define matrix and condition for non-invertibility
variables {R : Type*} [Field R]

def cyclicMatrix (a b c d : R) : Matrix (Fin 4) (Fin 4) R :=
  ![
    ![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]
  ]

-- The statement:
theorem matrix_non_invertible_value (a b c d : R) :
  deter (cyclicMatrix a b c d) = 0 ↔
  (a = d ∨ b = c ∨ c = a ∨ d = b) ∧
  (b + c + d ≠ 0 ∧ a + c + d ≠ 0 ∧ a + b + d ≠ 0 ∧ a + b + c ≠ 0) →
  (∑ x in [a/(b + c + d), b/(a + c + d), c/(a + b + d), d/(a + b + c)], x) = 4 / 3 :=
sorry

end matrix_non_invertible_value_l237_237918


namespace intersection_M_N_l237_237275

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237275


namespace trapezium_height_l237_237013

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

end trapezium_height_l237_237013


namespace five_digit_number_probability_l237_237443

-- Define a predicate for a five-digit number
def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (λ x acc => x + acc) 0

-- Define the alternating sum of digits function
def alternating_sum_of_digits (n : ℕ) : ℤ :=
  let digits := n.digits 10
  digits.enum.foldr (λ ⟨i, x⟩ acc => if i % 2 = 0 then acc + x else acc - x) 0

-- The divisible by 11 rule
def divisible_by_11 (n : ℕ) : Prop :=
  alternating_sum_of_digits n % 11 = 0

-- Prove the main statement
theorem five_digit_number_probability :
  let S := { n : ℕ | is_five_digit_number n ∧ sum_of_digits n = 43 }
  let D := { n ∈ S | divisible_by_11 n }
  (S.finite.toFinset.card : ℚ) ≠ 0 →
  (D.finite.toFinset.card : ℚ) / (S.finite.toFinset.card : ℚ) = 1 / 5 :=
by
  sorry

end five_digit_number_probability_l237_237443


namespace smallest_number_divisible_1_to_10_l237_237120

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237120


namespace ruby_height_l237_237439

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end ruby_height_l237_237439


namespace product_of_fractions_l237_237564

theorem product_of_fractions : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end product_of_fractions_l237_237564


namespace angle_Z_is_90_l237_237936

theorem angle_Z_is_90 (X Y Z : ℝ) (h_sum_XY : X + Y = 90) (h_Y_is_2X : Y = 2 * X) (h_sum_angles : X + Y + Z = 180) : Z = 90 :=
by
  sorry

end angle_Z_is_90_l237_237936


namespace find_numbers_l237_237178

/-- Given the sums of three pairs of numbers, we prove the individual numbers. -/
theorem find_numbers (x y z : ℕ) (h1 : x + y = 40) (h2 : y + z = 50) (h3 : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 :=
by
  sorry

end find_numbers_l237_237178


namespace smallest_number_div_by_1_to_10_l237_237162

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l237_237162


namespace arrange_rose_bushes_l237_237051

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

end arrange_rose_bushes_l237_237051


namespace sara_quarters_l237_237810

-- Conditions
def usd_to_eur (usd : ℝ) : ℝ := usd * 0.85
def eur_to_usd (eur : ℝ) : ℝ := eur * 1.15
def value_of_quarter_usd : ℝ := 0.25
def dozen : ℕ := 12

-- Theorem
theorem sara_quarters (sara_savings_usd : ℝ) (usd_to_eur_ratio : ℝ) (eur_to_usd_ratio : ℝ) (quarter_value_usd : ℝ) (doz : ℕ) : sara_savings_usd = 9 → usd_to_eur_ratio = 0.85 → eur_to_usd_ratio = 1.15 → quarter_value_usd = 0.25 → doz = 12 → 
  ∃ dozens : ℕ, dozens = 2 :=
by
  sorry

end sara_quarters_l237_237810


namespace remainder_when_x_plus_3uy_div_y_l237_237769

theorem remainder_when_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (v_lt_y : v < y) :
  ((x + 3 * u * y) % y) = v := 
sorry

end remainder_when_x_plus_3uy_div_y_l237_237769


namespace ellipse_hyperbola_tangent_m_eq_l237_237979

variable (x y m : ℝ)

def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 2)^2 = 1
def curves_tangent (x m : ℝ) : Prop := ∃ y, ellipse x y ∧ hyperbola x y m

theorem ellipse_hyperbola_tangent_m_eq :
  (∃ x, curves_tangent x (12/13)) ↔ true := 
by
  sorry

end ellipse_hyperbola_tangent_m_eq_l237_237979


namespace alex_cell_phone_cost_l237_237701

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.1
def extra_min_cost_per_minute : ℝ := 0.15
def text_messages_sent : ℕ := 150
def hours_talked : ℝ := 32
def included_hours : ℝ := 25

theorem alex_cell_phone_cost : base_cost 
  + (text_messages_sent * text_cost_per_message)
  + ((hours_talked - included_hours) * 60 * extra_min_cost_per_minute) = 98 := by
  sorry

end alex_cell_phone_cost_l237_237701


namespace tangent_line_f_at_one_l237_237224

noncomputable def g : ℝ → ℝ := sorry
noncomputable def f (x : ℝ) : ℝ := g (2 * x - 1) + x^2

axiom tangent_g_at_one : ∀ x, g 1 = 3 ∧ (deriv g 1 = 2)

theorem tangent_line_f_at_one : 
  let t : ℝ × ℝ := (1, f 1)
  in ∀ x y : ℝ, y - t.2 = (deriv f 1) * (x - t.1) ↔ 6 * x - y - 2 = 0 :=
begin
  sorry
end

end tangent_line_f_at_one_l237_237224


namespace prism_faces_l237_237855

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end prism_faces_l237_237855


namespace simplify_expression_l237_237484

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l237_237484


namespace triangle_isosceles_or_right_angled_l237_237482

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ) ∨ (β + γ = π / 2) :=
sorry

end triangle_isosceles_or_right_angled_l237_237482


namespace transylvanian_convinces_l237_237046

theorem transylvanian_convinces (s : Prop) (t : Prop) (h : s ↔ (¬t ∧ ¬s)) : t :=
by
  -- Leverage the existing equivalence to prove the desired result
  sorry

end transylvanian_convinces_l237_237046


namespace find_x_l237_237593

theorem find_x (x y : ℕ) (h1 : x / y = 6 / 3) (h2 : y = 27) : x = 54 :=
sorry

end find_x_l237_237593


namespace BC_at_least_17_l237_237371

-- Given conditions
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
-- Distances given
variables (AB AC EC BD BC : ℝ)
variables (AB_pos : AB = 7)
variables (AC_pos : AC = 15)
variables (EC_pos : EC = 9)
variables (BD_pos : BD = 26)
-- Triangle Inequalities
variables (triangle_ABC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], AC - AB < BC)
variables (triangle_DEC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], BD - EC < BC)

-- Proof statement
theorem BC_at_least_17 : BC ≥ 17 := by
  sorry

end BC_at_least_17_l237_237371


namespace wheel_speed_l237_237792

def original_circumference_in_miles := 10 / 5280
def time_factor := 3600
def new_time_factor := 3600 - (1/3)

theorem wheel_speed
  (r : ℝ) 
  (original_speed : r * time_factor = original_circumference_in_miles * 3600)
  (new_speed : (r + 5) * (time_factor - 1/10800) = original_circumference_in_miles * 3600) :
  r = 10 :=
sorry

end wheel_speed_l237_237792


namespace mr_blue_expected_rose_petals_l237_237463

def mr_blue_flower_bed_rose_petals (length_paces : ℕ) (width_paces : ℕ) (pace_length_ft : ℝ) (petals_per_sqft : ℝ) : ℝ :=
  let length_ft := length_paces * pace_length_ft
  let width_ft := width_paces * pace_length_ft
  let area_sqft := length_ft * width_ft
  area_sqft * petals_per_sqft

theorem mr_blue_expected_rose_petals :
  mr_blue_flower_bed_rose_petals 18 24 1.5 0.4 = 388.8 :=
by
  simp [mr_blue_flower_bed_rose_petals]
  norm_num

end mr_blue_expected_rose_petals_l237_237463


namespace intersection_eq_l237_237249

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l237_237249


namespace find_k_l237_237435

variables {x k : ℝ}

theorem find_k (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (h2 : k ≠ 0) : k = 8 :=
sorry

end find_k_l237_237435


namespace four_friends_total_fish_l237_237650

-- Define the number of fish each friend has based on the conditions
def micah_fish : ℕ := 7
def kenneth_fish : ℕ := 3 * micah_fish
def matthias_fish : ℕ := kenneth_fish - 15
def total_three_boys_fish : ℕ := micah_fish + kenneth_fish + matthias_fish
def gabrielle_fish : ℕ := 2 * total_three_boys_fish
def total_fish : ℕ := micah_fish + kenneth_fish + matthias_fish + gabrielle_fish

-- The proof goal
theorem four_friends_total_fish : total_fish = 102 :=
by
  -- We assume the proof steps are correct and leave the proof part as sorry
  sorry

end four_friends_total_fish_l237_237650


namespace sheena_weeks_to_complete_dresses_l237_237481

/- Sheena is sewing the bridesmaid's dresses for her sister's wedding.
There are 7 bridesmaids in the wedding.
Each bridesmaid's dress takes a different number of hours to sew due to different styles and sizes.
The hours needed to sew the bridesmaid's dresses are as follows: 15 hours, 18 hours, 20 hours, 22 hours, 24 hours, 26 hours, and 28 hours.
If Sheena sews the dresses 5 hours each week, prove that it will take her 31 weeks to complete all the dresses. -/

def bridesmaid_hours : List ℕ := [15, 18, 20, 22, 24, 26, 28]

def total_hours_needed (hours : List ℕ) : ℕ :=
  hours.sum

def weeks_needed (total_hours : ℕ) (hours_per_week : ℕ) : ℕ :=
  (total_hours + hours_per_week - 1) / hours_per_week

theorem sheena_weeks_to_complete_dresses :
  weeks_needed (total_hours_needed bridesmaid_hours) 5 = 31 := by
  sorry

end sheena_weeks_to_complete_dresses_l237_237481


namespace jon_payment_per_visit_l237_237050

theorem jon_payment_per_visit 
  (visits_per_hour : ℕ) (operating_hours_per_day : ℕ) (income_in_month : ℚ) (days_in_month : ℕ) 
  (visits_per_hour_eq : visits_per_hour = 50) 
  (operating_hours_per_day_eq : operating_hours_per_day = 24) 
  (income_in_month_eq : income_in_month = 3600) 
  (days_in_month_eq : days_in_month = 30) :
  (income_in_month / (visits_per_hour * operating_hours_per_day * days_in_month) : ℚ) = 0.10 := 
by
  sorry

end jon_payment_per_visit_l237_237050


namespace intersection_of_M_and_N_l237_237230

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l237_237230


namespace largest_integral_value_l237_237216

theorem largest_integral_value (y : ℤ) (h1 : 0 < y) (h2 : (1 : ℚ)/4 < y / 7) (h3 : y / 7 < 7 / 11) : y = 4 :=
sorry

end largest_integral_value_l237_237216


namespace independent_variable_range_l237_237331

/-- In the function y = 1 / (x - 2), the range of the independent variable x is all real numbers except 2. -/
theorem independent_variable_range (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end independent_variable_range_l237_237331


namespace stadium_length_in_feet_l237_237824

theorem stadium_length_in_feet (length_in_yards : ℕ) (conversion_factor : ℕ) (h1 : length_in_yards = 62) (h2 : conversion_factor = 3) : length_in_yards * conversion_factor = 186 :=
by
  sorry

end stadium_length_in_feet_l237_237824


namespace intersection_of_M_and_N_l237_237236

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l237_237236


namespace cone_from_sector_l237_237838

theorem cone_from_sector 
  (sector_angle : ℝ) (sector_radius : ℝ)
  (circumference : ℝ := (sector_angle / 360) * (2 * Real.pi * sector_radius))
  (base_radius : ℝ := circumference / (2 * Real.pi))
  (slant_height : ℝ := sector_radius) :
  sector_angle = 270 ∧ sector_radius = 12 → base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l237_237838


namespace find_f1_l237_237598

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = x * f (x)

theorem find_f1 (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : functional_equation f) : 
  f 1 = 0 :=
sorry

end find_f1_l237_237598


namespace hypotenuse_length_l237_237906

theorem hypotenuse_length (a b c : ℝ)
  (h_a : a = 12)
  (h_area : 54 = 1 / 2 * a * b)
  (h_py : c^2 = a^2 + b^2) :
    c = 15 := by
  sorry

end hypotenuse_length_l237_237906


namespace calculate_decimal_sum_and_difference_l237_237407

theorem calculate_decimal_sum_and_difference : 
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := 
by 
  sorry

end calculate_decimal_sum_and_difference_l237_237407


namespace option_C_is_proposition_l237_237839

def is_proposition (s : Prop) : Prop := ∃ p : Prop, s = p

theorem option_C_is_proposition : is_proposition (4 + 3 = 8) := sorry

end option_C_is_proposition_l237_237839


namespace smallest_divisible_by_1_to_10_l237_237147

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l237_237147


namespace inequality_solution_range_l237_237445

variable (a : ℝ)

def f (x : ℝ) := 2 * x^2 - 8 * x - 4

theorem inequality_solution_range :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x - a > 0) ↔ a < -4 := 
by
  sorry

end inequality_solution_range_l237_237445


namespace bundles_burned_in_afternoon_l237_237554

theorem bundles_burned_in_afternoon 
  (morning_burn : ℕ)
  (start_bundles : ℕ)
  (end_bundles : ℕ)
  (h_morning_burn : morning_burn = 4)
  (h_start : start_bundles = 10)
  (h_end : end_bundles = 3)
  : (start_bundles - morning_burn - end_bundles) = 3 := 
by 
  sorry

end bundles_burned_in_afternoon_l237_237554


namespace div_37_permutation_l237_237660

-- Let A, B, C be digits of a three-digit number
variables (A B C : ℕ) -- these can take values from 0 to 9
variables (p : ℕ) -- integer multiplier for the divisibility condition

-- The main theorem stated as a Lean 4 problem
theorem div_37_permutation (h : 100 * A + 10 * B + C = 37 * p) : 
  ∃ (M : ℕ), (M = 100 * B + 10 * C + A ∨ M = 100 * C + 10 * A + B ∨ M = 100 * A + 10 * C + B ∨ M = 100 * C + 10 * B + A ∨ M = 100 * B + 10 * A + C) ∧ 37 ∣ M :=
by
  sorry

end div_37_permutation_l237_237660


namespace interval_for_systematic_sampling_l237_237370

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the sample size
def sample_size : ℕ := 30

-- Define the interval for systematic sampling
def interval_k : ℕ := total_students / sample_size

-- The theorem to prove that the interval k should be 40
theorem interval_for_systematic_sampling :
  interval_k = 40 := sorry

end interval_for_systematic_sampling_l237_237370


namespace find_q_l237_237316

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by
  sorry

end find_q_l237_237316


namespace intersection_M_N_l237_237246

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237246


namespace natalia_crates_l237_237726

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l237_237726


namespace Olivia_spent_25_dollars_l237_237558

theorem Olivia_spent_25_dollars
    (initial_amount : ℕ)
    (final_amount : ℕ)
    (spent_amount : ℕ)
    (h_initial : initial_amount = 54)
    (h_final : final_amount = 29)
    (h_spent : spent_amount = initial_amount - final_amount) :
    spent_amount = 25 := by
  sorry

end Olivia_spent_25_dollars_l237_237558


namespace problem1_problem2_l237_237719

-- Statement for Problem ①
theorem problem1 
: ( (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2) := by
  sorry

-- Statement for Problem ②
theorem problem2
: ((-99 - 11 / 12) * 24 = -2398) := by
  sorry

end problem1_problem2_l237_237719


namespace chalk_boxes_needed_l237_237847

theorem chalk_boxes_needed (pieces_per_box : ℕ) (total_pieces : ℕ) (pieces_per_box_pos : pieces_per_box > 0) : 
  (total_pieces + pieces_per_box - 1) / pieces_per_box = 194 :=
by 
  let boxes_needed := (total_pieces + pieces_per_box - 1) / pieces_per_box
  have h: boxes_needed = 194 := sorry
  exact h

end chalk_boxes_needed_l237_237847


namespace bug_back_at_A_l237_237787

noncomputable def prob_back_at_A_after_6_meters : ℚ := 159 / 972

theorem bug_back_at_A (P : ℕ → ℚ) (P_0 : P 0 = 1) 
  (trans : ∀ n, P (n + 1) = (1/2) * P(0) + (1/2) * P(n-1) + (1/6) * P(n-1 - 1)) :
  P 6 = 53 / 324 :=
by 
  have P1 : P 1 = 0 := sorry
  have P2 : P 2 = 1/6 := sorry
  have P3 : P 3 = 7/36 := sorry
  have P4 : P 4 = 19/108 := sorry
  have P5 : P 5 = 55/324 := sorry
  have P6 : P 6 = 159/972 := by sorry
  exact sorry

end bug_back_at_A_l237_237787


namespace find_max_sum_of_squares_l237_237786

open Real

theorem find_max_sum_of_squares 
  (a b c d : ℝ)
  (h1 : a + b = 17)
  (h2 : ab + c + d = 98)
  (h3 : ad + bc = 176)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 :=
sorry

end find_max_sum_of_squares_l237_237786


namespace number_of_real_solutions_l237_237456

theorem number_of_real_solutions (floor : ℝ → ℤ) 
  (h_floor : ∀ x, floor x = ⌊x⌋)
  (h_eq : ∀ x, 9 * x^2 - 45 * floor (x^2 - 1) + 94 = 0) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_real_solutions_l237_237456


namespace sin_alpha_in_second_quadrant_l237_237606

theorem sin_alpha_in_second_quadrant
  (α : ℝ)
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.tan α = - (8 / 15)) :
  Real.sin α = 8 / 17 :=
sorry

end sin_alpha_in_second_quadrant_l237_237606


namespace cost_equality_store_comparison_for_10_l237_237373

-- price definitions
def teapot_price := 30
def teacup_price := 5
def teapot_count := 5

-- store A and B promotional conditions
def storeA_cost (x : Nat) : Real := 5 * x + 125
def storeB_cost (x : Nat) : Real := 4.5 * x + 135

theorem cost_equality (x : Nat) (h : x > 5) :
  storeA_cost x = storeB_cost x → x = 20 := by
  sorry

theorem store_comparison_for_10 (x : Nat) (h : x = 10) :
  storeA_cost x < storeB_cost x := by
  sorry

end cost_equality_store_comparison_for_10_l237_237373


namespace lcm_1_to_10_l237_237101

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l237_237101


namespace range_of_a_l237_237921

noncomputable def p (a : ℝ) : Prop := 
  (1 + a)^2 + (1 - a)^2 < 4

noncomputable def q (a : ℝ) : Prop := 
  ∀ x : ℝ, x^2 + a * x + 1 ≥ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) ↔ (-2 ≤ a ∧ a ≤ -1) ∨ (1 ≤ a ∧ a ≤ 2) := 
by
  sorry

end range_of_a_l237_237921


namespace taxi_fare_for_100_miles_l237_237194

theorem taxi_fare_for_100_miles
  (base_fare : ℝ := 10)
  (proportional_fare : ℝ := 140 / 80)
  (fare_for_80_miles : ℝ := 150)
  (distance_80 : ℝ := 80)
  (distance_100 : ℝ := 100) :
  let additional_fare := proportional_fare * distance_100
  let total_fare_for_100_miles := base_fare + additional_fare
  total_fare_for_100_miles = 185 :=
by
  sorry

end taxi_fare_for_100_miles_l237_237194


namespace smallest_divisible_1_to_10_l237_237114

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l237_237114


namespace smallest_divisible_by_1_to_10_l237_237143

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l237_237143


namespace total_students_in_class_l237_237848

def period_length : ℕ := 40
def periods_per_student : ℕ := 4
def time_per_student : ℕ := 5

theorem total_students_in_class :
  ((period_length / time_per_student) * periods_per_student) = 32 :=
by
  sorry

end total_students_in_class_l237_237848


namespace range_of_m_tangent_not_parallel_l237_237753

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 / 2) * x^2 - k * x
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := f x + g x (m + (1 / m))
noncomputable def M (x : ℝ) (m : ℝ) : ℝ := f x - g x (m + (1 / m))

theorem range_of_m (m : ℝ) (h_extreme : ∃ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, h y m ≤ h x m) : 
  (0 < m ∧ m ≤ 1 / 2) ∨ (m ≥ 2) :=
  sorry

theorem tangent_not_parallel (x1 x2 x0 : ℝ) (m : ℝ) (h_zeros : M x1 m = 0 ∧ M x2 m = 0 ∧ x1 > x2 ∧ 2 * x0 = x1 + x2) :
  ¬ (∃ l : ℝ, ∀ x : ℝ, M x m = l * (x - x0) + M x0 m ∧ l = 0) :=
  sorry

end range_of_m_tangent_not_parallel_l237_237753


namespace intersection_M_N_l237_237287

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237287


namespace simplified_expression_value_l237_237495

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l237_237495


namespace max_sum_of_digits_l237_237387

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_sum_of_digits : ∃ h m : ℕ, h < 24 ∧ m < 60 ∧
  sum_of_digits h + sum_of_digits m = 24 :=
by
  sorry

end max_sum_of_digits_l237_237387


namespace copy_pages_l237_237047

theorem copy_pages (cost_per_5_pages : ℝ) (total_dollars : ℝ) : 
  (cost_per_5_pages = 10) → (total_dollars = 15) → (15 * 100 / 10 * 5 = 750) :=
by
  intros
  sorry

end copy_pages_l237_237047


namespace smallest_divisible_by_1_to_10_l237_237144

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l237_237144


namespace parts_in_batch_l237_237773

theorem parts_in_batch :
  ∃ a : ℕ, 500 ≤ a ∧ a ≤ 600 ∧ a % 20 = 13 ∧ a % 27 = 20 ∧ a = 533 :=
begin
  sorry
end

end parts_in_batch_l237_237773


namespace greatest_common_divisor_of_98_and_n_l237_237831

theorem greatest_common_divisor_of_98_and_n (n : ℕ) (h1 : ∃ (d : Finset ℕ),  d = {1, 7, 49} ∧ ∀ x ∈ d, x ∣ 98 ∧ x ∣ n) :
  ∃ (g : ℕ), g = 49 :=
by
  sorry

end greatest_common_divisor_of_98_and_n_l237_237831


namespace inequality_a_b_c_l237_237654

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_a_b_c_l237_237654


namespace find_a_l237_237029

theorem find_a (a : ℝ) (h : (3 * a + 2) + (a + 14) = 0) : a = -4 :=
sorry

end find_a_l237_237029


namespace smallest_divisible_by_1_to_10_l237_237146

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l237_237146


namespace smallest_number_divisible_1_to_10_l237_237118

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237118


namespace quadratic_has_distinct_real_roots_l237_237674

theorem quadratic_has_distinct_real_roots : 
  ∀ (x : ℝ), x^2 - 3 * x + 1 = 0 → ∀ (a b c : ℝ), a = 1 ∧ b = -3 ∧ c = 1 →
  (b^2 - 4 * a * c) > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l237_237674


namespace simplify_expression_l237_237488

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l237_237488


namespace polynomial_simplification_l237_237664

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 :=
by
  sorry

end polynomial_simplification_l237_237664


namespace largest_square_plots_l237_237192

theorem largest_square_plots (width length pathway_material : Nat) (width_eq : width = 30) (length_eq : length = 60) (pathway_material_eq : pathway_material = 2010) : ∃ (n : Nat), n * (2 * n) = 578 := 
by
  sorry

end largest_square_plots_l237_237192


namespace fraction_to_decimal_l237_237995

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237995


namespace change_calculation_l237_237342

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l237_237342


namespace area_difference_l237_237931

theorem area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15 / 2) :
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end area_difference_l237_237931


namespace paint_gallons_l237_237189

theorem paint_gallons (W B : ℕ) (h1 : 5 * B = 8 * W) (h2 : W + B = 6689) : B = 4116 :=
by
  sorry

end paint_gallons_l237_237189


namespace route_time_saving_zero_l237_237651

theorem route_time_saving_zero 
  (distance_X : ℝ) (speed_X : ℝ) 
  (total_distance_Y : ℝ) (construction_distance_Y : ℝ) (construction_speed_Y : ℝ)
  (normal_distance_Y : ℝ) (normal_speed_Y : ℝ)
  (hx1 : distance_X = 7)
  (hx2 : speed_X = 35)
  (hy1 : total_distance_Y = 6)
  (hy2 : construction_distance_Y = 1)
  (hy3 : construction_speed_Y = 10)
  (hy4 : normal_distance_Y = 5)
  (hy5 : normal_speed_Y = 50) :
  (distance_X / speed_X * 60) - 
  ((construction_distance_Y / construction_speed_Y * 60) + 
  (normal_distance_Y / normal_speed_Y * 60)) = 0 := 
sorry

end route_time_saving_zero_l237_237651


namespace billion_in_scientific_notation_l237_237557

theorem billion_in_scientific_notation :
  (4.55 * 10^9) = (4.55 * 10^9) := by
  sorry

end billion_in_scientific_notation_l237_237557


namespace tan_arithmetic_seq_value_l237_237028

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Given conditions and the final proof goal
theorem tan_arithmetic_seq_value (h_arith : arithmetic_seq a d)
    (h_sum : a 0 + a 6 + a 12 = Real.pi) :
    Real.tan (a 1 + a 11) = -Real.sqrt 3 := sorry

end tan_arithmetic_seq_value_l237_237028


namespace decimal_equivalent_one_quarter_power_one_l237_237093

theorem decimal_equivalent_one_quarter_power_one : (1 / 4 : ℝ) ^ 1 = 0.25 := by
  sorry

end decimal_equivalent_one_quarter_power_one_l237_237093


namespace lcm_1_to_10_l237_237150

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

end lcm_1_to_10_l237_237150


namespace lcm_1_10_l237_237134

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l237_237134


namespace gcd_sum_lcm_eq_gcd_l237_237807

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end gcd_sum_lcm_eq_gcd_l237_237807


namespace find_number_l237_237973

theorem find_number (a p x : ℕ) (h1 : p = 36) (h2 : 6 * a = 6 * (2 * p + x)) : x = 9 :=
by
  sorry

end find_number_l237_237973


namespace smaller_number_is_17_l237_237825

theorem smaller_number_is_17 (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_is_17_l237_237825


namespace find_f_sqrt2_l237_237980

theorem find_f_sqrt2 (f : ℝ → ℝ)
  (hf : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x * y) = f x + f y)
  (hf8 : f 8 = 3) :
  f (Real.sqrt 2) = 1 / 2 := by
  sorry

end find_f_sqrt2_l237_237980


namespace intersection_M_N_l237_237286

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l237_237286


namespace max_composite_numbers_l237_237465

theorem max_composite_numbers (s : set ℕ) (hs : ∀ n ∈ s, n < 1500 ∧ ∃ p : ℕ, prime p ∧ p ∣ n) (hs_gcd : ∀ x y ∈ s, x ≠ y → Nat.gcd x y = 1) :
  s.card ≤ 12 := 
by sorry

end max_composite_numbers_l237_237465


namespace intersection_M_N_l237_237269

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237269


namespace smallest_divisible_1_to_10_l237_237113

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l237_237113


namespace find_symmetric_L_like_shape_l237_237892

-- Define the L-like shape and its mirror image
def L_like_shape : Type := sorry  -- Placeholder for the actual geometry definition
def mirrored_L_like_shape : Type := sorry  -- Placeholder for the actual mirrored shape

-- Condition: The vertical symmetry function
def symmetric_about_vertical_line (shape1 shape2 : Type) : Prop :=
   sorry  -- Define what it means for shape1 to be symmetric to shape2

-- Given conditions (A to E as L-like shape variations)
def option_A : Type := sorry  -- An inverted L-like shape
def option_B : Type := sorry  -- An upside-down T-like shape
def option_C : Type := mirrored_L_like_shape  -- A mirrored L-like shape
def option_D : Type := sorry  -- A rotated L-like shape by 180 degrees
def option_E : Type := L_like_shape  -- An unchanged L-like shape

-- The theorem statement
theorem find_symmetric_L_like_shape :
  symmetric_about_vertical_line L_like_shape option_C :=
  sorry

end find_symmetric_L_like_shape_l237_237892


namespace dilation_origin_distance_l237_237851

open Real

-- Definition of points and radii
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Given conditions as definitions
def original_circle := Circle.mk (3, 3) 3
def dilated_circle := Circle.mk (8, 10) 5
def dilation_factor := 5 / 3

-- Problem statement to prove
theorem dilation_origin_distance :
  let d₀ := dist (0, 0) (-6, -6)
  let d₁ := dilation_factor * d₀
  d₁ - d₀ = 4 * sqrt 2 :=
by
  sorry

end dilation_origin_distance_l237_237851


namespace find_value_of_expression_l237_237633

theorem find_value_of_expression (x : ℝ) (h : x^2 + (1 / x^2) = 5) : x^4 + (1 / x^4) = 23 :=
by
  sorry

end find_value_of_expression_l237_237633


namespace prism_faces_l237_237856

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end prism_faces_l237_237856


namespace odd_function_increasing_on_negative_interval_l237_237446

theorem odd_function_increasing_on_negative_interval {f : ℝ → ℝ}
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_min_value : f 3 = 1) :
  (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) ∧ f (-3) = -1 := 
sorry

end odd_function_increasing_on_negative_interval_l237_237446


namespace danny_bottle_caps_l237_237207

variable (caps_found : Nat) (caps_existing : Nat)
variable (wrappers_found : Nat) (wrappers_existing : Nat)

theorem danny_bottle_caps:
  caps_found = 58 → caps_existing = 12 →
  wrappers_found = 25 → wrappers_existing = 11 →
  (caps_found + caps_existing) - (wrappers_found + wrappers_existing) = 34 := 
by
  intros h1 h2 h3 h4
  sorry

end danny_bottle_caps_l237_237207


namespace Wang_returns_to_start_electricity_consumed_l237_237963

-- Definition of movements
def movements : List ℤ := [+6, -3, +10, -8, +12, -7, -10]

-- Definition of height per floor and electricity consumption per meter
def height_per_floor : ℝ := 3
def electricity_per_meter : ℝ := 0.2

-- Problem statement 1: Prove that Mr. Wang returned to the starting position
theorem Wang_returns_to_start : 
  List.sum movements = 0 :=
  sorry

-- Problem statement 2: Prove the total electricity consumption
theorem electricity_consumed : 
  let total_floors := List.sum (List.map Int.natAbs movements)
  let total_meters := total_floors * height_per_floor
  total_meters * electricity_per_meter = 33.6 := 
  sorry

end Wang_returns_to_start_electricity_consumed_l237_237963


namespace lcm_1_to_10_l237_237149

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

end lcm_1_to_10_l237_237149


namespace determine_n_l237_237458

theorem determine_n (n : ℕ) (h1 : 0 < n) 
(h2 : ∃ (sols : Finset (ℕ × ℕ × ℕ)), 
  (∀ (x y z : ℕ), (x, y, z) ∈ sols ↔ 3 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ sols.card = 55) : 
  n = 36 := 
by 
  sorry 

end determine_n_l237_237458


namespace ratio_suspension_to_fingers_toes_l237_237785

-- Definition of conditions
def suspension_days_per_instance : Nat := 3
def bullying_instances : Nat := 20
def fingers_and_toes : Nat := 20

-- Theorem statement
theorem ratio_suspension_to_fingers_toes :
  (suspension_days_per_instance * bullying_instances) / fingers_and_toes = 3 :=
by
  sorry

end ratio_suspension_to_fingers_toes_l237_237785


namespace intersection_M_N_l237_237281

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237281


namespace regular_polygon_sides_l237_237322

theorem regular_polygon_sides (n : ℕ) (h : ∀ (θ : ℝ), θ = 36 → θ = 360 / n) : n = 10 := by
  sorry

end regular_polygon_sides_l237_237322


namespace min_value_frac_sum_l237_237919

theorem min_value_frac_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) : 
  ∃ c : ℝ, c = 4 ∧ (∀ m n, 2 * m + n = 2 → m * n > 0 → (1 / m + 2 / n) ≥ c) :=
sorry

end min_value_frac_sum_l237_237919


namespace transitiveSim_l237_237833

def isGreat (f : ℕ × ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1, n + 1) * f (m, n) - f (m + 1, n) * f (m, n + 1) = 1

def seqSim (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ × ℕ → ℤ, isGreat f ∧ (∀ n, f (n, 0) = A n) ∧ (∀ n, f (0, n) = B n)

theorem transitiveSim (A B C D : ℕ → ℤ)
  (h1 : seqSim A B)
  (h2 : seqSim B C)
  (h3 : seqSim C D) : seqSim D A :=
sorry

end transitiveSim_l237_237833


namespace circle_area_difference_l237_237929

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 30) (h2 : r2 = 7.5) : 
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end circle_area_difference_l237_237929


namespace expand_and_simplify_product_l237_237211

variable (x : ℝ)

theorem expand_and_simplify_product :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 :=
by
  sorry

end expand_and_simplify_product_l237_237211


namespace divisible_bc_ad_l237_237075

theorem divisible_bc_ad
  (a b c d u : ℤ)
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) :
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end divisible_bc_ad_l237_237075


namespace solve_y_eq_l237_237587

theorem solve_y_eq :
  ∀ y: ℝ, y ≠ -1 → (y^3 - 3 * y^2) / (y^2 + 2 * y + 1) + 2 * y = -1 → 
  y = 1 / Real.sqrt 3 ∨ y = -1 / Real.sqrt 3 :=
by sorry

end solve_y_eq_l237_237587


namespace smallest_angle_in_trapezoid_l237_237939

theorem smallest_angle_in_trapezoid 
  (a d : ℝ) 
  (h1 : a + 2 * d = 150) 
  (h2 : a + d + a + 2 * d = 180) : 
  a = 90 := 
sorry

end smallest_angle_in_trapezoid_l237_237939


namespace compound_interest_is_correct_l237_237473

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

theorem compound_interest_is_correct :
  let P := 660 / (0.2 : ℝ)
  (compound_interest P 10 2) = 693 := 
by
  -- Definitions of simple_interest and compound_interest are used
  -- The problem conditions help us conclude
  let P := 660 / (0.2 : ℝ)
  have h1 : simple_interest P 10 2 = 660 := by sorry
  have h2 : compound_interest P 10 2 = 693 := by sorry
  exact h2

end compound_interest_is_correct_l237_237473


namespace alice_lost_second_game_l237_237395

/-- Alice, Belle, and Cathy had an arm-wrestling contest. In each game, two girls wrestled, while the third rested.
After each game, the winner played the next game against the girl who had rested.
Given that Alice played 10 times, Belle played 15 times, and Cathy played 17 times; prove Alice lost the second game. --/

theorem alice_lost_second_game (alice_plays : ℕ) (belle_plays : ℕ) (cathy_plays : ℕ) :
  alice_plays = 10 → belle_plays = 15 → cathy_plays = 17 → 
  ∃ (lost_second_game : String), lost_second_game = "Alice" := by
  intros hA hB hC
  sorry

end alice_lost_second_game_l237_237395


namespace find_alpha_l237_237971

-- Define the given condition that alpha is inversely proportional to beta
def inv_proportional (α β : ℝ) (k : ℝ) : Prop := α * β = k

-- Main theorem statement
theorem find_alpha (α β k : ℝ) (h1 : inv_proportional 2 5 k) (h2 : inv_proportional α (-10) k) : α = -1 := by
  -- Given the conditions, the proof would follow, but it's not required here.
  sorry

end find_alpha_l237_237971


namespace minimize_expression_l237_237449

theorem minimize_expression : 
  let a := -1
  let b := -0.5
  (a + b) ≤ (a - b) ∧ (a + b) ≤ (a * b) ∧ (a + b) ≤ (a / b) := by
  let a := -1
  let b := -0.5
  sorry

end minimize_expression_l237_237449


namespace find_other_number_l237_237818

noncomputable def calculateB (lcm hcf a : ℕ) : ℕ :=
  (lcm * hcf) / a

theorem find_other_number :
  ∃ B : ℕ, (calculateB 76176 116 8128) = 1087 :=
by
  use 1087
  sorry

end find_other_number_l237_237818


namespace smallest_multiple_1_through_10_l237_237137

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l237_237137


namespace simplified_expression_value_l237_237494

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l237_237494


namespace intersection_M_N_l237_237284

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l237_237284


namespace thief_speed_is_43_75_l237_237392

-- Given Information
def speed_owner : ℝ := 50
def time_head_start : ℝ := 0.5
def total_time_to_overtake : ℝ := 4

-- Question: What is the speed of the thief's car v?
theorem thief_speed_is_43_75 (v : ℝ) (hv : 4 * v = speed_owner * (total_time_to_overtake - time_head_start)) : v = 43.75 := 
by {
  -- The proof of this theorem is omitted as it is not required.
  sorry
}

end thief_speed_is_43_75_l237_237392


namespace simplify_and_evaluate_l237_237502

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l237_237502


namespace crayons_per_color_in_each_box_l237_237704

def crayons_in_each_box : ℕ := 2

theorem crayons_per_color_in_each_box
  (colors : ℕ)
  (boxes_per_hour : ℕ)
  (crayons_in_4_hours : ℕ)
  (hours : ℕ)
  (total_boxes : ℕ := boxes_per_hour * hours)
  (crayons_per_box : ℕ := crayons_in_4_hours / total_boxes)
  (crayons_per_color : ℕ := crayons_per_box / colors)
  (colors_eq : colors = 4)
  (boxes_per_hour_eq : boxes_per_hour = 5)
  (crayons_in_4_hours_eq : crayons_in_4_hours = 160)
  (hours_eq : hours = 4) : crayons_per_color = crayons_in_each_box :=
by {
  sorry
}

end crayons_per_color_in_each_box_l237_237704


namespace sofie_total_distance_l237_237071

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

end sofie_total_distance_l237_237071


namespace direct_proportion_function_l237_237923

theorem direct_proportion_function (m : ℝ) 
  (h1 : m + 1 ≠ 0) 
  (h2 : m^2 - 1 = 0) : 
  m = 1 :=
sorry

end direct_proportion_function_l237_237923


namespace percentage_fractions_l237_237990

theorem percentage_fractions : (3 / 8 / 100) * (160 : ℚ) = 3 / 5 :=
by
  sorry

end percentage_fractions_l237_237990


namespace intersection_M_N_l237_237293

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l237_237293


namespace evaluate_expression_l237_237902

theorem evaluate_expression : 3^(2 + 3 + 4) - (3^2 * 3^3 + 3^4) = 19359 :=
by
  sorry

end evaluate_expression_l237_237902


namespace find_height_of_cylinder_l237_237079

theorem find_height_of_cylinder (h r : ℝ) (π : ℝ) (SA : ℝ) (r_val : r = 3) (SA_val : SA = 36 * π) 
  (SA_formula : SA = 2 * π * r^2 + 2 * π * r * h) : h = 3 := 
by
  sorry

end find_height_of_cylinder_l237_237079


namespace prove_lesser_fraction_l237_237675

noncomputable def lesser_fraction (x y : ℚ) : Prop :=
  x + y = 8/9 ∧ x * y = 1/8 ∧ min x y = 7/40

theorem prove_lesser_fraction :
  ∃ x y : ℚ, lesser_fraction x y :=
sorry

end prove_lesser_fraction_l237_237675


namespace problem_divisibility_l237_237643

theorem problem_divisibility (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by
  sorry

end problem_divisibility_l237_237643


namespace smallest_number_divisible_by_1_to_10_l237_237095

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237095


namespace school_student_count_l237_237709

theorem school_student_count (pencils erasers pencils_per_student erasers_per_student students : ℕ) 
    (h1 : pencils = 195) 
    (h2 : erasers = 65) 
    (h3 : pencils_per_student = 3)
    (h4 : erasers_per_student = 1) :
    students = pencils / pencils_per_student ∧ students = erasers / erasers_per_student → students = 65 :=
by
  sorry

end school_student_count_l237_237709


namespace polynomial_has_real_root_l237_237413

open Real

theorem polynomial_has_real_root (a : ℝ) : 
  ∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 - x + a = 0 :=
sorry

end polynomial_has_real_root_l237_237413


namespace which_calc_is_positive_l237_237693

theorem which_calc_is_positive :
  (-3 + 7 - 5 < 0) ∧
  ((1 - 2) * 3 < 0) ∧
  (-16 / (↑(-3)^2) < 0) ∧
  (-2^4 * (-6) > 0) :=
by
sorry

end which_calc_is_positive_l237_237693


namespace wood_burned_in_afternoon_l237_237548

theorem wood_burned_in_afternoon 
  (burned_morning : ℕ) 
  (start_bundles : ℕ) 
  (end_bundles : ℕ) 
  (burned_afternoon : ℕ) 
  (h1 : burned_morning = 4) 
  (h2 : start_bundles = 10) 
  (h3 : end_bundles = 3) 
  (h4 : burned_morning + burned_afternoon = start_bundles - end_bundles) :
  burned_afternoon = 3 := 
sorry

end wood_burned_in_afternoon_l237_237548


namespace simplified_expression_value_l237_237496

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l237_237496


namespace range_of_f_l237_237003

noncomputable def f (x : ℝ) : ℝ := x + |x - 2|

theorem range_of_f : Set.range f = Set.Ici 2 :=
sorry

end range_of_f_l237_237003


namespace symmetric_points_sum_l237_237934

-- Definition of symmetry with respect to the origin for points M and N
def symmetric_with_origin (M N : ℝ × ℝ) : Prop :=
  M.1 = -N.1 ∧ M.2 = -N.2

-- Definition of the points M and N from the original problem
variables {a b : ℝ}
def M : ℝ × ℝ := (3, a - 2)
def N : ℝ × ℝ := (b, a)

-- The theorem statement
theorem symmetric_points_sum :
  symmetric_with_origin M N → a + b = -2 :=
by
  intro h
  cases h with hx hy
  -- here would go the detailed proof, which we're omitting
  sorry

end symmetric_points_sum_l237_237934


namespace tan_sum_l237_237568

theorem tan_sum (A B : ℝ) (h₁ : A = 17) (h₂ : B = 28) :
  Real.tan (A) + Real.tan (B) + Real.tan (A) * Real.tan (B) = 1 := 
by
  sorry

end tan_sum_l237_237568


namespace ramola_rank_from_first_l237_237352

-- Conditions definitions
def total_students : ℕ := 26
def ramola_rank_from_last : ℕ := 13

-- Theorem statement
theorem ramola_rank_from_first : total_students - (ramola_rank_from_last - 1) = 14 := 
by 
-- We use 'by' to begin the proof block
sorry 
-- We use 'sorry' to indicate the proof is omitted

end ramola_rank_from_first_l237_237352


namespace fraction_to_decimal_l237_237996

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237996


namespace remainder_determined_l237_237476

theorem remainder_determined (p a b : ℤ) (h₀: Nat.Prime (Int.natAbs p)) (h₁ : ¬ (p ∣ a)) (h₂ : ¬ (p ∣ b)) :
  ∃ (r : ℤ), (r ≡ a [ZMOD p]) ∧ (r ≡ b [ZMOD p]) ∧ (r ≡ (a * b) [ZMOD p]) →
  (a ≡ r [ZMOD p]) := sorry

end remainder_determined_l237_237476


namespace find_y_given_conditions_l237_237037

theorem find_y_given_conditions (x y : ℝ) (h1 : x^(3 * y) = 27) (h2 : x = 3) : y = 1 := 
by
  sorry

end find_y_given_conditions_l237_237037


namespace center_of_circle_l237_237186

theorem center_of_circle (
  center : ℝ × ℝ
) :
  (∀ (p : ℝ × ℝ), (p.1 * 3 + p.2 * 4 = 24) ∨ (p.1 * 3 + p.2 * 4 = -6) → (dist center p = dist center p)) ∧
  (center.1 * 3 - center.2 = 0)
  → center = (3 / 5, 9 / 5) :=
by
  sorry

end center_of_circle_l237_237186


namespace maximum_abc_827_l237_237644

noncomputable def maximum_abc (a b c : ℝ) := (a * b * c)

theorem maximum_abc_827 (a b c : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0) 
  (h4: (a * b) + c = (a + c) * (b + c)) 
  (h5: a + b + c = 2) : 
  maximum_abc a b c = 8 / 27 := 
by 
  sorry

end maximum_abc_827_l237_237644


namespace right_triangle_AB_CA_BC_l237_237691

namespace TriangleProof

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def A : point := (5, -2)
def B : point := (1, 5)
def C : point := (-1, 2)

def AB2 := dist A B
def BC2 := dist B C
def CA2 := dist C A

theorem right_triangle_AB_CA_BC : CA2 + BC2 = AB2 :=
by 
  -- proof will be filled here
  sorry

end TriangleProof

end right_triangle_AB_CA_BC_l237_237691


namespace company_employee_percentage_l237_237042

theorem company_employee_percentage (M : ℝ)
  (h1 : 0.20 * M + 0.40 * (1 - M) = 0.31000000000000007) :
  M = 0.45 :=
sorry

end company_employee_percentage_l237_237042


namespace horse_total_value_l237_237852

theorem horse_total_value (n : ℕ) (a r : ℕ) (h₁ : n = 32) (h₂ : a = 1) (h₃ : r = 2) :
  (a * (r ^ n - 1) / (r - 1)) = 4294967295 :=
by 
  rw [h₁, h₂, h₃]
  sorry

end horse_total_value_l237_237852


namespace number_of_red_balls_l237_237937

theorem number_of_red_balls (m : ℕ) (h1 : ∃ m : ℕ, (3 / (m + 3) : ℚ) = 1 / 4) : m = 9 :=
by
  obtain ⟨m, h1⟩ := h1
  sorry

end number_of_red_balls_l237_237937


namespace marble_problem_l237_237826

def total_marbles_originally 
  (white_marbles : ℕ := 20) 
  (blue_marbles : ℕ) 
  (red_marbles : ℕ := blue_marbles) 
  (total_left : ℕ := 40)
  (jack_removes : ℕ := 2 * (white_marbles - blue_marbles)) : ℕ :=
  white_marbles + blue_marbles + red_marbles

theorem marble_problem : 
  ∀ (white_marbles : ℕ := 20) 
    (blue_marbles red_marbles : ℕ) 
    (jack_removes total_left : ℕ),
    red_marbles = blue_marbles →
    jack_removes = 2 * (white_marbles - blue_marbles) →
    total_left = total_marbles_originally white_marbles blue_marbles red_marbles - jack_removes →
    total_left = 40 →
    total_marbles_originally white_marbles blue_marbles red_marbles = 50 :=
by
  intros white_marbles blue_marbles red_marbles jack_removes total_left h1 h2 h3 h4
  sorry

end marble_problem_l237_237826


namespace tan_beta_eq_neg13_l237_237018

variables (α β : Real)

theorem tan_beta_eq_neg13 (h1 : Real.tan α = 2) (h2 : Real.tan (α - β) = -3/5) : 
  Real.tan β = -13 := 
by 
  sorry

end tan_beta_eq_neg13_l237_237018


namespace prism_faces_l237_237860

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end prism_faces_l237_237860


namespace lcm_1_10_l237_237133

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l237_237133


namespace integer_values_abs_lt_5pi_l237_237306

theorem integer_values_abs_lt_5pi : 
  ∃ n : ℕ, n = 31 ∧ ∀ x : ℤ, |(x : ℝ)| < 5 * Real.pi → x ∈ (Finset.Icc (-15) 15) := 
sorry

end integer_values_abs_lt_5pi_l237_237306


namespace smallest_number_divisible_1_to_10_l237_237172

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237172


namespace smallest_number_divisible_by_1_to_10_l237_237096

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237096


namespace largest_non_representable_integer_l237_237375

theorem largest_non_representable_integer (n a b : ℕ) (h₁ : n = 42 * a + b)
  (h₂ : 0 ≤ b) (h₃ : b < 42) (h₄ : ¬ (b % 6 = 0)) :
  n ≤ 252 :=
sorry

end largest_non_representable_integer_l237_237375


namespace last_digit_2_pow_2023_l237_237464

-- Definitions
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem last_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 :=
by
  -- We will assume and use the properties mentioned in the solution steps.
  -- The proof process is skipped here with 'sorry'.
  sorry

end last_digit_2_pow_2023_l237_237464


namespace solve_system_of_equations_l237_237970

theorem solve_system_of_equations
  (x y : ℝ)
  (h1 : 1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2))
  (h2 : 1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) :
  x = (3 ^ (1 / 5) + 1) / 2 ∧ y = (3 ^ (1 / 5) - 1) / 2 :=
by
  sorry

end solve_system_of_equations_l237_237970


namespace possible_values_of_a_l237_237808

theorem possible_values_of_a (x y a : ℝ)
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 :=
sorry

end possible_values_of_a_l237_237808


namespace number_satisfies_equation_l237_237349

theorem number_satisfies_equation :
  ∃ x : ℝ, (x^2 + 100 = (x - 20)^2) ∧ x = 7.5 :=
by
  use 7.5
  sorry

end number_satisfies_equation_l237_237349


namespace remainder_when_7x_div_9_l237_237182

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end remainder_when_7x_div_9_l237_237182


namespace alcohol_to_water_ratio_l237_237368

theorem alcohol_to_water_ratio (p q r : ℝ) :
  let alcohol := (p / (p + 1) + q / (q + 1) + r / (r + 1))
  let water := (1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1))
  (alcohol / water) = (p * q * r + p * q + p * r + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 1) :=
sorry

end alcohol_to_water_ratio_l237_237368


namespace intersection_of_M_and_N_l237_237231

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l237_237231


namespace minimize_folded_area_l237_237225

-- defining the problem as statements in Lean
variables (a M N : ℝ) (M_on_AB : M > 0 ∧ M < a) (N_on_CD : N > 0 ∧ N < a)

-- main theorem statement
theorem minimize_folded_area :
  BM = 5 * a / 8 →
  CN = a / 8 →
  S = 3 * a ^ 2 / 8 := sorry

end minimize_folded_area_l237_237225


namespace quadratic_decreasing_right_of_axis_of_symmetry_l237_237908

theorem quadratic_decreasing_right_of_axis_of_symmetry :
  ∀ x : ℝ, -2 * (x - 1)^2 < -2 * (x + 1 - 1)^2 →
  (∀ x' : ℝ, x' > 1 → -2 * (x' - 1)^2 < -2 * (x + 1 - 1)^2) :=
by
  sorry

end quadratic_decreasing_right_of_axis_of_symmetry_l237_237908


namespace range_of_x_l237_237793

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) : x ≤ -2 ∨ x ≥ 3 :=
sorry

end range_of_x_l237_237793


namespace obtuse_angle_only_dihedral_planar_l237_237712

/-- Given the range of three types of angles, prove that only the dihedral angle's planar angle can be obtuse. -/
theorem obtuse_angle_only_dihedral_planar 
  (α : ℝ) (β : ℝ) (γ : ℝ) 
  (hα : 0 < α ∧ α ≤ 90)
  (hβ : 0 ≤ β ∧ β ≤ 90)
  (hγ : 0 ≤ γ ∧ γ < 180) : 
  (90 < γ ∧ (¬(90 < α)) ∧ (¬(90 < β))) :=
by 
  sorry

end obtuse_angle_only_dihedral_planar_l237_237712


namespace minimum_max_abs_x2_sub_2xy_l237_237584

theorem minimum_max_abs_x2_sub_2xy {y : ℝ} :
  ∃ y : ℝ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y) ≥ 0) ∧
           (∀ y' ∈ Set.univ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y') ≥ abs (x^2 - 2*x*y))) :=
sorry

end minimum_max_abs_x2_sub_2xy_l237_237584


namespace final_coordinates_of_A_l237_237803

-- Define the initial points
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (5, -5)
def C : ℝ × ℝ := (2, -4)

-- Define the translation operation
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

-- Define the rotation operation (180 degrees around a point (h, k))
def rotate180 (p : ℝ × ℝ) (h k : ℝ) : ℝ × ℝ :=
  (2 * h - p.1, 2 * k - p.2)

-- Translate point A
def A' := translate A 4 3

-- Rotate the translated point A' 180 degrees around the point (4, 0)
def A'' := rotate180 A' 4 0

-- The final coordinates of point A after transformations should be (1, -1)
theorem final_coordinates_of_A : A'' = (1, -1) :=
  sorry

end final_coordinates_of_A_l237_237803


namespace intersection_complement_M_N_l237_237304

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_complement_M_N :
  (U \ M) ∩ N = {-3, -4} :=
by {
  sorry
}

end intersection_complement_M_N_l237_237304


namespace pizza_slices_l237_237685

theorem pizza_slices (S L : ℕ) (h1 : S + L = 36) (h2 : L = 2 * S) :
  (8 * S + 12 * L) = 384 :=
by
  sorry

end pizza_slices_l237_237685


namespace sculpture_height_l237_237408

theorem sculpture_height (base_height : ℕ) (total_height_ft : ℝ) (inches_per_foot : ℕ) 
  (h1 : base_height = 8) (h2 : total_height_ft = 3.5) (h3 : inches_per_foot = 12) : 
  (total_height_ft * inches_per_foot - base_height) = 34 := 
by
  sorry

end sculpture_height_l237_237408


namespace average_price_of_six_toys_l237_237529

/-- Define the average cost of toys given the number of toys and their total cost -/
def avg_cost (total_cost : ℕ) (num_toys : ℕ) : ℕ :=
  total_cost / num_toys

/-- Define the total cost of toys given a list of individual toy costs -/
def total_cost (costs : List ℕ) : ℕ :=
  costs.foldl (· + ·) 0

/-- The main theorem -/
theorem average_price_of_six_toys :
  let dhoni_toys := 5
  let avg_cost_dhoni := 10
  let total_cost_dhoni := dhoni_toys * avg_cost_dhoni
  let david_toy_cost := 16
  let total_toys := dhoni_toys + 1
  total_cost_dhoni + david_toy_cost = 66 →
  avg_cost (66) (total_toys) = 11 :=
by
  -- Introduce the conditions and hypothesis
  intros total_cost_of_6_toys H
  -- Simplify the expression
  sorry  -- Proof skipped

end average_price_of_six_toys_l237_237529


namespace S_11_is_22_l237_237436

-- Definitions and conditions
variable (a_1 d : ℤ) -- first term and common difference of the arithmetic sequence
noncomputable def S (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- The given condition
variable (h : S a_1 d 8 - S a_1 d 3 = 10)

-- The proof goal
theorem S_11_is_22 : S a_1 d 11 = 22 :=
by
  sorry

end S_11_is_22_l237_237436


namespace min_value_x_add_y_l237_237074

variable {x y : ℝ}
variable (hx : 0 < x) (hy : 0 < y)
variable (h : 2 * x + 8 * y - x * y = 0)

theorem min_value_x_add_y : x + y ≥ 18 :=
by
  /- Proof goes here -/
  sorry

end min_value_x_add_y_l237_237074


namespace hyperbola_asymptotes_l237_237214

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the equations for the asymptotes
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

-- State the theorem
theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola_eq x y → (asymptote_pos x y ∨ asymptote_neg x y) := 
by
  sorry

end hyperbola_asymptotes_l237_237214


namespace second_root_l237_237720

variables {a b c x : ℝ}

theorem second_root (h : a * (b + c) * x ^ 2 - b * (c + a) * x + c * (a + b) = 0)
(hroot : a * (b + c) * (-1) ^ 2 - b * (c + a) * (-1) + c * (a + b) = 0) :
  ∃ k : ℝ, k = - c * (a + b) / (a * (b + c)) ∧ a * (b + c) * k ^ 2 - b * (c + a) * k + c * (a + b) = 0 :=
sorry

end second_root_l237_237720


namespace sum_of_permutations_is_divisible_by_37_l237_237887

theorem sum_of_permutations_is_divisible_by_37
  (A B C : ℕ)
  (h : 37 ∣ (100 * A + 10 * B + C)) :
  37 ∣ (100 * B + 10 * C + A + 100 * C + 10 * A + B) :=
by
  sorry

end sum_of_permutations_is_divisible_by_37_l237_237887


namespace exists_consecutive_integers_sum_cube_l237_237208

theorem exists_consecutive_integers_sum_cube :
  ∃ (n : ℤ), ∃ (k : ℤ), 1981 * (n + 990) = k^3 :=
by
  sorry

end exists_consecutive_integers_sum_cube_l237_237208


namespace maximum_OA_plus_OB_l237_237454

noncomputable def C (a : ℝ) (θ : ℝ) : ℝ := 2 * a * Real.cos θ
noncomputable def l (θ : ℝ) : ℝ := (3/2) / Real.cos (θ - π / 3)
noncomputable def OA (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.cos θ
noncomputable def OB (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.cos (θ + π / 3)

-- main theorem
theorem maximum_OA_plus_OB {a : ℝ} (h₀ : 0 < a) 
(h₁ : ∀ θ, C a θ = l θ) 
(h₂ : ∀ A B, C a (angle A) = 2 * Real.cos (angle A) /\ C a (angle B) = 2 * Real.cos (angle B))
(h₃ : ∀ θ, OA a θ + OB a θ = 2 * Real.sqrt 3 * Real.cos (θ + π / 6)) :
  ∃ θ, (OA a θ + OB a θ) = 2 * Real.sqrt 3 :=
by
  sorry

end maximum_OA_plus_OB_l237_237454


namespace last_digits_nn_periodic_l237_237351

theorem last_digits_nn_periodic (n : ℕ) : 
  ∃ p > 0, ∀ k, (n + k * p)^(n + k * p) % 10 = n^n % 10 := 
sorry

end last_digits_nn_periodic_l237_237351


namespace vowel_soup_sequences_count_l237_237196

theorem vowel_soup_sequences_count :
  let vowels := 5
  let sequence_length := 6
  vowels ^ sequence_length = 15625 :=
by
  sorry

end vowel_soup_sequences_count_l237_237196


namespace pipe_R_fill_time_l237_237653

theorem pipe_R_fill_time (P_rate Q_rate combined_rate : ℝ) (hP : P_rate = 1 / 2) (hQ : Q_rate = 1 / 4)
  (h_combined : combined_rate = 1 / 1.2) : (∃ R_rate : ℝ, R_rate = 1 / 12) :=
by
  sorry

end pipe_R_fill_time_l237_237653


namespace smallest_number_divisible_by_1_to_10_l237_237097

open Classical
open Finset

def is_lcm (a : ℕ) (S : Finset ℕ) : Prop :=
  ∀ b : ℕ, (∀ s ∈ S, s ∣ b) ↔ (a ∣ b)

theorem smallest_number_divisible_by_1_to_10 :
  ∃ a : ℕ, is_lcm a (range 1 11) ∧ a = 2520 := by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237097


namespace white_pairs_coincide_l237_237723

theorem white_pairs_coincide 
    (red_triangles : ℕ)
    (blue_triangles : ℕ)
    (white_triangles : ℕ)
    (red_pairs : ℕ)
    (blue_pairs : ℕ)
    (red_white_pairs : ℕ)
    (coinciding_white_pairs : ℕ) :
    red_triangles = 4 → 
    blue_triangles = 6 →
    white_triangles = 10 →
    red_pairs = 3 →
    blue_pairs = 4 →
    red_white_pairs = 3 →
    coinciding_white_pairs = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end white_pairs_coincide_l237_237723


namespace total_visible_legs_l237_237636

-- Defining the conditions
def num_crows : ℕ := 4
def num_pigeons : ℕ := 3
def num_flamingos : ℕ := 5
def num_sparrows : ℕ := 8

def legs_per_crow : ℕ := 2
def legs_per_pigeon : ℕ := 2
def legs_per_flamingo : ℕ := 3
def legs_per_sparrow : ℕ := 2

-- Formulating the theorem that we need to prove
theorem total_visible_legs :
  (num_crows * legs_per_crow) +
  (num_pigeons * legs_per_pigeon) +
  (num_flamingos * legs_per_flamingo) +
  (num_sparrows * legs_per_sparrow) = 45 := by sorry

end total_visible_legs_l237_237636


namespace inv_88_mod_89_l237_237213

theorem inv_88_mod_89 : (88 * 88) % 89 = 1 := by
  sorry

end inv_88_mod_89_l237_237213


namespace remi_water_bottle_capacity_l237_237480

-- Let's define the problem conditions
def daily_refills : ℕ := 3
def days : ℕ := 7
def total_spilled : ℕ := 5 + 8 -- Total spilled water in ounces
def total_intake : ℕ := 407 -- Total amount of water drunk in 7 days

-- The capacity of Remi's water bottle is the quantity we need to prove
def bottle_capacity (x : ℕ) : Prop :=
  daily_refills * days * x - total_spilled = total_intake

-- Statement of the proof problem
theorem remi_water_bottle_capacity : bottle_capacity 20 :=
by
  sorry

end remi_water_bottle_capacity_l237_237480


namespace truncated_quadrilateral_pyramid_exists_l237_237588

theorem truncated_quadrilateral_pyramid_exists :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by
  sorry

end truncated_quadrilateral_pyramid_exists_l237_237588


namespace prism_faces_l237_237864

theorem prism_faces (edges : ℕ) (h_edges : edges = 18) : ∃ faces : ℕ, faces = 8 :=
by
  -- Define L as the number of lateral faces
  let L := edges / 3
  have h_L : L = 6, from calc
    L = 18 / 3 : by rw [h_edges]
    ... = 6 : by norm_num
  -- Define the total number of faces
  let total_faces := L + 2
  have h_faces : total_faces = 8, from calc
    total_faces = 6 + 2 : by rw [h_L]
    ... = 8 : by norm_num
  use total_faces
  exact h_faces

end prism_faces_l237_237864


namespace proofs_l237_237590

theorem proofs (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : sin α = 4 / 5) (h4 : cos (α + β) = 5 / 13) :
  (cos β = 63 / 65) ∧
  ((sin α ^ 2 + sin (2 * α)) / (cos (2 * α) - 1) = -5 / 4) :=
by
  sorry

end proofs_l237_237590


namespace ascending_order_conversion_l237_237757

def convert_base (num : Nat) (base : Nat) : Nat :=
  match num with
  | 0 => 0
  | _ => (num / 10) * base + (num % 10)

theorem ascending_order_conversion :
  let num16 := 12
  let num7 := 25
  let num4 := 33
  let base16 := 16
  let base7 := 7
  let base4 := 4
  convert_base num4 base4 < convert_base num16 base16 ∧ 
  convert_base num16 base16 < convert_base num7 base7 :=
by
  -- Here would be the proof, but we skip it
  sorry

end ascending_order_conversion_l237_237757


namespace Albert_more_rocks_than_Joshua_l237_237335

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_rocks_than_Joshua_l237_237335


namespace probability_of_rolling_two_exactly_four_times_in_five_rolls_l237_237623

theorem probability_of_rolling_two_exactly_four_times_in_five_rolls :
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n-k)
  probability = (25 / 7776) :=
by
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n - k)
  have h : probability = (25 / 7776) := sorry
  exact h

end probability_of_rolling_two_exactly_four_times_in_five_rolls_l237_237623


namespace remainder_of_k_div_11_l237_237992

theorem remainder_of_k_div_11 {k : ℕ} (hk1 : k % 5 = 2) (hk2 : k % 6 = 5)
  (hk3 : 0 ≤ k % 7 ∧ k % 7 < 7) (hk4 : k < 38) : (k % 11) = 6 := 
by
  sorry

end remainder_of_k_div_11_l237_237992


namespace lcm_1_to_10_l237_237148

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

end lcm_1_to_10_l237_237148


namespace smallest_number_div_by_1_to_10_l237_237160

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l237_237160


namespace left_person_truthful_right_person_lies_l237_237828

theorem left_person_truthful_right_person_lies
  (L R M : Prop)
  (L_truthful_or_false : L ∨ ¬L)
  (R_truthful_or_false : R ∨ ¬R)
  (M_always_answers : M = (L → M) ∨ (¬L → M))
  (left_statement : L → (M = (L → M)))
  (right_statement : R → (M = (¬L → M))) :
  (L ∧ ¬R) ∨ (¬L ∧ R) :=
by
  sorry

end left_person_truthful_right_person_lies_l237_237828


namespace problem_statement_l237_237718

noncomputable def lhs: ℝ := 8^6 * 27^6 * 8^27 * 27^8
noncomputable def rhs: ℝ := 216^14 * 8^19

theorem problem_statement : lhs = rhs :=
by
  sorry

end problem_statement_l237_237718


namespace table_tennis_total_rounds_l237_237523

-- Mathematical equivalent proof problem in Lean 4 statement
theorem table_tennis_total_rounds
  (A_played : ℕ) (B_played : ℕ) (C_referee : ℕ) (total_rounds : ℕ)
  (hA : A_played = 5) (hB : B_played = 4) (hC : C_referee = 2) :
  total_rounds = 7 :=
by
  -- Proof omitted
  sorry

end table_tennis_total_rounds_l237_237523


namespace all_equal_l237_237672

theorem all_equal (xs xsp : Fin 2011 → ℝ) (h : ∀ i : Fin 2011, xs i + xs ((i + 1) % 2011) = 2 * xsp i) (perm : ∃ σ : Fin 2011 ≃ Fin 2011, ∀ i, xsp i = xs (σ i)) :
  ∀ i j : Fin 2011, xs i = xs j := 
sorry

end all_equal_l237_237672


namespace total_students_in_class_l237_237772

theorem total_students_in_class (S R : ℕ)
  (h1 : S = 2 + 12 + 4 + R)
  (h2 : 0 * 2 + 1 * 12 + 2 * 4 + 3 * R = 2 * S) : S = 34 :=
by { sorry }

end total_students_in_class_l237_237772


namespace intersection_M_N_l237_237296

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l237_237296


namespace problem_solution_l237_237632

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set N using the given condition
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Define the complement of N in U
def complement_N : Set ℝ := U \ N

-- Define the intersection of M and the complement of N
def result_set : Set ℝ := M ∩ complement_N

-- Prove the desired result
theorem problem_solution : result_set = {x | -2 ≤ x ∧ x < 0} :=
sorry

end problem_solution_l237_237632


namespace parts_in_batch_l237_237774

theorem parts_in_batch (a : ℕ) (h₁ : 20 * (a / 20) + 13 = a) (h₂ : 27 * (a / 27) + 20 = a) 
  (h₃ : 500 ≤ a) (h₄ : a ≤ 600) : a = 533 :=
by sorry

end parts_in_batch_l237_237774


namespace collinear_vectors_l237_237600

theorem collinear_vectors (m : ℝ) (h_collinear : 1 * m - (-2) * (-3) = 0) : m = 6 :=
by
  sorry

end collinear_vectors_l237_237600


namespace parallelepiped_properties_l237_237452

/--
In an oblique parallelepiped with the following properties:
- The height is 12 dm,
- The projection of the lateral edge on the base plane is 5 dm,
- A cross-section perpendicular to the lateral edge is a rhombus with:
  - An area of 24 dm²,
  - A diagonal of 8 dm,
Prove that:
1. The lateral surface area is 260 dm².
2. The volume is 312 dm³.
-/
theorem parallelepiped_properties
    (height : ℝ)
    (projection_lateral_edge : ℝ)
    (area_rhombus : ℝ)
    (diagonal_rhombus : ℝ)
    (lateral_surface_area : ℝ)
    (volume : ℝ) :
  height = 12 ∧
  projection_lateral_edge = 5 ∧
  area_rhombus = 24 ∧
  diagonal_rhombus = 8 ∧
  lateral_surface_area = 260 ∧
  volume = 312 :=
by
  sorry

end parallelepiped_properties_l237_237452


namespace total_goals_scored_l237_237910

theorem total_goals_scored (g1 t1 g2 t2 : ℕ)
  (h1 : g1 = 2)
  (h2 : g1 = t1 - 3)
  (h3 : t2 = 6)
  (h4 : g2 = t2 - 2) :
  g1 + t1 + g2 + t2 = 17 :=
by
  sorry

end total_goals_scored_l237_237910


namespace evaluate_expression_l237_237210

theorem evaluate_expression (a b c : ℚ) (h1 : c = b - 8) (h2 : b = a + 3) (h3 : a = 2) 
  (h4 : a + 1 ≠ 0) (h5 : b - 3 ≠ 0) (h6 : c + 5 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 7) / (c + 5) = 20 / 3 := by
  sorry

end evaluate_expression_l237_237210


namespace proportion_Q_to_R_l237_237713

theorem proportion_Q_to_R (q r : ℕ) (h1 : 3 * q + 5 * r = 1000) (h2 : 4 * r - 2 * q = 250) : q = r :=
by sorry

end proportion_Q_to_R_l237_237713


namespace prize_behind_door_4_eq_a_l237_237635

theorem prize_behind_door_4_eq_a :
  ∀ (prize : ℕ → ℕ)
    (h_prizes : ∀ i j, 1 ≤ prize i ∧ prize i ≤ 4 ∧ prize i = prize j → i = j)
    (hA1 : prize 1 = 2)
    (hA2 : prize 3 = 3)
    (hB1 : prize 2 = 2)
    (hB2 : prize 3 = 4)
    (hC1 : prize 4 = 2)
    (hC2 : prize 2 = 3)
    (hD1 : prize 4 = 1)
    (hD2 : prize 3 = 3),
    prize 4 = 1 :=
by
  intro prize h_prizes hA1 hA2 hB1 hB2 hC1 hC2 hD1 hD2
  sorry

end prize_behind_door_4_eq_a_l237_237635


namespace arrangement_count_l237_237658

-- Define the sets of books
def italian_books : Finset String := { "I1", "I2", "I3" }
def german_books : Finset String := { "G1", "G2", "G3" }
def french_books : Finset String := { "F1", "F2", "F3", "F4", "F5" }

-- Define the arrangement count as a noncomputable definition, because we are going to use factorial which involves an infinite structure
noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Prove the required arrangement
theorem arrangement_count : 
  (factorial 3) * ((factorial 3) * (factorial 3) * (factorial 5)) = 25920 := 
by
  -- Provide the solution steps here (omitted for now)
  sorry

end arrangement_count_l237_237658


namespace probability_equality_distributions_l237_237789

noncomputable def probability_of_equality 
  (X Y : ℝ → ℝ) -- X and Y are random variables
  (F G : ℝ → ℝ) -- F and G are distribution functions
  (independent : ∀ x, ∃ e, P(X=x, Y=x) = P(X=x)*P(Y=x)) :
  Prop :=
∃ X Y : ℝ → ℝ,
  ∃ F G : ℝ → ℝ,
  (∀ x, ∃ e, P(X = x ∧ Y = x) = P(X = x) * P(Y = x)) →
  (∑ (x : ℤ), (F(x) - F(x - 1)) * (G(x) - G(x - 1))) = P(X = Y)

theorem probability_equality_distributions 
  {X Y : ℝ → ℝ} 
  {F G : ℝ → ℝ}
  (h_independent : ∀ x, ∃ e, P(X=x, Y=x) = P(X=x)*P(Y=x)) :
  probability_of_equality X Y F G h_independent :=
sorry

end probability_equality_distributions_l237_237789


namespace wood_burned_afternoon_l237_237550

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon_l237_237550


namespace smallest_multiple_1_through_10_l237_237136

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l237_237136


namespace train_length_l237_237889

theorem train_length (speed_km_per_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_per_hr = 80) (h_time : time_sec = 9) :
  ∃ length_m : ℕ, length_m = 200 :=
by
  sorry

end train_length_l237_237889


namespace number_of_prime_divisors_420_l237_237314

theorem number_of_prime_divisors_420 : 
  ∃ (count : ℕ), (∀ (p : ℕ), prime p → p ∣ 420 → p ∈ {2, 3, 5, 7}) ∧ count = 4 := 
by
  sorry

end number_of_prime_divisors_420_l237_237314


namespace fixed_point_coordinates_l237_237752

theorem fixed_point_coordinates (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : (2, 2) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^(x-2) + 1)} := 
by
  -- Proof goes here
  sorry

end fixed_point_coordinates_l237_237752


namespace weaving_output_first_day_l237_237941

theorem weaving_output_first_day (x : ℝ) :
  (x + 2*x + 4*x + 8*x + 16*x = 5) → x = 5 / 31 :=
by
  intros h
  sorry

end weaving_output_first_day_l237_237941


namespace one_in_M_l237_237303

def N := { x : ℕ | true } -- Define the natural numbers ℕ

def M : Set ℕ := { x ∈ N | 1 / (x - 2) ≤ 0 }

theorem one_in_M : 1 ∈ M :=
  sorry

end one_in_M_l237_237303


namespace lcm_1_10_l237_237131

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l237_237131


namespace right_triangle_to_acute_triangle_l237_237631

theorem right_triangle_to_acute_triangle 
  (a b c d : ℝ) (h_triangle : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_increase : d > 0):
  (a + d)^2 + (b + d)^2 > (c + d)^2 := 
by {
  sorry
}

end right_triangle_to_acute_triangle_l237_237631


namespace problem_1_problem_2_problem_3_l237_237305

-- Definitions and conditions
def monomial_degree_condition (a : ℝ) : Prop := 2 + (1 + a) = 5

-- Proof goals
theorem problem_1 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = 9 := sorry
theorem problem_2 (a : ℝ) (h : monomial_degree_condition a) : (a + 1) * (a^2 - a + 1) = 9 := sorry
theorem problem_3 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = (a + 1) * (a^2 - a + 1) := sorry

end problem_1_problem_2_problem_3_l237_237305


namespace sum_of_coordinates_l237_237747

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 3) : 
  let x := 2 / 3
  let y := 2 * f (3 * x) + 4
  x + y = 32 / 3 :=
by
  sorry

end sum_of_coordinates_l237_237747


namespace initial_deposit_l237_237390

/-- 
A person deposits some money in a bank at an interest rate of 7% per annum (of the original amount). 
After two years, the total amount in the bank is $6384. Prove that the initial amount deposited is $5600.
-/
theorem initial_deposit (P : ℝ) (h : (P + 0.07 * P) + 0.07 * P = 6384) : P = 5600 :=
by
  sorry

end initial_deposit_l237_237390


namespace probability_of_rolling_two_exactly_four_times_in_five_rolls_l237_237622

theorem probability_of_rolling_two_exactly_four_times_in_five_rolls :
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n-k)
  probability = (25 / 7776) :=
by
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n - k)
  have h : probability = (25 / 7776) := sorry
  exact h

end probability_of_rolling_two_exactly_four_times_in_five_rolls_l237_237622


namespace infinite_x_differs_from_two_kth_powers_l237_237794

theorem infinite_x_differs_from_two_kth_powers (k : ℕ) (h : k > 1) : 
  ∃ (f : ℕ → ℕ), (∀ n, f n = (2^(n+1))^k - (2^n)^k) ∧ (∀ n, ∀ a b : ℕ, ¬ f n = a^k + b^k) :=
sorry

end infinite_x_differs_from_two_kth_powers_l237_237794


namespace _l237_237086

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end _l237_237086


namespace valid_triples_count_l237_237052

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

end valid_triples_count_l237_237052


namespace age_difference_l237_237383

variable (a b c d : ℕ)
variable (h1 : a + b = b + c + 11)
variable (h2 : a + c = c + d + 15)
variable (h3 : b + d = 36)
variable (h4 : a * 2 = 3 * d)

theorem age_difference :
  a - b = 39 :=
by
  sorry

end age_difference_l237_237383


namespace profit_percentage_previous_year_l237_237327

-- Declaring variables
variables (R P : ℝ) -- revenues and profits in the previous year
variable (revenues_1999 := 0.8 * R) -- revenues in 1999
variable (profits_1999 := 0.14 * revenues_1999) -- profits in 1999

-- Given condition: profits in 1999 were 112.00000000000001 percent of the profits in the previous year
axiom profits_ratio : 0.112 * R = 1.1200000000000001 * P

-- Prove the profit as a percentage of revenues in the previous year was 10%
theorem profit_percentage_previous_year : (P / R) * 100 = 10 := by
  sorry

end profit_percentage_previous_year_l237_237327


namespace part1_part2_l237_237066

open Real

-- Definitions used in the proof
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

theorem part1 (x : ℝ) : (p 1 x ∧ q x) → 2 < x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : (¬ (∃ x, p a x) → ¬ (∃ x, q x)) → a > 3 / 2 := by
  sorry

end part1_part2_l237_237066


namespace prism_faces_l237_237859

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end prism_faces_l237_237859


namespace lcm_1_10_l237_237130

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l237_237130


namespace distribution_ways_l237_237716

def count_distributions (n : ℕ) (k : ℕ) : ℕ :=
-- Calculation for count distributions will be implemented here
sorry

theorem distribution_ways (items bags : ℕ) (cond : items = 6 ∧ bags = 3):
  count_distributions items bags = 75 :=
by
  -- Proof would be implemented here
  sorry

end distribution_ways_l237_237716


namespace Hezekiah_age_l237_237967

def Ryanne_age_older_by := 7
def total_age := 15

theorem Hezekiah_age :
  ∃ H : ℕ, H + (H + Ryanne_age_older_by) = total_age ∧ H = 4 :=
begin
  sorry
end

end Hezekiah_age_l237_237967


namespace integer_quotient_is_perfect_square_l237_237461

theorem integer_quotient_is_perfect_square (a b : ℕ) (h : 0 < a ∧ 0 < b) (h_int : (a + b) ^ 2 % (4 * a * b + 1) = 0) :
  ∃ k : ℕ, (a + b) ^ 2 = k ^ 2 * (4 * a * b + 1) := sorry

end integer_quotient_is_perfect_square_l237_237461


namespace smallest_possible_odd_b_l237_237451

theorem smallest_possible_odd_b 
    (a b : ℕ) 
    (h1 : a + b = 90) 
    (h2 : Nat.Prime a) 
    (h3 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ b) 
    (h4 : a > b) 
    (h5 : b % 2 = 1) 
    : b = 85 := 
sorry

end smallest_possible_odd_b_l237_237451


namespace intersection_eq_l237_237602

-- Define Set A based on the given condition
def setA : Set ℝ := {x | 1 < (3:ℝ)^x ∧ (3:ℝ)^x ≤ 9}

-- Define Set B based on the given condition
def setB : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

-- Define the intersection of Set A and Set B
def intersection : Set ℝ := {x | x > 0 ∧ x < 1}

-- Prove that the intersection of setA and setB equals (0, 1)
theorem intersection_eq : {x | x > 0 ∧ x < 1} = {x | x ∈ setA ∧ x ∈ setB} :=
by
  sorry

end intersection_eq_l237_237602


namespace aj_ak_eq_ao_ar_j_is_incenter_l237_237459

open EuclideanGeometry

noncomputable theory

variables {A B C : Point} (is_isosceles : B ≠ C ∧ dist A B = dist A C)
(Gamma : Circle)
(hGamma : circumscribed_triangle Gamma A B C)
(gamma : Circle)
(hgamma : is_inscribed gamma A B C)
(P Q R : Point) (hP : tangent_at gamma A B P) (hQ : tangent_at gamma A C Q) (hR : tangent_at gamma Gamma R)
(O : Point) (hO : center O gamma)
(J : Point) (hJ : midpoint J P Q)
(K : Point) (hK : midpoint K B C)

theorem aj_ak_eq_ao_ar : 
  dist A J / dist A K = dist A O / dist A R :=
sorry

theorem j_is_incenter : 
  is_incenter J A B C :=
sorry

end aj_ak_eq_ao_ar_j_is_incenter_l237_237459


namespace change_calculation_l237_237341

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l237_237341


namespace parabola_x_coordinate_l237_237358

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p, 0)

theorem parabola_x_coordinate
  (M : ℝ × ℝ)
  (h_parabola : (M.2)^2 = 4 * M.1)
  (h_distance : dist M (parabola_focus 2) = 3) :
  M.1 = 1 :=
by
  sorry

end parabola_x_coordinate_l237_237358


namespace product_remainder_l237_237836

-- Define the product of the consecutive numbers
def product := 86 * 87 * 88 * 89 * 90 * 91 * 92

-- Lean statement to state the problem
theorem product_remainder :
  product % 7 = 0 :=
by
  sorry

end product_remainder_l237_237836


namespace M_inter_N_eq_2_4_l237_237237

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l237_237237


namespace prism_faces_l237_237876

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end prism_faces_l237_237876


namespace intersection_of_M_and_N_l237_237254

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l237_237254


namespace vertex_parabola_is_parabola_l237_237054

variables {a c : ℝ} (h_a : 0 < a) (h_c : 0 < c)

theorem vertex_parabola_is_parabola :
  ∀ (x y : ℝ), (∃ b : ℝ, x = -b / (2 * a) ∧ y = a * (-b / (2 * a)) ^ 2 + b * (-b / (2 * a)) + c) ↔ y = -a * x ^ 2 + c :=
by sorry

end vertex_parabola_is_parabola_l237_237054


namespace perfect_cubes_count_l237_237310

def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem perfect_cubes_count : 
  (∀ (n : ℕ), n ≥ 51 ∧ n ≤ 1999 → (is_perfect_cube n) -> 
  (n = 64 ∨ n = 125 ∨ n = 216 ∨ n = 343 ∨ n = 512 ∨ n = 729 ∨ n = 1000 ∨ n = 1331 ∨ n = 1728)) ∧
  (∀ (n : ℕ), n ≥ 50 ∧ n ≤ 2000 → (is_perfect_cube n) -> 
  ((n = 64 ∨ n = 125 ∨ n = 216 ∨ n = 343 ∨ n = 512 ∨ n = 729 ∨ n = 1000 ∨ n = 1331 ∨ n = 1728) -> True)) :=
begin
  sorry
end

end perfect_cubes_count_l237_237310


namespace min_fence_length_l237_237802

theorem min_fence_length (w l F: ℝ) (h1: l = 2 * w) (h2: 2 * w^2 ≥ 500) : F = 96 :=
by sorry

end min_fence_length_l237_237802


namespace jasmine_laps_l237_237048

theorem jasmine_laps (x : ℕ) :
  (∀ (x : ℕ), ∃ (y : ℕ), y = 60 * x) :=
by
  sorry

end jasmine_laps_l237_237048


namespace gcd_sum_lcm_eq_gcd_l237_237806

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end gcd_sum_lcm_eq_gcd_l237_237806


namespace construct_one_degree_l237_237426

theorem construct_one_degree (theta : ℝ) (h : theta = 19) : 1 = 19 * theta - 360 :=
by
  -- Proof here will be filled
  sorry

end construct_one_degree_l237_237426


namespace divide_decimals_l237_237410

theorem divide_decimals : (0.24 / 0.006) = 40 := by
  sorry

end divide_decimals_l237_237410


namespace solve_system_equations_l237_237812

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_system_equations :
  ∃ x y : ℝ, (y = 10^((log10 x)^(log10 x)) ∧ (log10 x)^(log10 (2 * x)) = (log10 y) * 10^((log10 (log10 x))^2))
  → ((x = 10 ∧ y = 10) ∨ (x = 100 ∧ y = 10000)) :=
by
  sorry

end solve_system_equations_l237_237812


namespace rectangular_eq_line_general_eq_curve_min_distance_AB_l237_237596

noncomputable def line_polar_eq := ∀ (ρ θ : ℝ), ρ * cos(θ - π / 4) = 5 + sqrt 2
noncomputable def curve_param_eq := ∀ (α : ℝ), (x, y) = (2 + 2 * cos α, 2 * sin α)

theorem rectangular_eq_line : ∀ x y : ℝ, (x + y = 5 * sqrt 2 + 2) :=
by
  sorry

theorem general_eq_curve : ∀ x y : ℝ, (x^2 + y^2 - 4 * x = 0) :=
by
  sorry

theorem min_distance_AB : ∀ (t α: ℝ), let A := (2 + 2 * cos α, 2 * sin α) in let B := (5 * sqrt 2 + sqrt 2 / 2 * t, 2 - sqrt 2 / 2 * t) in 
  dist A B >= 3 :=
by
  sorry

end rectangular_eq_line_general_eq_curve_min_distance_AB_l237_237596


namespace least_six_digit_cong_3_mod_17_l237_237689

theorem least_six_digit_cong_3_mod_17 :
  ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ x % 17 = 3 ∧ x = 100004 :=
by
  sorry

end least_six_digit_cong_3_mod_17_l237_237689


namespace prism_faces_l237_237863

theorem prism_faces (edges : ℕ) (h_edges : edges = 18) : ∃ faces : ℕ, faces = 8 :=
by
  -- Define L as the number of lateral faces
  let L := edges / 3
  have h_L : L = 6, from calc
    L = 18 / 3 : by rw [h_edges]
    ... = 6 : by norm_num
  -- Define the total number of faces
  let total_faces := L + 2
  have h_faces : total_faces = 8, from calc
    total_faces = 6 + 2 : by rw [h_L]
    ... = 8 : by norm_num
  use total_faces
  exact h_faces

end prism_faces_l237_237863


namespace range_of_x_plus_one_over_x_l237_237318

theorem range_of_x_plus_one_over_x (x : ℝ) (h : x < 0) : x + 1/x ≤ -2 := by
  sorry

end range_of_x_plus_one_over_x_l237_237318


namespace _l237_237085

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end _l237_237085


namespace lcm_1_to_10_l237_237103

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l237_237103


namespace lcm_1_10_l237_237132

theorem lcm_1_10 : Nat.lcm (List.range' 1 10) = 2520 := 
by sorry

end lcm_1_10_l237_237132


namespace sum_of_extrema_l237_237611

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- Main statement to prove
theorem sum_of_extrema :
  let a := -1
  let b := 1
  let f_min := f a
  let f_max := f b
  f_min + f_max = Real.exp 1 + Real.exp (-1) :=
by
  sorry

end sum_of_extrema_l237_237611


namespace income_increase_is_17_percent_l237_237541

def sales_percent_increase (original_items : ℕ) 
                           (original_price : ℝ) 
                           (discount_percent : ℝ) 
                           (sales_increase_percent : ℝ) 
                           (new_items_sold : ℕ) 
                           (new_income : ℝ)
                           (percent_increase : ℝ) : Prop :=
  let original_income := original_items * original_price
  let discounted_price := original_price * (1 - discount_percent / 100)
  let increased_sales := original_items + (original_items * sales_increase_percent / 100)
  original_income = original_items * original_price ∧
  new_income = discounted_price * increased_sales ∧
  new_items_sold = original_items * (1 + sales_increase_percent / 100) ∧
  percent_increase = ((new_income - original_income) / original_income) * 100 ∧
  original_items = 100 ∧ original_price = 1 ∧ discount_percent = 10 ∧ sales_increase_percent = 30 ∧ 
  new_items_sold = 130 ∧ new_income = 117 ∧ percent_increase = 17

theorem income_increase_is_17_percent :
  sales_percent_increase 100 1 10 30 130 117 17 :=
sorry

end income_increase_is_17_percent_l237_237541


namespace sequence_solution_l237_237905

theorem sequence_solution (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → |a n - a m| ≤ (2 * m * n) / (m ^ 2 + n ^ 2)) :
  ∀ (n : ℕ), a n = 1 :=
by
  sorry

end sequence_solution_l237_237905


namespace correct_system_of_equations_l237_237369

theorem correct_system_of_equations : 
  ∃ (x y : ℕ), x + y = 12 ∧ 4 * x + 3 * y = 40 := by
  -- we are stating the existence of x and y that satisfy both equations given as conditions.
  sorry

end correct_system_of_equations_l237_237369


namespace seating_arrangement_correct_l237_237532

noncomputable def seating_arrangements_around_table : Nat :=
  7

def B_G_next_to_C (A B C D E F G : Prop) (d : Nat) : Prop :=
  d = 48

theorem seating_arrangement_correct : ∃ d, d = 48 := sorry

end seating_arrangement_correct_l237_237532


namespace opposite_of_negative_five_l237_237084

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end opposite_of_negative_five_l237_237084


namespace polynomial_evaluation_l237_237378

theorem polynomial_evaluation :
  101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := sorry

end polynomial_evaluation_l237_237378


namespace correct_calculation_l237_237179

theorem correct_calculation (a b : ℝ) : (3 * a * b) ^ 2 = 9 * a ^ 2 * b ^ 2 :=
by
  sorry

end correct_calculation_l237_237179


namespace simplify_and_evaluate_l237_237500

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l237_237500


namespace fraction_product_l237_237567

theorem fraction_product : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end fraction_product_l237_237567


namespace arithmetic_sequence_sum_l237_237023

theorem arithmetic_sequence_sum (a₁ d S : ℤ)
  (ha : 10 * a₁ + 24 * d = 37) :
  19 * (a₁ + 2 * d) + (a₁ + 10 * d) = 74 :=
by
  sorry

end arithmetic_sequence_sum_l237_237023


namespace smallest_divisible_1_to_10_l237_237108

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l237_237108


namespace remainder_not_power_of_4_l237_237204

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end remainder_not_power_of_4_l237_237204


namespace y_intercept_of_line_b_l237_237797

theorem y_intercept_of_line_b
  (m : ℝ) (c₁ : ℝ) (c₂ : ℝ) (x₁ : ℝ) (y₁ : ℝ)
  (h_parallel : m = 3/2)
  (h_point : (4, 2) ∈ { p : ℝ × ℝ | p.2 = m * p.1 + c₂ }) :
  c₂ = -4 := by
  sorry

end y_intercept_of_line_b_l237_237797


namespace intersection_M_N_l237_237245

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237245


namespace find_a_l237_237025

theorem find_a (a x : ℝ) (h : x = 1) (h_eq : 2 - 3 * (a + x) = 2 * x) : a = -1 := by
  sorry

end find_a_l237_237025


namespace probability_not_passing_l237_237981

theorem probability_not_passing (P_passing : ℚ) (h : P_passing = 4/7) : (1 - P_passing = 3/7) :=
by
  rw [h]
  norm_num

end probability_not_passing_l237_237981


namespace intersection_of_M_and_N_l237_237255

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l237_237255


namespace A_subset_B_l237_237949

def inA (n : ℕ) : Prop := ∃ x y : ℕ, n = x^2 + 2 * y^2 ∧ x > y
def inB (n : ℕ) : Prop := ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ n = (a^3 + b^3 + c^3) / (a + b + c)

theorem A_subset_B : ∀ (n : ℕ), inA n → inB n := 
sorry

end A_subset_B_l237_237949


namespace intersection_M_N_l237_237267

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237267


namespace find_x_plus_y_l237_237438

variable (x y : ℝ)

theorem find_x_plus_y (h1 : |x| + x + y = 8) (h2 : x + |y| - y = 10) : x + y = 14 / 5 := 
by
  sorry

end find_x_plus_y_l237_237438


namespace shooter_hits_at_least_3_times_l237_237026

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

end shooter_hits_at_least_3_times_l237_237026


namespace total_games_in_season_is_correct_l237_237827

-- Definitions based on given conditions
def games_per_month : ℕ := 7
def season_months : ℕ := 2

-- The theorem to prove
theorem total_games_in_season_is_correct : 
  (games_per_month * season_months = 14) :=
by
  sorry

end total_games_in_season_is_correct_l237_237827


namespace square_divisibility_l237_237528

theorem square_divisibility (n : ℤ) : n^2 % 4 = 0 ∨ n^2 % 4 = 1 := sorry

end square_divisibility_l237_237528


namespace prism_faces_l237_237867

-- Define variables for number of edges E, lateral faces L, and total number of faces F
variables (E : ℕ := 18) (L : ℕ) (F : ℕ)

-- The relationship between edges and lateral faces for a prism
lemma prism_lateral_faces (hE : E = 18) : 3 * L = E :=
sorry

-- Calculate the number of lateral faces
lemma solve_lateral_faces (hE : E = 18) : L = 6 :=
begin
  have h : 3 * L = E := prism_lateral_faces hE,
  rw hE at h,
  exact (nat.mul_left_inj 3 (nat.zero_lt_succ 2)).mp h,
end

-- Prove the total number of faces
theorem prism_faces (hE : E = 18) : F = 8 :=
begin
  have hL : L = 6 := solve_lateral_faces hE,
  exact calc
    F = L + 2 : by rw [hL]
       ... = 6 + 2 : rfl
       ... = 8 : rfl,
end

end prism_faces_l237_237867


namespace smallest_number_divisible_by_1_to_10_l237_237128

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237128


namespace madeline_money_l237_237648

variable (M B : ℝ)

theorem madeline_money :
  B = 1/2 * M →
  M + B = 72 →
  M = 48 :=
  by
    intros h1 h2
    sorry

end madeline_money_l237_237648


namespace median_possible_values_l237_237455

variable {ι : Type} -- Representing the set S as a type
variable (S : Finset ℤ) -- S is a finite set of integers

def conditions (S: Finset ℤ) : Prop :=
  S.card = 9 ∧
  {5, 7, 10, 13, 17, 21} ⊆ S

theorem median_possible_values :
  ∀ S : Finset ℤ, conditions S → ∃ medians : Finset ℤ, medians.card = 7 :=
by
  sorry

end median_possible_values_l237_237455


namespace gina_tom_goals_l237_237913

theorem gina_tom_goals :
  let g_day1 := 2
  let t_day1 := g_day1 + 3
  let t_day2 := 6
  let g_day2 := t_day2 - 2
  let g_total := g_day1 + g_day2
  let t_total := t_day1 + t_day2
  g_total + t_total = 17 := by
  sorry

end gina_tom_goals_l237_237913


namespace min_a5_of_geom_seq_l237_237321

-- Definition of geometric sequence positivity and difference condition.
def geom_seq_pos_diff (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 3 - a 1 = 2)

-- The main theorem stating that the minimum value of a_5 is 8.
theorem min_a5_of_geom_seq {a : ℕ → ℝ} {q : ℝ} (h : geom_seq_pos_diff a q) :
  a 5 ≥ 8 :=
sorry

end min_a5_of_geom_seq_l237_237321


namespace intersection_M_N_l237_237243

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237243


namespace lcm_1_to_10_l237_237102

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  -- Proof can be provided here
  sorry

end lcm_1_to_10_l237_237102


namespace fraction_subtraction_l237_237569

theorem fraction_subtraction (a : ℝ) (h : a ≠ 0) : 1 / a - 3 / a = -2 / a := 
by
  sorry

end fraction_subtraction_l237_237569


namespace solve_equation_l237_237355

theorem solve_equation (x : ℝ) : (⌊Real.sin x⌋:ℝ)^2 = Real.cos x ^ 2 - 1 ↔ ∃ n : ℤ, x = n * Real.pi := by
  sorry

end solve_equation_l237_237355


namespace trig_identity_l237_237428

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) : 
  2 * Real.sin x + 3 * Real.cos x = 1 ∨ 2 * Real.sin x + 3 * Real.cos x = 3 :=
sorry

end trig_identity_l237_237428


namespace intersection_M_N_l237_237272

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237272


namespace power_of_negative_fraction_l237_237896

theorem power_of_negative_fraction :
  (- (1/3))^2 = 1/9 := 
by 
  sorry

end power_of_negative_fraction_l237_237896


namespace area_inequality_l237_237661

variable {a b c : ℝ} (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a)

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) : ℝ :=
  let p := semiperimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem area_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  (2 * (area a b c h))^3 < (a * b * c)^2 := sorry

end area_inequality_l237_237661


namespace find_x_plus_y_l237_237763

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end find_x_plus_y_l237_237763


namespace oleg_max_composite_numbers_l237_237466

theorem oleg_max_composite_numbers : 
  ∃ (S : Finset ℕ), 
    (∀ (n ∈ S), n < 1500 ∧ ∃ p q, prime p ∧ prime q ∧ p ≠ q ∧ p * q = n) ∧ 
    (∀ (a b ∈ S), a ≠ b → gcd a b = 1) ∧ 
    S.card = 12 :=
sorry

end oleg_max_composite_numbers_l237_237466


namespace is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l237_237916

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Given that a, b, c are the sides of the triangle
axiom lengths_of_triangle : a > 0 ∧ b > 0 ∧ c > 0

-- Problem 1: Prove that triangle is isosceles if x=1 is a root
theorem is_isosceles_of_x_eq_one_root  : ((a - c) * (1:ℝ)^2 - 2 * b * (1:ℝ) + (a + c) = 0) → a = b ∧ a ≠ c := 
by
  intros h
  sorry

-- Problem 2: Prove that triangle is right-angled if the equation has two equal real roots
theorem is_right_angled_of_equal_roots : (b^2 = a^2 - c^2) → (a^2 = b^2 + c^2) := 
by 
  intros h
  sorry

end is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l237_237916


namespace prism_faces_l237_237865

-- Define variables for number of edges E, lateral faces L, and total number of faces F
variables (E : ℕ := 18) (L : ℕ) (F : ℕ)

-- The relationship between edges and lateral faces for a prism
lemma prism_lateral_faces (hE : E = 18) : 3 * L = E :=
sorry

-- Calculate the number of lateral faces
lemma solve_lateral_faces (hE : E = 18) : L = 6 :=
begin
  have h : 3 * L = E := prism_lateral_faces hE,
  rw hE at h,
  exact (nat.mul_left_inj 3 (nat.zero_lt_succ 2)).mp h,
end

-- Prove the total number of faces
theorem prism_faces (hE : E = 18) : F = 8 :=
begin
  have hL : L = 6 := solve_lateral_faces hE,
  exact calc
    F = L + 2 : by rw [hL]
       ... = 6 + 2 : rfl
       ... = 8 : rfl,
end

end prism_faces_l237_237865


namespace correct_operation_l237_237525

theorem correct_operation :
  (∀ (a : ℤ), 3 * a + 2 * a ≠ 5 * a ^ 2) ∧
  (∀ (a : ℤ), a ^ 6 / a ^ 2 ≠ a ^ 3) ∧
  (∀ (a : ℤ), (-3 * a ^ 3) ^ 2 = 9 * a ^ 6) ∧
  (∀ (a : ℤ), (a + 2) ^ 2 ≠ a ^ 2 + 4) := 
by
  sorry

end correct_operation_l237_237525


namespace intersection_M_N_l237_237265

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237265


namespace complete_graph_k17_has_monochromatic_triangle_l237_237578

open SimpleGraph

theorem complete_graph_k17_has_monochromatic_triangle (C : SimpleGraph (Fin 17)) 
  [CompleteGraph 17 C] (f : C.Edge → Fin 3) : 
  ∃ (u v w : Fin 17), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ f ⟨u, v⟩ = f ⟨v, w⟩ ∧ f ⟨v, w⟩ = f ⟨w, u⟩ := 
by
  sorry

end complete_graph_k17_has_monochromatic_triangle_l237_237578


namespace average_of_last_three_numbers_l237_237819

theorem average_of_last_three_numbers (A B C D E F : ℕ) 
  (h1 : (A + B + C + D + E + F) / 6 = 30)
  (h2 : (A + B + C + D) / 4 = 25)
  (h3 : D = 25) :
  (D + E + F) / 3 = 35 :=
by
  sorry

end average_of_last_three_numbers_l237_237819


namespace distance_on_dirt_section_distance_on_mud_section_l237_237882

noncomputable def v_highway : ℝ := 120 -- km/h
noncomputable def v_dirt : ℝ := 40 -- km/h
noncomputable def v_mud : ℝ := 10 -- km/h
noncomputable def initial_distance : ℝ := 0.6 -- km

theorem distance_on_dirt_section : 
  ∃ s_1 : ℝ, 
  (s_1 = 0.2 * 1000 ∧ -- converting km to meters
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

theorem distance_on_mud_section : 
  ∃ s_2 : ℝ, 
  (s_2 = 50 ∧
  v_highway = 120 ∧ 
  v_dirt = 40 ∧ 
  v_mud = 10 ∧ 
  initial_distance = 0.6 ) :=
sorry

end distance_on_dirt_section_distance_on_mud_section_l237_237882


namespace solution_set_of_inequality_l237_237514

theorem solution_set_of_inequality (x : ℝ) :
  2 * x ≤ -1 → x > -1 → -1 < x ∧ x ≤ -1 / 2 :=
by
  intro h1 h2
  have h3 : x ≤ -1 / 2 := by linarith
  exact ⟨h2, h3⟩

end solution_set_of_inequality_l237_237514


namespace largest_three_digit_number_divisible_by_8_l237_237688

-- Define the properties of a number being a three-digit number
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of a number being divisible by 8
def isDivisibleBy8 (n : ℕ) : Prop := n % 8 = 0

-- The theorem we want to prove: the largest three-digit number divisible by 8 is 992
theorem largest_three_digit_number_divisible_by_8 : ∃ n, isThreeDigitNumber n ∧ isDivisibleBy8 n ∧ (∀ m, isThreeDigitNumber m ∧ isDivisibleBy8 m → m ≤ 992) :=
  sorry

end largest_three_digit_number_divisible_by_8_l237_237688


namespace find_natural_number_l237_237419

-- Define the problem statement
def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ (2 * n^2 - 2) = k * (n^3 - n)

-- The main theorem
theorem find_natural_number (n : ℕ) : satisfies_condition n ↔ n = 2 :=
sorry

end find_natural_number_l237_237419


namespace max_not_divisible_by_3_l237_237563

theorem max_not_divisible_by_3 (s : Finset ℕ) (h₁ : s.card = 7) (h₂ : ∃ p ∈ s, p % 3 = 0) : 
  ∃t : Finset ℕ, t.card = 6 ∧ (∀ x ∈ t, x % 3 ≠ 0) ∧ (t ⊆ s) :=
sorry

end max_not_divisible_by_3_l237_237563


namespace extreme_point_condition_l237_237751

variable {R : Type*} [OrderedRing R]

def f (x a b : R) : R := x^3 - a*x - b

theorem extreme_point_condition (a b x0 x1 : R) (h₁ : ∀ x : R, f x a b = x^3 - a*x - b)
  (h₂ : f x0 a b = x0^3 - a*x0 - b)
  (h₃ : f x1 a b = x1^3 - a*x1 - b)
  (has_extreme : ∃ x0 : R, 3*x0^2 = a) 
  (hx1_extreme : f x1 a b = f x0 a b) 
  (hx1_x0_diff : x1 ≠ x0) :
  x1 + 2*x0 = 0 :=
by
  sorry

end extreme_point_condition_l237_237751


namespace increasing_interval_of_f_maximum_value_of_f_l237_237080

open Real

def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Consider x in the interval [-2, 4]
def domain_x (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

theorem increasing_interval_of_f :
  ∃a b : ℝ, (a, b) = (1, 4) ∧ ∀ x y : ℝ, domain_x x → domain_x y → a ≤ x → x < y → y ≤ b → f x < f y := sorry

theorem maximum_value_of_f :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, domain_x x → f x ≤ M := sorry

end increasing_interval_of_f_maximum_value_of_f_l237_237080


namespace total_cost_price_l237_237850

theorem total_cost_price (C O B : ℝ) 
    (hC : 1.25 * C = 8340) 
    (hO : 1.30 * O = 4675) 
    (hB : 1.20 * B = 3600) : 
    C + O + B = 13268.15 := 
by 
    sorry

end total_cost_price_l237_237850


namespace area_difference_l237_237930

theorem area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15 / 2) :
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end area_difference_l237_237930


namespace polygon_sides_l237_237768

theorem polygon_sides :
  ∀ (n : ℕ), (n > 2) → (n - 2) * 180 < 360 → n = 3 :=
by
  intros n hn1 hn2
  sorry

end polygon_sides_l237_237768


namespace f_of_g_of_2_l237_237791

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end f_of_g_of_2_l237_237791


namespace prism_faces_l237_237866

-- Define variables for number of edges E, lateral faces L, and total number of faces F
variables (E : ℕ := 18) (L : ℕ) (F : ℕ)

-- The relationship between edges and lateral faces for a prism
lemma prism_lateral_faces (hE : E = 18) : 3 * L = E :=
sorry

-- Calculate the number of lateral faces
lemma solve_lateral_faces (hE : E = 18) : L = 6 :=
begin
  have h : 3 * L = E := prism_lateral_faces hE,
  rw hE at h,
  exact (nat.mul_left_inj 3 (nat.zero_lt_succ 2)).mp h,
end

-- Prove the total number of faces
theorem prism_faces (hE : E = 18) : F = 8 :=
begin
  have hL : L = 6 := solve_lateral_faces hE,
  exact calc
    F = L + 2 : by rw [hL]
       ... = 6 + 2 : rfl
       ... = 8 : rfl,
end

end prism_faces_l237_237866


namespace prism_faces_l237_237857

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end prism_faces_l237_237857


namespace prism_faces_l237_237879

-- Define basic properties and conditions
def is_prism_with_L_gon_base (edges : ℕ) (L : ℕ) :=
  edges = 3 * L

noncomputable def number_of_faces (L : ℕ) : ℕ :=
  L + 2  -- L lateral faces and 2 bases

-- Main statement
theorem prism_faces (edges : ℕ) (L : ℕ) (h : edges = 3 * L) : number_of_faces L = 8 :=
by
  -- Instantiate the given condition
  have hL : L = 6 := sorry
  -- Prove the number of faces calculation
  have h_faces : number_of_faces L = number_of_faces 6 := by rw hL
  have h_faces_calc : number_of_faces 6 = 8 := sorry
  exact (h_faces.trans h_faces_calc)

end prism_faces_l237_237879


namespace john_total_hours_l237_237829

def wall_area (length : ℕ) (width : ℕ) := length * width

def total_area (num_walls : ℕ) (wall_area : ℕ) := num_walls * wall_area

def time_to_paint (area : ℕ) (time_per_square_meter : ℕ) := area * time_per_square_meter

def hours_to_minutes (hours : ℕ) := hours * 60

def total_hours (painting_time : ℕ) (spare_time : ℕ) := painting_time + spare_time

theorem john_total_hours 
  (length width num_walls time_per_square_meter spare_hours : ℕ) 
  (H_length : length = 2) 
  (H_width : width = 3) 
  (H_num_walls : num_walls = 5)
  (H_time_per_square_meter : time_per_square_meter = 10)
  (H_spare_hours : spare_hours = 5) :
  total_hours (time_to_paint (total_area num_walls (wall_area length width)) time_per_square_meter / hours_to_minutes 1) spare_hours = 10 := 
by 
    rw [H_length, H_width, H_num_walls, H_time_per_square_meter, H_spare_hours]
    sorry

end john_total_hours_l237_237829


namespace factorization_correct_l237_237560

theorem factorization_correct :
  (∀ x : ℝ, x^2 - 6*x + 9 = (x - 3)^2) :=
by
  sorry

end factorization_correct_l237_237560


namespace sin_15_add_sin_75_l237_237677

theorem sin_15_add_sin_75 : 
  Real.sin (15 * Real.pi / 180) + Real.sin (75 * Real.pi / 180) = Real.sqrt 6 / 2 :=
by
  sorry

end sin_15_add_sin_75_l237_237677


namespace leaks_empty_time_l237_237474

theorem leaks_empty_time (A L1 L2: ℝ) (hA: A = 1/2) (hL1_rate: A - L1 = 1/3) 
  (hL2_rate: A - L1 - L2 = 1/4) : 1 / (L1 + L2) = 4 :=
by
  sorry

end leaks_empty_time_l237_237474


namespace ott_fraction_part_l237_237798

noncomputable def fractional_part_of_group_money (x : ℝ) (M L N P : ℝ) :=
  let total_initial := M + L + N + P + 2
  let money_received_by_ott := 4 * x
  let ott_final_money := 2 + money_received_by_ott
  let total_final := total_initial + money_received_by_ott
  (ott_final_money / total_final) = (3 / 14)

theorem ott_fraction_part (x : ℝ) (M L N P : ℝ)
    (hM : M = 6 * x) (hL : L = 5 * x) (hN : N = 4 * x) (hP : P = 7 * x) :
    fractional_part_of_group_money x M L N P :=
by
  sorry

end ott_fraction_part_l237_237798


namespace jiujiang_liansheng_sampling_l237_237043

def bag_numbers : List ℕ := [7, 17, 27, 37, 47]

def systematic_sampling (N n : ℕ) (selected_bags : List ℕ) : Prop :=
  ∃ k i, k = N / n ∧ ∀ j, j < List.length selected_bags → selected_bags.get? j = some (i + k * j)

theorem jiujiang_liansheng_sampling :
  systematic_sampling 50 5 bag_numbers :=
by
  sorry

end jiujiang_liansheng_sampling_l237_237043


namespace wendy_baked_29_cookies_l237_237374

variables (cupcakes : ℕ) (pastries_taken_home : ℕ) (pastries_sold : ℕ)

def total_initial_pastries (cupcakes pastries_taken_home pastries_sold : ℕ) : ℕ :=
  pastries_taken_home + pastries_sold

def cookies_baked (total_initial_pastries cupcakes : ℕ) : ℕ :=
  total_initial_pastries - cupcakes

theorem wendy_baked_29_cookies :
  cupcakes = 4 →
  pastries_taken_home = 24 →
  pastries_sold = 9 →
  cookies_baked (total_initial_pastries cupcakes pastries_taken_home pastries_sold) cupcakes = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end wendy_baked_29_cookies_l237_237374


namespace inequality_solution_set_l237_237984

open Real

-- Define the inequality condition and the proof statement
theorem inequality_solution_set (x : ℝ) :
  (4 ^ x - 3 * 2 ^ (x + 1) - 16 > 0) ↔ (x > 3) :=
by
  sorry

end inequality_solution_set_l237_237984


namespace purple_marbles_probability_l237_237891

noncomputable def purple_probability (n k : ℕ) (p q : ℚ) : ℚ :=
(prod (finset.range k) (λ _, p) * prod (finset.range (n - k)) (λ _, q)) * (nat.choose n k)

def total_probability : ℚ :=
(purple_probability 5 3 (7/12) (5/12)) +
(purple_probability 5 4 (7/12) (5/12)) +
(purple_probability 5 5 (7/12) (5/12))

theorem purple_marbles_probability :
  total_probability ≈ 0.054 :=
sorry

end purple_marbles_probability_l237_237891


namespace tan_ratio_l237_237056

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end tan_ratio_l237_237056


namespace find_total_worth_of_stock_l237_237527

theorem find_total_worth_of_stock (X : ℝ)
  (h1 : 0.20 * X * 0.10 = 0.02 * X)
  (h2 : 0.80 * X * 0.05 = 0.04 * X)
  (h3 : 0.04 * X - 0.02 * X = 200) :
  X = 10000 :=
sorry

end find_total_worth_of_stock_l237_237527


namespace smallest_number_divisible_by_1_through_10_l237_237157

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l237_237157


namespace prism_faces_l237_237858

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end prism_faces_l237_237858


namespace find_y_l237_237345

theorem find_y (a b y : ℝ) (h1 : s = (3 * a) ^ (2 * b)) (h2 : s = 5 * (a ^ b) * (y ^ b))
  (h3 : 0 < a) (h4 : 0 < b) : 
  y = 9 * a / 5 := by
  sorry

end find_y_l237_237345


namespace largest_of_eight_consecutive_integers_l237_237985

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h : 8 * n + 28 = 3652) : n + 7 = 460 := by 
  sorry

end largest_of_eight_consecutive_integers_l237_237985


namespace pos_diff_of_solutions_abs_eq_20_l237_237834

theorem pos_diff_of_solutions_abs_eq_20 : ∀ (x1 x2 : ℝ), (|x1 + 5| = 20 ∧ |x2 + 5| = 20) → x1 - x2 = 40 :=
  by
    intros x1 x2 h
    sorry

end pos_diff_of_solutions_abs_eq_20_l237_237834


namespace Lilith_caps_collection_l237_237957

theorem Lilith_caps_collection
  (caps_per_month_first_year : ℕ)
  (caps_per_month_after_first_year : ℕ)
  (caps_received_each_christmas : ℕ)
  (caps_lost_per_year : ℕ)
  (total_caps_collected : ℕ)
  (first_year_caps : ℕ := caps_per_month_first_year * 12)
  (years_after_first_year : ℕ)
  (total_years : ℕ := years_after_first_year + 1)
  (caps_collected_after_first_year : ℕ := caps_per_month_after_first_year * 12 * years_after_first_year)
  (caps_received_total : ℕ := caps_received_each_christmas * total_years)
  (caps_lost_total : ℕ := caps_lost_per_year * total_years)
  (total_calculated_caps : ℕ := first_year_caps + caps_collected_after_first_year + caps_received_total - caps_lost_total) :
  total_caps_collected = 401 → total_years = 5 :=
by
  sorry

end Lilith_caps_collection_l237_237957


namespace intersection_eq_0_l237_237641

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_eq_0 : M ∩ N = {0} := by
  sorry

end intersection_eq_0_l237_237641


namespace rectangle_vertices_complex_plane_l237_237004

theorem rectangle_vertices_complex_plane (b : ℝ) :
  (∀ (z : ℂ), z^4 - 10*z^3 + (16*b : ℂ)*z^2 - 2*(3*b^2 - 5*b + 4 : ℂ)*z + 6 = 0 →
    (∃ (w₁ w₂ : ℂ), z = w₁ ∨ z = w₂)) →
  (b = 5 / 3 ∨ b = 2) :=
sorry

end rectangle_vertices_complex_plane_l237_237004


namespace student_community_arrangement_l237_237538

theorem student_community_arrangement :
  let students := 4
  let communities := 3
  (students.choose 2) * (communities.factorial / (communities - (students - 1)).factorial) = 36 :=
by
  have students := 4
  have communities := 3
  sorry

end student_community_arrangement_l237_237538


namespace new_paint_intensity_l237_237813

def red_paint_intensity (initial_intensity replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

theorem new_paint_intensity :
  red_paint_intensity 0.1 0.2 0.5 = 0.15 :=
by sorry

end new_paint_intensity_l237_237813


namespace exponential_monotonicity_l237_237425

theorem exponential_monotonicity {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b :=
by 
  sorry 

end exponential_monotonicity_l237_237425


namespace convert_spherical_coords_l237_237940

theorem convert_spherical_coords (ρ θ φ : ℝ) (hρ : ρ > 0) (hθ : 0 ≤ θ ∧ θ < 2 * π) (hφ : 0 ≤ φ ∧ φ ≤ π) :
  (ρ = 4 ∧ θ = 4 * π / 3 ∧ φ = π / 4) ↔ (ρ, θ, φ) = (4, 4 * π / 3, π / 4) :=
by { sorry }

end convert_spherical_coords_l237_237940


namespace smallest_divisible_1_to_10_l237_237107

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l237_237107


namespace brown_eyed_brunettes_count_l237_237008

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

end brown_eyed_brunettes_count_l237_237008


namespace lcm_1_to_10_l237_237166

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l237_237166


namespace triangle_inequality_proof_l237_237346

theorem triangle_inequality_proof (a b c : ℝ) (PA QA PB QB PC QC : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hpa : PA ≥ 0) (hqa : QA ≥ 0) (hpb : PB ≥ 0) (hqb : QB ≥ 0) 
  (hpc : PC ≥ 0) (hqc : QC ≥ 0):
  a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c := 
sorry

end triangle_inequality_proof_l237_237346


namespace line_sum_slope_intercept_l237_237389

theorem line_sum_slope_intercept (m b : ℝ) (x y : ℝ)
  (hm : m = 3)
  (hpoint : (x, y) = (-2, 4))
  (heq : y = m * x + b) :
  m + b = 13 :=
by
  sorry

end line_sum_slope_intercept_l237_237389


namespace sqrt_diff_of_squares_l237_237406

theorem sqrt_diff_of_squares : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end sqrt_diff_of_squares_l237_237406


namespace change_calculation_l237_237343

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l237_237343


namespace garrison_provisions_last_initially_l237_237190

noncomputable def garrison_initial_provisions (x : ℕ) : Prop :=
  ∃ x : ℕ, 2000 * (x - 21) = 3300 * 20 ∧ x = 54

theorem garrison_provisions_last_initially :
  garrison_initial_provisions 54 :=
by
  sorry

end garrison_provisions_last_initially_l237_237190


namespace files_remaining_on_flash_drive_l237_237832

def initial_music_files : ℕ := 32
def initial_video_files : ℕ := 96
def deleted_files : ℕ := 60

def total_initial_files : ℕ := initial_music_files + initial_video_files

theorem files_remaining_on_flash_drive 
  (h : total_initial_files = 128) : (total_initial_files - deleted_files) = 68 := by
  sorry

end files_remaining_on_flash_drive_l237_237832


namespace natalia_crates_l237_237734

/- The definitions from the conditions -/
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

/- The proposition to prove -/
theorem natalia_crates : (novels + comics + documentaries + albums) / crate_capacity = 116 := by
  -- this skips the proof and assumes the theorem is true
  sorry

end natalia_crates_l237_237734


namespace at_least_4_stayed_l237_237323

-- We define the number of people and their respective probabilities of staying.
def numPeople : ℕ := 8
def numCertain : ℕ := 5
def numUncertain : ℕ := 3
def probUncertainStay : ℚ := 1 / 3

-- We state the problem formally:
theorem at_least_4_stayed :
  (probUncertainStay ^ 3 * 3 + (probUncertainStay ^ 2 * (2 / 3) * 3) + (probUncertainStay * (2 / 3)^2 * 3)) = 19 / 27 :=
by
  sorry

end at_least_4_stayed_l237_237323


namespace smallest_number_divisible_by_1_to_10_l237_237126

theorem smallest_number_divisible_by_1_to_10 : ∃ n : ℕ, (∀ i ∈ (finset.range 11 \ finset.singleton 0), i ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_to_10_l237_237126


namespace intersection_M_N_l237_237264

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237264


namespace num_non_fiction_books_l237_237185

-- Definitions based on the problem conditions
def num_fiction_configurations : ℕ := 24
def total_configurations : ℕ := 36

-- Non-computable definition for factorial
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

-- Theorem to prove the number of new non-fiction books
theorem num_non_fiction_books (n : ℕ) :
  num_fiction_configurations * factorial n = total_configurations → n = 2 :=
by
  sorry

end num_non_fiction_books_l237_237185


namespace lcm_1_to_10_l237_237167

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l237_237167


namespace not_sunny_prob_l237_237087

theorem not_sunny_prob (P_sunny : ℚ) (h : P_sunny = 5/7) : 1 - P_sunny = 2/7 :=
by sorry

end not_sunny_prob_l237_237087


namespace james_total_vegetables_l237_237638

def james_vegetable_count (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

theorem james_total_vegetables 
    (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
    a = 22 → b = 18 → c = 15 → d = 10 → e = 12 →
    james_vegetable_count a b c d e = 77 :=
by
  intros ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end james_total_vegetables_l237_237638


namespace joan_mortgage_payoff_l237_237842

/-- Joan's mortgage problem statement. -/
theorem joan_mortgage_payoff (a r : ℕ) (total : ℕ) (n : ℕ) : a = 100 → r = 3 → total = 12100 → 
    total = a * (1 - r^n) / (1 - r) → n = 5 :=
by intros ha hr htotal hgeom; sorry

end joan_mortgage_payoff_l237_237842


namespace intersection_M_N_l237_237290

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237290


namespace kolya_or_leva_l237_237337

theorem kolya_or_leva (k l : ℝ) (hkl : k > 0) (hll : l > 0) : 
  (k > l → ∃ a b c : ℝ, a = l + (2 / 3) * (k - l) ∧ b = (1 / 6) * (k - l) ∧ c = (1 / 6) * (k - l) ∧ a > b + c + l ∧ ¬(a < b + c + a)) ∨ 
  (k ≤ l → ∃ k1 k2 k3 : ℝ, k1 ≥ k2 ∧ k2 ≥ k3 ∧ k = k1 + k2 + k3 ∧ ∃ a' b' c' : ℝ, a' = k1 ∧ b' = (l - k1) / 2 ∧ c' = (l - k1) / 2 ∧ a' + a' > k2 ∧ b' + b' > k3) :=
by sorry

end kolya_or_leva_l237_237337


namespace const_sequence_l237_237845

theorem const_sequence (x y : ℝ) (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∀ n, a n - a (n + 1) = (a n ^ 2 - 1) / (a n + a (n - 1)))
  (h2 : ∀ n, a n = a (n + 1) → a n ^ 2 = 1 ∧ a n ≠ -a (n - 1))
  (h_init : a 1 = y ∧ a 0 = x)
  (hx : |x| = 1 ∧ y ≠ -x) :
  (∃ n0, ∀ n ≥ n0, a n = 1 ∨ a n = -1) := sorry

end const_sequence_l237_237845


namespace intersection_M_N_l237_237273

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237273


namespace intersection_M_N_l237_237283

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l237_237283


namespace johns_piano_total_cost_l237_237639

theorem johns_piano_total_cost : 
  let piano_cost := 500
  let original_lessons_cost := 20 * 40
  let discount := (25 / 100) * original_lessons_cost
  let discounted_lessons_cost := original_lessons_cost - discount
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  total_cost = 1275 := 
by
  let piano_cost := 500
  let original_lessons_cost := 800
  let discount := 200
  let discounted_lessons_cost := 600
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  -- Proof skipped
  sorry

end johns_piano_total_cost_l237_237639


namespace max_oleg_composite_numbers_l237_237468

/-- Oleg's list of composite numbers less than 1500 and pairwise coprime. --/
def oleg_composite_numbers (numbers : List ℕ) : Prop :=
  ∀ n ∈ numbers, Nat.isComposite n ∧ n < 1500 ∧ (∀ m ∈ numbers, n ≠ m → Nat.gcd n m = 1)

/-- Maximum number of such numbers that Oleg could write. --/
theorem max_oleg_composite_numbers : ∃ numbers : List ℕ, oleg_composite_numbers numbers ∧ numbers.length = 12 := 
sorry

end max_oleg_composite_numbers_l237_237468


namespace min_employees_needed_l237_237546

theorem min_employees_needed (forest_jobs : ℕ) (marine_jobs : ℕ) (both_jobs : ℕ)
    (h1 : forest_jobs = 95) (h2 : marine_jobs = 80) (h3 : both_jobs = 35) :
    (forest_jobs - both_jobs) + (marine_jobs - both_jobs) + both_jobs = 140 :=
by
  sorry

end min_employees_needed_l237_237546


namespace calculate_difference_of_squares_l237_237201

theorem calculate_difference_of_squares : (640^2 - 360^2) = 280000 := by
  sorry

end calculate_difference_of_squares_l237_237201


namespace number_of_ways_l237_237518

theorem number_of_ways (h_walk : ℕ) (h_drive : ℕ) (h_eq1 : h_walk = 3) (h_eq2 : h_drive = 4) : h_walk + h_drive = 7 :=
by 
  sorry

end number_of_ways_l237_237518


namespace smallest_number_divisible_1_to_10_l237_237176

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237176


namespace smallest_number_div_by_1_to_10_l237_237163

theorem smallest_number_div_by_1_to_10: 
  LCM [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 2520 :=
by
  sorry

end smallest_number_div_by_1_to_10_l237_237163


namespace count_divisors_multiple_of_5_l237_237307

-- Define the conditions as Lean definitions
def prime_factorization (n : ℕ) := 
  n = 2^2 * 3^3 * 5^2

def is_divisor (d : ℕ) (n : ℕ) :=
  d ∣ n

def is_multiple_of_5 (d : ℕ) :=
  ∃ a b c, d = 2^a * 3^b * 5^c ∧ 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2

-- The theorem to be proven
theorem count_divisors_multiple_of_5 (h: prime_factorization 5400) : 
  {d : ℕ | is_divisor d 5400 ∧ is_multiple_of_5 d}.to_finset.card = 24 :=
by {
  sorry -- Proof goes here
}

end count_divisors_multiple_of_5_l237_237307


namespace tiling_possible_with_one_type_l237_237646

theorem tiling_possible_with_one_type
  {a b m n : ℕ} (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (H : (∃ (k : ℕ), a = k * n) ∨ (∃ (l : ℕ), b = l * m)) :
  (∃ (i : ℕ), a = i * n) ∨ (∃ (j : ℕ), b = j * m) :=
  sorry

end tiling_possible_with_one_type_l237_237646


namespace alcohol_percentage_proof_l237_237385

noncomputable def percentage_alcohol_new_mixture 
  (original_solution_volume : ℕ)
  (percent_A : ℚ)
  (concentration_A : ℚ)
  (percent_B : ℚ)
  (concentration_B : ℚ)
  (percent_C : ℚ)
  (concentration_C : ℚ)
  (water_added_volume : ℕ) : ℚ :=
((original_solution_volume * percent_A * concentration_A) +
 (original_solution_volume * percent_B * concentration_B) +
 (original_solution_volume * percent_C * concentration_C)) /
 (original_solution_volume + water_added_volume) * 100

theorem alcohol_percentage_proof : 
  percentage_alcohol_new_mixture 24 0.30 0.80 0.40 0.90 0.30 0.95 16 = 53.1 := 
by 
  sorry

end alcohol_percentage_proof_l237_237385


namespace find_x_plus_y_l237_237764

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end find_x_plus_y_l237_237764


namespace intersection_M_N_l237_237285

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l237_237285


namespace train_B_speed_l237_237683

noncomputable def train_speed_B (V_A : ℕ) (T_A : ℕ) (T_B : ℕ) : ℕ :=
  V_A * T_A / T_B

theorem train_B_speed
  (V_A : ℕ := 60)
  (T_A : ℕ := 9)
  (T_B : ℕ := 4) :
  train_speed_B V_A T_A T_B = 135 := 
by
  sorry

end train_B_speed_l237_237683


namespace smallest_divisible_1_to_10_l237_237109

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l237_237109


namespace minimum_surface_area_of_cube_l237_237545

noncomputable def brick_length := 25
noncomputable def brick_width := 15
noncomputable def brick_height := 5
noncomputable def side_length := Nat.lcm brick_width brick_length
noncomputable def surface_area := 6 * side_length * side_length

theorem minimum_surface_area_of_cube : surface_area = 33750 := 
by
  sorry

end minimum_surface_area_of_cube_l237_237545


namespace intersection_eq_l237_237247

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l237_237247


namespace haley_more_than_josh_l237_237034

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

end haley_more_than_josh_l237_237034


namespace find_a9_l237_237330

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- conditions
def is_arithmetic_sequence := ∀ n : ℕ, a (n + 1) = a n + d
def given_condition1 := a 5 + a 7 = 16
def given_condition2 := a 3 = 4

-- theorem
theorem find_a9 (h1 : is_arithmetic_sequence a d) (h2 : given_condition1 a) (h3 : given_condition2 a) :
  a 9 = 12 :=
sorry

end find_a9_l237_237330


namespace smallest_number_divisible_1_to_10_l237_237173

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237173


namespace proportional_distribution_ratio_l237_237771

theorem proportional_distribution_ratio (B : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : B = 80) 
  (h2 : S = 164)
  (h3 : S = (B / (1 - r)) + (B * (1 - r))) : 
  r = 0.2 := 
sorry

end proportional_distribution_ratio_l237_237771


namespace wood_burned_afternoon_l237_237552

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon_l237_237552


namespace even_composite_fraction_l237_237415

theorem even_composite_fraction : 
  ((4 * 6 * 8 * 10 * 12) : ℚ) / (14 * 16 * 18 * 20 * 22) = 1 / 42 :=
by 
  sorry

end even_composite_fraction_l237_237415


namespace sign_up_ways_l237_237778

theorem sign_up_ways : 
  let num_ways_A := 2
  let num_ways_B := 2
  let num_ways_C := 2
  num_ways_A * num_ways_B * num_ways_C = 8 := 
by 
  -- show the proof (omitted for simplicity)
  sorry

end sign_up_ways_l237_237778


namespace rectangle_side_ratio_l237_237909

theorem rectangle_side_ratio
  (s : ℝ) -- side length of inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_square : y = s) -- shorter side aligns to form inner square
  (h_outer_area : (3 * s) ^ 2 = 9 * s ^ 2) -- area of outer square is 9 times the inner square
  (h_outer_side_relation : x + s = 3 * s) -- outer side length relation
  : x / y = 2 := 
by
  sorry

end rectangle_side_ratio_l237_237909


namespace natalia_crates_l237_237728

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l237_237728


namespace Isaabel_math_pages_l237_237782

theorem Isaabel_math_pages (x : ℕ) (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  (reading_pages * problems_per_page = 20) ∧ (total_problems = 30) →
  x * problems_per_page + 20 = total_problems →
  x = 2 := by
  sorry

end Isaabel_math_pages_l237_237782


namespace determine_digits_l237_237900

def digit (n : Nat) : Prop := n < 10

theorem determine_digits :
  ∃ (A B C D : Nat), digit A ∧ digit B ∧ digit C ∧ digit D ∧
    (1000 * A + 100 * B + 10 * B + B) ^ 2 = 10000 * A + 1000 * C + 100 * D + 10 * B + B ∧
    (1000 * C + 100 * D + 10 * D + D) ^ 3 = 10000 * A + 1000 * C + 100 * B + 10 * D + D ∧
    A = 9 ∧ B = 6 ∧ C = 2 ∧ D = 1 := 
by
  sorry

end determine_digits_l237_237900


namespace percentage_of_men_l237_237450

variable (M W : ℝ)
variable (h1 : M + W = 100)
variable (h2 : 0.20 * W + 0.70 * M = 40)

theorem percentage_of_men : M = 40 :=
by
  sorry

end percentage_of_men_l237_237450


namespace prism_faces_l237_237878

-- Define basic properties and conditions
def is_prism_with_L_gon_base (edges : ℕ) (L : ℕ) :=
  edges = 3 * L

noncomputable def number_of_faces (L : ℕ) : ℕ :=
  L + 2  -- L lateral faces and 2 bases

-- Main statement
theorem prism_faces (edges : ℕ) (L : ℕ) (h : edges = 3 * L) : number_of_faces L = 8 :=
by
  -- Instantiate the given condition
  have hL : L = 6 := sorry
  -- Prove the number of faces calculation
  have h_faces : number_of_faces L = number_of_faces 6 := by rw hL
  have h_faces_calc : number_of_faces 6 = 8 := sorry
  exact (h_faces.trans h_faces_calc)

end prism_faces_l237_237878


namespace smallest_multiple_1_through_10_l237_237141

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l237_237141


namespace arithmetic_geometric_properties_l237_237024

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

end arithmetic_geometric_properties_l237_237024


namespace weight_of_new_person_l237_237077

theorem weight_of_new_person (A : ℤ) (avg_weight_dec : ℤ) (n : ℤ) (new_avg : ℤ)
  (h1 : A = 102)
  (h2 : avg_weight_dec = 2)
  (h3 : n = 30) 
  (h4 : new_avg = A - avg_weight_dec) : 
  (31 * new_avg) - (30 * A) = 40 := 
by 
  sorry

end weight_of_new_person_l237_237077


namespace cannot_lie_on_line_l237_237932

theorem cannot_lie_on_line (m b : ℝ) (h : m * b < 0) : ¬ (0 = m * (-2022) + b) := 
  by
  sorry

end cannot_lie_on_line_l237_237932


namespace intersection_M_N_l237_237259

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237259


namespace least_odd_prime_factor_of_2023_pow_8_add_1_l237_237421

theorem least_odd_prime_factor_of_2023_pow_8_add_1 :
  ∃ (p : ℕ), Prime p ∧ (2023^8 + 1) % p = 0 ∧ p % 2 = 1 ∧ p = 97 :=
by
  sorry

end least_odd_prime_factor_of_2023_pow_8_add_1_l237_237421


namespace gina_tom_goals_l237_237912

theorem gina_tom_goals :
  let g_day1 := 2
  let t_day1 := g_day1 + 3
  let t_day2 := 6
  let g_day2 := t_day2 - 2
  let g_total := g_day1 + g_day2
  let t_total := t_day1 + t_day2
  g_total + t_total = 17 := by
  sorry

end gina_tom_goals_l237_237912


namespace find_m_l237_237758

open Real

noncomputable def vec_a : ℝ × ℝ := (-1, 2)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (m, 3)

theorem find_m (m : ℝ) (h : -1 * m + 2 * 3 = 0) : m = 6 :=
sorry

end find_m_l237_237758


namespace no_function_f_exists_l237_237007

theorem no_function_f_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 :=
by sorry

end no_function_f_exists_l237_237007


namespace intersection_M_N_l237_237270

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l237_237270


namespace average_speed_first_girl_l237_237989

theorem average_speed_first_girl (v : ℝ) 
  (start_same_point : True)
  (opp_directions : True)
  (avg_speed_second_girl : ℝ := 3)
  (distance_after_12_hours : (v + avg_speed_second_girl) * 12 = 120) :
  v = 7 :=
by
  sorry

end average_speed_first_girl_l237_237989


namespace determine_borrow_lend_years_l237_237708

theorem determine_borrow_lend_years (P : ℝ) (Rb Rl G : ℝ) (n : ℝ) 
  (hP : P = 9000) 
  (hRb : Rb = 4 / 100) 
  (hRl : Rl = 6 / 100) 
  (hG : G = 180) 
  (h_gain : G = P * Rl * n - P * Rb * n) : 
  n = 1 := 
sorry

end determine_borrow_lend_years_l237_237708


namespace prism_faces_l237_237874

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end prism_faces_l237_237874


namespace intersection_of_M_and_N_l237_237229

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l237_237229


namespace voice_of_china_signup_ways_l237_237780

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end voice_of_china_signup_ways_l237_237780


namespace find_quadruples_l237_237899

open Nat

/-- Define the primality property -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Define the problem conditions -/
def valid_quadruple (p1 p2 p3 p4 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  p1 * p2 + p2 * p3 + p3 * p4 + p4 * p1 = 882

/-- The final theorem stating the valid quadruples -/
theorem find_quadruples :
  ∀ (p1 p2 p3 p4 : ℕ), valid_quadruple p1 p2 p3 p4 ↔ 
  (p1 = 2 ∧ p2 = 5 ∧ p3 = 19 ∧ p4 = 37) ∨
  (p1 = 2 ∧ p2 = 11 ∧ p3 = 19 ∧ p4 = 31) ∨
  (p1 = 2 ∧ p2 = 13 ∧ p3 = 19 ∧ p4 = 29) :=
by
  sorry

end find_quadruples_l237_237899


namespace score_order_l237_237938

variables (L N O P : ℕ)

def conditions : Prop := 
  O = L ∧ 
  N < max O P ∧ 
  P > L

theorem score_order (h : conditions L N O P) : N < O ∧ O < P :=
by
  sorry

end score_order_l237_237938


namespace smallest_number_divisible_1_to_10_l237_237123

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237123


namespace grazing_months_of_A_l237_237694

-- Definitions of conditions
def oxen_months_A (x : ℕ) := 10 * x
def oxen_months_B := 12 * 5
def oxen_months_C := 15 * 3
def total_rent := 140
def rent_C := 36

-- Assuming a is the number of months a put his oxen for grazing, we need to prove that a = 7
theorem grazing_months_of_A (a : ℕ) :
  (45 * 140 = 36 * (10 * a + 60 + 45)) → a = 7 := 
by
  intro h
  sorry

end grazing_months_of_A_l237_237694


namespace percentage_sold_is_80_l237_237943

-- Definitions corresponding to conditions
def first_day_houses : Nat := 20
def items_per_house : Nat := 2
def total_items_sold : Nat := 104

-- Calculate the houses visited on the second day
def second_day_houses : Nat := 2 * first_day_houses

-- Calculate items sold on the first day
def items_sold_first_day : Nat := first_day_houses * items_per_house

-- Calculate items sold on the second day
def items_sold_second_day : Nat := total_items_sold - items_sold_first_day

-- Calculate houses sold to on the second day
def houses_sold_to_second_day : Nat := items_sold_second_day / items_per_house

-- Percentage calculation
def percentage_sold_second_day : Nat := (houses_sold_to_second_day * 100) / second_day_houses

-- Theorem proving that James sold to 80% of the houses on the second day
theorem percentage_sold_is_80 : percentage_sold_second_day = 80 := by
  sorry

end percentage_sold_is_80_l237_237943


namespace smallest_number_divisible_by_1_through_10_l237_237158

theorem smallest_number_divisible_by_1_through_10 : ∃ n : ℕ, (∀ k ∈ finset.range 1 11, k ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_number_divisible_by_1_through_10_l237_237158


namespace part_I_part_II_l237_237302

noncomputable def f (x a : ℝ) := |x - 4| + |x - a|

theorem part_I (x : ℝ) : (f x 2 > 10) ↔ (x > 8 ∨ x < -2) :=
by sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≥ 1) ↔ (a ≥ 5 ∨ a ≤ 3) :=
by sorry

end part_I_part_II_l237_237302


namespace martina_success_rate_l237_237958

theorem martina_success_rate
  (games_played : ℕ) (games_won : ℕ) (games_remaining : ℕ)
  (games_won_remaining : ℕ) :
  games_played = 15 → 
  games_won = 9 → 
  games_remaining = 5 → 
  games_won_remaining = 5 → 
  ((games_won + games_won_remaining) / (games_played + games_remaining) : ℚ) * 100 = 70 := 
by
  intros h1 h2 h3 h4
  sorry

end martina_success_rate_l237_237958


namespace smallest_n_l237_237219

-- Define the polynomial expression
def expression (x y : ℕ) := x*y - 3*x + 7*y - 21

-- Define the condition for the number of unique terms in the expansion
def number_of_unique_terms (n : ℕ) := (n + 1) * (n + 1)

-- Define the proof statement
theorem smallest_n (n : ℕ) : number_of_unique_terms n >= 1996 ↔ n = 44 := by
  sorry

end smallest_n_l237_237219


namespace books_about_sports_l237_237697

theorem books_about_sports (total_books school_books sports_books : ℕ) 
  (h1 : total_books = 58)
  (h2 : school_books = 19) 
  (h3 : sports_books = total_books - school_books) :
  sports_books = 39 :=
by 
  rw [h1, h2] at h3 
  exact h3

end books_about_sports_l237_237697


namespace intersection_M_N_l237_237276

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237276


namespace fraction_start_with_9_end_with_0_is_1_over_72_l237_237715

-- Definition of valid 8-digit telephone number
def valid_phone_number (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  2 ≤ d.val ∧ d.val ≤ 9 ∧ n.val ≤ 8

-- Definition of phone numbers that start with 9 and end with 0
def starts_with_9_ends_with_0 (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  d.val = 9 ∧ n.val = 0

-- The total number of valid 8-digit phone numbers
noncomputable def total_valid_numbers : ℕ :=
  8 * (10 ^ 6) * 9

-- The number of valid phone numbers that start with 9 and end with 0
noncomputable def valid_start_with_9_end_with_0 : ℕ :=
  10 ^ 6

-- The target fraction
noncomputable def target_fraction : ℚ :=
  valid_start_with_9_end_with_0 / total_valid_numbers

-- Main theorem
theorem fraction_start_with_9_end_with_0_is_1_over_72 :
  target_fraction = (1 / 72 : ℚ) :=
by
  sorry

end fraction_start_with_9_end_with_0_is_1_over_72_l237_237715


namespace tan_ratio_l237_237057

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end tan_ratio_l237_237057


namespace distance_between_parallel_lines_l237_237978

class ParallelLines (A B c1 c2 : ℝ)

theorem distance_between_parallel_lines (A B c1 c2 : ℝ)
  [h : ParallelLines A B c1 c2] : 
  A = 4 → B = 3 → c1 = 1 → c2 = -9 → 
  (|c1 - c2| / Real.sqrt (A^2 + B^2)) = 2 :=
by
  intros hA hB hc1 hc2
  rw [hA, hB, hc1, hc2]
  norm_num
  sorry

end distance_between_parallel_lines_l237_237978


namespace min_value_A2_minus_B2_l237_237796

noncomputable def A (p q r : ℝ) : ℝ := 
  Real.sqrt (p + 3) + Real.sqrt (q + 6) + Real.sqrt (r + 12)

noncomputable def B (p q r : ℝ) : ℝ :=
  Real.sqrt (p + 2) + Real.sqrt (q + 2) + Real.sqrt (r + 2)

theorem min_value_A2_minus_B2
  (h₁ : 0 ≤ p)
  (h₂ : 0 ≤ q)
  (h₃ : 0 ≤ r) :
  ∃ (p q r : ℝ), A p q r ^ 2 - B p q r ^ 2 = 35 + 10 * Real.sqrt 10 := 
sorry

end min_value_A2_minus_B2_l237_237796


namespace find_product_of_abc_l237_237515

theorem find_product_of_abc :
  ∃ (a b c m : ℝ), 
    a + b + c = 195 ∧
    m = 8 * a ∧
    m = b - 10 ∧
    m = c + 10 ∧
    a * b * c = 95922 := by
  sorry

end find_product_of_abc_l237_237515


namespace volume_Q3_l237_237021

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

end volume_Q3_l237_237021


namespace simplify_expression_l237_237486

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l237_237486


namespace lcm_1_to_10_l237_237152

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

end lcm_1_to_10_l237_237152


namespace capacity_of_smaller_bucket_l237_237925

theorem capacity_of_smaller_bucket (x : ℕ) (h1 : x < 5) (h2 : 5 - x = 2) : x = 3 := by
  sorry

end capacity_of_smaller_bucket_l237_237925


namespace simplify_and_evaluate_l237_237489

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l237_237489


namespace daisies_multiple_of_4_l237_237945

def num_roses := 8
def num_daisies (D : ℕ) := D
def num_marigolds := 48
def num_arrangements := 4

theorem daisies_multiple_of_4 (D : ℕ) 
  (h_roses_div_4 : num_roses % num_arrangements = 0)
  (h_marigolds_div_4 : num_marigolds % num_arrangements = 0)
  (h_total_div_4 : (num_roses + num_daisies D + num_marigolds) % num_arrangements = 0) :
  D % 4 = 0 :=
sorry

end daisies_multiple_of_4_l237_237945


namespace tan_pi_over_12_plus_tan_7pi_over_12_l237_237354

theorem tan_pi_over_12_plus_tan_7pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (7 * Real.pi / 12)) = -4 * (3 - Real.sqrt 3) / 5 :=
by
  sorry

end tan_pi_over_12_plus_tan_7pi_over_12_l237_237354


namespace lcm_1_to_10_l237_237153

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

end lcm_1_to_10_l237_237153


namespace students_count_l237_237814

theorem students_count (x : ℕ) (h1 : x / 2 + x / 4 + x / 7 + 3 = x) : x = 28 :=
  sorry

end students_count_l237_237814


namespace total_people_in_group_l237_237893

-- Given conditions as definitions
def numChinese : Nat := 22
def numAmericans : Nat := 16
def numAustralians : Nat := 11

-- Statement of the theorem to prove
theorem total_people_in_group : (numChinese + numAmericans + numAustralians) = 49 :=
by
  -- proof goes here
  sorry

end total_people_in_group_l237_237893


namespace f_of_g_of_2_l237_237790

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end f_of_g_of_2_l237_237790


namespace smallest_number_divisible_1_to_10_l237_237121

/-- The smallest number divisible by all integers from 1 to 10 is 2520. -/
theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ i : ℕ, i ∈ finset.range 11 → i ≠ 0 → i ∣ n) ∧ n = 2520 :=
by {
  sorry
}

end smallest_number_divisible_1_to_10_l237_237121


namespace select_team_ways_l237_237652

-- Definitions of the conditions and question
def boys := 7
def girls := 10
def boys_needed := 2
def girls_needed := 3
def total_team := 5

-- Theorem statement to prove the number of selecting the team
theorem select_team_ways : (Nat.choose boys boys_needed) * (Nat.choose girls girls_needed) = 2520 := 
by
  -- Place holder for proof
  sorry

end select_team_ways_l237_237652


namespace simplify_expression_l237_237483

theorem simplify_expression :
  ((5 ^ 7 + 2 ^ 8) * (1 ^ 5 - (-1) ^ 5) ^ 10) = 80263680 := by
  sorry

end simplify_expression_l237_237483


namespace find_sum_of_digits_l237_237637

theorem find_sum_of_digits (a b c d : ℕ) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : a = 1)
  (h3 : 1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100)
  : a + b + c + d = 2 := 
sorry

end find_sum_of_digits_l237_237637


namespace net_change_is_minus_0_19_l237_237365

-- Define the yearly change factors as provided in the conditions
def yearly_changes : List ℚ := [6/5, 11/10, 7/10, 4/5, 11/10]

-- Compute the net change over the five years
def net_change (changes : List ℚ) : ℚ :=
  changes.foldl (λ acc x => acc * x) 1 - 1

-- Define the target value for the net change
def target_net_change : ℚ := -19 / 100

-- The theorem to prove the net change calculated matches the target net change
theorem net_change_is_minus_0_19 : net_change yearly_changes = target_net_change :=
  by
    sorry

end net_change_is_minus_0_19_l237_237365


namespace apples_distribution_l237_237400

variable (p b t : ℕ)

theorem apples_distribution (p_eq : p = 40) (b_eq : b = p + 8) (t_eq : t = (3 * b) / 8) :
  t = 18 := by
  sorry

end apples_distribution_l237_237400


namespace cost_of_fencing_field_l237_237983

def ratio (a b : ℕ) : Prop := ∃ k : ℕ, (b = k * a)

def assume_fields : Prop :=
  ∃ (x : ℚ), (ratio 3 4) ∧ (3 * 4 * x^2 = 9408) ∧ (0.25 > 0)

theorem cost_of_fencing_field :
  assume_fields → 98 = 98 := by
  sorry

end cost_of_fencing_field_l237_237983


namespace problem_3_problem_4_l237_237737

open Classical

section
  variable {x₁ x₂ : ℝ}
  theorem problem_3 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) : (Real.log (x₁ * x₂) = Real.log x₁ + Real.log x₂) :=
  by
    sorry

  theorem problem_4 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hlt : x₁ < x₂) : ((Real.log x₁ - Real.log x₂) / (x₁ - x₂) > 0) :=
  by
    sorry
end

end problem_3_problem_4_l237_237737


namespace range_of_f_t_l237_237301

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (Real.exp x) + Real.log x - x

theorem range_of_f_t (a : ℝ) (t : ℝ) 
  (h_unique_critical : ∀ x, f a x = 0 → x = t) : 
  ∃ y : ℝ, y ≥ -2 ∧ ∀ z : ℝ, y = f a t :=
sorry

end range_of_f_t_l237_237301


namespace power_of_negative_125_l237_237725

theorem power_of_negative_125 : (-125 : ℝ)^(4/3) = 625 := by
  sorry

end power_of_negative_125_l237_237725


namespace simplify_and_evaluate_l237_237503

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end simplify_and_evaluate_l237_237503


namespace carpool_commute_distance_l237_237897

theorem carpool_commute_distance :
  (∀ (D : ℕ),
    4 * 5 * ((2 * D : ℝ) / 30) * 2.50 = 5 * 14 →
    D = 21) :=
by
  intro D
  intro h
  sorry

end carpool_commute_distance_l237_237897


namespace intersection_M_N_l237_237266

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l237_237266


namespace arithmetic_sequence_sum_9_l237_237742

theorem arithmetic_sequence_sum_9 :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a n = 2 + n * d) ∧ d ≠ 0 ∧ (2 : ℝ) + 2 * d ≠ 0 ∧ (2 + 5 * d) ≠ 0 ∧ d = 0.5 →
  (2 + 2 * d)^2 = 2 * (2 + 5 * d) →
  (9 * 2 + (9 * 8 / 2) * 0.5) = 36 :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_sum_9_l237_237742


namespace function_always_negative_iff_l237_237081

theorem function_always_negative_iff (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 :=
by
  -- Proof skipped
  sorry

end function_always_negative_iff_l237_237081


namespace value_of_expression_l237_237479

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by sorry

end value_of_expression_l237_237479


namespace number_of_senior_citizen_tickets_sold_on_first_day_l237_237815

theorem number_of_senior_citizen_tickets_sold_on_first_day 
  (S : ℤ) (x : ℤ)
  (student_ticket_price : ℤ := 9)
  (first_day_sales : ℤ := 79)
  (second_day_sales : ℤ := 246) 
  (first_day_student_tickets_sold : ℤ := 3)
  (second_day_senior_tickets_sold : ℤ := 12)
  (second_day_student_tickets_sold : ℤ := 10) 
  (h1 : 12 * S + 10 * student_ticket_price = second_day_sales)
  (h2 : S * x + first_day_student_tickets_sold * student_ticket_price = first_day_sales) : 
  x = 4 :=
by
  sorry

end number_of_senior_citizen_tickets_sold_on_first_day_l237_237815


namespace proof_problem_l237_237032

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

end proof_problem_l237_237032


namespace mr_bird_exact_speed_l237_237800

-- Define the properties and calculating the exact speed
theorem mr_bird_exact_speed (d t : ℝ) (h1 : d = 50 * (t + 1 / 12)) (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 :=
by 
  -- skipping the proof
  sorry

end mr_bird_exact_speed_l237_237800


namespace helga_article_count_l237_237927

theorem helga_article_count :
  let articles_per_30min := 5
  let articles_per_hour := 2 * articles_per_30min
  let hours_per_day := 4
  let days_per_week := 5
  let extra_hours_thursday := 2
  let extra_hours_friday := 3
  let usual_weekly_articles := hours_per_day * days_per_week * articles_per_hour
  let extra_articles_thursday := extra_hours_thursday * articles_per_hour
  let extra_articles_friday := extra_hours_friday * articles_per_hour
  let total_articles := usual_weekly_articles + extra_articles_thursday + extra_articles_friday
  total_articles = 250 :=
by 
  let articles_per_30min := 5
  let articles_per_hour := 2 * articles_per_30min
  let hours_per_day := 4
  let days_per_week := 5
  let extra_hours_thursday := 2
  let extra_hours_friday := 3
  let usual_weekly_articles := hours_per_day * days_per_week * articles_per_hour
  let extra_articles_thursday := extra_hours_thursday * articles_per_hour
  let extra_articles_friday := extra_hours_friday * articles_per_hour
  let total_articles := usual_weekly_articles + extra_articles_thursday + extra_articles_friday
  exact eq.refl 250

end helga_article_count_l237_237927


namespace smallest_divisible_1_to_10_l237_237116

theorem smallest_divisible_1_to_10 : 
  let N := 2520 in
  (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → N % k = 0) ∧
  (∀ m: ℕ, (∀ k: ℕ, k ∈ (finset.range 11) \ {0} → m % k = 0) → N ≤ m) :=
by
  sorry

end smallest_divisible_1_to_10_l237_237116


namespace seashells_remaining_l237_237663

def initial_seashells : ℕ := 35
def given_seashells : ℕ := 18

theorem seashells_remaining : initial_seashells - given_seashells = 17 := by
  sorry

end seashells_remaining_l237_237663


namespace intersection_of_M_and_N_l237_237227

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l237_237227


namespace forecast_interpretation_l237_237781

-- Define the conditions
def condition (precipitation_probability : ℕ) : Prop :=
  precipitation_probability = 78

-- Define the interpretation question as a proof
theorem forecast_interpretation (precipitation_probability: ℕ) (cond : condition precipitation_probability) :
  precipitation_probability = 78 :=
by
  sorry

end forecast_interpretation_l237_237781


namespace poultry_count_correct_l237_237944

noncomputable def total_poultry : ℝ :=
  let hens_total := 40
  let ducks_total := 20
  let geese_total := 10
  let pigeons_total := 30

  -- Calculate males and females
  let hens_males := (2/9) * hens_total
  let hens_females := hens_total - hens_males

  let ducks_males := (1/4) * ducks_total
  let ducks_females := ducks_total - ducks_males

  let geese_males := (3/11) * geese_total
  let geese_females := geese_total - geese_males

  let pigeons_males := (1/2) * pigeons_total
  let pigeons_females := pigeons_total - pigeons_males

  -- Offspring calculations using breeding success rates
  let hens_offspring := (0.85 * hens_females) * 7
  let ducks_offspring := (0.75 * ducks_females) * 9
  let geese_offspring := (0.9 * geese_females) * 5
  let pigeons_pairs := 0.8 * (pigeons_females / 2)
  let pigeons_offspring := pigeons_pairs * 2 * 0.8

  -- Total poultry count
  (hens_total + ducks_total + geese_total + pigeons_total) + (hens_offspring + ducks_offspring + geese_offspring + pigeons_offspring)

theorem poultry_count_correct : total_poultry = 442 := by
  sorry

end poultry_count_correct_l237_237944


namespace minimum_value_of_f_l237_237422

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, y = f x ∧ y >= 5 / 2 :=
by
  sorry

end minimum_value_of_f_l237_237422


namespace sum_of_positive_ks_l237_237666

theorem sum_of_positive_ks :
  ∃ (S : ℤ), S = 39 ∧ ∀ k : ℤ, 
  (∃ α β : ℤ, α * β = 18 ∧ α + β = k) →
  (k > 0 → S = 19 + 11 + 9) := sorry

end sum_of_positive_ks_l237_237666


namespace parabola_distance_from_focus_l237_237607

noncomputable def parabola_distance_proof : Prop :=
  ∃ (x y : ℝ), y^2 = 2 * x ∧ x = 3 ∧ dist (x, y) (1 / 2, 0) = 7 / 2

theorem parabola_distance_from_focus :
  parabola_distance_proof :=
sorry

end parabola_distance_from_focus_l237_237607


namespace intersection_of_M_and_N_l237_237235

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l237_237235


namespace find_S₁₀_l237_237609

noncomputable def sequence_sum (n : ℕ) : ℕ
| 0 => 0
| 1 => 1
| k + 1 => sequence_sum k + a (k + 1)

noncomputable def a : ℕ → ℕ
| 1 => 1
| 2 => 2
| n + 1 => sequence_sum (n - 1) = 2 * (sequence_sum n + 1) - sequence_sum (n + 1)

theorem find_S₁₀ : sequence_sum 10 = 91 := sorry

end find_S₁₀_l237_237609


namespace min_length_intersection_l237_237433

def set_with_length (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}
def length_of_set (a b : ℝ) := b - a
def M (m : ℝ) := set_with_length m (m + 3/4)
def N (n : ℝ) := set_with_length (n - 1/3) n

theorem min_length_intersection (m n : ℝ) (h₁ : 0 ≤ m) (h₂ : m + 3/4 ≤ 1) (h₃ : 0 ≤ n - 1/3) (h₄ : n ≤ 1) : 
  length_of_set (max m (n - 1/3)) (min (m + 3/4) n) = 1/12 :=
by
  sorry

end min_length_intersection_l237_237433


namespace lcm_factor_is_one_l237_237360

theorem lcm_factor_is_one
  (A B : ℕ)
  (hcf : A.gcd B = 42)
  (larger_A : A = 588)
  (other_factor : ∃ X, A.lcm B = 42 * X * 14) :
  ∃ X, X = 1 :=
  sorry

end lcm_factor_is_one_l237_237360


namespace no_real_solutions_l237_237359

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (2 - x^2) / x

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 → (f x + 2 * f (1 / x) = 3 * x)) →
  (∀ x : ℝ, f x = f (-x) → false) :=
by
  intro h1 h2
  sorry

end no_real_solutions_l237_237359


namespace intersection_of_M_and_N_l237_237234

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l237_237234


namespace removed_number_is_34_l237_237504
open Real

theorem removed_number_is_34 (n : ℕ) (x : ℕ) (h₁ : 946 = (43 * (43 + 1)) / 2) (h₂ : 912 = 43 * (152 / 7)) : x = 34 :=
by
  sorry

end removed_number_is_34_l237_237504


namespace tennis_tournament_boxes_needed_l237_237533

theorem tennis_tournament_boxes_needed (n : ℕ) (h : n = 199) : 
  ∃ m, m = 198 ∧
    (∀ k, k < n → (n - k - 1 = m)) :=
by
  sorry

end tennis_tournament_boxes_needed_l237_237533


namespace ratio_minutes_l237_237088

theorem ratio_minutes (x : ℝ) : 
  (12 / 8) = (6 / (x * 60)) → x = 1 / 15 :=
by
  sorry

end ratio_minutes_l237_237088


namespace final_result_l237_237579

-- Define the number of letters in each name
def letters_in_elida : ℕ := 5
def letters_in_adrianna : ℕ := 2 * letters_in_elida - 2

-- Define the alphabetical positions and their sums for each name
def sum_positions_elida : ℕ := 5 + 12 + 9 + 4 + 1
def sum_positions_adrianna : ℕ := 1 + 4 + 18 + 9 + 1 + 14 + 14 + 1
def sum_positions_belinda : ℕ := 2 + 5 + 12 + 9 + 14 + 4 + 1

-- Define the total sum of alphabetical positions
def total_sum_positions : ℕ := sum_positions_elida + sum_positions_adrianna + sum_positions_belinda

-- Define the average of the total sum
def average_sum_positions : ℕ := total_sum_positions / 3

-- Prove the final result
theorem final_result : (average_sum_positions * 3 - sum_positions_elida) = 109 :=
by
  -- Proof skipped
  sorry

end final_result_l237_237579


namespace M_inter_N_eq_2_4_l237_237239

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l237_237239


namespace minimize_distance_AP_BP_l237_237434

theorem minimize_distance_AP_BP :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = -1 ∧
    ∀ P' : ℝ × ℝ, P'.1 = 0 → 
      (dist (3, 2) P + dist (1, -2) P) ≤ (dist (3, 2) P' + dist (1, -2) P') := by
sorry

end minimize_distance_AP_BP_l237_237434


namespace f_x_f_2x_plus_1_l237_237030

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem f_x (x : ℝ) : f x = x^2 - 2 * x - 3 := 
by sorry

theorem f_2x_plus_1 (x : ℝ) : f (2 * x + 1) = 4 * x^2 - 4 := 
by sorry

end f_x_f_2x_plus_1_l237_237030


namespace intersection_M_N_l237_237289

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l237_237289


namespace lcm_1_to_10_l237_237168

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l237_237168


namespace natalia_crates_l237_237727

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l237_237727


namespace functional_equation_solution_l237_237418

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a :=
by
  intro h
  sorry

end functional_equation_solution_l237_237418


namespace greatest_possible_value_of_y_l237_237817

theorem greatest_possible_value_of_y 
  (x y : ℤ) 
  (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ (exists x, x * y + 7 * x + 6 * y = -8) := 
sorry

end greatest_possible_value_of_y_l237_237817


namespace smallest_divisible_1_to_10_l237_237106

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l237_237106


namespace inequality_b_c_a_l237_237766

-- Define the values of a, b, and c
def a := 8^53
def b := 16^41
def c := 64^27

-- State the theorem to prove the inequality b > c > a
theorem inequality_b_c_a : b > c ∧ c > a := by
  sorry

end inequality_b_c_a_l237_237766


namespace solve_math_problem_l237_237347

noncomputable def math_problem : Prop :=
  ∃ (ω α β : ℂ), (ω^5 = 1) ∧ (ω ≠ 1) ∧ (α = ω + ω^2) ∧ (β = ω^3 + ω^4) ∧
  (∀ x : ℂ, x^2 + x + 3 = 0 → x = α ∨ x = β) ∧ (α + β = -1) ∧ (α * β = 3)

theorem solve_math_problem : math_problem := sorry

end solve_math_problem_l237_237347
