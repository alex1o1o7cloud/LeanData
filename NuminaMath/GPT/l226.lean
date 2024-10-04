import Mathlib

namespace exists_common_ratio_of_geometric_progression_l226_226284

theorem exists_common_ratio_of_geometric_progression (a r : ℝ) (h_pos : 0 < r) 
(h_eq: a = a * r + a * r^2 + a * r^3) : ∃ r : ℝ, r^3 + r^2 + r - 1 = 0 :=
by sorry

end exists_common_ratio_of_geometric_progression_l226_226284


namespace product_of_last_two_digits_of_divisible_by_6_l226_226558

-- Definitions
def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def sum_of_last_two_digits (n : ℤ) (a b : ℤ) : Prop := (n % 100) = 10 * a + b

-- Theorem statement
theorem product_of_last_two_digits_of_divisible_by_6 (x a b : ℤ)
  (h1 : is_divisible_by_6 x)
  (h2 : sum_of_last_two_digits x a b)
  (h3 : a + b = 15) :
  (a * b = 54 ∨ a * b = 56) := 
sorry

end product_of_last_two_digits_of_divisible_by_6_l226_226558


namespace initial_volume_kola_solution_l226_226496

-- Initial composition of the kola solution
def initial_composition_sugar (V : ℝ) : ℝ := 0.20 * V

-- Final volume after additions
def final_volume (V : ℝ) : ℝ := V + 3.2 + 12 + 6.8

-- Final amount of sugar after additions
def final_amount_sugar (V : ℝ) : ℝ := initial_composition_sugar V + 3.2

-- Final percentage of sugar in the solution
def final_percentage_sugar (total_sol : ℝ) : ℝ := 0.1966850828729282 * total_sol

theorem initial_volume_kola_solution : 
  ∃ V : ℝ, final_amount_sugar V = final_percentage_sugar (final_volume V) :=
sorry

end initial_volume_kola_solution_l226_226496


namespace henry_kombucha_bottles_l226_226398

theorem henry_kombucha_bottles :
  ∀ (monthly_bottles: ℕ) (cost_per_bottle refund_rate: ℝ) (months_in_year total_bottles_in_year: ℕ),
  (monthly_bottles = 15) →
  (cost_per_bottle = 3.0) →
  (refund_rate = 0.10) →
  (months_in_year = 12) →
  (total_bottles_in_year = monthly_bottles * months_in_year) →
  (total_refund = refund_rate * total_bottles_in_year) →
  (bottles_bought_with_refund = total_refund / cost_per_bottle) →
  bottles_bought_with_refund = 6 :=
by
  intros monthly_bottles cost_per_bottle refund_rate months_in_year total_bottles_in_year
  sorry

end henry_kombucha_bottles_l226_226398


namespace part1_part2_l226_226563

noncomputable def inverse_function_constant (k : ℝ) : Prop :=
  (∀ x : ℝ, 0 < x → (x, 3) ∈ {p : ℝ × ℝ | p.snd = k / p.fst})

noncomputable def range_m (m : ℝ) : Prop :=
  0 < m → m < 3

theorem part1 (k : ℝ) (hk : k ≠ 0) (h : (1, 3).snd = k / (1, 3).fst) :
  k = 3 := by
  sorry

theorem part2 (m : ℝ) (hm : m ≠ 0) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) := by
  sorry

end part1_part2_l226_226563


namespace coffee_shop_sales_l226_226507

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l226_226507


namespace find_function_l226_226018

theorem find_function (α : ℝ) (hα : 0 < α) (f : ℕ+ → ℝ) 
  (h : ∀ k m : ℕ+, α * m ≤ k → k ≤ (α + 1) * m → f (k + m) = f k + f m) :
  ∃ D : ℝ, ∀ n : ℕ+, f n = n * D :=
sorry

end find_function_l226_226018


namespace return_to_freezer_probability_l226_226206

theorem return_to_freezer_probability :
  let cherry := 4
  let orange := 3
  let lemon_lime := 4
  let total := cherry + orange + lemon_lime
  (1 - (cherry / total * (cherry - 1) / (total - 1) 
     + orange / total * (orange - 1) / (total - 1)
     + lemon_lime / total * (lemon_lime - 1) / (total - 1)) : ℚ) = 8/11 :=
by
  let cherry := 4
  let orange := 3
  let lemon_lime := 4
  let total := cherry + orange + lemon_lime
  sorry

end return_to_freezer_probability_l226_226206


namespace polynomial_proof_l226_226532

theorem polynomial_proof (x : ℝ) : 
  (2 * x^2 + 5 * x + 4) = (2 * x^2 + 5 * x - 2) + (10 * x + 6) :=
by sorry

end polynomial_proof_l226_226532


namespace find_contributions_before_johns_l226_226997

-- Definitions based on the conditions provided
def avg_contrib_size_after (A : ℝ) := A + 0.5 * A = 75
def johns_contribution := 100
def total_amount_before (n : ℕ) (A : ℝ) := n * A
def total_amount_after (n : ℕ) (A : ℝ) := (n * A + johns_contribution)

-- Proposition we need to prove
theorem find_contributions_before_johns (n : ℕ) (A : ℝ) :
  avg_contrib_size_after A →
  total_amount_before n A + johns_contribution = (n + 1) * 75 →
  n = 1 :=
by
  sorry

end find_contributions_before_johns_l226_226997


namespace b_power_a_equals_nine_l226_226130

theorem b_power_a_equals_nine (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : b^a = 9 := by
  sorry

end b_power_a_equals_nine_l226_226130


namespace abs_diff_squares_1055_985_eq_1428_l226_226753

theorem abs_diff_squares_1055_985_eq_1428 :
  abs ((105.5: ℝ)^2 - (98.5: ℝ)^2) = 1428 :=
by
  sorry

end abs_diff_squares_1055_985_eq_1428_l226_226753


namespace proof_f_derivative_neg1_l226_226111

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x ^ 4 + b * x ^ 2 + c

noncomputable def f_derivative (x : ℝ) (a b : ℝ) : ℝ :=
  4 * a * x ^ 3 + 2 * b * x

theorem proof_f_derivative_neg1
  (a b c : ℝ) (h : f_derivative 1 a b = 2) :
  f_derivative (-1) a b = -2 :=
by
  sorry

end proof_f_derivative_neg1_l226_226111


namespace sphere_surface_area_ratio_l226_226688

theorem sphere_surface_area_ratio (V1 V2 r1 r2 A1 A2 : ℝ)
    (h_volume_ratio : V1 / V2 = 8 / 27)
    (h_volume_formula1 : V1 = (4/3) * Real.pi * r1^3)
    (h_volume_formula2 : V2 = (4/3) * Real.pi * r2^3)
    (h_surface_area_formula1 : A1 = 4 * Real.pi * r1^2)
    (h_surface_area_formula2 : A2 = 4 * Real.pi * r2^2)
    (h_radius_ratio : r1 / r2 = 2 / 3) :
  A1 / A2 = 4 / 9 :=
sorry

end sphere_surface_area_ratio_l226_226688


namespace find_k_l226_226082

noncomputable def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - (1 / x) + 2

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → 
  k = - 134 / 3 :=
by
  sorry

end find_k_l226_226082


namespace divisible_by_five_l226_226141

theorem divisible_by_five (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a * b = 15) : ∃ k : ℕ, 110 * 1000 + a * 100 + b * 10 ∗ 1 = k * 5 :=
by
  sorry

end divisible_by_five_l226_226141


namespace calculation_correct_l226_226078

theorem calculation_correct : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end calculation_correct_l226_226078


namespace car_speeds_l226_226052

noncomputable def distance_between_places : ℝ := 135
noncomputable def departure_time_diff : ℝ := 4 -- large car departs 4 hours before small car
noncomputable def arrival_time_diff : ℝ := 0.5 -- small car arrives 30 minutes earlier than large car
noncomputable def speed_ratio : ℝ := 5 / 2 -- ratio of speeds (small car : large car)

theorem car_speeds (v_small v_large : ℝ) (h1 : v_small / v_large = speed_ratio) :
    v_small = 45 ∧ v_large = 18 :=
sorry

end car_speeds_l226_226052


namespace common_difference_of_arithmetic_sequence_l226_226815

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 0 = 2) 
  (h2 : ∀ n, a (n+1) = a n + d)
  (h3 : a 9 = 20): 
  d = 2 := 
by
  sorry

end common_difference_of_arithmetic_sequence_l226_226815


namespace sufficient_not_necessary_l226_226063

theorem sufficient_not_necessary (x : ℝ) : (x > 3) → (abs (x - 3) > 0) ∧ (¬(abs (x - 3) > 0) → (¬(x > 3))) :=
by
  sorry

end sufficient_not_necessary_l226_226063


namespace lcm_18_30_is_90_l226_226475

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l226_226475


namespace correct_operation_l226_226615

-- Define the operations given in the conditions
def optionA (m : ℝ) := m^2 + m^2 = 2 * m^4
def optionB (a : ℝ) := a^2 * a^3 = a^5
def optionC (m n : ℝ) := (m * n^2) ^ 3 = m * n^6
def optionD (m : ℝ) := m^6 / m^2 = m^3

-- Theorem stating that option B is the correct operation
theorem correct_operation (a m n : ℝ) : optionB a :=
by sorry

end correct_operation_l226_226615


namespace no_real_roots_l226_226358

def op (m n : ℝ) : ℝ := n^2 - m * n + 1

theorem no_real_roots (x : ℝ) : op 1 x = 0 → ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by {
  sorry
}

end no_real_roots_l226_226358


namespace clear_board_possible_l226_226857

def operation (board : Array (Array Nat)) (op_type : String) (index : Fin 8) : Array (Array Nat) :=
  match op_type with
  | "column" => board.map (λ row => row.modify index fun x => x - 1)
  | "row" => board.modify index fun row => row.map (λ x => 2 * x)
  | _ => board

def isZeroBoard (board : Array (Array Nat)) : Prop :=
  board.all (λ row => row.all (λ x => x = 0))

theorem clear_board_possible (initial_board : Array (Array Nat)) : 
  ∃ (ops : List (String × Fin 8)), 
    isZeroBoard (ops.foldl (λ b ⟨t, i⟩ => operation b t i) initial_board) :=
sorry

end clear_board_possible_l226_226857


namespace necessary_but_not_sufficient_condition_l226_226717

variable (A B C : Set α) (a : α)
variable [Nonempty α]
variable (H1 : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C))

theorem necessary_but_not_sufficient_condition :
  (a ∈ B → a ∈ A) ∧ ¬(a ∈ A → a ∈ B) :=
by
  sorry

end necessary_but_not_sufficient_condition_l226_226717


namespace total_money_made_l226_226654

-- Define the given conditions.
def total_rooms : ℕ := 260
def single_rooms : ℕ := 64
def single_room_cost : ℕ := 35
def double_room_cost : ℕ := 60

-- Define the number of double rooms.
def double_rooms : ℕ := total_rooms - single_rooms

-- Define the total money made from single and double rooms.
def money_from_single_rooms : ℕ := single_rooms * single_room_cost
def money_from_double_rooms : ℕ := double_rooms * double_room_cost

-- State the theorem we want to prove.
theorem total_money_made : 
  (money_from_single_rooms + money_from_double_rooms) = 14000 :=
  by
    sorry -- Proof is omitted.

end total_money_made_l226_226654


namespace sum_of_squares_of_roots_eq_1853_l226_226454

theorem sum_of_squares_of_roots_eq_1853
  (α β : ℕ) (h_prime_α : Prime α) (h_prime_beta : Prime β) (h_sum : α + β = 45)
  (h_quadratic_eq : ∀ x, x^2 - 45*x + α*β = 0 → x = α ∨ x = β) :
  α^2 + β^2 = 1853 := 
by
  sorry

end sum_of_squares_of_roots_eq_1853_l226_226454


namespace lauren_change_l226_226151

-- Define the given conditions as Lean terms.
def price_meat_per_pound : ℝ := 3.5
def pounds_meat : ℝ := 2.0
def price_buns : ℝ := 1.5
def price_lettuce : ℝ := 1.0
def pounds_tomato : ℝ := 1.5
def price_tomato_per_pound : ℝ := 2.0
def price_pickles : ℝ := 2.5
def coupon_value : ℝ := 1.0
def amount_paid : ℝ := 20.0

-- Define the total cost of each item.
def cost_meat : ℝ := pounds_meat * price_meat_per_pound
def cost_tomato : ℝ := pounds_tomato * price_tomato_per_pound
def total_cost_before_coupon : ℝ := cost_meat + price_buns + price_lettuce + cost_tomato + price_pickles

-- Define the final total cost after applying the coupon.
def final_total_cost : ℝ := total_cost_before_coupon - coupon_value

-- Define the expected change.
def expected_change : ℝ := amount_paid - final_total_cost

-- Prove that the expected change is $6.00.
theorem lauren_change : expected_change = 6.0 := by
  sorry

end lauren_change_l226_226151


namespace age_of_vanya_and_kolya_l226_226598

theorem age_of_vanya_and_kolya (P V K : ℕ) (hP : P = 10)
  (hV : V = P - 1) (hK : K = P - 5 + 1) : V = 9 ∧ K = 6 :=
by
  sorry

end age_of_vanya_and_kolya_l226_226598


namespace arithmetic_sequence_property_l226_226380

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property
  (h1 : a 6 + a 8 = 10)
  (h2 : a 3 = 1)
  (property : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q)
  : a 11 = 9 :=
by
  sorry

end arithmetic_sequence_property_l226_226380


namespace min_travel_time_l226_226912

/-- Two people, who have one bicycle, need to travel from point A to point B, which is 40 km away from point A. 
The first person walks at a speed of 4 km/h and rides the bicycle at 30 km/h, 
while the second person walks at a speed of 6 km/h and rides the bicycle at 20 km/h. 
Prove that the minimum time in which they can both get to point B is 25/9 hours. -/
theorem min_travel_time (d : ℕ) (v_w1 v_c1 v_w2 v_c2 : ℕ) (min_time : ℚ) 
  (h_d : d = 40)
  (h_v1_w : v_w1 = 4)
  (h_v1_c : v_c1 = 30)
  (h_v2_w : v_w2 = 6)
  (h_v2_c : v_c2 = 20)
  (h_min_time : min_time = 25 / 9) :
  ∃ y x : ℚ, 4*y + (2/3)*y*30 = 40 ∧ min_time = y + (2/3)*y :=
sorry

end min_travel_time_l226_226912


namespace fraction_simplification_l226_226355

/-- Given x and y, under the conditions x ≠ 3y and x ≠ -3y, 
we want to prove that (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y). -/
theorem fraction_simplification (x y : ℝ) (h1 : x ≠ 3 * y) (h2 : x ≠ -3 * y) :
  (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y) :=
by
  sorry

end fraction_simplification_l226_226355


namespace roots_greater_than_half_iff_l226_226669

noncomputable def quadratic_roots (a : ℝ) (x1 x2 : ℝ) : Prop :=
  (2 - a) * x1^2 - 3 * a * x1 + 2 * a = 0 ∧ 
  (2 - a) * x2^2 - 3 * a * x2 + 2 * a = 0 ∧
  x1 > 1/2 ∧ x2 > 1/2

theorem roots_greater_than_half_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots a x1 x2) ↔ (16 / 17 < a ∧ a < 2) :=
sorry

end roots_greater_than_half_iff_l226_226669


namespace books_combination_l226_226550

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l226_226550


namespace additional_amount_needed_l226_226016

-- Define the amounts spent on shampoo, conditioner, and lotion
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost_per_bottle : ℝ := 6.00
def lotion_quantity : ℕ := 3

-- Define the amount required for free shipping
def free_shipping_threshold : ℝ := 50.00

-- Calculate the total amount spent
def total_spent : ℝ := shampoo_cost + conditioner_cost + (lotion_quantity * lotion_cost_per_bottle)

-- Define the additional amount needed for free shipping
def additional_needed_for_shipping : ℝ := free_shipping_threshold - total_spent

-- The final goal to prove
theorem additional_amount_needed : additional_needed_for_shipping = 12.00 :=
by
  sorry

end additional_amount_needed_l226_226016


namespace range_of_a_l226_226265

theorem range_of_a (a : ℝ) (h : ¬ (1^2 - 2*1 + a > 0)) : 1 ≤ a := sorry

end range_of_a_l226_226265


namespace ones_digit_of_9_pow_46_l226_226607

theorem ones_digit_of_9_pow_46 : (9 ^ 46) % 10 = 1 :=
by
  sorry

end ones_digit_of_9_pow_46_l226_226607


namespace train_length_l226_226501

theorem train_length (L : ℝ) 
    (cross_bridge : ∀ (t_bridge : ℝ), t_bridge = 10 → L + 200 = t_bridge * (L / 5))
    (cross_lamp_post : ∀ (t_lamp_post : ℝ), t_lamp_post = 5 → L = t_lamp_post * (L / 5)) :
  L = 200 := 
by 
  -- sorry is used to skip the proof part
  sorry

end train_length_l226_226501


namespace probability_x_lt_2y_l226_226635

noncomputable def rectangle := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }

noncomputable def region_of_interest := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 2 * p.2 }

noncomputable def area_rectangle := 6 * 2

noncomputable def area_trapezoid := (1 / 2) * (4 + 6) * 2

theorem probability_x_lt_2y : (area_trapezoid / area_rectangle) = 5 / 6 :=
by
  -- skip the proof
  sorry

end probability_x_lt_2y_l226_226635


namespace max_difference_and_max_value_of_multiple_of_5_l226_226745

theorem max_difference_and_max_value_of_multiple_of_5:
  ∀ (N : ℕ), 
  (∃ (d : ℕ), d = 0 ∨ d = 5 ∧ N = 740 + d) →
  (∃ (diff : ℕ), diff = 5) ∧ (∃ (max_num : ℕ), max_num = 745) :=
by
  intro N
  rintro ⟨d, (rfl | rfl), rfl⟩
  apply And.intro
  use 5
  use 745
  sorry

end max_difference_and_max_value_of_multiple_of_5_l226_226745


namespace math_problem_l226_226916

variable (x : ℕ)
variable (h : x + 7 = 27)

theorem math_problem : (x = 20) ∧ (((x / 5) + 5) * 7 = 63) :=
by
  have h1 : x = 20 := by {
    -- x can be solved here using the condition, but we use sorry to skip computation.
    sorry
  }
  have h2 : (((x / 5) + 5) * 7 = 63) := by {
    -- The second part result can be computed using the derived x value, but we use sorry to skip computation.
    sorry
  }
  exact ⟨h1, h2⟩

end math_problem_l226_226916


namespace part_one_part_two_l226_226392

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 / x

theorem part_one (x a : ℝ) (hx : x > 0) (ineq : x * f' x ≤ x^2 + a * x + 1) : a ∈ Set.Ici (-1) :=
by sorry

theorem part_two (x : ℝ) (hx : x > 0) : (x - 1) * f x ≥ 0 :=
by sorry

end part_one_part_two_l226_226392


namespace sum_of_three_squares_l226_226559

theorem sum_of_three_squares (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
  sorry

end sum_of_three_squares_l226_226559


namespace intersection_A_B_l226_226383

def A : Set ℝ := { x : ℝ | |x - 1| < 2 }
def B : Set ℝ := { x : ℝ | x^2 - x - 2 > 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l226_226383


namespace sum_of_three_consecutive_numbers_l226_226316

theorem sum_of_three_consecutive_numbers (smallest : ℕ) (h : smallest = 29) :
  (smallest + (smallest + 1) + (smallest + 2)) = 90 :=
by
  sorry

end sum_of_three_consecutive_numbers_l226_226316


namespace b_investment_correct_l226_226334

-- Constants for shares and investments
def a_investment : ℕ := 11000
def a_share : ℕ := 2431
def b_share : ℕ := 3315
def c_investment : ℕ := 23000

-- Goal: Prove b's investment given the conditions
theorem b_investment_correct (b_investment : ℕ) (h : 2431 * b_investment = 11000 * 3315) :
  b_investment = 15000 := by
  sorry

end b_investment_correct_l226_226334


namespace mrs_jensens_preschool_l226_226439

theorem mrs_jensens_preschool (total_students students_with_both students_with_neither students_with_green_eyes students_with_red_hair : ℕ) 
(h1 : total_students = 40) 
(h2 : students_with_red_hair = 3 * students_with_green_eyes) 
(h3 : students_with_both = 8) 
(h4 : students_with_neither = 4) :
students_with_green_eyes = 12 := 
sorry

end mrs_jensens_preschool_l226_226439


namespace product_increases_exactly_13_times_by_subtracting_3_l226_226834

theorem product_increases_exactly_13_times_by_subtracting_3 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    13 * (n1 * n2 * n3 * n4 * n5 * n6 * n7) =
      ((n1 - 3) * (n2 - 3) * (n3 - 3) * (n4 - 3) * (n5 - 3) * (n6 - 3) * (n7 - 3)) :=
sorry

end product_increases_exactly_13_times_by_subtracting_3_l226_226834


namespace mushroom_pickers_l226_226928

theorem mushroom_pickers (n : ℕ) (hn : n = 18) (total_mushrooms : ℕ) (h_total : total_mushrooms = 162) (h_each : ∀ i : ℕ, i < n → 0 < 1) : 
  ∃ i j : ℕ, i < n ∧ j < n ∧ i ≠ j ∧ (total_mushrooms / n = (total_mushrooms / n)) :=
sorry

end mushroom_pickers_l226_226928


namespace three_digit_number_l226_226367

/-- 
Prove there exists three-digit number N such that 
1. N is of form 100a + 10b + c
2. 1 ≤ a ≤ 9
3. 0 ≤ b, c ≤ 9
4. N = 11 * (a + b + c)
--/
theorem three_digit_number (N a b c : ℕ) 
  (hN: N = 100 * a + 10 * b + c) 
  (h_a: 1 ≤ a ∧ a ≤ 9)
  (h_b_c: 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_condition: N = 11 * (a + b + c)) :
  N = 198 := 
sorry

end three_digit_number_l226_226367


namespace remaining_distance_l226_226170

-- Definitions of conditions
def distance_to_grandmother : ℕ := 300
def speed_per_hour : ℕ := 60
def time_elapsed : ℕ := 2

-- Statement of the proof problem
theorem remaining_distance : distance_to_grandmother - (speed_per_hour * time_elapsed) = 180 :=
by 
  sorry

end remaining_distance_l226_226170


namespace probability_divisible_by_3_l226_226618

noncomputable def prime_digit_two_digit_integers : List ℕ :=
  [23, 25, 27, 32, 35, 37, 52, 53, 57, 72, 73, 75]

noncomputable def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- The statement
theorem probability_divisible_by_3 :
  let favorable_outcomes := prime_digit_two_digit_integers.filter is_divisible_by_3
  let total_outcomes := prime_digit_two_digit_integers.length
  let r := favorable_outcomes.length / total_outcomes
  r = 1 / 3 := 
sorry

end probability_divisible_by_3_l226_226618


namespace difference_in_roses_and_orchids_l226_226750

theorem difference_in_roses_and_orchids
    (initial_roses : ℕ) (initial_orchids : ℕ) (initial_tulips : ℕ)
    (final_roses : ℕ) (final_orchids : ℕ) (final_tulips : ℕ)
    (ratio_roses_orchids_num : ℕ) (ratio_roses_orchids_den : ℕ)
    (ratio_roses_tulips_num : ℕ) (ratio_roses_tulips_den : ℕ)
    (h1 : initial_roses = 7)
    (h2 : initial_orchids = 12)
    (h3 : initial_tulips = 5)
    (h4 : final_roses = 11)
    (h5 : final_orchids = 20)
    (h6 : final_tulips = 10)
    (h7 : ratio_roses_orchids_num = 2)
    (h8 : ratio_roses_orchids_den = 5)
    (h9 : ratio_roses_tulips_num = 3)
    (h10 : ratio_roses_tulips_den = 5)
    (h11 : (final_roses : ℚ) / final_orchids = (ratio_roses_orchids_num : ℚ) / ratio_roses_orchids_den)
    (h12 : (final_roses : ℚ) / final_tulips = (ratio_roses_tulips_num : ℚ) / ratio_roses_tulips_den)
    : final_orchids - final_roses = 9 :=
by
  sorry

end difference_in_roses_and_orchids_l226_226750


namespace fraction_numerator_greater_than_denominator_l226_226897

theorem fraction_numerator_greater_than_denominator {x : ℝ} : 
  -1 ≤ x ∧ x ≤ 3 ∧ 5 * x + 2 > 8 - 3 * x ↔ (3 / 4) < x ∧ x ≤ 3 :=
by 
  sorry

end fraction_numerator_greater_than_denominator_l226_226897


namespace plant_lamp_arrangements_l226_226582

/-- Rachel has two identical basil plants and an aloe plant.
Additionally, she has two identical white lamps, two identical red lamps, and 
two identical blue lamps she can put each plant under 
(she can put more than one plant under a lamp, but each plant is under exactly one lamp). 
-/
theorem plant_lamp_arrangements : 
  let plants := ["basil", "basil", "aloe"]
  let lamps := ["white", "white", "red", "red", "blue", "blue"]
  ∃ n, n = 27 := by
  sorry

end plant_lamp_arrangements_l226_226582


namespace geometric_sequence_a6_l226_226030

theorem geometric_sequence_a6 :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n) ∧ (a 4 * a 10 = 16) → (a 6 = 2) :=
by
  sorry

end geometric_sequence_a6_l226_226030


namespace martin_big_bell_rings_l226_226159

theorem martin_big_bell_rings (B S : ℚ) (h1 : S = B / 3 + B^2 / 4) (h2 : S + B = 52) : B = 12 :=
by
  sorry

end martin_big_bell_rings_l226_226159


namespace median_to_longest_side_l226_226533

theorem median_to_longest_side
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26)
  (h4 : a^2 + b^2 = c^2) :
  ∃ m : ℕ, m = c / 2 ∧ m = 13 := 
by {
  sorry
}

end median_to_longest_side_l226_226533


namespace park_maple_trees_total_l226_226047

theorem park_maple_trees_total (current_maples planted_maples : ℕ) 
    (h1 : current_maples = 2) (h2 : planted_maples = 9) 
    : current_maples + planted_maples = 11 := 
by
  sorry

end park_maple_trees_total_l226_226047


namespace product_quality_difference_l226_226605

variable (n a b c d : ℕ)
variable (P_K_2 : ℝ → ℝ)

def first_class_freq_A := a / (a + b : ℕ)
def first_class_freq_B := c / (c + d : ℕ)

def K2 := (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_difference
  (ha : a = 150) (hb : b = 50) 
  (hc : c = 120) (hd : d = 80)
  (hn : n = 400)
  (hK : P_K_2 0.010 = 6.635) : 
  first_class_freq_A a b = 3 / 4 ∧
  first_class_freq_B c d = 3 / 5 ∧
  K2 n a b c d > P_K_2 0.010 :=
by {
  sorry
}

end product_quality_difference_l226_226605


namespace dish_heats_up_by_5_degrees_per_minute_l226_226147

theorem dish_heats_up_by_5_degrees_per_minute
  (final_temperature initial_temperature : ℕ)
  (time_taken : ℕ)
  (h1 : final_temperature = 100)
  (h2 : initial_temperature = 20)
  (h3 : time_taken = 16) :
  (final_temperature - initial_temperature) / time_taken = 5 :=
by
  sorry

end dish_heats_up_by_5_degrees_per_minute_l226_226147


namespace unique_solution_exists_l226_226659

theorem unique_solution_exists (ell : ℚ) (h : ell ≠ -2) : 
  (∃! x : ℚ, (x + 3) / (ell * x + 2) = x) ↔ ell = -1 / 12 := 
by
  sorry

end unique_solution_exists_l226_226659


namespace max_matches_l226_226212

theorem max_matches (x y z m : ℕ) (h1 : x + y + z = 19) (h2 : x * y + y * z + x * z = m) : m ≤ 120 :=
sorry

end max_matches_l226_226212


namespace triangle_area_l226_226062

theorem triangle_area (a b c p : ℕ) (h_ratio : a = 5 * p) (h_ratio2 : b = 12 * p) (h_ratio3 : c = 13 * p) (h_perimeter : a + b + c = 300) : 
  (1 / 4) * Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) = 3000 := 
by 
  sorry

end triangle_area_l226_226062


namespace subset_N_M_l226_226543

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | x^2 - x < 0 }

-- The proof goal
theorem subset_N_M : N ⊆ M := by
  sorry

end subset_N_M_l226_226543


namespace sculpture_and_base_height_l226_226490

theorem sculpture_and_base_height :
  let sculpture_height_in_feet := 2
  let sculpture_height_in_inches := 10
  let base_height_in_inches := 2
  let total_height_in_inches := (sculpture_height_in_feet * 12) + sculpture_height_in_inches + base_height_in_inches
  let total_height_in_feet := total_height_in_inches / 12
  total_height_in_feet = 3 :=
by
  sorry

end sculpture_and_base_height_l226_226490


namespace system_sum_of_squares_l226_226312

theorem system_sum_of_squares :
  (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    9*y1^2 - 4*x1^2 = 144 - 48*x1 ∧ 9*y1^2 + 4*x1^2 = 144 + 18*x1*y1 ∧
    9*y2^2 - 4*x2^2 = 144 - 48*x2 ∧ 9*y2^2 + 4*x2^2 = 144 + 18*x2*y2 ∧
    9*y3^2 - 4*x3^2 = 144 - 48*x3 ∧ 9*y3^2 + 4*x3^2 = 144 + 18*x3*y3 ∧
    (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2 = 68)) :=
by sorry

end system_sum_of_squares_l226_226312


namespace number_of_correct_statements_l226_226892

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

axiom random_variable_X_distribution {σ : ℝ} : 
  ∀(X : ℝ → ℝ), X = normal_distribution 2 σ

axiom probability_X_lt_5 {σ : ℝ} :
  ∀(P : ℝ → ℝ), P (λ X, X < 5) = 0.8

axiom regression_equation :
  ∀(x y : ℝ), y = 0.85 * x - 82

axiom bags_setup :
  ∀ (whites_A blacks_A : ℕ) (whites_B blacks_B : ℕ),
    whites_A = 3 ∧ blacks_A = 2 ∧ whites_B = 4 ∧ blacks_B = 4

axiom probability_white_ball :
  ∀ (P : ℕ → ℝ), P (λ _, true) to_finite = 13/25

theorem number_of_correct_statements : 
  (statement_1_correct → statement_2_correct → statement_3_correct → ∃ n = 2) := 
sorry

end number_of_correct_statements_l226_226892


namespace divisor_of_first_division_l226_226619

theorem divisor_of_first_division (n d : ℕ) (hn_pos : 0 < n)
  (h₁ : (n + 1) % d = 4) (h₂ : n % 2 = 1) : 
  d = 6 :=
sorry

end divisor_of_first_division_l226_226619


namespace frac_wx_l226_226400

theorem frac_wx (x y z w : ℚ) (h1 : x / y = 5) (h2 : y / z = 1 / 2) (h3 : z / w = 7) : w / x = 2 / 35 :=
by
  sorry

end frac_wx_l226_226400


namespace polikarp_make_first_box_empty_l226_226861

theorem polikarp_make_first_box_empty (n : ℕ) (h : n ≤ 30) : ∃ (x y : ℕ), x + y ≤ 10 ∧ ∀ k : ℕ, k ≤ x → k + k * y = n :=
by
  sorry

end polikarp_make_first_box_empty_l226_226861


namespace correct_transformation_l226_226921

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end correct_transformation_l226_226921


namespace range_of_a_l226_226530

open Set Real

theorem range_of_a :
  let p := ∀ x : ℝ, |4 * x - 3| ≤ 1
  let q := ∀ x : ℝ, x^2 - (2 * a + 1) * x + (a * (a + 1)) ≤ 0
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q)
  → (∀ x : Icc (0 : ℝ) (1 / 2 : ℝ), a = x) :=
by
  intros
  sorry

end range_of_a_l226_226530


namespace a_is_5_if_extreme_at_neg3_l226_226112

-- Define the function f with parameter a
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 3

-- Define the given condition that f reaches an extreme value at x = -3
def reaches_extreme_at (a : ℝ) : Prop := f_prime a (-3) = 0

-- Prove that a = 5 if f reaches an extreme value at x = -3
theorem a_is_5_if_extreme_at_neg3 : ∀ a : ℝ, reaches_extreme_at a → a = 5 :=
by
  intros a h
  -- Proof omitted
  sorry

end a_is_5_if_extreme_at_neg3_l226_226112


namespace initial_overs_played_l226_226561

-- Define the conditions
def initial_run_rate : ℝ := 6.2
def remaining_overs : ℝ := 40
def remaining_run_rate : ℝ := 5.5
def target_runs : ℝ := 282

-- Define what we seek to prove
theorem initial_overs_played :
  ∃ x : ℝ, (6.2 * x) + (5.5 * 40) = 282 ∧ x = 10 :=
by
  sorry

end initial_overs_played_l226_226561


namespace y_coord_of_equidistant_point_on_y_axis_l226_226791

/-!
  # Goal
  Prove that the $y$-coordinate of the point P on the $y$-axis that is equidistant from points $A(5, 0)$ and $B(3, 6)$ is \( \frac{5}{3} \).
  Conditions:
  - Point A has coordinates (5, 0).
  - Point B has coordinates (3, 6).
-/

theorem y_coord_of_equidistant_point_on_y_axis :
  ∃ y : ℝ, y = 5 / 3 ∧ (dist (⟨0, y⟩ : ℝ × ℝ) (⟨5, 0⟩ : ℝ × ℝ) = dist (⟨0, y⟩ : ℝ × ℝ) (⟨3, 6⟩ : ℝ × ℝ)) :=
by
  sorry -- Proof omitted

end y_coord_of_equidistant_point_on_y_axis_l226_226791


namespace a_10_equals_1024_l226_226992

-- Define the sequence a_n and its properties
variable {a : ℕ → ℕ}
variable (h_prop : ∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)
variable (h_a2 : a 2 = 4)

-- Prove the statement that a_10 = 1024 given the above conditions.
theorem a_10_equals_1024 : a 10 = 1024 :=
sorry

end a_10_equals_1024_l226_226992


namespace volume_tetrahedron_PXYZ_l226_226180

noncomputable def volume_of_tetrahedron_PXYZ (x y z : ℝ) : ℝ :=
  (1 / 6) * x * y * z

theorem volume_tetrahedron_PXYZ :
  ∃ (x y z : ℝ), (x^2 + y^2 = 49) ∧ (y^2 + z^2 = 64) ∧ (z^2 + x^2 = 81) ∧
  volume_of_tetrahedron_PXYZ (Real.sqrt x) (Real.sqrt y) (Real.sqrt z) = 4 * Real.sqrt 11 := 
by {
  sorry
}

end volume_tetrahedron_PXYZ_l226_226180


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l226_226604

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l226_226604


namespace volume_of_inscribed_sphere_l226_226781

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end volume_of_inscribed_sphere_l226_226781


namespace brown_gumdrops_after_replacement_l226_226498

theorem brown_gumdrops_after_replacement
  (total_gumdrops : ℕ)
  (percent_blue : ℚ)
  (percent_brown : ℚ)
  (percent_red : ℚ)
  (percent_yellow : ℚ)
  (num_green : ℕ)
  (replace_half_blue_with_brown : ℕ) :
  total_gumdrops = 120 →
  percent_blue = 0.30 →
  percent_brown = 0.20 →
  percent_red = 0.15 →
  percent_yellow = 0.10 →
  num_green = 30 →
  replace_half_blue_with_brown = 18 →
  ((percent_brown * ↑total_gumdrops) + replace_half_blue_with_brown) = 42 :=
by sorry

end brown_gumdrops_after_replacement_l226_226498


namespace integer_solution_interval_l226_226541

theorem integer_solution_interval {f : ℝ → ℝ} (m : ℝ) :
  (∀ x : ℤ, (-x^2 + x + m + 2 ≥ |x| ↔ (x : ℝ) = n)) ↔ (-2 ≤ m ∧ m < -1) := 
sorry

end integer_solution_interval_l226_226541


namespace system_of_equations_correct_l226_226911

-- Define the problem conditions
variable (x y : ℝ) -- Define the productivity of large and small harvesters

-- Define the correct system of equations as per the problem
def system_correct : Prop := (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8)

-- State the theorem to prove the correctness of the system of equations under given conditions
theorem system_of_equations_correct (x y : ℝ) : (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8) :=
by
  sorry

end system_of_equations_correct_l226_226911


namespace find_range_of_a_l226_226970

def prop_p (a : ℝ) : Prop :=
∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def prop_q (a : ℝ) : Prop :=
(∃ x₁ x₂ : ℝ, x₁ * x₂ = 1 ∧ x₁ + x₂ = -(a - 1) ∧ (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2))

def range_a (a : ℝ) : Prop :=
(-2 < a ∧ a <= -3/2) ∨ (-1 <= a ∧ a <= 2)

theorem find_range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a) ↔ range_a a :=
sorry

end find_range_of_a_l226_226970


namespace sum_of_areas_of_squares_l226_226320

theorem sum_of_areas_of_squares (a b x : ℕ) 
  (h_overlapping_min : 9 ≤ (min a b) ^ 2)
  (h_overlapping_max : (min a b) ^ 2 ≤ 25)
  (h_sum_of_sides : a + b + x = 23) :
  a^2 + b^2 + x^2 = 189 := 
sorry

end sum_of_areas_of_squares_l226_226320


namespace relation_among_a_b_c_l226_226529

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relation_among_a_b_c : c > a ∧ a > b :=
by
  -- Prove that c > a and a > b
  sorry

end relation_among_a_b_c_l226_226529


namespace find_distinct_prime_triples_l226_226522

noncomputable def areDistinctPrimes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r

def satisfiesConditions (p q r : ℕ) : Prop :=
  p ∣ (q + r) ∧ q ∣ (r + 2 * p) ∧ r ∣ (p + 3 * q)

theorem find_distinct_prime_triples :
  { (p, q, r) : ℕ × ℕ × ℕ | areDistinctPrimes p q r ∧ satisfiesConditions p q r } =
  { (5, 3, 2), (2, 11, 7), (2, 3, 11) } :=
by
  sorry

end find_distinct_prime_triples_l226_226522


namespace trace_bag_weight_is_two_l226_226325

-- Given the conditions in the problem
def weight_gordon_bag₁ : ℕ := 3
def weight_gordon_bag₂ : ℕ := 7
def num_traces_bag : ℕ := 5

-- Total weight of Gordon's bags is 10
def total_weight_gordon := weight_gordon_bag₁ + weight_gordon_bag₂

-- Trace's bags weight
def total_weight_trace := total_weight_gordon

-- All conditions must imply this equation is true
theorem trace_bag_weight_is_two :
  (num_traces_bag * 2 = total_weight_trace) → (2 = 2) :=
  by
    sorry

end trace_bag_weight_is_two_l226_226325


namespace quadrant_and_terminal_angle_l226_226103

def alpha : ℝ := -1910 

noncomputable def normalize_angle (α : ℝ) : ℝ := 
  let β := α % 360
  if β < 0 then β + 360 else β

noncomputable def in_quadrant_3 (β : ℝ) : Prop :=
  180 ≤ β ∧ β < 270

noncomputable def equivalent_theta (α : ℝ) (θ : ℝ) : Prop :=
  (α % 360 = θ % 360) ∧ (-720 ≤ θ ∧ θ < 0)

theorem quadrant_and_terminal_angle :
  in_quadrant_3 (normalize_angle alpha) ∧ 
  (equivalent_theta alpha (-110) ∨ equivalent_theta alpha (-470)) :=
by 
  sorry

end quadrant_and_terminal_angle_l226_226103


namespace red_peaches_per_basket_l226_226237

theorem red_peaches_per_basket (R : ℕ) (green_peaches_per_basket : ℕ) (number_of_baskets : ℕ) (total_peaches : ℕ) (h1 : green_peaches_per_basket = 4) (h2 : number_of_baskets = 15) (h3 : total_peaches = 345) : R = 19 :=
by
  sorry

end red_peaches_per_basket_l226_226237


namespace license_plate_palindrome_probability_l226_226434

noncomputable def is_palindrome_prob : ℚ := 775 / 67600

theorem license_plate_palindrome_probability:
  is_palindrome_prob.num + is_palindrome_prob.denom = 68375 := by
  sorry

end license_plate_palindrome_probability_l226_226434


namespace common_difference_unique_l226_226109

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1 : ℝ, ∀ n : ℕ, a n = a1 + n * d

theorem common_difference_unique {a : ℕ → ℝ}
  (h1 : a 2 = 5)
  (h2 : a 3 + a 5 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) ∧ d = -2 :=
sorry

end common_difference_unique_l226_226109


namespace remainder_of_8_pow_2023_l226_226756

theorem remainder_of_8_pow_2023 :
  8^2023 % 100 = 12 :=
sorry

end remainder_of_8_pow_2023_l226_226756


namespace percent_increase_twice_eq_44_percent_l226_226898

variable (P : ℝ) (x : ℝ)

theorem percent_increase_twice_eq_44_percent (h : P * (1 + x)^2 = P * 1.44) : x = 0.2 :=
by sorry

end percent_increase_twice_eq_44_percent_l226_226898


namespace decreasing_function_range_l226_226259

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) (h_decreasing : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 → -1 < x2 ∧ x2 < 1 ∧ x1 > x2 → f x1 < f x2)
  (h_ineq: f (1 - a) < f (3 * a - 1)) : 0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l226_226259


namespace otimes_calculation_l226_226233

def otimes (x y : ℝ) : ℝ := x^2 + y^2

theorem otimes_calculation (x : ℝ) : otimes x (otimes x x) = x^2 + 4 * x^4 :=
by
  sorry

end otimes_calculation_l226_226233


namespace jerome_contacts_total_l226_226843

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end jerome_contacts_total_l226_226843


namespace greater_number_l226_226317

theorem greater_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : x = 35 := 
by sorry

end greater_number_l226_226317


namespace circle_parametric_solution_l226_226739

theorem circle_parametric_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (hx : 4 * Real.cos θ = -2) (hy : 4 * Real.sin θ = 2 * Real.sqrt 3) :
    θ = 2 * Real.pi / 3 :=
sorry

end circle_parametric_solution_l226_226739


namespace breadthOfRectangularPart_l226_226223

variable (b l : ℝ)

def rectangularAreaProblem : Prop :=
  (l * b + (1 / 12) * b * l = 24 * b) ∧ (l - b = 10)

theorem breadthOfRectangularPart :
  rectangularAreaProblem b l → b = 12.15 :=
by
  intros
  sorry

end breadthOfRectangularPart_l226_226223


namespace problem1_problem2_l226_226208

theorem problem1 :
  (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (60 * Real.pi / 180) - abs (1 - Real.sqrt 3) = 3 :=
by 
  sorry

theorem problem2 (x : ℝ) :
  (2 / (x + 1) + 1 = x / (x - 1)) → x = 3 :=
by 
  sorry

end problem1_problem2_l226_226208


namespace impossible_coins_l226_226864

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l226_226864


namespace coordinate_difference_l226_226835

theorem coordinate_difference (m n : ℝ) (h : m = 4 * n + 5) :
  (4 * (n + 0.5) + 5) - m = 2 :=
by
  -- proof skipped
  sorry

end coordinate_difference_l226_226835


namespace breakfast_problem_probability_l226_226774

def are_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem breakfast_problem_probability : 
  ∃ m n : ℕ, are_relatively_prime m n ∧ 
  (1 / 1 * 9 / 11 * 6 / 10 * 1 / 3) * 1 = 9 / 55 ∧ m + n = 64 :=
by
  sorry

end breakfast_problem_probability_l226_226774


namespace frequency_machine_A_frequency_machine_B_chi_square_test_significance_l226_226603

theorem frequency_machine_A (total_A first_class_A : ℕ) (h_total_A: total_A = 200) (h_first_class_A: first_class_A = 150) :
  first_class_A / total_A = 3 / 4 := by
  rw [h_total_A, h_first_class_A]
  norm_num

theorem frequency_machine_B (total_B first_class_B : ℕ) (h_total_B: total_B = 200) (h_first_class_B: first_class_B = 120) :
  first_class_B / total_B = 3 / 5 := by
  rw [h_total_B, h_first_class_B]
  norm_num

theorem chi_square_test_significance (n a b c d : ℕ) (h_n: n = 400) (h_a: a = 150) (h_b: b = 50) 
  (h_c: c = 120) (h_d: d = 80) :
  let K_squared := (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))
  in K_squared > 6.635 := by
  rw [h_n, h_a, h_b, h_c, h_d]
  let num := 400 * (150 * 80 - 50 * 120)^2
  let denom := (150 + 50) * (120 + 80) * (150 + 120) * (50 + 80)
  have : K_squared = num / denom := rfl
  norm_num at this
  sorry

end frequency_machine_A_frequency_machine_B_chi_square_test_significance_l226_226603


namespace inverse_function_solution_l226_226852

noncomputable def f (a b x : ℝ) := 2 / (a * x + b)

theorem inverse_function_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : f a b 2 = 1 / 2) : b = 1 - 2 * a :=
by
  -- Assuming the inverse function condition means f(2) should be evaluated.
  sorry

end inverse_function_solution_l226_226852


namespace area_of_sector_l226_226183

noncomputable def circleAreaAboveXAxisAndRightOfLine : ℝ :=
  let radius := 10
  let area_of_circle := Real.pi * radius^2
  area_of_circle / 4

theorem area_of_sector :
  circleAreaAboveXAxisAndRightOfLine = 25 * Real.pi := sorry

end area_of_sector_l226_226183


namespace correlate_height_weight_l226_226201

-- Define the problems as types
def heightWeightCorrelated : Prop := true
def distanceTimeConstantSpeed : Prop := true
def heightVisionCorrelated : Prop := false
def volumeEdgeLengthCorrelated : Prop := true

-- Define the equivalence for the problem
def correlated : Prop := heightWeightCorrelated

-- Now state that correlated == heightWeightCorrelated
theorem correlate_height_weight : correlated = heightWeightCorrelated :=
by sorry

end correlate_height_weight_l226_226201


namespace count_total_balls_l226_226460

def blue_balls : ℕ := 3
def red_balls : ℕ := 2

theorem count_total_balls : blue_balls + red_balls = 5 :=
by {
  sorry
}

end count_total_balls_l226_226460


namespace problem_six_circles_l226_226657

noncomputable def six_circles_centers : List (ℝ × ℝ) := [(1,1), (1,3), (3,1), (3,3), (5,1), (5,3)]

noncomputable def slope_of_line_dividing_circles := (2 : ℝ)

def gcd_is_1 (p q r : ℕ) : Prop := Nat.gcd (Nat.gcd p q) r = 1

theorem problem_six_circles (p q r : ℕ) (h_gcd : gcd_is_1 p q r)
  (h_line_eq : ∀ x y, y = slope_of_line_dividing_circles * x - 3 → px = qy + r) :
  p^2 + q^2 + r^2 = 14 :=
sorry

end problem_six_circles_l226_226657


namespace range_of_y_coordinate_of_C_l226_226110

-- Define the given parabola equation
def on_parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define the coordinates for point A
def A : (ℝ × ℝ) := (0, 2)

-- Determine if points B and C lies on the parabola
def point_on_parabola (B C : ℝ × ℝ) : Prop :=
  on_parabola B.1 B.2 ∧ on_parabola C.1 C.2

-- Determine if lines AB and BC are perpendicular
def perpendicular_slopes (B C : ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

-- Prove the range for y-coordinate of C
theorem range_of_y_coordinate_of_C (B C : ℝ × ℝ) (h1 : point_on_parabola B C) (h2 : perpendicular_slopes B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := sorry

end range_of_y_coordinate_of_C_l226_226110


namespace existence_of_M_lamps_on_power_of_two_lamps_on_power_of_two_plus_one_l226_226019

open Polynomial

variable {n : ℕ}

-- Condition: n is an integer greater than 1.
-- Question (a): There is a positive integer M(n) such that after M(n) steps, all lamps are ON again.
theorem existence_of_M (h_n : n > 1) : ∃ M, M > 0 ∧ ∀ k < M, (x^k) % (x^n + x^(n-1) + 1) = 1 := 
sorry

variable {k : ℕ}

-- Condition: n has the form 2^k.
-- Question (b): If n = 2^k, then all lamps are ON after n^2 - 1 steps.
theorem lamps_on_power_of_two (h_n : n = 2^k) : (x^(n^2 - 1)) % (x^n + x^(n-1) + 1) = 1 := 
sorry

-- Condition: n has the form 2^k + 1.
-- Question (c): If n = 2^k + 1, then all lamps are ON after n^2 - n + 1 steps.
theorem lamps_on_power_of_two_plus_one (h_n : n = 2^k + 1) : (x^(n^2 - n + 1)) % (x^n + x^(n-1) + 1) = 1 := 
sorry

end existence_of_M_lamps_on_power_of_two_lamps_on_power_of_two_plus_one_l226_226019


namespace functional_equation_solution_l226_226713

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * y * f x) :
  ∀ x : ℝ, f x = 0 := 
sorry

end functional_equation_solution_l226_226713


namespace coordinate_of_M_l226_226025

-- Definition and given conditions
def L : ℚ := 1 / 6
def P : ℚ := 1 / 12

def divides_into_three_equal_parts (L P M N : ℚ) : Prop :=
  M = L + (P - L) / 3 ∧ N = L + 2 * (P - L) / 3

theorem coordinate_of_M (M N : ℚ) 
  (h1 : divides_into_three_equal_parts L P M N) : 
  M = 1 / 9 := 
by 
  sorry
  
end coordinate_of_M_l226_226025


namespace sample_average_l226_226636

theorem sample_average (x : ℝ) 
  (h1 : (1 + 3 + 2 + 5 + x) / 5 = 3) : x = 4 := 
by 
  sorry

end sample_average_l226_226636


namespace dana_jellybeans_l226_226806

noncomputable def jellybeans_in_dana_box (alex_capacity : ℝ) (mul_factor : ℝ) : ℝ :=
  let alex_volume := 1 * 1 * 1.5
  let dana_volume := mul_factor * mul_factor * (mul_factor * 1.5)
  let volume_ratio := dana_volume / alex_volume
  volume_ratio * alex_capacity

theorem dana_jellybeans
  (alex_capacity : ℝ := 150)
  (mul_factor : ℝ := 3) :
  jellybeans_in_dana_box alex_capacity mul_factor = 4050 :=
by
  rw [jellybeans_in_dana_box]
  simp
  sorry

end dana_jellybeans_l226_226806


namespace identify_quadratic_equation_l226_226198

def is_quadratic_one_variable (eq : Type) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ eq = fun x : ℝ => a * x^2 + b * x + c = 0

theorem identify_quadratic_equation (E1 E2 E3 E4 : Type) :
  E1 = (λ x y : ℝ, x^2 - 4 * y = 0) →
  E2 = (λ x : ℝ, x^2 + x + 3 = 0) →
  E3 = (λ x : ℝ, 2 * x = 5) →
  E4 = (λ x : ℝ, x^2 + x⁻¹ - 2 = 0) →
  is_quadratic_one_variable E2 :=
begin
  sorry
end

end identify_quadratic_equation_l226_226198


namespace horner_method_multiplications_and_additions_l226_226354

noncomputable def f (x : ℕ) : ℕ :=
  12 * x ^ 6 + 5 * x ^ 5 + 11 * x ^ 2 + 2 * x + 5

theorem horner_method_multiplications_and_additions (x : ℕ) :
  let multiplications := 6
  let additions := 4
  multiplications = 6 ∧ additions = 4 :=
sorry

end horner_method_multiplications_and_additions_l226_226354


namespace first_class_rate_l226_226993

def pass_rate : ℝ := 0.95
def cond_first_class_rate : ℝ := 0.20

theorem first_class_rate :
  (pass_rate * cond_first_class_rate) = 0.19 :=
by
  -- The proof is omitted as we're not required to provide it.
  sorry

end first_class_rate_l226_226993


namespace evaluate_expression_l226_226365

theorem evaluate_expression :
  let a := 5 ^ 1001
  let b := 6 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end evaluate_expression_l226_226365


namespace train_distance_30_minutes_l226_226068

theorem train_distance_30_minutes (h : ∀ (t : ℝ), 0 < t → (1 / 2) * t = 1 / 2 * t) : 
  (1 / 2) * 30 = 15 :=
by
  sorry

end train_distance_30_minutes_l226_226068


namespace correct_system_of_equations_l226_226227

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : x - y = 5) (h2 : y - (1/2) * x = 5) : 
  (x - y = 5) ∧ (y - (1/2) * x = 5) :=
by { sorry }

end correct_system_of_equations_l226_226227


namespace boys_from_pine_l226_226423

/-- 
Given the following conditions:
1. There are 150 students at the camp.
2. There are 90 boys at the camp.
3. There are 60 girls at the camp.
4. There are 70 students from Maple High School.
5. There are 80 students from Pine High School.
6. There are 20 girls from Oak High School.
7. There are 30 girls from Maple High School.

Prove that the number of boys from Pine High School is 70.
--/
theorem boys_from_pine (total_students boys girls maple_high pine_high oak_girls maple_girls : ℕ)
  (H1 : total_students = 150)
  (H2 : boys = 90)
  (H3 : girls = 60)
  (H4 : maple_high = 70)
  (H5 : pine_high = 80)
  (H6 : oak_girls = 20)
  (H7 : maple_girls = 30) : 
  ∃ pine_boys : ℕ, pine_boys = 70 :=
by
  -- Proof goes here
  sorry

end boys_from_pine_l226_226423


namespace total_age_l226_226623

variable (A B : ℝ)

-- Conditions
def condition1 : Prop := A / B = 3 / 4
def condition2 : Prop := A - 10 = (1 / 2) * (B - 10)

-- Statement
theorem total_age : condition1 A B → condition2 A B → A + B = 35 := by
  sorry

end total_age_l226_226623


namespace coffee_shop_sales_l226_226508

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l226_226508


namespace original_number_of_men_l226_226057

/--A group of men decided to complete a work in 6 days. 
 However, 4 of them became absent, and the remaining men finished the work in 12 days. 
 Given these conditions, we need to prove that the original number of men was 8. --/
theorem original_number_of_men 
  (x : ℕ) -- original number of men
  (h1 : x * 6 = (x - 4) * 12) -- total work remains the same
  : x = 8 := 
sorry

end original_number_of_men_l226_226057


namespace square_b_perimeter_l226_226584

/-- Square A has an area of 121 square centimeters. Square B has a certain perimeter.
  If square B is placed within square A and a random point is chosen within square A,
  the probability that the point is not within square B is 0.8677685950413223.
  Prove the perimeter of square B is 16 centimeters. -/
theorem square_b_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) 
  (h1 : area_A = 121)
  (h2 : prob = 0.8677685950413223)
  (h3 : ∃ (a b : ℝ), area_A = a * a ∧ a * a - b * b = prob * area_A) :
  perimeter_B = 16 :=
sorry

end square_b_perimeter_l226_226584


namespace speed_W_B_l226_226179

-- Definitions for the conditions
def distance_W_B (D : ℝ) := 2 * D
def average_speed := 36
def speed_B_C := 20

-- The problem statement to be verified in Lean
theorem speed_W_B (D : ℝ) (S : ℝ) (h1: distance_W_B D = 2 * D) (h2: S ≠ 0 ∧ D ≠ 0)
(h3: (3 * D) / ((2 * D) / S + D / speed_B_C) = average_speed) : S = 60 := by
sorry

end speed_W_B_l226_226179


namespace probability_at_least_four_of_five_dice_same_number_l226_226664

noncomputable def probability_at_least_four_same : ℚ :=
  (1 / 1296) + (25 / 1296)

theorem probability_at_least_four_of_five_dice_same_number :
  let P := (1 : ℤ) / 1296 + 25 / 1296 in
  P = 13 / 648 :=
by
  let P := (1 : ℤ) / 1296 + 25 / 1296
  have : P = 26 / 1296 := by 
    calc
      P = (1 / 1296) + (25 / 1296) : by sorry -- simplify addition
      ... = 26 / 1296 : by sorry -- combine fractions
  show P = 13 / 648 from by
    calc
      26 / 1296 = 13 / 648 : by sorry -- reduce fraction

end probability_at_least_four_of_five_dice_same_number_l226_226664


namespace no_such_coins_l226_226867

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l226_226867


namespace mandy_total_shirts_l226_226158

-- Condition definitions
def black_packs : ℕ := 6
def black_shirts_per_pack : ℕ := 7
def yellow_packs : ℕ := 8
def yellow_shirts_per_pack : ℕ := 4

theorem mandy_total_shirts : 
  (black_packs * black_shirts_per_pack + yellow_packs * yellow_shirts_per_pack) = 74 :=
by
  sorry

end mandy_total_shirts_l226_226158


namespace b_alone_work_time_l226_226487

def work_rate_combined (a_rate b_rate : ℝ) : ℝ := a_rate + b_rate

theorem b_alone_work_time
  (a_rate b_rate : ℝ)
  (h1 : work_rate_combined a_rate b_rate = 1/16)
  (h2 : a_rate = 1/20) :
  b_rate = 1/80 := by
  sorry

end b_alone_work_time_l226_226487


namespace commute_days_l226_226495

theorem commute_days (a b d e x : ℕ) 
  (h1 : b + e = 12)
  (h2 : a + d = 20)
  (h3 : a + b = 15)
  (h4 : x = a + b + d + e) :
  x = 32 :=
by {
  sorry
}

end commute_days_l226_226495


namespace symmetrical_line_equation_l226_226735

-- Definitions for the conditions
def line_symmetrical (eq1 eq2 : String) : Prop :=
  eq1 = "x - 2y + 3 = 0" ∧ eq2 = "x + 2y + 3 = 0"

-- Prove the statement
theorem symmetrical_line_equation : line_symmetrical "x - 2y + 3 = 0" "x + 2y + 3 = 0" :=
  by
  -- This is just the proof skeleton; the actual proof is not required
  sorry

end symmetrical_line_equation_l226_226735


namespace sum_b4_b6_l226_226678

theorem sum_b4_b6
  (b : ℕ → ℝ)
  (h₁ : ∀ n : ℕ, n > 0 → ∃ d : ℝ, ∀ m : ℕ, m > 0 → (1 / b (m + 1) - 1 / b m) = d)
  (h₂ : b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 90) :
  b 4 + b 6 = 20 := by
  sorry

end sum_b4_b6_l226_226678


namespace tomorrowIsUncertain_l226_226927

-- Definitions as conditions
def isCertainEvent (e : Prop) : Prop := e = true
def isImpossibleEvent (e : Prop) : Prop := e = false
def isInevitableEvent (e : Prop) : Prop := e = true
def isUncertainEvent (e : Prop) : Prop := e ≠ true ∧ e ≠ false

-- Event: Tomorrow will be sunny
def tomorrowWillBeSunny : Prop := sorry -- Placeholder for the actual weather prediction model

-- Problem statement: Prove that "Tomorrow will be sunny" is an uncertain event
theorem tomorrowIsUncertain : isUncertainEvent tomorrowWillBeSunny := sorry

end tomorrowIsUncertain_l226_226927


namespace total_money_made_l226_226512

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l226_226512


namespace range_of_k_l226_226459

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x + k^2 - 1 ≤ 0) ↔ (-Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2) :=
by 
  sorry

end range_of_k_l226_226459


namespace find_a_l226_226272

-- Conditions: x = 5 is a solution to the equation 2x - a = -5
-- We need to prove that a = 15 under these conditions

theorem find_a (x a : ℤ) (h1 : x = 5) (h2 : 2 * x - a = -5) : a = 15 :=
by
  -- We are required to prove the statement, so we skip the proof part here
  sorry

end find_a_l226_226272


namespace larger_number_is_55_l226_226041

theorem larger_number_is_55 (x y : ℤ) (h1 : x + y = 70) (h2 : x = 3 * y + 10) (h3 : y = 15) : x = 55 :=
by
  sorry

end larger_number_is_55_l226_226041


namespace vertical_angles_equal_l226_226165

-- Define what it means for two angles to be vertical angles.
def are_vertical_angles (α β : ℝ) : Prop :=
  ∃ (γ δ : ℝ), α + γ = 180 ∧ β + δ = 180 ∧ γ = β ∧ δ = α

-- The theorem statement:
theorem vertical_angles_equal (α β : ℝ) : are_vertical_angles α β → α = β := 
  sorry

end vertical_angles_equal_l226_226165


namespace tissues_used_l226_226596

-- Define the conditions
def box_tissues : ℕ := 160
def boxes_bought : ℕ := 3
def tissues_left : ℕ := 270

-- Define the theorem that needs to be proven
theorem tissues_used (total_tissues := boxes_bought * box_tissues) : total_tissues - tissues_left = 210 := by
  sorry

end tissues_used_l226_226596


namespace number_of_rows_l226_226071

theorem number_of_rows (r : ℕ) (h1 : ∀ bus : ℕ, bus * (4 * r) = 240) : r = 10 :=
sorry

end number_of_rows_l226_226071


namespace find_x_eq_728_l226_226674

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l226_226674


namespace license_plate_palindrome_probability_l226_226433

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end license_plate_palindrome_probability_l226_226433


namespace distribute_pencils_l226_226094

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end distribute_pencils_l226_226094


namespace neg_ex_proposition_l226_226302

open Classical

theorem neg_ex_proposition :
  ¬ (∃ n : ℕ, n^2 > 2^n) ↔ ∀ n : ℕ, n^2 ≤ 2^n :=
by sorry

end neg_ex_proposition_l226_226302


namespace tissue_pallets_ratio_l226_226224

-- Define the total number of pallets received
def total_pallets : ℕ := 20

-- Define the number of pallets of each type
def paper_towels_pallets : ℕ := total_pallets / 2
def paper_plates_pallets : ℕ := total_pallets / 5
def paper_cups_pallets : ℕ := 1

-- Calculate the number of pallets of tissues
def tissues_pallets : ℕ := total_pallets - (paper_towels_pallets + paper_plates_pallets + paper_cups_pallets)

-- Prove the ratio of pallets of tissues to total pallets is 1/4
theorem tissue_pallets_ratio : (tissues_pallets : ℚ) / total_pallets = 1 / 4 :=
by
  -- Proof goes here
  sorry

end tissue_pallets_ratio_l226_226224


namespace age_solution_l226_226823

theorem age_solution :
  ∃ me you : ℕ, me + you = 63 ∧ 
  ∃ x : ℕ, me = 2 * x ∧ you = x ∧ me = 36 ∧ you = 27 :=
by
  sorry

end age_solution_l226_226823


namespace odd_increasing_min_5_then_neg5_max_on_neg_interval_l226_226006

-- Definitions using the conditions given in the problem statement
variable {f : ℝ → ℝ}

-- Condition 1: f is odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Condition 2: f is increasing on the interval [3, 7]
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f (x) ≤ f (y)

-- Condition 3: Minimum value of f on [3, 7] is 5
def min_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (min_val : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f (x) = min_val

-- Lean statement for the proof problem
theorem odd_increasing_min_5_then_neg5_max_on_neg_interval
  (f_odd: odd_function f)
  (f_increasing: increasing_on_interval f 3 7)
  (min_val: min_value_on_interval f 3 7 5) :
  increasing_on_interval f (-7) (-3) ∧ min_value_on_interval f (-7) (-3) (-5) :=
by sorry

end odd_increasing_min_5_then_neg5_max_on_neg_interval_l226_226006


namespace compute_expression_l226_226656

theorem compute_expression :
  21 * 47 + 21 * 53 = 2100 := 
by
  sorry

end compute_expression_l226_226656


namespace total_savings_l226_226280

-- Define the conditions
def cost_of_two_packs : ℝ := 2.50
def cost_of_single_pack : ℝ := 1.30

-- Define the problem statement
theorem total_savings :
  let price_per_pack_when_in_set := cost_of_two_packs / 2,
      savings_per_pack := cost_of_single_pack - price_per_pack_when_in_set,
      total_packs := 10 * 2,
      total_savings := savings_per_pack * total_packs in
  total_savings = 1 :=
by
  sorry

end total_savings_l226_226280


namespace integral_solutions_l226_226955

/-- 
  Prove that the integral solutions to the equation 
  (m^2 - n^2)^2 = 1 + 16n are exactly (m, n) = (±1, 0), (±4, 3), (±4, 5). 
--/
theorem integral_solutions (m n : ℤ) :
  (m^2 - n^2)^2 = 1 + 16 * n ↔ (m = 1 ∧ n = 0) ∨ (m = -1 ∧ n = 0) ∨
                        (m = 4 ∧ n = 3) ∨ (m = -4 ∧ n = 3) ∨
                        (m = 4 ∧ n = 5) ∨ (m = -4 ∧ n = 5) :=
by
  sorry

end integral_solutions_l226_226955


namespace correct_answer_A_correct_answer_C_correct_answer_D_l226_226381

variable (f g : ℝ → ℝ)

namespace ProofProblem

-- Assume the given conditions
axiom f_eq : ∀ x, f x = 6 - deriv g x
axiom f_compl : ∀ x, f (1 - x) = 6 + deriv g (1 + x)
axiom g_odd : ∀ x, g x - 2 = -(g (-x) - 2)

-- Proving the correct answers
theorem correct_answer_A : g 0 = 2 :=
sorry

theorem correct_answer_C : ∀ x, g (x + 4) = g x :=
sorry

theorem correct_answer_D : f 1 * g 1 + f 3 * g 3 = 24 :=
sorry

end ProofProblem

end correct_answer_A_correct_answer_C_correct_answer_D_l226_226381


namespace complement_union_l226_226544

open Set

namespace ProofFormalization

/-- Declaration of the universal set U, and sets A and B -/
def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

def complement {α : Type*} (s t : Set α) : Set α := t \ s

/-- Theorem statement that proves the complement of A ∪ B with respect to U is {5} -/
theorem complement_union :
  complement (A ∪ B) U = {5} :=
by
  sorry

end ProofFormalization

end complement_union_l226_226544


namespace lcm_18_30_is_90_l226_226473

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l226_226473


namespace trajectory_equation_l226_226989

theorem trajectory_equation :
  ∀ (N : ℝ × ℝ), (∃ (F : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (∃ b : ℝ, P = (0, b)) ∧ 
    (∃ a : ℝ, a ≠ 0 ∧ M = (a, 0)) ∧ 
    (N.fst = -(M.fst) ∧ N.snd = 2 * P.snd) ∧ 
    ((-M.fst) * F.fst + (-(M.snd)) * (-(P.snd)) = 0) ∧ 
    ((-M.fst, -M.snd) + (N.fst, N.snd) = (0,0))) → 
  (N.snd)^2 = 4 * (N.fst) :=
by
  intros N h
  sorry

end trajectory_equation_l226_226989


namespace cannot_be_written_as_square_l226_226871

theorem cannot_be_written_as_square (A B : ℤ) : 
  99999 + 111111 * Real.sqrt 3 ≠ (A + B * Real.sqrt 3) ^ 2 :=
by
  -- Here we would provide the actual mathematical proof
  sorry

end cannot_be_written_as_square_l226_226871


namespace village_population_percentage_l226_226931

theorem village_population_percentage 
  (part : ℝ)
  (whole : ℝ)
  (h_part : part = 8100)
  (h_whole : whole = 9000) : 
  (part / whole) * 100 = 90 :=
by
  sorry

end village_population_percentage_l226_226931


namespace valves_fill_pool_l226_226331

theorem valves_fill_pool
  (a b c d : ℝ)
  (h1 : 1 / a + 1 / b + 1 / c = 1 / 12)
  (h2 : 1 / b + 1 / c + 1 / d = 1 / 15)
  (h3 : 1 / a + 1 / d = 1 / 20) :
  1 / a + 1 / b + 1 / c + 1 / d = 1 / 10 := 
sorry

end valves_fill_pool_l226_226331


namespace firecracker_confiscation_l226_226847

variables
  (F : ℕ)   -- Total number of firecrackers bought
  (R : ℕ)   -- Number of firecrackers remaining after confiscation
  (D : ℕ)   -- Number of defective firecrackers
  (G : ℕ)   -- Number of good firecrackers before setting off half
  (C : ℕ)   -- Number of firecrackers confiscated

-- Define the conditions:
def conditions := 
  F = 48 ∧
  D = R / 6 ∧
  G = 2 * 15 ∧
  R - D = G ∧
  F - R = C

-- The theorem to prove:
theorem firecracker_confiscation (h : conditions F R D G C) : C = 12 := 
  sorry

end firecracker_confiscation_l226_226847


namespace perpendicular_condition_l226_226210

theorem perpendicular_condition (a : ℝ) :
  (a = 1) ↔ (∀ x : ℝ, (a*x + 1 - ((a - 2)*x - 1)) * ((a * x + 1 - (a * x + 1))) = 0) :=
by
  sorry

end perpendicular_condition_l226_226210


namespace fraction_left_handed_non_throwers_is_one_third_l226_226441

theorem fraction_left_handed_non_throwers_is_one_third :
  let total_players := 70
  let throwers := 31
  let right_handed := 57
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  (left_handed_non_throwers : ℝ) / non_throwers = 1 / 3 := by
  sorry

end fraction_left_handed_non_throwers_is_one_third_l226_226441


namespace vector_BC_l226_226675

/-- Given points A (0,1), B (3,2) and vector AC (-4,-3), prove that BC = (-7, -4) -/
theorem vector_BC
  (A B : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (hA : A = (0, 1))
  (hB : B = (3, 2))
  (hAC : AC = (-4, -3)) :
  (AC - (B - A)) = (-7, -4) :=
by
  sorry

end vector_BC_l226_226675


namespace new_people_moved_in_l226_226940

theorem new_people_moved_in (N : ℕ) : (∃ N, 1/16 * (780 - 400 + N : ℝ) = 60) → N = 580 := by
  intros hN
  sorry

end new_people_moved_in_l226_226940


namespace even_mult_expressions_divisible_by_8_l226_226163

theorem even_mult_expressions_divisible_by_8 {a : ℤ} (h : ∃ k : ℤ, a = 2 * k) :
  (8 ∣ a * (a^2 + 20)) ∧ (8 ∣ a * (a^2 - 20)) ∧ (8 ∣ a * (a^2 - 4)) := by
  sorry

end even_mult_expressions_divisible_by_8_l226_226163


namespace slope_angle_of_line_l226_226976

theorem slope_angle_of_line (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : (m^2 + n^2) / m^2 = 4) :
  ∃ θ : ℝ, θ = π / 6 ∨ θ = 5 * π / 6 :=
by
  sorry

end slope_angle_of_line_l226_226976


namespace impossible_all_black_l226_226585

def initial_white_chessboard (n : ℕ) : Prop :=
  n = 0

def move_inverts_three (move : ℕ → ℕ) : Prop :=
  ∀ n, move n = n + 3 ∨ move n = n - 3

theorem impossible_all_black (move : ℕ → ℕ) (n : ℕ) (initial : initial_white_chessboard n) (invert : move_inverts_three move) : ¬ ∃ k, move^[k] n = 64 :=
by sorry

end impossible_all_black_l226_226585


namespace mario_time_on_moving_sidewalk_l226_226221

theorem mario_time_on_moving_sidewalk (d w v : ℝ) (h_walk : d = 90 * w) (h_sidewalk : d = 45 * v) : 
  d / (w + v) = 30 :=
by
  sorry

end mario_time_on_moving_sidewalk_l226_226221


namespace total_legs_in_farm_l226_226595

def num_animals : Nat := 13
def num_chickens : Nat := 4
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

theorem total_legs_in_farm : 
  (num_chickens * legs_per_chicken) + ((num_animals - num_chickens) * legs_per_buffalo) = 44 :=
by
  sorry

end total_legs_in_farm_l226_226595


namespace avg_age_of_children_l226_226699

theorem avg_age_of_children 
  (participants : ℕ) (women : ℕ) (men : ℕ) (children : ℕ)
  (overall_avg_age : ℕ) (avg_age_women : ℕ) (avg_age_men : ℕ)
  (hp : participants = 50) (hw : women = 22) (hm : men = 18) (hc : children = 10)
  (ho : overall_avg_age = 20) (haw : avg_age_women = 24) (ham : avg_age_men = 19) :
  ∃ (avg_age_children : ℕ), avg_age_children = 13 :=
by
  -- Proof will be here.
  sorry

end avg_age_of_children_l226_226699


namespace range_of_m_l226_226978

-- Definitions based on the given conditions
def setA : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def setB (m : ℝ) : Set ℝ := {x | 2 * m - 1 < x ∧ x < m + 1}

-- Lean statement of the problem
theorem range_of_m (m : ℝ) (h : setB m ⊆ setA) : m ≥ -1 :=
sorry  -- proof is not required

end range_of_m_l226_226978


namespace find_pre_tax_remuneration_l226_226503

def pre_tax_remuneration (x : ℝ) : Prop :=
  let taxable_amount := if x <= 4000 then x - 800 else x * 0.8
  let tax_due := taxable_amount * 0.2
  let final_tax := tax_due * 0.7
  final_tax = 280

theorem find_pre_tax_remuneration : ∃ x : ℝ, pre_tax_remuneration x ∧ x = 2800 := by
  sorry

end find_pre_tax_remuneration_l226_226503


namespace income_to_expenditure_ratio_l226_226175

-- Define the constants based on the conditions in step a)
def income : ℕ := 36000
def savings : ℕ := 4000

-- Define the expenditure as a function of income and savings
def expenditure (I S : ℕ) : ℕ := I - S

-- Define the ratio of two natural numbers
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to be proved
theorem income_to_expenditure_ratio : 
  ratio income (expenditure income savings) = 9 / 8 :=
by
  sorry

end income_to_expenditure_ratio_l226_226175


namespace expression_evaluation_l226_226364

noncomputable def evaluate_expression : ℝ :=
  (Real.sin (38 * Real.pi / 180) * Real.sin (38 * Real.pi / 180) 
  + Real.cos (38 * Real.pi / 180) * Real.sin (52 * Real.pi / 180) 
  - Real.tan (15 * Real.pi / 180) ^ 2) / (3 * Real.tan (15 * Real.pi / 180))

theorem expression_evaluation : 
  evaluate_expression = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end expression_evaluation_l226_226364


namespace sufficient_not_necessary_p_q_l226_226382

theorem sufficient_not_necessary_p_q {m : ℝ} 
  (hp : ∀ x, (x^2 - 8*x - 20 ≤ 0) → (-2 ≤ x ∧ x ≤ 10))
  (hq : ∀ x, ((x - 1 - m) * (x - 1 + m) ≤ 0) → (1 - m ≤ x ∧ x ≤ 1 + m))
  (m_pos : 0 < m)  :
  (∀ x, (x - 1 - m) * (x - 1 + m) ≤ 0 → x^2 - 8*x - 20 ≤ 0) ∧ ¬ (∀ x, x^2 - 8*x - 20 ≤ 0 → (x - 1 - m) * (x - 1 + m) ≤ 0) →
  m ≤ 3 :=
sorry

end sufficient_not_necessary_p_q_l226_226382


namespace solution_l226_226967

variable (f g : ℝ → ℝ)

open Real

-- Define f(x) and g(x) as given in the problem
def isSolution (x : ℝ) : Prop :=
  f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x)) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = g x)

-- The theorem we want to prove
theorem solution (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2)
  (h : isSolution f g x) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end solution_l226_226967


namespace correct_word_is_tradition_l226_226906

-- Definitions of the words according to the problem conditions
def tradition : String := "custom, traditional practice"
def balance : String := "equilibrium"
def concern : String := "worry, care about"
def relationship : String := "relation"

-- The sentence to be filled
def sentence (word : String) : String :=
"There’s a " ++ word ++ " in our office that when it’s somebody’s birthday, they bring in a cake for us all to share."

-- The proof problem statement
theorem correct_word_is_tradition :
  ∀ word, (word ≠ tradition) → (sentence word ≠ "There’s a tradition in our office that when it’s somebody’s birthday, they bring in a cake for us all to share.") :=
by sorry

end correct_word_is_tradition_l226_226906


namespace range_of_3x_minus_2y_l226_226966

variable (x y : ℝ)

theorem range_of_3x_minus_2y (h1 : -1 ≤ x + y ∧ x + y ≤ 1) (h2 : 1 ≤ x - y ∧ x - y ≤ 5) :
  ∃ (a b : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ 13 ∧ (3 * x - 2 * y = a ∨ 3 * x - 2 * y = b) :=
by
  sorry

end range_of_3x_minus_2y_l226_226966


namespace polynomial_unique_f_g_l226_226366

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_unique_f_g :
  (∀ x : ℝ, (x^2 + x + 1) * f (x^2 - x + 1) = (x^2 - x + 1) * g (x^2 + x + 1)) →
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x ∧ g x = k * x) :=
sorry

end polynomial_unique_f_g_l226_226366


namespace intersection_of_A_and_B_l226_226430

open Set

def A : Set Int := {x | x + 2 = 0}

def B : Set Int := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by
  sorry

end intersection_of_A_and_B_l226_226430


namespace prob_female_l226_226287

/-- Define basic probabilities for names and their gender associations -/
variables (P_Alexander P_Alexandra P_Yevgeny P_Evgenia P_Valentin P_Valentina P_Vasily P_Vasilisa : ℝ)

-- Define the conditions for the probabilities
axiom h1 : P_Alexander = 3 * P_Alexandra
axiom h2 : P_Yevgeny = 3 * P_Evgenia
axiom h3 : P_Valentin = 1.5 * P_Valentina
axiom h4 : P_Vasily = 49 * P_Vasilisa

/-- The problem we need to prove: the probability that the lot was won by a female student is approximately 0.355 -/
theorem prob_female : 
  let P_female := (P_Alexandra * 1 / 4) + (P_Evgenia * 1 / 4) + (P_Valentina * 1 / 4) + (P_Vasilisa * 1 / 4) in
  abs (P_female - 0.355) < 0.001 :=
sorry

end prob_female_l226_226287


namespace find_linear_function_passing_A_B_l226_226029

-- Conditions
def line_function (k b x : ℝ) : ℝ := k * x + b

theorem find_linear_function_passing_A_B :
  (∃ k b : ℝ, k ≠ 0 ∧ line_function k b 1 = 3 ∧ line_function k b 0 = -2) → 
  ∃ k b : ℝ, k = 5 ∧ b = -2 ∧ ∀ x : ℝ, line_function k b x = 5 * x - 2 :=
by
  -- Proof will be added here
  sorry

end find_linear_function_passing_A_B_l226_226029


namespace flavoring_corn_syrup_ratio_comparison_l226_226142

-- Definitions and conditions derived from the problem
def standard_flavoring_to_water_ratio : ℝ := 1 / 30
def sport_flavoring_to_water_ratio : ℝ := standard_flavoring_to_water_ratio / 2
def sport_water_amount : ℝ := 75
def sport_flavoring_amount : ℝ := sport_water_amount / 60
def sport_corn_syrup_amount : ℝ := 5
def sport_flavoring_to_corn_syrup_ratio : ℝ := sport_flavoring_amount / sport_corn_syrup_amount
def standard_flavoring_to_corn_syrup_ratio : ℝ := 1 / 12

-- The statement to be proved
theorem flavoring_corn_syrup_ratio_comparison :
  sport_flavoring_to_corn_syrup_ratio / standard_flavoring_to_corn_syrup_ratio = 3 :=
by
  have h_sport_flavoring_to_corn_syrup : sport_flavoring_to_corn_syrup_ratio = 1 / 4,
  sorry

  have h_standard_flavoring_to_corn_syrup : standard_flavoring_to_corn_syrup_ratio = 1 / 12,
  sorry

  calc
    sport_flavoring_to_corn_syrup_ratio / standard_flavoring_to_corn_syrup_ratio
        = (1 / 4) / (1 / 12) : by rw [h_sport_flavoring_to_corn_syrup, h_standard_flavoring_to_corn_syrup]
    ... = (1 / 4) * 12 / 1 : by sorry
    ... = 12 / 4 : by sorry
    ... = 3 : by sorry

end flavoring_corn_syrup_ratio_comparison_l226_226142


namespace solution_l226_226670

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * 3 * x + 4

def problem (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : Prop :=
  f a b (-Real.logb 3 3) = 3

theorem solution (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : problem a b m h1 h2 :=
sorry

end solution_l226_226670


namespace triangle_equilateral_l226_226565

variables {A B C : ℝ} -- angles of the triangle
variables {a b c : ℝ} -- sides opposite to the angles

-- Given conditions
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C = c * Real.cos A ∧ (b * b = a * c)

-- The proof goal
theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c → a = b ∧ b = c :=
sorry

end triangle_equilateral_l226_226565


namespace fraction_is_one_fourth_l226_226402

theorem fraction_is_one_fourth (f N : ℝ) 
  (h1 : (1/3) * f * N = 15) 
  (h2 : (3/10) * N = 54) : 
  f = 1/4 :=
by
  sorry

end fraction_is_one_fourth_l226_226402


namespace quadratic_roots_r12_s12_l226_226301

theorem quadratic_roots_r12_s12 (r s : ℝ) (h1 : r + s = 2 * Real.sqrt 3) (h2 : r * s = 1) :
  r^12 + s^12 = 940802 :=
sorry

end quadratic_roots_r12_s12_l226_226301


namespace fraction_equivalence_l226_226336

theorem fraction_equivalence (a b c : ℝ) (h : (c - a) / (c - b) = 1) : 
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by
  sorry

end fraction_equivalence_l226_226336


namespace identical_cubes_probability_l226_226328

/-- Statement of the problem -/
theorem identical_cubes_probability :
  let total_ways := 3^8 * 3^8  -- Total ways to paint two cubes
  let identical_ways := 3 + 72 + 252 + 504  -- Ways for identical appearance after rotation
  (identical_ways : ℝ) / total_ways = 1 / 51814 :=
by
  sorry

end identical_cubes_probability_l226_226328


namespace monkeys_and_apples_l226_226275

theorem monkeys_and_apples
  {x a : ℕ}
  (h1 : a = 3 * x + 6)
  (h2 : 0 < a - 4 * (x - 1) ∧ a - 4 * (x - 1) < 4)
  : (x = 7 ∧ a = 27) ∨ (x = 8 ∧ a = 30) ∨ (x = 9 ∧ a = 33) :=
sorry

end monkeys_and_apples_l226_226275


namespace tommy_gum_given_l226_226436

variable (original_gum : ℕ) (luis_gum : ℕ) (final_total_gum : ℕ)

-- Defining the conditions
def conditions := original_gum = 25 ∧ luis_gum = 20 ∧ final_total_gum = 61

-- The theorem stating that Tommy gave Maria 16 pieces of gum
theorem tommy_gum_given (t_gum : ℕ) (h : conditions original_gum luis_gum final_total_gum) :
  t_gum = final_total_gum - (original_gum + luis_gum) → t_gum = 16 :=
by
  intros h
  sorry

end tommy_gum_given_l226_226436


namespace impossible_coins_l226_226870

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l226_226870


namespace find_m_eq_l226_226682

theorem find_m_eq : 
  (∀ (m : ℝ),
    ((m + 2)^2 + (m + 3)^2 = m^2 + 16 + 4 + (m - 1)^2) →
    m = 2 / 3 ) :=
by
  intros m h
  sorry

end find_m_eq_l226_226682


namespace speed_of_train_A_l226_226329

noncomputable def train_speed_A (V_B : ℝ) (T_A T_B : ℝ) : ℝ :=
  (T_B / T_A) * V_B

theorem speed_of_train_A : train_speed_A 165 9 4 = 73.33 :=
by
  sorry

end speed_of_train_A_l226_226329


namespace inequality_bi_l226_226851

variable {α : Type*} [LinearOrderedField α]

-- Sequence of positive real numbers
variable (a : ℕ → α)
-- Conditions for a_i
variable (ha : ∀ i, i > 0 → i * (a i)^2 ≥ (i + 1) * a (i - 1) * a (i + 1))
-- Positive real numbers x and y
variables (x y : α) (hx : x > 0) (hy : y > 0)
-- Definition of b_i
def b (i : ℕ) : α := x * a i + y * a (i - 1)

theorem inequality_bi (i : ℕ) (hi : i ≥ 2) : i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := 
sorry

end inequality_bi_l226_226851


namespace brians_gas_usage_l226_226515

theorem brians_gas_usage (miles_per_gallon : ℕ) (miles_traveled : ℕ) (gallons_used : ℕ) 
  (h1 : miles_per_gallon = 20) 
  (h2 : miles_traveled = 60) 
  (h3 : gallons_used = miles_traveled / miles_per_gallon) : 
  gallons_used = 3 := 
by 
  rw [h1, h2] at h3 
  exact h3

end brians_gas_usage_l226_226515


namespace license_plate_palindrome_probability_l226_226435

noncomputable def is_palindrome_prob : ℚ := 775 / 67600

theorem license_plate_palindrome_probability:
  is_palindrome_prob.num + is_palindrome_prob.denom = 68375 := by
  sorry

end license_plate_palindrome_probability_l226_226435


namespace inequality_abc_l226_226300

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c)) :=
by
  sorry

end inequality_abc_l226_226300


namespace ratio_of_DN_NF_l226_226692

theorem ratio_of_DN_NF (D E F N : Type) (DE EF DF DN NF p q: ℕ) (h1 : DE = 18) (h2 : EF = 28) (h3 : DF = 34) 
(h4 : DN + NF = DF) (h5 : DN = 22) (h6 : NF = 11) (h7 : p = 101) (h8 : q = 50) : p + q = 151 := 
by 
  sorry

end ratio_of_DN_NF_l226_226692


namespace find_value_of_fraction_l226_226153

noncomputable def a : ℝ := 5 * (Real.sqrt 2) + 7

theorem find_value_of_fraction (h : (20 * a) / (a^2 + 1) = Real.sqrt 2) (h1 : 1 < a) : 
  (14 * a) / (a^2 - 1) = 1 := 
by 
  have h_sqrt : 20 * a = Real.sqrt 2 * a^2 + Real.sqrt 2 := by sorry
  have h_rearrange : Real.sqrt 2 * a^2 - 20 * a + Real.sqrt 2 = 0 := by sorry
  have h_solution : a = 5 * (Real.sqrt 2) + 7 := by sorry
  have h_asquare : a^2 = 99 + 70 * (Real.sqrt 2) := by sorry
  exact sorry

end find_value_of_fraction_l226_226153


namespace estimate_m_value_l226_226620

-- Definition of polynomial P(x) and its roots related to the problem
noncomputable def P (x : ℂ) (a b c : ℂ) : ℂ := x^3 + a * x^2 + b * x + c

-- Statement of the problem in Lean 4
theorem estimate_m_value :
  ∀ (a b c : ℕ),
  a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧
  (∃ z1 z2 z3 : ℂ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧ 
  P z1 a b c = 0 ∧ P z2 a b c = 0 ∧ P z3 a b c = 0) →
  ∃ m : ℕ, m = 8097 :=
sorry

end estimate_m_value_l226_226620


namespace Paul_sold_350_pencils_l226_226859

-- Variables representing conditions
def pencils_per_day : ℕ := 100
def days_in_week : ℕ := 5
def starting_stock : ℕ := 80
def ending_stock : ℕ := 230

-- The total pencils Paul made in a week
def total_pencils_made : ℕ := pencils_per_day * days_in_week

-- The total pencils before selling any
def total_pencils_before_selling : ℕ := total_pencils_made + starting_stock

-- The number of pencils sold is the difference between total pencils before selling and ending stock
def pencils_sold : ℕ := total_pencils_before_selling - ending_stock

theorem Paul_sold_350_pencils :
  pencils_sold = 350 :=
by {
  -- The proof body is replaced with sorry to indicate a placeholder for the proof.
  sorry
}

end Paul_sold_350_pencils_l226_226859


namespace boats_equation_correct_l226_226564

theorem boats_equation_correct (x : ℕ) (h1 : x ≤ 8) (h2 : 4 * x + 6 * (8 - x) = 38) : 
    4 * x + 6 * (8 - x) = 38 :=
by
  sorry

end boats_equation_correct_l226_226564


namespace quadratic_inequality_solution_l226_226242

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end quadratic_inequality_solution_l226_226242


namespace probability_A_middle_BC_adjacent_l226_226028

noncomputable theory

def prob_person_A_middle_BC_adjacent : ℚ :=
  let total_arrangements := fact 9 in
  let valid_arrangements := choose 6 1 * 2! * fact 6 in
  valid_arrangements / total_arrangements

theorem probability_A_middle_BC_adjacent :
  prob_person_A_middle_BC_adjacent = 1 / 42 :=
by
  sorry

end probability_A_middle_BC_adjacent_l226_226028


namespace solve_for_x_l226_226878

theorem solve_for_x (x : ℝ) (h : (x - 5)^3 = (1 / 27)⁻¹) : x = 8 :=
sorry

end solve_for_x_l226_226878


namespace proof_problem_l226_226982

-- Given condition
variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b)
variable (h3 : Real.log a + Real.log (b ^ 2) ≥ 2 * a + (b ^ 2) / 2 - 2)

-- Proof statement
theorem proof_problem : a - 2 * b = 1/2 - 2 * Real.sqrt 2 :=
by
  sorry

end proof_problem_l226_226982


namespace min_sum_of_factors_of_9_factorial_l226_226587

theorem min_sum_of_factors_of_9_factorial (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (h : p * q * r * s = (9!)) : 
  p + q + r + s ≥ 132 := 
sorry

end min_sum_of_factors_of_9_factorial_l226_226587


namespace option_A_is_correct_l226_226918

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end option_A_is_correct_l226_226918


namespace ratio_singers_joined_second_to_remaining_first_l226_226339

-- Conditions
def total_singers : ℕ := 30
def singers_first_verse : ℕ := total_singers / 2
def remaining_after_first : ℕ := total_singers - singers_first_verse
def singers_joined_third_verse : ℕ := 10
def all_singing : ℕ := total_singers

-- Definition for singers who joined in the second verse
def singers_joined_second_verse : ℕ := all_singing - singers_joined_third_verse - singers_first_verse

-- The target proof
theorem ratio_singers_joined_second_to_remaining_first :
  (singers_joined_second_verse : ℚ) / remaining_after_first = 1 / 3 :=
by
  sorry

end ratio_singers_joined_second_to_remaining_first_l226_226339


namespace min_value_theorem_l226_226248

noncomputable def min_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_theorem (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  min_value a b h₀ h₁ h₂ ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_theorem_l226_226248


namespace cos_alpha_in_second_quadrant_l226_226557

variable (α : Real) -- Define the variable α as a Real number (angle in radians)
variable (h1 : α > π / 2 ∧ α < π) -- Condition that α is in the second quadrant
variable (h2 : Real.sin α = 2 / 3) -- Condition that sin(α) = 2/3

theorem cos_alpha_in_second_quadrant (α : Real) (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.sin α = 2 / 3) : Real.cos α = - Real.sqrt (1 - (2 / 3) ^ 2) :=
by
  sorry

end cos_alpha_in_second_quadrant_l226_226557


namespace cistern_filling_time_l226_226489

/-- Define the rates at which the cistern is filled and emptied -/
def fill_rate := (1 : ℚ) / 3
def empty_rate := (1 : ℚ) / 8

/-- Define the net rate of filling when both taps are open -/
def net_rate := fill_rate - empty_rate

/-- Define the volume of the cistern -/
def cistern_volume := (1 : ℚ)

/-- Compute the time to fill the cistern given the net rate -/
def fill_time := cistern_volume / net_rate

theorem cistern_filling_time :
  fill_time = 4.8 := by
sorry

end cistern_filling_time_l226_226489


namespace range_of_t_for_monotonicity_l226_226389

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  (∀ x y ∈ s, x ≤ y → f x ≤ f y) ∨ (∀ x y ∈ s, x ≤ y → f x ≥ f y)

theorem range_of_t_for_monotonicity :
  ∀ t : ℝ, t > -2 → (is_monotonic_on f (Set.Icc (-2) t) ↔ t ∈ Set.Ioo (-2:ℝ) (0:ℝ) ∨ t ∈ Set.Icc (-2:ℝ) 0) :=
by sorry

end range_of_t_for_monotonicity_l226_226389


namespace dinesh_loop_l226_226360

noncomputable def number_of_pentagons (n : ℕ) : ℕ :=
  if (20 * n) % 11 = 0 then 10 else 0

theorem dinesh_loop (n : ℕ) : number_of_pentagons n = 10 :=
by sorry

end dinesh_loop_l226_226360


namespace alpha_value_l226_226262

theorem alpha_value (α : ℝ) (h : (α * (α - 1) * (-1 : ℝ)^(α - 2)) = 4) : α = -4 :=
by
  sorry

end alpha_value_l226_226262


namespace find_number_l226_226326

theorem find_number (x : ℤ) (h : 27 + 2 * x = 39) : x = 6 :=
sorry

end find_number_l226_226326


namespace even_numbers_between_150_and_350_l226_226821

theorem even_numbers_between_150_and_350 : 
  let smallest_even := 152
  let largest_even := 348
  (∃ n, (2 * n > 150) ∧ (2 * n < 350) ∧ (n <= 174)) →
  (∑ n in (finset.range 100).filter (λ n, (2 * (75 + n) > 150) ∧ (2 * (75 + n) < 350)), n) = 99 :=
by
  sorry

end even_numbers_between_150_and_350_l226_226821


namespace perimeter_of_equilateral_triangle_l226_226885

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l226_226885


namespace find_a_value_l226_226117

-- Definitions of conditions
def eq_has_positive_root (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (x / (x - 5) = 3 - (a / (x - 5)))

-- Statement of the theorem
theorem find_a_value (a : ℝ) (h : eq_has_positive_root a) : a = -5 := 
  sorry

end find_a_value_l226_226117


namespace solve_for_x_l226_226310

theorem solve_for_x :
  ∀ x : ℚ, 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) → x = 22 / 5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l226_226310


namespace lcm_of_18_and_30_l226_226469

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l226_226469


namespace lcm_18_30_is_90_l226_226483

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l226_226483


namespace angle_B_in_triangle_l226_226685

theorem angle_B_in_triangle
  (a b c : ℝ)
  (h_area : 2 * (a * c * ((a^2 + c^2 - b^2) / (2 * a * c)).sin) = (a^2 + c^2 - b^2) * (Real.sqrt 3 / 6)) :
  ∃ B : ℝ, B = π / 6 :=
by
  sorry

end angle_B_in_triangle_l226_226685


namespace sequence_general_formula_l226_226689

theorem sequence_general_formula
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (3 / 2) * (a n) - 3) :
  ∀ n, a n = 3 * (2 : ℝ) ^ n :=
by sorry

end sequence_general_formula_l226_226689


namespace scientific_notation_l226_226641

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l226_226641


namespace simplify_polynomial_l226_226606

theorem simplify_polynomial (x : ℝ) :
  3 + 5 * x - 7 * x^2 - 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 = 9 - x - x^2 := 
  by {
  -- placeholder for the proof
  sorry
}

end simplify_polynomial_l226_226606


namespace volume_of_inscribed_sphere_l226_226779

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l226_226779


namespace C_eq_D_at_n_l226_226711

noncomputable def C_n (n : ℕ) : ℝ := 768 * (1 - (1 / (3^n)))
noncomputable def D_n (n : ℕ) : ℝ := (4096 / 5) * (1 - ((-1)^n / (4^n)))
noncomputable def n_ge_1 : ℕ := 4

theorem C_eq_D_at_n : ∀ n ≥ 1, C_n n = D_n n → n = n_ge_1 :=
by
  intro n hn heq
  sorry

end C_eq_D_at_n_l226_226711


namespace inequality_holds_l226_226853

theorem inequality_holds (x y : ℝ) (hx₀ : 0 < x) (hy₀ : 0 < y) (hxy : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 :=
sorry

end inequality_holds_l226_226853


namespace unknown_road_length_l226_226963

/-
  Given the lengths of four roads and the Triangle Inequality condition, 
  prove the length of the fifth road.
  Given lengths: a = 10 km, b = 5 km, c = 8 km, d = 21 km.
-/

theorem unknown_road_length
  (a b c d : ℕ) (h0 : a = 10) (h1 : b = 5) (h2 : c = 8) (h3 : d = 21)
  (x : ℕ) :
  2 < x ∧ x < 18 ∧ 16 < x ∧ x < 26 → x = 17 :=
by
  intros
  sorry

end unknown_road_length_l226_226963


namespace option_C_correct_l226_226485

theorem option_C_correct (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 :=
by
  sorry

end option_C_correct_l226_226485


namespace petya_payment_l226_226421

theorem petya_payment : 
  ∃ (x y : ℕ), 
  (14 * x + 3 * y = 107) ∧ 
  (|x - y| ≤ 5) ∧
  (x + y = 10) := 
sorry

end petya_payment_l226_226421


namespace employee_Y_payment_l226_226322

theorem employee_Y_payment :
  ∀ (X Y Z : ℝ),
  (X + Y + Z = 900) →
  (X = 1.2 * Y) →
  (Z = (X + Y) / 2) →
  (Y = 272.73) :=
begin
  intros X Y Z h1 h2 h3,
  have h4 : 3.3 * Y = 900,
  { rw [←h2, ←h3] at h1,
    linarith,},
  exact (div_eq_iff (by norm_num : 3.3 ≠ 0)).mp h4,
end

end employee_Y_payment_l226_226322


namespace students_basketball_not_table_tennis_l226_226698

theorem students_basketball_not_table_tennis :
  ∀ (total_students basketball_likers table_tennis_likers neither_likers : ℕ),
  total_students = 30 →
  basketball_likers = 15 →
  table_tennis_likers = 10 →
  neither_likers = 8 →
  ∃ (num_both : ℕ), (basketball_likers - num_both) - (total_students - table_tennis_likers - neither_likers - num_both) = 12 :=
by
  intros total_students basketball_likers table_tennis_likers neither_likers h_total h_basketball h_table_tennis h_neither
  use 3
  rw [h_total, h_basketball, h_table_tennis, h_neither]
  linarith

end students_basketball_not_table_tennis_l226_226698


namespace f_is_odd_f_is_decreasing_range_of_m_l226_226393

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) : f (-x) = - f x := by
  sorry

-- Prove that f(x) is decreasing on ℝ
theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  sorry

-- Prove the range of m if f(m-1) + f(2m-1) > 0
theorem range_of_m (m : ℝ) (h : f (m - 1) + f (2 * m - 1) > 0) : m < 2 / 3 := by
  sorry

end f_is_odd_f_is_decreasing_range_of_m_l226_226393


namespace quadrilateral_tile_angles_l226_226986

theorem quadrilateral_tile_angles :
  ∃ a b c d : ℝ, a + b + c + d = 360 ∧ a = 45 ∧ b = 60 ∧ c = 105 ∧ d = 150 := 
by {
  sorry
}

end quadrilateral_tile_angles_l226_226986


namespace total_cost_of_plates_and_cups_l226_226926

theorem total_cost_of_plates_and_cups 
  (P C : ℝ)
  (h : 100 * P + 200 * C = 7.50) :
  20 * P + 40 * C = 1.50 :=
by
  sorry

end total_cost_of_plates_and_cups_l226_226926


namespace right_triangle_sum_of_legs_l226_226010

theorem right_triangle_sum_of_legs (a b : ℝ) (h₁ : a^2 + b^2 = 2500) (h₂ : (1 / 2) * a * b = 600) : a + b = 70 :=
sorry

end right_triangle_sum_of_legs_l226_226010


namespace max_sum_arithmetic_sequence_l226_226178

theorem max_sum_arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ) (h1 : (a + 2) ^ 2 = (a + 8) * (a - 2))
  (h2 : ∀ k, S k = (k * (2 * a + (k - 1) * d)) / 2)
  (h3 : 10 = a) (h4 : -2 = d) :
  S 10 = 90 :=
sorry

end max_sum_arithmetic_sequence_l226_226178


namespace technicians_count_l226_226138

theorem technicians_count (avg_all : ℕ) (avg_tech : ℕ) (avg_other : ℕ) (total_workers : ℕ)
  (h1 : avg_all = 750) (h2 : avg_tech = 900) (h3 : avg_other = 700) (h4 : total_workers = 20) :
  ∃ T O : ℕ, (T + O = total_workers) ∧ ((T * avg_tech + O * avg_other) = total_workers * avg_all) ∧ (T = 5) :=
by
  sorry

end technicians_count_l226_226138


namespace original_price_l226_226795

theorem original_price (a b x : ℝ) (h : (x - a) * 0.60 = b) : x = (5 / 3 * b) + a :=
  sorry

end original_price_l226_226795


namespace percentage_decrease_l226_226513

-- Define conditions and variables
def original_selling_price : ℝ := 659.9999999999994
def profit_rate1 : ℝ := 0.10
def increase_in_selling_price : ℝ := 42
def profit_rate2 : ℝ := 0.30

-- Define the actual proof problem
theorem percentage_decrease (C C_prime : ℝ) 
    (h1 : 1.10 * C = original_selling_price) 
    (h2 : 1.30 * C_prime = original_selling_price + increase_in_selling_price) : 
    ((C - C_prime) / C) * 100 = 10 := 
sorry

end percentage_decrease_l226_226513


namespace total_money_made_l226_226511

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l226_226511


namespace solution_set_inequality_l226_226270

theorem solution_set_inequality (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - 1 / a) > 0) ↔ (a < x ∧ x < 1 / a) :=
by
  sorry

end solution_set_inequality_l226_226270


namespace find_circle_eqn_l226_226098

open Real

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4
def is_tangent (C1 C2 : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop := 
    ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧ P = (x, y)

theorem find_circle_eqn :
  ∃ C : ℝ → ℝ → Prop, is_tangent circle1 C (4, -1) ∧ (∀ x y, C x y ↔ (x - 5)^2 + (y + 1)^2 = 1) := sorry

end find_circle_eqn_l226_226098


namespace max_value_g_eq_3_in_interval_l226_226234

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end max_value_g_eq_3_in_interval_l226_226234


namespace angle_between_adjacent_triangles_l226_226100

-- Define the setup of the problem
def five_nonoverlapping_equilateral_triangles (angles : Fin 5 → ℝ) :=
  ∀ i, angles i = 60

def angles_between_adjacent_triangles (angles : Fin 5 → ℝ) :=
  ∀ i j, i ≠ j → angles i = angles j

-- State the main theorem
theorem angle_between_adjacent_triangles :
  ∀ (angles : Fin 5 → ℝ),
    five_nonoverlapping_equilateral_triangles angles →
    angles_between_adjacent_triangles angles →
    ((360 - 5 * 60) / 5) = 12 :=
by
  intros angles h1 h2
  sorry

end angle_between_adjacent_triangles_l226_226100


namespace range_of_a_l226_226819

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 4 * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 :=
sorry

end range_of_a_l226_226819


namespace find_prime_pairs_l226_226801

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end find_prime_pairs_l226_226801


namespace weighted_arithmetic_geometric_mean_l226_226323
-- Importing required library

-- Definitions of the problem variables and conditions
variables (a b c : ℝ)

-- Non-negative constraints on the lengths of the line segments
variables (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)

-- Problem statement, we need to prove
theorem weighted_arithmetic_geometric_mean :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c)^(1/3) :=
sorry

end weighted_arithmetic_geometric_mean_l226_226323


namespace probability_of_defective_l226_226050

theorem probability_of_defective (p_first_grade p_second_grade : ℝ) (h_fg : p_first_grade = 0.65) (h_sg : p_second_grade = 0.3) : (1 - (p_first_grade + p_second_grade) = 0.05) :=
by
  sorry

end probability_of_defective_l226_226050


namespace greatest_xy_value_l226_226556

theorem greatest_xy_value :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 5 * y = 200 ∧ x * y = 285 :=
by 
  sorry

end greatest_xy_value_l226_226556


namespace number_of_groups_of_three_books_l226_226555

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l226_226555


namespace tom_pie_portion_l226_226229

theorem tom_pie_portion :
  let pie_left := 5 / 8
  let friends := 4
  let portion_per_person := pie_left / friends
  portion_per_person = 5 / 32 := by
  sorry

end tom_pie_portion_l226_226229


namespace eighth_graders_ninth_grader_points_l226_226205

noncomputable def eighth_grader_points (y : ℚ) (x : ℕ) : Prop :=
  x * y + 8 = ((x + 2) * (x + 1)) / 2

theorem eighth_graders (x : ℕ) (y : ℚ) (hx : eighth_grader_points y x) :
  x = 7 ∨ x = 14 :=
sorry

noncomputable def tenth_grader_points (z y : ℚ) (x : ℕ) : Prop :=
  10 * z = 4.5 * y ∧ x * z = y

theorem ninth_grader_points (y : ℚ) (x : ℕ) (z : ℚ)
  (hx : tenth_grader_points z y x) :
  y = 10 :=
sorry

end eighth_graders_ninth_grader_points_l226_226205


namespace F_sum_l226_226535

noncomputable def f : ℝ → ℝ := sorry -- even function f(x)
noncomputable def F (x a c : ℝ) : ℝ := 
  let b := (a + c) / 2
  (x - b) * f (x - b) + 2016

theorem F_sum (a c : ℝ) : F a a c + F c a c = 4032 := 
by {
  sorry
}

end F_sum_l226_226535


namespace cake_recipe_l226_226666

theorem cake_recipe (flour : ℕ) (milk_per_200ml : ℕ) (egg_per_200ml : ℕ) (total_flour : ℕ)
  (h1 : milk_per_200ml = 60)
  (h2 : egg_per_200ml = 1)
  (h3 : total_flour = 800) :
  (total_flour / 200 * milk_per_200ml = 240) ∧ (total_flour / 200 * egg_per_200ml = 4) :=
by
  sorry

end cake_recipe_l226_226666


namespace arithmetic_sequence_d_range_l226_226011

theorem arithmetic_sequence_d_range (d : ℝ) :
  (10 + 4 * d > 0) ∧ (10 + 5 * d < 0) ↔ (-5/2 < d) ∧ (d < -2) :=
by
  sorry

end arithmetic_sequence_d_range_l226_226011


namespace manuscript_acceptance_prob_manuscript_acceptance_distribution_l226_226219

-- Define the probabilities given in the problem
def initial_review_pass_prob := 0.5
def third_review_pass_prob := 0.3

-- Define the events for passing both initial reviews and one initial review and the third review
variable (passes_both_initial : Event)
variable (passes_one_initial_one_third : Event)

-- Define the acceptance event
def acceptance : Event :=
  passes_both_initial ∨ passes_one_initial_one_third

theorem manuscript_acceptance_prob :
  P(acceptance) = 0.4 :=
  sorry

-- Define X as the number of accepted manuscripts out of 4, following binomial distribution
def X : ℕ → ℝ := binomial 4 0.4

theorem manuscript_acceptance_distribution :
  ∀ k, (k ∈ {0, 1, 2, 3, 4}) → (P(X = k) = (nat.choose 4 k) * (0.4)^k * (0.6)^(4 - k)) :=
  sorry

end manuscript_acceptance_prob_manuscript_acceptance_distribution_l226_226219


namespace water_left_in_bucket_l226_226231

theorem water_left_in_bucket :
  let initial_water := 3 / 4
  let poured_out := 1 / 3
  initial_water - poured_out = 5 / 12 :=
by
  sorry

end water_left_in_bucket_l226_226231


namespace find_x_l226_226706

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 119) : x = 39 :=
sorry

end find_x_l226_226706


namespace impossible_to_arrange_distinct_integers_in_grid_l226_226841

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end impossible_to_arrange_distinct_integers_in_grid_l226_226841


namespace time_for_b_alone_l226_226056

theorem time_for_b_alone (A B : ℝ) (h1 : A + B = 1 / 16) (h2 : A = 1 / 24) : B = 1 / 48 :=
by
  sorry

end time_for_b_alone_l226_226056


namespace parallel_vectors_y_value_l226_226397

theorem parallel_vectors_y_value (y : ℝ) :
  let a := (2, 3)
  let b := (4, y)
  ∃ y : ℝ, (2 : ℝ) / 4 = 3 / y → y = 6 :=
sorry

end parallel_vectors_y_value_l226_226397


namespace find_theta_l226_226813

noncomputable def P := (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4))

theorem find_theta
  (theta : ℝ)
  (h_theta_range : 0 ≤ theta ∧ theta < 2 * Real.pi)
  (h_P_theta : P = (Real.sin theta, Real.cos theta)) :
  theta = 7 * Real.pi / 4 :=
sorry

end find_theta_l226_226813


namespace sum_of_variables_is_233_l226_226875

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end sum_of_variables_is_233_l226_226875


namespace grocer_rows_count_l226_226772

theorem grocer_rows_count (n : ℕ) (a d S : ℕ) (h_a : a = 1) (h_d : d = 3) (h_S : S = 225)
  (h_sum : S = n * (2 * a + (n - 1) * d) / 2) : n = 16 :=
by {
  sorry
}

end grocer_rows_count_l226_226772


namespace lcm_18_30_eq_90_l226_226467

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l226_226467


namespace maximize_revenue_l226_226628

def revenue (p : ℝ) : ℝ := 150 * p - 4 * p^2

theorem maximize_revenue : 
  ∃ p, 0 ≤ p ∧ p ≤ 30 ∧ p = 18.75 ∧ (∀ q, 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue 18.75) :=
by
  sorry

end maximize_revenue_l226_226628


namespace audrey_older_than_heracles_l226_226230

variable (A H : ℕ)
variable (hH : H = 10)
variable (hFutureAge : A + 3 = 2 * H)

theorem audrey_older_than_heracles : A - H = 7 :=
by
  have h1 : H = 10 := by assumption
  have h2 : A + 3 = 2 * H := by assumption
  -- Proof is omitted
  sorry

end audrey_older_than_heracles_l226_226230


namespace range_of_k_l226_226116

theorem range_of_k (k : ℝ) : (4 < k ∧ k < 9 ∧ k ≠ 13 / 2) ↔ (k ∈ Set.Ioo 4 (13 / 2) ∪ Set.Ioo (13 / 2) 9) :=
by
  sorry

end range_of_k_l226_226116


namespace inequality_one_inequality_two_l226_226226

-- First Inequality Problem
theorem inequality_one (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) + (1 / d^2) ≤ (1 / (a^2 * b^2 * c^2 * d^2)) :=
sorry

-- Second Inequality Problem
theorem inequality_two (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + (1 / d^3) ≤ (1 / (a^3 * b^3 * c^3 * d^3)) :=
sorry

end inequality_one_inequality_two_l226_226226


namespace sin_double_angle_l226_226528

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l226_226528


namespace bob_distance_when_meet_l226_226762

theorem bob_distance_when_meet (total_distance : ℕ) (yolanda_speed : ℕ) (bob_speed : ℕ) 
    (yolanda_additional_distance : ℕ) (t : ℕ) :
    total_distance = 31 ∧ yolanda_speed = 3 ∧ bob_speed = 4 ∧ yolanda_additional_distance = 3 
    ∧ 7 * t = 28 → 4 * t = 16 := by
    sorry

end bob_distance_when_meet_l226_226762


namespace find_x_l226_226673

theorem find_x (n : ℕ) (x : ℕ) (hn : x = 9^n - 1) (h7 : ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ 7 ∧ p2 ≠ 7 ∧ p1 ≠ p2 ∧ 7 ∣ x ∧ p1 ∣ x ∧ p2 ∣ x) : x = 728 :=
by
  sorry

end find_x_l226_226673


namespace new_cube_weight_l226_226631

-- Define the weight function for a cube given side length and density.
def weight (ρ : ℝ) (s : ℝ) : ℝ := ρ * s^3

-- Given conditions: the weight of the original cube.
axiom original_weight : ∃ ρ s : ℝ, weight ρ s = 7

-- The goal is to prove that a new cube with sides twice as long weighs 56 pounds.
theorem new_cube_weight : 
  (∃ ρ s : ℝ, weight ρ (2 * s) = 56) := by
  sorry

end new_cube_weight_l226_226631


namespace container_unoccupied_volume_l226_226996

noncomputable def unoccupied_volume (side_length_container : ℝ) (side_length_ice : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let volume_container := side_length_container ^ 3
  let volume_water := (3 / 4) * volume_container
  let volume_ice := num_ice_cubes / 2 * side_length_ice ^ 3
  volume_container - (volume_water + volume_ice)

theorem container_unoccupied_volume :
  unoccupied_volume 12 1.5 12 = 411.75 :=
by
  sorry

end container_unoccupied_volume_l226_226996


namespace theater_workshop_l226_226987

-- Definitions of the conditions
def total_participants : ℕ := 120
def cannot_craft_poetry : ℕ := 52
def cannot_perform_painting : ℕ := 75
def not_skilled_in_photography : ℕ := 38
def participants_with_exactly_two_skills : ℕ := 195 - total_participants

-- The theorem stating the problem
theorem theater_workshop :
  participants_with_exactly_two_skills = 75 := by
  sorry

end theater_workshop_l226_226987


namespace percentage_conveyance_l226_226167

def percentage_on_food := 40 / 100
def percentage_on_rent := 20 / 100
def percentage_on_entertainment := 10 / 100
def salary := 12500
def savings := 2500

def total_percentage_spent := percentage_on_food + percentage_on_rent + percentage_on_entertainment
def total_spent := salary - savings
def amount_spent_on_conveyance := total_spent - (salary * total_percentage_spent)
def percentage_spent_on_conveyance := (amount_spent_on_conveyance / salary) * 100

theorem percentage_conveyance : percentage_spent_on_conveyance = 10 :=
by sorry

end percentage_conveyance_l226_226167


namespace survival_rate_is_100_percent_l226_226332

-- Definitions of conditions
def planted_trees : ℕ := 99
def survived_trees : ℕ := 99

-- Definition of survival rate
def survival_rate : ℕ := (survived_trees * 100) / planted_trees

-- Proof statement
theorem survival_rate_is_100_percent : survival_rate = 100 := by
  sorry

end survival_rate_is_100_percent_l226_226332


namespace power_of_thousand_l226_226924

-- Define the notion of googol
def googol := 10^100

-- Prove that 1000^100 is equal to googol^3
theorem power_of_thousand : (1000 ^ 100) = googol^3 := by
  -- proof step to be filled here
  sorry

end power_of_thousand_l226_226924


namespace joy_pencils_count_l226_226149

theorem joy_pencils_count :
  ∃ J, J = 30 ∧ (∃ (pencils_cost_J pencils_cost_C : ℕ), 
  pencils_cost_C = 50 * 4 ∧ pencils_cost_J = pencils_cost_C - 80 ∧ J = pencils_cost_J / 4) := sorry

end joy_pencils_count_l226_226149


namespace range_of_a_l226_226133

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x^2 + 2 * a * x + 2 * a + 3 < 0) ↔ a < -1 :=
sorry

end range_of_a_l226_226133


namespace impossible_grid_arrangement_l226_226838

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end impossible_grid_arrangement_l226_226838


namespace number_of_mismatching_socks_l226_226880

def SteveTotalSocks := 48
def StevePairsMatchingSocks := 11

theorem number_of_mismatching_socks :
  SteveTotalSocks - (StevePairsMatchingSocks * 2) = 26 := by
  sorry

end number_of_mismatching_socks_l226_226880


namespace total_minutes_exercised_l226_226295

-- Defining the conditions
def Javier_minutes_per_day : Nat := 50
def Javier_days : Nat := 10

def Sanda_minutes_day_90 : Nat := 90
def Sanda_days_90 : Nat := 3

def Sanda_minutes_day_75 : Nat := 75
def Sanda_days_75 : Nat := 2

def Sanda_minutes_day_45 : Nat := 45
def Sanda_days_45 : Nat := 4

-- Main statement to prove
theorem total_minutes_exercised : 
  (Javier_minutes_per_day * Javier_days) + 
  (Sanda_minutes_day_90 * Sanda_days_90) +
  (Sanda_minutes_day_75 * Sanda_days_75) +
  (Sanda_minutes_day_45 * Sanda_days_45) = 1100 := by
  sorry

end total_minutes_exercised_l226_226295


namespace sin_double_angle_l226_226126

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
  sorry

end sin_double_angle_l226_226126


namespace giyoon_above_average_subjects_l226_226267

def points_korean : ℕ := 80
def points_mathematics : ℕ := 94
def points_social_studies : ℕ := 82
def points_english : ℕ := 76
def points_science : ℕ := 100
def number_of_subjects : ℕ := 5

def total_points : ℕ := points_korean + points_mathematics + points_social_studies + points_english + points_science
def average_points : ℚ := total_points / number_of_subjects

def count_above_average_points : ℕ := 
  (if points_korean > average_points then 1 else 0) + 
  (if points_mathematics > average_points then 1 else 0) +
  (if points_social_studies > average_points then 1 else 0) +
  (if points_english > average_points then 1 else 0) +
  (if points_science > average_points then 1 else 0)

theorem giyoon_above_average_subjects : count_above_average_points = 2 := by
  sorry

end giyoon_above_average_subjects_l226_226267


namespace base3_20121_to_base10_l226_226083

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end base3_20121_to_base10_l226_226083


namespace sale_in_first_month_is_5000_l226_226771

def sales : List ℕ := [6524, 5689, 7230, 6000, 12557]
def avg_sales : ℕ := 7000
def total_months : ℕ := 6

theorem sale_in_first_month_is_5000 :
  (avg_sales * total_months) - sales.sum = 5000 :=
by sorry

end sale_in_first_month_is_5000_l226_226771


namespace rectangle_area_l226_226749

theorem rectangle_area (w l : ℝ) (h_width : w = 4) (h_perimeter : 2 * l + 2 * w = 30) :
    l * w = 44 :=
by 
  sorry

end rectangle_area_l226_226749


namespace photographer_choice_l226_226763

theorem photographer_choice : 
  (Nat.choose 7 4) + (Nat.choose 7 5) = 56 := 
by 
  sorry

end photographer_choice_l226_226763


namespace quadratic_inequality_solution_l226_226240

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end quadratic_inequality_solution_l226_226240


namespace sum_of_squares_of_solutions_l226_226311

def is_solution (x y : ℝ) : Prop := 
  9 * y^2 - 4 * x^2 = 144 - 48 * x ∧ 
  9 * y^2 + 4 * x^2 = 144 + 18 * x * y

theorem sum_of_squares_of_solutions : 
  (∀ (x y : ℝ), is_solution x y → x^2 + y^2 ∈ {0, 16, 36}) → 
  ∑ (x, y) in {(0, 4), (0, -4), (6, 0)}, x^2 + y^2 = 68 :=
by 
  intros h
  -- the steps can be filled later
  sorry

end sum_of_squares_of_solutions_l226_226311


namespace female_wins_probability_l226_226290

theorem female_wins_probability :
  let p_alexandr := 3 * p_alexandra,
      p_evgeniev := (1 / 3) * p_evgenii,
      p_valentinov := (3 / 2) * p_valentin,
      p_vasilev := 49 * p_vasilisa,
      p_alexandra := 1 / 4,
      p_alexandr := 3 / 4,
      p_evgeniev := 1 / 12,
      p_evgenii := 11 / 12,
      p_valentinov := 3 / 5,
      p_valentin := 2 / 5,
      p_vasilev := 49 / 50,
      p_vasilisa := 1 / 50,
      p_female := 
        (1 / 4) * p_alexandra + 
        (1 / 4) * p_evgeniev + 
        (1 / 4) * p_valentinov + 
        (1 / 4) * p_vasilisa 
  in p_female ≈ 0.355 := 
sorry

end female_wins_probability_l226_226290


namespace locus_of_P_l226_226324

-- Definitions based on conditions
def F : ℝ × ℝ := (2, 0)
def Q (k : ℝ) : ℝ × ℝ := (0, -2 * k)
def T (k : ℝ) : ℝ × ℝ := (-2 * k^2, 0)
def P (k : ℝ) : ℝ × ℝ := (2 * k^2, -4 * k)

-- Theorem statement based on the proof problem
theorem locus_of_P (x y : ℝ) (k : ℝ) (hf : F = (2, 0)) (hq : Q k = (0, -2 * k))
  (ht : T k = (-2 * k^2, 0)) (hp : P k = (2 * k^2, -4 * k)) :
  y^2 = 8 * x :=
sorry

end locus_of_P_l226_226324


namespace girls_collected_more_mushrooms_l226_226837

variables (N I A V : ℝ)

theorem girls_collected_more_mushrooms 
    (h1 : N > I) 
    (h2 : N > A) 
    (h3 : N > V) 
    (h4 : I ≤ N) 
    (h5 : I ≤ A) 
    (h6 : I ≤ V) 
    (h7 : A > V) : 
    N + I > A + V := 
by {
    sorry
}

end girls_collected_more_mushrooms_l226_226837


namespace sum_of_solutions_eq_9_l226_226186

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l226_226186


namespace oil_production_per_capita_l226_226464

-- Defining the given conditions
def oilProduction_West : ℝ := 55084
def oilProduction_NonWest : ℝ := 1480689
def population_NonWest : ℝ := 6900000
def oilProduction_Russia_part : ℝ := 13737.1
def percent_Russia_part : ℝ := 9 / 100
def population_Russia : ℝ := 147000000

-- Defining the correct answers
def perCapita_West : ℝ := oilProduction_West
def perCapita_NonWest : ℝ := oilProduction_NonWest / population_NonWest
def totalOilProduction_Russia : ℝ := oilProduction_Russia_part / percent_Russia_part
def perCapita_Russia : ℝ := totalOilProduction_Russia / population_Russia

-- Statement of the theorem
theorem oil_production_per_capita :
    (perCapita_West = 55084) ∧
    (perCapita_NonWest = 214.59) ∧
    (perCapita_Russia = 1038.33) :=
by sorry

end oil_production_per_capita_l226_226464


namespace no_hexagonal_pyramid_with_equal_edges_l226_226344

theorem no_hexagonal_pyramid_with_equal_edges (edges : ℕ → ℝ)
  (regular_polygon : ℕ → ℝ → Prop)
  (equal_length_edges : ∀ (n : ℕ), regular_polygon n (edges n) → ∀ i j, edges i = edges j)
  (apex_above_centroid : ∀ (n : ℕ) (h : regular_polygon n (edges n)), True) :
  ¬ regular_polygon 6 (edges 6) :=
by
  sorry

end no_hexagonal_pyramid_with_equal_edges_l226_226344


namespace jen_visits_exactly_two_countries_l226_226904

noncomputable def probability_of_visiting_exactly_two_countries (p_chile p_madagascar p_japan p_egypt : ℝ) : ℝ :=
  let p_chile_madagascar := (p_chile * p_madagascar) * (1 - p_japan) * (1 - p_egypt)
  let p_chile_japan := (p_chile * p_japan) * (1 - p_madagascar) * (1 - p_egypt)
  let p_chile_egypt := (p_chile * p_egypt) * (1 - p_madagascar) * (1 - p_japan)
  let p_madagascar_japan := (p_madagascar * p_japan) * (1 - p_chile) * (1 - p_egypt)
  let p_madagascar_egypt := (p_madagascar * p_egypt) * (1 - p_chile) * (1 - p_japan)
  let p_japan_egypt := (p_japan * p_egypt) * (1 - p_chile) * (1 - p_madagascar)
  p_chile_madagascar + p_chile_japan + p_chile_egypt + p_madagascar_japan + p_madagascar_egypt + p_japan_egypt

theorem jen_visits_exactly_two_countries :
  probability_of_visiting_exactly_two_countries 0.4 0.35 0.2 0.15 = 0.2432 :=
by
  sorry

end jen_visits_exactly_two_countries_l226_226904


namespace triangle_perimeter_l226_226590

theorem triangle_perimeter (MN NP MP : ℝ)
  (h1 : MN - NP = 18)
  (h2 : MP = 40)
  (h3 : MN / NP = 28 / 12) : 
  MN + NP + MP = 85 :=
by
  -- Proof is omitted
  sorry

end triangle_perimeter_l226_226590


namespace pencils_distributed_per_container_l226_226096

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end pencils_distributed_per_container_l226_226096


namespace equal_roots_quadratic_k_eq_one_l226_226386

theorem equal_roots_quadratic_k_eq_one
  (k : ℝ)
  (h : ∃ x : ℝ, x^2 - 2 * x + k == 0 ∧ x^2 - 2 * x + k == 0) :
  k = 1 :=
by {
  sorry
}

end equal_roots_quadratic_k_eq_one_l226_226386


namespace lcm_18_30_is_90_l226_226474

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l226_226474


namespace average_salary_l226_226742

theorem average_salary (A B C D E : ℕ) (hA : A = 8000) (hB : B = 5000) (hC : C = 14000) (hD : D = 7000) (hE : E = 9000) :
  (A + B + C + D + E) / 5 = 8800 :=
by
  -- the proof will be inserted here
  sorry

end average_salary_l226_226742


namespace vitamin_D_scientific_notation_l226_226649

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l226_226649


namespace matrix_no_solution_neg_two_l226_226680

-- Define the matrix and vector equation
def matrix_equation (a x y : ℝ) : Prop :=
  (a * x + 2 * y = a + 2) ∧ (2 * x + a * y = 2 * a)

-- Define the condition for no solution
def no_solution_condition (a : ℝ) : Prop :=
  (a/2 = 2/a) ∧ (a/2 ≠ (a + 2) / (2 * a))

-- Theorem stating that a = -2 is the necessary condition for no solution
theorem matrix_no_solution_neg_two (a : ℝ) : no_solution_condition a → a = -2 := by
  sorry

end matrix_no_solution_neg_two_l226_226680


namespace find_roots_l226_226960

theorem find_roots (x : ℝ) : x^2 - 2 * x - 2 / x + 1 / x^2 - 13 = 0 ↔ 
  (x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 ∨ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2) := by
  sorry

end find_roots_l226_226960


namespace nikita_productivity_l226_226600

theorem nikita_productivity 
  (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 5 * x + 3 * y = 11) : 
  y = 2 := 
sorry

end nikita_productivity_l226_226600


namespace equilateral_triangle_M_properties_l226_226102

-- Define the points involved
variables (A B C M P Q R : ℝ)
-- Define distances from M to the sides as given by perpendiculars
variables (d_AP d_BQ d_CR d_PB d_QC d_RA : ℝ)

-- Equilateral triangle assumption and perpendiculars from M to sides
def equilateral_triangle (A B C : ℝ) : Prop := sorry
def perpendicular_from_point (M P R : ℝ) (line : ℝ) : Prop := sorry

-- Problem statement encapsulating the given conditions and what needs to be proved:
theorem equilateral_triangle_M_properties
  (h_triangle: equilateral_triangle A B C)
  (h_perp_AP: perpendicular_from_point M P A B)
  (h_perp_BQ: perpendicular_from_point M Q B C)
  (h_perp_CR: perpendicular_from_point M R C A) :
  (d_AP^2 + d_BQ^2 + d_CR^2 = d_PB^2 + d_QC^2 + d_RA^2) ∧ 
  (d_AP + d_BQ + d_CR = d_PB + d_QC + d_RA) := sorry

end equilateral_triangle_M_properties_l226_226102


namespace avg_visitors_proof_l226_226775

-- Define the constants and conditions
def Sundays_visitors : ℕ := 500
def total_days : ℕ := 30
def avg_visitors_per_day : ℕ := 200

-- Total visits on Sundays within the month
def visits_on_Sundays := 5 * Sundays_visitors

-- Total visitors for the month
def total_visitors := total_days * avg_visitors_per_day

-- Average visitors on other days (Monday to Saturday)
def avg_visitors_other_days : ℕ :=
  (total_visitors - visits_on_Sundays) / (total_days - 5)

-- The theorem stating the problem and corresponding answer
theorem avg_visitors_proof (V : ℕ) 
  (h1 : Sundays_visitors = 500)
  (h2 : total_days = 30)
  (h3 : avg_visitors_per_day = 200)
  (h4 : visits_on_Sundays = 5 * Sundays_visitors)
  (h5 : total_visitors = total_days * avg_visitors_per_day)
  (h6 : avg_visitors_other_days = (total_visitors - visits_on_Sundays) / (total_days - 5))
  : V = 140 :=
by
  -- Proof is not required, just state the theorem
  sorry

end avg_visitors_proof_l226_226775


namespace ordered_sum_ways_l226_226830

theorem ordered_sum_ways (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 2) : 
  ∃ (ways : ℕ), ways = 70 :=
by
  sorry

end ordered_sum_ways_l226_226830


namespace find_value_of_y_l226_226703

noncomputable def angle_sum_triangle (A B C : ℝ) : Prop :=
A + B + C = 180

noncomputable def triangle_ABC : angle_sum_triangle 80 60 x := by
  sorry

noncomputable def triangle_CDE (x y : ℝ) : Prop :=
(x = 40) ∧ (90 + x + y = 180)

theorem find_value_of_y (x y : ℝ) 
  (h1 : angle_sum_triangle 80 60 x)
  (h2 : triangle_CDE x y) : 
  y = 50 := 
by
  sorry

end find_value_of_y_l226_226703


namespace simplify_expression_l226_226196

theorem simplify_expression :
  (Real.sqrt 2 * 2 ^ (1 / 2 : ℝ) + 18 / 3 * 3 - 8 ^ (3 / 2 : ℝ)) = (20 - 16 * Real.sqrt 2) :=
by sorry

end simplify_expression_l226_226196


namespace find_s_squared_l226_226934

-- Define the conditions and entities in Lean
variable (s : ℝ)
def passesThrough (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 / 9) - (x^2 / a^2) = 1

-- State the given conditions as hypotheses
axiom h₀ : passesThrough 0 3 3 1
axiom h₁ : passesThrough 5 (-3) 25 1
axiom h₂ : passesThrough s (-4) 25 1

-- State the theorem we want to prove
theorem find_s_squared : s^2 = 175 / 9 := by
  sorry

end find_s_squared_l226_226934


namespace symmetric_point_l226_226394

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1 : P = (2, 7)) (h2 : 1 * (a - 2) + (b - 7) * (-1) = 0) (h3 : (a + 2) / 2 + (b + 7) / 2 + 1 = 0) :
  (a, b) = (-8, -3) :=
sorry

end symmetric_point_l226_226394


namespace toms_profit_l226_226909

noncomputable def cost_of_flour : Int :=
  let flour_needed := 500
  let bag_size := 50
  let bag_cost := 20
  (flour_needed / bag_size) * bag_cost

noncomputable def cost_of_salt : Int :=
  let salt_needed := 10
  let salt_cost_per_pound := (2 / 10)  -- Represent $0.2 as a fraction to maintain precision with integers in Lean
  salt_needed * salt_cost_per_pound

noncomputable def total_expenses : Int :=
  let flour_cost := cost_of_flour
  let salt_cost := cost_of_salt
  let promotion_cost := 1000
  flour_cost + salt_cost + promotion_cost

noncomputable def revenue_from_tickets : Int :=
  let ticket_price := 20
  let tickets_sold := 500
  tickets_sold * ticket_price

noncomputable def profit : Int :=
  revenue_from_tickets - total_expenses

theorem toms_profit : profit = 8798 :=
  by
    sorry

end toms_profit_l226_226909


namespace average_score_of_seniors_l226_226228

theorem average_score_of_seniors
    (total_students : ℕ)
    (average_score_all : ℚ)
    (num_seniors num_non_seniors : ℕ)
    (mean_score_senior mean_score_non_senior : ℚ)
    (h1 : total_students = 120)
    (h2 : average_score_all = 84)
    (h3 : num_non_seniors = 2 * num_seniors)
    (h4 : mean_score_senior = 2 * mean_score_non_senior)
    (h5 : num_seniors + num_non_seniors = total_students)
    (h6 : num_seniors * mean_score_senior + num_non_seniors * mean_score_non_senior = total_students * average_score_all) :
  mean_score_senior = 126 :=
by
  sorry

end average_score_of_seniors_l226_226228


namespace describe_S_is_two_rays_l226_226299

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ common : ℝ, 
     (common = 5 ∧ (p.1 + 3 = common ∧ p.2 - 2 ≥ common ∨ p.1 + 3 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.1 + 3 ∧ (5 = common ∧ p.2 - 2 ≥ common ∨ 5 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.2 - 2 ∧ (5 = common ∧ p.1 + 3 ≥ common ∨ 5 ≥ common ∧ p.1 + 3 = common))}

theorem describe_S_is_two_rays :
  S = {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 ≥ 7) ∨ (p.2 = 7 ∧ p.1 ≥ 2)} :=
  by
    sorry

end describe_S_is_two_rays_l226_226299


namespace bugs_initial_count_l226_226352

theorem bugs_initial_count (B : ℝ) 
  (h_spray : ∀ (b : ℝ), b * 0.8 = b * (4 / 5)) 
  (h_spiders : ∀ (s : ℝ), s * 7 = 12 * 7) 
  (h_initial_spray_spiders : ∀ (b : ℝ), b * 0.8 - (12 * 7) = 236) 
  (h_final_bugs : 320 / 0.8 = 400) : 
  B = 400 :=
sorry

end bugs_initial_count_l226_226352


namespace election_result_l226_226700

theorem election_result (Vx Vy Vz : ℝ) (Pz : ℝ)
  (h1 : Vx = 3 * (Vx / 3)) (h2 : Vy = 2 * (Vy / 2)) (h3 : Vz = 1 * (Vz / 1))
  (h4 : 0.63 * (Vx + Vy + Vz) = 0.74 * Vx + 0.67 * Vy + Pz * Vz) :
  Pz = 0.22 :=
by
  -- proof steps would go here
  -- sorry to keep the proof incomplete
  sorry

end election_result_l226_226700


namespace probability_standard_bulb_l226_226577

structure FactoryConditions :=
  (P_H1 : ℝ)
  (P_H2 : ℝ)
  (P_H3 : ℝ)
  (P_A_H1 : ℝ)
  (P_A_H2 : ℝ)
  (P_A_H3 : ℝ)

theorem probability_standard_bulb (conditions : FactoryConditions) : 
  conditions.P_H1 = 0.45 → 
  conditions.P_H2 = 0.40 → 
  conditions.P_H3 = 0.15 →
  conditions.P_A_H1 = 0.70 → 
  conditions.P_A_H2 = 0.80 → 
  conditions.P_A_H3 = 0.81 → 
  (conditions.P_H1 * conditions.P_A_H1 + 
   conditions.P_H2 * conditions.P_A_H2 + 
   conditions.P_H3 * conditions.P_A_H3) = 0.7565 :=
by 
  intros h1 h2 h3 a_h1 a_h2 a_h3 
  sorry

end probability_standard_bulb_l226_226577


namespace production_steps_description_l226_226051

-- Definition of the choices
inductive FlowchartType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

-- Conditions
def describeProductionSteps (flowchart : FlowchartType) : Prop :=
flowchart = FlowchartType.ProcessFlowchart

-- The statement to be proved
theorem production_steps_description:
  describeProductionSteps FlowchartType.ProcessFlowchart := 
sorry -- proof to be provided

end production_steps_description_l226_226051


namespace condition_sufficient_not_necessary_l226_226428

theorem condition_sufficient_not_necessary
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B ↔ C)
  (h3 : C → D) :
  (A → D) ∧ ¬(D → A) :=
by
  sorry

end condition_sufficient_not_necessary_l226_226428


namespace inequality_holds_l226_226376

theorem inequality_holds (a : ℝ) (h : a ≠ 0) : |a + (1/a)| ≥ 2 :=
by
  sorry

end inequality_holds_l226_226376


namespace katya_can_write_number_with_conditions_l226_226947

open Finset List

def distinct_digits (l : List ℕ) : Prop :=
  l.nodup ∧ l.length = 10 ∧ (∀ d, d ∈ l → d < 10)

def distinct_absolute_differences (l : List ℕ) : Prop :=
  ∃ diffs : List ℕ, (∀ (i : ℕ), i < l.length - 1 → diffs.nth_le i sorry = |l.nth_le i sorry - l.nth_le (i + 1) sorry|) ∧
  diffs.nodup ∧
  diffs = List.map (λ d, d + 1) (List.range 9)

theorem katya_can_write_number_with_conditions :
  ∃ l : List ℕ, distinct_digits l ∧ distinct_absolute_differences l :=
sorry

end katya_can_write_number_with_conditions_l226_226947


namespace hearing_news_probability_l226_226338

noncomputable def probability_of_hearing_news : ℚ :=
  let broadcast_cycle := 30 -- total time in minutes for each broadcast cycle
  let news_duration := 5  -- duration of each news broadcast in minutes
  news_duration / broadcast_cycle

theorem hearing_news_probability : probability_of_hearing_news = 1 / 6 := by
  sorry

end hearing_news_probability_l226_226338


namespace pizza_problem_l226_226773

theorem pizza_problem (m d : ℕ) :
  (7 * m + 2 * d > 36) ∧ (8 * m + 4 * d < 48) ↔ (m = 5) ∧ (d = 1) := by
  sorry

end pizza_problem_l226_226773


namespace find_number_l226_226613

theorem find_number (x : ℕ) (h : x / 3 = 3) : x = 9 :=
sorry

end find_number_l226_226613


namespace factor_quadratic_l226_226954

theorem factor_quadratic (y : ℝ) : 9 * y ^ 2 - 30 * y + 25 = (3 * y - 5) ^ 2 := by
  sorry

end factor_quadratic_l226_226954


namespace sum_9_to_12_l226_226431

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variables {S : ℕ → ℝ} -- Define the sum function of the sequence

-- Define the conditions given in the problem
def S_4 : ℝ := 8
def S_8 : ℝ := 20

-- The goal is to show that the sum of the 9th to 12th terms is 16
theorem sum_9_to_12 : (a 9) + (a 10) + (a 11) + (a 12) = 16 :=
by
  sorry

end sum_9_to_12_l226_226431


namespace cmp_c_b_a_l226_226377

noncomputable def a : ℝ := 17 / 18
noncomputable def b : ℝ := Real.cos (1 / 3)
noncomputable def c : ℝ := 3 * Real.sin (1 / 3)

theorem cmp_c_b_a:
  c > b ∧ b > a := by
  sorry

end cmp_c_b_a_l226_226377


namespace value_of_a5_l226_226411

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a n * r ^ (m - n) = a m

theorem value_of_a5 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 :=
by
  sorry

end value_of_a5_l226_226411


namespace solve_inequality_l226_226662

theorem solve_inequality (x : ℝ) : (1 / (x + 2) + 4 / (x + 8) ≤ 3 / 4) ↔ ((-8 < x ∧ x ≤ -4) ∨ (-4 ≤ x ∧ x ≤ 4 / 3)) ∧ x ≠ -2 ∧ x ≠ -8 :=
by
  sorry

end solve_inequality_l226_226662


namespace spider_paths_l226_226450

-- Define the grid points and the binomial coefficient calculation.
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- The problem statement
theorem spider_paths : grid_paths 4 3 = 35 := by
  sorry

end spider_paths_l226_226450


namespace solution_part1_solution_part2_l226_226118

variable (f : ℝ → ℝ) (a x m : ℝ)

def problem_statement :=
  (∀ x : ℝ, f x = abs (x - a)) ∧
  (∀ x : ℝ, f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5)

theorem solution_part1 (x : ℝ) (h : problem_statement f a) : a = 2 :=
by
  sorry

theorem solution_part2 (x : ℝ) (h : problem_statement f a) :
  (∀ x : ℝ, f x + f (x + 5) ≥ m) → m ≤ 5 :=
by
  sorry

end solution_part1_solution_part2_l226_226118


namespace range_of_a_l226_226681

variable (a : ℝ)
def A (a : ℝ) := {x : ℝ | x^2 - 2*x + a > 0}

theorem range_of_a (h : 1 ∉ A a) : a ≤ 1 :=
by {
  sorry
}

end range_of_a_l226_226681


namespace range_of_a_for_f_zero_l226_226679

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_f_zero (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 :=
by
  sorry

end range_of_a_for_f_zero_l226_226679


namespace sum_of_solutions_l226_226187

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l226_226187


namespace vishal_investment_more_than_trishul_l226_226463

theorem vishal_investment_more_than_trishul:
  ∀ (V T R : ℝ),
  R = 2100 →
  T = 0.90 * R →
  V + T + R = 6069 →
  ((V - T) / T) * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l226_226463


namespace binomial_parameters_correct_l226_226253

open Probability

variables {n : ℕ} {p : ℝ} (ξ : binomial_distribution n p)
noncomputable def binomial_params :=
  E ξ = 2.4 ∧ variance ξ = 1.44 → (n = 6 ∧ p = 0.4)

theorem binomial_parameters_correct (ξ : binomial_distribution n p) :
  binomial_params ξ :=
by
  sorry

end binomial_parameters_correct_l226_226253


namespace equilateral_triangle_perimeter_l226_226886

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end equilateral_triangle_perimeter_l226_226886


namespace x_minus_y_values_l226_226537

theorem x_minus_y_values (x y : ℝ) 
  (h1 : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) : x - y = -1 ∨ x - y = -7 := 
  sorry

end x_minus_y_values_l226_226537


namespace ant_rest_position_l226_226505

noncomputable def percent_way_B_to_C (s : ℕ) : ℕ :=
  let perimeter := 3 * s
  let distance_traveled := (42 * perimeter) / 100
  let distance_AB := s
  let remaining_distance := distance_traveled - distance_AB
  (remaining_distance * 100) / s

theorem ant_rest_position :
  ∀ (s : ℕ), percent_way_B_to_C s = 26 :=
by
  intros
  unfold percent_way_B_to_C
  sorry

end ant_rest_position_l226_226505


namespace find_a1_l226_226807

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

theorem find_a1
  (h1 : ∀ n : ℕ, a_n 2 * a_n 8 = 2 * a_n 3 * a_n 6)
  (h2 : S_n 5 = -62) :
  a_n 1 = -2 :=
sorry

end find_a1_l226_226807


namespace octahedron_plane_pairs_l226_226979

-- A regular octahedron has 12 edges.
def edges_octahedron : ℕ := 12

-- Each edge determines a plane with 8 other edges.
def pairs_with_each_edge : ℕ := 8

-- The number of unordered pairs of edges that determine a plane
theorem octahedron_plane_pairs : (edges_octahedron * pairs_with_each_edge) / 2 = 48 :=
by
  -- sorry is used to skip the proof
  sorry

end octahedron_plane_pairs_l226_226979


namespace total_customers_served_l226_226945

theorem total_customers_served :
  let hours_ann_becky := 2 * 8 in
  let hours_julia := 6 in
  let total_hours := hours_ann_becky + hours_julia in
  let customers_per_hour := 7 in
  let total_customers := customers_per_hour * total_hours in
  total_customers = 154 :=
by {
  let hours_ann_becky := 2 * 8;
  let hours_julia := 6;
  let total_hours := hours_ann_becky + hours_julia;
  let customers_per_hour := 7;
  let total_customers := customers_per_hour * total_hours;
  sorry
}

end total_customers_served_l226_226945


namespace find_prime_pairs_l226_226800

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end find_prime_pairs_l226_226800


namespace food_coloring_for_hard_candy_l226_226929

theorem food_coloring_for_hard_candy :
  (∀ (food_coloring_per_lollipop total_food_coloring_per_day total_lollipops total_hard_candies : ℕ)
      (food_coloring_total : ℤ),
    food_coloring_per_lollipop = 5 →
    total_lollipops = 100 →
    total_hard_candies = 5 →
    food_coloring_total = 600 →
    food_coloring_per_lollipop * total_lollipops + (total_hard_candies * ?) = food_coloring_total →
    ? = 20)
:=
sorry

end food_coloring_for_hard_candy_l226_226929


namespace hyperbola_equation_l226_226531

noncomputable def hyperbola_eqn : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (b = (1/2) * a) ∧ (a^2 + b^2 = 25) ∧ 
    (∀ x y, (x^2 / (a^2)) - (y^2 / (b^2)) = 1 ↔ (x^2 / 20) - (y^2 / 5) = 1)

theorem hyperbola_equation : hyperbola_eqn := 
  sorry

end hyperbola_equation_l226_226531


namespace jason_money_in_usd_l226_226995

noncomputable def jasonTotalInUSD : ℝ :=
  let init_quarters_value := 49 * 0.25
  let init_dimes_value    := 32 * 0.10
  let init_nickels_value  := 18 * 0.05
  let init_euros_in_usd   := 22.50 * 1.20
  let total_initial       := init_quarters_value + init_dimes_value + init_nickels_value + init_euros_in_usd

  let dad_quarters_value  := 25 * 0.25
  let dad_dimes_value     := 15 * 0.10
  let dad_nickels_value   := 10 * 0.05
  let dad_euros_in_usd    := 12 * 1.20
  let total_additional    := dad_quarters_value + dad_dimes_value + dad_nickels_value + dad_euros_in_usd

  total_initial + total_additional

theorem jason_money_in_usd :
  jasonTotalInUSD = 66 := 
sorry

end jason_money_in_usd_l226_226995


namespace circle_equation_l226_226827

theorem circle_equation (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 1)^2 + (b - 1)^2 = 2 → (a, b) = (0, 0)) ∧
  ((0 - 1)^2 + (0 - 1)^2 = 2) → 
  (x - 1)^2 + (y - 1)^2 = 2 := 
by 
  sorry

end circle_equation_l226_226827


namespace find_number_l226_226594

-- Define the set of positive even integers less than a certain number that contain the digits 5 or 9.
def even_integers_with_5_or_9 (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x, ((x < n) ∧ (x % 2 = 0) ∧
     (x.toDigits 10).any (λ d, d = 5 ∨ d = 9)))
     (Finset.range n)

theorem find_number :
  ∃ n, even_integers_with_5_or_9 n = ({50, 52, 54, 56, 58, 90, 92, 94, 96, 98} : Finset ℕ) ∧
       n = 100 :=
by
  sorry

end find_number_l226_226594


namespace inequality_of_fractions_l226_226156

theorem inequality_of_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (x + y)) + (y / (y + z)) + (z / (z + x)) ≤ 2 := 
by 
  sorry

end inequality_of_fractions_l226_226156


namespace prime_sq_mod_12_l226_226026

theorem prime_sq_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) : (p * p) % 12 = 1 := by
  sorry

end prime_sq_mod_12_l226_226026


namespace frequencies_of_first_class_quality_difference_confidence_l226_226601

section quality_comparison

variables (n a b c d : ℕ)

-- Given conditions
def total_products : ℕ := 400
def machine_a_total : ℕ := 200
def machine_a_first : ℕ := 150
def machine_a_second : ℕ := 50
def machine_b_total : ℕ := 200
def machine_b_first : ℕ := 120
def machine_b_second : ℕ := 80

-- Defining the K^2 calculation formula
def K_squared : ℚ :=
  (total_products * (machine_a_first * machine_b_second - machine_a_second * machine_b_first) ^ 2 : ℚ) /
  ((machine_a_first + machine_a_second) * (machine_b_first + machine_b_second) * (machine_a_first + machine_b_first) * (machine_a_second + machine_b_second))

-- Proof statement for Q1: Frequencies of first-class products
theorem frequencies_of_first_class :
  machine_a_first / machine_a_total = 3 / 4 ∧ 
  machine_b_first / machine_b_total = 3 / 5 := 
sorry

-- Proof statement for Q2: Confidence level of difference in quality
theorem quality_difference_confidence :
  K_squared = 10.256 ∧ 10.256 > 6.635 → 0.99 :=
sorry

end quality_comparison

end frequencies_of_first_class_quality_difference_confidence_l226_226601


namespace ted_and_mike_seeds_l226_226704

noncomputable def ted_morning_seeds (T : ℕ) (mike_morning_seeds : ℕ) (mike_afternoon_seeds : ℕ) (total_seeds : ℕ) : Prop :=
  mike_morning_seeds = 50 ∧
  mike_afternoon_seeds = 60 ∧
  total_seeds = 250 ∧
  T + (mike_afternoon_seeds - 20) + (mike_morning_seeds + mike_afternoon_seeds) = total_seeds ∧
  2 * mike_morning_seeds = T

theorem ted_and_mike_seeds :
  ∃ T : ℕ, ted_morning_seeds T 50 60 250 :=
by {
  sorry
}

end ted_and_mike_seeds_l226_226704


namespace determine_pairs_l226_226089

theorem determine_pairs (a b : ℕ) (h : 2017^a = b^6 - 32 * b + 1) : 
  (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end determine_pairs_l226_226089


namespace exponentiation_product_rule_l226_226614

theorem exponentiation_product_rule (a : ℝ) : (3 * a) ^ 2 = 9 * a ^ 2 :=
by
  sorry

end exponentiation_product_rule_l226_226614


namespace true_propositions_l226_226973

-- Defining the propositions as functions for clarity
def proposition1 (L1 L2 P: Prop) : Prop := 
  (L1 ∧ L2 → P) → (P)

def proposition2 (plane1 plane2 line: Prop) : Prop := 
  (line → (plane1 ∧ plane2)) → (plane1 ∧ plane2)

def proposition3 (L1 L2 L3: Prop) : Prop := 
  (L1 ∧ L2 → L3) → L1

def proposition4 (plane1 plane2 line: Prop) : Prop := 
  (plane1 ∧ plane2 → (line → ¬ (plane1 ∧ plane2)))

-- Assuming the required mathematical hypothesis was valid within our formal system 
theorem true_propositions : proposition2 plane1 plane2 line ∧ proposition4 plane1 plane2 line := 
by sorry

end true_propositions_l226_226973


namespace num_adults_attended_l226_226314

-- Definitions for the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_children : ℕ := 28
def total_revenue : ℕ := 5122

-- The goal is to prove the number of adults who attended the show
theorem num_adults_attended :
  ∃ (A : ℕ), A * ticket_price_adult + num_children * ticket_price_child = total_revenue ∧ A = 183 :=
by
  sorry

end num_adults_attended_l226_226314


namespace unique_intersection_of_A_and_B_l226_226429

-- Define the sets A and B with their respective conditions
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 4 }

def B (r : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - 3)^2 + (y - 4)^2 = r^2 ∧ r > 0 }

-- Define the main theorem statement
theorem unique_intersection_of_A_and_B (r : ℝ) (h : r > 0) : 
  (∃! p, p ∈ A ∧ p ∈ B r) ↔ r = 3 ∨ r = 7 :=
sorry

end unique_intersection_of_A_and_B_l226_226429


namespace problem_statement_l226_226250

theorem problem_statement (m n : ℝ) (h : m + n = 1 / 2 * m * n) : (m - 2) * (n - 2) = 4 :=
by sorry

end problem_statement_l226_226250


namespace volume_of_inscribed_sphere_l226_226780

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l226_226780


namespace geometric_sequence_term_number_l226_226140

theorem geometric_sequence_term_number 
  (a_n : ℕ → ℝ)
  (a1 : ℝ) (q : ℝ) (n : ℕ)
  (h1 : a1 = 1/2)
  (h2 : q = 1/2)
  (h3 : a_n n = 1/32)
  (h4 : ∀ n, a_n n = a1 * (q^(n-1))) :
  n = 5 := 
by
  sorry

end geometric_sequence_term_number_l226_226140


namespace find_capacity_of_second_vessel_l226_226785

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

end find_capacity_of_second_vessel_l226_226785


namespace periodic_length_le_T_l226_226743

noncomputable def purely_periodic (a : ℚ) (T : ℕ) : Prop :=
∃ p : ℤ, a = p / (10^T - 1)

theorem periodic_length_le_T {a b : ℚ} {T : ℕ} 
  (ha : purely_periodic a T) 
  (hb : purely_periodic b T) 
  (hab_sum : purely_periodic (a + b) T)
  (hab_prod : purely_periodic (a * b) T) :
  ∃ Ta Tb : ℕ, Ta ≤ T ∧ Tb ≤ T ∧ purely_periodic a Ta ∧ purely_periodic b Tb := 
sorry

end periodic_length_le_T_l226_226743


namespace find_breadth_of_landscape_l226_226778

theorem find_breadth_of_landscape (L B A : ℕ) 
  (h1 : B = 8 * L)
  (h2 : 3200 = A / 9)
  (h3 : 3200 * 9 = A) :
  B = 480 :=
by 
  sorry

end find_breadth_of_landscape_l226_226778


namespace final_number_l226_226726

variables (crab goat bear cat hen : ℕ)

-- Given conditions
def row4_sum : Prop := 5 * crab = 10
def col5_sum : Prop := 4 * crab + goat = 11
def row2_sum : Prop := 2 * goat + crab + 2 * bear = 16
def col2_sum : Prop := cat + bear + 2 * goat + crab = 13
def col3_sum : Prop := 2 * crab + 2 * hen + goat = 17

-- Theorem statement
theorem final_number
  (hcrab : row4_sum crab)
  (hgoat_col5 : col5_sum crab goat)
  (hbear_row2 : row2_sum crab goat bear)
  (hcat_col2 : col2_sum cat crab bear goat)
  (hhen_col3 : col3_sum crab goat hen) :
  crab = 2 ∧ goat = 3 ∧ bear = 4 ∧ cat = 1 ∧ hen = 5 → (cat * 10000 + hen * 1000 + crab * 100 + bear * 10 + goat = 15243) :=
sorry

end final_number_l226_226726


namespace probability_not_siblings_l226_226285

noncomputable def num_individuals : ℕ := 6
noncomputable def num_pairs : ℕ := num_individuals / 2
noncomputable def total_pairs : ℕ := num_individuals * (num_individuals - 1) / 2
noncomputable def sibling_pairs : ℕ := num_pairs
noncomputable def non_sibling_pairs : ℕ := total_pairs - sibling_pairs

theorem probability_not_siblings :
  (non_sibling_pairs : ℚ) / total_pairs = 4 / 5 := 
by sorry

end probability_not_siblings_l226_226285


namespace find_x_l226_226211

theorem find_x (x : ℝ) (h : 121 * x^4 = 75625) : x = 5 :=
sorry

end find_x_l226_226211


namespace sum_of_three_squares_l226_226850

theorem sum_of_three_squares (n : ℕ) (h_pos : 0 < n) (h_square : ∃ m : ℕ, 3 * n + 1 = m^2) : ∃ x y z : ℕ, n + 1 = x^2 + y^2 + z^2 :=
by
  sorry

end sum_of_three_squares_l226_226850


namespace license_plate_palindrome_probability_l226_226432

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end license_plate_palindrome_probability_l226_226432


namespace find_AD_length_l226_226705

noncomputable def triangle_AD (A B C : Type) (AB AC : ℝ) (ratio_BD_CD : ℝ) (AD : ℝ) : Prop :=
  AB = 13 ∧ AC = 20 ∧ ratio_BD_CD = 3 / 4 → AD = 8 * Real.sqrt 2

theorem find_AD_length {A B C : Type} :
  triangle_AD A B C 13 20 (3/4) (8 * Real.sqrt 2) :=
by
  sorry

end find_AD_length_l226_226705


namespace probability_of_roots_condition_l226_226937

theorem probability_of_roots_condition :
  let k := 6 -- Lower bound of the interval
  let k' := 10 -- Upper bound of the interval
  let interval_length := k' - k
  let satisfying_interval_length := (22 / 3) - 6
  -- The probability that the roots of the quadratic equation satisfy x₁ ≤ 2x₂
  (satisfying_interval_length / interval_length) = (1 / 3) := by
    sorry

end probability_of_roots_condition_l226_226937


namespace sum_of_smallest_x_and_y_l226_226036

theorem sum_of_smallest_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (hx : ∃ k : ℕ, (480 * x) = k * k ∧ ∀ z : ℕ, 0 < z → (480 * z) = k * k → x ≤ z)
  (hy : ∃ n : ℕ, (480 * y) = n * n * n ∧ ∀ z : ℕ, 0 < z → (480 * z) = n * n * n → y ≤ z) :
  x + y = 480 := sorry

end sum_of_smallest_x_and_y_l226_226036


namespace total_paint_correct_l226_226667

-- Define the current gallons of paint he has
def current_paint : ℕ := 36

-- Define the gallons of paint he bought
def bought_paint : ℕ := 23

-- Define the additional gallons of paint he needs
def needed_paint : ℕ := 11

-- The total gallons of paint he needs for finishing touches
def total_paint_needed : ℕ := current_paint + bought_paint + needed_paint

-- The proof statement to show that the total paint needed is 70
theorem total_paint_correct : total_paint_needed = 70 := by
  sorry

end total_paint_correct_l226_226667


namespace p_at_zero_l226_226714

-- Definitions according to given conditions
def p (x : ℝ) : ℝ := sorry  -- Polynomial of degree 6 with specific values

-- Given condition: Degree of polynomial
def degree_p : Prop := (∀ n : ℕ, (n ≤ 6) → p (3 ^ n) = 1 / 3 ^ n)

-- Theorem that needs to be proved
theorem p_at_zero : degree_p → p 0 = 6560 / 2187 := 
by
  sorry

end p_at_zero_l226_226714


namespace max_third_side_of_triangle_l226_226913

theorem max_third_side_of_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : 
  ∃ c : ℕ, c < a + b ∧ c = 17 :=
by 
  sorry

end max_third_side_of_triangle_l226_226913


namespace simplify_expression_l226_226446

theorem simplify_expression (x y : ℝ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 :=
by
  sorry

end simplify_expression_l226_226446


namespace connection_no_values_l226_226658

def scm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem connection_no_values (y : ℕ) (h1 : y < 50)
  (h2 : (scm y 13 : ℚ) / (y * 13 : ℚ) = 3 / 5) : false :=
by
  sorry

end connection_no_values_l226_226658


namespace volume_less_than_1000_l226_226202

noncomputable def volume (x : ℕ) : ℤ :=
(x + 3) * (x - 1) * (x^3 - 20)

theorem volume_less_than_1000 : ∃ (n : ℕ), n = 2 ∧ 
  ∃ x1 x2, x1 ≠ x2 ∧ 0 < x1 ∧ 
  0 < x2 ∧
  volume x1 < 1000 ∧
  volume x2 < 1000 ∧
  ∀ x, 0 < x → volume x < 1000 → (x = x1 ∨ x = x2) :=
by
  sorry

end volume_less_than_1000_l226_226202


namespace pentagon_area_l226_226070

variable (a b c d e : ℕ)
variable (r s : ℕ)

-- Given conditions
axiom H₁: a = 14
axiom H₂: b = 35
axiom H₃: c = 42
axiom H₄: d = 14
axiom H₅: e = 35
axiom H₆: r = 21
axiom H₇: s = 28
axiom H₈: r^2 + s^2 = e^2

-- Question: Prove that the area of the pentagon is 1176
theorem pentagon_area : b * c - (1 / 2) * r * s = 1176 := 
by 
  sorry

end pentagon_area_l226_226070


namespace football_hits_ground_l226_226890

theorem football_hits_ground :
  ∃ t : ℚ, -16 * t^2 + 18 * t + 60 = 0 ∧ 0 < t ∧ t = 41 / 16 :=
by
  sorry

end football_hits_ground_l226_226890


namespace max_leap_years_in_200_years_l226_226653

-- Definitions based on conditions
def leap_year_occurrence (years : ℕ) : ℕ :=
  years / 4

-- Define the problem statement based on the given conditions and required proof
theorem max_leap_years_in_200_years : leap_year_occurrence 200 = 50 := 
by
  sorry

end max_leap_years_in_200_years_l226_226653


namespace jerome_contacts_total_l226_226844

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end jerome_contacts_total_l226_226844


namespace solve_inequality_l226_226457

theorem solve_inequality (x : ℝ) : x + 1 > 3 → x > 2 := 
sorry

end solve_inequality_l226_226457


namespace number_of_possible_values_of_k_l226_226514

-- Define the primary conditions and question
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def quadratic_roots_prime (p q k : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 72 ∧ p * q = k

theorem number_of_possible_values_of_k :
  ¬ ∃ k : ℕ, ∃ p q : ℕ, quadratic_roots_prime p q k :=
by
  sorry

end number_of_possible_values_of_k_l226_226514


namespace error_percent_in_area_l226_226409

theorem error_percent_in_area
  (L W : ℝ)
  (hL : L > 0)
  (hW : W > 0) :
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 0.8 := by
  sorry

end error_percent_in_area_l226_226409


namespace school_allocation_methods_l226_226492

-- Define the conditions
def doctors : ℕ := 3
def nurses : ℕ := 6
def schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- The combinatorial function for binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Verify the number of allocation methods
theorem school_allocation_methods : 
  C doctors doctors_per_school * C nurses nurses_per_school *
  C (doctors - 1) doctors_per_school * C (nurses - 2) nurses_per_school *
  C (doctors - 2) doctors_per_school * C (nurses - 4) nurses_per_school = 540 := 
sorry

end school_allocation_methods_l226_226492


namespace number_of_classes_l226_226721

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by {
  sorry -- Proof goes here
}

end number_of_classes_l226_226721


namespace no_such_coins_l226_226865

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l226_226865


namespace muffin_banana_ratio_l226_226882

variables (m b : ℝ)

theorem muffin_banana_ratio (h1 : 4 * m + 3 * b = x) 
                            (h2 : 2 * (4 * m + 3 * b) = 2 * m + 16 * b) : 
                            m / b = 5 / 3 :=
by sorry

end muffin_banana_ratio_l226_226882


namespace lcm_of_18_and_30_l226_226472

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l226_226472


namespace total_chapters_eq_l226_226964

-- Definitions based on conditions
def days : ℕ := 664
def chapters_per_day : ℕ := 332

-- Theorem to prove the total number of chapters in the book is 220448
theorem total_chapters_eq : (chapters_per_day * days = 220448) :=
by
  sorry

end total_chapters_eq_l226_226964


namespace sales_volume_relation_maximize_profit_l226_226303

-- Define the conditions as given in the problem
def cost_price : ℝ := 6
def sales_data : List (ℝ × ℝ) := [(10, 4000), (11, 3900), (12, 3800)]
def price_range (x : ℝ) : Prop := 6 ≤ x ∧ x ≤ 32

-- Define the functional relationship y in terms of x
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

-- Define the profit function w in terms of x
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - cost_price)

-- Prove that the functional relationship holds within the price range
theorem sales_volume_relation (x : ℝ) (h : price_range x) :
  ∀ (y : ℝ), (x, y) ∈ sales_data → y = sales_volume x := by
  sorry

-- Prove that the profit is maximized when x = 28 and the profit is 48400 yuan
theorem maximize_profit :
  ∃ x, price_range x ∧ x = 28 ∧ profit x = 48400 := by
  sorry

end sales_volume_relation_maximize_profit_l226_226303


namespace interest_rate_is_5_percent_l226_226936

noncomputable def interest_rate_1200_loan (R : ℝ) : Prop :=
  let time := 3.888888888888889
  let principal_1000 := 1000
  let principal_1200 := 1200
  let rate_1000 := 0.03
  let total_interest := 350
  principal_1000 * rate_1000 * time + principal_1200 * (R / 100) * time = total_interest

theorem interest_rate_is_5_percent :
  interest_rate_1200_loan 5 :=
by
  sorry

end interest_rate_is_5_percent_l226_226936


namespace trig_identity_l226_226128

theorem trig_identity (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 3 - Real.cos (2 * x)) : f (Real.cos x) = 3 + Real.cos (2 * x) :=
sorry

end trig_identity_l226_226128


namespace only_A_forms_triangle_l226_226486

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_A_forms_triangle :
  (triangle_inequality 5 6 10) ∧ ¬(triangle_inequality 5 2 9) ∧ ¬(triangle_inequality 5 7 12) ∧ ¬(triangle_inequality 3 4 8) :=
by
  sorry

end only_A_forms_triangle_l226_226486


namespace speed_ratio_l226_226333

variables (H D : ℝ)
variables (duck_leaps hen_leaps : ℕ)
-- hen_leaps and duck_leaps denote the leaps taken by hen and duck respectively

-- conditions given
axiom cond1 : hen_leaps = 6 ∧ duck_leaps = 8
axiom cond2 : 4 * D = 3 * H

-- goal to prove
theorem speed_ratio (H D : ℝ) (hen_leaps duck_leaps : ℕ) (cond1 : hen_leaps = 6 ∧ duck_leaps = 8) (cond2 : 4 * D = 3 * H) : 
  (6 * H) = (8 * D) :=
by
  intros
  sorry

end speed_ratio_l226_226333


namespace multiples_of_4_in_sequence_l226_226002

-- Define the arithmetic sequence terms
def nth_term (a d n : ℤ) : ℤ := a + (n - 1) * d

-- Define the conditions
def cond_1 : ℤ := 200 -- first term
def cond_2 : ℤ := -6 -- common difference
def smallest_term : ℤ := 2

-- Define the count of terms function
def num_terms (a d min : ℤ) : ℤ := (a - min) / -d + 1

-- The total number of terms in the sequence
def total_terms : ℤ := num_terms cond_1 cond_2 smallest_term

-- Define a function to get the ith term that is a multiple of 4
def ith_multiple_of_4 (n : ℤ) : ℤ := cond_1 + 18 * (n - 1)

-- Define the count of multiples of 4 within the given number of terms
def count_multiples_of_4 (total : ℤ) : ℤ := (total / 3) + 1

-- Final theorem statement
theorem multiples_of_4_in_sequence : count_multiples_of_4 total_terms = 12 := sorry

end multiples_of_4_in_sequence_l226_226002


namespace arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l226_226808

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * (-2)) → 
  a 2 = 1 → 
  a 5 = -5 → 
  ∀ n : ℕ, a n = -2 * n + 5 :=
by
  intros h₁ h₂ h₅
  sorry

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (-2)) →
  a 2 = 1 → 
  a 5 = -5 → 
  ∃ n : ℕ, n = 2 ∧ S n = 4 :=
by
  intros hSn h₂ h₅
  sorry

end arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l226_226808


namespace female_sample_count_is_correct_l226_226697

-- Definitions based on the given conditions
def total_students : ℕ := 900
def male_students : ℕ := 500
def sample_size : ℕ := 45
def female_students : ℕ := total_students - male_students
def female_sample_size : ℕ := (female_students * sample_size) / total_students

-- The lean statement to prove
theorem female_sample_count_is_correct : female_sample_size = 20 := 
by 
  -- A placeholder to indicate the proof needs to be filled in
  sorry

end female_sample_count_is_correct_l226_226697


namespace John_works_5_days_a_week_l226_226570

theorem John_works_5_days_a_week
  (widgets_per_hour : ℕ)
  (hours_per_day : ℕ)
  (widgets_per_week : ℕ)
  (H1 : widgets_per_hour = 20)
  (H2 : hours_per_day = 8)
  (H3 : widgets_per_week = 800) :
  widgets_per_week / (widgets_per_hour * hours_per_day) = 5 :=
by
  sorry

end John_works_5_days_a_week_l226_226570


namespace sum_of_roots_quadratic_eq_l226_226190

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l226_226190


namespace correct_description_of_sperm_l226_226504

def sperm_carries_almost_no_cytoplasm (sperm : Type) : Prop := sorry

theorem correct_description_of_sperm : sperm_carries_almost_no_cytoplasm sperm := 
sorry

end correct_description_of_sperm_l226_226504


namespace thickness_of_wall_l226_226001

theorem thickness_of_wall 
    (brick_length cm : ℝ)
    (brick_width cm : ℝ)
    (brick_height cm : ℝ)
    (num_bricks : ℝ)
    (wall_length cm : ℝ)
    (wall_height cm : ℝ)
    (wall_thickness cm : ℝ) :
    brick_length = 25 → 
    brick_width = 11.25 → 
    brick_height = 6 →
    num_bricks = 7200 → 
    wall_length = 900 → 
    wall_height = 600 →
    wall_length * wall_height * wall_thickness = num_bricks * (brick_length * brick_width * brick_height) →
    wall_thickness = 22.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end thickness_of_wall_l226_226001


namespace original_cost_111_l226_226225

theorem original_cost_111 (P : ℝ) (h1 : 0.76 * P * 0.90 = 760) : P = 111 :=
by sorry

end original_cost_111_l226_226225


namespace value_of_a1_l226_226809

def seq (a : ℕ → ℚ) (a_8 : ℚ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 1 / (1 - a n)) ∧ a 8 = 2

theorem value_of_a1 (a : ℕ → ℚ) (h : seq a 2) : a 1 = 1 / 2 :=
  sorry

end value_of_a1_l226_226809


namespace arithmetic_sequence_properties_l226_226622

def a_n (n : ℕ) : ℤ := 2 * n + 1

def S_n (n : ℕ) : ℤ := n * (n + 2)

theorem arithmetic_sequence_properties : 
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) :=
by {
  -- Proof to be filled
  sorry
}

end arithmetic_sequence_properties_l226_226622


namespace shopping_problem_l226_226789

theorem shopping_problem
  (D S H N : ℝ)
  (h1 : (D - (D / 2 - 10)) + (S - 0.85 * S) + (H - (H - 30)) + (N - N) = 120)
  (T_sale : ℝ := (D / 2 - 10) + 0.85 * S + (H - 30) + N) :
  (120 + 0.10 * T_sale = 0.10 * 1200) →
  D + S + H + N = 1200 :=
by
  sorry

end shopping_problem_l226_226789


namespace trains_cross_time_l226_226182

noncomputable def time_to_cross : ℝ := 
  let length_train1 := 110 -- length of the first train in meters
  let length_train2 := 150 -- length of the second train in meters
  let speed_train1 := 60 * 1000 / 3600 -- speed of the first train in meters per second
  let speed_train2 := 45 * 1000 / 3600 -- speed of the second train in meters per second
  let bridge_length := 340 -- length of the bridge in meters
  let total_distance := length_train1 + length_train2 + bridge_length -- total distance to be covered
  let relative_speed := speed_train1 + speed_train2 -- relative speed in meters per second
  total_distance / relative_speed

theorem trains_cross_time :
  abs (time_to_cross - 20.57) < 0.01 :=
sorry

end trains_cross_time_l226_226182


namespace algebra_inequality_l226_226575

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end algebra_inequality_l226_226575


namespace count_valid_six_digit_numbers_l226_226235

open Finset

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits: Finset ℕ := {1, 3, 5}
def even_digits: Finset ℕ := {0, 2, 4}

def permute_even_odd (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 0
  | _ => 6 * 6 + 2 * 6 * 2 -- This handles the cases of alternating digits

theorem count_valid_six_digit_numbers : permute_even_odd 6 = 60 := by
  sorry

end count_valid_six_digit_numbers_l226_226235


namespace total_calories_consumed_l226_226572

-- Definitions for conditions
def calories_per_chip : ℕ := 60 / 10
def extra_calories_per_cheezit := calories_per_chip / 3
def calories_per_cheezit: ℕ := calories_per_chip + extra_calories_per_cheezit
def total_calories_chips : ℕ := 60
def total_calories_cheezits : ℕ := 6 * calories_per_cheezit

-- Main statement to be proved
theorem total_calories_consumed : total_calories_chips + total_calories_cheezits = 108 := by 
  sorry

end total_calories_consumed_l226_226572


namespace number_of_solutions_l226_226042

-- Define the main theorem with the correct conditions
theorem number_of_solutions : 
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℕ), 
     x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₁ + x₂ + x₃ + x₄ + x₅ = 10) 
  → 
  (∃ t : ℕ, t = 70) :=
by 
  sorry

end number_of_solutions_l226_226042


namespace jerry_total_shingles_l226_226415

def roof_length : ℕ := 20
def roof_width : ℕ := 40
def num_roofs : ℕ := 3
def shingles_per_square_foot : ℕ := 8

def area_of_one_side (length width : ℕ) : ℕ :=
  length * width

def total_area_one_roof (area_one_side : ℕ) : ℕ :=
  area_one_side * 2

def total_area_three_roofs (total_area_one_roof : ℕ) : ℕ :=
  total_area_one_roof * num_roofs

def total_shingles_needed (total_area_all_roofs shingles_per_square_foot : ℕ) : ℕ :=
  total_area_all_roofs * shingles_per_square_foot

theorem jerry_total_shingles :
  total_shingles_needed (total_area_three_roofs (total_area_one_roof (area_of_one_side roof_length roof_width))) shingles_per_square_foot = 38400 :=
by
  sorry

end jerry_total_shingles_l226_226415


namespace josh_found_marbles_l226_226418

theorem josh_found_marbles :
  ∃ (F : ℕ), (F + 14 = 23) → (F = 9) :=
by
  existsi 9
  intro h
  linarith

end josh_found_marbles_l226_226418


namespace smallest_mu_real_number_l226_226236

theorem smallest_mu_real_number (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) :
  a^2 + b^2 + c^2 + d^2 ≤ ab + (3/2) * bc + cd :=
sorry

end smallest_mu_real_number_l226_226236


namespace inequality_proof_l226_226306

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) :=
by
  sorry

end inequality_proof_l226_226306


namespace second_car_distance_l226_226327

variables 
  (distance_apart : ℕ := 105)
  (d1 d2 d3 : ℕ := 25) -- distances 25 km, 15 km, 25 km respectively
  (d_road_back : ℕ := 15)
  (final_distance : ℕ := 20)

theorem second_car_distance 
  (car1_total_distance := d1 + d2 + d3 + d_road_back)
  (car2_distance : ℕ) :
  distance_apart - (car1_total_distance + car2_distance) = final_distance →
  car2_distance = 5 :=
sorry

end second_car_distance_l226_226327


namespace domain_of_g_l226_226054

theorem domain_of_g : ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 1 ≠ 0 :=
by
  intro t
  sorry

end domain_of_g_l226_226054


namespace reciprocal_of_negative_one_sixth_l226_226209

theorem reciprocal_of_negative_one_sixth : ∃ x : ℚ, - (1/6) * x = 1 ∧ x = -6 :=
by
  use -6
  constructor
  . sorry -- Need to prove - (1 / 6) * (-6) = 1
  . sorry -- Need to verify x = -6

end reciprocal_of_negative_one_sixth_l226_226209


namespace max_k_for_3_pow_11_as_sum_of_consec_integers_l226_226370

theorem max_k_for_3_pow_11_as_sum_of_consec_integers :
  ∃ k n : ℕ, (3^11 = k * (2 * n + k + 1) / 2) ∧ (k = 486) :=
by
  sorry

end max_k_for_3_pow_11_as_sum_of_consec_integers_l226_226370


namespace gcd_90_250_l226_226091

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end gcd_90_250_l226_226091


namespace movie_sale_price_l226_226770

/-- 
Given the conditions:
- cost of actors: $1200
- number of people: 50
- cost of food per person: $3
- equipment rental costs twice as much as food and actors combined
- profit made: $5950

Prove that the selling price of the movie was $10,000.
-/
theorem movie_sale_price :
  let cost_of_actors := 1200
  let num_people := 50
  let food_cost_per_person := 3
  let total_food_cost := num_people * food_cost_per_person
  let combined_cost := total_food_cost + cost_of_actors
  let equipment_rental_cost := 2 * combined_cost
  let total_cost := cost_of_actors + total_food_cost + equipment_rental_cost
  let profit := 5950
  let sale_price := total_cost + profit
  sale_price = 10000 := 
by
  sorry

end movie_sale_price_l226_226770


namespace first_programmer_loses_l226_226991

noncomputable def programSequence : List ℕ :=
  List.range 1999 |>.map (fun i => 2^i)

def validMove (sequence : List ℕ) (move : List ℕ) : Prop :=
  move.length = 5 ∧ move.all (λ i => i < sequence.length ∧ sequence.get! i > 0)

def applyMove (sequence : List ℕ) (move : List ℕ) : List ℕ :=
  move.foldl
    (λ seq i => seq.set i (seq.get! i - 1))
    sequence

def totalWeight (sequence : List ℕ) : ℕ :=
  sequence.foldl (· + ·) 0

theorem first_programmer_loses : ∀ seq moves,
  seq = programSequence →
  (∀ move, validMove seq move → False) →
  applyMove seq moves = seq →
  totalWeight seq = 2^1999 - 1 :=
by
  intro seq moves h_seq h_valid_move h_apply_move
  sorry

end first_programmer_loses_l226_226991


namespace sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l226_226388

variables (A B C a b c S : ℝ)
variables (h_area : S = (a + b) ^ 2 - c ^ 2) (h_sum : a + b = 4)
variables (h_triangle : ∀ (x : ℝ), x = sin C)

open Real

theorem sin_C_value_proof :
  sin C = 8 / 17 :=
sorry

theorem a2_b2_fraction_proof :
  (a ^ 2 - b ^ 2) / c ^ 2 = sin (A - B) / sin C :=
sorry

theorem sides_sum_comparison :
  a ^ 2 + b ^ 2 + c ^ 2 ≥ 4 * sqrt 3 * S :=
sorry

end sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l226_226388


namespace inequality_solution_set_l226_226523

theorem inequality_solution_set :
  {x : ℝ | (3 - x) * (1 + x) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end inequality_solution_set_l226_226523


namespace smallest_x_for_multiple_l226_226330

theorem smallest_x_for_multiple (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 640 = 2^7 * 5^1) :
  (450 * x) % 640 = 0 ↔ x = 64 :=
sorry

end smallest_x_for_multiple_l226_226330


namespace not_factorial_tail_numbers_lt_1992_l226_226279

noncomputable def factorial_tail_number_count (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + factorial_tail_number_count (n / 5)

theorem not_factorial_tail_numbers_lt_1992 :
  ∃ n, n < 1992 ∧ n = 1992 - (1992 / 5 + (1992 / 25 + (1992 / 125 + (1992 / 625 + 0)))) :=
sorry

end not_factorial_tail_numbers_lt_1992_l226_226279


namespace slices_with_both_l226_226626

theorem slices_with_both (n total_slices pepperoni_slices mushroom_slices other_slices : ℕ)
  (h1 : total_slices = 24) 
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 14)
  (h4 : (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices) :
  n = 5 :=
sorry

end slices_with_both_l226_226626


namespace total_money_made_l226_226510

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l226_226510


namespace pages_read_on_Monday_l226_226856

variable (P : Nat) (W : Nat)
def TotalPages : Nat := P + 12 + W

theorem pages_read_on_Monday :
  (TotalPages P W = 51) → (P = 39) :=
by
  sorry

end pages_read_on_Monday_l226_226856


namespace impossible_to_arrange_distinct_integers_in_grid_l226_226840

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end impossible_to_arrange_distinct_integers_in_grid_l226_226840


namespace final_price_including_tax_l226_226589

noncomputable def increasedPrice (originalPrice : ℝ) (increasePercentage : ℝ) : ℝ :=
  originalPrice + originalPrice * increasePercentage

noncomputable def discountedPrice (increasedPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  increasedPrice - increasedPrice * discountPercentage

noncomputable def finalPrice (discountedPrice : ℝ) (salesTax : ℝ) : ℝ :=
  discountedPrice + discountedPrice * salesTax

theorem final_price_including_tax :
  let originalPrice := 200
  let increasePercentage := 0.30
  let discountPercentage := 0.30
  let salesTax := 0.07
  let incPrice := increasedPrice originalPrice increasePercentage
  let disPrice := discountedPrice incPrice discountPercentage
  finalPrice disPrice salesTax = 194.74 :=
by
  simp [increasedPrice, discountedPrice, finalPrice]
  sorry

end final_price_including_tax_l226_226589


namespace first_person_amount_l226_226168

theorem first_person_amount (A B C : ℕ) (h1 : A = 28) (h2 : B = 72) (h3 : C = 98) (h4 : A + B + C = 198) (h5 : 99 ≤ max (A + B) (B + C) / 2) : 
  A = 28 :=
by
  -- placeholder for proof
  sorry

end first_person_amount_l226_226168


namespace friedEdgeProb_l226_226374

-- Define a data structure for positions on the grid
inductive Pos
| A1 | A2 | A3 | A4
| B1 | B2 | B3 | B4
| C1 | C2 | C3 | C4
| D1 | D2 | D3 | D4
deriving DecidableEq, Repr

-- Define whether a position is an edge square (excluding corners)
def isEdge : Pos → Prop
| Pos.A2 | Pos.A3 | Pos.B1 | Pos.B4 | Pos.C1 | Pos.C4 | Pos.D2 | Pos.D3 => True
| _ => False

-- Define the initial state and max hops
def initialState := Pos.B2
def maxHops := 5

-- Define the recursive probability function (details omitted for brevity)
noncomputable def probabilityEdge (p : Pos) (hops : Nat) : ℚ := sorry

-- The proof problem statement
theorem friedEdgeProb :
  probabilityEdge initialState maxHops = 94 / 256 := sorry

end friedEdgeProb_l226_226374


namespace hcf_of_three_numbers_l226_226174

def hcf (a b : ℕ) : ℕ := gcd a b

theorem hcf_of_three_numbers :
  let a := 136
  let b := 144
  let c := 168
  hcf (hcf a b) c = 8 :=
by
  sorry

end hcf_of_three_numbers_l226_226174


namespace volume_of_inscribed_sphere_l226_226782

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end volume_of_inscribed_sphere_l226_226782


namespace variance_of_data_is_0_02_l226_226108

def data : List ℝ := [10.1, 9.8, 10, 9.8, 10.2]

theorem variance_of_data_is_0_02 (h : (10.1 + 9.8 + 10 + 9.8 + 10.2) / 5 = 10) : 
  (1 / 5) * ((10.1 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10.2 - 10) ^ 2) = 0.02 :=
by
  sorry

end variance_of_data_is_0_02_l226_226108


namespace radicals_like_simplest_forms_l226_226816

theorem radicals_like_simplest_forms (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : a = b + 2) :
  a = 3 ∧ b = 1 :=
by
  sorry

end radicals_like_simplest_forms_l226_226816


namespace find_prime_pairs_l226_226802

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end find_prime_pairs_l226_226802


namespace christine_commission_rate_l226_226232

theorem christine_commission_rate (C : ℝ) (H1 : 24000 ≠ 0) (H2 : 0.4 * (C / 100 * 24000) = 1152) :
  C = 12 :=
by
  sorry

end christine_commission_rate_l226_226232


namespace problem_1_and_2_problem_1_infinite_solutions_l226_226491

open Nat

theorem problem_1_and_2 (k : ℕ) (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2 = k * a * b * c) →
  (k = 1 ∨ k = 3) :=
sorry

theorem problem_1_infinite_solutions (k : ℕ) (h_k : k = 1 ∨ k = 3) :
  ∃ (a_n b_n c_n : ℕ) (n : ℕ), 
  a_n > 0 ∧ b_n > 0 ∧ c_n > 0 ∧
  (a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n) ∧
  ∀ x y : ℕ, (x = a_n ∧ y = b_n) ∨ (x = a_n ∧ y = c_n) ∨ (x = b_n ∧ y = c_n) →
    ∃ p q : ℕ, x * y = p^2 + q^2 :=
sorry

end problem_1_and_2_problem_1_infinite_solutions_l226_226491


namespace john_buys_1000_balloons_l226_226571

-- Define conditions
def balloon_volume : ℕ := 10
def tank_volume : ℕ := 500
def num_tanks : ℕ := 20

-- Define the total volume of gas
def total_gas_volume : ℕ := num_tanks * tank_volume

-- Define the number of balloons
def num_balloons : ℕ := total_gas_volume / balloon_volume

-- Prove that the number of balloons is 1,000
theorem john_buys_1000_balloons : num_balloons = 1000 := by
  sorry

end john_buys_1000_balloons_l226_226571


namespace apple_distribution_ways_l226_226651

-- Definitions based on conditions
def distribute_apples (a b c : ℕ) : Prop := a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

-- Non-negative integer solutions to a' + b' + c' = 21
def num_solutions := Nat.choose 23 2

-- Theorem to prove
theorem apple_distribution_ways : distribute_apples 10 10 10 → num_solutions = 253 :=
by
  intros
  sorry

end apple_distribution_ways_l226_226651


namespace greatest_integer_jean_thinks_of_l226_226145

theorem greatest_integer_jean_thinks_of :
  ∃ n : ℕ, n < 150 ∧ (∃ a : ℤ, n + 2 = 9 * a) ∧ (∃ b : ℤ, n + 3 = 11 * b) ∧ n = 142 :=
by
  sorry

end greatest_integer_jean_thinks_of_l226_226145


namespace integers_sum_eighteen_l226_226900

theorem integers_sum_eighteen (a b : ℕ) (h₀ : a ≠ b) (h₁ : a < 20) (h₂ : b < 20) (h₃ : Nat.gcd a b = 1) 
(h₄ : a * b + a + b = 95) : a + b = 18 :=
by
  sorry

end integers_sum_eighteen_l226_226900


namespace parallel_lines_slope_equal_l226_226405

theorem parallel_lines_slope_equal (m : ℝ) : 
  (∃ m : ℝ, -(m+4)/(m+2) = -(m+2)/(m+1)) → m = 0 := 
by
  sorry

end parallel_lines_slope_equal_l226_226405


namespace math_problem_proof_l226_226080

-- Define the fractions involved
def frac1 : ℚ := -49
def frac2 : ℚ := 4 / 7
def frac3 : ℚ := -8 / 7

-- The original expression
def original_expr : ℚ :=
  frac1 * frac2 - frac2 / frac3

-- Declare the theorem to be proved
theorem math_problem_proof : original_expr = -27.5 :=
by
  sorry

end math_problem_proof_l226_226080


namespace vitamin_D_scientific_notation_l226_226647

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l226_226647


namespace Kirill_is_69_l226_226420

/-- Kirill is 14 centimeters shorter than his brother.
    Their sister's height is twice the height of Kirill.
    Their cousin's height is 3 centimeters more than the sister's height.
    Together, their heights equal 432 centimeters.
    We aim to prove that Kirill's height is 69 centimeters.
-/
def Kirill_height (K : ℕ) : Prop :=
  let brother_height := K + 14
  let sister_height := 2 * K
  let cousin_height := 2 * K + 3
  K + brother_height + sister_height + cousin_height = 432

theorem Kirill_is_69 {K : ℕ} (h : Kirill_height K) : K = 69 :=
by
  sorry

end Kirill_is_69_l226_226420


namespace swim_ratio_l226_226069

theorem swim_ratio
  (V_m : ℝ) (h1 : V_m = 4.5)
  (V_s : ℝ) (h2 : V_s = 1.5)
  (V_u : ℝ) (h3 : V_u = V_m - V_s)
  (V_d : ℝ) (h4 : V_d = V_m + V_s)
  (T_u T_d : ℝ) (h5 : T_u / T_d = V_d / V_u) :
  T_u / T_d = 2 :=
by {
  sorry
}

end swim_ratio_l226_226069


namespace sum_of_solutions_l226_226189

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l226_226189


namespace camera_sticker_price_l226_226125

theorem camera_sticker_price (p : ℝ)
  (h1 : p > 0)
  (hx : ∀ x, x = 0.80 * p - 50)
  (hy : ∀ y, y = 0.65 * p)
  (hs : 0.80 * p - 50 = 0.65 * p - 40) :
  p = 666.67 :=
by sorry

end camera_sticker_price_l226_226125


namespace petya_payment_l226_226422

theorem petya_payment (x y : ℤ) (h₁ : 14 * x + 3 * y = 107) (h₂ : |x - y| ≤ 5) : x + y = 10 :=
sorry

end petya_payment_l226_226422


namespace impossible_grid_arrangement_l226_226839

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end impossible_grid_arrangement_l226_226839


namespace find_x_with_three_prime_divisors_l226_226672

def x_with_conditions (n : Nat) (x : Nat) : Prop :=
  x = 9^n - 1 ∧
  nat.factors x ∧
  nat.factors x.count 7 = 1

theorem find_x_with_three_prime_divisors (n : Nat) (x : Nat) :
  x_with_conditions n x → x = 728 :=
by
  sorry

end find_x_with_three_prime_divisors_l226_226672


namespace mother_l226_226616

def age_relations (P M : ℕ) : Prop :=
  P = (2 * M) / 5 ∧ P + 10 = (M + 10) / 2

theorem mother's_present_age (P M : ℕ) (h : age_relations P M) : M = 50 :=
by
  sorry

end mother_l226_226616


namespace buying_pets_l226_226342

theorem buying_pets {puppies kittens hamsters birds : ℕ} :
(∃ pets : ℕ, pets = 12 * 8 * 10 * 5 * 4 * 3 * 2) ∧ 
puppies = 12 ∧ kittens = 8 ∧ hamsters = 10 ∧ birds = 5 → 
12 * 8 * 10 * 5 * 4 * 3 * 2 = 115200 :=
by
  intros h
  sorry

end buying_pets_l226_226342


namespace equal_total_areas_of_checkerboard_pattern_l226_226362

-- Definition representing the convex quadrilateral and its subdivisions
structure ConvexQuadrilateral :=
  (A B C D : ℝ × ℝ) -- vertices of the quadrilateral

-- Predicate indicating the subdivision and coloring pattern
inductive CheckerboardColor
  | Black
  | White

-- Function to determine the area of the resulting smaller quadrilateral
noncomputable def area_of_subquadrilateral 
  (quad : ConvexQuadrilateral) 
  (subdivision : ℕ) -- subdivision factor
  (color : CheckerboardColor) 
  : ℝ := -- returns the area based on the subdivision and color
  -- Simplified implementation of area calculation
  -- (detailed geometric computation should replace this placeholder)
  sorry

-- Function to determine the total area of quadrilaterals of a given color
noncomputable def total_area_of_color 
  (quad : ConvexQuadrilateral) 
  (substution : ℕ) 
  (color : CheckerboardColor) 
  : ℝ := -- Total area of subquadrilaterals of the given color
  sorry

-- Theorem stating the required proof
theorem equal_total_areas_of_checkerboard_pattern
  (quad : ConvexQuadrilateral)
  (subdivision : ℕ)
  : total_area_of_color quad subdivision CheckerboardColor.Black = total_area_of_color quad subdivision CheckerboardColor.White :=
  sorry

end equal_total_areas_of_checkerboard_pattern_l226_226362


namespace largest_triangle_perimeter_l226_226347

theorem largest_triangle_perimeter (x : ℤ) (hx1 : 7 + 11 > x) (hx2 : 7 + x > 11) (hx3 : 11 + x > 7) (hx4 : 5 ≤ x) (hx5 : x < 18) : 
  7 + 11 + x = 35 :=
sorry

end largest_triangle_perimeter_l226_226347


namespace jerry_needs_shingles_l226_226414

theorem jerry_needs_shingles :
  let roofs := 3 in
  let length := 20 in
  let width := 40 in
  let sides := 2 in
  let shingles_per_sqft := 8 in
  let area_one_side := length * width in
  let area_one_roof := area_one_side * sides in
  let total_area := area_one_roof * roofs in
  let total_shingles := total_area * shingles_per_sqft in
  total_shingles = 38400 :=
by
  sorry

end jerry_needs_shingles_l226_226414


namespace mean_weight_of_soccer_team_l226_226458

-- Define the weights as per the conditions
def weights : List ℕ := [64, 68, 71, 73, 76, 76, 77, 78, 80, 82, 85, 87, 89, 89]

-- Define the total weight
def total_weight : ℕ := 64 + 68 + 71 + 73 + 76 + 76 + 77 + 78 + 80 + 82 + 85 + 87 + 89 + 89

-- Define the number of players
def number_of_players : ℕ := 14

-- Calculate the mean weight
noncomputable def mean_weight : ℚ := total_weight / number_of_players

-- The proof problem statement
theorem mean_weight_of_soccer_team : mean_weight = 75.357 := by
  -- This is where the proof would go.
  sorry

end mean_weight_of_soccer_team_l226_226458


namespace curve_C2_eqn_l226_226258

theorem curve_C2_eqn (p : ℝ) (x y : ℝ) :
  (∃ x y, (x^2 - y^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (2 * p = 3/4)) →
  (y^2 = (3/2) * x) :=
by
  sorry

end curve_C2_eqn_l226_226258


namespace number_of_dogs_l226_226696

variable (D C : ℕ)
variable (x : ℚ)

-- Conditions
def ratio_dogs_to_cats := D = (x * (C: ℚ) / 7)
def new_ratio_dogs_to_cats := D = (15 / 11) * (C + 8)

theorem number_of_dogs (h1 : ratio_dogs_to_cats D C x) (h2 : new_ratio_dogs_to_cats D C) : D = 77 := 
by sorry

end number_of_dogs_l226_226696


namespace number_of_friends_l226_226842

-- Define the conditions
def initial_apples := 55
def apples_given_to_father := 10
def apples_per_person := 9

-- Define the formula to calculate the number of friends
def friends (initial_apples apples_given_to_father apples_per_person : ℕ) : ℕ :=
  (initial_apples - apples_given_to_father - apples_per_person) / apples_per_person

-- State the Lean theorem
theorem number_of_friends :
  friends initial_apples apples_given_to_father apples_per_person = 4 :=
by
  sorry

end number_of_friends_l226_226842


namespace students_enrolled_in_all_three_l226_226045

variables {total_students at_least_one robotics_students dance_students music_students at_least_two_students all_three_students : ℕ}

-- Given conditions
axiom H1 : total_students = 25
axiom H2 : at_least_one = total_students
axiom H3 : robotics_students = 15
axiom H4 : dance_students = 12
axiom H5 : music_students = 10
axiom H6 : at_least_two_students = 11

-- We need to prove the number of students enrolled in all three workshops is 1
theorem students_enrolled_in_all_three : all_three_students = 1 :=
sorry

end students_enrolled_in_all_three_l226_226045


namespace sum_of_solutions_l226_226191

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l226_226191


namespace geometric_sequences_l226_226891

theorem geometric_sequences :
  ∃ (a q : ℝ) (a1 a2 a3 : ℕ → ℝ), 
    (∀ n, a1 n = a * (q - 2) ^ n) ∧ 
    (∀ n, a2 n = 2 * a * (q - 1) ^ n) ∧ 
    (∀ n, a3 n = 4 * a * q ^ n) ∧
    a = 1 ∧ q = 4 ∨ a = 192 / 31 ∧ q = 9 / 8 ∧
    (a + 2 * a + 4 * a = 84) ∧
    (a * (q - 2) + 2 * a * (q - 1) + 4 * a * q = 24) :=
sorry

end geometric_sequences_l226_226891


namespace impossible_coins_l226_226862

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l226_226862


namespace quadratic_has_equal_roots_l226_226406

-- Proposition: If the quadratic equation 3x^2 + 6x + m = 0 has two equal real roots, then m = 3.

theorem quadratic_has_equal_roots (m : ℝ) : 3 * 6 - 12 * m = 0 → m = 3 :=
by
  intro h
  sorry

end quadratic_has_equal_roots_l226_226406


namespace sharon_trip_distance_l226_226361

noncomputable section

variable (x : ℝ)

def sharon_original_speed (x : ℝ) := x / 200

def sharon_reduced_speed (x : ℝ) := (x / 200) - 1 / 2

def time_before_traffic (x : ℝ) := (x / 2) / (sharon_original_speed x)

def time_after_traffic (x : ℝ) := (x / 2) / (sharon_reduced_speed x)

theorem sharon_trip_distance : 
  (time_before_traffic x) + (time_after_traffic x) = 300 → x = 200 := 
by
  sorry

end sharon_trip_distance_l226_226361


namespace triangle_third_side_l226_226914

theorem triangle_third_side (a b : ℝ) (h₁ : a = 7) (h₂ : b = 11) : ∃ k : ℕ, 4 < k ∧ k < 18 ∧ k = 17 := 
by {
  let s := a + b,
  let d := b - a,
  have h₃ : s > 17 := by linarith,
  have h₄ : 4 < d := by linarith,
  have h₅ : d < 18 := by linarith,
  have h₆ : 17 < 18 := by linarith,
  use 17,
  linarith,
  sorry
}

end triangle_third_side_l226_226914


namespace rationalize_denominator_sum_equals_49_l226_226873

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end rationalize_denominator_sum_equals_49_l226_226873


namespace team_expected_score_l226_226640

universe u

noncomputable def team_score_expected_value : ℝ :=
  let p1 := 0.4
  let p2 := 0.4
  let p3 := 0.5
  let p_correct := 1 - ((1 - p1) * (1 - p2) * (1 - p3))
  let questions := 10
  let points_per_correct := 10
  questions * p_correct * points_per_correct

theorem team_expected_score : team_score_expected_value = 82 := by
  sorry

end team_expected_score_l226_226640


namespace chromium_percentage_new_alloy_l226_226831

variable (w1 w2 : ℝ) (cr1 cr2 : ℝ)

theorem chromium_percentage_new_alloy (h_w1 : w1 = 15) (h_w2 : w2 = 30) (h_cr1 : cr1 = 0.12) (h_cr2 : cr2 = 0.08) :
  (cr1 * w1 + cr2 * w2) / (w1 + w2) * 100 = 9.33 := by
  sorry

end chromium_percentage_new_alloy_l226_226831


namespace equation_not_expression_with_unknowns_l226_226075

def is_equation (expr : String) : Prop :=
  expr = "equation"

def contains_unknowns (expr : String) : Prop :=
  expr = "contains unknowns"

theorem equation_not_expression_with_unknowns (expr : String) (h1 : is_equation expr) (h2 : contains_unknowns expr) : 
  (is_equation expr) = False := 
sorry

end equation_not_expression_with_unknowns_l226_226075


namespace calculate_f_50_l226_226313

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_50 (f : ℝ → ℝ) (h_fun : ∀ x y : ℝ, f (x * y) = y * f x) (h_f2 : f 2 = 10) :
  f 50 = 250 :=
by
  sorry

end calculate_f_50_l226_226313


namespace danny_steve_ratio_l226_226088

theorem danny_steve_ratio :
  ∀ (D S : ℝ),
  D = 29 →
  2 * (S / 2 - D / 2) = 29 →
  D / S = 1 / 2 :=
by
  intros D S hD h_eq
  sorry

end danny_steve_ratio_l226_226088


namespace percentage_of_import_tax_l226_226935

noncomputable def total_value : ℝ := 2560
noncomputable def taxable_threshold : ℝ := 1000
noncomputable def import_tax : ℝ := 109.20

theorem percentage_of_import_tax :
  let excess_value := total_value - taxable_threshold
  let percentage_tax := (import_tax / excess_value) * 100
  percentage_tax = 7 := 
by
  sorry

end percentage_of_import_tax_l226_226935


namespace correct_transformation_l226_226920

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end correct_transformation_l226_226920


namespace frequencies_first_class_confidence_difference_quality_l226_226602

theorem frequencies_first_class (a b c d n : ℕ) (Ha : a = 150) (Hb : b = 50) 
                                (Hc : c = 120) (Hd : d = 80) (Hn : n = 400) 
                                (totalA : a + b = 200) 
                                (totalB : c + d = 200) :
  (a / (a + b) = 3 / 4) ∧ (c / (c + d) = 3 / 5) := by
sorry

theorem confidence_difference_quality (a b c d n : ℕ) (Ha : a = 150)
                                       (Hb : b = 50) (Hc : c = 120)
                                       (Hd : d = 80) (Hn : n = 400)
                                       (total : n = 400)
                                       (first_class_total : a + c = 270)
                                       (second_class_total : b + d = 130) :
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared > 6.635 := by
sorry

end frequencies_first_class_confidence_difference_quality_l226_226602


namespace odd_multiple_of_9_is_multiple_of_3_l226_226039

theorem odd_multiple_of_9_is_multiple_of_3 (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 9 = 0) : n % 3 = 0 := 
by sorry

end odd_multiple_of_9_is_multiple_of_3_l226_226039


namespace flour_amount_second_combination_l226_226733

-- Define given conditions as parameters
variables {sugar_cost flour_cost : ℝ} (sugar_per_pound flour_per_pound : ℝ)
variable (cost1 cost2 : ℝ)

axiom cost1_eq :
  40 * sugar_per_pound + 16 * flour_per_pound = cost1

axiom cost2_eq :
  30 * sugar_per_pound + flour_cost = cost2

axiom sugar_rate :
  sugar_per_pound = 0.45

axiom flour_rate :
  flour_per_pound = 0.45

-- Define the target theorem
theorem flour_amount_second_combination : ∃ flour_amount : ℝ, flour_amount = 28 := by
  sorry

end flour_amount_second_combination_l226_226733


namespace job_completion_days_l226_226932

theorem job_completion_days :
  let days_total := 150
  let workers_initial := 25
  let workers_less_efficient := 15
  let workers_more_efficient := 10
  let days_elapsed := 40
  let efficiency_less := 1
  let efficiency_more := 1.5
  let work_fraction_completed := 1/3
  let workers_fired_less := 4
  let workers_fired_more := 3
  let units_per_day_initial := (workers_less_efficient * efficiency_less) + (workers_more_efficient * efficiency_more)
  let work_completed := units_per_day_initial * days_elapsed
  let total_work := work_completed / work_fraction_completed
  let workers_remaining_less := workers_less_efficient - workers_fired_less
  let workers_remaining_more := workers_more_efficient - workers_fired_more
  let units_per_day_new := (workers_remaining_less * efficiency_less) + (workers_remaining_more * efficiency_more)
  let work_remaining := total_work * (2/3)
  let remaining_days := work_remaining / units_per_day_new
  remaining_days.ceil = 112 :=
by
  sorry

end job_completion_days_l226_226932


namespace gcd_of_17934_23526_51774_l226_226957

-- Define the three integers
def a : ℕ := 17934
def b : ℕ := 23526
def c : ℕ := 51774

-- State the theorem
theorem gcd_of_17934_23526_51774 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_of_17934_23526_51774_l226_226957


namespace scientific_notation_of_1206_million_l226_226562

theorem scientific_notation_of_1206_million :
  (1206 * 10^6 : ℝ) = 1.206 * 10^7 :=
by
  sorry

end scientific_notation_of_1206_million_l226_226562


namespace average_pastries_per_day_l226_226064

def monday_sales : ℕ := 2
def increment_weekday : ℕ := 2
def increment_weekend : ℕ := 3

def tuesday_sales : ℕ := monday_sales + increment_weekday
def wednesday_sales : ℕ := tuesday_sales + increment_weekday
def thursday_sales : ℕ := wednesday_sales + increment_weekday
def friday_sales : ℕ := thursday_sales + increment_weekday
def saturday_sales : ℕ := friday_sales + increment_weekend
def sunday_sales : ℕ := saturday_sales + increment_weekend

def total_sales_week : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
def average_sales_per_day : ℚ := total_sales_week / 7

theorem average_pastries_per_day : average_sales_per_day = 59 / 7 := by
  sorry

end average_pastries_per_day_l226_226064


namespace prime_digit_B_l226_226638

-- Mathematical description
def six_digit_form (B : Nat) : Nat := 3 * 10^5 + 0 * 10^4 + 3 * 10^3 + 7 * 10^2 + 0 * 10^1 + B

-- Prime condition
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

theorem prime_digit_B (B : Nat) : is_prime (six_digit_form B) ↔ B = 3 :=
sorry

end prime_digit_B_l226_226638


namespace pencils_distributed_per_container_l226_226095

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end pencils_distributed_per_container_l226_226095


namespace jessica_quarters_l226_226568

theorem jessica_quarters (quarters_initial quarters_given : Nat) (h_initial : quarters_initial = 8) (h_given : quarters_given = 3) :
  quarters_initial + quarters_given = 11 := by
  sorry

end jessica_quarters_l226_226568


namespace number_minus_29_l226_226907

theorem number_minus_29 (x : ℕ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end number_minus_29_l226_226907


namespace work_done_time_l226_226760

/-
  Question: How many days does it take for \(a\) to do the work alone?

  Conditions:
  - \(b\) can do the work in 20 days.
  - \(c\) can do the work in 55 days.
  - \(a\) is assisted by \(b\) and \(c\) on alternate days, and the work can be done in 8 days.
  
  Correct Answer:
  - \(x = 8.8\)
-/

theorem work_done_time (x : ℝ) (h : 8 * x⁻¹ + 1 /  5 + 4 / 55 = 1): x = 8.8 :=
by sorry

end work_done_time_l226_226760


namespace rolls_combinations_l226_226065

theorem rolls_combinations (x1 x2 x3 : ℕ) (h1 : x1 + x2 + x3 = 2) : 
  (Nat.choose (2 + 3 - 1) (3 - 1) = 6) :=
by
  sorry

end rolls_combinations_l226_226065


namespace find_prime_pairs_l226_226803

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end find_prime_pairs_l226_226803


namespace lcm_18_30_is_90_l226_226476

noncomputable def LCM_of_18_and_30 : ℕ := 90

theorem lcm_18_30_is_90 : nat.lcm 18 30 = LCM_of_18_and_30 := 
by 
  -- definition of lcm should be used here eventually
  sorry

end lcm_18_30_is_90_l226_226476


namespace Merry_sold_470_apples_l226_226438

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

end Merry_sold_470_apples_l226_226438


namespace arrangement_of_letters_l226_226003

-- Define the set of letters with subscripts
def letters : Finset String := {"B", "A₁", "B₁", "A₂", "B₂", "A₃"}

-- Define the number of ways to arrange 6 distinct letters
theorem arrangement_of_letters : letters.card.factorial = 720 := 
by {
  sorry
}

end arrangement_of_letters_l226_226003


namespace lcm_18_30_eq_90_l226_226468

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l226_226468


namespace find_a_l226_226391

noncomputable def f (a x : ℝ) := a * Real.exp x + 2 * x^2

noncomputable def f' (a x : ℝ) := a * Real.exp x + 4 * x

theorem find_a (a : ℝ) (h : f' a 0 = 2) : a = 2 :=
by
  unfold f' at h
  simp at h
  exact h

end find_a_l226_226391


namespace mrs_heine_dogs_l226_226160

theorem mrs_heine_dogs (total_biscuits biscuits_per_dog : ℕ) (h1 : total_biscuits = 6) (h2 : biscuits_per_dog = 3) :
  total_biscuits / biscuits_per_dog = 2 :=
by
  sorry

end mrs_heine_dogs_l226_226160


namespace find_x_l226_226493

theorem find_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  -- proof is not required, so we insert sorry
  sorry

end find_x_l226_226493


namespace maximize_box_volume_l226_226087

noncomputable def volume (x : ℝ) := (16 - 2 * x) * (10 - 2 * x) * x

theorem maximize_box_volume :
  (∃ x : ℝ, volume x = 144 ∧ ∀ y : ℝ, 0 < y ∧ y < 5 → volume y ≤ volume 2) := 
by
  sorry

end maximize_box_volume_l226_226087


namespace a_general_term_b_general_term_sum_Tn_condition_l226_226379

/-
Given:
1. A sequence {a_n} such that s_n = 2a_n - 2 where s_n is the sum of the first n terms of {a_n}
2. A sequence {b_n} such that b_1 = 1 and b_{n+1} = b_n + 2

To prove:
1. The general term of {a_n} is a_n = 2^n 
2. The general term of {b_n} is b_n = 2n - 1
3. The sum of the first n terms of c_n = a_n * b_n is T_n, and the largest integer n such that T_n < 167 is 4
-/

def seq_s (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i+1)

def a_n (n : ℕ) : ℕ := 2 ^ n

def b_n (n : ℕ) : ℕ := 2 * n - 1

def c_n (n : ℕ) : ℕ := a_n n * b_n n

def T_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, c_n (i+1)

theorem a_general_term (n : ℕ) : 
  seq_s n a_n = 2 * a_n n - 2 := sorry

theorem b_general_term (n : ℕ) :
  ∀ n > 0, b_n(n+1) = b_n(n) + 2 ∧ b_n(1) = 1 := sorry

theorem sum_Tn_condition (n : ℕ) : 
  T_n 4 < 167 ∧ ∀ m > 4, T_n m ≥ 167 := sorry

end a_general_term_b_general_term_sum_Tn_condition_l226_226379


namespace base3_20121_to_base10_l226_226084

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end base3_20121_to_base10_l226_226084


namespace impossibility_triplet_2002x2002_grid_l226_226707

theorem impossibility_triplet_2002x2002_grid: 
  ∀ (M : Matrix ℕ (Fin 2002) (Fin 2002)),
    (∀ i j : Fin 2002, ∃ (r1 r2 r3 : Fin 2002), 
      (M i r1 > 0 ∧ M i r2 > 0 ∧ M i r3 > 0) ∨ 
      (M r1 j > 0 ∧ M r2 j > 0 ∧ M r3 j > 0)) →
    ¬ (∀ i j : Fin 2002, ∃ (a b c : ℕ), 
      M i j = a ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
      (∃ (r1 r2 r3 : Fin 2002), 
        (M i r1 = a ∨ M i r1 = b ∨ M i r1 = c) ∧ 
        (M i r2 = a ∨ M i r2 = b ∨ M i r2 = c) ∧ 
        (M i r3 = a ∨ M i r3 = b ∨ M i r3 = c)) ∨
      (∃ (c1 c2 c3 : Fin 2002), 
        (M c1 j = a ∨ M c1 j = b ∨ M c1 j = c) ∧ 
        (M c2 j = a ∨ M c2 j = b ∨ M c2 j = c) ∧ 
        (M c3 j = a ∨ M c3 j = b ∨ M c3 j = c)))
:= sorry

end impossibility_triplet_2002x2002_grid_l226_226707


namespace cakes_served_yesterday_l226_226345

theorem cakes_served_yesterday:
  ∃ y : ℕ, (5 + 6 + y = 14) ∧ y = 3 := 
by
  sorry

end cakes_served_yesterday_l226_226345


namespace meaningful_iff_x_ne_2_l226_226127

theorem meaningful_iff_x_ne_2 (x : ℝ) : (x ≠ 2) ↔ (∃ y : ℝ, y = (x - 3) / (x - 2)) := 
by
  sorry

end meaningful_iff_x_ne_2_l226_226127


namespace garden_length_l226_226687

theorem garden_length (P : ℕ) (breadth : ℕ) (length : ℕ) 
  (h1 : P = 600) (h2 : breadth = 95) (h3 : P = 2 * (length + breadth)) : 
  length = 205 :=
by
  sorry

end garden_length_l226_226687


namespace determine_values_l226_226520

theorem determine_values (A B : ℚ) :
  (A + B = 4) ∧ (2 * A - 7 * B = 3) →
  A = 31 / 9 ∧ B = 5 / 9 :=
by
  sorry

end determine_values_l226_226520


namespace lcm_18_30_is_90_l226_226482

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l226_226482


namespace zoo_gorilla_percentage_l226_226903

theorem zoo_gorilla_percentage :
  ∀ (visitors_per_hour : ℕ) (open_hours : ℕ) (gorilla_visitors : ℕ) (total_visitors : ℕ)
    (percentage : ℕ),
  visitors_per_hour = 50 → open_hours = 8 → gorilla_visitors = 320 →
  total_visitors = visitors_per_hour * open_hours →
  percentage = (gorilla_visitors * 100) / total_visitors →
  percentage = 80 :=
by
  intros visitors_per_hour open_hours gorilla_visitors total_visitors percentage
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3, h4] at h5
  exact h5

end zoo_gorilla_percentage_l226_226903


namespace two_digit_numbers_count_l226_226246

theorem two_digit_numbers_count : 
  ∃ (count : ℕ), (
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ b = 2 * a → 
      (10 * b + a = 7 / 4 * (10 * a + b))) 
      ∧ count = 4
  ) :=
sorry

end two_digit_numbers_count_l226_226246


namespace cyclist_speed_l226_226910

/-- 
  Two cyclists A and B start at the same time from Newton to Kingston, a distance of 50 miles. 
  Cyclist A travels 5 mph slower than cyclist B. After reaching Kingston, B immediately turns 
  back and meets A 10 miles from Kingston. --/
theorem cyclist_speed (a b : ℕ) (h1 : b = a + 5) (h2 : 40 / a = 60 / b) : a = 10 :=
by
  sorry

end cyclist_speed_l226_226910


namespace total_pizza_slices_correct_l226_226580

-- Define the conditions
def num_pizzas : Nat := 3
def slices_per_first_two_pizzas : Nat := 8
def num_first_two_pizzas : Nat := 2
def slices_third_pizza : Nat := 12

-- Define the total slices based on conditions
def total_slices : Nat := slices_per_first_two_pizzas * num_first_two_pizzas + slices_third_pizza

-- The theorem to be proven
theorem total_pizza_slices_correct : total_slices = 28 := by
  sorry

end total_pizza_slices_correct_l226_226580


namespace problem_l226_226736

noncomputable def p (k : ℝ) (x : ℝ) := k * (x - 5) * (x - 2)
noncomputable def q (x : ℝ) := (x - 5) * (x + 3)

theorem problem {p q : ℝ → ℝ} (k : ℝ) :
  (∀ x, q x = (x - 5) * (x + 3)) →
  (∀ x, p x = k * (x - 5) * (x - 2)) →
  (∀ x ≠ 5, (p x) / (q x) = (3 * (x - 2)) / (x + 3)) →
  p 3 / q 3 = 1 / 2 :=
by
  sorry

end problem_l226_226736


namespace sum_of_solutions_eq_9_l226_226193

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l226_226193


namespace probability_digits_different_l226_226788

theorem probability_digits_different : 
  let total_numbers := 490
  let same_digits_numbers := 13
  let different_digits_numbers := total_numbers - same_digits_numbers 
  let probability := different_digits_numbers / total_numbers 
  probability = 477 / 490 :=
by
  sorry

end probability_digits_different_l226_226788


namespace lion_cub_birth_rate_l226_226746

theorem lion_cub_birth_rate :
  ∀ (x : ℕ), 100 + 12 * (x - 1) = 148 → x = 5 :=
by
  intros x h
  sorry

end lion_cub_birth_rate_l226_226746


namespace rationalize_denominator_sum_equals_49_l226_226872

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end rationalize_denominator_sum_equals_49_l226_226872


namespace episode_length_l226_226855

/-- Subject to the conditions provided, we prove the length of each episode watched by Maddie. -/
theorem episode_length
  (total_episodes : ℕ)
  (monday_minutes : ℕ)
  (thursday_minutes : ℕ)
  (weekend_minutes : ℕ)
  (episodes_length : ℕ)
  (monday_watch : monday_minutes = 138)
  (thursday_watch : thursday_minutes = 21)
  (weekend_watch : weekend_minutes = 105)
  (total_episodes_watch : total_episodes = 8)
  (total_minutes : monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episodes_length) :
  episodes_length = 33 := 
by 
  sorry

end episode_length_l226_226855


namespace pow_mod_equality_l226_226755

theorem pow_mod_equality (h : 2^3 ≡ 1 [MOD 7]) : 2^30 ≡ 1 [MOD 7] :=
sorry

end pow_mod_equality_l226_226755


namespace mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l226_226579

noncomputable def Mork_base_income (M : ℝ) : ℝ := M
noncomputable def Mindy_base_income (M : ℝ) : ℝ := 4 * M
noncomputable def Mork_total_income (M : ℝ) : ℝ := 1.5 * M
noncomputable def Mindy_total_income (M : ℝ) : ℝ := 6 * M

noncomputable def Mork_total_tax (M : ℝ) : ℝ :=
  0.4 * M + 0.5 * 0.5 * M
noncomputable def Mindy_total_tax (M : ℝ) : ℝ :=
  0.3 * 4 * M + 0.35 * 2 * M

noncomputable def Mork_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M) / (Mork_total_income M)

noncomputable def Mindy_effective_tax_rate (M : ℝ) : ℝ :=
  (Mindy_total_tax M) / (Mindy_total_income M)

noncomputable def combined_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M + Mindy_total_tax M) / (Mork_total_income M + Mindy_total_income M)

theorem mork_effective_tax_rate_theorem (M : ℝ) : Mork_effective_tax_rate M = 43.33 / 100 := sorry
theorem mindy_effective_tax_rate_theorem (M : ℝ) : Mindy_effective_tax_rate M = 31.67 / 100 := sorry
theorem combined_effective_tax_rate_theorem (M : ℝ) : combined_effective_tax_rate M = 34 / 100 := sorry

end mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l226_226579


namespace move_3m_left_is_neg_3m_l226_226506

-- Define the notation for movements
def move_right (distance : Int) : Int := distance
def move_left (distance : Int) : Int := -distance

-- Define the specific condition
def move_1m_right : Int := move_right 1

-- Define the assertion for moving 3m to the left
def move_3m_left : Int := move_left 3

-- State the proof problem
theorem move_3m_left_is_neg_3m : move_3m_left = -3 := by
  unfold move_3m_left
  unfold move_left
  rfl

end move_3m_left_is_neg_3m_l226_226506


namespace range_of_quadratic_function_l226_226902

theorem range_of_quadratic_function : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 0 ≤ x^2 - 4 * x + 3 ∧ x^2 - 4 * x + 3 ≤ 8 :=
by
  intro x hx
  sorry

end range_of_quadratic_function_l226_226902


namespace find_p_minus_q_l226_226814

theorem find_p_minus_q (p q : ℝ) (h : ∀ x, x^2 - 6 * x + q = 0 ↔ (x - p)^2 = 7) : p - q = 1 :=
sorry

end find_p_minus_q_l226_226814


namespace prob_even_product_first_second_spinner_l226_226461

def first_spinner_values := [2, 3, 5, 7, 11]
def second_spinner_values := [4, 6, 9, 10, 13, 15]

def probability_even_product (spinner1 spinner2 : List ℕ) : ℚ :=
  let total_outcomes := spinner1.length * spinner2.length
  let is_even (n : ℕ) : Bool := (n % 2 = 0)
  let is_odd (n : ℕ) : Bool := ¬ is_even n
  let odd_values1 := spinner1.filter is_odd
  let odd_values2 := spinner2.filter is_odd
  let odd_outcomes := odd_values1.length * odd_values2.length
  1 - (odd_outcomes : ℚ) / (total_outcomes : ℚ)

theorem prob_even_product_first_second_spinner :
  probability_even_product first_spinner_values second_spinner_values = 7 / 10 :=
by
  sorry

end prob_even_product_first_second_spinner_l226_226461


namespace rides_ratio_l226_226683

theorem rides_ratio (total_money rides_spent dessert_spent money_left : ℕ) 
  (h1 : total_money = 30) 
  (h2 : dessert_spent = 5) 
  (h3 : money_left = 10) 
  (h4 : total_money - money_left = rides_spent + dessert_spent) : 
  (rides_spent : ℚ) / total_money = 1 / 2 := 
sorry

end rides_ratio_l226_226683


namespace lcm_of_18_and_30_l226_226471

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l226_226471


namespace find_g_of_3_l226_226452

theorem find_g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 5 * g (1 / x) = 2 * x) : g 3 = -32 / 63 :=
by sorry

end find_g_of_3_l226_226452


namespace find_C1_C2_value_l226_226990

-- Definitions and conditions
def C1_param_eqns (a b φ : ℝ) : ℝ × ℝ := (a * cos φ, b * sin φ)
def C2_eqn (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

theorem find_C1_C2_value :
  ∃ (a b : ℝ), (a > b ∧ b > 0) ∧ C1_param_eqns a b (π / 3) = (1, sqrt 3 / 2) ∧
  (∀ (x y : ℝ), ∃ (φ : ℝ), C1_param_eqns a b φ = (x, y) → x^2 / 4 + y^2 = 1) ∧
  ∀ (θ : ℝ), C2_eqn 1 (π / 3) ∧ (θ = π / 3 → ∀ (ρ1 ρ2 : ℝ), 
    (C1_param_eqns ρ1 θ = (ρ1 * cos θ, ρ1 * sin θ)) ∧ 
    (C1_param_eqns ρ2 (θ + π / 2) = (ρ2 * cos (θ + π / 2), ρ2 * sin (θ + π / 2))) → 
    (1 / ρ1^2 + 1 / ρ2^2 = 5 / 4)) :=
by
  sorry

end find_C1_C2_value_l226_226990


namespace angle_between_vectors_with_offset_l226_226245

noncomputable def vector_angle_with_offset : ℝ :=
  let v1 := (4, -1)
  let v2 := (6, 8)
  let dot_product := 4 * 6 + (-1) * 8
  let magnitude_v1 := Real.sqrt (4 ^ 2 + (-1) ^ 2)
  let magnitude_v2 := Real.sqrt (6 ^ 2 + 8 ^ 2)
  let cos_theta := dot_product / (magnitude_v1 * magnitude_v2)
  Real.arccos cos_theta + 30

theorem angle_between_vectors_with_offset :
  vector_angle_with_offset = Real.arccos (8 / (5 * Real.sqrt 17)) + 30 := 
sorry

end angle_between_vectors_with_offset_l226_226245


namespace third_shiny_penny_prob_l226_226494

open Nat

def num_shiny : Nat := 4
def num_dull : Nat := 5
def total_pennies : Nat := num_shiny + num_dull

theorem third_shiny_penny_prob :
  let a := 5
  let b := 9
  a + b = 14 := 
by
  sorry

end third_shiny_penny_prob_l226_226494


namespace skirt_price_l226_226709

theorem skirt_price (S : ℝ) 
  (h1 : 2 * 5 = 10) 
  (h2 : 1 * 4 = 4) 
  (h3 : 6 * (5 / 2) = 15) 
  (h4 : 10 + 4 + 15 + 4 * S = 53) 
  : S = 6 :=
sorry

end skirt_price_l226_226709


namespace option_A_is_correct_l226_226919

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end option_A_is_correct_l226_226919


namespace combination_15_choose_3_l226_226548

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l226_226548


namespace scientific_notation_of_number_l226_226644

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l226_226644


namespace smallest_positive_period_intervals_of_monotonicity_max_min_values_l226_226260

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

end smallest_positive_period_intervals_of_monotonicity_max_min_values_l226_226260


namespace mean_of_set_eq_10point6_l226_226793

open Real -- For real number operations

theorem mean_of_set_eq_10point6 (n : ℝ)
  (h : n + 7 = 11) :
  (4 + 7 + 11 + 13 + 18) / 5 = 10.6 :=
by
  have h1 : n = 4 := by linarith
  sorry -- skip the proof part

end mean_of_set_eq_10point6_l226_226793


namespace find_total_amount_l226_226073

variables (A B C : ℕ) (total_amount : ℕ) 

-- Conditions
def condition1 : Prop := B = 36
def condition2 : Prop := 100 * B / 45 = A
def condition3 : Prop := 100 * C / 30 = A

-- Proof statement
theorem find_total_amount (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 A C) :
  total_amount = 300 :=
sorry

end find_total_amount_l226_226073


namespace B_subset_A_iff_l226_226395

namespace MathProofs

def A (x : ℝ) : Prop := -2 < x ∧ x < 5

def B (x : ℝ) (m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem B_subset_A_iff (m : ℝ) :
  (∀ x : ℝ, B x m → A x) ↔ m < 3 :=
by
  sorry

end MathProofs

end B_subset_A_iff_l226_226395


namespace conditional_probability_correct_l226_226144

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define the events A and B
variable (A B : Event Ω)

-- Probability of A and AB
variable (P_A : ℝ) (P_AB : ℝ)

-- Given conditions: P(A) = 0.5 and P(AB) = 0.4
axiom P_A_def : P_A = 0.5
axiom P_AB_def : P_AB = 0.4

-- Define the conditional probability P(B|A)
noncomputable def P_B_given_A : ℝ := P_AB / P_A

-- Expected result: P(B|A) = 0.8
theorem conditional_probability_correct : P_B_given_A A B P_AB P_A = 0.8 :=
by
  -- Use the given conditions
  rw [P_A_def, P_AB_def]
  -- Simplify the expression
  sorry

end conditional_probability_correct_l226_226144


namespace parallel_transitive_l226_226787

-- Definition of parallel lines
def are_parallel (l1 l2 : Line) : Prop :=
  ∃ (P : Line), l1 = P ∧ l2 = P

-- Theorem stating that if two lines are parallel to the same line, then they are parallel to each other
theorem parallel_transitive (l1 l2 l3 : Line) (h1 : are_parallel l1 l3) (h2 : are_parallel l2 l3) :
  are_parallel l1 l2 :=
by
  sorry

end parallel_transitive_l226_226787


namespace sum_of_roots_l226_226188

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l226_226188


namespace average_temperature_correct_l226_226172

theorem average_temperature_correct (W T : ℝ) :
  (38 + W + T) / 3 = 32 →
  44 = 44 →
  38 = 38 →
  (W + T + 44) / 3 = 34 :=
by
  intros h1 h2 h3
  sorry

end average_temperature_correct_l226_226172


namespace y_z_add_x_eq_160_l226_226135

theorem y_z_add_x_eq_160 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 132) (h5 : z * (x + y) = 180) (h6 : x * y * z = 160) :
  y * (z + x) = 160 := 
by 
  sorry

end y_z_add_x_eq_160_l226_226135


namespace find_intersection_points_l226_226369

def intersection_points (t α : ℝ) : Prop :=
∃ t α : ℝ,
  (2 + t, -1 - t) = (3 * Real.cos α, 3 * Real.sin α) ∧
  ((2 + t = (1 + Real.sqrt 17) / 2 ∧ -1 - t = (1 - Real.sqrt 17) / 2) ∨
   (2 + t = (1 - Real.sqrt 17) / 2 ∧ -1 - t = (1 + Real.sqrt 17) / 2))

theorem find_intersection_points : intersection_points t α :=
sorry

end find_intersection_points_l226_226369


namespace ann_total_fare_for_100_miles_l226_226941

-- Conditions
def base_fare : ℕ := 20
def fare_per_distance (distance : ℕ) : ℕ := 180 * distance / 80

-- Question: How much would Ann be charged if she traveled 100 miles?
def total_fare (distance : ℕ) : ℕ := (fare_per_distance distance) + base_fare

-- Prove that the total fare for 100 miles is 245 dollars
theorem ann_total_fare_for_100_miles : total_fare 100 = 245 :=
by
  -- Adding your proof here
  sorry

end ann_total_fare_for_100_miles_l226_226941


namespace sum_series_equals_l226_226357

theorem sum_series_equals :
  (∑' n : ℕ, if n ≥ 2 then 1 / (n * (n + 3)) else 0) = 13 / 36 :=
by
  sorry

end sum_series_equals_l226_226357


namespace smallest_b_l226_226712

open Real

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 2 ∧ B = a ∧ C = b ∨ A = 2 ∧ B = b ∧ C = a ∨ A = a ∧ B = b ∧ C = 2) ∧ A + B > C ∧ A + C > B ∧ B + C > A)
  (h4 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 1 / b ∧ B = 1 / a ∧ C = 2 ∨ A = 1 / a ∧ B = 1 / b ∧ C = 2 ∨ A = 1 / b ∧ B = 2 ∧ C = 1 / a ∨ A = 1 / a ∧ B = 2 ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / a ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / b ∧ C = 1 / a) ∧ A + B > C ∧ A + C > B ∧ B + C > A) :
  b = 2 := 
sorry

end smallest_b_l226_226712


namespace weight_of_new_girl_l226_226925

theorem weight_of_new_girl (W N : ℝ) (h_weight_replacement: (20 * W / 20 + 40 - 40 + 40) / 20 = W / 20 + 2) :
  N = 80 :=
by
  sorry

end weight_of_new_girl_l226_226925


namespace count_powers_of_2_not_4_under_2000000_l226_226269

theorem count_powers_of_2_not_4_under_2000000 :
  ∃ n, ∀ x, x < 2000000 → (∃ k, x = 2 ^ k ∧ (∀ m, x ≠ 4 ^ m)) ↔ x > 0 ∧ x < 2 ^ (n + 1) := by
  sorry

end count_powers_of_2_not_4_under_2000000_l226_226269


namespace ratio_sum_ineq_l226_226257

theorem ratio_sum_ineq 
  (a b α β : ℝ) 
  (hαβ : 0 < α ∧ 0 < β) 
  (h_range : α ≤ a ∧ a ≤ β ∧ α ≤ b ∧ b ≤ β) : 
  (b / a + a / b ≤ β / α + α / β) ∧ 
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β ∨ a = β ∧ b = α)) :=
by
  sorry

end ratio_sum_ineq_l226_226257


namespace factorization_pq_difference_l226_226031

theorem factorization_pq_difference :
  ∃ (p q : ℤ), 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q) ∧ p - q = 36 := by
-- Given the conditions in the problem,
-- We assume ∃ integers p and q such that (5x + p)(5x + q) = 25x² - 135x - 150 and derive the difference p - q = 36.
  sorry

end factorization_pq_difference_l226_226031


namespace find_slant_height_l226_226106

-- Definitions of the given conditions
variable (r1 r2 L A1 A2 : ℝ)
variable (π : ℝ := Real.pi)

-- The conditions as given in the problem
def conditions : Prop := 
  r1 = 3 ∧ r2 = 4 ∧ 
  (π * L * (r1 + r2) = A1 + A2) ∧ 
  (A1 = π * r1^2) ∧ 
  (A2 = π * r2^2)

-- The theorem stating the question and the correct answer
theorem find_slant_height (h : conditions r1 r2 L A1 A2) : 
  L = 5 := 
sorry

end find_slant_height_l226_226106


namespace expression_equivalence_l226_226797

theorem expression_equivalence : (2 / 20) + (3 / 30) + (4 / 40) + (5 / 50) = 0.4 := by
  sorry

end expression_equivalence_l226_226797


namespace champion_is_C_l226_226694

-- Definitions of statements made by Zhang, Wang, and Li
def zhang_statement (winner : String) : Bool := winner = "A" ∨ winner = "B"
def wang_statement (winner : String) : Bool := winner ≠ "C"
def li_statement (winner : String) : Bool := winner ≠ "A" ∧ winner ≠ "B"

-- Predicate that indicates exactly one of the statements is correct
def exactly_one_correct (winner : String) : Prop :=
  (zhang_statement winner ∧ ¬wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ ¬wang_statement winner ∧ li_statement winner)

-- The theorem stating the correct answer to the problem
theorem champion_is_C : (exactly_one_correct "C") :=
  by
    sorry  -- Proof goes here

-- Note: The import statement and sorry definition are included to ensure the code builds.

end champion_is_C_l226_226694


namespace simplify_fraction_l226_226308

theorem simplify_fraction (a b : ℕ) (h₁ : a = 84) (h₂ : b = 144) :
  a / gcd a b = 7 ∧ b / gcd a b = 12 := 
by
  sorry

end simplify_fraction_l226_226308


namespace total_charge_correct_l226_226023

def boxwoodTrimCost (numBoxwoods : Nat) (trimCost : Nat) : Nat :=
  numBoxwoods * trimCost

def boxwoodShapeCost (numBoxwoods : Nat) (shapeCost : Nat) : Nat :=
  numBoxwoods * shapeCost

theorem total_charge_correct :
  let numBoxwoodsTrimmed := 30
  let trimCost := 5
  let numBoxwoodsShaped := 4
  let shapeCost := 15
  let totalTrimCost := boxwoodTrimCost numBoxwoodsTrimmed trimCost
  let totalShapeCost := boxwoodShapeCost numBoxwoodsShaped shapeCost
  let totalCharge := totalTrimCost + totalShapeCost
  totalCharge = 210 :=
by sorry

end total_charge_correct_l226_226023


namespace scientific_notation_of_number_l226_226646

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l226_226646


namespace find_other_two_sides_of_isosceles_right_triangle_l226_226408

noncomputable def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  ((AB.1 ^ 2 + AB.2 ^ 2 = AC.1 ^ 2 + AC.2 ^ 2 ∧ BC.1 ^ 2 + BC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AB.1 ^ 2 + AB.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AC.1 ^ 2 + AC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AC.1 ^ 2 + AC.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AB.1 ^ 2 + AB.2 ^ 2 = 2 * (AC.1 ^ 2 + AC.2 ^ 2)))

theorem find_other_two_sides_of_isosceles_right_triangle (A B C : ℝ × ℝ)
  (h : is_isosceles_right_triangle A B C)
  (line_AB : 2 * A.1 - A.2 = 0)
  (midpoint_hypotenuse : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 2) :
  (A.1 + 2 * A.2 = 2 ∨ A.1 + 2 * A.2 = 14) ∧ 
  ((A.2 = 2 * A.1) ∨ (A.1 = 4)) :=
sorry

end find_other_two_sides_of_isosceles_right_triangle_l226_226408


namespace dividend_calculation_l226_226053

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 18) (h2 : quotient = 9) (h3 : remainder = 5) : 
  (divisor * quotient + remainder = 167) :=
by
  sorry

end dividend_calculation_l226_226053


namespace shaded_region_area_l226_226455

-- Define the problem conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def area_of_shaded_region : ℝ := 50

-- State the theorem to prove the area of the shaded region
theorem shaded_region_area (n : ℕ) (d : ℝ) (area : ℝ) (h1 : n = num_squares) (h2 : d = diagonal_length) : 
  area = area_of_shaded_region :=
sorry

end shaded_region_area_l226_226455


namespace common_ratio_of_geometric_sequence_l226_226255

variable (a : ℕ → ℝ) (d : ℝ)
variable (a1 : ℝ) (h_d : d ≠ 0)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem common_ratio_of_geometric_sequence :
  (a 0 = a1) →
  (a 4 = a1 + 4 * d) →
  (a 16 = a1 + 16 * d) →
  (a1 + 4 * d) / a1 = (a1 + 16 * d) / (a1 + 4 * d) →
  (a1 + 16 * d) / (a1 + 4 * d) = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l226_226255


namespace find_sum_of_a_b_c_l226_226592

def a := 8
def b := 2
def c := 2

theorem find_sum_of_a_b_c : a + b + c = 12 :=
by
  have ha : a = 8 := rfl
  have hb : b = 2 := rfl
  have hc : c = 2 := rfl
  sorry

end find_sum_of_a_b_c_l226_226592


namespace scientific_notation_256000_l226_226730

theorem scientific_notation_256000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 256000 = a * 10^n ∧ a = 2.56 ∧ n = 5 :=
by
  sorry

end scientific_notation_256000_l226_226730


namespace alcohol_water_ratio_l226_226401

theorem alcohol_water_ratio (a b : ℚ) (h₁ : a = 3/5) (h₂ : b = 2/5) : a / b = 3 / 2 :=
by
  sorry

end alcohol_water_ratio_l226_226401


namespace number_of_groups_of_three_books_l226_226553

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l226_226553


namespace difference_in_sums_l226_226722

def sum_of_digits (n : ℕ) : ℕ := (toString n).foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

def Petrov_numbers := List.range' 1 2014 |>.filter (λ n => n % 2 = 1)
def Vasechkin_numbers := List.range' 2 2012 |>.filter (λ n => n % 2 = 0)

def sum_of_digits_Petrov := (Petrov_numbers.map sum_of_digits).sum
def sum_of_digits_Vasechkin := (Vasechkin_numbers.map sum_of_digits).sum

theorem difference_in_sums : sum_of_digits_Petrov - sum_of_digits_Vasechkin = 1007 := by
  sorry

end difference_in_sums_l226_226722


namespace sum_of_faces_of_rectangular_prism_l226_226309

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end sum_of_faces_of_rectangular_prism_l226_226309


namespace negation_if_positive_then_square_positive_l226_226896

theorem negation_if_positive_then_square_positive :
  (¬ (∀ x : ℝ, x > 0 → x^2 > 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end negation_if_positive_then_square_positive_l226_226896


namespace rate_second_year_l226_226244

/-- Define the principal amount at the start. -/
def P : ℝ := 4000

/-- Define the rate of interest for the first year. -/
def rate_first_year : ℝ := 0.04

/-- Define the final amount after 2 years. -/
def A : ℝ := 4368

/-- Define the amount after the first year. -/
def P1 : ℝ := P + P * rate_first_year

/-- Define the interest for the second year. -/
def Interest2 : ℝ := A - P1

/-- Define the principal amount for the second year, which is the amount after the first year. -/
def P2 : ℝ := P1

/-- Prove that the rate of interest for the second year is 5%. -/
theorem rate_second_year : (Interest2 / P2) * 100 = 5 :=
by
  sorry

end rate_second_year_l226_226244


namespace remainder_div_29_l226_226488

theorem remainder_div_29 (k : ℤ) (N : ℤ) (h : N = 899 * k + 63) : N % 29 = 10 :=
  sorry

end remainder_div_29_l226_226488


namespace jerome_contact_list_count_l226_226846

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end jerome_contact_list_count_l226_226846


namespace number_leaves_remainder_3_l226_226905

theorem number_leaves_remainder_3 (n : ℕ) (h1 : 1680 % 9 = 0) (h2 : 1680 = n * 9) : 1680 % 1677 = 3 := by
  sorry

end number_leaves_remainder_3_l226_226905


namespace isosceles_triangle_base_length_l226_226538

theorem isosceles_triangle_base_length
  (a b c : ℕ)
  (h_iso : a = b)
  (h_perimeter : a + b + c = 62)
  (h_leg_length : a = 25) :
  c = 12 :=
by
  sorry

end isosceles_triangle_base_length_l226_226538


namespace geometric_progression_identity_l226_226443

theorem geometric_progression_identity 
  (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 :=
by 
  sorry

end geometric_progression_identity_l226_226443


namespace hyperbola_equilateral_triangle_area_l226_226677

open Real

def point := (ℝ × ℝ)

noncomputable def hyperbola := {p : point | p.1^2 - p.2^2 = 1}

noncomputable def is_equilateral_triangle (A B C : point) :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem hyperbola_equilateral_triangle_area :
  ∀ (A B C : point),
    A = (-1, 0) →
    A ∈ hyperbola →
    B ∈ hyperbola →
    C ∈ hyperbola →
    is_equilateral_triangle A B C →
    area_of_triangle A B C = 3 * sqrt 3 :=
by
  sorry

end hyperbola_equilateral_triangle_area_l226_226677


namespace find_n_l226_226951

theorem find_n (n : ℕ) : (256 : ℝ)^(1/4) = (4 : ℝ)^n → n = 1 := 
by
  sorry

end find_n_l226_226951


namespace time_2517_hours_from_now_l226_226734

-- Define the initial time and the function to calculate time after certain hours on a 12-hour clock
def current_time := 3
def hours := 2517

noncomputable def final_time_mod_12 (current_time : ℕ) (hours : ℕ) : ℕ :=
  (current_time + (hours % 12)) % 12

theorem time_2517_hours_from_now :
  final_time_mod_12 current_time hours = 12 :=
by
  sorry

end time_2517_hours_from_now_l226_226734


namespace no_integer_b_two_distinct_roots_l226_226799

theorem no_integer_b_two_distinct_roots :
  ∀ b : ℤ, ¬ ∃ x y : ℤ, x ≠ y ∧ (x^4 + 4 * x^3 + b * x^2 + 16 * x + 8 = 0) ∧ (y^4 + 4 * y^3 + b * y^2 + 16 * y + 8 = 0) :=
by
  sorry

end no_integer_b_two_distinct_roots_l226_226799


namespace find_b_l226_226981

variable {a b d m : ℝ}

theorem find_b (h : m = d * a * b / (a + b)) : b = m * a / (d * a - m) :=
sorry

end find_b_l226_226981


namespace spending_spring_months_l226_226894

variable (s_feb s_may : ℝ)

theorem spending_spring_months (h1 : s_feb = 2.8) (h2 : s_may = 5.6) : s_may - s_feb = 2.8 := 
by
  sorry

end spending_spring_months_l226_226894


namespace total_customers_served_l226_226944

-- Definitions for the hours worked by Ann, Becky, and Julia
def hours_ann : ℕ := 8
def hours_becky : ℕ := 8
def hours_julia : ℕ := 6

-- Definition for the number of customers served per hour
def customers_per_hour : ℕ := 7

-- Total number of customers served by Ann, Becky, and Julia
def total_customers : ℕ :=
  (hours_ann * customers_per_hour) + 
  (hours_becky * customers_per_hour) + 
  (hours_julia * customers_per_hour)

theorem total_customers_served : total_customers = 154 :=
  by 
    -- This is where the proof would go, but we'll use sorry to indicate it's incomplete
    sorry

end total_customers_served_l226_226944


namespace max_strips_cut_l226_226965

-- Definitions: dimensions of the paper and the strips
def length_paper : ℕ := 14
def width_paper : ℕ := 11
def length_strip : ℕ := 4
def width_strip : ℕ := 1

-- States the main theorem: Maximum number of strips that can be cut from the rectangular piece of paper
theorem max_strips_cut (L W l w : ℕ) (H1 : L = 14) (H2 : W = 11) (H3 : l = 4) (H4 : w = 1) :
  ∃ n : ℕ, n = 33 :=
by
  sorry

end max_strips_cut_l226_226965


namespace N_is_necessary_but_not_sufficient_l226_226266

-- Define sets M and N
def M := { x : ℝ | 0 < x ∧ x < 1 }
def N := { x : ℝ | -2 < x ∧ x < 1 }

-- State the theorem to prove that "a belongs to N" is necessary but not sufficient for "a belongs to M"
theorem N_is_necessary_but_not_sufficient (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → a ∈ M → False) :=
by sorry

end N_is_necessary_but_not_sufficient_l226_226266


namespace smallest_value_for_x_9_l226_226665

theorem smallest_value_for_x_9 :
  let x := 9
  ∃ i, i = (8 / (x + 2)) ∧ 
  (i < (8 / x) ∧ 
   i < (8 / (x - 2)) ∧ 
   i < (x / 8) ∧ 
   i < ((x + 2) / 8)) :=
by
  let x := 9
  use (8 / (x + 2))
  sorry

end smallest_value_for_x_9_l226_226665


namespace range_of_a_l226_226817

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by 
sorry

end range_of_a_l226_226817


namespace simson_line_properties_l226_226027

-- Given a triangle ABC
variables {A B C M P Q R H : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] 
variables [Inhabited M] [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited H]

-- Conditions
def is_point_on_circumcircle (A B C : Type) (M : Type) : Prop :=
sorry  -- formal definition that M is on the circumcircle of triangle ABC

def perpendicular_dropped_to_side (M : Type) (side : Type) (foot : Type) : Prop :=
sorry  -- formal definition of a perpendicular dropping from M to a side

def is_orthocenter (A B C H : Type) : Prop := 
sorry  -- formal definition that H is the orthocenter of triangle ABC

-- Proof Goal 1: The points P, Q, R are collinear (Simson line)
def simson_line (A B C M P Q R : Type) : Prop :=
sorry  -- formal definition and proof that P, Q, R are collinear

-- Proof Goal 2: The Simson line is equidistant from point M and the orthocenter H
def simson_line_equidistant (M H P Q R : Type) : Prop :=
sorry  -- formal definition and proof that Simson line is equidistant from M and H

-- Main theorem combining both proof goals
theorem simson_line_properties 
  (A B C M P Q R H : Type)
  (M_on_circumcircle : is_point_on_circumcircle A B C M)
  (perp_to_BC : perpendicular_dropped_to_side M (B × C) P)
  (perp_to_CA : perpendicular_dropped_to_side M (C × A) Q)
  (perp_to_AB : perpendicular_dropped_to_side M (A × B) R)
  (H_is_orthocenter : is_orthocenter A B C H) :
  simson_line A B C M P Q R ∧ simson_line_equidistant M H P Q R := 
by sorry

end simson_line_properties_l226_226027


namespace high_card_point_value_l226_226008

theorem high_card_point_value :
  ∀ (H L : ℕ), 
  (L = 1) →
  ∀ (high low total_points : ℕ), 
  (total_points = 5) →
  (high + (L + L + L) = total_points) →
  high = 2 :=
by
  intros
  sorry

end high_card_point_value_l226_226008


namespace area_of_triangles_l226_226014

theorem area_of_triangles
  (ABC_area : ℝ)
  (AD : ℝ)
  (DB : ℝ)
  (h_AD_DB : AD + DB = 7)
  (h_equal_areas : ABC_area = 12) :
  (∃ ABE_area : ℝ, ABE_area = 36 / 7) ∧ (∃ DBF_area : ℝ, DBF_area = 36 / 7) :=
by
  sorry

end area_of_triangles_l226_226014


namespace constant_term_expansion_l226_226173

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : k ≤ n then Nat.choose n k else 0

theorem constant_term_expansion :
    ∀ x: ℂ, (x ≠ 0) → ∃ term: ℂ, 
    term = (-1 : ℂ) * binom 6 4 ∧ term = -15 := 
by
  intros x hx
  use (-1 : ℂ) * binom 6 4
  constructor
  · rfl
  · sorry

end constant_term_expansion_l226_226173


namespace preparation_start_month_l226_226166

variable (ExamMonth : ℕ)
def start_month (ExamMonth : ℕ) : ℕ :=
  (ExamMonth - 5) % 12

theorem preparation_start_month :
  ∀ (ExamMonth : ℕ), start_month ExamMonth = (ExamMonth - 5) % 12 :=
by
  sorry

end preparation_start_month_l226_226166


namespace quadratic_roots_min_value_l226_226999

theorem quadratic_roots_min_value (m α β : ℝ) (h_eq : 4 * α^2 - 4 * m * α + m + 2 = 0) (h_eq2 : 4 * β^2 - 4 * m * β + m + 2 = 0) :
  (∃ m_val : ℝ, m_val = -1 ∧ α^2 + β^2 = 1 / 2) :=
by
  sorry

end quadratic_roots_min_value_l226_226999


namespace equal_share_is_168_l226_226573

namespace StrawberryProblem

def brother_baskets : ℕ := 3
def strawberries_per_basket : ℕ := 15
def brother_strawberries : ℕ := brother_baskets * strawberries_per_basket

def kimberly_multiplier : ℕ := 8
def kimberly_strawberries : ℕ := kimberly_multiplier * brother_strawberries

def parents_difference : ℕ := 93
def parents_strawberries : ℕ := kimberly_strawberries - parents_difference

def total_strawberries : ℕ := kimberly_strawberries + brother_strawberries + parents_strawberries
def total_people : ℕ := 4

def equal_share : ℕ := total_strawberries / total_people

theorem equal_share_is_168 :
  equal_share = 168 := by
  -- We state that for the given problem conditions,
  -- the total number of strawberries divided equally among the family members results in 168 strawberries per person.
  sorry

end StrawberryProblem

end equal_share_is_168_l226_226573


namespace cover_tiles_count_l226_226499

-- Definitions corresponding to the conditions
def tile_side : ℕ := 6 -- in inches
def tile_area : ℕ := tile_side * tile_side -- area of one tile in square inches

def region_length : ℕ := 3 * 12 -- 3 feet in inches
def region_width : ℕ := 6 * 12 -- 6 feet in inches
def region_area : ℕ := region_length * region_width -- area of the region in square inches

-- The statement of the proof problem
theorem cover_tiles_count : (region_area / tile_area) = 72 :=
by
   -- Proof would be filled in here
   sorry

end cover_tiles_count_l226_226499


namespace six_positive_integers_solution_count_l226_226950

theorem six_positive_integers_solution_count :
  ∃ (S : Finset (Finset ℕ)) (n : ℕ) (a b c x y z : ℕ), 
  a ≥ b → b ≥ c → x ≥ y → y ≥ z → 
  a + b + c = x * y * z → 
  x + y + z = a * b * c → 
  S.card = 7 := by
    sorry

end six_positive_integers_solution_count_l226_226950


namespace purely_periodic_period_le_T_l226_226040

theorem purely_periodic_period_le_T {a b : ℚ} (T : ℕ) 
  (ha : ∃ m, a = m / (10^T - 1)) 
  (hb : ∃ n, b = n / (10^T - 1)) :
  (∃ T₁, T₁ ≤ T ∧ ∃ p, a = p / (10^T₁ - 1)) ∧ 
  (∃ T₂, T₂ ≤ T ∧ ∃ q, b = q / (10^T₂ - 1)) := 
sorry

end purely_periodic_period_le_T_l226_226040


namespace smallest_n_sum_gt_10_pow_5_l226_226593

theorem smallest_n_sum_gt_10_pow_5 :
  ∃ (n : ℕ), (n ≥ 142) ∧ (5 * n^2 + 4 * n ≥ 100000) :=
by
  use 142
  sorry

end smallest_n_sum_gt_10_pow_5_l226_226593


namespace proof_inequalities_l226_226860

theorem proof_inequalities (A B C D E : ℝ) (p q r s t : ℝ)
  (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D < E)
  (h5 : p = B - A) (h6 : q = C - A) (h7 : r = D - A)
  (h8 : s = E - B) (h9 : t = E - D)
  (ineq1 : p + 2 * s > r + t)
  (ineq2 : r + t > p)
  (ineq3 : r + t > s) :
  (p < r / 2) ∧ (s < t + p / 2) :=
by 
  sorry

end proof_inequalities_l226_226860


namespace max_value_xy_xz_yz_l226_226298

theorem max_value_xy_xz_yz (x y z : ℝ) (h : x + 2 * y + z = 6) :
  xy + xz + yz ≤ 6 :=
sorry

end max_value_xy_xz_yz_l226_226298


namespace ratio_G_to_C_is_1_1_l226_226820

variable (R C G : ℕ)

-- Given conditions
def Rover_has_46_spots : Prop := R = 46
def Cisco_has_half_R_minus_5 : Prop := C = R / 2 - 5
def Granger_Cisco_combined_108 : Prop := G + C = 108
def Granger_Cisco_equal : Prop := G = C

-- Theorem stating the final answer to the problem
theorem ratio_G_to_C_is_1_1 (h1 : Rover_has_46_spots R) 
                            (h2 : Cisco_has_half_R_minus_5 C R) 
                            (h3 : Granger_Cisco_combined_108 G C) 
                            (h4 : Granger_Cisco_equal G C) : 
                            G / C = 1 := by
  sorry

end ratio_G_to_C_is_1_1_l226_226820


namespace necessary_and_sufficient_condition_l226_226812

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (a^2 + 4 * a - 5 > 0) ↔ (|a + 2| > 3) := sorry

end necessary_and_sufficient_condition_l226_226812


namespace find_cost_price_l226_226058

noncomputable def cost_price (CP SP_loss SP_gain : ℝ) : Prop :=
SP_loss = 0.90 * CP ∧
SP_gain = 1.05 * CP ∧
(SP_gain - SP_loss = 225)

theorem find_cost_price (CP : ℝ) (h : cost_price CP (0.90 * CP) (1.05 * CP)) : CP = 1500 :=
by
  sorry

end find_cost_price_l226_226058


namespace calculate_regular_rate_l226_226597

def regular_hours_per_week : ℕ := 6 * 10
def total_weeks : ℕ := 4
def total_regular_hours : ℕ := regular_hours_per_week * total_weeks
def total_worked_hours : ℕ := 245
def overtime_hours : ℕ := total_worked_hours - total_regular_hours
def overtime_rate : ℚ := 4.20
def total_earning : ℚ := 525
def total_overtime_pay : ℚ := overtime_hours * overtime_rate
def total_regular_pay : ℚ := total_earning - total_overtime_pay
def regular_rate : ℚ := total_regular_pay / total_regular_hours

theorem calculate_regular_rate : regular_rate = 2.10 :=
by
  -- The proof would go here
  sorry

end calculate_regular_rate_l226_226597


namespace value_of_f_2012_l226_226033

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom odd_fn : odd_function f
axiom f_at_2 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem value_of_f_2012 : f 2012 = 0 :=
by
  sorry

end value_of_f_2012_l226_226033


namespace percentage_increase_l226_226686

theorem percentage_increase (P : ℕ) (x y : ℕ) (h1 : x = 5) (h2 : y = 7) 
    (h3 : (x * (1 + P / 100) / (y * (1 - 10 / 100))) = 20 / 21) : 
    P = 20 :=
by
  sorry

end percentage_increase_l226_226686


namespace quadratic_real_roots_l226_226962

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ k ≥ -5 ∧ k ≠ -1 :=
by
  sorry

end quadratic_real_roots_l226_226962


namespace division_quoitient_l226_226663

-- Let f be the polynomial we want to divide
def f : Polynomial ℤ := X^6 - 2*X^5 + 3*X^4 - 4*X^3 + 5*X^2 - 6*X + 12

-- Let g be the divisor
def g : Polynomial ℤ := X - 1

-- The quotient q derived from the division
def q : Polynomial ℤ := X^5 - X^4 + 2*X^3 - 2*X^2 + 3*X - 3

-- The remainder r derived from the division
def r : Polynomial ℤ := 9

theorem division_quoitient : f = g * q + r := by
  sorry

end division_quoitient_l226_226663


namespace multiplicative_inverse_modulo_l226_226157

theorem multiplicative_inverse_modulo :
  let A := 222222
  let B := 142857
  let M := 2000000
  let N := 126
  N < 1000000 ∧ N * (A * B) % M = 1 :=
by
  let A := 222222
  let B := 142857
  let M := 2000000
  let N := 126
  have h1 : A = 222222 := rfl
  have h2 : B = 142857 := rfl
  have h3 : M = 2000000 := rfl
  have h4 : N = 126 := rfl
  exact And.intro
    (show N < 1000000 from sorry)
    (show N * (A * B) % M = 1 from sorry)

end multiplicative_inverse_modulo_l226_226157


namespace triangle_area_correct_l226_226968

noncomputable def area_of_triangle 
  (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) : ℝ :=
  let cosC := (b^2 + c^2 - a^2) / (2 * b * c)
  let sinC := Real.sqrt (1 - cosC^2)
  (1 / 2) * b * c * sinC

theorem triangle_area_correct : area_of_triangle (Real.sqrt 29) (Real.sqrt 13) (Real.sqrt 34) 
  (by rfl) (by rfl) (by rfl) = 19 / 2 :=
sorry

end triangle_area_correct_l226_226968


namespace gcd_polynomial_l226_226676

-- Define conditions
variables (b : ℤ) (k : ℤ)

-- Assume b is an even multiple of 8753
def is_even_multiple_of_8753 (b : ℤ) : Prop := ∃ k : ℤ, b = 2 * 8753 * k

-- Statement to be proven
theorem gcd_polynomial (b : ℤ) (h : is_even_multiple_of_8753 b) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 :=
by sorry

end gcd_polynomial_l226_226676


namespace union_sets_l226_226849

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_sets : S ∪ T = {0, 1, 3} :=
by
  sorry

end union_sets_l226_226849


namespace fraction_b_not_whole_l226_226200

-- Defining the fractions as real numbers
def fraction_a := 60 / 12
def fraction_b := 60 / 8
def fraction_c := 60 / 5
def fraction_d := 60 / 4
def fraction_e := 60 / 3

-- Defining what it means to be a whole number
def is_whole_number (x : ℝ) : Prop := ∃ (n : ℤ), x = n

-- Theorem stating that fraction_b is not a whole number
theorem fraction_b_not_whole : ¬ is_whole_number fraction_b := 
by 
-- proof to be filled in
sorry

end fraction_b_not_whole_l226_226200


namespace max_discriminant_l226_226716

noncomputable def f (a b c x : ℤ) := a * x^2 + b * x + c

theorem max_discriminant (a b c u v w : ℤ)
  (h1 : u ≠ v) (h2 : v ≠ w) (h3 : u ≠ w)
  (hu : f a b c u = 0)
  (hv : f a b c v = 0)
  (hw : f a b c w = 2) :
  ∃ (a b c : ℤ), b^2 - 4 * a * c = 16 :=
sorry

end max_discriminant_l226_226716


namespace factorize_expression_l226_226798

theorem factorize_expression (x : ℝ) : 2 * x ^ 2 - 50 = 2 * (x + 5) * (x - 5) := 
  sorry

end factorize_expression_l226_226798


namespace gribblean_words_count_l226_226291

universe u

-- Define the Gribblean alphabet size
def alphabet_size : Nat := 3

-- Words of length 1 to 4
def words_of_length (n : Nat) : Nat :=
  alphabet_size ^ n

-- All possible words count
def total_words : Nat :=
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3) + (words_of_length 4)

-- Theorem statement
theorem gribblean_words_count : total_words = 120 :=
by
  sorry

end gribblean_words_count_l226_226291


namespace cuboid_height_l226_226958

-- Define the base area and volume of the cuboid
def base_area : ℝ := 50
def volume : ℝ := 2000

-- Prove that the height is 40 cm given the base area and volume
theorem cuboid_height : volume / base_area = 40 := by
  sorry

end cuboid_height_l226_226958


namespace quadratic_inequality_solution_l226_226241

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end quadratic_inequality_solution_l226_226241


namespace range_of_a_l226_226124

theorem range_of_a (M : Set ℝ) (a : ℝ) :
  (M = {x | x^2 - 4 * x + 4 * a < 0}) →
  ¬(2 ∈ M) →
  (1 ≤ a) :=
by
  -- Given assumptions
  intros hM h2_notin_M
  -- Convert h2_notin_M to an inequality and prove the desired result
  sorry

end range_of_a_l226_226124


namespace perimeter_of_equilateral_triangle_l226_226884

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l226_226884


namespace rubble_initial_money_l226_226307

def initial_money (cost_notebook cost_pen : ℝ) (num_notebooks num_pens : ℕ) (money_left : ℝ) : ℝ :=
  (num_notebooks * cost_notebook + num_pens * cost_pen) + money_left

theorem rubble_initial_money :
  initial_money 4 1.5 2 2 4 = 15 :=
by
  sorry

end rubble_initial_money_l226_226307


namespace kiki_total_money_l226_226017

theorem kiki_total_money 
  (S : ℕ) (H : ℕ) (M : ℝ)
  (h1: S = 18)
  (h2: H = 2 * S)
  (h3: 0.40 * M = 36) : 
  M = 90 :=
by
  sorry

end kiki_total_money_l226_226017


namespace find_positive_n_l226_226744

def arithmetic_sequence (a d : ℤ) (n : ℤ) := a + (n - 1) * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

theorem find_positive_n :
  ∃ (n : ℕ), n > 0 ∧ ∀ a d : ℤ, a = -12 → sum_of_first_n_terms a d 13 = 0 → arithmetic_sequence a d n > 0 ∧ n = 8 := 
sorry

end find_positive_n_l226_226744


namespace five_segments_acute_angle_l226_226251

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_obtuse (a b c : ℝ) : Prop :=
  c^2 > a^2 + b^2

def is_acute (a b c : ℝ) : Prop :=
  c^2 < a^2 + b^2

theorem five_segments_acute_angle (a b c d e : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (T1 : is_triangle a b c) (T2 : is_triangle a b d) (T3 : is_triangle a b e)
  (T4 : is_triangle a c d) (T5 : is_triangle a c e) (T6 : is_triangle a d e)
  (T7 : is_triangle b c d) (T8 : is_triangle b c e) (T9 : is_triangle b d e)
  (T10 : is_triangle c d e) : 
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
           is_triangle x y z ∧ is_acute x y z :=
by
  sorry

end five_segments_acute_angle_l226_226251


namespace total_hats_l226_226581

noncomputable def num_adults := 1500
noncomputable def proportion_men := (2 : ℚ) / 3
noncomputable def proportion_women := 1 - proportion_men
noncomputable def proportion_men_hats := (15 : ℚ) / 100
noncomputable def proportion_women_hats := (10 : ℚ) / 100

noncomputable def num_men := proportion_men * num_adults
noncomputable def num_women := proportion_women * num_adults
noncomputable def num_men_hats := proportion_men_hats * num_men
noncomputable def num_women_hats := proportion_women_hats * num_women

noncomputable def total_adults_with_hats := num_men_hats + num_women_hats

theorem total_hats : total_adults_with_hats = 200 := 
by
  sorry

end total_hats_l226_226581


namespace eccentricity_of_ellipse_l226_226007

-- Definitions
variable (a b c : ℝ)  -- semi-major axis, semi-minor axis, and distance from center to a focus
variable (h_c_eq_b : c = b)  -- given condition focal length equals length of minor axis
variable (h_a_eq_sqrt_sum : a = Real.sqrt (c^2 + b^2))  -- relationship in ellipse

-- Question: Prove the eccentricity of the ellipse e = √2 / 2
theorem eccentricity_of_ellipse : (c = b) → (a = Real.sqrt (c^2 + b^2)) → (c / a = Real.sqrt 2 / 2) :=
by
  intros h_c_eq_b h_a_eq_sqrt_sum
  sorry

end eccentricity_of_ellipse_l226_226007


namespace average_calculation_l226_226729

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 4 1) (average_two 3 2) 5 = 59 / 18 :=
by
  sorry

end average_calculation_l226_226729


namespace evaluation_l226_226525
-- Import the entire Mathlib library

-- Define the operations triangle and nabla
def triangle (a b : ℕ) : ℕ := 3 * a + 2 * b
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

-- The proof statement
theorem evaluation : triangle 2 (nabla 3 4) = 42 :=
by
  -- Provide a placeholder for the proof
  sorry

end evaluation_l226_226525


namespace find_real_numbers_l226_226956

theorem find_real_numbers :
  ∀ (x y z : ℝ), x^2 - y*z = |y - z| + 1 ∧ y^2 - z*x = |z - x| + 1 ∧ z^2 - x*y = |x - y| + 1 ↔
  (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
  (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
  (x = -5/3 ∧ y = 4/3 ∧ z = 4/3) ∨
  (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
  (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
  (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) :=
by
  sorry

end find_real_numbers_l226_226956


namespace eval_expression_l226_226796

theorem eval_expression :
  16^3 + 3 * (16^2) * 2 + 3 * 16 * (2^2) + 2^3 = 5832 :=
by
  sorry

end eval_expression_l226_226796


namespace roger_initial_money_l226_226444

theorem roger_initial_money (spent_on_game : ℕ) (cost_per_toy : ℕ) (num_toys : ℕ) (total_money_spent : ℕ) :
  spent_on_game = 48 →
  cost_per_toy = 3 →
  num_toys = 5 →
  total_money_spent = spent_on_game + num_toys * cost_per_toy →
  total_money_spent = 63 :=
by
  intros h_game h_toy_cost h_num_toys h_total_spent
  rw [h_game, h_toy_cost, h_num_toys] at h_total_spent
  exact h_total_spent

end roger_initial_money_l226_226444


namespace pair_C_does_not_produce_roots_l226_226741

theorem pair_C_does_not_produce_roots (x : ℝ) :
  (x = 0 ∨ x = 2) ↔ (∃ x, y = x ∧ y = x - 2) = false :=
by
  sorry

end pair_C_does_not_produce_roots_l226_226741


namespace largest_marbles_l226_226764

theorem largest_marbles {n : ℕ} (h1 : n < 400) (h2 : n % 3 = 1) (h3 : n % 7 = 2) (h4 : n % 5 = 0) : n = 310 :=
  sorry

end largest_marbles_l226_226764


namespace radius_of_circle_l226_226731

theorem radius_of_circle (r x y : ℝ): 
  x = π * r^2 → 
  y = 2 * π * r → 
  x - y = 72 * π → 
  r = 12 := 
by 
  sorry

end radius_of_circle_l226_226731


namespace product_fraction_l226_226356

theorem product_fraction :
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693 / 256 := by
  sorry

end product_fraction_l226_226356


namespace max_good_numberings_l226_226293

noncomputable def goodNumberings : finset (fin (8 × 8)) := sorry

theorem max_good_numberings (points : finset (ℝ × ℝ)) (h_distinct : points.card = 8) :
  goodNumberings.card = 56 :=
sorry

end max_good_numberings_l226_226293


namespace trig_intersection_identity_l226_226881

theorem trig_intersection_identity (x0 : ℝ) (hx0 : x0 ≠ 0) (htan : -x0 = Real.tan x0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
sorry

end trig_intersection_identity_l226_226881


namespace evaluate_expression_l226_226239

theorem evaluate_expression :
  3000 * (3000 ^ 1500 + 3000 ^ 1500) = 2 * 3000 ^ 1501 :=
by sorry

end evaluate_expression_l226_226239


namespace impossible_coins_l226_226868

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l226_226868


namespace equilateral_triangle_perimeter_l226_226887

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end equilateral_triangle_perimeter_l226_226887


namespace find_digit_n_l226_226038

theorem find_digit_n :
  let n := 2 in
  (9 * 11 * 13 * 15 * 17 = 3 * 100000 + n * 10000 + 8185) ∧
  ((3 + n + 8 + 1 + 8 + 1 + 5) % 9 = 0) :=
by
  sorry

end find_digit_n_l226_226038


namespace solve_equation_l226_226660

theorem solve_equation (x : ℝ) (h₀ : x ≠ -3) (h₁ : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : x = 9 :=
by
  sorry

end solve_equation_l226_226660


namespace probability_eq_l226_226766

noncomputable def probability_exactly_two_one_digit_and_three_two_digit : ℚ := 
  let n := 5
  let p_one_digit := 9 / 20
  let p_two_digit := 11 / 20
  let binomial_coeff := Nat.choose 5 2
  (binomial_coeff * p_one_digit^2 * p_two_digit^3)

theorem probability_eq : probability_exactly_two_one_digit_and_three_two_digit = 539055 / 1600000 := 
  sorry

end probability_eq_l226_226766


namespace evaluate_fraction_l226_226953

theorem evaluate_fraction (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 :=
sorry

end evaluate_fraction_l226_226953


namespace greatest_integer_x_l226_226055

theorem greatest_integer_x (x : ℤ) : 
  (∀ x : ℤ, (8 / 11 : ℝ) > (x / 17) → x ≤ 12) ∧ (8 / 11 : ℝ) > (12 / 17) :=
sorry

end greatest_integer_x_l226_226055


namespace mean_reciprocals_first_three_composites_l226_226368

theorem mean_reciprocals_first_three_composites :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = (13 : ℚ) / 72 := 
by
  sorry

end mean_reciprocals_first_three_composites_l226_226368


namespace problem_f_2009_plus_f_2010_l226_226115

theorem problem_f_2009_plus_f_2010 (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (2 * x + 1) = f (2 * (x + 5 / 2) + 1))
  (h_f1 : f 1 = 5) :
  f 2009 + f 2010 = 0 :=
sorry

end problem_f_2009_plus_f_2010_l226_226115


namespace golden_section_length_l226_226404

theorem golden_section_length (MN : ℝ) (MP NP : ℝ) (hMN : MN = 1) (hP : MP + NP = MN) (hgolden : MN / MP = MP / NP) (hMP_gt_NP : MP > NP) : MP = (Real.sqrt 5 - 1) / 2 :=
by sorry

end golden_section_length_l226_226404


namespace total_crayons_lost_or_given_away_l226_226858

def crayons_given_away : ℕ := 52
def crayons_lost : ℕ := 535

theorem total_crayons_lost_or_given_away :
  crayons_given_away + crayons_lost = 587 :=
by
  sorry

end total_crayons_lost_or_given_away_l226_226858


namespace smallest_odd_number_divisible_by_3_l226_226901

theorem smallest_odd_number_divisible_by_3 : ∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, (m % 2 = 1 ∧ m % 3 = 0) → m ≥ n := 
by
  sorry

end smallest_odd_number_divisible_by_3_l226_226901


namespace find_floor_of_apt_l226_226348

-- Define the conditions:
-- Number of stories
def num_stories : Nat := 9
-- Number of entrances
def num_entrances : Nat := 10
-- Total apartments in entrance 10
def apt_num : Nat := 333
-- Number of apartments per floor in each entrance (which is to be found)
def apts_per_floor_per_entrance : Nat := 4 -- from solution b)

-- Assertion: The floor number that apartment number 333 is on in entrance 10
theorem find_floor_of_apt (num_stories num_entrances apt_num apts_per_floor_per_entrance : ℕ) :
  1 ≤ apt_num ∧ apt_num ≤ num_stories * num_entrances * apts_per_floor_per_entrance →
  (apt_num - 1) / apts_per_floor_per_entrance + 1 = 3 :=
by
  sorry

end find_floor_of_apt_l226_226348


namespace find_number_l226_226274

theorem find_number (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := 
sorry

end find_number_l226_226274


namespace movie_hours_sum_l226_226150

noncomputable def total_movie_hours 
  (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : ℕ :=
  Joyce + Michael + Nikki + Ryn + Sam

theorem movie_hours_sum (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : 
  total_movie_hours Michael Joyce Nikki Ryn Sam h1 h2 h3 h4 h5 = 94 :=
by 
  -- The actual proof will go here, to demonstrate the calculations resulting in 94 hours
  sorry

end movie_hours_sum_l226_226150


namespace find_cost_price_l226_226335

def selling_price : ℝ := 150
def profit_percentage : ℝ := 25

theorem find_cost_price (cost_price : ℝ) (h : profit_percentage = ((selling_price - cost_price) / cost_price) * 100) : 
  cost_price = 120 := 
sorry

end find_cost_price_l226_226335


namespace number_of_possible_points_C_of_conditions_l226_226123

noncomputable def number_of_possible_points_C (line : ℝ × ℝ × ℝ) (circle_center : ℝ × ℝ) (circle_radius : ℝ) (area_triangle_ABC : ℝ) : ℕ :=
sorry

theorem number_of_possible_points_C_of_conditions :
  number_of_possible_points_C (3, 4, -15) (0, 0) 5 8 = 3 :=
by
  sorry

end number_of_possible_points_C_of_conditions_l226_226123


namespace percentage_defective_units_shipped_l226_226833

noncomputable def defective_percent : ℝ := 0.07
noncomputable def shipped_percent : ℝ := 0.05

theorem percentage_defective_units_shipped :
  defective_percent * shipped_percent * 100 = 0.35 :=
by
  -- Proof body here
  sorry

end percentage_defective_units_shipped_l226_226833


namespace f_sum_lt_zero_l226_226969

theorem f_sum_lt_zero {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_monotone : ∀ x y, x < y → f y < f x)
  (α β γ : ℝ) (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end f_sum_lt_zero_l226_226969


namespace millionaire_allocation_l226_226321

def numWaysToAllocateMillionaires (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem millionaire_allocation :
  numWaysToAllocateMillionaires 13 3 = 36 :=
by
  sorry

end millionaire_allocation_l226_226321


namespace john_running_speed_l226_226417

noncomputable def find_running_speed (x : ℝ) : Prop :=
  (12 / (3 * x + 2) + 8 / x = 2.2)

theorem john_running_speed : ∃ x : ℝ, find_running_speed x ∧ abs (x - 0.47) < 0.01 :=
by
  sorry

end john_running_speed_l226_226417


namespace no_valid_sequence_of_integers_from_1_to_2004_l226_226081

theorem no_valid_sequence_of_integers_from_1_to_2004 :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ 2004) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ k, 1 ≤ k ∧ k + 9 ≤ 2004 → 
      (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + 
       a (k + 6) + a (k + 7) + a (k + 8) + a (k + 9)) % 10 = 0) :=
  sorry

end no_valid_sequence_of_integers_from_1_to_2004_l226_226081


namespace hammerhead_teeth_fraction_l226_226500

theorem hammerhead_teeth_fraction (f : ℚ) : 
  let t := 180 
  let h := f * t
  let w := 2 * (t + h)
  w = 420 → f = (1 : ℚ) / 6 := by
  intros _ 
  sorry

end hammerhead_teeth_fraction_l226_226500


namespace measure_of_angle_F_l226_226566

theorem measure_of_angle_F (D E F : ℝ) (hD : D = E) 
  (hF : F = D + 40) (h_sum : D + E + F = 180) : F = 140 / 3 + 40 :=
by
  sorry

end measure_of_angle_F_l226_226566


namespace like_terms_sum_l226_226810

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 2) (h2 : n = 3) : m + n = 4 :=
sorry

end like_terms_sum_l226_226810


namespace total_toads_l226_226046

def pond_toads : ℕ := 12
def outside_toads : ℕ := 6

theorem total_toads : pond_toads + outside_toads = 18 :=
by
  -- Proof goes here
  sorry

end total_toads_l226_226046


namespace coffee_maker_capacity_l226_226893

theorem coffee_maker_capacity (x : ℝ) (h : 0.36 * x = 45) : x = 125 :=
sorry

end coffee_maker_capacity_l226_226893


namespace cows_sold_l226_226776

/-- 
A man initially had 39 cows, 25 of them died last year, he sold some remaining cows, this year,
the number of cows increased by 24, he bought 43 more cows, his friend gave him 8 cows.
Now, he has 83 cows. How many cows did he sell last year?
-/
theorem cows_sold (S : ℕ) : (39 - 25 - S + 24 + 43 + 8 = 83) → S = 6 :=
by
  intro h
  sorry

end cows_sold_l226_226776


namespace ratio_of_blue_fish_to_total_fish_l226_226747

-- Define the given conditions
def total_fish : ℕ := 30
def blue_spotted_fish : ℕ := 5
def half (n : ℕ) : ℕ := n / 2

-- Calculate the number of blue fish using the conditions
def blue_fish : ℕ := blue_spotted_fish * 2

-- Define the ratio of blue fish to total fish
def ratio (num denom : ℕ) : ℚ := num / denom

-- The theorem to prove
theorem ratio_of_blue_fish_to_total_fish :
  ratio blue_fish total_fish = 1 / 3 := by
  sorry

end ratio_of_blue_fish_to_total_fish_l226_226747


namespace geometric_sequence_value_of_b_l226_226318

theorem geometric_sequence_value_of_b : 
  ∃ b : ℝ, 180 * (b / 180) = b ∧ (b / 180) * b = 64 / 25 ∧ b > 0 ∧ b = 21.6 :=
by sorry

end geometric_sequence_value_of_b_l226_226318


namespace lcm_18_30_l226_226478

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l226_226478


namespace lisa_pizza_l226_226854

theorem lisa_pizza (P H S : ℕ) 
  (h1 : H = 2 * P) 
  (h2 : S = P + 12) 
  (h3 : P + H + S = 132) : 
  P = 30 := 
by
  sorry

end lisa_pizza_l226_226854


namespace mul_same_base_exp_ten_pow_1000_sq_l226_226337

theorem mul_same_base_exp (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

-- Given specific constants for this problem
theorem ten_pow_1000_sq : (10:ℝ)^(1000) * (10)^(1000) = (10)^(2000) := by
  exact mul_same_base_exp 10 1000 1000

end mul_same_base_exp_ten_pow_1000_sq_l226_226337


namespace quadratic_equation_formulation_l226_226101

theorem quadratic_equation_formulation (a b c : ℝ) (x₁ x₂ : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = -b / a)
  (h₅ : x₁ * x₂ = c / a) :
  ∃ (y : ℝ), a^2 * y^2 + a * (b - c) * y - b * c = 0 :=
by
  sorry

end quadratic_equation_formulation_l226_226101


namespace equation_has_real_roots_for_all_K_l226_226526

open Real

noncomputable def original_equation (K x : ℝ) : ℝ :=
  x - K^3 * (x - 1) * (x - 3)

theorem equation_has_real_roots_for_all_K :
  ∀ K : ℝ, ∃ x : ℝ, original_equation K x = 0 :=
sorry

end equation_has_real_roots_for_all_K_l226_226526


namespace simple_interest_rate_l226_226184

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 15000) (hSI : SI = 6000) (hT : T = 8) :
  ∃ R : ℝ, (SI = P * R * T / 100) ∧ R = 5 :=
by
  use 5
  field_simp [hP, hSI, hT]
  sorry

end simple_interest_rate_l226_226184


namespace pipe_A_fill_time_l226_226340

theorem pipe_A_fill_time (B C : ℝ) (hB : B = 8) (hC : C = 14.4) (hB_not_zero : B ≠ 0) (hC_not_zero : C ≠ 0) :
  ∃ (A : ℝ), (1 / A + 1 / B = 1 / C) ∧ A = 24 :=
by
  sorry

end pipe_A_fill_time_l226_226340


namespace line_parallel_plane_l226_226107

axiom line (m : Type) : Prop
axiom plane (α : Type) : Prop
axiom has_no_common_points (m : Type) (α : Type) : Prop
axiom parallel (m : Type) (α : Type) : Prop

theorem line_parallel_plane
  (m : Type) (α : Type)
  (h : has_no_common_points m α) : parallel m α := sorry

end line_parallel_plane_l226_226107


namespace remainder_of_x13_plus_1_by_x_minus_1_l226_226195

-- Define the polynomial f(x) = x^13 + 1
def f (x : ℕ) : ℕ := x ^ 13 + 1

-- State the theorem using the Polynomial Remainder Theorem
theorem remainder_of_x13_plus_1_by_x_minus_1 : f 1 = 2 := by
  -- Skip the proof
  sorry

end remainder_of_x13_plus_1_by_x_minus_1_l226_226195


namespace base_three_to_base_ten_l226_226086

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end base_three_to_base_ten_l226_226086


namespace f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l226_226536

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x / (1 + x) else 1 / (1 + x)

theorem f_property (x : ℝ) (hx : 0 < x) : 
  f x = f (1 / x) :=
by
  sorry

theorem f_equals_when_x_lt_1 (x : ℝ) (hx0 : 0 < x) (hx1 : x < 1) : 
  f x = 1 / (1 + x) :=
by
  sorry

theorem f_equals_when_x_gt_1 (x : ℝ) (hx : 1 < x) : 
  f x = x / (1 + x) :=
by
  sorry

end f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l226_226536


namespace probability_both_dice_greater_than_4_l226_226612

def ProbabilitySingleDieGreaterThan4 : ℚ := 2 / 6

theorem probability_both_dice_greater_than_4 :
  (ProbabilitySingleDieGreaterThan4 * ProbabilitySingleDieGreaterThan4) = (1 / 9) :=
by
  sorry

end probability_both_dice_greater_than_4_l226_226612


namespace tangent_line_parabola_l226_226805

theorem tangent_line_parabola (k : ℝ) 
  (h : ∀ (x y : ℝ), 4 * x + 6 * y + k = 0 → y^2 = 32 * x) : k = 72 := 
sorry

end tangent_line_parabola_l226_226805


namespace sqrt_t6_plus_t4_l226_226247

open Real

theorem sqrt_t6_plus_t4 (t : ℝ) : sqrt (t^6 + t^4) = t^2 * sqrt (t^2 + 1) :=
by sorry

end sqrt_t6_plus_t4_l226_226247


namespace fraction_division_l226_226608

theorem fraction_division (a b c d e : ℚ)
  (h1 : a = 3 / 7)
  (h2 : b = 1 / 3)
  (h3 : d = 2 / 5)
  (h4 : c = a + b)
  (h5 : e = c / d):
  e = 40 / 21 := by
  sorry

end fraction_division_l226_226608


namespace problem_xy_l226_226690

theorem problem_xy (x y : ℝ) (h1 : x + y = 25) (h2 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 :=
by
  sorry

end problem_xy_l226_226690


namespace cubic_equation_solution_bound_l226_226527

theorem cubic_equation_solution_bound (a : ℝ) :
  a ∈ Set.Ici (-15) → ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ → x₂ ≠ x₃ → x₁ ≠ x₃ →
  (x₁^3 + 6 * x₁^2 + a * x₁ + 8 = 0) →
  (x₂^3 + 6 * x₂^2 + a * x₂ + 8 = 0) →
  (x₃^3 + 6 * x₃^2 + a * x₃ + 8 = 0) →
  False := 
sorry

end cubic_equation_solution_bound_l226_226527


namespace even_numbers_count_l226_226822

theorem even_numbers_count (a b : ℕ) (h1 : 150 < a) (h2 : a % 2 = 0) (h3 : b < 350) (h4 : b % 2 = 0) (h5 : 150 < b) (h6 : a < 350) (h7 : 154 ≤ b) (h8 : a ≤ 152) :
  ∃ n : ℕ, ∀ k : ℕ, k = 99 ↔ 2 * k + 150 = b - a + 2 :=
by
  sorry

end even_numbers_count_l226_226822


namespace inequality_solution_l226_226972

noncomputable def inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : Prop :=
  (x^4 + y^4 + z^4) ≥ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ∧ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ≥ (x * y * z * (x + y + z))

theorem inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  inequality_proof x y z hx hy hz :=
by 
  sorry

end inequality_solution_l226_226972


namespace amount_after_two_years_l226_226059

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (hP : P = 3200) (hr : r = 1 / 8) (hn : n = 2) :
  P * (1 + r) ^ n = 4050 :=
by
  rw [hP, hr, hn]
  norm_num
  sorry

end amount_after_two_years_l226_226059


namespace NicoleEndsUpWith36Pieces_l226_226719

namespace ClothingProblem

noncomputable def NicoleClothesStart := 10
noncomputable def FirstOlderSisterClothes := NicoleClothesStart / 2
noncomputable def NextOldestSisterClothes := NicoleClothesStart + 2
noncomputable def OldestSisterClothes := (NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes) / 3

theorem NicoleEndsUpWith36Pieces : 
  NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes + OldestSisterClothes = 36 :=
  by
    sorry

end ClothingProblem

end NicoleEndsUpWith36Pieces_l226_226719


namespace parabola_equation_trajectory_midpoint_l226_226542

-- Given data and conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola_x3 (p : ℝ) : Prop := ∃ y, parabola p 3 y
def distance_point_to_line (x d : ℝ) : Prop := x + d = 5

-- Prove that given these conditions, the parabola equation is y^2 = 8x
theorem parabola_equation (p : ℝ) (h1 : point_on_parabola_x3 p) (h2 : distance_point_to_line (3 + p / 2) 2) : p = 4 :=
sorry

-- Prove the equation of the trajectory for the midpoint of the line segment FP
def point_on_parabola (p x y : ℝ) : Prop := y^2 = 8 * x
theorem trajectory_midpoint (p x y : ℝ) (h1 : parabola 4 x y) : y^2 = 4 * (x - 1) :=
sorry

end parabola_equation_trajectory_midpoint_l226_226542


namespace min_colors_needed_correct_l226_226754

-- Define the 5x5 grid as a type
def Grid : Type := Fin 5 × Fin 5

-- Define a coloring as a function from Grid to a given number of colors
def Coloring (colors : Type) : Type := Grid → colors

-- Define the property where in any row, column, or diagonal, no three consecutive cells have the same color
def valid_coloring (colors : Type) (C : Coloring colors) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 3, ( C (i, j) ≠ C (i, j + 1) ∧ C (i, j + 1) ≠ C (i, j + 2) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 5, ( C (i, j) ≠ C (i + 1, j) ∧ C (i + 1, j) ≠ C (i + 2, j) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 3, ( C (i, j) ≠ C (i + 1, j + 1) ∧ C (i + 1, j + 1) ≠ C (i + 2, j + 2) )

-- Define the minimum number of colors required
def min_colors_needed : Nat := 5

-- Prove the statement
theorem min_colors_needed_correct : ∃ C : Coloring (Fin min_colors_needed), valid_coloring (Fin min_colors_needed) C :=
sorry

end min_colors_needed_correct_l226_226754


namespace average_weight_of_rock_l226_226848

-- Define all the conditions
def price_per_pound : ℝ := 4
def total_amount : ℝ := 60
def number_of_rocks : ℕ := 10

-- The statement we need to prove
theorem average_weight_of_rock :
  (total_amount / price_per_pound) / number_of_rocks = 1.5 :=
sorry

end average_weight_of_rock_l226_226848


namespace g_neg_one_l226_226114

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h₀ : ∀ x : ℝ, f (-x) + x^2 = -(f x + x^2))
variables (h₁ : f 1 = 1)
variables (h₂ : ∀ x : ℝ, g x = f x + 2)

theorem g_neg_one : g (-1) = -1 :=
by
  sorry

end g_neg_one_l226_226114


namespace liquid_X_percentage_in_new_solution_l226_226061

noncomputable def solutionY_initial_kg : ℝ := 10
noncomputable def percentage_liquid_X : ℝ := 0.30
noncomputable def evaporated_water_kg : ℝ := 2
noncomputable def added_solutionY_kg : ℝ := 2

-- Calculate the amount of liquid X in the original solution
noncomputable def initial_liquid_X_kg : ℝ :=
  percentage_liquid_X * solutionY_initial_kg

-- Calculate the remaining weight after evaporation
noncomputable def remaining_weight_kg : ℝ :=
  solutionY_initial_kg - evaporated_water_kg

-- Calculate the amount of liquid X after evaporation
noncomputable def remaining_liquid_X_kg : ℝ := initial_liquid_X_kg

-- Since only water evaporates, remaining water weight
noncomputable def remaining_water_kg : ℝ :=
  remaining_weight_kg - remaining_liquid_X_kg

-- Calculate the amount of liquid X in the added solution
noncomputable def added_liquid_X_kg : ℝ :=
  percentage_liquid_X * added_solutionY_kg

-- Total liquid X in the new solution
noncomputable def new_liquid_X_kg : ℝ :=
  remaining_liquid_X_kg + added_liquid_X_kg

-- Calculate the water in the added solution
noncomputable def percentage_water : ℝ := 0.70
noncomputable def added_water_kg : ℝ :=
  percentage_water * added_solutionY_kg

-- Total water in the new solution
noncomputable def new_water_kg : ℝ :=
  remaining_water_kg + added_water_kg

-- Total weight of the new solution
noncomputable def new_total_weight_kg : ℝ :=
  remaining_weight_kg + added_solutionY_kg

-- Percentage of liquid X in the new solution
noncomputable def percentage_new_liquid_X : ℝ :=
  (new_liquid_X_kg / new_total_weight_kg) * 100

-- The proof statement
theorem liquid_X_percentage_in_new_solution :
  percentage_new_liquid_X = 36 :=
by
  sorry

end liquid_X_percentage_in_new_solution_l226_226061


namespace ratio_of_second_to_first_l226_226769

noncomputable def building_heights (H1 H2 H3 : ℝ) : Prop :=
  H1 = 600 ∧ H3 = 3 * (H1 + H2) ∧ H1 + H2 + H3 = 7200

theorem ratio_of_second_to_first (H1 H2 H3 : ℝ) (h : building_heights H1 H2 H3) :
  H1 ≠ 0 → (H2 / H1 = 2) :=
by
  unfold building_heights at h
  rcases h with ⟨h1, h3, h_total⟩
  sorry -- Steps of solving are skipped

end ratio_of_second_to_first_l226_226769


namespace scientific_notation_l226_226642

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l226_226642


namespace Rajesh_Spend_Salary_on_Food_l226_226724

theorem Rajesh_Spend_Salary_on_Food
    (monthly_salary : ℝ)
    (percentage_medicines : ℝ)
    (savings_percentage : ℝ)
    (savings : ℝ) :
    monthly_salary = 15000 ∧
    percentage_medicines = 0.20 ∧
    savings_percentage = 0.60 ∧
    savings = 4320 →
    (32 : ℝ) = ((monthly_salary * percentage_medicines + monthly_salary * (1 - (percentage_medicines + savings_percentage))) / monthly_salary) * 100 :=
by
  sorry

end Rajesh_Spend_Salary_on_Food_l226_226724


namespace smallest_n_mod_l226_226609

theorem smallest_n_mod :
  ∃ n : ℕ, (23 * n ≡ 5678 [MOD 11]) ∧ (∀ m : ℕ, (23 * m ≡ 5678 [MOD 11]) → (0 < n) ∧ (n ≤ m)) :=
  by
  sorry

end smallest_n_mod_l226_226609


namespace smallest_N_l226_226072

theorem smallest_N (l m n N : ℕ) (hl : l > 1) (hm : m > 1) (hn : n > 1) :
  (l - 1) * (m - 1) * (n - 1) = 231 → l * m * n = N → N = 384 :=
sorry

end smallest_N_l226_226072


namespace mean_absolute_temperature_correct_l226_226449

noncomputable def mean_absolute_temperature (temps : List ℝ) : ℝ :=
  (temps.map (λ x => |x|)).sum / temps.length

theorem mean_absolute_temperature_correct :
  mean_absolute_temperature [-6, -3, -3, -6, 0, 4, 3] = 25 / 7 :=
by
  sorry

end mean_absolute_temperature_correct_l226_226449


namespace groups_div_rem_l226_226630

noncomputable def numOfGroups (t b : ℕ) : ℕ :=
  Nat.choose 6 t * Nat.choose 8 b

def isValidGroup (t b : ℕ) : Prop :=
  (t - b) % 4 = 0 ∧ (t + b) > 0

def countValidGroups : ℕ :=
  Finset.sum (Finset.filter (λ pair, isValidGroup pair.1 pair.2) (Finset.product (Finset.range 7) (Finset.range 9))) (λ pair, numOfGroups pair.1 pair.2)

theorem groups_div_rem : countValidGroups % 100 = 95 := sorry

end groups_div_rem_l226_226630


namespace min_value_of_x_sq_plus_6x_l226_226610

theorem min_value_of_x_sq_plus_6x : ∃ x : ℝ, ∀ y : ℝ, y^2 + 6*y ≥ -9 :=
by
  sorry

end min_value_of_x_sq_plus_6x_l226_226610


namespace books_ratio_3_to_1_l226_226727

-- Definitions based on the conditions
def initial_books : ℕ := 220
def books_rebecca_received : ℕ := 40
def remaining_books : ℕ := 60
def total_books_given_away := initial_books - remaining_books
def books_mara_received := total_books_given_away - books_rebecca_received

-- The proof that the ratio of the number of books Mara received to the number of books Rebecca received is 3:1
theorem books_ratio_3_to_1 : (books_mara_received : ℚ) / books_rebecca_received = 3 := by
  sorry

end books_ratio_3_to_1_l226_226727


namespace external_tangent_twice_internal_tangent_l226_226034

noncomputable def distance_between_centers (r R : ℝ) : ℝ :=
  Real.sqrt (R^2 + r^2 + (10/3) * R * r)

theorem external_tangent_twice_internal_tangent 
  (r R O₁O₂ AB CD : ℝ)
  (h₁ : AB = 2 * CD)
  (h₂ : AB^2 = O₁O₂^2 - (R - r)^2)
  (h₃ : CD^2 = O₁O₂^2 - (R + r)^2) :
  O₁O₂ = distance_between_centers r R :=
by
  sorry

end external_tangent_twice_internal_tangent_l226_226034


namespace parabola_intersection_prob_l226_226181

noncomputable def prob_intersect_parabolas : ℚ :=
  57 / 64

theorem parabola_intersection_prob :
  ∀ (a b c d : ℤ), (1 ≤ a ∧ a ≤ 8) → (1 ≤ b ∧ b ≤ 8) →
  (1 ≤ c∧ c ≤ 8) → (1 ≤ d ∧ d ≤ 8) →
  prob_intersect_parabolas = 57 / 64 :=
by
  intros a b c d ha hb hc hd
  sorry

end parabola_intersection_prob_l226_226181


namespace degree_of_angle_C_l226_226836

theorem degree_of_angle_C 
  (A B C : ℝ) 
  (h1 : A = 4 * x) 
  (h2 : B = 4 * x) 
  (h3 : C = 7 * x) 
  (h_sum : A + B + C = 180) : 
  C = 84 := 
by 
  sorry

end degree_of_angle_C_l226_226836


namespace magic_8_ball_probability_l226_226146

theorem magic_8_ball_probability :
  let num_questions := 7
  let num_positive := 3
  let positive_probability := 3 / 7
  let negative_probability := 4 / 7
  let binomial_coefficient := Nat.choose num_questions num_positive
  let total_probability := binomial_coefficient * (positive_probability ^ num_positive) * (negative_probability ^ (num_questions - num_positive))
  total_probability = 242112 / 823543 :=
by
  sorry

end magic_8_ball_probability_l226_226146


namespace probability_odd_product_l226_226264

-- Given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the proof problem
theorem probability_odd_product (h: choices = 15 ∧ odd_choices = 3) :
  (odd_choices : ℚ) / choices = 1 / 5 :=
by sorry

end probability_odd_product_l226_226264


namespace math_problem_l226_226099

theorem math_problem (a b c d e : ℤ) (x : ℤ) (hx : x > 196)
  (h1 : a + b = 183) (h2 : a + c = 186) (h3 : d + e = x) (h4 : c + e = 196)
  (h5 : 183 < 186) (h6 : 186 < 187) (h7 : 187 < 190) (h8 : 190 < 191) (h9 : 191 < 192)
  (h10 : 192 < 193) (h11 : 193 < 194) (h12 : 194 < 196) (h13 : 196 < x) :
  (a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200) ∧ (∃ y, y = 10 * x + 3 ∧ y = 2003) :=
by
  sorry

end math_problem_l226_226099


namespace dress_price_l226_226462

namespace VanessaClothes

def priceOfDress (total_revenue : ℕ) (num_dresses num_shirts price_of_shirt : ℕ) : ℕ :=
  (total_revenue - num_shirts * price_of_shirt) / num_dresses

theorem dress_price :
  priceOfDress 69 7 4 5 = 7 :=
by 
  calc
    priceOfDress 69 7 4 5 = (69 - 4 * 5) / 7 : rfl
                     ... = 49 / 7 : by norm_num
                     ... = 7 : by norm_num

end VanessaClothes

end dress_price_l226_226462


namespace num_zeros_in_product_l226_226004

theorem num_zeros_in_product : ∀ (a b : ℕ), (a = 125) → (b = 960) → (∃ n, a * b = n * 10^4) :=
by
  sorry

end num_zeros_in_product_l226_226004


namespace exists_X_Y_l226_226277

theorem exists_X_Y {A n : ℤ} (h_coprime : Int.gcd A n = 1) :
  ∃ X Y : ℤ, |X| < Int.sqrt n ∧ |Y| < Int.sqrt n ∧ n ∣ (A * X - Y) :=
sorry

end exists_X_Y_l226_226277


namespace max_min_x2_min_xy_plus_y2_l226_226297

theorem max_min_x2_min_xy_plus_y2 (x y : ℝ) (h : x^2 + x * y + y^2 = 3) :
  1 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 9 :=
by sorry

end max_min_x2_min_xy_plus_y2_l226_226297


namespace proportional_stratified_sampling_probability_at_least_one_grade12_probability_classes_ab_l226_226136

def school_classes := {grade10 := 16, grade11 := 12, grade12 := 8}

-- Proving proportional stratified sampling
def stratified_sampling (total_classes: Fin 3 -> Nat) (selected: Fin 3 -> Nat) : Prop :=
  selected 0 = (16 * 9 / (16 + 12 + 8)) ∧
  selected 1 = (12 * 9 / (16 + 12 + 8)) ∧
  selected 2 = (8 * 9 / (16 + 12 + 8))

def selected_classes : Fin 3 -> Nat := fun i =>
  match i with
  | ⟨0, _⟩ => 4
  | ⟨1, _⟩ => 3
  | ⟨2, _⟩ => 2

theorem proportional_stratified_sampling :
  stratified_sampling (λ ⟨i, _⟩ => match i with
                              | 0 => school_classes.grade10
                              | 1 => school_classes.grade11
                              | 2 => school_classes.grade12
                      end) selected_classes := sorry

-- Proving probability that at least one of the 2 selected classes is from grade 12
def at_least_one_grade12 : Prop :=
  let grade11 := 3
  let grade12 := 2
  ∃ combs: Π n, Fin₃ (grade11 + grade12), true ∧
  (grade11 + grade12 = 5) ∧
  ∀ selections, ∃ count12, selections.filter (λ x, x < grade12) = count12 ∧
  (count12.toNat ≥ 1) = (7/10)

theorem probability_at_least_one_grade12 :
  at_least_one_grade12 := sorry

-- Proving probability that both class A from grade 11 and class B from grade 12 are selected
def classes_selection(A: Nat) (B: Nat) : Prop :=
  let total_combinations := 4 * 3 * 2
  ∃ prob: (4/total_combinations), true ∧
  prob = (1/6)

theorem probability_classes_ab :
  classes_selection 11 12 := sorry

end proportional_stratified_sampling_probability_at_least_one_grade12_probability_classes_ab_l226_226136


namespace arithmetic_expression_equality_l226_226516

theorem arithmetic_expression_equality : 18 * 36 - 27 * 18 = 162 := by
  sorry

end arithmetic_expression_equality_l226_226516


namespace bicycle_distance_l226_226768

theorem bicycle_distance (b t : ℝ) (h : t ≠ 0) :
  let rate := (b / 2) / t / 3
  let total_seconds := 5 * 60
  rate * total_seconds = 50 * b / t := by
    sorry

end bicycle_distance_l226_226768


namespace max_value_of_8q_minus_9p_is_zero_l226_226286

theorem max_value_of_8q_minus_9p_is_zero (p : ℝ) (q : ℝ) (h1 : 0 < p) (h2 : p < 1) (hq : q = 3 * p ^ 2 - 2 * p ^ 3) : 
  8 * q - 9 * p ≤ 0 :=
by
  sorry

end max_value_of_8q_minus_9p_is_zero_l226_226286


namespace impossible_coins_l226_226869

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l226_226869


namespace shekar_biology_marks_l226_226445

theorem shekar_biology_marks (M S SS E A n B : ℕ) 
  (hM : M = 76)
  (hS : S = 65)
  (hSS : SS = 82)
  (hE : E = 67)
  (hA : A = 73)
  (hn : n = 5)
  (hA_eq : A = (M + S + SS + E + B) / n) : 
  B = 75 := 
by
  rw [hM, hS, hSS, hE, hn, hA] at hA_eq
  sorry

end shekar_biology_marks_l226_226445


namespace min_value_four_l226_226977

noncomputable def min_value_T (a b c : ℝ) : ℝ :=
  1 / (2 * (a * b - 1)) + a * (b + 2 * c) / (a * b - 1)

theorem min_value_four (a b c : ℝ) (h1 : (1 / a) > 0)
  (h2 : b^2 - (4 * c) / a ≤ 0) (h3 : a * b > 1) : 
  min_value_T a b c = 4 := 
by 
  sorry

end min_value_four_l226_226977


namespace dist_between_centers_l226_226396

noncomputable def dist_centers_tangent_circles : ℝ :=
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  Real.sqrt 2 * (a₁ - a₂)

theorem dist_between_centers :
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  let C₁ := (a₁, a₁)
  let C₂ := (a₂, a₂)
  dist_centers_tangent_circles = 8 :=
by
  sorry

end dist_between_centers_l226_226396


namespace new_volume_is_80_gallons_l226_226633

-- Define the original volume
def V_original : ℝ := 5

-- Define the factors by which length, width, and height are increased
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 4

-- Define the new volume
def V_new : ℝ := V_original * (length_factor * width_factor * height_factor)

-- Theorem to prove the new volume is 80 gallons
theorem new_volume_is_80_gallons : V_new = 80 := 
by
  -- Proof goes here
  sorry

end new_volume_is_80_gallons_l226_226633


namespace largest_integral_x_l226_226804

theorem largest_integral_x (x : ℤ) (h1 : 1/4 < (x:ℝ)/6) (h2 : (x:ℝ)/6 < 7/9) : x ≤ 4 :=
by
  -- This is where the proof would go
  sorry

end largest_integral_x_l226_226804


namespace basketball_team_win_rate_l226_226215

theorem basketball_team_win_rate (won_first : ℕ) (total : ℕ) (remaining : ℕ)
    (desired_rate : ℚ) (x : ℕ) (H_won : won_first = 30) (H_total : total = 100)
    (H_remaining : remaining = 55) (H_desired : desired_rate = 13/20) :
    (30 + x) / 100 = 13 / 20 ↔ x = 35 := by
    sorry

end basketball_team_win_rate_l226_226215


namespace map_length_25_cm_represents_125_km_l226_226305

-- Define the conditions
def map_scale (cm: ℝ) : ℝ := 5 * cm

-- Define the main statement to be proved
theorem map_length_25_cm_represents_125_km : map_scale 25 = 125 := by
  sorry

end map_length_25_cm_represents_125_km_l226_226305


namespace smallest_N_for_equal_adults_and_children_l226_226637

theorem smallest_N_for_equal_adults_and_children :
  ∃ (N : ℕ), N > 0 ∧ (∀ a b : ℕ, 8 * N = a ∧ 12 * N = b ∧ a = b) ∧ N = 3 :=
sorry

end smallest_N_for_equal_adults_and_children_l226_226637


namespace combination_15_choose_3_l226_226547

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l226_226547


namespace lcm_18_30_l226_226479

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l226_226479


namespace total_cost_of_pencils_and_erasers_l226_226276

theorem total_cost_of_pencils_and_erasers 
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (pencils_bought : ℕ)
  (erasers_bought : ℕ)
  (total_cost_dollars : ℝ)
  (cents_to_dollars : ℝ)
  (hc : pencil_cost = 2)
  (he : eraser_cost = 5)
  (hp : pencils_bought = 500)
  (he2 : erasers_bought = 250)
  (cents_to_dollars_def : cents_to_dollars = 100)
  (total_cost_calc : total_cost_dollars = 
    ((pencils_bought * pencil_cost + erasers_bought * eraser_cost : ℕ) : ℝ) / cents_to_dollars) 
  : total_cost_dollars = 22.50 :=
sorry

end total_cost_of_pencils_and_erasers_l226_226276


namespace james_shirts_l226_226994

theorem james_shirts (S P : ℕ) (h1 : P = S / 2) (h2 : 6 * S + 8 * P = 100) : S = 10 :=
sorry

end james_shirts_l226_226994


namespace prime_sum_square_mod_3_l226_226105

theorem prime_sum_square_mod_3 (p : Fin 100 → ℕ) (h_prime : ∀ i, Nat.Prime (p i)) (h_distinct : Function.Injective p) :
  let N := (Finset.univ : Finset (Fin 100)).sum (λ i => (p i)^2)
  N % 3 = 1 := by
  sorry

end prime_sum_square_mod_3_l226_226105


namespace find_n_lcm_l226_226895

theorem find_n_lcm (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : n ≥ 100) (h3 : n < 1000) (h4 : ¬ (3 ∣ n)) (h5 : ¬ (2 ∣ m)) : n = 230 :=
sorry

end find_n_lcm_l226_226895


namespace michael_and_truck_meet_l226_226304

/--
Assume:
1. Michael walks at 6 feet per second.
2. Trash pails are every 240 feet.
3. A truck travels at 10 feet per second and stops for 36 seconds at each pail.
4. Initially, when Michael passes a pail, the truck is 240 feet ahead.

Prove:
Michael and the truck meet every 120 seconds starting from 120 seconds.
-/
theorem michael_and_truck_meet (t : ℕ) : t ≥ 120 → (t - 120) % 120 = 0 :=
sorry

end michael_and_truck_meet_l226_226304


namespace marvin_number_is_correct_l226_226437

theorem marvin_number_is_correct (y : ℤ) (h : y - 5 = 95) : y + 5 = 105 := by
  sorry

end marvin_number_is_correct_l226_226437


namespace power_function_value_at_half_l226_226119

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_value_at_half (a : ℝ) (α : ℝ) 
  (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1 / 4) (h4 : g α 2 = 1 / 4) : 
  g α (1/2) = 4 := 
by
  sorry

end power_function_value_at_half_l226_226119


namespace silk_pieces_count_l226_226020

theorem silk_pieces_count (S C : ℕ) (h1 : S = 2 * C) (h2 : S + C + 2 = 13) : S = 7 :=
by
  sorry

end silk_pieces_count_l226_226020


namespace inequality_solution_l226_226879

theorem inequality_solution 
  (x : ℝ) 
  (h : 2*x^4 + x^2 - 4*x - 3*x^2 * |x - 2| + 4 ≥ 0) : 
  x ∈ Set.Iic (-2) ∪ Set.Icc ((-1 - Real.sqrt 17) / 4) ((-1 + Real.sqrt 17) / 4) ∪ Set.Ici 1 :=
sorry

end inequality_solution_l226_226879


namespace sum_of_solutions_l226_226185

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l226_226185


namespace polynomial_p_at_0_l226_226715

theorem polynomial_p_at_0 {p : ℕ → ℝ} (h1 : ∀ x, polynomial.degree (p x) = 6)
  (h2 : ∀ n, n ∈ {0, 1, 2, 3, 4, 5, 6} → p (3 ^ n) = 1 / (3 ^ n)) :
  p 0 = 2186 / 2187 :=
sorry

end polynomial_p_at_0_l226_226715


namespace value_of_t_l226_226252

noncomputable def f (x t k : ℝ) : ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem value_of_t (a b t k : ℝ) (h1 : 0 < t) (h2 : 0 < k) 
  (h3 : a + b = t) (h4 : a * b = k) (h5 : 2 * a = b - 2) (h6 : (-2)^2 = a * b) : 
  t = 5 := 
  sorry

end value_of_t_l226_226252


namespace sum_of_variables_is_233_l226_226874

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end sum_of_variables_is_233_l226_226874


namespace blue_line_length_l226_226521

theorem blue_line_length (w b : ℝ) (h1 : w = 7.666666666666667) (h2 : w = b + 4.333333333333333) :
  b = 3.333333333333334 :=
by sorry

end blue_line_length_l226_226521


namespace smallest_positive_real_number_l226_226794

noncomputable def smallest_x : ℝ := 71 / 8

theorem smallest_positive_real_number (x : ℝ) (h₁ : ∀ y : ℝ, 0 < y ∧ (⌊y^2⌋ - y * ⌊y⌋ = 7) → x ≤ y) (h₂ : 0 < x) (h₃ : ⌊x^2⌋ - x * ⌊x⌋ = 7) : x = smallest_x :=
sorry

end smallest_positive_real_number_l226_226794


namespace slope_of_line_l226_226456

theorem slope_of_line : ∀ (x y : ℝ), (x - y + 1 = 0) → (1 = 1) :=
by
  intros x y h
  sorry

end slope_of_line_l226_226456


namespace part1_part2_part3_l226_226949

-- Part 1
theorem part1 : (1 > -1) ∧ (1 < 2) ∧ (-(1/2) > -1) ∧ (-(1/2) < 2) := 
  by sorry

-- Part 2
theorem part2 (k : Real) : (3 < k) ∧ (k ≤ 4) := 
  by sorry

-- Part 3
theorem part3 (m : Real) : (2 < m) ∧ (m ≤ 3) := 
  by sorry

end part1_part2_part3_l226_226949


namespace lcm_of_18_and_30_l226_226470

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l226_226470


namespace cylinder_lateral_surface_area_l226_226632

-- Define structures for the problem
structure Cylinder where
  generatrix : ℝ
  base_radius : ℝ

-- Define the conditions
def cylinder_conditions : Cylinder :=
  { generatrix := 1, base_radius := 1 }

-- The theorem statement
theorem cylinder_lateral_surface_area (cyl : Cylinder) (h_gen : cyl.generatrix = 1) (h_rad : cyl.base_radius = 1) :
  ∀ (area : ℝ), area = 2 * Real.pi :=
sorry

end cylinder_lateral_surface_area_l226_226632


namespace lcm_18_30_eq_90_l226_226465

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l226_226465


namespace mod_congruence_zero_iff_l226_226131

theorem mod_congruence_zero_iff
  (a b c d n : ℕ)
  (h1 : a * c ≡ 0 [MOD n])
  (h2 : b * c + a * d ≡ 0 [MOD n]) :
  b * c ≡ 0 [MOD n] ∧ a * d ≡ 0 [MOD n] :=
by
  sorry

end mod_congruence_zero_iff_l226_226131


namespace value_of_m_minus_n_l226_226129

theorem value_of_m_minus_n (m n : ℝ) (h : (-3)^2 + m * (-3) + 3 * n = 0) : m - n = 3 :=
sorry

end value_of_m_minus_n_l226_226129


namespace john_monthly_income_l226_226022

theorem john_monthly_income (I : ℝ) (h : I - 0.05 * I = 1900) : I = 2000 :=
by
  sorry

end john_monthly_income_l226_226022


namespace train_speed_l226_226767

theorem train_speed (length : ℕ) (cross_time : ℕ) (speed : ℝ)
    (h1 : length = 250)
    (h2 : cross_time = 3)
    (h3 : speed = (length / cross_time : ℝ) * 3.6) :
    speed = 300 := 
sorry

end train_speed_l226_226767


namespace sqrt_sum_equality_l226_226922

theorem sqrt_sum_equality :
  (Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6) :=
by
  sorry

end sqrt_sum_equality_l226_226922


namespace even_and_multiple_of_3_l226_226169

theorem even_and_multiple_of_3 (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) (h2 : ∃ n : ℤ, b = 6 * n) :
  (∃ m : ℤ, a + b = 2 * m) ∧ (∃ p : ℤ, a + b = 3 * p) :=
by
  sorry

end even_and_multiple_of_3_l226_226169


namespace distribute_pencils_l226_226093

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end distribute_pencils_l226_226093


namespace find_y_l226_226012

theorem find_y 
  (x y : ℝ) 
  (h1 : (6 : ℝ) = (1/2 : ℝ) * x) 
  (h2 : y = (1/2 : ℝ) * 10) 
  (h3 : x * y = 60) 
: y = 5 := 
by 
  sorry

end find_y_l226_226012


namespace range_of_m_l226_226534

theorem range_of_m (x m : ℝ) (h₁ : x^2 - 3 * x + 2 > 0) (h₂ : ¬(x^2 - 3 * x + 2 > 0) → x < m) : 2 < m :=
by
  sorry

end range_of_m_l226_226534


namespace second_day_more_than_third_day_l226_226777

-- Define the conditions
def total_people (d1 d2 d3 : ℕ) := d1 + d2 + d3 = 246 
def first_day := 79
def third_day := 120

-- Define the statement to prove
theorem second_day_more_than_third_day : 
  ∃ d2 : ℕ, total_people first_day d2 third_day ∧ (d2 - third_day) = 47 :=
by
  sorry

end second_day_more_than_third_day_l226_226777


namespace squirrels_and_nuts_l226_226049

theorem squirrels_and_nuts (number_of_squirrels number_of_nuts : ℕ) 
    (h1 : number_of_squirrels = 4) 
    (h2 : number_of_squirrels = number_of_nuts + 2) : 
    number_of_nuts = 2 :=
by
  sorry

end squirrels_and_nuts_l226_226049


namespace find_number_l226_226728

-- Define the variables and the conditions as theorems to be proven in Lean.
theorem find_number (x : ℤ) 
  (h1 : (x - 16) % 37 = 0)
  (h2 : (x - 16) / 37 = 23) :
  x = 867 :=
sorry

end find_number_l226_226728


namespace find_ab_unique_l226_226268

theorem find_ab_unique (a b : ℕ) (h1 : a > 1) (h2 : b > a) (h3 : a ≤ 20) (h4 : b ≤ 20) (h5 : a * b = 52) (h6 : a + b = 17) : a = 4 ∧ b = 13 :=
by {
  -- Proof goes here
  sorry
}

end find_ab_unique_l226_226268


namespace find_g_at_6_l226_226425

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 20 * x ^ 3 + 37 * x ^ 2 - 18 * x - 80

theorem find_g_at_6 : g 6 = 712 := by
  -- We apply the remainder theorem to determine the value of g(6).
  sorry

end find_g_at_6_l226_226425


namespace books_combination_l226_226552

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l226_226552


namespace mittens_per_box_l226_226518

theorem mittens_per_box (total_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) 
  (h_total_boxes : total_boxes = 4) 
  (h_scarves_per_box : scarves_per_box = 2) 
  (h_total_clothing : total_clothing = 32) : 
  (total_clothing - total_boxes * scarves_per_box) / total_boxes = 6 := 
by
  -- Sorry, proof is omitted
  sorry

end mittens_per_box_l226_226518


namespace average_speed_approx_l226_226066

noncomputable def average_speed : ℝ :=
  let distance1 := 7
  let speed1 := 10
  let distance2 := 10
  let speed2 := 7
  let distance3 := 5
  let speed3 := 12
  let distance4 := 8
  let speed4 := 6
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  total_distance / total_time

theorem average_speed_approx : abs (average_speed - 7.73) < 0.01 := by
  -- The necessary definitions fulfill the conditions and hence we put sorry here
  sorry

end average_speed_approx_l226_226066


namespace rectangle_area_error_percent_l226_226410

theorem rectangle_area_error_percent (L W : ℝ) :
  let measured_length := 1.05 * L,
      measured_width := 0.96 * W,
      actual_area := L * W,
      measured_area := measured_length * measured_width,
      error := measured_area - actual_area in
  (error / actual_area) * 100 = 0.8 := 
by
  sorry

end rectangle_area_error_percent_l226_226410


namespace permutations_with_property_P_greater_l226_226426

def has_property_P (n : ℕ) (p : (Fin (2 * n) → Fin (2 * n))) : Prop :=
∃ i : Fin (2 * n - 1), |(p i).val - (p ⟨i.val + 1, Nat.lt_of_succ_lt_succ i.property⟩).val| = n

theorem permutations_with_property_P_greater {n : ℕ} (hn : 0 < n) :
  let perms := equiv.perm (Fin (2 * n))
  ∃ (p₁ p₂ : perms), has_property_P n p₁ ∧ ¬ has_property_P n p₂ ∧
    (∑ p in perms, (if has_property_P n p then 1 else 0)) >
    (∑ p in perms, (if ¬ has_property_P n p then 1 else 0)) := 
sorry

end permutations_with_property_P_greater_l226_226426


namespace traveler_distance_l226_226346

theorem traveler_distance (a b c d : ℕ) (h1 : a = 24) (h2 : b = 15) (h3 : c = 10) (h4 : d = 9) :
  let net_ns := a - c
  let net_ew := b - d
  let distance := Real.sqrt ((net_ns ^ 2) + (net_ew ^ 2))
  distance = 2 * Real.sqrt 58 := 
by
  sorry

end traveler_distance_l226_226346


namespace andrew_bought_mangoes_l226_226943

theorem andrew_bought_mangoes (m : ℕ) 
    (grapes_cost : 6 * 74 = 444) 
    (mangoes_cost : m * 59 = total_mangoes_cost) 
    (total_cost_eq_975 : 444 + total_mangoes_cost = 975) 
    (total_cost := 444 + total_mangoes_cost) 
    (total_mangoes_cost := 59 * m) 
    : m = 9 := 
sorry

end andrew_bought_mangoes_l226_226943


namespace no_such_coins_l226_226866

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end no_such_coins_l226_226866


namespace total_ages_l226_226419

def Kate_age : ℕ := 19
def Maggie_age : ℕ := 17
def Sue_age : ℕ := 12

theorem total_ages : Kate_age + Maggie_age + Sue_age = 48 := sorry

end total_ages_l226_226419


namespace range_of_a_for_solution_set_l226_226524

theorem range_of_a_for_solution_set (a : ℝ) :
  ((∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1)) :=
sorry

end range_of_a_for_solution_set_l226_226524


namespace jerome_contact_list_count_l226_226845

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end jerome_contact_list_count_l226_226845


namespace number_of_round_table_arrangements_l226_226988

theorem number_of_round_table_arrangements : (Nat.factorial 5) / 5 = 24 := 
by
  sorry

end number_of_round_table_arrangements_l226_226988


namespace solution_x_alcohol_percentage_l226_226639

theorem solution_x_alcohol_percentage (P : ℝ) :
  let y_percentage := 0.30
  let mixture_percentage := 0.25
  let y_volume := 600
  let x_volume := 200
  let mixture_volume := y_volume + x_volume
  let y_alcohol_content := y_volume * y_percentage
  let mixture_alcohol_content := mixture_volume * mixture_percentage
  P * x_volume + y_alcohol_content = mixture_alcohol_content →
  P = 0.10 :=
by
  intros
  sorry

end solution_x_alcohol_percentage_l226_226639


namespace antonieta_tickets_needed_l226_226077

-- Definitions based on conditions:
def ferris_wheel_tickets : ℕ := 6
def roller_coaster_tickets : ℕ := 5
def log_ride_tickets : ℕ := 7
def antonieta_initial_tickets : ℕ := 2

-- Theorem to prove the required number of tickets Antonieta should buy
theorem antonieta_tickets_needed : ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - antonieta_initial_tickets = 16 :=
by
  sorry

end antonieta_tickets_needed_l226_226077


namespace length_OF_does_not_depend_on_C_l226_226723

open Euclidean_geometry

variables {A B C M P Q O D E F : Point ℝ}
variables (segment_AB: Segment (line A B))
variables (point_M_on_segment : ∃ (λ M, M ∈ segment_AB))
variables (point_P_is_midpoint: P = midpoint A M)
variables (point_Q_is_midpoint: Q = midpoint B M)
variables (point_O_is_midpoint: O = midpoint P Q)
variables (angle_ACB_right: ∠ A C B = 90°)
variables (MD_perpendicular_to_CA: perpendicular M D (line C A))
variables (ME_perpendicular_to_CB: perpendicular M E (line C B))
variables (point_F_is_midpoint: F = midpoint D E)

theorem length_OF_does_not_depend_on_C
  (fixed_PQ : (λ P, Q, dist P Q) : ℝ) :
  dist O F = dist P Q / 2 :=
begin
  sorry
end

end length_OF_does_not_depend_on_C_l226_226723


namespace john_income_l226_226569

theorem john_income 
  (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) (ingrid_income : ℝ) (combined_tax_rate : ℝ)
  (jt_30 : john_tax_rate = 0.30) (it_40 : ingrid_tax_rate = 0.40) (ii_72000 : ingrid_income = 72000) 
  (ctr_35625 : combined_tax_rate = 0.35625) :
  ∃ J : ℝ, (0.30 * J + ingrid_tax_rate * ingrid_income = combined_tax_rate * (J + ingrid_income)) ∧ (J = 56000) :=
by
  sorry

end john_income_l226_226569


namespace monotonic_range_of_t_l226_226390

noncomputable def f (x : ℝ) := (x^2 - 3 * x + 3) * Real.exp x

def is_monotonic_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

theorem monotonic_range_of_t (t : ℝ) (ht : t > -2) :
  is_monotonic_on_interval (-2) t f ↔ (-2 < t ∧ t ≤ 0) :=
sorry

end monotonic_range_of_t_l226_226390


namespace jane_wins_game_l226_226413

noncomputable def jane_win_probability : ℚ :=
  1/3 / (1 - (2/3 * 1/3 * 2/3))

theorem jane_wins_game :
  jane_win_probability = 9/23 :=
by
  -- detailed proof steps would be filled in here
  sorry

end jane_wins_game_l226_226413


namespace square_perimeter_lemma_l226_226899

theorem square_perimeter_lemma
  (t1 t2 t3 t4 k1 k2 k3 k4 : ℝ)
  (h1 : t1 + t2 + t3 = t4)
  (h2 : k1 = 4 * real.sqrt t1)
  (h3 : k2 = 4 * real.sqrt t2)
  (h4 : k3 = 4 * real.sqrt t3)
  (h5 : k4 = 4 * real.sqrt t4) :
  k1 + k2 + k3 ≤ k4 * real.sqrt 3 :=
by { sorry }

end square_perimeter_lemma_l226_226899


namespace mac_total_loss_is_correct_l226_226578

def day_1_value : ℝ := 6 * 0.075 + 2 * 0.0075
def day_2_value : ℝ := 10 * 0.0045 + 5 * 0.0036
def day_3_value : ℝ := 4 * 0.10 + 1 * 0.011
def day_4_value : ℝ := 7 * 0.013 + 5 * 0.038
def day_5_value : ℝ := 3 * 0.5 + 2 * 0.0019
def day_6_value : ℝ := 12 * 0.0072 + 3 * 0.0013
def day_7_value : ℝ := 8 * 0.045 + 6 * 0.0089

def total_value : ℝ := day_1_value + day_2_value + day_3_value + day_4_value + day_5_value + day_6_value + day_7_value

def daily_loss (total_value: ℝ): ℝ := total_value - 0.25

def total_loss : ℝ := daily_loss day_1_value + daily_loss day_2_value + daily_loss day_3_value + daily_loss day_4_value + daily_loss day_5_value + daily_loss day_6_value + daily_loss day_7_value

theorem mac_total_loss_is_correct : total_loss = 2.1619 := 
by 
  simp [day_1_value, day_2_value, day_3_value, day_4_value, day_5_value, day_6_value, day_7_value, daily_loss, total_loss]
  sorry

end mac_total_loss_is_correct_l226_226578


namespace value_of_p_l226_226740

theorem value_of_p (p q : ℚ) (h₁ : p > 0) (h₂ : q > 0) (h₃ : p + q = 1)
  (h_eq : 8 * p^7 * q = 28 * p^6 * q^2) : p = 7 / 9 :=
by sorry

end value_of_p_l226_226740


namespace evlyn_can_buy_grapes_l226_226825

theorem evlyn_can_buy_grapes 
  (price_pears price_oranges price_lemons price_grapes : ℕ)
  (h1 : 10 * price_pears = 5 * price_oranges)
  (h2 : 4 * price_oranges = 6 * price_lemons)
  (h3 : 3 * price_lemons = 2 * price_grapes) :
  (20 * price_pears = 10 * price_grapes) :=
by
  -- The proof is omitted using sorry
  sorry

end evlyn_can_buy_grapes_l226_226825


namespace vitamin_D_scientific_notation_l226_226648

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l226_226648


namespace james_distance_l226_226708

-- Definitions and conditions
def speed : ℝ := 80.0
def time : ℝ := 16.0

-- Proof problem statement
theorem james_distance : speed * time = 1280.0 := by
  sorry

end james_distance_l226_226708


namespace eighth_grade_probability_female_win_l226_226289

theorem eighth_grade_probability_female_win:
  let P_Alexandra : ℝ := 1 / 4,
      P_Alexander : ℝ := 3 / 4,
      P_Evgenia : ℝ := 1 / 4,
      P_Yevgeny : ℝ := 3 / 4,
      P_Valentina : ℝ := 2 / 5,
      P_Valentin : ℝ := 3 / 5,
      P_Vasilisa : ℝ := 1 / 50,
      P_Vasily : ℝ := 49 / 50 in
  let P_female : ℝ :=
    1 / 4 * (P_Alexandra + 
             P_Evgenia + 
             P_Valentina + 
             P_Vasilisa) in
  P_female = 1 / 16 + 1 / 48 + 3 / 20 + 1 / 200 :=
sorry

end eighth_grade_probability_female_win_l226_226289


namespace Eva_is_16_l226_226238

def Clara_age : ℕ := 12
def Nora_age : ℕ := Clara_age + 3
def Liam_age : ℕ := Nora_age - 4
def Eva_age : ℕ := Liam_age + 5

theorem Eva_is_16 : Eva_age = 16 := by
  sorry

end Eva_is_16_l226_226238


namespace production_volume_bounds_l226_226440

theorem production_volume_bounds:
  ∀ (x : ℕ),
  (10 * x ≤ 800 * 2400) ∧ 
  (10 * x ≤ 4000000 + 16000000) ∧
  (x ≥ 1800000) →
  (1800000 ≤ x ∧ x ≤ 1920000) :=
by
  sorry

end production_volume_bounds_l226_226440


namespace ratio_shortest_to_middle_tree_l226_226319

theorem ratio_shortest_to_middle_tree (height_tallest : ℕ) 
  (height_middle : ℕ) (height_shortest : ℕ)
  (h1 : height_tallest = 150) 
  (h2 : height_middle = (2 * height_tallest) / 3) 
  (h3 : height_shortest = 50) : 
  height_shortest / height_middle = 1 / 2 := by sorry

end ratio_shortest_to_middle_tree_l226_226319


namespace sequence_satisfies_conditions_l226_226691

theorem sequence_satisfies_conditions : 
  let seq1 := [4, 1, 3, 1, 2, 4, 3, 2]
  let seq2 := [2, 3, 4, 2, 1, 3, 1, 4]
  (seq1[0] = 4 ∧ seq1[1] = 1 ∧ seq1[2] = 3 ∧ seq1[3] = 1 ∧ seq1[4] = 2 ∧ seq1[5] = 4 ∧ seq1[6] = 3 ∧ seq1[7] = 2)
  ∨ (seq2[0] = 2 ∧ seq2[1] = 3 ∧ seq2[2] = 4 ∧ seq2[3] = 2 ∧ seq2[4] = 1 ∧ seq2[5] = 3 ∧ seq2[6] = 1 ∧ seq2[7] = 4)
  ∧ (seq1[1] = 1 ∧ seq1[3] - seq1[1] = 2 ∧ seq1[4] - seq1[2] = 3 ∧ seq1[5] - seq1[2] = 4) := 
  sorry

end sequence_satisfies_conditions_l226_226691


namespace teacher_periods_per_day_l226_226783

noncomputable def periods_per_day (days_per_month : ℕ) (months : ℕ) (period_rate : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_days := days_per_month * months
  let total_periods := total_earnings / period_rate
  let periods_per_day := total_periods / total_days
  periods_per_day

theorem teacher_periods_per_day :
  periods_per_day 24 6 5 3600 = 5 := by
  sorry

end teacher_periods_per_day_l226_226783


namespace impossible_coins_l226_226863

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l226_226863


namespace a_b_product_l226_226811

theorem a_b_product (a b : ℝ) (h1 : 2 * a - b = 1) (h2 : 2 * b - a = 7) : (a + b) * (a - b) = -16 :=
by
  -- The proof would be provided here.
  sorry

end a_b_product_l226_226811


namespace range_of_a_l226_226132

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 > 0) ↔ (0 ≤ a ∧ a < 12) :=
by
  sorry

end range_of_a_l226_226132


namespace cos_210_eq_neg_sqrt3_div_2_l226_226043

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (Real.pi + Real.pi / 6) = -Real.sqrt 3 / 2 := sorry

end cos_210_eq_neg_sqrt3_div_2_l226_226043


namespace simulation_probability_of_two_bullseyes_in_three_shots_l226_226015

/-
Conditions:
1. Probability of hitting the bullseye = 0.4
2. Representation in simulation:
   - $0, 1, 2, 3$ represent hitting the bullseye
   - $4, 5, 6, 7, 8, 9$ represent missing the bullseye
3. Given groups of random numbers (simulated shots):

Groups are:
321, 421, 191, 925, 271, 932, 800, 478, 589, 663,
531, 297, 396, 021, 546, 388, 230, 113, 507, 965

Question:
Estimate the probability that Xiao Li hits the bullseye exactly twice in three shots, which results to 0.30.

Problem: Prove that the estimated probability matches the expected results based on the simulation given the conditions.
-/

theorem simulation_probability_of_two_bullseyes_in_three_shots :
  let bullseye := λx : Nat, x <= 3 in
  let is_bullseye_twice_in_three := λ (u v w : Nat), bullseye u + bullseye v + bullseye w = 2 in
  let groups := [[3, 2, 1], [4, 2, 1], [1, 9, 1], [9, 2, 5], [2, 7, 1], [9, 3, 2],
                 [8, 0, 0], [4, 7, 8], [5, 8, 9], [6, 6, 3], [5, 3, 1], [2, 9, 7],
                 [3, 9, 6], [0, 2, 1], [5, 4, 6], [3, 8, 8], [2, 3, 0], [1, 1, 3],
                 [5, 0, 7], [9, 6, 5]] in
  (countp (λ g, is_bullseye_twice_in_three g.head g.get? 1 g.get? 2) groups.to_list : ℚ) / 20 = 0.3 :=
by
  sorry

end simulation_probability_of_two_bullseyes_in_three_shots_l226_226015


namespace larger_solution_quadratic_l226_226959

theorem larger_solution_quadratic (x : ℝ) : x^2 - 13 * x + 42 = 0 → x = 7 ∨ x = 6 ∧ x > 6 :=
by
  sorry

end larger_solution_quadratic_l226_226959


namespace solve_some_number_l226_226539

theorem solve_some_number (n : ℝ) (h : (n * 10) / 100 = 0.032420000000000004) : n = 0.32420000000000004 :=
by
  -- The proof steps are omitted with 'sorry' here.
  sorry

end solve_some_number_l226_226539


namespace simplify_cube_root_l226_226583

theorem simplify_cube_root (a : ℝ) (h : 0 ≤ a) : (a * a^(1/2))^(1/3) = a^(1/2) :=
sorry

end simplify_cube_root_l226_226583


namespace lean_proof_l226_226710

noncomputable def proof_problem (a b c d : ℝ) (habcd : a * b * c * d = 1) : Prop :=
  (1 + a * b) / (1 + a) ^ 2008 +
  (1 + b * c) / (1 + b) ^ 2008 +
  (1 + c * d) / (1 + c) ^ 2008 +
  (1 + d * a) / (1 + d) ^ 2008 ≥ 4

theorem lean_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_abcd : a * b * c * d = 1) : proof_problem a b c d h_abcd :=
  sorry

end lean_proof_l226_226710


namespace seventh_term_value_l226_226451

theorem seventh_term_value (a d : ℤ) (h1 : a = 12) (h2 : a + 3 * d = 18) : a + 6 * d = 24 := 
by
  sorry

end seventh_term_value_l226_226451


namespace probability_of_meeting_l226_226752

noncomputable def meeting_probability : ℝ :=
  let total_area := 10 * 10
  let favorable_area := 51
  favorable_area / total_area

theorem probability_of_meeting : meeting_probability = 51 / 100 :=
by
  sorry

end probability_of_meeting_l226_226752


namespace range_of_independent_variable_l226_226315

theorem range_of_independent_variable (x : ℝ) (h : ∃ y, y = 2 / (Real.sqrt (x - 3))) : x > 3 :=
sorry

end range_of_independent_variable_l226_226315


namespace player1_winning_strategy_l226_226519

/--
Player 1 has a winning strategy if and only if N is not an odd power of 2,
under the game rules where players alternately subtract proper divisors
and a player loses when given a prime number or 1.
-/
theorem player1_winning_strategy (N: ℕ) : 
  ¬ (∃ k: ℕ, k % 2 = 1 ∧ N = 2^k) ↔ (∃ strategy: ℕ → ℕ, ∀ n ≠ 1, n ≠ prime → n - strategy n = m) :=
sorry

end player1_winning_strategy_l226_226519


namespace find_abc_value_l226_226249

noncomputable def abc_value_condition (a b c : ℝ) : Prop := 
  a + b + c = 4 ∧
  b * c + c * a + a * b = 5 ∧
  a^3 + b^3 + c^3 = 10

theorem find_abc_value (a b c : ℝ) (h : abc_value_condition a b c) : a * b * c = 2 := 
sorry

end find_abc_value_l226_226249


namespace total_marbles_l226_226214

namespace MarbleBag

def numBlue : ℕ := 5
def numRed : ℕ := 9
def probRedOrWhite : ℚ := 5 / 6

theorem total_marbles (total_mar : ℕ) (numWhite : ℕ) (h1 : probRedOrWhite = (numRed + numWhite) / total_mar)
                      (h2 : total_mar = numBlue + numRed + numWhite) :
  total_mar = 30 :=
by
  sorry

end MarbleBag

end total_marbles_l226_226214


namespace red_balls_removed_to_certain_event_l226_226829

theorem red_balls_removed_to_certain_event (total_balls red_balls yellow_balls : ℕ) (m : ℕ)
  (total_balls_eq : total_balls = 8)
  (red_balls_eq : red_balls = 3)
  (yellow_balls_eq : yellow_balls = 5)
  (certain_event_A : ∀ remaining_red_balls remaining_yellow_balls,
    remaining_red_balls = red_balls - m → remaining_yellow_balls = yellow_balls →
    remaining_red_balls = 0) : m = 3 :=
by
  sorry

end red_balls_removed_to_certain_event_l226_226829


namespace count_books_in_row_on_tuesday_l226_226908

-- Define the given conditions
def tiles_count_monday : ℕ := 38
def books_count_monday : ℕ := 75
def total_count_tuesday : ℕ := 301
def tiles_count_tuesday := tiles_count_monday * 2

-- The Lean statement we need to prove
theorem count_books_in_row_on_tuesday (hcbooks : books_count_monday = 75) 
(hc1 : total_count_tuesday = 301) 
(hc2 : tiles_count_tuesday = tiles_count_monday * 2):
  (total_count_tuesday - tiles_count_tuesday) / books_count_monday = 3 :=
by
  sorry

end count_books_in_row_on_tuesday_l226_226908


namespace area_of_region_l226_226090

open Set

theorem area_of_region:
  let R := {p : ℝ × ℝ | |p.1 - 2| ≤ p.2 ∧ p.2 ≤ 5 - |p.1 + 1|} in
  measurable_set R →
  ∫ p in R, 1 = 10 :=
by
  sorry

end area_of_region_l226_226090


namespace steve_total_payment_l226_226021

def mike_dvd_cost : ℝ := 5
def steve_dvd_cost : ℝ := 2 * mike_dvd_cost
def additional_dvd_cost : ℝ := 7
def steve_additional_dvds : ℝ := 2 * additional_dvd_cost
def total_dvd_cost : ℝ := steve_dvd_cost + steve_additional_dvds
def shipping_cost : ℝ := 0.80 * total_dvd_cost
def subtotal_with_shipping : ℝ := total_dvd_cost + shipping_cost
def sales_tax : ℝ := 0.10 * subtotal_with_shipping
def total_amount_paid : ℝ := subtotal_with_shipping + sales_tax

theorem steve_total_payment : total_amount_paid = 47.52 := by
  sorry

end steve_total_payment_l226_226021


namespace lcm_18_30_eq_90_l226_226466

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l226_226466


namespace scientific_notation_of_number_l226_226645

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l226_226645


namespace giant_spider_leg_cross_sectional_area_l226_226634

theorem giant_spider_leg_cross_sectional_area :
  let previous_spider_weight := 6.4
  let weight_multiplier := 2.5
  let pressure := 4
  let num_legs := 8

  let giant_spider_weight := weight_multiplier * previous_spider_weight
  let weight_per_leg := giant_spider_weight / num_legs
  let cross_sectional_area := weight_per_leg / pressure

  cross_sectional_area = 0.5 :=
by 
  sorry

end giant_spider_leg_cross_sectional_area_l226_226634


namespace base_five_to_base_ten_modulo_seven_l226_226353

-- Define the base five number 21014_5 as the corresponding base ten conversion
def base_five_number : ℕ := 2 * 5^4 + 1 * 5^3 + 0 * 5^2 + 1 * 5^1 + 4 * 5^0

-- The equivalent base ten result
def base_ten_number : ℕ := 1384

-- Verify the base ten equivalent of 21014_5
theorem base_five_to_base_ten : base_five_number = base_ten_number :=
by
  -- The expected proof should compute the value of base_five_number
  -- and check that it equals 1384
  sorry

-- Find the modulo operation result of 1384 % 7
def modulo_seven_result : ℕ := 6

-- Verify 1384 % 7 gives 6
theorem modulo_seven : base_ten_number % 7 = modulo_seven_result :=
by
  -- The expected proof should compute 1384 % 7
  -- and check that it equals 6
  sorry

end base_five_to_base_ten_modulo_seven_l226_226353


namespace find_divisor_l226_226207

def dividend := 23
def quotient := 4
def remainder := 3

theorem find_divisor (d : ℕ) (h : dividend = (d * quotient) + remainder) : d = 5 :=
by {
  sorry
}

end find_divisor_l226_226207


namespace fred_red_marbles_l226_226373

variable (R G B : ℕ)
variable (total : ℕ := 63)
variable (B_val : ℕ := 6)
variable (G_def : G = (1 / 2) * R)
variable (eq1 : R + G + B = total)
variable (eq2 : B = B_val)

theorem fred_red_marbles : R = 38 := 
by
  sorry

end fred_red_marbles_l226_226373


namespace lap_distance_l226_226013

theorem lap_distance (boys_laps : ℕ) (girls_extra_laps : ℕ) (total_girls_miles : ℚ) : 
  boys_laps = 27 → girls_extra_laps = 9 → total_girls_miles = 27 →
  (total_girls_miles / (boys_laps + girls_extra_laps) = 3 / 4) :=
by
  intros hb hg hm
  sorry

end lap_distance_l226_226013


namespace g_at_3_l226_226122

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_at_3 : g 3 = -185 := by
  sorry

end g_at_3_l226_226122


namespace find_candies_l226_226092

variable (e : ℝ)

-- Given conditions
def candies_sum (e : ℝ) : ℝ := e + 4 * e + 16 * e + 96 * e

theorem find_candies (h : candies_sum e = 876) : e = 7.5 :=
by
  -- proof omitted
  sorry

end find_candies_l226_226092


namespace pipe_flow_rate_is_correct_l226_226213

-- Definitions for the conditions
def tank_capacity : ℕ := 10000
def initial_water : ℕ := tank_capacity / 2
def fill_time : ℕ := 60
def drain1_rate : ℕ := 1000
def drain1_interval : ℕ := 4
def drain2_rate : ℕ := 1000
def drain2_interval : ℕ := 6

-- Calculation based on conditions
def total_water_needed : ℕ := tank_capacity - initial_water
def drain1_loss (time : ℕ) : ℕ := (time / drain1_interval) * drain1_rate
def drain2_loss (time : ℕ) : ℕ := (time / drain2_interval) * drain2_rate
def total_drain_loss (time : ℕ) : ℕ := drain1_loss time + drain2_loss time

-- Target flow rate for the proof
def total_fill (time : ℕ) : ℕ := total_water_needed + total_drain_loss time
def pipe_flow_rate : ℕ := total_fill fill_time / fill_time

-- Statement to prove
theorem pipe_flow_rate_is_correct : pipe_flow_rate = 500 := by  
  sorry

end pipe_flow_rate_is_correct_l226_226213


namespace complete_the_square_l226_226915

theorem complete_the_square (x : ℝ) (h : x^2 - 8 * x - 1 = 0) : (x - 4)^2 = 17 :=
by
  -- proof steps would go here, but we use sorry for now
  sorry

end complete_the_square_l226_226915


namespace opposite_sign_pairs_l226_226652

theorem opposite_sign_pairs :
  ¬ ((- 2 ^ 3 < 0) ∧ (- (2 ^ 3) > 0)) ∧
  ¬ (|-4| < 0 ∧ -(-4) > 0) ∧
  ((- 3 ^ 4 < 0 ∧ (-(3 ^ 4)) = 81)) ∧
  ¬ (10 ^ 2 < 0 ∧ 2 ^ 10 > 0) :=
by
  sorry

end opposite_sign_pairs_l226_226652


namespace max_consecutive_sum_k_l226_226371

theorem max_consecutive_sum_k : 
  ∃ k n : ℕ, k = 486 ∧ 3^11 = (0 to k-1).sum + n * k := 
sorry

end max_consecutive_sum_k_l226_226371


namespace amount_saved_percent_l226_226060

variable (S : ℝ)

theorem amount_saved_percent :
  (0.165 * S) / (0.10 * S) * 100 = 165 := sorry

end amount_saved_percent_l226_226060


namespace f_monotonic_f_odd_find_a_k_range_l226_226261
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

-- (1) Prove the monotonicity of the function f
theorem f_monotonic (a : ℝ) : ∀ {x y : ℝ}, x < y → f a x < f a y := sorry

-- (2) If f is an odd function, find the value of the real number a
theorem f_odd_find_a : ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) → a = -1/2 := sorry

-- (3) Under the condition in (2), if the inequality holds for all x ∈ ℝ, find the range of values for k
theorem k_range (k : ℝ) :
  (∀ x : ℝ, f (-1/2) (x^2 - 2*x) + f (-1/2) (2*x^2 - k) > 0) → k < -1/3 := sorry

end f_monotonic_f_odd_find_a_k_range_l226_226261


namespace find_sum_l226_226203

-- Define the prime conditions
variables (P : ℝ) (SI15 SI12 : ℝ)

-- Assume conditions for the problem
axiom h1 : SI15 = P * 15 / 100 * 2
axiom h2 : SI12 = P * 12 / 100 * 2
axiom h3 : SI15 - SI12 = 840

-- Prove that P = 14000
theorem find_sum : P = 14000 :=
sorry

end find_sum_l226_226203


namespace jesse_total_carpet_l226_226567

theorem jesse_total_carpet : 
  let length_rect := 12
  let width_rect := 8
  let base_tri := 10
  let height_tri := 6
  let area_rect := length_rect * width_rect
  let area_tri := (base_tri * height_tri) / 2
  area_rect + area_tri = 126 :=
by
  sorry

end jesse_total_carpet_l226_226567


namespace choose_two_out_of_three_l226_226923

theorem choose_two_out_of_three : Nat.choose 3 2 = 3 := by
  sorry

end choose_two_out_of_three_l226_226923


namespace flavoring_ratio_comparison_l226_226143

theorem flavoring_ratio_comparison (f_st cs_st w_st : ℕ) (f_sp cs_sp w_sp : ℕ) :
  f_st = 1 → cs_st = 12 → w_st = 30 →
  w_sp = 75 → cs_sp = 5 →
  f_sp / w_sp = f_st / (2 * w_st) →
  (f_st / cs_st) * 3 = f_sp / cs_sp :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end flavoring_ratio_comparison_l226_226143


namespace find_P_plus_Q_l226_226005

theorem find_P_plus_Q (P Q : ℝ) (h : ∃ b c : ℝ, (x^2 + 3 * x + 4) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) : 
P + Q = 15 :=
by
  sorry

end find_P_plus_Q_l226_226005


namespace calc1_calc2_calc3_l226_226661

theorem calc1 : -4 - 4 = -8 := by
  sorry

theorem calc2 : (-32) / 4 = -8 := by
  sorry

theorem calc3 : -(-2)^3 = 8 := by
  sorry

end calc1_calc2_calc3_l226_226661


namespace calculate_cubic_sum_roots_l226_226948

noncomputable def α := (27 : ℝ)^(1/3)
noncomputable def β := (64 : ℝ)^(1/3)
noncomputable def γ := (125 : ℝ)^(1/3)

theorem calculate_cubic_sum_roots (u v w : ℝ) :
  (u - α) * (u - β) * (u - γ) = 1/2 ∧
  (v - α) * (v - β) * (v - γ) = 1/2 ∧
  (w - α) * (w - β) * (w - γ) = 1/2 →
  u^3 + v^3 + w^3 = 217.5 :=
by
  sorry

end calculate_cubic_sum_roots_l226_226948


namespace gaokao_probability_l226_226292

open Finset

noncomputable def choose_2_out_of_5 (s : Finset ℕ) : Finset (Sym2 ℕ) := 
  s.powerset.filter (λ x, x.card = 2).map (λ x, ⟨x.1, x.2⟩)

theorem gaokao_probability :
  let subjects : Finset ℕ := {1, 2, 3, 4, 5} -- where 1, 2, 3, 4, 5 represent Chemistry, Biology, IPE, History, Geography respectively
  let total_outcomes := 10  -- since choosing 2 out of 5 elements
  let history_geography := {4, 5}
  let favorable_outcomes := (choose_2_out_of_5 subjects).filter (λ x, x.1 ∈ history_geography ∨ x.2 ∈ history_geography)
  in
  (favorable_outcomes.card : ℚ) / total_outcomes = 7 / 10 := sorry

end gaokao_probability_l226_226292


namespace lcm_18_30_l226_226477

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l226_226477


namespace value_of_expression_l226_226671

noncomputable def x := (2 : ℚ) / 3
noncomputable def y := (5 : ℚ) / 2

theorem value_of_expression : (1 / 3) * x^8 * y^9 = (5^9 / (2 * 3^9)) := by
  sorry

end value_of_expression_l226_226671


namespace sparrows_among_non_robins_percentage_l226_226009

-- Define percentages of different birds
def finches_percentage : ℝ := 0.40
def sparrows_percentage : ℝ := 0.20
def owls_percentage : ℝ := 0.15
def robins_percentage : ℝ := 0.25

-- Define the statement to prove 
theorem sparrows_among_non_robins_percentage :
  ((sparrows_percentage / (1 - robins_percentage)) * 100) = 26.67 := by
  -- This is where the proof would go, but it's omitted as per instructions
  sorry

end sparrows_among_non_robins_percentage_l226_226009


namespace inhabitants_reach_ball_on_time_l226_226282

theorem inhabitants_reach_ball_on_time
  (kingdom_side_length : ℝ)
  (messenger_sent_at : ℕ)
  (ball_begins_at : ℕ)
  (inhabitant_speed : ℝ)
  (time_available : ℝ)
  (max_distance_within_square : ℝ)
  (H_side_length : kingdom_side_length = 2)
  (H_messenger_time : messenger_sent_at = 12)
  (H_ball_time : ball_begins_at = 19)
  (H_speed : inhabitant_speed = 3)
  (H_time_avail : time_available = 7)
  (H_max_distance : max_distance_within_square = 2 * Real.sqrt 2) :
  ∃ t : ℝ, t ≤ time_available ∧ max_distance_within_square / inhabitant_speed ≤ t :=
by
  -- You would write the proof here.
  sorry

end inhabitants_reach_ball_on_time_l226_226282


namespace pizza_topping_cost_l226_226877

/- 
   Given:
   1. Ruby ordered 3 pizzas.
   2. Each pizza costs $10.00.
   3. The total number of toppings were 4.
   4. Ruby added a $5.00 tip to the order.
   5. The total cost of the order, including tip, was $39.00.

   Prove: The cost per topping is $1.00.
-/
theorem pizza_topping_cost (cost_per_pizza : ℝ) (total_pizzas : ℕ) (tip : ℝ) (total_cost : ℝ) 
    (total_toppings : ℕ) (x : ℝ) : 
    cost_per_pizza = 10 → total_pizzas = 3 → tip = 5 → total_cost = 39 → total_toppings = 4 → 
    total_cost = cost_per_pizza * total_pizzas + x * total_toppings + tip →
    x = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end pizza_topping_cost_l226_226877


namespace circle_center_coordinates_l226_226732

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (∀ x y : ℝ, x^2 + y^2 - x + 2*y = 0 ↔ (x-c.1)^2 + (y-c.2)^2 = (5/4)) ∧ c = (1/2, -1) :=
sorry

end circle_center_coordinates_l226_226732


namespace side_length_S2_l226_226164

-- Define the variables
variables (r s : ℕ)

-- Given conditions
def condition1 : Prop := 2 * r + s = 2300
def condition2 : Prop := 2 * r + 3 * s = 4000

-- The main statement to be proven
theorem side_length_S2 (h1 : condition1 r s) (h2 : condition2 r s) : s = 850 := sorry

end side_length_S2_l226_226164


namespace print_time_is_fifteen_l226_226343

noncomputable def time_to_print (total_pages rate : ℕ) := 
  (total_pages : ℚ) / rate

theorem print_time_is_fifteen :
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  round time = 15 := by
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  have time_val : time = (350 : ℚ) / 24 := by rfl
  let rounded_time := round time
  have rounded_time_val : rounded_time = 15 := by sorry
  exact rounded_time_val

end print_time_is_fifteen_l226_226343


namespace sandra_tickets_relation_l226_226748

def volleyball_game : Prop :=
  ∃ (tickets_total tickets_left tickets_jude tickets_andrea tickets_sandra : ℕ),
    tickets_total = 100 ∧
    tickets_left = 40 ∧
    tickets_jude = 16 ∧
    tickets_andrea = 2 * tickets_jude ∧
    tickets_total - tickets_left = tickets_jude + tickets_andrea + tickets_sandra ∧
    tickets_sandra = tickets_jude - 4

theorem sandra_tickets_relation : volleyball_game :=
  sorry

end sandra_tickets_relation_l226_226748


namespace range_of_a_values_l226_226359

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0

theorem range_of_a_values (a : ℝ) : range_of_a a ↔ a ≥ 1/2 :=
by
  sorry

end range_of_a_values_l226_226359


namespace boys_planted_more_by_62_percent_girls_fraction_of_total_l226_226695

-- Define the number of trees planted by boys and girls
def boys_trees : ℕ := 130
def girls_trees : ℕ := 80

-- Statement 1: Boys planted 62% more trees than girls
theorem boys_planted_more_by_62_percent : (boys_trees - girls_trees) * 100 / girls_trees = 62 := by
  sorry

-- Statement 2: The number of trees planted by girls represents 4/7 of the total number of trees
theorem girls_fraction_of_total : girls_trees * 7 = 4 * (boys_trees + girls_trees) := by
  sorry

end boys_planted_more_by_62_percent_girls_fraction_of_total_l226_226695


namespace m_squared_minus_n_squared_plus_one_is_perfect_square_l226_226448

theorem m_squared_minus_n_squared_plus_one_is_perfect_square (m n : ℤ)
  (hm : m % 2 = 1) (hn : n % 2 = 1)
  (h : m^2 - n^2 + 1 ∣ n^2 - 1) :
  ∃ k : ℤ, k^2 = m^2 - n^2 + 1 :=
sorry

end m_squared_minus_n_squared_plus_one_is_perfect_square_l226_226448


namespace imag_part_of_complex_squared_is_2_l226_226453

-- Define the complex number 1 + i
def complex_num := (1 : ℂ) + (Complex.I : ℂ)

-- Define the squared value of the complex number
def complex_squared := complex_num ^ 2

-- Define the imaginary part of the squared value
def imag_part := complex_squared.im

-- State the theorem
theorem imag_part_of_complex_squared_is_2 : imag_part = 2 := sorry

end imag_part_of_complex_squared_is_2_l226_226453


namespace like_terms_exponents_product_l226_226980

theorem like_terms_exponents_product (m n : ℤ) (a b : ℝ) 
  (h1 : 3 * a^m * b^2 = -1 * a^2 * b^(n+3)) : m * n = -2 :=
  sorry

end like_terms_exponents_product_l226_226980


namespace sin_neg_seven_pi_over_three_correct_l226_226044

noncomputable def sin_neg_seven_pi_over_three : Prop :=
  (Real.sin (-7 * Real.pi / 3) = - (Real.sqrt 3 / 2))

theorem sin_neg_seven_pi_over_three_correct : sin_neg_seven_pi_over_three := 
by
  sorry

end sin_neg_seven_pi_over_three_correct_l226_226044


namespace parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l226_226137

theorem parts_from_blanks_9 : ∀ (produced_parts : ℕ), produced_parts = 13 :=
by
  sorry

theorem parts_from_blanks_14 : ∀ (produced_parts : ℕ), produced_parts = 20 :=
by
  sorry

theorem blanks_needed_for_40_parts : ∀ (required_blanks : ℕ), required_blanks = 27 :=
by
  sorry

end parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l226_226137


namespace work_completion_in_8_days_l226_226629

/-- Definition of the individual work rates and the combined work rate. -/
def work_rate_A := 1 / 12
def work_rate_B := 1 / 24
def combined_work_rate := work_rate_A + work_rate_B

/-- The main theorem stating that A and B together complete the job in 8 days. -/
theorem work_completion_in_8_days (h1 : work_rate_A = 1 / 12) (h2 : work_rate_B = 1 / 24) : 
  1 / combined_work_rate = 8 :=
by
  sorry

end work_completion_in_8_days_l226_226629


namespace number_of_solutions_eq_two_l226_226738

theorem number_of_solutions_eq_two : 
  (∃ (x y : ℝ), x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) ∧
  (∀ (x y : ℝ), (x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) → ((x = 4 ∨ x = -1) ∧ y = 3)) :=
by
  sorry

end number_of_solutions_eq_two_l226_226738


namespace number_of_groups_of_three_books_l226_226554

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l226_226554


namespace Julia_played_kids_on_Monday_l226_226998

theorem Julia_played_kids_on_Monday
  (t : ℕ) (w : ℕ) (h1 : t = 18) (h2 : w = 97) (h3 : t + m = 33) :
  ∃ m : ℕ, m = 15 :=
by
  sorry

end Julia_played_kids_on_Monday_l226_226998


namespace frequency_of_group_5_l226_226341

theorem frequency_of_group_5 (total_students freq1 freq2 freq3 freq4 : ℕ)
  (h_total: total_students = 50) 
  (h_freq1: freq1 = 7) 
  (h_freq2: freq2 = 12) 
  (h_freq3: freq3 = 13) 
  (h_freq4: freq4 = 8) :
  (50 - (7 + 12 + 13 + 8)) / 50 = 0.2 :=
by
  sorry

end frequency_of_group_5_l226_226341


namespace measure_angle_B_triangle_area_correct_l226_226254

noncomputable def triangle_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) → B = Real.pi / 3

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area1 := (3 + Real.sqrt 3)
  let area2 := Real.sqrt 3
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  let sinA1 := (Real.sqrt 2 / 2)
  let sinA2 := (Real.sqrt 6 - Real.sqrt 2) / 4
  let S1 := (1 / 2) * b * c * sinA1
  let S2 := (1 / 2) * b * c * sinA2
  S1 = area1 ∨ S2 = area2

theorem measure_angle_B :
  ∀ (a b c A B C : ℝ),
    triangle_angle_B a b c A B C := sorry

theorem triangle_area_correct :
  ∀ (a b c A B C : ℝ),
    triangle_area a b c A B C := sorry

end measure_angle_B_triangle_area_correct_l226_226254


namespace range_of_values_for_a_l226_226975

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.sin x - (1 / 2) * Real.cos (2 * x) + a - (3 / a) + (1 / 2)

theorem range_of_values_for_a (a : ℝ) (ha : a ≠ 0) : 
  (∀ x : ℝ, f x a ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
by 
  let g (t : ℝ) : ℝ := t^2 + a * t + a - (3 / a)
  have h1 : g (-1) ≤ 0 := by sorry
  have h2 : g (1) ≤ 0 := by sorry
  sorry

end range_of_values_for_a_l226_226975


namespace range_of_k_l226_226278

theorem range_of_k (k : ℝ) (H : ∀ x : ℤ, |(x : ℝ) - 1| < k * x ↔ x ∈ ({1, 2, 3} : Set ℤ)) : 
  (2 / 3 : ℝ) < k ∧ k ≤ (3 / 4 : ℝ) :=
by
  sorry

end range_of_k_l226_226278


namespace joan_gave_apples_l226_226148

theorem joan_gave_apples (initial_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : initial_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  -- Show that given_apples is obtained by subtracting remaining_apples from initial_apples
  sorry

end joan_gave_apples_l226_226148


namespace number_one_fourth_more_than_it_is_30_percent_less_than_80_l226_226599

theorem number_one_fourth_more_than_it_is_30_percent_less_than_80 :
    ∃ (n : ℝ), (5 / 4) * n = 56 ∧ n = 45 :=
by
  sorry

end number_one_fourth_more_than_it_is_30_percent_less_than_80_l226_226599


namespace sufficient_not_necessary_l226_226104

theorem sufficient_not_necessary (a b : ℝ) :
  (a = -1 ∧ b = 2 → a * b = -2) ∧ (a * b = -2 → ¬(a = -1 ∧ b = 2)) :=
by
  sorry

end sufficient_not_necessary_l226_226104


namespace calculate_expr_l226_226655

theorem calculate_expr :
  ( (5 / 12: ℝ) ^ 2022) * (-2.4) ^ 2023 = - (12 / 5: ℝ) := 
by 
  sorry

end calculate_expr_l226_226655


namespace valid_tickets_percentage_l226_226076

theorem valid_tickets_percentage (cars : ℕ) (people_without_payment : ℕ) (P : ℚ) 
  (h_cars : cars = 300) (h_people_without_payment : people_without_payment = 30) 
  (h_total_valid_or_passes : (cars - people_without_payment = 270)) :
  P + (P / 5) = 90 → P = 75 :=
by
  sorry

end valid_tickets_percentage_l226_226076


namespace discount_percentage_l226_226283

theorem discount_percentage (p : ℝ) : 
  (1 + 0.25) * p * (1 - 0.20) = p :=
by
  sorry

end discount_percentage_l226_226283


namespace quadratic_inequality_solution_l226_226243

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end quadratic_inequality_solution_l226_226243


namespace james_total_cost_l226_226220

def subscription_cost (base_cost : ℕ) (free_hours : ℕ) (extra_hour_cost : ℕ) (movie_rental_cost : ℝ) (streamed_hours : ℕ) (rented_movies : ℕ) : ℝ :=
  let extra_hours := max (streamed_hours - free_hours) 0
  base_cost + extra_hours * extra_hour_cost + rented_movies * movie_rental_cost

theorem james_total_cost 
  (base_cost : ℕ)
  (free_hours : ℕ)
  (extra_hour_cost : ℕ)
  (movie_rental_cost : ℝ)
  (streamed_hours : ℕ)
  (rented_movies : ℕ)
  (h_base_cost : base_cost = 15)
  (h_free_hours : free_hours = 50)
  (h_extra_hour_cost : extra_hour_cost = 2)
  (h_movie_rental_cost : movie_rental_cost = 0.10)
  (h_streamed_hours : streamed_hours = 53)
  (h_rented_movies : rented_movies = 30) :
  subscription_cost base_cost free_hours extra_hour_cost movie_rental_cost streamed_hours rented_movies = 24 := 
by {
  sorry
}

end james_total_cost_l226_226220


namespace at_least_one_inequality_holds_l226_226427

theorem at_least_one_inequality_holds
    (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l226_226427


namespace negation_prop_l226_226035

theorem negation_prop : (¬(∃ x : ℝ, x + 2 ≤ 0)) ↔ (∀ x : ℝ, x + 2 > 0) := 
  sorry

end negation_prop_l226_226035


namespace probability_female_win_l226_226288

variable (P_Alexandr P_Alexandra P_Evgeniev P_Evgenii P_Valentinov P_Valentin P_Vasilev P_Vasilisa : ℝ)

-- Conditions
axiom h1 : P_Alexandr = 3 * P_Alexandra
axiom h2 : P_Evgeniev = (1 / 3) * P_Evgenii
axiom h3 : P_Valentinov = (1.5) * P_Valentin
axiom h4 : P_Vasilev = 49 * P_Vasilisa
axiom h5 : P_Alexandr + P_Alexandra = 1
axiom h6 : P_Evgeniev + P_Evgenii = 1
axiom h7 : P_Valentinov + P_Valentin = 1
axiom h8 : P_Vasilev + P_Vasilisa = 1

-- Statement to prove
theorem probability_female_win : 
  let P_female := (1/4) * P_Alexandra + (1/4) * P_Evgeniev + (1/4) * P_Valentinov + (1/4) * P_Vasilisa in
  P_female = 0.355 :=
by
  sorry

end probability_female_win_l226_226288


namespace divisible_by_5886_l226_226737

theorem divisible_by_5886 (r b c : ℕ) (h1 : (523000 + r * 1000 + b * 100 + c * 10) % 89 = 0) (h2 : r * b * c = 180) : 
  (523000 + r * 1000 + b * 100 + c * 10) % 5886 = 0 := 
sorry

end divisible_by_5886_l226_226737


namespace rectangle_placement_l226_226154

theorem rectangle_placement (a b c d : ℝ)
  (h1 : a < c)
  (h2 : c < d)
  (h3 : d < b)
  (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
sorry

end rectangle_placement_l226_226154


namespace distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l226_226218

-- We will assume the depth of the well as a constant
def well_depth : ℝ := 4.0

-- Climb and slide distances as per each climb
def first_climb : ℝ := 1.2
def first_slide : ℝ := 0.4
def second_climb : ℝ := 1.4
def second_slide : ℝ := 0.5
def third_climb : ℝ := 1.1
def third_slide : ℝ := 0.3
def fourth_climb : ℝ := 1.2
def fourth_slide : ℝ := 0.2

noncomputable def net_gain_four_climbs : ℝ :=
  (first_climb - first_slide) + (second_climb - second_slide) +
  (third_climb - third_slide) + (fourth_climb - fourth_slide)

noncomputable def distance_from_top_after_four : ℝ := 
  well_depth - net_gain_four_climbs

noncomputable def total_distance_covered_four_climbs : ℝ :=
  first_climb + first_slide + second_climb + second_slide +
  third_climb + third_slide + fourth_climb + fourth_slide

noncomputable def can_climb_out_fifth_climb : Bool :=
  well_depth < (net_gain_four_climbs + first_climb)

-- Now we state the theorems we need to prove

theorem distance_from_top_correct :
  distance_from_top_after_four = 0.5 := by
  sorry

theorem total_distance_covered_correct :
  total_distance_covered_four_climbs = 6.3 := by
  sorry

theorem fifth_climb_success :
  can_climb_out_fifth_climb = true := by
  sorry

end distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l226_226218


namespace domain_of_function_l226_226889

noncomputable def is_domain_of_function (x : ℝ) : Prop :=
  (4 - x^2 ≥ 0) ∧ (x ≠ 1)

theorem domain_of_function :
  {x : ℝ | is_domain_of_function x} = {x : ℝ | -2 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l226_226889


namespace problem_statement_l226_226761

theorem problem_statement (x : ℝ) (hx : x^2 + 1/(x^2) = 2) : x^4 + 1/(x^4) = 2 := by
  sorry

end problem_statement_l226_226761


namespace ages_l226_226067

-- Definitions of ages
variables (S M : ℕ) -- S: son's current age, M: mother's current age

-- Given conditions
def father_age : ℕ := 44
def son_father_relationship (S : ℕ) : Prop := father_age = S + S
def son_mother_relationship (S M : ℕ) : Prop := (S - 5) = (M - 10)

-- Theorem to prove the ages
theorem ages (S M : ℕ) (h1 : son_father_relationship S) (h2 : son_mother_relationship S M) :
  S = 22 ∧ M = 27 :=
by 
  sorry

end ages_l226_226067


namespace hard_candy_food_colouring_l226_226930

theorem hard_candy_food_colouring :
  (∀ lollipop_colour hard_candy_count total_food_colouring lollipop_count hard_candy_food_total_per_lollipop,
    lollipop_colour = 5 →
    lollipop_count = 100 →
    hard_candy_count = 5 →
    total_food_colouring = 600 →
    hard_candy_food_total_per_lollipop = lollipop_colour * lollipop_count →
    total_food_colouring - hard_candy_food_total_per_lollipop = hard_candy_count * hard_candy_food_total_per_candy →
    hard_candy_food_total_per_candy = 20) :=
by
  sorry

end hard_candy_food_colouring_l226_226930


namespace rectangles_greater_than_one_area_l226_226399

theorem rectangles_greater_than_one_area (n : ℕ) (H : n = 5) : ∃ r, r = 84 :=
by
  sorry

end rectangles_greater_than_one_area_l226_226399


namespace inscribed_triangle_area_l226_226074

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin A + Real.sin B + Real.sin C)

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  ∀ (A B C : ℝ), A = 40 * Real.pi / 180 → B = 80 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  triangle_area r A B C = 359.4384 / Real.pi^2 :=
by
  intros
  unfold triangle_area
  sorry

end inscribed_triangle_area_l226_226074


namespace cooking_time_per_side_l226_226517

-- Defining the problem conditions
def total_guests : ℕ := 30
def guests_wanting_2_burgers : ℕ := total_guests / 2
def guests_wanting_1_burger : ℕ := total_guests / 2
def burgers_per_guest_2 : ℕ := 2
def burgers_per_guest_1 : ℕ := 1
def total_burgers : ℕ := guests_wanting_2_burgers * burgers_per_guest_2 + guests_wanting_1_burger * burgers_per_guest_1
def burgers_per_batch : ℕ := 5
def total_batches : ℕ := total_burgers / burgers_per_batch
def total_cooking_time : ℕ := 72
def time_per_batch : ℕ := total_cooking_time / total_batches
def sides_per_burger : ℕ := 2

-- the theorem to prove the desired cooking time per side
theorem cooking_time_per_side : (time_per_batch / sides_per_burger) = 4 := by {
    -- Here we would enter the proof steps, but this is omitted as per the instructions.
    sorry
}

end cooking_time_per_side_l226_226517


namespace area_of_triangle_PQS_l226_226139

-- Define a structure to capture the conditions of the trapezoid and its properties.
structure Trapezoid (P Q R S : Type) :=
(area : ℝ)
(PQ : ℝ)
(RS : ℝ)
(area_PQS : ℝ)
(condition1 : area = 18)
(condition2 : RS = 3 * PQ)

-- Here's the theorem we want to prove, stating the conclusion based on the given conditions.
theorem area_of_triangle_PQS {P Q R S : Type} (T : Trapezoid P Q R S) : T.area_PQS = 4.5 :=
by
  -- Proof will go here, but for now we use sorry.
  sorry

end area_of_triangle_PQS_l226_226139


namespace function_property_l226_226546

variable (g : ℝ × ℝ → ℝ)
variable (cond : ∀ x y : ℝ, g (x, y) = - g (y, x))

theorem function_property (x : ℝ) : g (x, x) = 0 :=
by
  sorry

end function_property_l226_226546


namespace price_of_72_cans_l226_226037

def regular_price_per_can : ℝ := 0.60
def discount_percentage : ℝ := 0.20
def total_price : ℝ := 34.56

theorem price_of_72_cans (discounted_price_per_can : ℝ) (number_of_cans : ℕ)
  (H1 : discounted_price_per_can = regular_price_per_can - (discount_percentage * regular_price_per_can))
  (H2 : number_of_cans = total_price / discounted_price_per_can) :
  total_price = number_of_cans * discounted_price_per_can := by
  sorry

end price_of_72_cans_l226_226037


namespace find_a_l226_226387

theorem find_a
  (a : ℝ)
  (h1 : ∃ P Q : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 - 2 * P.1 + 4 * P.2 + 1 = 0) ∧ (Q.1 ^ 2 + Q.2 ^ 2 - 2 * Q.1 + 4 * Q.2 + 1 = 0) ∧
                         (a * P.1 + 2 * P.2 + 6 = 0) ∧ (a * Q.1 + 2 * Q.2 + 6 = 0) ∧
                         ((P.1 - 1) * (Q.1 - 1) + (P.2 + 2) * (Q.2 + 2) = 0)) :
  a = 2 :=
by
  sorry

end find_a_l226_226387


namespace trajectory_of_intersection_l226_226256

-- Define the conditions and question in Lean
structure Point where
  x : ℝ
  y : ℝ

def on_circle (C : Point) : Prop :=
  C.x^2 + C.y^2 = 1

def perp_to_x_axis (C D : Point) : Prop :=
  C.x = D.x ∧ C.y = -D.y

theorem trajectory_of_intersection (A B C D M : Point)
  (hA : A = {x := -1, y := 0})
  (hB : B = {x := 1, y := 0})
  (hC : on_circle C)
  (hD : on_circle D)
  (hCD : perp_to_x_axis C D)
  (hM : ∃ m n : ℝ, C = {x := m, y := n} ∧ M = {x := 1 / m, y := n / m}) :
  M.x^2 - M.y^2 = 1 ∧ M.y ≠ 0 :=
by
  sorry

end trajectory_of_intersection_l226_226256


namespace cone_lateral_area_l226_226385

theorem cone_lateral_area (C l r A : ℝ) (hC : C = 4 * Real.pi) (hl : l = 3) 
  (hr : 2 * Real.pi * r = 4 * Real.pi) (hA : A = Real.pi * r * l) : A = 6 * Real.pi :=
by
  sorry

end cone_lateral_area_l226_226385


namespace integral_solutions_count_l226_226384

theorem integral_solutions_count (m : ℕ) (h : m > 0) :
  ∃ S : Finset (ℕ × ℕ), S.card = m ∧ 
  ∀ (p : ℕ × ℕ), p ∈ S → (p.1^2 + p.2^2 + 2 * p.1 * p.2 - m * p.1 - m * p.2 - m - 1 = 0) := 
sorry

end integral_solutions_count_l226_226384


namespace lcm_18_30_is_90_l226_226481

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l226_226481


namespace time_left_to_use_exerciser_l226_226416

-- Definitions based on the conditions
def total_time : ℕ := 2 * 60  -- Total time in minutes (120 minutes)
def piano_time : ℕ := 30  -- Time spent on piano
def writing_music_time : ℕ := 25  -- Time spent on writing music
def history_time : ℕ := 38  -- Time spent on history

-- The theorem statement that Joan has 27 minutes left
theorem time_left_to_use_exerciser : 
  total_time - (piano_time + writing_music_time + history_time) = 27 :=
by {
  sorry
}

end time_left_to_use_exerciser_l226_226416


namespace numerator_in_second_fraction_l226_226826

theorem numerator_in_second_fraction (p q x: ℚ) (h1 : p / q = 4 / 5) (h2 : 11 / 7 + x / (2 * q + p) = 2) : x = 6 :=
sorry

end numerator_in_second_fraction_l226_226826


namespace common_divisors_count_48_80_l226_226684

noncomputable def prime_factors_48 : Nat -> Prop
| n => n = 48

noncomputable def prime_factors_80 : Nat -> Prop
| n => n = 80

theorem common_divisors_count_48_80 :
  let gcd_48_80 := 2^4
  let divisors_of_gcd := [1, 2, 4, 8, 16]
  prime_factors_48 48 ∧ prime_factors_80 80 →
  List.length divisors_of_gcd = 5 :=
by
  intros
  sorry

end common_divisors_count_48_80_l226_226684


namespace tomato_puree_water_percentage_l226_226000

theorem tomato_puree_water_percentage :
  (∀ (juice_purity water_percentage : ℝ), 
    (juice_purity = 0.90) → 
    (20 * juice_purity = 18) →
    (2.5 - 2) = 0.5 →
    (2.5 * water_percentage - 0.5) = 0 →
    water_percentage = 0.20) :=
by
  intros juice_purity water_percentage h1 h2 h3 h4
  sorry

end tomato_puree_water_percentage_l226_226000


namespace proof_expression_l226_226952

open Real

theorem proof_expression (x y : ℝ) (h1 : P = 2 * (x + y)) (h2 : Q = 3 * (x - y)) :
  (P + Q) / (P - Q) - (P - Q) / (P + Q) + (x + y) / (x - y) = (28 * x^2 - 20 * y^2) / ((x - y) * (5 * x - y) * (-x + 5 * y)) :=
by
  sorry

end proof_expression_l226_226952


namespace triangle_area_l226_226177

theorem triangle_area :
  ∀ (k : ℝ), ∃ (area : ℝ), 
  (∃ (r : ℝ) (a b c : ℝ), 
      r = 2 * Real.sqrt 3 ∧
      a / b = 3 / 5 ∧ a / c = 3 / 7 ∧ b / c = 5 / 7 ∧
      (∃ (A B C : ℝ),
          A = 3 * k ∧ B = 5 * k ∧ C = 7 * k ∧
          area = (1/2) * a * b * Real.sin (2 * Real.pi / 3))) →
  area = (135 * Real.sqrt 3 / 49) :=
sorry

end triangle_area_l226_226177


namespace eccentricity_ellipse_l226_226701

variable (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variable (c : ℝ) (h3 : c = Real.sqrt (a ^ 2 - b ^ 2))
variable (h4 : b = c)
variable (ellipse_eq : ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1)

theorem eccentricity_ellipse :
  c / a = Real.sqrt 2 / 2 :=
by
  sorry

end eccentricity_ellipse_l226_226701


namespace average_height_students_count_l226_226560

-- Definitions based on the conditions
def total_students : ℕ := 400
def short_students : ℕ := (2 * total_students) / 5
def extremely_tall_students : ℕ := total_students / 10
def tall_students : ℕ := 90
def average_height_students : ℕ := total_students - (short_students + tall_students + extremely_tall_students)

-- Theorem to prove
theorem average_height_students_count : average_height_students = 110 :=
by
  -- This proof is omitted, we are only stating the theorem.
  sorry

end average_height_students_count_l226_226560


namespace children_got_on_bus_l226_226621

theorem children_got_on_bus (initial_children total_children children_added : ℕ) 
  (h_initial : initial_children = 64) 
  (h_total : total_children = 78) : 
  children_added = total_children - initial_children :=
by
  sorry

end children_got_on_bus_l226_226621


namespace race_distance_l226_226350

theorem race_distance {a b c : ℝ} (h1 : b = 0.9 * a) (h2 : c = 0.95 * b) :
  let andrei_distance := 1000
  let boris_distance := andrei_distance - 100
  let valentin_distance := boris_distance - 50
  let valentin_actual_distance := (c / a) * andrei_distance
  andrei_distance - valentin_actual_distance = 145 :=
by
  sorry

end race_distance_l226_226350


namespace number_of_pigs_l226_226725

theorem number_of_pigs (daily_feed_per_pig : ℕ) (weekly_feed_total : ℕ) (days_per_week : ℕ)
  (h1 : daily_feed_per_pig = 10) (h2 : weekly_feed_total = 140) (h3 : days_per_week = 7) : 
  (weekly_feed_total / days_per_week) / daily_feed_per_pig = 2 := by
  sorry

end number_of_pigs_l226_226725


namespace even_and_monotonically_decreasing_l226_226199

noncomputable def f_B (x : ℝ) : ℝ := 1 / (x^2)

theorem even_and_monotonically_decreasing (x : ℝ) (h : x > 0) :
  (f_B x = f_B (-x)) ∧ (∀ {a b : ℝ}, a < b → a > 0 → b > 0 → f_B a > f_B b) :=
by
  sorry

end even_and_monotonically_decreasing_l226_226199


namespace wrappers_after_collection_l226_226790

theorem wrappers_after_collection (caps_found : ℕ) (wrappers_found : ℕ) (current_caps : ℕ) (initial_caps : ℕ) : 
  caps_found = 22 → wrappers_found = 30 → current_caps = 17 → initial_caps = 0 → 
  wrappers_found ≥ 30 := 
by 
  intros h1 h2 h3 h4
  -- Solution steps are omitted on purpose
  --- This is where the proof is written
  sorry

end wrappers_after_collection_l226_226790


namespace divisors_congruent_mod8_l226_226155

theorem divisors_congruent_mod8 (n : ℕ) (hn : n % 2 = 1) :
  ∀ d, d ∣ (2^n - 1) → d % 8 = 1 ∨ d % 8 = 7 :=
by
  sorry

end divisors_congruent_mod8_l226_226155


namespace arithmetic_sequence_geometric_sum_l226_226424

theorem arithmetic_sequence_geometric_sum (a1 : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), S 1 = a1)
  (h2 : ∀ (n : ℕ), S 2 = 2 * a1 - 1)
  (h3 : ∀ (n : ℕ), S 4 = 4 * a1 - 6)
  (h4 : (2 * a1 - 1)^2 = a1 * (4 * a1 - 6)) 
  : a1 = -1/2 := 
sorry

end arithmetic_sequence_geometric_sum_l226_226424


namespace sum_of_w_l226_226574

def g (y : ℝ) : ℝ := (2 * y)^3 - 2 * (2 * y) + 5

theorem sum_of_w (w1 w2 w3 : ℝ)
  (hw1 : g (2 * w1) = 13)
  (hw2 : g (2 * w2) = 13)
  (hw3 : g (2 * w3) = 13) :
  w1 + w2 + w3 = -1 / 4 :=
sorry

end sum_of_w_l226_226574


namespace smallest_possible_N_l226_226576

theorem smallest_possible_N (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) 
  (hr : r > 0) (hs : s > 0) (ht : t > 0) (h_sum : p + q + r + s + t = 4020) :
  ∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1005 :=
sorry

end smallest_possible_N_l226_226576


namespace positive_integers_of_inequality_l226_226591

theorem positive_integers_of_inequality (x : ℕ) (h : 9 - 3 * x > 0) : x = 1 ∨ x = 2 :=
sorry

end positive_integers_of_inequality_l226_226591


namespace not_set_of_difficult_problems_l226_226917

-- Define the context and entities
inductive Exercise
| ex (n : Nat) : Exercise  -- Example definition for exercises, assumed to be numbered

def is_difficult (ex : Exercise) : Prop := sorry  -- Placeholder for the subjective predicate

-- Define the main problem statement
theorem not_set_of_difficult_problems
  (Difficult : Exercise → Prop) -- Subjective predicate defining difficult problems
  (H_subj : ∀ (e : Exercise), (Difficult e ↔ is_difficult e)) :
  ¬(∃ (S : Set Exercise), ∀ e, e ∈ S ↔ Difficult e) :=
sorry

end not_set_of_difficult_problems_l226_226917


namespace problem_equiv_proof_l226_226079

theorem problem_equiv_proof :
  2015 * (1 + 1999 / 2015) * (1 / 4) - (2011 / 2015) = 503 := 
by
  sorry

end problem_equiv_proof_l226_226079


namespace sin_cos_theta_l226_226971

open Real

theorem sin_cos_theta (θ : ℝ) (H1 : θ > π / 2 ∧ θ < π) (H2 : tan (θ + π / 4) = 1 / 2) :
  sin θ + cos θ = -sqrt 10 / 5 :=
by
  sorry

end sin_cos_theta_l226_226971


namespace abs_iff_sq_gt_l226_226824

theorem abs_iff_sq_gt (x y : ℝ) : (|x| > |y|) ↔ (x^2 > y^2) :=
by sorry

end abs_iff_sq_gt_l226_226824


namespace evaluate_ninth_roots_of_unity_product_l226_226097

theorem evaluate_ninth_roots_of_unity_product : 
  (3 - Complex.exp (2 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (4 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (6 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (8 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (10 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (12 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (14 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (16 * Real.pi * Complex.I / 9)) 
  = 9841 := 
by 
  sorry

end evaluate_ninth_roots_of_unity_product_l226_226097


namespace sum_of_solutions_of_quadratic_eq_l226_226192

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l226_226192


namespace remaining_gnomes_total_l226_226171

/--
The remaining number of gnomes in the three forests after the owner takes his specified percentages.
-/
theorem remaining_gnomes_total :
  let westerville_gnomes := 20
  let ravenswood_gnomes := 4 * westerville_gnomes
  let greenwood_grove_gnomes := ravenswood_gnomes + (25 * ravenswood_gnomes) / 100
  let remaining_ravenswood := ravenswood_gnomes - (40 * ravenswood_gnomes) / 100
  let remaining_westerville := westerville_gnomes - (30 * westerville_gnomes) / 100
  let remaining_greenwood_grove := greenwood_grove_gnomes - (50 * greenwood_grove_gnomes) / 100
  remaining_ravenswood + remaining_westerville + remaining_greenwood_grove = 112 := by
  sorry

end remaining_gnomes_total_l226_226171


namespace negation_proposition_l226_226588

theorem negation_proposition:
  (¬ (∀ x : ℝ, (1 ≤ x) → (x^2 - 2*x + 1 ≥ 0))) ↔ (∃ x : ℝ, (1 ≤ x) ∧ (x^2 - 2*x + 1 < 0)) := 
sorry

end negation_proposition_l226_226588


namespace arithmetic_mean_is_12_l226_226938

/-- The arithmetic mean of the numbers 3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, and 7 is equal to 12 -/
theorem arithmetic_mean_is_12 : 
  let numbers := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]
  let sum := numbers.foldl (· + ·) 0
  let count := numbers.length
  (sum / count) = 12 :=
by
  sorry

end arithmetic_mean_is_12_l226_226938


namespace math_problem_l226_226113

theorem math_problem (x y : ℝ) (h1 : x + Real.sin y = 2023) (h2 : x + 2023 * Real.cos y = 2022) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2022 + Real.pi / 2 :=
sorry

end math_problem_l226_226113


namespace xy_nonzero_implies_iff_l226_226273

variable {x y : ℝ}

theorem xy_nonzero_implies_iff (h : x * y ≠ 0) : (x + y = 0) ↔ (x / y + y / x = -2) :=
sorry

end xy_nonzero_implies_iff_l226_226273


namespace green_apples_count_l226_226161

-- Definitions for the conditions in the problem
def total_apples : ℕ := 19
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

-- Statement expressing that the number of green apples on the table is 2
theorem green_apples_count : (total_apples - red_apples - yellow_apples = 2) :=
by
  sorry

end green_apples_count_l226_226161


namespace apples_first_year_l226_226349

theorem apples_first_year (A : ℕ) 
  (second_year_prod : ℕ := 2 * A + 8)
  (third_year_prod : ℕ := 3 * (2 * A + 8) / 4)
  (total_prod : ℕ := A + second_year_prod + third_year_prod) :
  total_prod = 194 → A = 40 :=
by
  sorry

end apples_first_year_l226_226349


namespace axis_of_symmetry_l226_226263

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem axis_of_symmetry (φ : ℝ) (hφ : 0 ≤ φ ∧ φ ≤ 2 * Real.pi) :
  (Real.sin (2 * x + φ)) = Real.cos (2 * (x - Real.pi / 3)) →
  ∃ k : ℤ, 2 * x + φ = Real.pi / 2 + k * Real.pi → x = Real.pi / 6 :=
begin
  sorry
end

end axis_of_symmetry_l226_226263


namespace smallest_n_for_cubic_sum_inequality_l226_226650

theorem smallest_n_for_cubic_sum_inequality :
  ∃ n : ℕ, (∀ (a b c : ℕ), (a + b + c) ^ 3 ≤ n * (a ^ 3 + b ^ 3 + c ^ 3)) ∧ n = 9 :=
sorry

end smallest_n_for_cubic_sum_inequality_l226_226650


namespace quadratic_equation_with_one_variable_is_B_l226_226197

def is_quadratic_equation_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + x + 3 = 0"

theorem quadratic_equation_with_one_variable_is_B :
  is_quadratic_equation_with_one_variable "x^2 + x + 3 = 0" :=
by
  sorry

end quadratic_equation_with_one_variable_is_B_l226_226197


namespace triangle_side_length_l226_226294

theorem triangle_side_length (A B C : ℝ) (h1 : AC = Real.sqrt 2) (h2: AB = 2)
  (h3 : (Real.sqrt 3 * Real.sin A + Real.cos A) / (Real.sqrt 3 * Real.cos A - Real.sin A) = Real.tan (5 * Real.pi / 12)) :
  BC = Real.sqrt 2 := 
sorry

end triangle_side_length_l226_226294


namespace twenty_second_entry_l226_226372

-- Definition of r_9 which is the remainder left when n is divided by 9
def r_9 (n : ℕ) : ℕ := n % 9

-- Statement to prove that the 22nd entry in the ordered list of all nonnegative integers
-- that satisfy r_9(5n) ≤ 4 is 38
theorem twenty_second_entry (n : ℕ) (hn : 5 * n % 9 ≤ 4) :
  ∃ m : ℕ, m = 22 ∧ n = 38 :=
sorry

end twenty_second_entry_l226_226372


namespace zhang_hua_new_year_cards_l226_226759

theorem zhang_hua_new_year_cards (x y z : ℕ) 
  (h1 : Nat.lcm (Nat.lcm x y) z = 60)
  (h2 : Nat.gcd x y = 4)
  (h3 : Nat.gcd y z = 3) : 
  x = 4 ∨ x = 20 :=
by
  sorry

end zhang_hua_new_year_cards_l226_226759


namespace catch_up_distance_l226_226939

/-- 
  Assume that A walks at 10 km/h, starts at time 0, and B starts cycling at 20 km/h, 
  6 hours after A starts. Prove that B catches up with A 120 km from the start.
-/
theorem catch_up_distance (speed_A speed_B : ℕ) (initial_delay : ℕ) (distance : ℕ) : 
  initial_delay = 6 →
  speed_A = 10 →
  speed_B = 20 →
  distance = 120 →
  distance = speed_B * (initial_delay * speed_A / (speed_B - speed_A)) :=
by sorry

end catch_up_distance_l226_226939


namespace range_g_l226_226974

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x + (Real.cos x)^2 - 1/2

noncomputable def g (x : ℝ) : ℝ := 
  let h (x : ℝ) := (Real.sin (2 * x + Real.pi))
  h (x - (5 * Real.pi / 12))

theorem range_g :
  (Set.image g (Set.Icc (-Real.pi/12) (Real.pi/3))) = Set.Icc (-1) (1/2) :=
  sorry

end range_g_l226_226974


namespace journey_time_equality_l226_226296

variables {v : ℝ} (h : v > 0)

theorem journey_time_equality (v : ℝ) (hv : v > 0) :
  let t1 := 80 / v
  let t2 := 160 / (2 * v)
  t1 = t2 :=
by
  sorry

end journey_time_equality_l226_226296


namespace height_of_building_l226_226217

def flagpole_height : ℝ := 18
def flagpole_shadow_length : ℝ := 45

def building_shadow_length : ℝ := 65
def building_height : ℝ := 26

theorem height_of_building
  (hflagpole : flagpole_height / flagpole_shadow_length = building_height / building_shadow_length) :
  building_height = 26 :=
sorry

end height_of_building_l226_226217


namespace lcm_18_30_l226_226480

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l226_226480


namespace milk_savings_l226_226281

theorem milk_savings :
  let cost_for_two_packs : ℝ := 2.50
  let cost_per_pack_individual : ℝ := 1.30
  let num_packs_per_set := 2
  let num_sets := 10
  let cost_per_pack_set := cost_for_two_packs / num_packs_per_set
  let savings_per_pack := cost_per_pack_individual - cost_per_pack_set
  let total_packs := num_sets * num_packs_per_set
  let total_savings := savings_per_pack * total_packs
  total_savings = 1 :=
by
  sorry

end milk_savings_l226_226281


namespace base_three_to_base_ten_l226_226085

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end base_three_to_base_ten_l226_226085


namespace smaller_prime_is_x_l226_226176

theorem smaller_prime_is_x (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x = 17 :=
  sorry

end smaller_prime_is_x_l226_226176


namespace part1_part2_l226_226765

-- Part 1: Positive integers with leading digit 6 that become 1/25 of the original number when the leading digit is removed.
theorem part1 (n : ℕ) (m : ℕ) (h1 : m = 6 * 10^n + m) (h2 : m = (6 * 10^n + m) / 25) :
  m = 625 * 10^(n - 2) ∨
  m = 625 * 10^(n - 2 + 1) ∨
  ∃ k : ℕ, m = 625 * 10^(n - 2 + k) :=
sorry

-- Part 2: No positive integer exists which becomes 1/35 of the original number when its leading digit is removed.
theorem part2 (n : ℕ) (m : ℕ) (h : m = 6 * 10^n + m) :
  m ≠ (6 * 10^n + m) / 35 :=
sorry

end part1_part2_l226_226765


namespace number_of_units_sold_l226_226985

theorem number_of_units_sold (p : ℕ) (c : ℕ) (k : ℕ) (h : p * c = k) (h₁ : c = 800) (h₂ : k = 8000) : p = 10 :=
by
  sorry

end number_of_units_sold_l226_226985


namespace sqrt_expression_l226_226271

theorem sqrt_expression (h : n < m ∧ m < 0) : 
  (Real.sqrt (m^2 + 2 * m * n + n^2) - Real.sqrt (m^2 - 2 * m * n + n^2)) = -2 * m := 
by {
  sorry
}

end sqrt_expression_l226_226271


namespace average_goal_l226_226876

-- Define the list of initial rolls
def initial_rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

-- Define the next roll
def next_roll : ℕ := 2

-- Define the goal for the average
def goal_average : ℕ := 3

-- The theorem to prove that Ronald's goal for the average of all his rolls is 3
theorem average_goal : (List.sum (initial_rolls ++ [next_roll]) / (List.length (initial_rolls ++ [next_roll]))) = goal_average :=
by
  -- The proof will be provided later
  sorry

end average_goal_l226_226876


namespace total_plate_combinations_l226_226828

open Nat

def valid_letters := 24
def letter_positions := (choose 4 2)
def valid_digits := 10
def total_combinations := letter_positions * (valid_letters * valid_letters) * (valid_digits ^ 3)

theorem total_plate_combinations : total_combinations = 3456000 :=
  by
    -- Replace this sorry with steps to prove the theorem
    sorry

end total_plate_combinations_l226_226828


namespace interval_between_segments_l226_226832

def population_size : ℕ := 800
def sample_size : ℕ := 40

theorem interval_between_segments : population_size / sample_size = 20 :=
by
  -- Insert proof here
  sorry

end interval_between_segments_l226_226832


namespace pascal_50th_number_in_52_row_l226_226792

theorem pascal_50th_number_in_52_row : nat.binomial 51 2 = 1275 :=
by
  sorry

end pascal_50th_number_in_52_row_l226_226792


namespace milkshake_hours_l226_226351

theorem milkshake_hours (h : ℕ) : 
  (3 * h + 7 * h = 80) → h = 8 := 
by
  intro h_milkshake_eq
  sorry

end milkshake_hours_l226_226351


namespace votes_ratio_l226_226194

theorem votes_ratio (V : ℝ) 
  (counted_fraction : ℝ := 2/9) 
  (favor_fraction : ℝ := 3/4) 
  (against_fraction_remaining : ℝ := 0.7857142857142856) :
  let counted := counted_fraction * V
  let favor_counted := favor_fraction * counted
  let remaining := V - counted
  let against_remaining := against_fraction_remaining * remaining
  let against_counted := (1 - favor_fraction) * counted
  let total_against := against_counted + against_remaining
  let total_favor := favor_counted
  (total_against / total_favor) = 4 :=
by
  sorry

end votes_ratio_l226_226194


namespace lcm_18_30_is_90_l226_226484

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l226_226484


namespace expression_behavior_l226_226668

theorem expression_behavior (x : ℝ) (h1 : -3 < x) (h2 : x < 2) :
  ¬∃ m, ∀ y : ℝ, (h3 : -3 < y) → (h4 : y < 2) → (x ≠ 1) → (y ≠ 1) → 
    (m <= (y^2 - 3*y + 3) / (y - 1)) ∧ 
    (m >= (y^2 - 3*y + 3) / (y - 1)) :=
sorry

end expression_behavior_l226_226668


namespace calculate_120ab_l226_226617

variable (a b : ℚ)

theorem calculate_120ab (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * (a * b) = 800 := by
  sorry

end calculate_120ab_l226_226617


namespace additional_amount_per_10_cents_l226_226497

-- Definitions of the given conditions
def expected_earnings_per_share : ℝ := 0.80
def dividend_ratio : ℝ := 0.5
def actual_earnings_per_share : ℝ := 1.10
def shares_owned : ℕ := 600
def total_dividend_paid : ℝ := 312

-- Proof statement
theorem additional_amount_per_10_cents (additional_amount : ℝ) :
  (total_dividend_paid - (shares_owned * (expected_earnings_per_share * dividend_ratio))) / shares_owned / 
  ((actual_earnings_per_share - expected_earnings_per_share) / 0.10) = additional_amount :=
sorry

end additional_amount_per_10_cents_l226_226497


namespace solution_proof_l226_226946

noncomputable def problem_statement : Prop :=
  ((16^(1/4) * 32^(1/5)) + 64^(1/6)) = 6

theorem solution_proof : problem_statement :=
by
  sorry

end solution_proof_l226_226946


namespace alice_daily_savings_l226_226786

theorem alice_daily_savings :
  ∀ (d total_days : ℕ) (dime_value : ℝ),
  d = 4 → total_days = 40 → dime_value = 0.10 →
  (d * dime_value) / total_days = 0.01 :=
by
  intros d total_days dime_value h_d h_total_days h_dime_value
  sorry

end alice_daily_savings_l226_226786


namespace scientific_notation_l226_226643

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l226_226643


namespace solve_expression_l226_226624

noncomputable def expression : ℝ := 5 * 1.6 - 2 * 1.4 / 1.3

theorem solve_expression : expression = 5.8462 := 
by 
  sorry

end solve_expression_l226_226624


namespace triangular_prism_skew_pair_count_l226_226784

-- Definition of a triangular prism with 6 vertices and 15 lines through any two vertices
structure TriangularPrism :=
  (vertices : Fin 6)   -- 6 vertices
  (lines : Fin 15)     -- 15 lines through any two vertices

-- A function to check if two lines are skew lines 
-- (not intersecting and not parallel in three-dimensional space)
def is_skew (line1 line2 : Fin 15) : Prop := sorry

-- Function to count pairs of lines that are skew in a triangular prism
def count_skew_pairs (prism : TriangularPrism) : Nat := sorry

-- Theorem stating the number of skew pairs in a triangular prism is 36
theorem triangular_prism_skew_pair_count (prism : TriangularPrism) :
  count_skew_pairs prism = 36 := 
sorry

end triangular_prism_skew_pair_count_l226_226784


namespace people_to_right_of_taehyung_l226_226883

-- Given conditions
def total_people : Nat := 11
def people_to_left_of_taehyung : Nat := 5

-- Question and proof: How many people are standing to Taehyung's right?
theorem people_to_right_of_taehyung : total_people - people_to_left_of_taehyung - 1 = 4 :=
by
  sorry

end people_to_right_of_taehyung_l226_226883


namespace jumps_per_second_l226_226718

-- Define the conditions and known values
def record_jumps : ℕ := 54000
def hours : ℕ := 5
def seconds_per_hour : ℕ := 3600

-- Define the target question as a theorem to prove
theorem jumps_per_second :
  (record_jumps / (hours * seconds_per_hour)) = 3 := by
  sorry

end jumps_per_second_l226_226718


namespace simplify_complex_expression_l226_226447

open Complex

theorem simplify_complex_expression :
  let a := (4 : ℂ) + 6 * I
  let b := (4 : ℂ) - 6 * I
  ((a / b) - (b / a) = (24 * I) / 13) := by
  sorry

end simplify_complex_expression_l226_226447


namespace sum_of_number_and_reverse_l226_226888

theorem sum_of_number_and_reverse (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
(h : (10 * a + b) - (10 * b + a) = 3 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 33 := 
sorry

end sum_of_number_and_reverse_l226_226888


namespace netCaloriesConsumedIs1082_l226_226048

-- Given conditions
def caloriesPerCandyBar : ℕ := 347
def candyBarsEatenInAWeek : ℕ := 6
def caloriesBurnedInAWeek : ℕ := 1000

-- Net calories calculation
def netCaloriesInAWeek (calsPerBar : ℕ) (barsPerWeek : ℕ) (calsBurned : ℕ) : ℕ :=
  calsPerBar * barsPerWeek - calsBurned

-- The theorem to prove
theorem netCaloriesConsumedIs1082 :
  netCaloriesInAWeek caloriesPerCandyBar candyBarsEatenInAWeek caloriesBurnedInAWeek = 1082 :=
by
  sorry

end netCaloriesConsumedIs1082_l226_226048


namespace additional_profit_is_80000_l226_226586

-- Define the construction cost of a regular house
def construction_cost_regular (C : ℝ) : ℝ := C

-- Define the construction cost of the special house
def construction_cost_special (C : ℝ) : ℝ := C + 200000

-- Define the selling price of a regular house
def selling_price_regular : ℝ := 350000

-- Define the selling price of the special house
def selling_price_special : ℝ := 1.8 * 350000

-- Define the profit from selling a regular house
def profit_regular (C : ℝ) : ℝ := selling_price_regular - (construction_cost_regular C)

-- Define the profit from selling the special house
def profit_special (C : ℝ) : ℝ := selling_price_special - (construction_cost_special C)

-- Define the additional profit made by building and selling the special house compared to a regular house
def additional_profit (C : ℝ) : ℝ := (profit_special C) - (profit_regular C)

-- Theorem to prove the additional profit is $80,000
theorem additional_profit_is_80000 (C : ℝ) : additional_profit C = 80000 :=
sorry

end additional_profit_is_80000_l226_226586


namespace log_addition_property_l226_226540

noncomputable def logFunction (x : ℝ) : ℝ := Real.log x

theorem log_addition_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : logFunction (a * b) = 1) :
  logFunction (a^2) + logFunction (b^2) = 2 :=
by
  sorry

end log_addition_property_l226_226540


namespace translated_point_is_correct_l226_226702

-- Cartesian Point definition
structure Point where
  x : Int
  y : Int

-- Define the translation function
def translate (p : Point) (dx dy : Int) : Point :=
  Point.mk (p.x + dx) (p.y - dy)

-- Define the initial point A and the translation amounts
def A : Point := ⟨-3, 2⟩
def dx : Int := 3
def dy : Int := 2

-- The proof goal
theorem translated_point_is_correct :
  translate A dx dy = ⟨0, 0⟩ :=
by
  -- This is where the proof would normally go
  sorry

end translated_point_is_correct_l226_226702


namespace target_hit_probability_l226_226162

-- Define the probabilities of Person A and Person B hitting the target
def prob_A_hits := 0.8
def prob_B_hits := 0.7

-- Define the probability that the target is hit when both shoot independently at the same time
def prob_target_hit := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

theorem target_hit_probability : prob_target_hit = 0.94 := 
by
  sorry

end target_hit_probability_l226_226162


namespace tower_surface_area_l226_226363

noncomputable def total_visible_surface_area (volumes : List ℕ) : ℕ := sorry

theorem tower_surface_area :
  total_visible_surface_area [512, 343, 216, 125, 64, 27, 8, 1] = 882 :=
sorry

end tower_surface_area_l226_226363


namespace remainder_numGreenRedModal_l226_226545

def numGreenMarbles := 7
def numRedMarbles (n : ℕ) := 7 + n
def validArrangement (g r : ℕ) := (g + r = numGreenMarbles + numRedMarbles r) ∧ 
  (g = r)

theorem remainder_numGreenRedModal (N' : ℕ) :
  N' % 1000 = 432 :=
sorry

end remainder_numGreenRedModal_l226_226545


namespace books_combination_l226_226551

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l226_226551


namespace flour_needed_l226_226222

theorem flour_needed (flour_per_24_cookies : ℝ) (cookies_per_recipe : ℕ) (desired_cookies : ℕ) 
  (h : flour_per_24_cookies = 1.5) (h1 : cookies_per_recipe = 24) (h2 : desired_cookies = 72) : 
  flour_per_24_cookies / cookies_per_recipe * desired_cookies = 4.5 := 
  by {
    -- The proof is omitted
    sorry
  }

end flour_needed_l226_226222


namespace original_students_count_l226_226983

theorem original_students_count (N : ℕ) (T : ℕ) :
  (T = N * 85) →
  ((N - 5) * 90 = T - 300) →
  ((N - 8) * 95 = T - 465) →
  ((N - 15) * 100 = T - 955) →
  N = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end original_students_count_l226_226983


namespace triangle_ABC_two_solutions_l226_226032

theorem triangle_ABC_two_solutions (x : ℝ) (h1 : x > 0) : 
  2 < x ∧ x < 2 * Real.sqrt 2 ↔
  (∃ a b B, a = x ∧ b = 2 ∧ B = Real.pi / 4 ∧ a * Real.sin B < b ∧ b < a) := by
  sorry

end triangle_ABC_two_solutions_l226_226032


namespace coffee_shop_sales_l226_226509

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l226_226509


namespace part_I_part_II_l226_226818

noncomputable theory

open Real

section
variable (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 / a) * log x - x + 1 / x

theorem part_I (h : 0 < a) :
  (f' x = (a + 1 / a) / x - 1 - 1 / x^2) = 0 → x = (a + 1 / a + sqrt ((a + 1 / a)^2 - 4)) / 2 → 
  (a ≥ 1) :=
sorry

theorem part_II (a : ℝ) 
  (h1 : 1 < a) (h2 : a ≤ exp 1) (x₁ x₂ : ℝ) 
  (h₃ : 0 < x₁ ∧ x₁ < 1) (h₄ : 1 < x₂) :
  ∃ Mₐ, Mₐ = f a x₂ - f a x₁ ∧ (Mₐ = f e x₂ - f e x₁) :=
sorry
end

end part_I_part_II_l226_226818


namespace lauren_change_l226_226152

theorem lauren_change :
  let meat_cost      := 2 * 3.50
  let buns_cost      := 1.50
  let lettuce_cost   := 1.00
  let tomato_cost    := 1.5 * 2.00
  let pickles_cost   := 2.50 - 1.00
  let total_cost     := meat_cost + buns_cost + lettuce_cost + tomato_cost + pickles_cost
  let payment        := 20.00
  let change         := payment - total_cost
  change = 6.00 :=
by
  unfold meat_cost buns_cost lettuce_cost tomato_cost pickles_cost total_cost payment change
  -- Prove the main statement.
  sorry

end lauren_change_l226_226152


namespace original_price_of_cycle_l226_226933

theorem original_price_of_cycle (P : ℝ) (h1 : 1440 = P + 0.6 * P) : P = 900 :=
by
  sorry

end original_price_of_cycle_l226_226933


namespace division_remainder_l226_226758

def remainder (x y : ℕ) : ℕ := x % y

theorem division_remainder (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : (x : ℚ) / y = 96.15) (h4 : y = 20) : remainder x y = 3 :=
by
  sorry

end division_remainder_l226_226758


namespace race_head_start_l226_226204

theorem race_head_start (Va Vb L H : ℚ) (h : Va = 30 / 17 * Vb) :
  H = 13 / 30 * L :=
by
  sorry

end race_head_start_l226_226204


namespace smallest_y_condition_l226_226611

theorem smallest_y_condition : ∃ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 ∧ y = 167 :=
by 
  sorry

end smallest_y_condition_l226_226611


namespace problem1_problem2_l226_226378

variables {a b c : ℝ}

-- (1) Prove that a + b + c = 4 given the conditions
theorem problem1 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_min : ∀ x, abs (x + a) + abs (x - b) + c ≥ 4) : a + b + c = 4 := 
sorry

-- (2) Prove that the minimum value of (1/4)a^2 + (1/9)b^2 + c^2 is 8/7 given the conditions and that a + b + c = 4
theorem problem2 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8 / 7 := 
sorry

end problem1_problem2_l226_226378


namespace minimum_value_of_c_l226_226134

open Real

noncomputable
def question (a b c : ℝ) (log_base2 : ℝ → ℝ) :=
  2^a + 4^b = 2^c ∧ 4^a + 2^b = 4^c

noncomputable
def answer (c : ℝ) (log_base2 : ℝ → ℝ) :=
  c = log_base2 3 - 5 / 3

theorem minimum_value_of_c (a b c : ℝ) (log_base2 : ℝ → ℝ)
  (h : question a b c log_base2) : answer c log_base2 :=
sorry

end minimum_value_of_c_l226_226134


namespace coach_class_seats_l226_226942

variable (F C : ℕ)

-- Define the conditions
def totalSeats := F + C = 387
def coachSeats := C = 4 * F + 2

-- State the theorem
theorem coach_class_seats : totalSeats F C → coachSeats F C → C = 310 :=
by sorry

end coach_class_seats_l226_226942


namespace compute_c_plus_d_l226_226961

theorem compute_c_plus_d (c d : ℕ) (h1 : d = c^3) (h2 : d - c = 435) : c + d = 520 :=
sorry

end compute_c_plus_d_l226_226961


namespace find_f_l226_226403

variable (f : ℝ → ℝ)

open Function

theorem find_f (h : ∀ x: ℝ, f (3 * x + 2) = 9 * x + 8) : ∀ x: ℝ, f x = 3 * x + 2 := 
sorry

end find_f_l226_226403


namespace blue_balls_needed_l226_226720

theorem blue_balls_needed 
  (G B Y W : ℝ)
  (h1 : G = 2 * B)
  (h2 : Y = (8 / 3) * B)
  (h3 : W = (4 / 3) * B) :
  5 * G + 3 * Y + 4 * W = (70 / 3) * B :=
by
  sorry

end blue_balls_needed_l226_226720


namespace A_inter_B_A_subset_C_l226_226375

namespace MathProof

def A := {x : ℝ | x^2 - 6*x + 8 ≤ 0 }
def B := {x : ℝ | (x - 1)/(x - 3) ≥ 0 }
def C (a : ℝ) := {x : ℝ | x^2 - (2*a + 4)*x + a^2 + 4*a ≤ 0 }

theorem A_inter_B : (A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 4} := sorry

theorem A_subset_C (a : ℝ) : (A ⊆ C a) ↔ (0 ≤ a ∧ a ≤ 2) := sorry

end MathProof

end A_inter_B_A_subset_C_l226_226375


namespace max_levels_passed_prob_first_three_levels_l226_226693

-- Assuming X is a random variable representing the outcome of a single fair die roll
variable (X : ℕ → ℕ)

-- Definition for passing level k
def pass_level (k : ℕ) : Prop :=
  (finset.sum (finset.range k) (λ i, X i)) > 2^k

-- Maximum number of levels passed theorem
theorem max_levels_passed : ∃ (N : ℕ), ∀ (k : ℕ), (k ≤ 4 ↔ pass_level X k) := sorry

-- Probability of passing the first three levels consecutively
theorem prob_first_three_levels : 
  ∃ p : ℚ, p = (2 / 3) * (5 / 6) * (20 / 27) ∧ p = 100 / 243 := sorry

end max_levels_passed_prob_first_three_levels_l226_226693


namespace g_half_equals_four_l226_226120

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(2 - x) - 3 / 4

def A : ℝ × ℝ :=
  (2, 1 / 4)

noncomputable def g (α : ℝ) (x : ℝ) : ℝ :=
  x^α

theorem g_half_equals_four (a : ℝ) (α : ℝ)
  (ha : a > 0) (ha' : a ≠ 1) (hA1 : f a 2 = 1/4) (hA2 : g α 2 = 1/4) :
  g α (1 / 2) = 4 :=
by
  sorry

end g_half_equals_four_l226_226120


namespace paul_taxes_and_fees_l226_226442

theorem paul_taxes_and_fees 
  (hourly_wage: ℝ) 
  (hours_worked : ℕ)
  (spent_on_gummy_bears_percentage : ℝ)
  (final_amount : ℝ)
  (gross_earnings := hourly_wage * hours_worked)
  (taxes_and_fees := gross_earnings - final_amount / (1 - spent_on_gummy_bears_percentage)):
  hourly_wage = 12.50 →
  hours_worked = 40 →
  spent_on_gummy_bears_percentage = 0.15 →
  final_amount = 340 →
  taxes_and_fees / gross_earnings = 0.20 :=
by
  intros
  sorry

end paul_taxes_and_fees_l226_226442


namespace total_books_l226_226627

def initial_books : ℝ := 41.0
def first_addition : ℝ := 33.0
def second_addition : ℝ := 2.0

theorem total_books (h1 : initial_books = 41.0) (h2 : first_addition = 33.0) (h3 : second_addition = 2.0) :
  initial_books + first_addition + second_addition = 76.0 := 
by
  -- placeholders for the proof steps, omitting the detailed steps as instructed
  sorry

end total_books_l226_226627


namespace woman_born_1892_l226_226502

theorem woman_born_1892 (y : ℕ) (hy : 1850 ≤ y^2 - y ∧ y^2 - y < 1900) : y = 44 :=
by
  sorry

end woman_born_1892_l226_226502


namespace manny_remaining_money_l226_226751

def cost_chair (cost_total_chairs : ℕ) (number_of_chairs : ℕ) : ℕ :=
  cost_total_chairs / number_of_chairs

def cost_table (cost_chair : ℕ) (chairs_for_table : ℕ) : ℕ :=
  cost_chair * chairs_for_table

def total_cost (cost_table : ℕ) (cost_chairs : ℕ) : ℕ :=
  cost_table + cost_chairs

def remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem manny_remaining_money : remaining_money 100 (total_cost (cost_table (cost_chair 55 5) 3) ((cost_chair 55 5) * 2)) = 45 :=
by
  sorry

end manny_remaining_money_l226_226751


namespace correct_division_result_l226_226984

-- Define the conditions
def incorrect_divisor : ℕ := 48
def correct_divisor : ℕ := 36
def incorrect_quotient : ℕ := 24
def dividend : ℕ := incorrect_divisor * incorrect_quotient

-- Theorem statement
theorem correct_division_result : (dividend / correct_divisor) = 32 := by
  -- proof to be filled later
  sorry

end correct_division_result_l226_226984


namespace division_problem_l226_226757

theorem division_problem (x y n : ℕ) 
  (h1 : x = n * y + 4) 
  (h2 : 2 * x = 14 * y + 1) 
  (h3 : 5 * y - x = 3) : n = 4 := 
sorry

end division_problem_l226_226757


namespace combination_15_choose_3_l226_226549

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l226_226549


namespace neil_total_charge_l226_226024

theorem neil_total_charge 
  (trim_cost : ℕ) (shape_cost : ℕ) (total_boxwoods : ℕ) (shaped_boxwoods : ℕ) : 
  trim_cost = 5 → shape_cost = 15 → total_boxwoods = 30 → shaped_boxwoods = 4 → 
  trim_cost * total_boxwoods + shape_cost * shaped_boxwoods = 210 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end neil_total_charge_l226_226024


namespace probability_at_least_two_consecutive_heads_l226_226216

theorem probability_at_least_two_consecutive_heads :
  let Ω := (Fin 2 → Fin 2 → ℝ) in
  let P : MeasureTheory.Measure Ω := MeasureTheory.ProbabilityMeasure.uniform (MeasureTheory.fintypeFiniteOf Ω) in
  let event := {ω : Ω | ∃ i : Fin 4, ω i = 1 ∧ ω (i+1) = 1} in
  P event = 9/16 :=
begin
  sorry
end

end probability_at_least_two_consecutive_heads_l226_226216


namespace water_fraction_final_l226_226625

noncomputable def initial_water_volume : ℚ := 25
noncomputable def first_removal_water : ℚ := 5
noncomputable def first_add_antifreeze : ℚ := 5
noncomputable def first_water_fraction : ℚ := (initial_water_volume - first_removal_water) / initial_water_volume

noncomputable def second_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def second_water_fraction : ℚ := (initial_water_volume - first_removal_water - second_removal_fraction * (initial_water_volume - first_removal_water)) / initial_water_volume

noncomputable def third_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def third_water_fraction := (second_water_fraction * (initial_water_volume - 5) + 2) / initial_water_volume

theorem water_fraction_final :
  third_water_fraction = 14.8 / 25 := sorry

end water_fraction_final_l226_226625


namespace geometric_seq_value_l226_226412

variable (a : ℕ → ℝ)
variable (g : ∀ n m : ℕ, a n * a m = a ((n + m) / 2) ^ 2)

theorem geometric_seq_value (h1 : a 2 = 1 / 3) (h2 : a 8 = 27) : a 5 = 3 ∨ a 5 = -3 := by
  sorry

end geometric_seq_value_l226_226412


namespace part1_solution_set_eq_part2_a_range_l226_226121

theorem part1_solution_set_eq : {x : ℝ | |2 * x + 1| + |2 * x - 3| ≤ 6} = Set.Icc (-1) 2 :=
by sorry

theorem part2_a_range (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, |2 * x + 1| + |2 * x - 3| < |a - 2|) → 6 < a :=
by sorry

end part1_solution_set_eq_part2_a_range_l226_226121


namespace distance_interval_l226_226407

theorem distance_interval (d : ℝ) (h1 : ¬(d ≥ 8)) (h2 : ¬(d ≤ 7)) (h3 : ¬(d ≤ 6 → north)):
  7 < d ∧ d < 8 :=
by
  have h_d8 : d < 8 := by linarith
  have h_d7 : d > 7 := by linarith
  exact ⟨h_d7, h_d8⟩

end distance_interval_l226_226407
