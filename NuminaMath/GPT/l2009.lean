import Mathlib

namespace pizza_topping_cost_l2009_200938

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

end pizza_topping_cost_l2009_200938


namespace distance_M_to_AB_l2009_200965

noncomputable def distance_to_ab : ℝ := 5.8

theorem distance_M_to_AB
  (M : Point)
  (A B C : Point)
  (d_AC d_BC : ℝ)
  (AB BC AC : ℝ)
  (H1 : d_AC = 2)
  (H2 : d_BC = 4)
  (H3 : AB = 10)
  (H4 : BC = 17)
  (H5 : AC = 21) :
  distance_to_ab = 5.8 :=
by
  sorry

end distance_M_to_AB_l2009_200965


namespace encryption_of_hope_is_correct_l2009_200977

def shift_letter (c : Char) : Char :=
  if 'a' ≤ c ∧ c ≤ 'z' then
    Char.ofNat ((c.toNat - 'a'.toNat + 4) % 26 + 'a'.toNat)
  else 
    c

def encrypt (s : String) : String :=
  s.map shift_letter

theorem encryption_of_hope_is_correct : encrypt "hope" = "lsti" :=
by
  sorry

end encryption_of_hope_is_correct_l2009_200977


namespace largest_among_four_theorem_l2009_200937

noncomputable def largest_among_four (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) : Prop :=
  (a^2 + b^2 > 1) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a^2 + b^2 > a)

theorem largest_among_four_theorem (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) :
  largest_among_four a b h1 h2 :=
sorry

end largest_among_four_theorem_l2009_200937


namespace DF_is_5_point_5_l2009_200995

variables {A B C D E F : Type}
variables (congruent : triangle A B C ≃ triangle D E F)
variables (ac_length : AC = 5.5)

theorem DF_is_5_point_5 : DF = 5.5 :=
by
  -- skipped proof
  sorry

end DF_is_5_point_5_l2009_200995


namespace ratio_adidas_skechers_l2009_200940

-- Conditions
def total_expenditure : ℤ := 8000
def expenditure_adidas : ℤ := 600
def expenditure_clothes : ℤ := 2600
def expenditure_nike := 3 * expenditure_adidas

-- Calculation for sneakers
def total_sneakers := total_expenditure - expenditure_clothes
def expenditure_nike_adidas := expenditure_nike + expenditure_adidas
def expenditure_skechers := total_sneakers - expenditure_nike_adidas

-- Prove the ratio
theorem ratio_adidas_skechers (H1 : total_expenditure = 8000)
                              (H2 : expenditure_adidas = 600)
                              (H3 : expenditure_nike = 3 * expenditure_adidas)
                              (H4 : expenditure_clothes = 2600) :
  expenditure_adidas / expenditure_skechers = 1 / 5 :=
by
  sorry

end ratio_adidas_skechers_l2009_200940


namespace inequality_holds_l2009_200994

theorem inequality_holds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a + (1 / b))^2 + (b + (1 / c))^2 + (c + (1 / a))^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end inequality_holds_l2009_200994


namespace magician_card_pairs_l2009_200917

theorem magician_card_pairs:
  ∃ (f : Fin 65 → Fin 65 × Fin 65), 
  (∀ m n : Fin 65, ∃ k l : Fin 65, (f m = (k, l) ∧ f n = (l, k))) := 
sorry

end magician_card_pairs_l2009_200917


namespace total_blankets_collected_l2009_200944

theorem total_blankets_collected : 
  let original_members := 15
  let new_members := 5
  let blankets_per_original_member_first_day := 2
  let blankets_per_original_member_second_day := 2
  let blankets_per_new_member_second_day := 4
  let tripled_first_day_total := 3
  let blankets_school_third_day := 22
  let blankets_online_third_day := 30
  let first_day_blankets := original_members * blankets_per_original_member_first_day
  let second_day_original_members_blankets := original_members * blankets_per_original_member_second_day
  let second_day_new_members_blankets := new_members * blankets_per_new_member_second_day
  let second_day_additional_blankets := tripled_first_day_total * first_day_blankets
  let second_day_blankets := second_day_original_members_blankets + second_day_new_members_blankets + second_day_additional_blankets
  let third_day_blankets := blankets_school_third_day + blankets_online_third_day
  let total_blankets := first_day_blankets + second_day_blankets + third_day_blankets
  -- Prove that
  total_blankets = 222 :=
by 
  sorry

end total_blankets_collected_l2009_200944


namespace ticket_price_divisor_l2009_200958

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

def GCD (a b : ℕ) := Nat.gcd a b

theorem ticket_price_divisor :
  let total7 := 70
  let total8 := 98
  let y := 4
  is_divisor (GCD total7 total8) y :=
by
  sorry

end ticket_price_divisor_l2009_200958


namespace find_angle_F_l2009_200939

-- Define the angles of the triangle
variables (D E F : ℝ)

-- Define the conditions given in the problem
def angle_conditions (D E F : ℝ) : Prop :=
  (D = 3 * E) ∧ (E = 18) ∧ (D + E + F = 180)

-- The theorem to prove that angle F is 108 degrees
theorem find_angle_F (D E F : ℝ) (h : angle_conditions D E F) : 
  F = 108 :=
by
  -- The proof body is omitted
  sorry

end find_angle_F_l2009_200939


namespace costPerUse_l2009_200922

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end costPerUse_l2009_200922


namespace teams_equation_l2009_200904

theorem teams_equation (x : ℕ) (h1 : 100 = x + 4*x - 10) : 4 * x + x - 10 = 100 :=
by
  sorry

end teams_equation_l2009_200904


namespace probability_at_least_one_unqualified_l2009_200982

theorem probability_at_least_one_unqualified :
  let total_products := 6
  let qualified_products := 4
  let unqualified_products := 2
  let products_selected := 2
  (1 - (Nat.choose qualified_products 2 / Nat.choose total_products 2)) = 3/5 :=
by
  sorry

end probability_at_least_one_unqualified_l2009_200982


namespace sum_series_l2009_200942

noncomputable def b : ℕ → ℝ
| 0     => 2
| 1     => 2
| (n+2) => b (n+1) + b n

theorem sum_series : (∑' n, b n / 3^(n+1)) = 1 / 3 := by
  sorry

end sum_series_l2009_200942


namespace largest_number_formed_l2009_200970

-- Define the digits
def digit1 : ℕ := 2
def digit2 : ℕ := 6
def digit3 : ℕ := 9

-- Define the function to form the largest number using the given digits
def largest_three_digit_number (a b c : ℕ) : ℕ :=
  if a > b ∧ a > c then
    if b > c then 100 * a + 10 * b + c
    else 100 * a + 10 * c + b
  else if b > a ∧ b > c then
    if a > c then 100 * b + 10 * a + c
    else 100 * b + 10 * c + a
  else
    if a > b then 100 * c + 10 * a + b
    else 100 * c + 10 * b + a

-- Statement that this function correctly computes the largest number
theorem largest_number_formed :
  largest_three_digit_number digit1 digit2 digit3 = 962 :=
by
  sorry

end largest_number_formed_l2009_200970


namespace power_seven_evaluation_l2009_200948

theorem power_seven_evaluation (a b : ℝ) (h : a = (7 : ℝ)^(1/4) ∧ b = (7 : ℝ)^(1/7)) : 
  a / b = (7 : ℝ)^(3/28) :=
  sorry

end power_seven_evaluation_l2009_200948


namespace product_identity_l2009_200980

theorem product_identity (x y : ℝ) : (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l2009_200980


namespace division_multiplication_expression_l2009_200902

theorem division_multiplication_expression : 377 / 13 / 29 * 1 / 4 / 2 = 0.125 :=
by
  sorry

end division_multiplication_expression_l2009_200902


namespace problem_statement_l2009_200920

theorem problem_statement (x θ : ℝ) (h : Real.logb 2 x + Real.cos θ = 2) : |x - 8| + |x + 2| = 10 :=
sorry

end problem_statement_l2009_200920


namespace exists_t_perpendicular_min_dot_product_coordinates_l2009_200926

-- Definitions of points
def OA : ℝ × ℝ := (5, 1)
def OB : ℝ × ℝ := (1, 7)
def OC : ℝ × ℝ := (4, 2)

-- Definition of vector OM depending on t
def OM (t : ℝ) : ℝ × ℝ := (4 * t, 2 * t)

-- Definition of vector MA and MB
def MA (t : ℝ) : ℝ × ℝ := (5 - 4 * t, 1 - 2 * t)
def MB (t : ℝ) : ℝ × ℝ := (1 - 4 * t, 7 - 2 * t)

-- Dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Proof that there exists a t such that MA ⊥ MB
theorem exists_t_perpendicular : ∃ t : ℝ, dot_product (MA t) (MB t) = 0 :=
by 
  sorry

-- Proof that coordinates of M minimizing MA ⋅ MB is (4, 2)
theorem min_dot_product_coordinates : ∃ t : ℝ, t = 1 ∧ (OM t) = (4, 2) :=
by
  sorry

end exists_t_perpendicular_min_dot_product_coordinates_l2009_200926


namespace ratio_of_money_with_Ram_and_Gopal_l2009_200966

noncomputable section

variable (R K G : ℕ)

theorem ratio_of_money_with_Ram_and_Gopal 
  (hR : R = 735) 
  (hK : K = 4335) 
  (hRatio : G * 17 = 7 * K) 
  (hGCD : Nat.gcd 735 1785 = 105) :
  R * 17 = 7 * G := 
by
  sorry

end ratio_of_money_with_Ram_and_Gopal_l2009_200966


namespace number_of_ordered_triples_l2009_200907

/-- 
Prove the number of ordered triples (x, y, z) of positive integers that satisfy 
  lcm(x, y) = 180, lcm(x, z) = 210, and lcm(y, z) = 420 is 2.
-/
theorem number_of_ordered_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h₁ : Nat.lcm x y = 180) (h₂ : Nat.lcm x z = 210) (h₃ : Nat.lcm y z = 420) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end number_of_ordered_triples_l2009_200907


namespace hexagon_perimeter_arithmetic_sequence_l2009_200989

theorem hexagon_perimeter_arithmetic_sequence :
  let a₁ := 10
  let a₂ := 12
  let a₃ := 14
  let a₄ := 16
  let a₅ := 18
  let a₆ := 20
  let lengths := [a₁, a₂, a₃, a₄, a₅, a₆]
  let perimeter := lengths.sum
  perimeter = 90 :=
by
  sorry

end hexagon_perimeter_arithmetic_sequence_l2009_200989


namespace rita_coffee_cost_l2009_200906

noncomputable def costPerPound (initialAmount spentAmount pounds : ℝ) : ℝ :=
  spentAmount / pounds

theorem rita_coffee_cost :
  ∀ (initialAmount remainingAmount pounds : ℝ),
    initialAmount = 70 ∧ remainingAmount = 35.68 ∧ pounds = 4 →
    costPerPound initialAmount (initialAmount - remainingAmount) pounds = 8.58 :=
by
  intros initialAmount remainingAmount pounds h
  simp [costPerPound, h]
  sorry

end rita_coffee_cost_l2009_200906


namespace geometric_sequence_sum_is_120_l2009_200911

noncomputable def sum_first_four_geometric_seq (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4

theorem geometric_sequence_sum_is_120 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_pos_geometric : 0 < q ∧ q < 1)
  (h_a3_a5 : a 3 + a 5 = 20)
  (h_a3_a5_product : a 3 * a 5 = 64) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) :
  sum_first_four_geometric_seq a q = 120 :=
sorry

end geometric_sequence_sum_is_120_l2009_200911


namespace bob_total_distance_traveled_over_six_days_l2009_200951

theorem bob_total_distance_traveled_over_six_days (x : ℤ) (hx1 : 3 ≤ x) (hx2 : x % 3 = 0):
  (90 / x + 90 / (x + 3) + 90 / (x + 6) + 90 / (x + 9) + 90 / (x + 12) + 90 / (x + 15) : ℝ) = 73.5 :=
by
  sorry

end bob_total_distance_traveled_over_six_days_l2009_200951


namespace ticket_distribution_l2009_200986

theorem ticket_distribution 
    (A Ad C Cd S : ℕ) 
    (h1 : 25 * A + 20 * 50 + 15 * C + 10 * 30 + 20 * S = 7200) 
    (h2 : A + 50 + C + 30 + S = 400)
    (h3 : A + 50 = 2 * S)
    (h4 : Ad = 50)
    (h5 : Cd = 30) : 
    A = 102 ∧ Ad = 50 ∧ C = 142 ∧ Cd = 30 ∧ S = 76 := 
by 
    sorry

end ticket_distribution_l2009_200986


namespace roots_of_equation_l2009_200901

theorem roots_of_equation :
  ∀ x : ℚ, (3 * x^2 / (x - 2) - (5 * x + 10) / 4 + (9 - 9 * x) / (x - 2) + 2 = 0) ↔ 
           (x = 6 ∨ x = 17/3) := 
sorry

end roots_of_equation_l2009_200901


namespace ratio_brownies_to_cookies_l2009_200900

-- Conditions and definitions
def total_items : ℕ := 104
def cookies_sold : ℕ := 48
def brownies_sold : ℕ := total_items - cookies_sold

-- Problem statement
theorem ratio_brownies_to_cookies : (brownies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 7 ∧ (cookies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 6 :=
by
  sorry

end ratio_brownies_to_cookies_l2009_200900


namespace range_of_f_l2009_200978

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem range_of_f :
  ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc (1 : ℝ) (Real.sqrt 2) := 
by
  intro x hx
  rw [Set.mem_Icc] at hx
  have : ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc 1 (Real.sqrt 2) := sorry
  exact this x hx

end range_of_f_l2009_200978


namespace count_not_divisible_by_5_or_7_l2009_200969

theorem count_not_divisible_by_5_or_7 :
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  total_numbers - count_divisible_by_5_or_7 = 343 :=
by
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  have h : total_numbers - count_divisible_by_5_or_7 = 343 := by sorry
  exact h

end count_not_divisible_by_5_or_7_l2009_200969


namespace units_digit_of_17_pow_549_l2009_200967

theorem units_digit_of_17_pow_549 : (17 ^ 549) % 10 = 7 :=
by {
  -- Provide the necessary steps or strategies to prove the theorem
  sorry
}

end units_digit_of_17_pow_549_l2009_200967


namespace small_mold_radius_l2009_200919

theorem small_mold_radius (r : ℝ) (n : ℝ) (s : ℝ) :
    r = 2 ∧ n = 8 ∧ (1 / 2) * (2 / 3) * Real.pi * r^3 = (8 * (2 / 3) * Real.pi * s^3) → s = 1 :=
by
  sorry

end small_mold_radius_l2009_200919


namespace determine_x_l2009_200950

theorem determine_x (x : ℝ) (A B : Set ℝ) (H1 : A = {-1, 0}) (H2 : B = {0, 1, x + 2}) (H3 : A ⊆ B) : x = -3 :=
sorry

end determine_x_l2009_200950


namespace Mildred_heavier_than_Carol_l2009_200990

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end Mildred_heavier_than_Carol_l2009_200990


namespace distinct_real_roots_sum_l2009_200973

theorem distinct_real_roots_sum (p r_1 r_2 : ℝ) (h_eq : ∀ x, x^2 + p * x + 18 = 0)
  (h_distinct : r_1 ≠ r_2) (h_root1 : x^2 + p * x + 18 = 0)
  (h_root2 : x^2 + p * x + 18 = 0) : |r_1 + r_2| > 6 :=
sorry

end distinct_real_roots_sum_l2009_200973


namespace minimize_S_n_l2009_200962

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

axiom arithmetic_sequence : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
axiom sum_first_n_terms : ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * d)
axiom condition1 : a 0 + a 4 = -14
axiom condition2 : S 9 = -27

theorem minimize_S_n : ∃ n, ∀ m, S n ≤ S m := sorry

end minimize_S_n_l2009_200962


namespace determine_g_l2009_200959

variable {R : Type*} [CommRing R]

theorem determine_g (g : R → R) (x : R) :
  (4 * x^5 + 3 * x^3 - 2 * x + 1 + g x = 7 * x^3 - 5 * x^2 + 4 * x - 3) →
  g x = -4 * x^5 + 4 * x^3 - 5 * x^2 + 6 * x - 4 :=
by
  sorry

end determine_g_l2009_200959


namespace complex_number_real_l2009_200992

theorem complex_number_real (m : ℝ) (z : ℂ) 
  (h1 : z = ⟨1 / (m + 5), 0⟩ + ⟨0, m^2 + 2 * m - 15⟩)
  (h2 : m^2 + 2 * m - 15 = 0)
  (h3 : m ≠ -5) :
  m = 3 :=
sorry

end complex_number_real_l2009_200992


namespace simplify_expression_l2009_200987

variable {x y : ℝ}
variable (h : x * y ≠ 0)

theorem simplify_expression (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^2 + 1) / y) - ((x^2 - 1) / y) * ((y^3 - 1) / x) =
  (x^3*y^2 - x^2*y^3 + x^3 + x^2 + y^2 + y^3) / (x*y) :=
by sorry

end simplify_expression_l2009_200987


namespace min_value_inequality_l2009_200984

theorem min_value_inequality (a b c d e f : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f)
    (h_sum : a + b + c + d + e + f = 9) : 
    1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f ≥ 676 / 9 := 
by 
  sorry

end min_value_inequality_l2009_200984


namespace quadruple_pieces_sold_l2009_200999

theorem quadruple_pieces_sold (split_earnings : (2 : ℝ) * 5 = 10) 
  (single_pieces_sold : 100 * (0.01 : ℝ) = 1) 
  (double_pieces_sold : 45 * (0.02 : ℝ) = 0.9) 
  (triple_pieces_sold : 50 * (0.03 : ℝ) = 1.5) : 
  let total_earnings := 10
  let earnings_from_others := 3.4
  let quadruple_piece_price := 0.04
  total_earnings - earnings_from_others = 6.6 → 
  6.6 / quadruple_piece_price = 165 :=
by 
  intros 
  sorry

end quadruple_pieces_sold_l2009_200999


namespace complement_A_in_U_l2009_200930

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 + x - 2 < 0}

theorem complement_A_in_U :
  (U \ A) = {-2, 1, 2} :=
by 
  -- proof will be done here
  sorry

end complement_A_in_U_l2009_200930


namespace scooter_safety_gear_price_increase_l2009_200985

theorem scooter_safety_gear_price_increase :
  let last_year_scooter_price := 200
  let last_year_gear_price := 50
  let scooter_increase_rate := 0.08
  let gear_increase_rate := 0.15
  let total_last_year_price := last_year_scooter_price + last_year_gear_price
  let this_year_scooter_price := last_year_scooter_price * (1 + scooter_increase_rate)
  let this_year_gear_price := last_year_gear_price * (1 + gear_increase_rate)
  let total_this_year_price := this_year_scooter_price + this_year_gear_price
  let total_increase := total_this_year_price - total_last_year_price
  let percent_increase := (total_increase / total_last_year_price) * 100
  percent_increase = 9 :=
by
  -- sorry is added here to skip the proof steps
  sorry

end scooter_safety_gear_price_increase_l2009_200985


namespace area_is_12_l2009_200957

-- Definitions based on conditions
def isosceles_triangle (a b m : ℝ) : Prop :=
  a = b ∧ m > 0 ∧ a > 0

def median (height base_length : ℝ) : Prop :=
  height > 0 ∧ base_length > 0

noncomputable def area_of_isosceles_triangle_with_given_median (a m : ℝ) : ℝ :=
  let base_half := Real.sqrt (a^2 - m^2)
  let base := 2 * base_half
  (1 / 2) * base * m

-- Prove that the area of the isosceles triangle is correct given conditions
theorem area_is_12 :
  ∀ (a m : ℝ), isosceles_triangle a a m → median m (2 * Real.sqrt (a^2 - m^2)) → area_of_isosceles_triangle_with_given_median a m = 12 := 
by
  intros a m hiso hmed
  sorry  -- Proof steps are omitted

end area_is_12_l2009_200957


namespace mohamed_donated_more_l2009_200963

-- Definitions of the conditions
def toysLeilaDonated : ℕ := 2 * 25
def toysMohamedDonated : ℕ := 3 * 19

-- The theorem stating Mohamed donated 7 more toys than Leila
theorem mohamed_donated_more : toysMohamedDonated - toysLeilaDonated = 7 :=
by
  sorry

end mohamed_donated_more_l2009_200963


namespace brad_zip_code_l2009_200946

theorem brad_zip_code (x y : ℕ) (h1 : x + x + 0 + 2 * x + y = 10) : 2 * x + y = 8 :=
by 
  sorry

end brad_zip_code_l2009_200946


namespace value_of_expression_l2009_200916

theorem value_of_expression : (5^2 - 4^2 + 3^2) = 18 := 
by
  sorry

end value_of_expression_l2009_200916


namespace circumcircle_of_right_triangle_l2009_200991

theorem circumcircle_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  ∃ (x y : ℝ), (x - 0)^2 + (y - 0)^2 = 25 :=
by
  sorry

end circumcircle_of_right_triangle_l2009_200991


namespace correct_expression_l2009_200960

theorem correct_expression (a : ℝ) :
  (a^3 * a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ ¬((a - 1)^2 = a^2 - 1) :=
by
  sorry

end correct_expression_l2009_200960


namespace unique_midpoints_are_25_l2009_200941

/-- Define the properties of a parallelogram with marked points such as vertices, midpoints of sides, and intersection point of diagonals --/
structure Parallelogram :=
(vertices : Set ℝ)
(midpoints : Set ℝ)
(diagonal_intersection : ℝ)

def congruent_parallelograms (P P' : Parallelogram) : Prop :=
  P.vertices = P'.vertices ∧ P.midpoints = P'.midpoints ∧ P.diagonal_intersection = P'.diagonal_intersection

def unique_midpoints_count (P P' : Parallelogram) : ℕ := sorry

theorem unique_midpoints_are_25
  (P P' : Parallelogram)
  (h_congruent : congruent_parallelograms P P') :
  unique_midpoints_count P P' = 25 := sorry

end unique_midpoints_are_25_l2009_200941


namespace factorize_expression_l2009_200909

theorem factorize_expression (m n : ℝ) :
  2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) :=
by
  sorry

end factorize_expression_l2009_200909


namespace correct_operation_l2009_200953

theorem correct_operation (a b : ℝ) : 
  ¬(a^2 + a^3 = a^5) ∧ ¬((a^2)^3 = a^8) ∧ (a^3 / a^2 = a) ∧ ¬((a - b)^2 = a^2 - b^2) := 
by {
  sorry
}

end correct_operation_l2009_200953


namespace game_is_not_fair_l2009_200921

noncomputable def expected_winnings : ℚ := 
  let p_1 := 1 / 8
  let p_2 := 7 / 8
  let gain_case_1 := 2
  let loss_case_2 := -1 / 7
  (p_1 * gain_case_1) + (p_2 * loss_case_2)

theorem game_is_not_fair : expected_winnings = 1 / 8 :=
sorry

end game_is_not_fair_l2009_200921


namespace total_votes_is_5000_l2009_200996

theorem total_votes_is_5000 :
  ∃ (V : ℝ), 0.45 * V - 0.35 * V = 500 ∧ 0.35 * V - 0.20 * V = 350 ∧ V = 5000 :=
by
  sorry

end total_votes_is_5000_l2009_200996


namespace squares_perimeter_and_rectangle_area_l2009_200912

theorem squares_perimeter_and_rectangle_area (x y : ℝ) (hx : x^2 + y^2 = 145) (hy : x^2 - y^2 = 105) : 
  (4 * x + 4 * y = 28 * Real.sqrt 5) ∧ ((x + y) * x = 175) := 
by 
  sorry

end squares_perimeter_and_rectangle_area_l2009_200912


namespace problem_l2009_200947

theorem problem : 
  let b := 2 ^ 51
  let c := 4 ^ 25
  b > c :=
by 
  let b := 2 ^ 51
  let c := 4 ^ 25
  sorry

end problem_l2009_200947


namespace pyramid_property_l2009_200914

-- Define the areas of the faces of the right-angled triangular pyramid.
variables (S_ABC S_ACD S_ADB S_BCD : ℝ)

-- Define the condition that the areas correspond to a right-angled triangular pyramid.
def right_angled_triangular_pyramid (S_ABC S_ACD S_ADB S_BCD : ℝ) : Prop :=
  S_BCD^2 = S_ABC^2 + S_ACD^2 + S_ADB^2

-- State the theorem to be proven.
theorem pyramid_property : right_angled_triangular_pyramid S_ABC S_ACD S_ADB S_BCD :=
sorry

end pyramid_property_l2009_200914


namespace fraction_expression_value_l2009_200903

theorem fraction_expression_value:
  (1/4 - 1/5) / (1/3 - 1/6) = 3/10 :=
by
  sorry

end fraction_expression_value_l2009_200903


namespace total_handshakes_l2009_200927

section Handshakes

-- Define the total number of players
def total_players : ℕ := 4 + 6

-- Define the number of players in 2 and 3 player teams
def num_2player_teams : ℕ := 2
def num_3player_teams : ℕ := 2

-- Define the number of players per 2 player team and 3 player team
def players_per_2player_team : ℕ := 2
def players_per_3player_team : ℕ := 3

-- Define the total number of players in 2 player teams and in 3 player teams
def total_2player_team_players : ℕ := num_2player_teams * players_per_2player_team
def total_3player_team_players : ℕ := num_3player_teams * players_per_3player_team

-- Calculate handshakes
def handshakes (total_2player : ℕ) (total_3player : ℕ) : ℕ :=
  let h1 := total_2player * (total_players - players_per_2player_team) / 2
  let h2 := total_3player * (total_players - players_per_3player_team) / 2
  h1 + h2

-- Prove the total number of handshakes
theorem total_handshakes : handshakes total_2player_team_players total_3player_team_players = 37 :=
by
  have h1 := total_2player_team_players * (total_players - players_per_2player_team) / 2
  have h2 := total_3player_team_players * (total_players - players_per_3player_team) / 2
  have h_total := h1 + h2
  sorry

end Handshakes

end total_handshakes_l2009_200927


namespace polynomial_coeff_sum_abs_l2009_200983

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℤ) 
  (h : (2*x - 1)^5 + (x + 2)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  |a| + |a_2| + |a_4| = 30 :=
sorry

end polynomial_coeff_sum_abs_l2009_200983


namespace solve_for_x_l2009_200954

theorem solve_for_x (x : ℝ) (h : 3 * x = 16 - x + 4) : x = 5 := 
by
  sorry

end solve_for_x_l2009_200954


namespace fraction_meaningful_l2009_200943

theorem fraction_meaningful (x : ℝ) : (x-5) ≠ 0 ↔ (1 / (x - 5)) = (1 / (x - 5)) := 
by 
  sorry

end fraction_meaningful_l2009_200943


namespace tan_theta_l2009_200915

theorem tan_theta (θ : ℝ) (x y : ℝ) (hx : x = - (Real.sqrt 3) / 2) (hy : y = 1 / 2) (h_terminal : True) : 
  Real.tan θ = - (Real.sqrt 3) / 3 :=
sorry

end tan_theta_l2009_200915


namespace best_fitting_model_l2009_200932

/-- A type representing the coefficient of determination of different models -/
def r_squared (m : ℕ) : ℝ :=
  match m with
  | 1 => 0.98
  | 2 => 0.80
  | 3 => 0.50
  | 4 => 0.25
  | _ => 0 -- An auxiliary value for invalid model numbers

/-- The best fitting model is the one with the highest r_squared value --/
theorem best_fitting_model : r_squared 1 = max (r_squared 1) (max (r_squared 2) (max (r_squared 3) (r_squared 4))) :=
by
  sorry

end best_fitting_model_l2009_200932


namespace F_minimum_value_neg_inf_to_0_l2009_200981

variable (f g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) := ∀ x, h (-x) = - (h x)

theorem F_minimum_value_neg_inf_to_0 
  (hf_odd : is_odd f) 
  (hg_odd : is_odd g)
  (hF_max : ∀ x > 0, f x + g x + 2 ≤ 8) 
  (hF_reaches_max : ∃ x > 0, f x + g x + 2 = 8) :
  ∀ x < 0, f x + g x + 2 ≥ -4 :=
by
  sorry

end F_minimum_value_neg_inf_to_0_l2009_200981


namespace intersect_complementB_l2009_200961

def setA (x : ℝ) : Prop := ∃ y : ℝ, y = Real.log (9 - x^2)

def setB (x : ℝ) : Prop := ∃ y : ℝ, y = Real.sqrt (4 * x - x^2)

def complementB (x : ℝ) : Prop := x < 0 ∨ 4 < x

theorem intersect_complementB :
  { x : ℝ | setA x } ∩ { x : ℝ | complementB x } = { x : ℝ | -3 < x ∧ x < 0 } :=
sorry

end intersect_complementB_l2009_200961


namespace tan_sum_l2009_200976

theorem tan_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 96 / 65)
  (h2 : Real.cos x + Real.cos y = 72 / 65) :
  Real.tan x + Real.tan y = 507 / 112 := 
sorry

end tan_sum_l2009_200976


namespace sum_of_solutions_l2009_200956

theorem sum_of_solutions (y : ℤ) (x1 x2 : ℤ) (h1 : y = 8) (h2 : x1^2 + y^2 = 145) (h3 : x2^2 + y^2 = 145) : x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l2009_200956


namespace maximum_cards_l2009_200913

def total_budget : ℝ := 15
def card_cost : ℝ := 1.25
def transaction_fee : ℝ := 2
def desired_savings : ℝ := 3

theorem maximum_cards : ∃ n : ℕ, n ≤ 8 ∧ (card_cost * (n : ℝ) + transaction_fee ≤ total_budget - desired_savings) :=
by sorry

end maximum_cards_l2009_200913


namespace beth_gave_away_54_crayons_l2009_200928

-- Define the initial number of crayons
def initialCrayons : ℕ := 106

-- Define the number of crayons left
def remainingCrayons : ℕ := 52

-- Define the number of crayons given away
def crayonsGiven (initial remaining: ℕ) : ℕ := initial - remaining

-- The goal is to prove that Beth gave away 54 crayons
theorem beth_gave_away_54_crayons : crayonsGiven initialCrayons remainingCrayons = 54 :=
by
  sorry

end beth_gave_away_54_crayons_l2009_200928


namespace recurring_decimal_to_fraction_l2009_200997

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l2009_200997


namespace billy_piles_l2009_200910

theorem billy_piles (Q D : ℕ) (h : 2 * Q + 3 * D = 20) :
  Q = 4 ∧ D = 4 :=
sorry

end billy_piles_l2009_200910


namespace picture_edge_distance_l2009_200974

theorem picture_edge_distance 
    (wall_width : ℕ) 
    (picture_width : ℕ) 
    (centered : Bool) 
    (h_w : wall_width = 22) 
    (h_p : picture_width = 4) 
    (h_c : centered = true) : 
    ∃ (distance : ℕ), distance = 9 := 
by
  sorry

end picture_edge_distance_l2009_200974


namespace intersection_A_B_l2009_200925

-- Definition of sets A and B
def A := {x : ℝ | x > 2}
def B := { x : ℝ | (x - 1) * (x - 3) < 0 }

-- Claim that A ∩ B = {x : ℝ | 2 < x < 3}
theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l2009_200925


namespace expenditure_of_neg_50_l2009_200979

/-- In the book "Nine Chapters on the Mathematical Art," it is noted that
"when two calculations have opposite meanings, they should be named positive
and negative." This means: if an income of $80 is denoted as $+80, then $-50
represents an expenditure of $50. -/
theorem expenditure_of_neg_50 :
  (∀ (income : ℤ), income = 80 → -income = -50 → ∃ (expenditure : ℤ), expenditure = 50) := sorry

end expenditure_of_neg_50_l2009_200979


namespace fare_per_1_5_mile_l2009_200975

-- Definitions and conditions
def fare_first : ℝ := 1.0
def total_fare : ℝ := 7.3
def increments_per_mile : ℝ := 5.0
def total_miles : ℝ := 3.0
def remaining_increments : ℝ := (total_miles * increments_per_mile) - 1
def remaining_fare : ℝ := total_fare - fare_first

-- Theorem to prove
theorem fare_per_1_5_mile : remaining_fare / remaining_increments = 0.45 :=
by
  sorry

end fare_per_1_5_mile_l2009_200975


namespace pages_read_per_day_l2009_200949

-- Define the total number of pages in the book
def total_pages := 96

-- Define the number of days it took to finish the book
def number_of_days := 12

-- Define pages read per day for Charles
def pages_per_day := total_pages / number_of_days

-- Prove that the number of pages read per day is equal to 8
theorem pages_read_per_day : pages_per_day = 8 :=
by
  sorry

end pages_read_per_day_l2009_200949


namespace correct_scientific_notation_l2009_200993

def scientific_notation (n : ℝ) : ℝ × ℝ := 
  (4, 5)

theorem correct_scientific_notation : scientific_notation 400000 = (4, 5) :=
by {
  sorry
}

end correct_scientific_notation_l2009_200993


namespace angle_between_NE_and_SW_l2009_200972

theorem angle_between_NE_and_SW
  (n : ℕ) (hn : n = 12)
  (total_degrees : ℚ) (htotal : total_degrees = 360)
  (spaced_rays : ℚ) (hspaced : spaced_rays = total_degrees / n)
  (angles_between_NE_SW : ℕ) (hangles : angles_between_NE_SW = 4) :
  (angles_between_NE_SW * spaced_rays = 120) :=
by
  rw [htotal, hn] at hspaced
  rw [hangles]
  rw [hspaced]
  sorry

end angle_between_NE_and_SW_l2009_200972


namespace frequency_of_largest_rectangle_area_l2009_200936

theorem frequency_of_largest_rectangle_area (a : ℕ → ℝ) (sample_size : ℕ)
    (h_geom : ∀ n, a (n + 1) = 2 * a n) (h_sum : a 0 + a 1 + a 2 + a 3 = 1)
    (h_sample : sample_size = 300) : 
    sample_size * a 3 = 160 := by
  sorry

end frequency_of_largest_rectangle_area_l2009_200936


namespace div_sqrt_81_by_3_is_3_l2009_200934

-- Definitions based on conditions
def sqrt_81 := Nat.sqrt 81
def number_3 := 3

-- Problem statement
theorem div_sqrt_81_by_3_is_3 : sqrt_81 / number_3 = 3 := by
  sorry

end div_sqrt_81_by_3_is_3_l2009_200934


namespace reciprocal_of_sum_of_fractions_l2009_200931

theorem reciprocal_of_sum_of_fractions :
  (1 / (1 / 4 + 1 / 6)) = 12 / 5 :=
by
  sorry

end reciprocal_of_sum_of_fractions_l2009_200931


namespace gifts_left_l2009_200929

variable (initial_gifts : ℕ)
variable (gifts_sent : ℕ)

theorem gifts_left (h_initial : initial_gifts = 77) (h_sent : gifts_sent = 66) : initial_gifts - gifts_sent = 11 := by
  sorry

end gifts_left_l2009_200929


namespace marbles_in_larger_bottle_l2009_200918

theorem marbles_in_larger_bottle 
  (small_bottle_volume : ℕ := 20)
  (small_bottle_marbles : ℕ := 40)
  (larger_bottle_volume : ℕ := 60) :
  (small_bottle_marbles / small_bottle_volume) * larger_bottle_volume = 120 := 
by
  sorry

end marbles_in_larger_bottle_l2009_200918


namespace evaluate_expression_at_values_l2009_200968

theorem evaluate_expression_at_values (x y : ℤ) (h₁ : x = 1) (h₂ : y = -2) :
  (-2 * x ^ 2 + 2 * x - y) = 2 :=
by
  subst h₁
  subst h₂
  sorry

end evaluate_expression_at_values_l2009_200968


namespace total_salmon_now_l2009_200935

def initial_salmon : ℕ := 500

def increase_factor : ℕ := 10

theorem total_salmon_now : initial_salmon * increase_factor = 5000 := by
  sorry

end total_salmon_now_l2009_200935


namespace sufficient_condition_for_increasing_l2009_200924

theorem sufficient_condition_for_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^y < a^x) →
  (∀ x y : ℝ, x < y → (2 - a) * y ^ 3 > (2 - a) * x ^ 3) :=
sorry

end sufficient_condition_for_increasing_l2009_200924


namespace binomial_divisible_by_prime_l2009_200955

-- Define the conditions: p is prime and 0 < k < p
variables (p k : ℕ)
variable (hp : Nat.Prime p)
variable (hk : 0 < k ∧ k < p)

-- State that the binomial coefficient \(\binom{p}{k}\) is divisible by \( p \)
theorem binomial_divisible_by_prime
  (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_l2009_200955


namespace find_solution_l2009_200971

def satisfies_conditions (x y z : ℝ) :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solution (x y z : ℝ) :
  satisfies_conditions x y z →
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
by
  sorry

end find_solution_l2009_200971


namespace oil_amount_to_add_l2009_200933

variable (a b : ℝ)
variable (h1 : a = 0.16666666666666666)
variable (h2 : b = 0.8333333333333334)

theorem oil_amount_to_add (a b : ℝ) (h1 : a = 0.16666666666666666) (h2 : b = 0.8333333333333334) : 
  b - a = 0.6666666666666667 := by
  rw [h1, h2]
  norm_num
  sorry

end oil_amount_to_add_l2009_200933


namespace mary_brought_stickers_l2009_200945

theorem mary_brought_stickers (friends_stickers : Nat) (other_stickers : Nat) (left_stickers : Nat) 
                              (total_students : Nat) (num_friends : Nat) (stickers_per_friend : Nat) 
                              (stickers_per_other_student : Nat) :
  friends_stickers = num_friends * stickers_per_friend →
  left_stickers = 8 →
  total_students = 17 →
  num_friends = 5 →
  stickers_per_friend = 4 →
  stickers_per_other_student = 2 →
  other_stickers = (total_students - 1 - num_friends) * stickers_per_other_student →
  (friends_stickers + other_stickers + left_stickers) = 50 :=
by
  intros
  sorry

end mary_brought_stickers_l2009_200945


namespace black_pork_zongzi_price_reduction_l2009_200905

def price_reduction_15_dollars (initial_profit initial_boxes extra_boxes_per_dollar x : ℕ) : Prop :=
  initial_profit > x ∧ (initial_profit - x) * (initial_boxes + extra_boxes_per_dollar * x) = 2800 -> x = 15

-- Applying the problem conditions explicitly and stating the proposition to prove
theorem black_pork_zongzi_price_reduction:
  price_reduction_15_dollars 50 50 2 15 :=
by
  -- Here we state the question as a proposition based on the identified conditions and correct answer
  sorry

end black_pork_zongzi_price_reduction_l2009_200905


namespace S9_value_l2009_200964

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Define the arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a_n (n + 1) - a_n n) = (a_n 1 - a_n 0)

-- Sum of the first n terms of arithmetic sequence
def sum_first_n_terms (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n n = n * (a_n 0 + a_n (n - 1)) / 2

-- Given conditions: 
axiom a4_plus_a6 : a_n 4 + a_n 6 = 12
axiom S_definition : sum_first_n_terms S_n a_n

theorem S9_value : S_n 9 = 54 :=
by
  -- assuming the given conditions and definitions, we aim to prove the desired theorem.
  sorry

end S9_value_l2009_200964


namespace equivalence_a_gt_b_and_inv_a_lt_inv_b_l2009_200923

variable {a b : ℝ}

theorem equivalence_a_gt_b_and_inv_a_lt_inv_b (h : a * b > 0) : 
  (a > b) ↔ (1 / a < 1 / b) := 
sorry

end equivalence_a_gt_b_and_inv_a_lt_inv_b_l2009_200923


namespace work_completion_days_l2009_200998

theorem work_completion_days
  (E_q : ℝ) -- Efficiency of q
  (E_p : ℝ) -- Efficiency of p
  (E_r : ℝ) -- Efficiency of r
  (W : ℝ)  -- Total work
  (H1 : E_p = 1.5 * E_q) -- Condition 1
  (H2 : W = E_p * 25) -- Condition 2
  (H3 : E_r = 0.8 * E_q) -- Condition 3
  : (W / (E_p + E_q + E_r)) = 11.36 := -- Prove the days_needed is 11.36
by
  sorry

end work_completion_days_l2009_200998


namespace razorback_tshirt_profit_l2009_200952

theorem razorback_tshirt_profit
  (total_tshirts_sold : ℕ)
  (tshirts_sold_arkansas_game : ℕ)
  (money_made_arkansas_game : ℕ) :
  total_tshirts_sold = 163 →
  tshirts_sold_arkansas_game = 89 →
  money_made_arkansas_game = 8722 →
  money_made_arkansas_game / tshirts_sold_arkansas_game = 98 :=
by 
  intros _ _ _
  sorry

end razorback_tshirt_profit_l2009_200952


namespace prime_factor_of_sum_l2009_200908

theorem prime_factor_of_sum (n : ℤ) : ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ (2 * n + 1 + 2 * n + 3 + 2 * n + 5 + 2 * n + 7) % p = 0 :=
by
  sorry

end prime_factor_of_sum_l2009_200908


namespace diameter_correct_l2009_200988

noncomputable def diameter_of_circle (C : ℝ) (hC : C = 36) : ℝ :=
  let r := C / (2 * Real.pi)
  2 * r

theorem diameter_correct (C : ℝ) (hC : C = 36) : diameter_of_circle C hC = 36 / Real.pi := by
  sorry

end diameter_correct_l2009_200988
