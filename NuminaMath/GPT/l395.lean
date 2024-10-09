import Mathlib

namespace log_sum_l395_39569

-- Define the common logarithm function using Lean's natural logarithm with a change of base
noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum : log_base_10 5 + log_base_10 0.2 = 0 :=
by
  -- Placeholder for the proof to be completed
  sorry

end log_sum_l395_39569


namespace pencils_given_away_l395_39551

-- Define the basic values and conditions
def initial_pencils : ℕ := 39
def bought_pencils : ℕ := 22
def final_pencils : ℕ := 43

-- Let x be the number of pencils Brian gave away
variable (x : ℕ)

-- State the theorem we need to prove
theorem pencils_given_away : (initial_pencils - x) + bought_pencils = final_pencils → x = 18 := by
  sorry

end pencils_given_away_l395_39551


namespace apples_b_lighter_than_a_l395_39568

-- Definitions based on conditions
def total_weight : ℕ := 72
def weight_basket_a : ℕ := 42
def weight_basket_b : ℕ := total_weight - weight_basket_a

-- Theorem to prove the question equals the answer given the conditions
theorem apples_b_lighter_than_a : (weight_basket_a - weight_basket_b) = 12 := by
  -- Placeholder for proof
  sorry

end apples_b_lighter_than_a_l395_39568


namespace find_x_l395_39518

structure Vector2D where
  x : ℝ
  y : ℝ

def vecAdd (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

def vecScale (c : ℝ) (v : Vector2D) : Vector2D :=
  ⟨c * v.x, c * v.y⟩

def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem find_x (x : ℝ)
  (a : Vector2D := ⟨1, 2⟩)
  (b : Vector2D := ⟨x, 1⟩)
  (h : areParallel (vecAdd a (vecScale 2 b)) (vecAdd (vecScale 2 a) (vecScale (-2) b))) :
  x = 1 / 2 :=
by
  sorry

end find_x_l395_39518


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l395_39582

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l395_39582


namespace product_of_areas_eq_square_of_volume_l395_39512

theorem product_of_areas_eq_square_of_volume
    (a b c : ℝ)
    (bottom_area : ℝ) (side_area : ℝ) (front_area : ℝ)
    (volume : ℝ)
    (h1 : bottom_area = a * b)
    (h2 : side_area = b * c)
    (h3 : front_area = c * a)
    (h4 : volume = a * b * c) :
    bottom_area * side_area * front_area = volume ^ 2 := by
  -- proof omitted
  sorry

end product_of_areas_eq_square_of_volume_l395_39512


namespace find_k_of_geometric_mean_l395_39596

-- Let {a_n} be an arithmetic sequence with common difference d and a_1 = 9d.
-- Prove that if a_k is the geometric mean of a_1 and a_{2k}, then k = 4.
theorem find_k_of_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : ∀ n, a n = 9 * d + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a k ^ 2 = a 1 * a (2 * k)) : k = 4 :=
sorry

end find_k_of_geometric_mean_l395_39596


namespace ratio_of_chicken_to_beef_l395_39576

theorem ratio_of_chicken_to_beef
  (beef_pounds : ℕ)
  (chicken_price_per_pound : ℕ)
  (total_cost : ℕ)
  (beef_price_per_pound : ℕ)
  (beef_cost : ℕ)
  (chicken_cost : ℕ)
  (chicken_pounds : ℕ) :
  beef_pounds = 1000 →
  beef_price_per_pound = 8 →
  total_cost = 14000 →
  beef_cost = beef_pounds * beef_price_per_pound →
  chicken_cost = total_cost - beef_cost →
  chicken_price_per_pound = 3 →
  chicken_pounds = chicken_cost / chicken_price_per_pound →
  chicken_pounds / beef_pounds = 2 :=
by
  intros
  sorry

end ratio_of_chicken_to_beef_l395_39576


namespace intersection_of_M_and_N_l395_39531

def M := {x : ℝ | abs x ≤ 2}
def N := {x : ℝ | x^2 - 3 * x = 0}

theorem intersection_of_M_and_N : M ∩ N = {0} :=
by
  sorry

end intersection_of_M_and_N_l395_39531


namespace line_eq_l395_39565

theorem line_eq (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_eq : 1 / a + 9 / b = 1) (h_min_interp : a + b = 16) : 
  ∃ l : ℝ × ℝ → ℝ, ∀ x y : ℝ, l (x, y) = 3 * x + y - 12 :=
by
  sorry

end line_eq_l395_39565


namespace quotient_of_integers_l395_39520

variable {x y : ℤ}

theorem quotient_of_integers (h : 1996 * x + y / 96 = x + y) : 
  (x / y = 1 / 2016) ∨ (y / x = 2016) := by
  sorry

end quotient_of_integers_l395_39520


namespace solve_fraction_zero_l395_39546

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end solve_fraction_zero_l395_39546


namespace original_price_of_iWatch_l395_39513

theorem original_price_of_iWatch (P : ℝ) (h1 : 800 > 0) (h2 : P > 0)
    (h3 : 680 + 0.90 * P > 0) (h4 : 0.98 * (680 + 0.90 * P) = 931) :
    P = 300 := by
  sorry

end original_price_of_iWatch_l395_39513


namespace find_integer_x_l395_39574

theorem find_integer_x : ∃ x : ℤ, x^5 - 3 * x^2 = 216 ∧ x = 3 :=
by {
  sorry
}

end find_integer_x_l395_39574


namespace rate_per_kg_for_mangoes_l395_39566

theorem rate_per_kg_for_mangoes (quantity_grapes : ℕ)
    (rate_grapes : ℕ)
    (quantity_mangoes : ℕ)
    (total_payment : ℕ)
    (rate_mangoes : ℕ) :
    quantity_grapes = 8 →
    rate_grapes = 70 →
    quantity_mangoes = 9 →
    total_payment = 1055 →
    8 * 70 + 9 * rate_mangoes = 1055 →
    rate_mangoes = 55 := by
  intros h1 h2 h3 h4 h5
  have h6 : 8 * 70 = 560 := by norm_num
  have h7 : 560 + 9 * rate_mangoes = 1055 := by rw [h5]
  have h8 : 1055 - 560 = 495 := by norm_num
  have h9 : 9 * rate_mangoes = 495 := by linarith
  have h10 : rate_mangoes = 55 := by linarith
  exact h10

end rate_per_kg_for_mangoes_l395_39566


namespace average_weight_of_section_A_l395_39541

theorem average_weight_of_section_A (nA nB : ℕ) (WB WC : ℝ) (WA : ℝ) :
  nA = 50 →
  nB = 40 →
  WB = 70 →
  WC = 58.89 →
  50 * WA + 40 * WB = 58.89 * 90 →
  WA = 50.002 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_weight_of_section_A_l395_39541


namespace fish_tagged_initially_l395_39534

theorem fish_tagged_initially (N T : ℕ) (hN : N = 1500) 
  (h_ratio : 2 / 50 = (T:ℕ) / N) : T = 60 :=
by
  -- The proof is omitted
  sorry

end fish_tagged_initially_l395_39534


namespace original_number_is_two_over_three_l395_39558

theorem original_number_is_two_over_three (x : ℚ) (h : 1 + 1/x = 5/2) : x = 2/3 :=
sorry

end original_number_is_two_over_three_l395_39558


namespace remainder_when_1_stmt_l395_39557

-- Define the polynomial g(s)
def g (s : ℚ) : ℚ := s^15 + 1

-- Define the remainder theorem statement in the context of this problem
theorem remainder_when_1_stmt (s : ℚ) : g 1 = 2 :=
  sorry

end remainder_when_1_stmt_l395_39557


namespace marbles_per_friend_l395_39517

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) (h_total : total_marbles = 30) (h_friends : num_friends = 5) :
  total_marbles / num_friends = 6 :=
by
  -- Proof skipped
  sorry

end marbles_per_friend_l395_39517


namespace unique_integer_solution_l395_39504

theorem unique_integer_solution (x y z : ℤ) (h : 2 * x^2 + 3 * y^2 = z^2) : x = 0 ∧ y = 0 ∧ z = 0 :=
by {
  sorry
}

end unique_integer_solution_l395_39504


namespace problem1_problem2_problem3_problem4_l395_39539

-- Definitions of conversion rates used in the conditions
def sq_m_to_sq_dm : Nat := 100
def hectare_to_sq_m : Nat := 10000
def sq_cm_to_sq_dm_div : Nat := 100
def sq_km_to_hectare : Nat := 100

-- The problem statement with the expected values
theorem problem1 : 3 * sq_m_to_sq_dm = 300 := by
  sorry

theorem problem2 : 2 * hectare_to_sq_m = 20000 := by
  sorry

theorem problem3 : 5000 / sq_cm_to_sq_dm_div = 50 := by
  sorry

theorem problem4 : 8 * sq_km_to_hectare = 800 := by
  sorry

end problem1_problem2_problem3_problem4_l395_39539


namespace find_t_from_x_l395_39564

theorem find_t_from_x (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by
  sorry

end find_t_from_x_l395_39564


namespace sixteen_powers_five_equals_four_power_ten_l395_39525

theorem sixteen_powers_five_equals_four_power_ten : 
  (16 * 16 * 16 * 16 * 16 = 4 ^ 10) :=
by
  sorry

end sixteen_powers_five_equals_four_power_ten_l395_39525


namespace largest_multiple_of_15_under_500_l395_39592

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l395_39592


namespace subtracted_number_divisible_by_5_l395_39545

theorem subtracted_number_divisible_by_5 : ∃ k : ℕ, 9671 - 1 = 5 * k :=
by
  sorry

end subtracted_number_divisible_by_5_l395_39545


namespace total_cost_of_plates_and_cups_l395_39572

theorem total_cost_of_plates_and_cups 
  (P C : ℝ)
  (h : 100 * P + 200 * C = 7.50) :
  20 * P + 40 * C = 1.50 :=
by
  sorry

end total_cost_of_plates_and_cups_l395_39572


namespace max_liters_l395_39597

-- Declare the problem conditions as constants
def initialHeat : ℕ := 480 -- kJ for the first 5 min
def reductionRate : ℚ := 0.25 -- 25%
def initialTemp : ℚ := 20 -- °C
def boilingTemp : ℚ := 100 -- °C
def specificHeatCapacity : ℚ := 4.2 -- kJ/kg°C

-- Define the heat required to raise m kg of water from initialTemp to boilingTemp
def heatRequired (m : ℚ) : ℚ := m * specificHeatCapacity * (boilingTemp - initialTemp)

-- Define the total available heat from the fuel
noncomputable def totalAvailableHeat : ℚ :=
  initialHeat / (1 - (1 - reductionRate))

-- The proof problem statement
theorem max_liters :
  ∃ (m : ℕ), m ≤ 5 ∧ heatRequired m ≤ totalAvailableHeat :=
sorry

end max_liters_l395_39597


namespace calculation_l395_39535

theorem calculation : 
  let a := 20 / 9 
  let b := -53 / 4 
  (⌈ a * ⌈ b ⌉ ⌉ - ⌊ a * ⌊ b ⌋ ⌋) = 4 :=
by
  sorry

end calculation_l395_39535


namespace f_1986_eq_one_l395_39555

def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 2 * f (a * b) + 1
axiom f_one : f 1 = 1

theorem f_1986_eq_one : f 1986 = 1 :=
sorry

end f_1986_eq_one_l395_39555


namespace least_integer_value_x_l395_39571

theorem least_integer_value_x (x : ℤ) (h : |(2 : ℤ) * x + 3| ≤ 12) : x = -7 :=
by
  sorry

end least_integer_value_x_l395_39571


namespace remainder_when_divided_by_5_l395_39559

theorem remainder_when_divided_by_5 : (1234 * 1987 * 2013 * 2021) % 5 = 4 :=
by
  sorry

end remainder_when_divided_by_5_l395_39559


namespace find_x2017_l395_39584

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define that f is increasing
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y
  
-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + n * d

-- Main theorem
theorem find_x2017
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (Hodd : is_odd_function f)
  (Hinc : is_increasing_function f)
  (Hseq : ∀ n, x (n + 1) = x n + 2)
  (H7_8 : f (x 7) + f (x 8) = 0) :
  x 2017 = 4019 := 
sorry

end find_x2017_l395_39584


namespace sin_theta_value_l395_39554

open Real

noncomputable def sin_theta_sol (theta : ℝ) : ℝ :=
  (-5 + Real.sqrt 41) / 4

theorem sin_theta_value (theta : ℝ) (h1 : 5 * tan theta = 2 * cos theta) (h2 : 0 < theta) (h3 : theta < π) :
  sin theta = sin_theta_sol theta :=
by
  sorry

end sin_theta_value_l395_39554


namespace books_sold_on_friday_l395_39529

theorem books_sold_on_friday
  (total_books : ℕ)
  (books_sold_mon : ℕ)
  (books_sold_tue : ℕ)
  (books_sold_wed : ℕ)
  (books_sold_thu : ℕ)
  (pct_unsold : ℚ)
  (initial_stock : total_books = 1400)
  (sold_mon : books_sold_mon = 62)
  (sold_tue : books_sold_tue = 62)
  (sold_wed : books_sold_wed = 60)
  (sold_thu : books_sold_thu = 48)
  (percentage_unsold : pct_unsold = 0.8057142857142857) :
  total_books - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + 40) = total_books * pct_unsold :=
by
  sorry

end books_sold_on_friday_l395_39529


namespace company_sales_difference_l395_39593

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l395_39593


namespace prove_a_eq_1_l395_39581

variables {a b c d k m : ℕ}
variables (h_odd_a : a%2 = 1) 
          (h_odd_b : b%2 = 1) 
          (h_odd_c : c%2 = 1) 
          (h_odd_d : d%2 = 1)
          (h_a_pos : 0 < a) 
          (h_ineq1 : a < b) 
          (h_ineq2 : b < c) 
          (h_ineq3 : c < d)
          (h_eqn1 : a * d = b * c)
          (h_eqn2 : a + d = 2^k) 
          (h_eqn3 : b + c = 2^m)

theorem prove_a_eq_1 
  (h_odd_a : a%2 = 1) 
  (h_odd_b : b%2 = 1) 
  (h_odd_c : c%2 = 1) 
  (h_odd_d : d%2 = 1)
  (h_a_pos : 0 < a) 
  (h_ineq1 : a < b) 
  (h_ineq2 : b < c) 
  (h_ineq3 : c < d)
  (h_eqn1 : a * d = b * c)
  (h_eqn2 : a + d = 2^k) 
  (h_eqn3 : b + c = 2^m) :
  a = 1 := by
  sorry

end prove_a_eq_1_l395_39581


namespace ashley_family_spent_30_l395_39514

def cost_of_child_ticket : ℝ := 4.25
def cost_of_adult_ticket : ℝ := cost_of_child_ticket + 3.25
def discount : ℝ := 2.00
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4

def total_cost : ℝ := num_adult_tickets * cost_of_adult_ticket + num_child_tickets * cost_of_child_ticket - discount

theorem ashley_family_spent_30 :
  total_cost = 30.00 :=
sorry

end ashley_family_spent_30_l395_39514


namespace sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l395_39589

def original_price : ℝ := 150
def discount_monday_to_wednesday : ℝ := 0.20
def tax_monday_to_wednesday : ℝ := 0.05
def discount_thursday_to_saturday : ℝ := 0.15
def tax_thursday_to_saturday : ℝ := 0.04
def discount_super_saver_sunday1 : ℝ := 0.25
def discount_super_saver_sunday2 : ℝ := 0.10
def tax_super_saver_sunday : ℝ := 0.03
def discount_festive_friday : ℝ := 0.20
def tax_festive_friday : ℝ := 0.04
def additional_discount_festive_friday : ℝ := 0.05

theorem sale_price_monday_to_wednesday : (original_price * (1 - discount_monday_to_wednesday)) * (1 + tax_monday_to_wednesday) = 126 :=
by sorry

theorem sale_price_thursday_to_saturday : (original_price * (1 - discount_thursday_to_saturday)) * (1 + tax_thursday_to_saturday) = 132.60 :=
by sorry

theorem sale_price_super_saver_sunday : ((original_price * (1 - discount_super_saver_sunday1)) * (1 - discount_super_saver_sunday2)) * (1 + tax_super_saver_sunday) = 104.29 :=
by sorry

theorem sale_price_festive_friday_selected : ((original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday)) * (1 - additional_discount_festive_friday) = 118.56 :=
by sorry

theorem sale_price_festive_friday_non_selected : (original_price * (1 - discount_festive_friday)) * (1 + tax_festive_friday) = 124.80 :=
by sorry

end sale_price_monday_to_wednesday_sale_price_thursday_to_saturday_sale_price_super_saver_sunday_sale_price_festive_friday_selected_sale_price_festive_friday_non_selected_l395_39589


namespace subset_of_intervals_l395_39549

def A (x : ℝ) := -2 ≤ x ∧ x ≤ 5
def B (m x : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def is_subset_of (B A : ℝ → Prop) := ∀ x, B x → A x
def possible_values_m (m : ℝ) := m ≤ 3

theorem subset_of_intervals (m : ℝ) :
  is_subset_of (B m) A ↔ possible_values_m m := by
  sorry

end subset_of_intervals_l395_39549


namespace bike_growth_equation_l395_39542

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end bike_growth_equation_l395_39542


namespace last_donation_on_saturday_l395_39532

def total_amount : ℕ := 2010
def daily_donation : ℕ := 10
def first_day_donation : ℕ := 0 -- where 0 represents Monday, 6 represents Sunday

def total_days : ℕ := total_amount / daily_donation

def last_donation_day_of_week : ℕ := (total_days % 7 + first_day_donation) % 7

theorem last_donation_on_saturday : last_donation_day_of_week = 5 := by
  -- Prove it by calculation
  sorry

end last_donation_on_saturday_l395_39532


namespace find_divisor_l395_39594

theorem find_divisor
  (D dividend quotient remainder : ℤ)
  (h_dividend : dividend = 13787)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14)
  (h_relation : dividend = (D * quotient) + remainder) :
  D = 155 :=
by
  sorry

end find_divisor_l395_39594


namespace four_digit_positive_integers_count_l395_39575

theorem four_digit_positive_integers_count :
  let p := 17
  let a := 4582 % p
  let b := 902 % p
  let c := 2345 % p
  ∃ (n : ℕ), 
    (1000 ≤ 14 + p * n ∧ 14 + p * n ≤ 9999) ∧ 
    (4582 * (14 + p * n) + 902 ≡ 2345 [MOD p]) ∧ 
    n = 530 := sorry

end four_digit_positive_integers_count_l395_39575


namespace sum_of_three_numbers_l395_39585

theorem sum_of_three_numbers (a b c : ℕ) (h1 : b = 10)
                            (h2 : (a + b + c) / 3 = a + 15)
                            (h3 : (a + b + c) / 3 = c - 25) :
                            a + b + c = 60 :=
sorry

end sum_of_three_numbers_l395_39585


namespace Maggie_earnings_l395_39583

theorem Maggie_earnings :
  let family_commission := 7
  let neighbor_commission := 6
  let bonus_fixed := 10
  let bonus_threshold := 10
  let bonus_per_subscription := 1
  let monday_family := 4 + 1 
  let tuesday_neighbors := 2 + 2 * 2
  let wednesday_family := 3 + 1
  let total_family := monday_family + wednesday_family
  let total_neighbors := tuesday_neighbors
  let total_subscriptions := total_family + total_neighbors
  let bonus := if total_subscriptions > bonus_threshold then 
                 bonus_fixed + bonus_per_subscription * (total_subscriptions - bonus_threshold)
               else 0
  let total_earnings := total_family * family_commission + total_neighbors * neighbor_commission + bonus
  total_earnings = 114 := 
by {
  -- Placeholder for the proof. We assume this step will contain a verification of derived calculations.
  sorry
}

end Maggie_earnings_l395_39583


namespace ski_boat_rental_cost_per_hour_l395_39577

-- Let the cost per hour to rent a ski boat be x dollars
variable (x : ℝ)

-- Conditions
def cost_sailboat : ℝ := 60
def duration : ℝ := 3 * 2 -- 3 hours a day for 2 days
def cost_ken : ℝ := cost_sailboat * 2 -- Ken's total cost
def additional_cost : ℝ := 120
def cost_aldrich : ℝ := cost_ken + additional_cost -- Aldrich's total cost

-- Statement to prove
theorem ski_boat_rental_cost_per_hour (h : (duration * x = cost_aldrich)) : x = 40 := by
  sorry

end ski_boat_rental_cost_per_hour_l395_39577


namespace jack_further_down_l395_39507

-- Define the conditions given in the problem
def flights_up := 3
def flights_down := 6
def steps_per_flight := 12
def height_per_step_in_inches := 8
def inches_per_foot := 12

-- Define the number of steps and height calculations
def steps_up := flights_up * steps_per_flight
def steps_down := flights_down * steps_per_flight
def net_steps_down := steps_down - steps_up
def net_height_down_in_inches := net_steps_down * height_per_step_in_inches
def net_height_down_in_feet := net_height_down_in_inches / inches_per_foot

-- The proof statement to be shown
theorem jack_further_down : net_height_down_in_feet = 24 := sorry

end jack_further_down_l395_39507


namespace probability_product_divisible_by_4_gt_half_l395_39550

theorem probability_product_divisible_by_4_gt_half :
  let n := 2023
  let even_count := n / 2
  let four_div_count := n / 4
  let select_five := 5
  (true) ∧ (even_count = 1012) ∧ (four_div_count = 505)
  → 0.5 < (1 - ((2023 - even_count) / 2023) * ((2022 - (even_count - 1)) / 2022) * ((2021 - (even_count - 2)) / 2021) * ((2020 - (even_count - 3)) / 2020) * ((2019 - (even_count - 4)) / 2019)) :=
by
  sorry

end probability_product_divisible_by_4_gt_half_l395_39550


namespace max_m_value_l395_39523

theorem max_m_value (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^4 + 16 * m + 8 = k * (k + 1)) : m ≤ 2 :=
sorry

end max_m_value_l395_39523


namespace trader_profit_l395_39533

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def purchase_price (P : ℝ) : ℝ := 0.8 * P
noncomputable def depreciation1 (P : ℝ) : ℝ := 0.04 * P
noncomputable def depreciation2 (P : ℝ) : ℝ := 0.038 * P
noncomputable def value_after_depreciation (P : ℝ) : ℝ := 0.722 * P
noncomputable def taxes (P : ℝ) : ℝ := 0.024 * P
noncomputable def insurance (P : ℝ) : ℝ := 0.032 * P
noncomputable def maintenance (P : ℝ) : ℝ := 0.01 * P
noncomputable def total_cost (P : ℝ) : ℝ := value_after_depreciation P + taxes P + insurance P + maintenance P
noncomputable def selling_price (P : ℝ) : ℝ := 1.70 * total_cost P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def profit_percent (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) : profit_percent P = 33.96 :=
  by
    sorry

end trader_profit_l395_39533


namespace value_of_a_l395_39509

theorem value_of_a (a : ℤ) (h1 : 2 * a + 6 + (3 - a) = 0) : a = -9 :=
sorry

end value_of_a_l395_39509


namespace problem1_problem2_l395_39544

-- Problem (1)
theorem problem1 (x : ℝ) : (2 * |x - 1| ≥ 1) ↔ (x ≤ 1/2 ∨ x ≥ 3/2) := sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : a > 0) : (∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) ↔ a ≥ 2 := sorry

end problem1_problem2_l395_39544


namespace sector_area_l395_39501

noncomputable def radius_of_sector (l α : ℝ) : ℝ := l / α

noncomputable def area_of_sector (r l : ℝ) : ℝ := (1 / 2) * r * l

theorem sector_area {α l S : ℝ} (hα : α = 2) (hl : l = 3 * Real.pi) (hS : S = 9 * Real.pi ^ 2 / 4) :
  area_of_sector (radius_of_sector l α) l = S := 
by 
  rw [hα, hl, hS]
  rw [radius_of_sector, area_of_sector]
  sorry

end sector_area_l395_39501


namespace infinitesimal_alpha_as_t_to_zero_l395_39510

open Real

noncomputable def alpha (t : ℝ) : ℝ × ℝ :=
  (t, sin t)

theorem infinitesimal_alpha_as_t_to_zero : 
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, abs t < δ → abs (alpha t).fst + abs (alpha t).snd < ε := by
  sorry

end infinitesimal_alpha_as_t_to_zero_l395_39510


namespace quadratic_has_real_roots_b_3_c_1_l395_39573

theorem quadratic_has_real_roots_b_3_c_1 :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x * x + 3 * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  x₁ = (-3 + Real.sqrt 5) / 2 ∧
  x₂ = (-3 - Real.sqrt 5) / 2 :=
by
  sorry

end quadratic_has_real_roots_b_3_c_1_l395_39573


namespace number_of_cities_experienced_protests_l395_39530

variables (days_of_protest : ℕ) (arrests_per_day : ℕ) (days_pre_trial : ℕ) 
          (days_post_trial_in_weeks : ℕ) (combined_weeks_jail : ℕ)

def total_days_in_jail_per_person := days_pre_trial + (days_post_trial_in_weeks * 7) / 2

theorem number_of_cities_experienced_protests 
  (h1 : days_of_protest = 30) 
  (h2 : arrests_per_day = 10) 
  (h3 : days_pre_trial = 4) 
  (h4 : days_post_trial_in_weeks = 2) 
  (h5 : combined_weeks_jail = 9900) : 
  (combined_weeks_jail * 7) / total_days_in_jail_per_person 
  = 21 :=
by
  sorry

end number_of_cities_experienced_protests_l395_39530


namespace arithmetic_geometric_proof_l395_39522

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
∀ n, b (n + 1) = b n * r

theorem arithmetic_geometric_proof
  (a : ℕ → ℤ) (b : ℕ → ℤ) (d r : ℤ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence b r)
  (h_cond1 : 3 * a 1 - a 8 * a 8 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10):
  b 3 * b 17 = 36 :=
sorry

end arithmetic_geometric_proof_l395_39522


namespace quotient_of_5_divided_by_y_is_5_point_3_l395_39528

theorem quotient_of_5_divided_by_y_is_5_point_3 (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 :=
by
  sorry

end quotient_of_5_divided_by_y_is_5_point_3_l395_39528


namespace rectangular_solid_sum_of_edges_l395_39587

noncomputable def sum_of_edges (x y z : ℝ) := 4 * (x + y + z)

theorem rectangular_solid_sum_of_edges :
  ∃ (x y z : ℝ), (x * y * z = 512) ∧ (2 * (x * y + y * z + z * x) = 384) ∧
  (∃ (r a : ℝ), x = a / r ∧ y = a ∧ z = a * r) ∧ sum_of_edges x y z = 96 :=
by
  sorry

end rectangular_solid_sum_of_edges_l395_39587


namespace student_good_probability_l395_39506

-- Defining the conditions as given in the problem
def P_A1 := 0.25          -- Probability of selecting a student from School A
def P_A2 := 0.4           -- Probability of selecting a student from School B
def P_A3 := 0.35          -- Probability of selecting a student from School C

def P_B_given_A1 := 0.3   -- Probability that a student's level is good given they are from School A
def P_B_given_A2 := 0.6   -- Probability that a student's level is good given they are from School B
def P_B_given_A3 := 0.5   -- Probability that a student's level is good given they are from School C

-- Main theorem statement
theorem student_good_probability : 
  P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 0.49 := 
by sorry

end student_good_probability_l395_39506


namespace bees_lost_each_day_l395_39538

theorem bees_lost_each_day
    (initial_bees : ℕ)
    (daily_hatch : ℕ)
    (days : ℕ)
    (total_bees_after_days : ℕ)
    (bees_lost_each_day : ℕ) :
    initial_bees = 12500 →
    daily_hatch = 3000 →
    days = 7 →
    total_bees_after_days = 27201 →
    (initial_bees + days * (daily_hatch - bees_lost_each_day) = total_bees_after_days) →
    bees_lost_each_day = 899 :=
by
  intros h_initial h_hatch h_days h_total h_eq
  sorry

end bees_lost_each_day_l395_39538


namespace days_in_first_quarter_2010_l395_39591

theorem days_in_first_quarter_2010 : 
  let not_leap_year := ¬ (2010 % 4 = 0)
  let days_in_february := 28
  let days_in_january_and_march := 31
  not_leap_year → days_in_february = 28 → days_in_january_and_march = 31 → (31 + 28 + 31 = 90)
:= 
sorry

end days_in_first_quarter_2010_l395_39591


namespace find_number_l395_39598

theorem find_number (number : ℝ) (h : 0.75 / 100 * number = 0.06) : number = 8 := 
by
  sorry

end find_number_l395_39598


namespace total_dots_not_visible_l395_39521

-- Define the total dot sum for each die
def sum_of_dots_per_die : Nat := 1 + 2 + 3 + 4 + 5 + 6

-- Define the total number of dice
def number_of_dice : Nat := 4

-- Calculate the total dot sum for all dice
def total_dots_all_dice : Nat := sum_of_dots_per_die * number_of_dice

-- Sum of visible dots
def sum_of_visible_dots : Nat := 1 + 1 + 2 + 2 + 3 + 3 + 4 + 5 + 6 + 6

-- Prove the total dots not visible
theorem total_dots_not_visible : total_dots_all_dice - sum_of_visible_dots = 51 := by
  sorry

end total_dots_not_visible_l395_39521


namespace largest_of_six_consecutive_sum_2070_is_347_l395_39553

theorem largest_of_six_consecutive_sum_2070_is_347 (n : ℕ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070 → n + 5 = 347 :=
by
  intro h
  sorry

end largest_of_six_consecutive_sum_2070_is_347_l395_39553


namespace remainder_determined_l395_39548

theorem remainder_determined (p a b : ℤ) (h₀: Nat.Prime (Int.natAbs p)) (h₁ : ¬ (p ∣ a)) (h₂ : ¬ (p ∣ b)) :
  ∃ (r : ℤ), (r ≡ a [ZMOD p]) ∧ (r ≡ b [ZMOD p]) ∧ (r ≡ (a * b) [ZMOD p]) →
  (a ≡ r [ZMOD p]) := sorry

end remainder_determined_l395_39548


namespace problem1_problem2_l395_39503

-- Problem 1: Simplification and Evaluation
theorem problem1 (x : ℝ) : (x = -3) → 
  ((x^2 - 6*x + 9) / (x^2 - 1)) / ((x^2 - 3*x) / (x + 1))
  = -1 / 2 := sorry

-- Problem 2: Solving the Equation
theorem problem2 (x : ℝ) : 
  (∀ y, (y = x) → 
    (y / (y + 1) = 2*y / (3*y + 3) - 1)) → x = -3 / 4 := sorry

end problem1_problem2_l395_39503


namespace who_is_who_l395_39505

-- Defining the structure and terms
structure Brother :=
  (name : String)
  (has_purple_card : Bool)

-- Conditions
def first_brother := Brother.mk "Tralalya" true
def second_brother := Brother.mk "Trulalya" false

/-- Proof that the names and cards of the brothers are as stated. -/
theorem who_is_who :
  ((first_brother.name = "Tralalya" ∧ first_brother.has_purple_card = false) ∧
   (second_brother.name = "Trulalya" ∧ second_brother.has_purple_card = true)) :=
by sorry

end who_is_who_l395_39505


namespace inequality_solution_l395_39540

noncomputable def solve_inequality : Set ℝ :=
  {x | (x - 5) / ((x - 3)^2) < 0}

theorem inequality_solution :
  solve_inequality = {x | x < 3} ∪ {x | 3 < x ∧ x < 5} :=
by
  sorry

end inequality_solution_l395_39540


namespace rosie_laps_l395_39527

theorem rosie_laps (lou_distance : ℝ) (track_length : ℝ) (lou_speed_factor : ℝ) (rosie_speed_multiplier : ℝ) 
    (number_of_laps_by_lou : ℝ) (number_of_laps_by_rosie : ℕ) :
  lou_distance = 3 ∧ 
  track_length = 1 / 4 ∧ 
  lou_speed_factor = 0.75 ∧ 
  rosie_speed_multiplier = 2 ∧ 
  number_of_laps_by_lou = lou_distance / track_length ∧ 
  number_of_laps_by_rosie = rosie_speed_multiplier * number_of_laps_by_lou → 
  number_of_laps_by_rosie = 18 := 
sorry

end rosie_laps_l395_39527


namespace number_of_teams_l395_39560

-- Given the conditions and the required proof problem
theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_l395_39560


namespace probability_heads_at_least_9_l395_39586

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l395_39586


namespace tetrahedron_faces_congruent_iff_face_angle_sum_straight_l395_39570

-- Defining the Tetrahedron and its properties
structure Tetrahedron (V : Type*) :=
(A B C D : V)
(face_angle_sum_at_vertex : V → Prop)
(congruent_faces : Prop)

-- Translating the problem into a Lean 4 theorem statement
theorem tetrahedron_faces_congruent_iff_face_angle_sum_straight (V : Type*) 
  (T : Tetrahedron V) :
  T.face_angle_sum_at_vertex T.A = T.face_angle_sum_at_vertex T.B ∧ 
  T.face_angle_sum_at_vertex T.B = T.face_angle_sum_at_vertex T.C ∧ 
  T.face_angle_sum_at_vertex T.C = T.face_angle_sum_at_vertex T.D ↔ T.congruent_faces :=
sorry


end tetrahedron_faces_congruent_iff_face_angle_sum_straight_l395_39570


namespace line_perpendicular_l395_39537

theorem line_perpendicular (m : ℝ) : 
  -- Conditions
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → y = 1/2 * x + 5/2) →  -- Slope of the first line
  (∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = -2/m * x + 6/m) →  -- Slope of the second line
  -- Perpendicular condition
  ((1/2) * (-2/m) = -1) →
  -- Conclusion
  m = 1 := 
sorry

end line_perpendicular_l395_39537


namespace total_books_proof_l395_39502

noncomputable def economics_books (T : ℝ) := (1/4) * T + 10
noncomputable def rest_books (T : ℝ) := T - economics_books T
noncomputable def social_studies_books (T : ℝ) := (3/5) * rest_books T - 5
noncomputable def other_books := 13
noncomputable def science_books := 12
noncomputable def total_books_equation (T : ℝ) :=
  T = economics_books T + social_studies_books T + science_books + other_books

theorem total_books_proof : ∃ T : ℝ, total_books_equation T ∧ T = 80 := by
  sorry

end total_books_proof_l395_39502


namespace conditional_probability_correct_l395_39590

noncomputable def total_products : ℕ := 8
noncomputable def first_class_products : ℕ := 6
noncomputable def chosen_products : ℕ := 2

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def P_A : ℚ := 1 - (combination first_class_products chosen_products) / (combination total_products chosen_products)
noncomputable def P_AB : ℚ := (combination 2 1 * combination first_class_products 1) / (combination total_products chosen_products)

noncomputable def conditional_probability : ℚ := P_AB / P_A

theorem conditional_probability_correct :
  conditional_probability = 12 / 13 :=
  sorry

end conditional_probability_correct_l395_39590


namespace min_value_of_expression_l395_39508

theorem min_value_of_expression (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 :=
sorry

end min_value_of_expression_l395_39508


namespace total_money_shared_l395_39519

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end total_money_shared_l395_39519


namespace tom_books_total_l395_39579

theorem tom_books_total :
  (2 + 6 + 10 + 14 + 18) = 50 :=
by {
  -- Proof steps would go here.
  sorry
}

end tom_books_total_l395_39579


namespace max_annual_profit_at_x_9_l395_39500

noncomputable def annual_profit (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then
  8.1 * x - x^3 / 30 - 10
else
  98 - 1000 / (3 * x) - 2.7 * x

theorem max_annual_profit_at_x_9 (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 10) :
  annual_profit x ≤ annual_profit 9 :=
sorry

end max_annual_profit_at_x_9_l395_39500


namespace smaller_number_l395_39547

theorem smaller_number (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 16) : y = 4 := by
  sorry

end smaller_number_l395_39547


namespace intersection_S_T_l395_39536

def S := {x : ℝ | (x - 2) * (x - 3) ≥ 0}
def T := {x : ℝ | x > 0}

theorem intersection_S_T :
  (S ∩ T) = (Set.Ioc 0 2 ∪ Set.Ici 3) :=
by
  sorry

end intersection_S_T_l395_39536


namespace value_of_y_at_x_eq_1_l395_39563

noncomputable def quadractic_function (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem value_of_y_at_x_eq_1 (m : ℝ) (h1 : ∀ x : ℝ, x ≤ -2 → quadractic_function x m < quadractic_function (x + 1) m)
    (h2 : ∀ x : ℝ, x ≥ -2 → quadractic_function x m < quadractic_function (x + 1) m) :
    quadractic_function 1 16 = 25 :=
sorry

end value_of_y_at_x_eq_1_l395_39563


namespace inv_100_mod_101_l395_39543

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l395_39543


namespace simplify_expression_l395_39556

-- Define the statement we want to prove
theorem simplify_expression (s : ℕ) : (105 * s - 63 * s) = 42 * s :=
  by
    -- Placeholder for the proof
    sorry

end simplify_expression_l395_39556


namespace absolute_value_inequality_l395_39580

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end absolute_value_inequality_l395_39580


namespace prank_people_combinations_l395_39578

theorem prank_people_combinations (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (hMonday : Monday = 2)
  (hTuesday : Tuesday = 3)
  (hWednesday : Wednesday = 6)
  (hThursday : Thursday = 4)
  (hFriday : Friday = 3) :
  Monday * Tuesday * Wednesday * Thursday * Friday = 432 :=
  by sorry

end prank_people_combinations_l395_39578


namespace fewer_bronze_stickers_l395_39595

theorem fewer_bronze_stickers
  (gold_stickers : ℕ)
  (silver_stickers : ℕ)
  (each_student_stickers : ℕ)
  (students : ℕ)
  (total_stickers_given : ℕ)
  (bronze_stickers : ℕ)
  (total_gold_and_silver_stickers : ℕ)
  (gold_stickers_eq : gold_stickers = 50)
  (silver_stickers_eq : silver_stickers = 2 * gold_stickers)
  (each_student_stickers_eq : each_student_stickers = 46)
  (students_eq : students = 5)
  (total_stickers_given_eq : total_stickers_given = students * each_student_stickers)
  (total_gold_and_silver_stickers_eq : total_gold_and_silver_stickers = gold_stickers + silver_stickers)
  (bronze_stickers_eq : bronze_stickers = total_stickers_given - total_gold_and_silver_stickers) :
  silver_stickers - bronze_stickers = 20 :=
by
  sorry

end fewer_bronze_stickers_l395_39595


namespace geometric_series_sum_l395_39561

theorem geometric_series_sum :
  let a := 2
  let r := -2
  let n := 10
  let Sn := (a : ℚ) * (r^n - 1) / (r - 1)
  Sn = 2050 / 3 :=
by
  sorry

end geometric_series_sum_l395_39561


namespace bacteria_reaches_final_in_24_hours_l395_39511

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 200

-- Define the final number of bacteria
def final_bacteria : ℕ := 16200

-- Define the tripling period in hours
def tripling_period : ℕ := 6

-- Define the tripling factor
def tripling_factor : ℕ := 3

-- Define the number of hours needed to reach final number of bacteria
def hours_to_reach_final_bacteria : ℕ := 24

-- Define a function that models the number of bacteria after t hours
def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * tripling_factor^((t / tripling_period))

-- Main statement of the problem: prove that the number of bacteria is 16200 after 24 hours
theorem bacteria_reaches_final_in_24_hours :
  bacteria_after hours_to_reach_final_bacteria = final_bacteria :=
sorry

end bacteria_reaches_final_in_24_hours_l395_39511


namespace largest_prime_value_of_quadratic_expression_l395_39567

theorem largest_prime_value_of_quadratic_expression : 
  ∃ n : ℕ, n > 0 ∧ Prime (n^2 - 12 * n + 27) ∧ ∀ m : ℕ, m > 0 → Prime (m^2 - 12 * m + 27) → (n^2 - 12 * n + 27) ≥ (m^2 - 12 * m + 27) := 
by
  sorry


end largest_prime_value_of_quadratic_expression_l395_39567


namespace integer_solutions_count_l395_39515

theorem integer_solutions_count :
  ∃ (count : ℤ), (∀ (a : ℤ), 
  (∃ x : ℤ, x^2 + a * x + 8 * a = 0) ↔ count = 8) :=
sorry

end integer_solutions_count_l395_39515


namespace size_of_first_file_l395_39524

theorem size_of_first_file (internet_speed_mbps : ℝ) (time_hours : ℝ) (file2_mbps : ℝ) (file3_mbps : ℝ) (total_downloaded_mbps : ℝ) :
  internet_speed_mbps = 2 →
  time_hours = 2 →
  file2_mbps = 90 →
  file3_mbps = 70 →
  total_downloaded_mbps = internet_speed_mbps * 60 * time_hours →
  total_downloaded_mbps - (file2_mbps + file3_mbps) = 80 :=
by
  intros
  sorry

end size_of_first_file_l395_39524


namespace cheesecake_total_calories_l395_39599

-- Define the conditions
def slice_calories : ℕ := 350

def percent_eaten : ℕ := 25
def slices_eaten : ℕ := 2

-- Define the total number of slices in a cheesecake
def total_slices (percent_eaten slices_eaten : ℕ) : ℕ :=
  slices_eaten * (100 / percent_eaten)

-- Define the total calories in a cheesecake given the above conditions
def total_calories (slice_calories slices : ℕ) : ℕ :=
  slice_calories * slices

-- State the theorem
theorem cheesecake_total_calories :
  total_calories slice_calories (total_slices percent_eaten slices_eaten) = 2800 :=
by
  sorry

end cheesecake_total_calories_l395_39599


namespace true_supporters_of_rostov_l395_39526

theorem true_supporters_of_rostov
  (knights_liars_fraction : ℕ → ℕ)
  (rostov_support_yes : ℕ)
  (zenit_support_yes : ℕ)
  (lokomotiv_support_yes : ℕ)
  (cska_support_yes : ℕ)
  (h1 : knights_liars_fraction 100 = 10)
  (h2 : rostov_support_yes = 40)
  (h3 : zenit_support_yes = 30)
  (h4 : lokomotiv_support_yes = 50)
  (h5 : cska_support_yes = 0):
  rostov_support_yes - knights_liars_fraction 100 = 30 := 
sorry

end true_supporters_of_rostov_l395_39526


namespace ratio_of_amount_divided_to_total_savings_is_half_l395_39552

theorem ratio_of_amount_divided_to_total_savings_is_half :
  let husband_weekly_contribution := 335
  let wife_weekly_contribution := 225
  let weeks_in_six_months := 6 * 4
  let total_weekly_contribution := husband_weekly_contribution + wife_weekly_contribution
  let total_savings := total_weekly_contribution * weeks_in_six_months
  let amount_per_child := 1680
  let number_of_children := 4
  let total_amount_divided := amount_per_child * number_of_children
  (total_amount_divided : ℝ) / total_savings = 0.5 := 
by
  sorry

end ratio_of_amount_divided_to_total_savings_is_half_l395_39552


namespace air_conditioner_sales_l395_39588

/-- Represent the conditions -/
def conditions (x y m : ℕ) : Prop :=
  (3 * x + 5 * y = 23500) ∧
  (4 * x + 10 * y = 42000) ∧
  (x = 2500) ∧
  (y = 3200) ∧
  (700 * (50 - m) + 800 * m ≥ 38000)

/-- Prove that the unit selling prices of models A and B are 2500 yuan and 3200 yuan respectively,
    and at least 30 units of model B need to be purchased for a profit of at least 38000 yuan,
    given the conditions. -/
theorem air_conditioner_sales :
  ∃ (x y m : ℕ), conditions x y m ∧ m ≥ 30 := by
  sorry

end air_conditioner_sales_l395_39588


namespace length_of_rectangle_l395_39562

-- Given conditions as per the problem statement
variables {s l : ℝ} -- side length of the square, length of the rectangle
def width_rectangle : ℝ := 10 -- width of the rectangle

-- Conditions
axiom sq_perimeter : 4 * s = 200
axiom area_relation : s^2 = 5 * (l * width_rectangle)

-- Goal to prove
theorem length_of_rectangle : l = 50 :=
by
  sorry

end length_of_rectangle_l395_39562


namespace product_of_a_and_b_is_zero_l395_39516

theorem product_of_a_and_b_is_zero
  (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)
  (h2 : b < 10)
  (h3 : a * (b + 10) = 190) :
  a * b = 0 :=
sorry

end product_of_a_and_b_is_zero_l395_39516
