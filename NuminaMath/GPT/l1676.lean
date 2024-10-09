import Mathlib

namespace equalize_champagne_futile_l1676_167669

/-- Stepashka cannot distribute champagne into 2018 glasses in such a way 
that Kryusha's attempts to equalize the amount in all glasses become futile. -/
theorem equalize_champagne_futile (n : ℕ) (h : n = 2018) : 
∃ (a : ℕ), (∀ (A B : ℕ), A ≠ B ∧ A + B = 2019 → (A + B) % 2 = 1) := 
sorry

end equalize_champagne_futile_l1676_167669


namespace scrap_rate_independence_l1676_167617

theorem scrap_rate_independence (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - (1 - a) * (1 - b)) = 1 - (1 - a) * (1 - b) :=
by
  sorry

end scrap_rate_independence_l1676_167617


namespace y_is_75_percent_of_x_l1676_167619

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := 0.45 * z = 0.72 * y
def condition2 : Prop := z = 1.20 * x

-- Theorem to prove y = 0.75 * x
theorem y_is_75_percent_of_x (h1 : condition1 z y) (h2 : condition2 x z) : y = 0.75 * x :=
by sorry

end y_is_75_percent_of_x_l1676_167619


namespace number_of_round_trips_each_bird_made_l1676_167694

theorem number_of_round_trips_each_bird_made
  (distance_to_materials : ℕ)
  (total_distance_covered : ℕ)
  (distance_one_round_trip : ℕ)
  (total_number_of_trips : ℕ)
  (individual_bird_trips : ℕ) :
  distance_to_materials = 200 →
  total_distance_covered = 8000 →
  distance_one_round_trip = 2 * distance_to_materials →
  total_number_of_trips = total_distance_covered / distance_one_round_trip →
  individual_bird_trips = total_number_of_trips / 2 →
  individual_bird_trips = 10 :=
by
  intros
  sorry

end number_of_round_trips_each_bird_made_l1676_167694


namespace relationship_bx_x2_a2_l1676_167611

theorem relationship_bx_x2_a2 {a b x : ℝ} (h1 : b < x) (h2 : x < a) (h3 : 0 < a) (h4 : 0 < b) : 
  b * x < x^2 ∧ x^2 < a^2 :=
by sorry

end relationship_bx_x2_a2_l1676_167611


namespace volume_is_correct_l1676_167605

def volume_of_box (x : ℝ) : ℝ :=
  (14 - 2 * x) * (10 - 2 * x) * x

theorem volume_is_correct (x : ℝ) :
  volume_of_box x = 140 * x - 48 * x^2 + 4 * x^3 :=
by
  sorry

end volume_is_correct_l1676_167605


namespace right_triangle_hypotenuse_l1676_167651

theorem right_triangle_hypotenuse :
  ∃ b a : ℕ, a^2 + 1994^2 = b^2 ∧ b = 994010 :=
by
  sorry

end right_triangle_hypotenuse_l1676_167651


namespace number_exceeds_its_part_l1676_167634

theorem number_exceeds_its_part (x : ℝ) (h : x = 3/8 * x + 25) : x = 40 :=
by sorry

end number_exceeds_its_part_l1676_167634


namespace sum_of_ages_l1676_167696

theorem sum_of_ages (a b c : ℕ) (h1 : a * b * c = 72) (h2 : b = c) (h3 : a < b) : a + b + c = 14 :=
sorry

end sum_of_ages_l1676_167696


namespace ex1_simplified_ex2_simplified_l1676_167660

-- Definitions and problem setup
def ex1 (a : ℝ) : ℝ := ((-a^3)^2 * a^3 - 4 * a^2 * a^7)
def ex2 (a : ℝ) : ℝ := (2 * a + 1) * (-2 * a + 1)

-- Proof goals
theorem ex1_simplified (a : ℝ) : ex1 a = -3 * a^9 :=
by sorry

theorem ex2_simplified (a : ℝ) : ex2 a = 4 * a^2 - 1 :=
by sorry

end ex1_simplified_ex2_simplified_l1676_167660


namespace list_price_is_35_l1676_167675

-- Define the conditions in Lean
variable (x : ℝ)

def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * (alice_selling_price x)

def bob_selling_price (x : ℝ) : ℝ := x - 20
def bob_commission (x : ℝ) : ℝ := 0.20 * (bob_selling_price x)

-- Define the theorem to be proven
theorem list_price_is_35 (x : ℝ) 
  (h : alice_commission x = bob_commission x) : x = 35 :=
by sorry

end list_price_is_35_l1676_167675


namespace degree_to_radian_l1676_167624

theorem degree_to_radian (deg : ℝ) (h : deg = 50) : deg * (Real.pi / 180) = (5 / 18) * Real.pi :=
by
  -- placeholder for the proof
  sorry

end degree_to_radian_l1676_167624


namespace operation_result_l1676_167670

def a : ℝ := 0.8
def b : ℝ := 0.5
def c : ℝ := 0.40

theorem operation_result :
  (a ^ 3 - b ^ 3 / a ^ 2 + c + b ^ 2) = 0.9666875 := by
  sorry

end operation_result_l1676_167670


namespace number_of_ordered_triples_l1676_167616

theorem number_of_ordered_triples :
  ∃ n, (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.lcm a b = 12 ∧ Nat.gcd b c = 6 ∧ Nat.lcm c a = 24) ∧ n = 4 :=
sorry

end number_of_ordered_triples_l1676_167616


namespace calculate_f3_minus_f4_l1676_167630

-- Defining the function f and the given conditions
variables (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (periodic_f : ∀ x, f (x + 2) = -f x)
variable (f1 : f 1 = 1)

-- Proving the required equality
theorem calculate_f3_minus_f4 : f 3 - f 4 = -1 :=
by
  sorry

end calculate_f3_minus_f4_l1676_167630


namespace ratio_of_areas_l1676_167635

theorem ratio_of_areas (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
    let S₁ := (1 - p * q * r) * (1 - p * q * r)
    let S₂ := (1 + p + p * q) * (1 + q + q * r) * (1 + r + r * p)
    S₁ / S₂ = (S₁ / S₂) := sorry

end ratio_of_areas_l1676_167635


namespace algebraic_expression_value_l1676_167621

variable (x y A B : ℤ)
variable (x_val : x = -1)
variable (y_val : y = 2)
variable (A_def : A = 2*x + y)
variable (B_def : B = 2*x - y)

theorem algebraic_expression_value : 
  (A^2 - B^2) * (x - 2*y) = 80 := 
by
  rw [x_val, y_val, A_def, B_def]
  sorry

end algebraic_expression_value_l1676_167621


namespace smallest_possible_value_l1676_167649

-- Definitions of the digits
def P := 1
def A := 9
def B := 2
def H := 8
def O := 3

-- Expression for continued fraction T
noncomputable def T : ℚ :=
  P + 1 / (A + 1 / (B + 1 / (H + 1 / O)))

-- The goal is to prove that T is the smallest possible value given the conditions
theorem smallest_possible_value : T = 555 / 502 :=
by
  -- The detailed proof would be done here, but for now we use sorry because we only need the statement
  sorry

end smallest_possible_value_l1676_167649


namespace polynomial_remainder_l1676_167641

theorem polynomial_remainder (y : ℂ) (h1 : y^5 + y^4 + y^3 + y^2 + y + 1 = 0) (h2 : y^6 = 1) :
  (y^55 + y^40 + y^25 + y^10 + 1) % (y^5 + y^4 + y^3 + y^2 + y + 1) = 2 * y + 3 :=
sorry

end polynomial_remainder_l1676_167641


namespace remainder_division_l1676_167678

theorem remainder_division (N : ℤ) (R1 : ℤ) (Q2 : ℤ) 
  (h1 : N = 44 * 432 + R1)
  (h2 : N = 38 * Q2 + 8) : 
  R1 = 0 := by
  sorry

end remainder_division_l1676_167678


namespace evenness_oddness_of_f_min_value_of_f_l1676_167648

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + |x - a| + 1

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem evenness_oddness_of_f (a : ℝ) :
  (is_even (f a) ↔ a = 0) ∧ (a ≠ 0 → ¬ is_even (f a) ∧ ¬ is_odd (f a)) :=
by
  sorry

theorem min_value_of_f (a x : ℝ) (h : x ≥ a) :
  (a ≤ -1 / 2 → f a x = 3 / 4 - a) ∧ (a > -1 / 2 → f a x = a^2 + 1) :=
by
  sorry

end evenness_oddness_of_f_min_value_of_f_l1676_167648


namespace exists_non_decreasing_subsequences_l1676_167653

theorem exists_non_decreasing_subsequences {a b c : ℕ → ℕ} : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_non_decreasing_subsequences_l1676_167653


namespace base_of_numbering_system_l1676_167666

-- Definitions based on conditions
def num_children := 100
def num_boys := 24
def num_girls := 32

-- Problem statement: Prove the base of numbering system used is 6
theorem base_of_numbering_system (n: ℕ) (h: n ≠ 0):
    n^2 = (2 * n + 4) + (3 * n + 2) → n = 6 := 
  by
    sorry

end base_of_numbering_system_l1676_167666


namespace box_height_l1676_167645

theorem box_height (x h : ℕ) 
  (h1 : h = x + 5) 
  (h2 : 6 * x^2 + 20 * x ≥ 150) 
  (h3 : 5 * x + 5 ≥ 25) 
  : h = 9 :=
by 
  sorry

end box_height_l1676_167645


namespace number_of_sodas_l1676_167674

theorem number_of_sodas (cost_sandwich : ℝ) (num_sandwiches : ℕ) (cost_soda : ℝ) (total_cost : ℝ):
  cost_sandwich = 2.45 → 
  num_sandwiches = 2 → 
  cost_soda = 0.87 → 
  total_cost = 8.38 → 
  (total_cost - num_sandwiches * cost_sandwich) / cost_soda = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end number_of_sodas_l1676_167674


namespace compare_abc_l1676_167659

theorem compare_abc (a b c : ℝ)
  (h1 : a = Real.log 0.9 / Real.log 2)
  (h2 : b = 3 ^ (-1 / 3 : ℝ))
  (h3 : c = (1 / 3 : ℝ) ^ (1 / 2 : ℝ)) :
  a < c ∧ c < b := by
  sorry

end compare_abc_l1676_167659


namespace average_speed_of_car_l1676_167667

-- Definitions of the given conditions
def uphill_speed : ℝ := 30  -- km/hr
def downhill_speed : ℝ := 70  -- km/hr
def uphill_distance : ℝ := 100  -- km
def downhill_distance : ℝ := 50  -- km

-- Required proof statement (with the correct answer derived from the conditions)
theorem average_speed_of_car :
  (uphill_distance + downhill_distance) / 
  ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = 37.04 := by
  sorry

end average_speed_of_car_l1676_167667


namespace combined_ticket_cost_l1676_167652

variables (S K : ℕ)

theorem combined_ticket_cost (total_budget : ℕ) (samuel_food_drink : ℕ) (kevin_food : ℕ) (kevin_drink : ℕ) :
  total_budget = 20 →
  samuel_food_drink = 6 →
  kevin_food = 4 →
  kevin_drink = 2 →
  S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget →
  S + K = 8 :=
by
  intros h_total_budget h_samuel_food_drink h_kevin_food h_kevin_drink h_total_spent
  /-
  We have the following conditions:
  1. total_budget = 20
  2. samuel_food_drink = 6
  3. kevin_food = 4
  4. kevin_drink = 2
  5. S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget

  We need to prove that S + K = 8. We can use the conditions to derive this.
  -/
  rw [h_total_budget, h_samuel_food_drink, h_kevin_food, h_kevin_drink] at h_total_spent
  exact sorry

end combined_ticket_cost_l1676_167652


namespace eight_painters_finish_in_required_days_l1676_167689

/- Conditions setup -/
def initial_painters : ℕ := 6
def initial_days : ℕ := 2
def job_constant := initial_painters * initial_days

def new_painters : ℕ := 8
def required_days := 3 / 2

/- Theorem statement -/
theorem eight_painters_finish_in_required_days : new_painters * required_days = job_constant :=
sorry

end eight_painters_finish_in_required_days_l1676_167689


namespace intersection_of_A_and_B_l1676_167655

def A : Set ℚ := { x | x^2 - 4*x + 3 < 0 }
def B : Set ℚ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end intersection_of_A_and_B_l1676_167655


namespace largest_angle_in_triangle_PQR_is_75_degrees_l1676_167614

noncomputable def largest_angle (p q r : ℝ) : ℝ :=
  if p + q + 2 * r = p^2 ∧ p + q - 2 * r = -1 then 
    Real.arccos ((p^2 + q^2 - (p^2 + p*q + (1/2)*q^2)/2) / (2 * p * q)) * (180/Real.pi)
  else 
    0

theorem largest_angle_in_triangle_PQR_is_75_degrees (p q r : ℝ) (h1 : p + q + 2 * r = p^2) (h2 : p + q - 2 * r = -1) :
  largest_angle p q r = 75 :=
by sorry

end largest_angle_in_triangle_PQR_is_75_degrees_l1676_167614


namespace perfect_square_polynomial_l1676_167608

theorem perfect_square_polynomial (m : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, x^2 - (m + 1) * x + 1 = (f x) * (f x)) → (m = 1 ∨ m = -3) :=
by
  sorry

end perfect_square_polynomial_l1676_167608


namespace remainder_when_divided_l1676_167644

open Polynomial

noncomputable def poly : Polynomial ℚ := X^6 + X^5 + 2*X^3 - X^2 + 3
noncomputable def divisor : Polynomial ℚ := (X + 2) * (X - 1)
noncomputable def remainder : Polynomial ℚ := -X + 5

theorem remainder_when_divided :
  ∃ q : Polynomial ℚ, poly = divisor * q + remainder :=
sorry

end remainder_when_divided_l1676_167644


namespace smallest_two_digit_multiple_of_17_l1676_167625

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ n % 17 = 0 ∧ ∀ m, (10 ≤ m ∧ m < n ∧ m % 17 = 0) → false := sorry

end smallest_two_digit_multiple_of_17_l1676_167625


namespace exponential_rule_l1676_167633

theorem exponential_rule (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=  
  sorry

end exponential_rule_l1676_167633


namespace rectangle_area_l1676_167692

/-- 
In the rectangle \(ABCD\), \(AD - AB = 9\) cm. The area of trapezoid \(ABCE\) is 5 times 
the area of triangle \(ADE\). The perimeter of triangle \(ADE\) is 68 cm less than the 
perimeter of trapezoid \(ABCE\). Prove that the area of the rectangle \(ABCD\) 
is 3060 square centimeters.
-/
theorem rectangle_area (AB AD : ℝ) (S_ABC : ℝ) (S_ADE : ℝ) (P_ADE : ℝ) (P_ABC : ℝ) :
  AD - AB = 9 →
  S_ABC = 5 * S_ADE →
  P_ADE = P_ABC - 68 →
  (AB * AD = 3060) :=
by
  sorry

end rectangle_area_l1676_167692


namespace elementary_school_coats_correct_l1676_167647

def total_coats : ℕ := 9437
def high_school_coats : ℕ := (3 * total_coats) / 5
def elementary_school_coats := total_coats - high_school_coats

theorem elementary_school_coats_correct : 
  elementary_school_coats = 3775 :=
by
  sorry

end elementary_school_coats_correct_l1676_167647


namespace minimum_score_4th_quarter_l1676_167638

theorem minimum_score_4th_quarter (q1 q2 q3 : ℕ) (q4 : ℕ) :
  q1 = 85 → q2 = 80 → q3 = 90 →
  (q1 + q2 + q3 + q4) / 4 ≥ 85 →
  q4 ≥ 85 :=
by intros hq1 hq2 hq3 h_avg
   sorry

end minimum_score_4th_quarter_l1676_167638


namespace factorize_quadratic_l1676_167681

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end factorize_quadratic_l1676_167681


namespace A_alone_finishes_in_27_days_l1676_167687

noncomputable def work (B : ℝ) : ℝ := 54 * B  -- amount of work W
noncomputable def days_to_finish_alone (B : ℝ) : ℝ := (work B) / (2 * B)

theorem A_alone_finishes_in_27_days (B : ℝ) (h : (work B) / (2 * B + B) = 18) : 
  days_to_finish_alone B = 27 :=
by
  sorry

end A_alone_finishes_in_27_days_l1676_167687


namespace total_profit_l1676_167693

-- Definitions
def investment_a : ℝ := 45000
def investment_b : ℝ := 63000
def investment_c : ℝ := 72000
def c_share : ℝ := 24000

-- Theorem statement
theorem total_profit : (investment_a + investment_b + investment_c) * (c_share / investment_c) = 60000 := by
  sorry

end total_profit_l1676_167693


namespace sale_in_fifth_month_l1676_167620

-- Define the sales in the first, second, third, fourth, and sixth months
def a1 : ℕ := 7435
def a2 : ℕ := 7927
def a3 : ℕ := 7855
def a4 : ℕ := 8230
def a6 : ℕ := 5991

-- Define the average sale
def avg_sale : ℕ := 7500

-- Define the number of months
def months : ℕ := 6

-- The total sales required for the average sale to be 7500 over 6 months.
def total_sales : ℕ := avg_sale * months

-- Calculate the sales in the first four months
def sales_first_four_months : ℕ := a1 + a2 + a3 + a4

-- Calculate the total sales for the first four months plus the sixth month.
def sales_first_four_and_sixth : ℕ := sales_first_four_months + a6

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : ∃ a5 : ℕ, total_sales = sales_first_four_and_sixth + a5 ∧ a5 = 7562 :=
by
  sorry


end sale_in_fifth_month_l1676_167620


namespace crocodiles_count_l1676_167618

-- Definitions of constants
def alligators : Nat := 23
def vipers : Nat := 5
def total_dangerous_animals : Nat := 50

-- Theorem statement
theorem crocodiles_count :
  total_dangerous_animals - alligators - vipers = 22 :=
by
  sorry

end crocodiles_count_l1676_167618


namespace gage_skating_time_l1676_167662

theorem gage_skating_time :
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  minutes_needed_ninth_day = 120 :=
by
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  sorry

end gage_skating_time_l1676_167662


namespace jacob_find_more_l1676_167606

theorem jacob_find_more :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let total_shells := 30
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + initial_shells
  let jacob_shells := total_shells - ed_shells
  (jacob_shells - ed_limpet_shells - ed_oyster_shells - ed_conch_shells = 2) := 
by 
  sorry

end jacob_find_more_l1676_167606


namespace xiaoming_money_l1676_167602

open Real

noncomputable def verify_money_left (M P_L : ℝ) : Prop := M = 12 * P_L

noncomputable def verify_money_right (M P_R : ℝ) : Prop := M = 14 * P_R

noncomputable def price_relationship (P_L P_R : ℝ) : Prop := P_R = P_L - 1

theorem xiaoming_money (M P_L P_R : ℝ) 
  (h1 : verify_money_left M P_L) 
  (h2 : verify_money_right M P_R) 
  (h3 : price_relationship P_L P_R) : 
  M = 84 := 
  by
  sorry

end xiaoming_money_l1676_167602


namespace inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l1676_167615

theorem inequality_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 :=
by sorry

theorem equality_conditions_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2
  ↔ (a = 0 ∨ b = 0 ∨ x = y) :=
by sorry

end inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l1676_167615


namespace female_salmon_returned_l1676_167656

/-- The number of female salmon that returned to their rivers is 259378,
    given that the total number of salmon that made the trip is 971639 and
    the number of male salmon that returned is 712261. -/
theorem female_salmon_returned :
  let n := 971639
  let m := 712261
  let f := n - m
  f = 259378 :=
by
  rfl

end female_salmon_returned_l1676_167656


namespace no_three_positive_reals_l1676_167663

noncomputable def S (a : ℝ) : Set ℕ := { n | ∃ (k : ℕ), n = ⌊(k : ℝ) * a⌋ }

theorem no_three_positive_reals (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧ (S a ∪ S b ∪ S c = Set.univ) → false :=
sorry

end no_three_positive_reals_l1676_167663


namespace train_times_valid_l1676_167668

-- Define the parameters and conditions
def trainA_usual_time : ℝ := 180 -- minutes
def trainB_travel_time : ℝ := 810 -- minutes

theorem train_times_valid (t : ℝ) (T_B : ℝ) 
  (cond1 : (7 / 6) * t = t + 30)
  (cond2 : T_B = 4.5 * t) : 
  t = trainA_usual_time ∧ T_B = trainB_travel_time :=
by
  sorry

end train_times_valid_l1676_167668


namespace relationship_between_number_and_square_l1676_167677

theorem relationship_between_number_and_square (n : ℕ) (h : n = 9) :
  (n + n^2) / 2 = 5 * n := by
    sorry

end relationship_between_number_and_square_l1676_167677


namespace marbles_left_l1676_167690

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem marbles_left : (initial_marbles - marbles_given) = 50 := by
  sorry

end marbles_left_l1676_167690


namespace range_of_m_if_not_p_and_q_l1676_167600

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_if_not_p_and_q_l1676_167600


namespace identify_set_A_l1676_167680

open Set

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem identify_set_A : A = {0, 1, 2} := 
by
  sorry

end identify_set_A_l1676_167680


namespace solution_set_of_inequality_l1676_167610

theorem solution_set_of_inequality : 
  { x : ℝ | (3 - 2 * x) * (x + 1) ≤ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 / 2 } :=
sorry

end solution_set_of_inequality_l1676_167610


namespace find_subtracted_number_l1676_167665

theorem find_subtracted_number (t k x : ℝ) (h1 : t = 20) (h2 : k = 68) (h3 : t = 5/9 * (k - x)) :
  x = 32 :=
by
  sorry

end find_subtracted_number_l1676_167665


namespace total_votes_is_120_l1676_167695

-- Define the conditions
def Fiona_votes : ℕ := 48
def fraction_of_votes : ℚ := 2 / 5

-- The proof goal
theorem total_votes_is_120 (V : ℕ) (h : Fiona_votes = fraction_of_votes * V) : V = 120 :=
by
  sorry

end total_votes_is_120_l1676_167695


namespace valid_four_digit_number_count_l1676_167642

theorem valid_four_digit_number_count : 
  let first_digit_choices := 6 
  let last_digit_choices := 10 
  let middle_digits_valid_pairs := 9 * 9 - 18
  (first_digit_choices * middle_digits_valid_pairs * last_digit_choices = 3780) := by
  sorry

end valid_four_digit_number_count_l1676_167642


namespace work_problem_l1676_167646

theorem work_problem 
  (A_real : ℝ)
  (B_days : ℝ := 16)
  (C_days : ℝ := 16)
  (ABC_days : ℝ := 4)
  (H_b : (1 / B_days) = 1 / 16)
  (H_c : (1 / C_days) = 1 / 16)
  (H_abc : (1 / A_real + 1 / B_days + 1 / C_days) = 1 / ABC_days) : 
  A_real = 8 := 
sorry

end work_problem_l1676_167646


namespace speed_of_water_l1676_167639

-- Definitions based on conditions
def swim_speed_in_still_water : ℝ := 4
def distance_against_current : ℝ := 6
def time_against_current : ℝ := 3
def effective_speed (v : ℝ) : ℝ := swim_speed_in_still_water - v

-- Theorem to prove the speed of the water
theorem speed_of_water (v : ℝ) : 
  effective_speed v * time_against_current = distance_against_current → 
  v = 2 :=
by
  sorry

end speed_of_water_l1676_167639


namespace total_rainfall_2010_to_2012_l1676_167699

noncomputable def average_rainfall (year : ℕ) : ℕ :=
  if year = 2010 then 35
  else if year = 2011 then 38
  else if year = 2012 then 41
  else 0

theorem total_rainfall_2010_to_2012 :
  (12 * average_rainfall 2010) + 
  (12 * average_rainfall 2011) + 
  (12 * average_rainfall 2012) = 1368 :=
by
  sorry

end total_rainfall_2010_to_2012_l1676_167699


namespace evaluate_fraction_l1676_167626

open Complex

theorem evaluate_fraction (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 - a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 1 / 18 := by
  sorry

end evaluate_fraction_l1676_167626


namespace relationship_y_values_l1676_167654

theorem relationship_y_values (x1 x2 y1 y2 : ℝ) (h1 : x1 > x2) (h2 : 0 < x2) (h3 : y1 = - (3 / x1)) (h4 : y2 = - (3 / x2)) : y1 > y2 :=
by
  sorry

end relationship_y_values_l1676_167654


namespace cost_formula_l1676_167691

-- Definitions based on conditions
def base_cost : ℕ := 15
def additional_cost_per_pound : ℕ := 5
def environmental_fee : ℕ := 2

-- Definition of cost function
def cost (P : ℕ) : ℕ := base_cost + additional_cost_per_pound * (P - 1) + environmental_fee

-- Theorem stating the formula for the cost C
theorem cost_formula (P : ℕ) (h : 1 ≤ P) : cost P = 12 + 5 * P :=
by
  -- Proof would go here
  sorry

end cost_formula_l1676_167691


namespace sharks_problem_l1676_167683

variable (F : ℝ)
variable (S : ℝ := 0.25 * (F + 3 * F))
variable (total_sharks : ℝ := 15)

theorem sharks_problem : 
  (0.25 * (F + 3 * F) = 15) ↔ (F = 15) :=
by 
  sorry

end sharks_problem_l1676_167683


namespace a8_value_l1676_167623

def sequence_sum (n : ℕ) : ℕ := 2^n - 1

def nth_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem a8_value : nth_term sequence_sum 8 = 128 :=
by
  -- Proof goes here
  sorry

end a8_value_l1676_167623


namespace fraction_product_l1676_167658

theorem fraction_product (a b : ℕ) 
  (h1 : 1/5 < a / b)
  (h2 : a / b < 1/4)
  (h3 : b ≤ 19) :
  ∃ a1 a2 b1 b2, 4 * a2 < b1 ∧ b1 < 5 * a2 ∧ b2 ≤ 19 ∧ 4 * a2 < b2 ∧ b2 < 20 ∧ a = 4 ∧ b = 19 ∧ a1 = 2 ∧ b1 = 9 ∧ 
  (a + b = 23 ∨ a + b = 11) ∧ (23 * 11 = 253) := by
  sorry

end fraction_product_l1676_167658


namespace cookie_radius_l1676_167622

theorem cookie_radius (x y : ℝ) : x^2 + y^2 + 28 = 6*x + 20*y → ∃ r, r = 9 :=
by
  sorry

end cookie_radius_l1676_167622


namespace max_number_of_girls_l1676_167664

theorem max_number_of_girls (students : ℕ)
  (num_friends : ℕ → ℕ)
  (h_students : students = 25)
  (h_distinct_friends : ∀ (i j : ℕ), i ≠ j → num_friends i ≠ num_friends j)
  (h_girls_boys : ∃ (G B : ℕ), G + B = students) :
  ∃ G : ℕ, G = 13 := 
sorry

end max_number_of_girls_l1676_167664


namespace find_a_plus_b_plus_c_l1676_167676

-- Definitions of conditions
def is_vertex (a b c : ℝ) (vertex_x vertex_y : ℝ) := 
  ∀ x : ℝ, vertex_y = (a * (vertex_x ^ 2)) + (b * vertex_x) + c

def contains_point (a b c : ℝ) (x y : ℝ) := 
  y = (a * (x ^ 2)) + (b * x) + c

theorem find_a_plus_b_plus_c
  (a b c : ℝ)
  (h_vertex : is_vertex a b c 3 4)
  (h_symmetry : ∃ h : ℝ, ∀ x : ℝ, a * (x - h) ^ 2 = a * (h - x) ^ 2)
  (h_contains : contains_point a b c 1 0)
  : a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l1676_167676


namespace find_second_number_l1676_167688

theorem find_second_number (x y z : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x = (3/4) * y) 
  (h3 : z = (9/7) * y) 
  : y = 40 :=
sorry

end find_second_number_l1676_167688


namespace simplify_expr1_simplify_expr2_l1676_167661

-- Expression simplification proof statement 1
theorem simplify_expr1 (m n : ℤ) : 
  (5 * m + 3 * n - 7 * m - n) = (-2 * m + 2 * n) :=
sorry

-- Expression simplification proof statement 2
theorem simplify_expr2 (x : ℤ) : 
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2)) = (2 * x^2 - 5 * x + 6) :=
sorry

end simplify_expr1_simplify_expr2_l1676_167661


namespace smallest_b_l1676_167679

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) 
(h3 : 2 + a ≤ b) (h4 : 1 / a + 1 / b ≤ 2) : b = 2 :=
sorry

end smallest_b_l1676_167679


namespace find_digits_of_abc_l1676_167657

theorem find_digits_of_abc (a b c : ℕ) (h1 : a ≠ c) (h2 : c - a = 3) (h3 : (100 * a + 10 * b + c) - (100 * c + 10 * a + b) = 100 * (a - (c - 1)) + 0 + (b - b)) : 
  100 * a + 10 * b + c = 619 :=
by
  sorry

end find_digits_of_abc_l1676_167657


namespace sherman_weekend_driving_time_l1676_167672

def total_driving_time_per_week : ℕ := 9
def commute_time_per_day : ℕ := 1
def work_days_per_week : ℕ := 5
def weekend_days : ℕ := 2

theorem sherman_weekend_driving_time :
  (total_driving_time_per_week - commute_time_per_day * work_days_per_week) / weekend_days = 2 :=
sorry

end sherman_weekend_driving_time_l1676_167672


namespace garden_perimeter_l1676_167609

-- Definitions for length and breadth
def length := 150
def breadth := 100

-- Theorem that states the perimeter of the rectangular garden
theorem garden_perimeter : (2 * (length + breadth)) = 500 :=
by sorry

end garden_perimeter_l1676_167609


namespace sum_of_arithmetic_sequence_is_54_l1676_167629

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence_is_54 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 8 = 6 + a 11) : 
  S 9 = 54 :=
sorry

end sum_of_arithmetic_sequence_is_54_l1676_167629


namespace circle_equation_exists_l1676_167631

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)
def l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0
def is_on_circle (C : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2

theorem circle_equation_exists :
  ∃ C : ℝ × ℝ, C.1 - C.2 + 1 = 0 ∧
  (is_on_circle C A 5) ∧
  (is_on_circle C B 5) ∧
  is_on_circle C (-3, -2) 5 :=
sorry

end circle_equation_exists_l1676_167631


namespace mod_pow_solution_l1676_167685

def m (x : ℕ) := x

theorem mod_pow_solution :
  ∃ (m : ℕ), 0 ≤ m ∧ m < 8 ∧ 13^6 % 8 = m ∧ m = 1 :=
by
  use 1
  sorry

end mod_pow_solution_l1676_167685


namespace polygon_sides_16_l1676_167650

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

noncomputable def arithmetic_sequence_sum (a1 an : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a1 + an) / 2

theorem polygon_sides_16 (n : ℕ) (a1 an : ℝ) (d : ℝ) 
  (h1 : d = 5) (h2 : an = 160) (h3 : a1 = 160 - 5 * (n - 1))
  (h4 : arithmetic_sequence_sum a1 an d n = sum_of_interior_angles n)
  : n = 16 :=
sorry

end polygon_sides_16_l1676_167650


namespace square_perimeter_l1676_167640

theorem square_perimeter (s : ℝ)
  (h1 : ∃ (s : ℝ), 4 * s = s * 1 + s / 4 * 1 + s * 1 + s / 4 * 1)
  (h2 : ∃ (P : ℝ), P = 4 * s)
  : (5/2) * s = 40 → 4 * s = 64 :=
by
  intro h
  sorry

end square_perimeter_l1676_167640


namespace problem_l1676_167604

variable (m : ℝ)

def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

theorem problem (hpq : ¬ (p m ∧ q m)) (hlpq : p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end problem_l1676_167604


namespace Steve_pencils_left_l1676_167628

-- Define the initial number of boxes and pencils per box
def boxes := 2
def pencils_per_box := 12
def initial_pencils := boxes * pencils_per_box

-- Define the number of pencils given to Lauren and the additional pencils given to Matt
def pencils_to_Lauren := 6
def diff_Lauren_Matt := 3
def pencils_to_Matt := pencils_to_Lauren + diff_Lauren_Matt

-- Calculate the total pencils given away
def pencils_given_away := pencils_to_Lauren + pencils_to_Matt

-- Number of pencils left with Steve
def pencils_left := initial_pencils - pencils_given_away

-- The statement to prove
theorem Steve_pencils_left : pencils_left = 9 := by
  sorry

end Steve_pencils_left_l1676_167628


namespace union_of_A_and_B_l1676_167636

-- Define the sets A and B
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

-- Prove that the union of A and B is {-1, 0, 1}
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} :=
  by sorry

end union_of_A_and_B_l1676_167636


namespace find_general_students_l1676_167684

-- Define the conditions and the question
structure Halls :=
  (general : ℕ)
  (biology : ℕ)
  (math : ℕ)
  (total : ℕ)

def conditions_met (h : Halls) : Prop :=
  h.biology = 2 * h.general ∧
  h.math = (3 / 5 : ℚ) * (h.general + h.biology) ∧
  h.total = h.general + h.biology + h.math ∧
  h.total = 144

-- The proof problem statement
theorem find_general_students (h : Halls) (h_cond : conditions_met h) : h.general = 30 :=
sorry

end find_general_students_l1676_167684


namespace weight_of_new_boy_l1676_167637

theorem weight_of_new_boy (W : ℕ) (original_weight : ℕ) (total_new_weight : ℕ)
  (h_original_avg : original_weight = 5 * 35)
  (h_new_avg : total_new_weight = 6 * 36)
  (h_new_weight : total_new_weight = original_weight + W) :
  W = 41 := by
  sorry

end weight_of_new_boy_l1676_167637


namespace system_solution_exists_l1676_167607

theorem system_solution_exists (x y: ℝ) :
    (y^2 = (x + 8) * (x^2 + 2) ∧ y^2 - (8 + 4 * x) * y + (16 + 16 * x - 5 * x^2) = 0) → 
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨ (x = -2 ∧ (y = 6 ∨ y = -6)) ∨ (x = 19 ∧ (y = 99 ∨ y = -99))) :=
    sorry

end system_solution_exists_l1676_167607


namespace february_sales_increase_l1676_167632

theorem february_sales_increase (Slast : ℝ) (r : ℝ) (Sthis : ℝ) 
  (h_last_year_sales : Slast = 320) 
  (h_percent_increase : r = 0.25) : 
  Sthis = 400 :=
by
  have h1 : Sthis = Slast * (1 + r) := sorry
  sorry

end february_sales_increase_l1676_167632


namespace map_distance_representation_l1676_167698

-- Define the conditions and the question as a Lean statement
theorem map_distance_representation :
  (∀ (length_cm : ℕ), (length_cm : ℕ) = 23 → (length_cm * 50 / 10 : ℕ) = 115) :=
by
  sorry

end map_distance_representation_l1676_167698


namespace S7_is_28_l1676_167613

variables {a_n : ℕ → ℤ} -- Sequence definition
variables {S_n : ℕ → ℤ} -- Sum of the first n terms

-- Define an arithmetic sequence condition
def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Given conditions
axiom sum_condition : a_n 2 + a_n 4 + a_n 6 = 12
axiom sum_formula (n : ℕ) : S_n n = n * (a_n 1 + a_n n) / 2
axiom arith_seq : is_arithmetic_sequence a_n

-- The statement to be proven
theorem S7_is_28 : S_n 7 = 28 :=
sorry

end S7_is_28_l1676_167613


namespace cost_flying_X_to_Y_l1676_167612

def distance_XY : ℝ := 4500 -- Distance from X to Y in km
def cost_per_km_flying : ℝ := 0.12 -- Cost per km for flying in dollars
def booking_fee_flying : ℝ := 120 -- Booking fee for flying in dollars

theorem cost_flying_X_to_Y : 
    distance_XY * cost_per_km_flying + booking_fee_flying = 660 := by
  sorry

end cost_flying_X_to_Y_l1676_167612


namespace missing_number_is_6630_l1676_167643

theorem missing_number_is_6630 (x : ℕ) (h : 815472 / x = 123) : x = 6630 :=
by {
  sorry
}

end missing_number_is_6630_l1676_167643


namespace sum_of_squares_of_roots_of_quadratic_l1676_167601

noncomputable def sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) : Prop :=
  a^2 + b^2 = 4 * p^2 - 6 * q

theorem sum_of_squares_of_roots_of_quadratic
  (p q a b : ℝ)
  (h1 : a + b = 2 * p / 3)
  (h2 : a * b = q / 3)
  (h3 : a * a + b * b = 4 * p^2 - 6 * q) :
  sum_of_squares_of_roots p q a b :=
by
  sorry

end sum_of_squares_of_roots_of_quadratic_l1676_167601


namespace power_of_two_square_l1676_167673

theorem power_of_two_square (n : ℕ) : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2 ↔ n = 10 :=
by
  sorry

end power_of_two_square_l1676_167673


namespace intersect_single_point_l1676_167603

theorem intersect_single_point (k : ℝ) :
  (∃ x : ℝ, (x^2 + k * x + 1 = 0) ∧
   ∀ x y : ℝ, (x^2 + k * x + 1 = 0 → y^2 + k * y + 1 = 0 → x = y))
  ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end intersect_single_point_l1676_167603


namespace common_chord_eq_l1676_167671

theorem common_chord_eq : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x + 8*y - 8 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 2 = 0) → 
  (∀ x y : ℝ, x + 2*y - 1 = 0) :=
by 
  sorry

end common_chord_eq_l1676_167671


namespace f_2015_2016_l1676_167697

theorem f_2015_2016 (f : ℤ → ℤ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l1676_167697


namespace range_of_a_l1676_167686

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 ↔ x > Real.log a / Real.log 2) → 0 < a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l1676_167686


namespace solve_cubic_eq_solve_quadratic_eq_l1676_167682

-- Define the first equation and prove its solution
theorem solve_cubic_eq (x : ℝ) (h : x^3 + 64 = 0) : x = -4 :=
by
  -- skipped proof
  sorry

-- Define the second equation and prove its solutions
theorem solve_quadratic_eq (x : ℝ) (h : (x - 2)^2 = 81) : x = 11 ∨ x = -7 :=
by
  -- skipped proof
  sorry

end solve_cubic_eq_solve_quadratic_eq_l1676_167682


namespace option_d_is_right_triangle_l1676_167627

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end option_d_is_right_triangle_l1676_167627
