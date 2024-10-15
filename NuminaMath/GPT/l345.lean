import Mathlib

namespace NUMINAMATH_GPT_conic_section_is_hyperbola_l345_34562

theorem conic_section_is_hyperbola :
  ∀ (x y : ℝ), x^2 - 16 * y^2 - 8 * x + 16 * y + 32 = 0 → 
               (∃ h k a b : ℝ, h = 4 ∧ k = 0.5 ∧ a = b ∧ a^2 = 2 ∧ b^2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_is_hyperbola_l345_34562


namespace NUMINAMATH_GPT_roberts_monthly_expenses_l345_34518

-- Conditions
def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.1
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.2

-- Definitions derived from the conditions
noncomputable def commission : ℝ := commission_rate * total_sales
noncomputable def total_earnings : ℝ := basic_salary + commission
noncomputable def savings : ℝ := savings_rate * total_earnings
noncomputable def monthly_expenses : ℝ := total_earnings - savings

-- The statement to be proved
theorem roberts_monthly_expenses : monthly_expenses = 2888 := by
  sorry

end NUMINAMATH_GPT_roberts_monthly_expenses_l345_34518


namespace NUMINAMATH_GPT_mixed_tea_sale_price_l345_34558

noncomputable def sale_price_of_mixed_tea (weight1 weight2 weight3 price1 price2 price3 profit1 profit2 profit3 : ℝ) : ℝ :=
  let total_cost1 := weight1 * price1
  let total_cost2 := weight2 * price2
  let total_cost3 := weight3 * price3
  let total_profit1 := profit1 * total_cost1
  let total_profit2 := profit2 * total_cost2
  let total_profit3 := profit3 * total_cost3
  let selling_price1 := total_cost1 + total_profit1
  let selling_price2 := total_cost2 + total_profit2
  let selling_price3 := total_cost3 + total_profit3
  let total_selling_price := selling_price1 + selling_price2 + selling_price3
  let total_weight := weight1 + weight2 + weight3
  total_selling_price / total_weight

theorem mixed_tea_sale_price :
  sale_price_of_mixed_tea 120 45 35 30 40 60 0.50 0.30 0.25 = 51.825 :=
by
  sorry

end NUMINAMATH_GPT_mixed_tea_sale_price_l345_34558


namespace NUMINAMATH_GPT_middle_number_in_consecutive_nat_sum_squares_equals_2030_l345_34568

theorem middle_number_in_consecutive_nat_sum_squares_equals_2030 
  (n : ℕ)
  (h1 : (n - 1)^2 + n^2 + (n + 1)^2 = 2030)
  (h2 : (n^3 - n^2) % 7 = 0)
  : n = 26 := 
sorry

end NUMINAMATH_GPT_middle_number_in_consecutive_nat_sum_squares_equals_2030_l345_34568


namespace NUMINAMATH_GPT_helga_shoe_pairs_l345_34564

theorem helga_shoe_pairs
  (first_store_pairs: ℕ) 
  (second_store_pairs: ℕ) 
  (third_store_pairs: ℕ)
  (fourth_store_pairs: ℕ)
  (h1: first_store_pairs = 7)
  (h2: second_store_pairs = first_store_pairs + 2)
  (h3: third_store_pairs = 0)
  (h4: fourth_store_pairs = 2 * (first_store_pairs + second_store_pairs + third_store_pairs))
  : first_store_pairs + second_store_pairs + third_store_pairs + fourth_store_pairs = 48 := 
by
  sorry

end NUMINAMATH_GPT_helga_shoe_pairs_l345_34564


namespace NUMINAMATH_GPT_inequality_solution_set_l345_34522

theorem inequality_solution_set : 
  (∃ (x : ℝ), (4 / (x - 1) ≤ x - 1) ↔ (x ≥ 3 ∨ (-1 ≤ x ∧ x < 1))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l345_34522


namespace NUMINAMATH_GPT_tickets_spent_dunk_a_clown_booth_l345_34557

/-
The conditions given:
1. Tom bought 40 tickets.
2. Tom went on 3 rides.
3. Each ride costs 4 tickets.
-/
def total_tickets : ℕ := 40
def rides_count : ℕ := 3
def tickets_per_ride : ℕ := 4

/-
We aim to prove that Tom spent 28 tickets at the 'dunk a clown' booth.
-/
theorem tickets_spent_dunk_a_clown_booth :
  (total_tickets - rides_count * tickets_per_ride) = 28 :=
by
  sorry

end NUMINAMATH_GPT_tickets_spent_dunk_a_clown_booth_l345_34557


namespace NUMINAMATH_GPT_age_difference_l345_34559

theorem age_difference (A B : ℕ) (h1 : B = 38) (h2 : A + 10 = 2 * (B - 10)) : A - B = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l345_34559


namespace NUMINAMATH_GPT_find_fraction_l345_34561

theorem find_fraction (N : ℕ) (hN : N = 90) (f : ℚ)
  (h : 3 + (1/2) * f * (1/5) * N = (1/15) * N) :
  f = 1/3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_fraction_l345_34561


namespace NUMINAMATH_GPT_event_B_more_likely_l345_34555

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end NUMINAMATH_GPT_event_B_more_likely_l345_34555


namespace NUMINAMATH_GPT_find_cheesecake_price_l345_34595

def price_of_cheesecake (C : ℝ) (coffee_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  let original_price := coffee_price + C
  let discounted_price := discount_rate * original_price
  discounted_price = final_price

theorem find_cheesecake_price : ∃ C : ℝ,
  price_of_cheesecake C 6 0.75 12 ∧ C = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_cheesecake_price_l345_34595


namespace NUMINAMATH_GPT_chocolates_difference_l345_34596

-- Conditions
def Robert_chocolates : Nat := 13
def Nickel_chocolates : Nat := 4

-- Statement
theorem chocolates_difference : (Robert_chocolates - Nickel_chocolates) = 9 := by
  sorry

end NUMINAMATH_GPT_chocolates_difference_l345_34596


namespace NUMINAMATH_GPT_janet_earned_1390_in_interest_l345_34576

def janets_total_interest (total_investment investment_at_10_rate investment_at_10_interest investment_at_1_rate remaining_investment remaining_investment_interest : ℝ) : ℝ :=
    investment_at_10_interest + remaining_investment_interest

theorem janet_earned_1390_in_interest :
  janets_total_interest 31000 12000 0.10 (12000 * 0.10) 0.01 (19000 * 0.01) = 1390 :=
by
  sorry

end NUMINAMATH_GPT_janet_earned_1390_in_interest_l345_34576


namespace NUMINAMATH_GPT_sum_a_b_l345_34540

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) : a + b = 176 / 57 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_l345_34540


namespace NUMINAMATH_GPT_find_missing_number_l345_34553

theorem find_missing_number (x : ℤ) (h : 10010 - 12 * x * 2 = 9938) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l345_34553


namespace NUMINAMATH_GPT_divisor_greater_than_2_l345_34567

theorem divisor_greater_than_2 (w n d : ℕ) (h1 : ∃ q1 : ℕ, w = d * q1 + 2)
                                       (h2 : n % 8 = 5)
                                       (h3 : n < 180) : 2 < d :=
sorry

end NUMINAMATH_GPT_divisor_greater_than_2_l345_34567


namespace NUMINAMATH_GPT_solution_set_f_x_minus_1_lt_0_l345_34598

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then x - 1 else -x - 1

theorem solution_set_f_x_minus_1_lt_0 :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_f_x_minus_1_lt_0_l345_34598


namespace NUMINAMATH_GPT_range_of_b_l345_34581

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem range_of_b (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (f x1 = b) ∧ (f x2 = b) ∧ (f x3 = b))
  ↔ (-4 / 3 < b ∧ b < 28 / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l345_34581


namespace NUMINAMATH_GPT_ratio_right_to_left_l345_34572

theorem ratio_right_to_left (L C R : ℕ) (hL : L = 12) (hC : C = L + 2) (hTotal : L + C + R = 50) :
  R / L = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_right_to_left_l345_34572


namespace NUMINAMATH_GPT_problem1_problem2_l345_34542

-- Problem 1: Prove that (x + y + z)² - (x + y - z)² = 4z(x + y) for x, y, z ∈ ℝ
theorem problem1 (x y z : ℝ) : (x + y + z)^2 - (x + y - z)^2 = 4 * z * (x + y) := 
sorry

-- Problem 2: Prove that (a + 2b)² - 2(a + 2b)(a - 2b) + (a - 2b)² = 16b² for a, b ∈ ℝ
theorem problem2 (a b : ℝ) : (a + 2 * b)^2 - 2 * (a + 2 * b) * (a - 2 * b) + (a - 2 * b)^2 = 16 * b^2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l345_34542


namespace NUMINAMATH_GPT_hexagon_rotation_angle_l345_34523

theorem hexagon_rotation_angle (θ : ℕ) : θ = 90 → ¬ ∃ k, k * 60 = θ ∨ θ = 360 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_rotation_angle_l345_34523


namespace NUMINAMATH_GPT_slip_3_5_in_F_l345_34541

def slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]

def cup_sum (x : List ℝ) := List.sum x

def slips_dist (A B C D E F : List ℝ) : Prop :=
  cup_sum A + cup_sum B + cup_sum C + cup_sum D + cup_sum E + cup_sum F = 50 ∧ 
  cup_sum A = 6 ∧ cup_sum B = 8 ∧ cup_sum C = 10 ∧ cup_sum D = 12 ∧ cup_sum E = 14 ∧ cup_sum F = 16 ∧
  2.5 ∈ B ∧ 2.5 ∈ D ∧ 4 ∈ C

def contains_slip (c : List ℝ) (v : ℝ) : Prop := v ∈ c

theorem slip_3_5_in_F (A B C D E F : List ℝ) (h : slips_dist A B C D E F) : 
  contains_slip F 3.5 :=
sorry

end NUMINAMATH_GPT_slip_3_5_in_F_l345_34541


namespace NUMINAMATH_GPT_current_at_resistance_12_l345_34566

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end NUMINAMATH_GPT_current_at_resistance_12_l345_34566


namespace NUMINAMATH_GPT_f_increasing_on_pos_real_l345_34530

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 1)

theorem f_increasing_on_pos_real : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 < f x2 :=
by sorry

end NUMINAMATH_GPT_f_increasing_on_pos_real_l345_34530


namespace NUMINAMATH_GPT_max_intersections_l345_34501

theorem max_intersections (n1 n2 k : ℕ) 
  (h1 : n1 ≤ n2)
  (h2 : k ≤ n1) : 
  ∃ max_intersections : ℕ, 
  max_intersections = k * n2 :=
by
  sorry

end NUMINAMATH_GPT_max_intersections_l345_34501


namespace NUMINAMATH_GPT_sum_factors_of_18_l345_34584

theorem sum_factors_of_18 : (1 + 18 + 2 + 9 + 3 + 6) = 39 := by
  sorry

end NUMINAMATH_GPT_sum_factors_of_18_l345_34584


namespace NUMINAMATH_GPT_second_percentage_increase_l345_34544

theorem second_percentage_increase (P : ℝ) (x : ℝ) :
  1.25 * P * (1 + x / 100) = 1.625 * P ↔ x = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_percentage_increase_l345_34544


namespace NUMINAMATH_GPT_determine_K_class_comparison_l345_34583

variables (a b : ℕ) -- number of students in classes A and B respectively
variable (K : ℕ) -- amount that each A student would pay if they covered all cost

-- Conditions from the problem statement
def first_event_total (a b : ℕ) := 5 * a + 3 * b
def second_event_total (a b : ℕ) := 4 * a + 6 * b
def total_balance (a b K : ℕ) := 9 * (a + b) = K * (a + b)

-- Questions to be answered
theorem determine_K : total_balance a b K → K = 9 :=
by
  sorry

theorem class_comparison (a b : ℕ) : 5 * a + 3 * b = 4 * a + 6 * b → b > a :=
by
  sorry

end NUMINAMATH_GPT_determine_K_class_comparison_l345_34583


namespace NUMINAMATH_GPT_range_of_h_l345_34571

def f (x : ℝ) : ℝ := 4 * x - 3
def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h : 
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -127 ≤ h x ∧ h x ≤ 129) :=
by
  sorry

end NUMINAMATH_GPT_range_of_h_l345_34571


namespace NUMINAMATH_GPT_greatest_number_l345_34524

-- Define the base conversions
def octal_to_decimal (n : Nat) : Nat := 3 * 8^1 + 2
def quintal_to_decimal (n : Nat) : Nat := 1 * 5^2 + 1 * 5^1 + 1
def binary_to_decimal (n : Nat) : Nat := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0
def senary_to_decimal (n : Nat) : Nat := 5 * 6^1 + 4

theorem greatest_number :
  max (max (octal_to_decimal 32) (quintal_to_decimal 111)) (max (binary_to_decimal 101010) (senary_to_decimal 54))
  = binary_to_decimal 101010 := by sorry

end NUMINAMATH_GPT_greatest_number_l345_34524


namespace NUMINAMATH_GPT_common_ratio_arith_geo_sequence_l345_34580

theorem common_ratio_arith_geo_sequence (a : ℕ → ℝ) (d : ℝ) (q : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geo : (a 1 + 2) * q = a 5 + 5) 
  (h_geo' : (a 5 + 5) * q = a 9 + 8) :
  q = 1 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_arith_geo_sequence_l345_34580


namespace NUMINAMATH_GPT_smallest_m_for_integral_solutions_l345_34560

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), (12 * x^2 - m * x + 504 = 0 → ∃ (p q : ℤ), p + q = m / 12 ∧ p * q = 42)) ∧
  m = 156 := by
sorry

end NUMINAMATH_GPT_smallest_m_for_integral_solutions_l345_34560


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l345_34508

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a m + 1 - a m) 
  (h1 : a 2 = 3) (h2 : a 5 = 12) : a 8 = 21 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l345_34508


namespace NUMINAMATH_GPT_valentino_farm_birds_total_l345_34548

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_valentino_farm_birds_total_l345_34548


namespace NUMINAMATH_GPT_find_m_value_l345_34512

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l345_34512


namespace NUMINAMATH_GPT_find_square_sum_l345_34536

theorem find_square_sum :
  ∃ a b c : ℕ, a = 2494651 ∧ b = 1385287 ∧ c = 9406087 ∧ (a + b + c = 3645^2) :=
by
  have h1 : 2494651 + 1385287 + 9406087 = 13286025 := by norm_num
  have h2 : 3645^2 = 13286025 := by norm_num
  exact ⟨2494651, 1385287, 9406087, rfl, rfl, rfl, h2⟩

end NUMINAMATH_GPT_find_square_sum_l345_34536


namespace NUMINAMATH_GPT_triangle_angle_and_area_l345_34531

theorem triangle_angle_and_area (a b c A B C : ℝ)
  (h₁ : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))
  (h₂ : 0 < C ∧ C < Real.pi)
  (h₃ : c = 2 * Real.sqrt 3) :
  C = Real.pi / 3 ∧ 0 ≤ (1 / 2) * a * b * Real.sin C ∧ (1 / 2) * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_and_area_l345_34531


namespace NUMINAMATH_GPT_average_value_continuous_l345_34554

noncomputable def average_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / (b - a)) * ∫ x in a..b, f x

theorem average_value_continuous (f : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn f (Set.Icc a b)) :
  (average_value f a b) = (1 / (b - a)) * (∫ x in a..b, f x) :=
by
  sorry

end NUMINAMATH_GPT_average_value_continuous_l345_34554


namespace NUMINAMATH_GPT_r_daily_earnings_l345_34563

def earnings_problem (P Q R : ℝ) : Prop :=
  (9 * (P + Q + R) = 1890) ∧ 
  (5 * (P + R) = 600) ∧ 
  (7 * (Q + R) = 910)

theorem r_daily_earnings :
  ∃ P Q R : ℝ, earnings_problem P Q R ∧ R = 40 := sorry

end NUMINAMATH_GPT_r_daily_earnings_l345_34563


namespace NUMINAMATH_GPT_lcm_180_616_l345_34533

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := 
by
  sorry

end NUMINAMATH_GPT_lcm_180_616_l345_34533


namespace NUMINAMATH_GPT_mean_age_is_10_l345_34570

def ages : List ℤ := [7, 7, 7, 14, 15]

theorem mean_age_is_10 : (List.sum ages : ℤ) / (ages.length : ℤ) = 10 := by
-- sorry placeholder for the actual proof
sorry

end NUMINAMATH_GPT_mean_age_is_10_l345_34570


namespace NUMINAMATH_GPT_fg_of_3_eq_83_l345_34506

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_83 : f (g 3) = 83 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_83_l345_34506


namespace NUMINAMATH_GPT_distance_to_directrix_l345_34529

theorem distance_to_directrix (p : ℝ) (h1 : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2 * Real.sqrt 2)) :
  abs (2 - (-1)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_directrix_l345_34529


namespace NUMINAMATH_GPT_intersection_A_B_l345_34565

def A : Set ℝ := { x | (x + 1) / (x - 1) ≤ 0 }
def B : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l345_34565


namespace NUMINAMATH_GPT_maximum_negative_roots_l345_34539

theorem maximum_negative_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (discriminant1 : b^2 - 4 * a * c ≥ 0)
    (discriminant2 : c^2 - 4 * b * a ≥ 0)
    (discriminant3 : a^2 - 4 * c * b ≥ 0) :
    ∃ n : ℕ, n ≤ 2 ∧ ∀ x ∈ {x | a * x^2 + b * x + c = 0 ∨ b * x^2 + c * x + a = 0 ∨ c * x^2 + a * x + b = 0}, x < 0 ↔ n = 2 := 
sorry

end NUMINAMATH_GPT_maximum_negative_roots_l345_34539


namespace NUMINAMATH_GPT_clara_hardcover_books_l345_34528

-- Define the variables and conditions
variables (h p : ℕ)

-- Conditions based on the problem statement
def volumes_total : Prop := h + p = 12
def total_cost (total : ℕ) : Prop := 28 * h + 18 * p = total

-- The theorem to prove
theorem clara_hardcover_books (h p : ℕ) (H1 : volumes_total h p) (H2 : total_cost h p 270) : h = 6 :=
by
  sorry

end NUMINAMATH_GPT_clara_hardcover_books_l345_34528


namespace NUMINAMATH_GPT_find_a_l345_34597

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x >= 0 then a^x else a^(-x)

theorem find_a (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a)
(h_ge : ∀ x : ℝ, x >= 0 → f x a = a ^ x)
(h_a_gt_1 : a > 1)
(h_sol : ∀ x : ℝ, f x a ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) :
a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l345_34597


namespace NUMINAMATH_GPT_rectangle_other_side_l345_34599

theorem rectangle_other_side
  (a b : ℝ)
  (Area : ℝ := 12 * a ^ 2 - 6 * a * b)
  (side1 : ℝ := 3 * a)
  (side2 : ℝ := Area / side1) :
  side2 = 4 * a - 2 * b :=
by
  sorry

end NUMINAMATH_GPT_rectangle_other_side_l345_34599


namespace NUMINAMATH_GPT_games_played_by_player_3_l345_34502

theorem games_played_by_player_3 (games_1 games_2 : ℕ) (rotation_system : ℕ) :
  games_1 = 10 → games_2 = 21 →
  rotation_system = (games_2 - games_1) →
  rotation_system = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_games_played_by_player_3_l345_34502


namespace NUMINAMATH_GPT_other_root_l345_34535

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_other_root_l345_34535


namespace NUMINAMATH_GPT_circle_value_of_m_l345_34517

theorem circle_value_of_m (m : ℝ) : (∃ a b r : ℝ, r > 0 ∧ (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ m < 1/2 := by
  sorry

end NUMINAMATH_GPT_circle_value_of_m_l345_34517


namespace NUMINAMATH_GPT_number_of_possible_values_of_a_l345_34504

theorem number_of_possible_values_of_a :
  ∃ a_count : ℕ, (∃ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2020 ∧ a^2 - b^2 + c^2 - d^2 = 2020 ∧ a_count = 501) :=
sorry

end NUMINAMATH_GPT_number_of_possible_values_of_a_l345_34504


namespace NUMINAMATH_GPT_projection_problem_l345_34532

noncomputable def vector_proj (w v : ℝ × ℝ) : ℝ × ℝ := sorry -- assume this definition

variables (v w : ℝ × ℝ)

-- Given condition
axiom proj_v : vector_proj w v = ⟨4, 3⟩

-- Proof Statement
theorem projection_problem :
  vector_proj w (7 • v + 2 • w) = ⟨28, 21⟩ + 2 • w :=
sorry

end NUMINAMATH_GPT_projection_problem_l345_34532


namespace NUMINAMATH_GPT_simplify_abs_expression_l345_34594

theorem simplify_abs_expression (a b c : ℝ) (h1 : a + c > b) (h2 : b + c > a) (h3 : a + b > c) :
  |a - b + c| - |a - b - c| = 2 * a - 2 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_abs_expression_l345_34594


namespace NUMINAMATH_GPT_video_total_votes_l345_34593

theorem video_total_votes (x : ℕ) (L D : ℕ)
  (h1 : L + D = x)
  (h2 : L - D = 130)
  (h3 : 70 * x = 100 * L) :
  x = 325 :=
by
  sorry

end NUMINAMATH_GPT_video_total_votes_l345_34593


namespace NUMINAMATH_GPT_floor_ceil_sum_l345_34587

theorem floor_ceil_sum (x : ℝ) (h : Int.floor x + Int.ceil x = 7) : x ∈ { x : ℝ | 3 < x ∧ x < 4 } ∪ {3.5} :=
sorry

end NUMINAMATH_GPT_floor_ceil_sum_l345_34587


namespace NUMINAMATH_GPT_k_value_l345_34574

theorem k_value {x y k : ℝ} (h : ∃ c : ℝ, (x ^ 2 + k * x * y + 49 * y ^ 2) = c ^ 2) : k = 14 ∨ k = -14 :=
by sorry

end NUMINAMATH_GPT_k_value_l345_34574


namespace NUMINAMATH_GPT_wolf_nobel_laureates_l345_34590

theorem wolf_nobel_laureates (W N total W_prize N_prize N_noW N_W : ℕ)
  (h1 : W_prize = 31)
  (h2 : total = 50)
  (h3 : N_prize = 27)
  (h4 : N_noW + N_W = total - W_prize)
  (h5 : N_W = N_noW + 3)
  (h6 : N_prize = W + N_W) :
  W = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_wolf_nobel_laureates_l345_34590


namespace NUMINAMATH_GPT_peanut_butter_candy_pieces_l345_34534

theorem peanut_butter_candy_pieces :
  ∀ (pb_candy grape_candy banana_candy : ℕ),
  pb_candy = 4 * grape_candy →
  grape_candy = banana_candy + 5 →
  banana_candy = 43 →
  pb_candy = 192 :=
by
  sorry

end NUMINAMATH_GPT_peanut_butter_candy_pieces_l345_34534


namespace NUMINAMATH_GPT_area_of_union_of_triangles_l345_34519

-- Define the vertices of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (7, 3)

-- Define the reflection function across the line x=5
def reflect_x5 (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (10 - x, y)

-- Define the vertices of the reflected triangle
def A' : ℝ × ℝ := reflect_x5 A
def B' : ℝ × ℝ := reflect_x5 B
def C' : ℝ × ℝ := reflect_x5 C

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the union of both triangles is 22
theorem area_of_union_of_triangles : triangle_area A B C + triangle_area A' B' C' = 22 := by
  sorry

end NUMINAMATH_GPT_area_of_union_of_triangles_l345_34519


namespace NUMINAMATH_GPT_determine_phi_l345_34592

theorem determine_phi (phi : ℝ) (h : 0 < phi ∧ phi < π) :
  (∃ k : ℤ, phi = 2*k*π + (3*π/4)) :=
by
  sorry

end NUMINAMATH_GPT_determine_phi_l345_34592


namespace NUMINAMATH_GPT_solve_inequality_l345_34514

open Set

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 < 0

-- Define the solution sets for different cases of a
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x < 1 / a ∨ x > 1}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1 / a}
  else if a > 1 then {x | 1 / a < x ∧ x < 1}
  else ∅

-- State the theorem
theorem solve_inequality (a : ℝ) : 
  {x : ℝ | inequality a x} = solution_set a :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l345_34514


namespace NUMINAMATH_GPT_evaluate_special_operation_l345_34573

-- Define the operation @
def special_operation (a b : ℕ) : ℚ := (a * b) / (a - b)

-- State the theorem
theorem evaluate_special_operation : special_operation 6 3 = 6 := by
  sorry

end NUMINAMATH_GPT_evaluate_special_operation_l345_34573


namespace NUMINAMATH_GPT_grapes_purchased_l345_34588

variable (G : ℕ)
variable (rate_grapes : ℕ) (qty_mangoes : ℕ) (rate_mangoes : ℕ) (total_paid : ℕ)

theorem grapes_purchased (h1 : rate_grapes = 70)
                        (h2 : qty_mangoes = 9)
                        (h3 : rate_mangoes = 55)
                        (h4 : total_paid = 1055) :
                        70 * G + 9 * 55 = 1055 → G = 8 :=
by
  sorry

end NUMINAMATH_GPT_grapes_purchased_l345_34588


namespace NUMINAMATH_GPT_total_coins_Zain_l345_34547

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end NUMINAMATH_GPT_total_coins_Zain_l345_34547


namespace NUMINAMATH_GPT_gallons_10_percent_milk_needed_l345_34545

-- Definitions based on conditions
def amount_of_butterfat (x : ℝ) : ℝ := 0.10 * x
def total_butterfat_in_existing_milk : ℝ := 4
def final_butterfat (x : ℝ) : ℝ := amount_of_butterfat x + total_butterfat_in_existing_milk
def total_milk (x : ℝ) : ℝ := x + 8
def desired_butterfat (x : ℝ) : ℝ := 0.20 * total_milk x

-- Lean proof statement
theorem gallons_10_percent_milk_needed (x : ℝ) : final_butterfat x = desired_butterfat x → x = 24 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_gallons_10_percent_milk_needed_l345_34545


namespace NUMINAMATH_GPT_maximize_profit_l345_34503
-- Importing the entire necessary library

-- Definitions and conditions
def cost_price : ℕ := 40
def minimum_selling_price : ℕ := 44
def maximum_profit_margin : ℕ := 30
def sales_at_minimum_price : ℕ := 300
def price_increase_effect : ℕ := 10
def max_profit_price := 52
def max_profit := 2640

-- Function relationship between y and x
def sales_volume (x : ℕ) : ℕ := 300 - 10 * (x - 44)

-- Range of x
def valid_price (x : ℕ) : Prop := 44 ≤ x ∧ x ≤ 52

-- Statement of the problem
theorem maximize_profit (x : ℕ) (hx : valid_price x) : 
  sales_volume x = 300 - 10 * (x - 44) ∧
  44 ≤ x ∧ x ≤ 52 ∧
  x = 52 → 
  (x - cost_price) * (sales_volume x) = max_profit :=
sorry

end NUMINAMATH_GPT_maximize_profit_l345_34503


namespace NUMINAMATH_GPT_initial_momentum_eq_2Fx_div_v_l345_34591

variable (m v F x t : ℝ)
variable (H_initial_conditions : v ≠ 0)
variable (H_force : F > 0)
variable (H_distance : x > 0)
variable (H_time : t > 0)
variable (H_stopping_distance : x = (m * v^2) / (2 * F))
variable (H_stopping_time : t = (m * v) / F)

theorem initial_momentum_eq_2Fx_div_v :
  m * v = (2 * F * x) / v :=
sorry

end NUMINAMATH_GPT_initial_momentum_eq_2Fx_div_v_l345_34591


namespace NUMINAMATH_GPT_students_play_neither_l345_34507

-- Defining the problem parameters
def total_students : ℕ := 36
def football_players : ℕ := 26
def tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Statement to be proved
theorem students_play_neither : (total_students - (football_players + tennis_players - both_players)) = 7 :=
by show total_students - (football_players + tennis_players - both_players) = 7; sorry

end NUMINAMATH_GPT_students_play_neither_l345_34507


namespace NUMINAMATH_GPT_triangle_inequality_l345_34556

-- Define the side lengths of a triangle
variables {a b c : ℝ}

-- State the main theorem
theorem triangle_inequality :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l345_34556


namespace NUMINAMATH_GPT_simon_change_l345_34516

noncomputable def calculate_change 
  (pansies_count : ℕ) (pansies_price : ℚ) (pansies_discount : ℚ) 
  (hydrangea_count : ℕ) (hydrangea_price : ℚ) (hydrangea_discount : ℚ) 
  (petunias_count : ℕ) (petunias_price : ℚ) (petunias_discount : ℚ) 
  (lilies_count : ℕ) (lilies_price : ℚ) (lilies_discount : ℚ) 
  (orchids_count : ℕ) (orchids_price : ℚ) (orchids_discount : ℚ) 
  (sales_tax : ℚ) (payment : ℚ) : ℚ :=
  let pansies_total := (pansies_count * pansies_price) * (1 - pansies_discount)
  let hydrangea_total := (hydrangea_count * hydrangea_price) * (1 - hydrangea_discount)
  let petunias_total := (petunias_count * petunias_price) * (1 - petunias_discount)
  let lilies_total := (lilies_count * lilies_price) * (1 - lilies_discount)
  let orchids_total := (orchids_count * orchids_price) * (1 - orchids_discount)
  let total_price := pansies_total + hydrangea_total + petunias_total + lilies_total + orchids_total
  let final_price := total_price * (1 + sales_tax)
  payment - final_price

theorem simon_change : calculate_change
  5 2.50 0.10
  1 12.50 0.15
  5 1.00 0.20
  3 5.00 0.12
  2 7.50 0.08
  0.06 100 = 43.95 := by sorry

end NUMINAMATH_GPT_simon_change_l345_34516


namespace NUMINAMATH_GPT_problem_I_problem_II_problem_III_l345_34589

-- Problem (I)
noncomputable def f (x a : ℝ) := Real.log x - a * (x - 1)
noncomputable def tangent_line (x a : ℝ) := (1 - a) * (x - 1)

theorem problem_I (a : ℝ) :
  ∃ y, tangent_line y a = f 1 a / (1 : ℝ) :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x ≥ 1, f x a ≤ Real.log x / (x + 1) :=
sorry

-- Problem (III)
theorem problem_III (a : ℝ) :
  ∀ x ≥ 1, Real.exp (x - 1) - a * (x ^ 2 - x) ≥ x * f x a + 1 :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_problem_III_l345_34589


namespace NUMINAMATH_GPT_octahedron_sum_l345_34510

-- Define the properties of an octahedron
def octahedron_edges := 12
def octahedron_vertices := 6
def octahedron_faces := 8

theorem octahedron_sum : octahedron_edges + octahedron_vertices + octahedron_faces = 26 := by
  -- Here we state that the sum of edges, vertices, and faces equals 26
  sorry

end NUMINAMATH_GPT_octahedron_sum_l345_34510


namespace NUMINAMATH_GPT_bernoulli_inequality_l345_34520

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x ≥ -1) (hn : n ≥ 1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end NUMINAMATH_GPT_bernoulli_inequality_l345_34520


namespace NUMINAMATH_GPT_second_offset_length_l345_34578

-- Definitions based on the given conditions.
def diagonal : ℝ := 24
def offset1 : ℝ := 9
def area_quad : ℝ := 180

-- Statement to prove the length of the second offset.
theorem second_offset_length :
  ∃ h : ℝ, (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * h = area_quad ∧ h = 6 :=
by
  sorry

end NUMINAMATH_GPT_second_offset_length_l345_34578


namespace NUMINAMATH_GPT_son_l345_34537

theorem son's_age (F S : ℕ) (h1 : F + S = 75) (h2 : F = 8 * (S - (F - S))) : S = 27 :=
sorry

end NUMINAMATH_GPT_son_l345_34537


namespace NUMINAMATH_GPT_Mike_gave_marbles_l345_34511

variables (original_marbles given_marbles remaining_marbles : ℕ)

def Mike_original_marbles : ℕ := 8
def Mike_remaining_marbles : ℕ := 4
def Mike_given_marbles (original remaining : ℕ) : ℕ := original - remaining

theorem Mike_gave_marbles :
  Mike_given_marbles Mike_original_marbles Mike_remaining_marbles = 4 :=
sorry

end NUMINAMATH_GPT_Mike_gave_marbles_l345_34511


namespace NUMINAMATH_GPT_greatest_k_inequality_l345_34549

theorem greatest_k_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ( ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    (a / b + b / c + c / a - 3) ≥ k * (a / (b + c) + b / (c + a) + c / (a + b) - 3 / 2) ) ↔ k = 1 := 
sorry

end NUMINAMATH_GPT_greatest_k_inequality_l345_34549


namespace NUMINAMATH_GPT_solve_inequality_l345_34527

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if h : a > -1 then { x : ℝ | -1 < x ∧ x < a }
  else if h : a < -1 then { x : ℝ | a < x ∧ x < -1 }
  else ∅

theorem solve_inequality (x a : ℝ) :
  (x^2 + (1 - a)*x - a < 0) ↔ (
    (a > -1 → x ∈ { x : ℝ | -1 < x ∧ x < a }) ∧
    (a < -1 → x ∈ { x : ℝ | a < x ∧ x < -1 }) ∧
    (a = -1 → False)
  ) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l345_34527


namespace NUMINAMATH_GPT_circle_equation_l345_34546

theorem circle_equation 
  (circle_eq : ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = (x - 3)^2 + (y - 2)^2) 
  (tangent_to_line : ∀ (x y : ℝ), (2*x - y + 5) = 0 → 
    (x = -2 ∧ y = 1))
  (passes_through_N : ∀ (x y : ℝ), (x = 3 ∧ y = 2)) :
  ∀ (x y : ℝ), x^2 + y^2 - 9*x + (9/2)*y - (55/2) = 0 := 
sorry

end NUMINAMATH_GPT_circle_equation_l345_34546


namespace NUMINAMATH_GPT_second_closest_location_l345_34575
-- Import all necessary modules from the math library

-- Define the given distances (conditions)
def distance_library : ℝ := 1.912 * 1000  -- distance in meters
def distance_park : ℝ := 876              -- distance in meters
def distance_clothing_store : ℝ := 1.054 * 1000  -- distance in meters

-- State the proof problem
theorem second_closest_location :
  (distance_library = 1912) →
  (distance_park = 876) →
  (distance_clothing_store = 1054) →
  (distance_clothing_store = 1054) :=
by
  intros h1 h2 h3
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_second_closest_location_l345_34575


namespace NUMINAMATH_GPT_domain_of_function_l345_34579

theorem domain_of_function :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x ≠ 0} = {x : ℝ | -2 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l345_34579


namespace NUMINAMATH_GPT_frog_climbs_out_l345_34586

theorem frog_climbs_out (d climb slip : ℕ) (h : d = 20) (h_climb : climb = 3) (h_slip : slip = 2) :
  ∃ n : ℕ, n = 20 ∧ d ≤ n * (climb - slip) + climb :=
sorry

end NUMINAMATH_GPT_frog_climbs_out_l345_34586


namespace NUMINAMATH_GPT_connie_tickets_l345_34551

variable (T : ℕ)

theorem connie_tickets (h : T = T / 2 + 10 + 15) : T = 50 :=
by 
sorry

end NUMINAMATH_GPT_connie_tickets_l345_34551


namespace NUMINAMATH_GPT_no_solution_inequality_l345_34513

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_solution_inequality_l345_34513


namespace NUMINAMATH_GPT_simplify_and_evaluate_l345_34543

theorem simplify_and_evaluate :
  ∀ (a : ℚ), a = 3 → ((a - 1) / (a + 2) / ((a ^ 2 - 2 * a) / (a ^ 2 - 4)) - (a + 1) / a) = -2 / 3 :=
by
  intros a ha
  have : a = 3 := ha
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l345_34543


namespace NUMINAMATH_GPT_solution_set_of_inequality_l345_34521

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) ≤ 0 ↔ x ≤ 0 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l345_34521


namespace NUMINAMATH_GPT_measure_of_angle_R_l345_34577

variable (S T A R : ℝ) -- Represent the angles as real numbers.

-- The conditions given in the problem.
axiom angles_congruent : S = T ∧ T = A ∧ A = R
axiom angle_A_equals_angle_S : A = S

-- Statement: Prove that the measure of angle R is 108 degrees.
theorem measure_of_angle_R : R = 108 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_R_l345_34577


namespace NUMINAMATH_GPT_find_remainder_l345_34515

def dividend : ℕ := 997
def divisor : ℕ := 23
def quotient : ℕ := 43

theorem find_remainder : ∃ r : ℕ, dividend = (divisor * quotient) + r ∧ r = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_l345_34515


namespace NUMINAMATH_GPT_special_op_2_4_5_l345_34525

def special_op (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem special_op_2_4_5 : special_op 2 4 5 = -24 := by
  sorry

end NUMINAMATH_GPT_special_op_2_4_5_l345_34525


namespace NUMINAMATH_GPT_sum_of_primes_1_to_20_l345_34509

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end NUMINAMATH_GPT_sum_of_primes_1_to_20_l345_34509


namespace NUMINAMATH_GPT_sandwiches_lunch_monday_l345_34552

-- Define the conditions
variables (L : ℕ) 
variables (sandwiches_monday sandwiches_tuesday : ℕ)
variables (h1 : sandwiches_monday = L + 2 * L)
variables (h2 : sandwiches_tuesday = 1)

-- Define the fact that he ate 8 more sandwiches on Monday compared to Tuesday.
variables (h3 : sandwiches_monday = sandwiches_tuesday + 8)

theorem sandwiches_lunch_monday : L = 3 := 
by
  -- We need to prove L = 3 given the conditions (h1, h2, h3)
  -- Here is where the necessary proof would be constructed
  -- This placeholder indicates a proof needs to be inserted here
  sorry

end NUMINAMATH_GPT_sandwiches_lunch_monday_l345_34552


namespace NUMINAMATH_GPT_x_power_2023_zero_or_neg_two_l345_34585

variable {x : ℂ} -- Assuming x is a complex number to handle general roots of unity.

theorem x_power_2023_zero_or_neg_two 
  (h1 : (x - 1) * (x + 1) = x^2 - 1)
  (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
  (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
  (pattern : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) :
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 :=
by
  sorry

end NUMINAMATH_GPT_x_power_2023_zero_or_neg_two_l345_34585


namespace NUMINAMATH_GPT_number_of_valid_arrangements_l345_34538

def total_permutations (n : ℕ) : ℕ := n.factorial

def valid_permutations (total : ℕ) (block : ℕ) (specific_restriction : ℕ) : ℕ :=
  total - specific_restriction

theorem number_of_valid_arrangements : valid_permutations (total_permutations 5) 48 24 = 96 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_arrangements_l345_34538


namespace NUMINAMATH_GPT_isosceles_triangle_relation_range_l345_34505

-- Definitions of the problem conditions and goal
variables (x y : ℝ)

-- Given conditions
def isosceles_triangle (x y : ℝ) :=
  x + x + y = 10

-- Prove the relationship and range 
theorem isosceles_triangle_relation_range (h : isosceles_triangle x y) :
  y = 10 - 2 * x ∧ (5 / 2 < x ∧ x < 5) :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_relation_range_l345_34505


namespace NUMINAMATH_GPT_find_fraction_l345_34582

theorem find_fraction
  (F : ℚ) (m : ℕ) 
  (h1 : F^m * (1 / 4)^2 = 1 / 10^4)
  (h2 : m = 4) : 
  F = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l345_34582


namespace NUMINAMATH_GPT_train_length_l345_34550

theorem train_length (L : ℝ) 
    (cross_bridge : ∀ (t_bridge : ℝ), t_bridge = 10 → L + 200 = t_bridge * (L / 5))
    (cross_lamp_post : ∀ (t_lamp_post : ℝ), t_lamp_post = 5 → L = t_lamp_post * (L / 5)) :
  L = 200 := 
by 
  -- sorry is used to skip the proof part
  sorry

end NUMINAMATH_GPT_train_length_l345_34550


namespace NUMINAMATH_GPT_lions_deers_15_minutes_l345_34500

theorem lions_deers_15_minutes :
  ∀ (n : ℕ), (15 * n = 15 * 15 → n = 15 → ∀ t, t = 15) := by
  sorry

end NUMINAMATH_GPT_lions_deers_15_minutes_l345_34500


namespace NUMINAMATH_GPT_problem1_problem2_l345_34526

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- First problem: A ∩ B
theorem problem1 (a : ℝ) (ha : a = 4) : A a ∩ B = {x | 6 < x ∧ x ≤ 7} :=
by sorry

-- Second problem: A ∪ B = B
theorem problem2 (a : ℝ) : (A a ∪ B = B) ↔ (a < -4 ∨ a > 5) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l345_34526


namespace NUMINAMATH_GPT_abs_a_gt_neg_b_l345_34569

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end NUMINAMATH_GPT_abs_a_gt_neg_b_l345_34569
