import Mathlib

namespace NUMINAMATH_GPT_cloth_woven_on_30th_day_l1541_154195

theorem cloth_woven_on_30th_day :
  (∃ d : ℚ, (30 * 5 + ((30 * 29) / 2) * d = 390) ∧ (5 + 29 * d = 21)) :=
by sorry

end NUMINAMATH_GPT_cloth_woven_on_30th_day_l1541_154195


namespace NUMINAMATH_GPT_JohnsonsYield_l1541_154106

def JohnsonYieldPerTwoMonths (J : ℕ) : Prop :=
  ∀ (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ),
    neighbor_hectares = 2 →
    neighbor_yield_per_hectare = 2 * J →
    total_yield_six_months = 1200 →
    3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months →
    J = 80

theorem JohnsonsYield
  (J : ℕ)
  (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ)
  (h1 : neighbor_hectares = 2)
  (h2 : neighbor_yield_per_hectare = 2 * J)
  (h3 : total_yield_six_months = 1200)
  (h4 : 3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months) :
  J = 80 :=
by
  sorry

end NUMINAMATH_GPT_JohnsonsYield_l1541_154106


namespace NUMINAMATH_GPT_tan_double_angle_l1541_154101

theorem tan_double_angle (α : ℝ) (h1 : Real.cos (Real.pi - α) = 4 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan (2 * α) = 24 / 7 := 
sorry

end NUMINAMATH_GPT_tan_double_angle_l1541_154101


namespace NUMINAMATH_GPT_parabola_condition_l1541_154177

/-- Given the point (3,0) lies on the parabola y = 2x^2 + (k + 2)x - k,
    prove that k = -12. -/
theorem parabola_condition (k : ℝ) (h : 0 = 2 * 3^2 + (k + 2) * 3 - k) : k = -12 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_condition_l1541_154177


namespace NUMINAMATH_GPT_weight_of_packet_a_l1541_154143

theorem weight_of_packet_a
  (A B C D E F : ℝ)
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 79)
  (h5 : F = (A + E) / 2)
  (h6 : (B + C + D + E + F) / 5 = 81) :
  A = 75 :=
by sorry

end NUMINAMATH_GPT_weight_of_packet_a_l1541_154143


namespace NUMINAMATH_GPT_number_of_true_propositions_l1541_154148

open Classical

-- Define each proposition as a term or lemma in Lean
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 > 0
def prop2 : Prop := ∀ x : ℕ, x^4 ≥ 1
def prop3 : Prop := ∃ x : ℤ, x^3 < 1
def prop4 : Prop := ∀ x : ℚ, x^2 ≠ 2

-- The main theorem statement that the number of true propositions is 3 given the conditions
theorem number_of_true_propositions : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 → 3 = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1541_154148


namespace NUMINAMATH_GPT_incorrect_statement_l1541_154162

theorem incorrect_statement : ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_incorrect_statement_l1541_154162


namespace NUMINAMATH_GPT_sixth_number_of_11_consecutive_odd_sum_1991_is_181_l1541_154135

theorem sixth_number_of_11_consecutive_odd_sum_1991_is_181 :
  (∃ (n : ℤ), (2 * n + 1) + (2 * n + 3) + (2 * n + 5) + (2 * n + 7) + (2 * n + 9) + (2 * n + 11) + (2 * n + 13) + (2 * n + 15) + (2 * n + 17) + (2 * n + 19) + (2 * n + 21) = 1991) →
  2 * 85 + 11 = 181 := 
by
  sorry

end NUMINAMATH_GPT_sixth_number_of_11_consecutive_odd_sum_1991_is_181_l1541_154135


namespace NUMINAMATH_GPT_visual_range_percent_increase_l1541_154169

-- Define the original and new visual ranges
def original_range : ℝ := 90
def new_range : ℝ := 150

-- Define the desired percent increase as a real number
def desired_percent_increase : ℝ := 66.67

-- The theorem to prove that the visual range is increased by the desired percentage
theorem visual_range_percent_increase :
  ((new_range - original_range) / original_range) * 100 = desired_percent_increase := 
sorry

end NUMINAMATH_GPT_visual_range_percent_increase_l1541_154169


namespace NUMINAMATH_GPT_range_of_m_l1541_154128

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 + m * x + 1 = 0 → x ≠ 0) ∧ ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1541_154128


namespace NUMINAMATH_GPT_machineB_produces_100_parts_in_40_minutes_l1541_154121

-- Define the given conditions
def machineA_rate := 50 / 10 -- Machine A's rate in parts per minute
def machineB_rate := machineA_rate / 2 -- Machine B's rate in parts per minute

-- Machine A produces 50 parts in 10 minutes
def machineA_50_parts_time : ℝ := 10

-- Machine B's time to produce 100 parts (The question)
def machineB_100_parts_time : ℝ := 40

-- Proving that Machine B takes 40 minutes to produce 100 parts
theorem machineB_produces_100_parts_in_40_minutes :
    machineB_100_parts_time = 40 :=
by
  sorry

end NUMINAMATH_GPT_machineB_produces_100_parts_in_40_minutes_l1541_154121


namespace NUMINAMATH_GPT_error_percent_in_area_l1541_154102

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

end NUMINAMATH_GPT_error_percent_in_area_l1541_154102


namespace NUMINAMATH_GPT_find_a_l1541_154168

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (2 * x) - (1 / 3) * Real.sin (3 * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  2 * a * Real.cos (2 * x) - Real.cos (3 * x)

theorem find_a (a : ℝ) (h : f_prime a (Real.pi / 3) = 0) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1541_154168


namespace NUMINAMATH_GPT_prism_volume_l1541_154103

theorem prism_volume (a b c : ℝ) (h1 : a * b = 12) (h2 : b * c = 8) (h3 : a * c = 4) : a * b * c = 8 * Real.sqrt 6 :=
by 
  sorry

end NUMINAMATH_GPT_prism_volume_l1541_154103


namespace NUMINAMATH_GPT_problem1_problem2_l1541_154142

theorem problem1 (x y : ℝ) : (x + y) * (x - y) + y * (y - 2) = x^2 - 2 * y :=
by 
  sorry

theorem problem2 (m : ℝ) (h : m ≠ 2) : (1 - m / (m + 2)) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 2 / (m - 2) :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1541_154142


namespace NUMINAMATH_GPT_find_number_l1541_154107

theorem find_number (x : ℤ) (h : 45 - (28 - (37 - (x - 18))) = 57) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1541_154107


namespace NUMINAMATH_GPT_roots_cubic_sum_l1541_154189

theorem roots_cubic_sum:
  (∃ p q r : ℝ, 
     (p^3 - p^2 + p - 2 = 0) ∧ 
     (q^3 - q^2 + q - 2 = 0) ∧ 
     (r^3 - r^2 + r - 2 = 0)) 
  → 
  (∃ p q r : ℝ, p^3 + q^3 + r^3 = 4) := 
by 
  sorry

end NUMINAMATH_GPT_roots_cubic_sum_l1541_154189


namespace NUMINAMATH_GPT_unique_square_friendly_l1541_154198

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k^2 = n

def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18 * m + c)

theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 := 
sorry

end NUMINAMATH_GPT_unique_square_friendly_l1541_154198


namespace NUMINAMATH_GPT_factors_and_divisors_l1541_154179

theorem factors_and_divisors :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (¬(∃ n : ℕ, 209 = 19 * n ∧ ¬ (∃ m : ℕ, 57 = 19 * m))) ∧
  (¬(¬(∃ n : ℕ, 90 = 30 * n) ∧ ¬(∃ m : ℕ, 75 = 30 * m))) ∧
  (¬(∃ n : ℕ, 51 = 17 * n ∧ ¬ (∃ m : ℕ, 68 = 17 * m))) ∧
  (∃ n : ℕ, 171 = 9 * n) :=
by {
  sorry
}

end NUMINAMATH_GPT_factors_and_divisors_l1541_154179


namespace NUMINAMATH_GPT_find_t_for_area_of_triangle_l1541_154180

theorem find_t_for_area_of_triangle :
  ∃ (t : ℝ), 
  (∀ (A B C T U: ℝ × ℝ),
    A = (0, 10) → 
    B = (3, 0) → 
    C = (9, 0) → 
    T = (3/10 * (10 - t), t) →
    U = (9/10 * (10 - t), t) →
    2 * 15 = 3/10 * (10 - t) ^ 2) →
  t = 2.93 :=
by sorry

end NUMINAMATH_GPT_find_t_for_area_of_triangle_l1541_154180


namespace NUMINAMATH_GPT_parallel_lines_slopes_l1541_154144

theorem parallel_lines_slopes (k : ℝ) :
  (∀ x y : ℝ, x + (1 + k) * y = 2 - k → k * x + 2 * y + 8 = 0 → k = 1) :=
by
  intro h1 h2
  -- We can see that there should be specifics here about how the conditions lead to k = 1
  sorry

end NUMINAMATH_GPT_parallel_lines_slopes_l1541_154144


namespace NUMINAMATH_GPT_mean_cars_l1541_154132

theorem mean_cars (a b c d e : ℝ) (h1 : a = 30) (h2 : b = 14) (h3 : c = 14) (h4 : d = 21) (h5 : e = 25) : 
  (a + b + c + d + e) / 5 = 20.8 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_mean_cars_l1541_154132


namespace NUMINAMATH_GPT_f_fraction_neg_1987_1988_l1541_154146

-- Define the function f and its properties
def f : ℚ → ℝ := sorry

axiom functional_eq (x y : ℚ) : f (x + y) = f x * f y - f (x * y) + 1
axiom not_equal_f : f 1988 ≠ f 1987

-- Prove the desired equality
theorem f_fraction_neg_1987_1988 : f (-1987 / 1988) = 1 / 1988 :=
by
  sorry

end NUMINAMATH_GPT_f_fraction_neg_1987_1988_l1541_154146


namespace NUMINAMATH_GPT_jerusha_earnings_l1541_154160

theorem jerusha_earnings (L : ℕ) (h1 : 5 * L = 85) : 4 * L = 68 := 
by
  sorry

end NUMINAMATH_GPT_jerusha_earnings_l1541_154160


namespace NUMINAMATH_GPT_remainder_2011_2015_mod_23_l1541_154112

theorem remainder_2011_2015_mod_23 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_2011_2015_mod_23_l1541_154112


namespace NUMINAMATH_GPT_second_chick_eats_52_l1541_154145

theorem second_chick_eats_52 (days : ℕ) (first_chick_eats : ℕ → ℕ) (second_chick_eats : ℕ → ℕ) :
  (∀ n, first_chick_eats n + second_chick_eats n = 12) →
  (∃ a b, first_chick_eats a = 7 ∧ second_chick_eats a = 5 ∧
          first_chick_eats b = 7 ∧ second_chick_eats b = 5 ∧
          12 * days = first_chick_eats a * 2 + first_chick_eats b * 6 + second_chick_eats a * 2 + second_chick_eats b * 6) →
  (first_chick_eats a * 2 + first_chick_eats b * 6 = 44) →
  (second_chick_eats a * 2 + second_chick_eats b * 6 = 52) :=
by
  sorry

end NUMINAMATH_GPT_second_chick_eats_52_l1541_154145


namespace NUMINAMATH_GPT_amount_of_rice_distributed_in_first_5_days_l1541_154104

-- Definitions from conditions
def workers_day (d : ℕ) : ℕ := if d = 1 then 64 else 64 + 7 * (d - 1)

-- The amount of rice each worker receives per day
def rice_per_worker : ℕ := 3

-- Total workers dispatched in the first 5 days
def total_workers_first_5_days : ℕ := (workers_day 1 + workers_day 2 + workers_day 3 + workers_day 4 + workers_day 5)

-- Given these definitions, we now state the theorem to prove
theorem amount_of_rice_distributed_in_first_5_days : total_workers_first_5_days * rice_per_worker = 1170 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_rice_distributed_in_first_5_days_l1541_154104


namespace NUMINAMATH_GPT_restore_triangle_ABC_l1541_154122

-- let I be the incenter of triangle ABC
variable (I : Point)
-- let Ic be the C-excenter of triangle ABC
variable (I_c : Point)
-- let H be the foot of the altitude from vertex C to side AB
variable (H : Point)

-- Claim: Given I, I_c, H, we can recover the original triangle ABC
theorem restore_triangle_ABC (I I_c H : Point) : ExistsTriangleABC :=
sorry

end NUMINAMATH_GPT_restore_triangle_ABC_l1541_154122


namespace NUMINAMATH_GPT_product_roots_example_l1541_154170

def cubic_eq (a b c d : ℝ) (x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

noncomputable def product_of_roots (a b c d : ℝ) : ℝ := -d / a

theorem product_roots_example : product_of_roots 4 (-2) (-25) 36 = -9 := by
  sorry

end NUMINAMATH_GPT_product_roots_example_l1541_154170


namespace NUMINAMATH_GPT_medical_bills_value_l1541_154167

variable (M : ℝ)
variable (property_damage : ℝ := 40000)
variable (insurance_coverage : ℝ := 0.80)
variable (carl_coverage : ℝ := 0.20)
variable (carl_owes : ℝ := 22000)

theorem medical_bills_value : 0.20 * (property_damage + M) = carl_owes → M = 70000 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_medical_bills_value_l1541_154167


namespace NUMINAMATH_GPT_positive_integer_prime_condition_l1541_154172

theorem positive_integer_prime_condition (n : ℕ) 
  (h1 : 0 < n)
  (h2 : ∀ (k : ℕ), k < n → Nat.Prime (4 * k^2 + n)) : 
  n = 3 ∨ n = 7 := 
sorry

end NUMINAMATH_GPT_positive_integer_prime_condition_l1541_154172


namespace NUMINAMATH_GPT_relationship_between_n_and_m_l1541_154184

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def geometric_sequence (a q : ℝ) (m : ℕ) : ℝ :=
  a * q ^ (m - 1)

theorem relationship_between_n_and_m
  (a d q : ℝ) (n m : ℕ)
  (h_d_ne_zero : d ≠ 0)
  (h1 : arithmetic_sequence a d 1 = geometric_sequence a q 1)
  (h2 : arithmetic_sequence a d 3 = geometric_sequence a q 3)
  (h3 : arithmetic_sequence a d 7 = geometric_sequence a q 5)
  (q_pos : 0 < q) (q_sqrt2 : q^2 = 2)
  :
  n = 2 ^ ((m + 1) / 2) - 1 := sorry

end NUMINAMATH_GPT_relationship_between_n_and_m_l1541_154184


namespace NUMINAMATH_GPT_arnold_plates_count_l1541_154141

def arnold_barbell := 45
def mistaken_weight := 600
def actual_weight := 470
def weight_difference_per_plate := 10

theorem arnold_plates_count : 
  ∃ n : ℕ, mistaken_weight - actual_weight = n * weight_difference_per_plate ∧ n = 13 := 
sorry

end NUMINAMATH_GPT_arnold_plates_count_l1541_154141


namespace NUMINAMATH_GPT_Ann_is_16_l1541_154129

variable (A S : ℕ)

theorem Ann_is_16
  (h1 : A = S + 5)
  (h2 : A + S = 27) :
  A = 16 :=
by
  sorry

end NUMINAMATH_GPT_Ann_is_16_l1541_154129


namespace NUMINAMATH_GPT_probability_detecting_drunk_driver_l1541_154131

namespace DrunkDrivingProbability

def P_A : ℝ := 0.05
def P_B_given_A : ℝ := 0.99
def P_B_given_not_A : ℝ := 0.01

def P_not_A : ℝ := 1 - P_A

def P_B : ℝ := P_A * P_B_given_A + P_not_A * P_B_given_not_A

theorem probability_detecting_drunk_driver :
  P_B = 0.059 :=
by
  sorry

end DrunkDrivingProbability

end NUMINAMATH_GPT_probability_detecting_drunk_driver_l1541_154131


namespace NUMINAMATH_GPT_probability_A_wins_probability_A_wins_2_l1541_154150

def binomial (n k : ℕ) := Nat.choose n k

noncomputable def P (n : ℕ) : ℚ := 
  1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n))

theorem probability_A_wins (n : ℕ) : P n = 1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n)) := 
by sorry

theorem probability_A_wins_2 : P 2 = 5 / 16 := 
by sorry

end NUMINAMATH_GPT_probability_A_wins_probability_A_wins_2_l1541_154150


namespace NUMINAMATH_GPT_notebooks_ratio_l1541_154156

variable (C N : Nat)

theorem notebooks_ratio (h1 : 512 = C * N)
  (h2 : 512 = 16 * (C / 2)) :
  N = C / 8 :=
by
  sorry

end NUMINAMATH_GPT_notebooks_ratio_l1541_154156


namespace NUMINAMATH_GPT_good_oranges_per_month_l1541_154192

/-- Salaria has 50% of tree A and 50% of tree B, totaling to 10 trees.
    Tree A gives 10 oranges a month and 60% are good.
    Tree B gives 15 oranges a month and 1/3 are good.
    Prove that the total number of good oranges Salaria gets per month is 55. -/
theorem good_oranges_per_month 
  (total_trees : ℕ) 
  (percent_tree_A : ℝ) 
  (percent_tree_B : ℝ) 
  (oranges_tree_A : ℕ)
  (good_percent_A : ℝ)
  (oranges_tree_B : ℕ)
  (good_ratio_B : ℝ)
  (H1 : total_trees = 10)
  (H2 : percent_tree_A = 0.5)
  (H3 : percent_tree_B = 0.5)
  (H4 : oranges_tree_A = 10)
  (H5 : good_percent_A = 0.6)
  (H6 : oranges_tree_B = 15)
  (H7 : good_ratio_B = 1/3)
  : (total_trees * percent_tree_A * oranges_tree_A * good_percent_A) + 
    (total_trees * percent_tree_B * oranges_tree_B * good_ratio_B) = 55 := 
  by 
    sorry

end NUMINAMATH_GPT_good_oranges_per_month_l1541_154192


namespace NUMINAMATH_GPT_union_M_N_eq_interval_l1541_154119

variable {α : Type*} [PartialOrder α]

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem union_M_N_eq_interval :
  M ∪ N = {x | -1/2 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_union_M_N_eq_interval_l1541_154119


namespace NUMINAMATH_GPT_sausage_left_l1541_154137

variables (S x y : ℝ)

-- Conditions
axiom dog_bites : y = x + 300
axiom cat_bites : x = y + 500

-- Theorem Statement
theorem sausage_left {S x y : ℝ}
  (h1 : y = x + 300)
  (h2 : x = y + 500) : S - x - y = 400 :=
by
  sorry

end NUMINAMATH_GPT_sausage_left_l1541_154137


namespace NUMINAMATH_GPT_meters_of_cloth_sold_l1541_154133

-- Definitions based on conditions
def total_selling_price : ℕ := 8925
def profit_per_meter : ℕ := 20
def cost_price_per_meter : ℕ := 85
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof statement
theorem meters_of_cloth_sold : ∃ x : ℕ, selling_price_per_meter * x = total_selling_price ∧ x = 85 := by
  sorry

end NUMINAMATH_GPT_meters_of_cloth_sold_l1541_154133


namespace NUMINAMATH_GPT_hyperbola_condition_sufficiency_l1541_154120

theorem hyperbola_condition_sufficiency (k : ℝ) :
  (k > 3) → (∃ x y : ℝ, (x^2)/(3-k) + (y^2)/(k-1) = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_sufficiency_l1541_154120


namespace NUMINAMATH_GPT_min_value_of_x2_y2_z2_l1541_154100

noncomputable def min_square_sum (x y z k : ℝ) : ℝ :=
  x^2 + y^2 + z^2

theorem min_value_of_x2_y2_z2 (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = k) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (x y z k : ℝ), (x^3 + y^3 + z^3 - 3 * x * y * z = k ∧ k ≥ -1) -> min_square_sum x y z k ≥ min_val :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_x2_y2_z2_l1541_154100


namespace NUMINAMATH_GPT_geometric_series_sum_example_l1541_154111

-- Define the finite geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- State the theorem
theorem geometric_series_sum_example :
  geometric_series_sum (1/2) (1/2) 8 = 255 / 256 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_example_l1541_154111


namespace NUMINAMATH_GPT_find_alpha_l1541_154164

def point (α : ℝ) : Prop := 3^α = Real.sqrt 3

theorem find_alpha (α : ℝ) (h : point α) : α = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_find_alpha_l1541_154164


namespace NUMINAMATH_GPT_total_hours_worked_l1541_154157

theorem total_hours_worked
  (x : ℕ)
  (h1 : 5 * x = 55)
  : 2 * x + 3 * x + 5 * x = 110 :=
by 
  sorry

end NUMINAMATH_GPT_total_hours_worked_l1541_154157


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l1541_154197

-- Definition of a geometric sequence and given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 1 / 4) : q = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l1541_154197


namespace NUMINAMATH_GPT_sum_of_squares_eq_229_l1541_154115

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_eq_229_l1541_154115


namespace NUMINAMATH_GPT_find_probability_between_0_and_1_l1541_154127

-- Define a random variable X following a normal distribution N(μ, σ²)
variables {X : ℝ → ℝ} {μ σ : ℝ}
-- Define conditions:
-- Condition 1: X follows a normal distribution with mean μ and variance σ²
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  sorry  -- Assume properties of normal distribution are satisfied

-- Condition 2: P(X < 1) = 1/2
def P_X_lt_1 : Prop := 
  sorry  -- Assume that P(X < 1) = 1/2

-- Condition 3: P(X > 2) = p
def P_X_gt_2 (p : ℝ) : Prop := 
  sorry  -- Assume that P(X > 2) = p

noncomputable
def probability_X_between_0_and_1 (p : ℝ) : ℝ :=
  1/2 - p

theorem find_probability_between_0_and_1 (X : ℝ → ℝ) {μ σ p : ℝ} 
  (hX : normal_dist X μ σ)
  (h1 : P_X_lt_1)
  (h2 : P_X_gt_2 p) :
  probability_X_between_0_and_1 p = 1/2 - p := 
  sorry

end NUMINAMATH_GPT_find_probability_between_0_and_1_l1541_154127


namespace NUMINAMATH_GPT_prove_percentage_cats_adopted_each_month_l1541_154152

noncomputable def percentage_cats_adopted_each_month
    (initial_dogs : ℕ)
    (initial_cats : ℕ)
    (initial_lizards : ℕ)
    (adopted_dogs_percent : ℕ)
    (adopted_lizards_percent : ℕ)
    (new_pets_each_month : ℕ)
    (total_pets_after_month : ℕ)
    (adopted_cats_percent : ℕ) : Prop :=
  initial_dogs = 30 ∧
  initial_cats = 28 ∧
  initial_lizards = 20 ∧
  adopted_dogs_percent = 50 ∧
  adopted_lizards_percent = 20 ∧
  new_pets_each_month = 13 ∧
  total_pets_after_month = 65 →
  adopted_cats_percent = 25

-- The condition to prove
theorem prove_percentage_cats_adopted_each_month :
  percentage_cats_adopted_each_month 30 28 20 50 20 13 65 25 :=
by 
  sorry

end NUMINAMATH_GPT_prove_percentage_cats_adopted_each_month_l1541_154152


namespace NUMINAMATH_GPT_shortest_side_length_rectangular_solid_geometric_progression_l1541_154113

theorem shortest_side_length_rectangular_solid_geometric_progression
  (b s : ℝ)
  (h1 : (b^3 / s) = 512)
  (h2 : 2 * ((b^2 / s) + (b^2 * s) + b^2) = 384)
  : min (b / s) (min b (b * s)) = 8 := 
sorry

end NUMINAMATH_GPT_shortest_side_length_rectangular_solid_geometric_progression_l1541_154113


namespace NUMINAMATH_GPT_area_under_parabola_l1541_154108

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- State the theorem about the area under the curve
theorem area_under_parabola : (∫ x in (1 : ℝ)..3, parabola x) = 4 / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_area_under_parabola_l1541_154108


namespace NUMINAMATH_GPT_science_students_count_l1541_154166

def total_students := 400 + 120
def local_arts_students := 0.50 * 400
def local_commerce_students := 0.85 * 120
def total_local_students := 327

theorem science_students_count :
  0.25 * S = 25 →
  S = 100 :=
by
  sorry

end NUMINAMATH_GPT_science_students_count_l1541_154166


namespace NUMINAMATH_GPT_find_B_l1541_154155

theorem find_B (A B C : ℝ) (h1 : A = B + C) (h2 : A + B = 1/25) (h3 : C = 1/35) : B = 1/175 :=
by
  sorry

end NUMINAMATH_GPT_find_B_l1541_154155


namespace NUMINAMATH_GPT_slope_of_line_joining_solutions_l1541_154174

theorem slope_of_line_joining_solutions (x1 x2 y1 y2 : ℝ) :
  (4 / x1 + 5 / y1 = 1) → (4 / x2 + 5 / y2 = 1) →
  (x1 ≠ x2) → (y1 = 5 * x1 / (4 * x1 - 1)) → (y2 = 5 * x2 / (4 * x2 - 1)) →
  (x1 ≠ 1 / 4) → (x2 ≠ 1 / 4) →
  ((y2 - y1) / (x2 - x1) = - (5 / 21)) :=
by
  intros h_eq1 h_eq2 h_neq h_y1 h_y2 h_x1 h_x2
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_slope_of_line_joining_solutions_l1541_154174


namespace NUMINAMATH_GPT_ordering_of_variables_l1541_154123

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem ordering_of_variables 
  (a b c : ℝ)
  (ha : a - 2 = Real.log (a / 2))
  (hb : b - 3 = Real.log (b / 3))
  (hc : c - 3 = Real.log (c / 2))
  (ha_pos : 0 < a) (ha_lt_one : a < 1)
  (hb_pos : 0 < b) (hb_lt_one : b < 1)
  (hc_pos : 0 < c) (hc_lt_one : c < 1) :
  c < b ∧ b < a :=
sorry

end NUMINAMATH_GPT_ordering_of_variables_l1541_154123


namespace NUMINAMATH_GPT_find_k_l1541_154190

theorem find_k (x y k : ℤ) 
  (h1 : 2 * x - y = 5 * k + 6) 
  (h2 : 4 * x + 7 * y = k) 
  (h3 : x + y = 2023) : 
  k = 2022 := 
  by 
    sorry

end NUMINAMATH_GPT_find_k_l1541_154190


namespace NUMINAMATH_GPT_diameter_of_triple_sphere_l1541_154139

noncomputable def radius_of_sphere : ℝ := 6

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

noncomputable def triple_volume_of_sphere (r : ℝ) : ℝ := 3 * volume_of_sphere r

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem diameter_of_triple_sphere (r : ℝ) (V1 V2 : ℝ) (a b : ℝ) 
  (h_r : r = radius_of_sphere)
  (h_V1 : V1 = volume_of_sphere r)
  (h_V2 : V2 = triple_volume_of_sphere r)
  (h_d : 12 * cube_root 3 = 2 * (6 * cube_root 3))
  : a + b = 15 :=
sorry

end NUMINAMATH_GPT_diameter_of_triple_sphere_l1541_154139


namespace NUMINAMATH_GPT_find_a7_l1541_154165

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end NUMINAMATH_GPT_find_a7_l1541_154165


namespace NUMINAMATH_GPT_number_of_solutions_l1541_154194

theorem number_of_solutions : ∃! (xy : ℕ × ℕ), (xy.1 ^ 2 - xy.2 ^ 2 = 91 ∧ xy.1 > 0 ∧ xy.2 > 0) := sorry

end NUMINAMATH_GPT_number_of_solutions_l1541_154194


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l1541_154173

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 16) (h2 : a * r^6 = 2) : a * r^4 = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l1541_154173


namespace NUMINAMATH_GPT_unique_positive_integer_b_quadratic_solution_l1541_154147

theorem unique_positive_integer_b_quadratic_solution (c : ℝ) :
  (∃! (b : ℕ), ∀ (x : ℝ), x^2 + (b^2 + (1 / b^2)) * x + c = 3) ↔ c = 5 :=
sorry

end NUMINAMATH_GPT_unique_positive_integer_b_quadratic_solution_l1541_154147


namespace NUMINAMATH_GPT_arithmetic_geom_seq_l1541_154118

noncomputable def geom_seq (a q : ℝ) : ℕ → ℝ 
| 0     => a
| (n+1) => q * (geom_seq a q n)

theorem arithmetic_geom_seq
  (a q : ℝ)
  (h_arith : 2 * geom_seq a q 1 = 1 + (geom_seq a q 2 - 1))
  (h_q : q = 2) :
  (geom_seq a q 2 + geom_seq a q 3) / (geom_seq a q 4 + geom_seq a q 5) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geom_seq_l1541_154118


namespace NUMINAMATH_GPT_book_pages_l1541_154126

theorem book_pages (total_pages : ℝ) : 
  (0.1 * total_pages + 0.25 * total_pages + 30 = 0.5 * total_pages) → 
  total_pages = 240 :=
by
  sorry

end NUMINAMATH_GPT_book_pages_l1541_154126


namespace NUMINAMATH_GPT_unique_a_for_three_distinct_real_solutions_l1541_154186

theorem unique_a_for_three_distinct_real_solutions (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - 2 * x + 1 - 3 * |x|) ∧
  ((∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) ∧
  (∀ x4 : ℝ, f x4 = 0 → (x4 = x1 ∨ x4 = x2 ∨ x4 = x3) )) ) ↔
  a = 1 / 4 :=
sorry

end NUMINAMATH_GPT_unique_a_for_three_distinct_real_solutions_l1541_154186


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1541_154116

variables (a b : ℚ)

theorem value_of_a_minus_b (h1 : |a| = 5) (h2 : |b| = 2) (h3 : |a + b| = a + b) : a - b = 3 ∨ a - b = 7 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1541_154116


namespace NUMINAMATH_GPT_sin_value_l1541_154171

open Real

-- Define the given conditions
variables (x : ℝ) (h1 : cos (π + x) = 3 / 5) (h2 : π < x) (h3 : x < 2 * π)

-- State the problem to be proved
theorem sin_value : sin x = - 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_value_l1541_154171


namespace NUMINAMATH_GPT_product_of_roots_l1541_154163

theorem product_of_roots (Q : Polynomial ℚ) (hQ : Q.degree = 1) (h_root : Q.eval 6 = 0) :
  (Q.roots : Multiset ℚ).prod = 6 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l1541_154163


namespace NUMINAMATH_GPT_cone_height_l1541_154176

theorem cone_height (V : ℝ) (h : ℝ) (r : ℝ) (vertex_angle : ℝ) 
  (H1 : V = 16384 * Real.pi)
  (H2 : vertex_angle = 90) 
  (H3 : V = (1 / 3) * Real.pi * r^2 * h)
  (H4 : h = r) : 
  h = 36.6 :=
by
  sorry

end NUMINAMATH_GPT_cone_height_l1541_154176


namespace NUMINAMATH_GPT_sqrt_expression_value_l1541_154158

theorem sqrt_expression_value :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_value_l1541_154158


namespace NUMINAMATH_GPT_sum_powers_eq_34_over_3_l1541_154125

theorem sum_powers_eq_34_over_3 (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6):
  a^4 + b^4 + c^4 = 34 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_powers_eq_34_over_3_l1541_154125


namespace NUMINAMATH_GPT_total_cost_l1541_154196

-- Definition of the conditions
def cost_sharing (x : ℝ) : Prop :=
  let initial_cost := x / 5
  let new_cost := x / 7
  initial_cost - 15 = new_cost

-- The statement we need to prove
theorem total_cost (x : ℝ) (h : cost_sharing x) : x = 262.50 := by
  sorry

end NUMINAMATH_GPT_total_cost_l1541_154196


namespace NUMINAMATH_GPT_paint_cost_per_quart_l1541_154117

-- Definitions of conditions
def edge_length (cube_edge_length : ℝ) : Prop := cube_edge_length = 10
def surface_area (s_area : ℝ) : Prop := s_area = 6 * (10^2)
def coverage_per_quart (coverage : ℝ) : Prop := coverage = 120
def total_cost (cost : ℝ) : Prop := cost = 16
def required_quarts (quarts : ℝ) : Prop := quarts = 600 / 120
def cost_per_quart (cost : ℝ) (quarts : ℝ) (price_per_quart : ℝ) : Prop := price_per_quart = cost / quarts

-- Main theorem statement translating the problem into Lean
theorem paint_cost_per_quart {cube_edge_length s_area coverage cost quarts price_per_quart : ℝ} :
  edge_length cube_edge_length →
  surface_area s_area →
  coverage_per_quart coverage →
  total_cost cost →
  required_quarts quarts →
  quarts = s_area / coverage →
  cost_per_quart cost quarts 3.20 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- proof will go here
  sorry

end NUMINAMATH_GPT_paint_cost_per_quart_l1541_154117


namespace NUMINAMATH_GPT_smallest_n_l1541_154138

theorem smallest_n (n : ℕ) (h : n ≥ 2) : 
  (∃ m : ℕ, m * m = (n + 1) * (2 * n + 1) / 6) ↔ n = 337 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1541_154138


namespace NUMINAMATH_GPT_find_a_l1541_154134

theorem find_a (a b c : ℂ) (ha : a.re = a) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 6) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1541_154134


namespace NUMINAMATH_GPT_hyperbola_focus_l1541_154124

theorem hyperbola_focus (m : ℝ) (h : (0, 5) = (0, 5)) : 
  (∀ x y : ℝ, (y^2 / m - x^2 / 9 = 1) → m = 16) :=
sorry

end NUMINAMATH_GPT_hyperbola_focus_l1541_154124


namespace NUMINAMATH_GPT_probability_bernardo_larger_l1541_154185

-- Define the sets from which Bernardo and Silvia are picking numbers
def set_B : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to calculate the probability as described in the problem statement
def bernardo_larger_probability : ℚ := sorry -- The step by step calculations will be inserted here

-- Main theorem stating what needs to be proved
theorem probability_bernardo_larger : bernardo_larger_probability = 61 / 80 := 
sorry

end NUMINAMATH_GPT_probability_bernardo_larger_l1541_154185


namespace NUMINAMATH_GPT_total_customers_is_40_l1541_154105

-- The number of tables the waiter is attending
def num_tables : ℕ := 5

-- The number of women at each table
def women_per_table : ℕ := 5

-- The number of men at each table
def men_per_table : ℕ := 3

-- The total number of customers at each table
def customers_per_table : ℕ := women_per_table + men_per_table

-- The total number of customers the waiter has
def total_customers : ℕ := num_tables * customers_per_table

theorem total_customers_is_40 : total_customers = 40 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_customers_is_40_l1541_154105


namespace NUMINAMATH_GPT_reduced_rates_start_l1541_154178

theorem reduced_rates_start (reduced_fraction : ℝ) (total_hours : ℝ) (weekend_hours : ℝ) (weekday_hours : ℝ) 
  (start_time : ℝ) (end_time : ℝ) : 
  reduced_fraction = 0.6428571428571429 → 
  total_hours = 168 → 
  weekend_hours = 48 → 
  weekday_hours = 60 - weekend_hours → 
  end_time = 8 → 
  start_time = end_time - weekday_hours → 
  start_time = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_reduced_rates_start_l1541_154178


namespace NUMINAMATH_GPT_eq_nine_l1541_154188

theorem eq_nine (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_GPT_eq_nine_l1541_154188


namespace NUMINAMATH_GPT_inequalities_correct_l1541_154161

theorem inequalities_correct (a b : ℝ) (h : a * b > 0) :
  |b| > |a| ∧ |a + b| < |b| := sorry

end NUMINAMATH_GPT_inequalities_correct_l1541_154161


namespace NUMINAMATH_GPT_quotient_calculation_l1541_154159

theorem quotient_calculation (dividend divisor remainder expected_quotient : ℕ)
  (h₁ : dividend = 166)
  (h₂ : divisor = 18)
  (h₃ : remainder = 4)
  (h₄ : dividend = divisor * expected_quotient + remainder) :
  expected_quotient = 9 :=
by
  sorry

end NUMINAMATH_GPT_quotient_calculation_l1541_154159


namespace NUMINAMATH_GPT_prob_both_successful_prob_at_least_one_successful_l1541_154181

variables (P_A P_B : ℚ)
variables (h1 : P_A = 1 / 2)
variables (h2 : P_B = 2 / 5)

/-- Prove that the probability that both A and B score in one shot each is 1 / 5. -/
theorem prob_both_successful (P_A P_B : ℚ) (h1 : P_A = 1 / 2) (h2 : P_B = 2 / 5) :
  P_A * P_B = 1 / 5 :=
by sorry

variables (P_A_miss P_B_miss : ℚ)
variables (h3 : P_A_miss = 1 / 2)
variables (h4 : P_B_miss = 3 / 5)

/-- Prove that the probability that at least one shot is successful is 7 / 10. -/
theorem prob_at_least_one_successful (P_A_miss P_B_miss : ℚ) (h3 : P_A_miss = 1 / 2) (h4 : P_B_miss = 3 / 5) :
  1 - P_A_miss * P_B_miss = 7 / 10 :=
by sorry

end NUMINAMATH_GPT_prob_both_successful_prob_at_least_one_successful_l1541_154181


namespace NUMINAMATH_GPT_find_angle_A_find_area_l1541_154114

-- Define the geometric and trigonometric conditions of the triangle
def triangle (A B C a b c : ℝ) :=
  a = 4 * Real.sqrt 3 ∧ b + c = 8 ∧
  2 * Real.sin A * Real.cos B + Real.sin B = 2 * Real.sin C

-- Prove angle A is 60 degrees
theorem find_angle_A (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : A = Real.pi / 3 := sorry

-- Prove the area of triangle ABC is 4 * sqrt(3) / 3
theorem find_area (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : 
  (1 / 2) * (a * b * Real.sin C) = (4 * Real.sqrt 3) / 3 := sorry

end NUMINAMATH_GPT_find_angle_A_find_area_l1541_154114


namespace NUMINAMATH_GPT_john_piano_lessons_l1541_154191

theorem john_piano_lessons (total_cost piano_cost original_price_per_lesson discount : ℕ) 
    (total_spent : ℕ) : 
    total_spent = piano_cost + ((total_cost - piano_cost) / (original_price_per_lesson - discount)) → 
    total_cost = 1100 ∧ piano_cost = 500 ∧ original_price_per_lesson = 40 ∧ discount = 10 → 
    (total_cost - piano_cost) / (original_price_per_lesson - discount) = 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_john_piano_lessons_l1541_154191


namespace NUMINAMATH_GPT_frac_equiv_l1541_154110

theorem frac_equiv (a b : ℚ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end NUMINAMATH_GPT_frac_equiv_l1541_154110


namespace NUMINAMATH_GPT_total_planks_l1541_154182

-- Define the initial number of planks
def initial_planks : ℕ := 15

-- Define the planks Charlie got
def charlie_planks : ℕ := 10

-- Define the planks Charlie's father got
def father_planks : ℕ := 10

-- Prove the total number of planks
theorem total_planks : (initial_planks + charlie_planks + father_planks) = 35 :=
by sorry

end NUMINAMATH_GPT_total_planks_l1541_154182


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1541_154140

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : Real.tan α = -2)

theorem problem_1 : Real.sin (α + (π / 6)) = (2 * Real.sqrt 15 - Real.sqrt 5) / 10 := by
  sorry

theorem problem_2 : (2 * Real.cos ((π / 2) + α) - Real.cos (π - α)) / (Real.sin ((π / 2) - α) - 3 * Real.sin (π + α)) = 5 / 7 := by
  sorry

theorem problem_3 : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1541_154140


namespace NUMINAMATH_GPT_percent_increase_l1541_154183

def initial_price : ℝ := 15
def final_price : ℝ := 16

theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 6.67 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_l1541_154183


namespace NUMINAMATH_GPT_average_score_first_10_matches_l1541_154199

theorem average_score_first_10_matches (A : ℕ) 
  (h1 : 0 < A) 
  (h2 : 10 * A + 15 * 70 = 25 * 66) : A = 60 :=
by
  sorry

end NUMINAMATH_GPT_average_score_first_10_matches_l1541_154199


namespace NUMINAMATH_GPT_exam_students_count_l1541_154109

theorem exam_students_count (n : ℕ) (T : ℕ) (h1 : T = 90 * n) 
                            (h2 : (T - 90) / (n - 2) = 95) : n = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_exam_students_count_l1541_154109


namespace NUMINAMATH_GPT_jane_total_drawing_paper_l1541_154187

theorem jane_total_drawing_paper (brown_sheets : ℕ) (yellow_sheets : ℕ) 
    (h1 : brown_sheets = 28) (h2 : yellow_sheets = 27) : 
    brown_sheets + yellow_sheets = 55 := 
by
    sorry

end NUMINAMATH_GPT_jane_total_drawing_paper_l1541_154187


namespace NUMINAMATH_GPT_paul_is_19_years_old_l1541_154175

theorem paul_is_19_years_old
  (mark_age : ℕ)
  (alice_age : ℕ)
  (paul_age : ℕ)
  (h1 : mark_age = 20)
  (h2 : alice_age = mark_age + 4)
  (h3 : paul_age = alice_age - 5) : 
  paul_age = 19 := by 
  sorry

end NUMINAMATH_GPT_paul_is_19_years_old_l1541_154175


namespace NUMINAMATH_GPT_base_salary_l1541_154193

theorem base_salary {B : ℝ} {C : ℝ} :
  (B + 200 * C = 2000) → 
  (B + 200 * 15 = 4000) → 
  B = 1000 :=
by
  sorry

end NUMINAMATH_GPT_base_salary_l1541_154193


namespace NUMINAMATH_GPT_muffins_count_l1541_154130

-- Lean 4 Statement
theorem muffins_count (doughnuts muffins : ℕ) (ratio_doughnuts_muffins : ℕ → ℕ → Prop)
  (h_ratio : ratio_doughnuts_muffins 5 1) (h_doughnuts : doughnuts = 50) :
  muffins = 10 :=
by
  sorry

end NUMINAMATH_GPT_muffins_count_l1541_154130


namespace NUMINAMATH_GPT_inequality_solution_l1541_154153

theorem inequality_solution (x : ℝ) :
  (3 / 16) + abs (x - 17 / 64) < 7 / 32 ↔ (15 / 64) < x ∧ x < (19 / 64) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1541_154153


namespace NUMINAMATH_GPT_coin_toss_sequences_count_l1541_154149

theorem coin_toss_sequences_count :
  (∃ (seq : List Char), 
    seq.length = 15 ∧ 
    (seq == ['H', 'H']) = 5 ∧ 
    (seq == ['H', 'T']) = 3 ∧ 
    (seq == ['T', 'H']) = 2 ∧ 
    (seq == ['T', 'T']) = 4) → 
  (count_sequences == 775360) :=
by
  sorry

end NUMINAMATH_GPT_coin_toss_sequences_count_l1541_154149


namespace NUMINAMATH_GPT_coat_price_reduction_l1541_154136

theorem coat_price_reduction:
  ∀ (original_price reduction_amount : ℕ),
  original_price = 500 →
  reduction_amount = 350 →
  (reduction_amount : ℝ) / original_price * 100 = 70 :=
by
  intros original_price reduction_amount h1 h2
  sorry

end NUMINAMATH_GPT_coat_price_reduction_l1541_154136


namespace NUMINAMATH_GPT_least_positive_x_multiple_of_53_l1541_154154

theorem least_positive_x_multiple_of_53 :
  ∃ (x : ℕ), (x > 0) ∧ ((2 * x)^2 + 2 * 47 * (2 * x) + 47^2) % 53 = 0 ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_x_multiple_of_53_l1541_154154


namespace NUMINAMATH_GPT_repair_cost_total_l1541_154151

def hourly_labor_cost : ℝ := 75
def labor_hours : ℝ := 16
def part_cost : ℝ := 1200
def labor_cost : ℝ := hourly_labor_cost * labor_hours
def total_cost : ℝ := labor_cost + part_cost

theorem repair_cost_total : total_cost = 2400 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_repair_cost_total_l1541_154151
