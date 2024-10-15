import Mathlib

namespace NUMINAMATH_GPT_processing_time_l2210_221096

theorem processing_time 
  (pictures : ℕ) (minutes_per_picture : ℕ) (minutes_per_hour : ℕ)
  (h1 : pictures = 960) (h2 : minutes_per_picture = 2) (h3 : minutes_per_hour = 60) : 
  (pictures * minutes_per_picture) / minutes_per_hour = 32 :=
by 
  sorry

end NUMINAMATH_GPT_processing_time_l2210_221096


namespace NUMINAMATH_GPT_perpendicular_line_through_P_l2210_221029

open Real

-- Define the point (1, 0)
def P : ℝ × ℝ := (1, 0)

-- Define the initial line x - 2y - 2 = 0
def initial_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the desired line 2x + y - 2 = 0
def desired_line (x y : ℝ) : Prop := 2 * x + y = 2

-- State that the desired line passes through the point (1, 0) and is perpendicular to the initial line
theorem perpendicular_line_through_P :
  (∃ m b, b ∈ Set.univ ∧ (∀ x y, desired_line x y → y = m * x + b)) ∧ ∀ x y, 
  initial_line x y → x ≠ 0 → desired_line y (-x / 2) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_through_P_l2210_221029


namespace NUMINAMATH_GPT_rectangle_measurement_error_l2210_221077

theorem rectangle_measurement_error
  (L W : ℝ)
  (x : ℝ)
  (h1 : ∀ x, L' = L * (1 + x / 100))
  (h2 : W' = W * 0.9)
  (h3 : A = L * W)
  (h4 : A' = A * 1.08) :
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_measurement_error_l2210_221077


namespace NUMINAMATH_GPT_initial_number_of_mice_l2210_221063

theorem initial_number_of_mice (x : ℕ) 
  (h1 : x % 2 = 0)
  (h2 : (x / 2) % 3 = 0)
  (h3 : (x / 2 - x / 6) % 4 = 0)
  (h4 : (x / 2 - x / 6 - (x / 2 - x / 6) / 4) % 5 = 0)
  (h5 : (x / 5) = (x / 6) + 2) : 
  x = 60 := 
by sorry

end NUMINAMATH_GPT_initial_number_of_mice_l2210_221063


namespace NUMINAMATH_GPT_arc_length_of_sector_l2210_221089

theorem arc_length_of_sector (n r : ℝ) (h_angle : n = 60) (h_radius : r = 3) : 
  (n * Real.pi * r / 180) = Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l2210_221089


namespace NUMINAMATH_GPT_fill_tub_in_seconds_l2210_221020

theorem fill_tub_in_seconds 
  (faucet_rate : ℚ)
  (four_faucet_rate : ℚ := 4 * faucet_rate)
  (three_faucet_rate : ℚ := 3 * faucet_rate)
  (time_for_100_gallons_in_minutes : ℚ := 6)
  (time_for_100_gallons_in_seconds : ℚ := time_for_100_gallons_in_minutes * 60)
  (volume_100_gallons : ℚ := 100)
  (rate_per_three_faucets_in_gallons_per_second : ℚ := volume_100_gallons / time_for_100_gallons_in_seconds)
  (rate_per_faucet : ℚ := rate_per_three_faucets_in_gallons_per_second / 3)
  (rate_per_four_faucets : ℚ := 4 * rate_per_faucet)
  (volume_50_gallons : ℚ := 50)
  (expected_time_seconds : ℚ := volume_50_gallons / rate_per_four_faucets) :
  expected_time_seconds = 135 :=
sorry

end NUMINAMATH_GPT_fill_tub_in_seconds_l2210_221020


namespace NUMINAMATH_GPT_sum_of_squares_eq_frac_squared_l2210_221074

theorem sum_of_squares_eq_frac_squared (x y z a b c : ℝ) (hxya : x * y = a) (hxzb : x * z = b) (hyzc : y * z = c)
  (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hc0 : c ≠ 0) :
  x^2 + y^2 + z^2 = ((a * b)^2 + (a * c)^2 + (b * c)^2) / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_eq_frac_squared_l2210_221074


namespace NUMINAMATH_GPT_fall_increase_l2210_221055

noncomputable def percentage_increase_in_fall (x : ℝ) : ℝ :=
  x

theorem fall_increase :
  ∃ (x : ℝ), (1 + percentage_increase_in_fall x / 100) * (1 - 19 / 100) = 1 + 11.71 / 100 :=
by
  sorry

end NUMINAMATH_GPT_fall_increase_l2210_221055


namespace NUMINAMATH_GPT_interval_necessary_not_sufficient_l2210_221061

theorem interval_necessary_not_sufficient :
  (∀ x, x^2 - x - 2 = 0 → (-1 ≤ x ∧ x ≤ 2)) ∧ (∃ x, x^2 - x - 2 = 0 ∧ ¬(-1 ≤ x ∧ x ≤ 2)) → False :=
by
  sorry

end NUMINAMATH_GPT_interval_necessary_not_sufficient_l2210_221061


namespace NUMINAMATH_GPT_abs_value_x_minus_2_plus_x_plus_3_ge_4_l2210_221036

theorem abs_value_x_minus_2_plus_x_plus_3_ge_4 :
  ∀ x : ℝ, (|x - 2| + |x + 3| ≥ 4) ↔ (x ≤ - (5 / 2)) := 
sorry

end NUMINAMATH_GPT_abs_value_x_minus_2_plus_x_plus_3_ge_4_l2210_221036


namespace NUMINAMATH_GPT_negation_of_exists_l2210_221064

theorem negation_of_exists :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l2210_221064


namespace NUMINAMATH_GPT_calculate_A_l2210_221098

theorem calculate_A (D B E C A : ℝ) :
  D = 2 * 4 →
  B = 2 * D →
  E = 7 * 2 →
  C = 7 * E →
  A^2 = B * C →
  A = 28 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_A_l2210_221098


namespace NUMINAMATH_GPT_quadratic_real_roots_l2210_221082

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ k ≥ -9 / 4 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l2210_221082


namespace NUMINAMATH_GPT_find_large_no_l2210_221023

theorem find_large_no (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
by 
  sorry

end NUMINAMATH_GPT_find_large_no_l2210_221023


namespace NUMINAMATH_GPT_max_value_is_two_over_three_l2210_221000

noncomputable def max_value_expr (x : ℝ) : ℝ := 2^x - 8^x

theorem max_value_is_two_over_three :
  ∃ (x : ℝ), max_value_expr x = 2 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_is_two_over_three_l2210_221000


namespace NUMINAMATH_GPT_multiply_polynomials_l2210_221016

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l2210_221016


namespace NUMINAMATH_GPT_solve_for_y_l2210_221030

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l2210_221030


namespace NUMINAMATH_GPT_range_of_m_l2210_221042

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m) : -3 < m ∧ m < 1 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2210_221042


namespace NUMINAMATH_GPT_even_product_when_eight_cards_drawn_l2210_221068

theorem even_product_when_eight_cards_drawn :
  ∀ (s : Finset ℕ), (∀ n ∈ s, n ∈ Finset.range 15) →
  s.card ≥ 8 →
  (∃ m ∈ s, Even m) :=
by
  sorry

end NUMINAMATH_GPT_even_product_when_eight_cards_drawn_l2210_221068


namespace NUMINAMATH_GPT_total_books_is_10033_l2210_221059

variable (P C B M H : ℕ)
variable (x : ℕ) (h_P : P = 3 * x) (h_C : C = 2 * x)
variable (h_B : B = (3 / 2) * x)
variable (h_M : M = (3 / 5) * x)
variable (h_H : H = (4 / 5) * x)
variable (total_books : ℕ)
variable (h_total : total_books = P + C + B + M + H)
variable (h_bound : total_books > 10000)

theorem total_books_is_10033 : total_books = 10033 :=
  sorry

end NUMINAMATH_GPT_total_books_is_10033_l2210_221059


namespace NUMINAMATH_GPT_ladder_distance_from_wall_l2210_221035

noncomputable def dist_from_wall (ladder_length : ℝ) (angle_deg : ℝ) : ℝ :=
  ladder_length * Real.cos (angle_deg * Real.pi / 180)

theorem ladder_distance_from_wall :
  ∀ (ladder_length : ℝ) (angle_deg : ℝ), ladder_length = 19 → angle_deg = 60 → dist_from_wall ladder_length angle_deg = 9.5 :=
by
  intros ladder_length angle_deg h1 h2
  sorry

end NUMINAMATH_GPT_ladder_distance_from_wall_l2210_221035


namespace NUMINAMATH_GPT_find_sol_y_pct_l2210_221039

-- Define the conditions
def sol_x_vol : ℕ := 200            -- Volume of solution x in milliliters
def sol_y_vol : ℕ := 600            -- Volume of solution y in milliliters
def sol_x_pct : ℕ := 10             -- Percentage of alcohol in solution x
def final_sol_pct : ℕ := 25         -- Percentage of alcohol in the final solution
def final_sol_vol := sol_x_vol + sol_y_vol -- Total volume of the final solution

-- Define the problem statement
theorem find_sol_y_pct (sol_x_vol sol_y_vol final_sol_vol : ℕ) 
  (sol_x_pct final_sol_pct : ℕ) : 
  (600 * 10 + sol_y_vol * 30) / 800 = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_sol_y_pct_l2210_221039


namespace NUMINAMATH_GPT_range_of_a_l2210_221097

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≥ 4 ∧ y ≥ 4 ∧ x ≤ y → (x^2 + 2*(a-1)*x + 2) ≤ (y^2 + 2*(a-1)*y + 2)) ↔ a ∈ Set.Ici (-3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2210_221097


namespace NUMINAMATH_GPT_equation_solution_l2210_221099

theorem equation_solution (x : ℝ) : 
  (x - 3)^4 = 16 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l2210_221099


namespace NUMINAMATH_GPT_shaded_area_percentage_l2210_221047

-- Define the given conditions
def square_area := 6 * 6
def shaded_area_left := (1 / 2) * 2 * 6
def shaded_area_right := (1 / 2) * 4 * 6
def total_shaded_area := shaded_area_left + shaded_area_right

-- State the theorem
theorem shaded_area_percentage : (total_shaded_area / square_area) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_shaded_area_percentage_l2210_221047


namespace NUMINAMATH_GPT_positive_difference_of_R_coords_l2210_221041

theorem positive_difference_of_R_coords :
    ∀ (xR yR : ℝ),
    ∃ (k : ℝ),
    (∀ (A B C R S : ℝ × ℝ), 
    A = (-1, 6) ∧ B = (1, 2) ∧ C = (7, 2) ∧ 
    R = (k, -0.5 * k + 5.5) ∧ S = (k, 2) ∧
    (0.5 * |7 - k| * |0.5 * k - 3.5| = 8)) → 
    |xR - yR| = 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_R_coords_l2210_221041


namespace NUMINAMATH_GPT_allie_betty_total_points_product_l2210_221073

def score (n : Nat) : Nat :=
  if n % 3 == 0 then 9
  else if n % 2 == 0 then 3
  else if n % 2 == 1 then 1
  else 0

def allie_points : List Nat := [5, 2, 6, 1, 3]
def betty_points : List Nat := [6, 4, 1, 2, 5]

def total_points (rolls: List Nat) : Nat :=
  rolls.foldl (λ acc n => acc + score n) 0

theorem allie_betty_total_points_product : 
  total_points allie_points * total_points betty_points = 391 := by
  sorry

end NUMINAMATH_GPT_allie_betty_total_points_product_l2210_221073


namespace NUMINAMATH_GPT_find_k_l2210_221092

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem find_k (k : ℝ) (h_pos : 0 < k) (h_exists : ∃ x₀ : ℝ, 1 ≤ x₀ ∧ g x₀ ≤ k * (-x₀^2 + 3 * x₀)) : 
  k > (1 / 2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end NUMINAMATH_GPT_find_k_l2210_221092


namespace NUMINAMATH_GPT_cookie_sales_l2210_221032

theorem cookie_sales (n M A : ℕ) 
  (hM : M = n - 9)
  (hA : A = n - 2)
  (h_sum : M + A < n)
  (hM_positive : M ≥ 1)
  (hA_positive : A ≥ 1) : 
  n = 10 := 
sorry

end NUMINAMATH_GPT_cookie_sales_l2210_221032


namespace NUMINAMATH_GPT_fraction_less_than_40_percent_l2210_221001

theorem fraction_less_than_40_percent (x : ℝ) (h1 : x * 180 = 48) (h2 : x < 0.4) : x = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_less_than_40_percent_l2210_221001


namespace NUMINAMATH_GPT_deformable_to_triangle_l2210_221056

-- Definition of the planar polygon with n sides
structure Polygon (n : ℕ) := 
  (vertices : Fin n → ℝ × ℝ) -- This is a simplified representation of a planar polygon using vertex coordinates

noncomputable def canDeformToTriangle (poly : Polygon n) : Prop := sorry

theorem deformable_to_triangle (n : ℕ) (h : n > 4) (poly : Polygon n) : canDeformToTriangle poly := 
  sorry

end NUMINAMATH_GPT_deformable_to_triangle_l2210_221056


namespace NUMINAMATH_GPT_alexis_initial_budget_l2210_221050

-- Define all the given conditions
def cost_shirt : Int := 30
def cost_pants : Int := 46
def cost_coat : Int := 38
def cost_socks : Int := 11
def cost_belt : Int := 18
def cost_shoes : Int := 41
def amount_left : Int := 16

-- Define the total expenses
def total_expenses : Int := cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + cost_shoes

-- Define the initial budget
def initial_budget : Int := total_expenses + amount_left

-- The proof statement
theorem alexis_initial_budget : initial_budget = 200 := by
  sorry

end NUMINAMATH_GPT_alexis_initial_budget_l2210_221050


namespace NUMINAMATH_GPT_doubled_base_and_exponent_l2210_221084

theorem doubled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ 3) : 
  x = (4 ^ b * a ^ b) ^ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_doubled_base_and_exponent_l2210_221084


namespace NUMINAMATH_GPT_sum_fractions_l2210_221083

theorem sum_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end NUMINAMATH_GPT_sum_fractions_l2210_221083


namespace NUMINAMATH_GPT_brookdale_avg_temp_l2210_221015

def highs : List ℤ := [51, 64, 60, 59, 48, 55]
def lows : List ℤ := [42, 49, 47, 43, 41, 44]

def average_temperature : ℚ :=
  let total_sum := highs.sum + lows.sum
  let count := (highs.length + lows.length : ℚ)
  total_sum / count

theorem brookdale_avg_temp :
  average_temperature = 49.4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_brookdale_avg_temp_l2210_221015


namespace NUMINAMATH_GPT_find_length_of_bridge_l2210_221011

noncomputable def length_of_train : ℝ := 165
noncomputable def speed_of_train_kmph : ℝ := 54
noncomputable def time_to_cross_bridge_seconds : ℝ := 67.66125376636536

noncomputable def speed_of_train_mps : ℝ :=
  speed_of_train_kmph * (1000 / 3600)

noncomputable def total_distance_covered : ℝ :=
  speed_of_train_mps * time_to_cross_bridge_seconds

noncomputable def length_of_bridge : ℝ :=
  total_distance_covered - length_of_train

theorem find_length_of_bridge : length_of_bridge = 849.92 := by
  sorry

end NUMINAMATH_GPT_find_length_of_bridge_l2210_221011


namespace NUMINAMATH_GPT_jake_final_bitcoins_l2210_221087

def initial_bitcoins : ℕ := 120
def investment_bitcoins : ℕ := 40
def returned_investment : ℕ := investment_bitcoins * 2
def bitcoins_after_investment : ℕ := initial_bitcoins - investment_bitcoins + returned_investment
def first_charity_donation : ℕ := 25
def bitcoins_after_first_donation : ℕ := bitcoins_after_investment - first_charity_donation
def brother_share : ℕ := 67
def bitcoins_after_giving_to_brother : ℕ := bitcoins_after_first_donation - brother_share
def debt_payment : ℕ := 5
def bitcoins_after_taking_back : ℕ := bitcoins_after_giving_to_brother + debt_payment
def quadrupled_bitcoins : ℕ := bitcoins_after_taking_back * 4
def second_charity_donation : ℕ := 15
def final_bitcoins : ℕ := quadrupled_bitcoins - second_charity_donation

theorem jake_final_bitcoins : final_bitcoins = 277 := by
  unfold final_bitcoins
  unfold quadrupled_bitcoins
  unfold bitcoins_after_taking_back
  unfold debt_payment
  unfold bitcoins_after_giving_to_brother
  unfold brother_share
  unfold bitcoins_after_first_donation
  unfold first_charity_donation
  unfold bitcoins_after_investment
  unfold returned_investment
  unfold investment_bitcoins
  unfold initial_bitcoins
  sorry

end NUMINAMATH_GPT_jake_final_bitcoins_l2210_221087


namespace NUMINAMATH_GPT_boys_and_girls_solution_l2210_221090

theorem boys_and_girls_solution (x y : ℕ) 
  (h1 : 3 * x + y > 24) 
  (h2 : 7 * x + 3 * y < 60) : x = 8 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_boys_and_girls_solution_l2210_221090


namespace NUMINAMATH_GPT_rectangular_prism_has_8_vertices_l2210_221075

def rectangular_prism_vertices := 8

theorem rectangular_prism_has_8_vertices : rectangular_prism_vertices = 8 := by
  sorry

end NUMINAMATH_GPT_rectangular_prism_has_8_vertices_l2210_221075


namespace NUMINAMATH_GPT_no_five_consecutive_divisible_by_2005_l2210_221054

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2005 :
  ¬ (∃ m : ℕ, ∀ k : ℕ, k < 5 → (seq (m + k)) % 2005 = 0) :=
sorry

end NUMINAMATH_GPT_no_five_consecutive_divisible_by_2005_l2210_221054


namespace NUMINAMATH_GPT_modulus_of_complex_l2210_221091

open Complex

theorem modulus_of_complex (z : ℂ) (h : (1 + z) / (1 - z) = ⟨0, 1⟩) : abs z = 1 := 
sorry

end NUMINAMATH_GPT_modulus_of_complex_l2210_221091


namespace NUMINAMATH_GPT_portion_of_larger_jar_full_l2210_221013

noncomputable def smaller_jar_capacity (S L : ℝ) : Prop :=
  (1 / 5) * S = (1 / 4) * L

noncomputable def larger_jar_capacity (L : ℝ) : ℝ :=
  (1 / 5) * (5 / 4) * L

theorem portion_of_larger_jar_full (S L : ℝ) 
  (h1 : smaller_jar_capacity S L) : 
  (1 / 4) * L + (1 / 4) * L = (1 / 2) * L := 
sorry

end NUMINAMATH_GPT_portion_of_larger_jar_full_l2210_221013


namespace NUMINAMATH_GPT_prime_product_div_by_four_l2210_221018

theorem prime_product_div_by_four 
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq1 : Nat.Prime (p * q + 1)) : 
  4 ∣ (2 * p + q) * (p + 2 * q) := 
sorry

end NUMINAMATH_GPT_prime_product_div_by_four_l2210_221018


namespace NUMINAMATH_GPT_cheaperCandy_cost_is_5_l2210_221002

def cheaperCandy (C : ℝ) : Prop :=
  let expensiveCandyCost := 20 * 8
  let cheaperCandyCost := 40 * C
  let totalWeight := 20 + 40
  let totalCost := 60 * 6
  expensiveCandyCost + cheaperCandyCost = totalCost

theorem cheaperCandy_cost_is_5 : cheaperCandy 5 :=
by
  unfold cheaperCandy
  -- SORRY is a placeholder for the proof steps, which are not required
  sorry 

end NUMINAMATH_GPT_cheaperCandy_cost_is_5_l2210_221002


namespace NUMINAMATH_GPT_sum_of_reciprocals_factors_12_l2210_221053

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end NUMINAMATH_GPT_sum_of_reciprocals_factors_12_l2210_221053


namespace NUMINAMATH_GPT_remaining_money_l2210_221080

theorem remaining_money (m : ℝ) (c f t r : ℝ)
  (h_initial : m = 1500)
  (h_clothes : c = (1 / 3) * m)
  (h_food : f = (1 / 5) * (m - c))
  (h_travel : t = (1 / 4) * (m - c - f))
  (h_remaining : r = m - c - f - t) :
  r = 600 := 
by
  sorry

end NUMINAMATH_GPT_remaining_money_l2210_221080


namespace NUMINAMATH_GPT_increase_factor_l2210_221060

-- Definition of parameters: number of letters, digits, and symbols.
def num_letters : ℕ := 26
def num_digits : ℕ := 10
def num_symbols : ℕ := 5

-- Definition of the number of old license plates and new license plates.
def num_old_plates : ℕ := num_letters ^ 2 * num_digits ^ 3
def num_new_plates : ℕ := num_letters ^ 3 * num_digits ^ 3 * num_symbols

-- The proof problem statement: Prove that the increase factor is 130.
theorem increase_factor : num_new_plates / num_old_plates = 130 := by
  sorry

end NUMINAMATH_GPT_increase_factor_l2210_221060


namespace NUMINAMATH_GPT_max_value_m_l2210_221046

theorem max_value_m {m : ℝ} (h : ∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → m ≤ Real.tan x + 1) : m = 2 :=
sorry

end NUMINAMATH_GPT_max_value_m_l2210_221046


namespace NUMINAMATH_GPT_xiao_zhang_winning_probability_max_expected_value_l2210_221034

-- Definitions for the conditions
variables (a b c : ℕ)
variable (h_sum : a + b + c = 6)

-- Main theorem statement 1: Probability of Xiao Zhang winning
theorem xiao_zhang_winning_probability (h_sum : a + b + c = 6) :
  (3 * a + 2 * b + c) / 36 = a / 6 * 3 / 6 + b / 6 * 2 / 6 + c / 6 * 1 / 6 :=
sorry

-- Main theorem statement 2: Maximum expected value of Xiao Zhang's score
theorem max_expected_value (h_sum : a + b + c = 6) :
  (3 * a + 4 * b + 3 * c) / 36 = (1 / 2 + b / 36) →  (a = 0 ∧ b = 6 ∧ c = 0) :=
sorry

end NUMINAMATH_GPT_xiao_zhang_winning_probability_max_expected_value_l2210_221034


namespace NUMINAMATH_GPT_sum_six_times_product_l2210_221014

variable (a b x : ℝ)

theorem sum_six_times_product (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * x) (h4 : 1/a + 1/b = 6) :
  x = a * b := sorry

end NUMINAMATH_GPT_sum_six_times_product_l2210_221014


namespace NUMINAMATH_GPT_julia_drove_214_miles_l2210_221093

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end NUMINAMATH_GPT_julia_drove_214_miles_l2210_221093


namespace NUMINAMATH_GPT_sum_slopes_const_zero_l2210_221017

-- Define variables and constants
variable (p : ℝ) (h : 0 < p)

-- Define parabola and circle equations
def parabola_C1 (x y : ℝ) : Prop := y^2 = 2 * p * x
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 = p^2

-- Condition: The line segment length from circle cut by directrix
def segment_length_condition : Prop := ∃ d : ℝ, d^2 + 3 = p^2

-- The main theorem to prove
theorem sum_slopes_const_zero
  (A : ℝ × ℝ)
  (F : ℝ × ℝ := (p / 2, 0))
  (M N : ℝ × ℝ)
  (line_n_through_A : ∀ x : ℝ, x = 1 / p - 1 + 1 / p → (1 / p - 1 + x) = 0)
  (intersection_prop: parabola_C1 p M.1 M.2 ∧ parabola_C1 p N.1 N.2) 
  (slope_MF : ℝ := (M.2 / (p / 2 - M.1)) ) 
  (slope_NF : ℝ := (N.2 / (p / 2 - N.1))) :
  slope_MF + slope_NF = 0 := 
sorry

end NUMINAMATH_GPT_sum_slopes_const_zero_l2210_221017


namespace NUMINAMATH_GPT_solution_set_a1_range_of_a_l2210_221058

def f (x a : ℝ) : ℝ := abs (x - a) * abs (x + abs (x - 2)) * abs (x - a)

theorem solution_set_a1 (x : ℝ) : f x 1 < 0 ↔ x < 1 :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, x < 1 → f x a < 0) ↔ 1 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_solution_set_a1_range_of_a_l2210_221058


namespace NUMINAMATH_GPT_domain_of_p_l2210_221048

def is_domain_of_p (x : ℝ) : Prop := x > 5

theorem domain_of_p :
  {x : ℝ | ∃ y : ℝ, y = 5*x + 2 ∧ ∃ z : ℝ, z = 2*x - 10 ∧
    z ≥ 0 ∧ z ≠ 0 ∧ p = 5*x + 2} = {x : ℝ | x > 5} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_p_l2210_221048


namespace NUMINAMATH_GPT_smallest_positive_integer_l2210_221085

def smallest_x (x : ℕ) : Prop :=
  (540 * x) % 800 = 0

theorem smallest_positive_integer (x : ℕ) : smallest_x x → x = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_integer_l2210_221085


namespace NUMINAMATH_GPT_sequence_formula_l2210_221007

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l2210_221007


namespace NUMINAMATH_GPT_find_m_n_l2210_221081

theorem find_m_n 
  (a b c d m n : ℕ) 
  (h₁ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₂ : a + b + c + d = m^2)
  (h₃ : a = max (max a b) (max c d) ∨ b = max (max a b) (max c d) ∨ c = max (max a b) (max c d) ∨ d = max (max a b) (max c d))
  (h₄ : exists k, k^2 = max (max a b) (max c d))
  : m = 9 ∧ n = 6 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_m_n_l2210_221081


namespace NUMINAMATH_GPT_circular_seating_count_l2210_221044

theorem circular_seating_count :
  let D := 5 -- Number of Democrats
  let R := 5 -- Number of Republicans
  let total_politicians := D + R -- Total number of politicians
  let linear_arrangements := Nat.factorial total_politicians -- Total linear arrangements
  let unique_circular_arrangements := linear_arrangements / total_politicians -- Adjusting for circular rotations
  unique_circular_arrangements = 362880 :=
by
  sorry

end NUMINAMATH_GPT_circular_seating_count_l2210_221044


namespace NUMINAMATH_GPT_number_of_discounted_tickets_l2210_221003

def total_tickets : ℕ := 10
def full_price_ticket_cost : ℝ := 2.0
def discounted_ticket_cost : ℝ := 1.6
def total_spent : ℝ := 18.40

theorem number_of_discounted_tickets (F D : ℕ) : 
    F + D = total_tickets → 
    full_price_ticket_cost * ↑F + discounted_ticket_cost * ↑D = total_spent → 
    D = 4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_of_discounted_tickets_l2210_221003


namespace NUMINAMATH_GPT_greatest_k_dividing_abcdef_l2210_221021

theorem greatest_k_dividing_abcdef {a b c d e f : ℤ}
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = f^2) :
  ∃ k, (∀ a b c d e f, a^2 + b^2 + c^2 + d^2 + e^2 = f^2 → k ∣ (a * b * c * d * e * f)) ∧ k = 24 :=
sorry

end NUMINAMATH_GPT_greatest_k_dividing_abcdef_l2210_221021


namespace NUMINAMATH_GPT_exists_difference_divisible_by_11_l2210_221086

theorem exists_difference_divisible_by_11 (a : Fin 12 → ℤ) :
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
  sorry

end NUMINAMATH_GPT_exists_difference_divisible_by_11_l2210_221086


namespace NUMINAMATH_GPT_inverse_proposition_l2210_221040

theorem inverse_proposition (a b c : ℝ) : (a > b → a + c > b + c) → (a + c > b + c → a > b) :=
sorry

end NUMINAMATH_GPT_inverse_proposition_l2210_221040


namespace NUMINAMATH_GPT_find_number_l2210_221008

theorem find_number (x : ℝ) (h : x^2 + 50 = (x - 10)^2) : x = 2.5 :=
sorry

end NUMINAMATH_GPT_find_number_l2210_221008


namespace NUMINAMATH_GPT_gcd_154_and_90_l2210_221088

theorem gcd_154_and_90 : Nat.gcd 154 90 = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_154_and_90_l2210_221088


namespace NUMINAMATH_GPT_triangle_side_calculation_l2210_221066

theorem triangle_side_calculation
  (a : ℝ) (A B : ℝ)
  (ha : a = 3)
  (hA : A = 30)
  (hB : B = 15) :
  let C := 180 - A - B
  let c := a * (Real.sin C) / (Real.sin A)
  c = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_triangle_side_calculation_l2210_221066


namespace NUMINAMATH_GPT_man_older_than_son_l2210_221079

theorem man_older_than_son (S M : ℕ) (hS : S = 27) (hM : M + 2 = 2 * (S + 2)) : M - S = 29 := 
by {
  sorry
}

end NUMINAMATH_GPT_man_older_than_son_l2210_221079


namespace NUMINAMATH_GPT_sum_of_perimeters_geq_4400_l2210_221065

theorem sum_of_perimeters_geq_4400 (side original_side : ℕ) 
  (h_side_le_10 : ∀ s, s ≤ side → s ≤ 10) 
  (h_original_square : original_side = 100) 
  (h_cut_condition : side ≤ 10) : 
  ∃ (small_squares : ℕ → ℕ × ℕ), (original_side / side = n) → 4 * n * side ≥ 4400 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_geq_4400_l2210_221065


namespace NUMINAMATH_GPT_expression_evaluation_l2210_221026

theorem expression_evaluation :
  (0.86^3) - ((0.1^3) / (0.86^2)) + 0.086 + (0.1^2) = 0.730704 := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2210_221026


namespace NUMINAMATH_GPT_Papi_Calot_plants_l2210_221049

theorem Papi_Calot_plants :
  let initial_potatoes_plants := 10 * 25
  let initial_carrots_plants := 15 * 30
  let initial_onions_plants := 12 * 20
  let total_potato_plants := initial_potatoes_plants + 20
  let total_carrot_plants := initial_carrots_plants + 30
  let total_onion_plants := initial_onions_plants + 10
  total_potato_plants = 270 ∧
  total_carrot_plants = 480 ∧
  total_onion_plants = 250 := by
  sorry

end NUMINAMATH_GPT_Papi_Calot_plants_l2210_221049


namespace NUMINAMATH_GPT_second_odd_integer_l2210_221025

theorem second_odd_integer (n : ℤ) (h : (n - 2) + (n + 2) = 128) : n = 64 :=
by
  sorry

end NUMINAMATH_GPT_second_odd_integer_l2210_221025


namespace NUMINAMATH_GPT_instrument_price_problem_l2210_221069

theorem instrument_price_problem (v t p : ℝ) (h1 : 1.5 * v = 0.5 * t + 50) (h2 : 1.5 * t = 0.5 * p + 50) : 
  ∃ m n : ℤ, m = 80 ∧ n = 80 ∧ (100 + m) * v / 100 = n + (100 - m) * p / 100 := 
by
  use 80, 80
  sorry

end NUMINAMATH_GPT_instrument_price_problem_l2210_221069


namespace NUMINAMATH_GPT_proof_l2210_221038

noncomputable def problem_statement (a b : ℝ) :=
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1)

theorem proof : ∀ a b : ℝ, problem_statement a b := sorry

end NUMINAMATH_GPT_proof_l2210_221038


namespace NUMINAMATH_GPT_vertices_of_cube_l2210_221005

-- Given condition: geometric shape is a cube
def is_cube (x : Type) : Prop := true -- This is a placeholder declaration that x is a cube.

-- Question: How many vertices does a cube have?
-- Proof problem: Prove that the number of vertices of a cube is 8.
theorem vertices_of_cube (x : Type) (h : is_cube x) : true := 
  sorry

end NUMINAMATH_GPT_vertices_of_cube_l2210_221005


namespace NUMINAMATH_GPT_remove_least_candies_l2210_221037

theorem remove_least_candies (total_candies : ℕ) (friends : ℕ) (candies_remaining : ℕ) : total_candies = 34 ∧ friends = 5 ∧ candies_remaining = 4 → (total_candies % friends = candies_remaining) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_remove_least_candies_l2210_221037


namespace NUMINAMATH_GPT_range_of_m_l2210_221012

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 3) (h4 : ∀ x y, x > 0 → y > 0 → x + y = 3 → (4 / (x + 1) + 16 / y > m^2 - 3 * m + 11)) : 1 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2210_221012


namespace NUMINAMATH_GPT_price_per_jin_of_tomatoes_is_3yuan_3jiao_l2210_221076

/-- Definitions of the conditions --/
def cucumbers_cost_jin : ℕ := 5
def cucumbers_cost_yuan : ℕ := 11
def cucumbers_cost_jiao : ℕ := 8
def tomatoes_cost_jin : ℕ := 4
def difference_cost_yuan : ℕ := 1
def difference_cost_jiao : ℕ := 4

/-- Converting cost in yuan and jiao to decimal yuan --/
def cost_in_yuan (yuan jiao : ℕ) : ℕ := yuan + jiao / 10

/-- Given conditions in decimal --/
def cucumbers_cost := cost_in_yuan cucumbers_cost_yuan cucumbers_cost_jiao
def difference_cost := cost_in_yuan difference_cost_yuan difference_cost_jiao
def tomatoes_cost := cucumbers_cost + difference_cost

/-- Proof statement: price per jin of tomatoes in yuan and jiao --/
theorem price_per_jin_of_tomatoes_is_3yuan_3jiao :
  tomatoes_cost / tomatoes_cost_jin = 3 + 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_price_per_jin_of_tomatoes_is_3yuan_3jiao_l2210_221076


namespace NUMINAMATH_GPT_center_of_symmetry_l2210_221070

def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

/--
  Given the function f(x) := sin x - sqrt(3) * cos x,
  prove that (π/3, 0) is the center of symmetry for f.
-/
theorem center_of_symmetry : symmetry_center (fun x => Real.sin x - Real.sqrt 3 * Real.cos x) (Real.pi / 3, 0) :=
by
  sorry

end NUMINAMATH_GPT_center_of_symmetry_l2210_221070


namespace NUMINAMATH_GPT_floor_plus_x_eq_205_l2210_221024

theorem floor_plus_x_eq_205 (x : ℝ) (h : ⌊x⌋ + x = 20.5) : x = 10.5 :=
sorry

end NUMINAMATH_GPT_floor_plus_x_eq_205_l2210_221024


namespace NUMINAMATH_GPT_steve_bought_3_boxes_of_cookies_l2210_221078

variable (total_cost : ℝ)
variable (milk_cost : ℝ)
variable (cereal_cost : ℝ)
variable (banana_cost : ℝ)
variable (apple_cost : ℝ)
variable (chicken_cost : ℝ)
variable (peanut_butter_cost : ℝ)
variable (bread_cost : ℝ)
variable (cookie_box_cost : ℝ)
variable (cookie_box_count : ℝ)

noncomputable def proves_steve_cookie_boxes : Prop :=
  total_cost = 50 ∧
  milk_cost = 4 ∧
  cereal_cost = 3 ∧
  banana_cost = 0.2 ∧
  apple_cost = 0.75 ∧
  chicken_cost = 10 ∧
  peanut_butter_cost = 5 ∧
  bread_cost = (2 * cereal_cost) / 2 ∧
  cookie_box_cost = (milk_cost + peanut_butter_cost) / 3 ∧
  cookie_box_count = (total_cost - (milk_cost + 3 * cereal_cost + 6 * banana_cost + 8 * apple_cost + chicken_cost + peanut_butter_cost + bread_cost)) / cookie_box_cost

theorem steve_bought_3_boxes_of_cookies :
  proves_steve_cookie_boxes 50 4 3 0.2 0.75 10 5 3 ((4 + 5) / 3) 3 :=
by
  sorry

end NUMINAMATH_GPT_steve_bought_3_boxes_of_cookies_l2210_221078


namespace NUMINAMATH_GPT_cone_radius_from_melted_cylinder_l2210_221052

theorem cone_radius_from_melted_cylinder :
  ∀ (r_cylinder h_cylinder r_cone h_cone : ℝ),
  r_cylinder = 8 ∧ h_cylinder = 2 ∧ h_cone = 6 ∧
  (π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone) →
  r_cone = 8 :=
by
  sorry

end NUMINAMATH_GPT_cone_radius_from_melted_cylinder_l2210_221052


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l2210_221031

theorem simplify_expr1 (y : ℤ) (hy : y = 2) : -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := 
by sorry

theorem simplify_expr2 (a : ℤ) (ha : a = -2) : 15 * a^2 * (-4 * a^2 + (6 * a - a^2) - 3 * a) = -1560 :=
by sorry

theorem simplify_expr3 (x y : ℤ) (h1 : x * y = 2) (h2 : x + y = 3) : (3 * x * y + 10 * y) + (5 * x - (2 * x * y + 2 * y - 3 * x)) = 26 :=
by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l2210_221031


namespace NUMINAMATH_GPT_birds_left_in_tree_l2210_221043

-- Define the initial number of birds in the tree
def initialBirds : ℝ := 42.5

-- Define the number of birds that flew away
def birdsFlewAway : ℝ := 27.3

-- Theorem statement: Prove the number of birds left in the tree
theorem birds_left_in_tree : initialBirds - birdsFlewAway = 15.2 :=
by 
  sorry

end NUMINAMATH_GPT_birds_left_in_tree_l2210_221043


namespace NUMINAMATH_GPT_ninth_term_arith_seq_l2210_221067

-- Define the arithmetic sequence.
def arith_seq (a₁ d : ℚ) (n : ℕ) := a₁ + n * d

-- Define the third and fifteenth terms of the sequence.
def third_term := (5 : ℚ) / 11
def fifteenth_term := (7 : ℚ) / 8

-- Prove that the ninth term is 117/176 given the conditions.
theorem ninth_term_arith_seq :
    ∃ (a₁ d : ℚ), 
    arith_seq a₁ d 2 = third_term ∧ 
    arith_seq a₁ d 14 = fifteenth_term ∧
    arith_seq a₁ d 8 = 117 / 176 :=
by
  sorry

end NUMINAMATH_GPT_ninth_term_arith_seq_l2210_221067


namespace NUMINAMATH_GPT_voting_for_marty_l2210_221071

/-- Conditions provided in the problem -/
def total_people : ℕ := 400
def percentage_biff : ℝ := 0.30
def percentage_clara : ℝ := 0.20
def percentage_doc : ℝ := 0.10
def percentage_ein : ℝ := 0.05
def percentage_undecided : ℝ := 0.15

/-- Statement to prove the number of people voting for Marty -/
theorem voting_for_marty : 
  (1 - percentage_biff - percentage_clara - percentage_doc - percentage_ein - percentage_undecided) * total_people = 80 :=
by
  sorry

end NUMINAMATH_GPT_voting_for_marty_l2210_221071


namespace NUMINAMATH_GPT_Daisy_lunch_vs_breakfast_l2210_221028

noncomputable def breakfast_cost : ℝ := 2.0 + 3.0 + 4.0 + 3.5
noncomputable def lunch_cost_before_service_charge : ℝ := 3.75 + 5.75 + 1.0
noncomputable def service_charge : ℝ := 0.10 * lunch_cost_before_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_before_service_charge + service_charge

theorem Daisy_lunch_vs_breakfast : total_lunch_cost - breakfast_cost = -0.95 := by
  sorry

end NUMINAMATH_GPT_Daisy_lunch_vs_breakfast_l2210_221028


namespace NUMINAMATH_GPT_common_terms_only_1_and_7_l2210_221045

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * sequence_a (n - 1) - sequence_a (n - 2)

def sequence_b (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 7
  else 6 * sequence_b (n - 1) - sequence_b (n - 2)

theorem common_terms_only_1_and_7 :
  ∀ n m : ℕ, (sequence_a n = sequence_b m) → (sequence_a n = 1 ∨ sequence_a n = 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_common_terms_only_1_and_7_l2210_221045


namespace NUMINAMATH_GPT_reciprocal_of_repeating_decimal_l2210_221033

theorem reciprocal_of_repeating_decimal :
  let x := 0.36363636 -- simplified as .\overline{36}
  ∃ y : ℚ, x = 4 / 11 ∧ y = 1 / x ∧ y = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_repeating_decimal_l2210_221033


namespace NUMINAMATH_GPT_selling_price_of_cycle_l2210_221004

def cost_price : ℝ := 1400
def loss_percentage : ℝ := 18

theorem selling_price_of_cycle : 
    (cost_price - (loss_percentage / 100) * cost_price) = 1148 := 
by
  sorry

end NUMINAMATH_GPT_selling_price_of_cycle_l2210_221004


namespace NUMINAMATH_GPT_problem_statement_l2210_221010

noncomputable def verify_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : Prop :=
  c / d = -1/3

theorem problem_statement (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : verify_ratio x y c d h1 h2 h3 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l2210_221010


namespace NUMINAMATH_GPT_largest_y_coordinate_of_graph_l2210_221062

theorem largest_y_coordinate_of_graph :
  ∀ (x y : ℝ), (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_of_graph_l2210_221062


namespace NUMINAMATH_GPT_profit_percentage_l2210_221095

theorem profit_percentage (SP CP : ℝ) (h_SP : SP = 150) (h_CP : CP = 120) : 
  ((SP - CP) / CP) * 100 = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_profit_percentage_l2210_221095


namespace NUMINAMATH_GPT_divides_necklaces_l2210_221094

/-- Define the number of ways to make an even number of necklaces each of length at least 3. -/
def D_0 (n : ℕ) : ℕ := sorry

/-- Define the number of ways to make an odd number of necklaces each of length at least 3. -/
def D_1 (n : ℕ) : ℕ := sorry

/-- Main theorem: Prove that (n - 1) divides (D_1(n) - D_0(n)) for n ≥ 2 -/
theorem divides_necklaces (n : ℕ) (h : n ≥ 2) : (n - 1) ∣ (D_1 n - D_0 n) := sorry

end NUMINAMATH_GPT_divides_necklaces_l2210_221094


namespace NUMINAMATH_GPT_total_pages_read_l2210_221009

-- Define the reading rates
def ReneReadingRate : ℕ := 30  -- pages in 60 minutes
def LuluReadingRate : ℕ := 27  -- pages in 60 minutes
def CherryReadingRate : ℕ := 25  -- pages in 60 minutes

-- Total time in minutes
def totalTime : ℕ := 240  -- minutes

-- Define a function to calculate pages read in given time
def pagesRead (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem to prove the total number of pages read
theorem total_pages_read :
  pagesRead ReneReadingRate totalTime +
  pagesRead LuluReadingRate totalTime +
  pagesRead CherryReadingRate totalTime = 328 :=
by
  -- Proof is not required, hence replaced with sorry
  sorry

end NUMINAMATH_GPT_total_pages_read_l2210_221009


namespace NUMINAMATH_GPT_find_number_l2210_221027

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 := by 
  sorry

end NUMINAMATH_GPT_find_number_l2210_221027


namespace NUMINAMATH_GPT_remainder_approximately_14_l2210_221072

def dividend : ℝ := 14698
def quotient : ℝ := 89
def divisor : ℝ := 164.98876404494382
def remainder : ℝ := dividend - (quotient * divisor)

theorem remainder_approximately_14 : abs (remainder - 14) < 1e-10 := 
by
-- using abs since the problem is numerical/approximate
sorry

end NUMINAMATH_GPT_remainder_approximately_14_l2210_221072


namespace NUMINAMATH_GPT_total_shaded_area_l2210_221051

theorem total_shaded_area (r R : ℝ) (h1 : π * R^2 = 100 * π) (h2 : r = R / 2) : 
    (1/4) * π * R^2 + (1/4) * π * r^2 = 31.25 * π :=
by
  sorry

end NUMINAMATH_GPT_total_shaded_area_l2210_221051


namespace NUMINAMATH_GPT_monotone_intervals_range_of_t_for_three_roots_l2210_221022

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end NUMINAMATH_GPT_monotone_intervals_range_of_t_for_three_roots_l2210_221022


namespace NUMINAMATH_GPT_smallest_hope_number_l2210_221006

def is_square (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k
def is_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k
def is_fifth_power (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k * k * k

def is_hope_number (n : ℕ) : Prop :=
  is_square (n / 8) ∧ is_cube (n / 9) ∧ is_fifth_power (n / 25)

theorem smallest_hope_number : ∃ n, is_hope_number n ∧ n = 2^15 * 3^20 * 5^12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_hope_number_l2210_221006


namespace NUMINAMATH_GPT_anna_apple_ratio_l2210_221019

-- Definitions based on conditions
def tuesday_apples : ℕ := 4
def wednesday_apples : ℕ := 2 * tuesday_apples
def total_apples : ℕ := 14

-- Theorem statement
theorem anna_apple_ratio :
  ∃ thursday_apples : ℕ, 
  thursday_apples = total_apples - (tuesday_apples + wednesday_apples) ∧
  (thursday_apples : ℚ) / tuesday_apples = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_anna_apple_ratio_l2210_221019


namespace NUMINAMATH_GPT_construct_right_triangle_l2210_221057

theorem construct_right_triangle (c m n : ℝ) (hc : c > 0) (hm : m > 0) (hn : n > 0) : 
  ∃ a b : ℝ, a^2 + b^2 = c^2 ∧ a / b = m / n :=
by
  sorry

end NUMINAMATH_GPT_construct_right_triangle_l2210_221057
