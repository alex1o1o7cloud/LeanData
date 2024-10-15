import Mathlib

namespace NUMINAMATH_GPT_avg_ac_l1270_127005

-- Define the ages of a, b, and c as variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def avg_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 26
def age_b (B : ℕ) : Prop := B = 20

-- State the theorem to prove
theorem avg_ac {A B C : ℕ} (h1 : avg_abc A B C) (h2 : age_b B) : (A + C) / 2 = 29 := 
by sorry

end NUMINAMATH_GPT_avg_ac_l1270_127005


namespace NUMINAMATH_GPT_factorize_x4_minus_81_l1270_127039

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x4_minus_81_l1270_127039


namespace NUMINAMATH_GPT_prove_d_value_l1270_127053

-- Definitions of the conditions
def d (x : ℝ) : ℝ := x^4 - 2*x^3 + x^2 - 12*x - 5

-- The statement to prove
theorem prove_d_value (x : ℝ) (h : x^2 - 2*x - 5 = 0) : d x = 25 :=
sorry

end NUMINAMATH_GPT_prove_d_value_l1270_127053


namespace NUMINAMATH_GPT_quarters_remaining_l1270_127012

-- Define the number of quarters Sally originally had
def initialQuarters : Nat := 760

-- Define the number of quarters Sally spent
def spentQuarters : Nat := 418

-- Prove that the number of quarters she has now is 342
theorem quarters_remaining : initialQuarters - spentQuarters = 342 :=
by
  sorry

end NUMINAMATH_GPT_quarters_remaining_l1270_127012


namespace NUMINAMATH_GPT_BC_at_least_17_l1270_127079

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

end NUMINAMATH_GPT_BC_at_least_17_l1270_127079


namespace NUMINAMATH_GPT_solution_set_inequality_l1270_127069

theorem solution_set_inequality (x : ℝ) : |5 - x| < |x - 2| + |7 - 2 * x| ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3.5 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1270_127069


namespace NUMINAMATH_GPT_vertical_line_divides_triangle_l1270_127011

theorem vertical_line_divides_triangle (k : ℝ) :
  let triangle_area := 1 / 2 * |0 * (1 - 1) + 1 * (1 - 0) + 9 * (0 - 1)|
  let left_triangle_area := 1 / 2 * |0 * (1 - 1) + k * (1 - 0) + 1 * (0 - 1)|
  let right_triangle_area := triangle_area - left_triangle_area
  triangle_area = 4 
  ∧ left_triangle_area = 2
  ∧ right_triangle_area = 2
  ∧ (k = 5) ∨ (k = -3) → 
  k = 5 :=
by
  sorry

end NUMINAMATH_GPT_vertical_line_divides_triangle_l1270_127011


namespace NUMINAMATH_GPT_selling_price_correct_l1270_127074

-- Define the conditions
def cost_price : ℝ := 900
def gain_percentage : ℝ := 0.2222222222222222

-- Define the selling price calculation
def profit := cost_price * gain_percentage
def selling_price := cost_price + profit

-- The problem statement in Lean 4
theorem selling_price_correct : selling_price = 1100 := 
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_selling_price_correct_l1270_127074


namespace NUMINAMATH_GPT_sum_of_coefficients_l1270_127029

def original_function (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 4

def transformed_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2 * (x + 2) + 4 + 5

theorem sum_of_coefficients : (3 : ℝ) + 10 + 17 = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1270_127029


namespace NUMINAMATH_GPT_sum_of_five_distinct_integers_product_2022_l1270_127032

theorem sum_of_five_distinct_integers_product_2022 :
  ∃ (a b c d e : ℤ), 
    a * b * c * d * e = 2022 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧ 
    (a + b + c + d + e = 342 ∨
     a + b + c + d + e = 338 ∨
     a + b + c + d + e = 336 ∨
     a + b + c + d + e = -332) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_five_distinct_integers_product_2022_l1270_127032


namespace NUMINAMATH_GPT_worker_savings_fraction_l1270_127060

theorem worker_savings_fraction (P : ℝ) (F : ℝ) (h1 : P > 0) (h2 : 12 * F * P = 5 * (1 - F) * P) : F = 5 / 17 :=
by
  sorry

end NUMINAMATH_GPT_worker_savings_fraction_l1270_127060


namespace NUMINAMATH_GPT_rectangle_dimensions_l1270_127026

-- Definitions from conditions
def is_rectangle (length width : ℝ) : Prop :=
  3 * width = length ∧ 3 * width^2 = 8 * width

-- The theorem to prove
theorem rectangle_dimensions :
  ∃ (length width : ℝ), is_rectangle length width ∧ width = 8 / 3 ∧ length = 8 := by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1270_127026


namespace NUMINAMATH_GPT_percentage_decrease_second_year_l1270_127023

-- Define initial population
def initial_population : ℝ := 14999.999999999998

-- Define the population at the end of the first year after 12% increase
def population_end_year_1 : ℝ := initial_population * 1.12

-- Define the final population at the end of the second year
def final_population : ℝ := 14784.0

-- Define the proof statement
theorem percentage_decrease_second_year :
  ∃ D : ℝ, final_population = population_end_year_1 * (1 - D / 100) ∧ D = 12 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_second_year_l1270_127023


namespace NUMINAMATH_GPT_exists_solution_iff_l1270_127025

theorem exists_solution_iff (m : ℝ) (x y : ℝ) :
  ((y = (3 * m + 2) * x + 1) ∧ (y = (5 * m - 4) * x + 5)) ↔ m ≠ 3 :=
by sorry

end NUMINAMATH_GPT_exists_solution_iff_l1270_127025


namespace NUMINAMATH_GPT_rectangle_with_perpendicular_diagonals_is_square_l1270_127087

-- Define rectangle and its properties
structure Rectangle where
  length : ℝ
  width : ℝ
  opposite_sides_equal : length = width

-- Define the condition that the diagonals of the rectangle are perpendicular
axiom perpendicular_diagonals {r : Rectangle} : r.length = r.width → True

-- Define the square property that a rectangle with all sides equal is a square
structure Square extends Rectangle where
  all_sides_equal : length = width

-- The main theorem to be proven
theorem rectangle_with_perpendicular_diagonals_is_square (r : Rectangle) (h : r.length = r.width) : Square := by
  sorry

end NUMINAMATH_GPT_rectangle_with_perpendicular_diagonals_is_square_l1270_127087


namespace NUMINAMATH_GPT_multiple_of_6_is_multiple_of_3_l1270_127021

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) : (∃ k : ℕ, n = 6 * k) → (∃ m : ℕ, n = 3 * m) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_6_is_multiple_of_3_l1270_127021


namespace NUMINAMATH_GPT_smallest_x_for_multiple_of_450_and_648_l1270_127096

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_of_450_and_648_l1270_127096


namespace NUMINAMATH_GPT_power_function_point_l1270_127015

theorem power_function_point (a : ℝ) (h : (2 : ℝ) ^ a = (1 / 2 : ℝ)) : a = -1 :=
by sorry

end NUMINAMATH_GPT_power_function_point_l1270_127015


namespace NUMINAMATH_GPT_a1_is_1_l1270_127036

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = (2^n - 1)

theorem a1_is_1 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : sequence_sum a S) : 
  a 1 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_a1_is_1_l1270_127036


namespace NUMINAMATH_GPT_number_of_students_l1270_127052

theorem number_of_students (n T : ℕ) (h1 : T = n * 90) 
(h2 : T - 120 = (n - 3) * 95) : n = 33 := 
by
sorry

end NUMINAMATH_GPT_number_of_students_l1270_127052


namespace NUMINAMATH_GPT_intersection_M_N_l1270_127045

def M : Set ℝ := { x | x^2 - x - 2 = 0 }
def N : Set ℝ := { -1, 0 }

theorem intersection_M_N : M ∩ N = {-1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1270_127045


namespace NUMINAMATH_GPT_avg_temp_correct_l1270_127076

-- Defining the temperatures for each day from March 1st to March 5th
def day_1_temp := 55.0
def day_2_temp := 59.0
def day_3_temp := 60.0
def day_4_temp := 57.0
def day_5_temp := 64.0

-- Calculating the average temperature
def avg_temp := (day_1_temp + day_2_temp + day_3_temp + day_4_temp + day_5_temp) / 5.0

-- Proving that the average temperature equals 59.0°F
theorem avg_temp_correct : avg_temp = 59.0 := sorry

end NUMINAMATH_GPT_avg_temp_correct_l1270_127076


namespace NUMINAMATH_GPT_circle_symmetric_about_line_l1270_127030

theorem circle_symmetric_about_line :
  ∃ b : ℝ, (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 4 = 0 → y = 2*x + b) → b = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_symmetric_about_line_l1270_127030


namespace NUMINAMATH_GPT_solve_for_x_l1270_127050

theorem solve_for_x (x : ℝ) (h : (2 / 7) * (1 / 3) * x = 14) : x = 147 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1270_127050


namespace NUMINAMATH_GPT_percentage_of_engineers_from_university_A_l1270_127034

theorem percentage_of_engineers_from_university_A :
  let original_engineers := 20
  let new_hired_engineers := 8
  let percentage_original_from_A := 0.65
  let original_from_A := percentage_original_from_A * original_engineers
  let total_engineers := original_engineers + new_hired_engineers
  let total_from_A := original_from_A + new_hired_engineers
  (total_from_A / total_engineers) * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_engineers_from_university_A_l1270_127034


namespace NUMINAMATH_GPT_inequality_solution_l1270_127009

theorem inequality_solution {x : ℝ} : (1 / 2 - (x - 2) / 3 > 1) → (x < 1 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_l1270_127009


namespace NUMINAMATH_GPT_simplify_expression_l1270_127092

variable (a b : ℚ)

theorem simplify_expression (ha : a = -2) (hb : b = 1/5) :
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  -- Proof can be filled here
  sorry

end NUMINAMATH_GPT_simplify_expression_l1270_127092


namespace NUMINAMATH_GPT_common_ratio_infinite_geometric_series_l1270_127090

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_infinite_geometric_series_l1270_127090


namespace NUMINAMATH_GPT_incenter_sum_equals_one_l1270_127018

noncomputable def incenter (A B C : Point) : Point := sorry -- Definition goes here

def side_length (A B C : Point) (a b c : ℝ) : Prop :=
  -- Definitions relating to side lengths go here
  sorry

theorem incenter_sum_equals_one (A B C I : Point) (a b c IA IB IC : ℝ) (h_incenter : I = incenter A B C)
    (h_sides : side_length A B C a b c) :
    (IA ^ 2 / (b * c)) + (IB ^ 2 / (a * c)) + (IC ^ 2 / (a * b)) = 1 :=
  sorry

end NUMINAMATH_GPT_incenter_sum_equals_one_l1270_127018


namespace NUMINAMATH_GPT_k_times_a_plus_b_l1270_127073

/-- Given a quadrilateral with vertices P(ka, kb), Q(kb, ka), R(-ka, -kb), and S(-kb, -ka),
where a and b are consecutive integers with a > b > 0, and k is an odd integer.
It is given that the area of PQRS is 50.
Prove that k(a + b) = 5. -/
theorem k_times_a_plus_b (a b k : ℤ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a = b + 1)
  (h4 : Odd k)
  (h5 : 2 * k^2 * (a - b) * (a + b) = 50) :
  k * (a + b) = 5 := by
  sorry

end NUMINAMATH_GPT_k_times_a_plus_b_l1270_127073


namespace NUMINAMATH_GPT_central_angle_of_sector_l1270_127098

noncomputable def central_angle (l S r : ℝ) : ℝ :=
  2 * S / r^2

theorem central_angle_of_sector (r : ℝ) (h₁ : 4 * r / 2 = 4) (h₂ : r = 2) : central_angle 4 4 r = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1270_127098


namespace NUMINAMATH_GPT_running_current_each_unit_l1270_127020

theorem running_current_each_unit (I : ℝ) (h1 : ∀i, i = 2 * I) (h2 : ∀i, i * 3 = 6 * I) (h3 : 6 * I = 240) : I = 40 :=
by
  sorry

end NUMINAMATH_GPT_running_current_each_unit_l1270_127020


namespace NUMINAMATH_GPT_algebra_expression_value_l1270_127057

theorem algebra_expression_value (a b : ℝ) 
  (h₁ : a - b = 5) 
  (h₂ : a * b = -1) : 
  (2 * a + 3 * b - 2 * a * b) 
  - (a + 4 * b + a * b) 
  - (3 * a * b + 2 * b - 2 * a) = 21 := 
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l1270_127057


namespace NUMINAMATH_GPT_base_rate_of_first_company_is_7_l1270_127099

noncomputable def telephone_company_base_rate_proof : Prop :=
  ∃ (base_rate1 base_rate2 charge_per_minute1 charge_per_minute2 minutes : ℝ),
  base_rate1 = 7 ∧
  charge_per_minute1 = 0.25 ∧
  base_rate2 = 12 ∧
  charge_per_minute2 = 0.20 ∧
  minutes = 100 ∧
  (base_rate1 + charge_per_minute1 * minutes) =
  (base_rate2 + charge_per_minute2 * minutes) ∧
  base_rate1 = 7

theorem base_rate_of_first_company_is_7 :
  telephone_company_base_rate_proof :=
by
  -- The proof step will go here
  sorry

end NUMINAMATH_GPT_base_rate_of_first_company_is_7_l1270_127099


namespace NUMINAMATH_GPT_maximize_profit_l1270_127067

theorem maximize_profit : 
  ∃ (a b : ℕ), 
  a ≤ 8 ∧ 
  b ≤ 7 ∧ 
  2 * a + b ≤ 19 ∧ 
  a + b ≤ 12 ∧ 
  10 * a + 6 * b ≥ 72 ∧ 
  (a * 450 + b * 350) = 4900 :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l1270_127067


namespace NUMINAMATH_GPT_no_valid_x_l1270_127049

-- Definitions based on given conditions
variables {m n x : ℝ}
variables (hm : m > 0) (hn : n < 0)

-- Theorem statement
theorem no_valid_x (hm : m > 0) (hn : n < 0) :
  ¬ ∃ x, (x - m)^2 - (x - n)^2 = (m - n)^2 :=
by
  sorry

end NUMINAMATH_GPT_no_valid_x_l1270_127049


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l1270_127024

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l1270_127024


namespace NUMINAMATH_GPT_exists_special_N_l1270_127091

open Nat

theorem exists_special_N :
  ∃ N : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 150 → N % i = 0 ∨ i = 127 ∨ i = 128) ∧ 
  ¬ (N % 127 = 0) ∧ ¬ (N % 128 = 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_special_N_l1270_127091


namespace NUMINAMATH_GPT_new_releases_fraction_is_2_over_5_l1270_127031

def fraction_new_releases (total_books : ℕ) (frac_historical_fiction : ℚ) (frac_new_historical_fiction : ℚ) (frac_new_non_historical_fiction : ℚ) : ℚ :=
  let num_historical_fiction := frac_historical_fiction * total_books
  let num_new_historical_fiction := frac_new_historical_fiction * num_historical_fiction
  let num_non_historical_fiction := total_books - num_historical_fiction
  let num_new_non_historical_fiction := frac_new_non_historical_fiction * num_non_historical_fiction
  let total_new_releases := num_new_historical_fiction + num_new_non_historical_fiction
  num_new_historical_fiction / total_new_releases

theorem new_releases_fraction_is_2_over_5 :
  ∀ (total_books : ℕ), total_books > 0 →
    fraction_new_releases total_books (40 / 100) (40 / 100) (40 / 100) = 2 / 5 :=
by 
  intro total_books h
  sorry

end NUMINAMATH_GPT_new_releases_fraction_is_2_over_5_l1270_127031


namespace NUMINAMATH_GPT_remainder_of_x_div_9_l1270_127063

theorem remainder_of_x_div_9 (x : ℕ) (hx_pos : 0 < x) (h : (6 * x) % 9 = 3) : x % 9 = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_x_div_9_l1270_127063


namespace NUMINAMATH_GPT_transformed_expression_value_l1270_127002

-- Defining the new operations according to the problem's conditions
def new_minus (a b : ℕ) : ℕ := a + b
def new_plus (a b : ℕ) : ℕ := a * b
def new_times (a b : ℕ) : ℕ := a / b
def new_div (a b : ℕ) : ℕ := a - b

-- Problem statement
theorem transformed_expression_value : new_minus 6 (new_plus 9 (new_times 8 (new_div 3 25))) = 5 :=
sorry

end NUMINAMATH_GPT_transformed_expression_value_l1270_127002


namespace NUMINAMATH_GPT_range_of_m_l1270_127085

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h1 : 1/x + 1/y = 1) (h2 : x + y > m) : m < 4 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1270_127085


namespace NUMINAMATH_GPT_solve_for_a_l1270_127095

theorem solve_for_a (a x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
by sorry

end NUMINAMATH_GPT_solve_for_a_l1270_127095


namespace NUMINAMATH_GPT_birds_flew_up_l1270_127019

theorem birds_flew_up (initial_birds new_birds total_birds : ℕ) 
    (h_initial : initial_birds = 29) 
    (h_total : total_birds = 42) : 
    new_birds = total_birds - initial_birds := 
by 
    sorry

end NUMINAMATH_GPT_birds_flew_up_l1270_127019


namespace NUMINAMATH_GPT_cone_base_radius_l1270_127043

open Real

theorem cone_base_radius (r_sector : ℝ) (θ_sector : ℝ) : 
    r_sector = 6 ∧ θ_sector = 120 → (∃ r : ℝ, 2 * π * r = θ_sector * π * r_sector / 180 ∧ r = 2) :=
by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l1270_127043


namespace NUMINAMATH_GPT_ratio_of_cream_l1270_127017

theorem ratio_of_cream (coffee_init : ℕ) (joe_coffee_drunk : ℕ) (cream_added : ℕ) (joann_total_drunk : ℕ) 
  (joann_coffee_init : ℕ := coffee_init)
  (joe_coffee_init : ℕ := coffee_init) (joann_cream_init : ℕ := cream_added)
  (joe_cream_init : ℕ := cream_added)
  (joann_drunk_cream_ratio : ℚ := joann_cream_init / (joann_coffee_init + joann_cream_init)) :
  (joe_cream_init / (joann_cream_init - joann_total_drunk * (joann_drunk_cream_ratio))) = 
  (6 / 5) := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_cream_l1270_127017


namespace NUMINAMATH_GPT_probability_jqk_3_13_l1270_127089

def probability_jack_queen_king (total_cards jacks queens kings : ℕ) : ℚ :=
  (jacks + queens + kings) / total_cards

theorem probability_jqk_3_13 :
  probability_jack_queen_king 52 4 4 4 = 3 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_jqk_3_13_l1270_127089


namespace NUMINAMATH_GPT_meadow_area_l1270_127040

theorem meadow_area (x : ℝ) (h1 : ∀ y : ℝ, y = x / 2 + 3) (h2 : ∀ z : ℝ, z = 1 / 3 * (x / 2 - 3) + 6) :
  (x / 2 + 3) + (1 / 3 * (x / 2 - 3) + 6) = x → x = 24 := by
  sorry

end NUMINAMATH_GPT_meadow_area_l1270_127040


namespace NUMINAMATH_GPT_sum_series_equals_4_div_9_l1270_127008

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_sum_series_equals_4_div_9_l1270_127008


namespace NUMINAMATH_GPT_find_P_l1270_127051

noncomputable def parabola_vertex : ℝ × ℝ := (0, 0)
noncomputable def parabola_focus : ℝ × ℝ := (0, -1)
noncomputable def point_P : ℝ × ℝ := (20 * Real.sqrt 6, -120)
noncomputable def PF_distance : ℝ := 121

def parabola_equation (x y : ℝ) : Prop :=
  x^2 = -4 * y

def parabola_condition (x y : ℝ) : Prop :=
  (parabola_equation x y) ∧ 
  (Real.sqrt (x^2 + (y + 1)^2) = PF_distance)

theorem find_P : parabola_condition (point_P.1) (point_P.2) :=
by
  sorry

end NUMINAMATH_GPT_find_P_l1270_127051


namespace NUMINAMATH_GPT_reservoir_water_level_at_6_pm_l1270_127068

/-
  Initial conditions:
  - initial_water_level: Water level at 8 a.m.
  - increase_rate: Rate of increase in water level from 8 a.m. to 12 p.m.
  - decrease_rate: Rate of decrease in water level from 12 p.m. to 6 p.m.
  - start_increase_time: Starting time of increase (in hours from 8 a.m.)
  - end_increase_time: Ending time of increase (in hours from 8 a.m.)
  - start_decrease_time: Starting time of decrease (in hours from 12 p.m.)
  - end_decrease_time: Ending time of decrease (in hours from 12 p.m.)
-/
def initial_water_level : ℝ := 45
def increase_rate : ℝ := 0.6
def decrease_rate : ℝ := 0.3
def start_increase_time : ℝ := 0 -- 8 a.m. in hours from 8 a.m.
def end_increase_time : ℝ := 4 -- 12 p.m. in hours from 8 a.m.
def start_decrease_time : ℝ := 0 -- 12 p.m. in hours from 12 p.m.
def end_decrease_time : ℝ := 6 -- 6 p.m. in hours from 12 p.m.

theorem reservoir_water_level_at_6_pm :
  initial_water_level
  + (end_increase_time - start_increase_time) * increase_rate
  - (end_decrease_time - start_decrease_time) * decrease_rate
  = 45.6 :=
by
  sorry

end NUMINAMATH_GPT_reservoir_water_level_at_6_pm_l1270_127068


namespace NUMINAMATH_GPT_max_grapes_leftover_l1270_127081

-- Define variables and conditions
def total_grapes (n : ℕ) : ℕ := n
def kids : ℕ := 5
def grapes_leftover (n : ℕ) : ℕ := n % kids

-- The proposition we need to prove
theorem max_grapes_leftover (n : ℕ) (h : n ≥ 5) : grapes_leftover n = 4 :=
sorry

end NUMINAMATH_GPT_max_grapes_leftover_l1270_127081


namespace NUMINAMATH_GPT_triangle_sides_possible_k_l1270_127000

noncomputable def f (x k : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_sides_possible_k (a b c k : ℝ) (ha : 0 ≤ a) (hb : a ≤ 3) (ha' : 0 ≤ b) (hb' : b ≤ 3) (ha'' : 0 ≤ c) (hb'' : c ≤ 3) :
  (f a k + f b k > f c k) ∧ (f a k + f c k > f b k) ∧ (f b k + f c k > f a k) ↔ k = 3 ∨ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sides_possible_k_l1270_127000


namespace NUMINAMATH_GPT_max_value_npk_l1270_127093

theorem max_value_npk : 
  ∃ (M K : ℕ), 
    (M ≠ K) ∧ (1 ≤ M ∧ M ≤ 9) ∧ (1 ≤ K ∧ K ≤ 9) ∧ 
    (NPK = 11 * M * K ∧ 100 ≤ NPK ∧ NPK < 1000 ∧ NPK = 891) :=
sorry

end NUMINAMATH_GPT_max_value_npk_l1270_127093


namespace NUMINAMATH_GPT_range_g_l1270_127086

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x + 1
noncomputable def g (a x : ℝ) : ℝ := x^2 + a * x + 1

theorem range_g (a : ℝ) (h : Set.range (λ x => f a x) = Set.univ) : Set.range (λ x => g a x) = { y : ℝ | 1 ≤ y } := by
  sorry

end NUMINAMATH_GPT_range_g_l1270_127086


namespace NUMINAMATH_GPT_remainder_N_div_5_is_1_l1270_127064

-- The statement proving the remainder of N when divided by 5 is 1
theorem remainder_N_div_5_is_1 (N : ℕ) (h1 : N % 2 = 1) (h2 : N % 35 = 1) : N % 5 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_N_div_5_is_1_l1270_127064


namespace NUMINAMATH_GPT_apple_tree_fruits_production_l1270_127044

def apple_production (first_season : ℕ) (second_season : ℕ) (third_season : ℕ): ℕ :=
  first_season + second_season + third_season

theorem apple_tree_fruits_production :
  let first_season := 200
  let second_season := 160    -- 200 - 20% of 200
  let third_season := 320     -- 2 * 160
  apple_production first_season second_season third_season = 680 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_apple_tree_fruits_production_l1270_127044


namespace NUMINAMATH_GPT_john_cuts_his_grass_to_l1270_127033

theorem john_cuts_his_grass_to (growth_rate monthly_cost annual_cost cut_height : ℝ)
  (h : ℝ) : 
  growth_rate = 0.5 ∧ monthly_cost = 100 ∧ annual_cost = 300 ∧ cut_height = 4 →
  h = 2 := by
  intros conditions
  sorry

end NUMINAMATH_GPT_john_cuts_his_grass_to_l1270_127033


namespace NUMINAMATH_GPT_find_percentage_l1270_127066

variable (P : ℝ)

/-- A number P% that satisfies the condition is 65. -/
theorem find_percentage (h : ((P / 100) * 40 = ((5 / 100) * 60) + 23)) : P = 65 :=
sorry

end NUMINAMATH_GPT_find_percentage_l1270_127066


namespace NUMINAMATH_GPT_train_length_proof_l1270_127088

def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 5 / 18

theorem train_length_proof (speed_kmph : ℕ) (platform_length_m : ℕ) (crossing_time_s : ℕ) (speed_mps : ℕ) (distance_covered_m : ℕ) (train_length_m : ℕ) :
  speed_kmph = 72 →
  platform_length_m = 270 →
  crossing_time_s = 26 →
  speed_mps = convert_kmph_to_mps speed_kmph →
  distance_covered_m = speed_mps * crossing_time_s →
  train_length_m = distance_covered_m - platform_length_m →
  train_length_m = 250 :=
by
  intros h_speed h_platform h_time h_conv h_dist h_train_length
  sorry

end NUMINAMATH_GPT_train_length_proof_l1270_127088


namespace NUMINAMATH_GPT_brownies_pieces_l1270_127061

theorem brownies_pieces (tray_length tray_width piece_length piece_width : ℕ) 
  (h1 : tray_length = 24) 
  (h2 : tray_width = 16) 
  (h3 : piece_length = 2) 
  (h4 : piece_width = 2) : 
  tray_length * tray_width / (piece_length * piece_width) = 96 :=
by sorry

end NUMINAMATH_GPT_brownies_pieces_l1270_127061


namespace NUMINAMATH_GPT_three_lines_l1270_127058

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem three_lines (x y : ℝ) : (diamond x y = diamond y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) := 
by sorry

end NUMINAMATH_GPT_three_lines_l1270_127058


namespace NUMINAMATH_GPT_pipe_fill_time_without_leak_l1270_127071

theorem pipe_fill_time_without_leak (T : ℕ) :
  let pipe_with_leak_time := 10
  let leak_empty_time := 10
  ((1 / T : ℚ) - (1 / leak_empty_time) = (1 / pipe_with_leak_time)) →
  T = 5 := 
sorry

end NUMINAMATH_GPT_pipe_fill_time_without_leak_l1270_127071


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1270_127046

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1270_127046


namespace NUMINAMATH_GPT_smallest_positive_integer_l1270_127004

-- Definitions of the conditions
def condition1 (k : ℕ) : Prop := k % 10 = 9
def condition2 (k : ℕ) : Prop := k % 9 = 8
def condition3 (k : ℕ) : Prop := k % 8 = 7
def condition4 (k : ℕ) : Prop := k % 7 = 6
def condition5 (k : ℕ) : Prop := k % 6 = 5
def condition6 (k : ℕ) : Prop := k % 5 = 4
def condition7 (k : ℕ) : Prop := k % 4 = 3
def condition8 (k : ℕ) : Prop := k % 3 = 2
def condition9 (k : ℕ) : Prop := k % 2 = 1

-- Statement of the problem
theorem smallest_positive_integer : ∃ k : ℕ, 
  k > 0 ∧
  condition1 k ∧ 
  condition2 k ∧ 
  condition3 k ∧ 
  condition4 k ∧ 
  condition5 k ∧ 
  condition6 k ∧ 
  condition7 k ∧ 
  condition8 k ∧ 
  condition9 k ∧
  k = 2519 := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1270_127004


namespace NUMINAMATH_GPT_plane_passes_through_line_l1270_127038

-- Definition for a plane α and a line l
variable {α : Set Point} -- α represents the set of points in plane α
variable {l : Set Point} -- l represents the set of points in line l

-- The condition given
def passes_through (α : Set Point) (l : Set Point) : Prop :=
  l ⊆ α

-- The theorem statement
theorem plane_passes_through_line (α : Set Point) (l : Set Point) :
  passes_through α l = (l ⊆ α) :=
by
  sorry

end NUMINAMATH_GPT_plane_passes_through_line_l1270_127038


namespace NUMINAMATH_GPT_price_of_turban_l1270_127083

theorem price_of_turban : 
  ∃ T : ℝ, (9 / 12) * (90 + T) = 40 + T ↔ T = 110 :=
by
  sorry

end NUMINAMATH_GPT_price_of_turban_l1270_127083


namespace NUMINAMATH_GPT_smallest_n_l1270_127047

theorem smallest_n (n : ℕ) : 
  (25 * n = (Nat.lcm 10 (Nat.lcm 16 18)) → n = 29) :=
by sorry

end NUMINAMATH_GPT_smallest_n_l1270_127047


namespace NUMINAMATH_GPT_win_probability_l1270_127048

theorem win_probability (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose = 3 / 8) :=
by
  -- Provide the proof here if needed, but skip it
  sorry

end NUMINAMATH_GPT_win_probability_l1270_127048


namespace NUMINAMATH_GPT_geometric_sequence_second_term_l1270_127059

theorem geometric_sequence_second_term (b : ℝ) (hb : b > 0) 
  (h1 : ∃ r : ℝ, 210 * r = b) 
  (h2 : ∃ r : ℝ, b * r = 135 / 56) : 
  b = 22.5 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_l1270_127059


namespace NUMINAMATH_GPT_average_of_first_12_is_14_l1270_127006

-- Definitions based on given conditions
def average_of_25 := 19
def sum_of_25 := average_of_25 * 25

def average_of_last_12 := 17
def sum_of_last_12 := average_of_last_12 * 12

def result_13 := 103

-- Main proof statement to be checked
theorem average_of_first_12_is_14 (A : ℝ) (h1 : sum_of_25 = sum_of_25) (h2 : sum_of_last_12 = sum_of_last_12) (h3 : result_13 = 103) :
  (A * 12 + result_13 + sum_of_last_12 = sum_of_25) → (A = 14) :=
by
  sorry

end NUMINAMATH_GPT_average_of_first_12_is_14_l1270_127006


namespace NUMINAMATH_GPT_dino_finances_l1270_127001

def earnings_per_gig (hours: ℕ) (rate: ℕ) : ℕ := hours * rate

def dino_total_income : ℕ :=
  earnings_per_gig 20 10 + -- Earnings from the first gig
  earnings_per_gig 30 20 + -- Earnings from the second gig
  earnings_per_gig 5 40    -- Earnings from the third gig

def dino_expenses : ℕ := 500

def dino_net_income : ℕ :=
  dino_total_income - dino_expenses

theorem dino_finances : 
  dino_net_income = 500 :=
by
  -- Here, the actual proof would be constructed.
  sorry

end NUMINAMATH_GPT_dino_finances_l1270_127001


namespace NUMINAMATH_GPT_henry_added_water_l1270_127062

theorem henry_added_water (F : ℕ) (h2 : F = 32) (α β : ℚ) (h3 : α = 3/4) (h4 : β = 7/8) :
  (F * β) - (F * α) = 4 := by
  sorry

end NUMINAMATH_GPT_henry_added_water_l1270_127062


namespace NUMINAMATH_GPT_Bobby_has_27_pairs_l1270_127013

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end NUMINAMATH_GPT_Bobby_has_27_pairs_l1270_127013


namespace NUMINAMATH_GPT_evaluate_expression_l1270_127094

theorem evaluate_expression :
  2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1270_127094


namespace NUMINAMATH_GPT_arithmetic_identity_l1270_127075

theorem arithmetic_identity : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_identity_l1270_127075


namespace NUMINAMATH_GPT_poly_comp_eq_l1270_127014

variable {K : Type*} [Field K]

theorem poly_comp_eq {Q1 Q2 : Polynomial K} (P : Polynomial K) (hP : ¬P.degree = 0) :
  Q1.comp P = Q2.comp P → Q1 = Q2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_poly_comp_eq_l1270_127014


namespace NUMINAMATH_GPT_general_formula_a_sum_T_max_k_value_l1270_127070

-- Given conditions
noncomputable def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 6 else n + 5
noncomputable def b (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))
noncomputable def T (n : ℕ) : ℚ := (3 * n) / (2 * n + 1)

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = if n = 1 then 6 else n + 5 :=
by sorry

theorem sum_T (n : ℕ) : T n = (3 * n) / (2 * n + 1) :=
by sorry

theorem max_k_value (k : ℕ) : k = 19 → ∀ n : ℕ, T n > k / 20 :=
by sorry

end NUMINAMATH_GPT_general_formula_a_sum_T_max_k_value_l1270_127070


namespace NUMINAMATH_GPT_austin_needs_six_weeks_l1270_127082

theorem austin_needs_six_weeks
  (work_rate: ℕ) (hours_monday hours_wednesday hours_friday: ℕ) (bicycle_cost: ℕ) 
  (weekly_hours: ℕ := hours_monday + hours_wednesday + hours_friday) 
  (weekly_earnings: ℕ := weekly_hours * work_rate) 
  (weeks_needed: ℕ := bicycle_cost / weekly_earnings):
  work_rate = 5 ∧ hours_monday = 2 ∧ hours_wednesday = 1 ∧ hours_friday = 3 ∧ bicycle_cost = 180 ∧ weeks_needed = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_austin_needs_six_weeks_l1270_127082


namespace NUMINAMATH_GPT_total_snakes_l1270_127042

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end NUMINAMATH_GPT_total_snakes_l1270_127042


namespace NUMINAMATH_GPT_part_a_roots_part_b_sum_l1270_127041

theorem part_a_roots : ∀ x : ℝ, 2^x = x + 1 ↔ x = 0 ∨ x = 1 :=
by 
  intros x
  sorry

theorem part_b_sum (f : ℝ → ℝ) (h : ∀ x : ℝ, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_part_a_roots_part_b_sum_l1270_127041


namespace NUMINAMATH_GPT_find_AD_length_l1270_127078

variables (A B C D O : Point)
variables (BO OD AO OC AB AD : ℝ)

def quadrilateral_properties (BO OD AO OC AB : ℝ) (O : Point) : Prop :=
  BO = 3 ∧ OD = 9 ∧ AO = 5 ∧ OC = 2 ∧ AB = 7

theorem find_AD_length (h : quadrilateral_properties BO OD AO OC AB O) : AD = Real.sqrt 151 :=
by
  sorry

end NUMINAMATH_GPT_find_AD_length_l1270_127078


namespace NUMINAMATH_GPT_can_divide_cube_into_71_l1270_127080

theorem can_divide_cube_into_71 : 
  ∃ (n : ℕ), n = 71 ∧ 
  (∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = f k + 7) ∧ f n = 71) :=
by
  sorry

end NUMINAMATH_GPT_can_divide_cube_into_71_l1270_127080


namespace NUMINAMATH_GPT_spacy_subsets_15_l1270_127055

def spacy_subsets_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | (k + 5) => spacy_subsets_count (k + 4) + spacy_subsets_count k

theorem spacy_subsets_15 : spacy_subsets_count 15 = 181 :=
sorry

end NUMINAMATH_GPT_spacy_subsets_15_l1270_127055


namespace NUMINAMATH_GPT_slow_train_speed_l1270_127028

/-- Given the conditions of two trains traveling towards each other and their meeting times,
     prove the speed of the slow train. -/
theorem slow_train_speed :
  let distance_AB := 901
  let slow_train_departure := 5 + 30 / 60 -- 5:30 AM in decimal hours
  let fast_train_departure := 9 + 30 / 60 -- 9:30 AM in decimal hours
  let meeting_time := 16 + 30 / 60 -- 4:30 PM in decimal hours
  let fast_train_speed := 58 -- speed in km/h
  let slow_train_time := meeting_time - slow_train_departure
  let fast_train_time := meeting_time - fast_train_departure
  let fast_train_distance := fast_train_speed * fast_train_time
  let slow_train_distance := distance_AB - fast_train_distance
  let slow_train_speed := slow_train_distance / slow_train_time
  slow_train_speed = 45 := sorry

end NUMINAMATH_GPT_slow_train_speed_l1270_127028


namespace NUMINAMATH_GPT_reflect_center_is_image_center_l1270_127056

def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem reflect_center_is_image_center : 
  reflect_over_y_eq_neg_x (3, -4) = (4, -3) :=
by
  -- Proof is omitted as per instructions.
  -- This proof would show the reflection of the point (3, -4) over the line y = -x resulting in (4, -3).
  sorry

end NUMINAMATH_GPT_reflect_center_is_image_center_l1270_127056


namespace NUMINAMATH_GPT_deepak_age_l1270_127097

theorem deepak_age (A D : ℕ)
  (h1 : A / D = 2 / 3)
  (h2 : A + 5 = 25) :
  D = 30 := 
by
  sorry

end NUMINAMATH_GPT_deepak_age_l1270_127097


namespace NUMINAMATH_GPT_polynomial_identity_equals_neg_one_l1270_127027

theorem polynomial_identity_equals_neg_one
  (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_identity_equals_neg_one_l1270_127027


namespace NUMINAMATH_GPT_find_larger_box_ounces_l1270_127084

-- Define the conditions
def ounces_smaller_box : ℕ := 20
def cost_smaller_box : ℝ := 3.40
def cost_larger_box : ℝ := 4.80
def best_value_price_per_ounce : ℝ := 0.16

-- Define the question and its expected answer
def expected_ounces_larger_box : ℕ := 30

-- Proof statement
theorem find_larger_box_ounces :
  (cost_larger_box / best_value_price_per_ounce = expected_ounces_larger_box) :=
by
  sorry

end NUMINAMATH_GPT_find_larger_box_ounces_l1270_127084


namespace NUMINAMATH_GPT_prop_logic_example_l1270_127037

theorem prop_logic_example (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end NUMINAMATH_GPT_prop_logic_example_l1270_127037


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1270_127016

theorem quadratic_two_distinct_real_roots : 
  ∀ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 → 
  b^2 - 4 * a * c > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1270_127016


namespace NUMINAMATH_GPT_exists_nonneg_coefs_some_n_l1270_127054

-- Let p(x) be a polynomial with real coefficients
variable (p : Polynomial ℝ)

-- Assumption: p(x) > 0 for all x >= 0
axiom positive_poly : ∀ x : ℝ, x ≥ 0 → p.eval x > 0 

theorem exists_nonneg_coefs_some_n :
  ∃ n : ℕ, ∀ k : ℕ, Polynomial.coeff ((1 + Polynomial.X)^n * p) k ≥ 0 :=
sorry

end NUMINAMATH_GPT_exists_nonneg_coefs_some_n_l1270_127054


namespace NUMINAMATH_GPT_cards_given_l1270_127003

/-- Martha starts with 3 cards. She ends up with 79 cards after receiving some from Emily. We need to prove that Emily gave her 76 cards. -/
theorem cards_given (initial_cards final_cards cards_given : ℕ) (h1 : initial_cards = 3) (h2 : final_cards = 79) (h3 : final_cards = initial_cards + cards_given) :
  cards_given = 76 :=
sorry

end NUMINAMATH_GPT_cards_given_l1270_127003


namespace NUMINAMATH_GPT_solve_for_x_l1270_127010

theorem solve_for_x (x : ℚ) (h : x + 3 * x = 300 - (4 * x + 5 * x)) : x = 300 / 13 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1270_127010


namespace NUMINAMATH_GPT_expected_value_is_20_point_5_l1270_127072

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def coin_heads_probability : ℚ := 1 / 2

noncomputable def expected_value : ℚ :=
  coin_heads_probability * (penny_value + nickel_value + dime_value + quarter_value)

theorem expected_value_is_20_point_5 :
  expected_value = 20.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_20_point_5_l1270_127072


namespace NUMINAMATH_GPT_necessary_not_sufficient_to_form_triangle_l1270_127007

-- Define the vectors and the condition
variables (a b c : ℝ × ℝ)

-- Define the condition that these vectors form a closed loop (triangle)
def forms_closed_loop (a b c : ℝ × ℝ) : Prop :=
  a + b + c = (0, 0)

-- Prove that the condition is necessary but not sufficient
theorem necessary_not_sufficient_to_form_triangle :
  forms_closed_loop a b c → ∃ (x : ℝ × ℝ), a ≠ x ∧ b ≠ -2 * x ∧ c ≠ x :=
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_to_form_triangle_l1270_127007


namespace NUMINAMATH_GPT_find_larger_number_l1270_127077

theorem find_larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : 2 * (x + y) = 40) : x = 12.5 :=
by 
  sorry

end NUMINAMATH_GPT_find_larger_number_l1270_127077


namespace NUMINAMATH_GPT_original_decimal_l1270_127022

theorem original_decimal (x : ℝ) : (10 * x = x + 2.7) → x = 0.3 := 
by
    intro h
    sorry

end NUMINAMATH_GPT_original_decimal_l1270_127022


namespace NUMINAMATH_GPT_no_such_n_l1270_127065

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end NUMINAMATH_GPT_no_such_n_l1270_127065


namespace NUMINAMATH_GPT_friend1_reading_time_friend2_reading_time_l1270_127035

theorem friend1_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time / 2) : 
  ∃ t1 : ℕ, t1 = 90 := by
  sorry

theorem friend2_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time * 2) : 
  ∃ t2 : ℕ, t2 = 360 := by
  sorry

end NUMINAMATH_GPT_friend1_reading_time_friend2_reading_time_l1270_127035
