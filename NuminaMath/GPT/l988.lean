import Mathlib

namespace NUMINAMATH_GPT_simplify_and_evaluate_l988_98854

-- Definitions of given conditions
def a := 1
def b := 2

-- Statement of the theorem
theorem simplify_and_evaluate : (a * b + (a^2 - a * b) - (a^2 - 2 * a * b) = 4) :=
by
  -- Using sorry to indicate the proof is to be completed
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l988_98854


namespace NUMINAMATH_GPT_not_always_divisible_by_40_l988_98874

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem not_always_divisible_by_40 (p : ℕ) (hp_prime : is_prime p) (hp_geq7 : p ≥ 7) : ¬ (∀ p : ℕ, is_prime p ∧ p ≥ 7 → 40 ∣ (p^2 - 1)) := 
sorry

end NUMINAMATH_GPT_not_always_divisible_by_40_l988_98874


namespace NUMINAMATH_GPT_resulting_polygon_sides_l988_98855

theorem resulting_polygon_sides :
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let decagon_sides := 10
  let shared_square_decagon := 2
  let shared_between_others := 2 * 5 -- 2 sides shared for pentagon to nonagon
  let total_shared_sides := shared_square_decagon + shared_between_others
  let total_unshared_sides := 
    square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides + decagon_sides
  total_unshared_sides - total_shared_sides = 37 := by
  sorry

end NUMINAMATH_GPT_resulting_polygon_sides_l988_98855


namespace NUMINAMATH_GPT_distance_PQ_is_12_miles_l988_98815

-- Define the conditions
def average_speed_PQ := 40 -- mph
def average_speed_QP := 45 -- mph
def time_difference := 2 -- minutes

-- Main proof statement to show that the distance is 12 miles
theorem distance_PQ_is_12_miles 
    (x : ℝ) 
    (h1 : average_speed_PQ > 0) 
    (h2 : average_speed_QP > 0) 
    (h3 : abs ((x / average_speed_PQ * 60) - (x / average_speed_QP * 60)) = time_difference) 
    : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_distance_PQ_is_12_miles_l988_98815


namespace NUMINAMATH_GPT_total_wheels_of_four_wheelers_l988_98818

-- Define the number of four-wheelers and wheels per four-wheeler
def number_of_four_wheelers : ℕ := 13
def wheels_per_four_wheeler : ℕ := 4

-- Prove the total number of wheels for the 13 four-wheelers
theorem total_wheels_of_four_wheelers : (number_of_four_wheelers * wheels_per_four_wheeler) = 52 :=
by sorry

end NUMINAMATH_GPT_total_wheels_of_four_wheelers_l988_98818


namespace NUMINAMATH_GPT_average_age_choir_l988_98882

theorem average_age_choir (S_f S_m S_total : ℕ) (avg_f : ℕ) (avg_m : ℕ) (females males total : ℕ)
  (h1 : females = 8) (h2 : males = 12) (h3 : total = 20)
  (h4 : avg_f = 25) (h5 : avg_m = 40)
  (h6 : S_f = avg_f * females) 
  (h7 : S_m = avg_m * males) 
  (h8 : S_total = S_f + S_m) :
  (S_total / total) = 34 := by
  sorry

end NUMINAMATH_GPT_average_age_choir_l988_98882


namespace NUMINAMATH_GPT_no_perfect_square_integers_l988_98850

open Nat

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 10 * x^2 + 4 * x + 29

theorem no_perfect_square_integers : ∀ x : ℤ, ¬∃ a : ℤ, Q x = a^2 :=
by
  sorry

end NUMINAMATH_GPT_no_perfect_square_integers_l988_98850


namespace NUMINAMATH_GPT_distance_between_points_l988_98827

def distance_on_line (a b : ℝ) : ℝ := |b - a|

theorem distance_between_points (a b : ℝ) : distance_on_line a b = |b - a| :=
by sorry

end NUMINAMATH_GPT_distance_between_points_l988_98827


namespace NUMINAMATH_GPT_arithmetic_square_root_of_3_neg_2_l988_98825

theorem arithmetic_square_root_of_3_neg_2 : Real.sqrt (3 ^ (-2: Int)) = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_3_neg_2_l988_98825


namespace NUMINAMATH_GPT_shoe_size_15_is_9point25_l988_98866

noncomputable def smallest_shoe_length (L : ℝ) := L
noncomputable def largest_shoe_length (L : ℝ) := L + 9 * (1/4 : ℝ)
noncomputable def length_ratio_condition (L : ℝ) := largest_shoe_length L = 1.30 * smallest_shoe_length L
noncomputable def shoe_length_size_15 (L : ℝ) := L + 7 * (1/4 : ℝ)

theorem shoe_size_15_is_9point25 : ∃ L : ℝ, length_ratio_condition L → shoe_length_size_15 L = 9.25 :=
by
  sorry

end NUMINAMATH_GPT_shoe_size_15_is_9point25_l988_98866


namespace NUMINAMATH_GPT_min_value_lemma_min_value_achieved_l988_98833

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2)

theorem min_value_lemma : ∀ (x : ℝ), f x ≥ Real.sqrt 5 := 
by
  intro x
  sorry

theorem min_value_achieved : ∃ (x : ℝ), f x = Real.sqrt 5 :=
by
  use 1 / 3
  sorry

end NUMINAMATH_GPT_min_value_lemma_min_value_achieved_l988_98833


namespace NUMINAMATH_GPT_common_ratio_is_4_l988_98879

theorem common_ratio_is_4 
  (a : ℕ → ℝ) -- The geometric sequence
  (r : ℝ) -- The common ratio
  (h_geo_seq : ∀ n, a (n + 1) = r * a n) -- Definition of geometric sequence
  (h_condition : ∀ n, a n * a (n + 1) = 16 ^ n) -- Given condition
  : r = 4 := 
  sorry

end NUMINAMATH_GPT_common_ratio_is_4_l988_98879


namespace NUMINAMATH_GPT_sum_fractions_l988_98841

theorem sum_fractions : 
  (1/2 + 1/6 + 1/12 + 1/20 + 1/30 + 1/42 = 6/7) :=
by
  sorry

end NUMINAMATH_GPT_sum_fractions_l988_98841


namespace NUMINAMATH_GPT_wrestler_teams_possible_l988_98861

theorem wrestler_teams_possible :
  ∃ (team1 team2 team3 : Finset ℕ),
  (team1 ∪ team2 ∪ team3 = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (team1 ∩ team2 = ∅) ∧ (team1 ∩ team3 = ∅) ∧ (team2 ∩ team3 = ∅) ∧
  (team1.card = 3) ∧ (team2.card = 3) ∧ (team3.card = 3) ∧
  (team1.sum id = 15) ∧ (team2.sum id = 15) ∧ (team3.sum id = 15) ∧
  (∀ x ∈ team1, ∀ y ∈ team2, x > y) ∧
  (∀ x ∈ team2, ∀ y ∈ team3, x > y) ∧
  (∀ x ∈ team3, ∀ y ∈ team1, x > y) := sorry

end NUMINAMATH_GPT_wrestler_teams_possible_l988_98861


namespace NUMINAMATH_GPT_min_total_fund_Required_l988_98822

noncomputable def sell_price_A (x : ℕ) : ℕ := x + 10
noncomputable def cost_A (x : ℕ) : ℕ := 600
noncomputable def cost_B (x : ℕ) : ℕ := 400

def num_barrels_A_B_purchased (x : ℕ) := cost_A x / (sell_price_A x) = cost_B x / x

noncomputable def total_cost (m : ℕ) : ℕ := 10 * m + 10000

theorem min_total_fund_Required (price_A price_B m total : ℕ) :
  price_B = 20 →
  price_A = 30 →
  price_A = price_B + 10 →
  (num_barrels_A_B_purchased price_B) →
  total = total_cost m →
  m = 250 →
  total = 12500 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_min_total_fund_Required_l988_98822


namespace NUMINAMATH_GPT_problem_l988_98859

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x ^ 2 - x - a * Real.log (x - a)

def monotonicity_f (a : ℝ) : Prop :=
  if a = 0 then
    ∀ x : ℝ, 0 < x → (x < 1 → f x 0 < f (x + 1) 0) ∧ (x > 1 → f x 0 > f (x + 1) 0)
  else if a > 0 then
    ∀ x : ℝ, a < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f x a > f (x + 1) a)
  else if -1 < a ∧ a < 0 then
    ∀ x : ℝ, 0 < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f (x + 1) a > f x a)
  else if a = -1 then
    ∀ x : ℝ, -1 < x → f x (-1) < f (x + 1) (-1)
  else
    ∀ x : ℝ, a < x → (x < 0 → f (x + 1) a > f x a) ∧ (0 < x → f x a > f (x + 1) a)

noncomputable def g (x a : ℝ) : ℝ := f (x + a) a - a * (x + (1/2) * a - 1)

def extreme_points (x₁ x₂ a : ℝ) : Prop :=
  x₁ < x₂ ∧ ∀ x : ℝ, 0 < x → x < 1 → g x a = 0

theorem problem (a : ℝ) (x₁ x₂ : ℝ) (hx : extreme_points x₁ x₂ a) (h_dom : -1/4 < a ∧ a < 0) :
  0 < f x₁ a - f x₂ a ∧ f x₁ a - f x₂ a < 1/2 := sorry

end NUMINAMATH_GPT_problem_l988_98859


namespace NUMINAMATH_GPT_sum_of_numbers_l988_98898

theorem sum_of_numbers :
  2.12 + 0.004 + 0.345 = 2.469 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l988_98898


namespace NUMINAMATH_GPT_green_tractor_price_l988_98851

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end NUMINAMATH_GPT_green_tractor_price_l988_98851


namespace NUMINAMATH_GPT_smallest_m_l988_98852

theorem smallest_m (m : ℤ) (h : 2 * m + 1 ≥ 0) : m ≥ 0 :=
sorry

end NUMINAMATH_GPT_smallest_m_l988_98852


namespace NUMINAMATH_GPT_ferris_wheel_seats_l988_98837

theorem ferris_wheel_seats (S : ℕ) (h1 : ∀ (p : ℕ), p = 9) (h2 : ∀ (r : ℕ), r = 18) (h3 : 9 * S = 18) : S = 2 :=
by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seats_l988_98837


namespace NUMINAMATH_GPT_sin_half_angle_product_lt_quarter_l988_98820

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end NUMINAMATH_GPT_sin_half_angle_product_lt_quarter_l988_98820


namespace NUMINAMATH_GPT_quadratic_inequality_l988_98807

theorem quadratic_inequality (a b c d x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 = -a) 
  (h2 : x1 * x2 = b)
  (h3 : x3 + x4 = -c)
  (h4 : x3 * x4 = d)
  (h5 : b > d)
  (h6 : b > 0)
  (h7 : d > 0) :
  a^2 - c^2 > b - d :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l988_98807


namespace NUMINAMATH_GPT_gum_boxes_l988_98845

theorem gum_boxes (c s t g : ℕ) (h1 : c = 2) (h2 : s = 5) (h3 : t = 9) (h4 : c + s + g = t) : g = 2 := by
  sorry

end NUMINAMATH_GPT_gum_boxes_l988_98845


namespace NUMINAMATH_GPT_conic_sections_parabolas_l988_98869

theorem conic_sections_parabolas (x y : ℝ) :
  (y^6 - 9*x^6 = 3*y^3 - 1) → 
  ((y^3 = 3*x^3 + 1) ∨ (y^3 = -3*x^3 + 1)) := 
by 
  sorry

end NUMINAMATH_GPT_conic_sections_parabolas_l988_98869


namespace NUMINAMATH_GPT_speed_conversion_l988_98896

theorem speed_conversion (speed_mps: ℝ) (conversion_factor: ℝ) (expected_speed_kmph: ℝ):
  speed_mps * conversion_factor = expected_speed_kmph :=
by
  let speed_mps := 115.00919999999999
  let conversion_factor := 3.6
  let expected_speed_kmph := 414.03312
  sorry

end NUMINAMATH_GPT_speed_conversion_l988_98896


namespace NUMINAMATH_GPT_range_of_a_l988_98824

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 1 then a^x else (a - 3) * x + 4 * a

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 0 < a ∧ a ≤ 3 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l988_98824


namespace NUMINAMATH_GPT_ratio_brother_to_joanna_l988_98801

/-- Definitions for the conditions -/
def joanna_money : ℝ := 8
def sister_money : ℝ := 4 -- since it's half of Joanna's money
def total_money : ℝ := 36

/-- Stating the theorem -/
theorem ratio_brother_to_joanna (x : ℝ) (h : joanna_money + 8*x + sister_money = total_money) :
  x = 3 :=
by 
  -- The ratio of brother's money to Joanna's money is 3:1
  sorry

end NUMINAMATH_GPT_ratio_brother_to_joanna_l988_98801


namespace NUMINAMATH_GPT_unique_real_solution_l988_98884

theorem unique_real_solution :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (ab + 1) ∧ a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_real_solution_l988_98884


namespace NUMINAMATH_GPT_find_2023rd_digit_of_11_div_13_l988_98862

noncomputable def decimal_expansion_repeating (n d : Nat) : List Nat := sorry

noncomputable def decimal_expansion_digit (n d pos : Nat) : Nat :=
  let repeating_block := decimal_expansion_repeating n d
  repeating_block.get! ((pos - 1) % repeating_block.length)

theorem find_2023rd_digit_of_11_div_13 :
  decimal_expansion_digit 11 13 2023 = 8 := by
  sorry

end NUMINAMATH_GPT_find_2023rd_digit_of_11_div_13_l988_98862


namespace NUMINAMATH_GPT_mushroom_mass_decrease_l988_98847

theorem mushroom_mass_decrease :
  ∀ (initial_mass water_content_fresh water_content_dry : ℝ),
  water_content_fresh = 0.8 →
  water_content_dry = 0.2 →
  (initial_mass * (1 - water_content_fresh) / (1 - water_content_dry) = initial_mass * 0.25) →
  (initial_mass - initial_mass * 0.25) / initial_mass = 0.75 :=
by
  intros initial_mass water_content_fresh water_content_dry h_fresh h_dry h_dry_mass
  sorry

end NUMINAMATH_GPT_mushroom_mass_decrease_l988_98847


namespace NUMINAMATH_GPT_tenth_number_in_twentieth_row_l988_98856

def arrangement : ∀ n : ℕ, ℕ := -- A function defining the nth number in the sequence.
  sorry

-- A function to get the nth number in the mth row, respecting the arithmetic sequence property.
def number_in_row (m n : ℕ) : ℕ := 
  sorry

theorem tenth_number_in_twentieth_row : number_in_row 20 10 = 426 :=
  sorry

end NUMINAMATH_GPT_tenth_number_in_twentieth_row_l988_98856


namespace NUMINAMATH_GPT_quadratic_roots_l988_98858

theorem quadratic_roots {x y : ℝ} (h1 : x + y = 8) (h2 : |x - y| = 10) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (x^2 - 8*x - 9 = 0) ∧ (y^2 - 8*y - 9 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l988_98858


namespace NUMINAMATH_GPT_find_coals_per_bag_l988_98830

open Nat

variable (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ)

def coal_per_bag (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ) : ℕ :=
  (totalTime / timePerSet) * burnRate / totalBags

theorem find_coals_per_bag :
  coal_per_bag 15 20 240 3 = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_coals_per_bag_l988_98830


namespace NUMINAMATH_GPT_day_100_M_minus_1_is_Tuesday_l988_98890

variable {M : ℕ}

-- Given conditions
def day_200_M_is_Monday (M : ℕ) : Prop :=
  ((200 % 7) = 6)

def day_300_M_plus_2_is_Monday (M : ℕ) : Prop :=
  ((300 % 7) = 6)

-- Statement to prove
theorem day_100_M_minus_1_is_Tuesday (M : ℕ) 
  (h1 : day_200_M_is_Monday M) 
  (h2 : day_300_M_plus_2_is_Monday M) 
  : (((100 + (365 - 200)) % 7 + 7 - 1) % 7 = 2) :=
sorry

end NUMINAMATH_GPT_day_100_M_minus_1_is_Tuesday_l988_98890


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l988_98826

theorem breadth_of_rectangular_plot (b l A : ℝ) (h1 : l = 3 * b) (h2 : A = 588) (h3 : A = l * b) : b = 14 :=
by
  -- We start our proof here
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l988_98826


namespace NUMINAMATH_GPT_complement_intersection_l988_98892

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) : (U \ (M ∩ N)) = {1, 4} := 
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l988_98892


namespace NUMINAMATH_GPT_ab_plus_cd_111_333_l988_98808

theorem ab_plus_cd_111_333 (a b c d : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a + b + d = 5) 
  (h3 : a + c + d = 20) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 111.333 := 
by
  sorry

end NUMINAMATH_GPT_ab_plus_cd_111_333_l988_98808


namespace NUMINAMATH_GPT_not_exists_implies_bounds_l988_98828

variable (a : ℝ)

/-- If there does not exist an x such that x^2 + (a - 1) * x + 1 < 0, then -1 ≤ a ∧ a ≤ 3. -/
theorem not_exists_implies_bounds : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_GPT_not_exists_implies_bounds_l988_98828


namespace NUMINAMATH_GPT_sandra_coffee_l988_98864

theorem sandra_coffee (S : ℕ) (H1 : 2 + S = 8) : S = 6 :=
by
  sorry

end NUMINAMATH_GPT_sandra_coffee_l988_98864


namespace NUMINAMATH_GPT_solve_for_b_l988_98895

noncomputable def P (x a b d c : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + d * x + c

theorem solve_for_b (a b d c : ℝ) (h1 : -a = d) (h2 : d = 1 + a + b + d + c) (h3 : c = 8) :
    b = -17 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l988_98895


namespace NUMINAMATH_GPT_odd_function_neg_value_l988_98883

theorem odd_function_neg_value
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  ∀ x, x < 0 → f x = -x^2 + 2 * x :=
by
  intros x hx
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_odd_function_neg_value_l988_98883


namespace NUMINAMATH_GPT_problem_statement_l988_98888

def f (x : ℝ) : ℝ := sorry

theorem problem_statement
  (cond1 : ∀ {x y w : ℝ}, x > y → f x + x ≥ w → w ≥ f y + y → ∃ (z : ℝ), z ∈ Set.Icc y x ∧ f z = w - z)
  (cond2 : ∃ (u : ℝ), 0 ∈ Set.range f ∧ ∀ a ∈ Set.range f, u ≤ a)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 := sorry

end NUMINAMATH_GPT_problem_statement_l988_98888


namespace NUMINAMATH_GPT_pirates_gold_coins_l988_98802

theorem pirates_gold_coins (S a b c d e : ℕ) (h1 : a = S / 3) (h2 : b = S / 4) (h3 : c = S / 5) (h4 : d = S / 6) (h5 : e = 90) :
  S = 1800 :=
by
  -- Definitions and assumptions would go here
  sorry

end NUMINAMATH_GPT_pirates_gold_coins_l988_98802


namespace NUMINAMATH_GPT_solve_equations_l988_98813

-- Prove that the solutions to the given equations are correct.
theorem solve_equations :
  (∀ x : ℝ, (x * (x - 4) = 2 * x - 8) ↔ (x = 4 ∨ x = 2)) ∧
  (∀ x : ℝ, ((2 * x) / (2 * x - 3) - (4 / (2 * x + 3)) = 1) ↔ (x = 10.5)) :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l988_98813


namespace NUMINAMATH_GPT_probability_green_ball_l988_98812

/-- 
Given three containers with specific numbers of red and green balls, 
and the probability of selecting each container being equal, 
the probability of picking a green ball when choosing a container randomly is 7/12.
-/
theorem probability_green_ball :
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  (green_I + green_II + green_III) = 7 / 12 :=
by 
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  have : (green_I + green_II + green_III) = (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) := by rfl
  have : (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) = (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) := by rfl
  have : (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) = (1 / 9 + 2 / 9 + 1 / 4) := by rfl
  have : (1 / 9 + 2 / 9 + 1 / 4) = (4 / 36 + 8 / 36 + 9 / 36) := by rfl
  have : (4 / 36 + 8 / 36 + 9 / 36) = 21 / 36 := by rfl
  have : 21 / 36 = 7 / 12 := by rfl
  rfl

end NUMINAMATH_GPT_probability_green_ball_l988_98812


namespace NUMINAMATH_GPT_number_of_bags_of_chips_l988_98843

theorem number_of_bags_of_chips (friends : ℕ) (amount_per_friend : ℕ) (cost_per_bag : ℕ) (total_amount : ℕ) (number_of_bags : ℕ) : 
  friends = 3 → amount_per_friend = 5 → cost_per_bag = 3 → total_amount = friends * amount_per_friend → number_of_bags = total_amount / cost_per_bag → number_of_bags = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_number_of_bags_of_chips_l988_98843


namespace NUMINAMATH_GPT_sum_of_six_angles_l988_98897

theorem sum_of_six_angles (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 + a3 + a5 = 180)
                           (h2 : a2 + a4 + a6 = 180) : 
                           a1 + a2 + a3 + a4 + a5 + a6 = 360 := 
by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_sum_of_six_angles_l988_98897


namespace NUMINAMATH_GPT_difference_divisible_l988_98848

theorem difference_divisible (a b n : ℕ) (h : n % 2 = 0) (hab : a + b = 61) :
  (47^100 - 14^100) % 61 = 0 := by
  sorry

end NUMINAMATH_GPT_difference_divisible_l988_98848


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l988_98835

-- Equation 1 Statement
theorem equation1_solution (x : ℝ) : 
  (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ↔ x = -20 :=
by sorry

-- Equation 2 Statement
theorem equation2_solution (x : ℝ) : 
  (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ↔ x = 67 / 23 :=
by sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l988_98835


namespace NUMINAMATH_GPT_determine_d_l988_98839

variables (a b c d : ℝ)

-- Conditions given in the problem
def condition1 (a b d : ℝ) : Prop := d / a = (d - 25) / b
def condition2 (b c d : ℝ) : Prop := d / b = (d - 15) / c
def condition3 (a c d : ℝ) : Prop := d / a = (d - 35) / c

-- Final statement to prove
theorem determine_d (a b c : ℝ) (d : ℝ) :
    condition1 a b d ∧ condition2 b c d ∧ condition3 a c d → d = 75 :=
by sorry

end NUMINAMATH_GPT_determine_d_l988_98839


namespace NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_l988_98838

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end NUMINAMATH_GPT_greatest_two_digit_multiple_of_17_l988_98838


namespace NUMINAMATH_GPT_range_of_f_l988_98829

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f : Set.Icc (-(3 / 2)) 3 = Set.image f (Set.Icc 0 (Real.pi / 2)) :=
  sorry

end NUMINAMATH_GPT_range_of_f_l988_98829


namespace NUMINAMATH_GPT_minimum_milk_candies_l988_98803

/-- A supermarket needs to purchase candies with the following conditions:
 1. The number of watermelon candies is at most 3 times the number of chocolate candies.
 2. The number of milk candies is at least 4 times the number of chocolate candies.
 3. The sum of chocolate candies and watermelon candies is at least 2020.

 Prove that the minimum number of milk candies that need to be purchased is 2020. -/
theorem minimum_milk_candies (x y z : ℕ)
  (h1 : y ≤ 3 * x)
  (h2 : z ≥ 4 * x)
  (h3 : x + y ≥ 2020) :
  z ≥ 2020 :=
sorry

end NUMINAMATH_GPT_minimum_milk_candies_l988_98803


namespace NUMINAMATH_GPT_max_value_ineq_l988_98809

theorem max_value_ineq (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_ineq_l988_98809


namespace NUMINAMATH_GPT_hillary_descending_rate_l988_98893

def baseCampDistance : ℕ := 4700
def hillaryClimbingRate : ℕ := 800
def eddyClimbingRate : ℕ := 500
def hillaryStopShort : ℕ := 700
def departTime : ℕ := 6 -- time is represented in hours from midnight
def passTime : ℕ := 12 -- time is represented in hours from midnight

theorem hillary_descending_rate :
  ∃ r : ℕ, r = 1000 := by
  sorry

end NUMINAMATH_GPT_hillary_descending_rate_l988_98893


namespace NUMINAMATH_GPT_jane_book_pages_l988_98877

theorem jane_book_pages (x : ℝ) :
  (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20) - (1 / 2 * (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20)) + 25) = 75) → x = 380 :=
by
  sorry

end NUMINAMATH_GPT_jane_book_pages_l988_98877


namespace NUMINAMATH_GPT_turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l988_98819

-- Definitions based on conditions
def turnover_first_four_days : ℝ := 450
def turnover_fifth_day : ℝ := 0.12 * turnover_first_four_days
def total_turnover_five_days : ℝ := turnover_first_four_days + turnover_fifth_day

-- Proof statement for part 1
theorem turnover_five_days_eq_504 :
  total_turnover_five_days = 504 := 
sorry

-- Definitions and conditions for part 2
def turnover_february : ℝ := 350
def turnover_april : ℝ := total_turnover_five_days
def growth_rate (x : ℝ) : Prop := (1 + x)^2 * turnover_february = turnover_april

-- Proof statement for part 2
theorem monthly_growth_rate_eq_20_percent :
  ∃ x : ℝ, growth_rate x ∧ x = 0.2 := 
sorry

end NUMINAMATH_GPT_turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l988_98819


namespace NUMINAMATH_GPT_value_of_k_l988_98894

def f (x : ℝ) := 4 * x ^ 2 - 5 * x + 6
def g (x : ℝ) (k : ℝ) := 2 * x ^ 2 - k * x + 1

theorem value_of_k :
  (f 5 - g 5 k = 30) → k = -10 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_k_l988_98894


namespace NUMINAMATH_GPT_point_of_tangency_l988_98878

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)
noncomputable def f_deriv (x a : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem point_of_tangency (a : ℝ) (h1 : ∀ x, f_deriv (-x) a = -f_deriv x a)
  (h2 : ∃ x0, f_deriv x0 1 = 3/2) :
  ∃ x0 y0, x0 = Real.log 2 ∧ y0 = f (Real.log 2) 1 ∧ y0 = 5/2 :=
by
  sorry

end NUMINAMATH_GPT_point_of_tangency_l988_98878


namespace NUMINAMATH_GPT_second_train_further_l988_98872

-- Define the speeds of the two trains
def speed_train1 : ℝ := 50
def speed_train2 : ℝ := 60

-- Define the total distance between points A and B
def total_distance : ℝ := 1100

-- Define the distances traveled by the two trains when they meet
def distance_train1 (t: ℝ) : ℝ := speed_train1 * t
def distance_train2 (t: ℝ) : ℝ := speed_train2 * t

-- Define the meeting condition
def meeting_condition (t: ℝ) : Prop := distance_train1 t + distance_train2 t = total_distance

-- Prove the distance difference
theorem second_train_further (t: ℝ) (h: meeting_condition t) : distance_train2 t - distance_train1 t = 100 :=
sorry

end NUMINAMATH_GPT_second_train_further_l988_98872


namespace NUMINAMATH_GPT_potassium_bromate_molecular_weight_l988_98880

def potassium_atomic_weight : Real := 39.10
def bromine_atomic_weight : Real := 79.90
def oxygen_atomic_weight : Real := 16.00
def oxygen_atoms : Nat := 3

theorem potassium_bromate_molecular_weight :
  potassium_atomic_weight + bromine_atomic_weight + oxygen_atoms * oxygen_atomic_weight = 167.00 :=
by
  sorry

end NUMINAMATH_GPT_potassium_bromate_molecular_weight_l988_98880


namespace NUMINAMATH_GPT_evaluate_f_at_2_l988_98857

def f (x : ℕ) : ℕ := 5 * x + 2

theorem evaluate_f_at_2 : f 2 = 12 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l988_98857


namespace NUMINAMATH_GPT_football_basketball_problem_l988_98867

theorem football_basketball_problem :
  ∃ (football_cost basketball_cost : ℕ),
    (3 * football_cost + basketball_cost = 230) ∧
    (2 * football_cost + 3 * basketball_cost = 340) ∧
    football_cost = 50 ∧
    basketball_cost = 80 ∧
    ∃ (basketballs footballs : ℕ),
      (basketballs + footballs = 20) ∧
      (footballs < basketballs) ∧
      (80 * basketballs + 50 * footballs ≤ 1400) ∧
      ((basketballs = 11 ∧ footballs = 9) ∨
       (basketballs = 12 ∧ footballs = 8) ∨
       (basketballs = 13 ∧ footballs = 7)) :=
by
  sorry

end NUMINAMATH_GPT_football_basketball_problem_l988_98867


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l988_98868

theorem sufficient_but_not_necessary_condition (b c : ℝ) :
  (∃ x0 : ℝ, (x0^2 + b * x0 + c) < 0) ↔ (c < 0) ∨ true :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l988_98868


namespace NUMINAMATH_GPT_area_correct_l988_98821

open BigOperators

def Rectangle (PQ RS : ℕ) := PQ * RS

def PointOnSegment (a b : ℕ) (ratio : ℚ) : ℚ :=
ratio * (b - a)

def area_of_PTUS : ℚ :=
Rectangle 10 6 - (0.5 * 6 * (10 / 3) + 0.5 * 10 * 6)

theorem area_correct :
  area_of_PTUS = 20 := by
  sorry

end NUMINAMATH_GPT_area_correct_l988_98821


namespace NUMINAMATH_GPT_cannot_use_square_difference_formula_l988_98811

theorem cannot_use_square_difference_formula (x y : ℝ) :
  ¬ ∃ a b : ℝ, (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) :=
sorry

end NUMINAMATH_GPT_cannot_use_square_difference_formula_l988_98811


namespace NUMINAMATH_GPT_packs_of_green_bouncy_balls_l988_98806

/-- Maggie bought 10 bouncy balls in each pack of red, yellow, and green bouncy balls.
    She bought 4 packs of red bouncy balls, 8 packs of yellow bouncy balls, and some 
    packs of green bouncy balls. In total, she bought 160 bouncy balls. This theorem 
    aims to prove how many packs of green bouncy balls Maggie bought. 
 -/
theorem packs_of_green_bouncy_balls (red_packs : ℕ) (yellow_packs : ℕ) (total_balls : ℕ) (balls_per_pack : ℕ) 
(pack : ℕ) :
  red_packs = 4 →
  yellow_packs = 8 →
  balls_per_pack = 10 →
  total_balls = 160 →
  red_packs * balls_per_pack + yellow_packs * balls_per_pack + pack * balls_per_pack = total_balls →
  pack = 4 :=
by
  intros h_red h_yellow h_balls_per_pack h_total_balls h_eq
  sorry

end NUMINAMATH_GPT_packs_of_green_bouncy_balls_l988_98806


namespace NUMINAMATH_GPT_eq_circle_value_of_k_l988_98817

noncomputable def circle_center : Prod ℝ ℝ := (2, 3)
noncomputable def circle_radius := 2
noncomputable def line_equation (k : ℝ) : ℝ → ℝ := fun x => k * x - 1
noncomputable def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

theorem eq_circle (x y : ℝ) : 
  circle_equation x y ↔ (x - 2)^2 + (y - 3)^2 = 4 := 
by sorry

theorem value_of_k (k : ℝ) : 
  (∀ M N : Prod ℝ ℝ, 
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 ∧ 
  line_equation k M.1 = M.2 ∧ line_equation k N.1 = N.2 ∧ 
  M ≠ N ∧ 
  (circle_center.1 - M.1) * (circle_center.1 - N.1) + 
  (circle_center.2 - M.2) * (circle_center.2 - N.2) = 0) → 
  (k = 1 ∨ k = 7) := 
by sorry

end NUMINAMATH_GPT_eq_circle_value_of_k_l988_98817


namespace NUMINAMATH_GPT_find_interval_l988_98816

theorem find_interval (n : ℕ) 
  (h1 : n < 500) 
  (h2 : n ∣ 9999) 
  (h3 : n + 4 ∣ 99) : (1 ≤ n) ∧ (n ≤ 125) := 
sorry

end NUMINAMATH_GPT_find_interval_l988_98816


namespace NUMINAMATH_GPT_f_12_16_plus_f_16_12_l988_98844

noncomputable def f : ℕ × ℕ → ℕ :=
sorry

axiom ax1 : ∀ (x : ℕ), f (x, x) = x
axiom ax2 : ∀ (x y : ℕ), f (x, y) = f (y, x)
axiom ax3 : ∀ (x y : ℕ), (x + y) * f (x, y) = y * f (x, x + y)

theorem f_12_16_plus_f_16_12 : f (12, 16) + f (16, 12) = 96 :=
by sorry

end NUMINAMATH_GPT_f_12_16_plus_f_16_12_l988_98844


namespace NUMINAMATH_GPT_identify_jars_l988_98814

namespace JarIdentification

/-- Definitions of Jar labels -/
inductive JarLabel
| Nickels
| Dimes
| Nickels_and_Dimes

open JarLabel

/-- Mislabeling conditions for each jar -/
def mislabeled (jarA : JarLabel) (jarB : JarLabel) (jarC : JarLabel) : Prop :=
  ((jarA ≠ Nickels) ∧ (jarB ≠ Dimes) ∧ (jarC ≠ Nickels_and_Dimes)) ∧
  ((jarC = Nickels ∨ jarC = Dimes))

/-- Given the result of a coin draw from the jar labeled "Nickels and Dimes" -/
def jarIdentity (jarA jarB jarC : JarLabel) (drawFromC : String) : Prop :=
  if drawFromC = "Nickel" then
    jarC = Nickels ∧ jarA = Nickels_and_Dimes ∧ jarB = Dimes
  else if drawFromC = "Dime" then
    jarC = Dimes ∧ jarB = Nickels_and_Dimes ∧ jarA = Nickels
  else 
    false

/-- Main theorem to prove the identification of jars -/
theorem identify_jars (jarA jarB jarC : JarLabel) (draw : String)
  (h1 : mislabeled jarA jarB jarC) :
  jarIdentity jarA jarB jarC draw :=
by
  sorry

end JarIdentification

end NUMINAMATH_GPT_identify_jars_l988_98814


namespace NUMINAMATH_GPT_intersect_setA_setB_l988_98800

def setA : Set ℝ := {x | x < 2}
def setB : Set ℝ := {x | 3 - 2 * x > 0}

theorem intersect_setA_setB :
  setA ∩ setB = {x | x < 3 / 2} :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_intersect_setA_setB_l988_98800


namespace NUMINAMATH_GPT_differences_multiple_of_nine_l988_98853

theorem differences_multiple_of_nine (S : Finset ℕ) (hS : S.card = 10) (h_unique : ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → x ≠ y) : 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_differences_multiple_of_nine_l988_98853


namespace NUMINAMATH_GPT_max_distance_convoy_l988_98899

structure Vehicle :=
  (mpg : ℝ) (min_gallons : ℝ)

def SUV : Vehicle := ⟨12.2, 10⟩
def Sedan : Vehicle := ⟨52, 5⟩
def Motorcycle : Vehicle := ⟨70, 2⟩

def total_gallons : ℝ := 21

def total_distance (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) : ℝ :=
  SUV.mpg * SUV_gallons + Sedan.mpg * Sedan_gallons + Motorcycle.mpg * Motorcycle_gallons

theorem max_distance_convoy (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) :
  SUV_gallons + Sedan_gallons + Motorcycle_gallons = total_gallons →
  SUV_gallons >= SUV.min_gallons →
  Sedan_gallons >= Sedan.min_gallons →
  Motorcycle_gallons >= Motorcycle.min_gallons →
  total_distance SUV_gallons Sedan_gallons Motorcycle_gallons = 802 :=
sorry

end NUMINAMATH_GPT_max_distance_convoy_l988_98899


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l988_98840

theorem quadratic_has_real_roots (k : ℝ) :
  (∃ (x : ℝ), (k-2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ (3 / 2) ∧ k ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l988_98840


namespace NUMINAMATH_GPT_smallest_n_divisibility_l988_98832

theorem smallest_n_divisibility (n : ℕ) (h : 1 ≤ n) :
  (∀ k, 1 ≤ k ∧ k ≤ n → n^3 - n ∣ k) ∨ (∃ k, 1 ≤ k ∧ k ≤ n ∧ ¬ (n^3 - n ∣ k)) :=
sorry

end NUMINAMATH_GPT_smallest_n_divisibility_l988_98832


namespace NUMINAMATH_GPT_largest_k_l988_98876

-- Define the system of equations and conditions
def system_valid (x y k : ℝ) : Prop := 
  2 * x + y = k ∧ 
  3 * x + y = 3 ∧ 
  x - 2 * y ≥ 1

-- Define the proof problem as a theorem in Lean
theorem largest_k (x y : ℝ) :
  ∀ k : ℝ, system_valid x y k → k ≤ 2 := 
sorry

end NUMINAMATH_GPT_largest_k_l988_98876


namespace NUMINAMATH_GPT_arithmetic_sequence_75th_term_l988_98846

theorem arithmetic_sequence_75th_term (a d n : ℕ) (h1 : a = 2) (h2 : d = 4) (h3 : n = 75) : 
  a + (n - 1) * d = 298 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_75th_term_l988_98846


namespace NUMINAMATH_GPT_probability_of_selection_l988_98875

theorem probability_of_selection : 
  ∀ (n k : ℕ), n = 121 ∧ k = 20 → (P : ℚ) = 20 / 121 :=
by
  intros n k h
  sorry

end NUMINAMATH_GPT_probability_of_selection_l988_98875


namespace NUMINAMATH_GPT_pqrs_sum_l988_98873

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end NUMINAMATH_GPT_pqrs_sum_l988_98873


namespace NUMINAMATH_GPT_sqrt_product_simplification_l988_98870

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l988_98870


namespace NUMINAMATH_GPT_total_test_points_l988_98804

theorem total_test_points (total_questions two_point_questions four_point_questions points_per_two_question points_per_four_question : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10)
  (h3 : points_per_two_question = 2)
  (h4 : points_per_four_question = 4)
  (h5 : two_point_questions = total_questions - four_point_questions)
  : (two_point_questions * points_per_two_question) + (four_point_questions * points_per_four_question) = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_test_points_l988_98804


namespace NUMINAMATH_GPT_quadratic_complete_square_l988_98849

theorem quadratic_complete_square :
  ∃ a b c : ℤ, (8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) ∧ (a + b + c = -387) :=
sorry

end NUMINAMATH_GPT_quadratic_complete_square_l988_98849


namespace NUMINAMATH_GPT_inequality_problem_l988_98887

-- Define a and the condition that expresses the given problem as an inequality
variable (a : ℝ)

-- The inequality to prove
theorem inequality_problem : a - 5 > 2 * a := sorry

end NUMINAMATH_GPT_inequality_problem_l988_98887


namespace NUMINAMATH_GPT_is_decreasing_on_interval_l988_98865

open Set Real

def f (x : ℝ) : ℝ := x^3 - x^2 - x

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem is_decreasing_on_interval :
  ∀ x ∈ Ioo (-1 / 3 : ℝ) 1, f' x < 0 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_is_decreasing_on_interval_l988_98865


namespace NUMINAMATH_GPT_solve_inequality_l988_98860

def solution_set_of_inequality : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

theorem solve_inequality (x : ℝ) (h : (2 - x) / (x + 4) > 0) : x ∈ solution_set_of_inequality :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l988_98860


namespace NUMINAMATH_GPT_ratio_of_area_to_perimeter_l988_98863

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_GPT_ratio_of_area_to_perimeter_l988_98863


namespace NUMINAMATH_GPT_Chicago_White_Sox_loss_l988_98886

theorem Chicago_White_Sox_loss :
  ∃ (L : ℕ), (99 = L + 36) ∧ (L = 63) :=
by
  sorry

end NUMINAMATH_GPT_Chicago_White_Sox_loss_l988_98886


namespace NUMINAMATH_GPT_tan_alpha_l988_98823

theorem tan_alpha (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan α = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_l988_98823


namespace NUMINAMATH_GPT_minimum_shirts_to_save_money_by_using_Acme_l988_98805

-- Define the cost functions for Acme and Gamma
def Acme_cost (x : ℕ) : ℕ := 60 + 8 * x
def Gamma_cost (x : ℕ) : ℕ := 12 * x

-- State the theorem to prove that for x = 16, Acme is cheaper than Gamma
theorem minimum_shirts_to_save_money_by_using_Acme : ∀ x ≥ 16, Acme_cost x < Gamma_cost x :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_minimum_shirts_to_save_money_by_using_Acme_l988_98805


namespace NUMINAMATH_GPT_max_paths_from_A_to_F_l988_98885

-- Define the points and line segments.
inductive Point
| A | B | C | D | E | F

-- Define the edges of the graph as pairs of points.
def edges : List (Point × Point) :=
  [(Point.A, Point.B), (Point.A, Point.E), (Point.A, Point.D),
   (Point.B, Point.C), (Point.B, Point.E),
   (Point.C, Point.F),
   (Point.D, Point.E), (Point.D, Point.F),
   (Point.E, Point.F)]

-- A path is valid if it passes through each point and line segment only once.
def valid_path (path : List (Point × Point)) : Bool :=
  -- Check that each edge in the path is unique and forms a sequence from A to F.
  sorry

-- Calculate the maximum number of different valid paths from point A to point F.
def max_paths : Nat :=
  List.length (List.filter valid_path (List.permutations edges))

theorem max_paths_from_A_to_F : max_paths = 9 :=
by sorry

end NUMINAMATH_GPT_max_paths_from_A_to_F_l988_98885


namespace NUMINAMATH_GPT_like_terms_sum_l988_98881

theorem like_terms_sum (m n : ℕ) (h1 : m + 1 = 1) (h2 : 3 = n) : m + n = 3 :=
by sorry

end NUMINAMATH_GPT_like_terms_sum_l988_98881


namespace NUMINAMATH_GPT_max_value_of_function_l988_98871

noncomputable def max_value (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem max_value_of_function : 
  ∀ x : ℝ, (- (Real.pi / 2)) ≤ x ∧ x ≤ 0 → max_value x ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_function_l988_98871


namespace NUMINAMATH_GPT_focus_of_parabola_l988_98834

theorem focus_of_parabola (a k : ℝ) (h_eq : ∀ x : ℝ, k = 6 ∧ a = 9) :
  (0, (1 / (4 * a)) + k) = (0, 217 / 36) := sorry

end NUMINAMATH_GPT_focus_of_parabola_l988_98834


namespace NUMINAMATH_GPT_count_perfect_squares_diff_two_consecutive_squares_l988_98889

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end NUMINAMATH_GPT_count_perfect_squares_diff_two_consecutive_squares_l988_98889


namespace NUMINAMATH_GPT_greatest_consecutive_integers_sum_55_l988_98831

theorem greatest_consecutive_integers_sum_55 :
  ∃ N a : ℤ, (N * (2 * a + N - 1)) = 110 ∧ (∀ M a' : ℤ, (M * (2 * a' + M - 1)) = 110 → N ≥ M) :=
sorry

end NUMINAMATH_GPT_greatest_consecutive_integers_sum_55_l988_98831


namespace NUMINAMATH_GPT_ratio_mara_janet_l988_98891

variables {B J M : ℕ}

/-- Janet has 9 cards more than Brenda --/
def janet_cards (B : ℕ) : ℕ := B + 9

/-- Mara has 40 cards less than 150 --/
def mara_cards : ℕ := 150 - 40

/-- They have a total of 211 cards --/
axiom total_cards_eq (B : ℕ) : B + janet_cards B + mara_cards = 211

/-- Mara has a multiple of Janet's number of cards --/
axiom multiples_cards (J M : ℕ) : J * 2 = M

theorem ratio_mara_janet (B J M : ℕ) (h1 : janet_cards B = J)
  (h2 : mara_cards = M) (h3 : J * 2 = M) :
  (M / J : ℕ) = 2 :=
sorry

end NUMINAMATH_GPT_ratio_mara_janet_l988_98891


namespace NUMINAMATH_GPT_sector_area_l988_98836

theorem sector_area (r : ℝ) (α : ℝ) (h1 : 2 * r + α * r = 16) (h2 : α = 2) :
  1 / 2 * α * r^2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l988_98836


namespace NUMINAMATH_GPT_anna_more_candy_than_billy_l988_98842

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end NUMINAMATH_GPT_anna_more_candy_than_billy_l988_98842


namespace NUMINAMATH_GPT_jose_speed_l988_98810

theorem jose_speed
  (distance : ℕ) (time : ℕ)
  (h_distance : distance = 4)
  (h_time : time = 2) :
  distance / time = 2 := by
  sorry

end NUMINAMATH_GPT_jose_speed_l988_98810
