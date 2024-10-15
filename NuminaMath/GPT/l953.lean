import Mathlib

namespace NUMINAMATH_GPT_buses_required_l953_95373

theorem buses_required (students : ℕ) (bus_capacity : ℕ) (h_students : students = 325) (h_bus_capacity : bus_capacity = 45) : 
∃ n : ℕ, n = 8 ∧ bus_capacity * n ≥ students :=
by
  sorry

end NUMINAMATH_GPT_buses_required_l953_95373


namespace NUMINAMATH_GPT_numberOfWaysToPlaceCoinsSix_l953_95324

def numberOfWaysToPlaceCoins (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * numberOfWaysToPlaceCoins (n - 1)

theorem numberOfWaysToPlaceCoinsSix : numberOfWaysToPlaceCoins 6 = 32 :=
by
  sorry

end NUMINAMATH_GPT_numberOfWaysToPlaceCoinsSix_l953_95324


namespace NUMINAMATH_GPT_recruits_line_l953_95333

theorem recruits_line
  (x y z : ℕ) 
  (hx : x + y + z + 3 = 211) 
  (hx_peter : x = 50) 
  (hy_nikolai : y = 100) 
  (hz_denis : z = 170) 
  (hxy_ratio : x = 4 * z) : 
  x + y + z + 3 = 211 :=
by
  sorry

end NUMINAMATH_GPT_recruits_line_l953_95333


namespace NUMINAMATH_GPT_base_seven_to_base_ten_l953_95358

theorem base_seven_to_base_ten (n : ℕ) (h : n = 54231) : 
  (1 * 7^0 + 3 * 7^1 + 2 * 7^2 + 4 * 7^3 + 5 * 7^4) = 13497 :=
by
  sorry

end NUMINAMATH_GPT_base_seven_to_base_ten_l953_95358


namespace NUMINAMATH_GPT_f_decreasing_on_neg_infty_2_l953_95315

def f (x : ℝ) := x^2 - 4 * x + 3

theorem f_decreasing_on_neg_infty_2 :
  ∀ x y : ℝ, x < y → y ≤ 2 → f y < f x :=
by
  sorry

end NUMINAMATH_GPT_f_decreasing_on_neg_infty_2_l953_95315


namespace NUMINAMATH_GPT_time_interval_between_recordings_is_5_seconds_l953_95322

theorem time_interval_between_recordings_is_5_seconds
  (instances_per_hour : ℕ)
  (seconds_per_hour : ℕ)
  (h1 : instances_per_hour = 720)
  (h2 : seconds_per_hour = 3600) :
  seconds_per_hour / instances_per_hour = 5 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_time_interval_between_recordings_is_5_seconds_l953_95322


namespace NUMINAMATH_GPT_expression_equals_required_value_l953_95323

-- Define the expression as needed
def expression : ℚ := (((((4 + 2)⁻¹ + 2)⁻¹) + 2)⁻¹) + 2

-- Define the theorem stating that the expression equals the required value
theorem expression_equals_required_value : 
  expression = 77 / 32 := 
sorry

end NUMINAMATH_GPT_expression_equals_required_value_l953_95323


namespace NUMINAMATH_GPT_yellow_jelly_bean_probability_l953_95372

theorem yellow_jelly_bean_probability :
  ∀ (p_red p_orange p_green p_total p_yellow : ℝ),
    p_red = 0.15 →
    p_orange = 0.35 →
    p_green = 0.25 →
    p_total = 1 →
    p_red + p_orange + p_green + p_yellow = p_total →
    p_yellow = 0.25 :=
by
  intros p_red p_orange p_green p_total p_yellow h_red h_orange h_green h_total h_sum
  sorry

end NUMINAMATH_GPT_yellow_jelly_bean_probability_l953_95372


namespace NUMINAMATH_GPT_problem_l953_95391

theorem problem (a : ℝ) (h : a^2 - 2 * a - 2 = 0) :
  (1 - 1 / (a + 1)) / (a^3 / (a^2 + 2 * a + 1)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l953_95391


namespace NUMINAMATH_GPT_common_ratio_is_half_l953_95338

variable {a₁ q : ℝ}

-- Given the conditions of the geometric sequence

-- First condition
axiom h1 : a₁ + a₁ * q ^ 2 = 10

-- Second condition
axiom h2 : a₁ * q ^ 3 + a₁ * q ^ 5 = 5 / 4

-- Proving that the common ratio q is 1/2
theorem common_ratio_is_half : q = 1 / 2 :=
by
  -- The proof details will be filled in here.
  sorry

end NUMINAMATH_GPT_common_ratio_is_half_l953_95338


namespace NUMINAMATH_GPT_sqrt_meaningful_l953_95394

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_l953_95394


namespace NUMINAMATH_GPT_cricket_throwers_l953_95354

theorem cricket_throwers (T L R : ℕ) 
  (h1 : T + L + R = 55)
  (h2 : T + R = 49) 
  (h3 : L = (1/3) * (L + R))
  (h4 : R = (2/3) * (L + R)) :
  T = 37 :=
by sorry

end NUMINAMATH_GPT_cricket_throwers_l953_95354


namespace NUMINAMATH_GPT_manuscript_fee_tax_l953_95337

theorem manuscript_fee_tax (fee : ℕ) (tax_paid : ℕ) :
  (tax_paid = 0 ∧ fee ≤ 800) ∨ 
  (tax_paid = (14 * (fee - 800) / 100) ∧ 800 < fee ∧ fee ≤ 4000) ∨ 
  (tax_paid = 11 * fee / 100 ∧ fee > 4000) →
  tax_paid = 420 →
  fee = 3800 :=
by 
  intro h_eq h_tax;
  sorry

end NUMINAMATH_GPT_manuscript_fee_tax_l953_95337


namespace NUMINAMATH_GPT_power_of_four_l953_95374

-- Definition of the conditions
def prime_factors (x: ℕ): ℕ := 2 * x + 5 + 2

-- The statement we need to prove given the conditions
theorem power_of_four (x: ℕ) (h: prime_factors x = 33) : x = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_power_of_four_l953_95374


namespace NUMINAMATH_GPT_medals_award_count_l953_95306

theorem medals_award_count :
  let total_ways (n k : ℕ) := n.factorial / (n - k).factorial
  ∃ (award_ways : ℕ), 
    let no_americans := total_ways 6 3
    let one_american := 4 * 3 * total_ways 6 2
    award_ways = no_americans + one_american ∧
    award_ways = 480 :=
by
  sorry

end NUMINAMATH_GPT_medals_award_count_l953_95306


namespace NUMINAMATH_GPT_gcd_90_150_l953_95367

theorem gcd_90_150 : Int.gcd 90 150 = 30 := 
by sorry

end NUMINAMATH_GPT_gcd_90_150_l953_95367


namespace NUMINAMATH_GPT_geom_progression_contra_l953_95384

theorem geom_progression_contra (q : ℝ) (p n : ℕ) (hp : p > 0) (hn : n > 0) :
  (11 = 10 * q^p) → (12 = 10 * q^n) → False :=
by
  -- proof steps should follow here
  sorry

end NUMINAMATH_GPT_geom_progression_contra_l953_95384


namespace NUMINAMATH_GPT_zero_and_one_positions_l953_95381

theorem zero_and_one_positions (a : ℝ) :
    (0 = (a + (-a)) / 2) ∧ (1 = ((a + (-a)) / 2 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_zero_and_one_positions_l953_95381


namespace NUMINAMATH_GPT_eval_diamond_expr_l953_95318

def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4
  | (1, 2) => 3
  | (1, 3) => 2
  | (1, 4) => 1
  | (2, 1) => 1
  | (2, 2) => 4
  | (2, 3) => 3
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 1
  | (3, 3) => 4
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 1
  | (4, 4) => 4
  | (_, _) => 0  -- This handles any case outside of 1,2,3,4 which should ideally not happen

theorem eval_diamond_expr : diamond (diamond 3 4) (diamond 2 1) = 2 := by
  sorry

end NUMINAMATH_GPT_eval_diamond_expr_l953_95318


namespace NUMINAMATH_GPT_find_triangle_angles_l953_95348

theorem find_triangle_angles 
  (α β γ : ℝ)
  (a b : ℝ)
  (h1 : γ = 2 * α)
  (h2 : b = 2 * a)
  (h3 : α + β + γ = 180) :
  α = 30 ∧ β = 90 ∧ γ = 60 := 
by 
  sorry

end NUMINAMATH_GPT_find_triangle_angles_l953_95348


namespace NUMINAMATH_GPT_time_for_P_to_finish_job_alone_l953_95375

variable (T : ℝ)

theorem time_for_P_to_finish_job_alone (h1 : 0 < T) (h2 : 3 * (1 / T + 1 / 20) + 0.4 * (1 / T) = 1) : T = 4 :=
by
  sorry

end NUMINAMATH_GPT_time_for_P_to_finish_job_alone_l953_95375


namespace NUMINAMATH_GPT_at_least_one_not_less_than_2_l953_95336

-- Definitions for the problem
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The Lean 4 statement for the problem
theorem at_least_one_not_less_than_2 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (2 ≤ a + 1/b) ∨ (2 ≤ b + 1/c) ∨ (2 ≤ c + 1/a) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_2_l953_95336


namespace NUMINAMATH_GPT_find_angle_A_l953_95376

theorem find_angle_A (a b : ℝ) (B A : ℝ)
  (h1 : a = 2) 
  (h2 : b = Real.sqrt 3) 
  (h3 : B = Real.pi / 3) : 
  A = Real.pi / 2 := 
sorry

end NUMINAMATH_GPT_find_angle_A_l953_95376


namespace NUMINAMATH_GPT_range_of_m_l953_95319

-- Define the variables and main theorem
theorem range_of_m (m : ℝ) (a b c : ℝ) 
  (h₀ : a = 3) (h₁ : b = (1 - 2 * m)) (h₂ : c = 8)
  : -5 < m ∧ m < -2 :=
by
  -- Given that a, b, and c are sides of a triangle, we use the triangle inequality theorem
  -- This code will remain as a placeholder of that proof
  sorry

end NUMINAMATH_GPT_range_of_m_l953_95319


namespace NUMINAMATH_GPT_max_value_f_l953_95329

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_f : ∃ x : ℝ, (f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) :=
sorry

end NUMINAMATH_GPT_max_value_f_l953_95329


namespace NUMINAMATH_GPT_paintings_in_four_weeks_l953_95377

theorem paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → num_weeks = 4 → 
  (hours_per_week / hours_per_painting) * num_weeks = 40 :=
by
  -- Sorry is used since we are not providing the proof
  sorry

end NUMINAMATH_GPT_paintings_in_four_weeks_l953_95377


namespace NUMINAMATH_GPT_percent_of_a_is_4b_l953_95344

theorem percent_of_a_is_4b (b : ℝ) (a : ℝ) (h : a = 1.8 * b) : (4 * b / a) * 100 = 222.22 := 
by {
  sorry
}

end NUMINAMATH_GPT_percent_of_a_is_4b_l953_95344


namespace NUMINAMATH_GPT_jelly_price_l953_95397

theorem jelly_price (d1 h1 d2 h2 : ℝ) (P1 : ℝ)
    (hd1 : d1 = 2) (hh1 : h1 = 5) (hd2 : d2 = 4) (hh2 : h2 = 8) (P1_cond : P1 = 0.75) :
    ∃ P2 : ℝ, P2 = 2.40 :=
by
  sorry

end NUMINAMATH_GPT_jelly_price_l953_95397


namespace NUMINAMATH_GPT_probJackAndJillChosen_l953_95359

-- Define the probabilities of each worker being chosen
def probJack : ℝ := 0.20
def probJill : ℝ := 0.15

-- Define the probability that Jack and Jill are both chosen
def probJackAndJill : ℝ := probJack * probJill

-- Theorem stating the probability that Jack and Jill are both chosen
theorem probJackAndJillChosen : probJackAndJill = 0.03 := 
by
  -- Replace this sorry with the complete proof
  sorry

end NUMINAMATH_GPT_probJackAndJillChosen_l953_95359


namespace NUMINAMATH_GPT_number_of_workers_is_25_l953_95364

noncomputable def original_workers (W : ℕ) :=
  W * 35 = (W + 10) * 25

theorem number_of_workers_is_25 : ∃ W, original_workers W ∧ W = 25 :=
by
  use 25
  unfold original_workers
  sorry

end NUMINAMATH_GPT_number_of_workers_is_25_l953_95364


namespace NUMINAMATH_GPT_smallest_positive_integer_x_l953_95305

-- Definitions based on the conditions given
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the problem
theorem smallest_positive_integer_x (x : ℕ) :
  (is_multiple (900 * x) 640) → x = 32 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_l953_95305


namespace NUMINAMATH_GPT_gcd_pow_sub_l953_95310

theorem gcd_pow_sub (h1001 h1012 : ℕ) (h : 1001 ≤ 1012) : 
  (Nat.gcd (2 ^ 1001 - 1) (2 ^ 1012 - 1)) = 2047 := sorry

end NUMINAMATH_GPT_gcd_pow_sub_l953_95310


namespace NUMINAMATH_GPT_find_T_l953_95368

theorem find_T (T : ℝ) : (1 / 2) * (1 / 7) * T = (1 / 3) * (1 / 5) * 90 → T = 84 :=
by sorry

end NUMINAMATH_GPT_find_T_l953_95368


namespace NUMINAMATH_GPT_central_angle_of_sector_l953_95355

variable (A : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions: A is the area of the sector, and r is the radius.
def is_sector (A : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  A = (1 / 2) * α * r^2

-- Proof that the central angle α given the conditions is 3π/4.
theorem central_angle_of_sector (h1 : is_sector (3 * Real.pi / 8) 1 α) : 
  α = 3 * Real.pi / 4 := 
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l953_95355


namespace NUMINAMATH_GPT_perfect_cubes_between_100_and_900_l953_95370

theorem perfect_cubes_between_100_and_900:
  ∃ n, n = 5 ∧ (∀ k, (k ≥ 5 ∧ k ≤ 9) → (k^3 ≥ 100 ∧ k^3 ≤ 900)) :=
by
  sorry

end NUMINAMATH_GPT_perfect_cubes_between_100_and_900_l953_95370


namespace NUMINAMATH_GPT_sum_of_two_digit_odd_numbers_l953_95380

-- Define the set of all two-digit numbers with both digits odd
def two_digit_odd_numbers : List ℕ := 
  [11, 13, 15, 17, 19, 31, 33, 35, 37, 39,
   51, 53, 55, 57, 59, 71, 73, 75, 77, 79,
   91, 93, 95, 97, 99]

-- Define a function to compute the sum of elements in a list
def list_sum (l : List ℕ) : ℕ := l.foldl (.+.) 0

theorem sum_of_two_digit_odd_numbers :
  list_sum two_digit_odd_numbers = 1375 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_digit_odd_numbers_l953_95380


namespace NUMINAMATH_GPT_log_three_twenty_seven_sqrt_three_l953_95398

noncomputable def twenty_seven : ℝ := 27
noncomputable def sqrt_three : ℝ := Real.sqrt 3

theorem log_three_twenty_seven_sqrt_three :
  Real.logb 3 (twenty_seven * sqrt_three) = 7 / 2 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_log_three_twenty_seven_sqrt_three_l953_95398


namespace NUMINAMATH_GPT_parallel_lines_slope_l953_95361

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 2 = 0 → ax * x - 8 * y - 3 = 0 → a = -6) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l953_95361


namespace NUMINAMATH_GPT_variance_of_sample_l953_95330

theorem variance_of_sample
  (x : ℝ)
  (h : (2 + 3 + x + 6 + 8) / 5 = 5) : 
  (1 / 5) * ((2 - 5) ^ 2 + (3 - 5) ^ 2 + (x - 5) ^ 2 + (6 - 5) ^ 2 + (8 - 5) ^ 2) = 24 / 5 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_sample_l953_95330


namespace NUMINAMATH_GPT_equation_has_solution_iff_l953_95345

open Real

theorem equation_has_solution_iff (a : ℝ) : 
  (∃ x : ℝ, (1/3)^|x| + a - 1 = 0) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_equation_has_solution_iff_l953_95345


namespace NUMINAMATH_GPT_dog_roaming_area_l953_95357

theorem dog_roaming_area :
  let shed_radius := 20
  let rope_length := 10
  let distance_from_edge := 10
  let radius_from_center := shed_radius - distance_from_edge
  radius_from_center = rope_length →
  (π * rope_length^2 = 100 * π) :=
by
  intros shed_radius rope_length distance_from_edge radius_from_center h
  sorry

end NUMINAMATH_GPT_dog_roaming_area_l953_95357


namespace NUMINAMATH_GPT_polynomial_degree_add_sub_l953_95379

noncomputable def degree (p : Polynomial ℂ) : ℕ := 
p.natDegree

variable (M N : Polynomial ℂ)

def is_fifth_degree (M : Polynomial ℂ) : Prop :=
degree M = 5

def is_third_degree (N : Polynomial ℂ) : Prop :=
degree N = 3

theorem polynomial_degree_add_sub (hM : is_fifth_degree M) (hN : is_third_degree N) :
  degree (M + N) = 5 ∧ degree (M - N) = 5 :=
by sorry

end NUMINAMATH_GPT_polynomial_degree_add_sub_l953_95379


namespace NUMINAMATH_GPT_frac_nonneg_iff_pos_l953_95349

theorem frac_nonneg_iff_pos (x : ℝ) : (2 / x ≥ 0) ↔ (x > 0) :=
by sorry

end NUMINAMATH_GPT_frac_nonneg_iff_pos_l953_95349


namespace NUMINAMATH_GPT_flour_per_new_base_is_one_fifth_l953_95387

def total_flour : ℚ := 40 * (1 / 8)

def flour_per_new_base (p : ℚ) (total_flour : ℚ) : ℚ := total_flour / p

theorem flour_per_new_base_is_one_fifth :
  flour_per_new_base 25 total_flour = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_flour_per_new_base_is_one_fifth_l953_95387


namespace NUMINAMATH_GPT_min_value_2x_y_l953_95300

noncomputable def min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (heq : Real.log (x + 2 * y) = Real.log x + Real.log y) : ℝ :=
  2 * x + y

theorem min_value_2x_y : ∀ (x y : ℝ), 0 < x → 0 < y → Real.log (x + 2 * y) = Real.log x + Real.log y → 2 * x + y ≥ 9 :=
by
  intros x y hx hy heq
  sorry

end NUMINAMATH_GPT_min_value_2x_y_l953_95300


namespace NUMINAMATH_GPT_pages_copied_l953_95313

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end NUMINAMATH_GPT_pages_copied_l953_95313


namespace NUMINAMATH_GPT_elf_distribution_finite_l953_95312

theorem elf_distribution_finite (infinite_rubies : ℕ → ℕ) (infinite_sapphires : ℕ → ℕ) :
  (∃ n : ℕ, ∀ i j : ℕ, i < n → j < n → (infinite_rubies i > infinite_rubies j → infinite_sapphires i < infinite_sapphires j) ∧
  (infinite_rubies i ≥ infinite_rubies j → infinite_sapphires i < infinite_sapphires j)) ↔
  ∃ k : ℕ, ∀ j : ℕ, j < k :=
sorry

end NUMINAMATH_GPT_elf_distribution_finite_l953_95312


namespace NUMINAMATH_GPT_problem1_problem2_problem2_zero_problem2_neg_l953_95332

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + a*x + a
def g (a x : ℝ) : ℝ := a*(f a x) - a^2*(x + 1) - 2*x

-- Problem 1
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f a x1 - x1 = 0 ∧ f a x2 - x2 = 0) →
  (0 < a ∧ a < 3 - 2*Real.sqrt 2) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ 
    if a < 1 then a-2 
    else -1/a) :=
sorry

theorem problem2_zero (h2 : a = 0) : 
  g a 1 = -2 :=
sorry

theorem problem2_neg (a : ℝ) (h3 : a < 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ a - 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem2_zero_problem2_neg_l953_95332


namespace NUMINAMATH_GPT_circle_radius_is_six_l953_95304

open Real

theorem circle_radius_is_six
  (r : ℝ)
  (h : 2 * 3 * 2 * π * r = 2 * π * r^2) :
  r = 6 := sorry

end NUMINAMATH_GPT_circle_radius_is_six_l953_95304


namespace NUMINAMATH_GPT_rate_of_discount_l953_95382

theorem rate_of_discount (marked_price selling_price : ℝ) (h1 : marked_price = 200) (h2 : selling_price = 120) : 
  ((marked_price - selling_price) / marked_price) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_discount_l953_95382


namespace NUMINAMATH_GPT_unique_solution_c_value_l953_95352

-- Define the main problem: the parameter c for which a given system of equations has a unique solution.
theorem unique_solution_c_value (c : ℝ) : 
  (∀ x y : ℝ, 2 * abs (x + 7) + abs (y - 4) = c ∧ abs (x + 4) + 2 * abs (y - 7) = c → 
   (x = -7 ∧ y = 7)) ↔ c = 3 :=
by sorry

end NUMINAMATH_GPT_unique_solution_c_value_l953_95352


namespace NUMINAMATH_GPT_find_angle_x_l953_95311

theorem find_angle_x (x : ℝ) (h1 : x + x + 140 = 360) : x = 110 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_x_l953_95311


namespace NUMINAMATH_GPT_second_pipe_fill_time_l953_95317

theorem second_pipe_fill_time :
  ∃ x : ℝ, x ≠ 0 ∧ (1 / 10 + 1 / x - 1 / 20 = 1 / 7.5) ∧ x = 60 :=
by
  sorry

end NUMINAMATH_GPT_second_pipe_fill_time_l953_95317


namespace NUMINAMATH_GPT_number_of_females_l953_95399

theorem number_of_females 
  (total_students : ℕ) 
  (sampled_students : ℕ) 
  (sampled_female_less_than_male : ℕ) 
  (h_total : total_students = 1600)
  (h_sample : sampled_students = 200)
  (h_diff : sampled_female_less_than_male = 20) : 
  ∃ F M : ℕ, F + M = total_students ∧ (F / M : ℝ) = 9 / 11 ∧ F = 720 :=
by
  sorry

end NUMINAMATH_GPT_number_of_females_l953_95399


namespace NUMINAMATH_GPT_polynomial_equality_l953_95366

theorem polynomial_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_equality_l953_95366


namespace NUMINAMATH_GPT_solve_inequality_l953_95340

theorem solve_inequality {x : ℝ} :
  (3 / (5 - 3 * x) > 1) ↔ (2/3 < x ∧ x < 5/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l953_95340


namespace NUMINAMATH_GPT_pizza_slices_needed_l953_95342

theorem pizza_slices_needed (couple_slices : ℕ) (children : ℕ) (children_slices : ℕ) (pizza_slices : ℕ)
    (hc : couple_slices = 3)
    (hcouple : children = 6)
    (hch : children_slices = 1)
    (hpizza : pizza_slices = 4) : 
    (2 * couple_slices + children * children_slices) / pizza_slices = 3 := 
by
    sorry

end NUMINAMATH_GPT_pizza_slices_needed_l953_95342


namespace NUMINAMATH_GPT_solve_for_x_l953_95363

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l953_95363


namespace NUMINAMATH_GPT_find_cost_price_l953_95341

variables (SP CP : ℝ)
variables (discount profit : ℝ)
variable (h1 : SP = 24000)
variable (h2 : discount = 0.10)
variable (h3 : profit = 0.08)

theorem find_cost_price 
  (h1 : SP = 24000)
  (h2 : discount = 0.10)
  (h3 : profit = 0.08)
  (h4 : SP * (1 - discount) = CP * (1 + profit)) :
  CP = 20000 := 
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l953_95341


namespace NUMINAMATH_GPT_father_l953_95385

theorem father's_age (M F : ℕ) 
  (h1 : M = (2 / 5 : ℝ) * F)
  (h2 : M + 14 = (1 / 2 : ℝ) * (F + 14)) : 
  F = 70 := 
  sorry

end NUMINAMATH_GPT_father_l953_95385


namespace NUMINAMATH_GPT_regular_polygon_sides_l953_95389

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l953_95389


namespace NUMINAMATH_GPT_least_number_to_add_l953_95350

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) (k : ℕ) (l : ℕ) (h₁ : n = 1077) (h₂ : d = 23) (h₃ : n % d = r) (h₄ : d - r = k) (h₅ : r = 19) (h₆ : k = l) : l = 4 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l953_95350


namespace NUMINAMATH_GPT_bus_avg_speed_l953_95309

noncomputable def average_speed_of_bus 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) :
  ℕ :=
  (initial_distance_behind + bicycle_speed * catch_up_time) / catch_up_time

theorem bus_avg_speed 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) 
  (h_bicycle_speed : bicycle_speed = 15) 
  (h_initial_distance_behind : initial_distance_behind = 195)
  (h_catch_up_time : catch_up_time = 3) :
  average_speed_of_bus bicycle_speed initial_distance_behind catch_up_time = 80 :=
by
  sorry

end NUMINAMATH_GPT_bus_avg_speed_l953_95309


namespace NUMINAMATH_GPT_daily_sales_change_l953_95343

theorem daily_sales_change
    (mon_sales : ℕ)
    (week_total_sales : ℕ)
    (days_in_week : ℕ)
    (avg_sales_per_day : ℕ)
    (other_days_total_sales : ℕ)
    (x : ℕ)
    (h1 : days_in_week = 7)
    (h2 : avg_sales_per_day = 5)
    (h3 : week_total_sales = avg_sales_per_day * days_in_week)
    (h4 : mon_sales = 2)
    (h5 : week_total_sales = mon_sales + other_days_total_sales)
    (h6 : other_days_total_sales = 33)
    (h7 : 2 + x + 2 + 2*x + 2 + 3*x + 2 + 4*x + 2 + 5*x + 2 + 6*x = other_days_total_sales) : 
  x = 1 :=
by
sorry

end NUMINAMATH_GPT_daily_sales_change_l953_95343


namespace NUMINAMATH_GPT_total_seats_l953_95392

theorem total_seats (F : ℕ) 
  (h1 : 305 = 4 * F + 2) 
  (h2 : 310 = 4 * F + 2) : 
  310 + F = 387 :=
by
  sorry

end NUMINAMATH_GPT_total_seats_l953_95392


namespace NUMINAMATH_GPT_blue_balls_count_l953_95327

theorem blue_balls_count:
  ∀ (T : ℕ),
  (1/4 * T) + (1/8 * T) + (1/12 * T) + 26 = T → 
  (1 / 8) * T = 6 := by
  intros T h
  sorry

end NUMINAMATH_GPT_blue_balls_count_l953_95327


namespace NUMINAMATH_GPT_dot_product_correct_l953_95353

theorem dot_product_correct:
  let a : ℝ × ℝ := (5, -7)
  let b : ℝ × ℝ := (-6, -4)
  (a.1 * b.1) + (a.2 * b.2) = -2 := by
sorry

end NUMINAMATH_GPT_dot_product_correct_l953_95353


namespace NUMINAMATH_GPT_cans_purchased_l953_95371

theorem cans_purchased (S Q E : ℕ) (hQ : Q ≠ 0) :
  (∃ x : ℕ, x = (5 * S * E) / Q) := by
  sorry

end NUMINAMATH_GPT_cans_purchased_l953_95371


namespace NUMINAMATH_GPT_cubic_sum_identity_l953_95303

variables (x y z : ℝ)

theorem cubic_sum_identity (h1 : x + y + z = 10) (h2 : xy + xz + yz = 30) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 100 :=
sorry

end NUMINAMATH_GPT_cubic_sum_identity_l953_95303


namespace NUMINAMATH_GPT_powers_of_i_l953_95335

theorem powers_of_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i^22 + i^222 = -2 :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_powers_of_i_l953_95335


namespace NUMINAMATH_GPT_missed_angle_l953_95302

theorem missed_angle (n : ℕ) (h1 : (n - 2) * 180 ≥ 3239) (h2 : n ≥ 3) : 3240 - 3239 = 1 :=
by
  sorry

end NUMINAMATH_GPT_missed_angle_l953_95302


namespace NUMINAMATH_GPT_fraction_addition_simplest_form_l953_95328

theorem fraction_addition_simplest_form :
  (7 / 8) + (3 / 5) = 59 / 40 :=
by sorry

end NUMINAMATH_GPT_fraction_addition_simplest_form_l953_95328


namespace NUMINAMATH_GPT_mohit_discount_l953_95390

variable (SP : ℝ) -- Selling price
variable (CP : ℝ) -- Cost price
variable (discount_percentage : ℝ) -- Discount percentage

-- Conditions
axiom h1 : SP = 21000
axiom h2 : CP = 17500
axiom h3 : discount_percentage = ( (SP - (CP + 0.08 * CP)) / SP) * 100

-- Theorem to prove
theorem mohit_discount : discount_percentage = 10 :=
  sorry

end NUMINAMATH_GPT_mohit_discount_l953_95390


namespace NUMINAMATH_GPT_father_age_when_rachel_is_25_l953_95314

-- Definitions for Rachel's age, Grandfather's age, Mother's age, and Father's age
def rachel_age : ℕ := 12
def grandfather_age : ℕ := 7 * rachel_age
def mother_age : ℕ := grandfather_age / 2
def father_age : ℕ := mother_age + 5
def years_until_rachel_is_25 : ℕ := 25 - rachel_age
def fathers_age_when_rachel_is_25 : ℕ := father_age + years_until_rachel_is_25

-- Theorem to prove that Rachel's father will be 60 years old when Rachel is 25 years old
theorem father_age_when_rachel_is_25 : fathers_age_when_rachel_is_25 = 60 := by
  sorry

end NUMINAMATH_GPT_father_age_when_rachel_is_25_l953_95314


namespace NUMINAMATH_GPT_solutionToSystemOfEquations_solutionToSystemOfInequalities_l953_95378

open Classical

noncomputable def solveSystemOfEquations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solutionToSystemOfEquations : ∃ (x y : ℝ), solveSystemOfEquations x y ∧ x = 4 ∧ y = 5 := by
  sorry

def solveSystemOfInequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solutionToSystemOfInequalities : ∃ x : ℝ, solveSystemOfInequalities x ∧ 1 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_GPT_solutionToSystemOfEquations_solutionToSystemOfInequalities_l953_95378


namespace NUMINAMATH_GPT_range_of_m_l953_95395

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l953_95395


namespace NUMINAMATH_GPT_numbers_not_crossed_out_l953_95316

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

end NUMINAMATH_GPT_numbers_not_crossed_out_l953_95316


namespace NUMINAMATH_GPT_total_doors_needed_correct_l953_95369

-- Define the conditions
def buildings : ℕ := 2
def floors_per_building : ℕ := 12
def apartments_per_floor : ℕ := 6
def doors_per_apartment : ℕ := 7

-- Define the total number of doors needed
def total_doors_needed : ℕ := buildings * floors_per_building * apartments_per_floor * doors_per_apartment

-- State the theorem to prove the total number of doors needed is 1008
theorem total_doors_needed_correct : total_doors_needed = 1008 := by
  sorry

end NUMINAMATH_GPT_total_doors_needed_correct_l953_95369


namespace NUMINAMATH_GPT_total_amount_paid_is_correct_l953_95339

-- Define the initial conditions
def tireA_price : ℕ := 75
def tireA_discount : ℕ := 20
def tireB_price : ℕ := 90
def tireB_discount : ℕ := 30
def tireC_price : ℕ := 120
def tireC_discount : ℕ := 45
def tireD_price : ℕ := 150
def tireD_discount : ℕ := 60
def installation_fee : ℕ := 15
def disposal_fee : ℕ := 5

-- Calculate the total amount paid
def total_paid : ℕ :=
  let tireA_total := (tireA_price - tireA_discount) + installation_fee + disposal_fee
  let tireB_total := (tireB_price - tireB_discount) + installation_fee + disposal_fee
  let tireC_total := (tireC_price - tireC_discount) + installation_fee + disposal_fee
  let tireD_total := (tireD_price - tireD_discount) + installation_fee + disposal_fee
  tireA_total + tireB_total + tireC_total + tireD_total

-- Statement of the theorem
theorem total_amount_paid_is_correct :
  total_paid = 360 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_amount_paid_is_correct_l953_95339


namespace NUMINAMATH_GPT_percentage_of_customers_purchased_l953_95301

theorem percentage_of_customers_purchased (ad_cost : ℕ) (customers : ℕ) (price_per_sale : ℕ) (profit : ℕ)
  (h1 : ad_cost = 1000)
  (h2 : customers = 100)
  (h3 : price_per_sale = 25)
  (h4 : profit = 1000) :
  (profit / price_per_sale / customers) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_customers_purchased_l953_95301


namespace NUMINAMATH_GPT_square_area_l953_95360

theorem square_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  ∃ (s : ℝ), (s * Real.sqrt 2 = d) ∧ (s^2 = 144) := by
  sorry

end NUMINAMATH_GPT_square_area_l953_95360


namespace NUMINAMATH_GPT_P_Q_sum_l953_95307

noncomputable def find_P_Q_sum (P Q : ℚ) : Prop :=
  ∀ x : ℚ, (x^2 + 3 * x + 7) * (x^2 + (51/7) * x - 2) = x^4 + P * x^3 + Q * x^2 + 45 * x - 14

theorem P_Q_sum :
  ∃ P Q : ℚ, find_P_Q_sum P Q ∧ (P + Q = 260 / 7) :=
by
  sorry

end NUMINAMATH_GPT_P_Q_sum_l953_95307


namespace NUMINAMATH_GPT_pedro_squares_correct_l953_95347

def squares_jesus : ℕ := 60
def squares_linden : ℕ := 75
def squares_pedro (s_jesus s_linden : ℕ) : ℕ := (s_jesus + s_linden) + 65

theorem pedro_squares_correct :
  squares_pedro squares_jesus squares_linden = 200 :=
by
  sorry

end NUMINAMATH_GPT_pedro_squares_correct_l953_95347


namespace NUMINAMATH_GPT_pencils_combined_length_l953_95308

theorem pencils_combined_length (length_pencil1 length_pencil2 : Nat) (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) :
  length_pencil1 + length_pencil2 = 24 := by
  sorry

end NUMINAMATH_GPT_pencils_combined_length_l953_95308


namespace NUMINAMATH_GPT_sum_of_coefficients_correct_l953_95388

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (x + 3 * y) ^ 17

-- Define the sum of coefficients by substituting x = 1 and y = 1
def sum_of_coefficients : ℤ := polynomial 1 1

-- Statement of the mathematical proof problem
theorem sum_of_coefficients_correct :
  sum_of_coefficients = 17179869184 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_correct_l953_95388


namespace NUMINAMATH_GPT_quadratic_condition_l953_95386

theorem quadratic_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1^2 - x2^2 = c^2 / a^2) ↔
  b^4 - c^4 = 4 * a * b^2 * c :=
sorry

end NUMINAMATH_GPT_quadratic_condition_l953_95386


namespace NUMINAMATH_GPT_positive_integer_solutions_l953_95362

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end NUMINAMATH_GPT_positive_integer_solutions_l953_95362


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l953_95351

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 < x ∧ x < 2) : x < 2 ∧ ∀ y, (y < 2 → y ≤ 1 ∨ y ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l953_95351


namespace NUMINAMATH_GPT_calculate_yield_l953_95346

-- Define the conditions
def x := 6
def x_pos := 3
def x_tot := 3 * x
def nuts_x_pos := x + x_pos
def nuts_x := x
def nuts_x_neg := x - x_pos
def yield_x_pos := 60
def yield_x := 120
def avg_yield := 100

-- Calculate yields
def nuts_x_pos_yield : ℕ := nuts_x_pos * yield_x_pos
def nuts_x_yield : ℕ := nuts_x * yield_x
noncomputable def total_yield (yield_x_neg : ℕ) : ℕ :=
  nuts_x_pos_yield + nuts_x_yield + nuts_x_neg * yield_x_neg

-- Equation combining all
lemma yield_per_tree : (total_yield Y) / x_tot = avg_yield := sorry

-- Prove Y = 180
theorem calculate_yield : (x = 6 → ((nuts_x_neg * 180 = 540) ∧ rate = 180)) := sorry

end NUMINAMATH_GPT_calculate_yield_l953_95346


namespace NUMINAMATH_GPT_dongzhi_daylight_hours_l953_95325

theorem dongzhi_daylight_hours:
  let total_hours_in_day := 24
  let daytime_ratio := 5
  let nighttime_ratio := 7
  let total_parts := daytime_ratio + nighttime_ratio
  let daylight_hours := total_hours_in_day * daytime_ratio / total_parts
  daylight_hours = 10 :=
by
  sorry

end NUMINAMATH_GPT_dongzhi_daylight_hours_l953_95325


namespace NUMINAMATH_GPT_seventy_five_inverse_mod_seventy_six_l953_95321

-- Lean 4 statement for the problem.
theorem seventy_five_inverse_mod_seventy_six : (75 : ℤ) * 75 % 76 = 1 :=
by
  sorry

end NUMINAMATH_GPT_seventy_five_inverse_mod_seventy_six_l953_95321


namespace NUMINAMATH_GPT_intersection_A_B_l953_95320

-- Define set A based on the given condition.
def setA : Set ℝ := {x | x^2 - 4 < 0}

-- Define set B based on the given condition.
def setB : Set ℝ := {x | x < 0}

-- Prove that the intersection of sets A and B is the given set.
theorem intersection_A_B : setA ∩ setB = {x | -2 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l953_95320


namespace NUMINAMATH_GPT_inscribed_circle_radius_in_quadrilateral_pyramid_l953_95356

theorem inscribed_circle_radius_in_quadrilateral_pyramid
  (a : ℝ) (α : ℝ)
  (h_pos : 0 < a) (h_α : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r = a * Real.sqrt 2 / (1 + 2 * Real.cos α + Real.sqrt (4 * Real.cos α ^ 2 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_in_quadrilateral_pyramid_l953_95356


namespace NUMINAMATH_GPT_hexagon_diagonals_l953_95383

-- Define a hexagon as having 6 vertices
def hexagon_vertices : ℕ := 6

-- From one vertex of a hexagon, there are (6 - 1) vertices it can potentially connect to
def potential_connections (vertices : ℕ) : ℕ := vertices - 1

-- Remove the two adjacent vertices to count diagonals
def diagonals_from_vertex (connections : ℕ) : ℕ := connections - 2

theorem hexagon_diagonals : diagonals_from_vertex (potential_connections hexagon_vertices) = 3 := by
  -- The proof is intentionally left as a sorry placeholder.
  sorry

end NUMINAMATH_GPT_hexagon_diagonals_l953_95383


namespace NUMINAMATH_GPT_estimate_mass_of_ice_floe_l953_95334

noncomputable def mass_of_ice_floe (d : ℝ) (D : ℝ) (m : ℝ) : ℝ :=
  (m * d) / (D - d)

theorem estimate_mass_of_ice_floe :
  mass_of_ice_floe 9.5 10 600 = 11400 := 
by
  sorry

end NUMINAMATH_GPT_estimate_mass_of_ice_floe_l953_95334


namespace NUMINAMATH_GPT_intersection_of_multiples_of_2_l953_95396

theorem intersection_of_multiples_of_2 : 
  let M := {1, 2, 4, 8}
  let N := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  M ∩ N = {2, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_multiples_of_2_l953_95396


namespace NUMINAMATH_GPT_cos_two_pi_over_three_l953_95331

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_two_pi_over_three_l953_95331


namespace NUMINAMATH_GPT_constant_term_of_expansion_l953_95365

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the expansion term
def expansion_term (n k : ℕ) (a b : ℚ) : ℚ :=
  (binom n k) * (a ^ k) * (b ^ (n - k))

-- Define the specific example
def specific_expansion_term : ℚ :=
  expansion_term 8 4 3 (2 : ℚ)

theorem constant_term_of_expansion : specific_expansion_term = 90720 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_constant_term_of_expansion_l953_95365


namespace NUMINAMATH_GPT_intersection_of_sets_l953_95393

def setA : Set ℝ := {x | x^2 ≤ 4 * x}
def setB : Set ℝ := {x | x < 1}

theorem intersection_of_sets : setA ∩ setB = {x | x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l953_95393


namespace NUMINAMATH_GPT_savings_amount_l953_95326

-- Define the conditions for Celia's spending
def food_spending_per_week : ℝ := 100
def weeks : ℕ := 4
def rent_spending : ℝ := 1500
def video_streaming_services_spending : ℝ := 30
def cell_phone_usage_spending : ℝ := 50
def savings_rate : ℝ := 0.10

-- Define the total spending calculation
def total_spending : ℝ :=
  food_spending_per_week * weeks + rent_spending + video_streaming_services_spending + cell_phone_usage_spending

-- Define the savings calculation
def savings : ℝ :=
  savings_rate * total_spending

-- Prove the amount of savings
theorem savings_amount : savings = 198 :=
by
  -- This is the statement that needs to be proven, hence adding a placeholder proof.
  sorry

end NUMINAMATH_GPT_savings_amount_l953_95326
