import Mathlib

namespace NUMINAMATH_GPT_how_many_years_later_will_tom_be_twice_tim_l2139_213997

-- Conditions
def toms_age := 15
def total_age := 21
def tims_age := total_age - toms_age

-- Define the problem statement
theorem how_many_years_later_will_tom_be_twice_tim (x : ℕ) 
  (h1 : toms_age + tims_age = total_age) 
  (h2 : toms_age = 15) 
  (h3 : ∀ y : ℕ, toms_age + y = 2 * (tims_age + y) ↔ y = x) : 
  x = 3 
:= sorry

end NUMINAMATH_GPT_how_many_years_later_will_tom_be_twice_tim_l2139_213997


namespace NUMINAMATH_GPT_base_addition_is_10_l2139_213987

-- The problem states that adding two numbers in a particular base results in a third number in the same base.
def valid_base_10_addition (n m k b : ℕ) : Prop :=
  let n_b := n / b^2 * b^2 + (n / b % b) * b + n % b
  let m_b := m / b^2 * b^2 + (m / b % b) * b + m % b
  let k_b := k / b^2 * b^2 + (k / b % b) * b + k % b
  n_b + m_b = k_b

theorem base_addition_is_10 : valid_base_10_addition 172 156 340 10 :=
  sorry

end NUMINAMATH_GPT_base_addition_is_10_l2139_213987


namespace NUMINAMATH_GPT_ratio_diff_squares_eq_16_l2139_213919

theorem ratio_diff_squares_eq_16 (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  (x^2 - y^2) / (x - y) = 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_diff_squares_eq_16_l2139_213919


namespace NUMINAMATH_GPT_mark_charged_more_hours_l2139_213962

variable {p k m : ℕ}

theorem mark_charged_more_hours (h1 : p + k + m = 216)
                                (h2 : p = 2 * k)
                                (h3 : p = m / 3) :
                                m - k = 120 :=
sorry

end NUMINAMATH_GPT_mark_charged_more_hours_l2139_213962


namespace NUMINAMATH_GPT_Kyle_older_than_Julian_l2139_213980

variable (Tyson_age : ℕ)
variable (Frederick_age Julian_age Kyle_age : ℕ)

-- Conditions
def condition1 := Tyson_age = 20
def condition2 := Frederick_age = 2 * Tyson_age
def condition3 := Julian_age = Frederick_age - 20
def condition4 := Kyle_age = 25

-- The proof problem (statement only)
theorem Kyle_older_than_Julian :
  Tyson_age = 20 ∧
  Frederick_age = 2 * Tyson_age ∧
  Julian_age = Frederick_age - 20 ∧
  Kyle_age = 25 →
  Kyle_age - Julian_age = 5 := by
  intro h
  sorry

end NUMINAMATH_GPT_Kyle_older_than_Julian_l2139_213980


namespace NUMINAMATH_GPT_hyperbola_distance_condition_l2139_213912

open Real

theorem hyperbola_distance_condition (a b c x: ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_dist : abs (b^4 / a^2 / (a - c)) < a + sqrt (a^2 + b^2)) :
    0 < b / a ∧ b / a < 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_distance_condition_l2139_213912


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l2139_213966

theorem inequality_for_positive_reals
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := 
sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l2139_213966


namespace NUMINAMATH_GPT_slope_tangent_at_point_l2139_213992

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem slope_tangent_at_point : (deriv f 1) = 1 := 
by
  sorry

end NUMINAMATH_GPT_slope_tangent_at_point_l2139_213992


namespace NUMINAMATH_GPT_area_of_ground_l2139_213921

def height_of_rain : ℝ := 0.05
def volume_of_water : ℝ := 750

theorem area_of_ground : ∃ A : ℝ, A = (volume_of_water / height_of_rain) ∧ A = 15000 := by
  sorry

end NUMINAMATH_GPT_area_of_ground_l2139_213921


namespace NUMINAMATH_GPT_geometric_sequence_arith_condition_l2139_213933

-- Definitions of geometric sequence and arithmetic sequence condition
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions: \( \{a_n\} \) is a geometric sequence with \( a_2 \), \( \frac{1}{2}a_3 \), \( a_1 \) forming an arithmetic sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop := a 2 = (1 / 2) * a 3 + a 1

-- Final theorem to prove
theorem geometric_sequence_arith_condition (hq : q^2 - q - 1 = 0) 
  (hgeo : is_geometric_sequence a q) 
  (harith : arithmetic_sequence_condition a) : 
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_arith_condition_l2139_213933


namespace NUMINAMATH_GPT_eighth_term_geometric_seq_l2139_213946

theorem eighth_term_geometric_seq (a1 a2 : ℚ) (a1_val : a1 = 3) (a2_val : a2 = 9 / 2) :
  (a1 * (a2 / a1)^(7) = 6561 / 128) :=
  by
    sorry

end NUMINAMATH_GPT_eighth_term_geometric_seq_l2139_213946


namespace NUMINAMATH_GPT_white_balls_in_bag_l2139_213968

theorem white_balls_in_bag:
  ∀ (total balls green yellow red purple : Nat),
  total = 60 →
  green = 18 →
  yellow = 8 →
  red = 5 →
  purple = 7 →
  (1 - 0.8) = (red + purple : ℚ) / total →
  (W + green + yellow = total - (red + purple : ℚ)) →
  W = 22 :=
by
  intros total balls green yellow red purple ht hg hy hr hp hprob heqn
  sorry

end NUMINAMATH_GPT_white_balls_in_bag_l2139_213968


namespace NUMINAMATH_GPT_median_price_l2139_213905

-- Definitions from conditions
def price1 : ℝ := 10
def price2 : ℝ := 12
def price3 : ℝ := 15

def sales1 : ℝ := 0.50
def sales2 : ℝ := 0.30
def sales3 : ℝ := 0.20

-- Statement of the problem
theorem median_price : (price1 * sales1 + price2 * sales2 + price3 * sales3) / 2 = 11 := by
  sorry

end NUMINAMATH_GPT_median_price_l2139_213905


namespace NUMINAMATH_GPT_number_of_points_marked_l2139_213958

theorem number_of_points_marked (a₁ a₂ b₁ b₂ : ℕ) 
  (h₁ : a₁ * a₂ = 50) (h₂ : b₁ * b₂ = 56) (h₃ : a₁ + a₂ = b₁ + b₂) : 
  (a₁ + a₂ + 1 = 16) :=
sorry

end NUMINAMATH_GPT_number_of_points_marked_l2139_213958


namespace NUMINAMATH_GPT_thirteen_power_1997_tens_digit_l2139_213939

def tens_digit (n : ℕ) := (n / 10) % 10

theorem thirteen_power_1997_tens_digit :
  tens_digit (13 ^ 1997 % 100) = 5 := by
  sorry

end NUMINAMATH_GPT_thirteen_power_1997_tens_digit_l2139_213939


namespace NUMINAMATH_GPT_balloon_altitude_l2139_213942

theorem balloon_altitude 
  (temp_diff_per_1000m : ℝ)
  (altitude_temp : ℝ) 
  (ground_temp : ℝ)
  (altitude : ℝ) 
  (h1 : temp_diff_per_1000m = 6) 
  (h2 : altitude_temp = -2)
  (h3 : ground_temp = 5) :
  altitude = 7/6 :=
by sorry

end NUMINAMATH_GPT_balloon_altitude_l2139_213942


namespace NUMINAMATH_GPT_inequality_holds_iff_l2139_213976

theorem inequality_holds_iff (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → x^2 + (a - 4) * x + 4 > 0) ↔ a > 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_iff_l2139_213976


namespace NUMINAMATH_GPT_sum_gcf_lcm_36_56_84_l2139_213945

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_gcf_lcm_36_56_84 :
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  gcf_36_56_84 + lcm_36_56_84 = 516 :=
by
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  show gcf_36_56_84 + lcm_36_56_84 = 516
  sorry

end NUMINAMATH_GPT_sum_gcf_lcm_36_56_84_l2139_213945


namespace NUMINAMATH_GPT_product_gcd_lcm_4000_l2139_213924

-- Definitions of gcd and lcm for the given numbers
def gcd_40_100 := Nat.gcd 40 100
def lcm_40_100 := Nat.lcm 40 100

-- Problem: Prove that the product of the gcd and lcm of 40 and 100 equals 4000
theorem product_gcd_lcm_4000 : gcd_40_100 * lcm_40_100 = 4000 := by
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_4000_l2139_213924


namespace NUMINAMATH_GPT_solve_linear_eq_l2139_213922

theorem solve_linear_eq (x y : ℤ) : 2 * x + 3 * y = 0 ↔ (x, y) = (3, -2) := sorry

end NUMINAMATH_GPT_solve_linear_eq_l2139_213922


namespace NUMINAMATH_GPT_amina_wins_is_21_over_32_l2139_213964

/--
Amina and Bert alternate turns tossing a fair coin. Amina goes first and each player takes three turns.
The first player to toss a tail wins. If neither Amina nor Bert tosses a tail, then neither wins.
Prove that the probability that Amina wins is \( \frac{21}{32} \).
-/
def amina_wins_probability : ℚ :=
  let p_first_turn := 1 / 2
  let p_second_turn := (1 / 2) ^ 3
  let p_third_turn := (1 / 2) ^ 5
  p_first_turn + p_second_turn + p_third_turn

theorem amina_wins_is_21_over_32 :
  amina_wins_probability = 21 / 32 :=
sorry

end NUMINAMATH_GPT_amina_wins_is_21_over_32_l2139_213964


namespace NUMINAMATH_GPT_inequality_S_l2139_213991

def S (n m : ℕ) : ℕ := sorry

theorem inequality_S (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n :=
sorry

end NUMINAMATH_GPT_inequality_S_l2139_213991


namespace NUMINAMATH_GPT_relay_race_arrangements_l2139_213977

noncomputable def number_of_arrangements (athletes : Finset ℕ) (a b : ℕ) : ℕ :=
  (athletes.erase a).card.factorial * ((athletes.erase b).card.factorial - 2) * (athletes.card.factorial / ((athletes.card - 4).factorial)) / 4

theorem relay_race_arrangements :
  let athletes := {0, 1, 2, 3, 4, 5}
  number_of_arrangements athletes 0 1 = 252 := 
by
  sorry

end NUMINAMATH_GPT_relay_race_arrangements_l2139_213977


namespace NUMINAMATH_GPT_difference_of_digits_l2139_213911

theorem difference_of_digits (p q : ℕ) (h1 : ∀ n, n < 100 → n ≥ 10 → ∀ m, m < 100 → m ≥ 10 → 9 * (p - q) = 9) : 
  p - q = 1 :=
sorry

end NUMINAMATH_GPT_difference_of_digits_l2139_213911


namespace NUMINAMATH_GPT_value_of_a7_minus_a8_l2139_213952

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem value_of_a7_minus_a8
  (h_seq: arithmetic_sequence a d)
  (h_sum: a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = d :=
sorry

end NUMINAMATH_GPT_value_of_a7_minus_a8_l2139_213952


namespace NUMINAMATH_GPT_washing_time_per_cycle_l2139_213948

theorem washing_time_per_cycle
    (shirts pants sweaters jeans : ℕ)
    (items_per_cycle total_hours : ℕ)
    (h1 : shirts = 18)
    (h2 : pants = 12)
    (h3 : sweaters = 17)
    (h4 : jeans = 13)
    (h5 : items_per_cycle = 15)
    (h6 : total_hours = 3) :
    ((shirts + pants + sweaters + jeans) / items_per_cycle) * (total_hours * 60) / ((shirts + pants + sweaters + jeans) / items_per_cycle) = 45 := 
by
  sorry

end NUMINAMATH_GPT_washing_time_per_cycle_l2139_213948


namespace NUMINAMATH_GPT_ratio_of_girls_with_long_hair_l2139_213954

theorem ratio_of_girls_with_long_hair (total_people boys girls short_hair long_hair : ℕ)
  (h1 : total_people = 55)
  (h2 : boys = 30)
  (h3 : girls = total_people - boys)
  (h4 : short_hair = 10)
  (h5 : long_hair = girls - short_hhair) :
  long_hair / gcd long_hair girls = 3 ∧ girls / gcd long_hair girls = 5 := 
by {
  -- This placeholder indicates where the proof should be.
  sorry
}

end NUMINAMATH_GPT_ratio_of_girls_with_long_hair_l2139_213954


namespace NUMINAMATH_GPT_total_notes_count_l2139_213986

theorem total_notes_count :
  ∀ (rows : ℕ) (notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ),
  rows = 5 →
  notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  (rows * notes_per_row + (rows * notes_per_row * blue_notes_per_red + additional_blue_notes)) = 100 := by
  intros rows notes_per_row blue_notes_per_red additional_blue_notes
  sorry

end NUMINAMATH_GPT_total_notes_count_l2139_213986


namespace NUMINAMATH_GPT_part1_part2_l2139_213927

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (a x : ℝ) := a * |x - 1|

theorem part1 (a : ℝ) :
  (∀ x : ℝ, |f x| = g a x → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2139_213927


namespace NUMINAMATH_GPT_daniel_fraction_l2139_213915

theorem daniel_fraction (A B C D : Type) (money : A → ℝ) 
  (adriano bruno cesar daniel : A)
  (h1 : money daniel = 0)
  (given_amount : ℝ)
  (h2 : money adriano = 5 * given_amount)
  (h3 : money bruno = 4 * given_amount)
  (h4 : money cesar = 3 * given_amount)
  (h5 : money daniel = (1 / 5) * money adriano + (1 / 4) * money bruno + (1 / 3) * money cesar) :
  money daniel / (money adriano + money bruno + money cesar) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_daniel_fraction_l2139_213915


namespace NUMINAMATH_GPT_range_of_a_l2139_213979

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 16) / x

theorem range_of_a (a : ℝ) (h1 : 2 ≤ a) (h2 : (∀ x, 2 ≤ x ∧ x ≤ a → 9 ≤ f x ∧ f x ≤ 11)) : 4 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2139_213979


namespace NUMINAMATH_GPT_product_of_numbers_l2139_213902

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l2139_213902


namespace NUMINAMATH_GPT_fermat_coprime_l2139_213928

theorem fermat_coprime (m n : ℕ) (hmn : m ≠ n) (hm_pos : m > 0) (hn_pos : n > 0) :
  gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 :=
sorry

end NUMINAMATH_GPT_fermat_coprime_l2139_213928


namespace NUMINAMATH_GPT_opposite_of_negative_six_is_six_l2139_213983

theorem opposite_of_negative_six_is_six : ∀ (x : ℤ), (-6 + x = 0) → x = 6 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_opposite_of_negative_six_is_six_l2139_213983


namespace NUMINAMATH_GPT_problem1_problem2_l2139_213904

variable {n : ℕ}
variable {a b : ℝ}

-- Part 1
theorem problem1 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(2 * n^2 / (n + 2) - n * a) - b| < ε) :
  a = 2 ∧ b = 4 := sorry

-- Part 2
theorem problem2 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(3^n / (3^(n + 1) + (a + 1)^n) - 1/3)| < ε) :
  -4 < a ∧ a < 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l2139_213904


namespace NUMINAMATH_GPT_smallest_sum_minimum_l2139_213917

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end NUMINAMATH_GPT_smallest_sum_minimum_l2139_213917


namespace NUMINAMATH_GPT_algebra_expression_correct_l2139_213985

theorem algebra_expression_correct {x y : ℤ} (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
  sorry

end NUMINAMATH_GPT_algebra_expression_correct_l2139_213985


namespace NUMINAMATH_GPT_twelve_xy_leq_fourx_1_y_9y_1_x_l2139_213996

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end NUMINAMATH_GPT_twelve_xy_leq_fourx_1_y_9y_1_x_l2139_213996


namespace NUMINAMATH_GPT_max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l2139_213971

theorem max_lg_sum_eq_one {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ u, u = Real.log x + Real.log y → u ≤ 1 :=
sorry

theorem min_inv_sum_eq_specific_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ v, v = (1 / x) + (1 / y) → v ≥ (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end NUMINAMATH_GPT_max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l2139_213971


namespace NUMINAMATH_GPT_solve_problem_l2139_213941

-- Conditions from the problem
def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_conditions (n p : ℕ) : Prop := 
  (p > 1) ∧ is_prime p ∧ (n > 0) ∧ (n ≤ 2 * p)

-- Main proof statement
theorem solve_problem (n p : ℕ) (h1 : satisfies_conditions n p)
    (h2 : (p - 1) ^ n + 1 ∣ n ^ (p - 1)) :
    (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
sorry

end NUMINAMATH_GPT_solve_problem_l2139_213941


namespace NUMINAMATH_GPT_unique_two_digit_integer_l2139_213950

theorem unique_two_digit_integer (s : ℕ) (hs : s > 9 ∧ s < 100) (h : 13 * s ≡ 42 [MOD 100]) : s = 34 :=
by sorry

end NUMINAMATH_GPT_unique_two_digit_integer_l2139_213950


namespace NUMINAMATH_GPT_eval_expression_l2139_213925

theorem eval_expression : 
  3000^3 - 2998 * 3000^2 - 2998^2 * 3000 + 2998^3 = 23992 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l2139_213925


namespace NUMINAMATH_GPT_degree_g_of_degree_f_and_h_l2139_213989

noncomputable def degree (p : ℕ) := p -- definition to represent degree of polynomials

theorem degree_g_of_degree_f_and_h (f g : ℕ → ℕ) (h : ℕ → ℕ) 
  (deg_h : ℕ) (deg_f : ℕ) (deg_10 : deg_h = 10) (deg_3 : deg_f = 3) 
  (h_eq : ∀ x, degree (h x) = degree (f (g x)) + degree x ^ 5) :
  degree (g 0) = 4 :=
by
  sorry

end NUMINAMATH_GPT_degree_g_of_degree_f_and_h_l2139_213989


namespace NUMINAMATH_GPT_required_number_of_shirts_l2139_213951

/-
In a shop, there is a sale of clothes. Every shirt costs $5, every hat $4, and a pair of jeans $10.
You need to pay $51 for a certain number of shirts, two pairs of jeans, and four hats.
Prove that the number of shirts you need to buy is 3.
-/

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_payment : ℕ := 51
def number_of_jeans : ℕ := 2
def number_of_hats : ℕ := 4

theorem required_number_of_shirts (S : ℕ) (h : 5 * S + 2 * jeans_cost + 4 * hat_cost = total_payment) : S = 3 :=
by
  -- This statement asserts that given the defined conditions, the number of shirts that satisfies the equation is 3.
  sorry

end NUMINAMATH_GPT_required_number_of_shirts_l2139_213951


namespace NUMINAMATH_GPT_find_n_l2139_213982

theorem find_n (n : ℕ) (h : (1 + n + (n * (n - 1)) / 2) / 2^n = 7 / 32) : n = 6 :=
sorry

end NUMINAMATH_GPT_find_n_l2139_213982


namespace NUMINAMATH_GPT_problem1_problem2_l2139_213907

theorem problem1 (a b : ℤ) (h : Even (5 * b + a)) : Even (a - 3 * b) :=
sorry

theorem problem2 (a b : ℤ) (h : Odd (5 * b + a)) : Odd (a - 3 * b) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2139_213907


namespace NUMINAMATH_GPT_tan_simplify_l2139_213956

theorem tan_simplify (α : ℝ) (h : Real.tan α = 1 / 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = - 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_simplify_l2139_213956


namespace NUMINAMATH_GPT_find_f_nine_l2139_213938

-- Define the function f that satisfies the conditions
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x + y) = f(x) * f(y) for all real x and y
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y

-- Define the condition that f(3) = 4
axiom f_three : f 3 = 4

-- State the main theorem to prove that f(9) = 64
theorem find_f_nine : f 9 = 64 := by
  sorry

end NUMINAMATH_GPT_find_f_nine_l2139_213938


namespace NUMINAMATH_GPT_ordered_pair_unique_l2139_213975

theorem ordered_pair_unique (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x, y) = (1, 14) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_unique_l2139_213975


namespace NUMINAMATH_GPT_winning_strategy_l2139_213914

theorem winning_strategy (n : ℕ) (take_stones : ℕ → Prop) :
  n = 13 ∧ (∀ k, (k = 1 ∨ k = 2) → take_stones k) →
  (take_stones 12 ∨ take_stones 9 ∨ take_stones 6 ∨ take_stones 3) :=
by sorry

end NUMINAMATH_GPT_winning_strategy_l2139_213914


namespace NUMINAMATH_GPT_find_dividend_and_divisor_l2139_213978

theorem find_dividend_and_divisor (quotient : ℕ) (remainder : ℕ) (total : ℕ) (dividend divisor : ℕ) :
  quotient = 13 ∧ remainder = 6 ∧ total = 137 ∧ (dividend + divisor + quotient + remainder = total)
  ∧ dividend = 13 * divisor + remainder → 
  dividend = 110 ∧ divisor = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_dividend_and_divisor_l2139_213978


namespace NUMINAMATH_GPT_adoption_time_l2139_213998

theorem adoption_time
  (p0 : ℕ) (p1 : ℕ) (rate : ℕ)
  (p0_eq : p0 = 10) (p1_eq : p1 = 15) (rate_eq : rate = 7) :
  Nat.ceil ((p0 + p1) / rate) = 4 := by
  sorry

end NUMINAMATH_GPT_adoption_time_l2139_213998


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l2139_213990

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
    x^2 / 16 + y^2 / 25 = 1 → (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l2139_213990


namespace NUMINAMATH_GPT_pathway_area_ratio_l2139_213984

theorem pathway_area_ratio (AB AD: ℝ) (r: ℝ) (A_rectangle A_circles: ℝ):
  AB = 24 → (AD / AB) = (4 / 3) → r = AB / 2 → 
  A_rectangle = AD * AB → A_circles = π * r^2 →
  (A_rectangle / A_circles) = 16 / (3 * π) :=
by
  sorry

end NUMINAMATH_GPT_pathway_area_ratio_l2139_213984


namespace NUMINAMATH_GPT_entry_cost_proof_l2139_213929

variable (hitting_rate : ℕ → ℝ)
variable (entry_cost : ℝ)
variable (total_hits : ℕ)
variable (money_lost : ℝ)

-- Conditions
axiom hitting_rate_condition : hitting_rate 200 = 0.025
axiom total_hits_condition : total_hits = 300
axiom money_lost_condition : money_lost = 7.5

-- Question: Prove that the cost to enter the contest equals $10.00
theorem entry_cost_proof : entry_cost = 10 := by
  sorry

end NUMINAMATH_GPT_entry_cost_proof_l2139_213929


namespace NUMINAMATH_GPT_inequality_not_always_true_l2139_213900

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) : ¬ (∀ c, (a - b) / c > 0) := 
sorry

end NUMINAMATH_GPT_inequality_not_always_true_l2139_213900


namespace NUMINAMATH_GPT_closest_integer_to_2_plus_sqrt_6_l2139_213926

theorem closest_integer_to_2_plus_sqrt_6 (sqrt6_lower : 2 < Real.sqrt 6) (sqrt6_upper : Real.sqrt 6 < 2.5) : 
  abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 3) ∧ abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 5) :=
by
  sorry

end NUMINAMATH_GPT_closest_integer_to_2_plus_sqrt_6_l2139_213926


namespace NUMINAMATH_GPT_height_of_smaller_cone_is_18_l2139_213923

theorem height_of_smaller_cone_is_18
  (height_frustum : ℝ)
  (area_larger_base : ℝ)
  (area_smaller_base : ℝ) :
  let R := (area_larger_base / π).sqrt
  let r := (area_smaller_base / π).sqrt
  let ratio := r / R
  let H := height_frustum / (1 - ratio)
  let h := ratio * H
  height_frustum = 18 ∧ area_larger_base = 400 * π ∧ area_smaller_base = 100 * π
  → h = 18 := by
  sorry

end NUMINAMATH_GPT_height_of_smaller_cone_is_18_l2139_213923


namespace NUMINAMATH_GPT_quadratic_equation_with_product_of_roots_20_l2139_213967

theorem quadratic_equation_with_product_of_roots_20
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c / a = 20) :
  ∃ b : ℝ, ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  use 1
  use 20
  sorry

end NUMINAMATH_GPT_quadratic_equation_with_product_of_roots_20_l2139_213967


namespace NUMINAMATH_GPT_inequality_solver_l2139_213988

variable {m n x : ℝ}

-- Main theorem statement validating the instances described above.
theorem inequality_solver (h : 2 * m * x + 3 < 3 * x + n) :
  (2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨ 
  (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨ 
  (m = 3 / 2 ∧ n > 3 ∧ ∀ x : ℝ, true) ∨ 
  (m = 3 / 2 ∧ n ≤ 3 ∧ ∀ x : ℝ, false) :=
sorry

end NUMINAMATH_GPT_inequality_solver_l2139_213988


namespace NUMINAMATH_GPT_spending_total_march_to_july_l2139_213999

/-- Given the conditions:
  1. Total amount spent by the beginning of March is 1.2 million,
  2. Total amount spent by the end of July is 5.4 million,
  Prove that the total amount spent during March, April, May, June, and July is 4.2 million. -/
theorem spending_total_march_to_july
  (spent_by_end_of_feb : ℝ)
  (spent_by_end_of_july : ℝ)
  (h1 : spent_by_end_of_feb = 1.2)
  (h2 : spent_by_end_of_july = 5.4) :
  spent_by_end_of_july - spent_by_end_of_feb = 4.2 :=
by
  sorry

end NUMINAMATH_GPT_spending_total_march_to_july_l2139_213999


namespace NUMINAMATH_GPT_n_greater_than_sqrt_p_sub_1_l2139_213955

theorem n_greater_than_sqrt_p_sub_1 {p n : ℕ} (hp : Nat.Prime p) (hn : n ≥ 2) (hdiv : p ∣ (n^6 - 1)) : n > Nat.sqrt p - 1 := 
by
  sorry

end NUMINAMATH_GPT_n_greater_than_sqrt_p_sub_1_l2139_213955


namespace NUMINAMATH_GPT_integer_roots_of_polynomial_l2139_213974

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_of_polynomial_l2139_213974


namespace NUMINAMATH_GPT_combined_weight_of_three_boxes_l2139_213965

theorem combined_weight_of_three_boxes (a b c d : ℕ) (h₁ : a + b = 132) (h₂ : a + c = 136) (h₃ : b + c = 138) (h₄ : d = 60) : 
  a + b + c = 203 :=
sorry

end NUMINAMATH_GPT_combined_weight_of_three_boxes_l2139_213965


namespace NUMINAMATH_GPT_cubical_pyramidal_segment_volume_and_area_l2139_213937

noncomputable def volume_and_area_sum (a : ℝ) : ℝ :=
  (1/4 * (9 + 27 * Real.sqrt 13))

theorem cubical_pyramidal_segment_volume_and_area :
  ∀ a : ℝ, a = 3 → volume_and_area_sum a = (9/2 + 27 * Real.sqrt 13 / 8) := by
  intro a ha
  sorry

end NUMINAMATH_GPT_cubical_pyramidal_segment_volume_and_area_l2139_213937


namespace NUMINAMATH_GPT_age_difference_64_l2139_213993

variables (Patrick Michael Monica : ℕ)
axiom age_ratio_1 : ∃ (x : ℕ), Patrick = 3 * x ∧ Michael = 5 * x
axiom age_ratio_2 : ∃ (y : ℕ), Michael = 3 * y ∧ Monica = 5 * y
axiom age_sum : Patrick + Michael + Monica = 196

theorem age_difference_64 : Monica - Patrick = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_difference_64_l2139_213993


namespace NUMINAMATH_GPT_probability_not_within_B_l2139_213944

-- Definition representing the problem context
structure Squares where
  areaA : ℝ
  areaA_pos : areaA = 65
  perimeterB : ℝ
  perimeterB_pos : perimeterB = 16

-- The theorem to be proved
theorem probability_not_within_B (s : Squares) : 
  let sideA := Real.sqrt s.areaA
  let sideB := s.perimeterB / 4
  let areaB := sideB^2
  let area_not_covered := s.areaA - areaB
  let probability := area_not_covered / s.areaA
  probability = 49 / 65 := 
by
  sorry

end NUMINAMATH_GPT_probability_not_within_B_l2139_213944


namespace NUMINAMATH_GPT_calculate_value_of_squares_difference_l2139_213913

theorem calculate_value_of_squares_difference : 305^2 - 301^2 = 2424 :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_value_of_squares_difference_l2139_213913


namespace NUMINAMATH_GPT_correct_speed_to_reach_on_time_l2139_213940

theorem correct_speed_to_reach_on_time
  (d : ℝ)
  (t : ℝ)
  (h1 : d = 50 * (t + 1 / 12))
  (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 := 
by
  sorry

end NUMINAMATH_GPT_correct_speed_to_reach_on_time_l2139_213940


namespace NUMINAMATH_GPT_jill_sod_area_l2139_213934

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end NUMINAMATH_GPT_jill_sod_area_l2139_213934


namespace NUMINAMATH_GPT_scientific_notation_110_billion_l2139_213936

theorem scientific_notation_110_billion :
  ∃ (n : ℝ) (e : ℤ), 110000000000 = n * 10 ^ e ∧ 1 ≤ n ∧ n < 10 ∧ n = 1.1 ∧ e = 11 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_110_billion_l2139_213936


namespace NUMINAMATH_GPT_number_of_workers_l2139_213932

-- Definitions for conditions
def initial_contribution (W C : ℕ) : Prop := W * C = 300000
def additional_contribution (W C : ℕ) : Prop := W * (C + 50) = 350000

-- Proof statement
theorem number_of_workers (W C : ℕ) (h1 : initial_contribution W C) (h2 : additional_contribution W C) : W = 1000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_workers_l2139_213932


namespace NUMINAMATH_GPT_magician_draws_two_cards_l2139_213957

-- Define the total number of unique cards
def total_cards : ℕ := 15^2

-- Define the number of duplicate cards
def duplicate_cards : ℕ := 15

-- Define the number of ways to choose 2 cards from the duplicate cards
def choose_two_duplicates : ℕ := Nat.choose 15 2

-- Define the number of ways to choose 1 duplicate card and 1 non-duplicate card
def choose_one_duplicate_one_nonduplicate : ℕ := (15 * (total_cards - 15 - 14 - 14))

-- The main theorem to prove
theorem magician_draws_two_cards : choose_two_duplicates + choose_one_duplicate_one_nonduplicate = 2835 := by
  sorry

end NUMINAMATH_GPT_magician_draws_two_cards_l2139_213957


namespace NUMINAMATH_GPT_solution_l2139_213969

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem solution (a b : ℝ) (H : a = 5 * Real.pi / 8 ∧ b = 7 * Real.pi / 8) :
  is_monotonically_increasing g a b :=
sorry

end NUMINAMATH_GPT_solution_l2139_213969


namespace NUMINAMATH_GPT_average_score_for_girls_l2139_213994

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end NUMINAMATH_GPT_average_score_for_girls_l2139_213994


namespace NUMINAMATH_GPT_bounds_for_f3_l2139_213949

variable (a c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - c

theorem bounds_for_f3 (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
                      (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end NUMINAMATH_GPT_bounds_for_f3_l2139_213949


namespace NUMINAMATH_GPT_tank_capacity_is_24_l2139_213981

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end NUMINAMATH_GPT_tank_capacity_is_24_l2139_213981


namespace NUMINAMATH_GPT_average_of_angles_l2139_213918

theorem average_of_angles (p q r s t : ℝ) (h : p + q + r + s + t = 180) : 
  (p + q + r + s + t) / 5 = 36 :=
by
  sorry

end NUMINAMATH_GPT_average_of_angles_l2139_213918


namespace NUMINAMATH_GPT_max_ahead_distance_l2139_213961

noncomputable def distance_run_by_alex (initial_distance ahead1 ahead_max_runs final_ahead : ℝ) : ℝ :=
  initial_distance + ahead1 + ahead_max_runs + final_ahead

theorem max_ahead_distance :
  let initial_distance := 200
  let ahead1 := 300
  let final_ahead := 440
  let total_road := 5000
  let distance_remaining := 3890
  let distance_run_alex := total_road - distance_remaining
  ∃ X : ℝ, distance_run_by_alex initial_distance ahead1 X final_ahead = distance_run_alex ∧ X = 170 :=
by
  intro initial_distance ahead1 final_ahead total_road distance_remaining distance_run_alex
  use 170
  simp [initial_distance, ahead1, final_ahead, total_road, distance_remaining, distance_run_alex, distance_run_by_alex]
  sorry

end NUMINAMATH_GPT_max_ahead_distance_l2139_213961


namespace NUMINAMATH_GPT_sum_divisible_by_17_l2139_213916

theorem sum_divisible_by_17 :
    (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 17 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_divisible_by_17_l2139_213916


namespace NUMINAMATH_GPT_combined_distance_proof_l2139_213972

/-- Define the distances walked by Lionel, Esther, and Niklaus in their respective units -/
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

/-- Define the conversion factors -/
def miles_to_feet : ℕ := 5280
def yards_to_feet : ℕ := 3

/-- The total combined distance in feet -/
def total_distance_feet : ℕ :=
  (lionel_miles * miles_to_feet) + (esther_yards * yards_to_feet) + niklaus_feet

theorem combined_distance_proof : total_distance_feet = 24332 := by
  -- expand definitions and calculations here...
  -- lionel = 4 * 5280 = 21120
  -- esther = 975 * 3 = 2925
  -- niklaus = 1287
  -- sum = 21120 + 2925 + 1287 = 24332
  sorry

end NUMINAMATH_GPT_combined_distance_proof_l2139_213972


namespace NUMINAMATH_GPT_quadratic_polynomial_fourth_power_l2139_213947

theorem quadratic_polynomial_fourth_power {a b c : ℤ} (h : ∀ x : ℤ, ∃ k : ℤ, ax^2 + bx + c = k^4) : a = 0 ∧ b = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_polynomial_fourth_power_l2139_213947


namespace NUMINAMATH_GPT_time_to_fill_pool_l2139_213910

noncomputable def slower_pump_rate : ℝ := 1 / 12.5
noncomputable def faster_pump_rate : ℝ := 1.5 * slower_pump_rate
noncomputable def combined_rate : ℝ := slower_pump_rate + faster_pump_rate

theorem time_to_fill_pool : (1 / combined_rate) = 5 := 
by
  sorry

end NUMINAMATH_GPT_time_to_fill_pool_l2139_213910


namespace NUMINAMATH_GPT_remaining_episodes_l2139_213906

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end NUMINAMATH_GPT_remaining_episodes_l2139_213906


namespace NUMINAMATH_GPT_increasing_function_on_interval_l2139_213970

section
  variable (a b : ℝ)
  def f (x : ℝ) : ℝ := |x^2 - 2*a*x + b|

  theorem increasing_function_on_interval (h : a^2 - b ≤ 0) :
    ∀ x y : ℝ, a ≤ x → x ≤ y → f x ≤ f y := 
  sorry
end

end NUMINAMATH_GPT_increasing_function_on_interval_l2139_213970


namespace NUMINAMATH_GPT_smallest_possible_n_l2139_213930

theorem smallest_possible_n (x n : ℤ) (hx : 0 < x) (m : ℤ) (hm : m = 30) (h1 : m.gcd n = x + 1) (h2 : m.lcm n = x * (x + 1)) : n = 6 := sorry

end NUMINAMATH_GPT_smallest_possible_n_l2139_213930


namespace NUMINAMATH_GPT_scrap_cookie_radius_is_correct_l2139_213943

noncomputable def radius_of_scrap_cookie (large_radius small_radius : ℝ) (number_of_cookies : ℕ) : ℝ :=
  have large_area : ℝ := Real.pi * large_radius^2
  have small_area : ℝ := Real.pi * small_radius^2
  have total_small_area : ℝ := small_area * number_of_cookies
  have scrap_area : ℝ := large_area - total_small_area
  Real.sqrt (scrap_area / Real.pi)

theorem scrap_cookie_radius_is_correct :
  radius_of_scrap_cookie 8 2 9 = 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_scrap_cookie_radius_is_correct_l2139_213943


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l2139_213953

theorem remainder_when_divided_by_6 :
  ∃ (n : ℕ), (∃ k : ℕ, n = 3 * k + 2 ∧ ∃ m : ℕ, k = 4 * m + 3) → n % 6 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l2139_213953


namespace NUMINAMATH_GPT_weighted_averages_correct_l2139_213960

def group_A_boys : ℕ := 20
def group_B_boys : ℕ := 25
def group_C_boys : ℕ := 15

def group_A_weight : ℝ := 50.25
def group_B_weight : ℝ := 45.15
def group_C_weight : ℝ := 55.20

def group_A_height : ℝ := 160
def group_B_height : ℝ := 150
def group_C_height : ℝ := 165

def group_A_age : ℝ := 15
def group_B_age : ℝ := 14
def group_C_age : ℝ := 16

def group_A_athletic : ℝ := 0.60
def group_B_athletic : ℝ := 0.40
def group_C_athletic : ℝ := 0.75

noncomputable def total_boys : ℕ := group_A_boys + group_B_boys + group_C_boys

noncomputable def weighted_average_height : ℝ := 
    (group_A_boys * group_A_height + group_B_boys * group_B_height + group_C_boys * group_C_height) / total_boys

noncomputable def weighted_average_weight : ℝ := 
    (group_A_boys * group_A_weight + group_B_boys * group_B_weight + group_C_boys * group_C_weight) / total_boys

noncomputable def weighted_average_age : ℝ := 
    (group_A_boys * group_A_age + group_B_boys * group_B_age + group_C_boys * group_C_age) / total_boys

noncomputable def weighted_average_athletic : ℝ := 
    (group_A_boys * group_A_athletic + group_B_boys * group_B_athletic + group_C_boys * group_C_athletic) / total_boys

theorem weighted_averages_correct :
  weighted_average_height = 157.08 ∧
  weighted_average_weight = 49.36 ∧
  weighted_average_age = 14.83 ∧
  weighted_average_athletic = 0.5542 := 
  by
    sorry

end NUMINAMATH_GPT_weighted_averages_correct_l2139_213960


namespace NUMINAMATH_GPT_valid_N_values_l2139_213963

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end NUMINAMATH_GPT_valid_N_values_l2139_213963


namespace NUMINAMATH_GPT_quad_roots_sum_l2139_213973

theorem quad_roots_sum {x₁ x₂ : ℝ} (h1 : x₁ + x₂ = 5) (h2 : x₁ * x₂ = -6) :
  1 / x₁ + 1 / x₂ = -5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_quad_roots_sum_l2139_213973


namespace NUMINAMATH_GPT_sodas_per_pack_l2139_213903

theorem sodas_per_pack 
  (packs : ℕ) (initial_sodas : ℕ) (days_in_a_week : ℕ) (sodas_per_day : ℕ) 
  (total_sodas_consumed : ℕ) (sodas_per_pack : ℕ) :
  packs = 5 →
  initial_sodas = 10 →
  days_in_a_week = 7 →
  sodas_per_day = 10 →
  total_sodas_consumed = 70 →
  total_sodas_consumed - initial_sodas = packs * sodas_per_pack →
  sodas_per_pack = 12 :=
by
  intros hpacks hinitial hsodas hdaws htpd htcs
  sorry

end NUMINAMATH_GPT_sodas_per_pack_l2139_213903


namespace NUMINAMATH_GPT_cost_of_show_dogs_l2139_213995

noncomputable def cost_per_dog : ℕ → ℕ → ℕ → ℕ
| total_revenue, total_profit, number_of_dogs => (total_revenue - total_profit) / number_of_dogs

theorem cost_of_show_dogs {revenue_per_puppy number_of_puppies profit number_of_dogs : ℕ}
  (h_puppies: number_of_puppies = 6)
  (h_revenue_per_puppy : revenue_per_puppy = 350)
  (h_profit : profit = 1600)
  (h_number_of_dogs : number_of_dogs = 2)
:
  cost_per_dog (number_of_puppies * revenue_per_puppy) profit number_of_dogs = 250 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_show_dogs_l2139_213995


namespace NUMINAMATH_GPT_constant_function_on_chessboard_l2139_213935

theorem constant_function_on_chessboard
  (f : ℤ × ℤ → ℝ)
  (h_nonneg : ∀ (m n : ℤ), 0 ≤ f (m, n))
  (h_mean : ∀ (m n : ℤ), f (m, n) = (f (m + 1, n) + f (m - 1, n) + f (m, n + 1) + f (m, n - 1)) / 4) :
  ∃ c : ℝ, ∀ (m n : ℤ), f (m, n) = c :=
sorry

end NUMINAMATH_GPT_constant_function_on_chessboard_l2139_213935


namespace NUMINAMATH_GPT_popsicle_sticks_left_correct_l2139_213959

noncomputable def popsicle_sticks_left (initial : ℝ) (given : ℝ) : ℝ :=
  initial - given

theorem popsicle_sticks_left_correct :
  popsicle_sticks_left 63 50 = 13 :=
by
  sorry

end NUMINAMATH_GPT_popsicle_sticks_left_correct_l2139_213959


namespace NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l2139_213908

-- Proof problem definitions
theorem calc1 : 15 + (-22) = -7 := sorry

theorem calc2 : (-13) + (-8) = -21 := sorry

theorem calc3 : (-0.9) + 1.5 = 0.6 := sorry

theorem calc4 : (1 / 2) + (-2 / 3) = -1 / 6 := sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l2139_213908


namespace NUMINAMATH_GPT_tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l2139_213931

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x) - x * Real.exp x

theorem tangent_line_a_zero (x : ℝ) (y : ℝ) : 
  a = 0 ∧ x = 1 → (2 * Real.exp 1) * x + y - Real.exp 1 = 0 :=
sorry

theorem range_a_if_fx_neg (a : ℝ) : 
  (∀ x ≥ 1, f a x < 0) → a < Real.exp 1 :=
sorry

theorem max_value_a_one (x : ℝ) : 
  a = 1 → x = (Real.exp 1)⁻¹ → f 1 x = -1 :=
sorry

end NUMINAMATH_GPT_tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l2139_213931


namespace NUMINAMATH_GPT_find_x_plus_y_l2139_213909

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l2139_213909


namespace NUMINAMATH_GPT_larger_integer_is_21_l2139_213901

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end NUMINAMATH_GPT_larger_integer_is_21_l2139_213901


namespace NUMINAMATH_GPT_larger_number_is_17_l2139_213920

noncomputable def x : ℤ := 17
noncomputable def y : ℤ := 12

def sum_condition : Prop := x + y = 29
def diff_condition : Prop := x - y = 5

theorem larger_number_is_17 (h_sum : sum_condition) (h_diff : diff_condition) : x = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_number_is_17_l2139_213920
