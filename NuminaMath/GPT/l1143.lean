import Mathlib

namespace steven_total_seeds_l1143_114369

-- Definitions based on the conditions
def apple_seed_count := 6
def pear_seed_count := 2
def grape_seed_count := 3

def apples_set_aside := 4
def pears_set_aside := 3
def grapes_set_aside := 9

def additional_seeds_needed := 3

-- The total seeds Steven already has
def total_seeds_from_fruits : ℕ :=
  apples_set_aside * apple_seed_count +
  pears_set_aside * pear_seed_count +
  grapes_set_aside * grape_seed_count

-- The total number of seeds Steven needs to collect, as given by the problem's solution
def total_seeds_needed : ℕ :=
  total_seeds_from_fruits + additional_seeds_needed

-- The actual proof statement
theorem steven_total_seeds : total_seeds_needed = 60 :=
  by
    sorry

end steven_total_seeds_l1143_114369


namespace quarter_pounder_cost_l1143_114392

theorem quarter_pounder_cost :
  let fries_cost := 2 * 1.90
  let milkshakes_cost := 2 * 2.40
  let min_purchase := 18
  let current_total := fries_cost + milkshakes_cost
  let amount_needed := min_purchase - current_total
  let additional_spending := 3
  let total_cost := amount_needed + additional_spending
  total_cost = 12.40 :=
by
  sorry

end quarter_pounder_cost_l1143_114392


namespace evaluate_expression_at_minus_one_l1143_114321

theorem evaluate_expression_at_minus_one :
  ((-1 + 1) * (-1 - 2) + 2 * (-1 + 4) * (-1 - 4)) = -30 := by
  sorry

end evaluate_expression_at_minus_one_l1143_114321


namespace min_value_exp_sum_eq_4sqrt2_l1143_114350

theorem min_value_exp_sum_eq_4sqrt2 {a b : ℝ} (h : a + b = 3) : 2^a + 2^b ≥ 4 * Real.sqrt 2 :=
by
  sorry

end min_value_exp_sum_eq_4sqrt2_l1143_114350


namespace pythagorean_theorem_l1143_114323

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l1143_114323


namespace part1_part2_l1143_114353

-- Define the conditions that translate the quadratic equation having distinct real roots
def discriminant_condition (m : ℝ) : Prop :=
  let a := 1
  let b := -4
  let c := 3 - 2 * m
  b ^ 2 - 4 * a * c > 0

-- Define the root condition from Vieta's formulas and the additional given condition
def additional_condition (m : ℝ) : Prop :=
  let x1_plus_x2 := 4
  let x1_times_x2 := 3 - 2 * m
  x1_times_x2 + x1_plus_x2 - m^2 = 4

-- Prove the range of m for part 1
theorem part1 (m : ℝ) : discriminant_condition m → m ≥ -1/2 := by
  sorry

-- Prove the value of m for part 2 with the range condition
theorem part2 (m : ℝ) : discriminant_condition m → additional_condition m → m = 1 := by
  sorry

end part1_part2_l1143_114353


namespace prove_a_21022_le_1_l1143_114360

-- Define the sequence a_n
variable (a : ℕ → ℝ)

-- Conditions for the sequence
axiom seq_condition {n : ℕ} (hn : n ≥ 1) :
  (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)

-- Positive real numbers condition
axiom seq_positive {n : ℕ} (hn : n ≥ 1) :
  a n > 0

-- The main theorem to prove
theorem prove_a_21022_le_1 :
  a 21022 ≤ 1 :=
sorry

end prove_a_21022_le_1_l1143_114360


namespace weekly_earnings_before_rent_l1143_114356

theorem weekly_earnings_before_rent (EarningsAfterRent : ℝ) (weeks : ℕ) (rentPerWeek : ℝ) :
  EarningsAfterRent = 93899 → weeks = 233 → rentPerWeek = 49 →
  ((EarningsAfterRent + rentPerWeek * weeks) / weeks) = 451.99 :=
by
  intros H1 H2 H3
  -- convert the assumptions to the required form
  rw [H1, H2, H3]
  -- provide the objective statement
  change ((93899 + 49 * 233) / 233) = 451.99
  -- leave the final proof details as a sorry for now
  sorry

end weekly_earnings_before_rent_l1143_114356


namespace kevin_bucket_size_l1143_114355

def rate_of_leakage (r : ℝ) : Prop := r = 1.5
def time_away (t : ℝ) : Prop := t = 12
def bucket_size (b : ℝ) (r t : ℝ) : Prop := b = 2 * r * t

theorem kevin_bucket_size
  (r t b : ℝ)
  (H1 : rate_of_leakage r)
  (H2 : time_away t) :
  bucket_size b r t :=
by
  simp [rate_of_leakage, time_away, bucket_size] at *
  sorry

end kevin_bucket_size_l1143_114355


namespace average_weight_of_dogs_is_5_l1143_114344

def weight_of_brown_dog (B : ℝ) : ℝ := B
def weight_of_black_dog (B : ℝ) : ℝ := B + 1
def weight_of_white_dog (B : ℝ) : ℝ := 2 * B
def weight_of_grey_dog (B : ℝ) : ℝ := B - 1

theorem average_weight_of_dogs_is_5 (B : ℝ) (h : (weight_of_brown_dog B + weight_of_black_dog B + weight_of_white_dog B + weight_of_grey_dog B) / 4 = 5) :
  5 = 5 :=
by sorry

end average_weight_of_dogs_is_5_l1143_114344


namespace alley_width_l1143_114380

theorem alley_width (L w : ℝ) (k h : ℝ)
    (h1 : k = L / 2)
    (h2 : h = L * (Real.sqrt 3) / 2)
    (h3 : w^2 + (L / 2)^2 = L^2)
    (h4 : w^2 + (L * (Real.sqrt 3) / 2)^2 = L^2):
    w = (Real.sqrt 3) * L / 2 := 
sorry

end alley_width_l1143_114380


namespace range_of_k_for_real_roots_l1143_114382

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 :=
by
  sorry

end range_of_k_for_real_roots_l1143_114382


namespace geometric_series_sum_l1143_114305

theorem geometric_series_sum {a r : ℚ} (n : ℕ) (h_a : a = 3/4) (h_r : r = 3/4) (h_n : n = 8) : 
       a * (1 - r^n) / (1 - r) = 176925 / 65536 :=
by
  -- Utilizing the provided conditions
  have h_a := h_a
  have h_r := h_r
  have h_n := h_n
  -- Proving the theorem using sorry as a placeholder for the detailed steps
  sorry

end geometric_series_sum_l1143_114305


namespace arithmetic_seq_20th_term_l1143_114391

variable (a : ℕ → ℤ) -- a_n is an arithmetic sequence
variable (d : ℤ) -- common difference of the arithmetic sequence

-- Condition for arithmetic sequence
variable (h_seq : ∀ n, a (n+1) = a n + d)

-- Given conditions
axiom h1 : a 1 + a 3 + a 5 = 105
axiom h2 : a 2 + a 4 + a 6 = 99

-- Goal: prove that a 20 = 1
theorem arithmetic_seq_20th_term :
  a 20 = 1 :=
sorry

end arithmetic_seq_20th_term_l1143_114391


namespace unw_touchable_area_l1143_114345

-- Define the conditions
def ball_radius : ℝ := 1
def container_edge_length : ℝ := 5

-- Define the surface area that the ball can never touch
theorem unw_touchable_area : (ball_radius = 1) ∧ (container_edge_length = 5) → 
  let total_unreachable_area := 120
  let overlapping_area := 24
  let unreachable_area := total_unreachable_area - overlapping_area
  unreachable_area = 96 :=
by
  intros
  sorry

end unw_touchable_area_l1143_114345


namespace find_C_l1143_114309

theorem find_C (C : ℤ) (h : 4 * C + 3 = 31) : C = 7 := by
  sorry

end find_C_l1143_114309


namespace largest_common_divisor_l1143_114300

theorem largest_common_divisor (a b : ℕ) (h1 : a = 360) (h2 : b = 315) : 
  ∃ d : ℕ, d ∣ a ∧ d ∣ b ∧ ∀ e : ℕ, (e ∣ a ∧ e ∣ b) → e ≤ d ∧ d = 45 :=
by
  sorry

end largest_common_divisor_l1143_114300


namespace quadrilateral_circumscribed_circle_l1143_114316

theorem quadrilateral_circumscribed_circle (a : ℝ) :
  ((a + 2) * x + (1 - a) * y - 3 = 0) ∧ ((a - 1) * x + (2 * a + 3) * y + 2 = 0) →
  ( a = 1 ∨ a = -1 ) :=
by
  intro h
  sorry

end quadrilateral_circumscribed_circle_l1143_114316


namespace longest_chord_length_of_circle_l1143_114376

theorem longest_chord_length_of_circle (r : ℝ) (h : r = 5) : ∃ d, d = 10 :=
by
  sorry

end longest_chord_length_of_circle_l1143_114376


namespace calc_g_inv_sum_l1143_114364

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x * x

noncomputable def g_inv (y : ℝ) : ℝ := 
  if y = -4 then 4
  else if y = 0 then 3
  else if y = 4 then -1
  else 0

theorem calc_g_inv_sum : g_inv (-4) + g_inv 0 + g_inv 4 = 6 :=
by
  sorry

end calc_g_inv_sum_l1143_114364


namespace sum_of_trinomials_1_l1143_114319

theorem sum_of_trinomials_1 (p q : ℝ) :
  (p + q = 0 ∨ p + q = 8) →
  (2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 2 ∨ 2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 18) :=
by sorry

end sum_of_trinomials_1_l1143_114319


namespace tan_of_geometric_sequence_is_negative_sqrt_3_l1143_114328

variable {a : ℕ → ℝ} 

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q, m + n = p + q → a m * a n = a p * a q

theorem tan_of_geometric_sequence_is_negative_sqrt_3 
  (hgeo : is_geometric_sequence a)
  (hcond : a 2 * a 3 * a 4 = - a 7 ^ 2 ∧ a 7 ^ 2 = 64) :
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = - Real.sqrt 3 :=
sorry

end tan_of_geometric_sequence_is_negative_sqrt_3_l1143_114328


namespace cereal_difference_l1143_114341

-- Variables to represent the amounts of cereal in each box
variable (A B C : ℕ)

-- Define the conditions given in the problem
def problem_conditions : Prop :=
  A = 14 ∧
  B = A / 2 ∧
  A + B + C = 33

-- Prove the desired conclusion under these conditions
theorem cereal_difference
  (h : problem_conditions A B C) :
  C - B = 5 :=
sorry

end cereal_difference_l1143_114341


namespace imaginary_part_is_neg_two_l1143_114397

open Complex

noncomputable def imaginary_part_of_square : ℂ := (1 - I)^2

theorem imaginary_part_is_neg_two : imaginary_part_of_square.im = -2 := by
  sorry

end imaginary_part_is_neg_two_l1143_114397


namespace repeating_decimal_427_diff_l1143_114370

theorem repeating_decimal_427_diff :
  let G := 0.427427427427
  let num := 427
  let denom := 999
  num.gcd denom = 1 →
  denom - num = 572 :=
by
  intros G num denom gcd_condition
  sorry

end repeating_decimal_427_diff_l1143_114370


namespace last_digit_322_power_111569_l1143_114358

theorem last_digit_322_power_111569 : (322 ^ 111569) % 10 = 2 := 
by {
  sorry
}

end last_digit_322_power_111569_l1143_114358


namespace dvaneft_percentage_bounds_l1143_114357

theorem dvaneft_percentage_bounds (x y z : ℝ) (n m : ℕ) 
  (h1 : x * n + y * m = z * (m + n))
  (h2 : 3 * x * n = y * m)
  (h3_1 : 10 ≤ y - x)
  (h3_2 : y - x ≤ 18)
  (h4_1 : 18 ≤ z)
  (h4_2 : z ≤ 42)
  : (15 ≤ (n:ℝ) / (2 * (n + m)) * 100) ∧ ((n:ℝ) / (2 * (n + m)) * 100 ≤ 25) :=
by
  sorry

end dvaneft_percentage_bounds_l1143_114357


namespace mary_total_spent_l1143_114336

-- The conditions given in the problem
def cost_berries : ℝ := 11.08
def cost_apples : ℝ := 14.33
def cost_peaches : ℝ := 9.31

-- The theorem to prove the total cost
theorem mary_total_spent : cost_berries + cost_apples + cost_peaches = 34.72 := 
by
  sorry

end mary_total_spent_l1143_114336


namespace bubbleSort_iter_count_l1143_114393

/-- Bubble sort iterates over the list repeatedly, swapping adjacent elements if they are in the wrong order. -/
def bubbleSortSteps (lst : List Int) : List (List Int) :=
sorry -- Implementation of bubble sort to capture each state after each iteration

/-- Prove that sorting [6, -3, 0, 15] in descending order using bubble sort requires exactly 3 iterations. -/
theorem bubbleSort_iter_count : 
  (bubbleSortSteps [6, -3, 0, 15]).length = 3 :=
sorry

end bubbleSort_iter_count_l1143_114393


namespace value_of_a_m_minus_3n_l1143_114363

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end value_of_a_m_minus_3n_l1143_114363


namespace problem1_problem2_l1143_114302

-- Theorem 1: Given a^2 - b^2 = 1940:
theorem problem1 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1940 → 
  (a = 102 ∧ b = 92) := 
by 
  sorry

-- Theorem 2: Given a^2 - b^2 = 1920:
theorem problem2 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1920 → 
  (a = 101 ∧ b = 91) ∨ 
  (a = 58 ∧ b = 38) ∨ 
  (a = 47 ∧ b = 17) ∨ 
  (a = 44 ∧ b = 4) := 
by 
  sorry

end problem1_problem2_l1143_114302


namespace total_colors_over_two_hours_l1143_114389

def colors_in_first_hour : Nat :=
  let quick_colors := 5 * 3
  let slow_colors := 2 * 3
  quick_colors + slow_colors

def colors_in_second_hour : Nat :=
  let quick_colors := (5 * 2) * 3
  let slow_colors := (2 * 2) * 3
  quick_colors + slow_colors

theorem total_colors_over_two_hours : colors_in_first_hour + colors_in_second_hour = 63 := by
  sorry

end total_colors_over_two_hours_l1143_114389


namespace average_time_relay_race_l1143_114372

theorem average_time_relay_race :
  let dawson_time := 38
  let henry_time := 7
  let total_legs := 2
  (dawson_time + henry_time) / total_legs = 22.5 :=
by
  sorry

end average_time_relay_race_l1143_114372


namespace pirate_total_dollar_amount_l1143_114374

def base_5_to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨p, d⟩ => d * base^p) |>.sum

def jewelry_base5 := [3, 1, 2, 4]
def gold_coins_base5 := [3, 1, 2, 2]
def alcohol_base5 := [1, 2, 4]

def jewelry_base10 := base_5_to_base_10 jewelry_base5 5
def gold_coins_base10 := base_5_to_base_10 gold_coins_base5 5
def alcohol_base10 := base_5_to_base_10 alcohol_base5 5

def total_base10 := jewelry_base10 + gold_coins_base10 + alcohol_base10

theorem pirate_total_dollar_amount :
  total_base10 = 865 :=
by
  unfold total_base10 jewelry_base10 gold_coins_base10 alcohol_base10 base_5_to_base_10
  simp
  sorry

end pirate_total_dollar_amount_l1143_114374


namespace tea_bags_count_l1143_114373

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l1143_114373


namespace parallel_lines_slope_equal_l1143_114313

theorem parallel_lines_slope_equal (m : ℝ) : 
  (∃ m : ℝ, -(m+4)/(m+2) = -(m+2)/(m+1)) → m = 0 := 
by
  sorry

end parallel_lines_slope_equal_l1143_114313


namespace smallest_c_d_sum_l1143_114367

theorem smallest_c_d_sum : ∃ (c d : ℕ), 2^12 * 7^6 = c^d ∧  (∀ (c' d' : ℕ), 2^12 * 7^6 = c'^d'  → (c + d) ≤ (c' + d')) ∧ c + d = 21954 := by
  sorry

end smallest_c_d_sum_l1143_114367


namespace degree_to_radian_l1143_114347

theorem degree_to_radian : (855 : ℝ) * (Real.pi / 180) = (59 / 12) * Real.pi :=
by
  sorry

end degree_to_radian_l1143_114347


namespace nat_n_divisibility_cond_l1143_114301

theorem nat_n_divisibility_cond (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end nat_n_divisibility_cond_l1143_114301


namespace tangent_line_at_2_number_of_zeros_l1143_114330

noncomputable def f (x : ℝ) := 3 * Real.log x + (1/2) * x^2 - 4 * x + 1

theorem tangent_line_at_2 :
  let x := 2
  ∃ k b : ℝ, (∀ y : ℝ, y = k * x + b) ∧ (k = -1/2) ∧ (b = 3 * Real.log 2 - 5) ∧ (∀ x y : ℝ, (y - (3 * Real.log 2 - 5) = -1/2 * (x - 2)) ↔ (x + 2 * y - 6 * Real.log 2 + 8 = 0)) :=
by
  sorry

noncomputable def g (x : ℝ) (m : ℝ) := f x - m

theorem number_of_zeros (m : ℝ) :
  let g := g
  (m > -5/2 ∨ m < 3 * Real.log 3 - 13/2 → ∃ x : ℝ, g x = 0) ∧ 
  (m = -5/2 ∨ m = 3 * Real.log 3 - 13/2 → ∃ x y : ℝ, g x = 0 ∧ g y = 0 ∧ x ≠ y) ∧
  (3 * Real.log 3 - 13/2 < m ∧ m < -5/2 → ∃ x y z : ℝ, g x = 0 ∧ g y = 0 ∧ g z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by
  sorry

end tangent_line_at_2_number_of_zeros_l1143_114330


namespace resulting_polygon_sides_l1143_114348

/-
Problem statement: 

Construct a regular pentagon on one side of a regular heptagon.
On one non-adjacent side of the pentagon, construct a regular hexagon.
On a non-adjacent side of the hexagon, construct an octagon.
Continue to construct regular polygons in the same way, until you construct a nonagon.
How many sides does the resulting polygon have?

Given facts:
1. Start with a heptagon (7 sides).
2. Construct a pentagon (5 sides) on one side of the heptagon.
3. Construct a hexagon (6 sides) on a non-adjacent side of the pentagon.
4. Construct an octagon (8 sides) on a non-adjacent side of the hexagon.
5. Construct a nonagon (9 sides) on a non-adjacent side of the octagon.
-/

def heptagon_sides : ℕ := 7
def pentagon_sides : ℕ := 5
def hexagon_sides : ℕ := 6
def octagon_sides : ℕ := 8
def nonagon_sides : ℕ := 9

theorem resulting_polygon_sides : 
  (heptagon_sides + nonagon_sides - 2 * 1) + (pentagon_sides + hexagon_sides + octagon_sides - 3 * 2) = 27 := by
  sorry

end resulting_polygon_sides_l1143_114348


namespace value_of_3a_plus_6b_l1143_114375

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2 * b = 1) : 3 * a + 6 * b = 3 :=
sorry

end value_of_3a_plus_6b_l1143_114375


namespace carriage_problem_l1143_114396

theorem carriage_problem (x : ℕ) : 
  3 * (x - 2) = 2 * x + 9 := 
sorry

end carriage_problem_l1143_114396


namespace Rachel_total_earnings_l1143_114349

-- Define the constants for the conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def tip_per_person : ℝ := 1.25

-- Define the problem
def total_money_made : ℝ := hourly_wage + (people_served * tip_per_person)

-- State the theorem to be proved
theorem Rachel_total_earnings : total_money_made = 37 := by
  sorry

end Rachel_total_earnings_l1143_114349


namespace most_colored_pencils_l1143_114394

theorem most_colored_pencils (total red blue yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - (red + blue)) :
  blue = 12 :=
by
  sorry

end most_colored_pencils_l1143_114394


namespace value_of_a_l1143_114352

-- Define the quadratic function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

-- Define the condition f(1) = f(2)
def condition (a b : ℝ) : Prop := f 1 a b = f 2 a b

-- The proof problem statement
theorem value_of_a (a b : ℝ) (h : condition a b) : a = -3 :=
by sorry

end value_of_a_l1143_114352


namespace oranges_in_first_bucket_l1143_114378

theorem oranges_in_first_bucket
  (x : ℕ) -- number of oranges in the first bucket
  (h1 : ∃ n, n = x) -- condition: There are some oranges in the first bucket
  (h2 : ∃ y, y = x + 17) -- condition: The second bucket has 17 more oranges than the first bucket
  (h3 : ∃ z, z = x + 6) -- condition: The third bucket has 11 fewer oranges than the second bucket
  (h4 : x + (x + 17) + (x + 6) = 89) -- condition: There are 89 oranges in all the buckets
  : x = 22 := -- conclusion: number of oranges in the first bucket is 22
sorry

end oranges_in_first_bucket_l1143_114378


namespace length_of_other_parallel_side_l1143_114390

theorem length_of_other_parallel_side (a b h area : ℝ) 
  (h_area : area = 190) 
  (h_parallel1 : b = 18) 
  (h_height : h = 10) : 
  a = 20 :=
by
  sorry

end length_of_other_parallel_side_l1143_114390


namespace ratio_diminished_to_total_l1143_114318

-- Definitions related to the conditions
def N := 240
def P := 60
def fifth_part_increased (N : ℕ) : ℕ := (N / 5) + 6
def part_diminished (P : ℕ) : ℕ := P - 6

-- The proof problem statement
theorem ratio_diminished_to_total 
  (h1 : fifth_part_increased N = part_diminished P) : 
  (P - 6) / N = 9 / 40 :=
by sorry

end ratio_diminished_to_total_l1143_114318


namespace sum_of_distinct_squares_l1143_114332

theorem sum_of_distinct_squares:
  ∀ (a b c : ℕ),
  a + b + c = 23 ∧ Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 9 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 + c^2 = 179 ∨ a^2 + b^2 + c^2 = 259 →
  a^2 + b^2 + c^2 = 438 :=
by
  sorry

end sum_of_distinct_squares_l1143_114332


namespace area_of_trapezoid_l1143_114385

variable (a d : ℝ)
variable (h b1 b2 : ℝ)

def is_arithmetic_progression (a d : ℝ) (h b1 b2 : ℝ) : Prop :=
  h = a ∧ b1 = a + d ∧ b2 = a - d

theorem area_of_trapezoid (a d : ℝ) (h b1 b2 : ℝ) (hAP : is_arithmetic_progression a d h b1 b2) :
  ∃ J : ℝ, J = a^2 ∧ ∀ x : ℝ, 0 ≤ x → (J = x → x ≥ 0) :=
by
  sorry

end area_of_trapezoid_l1143_114385


namespace interchangeable_statements_l1143_114320

-- Modeled conditions and relationships
def perpendicular (l p: Type) : Prop := sorry -- Definition of perpendicularity between a line and a plane
def parallel (a b: Type) : Prop := sorry -- Definition of parallelism between two objects (lines or planes)

-- Original Statements
def statement_1 := ∀ (l₁ l₂ p: Type), (perpendicular l₁ p) ∧ (perpendicular l₂ p) → parallel l₁ l₂
def statement_2 := ∀ (p₁ p₂ p: Type), (perpendicular p₁ p) ∧ (perpendicular p₂ p) → parallel p₁ p₂
def statement_3 := ∀ (l₁ l₂ l: Type), (parallel l₁ l) ∧ (parallel l₂ l) → parallel l₁ l₂
def statement_4 := ∀ (l₁ l₂ p: Type), (parallel l₁ p) ∧ (parallel l₂ p) → parallel l₁ l₂

-- Swapped Statements
def swapped_1 := ∀ (p₁ p₂ l: Type), (perpendicular p₁ l) ∧ (perpendicular p₂ l) → parallel p₁ p₂
def swapped_2 := ∀ (l₁ l₂ l: Type), (perpendicular l₁ l) ∧ (perpendicular l₂ l) → parallel l₁ l₂
def swapped_3 := ∀ (p₁ p₂ p: Type), (parallel p₁ p) ∧ (parallel p₂ p) → parallel p₁ p₂
def swapped_4 := ∀ (p₁ p₂ l: Type), (parallel p₁ l) ∧ (parallel p₂ l) → parallel p₁ p₂

-- Proof Problem: Verify which statements are interchangeable
theorem interchangeable_statements :
  (statement_1 ↔ swapped_1) ∧
  (statement_2 ↔ swapped_2) ∧
  (statement_3 ↔ swapped_3) ∧
  (statement_4 ↔ swapped_4) :=
sorry

end interchangeable_statements_l1143_114320


namespace rightmost_three_digits_of_7_pow_1987_l1143_114351

theorem rightmost_three_digits_of_7_pow_1987 :
  (7^1987 : ℕ) % 1000 = 643 := 
by 
  sorry

end rightmost_three_digits_of_7_pow_1987_l1143_114351


namespace cube_sufficient_but_not_necessary_l1143_114379

theorem cube_sufficient_but_not_necessary (x : ℝ) : (x^3 > 27 → |x| > 3) ∧ (¬(|x| > 3 → x^3 > 27)) :=
by
  sorry

end cube_sufficient_but_not_necessary_l1143_114379


namespace x_is_36_percent_of_z_l1143_114381

variable (x y z : ℝ)

theorem x_is_36_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.30 * z) : x = 0.36 * z :=
by
  sorry

end x_is_36_percent_of_z_l1143_114381


namespace correct_combined_average_l1143_114333

noncomputable def average_marks : ℝ :=
  let num_students : ℕ := 100
  let avg_math_marks : ℝ := 85
  let avg_science_marks : ℝ := 89
  let incorrect_math_marks : List ℝ := [76, 80, 95, 70, 90]
  let correct_math_marks : List ℝ := [86, 70, 75, 90, 100]
  let incorrect_science_marks : List ℝ := [105, 60, 80, 92, 78]
  let correct_science_marks : List ℝ := [95, 70, 90, 82, 88]

  let total_incorrect_math := incorrect_math_marks.sum
  let total_correct_math := correct_math_marks.sum
  let diff_math := total_correct_math - total_incorrect_math

  let total_incorrect_science := incorrect_science_marks.sum
  let total_correct_science := correct_science_marks.sum
  let diff_science := total_correct_science - total_incorrect_science

  let incorrect_total_math := avg_math_marks * num_students
  let correct_total_math := incorrect_total_math + diff_math

  let incorrect_total_science := avg_science_marks * num_students
  let correct_total_science := incorrect_total_science + diff_science

  let combined_total := correct_total_math + correct_total_science
  combined_total / (num_students * 2)

theorem correct_combined_average :
  average_marks = 87.1 :=
by
  sorry

end correct_combined_average_l1143_114333


namespace divisible_by_9_l1143_114399

-- Definition of the sum of digits function S
def sum_of_digits (n : ℕ) : ℕ := sorry  -- Assume we have a function that sums the digits of n

theorem divisible_by_9 (a : ℕ) (h₁ : sum_of_digits a = sum_of_digits (2 * a)) 
  (h₂ : a % 9 = sum_of_digits a % 9) (h₃ : (2 * a) % 9 = sum_of_digits (2 * a) % 9) : 
  a % 9 = 0 :=
by
  sorry

end divisible_by_9_l1143_114399


namespace consecutive_odd_numbers_square_difference_l1143_114346

theorem consecutive_odd_numbers_square_difference (a b : ℤ) :
  (a - b = 2 ∨ b - a = 2) → (a^2 - b^2 = 2000) → (a = 501 ∧ b = 499 ∨ a = -501 ∧ b = -499) :=
by 
  intros h1 h2
  sorry

end consecutive_odd_numbers_square_difference_l1143_114346


namespace find_numbers_l1143_114310

def is_solution (a b : ℕ) : Prop :=
  a + b = 432 ∧ (max a b) = 5 * (min a b) ∧ (max a b = 360 ∧ min a b = 72)

theorem find_numbers : ∃ a b : ℕ, is_solution a b :=
by
  sorry

end find_numbers_l1143_114310


namespace initial_volume_solution_l1143_114322

variable (V : ℝ)

theorem initial_volume_solution
  (h1 : 0.35 * V + 1.8 = 0.50 * (V + 1.8)) :
  V = 6 :=
by
  sorry

end initial_volume_solution_l1143_114322


namespace fish_buckets_last_l1143_114335

theorem fish_buckets_last (buckets_sharks : ℕ) (buckets_total : ℕ) 
  (h1 : buckets_sharks = 4)
  (h2 : ∀ (buckets_dolphins : ℕ), buckets_dolphins = buckets_sharks / 2)
  (h3 : ∀ (buckets_other : ℕ), buckets_other = 5 * buckets_sharks)
  (h4 : buckets_total = 546)
  : 546 / ((buckets_sharks + (buckets_sharks / 2) + (5 * buckets_sharks)) * 7) = 3 :=
by
  -- Calculation steps skipped for brevity
  sorry

end fish_buckets_last_l1143_114335


namespace intersection_M_N_l1143_114354

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | |x| > 1}

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l1143_114354


namespace moles_of_HC2H3O2_needed_l1143_114343

theorem moles_of_HC2H3O2_needed :
  (∀ (HC2H3O2 NaHCO3 H2O : ℕ), 
    (HC2H3O2 + NaHCO3 = NaC2H3O2 + H2O + CO2) → 
    (H2O = 3) → 
    (NaHCO3 = 3) → 
    HC2H3O2 = 3) :=
by
  intros HC2H3O2 NaHCO3 H2O h_eq h_H2O h_NaHCO3
  -- Hint: You can use the balanced chemical equation to derive that HC2H3O2 must be 3
  sorry

end moles_of_HC2H3O2_needed_l1143_114343


namespace smallest_x_y_sum_l1143_114368

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hne : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 24) :
  x + y = 100 :=
sorry

end smallest_x_y_sum_l1143_114368


namespace ratio_of_distance_l1143_114315

noncomputable def initial_distance : ℝ := 30 * 20

noncomputable def total_distance : ℝ := 2 * initial_distance

noncomputable def distance_after_storm : ℝ := initial_distance - 200

theorem ratio_of_distance (initial_distance : ℝ) (total_distance : ℝ) (distance_after_storm : ℝ) : 
  distance_after_storm / total_distance = 1 / 3 :=
by
  -- Given conditions
  have h1 : initial_distance = 30 * 20 := by sorry
  have h2 : total_distance = 2 * initial_distance := by sorry
  have h3 : distance_after_storm = initial_distance - 200 := by sorry
  -- Prove the ratio is 1 / 3
  sorry

end ratio_of_distance_l1143_114315


namespace zeros_of_f_on_interval_l1143_114304

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end zeros_of_f_on_interval_l1143_114304


namespace quadratic_roots_range_quadratic_root_condition_l1143_114377

-- Problem 1: Prove that the range of real number \(k\) for which the quadratic 
-- equation \(x^{2} + (2k + 1)x + k^{2} + 1 = 0\) has two distinct real roots is \(k > \frac{3}{4}\). 
theorem quadratic_roots_range (k : ℝ) : 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x^2 + (2*k+1)*x + k^2 + 1 = 0) ↔ (k > 3/4) := 
sorry

-- Problem 2: Given \(k > \frac{3}{4}\), prove that if the roots \(x₁\) and \(x₂\) of 
-- the equation satisfy \( |x₁| + |x₂| = x₁ \cdot x₂ \), then \( k = 2 \).
theorem quadratic_root_condition (k : ℝ) 
    (hk : k > 3 / 4)
    (x₁ x₂ : ℝ)
    (h₁ : x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0)
    (h₂ : x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0)
    (h3 : |x₁| + |x₂| = x₁ * x₂) : 
    k = 2 := 
sorry

end quadratic_roots_range_quadratic_root_condition_l1143_114377


namespace find_value_of_abc_cubed_l1143_114329

-- Variables and conditions
variables {a b c : ℝ}
variables (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4)

-- The statement
theorem find_value_of_abc_cubed (ha : a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0) :
  a^3 + b^3 + c^3 = -3 * a * b * (a + b) :=
by
  sorry

end find_value_of_abc_cubed_l1143_114329


namespace find_f_1991_l1143_114331

namespace FunctionProof

-- Defining the given conditions as statements in Lean
def func_f (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

def poly_g (f g : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, g n = g (f n)

-- Statement of the problem
theorem find_f_1991 
  (f g : ℤ → ℤ)
  (Hf : func_f f)
  (Hg : poly_g f g) :
  f 1991 = -1992 := 
sorry

end FunctionProof

end find_f_1991_l1143_114331


namespace perpendicular_lines_a_value_l1143_114383

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (x + a * y - a = 0) ∧ (a * x - (2 * a - 3) * y - 1 = 0) → x ≠ y) →
  a = 0 ∨ a = 2 :=
sorry

end perpendicular_lines_a_value_l1143_114383


namespace train_length_l1143_114386

theorem train_length (speed_fast speed_slow : ℝ) (time_pass : ℝ)
  (L : ℝ)
  (hf : speed_fast = 46 * (1000/3600))
  (hs : speed_slow = 36 * (1000/3600))
  (ht : time_pass = 36)
  (hL : (2 * L = (speed_fast - speed_slow) * time_pass)) :
  L = 50 := by
  sorry

end train_length_l1143_114386


namespace problem_statement_l1143_114371

def is_ideal_circle (circle : ℝ × ℝ → ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, (circle P = 0 ∧ circle Q = 0) ∧ (abs (l P) = 1 ∧ abs (l Q) = 1)

noncomputable def line_l (p : ℝ × ℝ) : ℝ := 3 * p.1 + 4 * p.2 - 12

noncomputable def circle_D (p : ℝ × ℝ) : ℝ := (p.1 - 4) ^ 2 + (p.2 - 4) ^ 2 - 16

theorem problem_statement : is_ideal_circle circle_D line_l :=
sorry  -- The proof would go here

end problem_statement_l1143_114371


namespace ratio_of_sums_l1143_114337

noncomputable def first_sum : Nat := 
  let sequence := (List.range' 1 15)
  let differences := (List.range' 2 30).map (fun x => 2 * x)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum))

noncomputable def second_sum : Nat :=
  let sequence := (List.range' 1 15)
  let differences := (List.range' 1 29).filterMap (fun x => if x % 2 = 1 then some x else none)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum) - 135)

theorem ratio_of_sums : (first_sum / second_sum : Rat) = (160 / 151 : Rat) :=
  sorry

end ratio_of_sums_l1143_114337


namespace ratio_of_height_to_radius_l1143_114388

theorem ratio_of_height_to_radius (r h : ℝ)
  (h_cone : r > 0 ∧ h > 0)
  (circumference_cone_base : 20 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2))
  : h / r = Real.sqrt 399 := by
  sorry

end ratio_of_height_to_radius_l1143_114388


namespace total_interest_obtained_l1143_114303

-- Define the interest rates and face values
def interest_16 := 0.16 * 100
def interest_12 := 0.12 * 100
def interest_20 := 0.20 * 100

-- State the theorem to be proved
theorem total_interest_obtained : 
  interest_16 + interest_12 + interest_20 = 48 :=
by
  sorry

end total_interest_obtained_l1143_114303


namespace flight_duration_l1143_114306

theorem flight_duration (departure_time arrival_time : ℕ) (time_difference : ℕ) (h m : ℕ) (m_bound : 0 < m ∧ m < 60) 
  (h_val : h = 1) (m_val : m = 35)  : h + m = 36 := by
  sorry

end flight_duration_l1143_114306


namespace largest_base_6_five_digits_l1143_114366

-- Define the base-6 number 55555 in base 10
def base_6_to_base_10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 10000) % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem largest_base_6_five_digits : base_6_to_base_10 55555 = 7775 := by
  sorry

end largest_base_6_five_digits_l1143_114366


namespace eric_time_ratio_l1143_114339

-- Defining the problem context
def eric_runs : ℕ := 20
def eric_jogs : ℕ := 10
def eric_return_time : ℕ := 90

-- The ratio is represented as a fraction
def ratio (a b : ℕ) := a / b

-- Stating the theorem
theorem eric_time_ratio :
  ratio eric_return_time (eric_runs + eric_jogs) = 3 :=
by
  sorry

end eric_time_ratio_l1143_114339


namespace product_of_fractions_l1143_114362

open BigOperators

theorem product_of_fractions :
  (∏ n in Finset.range 9, (n + 2)^3 - 1) / (∏ n in Finset.range 9, (n + 2)^3 + 1) = 74 / 55 :=
by
  sorry

end product_of_fractions_l1143_114362


namespace slices_left_for_lunch_tomorrow_l1143_114324

-- Definitions according to conditions
def initial_slices : ℕ := 12
def slices_eaten_for_lunch := initial_slices / 2
def remaining_slices_after_lunch := initial_slices - slices_eaten_for_lunch
def slices_eaten_for_dinner := 1 / 3 * remaining_slices_after_lunch
def remaining_slices_after_dinner := remaining_slices_after_lunch - slices_eaten_for_dinner
def slices_shared_with_friend := 1 / 4 * remaining_slices_after_dinner
def remaining_slices_after_sharing := remaining_slices_after_dinner - slices_shared_with_friend
def slices_eaten_by_sibling := if (1 / 5 * remaining_slices_after_sharing < 1) then 0 else 1 / 5 * remaining_slices_after_sharing
def remaining_slices_after_sibling := remaining_slices_after_sharing - slices_eaten_by_sibling

-- Lean statement of the proof problem
theorem slices_left_for_lunch_tomorrow : remaining_slices_after_sibling = 3 := by
  sorry

end slices_left_for_lunch_tomorrow_l1143_114324


namespace combination_eq_permutation_div_factorial_l1143_114307

-- Step d): Lean 4 Statement

variables (n k : ℕ)

-- Define combination C_n^k is any k-element subset of an n-element set
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define permutation A_n^k is the number of ways to arrange k elements out of n elements
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Statement to prove: C_n^k = A_n^k / k!
theorem combination_eq_permutation_div_factorial :
  combination n k = permutation n k / (Nat.factorial k) :=
by
  sorry

end combination_eq_permutation_div_factorial_l1143_114307


namespace extremum_value_of_a_g_monotonicity_l1143_114365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

theorem extremum_value_of_a (a : ℝ) (h : (3 * a * (-4 / 3) ^ 2 + 2 * (-4 / 3) = 0)) : a = 1 / 2 :=
by
  -- We need to prove that a = 1 / 2 given the extremum condition.
  sorry

noncomputable def g (x : ℝ) : ℝ := (1 / 2 * x ^ 3 + x ^ 2) * Real.exp x

theorem g_monotonicity :
  (∀ x < -4, deriv g x < 0) ∧
  (∀ x, -4 < x ∧ x < -1 → deriv g x > 0) ∧
  (∀ x, -1 < x ∧ x < 0 → deriv g x < 0) ∧
  (∀ x > 0, deriv g x > 0) :=
by
  -- We need to prove the monotonicity of the function g in the specified intervals.
  sorry

end extremum_value_of_a_g_monotonicity_l1143_114365


namespace correct_avg_and_mode_l1143_114340

-- Define the conditions and correct answers
def avgIncorrect : ℚ := 13.5
def medianIncorrect : ℚ := 12
def modeCorrect : ℚ := 16
def totalNumbers : ℕ := 25
def incorrectNums : List ℚ := [33.5, 47.75, 58.5, 19/2]
def correctNums : List ℚ := [43.5, 56.25, 68.5, 21/2]

noncomputable def correctSum : ℚ := (avgIncorrect * totalNumbers) + (correctNums.sum - incorrectNums.sum)
noncomputable def correctAvg : ℚ := correctSum / totalNumbers

theorem correct_avg_and_mode :
  correctAvg = 367 / 25 ∧ modeCorrect = 16 :=
by
  sorry

end correct_avg_and_mode_l1143_114340


namespace probability_neither_defective_l1143_114359

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def non_defective_pens : ℕ := total_pens - defective_pens
def draw_count : ℕ := 2

def probability_of_non_defective (total : ℕ) (defective : ℕ) (draws : ℕ) : ℚ :=
  let non_defective := total - defective
  (non_defective / total) * ((non_defective - 1) / (total - 1))

theorem probability_neither_defective :
  probability_of_non_defective total_pens defective_pens draw_count = 5 / 14 :=
by sorry

end probability_neither_defective_l1143_114359


namespace sum_of_areas_of_two_parks_l1143_114312

theorem sum_of_areas_of_two_parks :
  let side1 := 11
  let side2 := 5
  let area1 := side1 * side1
  let area2 := side2 * side2
  area1 + area2 = 146 := 
by 
  sorry

end sum_of_areas_of_two_parks_l1143_114312


namespace jake_needs_total_hours_to_pay_off_debts_l1143_114384

-- Define the conditions for the debts and payments
variable (debtA debtB debtC : ℝ)
variable (paymentA paymentB paymentC : ℝ)
variable (task1P task2P task3P task4P task5P task6P : ℝ)
variable (task2Payoff task4Payoff task6Payoff : ℝ)

-- Assume provided values
noncomputable def total_hours_needed : ℝ :=
  let remainingA := debtA - paymentA
  let remainingB := debtB - paymentB
  let remainingC := debtC - paymentC
  let hoursTask1 := (remainingA - task2Payoff) / task1P
  let hoursTask2 := task2Payoff / task2P
  let hoursTask3 := (remainingB - task4Payoff) / task3P
  let hoursTask4 := task4Payoff / task4P
  let hoursTask5 := (remainingC - task6Payoff) / task5P
  let hoursTask6 := task6Payoff / task6P
  hoursTask1 + hoursTask2 + hoursTask3 + hoursTask4 + hoursTask5 + hoursTask6

-- Given our specific problem conditions
theorem jake_needs_total_hours_to_pay_off_debts :
  total_hours_needed 150 200 250 60 80 100 15 12 20 10 25 30 30 40 60 = 20.1 :=
by
  sorry

end jake_needs_total_hours_to_pay_off_debts_l1143_114384


namespace sequence_is_aperiodic_l1143_114311

noncomputable def sequence_a (a : ℕ → ℕ) : Prop :=
∀ k n : ℕ, k < 2^n → a k ≠ a (k + 2^n)

theorem sequence_is_aperiodic (a : ℕ → ℕ) (h_a : sequence_a a) : ¬(∃ p : ℕ, ∀ n k : ℕ, a k = a (k + n * p)) :=
sorry

end sequence_is_aperiodic_l1143_114311


namespace even_fn_increasing_max_val_l1143_114398

variable {f : ℝ → ℝ}

theorem even_fn_increasing_max_val (h_even : ∀ x, f x = f (-x))
    (h_inc_0_5 : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 5 → f x ≤ f y)
    (h_dec_5_inf : ∀ x y, 5 ≤ x → x ≤ y → f y ≤ f x)
    (h_f5 : f 5 = 2) :
    (∀ x y, -5 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y) ∧ (∀ x, -5 ≤ x → x ≤ 0 → f x ≤ 2) :=
by
    sorry

end even_fn_increasing_max_val_l1143_114398


namespace second_part_shorter_l1143_114327

def length_wire : ℕ := 180
def length_part1 : ℕ := 106
def length_part2 : ℕ := length_wire - length_part1
def length_difference : ℕ := length_part1 - length_part2

theorem second_part_shorter :
  length_difference = 32 :=
by
  sorry

end second_part_shorter_l1143_114327


namespace range_of_x_plus_y_l1143_114338

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + 2 * x * y + 4 * y^2 = 1) : 0 < x + y ∧ x + y < 1 :=
by
  sorry

end range_of_x_plus_y_l1143_114338


namespace reciprocal_div_calculate_fraction_reciprocal_div_result_l1143_114387

-- Part 1
theorem reciprocal_div {a b c : ℚ} (h : (a + b) / c = -2) : c / (a + b) = -1 / 2 :=
sorry

-- Part 2
theorem calculate_fraction : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 :=
sorry

-- Part 3
theorem reciprocal_div_result : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 →
 (-1 / 36) / (5 / 12 - 1 / 9 + 2 / 3) = -1 / 35 :=
sorry

end reciprocal_div_calculate_fraction_reciprocal_div_result_l1143_114387


namespace find_c_l1143_114325

variable {a b c : ℝ} 
variable (h_perpendicular : (a / 3) * (-3 / b) = -1)
variable (h_intersect1 : 2 * a + 9 = c)
variable (h_intersect2 : 6 - 3 * b = -c)
variable (h_ab_equal : a = b)

theorem find_c : c = 39 := 
by
  sorry

end find_c_l1143_114325


namespace julie_same_hours_september_october_l1143_114314

-- Define Julie's hourly rates and work hours
def rate_mowing : ℝ := 4
def rate_weeding : ℝ := 8
def september_mowing_hours : ℕ := 25
def september_weeding_hours : ℕ := 3
def total_earnings_september_october : ℤ := 248

-- Define Julie's earnings for each activity and total earnings for September
def september_earnings_mowing : ℝ := september_mowing_hours * rate_mowing
def september_earnings_weeding : ℝ := september_weeding_hours * rate_weeding
def september_total_earnings : ℝ := september_earnings_mowing + september_earnings_weeding

-- Define earnings in October
def october_earnings : ℝ := total_earnings_september_october - september_total_earnings

-- Define the theorem to prove Julie worked the same number of hours in October as in September
theorem julie_same_hours_september_october :
  october_earnings = september_total_earnings :=
by
  sorry

end julie_same_hours_september_october_l1143_114314


namespace sum_of_midpoints_y_coordinates_l1143_114342

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_y_coordinates_l1143_114342


namespace histogram_groups_l1143_114334

theorem histogram_groups 
  (max_height : ℕ)
  (min_height : ℕ)
  (class_interval : ℕ)
  (h_max : max_height = 176)
  (h_min : min_height = 136)
  (h_interval : class_interval = 6) :
  Nat.ceil ((max_height - min_height) / class_interval) = 7 :=
by
  sorry

end histogram_groups_l1143_114334


namespace problem_I_problem_II_l1143_114395

open Set

-- Definitions of the sets A and B, and their intersections would be needed
def A := {x : ℝ | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 3 * a}

-- (I) When a = 1, find A ∩ B
theorem problem_I : A ∩ (B 1) = {x : ℝ | (2 ≤ x ∧ x ≤ 3) ∨ x = 1} := by
  sorry

-- (II) When A ∩ B = B, find the range of a
theorem problem_II : {a : ℝ | a > 0 ∧ ∀ x, x ∈ B a → x ∈ A} = {a : ℝ | (0 < a ∧ a ≤ 1 / 3) ∨ a ≥ 2} := by
  sorry

end problem_I_problem_II_l1143_114395


namespace find_b_value_l1143_114317

theorem find_b_value : 
  ∀ (a b : ℝ), 
    (a^3 * b^4 = 2048) ∧ (a = 8) → b = Real.sqrt 2 := 
by 
sorry

end find_b_value_l1143_114317


namespace students_received_B_l1143_114326

theorem students_received_B (x : ℕ) 
  (h1 : (0.8 * x : ℝ) + x + (1.2 * x : ℝ) = 28) : 
  x = 9 := 
by
  sorry

end students_received_B_l1143_114326


namespace max_xy_of_conditions_l1143_114308

noncomputable def max_xy : ℝ := 37.5

theorem max_xy_of_conditions (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 10 * x + 15 * y = 150) (h4 : x^2 + y^2 ≤ 100) :
  xy ≤ max_xy :=
by sorry

end max_xy_of_conditions_l1143_114308


namespace apples_per_basket_holds_15_l1143_114361

-- Conditions as Definitions
def trees := 10
def total_apples := 3000
def baskets_per_tree := 20

-- Definition for apples per tree (from the given total apples and number of trees)
def apples_per_tree : ℕ := total_apples / trees

-- Definition for apples per basket (from apples per tree and baskets per tree)
def apples_per_basket : ℕ := apples_per_tree / baskets_per_tree

-- The statement to prove the equivalent mathematical problem
theorem apples_per_basket_holds_15 
  (H1 : trees = 10)
  (H2 : total_apples = 3000)
  (H3 : baskets_per_tree = 20) :
  apples_per_basket = 15 :=
by 
  sorry

end apples_per_basket_holds_15_l1143_114361
