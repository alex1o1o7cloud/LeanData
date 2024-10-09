import Mathlib

namespace sum_of_possible_values_of_g1_l1643_164376

def g (x : ℝ) : ℝ := sorry

axiom g_prop : ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - x^2 * y^2

theorem sum_of_possible_values_of_g1 : g 1 = -1 := by sorry

end sum_of_possible_values_of_g1_l1643_164376


namespace simplify_fraction_l1643_164314

theorem simplify_fraction (a b : ℕ) (h1 : a = 252) (h2 : b = 248) :
  (1000 ^ 2 : ℤ) / ((a ^ 2 - b ^ 2) : ℤ) = 500 := by
  sorry

end simplify_fraction_l1643_164314


namespace andrew_purchased_mangoes_l1643_164357

variable (m : ℕ)

def cost_of_grapes := 8 * 70
def cost_of_mangoes (m : ℕ) := 55 * m
def total_cost (m : ℕ) := cost_of_grapes + cost_of_mangoes m

theorem andrew_purchased_mangoes :
  total_cost m = 1055 → m = 9 := by
  intros h_total_cost
  sorry

end andrew_purchased_mangoes_l1643_164357


namespace minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l1643_164339

theorem minimum_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

theorem minimum_value_x_add_2y_achieved (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  ∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 9/y = 1 ∧ x + 2 * y = 19 + 6 * Real.sqrt 2 :=
sorry

end minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l1643_164339


namespace n_is_prime_l1643_164311

theorem n_is_prime (p : ℕ) (h : ℕ) (n : ℕ)
  (hp : Nat.Prime p)
  (hh : h < p)
  (hn : n = p * h + 1)
  (div_n : n ∣ (2^(n-1) - 1))
  (not_div_n : ¬ n ∣ (2^h - 1)) : Nat.Prime n := sorry

end n_is_prime_l1643_164311


namespace sum_of_products_l1643_164324

theorem sum_of_products : 1 * 15 + 2 * 14 + 3 * 13 + 4 * 12 + 5 * 11 + 6 * 10 + 7 * 9 + 8 * 8 = 372 := by
  sorry

end sum_of_products_l1643_164324


namespace count_two_digit_perfect_squares_divisible_by_4_l1643_164346

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l1643_164346


namespace janet_total_l1643_164366

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end janet_total_l1643_164366


namespace quadratic_m_ge_neg2_l1643_164308

-- Define the quadratic equation and condition for real roots
def quadratic_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, (x + 2) ^ 2 = m + 2

-- The theorem to prove
theorem quadratic_m_ge_neg2 (m : ℝ) (h : quadratic_has_real_roots m) : m ≥ -2 :=
by {
  sorry
}

end quadratic_m_ge_neg2_l1643_164308


namespace platform_length_150_l1643_164342

def speed_kmph : ℕ := 54  -- Speed in km/hr

def speed_mps : ℚ := speed_kmph * 1000 / 3600  -- Speed in m/s

def time_pass_man : ℕ := 20  -- Time to pass a man in seconds
def time_pass_platform : ℕ := 30  -- Time to pass a platform in seconds

def length_train : ℚ := speed_mps * time_pass_man  -- Length of the train in meters

def length_platform (P : ℚ) : Prop :=
  length_train + P = speed_mps * time_pass_platform  -- The condition involving platform length

theorem platform_length_150 :
  length_platform 150 := by
  -- We would provide a proof here.
  sorry

end platform_length_150_l1643_164342


namespace repeating_decimal_product_l1643_164385

theorem repeating_decimal_product 
  (x : ℚ) 
  (h1 : x = (0.0126 : ℚ)) 
  (h2 : 9999 * x = 126) 
  (h3 : x = 14 / 1111) : 
  14 * 1111 = 15554 := 
by
  sorry

end repeating_decimal_product_l1643_164385


namespace factor_expression_l1643_164343

theorem factor_expression (x : ℝ) :
  (3*x^3 + 48*x^2 - 14) - (-9*x^3 + 2*x^2 - 14) =
  2*x^2 * (6*x + 23) :=
by
  sorry

end factor_expression_l1643_164343


namespace measure_AX_l1643_164394

-- Definitions based on conditions
def circle_radii (r_A r_B r_C : ℝ) : Prop :=
  r_A - r_B = 6 ∧
  r_A - r_C = 5 ∧
  r_B + r_C = 9

-- Theorem statement
theorem measure_AX (r_A r_B r_C : ℝ) (h : circle_radii r_A r_B r_C) : r_A = 10 :=
by
  sorry

end measure_AX_l1643_164394


namespace polynomial_divisibility_l1643_164336

open Polynomial

noncomputable def f (n : ℕ) : ℤ[X] :=
  (X + 1) ^ (2 * n + 1) + X ^ (n + 2)

noncomputable def p : ℤ[X] :=
  X ^ 2 + X + 1

theorem polynomial_divisibility (n : ℕ) : p ∣ f n :=
  sorry

end polynomial_divisibility_l1643_164336


namespace elephant_distribution_l1643_164303

theorem elephant_distribution (unions nonunions : ℕ) (elephants : ℕ) :
  unions = 28 ∧ nonunions = 37 ∧ (∀ k : ℕ, elephants = 28 * k ∨ elephants = 37 * k) ∧ (∀ k : ℕ, ((28 * k ≤ elephants) ∧ (37 * k ≤ elephants))) → 
  elephants = 2072 :=
by
  sorry

end elephant_distribution_l1643_164303


namespace lemonade_quarts_water_l1643_164330

-- Definitions derived from the conditions
def total_parts := 6 + 2 + 1 -- Sum of all ratio parts
def parts_per_gallon : ℚ := 1.5 / total_parts -- Volume per part in gallons
def parts_per_quart : ℚ := parts_per_gallon * 4 -- Volume per part in quarts
def water_needed : ℚ := 6 * parts_per_quart -- Quarts of water needed

-- Statement to prove
theorem lemonade_quarts_water : water_needed = 4 := 
by sorry

end lemonade_quarts_water_l1643_164330


namespace max_levels_passable_prob_pass_three_levels_l1643_164344

-- Define the condition for passing a level
def passes_level (n : ℕ) (sum : ℕ) : Prop :=
  sum > 2^n

-- Define the maximum sum possible for n dice rolls
def max_sum (n : ℕ) : ℕ :=
  6 * n

-- Define the probability of passing the n-th level
def prob_passing_level (n : ℕ) : ℚ :=
  if n = 1 then 2/3
  else if n = 2 then 5/6
  else if n = 3 then 20/27
  else 0 

-- Combine probabilities for passing the first three levels
def prob_passing_three_levels : ℚ :=
  (2/3) * (5/6) * (20/27)

-- Theorem statement for the maximum number of levels passable
theorem max_levels_passable : 4 = 4 :=
sorry

-- Theorem statement for the probability of passing the first three levels
theorem prob_pass_three_levels : prob_passing_three_levels = 100 / 243 :=
sorry

end max_levels_passable_prob_pass_three_levels_l1643_164344


namespace geometric_sequence_xz_eq_three_l1643_164352

theorem geometric_sequence_xz_eq_three 
  (x y z : ℝ)
  (h1 : ∃ r : ℝ, x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * z = 3 :=
by
  -- skip the proof
  sorry

end geometric_sequence_xz_eq_three_l1643_164352


namespace triple_comp_g_of_2_l1643_164355

def g (n : ℕ) : ℕ :=
  if n ≤ 3 then n^3 - 2 else 4 * n + 1

theorem triple_comp_g_of_2 : g (g (g 2)) = 101 := by
  sorry

end triple_comp_g_of_2_l1643_164355


namespace dog_paws_ground_l1643_164322

theorem dog_paws_ground (total_dogs : ℕ) (two_thirds_back_legs : ℕ) (remaining_dogs_four_legs : ℕ) (two_paws_per_back_leg_dog : ℕ) (four_paws_per_four_leg_dog : ℕ) :
  total_dogs = 24 →
  two_thirds_back_legs = 2 * total_dogs / 3 →
  remaining_dogs_four_legs = total_dogs - two_thirds_back_legs →
  two_paws_per_back_leg_dog = 2 →
  four_paws_per_four_leg_dog = 4 →
  (two_thirds_back_legs * two_paws_per_back_leg_dog + remaining_dogs_four_legs * four_paws_per_four_leg_dog) = 64 := 
by 
  sorry

end dog_paws_ground_l1643_164322


namespace exists_perfect_square_of_the_form_l1643_164341

theorem exists_perfect_square_of_the_form (k : ℕ) (h : k > 0) : ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * m = n * 2^k - 7 :=
by sorry

end exists_perfect_square_of_the_form_l1643_164341


namespace initial_investment_amount_l1643_164306

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment_amount (P A r t : ℝ) (n : ℕ) (hA : A = 992.25) 
  (hr : r = 0.10) (hn : n = 2) (ht : t = 1) : P = 900 :=
by
  have h : compoundInterest P r n t = A := by sorry
  rw [hA, hr, hn, ht] at h
  simp at h
  exact sorry

end initial_investment_amount_l1643_164306


namespace evaluate_neg_64_pow_two_thirds_l1643_164381

theorem evaluate_neg_64_pow_two_thirds 
  (h : -64 = (-4)^3) : (-64)^(2/3) = 16 := 
by 
  -- sorry added to skip the proof.
  sorry  

end evaluate_neg_64_pow_two_thirds_l1643_164381


namespace triangle_area_is_correct_l1643_164345

noncomputable def triangle_area : ℝ :=
  let A := (3, 3)
  let B := (4.5, 7.5)
  let C := (7.5, 4.5)
  1 / 2 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℝ)|

theorem triangle_area_is_correct : triangle_area = 9 := by
  sorry

end triangle_area_is_correct_l1643_164345


namespace greenville_height_of_boxes_l1643_164379

theorem greenville_height_of_boxes:
  ∃ h : ℝ, 
    (20 * 20 * h) * (2160000 / (20 * 20 * h)) * 0.40 = 180 ∧ 
    400 * h = 2160000 / (2160000 / (20 * 20 * h)) ∧
    400 * h = 5400 ∧
    h = 12 :=
    sorry

end greenville_height_of_boxes_l1643_164379


namespace compute_cos_l1643_164383

noncomputable def angle1 (A C B : ℝ) : Prop := A + C = 2 * B
noncomputable def angle2 (A C B : ℝ) : Prop := 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B

theorem compute_cos (A B C : ℝ) (h1 : angle1 A C B) (h2 : angle2 A C B) : 
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 :=
sorry

end compute_cos_l1643_164383


namespace dorothy_will_be_twice_as_old_l1643_164374

-- Define some variables
variables (D S Y : ℕ)

-- Hypothesis
def dorothy_age_condition (D S : ℕ) : Prop := D = 3 * S
def dorothy_current_age (D : ℕ) : Prop := D = 15

-- Theorems we want to prove
theorem dorothy_will_be_twice_as_old (D S Y : ℕ) 
  (h1 : dorothy_age_condition D S)
  (h2 : dorothy_current_age D)
  (h3 : D = 15)
  (h4 : S = 5)
  (h5 : D + Y = 2 * (S + Y)) : Y = 5 := 
sorry

end dorothy_will_be_twice_as_old_l1643_164374


namespace carA_travel_time_l1643_164331

theorem carA_travel_time 
    (speedA speedB distanceB : ℕ)
    (ratio : ℕ)
    (timeB : ℕ)
    (h_speedA : speedA = 50)
    (h_speedB : speedB = 100)
    (h_distanceB : distanceB = speedB * timeB)
    (h_ratio : distanceA / distanceB = ratio)
    (h_ratio_value : ratio = 3)
    (h_timeB : timeB = 1)
  : distanceA / speedA = 6 :=
by sorry

end carA_travel_time_l1643_164331


namespace least_number_to_add_l1643_164320

theorem least_number_to_add (a b n : ℕ) (h₁ : a = 1056) (h₂ : b = 29) (h₃ : (a + n) % b = 0) : n = 17 :=
sorry

end least_number_to_add_l1643_164320


namespace curve_distance_bound_l1643_164315

/--
Given the point A on the curve y = e^x and point B on the curve y = ln(x),
prove that |AB| >= a always holds if and only if a <= sqrt(2).
-/
theorem curve_distance_bound {A B : ℝ × ℝ} (a : ℝ)
  (hA : A.2 = Real.exp A.1) (hB : B.2 = Real.log B.1) :
  (dist A B ≥ a) ↔ (a ≤ Real.sqrt 2) :=
sorry

end curve_distance_bound_l1643_164315


namespace abs_fraction_inequality_solution_l1643_164356

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l1643_164356


namespace divisibility_l1643_164337

def Q (X : ℤ) := (X - 1) ^ 3

def P_n (n : ℕ) (X : ℤ) : ℤ :=
  n * X ^ (n + 2) - (n + 2) * X ^ (n + 1) + (n + 2) * X - n

theorem divisibility (n : ℕ) (h : n > 0) : ∀ X : ℤ, Q X ∣ P_n n X :=
by
  sorry

end divisibility_l1643_164337


namespace men_work_problem_l1643_164307

theorem men_work_problem (x : ℕ) (h1 : x * 70 = 40 * 63) : x = 36 := 
by
  sorry

end men_work_problem_l1643_164307


namespace range_of_a_l1643_164371

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ f 0 a) ↔ 0 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l1643_164371


namespace percentage_bob_is_36_l1643_164340

def water_per_acre_corn : ℕ := 20
def water_per_acre_cotton : ℕ := 80
def water_per_acre_beans : ℕ := 2 * water_per_acre_corn

def acres_bob_corn : ℕ := 3
def acres_bob_cotton : ℕ := 9
def acres_bob_beans : ℕ := 12

def acres_brenda_corn : ℕ := 6
def acres_brenda_cotton : ℕ := 7
def acres_brenda_beans : ℕ := 14

def acres_bernie_corn : ℕ := 2
def acres_bernie_cotton : ℕ := 12

def water_bob : ℕ := (acres_bob_corn * water_per_acre_corn) +
                      (acres_bob_cotton * water_per_acre_cotton) +
                      (acres_bob_beans * water_per_acre_beans)

def water_brenda : ℕ := (acres_brenda_corn * water_per_acre_corn) +
                         (acres_brenda_cotton * water_per_acre_cotton) +
                         (acres_brenda_beans * water_per_acre_beans)

def water_bernie : ℕ := (acres_bernie_corn * water_per_acre_corn) +
                         (acres_bernie_cotton * water_per_acre_cotton)

def total_water : ℕ := water_bob + water_brenda + water_bernie

def percentage_bob : ℚ := (water_bob : ℚ) / (total_water : ℚ) * 100

theorem percentage_bob_is_36 : percentage_bob = 36 := by
  sorry

end percentage_bob_is_36_l1643_164340


namespace pipe_fill_time_without_leakage_l1643_164393

theorem pipe_fill_time_without_leakage (t : ℕ) (h1 : 7 * t * (1/t - 1/70) = 1) : t = 60 :=
by
  sorry

end pipe_fill_time_without_leakage_l1643_164393


namespace log_inequality_l1643_164362

theorem log_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
    (Real.log (c ^ 2) / Real.log (a + b) + Real.log (a ^ 2) / Real.log (b + c) + Real.log (b ^ 2) / Real.log (c + a)) ≥ 3 :=
sorry

end log_inequality_l1643_164362


namespace distance_from_origin_to_midpoint_l1643_164373

theorem distance_from_origin_to_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 10) → (y1 = 20) → (x2 = -10) → (y2 = -20) → 
  dist (0 : ℝ × ℝ) ((x1 + x2) / 2, (y1 + y2) / 2) = 0 := 
by
  intros x1 y1 x2 y2 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- remaining proof goes here
  sorry

end distance_from_origin_to_midpoint_l1643_164373


namespace oldest_daily_cheese_l1643_164317

-- Given conditions
def days_per_week : ℕ := 5
def weeks : ℕ := 4
def youngest_daily : ℕ := 1
def cheeses_per_pack : ℕ := 30
def packs_needed : ℕ := 2

-- Derived conditions
def total_days : ℕ := days_per_week * weeks
def total_cheeses : ℕ := packs_needed * cheeses_per_pack
def youngest_total_cheeses : ℕ := youngest_daily * total_days
def oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses

-- Prove that the oldest child wants 2 string cheeses per day
theorem oldest_daily_cheese : oldest_total_cheeses / total_days = 2 := by
  sorry

end oldest_daily_cheese_l1643_164317


namespace length_of_AC_l1643_164347

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end length_of_AC_l1643_164347


namespace seventh_term_of_arithmetic_sequence_l1643_164353

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : 5 * a + 10 * d = 35)
  (h2 : a + 5 * d = 10) :
  a + 6 * d = 11 :=
by
  sorry

end seventh_term_of_arithmetic_sequence_l1643_164353


namespace prove_heron_formula_prove_S_squared_rrarc_l1643_164301

variables {r r_a r_b r_c p a b c S : ℝ}

-- Problem 1: Prove Heron's Formula
theorem prove_heron_formula (h1 : r * p = r_a * (p - a))
                            (h2 : r * r_a = (p - b) * (p - c))
                            (h3 : r_b * r_c = p * (p - a)) :
  S^2 = p * (p - a) * (p - b) * (p - c) :=
sorry

-- Problem 2: Prove S^2 = r * r_a * r_b * r_c
theorem prove_S_squared_rrarc (h1 : r * p = r_a * (p - a))
                              (h2 : r * r_a = (p - b) * (p - c))
                              (h3 : r_b * r_c = p * (p - a)) :
  S^2 = r * r_a * r_b * r_c :=
sorry

end prove_heron_formula_prove_S_squared_rrarc_l1643_164301


namespace exists_integers_a_b_part_a_l1643_164392

theorem exists_integers_a_b_part_a : 
  ∃ a b : ℤ, (∀ x : ℝ, x^2 + a * x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a * x + (b : ℝ) = 0) := 
sorry

end exists_integers_a_b_part_a_l1643_164392


namespace q_at_2_l1643_164370

noncomputable def q (x : ℝ) : ℝ :=
  Real.sign (3 * x - 2) * |3 * x - 2|^(1/4) +
  2 * Real.sign (3 * x - 2) * |3 * x - 2|^(1/6) +
  |3 * x - 2|^(1/8)

theorem q_at_2 : q 2 = 4 := by
  -- Proof attempt needed
  sorry

end q_at_2_l1643_164370


namespace calculate_unshaded_perimeter_l1643_164332

-- Defining the problem's conditions and results.
def total_length : ℕ := 20
def total_width : ℕ := 12
def shaded_area : ℕ := 65
def inner_shaded_width : ℕ := 5
def total_area : ℕ := total_length * total_width
def unshaded_area : ℕ := total_area - shaded_area

-- Define dimensions for the unshaded region based on the problem conditions.
def unshaded_width : ℕ := total_width - inner_shaded_width
def unshaded_height : ℕ := unshaded_area / unshaded_width

-- Calculate perimeter of the unshaded region.
def unshaded_perimeter : ℕ := 2 * (unshaded_width + unshaded_height)

-- Stating the theorem to be proved.
theorem calculate_unshaded_perimeter : unshaded_perimeter = 64 := 
sorry

end calculate_unshaded_perimeter_l1643_164332


namespace problem_solution_l1643_164333

noncomputable def circles_intersect (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (A ∈ { p | p.1^2 + p.2^2 = 1 }) ∧ (B ∈ { p | p.1^2 + p.2^2 = 1 }) ∧
  (A ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ (B ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ 
  (dist A B = (4 * Real.sqrt 5) / 5)

theorem problem_solution (m : ℝ) : circles_intersect m ↔ (m = 1 ∨ m = -3) := by
  sorry

end problem_solution_l1643_164333


namespace f_at_1_over_11_l1643_164398

noncomputable def f : (ℝ → ℝ) := sorry

axiom f_domain : ∀ x, 0 < x → 0 < f x

axiom f_eq : ∀ x y, 0 < x → 0 < y → 10 * ((x + y) / (x * y)) = (f x) * (f y) - f (x * y) - 90

theorem f_at_1_over_11 : f (1 / 11) = 21 := by
  -- proof is omitted
  sorry

end f_at_1_over_11_l1643_164398


namespace repeating_decimal_to_fraction_l1643_164364

noncomputable def repeating_decimal := 0.6 + 3 / 100

theorem repeating_decimal_to_fraction :
  repeating_decimal = 19 / 30 :=
  sorry

end repeating_decimal_to_fraction_l1643_164364


namespace greatest_natural_number_l1643_164375

theorem greatest_natural_number (n q r : ℕ) (h1 : n = 91 * q + r)
  (h2 : r = q^2) (h3 : r < 91) : n = 900 :=
sorry

end greatest_natural_number_l1643_164375


namespace polygon_sides_count_l1643_164321

def sides_square : ℕ := 4
def sides_triangle : ℕ := 3
def sides_hexagon : ℕ := 6
def sides_heptagon : ℕ := 7
def sides_octagon : ℕ := 8
def sides_nonagon : ℕ := 9

def total_sides_exposed : ℕ :=
  let adjacent_1side := sides_square + sides_nonagon - 2 * 1
  let adjacent_2sides :=
    sides_triangle + sides_hexagon +
    sides_heptagon + sides_octagon - 4 * 2
  adjacent_1side + adjacent_2sides

theorem polygon_sides_count : total_sides_exposed = 27 := by
  sorry

end polygon_sides_count_l1643_164321


namespace inequality_holds_l1643_164309

theorem inequality_holds (a b : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (hb : 0 ≤ b) (hb' : b ≤ 1) : 
  a^5 + b^3 + (a - b)^2 ≤ 2 :=
sorry

end inequality_holds_l1643_164309


namespace problem1_problem2_l1643_164335

section proof_problem

-- Define the sets as predicate functions
def A (x : ℝ) : Prop := x > 1
def B (x : ℝ) : Prop := -2 < x ∧ x < 2
def C (x : ℝ) : Prop := -3 < x ∧ x < 5

-- Define the union and intersection of sets
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x
def inter (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x

-- Proving that (A ∪ B) ∩ C = {x | -2 < x < 5}
theorem problem1 : ∀ x, (inter (union A B) C) x ↔ (-2 < x ∧ x < 5) := 
by
  sorry

-- Proving the arithmetic expression result
theorem problem2 : 
  ((2 + 1/4) ^ (1/2)) - ((-9.6) ^ 0) - ((3 + 3/8) ^ (-2/3)) + ((1.5) ^ (-2)) = 1/2 := 
by
  sorry

end proof_problem

end problem1_problem2_l1643_164335


namespace more_cats_than_dogs_l1643_164326

-- Define the initial conditions
def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_adopted : ℕ := 3

-- Compute the number of cats after adoption
def cats_now : ℕ := initial_cats - cats_adopted

-- Define the target statement
theorem more_cats_than_dogs : cats_now - initial_dogs = 7 := by
  unfold cats_now
  unfold initial_cats
  unfold cats_adopted
  unfold initial_dogs
  sorry

end more_cats_than_dogs_l1643_164326


namespace pants_cost_l1643_164361

/-- Given:
- 3 skirts with each costing $20.00
- 5 blouses with each costing $15.00
- The total spending is $180.00
- A discount on pants: buy 1 pair get 1 pair 1/2 off

Prove that each pair of pants costs $30.00 before the discount. --/
theorem pants_cost (cost_skirt cost_blouse total_amount : ℤ) (pants_discount: ℚ) (total_cost: ℤ) :
  cost_skirt = 20 ∧ cost_blouse = 15 ∧ total_amount = 180 
  ∧ pants_discount * 2 = 1 
  ∧ total_cost = 3 * cost_skirt + 5 * cost_blouse + 3/2 * pants_discount → 
  pants_discount = 30 := by
  sorry

end pants_cost_l1643_164361


namespace part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l1643_164350

-- Definition of a good number
def is_good (n : ℕ) : Prop := (n % 6 = 3)

-- Lean 4 statements

-- 1. 2001 is good
theorem part_a_2001_good : is_good 2001 :=
by sorry

-- 2. 3001 isn't good
theorem part_a_3001_not_good : ¬ is_good 3001 :=
by sorry

-- 3. The product of two good numbers is a good number
theorem part_b_product_of_good_is_good (x y : ℕ) (hx : is_good x) (hy : is_good y) : is_good (x * y) :=
by sorry

-- 4. If the product of two numbers is good, then at least one of the numbers is good
theorem part_c_product_good_then_one_good (x y : ℕ) (hxy : is_good (x * y)) : is_good x ∨ is_good y :=
by sorry

end part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l1643_164350


namespace abc_gt_16_abc_geq_3125_div_108_l1643_164388

variables {a b c α β : ℝ}

-- Define the conditions
def conditions (a b c α β : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b > 0 ∧
  (a * α^2 + b * α - c = 0) ∧
  (a * β^2 + b * β - c = 0) ∧
  (α ≠ β) ∧
  (α^3 + b * α^2 + a * α - c = 0) ∧
  (β^3 + b * β^2 + a * β - c = 0)

-- State the first proof problem
theorem abc_gt_16 (h : conditions a b c α β) : a * b * c > 16 :=
sorry

-- State the second proof problem
theorem abc_geq_3125_div_108 (h : conditions a b c α β) : a * b * c ≥ 3125 / 108 :=
sorry

end abc_gt_16_abc_geq_3125_div_108_l1643_164388


namespace math_problem_l1643_164319

noncomputable def proof_statement : Prop :=
  ∃ (a b m : ℝ),
    0 < a ∧ 0 < b ∧ 0 < m ∧
    (5 = m^2 * ((a^2 / b^2) + (b^2 / a^2)) + m * (a/b + b/a)) ∧
    m = (-1 + Real.sqrt 21) / 2

theorem math_problem : proof_statement :=
  sorry

end math_problem_l1643_164319


namespace find_f_inv_128_l1643_164368

noncomputable def f : ℕ → ℕ := sorry

axiom f_at_5 : f 5 = 2
axiom f_doubling : ∀ x : ℕ, f (2 * x) = 2 * f x

theorem find_f_inv_128 : f 320 = 128 :=
by sorry

end find_f_inv_128_l1643_164368


namespace polynomial_no_strictly_positive_roots_l1643_164338

-- Define the necessary conditions and prove the main statement

variables (n : ℕ)
variables (a : Fin n → ℕ) (k : ℕ) (M : ℕ)

-- Axioms/Conditions
axiom pos_a (i : Fin n) : 0 < a i
axiom pos_k : 0 < k
axiom pos_M : 0 < M
axiom M_gt_1 : M > 1

axiom sum_reciprocals : (Finset.univ.sum (λ i => (1 : ℚ) / a i)) = k
axiom product_a : (Finset.univ.prod a) = M

noncomputable def polynomial_has_no_positive_roots : Prop :=
  ∀ x : ℝ, 0 < x →
    M * (1 + x)^k > (Finset.univ.prod (λ i => x + a i))

theorem polynomial_no_strictly_positive_roots (h : polynomial_has_no_positive_roots n a k M) : 
  ∀ x : ℝ, 0 < x → (M * (1 + x)^k - (Finset.univ.prod (λ i => x + a i)) ≠ 0) :=
by
  sorry

end polynomial_no_strictly_positive_roots_l1643_164338


namespace vertical_asymptote_unique_d_values_l1643_164323

theorem vertical_asymptote_unique_d_values (d : ℝ) :
  (∃! x : ℝ, ∃ c : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 - 2*x + d) = 0) ↔ (d = 0 ∨ d = -3) := 
sorry

end vertical_asymptote_unique_d_values_l1643_164323


namespace DVDs_sold_168_l1643_164359

-- Definitions of the conditions
def CDs_sold := ℤ
def DVDs_sold := ℤ

def ratio_condition (C D : ℤ) : Prop := D = 16 * C / 10
def total_condition (C D : ℤ) : Prop := D + C = 273

-- The main statement to prove
theorem DVDs_sold_168 (C D : ℤ) 
  (h1 : ratio_condition C D) 
  (h2 : total_condition C D) : D = 168 :=
sorry

end DVDs_sold_168_l1643_164359


namespace amount_paid_Y_l1643_164395

theorem amount_paid_Y (X Y : ℝ) (h1 : X + Y = 330) (h2 : X = 1.2 * Y) : Y = 150 := 
by
  sorry

end amount_paid_Y_l1643_164395


namespace sages_success_l1643_164390

-- Assume we have a finite type representing our 1000 colors
inductive Color
| mk : Fin 1000 → Color

open Color

-- Define the sages
def Sage : Type := Fin 11

-- Define the problem conditions into a Lean structure
structure Problem :=
  (sages : Fin 11)
  (colors : Fin 1000)
  (assignments : Sage → Color)
  (strategies : Sage → (Fin 1024 → Fin 2))

-- Define the success condition
def success (p : Problem) : Prop :=
  ∃ (strategies : Sage → (Fin 1024 → Fin 2)),
    ∀ (assignment : Sage → Color),
      ∃ (color_guesses : Sage → Color),
        (∀ s, color_guesses s = assignment s)

-- The sages will succeed in determining the colors of their hats.
theorem sages_success : ∀ (p : Problem), success p := by
  sorry

end sages_success_l1643_164390


namespace molecular_weight_al_fluoride_l1643_164396

/-- Proving the molecular weight of Aluminum fluoride calculation -/
theorem molecular_weight_al_fluoride (x : ℕ) (h : 10 * x = 840) : x = 84 :=
by sorry

end molecular_weight_al_fluoride_l1643_164396


namespace remainder_47_mod_288_is_23_mod_24_l1643_164328

theorem remainder_47_mod_288_is_23_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := 
sorry

end remainder_47_mod_288_is_23_mod_24_l1643_164328


namespace relationship_a_b_l1643_164300

theorem relationship_a_b
  (m a b : ℝ)
  (h1 : ∃ m, ∀ x, -2 * x + m = y)
  (h2 : ∃ x₁ y₁, (x₁ = -2) ∧ (y₁ = a) ∧ (-2 * x₁ + m = y₁))
  (h3 : ∃ x₂ y₂, (x₂ = 2) ∧ (y₂ = b) ∧ (-2 * x₂ + m = y₂)) :
  a > b :=
sorry

end relationship_a_b_l1643_164300


namespace simplify_expression_l1643_164372

theorem simplify_expression (x : ℤ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 24 = 45 * x + 42 := 
by 
  -- proof steps
  sorry

end simplify_expression_l1643_164372


namespace option_B_more_cost_effective_l1643_164354

def cost_option_A (x : ℕ) : ℕ := 60 + 18 * x
def cost_option_B (x : ℕ) : ℕ := 150 + 15 * x
def x : ℕ := 40

theorem option_B_more_cost_effective : cost_option_B x < cost_option_A x := by
  -- Placeholder for the proof steps
  sorry

end option_B_more_cost_effective_l1643_164354


namespace polygon_sides_eq_eight_l1643_164363

theorem polygon_sides_eq_eight (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
sorry

end polygon_sides_eq_eight_l1643_164363


namespace range_of_a_l1643_164365

theorem range_of_a 
  (a : ℝ) 
  (h₀ : ∀ x : ℝ, (3 ≤ x ∧ x ≤ 4) ↔ (y = 2 * x + (3 - a))) : 
  9 ≤ a ∧ a ≤ 11 := 
sorry

end range_of_a_l1643_164365


namespace vendor_has_1512_liters_of_sprite_l1643_164367

-- Define the conditions
def liters_of_maaza := 60
def liters_of_pepsi := 144
def least_number_of_cans := 143
def gcd_maaza_pepsi := Nat.gcd liters_of_maaza liters_of_pepsi --let Lean compute GCD

-- Define the liters per can as the GCD of Maaza and Pepsi
def liters_per_can := gcd_maaza_pepsi

-- Define the number of cans for Maaza and Pepsi respectively
def cans_of_maaza := liters_of_maaza / liters_per_can
def cans_of_pepsi := liters_of_pepsi / liters_per_can

-- Define total cans for Maaza and Pepsi
def total_cans_for_maaza_and_pepsi := cans_of_maaza + cans_of_pepsi

-- Define the number of cans for Sprite
def cans_of_sprite := least_number_of_cans - total_cans_for_maaza_and_pepsi

-- The total liters of Sprite the vendor has
def liters_of_sprite := cans_of_sprite * liters_per_can

-- Statement to prove
theorem vendor_has_1512_liters_of_sprite : 
  liters_of_sprite = 1512 :=
by
  -- solution omitted 
  sorry

end vendor_has_1512_liters_of_sprite_l1643_164367


namespace greater_number_is_64_l1643_164313

-- Proof statement: The greater number (y) is 64 given the conditions
theorem greater_number_is_64 (x y : ℕ) 
    (h1 : y = 2 * x) 
    (h2 : x + y = 96) : 
    y = 64 := 
sorry

end greater_number_is_64_l1643_164313


namespace chain_of_inequalities_l1643_164327

theorem chain_of_inequalities (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  9 / (a + b + c) ≤ (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ∧ 
  (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ≤ (1 / a + 1 / b + 1 / c) := 
by 
  sorry

end chain_of_inequalities_l1643_164327


namespace math_problem_l1643_164349

theorem math_problem (a b c d x : ℝ)
  (h1 : a = -(-b))
  (h2 : c = -1 / d)
  (h3 : |x| = 3) :
  x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36 :=
by sorry

end math_problem_l1643_164349


namespace factorization_correct_l1643_164304

theorem factorization_correct : ∀ (x : ℕ), x^2 - x = x * (x - 1) :=
by
  intro x
  -- We know the problem reduces to algebraic identity proof
  sorry

end factorization_correct_l1643_164304


namespace number_of_buckets_after_reduction_l1643_164369

def initial_buckets : ℕ := 25
def reduction_factor : ℚ := 2 / 5

theorem number_of_buckets_after_reduction :
  (initial_buckets : ℚ) * (1 / reduction_factor) = 63 := by
  sorry

end number_of_buckets_after_reduction_l1643_164369


namespace monthly_rent_calc_l1643_164377

def monthly_rent (length width annual_rent_per_sq_ft : ℕ) : ℕ :=
  (length * width * annual_rent_per_sq_ft) / 12

theorem monthly_rent_calc :
  monthly_rent 10 8 360 = 2400 := 
  sorry

end monthly_rent_calc_l1643_164377


namespace rate_of_change_l1643_164310

noncomputable def radius : ℝ := 12
noncomputable def θ (t : ℝ) : ℝ := (38 + 5 * t) * (Real.pi / 180)
noncomputable def area (t : ℝ) : ℝ := (1/2) * radius^2 * θ t

theorem rate_of_change (t : ℝ) : deriv area t = 2 * Real.pi :=
by
  sorry

end rate_of_change_l1643_164310


namespace pens_to_sell_to_make_profit_l1643_164334

theorem pens_to_sell_to_make_profit (initial_pens : ℕ) (purchase_price selling_price profit : ℝ) :
  initial_pens = 2000 →
  purchase_price = 0.15 →
  selling_price = 0.30 →
  profit = 150 →
  (initial_pens * selling_price - initial_pens * purchase_price = profit) →
  initial_pens * profit / (selling_price - purchase_price) = 1500 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pens_to_sell_to_make_profit_l1643_164334


namespace minimum_toothpicks_removal_l1643_164312

theorem minimum_toothpicks_removal
  (total_toothpicks : ℕ)
  (grid_size : ℕ)
  (toothpicks_per_square : ℕ)
  (shared_sides : ℕ)
  (interior_toothpicks : ℕ) 
  (diagonal_toothpicks : ℕ)
  (min_removal : ℕ) 
  (no_squares_or_triangles : Bool)
  (h1 : total_toothpicks = 40)
  (h2 : grid_size = 3)
  (h3 : toothpicks_per_square = 4)
  (h4 : shared_sides = 16)
  (h5 : interior_toothpicks = 16) 
  (h6 : diagonal_toothpicks = 12)
  (h7 : min_removal = 16)
: no_squares_or_triangles := 
sorry

end minimum_toothpicks_removal_l1643_164312


namespace volume_of_circumscribed_polyhedron_l1643_164387

theorem volume_of_circumscribed_polyhedron (R : ℝ) (V : ℝ) (S_n : ℝ) (h : Π (F_i : ℝ), V = (1/3) * S_n * R) : V = (1/3) * S_n * R :=
sorry

end volume_of_circumscribed_polyhedron_l1643_164387


namespace f_lt_2_l1643_164318

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f (x + 2) = f (-x + 2)

axiom f_ge_2 (x : ℝ) (h : x ≥ 2) : f x = x^2 - 6 * x + 4

theorem f_lt_2 (x : ℝ) (h : x < 2) : f x = x^2 - 2 * x - 4 :=
by
  sorry

end f_lt_2_l1643_164318


namespace symmetric_point_coords_l1643_164305

def pointA : ℝ × ℝ := (1, 2)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def pointB : ℝ × ℝ := translate_left pointA 2

def pointC : ℝ × ℝ := reflect_origin pointB

theorem symmetric_point_coords :
  pointC = (1, -2) :=
by
  -- Proof omitted as instructed
  sorry

end symmetric_point_coords_l1643_164305


namespace min_pairs_l1643_164391

-- Define the types for knights and liars
inductive Residents
| Knight : Residents
| Liar : Residents

def total_residents : ℕ := 200
def knights : ℕ := 100
def liars : ℕ := 100

-- Additional conditions
def conditions (friend_claims_knights friend_claims_liars : ℕ) : Prop :=
  friend_claims_knights = 100 ∧
  friend_claims_liars = 100 ∧
  knights + liars = total_residents

-- Minimum number of knight-liar pairs to prove
def min_knight_liar_pairs : ℕ := 50

theorem min_pairs {friend_claims_knights friend_claims_liars : ℕ} (h : conditions friend_claims_knights friend_claims_liars) :
    min_knight_liar_pairs = 50 :=
sorry

end min_pairs_l1643_164391


namespace determine_x_value_l1643_164302

variable {a b x r : ℝ}
variable (b_nonzero : b ≠ 0)

theorem determine_x_value (h1 : r = (3 * a)^(3 * b)) (h2 : r = a^b * x^b) : x = 27 * a^2 :=
by
  sorry

end determine_x_value_l1643_164302


namespace usual_time_is_49_l1643_164360

variable (R T : ℝ)
variable (h1 : R > 0) -- Usual rate is positive
variable (h2 : T > 0) -- Usual time is positive
variable (condition : T * R = (T - 7) * (7 / 6 * R)) -- Main condition derived from the problem

theorem usual_time_is_49 (h1 : R > 0) (h2 : T > 0) (condition : T * R = (T - 7) * (7 / 6 * R)) : T = 49 := by
  sorry -- Proof goes here

end usual_time_is_49_l1643_164360


namespace only_root_is_4_l1643_164316

noncomputable def equation_one (x : ℝ) : ℝ := (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1

noncomputable def equation_two (x : ℝ) : ℝ := x^2 - 5 * x + 4

theorem only_root_is_4 (x : ℝ) (h: equation_one x = 0) (h_transformation: equation_two x = 0) : x = 4 := sorry

end only_root_is_4_l1643_164316


namespace initial_number_of_quarters_l1643_164389

theorem initial_number_of_quarters 
  (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (half_dollars : ℕ) (dollar_coins : ℕ) 
  (two_dollar_coins : ℕ) (quarters : ℕ)
  (cost_per_sundae : ℝ) 
  (special_topping_cost : ℝ)
  (featured_flavor_discount : ℝ)
  (members_with_special_topping : ℕ)
  (members_with_featured_flavor : ℕ)
  (left_over : ℝ)
  (expected_quarters : ℕ) :
  pennies = 123 ∧
  nickels = 85 ∧
  dimes = 35 ∧
  half_dollars = 15 ∧
  dollar_coins = 5 ∧
  quarters = expected_quarters ∧
  two_dollar_coins = 4 ∧
  cost_per_sundae = 5.25 ∧
  special_topping_cost = 0.50 ∧
  featured_flavor_discount = 0.25 ∧
  members_with_special_topping = 3 ∧
  members_with_featured_flavor = 5 ∧
  left_over = 0.97 →
  expected_quarters = 54 :=
  by
  sorry

end initial_number_of_quarters_l1643_164389


namespace a_minus_b_7_l1643_164329

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end a_minus_b_7_l1643_164329


namespace range_func_l1643_164325

noncomputable def func (x : ℝ) : ℝ := x + 4 / x

theorem range_func (x : ℝ) (hx : x ≠ 0) : func x ≤ -4 ∨ func x ≥ 4 := by
  sorry

end range_func_l1643_164325


namespace eccentricity_is_two_l1643_164384

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_is_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : 
  eccentricity_of_hyperbola a b h1 h2 h3 = 2 := 
  sorry

end eccentricity_is_two_l1643_164384


namespace arithmetic_expression_value_l1643_164397

def mixed_to_frac (a b c : ℕ) : ℚ := a + b / c

theorem arithmetic_expression_value :
  ( ( (mixed_to_frac 5 4 45 - mixed_to_frac 4 1 6) / mixed_to_frac 5 8 15 ) / 
    ( (mixed_to_frac 4 2 3 + 3 / 4) * mixed_to_frac 3 9 13 ) * mixed_to_frac 34 2 7 + 
    (3 / 10 / (1 / 100) / 70) + 2 / 7 ) = 1 :=
by
  -- We need to convert the mixed numbers to fractions using mixed_to_frac
  -- Then, we simplify step-by-step as in the problem solution, but for now we just use sorry
  sorry

end arithmetic_expression_value_l1643_164397


namespace find_value_of_expression_l1643_164358

theorem find_value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2 * m * n = 3) 
  (h2 : m * n + n^2 = 4) : 
  m^2 + 3 * m * n + n^2 = 7 := 
by
  sorry

end find_value_of_expression_l1643_164358


namespace smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l1643_164348

noncomputable def smaller_angle_at_715 : ℝ :=
  let hour_position := 7 * 30 + 30 / 4
  let minute_position := 15 * (360 / 60)
  let angle_between := abs (hour_position - minute_position)
  if angle_between > 180 then 360 - angle_between else angle_between

theorem smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m :
  smaller_angle_at_715 = 127.5 := 
sorry

end smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l1643_164348


namespace janina_spend_on_supplies_each_day_l1643_164380

theorem janina_spend_on_supplies_each_day 
  (rent : ℝ)
  (p : ℝ)
  (n : ℕ)
  (H1 : rent = 30)
  (H2 : p = 2)
  (H3 : n = 21) :
  (n : ℝ) * p - rent = 12 := 
by
  sorry

end janina_spend_on_supplies_each_day_l1643_164380


namespace sequence_formula_l1643_164351

theorem sequence_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 3 * a n + 3 ^ n) → 
  ∀ n : ℕ, 0 < n → a n = n * 3 ^ (n - 1) :=
by
  sorry

end sequence_formula_l1643_164351


namespace prime_divisors_of_n_congruent_to_1_mod_4_l1643_164382

theorem prime_divisors_of_n_congruent_to_1_mod_4
  (x y n : ℕ)
  (hx : x ≥ 3)
  (hn : n ≥ 2)
  (h_eq : x^2 + 5 = y^n) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≡ 1 [MOD 4] :=
by
  sorry

end prime_divisors_of_n_congruent_to_1_mod_4_l1643_164382


namespace length_PX_l1643_164378

theorem length_PX (CX DP PW PX : ℕ) (hCX : CX = 60) (hDP : DP = 20) (hPW : PW = 40)
  (parallel_CD_WX : true)  -- We use a boolean to denote the parallel condition for simplicity
  (h1 : DP + PW = CX)  -- The sum of the segments from point C through P to point X
  (h2 : DP * 2 = PX)  -- The ratio condition derived from the similarity of triangles
  : PX = 40 := 
by
  -- using the given conditions and h2 to solve for PX
  sorry

end length_PX_l1643_164378


namespace pieces_from_sister_calculation_l1643_164386

-- Definitions for the conditions
def pieces_from_neighbors : ℕ := 5
def pieces_per_day : ℕ := 9
def duration : ℕ := 2

-- Definition to calculate the total number of pieces Emily ate
def total_pieces : ℕ := pieces_per_day * duration

-- Proof Problem: Prove Emily received 13 pieces of candy from her older sister
theorem pieces_from_sister_calculation :
  ∃ (pieces_from_sister : ℕ), pieces_from_sister = total_pieces - pieces_from_neighbors ∧ pieces_from_sister = 13 :=
by
  sorry

end pieces_from_sister_calculation_l1643_164386


namespace find_clique_of_size_6_l1643_164399

-- Defining the conditions of the graph G
variable (G : SimpleGraph (Fin 12))

-- Condition: For any subset of 9 vertices, there exists a subset of 5 vertices that form a complete subgraph K_5.
def condition (s : Finset (Fin 12)) : Prop :=
  s.card = 9 → ∃ t : Finset (Fin 12), t ⊆ s ∧ t.card = 5 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v)

-- The theorem to prove given the conditions
theorem find_clique_of_size_6 (h : ∀ s : Finset (Fin 12), condition G s) : 
  ∃ t : Finset (Fin 12), t.card = 6 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v) :=
sorry

end find_clique_of_size_6_l1643_164399
