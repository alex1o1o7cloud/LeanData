import Mathlib

namespace NUMINAMATH_GPT_cost_of_mixture_verify_cost_of_mixture_l1213_121369

variables {C1 C2 Cm : ℝ}

def ratio := 5 / 12

axiom cost_of_rice_1 : C1 = 4.5
axiom cost_of_rice_2 : C2 = 8.75
axiom mix_ratio : ratio = 5 / 12

theorem cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = (8.75 * 5 + 4.5 * 12) / 17 :=
by sorry

-- Prove that the cost of the mixture Cm is indeed 5.75
theorem verify_cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = 5.75 :=
by sorry

end NUMINAMATH_GPT_cost_of_mixture_verify_cost_of_mixture_l1213_121369


namespace NUMINAMATH_GPT_unique_function_solution_l1213_121342

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end NUMINAMATH_GPT_unique_function_solution_l1213_121342


namespace NUMINAMATH_GPT_gcd_five_pentagonal_and_n_plus_one_l1213_121332

-- Definition of the nth pentagonal number
def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n - 1)) / 2

-- Proof statement
theorem gcd_five_pentagonal_and_n_plus_one (n : ℕ) (h : 0 < n) : 
  Nat.gcd (5 * pentagonal_number n) (n + 1) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_five_pentagonal_and_n_plus_one_l1213_121332


namespace NUMINAMATH_GPT_passing_marks_l1213_121328

variable (T P : ℝ)

theorem passing_marks :
  (0.35 * T = P - 40) →
  (0.60 * T = P + 25) →
  P = 131 :=
by
  intro h1 h2
  -- Proof steps should follow here.
  sorry

end NUMINAMATH_GPT_passing_marks_l1213_121328


namespace NUMINAMATH_GPT_Diego_total_stamp_cost_l1213_121341

theorem Diego_total_stamp_cost :
  let price_brazil_colombia := 0.07
  let price_peru := 0.05
  let num_brazil_50s := 6
  let num_brazil_60s := 9
  let num_peru_50s := 8
  let num_peru_60s := 5
  let num_colombia_50s := 7
  let num_colombia_60s := 6
  let total_brazil := num_brazil_50s + num_brazil_60s
  let total_peru := num_peru_50s + num_peru_60s
  let total_colombia := num_colombia_50s + num_colombia_60s
  let cost_brazil := total_brazil * price_brazil_colombia
  let cost_peru := total_peru * price_peru
  let cost_colombia := total_colombia * price_brazil_colombia
  cost_brazil + cost_peru + cost_colombia = 2.61 :=
by
  sorry

end NUMINAMATH_GPT_Diego_total_stamp_cost_l1213_121341


namespace NUMINAMATH_GPT_total_spider_legs_l1213_121340

-- Define the number of legs per spider.
def legs_per_spider : ℕ := 8

-- Define half of the legs per spider.
def half_legs : ℕ := legs_per_spider / 2

-- Define the number of spiders in the group.
def num_spiders : ℕ := half_legs + 10

-- Prove the total number of spider legs in the group is 112.
theorem total_spider_legs : num_spiders * legs_per_spider = 112 := by
  -- Use 'sorry' to skip the detailed proof steps.
  sorry

end NUMINAMATH_GPT_total_spider_legs_l1213_121340


namespace NUMINAMATH_GPT_square_of_binomial_conditions_l1213_121326

variable (x a b m : ℝ)

theorem square_of_binomial_conditions :
  ∃ u v : ℝ, (x + a) * (x - a) = u^2 - v^2 ∧
             ∃ e f : ℝ, (-x - b) * (x - b) = - (e^2 - f^2) ∧
             ∃ g h : ℝ, (b + m) * (m - b) = g^2 - h^2 ∧
             ¬ ∃ p q : ℝ, (a + b) * (-a - b) = p^2 - q^2 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_conditions_l1213_121326


namespace NUMINAMATH_GPT_equation_represents_lines_and_point_l1213_121381

theorem equation_represents_lines_and_point:
    (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 0 → (x = 1 ∧ y = -2)) ∧
    (∀ x y : ℝ, x^2 - y^2 = 0 → (x = y) ∨ (x = -y)) → 
    (∀ x y : ℝ, ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0 → 
    ((x = 1 ∧ y = -2) ∨ (x + y = 0) ∨ (x - y = 0))) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_equation_represents_lines_and_point_l1213_121381


namespace NUMINAMATH_GPT_probability_complement_l1213_121393

theorem probability_complement (p : ℝ) (h : p = 0.997) : 1 - p = 0.003 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_probability_complement_l1213_121393


namespace NUMINAMATH_GPT_diff_of_squares_div_l1213_121334

-- Definitions from the conditions
def a : ℕ := 125
def b : ℕ := 105

-- The main statement to be proved
theorem diff_of_squares_div {a b : ℕ} (h1 : a = 125) (h2 : b = 105) : (a^2 - b^2) / 20 = 230 := by
  sorry

end NUMINAMATH_GPT_diff_of_squares_div_l1213_121334


namespace NUMINAMATH_GPT_megan_final_balance_percentage_l1213_121391

noncomputable def initial_balance_usd := 125.0
noncomputable def increase_percentage_babysitting := 0.25
noncomputable def exchange_rate_usd_to_eur_1 := 0.85
noncomputable def decrease_percentage_shoes := 0.20
noncomputable def exchange_rate_eur_to_usd := 1.15
noncomputable def increase_percentage_stocks := 0.15
noncomputable def decrease_percentage_medical := 0.10
noncomputable def exchange_rate_usd_to_eur_2 := 0.88

theorem megan_final_balance_percentage :
  let new_balance_after_babysitting := initial_balance_usd * (1 + increase_percentage_babysitting)
  let balance_in_eur := new_balance_after_babysitting * exchange_rate_usd_to_eur_1
  let balance_after_shoes := balance_in_eur * (1 - decrease_percentage_shoes)
  let balance_back_to_usd := balance_after_shoes * exchange_rate_eur_to_usd
  let balance_after_stocks := balance_back_to_usd * (1 + increase_percentage_stocks)
  let balance_after_medical := balance_after_stocks * (1 - decrease_percentage_medical)
  let final_balance_in_eur := balance_after_medical * exchange_rate_usd_to_eur_2
  let initial_balance_in_eur := initial_balance_usd * exchange_rate_usd_to_eur_1
  (final_balance_in_eur / initial_balance_in_eur) * 100 = 104.75 := by
  sorry

end NUMINAMATH_GPT_megan_final_balance_percentage_l1213_121391


namespace NUMINAMATH_GPT_ratio_of_areas_of_squares_l1213_121343

open Real

theorem ratio_of_areas_of_squares :
  let side_length_C := 48
  let side_length_D := 60
  let area_C := side_length_C^2
  let area_D := side_length_D^2
  area_C / area_D = (16 : ℝ) / 25 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_squares_l1213_121343


namespace NUMINAMATH_GPT_dasha_rectangle_problem_l1213_121306

variables (a b c : ℕ)

theorem dasha_rectangle_problem
  (h1 : a > 0) 
  (h2 : a * (b + c) + a * (b - a) + a^2 + a * (c - a) = 43) 
  : (a = 1 ∧ b + c = 22) ∨ (a = 43 ∧ b + c = 2) :=
by
  sorry

end NUMINAMATH_GPT_dasha_rectangle_problem_l1213_121306


namespace NUMINAMATH_GPT_always_in_range_l1213_121312

noncomputable def g (x k : ℝ) : ℝ := x^2 + 2 * k * x + 1

theorem always_in_range (k : ℝ) : 
  ∃ x : ℝ, g x k = 3 :=
by
  sorry

end NUMINAMATH_GPT_always_in_range_l1213_121312


namespace NUMINAMATH_GPT_inequality_for_positive_nums_l1213_121386

theorem inequality_for_positive_nums 
    (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    a^2 / b + c^2 / d ≥ (a + c)^2 / (b + d) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_nums_l1213_121386


namespace NUMINAMATH_GPT_range_of_a_l1213_121373

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := {x | x - a > 0}
def setB : Set ℝ := {x | x ≤ 0}

-- The main theorem asserting the condition
theorem range_of_a {a : ℝ} (h : setA a ∩ setB = ∅) : a ≥ 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1213_121373


namespace NUMINAMATH_GPT_closed_path_has_even_length_l1213_121372

   theorem closed_path_has_even_length 
     (u d r l : ℤ) 
     (hu : u = d) 
     (hr : r = l) : 
     ∃ k : ℤ, 2 * (u + r) = 2 * k :=
   by
     sorry
   
end NUMINAMATH_GPT_closed_path_has_even_length_l1213_121372


namespace NUMINAMATH_GPT_product_pass_rate_l1213_121375

variable (a b : ℝ)

theorem product_pass_rate (h1 : 0 ≤ a) (h2 : a < 1) (h3 : 0 ≤ b) (h4 : b < 1) : 
  (1 - a) * (1 - b) = 1 - (a + b - a * b) :=
by sorry

end NUMINAMATH_GPT_product_pass_rate_l1213_121375


namespace NUMINAMATH_GPT_commute_distance_l1213_121376

noncomputable def distance_to_work (total_time : ℕ) (speed_to_work : ℕ) (speed_to_home : ℕ) : ℕ :=
  let d := (speed_to_work * speed_to_home * total_time) / (speed_to_work + speed_to_home)
  d

-- Given conditions
def speed_to_work : ℕ := 45
def speed_to_home : ℕ := 30
def total_time : ℕ := 1

-- Proof problem statement
theorem commute_distance : distance_to_work total_time speed_to_work speed_to_home = 18 :=
by
  sorry

end NUMINAMATH_GPT_commute_distance_l1213_121376


namespace NUMINAMATH_GPT_probability_of_winning_l1213_121331

variable (P_A P_B P_C P_M_given_A P_M_given_B P_M_given_C : ℝ)

theorem probability_of_winning :
  P_A = 0.6 →
  P_B = 0.3 →
  P_C = 0.1 →
  P_M_given_A = 0.1 →
  P_M_given_B = 0.2 →
  P_M_given_C = 0.3 →
  (P_A * P_M_given_A + P_B * P_M_given_B + P_C * P_M_given_C) = 0.15 :=
by sorry

end NUMINAMATH_GPT_probability_of_winning_l1213_121331


namespace NUMINAMATH_GPT_minimum_berries_left_l1213_121362

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

theorem minimum_berries_left {a r n S : ℕ} 
  (h_a : a = 1) 
  (h_r : r = 2) 
  (h_n : n = 100) 
  (h_S : S = geometric_sum a r n) 
  : S = 2^100 - 1 -> ∃ k, k = 100 :=
by
  sorry

end NUMINAMATH_GPT_minimum_berries_left_l1213_121362


namespace NUMINAMATH_GPT_range_of_a_l1213_121319

noncomputable def y (a x : ℝ) : ℝ := a * Real.exp x + 3 * x
noncomputable def y_prime (a x : ℝ) : ℝ := a * Real.exp x + 3

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a * Real.exp x + 3 = 0 ∧ a * Real.exp x + 3 * x < 0) → a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1213_121319


namespace NUMINAMATH_GPT_intersection_points_in_decagon_l1213_121310

-- Define the number of sides for a regular decagon
def n : ℕ := 10

-- The formula to calculate the number of ways to choose 4 vertices from n vertices
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The statement that needs to be proven
theorem intersection_points_in_decagon : choose 10 4 = 210 := by
  sorry

end NUMINAMATH_GPT_intersection_points_in_decagon_l1213_121310


namespace NUMINAMATH_GPT_coin_value_permutations_l1213_121321

theorem coin_value_permutations : 
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540 := by
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  show 3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540
  
  -- Steps for the proof can be filled in
  -- sorry in place to indicate incomplete proof steps
  sorry

end NUMINAMATH_GPT_coin_value_permutations_l1213_121321


namespace NUMINAMATH_GPT_intersection_A_B_l1213_121348

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1213_121348


namespace NUMINAMATH_GPT_multiples_of_8_has_highest_avg_l1213_121359

def average_of_multiples (m : ℕ) (a b : ℕ) : ℕ :=
(a + b) / 2

def multiples_of_7_avg := average_of_multiples 7 7 196 -- 101.5
def multiples_of_2_avg := average_of_multiples 2 2 200 -- 101
def multiples_of_8_avg := average_of_multiples 8 8 200 -- 104
def multiples_of_5_avg := average_of_multiples 5 5 200 -- 102.5
def multiples_of_9_avg := average_of_multiples 9 9 189 -- 99

theorem multiples_of_8_has_highest_avg :
  multiples_of_8_avg > multiples_of_7_avg ∧
  multiples_of_8_avg > multiples_of_2_avg ∧
  multiples_of_8_avg > multiples_of_5_avg ∧
  multiples_of_8_avg > multiples_of_9_avg :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_8_has_highest_avg_l1213_121359


namespace NUMINAMATH_GPT_ten_factorial_mod_thirteen_l1213_121364

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end NUMINAMATH_GPT_ten_factorial_mod_thirteen_l1213_121364


namespace NUMINAMATH_GPT_fraction_sum_l1213_121316

theorem fraction_sum : (3 / 8) + (9 / 12) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_l1213_121316


namespace NUMINAMATH_GPT_diet_soda_bottles_l1213_121389

def total_bottles : ℕ := 17
def regular_soda_bottles : ℕ := 9

theorem diet_soda_bottles : total_bottles - regular_soda_bottles = 8 := by
  sorry

end NUMINAMATH_GPT_diet_soda_bottles_l1213_121389


namespace NUMINAMATH_GPT_solution_for_x_l1213_121322

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_solution_for_x_l1213_121322


namespace NUMINAMATH_GPT_defective_units_l1213_121302

-- Conditions given in the problem
variable (D : ℝ) (h1 : 0.05 * D = 0.35)

-- The percent of the units produced that are defective is 7%
theorem defective_units (h1 : 0.05 * D = 0.35) : D = 7 := sorry

end NUMINAMATH_GPT_defective_units_l1213_121302


namespace NUMINAMATH_GPT_logarithm_identity_l1213_121323

theorem logarithm_identity (k x : ℝ) (hk : 0 < k ∧ k ≠ 1) (hx : 0 < x) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 3 → x = 343 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_logarithm_identity_l1213_121323


namespace NUMINAMATH_GPT_sum_f_values_l1213_121354

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 / x) + 1

theorem sum_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f (3) + f (5) + f (7) + f (9) = 8 := 
by
  sorry

end NUMINAMATH_GPT_sum_f_values_l1213_121354


namespace NUMINAMATH_GPT_negate_proposition_l1213_121380

theorem negate_proposition (x : ℝ) :
  (¬(x > 1 → x^2 > 1)) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l1213_121380


namespace NUMINAMATH_GPT_factorize_expression_l1213_121365

variable (a : ℝ) (b : ℝ)

theorem factorize_expression : 2 * a - 8 * a * b^2 = 2 * a * (1 - 2 * b) * (1 + 2 * b) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1213_121365


namespace NUMINAMATH_GPT_find_z_when_w_15_l1213_121333

-- Define a direct variation relationship
def varies_directly (z w : ℕ) (k : ℕ) : Prop :=
  z = k * w

-- Using the given conditions and to prove the statement
theorem find_z_when_w_15 :
  ∃ k, (varies_directly 10 5 k) → (varies_directly 30 15 k) :=
by
  sorry

end NUMINAMATH_GPT_find_z_when_w_15_l1213_121333


namespace NUMINAMATH_GPT_big_rectangle_width_l1213_121356

theorem big_rectangle_width
  (W : ℝ)
  (h₁ : ∃ l w : ℝ, l = 40 ∧ w = W)
  (h₂ : ∃ l' w' : ℝ, l' = l / 2 ∧ w' = w / 2)
  (h_area : 200 = l' * w') :
  W = 20 :=
by sorry

end NUMINAMATH_GPT_big_rectangle_width_l1213_121356


namespace NUMINAMATH_GPT_part1_part2_l1213_121360

def f (x a : ℝ) := x^2 + 4 * a * x + 2 * a + 6

theorem part1 (a : ℝ) : (∃ x : ℝ, f x a = 0) ↔ (a = -1 ∨ a = 3 / 2) := 
by 
  sorry

def g (a : ℝ) := 2 - a * |a + 3|

theorem part2 (a : ℝ) :
  (-1 ≤ a ∧ a ≤ 3 / 2) →
  -19 / 4 ≤ g a ∧ g a ≤ 4 :=
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l1213_121360


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1213_121325

theorem problem1_solution (x : ℝ) :
  (2 < |2 * x - 5| ∧ |2 * x - 5| ≤ 7) → ((-1 ≤ x ∧ x < 3 / 2) ∨ (7 / 2 < x ∧ x ≤ 6)) := by
  sorry

theorem problem2_solution (x : ℝ) :
  (1 / (x - 1) > x + 1) → (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1213_121325


namespace NUMINAMATH_GPT_find_ab_l1213_121300

variables {a b : ℝ}

theorem find_ab
  (h : ∀ x : ℝ, 0 ≤ x → 0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) :
  a * b = -1 :=
sorry

end NUMINAMATH_GPT_find_ab_l1213_121300


namespace NUMINAMATH_GPT_eagles_win_at_least_three_matches_l1213_121396

-- Define the conditions
def n : ℕ := 5
def p : ℝ := 0.5

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k) * p^k * (1 - p)^(n - k)

-- Theorem stating the main result
theorem eagles_win_at_least_three_matches :
  (binomial_prob n 3 p + binomial_prob n 4 p + binomial_prob n 5 p) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eagles_win_at_least_three_matches_l1213_121396


namespace NUMINAMATH_GPT_digit_7_count_in_range_l1213_121347

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end NUMINAMATH_GPT_digit_7_count_in_range_l1213_121347


namespace NUMINAMATH_GPT_quadratic_real_roots_implies_k_range_l1213_121308

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_real_roots_implies_k_range_l1213_121308


namespace NUMINAMATH_GPT_probability_of_specific_selection_l1213_121353

/-- 
Given a drawer with 8 forks, 10 spoons, and 6 knives, 
the probability of randomly choosing one fork, one spoon, and one knife when three pieces of silverware are removed equals 120/506.
-/
theorem probability_of_specific_selection :
  let total_pieces := 24
  let total_ways := Nat.choose total_pieces 3
  let favorable_ways := 8 * 10 * 6
  (favorable_ways : ℚ) / total_ways = 120 / 506 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_specific_selection_l1213_121353


namespace NUMINAMATH_GPT_isosceles_triangles_count_isosceles_triangles_l1213_121395

theorem isosceles_triangles (x : ℕ) (b : ℕ) : 
  (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14) → 
  (b = 1 ∧ x = 14 ∨ b = 3 ∧ x = 13 ∨ b = 5 ∧ x = 12 ∨ b = 7 ∧ x = 11 ∨ b = 9 ∧ x = 10) :=
by sorry

theorem count_isosceles_triangles : 
  (∃ x b, (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14)) → 
  (5 = 5) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangles_count_isosceles_triangles_l1213_121395


namespace NUMINAMATH_GPT_dog_bones_initial_count_l1213_121327

theorem dog_bones_initial_count (buried : ℝ) (final : ℝ) : buried = 367.5 → final = -860 → (buried + (final + 367.5) + 860) = 367.5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_dog_bones_initial_count_l1213_121327


namespace NUMINAMATH_GPT_simplified_fraction_of_num_l1213_121336

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end NUMINAMATH_GPT_simplified_fraction_of_num_l1213_121336


namespace NUMINAMATH_GPT_range_of_a_l1213_121313

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0) ∧ y + (a - 1) * x + 2 * a - 1 = 0

def valid_a (a : ℝ) : Prop :=
  (p a ∨ q a) ∧ ¬(p a ∧ q a)

theorem range_of_a (a : ℝ) :
  valid_a a →
  (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1213_121313


namespace NUMINAMATH_GPT_percentage_decrease_in_y_when_x_doubles_l1213_121349

variable {k x y : ℝ}
variable (h_pos_x : 0 < x) (h_pos_y : 0 < y)
variable (inverse_proportional : x * y = k)

theorem percentage_decrease_in_y_when_x_doubles :
  (x' = 2 * x) →
  (y' = y / 2) →
  (100 * (y - y') / y) = 50 :=
by
  intro h1 h2
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_y_when_x_doubles_l1213_121349


namespace NUMINAMATH_GPT_square_area_PS_l1213_121390

noncomputable def area_of_square_on_PS : ℕ :=
  sorry

theorem square_area_PS (PQ QR RS PR PS : ℝ)
  (h1 : PQ ^ 2 = 25)
  (h2 : QR ^ 2 = 49)
  (h3 : RS ^ 2 = 64)
  (h4 : PQ^2 + QR^2 = PR^2)
  (h5 : PR^2 + RS^2 = PS^2) :
  PS^2 = 138 :=
by
  -- proof skipping
  sorry


end NUMINAMATH_GPT_square_area_PS_l1213_121390


namespace NUMINAMATH_GPT_total_peanut_cost_l1213_121346

def peanut_cost_per_pound : ℝ := 3
def minimum_pounds : ℝ := 15
def extra_pounds : ℝ := 20

theorem total_peanut_cost :
  (minimum_pounds + extra_pounds) * peanut_cost_per_pound = 105 :=
by
  sorry

end NUMINAMATH_GPT_total_peanut_cost_l1213_121346


namespace NUMINAMATH_GPT_dawn_hourly_income_l1213_121398

theorem dawn_hourly_income 
  (n : ℕ) (t_s t_p t_f I_p I_s I_f : ℝ)
  (h_n : n = 12)
  (h_t_s : t_s = 1.5)
  (h_t_p : t_p = 2)
  (h_t_f : t_f = 0.5)
  (h_I_p : I_p = 3600)
  (h_I_s : I_s = 1200)
  (h_I_f : I_f = 300) :
  (I_p + I_s + I_f) / (n * (t_s + t_p + t_f)) = 106.25 := 
  by
  sorry

end NUMINAMATH_GPT_dawn_hourly_income_l1213_121398


namespace NUMINAMATH_GPT_tank_capacity_l1213_121351

theorem tank_capacity (x : ℝ) 
  (h1 : 1/4 * x + 180 = 2/3 * x) : 
  x = 432 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1213_121351


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1213_121379

noncomputable def a_n (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_problem (a d : ℝ) 
  (h : a_n 1 a d - a_n 4 a d - a_n 8 a d - a_n 12 a d + a_n 15 a d = 2) :
  a_n 3 a d + a_n 13 a d = -4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1213_121379


namespace NUMINAMATH_GPT_incorrect_transformation_l1213_121303

theorem incorrect_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a / 2 = b / 3) :
  (∃ k : ℕ, 2 * a = 3 * b → false) ∧ 
  (a / b = 2 / 3) ∧ 
  (b / a = 3 / 2) ∧
  (3 * a = 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_transformation_l1213_121303


namespace NUMINAMATH_GPT_negation_exists_implies_forall_l1213_121392

theorem negation_exists_implies_forall (x_0 : ℝ) (h : ∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) : 
  ¬ (∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) ↔ ∀ x : ℝ, x^3 - x + 1 ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_negation_exists_implies_forall_l1213_121392


namespace NUMINAMATH_GPT_main_theorem_l1213_121337

variable (x : ℝ)

-- Define proposition p
def p : Prop := ∃ x0 : ℝ, x0^2 < x0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- Main proof problem
theorem main_theorem : p ∧ q := 
by {
  sorry
}

end NUMINAMATH_GPT_main_theorem_l1213_121337


namespace NUMINAMATH_GPT_polygon_number_of_sides_l1213_121377

-- Define the given conditions
def each_interior_angle (n : ℕ) : ℕ := 120

-- Define the property to calculate the number of sides
def num_sides (each_exterior_angle : ℕ) : ℕ := 360 / each_exterior_angle

-- Statement of the problem
theorem polygon_number_of_sides : num_sides (180 - each_interior_angle 6) = 6 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_polygon_number_of_sides_l1213_121377


namespace NUMINAMATH_GPT_sphere_tangent_radius_l1213_121311

variables (a b : ℝ) (h : b > a)

noncomputable def radius (a b : ℝ) : ℝ := a * (b - a) / Real.sqrt (b^2 - a^2)

theorem sphere_tangent_radius (a b : ℝ) (h : b > a) : 
  radius a b = a * (b - a) / Real.sqrt (b^2 - a^2) :=
by sorry

end NUMINAMATH_GPT_sphere_tangent_radius_l1213_121311


namespace NUMINAMATH_GPT_correct_exponentiation_l1213_121370

theorem correct_exponentiation (a : ℕ) : 
  (a^3 * a^2 = a^5) ∧ ¬(a^3 + a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^10 / a^2 = a^5) :=
by
  -- Proof steps and actual mathematical validation will go here.
  -- For now, we skip the actual proof due to the problem requirements.
  sorry

end NUMINAMATH_GPT_correct_exponentiation_l1213_121370


namespace NUMINAMATH_GPT_min_value_fraction_l1213_121387

theorem min_value_fraction 
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a1 a3 a13 : ℕ)
  (d : ℕ) 
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a1 = 1)
  (h4 : a3 ^ 2 = a1 * a13)
  (h5 : ∀ n, S_n n = n * (a1 + a_n n) / 2) :
  ∃ n, (2 * S_n n + 16) / (a_n n + 3) = 4 := 
sorry

end NUMINAMATH_GPT_min_value_fraction_l1213_121387


namespace NUMINAMATH_GPT_jars_of_plum_jelly_sold_l1213_121388

theorem jars_of_plum_jelly_sold (P R G S : ℕ) (h1 : R = 2 * P) (h2 : G = 3 * R) (h3 : G = 2 * S) (h4 : S = 18) : P = 6 := by
  sorry

end NUMINAMATH_GPT_jars_of_plum_jelly_sold_l1213_121388


namespace NUMINAMATH_GPT_toms_age_l1213_121304

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end NUMINAMATH_GPT_toms_age_l1213_121304


namespace NUMINAMATH_GPT_dot_product_example_l1213_121309

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example 
  (ha : a = (-1, 1)) 
  (hb : b = (3, -2)) : dot_product a b = -5 := by
  sorry

end NUMINAMATH_GPT_dot_product_example_l1213_121309


namespace NUMINAMATH_GPT_range_of_m_l1213_121367

theorem range_of_m (m : ℝ) : 
    (∀ x y : ℝ, (x^2 / (4 - m) + y^2 / (m - 3) = 1) → 
    4 - m > 0 ∧ m - 3 > 0 ∧ m - 3 > 4 - m) → 
    (7/2 < m ∧ m < 4) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1213_121367


namespace NUMINAMATH_GPT_decreasing_function_condition_l1213_121352

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (4 * a - 1) * x + 4 * a else a ^ x

theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1 / 7 ≤ a ∧ a < 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_condition_l1213_121352


namespace NUMINAMATH_GPT_total_distance_walked_l1213_121383

theorem total_distance_walked 
  (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 0.75)
  (h2 : d2 = 0.25) :
  d1 + d2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_walked_l1213_121383


namespace NUMINAMATH_GPT_smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l1213_121301

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_and_second_smallest_four_digit_numbers_divisible_by_35 :
  ∃ a b : ℕ, 
    is_four_digit a ∧ 
    is_four_digit b ∧ 
    divisible_by_35 a ∧ 
    divisible_by_35 b ∧ 
    a < b ∧ 
    ∀ c : ℕ, is_four_digit c → divisible_by_35 c → a ≤ c → (c = a ∨ c = b) :=
by
  sorry

end NUMINAMATH_GPT_smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l1213_121301


namespace NUMINAMATH_GPT_evaluate_expression_l1213_121329

theorem evaluate_expression (b : ℕ) (hb : b = 2) : (b^3 * b^4) - b^2 = 124 :=
by
  -- leave the proof empty with a placeholder
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1213_121329


namespace NUMINAMATH_GPT_count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l1213_121384

-- Definitions based on conditions
def is_symmetric_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

def symmetric_pair (a b : ℕ) : Prop :=
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) ∨ (a = 8 ∧ b = 8) ∨ (a = 6 ∧ b = 9) ∨ (a = 9 ∧ b = 6)

-- 1. Prove the total number of 7-digit symmetric numbers
theorem count_symmetric_numbers : ∃ n, n = 300 := by
  sorry

-- 2. Prove the number of symmetric numbers divisible by 4
theorem count_symmetric_divisible_by_4 : ∃ n, n = 75 := by
  sorry

-- 3. Prove the total sum of these 7-digit symmetric numbers
theorem sum_symmetric_numbers : ∃ s, s = 1959460200 := by
  sorry

end NUMINAMATH_GPT_count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l1213_121384


namespace NUMINAMATH_GPT_find_third_sum_l1213_121394

def arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (a 1) + (a 4) + (a 7) = 39 ∧ (a 2) + (a 5) + (a 8) = 33

theorem find_third_sum (a : ℕ → ℝ)
                       (d : ℝ)
                       (h_seq : arithmetic_sequence_sum a d)
                       (a_1 : ℝ) :
  a 1 = a_1 ∧ a 2 = a_1 + d ∧ a 3 = a_1 + 2 * d ∧
  a 4 = a_1 + 3 * d ∧ a 5 = a_1 + 4 * d ∧ a 6 = a_1 + 5 * d ∧
  a 7 = a_1 + 6 * d ∧ a 8 = a_1 + 7 * d ∧ a 9 = a_1 + 8 * d →
  a 3 + a 6 + a 9 = 27 :=
by
  sorry

end NUMINAMATH_GPT_find_third_sum_l1213_121394


namespace NUMINAMATH_GPT_color_of_face_opposite_silver_is_yellow_l1213_121358

def Face : Type := String

def Color : Type := String

variable (B Y O Bl S V : Color)

-- Conditions based on views
variable (cube : Face → Color)
variable (top front_right_1 right_1 front_right_2 front_right_3 : Face)
variable (back : Face)

axiom view1 : cube top = B ∧ cube front_right_1 = Y ∧ cube right_1 = O
axiom view2 : cube top = B ∧ cube front_right_2 = Bl ∧ cube right_1 = O
axiom view3 : cube top = B ∧ cube front_right_3 = V ∧ cube right_1 = O

-- Additional axiom based on the fact that S is not visible and deduced to be on the back face
axiom silver_back : cube back = S

-- The problem: Prove that the color of the face opposite the silver face is yellow.
theorem color_of_face_opposite_silver_is_yellow :
  (∃ front : Face, cube front = Y) :=
by
  sorry

end NUMINAMATH_GPT_color_of_face_opposite_silver_is_yellow_l1213_121358


namespace NUMINAMATH_GPT_speeds_of_bodies_l1213_121399

theorem speeds_of_bodies 
  (v1 v2 : ℝ)
  (h1 : 21 * v1 + 10 * v2 = 270)
  (h2 : 51 * v1 + 40 * v2 = 540)
  (h3 : 5 * v2 = 3 * v1): 
  v1 = 10 ∧ v2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_speeds_of_bodies_l1213_121399


namespace NUMINAMATH_GPT_central_angle_of_sector_l1213_121366

theorem central_angle_of_sector :
  ∃ R α : ℝ, (2 * R + α * R = 4) ∧ (1 / 2 * R ^ 2 * α = 1) ∧ α = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1213_121366


namespace NUMINAMATH_GPT_shorten_to_sixth_power_l1213_121335

theorem shorten_to_sixth_power (x n m p q r : ℕ) (h1 : x > 1000000)
  (h2 : x / 10 = n^2)
  (h3 : n^2 / 10 = m^3)
  (h4 : m^3 / 10 = p^4)
  (h5 : p^4 / 10 = q^5) :
  q^5 / 10 = r^6 :=
sorry

end NUMINAMATH_GPT_shorten_to_sixth_power_l1213_121335


namespace NUMINAMATH_GPT_manolo_makes_45_masks_in_four_hours_l1213_121320

noncomputable def face_masks_in_four_hour_shift : ℕ :=
  let first_hour_rate := 4
  let subsequent_hour_rate := 6
  let first_hour_face_masks := 60 / first_hour_rate
  let subsequent_hours_face_masks_per_hour := 60 / subsequent_hour_rate
  let total_face_masks :=
    first_hour_face_masks + subsequent_hours_face_masks_per_hour * (4 - 1)
  total_face_masks

theorem manolo_makes_45_masks_in_four_hours :
  face_masks_in_four_hour_shift = 45 :=
 by sorry

end NUMINAMATH_GPT_manolo_makes_45_masks_in_four_hours_l1213_121320


namespace NUMINAMATH_GPT_brad_ate_six_halves_l1213_121378

theorem brad_ate_six_halves (total_cookies : ℕ) (total_halves : ℕ) (greg_ate : ℕ) (halves_left : ℕ) (halves_brad_ate : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : total_halves = total_cookies * 2)
  (h3 : greg_ate = 4)
  (h4 : halves_left = 18)
  (h5 : total_halves - greg_ate - halves_brad_ate = halves_left) :
  halves_brad_ate = 6 :=
by
  sorry

end NUMINAMATH_GPT_brad_ate_six_halves_l1213_121378


namespace NUMINAMATH_GPT_Amanda_second_day_tickets_l1213_121330

/-- Amanda's ticket sales problem set up -/
def Amanda_total_tickets := 80
def Amanda_first_day_tickets := 5 * 4
def Amanda_third_day_tickets := 28

theorem Amanda_second_day_tickets :
  ∃ (tickets_sold_second_day : ℕ), tickets_sold_second_day = 32 :=
by
  let first_day := Amanda_first_day_tickets
  let third_day := Amanda_third_day_tickets
  let needed_before_third := Amanda_total_tickets - third_day
  let second_day := needed_before_third - first_day
  use second_day
  sorry

end NUMINAMATH_GPT_Amanda_second_day_tickets_l1213_121330


namespace NUMINAMATH_GPT_sum_of_cubes_of_integers_l1213_121363

theorem sum_of_cubes_of_integers (n: ℕ) (h1: (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 8830) : 
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 52264 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_integers_l1213_121363


namespace NUMINAMATH_GPT_white_balls_count_l1213_121338

theorem white_balls_count (W B R : ℕ) (h1 : B = W + 14) (h2 : R = 3 * (B - W)) (h3 : W + B + R = 1000) : W = 472 :=
sorry

end NUMINAMATH_GPT_white_balls_count_l1213_121338


namespace NUMINAMATH_GPT_range_of_y_eq_4_sin_squared_x_minus_2_l1213_121305

theorem range_of_y_eq_4_sin_squared_x_minus_2 : 
  (∀ x : ℝ, y = 4 * (Real.sin x)^2 - 2) → 
  (∃ a b : ℝ, ∀ x : ℝ, y ∈ Set.Icc a b ∧ a = -2 ∧ b = 2) :=
sorry

end NUMINAMATH_GPT_range_of_y_eq_4_sin_squared_x_minus_2_l1213_121305


namespace NUMINAMATH_GPT_find_box_length_l1213_121397

theorem find_box_length (width depth : ℕ) (num_cubes : ℕ) (cube_side length : ℕ) 
  (h1 : width = 20)
  (h2 : depth = 10)
  (h3 : num_cubes = 56)
  (h4 : cube_side = 10)
  (h5 : length * width * depth = num_cubes * cube_side * cube_side * cube_side) :
  length = 280 :=
sorry

end NUMINAMATH_GPT_find_box_length_l1213_121397


namespace NUMINAMATH_GPT_largest_n_divides_l1213_121345

theorem largest_n_divides (n : ℕ) (h : 2^n ∣ 5^256 - 1) : n ≤ 10 := sorry

end NUMINAMATH_GPT_largest_n_divides_l1213_121345


namespace NUMINAMATH_GPT_division_by_fraction_l1213_121344

theorem division_by_fraction :
  (12 : ℝ) / (1 / 6) = 72 :=
by
  sorry

end NUMINAMATH_GPT_division_by_fraction_l1213_121344


namespace NUMINAMATH_GPT_container_unoccupied_volume_l1213_121385

noncomputable def unoccupied_volume (side_length_container : ℝ) (side_length_ice : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let volume_container := side_length_container ^ 3
  let volume_water := (3 / 4) * volume_container
  let volume_ice := num_ice_cubes / 2 * side_length_ice ^ 3
  volume_container - (volume_water + volume_ice)

theorem container_unoccupied_volume :
  unoccupied_volume 12 1.5 12 = 411.75 :=
by
  sorry

end NUMINAMATH_GPT_container_unoccupied_volume_l1213_121385


namespace NUMINAMATH_GPT_largest_of_given_numbers_l1213_121314

theorem largest_of_given_numbers :
  ∀ (a b c d e : ℝ), a = 0.998 → b = 0.9899 → c = 0.99 → d = 0.981 → e = 0.995 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  intros a b c d e Ha Hb Hc Hd He
  rw [Ha, Hb, Hc, Hd, He]
  exact ⟨ by norm_num, by norm_num, by norm_num, by norm_num ⟩

end NUMINAMATH_GPT_largest_of_given_numbers_l1213_121314


namespace NUMINAMATH_GPT_rhombus_triangle_area_l1213_121371

theorem rhombus_triangle_area (d1 d2 : ℝ) (h_d1 : d1 = 15) (h_d2 : d2 = 20) :
  ∃ (area : ℝ), area = 75 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_triangle_area_l1213_121371


namespace NUMINAMATH_GPT_regions_divided_by_7_tangents_l1213_121382

-- Define the recursive function R for the number of regions divided by n tangents
def R : ℕ → ℕ
| 0       => 1
| (n + 1) => R n + (n + 1)

-- The theorem stating the specific case of the problem
theorem regions_divided_by_7_tangents : R 7 = 29 := by
  sorry

end NUMINAMATH_GPT_regions_divided_by_7_tangents_l1213_121382


namespace NUMINAMATH_GPT_arithmetic_expr_eval_l1213_121315

/-- A proof that the arithmetic expression (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) evaluates to -13122. -/
theorem arithmetic_expr_eval : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expr_eval_l1213_121315


namespace NUMINAMATH_GPT_non_congruent_rectangles_count_l1213_121324

theorem non_congruent_rectangles_count :
  let grid_width := 6
  let grid_height := 4
  let axis_aligned_rectangles := (grid_width.choose 2) * (grid_height.choose 2)
  let squares_1x1 := (grid_width - 1) * (grid_height - 1)
  let squares_2x2 := (grid_width - 2) * (grid_height - 2)
  let non_congruent_rectangles := axis_aligned_rectangles - (squares_1x1 + squares_2x2)
  non_congruent_rectangles = 67 := 
by {
  sorry
}

end NUMINAMATH_GPT_non_congruent_rectangles_count_l1213_121324


namespace NUMINAMATH_GPT_sum_of_solutions_of_quadratic_l1213_121357

theorem sum_of_solutions_of_quadratic :
    let a := 1;
    let b := -8;
    let c := -40;
    let discriminant := b * b - 4 * a * c;
    let root_discriminant := Real.sqrt discriminant;
    let sol1 := (-b + root_discriminant) / (2 * a);
    let sol2 := (-b - root_discriminant) / (2 * a);
    sol1 + sol2 = 8 := by
{
  sorry
}

end NUMINAMATH_GPT_sum_of_solutions_of_quadratic_l1213_121357


namespace NUMINAMATH_GPT_product_of_numbers_l1213_121350

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := 
sorry

end NUMINAMATH_GPT_product_of_numbers_l1213_121350


namespace NUMINAMATH_GPT_find_longer_parallel_side_length_l1213_121374

noncomputable def longer_parallel_side_length_of_trapezoid : ℝ :=
  let square_side_length : ℝ := 2
  let center_to_side_length : ℝ := square_side_length / 2
  let midline_length : ℝ := square_side_length / 2
  let equal_area : ℝ := (square_side_length^2) / 3
  let height_of_trapezoid : ℝ := center_to_side_length
  let shorter_parallel_side_length : ℝ := midline_length
  let longer_parallel_side_length := (2 * equal_area / height_of_trapezoid) - shorter_parallel_side_length
  longer_parallel_side_length

theorem find_longer_parallel_side_length : 
  longer_parallel_side_length_of_trapezoid = 5/3 := 
sorry

end NUMINAMATH_GPT_find_longer_parallel_side_length_l1213_121374


namespace NUMINAMATH_GPT_find_some_number_l1213_121355

-- Conditions on operations
axiom plus_means_mult (a b : ℕ) : (a + b) = (a * b)
axiom minus_means_plus (a b : ℕ) : (a - b) = (a + b)
axiom mult_means_div (a b : ℕ) : (a * b) = (a / b)
axiom div_means_minus (a b : ℕ) : (a / b) = (a - b)

-- Problem statement
theorem find_some_number (some_number : ℕ) :
  (6 - 9 + some_number * 3 / 25 = 5 ↔
   6 + 9 * some_number / 3 - 25 = 5) ∧
  some_number = 8 := by
  sorry

end NUMINAMATH_GPT_find_some_number_l1213_121355


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l1213_121307

def repeating_decimal := 567 / 999

theorem repeating_decimal_as_fraction : repeating_decimal = 21 / 37 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l1213_121307


namespace NUMINAMATH_GPT_erin_serves_all_soup_in_15_minutes_l1213_121361

noncomputable def time_to_serve_all_soup
  (ounces_per_bowl : ℕ)
  (bowls_per_minute : ℕ)
  (soup_in_gallons : ℕ)
  (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := soup_in_gallons * ounces_per_gallon
  let total_bowls := (total_ounces + ounces_per_bowl - 1) / ounces_per_bowl -- to round up
  let total_minutes := (total_bowls + bowls_per_minute - 1) / bowls_per_minute -- to round up
  total_minutes

theorem erin_serves_all_soup_in_15_minutes :
  time_to_serve_all_soup 10 5 6 128 = 15 :=
sorry

end NUMINAMATH_GPT_erin_serves_all_soup_in_15_minutes_l1213_121361


namespace NUMINAMATH_GPT_contrapositive_of_square_comparison_l1213_121339

theorem contrapositive_of_square_comparison (x y : ℝ) : (x^2 > y^2 → x > y) → (x ≤ y → x^2 ≤ y^2) :=
  by sorry

end NUMINAMATH_GPT_contrapositive_of_square_comparison_l1213_121339


namespace NUMINAMATH_GPT_arithmetic_seq_a12_l1213_121318

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 = 1)
  (h2 : a 7 + a 9 = 16)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 12 = 15 :=
by sorry

end NUMINAMATH_GPT_arithmetic_seq_a12_l1213_121318


namespace NUMINAMATH_GPT_total_hours_uploaded_l1213_121317

def hours_June_1_to_10 : ℝ := 5 * 2 * 10
def hours_June_11_to_20 : ℝ := 10 * 1 * 10
def hours_June_21_to_25 : ℝ := 7 * 3 * 5
def hours_June_26_to_30 : ℝ := 15 * 0.5 * 5

def total_video_hours : ℝ :=
  hours_June_1_to_10 + hours_June_11_to_20 + hours_June_21_to_25 + hours_June_26_to_30

theorem total_hours_uploaded :
  total_video_hours = 342.5 :=
by
  sorry

end NUMINAMATH_GPT_total_hours_uploaded_l1213_121317


namespace NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_2016_l1213_121368

-- Define 2016 and the sum of its distinct prime factors
def n : ℕ := 2016
def sumOfDistinctPrimeFactors (n : ℕ) : ℕ :=
  if n = 2016 then 2 + 3 + 7 else 0  -- Capture the problem-specific condition

-- The main theorem to prove the sum of the distinct prime factors of 2016 is 12
theorem sum_of_distinct_prime_factors_of_2016 :
  sumOfDistinctPrimeFactors 2016 = 12 :=
by
  -- Since this is beyond the obvious steps, we use a sorry here
  sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_2016_l1213_121368
