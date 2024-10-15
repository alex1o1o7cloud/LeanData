import Mathlib

namespace NUMINAMATH_GPT_Maggie_takes_75_percent_l1116_111637

def Debby's_portion : ℚ := 0.25
def Maggie's_share : ℚ := 4500
def Total_amount : ℚ := 6000
def Maggie's_portion : ℚ := Maggie's_share / Total_amount

theorem Maggie_takes_75_percent : Maggie's_portion = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_Maggie_takes_75_percent_l1116_111637


namespace NUMINAMATH_GPT_general_formula_l1116_111634

noncomputable def a : ℕ → ℕ
| 0       => 5
| (n + 1) => 2 * a n + 3

theorem general_formula : ∀ n, a n = 2 ^ (n + 2) - 3 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_l1116_111634


namespace NUMINAMATH_GPT_gh_two_value_l1116_111622

def g (x : ℤ) : ℤ := 3 * x ^ 2 + 2
def h (x : ℤ) : ℤ := -5 * x ^ 3 + 2

theorem gh_two_value : g (h 2) = 4334 := by
  sorry

end NUMINAMATH_GPT_gh_two_value_l1116_111622


namespace NUMINAMATH_GPT_solve_f_435_l1116_111667

variable (f : ℝ → ℝ)

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (3 - x) = f x

-- To Prove
theorem solve_f_435 : f 435 = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_f_435_l1116_111667


namespace NUMINAMATH_GPT_number_of_sheets_l1116_111630

theorem number_of_sheets (S E : ℕ) (h1 : S - E = 60) (h2 : 5 * E = S) : S = 150 := by
  sorry

end NUMINAMATH_GPT_number_of_sheets_l1116_111630


namespace NUMINAMATH_GPT_total_crayons_l1116_111663

-- Define the number of crayons Billy has
def billy_crayons : ℝ := 62.0

-- Define the number of crayons Jane has
def jane_crayons : ℝ := 52.0

-- Formulate the theorem to prove the total number of crayons
theorem total_crayons : billy_crayons + jane_crayons = 114.0 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l1116_111663


namespace NUMINAMATH_GPT_keith_initial_cards_l1116_111652

theorem keith_initial_cards (new_cards : ℕ) (cards_after_incident : ℕ) (total_cards : ℕ) :
  new_cards = 8 →
  cards_after_incident = 46 →
  total_cards = 2 * cards_after_incident →
  (total_cards - new_cards) = 84 :=
by
  intros
  sorry

end NUMINAMATH_GPT_keith_initial_cards_l1116_111652


namespace NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l1116_111629

-- Definitions based on the problem conditions
def is_isosceles_triangle (A B C: ℝ) := (A = B) ∨ (B = C) ∨ (C = A)
def angle_sum_triangle (A B C: ℝ) := A + B + C = 180

-- The main theorem we want to prove
theorem base_angle_of_isosceles_triangle (A B C: ℝ)
(h1: is_isosceles_triangle A B C)
(h2: A = 50 ∨ B = 50 ∨ C = 50):
C = 50 ∨ C = 65 :=
by
  sorry

end NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l1116_111629


namespace NUMINAMATH_GPT_grade_above_B_l1116_111693

theorem grade_above_B (total_students : ℕ) (percentage_below_B : ℕ) (students_above_B : ℕ) :
  total_students = 60 ∧ percentage_below_B = 40 ∧ students_above_B = total_students * (100 - percentage_below_B) / 100 →
  students_above_B = 36 :=
by
  sorry

end NUMINAMATH_GPT_grade_above_B_l1116_111693


namespace NUMINAMATH_GPT_intersection_A_B_l1116_111647

def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | -1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1116_111647


namespace NUMINAMATH_GPT_maximum_M_l1116_111602

-- Define the sides of a triangle condition
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Theorem statement
theorem maximum_M (a b c : ℝ) (h : is_triangle a b c) : 
  (a^2 + b^2) / (c^2) > (1/2) :=
sorry

end NUMINAMATH_GPT_maximum_M_l1116_111602


namespace NUMINAMATH_GPT_k_domain_all_reals_l1116_111646

noncomputable def domain_condition (k : ℝ) : Prop :=
  9 + 28 * k < 0

noncomputable def k_values : Set ℝ :=
  {k : ℝ | domain_condition k}

theorem k_domain_all_reals :
  k_values = {k : ℝ | k < -9 / 28} :=
by
  sorry

end NUMINAMATH_GPT_k_domain_all_reals_l1116_111646


namespace NUMINAMATH_GPT_scale_model_height_l1116_111695

/-- 
Given a scale model ratio and the actual height of the skyscraper in feet,
we can deduce the height of the model in inches.
-/
theorem scale_model_height
  (scale_ratio : ℕ := 25)
  (actual_height_feet : ℕ := 1250) :
  (actual_height_feet / scale_ratio) * 12 = 600 :=
by 
  sorry

end NUMINAMATH_GPT_scale_model_height_l1116_111695


namespace NUMINAMATH_GPT_interval_of_segmentation_l1116_111699

-- Define the population size and sample size as constants.
def population_size : ℕ := 2000
def sample_size : ℕ := 40

-- State the theorem for the interval of segmentation.
theorem interval_of_segmentation :
  population_size / sample_size = 50 :=
sorry

end NUMINAMATH_GPT_interval_of_segmentation_l1116_111699


namespace NUMINAMATH_GPT_minimum_a_l1116_111621

theorem minimum_a (a b x : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : b - a = 2013) (h₃ : x > 0) (h₄ : x^2 - a * x + b = 0) : a = 93 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_l1116_111621


namespace NUMINAMATH_GPT_pairs_of_socks_calculation_l1116_111679

variable (num_pairs_socks : ℤ)
variable (cost_per_pair : ℤ := 950) -- in cents
variable (cost_shoes : ℤ := 9200) -- in cents
variable (money_jack_has : ℤ := 4000) -- in cents
variable (money_needed : ℤ := 7100) -- in cents
variable (total_money_needed : ℤ := money_jack_has + money_needed)

theorem pairs_of_socks_calculation (x : ℤ) (h : cost_per_pair * x + cost_shoes = total_money_needed) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_pairs_of_socks_calculation_l1116_111679


namespace NUMINAMATH_GPT_tim_final_soda_cans_l1116_111657

-- Definitions based on given conditions
def initialSodaCans : ℕ := 22
def cansTakenByJeff : ℕ := 6
def remainingCans (t0 j : ℕ) : ℕ := t0 - j
def additionalCansBought (remaining : ℕ) : ℕ := remaining / 2

-- Function to calculate final number of soda cans
def finalSodaCans (t0 j : ℕ) : ℕ :=
  let remaining := remainingCans t0 j
  remaining + additionalCansBought remaining

-- Theorem to prove the final number of soda cans
theorem tim_final_soda_cans : finalSodaCans initialSodaCans cansTakenByJeff = 24 :=
by
  sorry

end NUMINAMATH_GPT_tim_final_soda_cans_l1116_111657


namespace NUMINAMATH_GPT_max_M_correct_l1116_111696

variable (A : ℝ) (x y : ℝ)

axiom A_pos : A > 0

noncomputable def max_M : ℝ :=
if A ≤ 4 then 2 + A / 2 else 2 * Real.sqrt A

theorem max_M_correct : 
  (∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y + A/(x + y) ≥ max_M A / Real.sqrt (x * y)) ∧ 
  (A ≤ 4 → max_M A = 2 + A / 2) ∧ 
  (A > 4 → max_M A = 2 * Real.sqrt A) :=
sorry

end NUMINAMATH_GPT_max_M_correct_l1116_111696


namespace NUMINAMATH_GPT_lambda_range_l1116_111606

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end NUMINAMATH_GPT_lambda_range_l1116_111606


namespace NUMINAMATH_GPT_divisibility_by_29_and_29pow4_l1116_111603

theorem divisibility_by_29_and_29pow4 (x y z : ℤ) (h : 29 ∣ (x^4 + y^4 + z^4)) : 29^4 ∣ (x^4 + y^4 + z^4) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_29_and_29pow4_l1116_111603


namespace NUMINAMATH_GPT_market_value_of_stock_l1116_111662

def face_value : ℝ := 100
def dividend_percentage : ℝ := 0.13
def yield : ℝ := 0.08

theorem market_value_of_stock : 
  (dividend_percentage * face_value / yield) * 100 = 162.50 :=
by
  sorry

end NUMINAMATH_GPT_market_value_of_stock_l1116_111662


namespace NUMINAMATH_GPT_werewolf_eats_per_week_l1116_111675
-- First, we import the necessary libraries

-- We define the conditions using Lean definitions

-- The vampire drains 3 people a week
def vampire_drains_per_week : Nat := 3

-- The total population of the village
def village_population : Nat := 72

-- The number of weeks both can live off the population
def weeks : Nat := 9

-- Prove the number of people the werewolf eats per week (W) given the conditions
theorem werewolf_eats_per_week :
  ∃ W : Nat, vampire_drains_per_week * weeks + weeks * W = village_population ∧ W = 5 :=
by
  sorry

end NUMINAMATH_GPT_werewolf_eats_per_week_l1116_111675


namespace NUMINAMATH_GPT_necessary_not_sufficient_l1116_111666

-- Definitions and conditions based on the problem statement
def x_ne_1 (x : ℝ) : Prop := x ≠ 1
def polynomial_ne_zero (x : ℝ) : Prop := (x^2 - 3 * x + 2) ≠ 0

-- The theorem statement
theorem necessary_not_sufficient (x : ℝ) : 
  (∀ x, polynomial_ne_zero x → x_ne_1 x) ∧ ¬ (∀ x, x_ne_1 x → polynomial_ne_zero x) :=
by 
  intros
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l1116_111666


namespace NUMINAMATH_GPT_average_of_11_numbers_l1116_111641

theorem average_of_11_numbers (a b c d e f g h i j k : ℕ) 
  (h₀ : (a + b + c + d + e + f) / 6 = 19)
  (h₁ : (f + g + h + i + j + k) / 6 = 27)
  (h₂ : f = 34) :
  (a + b + c + d + e + f + g + h + i + j + k) / 11 = 22 := 
by
  sorry

end NUMINAMATH_GPT_average_of_11_numbers_l1116_111641


namespace NUMINAMATH_GPT_inverse_variation_l1116_111616

theorem inverse_variation (x y k : ℝ) (h1 : y = k / x^2) (h2 : k = 8) (h3 : y = 0.5) : x = 4 := by
  sorry

end NUMINAMATH_GPT_inverse_variation_l1116_111616


namespace NUMINAMATH_GPT_at_least_two_participants_solved_exactly_five_l1116_111690

open Nat Real

variable {n : ℕ}  -- Number of participants
variable {pij : ℕ → ℕ → ℕ} -- Number of contestants who correctly answered both the i-th and j-th problems

-- Conditions as definitions in Lean 4
def conditions (n : ℕ) (pij : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 6 → pij i j > (2 * n) / 5) ∧
  (∀ k, ¬ (∀ i, 1 ≤ i ∧ i ≤ 6 → pij k i = 1))

-- Main theorem statement
theorem at_least_two_participants_solved_exactly_five (n : ℕ) (pij : ℕ → ℕ → ℕ) (h : conditions n pij) : ∃ k₁ k₂, k₁ ≠ k₂ ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₁ i = 1) ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₂ i = 1) := sorry

end NUMINAMATH_GPT_at_least_two_participants_solved_exactly_five_l1116_111690


namespace NUMINAMATH_GPT_increase_80_by_150_percent_l1116_111669

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end NUMINAMATH_GPT_increase_80_by_150_percent_l1116_111669


namespace NUMINAMATH_GPT_pen_and_pencil_total_cost_l1116_111672

theorem pen_and_pencil_total_cost :
  ∀ (pen pencil : ℕ), pen = 4 → pen = 2 * pencil → pen + pencil = 6 :=
by
  intros pen pencil
  intro h1
  intro h2
  sorry

end NUMINAMATH_GPT_pen_and_pencil_total_cost_l1116_111672


namespace NUMINAMATH_GPT_jamal_green_marbles_l1116_111668

theorem jamal_green_marbles
  (Y B K T : ℕ)
  (hY : Y = 12)
  (hB : B = 10)
  (hK : K = 1)
  (h_total : 1 / T = 1 / 28) :
  T - (Y + B + K) = 5 :=
by
  -- sorry, proof goes here
  sorry

end NUMINAMATH_GPT_jamal_green_marbles_l1116_111668


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1116_111619

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 : ℤ) (h1 : a_1 = -2017) 
  (h2 : (S 2009) / 2009 - (S 2007) / 2007 = 2) : 
  S 2017 = -2017 :=
by
  -- definitions and steps would go here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1116_111619


namespace NUMINAMATH_GPT_sum_first_9000_terms_l1116_111611

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sum_first_9000_terms_l1116_111611


namespace NUMINAMATH_GPT_complex_expression_is_none_of_the_above_l1116_111691

-- We define the problem in Lean, stating that the given complex expression is not equal to any of the simplified forms
theorem complex_expression_is_none_of_the_above (x : ℝ) :
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x^3+1)^4 ) ∧
  ( ( ((x+1)^2*(x^2-x+2)^2) / (x^3+1)^2 )^2 * ( ((x-1)^2*(x^2+x+2)^2) / (x^3-2)^2 )^2 ≠ (x-1)^4 ) :=
sorry

end NUMINAMATH_GPT_complex_expression_is_none_of_the_above_l1116_111691


namespace NUMINAMATH_GPT_additional_songs_added_l1116_111656

theorem additional_songs_added (original_songs : ℕ) (song_duration : ℕ) (total_duration : ℕ) :
  original_songs = 25 → song_duration = 3 → total_duration = 105 → 
  (total_duration - original_songs * song_duration) / song_duration = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_additional_songs_added_l1116_111656


namespace NUMINAMATH_GPT_erica_riding_time_is_65_l1116_111692

-- Definition of Dave's riding time
def dave_time : ℕ := 10

-- Definition of Chuck's riding time based on Dave's time
def chuck_time (dave_time : ℕ) : ℕ := 5 * dave_time

-- Definition of Erica's additional riding time calculated as 30% of Chuck's time
def erica_additional_time (chuck_time : ℕ) : ℕ := (30 * chuck_time) / 100

-- Definition of Erica's total riding time as Chuck's time plus her additional time
def erica_total_time (chuck_time : ℕ) (erica_additional_time : ℕ) : ℕ := chuck_time + erica_additional_time

-- The proof problem: Erica's total riding time should be 65 minutes.
theorem erica_riding_time_is_65 : erica_total_time (chuck_time dave_time) (erica_additional_time (chuck_time dave_time)) = 65 :=
by
  -- The proof is skipped here
  sorry

end NUMINAMATH_GPT_erica_riding_time_is_65_l1116_111692


namespace NUMINAMATH_GPT_find_single_digit_number_l1116_111676

-- Define the given conditions:
def single_digit (A : ℕ) := A < 10
def rounded_down_tens (x : ℕ) (result: ℕ) := (x / 10) * 10 = result

-- Lean statement of the problem:
theorem find_single_digit_number (A : ℕ) (H1 : single_digit A) (H2 : rounded_down_tens (A * 1000 + 567) 2560) : A = 2 :=
sorry

end NUMINAMATH_GPT_find_single_digit_number_l1116_111676


namespace NUMINAMATH_GPT_number_of_days_at_Tom_house_l1116_111681

-- Define the constants and conditions
def total_people := 6
def plates_per_person_per_day := 6
def total_plates := 144

-- Prove that the number of days they were at Tom's house is 4
theorem number_of_days_at_Tom_house : total_plates / (total_people * plates_per_person_per_day) = 4 :=
  sorry

end NUMINAMATH_GPT_number_of_days_at_Tom_house_l1116_111681


namespace NUMINAMATH_GPT_min_value_problem_inequality_solution_l1116_111625

-- Definition of the function
noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Part (i): Minimum value problem
theorem min_value_problem (a : ℝ) (minF : ∀ x : ℝ, f x a ≥ 2) : a = 0 ∨ a = -4 :=
by
  sorry

-- Part (ii): Inequality solving problem
theorem inequality_solution (x : ℝ) (a : ℝ := 2) : f x a ≤ 6 ↔ -3 ≤ x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_problem_inequality_solution_l1116_111625


namespace NUMINAMATH_GPT_maximum_value_of_a_l1116_111686

theorem maximum_value_of_a {x y a : ℝ} (hx : x > 1 / 3) (hy : y > 1) :
  (∀ x y, x > 1 / 3 → y > 1 → 9 * x^2 / (a^2 * (y - 1)) + y^2 / (a^2 * (3 * x - 1)) ≥ 1)
  ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_a_l1116_111686


namespace NUMINAMATH_GPT_min_k_l1116_111635

def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℚ :=
  a_n n / 3^n

def T_n (n : ℕ) : ℚ :=
  (List.range n).foldl (λ acc i => acc + b_n (i + 1)) 0

theorem min_k (k : ℕ) (h : ∀ n : ℕ, n ≥ k → |T_n n - 3/4| < 1/(4*n)) : k = 4 :=
  sorry

end NUMINAMATH_GPT_min_k_l1116_111635


namespace NUMINAMATH_GPT_car_speed_first_hour_l1116_111653

theorem car_speed_first_hour (x : ℝ) (h_second_hour_speed : x + 80 / 2 = 85) : x = 90 :=
sorry

end NUMINAMATH_GPT_car_speed_first_hour_l1116_111653


namespace NUMINAMATH_GPT_original_price_calculation_l1116_111628

variable (P : ℝ)
variable (selling_price : ℝ := 1040)
variable (loss_percentage : ℝ := 20)

theorem original_price_calculation :
  P = 1300 :=
by
  have sell_percent := 100 - loss_percentage
  have SP_eq := selling_price = (sell_percent / 100) * P
  sorry

end NUMINAMATH_GPT_original_price_calculation_l1116_111628


namespace NUMINAMATH_GPT_vertex_of_parabola_l1116_111604

theorem vertex_of_parabola (a : ℝ) :
  (∃ (k : ℝ), ∀ x : ℝ, y = -4*x - 1 → x = 2 ∧ (a - 4) = -4 * 2 - 1) → 
  (2, -9) = (2, a - 4) → a = -5 :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1116_111604


namespace NUMINAMATH_GPT_base_k_number_eq_binary_l1116_111674

theorem base_k_number_eq_binary (k : ℕ) (h : k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end NUMINAMATH_GPT_base_k_number_eq_binary_l1116_111674


namespace NUMINAMATH_GPT_base_representing_350_as_four_digit_number_with_even_final_digit_l1116_111665

theorem base_representing_350_as_four_digit_number_with_even_final_digit {b : ℕ} :
  b ^ 3 ≤ 350 ∧ 350 < b ^ 4 ∧ (∃ d1 d2 d3 d4, 350 = d1 * b^3 + d2 * b^2 + d3 * b + d4 ∧ d4 % 2 = 0) ↔ b = 6 :=
by sorry

end NUMINAMATH_GPT_base_representing_350_as_four_digit_number_with_even_final_digit_l1116_111665


namespace NUMINAMATH_GPT_num_20_paise_coins_l1116_111661

theorem num_20_paise_coins (x y : ℕ) (h1 : x + y = 344) (h2 : 20 * x + 25 * y = 7100) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_num_20_paise_coins_l1116_111661


namespace NUMINAMATH_GPT_sale_price_60_l1116_111678

theorem sale_price_60 (original_price : ℕ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) 
  (h2 : discount_percentage = 0.40) :
  sale_price = (original_price : ℝ) * (1 - discount_percentage) :=
by
  sorry

end NUMINAMATH_GPT_sale_price_60_l1116_111678


namespace NUMINAMATH_GPT_not_divisor_60_l1116_111673

variable (k : ℤ)
def n : ℤ := k * (k + 1) * (k + 2)

theorem not_divisor_60 
  (h₁ : ∃ k, n = k * (k + 1) * (k + 2) ∧ 5 ∣ n) : ¬(60 ∣ n) := 
sorry

end NUMINAMATH_GPT_not_divisor_60_l1116_111673


namespace NUMINAMATH_GPT_jovana_added_23_pounds_l1116_111632

def initial_weight : ℕ := 5
def final_weight : ℕ := 28

def added_weight : ℕ := final_weight - initial_weight

theorem jovana_added_23_pounds : added_weight = 23 := 
by sorry

end NUMINAMATH_GPT_jovana_added_23_pounds_l1116_111632


namespace NUMINAMATH_GPT_smallest_number_value_l1116_111601

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  c = 2 * a ∧
  c - b = 10

theorem smallest_number_value (h : conditions a b c) : a = 22 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_value_l1116_111601


namespace NUMINAMATH_GPT_range_of_function_l1116_111627

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x < 5 → -4 ≤ f x ∧ f x < 5 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_range_of_function_l1116_111627


namespace NUMINAMATH_GPT_distinct_meals_l1116_111614

-- Define the conditions
def number_of_entrees : ℕ := 4
def number_of_drinks : ℕ := 3
def number_of_desserts : ℕ := 2

-- Define the main theorem
theorem distinct_meals : number_of_entrees * number_of_drinks * number_of_desserts = 24 := 
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_distinct_meals_l1116_111614


namespace NUMINAMATH_GPT_crushing_load_l1116_111651

theorem crushing_load (T H C : ℝ) (L : ℝ) 
  (h1 : T = 5) (h2 : H = 10) (h3 : C = 3)
  (h4 : L = C * 25 * T^4 / H^2) : 
  L = 468.75 :=
by
  sorry

end NUMINAMATH_GPT_crushing_load_l1116_111651


namespace NUMINAMATH_GPT_triangle_angle_l1116_111613

variable (a b c : ℝ)
variable (C : ℝ)

theorem triangle_angle (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) :
  C = Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2))) :=
sorry

end NUMINAMATH_GPT_triangle_angle_l1116_111613


namespace NUMINAMATH_GPT_marble_ratio_is_two_to_one_l1116_111664

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end NUMINAMATH_GPT_marble_ratio_is_two_to_one_l1116_111664


namespace NUMINAMATH_GPT_sum_of_104th_parenthesis_is_correct_l1116_111633

def b (n : ℕ) : ℕ := 2 * n + 1

def sumOf104thParenthesis : ℕ :=
  let cycleCount := 104 / 4
  let numbersBefore104 := 260
  let firstNumIndex := numbersBefore104 + 1
  let firstNum := b firstNumIndex
  let secondNum := b (firstNumIndex + 1)
  let thirdNum := b (firstNumIndex + 2)
  let fourthNum := b (firstNumIndex + 3)
  firstNum + secondNum + thirdNum + fourthNum

theorem sum_of_104th_parenthesis_is_correct : sumOf104thParenthesis = 2104 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_104th_parenthesis_is_correct_l1116_111633


namespace NUMINAMATH_GPT_total_winter_clothing_l1116_111649

def num_scarves (boxes : ℕ) (scarves_per_box : ℕ) : ℕ := boxes * scarves_per_box
def num_mittens (boxes : ℕ) (mittens_per_box : ℕ) : ℕ := boxes * mittens_per_box
def num_hats (boxes : ℕ) (hats_per_box : ℕ) : ℕ := boxes * hats_per_box
def num_jackets (boxes : ℕ) (jackets_per_box : ℕ) : ℕ := boxes * jackets_per_box

theorem total_winter_clothing :
    num_scarves 4 8 + num_mittens 3 6 + num_hats 2 5 + num_jackets 1 3 = 63 :=
by
  -- The proof will use the given definitions and calculate the total
  sorry

end NUMINAMATH_GPT_total_winter_clothing_l1116_111649


namespace NUMINAMATH_GPT_sandy_more_tokens_than_siblings_l1116_111643

-- Define the initial conditions
def initial_tokens : ℕ := 3000000
def initial_transaction_fee_percent : ℚ := 0.10
def value_increase_percent : ℚ := 0.20
def additional_tokens : ℕ := 500000
def additional_transaction_fee_percent : ℚ := 0.07
def sandy_keep_percent : ℚ := 0.40
def siblings : ℕ := 7
def sibling_transaction_fee_percent : ℚ := 0.05

-- Define the main theorem to prove
theorem sandy_more_tokens_than_siblings :
  let received_initial_tokens := initial_tokens * (1 - initial_transaction_fee_percent)
  let increased_tokens := received_initial_tokens * (1 + value_increase_percent)
  let received_additional_tokens := additional_tokens * (1 - additional_transaction_fee_percent)
  let total_tokens := increased_tokens + received_additional_tokens
  let sandy_tokens := total_tokens * sandy_keep_percent
  let remaining_tokens := total_tokens * (1 - sandy_keep_percent)
  let each_sibling_tokens := remaining_tokens / siblings * (1 - sibling_transaction_fee_percent)
  sandy_tokens - each_sibling_tokens = 1180307.1428 := sorry

end NUMINAMATH_GPT_sandy_more_tokens_than_siblings_l1116_111643


namespace NUMINAMATH_GPT_explicit_form_of_function_l1116_111660

theorem explicit_form_of_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f x + f x * f y + y - 1) = f (x * f x + x * y) + y - 1) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_explicit_form_of_function_l1116_111660


namespace NUMINAMATH_GPT_value_of_fraction_l1116_111698

open Real

theorem value_of_fraction (a : ℝ) (h : a^2 + a - 1 = 0) : (1 - a) / a + a / (1 + a) = 1 := 
by { sorry }

end NUMINAMATH_GPT_value_of_fraction_l1116_111698


namespace NUMINAMATH_GPT_john_yasmin_child_ratio_l1116_111609

theorem john_yasmin_child_ratio
  (gabriel_grandkids : ℕ)
  (yasmin_children : ℕ)
  (john_children : ℕ)
  (h1 : gabriel_grandkids = 6)
  (h2 : yasmin_children = 2)
  (h3 : john_children + yasmin_children = gabriel_grandkids) :
  john_children / yasmin_children = 2 :=
by 
  sorry

end NUMINAMATH_GPT_john_yasmin_child_ratio_l1116_111609


namespace NUMINAMATH_GPT_remainder_of_towers_l1116_111654

open Nat

def count_towers (m : ℕ) : ℕ :=
  match m with
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 18
  | 5 => 54
  | 6 => 162
  | _ => 0

theorem remainder_of_towers : (count_towers 6) % 100 = 62 :=
  by
  sorry

end NUMINAMATH_GPT_remainder_of_towers_l1116_111654


namespace NUMINAMATH_GPT_max_unique_sums_l1116_111607

-- Define the coin values in cents
def penny := 1
def nickel := 5
def quarter := 25
def half_dollar := 50

-- Define the set of all coins and their counts
structure Coins :=
  (pennies : ℕ := 3)
  (nickels : ℕ := 3)
  (quarters : ℕ := 1)
  (half_dollars : ℕ := 2)

-- Define the list of all possible pairs and their sums
def possible_sums : Finset ℕ :=
  { 2, 6, 10, 26, 30, 51, 55, 75, 100 }

-- Prove that the count of unique sums is 9
theorem max_unique_sums (c : Coins) : c.pennies = 3 → c.nickels = 3 → c.quarters = 1 → c.half_dollars = 2 →
  possible_sums.card = 9 := 
by
  intros
  sorry

end NUMINAMATH_GPT_max_unique_sums_l1116_111607


namespace NUMINAMATH_GPT_card_tag_sum_l1116_111642

noncomputable def W : ℕ := 200
noncomputable def X : ℝ := 2 / 3 * W
noncomputable def Y : ℝ := W + X
noncomputable def Z : ℝ := Real.sqrt Y
noncomputable def P : ℝ := X^3
noncomputable def Q : ℝ := Nat.factorial W / 100000
noncomputable def R : ℝ := 3 / 5 * (P + Q)
noncomputable def S : ℝ := W^1 + X^2 + Z^3

theorem card_tag_sum :
  W + X + Y + Z + P + S = 2373589.26 + Q + R :=
by
  sorry

end NUMINAMATH_GPT_card_tag_sum_l1116_111642


namespace NUMINAMATH_GPT_correct_time_fraction_l1116_111605

theorem correct_time_fraction : (3 / 4 : ℝ) * (3 / 4 : ℝ) = (9 / 16 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_correct_time_fraction_l1116_111605


namespace NUMINAMATH_GPT_sqrt_function_of_x_l1116_111624

theorem sqrt_function_of_x (x : ℝ) (h : x > 0) : ∃! y : ℝ, y = Real.sqrt x :=
by
  sorry

end NUMINAMATH_GPT_sqrt_function_of_x_l1116_111624


namespace NUMINAMATH_GPT_right_triangle_to_acute_triangle_l1116_111610

theorem right_triangle_to_acute_triangle 
  (a b c d : ℝ) (h_triangle : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_increase : d > 0):
  (a + d)^2 + (b + d)^2 > (c + d)^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_to_acute_triangle_l1116_111610


namespace NUMINAMATH_GPT_exists_a_max_value_of_four_l1116_111670

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * a * Real.sin x + 3 * a - 1

theorem exists_a_max_value_of_four :
  ∃ a : ℝ, (a = 1) ∧ ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f a x ≤ 4 := 
sorry

end NUMINAMATH_GPT_exists_a_max_value_of_four_l1116_111670


namespace NUMINAMATH_GPT_gear_q_revolutions_per_minute_l1116_111638

-- Define the constants and conditions
def revolutions_per_minute_p : ℕ := 10
def revolutions_per_minute_q : ℕ := sorry
def time_in_minutes : ℝ := 1.5
def extra_revolutions_q : ℕ := 45

-- Calculate the number of revolutions for gear p in 90 seconds
def revolutions_p_in_90_seconds := revolutions_per_minute_p * time_in_minutes

-- Condition that gear q makes exactly 45 more revolutions than gear p in 90 seconds
def revolutions_q_in_90_seconds := revolutions_p_in_90_seconds + extra_revolutions_q

-- Correct answer
def correct_answer : ℕ := 40

-- Prove that gear q makes 40 revolutions per minute
theorem gear_q_revolutions_per_minute : 
    revolutions_per_minute_q = correct_answer :=
sorry

end NUMINAMATH_GPT_gear_q_revolutions_per_minute_l1116_111638


namespace NUMINAMATH_GPT_math_problem_l1116_111682

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end NUMINAMATH_GPT_math_problem_l1116_111682


namespace NUMINAMATH_GPT_sum_of_coordinates_B_l1116_111683

theorem sum_of_coordinates_B
  (x y : ℤ)
  (Mx My : ℤ)
  (Ax Ay : ℤ)
  (M : Mx = 2 ∧ My = -3)
  (A : Ax = -4 ∧ Ay = -5)
  (midpoint_x : (x + Ax) / 2 = Mx)
  (midpoint_y : (y + Ay) / 2 = My) :
  x + y = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_B_l1116_111683


namespace NUMINAMATH_GPT_value_of_livestock_l1116_111600

variable (x y : ℝ)

theorem value_of_livestock :
  (5 * x + 2 * y = 10) ∧ (2 * x + 5 * y = 8) :=
sorry

end NUMINAMATH_GPT_value_of_livestock_l1116_111600


namespace NUMINAMATH_GPT_platform_length_l1116_111636

theorem platform_length (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) (speed : ℝ) (platform_length : ℝ) :
  train_length = 300 → time_pole = 18 → time_platform = 38 → speed = train_length / time_pole →
  platform_length = (speed * time_platform) - train_length → platform_length = 333.46 :=
by
  introv h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_platform_length_l1116_111636


namespace NUMINAMATH_GPT_acute_angle_30_l1116_111680

theorem acute_angle_30 (α : ℝ) (h : Real.cos (π / 6) * Real.sin α = Real.sqrt 3 / 4) : α = π / 6 := 
by 
  sorry

end NUMINAMATH_GPT_acute_angle_30_l1116_111680


namespace NUMINAMATH_GPT_eval_expression_l1116_111659

theorem eval_expression : -30 + 12 * (8 / 4)^2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1116_111659


namespace NUMINAMATH_GPT_problem_statement_l1116_111631

theorem problem_statement : ¬ (487.5 * 10^(-10) = 0.0000004875) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1116_111631


namespace NUMINAMATH_GPT_number_of_distinguishable_large_triangles_l1116_111639

theorem number_of_distinguishable_large_triangles (colors : Fin 8) :
  ∃(large_triangles : Fin 960), true :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinguishable_large_triangles_l1116_111639


namespace NUMINAMATH_GPT_total_possible_match_sequences_l1116_111658

theorem total_possible_match_sequences :
  let num_teams := 2
  let team_size := 7
  let possible_sequences := 2 * (Nat.choose (2 * team_size - 1) (team_size - 1))
  possible_sequences = 3432 :=
by
  sorry

end NUMINAMATH_GPT_total_possible_match_sequences_l1116_111658


namespace NUMINAMATH_GPT_remainder_23_pow_2003_mod_7_l1116_111623

theorem remainder_23_pow_2003_mod_7 : 23 ^ 2003 % 7 = 4 :=
by sorry

end NUMINAMATH_GPT_remainder_23_pow_2003_mod_7_l1116_111623


namespace NUMINAMATH_GPT_principal_sum_investment_l1116_111626

theorem principal_sum_investment 
    (P R : ℝ) 
    (h1 : (P * 5 * (R + 2)) / 100 - (P * 5 * R) / 100 = 180)
    (h2 : (P * 5 * (R + 3)) / 100 - (P * 5 * R) / 100 = 270) :
    P = 1800 :=
by
  -- These are the hypotheses generated for Lean, the proof steps are omitted
  sorry

end NUMINAMATH_GPT_principal_sum_investment_l1116_111626


namespace NUMINAMATH_GPT_total_shaded_area_approx_l1116_111697

noncomputable def area_of_shaded_regions (r1 r2 : ℝ) :=
  let area_smaller_circle := 3 * 6 - (1 / 2) * Real.pi * r1^2
  let area_larger_circle := 6 * 12 - (1 / 2) * Real.pi * r2^2
  area_smaller_circle + area_larger_circle

theorem total_shaded_area_approx :
  abs (area_of_shaded_regions 3 6 - 19.4) < 0.05 :=
by
  sorry

end NUMINAMATH_GPT_total_shaded_area_approx_l1116_111697


namespace NUMINAMATH_GPT_periodic_function_of_f_l1116_111608

theorem periodic_function_of_f (f : ℝ → ℝ) (c : ℝ) (h : ∀ x, f (x + c) = (2 / (1 + f x)) - 1) : ∀ x, f (x + 2 * c) = f x :=
sorry

end NUMINAMATH_GPT_periodic_function_of_f_l1116_111608


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l1116_111684

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α < 60) (h3 : β < 60) (h4 : γ < 60) : false := 
sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l1116_111684


namespace NUMINAMATH_GPT_exist_two_divisible_by_n_l1116_111671

theorem exist_two_divisible_by_n (n : ℤ) (a : Fin (n.toNat + 1) → ℤ) :
  ∃ (i j : Fin (n.toNat + 1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_exist_two_divisible_by_n_l1116_111671


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1116_111618

theorem eccentricity_of_hyperbola :
  let a := Real.sqrt 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (∃ (x y : ℝ), (x^2 / 5) - (y^2 / 4) = 1 ∧ e = (3 * Real.sqrt 5) / 5) := sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1116_111618


namespace NUMINAMATH_GPT_trajectory_eqn_l1116_111655

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Conditions given in the problem
def PA_squared (P : ℝ × ℝ) : ℝ := (P.1 + 1)^2 + P.2^2
def PB_squared (P : ℝ × ℝ) : ℝ := (P.1 - 1)^2 + P.2^2

-- The main statement to prove
theorem trajectory_eqn (P : ℝ × ℝ) (h : PA_squared P = 3 * PB_squared P) : 
  P.1^2 + P.2^2 - 4 * P.1 + 1 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_trajectory_eqn_l1116_111655


namespace NUMINAMATH_GPT_measure_of_angle_B_l1116_111650

theorem measure_of_angle_B (a b c R : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C)
  (h4 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_l1116_111650


namespace NUMINAMATH_GPT_arithmetic_series_sum_proof_middle_term_proof_l1116_111617

def arithmetic_series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

def middle_term (a l : ℤ) : ℤ :=
  (a + l) / 2

theorem arithmetic_series_sum_proof :
  let a := -51
  let d := 2
  let n := 27
  let l := 1
  arithmetic_series_sum a d n = -675 :=
by
  sorry

theorem middle_term_proof :
  let a := -51
  let l := 1
  middle_term a l = -25 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_proof_middle_term_proof_l1116_111617


namespace NUMINAMATH_GPT_sum_of_opposite_numbers_is_zero_l1116_111648

theorem sum_of_opposite_numbers_is_zero {a b : ℝ} (h : a + b = 0) : a + b = 0 := 
h

end NUMINAMATH_GPT_sum_of_opposite_numbers_is_zero_l1116_111648


namespace NUMINAMATH_GPT_problem_part_I_problem_part_II_l1116_111612

theorem problem_part_I (A B C : ℝ)
  (h1 : 0 < A) 
  (h2 : A < π / 2)
  (h3 : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2) : 
  A = π / 3 := 
sorry

theorem problem_part_II (A B C R S : ℝ)
  (h1 : A = π / 3)
  (h2 : R = 2 * Real.sqrt 3) 
  (h3 : S = (1 / 2) * (6 * (Real.sin A)) * (Real.sqrt 3 / 2)) :
  S = 9 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_problem_part_I_problem_part_II_l1116_111612


namespace NUMINAMATH_GPT_find_f_1_div_2007_l1116_111689

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_1_div_2007 :
  f 0 = 0 ∧
  (∀ x, f x + f (1 - x) = 1) ∧
  (∀ x, f (x / 5) = f x / 2) ∧
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f x1 ≤ f x2) →
  f (1 / 2007) = 1 / 32 :=
sorry

end NUMINAMATH_GPT_find_f_1_div_2007_l1116_111689


namespace NUMINAMATH_GPT_statement_B_statement_D_l1116_111685

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.cos x + Real.sqrt 3 * Real.sin x) - Real.sqrt 3 + 1

theorem statement_B (x₁ x₂ : ℝ) (h1 : -π / 12 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 5 * π / 12) :
  f x₁ < f x₂ := sorry

theorem statement_D (x₁ x₂ x₃ : ℝ) (h1 : π / 3 ≤ x₁) (h2 : x₁ ≤ π / 2) (h3 : π / 3 ≤ x₂) (h4 : x₂ ≤ π / 2) (h5 : π / 3 ≤ x₃) (h6 : x₃ ≤ π / 2) :
  f x₁ + f x₂ - f x₃ > 2 := sorry

end NUMINAMATH_GPT_statement_B_statement_D_l1116_111685


namespace NUMINAMATH_GPT_mass_of_fat_max_mass_of_carbohydrates_l1116_111615

-- Definitions based on conditions
def total_mass : ℤ := 500
def fat_percentage : ℚ := 5 / 100
def protein_to_mineral_ratio : ℤ := 4

-- Lean 4 statement for Part 1: mass of fat
theorem mass_of_fat : (total_mass : ℚ) * fat_percentage = 25 := sorry

-- Definitions to utilize in Part 2
def max_percentage_protein_carbs : ℚ := 85 / 100
def mass_protein (x : ℚ) : ℚ := protein_to_mineral_ratio * x

-- Lean 4 statement for Part 2: maximum mass of carbohydrates
theorem max_mass_of_carbohydrates (x : ℚ) :
  x ≥ 50 → (total_mass - 25 - x - mass_protein x) ≤ 225 := sorry

end NUMINAMATH_GPT_mass_of_fat_max_mass_of_carbohydrates_l1116_111615


namespace NUMINAMATH_GPT_standard_equation_of_circle_tangent_to_x_axis_l1116_111688

theorem standard_equation_of_circle_tangent_to_x_axis :
  ∀ (x y : ℝ), ((x + 3) ^ 2 + (y - 4) ^ 2 = 16) :=
by
  -- Definitions based on the conditions
  let center_x := -3
  let center_y := 4
  let radius := 4

  sorry

end NUMINAMATH_GPT_standard_equation_of_circle_tangent_to_x_axis_l1116_111688


namespace NUMINAMATH_GPT_average_speed_of_car_l1116_111645

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end NUMINAMATH_GPT_average_speed_of_car_l1116_111645


namespace NUMINAMATH_GPT_max_k_value_l1116_111687

theorem max_k_value :
  ∃ A B C k : ℕ, 
  (A ≠ 0) ∧ 
  (A < 10) ∧ 
  (B < 10) ∧ 
  (C < 10) ∧
  (10 * A + B) * k = 100 * A + 10 * C + B ∧
  (∀ k' : ℕ, 
     ((A ≠ 0) ∧ (A < 10) ∧ (B < 10) ∧ (C < 10) ∧
     (10 * A + B) * k' = 100 * A + 10 * C + B) 
     → k' ≤ 19) ∧
  k = 19 :=
sorry

end NUMINAMATH_GPT_max_k_value_l1116_111687


namespace NUMINAMATH_GPT_simplify_expression_l1116_111640

variable (a : ℝ) (ha : a ≠ -3)

theorem simplify_expression : (a^2) / (a + 3) - 9 / (a + 3) = a - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1116_111640


namespace NUMINAMATH_GPT_factor_expression_l1116_111644

theorem factor_expression (x : ℝ) :
  x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := 
  sorry

end NUMINAMATH_GPT_factor_expression_l1116_111644


namespace NUMINAMATH_GPT_second_quadrant_coordinates_l1116_111620

theorem second_quadrant_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : y^2 = 1) :
    (x, y) = (-2, 1) :=
  sorry

end NUMINAMATH_GPT_second_quadrant_coordinates_l1116_111620


namespace NUMINAMATH_GPT_range_of_a_l1116_111677

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ -1 → f a x ≥ a) : -3 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1116_111677


namespace NUMINAMATH_GPT_suitable_sampling_method_l1116_111694

noncomputable def is_stratified_sampling_suitable (mountainous hilly flat low_lying sample_size : ℕ) (yield_dependent_on_land_type : Bool) : Bool :=
  if yield_dependent_on_land_type && mountainous + hilly + flat + low_lying > 0 then true else false

theorem suitable_sampling_method :
  is_stratified_sampling_suitable 8000 12000 24000 4000 480 true = true :=
by
  sorry

end NUMINAMATH_GPT_suitable_sampling_method_l1116_111694
