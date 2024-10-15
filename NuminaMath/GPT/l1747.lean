import Mathlib

namespace NUMINAMATH_GPT_find_pairs_l1747_174756

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem find_pairs (a n : ℕ) (h1 : a ≥ n) (h2 : is_power_of_two ((a + 1)^n + a - 1)) :
  (a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1747_174756


namespace NUMINAMATH_GPT_term_2005_is_1004th_l1747_174790

-- Define the first term and the common difference
def a1 : Int := -1
def d : Int := 2

-- Define the general term formula of the arithmetic sequence
def a_n (n : Nat) : Int :=
  a1 + (n - 1) * d

-- State the theorem that the year 2005 is the 1004th term in the sequence
theorem term_2005_is_1004th : ∃ n : Nat, a_n n = 2005 ∧ n = 1004 := by
  sorry

end NUMINAMATH_GPT_term_2005_is_1004th_l1747_174790


namespace NUMINAMATH_GPT_triangle_problem_l1747_174706

noncomputable def find_b (a b c : ℝ) : Prop :=
  let B : ℝ := 60 * Real.pi / 180 -- converting 60 degrees to radians
  b = 2 * Real.sqrt 2

theorem triangle_problem
  (a b c : ℝ)
  (h_area : (1 / 2) * a * c * Real.sin (60 * Real.pi / 180) = Real.sqrt 3)
  (h_cosine : a^2 + c^2 = 3 * a * c) : find_b a b c :=
by
  -- The proof would go here, but we're skipping it as per the instructions.
  sorry

end NUMINAMATH_GPT_triangle_problem_l1747_174706


namespace NUMINAMATH_GPT_minimum_value_18_sqrt_3_minimum_value_at_x_3_l1747_174718

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 12*x + 81 / x^3

theorem minimum_value_18_sqrt_3 (x : ℝ) (hx : x > 0) :
  f x ≥ 18 * Real.sqrt 3 :=
by
  sorry

theorem minimum_value_at_x_3 : f 3 = 18 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_18_sqrt_3_minimum_value_at_x_3_l1747_174718


namespace NUMINAMATH_GPT_meaningful_iff_x_ne_1_l1747_174743

theorem meaningful_iff_x_ne_1 (x : ℝ) : (x - 1) ≠ 0 ↔ (x ≠ 1) :=
by 
  sorry

end NUMINAMATH_GPT_meaningful_iff_x_ne_1_l1747_174743


namespace NUMINAMATH_GPT_negation_proposition_l1747_174734

open Classical

variable (x : ℝ)

def proposition (x : ℝ) : Prop := ∀ x > 1, Real.log x / Real.log 2 > 0

theorem negation_proposition (h : ¬ proposition x) : 
  ∃ x > 1, Real.log x / Real.log 2 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1747_174734


namespace NUMINAMATH_GPT_total_items_on_shelf_l1747_174711

-- Given conditions
def initial_action_figures : Nat := 4
def initial_books : Nat := 22
def initial_video_games : Nat := 10

def added_action_figures : Nat := 6
def added_video_games : Nat := 3
def removed_books : Nat := 5

-- Definitions based on conditions
def final_action_figures : Nat := initial_action_figures + added_action_figures
def final_books : Nat := initial_books - removed_books
def final_video_games : Nat := initial_video_games + added_video_games

-- Claim to prove
theorem total_items_on_shelf : final_action_figures + final_books + final_video_games = 40 := by
  sorry

end NUMINAMATH_GPT_total_items_on_shelf_l1747_174711


namespace NUMINAMATH_GPT_mrs_franklin_needs_more_valentines_l1747_174701

theorem mrs_franklin_needs_more_valentines (valentines_have : ℝ) (students : ℝ) : valentines_have = 58 ∧ students = 74 → students - valentines_have = 16 :=
by
  sorry

end NUMINAMATH_GPT_mrs_franklin_needs_more_valentines_l1747_174701


namespace NUMINAMATH_GPT_n_plus_5_divisible_by_6_l1747_174712

theorem n_plus_5_divisible_by_6 (n : ℕ) (h1 : (n + 2) % 3 = 0) (h2 : (n + 3) % 4 = 0) : (n + 5) % 6 = 0 := 
sorry

end NUMINAMATH_GPT_n_plus_5_divisible_by_6_l1747_174712


namespace NUMINAMATH_GPT_age_proof_l1747_174795

theorem age_proof (A B C D : ℕ) 
  (h1 : A = D + 16)
  (h2 : B = D + 8)
  (h3 : C = D + 4)
  (h4 : A - 6 = 3 * (D - 6))
  (h5 : A - 6 = 2 * (B - 6))
  (h6 : A - 6 = (C - 6) + 4) 
  : A = 30 ∧ B = 22 ∧ C = 18 ∧ D = 14 :=
sorry

end NUMINAMATH_GPT_age_proof_l1747_174795


namespace NUMINAMATH_GPT_sprinkler_days_needed_l1747_174785

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end NUMINAMATH_GPT_sprinkler_days_needed_l1747_174785


namespace NUMINAMATH_GPT_not_obtain_other_than_given_set_l1747_174763

theorem not_obtain_other_than_given_set : 
  ∀ (x : ℝ), x = 1 → 
  ∃ (n : ℕ → ℝ), (n 0 = 1) ∧ 
  (∀ k, n (k + 1) = n k + 1 ∨ n (k + 1) = -1 / n k) ∧
  (x = -2 ∨ x = 1/2 ∨ x = 5/3 ∨ x = 7) → 
  ∃ k, x = n k :=
sorry

end NUMINAMATH_GPT_not_obtain_other_than_given_set_l1747_174763


namespace NUMINAMATH_GPT_jenna_age_l1747_174736

theorem jenna_age (D J : ℕ) (h1 : J = D + 5) (h2 : J + D = 21) (h3 : D = 8) : J = 13 :=
by
  sorry

end NUMINAMATH_GPT_jenna_age_l1747_174736


namespace NUMINAMATH_GPT_percentage_profit_without_discount_l1747_174724

variable (CP : ℝ) (discountRate profitRate noDiscountProfitRate : ℝ)

theorem percentage_profit_without_discount 
  (hCP : CP = 100)
  (hDiscount : discountRate = 0.04)
  (hProfit : profitRate = 0.26)
  (hNoDiscountProfit : noDiscountProfitRate = 0.3125) :
  let SP := CP * (1 + profitRate)
  let MP := SP / (1 - discountRate)
  noDiscountProfitRate = (MP - CP) / CP :=
by
  sorry

end NUMINAMATH_GPT_percentage_profit_without_discount_l1747_174724


namespace NUMINAMATH_GPT_polynomial_coefficient_sum_l1747_174770

theorem polynomial_coefficient_sum :
  ∀ (a0 a1 a2 a3 a4 a5 : ℤ), 
  (3 - 2 * x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 → 
  a0 + a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 = 233 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_sum_l1747_174770


namespace NUMINAMATH_GPT_store_total_income_l1747_174742

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end NUMINAMATH_GPT_store_total_income_l1747_174742


namespace NUMINAMATH_GPT_area_of_rectangle_l1747_174793

-- Define the conditions
variable {S1 S2 S3 S4 : ℝ} -- side lengths of the four squares

-- The conditions:
-- 1. Four non-overlapping squares
-- 2. The area of the shaded square is 4 square inches
def conditions (S1 S2 S3 S4 : ℝ) : Prop :=
    S1^2 = 4 -- Given that one of the squares has an area of 4 square inches

-- The proof problem:
theorem area_of_rectangle (S1 S2 S3 S4 : ℝ) (h1 : 2 * S1 = S2) (h2 : 2 * S2 = S3) (h3 : conditions S1 S2 S3 S4) : 
    S1^2 + S2^2 + S3^2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1747_174793


namespace NUMINAMATH_GPT_students_with_all_three_pets_l1747_174728

variables (TotalStudents HaveDogs HaveCats HaveOtherPets NoPets x y z w : ℕ)

theorem students_with_all_three_pets :
  TotalStudents = 40 →
  HaveDogs = 20 →
  HaveCats = 16 →
  HaveOtherPets = 8 →
  NoPets = 7 →
  x = 12 →
  y = 3 →
  z = 11 →
  TotalStudents - NoPets = 33 →
  x + y + w = HaveDogs →
  z + w = HaveCats →
  y + w = HaveOtherPets →
  x + y + z + w = 33 →
  w = 5 :=
by
  intros h1 h2 h3 h4 h5 hx hy hz h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_students_with_all_three_pets_l1747_174728


namespace NUMINAMATH_GPT_train_cars_estimate_l1747_174775

noncomputable def train_cars_count (total_time_secs : ℕ) (delay_secs : ℕ) (cars_counted : ℕ) (count_time_secs : ℕ): ℕ := 
  let rate_per_sec := cars_counted / count_time_secs
  let cars_missed := delay_secs * rate_per_sec
  let cars_in_remaining_time := rate_per_sec * (total_time_secs - delay_secs)
  cars_missed + cars_in_remaining_time

theorem train_cars_estimate :
  train_cars_count 210 15 8 20 = 120 :=
sorry

end NUMINAMATH_GPT_train_cars_estimate_l1747_174775


namespace NUMINAMATH_GPT_square_value_zero_l1747_174794

variable {a b : ℝ}

theorem square_value_zero (h1 : a > b) (h2 : -2 * a - 1 < -2 * b + 0) : 0 = 0 := 
by
  sorry

end NUMINAMATH_GPT_square_value_zero_l1747_174794


namespace NUMINAMATH_GPT_bc_over_ad_eq_50_point_4_l1747_174792

theorem bc_over_ad_eq_50_point_4 :
  let B := (2, 2, 5)
  let S (r : ℝ) (B : ℝ × ℝ × ℝ) := {p | dist p B ≤ r }
  let d := (20 : ℝ)
  let c := (48 : ℝ)
  let b := (28 * Real.pi : ℝ)
  let a := ((4 * Real.pi) / 3 : ℝ)
  let bc := b * c
  let ad := a * d
  bc / ad = 50.4 := by
    sorry

end NUMINAMATH_GPT_bc_over_ad_eq_50_point_4_l1747_174792


namespace NUMINAMATH_GPT_pow_mod_remainder_l1747_174700

theorem pow_mod_remainder : (3 ^ 304) % 11 = 4 := by
  sorry

end NUMINAMATH_GPT_pow_mod_remainder_l1747_174700


namespace NUMINAMATH_GPT_product_of_roots_l1747_174777

theorem product_of_roots : ∀ (x : ℝ), (x + 3) * (x - 4) = 2 * (x + 1) → 
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  product_of_roots = -14 :=
by
  intros x h
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  sorry

end NUMINAMATH_GPT_product_of_roots_l1747_174777


namespace NUMINAMATH_GPT_exponentiation_rule_l1747_174798

theorem exponentiation_rule (b : ℝ) : (-2 * b) ^ 3 = -8 * b ^ 3 :=
by sorry

end NUMINAMATH_GPT_exponentiation_rule_l1747_174798


namespace NUMINAMATH_GPT_solve_fractional_eq_l1747_174729

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) : (x / (x + 1) - 1 = 3 / (x - 1)) → x = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l1747_174729


namespace NUMINAMATH_GPT_water_percentage_in_fresh_mushrooms_l1747_174799

theorem water_percentage_in_fresh_mushrooms
  (fresh_mushrooms_mass : ℝ)
  (dried_mushrooms_mass : ℝ)
  (dried_mushrooms_water_percentage : ℝ)
  (dried_mushrooms_non_water_mass : ℝ)
  (fresh_mushrooms_dry_percentage : ℝ)
  (fresh_mushrooms_water_percentage : ℝ)
  (h1 : fresh_mushrooms_mass = 22)
  (h2 : dried_mushrooms_mass = 2.5)
  (h3 : dried_mushrooms_water_percentage = 12 / 100)
  (h4 : dried_mushrooms_non_water_mass = dried_mushrooms_mass * (1 - dried_mushrooms_water_percentage))
  (h5 : fresh_mushrooms_dry_percentage = dried_mushrooms_non_water_mass / fresh_mushrooms_mass * 100)
  (h6 : fresh_mushrooms_water_percentage = 100 - fresh_mushrooms_dry_percentage) :
  fresh_mushrooms_water_percentage = 90 := 
by
  sorry

end NUMINAMATH_GPT_water_percentage_in_fresh_mushrooms_l1747_174799


namespace NUMINAMATH_GPT_max_tan_B_l1747_174735

theorem max_tan_B (A B : ℝ) (C : Prop) 
  (sin_pos_A : 0 < Real.sin A) 
  (sin_pos_B : 0 < Real.sin B) 
  (angle_condition : Real.sin B / Real.sin A = Real.cos (A + B)) :
  Real.tan B ≤ Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_tan_B_l1747_174735


namespace NUMINAMATH_GPT_tshirts_per_package_l1747_174725

def number_of_packages := 28
def total_white_tshirts := 56
def white_tshirts_per_package : Nat :=
  total_white_tshirts / number_of_packages

theorem tshirts_per_package :
  white_tshirts_per_package = 2 :=
by
  -- Assuming the definitions and the proven facts
  sorry

end NUMINAMATH_GPT_tshirts_per_package_l1747_174725


namespace NUMINAMATH_GPT_trays_needed_to_fill_ice_cubes_l1747_174717

-- Define the initial conditions
def ice_cubes_in_glass : Nat := 8
def multiplier_for_pitcher : Nat := 2
def spaces_per_tray : Nat := 12

-- Define the total ice cubes used
def total_ice_cubes_used : Nat := ice_cubes_in_glass + multiplier_for_pitcher * ice_cubes_in_glass

-- State the Lean theorem to be proven: The number of trays needed
theorem trays_needed_to_fill_ice_cubes : 
  total_ice_cubes_used / spaces_per_tray = 2 :=
  by 
  sorry

end NUMINAMATH_GPT_trays_needed_to_fill_ice_cubes_l1747_174717


namespace NUMINAMATH_GPT_investment_scientific_notation_l1747_174704

def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ (1650000000 = a * 10^n)

theorem investment_scientific_notation :
  ∃ a n, is_scientific_notation a n ∧ a = 1.65 ∧ n = 9 :=
sorry

end NUMINAMATH_GPT_investment_scientific_notation_l1747_174704


namespace NUMINAMATH_GPT_students_in_class_l1747_174731

theorem students_in_class (S : ℕ) 
  (h1 : chess_students = S / 3)
  (h2 : tournament_students = chess_students / 2)
  (h3 : tournament_students = 4) : 
  S = 24 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l1747_174731


namespace NUMINAMATH_GPT_total_quartet_songs_l1747_174778

/-- 
Five girls — Mary, Alina, Tina, Hanna, and Elsa — sang songs in a concert as quartets,
with one girl sitting out each time. Hanna sang 9 songs, which was more than any other girl,
and Mary sang 3 songs, which was fewer than any other girl. If the total number of songs
sung by Alina and Tina together was 16, then the total number of songs sung by these quartets is 8. -/
theorem total_quartet_songs
  (hanna_songs : ℕ) (mary_songs : ℕ) (alina_tina_songs : ℕ) (total_songs : ℕ)
  (h_hanna : hanna_songs = 9)
  (h_mary : mary_songs = 3)
  (h_alina_tina : alina_tina_songs = 16) :
  total_songs = 8 :=
sorry

end NUMINAMATH_GPT_total_quartet_songs_l1747_174778


namespace NUMINAMATH_GPT_units_digit_2_104_5_205_11_302_l1747_174788

theorem units_digit_2_104_5_205_11_302 : 
  ((2 ^ 104) * (5 ^ 205) * (11 ^ 302)) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_2_104_5_205_11_302_l1747_174788


namespace NUMINAMATH_GPT_syllogism_major_minor_premise_l1747_174753

theorem syllogism_major_minor_premise
(people_of_Yaan_strong_unyielding : Prop)
(people_of_Yaan_Chinese : Prop)
(all_Chinese_strong_unyielding : Prop) :
  all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese → (all_Chinese_strong_unyielding = all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese = people_of_Yaan_Chinese) :=
by
  intros h
  exact ⟨rfl, rfl⟩

end NUMINAMATH_GPT_syllogism_major_minor_premise_l1747_174753


namespace NUMINAMATH_GPT_factorize_expression_l1747_174715

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1747_174715


namespace NUMINAMATH_GPT_sqrt_ceil_eq_one_range_of_x_l1747_174762

/-- Given $[m]$ represents the largest integer not greater than $m$, prove $[\sqrt{2}] = 1$. -/
theorem sqrt_ceil_eq_one (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) :
  floor (Real.sqrt 2) = 1 :=
sorry

/-- Given $[m]$ represents the largest integer not greater than $m$ and $[3 + \sqrt{x}] = 6$, 
  prove $9 \leq x < 16$. -/
theorem range_of_x (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) 
  (x : ℝ) (h : floor (3 + Real.sqrt x) = 6) :
  9 ≤ x ∧ x < 16 :=
sorry

end NUMINAMATH_GPT_sqrt_ceil_eq_one_range_of_x_l1747_174762


namespace NUMINAMATH_GPT_continuous_stripe_probability_l1747_174752

def cube_stripe_probability : ℚ :=
  let stripe_combinations_per_face := 8
  let total_combinations := stripe_combinations_per_face ^ 6
  let valid_combinations := 4 * 3 * 8 * 64
  let probability := valid_combinations / total_combinations
  probability

theorem continuous_stripe_probability :
  cube_stripe_probability = 3 / 128 := by
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l1747_174752


namespace NUMINAMATH_GPT_vasya_tolya_badges_l1747_174759

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end NUMINAMATH_GPT_vasya_tolya_badges_l1747_174759


namespace NUMINAMATH_GPT_amit_work_days_l1747_174782

variable (x : ℕ)

theorem amit_work_days
  (ananthu_rate : ℚ := 1/30) -- Ananthu's work rate is 1/30
  (amit_days : ℕ := 3) -- Amit worked for 3 days
  (ananthu_days : ℕ := 24) -- Ananthu worked for remaining 24 days
  (total_days : ℕ := 27) -- Total work completed in 27 days
  (amit_work: ℚ := amit_days * 1/x) -- Amit's work rate
  (ananthu_work: ℚ := ananthu_days * ananthu_rate) -- Ananthu's work rate
  (total_work : ℚ := 1) -- Total work completed  
  : 3 * (1/x) + 24 * (1/30) = 1 ↔ x = 15 := 
by
  sorry

end NUMINAMATH_GPT_amit_work_days_l1747_174782


namespace NUMINAMATH_GPT_fuel_tank_capacity_l1747_174740

theorem fuel_tank_capacity (C : ℝ) 
  (h1 : 0.12 * 98 + 0.16 * (C - 98) = 30) : 
  C = 212 :=
by
  sorry

end NUMINAMATH_GPT_fuel_tank_capacity_l1747_174740


namespace NUMINAMATH_GPT_sum_of_roots_ln_abs_eq_l1747_174787

theorem sum_of_roots_ln_abs_eq (m : ℝ) (x1 x2 : ℝ) (hx1 : Real.log (|x1|) = m) (hx2 : Real.log (|x2|) = m) : x1 + x2 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_ln_abs_eq_l1747_174787


namespace NUMINAMATH_GPT_divide_numbers_into_consecutive_products_l1747_174747

theorem divide_numbers_into_consecutive_products :
  ∃ (A B : Finset ℕ), A ∪ B = {2, 3, 5, 7, 11, 13, 17} ∧ A ∩ B = ∅ ∧ 
  (A.prod id = 714 ∧ B.prod id = 715 ∨ A.prod id = 715 ∧ B.prod id = 714) :=
sorry

end NUMINAMATH_GPT_divide_numbers_into_consecutive_products_l1747_174747


namespace NUMINAMATH_GPT_find_b_l1747_174719

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem find_b (a b : ℝ) (h1 : f a = -1 / 3) (h2 : f (a * b) = 1 / 6) : b = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l1747_174719


namespace NUMINAMATH_GPT_shaded_area_in_rectangle_is_correct_l1747_174720

noncomputable def percentage_shaded_area : ℝ :=
  let side_length_congruent_squares := 10
  let side_length_small_square := 5
  let rect_length := 20
  let rect_width := 15
  let rect_area := rect_length * rect_width
  let overlap_congruent_squares := side_length_congruent_squares * rect_width
  let overlap_small_square := (side_length_small_square / 2) * side_length_small_square
  let total_shaded_area := overlap_congruent_squares + overlap_small_square
  (total_shaded_area / rect_area) * 100

theorem shaded_area_in_rectangle_is_correct :
  percentage_shaded_area = 54.17 :=
sorry

end NUMINAMATH_GPT_shaded_area_in_rectangle_is_correct_l1747_174720


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1747_174754

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1747_174754


namespace NUMINAMATH_GPT_email_sending_ways_l1747_174783

theorem email_sending_ways (n k : ℕ) (hn : n = 3) (hk : k = 5) : n^k = 243 := 
by
  sorry

end NUMINAMATH_GPT_email_sending_ways_l1747_174783


namespace NUMINAMATH_GPT_yura_catches_up_l1747_174786

theorem yura_catches_up (a : ℕ) (x : ℕ) (h1 : 2 * a * x = a * (x + 5)) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_yura_catches_up_l1747_174786


namespace NUMINAMATH_GPT_phase_shift_cos_l1747_174713

theorem phase_shift_cos (b c : ℝ) (h_b : b = 2) (h_c : c = π / 2) :
  (-c / b) = -π / 4 :=
by
  rw [h_b, h_c]
  sorry

end NUMINAMATH_GPT_phase_shift_cos_l1747_174713


namespace NUMINAMATH_GPT_incorrect_statement_l1747_174771

def vector_mult (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem incorrect_statement (a b : ℝ × ℝ) : vector_mult a b ≠ vector_mult b a :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l1747_174771


namespace NUMINAMATH_GPT_compute_value_l1747_174784

theorem compute_value (a b c : ℕ) (h : a = 262 ∧ b = 258 ∧ c = 150) : 
  (a^2 - b^2) + c = 2230 := 
by
  sorry

end NUMINAMATH_GPT_compute_value_l1747_174784


namespace NUMINAMATH_GPT_count_integer_solutions_less_than_zero_l1747_174776

theorem count_integer_solutions_less_than_zero : 
  ∃ k : ℕ, k = 4 ∧ (∀ n : ℤ, n^4 - n^3 - 3 * n^2 - 3 * n - 17 < 0 → k = 4) :=
by
  sorry

end NUMINAMATH_GPT_count_integer_solutions_less_than_zero_l1747_174776


namespace NUMINAMATH_GPT_evaluate_sqrt_sum_l1747_174737

theorem evaluate_sqrt_sum : (Real.sqrt 1 + Real.sqrt 9) = 4 := by
  sorry

end NUMINAMATH_GPT_evaluate_sqrt_sum_l1747_174737


namespace NUMINAMATH_GPT_determinant_zero_implies_sum_l1747_174781

open Matrix

noncomputable def matrix_example (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![2, 5, 8],
    ![4, a, b],
    ![4, b, a]
  ]

theorem determinant_zero_implies_sum (a b : ℝ) (h : a ≠ b) (h_det : det (matrix_example a b) = 0) : a + b = 26 :=
by
  sorry

end NUMINAMATH_GPT_determinant_zero_implies_sum_l1747_174781


namespace NUMINAMATH_GPT_unique_flavors_l1747_174703

theorem unique_flavors (x y : ℕ) (h₀ : x = 5) (h₁ : y = 4) : 
  (∃ f : ℕ, f = 17) :=
sorry

end NUMINAMATH_GPT_unique_flavors_l1747_174703


namespace NUMINAMATH_GPT_bob_final_amount_l1747_174727

noncomputable def final_amount (start: ℝ) : ℝ :=
  let day1 := start - (3/5) * start
  let day2 := day1 - (7/12) * day1
  let day3 := day2 - (2/3) * day2
  let day4 := day3 - (1/6) * day3
  let day5 := day4 - (5/8) * day4
  let day6 := day5 - (3/5) * day5
  day6

theorem bob_final_amount : final_amount 500 = 3.47 := by
  sorry

end NUMINAMATH_GPT_bob_final_amount_l1747_174727


namespace NUMINAMATH_GPT_solve_absolute_value_equation_l1747_174757

theorem solve_absolute_value_equation (y : ℝ) :
  (|y - 8| + 3 * y = 11) → (y = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_equation_l1747_174757


namespace NUMINAMATH_GPT_combined_bus_capacity_l1747_174748

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end NUMINAMATH_GPT_combined_bus_capacity_l1747_174748


namespace NUMINAMATH_GPT_find_number_l1747_174746

theorem find_number (n : ℝ) (x : ℕ) (h1 : x = 4) (h2 : n^(2*x) = 3^(12-x)) : n = 3 := by
  sorry

end NUMINAMATH_GPT_find_number_l1747_174746


namespace NUMINAMATH_GPT_quadratic_real_roots_l1747_174745

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1747_174745


namespace NUMINAMATH_GPT_page_added_twice_l1747_174758

theorem page_added_twice (n k : ℕ) (h1 : (n * (n + 1)) / 2 + k = 1986) : k = 33 :=
sorry

end NUMINAMATH_GPT_page_added_twice_l1747_174758


namespace NUMINAMATH_GPT_temperature_on_tuesday_l1747_174779

theorem temperature_on_tuesday 
  (T W Th F : ℝ)
  (H1 : (T + W + Th) / 3 = 45)
  (H2 : (W + Th + F) / 3 = 50)
  (H3 : F = 53) :
  T = 38 :=
by 
  sorry

end NUMINAMATH_GPT_temperature_on_tuesday_l1747_174779


namespace NUMINAMATH_GPT_scientific_notation_80000000_l1747_174797

-- Define the given number
def number : ℕ := 80000000

-- Define the scientific notation form
def scientific_notation (n k : ℕ) (a : ℝ) : Prop :=
  n = (a * (10 : ℝ) ^ k)

-- The theorem to prove scientific notation of 80,000,000
theorem scientific_notation_80000000 : scientific_notation number 7 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_80000000_l1747_174797


namespace NUMINAMATH_GPT_remainder_is_zero_l1747_174733

def remainder_when_multiplied_then_subtracted (a b : ℕ) : ℕ :=
  (a * b - 8) % 8

theorem remainder_is_zero : remainder_when_multiplied_then_subtracted 104 106 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_is_zero_l1747_174733


namespace NUMINAMATH_GPT_units_digit_b_l1747_174791

theorem units_digit_b (a b : ℕ) (h1 : a % 10 = 9) (h2 : a * b = 34^8) : b % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_b_l1747_174791


namespace NUMINAMATH_GPT_solve_equation_l1747_174708

theorem solve_equation {x : ℝ} (h : x ≠ -2) : (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) → x = 1 :=
by
  intro h_eq
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_solve_equation_l1747_174708


namespace NUMINAMATH_GPT_factors_of_P_factorization_of_P_factorize_expression_l1747_174765

noncomputable def P (a b c : ℝ) : ℝ :=
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b)

theorem factors_of_P (a b c : ℝ) :
  (a - b ∣ P a b c) ∧ (b - c ∣ P a b c) ∧ (c - a ∣ P a b c) :=
sorry

theorem factorization_of_P (a b c : ℝ) :
  P a b c = -(a - b) * (b - c) * (c - a) :=
sorry

theorem factorize_expression (x y z : ℝ) :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3 * (x + y) * (y + z) * (z + x) :=
sorry

end NUMINAMATH_GPT_factors_of_P_factorization_of_P_factorize_expression_l1747_174765


namespace NUMINAMATH_GPT_problem_solve_l1747_174750

theorem problem_solve (n : ℕ) (h_pos : 0 < n) 
    (h_eq : Real.sin (Real.pi / (3 * n)) + Real.cos (Real.pi / (3 * n)) = Real.sqrt (2 * n) / 3) : 
    n = 6 := 
  sorry

end NUMINAMATH_GPT_problem_solve_l1747_174750


namespace NUMINAMATH_GPT_bucket_weight_l1747_174723

variable {p q x y : ℝ}

theorem bucket_weight (h1 : x + (1 / 4) * y = p) (h2 : x + (3 / 4) * y = q) :
  x + y = - (1 / 2) * p + (3 / 2) * q := by
  sorry

end NUMINAMATH_GPT_bucket_weight_l1747_174723


namespace NUMINAMATH_GPT_letters_by_30_typists_in_1_hour_l1747_174767

-- Definitions from the conditions
def lettersTypedByOneTypistIn20Minutes := 44 / 20

def lettersTypedBy30TypistsIn20Minutes := 30 * (lettersTypedByOneTypistIn20Minutes)

def conversionToHours := 3

-- Theorem statement
theorem letters_by_30_typists_in_1_hour : lettersTypedBy30TypistsIn20Minutes * conversionToHours = 198 := by
  sorry

end NUMINAMATH_GPT_letters_by_30_typists_in_1_hour_l1747_174767


namespace NUMINAMATH_GPT_spent_on_burgers_l1747_174722

noncomputable def money_spent_on_burgers (total_allowance : ℝ) (movie_fraction music_fraction ice_cream_fraction : ℝ) : ℝ :=
  let movie_expense := (movie_fraction * total_allowance)
  let music_expense := (music_fraction * total_allowance)
  let ice_cream_expense := (ice_cream_fraction * total_allowance)
  total_allowance - (movie_expense + music_expense + ice_cream_expense)

theorem spent_on_burgers : 
  money_spent_on_burgers 50 (1/4) (3/10) (2/5) = 2.5 :=
by sorry

end NUMINAMATH_GPT_spent_on_burgers_l1747_174722


namespace NUMINAMATH_GPT_compute_100p_plus_q_l1747_174707

-- Given constants p, q under the provided conditions,
-- prove the result: 100p + q = 430 / 3.
theorem compute_100p_plus_q (p q : ℚ) 
  (h1 : ∀ x : ℚ, (x + p) * (x + q) * (x + 20) = 0 → x ≠ -4)
  (h2 : ∀ x : ℚ, (x + 3 * p) * (x + 4) * (x + 10) = 0 → (x = -4 ∨ x ≠ -4)) :
  100 * p + q = 430 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_compute_100p_plus_q_l1747_174707


namespace NUMINAMATH_GPT_isosceles_trapezoid_ratio_l1747_174773

theorem isosceles_trapezoid_ratio (a b d : ℝ) (h1 : b = 2 * d) (h2 : a = d) : a / b = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_ratio_l1747_174773


namespace NUMINAMATH_GPT_cost_of_ice_cream_l1747_174716

theorem cost_of_ice_cream 
  (meal_cost : ℕ)
  (number_of_people : ℕ)
  (total_money : ℕ)
  (total_cost : ℕ := meal_cost * number_of_people) 
  (remaining_money : ℕ := total_money - total_cost) 
  (ice_cream_cost_per_scoop : ℕ := remaining_money / number_of_people) :
  meal_cost = 10 ∧ number_of_people = 3 ∧ total_money = 45 →
  ice_cream_cost_per_scoop = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_ice_cream_l1747_174716


namespace NUMINAMATH_GPT_smallest_n_l1747_174772

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3^n = k^4) (h2 : ∃ l : ℕ, 2^n = l^6) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1747_174772


namespace NUMINAMATH_GPT_not_in_M_4n2_l1747_174741

def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

theorem not_in_M_4n2 (n : ℤ) : ¬ (4 * n + 2 ∈ M) :=
by
sorry

end NUMINAMATH_GPT_not_in_M_4n2_l1747_174741


namespace NUMINAMATH_GPT_fair_tickets_sold_l1747_174761

theorem fair_tickets_sold (F : ℕ) (number_of_baseball_game_tickets : ℕ) 
  (h1 : F = 2 * number_of_baseball_game_tickets + 6) (h2 : number_of_baseball_game_tickets = 56) :
  F = 118 :=
by
  sorry

end NUMINAMATH_GPT_fair_tickets_sold_l1747_174761


namespace NUMINAMATH_GPT_problem_rewrite_equation_l1747_174751

theorem problem_rewrite_equation :
  ∃ a b c : ℤ, a > 0 ∧ (64*(x^2) + 96*x - 81 = 0) → ((a*x + b)^2 = c) ∧ (a + b + c = 131) :=
sorry

end NUMINAMATH_GPT_problem_rewrite_equation_l1747_174751


namespace NUMINAMATH_GPT_circle_centers_connection_line_eq_l1747_174710

-- Define the first circle equation
def circle1 (x y : ℝ) := (x^2 + y^2 - 4*x + 6*y = 0)

-- Define the second circle equation
def circle2 (x y : ℝ) := (x^2 + y^2 - 6*x = 0)

-- Given the centers of the circles, prove the equation of the line connecting them
theorem circle_centers_connection_line_eq (x y : ℝ) :
  (∀ (x y : ℝ), circle1 x y → (x = 2 ∧ y = -3)) →
  (∀ (x y : ℝ), circle2 x y → (x = 3 ∧ y = 0)) →
  (3 * x - y - 9 = 0) :=
by
  -- Here we would sketch the proof but skip it with sorry
  sorry

end NUMINAMATH_GPT_circle_centers_connection_line_eq_l1747_174710


namespace NUMINAMATH_GPT_problem_statement_l1747_174749

theorem problem_statement (x y : ℝ) (h : |x + 1| + |y + 2 * x| = 0) : (x + y) ^ 2004 = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1747_174749


namespace NUMINAMATH_GPT_least_candies_to_remove_for_equal_distribution_l1747_174768

theorem least_candies_to_remove_for_equal_distribution :
  ∃ k : ℕ, k = 4 ∧ ∀ n : ℕ, 24 - k = 5 * n :=
sorry

end NUMINAMATH_GPT_least_candies_to_remove_for_equal_distribution_l1747_174768


namespace NUMINAMATH_GPT_distribution_scheme_count_l1747_174796

-- Define the people and communities
inductive Person
| A | B | C
deriving DecidableEq, Repr

inductive Community
| C1 | C2 | C3 | C4 | C5 | C6 | C7
deriving DecidableEq, Repr

-- Define a function to count the number of valid distribution schemes
def countDistributionSchemes : Nat :=
  -- This counting is based on recognizing the problem involves permutations and combinations,
  -- the specific detail logic is omitted since we are only writing the statement, no proof.
  336

-- The main theorem statement
theorem distribution_scheme_count :
  countDistributionSchemes = 336 :=
sorry

end NUMINAMATH_GPT_distribution_scheme_count_l1747_174796


namespace NUMINAMATH_GPT_probability_both_hit_l1747_174702

-- Conditions
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Question and proof problem
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.72 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_hit_l1747_174702


namespace NUMINAMATH_GPT_fraction_of_liars_l1747_174760

theorem fraction_of_liars (n : ℕ) (villagers : Fin n → Prop) (right_neighbor : ∀ i, villagers i ↔ ∀ j : Fin n, j = (i + 1) % n → villagers j) :
  ∃ (x : ℚ), x = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_of_liars_l1747_174760


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_primes_l1747_174766

theorem smallest_four_digit_divisible_by_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ n = 1050 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_primes_l1747_174766


namespace NUMINAMATH_GPT_term_100_is_981_l1747_174780

def sequence_term (n : ℕ) : ℕ :=
  if n = 100 then 981 else sorry

theorem term_100_is_981 : sequence_term 100 = 981 := by
  rfl

end NUMINAMATH_GPT_term_100_is_981_l1747_174780


namespace NUMINAMATH_GPT_john_tanks_needed_l1747_174732

theorem john_tanks_needed 
  (num_balloons : ℕ) 
  (volume_per_balloon : ℕ) 
  (volume_per_tank : ℕ) 
  (H1 : num_balloons = 1000) 
  (H2 : volume_per_balloon = 10) 
  (H3 : volume_per_tank = 500) 
: (num_balloons * volume_per_balloon) / volume_per_tank = 20 := 
by 
  sorry

end NUMINAMATH_GPT_john_tanks_needed_l1747_174732


namespace NUMINAMATH_GPT_michael_needs_more_money_l1747_174730

-- Define the initial conditions
def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_gbp : ℝ := 30
def gbp_to_usd : ℝ := 1.4
def perfume_cost : ℝ := perfume_gbp * gbp_to_usd
def photo_album_eur : ℝ := 25
def eur_to_usd : ℝ := 1.2
def photo_album_cost : ℝ := photo_album_eur * eur_to_usd

-- Sum the costs
def total_cost : ℝ := cake_cost + bouquet_cost + balloons_cost + perfume_cost + photo_album_cost

-- Define the required amount
def additional_money_needed : ℝ := total_cost - michael_money

-- The theorem statement
theorem michael_needs_more_money : additional_money_needed = 83 := by
  sorry

end NUMINAMATH_GPT_michael_needs_more_money_l1747_174730


namespace NUMINAMATH_GPT_part1_part2_l1747_174789

def quadratic_inequality_A (x m : ℝ) := -x^2 + 2 * m * x + 4 - m^2 ≥ 0
def quadratic_inequality_B (x : ℝ) := 2 * x^2 - 5 * x - 7 < 0

theorem part1 (m : ℝ) :
  (∀ x, quadratic_inequality_A x m ∧ quadratic_inequality_B x ↔ 0 ≤ x ∧ x < 7 / 2) →
  m = 2 := by sorry

theorem part2 (m : ℝ) :
  (∀ x, quadratic_inequality_B x → ¬ quadratic_inequality_A x m) →
  m ≤ -3 ∨ 11 / 2 ≤ m := by sorry

end NUMINAMATH_GPT_part1_part2_l1747_174789


namespace NUMINAMATH_GPT_symmetry_construction_complete_l1747_174709

-- Conditions: The word and the chosen axis of symmetry
def word : String := "ГЕОМЕТРИя"

inductive Axis
| horizontal
| vertical

-- The main theorem which states that a symmetrical figure can be constructed for the given word and axis
theorem symmetry_construction_complete (axis : Axis) : ∃ (symmetrical : String), 
  (axis = Axis.horizontal ∨ axis = Axis.vertical) → 
   symmetrical = "яИРТЕМОЕГ" := 
by
  sorry

end NUMINAMATH_GPT_symmetry_construction_complete_l1747_174709


namespace NUMINAMATH_GPT_problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l1747_174714

theorem problem_a_eq_2 (x : ℝ) : (12 * x^2 - 2 * x > 4) ↔ (x < -1 / 2 ∨ x > 2 / 3) := sorry

theorem problem_a_real_pos (a x : ℝ) (h : a > 0) : (12 * x^2 - a * x > a^2) ↔ (x < -a / 4 ∨ x > a / 3) := sorry

theorem problem_a_real_zero (x : ℝ) : (12 * x^2 > 0) ↔ (x ≠ 0) := sorry

theorem problem_a_real_neg (a x : ℝ) (h : a < 0) : (12 * x^2 - a * x > a^2) ↔ (x < a / 3 ∨ x > -a / 4) := sorry

end NUMINAMATH_GPT_problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l1747_174714


namespace NUMINAMATH_GPT_bill_take_home_salary_l1747_174755

-- Define the parameters
def property_taxes : ℝ := 2000
def sales_taxes : ℝ := 3000
def gross_salary : ℝ := 50000
def income_tax_rate : ℝ := 0.10

-- Define income tax calculation
def income_tax : ℝ := income_tax_rate * gross_salary

-- Define total taxes calculation
def total_taxes : ℝ := property_taxes + sales_taxes + income_tax

-- Define the take-home salary calculation
def take_home_salary : ℝ := gross_salary - total_taxes

-- Statement of the theorem
theorem bill_take_home_salary : take_home_salary = 40000 := by
  -- Sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_bill_take_home_salary_l1747_174755


namespace NUMINAMATH_GPT_smallest_angle_of_isosceles_trapezoid_l1747_174738

def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  a = c ∧ b = d ∧ a + b + c + d = 360 ∧ a + 3 * b = 150

theorem smallest_angle_of_isosceles_trapezoid (a b : ℝ) (h1 : is_isosceles_trapezoid a b a (a + 2 * b))
  : a = 47 :=
sorry

end NUMINAMATH_GPT_smallest_angle_of_isosceles_trapezoid_l1747_174738


namespace NUMINAMATH_GPT_compare_abc_l1747_174739

noncomputable def a : ℝ := 2 + (1 / 5) * Real.log 2
noncomputable def b : ℝ := 1 + Real.exp (0.2 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.1 * Real.log 2)

theorem compare_abc : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_compare_abc_l1747_174739


namespace NUMINAMATH_GPT_limit_of_R_l1747_174744

noncomputable def R (m b : ℝ) : ℝ :=
  let x := ((-b) + Real.sqrt (b^2 + 4 * m)) / 2
  m * x + 3 

theorem limit_of_R (b : ℝ) (hb : b ≠ 0) : 
  (∀ m : ℝ, m < 3) → 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 0) < δ → abs ((R x (-b) - R x b) / x - b) < ε) :=
by
  sorry

end NUMINAMATH_GPT_limit_of_R_l1747_174744


namespace NUMINAMATH_GPT_detergent_per_pound_l1747_174774

theorem detergent_per_pound (detergent clothes_per_det: ℝ) (h: detergent = 18 ∧ clothes_per_det = 9) :
  detergent / clothes_per_det = 2 :=
by
  sorry

end NUMINAMATH_GPT_detergent_per_pound_l1747_174774


namespace NUMINAMATH_GPT_sum_of_five_consecutive_integers_l1747_174726

theorem sum_of_five_consecutive_integers : ∀ (n : ℤ), (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := 
by
  -- This would be where the proof goes
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_integers_l1747_174726


namespace NUMINAMATH_GPT_xunzi_statement_l1747_174764

/-- 
Given the conditions:
  "If not accumulating small steps, then not reaching a thousand miles."
  Which can be represented as: ¬P → ¬q.
Prove that accumulating small steps (P) is a necessary but not sufficient condition for
reaching a thousand miles (q).
-/
theorem xunzi_statement (P q : Prop) (h : ¬P → ¬q) : (q → P) ∧ ¬(P → q) :=
by sorry

end NUMINAMATH_GPT_xunzi_statement_l1747_174764


namespace NUMINAMATH_GPT_coleen_sprinkles_l1747_174769

theorem coleen_sprinkles : 
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  remaining_sprinkles = 3 :=
by
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  sorry

end NUMINAMATH_GPT_coleen_sprinkles_l1747_174769


namespace NUMINAMATH_GPT_solve_abs_inequality_l1747_174705

theorem solve_abs_inequality (x : ℝ) : abs ((7 - 2 * x) / 4) < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l1747_174705


namespace NUMINAMATH_GPT_percentage_of_women_in_study_group_l1747_174721

theorem percentage_of_women_in_study_group
  (W : ℝ)
  (H1 : 0 ≤ W ∧ W ≤ 1)
  (H2 : 0.60 * W = 0.54) :
  W = 0.9 :=
sorry

end NUMINAMATH_GPT_percentage_of_women_in_study_group_l1747_174721
