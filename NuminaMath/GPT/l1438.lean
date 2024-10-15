import Mathlib

namespace NUMINAMATH_GPT_min_value_fraction_solve_inequality_l1438_143834

-- Part 1
theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (f : ℝ → ℝ)
  (h3 : f 1 = 2) (h4 : ∀ x, f x = a * x^2 + b * x + 1) :
  (a + b = 1) → (∃ z, z = (1 / a + 4 / b) ∧ z = 9) := 
by {
  sorry
}

-- Part 2
theorem solve_inequality (a : ℝ) (x : ℝ) (h1 : b = -a - 1) (f : ℝ → ℝ)
  (h2 : ∀ x, f x = a * x^2 + b * x + 1) :
  (f x ≤ 0) → 
  (if a = 0 then 
      {x | x ≥ 1}
  else if a > 0 then
      if a = 1 then 
          {x | x = 1}
      else if 0 < a ∧ a < 1 then 
          {x | 1 ≤ x ∧ x ≤ 1 / a}
      else 
          {x | 1 / a ≤ x ∧ x ≤ 1}
  else 
      {x | x ≥ 1 ∨ x ≤ 1 / a}) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_fraction_solve_inequality_l1438_143834


namespace NUMINAMATH_GPT_chess_or_basketball_students_l1438_143836

-- Definitions based on the conditions
def percentage_likes_basketball : ℝ := 0.4
def percentage_likes_chess : ℝ := 0.1
def total_students : ℕ := 250

-- Main statement to prove
theorem chess_or_basketball_students : 
  (percentage_likes_basketball + percentage_likes_chess) * total_students = 125 :=
by
  sorry

end NUMINAMATH_GPT_chess_or_basketball_students_l1438_143836


namespace NUMINAMATH_GPT_primes_between_2_and_100_l1438_143825

open Nat

theorem primes_between_2_and_100 :
  { p : ℕ | 2 ≤ p ∧ p ≤ 100 ∧ Nat.Prime p } = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} :=
by
  sorry

end NUMINAMATH_GPT_primes_between_2_and_100_l1438_143825


namespace NUMINAMATH_GPT_rice_amount_previously_l1438_143880

variables (P X : ℝ) (hP : P > 0) (h : 0.8 * P * 50 = P * X)

theorem rice_amount_previously (hP : P > 0) (h : 0.8 * P * 50 = P * X) : X = 40 := 
by 
  sorry

end NUMINAMATH_GPT_rice_amount_previously_l1438_143880


namespace NUMINAMATH_GPT_ratio_of_building_heights_l1438_143845

theorem ratio_of_building_heights (F_h F_s A_s B_s : ℝ) (hF_h : F_h = 18) (hF_s : F_s = 45)
  (hA_s : A_s = 60) (hB_s : B_s = 72) :
  let h_A := (F_h / F_s) * A_s
  let h_B := (F_h / F_s) * B_s
  (h_A / h_B) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_building_heights_l1438_143845


namespace NUMINAMATH_GPT_sum_is_402_3_l1438_143887

def sum_of_numbers := 3 + 33 + 333 + 33.3

theorem sum_is_402_3 : sum_of_numbers = 402.3 := by
  sorry

end NUMINAMATH_GPT_sum_is_402_3_l1438_143887


namespace NUMINAMATH_GPT_percentage_correct_l1438_143867

theorem percentage_correct (x : ℕ) (h : x > 0) : 
  (4 * x / (6 * x) * 100 = 200 / 3) :=
by
  sorry

end NUMINAMATH_GPT_percentage_correct_l1438_143867


namespace NUMINAMATH_GPT_symmetric_about_origin_implies_odd_l1438_143866

variable {F : Type} [Field F] (f : F → F)
variable (x : F)

theorem symmetric_about_origin_implies_odd (H : ∀ x, f (-x) = -f x) : f x + f (-x) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_symmetric_about_origin_implies_odd_l1438_143866


namespace NUMINAMATH_GPT_not_always_product_greater_l1438_143878

-- Define the premise and the conclusion
theorem not_always_product_greater (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b < 1) : a * b < a :=
sorry

end NUMINAMATH_GPT_not_always_product_greater_l1438_143878


namespace NUMINAMATH_GPT_f_correct_l1438_143833

noncomputable def f : ℕ → ℝ
| 0       => 0 -- undefined for 0, start from 1
| (n + 1) => if n = 0 then 1/2 else sorry -- recursion undefined for now

theorem f_correct : ∀ n ≥ 1, f n = (3^(n-1) / (3^(n-1) + 1)) :=
by
  -- Initial conditions
  have h0 : f 1 = 1/2 := sorry
  -- Recurrence relations
  have h1 : ∀ n, n ≥ 1 → f (n + 1) ≥ (3 * f n) / (2 * f n + 1) := sorry
  -- Prove the function form
  sorry

end NUMINAMATH_GPT_f_correct_l1438_143833


namespace NUMINAMATH_GPT_complement_of_union_l1438_143886

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 2) * (x + 1) ≤ 0 }
def B : Set ℝ := { x | 0 ≤ x ∧ x < 3 }

theorem complement_of_union :
  Set.compl (A ∪ B) = { x : ℝ | x < -1 } ∪ { x | x ≥ 3 } := by
  sorry

end NUMINAMATH_GPT_complement_of_union_l1438_143886


namespace NUMINAMATH_GPT_cos_54_deg_l1438_143883

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end NUMINAMATH_GPT_cos_54_deg_l1438_143883


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1438_143897

theorem distance_between_A_and_B :
  let A := (0, 0)
  let B := (-10, 24)
  dist A B = 26 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1438_143897


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l1438_143811

theorem isosceles_triangle_largest_angle (A B C : Type) (α β γ : ℝ)
  (h_iso : α = β) (h_angles : α = 50) (triangle: α + β + γ = 180) : γ = 80 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l1438_143811


namespace NUMINAMATH_GPT_find_angle_A_l1438_143829

variable (a b c : ℝ)
variable (A : ℝ)

axiom triangle_ABC : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2

theorem find_angle_A : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2 → A = Real.pi / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_angle_A_l1438_143829


namespace NUMINAMATH_GPT_find_angle_E_l1438_143805

def trapezoid_angles (E H F G : ℝ) : Prop :=
  E + H = 180 ∧ E = 3 * H ∧ G = 4 * F

theorem find_angle_E (E H F G : ℝ) 
  (h1 : E + H = 180)
  (h2 : E = 3 * H)
  (h3 : G = 4 * F) : 
  E = 135 := by
    sorry

end NUMINAMATH_GPT_find_angle_E_l1438_143805


namespace NUMINAMATH_GPT_sum_of_areas_l1438_143846

def base_width : ℕ := 3
def lengths : List ℕ := [1, 8, 27, 64, 125, 216]
def area (w l : ℕ) : ℕ := w * l
def total_area : ℕ := (lengths.map (area base_width)).sum

theorem sum_of_areas : total_area = 1323 := 
by sorry

end NUMINAMATH_GPT_sum_of_areas_l1438_143846


namespace NUMINAMATH_GPT_train_length_l1438_143882

noncomputable def speed_kph := 56  -- speed in km/hr
def time_crossing := 9  -- time in seconds
noncomputable def speed_mps := speed_kph * 1000 / 3600  -- converting km/hr to m/s

theorem train_length : speed_mps * time_crossing = 140 := by
  -- conversion and result approximation
  sorry

end NUMINAMATH_GPT_train_length_l1438_143882


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1438_143885

variable (x y : ℝ)

theorem sufficient_not_necessary_condition (h : x + y ≤ 1) : x ≤ 1/2 ∨ y ≤ 1/2 := 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1438_143885


namespace NUMINAMATH_GPT_kendra_packs_l1438_143835

/-- Kendra has some packs of pens. Tony has 2 packs of pens. There are 3 pens in each pack. 
Kendra and Tony decide to keep two pens each and give the remaining pens to their friends 
one pen per friend. They give pens to 14 friends. Prove that Kendra has 4 packs of pens. --/
theorem kendra_packs : ∀ (kendra_pens tony_pens pens_per_pack pens_kept pens_given friends : ℕ),
  tony_pens = 2 →
  pens_per_pack = 3 →
  pens_kept = 2 →
  pens_given = 14 →
  tony_pens * pens_per_pack - pens_kept + kendra_pens - pens_kept = pens_given →
  kendra_pens / pens_per_pack = 4 :=
by
  intros kendra_pens tony_pens pens_per_pack pens_kept pens_given friends
  intro h1
  intro h2
  intro h3
  intro h4
  intro h5
  sorry

end NUMINAMATH_GPT_kendra_packs_l1438_143835


namespace NUMINAMATH_GPT_books_initially_l1438_143862

theorem books_initially (A B : ℕ) (h1 : A = 3) (h2 : B = (A + 2) + 2) : B = 7 :=
by
  -- Using the given facts, we need to show B = 7
  sorry

end NUMINAMATH_GPT_books_initially_l1438_143862


namespace NUMINAMATH_GPT_find_a3_minus_b3_l1438_143839

theorem find_a3_minus_b3 (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 47) : a^3 - b^3 = 322 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_minus_b3_l1438_143839


namespace NUMINAMATH_GPT_find_a_45_l1438_143813

theorem find_a_45 (a : ℕ → ℝ) 
  (h0 : a 0 = 11) 
  (h1 : a 1 = 11) 
  (h_rec : ∀ m n : ℕ, a (m + n) = (1 / 2) * (a (2 * m) + a (2 * n)) - (m - n) ^ 2) 
  : a 45 = 1991 :=
sorry

end NUMINAMATH_GPT_find_a_45_l1438_143813


namespace NUMINAMATH_GPT_Marty_combination_count_l1438_143857

theorem Marty_combination_count :
  let num_colors := 4
  let num_methods := 3
  num_colors * num_methods = 12 :=
by
  let num_colors := 4
  let num_methods := 3
  sorry

end NUMINAMATH_GPT_Marty_combination_count_l1438_143857


namespace NUMINAMATH_GPT_scientific_notation_120_million_l1438_143863

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_120_million_l1438_143863


namespace NUMINAMATH_GPT_partial_fraction_sum_zero_l1438_143817

variable {A B C D E : ℝ}
variable {x : ℝ}

theorem partial_fraction_sum_zero (h : 
  (1:ℝ) / ((x-1)*x*(x+1)*(x+2)*(x+3)) = 
  A / (x-1) + B / x + C / (x+1) + D / (x+2) + E / (x+3)) : 
  A + B + C + D + E = 0 :=
by sorry

end NUMINAMATH_GPT_partial_fraction_sum_zero_l1438_143817


namespace NUMINAMATH_GPT_find_t_l1438_143869

-- Define the logarithm base 3 function
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Given Condition
def condition (t : ℝ) : Prop := 4 * log_base_3 t = log_base_3 (4 * t) + 2

-- Theorem stating if the given condition holds, then t must be 6
theorem find_t (t : ℝ) (ht : condition t) : t = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_t_l1438_143869


namespace NUMINAMATH_GPT_abs_eq_of_unique_solution_l1438_143879

theorem abs_eq_of_unique_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) :
    |a| = |b| :=
sorry

end NUMINAMATH_GPT_abs_eq_of_unique_solution_l1438_143879


namespace NUMINAMATH_GPT_equivalent_statements_l1438_143806

variables (P Q : Prop)

theorem equivalent_statements : (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_equivalent_statements_l1438_143806


namespace NUMINAMATH_GPT_cubic_expression_identity_l1438_143809

theorem cubic_expression_identity (x : ℝ) (hx : x + 1/x = 8) : 
  x^3 + 1/x^3 = 332 :=
sorry

end NUMINAMATH_GPT_cubic_expression_identity_l1438_143809


namespace NUMINAMATH_GPT_vertical_asymptotes_sum_l1438_143865

theorem vertical_asymptotes_sum : 
  let f (x : ℝ) := (6 * x^2 + 1) / (4 * x^2 + 6 * x + 3)
  let den := 4 * x^2 + 6 * x + 3
  let p := -(3 / 2)
  let q := -(1 / 2)
  (den = 0) → (p + q = -2) :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptotes_sum_l1438_143865


namespace NUMINAMATH_GPT_sets_equivalence_l1438_143822

theorem sets_equivalence :
  (∀ M N, (M = {(3, 2)} ∧ N = {(2, 3)} → M ≠ N) ∧
          (M = {4, 5} ∧ N = {5, 4} → M = N) ∧
          (M = {1, 2} ∧ N = {(1, 2)} → M ≠ N) ∧
          (M = {(x, y) | x + y = 1} ∧ N = {y | ∃ x, x + y = 1} → M ≠ N)) :=
by sorry

end NUMINAMATH_GPT_sets_equivalence_l1438_143822


namespace NUMINAMATH_GPT_dimes_max_diff_l1438_143816

-- Definitions and conditions
def num_coins (a b c : ℕ) : Prop := a + b + c = 120
def coin_values (a b c : ℕ) : Prop := 5 * a + 10 * b + 50 * c = 1050
def dimes_difference (a1 a2 b1 b2 c1 c2 : ℕ) : Prop := num_coins a1 b1 c1 ∧ num_coins a2 b2 c2 ∧ coin_values a1 b1 c1 ∧ coin_values a2 b2 c2 ∧ a1 = a2 ∧ c1 = c2

-- Theorem statement
theorem dimes_max_diff : ∃ (a b1 b2 c : ℕ), dimes_difference a a b1 b2 c c ∧ b1 - b2 = 90 :=
by sorry

end NUMINAMATH_GPT_dimes_max_diff_l1438_143816


namespace NUMINAMATH_GPT_solution_set_l1438_143891

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom monotone_increasing : ∀ x y, x < y → f x ≤ f y
axiom f_at_3 : f 3 = 2

-- Proof statement
theorem solution_set : {x : ℝ | -2 ≤ f (3 - x) ∧ f (3 - x) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 6} :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_l1438_143891


namespace NUMINAMATH_GPT_find_nat_numbers_for_divisibility_l1438_143840

theorem find_nat_numbers_for_divisibility :
  ∃ (a b : ℕ), (7^3 ∣ a^2 + a * b + b^2) ∧ (¬ 7 ∣ a) ∧ (¬ 7 ∣ b) ∧ (a = 1) ∧ (b = 18) := by
  sorry

end NUMINAMATH_GPT_find_nat_numbers_for_divisibility_l1438_143840


namespace NUMINAMATH_GPT_find_coordinates_of_Q_l1438_143870

theorem find_coordinates_of_Q (x y : ℝ) (P : ℝ × ℝ) (hP : P = (1, 2))
    (perp : x + 2 * y = 0) (length : x^2 + y^2 = 5) :
    (x, y) = (-2, 1) :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_find_coordinates_of_Q_l1438_143870


namespace NUMINAMATH_GPT_greatest_x_for_quadratic_inequality_l1438_143888

theorem greatest_x_for_quadratic_inequality (x : ℝ) (h : x^2 - 12 * x + 35 ≤ 0) : x ≤ 7 :=
sorry

end NUMINAMATH_GPT_greatest_x_for_quadratic_inequality_l1438_143888


namespace NUMINAMATH_GPT_initial_amount_l1438_143830

theorem initial_amount (P : ℝ) :
  (P * 1.0816 - P * 1.08 = 3.0000000000002274) → P = 1875.0000000001421 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l1438_143830


namespace NUMINAMATH_GPT_value_of_square_of_sum_l1438_143894

theorem value_of_square_of_sum (x y: ℝ) 
(h1: 2 * x * (x + y) = 58) 
(h2: 3 * y * (x + y) = 111):
  (x + y)^2 = (169/5)^2 := by
  sorry

end NUMINAMATH_GPT_value_of_square_of_sum_l1438_143894


namespace NUMINAMATH_GPT_sum_fractions_eq_l1438_143861

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end NUMINAMATH_GPT_sum_fractions_eq_l1438_143861


namespace NUMINAMATH_GPT_toilet_paper_squares_per_roll_l1438_143800

theorem toilet_paper_squares_per_roll
  (trips_per_day : ℕ)
  (squares_per_trip : ℕ)
  (num_rolls : ℕ)
  (supply_days : ℕ)
  (total_squares : ℕ)
  (squares_per_roll : ℕ)
  (h1 : trips_per_day = 3)
  (h2 : squares_per_trip = 5)
  (h3 : num_rolls = 1000)
  (h4 : supply_days = 20000)
  (h5 : total_squares = trips_per_day * squares_per_trip * supply_days)
  (h6 : squares_per_roll = total_squares / num_rolls) :
  squares_per_roll = 300 :=
by sorry

end NUMINAMATH_GPT_toilet_paper_squares_per_roll_l1438_143800


namespace NUMINAMATH_GPT_find_number_of_packs_l1438_143828

-- Define the cost of a pack of Digimon cards
def cost_pack_digimon : ℝ := 4.45

-- Define the cost of the deck of baseball cards
def cost_deck_baseball : ℝ := 6.06

-- Define the total amount spent
def total_spent : ℝ := 23.86

-- Define the number of packs of Digimon cards Keith bought
def number_of_packs (D : ℝ) : Prop :=
  cost_pack_digimon * D + cost_deck_baseball = total_spent

-- Prove the number of packs is 4
theorem find_number_of_packs : ∃ D, number_of_packs D ∧ D = 4 :=
by
  -- the proof will be inserted here
  sorry

end NUMINAMATH_GPT_find_number_of_packs_l1438_143828


namespace NUMINAMATH_GPT_inequality_solution_l1438_143854

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4 / 3 ∨ -3 / 2 < x := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1438_143854


namespace NUMINAMATH_GPT_sum_squares_l1438_143841

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end NUMINAMATH_GPT_sum_squares_l1438_143841


namespace NUMINAMATH_GPT_factor_expression_l1438_143849

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end NUMINAMATH_GPT_factor_expression_l1438_143849


namespace NUMINAMATH_GPT_smallest_value_N_l1438_143856

theorem smallest_value_N (l m n N : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 143) (h2 : N = l * m * n) :
  N = 336 :=
sorry

end NUMINAMATH_GPT_smallest_value_N_l1438_143856


namespace NUMINAMATH_GPT_find_angle_l1438_143807

theorem find_angle (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_find_angle_l1438_143807


namespace NUMINAMATH_GPT_difference_between_second_and_third_levels_l1438_143804

def total_parking_spots : ℕ := 400
def first_level_open_spots : ℕ := 58
def second_level_open_spots : ℕ := first_level_open_spots + 2
def fourth_level_open_spots : ℕ := 31
def total_full_spots : ℕ := 186

def total_open_spots : ℕ := total_parking_spots - total_full_spots

def third_level_open_spots : ℕ := 
  total_open_spots - (first_level_open_spots + second_level_open_spots + fourth_level_open_spots)

def difference_open_spots : ℕ := third_level_open_spots - second_level_open_spots

theorem difference_between_second_and_third_levels : difference_open_spots = 5 :=
sorry

end NUMINAMATH_GPT_difference_between_second_and_third_levels_l1438_143804


namespace NUMINAMATH_GPT_mean_of_remaining_two_numbers_l1438_143881

/-- 
Given seven numbers:
a = 1870, b = 1995, c = 2020, d = 2026, e = 2110, f = 2124, g = 2500
and the condition that the mean of five of these numbers is 2100,
prove that the mean of the remaining two numbers is 2072.5.
-/
theorem mean_of_remaining_two_numbers :
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  a + b + c + d + e + f + g = 14645 →
  (a + b + c + d + e + f + g) = 14645 →
  (a + b + c + d + e) / 5 = 2100 →
  (f + g) / 2 = 2072.5 :=
by
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  sorry

end NUMINAMATH_GPT_mean_of_remaining_two_numbers_l1438_143881


namespace NUMINAMATH_GPT_max_parts_by_rectangles_l1438_143860

theorem max_parts_by_rectangles (n : ℕ) : 
  ∃ S : ℕ, S = 2 * n^2 - 2 * n + 2 :=
by
  sorry

end NUMINAMATH_GPT_max_parts_by_rectangles_l1438_143860


namespace NUMINAMATH_GPT_smallest_n_45_l1438_143803

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_n_45_l1438_143803


namespace NUMINAMATH_GPT_B_k_largest_at_45_l1438_143877

def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1)^k

theorem B_k_largest_at_45 : ∀ k : ℕ, k = 45 → ∀ m : ℕ, m ≠ 45 → B_k 45 > B_k m :=
by
  intro k h_k m h_m
  sorry

end NUMINAMATH_GPT_B_k_largest_at_45_l1438_143877


namespace NUMINAMATH_GPT_some_number_value_l1438_143847

theorem some_number_value (some_number : ℝ) (h : (some_number * 14) / 100 = 0.045388) :
  some_number = 0.3242 :=
sorry

end NUMINAMATH_GPT_some_number_value_l1438_143847


namespace NUMINAMATH_GPT_minimum_value_of_function_l1438_143874

theorem minimum_value_of_function (x : ℝ) (h : x > 1) : 
  (x + (1 / x) + (16 * x) / (x^2 + 1)) ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_function_l1438_143874


namespace NUMINAMATH_GPT_a_squared_plus_b_squared_equals_61_l1438_143802

theorem a_squared_plus_b_squared_equals_61 (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 :=
sorry

end NUMINAMATH_GPT_a_squared_plus_b_squared_equals_61_l1438_143802


namespace NUMINAMATH_GPT_parabola_tangent_xsum_l1438_143819

theorem parabola_tangent_xsum
  (p : ℝ) (hp : p > 0) 
  (X_A X_B X_M : ℝ) 
  (hxM_line : ∃ y, y = -2 * p ∧ y = -2 * p)
  (hxA_tangent : ∃ y, y = (X_A / p) * (X_A - X_M) - 2 * p)
  (hxB_tangent : ∃ y, y = (X_B / p) * (X_B - X_M) - 2 * p) :
  2 * X_M = X_A + X_B :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_xsum_l1438_143819


namespace NUMINAMATH_GPT_simple_random_sampling_correct_statements_l1438_143876

theorem simple_random_sampling_correct_statements :
  let N : ℕ := 10
  -- Conditions for simple random sampling
  let is_finite (N : ℕ) := N > 0
  let is_non_sequential (N : ℕ) := N > 0 -- represents sampling does not require sequential order
  let without_replacement := true
  let equal_probability := true
  -- Verification
  (is_finite N) ∧ 
  (¬ is_non_sequential N) ∧ 
  without_replacement ∧ 
  equal_probability = true :=
by
  sorry

end NUMINAMATH_GPT_simple_random_sampling_correct_statements_l1438_143876


namespace NUMINAMATH_GPT_fill_in_the_blanks_l1438_143898

theorem fill_in_the_blanks :
  (9 / 18 = 0.5) ∧
  (27 / 54 = 0.5) ∧
  (50 / 100 = 0.5) ∧
  (10 / 20 = 0.5) ∧
  (5 / 10 = 0.5) :=
by
  sorry

end NUMINAMATH_GPT_fill_in_the_blanks_l1438_143898


namespace NUMINAMATH_GPT_russian_players_pairing_probability_l1438_143810

theorem russian_players_pairing_probability :
  let total_players := 10
  let russian_players := 4
  (russian_players * (russian_players - 1)) / (total_players * (total_players - 1)) * 
  ((russian_players - 2) * (russian_players - 3)) / ((total_players - 2) * (total_players - 3)) = 1 / 21 :=
by
  sorry

end NUMINAMATH_GPT_russian_players_pairing_probability_l1438_143810


namespace NUMINAMATH_GPT_johns_outfit_cost_l1438_143818

theorem johns_outfit_cost (pants_cost shirt_cost outfit_cost : ℝ)
    (h_pants : pants_cost = 50)
    (h_shirt : shirt_cost = pants_cost + 0.6 * pants_cost)
    (h_outfit : outfit_cost = pants_cost + shirt_cost) :
    outfit_cost = 130 :=
by
  sorry

end NUMINAMATH_GPT_johns_outfit_cost_l1438_143818


namespace NUMINAMATH_GPT_angle_SR_XY_is_70_l1438_143892

-- Define the problem conditions
variables (X Y Z V H S R : Type) 
variables (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ)

-- Set the conditions
def triangleXYZ (X Y Z V H S R : Type) (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ) : Prop :=
  angleX = 40 ∧ angleY = 70 ∧ XY = 12 ∧ XV = 2 ∧ YH = 2 ∧
  ∃ S R, S = (XY / 2) ∧ R = ((XV + YH) / 2)

-- Construct the theorem to be proven
theorem angle_SR_XY_is_70 {X Y Z V H S R : Type} 
  {angleX angleY angleZ angleSRXY : ℝ} 
  {XY XV YH : ℝ} : 
  triangleXYZ X Y Z V H S R angleX angleY angleZ angleSRXY XY XV YH →
  angleSRXY = 70 :=
by
  -- Placeholder proof steps
  sorry

end NUMINAMATH_GPT_angle_SR_XY_is_70_l1438_143892


namespace NUMINAMATH_GPT_general_term_a_general_term_b_l1438_143812

def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
∀ n, a_n n = n ∧ S_n n = (n^2 + n) / 2

def sequence_b (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :=
  (b_n 1 = 1/2) ∧
  (∀ n, b_n (n+1) = (n+1) / n * b_n n) ∧ 
  (∀ n, b_n n = n / 2) ∧ 
  (∀ n, T_n n = (n^2 + n) / 4) ∧ 
  (∀ m, m = 1 → T_n m = 1/2)

-- Arithmetic sequence {a_n}
theorem general_term_a (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 2 = 2) (h2 : S 5 = 15) :
  arithmetic_sequence a S := sorry

-- Sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (T : ℕ → ℝ) (h1 : b 1 = 1/2) (h2 : ∀ n, b (n+1) = (n+1) / n * b n) :
  sequence_b b T := sorry

end NUMINAMATH_GPT_general_term_a_general_term_b_l1438_143812


namespace NUMINAMATH_GPT_solve_equation_l1438_143844

theorem solve_equation (a b : ℕ) : 
  (a^2 = b * (b + 7) ∧ a ≥ 0 ∧ b ≥ 0) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1438_143844


namespace NUMINAMATH_GPT_smallest_positive_integer_x_l1438_143884

theorem smallest_positive_integer_x :
  ∃ x : ℕ, 42 * x + 14 ≡ 4 [MOD 26] ∧ x ≡ 3 [MOD 5] ∧ x = 38 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_l1438_143884


namespace NUMINAMATH_GPT_time_correct_l1438_143832

theorem time_correct {t : ℝ} (h : 0 < t ∧ t < 60) :
  |6 * (t + 5) - (90 + 0.5 * (t - 4))| = 180 → t = 43 := by
  sorry

end NUMINAMATH_GPT_time_correct_l1438_143832


namespace NUMINAMATH_GPT_vector_definition_l1438_143838

-- Definition of a vector's characteristics
def hasCharacteristics (vector : Type) := ∃ (magnitude : ℝ) (direction : ℂ), true

-- The statement to prove: a vector is defined by having both magnitude and direction
theorem vector_definition (vector : Type) : hasCharacteristics vector := 
sorry

end NUMINAMATH_GPT_vector_definition_l1438_143838


namespace NUMINAMATH_GPT_ratio_of_turtles_l1438_143826

noncomputable def initial_turtles_owen : ℕ := 21
noncomputable def initial_turtles_johanna : ℕ := initial_turtles_owen - 5
noncomputable def turtles_johanna_after_month : ℕ := initial_turtles_johanna / 2
noncomputable def turtles_owen_after_month : ℕ := 50 - turtles_johanna_after_month

theorem ratio_of_turtles (a b : ℕ) (h1 : a = 21) (h2 : b = 5) (h3 : initial_turtles_owen = a) (h4 : initial_turtles_johanna = initial_turtles_owen - b) 
(h5 : turtles_johanna_after_month = initial_turtles_johanna / 2) (h6 : turtles_owen_after_month = 50 - turtles_johanna_after_month) : 
turtles_owen_after_month / initial_turtles_owen = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_turtles_l1438_143826


namespace NUMINAMATH_GPT_simplify_expression_l1438_143895

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) =
  16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1438_143895


namespace NUMINAMATH_GPT_speedster_convertibles_count_l1438_143823

-- Definitions of conditions
def total_inventory (T : ℕ) : Prop := (T / 3) = 60
def number_of_speedsters (T S : ℕ) : Prop := S = (2 / 3) * T
def number_of_convertibles (S C : ℕ) : Prop := C = (4 / 5) * S

-- Primary statement to prove
theorem speedster_convertibles_count (T S C : ℕ) (h1 : total_inventory T) (h2 : number_of_speedsters T S) (h3 : number_of_convertibles S C) : C = 96 :=
by
  -- Conditions and given values are defined
  sorry

end NUMINAMATH_GPT_speedster_convertibles_count_l1438_143823


namespace NUMINAMATH_GPT_binary_arithmetic_l1438_143851

theorem binary_arithmetic :
  let a := 0b11101
  let b := 0b10011
  let c := 0b101
  (a * b) / c = 0b11101100 :=
by
  sorry

end NUMINAMATH_GPT_binary_arithmetic_l1438_143851


namespace NUMINAMATH_GPT_original_cost_of_tshirt_l1438_143815

theorem original_cost_of_tshirt
  (backpack_cost : ℕ := 10)
  (cap_cost : ℕ := 5)
  (total_spent_after_discount : ℕ := 43)
  (discount : ℕ := 2)
  (tshirt_cost_before_discount : ℕ) :
  total_spent_after_discount + discount - (backpack_cost + cap_cost) = tshirt_cost_before_discount :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_tshirt_l1438_143815


namespace NUMINAMATH_GPT_sum_of_first_five_integers_l1438_143808

theorem sum_of_first_five_integers : (1 + 2 + 3 + 4 + 5) = 15 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_five_integers_l1438_143808


namespace NUMINAMATH_GPT_min_value_frac_sum_l1438_143837

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l1438_143837


namespace NUMINAMATH_GPT_triangle_perimeter_sqrt_l1438_143848

theorem triangle_perimeter_sqrt :
  let a := Real.sqrt 8
  let b := Real.sqrt 18
  let c := Real.sqrt 32
  a + b + c = 9 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_sqrt_l1438_143848


namespace NUMINAMATH_GPT_left_handed_jazz_lovers_count_l1438_143850

noncomputable def club_members := 30
noncomputable def left_handed := 11
noncomputable def like_jazz := 20
noncomputable def right_handed_dislike_jazz := 4

theorem left_handed_jazz_lovers_count : 
  ∃ x, x + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = club_members ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_left_handed_jazz_lovers_count_l1438_143850


namespace NUMINAMATH_GPT_sum_of_variables_is_16_l1438_143896

theorem sum_of_variables_is_16 (A B C D E : ℕ)
    (h1 : C + E = 4) 
    (h2 : B + E = 7) 
    (h3 : B + D = 6) 
    (h4 : A = 6)
    (hdistinct : ∀ x y, x ≠ y → (x ≠ A ∧ x ≠ B ∧ x ≠ C ∧ x ≠ D ∧ x ≠ E) ∧ (y ≠ A ∧ y ≠ B ∧ y ≠ C ∧ y ≠ D ∧ y ≠ E)) :
    A + B + C + D + E = 16 :=
by
    sorry

end NUMINAMATH_GPT_sum_of_variables_is_16_l1438_143896


namespace NUMINAMATH_GPT_arithmetic_mean_18_27_45_l1438_143831

theorem arithmetic_mean_18_27_45 : 
  (18 + 27 + 45) / 3 = 30 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_arithmetic_mean_18_27_45_l1438_143831


namespace NUMINAMATH_GPT_converse_of_prop1_true_l1438_143853

theorem converse_of_prop1_true
  (h1 : ∀ {x : ℝ}, x^2 - 3 * x + 2 = 0 → x = 1 ∨ x = 2)
  (h2 : ∀ {x : ℝ}, -2 ≤ x ∧ x < 3 → (x - 2) * (x - 3) ≤ 0)
  (h3 : ∀ {x y : ℝ}, x = 0 ∧ y = 0 → x^2 + y^2 = 0)
  (h4 : ∀ {x y : ℕ}, x > 0 ∧ y > 0 ∧ (x + y) % 2 = 1 → (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)) :
  (∀ {x : ℝ}, x = 1 ∨ x = 2 → x^2 - 3 * x + 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_converse_of_prop1_true_l1438_143853


namespace NUMINAMATH_GPT_b_is_some_even_number_l1438_143873

noncomputable def factorable_b (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    (m * p = 15 ∧ n * q = 15) ∧ 
    (b = m * q + n * p)

theorem b_is_some_even_number (b : ℤ) 
  (h : factorable_b b) : ∃ k : ℤ, b = 2 * k := 
by
  sorry

end NUMINAMATH_GPT_b_is_some_even_number_l1438_143873


namespace NUMINAMATH_GPT_arcsin_range_l1438_143890

theorem arcsin_range (α : ℝ ) (x : ℝ ) (h₁ : x = Real.cos α) (h₂ : -Real.pi / 4 ≤ α ∧ α ≤ 3 * Real.pi / 4) : 
-Real.pi / 4 ≤ Real.arcsin x ∧ Real.arcsin x ≤ Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_arcsin_range_l1438_143890


namespace NUMINAMATH_GPT_arithmetic_sequence_l1438_143889

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence (a1 d : ℝ) (h_d : d ≠ 0) 
  (h1 : a1 + (a1 + 2 * d) = 8) 
  (h2 : (a1 + d) * (a1 + 8 * d) = (a1 + 3 * d) * (a1 + 3 * d)) :
  a_n a1 d 5 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l1438_143889


namespace NUMINAMATH_GPT_number_in_circle_Y_l1438_143814

section
variables (a b c d X Y : ℕ)

theorem number_in_circle_Y :
  a + b + X = 30 ∧
  c + d + Y = 30 ∧
  a + b + c + d = 40 ∧
  X + Y + c + b = 40 ∧
  X = 9 → Y = 11 := by
  intros h
  sorry
end

end NUMINAMATH_GPT_number_in_circle_Y_l1438_143814


namespace NUMINAMATH_GPT_second_day_speed_faster_l1438_143899

def first_day_distance := 18
def first_day_speed := 3
def first_day_time := first_day_distance / first_day_speed
def second_day_time := first_day_time - 1
def third_day_speed := 5
def third_day_time := 3
def third_day_distance := third_day_speed * third_day_time
def total_distance := 53

theorem second_day_speed_faster :
  ∃ r2, (first_day_distance + (second_day_time * r2) + third_day_distance = total_distance) → (r2 - first_day_speed = 1) :=
by
  sorry

end NUMINAMATH_GPT_second_day_speed_faster_l1438_143899


namespace NUMINAMATH_GPT_fountains_fill_pool_together_l1438_143868

-- Define the times in hours for each fountain to fill the pool
def time_fountain1 : ℚ := 5 / 2  -- 2.5 hours
def time_fountain2 : ℚ := 15 / 4 -- 3.75 hours

-- Define the rates at which each fountain can fill the pool
def rate_fountain1 : ℚ := 1 / time_fountain1
def rate_fountain2 : ℚ := 1 / time_fountain2

-- Calculate the combined rate
def combined_rate : ℚ := rate_fountain1 + rate_fountain2

-- Define the time for both fountains working together to fill the pool
def combined_time : ℚ := 1 / combined_rate

-- Prove that the combined time is indeed 1.5 hours
theorem fountains_fill_pool_together : combined_time = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_fountains_fill_pool_together_l1438_143868


namespace NUMINAMATH_GPT_value_of_f_neg_a_l1438_143858

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_neg_a_l1438_143858


namespace NUMINAMATH_GPT_bridge_length_is_115_meters_l1438_143859

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_km_per_hr : ℝ) (time_to_pass : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * (1000 / 3600)
  let total_distance := speed_m_per_s * time_to_pass
  total_distance - length_of_train

theorem bridge_length_is_115_meters :
  length_of_bridge 300 35 42.68571428571429 = 115 :=
by
  -- Here the proof has to show the steps for converting speed and calculating distances
  sorry

end NUMINAMATH_GPT_bridge_length_is_115_meters_l1438_143859


namespace NUMINAMATH_GPT_opening_night_ticket_price_l1438_143821

theorem opening_night_ticket_price :
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let matinee_price := 5
  let evening_price := 7
  let popcorn_price := 10
  let total_revenue := 1670
  let total_customers := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers := total_customers / 2
  let total_matinee_revenue := matinee_customers * matinee_price
  let total_evening_revenue := evening_customers * evening_price
  let total_popcorn_revenue := popcorn_customers * popcorn_price
  let known_revenue := total_matinee_revenue + total_evening_revenue + total_popcorn_revenue
  let opening_night_revenue := total_revenue - known_revenue
  let opening_night_price := opening_night_revenue / opening_night_customers
  opening_night_price = 10 := by
  sorry

end NUMINAMATH_GPT_opening_night_ticket_price_l1438_143821


namespace NUMINAMATH_GPT_box_dimensions_correct_l1438_143893

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end NUMINAMATH_GPT_box_dimensions_correct_l1438_143893


namespace NUMINAMATH_GPT_train_length_equals_750_l1438_143852

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end NUMINAMATH_GPT_train_length_equals_750_l1438_143852


namespace NUMINAMATH_GPT_negation_of_proposition_l1438_143842

-- Conditions
variable {x : ℝ}

-- The proposition
def proposition : Prop := ∃ x : ℝ, Real.exp x > x

-- The proof problem: proving the negation of the proposition
theorem negation_of_proposition : (¬ proposition) ↔ ∀ x : ℝ, Real.exp x ≤ x := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1438_143842


namespace NUMINAMATH_GPT_only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l1438_143875

theorem only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime (n : ℕ) : 
  Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l1438_143875


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l1438_143872

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l1438_143872


namespace NUMINAMATH_GPT_carlos_meeting_percentage_l1438_143824

-- Definitions for the given conditions
def work_day_minutes : ℕ := 10 * 60
def first_meeting_minutes : ℕ := 80
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def break_minutes : ℕ := 15
def total_meeting_and_break_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + break_minutes

-- Statement to prove
theorem carlos_meeting_percentage : 
  (total_meeting_and_break_minutes * 100 / work_day_minutes) = 56 := 
by
  sorry

end NUMINAMATH_GPT_carlos_meeting_percentage_l1438_143824


namespace NUMINAMATH_GPT_min_packs_for_126_cans_l1438_143827

-- Definition of pack sizes
def pack_sizes : List ℕ := [15, 18, 36]

-- The given total cans of soda
def total_cans : ℕ := 126

-- The minimum number of packs needed to buy exactly 126 cans of soda
def min_packs_needed (total : ℕ) (packs : List ℕ) : ℕ :=
  -- Function definition to calculate the minimum packs needed
  -- This function needs to be implemented or proven
  sorry

-- The proof that the minimum number of packs needed to buy exactly 126 cans of soda is 4
theorem min_packs_for_126_cans : min_packs_needed total_cans pack_sizes = 4 :=
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_min_packs_for_126_cans_l1438_143827


namespace NUMINAMATH_GPT_post_height_l1438_143864

theorem post_height 
  (circumference : ℕ) 
  (rise_per_circuit : ℕ) 
  (travel_distance : ℕ)
  (circuits : ℕ := travel_distance / circumference) 
  (total_rise : ℕ := circuits * rise_per_circuit) 
  (c : circumference = 3)
  (r : rise_per_circuit = 4)
  (t : travel_distance = 9) :
  total_rise = 12 := by
  sorry

end NUMINAMATH_GPT_post_height_l1438_143864


namespace NUMINAMATH_GPT_count_triangles_l1438_143855

-- Assuming the conditions are already defined and given as parameters  
-- Let's define a proposition to prove the solution

noncomputable def total_triangles_in_figure : ℕ := 68

-- Create the theorem statement:
theorem count_triangles : total_triangles_in_figure = 68 := 
by
  sorry

end NUMINAMATH_GPT_count_triangles_l1438_143855


namespace NUMINAMATH_GPT_sum_of_possible_values_for_a_l1438_143843

-- Define the conditions
variables (a b c d : ℤ)
variables (h1 : a > b) (h2 : b > c) (h3 : c > d)
variables (h4 : a + b + c + d = 52)
variables (differences : finset ℤ)

-- Hypotheses about the pairwise differences
variable (h_diff : differences = {2, 3, 5, 6, 8, 11})
variable (h_ad : a - d = 11)

-- The pairs of differences adding up to 11
variable (h_pairs1 : a - b + b - d = 11)
variable (h_pairs2 : a - c + c - d = 11)

-- The theorem to be proved
theorem sum_of_possible_values_for_a : a = 19 :=
by
-- Implemented variables and conditions correctly, and the proof is outlined.
sorry

end NUMINAMATH_GPT_sum_of_possible_values_for_a_l1438_143843


namespace NUMINAMATH_GPT_expected_total_cost_of_removing_blocks_l1438_143871

/-- 
  There are six blocks in a row labeled 1 through 6, each with weight 1.
  Two blocks x ≤ y are connected if for all x ≤ z ≤ y, block z has not been removed.
  While there is at least one block remaining, a block is chosen uniformly at random and removed.
  The cost of removing a block is the sum of the weights of the blocks that are connected to it.
  Prove that the expected total cost of removing all blocks is 163 / 10.
-/
theorem expected_total_cost_of_removing_blocks : (6:ℚ) + 5 + 8/3 + 3/2 + 4/5 + 1/3 = 163 / 10 := sorry

end NUMINAMATH_GPT_expected_total_cost_of_removing_blocks_l1438_143871


namespace NUMINAMATH_GPT_intersection_A_B_l1438_143820

def A : Set ℤ := {x | abs x < 2}
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1438_143820


namespace NUMINAMATH_GPT_quadrilateral_area_l1438_143801

theorem quadrilateral_area (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * |a - b| * |a + b| = 32) : a + b = 8 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1438_143801
