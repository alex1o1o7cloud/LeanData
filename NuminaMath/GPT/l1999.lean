import Mathlib

namespace part1_part2_l1999_199937

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l1999_199937


namespace triangle_area_eq_l1999_199997

theorem triangle_area_eq :
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  area = 9 / 4 :=
by
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  sorry

end triangle_area_eq_l1999_199997


namespace difference_digits_in_base2_l1999_199924

def binaryDigitCount (n : Nat) : Nat := Nat.log2 n + 1

theorem difference_digits_in_base2 : binaryDigitCount 1400 - binaryDigitCount 300 = 2 :=
by
  sorry

end difference_digits_in_base2_l1999_199924


namespace largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l1999_199931

noncomputable def largest_integral_x_in_ineq (x : ℤ) : Prop :=
  (2 / 5 : ℚ) < (x / 7 : ℚ) ∧ (x / 7 : ℚ) < (8 / 11 : ℚ)

theorem largest_integral_x_satisfies_ineq : largest_integral_x_in_ineq 5 :=
sorry

theorem largest_integral_x_is_5 (x : ℤ) (h : largest_integral_x_in_ineq x) : x ≤ 5 :=
sorry

end largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l1999_199931


namespace part1_part2_l1999_199991

noncomputable def f (a : ℝ) (x : ℝ) := (a - 1/2) * x^2 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f a x - 2 * a * x

theorem part1 (x : ℝ) (hxe : Real.exp (-1) ≤ x ∧ x ≤ Real.exp (1)) : 
    f (-1/2) x ≤ -1/2 - 1/2 * Real.log 2 ∧ f (-1/2) x ≥ 1 - Real.exp 2 := sorry

theorem part2 (h : ∀ x > 2, g a x < 0) : a ≤ 1/2 := sorry

end part1_part2_l1999_199991


namespace white_to_brown_eggs_ratio_l1999_199958

-- Define variables W and B (the initial numbers of white and brown eggs respectively)
variable (W B : ℕ)

-- Conditions: 
-- 1. All 5 brown eggs survived.
-- 2. Total number of eggs after dropping is 12.
def egg_conditions : Prop :=
  B = 5 ∧ (W + B) = 12

-- Prove the ratio of white eggs to brown eggs is 7/5 given these conditions.
theorem white_to_brown_eggs_ratio (h : egg_conditions W B) : W / B = 7 / 5 :=
by 
  sorry

end white_to_brown_eggs_ratio_l1999_199958


namespace wage_increase_percentage_l1999_199986

theorem wage_increase_percentage (new_wage old_wage : ℝ) (h1 : new_wage = 35) (h2 : old_wage = 25) : 
  ((new_wage - old_wage) / old_wage) * 100 = 40 := 
by
  sorry

end wage_increase_percentage_l1999_199986


namespace find_partition_l1999_199976

open Nat

def isBad (S : Finset ℕ) : Prop :=
  ∃ T : Finset ℕ, T ⊆ S ∧ T.sum id = 2012

def partition_not_bad (S : Finset ℕ) (n : ℕ) : Prop :=
  ∃ (P : Finset (Finset ℕ)), P.card = n ∧ (∀ p ∈ P, isBad p = false) ∧ (S = P.sup id)

theorem find_partition :
  ∃ n : ℕ, n = 2 ∧ partition_not_bad (Finset.range (2012 - 503) \ Finset.range 503) n :=
by
  sorry

end find_partition_l1999_199976


namespace value_to_subtract_l1999_199984

theorem value_to_subtract (N x : ℕ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 2) / 13 = 4) : x = 5 :=
by 
  sorry

end value_to_subtract_l1999_199984


namespace yogurt_count_l1999_199966

theorem yogurt_count (Y : ℕ) 
  (ice_cream_cartons : ℕ := 20)
  (cost_ice_cream_per_carton : ℕ := 6)
  (cost_yogurt_per_carton : ℕ := 1)
  (spent_more_on_ice_cream : ℕ := 118)
  (total_cost_ice_cream : ℕ := ice_cream_cartons * cost_ice_cream_per_carton)
  (total_cost_yogurt : ℕ := Y * cost_yogurt_per_carton)
  (expenditure_condition : total_cost_ice_cream = total_cost_yogurt + spent_more_on_ice_cream) :
  Y = 2 :=
by {
  sorry
}

end yogurt_count_l1999_199966


namespace digit_makes_5678d_multiple_of_9_l1999_199904

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l1999_199904


namespace exists_same_color_ratios_l1999_199982

-- Definition of coloring function.
def coloring : ℕ → Fin 2 := sorry

-- Definition of the problem: there exist A, B, C such that A : C = C : B,
-- and A, B, C are of same color.
theorem exists_same_color_ratios :
  ∃ A B C : ℕ, coloring A = coloring B ∧ coloring B = coloring C ∧ 
  (A : ℚ) / C = (C : ℚ) / B := 
sorry

end exists_same_color_ratios_l1999_199982


namespace percentage_passed_both_subjects_l1999_199945

def failed_H : ℝ := 0.35
def failed_E : ℝ := 0.45
def failed_HE : ℝ := 0.20

theorem percentage_passed_both_subjects :
  (100 - (failed_H * 100 + failed_E * 100 - failed_HE * 100)) = 40 := 
by
  sorry

end percentage_passed_both_subjects_l1999_199945


namespace problem_statement_l1999_199942

noncomputable def f (x : ℝ) := 3 * x ^ 5 + 4 * x ^ 4 - 5 * x ^ 3 + 2 * x ^ 2 + x + 6
noncomputable def d (x : ℝ) := x ^ 3 + 2 * x ^ 2 - x - 3
noncomputable def q (x : ℝ) := 3 * x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) := 19 * x ^ 2 - 11 * x - 57

theorem problem_statement : (f 1 = q 1 * d 1 + r 1) ∧ q 1 + r 1 = -47 := by
  sorry

end problem_statement_l1999_199942


namespace vanya_number_l1999_199920

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end vanya_number_l1999_199920


namespace line_equation_slope_intercept_l1999_199903

theorem line_equation_slope_intercept (m b : ℝ) (h1 : m = -1) (h2 : b = -1) :
  ∀ x y : ℝ, y = m * x + b → x + y + 1 = 0 :=
by
  intros x y h
  sorry

end line_equation_slope_intercept_l1999_199903


namespace angle_B_in_equilateral_triangle_l1999_199918

theorem angle_B_in_equilateral_triangle (A B C : ℝ) (h_angle_sum : A + B + C = 180) (h_A : A = 80) (h_BC : B = C) :
  B = 50 :=
by
  -- Conditions
  have h1 : A = 80 := by exact h_A
  have h2 : B = C := by exact h_BC
  have h3 : A + B + C = 180 := by exact h_angle_sum

  sorry -- completing the proof is not required

end angle_B_in_equilateral_triangle_l1999_199918


namespace annalise_total_cost_l1999_199960

/-- 
Given conditions:
- 25 boxes of tissues.
- Each box contains 18 packs.
- Each pack contains 150 tissues.
- Each tissue costs $0.06.
- A 10% discount on the total price of the packs in each box.

Prove:
The total amount of money Annalise spent is $3645.
-/
theorem annalise_total_cost :
  let boxes := 25
  let packs_per_box := 18
  let tissues_per_pack := 150
  let cost_per_tissue := 0.06
  let discount_rate := 0.10
  let price_per_box := (packs_per_box * tissues_per_pack * cost_per_tissue)
  let discount_per_box := discount_rate * price_per_box
  let discounted_price_per_box := price_per_box - discount_per_box
  let total_cost := discounted_price_per_box * boxes
  total_cost = 3645 :=
by
  sorry

end annalise_total_cost_l1999_199960


namespace pq_sum_l1999_199983

theorem pq_sum {p q : ℤ}
  (h : ∀ x : ℤ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) :
  p + q = 20 :=
sorry

end pq_sum_l1999_199983


namespace problem_l1999_199985

def f (u : ℝ) : ℝ := u^2 - 2

theorem problem : f 3 = 7 := 
by sorry

end problem_l1999_199985


namespace second_term_of_geometric_series_l1999_199915

theorem second_term_of_geometric_series 
  (a : ℝ) (r : ℝ) (S : ℝ) :
  r = 1 / 4 → S = 40 → S = a / (1 - r) → a * r = 7.5 :=
by
  intros hr hS hSum
  sorry

end second_term_of_geometric_series_l1999_199915


namespace candy_pieces_total_l1999_199934

def number_of_packages_of_candy := 45
def pieces_per_package := 9

theorem candy_pieces_total : number_of_packages_of_candy * pieces_per_package = 405 :=
by
  sorry

end candy_pieces_total_l1999_199934


namespace bullet_train_pass_time_l1999_199900

noncomputable def time_to_pass (length_train : ℕ) (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) : ℝ := 
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * 1000 / 3600
  length_train / relative_speed_mps

def length_train := 350
def speed_train_kmph := 75
def speed_man_kmph := 12

theorem bullet_train_pass_time : 
  abs (time_to_pass length_train speed_train_kmph speed_man_kmph - 14.47) < 0.01 :=
by
  sorry

end bullet_train_pass_time_l1999_199900


namespace transformed_system_solution_l1999_199916

theorem transformed_system_solution :
  (∀ (a b : ℝ), 2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 → a = 8.3 ∧ b = 1.2) →
  (∀ (x y : ℝ), 2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9 →
    x = 6.3 ∧ y = 2.2) :=
by
  intro h1
  intro x y
  intro hy
  sorry

end transformed_system_solution_l1999_199916


namespace gondor_repaired_3_phones_on_monday_l1999_199979

theorem gondor_repaired_3_phones_on_monday :
  ∃ P : ℕ, 
    (10 * P + 10 * 5 + 20 * 2 + 20 * 4 = 200) ∧
    P = 3 :=
by
  sorry

end gondor_repaired_3_phones_on_monday_l1999_199979


namespace bekahs_reading_l1999_199955

def pages_per_day (total_pages read_pages days_left : ℕ) : ℕ :=
  (total_pages - read_pages) / days_left

theorem bekahs_reading :
  pages_per_day 408 113 5 = 59 := by
  sorry

end bekahs_reading_l1999_199955


namespace sum_smallest_and_largest_prime_between_1_and_50_l1999_199987

noncomputable def smallest_prime_between_1_and_50 : ℕ := 2
noncomputable def largest_prime_between_1_and_50 : ℕ := 47

theorem sum_smallest_and_largest_prime_between_1_and_50 : 
  smallest_prime_between_1_and_50 + largest_prime_between_1_and_50 = 49 := 
by
  sorry

end sum_smallest_and_largest_prime_between_1_and_50_l1999_199987


namespace unread_pages_when_a_is_11_l1999_199914

variable (a : ℕ)

def total_pages : ℕ := 250
def pages_per_day : ℕ := 15

def unread_pages_after_a_days (a : ℕ) : ℕ := total_pages - pages_per_day * a

theorem unread_pages_when_a_is_11 : unread_pages_after_a_days 11 = 85 :=
by
  sorry

end unread_pages_when_a_is_11_l1999_199914


namespace min_value_l1999_199943

theorem min_value (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/29 :=
sorry

end min_value_l1999_199943


namespace gcd_binom_integer_l1999_199951

theorem gcd_binom_integer (n m : ℕ) (hnm : n ≥ m) (hm : m ≥ 1) :
  (Nat.gcd m n) * Nat.choose n m % n = 0 := sorry

end gcd_binom_integer_l1999_199951


namespace range_of_a_l1999_199940

def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

theorem range_of_a (a : ℝ) : (-2 / 3 : ℝ) ≤ a ∧ a < 0 := sorry

end range_of_a_l1999_199940


namespace negation_correct_l1999_199907

variable {α : Type*} (A B : Set α)

-- Define the original proposition
def original_proposition : Prop := A ∪ B = A → A ∩ B = B

-- Define the negation of the original proposition
def negation_proposition : Prop := A ∪ B ≠ A → A ∩ B ≠ B

-- State that the negation of the original proposition is equivalent to the negation proposition
theorem negation_correct : ¬(original_proposition A B) ↔ negation_proposition A B := by sorry

end negation_correct_l1999_199907


namespace isosceles_triangle_perimeter_l1999_199999

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Define the roots based on factorization of the given equation
def root1 := 2
def root2 := 4

-- Define the perimeter of the isosceles triangle given the roots
def triangle_perimeter := root2 + root2 + root1

-- Prove that the perimeter of the isosceles triangle is 10
theorem isosceles_triangle_perimeter : triangle_perimeter = 10 :=
by
  -- We need to verify the solution without providing the steps explicitly
  sorry

end isosceles_triangle_perimeter_l1999_199999


namespace vector_addition_example_l1999_199989

def vector_addition (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem vector_addition_example : vector_addition (1, -1) (-1, 2) = (0, 1) := 
by 
  unfold vector_addition 
  simp
  sorry

end vector_addition_example_l1999_199989


namespace possible_permutations_100_l1999_199946

def tasty_permutations (n : ℕ) : ℕ := sorry

theorem possible_permutations_100 :
  2^100 ≤ tasty_permutations 100 ∧ tasty_permutations 100 ≤ 4^100 :=
sorry

end possible_permutations_100_l1999_199946


namespace largest_possible_s_l1999_199971

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) (h3 : (r - 2) * 180 * s = (s - 2) * 180 * r * 61 / 60) : s = 118 :=
sorry

end largest_possible_s_l1999_199971


namespace bread_problem_l1999_199922

variable (x : ℝ)

theorem bread_problem (h1 : x > 0) :
  (15 / x) - 1 = 14 / (x + 2) :=
sorry

end bread_problem_l1999_199922


namespace part1_part2_l1999_199978

-- Part 1
theorem part1 (x : ℝ) (h1 : 2 * x = 3 * x - 1) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h2 : x < 0) (h3 : |2 * x| + |3 * x - 1| = 16) : x = -3 :=
by
  sorry

end part1_part2_l1999_199978


namespace no_pos_int_sol_l1999_199944

theorem no_pos_int_sol (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ¬ ∃ (k : ℕ), (15 * a + b) * (a + 15 * b) = 3^k := 
sorry

end no_pos_int_sol_l1999_199944


namespace min_digits_decimal_correct_l1999_199923

noncomputable def min_digits_decimal : ℕ := 
  let n : ℕ := 123456789
  let d : ℕ := 2^26 * 5^4
  26 -- As per the problem statement

theorem min_digits_decimal_correct :
  let n := 123456789
  let d := 2^26 * 5^4
  ∀ x:ℕ, (∃ k:ℕ, n = k * 10^x) → x ≥ min_digits_decimal := 
by
  sorry

end min_digits_decimal_correct_l1999_199923


namespace students_present_in_class_l1999_199936

theorem students_present_in_class :
  ∀ (total_students absent_percentage : ℕ), 
    total_students = 50 → absent_percentage = 12 → 
    (88 * total_students / 100) = 44 :=
by
  intros total_students absent_percentage h1 h2
  sorry

end students_present_in_class_l1999_199936


namespace window_treatments_total_cost_l1999_199990

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end window_treatments_total_cost_l1999_199990


namespace product_of_solutions_of_t_squared_eq_49_l1999_199953

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end product_of_solutions_of_t_squared_eq_49_l1999_199953


namespace omega_value_l1999_199917

theorem omega_value (ω : ℕ) (h : ω > 0) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + Real.pi / 4)) 
  (h2 : ∀ x y, (Real.pi / 6 < x ∧ x < Real.pi / 3) → (Real.pi / 6 < y ∧ y < Real.pi / 3) → x < y → f y < f x) :
    ω = 2 ∨ ω = 3 := 
sorry

end omega_value_l1999_199917


namespace toothpicks_at_200th_stage_l1999_199910

-- Define initial number of toothpicks at stage 1
def a_1 : ℕ := 4

-- Define the function to compute the number of toothpicks at stage n, taking into account the changing common difference
def a (n : ℕ) : ℕ :=
  if n = 1 then 4
  else if n <= 49 then 4 + 4 * (n - 1)
  else if n <= 99 then 200 + 5 * (n - 50)
  else if n <= 149 then 445 + 6 * (n - 100)
  else if n <= 199 then 739 + 7 * (n - 150)
  else 0  -- This covers cases not considered in the problem for clarity

-- State the theorem to check the number of toothpicks at stage 200
theorem toothpicks_at_200th_stage : a 200 = 1082 :=
  sorry

end toothpicks_at_200th_stage_l1999_199910


namespace series_sum_l1999_199975

open BigOperators

theorem series_sum :
  (∑ n in Finset.range 99, (1 : ℝ) / ((n + 1) * (n + 2))) = 99 / 100 :=
by
  sorry

end series_sum_l1999_199975


namespace insulin_pills_per_day_l1999_199957

def conditions (I B A : ℕ) : Prop := 
  B = 3 ∧ A = 2 * B ∧ 7 * (I + B + A) = 77

theorem insulin_pills_per_day : ∃ (I : ℕ), ∀ (B A : ℕ), conditions I B A → I = 2 := by
  sorry

end insulin_pills_per_day_l1999_199957


namespace calculation_l1999_199963

theorem calculation : 
  ((18 ^ 13 * 18 ^ 11) ^ 2 / 6 ^ 8) * 3 ^ 4 = 2 ^ 40 * 3 ^ 92 :=
by sorry

end calculation_l1999_199963


namespace problem1_problem2_l1999_199962

def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x | x > 1 ∨ x < -6}

theorem problem1 (a : ℝ) : (setA a ∩ setB = ∅) → (-6 ≤ a ∧ a ≤ -2) := by
  intro h
  sorry

theorem problem2 (a : ℝ) : (setA a ∪ setB = setB) → (a < -9 ∨ a > 1) := by
  intro h
  sorry

end problem1_problem2_l1999_199962


namespace solution_set_inequality_l1999_199972

variable {x : ℝ}
variable {a b : ℝ}

theorem solution_set_inequality (h₁ : ∀ x : ℝ, (ax^2 + bx - 1 > 0) ↔ (-1/2 < x ∧ x < -1/3)) :
  ∀ x : ℝ, (x^2 - bx - a ≥ 0) ↔ (x ≤ -3 ∨ x ≥ -2) := 
sorry

end solution_set_inequality_l1999_199972


namespace max_sum_of_segments_l1999_199930

theorem max_sum_of_segments (A B C D : ℝ × ℝ × ℝ)
    (h : (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D ≤ 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1)
      ∨ (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D > 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1))
    : dist A B + dist A C + dist A D + dist B C + dist B D + dist C D ≤ 5 + Real.sqrt 3 := sorry

end max_sum_of_segments_l1999_199930


namespace point_2000_coordinates_l1999_199932

-- Definition to describe the spiral numbering system in the first quadrant
def spiral_number (n : ℕ) : ℕ × ℕ := sorry

-- The task is to prove that the coordinates of the 2000th point are (44, 25).
theorem point_2000_coordinates : spiral_number 2000 = (44, 25) :=
by
  sorry

end point_2000_coordinates_l1999_199932


namespace school_spent_440_l1999_199996

-- Definition based on conditions listed in part a)
def cost_of_pencils (cartons_pencils : ℕ) (boxes_per_carton_pencils : ℕ) (cost_per_box_pencils : ℕ) : ℕ := 
  cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils

def cost_of_markers (cartons_markers : ℕ) (cost_per_carton_markers : ℕ) : ℕ := 
  cartons_markers * cost_per_carton_markers

noncomputable def total_cost (cartons_pencils cartons_markers boxes_per_carton_pencils cost_per_box_pencils cost_per_carton_markers : ℕ) : ℕ := 
  cost_of_pencils cartons_pencils boxes_per_carton_pencils cost_per_box_pencils + 
  cost_of_markers cartons_markers cost_per_carton_markers

-- Theorem statement to prove the total cost is $440 given the conditions
theorem school_spent_440 : total_cost 20 10 10 2 4 = 440 := by 
  sorry

end school_spent_440_l1999_199996


namespace dawson_failed_by_36_l1999_199933

-- Define the constants and conditions
def max_marks : ℕ := 220
def passing_percentage : ℝ := 0.3
def marks_obtained : ℕ := 30

-- Calculate the minimum passing marks
noncomputable def min_passing_marks : ℝ :=
  passing_percentage * max_marks

-- Calculate the marks Dawson failed by
noncomputable def marks_failed_by : ℝ :=
  min_passing_marks - marks_obtained

-- State the theorem
theorem dawson_failed_by_36 :
  marks_failed_by = 36 := by
  -- Proof is omitted
  sorry

end dawson_failed_by_36_l1999_199933


namespace train_a_distance_at_meeting_l1999_199925

noncomputable def train_a_speed : ℝ := 75 / 3
noncomputable def train_b_speed : ℝ := 75 / 2
noncomputable def relative_speed : ℝ := train_a_speed + train_b_speed
noncomputable def time_until_meet : ℝ := 75 / relative_speed
noncomputable def distance_traveled_by_train_a : ℝ := train_a_speed * time_until_meet

theorem train_a_distance_at_meeting : distance_traveled_by_train_a = 30 := by
  sorry

end train_a_distance_at_meeting_l1999_199925


namespace value_of_a3_l1999_199901

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem value_of_a3 (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 + a 4 = 20) :
  a 2 = 4 :=
sorry

end value_of_a3_l1999_199901


namespace minimum_value_expression_l1999_199938

-- Define the conditions for positive real numbers
variables (a b c : ℝ)
variable (h_a : 0 < a)
variable (h_b : 0 < b)
variable (h_c : 0 < c)

-- State the theorem to prove the minimum value of the expression
theorem minimum_value_expression (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : 
  (a / b) + (b / c) + (c / a) ≥ 3 := 
sorry

end minimum_value_expression_l1999_199938


namespace sugar_solution_sweeter_l1999_199965

variables (a b m : ℝ)

theorem sugar_solution_sweeter (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  (a / b < (a + m) / (b + m)) :=
sorry

end sugar_solution_sweeter_l1999_199965


namespace tickets_per_friend_l1999_199959

-- Defining the conditions
def initial_tickets := 11
def remaining_tickets := 3
def friends := 4

-- Statement to prove
theorem tickets_per_friend (h_tickets_given : initial_tickets - remaining_tickets = 8) : (initial_tickets - remaining_tickets) / friends = 2 :=
by
  sorry

end tickets_per_friend_l1999_199959


namespace parameter_values_for_three_distinct_roots_l1999_199905

theorem parameter_values_for_three_distinct_roots (a : ℝ) :
  (∀ x : ℝ, (|x^3 - a^3| = x - a) → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ↔ 
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end parameter_values_for_three_distinct_roots_l1999_199905


namespace profit_percentage_approx_l1999_199967

-- Define the cost price of the first item
def CP1 (S1 : ℚ) : ℚ := 0.81 * S1

-- Define the selling price of the second item as 10% less than the first
def S2 (S1 : ℚ) : ℚ := 0.90 * S1

-- Define the cost price of the second item as 81% of its selling price
def CP2 (S1 : ℚ) : ℚ := 0.81 * (S2 S1)

-- Define the total selling price before tax
def TSP (S1 : ℚ) : ℚ := S1 + S2 S1

-- Define the total amount received after a 5% tax
def TAR (S1 : ℚ) : ℚ := TSP S1 * 0.95

-- Define the total cost price of both items
def TCP (S1 : ℚ) : ℚ := CP1 S1 + CP2 S1

-- Define the profit
def P (S1 : ℚ) : ℚ := TAR S1 - TCP S1

-- Define the profit percentage
def ProfitPercentage (S1 : ℚ) : ℚ := (P S1 / TCP S1) * 100

-- Prove the profit percentage is approximately 17.28%
theorem profit_percentage_approx (S1 : ℚ) : abs (ProfitPercentage S1 - 17.28) < 0.01 :=
by
  sorry

end profit_percentage_approx_l1999_199967


namespace triangle_area_ratio_l1999_199935

/-
In triangle XYZ, XY=12, YZ=16, and XZ=20. Point D is on XY,
E is on YZ, and F is on XZ. Let XD=p*XY, YE=q*YZ, and ZF=r*XZ,
where p, q, r are positive and satisfy p+q+r=0.9 and p^2+q^2+r^2=0.29.
Prove that the ratio of the area of triangle DEF to the area of triangle XYZ 
can be written in the form m/n where m, n are relatively prime positive 
integers and m+n=137.
-/

theorem triangle_area_ratio :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m + n = 137 ∧ 
  ∃ (p q r : ℝ), p + q + r = 0.9 ∧ p^2 + q^2 + r^2 = 0.29 ∧ 
                  ∀ (XY YZ XZ : ℝ), XY = 12 ∧ YZ = 16 ∧ XZ = 20 → 
                  (1 - (p * (1 - r) + q * (1 - p) + r * (1 - q))) = (37 / 100) :=
by
   sorry

end triangle_area_ratio_l1999_199935


namespace random_event_proof_l1999_199906

-- Definitions based on conditions
def event1 := "Tossing a coin twice in a row, and both times it lands heads up."
def event2 := "Opposite charges attract each other."
def event3 := "Water freezes at 1℃ under standard atmospheric pressure."

def is_random_event (event: String) : Prop :=
  event = event1 ∨ event = event2 ∨ event = event3 → event = event1

theorem random_event_proof : is_random_event event1 ∧ ¬is_random_event event2 ∧ ¬is_random_event event3 :=
by
  -- Proof goes here
  sorry

end random_event_proof_l1999_199906


namespace parabola_vertex_l1999_199969

-- Definition of the quadratic function representing the parabola
def parabola (x : ℝ) : ℝ := (3 * x - 1) ^ 2 + 2

-- Statement asserting the coordinates of the vertex of the given parabola
theorem parabola_vertex :
  ∃ h k : ℝ, ∀ x : ℝ, parabola x = 9 * (x - h) ^ 2 + k ∧ h = 1/3 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l1999_199969


namespace average_after_discard_l1999_199988

theorem average_after_discard (avg : ℝ) (n : ℕ) (a b : ℝ) (new_avg : ℝ) :
  avg = 62 →
  n = 50 →
  a = 45 →
  b = 55 →
  new_avg = 62.5 →
  (avg * n - (a + b)) / (n - 2) = new_avg := 
by
  intros h_avg h_n h_a h_b h_new_avg
  rw [h_avg, h_n, h_a, h_b, h_new_avg]
  sorry

end average_after_discard_l1999_199988


namespace necessary_not_sufficient_l1999_199954

theorem necessary_not_sufficient (x : ℝ) : (x > 5) → (x > 2) ∧ ¬((x > 2) → (x > 5)) :=
by
  sorry

end necessary_not_sufficient_l1999_199954


namespace Maria_drove_approximately_517_miles_l1999_199949

noncomputable def carRentalMaria (daily_rate per_mile_charge discount_rate insurance_rate rental_duration total_invoice : ℝ) (discount_threshold : ℕ) : ℝ :=
  let total_rental_cost := rental_duration * daily_rate
  let discount := if rental_duration ≥ discount_threshold then discount_rate * total_rental_cost else 0
  let discounted_cost := total_rental_cost - discount
  let insurance_cost := rental_duration * insurance_rate
  let cost_without_mileage := discounted_cost + insurance_cost
  let mileage_cost := total_invoice - cost_without_mileage
  mileage_cost / per_mile_charge

noncomputable def approx_equal (a b : ℝ) (epsilon : ℝ := 1) : Prop :=
  abs (a - b) < epsilon

theorem Maria_drove_approximately_517_miles :
  approx_equal (carRentalMaria 35 0.09 0.10 5 4 192.50 3) 517 :=
by
  sorry

end Maria_drove_approximately_517_miles_l1999_199949


namespace inequality_solution_l1999_199993

theorem inequality_solution (x : ℝ) (h : (x + 1) / 2 ≥ x / 3) : x ≥ -3 :=
by
  sorry

end inequality_solution_l1999_199993


namespace xyz_value_l1999_199941

noncomputable def positive (x : ℝ) : Prop := 0 < x

theorem xyz_value (x y z : ℝ) (hx : positive x) (hy : positive y) (hz : positive z): 
  (x + 1/y = 5) → (y + 1/z = 2) → (z + 1/x = 8/3) → x * y * z = (17 + Real.sqrt 285) / 2 :=
by
  sorry

end xyz_value_l1999_199941


namespace equilateral_triangle_l1999_199947

variable {a b c : ℝ}

-- Conditions
def condition1 (a b c : ℝ) : Prop :=
  (a + b + c) * (b + c - a) = 3 * b * c

def condition2 (a b c : ℝ) (cos_B cos_C : ℝ) : Prop :=
  c * cos_B = b * cos_C

-- Theorem statement
theorem equilateral_triangle (a b c : ℝ) (cos_B cos_C : ℝ)
  (h1 : condition1 a b c)
  (h2 : condition2 a b c cos_B cos_C) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l1999_199947


namespace minimum_b_l1999_199911

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

noncomputable def g (a b x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ a then f a b x else f a b (f a b x)

theorem minimum_b {a b : ℝ} (ha : 0 < a) :
  (∀ x : ℝ, 0 ≤ x → g a b x > g a b (x - 1)) → b ≥ 1 / 4 :=
sorry

end minimum_b_l1999_199911


namespace brownies_per_person_l1999_199961

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_l1999_199961


namespace perpendicularity_condition_l1999_199973

theorem perpendicularity_condition 
  (A B C D E F k b : ℝ) 
  (h1 : b ≠ 0)
  (line : ∀ (x : ℝ), y = k * x + b)
  (curve : ∀ (x y : ℝ), A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F = 0):
  A * b^2 - 2 * D * k * b + F * k^2 + C * b^2 + 2 * E * b + F = 0 :=
sorry

end perpendicularity_condition_l1999_199973


namespace find_three_numbers_l1999_199950

theorem find_three_numbers (x y z : ℝ) 
  (h1 : x - y = 12) 
  (h2 : (x + y) / 4 = 7) 
  (h3 : z = 2 * y) 
  (h4 : x + z = 24) : 
  x = 20 ∧ y = 8 ∧ z = 16 := by
  sorry

end find_three_numbers_l1999_199950


namespace factorization_correct_l1999_199995

variable (a b : ℝ)

theorem factorization_correct :
  12 * a ^ 3 * b - 12 * a ^ 2 * b + 3 * a * b = 3 * a * b * (2 * a - 1) ^ 2 :=
by 
  sorry

end factorization_correct_l1999_199995


namespace number_of_apple_trees_l1999_199919

variable (T : ℕ) -- Declare the number of apple trees as a natural number

-- Define the conditions
def picked_apples := 8 * T
def remaining_apples := 9
def initial_apples := 33

-- The statement to prove Rachel has 3 apple trees
theorem number_of_apple_trees :
  initial_apples - picked_apples + remaining_apples = initial_apples → T = 3 := 
by
  sorry

end number_of_apple_trees_l1999_199919


namespace solve_for_y_l1999_199964

theorem solve_for_y : ∀ (y : ℝ), 4 + 2.3 * y = 1.7 * y - 20 → y = -40 :=
by
  sorry

end solve_for_y_l1999_199964


namespace possible_to_fill_grid_l1999_199977

/-- Define the grid as a 2D array where each cell contains either 0 or 1. --/
def grid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), i < 5 → j < 5 → f i j = 0 ∨ f i j = 1

/-- Ensure the sum of every 2x2 subgrid is divisible by 3. --/
def divisible_by_3_in_subgrid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < 4 → j < 4 → (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 3 = 0

/-- Ensure both 0 and 1 are present in the grid. --/
def contains_0_and_1 (f : ℕ → ℕ → ℕ) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 0) ∧ (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 1)

/-- The main theorem stating the possibility of such a grid. --/
theorem possible_to_fill_grid :
  ∃ f, grid f ∧ divisible_by_3_in_subgrid f ∧ contains_0_and_1 f :=
sorry

end possible_to_fill_grid_l1999_199977


namespace Jennifer_has_24_dollars_left_l1999_199980

def remaining_money (initial amount: ℕ) (spent_sandwich spent_museum_ticket spent_book: ℕ) : ℕ :=
  initial - (spent_sandwich + spent_museum_ticket + spent_book)

theorem Jennifer_has_24_dollars_left :
  remaining_money 180 (1/5*180) (1/6*180) (1/2*180) = 24 :=
by
  sorry

end Jennifer_has_24_dollars_left_l1999_199980


namespace largest_multiple_5_6_lt_1000_is_990_l1999_199998

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end largest_multiple_5_6_lt_1000_is_990_l1999_199998


namespace proof_a_eq_neg2x_or_3x_l1999_199928

theorem proof_a_eq_neg2x_or_3x (a b x : ℝ) (h1 : a - b = x) (h2 : a^3 - b^3 = 19 * x^3) (h3 : x ≠ 0) : 
  a = -2 * x ∨ a = 3 * x :=
  sorry

end proof_a_eq_neg2x_or_3x_l1999_199928


namespace h_has_only_one_zero_C2_below_C1_l1999_199948

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 - 1/x
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem h_has_only_one_zero (x : ℝ) (hx : x > 0) : 
  ∃! (x0 : ℝ), x0 > 0 ∧ h x0 = 0 := sorry

theorem C2_below_C1 (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) : 
  g x < f x := sorry

end h_has_only_one_zero_C2_below_C1_l1999_199948


namespace total_money_spent_l1999_199994

/-- Erika, Elizabeth, Emma, and Elsa went shopping on Wednesday.
Emma spent $58.
Erika spent $20 more than Emma.
Elsa spent twice as much as Emma.
Elizabeth spent four times as much as Elsa.
Erika received a 10% discount on what she initially spent.
Elizabeth had to pay a 6% tax on her purchases.
Prove that the total amount of money they spent is $736.04.
-/
theorem total_money_spent :
  let emma_spent := 58
  let erika_initial_spent := emma_spent + 20
  let erika_discount := 0.10 * erika_initial_spent
  let erika_final_spent := erika_initial_spent - erika_discount
  let elsa_spent := 2 * emma_spent
  let elizabeth_initial_spent := 4 * elsa_spent
  let elizabeth_tax := 0.06 * elizabeth_initial_spent
  let elizabeth_final_spent := elizabeth_initial_spent + elizabeth_tax
  let total_spent := emma_spent + erika_final_spent + elsa_spent + elizabeth_final_spent
  total_spent = 736.04 := by
  sorry

end total_money_spent_l1999_199994


namespace line_repr_exists_same_line_iff_scalar_multiple_l1999_199921

-- Given that D is a line in 3D space, there exist a, b, c not all zero
theorem line_repr_exists
  (D : Set (ℝ × ℝ × ℝ)) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  (D = {p | ∃ (u v w : ℝ), p = (u, v, w) ∧ a * u + b * v + c * w = 0}) :=
sorry

-- Given two lines represented by different coefficients being the same
-- Prove that the coefficients are scalar multiples of each other
theorem same_line_iff_scalar_multiple
  (α1 β1 γ1 α2 β2 γ2 : ℝ) :
  (∀ (u v w : ℝ), α1 * u + β1 * v + γ1 * w = 0 ↔ α2 * u + β2 * v + γ2 * w = 0) ↔
  (∃ k : ℝ, k ≠ 0 ∧ α2 = k * α1 ∧ β2 = k * β1 ∧ γ2 = k * γ1) :=
sorry

end line_repr_exists_same_line_iff_scalar_multiple_l1999_199921


namespace marks_in_biology_l1999_199974

theorem marks_in_biology (E M P C : ℝ) (A B : ℝ)
  (h1 : E = 90)
  (h2 : M = 92)
  (h3 : P = 85)
  (h4 : C = 87)
  (h5 : A = 87.8) 
  (h6 : (E + M + P + C + B) / 5 = A) : 
  B = 85 := 
by
  -- Placeholder for the proof
  sorry

end marks_in_biology_l1999_199974


namespace economical_club_l1999_199912

-- Definitions of cost functions for Club A and Club B
def f (x : ℕ) : ℕ := 5 * x

def g (x : ℕ) : ℕ := if x ≤ 30 then 90 else 2 * x + 30

-- Theorem to determine the more economical club
theorem economical_club (x : ℕ) (hx : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 30 → f x > g x) ∧
  (30 < x ∧ x ≤ 40 → f x > g x) :=
sorry

end economical_club_l1999_199912


namespace petya_vasya_three_numbers_equal_l1999_199939

theorem petya_vasya_three_numbers_equal (a b c : ℕ) :
  gcd a b = lcm a b ∧ gcd b c = lcm b c ∧ gcd a c = lcm a c → a = b ∧ b = c :=
by
  sorry

end petya_vasya_three_numbers_equal_l1999_199939


namespace range_of_a_l1999_199981

theorem range_of_a (x y a : ℝ) (h1 : x - y = 2) (h2 : x + y = a) (h3 : x > -1) (h4 : y < 0) : -4 < a ∧ a < 2 :=
sorry

end range_of_a_l1999_199981


namespace point_line_real_assoc_l1999_199927

theorem point_line_real_assoc : 
  ∀ (p : ℝ), ∃! (r : ℝ), p = r := 
by 
  sorry

end point_line_real_assoc_l1999_199927


namespace solve_equation_1_solve_equation_2_l1999_199992

theorem solve_equation_1 (x : ℝ) (h : 0.5 * x + 1.1 = 6.5 - 1.3 * x) : x = 3 :=
  by sorry

theorem solve_equation_2 (x : ℝ) (h : (1 / 6) * (3 * x - 9) = (2 / 5) * x - 3) : x = -15 :=
  by sorry

end solve_equation_1_solve_equation_2_l1999_199992


namespace johns_previous_earnings_l1999_199926

theorem johns_previous_earnings (new_earnings raise_percentage old_earnings : ℝ) 
  (h1 : new_earnings = 68) (h2 : raise_percentage = 0.1333333333333334)
  (h3 : new_earnings = old_earnings * (1 + raise_percentage)) : old_earnings = 60 :=
sorry

end johns_previous_earnings_l1999_199926


namespace butterfat_in_final_mixture_l1999_199913

noncomputable def final_butterfat_percentage (gallons_of_35_percentage : ℕ) 
                                             (percentage_of_35_butterfat : ℝ) 
                                             (total_gallons : ℕ)
                                             (percentage_of_10_butterfat : ℝ) : ℝ :=
  let gallons_of_10 := total_gallons - gallons_of_35_percentage
  let butterfat_35 := gallons_of_35_percentage * percentage_of_35_butterfat
  let butterfat_10 := gallons_of_10 * percentage_of_10_butterfat
  let total_butterfat := butterfat_35 + butterfat_10
  (total_butterfat / total_gallons) * 100

theorem butterfat_in_final_mixture : 
  final_butterfat_percentage 8 0.35 12 0.10 = 26.67 :=
sorry

end butterfat_in_final_mixture_l1999_199913


namespace polynomial_value_sum_l1999_199968

theorem polynomial_value_sum
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (Hf : ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d)
  (H1 : f 1 = 1) (H2 : f 2 = 2) (H3 : f 3 = 3) :
  f 0 + f 4 = 28 :=
sorry

end polynomial_value_sum_l1999_199968


namespace average_of_middle_three_l1999_199909

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end average_of_middle_three_l1999_199909


namespace reciprocal_of_2_l1999_199929

theorem reciprocal_of_2 : 1 / 2 = 1 / (2 : ℝ) := by
  sorry

end reciprocal_of_2_l1999_199929


namespace min_value_of_reciprocals_l1999_199956

open Real

theorem min_value_of_reciprocals (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) :
  (1 / a) + (1 / (b + 1)) ≥ 2 :=
sorry

end min_value_of_reciprocals_l1999_199956


namespace g_2002_eq_1_l1999_199952

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ := λ x => f x + 1 - x)

axiom f_one : f 1 = 1
axiom f_inequality_1 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

theorem g_2002_eq_1 : g 2002 = 1 := by
  sorry

end g_2002_eq_1_l1999_199952


namespace largest_multiple_of_9_lt_120_is_117_l1999_199970

theorem largest_multiple_of_9_lt_120_is_117 : ∃ k : ℕ, 9 * k < 120 ∧ (∀ m : ℕ, 9 * m < 120 → 9 * m ≤ 9 * k) ∧ 9 * k = 117 := 
by 
  sorry

end largest_multiple_of_9_lt_120_is_117_l1999_199970


namespace repeating_decimal_conversion_l1999_199902

-- Definition of 0.\overline{23} as a rational number
def repeating_decimal_fraction : ℚ := 23 / 99

-- The main statement to prove
theorem repeating_decimal_conversion : (3 / 10) + (repeating_decimal_fraction) = 527 / 990 := 
by
  -- Placeholder for proof steps
  sorry

end repeating_decimal_conversion_l1999_199902


namespace find_c_of_triangle_area_l1999_199908

-- Define the problem in Lean 4 statement.
theorem find_c_of_triangle_area (A : ℝ) (b c : ℝ) (area : ℝ)
  (hA : A = 60 * Real.pi / 180)  -- Converting degrees to radians
  (hb : b = 1)
  (hArea : area = Real.sqrt 3) :
  c = 4 :=
by 
  -- Lean proof goes here (we include sorry to skip)
  sorry

end find_c_of_triangle_area_l1999_199908
