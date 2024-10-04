import Mathlib

namespace quadratic_form_proof_l72_72880

theorem quadratic_form_proof (k : ℝ) (a b c : ℝ) (h1 : 8*k^2 - 16*k + 28 = a * (k + b)^2 + c) (h2 : a = 8) (h3 : b = -1) (h4 : c = 20) : c / b = -20 :=
by {
  sorry
}

end quadratic_form_proof_l72_72880


namespace fraction_equality_l72_72038

theorem fraction_equality :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 :=
by
  sorry

end fraction_equality_l72_72038


namespace solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l72_72107

theorem solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1 :
  ∀ x : ℝ, 2 * x ^ 2 + 5 * x - 3 ≠ 0 ∧ 2 * x - 1 ≠ 0 → 
  (5 * x + 1) / (2 * x ^ 2 + 5 * x - 3) = (2 * x) / (2 * x - 1) → 
  x = -1 :=
by
  intro x h_cond h_eq
  sorry

end solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l72_72107


namespace flowers_per_vase_l72_72434

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end flowers_per_vase_l72_72434


namespace inequality_proof_l72_72221

variable {a b c : ℝ}
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)

theorem inequality_proof :
  (a + 3 * c) / (a + 2 * b + c) + 
  (4 * b) / (a + b + 2 * c) - 
  (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 := 
sorry

end inequality_proof_l72_72221


namespace compute_expression_l72_72025

open Real

theorem compute_expression : 
  sqrt (1 / 4) * sqrt 16 - (sqrt (1 / 9))⁻¹ - sqrt 0 + sqrt (45 / 5) = 2 := 
by
  -- The proof details would go here, but they are omitted.
  sorry

end compute_expression_l72_72025


namespace proof_equivalence_l72_72332

variable {x y : ℝ}

theorem proof_equivalence (h : x - y = 1) : x^3 - 3 * x * y - y^3 = 1 := by
  sorry

end proof_equivalence_l72_72332


namespace trigonometric_identity_l72_72338

theorem trigonometric_identity (α : Real) (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 9 / 5 :=
sorry

end trigonometric_identity_l72_72338


namespace parallelepiped_diagonal_l72_72391

theorem parallelepiped_diagonal 
  (x y z m n p d : ℝ)
  (h1 : x^2 + y^2 = m^2)
  (h2 : x^2 + z^2 = n^2)
  (h3 : y^2 + z^2 = p^2)
  : d = Real.sqrt ((m^2 + n^2 + p^2) / 2) := 
sorry

end parallelepiped_diagonal_l72_72391


namespace successful_experimental_operation_l72_72956

/-- Problem statement:
Given the following biological experimental operations:
1. spreading diluted E. coli culture on solid medium,
2. introducing sterile air into freshly inoculated grape juice with yeast,
3. inoculating soil leachate on beef extract peptone medium,
4. using slightly opened rose flowers as experimental material for anther culture.

Prove that spreading diluted E. coli culture on solid medium can successfully achieve the experimental objective of obtaining single colonies.
-/
theorem successful_experimental_operation :
  ∃ objective_result,
    (objective_result = "single_colonies" →
     let operation_A := "spreading diluted E. coli culture on solid medium"
     let operation_B := "introducing sterile air into freshly inoculated grape juice with yeast"
     let operation_C := "inoculating soil leachate on beef extract peptone medium"
     let operation_D := "slightly opened rose flowers as experimental material for anther culture"
     ∃ successful_operation,
       successful_operation = operation_A
       ∧ (successful_operation = operation_A → objective_result = "single_colonies")
       ∧ (successful_operation = operation_B → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_C → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_D → objective_result ≠ "single_colonies")) :=
sorry

end successful_experimental_operation_l72_72956


namespace initial_blueberry_jelly_beans_l72_72027

-- Definitions for initial numbers of jelly beans and modified quantities after eating
variables (b c : ℕ)

-- Conditions stated as Lean hypothesis
axiom initial_relation : b = 2 * c
axiom new_relation : b - 5 = 4 * (c - 5)

-- Theorem statement to prove the initial number of blueberry jelly beans is 30
theorem initial_blueberry_jelly_beans : b = 30 :=
by
  sorry

end initial_blueberry_jelly_beans_l72_72027


namespace contradiction_proof_l72_72879

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →
    false := 
by
  intros a b c d h1 h2 h3 h4
  sorry

end contradiction_proof_l72_72879


namespace no_solution_for_factorial_equation_l72_72495

theorem no_solution_for_factorial_equation :
  ∀ m : ℕ, 7! * 4! ≠ m! := by
  sorry

end no_solution_for_factorial_equation_l72_72495


namespace solve_fraction_problem_l72_72909

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l72_72909


namespace probability_sum_is_odd_given_product_is_even_dice_problem_l72_72623

def dice_rolls := Fin 6 → Fin 6

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ¬ is_even n

def sum_is_odd (rolls : dice_rolls) : Prop := 
  is_odd (∑ i, rolls i)

def product_is_even (rolls : dice_rolls) : Prop := 
  is_even (∏ i, rolls i)

def possible_rolls : Fin 7776 := 6^5

def valid_favorable_rolls : Fin 1443 := 5 * 3 * 2^4 + 10 * 3^3 * 2^2 + 3^5

theorem probability_sum_is_odd_given_product_is_even : 
  (num_favorable : ℚ) := valid_favorable_rolls / (possible_rolls - 3^5)

theorem dice_problem (rolls : dice_rolls) (h : product_is_even rolls) : 
  probability_sum_is_odd_given_product_is_even = 481/2511 := 
sorry

end probability_sum_is_odd_given_product_is_even_dice_problem_l72_72623


namespace son_l72_72783

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 24) 
  (h2 : M + 2 = 2 * (S + 2)) : S = 22 := 
by 
  sorry

end son_l72_72783


namespace min_value_of_a_l72_72248

theorem min_value_of_a (r s t : ℕ) (h1 : r > 0) (h2 : s > 0) (h3 : t > 0)
  (h4 : r * s * t = 2310) (h5 : r + s + t = a) : 
  a = 390 → True :=
by { 
  intros, 
  sorry 
}

end min_value_of_a_l72_72248


namespace smallest_of_consecutive_even_numbers_l72_72073

theorem smallest_of_consecutive_even_numbers (n : ℤ) (h : ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ c = 2 * n + 1) :
  ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ a = 2 * n - 3 :=
by
  sorry

end smallest_of_consecutive_even_numbers_l72_72073


namespace anticipated_sedans_l72_72801

theorem anticipated_sedans (sales_sports_cars sedans_ratio sports_ratio sports_forecast : ℕ) 
  (h_ratio : sports_ratio = 5) (h_sedans_ratio : sedans_ratio = 8) (h_sports_forecast : sports_forecast = 35)
  (h_eq : sales_sports_cars = sports_ratio * sports_forecast) :
  sales_sports_cars * 8 / 5 = 56 :=
by
  sorry

end anticipated_sedans_l72_72801


namespace problem_l72_72526

theorem problem (p q : Prop) (m : ℝ):
  (p = (m > 1)) →
  (q = (-2 ≤ m ∧ m ≤ 2)) →
  (¬q = (m < -2 ∨ m > 2)) →
  (¬(p ∧ q)) →
  (p ∨ q) →
  (¬q) →
  m > 2 :=
by
  sorry

end problem_l72_72526


namespace correct_value_of_a_l72_72373

namespace ProofProblem

-- Condition 1: Definition of set M
def M : Set ℤ := {x | x^2 ≤ 1}

-- Condition 2: Definition of set N dependent on a parameter a
def N (a : ℤ) : Set ℤ := {a, a * a}

-- Question translated: Correct value of a such that M ∪ N = M
theorem correct_value_of_a (a : ℤ) : (M ∪ N a = M) → a = -1 :=
by
  sorry

end ProofProblem

end correct_value_of_a_l72_72373


namespace interior_angle_of_regular_hexagon_l72_72771

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l72_72771


namespace only_integer_square_less_than_three_times_self_l72_72129

theorem only_integer_square_less_than_three_times_self :
  ∃! (x : ℤ), x^2 < 3 * x :=
by
  use 1
  split
  · -- Show that 1^2 < 3 * 1
    calc 1^2 = 1 : by norm_num
            ... < 3 : by norm_num
            ... = 3 * 1 : by norm_num
  · -- Show that for any x, if x^2 < 3 * x then x = 1
    intro y hy
    cases lt_or_ge y 1 with hy1 hy1
    · -- Case: y < 1
      exfalso
      calc y^2 ≥ 0 : by exact pow_two_nonneg y
              ... ≥ y * 3 - y : by linarith
              ...   = 3 * y - y : by ring
              ...   = 2 * y : by ring
      linarith
    cases lt_or_eq_of_le hy1 with hy1 hy1
    · -- Case: y = 2
      exfalso
      have h' := by linarith
      linarith
    · -- Case: y = 1
      exact hy1
    -- Case: y > 2
    exfalso
    calc y^2 ≥ y * 3 : by nlinarith
            ...   > y * 3 : by linarith
    linarith

end only_integer_square_less_than_three_times_self_l72_72129


namespace complex_division_result_l72_72185

theorem complex_division_result :
  let z := (⟨0, 1⟩ - ⟨2, 0⟩) / (⟨1, 0⟩ + ⟨0, 1⟩ : ℂ)
  let a := z.re
  let b := z.im
  a + b = 1 :=
by
  sorry

end complex_division_result_l72_72185


namespace g_range_l72_72669

noncomputable def g (x y z : ℝ) : ℝ := 
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_range (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 / 2 ≤ g x y z ∧ g x y z ≤ 2 :=
sorry

end g_range_l72_72669


namespace collinear_points_l72_72059

-- Define the given points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := {x := 1, y := 5, z := -2}
def B : Point := {x := 2, y := 4, z := 1}

-- Define C such that C(p, 3, q+2)
def C (p q : ℝ) : Point := {x := p, y := 3, z := q + 2}

-- Define the vectors AB and AC
def vector (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z}

def AB : Point := vector A B
def AC (p q : ℝ) : Point := vector A (C p q)

-- Define the proof problem that p = 3 and q = 2 if A, B, and C are collinear
theorem collinear_points (p q : ℝ) :
  (∃ λ : ℝ, AB.x = λ * (AC p q).x ∧ AB.y = λ * (AC p q).y ∧ AB.z = λ * (AC p q).z) → p = 3 ∧ q = 2 :=
by
  sorry

end collinear_points_l72_72059


namespace a_beats_b_by_7_seconds_l72_72507

/-
  Given:
  1. A's time to finish the race is 28 seconds (tA = 28).
  2. The race distance is 280 meters (d = 280).
  3. A beats B by 56 meters (dA - dB = 56).
  
  Prove:
  A beats B by 7 seconds (tB - tA = 7).
-/

theorem a_beats_b_by_7_seconds 
  (tA : ℕ) (d : ℕ) (speedA : ℕ) (dB : ℕ) (tB : ℕ) 
  (h1 : tA = 28) 
  (h2 : d = 280) 
  (h3 : d - dB = 56) 
  (h4 : speedA = d / tA) 
  (h5 : dB = speedA * tA) 
  (h6 : tB = d / speedA) :
  tB - tA = 7 := 
sorry

end a_beats_b_by_7_seconds_l72_72507


namespace rainfall_difference_l72_72563

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l72_72563


namespace percentage_female_on_duty_l72_72376

-- Definition of conditions
def on_duty_officers : ℕ := 152
def female_on_duty : ℕ := on_duty_officers / 2
def total_female_officers : ℕ := 400

-- Proof goal
theorem percentage_female_on_duty : (female_on_duty * 100) / total_female_officers = 19 := by
  -- We would complete the proof here
  sorry

end percentage_female_on_duty_l72_72376


namespace a9_value_l72_72824

theorem a9_value (a : ℕ → ℝ) (x : ℝ) (h : (1 + x) ^ 10 = 
  (a 0) + (a 1) * (1 - x) + (a 2) * (1 - x)^2 + 
  (a 3) * (1 - x)^3 + (a 4) * (1 - x)^4 + 
  (a 5) * (1 - x)^5 + (a 6) * (1 - x)^6 + 
  (a 7) * (1 - x)^7 + (a 8) * (1 - x)^8 + 
  (a 9) * (1 - x)^9 + (a 10) * (1 - x)^10) : 
  a 9 = -20 :=
sorry

end a9_value_l72_72824


namespace schedule_lectures_correct_l72_72595

noncomputable def ways_to_schedule_lectures : ℕ :=
  let lecturers := ["Dr. Jones", "Dr. Smith", "Dr. Allen", "L4", "L5", "L6"]
  let permutations := Multiset.permute lecturers
  (permutations.count (λ p : List String, p.indexOf "Dr. Jones" < p.indexOf "Dr. Smith" ∧ p.indexOf "Dr. Smith" < p.indexOf "Dr. Allen"))

theorem schedule_lectures_correct : ways_to_schedule_lectures = 228 := by
  sorry

end schedule_lectures_correct_l72_72595


namespace triangle_is_obtuse_l72_72732

-- Define the conditions of the problem
def angles (x : ℝ) : Prop :=
  2 * x + 3 * x + 6 * x = 180

def obtuse_angle (x : ℝ) : Prop :=
  6 * x > 90

-- State the theorem
theorem triangle_is_obtuse (x : ℝ) (hx : angles x) : obtuse_angle x :=
sorry

end triangle_is_obtuse_l72_72732


namespace add_to_frac_eq_l72_72921

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l72_72921


namespace distinct_total_prices_count_l72_72122

open Finset

def gift_prices : Finset ℕ := {2, 5, 8, 11, 14}
def box_prices : Finset ℕ := {3, 5, 7, 9, 11}

theorem distinct_total_prices_count : 
  (gift_prices.product box_prices).image (λ p => p.1 + p.2)).card = 19 :=
by
  sorry

end distinct_total_prices_count_l72_72122


namespace geometric_sum_3030_l72_72900

theorem geometric_sum_3030 {a r : ℝ}
  (h1 : a * (1 - r ^ 1010) / (1 - r) = 300)
  (h2 : a * (1 - r ^ 2020) / (1 - r) = 540) :
  a * (1 - r ^ 3030) / (1 - r) = 732 :=
sorry

end geometric_sum_3030_l72_72900


namespace negation_equivalence_l72_72723

-- Define the propositions
def proposition (a b : ℝ) : Prop := a > b → a + 1 > b

def negation_proposition (a b : ℝ) : Prop := a ≤ b → a + 1 ≤ b

-- Statement to prove
theorem negation_equivalence (a b : ℝ) : ¬(proposition a b) ↔ negation_proposition a b := 
sorry

end negation_equivalence_l72_72723


namespace find_factor_l72_72598

variable (x : ℕ) (f : ℕ)

def original_number := x = 20
def resultant := f * (2 * x + 5) = 135

theorem find_factor (h1 : original_number x) (h2 : resultant x f) : f = 3 := by
  sorry

end find_factor_l72_72598


namespace sequence_of_arrows_l72_72003

theorem sequence_of_arrows (n : ℕ) (h : n % 5 = 0) : 
  (n < 570 ∧ n % 5 = 0) → 
  (n + 1 < 573 ∧ (n + 1) % 5 = 1) → 
  (n + 2 < 573 ∧ (n + 2) % 5 = 2) → 
  (n + 3 < 573 ∧ (n + 3) % 5 = 3) →
    true :=
by
  sorry

end sequence_of_arrows_l72_72003


namespace line_passes_through_fixed_point_l72_72245

-- Given a line equation kx - y + 1 - 3k = 0
def line_equation (k x y : ℝ) : Prop := k * x - y + 1 - 3 * k = 0

-- We need to prove that this line passes through the point (3,1)
theorem line_passes_through_fixed_point (k : ℝ) : line_equation k 3 1 :=
by
  sorry

end line_passes_through_fixed_point_l72_72245


namespace oranges_taken_from_basket_l72_72401

-- Define the original number of oranges and the number left after taking some out.
def original_oranges : ℕ := 8
def oranges_left : ℕ := 3

-- Prove that the number of oranges taken from the basket equals 5.
theorem oranges_taken_from_basket : original_oranges - oranges_left = 5 := by
  sorry

end oranges_taken_from_basket_l72_72401


namespace friends_for_picnic_only_l72_72784

theorem friends_for_picnic_only (M MP MG G PG A P : ℕ) 
(h1 : M + MP + MG + A = 10)
(h2 : G + MG + A = 5)
(h3 : MP = 4)
(h4 : MG = 2)
(h5 : PG = 0)
(h6 : A = 2)
(h7 : M + P + G + MP + MG + PG + A = 31) : 
    P = 20 := by {
  sorry
}

end friends_for_picnic_only_l72_72784


namespace table_seating_problem_l72_72700

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l72_72700


namespace people_at_table_l72_72695

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l72_72695


namespace quadratic_has_two_distinct_real_roots_l72_72550

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l72_72550


namespace right_triangle_exists_with_area_ab_l72_72063

theorem right_triangle_exists_with_area_ab (a b c d : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
    (h1 : a * b = c * d) (h2 : a + b = c - d) :
    ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ (x * y / 2 = a * b) := sorry

end right_triangle_exists_with_area_ab_l72_72063


namespace area_diff_l72_72005

-- Defining the side lengths of squares
def side_length_small_square : ℕ := 4
def side_length_large_square : ℕ := 10

-- Calculating the areas
def area_small_square : ℕ := side_length_small_square ^ 2
def area_large_square : ℕ := side_length_large_square ^ 2

-- Theorem statement
theorem area_diff (a_small a_large : ℕ) (h1 : a_small = side_length_small_square ^ 2) (h2 : a_large = side_length_large_square ^ 2) : 
  a_large - a_small = 84 :=
by
  sorry

end area_diff_l72_72005


namespace total_people_seated_l72_72690

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l72_72690


namespace sum_of_factors_of_30_is_72_l72_72775

-- Condition: given the number 30
def number := 30

-- Define the positive factors of 30
def factors : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- Statement to prove the sum of the positive factors
theorem sum_of_factors_of_30_is_72 : (factors.sum) = 72 := 
by
  sorry

end sum_of_factors_of_30_is_72_l72_72775


namespace product_of_slopes_hyperbola_l72_72096

theorem product_of_slopes_hyperbola (a b x0 y0 : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : (x0, y0) ≠ (-a, 0)) (h4 : (x0, y0) ≠ (a, 0)) 
(h5 : x0^2 / a^2 - y0^2 / b^2 = 1) : 
(y0 / (x0 + a) * (y0 / (x0 - a)) = b^2 / a^2) :=
sorry

end product_of_slopes_hyperbola_l72_72096


namespace product_of_solutions_eq_zero_l72_72727

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end product_of_solutions_eq_zero_l72_72727


namespace algorithm_must_have_sequential_structure_l72_72717

-- Definitions for types of structures used in algorithm definitions.
inductive Structure
| Logical
| Selection
| Loop
| Sequential

-- Predicate indicating whether a given Structure is necessary for any algorithm.
def necessary (s : Structure) : Prop :=
  match s with
  | Structure.Logical => False
  | Structure.Selection => False
  | Structure.Loop => False
  | Structure.Sequential => True

-- The theorem statement to prove that the sequential structure is necessary for any algorithm.
theorem algorithm_must_have_sequential_structure :
  necessary Structure.Sequential :=
by
  sorry

end algorithm_must_have_sequential_structure_l72_72717


namespace who_is_first_l72_72958

def positions (A B C D : ℕ) : Prop :=
  A + B + D = 6 ∧ B + C = 6 ∧ B < A ∧ A + B + C + D = 10

theorem who_is_first (A B C D : ℕ) (h : positions A B C D) : D = 1 :=
sorry

end who_is_first_l72_72958


namespace sqrt_sum_ge_two_l72_72218

theorem sqrt_sum_ge_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 := 
by
  sorry

end sqrt_sum_ge_two_l72_72218


namespace sum_of_digits_of_special_two_digit_number_l72_72735

theorem sum_of_digits_of_special_two_digit_number (x : ℕ) (h1 : 1 ≤ x ∧ x < 10) 
  (h2 : ∃ (n : ℕ), n = 11 * x + 30) 
  (h3 : ∃ (sum_digits : ℕ), sum_digits = (x + 3) + x) 
  (h4 : (11 * x + 30) % ((x + 3) + x) = 3)
  (h5 : (11 * x + 30) / ((x + 3) + x) = 7) :
  (x + 3) + x = 7 := 
by 
  sorry

end sum_of_digits_of_special_two_digit_number_l72_72735


namespace flowers_per_vase_l72_72433

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end flowers_per_vase_l72_72433


namespace arts_school_probability_l72_72202

theorem arts_school_probability :
  let cultural_courses := 3
  let arts_courses := 3
  let total_periods := 6
  let total_arrangements := Nat.factorial total_periods
  let no_adjacent_more_than_one_separator := (72 + 216 + 144)
  (no_adjacent_more_than_one_separator : ℝ) / (total_arrangements : ℝ) = (3 / 5 : ℝ) := 
by 
  sorry

end arts_school_probability_l72_72202


namespace second_shipment_is_13_l72_72002

-- Definitions based on the conditions
def first_shipment : ℕ := 7
def third_shipment : ℕ := 45
def total_couscous_used : ℕ := 13 * 5 -- 65
def total_couscous_from_three_shipments (second_shipment : ℕ) : ℕ :=
  first_shipment + second_shipment + third_shipment

-- Statement of the proof problem corresponding to the conditions and question
theorem second_shipment_is_13 (x : ℕ) 
  (h : total_couscous_used = total_couscous_from_three_shipments x) : x = 13 := 
by
  sorry

end second_shipment_is_13_l72_72002


namespace perfect_square_m_value_l72_72501

theorem perfect_square_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ x : ℝ, (x^2 + (m : ℝ)*x + 1 : ℝ) = (x + (a : ℝ))^2) → m = 2 ∨ m = -2 :=
by
  sorry

end perfect_square_m_value_l72_72501


namespace solve_equation_l72_72534

theorem solve_equation : ∃ x : ℚ, (2*x + 1) / 4 - 1 = x - (10*x + 1) / 12 ∧ x = 5 / 2 :=
by
  sorry

end solve_equation_l72_72534


namespace binders_can_bind_books_l72_72195

theorem binders_can_bind_books :
  (∀ (binders books days : ℕ), binders * days * books = 18 * 10 * 900 → 
    11 * binders * 12 = 660) :=
sorry

end binders_can_bind_books_l72_72195


namespace min_value_of_expression_l72_72467

theorem min_value_of_expression : ∀ x : ℝ, ∃ (M : ℝ), (∀ x, 16^x - 4^x - 4^(x+1) + 3 ≥ M) ∧ M = -4 :=
by
  sorry

end min_value_of_expression_l72_72467


namespace probability_sum_is_odd_given_product_is_even_dice_problem_l72_72624

def dice_rolls := Fin 6 → Fin 6

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ¬ is_even n

def sum_is_odd (rolls : dice_rolls) : Prop := 
  is_odd (∑ i, rolls i)

def product_is_even (rolls : dice_rolls) : Prop := 
  is_even (∏ i, rolls i)

def possible_rolls : Fin 7776 := 6^5

def valid_favorable_rolls : Fin 1443 := 5 * 3 * 2^4 + 10 * 3^3 * 2^2 + 3^5

theorem probability_sum_is_odd_given_product_is_even : 
  (num_favorable : ℚ) := valid_favorable_rolls / (possible_rolls - 3^5)

theorem dice_problem (rolls : dice_rolls) (h : product_is_even rolls) : 
  probability_sum_is_odd_given_product_is_even = 481/2511 := 
sorry

end probability_sum_is_odd_given_product_is_even_dice_problem_l72_72624


namespace smallest_abcd_value_l72_72411

theorem smallest_abcd_value (A B C D : ℕ) (h1 : A ≠ B) (h2 : 1 ≤ A) (h3 : A ≤ 9) (h4 : 0 ≤ B) 
                            (h5 : B ≤ 9) (h6 : 1 ≤ C) (h7 : C ≤ 9) (h8 : 1 ≤ D) (h9 : D ≤ 9)
                            (h10 : 10 * A * A + A * B = 1000 * A + 100 * B + 10 * C + D)
                            (h11 : A ≠ C) (h12 : A ≠ D) (h13 : B ≠ C) (h14 : B ≠ D) (h15 : C ≠ D) :
  1000 * A + 100 * B + 10 * C + D = 2046 :=
sorry

end smallest_abcd_value_l72_72411


namespace total_drums_l72_72868

theorem total_drums (x y : ℕ) (hx : 30 * x + 20 * y = 160) : x + y = 7 :=
sorry

end total_drums_l72_72868


namespace evaluate_expression_l72_72036

variables (a b c : ℝ)

theorem evaluate_expression (h1 : c = b - 20) (h2 : b = a + 4) (h3 : a = 2)
  (h4 : a^2 + a ≠ 0) (h5 : b^2 - 6 * b + 8 ≠ 0) (h6 : c^2 + 12 * c + 36 ≠ 0):
  (a^2 + 2 * a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6 * b + 8) * (c^2 + 16 * c + 64) / (c^2 + 12 * c + 36) = 3 / 4 :=
by sorry

end evaluate_expression_l72_72036


namespace factor_expression_l72_72457

variable (a : ℤ)

theorem factor_expression : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end factor_expression_l72_72457


namespace balance_proof_l72_72817

variable (a b c : ℕ)

theorem balance_proof (h1 : 5 * a + 2 * b = 15 * c) (h2 : 2 * a = b + 3 * c) : 4 * b = 7 * c :=
sorry

end balance_proof_l72_72817


namespace boys_to_girls_ratio_l72_72273

theorem boys_to_girls_ratio (S G B : ℕ) (h1 : 1 / 2 * G = 1 / 3 * S) (h2 : S = B + G) : B / G = 1 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end boys_to_girls_ratio_l72_72273


namespace calculate_v_sum_l72_72446

def v (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem calculate_v_sum :
  v (2) + v (-2) + v (1) + v (-1) = 4 :=
by
  sorry

end calculate_v_sum_l72_72446


namespace geometric_progression_general_term_l72_72653

noncomputable def a_n (n : ℕ) : ℝ := 2^(n-1)

theorem geometric_progression_general_term :
  (∀ n : ℕ, n ≥ 1 → a_n n > 0) ∧
  a_n 1 = 1 ∧
  a_n 2 + a_n 3 = 6 →
  ∀ n, a_n n = 2^(n-1) :=
by
  intros h
  sorry

end geometric_progression_general_term_l72_72653


namespace rainfall_difference_l72_72561

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l72_72561


namespace solve_for_y_l72_72712

noncomputable def log5 (x : ℝ) : ℝ := (Real.log x) / (Real.log 5)

theorem solve_for_y (y : ℝ) (h₀ : log5 ((2 * y + 10) / (3 * y - 6)) + log5 ((3 * y - 6) / (y - 4)) = 3) : 
  y = 170 / 41 :=
sorry

end solve_for_y_l72_72712


namespace marissa_lunch_calories_l72_72093

theorem marissa_lunch_calories :
  (1 * 400) + (5 * 20) + (5 * 50) = 750 :=
by
  sorry

end marissa_lunch_calories_l72_72093


namespace max_n_for_positive_sum_l72_72841

-- Define the arithmetic sequence \(a_n\)
def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (a d : ℤ) (n : ℕ) := n * (2 * a + (n-1) * d) / 2

theorem max_n_for_positive_sum 
  (a : ℤ) 
  (d : ℤ) 
  (h_max_sum : ∃ m : ℕ, S_n a d m = S_n a d (m+1))
  (h_ratio : (arithmetic_sequence a d 15) / (arithmetic_sequence a d 14) < -1) :
  27 = 27 :=
sorry

end max_n_for_positive_sum_l72_72841


namespace remainder_when_divided_by_6_l72_72131

theorem remainder_when_divided_by_6 (n : ℕ) (h₁ : n = 482157)
  (odd_n : n % 2 ≠ 0) (div_by_3 : n % 3 = 0) : n % 6 = 3 :=
by
  -- Proof goes here
  sorry

end remainder_when_divided_by_6_l72_72131


namespace least_number_to_add_l72_72584

theorem least_number_to_add {n : ℕ} (h : n = 1202) : (∃ k : ℕ, (n + k) % 4 = 0 ∧ ∀ m : ℕ, (m < k → (n + m) % 4 ≠ 0)) ∧ k = 2 := by
  sorry

end least_number_to_add_l72_72584


namespace initial_wage_of_illiterate_l72_72655

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end initial_wage_of_illiterate_l72_72655


namespace find_principal_sum_l72_72114

theorem find_principal_sum (P R : ℝ) (SI CI : ℝ) 
  (h1 : SI = 10200) 
  (h2 : CI = 11730) 
  (h3 : SI = P * R * 2 / 100)
  (h4 : CI = P * (1 + R / 100)^2 - P) :
  P = 17000 :=
by
  sorry

end find_principal_sum_l72_72114


namespace paintings_correct_l72_72660

def octagon_paintings : Prop :=
  let num_disks := 8
  let disks := Fin num_disks
  let colors := {blue := 4, red := 2, green := 2}
  let symmetries := -- Represent rotations and reflections of the octagon
    [(0 : Int), 45, 90, 135, 180, 225, 270, 315, -- rotations
     "ref_v1", "ref_v2", "ref_v3", "ref_v4", -- reflections through vertices
     "ref_m1", "ref_m2", "ref_m3", "ref_m4"] -- reflections through midpoints
  ∀ (paintings : {arrangement : disks → Fin 3 // 
                    arrangement.parity colors -- Each color used the required number of times
                    }), 
    let count_fixed := Burnside.fixed_count symmetries paintings
    count_fixed = 34

theorem paintings_correct : octagon_paintings :=
  sorry

end paintings_correct_l72_72660


namespace table_seating_problem_l72_72704

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l72_72704


namespace last_two_digits_of_7_pow_2015_l72_72870

theorem last_two_digits_of_7_pow_2015 : ((7 ^ 2015) % 100) = 43 := 
by
  sorry

end last_two_digits_of_7_pow_2015_l72_72870


namespace does_not_pass_first_quadrant_l72_72892

def linear_function (x : ℝ) : ℝ := -3 * x - 2

def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem does_not_pass_first_quadrant : ∀ (x : ℝ), ¬ in_first_quadrant x (linear_function x) := 
sorry

end does_not_pass_first_quadrant_l72_72892


namespace remaining_bottle_caps_l72_72159

-- Definitions based on conditions
def initial_bottle_caps : ℕ := 65
def eaten_bottle_caps : ℕ := 4

-- Theorem
theorem remaining_bottle_caps : initial_bottle_caps - eaten_bottle_caps = 61 :=
by
  sorry

end remaining_bottle_caps_l72_72159


namespace discount_savings_l72_72215

theorem discount_savings (initial_price discounted_price : ℝ)
  (h_initial : initial_price = 475)
  (h_discounted : discounted_price = 199) :
  initial_price - discounted_price = 276 :=
by
  rw [h_initial, h_discounted]
  sorry

end discount_savings_l72_72215


namespace product_ab_zero_l72_72483

variable {a b : ℝ}

theorem product_ab_zero (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
  sorry

end product_ab_zero_l72_72483


namespace busy_squirrels_count_l72_72865

variable (B : ℕ)
variable (busy_squirrel_nuts_per_day : ℕ := 30)
variable (sleepy_squirrel_nuts_per_day : ℕ := 20)
variable (days : ℕ := 40)
variable (total_nuts : ℕ := 3200)

theorem busy_squirrels_count : busy_squirrel_nuts_per_day * days * B + sleepy_squirrel_nuts_per_day * days = total_nuts → B = 2 := by
  sorry

end busy_squirrels_count_l72_72865


namespace box_volume_max_l72_72197

noncomputable def volume (a x : ℝ) : ℝ :=
  (a - 2 * x) ^ 2 * x

theorem box_volume_max (a : ℝ) (h : 0 < a) :
  ∃ x, 0 < x ∧ x < a / 2 ∧ volume a x = volume a (a / 6) ∧ volume a (a / 6) = (2 * a^3) / 27 :=
by
  sorry

end box_volume_max_l72_72197


namespace largest_angle_in_pentagon_l72_72514

theorem largest_angle_in_pentagon (P Q R S T : ℝ) 
          (h1 : P = 70) 
          (h2 : Q = 100)
          (h3 : R = S) 
          (h4 : T = 3 * R - 25)
          (h5 : P + Q + R + S + T = 540) : 
          T = 212 :=
by
  sorry

end largest_angle_in_pentagon_l72_72514


namespace determine_min_bottles_l72_72285

-- Define the capacities and constraints
def mediumBottleCapacity : ℕ := 80
def largeBottleCapacity : ℕ := 1200
def additionalBottles : ℕ := 5

-- Define the minimum number of medium-sized bottles Jasmine needs to buy
def minimumMediumBottles (mediumCapacity largeCapacity extras : ℕ) : ℕ :=
  let requiredBottles := largeCapacity / mediumCapacity
  requiredBottles

theorem determine_min_bottles :
  minimumMediumBottles mediumBottleCapacity largeBottleCapacity additionalBottles = 15 :=
by
  sorry

end determine_min_bottles_l72_72285


namespace value_of_nested_fraction_l72_72812

theorem value_of_nested_fraction :
  10 + 5 + (1 / 2) * (9 + 5 + (1 / 2) * (8 + 5 + (1 / 2) * (7 + 5 + (1 / 2) * (6 + 5 + (1 / 2) * (5 + 5 + (1 / 2) * (4 + 5 + (1 / 2) * (3 + 5 ))))))) = 28 + (1 / 128) :=
sorry

end value_of_nested_fraction_l72_72812


namespace price_of_each_pizza_l72_72046

variable (P : ℝ)

theorem price_of_each_pizza (h1 : 4 * P + 5 = 45) : P = 10 := by
  sorry

end price_of_each_pizza_l72_72046


namespace range_of_a_l72_72827

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then 2^x + 1 else -x^2 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a < 3) ↔ (2 ≤ a ∧ a < 2 * Real.sqrt 3) := by
  sorry

end range_of_a_l72_72827


namespace simplify_expression_find_value_a_m_2n_l72_72444

-- Proof Problem 1
theorem simplify_expression : ( (-2 : ℤ) * x )^3 * x^2 + ( (3 : ℤ) * x^4 )^2 / x^3 = x^5 := by
  sorry

-- Proof Problem 2
theorem find_value_a_m_2n (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2*n) = 18 := by
  sorry

end simplify_expression_find_value_a_m_2n_l72_72444


namespace base7_of_2345_l72_72809

def decimal_to_base7 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 6 * 7^1 + 0 * 7^0

theorem base7_of_2345 : decimal_to_base7 2345 = 6560 := by
  sorry

end base7_of_2345_l72_72809


namespace num_ways_to_have_5_consecutive_empty_seats_l72_72076

theorem num_ways_to_have_5_consecutive_empty_seats :
  let n := 10
  let k := 4
  let m := 5
  ( ∃ S : set ℕ, S.card = k ∧ 
    0 ≤ min S ∧ max S < n ∧ 
    ∃ I : finset ℕ, I.card = m ∧ 
      disjoint I S ∧ I = (finset.range n).erase' finset.card ) → 
  ∃ N : ℕ, N = 480 := by
  sorry

end num_ways_to_have_5_consecutive_empty_seats_l72_72076


namespace scientific_notation_570_million_l72_72538

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end scientific_notation_570_million_l72_72538


namespace triangle_to_square_difference_l72_72019

noncomputable def number_of_balls_in_triangle (T : ℕ) : ℕ :=
  T * (T + 1) / 2

noncomputable def number_of_balls_in_square (S : ℕ) : ℕ :=
  S * S

theorem triangle_to_square_difference (T S : ℕ) 
  (h1 : number_of_balls_in_triangle T = 1176) 
  (h2 : number_of_balls_in_square S = 1600) :
  T - S = 8 :=
by
  sorry

end triangle_to_square_difference_l72_72019


namespace union_of_sets_l72_72482

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end union_of_sets_l72_72482


namespace evaluate_expression_l72_72613

variable {c d : ℝ}

theorem evaluate_expression (h : c ≠ d ∧ c ≠ -d) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 :=
by sorry

end evaluate_expression_l72_72613


namespace discount_rate_on_pony_jeans_is_15_l72_72582

noncomputable def discountProblem : Prop :=
  ∃ (F P : ℝ),
    (15 * 3 * F / 100 + 18 * 2 * P / 100 = 8.55) ∧ 
    (F + P = 22) ∧ 
    (P = 15)

theorem discount_rate_on_pony_jeans_is_15 : discountProblem :=
sorry

end discount_rate_on_pony_jeans_is_15_l72_72582


namespace find_XY_squared_l72_72522

variables {A B C T X Y : Type}

-- Conditions
variables (is_acute_scalene_triangle : ∀ A B C : Type, Prop) -- Assume scalene and acute properties
variable  (circumcircle : ∀ A B C : Type, Type) -- Circumcircle of the triangle
variable  (tangent_at : ∀ (ω : Type) B C, Type) -- Tangents at B and C
variables (BT CT : ℝ)
variables (BC : ℝ)
variables (projections : ∀ T (line : Type), Type)
variables (TX TY XY : ℝ)

-- Given conditions
axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom final_equation : TX^2 + TY^2 + XY^2 = 1552

-- Goal
theorem find_XY_squared : XY^2 = 884 := by
  sorry

end find_XY_squared_l72_72522


namespace find_y_l72_72896

noncomputable def inverse_proportion_y_value (x y k : ℝ) : Prop :=
  (x * y = k) ∧ (x + y = 52) ∧ (x = 3 * y) ∧ (x = -10) → (y = -50.7)

theorem find_y (x y k : ℝ) (h : inverse_proportion_y_value x y k) : y = -50.7 :=
  sorry

end find_y_l72_72896


namespace worker_savings_l72_72437

theorem worker_savings (P : ℝ) (f : ℝ) (h : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  have h1 : 12 * f * P = 4 * (1 - f) * P := h
  have h2 : P ≠ 0 := sorry  -- P should not be 0 for the worker to have a meaningful income.
  field_simp [h2] at h1
  linarith

end worker_savings_l72_72437


namespace smallest_n_is_589_l72_72715

noncomputable def smallest_n (a b c : ℕ) : ℕ :=
  let f : ℕ → ℕ := λ x, x / 5 + x / 25 + x / 125 + x / 625 + x / 3125 + x / 15625 + x / 78125
  in 2 * f(a) + f(2 * a) + f(c)

theorem smallest_n_is_589 (a b c : ℕ) (h1 : a + b + c = 2010) (h2 : b = 2 * a) :
  smallest_n a b (2010 - 3 * a) = 589 := sorry

end smallest_n_is_589_l72_72715


namespace Carrie_hourly_wage_l72_72806

theorem Carrie_hourly_wage (hours_per_week : ℕ) (weeks_per_month : ℕ) (cost_bike : ℕ) (remaining_money : ℕ)
  (total_hours : ℕ) (total_savings : ℕ) (x : ℕ) :
  hours_per_week = 35 → 
  weeks_per_month = 4 → 
  cost_bike = 400 → 
  remaining_money = 720 → 
  total_hours = hours_per_week * weeks_per_month → 
  total_savings = cost_bike + remaining_money → 
  total_savings = total_hours * x → 
  x = 8 :=
by 
  intros h_hw h_wm h_cb h_rm h_th h_ts h_tx
  sorry

end Carrie_hourly_wage_l72_72806


namespace parrots_are_red_l72_72224

-- Definitions for fractions.
def total_parrots : ℕ := 160
def green_fraction : ℚ := 5 / 8
def blue_fraction : ℚ := 1 / 4

-- Definition for calculating the number of parrots.
def number_of_green_parrots : ℚ := green_fraction * total_parrots
def number_of_blue_parrots : ℚ := blue_fraction * total_parrots
def number_of_red_parrots : ℚ := total_parrots - number_of_green_parrots - number_of_blue_parrots

-- The theorem to prove.
theorem parrots_are_red : number_of_red_parrots = 20 := by
  -- Proof is omitted.
  sorry

end parrots_are_red_l72_72224


namespace geometric_sequence_seventh_term_l72_72175

variable {G : Type*} [Field G]

def is_geometric (a : ℕ → G) (q : G) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → G) (q : G)
  (h1 : a 0 + a 1 = 3)
  (h2 : a 1 + a 2 = 6)
  (hq : is_geometric a q) :
  a 6 = 64 := 
sorry

end geometric_sequence_seventh_term_l72_72175


namespace remainder_mul_three_division_l72_72948

theorem remainder_mul_three_division
    (N : ℤ) (k : ℤ)
    (h1 : N = 1927 * k + 131) :
    ((3 * N) % 43) = 6 :=
by
  sorry

end remainder_mul_three_division_l72_72948


namespace fireflies_remaining_l72_72876

theorem fireflies_remaining
  (initial_fireflies : ℕ)
  (fireflies_joined : ℕ)
  (fireflies_flew_away : ℕ)
  (h_initial : initial_fireflies = 3)
  (h_joined : fireflies_joined = 12 - 4)
  (h_flew_away : fireflies_flew_away = 2)
  : initial_fireflies + fireflies_joined - fireflies_flew_away = 9 := by
  sorry

end fireflies_remaining_l72_72876


namespace surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l72_72092

-- Given conditions
def income_per_day : List Int := [65, 68, 50, 66, 50, 75, 74]
def expenditure_per_day : List Int := [-60, -64, -63, -58, -60, -64, -65]

-- Part 1: Proving the surplus by the end of the week is 14 yuan
theorem surplus_by_end_of_week_is_14 :
  List.sum income_per_day + List.sum expenditure_per_day = 14 :=
by
  sorry

-- Part 2: Proving the estimated income needed per month to maintain normal expenses is 1860 yuan
theorem estimated_monthly_income_is_1860 :
  (List.sum (List.map Int.natAbs expenditure_per_day) / 7) * 30 = 1860 :=
by
  sorry

end surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l72_72092


namespace product_of_integers_l72_72117

theorem product_of_integers (x y : ℕ) (h_gcd : Nat.gcd x y = 10) (h_lcm : Nat.lcm x y = 60) : x * y = 600 := by
  sorry

end product_of_integers_l72_72117


namespace subtraction_of_tenths_l72_72418

theorem subtraction_of_tenths (a b : ℝ) (n : ℕ) (h1 : a = (1 / 10) * 6000) (h2 : b = (1 / 10 / 100) * 6000) : (a - b) = 594 := by
sorry

end subtraction_of_tenths_l72_72418


namespace sum_first_110_terms_l72_72834

noncomputable def sum_arithmetic (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem sum_first_110_terms (a1 d : ℚ) (h1 : sum_arithmetic a1 d 10 = 100)
  (h2 : sum_arithmetic a1 d 100 = 10) : sum_arithmetic a1 d 110 = -110 := by
  sorry

end sum_first_110_terms_l72_72834


namespace find_rate_of_current_l72_72252

-- Define the conditions
def speed_in_still_water (speed : ℝ) : Prop := speed = 15
def distance_downstream (distance : ℝ) : Prop := distance = 7.2
def time_in_hours (time : ℝ) : Prop := time = 0.4

-- Define the effective speed downstream
def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ := boat_speed + current_speed

-- Define rate of current
def rate_of_current (current_speed : ℝ) : Prop :=
  ∃ (c : ℝ), effective_speed_downstream 15 c * 0.4 = 7.2 ∧ c = current_speed

-- The theorem stating the proof problem
theorem find_rate_of_current : rate_of_current 3 :=
by
  sorry

end find_rate_of_current_l72_72252


namespace simplify_and_evaluate_l72_72106

noncomputable section

def x := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  (x / (x^2 - 1) / (1 - (1 / (x + 1)))) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l72_72106


namespace add_to_frac_eq_l72_72923

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l72_72923


namespace proper_subset_f_l72_72860

noncomputable def h (x : ℚ) : ℕ := sorry  -- bijection from ℚ to ℕ 

def f (a : ℝ) : Set ℕ := 
  {n : ℕ | ∃ (x : ℚ), x < a ∧ h x = n}

theorem proper_subset_f {a b : ℝ} (h₀ : a < b) : f a ⊂ f b :=
by {
  -- Proof would go here
  sorry
}

end proper_subset_f_l72_72860


namespace ratio_of_milk_and_water_l72_72125

theorem ratio_of_milk_and_water (x y : ℝ) (hx : 9 * x = 9 * y) : 
  let total_milk := (7 * x + 8 * y)
  let total_water := (2 * x + y)
  (total_milk / total_water) = 5 :=
by
  sorry

end ratio_of_milk_and_water_l72_72125


namespace problem_difference_l72_72101

-- Define the sum of first n natural numbers
def sumFirstN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Define the rounding rule to the nearest multiple of 5
def roundToNearest5 (x : ℕ) : ℕ :=
  match x % 5 with
  | 0 => x
  | 1 => x - 1
  | 2 => x - 2
  | 3 => x + 2
  | 4 => x + 1
  | _ => x  -- This case is theoretically unreachable

-- Define the sum of the first n natural numbers after rounding to nearest 5
def sumRoundedFirstN (n : ℕ) : ℕ :=
  (List.range (n + 1)).map roundToNearest5 |>.sum

theorem problem_difference : sumFirstN 120 - sumRoundedFirstN 120 = 6900 := by
  sorry

end problem_difference_l72_72101


namespace sum_first_70_odd_eq_4900_l72_72275

theorem sum_first_70_odd_eq_4900 (h : (70 * (70 + 1) = 4970)) :
  (70 * 70 = 4900) :=
by
  sorry

end sum_first_70_odd_eq_4900_l72_72275


namespace sum_at_simple_interest_l72_72951

theorem sum_at_simple_interest (P R : ℝ) (h1: ((3 * P * (R + 1))/ 100) = ((3 * P * R) / 100 + 72)) : P = 2400 := 
by 
  sorry

end sum_at_simple_interest_l72_72951


namespace graph_is_two_lines_l72_72450

theorem graph_is_two_lines (x y : ℝ) : (x^2 - 25 * y^2 - 10 * x + 50 = 0) ↔
  (x = 5 + 5 * y) ∨ (x = 5 - 5 * y) :=
by
  sorry

end graph_is_two_lines_l72_72450


namespace investment_Q_correct_l72_72377

-- Define the investments of P and Q
def investment_P : ℝ := 40000
def investment_Q : ℝ := 60000

-- Define the profit share ratio
def profit_ratio_PQ : ℝ × ℝ := (2, 3)

-- State the theorem to prove
theorem investment_Q_correct :
  (investment_P / investment_Q = (profit_ratio_PQ.1 / profit_ratio_PQ.2)) → 
  investment_Q = 60000 := 
by 
  sorry

end investment_Q_correct_l72_72377


namespace tangent_slope_angle_expression_l72_72328

open Real

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * x^3

theorem tangent_slope_angle_expression :
  let α := atan (2 : ℝ)
  in  (sin α ^ 2 - cos α ^ 2) / (2 * sin α * cos α + cos α ^ 2) = 3 / 5 :=
by
  sorry

end tangent_slope_angle_expression_l72_72328


namespace solve_fraction_problem_l72_72910

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l72_72910


namespace grid_black_probability_l72_72001

theorem grid_black_probability :
  let p_black_each_cell : ℝ := 1 / 3 
  let p_not_black : ℝ := (2 / 3) * (2 / 3)
  let p_one_black : ℝ := 1 - p_not_black
  let total_pairs : ℕ := 8
  (p_one_black ^ total_pairs) = (5 / 9) ^ 8 :=
sorry

end grid_black_probability_l72_72001


namespace scientific_notation_of_570_million_l72_72539

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end scientific_notation_of_570_million_l72_72539


namespace sum_gcd_lcm_eq_4851_l72_72777

theorem sum_gcd_lcm_eq_4851 (a b : ℕ) (ha : a = 231) (hb : b = 4620) :
  Nat.gcd a b + Nat.lcm a b = 4851 :=
by
  rw [ha, hb]
  sorry

end sum_gcd_lcm_eq_4851_l72_72777


namespace jerry_mowing_income_l72_72518

theorem jerry_mowing_income (M : ℕ) (week_spending : ℕ) (money_weed_eating : ℕ) (weeks : ℕ)
  (H1 : week_spending = 5)
  (H2 : money_weed_eating = 31)
  (H3 : weeks = 9)
  (H4 : (M + money_weed_eating) = week_spending * weeks)
  : M = 14 :=
by {
  sorry
}

end jerry_mowing_income_l72_72518


namespace number_of_people_seated_l72_72708

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l72_72708


namespace percentage_cd_only_l72_72678

noncomputable def percentage_power_windows : ℝ := 0.60
noncomputable def percentage_anti_lock_brakes : ℝ := 0.40
noncomputable def percentage_cd_player : ℝ := 0.75
noncomputable def percentage_gps_system : ℝ := 0.50
noncomputable def percentage_pw_and_abs : ℝ := 0.10
noncomputable def percentage_abs_and_cd : ℝ := 0.15
noncomputable def percentage_pw_and_cd : ℝ := 0.20
noncomputable def percentage_gps_and_abs : ℝ := 0.12
noncomputable def percentage_gps_and_cd : ℝ := 0.18
noncomputable def percentage_pw_and_gps : ℝ := 0.25

theorem percentage_cd_only : 
  percentage_cd_player - (percentage_abs_and_cd + percentage_pw_and_cd + percentage_gps_and_cd) = 0.22 := 
by
  sorry

end percentage_cd_only_l72_72678


namespace triangle_cannot_have_two_right_angles_l72_72380

theorem triangle_cannot_have_two_right_angles (A B C : ℝ) (h : A + B + C = 180) : 
  ¬ (A = 90 ∧ B = 90) :=
by {
  sorry
}

end triangle_cannot_have_two_right_angles_l72_72380


namespace find_n_l72_72205

theorem find_n (n : ℕ) (a_n D_n d_n : ℕ) (h1 : n > 5) (h2 : D_n - d_n = a_n) : n = 9 := 
by 
  sorry

end find_n_l72_72205


namespace triplet_solution_l72_72041

theorem triplet_solution (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  (a + b + c = (1 / a) + (1 / b) + (1 / c) ∧ a ^ 2 + b ^ 2 + c ^ 2 = (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2))
  ↔ (∃ x, (a = 1 ∨ a = -1 ∨ a = x ∨ a = 1/x) ∧
           (b = 1 ∨ b = -1 ∨ b = x ∨ b = 1/x) ∧
           (c = 1 ∨ c = -1 ∨ c = x ∨ c = 1/x)) := 
sorry

end triplet_solution_l72_72041


namespace rita_daily_minimum_payment_l72_72881

theorem rita_daily_minimum_payment (total_cost down_payment balance daily_payment : ℝ) 
    (h1 : total_cost = 120)
    (h2 : down_payment = total_cost / 2)
    (h3 : balance = total_cost - down_payment)
    (h4 : daily_payment = balance / 10) : daily_payment = 6 :=
by
  sorry

end rita_daily_minimum_payment_l72_72881


namespace find_subtracted_value_l72_72599

theorem find_subtracted_value (N : ℤ) (V : ℤ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 :=
by
  sorry

end find_subtracted_value_l72_72599


namespace sequence_property_l72_72832

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end sequence_property_l72_72832


namespace evaluation_expression_l72_72558

theorem evaluation_expression : 
  20 * (10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5))) = 192.6 := 
by
  sorry

end evaluation_expression_l72_72558


namespace find_b_l72_72647

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 35 * b) : b = 63 := 
by 
  sorry

end find_b_l72_72647


namespace staff_discount_price_l72_72790

theorem staff_discount_price (d : ℝ) : (d - 0.15*d) * 0.90 = 0.765 * d :=
by
  have discount1 : d - 0.15 * d = d * 0.85 :=
    by ring
  have discount2 : (d * 0.85) * 0.90 = d * (0.85 * 0.90) :=
    by ring
  have final_price : d * (0.85 * 0.90) = d * 0.765 :=
    by norm_num
  rw [discount1, discount2, final_price]
  sorry

end staff_discount_price_l72_72790


namespace geometric_sequence_third_term_l72_72946

theorem geometric_sequence_third_term :
  ∃ (a : ℕ) (r : ℝ), a = 5 ∧ a * r^3 = 500 ∧ a * r^2 = 5 * 100^(2/3) :=
by
  sorry

end geometric_sequence_third_term_l72_72946


namespace B_is_criminal_l72_72980

-- Introduce the conditions
variable (A B C : Prop)  -- A, B, and C represent whether each individual is the criminal.

-- A says they did not commit the crime
axiom A_says_innocent : ¬A

-- Exactly one of A_says_innocent must hold true (A says ¬A, so B or C must be true)
axiom exactly_one_assertion_true : (¬A ∨ B ∨ C)

-- Problem Statement: Prove that B is the criminal
theorem B_is_criminal : B :=
by
  -- Solution steps would go here
  sorry

end B_is_criminal_l72_72980


namespace quadratic_to_vertex_form_l72_72382

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (1/2) * x^2 - 2 * x + 1 = (1/2) * (x - 2)^2 - 1 :=
by
  intro x
  -- full proof omitted
  sorry

end quadratic_to_vertex_form_l72_72382


namespace sheets_of_paper_l72_72124

theorem sheets_of_paper (S E : ℕ) (h1 : S - E = 100) (h2 : E = S / 3 - 25) : S = 120 :=
sorry

end sheets_of_paper_l72_72124


namespace smallest_x_with_18_factors_and_factors_18_24_l72_72893

theorem smallest_x_with_18_factors_and_factors_18_24 :
  ∃ (x : ℕ), (∃ (a b : ℕ), x = 2^a * 3^b ∧ 18 ∣ x ∧ 24 ∣ x ∧ (a + 1) * (b + 1) = 18) ∧
    (∀ y, (∃ (c d : ℕ), y = 2^c * 3^d ∧ 18 ∣ y ∧ 24 ∣ y ∧ (c + 1) * (d + 1) = 18) → x ≤ y) :=
by
  sorry

end smallest_x_with_18_factors_and_factors_18_24_l72_72893


namespace trigonometric_identity_l72_72178

open Real

theorem trigonometric_identity
  (α β γ φ : ℝ)
  (h1 : sin α + 7 * sin β = 4 * (sin γ + 2 * sin φ))
  (h2 : cos α + 7 * cos β = 4 * (cos γ + 2 * cos φ)) :
  2 * cos (α - φ) = 7 * cos (β - γ) :=
by sorry

end trigonometric_identity_l72_72178


namespace sum_of_digits_in_T_shape_35_l72_72455

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the problem variables and conditions
theorem sum_of_digits_in_T_shape_35
  (a b c d e f g h : ℕ)
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
        d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
        e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
        f ≠ g ∧ f ≠ h ∧
        g ≠ h)
  (h2 : a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
        e ∈ digits ∧ f ∈ digits ∧ g ∈ digits ∧ h ∈ digits)
  (h3 : a + b + c + d = 26)
  (h4 : e + b + f + g + h = 20) :
  a + b + c + d + e + f + g + h = 35 := by
  sorry

end sum_of_digits_in_T_shape_35_l72_72455


namespace thrown_away_oranges_l72_72292

theorem thrown_away_oranges (x : ℕ) (h : 40 - x + 7 = 10) : x = 37 :=
by sorry

end thrown_away_oranges_l72_72292


namespace roots_of_quadratic_l72_72553

theorem roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -2 ∧ c = -6) :
  let Δ := b^2 - 4 * a * c in
  Δ > 0 :=
by
  obtain ⟨ha, hb, hc⟩ := h_eq
  have Δ_def : Δ = b^2 - 4 * a * c := rfl
  rw [ha, hb, hc] at Δ_def
  sorry

end roots_of_quadratic_l72_72553


namespace mod_equiv_22_l72_72108

theorem mod_equiv_22 : ∃ m : ℕ, (198 * 864) % 50 = m ∧ 0 ≤ m ∧ m < 50 ∧ m = 22 := by
  sorry

end mod_equiv_22_l72_72108


namespace unplanted_fraction_l72_72200

theorem unplanted_fraction (a b hypotenuse : ℕ) (side_length_P : ℚ) 
                          (h1 : a = 5) (h2 : b = 12) (h3 : hypotenuse = 13)
                          (h4 : side_length_P = 5 / 3) : 
                          (side_length_P * side_length_P) / ((a * b) / 2) = 5 / 54 :=
by
  sorry

end unplanted_fraction_l72_72200


namespace group_capacity_l72_72403

theorem group_capacity (total_students : ℕ) (selected_students : ℕ) (removed_students : ℕ) :
  total_students = 5008 → selected_students = 200 → removed_students = 8 →
  (total_students - removed_students) / selected_students = 25 :=
by
  intros h1 h2 h3
  sorry

end group_capacity_l72_72403


namespace sampling_methods_correct_l72_72263

def condition1 : Prop :=
  ∃ yogurt_boxes : ℕ, yogurt_boxes = 10 ∧ ∃ sample_boxes : ℕ, sample_boxes = 3

def condition2 : Prop :=
  ∃ rows seats_per_row attendees sample_size : ℕ,
    rows = 32 ∧ seats_per_row = 40 ∧ attendees = rows * seats_per_row ∧ sample_size = 32

def condition3 : Prop :=
  ∃ liberal_arts_classes science_classes total_classes sample_size : ℕ,
    liberal_arts_classes = 4 ∧ science_classes = 8 ∧ total_classes = liberal_arts_classes + science_classes ∧ sample_size = 50

def simple_random_sampling (s : Prop) : Prop := sorry -- definition for simple random sampling
def systematic_sampling (s : Prop) : Prop := sorry -- definition for systematic sampling
def stratified_sampling (s : Prop) : Prop := sorry -- definition for stratified sampling

theorem sampling_methods_correct :
  (condition1 → simple_random_sampling condition1) ∧
  (condition2 → systematic_sampling condition2) ∧
  (condition3 → stratified_sampling condition3) :=
by {
  sorry
}

end sampling_methods_correct_l72_72263


namespace circle_area_l72_72577

noncomputable def circle_area_above_line (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 - 18 * y + 61 = 0

theorem circle_area (A : ℝ) (H : ∀ (x y : ℝ), circle_area_above_line x y → y >= 4) :
  A = π :=
by
  sorry

end circle_area_l72_72577


namespace factor_expression_zero_l72_72976

theorem factor_expression_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2 = 0 :=
sorry

end factor_expression_zero_l72_72976


namespace alice_arrives_earlier_l72_72015

/-
Alice and Bob are heading to a park that is 2 miles away from their home. 
They leave home at the same time. 
Alice cycles to the park at a speed of 12 miles per hour, 
while Bob jogs there at a speed of 6 miles per hour. 
Prove that Alice arrives 10 minutes earlier at the park than Bob.
-/

theorem alice_arrives_earlier 
  (d : ℕ) (a_speed : ℕ) (b_speed : ℕ) (arrival_difference_minutes : ℕ) 
  (h1 : d = 2) 
  (h2 : a_speed = 12) 
  (h3 : b_speed = 6) 
  (h4 : arrival_difference_minutes = 10) 
  : (d / a_speed * 60) + arrival_difference_minutes = d / b_speed * 60 :=
by
  sorry

end alice_arrives_earlier_l72_72015


namespace peter_money_left_l72_72378

variable (soda_cost : ℝ) (money_brought : ℝ) (soda_ounces : ℝ)

theorem peter_money_left (h1 : soda_cost = 0.25) (h2 : money_brought = 2) (h3 : soda_ounces = 6) : 
    money_brought - soda_ounces * soda_cost = 0.50 := 
by 
  sorry

end peter_money_left_l72_72378


namespace billiard_ball_weight_l72_72255

theorem billiard_ball_weight (w_box w_box_with_balls : ℝ) (h_w_box : w_box = 0.5) 
(h_w_box_with_balls : w_box_with_balls = 1.82) : 
    let total_weight_balls := w_box_with_balls - w_box;
    let weight_one_ball := total_weight_balls / 6;
    weight_one_ball = 0.22 :=
by
  sorry

end billiard_ball_weight_l72_72255


namespace add_to_frac_eq_l72_72919

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l72_72919


namespace solve_system_of_equations_l72_72237

theorem solve_system_of_equations (x y : ℝ) :
    (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧ 5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
    (x = 2 ∧ y = 1) ∨ (x = 2 / 5 ∧ y = -(1 / 5)) :=
by
  sorry

end solve_system_of_equations_l72_72237


namespace paint_proof_l72_72083

/-- 
Suppose Jack's room has 27 square meters of wall and ceiling area. He has three choices for paint:
- Using 1 can of paint leaves 1 liter of paint left over,
- Using 5 gallons of paint leaves 1 liter of paint left over,
- Using 4 gallons and 2.8 liters of paint.

1. Prove: The ratio between the volume of a can and the volume of a gallon is 1:5.
2. Prove: The volume of a gallon is 3.8 liters.
3. Prove: The paint's coverage is 1.5 square meters per liter.
-/
theorem paint_proof (A : ℝ) (C G : ℝ) (R : ℝ):
  ∀ (H1: A = 27) (H2: C - 1 = 27) (H3: 5 * G - 1 = 27) (H4: 4 * G + 2.8 = 27), 
  (C / G = 1 / 5) ∧ (G = 3.8) ∧ ((A / (5 * G - 1)) = 1.5) :=
by
  sorry

end paint_proof_l72_72083


namespace probability_of_A_l72_72950

noncomputable def ξ : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.Measure.gaussian 0 4

def A : Set ℝ := { x | (1 / x) > (1 / (1 + x)) }

theorem probability_of_A :
  MeasureTheory.Measure (A ∩ Range (MeasureTheory.Measure.toOuterMeasure ξ))
  = MeasureTheory.Measure.gennnicely 0.6587 := 
by
  sorry

end probability_of_A_l72_72950


namespace area_difference_of_circles_l72_72390

theorem area_difference_of_circles (circumference_large: ℝ) (half_radius_relation: ℝ → ℝ) (hl: circumference_large = 36) (hr: ∀ R, half_radius_relation R = R / 2) :
  ∃ R r, R = 18 / π ∧ r = 9 / π ∧ (π * R ^ 2 - π * r ^ 2) = 243 / π :=
by 
  sorry

end area_difference_of_circles_l72_72390


namespace remainder_of_9_pow_333_div_50_l72_72130

theorem remainder_of_9_pow_333_div_50 : (9 ^ 333) % 50 = 29 :=
by
  sorry

end remainder_of_9_pow_333_div_50_l72_72130


namespace ratio_white_to_remaining_l72_72536

def total_beans : ℕ := 572

def red_beans (total : ℕ) : ℕ := total / 4

def remaining_beans_after_red (total : ℕ) (red : ℕ) : ℕ := total - red

def green_beans : ℕ := 143

def remaining_beans_after_green (remaining : ℕ) (green : ℕ) : ℕ := remaining - green

def white_beans (remaining : ℕ) : ℕ := remaining / 2

theorem ratio_white_to_remaining (total : ℕ) (red : ℕ) (remaining : ℕ) (green : ℕ) (white : ℕ) 
  (H_total : total = 572)
  (H_red : red = red_beans total)
  (H_remaining : remaining = remaining_beans_after_red total red)
  (H_green : green = 143)
  (H_remaining_after_green : remaining_beans_after_green remaining green = white)
  (H_white : white = white_beans remaining) :
  (white : ℚ) / (remaining : ℚ) = (1 : ℚ) / 2 := 
by sorry

end ratio_white_to_remaining_l72_72536


namespace slices_left_for_lunch_tomorrow_l72_72306

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end slices_left_for_lunch_tomorrow_l72_72306


namespace average_speed_l72_72940

theorem average_speed (v : ℝ) (v_pos : 0 < v) (v_pos_10 : 0 < v + 10):
  420 / v - 420 / (v + 10) = 2 → v = 42 :=
by
  sorry

end average_speed_l72_72940


namespace twenty_seven_divides_sum_l72_72629

theorem twenty_seven_divides_sum (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) : 27 ∣ x + y + z := sorry

end twenty_seven_divides_sum_l72_72629


namespace externally_tangent_circles_radius_l72_72485

theorem externally_tangent_circles_radius :
  ∃ r : ℝ, r > 0 ∧ (∀ x y, (x^2 + y^2 = 1 ∧ ((x - 3)^2 + y^2 = r^2)) → r = 2) :=
sorry

end externally_tangent_circles_radius_l72_72485


namespace flowers_per_vase_l72_72435

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end flowers_per_vase_l72_72435


namespace largest_integer_sol_l72_72127

theorem largest_integer_sol (x : ℤ) : (3 * x + 4 < 5 * x - 2) -> x = 3 :=
by
  sorry

end largest_integer_sol_l72_72127


namespace find_side_length_a_l72_72180

noncomputable def length_of_a (A B : ℝ) (b : ℝ) : ℝ :=
  b * Real.sin A / Real.sin B

theorem find_side_length_a :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = Real.pi / 3 → B = Real.pi / 4 → b = Real.sqrt 6 →
  a = length_of_a A B b →
  a = 3 :=
by
  intros a b c A B C hA hB hb ha
  rw [hA, hB, hb] at ha
  sorry

end find_side_length_a_l72_72180


namespace perpendicular_lines_l72_72993

def line1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y) ∧ (∀ x y : ℝ, line2 a x y) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, 
    (line1 a x1 y1) ∧ (line2 a x2 y2) → 
    (-a / 2) * (-1 / (a - 1)) = -1) → a = 2 / 3 :=
sorry

end perpendicular_lines_l72_72993


namespace men_seated_l72_72419

theorem men_seated (total_passengers : ℕ) (women_ratio : ℚ) (children_count : ℕ) (men_standing_ratio : ℚ) 
  (women_with_prams : ℕ) (disabled_passengers : ℕ) 
  (h_total_passengers : total_passengers = 48) 
  (h_women_ratio : women_ratio = 2 / 3) 
  (h_children_count : children_count = 5) 
  (h_men_standing_ratio : men_standing_ratio = 1 / 8) 
  (h_women_with_prams : women_with_prams = 3) 
  (h_disabled_passengers : disabled_passengers = 2) : 
  (total_passengers * (1 - women_ratio) - total_passengers * (1 - women_ratio) * men_standing_ratio = 14) :=
by sorry

end men_seated_l72_72419


namespace number_of_people_third_day_l72_72424

variable (X : ℕ)
variable (total : ℕ := 246)
variable (first_day : ℕ := 79)
variable (second_day_third_day_diff : ℕ := 47)

theorem number_of_people_third_day :
  (first_day + (X + second_day_third_day_diff) + X = total) → 
  X = 60 := by
  sorry

end number_of_people_third_day_l72_72424


namespace inverse_of_73_mod_74_l72_72814

theorem inverse_of_73_mod_74 :
  73 * 73 ≡ 1 [MOD 74] :=
by
  sorry

end inverse_of_73_mod_74_l72_72814


namespace junior_score_l72_72432

variables (n : ℕ) (j s : ℕ)
  (num_juniors num_seniors : ℕ)
  (average_class_score average_senior_score : ℕ)

-- Given conditions
def cond1 := num_juniors = 0.1 * n
def cond2 := num_seniors = 0.9 * n
def cond3 := average_class_score = 84
def cond4 := average_senior_score = 83

-- Total scores
def total_class_score := n * average_class_score
def total_senior_score := num_seniors * average_senior_score
def total_junior_score := j * num_juniors

-- Assert total score consistency
def cons_total_score := total_class_score = total_senior_score + total_junior_score

-- Proof statement
theorem junior_score :
  cond1 →
  cond2 →
  cond3 →
  cond4 →
  cons_total_score →
  j = 93 :=
by
  sorry

end junior_score_l72_72432


namespace pond_sustain_capacity_l72_72144

-- Defining the initial number of frogs
def initial_frogs : ℕ := 5

-- Defining the number of tadpoles
def number_of_tadpoles (frogs: ℕ) : ℕ := 3 * frogs

-- Defining the number of matured tadpoles (those that survive to become frogs)
def matured_tadpoles (tadpoles: ℕ) : ℕ := (2 * tadpoles) / 3

-- Defining the total number of frogs after tadpoles mature
def total_frogs_after_mature (initial_frogs: ℕ) (matured_tadpoles: ℕ) : ℕ :=
  initial_frogs + matured_tadpoles

-- Defining the number of frogs that need to find a new pond
def frogs_to_leave : ℕ := 7

-- Defining the number of frogs the pond can sustain
def frogs_pond_can_sustain (total_frogs: ℕ) (frogs_to_leave: ℕ) : ℕ :=
  total_frogs - frogs_to_leave

-- The main theorem stating the number of frogs the pond can sustain given the conditions
theorem pond_sustain_capacity : frogs_pond_can_sustain
  (total_frogs_after_mature initial_frogs (matured_tadpoles (number_of_tadpoles initial_frogs)))
  frogs_to_leave = 8 := by
  -- proof goes here
  sorry

end pond_sustain_capacity_l72_72144


namespace interval_width_and_count_l72_72781

def average_income_intervals := [3000, 4000, 5000, 6000, 7000]
def frequencies := [5, 9, 4, 2]

theorem interval_width_and_count:
  (average_income_intervals[1] - average_income_intervals[0] = 1000) ∧
  (frequencies.length = 4) :=
by
  sorry

end interval_width_and_count_l72_72781


namespace cargo_to_passenger_ratio_l72_72600

def total_cars : Nat := 71
def passenger_cars : Nat := 44
def engine_and_caboose : Nat := 2
def cargo_cars : Nat := total_cars - passenger_cars - engine_and_caboose

theorem cargo_to_passenger_ratio : cargo_cars = 25 ∧ passenger_cars = 44 →
  cargo_cars.toFloat / passenger_cars.toFloat = 25.0 / 44.0 :=
by
  intros h
  rw [h.1]
  rw [h.2]
  sorry

end cargo_to_passenger_ratio_l72_72600


namespace baker_bought_131_new_cakes_l72_72803

def number_of_new_cakes_bought (initial_cakes: ℕ) (cakes_sold: ℕ) (excess_sold: ℕ): ℕ :=
    cakes_sold - excess_sold - initial_cakes

theorem baker_bought_131_new_cakes :
    number_of_new_cakes_bought 8 145 6 = 131 :=
by
  -- This is where the proof would normally go
  sorry

end baker_bought_131_new_cakes_l72_72803


namespace expression_for_A_l72_72603

theorem expression_for_A (A k : ℝ)
  (h : ∀ k : ℝ, Ax^2 + 6 * k * x + 2 = 0 → k = 0.4444444444444444 → (6 * k)^2 - 4 * A * 2 = 0) :
  A = 9 * k^2 / 2 := 
sorry

end expression_for_A_l72_72603


namespace wenlock_olympian_games_first_held_year_difference_l72_72677

theorem wenlock_olympian_games_first_held_year_difference :
  2012 - 1850 = 162 :=
sorry

end wenlock_olympian_games_first_held_year_difference_l72_72677


namespace max_area_of_garden_l72_72476

theorem max_area_of_garden (total_fence : ℝ) (gate : ℝ) (remaining_fence := total_fence - gate) :
  total_fence = 60 → gate = 4 → (remaining_fence / 2) * (remaining_fence / 2) = 196 :=
by 
  sorry

end max_area_of_garden_l72_72476


namespace cheaper_lens_price_l72_72374

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) 
  (h₁ : original_price = 300) 
  (h₂ : discount_rate = 0.20) 
  (h₃ : savings = 20) 
  (discounted_price : ℝ) 
  (cheaper_lens_price : ℝ)
  (discount_eq : discounted_price = original_price * (1 - discount_rate))
  (savings_eq : cheaper_lens_price = discounted_price - savings) :
  cheaper_lens_price = 220 := 
by sorry

end cheaper_lens_price_l72_72374


namespace inequality_proof_l72_72231

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end inequality_proof_l72_72231


namespace domain_of_sqrt_fraction_l72_72032

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end domain_of_sqrt_fraction_l72_72032


namespace gcd_pow_minus_one_l72_72043

theorem gcd_pow_minus_one {m n : ℕ} (hm : 0 < m) (hn : 0 < n) :
  Nat.gcd (2^m - 1) (2^n - 1) = 2^Nat.gcd m n - 1 :=
sorry

end gcd_pow_minus_one_l72_72043


namespace find_length_of_FC_l72_72172

theorem find_length_of_FC (DC CB AD AB ED FC : ℝ) (h1 : DC = 9) (h2 : CB = 10) (h3 : AB = (1 / 3) * AD) (h4 : ED = (2 / 3) * AD) : 
  FC = 13 := by
  sorry

end find_length_of_FC_l72_72172


namespace felicity_gas_usage_l72_72458

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l72_72458


namespace min_value_of_fraction_sum_l72_72372

theorem min_value_of_fraction_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x^2 + y^2 + z^2 = 1) :
  (2 * (1/(1-x^2) + 1/(1-y^2) + 1/(1-z^2))) = 3 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_sum_l72_72372


namespace find_sum_l72_72082

variable {a : ℕ → ℝ} {r : ℝ}

-- Conditions: a_n > 0 for all n
axiom pos : ∀ n : ℕ, a n > 0

-- Given equation: a_1 * a_5 + 2 * a_3 * a_5 + a_3 * a_7 = 25
axiom given_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

theorem find_sum : a 3 + a 5 = 5 :=
by
  sorry

end find_sum_l72_72082


namespace abhinav_annual_salary_l72_72000

def RamMontlySalary : ℝ := 25600
def ShyamMontlySalary (A : ℝ) := 2 * A
def AbhinavAnnualSalary (A : ℝ) := 12 * A

theorem abhinav_annual_salary (A : ℝ) : 
  0.10 * RamMontlySalary = 0.08 * ShyamMontlySalary A → 
  AbhinavAnnualSalary A = 192000 :=
by
  sorry

end abhinav_annual_salary_l72_72000


namespace cos_diff_angle_l72_72068

theorem cos_diff_angle
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (h : 3 * Real.sin α = Real.tan α) :
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 :=
sorry

end cos_diff_angle_l72_72068


namespace ratio_of_numbers_l72_72120

theorem ratio_of_numbers (a b : ℕ) (h1 : a.gcd b = 5) (h2 : a.lcm b = 60) (h3 : a = 3 * 5) (h4 : b = 4 * 5) : (a / a.gcd b) / (b / a.gcd b) = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l72_72120


namespace polynomial_square_b_value_l72_72546

theorem polynomial_square_b_value (a b p q : ℝ) (h : (∀ x : ℝ, x^4 + x^3 - x^2 + a * x + b = (x^2 + p * x + q)^2)) : b = 25/64 := by
  sorry

end polynomial_square_b_value_l72_72546


namespace number_of_ways_to_choose_museums_l72_72796

-- Define the conditions
def number_of_grades : Nat := 6
def number_of_museums : Nat := 6
def number_of_grades_Museum_A : Nat := 2

-- Prove the number of ways to choose museums such that exactly two grades visit Museum A
theorem number_of_ways_to_choose_museums :
  (Nat.choose number_of_grades number_of_grades_Museum_A) * (5 ^ (number_of_grades - number_of_grades_Museum_A)) = Nat.choose 6 2 * 5 ^ 4 :=
by
  sorry

end number_of_ways_to_choose_museums_l72_72796


namespace total_people_seated_l72_72693

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l72_72693


namespace simplify_fraction_product_l72_72882

theorem simplify_fraction_product :
  8 * (15 / 14) * (-49 / 45) = - (28 / 3) :=
by
  sorry

end simplify_fraction_product_l72_72882


namespace perpendicular_vectors_x_value_l72_72673

theorem perpendicular_vectors_x_value
  (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (1, -2)) (hb : b = (-3, x))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -3 / 2 := by
  sorry

end perpendicular_vectors_x_value_l72_72673


namespace total_sandwiches_l72_72020

theorem total_sandwiches :
  let billy := 49
  let katelyn := billy + 47
  let chloe := katelyn / 4
  billy + katelyn + chloe = 169 :=
by
  sorry

end total_sandwiches_l72_72020


namespace circles_disjoint_l72_72331

-- Definitions of the circles
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 1
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Prove that the circles are disjoint
theorem circles_disjoint : 
  (¬ ∃ (x y : ℝ), circleM x y ∧ circleN x y) :=
by sorry

end circles_disjoint_l72_72331


namespace value_of_c7_l72_72872

theorem value_of_c7 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : ∀ n, b n = 2^(n-1))
  (h3 : ∀ n, c n = a n * b n) :
  c 7 = 448 :=
by
  sorry

end value_of_c7_l72_72872


namespace colten_chickens_l72_72100

/-
Define variables to represent the number of chickens each person has.
-/

variables (C : ℕ)   -- Number of chickens Colten has.
variables (S : ℕ)   -- Number of chickens Skylar has.
variables (Q : ℕ)   -- Number of chickens Quentin has.

/-
Define the given conditions
-/
def condition1 := Q + S + C = 383
def condition2 := Q = 2 * S + 25
def condition3 := S = 3 * C - 4

theorem colten_chickens : C = 37 :=
by
  -- Proof elaboration to be done with sorry for the auto proof
  sorry

end colten_chickens_l72_72100


namespace smallest_int_solution_l72_72265

theorem smallest_int_solution : ∃ y : ℤ, y = 6 ∧ ∀ z : ℤ, z > 5 → y ≤ z := sorry

end smallest_int_solution_l72_72265


namespace geometric_seq_general_formula_sum_c_seq_terms_l72_72176

noncomputable def a_seq (n : ℕ) : ℕ := 2 * 3 ^ (n - 1)

noncomputable def S_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (a_seq n - 2) / 2

theorem geometric_seq_general_formula (n : ℕ) (h : n > 0) : 
  a_seq n = 2 * 3 ^ (n - 1) := 
by {
  sorry
}

noncomputable def d_n (n : ℕ) : ℕ :=
  (a_seq (n + 1) - a_seq n) / (n + 1)

noncomputable def c_seq (n : ℕ) : ℕ :=
  d_n n / (n * a_seq n)

noncomputable def T_n (n : ℕ) : ℕ :=
  2 * (1 - 1 / (n + 1)) * n

theorem sum_c_seq_terms (n : ℕ) (h : n > 0) : 
  T_n n = 2 * n / (n + 1) :=
by {
  sorry
}

end geometric_seq_general_formula_sum_c_seq_terms_l72_72176


namespace greater_number_l72_72901

theorem greater_number (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 2) (h3 : a > b) : a = 21 := by
  sorry

end greater_number_l72_72901


namespace length_of_path_along_arrows_l72_72383

theorem length_of_path_along_arrows (s : List ℝ) (h : s.sum = 73) :
  (3 * s.sum = 219) :=
by
  sorry

end length_of_path_along_arrows_l72_72383


namespace find_initial_amount_l72_72674

-- Let x be the initial amount Mark paid for the Magic card
variable {x : ℝ}

-- Condition 1: The card triples in value, resulting in 3x
-- Condition 2: Mark makes a profit of 200
def initial_amount (x : ℝ) : Prop := (3 * x - x = 200)

-- Theorem: Prove that the initial amount x equals 100 given the conditions
theorem find_initial_amount (h : initial_amount x) : x = 100 := by
  sorry

end find_initial_amount_l72_72674


namespace octahedron_tetrahedron_volume_ratio_l72_72426

theorem octahedron_tetrahedron_volume_ratio (a : ℝ) :
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3
  V_o / V_t = 1 :=
by 
  -- Definitions from conditions
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3

  -- Proof omitted
  -- Proof goes here
  sorry

end octahedron_tetrahedron_volume_ratio_l72_72426


namespace total_gold_cost_l72_72821

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end total_gold_cost_l72_72821


namespace product_roots_l72_72181

noncomputable def root1 (x1 : ℝ) : Prop := x1 * Real.log x1 = 2006
noncomputable def root2 (x2 : ℝ) : Prop := x2 * Real.exp x2 = 2006

theorem product_roots (x1 x2 : ℝ) (h1 : root1 x1) (h2 : root2 x2) : x1 * x2 = 2006 := sorry

end product_roots_l72_72181


namespace balloon_difference_l72_72091

theorem balloon_difference (x y : ℝ) (h1 : x = 2 * y - 3) (h2 : y = x / 4 + 1) : x - y = -2.5 :=
by 
  sorry

end balloon_difference_l72_72091


namespace max_elements_in_S_l72_72451

-- Definitions to establish the problem domain
def valid_element (a : ℕ) : Prop := a > 0 ∧ a ≤ 100

def condition_two (S : Finset ℕ) : Prop :=
  ∀ a b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd a c = 1 ∧ Nat.gcd b c = 1

def condition_three (S : Finset ℕ) : Prop :=
  ∀ a b ∈ S, a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ Nat.gcd a d > 1 ∧ Nat.gcd b d > 1

-- Statement of the theorem
theorem max_elements_in_S (S : Finset ℕ) : 
  (∀ s ∈ S, valid_element s) ∧ 
  condition_two S ∧ 
  condition_three S →
  Finset.card S ≤ 72 := 
sorry

end max_elements_in_S_l72_72451


namespace hot_air_balloon_height_l72_72987

theorem hot_air_balloon_height (altitude_temp_decrease_per_1000m : ℝ) 
  (ground_temp : ℝ) (high_altitude_temp : ℝ) :
  altitude_temp_decrease_per_1000m = 6 →
  ground_temp = 8 →
  high_altitude_temp = -1 →
  ∃ (height : ℝ), height = 1500 :=
by
  intro h1 h2 h3
  have temp_change := ground_temp - high_altitude_temp
  have height := (temp_change / altitude_temp_decrease_per_1000m) * 1000
  exact Exists.intro height sorry -- height needs to be computed here

end hot_air_balloon_height_l72_72987


namespace intersection_of_set_M_with_complement_of_set_N_l72_72668

theorem intersection_of_set_M_with_complement_of_set_N (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 4, 5}) (hN : N = {1, 3}) : M ∩ (U \ N) = {4, 5} :=
by
  sorry

end intersection_of_set_M_with_complement_of_set_N_l72_72668


namespace no_real_solution_for_t_l72_72368

open Real

theorem no_real_solution_for_t :
  ∀ t : ℝ, let P := (t - 5, -2) in
           let Q := (-3, t + 4) in
           let midpoint := ((t - 5 - 3) / 2, (-2 + (t + 4)) / 2) in
           ¬ ((dist midpoint P / 2)^2 = t^2 + t - 1) :=
by
  sorry

end no_real_solution_for_t_l72_72368


namespace jack_salt_amount_l72_72211

noncomputable def amount_of_salt (volume_salt_1 : ℝ) (volume_salt_2 : ℝ) : ℝ :=
  volume_salt_1 + volume_salt_2

noncomputable def total_salt_ml (total_salt_l : ℝ) : ℝ :=
  total_salt_l * 1000

theorem jack_salt_amount :
  let day1_water_l := 4.0
  let day2_water_l := 4.0
  let day1_salt_percentage := 0.18
  let day2_salt_percentage := 0.22
  let total_salt_before_evaporation := amount_of_salt (day1_water_l * day1_salt_percentage) (day2_water_l * day2_salt_percentage)
  let final_salt_ml := total_salt_ml total_salt_before_evaporation
  final_salt_ml = 1600 :=
by
  sorry

end jack_salt_amount_l72_72211


namespace count_and_largest_special_numbers_l72_72061

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_four_digit_number (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 10000

theorem count_and_largest_special_numbers :
  ∃ (nums : List ℕ), 
    (∀ n ∈ nums, ∃ x y : ℕ, is_prime x ∧ is_prime y ∧ 
      55 * x * y = n ∧ is_four_digit_number (n * 5))
    ∧ nums.length = 3
    ∧ nums.maximum = some 4785 :=
sorry

end count_and_largest_special_numbers_l72_72061


namespace total_carrots_l72_72532

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end total_carrots_l72_72532


namespace intersection_of_sets_l72_72984

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3}) (hB : B = { x | x < 3 ∧ x ∈ Set.univ }) :
  A ∩ B = {0, 1, 2} :=
by
  sorry

end intersection_of_sets_l72_72984


namespace denise_travel_l72_72303

theorem denise_travel (a b c : ℕ) (h₀ : a ≥ 1) (h₁ : a + b + c = 8) (h₂ : 90 * (b - a) % 48 = 0) : a^2 + b^2 + c^2 = 26 :=
sorry

end denise_travel_l72_72303


namespace num_functions_l72_72087

theorem num_functions (A : Finset ℕ) :
  A = {1, 2, 3, 4, 5, 6, 7, 8} →
  (Σ f : {f : ℕ → ℕ // ∀ x, x ∈ A → f(f x) = c ∨ (∃ a b, a ∈ A ∧ b ∈ A ∧ b ≠ c ∧ f(f a) = b ∧ f(f x) = c) }) = 63488 :=
by
  intro hA
  sorry

end num_functions_l72_72087


namespace spherical_to_rectangular_coordinates_l72_72448

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ),
  ρ = 5 → θ = π / 6 → φ = π / 3 →
  let x := ρ * (Real.sin φ * Real.cos θ)
  let y := ρ * (Real.sin φ * Real.sin θ)
  let z := ρ * Real.cos φ
  x = 15 / 4 ∧ y = 5 * Real.sqrt 3 / 4 ∧ z = 2.5 :=
by
  intros ρ θ φ hρ hθ hφ
  sorry

end spherical_to_rectangular_coordinates_l72_72448


namespace probability_not_both_ends_l72_72170

theorem probability_not_both_ends :
  let total_arrangements := 120
  let both_ends_arrangements := 12
  let favorable_arrangements := total_arrangements - both_ends_arrangements
  let probability := favorable_arrangements / total_arrangements
  total_arrangements = 120 ∧ both_ends_arrangements = 12 ∧ favorable_arrangements = 108 ∧ probability = 0.9 :=
by
  sorry

end probability_not_both_ends_l72_72170


namespace degree_measure_of_regular_hexagon_interior_angle_l72_72755

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l72_72755


namespace interior_angle_regular_hexagon_l72_72762

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l72_72762


namespace arithmetic_sequence_length_l72_72334

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a d l : ℕ), a = 2 → d = 5 → l = 3007 → l = a + (n-1) * d → n = 602 :=
by
  sorry

end arithmetic_sequence_length_l72_72334


namespace find_x2_times_sum_roots_l72_72672

noncomputable def sqrt2015 := Real.sqrt 2015

theorem find_x2_times_sum_roots
  (x1 x2 x3 : ℝ)
  (h_eq : ∀ x : ℝ, sqrt2015 * x^3 - 4030 * x^2 + 2 = 0 → x = x1 ∨ x = x2 ∨ x = x3)
  (h_ineq : x1 < x2 ∧ x2 < x3) :
  x2 * (x1 + x3) = 2 := by
  sorry

end find_x2_times_sum_roots_l72_72672


namespace droneSystemEquations_l72_72283

-- Definitions based on conditions
def typeADrones (x y : ℕ) : Prop := x = (1/2 : ℝ) * (x + y) + 11
def typeBDrones (x y : ℕ) : Prop := y = (1/3 : ℝ) * (x + y) - 2

-- Theorem statement
theorem droneSystemEquations (x y : ℕ) :
  typeADrones x y ∧ typeBDrones x y ↔
  (x = (1/2 : ℝ) * (x + y) + 11 ∧ y = (1/3 : ℝ) * (x + y) - 2) :=
by sorry

end droneSystemEquations_l72_72283


namespace smaller_of_two_numbers_l72_72738

theorem smaller_of_two_numbers 
  (a b d : ℝ) (h : 0 < a ∧ a < b) (u v : ℝ) 
  (huv : u / v = b / a) (sum_uv : u + v = d) : 
  min u v = (a * d) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_l72_72738


namespace johnny_words_l72_72859

def words_johnny (J : ℕ) :=
  let words_madeline := 2 * J
  let words_timothy := 2 * J + 30
  let total_words := J + words_madeline + words_timothy
  total_words = 3 * 260 → J = 150

-- Statement of the main theorem (no proof provided, hence sorry is used)
theorem johnny_words (J : ℕ) : words_johnny J :=
by sorry

end johnny_words_l72_72859


namespace original_number_l72_72941

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_list (n : ℕ) (a b c d e : ℕ) : Prop :=
  n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e

def four_digit_variant (N n : ℕ) (a b c d e : ℕ) : Prop :=
  (n = 10^3 * b + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d)

theorem original_number (N : ℕ) (a b c d e : ℕ) 
  (h1 : is_five_digit N) 
  (h2 : digit_list N a b c d e)
  (h3 : ∃ n, is_five_digit n ∧ four_digit_variant N n a b c d e ∧ N + n = 54321) :
  N = 49383 := 
sorry

end original_number_l72_72941


namespace tenth_number_drawn_eq_195_l72_72012

noncomputable def total_students : Nat := 1000
noncomputable def sample_size : Nat := 50
noncomputable def first_selected_number : Nat := 15  -- Note: 0015 is 15 in natural number

theorem tenth_number_drawn_eq_195 
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_selected_number = 15) :
  15 + (20 * 9) = 195 := 
by
  sorry

end tenth_number_drawn_eq_195_l72_72012


namespace builder_installed_windows_l72_72596

-- Conditions
def total_windows : ℕ := 14
def hours_per_window : ℕ := 8
def remaining_hours : ℕ := 48

-- Definition for the problem statement
def installed_windows := total_windows - remaining_hours / hours_per_window

-- The hypothesis we need to prove
theorem builder_installed_windows : installed_windows = 8 := by
  sorry

end builder_installed_windows_l72_72596


namespace cars_meet_cars_apart_l72_72183

section CarsProblem

variable (distance : ℕ) (speedA speedB : ℕ) (distanceToMeet distanceApart : ℕ)

def meetTime := distance / (speedA + speedB)
def apartTime1 := (distance - distanceApart) / (speedA + speedB)
def apartTime2 := (distance + distanceApart) / (speedA + speedB)

theorem cars_meet (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85):
  meetTime distance speedA speedB = 9 / 4 := by
  sorry

theorem cars_apart (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85) (h4: distanceApart = 50):
  apartTime1 distance speedA speedB distanceApart = 2 ∧ apartTime2 distance speedA speedB distanceApart = 5 / 2 := by
  sorry

end CarsProblem

end cars_meet_cars_apart_l72_72183


namespace numbers_are_odd_l72_72556

theorem numbers_are_odd (n : ℕ) (sum : ℕ) (h1 : n = 49) (h2 : sum = 2401) : 
      (∀ i < n, ∃ j, sum = j * 2 * i + 1) :=
by
  sorry

end numbers_are_odd_l72_72556


namespace fraction_of_usual_speed_l72_72589

-- Definitions based on conditions
variable (S R : ℝ)
variable (h1 : S * 60 = R * 72)

-- Goal statement
theorem fraction_of_usual_speed (h1 : S * 60 = R * 72) : R / S = 5 / 6 :=
by
  sorry

end fraction_of_usual_speed_l72_72589


namespace parabola_point_l72_72634

theorem parabola_point (a b c : ℝ) (hA : 0.64 * a - 0.8 * b + c = 4.132)
  (hB : 1.44 * a + 1.2 * b + c = -1.948) (hC : 7.84 * a + 2.8 * b + c = -3.932) :
  0.5 * (1.8)^2 - 3.24 * 1.8 + 1.22 = -2.992 :=
by
  -- Proof is intentionally omitted
  sorry

end parabola_point_l72_72634


namespace prod_of_real_roots_equation_l72_72730

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end prod_of_real_roots_equation_l72_72730


namespace logic_problem_l72_72199

variables (p q : Prop)

theorem logic_problem (hnp : ¬ p) (hpq : ¬ (p ∧ q)) : ¬ (p ∨ q) ∨ (p ∨ q) :=
by 
  sorry

end logic_problem_l72_72199


namespace smallest_value_of_a_l72_72247

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l72_72247


namespace john_initial_payment_l72_72364

-- Definitions based on the conditions from step a)
def cost_per_soda : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

-- Problem Statement: Prove that the total amount of money John paid initially is $20
theorem john_initial_payment :
  cost_per_soda * num_sodas + change_received = 20 := 
by
  sorry -- Proof steps are omitted as per instructions

end john_initial_payment_l72_72364


namespace domain_of_log_function_l72_72240

theorem domain_of_log_function (x : ℝ) :
  (-1 < x ∧ x < 1) ↔ (1 - x) / (1 + x) > 0 :=
by sorry

end domain_of_log_function_l72_72240


namespace sum_of_factors_of_30_l72_72776

theorem sum_of_factors_of_30 : 
  let n := 30 in sum (filter (λ d, n % d = 0) (list.range (n + 1))) = 72 :=
by 
  let n := 30
  sorry

end sum_of_factors_of_30_l72_72776


namespace quadratic_csq_l72_72236

theorem quadratic_csq (x q t : ℝ) (h : 9 * x^2 - 36 * x - 81 = 0) (hq : q = -2) (ht : t = 13) :
  q + t = 11 :=
by
  sorry

end quadratic_csq_l72_72236


namespace pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l72_72049

-- Definitions based on the problem's conditions
def a (n : Nat) : Nat := n * n

def pos_count (n : Nat) : Nat :=
  List.length (List.filter (λ m : Nat => a m < n) (List.range (n + 1)))

def pos_pos_count (n : Nat) : Nat :=
  pos_count (pos_count n)

-- Theorem statements
theorem pos_count_a5_eq_2 : pos_count 5 = 2 := 
by
  -- Proof would go here
  sorry

theorem pos_pos_count_an_eq_n2 (n : Nat) : pos_pos_count n = n * n :=
by
  -- Proof would go here
  sorry

end pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l72_72049


namespace calculate_fraction_l72_72805

theorem calculate_fraction :
  (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end calculate_fraction_l72_72805


namespace not_both_hit_bullseye_probability_l72_72017

-- Definitions based on the conditions
def prob_A_bullseye : ℚ := 1 / 3
def prob_B_bullseye : ℚ := 1 / 2
def prob_both_hit_bullseye : ℚ := prob_A_bullseye * prob_B_bullseye

-- Statement of the proof problem
theorem not_both_hit_bullseye_probability : 1 - prob_both_hit_bullseye = 5 / 6 := by
  sorry

end not_both_hit_bullseye_probability_l72_72017


namespace rain_probability_weekend_l72_72955

/-- Prove that given the probabilities of rain on each day of a weekend, 
    the probability that it rains at least once during the weekend is approximately 82.675%. 
    The probabilities for Friday, Saturday, and Sunday are 0.30, 0.45, and 0.55, respectively, 
    and they are independent. -/
theorem rain_probability_weekend :
  let P_A := 0.30
  let P_B := 0.45
  let P_C := 0.55
  P (¬A) = 0.70 ∧ P (¬B) = 0.55 ∧ P (¬C) = 0.45 ∧
  independent [A, B, C] →
  P (A ∪ B ∪ C) ≈ 0.82675 := sorry

end rain_probability_weekend_l72_72955


namespace peanut_butter_candy_pieces_l72_72123

theorem peanut_butter_candy_pieces :
  ∀ (pb_candy grape_candy banana_candy : ℕ),
  pb_candy = 4 * grape_candy →
  grape_candy = banana_candy + 5 →
  banana_candy = 43 →
  pb_candy = 192 :=
by
  sorry

end peanut_butter_candy_pieces_l72_72123


namespace trig_identity_l72_72970

theorem trig_identity :
  sin (12 * Real.pi / 180) * sin (48 * Real.pi / 180) * sin (72 * Real.pi / 180) * sin (84 * Real.pi / 180) = 1 / 8 := sorry

end trig_identity_l72_72970


namespace b7_in_form_l72_72671

theorem b7_in_form (a : ℕ → ℚ) (b : ℕ → ℚ) : 
  a 0 = 3 → 
  b 0 = 5 → 
  (∀ n : ℕ, a (n + 1) = (a n)^2 / (b n)) → 
  (∀ n : ℕ, b (n + 1) = (b n)^2 / (a n)) → 
  b 7 = (5^50 : ℚ) / (3^41 : ℚ) := 
by 
  intros h1 h2 h3 h4 
  sorry

end b7_in_form_l72_72671


namespace people_at_table_l72_72697

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l72_72697


namespace probability_sum_div_by_3_l72_72281

/-- 
A bag contains 30 balls that are numbered 1 through 30. Two balls are randomly chosen from the bag. 
This theorem proves that the probability that the sum of the two numbers is divisible by 3 is 1/3.
-/
theorem probability_sum_div_by_3 :
  (∃ s : set ℕ, s = {1, 2, ..., 30} ∧
      (∀ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b → 
        (a + b) % 3 = 0 ↔ rational_number = 1/3)) :=
begin
  sorry
end

end probability_sum_div_by_3_l72_72281


namespace regular_hexagon_interior_angle_l72_72744

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l72_72744


namespace village_Y_initial_population_l72_72575

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end village_Y_initial_population_l72_72575


namespace second_polygon_sides_l72_72261

/--
Given two regular polygons where:
- The first polygon has 42 sides.
- Each side of the first polygon is three times the length of each side of the second polygon.
- The perimeters of both polygons are equal.
Prove that the second polygon has 126 sides.
-/
theorem second_polygon_sides
  (s : ℝ) -- the side length of the second polygon
  (h1 : ∃ n : ℕ, n = 42) -- the first polygon has 42 sides
  (h2 : ∃ m : ℝ, m = 3 * s) -- the side length of the first polygon is three times the side length of the second polygon
  (h3 : ∃ k : ℕ, k * (3 * s) = n * s) -- the perimeters of both polygons are equal
  : ∃ n2 : ℕ, n2 = 126 := 
by
  sorry

end second_polygon_sides_l72_72261


namespace max_y_midpoint_l72_72357

open Real

-- Definitions based on conditions in the problem
noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def tangent_line (m : ℝ) : ℝ → ℝ := λ x, exp m * (x - m) + exp m
noncomputable def y_coord_M (m : ℝ) : ℝ := (1 - m) * exp m
noncomputable def perp_line (m : ℝ) : ℝ → ℝ := λ x, -exp (-m) * (x - m) + exp m
noncomputable def y_coord_N (m : ℝ) : ℝ := exp m + m * exp (-m)
noncomputable def y_midpoint (m : ℝ) : ℝ := 0.5 * ((2 - m) * exp m + m * exp (-m))

-- The theorem statement
theorem max_y_midpoint : ∃ m : ℝ, y_midpoint m = 0.5 * (exp 1 + exp (-1)) :=
by
  use 1
  sorry

end max_y_midpoint_l72_72357


namespace total_carrots_l72_72531

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end total_carrots_l72_72531


namespace initial_overs_l72_72209

theorem initial_overs (x : ℝ) (r1 : ℝ) (r2 : ℝ) (target : ℝ) (overs_remaining : ℝ) :
  r1 = 3.2 ∧ overs_remaining = 22 ∧ r2 = 11.363636363636363 ∧ target = 282 ∧
  (r1 * x + r2 * overs_remaining = target) → x = 10 :=
by
  intro h
  obtain ⟨hr1, ho, hr2, ht, heq⟩ := h
  sorry

end initial_overs_l72_72209


namespace interior_angle_regular_hexagon_l72_72760

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l72_72760


namespace factorize_expression_l72_72163

variable (a b : ℝ)

theorem factorize_expression : (a - b)^2 + 6 * (b - a) + 9 = (a - b - 3)^2 :=
by
  sorry

end factorize_expression_l72_72163


namespace miranda_monthly_savings_l72_72866

noncomputable def total_cost := 260
noncomputable def sister_contribution := 50
noncomputable def months := 3

theorem miranda_monthly_savings : 
  (total_cost - sister_contribution) / months = 70 := 
by
  sorry

end miranda_monthly_savings_l72_72866


namespace Felicity_used_23_gallons_l72_72461

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l72_72461


namespace inequality_proof_l72_72220

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
by 
  sorry

end inequality_proof_l72_72220


namespace angle_A_area_triangle_l72_72505

-- The first problem: Proving angle A
theorem angle_A (a b c : ℝ) (A C : ℝ) 
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C) : 
  A = Real.pi / 3 :=
by sorry

-- The second problem: Finding the area of triangle ABC
theorem area_triangle (a b c : ℝ) (A : ℝ)
  (h1 : a = 3)
  (h2 : b = 2 * c)
  (h3 : A = Real.pi / 3) :
  0.5 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end angle_A_area_triangle_l72_72505


namespace geometric_seq_a5_a7_l72_72478

theorem geometric_seq_a5_a7 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 3 + a 5 = 6)
  (q : ℝ) :
  (a 5 + a 7 = 12) :=
sorry

end geometric_seq_a5_a7_l72_72478


namespace polar_circle_l72_72541

def is_circle (ρ θ : ℝ) : Prop :=
  ρ = Real.cos (Real.pi / 4 - θ)

theorem polar_circle : 
  ∀ ρ θ : ℝ, is_circle ρ θ ↔ ∃ (x y : ℝ), (x - 1/(2 * Real.sqrt 2))^2 + (y - 1/(2 * Real.sqrt 2))^2 = (1/(2 * Real.sqrt 2))^2 :=
by
  intro ρ θ
  sorry

end polar_circle_l72_72541


namespace interior_angle_of_regular_hexagon_is_120_degrees_l72_72766

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l72_72766


namespace AB_length_l72_72878

noncomputable def length_of_AB (x y : ℝ) (P_ratio Q_ratio : ℝ × ℝ) (PQ_distance : ℝ) : ℝ :=
    x + y

theorem AB_length (x y : ℝ) (P_ratio : ℝ × ℝ := (3, 5)) (Q_ratio : ℝ × ℝ := (4, 5)) (PQ_distance : ℝ := 3) 
    (h1 : 5 * x = 3 * y) -- P divides AB in the ratio 3:5
    (h2 : 5 * (x + 3) = 4 * (y - 3)) -- Q divides AB in the ratio 4:5 and PQ = 3 units
    : length_of_AB x y P_ratio Q_ratio PQ_distance = 43.2 := 
by sorry

end AB_length_l72_72878


namespace common_ratio_of_geometric_sequence_l72_72077

open BigOperators

theorem common_ratio_of_geometric_sequence
  (a1 : ℝ) (q : ℝ)
  (h1 : 2 * (a1 * q^5) = 3 * (a1 * (1 - q^4) / (1 - q)) + 1)
  (h2 : a1 * q^6 = 3 * (a1 * (1 - q^5) / (1 - q)) + 1)
  (h_pos : a1 > 0) :
  q = 3 :=
sorry

end common_ratio_of_geometric_sequence_l72_72077


namespace donna_pizza_slices_l72_72308

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end donna_pizza_slices_l72_72308


namespace aluminum_weight_proportional_l72_72011

noncomputable def area_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * side_length * Real.sqrt 3) / 4

theorem aluminum_weight_proportional (weight1 weight2 : ℝ) 
  (side_length1 side_length2 : ℝ)
  (h_density_thickness : ∀ s t, area_equilateral_triangle s * weight1 = area_equilateral_triangle t * weight2)
  (h_weight1 : weight1 = 20)
  (h_side_length1 : side_length1 = 2)
  (h_side_length2 : side_length2 = 4) : 
  weight2 = 80 :=
by
  sorry

end aluminum_weight_proportional_l72_72011


namespace evaluate_expression_l72_72974

theorem evaluate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 2 → (5 * x^(y + 1) + 6 * y^(x + 1) = 231) := by 
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l72_72974


namespace pizza_consumption_order_l72_72318

theorem pizza_consumption_order :
  let total_slices := 168
  let alex_slices := (1/6) * total_slices
  let beth_slices := (2/7) * total_slices
  let cyril_slices := (1/3) * total_slices
  let eve_slices_initial := (1/8) * total_slices
  let dan_slices_initial := total_slices - (alex_slices + beth_slices + cyril_slices + eve_slices_initial)
  let eve_slices := eve_slices_initial + 2
  let dan_slices := dan_slices_initial - 2
  (cyril_slices > beth_slices ∧ beth_slices > eve_slices ∧ eve_slices > alex_slices ∧ alex_slices > dan_slices) :=
  sorry

end pizza_consumption_order_l72_72318


namespace dinosaur_count_l72_72787

theorem dinosaur_count (h : ℕ) (l : ℕ) (H1 : h = 1) (H2 : l = 3) (total_hl : ℕ) (H3 : total_hl = 20) :
  ∃ D : ℕ, 4 * D = total_hl := 
by
  use 5
  sorry

end dinosaur_count_l72_72787


namespace sequence_bound_l72_72830

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l72_72830


namespace MrsHiltTravelMiles_l72_72375

theorem MrsHiltTravelMiles
  (one_book_miles : ℕ)
  (finished_books : ℕ)
  (total_miles : ℕ)
  (h1 : one_book_miles = 450)
  (h2 : finished_books = 15)
  (h3 : total_miles = one_book_miles * finished_books) :
  total_miles = 6750 :=
by
  sorry

end MrsHiltTravelMiles_l72_72375


namespace common_difference_of_consecutive_multiples_l72_72733

/-- The sides of a rectangular prism are consecutive multiples of a certain number n. The base area is 450.
    Prove that the common difference between the consecutive multiples is 15. -/
theorem common_difference_of_consecutive_multiples (n d : ℕ) (h₁ : n * (n + d) = 450) : d = 15 :=
sorry

end common_difference_of_consecutive_multiples_l72_72733


namespace harry_total_cost_in_silver_l72_72846

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end harry_total_cost_in_silver_l72_72846


namespace pie_count_correct_l72_72807

structure Berries :=
  (strawberries : ℕ)
  (blueberries : ℕ)
  (raspberries : ℕ)

def christine_picking : Berries := {strawberries := 10, blueberries := 8, raspberries := 20}

def rachel_picking : Berries :=
  let c := christine_picking
  {strawberries := 2 * c.strawberries,
   blueberries := 2 * c.blueberries,
   raspberries := c.raspberries / 2}

def total_berries (b1 b2 : Berries) : Berries :=
  {strawberries := b1.strawberries + b2.strawberries,
   blueberries := b1.blueberries + b2.blueberries,
   raspberries := b1.raspberries + b2.raspberries}

def pie_requirements : Berries := {strawberries := 3, blueberries := 2, raspberries := 4}

def max_pies (total : Berries) (requirements : Berries) : Berries :=
  {strawberries := total.strawberries / requirements.strawberries,
   blueberries := total.blueberries / requirements.blueberries,
   raspberries := total.raspberries / requirements.raspberries}

def correct_pies : Berries := {strawberries := 10, blueberries := 12, raspberries := 7}

theorem pie_count_correct :
  let total := total_berries christine_picking rachel_picking;
  max_pies total pie_requirements = correct_pies :=
by {
  sorry
}

end pie_count_correct_l72_72807


namespace problem_statement_l72_72340

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l72_72340


namespace total_additions_and_multiplications_l72_72606

def f(x : ℝ) : ℝ := 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem total_additions_and_multiplications {x : ℝ} (h : x = 0.6) :
  let horner_f := ((((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x + 7)
  (horner_f = f x) ∧ (6 + 6 = 12) :=
by
  sorry

end total_additions_and_multiplications_l72_72606


namespace mean_of_first_set_is_67_l72_72395

theorem mean_of_first_set_is_67 (x : ℝ) 
  (h : (50 + 62 + 97 + 124 + x) / 5 = 75.6) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 := 
by
  sorry

end mean_of_first_set_is_67_l72_72395


namespace add_to_fraction_l72_72928

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l72_72928


namespace pipe_fill_time_without_leakage_l72_72793

theorem pipe_fill_time_without_leakage (t : ℕ) (h1 : 7 * t * (1/t - 1/70) = 1) : t = 60 :=
by
  sorry

end pipe_fill_time_without_leakage_l72_72793


namespace rainfall_difference_l72_72566

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l72_72566


namespace pascal_triangle_21st_number_l72_72405

theorem pascal_triangle_21st_number 
: (Nat.choose 22 2) = 231 :=
by 
  sorry

end pascal_triangle_21st_number_l72_72405


namespace construct_1_degree_l72_72267

def canConstruct1DegreeUsing19Degree : Prop :=
  ∃ (n : ℕ), n * 19 = 360 + 1

theorem construct_1_degree (h : ∃ (x : ℕ), x * 19 = 360 + 1) : canConstruct1DegreeUsing19Degree := by
  sorry

end construct_1_degree_l72_72267


namespace number_of_integer_pairs_l72_72849

theorem number_of_integer_pairs (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_ineq : m^2 + m * n < 30) :
  ∃ k : ℕ, k = 48 :=
sorry

end number_of_integer_pairs_l72_72849


namespace estimate_num_2016_digit_squares_l72_72310

noncomputable def num_estimate_2016_digit_squares : ℕ := 2016

theorem estimate_num_2016_digit_squares :
  let t1 := (10 ^ (2016 / 2) - 10 ^ (2015 / 2) - 1)
  let t2 := (2017 ^ 10)
  let result := t1 / t2
  t1 > 10 ^ 1000 → 
  result > 10 ^ 900 →
  result == num_estimate_2016_digit_squares :=
by
  intros
  sorry

end estimate_num_2016_digit_squares_l72_72310


namespace swim_back_distance_l72_72287

variables (swimming_speed_still_water : ℝ) (water_speed : ℝ) (time_back : ℝ) (distance_back : ℝ)

theorem swim_back_distance :
  swimming_speed_still_water = 12 → 
  water_speed = 10 → 
  time_back = 4 →
  distance_back = (swimming_speed_still_water - water_speed) * time_back →
  distance_back = 8 :=
by
  intros swimming_speed_still_water_eq water_speed_eq time_back_eq distance_back_eq
  have swim_speed : (swimming_speed_still_water - water_speed) = 2 := by sorry
  rw [swim_speed, time_back_eq] at distance_back_eq
  sorry

end swim_back_distance_l72_72287


namespace regular_hexagon_interior_angle_measure_l72_72749

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l72_72749


namespace slices_left_for_lunch_tomorrow_l72_72307

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end slices_left_for_lunch_tomorrow_l72_72307


namespace ruda_received_clock_on_correct_date_l72_72684

/-- Ruda's clock problem -/
def ruda_clock_problem : Prop :=
  ∃ receive_date : ℕ → ℕ × ℕ × ℕ, -- A function mapping the number of presses to a date (Year, Month, Day)
  (∀ days_after_received, 
    receive_date days_after_received = 
    if days_after_received <= 45 then (2022, 10, 27 - (45 - days_after_received)) -- Calculating the receive date.
    else receive_date 45)
  ∧
  receive_date 45 = (2022, 12, 11) -- The day he checked the clock has to be December 11th

-- We want to prove that:
theorem ruda_received_clock_on_correct_date : ruda_clock_problem :=
by
  sorry

end ruda_received_clock_on_correct_date_l72_72684


namespace number_in_2019th_field_l72_72244

theorem number_in_2019th_field (f : ℕ → ℕ) (h1 : ∀ n, 0 < f n) (h2 : ∀ n, f n * f (n+1) * f (n+2) = 2018) :
  f 2018 = 1009 := sorry

end number_in_2019th_field_l72_72244


namespace cherry_orange_punch_ratio_l72_72388

theorem cherry_orange_punch_ratio 
  (C : ℝ)
  (h_condition1 : 4.5 + C + (C - 1.5) = 21) : 
  C / 4.5 = 2 :=
by
  sorry

end cherry_orange_punch_ratio_l72_72388


namespace total_cookies_l72_72142

def total_chocolate_chip_batches := 5
def cookies_per_chocolate_chip_batch := 8
def total_oatmeal_batches := 3
def cookies_per_oatmeal_batch := 7
def total_sugar_batches := 1
def cookies_per_sugar_batch := 10
def total_double_chocolate_batches := 1
def cookies_per_double_chocolate_batch := 6

theorem total_cookies : 
  (total_chocolate_chip_batches * cookies_per_chocolate_chip_batch) +
  (total_oatmeal_batches * cookies_per_oatmeal_batch) +
  (total_sugar_batches * cookies_per_sugar_batch) +
  (total_double_chocolate_batches * cookies_per_double_chocolate_batch) = 77 :=
by sorry

end total_cookies_l72_72142


namespace tree_current_height_l72_72520

theorem tree_current_height 
  (growth_rate_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_height_after_4_months : ℕ) 
  (growth_rate_per_week_eq : growth_rate_per_week = 2)
  (weeks_per_month_eq : weeks_per_month = 4)
  (total_height_after_4_months_eq : total_height_after_4_months = 42) : 
  (∃ (current_height : ℕ), current_height = 10) :=
by
  sorry

end tree_current_height_l72_72520


namespace joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l72_72086

noncomputable def distance_joseph : ℝ := 48 * 2.5 + 60 * 1.5
noncomputable def distance_kyle : ℝ := 70 * 2 + 63 * 2.5
noncomputable def distance_emily : ℝ := 65 * 3

theorem joseph_vs_kyle : distance_joseph - distance_kyle = -87.5 := by
  unfold distance_joseph
  unfold distance_kyle
  sorry

theorem emily_vs_joseph : distance_emily - distance_joseph = -15 := by
  unfold distance_emily
  unfold distance_joseph
  sorry

theorem emily_vs_kyle : distance_emily - distance_kyle = -102.5 := by
  unfold distance_emily
  unfold distance_kyle
  sorry

end joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l72_72086


namespace percent_greater_than_average_l72_72492

variable (M N : ℝ)

theorem percent_greater_than_average (h : M > N) :
  (200 * (M - N)) / (M + N) = ((M - ((M + N) / 2)) / ((M + N) / 2)) * 100 :=
by 
  sorry

end percent_greater_than_average_l72_72492


namespace symmetric_line_eq_l72_72243

theorem symmetric_line_eq (x y : ℝ) :  
  (x - 2 * y + 3 = 0) → (x + 2 * y + 3 = 0) :=
sorry

end symmetric_line_eq_l72_72243


namespace C_necessary_but_not_sufficient_for_A_l72_72055

variable {A B C : Prop}

-- Given conditions
def sufficient_not_necessary (h : A → B) (hn : ¬(B → A)) := h
def necessary_sufficient := B ↔ C

-- Prove that C is a necessary but not sufficient condition for A
theorem C_necessary_but_not_sufficient_for_A (h₁ : A → B) (hn : ¬(B → A)) (h₂ : B ↔ C) : (C → A) ∧ ¬(A → C) :=
  by
  sorry

end C_necessary_but_not_sufficient_for_A_l72_72055


namespace book_purchase_schemes_l72_72592

theorem book_purchase_schemes :
  let num_schemes (a b c : ℕ) := 500 = 30 * a + 25 * b + 20 * c
  in
  (∑ a in {5, 6}, ∑ b in {b | ∃ c, b > 0 ∧ c > 0 ∧ num_schemes a b c} ) = 6 :=
by sorry

end book_purchase_schemes_l72_72592


namespace max_goods_purchased_l72_72010

theorem max_goods_purchased (initial_spend : ℕ) (reward_rate : ℕ → ℕ → ℕ) (continuous_reward : Prop) :
  initial_spend = 7020 →
  (∀ x y, reward_rate x y = (x / y) * 20) →
  continuous_reward →
  initial_spend + reward_rate initial_spend 100 + reward_rate (reward_rate initial_spend 100) 100 + 
  reward_rate (reward_rate (reward_rate initial_spend 100) 100) 100 = 8760 :=
by
  intros h1 h2 h3
  sorry

end max_goods_purchased_l72_72010


namespace recurring_decimal_to_fraction_l72_72999

theorem recurring_decimal_to_fraction (a b : ℕ) (ha : a = 356) (hb : b = 999) (hab_gcd : Nat.gcd a b = 1)
  (x : ℚ) (hx : x = 356 / 999) 
  (hx_recurring : x = {num := 356, den := 999}): a + b = 1355 :=
by
  sorry  -- Proof is not required as per the instructions

end recurring_decimal_to_fraction_l72_72999


namespace find_q_l72_72646

theorem find_q (p q : ℝ) (h : ∀ x : ℝ, (x^2 + p * x + q) ≥ 1) : q = 1 + (p^2 / 4) :=
sorry

end find_q_l72_72646


namespace find_a_plus_b_l72_72054

theorem find_a_plus_b (a b : ℝ) 
  (h_a : a^3 - 3 * a^2 + 5 * a = 1) 
  (h_b : b^3 - 3 * b^2 + 5 * b = 5) : 
  a + b = 2 := 
sorry

end find_a_plus_b_l72_72054


namespace combined_gold_cost_l72_72822

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end combined_gold_cost_l72_72822


namespace prove_f_neg1_eq_0_l72_72489

def f : ℝ → ℝ := sorry

theorem prove_f_neg1_eq_0
  (h1 : ∀ x : ℝ, f(x + 2) = f(2 - x))
  (h2 : ∀ x : ℝ, f(1 - 2 * x) = -f(2 * x + 1))
  : f(-1) = 0 := sorry

end prove_f_neg1_eq_0_l72_72489


namespace Felicity_used_23_gallons_l72_72463

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l72_72463


namespace minimum_value_of_x_plus_y_l72_72497

-- Define the conditions as a hypothesis and the goal theorem statement.
theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 9 / y = 1) :
  x + y = 16 :=
by
  sorry

end minimum_value_of_x_plus_y_l72_72497


namespace corresponding_side_of_larger_triangle_l72_72392

-- Conditions
variables (A1 A2 : ℕ) (s1 s2 : ℕ)
-- A1 is the area of the larger triangle
-- A2 is the area of the smaller triangle
-- s1 is a side of the smaller triangle = 4 feet
-- s2 is the corresponding side of the larger triangle

-- Given conditions as hypotheses
axiom diff_in_areas : A1 - A2 = 32
axiom ratio_of_areas : A1 = 9 * A2
axiom side_of_smaller_triangle : s1 = 4

-- Theorem to prove the corresponding side of the larger triangle
theorem corresponding_side_of_larger_triangle 
  (h1 : A1 - A2 = 32)
  (h2 : A1 = 9 * A2)
  (h3 : s1 = 4) : 
  s2 = 12 :=
sorry

end corresponding_side_of_larger_triangle_l72_72392


namespace complement_angle_l72_72826

theorem complement_angle (A : ℝ) (hA : A = 35) : 90 - A = 55 := by
  sorry

end complement_angle_l72_72826


namespace symmetric_function_expression_l72_72840

variable (f : ℝ → ℝ)
variable (h_sym : ∀ x y, f (-2 - x) = - f x)
variable (h_def : ∀ x, 0 < x → f x = 1 / x)

theorem symmetric_function_expression : ∀ x, x < -2 → f x = 1 / (2 + x) :=
by
  intro x
  intro hx
  sorry

end symmetric_function_expression_l72_72840


namespace days_required_for_C_l72_72938

noncomputable def rate_A (r_A r_B r_C : ℝ) : Prop := r_A + r_B = 1 / 3
noncomputable def rate_B (r_A r_B r_C : ℝ) : Prop := r_B + r_C = 1 / 6
noncomputable def rate_C (r_A r_B r_C : ℝ) : Prop := r_C + r_A = 1 / 4
noncomputable def days_for_C (r_C : ℝ) : ℝ := 1 / r_C

theorem days_required_for_C
  (r_A r_B r_C : ℝ)
  (h1 : rate_A r_A r_B r_C)
  (h2 : rate_B r_A r_B r_C)
  (h3 : rate_C r_A r_B r_C) :
  days_for_C r_C = 4.8 :=
sorry

end days_required_for_C_l72_72938


namespace problem_proof_l72_72302

variable {α : Type*}
noncomputable def op (a b : ℝ) : ℝ := 1/a + 1/b
theorem problem_proof (a b : ℝ) (h : op a (-b) = 2) : (3 * a * b) / (2 * a - 2 * b) = -3/4 :=
by
  sorry

end problem_proof_l72_72302


namespace num_girls_at_park_l72_72257

theorem num_girls_at_park (G : ℕ) (h1 : 11 + 50 + G = 3 * 25) : G = 14 := by
  sorry

end num_girls_at_park_l72_72257


namespace parabola_focus_directrix_distance_l72_72239

theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), 
    y^2 = 8 * x → 
    ∃ p : ℝ, 2 * p = 8 ∧ p = 4 := by
  sorry

end parabola_focus_directrix_distance_l72_72239


namespace sampled_students_within_interval_l72_72510

/-- Define the conditions for the student's problem --/
def student_count : ℕ := 1221
def sampled_students : ℕ := 37
def sampling_interval : ℕ := student_count / sampled_students
def interval_lower_bound : ℕ := 496
def interval_upper_bound : ℕ := 825
def interval_range : ℕ := interval_upper_bound - interval_lower_bound + 1

/-- State the goal within the above conditions --/
theorem sampled_students_within_interval :
  interval_range / sampling_interval = 10 :=
sorry

end sampled_students_within_interval_l72_72510


namespace find_f_2_solve_inequality_l72_72862

noncomputable def f : ℝ → ℝ :=
  sorry -- definition of f cannot be constructed without further info

axiom f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → (x ≤ y → f x ≥ f y)

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f x + f y - 1

axiom f_4 : f 4 = 5

theorem find_f_2 : f 2 = 3 :=
  sorry

theorem solve_inequality (m : ℝ) (h : f (m - 2) ≤ 3) : m ≥ 4 :=
  sorry

end find_f_2_solve_inequality_l72_72862


namespace quadratic_inequality_l72_72387

theorem quadratic_inequality : ∀ x : ℝ, -7 * x ^ 2 + 4 * x - 6 < 0 :=
by
  intro x
  have delta : 4 ^ 2 - 4 * (-7) * (-6) = -152 := by norm_num
  have neg_discriminant : -152 < 0 := by norm_num
  have coef : -7 < 0 := by norm_num
  sorry

end quadratic_inequality_l72_72387


namespace roots_of_quadratic_l72_72552

theorem roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -2 ∧ c = -6) :
  let Δ := b^2 - 4 * a * c in
  Δ > 0 :=
by
  obtain ⟨ha, hb, hc⟩ := h_eq
  have Δ_def : Δ = b^2 - 4 * a * c := rfl
  rw [ha, hb, hc] at Δ_def
  sorry

end roots_of_quadratic_l72_72552


namespace candy_cost_l72_72296

theorem candy_cost
    (grape_candies : ℕ)
    (cherry_candies : ℕ)
    (apple_candies : ℕ)
    (total_cost : ℝ)
    (total_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 24)
    (h2 : grape_candies = 3 * cherry_candies)
    (h3 : apple_candies = 2 * grape_candies)
    (h4 : total_cost = 200)
    (h5 : total_candies = cherry_candies + grape_candies + apple_candies)
    (h6 : cost_per_candy = total_cost / total_candies) :
    cost_per_candy = 2.50 :=
by
    sorry

end candy_cost_l72_72296


namespace find_positive_solutions_l72_72464

noncomputable def satisfies_eq1 (x y : ℝ) : Prop :=
  2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0

noncomputable def satisfies_eq2 (x y : ℝ) : Prop :=
  2 * x^2 + x^2 * y^4 = 18 * y^2

theorem find_positive_solutions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  satisfies_eq1 x y ∧ satisfies_eq2 x y ↔ 
  (x = 2 ∧ y = 2) ∨ 
  (x = Real.sqrt 286^(1/4) / 4 ∧ y = Real.sqrt 286^(1/4)) :=
sorry

end find_positive_solutions_l72_72464


namespace geometric_sequence_a3_is_15_l72_72853

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
a1 * q^(n - 1)

theorem geometric_sequence_a3_is_15 (q : ℝ) (a1 : ℝ) (a5 : ℝ) 
  (h1 : a1 = 3) (h2 : a5 = 75) (h_seq : ∀ n, a5 = geometric_sequence a1 q n) :
  geometric_sequence a1 q 3 = 15 :=
by 
  sorry

end geometric_sequence_a3_is_15_l72_72853


namespace quadratic_radical_type_l72_72179

-- Problem statement: Given that sqrt(2a + 1) is a simplest quadratic radical and the same type as sqrt(48), prove that a = 1.

theorem quadratic_radical_type (a : ℝ) (h1 : ((2 * a) + 1) = 3) : a = 1 :=
by
  sorry

end quadratic_radical_type_l72_72179


namespace sum_three_numbers_l72_72936

theorem sum_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 23 := by
  sorry

end sum_three_numbers_l72_72936


namespace total_bill_is_correct_l72_72440

def number_of_adults : ℕ := 2
def number_of_children : ℕ := 5
def meal_cost : ℕ := 8

-- Define total number of people
def total_people : ℕ := number_of_adults + number_of_children

-- Define the total bill
def total_bill : ℕ := total_people * meal_cost

-- Theorem stating the total bill amount
theorem total_bill_is_correct : total_bill = 56 := by
  sorry

end total_bill_is_correct_l72_72440


namespace blocks_difference_l72_72381

def blocks_house := 89
def blocks_tower := 63

theorem blocks_difference : (blocks_house - blocks_tower = 26) :=
by sorry

end blocks_difference_l72_72381


namespace percentage_increase_is_20_percent_l72_72384

noncomputable def originalSalary : ℝ := 575 / 1.15
noncomputable def increasedSalary : ℝ := 600
noncomputable def percentageIncreaseTo600 : ℝ := (increasedSalary - originalSalary) / originalSalary * 100

theorem percentage_increase_is_20_percent :
  percentageIncreaseTo600 = 20 := 
by
  sorry -- The proof will go here

end percentage_increase_is_20_percent_l72_72384


namespace trapezium_area_example_l72_72617

noncomputable def trapezium_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

theorem trapezium_area_example :
  trapezium_area 20 18 16 = 304 :=
by
  -- The proof steps would go here, but we're skipping them.
  sorry

end trapezium_area_example_l72_72617


namespace find_h_l72_72504

def quadratic_expr : ℝ → ℝ := λ x, 3 * x^2 + 9 * x + 20

theorem find_h : ∃ (a k : ℝ) (h : ℝ), 
  (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 :=
sorry

end find_h_l72_72504


namespace merry_go_round_times_l72_72966

theorem merry_go_round_times
  (dave_time : ℕ := 10)
  (chuck_multiplier : ℕ := 5)
  (erica_increase : ℕ := 30) : 
  let chuck_time := chuck_multiplier * dave_time,
      erica_time := chuck_time + (erica_increase * chuck_time / 100)
  in erica_time = 65 :=
by 
  let dave_time := 10
  let chuck_multiplier := 5
  let erica_increase := 30
  let chuck_time := chuck_multiplier * dave_time
  let erica_time := chuck_time + (erica_increase * chuck_time / 100)
  exact Nat.succ 64 -- directly providing the evaluated result to match the problem statement specification

end merry_go_round_times_l72_72966


namespace tan_alpha_l72_72637

theorem tan_alpha (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan α = 2 / 9 := by
  sorry

end tan_alpha_l72_72637


namespace purchasing_schemes_l72_72594

-- Define the cost of each type of book
def cost_A : ℕ := 30
def cost_B : ℕ := 25
def cost_C : ℕ := 20

-- Define the total budget available
def budget : ℕ := 500

-- Define the range of type A books that must be bought
def min_A : ℕ := 5
def max_A : ℕ := 6

-- Condition that all three types of books must be purchased
def all_types_purchased (A B C : ℕ) : Prop := A > 0 ∧ B > 0 ∧ C > 0

-- Condition that calculates the total cost
def total_cost (A B C : ℕ) : ℕ := cost_A * A + cost_B * B + cost_C * C

theorem purchasing_schemes :
  (∑ A in finset.range (max_A + 1), 
    if min_A ≤ A ∧ all_types_purchased A B C ∧ total_cost A B C = budget 
    then 1 else 0) = 6 :=
by {
  sorry
}

end purchasing_schemes_l72_72594


namespace symmetric_line_eq_l72_72355

theorem symmetric_line_eq (x y: ℝ) :
    (∃ (a b: ℝ), 3 * a - b + 2 = 0 ∧ a = 2 - x ∧ b = 2 - y) → 3 * x - y - 6 = 0 :=
by
    intro h
    sorry

end symmetric_line_eq_l72_72355


namespace sqrt_25_eq_pm_5_l72_72254

theorem sqrt_25_eq_pm_5 : {x : ℝ | x^2 = 25} = {5, -5} :=
by
  sorry

end sqrt_25_eq_pm_5_l72_72254


namespace problem1_problem2_l72_72322

-- Define the triangle and the condition a + 2a * cos B = c
variable {A B C : ℝ} (a b c : ℝ)
variable (cos_B : ℝ) -- cosine of angle B

-- Condition: a + 2a * cos B = c
variable (h1 : a + 2 * a * cos_B = c)

-- (I) Prove B = 2A
theorem problem1 (h1 : a + 2 * a * cos_B = c) : B = 2 * A :=
sorry

-- Define the acute triangle condition
variable (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Given: c = 2
variable (h2 : c = 2)

-- (II) Determine the range for a if the triangle is acute and c = 2
theorem problem2 (h1 : a + 2 * a * cos_B = 2) (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : 1 < a ∧ a < 2 :=
sorry

end problem1_problem2_l72_72322


namespace add_to_fraction_l72_72924

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l72_72924


namespace sum_of_squares_bounds_l72_72734

theorem sum_of_squares_bounds (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 10) : 
  (x^2 + y^2 ≤ 100) ∧ (x^2 + y^2 ≥ 50) :=
by 
  sorry

end sum_of_squares_bounds_l72_72734


namespace solution_set_inequality_l72_72058

theorem solution_set_inequality (a b x : ℝ) (h₀ : {x : ℝ | ax - b < 0} = {x : ℝ | 1 < x}) :
  {x : ℝ | (ax + b) * (x - 3) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_inequality_l72_72058


namespace smallest_possible_value_of_a_l72_72250

theorem smallest_possible_value_of_a (a b : ℕ) :
  (∃ (r1 r2 r3 : ℕ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    (r1 * r2 * r3 = 2310) ∧
    (a = r1 + r2 + r3)) →
  a = 52 :=
begin
  sorry
end

end smallest_possible_value_of_a_l72_72250


namespace sin_product_identity_l72_72969

theorem sin_product_identity :
  sin (12 * Real.pi / 180) * sin (48 * Real.pi / 180) * sin (72 * Real.pi / 180) * sin (84 * Real.pi / 180) =
  (1 / 8) * (1 + cos (24 * Real.pi / 180)) :=
sorry

end sin_product_identity_l72_72969


namespace total_trees_l72_72854

theorem total_trees (apricot_trees : ℕ) (peach_mult : ℕ) (peach_trees : ℕ) (total_trees : ℕ) :
  apricot_trees = 58 → peach_mult = 3 → peach_trees = peach_mult * apricot_trees → total_trees = apricot_trees + peach_trees → total_trees = 232 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw h3 at h4
  rw h4
  exact rfl

end total_trees_l72_72854


namespace table_seating_problem_l72_72701

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l72_72701


namespace vasya_made_a_mistake_l72_72262

theorem vasya_made_a_mistake (A B V G D E : ℕ)
  (h1 : A ≠ B)
  (h2 : V ≠ G)
  (h3 : (10 * A + B) * (10 * V + G) = 1000 * D + 100 * D + 10 * E + E)
  (h4 : ∀ {X Y : ℕ}, X ≠ Y → D ≠ E) :
  False :=
by
  -- Proof goes here (skipped)
  sorry

end vasya_made_a_mistake_l72_72262


namespace isosceles_right_triangle_area_l72_72116

theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) :
  (h = 5 * Real.sqrt 2) →
  (A = 12.5) →
  ∃ (leg : ℝ), (leg = 5) ∧ (A = 1 / 2 * leg^2) := by
  sorry

end isosceles_right_triangle_area_l72_72116


namespace problem_statement_l72_72997

-- Define the repeating decimal and the required gcd condition
def repeating_decimal_value := (356 : ℚ) / 999
def gcd_condition (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the main theorem stating the required sum
theorem problem_statement (a b : ℕ) 
                          (h_a : a = 356) 
                          (h_b : b = 999) 
                          (h_gcd : gcd_condition a b) : 
    a + b = 1355 := by
  sorry

end problem_statement_l72_72997


namespace solve_fractional_eq_l72_72959

theorem solve_fractional_eq (x : ℝ) (h₀ : x ≠ 2) (h₁ : x ≠ -2) :
  (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) → (x = 3 / 2) :=
by sorry

end solve_fractional_eq_l72_72959


namespace find_number_to_add_l72_72917

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l72_72917


namespace number_of_people_seated_l72_72707

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l72_72707


namespace expression_divisible_by_7_l72_72412

theorem expression_divisible_by_7 (n : ℕ) (hn : n > 0) :
  7 ∣ (3^(3*n+1) + 5^(3*n+2) + 7^(3*n+3)) :=
sorry

end expression_divisible_by_7_l72_72412


namespace victor_weight_is_correct_l72_72785

-- Define the given conditions
def bear_daily_food : ℕ := 90
def victors_food_in_3_weeks : ℕ := 15
def days_in_3_weeks : ℕ := 21

-- Define the equivalent weight of Victor based on the given conditions
def victor_weight : ℕ := bear_daily_food * days_in_3_weeks / victors_food_in_3_weeks

-- Prove that the weight of Victor is 126 pounds
theorem victor_weight_is_correct : victor_weight = 126 := by
  sorry

end victor_weight_is_correct_l72_72785


namespace at_least_two_equal_l72_72469

-- Define the problem
theorem at_least_two_equal (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x^2 / y) + (y^2 / z) + (z^2 / x) = (x^2 / z) + (y^2 / x) + (z^2 / y)) :
  x = y ∨ y = z ∨ z = x := 
by 
  sorry

end at_least_two_equal_l72_72469


namespace average_speed_of_train_b_l72_72570

-- Given conditions
def distance_between_trains_initially := 13
def speed_of_train_a := 37
def time_to_overtake := 5
def distance_a_in_5_hours := speed_of_train_a * time_to_overtake
def distance_b_to_overtake := distance_between_trains_initially + distance_a_in_5_hours + 17

-- Prove: The average speed of Train B
theorem average_speed_of_train_b : 
  ∃ v_B, v_B = distance_b_to_overtake / time_to_overtake ∧ v_B = 43 :=
by
  -- The proof should go here, but we use sorry to skip it.
  sorry

end average_speed_of_train_b_l72_72570


namespace tan_alpha_plus_pi_over_4_l72_72842

theorem tan_alpha_plus_pi_over_4 (x y : ℝ) (h1 : 3 * x + 4 * y = 0) : 
  Real.tan ((Real.arctan (- 3 / 4)) + π / 4) = 1 / 7 := 
by
  sorry

end tan_alpha_plus_pi_over_4_l72_72842


namespace parallel_lines_determine_plane_l72_72957

def determine_plane_by_parallel_lines := 
  let condition_4 := true -- Two parallel lines
  condition_4 = true

theorem parallel_lines_determine_plane : determine_plane_by_parallel_lines = true :=
by 
  sorry

end parallel_lines_determine_plane_l72_72957


namespace probability_is_3888_over_7533_l72_72625

noncomputable def probability_odd_sum_given_even_product : ℚ := 
  let total_outcomes := 6^5
  let all_odd_outcomes := 3^5
  let at_least_one_even_outcomes := total_outcomes - all_odd_outcomes
  let favorable_outcomes := 5 * 3^4 + 10 * 3^4 + 3^5
  favorable_outcomes / at_least_one_even_outcomes

theorem probability_is_3888_over_7533 :
  probability_odd_sum_given_even_product = 3888 / 7533 := 
sorry

end probability_is_3888_over_7533_l72_72625


namespace cannot_fit_all_pictures_l72_72888

theorem cannot_fit_all_pictures 
  (typeA_capacity : Nat) (typeB_capacity : Nat) (typeC_capacity : Nat)
  (typeA_count : Nat) (typeB_count : Nat) (typeC_count : Nat)
  (total_pictures : Nat)
  (h1 : typeA_capacity = 12)
  (h2 : typeB_capacity = 18)
  (h3 : typeC_capacity = 24)
  (h4 : typeA_count = 6)
  (h5 : typeB_count = 4)
  (h6 : typeC_count = 3)
  (h7 : total_pictures = 480) :
  (typeA_capacity * typeA_count + typeB_capacity * typeB_count + typeC_capacity * typeC_count < total_pictures) :=
  by sorry

end cannot_fit_all_pictures_l72_72888


namespace white_paint_amount_is_correct_l72_72863

noncomputable def totalAmountOfPaint (bluePaint: ℝ) (bluePercentage: ℝ): ℝ :=
  bluePaint / bluePercentage

noncomputable def whitePaintAmount (totalPaint: ℝ) (whitePercentage: ℝ): ℝ :=
  totalPaint * whitePercentage

theorem white_paint_amount_is_correct (bluePaint: ℝ) (bluePercentage: ℝ) (whitePercentage: ℝ) (totalPaint: ℝ) :
  bluePaint = 140 → bluePercentage = 0.7 → whitePercentage = 0.1 → totalPaint = totalAmountOfPaint 140 0.7 →
  whitePaintAmount totalPaint 0.1 = 20 :=
by
  intros
  sorry

end white_paint_amount_is_correct_l72_72863


namespace interior_angle_of_regular_hexagon_l72_72757

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l72_72757


namespace find_a_l72_72645

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a_l72_72645


namespace green_peaches_per_basket_l72_72121

/-- Define the conditions given in the problem. -/
def n_baskets : ℕ := 7
def n_red_each : ℕ := 10
def n_green_total : ℕ := 14

/-- Prove that there are 2 green peaches in each basket. -/
theorem green_peaches_per_basket : n_green_total / n_baskets = 2 := by
  sorry

end green_peaches_per_basket_l72_72121


namespace binary_addition_subtraction_l72_72148

def bin_10101 : ℕ := 0b10101
def bin_1011 : ℕ := 0b1011
def bin_1110 : ℕ := 0b1110
def bin_110001 : ℕ := 0b110001
def bin_1101 : ℕ := 0b1101
def bin_101100 : ℕ := 0b101100

theorem binary_addition_subtraction :
  bin_10101 + bin_1011 + bin_1110 + bin_110001 - bin_1101 = bin_101100 := 
sorry

end binary_addition_subtraction_l72_72148


namespace interior_angle_of_regular_hexagon_l72_72769

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l72_72769


namespace fewest_students_possible_l72_72947

theorem fewest_students_possible : 
  ∃ n : ℕ, n % 3 = 1 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ ∀ m, m % 3 = 1 ∧ m % 6 = 4 ∧ m % 8 = 5 → n ≤ m := 
by
  sorry

end fewest_students_possible_l72_72947


namespace calculate_expression_l72_72026

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := 
by
  sorry

end calculate_expression_l72_72026


namespace quadratic_prob_correct_l72_72109

noncomputable def quadratic_real_roots_probability : ℝ :=
let interval : Set ℝ := Set.Icc 0 5 in
let valid_p : Set ℝ := {p | p ∈ interval ∧ p^2 - 4 ≥ 0} in
MeasureTheory.Measure.count (valid_p) / MeasureTheory.Measure.count (interval)

theorem quadratic_prob_correct :
  quadratic_real_roots_probability = 3 / 5 :=
by
  sorry

end quadratic_prob_correct_l72_72109


namespace find_k_l72_72294

-- Define the conditions
variables (k : ℝ) -- the variable k
variables (x1 : ℝ) -- x1 coordinate of point A on the graph y = k/x
variable (AREA_ABCD : ℝ := 10) -- the area of the quadrilateral ABCD

-- The statement to be proven
theorem find_k (k : ℝ) (h1 : ∀ x1 : ℝ, (0 < x1 ∧ 2 * abs k = AREA_ABCD → x1 * abs k * 2 = AREA_ABCD)) : k = -5 :=
sorry

end find_k_l72_72294


namespace principal_amount_l72_72932

theorem principal_amount (SI P R T : ℝ) 
  (h1 : R = 12) (h2 : T = 3) (h3 : SI = 3600) : 
  SI = P * R * T / 100 → P = 10000 :=
by
  intros h
  sorry

end principal_amount_l72_72932


namespace polynomial_perfect_square_value_of_k_l72_72350

noncomputable def is_perfect_square (p : Polynomial ℝ) : Prop :=
  ∃ (q : Polynomial ℝ), p = q^2

theorem polynomial_perfect_square_value_of_k {k : ℝ} :
  is_perfect_square (Polynomial.X^2 - Polynomial.C k * Polynomial.X + Polynomial.C 25) ↔ (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_value_of_k_l72_72350


namespace sqrt_div_sqrt_eq_sqrt_fraction_l72_72037

theorem sqrt_div_sqrt_eq_sqrt_fraction
  (x y : ℝ)
  (h : ((1 / 2) ^ 2 + (1 / 3) ^ 2) / ((1 / 3) ^ 2 + (1 / 6) ^ 2) = 13 * x / (47 * y)) :
  (Real.sqrt x / Real.sqrt y) = (Real.sqrt 47 / Real.sqrt 5) :=
by
  sorry

end sqrt_div_sqrt_eq_sqrt_fraction_l72_72037


namespace at_least_one_negative_root_l72_72171

theorem at_least_one_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0)) ↔ a < (-1 + Real.sqrt 19) / 9 := by
  sorry

end at_least_one_negative_root_l72_72171


namespace value_of_x_minus_y_l72_72829

theorem value_of_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end value_of_x_minus_y_l72_72829


namespace number_of_terms_in_arithmetic_sequence_l72_72191

-- Definitions derived directly from the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 4
def last_term : ℕ := 2010

-- Lean statement for the proof problem
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 503 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l72_72191


namespace james_marbles_left_l72_72362

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end james_marbles_left_l72_72362


namespace flowers_per_vase_l72_72436

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end flowers_per_vase_l72_72436


namespace partition_natural_numbers_l72_72228

theorem partition_natural_numbers :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ f n ∧ f n ≤ 100) ∧
  (∀ a b c, a + 99 * b = c → f a = f c ∨ f a = f b ∨ f b = f c) :=
sorry

end partition_natural_numbers_l72_72228


namespace find_f_of_2_l72_72325

-- Definitions based on problem conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x) + 9

-- The main statement to proof that f(2) = 6 under the given conditions
theorem find_f_of_2 (f : ℝ → ℝ)
  (hf : is_odd_function f)
  (hg : ∀ x, g f x = f x + 9)
  (h : g f (-2) = 3) :
  f 2 = 6 := 
sorry

end find_f_of_2_l72_72325


namespace compare_f_values_l72_72844

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * Real.cos x

theorem compare_f_values :
  f 0 < f (-1 / 3) ∧ f (-1 / 3) < f (2 / 5) :=
by
  sorry

end compare_f_values_l72_72844


namespace sum_of_cubes_l72_72075

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 :=
sorry

end sum_of_cubes_l72_72075


namespace solve_for_d_l72_72990

theorem solve_for_d (n k c d : ℝ) (h₁ : n = 2 * k * c * d / (c + d)) (h₂ : 2 * k * c ≠ n) :
  d = n * c / (2 * k * c - n) :=
by
  sorry

end solve_for_d_l72_72990


namespace xyz_inequality_l72_72174

theorem xyz_inequality : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by
  intros
  sorry

end xyz_inequality_l72_72174


namespace jawbreakers_in_package_correct_l72_72085

def jawbreakers_ate : Nat := 20
def jawbreakers_left : Nat := 4
def jawbreakers_in_package : Nat := jawbreakers_ate + jawbreakers_left

theorem jawbreakers_in_package_correct : jawbreakers_in_package = 24 := by
  sorry

end jawbreakers_in_package_correct_l72_72085


namespace village_population_500_l72_72352

variable (n : ℝ) -- Define the variable for population increase
variable (initial_population : ℝ) -- Define the variable for the initial population

-- Conditions from the problem
def first_year_increase : Prop := initial_population * (3 : ℝ) = n
def initial_population_def : Prop := initial_population = n / 3
def second_year_increase_def := ((n / 3 + n) * (n / 100 )) = 300

-- Define the final population formula
def population_after_two_years : ℝ := (initial_population + n + 300)

theorem village_population_500 (n : ℝ) (initial_population: ℝ) :
  first_year_increase n initial_population →
  initial_population_def n initial_population →
  second_year_increase_def n →
  population_after_two_years n initial_population = 500 :=
by sorry

#check village_population_500

end village_population_500_l72_72352


namespace wrapping_paper_per_present_l72_72102

theorem wrapping_paper_per_present :
  ∀ (total: ℚ) (presents: ℚ) (frac_used: ℚ),
  total = 3 / 10 → presents = 3 → frac_used = total / presents → frac_used = 1 / 10 :=
by
  intros total presents frac_used htotal hpresents hfrac
  rw [htotal, hpresents, hfrac]
  sorry

end wrapping_paper_per_present_l72_72102


namespace polygon_sides_l72_72349

theorem polygon_sides (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∃ (theta theta' : ℝ), theta = (n - 2) * 180 / n ∧ theta' = (n + 7) * 180 / (n + 9) ∧ theta' = theta + 9) : n = 15 :=
sorry

end polygon_sides_l72_72349


namespace correct_operations_result_greater_than_1000_l72_72095

theorem correct_operations_result_greater_than_1000
    (finalResultIncorrectOps : ℕ)
    (originalNumber : ℕ)
    (finalResultCorrectOps : ℕ)
    (H1 : finalResultIncorrectOps = 40)
    (H2 : originalNumber = (finalResultIncorrectOps + 12) * 8)
    (H3 : finalResultCorrectOps = (originalNumber * 8) + (2 * originalNumber) + 12) :
  finalResultCorrectOps > 1000 := 
sorry

end correct_operations_result_greater_than_1000_l72_72095


namespace number_thought_of_eq_95_l72_72937

theorem number_thought_of_eq_95 (x : ℝ) (h : (x / 5) + 23 = 42) : x = 95 := 
by
  sorry

end number_thought_of_eq_95_l72_72937


namespace merchant_marking_percentage_l72_72008

theorem merchant_marking_percentage (L : ℝ) (p : ℝ) (d : ℝ) (c : ℝ) (profit : ℝ) 
  (purchase_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (list_price : ℝ) : 
  L = 100 ∧ p = 30 ∧ d = 20 ∧ c = 20 ∧ profit = 20 ∧ 
  purchase_price = L - L * (p / 100) ∧ 
  marked_price = 109.375 ∧ 
  selling_price = marked_price - marked_price * (d / 100) ∧ 
  selling_price - purchase_price = profit * (selling_price / 100) 
  → marked_price = 109.375 := by sorry

end merchant_marking_percentage_l72_72008


namespace inequality_abc_l72_72230

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end inequality_abc_l72_72230


namespace tiered_water_pricing_l72_72513

theorem tiered_water_pricing (x : ℝ) (y : ℝ) : 
  (∀ z, 0 ≤ z ∧ z ≤ 12 → y = 3 * z ∨
        12 < z ∧ z ≤ 18 → y = 36 + 6 * (z - 12) ∨
        18 < z → y = 72 + 9 * (z - 18)) → 
  y = 54 → 
  x = 15 :=
by
  sorry

end tiered_water_pricing_l72_72513


namespace inversely_proportional_l72_72389

theorem inversely_proportional (x y : ℕ) (c : ℕ) 
  (h1 : x * y = c)
  (hx1 : x = 40) 
  (hy1 : y = 5) 
  (hy2 : y = 10) : x = 20 :=
by
  sorry

end inversely_proportional_l72_72389


namespace Simplify_division_l72_72105

theorem Simplify_division :
  (5 * 10^9) / (2 * 10^5 * 5) = 5000 := sorry

end Simplify_division_l72_72105


namespace find_x_l72_72597

theorem find_x (a x : ℤ) (h1 : -6 * a^2 = x * (4 * a + 2)) (h2 : a = 1) : x = -1 :=
sorry

end find_x_l72_72597


namespace find_x_l72_72985

theorem find_x : ∃ x : ℕ, 6 * 2^x = 2048 ∧ x = 10 := by
  sorry

end find_x_l72_72985


namespace second_most_eater_l72_72978

variable (C M K B T : ℕ)  -- Assuming the quantities of food each child ate are positive integers

theorem second_most_eater
  (h1 : C > M)
  (h2 : B < K)
  (h3 : T < K)
  (h4 : K < M) :
  ∃ x, x = M ∧ (∀ y, y ≠ C → x ≥ y) ∧ (∃ z, z ≠ C ∧ z > M) :=
by {
  sorry
}

end second_most_eater_l72_72978


namespace smallest_value_of_linear_expression_l72_72069

theorem smallest_value_of_linear_expression :
  (∃ a, 8 * a^2 + 6 * a + 5 = 7 ∧ (∃ b, b = 3 * a + 2 ∧ ∀ c, (8 * c^2 + 6 * c + 5 = 7 → 3 * c + 2 ≥ b))) → -1 = b :=
by
  sorry

end smallest_value_of_linear_expression_l72_72069


namespace find_a_l72_72548

noncomputable def P (a : ℚ) (k : ℕ) : ℚ := a * (1 / 2)^(k)

theorem find_a (a : ℚ) : (P a 1 + P a 2 + P a 3 = 1) → (a = 8 / 7) :=
by
  sorry

end find_a_l72_72548


namespace eccentricity_proof_l72_72633

variables (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
def ellipse_eq (x y: ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def circle_eq (x y: ℝ) := x^2 + y^2 = b^2

-- Conditions
def a_eq_3b : Prop := a = 3 * b
def major_minor_axis_relation : Prop := a^2 = b^2 + c^2

-- To prove
theorem eccentricity_proof 
  (h3 : a_eq_3b a b)
  (h4 : major_minor_axis_relation a b c) :
  (c / a) = (2 * Real.sqrt 2 / 3) := 
  sorry

end eccentricity_proof_l72_72633


namespace problem_l72_72219

variable {f : ℝ → ℝ}

-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Condition: f is monotonically decreasing on (0, +∞)
def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f y < f x

theorem problem (h_even : even_function f) (h_mon_dec : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l72_72219


namespace total_cost_is_correct_l72_72233

def cost_shirt (S : ℝ) : Prop := S = 12
def cost_shoes (Sh S : ℝ) : Prop := Sh = S + 5
def cost_dress (D : ℝ) : Prop := D = 25
def discount_shoes (Sh Sh' : ℝ) : Prop := Sh' = Sh - 0.10 * Sh
def discount_dress (D D' : ℝ) : Prop := D' = D - 0.05 * D
def cost_bag (B twoS Sh' D' : ℝ) : Prop := B = (twoS + Sh' + D') / 2
def total_cost_before_tax (T_before twoS Sh' D' B : ℝ) : Prop := T_before = twoS + Sh' + D' + B
def sales_tax (tax T_before : ℝ) : Prop := tax = 0.07 * T_before
def total_cost_including_tax (T_total T_before tax : ℝ) : Prop := T_total = T_before + tax
def convert_to_usd (T_usd T_total : ℝ) : Prop := T_usd = T_total * 1.18

theorem total_cost_is_correct (S Sh D Sh' D' twoS B T_before tax T_total T_usd : ℝ) :
  cost_shirt S →
  cost_shoes Sh S →
  cost_dress D →
  discount_shoes Sh Sh' →
  discount_dress D D' →
  twoS = 2 * S →
  cost_bag B twoS Sh' D' →
  total_cost_before_tax T_before twoS Sh' D' B →
  sales_tax tax T_before →
  total_cost_including_tax T_total T_before tax →
  convert_to_usd T_usd T_total →
  T_usd = 119.42 :=
by
  sorry

end total_cost_is_correct_l72_72233


namespace david_marks_physics_l72_72810

def marks_english := 96
def marks_math := 95
def marks_chemistry := 97
def marks_biology := 95
def average_marks := 93
def number_of_subjects := 5

theorem david_marks_physics : 
  let total_marks := average_marks * number_of_subjects 
  let total_known_marks := marks_english + marks_math + marks_chemistry + marks_biology
  let marks_physics := total_marks - total_known_marks
  marks_physics = 82 :=
by
  sorry

end david_marks_physics_l72_72810


namespace james_marbles_left_l72_72360

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end james_marbles_left_l72_72360


namespace inequality_proof_l72_72217

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b + c = 1) :
  a * (1 + b - c) ^ (1 / 3) + b * (1 + c - a) ^ (1 / 3) + c * (1 + a - b) ^ (1 / 3) ≤ 1 := 
by
  sorry

end inequality_proof_l72_72217


namespace interior_angle_regular_hexagon_l72_72761

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l72_72761


namespace find_number_to_add_l72_72916

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l72_72916


namespace circle_center_sum_l72_72890

theorem circle_center_sum (h k : ℝ) :
  (∃ h k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = 6 * x + 8 * y - 15) → (h, k) = (3, 4)) →
  h + k = 7 :=
by
  sorry

end circle_center_sum_l72_72890


namespace good_pair_exists_l72_72961

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ (∃ k1 k2 : ℕ, m * n = k1 * k1 ∧ (m + 1) * (n + 1) = k2 * k2) :=
by
  sorry

end good_pair_exists_l72_72961


namespace abs_ineq_solution_l72_72535

theorem abs_ineq_solution (x : ℝ) :
  (|x - 2| + |x + 1| < 4) ↔ (x ∈ Set.Ioo (-7 / 2) (-1) ∪ Set.Ico (-1) (5 / 2)) := by
  sorry

end abs_ineq_solution_l72_72535


namespace arsenic_acid_concentration_equilibrium_l72_72611

noncomputable def dissociation_constants 
  (Kd1 Kd2 Kd3 : ℝ) (H3AsO4 H2AsO4 HAsO4 AsO4 H : ℝ) : Prop :=
  Kd1 = (H * H2AsO4) / H3AsO4 ∧ Kd2 = (H * HAsO4) / H2AsO4 ∧ Kd3 = (H * AsO4) / HAsO4

theorem arsenic_acid_concentration_equilibrium :
  dissociation_constants 5.6e-3 1.7e-7 2.95e-12 0.1 (2e-2) (1.7e-7) (0) (2e-2) :=
by sorry

end arsenic_acid_concentration_equilibrium_l72_72611


namespace find_value_of_S_l72_72877

theorem find_value_of_S (S : ℝ)
  (h1 : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 180) :
  S = 180 :=
sorry

end find_value_of_S_l72_72877


namespace correct_operation_l72_72579

theorem correct_operation {a : ℝ} : (a ^ 6 / a ^ 2 = a ^ 4) :=
by sorry

end correct_operation_l72_72579


namespace probability_is_3888_over_7533_l72_72626

noncomputable def probability_odd_sum_given_even_product : ℚ := 
  let total_outcomes := 6^5
  let all_odd_outcomes := 3^5
  let at_least_one_even_outcomes := total_outcomes - all_odd_outcomes
  let favorable_outcomes := 5 * 3^4 + 10 * 3^4 + 3^5
  favorable_outcomes / at_least_one_even_outcomes

theorem probability_is_3888_over_7533 :
  probability_odd_sum_given_even_product = 3888 / 7533 := 
sorry

end probability_is_3888_over_7533_l72_72626


namespace work_finish_in_3_days_l72_72590

-- Define the respective rates of work
def A_rate := 1/4
def B_rate := 1/14
def C_rate := 1/7

-- Define the duration they start working together
def initial_duration := 2
def after_C_joining := 1 -- time after C joins before A leaves

-- From the third day, consider A leaving the job
theorem work_finish_in_3_days :
  (initial_duration * (A_rate + B_rate)) + 
  (after_C_joining * (A_rate + B_rate + C_rate)) + 
  ((1 : ℝ) - after_C_joining) * (B_rate + C_rate) >= 1 :=
by
  sorry

end work_finish_in_3_days_l72_72590


namespace solve_problem_l72_72264

def problem_statement : Prop := (245245 % 35 = 0)

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l72_72264


namespace interior_angle_regular_hexagon_l72_72763

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l72_72763


namespace total_students_l72_72509

-- Definitions based on problem conditions
def H := 36
def S := 32
def union_H_S := 59
def history_not_statistics := 27

-- The proof statement
theorem total_students : H + S - (H - history_not_statistics) = union_H_S :=
by sorry

end total_students_l72_72509


namespace marla_adds_white_paint_l72_72864

-- Define the conditions as hypotheses.
variables (total_percent blue_percent red_percent white_percent proportion_of_blue x : ℕ)
variable (total_ounces : ℕ)
hypothesis (H1 : total_percent = 100)
hypothesis (H2 : blue_percent = 70)
hypothesis (H3 : red_percent = 20)
hypothesis (H4 : white_percent = total_percent - blue_percent - red_percent)
hypothesis (H5 : total_ounces = 140)
hypothesis (H6 : blue_percent * x = white_percent * total_ounces)

-- The problem statement
theorem marla_adds_white_paint : 
  blue_percent * x = white_percent * total_ounces → 
  (x = 20)
:= sorry

end marla_adds_white_paint_l72_72864


namespace demokhar_lifespan_l72_72610

-- Definitions based on the conditions
def boy_fraction := 1 / 4
def young_man_fraction := 1 / 5
def adult_man_fraction := 1 / 3
def old_man_years := 13

-- Statement without proof
theorem demokhar_lifespan :
  ∀ (x : ℕ), (boy_fraction * x) + (young_man_fraction * x) + (adult_man_fraction * x) + old_man_years = x → x = 60 :=
by
  sorry

end demokhar_lifespan_l72_72610


namespace dice_sum_18_l72_72929

theorem dice_sum_18 (n : ℕ) : 
  (∃ k : ℕ, k = (5.choose k) ∧ 
    sum (λ i : fin 5, (nat.bounded 1 8) i ≥ n) = 18) →
  n = 2380 :=
sorry

end dice_sum_18_l72_72929


namespace solution_to_equation_l72_72314

theorem solution_to_equation (x : ℝ) (h : (5 - x / 2)^(1/3) = 2) : x = -6 :=
sorry

end solution_to_equation_l72_72314


namespace area_of_square_land_l72_72113

-- Define the problem conditions
variable (A P : ℕ)

-- Define the main theorem statement: proving area A given the conditions
theorem area_of_square_land (h₁ : 5 * A = 10 * P + 45) (h₂ : P = 36) : A = 81 := by
  sorry

end area_of_square_land_l72_72113


namespace servings_per_bottle_l72_72439

-- Definitions based on conditions
def total_guests : ℕ := 120
def servings_per_guest : ℕ := 2
def total_bottles : ℕ := 40

-- Theorem stating that given the conditions, the servings per bottle is 6
theorem servings_per_bottle : (total_guests * servings_per_guest) / total_bottles = 6 := by
  sorry

end servings_per_bottle_l72_72439


namespace div120_l72_72680

theorem div120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end div120_l72_72680


namespace next_ten_winners_each_receive_160_l72_72280

def total_prize : ℕ := 2400
def first_winner_share : ℚ := 1 / 3 * total_prize
def remaining_after_first : ℚ := total_prize - first_winner_share
def next_ten_winners_share : ℚ := remaining_after_first / 10

theorem next_ten_winners_each_receive_160 :
  next_ten_winners_share = 160 := by
sorry

end next_ten_winners_each_receive_160_l72_72280


namespace sector_area_l72_72428

theorem sector_area (R : ℝ) (hR_pos : R > 0) (h_circumference : 4 * R = 2 * R + arc_length) :
  (1 / 2) * arc_length * R = R^2 :=
by sorry

end sector_area_l72_72428


namespace find_x_l72_72648

theorem find_x (x : ℝ) (h : x^29 * 4^15 = 2 * 10^29) : x = 5 := 
by 
  sorry

end find_x_l72_72648


namespace correct_combined_average_l72_72203

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

end correct_combined_average_l72_72203


namespace probability_of_odd_sum_given_even_product_l72_72628

open Nat

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_outcomes := total_outcomes - odd_outcomes
  let favorable_outcomes := 15 * 3^5
  favorable_outcomes / even_outcomes

theorem probability_of_odd_sum_given_even_product :
  probability_odd_sum_given_even_product = 91 / 324 :=
by
  sorry

end probability_of_odd_sum_given_even_product_l72_72628


namespace probability_of_non_defective_product_l72_72290

-- Define the probability of producing a grade B product
def P_B : ℝ := 0.03

-- Define the probability of producing a grade C product
def P_C : ℝ := 0.01

-- Define the probability of producing a non-defective product (grade A)
def P_A : ℝ := 1 - P_B - P_C

-- The theorem to prove: The probability of producing a non-defective product is 0.96
theorem probability_of_non_defective_product : P_A = 0.96 := by
  -- Insert proof here
  sorry

end probability_of_non_defective_product_l72_72290


namespace value_of_c7_l72_72874

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end value_of_c7_l72_72874


namespace proof_of_expression_value_l72_72067

theorem proof_of_expression_value (m n : ℝ) 
  (h1 : m^2 - 2019 * m = 1) 
  (h2 : n^2 - 2019 * n = 1) : 
  (m^2 - 2019 * m + 3) * (n^2 - 2019 * n + 4) = 20 := 
by 
  sorry

end proof_of_expression_value_l72_72067


namespace range_of_m_l72_72828

theorem range_of_m (m : ℝ) :
  ¬(1^2 + 2*1 - m > 0) ∧ (2^2 + 2*2 - m > 0) ↔ (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l72_72828


namespace total_gold_cost_l72_72820

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end total_gold_cost_l72_72820


namespace system_of_equations_solution_l72_72714

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 : ℝ), 
    (x1 + 2 * x2 = 10) ∧
    (3 * x1 + 2 * x2 + x3 = 23) ∧
    (x2 + 2 * x3 = 13) ∧
    (x1 = 4) ∧
    (x2 = 3) ∧
    (x3 = 5) :=
sorry

end system_of_equations_solution_l72_72714


namespace protein_in_steak_is_correct_l72_72293

-- Definitions of the conditions
def collagen_protein_per_scoop : ℕ := 18 / 2 -- 9 grams
def protein_powder_per_scoop : ℕ := 21 -- 21 grams

-- Define the total protein consumed
def total_protein (collagen_scoops protein_scoops : ℕ) (protein_from_steak : ℕ) : ℕ :=
  collagen_protein_per_scoop * collagen_scoops + protein_powder_per_scoop * protein_scoops + protein_from_steak

-- Condition in the problem
def total_protein_consumed : ℕ := 86

-- Prove that the protein in the steak is 56 grams
theorem protein_in_steak_is_correct : 
  total_protein 1 1 56 = total_protein_consumed :=
sorry

end protein_in_steak_is_correct_l72_72293


namespace interior_triangles_from_chords_l72_72676

theorem interior_triangles_from_chords (h₁ : ∀ p₁ p₂ p₃ : Prop, ¬(p₁ ∧ p₂ ∧ p₃)) : 
  ∀ (nine_points_on_circle : Finset ℝ) (h₂ : nine_points_on_circle.card = 9), 
    ∃ (triangles : ℕ), triangles = 210 := 
by 
  sorry

end interior_triangles_from_chords_l72_72676


namespace length_of_rectangular_garden_l72_72417

theorem length_of_rectangular_garden (P B : ℝ) (h₁ : P = 1200) (h₂ : B = 240) :
  ∃ L : ℝ, P = 2 * (L + B) ∧ L = 360 :=
by
  sorry

end length_of_rectangular_garden_l72_72417


namespace smallest_x_l72_72133

theorem smallest_x (x : ℕ) :
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 279 :=
by
  sorry

end smallest_x_l72_72133


namespace max_sum_after_swap_l72_72528

section
variables (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ)
  (h1 : 100 * a1 + 10 * b1 + c1 + 100 * a2 + 10 * b2 + c2 + 100 * a3 + 10 * b3 + c3 = 2019)
  (h2 : 1 ≤ a1 ∧ a1 ≤ 9 ∧ 0 ≤ b1 ∧ b1 ≤ 9 ∧ 0 ≤ c1 ∧ c1 ≤ 9)
  (h3 : 1 ≤ a2 ∧ a2 ≤ 9 ∧ 0 ≤ b2 ∧ b2 ≤ 9 ∧ 0 ≤ c2 ∧ c2 ≤ 9)
  (h4 : 1 ≤ a3 ∧ a3 ≤ 9 ∧ 0 ≤ b3 ∧ b3 ≤ 9 ∧ 0 ≤ c3 ∧ c3 ≤ 9)

theorem max_sum_after_swap : 100 * c1 + 10 * b1 + a1 + 100 * c2 + 10 * b2 + a2 + 100 * c3 + 10 * b3 + a3 ≤ 2118 := 
  sorry

end

end max_sum_after_swap_l72_72528


namespace rectangle_length_15_l72_72119

theorem rectangle_length_15
  (w l : ℝ)
  (h_ratio : 5 * w = 2 * l + 2 * w)
  (h_area : l * w = 150) :
  l = 15 :=
sorry

end rectangle_length_15_l72_72119


namespace table_height_l72_72906

theorem table_height
  (l d h : ℤ)
  (h_eq1 : l + h - d = 36)
  (h_eq2 : 2 * l + h = 46)
  (l_eq_d : l = d) :
  h = 36 :=
by
  sorry

end table_height_l72_72906


namespace bug_visits_tiles_l72_72425

theorem bug_visits_tiles :
  let width : ℕ := 11
  let length : ℕ := 19
  width + length - Nat.gcd width length = 29 :=
by
  let width : ℕ := 11
  let length : ℕ := 19
  have h1 : Nat.gcd width length = 1 := by
    sorry
  calc
    width + length - Nat.gcd width length
      = 11 + 19 - Nat.gcd 11 19 : by rfl
  ... = 11 + 19 - 1 : by rw [h1]
  ... = 29 : by norm_num

end bug_visits_tiles_l72_72425


namespace total_sandwiches_l72_72021

theorem total_sandwiches (billy : ℕ) (katelyn_more : ℕ) (katelyn_quarter : ℕ) :
  billy = 49 → katelyn_more = 47 → katelyn_quarter = 4 → 
  billy + (billy + katelyn_more) + ((billy + katelyn_more) / katelyn_quarter) = 169 :=
by
  intros hb hk hq
  rw [hb, hk, hq]
  calc
    49 + (49 + 47) + ((49 + 47) / 4) = 49 + 96 + 24 : by simp
                                ... = 169 : by simp

sorry

end total_sandwiches_l72_72021


namespace centroid_midpoint_triangle_eq_centroid_original_triangle_l72_72227

/-
Prove that the centroid of the triangle formed by the midpoints of the sides of another triangle
is the same as the centroid of the original triangle.
-/
theorem centroid_midpoint_triangle_eq_centroid_original_triangle
  (A B C M N P : ℝ × ℝ)
  (hM : M = (A + B) / 2)
  (hN : N = (A + C) / 2)
  (hP : P = (B + C) / 2) :
  (M.1 + N.1 + P.1) / 3 = (A.1 + B.1 + C.1) / 3 ∧
  (M.2 + N.2 + P.2) / 3 = (A.2 + B.2 + C.2) / 3 :=
by
  sorry

end centroid_midpoint_triangle_eq_centroid_original_triangle_l72_72227


namespace Sarah_total_weeds_l72_72533

noncomputable def Tuesday_weeds : ℕ := 25
noncomputable def Wednesday_weeds : ℕ := 3 * Tuesday_weeds
noncomputable def Thursday_weeds : ℕ := (1 / 5) * Tuesday_weeds
noncomputable def Friday_weeds : ℕ := (3 / 4) * Tuesday_weeds - 10

noncomputable def Total_weeds : ℕ := Tuesday_weeds + Wednesday_weeds + Thursday_weeds + Friday_weeds

theorem Sarah_total_weeds : Total_weeds = 113 := by
  sorry

end Sarah_total_weeds_l72_72533


namespace arithmetic_sequence_ratios_l72_72994

theorem arithmetic_sequence_ratios
  (a : ℕ → ℝ) (b : ℕ → ℝ) (A : ℕ → ℝ) (B : ℕ → ℝ)
  (d1 d2 a1 b1 : ℝ)
  (hA_sum : ∀ n : ℕ, A n = n * a1 + (n * (n - 1)) * d1 / 2)
  (hB_sum : ∀ n : ℕ, B n = n * b1 + (n * (n - 1)) * d2 / 2)
  (h_ratio : ∀ n : ℕ, B n ≠ 0 → A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b n ≠ 0 → a n / b n = (4 * n - 3) / (6 * n - 2) := sorry

end arithmetic_sequence_ratios_l72_72994


namespace correct_equation_l72_72779

theorem correct_equation (x y a b : ℝ) :
  ¬ (-(x - 6) = -x - 6) ∧
  ¬ (-y^2 - y^2 = 0) ∧
  ¬ (9 * a^2 * b - 9 * a * b^2 = 0) ∧
  (-9 * y^2 + 16 * y^2 = 7 * y^2) :=
by
  sorry

end correct_equation_l72_72779


namespace problem_statement_l72_72341

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l72_72341


namespace necessary_and_sufficient_condition_l72_72986

theorem necessary_and_sufficient_condition (a b : ℝ) : a^2 * b > a * b^2 ↔ 1/a < 1/b := 
sorry

end necessary_and_sufficient_condition_l72_72986


namespace num_schemes_l72_72593

-- Definitions for the costs of book types
def cost_A := 30
def cost_B := 25
def cost_C := 20

-- The total budget
def budget := 500

-- Constraints for the number of books of type A
def min_books_A := 5
def max_books_A := 6

-- Definition of a scheme
structure Scheme :=
  (num_A : ℕ)
  (num_B : ℕ)
  (num_C : ℕ)

-- Function to calculate the total cost of a scheme
def total_cost (s : Scheme) : ℕ :=
  cost_A * s.num_A + cost_B * s.num_B + cost_C * s.num_C

-- Valid scheme predicate
def valid_scheme (s : Scheme) : Prop :=
  total_cost(s) = budget ∧
  s.num_A ≥ min_books_A ∧ s.num_A ≤ max_books_A ∧
  s.num_B > 0 ∧ s.num_C > 0

-- Theorem statement: Prove the number of valid purchasing schemes is 6
theorem num_schemes : (finset.filter valid_scheme
  (finset.product (finset.range (max_books_A + 1)) 
                  (finset.product (finset.range (budget / cost_B + 1)) (finset.range (budget / cost_C + 1)))).to_finset).card = 6 := sorry

end num_schemes_l72_72593


namespace measured_diagonal_length_l72_72800

theorem measured_diagonal_length (a b c d diag : Real)
  (h1 : a = 1) (h2 : b = 2) (h3 : c = 2.8) (h4 : d = 5) (hd : diag = 7.5) :
  diag = 2.8 :=
sorry

end measured_diagonal_length_l72_72800


namespace sum_of_ages_of_henrys_brothers_l72_72644

theorem sum_of_ages_of_henrys_brothers (a b c : ℕ) : 
  a = 2 * b → 
  b = c ^ 2 →
  a ≠ b ∧ a ≠ c ∧ b ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 14 :=
by
  intro h₁ h₂ h₃ h₄
  sorry

end sum_of_ages_of_henrys_brothers_l72_72644


namespace quadratic_inequality_solution_l72_72399

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0) ↔ (-3 < k ∧ k < 0) :=
sorry

end quadratic_inequality_solution_l72_72399


namespace complete_the_square_b_26_l72_72778

theorem complete_the_square_b_26 :
  ∃ (a b : ℝ), (∀ x : ℝ, x^2 + 10 * x - 1 = 0 ↔ (x + a)^2 = b) ∧ b = 26 :=
sorry

end complete_the_square_b_26_l72_72778


namespace two_pow_geq_n_cubed_for_n_geq_ten_l72_72044

theorem two_pow_geq_n_cubed_for_n_geq_ten (n : ℕ) (hn : n ≥ 10) : 2^n ≥ n^3 := 
sorry

end two_pow_geq_n_cubed_for_n_geq_ten_l72_72044


namespace lines_intersection_l72_72572

/-- Two lines are defined by the equations y = 2x + c and y = 4x + d.
These lines intersect at the point (8, 12).
Prove that c + d = -24. -/
theorem lines_intersection (c d : ℝ) (h1 : 12 = 2 * 8 + c) (h2 : 12 = 4 * 8 + d) :
    c + d = -24 :=
by
  sorry

end lines_intersection_l72_72572


namespace sin_product_l72_72048

theorem sin_product (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.sin (π / 2 - α) = 2 / 5 :=
by
  -- proof shorter placeholder
  sorry

end sin_product_l72_72048


namespace find_parabola_eq_find_range_of_b_l72_72327

-- Problem 1: Finding the equation of the parabola
theorem find_parabola_eq (p : ℝ) (h1 : p > 0) (x1 x2 y1 y2 : ℝ) 
  (A : (x1 + 4) * 2 = 2 * p * y1) (C : (x2 + 4) * 2 = 2 * p * y2)
  (h3 : x1^2 = 2 * p * y1) (h4 : x2^2 = 2 * p * y2) 
  (h5 : y2 = 4 * y1) :
  x1^2 = 4 * y1 :=
sorry

-- Problem 2: Finding the range of b
theorem find_range_of_b (k : ℝ) (h : k > 0 ∨ k < -4) : 
  ∃ b : ℝ, b = 2 * (k + 1)^2 ∧ b > 2 :=
sorry

end find_parabola_eq_find_range_of_b_l72_72327


namespace probability_of_neither_is_correct_l72_72782

-- Definitions of the given conditions
def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_cake_and_muffin_buyers : ℕ := 19

-- Define the probability calculation function
def probability_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  let buyers_neither := total - (cake + muffin - both)
  (buyers_neither : ℚ) / (total : ℚ)

-- State the main theorem to ensure it is equivalent to our mathematical problem
theorem probability_of_neither_is_correct :
  probability_neither total_buyers cake_buyers muffin_buyers both_cake_and_muffin_buyers = 0.29 := 
sorry

end probability_of_neither_is_correct_l72_72782


namespace repeating_decimal_to_fraction_l72_72313

noncomputable def repeating_decimal_solution : ℚ := 7311 / 999

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 7 + 318 / 999) : x = repeating_decimal_solution := 
by
  sorry

end repeating_decimal_to_fraction_l72_72313


namespace condition_1_valid_for_n_condition_2_valid_for_n_l72_72320

-- Definitions from the conditions
def is_cube_root_of_unity (ω : ℂ) : Prop := ω^3 = 1

def roots_of_polynomial (ω : ℂ) (ω2 : ℂ) : Prop :=
  ω^2 + ω + 1 = 0 ∧ is_cube_root_of_unity ω ∧ is_cube_root_of_unity ω2

-- Problem statements
theorem condition_1_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n - x^n - 1 ↔ ∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k - 1 := sorry

theorem condition_2_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n + x^n + 1 ↔ ∃ k : ℕ, n = 6 * k + 2 ∨ n = 6 * k - 2 := sorry

end condition_1_valid_for_n_condition_2_valid_for_n_l72_72320


namespace sum_divisible_by_17_l72_72316

theorem sum_divisible_by_17 :
    (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_17_l72_72316


namespace function_at_neg_one_zero_l72_72487

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l72_72487


namespace min_value_am_hm_l72_72370

theorem min_value_am_hm (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end min_value_am_hm_l72_72370


namespace jordan_machine_solution_l72_72214

theorem jordan_machine_solution (x : ℝ) (h : 2 * x + 3 - 5 = 27) : x = 14.5 :=
sorry

end jordan_machine_solution_l72_72214


namespace interior_angle_of_regular_hexagon_l72_72758

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l72_72758


namespace incorrect_quotient_l72_72852

theorem incorrect_quotient
    (correct_quotient : ℕ)
    (correct_divisor : ℕ)
    (incorrect_divisor : ℕ)
    (h1 : correct_quotient = 28)
    (h2 : correct_divisor = 21)
    (h3 : incorrect_divisor = 12) :
  correct_divisor * correct_quotient / incorrect_divisor = 49 :=
by
  sorry

end incorrect_quotient_l72_72852


namespace add_base6_l72_72147

def base6_to_base10 (n : Nat) : Nat :=
  let rec aux (n : Nat) (exp : Nat) : Nat :=
    match n with
    | 0     => 0
    | n + 1 => aux n (exp + 1) + (n % 6) * (6 ^ exp)
  aux n 0

def base10_to_base6 (n : Nat) : Nat :=
  let rec aux (n : Nat) : Nat :=
    if n = 0 then 0
    else
      let q := n / 6
      let r := n % 6
      r + 10 * aux q
  aux n

theorem add_base6 (a b : Nat) (h1 : base6_to_base10 a = 5) (h2 : base6_to_base10 b = 13) : base10_to_base6 (base6_to_base10 a + base6_to_base10 b) = 30 :=
by
  sorry

end add_base6_l72_72147


namespace insurance_compensation_l72_72278

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end insurance_compensation_l72_72278


namespace monotonic_decreasing_interval_l72_72396

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < (1 / 2) → (0 < x ∧ x < (1 / 2)) ∧ (f (1 / 2) - f x) > 0 :=
sorry

end monotonic_decreasing_interval_l72_72396


namespace insurance_compensation_correct_l72_72277

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end insurance_compensation_correct_l72_72277


namespace inequality_proof_l72_72232

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end inequality_proof_l72_72232


namespace Mrs_Hilt_walks_to_fountain_l72_72675

theorem Mrs_Hilt_walks_to_fountain :
  ∀ (distance trips : ℕ), distance = 30 → trips = 4 → distance * trips = 120 :=
by
  intros distance trips h_distance h_trips
  sorry

end Mrs_Hilt_walks_to_fountain_l72_72675


namespace sequence_property_l72_72833

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end sequence_property_l72_72833


namespace marbles_solution_l72_72521

open Nat

def marbles_problem : Prop :=
  ∃ J_k J_j : Nat, (J_k = 3) ∧ (J_k = J_j - 4) ∧ (J_k + J_j = 10)

theorem marbles_solution : marbles_problem := by
  sorry

end marbles_solution_l72_72521


namespace product_of_solutions_eq_zero_l72_72728

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end product_of_solutions_eq_zero_l72_72728


namespace count_positive_integers_satisfying_inequality_l72_72620

theorem count_positive_integers_satisfying_inequality :
  (∃ n : ℕ, 0 < n ∧ (∏ i in (finset.range 50).image (λ k, n - (2 * k + 1)), i) < 0) = 49 :=
sorry

end count_positive_integers_satisfying_inequality_l72_72620


namespace people_at_table_l72_72696

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l72_72696


namespace natural_number_squares_l72_72009

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end natural_number_squares_l72_72009


namespace total_trees_correct_l72_72855

def apricot_trees : ℕ := 58
def peach_trees : ℕ := 3 * apricot_trees
def total_trees : ℕ := apricot_trees + peach_trees

theorem total_trees_correct : total_trees = 232 :=
by
  sorry

end total_trees_correct_l72_72855


namespace initial_wage_illiterate_l72_72657

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end initial_wage_illiterate_l72_72657


namespace unique_real_root_t_l72_72081

theorem unique_real_root_t (t : ℝ) :
  (∃ x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0 ∧ 
  ∀ y : ℝ, 3 * y + 7 * t - 2 + (2 * t * y^2 + 7 * t^2 - 9) / (y - t) = 0 ∧ x ≠ y → false) →
  t = -3 ∨ t = -7 / 2 ∨ t = 1 :=
by
  sorry

end unique_real_root_t_l72_72081


namespace fedya_incorrect_l72_72359

theorem fedya_incorrect 
  (a b c d : ℕ) 
  (a_ends_in_9 : a % 10 = 9)
  (b_ends_in_7 : b % 10 = 7)
  (c_ends_in_3 : c % 10 = 3)
  (d_is_1 : d = 1) : 
  a ≠ b * c + d :=
by {
  sorry
}

end fedya_incorrect_l72_72359


namespace typing_speed_ratio_l72_72581

-- Defining the conditions for the problem
def typing_speeds (T M : ℝ) : Prop :=
  (T + M = 12) ∧ (T + 1.25 * M = 14)

-- Stating the theorem with conditions and the expected result
theorem typing_speed_ratio (T M : ℝ) (h : typing_speeds T M) : M / T = 2 :=
by
  cases h
  sorry

end typing_speed_ratio_l72_72581


namespace value_of_c7_l72_72871

theorem value_of_c7 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : ∀ n, b n = 2^(n-1))
  (h3 : ∀ n, c n = a n * b n) :
  c 7 = 448 :=
by
  sorry

end value_of_c7_l72_72871


namespace product_of_decimals_l72_72155

theorem product_of_decimals :
  0.5 * 0.8 = 0.40 :=
by
  -- Proof will go here; using sorry to skip for now
  sorry

end product_of_decimals_l72_72155


namespace inequality_solution_l72_72884

theorem inequality_solution :
  ∀ x : ℝ, ( (x - 3) / ( (x - 2) ^ 2 ) < 0 ) ↔ ( x < 2 ∨ (2 < x ∧ x < 3) ) :=
by
  sorry

end inequality_solution_l72_72884


namespace ratio_of_a_to_b_l72_72731

theorem ratio_of_a_to_b 
  (b c a : ℝ)
  (h1 : b / c = 1 / 5) 
  (h2 : a / c = 1 / 7.5) : 
  a / b = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_l72_72731


namespace rainfall_difference_l72_72568

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l72_72568


namespace total_pay_per_week_l72_72260

variable (X Y : ℝ)
variable (hx : X = 1.2 * Y)
variable (hy : Y = 240)

theorem total_pay_per_week : X + Y = 528 := by
  sorry

end total_pay_per_week_l72_72260


namespace total_number_of_animals_l72_72164

-- Definitions for the animal types
def heads_per_hen := 2
def legs_per_hen := 8
def heads_per_peacock := 3
def legs_per_peacock := 9
def heads_per_zombie_hen := 6
def legs_per_zombie_hen := 12

-- Given total heads and legs
def total_heads := 800
def total_legs := 2018

-- Proof that the total number of animals is 203
theorem total_number_of_animals : 
  ∀ (H P Z : ℕ), 
    heads_per_hen * H + heads_per_peacock * P + heads_per_zombie_hen * Z = total_heads
    ∧ legs_per_hen * H + legs_per_peacock * P + legs_per_zombie_hen * Z = total_legs 
    → H + P + Z = 203 :=
by
  sorry

end total_number_of_animals_l72_72164


namespace smallest_x_l72_72132

theorem smallest_x (x : ℤ) (h : x + 3 < 3 * x - 4) : x = 4 :=
by
  sorry

end smallest_x_l72_72132


namespace projective_iff_fractional_linear_l72_72679

def projective_transformation (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))

theorem projective_iff_fractional_linear (P : ℝ → ℝ) : 
  projective_transformation P ↔ ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d)) :=
by 
  sorry

end projective_iff_fractional_linear_l72_72679


namespace sequence_bound_l72_72831

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l72_72831


namespace class_average_l72_72343

theorem class_average (x : ℝ) :
  (0.25 * 80 + 0.5 * x + 0.25 * 90 = 75) → x = 65 := by
  sorry

end class_average_l72_72343


namespace average_percent_increase_in_profit_per_car_l72_72802

theorem average_percent_increase_in_profit_per_car
  (N P : ℝ) -- N: Number of cars sold last year, P: Profit per car last year
  (HP1 : N > 0) -- Non-zero number of cars
  (HP2 : P > 0) -- Non-zero profit
  (HProfitIncrease : 1.3 * (N * P) = 1.3 * N * P) -- Total profit increased by 30%
  (HCarDecrease : 0.7 * N = 0.7 * N) -- Number of cars decreased by 30%
  : ((1.3 / 0.7) - 1) * 100 = 85.7 := sorry

end average_percent_increase_in_profit_per_car_l72_72802


namespace average_licks_to_center_l72_72609

theorem average_licks_to_center (Dan_lcks Michael_lcks Sam_lcks David_lcks Lance_lcks : ℕ)
  (h1 : Dan_lcks = 58) 
  (h2 : Michael_lcks = 63) 
  (h3 : Sam_lcks = 70) 
  (h4 : David_lcks = 70) 
  (h5 : Lance_lcks = 39) :
  (Dan_lcks + Michael_lcks + Sam_lcks + David_lcks + Lance_lcks) / 5 = 60 :=
by {
  sorry
}

end average_licks_to_center_l72_72609


namespace trader_loses_l72_72953

theorem trader_loses 
  (l_1 l_2 q : ℝ) 
  (h1 : l_1 ≠ l_2) 
  (p_1 p_2 : ℝ) 
  (h2 : p_1 = q * (l_2 / l_1)) 
  (h3 : p_2 = q * (l_1 / l_2)) :
  p_1 + p_2 > 2 * q :=
by {
  sorry
}

end trader_loses_l72_72953


namespace last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l72_72772

noncomputable def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_1989_1989:
  last_digit (1989 ^ 1989) = 9 := 
sorry

theorem last_digit_1989_1992:
  last_digit (1989 ^ 1992) = 1 := 
sorry

theorem last_digit_1992_1989:
  last_digit (1992 ^ 1989) = 2 := 
sorry

theorem last_digit_1992_1992:
  last_digit (1992 ^ 1992) = 6 := 
sorry

end last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l72_72772


namespace accurate_bottle_weight_l72_72016

-- Define the options as constants
def OptionA : ℕ := 500 -- milligrams
def OptionB : ℕ := 500 * 1000 -- grams
def OptionC : ℕ := 500 * 1000 * 1000 -- kilograms
def OptionD : ℕ := 500 * 1000 * 1000 * 1000 -- tons

-- Define a threshold range for the weight of a standard bottle of mineral water in grams
def typicalBottleWeightMin : ℕ := 400 -- for example
def typicalBottleWeightMax : ℕ := 600 -- for example

-- Translate the question and conditions into a proof statement
theorem accurate_bottle_weight : OptionB = 500 * 1000 :=
by
  -- Normally, we would add the necessary steps here to prove the statement
  sorry

end accurate_bottle_weight_l72_72016


namespace rightmost_three_digits_seven_pow_1983_add_123_l72_72126

theorem rightmost_three_digits_seven_pow_1983_add_123 :
  (7 ^ 1983 + 123) % 1000 = 466 := 
by 
  -- Proof steps are omitted
  sorry 

end rightmost_three_digits_seven_pow_1983_add_123_l72_72126


namespace regular_hexagon_interior_angle_l72_72743

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l72_72743


namespace JohnNeeds72Strings_l72_72365

def JohnHasToRestring3Basses : Nat := 3
def StringsPerBass : Nat := 4

def TwiceAsManyGuitarsAsBasses : Nat := 2 * JohnHasToRestring3Basses
def StringsPerNormalGuitar : Nat := 6

def ThreeFewerEightStringGuitarsThanNormal : Nat := TwiceAsManyGuitarsAsBasses - 3
def StringsPerEightStringGuitar : Nat := 8

def TotalStringsNeeded : Nat := 
  (JohnHasToRestring3Basses * StringsPerBass) +
  (TwiceAsManyGuitarsAsBasses * StringsPerNormalGuitar) +
  (ThreeFewerEightStringGuitarsThanNormal * StringsPerEightStringGuitar)

theorem JohnNeeds72Strings : TotalStringsNeeded = 72 := by
  calculate
  sorry

end JohnNeeds72Strings_l72_72365


namespace find_q_r_s_l72_72666

noncomputable def is_valid_geometry 
  (AD : ℝ) (AL : ℝ) (AM : ℝ) (AN : ℝ) (q : ℕ) (r : ℕ) (s : ℕ) : Prop :=
  AD = 10 ∧ AL = 3 ∧ AM = 3 ∧ AN = 3 ∧ ¬(∃ p : ℕ, p^2 ∣ s)

theorem find_q_r_s : ∃ (q r s : ℕ), is_valid_geometry 10 3 3 3 q r s ∧ q + r + s = 711 :=
by
  sorry

end find_q_r_s_l72_72666


namespace minimum_expression_l72_72323

variable (a b : ℝ)

theorem minimum_expression (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x + y = 3 → 
  x = a ∧ y = b  → ∃ m : ℝ, m ≥ 1 ∧ (m = (1/(a+1)) + 1/b))) := sorry

end minimum_expression_l72_72323


namespace range_of_k_intersecting_hyperbola_l72_72182

theorem range_of_k_intersecting_hyperbola :
  (∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1) →
  -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 :=
sorry

end range_of_k_intersecting_hyperbola_l72_72182


namespace even_function_value_of_a_l72_72188

theorem even_function_value_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x * (Real.exp x + a * Real.exp (-x))) (h_even : ∀ x : ℝ, f x = f (-x)) : a = -1 := 
by
  sorry

end even_function_value_of_a_l72_72188


namespace fractional_inequality_solution_l72_72719

theorem fractional_inequality_solution :
  ∃ (m n : ℕ), n = m^2 - 1 ∧ 
               (m + 2) / (n + 2 : ℝ) > 1 / 3 ∧ 
               (m - 3) / (n - 3 : ℝ) < 1 / 10 ∧ 
               1 ≤ m ∧ m ≤ 9 ∧ 1 ≤ n ∧ n ≤ 9 ∧ 
               (m = 3) ∧ (n = 8) := 
by
  sorry

end fractional_inequality_solution_l72_72719


namespace adam_total_spending_l72_72798

def first_laptop_cost : ℤ := 500
def second_laptop_cost : ℤ := 3 * first_laptop_cost
def total_cost : ℤ := first_laptop_cost + second_laptop_cost

theorem adam_total_spending : total_cost = 2000 := by
  sorry

end adam_total_spending_l72_72798


namespace line_passes_through_fixed_point_l72_72838

-- Statement to prove that the line always passes through the point (2, 2)
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, ∃ x y : ℝ, 
  (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0 ∧ x = 2 ∧ y = 2 :=
sorry

end line_passes_through_fixed_point_l72_72838


namespace rainfall_difference_l72_72565

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l72_72565


namespace ratio_BF_FC_l72_72295

variable (a b : ℝ) (A B C D E F O : Point)

variables [rect : Rectangle A B C D]
variables [hE : PointOn E (line_through D C)]
variables [hF : PointOn F (line_through B C)]
variables [hRatio_E : SegmentRatio D E E C (2:3)]
variables [hAF_BE : IntersectLinesPoint (line_through A F) (line_through B E) O]
variables [hRatio_AO_OF : SegmentRatio A O O F (5:2)]

theorem ratio_BF_FC : RatioOfSegments B F F C (2:1) :=
sorry

end ratio_BF_FC_l72_72295


namespace toby_steps_needed_l72_72258

noncomputable def total_steps_needed : ℕ := 10000 * 9

noncomputable def first_sunday_steps : ℕ := 10200
noncomputable def first_monday_steps : ℕ := 10400
noncomputable def tuesday_steps : ℕ := 9400
noncomputable def wednesday_steps : ℕ := 9100
noncomputable def thursday_steps : ℕ := 8300
noncomputable def friday_steps : ℕ := 9200
noncomputable def saturday_steps : ℕ := 8900
noncomputable def second_sunday_steps : ℕ := 9500

noncomputable def total_steps_walked := 
  first_sunday_steps + 
  first_monday_steps + 
  tuesday_steps + 
  wednesday_steps + 
  thursday_steps + 
  friday_steps + 
  saturday_steps + 
  second_sunday_steps

noncomputable def remaining_steps_needed := total_steps_needed - total_steps_walked

noncomputable def days_left : ℕ := 3

noncomputable def average_steps_needed := remaining_steps_needed / days_left

theorem toby_steps_needed : average_steps_needed = 5000 := by
  sorry

end toby_steps_needed_l72_72258


namespace sacks_per_day_l72_72066

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (harvest_rate : ℕ)
  (h1 : total_sacks = 498)
  (h2 : days = 6)
  (h3 : harvest_rate = total_sacks / days) :
  harvest_rate = 83 := by
  sorry

end sacks_per_day_l72_72066


namespace smallest_a_l72_72249

def root_product (P : Polynomial ℚ) : ℚ :=
  P.coeff 0

def poly_sum_roots_min_a (r1 r2 r3 : ℤ) (a b c : ℚ) : Prop :=
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
  r1 * r2 * r3 = 2310 ∧
  root_product (Polynomial.monomial 3 1 - Polynomial.monomial 2 a + Polynomial.monomial 1 b - Polynomial.monomial 0 2310) = 2310 ∧
  r1 + r2 + r3 = a

theorem smallest_a : ∃ a b : ℚ, ∀ r1 r2 r3 : ℤ, poly_sum_roots_min_a r1 r2 r3 a b 2310 → a = 28
  by sorry

end smallest_a_l72_72249


namespace polynomial_without_xy_l72_72074

theorem polynomial_without_xy (k : ℝ) (x y : ℝ) :
  ¬(∃ c : ℝ, (x^2 + k * x * y + 4 * x - 2 * x * y + y^2 - 1 = c * x * y)) → k = 2 := by
  sorry

end polynomial_without_xy_l72_72074


namespace value_of_a_l72_72351

-- Definitions based on conditions
def cond1 (a : ℝ) := |a| - 1 = 0
def cond2 (a : ℝ) := a + 1 ≠ 0

-- The main proof problem
theorem value_of_a (a : ℝ) : (cond1 a ∧ cond2 a) → a = 1 :=
by
  sorry

end value_of_a_l72_72351


namespace purchasing_methods_count_l72_72789

theorem purchasing_methods_count :
  ∃ n, n = 6 ∧
    ∃ (x y : ℕ), 
      60 * x + 70 * y ≤ 500 ∧
      x ≥ 3 ∧
      y ≥ 2 :=
sorry

end purchasing_methods_count_l72_72789


namespace union_eq_l72_72479

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end union_eq_l72_72479


namespace third_side_length_l72_72348

theorem third_side_length (x : ℝ) (h1 : 2 + 4 > x) (h2 : 4 + x > 2) (h3 : x + 2 > 4) : x = 4 :=
by {
  sorry
}

end third_side_length_l72_72348


namespace twelve_position_in_circle_l72_72559

theorem twelve_position_in_circle (a : ℕ → ℕ) (h_cyclic : ∀ i, a (i + 20) = a i)
  (h_sum_six : ∀ i, a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) = 24)
  (h_first : a 1 = 1) :
  a 12 = 7 :=
sorry

end twelve_position_in_circle_l72_72559


namespace line_does_not_intersect_circle_l72_72988

theorem line_does_not_intersect_circle (a : ℝ) : 
  (a > 1 ∨ a < -1) → ¬ ∃ (x y : ℝ), (x + y = a) ∧ (x^2 + y^2 = 1) :=
by
  sorry

end line_does_not_intersect_circle_l72_72988


namespace arithmetic_sequence_a9_l72_72632

theorem arithmetic_sequence_a9 (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * a 0 + (n - 1))) →
  S 6 = 3 * S 3 →
  a 9 = 10 := by
  sorry

end arithmetic_sequence_a9_l72_72632


namespace vertices_of_square_l72_72094

-- Define lattice points as points with integer coordinates
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define the distance between two lattice points
def distance (P Q : LatticePoint) : ℤ :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y)

-- Define the area of a triangle formed by three lattice points using the determinant method
def area (P Q R : LatticePoint) : ℤ :=
  (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)

-- Prove that three distinct lattice points form the vertices of a square given the condition
theorem vertices_of_square (P Q R : LatticePoint) (h₀ : P ≠ Q) (h₁ : Q ≠ R) (h₂ : P ≠ R)
    (h₃ : (distance P Q + distance Q R) < 8 * (area P Q R) + 1) :
    ∃ S : LatticePoint, S ≠ P ∧ S ≠ Q ∧ S ≠ R ∧
    (distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P) := 
by sorry

end vertices_of_square_l72_72094


namespace number_of_people_seated_l72_72709

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l72_72709


namespace base_conversion_subtraction_l72_72441

theorem base_conversion_subtraction :
  let n1_base9 := 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  let n2_base7 := 1 * 7^2 + 6 * 7^1 + 5 * 7^0
  n1_base9 - n2_base7 = 169 :=
by
  sorry

end base_conversion_subtraction_l72_72441


namespace teena_speed_l72_72716

theorem teena_speed (T : ℝ) : 
  (∀ (d₀ d_poe d_ahead : ℝ), 
    d₀ = 7.5 ∧ d_poe = 40 * 1.5 ∧ d_ahead = 15 →
    T = (d₀ + d_poe + d_ahead) / 1.5) → 
  T = 55 :=
by
  intros
  sorry

end teena_speed_l72_72716


namespace total_money_taken_in_l72_72111

-- Define the conditions as constants
def total_tickets : ℕ := 800
def advanced_ticket_price : ℝ := 14.5
def door_ticket_price : ℝ := 22.0
def door_tickets_sold : ℕ := 672
def advanced_tickets_sold : ℕ := total_tickets - door_tickets_sold
def total_revenue_advanced : ℝ := advanced_tickets_sold * advanced_ticket_price
def total_revenue_door : ℝ := door_tickets_sold * door_ticket_price
def total_revenue : ℝ := total_revenue_advanced + total_revenue_door

-- State the mathematical proof problem
theorem total_money_taken_in : total_revenue = 16640.00 := by
  sorry

end total_money_taken_in_l72_72111


namespace find_m_l72_72189

def U : Set Nat := {1, 2, 3}
def A (m : Nat) : Set Nat := {1, m}
def complement (s t : Set Nat) : Set Nat := {x | x ∈ s ∧ x ∉ t}

theorem find_m (m : Nat) (h1 : complement U (A m) = {2}) : m = 3 :=
by
  sorry

end find_m_l72_72189


namespace erica_duration_is_correct_l72_72965

-- Define the durations for Dave, Chuck, and Erica
def dave_duration : ℝ := 10
def chuck_duration : ℝ := 5 * dave_duration
def erica_duration : ℝ := chuck_duration + 0.30 * chuck_duration

-- State the theorem
theorem erica_duration_is_correct : erica_duration = 65 := by
  sorry

end erica_duration_is_correct_l72_72965


namespace group_B_equal_l72_72152

noncomputable def neg_two_pow_three := (-2)^3
noncomputable def minus_two_pow_three := -(2^3)

theorem group_B_equal : neg_two_pow_three = minus_two_pow_three :=
by sorry

end group_B_equal_l72_72152


namespace cylinder_sphere_ratio_l72_72477

theorem cylinder_sphere_ratio (r R : ℝ) (h : 8 * r^2 = 4 * R^2) : R / r = Real.sqrt 2 :=
by
  sorry

end cylinder_sphere_ratio_l72_72477


namespace rainfall_difference_l72_72560

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l72_72560


namespace parallelogram_vector_sum_l72_72208

theorem parallelogram_vector_sum (A B C D : ℝ × ℝ) (parallelogram : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ (C - A = D - B) ∧ (B - D = A - C)) :
  (B - A) + (C - B) = C - A :=
by
  sorry

end parallelogram_vector_sum_l72_72208


namespace solution_set_of_inequality_l72_72057

noncomputable def f (x : ℝ) : ℝ := (1 / x) * (1 / 2 * (Real.log x) ^ 2 + 1 / 2)

theorem solution_set_of_inequality :
  (∀ x : ℝ, x > 0 → x < e → f x - x > f e - e) ↔ (∀ x : ℝ, 0 < x ∧ x < e) :=
by
  sorry

end solution_set_of_inequality_l72_72057


namespace regular_hexagon_interior_angle_l72_72742

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l72_72742


namespace inequality_solution_l72_72470

noncomputable def inequality_proof (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 2) : Prop :=
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a)) ≥ (27 / 13)

theorem inequality_solution (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 2) : 
  inequality_proof a b c h_positive h_sum :=
sorry

end inequality_solution_l72_72470


namespace sum_of_readings_ammeters_l72_72169

variables (I1 I2 I3 I4 I5 : ℝ)

noncomputable def sum_of_ammeters (I1 I2 I3 I4 I5 : ℝ) : ℝ :=
  I1 + I2 + I3 + I4 + I5

theorem sum_of_readings_ammeters :
  I1 = 2 ∧ I2 = I1 ∧ I3 = 2 * I1 ∧ I5 = I3 + I1 ∧ I4 = (5 / 3) * I5 →
  sum_of_ammeters I1 I2 I3 I4 I5 = 24 :=
by
  sorry

end sum_of_readings_ammeters_l72_72169


namespace second_train_start_time_l72_72739

-- Define the conditions as hypotheses
def station_distance : ℝ := 200
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def meet_time : ℝ := 12 - 7 -- Time they meet after the first train starts, in hours.

-- The theorem statement corresponding to the proof problem
theorem second_train_start_time :
  ∃ T : ℝ, 0 <= T ∧ T <= 5 ∧ (5 * speed_train_A) + ((5 - T) * speed_train_B) = station_distance → T = 1 :=
by
  -- Placeholder for actual proof
  sorry

end second_train_start_time_l72_72739


namespace min_a1_a7_l72_72079

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem min_a1_a7 (a : ℕ → ℝ) (h : geom_seq a)
  (h1 : a 3 * a 5 = 64) :
  ∃ m, m = (a 1 + a 7) ∧ m = 16 :=
by
  sorry

end min_a1_a7_l72_72079


namespace john_investment_in_bankA_l72_72664

-- Definitions to set up the conditions
def total_investment : ℝ := 1500
def bankA_rate : ℝ := 0.04
def bankB_rate : ℝ := 0.06
def final_amount : ℝ := 1575

-- Definition of the question to be proved
theorem john_investment_in_bankA (x : ℝ) (h : 0 ≤ x ∧ x ≤ total_investment) :
  (x * (1 + bankA_rate) + (total_investment - x) * (1 + bankB_rate) = final_amount) -> x = 750 := sorry


end john_investment_in_bankA_l72_72664


namespace exists_25_pos_integers_l72_72619

theorem exists_25_pos_integers (n : ℕ) :
  (n - 1)*(n - 3)*(n - 5) * ... * (n - 99) < 0 ↔ n ∈ {4, 8, 12, ..., 96}.size = 25 :=
sorry

end exists_25_pos_integers_l72_72619


namespace total_games_l72_72519

-- Define the conditions
def games_this_year : ℕ := 4
def games_last_year : ℕ := 9

-- Define the proposition that we want to prove
theorem total_games : games_this_year + games_last_year = 13 := by
  sorry

end total_games_l72_72519


namespace interior_angle_of_regular_hexagon_is_120_degrees_l72_72765

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l72_72765


namespace not_divisible_by_5_l72_72472

theorem not_divisible_by_5 (b : ℕ) : b = 6 ↔ ¬ (5 ∣ (2 * b ^ 3 - 2 * b ^ 2 + 2 * b - 1)) :=
sorry

end not_divisible_by_5_l72_72472


namespace hyperbola_asymptote_l72_72542

theorem hyperbola_asymptote (x y : ℝ) : 
  (∀ x y : ℝ, (x^2 / 25 - y^2 / 16 = 1) → (y = (4 / 5) * x ∨ y = -(4 / 5) * x)) := 
by 
  sorry

end hyperbola_asymptote_l72_72542


namespace symmetric_points_x_axis_l72_72177

theorem symmetric_points_x_axis (a b : ℤ) 
  (h1 : a - 1 = 2) (h2 : 5 = -(b - 1)) : (a + b) ^ 2023 = -1 := 
by
  -- The proof steps will go here.
  sorry

end symmetric_points_x_axis_l72_72177


namespace harmonic_mean_of_3_6_12_l72_72443

-- Defining the harmonic mean function
def harmonic_mean (a b c : ℕ) : ℚ := 
  3 / ((1 / (a : ℚ)) + (1 / (b : ℚ)) + (1 / (c : ℚ)))

-- Stating the theorem
theorem harmonic_mean_of_3_6_12 : harmonic_mean 3 6 12 = 36 / 7 :=
by
  sorry

end harmonic_mean_of_3_6_12_l72_72443


namespace production_average_lemma_l72_72935

theorem production_average_lemma (n : ℕ) (h1 : 50 * n + 60 = 55 * (n + 1)) : n = 1 :=
by
  sorry

end production_average_lemma_l72_72935


namespace total_people_seated_l72_72691

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l72_72691


namespace maria_miles_after_second_stop_l72_72454

theorem maria_miles_after_second_stop (total_distance : ℕ)
    (h1 : total_distance = 360)
    (distance_first_stop : ℕ)
    (h2 : distance_first_stop = total_distance / 2)
    (remaining_distance_after_first_stop : ℕ)
    (h3 : remaining_distance_after_first_stop = total_distance - distance_first_stop)
    (distance_second_stop : ℕ)
    (h4 : distance_second_stop = remaining_distance_after_first_stop / 4)
    (remaining_distance_after_second_stop : ℕ)
    (h5 : remaining_distance_after_second_stop = remaining_distance_after_first_stop - distance_second_stop) :
    remaining_distance_after_second_stop = 135 := by
  sorry

end maria_miles_after_second_stop_l72_72454


namespace convert_angle_degrees_to_radians_l72_72447

theorem convert_angle_degrees_to_radians :
  ∃ (k : ℤ) (α : ℝ), -1125 * (Real.pi / 180) = 2 * k * Real.pi + α ∧ 0 ≤ α ∧ α < 2 * Real.pi ∧ (-8 * Real.pi + 7 * Real.pi / 4) = 2 * k * Real.pi + α :=
by {
  sorry
}

end convert_angle_degrees_to_radians_l72_72447


namespace average_percentage_decrease_l72_72151

theorem average_percentage_decrease :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 100 * (1 - x)^2 = 81 ∧ x = 0.1 :=
by
  sorry

end average_percentage_decrease_l72_72151


namespace train_speed_l72_72415

theorem train_speed
  (length_of_train : ℝ)
  (time_to_cross_pole : ℝ)
  (h1 : length_of_train = 3000)
  (h2 : time_to_cross_pole = 120) :
  length_of_train / time_to_cross_pole = 25 :=
by {
  sorry
}

end train_speed_l72_72415


namespace binom_sum_l72_72029

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_sum : binom 7 4 + binom 6 5 = 41 := by
  sorry

end binom_sum_l72_72029


namespace simplify_and_evaluate_expression_l72_72385

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.pi^0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expression_l72_72385


namespace minimum_components_needed_l72_72143

-- Define the parameters of the problem
def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 7
def fixed_monthly_cost : ℝ := 16500
def selling_price_per_component : ℝ := 198.33

-- Define the total cost as a function of the number of components
def total_cost (x : ℝ) : ℝ :=
  fixed_monthly_cost + (production_cost_per_component + shipping_cost_per_component) * x

-- Define the revenue as a function of the number of components
def revenue (x : ℝ) : ℝ :=
  selling_price_per_component * x

-- Define the theorem to be proved
theorem minimum_components_needed (x : ℝ) : x = 149 ↔ total_cost x ≤ revenue x := sorry

end minimum_components_needed_l72_72143


namespace interior_angle_of_regular_hexagon_is_120_degrees_l72_72767

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l72_72767


namespace smallest_prime_dividing_7pow15_plus_9pow17_l72_72908

theorem smallest_prime_dividing_7pow15_plus_9pow17 :
  Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p → p ∣ (7^15 + 9^17) → 2 ≤ p) :=
by
  sorry

end smallest_prime_dividing_7pow15_plus_9pow17_l72_72908


namespace bob_km_per_gallon_l72_72299

-- Define the total distance Bob can drive.
def total_distance : ℕ := 100

-- Define the total amount of gas in gallons Bob's car uses.
def total_gas : ℕ := 10

-- Define the expected kilometers per gallon
def expected_km_per_gallon : ℕ := 10

-- Define the statement we want to prove
theorem bob_km_per_gallon : total_distance / total_gas = expected_km_per_gallon :=
by 
  sorry

end bob_km_per_gallon_l72_72299


namespace highest_value_of_a_divisible_by_8_l72_72815

theorem highest_value_of_a_divisible_by_8 :
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (8 ∣ (100 * a + 16)) ∧ 
  (∀ (b : ℕ), (0 ≤ b ∧ b ≤ 9) → 8 ∣ (100 * b + 16) → b ≤ a) :=
sorry

end highest_value_of_a_divisible_by_8_l72_72815


namespace cycling_problem_l72_72973

theorem cycling_problem (x : ℚ) (h1 : 25 * x + 15 * (7 - x) = 140) : x = 7 / 2 := 
sorry

end cycling_problem_l72_72973


namespace katie_baked_5_cookies_l72_72319

theorem katie_baked_5_cookies (cupcakes cookies sold left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : sold = 4) 
  (h3 : left = 8) 
  (h4 : cupcakes + cookies = sold + left) : 
  cookies = 5 :=
by sorry

end katie_baked_5_cookies_l72_72319


namespace sum_of_cubes_pattern_l72_72869

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  sorry

end sum_of_cubes_pattern_l72_72869


namespace senior_ticket_cost_l72_72574

variable (tickets_total : ℕ)
variable (adult_ticket_price senior_ticket_price : ℕ)
variable (total_receipts : ℕ)
variable (senior_tickets_sold : ℕ)

theorem senior_ticket_cost (h1 : tickets_total = 529) 
                           (h2 : adult_ticket_price = 25)
                           (h3 : total_receipts = 9745)
                           (h4 : senior_tickets_sold = 348) 
                           (h5 : senior_ticket_price * 348 + 25 * (529 - 348) = 9745) : 
                           senior_ticket_price = 15 := by
  sorry

end senior_ticket_cost_l72_72574


namespace number_is_multiple_of_15_l72_72356

theorem number_is_multiple_of_15
  (W X Y Z D : ℤ)
  (h1 : X - W = 1)
  (h2 : Y - W = 9)
  (h3 : Y - X = 8)
  (h4 : Z - W = 11)
  (h5 : Z - X = 10)
  (h6 : Z - Y = 2)
  (hD : D - X = 5) :
  15 ∣ D :=
by
  sorry -- Proof goes here

end number_is_multiple_of_15_l72_72356


namespace michael_remaining_yards_l72_72153

theorem michael_remaining_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (m y : ℕ)
  (h1 : miles_per_marathon = 50)
  (h2 : yards_per_marathon = 800)
  (h3 : yards_per_mile = 1760)
  (h4 : num_marathons = 5)
  (h5 : y = (yards_per_marathon * num_marathons) % yards_per_mile)
  (h6 : m = miles_per_marathon * num_marathons + (yards_per_marathon * num_marathons) / yards_per_mile) :
  y = 480 :=
sorry

end michael_remaining_yards_l72_72153


namespace tanya_erasers_l72_72333

theorem tanya_erasers (H R TR T : ℕ) 
  (h1 : H = 2 * R) 
  (h2 : R = TR / 2 - 3) 
  (h3 : H = 4) 
  (h4 : TR = T / 2) : 
  T = 20 := 
by 
  sorry

end tanya_erasers_l72_72333


namespace evaluate_infinite_series_l72_72614

noncomputable def infinite_series (n : ℕ) : ℝ := (n^2) / (3^n)

theorem evaluate_infinite_series :
  (∑' k : ℕ, infinite_series (k+1)) = 4.5 :=
by sorry

end evaluate_infinite_series_l72_72614


namespace total_people_seated_l72_72694

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l72_72694


namespace no_real_x_condition_l72_72465

theorem no_real_x_condition (x : ℝ) : 
(∃ a b : ℕ, 4 * x^5 - 7 = a^2 ∧ 4 * x^13 - 7 = b^2) → false := 
by {
  sorry
}

end no_real_x_condition_l72_72465


namespace Felicity_used_23_gallons_l72_72462

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l72_72462


namespace find_a_degree_l72_72502

-- Definitions from conditions
def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

-- Statement of the proof problem
theorem find_a_degree (a : ℕ) (h : monomial_degree 2 a = 6) : a = 4 :=
by
  sorry

end find_a_degree_l72_72502


namespace geometric_seq_min_3b2_7b3_l72_72523

theorem geometric_seq_min_3b2_7b3 (b_1 b_2 b_3 : ℝ) (r : ℝ) 
  (h_seq : b_1 = 2) (h_geom : b_2 = b_1 * r) (h_geom2 : b_3 = b_1 * r^2) :
  3 * b_2 + 7 * b_3 ≥ -16 / 7 :=
by
  -- Include the necessary definitions to support the setup
  have h_b1 : b_1 = 2 := h_seq
  have h_b2 : b_2 = 2 * r := by rw [h_geom, h_b1]
  have h_b3 : b_3 = 2 * r^2 := by rw [h_geom2, h_b1]
  sorry

end geometric_seq_min_3b2_7b3_l72_72523


namespace no_perfect_square_for_nnplus1_l72_72858

theorem no_perfect_square_for_nnplus1 :
  ¬ ∃ (n : ℕ), 0 < n ∧ ∃ (k : ℕ), n * (n + 1) = k * k :=
sorry

end no_perfect_square_for_nnplus1_l72_72858


namespace find_smallest_x_l72_72317

noncomputable def smallest_pos_real_x : ℝ :=
  55 / 7

theorem find_smallest_x (x : ℝ) (h : x > 0) (hx : ⌊x^2⌋ - x * ⌊x⌋ = 6) : x = smallest_pos_real_x :=
  sorry

end find_smallest_x_l72_72317


namespace hyperbola_eccentricity_eq_two_l72_72242

theorem hyperbola_eccentricity_eq_two :
  (∀ x y : ℝ, ((x^2 / 2) - (y^2 / 6) = 1) → 
    let a_squared := 2
    let b_squared := 6
    let a := Real.sqrt a_squared
    let b := Real.sqrt b_squared
    let e := Real.sqrt (1 + b_squared / a_squared)
    e = 2) := 
sorry

end hyperbola_eccentricity_eq_two_l72_72242


namespace insurance_compensation_l72_72279

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end insurance_compensation_l72_72279


namespace regular_hexagon_interior_angle_measure_l72_72751

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l72_72751


namespace equal_sum_seq_value_at_18_l72_72797

-- Define what it means for a sequence to be an equal-sum sequence with a common sum
def equal_sum_seq (a : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_seq_value_at_18
  (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : equal_sum_seq a 5) :
  a 18 = 3 :=
sorry

end equal_sum_seq_value_at_18_l72_72797


namespace harry_total_payment_in_silvers_l72_72848

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end harry_total_payment_in_silvers_l72_72848


namespace friends_can_reach_destinations_l72_72393

/-- The distance between Coco da Selva and Quixajuba is 24 km. 
    The walking speed is 6 km/h and the biking speed is 18 km/h. 
    Show that the friends can proceed to reach their destinations in at most 2 hours 40 minutes, with the bicycle initially in Quixajuba. -/
theorem friends_can_reach_destinations (d q c : ℕ) (vw vb : ℕ) (h1 : d = 24) (h2 : vw = 6) (h3 : vb = 18): 
  (∃ ta tb tc : ℕ, ta ≤ 2 * 60 + 40 ∧ tb ≤ 2 * 60 + 40 ∧ tc ≤ 2 * 60 + 40 ∧ 
     True) :=
sorry

end friends_can_reach_destinations_l72_72393


namespace not_perfect_square_4_2021_l72_72134

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * x

-- State the non-perfect square problem for the given choices
theorem not_perfect_square_4_2021 :
  ¬ is_perfect_square (4 ^ 2021) ∧
  is_perfect_square (1 ^ 2018) ∧
  is_perfect_square (6 ^ 2020) ∧
  is_perfect_square (5 ^ 2022) :=
by
  sorry

end not_perfect_square_4_2021_l72_72134


namespace share_of_e_l72_72934

variable (E F : ℝ)
variable (D : ℝ := (5/3) * E)
variable (D_alt : ℝ := (1/2) * F)
variable (E_alt : ℝ := (3/2) * F)
variable (profit : ℝ := 25000)

theorem share_of_e (h1 : D = (5/3) * E) (h2 : D = (1/2) * F) (h3 : E = (3/2) * F) :
  (E / ((5/2) * F + (3/2) * F + F)) * profit = 7500 :=
by
  sorry

end share_of_e_l72_72934


namespace proportionality_intersect_calculation_l72_72649

variables {x1 x2 y1 y2 : ℝ}

/-- Proof that (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15,
    given specific conditions on x1, x2, y1, and y2. -/
theorem proportionality_intersect_calculation
  (h1 : y1 = 5 / x1) 
  (h2 : y2 = 5 / x2)
  (h3 : x1 * y1 = 5)
  (h4 : x2 * y2 = 5)
  (h5 : x1 = -x2)
  (h6 : y1 = -y2) :
  (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15 := 
sorry

end proportionality_intersect_calculation_l72_72649


namespace add_to_frac_eq_l72_72920

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l72_72920


namespace union_of_sets_l72_72481

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end union_of_sets_l72_72481


namespace correct_conclusion_l72_72898

noncomputable def proof_problem (a x : ℝ) (x1 x2 : ℝ) :=
  (a * (x - 1) * (x - 3) + 2 > 0 ∧ x1 < x2 ∧ 
   (∀ x, a * (x - 1) * (x - 3) + 2 > 0 ↔ x < x1 ∨ x > x2)) →
  (x1 + x2 = 4 ∧ 3 < x1 * x2 ∧ x1 * x2 < 4 ∧ 
   (∀ x, ((3 * a + 2) * x^2 - 4 * a * x + a < 0) ↔ (1 / x2 < x ∧ x < 1 / x1)))

theorem correct_conclusion (a x x1 x2 : ℝ) : 
proof_problem a x x1 x2 :=
by 
  unfold proof_problem 
  sorry

end correct_conclusion_l72_72898


namespace fraction_power_evaluation_l72_72339

theorem fraction_power_evaluation (x y : ℚ) (h1 : x = 2 / 3) (h2 : y = 3 / 2) : 
  (3 / 4) * x^8 * y^9 = 9 / 8 := 
by
  sorry

end fraction_power_evaluation_l72_72339


namespace max_tan_B_l72_72056

theorem max_tan_B (A B : ℝ) (C : Prop) 
  (sin_pos_A : 0 < Real.sin A) 
  (sin_pos_B : 0 < Real.sin B) 
  (angle_condition : Real.sin B / Real.sin A = Real.cos (A + B)) :
  Real.tan B ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l72_72056


namespace pq_implies_q_l72_72850

theorem pq_implies_q (p q : Prop) (h₁ : p ∨ q) (h₂ : ¬p) : q :=
by
  sorry

end pq_implies_q_l72_72850


namespace total_carrots_l72_72529

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end total_carrots_l72_72529


namespace abs_inequality_solution_l72_72235

theorem abs_inequality_solution (x : ℝ) :
  |2 * x - 2| + |2 * x + 4| < 10 ↔ x ∈ Set.Ioo (-4 : ℝ) (2 : ℝ) := 
by sorry

end abs_inequality_solution_l72_72235


namespace only_integers_square_less_than_three_times_l72_72128

-- We want to prove that the only integers n that satisfy n^2 < 3n are 1 and 2.
theorem only_integers_square_less_than_three_times (n : ℕ) (h : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
sorry

end only_integers_square_less_than_three_times_l72_72128


namespace roots_of_quadratic_equation_are_real_and_distinct_l72_72554

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l72_72554


namespace smallest_integer_with_conditions_l72_72894

theorem smallest_integer_with_conditions (x : ℕ) : 
  (∃ x, x.factors.count = 18 ∧ 18 ∣ x ∧ 24 ∣ x) → x = 972 :=
by
  sorry

end smallest_integer_with_conditions_l72_72894


namespace minimize_a_plus_b_l72_72053

theorem minimize_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 4 * a + b = 30) :
  a + b = 9 → (a, b) = (7, 2) := sorry

end minimize_a_plus_b_l72_72053


namespace points_and_conditions_proof_l72_72525

noncomputable def points_and_conditions (x y : ℝ) : Prop := 
|x - 3| + |y + 5| = 0

noncomputable def min_AM_BM (m : ℝ) : Prop :=
|3 - m| + |-5 - m| = 7 / 4 * |8|

noncomputable def min_PA_PB (p : ℝ) : Prop :=
|p - 3| + |p + 5| = 8

noncomputable def min_PD_PO (p : ℝ) : Prop :=
|p + 1| - |p| = -1

noncomputable def range_of_a (a : ℝ) : Prop :=
a ∈ Set.Icc (-5) (-1)

theorem points_and_conditions_proof (x y : ℝ) (m p a : ℝ) :
  points_and_conditions x y → 
  x = 3 ∧ y = -5 ∧ 
  ((m = -8 ∨ m = 6) → min_AM_BM m) ∧ 
  (min_PA_PB p) ∧ 
  (min_PD_PO p) ∧ 
  (range_of_a a) :=
by 
  sorry

end points_and_conditions_proof_l72_72525


namespace collinearity_necessary_but_not_sufficient_l72_72573

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u

def equal (u v : V) : Prop := u = v

theorem collinearity_necessary_but_not_sufficient (u v : V) :
  (collinear u v → equal u v) ∧ (equal u v → collinear u v) → collinear u v ∧ ¬(collinear u v ↔ equal u v) :=
sorry

end collinearity_necessary_but_not_sufficient_l72_72573


namespace value_of_y_l72_72193

theorem value_of_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end value_of_y_l72_72193


namespace table_seating_problem_l72_72702

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l72_72702


namespace complex_square_eq_l72_72836

open Complex

theorem complex_square_eq {a b : ℝ} (h : (a + b * Complex.I)^2 = Complex.mk 3 4) : a^2 + b^2 = 5 :=
by {
  sorry
}

end complex_square_eq_l72_72836


namespace trigonometric_order_l72_72612

theorem trigonometric_order :
  (Real.sin 2 > Real.sin 1) ∧
  (Real.sin 1 > Real.sin 3) ∧
  (Real.sin 3 > Real.sin 4) := 
by
  sorry

end trigonometric_order_l72_72612


namespace true_statement_count_l72_72808

def reciprocal (n : ℕ) : ℚ := 1 / n

def statement_i := (reciprocal 4 + reciprocal 8 = reciprocal 12)
def statement_ii := (reciprocal 9 - reciprocal 3 = reciprocal 6)
def statement_iii := (reciprocal 3 * reciprocal 9 = reciprocal 27)
def statement_iv := (reciprocal 15 / reciprocal 3 = reciprocal 5)

theorem true_statement_count :
  (¬statement_i ∧ ¬statement_ii ∧ statement_iii ∧ statement_iv) ↔ (2 = 2) :=
by sorry

end true_statement_count_l72_72808


namespace num_decompositions_144_l72_72023

theorem num_decompositions_144 : ∃ D, D = 45 ∧ 
  (∀ (factors : List ℕ), 
    (∀ x, x ∈ factors → x > 1) ∧ factors.prod = 144 → 
    factors.permutations.length = D) :=
sorry

end num_decompositions_144_l72_72023


namespace rotation_problem_l72_72571

-- Define the coordinates of the points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangles with given vertices
def P : Point := {x := 0, y := 0}
def Q : Point := {x := 0, y := 13}
def R : Point := {x := 17, y := 0}

def P' : Point := {x := 34, y := 26}
def Q' : Point := {x := 46, y := 26}
def R' : Point := {x := 34, y := 0}

-- Rotation parameters
variables (n : ℝ) (x y : ℝ) (h₀ : 0 < n) (h₁ : n < 180)

-- The mathematical proof problem
theorem rotation_problem :
  n + x + y = 180 := by
  sorry

end rotation_problem_l72_72571


namespace find_a2023_l72_72051

theorem find_a2023 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a n + a (n + 1) = n) : a 2023 = 1012 :=
sorry

end find_a2023_l72_72051


namespace find_C_l72_72088

noncomputable def h (C D : ℝ) (x : ℝ) : ℝ := 2 * C * x - 3 * D ^ 2
def k (D : ℝ) (x : ℝ) := D * x

theorem find_C (C D : ℝ) (h_eq : h C D (k D 2) = 0) (hD : D ≠ 0) : C = 3 * D / 4 :=
by
  unfold h k at h_eq
  sorry

end find_C_l72_72088


namespace Bernoulli_inequality_l72_72710

theorem Bernoulli_inequality (n : ℕ) (a : ℝ) (h : a > -1) : (1 + a)^n ≥ n * a + 1 := 
sorry

end Bernoulli_inequality_l72_72710


namespace number_of_games_played_l72_72851

-- Define our conditions
def teams : ℕ := 14
def games_per_pair : ℕ := 5

-- Define the function to calculate the number of combinations
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expected total games
def total_games : ℕ := 455

-- Statement asserting that given the conditions, the number of games played in the season is total_games
theorem number_of_games_played : (combinations teams 2) * games_per_pair = total_games := 
by 
  sorry

end number_of_games_played_l72_72851


namespace coeff_x4_in_expansion_correct_l72_72466

noncomputable def coeff_x4_in_expansion (f g : ℕ → ℤ) := 
  ∀ (c : ℤ), c = 80 → f 4 + g 1 * g 3 = c

-- Definitions of the individual polynomials
def poly1 (x : ℤ) : ℤ := 4 * x^2 - 2 * x + 1
def poly2 (x : ℤ) : ℤ := 2 * x + 1

-- Expanded form coefficients
def coeff_poly1 : ℕ → ℤ
  | 0       => 1
  | 1       => -2
  | 2       => 4
  | _       => 0

def coeff_poly2_pow4 : ℕ → ℤ
  | 0       => 1
  | 1       => 8
  | 2       => 24
  | 3       => 32
  | 4       => 16
  | _       => 0

-- The theorem we want to prove
theorem coeff_x4_in_expansion_correct :
  coeff_x4_in_expansion coeff_poly1 coeff_poly2_pow4 := 
by
  sorry

end coeff_x4_in_expansion_correct_l72_72466


namespace train_cross_time_l72_72358

noncomputable def speed_conversion (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def time_to_cross_pole (length_m speed_kmh : ℝ) : ℝ :=
  length_m / speed_conversion speed_kmh

theorem train_cross_time (length_m : ℝ) (speed_kmh : ℝ) :
  length_m = 225 → speed_kmh = 250 → time_to_cross_pole length_m speed_kmh = 3.24 := by
  intros hlen hspeed
  simp [time_to_cross_pole, speed_conversion, hlen, hspeed]
  sorry

end train_cross_time_l72_72358


namespace chapters_page_difference_l72_72282

def chapter1_pages : ℕ := 37
def chapter2_pages : ℕ := 80

theorem chapters_page_difference : chapter2_pages - chapter1_pages = 43 := by
  -- Proof goes here
  sorry

end chapters_page_difference_l72_72282


namespace white_ducks_count_l72_72819

theorem white_ducks_count (W : ℕ) : 
  (5 * W + 10 * 7 + 12 * 6 = 157) → W = 3 :=
by
  sorry

end white_ducks_count_l72_72819


namespace erica_riding_time_is_65_l72_72963

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

end erica_riding_time_is_65_l72_72963


namespace square_field_area_l72_72576

def square_area (side_length : ℝ) : ℝ :=
  side_length * side_length

theorem square_field_area :
  square_area 20 = 400 := by
  sorry

end square_field_area_l72_72576


namespace regular_hexagon_interior_angle_l72_72746

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l72_72746


namespace max_value_k_eq_1_range_k_no_zeros_l72_72062

-- Define the function f(x)
noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Note: 'by' and 'sorry' are placeholders to skip the proof; actual proofs are not required.

-- Proof Problem 1: Prove that when k = 1, the maximum value of f(x) is 0.
theorem max_value_k_eq_1 : ∀ x : ℝ, 1 < x → f x 1 ≤ 0 := 
by
  sorry

-- Proof Problem 2: Prove that k ∈ (1, +∞) is the range such that f(x) has no zeros.
theorem range_k_no_zeros : ∀ k : ℝ, (∀ x : ℝ, 1 < x → f x k ≠ 0) → 1 < k :=
by
  sorry

end max_value_k_eq_1_range_k_no_zeros_l72_72062


namespace equilateral_right_triangle_impossible_l72_72580
-- Import necessary library

-- Define the conditions and the problem statement
theorem equilateral_right_triangle_impossible :
  ¬(∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A = B ∧ B = C ∧ (A^2 + B^2 = C^2) ∧ (A + B + C = 180)) := sorry

end equilateral_right_triangle_impossible_l72_72580


namespace remainder_of_sum_of_ns_l72_72667

theorem remainder_of_sum_of_ns (S : ℕ) :
  (∃ (ns : List ℕ), (∀ n ∈ ns, ∃ m : ℕ, n^2 + 12*n - 1997 = m^2) ∧ S = ns.sum) →
  S % 1000 = 154 :=
by
  sorry

end remainder_of_sum_of_ns_l72_72667


namespace multiplication_as_sum_of_squares_l72_72665

theorem multiplication_as_sum_of_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end multiplication_as_sum_of_squares_l72_72665


namespace inverse_proposition_l72_72720

   theorem inverse_proposition (x a b : ℝ) :
     (x ≥ a^2 + b^2 → x ≥ 2 * a * b) →
     (x ≥ 2 * a * b → x ≥ a^2 + b^2) :=
   sorry
   
end inverse_proposition_l72_72720


namespace triangle_inequality_l72_72198

theorem triangle_inequality (a : ℝ) (h1 : a + 3 > 5) (h2 : a + 5 > 3) (h3 : 3 + 5 > a) :
  2 < a ∧ a < 8 :=
by {
  sorry
}

end triangle_inequality_l72_72198


namespace quadratic_has_two_distinct_real_roots_l72_72551

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l72_72551


namespace digits_same_l72_72616

theorem digits_same (k : ℕ) (hk : k ≥ 2) :
  (∃ n : ℕ, (10^(10^n) - 9^(9^n)) % (10^k) = 0) ↔ (k = 2 ∨ k = 3 ∨ k = 4) :=
sorry

end digits_same_l72_72616


namespace bigger_number_l72_72931

theorem bigger_number (yoongi : ℕ) (jungkook : ℕ) (h1 : yoongi = 4) (h2 : jungkook = 6 + 3) : jungkook > yoongi :=
by
  sorry

end bigger_number_l72_72931


namespace find_number_to_add_l72_72915

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l72_72915


namespace score_seventy_five_can_be_achieved_three_ways_l72_72080

-- Defining the problem constraints and goal
def quiz_problem (c u i : ℕ) (S : ℝ) : Prop :=
  c + u + i = 20 ∧ S = 5 * (c : ℝ) + 1.5 * (u : ℝ)

theorem score_seventy_five_can_be_achieved_three_ways :
  ∃ (c1 u1 c2 u2 c3 u3 : ℕ), 0 ≤ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ∧ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ≤ 100 ∧
  (5 * (c2 : ℝ) + 1.5 * (u2 : ℝ)) = 75 ∧ (5 * (c3 : ℝ) + 1.5 * (u3 : ℝ)) = 75 ∧
  (c1 ≠ c2 ∧ u1 ≠ u2) ∧ (c2 ≠ c3 ∧ u2 ≠ u3) ∧ (c3 ≠ c1 ∧ u3 ≠ u1) ∧ 
  quiz_problem c1 u1 (20 - c1 - u1) 75 ∧
  quiz_problem c2 u2 (20 - c2 - u2) 75 ∧
  quiz_problem c3 u3 (20 - c3 - u3) 75 :=
sorry

end score_seventy_five_can_be_achieved_three_ways_l72_72080


namespace find_FC_l72_72047

-- Define all given values and relationships
variables (DC CB AD AB ED FC : ℝ)
variables (h1 : DC = 9) (h2 : CB = 6)
variables (h3 : AB = (1/3) * AD)
variables (h4 : ED = (2/3) * AD)

-- Define the goal
theorem find_FC :
  FC = 9 :=
sorry

end find_FC_l72_72047


namespace original_number_l72_72943

theorem original_number (N : ℕ) (a b c d e : ℕ)
  (hN : N = 10^4 * a + 10^3 * b + 10^2 * c + 10^1 * d + e)
  (h1 : N + (10^3 * b + 10^2 * c + 10^1 * d + e) = 54321 ∨
        N + (10^4 * a + 10^2 * c + 10^1 * d + e) = 54321 ∨
        N + (10^4 * a + 10^3 * b + 10^1 * d + e) = 54321 ∨
        N + (10^4 * a + 10^3 * b + 10^2 * c + e) = 54321 ∨
        N + (10^4 * a + 10^3 * b + 10^2 * c + 10^0 * d) = 54321) :
  N = 49383 :=
begin
  sorry
end

end original_number_l72_72943


namespace k_of_neg7_l72_72089

noncomputable def h (x : ℝ) : ℝ := 4 * x - 9
noncomputable def k (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 2

theorem k_of_neg7 : k (-7) = 3 / 4 :=
by
  sorry

end k_of_neg7_l72_72089


namespace union_eq_l72_72480

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end union_eq_l72_72480


namespace largest_k_inequality_l72_72165

theorem largest_k_inequality
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_pos : (a + b) * (b + c) * (c + a) > 0) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a ≥ 
  (1 / 2) * abs ((a^3 - b^3) / (a + b) + (b^3 - c^3) / (b + c) + (c^3 - a^3) / (c + a)) :=
by
  sorry

end largest_k_inequality_l72_72165


namespace problem_statement_l72_72039

theorem problem_statement :
  ((8^5 / 8^2) * 2^10 - 2^2) = 2^19 - 4 := 
by 
  sorry

end problem_statement_l72_72039


namespace find_number_to_add_l72_72914

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l72_72914


namespace total_money_spent_l72_72662

theorem total_money_spent {s j : ℝ} (hs : s = 14.28) (hj : j = 4.74) : s + j = 19.02 :=
by
  sorry

end total_money_spent_l72_72662


namespace total_weight_of_5_moles_of_cai2_l72_72773

-- Definitions based on the conditions
def weight_of_calcium : Real := 40.08
def weight_of_iodine : Real := 126.90
def iodine_atoms_in_cai2 : Nat := 2
def moles_of_calcium_iodide : Nat := 5

-- Lean 4 statement for the proof problem
theorem total_weight_of_5_moles_of_cai2 :
  (weight_of_calcium + (iodine_atoms_in_cai2 * weight_of_iodine)) * moles_of_calcium_iodide = 1469.4 := by
  sorry

end total_weight_of_5_moles_of_cai2_l72_72773


namespace not_sufficient_nor_necessary_geometric_seq_l72_72661

theorem not_sufficient_nor_necessary_geometric_seq {a : ℕ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q) :
    (a 1 < a 3) ↔ (¬(a 2 < a 4) ∨ ¬(a 4 < a 2)) :=
by
  sorry

end not_sufficient_nor_necessary_geometric_seq_l72_72661


namespace zoo_ticket_sales_l72_72141

-- Define the number of total people, number of adults, and ticket prices
def total_people : ℕ := 254
def num_adults : ℕ := 51
def adult_ticket_price : ℕ := 28
def kid_ticket_price : ℕ := 12

-- Define the number of kids as the difference between total people and number of adults
def num_kids : ℕ := total_people - num_adults

-- Define the revenue from adult tickets and kid tickets
def revenue_adult_tickets : ℕ := num_adults * adult_ticket_price
def revenue_kid_tickets : ℕ := num_kids * kid_ticket_price

-- Define the total revenue
def total_revenue : ℕ := revenue_adult_tickets + revenue_kid_tickets

-- Theorem to prove the total revenue equals 3864
theorem zoo_ticket_sales : total_revenue = 3864 :=
  by {
    -- sorry allows us to skip the proof
    sorry
  }

end zoo_ticket_sales_l72_72141


namespace loaves_at_start_l72_72952

variable (X : ℕ) -- X represents the number of loaves at the start of the day.

-- Conditions given in the problem:
def final_loaves (X : ℕ) : Prop := X - 629 + 489 = 2215

-- The theorem to be proved:
theorem loaves_at_start (h : final_loaves X) : X = 2355 :=
by sorry

end loaves_at_start_l72_72952


namespace harry_total_payment_in_silvers_l72_72847

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end harry_total_payment_in_silvers_l72_72847


namespace seated_people_count_l72_72686

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l72_72686


namespace count_solutions_eq_4_l72_72891

theorem count_solutions_eq_4 :
  ∀ x : ℝ, (x^2 - 5)^2 = 16 → x = 3 ∨ x = -3 ∨ x = 1 ∨ x = -1  := sorry

end count_solutions_eq_4_l72_72891


namespace sin_225_plus_alpha_l72_72636

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 5 / 13) :
    Real.sin (5 * Real.pi / 4 + α) = -5 / 13 :=
by
  sorry

end sin_225_plus_alpha_l72_72636


namespace scientific_notation_of_570_million_l72_72540

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end scientific_notation_of_570_million_l72_72540


namespace production_today_l72_72630

-- Conditions
def average_daily_production_past_n_days (P : ℕ) (n : ℕ) := P = n * 50
def new_average_daily_production (P : ℕ) (T : ℕ) (new_n : ℕ) := (P + T) / new_n = 55

-- Values from conditions
def n := 11
def P := 11 * 50

-- Mathematically equivalent proof problem
theorem production_today :
  ∃ (T : ℕ), average_daily_production_past_n_days P n ∧ new_average_daily_production P T 12 → T = 110 :=
by
  sorry

end production_today_l72_72630


namespace james_marbles_left_l72_72363

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end james_marbles_left_l72_72363


namespace basketball_lineup_ways_l72_72226

theorem basketball_lineup_ways :
  let players := 16
  let twins := 2 -- Betty and Bobbi
  let seniors := 5
  let lineup_size := 7
  (∃ (ways : ℕ), 
    ways = (2 * (binomial seniors 2 * binomial (players - twins - seniors) 4 +
                 binomial seniors 3 * binomial (players - twins - seniors) 3)) ∧
    ways = 4200)
:=
begin
  sorry
end

end basketball_lineup_ways_l72_72226


namespace hyperbola_eccentricity_l72_72241

noncomputable def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 + (b^2) / (a^2))

theorem hyperbola_eccentricity :
  (eccentricity (real.sqrt 2) (real.sqrt 6)) = 2 := by
sorry

end hyperbola_eccentricity_l72_72241


namespace pencil_cost_l72_72569

theorem pencil_cost (p q : ℤ) (H1 : 3 * p + 4 * q = 287) (H2 : 5 * p + 2 * q = 236) : q = 52 :=
by
  -- Set up the system of linear equations
  let eq1 := H1
  let eq2 := H2

  -- Manipulate the equations (steps omitted for brevity)
  sorry

end pencil_cost_l72_72569


namespace trihedral_angle_plane_angles_acute_l72_72799

open Real

-- Define what it means for an angle to be acute
def is_acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Define the given conditions
variable {A B C α β γ : ℝ}
variable (hA : is_acute A)
variable (hB : is_acute B)
variable (hC : is_acute C)

-- State the problem: if dihedral angles are acute, then plane angles are also acute
theorem trihedral_angle_plane_angles_acute :
  is_acute A → is_acute B → is_acute C → is_acute α ∧ is_acute β ∧ is_acute γ :=
sorry

end trihedral_angle_plane_angles_acute_l72_72799


namespace find_fraction_l72_72409

theorem find_fraction (x : ℝ) (h1 : 7 = (1 / 10) / 100 * 7000) (h2 : x * 7000 - 7 = 700) : x = 707 / 7000 :=
by sorry

end find_fraction_l72_72409


namespace linear_equation_a_ne_1_l72_72995

theorem linear_equation_a_ne_1 (a : ℝ) : (∀ x : ℝ, (a - 1) * x - 6 = 0 → a ≠ 1) :=
sorry

end linear_equation_a_ne_1_l72_72995


namespace solve_fraction_problem_l72_72911

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l72_72911


namespace inequality_nonnegative_reals_l72_72468

theorem inequality_nonnegative_reals (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 :=
sorry

end inequality_nonnegative_reals_l72_72468


namespace scientific_notation_570_million_l72_72537

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end scientific_notation_570_million_l72_72537


namespace Q_2_plus_Q_neg_2_l72_72670

noncomputable def cubic_polynomial (a b c k : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + k

theorem Q_2_plus_Q_neg_2 (a b c k : ℝ) 
  (h0 : cubic_polynomial a b c k 0 = k)
  (h1 : cubic_polynomial a b c k 1 = 3 * k)
  (hneg1 : cubic_polynomial a b c k (-1) = 4 * k) :
  cubic_polynomial a b c k 2 + cubic_polynomial a b c k (-2) = 22 * k :=
sorry

end Q_2_plus_Q_neg_2_l72_72670


namespace solve_quadratic1_solve_quadratic2_l72_72386

-- For the first quadratic equation: 3x^2 = 6x
theorem solve_quadratic1 (x : ℝ) (h : 3 * x^2 = 6 * x) : x = 0 ∨ x = 2 :=
sorry

-- For the second quadratic equation: x^2 - 6x + 5 = 0
theorem solve_quadratic2 (x : ℝ) (h : x^2 - 6 * x + 5 = 0) : x = 5 ∨ x = 1 :=
sorry

end solve_quadratic1_solve_quadratic2_l72_72386


namespace original_amount_of_rice_l72_72718

theorem original_amount_of_rice
  (x : ℕ) -- the total amount of rice in kilograms
  (h1 : x = 10 * 500) -- statement that needs to be proven
  (h2 : 210 = x * (21 / 50)) -- remaining rice condition after given fractions are consumed
  (consume_day_one : x - (3 / 10) * x  = (7 / 10) * x) -- after the first day's consumption
  (consume_day_two : ((7 / 10) * x) - ((2 / 5) * ((7 / 10) * x)) = 210) -- after the second day's consumption
  : x = 500 :=
by
  sorry

end original_amount_of_rice_l72_72718


namespace perpendicular_vectors_m_eq_half_l72_72493

theorem perpendicular_vectors_m_eq_half (m : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, m)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = 1 / 2 :=
sorry

end perpendicular_vectors_m_eq_half_l72_72493


namespace find_value_of_m_l72_72725

-- Define the quadratic function and the values in the given table
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
variables (a b c m : ℝ)
variables (h1 : quadratic_function a b c (-1) = m)
variables (h2 : quadratic_function a b c 0 = 2)
variables (h3 : quadratic_function a b c 1 = 1)
variables (h4 : quadratic_function a b c 2 = 2)
variables (h5 : quadratic_function a b c 3 = 5)
variables (h6 : quadratic_function a b c 4 = 10)

-- Theorem stating that the value of m is 5
theorem find_value_of_m : m = 5 :=
by
  sorry

end find_value_of_m_l72_72725


namespace side_length_S2_l72_72683

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end side_length_S2_l72_72683


namespace function_at_neg_one_zero_l72_72486

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l72_72486


namespace ratio_of_points_to_away_home_game_l72_72222

-- Definitions
def first_away_game_points (A : ℕ) : ℕ := A
def second_away_game_points (A : ℕ) : ℕ := A + 18
def third_away_game_points (A : ℕ) : ℕ := A + 20
def last_home_game_points : ℕ := 62
def next_game_points : ℕ := 55
def total_points (A : ℕ) : ℕ := A + (A + 18) + (A + 20) + 62 + 55

-- Given that the total points should be four times the points of the last home game
def target_points : ℕ := 4 * 62

-- The main theorem to prove
theorem ratio_of_points_to_away_home_game : ∀ A : ℕ,
  total_points A = target_points → 62 = 2 * A :=
by
  sorry

end ratio_of_points_to_away_home_game_l72_72222


namespace max_value_of_abc_sum_l72_72484

theorem max_value_of_abc_sum (a b c : ℕ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a < b ∧ b < c) (h3 : b^2 = a * c) (h4 : log 2016 (a * b * c) = 3) :
  a + b + c ≤ 4066273 :=
sorry

end max_value_of_abc_sum_l72_72484


namespace typing_speed_ratio_l72_72266

theorem typing_speed_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.25 * M = 14) : M / T = 2 :=
by
  sorry

end typing_speed_ratio_l72_72266


namespace initial_percentage_water_l72_72034

theorem initial_percentage_water (W_initial W_final N_initial N_final : ℝ) (h1 : W_initial = 100) 
    (h2 : N_initial = W_initial - W_final) (h3 : W_final = 25) (h4 : W_final / N_final = 0.96) : N_initial / W_initial = 0.99 := 
by
  sorry

end initial_percentage_water_l72_72034


namespace relationship_between_P_and_Q_l72_72173

-- Define the sets P and Q
def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem relationship_between_P_and_Q : P ⊇ Q :=
by
  sorry

end relationship_between_P_and_Q_l72_72173


namespace tim_more_points_than_joe_l72_72201

variable (J K T : ℕ)

theorem tim_more_points_than_joe (h1 : T = 30) (h2 : T = K / 2) (h3 : J + T + K = 100) : T - J = 20 :=
by
  sorry

end tim_more_points_than_joe_l72_72201


namespace brendas_age_l72_72149

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end brendas_age_l72_72149


namespace rainfall_difference_l72_72562

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l72_72562


namespace regular_hexagon_interior_angle_l72_72745

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l72_72745


namespace problem_statement_l72_72996

-- Define the repeating decimal and the required gcd condition
def repeating_decimal_value := (356 : ℚ) / 999
def gcd_condition (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the main theorem stating the required sum
theorem problem_statement (a b : ℕ) 
                          (h_a : a = 356) 
                          (h_b : b = 999) 
                          (h_gcd : gcd_condition a b) : 
    a + b = 1355 := by
  sorry

end problem_statement_l72_72996


namespace simplify_expression_l72_72711

variable (y : ℝ)

theorem simplify_expression : (5 * y + 6 * y + 7 * y + 2) = (18 * y + 2) := 
by
  sorry

end simplify_expression_l72_72711


namespace erica_riding_time_is_65_l72_72962

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

end erica_riding_time_is_65_l72_72962


namespace parabola_intersection_at_1_2003_l72_72099

theorem parabola_intersection_at_1_2003 (p q : ℝ) (h : p + q = 2002) :
  (1, (1 : ℝ)^2 + p * 1 + q) = (1, 2003) :=
by
  sorry

end parabola_intersection_at_1_2003_l72_72099


namespace regular_hexagon_interior_angle_l72_72740

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l72_72740


namespace find_first_two_solutions_l72_72321

theorem find_first_two_solutions :
  ∃ (n1 n2 : ℕ), 
    (n1 ≡ 3 [MOD 7]) ∧ (n1 ≡ 4 [MOD 9]) ∧ 
    (n2 ≡ 3 [MOD 7]) ∧ (n2 ≡ 4 [MOD 9]) ∧ 
    n1 < n2 ∧ 
    n1 = 31 ∧ n2 = 94 := 
by 
  sorry

end find_first_two_solutions_l72_72321


namespace isosceles_triangle_side_length_l72_72146

theorem isosceles_triangle_side_length :
  let a := 1
  let b := Real.sqrt 3
  let right_triangle_area := (1 / 2) * a * b
  let isosceles_triangle_area := right_triangle_area / 3
  ∃ s, s = Real.sqrt 109 / 6 ∧ 
    (∀ (base height : ℝ), 
      (base = a / 3 ∨ base = b / 3) ∧
      height = (2 * isosceles_triangle_area) / base → 
      1 / 2 * base * height = isosceles_triangle_area) :=
by
  sorry

end isosceles_triangle_side_length_l72_72146


namespace number_of_solutions_is_zero_l72_72452

theorem number_of_solutions_is_zero : 
  ∀ x : ℝ, (x ≠ 0 ∧ x ≠ 5) → (3 * x^2 - 15 * x) / (x^2 - 5 * x) ≠ x - 2 :=
by
  sorry

end number_of_solutions_is_zero_l72_72452


namespace difference_in_average_speed_l72_72404

theorem difference_in_average_speed 
  (distance : ℕ) 
  (time_diff : ℕ) 
  (speed_B : ℕ) 
  (time_B : ℕ) 
  (time_A : ℕ) 
  (speed_A : ℕ)
  (h1 : distance = 300)
  (h2 : time_diff = 3)
  (h3 : speed_B = 20)
  (h4 : time_B = distance / speed_B)
  (h5 : time_A = time_B - time_diff)
  (h6 : speed_A = distance / time_A) 
  : speed_A - speed_B = 5 := 
sorry

end difference_in_average_speed_l72_72404


namespace ordered_pair_a_c_l72_72549

theorem ordered_pair_a_c (a c : ℝ) (h_quad: ∀ x : ℝ, a * x^2 + 16 * x + c = 0)
    (h_sum: a + c = 25) (h_ineq: a < c) : (a = 3 ∧ c = 22) :=
by
  -- The proof is omitted
  sorry

end ordered_pair_a_c_l72_72549


namespace complement_event_prob_l72_72639

variable {Ω : Type*} [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}

theorem complement_event_prob (A : Set Ω) (hA : P A = 0.5) : P (Aᶜ) = 0.5 := by
  rw [ProbabilityMeasure.compl_eq_one_sub]
  rw [hA]
  norm_num
  -- This theorem states that, given the probability of event A is 0.5,
  -- the probability of its complementary event must also be 0.5.

end complement_event_prob_l72_72639


namespace certain_number_is_2_l72_72344

theorem certain_number_is_2 
    (X : ℕ) 
    (Y : ℕ) 
    (h1 : X = 15) 
    (h2 : 0.40 * (X : ℝ) = 0.80 * 5 + (Y : ℝ)) : 
    Y = 2 := 
  sorry

end certain_number_is_2_l72_72344


namespace first_platform_length_is_150_l72_72954

-- Defining the conditions
def train_length : ℝ := 150
def first_platform_time : ℝ := 15
def second_platform_length : ℝ := 250
def second_platform_time : ℝ := 20

-- The distance covered when crossing the first platform is length of train + length of first platform
def distance_first_platform (L : ℝ) : ℝ := train_length + L

-- The distance covered when crossing the second platform is length of train + length of a known 250 m platform
def distance_second_platform : ℝ := train_length + second_platform_length

-- We are to prove that the length of the first platform, given the conditions, is 150 meters.
theorem first_platform_length_is_150 : ∃ L : ℝ, (distance_first_platform L / distance_second_platform) = (first_platform_time / second_platform_time) ∧ L = 150 :=
by
  let L := 150
  have h1 : distance_first_platform L = train_length + L := rfl
  have h2 : distance_second_platform = train_length + second_platform_length := rfl
  have h3 : distance_first_platform L / distance_second_platform = first_platform_time / second_platform_time :=
    by sorry
  use L
  exact ⟨h3, rfl⟩

end first_platform_length_is_150_l72_72954


namespace interior_angle_of_regular_hexagon_l72_72756

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l72_72756


namespace find_a_l72_72641

noncomputable theory

open Real

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, f x = a * (x + 1) - exp x) ∧
  f' 0 = -2 → a = -1 :=
by
  sorry

def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) - exp x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := (deriv (f a)) x

end find_a_l72_72641


namespace arithmetic_sequence_length_l72_72335

theorem arithmetic_sequence_length :
  ∃ n, (2 + (n - 1) * 5 = 3007) ∧ n = 602 :=
by
  use 602
  sorry

end arithmetic_sequence_length_l72_72335


namespace add_to_fraction_l72_72927

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l72_72927


namespace false_proposition_l72_72635

open Classical

variables (a b : ℝ) (x : ℝ)

def P := ∃ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ (a + b = 1) ∧ ((1 / a) + (1 / b) = 3)
def Q := ∀ (x : ℝ), x^2 - x + 1 ≥ 0

theorem false_proposition :
  (¬ P ∧ ¬ Q) = false → (¬ P ∨ ¬ Q) = true → (¬ P ∨ Q) = true → (¬ P ∧ Q) = true :=
sorry

end false_proposition_l72_72635


namespace integer_roots_of_polynomial_l72_72315

def polynomial : Polynomial ℤ := Polynomial.C 24 + Polynomial.C (-11) * Polynomial.X + Polynomial.C (-4) * Polynomial.X^2 + Polynomial.X^3

theorem integer_roots_of_polynomial :
  Set { x : ℤ | polynomial.eval x polynomial = 0 } = Set ({-4, 3, 8} : Set ℤ) :=
sorry

end integer_roots_of_polynomial_l72_72315


namespace speed_of_goods_train_l72_72269

theorem speed_of_goods_train 
  (t₁ t₂ v_express : ℝ)
  (h1 : v_express = 90) 
  (h2 : t₁ = 6) 
  (h3 : t₂ = 4)
  (h4 : v_express * t₂ = v * (t₁ + t₂)) : 
  v = 36 :=
by
  sorry

end speed_of_goods_train_l72_72269


namespace train_speed_ratio_l72_72253

theorem train_speed_ratio 
  (distance_2nd_train : ℕ)
  (time_2nd_train : ℕ)
  (speed_1st_train : ℚ)
  (H1 : distance_2nd_train = 400)
  (H2 : time_2nd_train = 4)
  (H3 : speed_1st_train = 87.5) :
  distance_2nd_train / time_2nd_train = 100 ∧ 
  (speed_1st_train / (distance_2nd_train / time_2nd_train)) = 7 / 8 :=
by
  sorry

end train_speed_ratio_l72_72253


namespace cost_price_of_cricket_bat_l72_72270

variable (CP_A CP_B SP_C : ℝ)

-- Conditions
def condition1 : CP_B = 1.20 * CP_A := sorry
def condition2 : SP_C = 1.25 * CP_B := sorry
def condition3 : SP_C = 234 := sorry

-- The statement to prove
theorem cost_price_of_cricket_bat : CP_A = 156 := sorry

end cost_price_of_cricket_bat_l72_72270


namespace sum_first_2018_terms_of_given_sequence_l72_72052

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_first_2018_terms_of_given_sequence :
  let a := 1
  let d := -1 / 2017
  S_2018 = 1009 :=
by
  sorry

end sum_first_2018_terms_of_given_sequence_l72_72052


namespace smallest_number_l72_72835

def binary_101010 : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0
def base5_111 : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0
def octal_32 : ℕ := 3 * 8^1 + 2 * 8^0
def base6_54 : ℕ := 5 * 6^1 + 4 * 6^0

theorem smallest_number : octal_32 < binary_101010 ∧ octal_32 < base5_111 ∧ octal_32 < base6_54 :=
by
  sorry

end smallest_number_l72_72835


namespace hardest_work_diff_l72_72112

theorem hardest_work_diff 
  (A B C D : ℕ) 
  (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x ∧ D = 4 * x)
  (h_total : A + B + C + D = 240) :
  (D - A) = 72 :=
by
  sorry

end hardest_work_diff_l72_72112


namespace max_sum_arith_seq_l72_72207

theorem max_sum_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = 8 + (n - 1) * d) →
  d ≠ 0 →
  a 1 = 8 →
  a 5 ^ 2 = a 1 * a 7 →
  S n = n * a 1 + (n * (n - 1) * d) / 2 →
  ∃ n : ℕ, S n = 36 :=
by
  intros
  sorry

end max_sum_arith_seq_l72_72207


namespace sum_x_y_eq_l72_72494

noncomputable def equation (x y : ℝ) : Prop :=
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0

theorem sum_x_y_eq (x y : ℝ) (h : equation x y) : x + y = -9 / 2 :=
by sorry

end sum_x_y_eq_l72_72494


namespace initial_wage_illiterate_l72_72658

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end initial_wage_illiterate_l72_72658


namespace expand_product_l72_72456

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7 * x + 10 := 
by 
  sorry

end expand_product_l72_72456


namespace calc_expression_l72_72156

theorem calc_expression : (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 :=
  sorry

end calc_expression_l72_72156


namespace ratio_yuan_david_l72_72414

-- Definitions
def yuan_age (david_age : ℕ) : ℕ := david_age + 7
def ratio (a b : ℕ) : ℚ := a / b

-- Conditions
variable (david_age : ℕ) (h_david : david_age = 7)

-- Proof Statement
theorem ratio_yuan_david : ratio (yuan_age david_age) david_age = 2 :=
by
  sorry

end ratio_yuan_david_l72_72414


namespace Elza_winning_strategy_l72_72161

-- Define a hypothetical graph structure
noncomputable def cities := {i : ℕ // 1 ≤ i ∧ i ≤ 2013}
def connected (c1 c2 : cities) : Prop := sorry

theorem Elza_winning_strategy 
  (N : ℕ) 
  (roads : (cities × cities) → Prop) 
  (h1 : ∀ c1 c2, roads (c1, c2) → connected c1 c2)
  (h2 : N = 1006): 
  ∃ (strategy : cities → Prop), 
  (∃ c1 c2 : cities, (strategy c1 ∧ strategy c2)) ∧ connected c1 c2 :=
by 
  sorry

end Elza_winning_strategy_l72_72161


namespace rainfall_difference_l72_72564

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l72_72564


namespace seated_people_count_l72_72687

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l72_72687


namespace teorema_dos_bicos_white_gray_eq_angle_x_l72_72933

-- Define the problem statement
theorem teorema_dos_bicos_white_gray_eq
    (n : ℕ)
    (AB CD : ℝ)
    (peaks : Fin n → ℝ)
    (white_angles gray_angles : Fin n → ℝ)
    (h_parallel : AB = CD)
    (h_white_angles : ∀ i, white_angles i = peaks i)
    (h_gray_angles : ∀ i, gray_angles i = peaks i):
    (Finset.univ.sum white_angles) = (Finset.univ.sum gray_angles) := sorry

theorem angle_x
    (AB CD : ℝ)
    (x : ℝ)
    (h_parallel : AB = CD):
    x = 32 := sorry

end teorema_dos_bicos_white_gray_eq_angle_x_l72_72933


namespace speed_of_mans_train_is_80_kmph_l72_72007

-- Define the given constants
def length_goods_train : ℤ := 280 -- length in meters
def time_to_pass : ℤ := 9 -- time in seconds
def speed_goods_train : ℤ := 32 -- speed in km/h

-- Define the conversion factor from km/h to m/s
def kmh_to_ms (v : ℤ) : ℤ := v * 1000 / 3600

-- Define the speed of the goods train in m/s
def speed_goods_train_ms := kmh_to_ms speed_goods_train

-- Define the speed of the man's train in km/h
def speed_mans_train : ℤ := 80

-- Prove that the speed of the man's train is 80 km/h given the conditions
theorem speed_of_mans_train_is_80_kmph :
  ∃ V : ℤ,
    (V + speed_goods_train) * 1000 / 3600 = length_goods_train / time_to_pass → 
    V = speed_mans_train :=
by
  sorry

end speed_of_mans_train_is_80_kmph_l72_72007


namespace runway_show_total_time_l72_72889

-- Define the conditions
def time_per_trip : Nat := 2
def num_models : Nat := 6
def trips_bathing_suits_per_model : Nat := 2
def trips_evening_wear_per_model : Nat := 3
def trips_per_model : Nat := trips_bathing_suits_per_model + trips_evening_wear_per_model
def total_trips : Nat := trips_per_model * num_models

-- State the theorem
theorem runway_show_total_time : total_trips * time_per_trip = 60 := by
  -- fill in the proof here
  sorry

end runway_show_total_time_l72_72889


namespace card_dealing_probability_l72_72045

theorem card_dealing_probability :
  let total_cards := 52
  let first_card_probability := 3 / total_cards
  let second_card_probability := 12 / (total_cards - 1)
  let third_card_probability := 3 / (total_cards - 2)
  let fourth_card_probability := 11 / (total_cards - 3)
  let case1_probability := first_card_probability * second_card_probability * third_card_probability * fourth_card_probability
  let fifth_card_probability := 3 / total_cards
  let sixth_card_probability := 1 / (total_cards - 1)
  let seventh_card_probability := 2 / (total_cards - 2)
  let eighth_card_probability := 11 / (total_cards - 3)
  let case2_probability := fifth_card_probability * sixth_card_probability * seventh_card_probability * eighth_card_probability
  let total_probability := case1_probability + case2_probability
  in total_probability = 627 / 3248700 :=
begin
  sorry -- Proof goes here.
end

end card_dealing_probability_l72_72045


namespace people_at_table_l72_72699

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l72_72699


namespace solve_fraction_problem_l72_72913

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l72_72913


namespace interior_angle_of_regular_hexagon_l72_72770

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l72_72770


namespace sum_of_fractions_l72_72816

theorem sum_of_fractions:
  (2 / 5) + (3 / 8) + (1 / 4) = 1 + (1 / 40) :=
by
  sorry

end sum_of_fractions_l72_72816


namespace correct_operation_l72_72135

variable (a : ℝ)

theorem correct_operation : 
  (3 * a^2 + 2 * a^4 ≠ 5 * a^6) ∧
  (a^2 * a^3 ≠ a^6) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) ∧
  ((-2 * a^3)^2 = 4 * a^6) := by
  sorry

end correct_operation_l72_72135


namespace hot_dogs_served_today_l72_72427

theorem hot_dogs_served_today : 9 + 2 = 11 :=
by
  sorry

end hot_dogs_served_today_l72_72427


namespace range_of_a_l72_72347

theorem range_of_a
  (h : ∀ x : ℝ, |x - 1| + |x - 2| > Real.log (a ^ 2) / Real.log 4) :
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
sorry

end range_of_a_l72_72347


namespace add_to_fraction_l72_72926

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l72_72926


namespace percentage_profit_without_discount_l72_72271

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

end percentage_profit_without_discount_l72_72271


namespace average_price_of_returned_cans_l72_72213

theorem average_price_of_returned_cans (total_cans : ℕ) (returned_cans : ℕ) (remaining_cans : ℕ)
  (avg_price_total : ℚ) (avg_price_remaining : ℚ) :
  total_cans = 6 →
  returned_cans = 2 →
  remaining_cans = 4 →
  avg_price_total = 36.5 →
  avg_price_remaining = 30 →
  (avg_price_total * total_cans - avg_price_remaining * remaining_cans) / returned_cans = 49.5 :=
by
  intros h_total_cans h_returned_cans h_remaining_cans h_avg_price_total h_avg_price_remaining
  rw [h_total_cans, h_returned_cans, h_remaining_cans, h_avg_price_total, h_avg_price_remaining]
  sorry

end average_price_of_returned_cans_l72_72213


namespace no_integer_solutions_l72_72160

theorem no_integer_solutions (m n : ℤ) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2011) :=
by sorry

end no_integer_solutions_l72_72160


namespace total_carrots_l72_72530

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end total_carrots_l72_72530


namespace find_number_to_add_l72_72918

theorem find_number_to_add : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 ∧ n = 13 / 2 :=
by
  sorry

end find_number_to_add_l72_72918


namespace inequality_proof_l72_72861

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_condition : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/(a^2) + 1/(b^2) + 1/(c^2) + 1/(d^2)) ≥ 36 :=
by
  sorry

end inequality_proof_l72_72861


namespace harry_total_cost_in_silver_l72_72845

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end harry_total_cost_in_silver_l72_72845


namespace solution_set_of_inequality_l72_72899

theorem solution_set_of_inequality :
  { x : ℝ | x^2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l72_72899


namespace num_positive_integer_solutions_l72_72895

theorem num_positive_integer_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x ≤ n → x - 1 < Real.sqrt 5) ∧ n = 3 :=
by
  sorry

end num_positive_integer_solutions_l72_72895


namespace forest_leaves_count_correct_l72_72791

def number_of_trees : ℕ := 20
def number_of_main_branches_per_tree : ℕ := 15
def number_of_sub_branches_per_main_branch : ℕ := 25
def number_of_tertiary_branches_per_sub_branch : ℕ := 30
def number_of_leaves_per_sub_branch : ℕ := 75
def number_of_leaves_per_tertiary_branch : ℕ := 45

def total_leaves_on_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch * number_of_leaves_per_sub_branch

def total_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch

def total_leaves_on_tertiary_branches_per_tree :=
  total_sub_branches_per_tree * number_of_tertiary_branches_per_sub_branch * number_of_leaves_per_tertiary_branch

def total_leaves_per_tree :=
  total_leaves_on_sub_branches_per_tree + total_leaves_on_tertiary_branches_per_tree

def total_leaves_in_forest :=
  total_leaves_per_tree * number_of_trees

theorem forest_leaves_count_correct :
  total_leaves_in_forest = 10687500 := 
by sorry

end forest_leaves_count_correct_l72_72791


namespace smallest_real_number_l72_72602

theorem smallest_real_number :
  ∃ (x : ℝ), x = -3 ∧ (∀ (y : ℝ), y = 0 ∨ y = (-1/3)^2 ∨ y = -((27:ℝ)^(1/3)) ∨ y = -2 → x ≤ y) := 
by 
  sorry

end smallest_real_number_l72_72602


namespace count_4_digit_divisible_by_45_l72_72792

theorem count_4_digit_divisible_by_45 : 
  ∃ n, n = 11 ∧ (∀ a b : ℕ, a + b = 2 ∨ a + b = 11 → (20 + b * 10 + 5) % 45 = 0) :=
sorry

end count_4_digit_divisible_by_45_l72_72792


namespace trig_identity_l72_72968

theorem trig_identity : 
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (72 * Real.pi / 180)) * (Real.sin (84 * Real.pi / 180)) = 1 / 32 :=
by sorry

end trig_identity_l72_72968


namespace degree_measure_of_regular_hexagon_interior_angle_l72_72752

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l72_72752


namespace kibble_recommendation_difference_l72_72140

theorem kibble_recommendation_difference :
  (0.2 * 1000 : ℝ) < (0.3 * 1000) ∧ ((0.3 * 1000) - (0.2 * 1000)) = 100 :=
by
  sorry

end kibble_recommendation_difference_l72_72140


namespace solve_fraction_problem_l72_72912

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l72_72912


namespace find_a_l72_72475

theorem find_a (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 0) :
  x + n^n * (1 / (x^n)) ≥ n + 1 :=
sorry

end find_a_l72_72475


namespace units_digit_29_pow_8_pow_7_l72_72168

/-- The units digit of 29 raised to an arbitrary power follows a cyclical pattern. 
    For the purposes of this proof, we use that 29^k for even k ends in 1.
    Since 8^7 is even, we prove the units digit of 29^(8^7) is 1. -/
theorem units_digit_29_pow_8_pow_7 : (29^(8^7)) % 10 = 1 :=
by
  have even_power_cycle : ∀ k, k % 2 = 0 → (29^k) % 10 = 1 := sorry
  have eight_power_seven_even : (8^7) % 2 = 0 := by norm_num
  exact even_power_cycle (8^7) eight_power_seven_even

end units_digit_29_pow_8_pow_7_l72_72168


namespace evaluate_expression_l72_72312

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end evaluate_expression_l72_72312


namespace total_strings_needed_l72_72366

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end total_strings_needed_l72_72366


namespace direct_variation_y_value_l72_72498

theorem direct_variation_y_value (x y k : ℝ) (h1 : y = k * x) (h2 : ∀ x, x = 5 → y = 10) 
                                 (h3 : ∀ x, x < 0 → k = 4) (hx : x = -6) : y = -24 :=
sorry

end direct_variation_y_value_l72_72498


namespace jimmy_yellow_marbles_correct_l72_72524

def lorin_black_marbles : ℕ := 4
def alex_black_marbles : ℕ := 2 * lorin_black_marbles
def alex_total_marbles : ℕ := 19
def alex_yellow_marbles : ℕ := alex_total_marbles - alex_black_marbles
def jimmy_yellow_marbles : ℕ := 2 * alex_yellow_marbles

theorem jimmy_yellow_marbles_correct : jimmy_yellow_marbles = 22 := by
  sorry

end jimmy_yellow_marbles_correct_l72_72524


namespace friday_profit_l72_72587

noncomputable def total_weekly_profit : ℝ := 2000
noncomputable def profit_on_monday (total : ℝ) : ℝ := total / 3
noncomputable def profit_on_tuesday (total : ℝ) : ℝ := total / 4
noncomputable def profit_on_thursday (total : ℝ) : ℝ := 0.35 * total
noncomputable def profit_on_friday (total : ℝ) : ℝ :=
  total - (profit_on_monday total + profit_on_tuesday total + profit_on_thursday total)

theorem friday_profit (total : ℝ) : profit_on_friday total = 133.33 :=
by
  sorry

end friday_profit_l72_72587


namespace passenger_difference_l72_72875

theorem passenger_difference {x : ℕ} :
  (30 + x = 3 * x + 14) →
  6 = 3 * x - x - 16 :=
by
  sorry

end passenger_difference_l72_72875


namespace donna_pizza_slices_l72_72309

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end donna_pizza_slices_l72_72309


namespace domain_of_sqrt_fraction_l72_72033

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end domain_of_sqrt_fraction_l72_72033


namespace initial_wage_of_illiterate_l72_72656

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end initial_wage_of_illiterate_l72_72656


namespace gear_ratio_proportion_l72_72979

variables {x y z w : ℕ} {ω_A ω_B ω_C ω_D : ℝ}

theorem gear_ratio_proportion 
  (h1: x * ω_A = y * ω_B) 
  (h2: y * ω_B = z * ω_C) 
  (h3: z * ω_C = w * ω_D):
  ω_A / ω_B = y * z * w / (x * z * w) ∧ 
  ω_B / ω_C = x * z * w / (y * x * w) ∧ 
  ω_C / ω_D = x * y * w / (z * y * w) ∧ 
  ω_D / ω_A = x * y * z / (w * z * y) :=
sorry  -- Proof is not included

end gear_ratio_proportion_l72_72979


namespace sum_of_two_digit_numbers_l72_72930

/-- Given two conditions regarding multiplication mistakes, we prove the sum of the numbers. -/
theorem sum_of_two_digit_numbers
  (A B C D : ℕ)
  (h1 : (10 * A + B) * (60 + D) = 2496)
  (h2 : (10 * A + B) * (20 + D) = 936) :
  (10 * A + B) + (10 * C + D) = 63 :=
by
  -- Conditions and necessary steps for solving the problem would go here.
  -- We're focusing on stating the problem, not the solution.
  sorry

end sum_of_two_digit_numbers_l72_72930


namespace seated_people_count_l72_72688

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l72_72688


namespace seated_people_count_l72_72689

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l72_72689


namespace degree_measure_of_regular_hexagon_interior_angle_l72_72753

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l72_72753


namespace platform_length_correct_l72_72284

noncomputable def platform_length (train_speed_kmph : ℝ) (crossing_time_s : ℝ) (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * crossing_time_s
  distance_covered - train_length_m

theorem platform_length_correct :
  platform_length 72 26 260.0416 = 259.9584 :=
by
  sorry

end platform_length_correct_l72_72284


namespace find_common_ratio_l72_72065

theorem find_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h₀ : ∀ n, a (n + 1) = q * a n)
  (h₁ : a 0 = 4)
  (h₂ : q ≠ 1)
  (h₃ : 2 * a 4 = 4 * a 0 - 2 * a 2) :
  q = -1 := 
sorry

end find_common_ratio_l72_72065


namespace insurance_compensation_correct_l72_72276

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end insurance_compensation_correct_l72_72276


namespace removed_term_sequence_l72_72050

theorem removed_term_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (k : ℕ) :
  (∀ n, S n = 2 * n^2 - n) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (S 21 - a k = 40 * 20) →
  a k = 4 * k - 3 →
  k = 16 :=
by
  intros hs ha h_avg h_ak
  sorry

end removed_term_sequence_l72_72050


namespace merry_go_round_times_l72_72967

theorem merry_go_round_times
  (dave_time : ℕ := 10)
  (chuck_multiplier : ℕ := 5)
  (erica_increase : ℕ := 30) : 
  let chuck_time := chuck_multiplier * dave_time,
      erica_time := chuck_time + (erica_increase * chuck_time / 100)
  in erica_time = 65 :=
by 
  let dave_time := 10
  let chuck_multiplier := 5
  let erica_increase := 30
  let chuck_time := chuck_multiplier * dave_time
  let erica_time := chuck_time + (erica_increase * chuck_time / 100)
  exact Nat.succ 64 -- directly providing the evaluated result to match the problem statement specification

end merry_go_round_times_l72_72967


namespace number_of_books_in_box_l72_72588

theorem number_of_books_in_box :
  ∀ (total_weight : ℕ) (empty_box_weight : ℕ) (book_weight : ℕ),
  total_weight = 42 →
  empty_box_weight = 6 →
  book_weight = 3 →
  (total_weight - empty_box_weight) / book_weight = 12 :=
by
  intros total_weight empty_box_weight book_weight htwe hebe hbw
  sorry

end number_of_books_in_box_l72_72588


namespace razorback_shop_revenue_from_jerseys_zero_l72_72110

theorem razorback_shop_revenue_from_jerseys_zero:
  let num_tshirts := 20
  let num_jerseys := 64
  let revenue_per_tshirt := 215
  let total_revenue_tshirts := 4300
  let total_revenue := total_revenue_tshirts
  let revenue_from_jerseys := total_revenue - total_revenue_tshirts
  revenue_from_jerseys = 0 := by
  sorry

end razorback_shop_revenue_from_jerseys_zero_l72_72110


namespace calculate_f_f_f_one_l72_72070

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem calculate_f_f_f_one : f (f (f 1)) = 9184 :=
by
  sorry

end calculate_f_f_f_one_l72_72070


namespace roots_exist_range_k_l72_72060

theorem roots_exist_range_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, (2 * k * x1^2 + (8 * k + 1) * x1 + 8 * k = 0) ∧ 
                 (2 * k * x2^2 + (8 * k + 1) * x2 + 8 * k = 0)) ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end roots_exist_range_k_l72_72060


namespace arithmetic_sequence_sum_l72_72184

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n+1) = a n + d)
  (h1 : a 2 + a 3 = 1)
  (h2 : a 10 + a 11 = 9) :
  a 5 + a 6 = 4 :=
sorry

end arithmetic_sequence_sum_l72_72184


namespace cos6_plus_sin6_equal_19_div_64_l72_72369

noncomputable def cos6_plus_sin6 (θ : ℝ) : ℝ :=
  (Real.cos θ) ^ 6 + (Real.sin θ) ^ 6

theorem cos6_plus_sin6_equal_19_div_64 (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) :
  cos6_plus_sin6 θ = 19 / 64 := by
  sorry

end cos6_plus_sin6_equal_19_div_64_l72_72369


namespace people_at_table_l72_72698

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l72_72698


namespace count_perfect_fourth_powers_l72_72336

theorem count_perfect_fourth_powers: 
  ∃ n_count: ℕ, n_count = 4 ∧ ∀ n: ℕ, (50 ≤ n^4 ∧ n^4 ≤ 2000) → (n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) :=
by {
  sorry
}

end count_perfect_fourth_powers_l72_72336


namespace interior_angle_of_regular_hexagon_l72_72768

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l72_72768


namespace chord_slope_range_l72_72186

theorem chord_slope_range (x1 y1 x2 y2 x0 y0 : ℝ) (h1 : x1^2 + (y1^2)/4 = 1) (h2 : x2^2 + (y2^2)/4 = 1)
  (h3 : x0 = (x1 + x2) / 2) (h4 : y0 = (y1 + y2) / 2)
  (h5 : x0 = 1/2) (h6 : 1/2 ≤ y0 ∧ y0 ≤ 1) :
  -4 ≤ (-2 / y0) ∧ -2 ≤ (-2 / y0) :=
by
  sorry

end chord_slope_range_l72_72186


namespace brendas_age_l72_72150

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end brendas_age_l72_72150


namespace conditional_probability_l72_72601

variable (P : Set → ℝ) (A B : Set)

-- Conditions
axiom P_B : P B = 0.75
axiom P_AB : P (A ∩ B) = 0.6

-- Question as a proof goal
theorem conditional_probability :
  P (A ∩ B) / P B = 0.8 :=
by
  rw [P_B, P_AB]
  -- Placeholder for more steps to manipulate the expressions
  sorry

end conditional_probability_l72_72601


namespace gear_p_revolutions_per_minute_l72_72300

theorem gear_p_revolutions_per_minute (r : ℝ) 
  (cond2 : ℝ := 40) 
  (cond3 : 1.5 * r + 45 = 1.5 * 40) :
  r = 10 :=
by
  sorry

end gear_p_revolutions_per_minute_l72_72300


namespace felicity_gas_usage_l72_72460

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l72_72460


namespace arithmetic_expression_evaluation_l72_72030

theorem arithmetic_expression_evaluation : 
  -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := 
by
  sorry

end arithmetic_expression_evaluation_l72_72030


namespace modulus_of_complex_division_l72_72024

noncomputable def complexDivisionModulus : ℂ := Complex.normSq (2 * Complex.I / (Complex.I - 1))

theorem modulus_of_complex_division : complexDivisionModulus = Real.sqrt 2 := by
  sorry

end modulus_of_complex_division_l72_72024


namespace cuboid_edge_sum_l72_72736

-- Define the properties of a cuboid
structure Cuboid (α : Type) [LinearOrderedField α] where
  length : α
  width : α
  height : α

-- Define the volume of a cuboid
def volume {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  c.length * c.width * c.height

-- Define the surface area of a cuboid
def surface_area {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

-- Define the sum of all edges of a cuboid
def edge_sum {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  4 * (c.length + c.width + c.height)

-- Given a geometric progression property
def gp_property {α : Type} [LinearOrderedField α] (c : Cuboid α) (q a : α) : Prop :=
  c.length = q * a ∧ c.width = a ∧ c.height = a / q

-- The main problem to be stated in Lean
theorem cuboid_edge_sum (α : Type) [LinearOrderedField α] (c : Cuboid α) (a q : α)
  (h1 : volume c = 8)
  (h2 : surface_area c = 32)
  (h3 : gp_property c q a) :
  edge_sum c = 32 := by
    sorry

end cuboid_edge_sum_l72_72736


namespace jana_walk_distance_l72_72516

theorem jana_walk_distance :
  (1 / 20 * 15 : ℝ) = 0.8 :=
by sorry

end jana_walk_distance_l72_72516


namespace remainder_of_3042_div_98_l72_72407

theorem remainder_of_3042_div_98 : 3042 % 98 = 4 := 
by
  sorry

end remainder_of_3042_div_98_l72_72407


namespace papaya_tree_growth_ratio_l72_72949

theorem papaya_tree_growth_ratio :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
    a1 = 2 ∧
    a2 = a1 * 1.5 ∧
    a3 = a2 * 1.5 ∧
    a4 = a3 * 2 ∧
    a1 + a2 + a3 + a4 + a5 = 23 ∧
    a5 = 4.5 ∧
    (a5 / a4) = 0.5 :=
sorry

end papaya_tree_growth_ratio_l72_72949


namespace tangent_segments_area_l72_72031

theorem tangent_segments_area (r : ℝ) (l : ℝ) (area : ℝ) :
  r = 4 ∧ l = 6 → area = 9 * Real.pi :=
by
  sorry

end tangent_segments_area_l72_72031


namespace find_a_for_quadratic_max_l72_72064

theorem find_a_for_quadratic_max :
  ∃ a : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ 1/2 → (x^2 + 2 * x - 2 ≤ 1)) ∧
           (∃ x : ℝ, a ≤ x ∧ x ≤ 1/2 ∧ (x^2 + 2 * x - 2 = 1)) ∧ 
           a = -3 :=
sorry

end find_a_for_quadratic_max_l72_72064


namespace triangle_cut_20_sided_polygon_l72_72449

-- Definitions based on the conditions
def is_triangle (T : Type) : Prop := ∃ (a b c : ℝ), a + b + c = 180 

def can_form_20_sided_polygon (pieces : List (ℝ × ℝ)) : Prop := pieces.length = 20

-- Theorem statement
theorem triangle_cut_20_sided_polygon (T : Type) (P1 P2 : (ℝ × ℝ)) :
  is_triangle T → 
  (P1 ≠ P2) → 
  can_form_20_sided_polygon [P1, P2] :=
sorry

end triangle_cut_20_sided_polygon_l72_72449


namespace percent_employed_l72_72210

theorem percent_employed (E : ℝ) : 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30 -- 1 - percent_females
  (percent_males * E = employed_males) → E = 70 := 
by 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30
  intro h
  sorry

end percent_employed_l72_72210


namespace horse_revolutions_l72_72420

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end horse_revolutions_l72_72420


namespace combined_gold_cost_l72_72823

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end combined_gold_cost_l72_72823


namespace combined_length_of_straight_parts_l72_72795

noncomputable def length_of_straight_parts (R : ℝ) (p : ℝ) : ℝ := p * R

theorem combined_length_of_straight_parts :
  ∀ (R : ℝ) (p : ℝ), R = 80 ∧ p = 0.25 → length_of_straight_parts R p = 20 :=
by
  intros R p h
  cases' h with hR hp
  rw [hR, hp]
  simp [length_of_straight_parts]
  sorry

end combined_length_of_straight_parts_l72_72795


namespace log_product_eq_3_div_4_l72_72607

theorem log_product_eq_3_div_4 : (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 9) = 3 / 4 :=
by
  sorry

end log_product_eq_3_div_4_l72_72607


namespace math_problem_l72_72982

variable (f : ℝ → ℝ)

-- Conditions
axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

-- Proof goals
theorem math_problem :
  (f 0 = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f (x + 6) = f x) :=
by 
  sorry

end math_problem_l72_72982


namespace randy_blocks_left_l72_72682

theorem randy_blocks_left 
  (initial_blocks : ℕ := 78)
  (used_blocks : ℕ := 19)
  (given_blocks : ℕ := 25)
  (bought_blocks : ℕ := 36)
  (sets_from_sister : ℕ := 3)
  (blocks_per_set : ℕ := 12) :
  (initial_blocks - used_blocks - given_blocks + bought_blocks + (sets_from_sister * blocks_per_set)) / 2 = 53 := 
by
  sorry

end randy_blocks_left_l72_72682


namespace selection_ways_l72_72104

/-- There are a total of 70 ways to select 3 people from 4 teachers and 5 students,
with the condition that there must be at least one teacher and one student among the selected. -/
theorem selection_ways (teachers students : ℕ) (T : 4 = teachers) (S : 5 = students) :
  ∃ (ways : ℕ), ways = 70 := by
  sorry

end selection_ways_l72_72104


namespace coplanar_lines_l72_72225

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 5 - k * s, 3 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * t, 4 + 2 * t, 6 - 2 * t)

theorem coplanar_lines (k : ℝ) :
  (exists s t : ℝ, line1 s k = line2 t) ∨ line1 1 k = (1, -k, k) ∧ line2 1 = (2, 2, -2) → k = -1 :=
by sorry

end coplanar_lines_l72_72225


namespace parrots_false_statements_l72_72512

theorem parrots_false_statements (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 140 ∧ 
    (∀ statements : ℕ → Prop, 
      (statements 0 = false) ∧ 
      (∀ i : ℕ, 1 ≤ i → i < n → 
          (statements i = true → 
            (∃ fp : ℕ, fp < i ∧ 7 * (fp + 1) > 10 * i)))) := 
by
  sorry

end parrots_false_statements_l72_72512


namespace david_initial_money_l72_72158

theorem david_initial_money (S X : ℕ) (h1 : S - 800 = 500) (h2 : X = S + 500) : X = 1800 :=
by
  sorry

end david_initial_money_l72_72158


namespace degree_measure_of_regular_hexagon_interior_angle_l72_72754

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l72_72754


namespace average_marks_l72_72138

theorem average_marks :
  let class1_students := 26
  let class1_avg_marks := 40
  let class2_students := 50
  let class2_avg_marks := 60
  let total_students := class1_students + class2_students
  let total_marks := (class1_students * class1_avg_marks) + (class2_students * class2_avg_marks)
  (total_marks / total_students : ℝ) = 53.16 := by
sorry

end average_marks_l72_72138


namespace problem1_problem2_l72_72157

variable (x y : ℝ)

theorem problem1 :
  x^4 * x^3 * x - (x^4)^2 + (-2 * x)^3 * x^5 = -8 * x^8 :=
by sorry

theorem problem2 :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end problem1_problem2_l72_72157


namespace gcd_g_x_1155_l72_72018

def g (x : ℕ) := (4 * x + 5) * (5 * x + 3) * (6 * x + 7) * (3 * x + 11)

theorem gcd_g_x_1155 (x : ℕ) (h : x % 18711 = 0) : Nat.gcd (g x) x = 1155 := by
  sorry

end gcd_g_x_1155_l72_72018


namespace tina_days_to_use_pink_pens_tina_total_pens_l72_72402

-- Definitions based on the problem conditions.
def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def total_pink_green := pink_pens + green_pens
def yellow_pens : ℕ := total_pink_green - 5
def pink_pens_per_day := 4

-- Prove the two statements based on the definitions.
theorem tina_days_to_use_pink_pens 
  (h1 : pink_pens = 15)
  (h2 : pink_pens_per_day = 4) :
  4 = 4 :=
by sorry

theorem tina_total_pens 
  (h1 : pink_pens = 15)
  (h2 : green_pens = pink_pens - 9)
  (h3 : blue_pens = green_pens + 3)
  (h4 : yellow_pens = total_pink_green - 5) :
  pink_pens + green_pens + blue_pens + yellow_pens = 46 :=
by sorry

end tina_days_to_use_pink_pens_tina_total_pens_l72_72402


namespace find_y_l72_72072

-- Hypotheses
variable (x y : ℤ)

-- Given conditions
def condition1 : Prop := x = 4
def condition2 : Prop := x + y = 0

-- The goal is to prove y = -4 given the conditions
theorem find_y (h1 : condition1 x) (h2 : condition2 x y) : y = -4 := by
  sorry

end find_y_l72_72072


namespace fg_difference_l72_72371

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 4 * x - 1

theorem fg_difference : f (g 3) - g (f 3) = -16 := by
  sorry

end fg_difference_l72_72371


namespace pie_shop_total_earnings_l72_72145

theorem pie_shop_total_earnings :
  let price_per_slice_custard := 3
  let price_per_slice_apple := 4
  let price_per_slice_blueberry := 5
  let slices_per_whole_custard := 10
  let slices_per_whole_apple := 8
  let slices_per_whole_blueberry := 12
  let num_whole_custard_pies := 6
  let num_whole_apple_pies := 4
  let num_whole_blueberry_pies := 5
  let total_earnings :=
    (num_whole_custard_pies * slices_per_whole_custard * price_per_slice_custard) +
    (num_whole_apple_pies * slices_per_whole_apple * price_per_slice_apple) +
    (num_whole_blueberry_pies * slices_per_whole_blueberry * price_per_slice_blueberry)
  total_earnings = 608 := by
  sorry

end pie_shop_total_earnings_l72_72145


namespace sum_of_any_three_on_line_is_30_l72_72379

/-- Define the list of numbers from 1 to 19 -/
def numbers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Define the specific sequence found in the solution -/
def arrangement :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 18,
   17, 16, 15, 14, 13, 12, 11]

/-- Define the function to compute the sum of any three numbers on a straight line -/
def sum_on_line (a b c : ℕ) := a + b + c

theorem sum_of_any_three_on_line_is_30 :
  ∀ i j k : ℕ, 
  i ∈ numbers ∧ j ∈ numbers ∧ k ∈ numbers ∧ (i = 10 ∨ j = 10 ∨ k = 10) →
  sum_on_line i j k = 30 :=
by
  sorry

end sum_of_any_three_on_line_is_30_l72_72379


namespace probability_red_joker_is_1_over_54_l72_72286

-- Define the conditions as given in the problem
def total_cards : ℕ := 54
def red_joker_count : ℕ := 1

-- Define the function to calculate the probability
def probability_red_joker_top_card : ℚ := red_joker_count / total_cards

-- Problem: Prove that the probability of drawing the red joker as the top card is 1/54
theorem probability_red_joker_is_1_over_54 :
  probability_red_joker_top_card = 1 / 54 :=
by
  sorry

end probability_red_joker_is_1_over_54_l72_72286


namespace Barons_theorem_correct_l72_72804

theorem Barons_theorem_correct (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ ∃ k1 k2 : ℕ, an = k1 ^ 2 ∧ bn = k2 ^ 3 := 
sorry

end Barons_theorem_correct_l72_72804


namespace number_of_keepers_l72_72508

theorem number_of_keepers
  (h₁ : 50 * 2 = 100)
  (h₂ : 45 * 4 = 180)
  (h₃ : 8 * 4 = 32)
  (h₄ : 12 * 8 = 96)
  (h₅ : 6 * 8 = 48)
  (h₆ : 100 + 180 + 32 + 96 + 48 = 456)
  (h₇ : 50 + 45 + 8 + 12 + 6 = 121)
  (h₈ : ∀ K : ℕ, (2 * (K - 5) + 6 + 2 = 2 * K - 2))
  (h₉ : ∀ K : ℕ, 121 + K + 372 = 456 + (2 * K - 2)) :
  ∃ K : ℕ, K = 39 :=
by
  sorry

end number_of_keepers_l72_72508


namespace add_to_frac_eq_l72_72922

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l72_72922


namespace f_neg_eq_f_l72_72071

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero :
  ∃ x, f x ≠ 0

axiom functional_equation :
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_neg_eq_f (x : ℝ) : f (-x) = f x := 
sorry

end f_neg_eq_f_l72_72071


namespace ellipse_major_axis_length_l72_72721

theorem ellipse_major_axis_length : 
  ∀ (x y : ℝ), x^2 + 2 * y^2 = 2 → 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_major_axis_length_l72_72721


namespace exists_decreasing_lcm_sequence_l72_72813

theorem exists_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
sorry

end exists_decreasing_lcm_sequence_l72_72813


namespace distinct_rationals_count_l72_72977

theorem distinct_rationals_count : ∃ N : ℕ, (N = 40) ∧ ∀ k : ℚ, (|k| < 100) → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) :=
by
  sorry

end distinct_rationals_count_l72_72977


namespace complex_expression_evaluation_l72_72975

theorem complex_expression_evaluation : 
  ( (2 + Complex.i) * (3 + Complex.i) ) / (1 + Complex.i) = 5 := 
by
  sorry

end complex_expression_evaluation_l72_72975


namespace felicity_gas_usage_l72_72459

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l72_72459


namespace determinant_trig_matrix_eq_one_l72_72162

theorem determinant_trig_matrix_eq_one (α θ : ℝ) :
  Matrix.det ![
  ![Real.cos α * Real.cos θ, Real.cos α * Real.sin θ, Real.sin α],
  ![Real.sin θ, -Real.cos θ, 0],
  ![Real.sin α * Real.cos θ, Real.sin α * Real.sin θ, -Real.cos α]
  ] = 1 :=
by
  sorry

end determinant_trig_matrix_eq_one_l72_72162


namespace no_integer_solution_for_z_l72_72499

theorem no_integer_solution_for_z (z : ℤ) (h : 2 / z = 2 / (z + 1) + 2 / (z + 25)) : false :=
by
  sorry

end no_integer_solution_for_z_l72_72499


namespace original_number_l72_72942

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_list (n : ℕ) (a b c d e : ℕ) : Prop :=
  n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e

def four_digit_variant (N n : ℕ) (a b c d e : ℕ) : Prop :=
  (n = 10^3 * b + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d)

theorem original_number (N : ℕ) (a b c d e : ℕ) 
  (h1 : is_five_digit N) 
  (h2 : digit_list N a b c d e)
  (h3 : ∃ n, is_five_digit n ∧ four_digit_variant N n a b c d e ∧ N + n = 54321) :
  N = 49383 := 
sorry

end original_number_l72_72942


namespace find_c_l72_72506

theorem find_c {A B C : ℝ} (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) 
(h3 : a * Real.sin A + b * Real.sin B - c * Real.sin C = (6 * Real.sqrt 7 / 7) * a * Real.sin B * Real.sin C) :
  c = 2 :=
sorry

end find_c_l72_72506


namespace exists_a_b_l72_72608

theorem exists_a_b (r : Fin 5 → ℝ) : ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 :=
by
  sorry

end exists_a_b_l72_72608


namespace number_of_pairs_l72_72811

theorem number_of_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), (∀ (pair : ℕ × ℕ), pair ∈ pairs → 1 ≤ pair.1 ∧ pair.1 ≤ 30 ∧ 3 ≤ pair.2 ∧ pair.2 ≤ 30 ∧ (pair.1 % pair.2 = 0) ∧ (pair.1 % (pair.2 - 2) = 0)) ∧ pairs.card = 22) := by
  sorry

end number_of_pairs_l72_72811


namespace pure_acid_total_is_3_8_l72_72078

/-- Volume of Solution A in liters -/
def volume_A : ℝ := 8

/-- Concentration of Solution A (in decimals, i.e., 20% as 0.20) -/
def concentration_A : ℝ := 0.20

/-- Volume of Solution B in liters -/
def volume_B : ℝ := 5

/-- Concentration of Solution B (in decimals, i.e., 35% as 0.35) -/
def concentration_B : ℝ := 0.35

/-- Volume of Solution C in liters -/
def volume_C : ℝ := 3

/-- Concentration of Solution C (in decimals, i.e., 15% as 0.15) -/
def concentration_C : ℝ := 0.15

/-- Total amount of pure acid in the resulting mixture -/
def total_pure_acid : ℝ :=
  (volume_A * concentration_A) +
  (volume_B * concentration_B) +
  (volume_C * concentration_C)

theorem pure_acid_total_is_3_8 : total_pure_acid = 3.8 := by
  sorry

end pure_acid_total_is_3_8_l72_72078


namespace reflect_across_y_axis_l72_72856

-- Definition of the original point A
def pointA : ℝ × ℝ := (2, 3)

-- Definition of the reflected point across the y-axis
def reflectedPoint (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- The theorem stating the reflection result
theorem reflect_across_y_axis : reflectedPoint pointA = (-2, 3) :=
by
  -- Proof (skipped)
  sorry

end reflect_across_y_axis_l72_72856


namespace tangent_line_circle_midpoint_locus_l72_72843

/-- 
Let O be the circle x^2 + y^2 = 1,
M be the point (-1, -4), and
N be the point (2, 0).
-/
structure CircleTangentMidpointProblem where
  (x y : ℝ)
  (O_eq : x^2 + y^2 = 1)
  (M_eq : x = -1 ∧ y = -4)
  (N_eq : x = 2 ∧ y = 0)

/- Part (1) -/
theorem tangent_line_circle (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                            (Mx My : ℝ) : ((Mx = -1 ∧ My = -4) → 
                          
                            (x = -1 ∨ 15 * x - 8 * y - 17 = 0)) := by
  sorry

/- Part (2) -/
theorem midpoint_locus (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                       (Nx Ny : ℝ) : ((Nx = 2 ∧ Ny = 0) → 
                       
                       ((x-1)^2 + y^2 = 1 ∧ (0 ≤ x ∧ x < 1 / 2))) := by
  sorry

end tangent_line_circle_midpoint_locus_l72_72843


namespace interior_angle_of_regular_hexagon_l72_72759

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l72_72759


namespace find_n_value_l72_72471

theorem find_n_value (n : ℤ) : (5^3 - 7 = 6^2 + n) ↔ (n = 82) :=
by
  sorry

end find_n_value_l72_72471


namespace trajectory_eq_l72_72196

theorem trajectory_eq (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 6 → x^2 + y^2 + 2 * x + 2 * y - 3 = 0 → 
    ∃ p q : ℝ, p = a + 1 ∧ q = b + 1 ∧ (p * x + q * y = (a^2 + b^2 - 3)/2)) →
  a^2 + b^2 + 2 * a + 2 * b + 1 = 0 :=
by
  intros h
  sorry

end trajectory_eq_l72_72196


namespace seated_people_count_l72_72685

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l72_72685


namespace original_number_l72_72944

theorem original_number (N : ℕ) (a b c d e : ℕ)
  (hN : N = 10^4 * a + 10^3 * b + 10^2 * c + 10^1 * d + e)
  (h1 : N + (10^3 * b + 10^2 * c + 10^1 * d + e) = 54321 ∨
        N + (10^4 * a + 10^2 * c + 10^1 * d + e) = 54321 ∨
        N + (10^4 * a + 10^3 * b + 10^1 * d + e) = 54321 ∨
        N + (10^4 * a + 10^3 * b + 10^2 * c + e) = 54321 ∨
        N + (10^4 * a + 10^3 * b + 10^2 * c + 10^0 * d) = 54321) :
  N = 49383 :=
begin
  sorry
end

end original_number_l72_72944


namespace hyperbola_equation_l72_72839

open Real

-- Define the conditions in Lean
def is_hyperbola_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def is_positive (x : ℝ) : Prop := x > 0

def parabola_focus : (ℝ × ℝ) := (1, 0)

def hyperbola_vertex_eq_focus (a : ℝ) : Prop := a = parabola_focus.1

def hyperbola_eccentricity (e a c : ℝ) : Prop := e = c / a

-- Our proof statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), is_positive a ∧ is_positive b ∧
  hyperbola_vertex_eq_focus a ∧
  hyperbola_eccentricity (sqrt 5) a (sqrt 5) ∧
  is_hyperbola_form a b 1 0 :=
by sorry

end hyperbola_equation_l72_72839


namespace infinitely_many_primes_l72_72681

theorem infinitely_many_primes : ∀ (p : ℕ) (h_prime : Nat.Prime p), ∃ (q : ℕ), Nat.Prime q ∧ q > p :=
by
  sorry

end infinitely_many_primes_l72_72681


namespace graph_shift_l72_72326

theorem graph_shift (f : ℝ → ℝ) (h : f 0 = 2) : f (-1 + 1) = 2 :=
by
  have h1 : f 0 = 2 := h
  sorry

end graph_shift_l72_72326


namespace nonneg_solution_iff_m_range_l72_72991

theorem nonneg_solution_iff_m_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 1) + 3 / (1 - x) = 1)) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end nonneg_solution_iff_m_range_l72_72991


namespace inequality_abc_l72_72229

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end inequality_abc_l72_72229


namespace interior_angle_of_regular_hexagon_is_120_degrees_l72_72764

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l72_72764


namespace houses_per_block_correct_l72_72421

-- Define the conditions
def total_mail_per_block : ℕ := 32
def mail_per_house : ℕ := 8

-- Define the correct answer
def houses_per_block : ℕ := 4

-- Theorem statement
theorem houses_per_block_correct (total_mail_per_block mail_per_house : ℕ) : 
  total_mail_per_block = 32 →
  mail_per_house = 8 →
  total_mail_per_block / mail_per_house = houses_per_block :=
by
  intros h1 h2
  sorry

end houses_per_block_correct_l72_72421


namespace calculate_speed_l72_72288

variable (time : ℝ) (distance : ℝ)

theorem calculate_speed (h_time : time = 5) (h_distance : distance = 500) : 
  distance / time = 100 := 
by 
  sorry

end calculate_speed_l72_72288


namespace Dodo_is_sane_l72_72305

-- Declare the names of the characters
inductive Character
| Dodo : Character
| Lori : Character
| Eagle : Character

open Character

-- Definitions of sanity state
def sane (c : Character) : Prop := sorry
def insane (c : Character) : Prop := ¬ sane c

-- Conditions based on the problem statement
axiom Dodo_thinks_Lori_thinks_Eagle_not_sane : (sane Lori → insane Eagle)
axiom Lori_thinks_Dodo_not_sane : insane Dodo
axiom Eagle_thinks_Dodo_sane : sane Dodo

-- Theorem to prove Dodo is sane
theorem Dodo_is_sane : sane Dodo :=
by {
    sorry
}

end Dodo_is_sane_l72_72305


namespace elevator_initial_floors_down_l72_72438

theorem elevator_initial_floors_down (x : ℕ) (h1 : 9 - x + 3 + 8 = 13) : x = 7 := 
by
  -- Proof
  sorry

end elevator_initial_floors_down_l72_72438


namespace number_of_people_seated_l72_72705

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l72_72705


namespace find_v_l72_72971

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 2], ![0, 1]]

def v : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![3], ![1]]

def target : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![15], ![5]]

theorem find_v :
  let B2 := B * B
  let B3 := B2 * B
  let B4 := B3 * B
  (B4 + B3 + B2 + B + (1 : Matrix (Fin 2) (Fin 2) ℚ)) * v = target :=
by
  sorry

end find_v_l72_72971


namespace cube_inscribed_sphere_volume_l72_72640

noncomputable def cubeSurfaceArea (a : ℝ) : ℝ := 6 * a^2
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def inscribedSphereRadius (a : ℝ) : ℝ := a / 2

theorem cube_inscribed_sphere_volume :
  ∀ (a : ℝ), cubeSurfaceArea a = 24 → sphereVolume (inscribedSphereRadius a) = (4 / 3) * Real.pi := 
by 
  intros a h₁
  sorry

end cube_inscribed_sphere_volume_l72_72640


namespace sum_log_ceiling_floor_l72_72028

theorem sum_log_ceiling_floor : 
  (∑ k in Finset.range (1500 + 1), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 1124657 :=
  by
  sorry

end sum_log_ceiling_floor_l72_72028


namespace work_days_together_l72_72416

variable (d : ℝ) (j : ℝ)

theorem work_days_together (hd : d = 1 / 5) (hj : j = 1 / 9) :
  1 / (d + j) = 45 / 14 := by
  sorry

end work_days_together_l72_72416


namespace jamie_dimes_l72_72084

theorem jamie_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 240) : d = 10 :=
sorry

end jamie_dimes_l72_72084


namespace bread_cost_is_30_l72_72212

variable (cost_sandwich : ℝ)
variable (cost_ham : ℝ)
variable (cost_cheese : ℝ)

def cost_bread (cost_sandwich cost_ham cost_cheese : ℝ) : ℝ :=
  cost_sandwich - cost_ham - cost_cheese

theorem bread_cost_is_30 (H1 : cost_sandwich = 0.90)
  (H2 : cost_ham = 0.25)
  (H3 : cost_cheese = 0.35) :
  cost_bread cost_sandwich cost_ham cost_cheese = 0.30 :=
by
  rw [H1, H2, H3]
  simp [cost_bread]
  sorry

end bread_cost_is_30_l72_72212


namespace bills_are_fake_bart_can_give_exact_amount_l72_72297

-- Problem (a)
theorem bills_are_fake : 
  (∀ x, x = 17 ∨ x = 19 → false) :=
sorry

-- Problem (b)
theorem bart_can_give_exact_amount (n : ℕ) :
  (∀ m, m = 323  → (n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b)) :=
sorry

end bills_are_fake_bart_can_give_exact_amount_l72_72297


namespace triangle_angle_distance_l72_72651

noncomputable def triangle_properties (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) : Prop :=
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R

theorem triangle_angle_distance (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) :
  triangle_properties ABC P Q R angle dist →
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R :=
by intros; sorry

end triangle_angle_distance_l72_72651


namespace length_OR_coordinates_Q_area_OPQR_8_p_value_l72_72722

noncomputable def point_R : (ℝ × ℝ) := (0, 4)

noncomputable def OR_distance : ℝ := 0 - 4 -- the vertical distance from O to R

theorem length_OR : OR_distance = 4 := sorry

noncomputable def point_Q (p : ℝ) : (ℝ × ℝ) := (p, 2 * p + 4)

theorem coordinates_Q (p : ℝ) : point_Q p = (p, 2 * p + 4) := sorry

noncomputable def area_OPQR (p : ℝ) : ℝ := 
  let OR : ℝ := 4
  let PQ : ℝ := 2 * p + 4
  let OP : ℝ := p
  1 / 2 * (OR + PQ) * OP

theorem area_OPQR_8 : area_OPQR 8 = 96 := sorry

theorem p_value (h : area_OPQR p = 77) : p = 7 := sorry

end length_OR_coordinates_Q_area_OPQR_8_p_value_l72_72722


namespace total_payment_correct_l72_72453

noncomputable def calculate_total_payment : ℝ :=
  let original_price_vase := 200
  let discount_vase := 0.35 * original_price_vase
  let sale_price_vase := original_price_vase - discount_vase
  let tax_vase := 0.10 * sale_price_vase

  let original_price_teacups := 300
  let discount_teacups := 0.20 * original_price_teacups
  let sale_price_teacups := original_price_teacups - discount_teacups
  let tax_teacups := 0.08 * sale_price_teacups

  let original_price_plate := 500
  let sale_price_plate := original_price_plate
  let tax_plate := 0.10 * sale_price_plate

  (sale_price_vase + tax_vase) + (sale_price_teacups + tax_teacups) + (sale_price_plate + tax_plate)

theorem total_payment_correct : calculate_total_payment = 952.20 :=
by sorry

end total_payment_correct_l72_72453


namespace triangle_incenter_equilateral_l72_72527

theorem triangle_incenter_equilateral (a b c : ℝ) (h : (b + c) / a = (a + c) / b ∧ (a + c) / b = (a + b) / c) : a = b ∧ b = c :=
by
  sorry

end triangle_incenter_equilateral_l72_72527


namespace student_tickets_second_day_l72_72886

variable (S T x: ℕ)

theorem student_tickets_second_day (hT : T = 9) (h_eq1 : 4 * S + 3 * T = 79) (h_eq2 : 12 * S + x * T = 246) : x = 10 :=
by
  sorry

end student_tickets_second_day_l72_72886


namespace probability_not_snowing_l72_72398

theorem probability_not_snowing (P_snowing : ℚ) (h : P_snowing = 2/7) :
  (1 - P_snowing) = 5/7 :=
sorry

end probability_not_snowing_l72_72398


namespace monitor_height_l72_72737

theorem monitor_height (width circumference : ℕ) (h_width : width = 12) (h_circumference : circumference = 38) :
  2 * (width + 7) = circumference :=
by
  sorry

end monitor_height_l72_72737


namespace hyperbola_equation_l72_72490

-- Fixed points F_1 and F_2
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Condition: The absolute value of the difference in distances from P to F1 and F2 is 6
def distance_condition (P : ℝ × ℝ) : Prop :=
  abs ((dist P F1) - (dist P F2)) = 6

theorem hyperbola_equation : 
  ∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ ∀ (x y : ℝ), distance_condition (x, y) → 
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1 :=
by
  -- We state the conditions and result derived from them
  sorry

end hyperbola_equation_l72_72490


namespace max_tetrahedron_volume_l72_72354

theorem max_tetrahedron_volume 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (right_triangle : ∃ A B C : Type, 
    ∃ (angle_C : ℝ) (h_angle_C : angle_C = π / 2), 
    ∃ (BC CA : ℝ), BC = a ∧ CA = b) : 
  ∃ V : ℝ, V = (a^2 * b^2) / (6 * (a^(2/3) + b^(2/3))^(3/2)) := 
sorry

end max_tetrahedron_volume_l72_72354


namespace total_students_in_class_l72_72652

-- Definitions based on the conditions
def num_girls : ℕ := 140
def num_boys_absent : ℕ := 40
def num_boys_present := num_girls / 2
def num_boys := num_boys_present + num_boys_absent
def total_students := num_girls + num_boys

-- Theorem to be proved
theorem total_students_in_class : total_students = 250 :=
by
  sorry

end total_students_in_class_l72_72652


namespace correct_product_of_a_and_b_l72_72659

-- Define reversal function for two-digit numbers
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- State the main problem
theorem correct_product_of_a_and_b (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 0 < b) 
  (h : (reverse_digits a) * b = 284) : a * b = 68 :=
sorry

end correct_product_of_a_and_b_l72_72659


namespace prod_of_real_roots_equation_l72_72729

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end prod_of_real_roots_equation_l72_72729


namespace ratio_r_to_pq_l72_72137

theorem ratio_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 5000) (h₂ : r = 2000) :
  r / (p + q) = 2 / 3 := 
by
  sorry

end ratio_r_to_pq_l72_72137


namespace regular_hexagon_interior_angle_measure_l72_72750

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l72_72750


namespace distribution_plans_equiv_210_l72_72154

noncomputable def number_of_distribution_plans : ℕ := sorry -- we will skip the proof

theorem distribution_plans_equiv_210 :
  number_of_distribution_plans = 210 := by
  sorry

end distribution_plans_equiv_210_l72_72154


namespace sum_of_coefficients_l72_72543

theorem sum_of_coefficients (a b c d e x : ℝ) (h : 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) :
  a + b + c + d + e = 36 :=
by
  sorry

end sum_of_coefficients_l72_72543


namespace fraction_spent_on_museum_ticket_l72_72517

theorem fraction_spent_on_museum_ticket (initial_money : ℝ) (sandwich_fraction : ℝ) (book_fraction : ℝ) (remaining_money : ℝ) (h1 : initial_money = 90) (h2 : sandwich_fraction = 1/5) (h3 : book_fraction = 1/2) (h4 : remaining_money = 12) : (initial_money - remaining_money) / initial_money - (sandwich_fraction * initial_money + book_fraction * initial_money) / initial_money = 1/6 :=
by
  sorry

end fraction_spent_on_museum_ticket_l72_72517


namespace problem_statement_l72_72342

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l72_72342


namespace sum_roots_l72_72167

theorem sum_roots :
  (∀ (x : ℂ), (3 * x^3 - 2 * x^2 + 4 * x - 15 = 0) → 
              x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (∀ (x : ℂ), (4 * x^3 - 16 * x^2 - 28 * x + 35 = 0) → 
              x = y₁ ∨ x = y₂ ∨ x = y₃) →
  (x₁ + x₂ + x₃ + y₁ + y₂ + y₃ = 14 / 3) :=
by
  sorry

end sum_roots_l72_72167


namespace consecutive_negative_integers_product_sum_l72_72118

theorem consecutive_negative_integers_product_sum (n : ℤ) 
  (h_neg1 : n < 0) 
  (h_neg2 : n + 1 < 0) 
  (h_product : n * (n + 1) = 2720) :
  n + (n + 1) = -105 :=
sorry

end consecutive_negative_integers_product_sum_l72_72118


namespace angle_relation_l72_72491

theorem angle_relation (R : ℝ) (hR : R > 0) (d : ℝ) (hd : d > R) 
  (α β : ℝ) : β = 3 * α :=
sorry

end angle_relation_l72_72491


namespace chord_length_l72_72397

theorem chord_length (a b : ℝ) (M : ℝ) (h : M * M = a * b) : ∃ AB : ℝ, AB = 2 * Real.sqrt (a * b) :=
by
  sorry

end chord_length_l72_72397


namespace probability_red_or_black_probability_red_black_or_white_l72_72939

theorem probability_red_or_black (total_balls red_balls black_balls : ℕ) : 
  total_balls = 12 → red_balls = 5 → black_balls = 4 → 
  (red_balls + black_balls) / total_balls = 3 / 4 :=
by
  intros
  sorry

theorem probability_red_black_or_white (total_balls red_balls black_balls white_balls : ℕ) :
  total_balls = 12 → red_balls = 5 → black_balls = 4 → white_balls = 2 → 
  (red_balls + black_balls + white_balls) / total_balls = 11 / 12 :=
by
  intros
  sorry

end probability_red_or_black_probability_red_black_or_white_l72_72939


namespace compute_XY_l72_72216

theorem compute_XY (BC AC AB : ℝ) (hBC : BC = 30) (hAC : AC = 50) (hAB : AB = 60) :
  let XA := (BC * AB) / AC 
  let AY := (BC * AC) / AB
  let XY := XA + AY
  XY = 61 :=
by
  sorry

end compute_XY_l72_72216


namespace meat_per_deer_is_200_l72_72256

namespace wolf_pack

def number_hunting_wolves : ℕ := 4
def number_additional_wolves : ℕ := 16
def meat_needed_per_day : ℕ := 8
def days : ℕ := 5

def total_wolves : ℕ := number_hunting_wolves + number_additional_wolves

def total_meat_needed : ℕ := total_wolves * meat_needed_per_day * days

def number_deer : ℕ := number_hunting_wolves

def meat_per_deer : ℕ := total_meat_needed / number_deer

theorem meat_per_deer_is_200 : meat_per_deer = 200 := by
  sorry

end wolf_pack

end meat_per_deer_is_200_l72_72256


namespace book_purchasing_schemes_l72_72591

theorem book_purchasing_schemes :
  let investment := 500
  let cost_A := 30
  let cost_B := 25
  let cost_C := 20
  let min_books_A := 5
  let max_books_A := 6
  (Σ (a : ℕ) (b : ℕ) (c : ℕ), 
    (min_books_A ≤ a ∧ a ≤ max_books_A) ∧ 
    (cost_A * a + cost_B * b + cost_C * c = investment)) = 6 := 
by
  sorry

end book_purchasing_schemes_l72_72591


namespace linear_relationship_increase_in_y_l72_72511

theorem linear_relationship_increase_in_y (x y : ℝ) (hx : x = 12) (hy : y = 10 / 4 * x) : y = 30 := by
  sorry

end linear_relationship_increase_in_y_l72_72511


namespace percentage_neither_bp_nor_ht_l72_72430

noncomputable def percentage_teachers_neither_condition (total: ℕ) (high_bp: ℕ) (heart_trouble: ℕ) (both: ℕ) : ℚ :=
  let either_condition := high_bp + heart_trouble - both
  let neither_condition := total - either_condition
  (neither_condition * 100 : ℚ) / total

theorem percentage_neither_bp_nor_ht :
  percentage_teachers_neither_condition 150 90 50 30 = 26.67 :=
by
  sorry

end percentage_neither_bp_nor_ht_l72_72430


namespace y_equals_4_if_abs_diff_eq_l72_72408

theorem y_equals_4_if_abs_diff_eq (y : ℝ) (h : |y - 3| = |y - 5|) : y = 4 :=
sorry

end y_equals_4_if_abs_diff_eq_l72_72408


namespace find_special_N_l72_72615

theorem find_special_N : ∃ N : ℕ, 
  (Nat.digits 10 N).length = 1112 ∧
  (Nat.digits 10 N).sum % 2000 = 0 ∧
  (Nat.digits 10 (N + 1)).sum % 2000 = 0 ∧
  (Nat.digits 10 N).contains 1 ∧
  (N = 9 * 10^1111 + 1 * 10^221 + 9 * (10^220 - 1) / 9 + 10^890 - 1) :=
sorry

end find_special_N_l72_72615


namespace james_marbles_left_l72_72361

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end james_marbles_left_l72_72361


namespace solve_equation1_solve_equation2_l72_72234

open Real

theorem solve_equation1 (x : ℝ) : (x^2 - 4 * x + 3 = 0) ↔ (x = 1 ∨ x = 3) := by
  sorry

theorem solve_equation2 (x : ℝ) : (x * (x - 2) = 2 * (2 - x)) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l72_72234


namespace theo_drinks_8_cups_per_day_l72_72400

/--
Theo, Mason, and Roxy are siblings. 
Mason drinks 7 cups of water every day.
Roxy drinks 9 cups of water every day. 
In one week, the siblings drink 168 cups of water together. 

Prove that Theo drinks 8 cups of water every day.
-/
theorem theo_drinks_8_cups_per_day (T : ℕ) :
  (∀ (d m r : ℕ), 
    (m = 7 ∧ r = 9 ∧ d + m + r = 168) → 
    (T * 7 = d) → T = 8) :=
by
  intros d m r cond1 cond2
  have h1 : d + 49 + 63 = 168 := by sorry
  have h2 : T * 7 = d := cond2
  have goal : T = 8 := by sorry
  exact goal

end theo_drinks_8_cups_per_day_l72_72400


namespace correct_statement_l72_72413

def is_accurate_to (value : ℝ) (place : ℝ) : Prop :=
  ∃ k : ℤ, value = k * place

def statement_A : Prop := is_accurate_to 51000 0.1
def statement_B : Prop := is_accurate_to 0.02 1
def statement_C : Prop := (2.8 = 2.80)
def statement_D : Prop := is_accurate_to (2.3 * 10^4) 1000

theorem correct_statement : statement_D :=
by
  sorry

end correct_statement_l72_72413


namespace prime_div_p_sq_minus_one_l72_72500

theorem prime_div_p_sq_minus_one {p : ℕ} (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 
  (p % 10 = 1 ∨ p % 10 = 9) → 40 ∣ (p^2 - 1) :=
sorry

end prime_div_p_sq_minus_one_l72_72500


namespace negation_of_existence_l72_72724

theorem negation_of_existence (h : ¬ (∃ x : ℝ, x^2 - x - 1 > 0)) : ∀ x : ℝ, x^2 - x - 1 ≤ 0 :=
sorry

end negation_of_existence_l72_72724


namespace recurring_decimal_to_fraction_l72_72998

theorem recurring_decimal_to_fraction (a b : ℕ) (ha : a = 356) (hb : b = 999) (hab_gcd : Nat.gcd a b = 1)
  (x : ℚ) (hx : x = 356 / 999) 
  (hx_recurring : x = {num := 356, den := 999}): a + b = 1355 :=
by
  sorry  -- Proof is not required as per the instructions

end recurring_decimal_to_fraction_l72_72998


namespace tangerine_initial_count_l72_72311

theorem tangerine_initial_count 
  (X : ℕ) 
  (h1 : X - 9 + 5 = 20) : 
  X = 24 :=
sorry

end tangerine_initial_count_l72_72311


namespace find_constants_l72_72605

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (period : 3 * Real.pi = 2 * Real.pi / b)
variable (low_peak : a = 3)

theorem find_constants (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (period : 3 * Real.pi = 2 * Real.pi / b) (low_peak : a = 3) :
  a = 3 ∧ b = 2 / 3 :=
by
  split
  { exact low_peak }
  {
    rw [←period, ←eq_div_iff (by norm_num [Real.pi_ne_zero])]
    norm_num
  }
  sorry

end find_constants_l72_72605


namespace regular_hexagon_interior_angle_measure_l72_72748

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l72_72748


namespace mod_remainder_l72_72410

theorem mod_remainder (n : ℤ) (h : n % 5 = 3) : (4 * n - 5) % 5 = 2 := by
  sorry

end mod_remainder_l72_72410


namespace simplify_expression_l72_72194

theorem simplify_expression (k : ℤ) (c d : ℤ) 
(h1 : (5 * k + 15) / 5 = c * k + d) 
(h2 : ∀ k, d + c * k = k + 3) : 
c / d = 1 / 3 := 
by 
  sorry

end simplify_expression_l72_72194


namespace probability_of_drawing_red_or_green_l72_72774

def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def total_marbles : ℕ := red_marbles + green_marbles + yellow_marbles
def favorable_marbles : ℕ := red_marbles + green_marbles
def probability_of_red_or_green : ℚ := favorable_marbles / total_marbles

theorem probability_of_drawing_red_or_green :
  probability_of_red_or_green = 7 / 13 := by
  sorry

end probability_of_drawing_red_or_green_l72_72774


namespace lemon_loaf_each_piece_weight_l72_72006

def pan_length := 20  -- cm
def pan_width := 18   -- cm
def pan_height := 5   -- cm
def total_pieces := 25
def density := 2      -- g/cm³

noncomputable def weight_of_each_piece : ℕ := by
  have volume := pan_length * pan_width * pan_height
  have volume_of_each_piece := volume / total_pieces
  have mass_of_each_piece := volume_of_each_piece * density
  exact mass_of_each_piece

theorem lemon_loaf_each_piece_weight :
  weight_of_each_piece = 144 :=
sorry

end lemon_loaf_each_piece_weight_l72_72006


namespace no_prime_satisfies_condition_l72_72867

theorem no_prime_satisfies_condition :
  ¬ ∃ p : ℕ, p > 1 ∧ 10 * (p : ℝ) = (p : ℝ) + 5.4 := by {
  sorry
}

end no_prime_satisfies_condition_l72_72867


namespace average_employees_per_week_l72_72004

-- Define the number of employees hired each week
variables (x : ℕ)
noncomputable def employees_first_week := x + 200
noncomputable def employees_second_week := x
noncomputable def employees_third_week := x + 150
noncomputable def employees_fourth_week := 400

-- Given conditions as hypotheses
axiom h1 : employees_third_week / 2 = employees_fourth_week / 2
axiom h2 : employees_fourth_week = 400

-- Prove the average number of employees hired per week is 225
theorem average_employees_per_week :
  (employees_first_week + employees_second_week + employees_third_week + employees_fourth_week) / 4 = 225 :=
by
  sorry

end average_employees_per_week_l72_72004


namespace intersection_is_as_expected_l72_72643

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

noncomputable def logarithmic_condition : Set ℝ :=
  { x | x > 0 ∧ x ≠ 1 }

noncomputable def intersection_of_sets : Set ℝ :=
  (quadratic_inequality_solution ∩ logarithmic_condition)

theorem intersection_is_as_expected :
  intersection_of_sets = { x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end intersection_is_as_expected_l72_72643


namespace denver_wood_used_per_birdhouse_l72_72304

-- Definitions used in the problem
def cost_per_piee_of_wood : ℝ := 1.50
def profit_per_birdhouse : ℝ := 5.50
def price_for_two_birdhouses : ℝ := 32
def num_birdhouses_purchased : ℝ := 2

-- Property to prove
theorem denver_wood_used_per_birdhouse (W : ℝ) 
  (h : num_birdhouses_purchased * (cost_per_piee_of_wood * W + profit_per_birdhouse) = price_for_two_birdhouses) : 
  W = 7 :=
sorry

end denver_wood_used_per_birdhouse_l72_72304


namespace monotonic_decreasing_interval_l72_72544

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem monotonic_decreasing_interval : 
  ∀ x, -1 < x ∧ x < 3 →  (deriv function_y x < 0) :=
by
  sorry

end monotonic_decreasing_interval_l72_72544


namespace parabola_properties_l72_72902

-- Given conditions
variables (a b c : ℝ)
variable (h_vertex : ∃ a b c : ℝ, (∀ x, a * (x+1)^2 + 4 = ax^2 + b * x + c))
variable (h_intersection : ∃ A : ℝ, 2 < A ∧ A < 3 ∧ a * A^2 + b * A + c = 0)

-- Define the proof problem
theorem parabola_properties (h_vertex : (b = 2 * a)) (h_a : a < 0) (h_c : c = 4 + a) : 
  ∃ x : ℕ, x = 2 ∧ 
  (∀ a b c : ℝ, a * b * c < 0 → false) ∧ 
  (-4 < a ∧ a < -1 → false) ∧
  (a * c + 2 * b > 1 → false) :=
sorry

end parabola_properties_l72_72902


namespace find_angle_C_l72_72650

open Real

theorem find_angle_C (a b C A B : ℝ) 
  (h1 : a^2 + b^2 = 6 * a * b * cos C)
  (h2 : sin C ^ 2 = 2 * sin A * sin B) :
  C = π / 3 := 
  sorry

end find_angle_C_l72_72650


namespace john_total_strings_l72_72367

theorem john_total_strings :
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := 
by
  let basses := 3
  let strings_per_bass := 4
  let guitars := 2 * basses
  let strings_per_guitar := 6
  let eight_string_guitars := guitars - 3
  let strings_per_eight_string_guitar := 8
  have H_bass := basses * strings_per_bass
  have H_guitar := guitars * strings_per_guitar
  have H_eight_string_guitar := eight_string_guitars * strings_per_eight_string_guitar
  have H_total := H_bass + H_guitar + H_eight_string_guitar
  show H_total = 72 from sorry

end john_total_strings_l72_72367


namespace max_table_rows_l72_72818

open Function

-- Definitions
def is_permutation {α : Type*} [DecidableEq α] (s : Finset α) (f : Fin (s.card) → α) : Prop :=
  ∀ {a}, a ∈ s ↔ ∃ i, f i = a

def table_conditions (n : ℕ) (table : Fin n → Fin 9 → ℕ) : Prop :=
  ∀ row : Fin n,
    (Finset.univ.image (table row)).card = 9 ∧
    (∀ row2 : Fin n, row ≠ row2 → ∃ col, table row col = table row2 col)

-- Main theorem
theorem max_table_rows : ∃ n, table_conditions n (λ i j, ...) ∧ n = 8! := sorry

end max_table_rows_l72_72818


namespace shooter_random_event_l72_72780

def eventA := "The sun rises from the east"
def eventB := "A coin thrown up from the ground will fall down"
def eventC := "A shooter hits the target with 10 points in one shot"
def eventD := "Xiao Ming runs at a speed of 30 meters per second"

def is_random_event (event : String) := event = eventC

theorem shooter_random_event : is_random_event eventC := 
by
  sorry

end shooter_random_event_l72_72780


namespace find_certain_number_l72_72578

-- Define the conditions as constants
def n1 : ℕ := 9
def n2 : ℕ := 70
def n3 : ℕ := 25
def n4 : ℕ := 21
def smallest_given_number : ℕ := 3153
def certain_number : ℕ := 3147

-- Lean theorem statement
theorem find_certain_number (n1 n2 n3 n4 smallest_given_number certain_number: ℕ) :
  (∀ x, (∀ y ∈ [n1, n2, n3, n4], y ∣ x) → x ≥ smallest_given_number → x = smallest_given_number + certain_number) :=
sorry -- Skips the proof

end find_certain_number_l72_72578


namespace find_a2015_l72_72983

variable (a : ℕ → ℝ)

-- Conditions
axiom h1 : a 1 = 1
axiom h2 : a 2 = 3
axiom h3 : ∀ n : ℕ, n > 0 → a (n + 1) - a n ≤ 2 ^ n
axiom h4 : ∀ n : ℕ, n > 0 → a (n + 2) - a n ≥ 3 * 2 ^ n

-- Theorem stating the solution
theorem find_a2015 : a 2015 = 2 ^ 2015 - 1 :=
by sorry

end find_a2015_l72_72983


namespace rectangle_area_in_inscribed_triangle_l72_72013

theorem rectangle_area_in_inscribed_triangle (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  ∃ (y : ℝ), y = (b * (h - x)) / h ∧ (x * y) = (b * x * (h - x)) / h :=
by
  sorry

end rectangle_area_in_inscribed_triangle_l72_72013


namespace mixed_number_calculation_l72_72022

theorem mixed_number_calculation :
  47 * (4 + 3/7 - (5 + 1/3)) / (3 + 1/2 + (2 + 1/5)) = -7 - 119/171 := by
  sorry

end mixed_number_calculation_l72_72022


namespace rectangle_length_l72_72394

theorem rectangle_length : 
  ∃ l b : ℝ, 
    (l = 2 * b) ∧ 
    (20 < l ∧ l < 50) ∧ 
    (10 < b ∧ b < 30) ∧ 
    ((l - 5) * (b + 5) = l * b + 75) ∧ 
    (l = 40) :=
sorry

end rectangle_length_l72_72394


namespace interest_rate_calculation_l72_72429

theorem interest_rate_calculation (P : ℝ) (r : ℝ) (h1 : P * (1 + r / 100)^3 = 800) (h2 : P * (1 + r / 100)^4 = 820) :
  r = 2.5 := 
  sorry

end interest_rate_calculation_l72_72429


namespace rainfall_difference_l72_72567

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l72_72567


namespace hyperbola_equation_sum_of_slopes_l72_72992

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 3

theorem hyperbola_equation :
  ∀ (a b : ℝ) (H1 : a > 0) (H2 : b > 0) (H3 : (2^2) = a^2 + b^2)
    (H4 : ∀ (x₀ y₀ : ℝ), (x₀ ≠ -a) ∧ (x₀ ≠ a) → (y₀^2 = (b^2 / a^2) * (x₀^2 - a^2)) ∧ ((y₀ / (x₀ + a) * y₀ / (x₀ - a)) = 3)),
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 - y^2 / 3 = 1)) :=
by
  intros a b H1 H2 H3 H4 x y Hxy
  sorry

theorem sum_of_slopes (m n : ℝ) (H1 : m < 1) :
  ∀ (k1 k2 : ℝ) (H2 : A ≠ B) (H3 : ((k1 ≠ k2) ∧ (1 + k1^2) / (3 - k1^2) = (1 + k2^2) / (3 - k2^2))),
  k1 + k2 = 0 :=
by
  intros k1 k2 H2 H3
  exact sorry

end hyperbola_equation_sum_of_slopes_l72_72992


namespace quadratic_roots_property_l72_72496

theorem quadratic_roots_property (a b : ℝ)
  (h1 : a^2 - 2 * a - 1 = 0)
  (h2 : b^2 - 2 * b - 1 = 0)
  (ha_b_sum : a + b = 2)
  (ha_b_product : a * b = -1) :
  a^2 + 2 * b - a * b = 6 :=
sorry

end quadratic_roots_property_l72_72496


namespace distance_between_P1_and_P2_l72_72442

-- Define the two points
def P1 : ℝ × ℝ := (2, 3)
def P2 : ℝ × ℝ := (5, 10)

-- Define the distance function
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Define the theorem we want to prove
theorem distance_between_P1_and_P2 :
  distance P1 P2 = Real.sqrt 58 :=
by sorry

end distance_between_P1_and_P2_l72_72442


namespace agent_takes_19_percent_l72_72887

def agentPercentage (copies_sold : ℕ) (advance_copies : ℕ) (price_per_copy : ℕ) (steve_earnings : ℕ) : ℕ :=
  let total_earnings := copies_sold * price_per_copy
  let agent_earnings := total_earnings - steve_earnings
  let percentage_agent := 100 * agent_earnings / total_earnings
  percentage_agent

theorem agent_takes_19_percent :
  agentPercentage 1000000 100000 2 1620000 = 19 :=
by 
  sorry

end agent_takes_19_percent_l72_72887


namespace income_ratio_l72_72897

-- Define the conditions
variables (I_A I_B E_A E_B : ℝ)
variables (Savings_A Savings_B : ℝ)

-- Given conditions
def expenditure_ratio : E_A / E_B = 3 / 2 := sorry
def savings_A : Savings_A = 1600 := sorry
def savings_B : Savings_B = 1600 := sorry
def income_A : I_A = 4000 := sorry
def expenditure_A : E_A = I_A - Savings_A := sorry
def expenditure_B : E_B = I_B - Savings_B := sorry

-- Prove it's implied that the ratio of incomes is 5:4
theorem income_ratio : I_A / I_B = 5 / 4 :=
by
  sorry

end income_ratio_l72_72897


namespace exists_five_consecutive_divisible_by_2014_l72_72166

theorem exists_five_consecutive_divisible_by_2014 :
  ∃ (a b c d e : ℕ), 53 = a ∧ 54 = b ∧ 55 = c ∧ 56 = d ∧ 57 = e ∧ 100 > a ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ 2014 ∣ (a * b * c * d * e) :=
by 
  sorry

end exists_five_consecutive_divisible_by_2014_l72_72166


namespace yard_length_l72_72206

theorem yard_length :
  let num_trees := 11
  let distance_between_trees := 18
  (num_trees - 1) * distance_between_trees = 180 :=
by
  let num_trees := 11
  let distance_between_trees := 18
  sorry

end yard_length_l72_72206


namespace unit_circle_root_l72_72638

noncomputable def z_eq (n : ℕ) (a : ℝ) (z : ℂ) : Prop := z^(n+1) - a * z^n + a * z - 1 = 0

theorem unit_circle_root (n : ℕ) (a : ℝ) (z : ℂ)
  (h1 : 2 ≤ n)
  (h2 : 0 < a)
  (h3 : a < (n+1) / (n-1))
  (h4 : z_eq n a z) :
  ∥z∥ = 1 := sorry

end unit_circle_root_l72_72638


namespace isosceles_triangle_perimeter_l72_72726

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_eq_triangle : a + b + c = 60) (h_eq_sides : a = b) 
  (isosceles_base : c = 15) (isosceles_side1_eq : a = 20) : a + b + c = 55 :=
by
  sorry

end isosceles_triangle_perimeter_l72_72726


namespace car_speed_15_seconds_less_l72_72786

theorem car_speed_15_seconds_less (v : ℝ) : 
  (∀ v, 75 = 3600 / v + 15) → v = 60 :=
by
  intro H
  -- Proof goes here
  sorry

end car_speed_15_seconds_less_l72_72786


namespace batsman_average_after_17th_inning_l72_72268

theorem batsman_average_after_17th_inning
  (A : ℕ)
  (h1 : (16 * A + 88) / 17 = A + 3) :
  37 + 3 = 40 :=
by sorry

end batsman_average_after_17th_inning_l72_72268


namespace remainder_of_k_divided_by_7_l72_72139

theorem remainder_of_k_divided_by_7 :
  ∃ k < 42, k % 5 = 2 ∧ k % 6 = 5 ∧ k % 7 = 3 :=
by {
  -- The proof is supplied here
  sorry
}

end remainder_of_k_divided_by_7_l72_72139


namespace carter_baseball_cards_l72_72585

theorem carter_baseball_cards (M C : ℕ) (h1 : M = 210) (h2 : M = C + 58) : C = 152 :=
by
  sorry

end carter_baseball_cards_l72_72585


namespace commute_days_l72_72422

-- Definitions of the variables
variables (a b c x : ℕ)

-- Given conditions
def condition1 : Prop := a + c = 12
def condition2 : Prop := b + c = 20
def condition3 : Prop := a + b = 14

-- The theorem to prove
theorem commute_days (h1 : condition1 a c) (h2 : condition2 b c) (h3 : condition3 a b) : a + b + c = 23 :=
sorry

end commute_days_l72_72422


namespace truck_travel_distance_l72_72014

noncomputable def truck_distance (gallons: ℕ) : ℕ :=
  let efficiency_10_gallons := 300 / 10 -- miles per gallon
  let efficiency_initial := efficiency_10_gallons
  let efficiency_decreased := efficiency_initial * 9 / 10 -- 10% decrease
  if gallons <= 12 then
    gallons * efficiency_initial
  else
    12 * efficiency_initial + (gallons - 12) * efficiency_decreased

theorem truck_travel_distance (gallons: ℕ) :
  gallons = 15 → truck_distance gallons = 441 :=
by
  intros h
  rw [h]
  -- skipping proof
  sorry

end truck_travel_distance_l72_72014


namespace number_of_players_l72_72515

-- Definitions based on conditions in the problem
def cost_of_gloves : ℕ := 6
def cost_of_helmet : ℕ := cost_of_gloves + 7
def cost_of_cap : ℕ := 3
def total_expenditure : ℕ := 2968

-- Total cost for one player
def cost_per_player : ℕ := 2 * (cost_of_gloves + cost_of_helmet) + cost_of_cap

-- Statement to prove: number of players
theorem number_of_players : total_expenditure / cost_per_player = 72 := 
by
  sorry

end number_of_players_l72_72515


namespace students_in_class_l72_72545

theorem students_in_class (n m f r u : ℕ) (cond1 : 20 < n ∧ n < 30)
  (cond2 : f = 2 * m) (cond3 : n = m + f)
  (cond4 : r = 3 * u - 1) (cond5 : r + u = n) :
  n = 27 :=
sorry

end students_in_class_l72_72545


namespace part1_part2_l72_72330

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (x : ℝ) : f x 1 >= f x 1 := sorry

theorem part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) : b / a ≥ 0 := sorry

end part1_part2_l72_72330


namespace problem_l72_72604

noncomputable def a : ℝ := Real.log 8 / Real.log 3
noncomputable def b : ℝ := Real.log 25 / Real.log 4
noncomputable def c : ℝ := Real.log 24 / Real.log 4

theorem problem : a < c ∧ c < b :=
by
  sorry

end problem_l72_72604


namespace max_small_boxes_l72_72136

-- Define the dimensions of the larger box in meters
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5
def large_box_height : ℝ := 4

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.60
def small_box_width : ℝ := 0.50
def small_box_height : ℝ := 0.40

-- Calculate the volume of the larger box
def large_box_volume : ℝ := large_box_length * large_box_width * large_box_height

-- Calculate the volume of the smaller box
def small_box_volume : ℝ := small_box_length * small_box_width * small_box_height

-- State the theorem to prove the maximum number of smaller boxes that can fit in the larger box
theorem max_small_boxes : large_box_volume / small_box_volume = 1000 :=
by
  sorry

end max_small_boxes_l72_72136


namespace find_a_l72_72989

theorem find_a (a : ℝ) : 
  let term_coeff (r : ℕ) := (Nat.choose 10 r : ℝ)
  let coeff_x6 := term_coeff 3 - (a * term_coeff 2)
  coeff_x6 = 30 → a = 2 :=
by
  intro h
  sorry

end find_a_l72_72989


namespace erica_duration_is_correct_l72_72964

-- Define the durations for Dave, Chuck, and Erica
def dave_duration : ℝ := 10
def chuck_duration : ℝ := 5 * dave_duration
def erica_duration : ℝ := chuck_duration + 0.30 * chuck_duration

-- State the theorem
theorem erica_duration_is_correct : erica_duration = 65 := by
  sorry

end erica_duration_is_correct_l72_72964


namespace chance_Z_winning_l72_72204

-- Given conditions as Lean definitions
def p_x : ℚ := 1 / (3 + 1)
def p_y : ℚ := 3 / (2 + 3)
def p_z : ℚ := 1 - (p_x + p_y)

-- Theorem statement: Prove the equivalence of the winning ratio for Z
theorem chance_Z_winning : 
  p_z = 3 / (3 + 17) :=
by
  -- Since we include no proof, we use sorry to indicate it
  sorry

end chance_Z_winning_l72_72204


namespace problem_statement_l72_72345

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end problem_statement_l72_72345


namespace edge_length_of_cube_l72_72406

theorem edge_length_of_cube {V_cube V_cuboid : ℝ} (base_area : ℝ) (height : ℝ)
  (h1 : base_area = 10) (h2 : height = 73) (h3 : V_cube = V_cuboid - 1)
  (h4 : V_cuboid = base_area * height) :
  ∃ (a : ℝ), a^3 = V_cube ∧ a = 9 :=
by
  /- The proof is omitted -/
  sorry

end edge_length_of_cube_l72_72406


namespace trig_identity_l72_72825

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  (1 / (Real.cos α ^ 2 + Real.sin (2 * α))) = 10 / 3 := 
by 
  sorry

end trig_identity_l72_72825


namespace mixture_ratio_l72_72346

theorem mixture_ratio (V : ℝ) (a b c : ℕ)
  (h_pos : V > 0)
  (h_ratio : V = (3/8) * V + (5/11) * V + ((88 - 33 - 40)/88) * V) :
  a = 33 ∧ b = 40 ∧ c = 15 :=
by
  sorry

end mixture_ratio_l72_72346


namespace arithmetic_sequence_terms_count_l72_72192

theorem arithmetic_sequence_terms_count (a d l : Int) (h1 : a = 20) (h2 : d = -3) (h3 : l = -5) :
  ∃ n : Int, l = a + (n - 1) * d ∧ n = 8 :=
by
  sorry

end arithmetic_sequence_terms_count_l72_72192


namespace find_n_l72_72586

theorem find_n (n : ℕ) (h : (1 + n + (n * (n - 1)) / 2) / 2^n = 7 / 32) : n = 6 :=
sorry

end find_n_l72_72586


namespace basketball_weight_l72_72883

theorem basketball_weight (b k : ℝ) (h1 : 6 * b = 4 * k) (h2 : 3 * k = 72) : b = 16 :=
by
  sorry

end basketball_weight_l72_72883


namespace original_length_wire_l72_72423

-- Define the conditions.
def length_cut_off_parts : ℕ := 10
def remaining_length_relation (L_remaining : ℕ) : Prop :=
  L_remaining = 4 * (2 * length_cut_off_parts) + 10

-- Define the theorem to prove the original length of the wire.
theorem original_length_wire (L_remaining : ℕ) (H : remaining_length_relation L_remaining) : 
  L_remaining + 2 * length_cut_off_parts = 110 :=
by 
  -- Use the given conditions
  unfold remaining_length_relation at H
  -- The proof would show that the equation holds true.
  sorry

end original_length_wire_l72_72423


namespace horner_eval_v4_at_2_l72_72960

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

theorem horner_eval_v4_at_2 : 
  let x := 2
  let v_0 := 1
  let v_1 := (v_0 * x) - 12 
  let v_2 := (v_1 * x) + 60 
  let v_3 := (v_2 * x) - 160 
  let v_4 := (v_3 * x) + 240 
  v_4 = 80 := 
by 
  sorry

end horner_eval_v4_at_2_l72_72960


namespace add_to_fraction_l72_72925

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l72_72925


namespace probability_of_odd_sum_given_even_product_l72_72627

open Nat

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_outcomes := total_outcomes - odd_outcomes
  let favorable_outcomes := 15 * 3^5
  favorable_outcomes / even_outcomes

theorem probability_of_odd_sum_given_even_product :
  probability_odd_sum_given_even_product = 91 / 324 :=
by
  sorry

end probability_of_odd_sum_given_even_product_l72_72627


namespace ladybugs_without_spots_l72_72885

-- Defining the conditions given in the problem
def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170

-- Proving the number of ladybugs without spots
theorem ladybugs_without_spots : total_ladybugs - ladybugs_with_spots = 54912 := by
  sorry

end ladybugs_without_spots_l72_72885


namespace value_of_72_a_in_terms_of_m_and_n_l72_72473

theorem value_of_72_a_in_terms_of_m_and_n (a m n : ℝ) (hm : 2^a = m) (hn : 3^a = n) :
  72^a = m^3 * n^2 :=
by sorry

end value_of_72_a_in_terms_of_m_and_n_l72_72473


namespace find_x_l72_72042

-- Conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 8 * x
def area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x / 2

-- Theorem to prove
theorem find_x (x s : ℝ) (h1 : volume_condition x s) (h2 : area_condition x s) : x = 110592 := sorry

end find_x_l72_72042


namespace number_of_family_members_l72_72274

-- Define the number of legs for each type of animal.
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def cat_legs : ℕ := 4

-- Define the number of animals.
def birds : ℕ := 4
def dogs : ℕ := 3
def cats : ℕ := 18

-- Define the total number of legs of all animals.
def total_animal_feet : ℕ := birds * bird_legs + dogs * dog_legs + cats * cat_legs

-- Define the total number of heads of all animals.
def total_animal_heads : ℕ := birds + dogs + cats

-- Main theorem: If the total number of feet in the house is 74 more than the total number of heads, find the number of family members.
theorem number_of_family_members (F : ℕ) (h : total_animal_feet + 2 * F = total_animal_heads + F + 74) : F = 7 :=
by
  sorry

end number_of_family_members_l72_72274


namespace order_of_real_numbers_l72_72622

noncomputable def a : ℝ := Real.arcsin (3 / 4)
noncomputable def b : ℝ := Real.arccos (1 / 5)
noncomputable def c : ℝ := 1 + Real.arctan (2 / 3)

theorem order_of_real_numbers : a < b ∧ b < c :=
by sorry

end order_of_real_numbers_l72_72622


namespace total_area_of_room_l72_72291

theorem total_area_of_room : 
  let length_rect := 8 
  let width_rect := 6 
  let base_triangle := 6 
  let height_triangle := 3 
  let area_rect := length_rect * width_rect 
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle 
  let total_area := area_rect + area_triangle 
  total_area = 57 := 
by 
  sorry

end total_area_of_room_l72_72291


namespace find_m_of_slope_is_12_l72_72246

theorem find_m_of_slope_is_12 (m : ℝ) :
  let A := (-m, 6)
  let B := (1, 3 * m)
  let slope := (3 * m - 6) / (1 + m)
  slope = 12 → m = -2 :=
by
  sorry

end find_m_of_slope_is_12_l72_72246


namespace derivative_at_0_l72_72238

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- State the theorem
theorem derivative_at_0 : f' 0 = 4 :=
by {
  -- Inserting sorry to skip the proof
  sorry
}

end derivative_at_0_l72_72238


namespace crude_oil_mixture_l72_72289

theorem crude_oil_mixture (x y : ℝ) 
  (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 0.55 * 50) : 
  y = 30 :=
by
  sorry

end crude_oil_mixture_l72_72289


namespace max_integer_value_of_x_l72_72557

theorem max_integer_value_of_x (x : ℤ) : 3 * x - (1 / 4 : ℚ) ≤ (1 / 3 : ℚ) * x - 2 → x ≤ -1 :=
by
  intro h
  sorry

end max_integer_value_of_x_l72_72557


namespace average_apples_per_guest_l72_72663

theorem average_apples_per_guest
  (servings_per_pie : ℕ)
  (pies : ℕ)
  (apples_per_serving : ℚ)
  (total_guests : ℕ)
  (red_delicious_proportion : ℚ)
  (granny_smith_proportion : ℚ)
  (total_servings := pies * servings_per_pie)
  (total_apples := total_servings * apples_per_serving)
  (total_red_delicious := (red_delicious_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (total_granny_smith := (granny_smith_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (average_apples_per_guest := total_apples / total_guests) :
  servings_per_pie = 8 →
  pies = 3 →
  apples_per_serving = 1.5 →
  total_guests = 12 →
  red_delicious_proportion = 2 →
  granny_smith_proportion = 1 →
  average_apples_per_guest = 3 :=
by
  intros;
  sorry

end average_apples_per_guest_l72_72663


namespace converse_of_posImpPosSquare_l72_72115

-- Let's define the condition proposition first
def posImpPosSquare (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Now, we state the converse we need to prove
theorem converse_of_posImpPosSquare (x : ℝ) (h : posImpPosSquare x) : x^2 > 0 → x > 0 := sorry

end converse_of_posImpPosSquare_l72_72115


namespace gcd_lcm_of_a_b_l72_72972

def a := 1560
def b := 1040

theorem gcd_lcm_of_a_b :
  (Nat.gcd a b = 520) ∧ (Nat.lcm a b = 1560) :=
by
  -- Proof is omitted.
  sorry

end gcd_lcm_of_a_b_l72_72972


namespace cyclists_meet_time_l72_72259

theorem cyclists_meet_time 
  (v1 v2 : ℕ) (C : ℕ) (h1 : v1 = 7) (h2 : v2 = 8) (hC : C = 675) : 
  C / (v1 + v2) = 45 :=
by
  sorry

end cyclists_meet_time_l72_72259


namespace range_of_a_l72_72337

theorem range_of_a (a : ℝ) (h : Real.sqrt ((2 * a - 1)^2) = 1 - 2 * a) : a ≤ 1 / 2 :=
sorry

end range_of_a_l72_72337


namespace total_surface_area_first_rectangular_parallelepiped_equals_22_l72_72301

theorem total_surface_area_first_rectangular_parallelepiped_equals_22
  (x y z : ℝ)
  (h1 : (x + 1) * (y + 1) * (z + 1) = x * y * z + 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) = 2 * (x * y + x * z + y * z) + 30) :
  2 * (x * y + x * z + y * z) = 22 := sorry

end total_surface_area_first_rectangular_parallelepiped_equals_22_l72_72301


namespace question1_question2_l72_72903

-- Define the conditions
def numTraditionalChinesePaintings : Nat := 5
def numOilPaintings : Nat := 2
def numWatercolorPaintings : Nat := 7

-- Define the number of ways to choose one painting from each category
def numWaysToChooseOnePaintingFromEachCategory : Nat :=
  numTraditionalChinesePaintings * numOilPaintings * numWatercolorPaintings

-- Define the number of ways to choose two paintings of different types
def numWaysToChooseTwoPaintingsOfDifferentTypes : Nat :=
  (numTraditionalChinesePaintings * numOilPaintings) +
  (numTraditionalChinesePaintings * numWatercolorPaintings) +
  (numOilPaintings * numWatercolorPaintings)

-- Theorems to prove the required results
theorem question1 : numWaysToChooseOnePaintingFromEachCategory = 70 := by
  sorry

theorem question2 : numWaysToChooseTwoPaintingsOfDifferentTypes = 59 := by
  sorry

end question1_question2_l72_72903


namespace total_students_l72_72654

variable (A B AB : ℕ)

-- Conditions
axiom h1 : AB = (1 / 5) * (A + AB)
axiom h2 : AB = (1 / 4) * (B + AB)
axiom h3 : A - B = 75

-- Proof problem
theorem total_students : A + B + AB = 600 :=
by
  sorry

end total_students_l72_72654


namespace evaluate_division_l72_72035

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l72_72035


namespace table_seating_problem_l72_72703

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l72_72703


namespace percentage_discount_proof_l72_72445

noncomputable def ticket_price : ℝ := 25
noncomputable def price_to_pay : ℝ := 18.75
noncomputable def discount_amount : ℝ := ticket_price - price_to_pay
noncomputable def percentage_discount : ℝ := (discount_amount / ticket_price) * 100

theorem percentage_discount_proof : percentage_discount = 25 := by
  sorry

end percentage_discount_proof_l72_72445


namespace length_AB_given_conditions_l72_72097

variable {A B P Q : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField P] [LinearOrderedField Q]

def length_of_AB (x y : A) : A := x + y

theorem length_AB_given_conditions (x y u v : A) (hx : y = 4 * x) (hv : 5 * u = 2 * v) (hu : u = x + 3) (hv' : v = y - 3) (hPQ : PQ = 3) : length_of_AB x y = 35 :=
by
  sorry

end length_AB_given_conditions_l72_72097


namespace termite_ridden_fraction_l72_72223

theorem termite_ridden_fraction (T : ℝ)
  (h1 : (3 / 10) * T = 0.1) : T = 1 / 3 :=
by
  -- proof goes here
  sorry

end termite_ridden_fraction_l72_72223


namespace product_of_roots_eq_neg_14_l72_72040

theorem product_of_roots_eq_neg_14 :
  ∀ (x : ℝ), 25 * x^2 + 60 * x - 350 = 0 → ((-350) / 25) = -14 :=
by
  intros x h
  sorry

end product_of_roots_eq_neg_14_l72_72040


namespace vector_parallel_l72_72190

theorem vector_parallel {x : ℝ} (h : (4 / x) = (-2 / 5)) : x = -10 :=
  by
  sorry

end vector_parallel_l72_72190


namespace prove_f_neg1_eq_0_l72_72488

def f : ℝ → ℝ := sorry

theorem prove_f_neg1_eq_0
  (h1 : ∀ x : ℝ, f(x + 2) = f(2 - x))
  (h2 : ∀ x : ℝ, f(1 - 2 * x) = -f(2 * x + 1))
  : f(-1) = 0 := sorry

end prove_f_neg1_eq_0_l72_72488


namespace range_of_f_l72_72642

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem range_of_f : ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), 
  -Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end range_of_f_l72_72642


namespace trigonometric_identity_l72_72837

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := 
by
  -- proof steps are omitted, using sorry to skip the proof.
  sorry

end trigonometric_identity_l72_72837


namespace intersecting_circles_l72_72905

theorem intersecting_circles (m n : ℝ) (h_intersect : ∃ c1 c2 : ℝ × ℝ, 
  (c1.1 - c1.2 - 2 = 0) ∧ (c2.1 - c2.2 - 2 = 0) ∧
  ∃ r1 r2 : ℝ, (c1.1 - 1)^2 + (c1.2 - 3)^2 = r1^2 ∧ (c2.1 - 1)^2 + (c2.2 - 3)^2 = r2^2 ∧
  (c1.1 - m)^2 + (c1.2 - n)^2 = r1^2 ∧ (c2.1 - m)^2 + (c2.2 - n)^2 = r2^2) :
  m + n = 4 :=
sorry

end intersecting_circles_l72_72905


namespace find_positive_integers_satisfying_inequality_l72_72621

theorem find_positive_integers_satisfying_inequality :
  (∃ n : ℕ, (n - 1) * (n - 3) * (n - 5) * (n - 7) * (n - 9) * (n - 11) * (n - 13) * (n - 15) *
    (n - 17) * (n - 19) * (n - 21) * (n - 23) * (n - 25) * (n - 27) * (n - 29) * (n - 31) *
    (n - 33) * (n - 35) * (n - 37) * (n - 39) * (n - 41) * (n - 43) * (n - 45) * (n - 47) *
    (n - 49) * (n - 51) * (n - 53) * (n - 55) * (n - 57) * (n - 59) * (n - 61) * (n - 63) *
    (n - 65) * (n - 67) * (n - 69) * (n - 71) * (n - 73) * (n - 75) * (n - 77) * (n - 79) *
    (n - 81) * (n - 83) * (n - 85) * (n - 87) * (n - 89) * (n - 91) * (n - 93) * (n - 95) *
    (n - 97) * (n - 99) < 0 ∧ 1 ≤ n ∧ n ≤ 99) 
  → ∃ f : ℕ → ℕ, (∀ i, f i = 2 + 4 * i) ∧ (∀ i, 1 ≤ f i ∧ f i ≤ 24) :=
by
  sorry

end find_positive_integers_satisfying_inequality_l72_72621


namespace twenty_percent_correct_l72_72583

def certain_number := 400
def forty_percent (x : ℕ) : ℕ := 40 * x / 100
def twenty_percent_of_certain_number (x : ℕ) : ℕ := 20 * x / 100

theorem twenty_percent_correct : 
  (∃ x : ℕ, forty_percent x = 160) → twenty_percent_of_certain_number certain_number = 80 :=
by
  sorry

end twenty_percent_correct_l72_72583


namespace length_FD_of_folded_square_l72_72857

theorem length_FD_of_folded_square :
  let A := (0, 0)
  let B := (8, 0)
  let D := (0, 8)
  let C := (8, 8)
  let E := (6, 0)
  let F := (8, 8 - (FD : ℝ))
  (ABCD_square : ∀ {x y : ℝ}, (x = 0 ∨ x = 8) ∧ (y = 0 ∨ y = 8)) →  
  let DE := (6 - 0 : ℝ)
  let Pythagorean_statement := (8 - FD) ^ 2 = FD ^ 2 + 6 ^ 2
  ∃ FD : ℝ, FD = 7 / 4 :=
sorry

end length_FD_of_folded_square_l72_72857


namespace total_people_seated_l72_72692

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l72_72692


namespace juniors_score_l72_72431

/-- Mathematical proof problem stated in Lean 4 -/
theorem juniors_score 
  (total_students : ℕ) 
  (juniors seniors : ℕ)
  (junior_score senior_avg total_avg : ℝ)
  (h_total_students : total_students > 0)
  (h_juniors : juniors = total_students / 10)
  (h_seniors : seniors = (total_students * 9) / 10)
  (h_total_avg : total_avg = 84)
  (h_senior_avg : senior_avg = 83)
  (h_junior_score_same : ∀ j : ℕ, j < juniors → ∃ s : ℝ, s = junior_score)
  :
  junior_score = 93 :=
by
  sorry

end juniors_score_l72_72431


namespace monotonicity_and_range_of_a_l72_72187

noncomputable def f (x a : ℝ) := Real.log x - a * x - 2

theorem monotonicity_and_range_of_a (a : ℝ) (h : a ≠ 0) :
  ((∀ x > 0, (Real.log x - a * x - 2) < (Real.log (x + 1) - a * (x + 1) - 2)) ↔ (a < 0)) ∧
  ((∃ M, M = Real.log (1/a) - a * (1/a) - 2 ∧ M > a - 4) → 0 < a ∧ a < 1) := sorry

end monotonicity_and_range_of_a_l72_72187


namespace burger_cost_proof_l72_72298

variable {burger_cost fries_cost salad_cost total_cost : ℕ}
variable {quantity_of_fries : ℕ}

theorem burger_cost_proof (h_fries_cost : fries_cost = 2)
    (h_salad_cost : salad_cost = 3 * fries_cost)
    (h_quantity_of_fries : quantity_of_fries = 2)
    (h_total_cost : total_cost = 15)
    (h_equation : burger_cost + (quantity_of_fries * fries_cost) + salad_cost = total_cost) :
    burger_cost = 5 :=
by 
  sorry

end burger_cost_proof_l72_72298


namespace number_of_divisors_3465_l72_72618

def prime_factors_3465 : Prop := 3465 = 3^2 * 5 * 7^2

theorem number_of_divisors_3465 (h : prime_factors_3465) : Nat.totient 3465 = 18 :=
  sorry

end number_of_divisors_3465_l72_72618


namespace solve_for_x_l72_72713

-- Assumptions and conditions of the problem
def a : ℚ := 4 / 7
def b : ℚ := 1 / 5
def c : ℚ := 12
def d : ℚ := 105

-- The statement of the problem
theorem solve_for_x (x : ℚ) (h : a * b * x = c) : x = d :=
by sorry

end solve_for_x_l72_72713


namespace regular_hexagon_interior_angle_l72_72741

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l72_72741


namespace non_participating_members_l72_72353

noncomputable def members := 35
noncomputable def badminton_players := 15
noncomputable def tennis_players := 18
noncomputable def both_players := 3

theorem non_participating_members : 
  members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end non_participating_members_l72_72353


namespace number_of_people_seated_l72_72706

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l72_72706


namespace evaluate_expression_l72_72090

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end evaluate_expression_l72_72090


namespace value_of_c7_l72_72873

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end value_of_c7_l72_72873


namespace trigonometric_identity_l72_72098

open Real

-- Lean 4 statement
theorem trigonometric_identity (α β γ x : ℝ) :
  (sin (x - β) * sin (x - γ) / (sin (α - β) * sin (α - γ))) +
  (sin (x - γ) * sin (x - α) / (sin (β - γ) * sin (β - α))) +
  (sin (x - α) * sin (x - β) / (sin (γ - α) * sin (γ - β))) = 1 := 
sorry

end trigonometric_identity_l72_72098


namespace range_of_a_l72_72324

noncomputable def set_A : Set ℝ := { x | x^2 - 3 * x - 10 < 0 }
noncomputable def set_B : Set ℝ := { x | x^2 + 2 * x - 8 > 0 }
def set_C (a : ℝ) : Set ℝ := { x | 2 * a < x ∧ x < a + 3 }

theorem range_of_a (a : ℝ) :
  (A ∩ B) ∩ set_C a = set_C a → 1 ≤ a := 
sorry

end range_of_a_l72_72324


namespace initial_bottles_count_l72_72904

theorem initial_bottles_count : 
  ∀ (jason_buys harry_buys bottles_left initial_bottles : ℕ), 
  jason_buys = 5 → 
  harry_buys = 6 → 
  bottles_left = 24 → 
  initial_bottles = bottles_left + jason_buys + harry_buys → 
  initial_bottles = 35 :=
by
  intros jason_buys harry_buys bottles_left initial_bottles
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end initial_bottles_count_l72_72904


namespace greatest_price_drop_is_april_l72_72788

-- Define the price changes for each month
def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => 1.00
  | 2 => -1.50
  | 3 => -0.50
  | 4 => -3.75 -- including the -1.25 adjustment
  | 5 => 0.50
  | 6 => -2.25
  | _ => 0 -- default case, although we only deal with months 1-6

-- Define a predicate for the month with the greatest drop
def greatest_drop_month (m : ℕ) : Prop :=
  m = 4

-- Main theorem: Prove that the month with the greatest price drop is April
theorem greatest_price_drop_is_april : greatest_drop_month 4 :=
by
  -- Use Lean tactics to prove the statement
  sorry

end greatest_price_drop_is_april_l72_72788


namespace income_exceeds_previous_l72_72945

noncomputable def a_n (a b : ℝ) (n : ℕ) : ℝ :=
if n = 1 then a
else a * (2 / 3)^(n - 1) + b * (3 / 2)^(n - 2)

theorem income_exceeds_previous (a b : ℝ) (h : b ≥ 3 * a / 8) (n : ℕ) (hn : n ≥ 2) : 
  a_n a b n ≥ a :=
sorry

end income_exceeds_previous_l72_72945


namespace rational_eq_reciprocal_l72_72794

theorem rational_eq_reciprocal (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 :=
by {
  sorry
}

end rational_eq_reciprocal_l72_72794


namespace fraction_equality_l72_72474

variables (x y : ℝ)

theorem fraction_equality (h : y / 2 = (2 * y - x) / 3) : y / x = 2 :=
sorry

end fraction_equality_l72_72474


namespace prod_mod_11_remainder_zero_l72_72907

theorem prod_mod_11_remainder_zero : (108 * 110) % 11 = 0 := 
by sorry

end prod_mod_11_remainder_zero_l72_72907


namespace statement_A_statement_C_statement_D_l72_72503

variable (a : ℕ → ℝ) (A B : ℝ)

-- Condition: The sequence satisfies the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  a (n + 2) = A * a (n + 1) + B * a n

-- Statement A: A=1 and B=-1 imply periodic with period 6
theorem statement_A (h : ∀ n, recurrence_relation a 1 (-1) n) :
  ∀ n, a (n + 6) = a n := 
sorry

-- Statement C: A=3 and B=-2 imply the derived sequence is geometric
theorem statement_C (h : ∀ n, recurrence_relation a 3 (-2) n) :
  ∃ r : ℝ, ∀ n, a (n + 1) - a n = r * (a n - a (n - 1)) :=
sorry

-- Statement D: A+1=B, a1=0, a2=B imply {a_{2n}} is increasing
theorem statement_D (hA : ∀ n, recurrence_relation a A (A + 1) n)
  (h1 : a 1 = 0) (h2 : a 2 = A + 1) :
  ∀ n, a (2 * (n + 1)) > a (2 * n) :=
sorry

end statement_A_statement_C_statement_D_l72_72503


namespace find_b_of_square_polynomial_l72_72547

theorem find_b_of_square_polynomial 
  (a b : ℚ)
  (h : ∃ p q : ℚ, (x^4 + x^3 - x^2 + a * x + b) = (x^2 + p * x + q)^2) :
  b = 25 / 64 :=
by 
  cases h with p hp
  cases hp with q hq
  sorry 

end find_b_of_square_polynomial_l72_72547


namespace roots_of_quadratic_equation_are_real_and_distinct_l72_72555

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l72_72555


namespace regular_hexagon_interior_angle_l72_72747

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l72_72747


namespace min_value_f_x_gt_1_min_value_a_f_x_lt_1_l72_72329

def f (x : ℝ) : ℝ := 4 * x + 1 / (x - 1)

theorem min_value_f_x_gt_1 : 
  (∀ x : ℝ, x > 1 → f x ≥ 8) ∧ (∃ x : ℝ, x > 1 ∧ f x = 8) := 
by 
  sorry

theorem min_value_a_f_x_lt_1 : 
  (∀ x : ℝ, x < 1 → f x ≤ 0) ∧ (∀ a : ℝ, (∀ x : ℝ, x < 1 → f x ≤ a) → a ≥ 0 ∧ (∃ x : ℝ, f x = 0)) := 
by 
  sorry

end min_value_f_x_gt_1_min_value_a_f_x_lt_1_l72_72329


namespace find_cost_price_l72_72272

variable (CP : ℝ) -- cost price
variable (SP_loss SP_gain : ℝ) -- selling prices

-- Conditions
def loss_condition := SP_loss = 0.9 * CP
def gain_condition := SP_gain = 1.04 * CP
def difference_condition := SP_gain - SP_loss = 190

-- Theorem to prove
theorem find_cost_price (h_loss : loss_condition CP SP_loss)
                        (h_gain : gain_condition CP SP_gain)
                        (h_diff : difference_condition SP_loss SP_gain) :
  CP = 1357.14 := 
sorry

end find_cost_price_l72_72272


namespace sum_of_squares_l72_72981

theorem sum_of_squares (x y z a b c k : ℝ)
  (h₁ : x * y = k * a)
  (h₂ : x * z = b)
  (h₃ : y * z = c)
  (hk : k ≠ 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) :=
by
  sorry

end sum_of_squares_l72_72981


namespace manuscript_pages_l72_72251

theorem manuscript_pages (P : ℕ) (rate_first : ℕ) (rate_revision : ℕ) 
  (revised_once_pages : ℕ) (revised_twice_pages : ℕ) (total_cost : ℕ) :
  rate_first = 6 →
  rate_revision = 4 →
  revised_once_pages = 35 →
  revised_twice_pages = 15 →
  total_cost = 860 →
  6 * (P - 35 - 15) + 10 * 35 + 14 * 15 = total_cost →
  P = 100 :=
by
  intros h_first h_revision h_once h_twice h_cost h_eq
  sorry

end manuscript_pages_l72_72251


namespace geometric_sequence_product_l72_72631

theorem geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (hA_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (hA_not_zero : ∀ n, a n ≠ 0)
  (h_condition : a 4 - 2 * (a 7)^2 + 3 * a 8 = 0)
  (hB_seq : ∀ n, b n = b 1 * (b 2 / b 1)^(n - 1))
  (hB7 : b 7 = a 7) :
  b 3 * b 7 * b 11 = 8 := 
sorry

end geometric_sequence_product_l72_72631


namespace wrapping_paper_fraction_each_present_l72_72103

theorem wrapping_paper_fraction_each_present (total_fraction : ℚ) (num_presents : ℕ) 
  (H : total_fraction = 3/10) (H1 : num_presents = 3) :
  total_fraction / num_presents = 1/10 :=
by sorry

end wrapping_paper_fraction_each_present_l72_72103
