import Mathlib

namespace NUMINAMATH_GPT_expression_simplification_l1664_166433

theorem expression_simplification :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := 
sorry

end NUMINAMATH_GPT_expression_simplification_l1664_166433


namespace NUMINAMATH_GPT_cube_sum_identity_l1664_166494

theorem cube_sum_identity (p q r : ℝ)
  (h₁ : p + q + r = 4)
  (h₂ : pq + qr + rp = 6)
  (h₃ : pqr = -8) :
  p^3 + q^3 + r^3 = 64 := 
by
  sorry

end NUMINAMATH_GPT_cube_sum_identity_l1664_166494


namespace NUMINAMATH_GPT_find_angle_C_l1664_166477

theorem find_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 10 * a * Real.cos B = 3 * b * Real.cos A) 
  (h2 : Real.cos A = (5 * Real.sqrt 26) / 26) 
  (h3 : A + B + C = π) : 
  C = (3 * π) / 4 :=
sorry

end NUMINAMATH_GPT_find_angle_C_l1664_166477


namespace NUMINAMATH_GPT_survey_is_sample_of_population_l1664_166462

-- Definitions based on the conditions in a)
def population_size := 50000
def sample_size := 2000
def is_comprehensive_survey := false
def is_sampling_survey := true
def is_population_student (n : ℕ) : Prop := n ≤ population_size
def is_individual_unit (n : ℕ) : Prop := n ≤ sample_size

-- Theorem that encapsulates the proof problem
theorem survey_is_sample_of_population : is_sampling_survey ∧ ∃ n, is_individual_unit n :=
by
  sorry

end NUMINAMATH_GPT_survey_is_sample_of_population_l1664_166462


namespace NUMINAMATH_GPT_ratio_of_distances_l1664_166444

-- Define the speeds and times for ferries P and Q
def speed_P : ℝ := 8
def time_P : ℝ := 3
def speed_Q : ℝ := speed_P + 1
def time_Q : ℝ := time_P + 5

-- Define the distances covered by ferries P and Q
def distance_P : ℝ := speed_P * time_P
def distance_Q : ℝ := speed_Q * time_Q

-- The statement to prove: the ratio of the distances
theorem ratio_of_distances : distance_Q / distance_P = 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_distances_l1664_166444


namespace NUMINAMATH_GPT_find_possible_values_l1664_166438

theorem find_possible_values (a b c k : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b * c) * (b^2 - a * c) + 
   (a^2 - b * c) * (c^2 - a * b) + 
   (b^2 - a * c) * (c^2 - a * b)) 
  = k / 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_possible_values_l1664_166438


namespace NUMINAMATH_GPT_cost_of_parts_l1664_166457

theorem cost_of_parts (C : ℝ) 
  (h1 : ∀ n ∈ List.range 60, (1.4 * C * n) = (1.4 * C * 60))
  (h2 : 5000 + 3000 = 8000)
  (h3 : 60 * C * 1.4 - (60 * C + 8000) = 11200) : 
  C = 800 := by
  sorry

end NUMINAMATH_GPT_cost_of_parts_l1664_166457


namespace NUMINAMATH_GPT_sam_quarters_l1664_166488

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sam_quarters_l1664_166488


namespace NUMINAMATH_GPT_B_share_is_102_l1664_166453

variables (A B C : ℝ)
variables (total : ℝ)
variables (rA_B : ℝ) (rB_C : ℝ)

-- Conditions
def conditions : Prop :=
  (total = 578) ∧
  (rA_B = 2 / 3) ∧
  (rB_C = 1 / 4) ∧
  (A = rA_B * B) ∧
  (B = rB_C * C) ∧
  (A + B + C = total)

-- Theorem to prove B's share
theorem B_share_is_102 (h : conditions A B C total rA_B rB_C) : B = 102 :=
by sorry

end NUMINAMATH_GPT_B_share_is_102_l1664_166453


namespace NUMINAMATH_GPT_probability_of_both_making_basket_l1664_166432

noncomputable def P : Set ℕ → ℚ :=
  sorry

def A : Set ℕ := sorry
def B : Set ℕ := sorry

axiom prob_A : P A = 2 / 5
axiom prob_B : P B = 1 / 2
axiom independent : P (A ∩ B) = P A * P B

theorem probability_of_both_making_basket :
  P (A ∩ B) = 1 / 5 :=
by
  rw [independent, prob_A, prob_B]
  norm_num

end NUMINAMATH_GPT_probability_of_both_making_basket_l1664_166432


namespace NUMINAMATH_GPT_evaluate_expression_l1664_166413

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1664_166413


namespace NUMINAMATH_GPT_ratio_new_average_to_original_l1664_166454

theorem ratio_new_average_to_original (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / scores.length : ℝ)
  let new_sum := scores.sum + 2 * A
  let new_avg := new_sum / (scores.length + 2)
  new_avg / A = 1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_new_average_to_original_l1664_166454


namespace NUMINAMATH_GPT_price_increase_percentage_l1664_166470

variables
  (coffees_daily_before : ℕ := 4)
  (price_per_coffee_before : ℝ := 2)
  (coffees_daily_after : ℕ := 2)
  (price_increase_savings : ℝ := 2)
  (spending_before := coffees_daily_before * price_per_coffee_before)
  (spending_after := spending_before - price_increase_savings)
  (price_per_coffee_after := spending_after / coffees_daily_after)

theorem price_increase_percentage :
  ((price_per_coffee_after - price_per_coffee_before) / price_per_coffee_before) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_price_increase_percentage_l1664_166470


namespace NUMINAMATH_GPT_remainder_division_39_l1664_166456

theorem remainder_division_39 (N : ℕ) (k m R1 : ℕ) (hN1 : N = 39 * k + R1) (hN2 : N % 13 = 5) (hR1_lt_39 : R1 < 39) :
  R1 = 5 :=
by sorry

end NUMINAMATH_GPT_remainder_division_39_l1664_166456


namespace NUMINAMATH_GPT_next_four_customers_cases_l1664_166464

theorem next_four_customers_cases (total_people : ℕ) (first_eight_cases : ℕ) (last_eight_cases : ℕ) (total_cases : ℕ) :
    total_people = 20 →
    first_eight_cases = 24 →
    last_eight_cases = 8 →
    total_cases = 40 →
    (total_cases - (first_eight_cases + last_eight_cases)) / 4 = 2 :=
by
  intro h1 h2 h3 h4
  -- Fill in the proof steps using h1, h2, h3, and h4
  sorry

end NUMINAMATH_GPT_next_four_customers_cases_l1664_166464


namespace NUMINAMATH_GPT_rectangle_area_l1664_166487

theorem rectangle_area :
  ∃ (a b : ℕ), a ≠ b ∧ Even a ∧ (a * b = 3 * (2 * a + 2 * b)) ∧ (a * b = 162) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1664_166487


namespace NUMINAMATH_GPT_sector_area_l1664_166471

-- Define the given parameters
def central_angle : ℝ := 2
def radius : ℝ := 3

-- Define the statement about the area of the sector
theorem sector_area (α r : ℝ) (hα : α = 2) (hr : r = 3) :
  let l := α * r
  let A := 0.5 * l * r
  A = 9 :=
by
  -- The proof is not required
  sorry

end NUMINAMATH_GPT_sector_area_l1664_166471


namespace NUMINAMATH_GPT_paula_bracelets_count_l1664_166481

-- Defining the given conditions
def cost_bracelet := 4
def cost_keychain := 5
def cost_coloring_book := 3
def total_spent := 20

-- Defining the cost for Paula's items
def cost_paula (B : ℕ) := B * cost_bracelet + cost_keychain

-- Defining the cost for Olive's items
def cost_olive := cost_coloring_book + cost_bracelet

-- Defining the main problem
theorem paula_bracelets_count (B : ℕ) (h : cost_paula B + cost_olive = total_spent) : B = 2 := by
  sorry

end NUMINAMATH_GPT_paula_bracelets_count_l1664_166481


namespace NUMINAMATH_GPT_find_divisor_l1664_166408

variable (Dividend : ℕ) (Quotient : ℕ) (Divisor : ℕ)
variable (h1 : Dividend = 64)
variable (h2 : Quotient = 8)
variable (h3 : Dividend = Divisor * Quotient)

theorem find_divisor : Divisor = 8 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1664_166408


namespace NUMINAMATH_GPT_percent_absent_math_dept_l1664_166482

theorem percent_absent_math_dept (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_absent_fraction : ℚ) (female_absent_fraction : ℚ)
  (h1 : total_students = 160) 
  (h2 : male_students = 90) 
  (h3 : female_students = 70) 
  (h4 : male_absent_fraction = 1 / 5) 
  (h5 : female_absent_fraction = 2 / 7) :
  ((male_absent_fraction * male_students + female_absent_fraction * female_students) / total_students) * 100 = 23.75 :=
by
  sorry

end NUMINAMATH_GPT_percent_absent_math_dept_l1664_166482


namespace NUMINAMATH_GPT_total_students_surveyed_l1664_166439

-- Define the constants for liked and disliked students.
def liked_students : ℕ := 235
def disliked_students : ℕ := 165

-- The theorem to prove the total number of students surveyed.
theorem total_students_surveyed : liked_students + disliked_students = 400 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_total_students_surveyed_l1664_166439


namespace NUMINAMATH_GPT_joao_chocolates_l1664_166449

theorem joao_chocolates (n : ℕ) (hn1 : 30 < n) (hn2 : n < 100) (h1 : n % 7 = 1) (h2 : n % 10 = 2) : n = 92 :=
sorry

end NUMINAMATH_GPT_joao_chocolates_l1664_166449


namespace NUMINAMATH_GPT_problem1_problem2_l1664_166431

-- Problem 1: Prove that (1) - 8 + 12 - 16 - 23 = -35
theorem problem1 : (1 - 8 + 12 - 16 - 23 = -35) :=
by
  sorry

-- Problem 2: Prove that (3 / 4) + (-1 / 6) - (1 / 3) - (-1 / 8) = 3 / 8
theorem problem2 : (3 / 4 + (-1 / 6) - 1 / 3 + 1 / 8 = 3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1664_166431


namespace NUMINAMATH_GPT_greatest_number_of_consecutive_integers_sum_36_l1664_166411

theorem greatest_number_of_consecutive_integers_sum_36 :
  ∃ (N : ℕ), 
    (∃ a : ℤ, N * a + ((N - 1) * N) / 2 = 36) ∧ 
    (∀ N' : ℕ, (∃ a' : ℤ, N' * a' + ((N' - 1) * N') / 2 = 36) → N' ≤ 72) := by
  sorry

end NUMINAMATH_GPT_greatest_number_of_consecutive_integers_sum_36_l1664_166411


namespace NUMINAMATH_GPT_inequality_proof_l1664_166479

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1664_166479


namespace NUMINAMATH_GPT_solve_arcsin_eq_l1664_166492

noncomputable def arcsin (x : ℝ) : ℝ := Real.arcsin x
noncomputable def pi : ℝ := Real.pi

theorem solve_arcsin_eq :
  ∃ x : ℝ, arcsin x + arcsin (3 * x) = pi / 4 ∧ x = 1 / Real.sqrt 19 :=
sorry

end NUMINAMATH_GPT_solve_arcsin_eq_l1664_166492


namespace NUMINAMATH_GPT_inequality_proof_l1664_166486

theorem inequality_proof (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1664_166486


namespace NUMINAMATH_GPT_lateral_surface_area_of_cube_l1664_166406

-- Define the side length of the cube
def side_length : ℕ := 12

-- Define the area of one face of the cube
def area_of_one_face (s : ℕ) : ℕ := s * s

-- Define the lateral surface area of the cube
def lateral_surface_area (s : ℕ) : ℕ := 4 * (area_of_one_face s)

-- Prove the lateral surface area of a cube with side length 12 m is equal to 576 m²
theorem lateral_surface_area_of_cube : lateral_surface_area side_length = 576 := by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cube_l1664_166406


namespace NUMINAMATH_GPT_base9_minus_base6_to_decimal_l1664_166452

theorem base9_minus_base6_to_decimal :
  let b9 := 3 * 9^2 + 2 * 9^1 + 1 * 9^0
  let b6 := 2 * 6^2 + 5 * 6^1 + 4 * 6^0
  b9 - b6 = 156 := by
sorry

end NUMINAMATH_GPT_base9_minus_base6_to_decimal_l1664_166452


namespace NUMINAMATH_GPT_train_speed_excluding_stoppages_l1664_166483

-- Define the speed of the train excluding stoppages and including stoppages
variables (S : ℕ) -- S is the speed of the train excluding stoppages
variables (including_stoppages_speed : ℕ := 40) -- The speed including stoppages is 40 kmph

-- The train stops for 20 minutes per hour. This means it runs for (60 - 20) minutes per hour.
def running_time_per_hour := 40

-- Converting 40 minutes to hours
def running_fraction_of_hour : ℚ := 40 / 60

-- Formulate the main theorem:
theorem train_speed_excluding_stoppages
    (H1 : including_stoppages_speed = 40)
    (H2 : running_fraction_of_hour = 2 / 3) :
    S = 60 :=
by
    sorry

end NUMINAMATH_GPT_train_speed_excluding_stoppages_l1664_166483


namespace NUMINAMATH_GPT_area_of_triangle_l1664_166420

def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -2 * x + 10

theorem area_of_triangle : 
  let inter_x := (10 - 6) / (3 + 2)
  let inter_y := line1 inter_x
  let base := (10 - 6 : ℝ)
  let height := inter_x
  base * height / 2 = 8 / 5 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1664_166420


namespace NUMINAMATH_GPT_lucas_purchase_l1664_166493

-- Define the variables and assumptions.
variables (a b c : ℕ)
variables (h1 : a + b + c = 50) (h2 : 50 * a + 400 * b + 500 * c = 10000)

-- Goal: Prove that the number of 50-cent items (a) is 30.
theorem lucas_purchase : a = 30 :=
by sorry

end NUMINAMATH_GPT_lucas_purchase_l1664_166493


namespace NUMINAMATH_GPT_percentage_difference_l1664_166415

theorem percentage_difference (N : ℝ) (hN : N = 160) : 0.50 * N - 0.35 * N = 24 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1664_166415


namespace NUMINAMATH_GPT_dorothy_money_left_l1664_166410

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end NUMINAMATH_GPT_dorothy_money_left_l1664_166410


namespace NUMINAMATH_GPT_sin_from_tan_l1664_166440

theorem sin_from_tan (A : ℝ) (h : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_sin_from_tan_l1664_166440


namespace NUMINAMATH_GPT_quadrilateral_inscribed_circumscribed_l1664_166480

theorem quadrilateral_inscribed_circumscribed 
  (r R d : ℝ) --Given variables with their types
  (K O : Type) (radius_K : K → ℝ) (radius_O : O → ℝ) (dist : (K × O) → ℝ)  -- Defining circles properties
  (K_inside_O : ∀ p : K × O, radius_K p.fst < radius_O p.snd) 
  (dist_centers : ∀ p : K × O, dist p = d) -- Distance between the centers
  : 
  (1 / (R + d)^2) + (1 / (R - d)^2) = (1 / r^2) := 
by 
  sorry

end NUMINAMATH_GPT_quadrilateral_inscribed_circumscribed_l1664_166480


namespace NUMINAMATH_GPT_archie_initial_marbles_l1664_166416

theorem archie_initial_marbles (M : ℝ) (h1 : 0.6 * M + 0.5 * 0.4 * M = M - 20) : M = 100 :=
sorry

end NUMINAMATH_GPT_archie_initial_marbles_l1664_166416


namespace NUMINAMATH_GPT_cross_number_puzzle_digit_star_l1664_166460

theorem cross_number_puzzle_digit_star :
  ∃ N₁ N₂ N₃ N₄ : ℕ,
    N₁ % 1000 / 100 = 4 ∧ N₁ % 10 = 1 ∧ ∃ n : ℕ, N₁ = n ^ 2 ∧
    N₃ % 1000 / 100 = 6 ∧ ∃ m : ℕ, N₃ = m ^ 4 ∧
    ∃ p : ℕ, N₂ = 2 * p ^ 5 ∧ 100 ≤ N₂ ∧ N₂ < 1000 ∧
    N₄ % 10 = 5 ∧ ∃ q : ℕ, N₄ = q ^ 3 ∧ 100 ≤ N₄ ∧ N₄ < 1000 ∧
    (N₁ % 10 = 4) :=
by
  sorry

end NUMINAMATH_GPT_cross_number_puzzle_digit_star_l1664_166460


namespace NUMINAMATH_GPT_total_pears_picked_l1664_166418

theorem total_pears_picked (keith_pears jason_pears : ℕ) (h1 : keith_pears = 3) (h2 : jason_pears = 2) : keith_pears + jason_pears = 5 :=
by
  sorry

end NUMINAMATH_GPT_total_pears_picked_l1664_166418


namespace NUMINAMATH_GPT_find_a_5_l1664_166443

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem find_a_5 {a : ℕ → ℤ} {S : ℕ → ℤ}
  (h_seq : arithmetic_sequence a)
  (h_S6 : S 6 = 3)
  (h_a4 : a 4 = 2)
  (h_sum_first_n : sum_first_n a S) :
  a 5 = 5 := 
sorry

end NUMINAMATH_GPT_find_a_5_l1664_166443


namespace NUMINAMATH_GPT_complex_arithmetic_l1664_166445

def Q : ℂ := 7 + 3 * Complex.I
def E : ℂ := 2 * Complex.I
def D : ℂ := 7 - 3 * Complex.I
def F : ℂ := 1 + Complex.I

theorem complex_arithmetic : (Q * E * D) + F = 1 + 117 * Complex.I := by
  sorry

end NUMINAMATH_GPT_complex_arithmetic_l1664_166445


namespace NUMINAMATH_GPT_find_S11_l1664_166403

variable (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

axiom sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n, S n = n * (a 1 + a n) / 2
axiom condition1 : is_arithmetic_sequence a
axiom condition2 : a 5 + a 7 = (a 6)^2

-- Proof (statement) that the sum of the first 11 terms is 22
theorem find_S11 : S 11 = 22 :=
  sorry

end NUMINAMATH_GPT_find_S11_l1664_166403


namespace NUMINAMATH_GPT_trigonometric_identity_l1664_166425

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) * Real.cos (Real.pi / 4 - α) = 7 / 18 :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l1664_166425


namespace NUMINAMATH_GPT_smallest_a_l1664_166435

theorem smallest_a (a : ℕ) (h_a : a > 8) : (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ↔ a = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l1664_166435


namespace NUMINAMATH_GPT_new_remainder_when_scaled_l1664_166450

theorem new_remainder_when_scaled (a b c : ℕ) (h : a = b * c + 7) : (10 * a) % (10 * b) = 70 := by
  sorry

end NUMINAMATH_GPT_new_remainder_when_scaled_l1664_166450


namespace NUMINAMATH_GPT_range_of_a_for_decreasing_f_l1664_166455

theorem range_of_a_for_decreasing_f :
  (∀ x : ℝ, (-3) * x^2 + 2 * a * x - 1 ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_range_of_a_for_decreasing_f_l1664_166455


namespace NUMINAMATH_GPT_arithmetic_mean_of_three_digit_multiples_of_8_l1664_166404

-- Define the conditions given in the problem
def smallest_three_digit_multiple_of_8 := 104
def largest_three_digit_multiple_of_8 := 992
def common_difference := 8

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  smallest_three_digit_multiple_of_8 + n * common_difference

-- Calculate the number of terms in the sequence
def number_of_terms : ℕ :=
  (largest_three_digit_multiple_of_8 - smallest_three_digit_multiple_of_8) / common_difference + 1

-- Calculate the sum of the arithmetic sequence
def sum_of_sequence : ℕ :=
  (number_of_terms * (smallest_three_digit_multiple_of_8 + largest_three_digit_multiple_of_8)) / 2

-- Calculate the arithmetic mean
def arithmetic_mean : ℕ :=
  sum_of_sequence / number_of_terms

-- The statement to be proved
theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  arithmetic_mean = 548 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_three_digit_multiples_of_8_l1664_166404


namespace NUMINAMATH_GPT_johns_weight_l1664_166421

-- Definitions based on the given conditions
def max_weight : ℝ := 1000
def safety_percentage : ℝ := 0.20
def bar_weight : ℝ := 550

-- Theorem stating the mathematically equivalent proof problem
theorem johns_weight : 
  (johns_safe_weight : ℝ) = max_weight - safety_percentage * max_weight 
  → (johns_safe_weight - bar_weight = 250) :=
by
  sorry

end NUMINAMATH_GPT_johns_weight_l1664_166421


namespace NUMINAMATH_GPT_find_original_number_l1664_166442

/-- Given that one less than the reciprocal of a number is 5/2, the original number must be -2/3. -/
theorem find_original_number (y : ℚ) (h : 1 - 1 / y = 5 / 2) : y = -2 / 3 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1664_166442


namespace NUMINAMATH_GPT_min_value_z_l1664_166459

theorem min_value_z : ∀ (x y : ℝ), ∃ z, z = 3 * x^2 + y^2 + 12 * x - 6 * y + 40 ∧ z = 19 :=
by
  intro x y
  use 3 * x^2 + y^2 + 12 * x - 6 * y + 40 -- Define z
  sorry -- Proof is skipped for now

end NUMINAMATH_GPT_min_value_z_l1664_166459


namespace NUMINAMATH_GPT_binom_sum_l1664_166423

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_sum : binom 7 4 + binom 6 5 = 41 := by
  sorry

end NUMINAMATH_GPT_binom_sum_l1664_166423


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1664_166426

theorem solution_set_of_inequality (x : ℝ) : x < (1 / x) ↔ (x < -1 ∨ (0 < x ∧ x < 1)) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1664_166426


namespace NUMINAMATH_GPT_seq_a2010_l1664_166429

-- Definitions and conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 3 ∧ 
  ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

-- Proof statement
theorem seq_a2010 {a : ℕ → ℕ} (h : seq a) : a 2010 = 4 := 
  sorry

end NUMINAMATH_GPT_seq_a2010_l1664_166429


namespace NUMINAMATH_GPT_moles_of_nacl_formed_l1664_166468

noncomputable def reaction (nh4cl: ℕ) (naoh: ℕ) : ℕ :=
  if nh4cl = naoh then nh4cl else min nh4cl naoh

theorem moles_of_nacl_formed (nh4cl: ℕ) (naoh: ℕ) (h_nh4cl: nh4cl = 2) (h_naoh: naoh = 2) :
  reaction nh4cl naoh = 2 :=
by
  rw [h_nh4cl, h_naoh]
  sorry

end NUMINAMATH_GPT_moles_of_nacl_formed_l1664_166468


namespace NUMINAMATH_GPT_apples_left_correct_l1664_166424

noncomputable def apples_left (initial_apples : ℝ) (additional_apples : ℝ) (apples_for_pie : ℝ) : ℝ :=
  initial_apples + additional_apples - apples_for_pie

theorem apples_left_correct :
  apples_left 10.0 5.5 4.25 = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_apples_left_correct_l1664_166424


namespace NUMINAMATH_GPT_problem_statement_l1664_166463

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

variable (f g : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_neg : ∀ x : ℝ, x < 0 → f x = x^3 - 1
axiom f_pos : ∀ x : ℝ, x > 0 → f x = g x

theorem problem_statement : f (-1) + g 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1664_166463


namespace NUMINAMATH_GPT_total_length_remaining_l1664_166407

def initial_figure_height : ℕ := 10
def initial_figure_width : ℕ := 7
def top_right_removed : ℕ := 2
def middle_left_removed : ℕ := 2
def bottom_removed : ℕ := 3
def near_top_left_removed : ℕ := 1

def remaining_top_length : ℕ := initial_figure_width - top_right_removed
def remaining_left_length : ℕ := initial_figure_height - middle_left_removed
def remaining_bottom_length : ℕ := initial_figure_width - bottom_removed
def remaining_right_length : ℕ := initial_figure_height - near_top_left_removed

theorem total_length_remaining :
  remaining_top_length + remaining_left_length + remaining_bottom_length + remaining_right_length = 26 := by
  sorry

end NUMINAMATH_GPT_total_length_remaining_l1664_166407


namespace NUMINAMATH_GPT_third_test_point_l1664_166422

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end NUMINAMATH_GPT_third_test_point_l1664_166422


namespace NUMINAMATH_GPT_smallest_n_with_digits_315_l1664_166472

-- Defining the conditions
def relatively_prime (m n : ℕ) := Nat.gcd m n = 1
def valid_fraction (m n : ℕ) := (m < n) ∧ relatively_prime m n

-- Predicate for the sequence 3, 1, 5 in the decimal representation of m/n
def contains_digits_315 (m n : ℕ) : Prop :=
  ∃ k d : ℕ, 10^k * m % n = 315 * 10^(d - 3) ∧ d ≥ 3

-- The main theorem: smallest n for which the conditions are satisfied
theorem smallest_n_with_digits_315 :
  ∃ n : ℕ, valid_fraction m n ∧ contains_digits_315 m n ∧ n = 159 :=
sorry

end NUMINAMATH_GPT_smallest_n_with_digits_315_l1664_166472


namespace NUMINAMATH_GPT_max_value_frac_sqrt_eq_sqrt_35_l1664_166451

theorem max_value_frac_sqrt_eq_sqrt_35 :
  ∀ x y : ℝ, 
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 
  ∧ (∃ x y : ℝ, x = 2 / 5 ∧ y = 6 / 5 ∧ (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_frac_sqrt_eq_sqrt_35_l1664_166451


namespace NUMINAMATH_GPT_molecular_weight_calculation_l1664_166401

def molecular_weight (n_Ar n_Si n_H n_O : ℕ) (w_Ar w_Si w_H w_O : ℝ) : ℝ :=
  n_Ar * w_Ar + n_Si * w_Si + n_H * w_H + n_O * w_O

theorem molecular_weight_calculation :
  molecular_weight 2 3 12 8 39.948 28.085 1.008 15.999 = 304.239 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_calculation_l1664_166401


namespace NUMINAMATH_GPT_frac_sum_equals_seven_eights_l1664_166489

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end NUMINAMATH_GPT_frac_sum_equals_seven_eights_l1664_166489


namespace NUMINAMATH_GPT_median_a_sq_correct_sum_of_medians_sq_l1664_166484

noncomputable def median_a_sq (a b c : ℝ) := (2 * b^2 + 2 * c^2 - a^2) / 4
noncomputable def median_b_sq (a b c : ℝ) := (2 * a^2 + 2 * c^2 - b^2) / 4
noncomputable def median_c_sq (a b c : ℝ) := (2 * a^2 + 2 * b^2 - c^2) / 4

theorem median_a_sq_correct (a b c : ℝ) : 
  median_a_sq a b c = (2 * b^2 + 2 * c^2 - a^2) / 4 :=
sorry

theorem sum_of_medians_sq (a b c : ℝ) :
  median_a_sq a b c + median_b_sq a b c + median_c_sq a b c = 
  3 * (a^2 + b^2 + c^2) / 4 :=
sorry

end NUMINAMATH_GPT_median_a_sq_correct_sum_of_medians_sq_l1664_166484


namespace NUMINAMATH_GPT_range_of_f_area_of_triangle_l1664_166427

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi / 6)

-- Problem Part (I)
theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      -1/2 ≤ f x ∧ f x ≤ 1/4) :=
sorry

-- Problem Part (II)
theorem area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ) 
  (hA0 : 0 < A ∧ A < Real.pi)
  (hS1 : a = Real.sqrt 3)
  (hS2 : b = 2 * c)
  (hF : f A = 1/4) :
  (∃ (area : ℝ), area = (1/2) * b * c * Real.sin A ∧ area = Real.sqrt 3 / 3)
:=
sorry

end NUMINAMATH_GPT_range_of_f_area_of_triangle_l1664_166427


namespace NUMINAMATH_GPT_yulgi_allowance_l1664_166495

theorem yulgi_allowance (Y G : ℕ) (h₁ : Y + G = 6000) (h₂ : (Y + G) - (Y - G) = 4800) (h₃ : Y > G) : Y = 3600 :=
sorry

end NUMINAMATH_GPT_yulgi_allowance_l1664_166495


namespace NUMINAMATH_GPT_age_of_oldest_child_l1664_166441

def average_age_of_children (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem age_of_oldest_child :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → average_age_of_children a b c d = 9 → d = 9 :=
by
  intros a b c d h_a h_b h_c h_avg
  sorry

end NUMINAMATH_GPT_age_of_oldest_child_l1664_166441


namespace NUMINAMATH_GPT_evaluate_expression_l1664_166491

theorem evaluate_expression : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1664_166491


namespace NUMINAMATH_GPT_intersection_of_sets_l1664_166434

def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

def is_acute_angle (α : ℝ) : Prop :=
  α < 90

theorem intersection_of_sets (α : ℝ) :
  (is_acute_angle α ∧ is_angle_in_first_quadrant α) ↔
  (∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90) := 
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1664_166434


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1664_166428

theorem other_root_of_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + m = 0 → x = -1) → (∀ y : ℝ, y^2 - 4 * y + m = 0 → y = 5) :=
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1664_166428


namespace NUMINAMATH_GPT_sum_boundary_values_of_range_l1664_166497

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 3 * x^2 + 6 * x)

theorem sum_boundary_values_of_range : 
  let c := 0
  let d := 1
  c + d = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_boundary_values_of_range_l1664_166497


namespace NUMINAMATH_GPT_faye_rows_l1664_166498

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h_total_pencils : total_pencils = 720)
  (h_pencils_per_row : pencils_per_row = 24) : 
  total_pencils / pencils_per_row = 30 := by 
  sorry

end NUMINAMATH_GPT_faye_rows_l1664_166498


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1664_166474

noncomputable def expr (x : ℝ) : ℝ :=
  ((x^2 + x - 2) / (x - 2) - x - 2) / ((x^2 + 4 * x + 4) / x)

theorem simplify_and_evaluate : expr 1 = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1664_166474


namespace NUMINAMATH_GPT_remainder_when_divided_by_9_l1664_166400

theorem remainder_when_divided_by_9 (x : ℕ) (h1 : x > 0) (h2 : (5 * x) % 9 = 7) : x % 9 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_9_l1664_166400


namespace NUMINAMATH_GPT_one_and_two_thirds_of_what_number_is_45_l1664_166414

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end NUMINAMATH_GPT_one_and_two_thirds_of_what_number_is_45_l1664_166414


namespace NUMINAMATH_GPT_mandy_yoga_time_l1664_166417

theorem mandy_yoga_time 
  (gym_ratio : ℕ)
  (bike_ratio : ℕ)
  (yoga_exercise_ratio : ℕ)
  (bike_time : ℕ) 
  (exercise_ratio : ℕ) 
  (yoga_ratio : ℕ)
  (h1 : gym_ratio = 2)
  (h2 : bike_ratio = 3)
  (h3 : yoga_exercise_ratio = 2)
  (h4 : exercise_ratio = 3)
  (h5 : bike_time = 18)
  (total_exercise_time : ℕ)
  (yoga_time : ℕ)
  (h6: total_exercise_time = ((gym_ratio * bike_time) / bike_ratio) + bike_time)
  (h7 : yoga_time = (yoga_exercise_ratio * total_exercise_time) / exercise_ratio) :
  yoga_time = 20 := 
by 
  sorry

end NUMINAMATH_GPT_mandy_yoga_time_l1664_166417


namespace NUMINAMATH_GPT_root_quadratic_eq_l1664_166465

theorem root_quadratic_eq (n m : ℝ) (h : n ≠ 0) (root_condition : n^2 + m * n + 3 * n = 0) : m + n = -3 :=
  sorry

end NUMINAMATH_GPT_root_quadratic_eq_l1664_166465


namespace NUMINAMATH_GPT_triangle_perimeter_triangle_side_c_l1664_166437

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) (h2 : c = 2) : 
  a + b + c = 6 := 
sorry

theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) 
(h2 : C = Real.pi / 3) (h3 : 2 * Real.sqrt 3 = (1/2) * a * b * Real.sin (Real.pi / 3)) : 
c = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_triangle_perimeter_triangle_side_c_l1664_166437


namespace NUMINAMATH_GPT_hexagon_angle_arith_prog_l1664_166412

theorem hexagon_angle_arith_prog (x d : ℝ) (hx : x > 0) (hd : d > 0) 
  (h_eq : 6 * x + 15 * d = 720) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_angle_arith_prog_l1664_166412


namespace NUMINAMATH_GPT_find_f2_g2_l1664_166499

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

theorem find_f2_g2 (f g : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : odd_function g)
  (h3 : equation f g) :
  f 2 + g 2 = -2 :=
sorry

end NUMINAMATH_GPT_find_f2_g2_l1664_166499


namespace NUMINAMATH_GPT_Tom_completes_wall_l1664_166485

theorem Tom_completes_wall :
  let avery_rate_per_hour := (1:ℝ)/3
  let tom_rate_per_hour := (1:ℝ)/2
  let combined_rate_per_hour := avery_rate_per_hour + tom_rate_per_hour
  let portion_completed_together := combined_rate_per_hour * 1 
  let remaining_wall := 1 - portion_completed_together
  let time_for_tom := remaining_wall / tom_rate_per_hour
  time_for_tom = (1:ℝ)/3 := 
by 
  sorry

end NUMINAMATH_GPT_Tom_completes_wall_l1664_166485


namespace NUMINAMATH_GPT_boxes_of_orange_crayons_l1664_166461

theorem boxes_of_orange_crayons
  (n_orange_boxes : ℕ)
  (orange_crayons_per_box : ℕ := 8)
  (blue_boxes : ℕ := 7) (blue_crayons_per_box : ℕ := 5)
  (red_boxes : ℕ := 1) (red_crayons_per_box : ℕ := 11)
  (total_crayons : ℕ := 94)
  (h_total_crayons : (n_orange_boxes * orange_crayons_per_box) + (blue_boxes * blue_crayons_per_box) + (red_boxes * red_crayons_per_box) = total_crayons):
  n_orange_boxes = 6 := 
by sorry

end NUMINAMATH_GPT_boxes_of_orange_crayons_l1664_166461


namespace NUMINAMATH_GPT_students_at_1544_l1664_166476

noncomputable def students_in_lab : Nat := 44

theorem students_at_1544 :
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8

  ∃ students : Nat,
    students = initial_students
    + (34 / enter_interval) * enter_students
    - (34 / leave_interval) * leave_students
    ∧ students = students_in_lab :=
by
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8
  use 20 + (34 / 3) * 4 - (34 / 10) * 8
  sorry

end NUMINAMATH_GPT_students_at_1544_l1664_166476


namespace NUMINAMATH_GPT_fourth_vertex_l1664_166447

-- Define the given vertices
def vertex1 := (2, 1)
def vertex2 := (4, 1)
def vertex3 := (2, 5)

-- Define what it means to be a rectangle in this context
def is_vertical_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.1 = p2.1

def is_horizontal_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.2 = p2.2

def is_rectangle (v1 v2 v3 v4: (ℕ × ℕ)) : Prop :=
  is_vertical_segment v1 v3 ∧
  is_horizontal_segment v1 v2 ∧
  is_vertical_segment v2 v4 ∧
  is_horizontal_segment v3 v4 ∧
  is_vertical_segment v1 v4 ∧ -- additional condition to ensure opposite sides are equal
  is_horizontal_segment v2 v3

-- Prove the coordinates of the fourth vertex of the rectangle
theorem fourth_vertex (v4 : ℕ × ℕ) : 
  is_rectangle vertex1 vertex2 vertex3 v4 → v4 = (4, 5) := 
by
  intro h_rect
  sorry

end NUMINAMATH_GPT_fourth_vertex_l1664_166447


namespace NUMINAMATH_GPT_find_a_l1664_166469

theorem find_a :
  ∃ a : ℝ, 
    (∀ x : ℝ, f x = 3 * x + a * x^3) ∧ 
    (f 1 = a + 3) ∧ 
    (∃ k : ℝ, k = 6 ∧ k = deriv f 1 ∧ ((∀ x : ℝ, deriv f x = 3 + 3 * a * x^2))) → 
    a = 1 :=
by sorry

end NUMINAMATH_GPT_find_a_l1664_166469


namespace NUMINAMATH_GPT_no_solution_exists_l1664_166475

theorem no_solution_exists :
  ∀ a b : ℕ, a - b = 5 ∨ b - a = 5 → a * b = 132 → false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l1664_166475


namespace NUMINAMATH_GPT_sequence_value_2016_l1664_166490

theorem sequence_value_2016 (a : ℕ → ℕ) (h₁ : a 1 = 0) (h₂ : ∀ n, a (n + 1) = a n + 2 * n) : a 2016 = 2016 * 2015 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_value_2016_l1664_166490


namespace NUMINAMATH_GPT_shadow_length_false_if_approaching_lamp_at_night_l1664_166402

theorem shadow_length_false_if_approaching_lamp_at_night
  (night : Prop)
  (approaches_lamp : Prop)
  (shadow_longer : Prop) :
  night → approaches_lamp → ¬shadow_longer :=
by
  -- assume it is night and person is approaching lamp
  intros h_night h_approaches
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_shadow_length_false_if_approaching_lamp_at_night_l1664_166402


namespace NUMINAMATH_GPT_tournament_teams_matches_l1664_166446

theorem tournament_teams_matches (teams : Fin 10 → ℕ) 
  (h : ∀ i, teams i ≤ 9) : 
  ∃ i j : Fin 10, i ≠ j ∧ teams i = teams j := 
by 
  sorry

end NUMINAMATH_GPT_tournament_teams_matches_l1664_166446


namespace NUMINAMATH_GPT_sum_le_two_of_cubics_sum_to_two_l1664_166458

theorem sum_le_two_of_cubics_sum_to_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : a + b ≤ 2 := 
sorry

end NUMINAMATH_GPT_sum_le_two_of_cubics_sum_to_two_l1664_166458


namespace NUMINAMATH_GPT_radius_increase_l1664_166478

-- Definitions and conditions
def initial_circumference : ℝ := 24
def final_circumference : ℝ := 30
def circumference_radius_relation (C : ℝ) (r : ℝ) : Prop := C = 2 * Real.pi * r

-- Required proof statement
theorem radius_increase (r1 r2 Δr : ℝ)
  (h1 : circumference_radius_relation initial_circumference r1)
  (h2 : circumference_radius_relation final_circumference r2)
  (h3 : Δr = r2 - r1) :
  Δr = 3 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_radius_increase_l1664_166478


namespace NUMINAMATH_GPT_sqrt_121_pm_11_l1664_166419

theorem sqrt_121_pm_11 :
  (∃ y : ℤ, y * y = 121) ∧ (∃ x : ℤ, x = 11 ∨ x = -11) → (∃ x : ℤ, x * x = 121 ∧ (x = 11 ∨ x = -11)) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_121_pm_11_l1664_166419


namespace NUMINAMATH_GPT_option_D_is_negative_l1664_166466

theorem option_D_is_negative :
  let A := abs (-4)
  let B := -(-4)
  let C := (-4) ^ 2
  let D := -(4 ^ 2)
  D < 0 := by
{
  -- Place sorry here since we are not required to provide the proof
  sorry
}

end NUMINAMATH_GPT_option_D_is_negative_l1664_166466


namespace NUMINAMATH_GPT_solution_is_unique_l1664_166448

noncomputable def solution (f : ℝ → ℝ) (α : ℝ) :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem solution_is_unique (f : ℝ → ℝ) (α : ℝ)
  (h : solution f α) :
  f = id ∧ α = -1 :=
sorry

end NUMINAMATH_GPT_solution_is_unique_l1664_166448


namespace NUMINAMATH_GPT_systematic_sampling_l1664_166496

theorem systematic_sampling (E P: ℕ) (a b: ℕ) (g: ℕ) 
  (hE: E = 840)
  (hP: P = 42)
  (ha: a = 61)
  (hb: b = 140)
  (hg: g = E / P)
  (hEpos: 0 < E)
  (hPpos: 0 < P)
  (hgpos: 0 < g):
  (b - a + 1) / g = 4 := 
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_l1664_166496


namespace NUMINAMATH_GPT_max_value_of_squares_l1664_166473

theorem max_value_of_squares (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  a^2 + b^2 + c^2 + d^2 ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_squares_l1664_166473


namespace NUMINAMATH_GPT_second_group_members_l1664_166409

theorem second_group_members (total first third : ℕ) (h1 : total = 70) (h2 : first = 25) (h3 : third = 15) :
  (total - first - third) = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_group_members_l1664_166409


namespace NUMINAMATH_GPT_joined_after_8_months_l1664_166405

theorem joined_after_8_months
  (investment_A investment_B : ℕ)
  (time_A time_B : ℕ)
  (profit_ratio : ℕ × ℕ)
  (h_A : investment_A = 36000)
  (h_B : investment_B = 54000)
  (h_ratio : profit_ratio = (2, 1))
  (h_time_A : time_A = 12)
  (h_eq : (investment_A * time_A) / (investment_B * time_B) = (profit_ratio.1 / profit_ratio.2)) :
  time_B = 4 := by
  sorry

end NUMINAMATH_GPT_joined_after_8_months_l1664_166405


namespace NUMINAMATH_GPT_remainder_of_125_div_j_l1664_166430

theorem remainder_of_125_div_j (j : ℕ) (h1 : j > 0) (h2 : 75 % (j^2) = 3) : 125 % j = 5 :=
sorry

end NUMINAMATH_GPT_remainder_of_125_div_j_l1664_166430


namespace NUMINAMATH_GPT_sqrt_expression_nonneg_l1664_166436

theorem sqrt_expression_nonneg {b : ℝ} : b - 3 ≥ 0 ↔ b ≥ 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_expression_nonneg_l1664_166436


namespace NUMINAMATH_GPT_quadrilaterals_property_A_false_l1664_166467

theorem quadrilaterals_property_A_false (Q A : Type → Prop) 
  (h : ¬ ∃ x, Q x ∧ A x) : ¬ ∀ x, Q x → A x :=
by
  sorry

end NUMINAMATH_GPT_quadrilaterals_property_A_false_l1664_166467
