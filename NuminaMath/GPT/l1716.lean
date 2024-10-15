import Mathlib

namespace NUMINAMATH_GPT_probability_green_ball_eq_l1716_171618

noncomputable def prob_green_ball : ℚ := 
  1 / 3 * (5 / 18) + 1 / 3 * (1 / 2) + 1 / 3 * (1 / 2)

theorem probability_green_ball_eq : 
  prob_green_ball = 23 / 54 := 
  by
  sorry

end NUMINAMATH_GPT_probability_green_ball_eq_l1716_171618


namespace NUMINAMATH_GPT_smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l1716_171602

theorem smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday :
  ∃ d : ℕ, d = 17 :=
by
  -- Assuming the starting condition that the month starts such that the second Thursday is on the 8th
  let second_thursday := 8

  -- Calculate second Monday after the second Thursday
  let second_monday := second_thursday + 4
  
  -- Calculate first Saturday after the second Monday
  let first_saturday := second_monday + 5

  have smallest_date : first_saturday = 17 := rfl
  
  exact ⟨first_saturday, smallest_date⟩

end NUMINAMATH_GPT_smallest_date_for_first_Saturday_after_second_Monday_following_second_Thursday_l1716_171602


namespace NUMINAMATH_GPT_loan_difference_eq_1896_l1716_171651

/-- 
  Samantha borrows $12,000 with two repayment schemes:
  1. A twelve-year loan with an annual interest rate of 8% compounded semi-annually. 
     At the end of 6 years, she must make a payment equal to half of what she owes, 
     and the remaining balance accrues interest until the end of 12 years.
  2. A twelve-year loan with a simple annual interest rate of 10%, paid as a lump-sum at the end.

  Prove that the positive difference between the total amounts to be paid back 
  under the two schemes is $1,896, rounded to the nearest dollar.
-/
theorem loan_difference_eq_1896 :
  let P := 12000
  let r1 := 0.08
  let r2 := 0.10
  let n := 2
  let t := 12
  let t1 := 6
  let A1 := P * (1 + r1 / n) ^ (n * t1)
  let payment_after_6_years := A1 / 2
  let remaining_balance := A1 / 2
  let compounded_remaining := remaining_balance * (1 + r1 / n) ^ (n * t1)
  let total_compound := payment_after_6_years + compounded_remaining
  let total_simple := P * (1 + r2 * t)
  (total_simple - total_compound).round = 1896 := 
by
  sorry

end NUMINAMATH_GPT_loan_difference_eq_1896_l1716_171651


namespace NUMINAMATH_GPT_true_proposition_l1716_171630

def proposition_p := ∀ (x : ℤ), x^2 > x
def proposition_q := ∃ (x : ℝ) (hx : x > 0), x + (2 / x) > 4

theorem true_proposition :
  (¬ proposition_p) ∨ proposition_q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l1716_171630


namespace NUMINAMATH_GPT_correct_calculation_of_mistake_l1716_171610

theorem correct_calculation_of_mistake (x : ℝ) (h : x - 48 = 52) : x + 48 = 148 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_of_mistake_l1716_171610


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1716_171600

theorem simplify_and_evaluate_expression : 
  ∀ x : ℝ, x = 1 → ( (x^2 - 5) / (x - 3) - 4 / (x - 3) ) = 4 :=
by
  intros x hx
  simp [hx]
  have eq : (1 * 1 - 5) = -4 := by norm_num -- Verify that the expression simplifies correctly
  sorry -- Skip the actual complex proof steps

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1716_171600


namespace NUMINAMATH_GPT_max_colors_l1716_171698

theorem max_colors (n : ℕ) (color : ℕ → ℕ → ℕ)
  (h_color_property : ∀ i j : ℕ, i < 2^n → j < 2^n → color i j = color j ((i + j) % 2^n)) :
  ∃ (c : ℕ), c ≤ 2^n ∧ (∀ i j : ℕ, i < 2^n → j < 2^n → color i j < c) :=
sorry

end NUMINAMATH_GPT_max_colors_l1716_171698


namespace NUMINAMATH_GPT_poem_lines_added_l1716_171655

theorem poem_lines_added (x : ℕ) 
  (initial_lines : ℕ)
  (months : ℕ)
  (final_lines : ℕ)
  (h_init : initial_lines = 24)
  (h_months : months = 22)
  (h_final : final_lines = 90)
  (h_equation : initial_lines + months * x = final_lines) :
  x = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_poem_lines_added_l1716_171655


namespace NUMINAMATH_GPT_equal_share_payment_l1716_171690

theorem equal_share_payment (A B C : ℝ) (h : A < B ∧ B < C) :
  (B + C - 2 * A) / 3 + (A + B - 2 * C) / 3 = ((A + B + C) * 2 / 3) - B :=
by
  sorry

end NUMINAMATH_GPT_equal_share_payment_l1716_171690


namespace NUMINAMATH_GPT_least_possible_value_l1716_171633

theorem least_possible_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 4 * x = 5 * y ∧ 5 * y = 6 * z) : x + y + z = 37 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_l1716_171633


namespace NUMINAMATH_GPT_log_function_domain_l1716_171622

theorem log_function_domain {x : ℝ} (h : 1 / x - 1 > 0) : 0 < x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_log_function_domain_l1716_171622


namespace NUMINAMATH_GPT_max_edges_partitioned_square_l1716_171687

theorem max_edges_partitioned_square (n v e : ℕ) 
  (h : v - e + n = 1) : e ≤ 3 * n + 1 := 
sorry

end NUMINAMATH_GPT_max_edges_partitioned_square_l1716_171687


namespace NUMINAMATH_GPT_Ashis_height_more_than_Babji_height_l1716_171626

-- Definitions based on conditions
variables {A B : ℝ}
-- Condition expressing the relationship between Ashis's and Babji's height
def Babji_height (A : ℝ) : ℝ := 0.80 * A

-- The proof problem to show the percentage increase
theorem Ashis_height_more_than_Babji_height :
  B = Babji_height A → (A - B) / B * 100 = 25 :=
sorry

end NUMINAMATH_GPT_Ashis_height_more_than_Babji_height_l1716_171626


namespace NUMINAMATH_GPT_cube_sum_l1716_171616

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_cube_sum_l1716_171616


namespace NUMINAMATH_GPT_monogram_count_l1716_171667

theorem monogram_count :
  ∃ (n : ℕ), n = 156 ∧
    (∃ (beforeM : Fin 13) (afterM : Fin 14),
      ∀ (a : Fin 13) (b : Fin 14),
        a < b → (beforeM = a ∧ afterM = b) → n = 12 * 13
    ) :=
by {
  sorry
}

end NUMINAMATH_GPT_monogram_count_l1716_171667


namespace NUMINAMATH_GPT_bean_seedlings_l1716_171611

theorem bean_seedlings
  (beans_per_row : ℕ)
  (pumpkins : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (radishes_per_row : ℕ)
  (rows_per_bed : ℕ) (beds : ℕ)
  (H_beans_per_row : beans_per_row = 8)
  (H_pumpkins : pumpkins = 84)
  (H_pumpkins_per_row : pumpkins_per_row = 7)
  (H_radishes : radishes = 48)
  (H_radishes_per_row : radishes_per_row = 6)
  (H_rows_per_bed : rows_per_bed = 2)
  (H_beds : beds = 14) :
  (beans_per_row * ((beds * rows_per_bed) - (pumpkins / pumpkins_per_row) - (radishes / radishes_per_row)) = 64) :=
by
  sorry

end NUMINAMATH_GPT_bean_seedlings_l1716_171611


namespace NUMINAMATH_GPT_shaded_areas_total_l1716_171623

theorem shaded_areas_total (r R : ℝ) (h_divides : ∀ (A : ℝ), ∃ (B : ℝ), B = A / 3)
  (h_center : True) (h_area : π * R^2 = 81 * π) :
  (π * R^2 / 3) + (π * (R / 2)^2 / 3) = 33.75 * π :=
by
  -- The proof here will be added.
  sorry

end NUMINAMATH_GPT_shaded_areas_total_l1716_171623


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1716_171675

theorem arithmetic_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) 
(h1 : S 3 = 2) 
(h2 : S 6 = 18) 
(h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
: S 10 / S 5 = 33 := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1716_171675


namespace NUMINAMATH_GPT_largest_sampled_number_l1716_171635

theorem largest_sampled_number (N : ℕ) (a₁ a₂ : ℕ) (k : ℕ) (H_N : N = 1500)
  (H_a₁ : a₁ = 18) (H_a₂ : a₂ = 68) (H_k : k = a₂ - a₁) :
  ∃ m, m ≤ N ∧ (m % k = 18 % k) ∧ ∀ n, (n % k = 18 % k) → n ≤ N → n ≤ m :=
by {
  -- sorry
  sorry
}

end NUMINAMATH_GPT_largest_sampled_number_l1716_171635


namespace NUMINAMATH_GPT_soccer_claim_fraction_l1716_171606

theorem soccer_claim_fraction 
  (total_students enjoy_soccer do_not_enjoy_soccer claim_do_not_enjoy honesty fraction_3_over_11 : ℕ)
  (h1 : enjoy_soccer = total_students / 2)
  (h2 : do_not_enjoy_soccer = total_students / 2)
  (h3 : claim_do_not_enjoy = enjoy_soccer * 3 / 10)
  (h4 : honesty = do_not_enjoy_soccer * 8 / 10)
  (h5 : fraction_3_over_11 = enjoy_soccer * 3 / (10 * (enjoy_soccer * 3 / 10 + do_not_enjoy_soccer * 2 / 10)))
  : fraction_3_over_11 = 3 / 11 :=
sorry

end NUMINAMATH_GPT_soccer_claim_fraction_l1716_171606


namespace NUMINAMATH_GPT_red_paint_amount_l1716_171681

theorem red_paint_amount (r w : ℕ) (hrw : r / w = 5 / 7) (hwhite : w = 21) : r = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_paint_amount_l1716_171681


namespace NUMINAMATH_GPT_smallest_n_l1716_171697

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1716_171697


namespace NUMINAMATH_GPT_part1_solution_set_a_eq_2_part2_range_of_a_l1716_171654

noncomputable def f (a x : ℝ) : ℝ := abs (x - a) + abs (2 * x - 2)

theorem part1_solution_set_a_eq_2 :
  { x : ℝ | f 2 x > 2 } = { x | x < (2 / 3) } ∪ { x | x > 2 } :=
by
  sorry

theorem part2_range_of_a :
  { a : ℝ | ∀ x : ℝ, f a x ≥ 2 } = { a | a ≤ -1 } ∪ { a | a ≥ 3 } :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_a_eq_2_part2_range_of_a_l1716_171654


namespace NUMINAMATH_GPT_contractor_total_received_l1716_171612

-- Define the conditions
def days_engaged : ℕ := 30
def daily_earnings : ℝ := 25
def fine_per_absence_day : ℝ := 7.50
def days_absent : ℕ := 4

-- Define the days worked based on conditions
def days_worked : ℕ := days_engaged - days_absent

-- Define the total earnings and total fines
def total_earnings : ℝ := days_worked * daily_earnings
def total_fines : ℝ := days_absent * fine_per_absence_day

-- Define the total amount received
def total_amount_received : ℝ := total_earnings - total_fines

-- State the theorem
theorem contractor_total_received :
  total_amount_received = 620 := 
by
  sorry

end NUMINAMATH_GPT_contractor_total_received_l1716_171612


namespace NUMINAMATH_GPT_solution_l1716_171631

-- Define the problem.
def problem (CD : ℝ) (hexagon_side : ℝ) (CY : ℝ) (BY : ℝ) : Prop :=
  CD = 2 ∧ hexagon_side = 2 ∧ CY = 4 * CD ∧ BY = 9 * Real.sqrt 2 → BY = 9 * Real.sqrt 2

theorem solution : problem 2 2 8 (9 * Real.sqrt 2) :=
by
  -- Contextualize the given conditions and directly link to the desired proof.
  intro h
  sorry

end NUMINAMATH_GPT_solution_l1716_171631


namespace NUMINAMATH_GPT_range_of_m_l1716_171641

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 - m * x + 1 > 0 → -2 < m ∧ m < 2)) ∧
  (∃ x : ℝ, x^2 < 9 - m^2) ∧
  (-3 < m ∧ m < 3) →
  ((-3 < m ∧ m ≤ -2) ∨ (2 ≤ m ∧ m < 3)) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1716_171641


namespace NUMINAMATH_GPT_part_a_part_b_l1716_171691

-- Definition for combination
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Proof problems as Lean statements
theorem part_a : combination 30 2 = 435 := by
  sorry

theorem part_b : combination 30 3 = 4060 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1716_171691


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l1716_171673

-- Proof for 238 + 45 × 5 = 463
theorem problem1 : 238 + 45 * 5 = 463 := by
  sorry

-- Proof for 65 × 4 - 128 = 132
theorem problem2 : 65 * 4 - 128 = 132 := by
  sorry

-- Proof for 900 - 108 × 4 = 468
theorem problem3 : 900 - 108 * 4 = 468 := by
  sorry

-- Proof for 369 + (512 - 215) = 666
theorem problem4 : 369 + (512 - 215) = 666 := by
  sorry

-- Proof for 758 - 58 × 9 = 236
theorem problem5 : 758 - 58 * 9 = 236 := by
  sorry

-- Proof for 105 × (81 ÷ 9 - 3) = 630
theorem problem6 : 105 * (81 / 9 - 3) = 630 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l1716_171673


namespace NUMINAMATH_GPT_sum_of_first_15_odd_positive_integers_l1716_171669

theorem sum_of_first_15_odd_positive_integers :
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  S_n = 225 :=
by
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  have : S_n = 225 := sorry
  exact this

end NUMINAMATH_GPT_sum_of_first_15_odd_positive_integers_l1716_171669


namespace NUMINAMATH_GPT_suitcase_weight_on_return_l1716_171674

def initial_weight : ℝ := 5
def perfume_count : ℝ := 5
def perfume_weight_oz : ℝ := 1.2
def chocolate_weight_lb : ℝ := 4
def soap_count : ℝ := 2
def soap_weight_oz : ℝ := 5
def jam_count : ℝ := 2
def jam_weight_oz : ℝ := 8
def oz_per_lb : ℝ := 16

theorem suitcase_weight_on_return :
  initial_weight + (perfume_count * perfume_weight_oz / oz_per_lb) + chocolate_weight_lb +
  (soap_count * soap_weight_oz / oz_per_lb) + (jam_count * jam_weight_oz / oz_per_lb) = 11 := 
  by
  sorry

end NUMINAMATH_GPT_suitcase_weight_on_return_l1716_171674


namespace NUMINAMATH_GPT_find_f_0_abs_l1716_171632

noncomputable def f : ℝ → ℝ := sorry -- f is a second-degree polynomial with real coefficients

axiom h1 : ∀ (x : ℝ), x = 1 → |f x| = 9
axiom h2 : ∀ (x : ℝ), x = 2 → |f x| = 9
axiom h3 : ∀ (x : ℝ), x = 3 → |f x| = 9

theorem find_f_0_abs : |f 0| = 9 := sorry

end NUMINAMATH_GPT_find_f_0_abs_l1716_171632


namespace NUMINAMATH_GPT_talia_mom_age_to_talia_age_ratio_l1716_171629

-- Definitions for the problem
def Talia_current_age : ℕ := 13
def Talia_mom_current_age : ℕ := 39
def Talia_father_current_age : ℕ := 36

-- These definitions match the conditions in the math problem
def condition1 : Prop := Talia_current_age + 7 = 20
def condition2 : Prop := Talia_father_current_age + 3 = Talia_mom_current_age
def condition3 : Prop := Talia_father_current_age = 36

-- The ratio calculation
def ratio := Talia_mom_current_age / Talia_current_age

-- The main theorem to prove
theorem talia_mom_age_to_talia_age_ratio :
  condition1 ∧ condition2 ∧ condition3 → ratio = 3 := by
  sorry

end NUMINAMATH_GPT_talia_mom_age_to_talia_age_ratio_l1716_171629


namespace NUMINAMATH_GPT_fill_cistern_time_l1716_171624

-- Definitions based on conditions
def rate_A : ℚ := 1 / 8
def rate_B : ℚ := 1 / 16
def rate_C : ℚ := -1 / 12

-- Combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Time to fill the cistern
def time_to_fill := 1 / combined_rate

-- Lean statement of the proof
theorem fill_cistern_time : time_to_fill = 9.6 := by
  sorry

end NUMINAMATH_GPT_fill_cistern_time_l1716_171624


namespace NUMINAMATH_GPT_relationship_M_N_l1716_171677

theorem relationship_M_N (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) 
  (M : ℝ) (hM : M = a * b) (N : ℝ) (hN : N = a + b - 1) : M > N :=
by
  sorry

end NUMINAMATH_GPT_relationship_M_N_l1716_171677


namespace NUMINAMATH_GPT_length_of_rectangular_garden_l1716_171619

-- Define the perimeter and breadth conditions
def perimeter : ℕ := 950
def breadth : ℕ := 100

-- The formula for the perimeter of a rectangle
def formula (L B : ℕ) : ℕ := 2 * (L + B)

-- State the theorem
theorem length_of_rectangular_garden (L : ℕ) 
  (h1 : perimeter = 2 * (L + breadth)) : 
  L = 375 := 
by
  sorry

end NUMINAMATH_GPT_length_of_rectangular_garden_l1716_171619


namespace NUMINAMATH_GPT_division_decomposition_l1716_171682

theorem division_decomposition (a b : ℕ) (h₁ : a = 36) (h₂ : b = 3)
    (h₃ : 30 / b = 10) (h₄ : 6 / b = 2) (h₅ : 10 + 2 = 12) :
    a / b = (30 / b) + (6 / b) := 
sorry

end NUMINAMATH_GPT_division_decomposition_l1716_171682


namespace NUMINAMATH_GPT_original_mixture_litres_l1716_171660

theorem original_mixture_litres 
  (x : ℝ)
  (h1 : 0.20 * x = 0.15 * (x + 5)) :
  x = 15 :=
sorry

end NUMINAMATH_GPT_original_mixture_litres_l1716_171660


namespace NUMINAMATH_GPT_inverse_110_mod_667_l1716_171609

theorem inverse_110_mod_667 :
  (∃ (a b c : ℕ), a = 65 ∧ b = 156 ∧ c = 169 ∧ c^2 = a^2 + b^2) →
  (∃ n : ℕ, 110 * n % 667 = 1 ∧ 0 ≤ n ∧ n < 667 ∧ n = 608) :=
by
  sorry

end NUMINAMATH_GPT_inverse_110_mod_667_l1716_171609


namespace NUMINAMATH_GPT_phase_shift_sin_l1716_171652

theorem phase_shift_sin (x : ℝ) : 
  let B := 4
  let C := - (π / 2)
  let φ := - C / B
  φ = π / 8 := 
by 
  sorry

end NUMINAMATH_GPT_phase_shift_sin_l1716_171652


namespace NUMINAMATH_GPT_exist_n_exactly_3_rainy_days_l1716_171644

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the binomial probability
def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exist_n_exactly_3_rainy_days (p : ℝ) (k : ℕ) (prob : ℝ) :
  p = 0.5 → k = 3 → prob = 0.25 →
  ∃ n : ℕ, binomial_prob n k p = prob :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_exist_n_exactly_3_rainy_days_l1716_171644


namespace NUMINAMATH_GPT_estimated_germination_probability_stable_l1716_171603

structure ExperimentData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations

def experimentalData : List ExperimentData := [
  ⟨50, 47⟩,
  ⟨100, 89⟩,
  ⟨200, 188⟩,
  ⟨500, 461⟩,
  ⟨1000, 892⟩,
  ⟨2000, 1826⟩,
  ⟨3000, 2733⟩
]

def germinationFrequency (data : ExperimentData) : ℚ :=
  data.m / data.n

def closeTo (x y : ℚ) (ε : ℚ) : Prop :=
  |x - y| < ε

theorem estimated_germination_probability_stable :
  ∃ ε > 0, ∀ data ∈ experimentalData, closeTo (germinationFrequency data) 0.91 ε :=
by
  sorry

end NUMINAMATH_GPT_estimated_germination_probability_stable_l1716_171603


namespace NUMINAMATH_GPT_intersection_a_four_range_of_a_l1716_171695

variable {x a : ℝ}

-- Problem 1: Intersection of A and B for a = 4
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a^2 + 2}

theorem intersection_a_four : A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} := 
by  sorry

-- Problem 2: Range of a given condition
theorem range_of_a (a : ℝ) (h1 : a > -3/2) (h2 : ∀ x ∈ A a, x ∈ B a) : 1 ≤ a ∧ a ≤ 3 := 
by  sorry

end NUMINAMATH_GPT_intersection_a_four_range_of_a_l1716_171695


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1716_171676

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x < 5) : (x < 2 → x < 5) ∧ ¬(x < 5 → x < 2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1716_171676


namespace NUMINAMATH_GPT_smallest_digit_N_divisible_by_6_l1716_171613

theorem smallest_digit_N_divisible_by_6 : 
  ∃ N : ℕ, N < 10 ∧ 
          (14530 + N) % 6 = 0 ∧
          ∀ M : ℕ, M < N → (14530 + M) % 6 ≠ 0 := sorry

end NUMINAMATH_GPT_smallest_digit_N_divisible_by_6_l1716_171613


namespace NUMINAMATH_GPT_ratio_of_areas_l1716_171653

theorem ratio_of_areas (C1 C2 : ℝ) (h1 : (60 : ℝ) / 360 * C1 = (48 : ℝ) / 360 * C2) : 
  (C1 / C2) ^ 2 = 16 / 25 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1716_171653


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1716_171638

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h1 : (a 1 - 3) ^ 3 + 3 * (a 1 - 3) = -3)
  (h12 : (a 12 - 3) ^ 3 + 3 * (a 12 - 3) = 3) :
  a 1 < a 12 ∧ (12 * (a 1 + a 12)) / 2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1716_171638


namespace NUMINAMATH_GPT_aubrey_distance_from_school_l1716_171620

-- Define average speed and travel time
def average_speed : ℝ := 22 -- in miles per hour
def travel_time : ℝ := 4 -- in hours

-- Define the distance function
def calc_distance (speed time : ℝ) : ℝ := speed * time

-- State the theorem
theorem aubrey_distance_from_school : calc_distance average_speed travel_time = 88 := 
by
  sorry

end NUMINAMATH_GPT_aubrey_distance_from_school_l1716_171620


namespace NUMINAMATH_GPT_floor_S_value_l1716_171696

noncomputable def floor_S (a b c d : ℝ) : ℝ :=
  a + b + c + d

theorem floor_S_value (a b c d : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_sum_sq : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (h_product : a * c = 1008 ∧ b * d = 1008) :
  ⌊floor_S a b c d⌋ = 117 :=
by
  sorry

end NUMINAMATH_GPT_floor_S_value_l1716_171696


namespace NUMINAMATH_GPT_greatest_integer_less_than_PS_l1716_171648

noncomputable def PS := (150 * Real.sqrt 2)

theorem greatest_integer_less_than_PS
  (PQ RS : ℝ)
  (PS : ℝ := PQ * Real.sqrt 2)
  (h₁ : PQ = 150)
  (h_midpoint : PS / 2 = PQ) :
  ∀ n : ℤ, n < PS → n = 212 :=
by
  -- Proof to be completed later
  sorry

end NUMINAMATH_GPT_greatest_integer_less_than_PS_l1716_171648


namespace NUMINAMATH_GPT_least_number_to_subtract_from_724946_l1716_171605

def divisible_by_10 (n : ℕ) : Prop :=
  n % 10 = 0

theorem least_number_to_subtract_from_724946 :
  ∃ k : ℕ, k = 6 ∧ divisible_by_10 (724946 - k) :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_from_724946_l1716_171605


namespace NUMINAMATH_GPT_gcd_10010_15015_l1716_171659

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end NUMINAMATH_GPT_gcd_10010_15015_l1716_171659


namespace NUMINAMATH_GPT_basin_capacity_l1716_171656

-- Defining the flow rate of water into the basin
def inflow_rate : ℕ := 24

-- Defining the leak rate of the basin
def leak_rate : ℕ := 4

-- Defining the time taken to fill the basin in seconds
def fill_time : ℕ := 13

-- Net rate of filling the basin
def net_rate : ℕ := inflow_rate - leak_rate

-- Volume of the basin
def basin_volume : ℕ := net_rate * fill_time

-- The goal is to prove that the volume of the basin is 260 gallons
theorem basin_capacity : basin_volume = 260 := by
  sorry

end NUMINAMATH_GPT_basin_capacity_l1716_171656


namespace NUMINAMATH_GPT_range_of_alpha_minus_beta_l1716_171645

open Real

theorem range_of_alpha_minus_beta (
    α β : ℝ) 
    (h1 : -π / 2 < α) 
    (h2 : α < 0)
    (h3 : 0 < β)
    (h4 : β < π / 3)
  : -5 * π / 6 < α - β ∧ α - β < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_alpha_minus_beta_l1716_171645


namespace NUMINAMATH_GPT_village_population_l1716_171646

theorem village_population (P : ℝ) (h1 : 0.08 * P = 4554) : P = 6325 :=
by
  sorry

end NUMINAMATH_GPT_village_population_l1716_171646


namespace NUMINAMATH_GPT_sample_capacity_l1716_171678

theorem sample_capacity (f : ℕ) (r : ℚ) (n : ℕ) (h₁ : f = 40) (h₂ : r = 0.125) (h₃ : r * n = f) : n = 320 :=
sorry

end NUMINAMATH_GPT_sample_capacity_l1716_171678


namespace NUMINAMATH_GPT_guessing_game_l1716_171679

-- Define the conditions
def number : ℕ := 33
def result : ℕ := 2 * 51 - 3

-- Define the factor (to be proven)
def factor (n r : ℕ) : ℕ := r / n

-- The theorem to be proven
theorem guessing_game (n r : ℕ) (h1 : n = 33) (h2 : r = 2 * 51 - 3) : 
  factor n r = 3 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_guessing_game_l1716_171679


namespace NUMINAMATH_GPT_find_x_l1716_171614

theorem find_x (y : ℝ) (x : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  x = (-19 + Real.sqrt 329) / 16 ∨ x = (-19 - Real.sqrt 329) / 16 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1716_171614


namespace NUMINAMATH_GPT_total_seeds_planted_l1716_171692

theorem total_seeds_planted 
    (seeds_per_bed : ℕ) 
    (seeds_grow_per_bed : ℕ) 
    (total_flowers : ℕ) 
    (h1 : seeds_per_bed = 15) 
    (h2 : seeds_grow_per_bed = 60) 
    (h3 : total_flowers = 220) : 
    ∃ (total_seeds : ℕ), total_seeds = 85 := 
by
    sorry

end NUMINAMATH_GPT_total_seeds_planted_l1716_171692


namespace NUMINAMATH_GPT_max_plates_l1716_171693

def cost_pan : ℕ := 3
def cost_pot : ℕ := 5
def cost_plate : ℕ := 11
def total_cost : ℕ := 100
def min_pans : ℕ := 2
def min_pots : ℕ := 2

theorem max_plates (p q r : ℕ) :
  p >= min_pans → q >= min_pots → (cost_pan * p + cost_pot * q + cost_plate * r = total_cost) → r = 7 :=
by
  intros h_p h_q h_cost
  sorry

end NUMINAMATH_GPT_max_plates_l1716_171693


namespace NUMINAMATH_GPT_coordinate_sum_condition_l1716_171636

open Function

theorem coordinate_sum_condition :
  (∃ (g : ℝ → ℝ), g 6 = 5 ∧
    (∃ y : ℝ, 4 * y = g (3 * 2) + 4 ∧ y = 9 / 4 ∧ 2 + y = 17 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_coordinate_sum_condition_l1716_171636


namespace NUMINAMATH_GPT_intersection_points_count_l1716_171684

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := |3 * x + 6|
def f2 (x : ℝ) : ℝ := -|4 * x - 4|

-- Prove the number of intersection points is 2
theorem intersection_points_count : 
  (∃ x1 y1, (f1 x1 = y1) ∧ (f2 x1 = y1)) ∧ 
  (∃ x2 y2, (f1 x2 = y2) ∧ (f2 x2 = y2) ∧ x1 ≠ x2) :=
sorry

end NUMINAMATH_GPT_intersection_points_count_l1716_171684


namespace NUMINAMATH_GPT_garden_length_l1716_171621

def PerimeterLength (P : ℕ) (length : ℕ) (breadth : ℕ) : Prop :=
  P = 2 * (length + breadth)

theorem garden_length
  (P : ℕ)
  (breadth : ℕ)
  (h1 : P = 480)
  (h2 : breadth = 100):
  ∃ length : ℕ, PerimeterLength P length breadth ∧ length = 140 :=
by
  use 140
  sorry

end NUMINAMATH_GPT_garden_length_l1716_171621


namespace NUMINAMATH_GPT_unique_solution_l1716_171699

theorem unique_solution:
  ∃! (x y z : ℕ), 2^x + 9 * 7^y = z^3 ∧ x = 0 ∧ y = 1 ∧ z = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l1716_171699


namespace NUMINAMATH_GPT_scientific_notation_320000_l1716_171634

theorem scientific_notation_320000 : 320000 = 3.2 * 10^5 :=
  by sorry

end NUMINAMATH_GPT_scientific_notation_320000_l1716_171634


namespace NUMINAMATH_GPT_wrongly_copied_value_l1716_171628

theorem wrongly_copied_value (mean_initial mean_correct : ℕ) (n : ℕ) 
  (wrong_copied_value : ℕ) (total_sum_initial total_sum_correct : ℕ) : 
  (mean_initial = 150) ∧ (mean_correct = 151) ∧ (n = 30) ∧ 
  (wrong_copied_value = 135) ∧ (total_sum_initial = n * mean_initial) ∧ 
  (total_sum_correct = n * mean_correct) → 
  (total_sum_correct - (total_sum_initial - wrong_copied_value) + wrong_copied_value = 300) :=
by
  intros h
  have h1 : mean_initial = 150 := by sorry
  have h2 : mean_correct = 151 := by sorry
  have h3 : n = 30 := by sorry
  have h4 : wrong_copied_value = 135 := by sorry
  have h5 : total_sum_initial = n * mean_initial := by sorry
  have h6 : total_sum_correct = n * mean_correct := by sorry
  sorry -- This is where the proof would go, but is not required per instructions.

end NUMINAMATH_GPT_wrongly_copied_value_l1716_171628


namespace NUMINAMATH_GPT_min_distance_ellipse_line_l1716_171608

theorem min_distance_ellipse_line :
  let ellipse (x y : ℝ) := (x ^ 2) / 16 + (y ^ 2) / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
             (∀ (x y : ℝ), ellipse x y → ∃ (d' : ℝ), line x y → d' ≥ d) :=
  sorry

end NUMINAMATH_GPT_min_distance_ellipse_line_l1716_171608


namespace NUMINAMATH_GPT_eggs_town_hall_l1716_171670

-- Definitions of given conditions
def eggs_club_house : ℕ := 40
def eggs_park : ℕ := 25
def total_eggs_found : ℕ := 80

-- Problem statement
theorem eggs_town_hall : total_eggs_found - (eggs_club_house + eggs_park) = 15 := by
  sorry

end NUMINAMATH_GPT_eggs_town_hall_l1716_171670


namespace NUMINAMATH_GPT_serena_mother_age_l1716_171604

theorem serena_mother_age {x : ℕ} (h : 39 + x = 3 * (9 + x)) : x = 6 := 
by
  sorry

end NUMINAMATH_GPT_serena_mother_age_l1716_171604


namespace NUMINAMATH_GPT_scientific_notation_of_1_656_million_l1716_171662

theorem scientific_notation_of_1_656_million :
  (1.656 * 10^6 = 1656000) := by
sorry

end NUMINAMATH_GPT_scientific_notation_of_1_656_million_l1716_171662


namespace NUMINAMATH_GPT_different_suits_card_combinations_l1716_171637

theorem different_suits_card_combinations :
  let num_suits := 4
  let suit_cards := 13
  let choose_suits := Nat.choose 4 4
  let ways_per_suit := suit_cards ^ num_suits
  choose_suits * ways_per_suit = 28561 :=
  sorry

end NUMINAMATH_GPT_different_suits_card_combinations_l1716_171637


namespace NUMINAMATH_GPT_initial_crayons_per_box_l1716_171683

-- Define the initial total number of crayons in terms of x
def total_initial_crayons (x : ℕ) : ℕ := 4 * x

-- Define the crayons given to Mae
def crayons_to_Mae : ℕ := 5

-- Define the crayons given to Lea
def crayons_to_Lea : ℕ := 12

-- Define the remaining crayons
def remaining_crayons : ℕ := 15

-- Prove that the initial number of crayons per box is 8 given the conditions
theorem initial_crayons_per_box (x : ℕ) : total_initial_crayons x - crayons_to_Mae - crayons_to_Lea = remaining_crayons → x = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_initial_crayons_per_box_l1716_171683


namespace NUMINAMATH_GPT_wine_cost_is_3_60_l1716_171650

noncomputable def appetizer_cost : ℕ := 8
noncomputable def steak_cost : ℕ := 20
noncomputable def dessert_cost : ℕ := 6
noncomputable def total_spent : ℝ := 38
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def number_of_wines : ℕ := 2

noncomputable def discounted_steak_cost : ℝ := steak_cost / 2
noncomputable def full_meal_cost : ℝ := appetizer_cost + steak_cost + dessert_cost
noncomputable def meal_cost_after_discount : ℝ := appetizer_cost + discounted_steak_cost + dessert_cost
noncomputable def full_meal_tip := tip_percentage * full_meal_cost
noncomputable def meal_cost_with_tip := meal_cost_after_discount + full_meal_tip
noncomputable def total_wine_cost := total_spent - meal_cost_with_tip
noncomputable def cost_per_wine := total_wine_cost / number_of_wines

theorem wine_cost_is_3_60 : cost_per_wine = 3.60 := by
  sorry

end NUMINAMATH_GPT_wine_cost_is_3_60_l1716_171650


namespace NUMINAMATH_GPT_floor_length_l1716_171686

theorem floor_length (b l : ℝ)
  (h1 : l = 3 * b)
  (h2 : 3 * b^2 = 484 / 3) :
  l = 22 := 
sorry

end NUMINAMATH_GPT_floor_length_l1716_171686


namespace NUMINAMATH_GPT_count_three_digit_with_f_l1716_171671

open Nat

def f : ℕ → ℕ := sorry 

axiom f_add_add (a b : ℕ) : f (a + b) = f (f a + b)
axiom f_add_small (a b : ℕ) (h : a + b < 10) : f (a + b) = f a + f b
axiom f_10 : f 10 = 1

theorem count_three_digit_with_f (hN : ∀ n : ℕ, f 2^(3^(4^5)) = f n):
  ∃ k, k = 100 ∧ ∀ n, 100 ≤ n ∧ n < 1000 → (f n = f 2^(3^(4^5))) :=
sorry

end NUMINAMATH_GPT_count_three_digit_with_f_l1716_171671


namespace NUMINAMATH_GPT_value_of_expression_l1716_171689

theorem value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) : 12 * a - 6 * b + 3 * c - 2 * d = 40 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1716_171689


namespace NUMINAMATH_GPT_length_cd_l1716_171647

noncomputable def isosceles_triangle (A B E : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ) :
  ℝ := sorry

theorem length_cd (A B E C D : Type*) (area_abe : ℝ) (trapezoid_area : ℝ) (altitude_abe : ℝ)
  (h1 : area_abe = 144) (h2 : trapezoid_area = 108) (h3 : altitude_abe = 24) :
  isosceles_triangle A B E area_abe trapezoid_area altitude_abe = 6 := by
  sorry

end NUMINAMATH_GPT_length_cd_l1716_171647


namespace NUMINAMATH_GPT_dogwood_trees_after_5_years_l1716_171685

theorem dogwood_trees_after_5_years :
  let current_trees := 39
  let trees_planted_today := 41
  let growth_rate_today := 2 -- trees per year
  let trees_planted_tomorrow := 20
  let growth_rate_tomorrow := 4 -- trees per year
  let years := 5
  let total_planted_trees := trees_planted_today + trees_planted_tomorrow
  let total_initial_trees := current_trees + total_planted_trees
  let total_growth_today := growth_rate_today * years
  let total_growth_tomorrow := growth_rate_tomorrow * years
  let total_growth := total_growth_today + total_growth_tomorrow
  let final_tree_count := total_initial_trees + total_growth
  final_tree_count = 130 := by
  sorry

end NUMINAMATH_GPT_dogwood_trees_after_5_years_l1716_171685


namespace NUMINAMATH_GPT_greatest_prime_factor_5pow8_plus_10pow7_l1716_171625

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem greatest_prime_factor_5pow8_plus_10pow7 : greatest_prime_factor (5^8 + 10^7) = 19 := by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_5pow8_plus_10pow7_l1716_171625


namespace NUMINAMATH_GPT_a_75_eq_24_l1716_171640

variable {a : ℕ → ℤ}

-- Conditions for the problem
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def a_15_eq_8 : a 15 = 8 := sorry

def a_60_eq_20 : a 60 = 20 := sorry

-- The theorem we want to prove
theorem a_75_eq_24 (d : ℤ) (h_seq : is_arithmetic_sequence a d) (h15 : a 15 = 8) (h60 : a 60 = 20) : a 75 = 24 :=
  by
    sorry

end NUMINAMATH_GPT_a_75_eq_24_l1716_171640


namespace NUMINAMATH_GPT_divides_prime_factors_l1716_171672

theorem divides_prime_factors (a b : ℕ) (p : ℕ → ℕ → Prop) (k l : ℕ → ℕ) (n : ℕ) : 
  (a ∣ b) ↔ (∀ i : ℕ, i < n → k i ≤ l i) :=
by
  sorry

end NUMINAMATH_GPT_divides_prime_factors_l1716_171672


namespace NUMINAMATH_GPT_geometric_sequence_a3_l1716_171661

noncomputable def a_1 (S_4 : ℕ) (q : ℕ) : ℕ :=
  S_4 * (q - 1) / (1 - q^4)

noncomputable def a_3 (a_1 : ℕ) (q : ℕ) : ℕ :=
  a_1 * q^(3 - 1)

theorem geometric_sequence_a3 (a_n : ℕ → ℕ) (S_4 : ℕ) (q : ℕ) :
  (q = 2) →
  (S_4 = 60) →
  a_3 (a_1 S_4 q) q = 16 :=
by
  intro hq hS4
  rw [hq, hS4]
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l1716_171661


namespace NUMINAMATH_GPT_words_per_page_l1716_171639

/-- 
  Let p denote the number of words per page.
  Given conditions:
  - A book contains 154 pages.
  - Each page has the same number of words, p, and no page contains more than 120 words.
  - The total number of words in the book (154p) is congruent to 250 modulo 227.
  Prove that the number of words in each page p is congruent to 49 modulo 227.
 -/
theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p ≡ 250 [MOD 227]) : p ≡ 49 [MOD 227] :=
sorry

end NUMINAMATH_GPT_words_per_page_l1716_171639


namespace NUMINAMATH_GPT_range_of_m_l1716_171680

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) (h : f (2 * m - 1) + f (3 - m) > 0) : m > -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1716_171680


namespace NUMINAMATH_GPT_unique_real_x_satisfies_eq_l1716_171668

theorem unique_real_x_satisfies_eq (x : ℝ) (h : x ≠ 0) : (7 * x)^5 = (14 * x)^4 ↔ x = 16 / 7 :=
by sorry

end NUMINAMATH_GPT_unique_real_x_satisfies_eq_l1716_171668


namespace NUMINAMATH_GPT_simplified_expr_l1716_171694

theorem simplified_expr : 
  (Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2) ^ 2) = (8 + 2 * Real.sqrt 2) := 
by 
  sorry

end NUMINAMATH_GPT_simplified_expr_l1716_171694


namespace NUMINAMATH_GPT_gumballs_remaining_l1716_171664

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end NUMINAMATH_GPT_gumballs_remaining_l1716_171664


namespace NUMINAMATH_GPT_interval_where_f_decreasing_minimum_value_of_a_l1716_171642

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x
noncomputable def h (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * a * x - 1

theorem interval_where_f_decreasing :
  {x : ℝ | 1 < x} = {x : ℝ | deriv f x < 0} :=
by sorry

theorem minimum_value_of_a (a : ℤ) (ha : ∀ x : ℝ, 0 < x → (a - 1) * x^2 + 2 * a * x - 1 ≥ log x - x^2 + x) :
  a ≥ 1 :=
by sorry

end NUMINAMATH_GPT_interval_where_f_decreasing_minimum_value_of_a_l1716_171642


namespace NUMINAMATH_GPT_rose_clothing_tax_l1716_171615

theorem rose_clothing_tax {total_spent total_tax tax_other tax_clothing amount_clothing amount_food amount_other clothing_tax_rate : ℝ} 
  (h_total_spent : total_spent = 100)
  (h_amount_clothing : amount_clothing = 0.5 * total_spent)
  (h_amount_food : amount_food = 0.2 * total_spent)
  (h_amount_other : amount_other = 0.3 * total_spent)
  (h_no_tax_food : True)
  (h_tax_other_rate : tax_other = 0.08 * amount_other)
  (h_total_tax_rate : total_tax = 0.044 * total_spent)
  (h_calculate_tax_clothing : tax_clothing = total_tax - tax_other) :
  clothing_tax_rate = (tax_clothing / amount_clothing) * 100 → 
  clothing_tax_rate = 4 := 
by
  sorry

end NUMINAMATH_GPT_rose_clothing_tax_l1716_171615


namespace NUMINAMATH_GPT_solve_x_l1716_171688

theorem solve_x (x : ℝ) (hx : (1/x + 1/(2*x) + 1/(3*x) = 1/12)) : x = 22 :=
  sorry

end NUMINAMATH_GPT_solve_x_l1716_171688


namespace NUMINAMATH_GPT_meaningful_expression_range_l1716_171665

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = (1 / (Real.sqrt (x - 2)))) ↔ (x > 2) := 
sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1716_171665


namespace NUMINAMATH_GPT_summer_sales_is_2_million_l1716_171617

def spring_sales : ℝ := 4.8
def autumn_sales : ℝ := 7
def winter_sales : ℝ := 2.2
def spring_percentage : ℝ := 0.3

theorem summer_sales_is_2_million :
  ∃ (total_sales : ℝ), total_sales = (spring_sales / spring_percentage) ∧
  ∃ summer_sales : ℝ, total_sales = spring_sales + summer_sales + autumn_sales + winter_sales ∧
  summer_sales = 2 :=
by
  sorry

end NUMINAMATH_GPT_summer_sales_is_2_million_l1716_171617


namespace NUMINAMATH_GPT_abs_neg_product_eq_product_l1716_171663

variable (a b : ℝ)

theorem abs_neg_product_eq_product (h1 : a < 0) (h2 : 0 < b) : |-a * b| = a * b := by
  sorry

end NUMINAMATH_GPT_abs_neg_product_eq_product_l1716_171663


namespace NUMINAMATH_GPT_planes_perpendicular_l1716_171601

variables {m n : Type} -- lines
variables {α β : Type} -- planes

axiom lines_different : m ≠ n
axiom planes_different : α ≠ β
axiom parallel_lines : ∀ (m n : Type), Prop -- m ∥ n
axiom parallel_plane_line : ∀ (m α : Type), Prop -- m ∥ α
axiom perp_plane_line : ∀ (n β : Type), Prop -- n ⊥ β
axiom perp_planes : ∀ (α β : Type), Prop -- α ⊥ β

theorem planes_perpendicular 
  (h1 : parallel_lines m n) 
  (h2 : parallel_plane_line m α) 
  (h3 : perp_plane_line n β) 
: perp_planes α β := 
sorry

end NUMINAMATH_GPT_planes_perpendicular_l1716_171601


namespace NUMINAMATH_GPT_walk_time_is_correct_l1716_171657

noncomputable def time_to_walk_one_block := 
  let blocks := 18
  let bike_time_per_block := 20 -- seconds
  let additional_walk_time := 12 * 60 -- 12 minutes in seconds
  let walk_time := blocks * bike_time_per_block + additional_walk_time
  walk_time / blocks

theorem walk_time_is_correct : 
  let W := time_to_walk_one_block
  W = 60 := by
    sorry -- proof goes here

end NUMINAMATH_GPT_walk_time_is_correct_l1716_171657


namespace NUMINAMATH_GPT_eval_expression_l1716_171666

theorem eval_expression : (2: ℤ)^2 - 3 * (2: ℤ) + 2 = 0 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1716_171666


namespace NUMINAMATH_GPT_problem_1_problem_2_l1716_171658

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -4 ≤ x ∧ x < 2 }
def B : Set ℝ := { x | -1 < x ∧ x ≤ 3 }
def P : Set ℝ := { x | x ≤ 0 ∨ x ≥ 5 / 2 }

theorem problem_1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem problem_2 : (U \ B) ∪ P = { x | x ≤ 0 ∨ x ≥ 5 / 2 } :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1716_171658


namespace NUMINAMATH_GPT_gerry_bananas_eaten_l1716_171607

theorem gerry_bananas_eaten (b : ℝ) : 
  (b + (b + 8) + (b + 16) + 0 + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 220) →
  b + 48 = 56.67 :=
by
  sorry

end NUMINAMATH_GPT_gerry_bananas_eaten_l1716_171607


namespace NUMINAMATH_GPT_maximize_profits_l1716_171643

variable (m : ℝ) (x : ℝ)

def w1 (m x : ℝ) := (8 - m) * x - 30
def w2 (x : ℝ) := -0.01 * x^2 + 8 * x - 80

theorem maximize_profits : 
  (4 ≤ m ∧ m < 5.1 → ∀ x, 0 ≤ x ∧ x ≤ 500 → w1 m x ≥ w2 x) ∧
  (m = 5.1 → ∀ x ≤ 300, w1 m 500 = w2 300) ∧
  (m > 5.1 ∧ m ≤ 6 → ∀ x, 0 ≤ x ∧ x ≤ 300 → w2 x ≥ w1 m x) :=
  sorry

end NUMINAMATH_GPT_maximize_profits_l1716_171643


namespace NUMINAMATH_GPT_raw_materials_amount_true_l1716_171649

def machinery_cost : ℝ := 2000
def total_amount : ℝ := 5555.56
def cash (T : ℝ) : ℝ := 0.10 * T
def raw_materials_cost (T : ℝ) : ℝ := T - machinery_cost - cash T

theorem raw_materials_amount_true :
  raw_materials_cost total_amount = 3000 := 
  by
  sorry

end NUMINAMATH_GPT_raw_materials_amount_true_l1716_171649


namespace NUMINAMATH_GPT_factorization_correct_l1716_171627

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorization_correct_l1716_171627
