import Mathlib

namespace NUMINAMATH_GPT_unique_solution_l15_1579

theorem unique_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a * b - a - b = 1) : (a, b) = (3, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l15_1579


namespace NUMINAMATH_GPT_gcd_36_54_l15_1551

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_GPT_gcd_36_54_l15_1551


namespace NUMINAMATH_GPT_intersecting_lines_l15_1568

theorem intersecting_lines (p : ℝ) :
    (∃ x y : ℝ, y = 3 * x - 6 ∧ y = -4 * x + 8 ∧ y = 7 * x + p) ↔ p = -14 :=
by {
    sorry
}

end NUMINAMATH_GPT_intersecting_lines_l15_1568


namespace NUMINAMATH_GPT_inequality_proof_l15_1562

theorem inequality_proof
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l15_1562


namespace NUMINAMATH_GPT_students_not_made_the_cut_l15_1566

-- Define the constants for the number of girls, boys, and students called back
def girls := 17
def boys := 32
def called_back := 10

-- Total number of students trying out for the team
def total_try_out := girls + boys

-- Number of students who didn't make the cut
def not_made_the_cut := total_try_out - called_back

-- The theorem to be proved
theorem students_not_made_the_cut : not_made_the_cut = 39 := by
  -- Adding the proof is not required, so we use sorry
  sorry

end NUMINAMATH_GPT_students_not_made_the_cut_l15_1566


namespace NUMINAMATH_GPT_find_m_range_l15_1528

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2 * (m - 1) * x + 2 

theorem find_m_range (m : ℝ) : (∀ x ≤ 4, f x m ≤ f (x + 1) m) → m ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l15_1528


namespace NUMINAMATH_GPT_point_distance_l15_1585

theorem point_distance (x : ℤ) : abs x = 2021 → (x = 2021 ∨ x = -2021) := 
sorry

end NUMINAMATH_GPT_point_distance_l15_1585


namespace NUMINAMATH_GPT_part1_part2_l15_1569

-- Define the system of equations
def system_eq (x y k : ℝ) : Prop := 
  3 * x + y = k + 1 ∧ x + 3 * y = 3

-- Part (1): x and y are opposite in sign implies k = -4
theorem part1 (x y k : ℝ) (h_eq : system_eq x y k) (h_sign : x * y < 0) : k = -4 := by
  sorry

-- Part (2): range of values for k given extra inequalities
theorem part2 (x y k : ℝ) (h_eq : system_eq x y k) 
  (h_ineq1 : x + y < 3) (h_ineq2 : x - y > 1) : 4 < k ∧ k < 8 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l15_1569


namespace NUMINAMATH_GPT_number_in_eighth_group_l15_1578

theorem number_in_eighth_group (employees groups n l group_size numbering_drawn starting_number: ℕ) 
(h1: employees = 200) 
(h2: groups = 40) 
(h3: n = 5) 
(h4: number_in_fifth_group = 23) 
(h5: starting_number + 4 * n = number_in_fifth_group) : 
  starting_number + 7 * n = 38 :=
by
  sorry

end NUMINAMATH_GPT_number_in_eighth_group_l15_1578


namespace NUMINAMATH_GPT_product_equals_one_l15_1510

theorem product_equals_one (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 / (1 + x + x^2)) + (1 / (1 + y + y^2)) + (1 / (1 + x + y)) = 1) : 
  x * y = 1 :=
by
  sorry

end NUMINAMATH_GPT_product_equals_one_l15_1510


namespace NUMINAMATH_GPT_range_of_a_l15_1582

-- Definitions of the propositions in Lean terms
def proposition_p (a : ℝ) := 
  ∃ x : ℝ, x ∈ [-1, 1] ∧ x^2 - (2 + a) * x + 2 * a = 0

def proposition_q (a : ℝ) := 
  ∃ x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The main theorem to prove that the range of values for a is [-1, 0]
theorem range_of_a {a : ℝ} (h : proposition_p a ∧ proposition_q a) : 
  -1 ≤ a ∧ a ≤ 0 := sorry

end NUMINAMATH_GPT_range_of_a_l15_1582


namespace NUMINAMATH_GPT_james_twitch_income_l15_1577

theorem james_twitch_income :
  let tier1_base := 120
  let tier2_base := 50
  let tier3_base := 30
  let tier1_gifted := 10
  let tier2_gifted := 25
  let tier3_gifted := 15
  let tier1_new := tier1_base + tier1_gifted
  let tier2_new := tier2_base + tier2_gifted
  let tier3_new := tier3_base + tier3_gifted
  let tier1_income := tier1_new * 4.99
  let tier2_income := tier2_new * 9.99
  let tier3_income := tier3_new * 24.99
  let total_income := tier1_income + tier2_income + tier3_income
  total_income = 2522.50 :=
by
  sorry

end NUMINAMATH_GPT_james_twitch_income_l15_1577


namespace NUMINAMATH_GPT_factor_polynomial_l15_1521

theorem factor_polynomial :
  ∀ u : ℝ, (u^4 - 81 * u^2 + 144) = (u^2 - 72) * (u - 3) * (u + 3) :=
by
  intro u
  -- Establish the polynomial and its factorization in Lean
  have h : u^4 - 81 * u^2 + 144 = (u^2 - 72) * (u - 3) * (u + 3) := sorry
  exact h

end NUMINAMATH_GPT_factor_polynomial_l15_1521


namespace NUMINAMATH_GPT_number_leaves_remainder_five_l15_1529

theorem number_leaves_remainder_five (k : ℕ) (n : ℕ) (least_num : ℕ) 
  (h₁ : least_num = 540)
  (h₂ : ∀ m, m % 12 = 5 → m ≥ least_num)
  (h₃ : n = 107) 
  : 540 % 107 = 5 :=
by sorry

end NUMINAMATH_GPT_number_leaves_remainder_five_l15_1529


namespace NUMINAMATH_GPT_sequence_sum_l15_1538

theorem sequence_sum (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l15_1538


namespace NUMINAMATH_GPT_find_triples_l15_1548

theorem find_triples (x y p : ℤ) (prime_p : Prime p) :
  x^2 - 3 * x * y + p^2 * y^2 = 12 * p ↔ 
  (p = 3 ∧ ( (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) ∨ (x = 4 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -4 ∧ y = -2) ) ) := 
by
  sorry

end NUMINAMATH_GPT_find_triples_l15_1548


namespace NUMINAMATH_GPT_work_efficiency_ratio_l15_1549

theorem work_efficiency_ratio (A B : ℝ) (k : ℝ)
  (h1 : A = k * B)
  (h2 : B = 1 / 27)
  (h3 : A + B = 1 / 9) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_work_efficiency_ratio_l15_1549


namespace NUMINAMATH_GPT_term_of_arithmetic_sequence_l15_1542

variable (a₁ : ℕ) (d : ℕ) (n : ℕ)

theorem term_of_arithmetic_sequence (h₁: a₁ = 2) (h₂: d = 5) (h₃: n = 50) :
    a₁ + (n - 1) * d = 247 := by
  sorry

end NUMINAMATH_GPT_term_of_arithmetic_sequence_l15_1542


namespace NUMINAMATH_GPT_bill_sunday_miles_l15_1574

-- Define the variables
variables (B S J : ℕ) -- B for miles Bill ran on Saturday, S for miles Bill ran on Sunday, J for miles Julia ran on Sunday

-- State the conditions
def condition1 (B S : ℕ) : Prop := S = B + 4
def condition2 (B S J : ℕ) : Prop := J = 2 * S
def condition3 (B S J : ℕ) : Prop := B + S + J = 20

-- The final theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (B S J : ℕ) 
  (h1 : condition1 B S)
  (h2 : condition2 B S J)
  (h3 : condition3 B S J) : 
  S = 6 := 
sorry

end NUMINAMATH_GPT_bill_sunday_miles_l15_1574


namespace NUMINAMATH_GPT_sarah_books_check_out_l15_1594

theorem sarah_books_check_out
  (words_per_minute : ℕ)
  (words_per_page : ℕ)
  (pages_per_book : ℕ)
  (reading_hours : ℕ)
  (number_of_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  pages_per_book = 80 →
  reading_hours = 20 →
  number_of_books = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sarah_books_check_out_l15_1594


namespace NUMINAMATH_GPT_circle_center_radius_l15_1564

theorem circle_center_radius :
  ∀ x y : ℝ,
  x^2 + y^2 + 4 * x - 6 * y - 3 = 0 →
  (∃ h k r : ℝ, (x + h)^2 + (y + k)^2 = r^2 ∧ h = -2 ∧ k = 3 ∧ r = 4) :=
by
  intros x y hxy
  sorry

end NUMINAMATH_GPT_circle_center_radius_l15_1564


namespace NUMINAMATH_GPT_division_multiplication_example_l15_1537

theorem division_multiplication_example : 120 / 4 / 2 * 3 = 45 := by
  sorry

end NUMINAMATH_GPT_division_multiplication_example_l15_1537


namespace NUMINAMATH_GPT_range_of_m_l15_1596

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)) :
  0 < m ∧ m ≤ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l15_1596


namespace NUMINAMATH_GPT_smallest_number_l15_1544

theorem smallest_number (N : ℤ) : (∃ (k : ℤ), N = 24 * k + 34) ∧ ∀ n, (∃ (k : ℤ), n = 24 * k + 10) -> n ≥ 34 := sorry

end NUMINAMATH_GPT_smallest_number_l15_1544


namespace NUMINAMATH_GPT_mean_temperature_correct_l15_1595

-- Define the list of temperatures
def temperatures : List ℤ := [-8, -5, -5, -6, 0, 4]

-- Define the mean temperature calculation
def mean_temperature (temps: List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

-- The theorem we want to prove
theorem mean_temperature_correct :
  mean_temperature temperatures = -10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_temperature_correct_l15_1595


namespace NUMINAMATH_GPT_minimize_quadratic_l15_1553

theorem minimize_quadratic (x : ℝ) :
  (∀ y : ℝ, x^2 + 14*x + 6 ≤ y^2 + 14*y + 6) ↔ x = -7 :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l15_1553


namespace NUMINAMATH_GPT_lcm_of_132_and_315_l15_1586

def n1 : ℕ := 132
def n2 : ℕ := 315

theorem lcm_of_132_and_315 :
  (Nat.lcm n1 n2) = 13860 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_lcm_of_132_and_315_l15_1586


namespace NUMINAMATH_GPT_maxProfitAchievable_l15_1534

namespace BarrelProduction

structure ProductionPlan where
  barrelsA : ℕ
  barrelsB : ℕ

def profit (plan : ProductionPlan) : ℕ :=
  300 * plan.barrelsA + 400 * plan.barrelsB

def materialAUsage (plan : ProductionPlan) : ℕ :=
  plan.barrelsA + 2 * plan.barrelsB

def materialBUsage (plan : ProductionPlan) : ℕ :=
  2 * plan.barrelsA + plan.barrelsB

def isValidPlan (plan : ProductionPlan) : Prop :=
  materialAUsage plan ≤ 12 ∧ materialBUsage plan ≤ 12

def maximumProfit : ℕ :=
  2800

theorem maxProfitAchievable : 
  ∃ (plan : ProductionPlan), isValidPlan plan ∧ profit plan = maximumProfit :=
sorry

end BarrelProduction

end NUMINAMATH_GPT_maxProfitAchievable_l15_1534


namespace NUMINAMATH_GPT_inequality_abc_squared_l15_1505

theorem inequality_abc_squared (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := 
sorry

end NUMINAMATH_GPT_inequality_abc_squared_l15_1505


namespace NUMINAMATH_GPT_arithmetic_series_remainder_l15_1592

noncomputable def arithmetic_series_sum_mod (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d) / 2) % 10

theorem arithmetic_series_remainder :
  let a := 3
  let d := 5
  let n := 21
  arithmetic_series_sum_mod a d n = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_remainder_l15_1592


namespace NUMINAMATH_GPT_number_is_580_l15_1593

noncomputable def find_number (x : ℝ) : Prop :=
  0.20 * x = 116

theorem number_is_580 (x : ℝ) (h : find_number x) : x = 580 :=
  by sorry

end NUMINAMATH_GPT_number_is_580_l15_1593


namespace NUMINAMATH_GPT_sequence_solution_exists_l15_1558

noncomputable def math_problem (a : ℕ → ℝ) : Prop :=
  ∀ n < 1990, a n > 0 ∧ a 1990 < 0

theorem sequence_solution_exists {a0 c : ℝ} (h_a0 : a0 > 0) (h_c : c > 0) :
  ∃ (a : ℕ → ℝ),
    a 0 = a0 ∧
    (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
    math_problem a :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_exists_l15_1558


namespace NUMINAMATH_GPT_age_of_b_l15_1598

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 := 
  sorry

end NUMINAMATH_GPT_age_of_b_l15_1598


namespace NUMINAMATH_GPT_b_money_used_for_10_months_l15_1506

theorem b_money_used_for_10_months
  (a_capital_ratio : ℚ)
  (a_time_used : ℕ)
  (b_profit_share : ℚ)
  (h1 : a_capital_ratio = 1 / 4)
  (h2 : a_time_used = 15)
  (h3 : b_profit_share = 2 / 3) :
  ∃ (b_time_used : ℕ), b_time_used = 10 :=
by
  sorry

end NUMINAMATH_GPT_b_money_used_for_10_months_l15_1506


namespace NUMINAMATH_GPT_maurice_late_467th_trip_l15_1589

-- Define the recurrence relation
def p (n : ℕ) : ℚ := 
  if n = 0 then 0
  else 1 / 4 * (p (n - 1) + 1)

-- Define the steady-state probability
def steady_state_p : ℚ := 1 / 3

-- Define L_n as the probability Maurice is late on the nth day
def L (n : ℕ) : ℚ := 1 - p n

-- The main goal (probability Maurice is late on his 467th trip)
theorem maurice_late_467th_trip :
  L 467 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_maurice_late_467th_trip_l15_1589


namespace NUMINAMATH_GPT_students_not_solving_any_problem_l15_1533

variable (A_0 A_1 A_2 A_3 A_4 A_5 A_6 : ℕ)

-- Given conditions
def number_of_students := 2006
def condition_1 := A_1 = 4 * A_2
def condition_2 := A_2 = 4 * A_3
def condition_3 := A_3 = 4 * A_4
def condition_4 := A_4 = 4 * A_5
def condition_5 := A_5 = 4 * A_6
def total_students := A_0 + A_1 = 2006

-- The final statement to be proven
theorem students_not_solving_any_problem : 
  (A_1 = 4 * A_2) →
  (A_2 = 4 * A_3) →
  (A_3 = 4 * A_4) →
  (A_4 = 4 * A_5) →
  (A_5 = 4 * A_6) →
  (A_0 + A_1 = 2006) →
  (A_0 = 982) :=
by
  intro h1 h2 h3 h4 h5 h6
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_students_not_solving_any_problem_l15_1533


namespace NUMINAMATH_GPT_positive_multiples_of_11_ending_with_7_l15_1513

-- Definitions for conditions
def is_multiple_of_11 (n : ℕ) : Prop := (n % 11 = 0)
def ends_with_7 (n : ℕ) : Prop := (n % 10 = 7)

-- Main theorem statement
theorem positive_multiples_of_11_ending_with_7 :
  ∃ n, (n = 13) ∧ ∀ k, is_multiple_of_11 k ∧ ends_with_7 k ∧ 0 < k ∧ k < 1500 → k = 77 + (k / 110) * 110 := 
sorry

end NUMINAMATH_GPT_positive_multiples_of_11_ending_with_7_l15_1513


namespace NUMINAMATH_GPT_smallest_odd_integer_of_set_l15_1565

theorem smallest_odd_integer_of_set (S : Set Int) 
  (h1 : ∃ m : Int, m ∈ S ∧ m = 149)
  (h2 : ∃ n : Int, n ∈ S ∧ n = 159)
  (h3 : ∀ a b : Int, a ∈ S → b ∈ S → a ≠ b → (a - b) % 2 = 0) : 
  ∃ s : Int, s ∈ S ∧ s = 137 :=
by sorry

end NUMINAMATH_GPT_smallest_odd_integer_of_set_l15_1565


namespace NUMINAMATH_GPT_power_of_two_ends_with_identical_digits_l15_1522

theorem power_of_two_ends_with_identical_digits : ∃ (k : ℕ), k ≥ 10 ∧ (∀ (x y : ℕ), 2^k = 1000 * x + 111 * y → y = 8 → (2^k % 1000 = 888)) :=
by sorry

end NUMINAMATH_GPT_power_of_two_ends_with_identical_digits_l15_1522


namespace NUMINAMATH_GPT_sam_watermelons_second_batch_l15_1590

theorem sam_watermelons_second_batch
  (initial_watermelons : ℕ)
  (total_watermelons : ℕ)
  (second_batch_watermelons : ℕ) :
  initial_watermelons = 4 →
  total_watermelons = 7 →
  second_batch_watermelons = total_watermelons - initial_watermelons →
  second_batch_watermelons = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_sam_watermelons_second_batch_l15_1590


namespace NUMINAMATH_GPT_point_to_polar_coordinates_l15_1563

noncomputable def convert_to_polar_coordinates (x y : ℝ) (r θ : ℝ) : Prop :=
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x)

theorem point_to_polar_coordinates :
  convert_to_polar_coordinates 8 (2 * Real.sqrt 6) 
    (2 * Real.sqrt 22) (Real.arctan (Real.sqrt 6 / 4)) :=
sorry

end NUMINAMATH_GPT_point_to_polar_coordinates_l15_1563


namespace NUMINAMATH_GPT_lucy_additional_kilometers_l15_1525

theorem lucy_additional_kilometers
  (mary_distance : ℚ := (3/8) * 24)
  (edna_distance : ℚ := (2/3) * mary_distance)
  (lucy_distance : ℚ := (5/6) * edna_distance) :
  (mary_distance - lucy_distance) = 4 :=
by
  sorry

end NUMINAMATH_GPT_lucy_additional_kilometers_l15_1525


namespace NUMINAMATH_GPT_floor_expression_correct_l15_1572

theorem floor_expression_correct :
  (∃ x : ℝ, x = 2007 ^ 3 / (2005 * 2006) - 2005 ^ 3 / (2006 * 2007) ∧ ⌊x⌋ = 8) := 
sorry

end NUMINAMATH_GPT_floor_expression_correct_l15_1572


namespace NUMINAMATH_GPT_undefined_expression_l15_1581

theorem undefined_expression (y : ℝ) : (y^2 - 16 * y + 64 = 0) ↔ (y = 8) := by
  sorry

end NUMINAMATH_GPT_undefined_expression_l15_1581


namespace NUMINAMATH_GPT_value_of_x_minus_2y_l15_1519

theorem value_of_x_minus_2y (x y : ℝ) (h1 : 0.5 * x = y + 20) : x - 2 * y = 40 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_minus_2y_l15_1519


namespace NUMINAMATH_GPT_correct_option_is_A_l15_1507

def second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (0, 4)
def point_D : ℝ × ℝ := (5, -6)

theorem correct_option_is_A :
  (second_quadrant point_A) ∧
  ¬(second_quadrant point_B) ∧
  ¬(second_quadrant point_C) ∧
  ¬(second_quadrant point_D) :=
by sorry

end NUMINAMATH_GPT_correct_option_is_A_l15_1507


namespace NUMINAMATH_GPT_unique_n_degree_polynomial_exists_l15_1536

theorem unique_n_degree_polynomial_exists (n : ℕ) (h : n > 0) :
  ∃! (f : Polynomial ℝ), Polynomial.degree f = n ∧
    f.eval 0 = 1 ∧
    ∀ x : ℝ, (x + 1) * (f.eval x)^2 - 1 = -((x + 1) * (f.eval (-x))^2 - 1) := 
sorry

end NUMINAMATH_GPT_unique_n_degree_polynomial_exists_l15_1536


namespace NUMINAMATH_GPT_find_min_k_l15_1576

theorem find_min_k (k : ℕ) 
  (h1 : k > 0) 
  (h2 : ∀ (A : Finset ℕ), A ⊆ (Finset.range 26).erase 0 → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (2 / 3 : ℝ) ≤ x / y ∧ x / y ≤ (3 / 2 : ℝ)) : 
  k = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_min_k_l15_1576


namespace NUMINAMATH_GPT_peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l15_1550

-- Define the conditions
variable (a b c : ℕ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)

-- Part 1
theorem peter_can_transfer_all_money_into_two_accounts :
  ∃ x y, (x + y = a + b + c ∧ y = 0) ∨
          (∃ z, (a + b + c = x + y + z ∧ y = 0 ∧ z = 0)) :=
  sorry

-- Part 2
theorem peter_cannot_always_transfer_all_money_into_one_account :
  ((a + b + c) % 2 = 1 → ¬ ∃ x, x = a + b + c) :=
  sorry

end NUMINAMATH_GPT_peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l15_1550


namespace NUMINAMATH_GPT_halfway_fraction_between_is_one_fourth_l15_1540

theorem halfway_fraction_between_is_one_fourth : 
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  ((f1 + f2 + f3) / 3) = (1 / 4) := 
by
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  sorry

end NUMINAMATH_GPT_halfway_fraction_between_is_one_fourth_l15_1540


namespace NUMINAMATH_GPT_correct_option_B_l15_1584

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_B_l15_1584


namespace NUMINAMATH_GPT_problem_solution_l15_1517

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end NUMINAMATH_GPT_problem_solution_l15_1517


namespace NUMINAMATH_GPT_determine_S5_l15_1545

noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1 / x^m

theorem determine_S5 (x : ℝ) (h : x + 1 / x = 3) : S x 5 = 123 :=
by
  sorry

end NUMINAMATH_GPT_determine_S5_l15_1545


namespace NUMINAMATH_GPT_pie_remaining_portion_l15_1599

theorem pie_remaining_portion (Carlos_share Maria_share remaining: ℝ)
  (hCarlos : Carlos_share = 0.65)
  (hRemainingAfterCarlos : remaining = 1 - Carlos_share)
  (hMaria : Maria_share = remaining / 2) :
  remaining - Maria_share = 0.175 :=
by
  sorry

end NUMINAMATH_GPT_pie_remaining_portion_l15_1599


namespace NUMINAMATH_GPT_max_quarters_l15_1556

/-- Prove that given the conditions for the number of nickels, dimes, and quarters,
    the maximum number of quarters can be 20. --/
theorem max_quarters {a b c : ℕ} (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) :
  c ≤ 20 :=
sorry

end NUMINAMATH_GPT_max_quarters_l15_1556


namespace NUMINAMATH_GPT_roots_poly_cond_l15_1573

theorem roots_poly_cond (α β p q γ δ : ℝ) 
  (h1 : α ^ 2 + p * α - 1 = 0) 
  (h2 : β ^ 2 + p * β - 1 = 0) 
  (h3 : γ ^ 2 + q * γ - 1 = 0) 
  (h4 : δ ^ 2 + q * δ - 1 = 0)
  (h5 : γ * δ = -1) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = -(p - q) ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_roots_poly_cond_l15_1573


namespace NUMINAMATH_GPT_initial_bananas_per_child_l15_1591

theorem initial_bananas_per_child (B x : ℕ) (total_children : ℕ := 780) (absent_children : ℕ := 390) :
  390 * (x + 2) = total_children * x → x = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_initial_bananas_per_child_l15_1591


namespace NUMINAMATH_GPT_trajectory_of_moving_circle_l15_1571

noncomputable def trajectory_equation_of_moving_circle_center 
  (x y : Real) : Prop :=
  (∃ r : Real, 
    ((x + 5)^2 + y^2 = 16) ∧ 
    ((x - 5)^2 + y^2 = 16)
  ) → (x > 0 → x^2 / 16 - y^2 / 9 = 1)

-- here's the statement of the proof problem
theorem trajectory_of_moving_circle
  (h₁ : ∀ x y : Real, (x + 5)^2 + y^2 = 16)
  (h₂ : ∀ x y : Real, (x - 5)^2 + y^2 = 16) :
  ∀ x y : Real, trajectory_equation_of_moving_circle_center x y :=
sorry

end NUMINAMATH_GPT_trajectory_of_moving_circle_l15_1571


namespace NUMINAMATH_GPT_intersection_M_N_l15_1527

open Set Real

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | abs (x - 1) ≤ 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l15_1527


namespace NUMINAMATH_GPT_triangle_area_l15_1559

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a + b > c) (h3 : a + c > b) (h4 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 3 ∧ c = 5 ∨ a = 5 ∧ b = 4 ∧ c = 3 ∨
  a = 5 ∧ b = 3 ∧ c = 4 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 3 ∧ b = 5 ∧ c = 4 → 
  (1 / 2 : ℝ) * ↑a * ↑b = 6 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l15_1559


namespace NUMINAMATH_GPT_find_number_l15_1555

theorem find_number (x : ℚ) (h : (3 * x / 2) + 6 = 11) : x = 10 / 3 :=
sorry

end NUMINAMATH_GPT_find_number_l15_1555


namespace NUMINAMATH_GPT_marked_price_l15_1508

theorem marked_price (x : ℝ) (payment : ℝ) (discount : ℝ) (hx : (payment = 90) ∧ ((x ≤ 100 ∧ discount = 0.1) ∨ (x > 100 ∧ discount = 0.2))) :
  (x = 100 ∨ x = 112.5) := by
  sorry

end NUMINAMATH_GPT_marked_price_l15_1508


namespace NUMINAMATH_GPT_jogger_ahead_of_train_l15_1552

theorem jogger_ahead_of_train (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (time_to_pass : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 100) 
  (h4 : time_to_pass = 34) : 
  ∃ d : ℝ, d = 240 :=
by
  sorry

end NUMINAMATH_GPT_jogger_ahead_of_train_l15_1552


namespace NUMINAMATH_GPT_probability_at_least_one_visits_guangzhou_l15_1512

-- Define the probabilities of visiting for persons A, B, and C
def p_A : ℚ := 2 / 3
def p_B : ℚ := 1 / 4
def p_C : ℚ := 3 / 5

-- Calculate the probability that no one visits
def p_not_A : ℚ := 1 - p_A
def p_not_B : ℚ := 1 - p_B
def p_not_C : ℚ := 1 - p_C

-- Calculate the probability that at least one person visits
def p_none_visit : ℚ := p_not_A * p_not_B * p_not_C
def p_at_least_one_visit : ℚ := 1 - p_none_visit

-- The statement we need to prove
theorem probability_at_least_one_visits_guangzhou : p_at_least_one_visit = 9 / 10 :=
by 
  sorry

end NUMINAMATH_GPT_probability_at_least_one_visits_guangzhou_l15_1512


namespace NUMINAMATH_GPT_prove_mutually_exclusive_l15_1502

def bag : List String := ["red", "red", "red", "black", "black"]

def at_least_one_black (drawn : List String) : Prop :=
  "black" ∈ drawn

def all_red (drawn : List String) : Prop :=
  ∀ b ∈ drawn, b = "red"

def events_mutually_exclusive : Prop :=
  ∀ drawn, at_least_one_black drawn → ¬all_red drawn

theorem prove_mutually_exclusive :
  events_mutually_exclusive
:= by
  sorry

end NUMINAMATH_GPT_prove_mutually_exclusive_l15_1502


namespace NUMINAMATH_GPT_ping_pong_tournament_l15_1524

theorem ping_pong_tournament :
  ∃ n: ℕ, 
    (∃ m: ℕ, m ≥ 0 ∧ m ≤ 2 ∧ 2 * n + m = 29) ∧
    n = 14 ∧
    (n + 2 = 16) := 
by {
  sorry
}

end NUMINAMATH_GPT_ping_pong_tournament_l15_1524


namespace NUMINAMATH_GPT_c_share_of_profit_l15_1535

-- Definitions for the investments and total profit
def investments_a := 800
def investments_b := 1000
def investments_c := 1200
def total_profit := 1000

-- Definition for the share of profits based on the ratio of investments
def share_of_c : ℕ :=
  let ratio_a := 4
  let ratio_b := 5
  let ratio_c := 6
  let total_ratio := ratio_a + ratio_b + ratio_c
  (ratio_c * total_profit) / total_ratio

-- The theorem to be proved
theorem c_share_of_profit : share_of_c = 400 := by
  sorry

end NUMINAMATH_GPT_c_share_of_profit_l15_1535


namespace NUMINAMATH_GPT_fraction_paint_used_second_week_l15_1546

noncomputable def total_paint : ℕ := 360
noncomputable def paint_used_first_week : ℕ := total_paint / 4
noncomputable def remaining_paint_after_first_week : ℕ := total_paint - paint_used_first_week
noncomputable def total_paint_used : ℕ := 135
noncomputable def paint_used_second_week : ℕ := total_paint_used - paint_used_first_week
noncomputable def remaining_paint_after_first_week_fraction : ℚ := paint_used_second_week / remaining_paint_after_first_week

theorem fraction_paint_used_second_week : remaining_paint_after_first_week_fraction = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_fraction_paint_used_second_week_l15_1546


namespace NUMINAMATH_GPT_sequences_count_equals_fibonacci_n_21_l15_1557

noncomputable def increasing_sequences_count (n: ℕ) : ℕ := 
  -- Function to count the number of valid increasing sequences
  sorry

def fibonacci : ℕ → ℕ 
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sequences_count_equals_fibonacci_n_21 :
  increasing_sequences_count 20 = fibonacci 21 :=
sorry

end NUMINAMATH_GPT_sequences_count_equals_fibonacci_n_21_l15_1557


namespace NUMINAMATH_GPT_students_at_end_l15_1554

def initial_students : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0

theorem students_at_end : initial_students - students_left - students_transferred = 28.0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_students_at_end_l15_1554


namespace NUMINAMATH_GPT_trapezoid_area_correct_l15_1583

noncomputable def trapezoid_area : ℝ := 
  let base1 : ℝ := 8
  let base2 : ℝ := 4
  let height : ℝ := 2
  (1 / 2) * (base1 + base2) * height

theorem trapezoid_area_correct :
  trapezoid_area = 12.0 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end NUMINAMATH_GPT_trapezoid_area_correct_l15_1583


namespace NUMINAMATH_GPT_range_of_a_l15_1511

-- Define the propositions
def p (x : ℝ) := (x - 1) * (x - 2) > 0
def q (a x : ℝ) := x^2 + (a - 1) * x - a > 0

-- Define the solution sets
def A := {x : ℝ | p x}
def B (a : ℝ) := {x : ℝ | q a x}

-- State the proof problem
theorem range_of_a (a : ℝ) : 
  (∀ x, p x → q a x) ∧ (∃ x, ¬p x ∧ q a x) → -2 < a ∧ a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l15_1511


namespace NUMINAMATH_GPT_find_x_from_percentage_l15_1500

theorem find_x_from_percentage : 
  ∃ x : ℚ, 0.65 * x = 0.20 * 487.50 := 
sorry

end NUMINAMATH_GPT_find_x_from_percentage_l15_1500


namespace NUMINAMATH_GPT_cost_difference_of_buses_l15_1526

-- Definitions from the conditions
def bus_cost_equations (x y : ℝ) :=
  (x + 2 * y = 260) ∧ (2 * x + y = 280)

-- The statement to prove
theorem cost_difference_of_buses (x y : ℝ) (h : bus_cost_equations x y) :
  x - y = 20 :=
sorry

end NUMINAMATH_GPT_cost_difference_of_buses_l15_1526


namespace NUMINAMATH_GPT_majority_vote_is_280_l15_1561

-- Definitions based on conditions from step (a)
def totalVotes : ℕ := 1400
def winningPercentage : ℝ := 0.60
def losingPercentage : ℝ := 0.40

-- Majority computation based on the winning and losing percentages
def majorityVotes : ℝ := totalVotes * winningPercentage - totalVotes * losingPercentage

-- Theorem statement
theorem majority_vote_is_280 : majorityVotes = 280 := by
  sorry

end NUMINAMATH_GPT_majority_vote_is_280_l15_1561


namespace NUMINAMATH_GPT_min_value_x2y2z2_l15_1541

open Real

noncomputable def condition (x y z : ℝ) : Prop := (1 / x + 1 / y + 1 / z = 3)

theorem min_value_x2y2z2 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : condition x y z) :
  x^2 * y^2 * z^2 ≥ 1 / 64 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x2y2z2_l15_1541


namespace NUMINAMATH_GPT_determinant_modified_l15_1575

variable (a b c d : ℝ)

theorem determinant_modified (h : a * d - b * c = 10) :
  (a + 2 * c) * d - (b + 3 * d) * c = 10 - c * d := by
  sorry

end NUMINAMATH_GPT_determinant_modified_l15_1575


namespace NUMINAMATH_GPT_nonneg_real_sum_inequality_l15_1516

theorem nonneg_real_sum_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end NUMINAMATH_GPT_nonneg_real_sum_inequality_l15_1516


namespace NUMINAMATH_GPT_count_positive_integers_in_range_l15_1531

theorem count_positive_integers_in_range :
  ∃ (count : ℕ), count = 11 ∧
    ∀ (n : ℕ), 300 < n^2 ∧ n^2 < 800 → (n ≥ 18 ∧ n ≤ 28) :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_in_range_l15_1531


namespace NUMINAMATH_GPT_david_ate_more_than_emma_l15_1514

-- Definitions and conditions
def contestants : Nat := 8
def pies_david_ate : Nat := 8
def pies_emma_ate : Nat := 2
def pies_by_david (contestants pies_david_ate: Nat) : Prop := pies_david_ate = 8
def pies_by_emma (contestants pies_emma_ate: Nat) : Prop := pies_emma_ate = 2

-- Theorem statement
theorem david_ate_more_than_emma (contestants pies_david_ate pies_emma_ate : Nat) (h_david : pies_by_david contestants pies_david_ate) (h_emma : pies_by_emma contestants pies_emma_ate) : pies_david_ate - pies_emma_ate = 6 :=
by
  sorry

end NUMINAMATH_GPT_david_ate_more_than_emma_l15_1514


namespace NUMINAMATH_GPT_domain_composite_l15_1539

-- Define the conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- The theorem statement
theorem domain_composite (h : ∀ x, domain_f x → 0 ≤ x ∧ x ≤ 4) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_domain_composite_l15_1539


namespace NUMINAMATH_GPT_t_bounds_f_bounds_l15_1547

noncomputable def t (x : ℝ) : ℝ := 3^x

noncomputable def f (x : ℝ) : ℝ := 9^x - 2 * 3^x + 4

theorem t_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (1/3 ≤ t x ∧ t x ≤ 9) :=
sorry

theorem f_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (3 ≤ f x ∧ f x ≤ 67) :=
sorry

end NUMINAMATH_GPT_t_bounds_f_bounds_l15_1547


namespace NUMINAMATH_GPT_initial_students_began_contest_l15_1504

theorem initial_students_began_contest
  (n : ℕ)
  (first_round_fraction : ℚ)
  (second_round_fraction : ℚ)
  (remaining_students : ℕ) :
  first_round_fraction * second_round_fraction * n = remaining_students →
  remaining_students = 18 →
  first_round_fraction = 0.3 →
  second_round_fraction = 0.5 →
  n = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_students_began_contest_l15_1504


namespace NUMINAMATH_GPT_average_score_difference_l15_1567

theorem average_score_difference {A B : ℝ} (hA : (19 * A + 125) / 20 = A + 5) (hB : (17 * B + 145) / 18 = B + 6) :
  (B + 6) - (A + 5) = 13 :=
  sorry

end NUMINAMATH_GPT_average_score_difference_l15_1567


namespace NUMINAMATH_GPT_ball_radius_and_surface_area_l15_1523

theorem ball_radius_and_surface_area (d h r : ℝ) (radius_eq : d / 2 = 6) (depth_eq : h = 2) 
  (pythagorean : (r - h)^2 + (d / 2)^2 = r^2) :
  r = 10 ∧ (4 * Real.pi * r^2 = 400 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_ball_radius_and_surface_area_l15_1523


namespace NUMINAMATH_GPT_least_hourly_number_l15_1520

def is_clock_equivalent (a b : ℕ) : Prop := (a - b) % 12 = 0

theorem least_hourly_number : ∃ n ≥ 6, is_clock_equivalent n (n * n) ∧ ∀ m ≥ 6, is_clock_equivalent m (m * m) → 9 ≤ m → n = 9 := 
by
  sorry

end NUMINAMATH_GPT_least_hourly_number_l15_1520


namespace NUMINAMATH_GPT_division_result_l15_1518

def numerator : ℕ := 3 * 4 * 5
def denominator : ℕ := 2 * 3
def quotient : ℕ := numerator / denominator

theorem division_result : quotient = 10 := by
  sorry

end NUMINAMATH_GPT_division_result_l15_1518


namespace NUMINAMATH_GPT_spaces_per_row_l15_1509

theorem spaces_per_row 
  (kind_of_tomatoes : ℕ)
  (tomatoes_per_kind : ℕ)
  (kind_of_cucumbers : ℕ)
  (cucumbers_per_kind : ℕ)
  (potatoes : ℕ)
  (rows : ℕ)
  (additional_spaces : ℕ)
  (h1 : kind_of_tomatoes = 3)
  (h2 : tomatoes_per_kind = 5)
  (h3 : kind_of_cucumbers = 5)
  (h4 : cucumbers_per_kind = 4)
  (h5 : potatoes = 30)
  (h6 : rows = 10)
  (h7 : additional_spaces = 85) :
  (kind_of_tomatoes * tomatoes_per_kind + kind_of_cucumbers * cucumbers_per_kind + potatoes + additional_spaces) / rows = 15 :=
by
  sorry

end NUMINAMATH_GPT_spaces_per_row_l15_1509


namespace NUMINAMATH_GPT_probability_one_solves_l15_1532

theorem probability_one_solves :
  let pA := 0.8
  let pB := 0.7
  (pA * (1 - pB) + pB * (1 - pA)) = 0.38 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_solves_l15_1532


namespace NUMINAMATH_GPT_integer_count_between_cubes_l15_1501

-- Definitions and conditions
def a : ℝ := 10.7
def b : ℝ := 10.8

-- Precomputed values
def a_cubed : ℝ := 1225.043
def b_cubed : ℝ := 1259.712

-- The theorem to prove
theorem integer_count_between_cubes (ha : a ^ 3 = a_cubed) (hb : b ^ 3 = b_cubed) :
  let start := Int.ceil a_cubed
  let end_ := Int.floor b_cubed
  end_ - start + 1 = 34 :=
by
  sorry

end NUMINAMATH_GPT_integer_count_between_cubes_l15_1501


namespace NUMINAMATH_GPT_initial_children_on_bus_l15_1570

-- Definitions based on conditions
variable (x : ℕ) -- number of children who got off the bus
variable (y : ℕ) -- initial number of children on the bus
variable (after_exchange : ℕ := 30) -- number of children on the bus after exchange
variable (got_on : ℕ := 82) -- number of children who got on the bus
variable (extra_on : ℕ := 2) -- extra children who got on compared to got off

-- Problem translated to Lean 4 statement
theorem initial_children_on_bus (h : got_on = x + extra_on) (hx : y + got_on - x = after_exchange) : y = 28 :=
by
  sorry

end NUMINAMATH_GPT_initial_children_on_bus_l15_1570


namespace NUMINAMATH_GPT_time_to_cross_stationary_train_l15_1580

theorem time_to_cross_stationary_train (t_pole : ℝ) (speed_train : ℝ) (length_stationary_train : ℝ) 
  (t_pole_eq : t_pole = 5) (speed_train_eq : speed_train = 64.8) (length_stationary_train_eq : length_stationary_train = 360) :
  (t_pole * speed_train + length_stationary_train) / speed_train = 10.56 := 
by
  rw [t_pole_eq, speed_train_eq, length_stationary_train_eq]
  norm_num
  sorry

end NUMINAMATH_GPT_time_to_cross_stationary_train_l15_1580


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l15_1515

theorem option_A (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) : a 20 = 211 :=
sorry

theorem option_B (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2^n * a n) : a 5 = 2^10 :=
sorry

theorem option_C (S : ℕ → ℝ) (h₀ : ∀ n, S n = 3^n + 1/2) : ¬(∃ r : ℝ, ∀ n, S n = S 1 * r ^ (n - 1)) :=
sorry

theorem option_D (S : ℕ → ℝ) (a : ℕ → ℝ) (h₀ : S 1 = 1) 
  (h₁ : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1))
  (h₂ : (S 8) / 8 - (S 4) / 4 = 8) : a 6 = 21 :=
sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l15_1515


namespace NUMINAMATH_GPT_steve_speed_during_race_l15_1543

theorem steve_speed_during_race 
  (distance_gap : ℝ) 
  (john_speed : ℝ) 
  (time : ℝ) 
  (john_ahead : ℝ)
  (steve_speed : ℝ) :
  distance_gap = 16 →
  john_speed = 4.2 →
  time = 36 →
  john_ahead = 2 →
  steve_speed = (151.2 - 18) / 36 :=
by
  sorry

end NUMINAMATH_GPT_steve_speed_during_race_l15_1543


namespace NUMINAMATH_GPT_prime_factors_1260_l15_1588

theorem prime_factors_1260 (w x y z : ℕ) (h : 2 ^ w * 3 ^ x * 5 ^ y * 7 ^ z = 1260) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
by sorry

end NUMINAMATH_GPT_prime_factors_1260_l15_1588


namespace NUMINAMATH_GPT_axis_of_symmetry_shifted_cos_l15_1503

noncomputable def shifted_cos_axis_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * (Real.pi / 2) - (Real.pi / 12)

theorem axis_of_symmetry_shifted_cos :
  shifted_cos_axis_symmetry x :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_shifted_cos_l15_1503


namespace NUMINAMATH_GPT_balls_into_boxes_problem_l15_1597

theorem balls_into_boxes_problem :
  ∃ (n : ℕ), n = 144 ∧ ∃ (balls : Fin 4 → ℕ), 
  (∃ (boxes : Fin 4 → Fin 4), 
    (∀ (b : Fin 4), boxes b < 4 ∧ boxes b ≠ b) ∧ 
    (∃! (empty_box : Fin 4), ∀ (b : Fin 4), (boxes b = empty_box) → false)) := 
by
  sorry

end NUMINAMATH_GPT_balls_into_boxes_problem_l15_1597


namespace NUMINAMATH_GPT_car_R_average_speed_l15_1587

theorem car_R_average_speed 
  (R P S: ℝ)
  (h1: S = 2 * P)
  (h2: P + 2 = R)
  (h3: P = R + 10)
  (h4: S = R + 20) :
  R = 25 :=
by 
  sorry

end NUMINAMATH_GPT_car_R_average_speed_l15_1587


namespace NUMINAMATH_GPT_total_bill_is_correct_l15_1560

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

end NUMINAMATH_GPT_total_bill_is_correct_l15_1560


namespace NUMINAMATH_GPT_nat_gt_10_is_diff_of_hypotenuse_numbers_l15_1530

def is_hypotenuse_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem nat_gt_10_is_diff_of_hypotenuse_numbers (n : ℕ) (h : n > 10) : 
  ∃ (n₁ n₂ : ℕ), is_hypotenuse_number n₁ ∧ is_hypotenuse_number n₂ ∧ n = n₁ - n₂ :=
by
  sorry

end NUMINAMATH_GPT_nat_gt_10_is_diff_of_hypotenuse_numbers_l15_1530
