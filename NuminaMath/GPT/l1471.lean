import Mathlib

namespace NUMINAMATH_GPT_largest_in_given_numbers_l1471_147111

noncomputable def A := 5.14322
noncomputable def B := 5.1432222222222222222 -- B = 5.143(bar)2
noncomputable def C := 5.1432323232323232323 -- C = 5.14(bar)32
noncomputable def D := 5.1432432432432432432 -- D = 5.1(bar)432
noncomputable def E := 5.1432143214321432143 -- E = 5.(bar)4321

theorem largest_in_given_numbers : D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end NUMINAMATH_GPT_largest_in_given_numbers_l1471_147111


namespace NUMINAMATH_GPT_yogurt_cost_l1471_147106

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end NUMINAMATH_GPT_yogurt_cost_l1471_147106


namespace NUMINAMATH_GPT_interval_for_f_l1471_147182

noncomputable def f (x : ℝ) : ℝ :=
-0.5 * x ^ 2 + 13 / 2

theorem interval_for_f (a b : ℝ) :
  f a = 2 * b ∧ f b = 2 * a ∧ (a ≤ 0 ∨ 0 ≤ b) → 
  ([a, b] = [1, 3] ∨ [a, b] = [-2 - Real.sqrt 17, 13 / 4]) :=
by sorry

end NUMINAMATH_GPT_interval_for_f_l1471_147182


namespace NUMINAMATH_GPT_total_skips_correct_l1471_147116

def bob_skip_rate := 12
def jim_skip_rate := 15
def sally_skip_rate := 18

def bob_rocks := 10
def jim_rocks := 8
def sally_rocks := 12

theorem total_skips_correct : 
  (bob_skip_rate * bob_rocks) + (jim_skip_rate * jim_rocks) + (sally_skip_rate * sally_rocks) = 456 := by
  sorry

end NUMINAMATH_GPT_total_skips_correct_l1471_147116


namespace NUMINAMATH_GPT_SplitWinnings_l1471_147147

noncomputable def IstvanInitialContribution : ℕ := 5000 * 20
noncomputable def IstvanSecondPeriodContribution : ℕ := (5000 + 4000) * 30
noncomputable def IstvanThirdPeriodContribution : ℕ := (5000 + 4000 - 2500) * 40
noncomputable def IstvanTotalContribution : ℕ := IstvanInitialContribution + IstvanSecondPeriodContribution + IstvanThirdPeriodContribution

noncomputable def KalmanContribution : ℕ := 4000 * 70
noncomputable def LaszloContribution : ℕ := 2500 * 40
noncomputable def MiklosContributionAdjustment : ℕ := 2000 * 90

noncomputable def IstvanExpectedShare : ℕ := IstvanTotalContribution * 12 / 100
noncomputable def KalmanExpectedShare : ℕ := KalmanContribution * 12 / 100
noncomputable def LaszloExpectedShare : ℕ := LaszloContribution * 12 / 100
noncomputable def MiklosExpectedShare : ℕ := MiklosContributionAdjustment * 12 / 100

noncomputable def IstvanActualShare : ℕ := IstvanExpectedShare * 7 / 8
noncomputable def KalmanActualShare : ℕ := (KalmanExpectedShare - MiklosExpectedShare) * 7 / 8
noncomputable def LaszloActualShare : ℕ := LaszloExpectedShare * 7 / 8
noncomputable def MiklosActualShare : ℕ := MiklosExpectedShare * 7 / 8

theorem SplitWinnings :
  IstvanActualShare = 54600 ∧ KalmanActualShare = 7800 ∧ LaszloActualShare = 10500 ∧ MiklosActualShare = 18900 :=
by
  sorry

end NUMINAMATH_GPT_SplitWinnings_l1471_147147


namespace NUMINAMATH_GPT_portraits_after_lunch_before_gym_class_l1471_147101

-- Define the total number of students in the class
def total_students : ℕ := 24

-- Define the number of students who had their portraits taken before lunch
def students_before_lunch : ℕ := total_students / 3

-- Define the number of students who have not yet had their picture taken after gym class
def students_after_gym_class : ℕ := 6

-- Define the number of students who had their portraits taken before gym class
def students_before_gym_class : ℕ := total_students - students_after_gym_class

-- Define the number of students who had their portraits taken after lunch but before gym class
def students_after_lunch_before_gym_class : ℕ := students_before_gym_class - students_before_lunch

-- Statement of the theorem
theorem portraits_after_lunch_before_gym_class :
  students_after_lunch_before_gym_class = 10 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_portraits_after_lunch_before_gym_class_l1471_147101


namespace NUMINAMATH_GPT_negative_integer_solution_l1471_147127

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end NUMINAMATH_GPT_negative_integer_solution_l1471_147127


namespace NUMINAMATH_GPT_value_of_expression_l1471_147132

theorem value_of_expression : 
  ∀ (a x y : ℤ), 
  (x = a + 5) → 
  (a = 20) → 
  (y = 25) → 
  (x - y) * (x + y) = 0 :=
by
  intros a x y h1 h2 h3
  -- proof goes here
  sorry

end NUMINAMATH_GPT_value_of_expression_l1471_147132


namespace NUMINAMATH_GPT_unique_pair_odd_prime_l1471_147197

theorem unique_pair_odd_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃! (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n) + (1 / m) ∧ 
  n = (p + 1) / 2 ∧ m = (p * (p + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_pair_odd_prime_l1471_147197


namespace NUMINAMATH_GPT_isosceles_triangle_solution_l1471_147123

noncomputable def isosceles_triangle_sides (x y : ℝ) : Prop :=
(x + 1/2 * y = 6 ∧ 1/2 * x + y = 12) ∨ (x + 1/2 * y = 12 ∧ 1/2 * x + y = 6)

theorem isosceles_triangle_solution :
  ∃ (x y : ℝ), isosceles_triangle_sides x y ∧ x = 8 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_solution_l1471_147123


namespace NUMINAMATH_GPT_resistance_at_least_2000_l1471_147148

variable (U : ℝ) (I : ℝ) (R : ℝ)

-- Given conditions:
def voltage := U = 220
def max_current := I ≤ 0.11

-- Ohm's law in this context
def ohms_law := I = U / R

-- Proof problem statement:
theorem resistance_at_least_2000 (voltage : U = 220) (max_current : I ≤ 0.11) (ohms_law : I = U / R) : R ≥ 2000 :=
sorry

end NUMINAMATH_GPT_resistance_at_least_2000_l1471_147148


namespace NUMINAMATH_GPT_seating_capacity_for_ten_tables_in_two_rows_l1471_147129

-- Definitions based on the problem conditions
def seating_for_one_table : ℕ := 6

def seating_for_two_tables : ℕ := 10

def seating_for_three_tables : ℕ := 14

def additional_people_per_table : ℕ := 4

-- Calculating the seating capacity for n tables based on the pattern
def seating_capacity (n : ℕ) : ℕ :=
  if n = 1 then seating_for_one_table
  else seating_for_one_table + (n - 1) * additional_people_per_table

-- Proof statement without the proof
theorem seating_capacity_for_ten_tables_in_two_rows :
  (seating_capacity 5) * 2 = 44 :=
by sorry

end NUMINAMATH_GPT_seating_capacity_for_ten_tables_in_two_rows_l1471_147129


namespace NUMINAMATH_GPT_set_union_inter_eq_l1471_147119

open Set

-- Conditions: Definitions of sets M, N, and P
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3, 4, 5}

-- Claim: The result of (M ∩ N) ∪ P equals {1, 2, 3, 4, 5}
theorem set_union_inter_eq :
  (M ∩ N ∪ P) = {1, 2, 3, 4, 5} := 
by
  sorry

end NUMINAMATH_GPT_set_union_inter_eq_l1471_147119


namespace NUMINAMATH_GPT_A_plus_B_l1471_147180

theorem A_plus_B {A B : ℚ} (h : ∀ x : ℚ, (Bx - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) : 
  A + B = 33 / 5 := sorry

end NUMINAMATH_GPT_A_plus_B_l1471_147180


namespace NUMINAMATH_GPT_original_smallest_element_l1471_147122

theorem original_smallest_element (x : ℤ) 
  (h1 : x < -1) 
  (h2 : x + 14 + 0 + 6 + 9 = 2 * (2 + 3 + 0 + 6 + 9)) : 
  x = -4 :=
by sorry

end NUMINAMATH_GPT_original_smallest_element_l1471_147122


namespace NUMINAMATH_GPT_kanul_total_amount_l1471_147105

-- Definitions based on the conditions
def raw_materials_cost : ℝ := 35000
def machinery_cost : ℝ := 40000
def marketing_cost : ℝ := 15000
def total_spent : ℝ := raw_materials_cost + machinery_cost + marketing_cost
def spending_percentage : ℝ := 0.25

-- The statement we want to prove
theorem kanul_total_amount (T : ℝ) (h : total_spent = spending_percentage * T) : T = 360000 :=
by
  sorry

end NUMINAMATH_GPT_kanul_total_amount_l1471_147105


namespace NUMINAMATH_GPT_floor_equation_l1471_147156

theorem floor_equation (n : ℤ) (h : ⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋^2 = 5) : n = 11 :=
sorry

end NUMINAMATH_GPT_floor_equation_l1471_147156


namespace NUMINAMATH_GPT_expression_value_l1471_147133

theorem expression_value : ((40 + 15) ^ 2 - 15 ^ 2) = 2800 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l1471_147133


namespace NUMINAMATH_GPT_manager_salary_is_3600_l1471_147178

-- Definitions based on the conditions
def average_salary_20_employees := 1500
def number_of_employees := 20
def new_average_salary := 1600
def number_of_people_incl_manager := number_of_employees + 1

-- Calculate necessary total salaries and manager's salary
def total_salary_of_20_employees := number_of_employees * average_salary_20_employees
def new_total_salary_with_manager := number_of_people_incl_manager * new_average_salary
def manager_monthly_salary := new_total_salary_with_manager - total_salary_of_20_employees

-- The statement to be proved
theorem manager_salary_is_3600 : manager_monthly_salary = 3600 :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_is_3600_l1471_147178


namespace NUMINAMATH_GPT_calculate_ratio_milk_l1471_147135

def ratio_milk_saturdays_weekdays (S : ℕ) : Prop :=
  let Weekdays := 15 -- total milk on weekdays
  let Sundays := 9 -- total milk on Sundays
  S + Weekdays + Sundays = 30 → S / Weekdays = 2 / 5

theorem calculate_ratio_milk : ratio_milk_saturdays_weekdays 6 :=
by
  unfold ratio_milk_saturdays_weekdays
  intros
  apply sorry -- Proof goes here

end NUMINAMATH_GPT_calculate_ratio_milk_l1471_147135


namespace NUMINAMATH_GPT_determine_right_triangle_l1471_147150

theorem determine_right_triangle (a b c : ℕ) :
  (∀ c b, (c + b) * (c - b) = a^2 → c^2 = a^2 + b^2) ∧
  (∀ A B C, A + B = C → C = 90) ∧
  (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 → a^2 + b^2 ≠ c^2) ∧
  (a = 5 ∧ b = 12 ∧ c = 13 → a^2 + b^2 = c^2) → 
  ( ∃ x y z : ℕ, x = a ∧ y = b ∧ z = c ∧ x^2 + y^2 ≠ z^2 )
:= by
  sorry

end NUMINAMATH_GPT_determine_right_triangle_l1471_147150


namespace NUMINAMATH_GPT_quadratic_intersection_with_x_axis_l1471_147124

theorem quadratic_intersection_with_x_axis :
  ∃ x : ℝ, (x^2 - 4*x + 4 = 0) ∧ (x = 2) ∧ (x, 0) = (2, 0) :=
sorry

end NUMINAMATH_GPT_quadratic_intersection_with_x_axis_l1471_147124


namespace NUMINAMATH_GPT_find_directrix_of_parabola_l1471_147158

open Real

theorem find_directrix_of_parabola (O : ℝ × ℝ) (p : ℝ) (F P Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hp_pos : p > 0)
  (hC : ∀ x y, (x, y) = P → y^2 = 2 * p * x)
  (hF : F = (p / 2, 0))
  (hPF_perpendicular_to_x : P.1 = p / 2 ∧ P.2 = p)
  (hQ_on_x_axis : Q.2 = 0)
  (hPQ_perpendicular_OP : (P.1, P.2) ≠ Q ∧ ((P.2 - Q.2) / (P.1 - Q.1) = -1 / ((P.2 - O.2) / (P.1 - O.1))))
  (hFQ_distance : abs (F.1 - Q.1) = 6) :
  x = -3 / 2 :=
sorry

end NUMINAMATH_GPT_find_directrix_of_parabola_l1471_147158


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1471_147174

theorem quadratic_has_distinct_real_roots : 
  ∀ (x : ℝ), x^2 - 3 * x + 1 = 0 → ∀ (a b c : ℝ), a = 1 ∧ b = -3 ∧ c = 1 →
  (b^2 - 4 * a * c) > 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1471_147174


namespace NUMINAMATH_GPT_fem_current_age_l1471_147109

theorem fem_current_age (F : ℕ) 
  (h1 : ∃ M : ℕ, M = 4 * F) 
  (h2 : (F + 2) + (4 * F + 2) = 59) : 
  F = 11 :=
sorry

end NUMINAMATH_GPT_fem_current_age_l1471_147109


namespace NUMINAMATH_GPT_simplify_expression_inequality_solution_l1471_147145

-- Simplification part
theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2):
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10 * x + 25) / (x^2 - 4)) = 
  (x - 2) / (x + 5) :=
sorry

-- Inequality system part
theorem inequality_solution (x : ℝ):
  (2 * x + 7 > 3) ∧ ((x + 1) / 3 > (x - 1) / 2) → -2 < x ∧ x < 5 :=
sorry

end NUMINAMATH_GPT_simplify_expression_inequality_solution_l1471_147145


namespace NUMINAMATH_GPT_rational_power_sum_l1471_147176

theorem rational_power_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = - b) : a ^ 2007 + b ^ 2007 = 1 ∨ a ^ 2007 + b ^ 2007 = -1 := by
  sorry

end NUMINAMATH_GPT_rational_power_sum_l1471_147176


namespace NUMINAMATH_GPT_find_f_neg_l1471_147160

noncomputable def f (a b x : ℝ) := a * x^3 + b * x - 2

theorem find_f_neg (a b : ℝ) (f_2017 : f a b 2017 = 7) : f a b (-2017) = -11 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_l1471_147160


namespace NUMINAMATH_GPT_singing_only_pupils_l1471_147166

theorem singing_only_pupils (total_pupils debate_only both : ℕ) (h1 : total_pupils = 55) (h2 : debate_only = 10) (h3 : both = 17) :
  total_pupils - debate_only = 45 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_singing_only_pupils_l1471_147166


namespace NUMINAMATH_GPT_volume_multiplication_factor_l1471_147112

variables (r h : ℝ) (π : ℝ := Real.pi)

def original_volume : ℝ := π * r^2 * h
def new_height : ℝ := 3 * h
def new_radius : ℝ := 2.5 * r
def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

theorem volume_multiplication_factor :
  new_volume r h / original_volume r h = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_volume_multiplication_factor_l1471_147112


namespace NUMINAMATH_GPT_laps_needed_to_reach_total_distance_l1471_147115

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end NUMINAMATH_GPT_laps_needed_to_reach_total_distance_l1471_147115


namespace NUMINAMATH_GPT_avg_visitors_proof_l1471_147189

-- Define the constants and conditions
def Sundays_visitors : ℕ := 500
def total_days : ℕ := 30
def avg_visitors_per_day : ℕ := 200

-- Total visits on Sundays within the month
def visits_on_Sundays := 5 * Sundays_visitors

-- Total visitors for the month
def total_visitors := total_days * avg_visitors_per_day

-- Average visitors on other days (Monday to Saturday)
def avg_visitors_other_days : ℕ :=
  (total_visitors - visits_on_Sundays) / (total_days - 5)

-- The theorem stating the problem and corresponding answer
theorem avg_visitors_proof (V : ℕ) 
  (h1 : Sundays_visitors = 500)
  (h2 : total_days = 30)
  (h3 : avg_visitors_per_day = 200)
  (h4 : visits_on_Sundays = 5 * Sundays_visitors)
  (h5 : total_visitors = total_days * avg_visitors_per_day)
  (h6 : avg_visitors_other_days = (total_visitors - visits_on_Sundays) / (total_days - 5))
  : V = 140 :=
by
  -- Proof is not required, just state the theorem
  sorry

end NUMINAMATH_GPT_avg_visitors_proof_l1471_147189


namespace NUMINAMATH_GPT_geometric_sequence_product_l1471_147149

/-- Given a geometric sequence with positive terms where a_3 = 3 and a_6 = 1/9,
    prove that a_4 * a_5 = 1/3. -/
theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
    (h_geometric : ∀ n, a (n + 1) = a n * q) (ha3 : a 3 = 3) (ha6 : a 6 = 1 / 9) :
  a 4 * a 5 = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1471_147149


namespace NUMINAMATH_GPT_g_value_at_6_l1471_147159

noncomputable def g (v : ℝ) : ℝ :=
  let x := (v + 2) / 4
  x^2 - x + 2

theorem g_value_at_6 :
  g 6 = 4 := by
  sorry

end NUMINAMATH_GPT_g_value_at_6_l1471_147159


namespace NUMINAMATH_GPT_arithmetic_sequence_term_count_l1471_147104

theorem arithmetic_sequence_term_count (a d l n : ℕ) (h1 : a = 11) (h2 : d = 4) (h3 : l = 107) :
  l = a + (n - 1) * d → n = 25 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_count_l1471_147104


namespace NUMINAMATH_GPT_inequality_convex_l1471_147195

theorem inequality_convex (x y a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : a + b = 1) : 
  (a * x + b * y) ^ 2 ≤ a * x ^ 2 + b * y ^ 2 := 
sorry

end NUMINAMATH_GPT_inequality_convex_l1471_147195


namespace NUMINAMATH_GPT_gg3_eq_585_over_368_l1471_147187

def g (x : ℚ) : ℚ := 2 * x⁻¹ + (2 * x⁻¹) / (1 + 2 * x⁻¹)

theorem gg3_eq_585_over_368 : g (g 3) = 585 / 368 := 
  sorry

end NUMINAMATH_GPT_gg3_eq_585_over_368_l1471_147187


namespace NUMINAMATH_GPT_incorrect_statement_among_options_l1471_147113

/- Definitions and Conditions -/
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 1) + (n * (n - 1) / 2) * d

/- Conditions given in the problem -/
axiom S_6_gt_S_7 : S 6 > S 7
axiom S_7_gt_S_5 : S 7 > S 5

/- Incorrect statement to be proved -/
theorem incorrect_statement_among_options :
  ¬ (∀ n, S n ≤ S 11) := sorry

end NUMINAMATH_GPT_incorrect_statement_among_options_l1471_147113


namespace NUMINAMATH_GPT_frac_pow_eq_l1471_147179

theorem frac_pow_eq : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by 
  sorry

end NUMINAMATH_GPT_frac_pow_eq_l1471_147179


namespace NUMINAMATH_GPT_part1_part2_l1471_147192

theorem part1 (x : ℝ) : |x + 3| - 2 * x - 1 < 0 → 2 < x :=
by sorry

theorem part2 (m : ℝ) : (m > 0) →
  (∃ x : ℝ, |x - m| + |x + 1/m| = 2) → m = 1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1471_147192


namespace NUMINAMATH_GPT_events_are_mutually_exclusive_but_not_opposite_l1471_147136

-- Definitions based on the conditions:
structure BallBoxConfig where
  ball1 : Fin 4 → ℕ     -- Function representing the placement of ball number 1 into one of the 4 boxes
  h_distinct : ∀ i j, i ≠ j → ball1 i ≠ ball1 j

def event_A (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 1
def event_B (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 2

-- The proof problem:
theorem events_are_mutually_exclusive_but_not_opposite (cfg : BallBoxConfig) :
  (event_A cfg ∨ event_B cfg) ∧ ¬ (event_A cfg ∧ event_B cfg) :=
sorry

end NUMINAMATH_GPT_events_are_mutually_exclusive_but_not_opposite_l1471_147136


namespace NUMINAMATH_GPT_range_of_m_l1471_147108

theorem range_of_m (m : Real) :
  (∀ x y : Real, 0 < x ∧ x < y ∧ y < (π / 2) → 
    (m - 2 * Real.sin x) / Real.cos x > (m - 2 * Real.sin y) / Real.cos y) →
  m ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1471_147108


namespace NUMINAMATH_GPT_combined_distance_20_birds_two_seasons_l1471_147138

theorem combined_distance_20_birds_two_seasons :
  let distance_jim_to_disney := 50
  let distance_disney_to_london := 60
  let number_of_birds := 20
  (number_of_birds * (distance_jim_to_disney + distance_disney_to_london)) = 2200 := by
  sorry

end NUMINAMATH_GPT_combined_distance_20_birds_two_seasons_l1471_147138


namespace NUMINAMATH_GPT_pyramid_volume_correct_l1471_147114

noncomputable def volume_of_pyramid (l α β : ℝ) (Hα : α = π/8) (Hβ : β = π/4) :=
  (1 / 3) * (l^3 / 24) * Real.sqrt (Real.sqrt 2 + 1)

theorem pyramid_volume_correct :
  ∀ (l : ℝ), l = 6 → volume_of_pyramid l (π/8) (π/4) (rfl) (rfl) = 9 * Real.sqrt (Real.sqrt 2 + 1) :=
by
  intros l hl
  rw [hl]
  norm_num
  sorry

end NUMINAMATH_GPT_pyramid_volume_correct_l1471_147114


namespace NUMINAMATH_GPT_area_of_square_with_adjacent_points_l1471_147196

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end NUMINAMATH_GPT_area_of_square_with_adjacent_points_l1471_147196


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l1471_147199

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3 / 4) = 2 * s) : 3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l1471_147199


namespace NUMINAMATH_GPT_cells_sequence_exists_l1471_147110

theorem cells_sequence_exists :
  ∃ (a : Fin 10 → ℚ), 
    a 0 = 9 ∧
    a 8 = 5 ∧
    (∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14) :=
sorry

end NUMINAMATH_GPT_cells_sequence_exists_l1471_147110


namespace NUMINAMATH_GPT_chess_game_probability_l1471_147151

theorem chess_game_probability (p_A_wins p_draw : ℝ) (h1 : p_A_wins = 0.3) (h2 : p_draw = 0.2) :
  p_A_wins + p_draw = 0.5 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_chess_game_probability_l1471_147151


namespace NUMINAMATH_GPT_arithmetic_seq_a6_l1471_147185

variable (a : ℕ → ℝ)

-- Conditions
axiom a3 : a 3 = 16
axiom a9 : a 9 = 80

-- Theorem to prove
theorem arithmetic_seq_a6 : a 6 = 48 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a6_l1471_147185


namespace NUMINAMATH_GPT_fred_gave_sandy_balloons_l1471_147139

theorem fred_gave_sandy_balloons :
  ∀ (original_balloons given_balloons final_balloons : ℕ),
    original_balloons = 709 →
    final_balloons = 488 →
    given_balloons = original_balloons - final_balloons →
    given_balloons = 221 := by
  sorry

end NUMINAMATH_GPT_fred_gave_sandy_balloons_l1471_147139


namespace NUMINAMATH_GPT_swimmer_speed_proof_l1471_147186

-- Definition of the conditions
def current_speed : ℝ := 2
def swimming_time : ℝ := 1.5
def swimming_distance : ℝ := 3

-- Prove: Swimmer's speed in still water
def swimmer_speed_in_still_water : ℝ := 4

-- Statement: Given the conditions, the swimmer's speed in still water equals 4 km/h
theorem swimmer_speed_proof :
  (swimming_distance = (swimmer_speed_in_still_water - current_speed) * swimming_time) →
  swimmer_speed_in_still_water = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_swimmer_speed_proof_l1471_147186


namespace NUMINAMATH_GPT_courier_problem_l1471_147157

variable (x : ℝ) -- Let x represent the specified time in minutes
variable (d : ℝ) -- Let d represent the total distance traveled in km

theorem courier_problem
  (h1 : 1.2 * (x - 10) = d)
  (h2 : 0.8 * (x + 5) = d) :
  x = 40 ∧ d = 36 :=
by
  -- This theorem statement encapsulates the conditions and the answer.
  sorry

end NUMINAMATH_GPT_courier_problem_l1471_147157


namespace NUMINAMATH_GPT_oliver_earning_correct_l1471_147183

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end NUMINAMATH_GPT_oliver_earning_correct_l1471_147183


namespace NUMINAMATH_GPT_cost_effectiveness_l1471_147162

-- Define general parameters and conditions given in the problem
def a : ℕ := 70 -- We use 70 since it must be greater than 50

-- Define the scenarios
def cost_scenario1 (a: ℕ) : ℕ := 4500 + 27 * a
def cost_scenario2 (a: ℕ) : ℕ := 4400 + 30 * a

-- The theorem to be proven
theorem cost_effectiveness (h : a > 50) : cost_scenario1 a < cost_scenario2 a :=
  by
  -- First, let's replace a with 70 (this step is unnecessary in the proof since a = 70 is fixed)
  let a := 70
  -- Now, prove the inequality
  sorry

end NUMINAMATH_GPT_cost_effectiveness_l1471_147162


namespace NUMINAMATH_GPT_problem_statement_l1471_147184

variables {α β : Plane} {m : Line}

def parallel (a b : Plane) : Prop := sorry
def perpendicular (m : Line) (π : Plane) : Prop := sorry

axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_trans {m : Line} {a b : Plane} : perpendicular m a → parallel a b → perpendicular m b

theorem problem_statement (h1 : parallel α β) (h2 : perpendicular m α) : perpendicular m β :=
  perpendicular_trans h2 (parallel_symm h1)

end NUMINAMATH_GPT_problem_statement_l1471_147184


namespace NUMINAMATH_GPT_factorial_div_sub_factorial_equality_l1471_147177

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n + 1) * factorial n

theorem factorial_div_sub_factorial_equality :
  (factorial 12 - factorial 11) / factorial 10 = 121 :=
by
  sorry

end NUMINAMATH_GPT_factorial_div_sub_factorial_equality_l1471_147177


namespace NUMINAMATH_GPT_x_mul_y_eq_4_l1471_147131

theorem x_mul_y_eq_4 (x y z w : ℝ) (hw_pos : w > 0) 
  (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) 
  (h4 : y = w) (h5 : z = 3) (h6 : w + w = w * w) : 
  x * y = 4 := by
  sorry

end NUMINAMATH_GPT_x_mul_y_eq_4_l1471_147131


namespace NUMINAMATH_GPT_ella_savings_l1471_147181

theorem ella_savings
  (initial_cost_per_lamp : ℝ)
  (num_lamps : ℕ)
  (discount_rate : ℝ)
  (additional_discount : ℝ)
  (initial_total_cost : ℝ := num_lamps * initial_cost_per_lamp)
  (discounted_lamp_cost : ℝ := initial_cost_per_lamp - (initial_cost_per_lamp * discount_rate))
  (total_cost_with_discount : ℝ := num_lamps * discounted_lamp_cost)
  (total_cost_after_additional_discount : ℝ := total_cost_with_discount - additional_discount) :
  initial_cost_per_lamp = 15 →
  num_lamps = 3 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  initial_total_cost - total_cost_after_additional_discount = 16.25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ella_savings_l1471_147181


namespace NUMINAMATH_GPT_gcd_153_119_l1471_147103

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end NUMINAMATH_GPT_gcd_153_119_l1471_147103


namespace NUMINAMATH_GPT_intersection_A_B_l1471_147126

-- Defining sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1471_147126


namespace NUMINAMATH_GPT_positive_expressions_l1471_147146

-- Define the approximate values for A, B, C, D, and E.
def A := 2.5
def B := -2.1
def C := -0.3
def D := 1.0
def E := -0.7

-- Define the expressions that we need to prove as positive numbers.
def exprA := A + B
def exprB := B * C
def exprD := E / (A * B)

-- The theorem states that expressions (A + B), (B * C), and (E / (A * B)) are positive.
theorem positive_expressions : exprA > 0 ∧ exprB > 0 ∧ exprD > 0 := 
by sorry

end NUMINAMATH_GPT_positive_expressions_l1471_147146


namespace NUMINAMATH_GPT_present_population_l1471_147175

theorem present_population (P : ℕ) (h1 : P * 11 / 10 = 264) : P = 240 :=
by sorry

end NUMINAMATH_GPT_present_population_l1471_147175


namespace NUMINAMATH_GPT_jesse_rooms_l1471_147155

theorem jesse_rooms:
  ∀ (l w A n: ℕ), 
  l = 19 ∧ 
  w = 18 ∧ 
  A = 6840 ∧ 
  n = A / (l * w) → 
  n = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jesse_rooms_l1471_147155


namespace NUMINAMATH_GPT_gears_together_again_l1471_147198

theorem gears_together_again (r₁ r₂ : ℕ) (h₁ : r₁ = 3) (h₂ : r₂ = 5) : 
  (∃ t : ℕ, t = Nat.lcm r₁ r₂ / r₁ ∨ t = Nat.lcm r₁ r₂ / r₂) → 5 = Nat.lcm r₁ r₂ / min r₁ r₂ := 
by
  sorry

end NUMINAMATH_GPT_gears_together_again_l1471_147198


namespace NUMINAMATH_GPT_non_similar_triangles_with_arithmetic_angles_l1471_147171

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end NUMINAMATH_GPT_non_similar_triangles_with_arithmetic_angles_l1471_147171


namespace NUMINAMATH_GPT_smallest_multiple_36_45_not_11_l1471_147140

theorem smallest_multiple_36_45_not_11 (n : ℕ) :
  (n = 180) ↔ (n > 0 ∧ (36 ∣ n) ∧ (45 ∣ n) ∧ ¬ (11 ∣ n)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_36_45_not_11_l1471_147140


namespace NUMINAMATH_GPT_mechanic_worked_hours_l1471_147169

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end NUMINAMATH_GPT_mechanic_worked_hours_l1471_147169


namespace NUMINAMATH_GPT_problem_solution_l1471_147173

theorem problem_solution :
  (2200 - 2089)^2 / 196 = 63 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1471_147173


namespace NUMINAMATH_GPT_meeting_equation_correct_l1471_147168

-- Define the conditions
def distance : ℝ := 25
def time : ℝ := 3
def speed_Xiaoming : ℝ := 4
def speed_Xiaogang (x : ℝ) : ℝ := x

-- The target equation derived from conditions which we need to prove valid.
theorem meeting_equation_correct (x : ℝ) : 3 * (speed_Xiaoming + speed_Xiaogang x) = distance :=
by
  sorry

end NUMINAMATH_GPT_meeting_equation_correct_l1471_147168


namespace NUMINAMATH_GPT_total_time_before_main_game_l1471_147141

-- Define the time spent on each activity according to the conditions
def download_time := 10
def install_time := download_time / 2
def update_time := 2 * download_time
def account_time := 5
def internet_issues_time := 15
def discussion_time := 20
def video_time := 8

-- Define the total preparation time
def preparation_time := download_time + install_time + update_time + account_time + internet_issues_time + discussion_time + video_time

-- Define the in-game tutorial time
def tutorial_time := 3 * preparation_time

-- Prove that the total time before playing the main game is 332 minutes
theorem total_time_before_main_game : preparation_time + tutorial_time = 332 := by
  -- Provide a detailed proof here
  sorry

end NUMINAMATH_GPT_total_time_before_main_game_l1471_147141


namespace NUMINAMATH_GPT_Jake_weight_196_l1471_147134

def Jake_and_Sister : Prop :=
  ∃ (J S : ℕ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196)

theorem Jake_weight_196 : Jake_and_Sister :=
by
  sorry

end NUMINAMATH_GPT_Jake_weight_196_l1471_147134


namespace NUMINAMATH_GPT_unbroken_seashells_l1471_147191

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) (h1 : total_seashells = 23) (h2 : broken_seashells = 11) : total_seashells - broken_seashells = 12 := by
  sorry

end NUMINAMATH_GPT_unbroken_seashells_l1471_147191


namespace NUMINAMATH_GPT_rectangle_sides_l1471_147161

theorem rectangle_sides (S d : ℝ) (a b : ℝ) : 
  a = Real.sqrt (S + d^2 / 4) + d / 2 ∧ 
  b = Real.sqrt (S + d^2 / 4) - d / 2 →
  S = a * b ∧ d = a - b :=
by
  -- definitions and conditions will be used here in the proofs
  sorry

end NUMINAMATH_GPT_rectangle_sides_l1471_147161


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1471_147193

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) -- sequence a_n
  (r : ℝ) -- common ratio
  (h1 : r = 2) -- given common ratio
  (h2 : a 4 = 16) -- given a_4 = 16
  (h3 : ∀ n, a n = a 1 * r^(n-1)) -- definition of geometric sequence
  : a 1 = 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1471_147193


namespace NUMINAMATH_GPT_find_s_l1471_147144

variable {a b n r s : ℝ}

theorem find_s (h1 : Polynomial.aeval a (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h2 : Polynomial.aeval b (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h_ab : a * b = 6)
              (h_roots : Polynomial.aeval (a + 2/b) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0)
              (h_roots2 : Polynomial.aeval (b + 2/a) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0) :
  s = 32/3 := 
sorry

end NUMINAMATH_GPT_find_s_l1471_147144


namespace NUMINAMATH_GPT_simplify_expression_l1471_147118

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1471_147118


namespace NUMINAMATH_GPT_car_distance_ratio_l1471_147117

theorem car_distance_ratio (speed_A time_A speed_B time_B : ℕ) 
  (hA : speed_A = 70) (hTA : time_A = 10) 
  (hB : speed_B = 35) (hTB : time_B = 10) : 
  (speed_A * time_A) / gcd (speed_A * time_A) (speed_B * time_B) = 2 :=
by
  sorry

end NUMINAMATH_GPT_car_distance_ratio_l1471_147117


namespace NUMINAMATH_GPT_total_shaded_area_of_rectangles_l1471_147188

theorem total_shaded_area_of_rectangles (w1 l1 w2 l2 ow ol : ℕ) 
  (h1 : w1 = 4) (h2 : l1 = 12) (h3 : w2 = 5) (h4 : l2 = 10) (h5 : ow = 4) (h6 : ol = 5) :
  (w1 * l1 + w2 * l2 - ow * ol = 78) :=
by
  sorry

end NUMINAMATH_GPT_total_shaded_area_of_rectangles_l1471_147188


namespace NUMINAMATH_GPT_find_x1_l1471_147128

noncomputable def parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem find_x1 
  (a h k m x1 : ℝ)
  (h1 : parabola a h k (-1) = 2)
  (h2 : parabola a h k 1 = -2)
  (h3 : parabola a h k 3 = 2)
  (h4 : parabola a h k (-2) = m)
  (h5 : parabola a h k x1 = m) :
  x1 = 4 := 
sorry

end NUMINAMATH_GPT_find_x1_l1471_147128


namespace NUMINAMATH_GPT_range_of_m_for_inequality_l1471_147137

theorem range_of_m_for_inequality (m : Real) : 
  (∀ (x : Real), 1 < x ∧ x < 2 → x^2 + m * x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end NUMINAMATH_GPT_range_of_m_for_inequality_l1471_147137


namespace NUMINAMATH_GPT_sqrt_domain_l1471_147120

theorem sqrt_domain (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
sorry

end NUMINAMATH_GPT_sqrt_domain_l1471_147120


namespace NUMINAMATH_GPT_elberta_amount_l1471_147154

theorem elberta_amount (grannySmith_amount : ℝ) (Anjou_factor : ℝ) (extra_amount : ℝ) :
  grannySmith_amount = 45 →
  Anjou_factor = 1 / 4 →
  extra_amount = 4 →
  (extra_amount + Anjou_factor * grannySmith_amount) = 15.25 :=
by
  intros h_grannySmith h_AnjouFactor h_extraAmount
  sorry

end NUMINAMATH_GPT_elberta_amount_l1471_147154


namespace NUMINAMATH_GPT_cricket_run_rate_l1471_147107

theorem cricket_run_rate 
  (run_rate_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_played : ℕ)
  (remaining_overs : ℕ)
  (correct_run_rate : ℝ)
  (h1 : run_rate_10_overs = 3.6)
  (h2 : target_runs = 282)
  (h3 : overs_played = 10)
  (h4 : remaining_overs = 40)
  (h5 : correct_run_rate = 6.15) :
  (target_runs - run_rate_10_overs * overs_played) / remaining_overs = correct_run_rate :=
sorry

end NUMINAMATH_GPT_cricket_run_rate_l1471_147107


namespace NUMINAMATH_GPT_determine_digits_l1471_147190

def product_consecutive_eq_120_times_ABABAB (n A B : ℕ) : Prop :=
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 * (A * 101010101 + B * 10101010 + A * 1010101 + B * 101010 + A * 10101 + B * 1010 + A * 101 + B * 10 + A)

theorem determine_digits (A B : ℕ) (h : ∃ n, product_consecutive_eq_120_times_ABABAB n A B):
  A = 5 ∧ B = 7 :=
sorry

end NUMINAMATH_GPT_determine_digits_l1471_147190


namespace NUMINAMATH_GPT_find_a_plus_b_l1471_147130

theorem find_a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = a * x + b) (h2 : ∀ x, g x = 3 * x - 4) 
(h3 : ∀ x, g (f x) = 4 * x + 5) : a + b = 13 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1471_147130


namespace NUMINAMATH_GPT_functional_relationship_find_selling_price_maximum_profit_l1471_147167

noncomputable def linear_relation (x : ℤ) : ℤ := -5 * x + 150
def profit_function (x : ℤ) : ℤ := -5 * x * x + 200 * x - 1500

theorem functional_relationship (x : ℤ) (hx : 10 ≤ x ∧ x ≤ 15) : linear_relation x = -5 * x + 150 :=
by sorry

theorem find_selling_price (h : ∃ x : ℤ, (10 ≤ x ∧ x ≤ 15) ∧ ((-5 * x + 150) * (x - 10) = 320)) :
  ∃ x : ℤ, x = 14 :=
by sorry

theorem maximum_profit (hx : 10 ≤ 15 ∧ 15 ≤ 15) : profit_function 15 = 375 :=
by sorry

end NUMINAMATH_GPT_functional_relationship_find_selling_price_maximum_profit_l1471_147167


namespace NUMINAMATH_GPT_minimum_bailing_rate_l1471_147125

theorem minimum_bailing_rate
  (distance : ℝ) (to_shore_rate : ℝ) (water_in_rate : ℝ) (submerge_limit : ℝ) (r : ℝ)
  (h_distance : distance = 0.5) 
  (h_speed : to_shore_rate = 6) 
  (h_water_intake : water_in_rate = 12) 
  (h_submerge_limit : submerge_limit = 50)
  (h_time : (distance / to_shore_rate) * 60 = 5)
  (h_total_intake : water_in_rate * 5 = 60)
  (h_max_intake : submerge_limit - 60 = -10) :
  r = 2 := sorry

end NUMINAMATH_GPT_minimum_bailing_rate_l1471_147125


namespace NUMINAMATH_GPT_calculation_result_l1471_147152

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l1471_147152


namespace NUMINAMATH_GPT_same_terminal_side_of_minus_80_l1471_147121

theorem same_terminal_side_of_minus_80 :
  ∃ k : ℤ, 1 * 360 - 80 = 280 := 
  sorry

end NUMINAMATH_GPT_same_terminal_side_of_minus_80_l1471_147121


namespace NUMINAMATH_GPT_candidate_percentage_l1471_147143

theorem candidate_percentage (P : ℚ) (votes_cast : ℚ) (loss : ℚ)
  (h1 : votes_cast = 2000) 
  (h2 : loss = 640) 
  (h3 : (P / 100) * votes_cast + (P / 100) * votes_cast + loss = votes_cast) :
  P = 34 :=
by 
  sorry

end NUMINAMATH_GPT_candidate_percentage_l1471_147143


namespace NUMINAMATH_GPT_triangles_needed_for_hexagon_with_perimeter_19_l1471_147194

def num_triangles_to_construct_hexagon (perimeter : ℕ) : ℕ :=
  match perimeter with
  | 19 => 59
  | _ => 0  -- We handle only the case where perimeter is 19

theorem triangles_needed_for_hexagon_with_perimeter_19 :
  num_triangles_to_construct_hexagon 19 = 59 :=
by
  -- Here we assert that the number of triangles to construct the hexagon with perimeter 19 is 59
  sorry

end NUMINAMATH_GPT_triangles_needed_for_hexagon_with_perimeter_19_l1471_147194


namespace NUMINAMATH_GPT_ratio_of_ages_l1471_147142

variables (X Y : ℕ)

theorem ratio_of_ages (h1 : X - 6 = 24) (h2 : X + Y = 36) : X / Y = 2 :=
by 
  have h3 : X = 30 - 6 := by sorry
  have h4 : X = 24 := by sorry
  have h5 : X + Y = 36 := by sorry
  have h6 : Y = 12 := by sorry
  have h7 : X / Y = 2 := by sorry
  exact h7

end NUMINAMATH_GPT_ratio_of_ages_l1471_147142


namespace NUMINAMATH_GPT_inequality_always_holds_true_l1471_147153

theorem inequality_always_holds_true (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) :
  (a / (c^2 + 1)) > (b / (c^2 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_holds_true_l1471_147153


namespace NUMINAMATH_GPT_shaded_percentage_of_grid_l1471_147172

def percent_shaded (total_squares shaded_squares : ℕ) : ℚ :=
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100

theorem shaded_percentage_of_grid :
  percent_shaded 36 16 = 44.44 :=
by 
  sorry

end NUMINAMATH_GPT_shaded_percentage_of_grid_l1471_147172


namespace NUMINAMATH_GPT_present_age_of_B_l1471_147164

-- Definitions
variables (a b : ℕ)

-- Conditions
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 7

-- Theorem to prove
theorem present_age_of_B (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 37 := by
  sorry

end NUMINAMATH_GPT_present_age_of_B_l1471_147164


namespace NUMINAMATH_GPT_a_can_finish_remaining_work_in_5_days_l1471_147165

theorem a_can_finish_remaining_work_in_5_days (a_work_rate b_work_rate : ℝ) (total_days_b_works : ℝ):
  a_work_rate = 1/15 → 
  b_work_rate = 1/15 → 
  total_days_b_works = 10 → 
  ∃ (remaining_days_for_a : ℝ), remaining_days_for_a = 5 :=
by
  intros h1 h2 h3
  -- We are skipping the proof itself
  sorry

end NUMINAMATH_GPT_a_can_finish_remaining_work_in_5_days_l1471_147165


namespace NUMINAMATH_GPT_cube_volume_is_216_l1471_147170

-- Define the conditions
def total_edge_length : ℕ := 72
def num_edges_of_cube : ℕ := 12

-- The side length of the cube can be calculated as
def side_length (E : ℕ) (n : ℕ) : ℕ := E / n

-- The volume of the cube is the cube of its side length
def volume (s : ℕ) : ℕ := s ^ 3

theorem cube_volume_is_216 (E : ℕ) (n : ℕ) (V : ℕ) 
  (hE : E = total_edge_length) 
  (hn : n = num_edges_of_cube) 
  (hv : V = volume (side_length E n)) : 
  V = 216 := by
  sorry

end NUMINAMATH_GPT_cube_volume_is_216_l1471_147170


namespace NUMINAMATH_GPT_amplitude_combined_wave_l1471_147163

noncomputable def y1 (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y1 t + y2 t
noncomputable def amplitude : ℝ := 3 * Real.sqrt 5

theorem amplitude_combined_wave : ∀ t : ℝ, ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  intro t
  use amplitude
  exact sorry

end NUMINAMATH_GPT_amplitude_combined_wave_l1471_147163


namespace NUMINAMATH_GPT_inscribed_rectangle_area_l1471_147102

theorem inscribed_rectangle_area (h a b x : ℝ) (ha_gt_b : a > b) :
  ∃ A : ℝ, A = (b * x / h) * (h - x) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_rectangle_area_l1471_147102


namespace NUMINAMATH_GPT_student_total_marks_l1471_147100

theorem student_total_marks (total_questions correct_answers incorrect_mark correct_mark : ℕ) 
                             (H1 : total_questions = 60) 
                             (H2 : correct_answers = 34)
                             (H3 : incorrect_mark = 1)
                             (H4 : correct_mark = 4) :
  ((correct_answers * correct_mark) - ((total_questions - correct_answers) * incorrect_mark)) = 110 := 
by {
  -- The proof goes here.
  sorry
}

end NUMINAMATH_GPT_student_total_marks_l1471_147100
