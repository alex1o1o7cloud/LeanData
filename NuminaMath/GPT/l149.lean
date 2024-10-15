import Mathlib

namespace NUMINAMATH_GPT_game_winning_strategy_l149_14999

theorem game_winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 1) ∧ (n % 2 = 1 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 2) :=
by
  sorry

end NUMINAMATH_GPT_game_winning_strategy_l149_14999


namespace NUMINAMATH_GPT_percentage_problem_l149_14901

variable (N P : ℝ)

theorem percentage_problem (h1 : 0.3 * N = 120) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l149_14901


namespace NUMINAMATH_GPT_original_gift_card_value_l149_14992

def gift_card_cost_per_pound : ℝ := 8.58
def coffee_pounds_bought : ℕ := 4
def remaining_balance_after_purchase : ℝ := 35.68

theorem original_gift_card_value :
  (remaining_balance_after_purchase + coffee_pounds_bought * gift_card_cost_per_pound) = 70.00 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_original_gift_card_value_l149_14992


namespace NUMINAMATH_GPT_polynomial_sum_l149_14981

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l149_14981


namespace NUMINAMATH_GPT_total_apples_l149_14938

variable (A : ℕ)
variables (too_small not_ripe perfect : ℕ)

-- Conditions
axiom small_fraction : too_small = A / 6
axiom ripe_fraction  : not_ripe = A / 3
axiom remaining_fraction : perfect = A / 2
axiom perfect_count : perfect = 15

theorem total_apples : A = 30 :=
sorry

end NUMINAMATH_GPT_total_apples_l149_14938


namespace NUMINAMATH_GPT_quadratic_solution_l149_14989

theorem quadratic_solution (x : ℝ) : (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l149_14989


namespace NUMINAMATH_GPT_output_correct_l149_14910

-- Definitions derived from the conditions
def initial_a : Nat := 3
def initial_b : Nat := 4

-- Proof that the final output of PRINT a, b is (4, 4)
theorem output_correct : 
  let a := initial_a;
  let b := initial_b;
  let a := b;
  let b := a;
  (a, b) = (4, 4) :=
by
  sorry

end NUMINAMATH_GPT_output_correct_l149_14910


namespace NUMINAMATH_GPT_proof_goats_minus_pigs_l149_14906

noncomputable def number_of_goats : ℕ := 66
noncomputable def number_of_chickens : ℕ := 2 * number_of_goats - 10
noncomputable def number_of_ducks : ℕ := (number_of_goats + number_of_chickens) / 2
noncomputable def number_of_pigs : ℕ := number_of_ducks / 3
noncomputable def number_of_rabbits : ℕ := Nat.floor (Real.sqrt (2 * number_of_ducks - number_of_pigs))
noncomputable def number_of_cows : ℕ := number_of_rabbits ^ number_of_pigs / Nat.factorial (number_of_goats / 2)

theorem proof_goats_minus_pigs : number_of_goats - number_of_pigs = 35 := by
  sorry

end NUMINAMATH_GPT_proof_goats_minus_pigs_l149_14906


namespace NUMINAMATH_GPT_magic_grid_product_l149_14970

theorem magic_grid_product (p q r s t x : ℕ) 
  (h1: p * 6 * 3 = q * r * s)
  (h2: p * q * t = 6 * r * 2)
  (h3: p * r * x = 6 * 2 * t)
  (h4: q * 2 * 3 = r * s * x)
  (h5: t * 2 * x = 6 * s * 3)
  (h6: 6 * q * 3 = r * s * t)
  (h7: p * r * s = 6 * 2 * q)
  : x = 36 := 
by
  sorry

end NUMINAMATH_GPT_magic_grid_product_l149_14970


namespace NUMINAMATH_GPT_ROI_difference_is_correct_l149_14957

noncomputable def compound_interest (P : ℝ) (rates : List ℝ) : ℝ :=
rates.foldl (λ acc rate => acc * (1 + rate)) P

noncomputable def Emma_investment := compound_interest 300 [0.15, 0.12, 0.18]

noncomputable def Briana_investment := compound_interest 500 [0.10, 0.08, 0.14]

noncomputable def ROI_difference := Briana_investment - Emma_investment

theorem ROI_difference_is_correct : ROI_difference = 220.808 := 
sorry

end NUMINAMATH_GPT_ROI_difference_is_correct_l149_14957


namespace NUMINAMATH_GPT_domain_of_sqrt_fn_l149_14920

theorem domain_of_sqrt_fn : {x : ℝ | -2 ≤ x ∧ x ≤ 2} = {x : ℝ | 4 - x^2 ≥ 0} := 
by sorry

end NUMINAMATH_GPT_domain_of_sqrt_fn_l149_14920


namespace NUMINAMATH_GPT_total_biscuits_needed_l149_14964

-- Definitions
def number_of_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem statement
theorem total_biscuits_needed : number_of_dogs * biscuits_per_dog = 6 :=
by sorry

end NUMINAMATH_GPT_total_biscuits_needed_l149_14964


namespace NUMINAMATH_GPT_john_weekly_allowance_l149_14954

noncomputable def weekly_allowance (A : ℝ) :=
  (3/5) * A + (1/3) * ((2/5) * A) + 0.60 <= A

theorem john_weekly_allowance : ∃ A : ℝ, (3/5) * A + (1/3) * ((2/5) * A) + 0.60 = A := by
  let A := 2.25
  sorry

end NUMINAMATH_GPT_john_weekly_allowance_l149_14954


namespace NUMINAMATH_GPT_positive_solutions_l149_14930

theorem positive_solutions (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔
  x = 1 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_solutions_l149_14930


namespace NUMINAMATH_GPT_find_A_of_trig_max_bsquared_plus_csquared_l149_14987

-- Given the geometric conditions and trigonometric identities.

-- Prove: Given 2a * sin B = b * tan A, we have A = π / 3
theorem find_A_of_trig (a b c A B C : Real) (h1 : 2 * a * Real.sin B = b * Real.tan A) :
  A = Real.pi / 3 := sorry

-- Prove: Given a = 2, the maximum value of b^2 + c^2 is 8
theorem max_bsquared_plus_csquared (a b c A : Real) (hA : A = Real.pi / 3) (ha : a = 2) :
  b^2 + c^2 ≤ 8 :=
by
  have hcos : Real.cos A = 1 / 2 := by sorry
  have h : 4 = b^2 + c^2 - b * c * (1/2) := by sorry
  have hmax : b^2 + c^2 + b * c ≤ 8 := by sorry
  sorry -- Proof steps to reach the final result

end NUMINAMATH_GPT_find_A_of_trig_max_bsquared_plus_csquared_l149_14987


namespace NUMINAMATH_GPT_total_students_in_class_l149_14960

theorem total_students_in_class (F G B N T : ℕ)
  (hF : F = 41)
  (hG : G = 22)
  (hB : B = 9)
  (hN : N = 15)
  (hT : T = (F + G - B) + N) :
  T = 69 :=
by
  -- This is a theorem statement, proof is intentionally omitted.
  sorry

end NUMINAMATH_GPT_total_students_in_class_l149_14960


namespace NUMINAMATH_GPT_sum_geometric_seq_eq_l149_14951

-- Defining the parameters of the geometric sequence
def a : ℚ := 1 / 5
def r : ℚ := 2 / 5
def n : ℕ := 8

-- Required to prove the sum of the first eight terms equals the given fraction
theorem sum_geometric_seq_eq :
  (a * (1 - r^n) / (1 - r)) = (390369 / 1171875) :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_sum_geometric_seq_eq_l149_14951


namespace NUMINAMATH_GPT_volume_not_determined_l149_14925

noncomputable def tetrahedron_volume_not_unique 
  (area1 area2 area3 : ℝ) (circumradius : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∃ a' b' c', 
      (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧ 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)))

theorem volume_not_determined 
  (area1 area2 area3 circumradius: ℝ) 
  (h: tetrahedron_volume_not_unique area1 area2 area3 circumradius) : 
  ¬ ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∀ a' b' c', 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)) → 
      (a = a' ∧ b = b' ∧ c = c')) := 
by sorry

end NUMINAMATH_GPT_volume_not_determined_l149_14925


namespace NUMINAMATH_GPT_statement_B_false_l149_14985

def f (x : ℝ) : ℝ := 3 * x

def diamondsuit (x y : ℝ) : ℝ := abs (f x - f y)

theorem statement_B_false (x y : ℝ) : 3 * diamondsuit x y ≠ diamondsuit (3 * x) (3 * y) :=
by
  sorry

end NUMINAMATH_GPT_statement_B_false_l149_14985


namespace NUMINAMATH_GPT_percentage_seniors_with_cars_is_40_l149_14949

noncomputable def percentage_of_seniors_with_cars 
  (total_students: ℕ) (seniors: ℕ) (lower_grades: ℕ) (percent_cars_all: ℚ) (percent_cars_lower_grades: ℚ) : ℚ :=
  let total_with_cars := percent_cars_all * total_students
  let lower_grades_with_cars := percent_cars_lower_grades * lower_grades
  let seniors_with_cars := total_with_cars - lower_grades_with_cars
  (seniors_with_cars / seniors) * 100

theorem percentage_seniors_with_cars_is_40
  : percentage_of_seniors_with_cars 1800 300 1500 0.15 0.10 = 40 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_percentage_seniors_with_cars_is_40_l149_14949


namespace NUMINAMATH_GPT_contradiction_proof_l149_14974

theorem contradiction_proof (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : 0 < b ∧ b < 2) (h3 : 0 < c ∧ c < 2) :
  ¬ (a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) :=
sorry

end NUMINAMATH_GPT_contradiction_proof_l149_14974


namespace NUMINAMATH_GPT_perfect_square_of_expression_l149_14931

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem perfect_square_of_expression : 
  (∃ k : ℕ, (factorial 19 * 2 = k ∧ (factorial 20 * factorial 19) / 5 = k * k)) := sorry

end NUMINAMATH_GPT_perfect_square_of_expression_l149_14931


namespace NUMINAMATH_GPT_steve_marbles_after_trans_l149_14937

def initial_marbles (S T L H : ℕ) : Prop :=
  S = 2 * T ∧
  L = S - 5 ∧
  H = T + 3

def transactions (S T L H : ℕ) (new_S new_T new_L new_H : ℕ) : Prop :=
  new_S = S - 10 ∧
  new_L = L - 4 ∧
  new_T = T + 4 ∧
  new_H = H - 6

theorem steve_marbles_after_trans (S T L H new_S new_T new_L new_H : ℕ) :
  initial_marbles S T L H →
  transactions S T L H new_S new_T new_L new_H →
  new_S = 6 →
  new_T = 12 :=
by
  sorry

end NUMINAMATH_GPT_steve_marbles_after_trans_l149_14937


namespace NUMINAMATH_GPT_total_history_and_maths_l149_14946

-- Defining the conditions
def total_students : ℕ := 25
def fraction_like_maths : ℚ := 2 / 5
def fraction_like_science : ℚ := 1 / 3

-- Theorem statement
theorem total_history_and_maths : (total_students * fraction_like_maths + (total_students * (1 - fraction_like_maths) * (1 - fraction_like_science))) = 20 := by
  sorry

end NUMINAMATH_GPT_total_history_and_maths_l149_14946


namespace NUMINAMATH_GPT_purchase_price_of_radio_l149_14959

theorem purchase_price_of_radio 
  (selling_price : ℚ) (loss_percentage : ℚ) (purchase_price : ℚ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 0.05):
  purchase_price = 490 :=
by 
  sorry

end NUMINAMATH_GPT_purchase_price_of_radio_l149_14959


namespace NUMINAMATH_GPT_fuel_consumption_per_100_km_l149_14940

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end NUMINAMATH_GPT_fuel_consumption_per_100_km_l149_14940


namespace NUMINAMATH_GPT_people_attend_both_reunions_l149_14967

theorem people_attend_both_reunions (N D H x : ℕ) 
  (hN : N = 50)
  (hD : D = 50)
  (hH : H = 60)
  (h_total : N = D + H - x) : 
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_people_attend_both_reunions_l149_14967


namespace NUMINAMATH_GPT_find_annual_compound_interest_rate_l149_14918

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_compound_interest_rate :
  compound_interest_rate 10000 24882.50 1 7 0.125 :=
by sorry

end NUMINAMATH_GPT_find_annual_compound_interest_rate_l149_14918


namespace NUMINAMATH_GPT_min_value_a_l149_14973

theorem min_value_a (a : ℕ) :
  (6 * (a + 1)) / (a^2 + 8 * a + 6) ≤ 1 / 100 ↔ a ≥ 594 := sorry

end NUMINAMATH_GPT_min_value_a_l149_14973


namespace NUMINAMATH_GPT_silver_cost_l149_14962

theorem silver_cost (S : ℝ) : 
  (1.5 * S) + (3 * 50 * S) = 3030 → S = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_silver_cost_l149_14962


namespace NUMINAMATH_GPT_max_value_of_function_l149_14958

open Real 

theorem max_value_of_function : ∀ x : ℝ, 
  cos (2 * x) + 6 * cos (π / 2 - x) ≤ 5 ∧ 
  ∃ x' : ℝ, cos (2 * x') + 6 * cos (π / 2 - x') = 5 :=
by 
  sorry

end NUMINAMATH_GPT_max_value_of_function_l149_14958


namespace NUMINAMATH_GPT_part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l149_14965

noncomputable def A : Set ℝ := { x : ℝ | 3 < x ∧ x < 10 }
noncomputable def B : Set ℝ := { x : ℝ | x^2 - 9 * x + 14 < 0 }
noncomputable def C (m : ℝ) : Set ℝ := { x : ℝ | 5 - m < x ∧ x < 2 * m }

theorem part_I_A_inter_B : A ∩ B = { x : ℝ | 3 < x ∧ x < 7 } :=
sorry

theorem part_I_complement_A_union_B :
  (Aᶜ) ∪ B = { x : ℝ | x < 7 ∨ x ≥ 10 } :=
sorry

theorem part_II_range_of_m :
  {m : ℝ | C m ⊆ A ∩ B} = {m : ℝ | m ≤ 2} :=
sorry

end NUMINAMATH_GPT_part_I_A_inter_B_part_I_complement_A_union_B_part_II_range_of_m_l149_14965


namespace NUMINAMATH_GPT_rational_expression_nonnegative_l149_14908

theorem rational_expression_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 :=
  sorry

end NUMINAMATH_GPT_rational_expression_nonnegative_l149_14908


namespace NUMINAMATH_GPT_incorrect_major_premise_l149_14903

noncomputable def Line := Type
noncomputable def Plane := Type

-- Conditions: Definitions
variable (b a : Line) (α : Plane)

-- Assumption: Line b is parallel to Plane α
axiom parallel_to_plane (p : Line) (π : Plane) : Prop

-- Assumption: Line a is in Plane α
axiom line_in_plane (l : Line) (π : Plane) : Prop

-- Define theorem stating the incorrect major premise
theorem incorrect_major_premise 
  (hb_par_α : parallel_to_plane b α)
  (ha_in_α : line_in_plane a α) : ¬ (parallel_to_plane b α → ∀ l, line_in_plane l α → b = l) := 
sorry

end NUMINAMATH_GPT_incorrect_major_premise_l149_14903


namespace NUMINAMATH_GPT_student_age_is_17_in_1960_l149_14971

noncomputable def student's_age_in_1960 (x y : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) : ℕ := 
  let birth_year : ℕ := 1900 + 10 * x + y
  let age_in_1960 : ℕ := 1960 - birth_year
  age_in_1960

theorem student_age_is_17_in_1960 :
  ∃ x y : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ (1960 - (1900 + 10 * x + y) = 1 + 9 + x + y) ∧ (1960 - (1900 + 10 * x + y) = 17) :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_student_age_is_17_in_1960_l149_14971


namespace NUMINAMATH_GPT_domain_log_base_4_l149_14984

theorem domain_log_base_4 (x : ℝ) : {x // x + 2 > 0} = {x | x > -2} :=
by
  sorry

end NUMINAMATH_GPT_domain_log_base_4_l149_14984


namespace NUMINAMATH_GPT_change_given_back_l149_14991

theorem change_given_back
  (p s t a : ℕ)
  (hp : p = 140)
  (hs : s = 43)
  (ht : t = 15)
  (ha : a = 200) :
  (a - (p + s + t)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_change_given_back_l149_14991


namespace NUMINAMATH_GPT_weight_of_currants_l149_14978

noncomputable def packing_density : ℝ := 0.74
noncomputable def water_density : ℝ := 1000
noncomputable def bucket_volume : ℝ := 0.01

theorem weight_of_currants :
  (water_density * (packing_density * bucket_volume)) = 7.4 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_currants_l149_14978


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l149_14979

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := { x | ∃ m : ℕ, x = 2 * m }

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l149_14979


namespace NUMINAMATH_GPT_sum_of_elements_in_T_l149_14936

noncomputable def digit_sum : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 504
noncomputable def repeating_sum : ℕ := digit_sum * 1111
noncomputable def sum_T : ℚ := repeating_sum / 9999

theorem sum_of_elements_in_T : sum_T = 2523 := by
  sorry

end NUMINAMATH_GPT_sum_of_elements_in_T_l149_14936


namespace NUMINAMATH_GPT_speed_upstream_l149_14907

-- Conditions definitions
def speed_of_boat_still_water : ℕ := 50
def speed_of_current : ℕ := 20

-- Theorem stating the problem
theorem speed_upstream : (speed_of_boat_still_water - speed_of_current = 30) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_speed_upstream_l149_14907


namespace NUMINAMATH_GPT_smallest_four_digit_remainder_l149_14966

theorem smallest_four_digit_remainder :
  ∃ N : ℕ, (N % 6 = 5) ∧ (1000 ≤ N ∧ N ≤ 9999) ∧ (∀ M : ℕ, (M % 6 = 5) ∧ (1000 ≤ M ∧ M ≤ 9999) → N ≤ M) ∧ N = 1001 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_remainder_l149_14966


namespace NUMINAMATH_GPT_rattlesnakes_count_l149_14969

theorem rattlesnakes_count (total_snakes : ℕ) (boa_constrictors pythons rattlesnakes : ℕ)
  (h1 : total_snakes = 200)
  (h2 : boa_constrictors = 40)
  (h3 : pythons = 3 * boa_constrictors)
  (h4 : total_snakes = boa_constrictors + pythons + rattlesnakes) :
  rattlesnakes = 40 :=
by
  sorry

end NUMINAMATH_GPT_rattlesnakes_count_l149_14969


namespace NUMINAMATH_GPT_Jason_spent_on_jacket_l149_14933

/-
Given:
- Amount_spent_on_shorts: ℝ := 14.28
- Total_spent_on_clothing: ℝ := 19.02

Prove:
- Amount_spent_on_jacket = 4.74
-/
def Amount_spent_on_shorts : ℝ := 14.28
def Total_spent_on_clothing : ℝ := 19.02

-- We need to prove:
def Amount_spent_on_jacket : ℝ := Total_spent_on_clothing - Amount_spent_on_shorts 

theorem Jason_spent_on_jacket : Amount_spent_on_jacket = 4.74 := by
  sorry

end NUMINAMATH_GPT_Jason_spent_on_jacket_l149_14933


namespace NUMINAMATH_GPT_bus_stops_per_hour_l149_14913

theorem bus_stops_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h₁ : speed_without_stoppages = 50)
  (h₂ : speed_with_stoppages = 40) :
  ∃ (minutes_stopped : ℝ), minutes_stopped = 12 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_per_hour_l149_14913


namespace NUMINAMATH_GPT_probability_x_gt_3y_correct_l149_14917

noncomputable def probability_x_gt_3y : ℚ :=
  let rect_area := (2010 : ℚ) * 2011
  let triangle_area := (2010 : ℚ) * (2010 / 3) / 2
  (triangle_area / rect_area)

theorem probability_x_gt_3y_correct :
  probability_x_gt_3y = 670 / 2011 := 
by
  sorry

end NUMINAMATH_GPT_probability_x_gt_3y_correct_l149_14917


namespace NUMINAMATH_GPT_exterior_angle_regular_polygon_l149_14956

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end NUMINAMATH_GPT_exterior_angle_regular_polygon_l149_14956


namespace NUMINAMATH_GPT_students_drawn_from_A_l149_14943

-- Define the conditions as variables (number of students in each school)
def studentsA := 3600
def studentsB := 5400
def studentsC := 1800
def sampleSize := 90

-- Define the total number of students
def totalStudents := studentsA + studentsB + studentsC

-- Define the proportion of students in School A
def proportionA := studentsA / totalStudents

-- Define the number of students to be drawn from School A using stratified sampling
def drawnFromA := sampleSize * proportionA

-- The theorem to prove
theorem students_drawn_from_A : drawnFromA = 30 :=
by
  sorry

end NUMINAMATH_GPT_students_drawn_from_A_l149_14943


namespace NUMINAMATH_GPT_Hayley_l149_14996

-- Definitions based on the given conditions
def num_friends : ℕ := 9
def stickers_per_friend : ℕ := 8

-- Theorem statement
theorem Hayley's_total_stickers : num_friends * stickers_per_friend = 72 := by
  sorry

end NUMINAMATH_GPT_Hayley_l149_14996


namespace NUMINAMATH_GPT_cos_240_eq_neg_half_l149_14924

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_cos_240_eq_neg_half_l149_14924


namespace NUMINAMATH_GPT_negation_of_universal_prop_l149_14977

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.sin x > 1

-- The theorem stating the equivalence
theorem negation_of_universal_prop : ¬p ↔ neg_p := 
by sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l149_14977


namespace NUMINAMATH_GPT_part1_part2_l149_14955

noncomputable def f (a x : ℝ) := a * Real.log x - x / 2

theorem part1 (a : ℝ) : (∀ x, f a x = a * Real.log x - x / 2) → (∃ x, x = 2 ∧ deriv (f a) x = 0) → a = 1 :=
by sorry

theorem part2 (k : ℝ) : (∀ x, x > 1 → f 1 x + k / x < 0) → k ≤ 1 / 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l149_14955


namespace NUMINAMATH_GPT_find_N_l149_14935

def f (N : ℕ) : ℕ :=
  if N % 2 = 0 then 5 * N else 3 * N + 2

theorem find_N (N : ℕ) :
  f (f (f (f (f N)))) = 542 ↔ N = 112500 := by
  sorry

end NUMINAMATH_GPT_find_N_l149_14935


namespace NUMINAMATH_GPT_train_length_is_330_meters_l149_14927

noncomputable def train_speed : ℝ := 60 -- in km/hr
noncomputable def man_speed : ℝ := 6    -- in km/hr
noncomputable def time : ℝ := 17.998560115190788  -- in seconds

noncomputable def relative_speed_km_per_hr : ℝ := train_speed + man_speed
noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def relative_speed_m_per_s : ℝ := 
  relative_speed_km_per_hr * conversion_factor

theorem train_length_is_330_meters : 
  (relative_speed_m_per_s * time) = 330 := 
sorry

end NUMINAMATH_GPT_train_length_is_330_meters_l149_14927


namespace NUMINAMATH_GPT_set_M_properties_l149_14995

def f (x : ℝ) : ℝ := |x| - |2 * x - 1|

def M : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem set_M_properties :
  M = { x | 0 < x ∧ x < 2 } ∧
  (∀ a, a ∈ M → 
    ((0 < a ∧ a < 1) → (a^2 - a + 1 < 1 / a)) ∧
    (a = 1 → (a^2 - a + 1 = 1 / a)) ∧
    ((1 < a ∧ a < 2) → (a^2 - a + 1 > 1 / a))) := 
by
  sorry

end NUMINAMATH_GPT_set_M_properties_l149_14995


namespace NUMINAMATH_GPT_range_of_m_l149_14975

theorem range_of_m {x m : ℝ} 
  (h1 : 1 / 3 < x) 
  (h2 : x < 1 / 2) 
  (h3 : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l149_14975


namespace NUMINAMATH_GPT_olaf_travels_miles_l149_14986

-- Define the given conditions
def men : ℕ := 25
def per_day_water_per_man : ℚ := 1 / 2
def boat_mileage_per_day : ℕ := 200
def total_water : ℚ := 250

-- Define the daily water consumption for the crew
def daily_water_consumption : ℚ := men * per_day_water_per_man

-- Define the number of days the water will last
def days_water_lasts : ℚ := total_water / daily_water_consumption

-- Define the total miles traveled
def total_miles_traveled : ℚ := days_water_lasts * boat_mileage_per_day

-- Theorem statement to prove the total miles traveled is 4000 miles
theorem olaf_travels_miles : total_miles_traveled = 4000 := by
  sorry

end NUMINAMATH_GPT_olaf_travels_miles_l149_14986


namespace NUMINAMATH_GPT_oranges_and_apples_costs_l149_14915

theorem oranges_and_apples_costs :
  ∃ (x y : ℚ), 7 * x + 5 * y = 13 ∧ 3 * x + 4 * y = 8 ∧ 37 * x + 45 * y = 93 :=
by 
  sorry

end NUMINAMATH_GPT_oranges_and_apples_costs_l149_14915


namespace NUMINAMATH_GPT_total_points_of_players_l149_14948

variables (Samanta Mark Eric Daisy Jake : ℕ)
variables (h1 : Samanta = Mark + 8)
variables (h2 : Mark = 3 / 2 * Eric)
variables (h3 : Eric = 6)
variables (h4 : Daisy = 3 / 4 * (Samanta + Mark + Eric))
variables (h5 : Jake = Samanta - Eric)
 
theorem total_points_of_players :
  Samanta + Mark + Eric + Daisy + Jake = 67 :=
sorry

end NUMINAMATH_GPT_total_points_of_players_l149_14948


namespace NUMINAMATH_GPT_projectile_reaches_45_feet_first_time_l149_14953

theorem projectile_reaches_45_feet_first_time :
  ∃ t : ℝ, (-20 * t^2 + 90 * t = 45) ∧ abs (t - 0.9) < 0.1 := sorry

end NUMINAMATH_GPT_projectile_reaches_45_feet_first_time_l149_14953


namespace NUMINAMATH_GPT_subset_implies_range_of_a_l149_14934

theorem subset_implies_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → x > a) → a < -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_subset_implies_range_of_a_l149_14934


namespace NUMINAMATH_GPT_jack_final_apples_l149_14950

-- Jack's transactions and initial count as conditions
def initial_count : ℕ := 150
def sold_to_jill : ℕ := initial_count * 30 / 100
def remaining_after_jill : ℕ := initial_count - sold_to_jill
def sold_to_june : ℕ := remaining_after_jill * 20 / 100
def remaining_after_june : ℕ := remaining_after_jill - sold_to_june
def donated_to_charity : ℕ := 5
def final_count : ℕ := remaining_after_june - donated_to_charity

-- Proof statement
theorem jack_final_apples : final_count = 79 := by
  sorry

end NUMINAMATH_GPT_jack_final_apples_l149_14950


namespace NUMINAMATH_GPT_purely_periodic_fraction_period_length_divisible_l149_14939

noncomputable def purely_periodic_fraction (p q n : ℕ) : Prop :=
  ∃ (r : ℕ), 10 ^ n - 1 = r * q ∧ (∃ (k : ℕ), q * (10 ^ (n * k)) ∣ p)

theorem purely_periodic_fraction_period_length_divisible
  (p q n : ℕ) (hq : ¬ (2 ∣ q) ∧ ¬ (5 ∣ q)) (hpq : p < q) (hn : 10 ^ n - 1 ∣ q) :
  purely_periodic_fraction p q n :=
by
  sorry

end NUMINAMATH_GPT_purely_periodic_fraction_period_length_divisible_l149_14939


namespace NUMINAMATH_GPT_max_halls_l149_14983

theorem max_halls (n : ℕ) (hall : ℕ → ℕ) (H : ∀ n, hall n = hall (3 * n + 1) ∧ hall n = hall (n + 10)) :
  ∃ (m : ℕ), m = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_halls_l149_14983


namespace NUMINAMATH_GPT_simplify_expression_l149_14976

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l149_14976


namespace NUMINAMATH_GPT_arrange_students_l149_14980

theorem arrange_students (students : Fin 7 → Prop) : 
  ∃ arrangements : ℕ, arrangements = 140 :=
by
  -- Define selection of 6 out of 7
  let selection_ways := Nat.choose 7 6
  -- Define arrangement of 6 into two groups of 3 each
  let arrangement_ways := (Nat.choose 6 3) * (Nat.choose 3 3)
  -- Calculate total arrangements by multiplying the two values
  let total_arrangements := selection_ways * arrangement_ways
  use total_arrangements
  simp [selection_ways, arrangement_ways, total_arrangements]
  exact rfl

end NUMINAMATH_GPT_arrange_students_l149_14980


namespace NUMINAMATH_GPT_g_g_g_25_l149_14926

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 18

theorem g_g_g_25 :
  g (g (g 25)) = 22 :=
by
  sorry

end NUMINAMATH_GPT_g_g_g_25_l149_14926


namespace NUMINAMATH_GPT_find_number_l149_14947

theorem find_number (x : ℝ) (h : (x - 8 - 12) / 5 = 7) : x = 55 :=
sorry

end NUMINAMATH_GPT_find_number_l149_14947


namespace NUMINAMATH_GPT_power_mod_l149_14911

theorem power_mod (n : ℕ) : 2^99 % 7 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_power_mod_l149_14911


namespace NUMINAMATH_GPT_width_of_second_square_is_seven_l149_14968

-- The conditions translated into Lean definitions
def first_square : ℕ × ℕ := (8, 5)
def third_square : ℕ × ℕ := (5, 5)
def flag_dimensions : ℕ × ℕ := (15, 9)

-- The area calculation functions
def area (dim : ℕ × ℕ) : ℕ := dim.fst * dim.snd

-- Given areas for the first and third square
def area_first_square : ℕ := area first_square
def area_third_square : ℕ := area third_square

-- Desired flag area
def flag_area : ℕ := area flag_dimensions

-- Total area of first and third squares
def total_area_first_and_third : ℕ := area_first_square + area_third_square

-- Required area for the second square
def area_needed_second_square : ℕ := flag_area - total_area_first_and_third

-- Given length of the second square
def second_square_length : ℕ := 10

-- Solve for the width of the second square
def second_square_width : ℕ := area_needed_second_square / second_square_length

-- The proof goal
theorem width_of_second_square_is_seven : second_square_width = 7 := by
  sorry

end NUMINAMATH_GPT_width_of_second_square_is_seven_l149_14968


namespace NUMINAMATH_GPT_carousel_seat_count_l149_14961

theorem carousel_seat_count
  (total_seats : ℕ)
  (colors : ℕ → Prop)
  (num_yellow num_blue num_red : ℕ)
  (num_colors : ∀ n, colors n → n = num_yellow ∨ n = num_blue ∨ n = num_red)
  (opposite_blue_red_7_3 : ∀ n, n = 7 ↔ n + 50 = 3)
  (opposite_yellow_red_7_23 : ∀ n, n = 7 ↔ n + 50 = 23)
  (total := 100)
 :
 (num_yellow = 34 ∧ num_blue = 20 ∧ num_red = 46) :=
by
  sorry

end NUMINAMATH_GPT_carousel_seat_count_l149_14961


namespace NUMINAMATH_GPT_x_greater_than_y_l149_14909

theorem x_greater_than_y (x y z : ℝ) (h1 : x + y + z = 28) (h2 : 2 * x - y = 32) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 
  x > y :=
by 
  sorry

end NUMINAMATH_GPT_x_greater_than_y_l149_14909


namespace NUMINAMATH_GPT_find_a_l149_14994
-- Import the entire Mathlib to ensure all necessary primitives and theorems are available.

-- Define a constant equation representing the conditions.
def equation (x a : ℝ) := 3 * x + 2 * a

-- Define a theorem to prove the condition => result structure.
theorem find_a (h : equation 2 a = 0) : a = -3 :=
by sorry

end NUMINAMATH_GPT_find_a_l149_14994


namespace NUMINAMATH_GPT_negation_of_exisential_inequality_l149_14963

open Classical

theorem negation_of_exisential_inequality :
  ¬ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 := 
by 
sorry

end NUMINAMATH_GPT_negation_of_exisential_inequality_l149_14963


namespace NUMINAMATH_GPT_total_expenditure_l149_14941

-- Define the conditions
def cost_per_acre : ℕ := 20
def acres_bought : ℕ := 30
def house_cost : ℕ := 120000
def cost_per_cow : ℕ := 1000
def cows_bought : ℕ := 20
def cost_per_chicken : ℕ := 5
def chickens_bought : ℕ := 100
def hourly_installation_cost : ℕ := 100
def installation_hours : ℕ := 6
def solar_equipment_cost : ℕ := 6000

-- Define the total cost breakdown
def land_cost : ℕ := cost_per_acre * acres_bought
def cows_cost : ℕ := cost_per_cow * cows_bought
def chickens_cost : ℕ := cost_per_chicken * chickens_bought
def solar_installation_cost : ℕ := (hourly_installation_cost * installation_hours) + solar_equipment_cost

-- Define the total cost
def total_cost : ℕ :=
  land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost

-- The theorem statement
theorem total_expenditure : total_cost = 147700 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_expenditure_l149_14941


namespace NUMINAMATH_GPT_denominator_of_second_fraction_l149_14932

theorem denominator_of_second_fraction :
  let a := 2007
  let b := 2999
  let c := 8001
  let d := 2001
  let e := 3999
  let sum := 3.0035428163476343
  let first_fraction := (2007 : ℝ) / 2999
  let third_fraction := (2001 : ℝ) / 3999
  ∃ x : ℤ, (first_fraction + (8001 : ℝ) / x + third_fraction) = 3.0035428163476343 ∧ x = 4362 := 
by
  sorry

end NUMINAMATH_GPT_denominator_of_second_fraction_l149_14932


namespace NUMINAMATH_GPT_find_x_l149_14902

theorem find_x (x : ℝ) (h : x - 2 * x + 3 * x = 100) : x = 50 := by
  sorry

end NUMINAMATH_GPT_find_x_l149_14902


namespace NUMINAMATH_GPT_total_notes_l149_14929

theorem total_notes (total_amount : ℤ) (num_50_notes : ℤ) (value_50 : ℤ) (value_500 : ℤ) (total_notes : ℤ) :
  total_amount = num_50_notes * value_50 + (total_notes - num_50_notes) * value_500 → 
  total_amount = 10350 → num_50_notes = 77 → value_50 = 50 → value_500 = 500 → total_notes = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_notes_l149_14929


namespace NUMINAMATH_GPT_remainder_of_division_l149_14914

def dividend := 1234567
def divisor := 257

theorem remainder_of_division : dividend % divisor = 774 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l149_14914


namespace NUMINAMATH_GPT_stratified_sample_over_30_l149_14945

-- Define the total number of employees and conditions
def total_employees : ℕ := 49
def employees_over_30 : ℕ := 14
def employees_30_or_younger : ℕ := 35
def sample_size : ℕ := 7

-- State the proportion and the final required count
def proportion_over_30 (total : ℕ) (over_30 : ℕ) : ℚ := (over_30 : ℚ) / (total : ℚ)
def required_count (proportion : ℚ) (sample : ℕ) : ℚ := proportion * (sample : ℚ)

theorem stratified_sample_over_30 :
  required_count (proportion_over_30 total_employees employees_over_30) sample_size = 2 := 
by sorry

end NUMINAMATH_GPT_stratified_sample_over_30_l149_14945


namespace NUMINAMATH_GPT_max_consecutive_integers_sum_48_l149_14921

-- Define the sum of consecutive integers
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Define the main theorem
theorem max_consecutive_integers_sum_48 : 
  ∃ N a : ℤ, sum_consecutive_integers a N = 48 ∧ (∀ N' : ℤ, ((N' * (2 * a + N' - 1)) / 2 = 48) → N' ≤ N) :=
sorry

end NUMINAMATH_GPT_max_consecutive_integers_sum_48_l149_14921


namespace NUMINAMATH_GPT_solution_set_of_f_double_exp_inequality_l149_14928

theorem solution_set_of_f_double_exp_inequality (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ 0 < f x) :
  {x : ℝ | f (2^x) < 0} = {x : ℝ | x > 0} :=
sorry

end NUMINAMATH_GPT_solution_set_of_f_double_exp_inequality_l149_14928


namespace NUMINAMATH_GPT_gcd_840_1764_l149_14982

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l149_14982


namespace NUMINAMATH_GPT_inequality_system_range_l149_14990

theorem inequality_system_range (a : ℝ) :
  (∃ (x : ℤ), (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0)) ∧
  (∀ x : ℤ, (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0) → (x = 2 ∨ x = 3)) →
  6 ≤ a ∧ a < 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_range_l149_14990


namespace NUMINAMATH_GPT_combined_gravitational_force_l149_14905

theorem combined_gravitational_force 
    (d_E_surface : ℝ) (f_E_surface : ℝ) (d_M_surface : ℝ) (f_M_surface : ℝ) 
    (d_E_new : ℝ) (d_M_new : ℝ) 
    (k_E : ℝ) (k_M : ℝ) 
    (h1 : k_E = f_E_surface * d_E_surface^2)
    (h2 : k_M = f_M_surface * d_M_surface^2)
    (h3 : f_E_new = k_E / d_E_new^2)
    (h4 : f_M_new = k_M / d_M_new^2) : 
  f_E_new + f_M_new = 755.7696 :=
by
  sorry

end NUMINAMATH_GPT_combined_gravitational_force_l149_14905


namespace NUMINAMATH_GPT_f_2020_minus_f_2018_l149_14904

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 5) = f x
axiom f_seven : f 7 = 9

theorem f_2020_minus_f_2018 : f 2020 - f 2018 = 9 := by
  sorry

end NUMINAMATH_GPT_f_2020_minus_f_2018_l149_14904


namespace NUMINAMATH_GPT_geom_seq_decreasing_l149_14942

theorem geom_seq_decreasing :
  (∀ n : ℕ, (4 : ℝ) * 3^(1 - (n + 1) : ℤ) < (4 : ℝ) * 3^(1 - n : ℤ)) :=
sorry

end NUMINAMATH_GPT_geom_seq_decreasing_l149_14942


namespace NUMINAMATH_GPT_combined_teaching_experience_l149_14900

def james_teaching_years : ℕ := 40
def partner_teaching_years : ℕ := james_teaching_years - 10

theorem combined_teaching_experience : james_teaching_years + partner_teaching_years = 70 :=
by
  sorry

end NUMINAMATH_GPT_combined_teaching_experience_l149_14900


namespace NUMINAMATH_GPT_odd_function_decreasing_l149_14952

theorem odd_function_decreasing (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, x < y → y < 0 → f x > f y) :
  ∀ x y, 0 < x → x < y → f y < f x :=
by
  sorry

end NUMINAMATH_GPT_odd_function_decreasing_l149_14952


namespace NUMINAMATH_GPT_area_of_CEF_l149_14997

-- Definitions of points and triangles based on given ratios
def is_right_triangle (A B C : Type) : Prop := sorry -- Placeholder for right triangle condition

def divides_ratio (A B : Type) (ratio : ℚ) : Prop := sorry -- Placeholder for ratio division condition

def area_of_triangle (A B C : Type) : ℚ := sorry -- Function to calculate area of triangle - placeholder

theorem area_of_CEF {A B C E F : Type} 
  (h1 : is_right_triangle A B C)
  (h2 : divides_ratio A C (1/4))
  (h3 : divides_ratio A B (2/3))
  (h4 : area_of_triangle A B C = 50) : 
  area_of_triangle C E F = 25 :=
sorry

end NUMINAMATH_GPT_area_of_CEF_l149_14997


namespace NUMINAMATH_GPT_fraction_simplification_l149_14998

theorem fraction_simplification : 
  (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_simplification_l149_14998


namespace NUMINAMATH_GPT_harmonic_point_P_3_m_harmonic_point_hyperbola_l149_14919

-- Part (1)
theorem harmonic_point_P_3_m (t : ℝ) (m : ℝ) (P : ℝ × ℝ → Prop)
  (h₁ : P ⟨ 3, m ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  m = -7 :=
by sorry

-- Part (2)
theorem harmonic_point_hyperbola (k : ℝ) (P : ℝ × ℝ → Prop)
  (h_hb : ∀ x, -3 < x ∧ x < -1 → P ⟨ x, k / x ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  3 < k ∧ k < 4 :=
by sorry

end NUMINAMATH_GPT_harmonic_point_P_3_m_harmonic_point_hyperbola_l149_14919


namespace NUMINAMATH_GPT_find_complex_number_l149_14916

-- Define the complex number z and the condition
variable (z : ℂ)
variable (h : (conj z) / (1 + I) = 1 - 2 * I)

-- State the theorem
theorem find_complex_number (hz : h) : z = 3 + I := 
sorry

end NUMINAMATH_GPT_find_complex_number_l149_14916


namespace NUMINAMATH_GPT_time_to_run_above_tree_l149_14944

-- Defining the given conditions
def tiger_length : ℕ := 5
def tree_trunk_length : ℕ := 20
def time_to_pass_grass : ℕ := 1

-- Defining the speed of the tiger
def tiger_speed : ℕ := tiger_length / time_to_pass_grass

-- Defining the total distance the tiger needs to run
def total_distance : ℕ := tree_trunk_length + tiger_length

-- The theorem stating the time it takes for the tiger to run above the fallen tree trunk
theorem time_to_run_above_tree :
  (total_distance / tiger_speed) = 5 :=
by
  -- Trying to fit the solution steps as formal Lean statements
  sorry

end NUMINAMATH_GPT_time_to_run_above_tree_l149_14944


namespace NUMINAMATH_GPT_pencils_left_l149_14922

theorem pencils_left (anna_pencils : ℕ) (harry_pencils : ℕ)
  (h_anna : anna_pencils = 50) (h_harry : harry_pencils = 2 * anna_pencils)
  (lost_pencils : ℕ) (h_lost : lost_pencils = 19) :
  harry_pencils - lost_pencils = 81 :=
by
  sorry

end NUMINAMATH_GPT_pencils_left_l149_14922


namespace NUMINAMATH_GPT_volume_of_tetrahedron_l149_14988

theorem volume_of_tetrahedron 
  (A B C D E : ℝ)
  (AB AD AE: ℝ)
  (h_AB : AB = 3)
  (h_AD : AD = 4)
  (h_AE : AE = 1)
  (V : ℝ) :
  (V = (4 * Real.sqrt 3) / 3) :=
sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_l149_14988


namespace NUMINAMATH_GPT_sequence_S_n_a_n_l149_14993

noncomputable def sequence_S (n : ℕ) : ℝ := -1 / (n : ℝ)

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then -1 else 1 / ((n : ℝ) * (n - 1))

theorem sequence_S_n_a_n (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = -1 →
  (∀ n, (a (n + 1)) / (S (n + 1)) = S n) →
  S n = sequence_S n ∧ a n = sequence_a n :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sequence_S_n_a_n_l149_14993


namespace NUMINAMATH_GPT_number_added_after_division_is_5_l149_14912

noncomputable def number_thought_of : ℕ := 72
noncomputable def result_after_division (n : ℕ) : ℕ := n / 6
noncomputable def final_result (n x : ℕ) : ℕ := result_after_division n + x

theorem number_added_after_division_is_5 :
  ∃ x : ℕ, final_result number_thought_of x = 17 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_added_after_division_is_5_l149_14912


namespace NUMINAMATH_GPT_will_buy_toys_l149_14923

theorem will_buy_toys : 
  ∀ (initialMoney spentMoney toyCost : ℕ), 
  initialMoney = 83 → spentMoney = 47 → toyCost = 4 → 
  (initialMoney - spentMoney) / toyCost = 9 :=
by
  intros initialMoney spentMoney toyCost hInit hSpent hCost
  sorry

end NUMINAMATH_GPT_will_buy_toys_l149_14923


namespace NUMINAMATH_GPT_problem_solution_l149_14972

theorem problem_solution :
  (3012 - 2933)^2 / 196 = 32 := sorry

end NUMINAMATH_GPT_problem_solution_l149_14972
