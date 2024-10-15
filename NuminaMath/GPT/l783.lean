import Mathlib

namespace NUMINAMATH_GPT_complement_intersection_l783_78386

def U : Set ℤ := Set.univ
def A : Set ℤ := {1, 2}
def B : Set ℤ := {3, 4}

-- A ∪ B should equal {1, 2, 3, 4}
axiom AUeq : A ∪ B = {1, 2, 3, 4}

theorem complement_intersection : (U \ A) ∩ B = {3, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l783_78386


namespace NUMINAMATH_GPT_calculate_value_l783_78314

theorem calculate_value (x y : ℝ) (h : 2 * x + y = 6) : 
    ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_value_l783_78314


namespace NUMINAMATH_GPT_sequence_of_arrows_512_to_517_is_B_C_D_E_A_l783_78375

noncomputable def sequence_from_512_to_517 : List Char :=
  let pattern := ['A', 'B', 'C', 'D', 'E']
  pattern.drop 2 ++ pattern.take 2

theorem sequence_of_arrows_512_to_517_is_B_C_D_E_A : sequence_from_512_to_517 = ['B', 'C', 'D', 'E', 'A'] :=
  sorry

end NUMINAMATH_GPT_sequence_of_arrows_512_to_517_is_B_C_D_E_A_l783_78375


namespace NUMINAMATH_GPT_max_value_expr_l783_78379

def point_on_line (m n : ℝ) : Prop :=
  3 * m + n = -1

def mn_positive (m n : ℝ) : Prop :=
  m * n > 0

theorem max_value_expr (m n : ℝ) (h1 : point_on_line m n) (h2 : mn_positive m n) :
  (3 / m + 1 / n) = -16 :=
sorry

end NUMINAMATH_GPT_max_value_expr_l783_78379


namespace NUMINAMATH_GPT_find_b_c_find_a_range_l783_78383

noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c
noncomputable def g (a b c x : ℝ) : ℝ := f a b c x + 2 * x
noncomputable def f_prime (a b x : ℝ) : ℝ := x^2 - a * x + b
noncomputable def g_prime (a b x : ℝ) : ℝ := f_prime a b x + 2

theorem find_b_c (a c : ℝ) (h_f0 : f a 0 c 0 = c) (h_tangent_y_eq_1 : 1 = c) : 
  b = 0 ∧ c = 1 :=
by
  sorry

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, g_prime a 0 x ≥ 0) ↔ a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_c_find_a_range_l783_78383


namespace NUMINAMATH_GPT_solve_equation_l783_78360

theorem solve_equation :
  ∃ (a b c d : ℚ), 
  (a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + 2 / 5 = 0) ∧ 
  (a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5) := sorry

end NUMINAMATH_GPT_solve_equation_l783_78360


namespace NUMINAMATH_GPT_inequality_abcde_l783_78352

theorem inequality_abcde
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d) : 
  1 / a + 1 / b + 4 / c + 16 / d ≥ 64 / (a + b + c + d) := 
  sorry

end NUMINAMATH_GPT_inequality_abcde_l783_78352


namespace NUMINAMATH_GPT_Jermaine_more_than_Terrence_l783_78338

theorem Jermaine_more_than_Terrence :
  ∀ (total_earnings Terrence_earnings Emilee_earnings : ℕ),
    total_earnings = 90 →
    Terrence_earnings = 30 →
    Emilee_earnings = 25 →
    (total_earnings - Terrence_earnings - Emilee_earnings) - Terrence_earnings = 5 := by
  sorry

end NUMINAMATH_GPT_Jermaine_more_than_Terrence_l783_78338


namespace NUMINAMATH_GPT_amanda_days_needed_to_meet_goal_l783_78316

def total_tickets : ℕ := 80
def first_day_friends : ℕ := 5
def first_day_per_friend : ℕ := 4
def first_day_tickets : ℕ := first_day_friends * first_day_per_friend
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_days_needed_to_meet_goal : 
  first_day_tickets + second_day_tickets + third_day_tickets = total_tickets → 
  3 = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_amanda_days_needed_to_meet_goal_l783_78316


namespace NUMINAMATH_GPT_find_m_l783_78348

theorem find_m 
  (m : ℝ) 
  (h1 : |m + 1| ≠ 0)
  (h2 : m^2 = 1) : 
  m = 1 := sorry

end NUMINAMATH_GPT_find_m_l783_78348


namespace NUMINAMATH_GPT_collinear_A₁_F_B_iff_q_eq_4_l783_78358

open Real

theorem collinear_A₁_F_B_iff_q_eq_4
  (m q : ℝ) (h_m : m ≠ 0)
  (A B : ℝ × ℝ)
  (h_A : 3 * (m * A.snd + q)^2 + 4 * A.snd^2 = 12)
  (h_B : 3 * (m * B.snd + q)^2 + 4 * B.snd^2 = 12)
  (A₁ : ℝ × ℝ := (A.fst, -A.snd))
  (F : ℝ × ℝ := (1, 0)) :
  ((q = 4) ↔ (∃ k : ℝ, k * (F.fst - A₁.fst) = F.snd - A₁.snd ∧ k * (B.fst - F.fst) = B.snd - F.snd)) :=
sorry

end NUMINAMATH_GPT_collinear_A₁_F_B_iff_q_eq_4_l783_78358


namespace NUMINAMATH_GPT_moles_NaOH_to_form_H2O_2_moles_l783_78363

-- Define the reaction and moles involved
def reaction : String := "NH4NO3 + NaOH -> NaNO3 + NH3 + H2O"
def moles_H2O_produced : Nat := 2
def moles_NaOH_required (moles_H2O : Nat) : Nat := moles_H2O

-- Theorem stating the required moles of NaOH to produce 2 moles of H2O
theorem moles_NaOH_to_form_H2O_2_moles : moles_NaOH_required moles_H2O_produced = 2 := 
by
  sorry

end NUMINAMATH_GPT_moles_NaOH_to_form_H2O_2_moles_l783_78363


namespace NUMINAMATH_GPT_least_multiple_of_25_gt_450_correct_l783_78340

def least_multiple_of_25_gt_450 : ℕ :=
  475

theorem least_multiple_of_25_gt_450_correct (n : ℕ) (h1 : 25 ∣ n) (h2 : n > 450) : n ≥ least_multiple_of_25_gt_450 :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_25_gt_450_correct_l783_78340


namespace NUMINAMATH_GPT_parallelogram_sides_l783_78370

theorem parallelogram_sides (x y : ℝ) (h₁ : 4 * x + 1 = 11) (h₂ : 10 * y - 3 = 5) : x + y = 3.3 :=
sorry

end NUMINAMATH_GPT_parallelogram_sides_l783_78370


namespace NUMINAMATH_GPT_slope_angle_line_l783_78346
open Real

theorem slope_angle_line (x y : ℝ) :
  x + sqrt 3 * y - 1 = 0 → ∃ θ : ℝ, θ = 150 ∧
  ∃ (m : ℝ), m = -sqrt 3 / 3 ∧ θ = arctan m :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_line_l783_78346


namespace NUMINAMATH_GPT_calculate_uphill_distance_l783_78307

noncomputable def uphill_speed : ℝ := 30
noncomputable def downhill_speed : ℝ := 40
noncomputable def downhill_distance : ℝ := 50
noncomputable def average_speed : ℝ := 32.73

theorem calculate_uphill_distance : ∃ d : ℝ, d = 99.86 ∧ 
  32.73 = (d + downhill_distance) / (d / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end NUMINAMATH_GPT_calculate_uphill_distance_l783_78307


namespace NUMINAMATH_GPT_seq_is_geometric_from_second_l783_78388

namespace sequence_problem

-- Definitions and conditions
def S : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 1) => 3 * S n - 2 * S (n - 1)

-- Recursive definition for sum of sequence terms
axiom S_rec_relation (n : ℕ) (h : n ≥ 2) : 
  S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Prove the sequence is geometric from the second term
theorem seq_is_geometric_from_second :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 2, a (n + 1) = 2 * a n) ∧ 
  (a 1 = 1) ∧ 
  (a 2 = 1) :=
by
  sorry

end sequence_problem

end NUMINAMATH_GPT_seq_is_geometric_from_second_l783_78388


namespace NUMINAMATH_GPT_range_of_linear_function_l783_78371

theorem range_of_linear_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  3 < -2 * x + 5 ∧ -2 * x + 5 < 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_linear_function_l783_78371


namespace NUMINAMATH_GPT_a_100_value_l783_78325

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     => 0    -- using 0-index for convenience
| (n+1) => a n + 4

-- Prove the value of the 100th term in the sequence
theorem a_100_value : a 100 = 397 := 
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_a_100_value_l783_78325


namespace NUMINAMATH_GPT_lcm_of_numbers_is_750_l783_78389

-- Define the two numbers x and y
variables (x y : ℕ)

-- Given conditions as hypotheses
def product_of_numbers := 18750
def hcf_of_numbers := 25

-- The proof problem statement
theorem lcm_of_numbers_is_750 (h_product : x * y = product_of_numbers) 
                              (h_hcf : Nat.gcd x y = hcf_of_numbers) : Nat.lcm x y = 750 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_numbers_is_750_l783_78389


namespace NUMINAMATH_GPT_min_value_of_expression_l783_78374

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y = 5 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l783_78374


namespace NUMINAMATH_GPT_andrew_paid_1428_l783_78355

-- Define the constants for the problem
def rate_per_kg_grapes : ℕ := 98
def kg_grapes : ℕ := 11

def rate_per_kg_mangoes : ℕ := 50
def kg_mangoes : ℕ := 7

-- Calculate the cost of grapes and mangoes
def cost_grapes := rate_per_kg_grapes * kg_grapes
def cost_mangoes := rate_per_kg_mangoes * kg_mangoes

-- Calculate the total amount paid
def total_amount_paid := cost_grapes + cost_mangoes

-- State the proof problem
theorem andrew_paid_1428 :
  total_amount_paid = 1428 :=
by
  -- Add the proof to verify the calculations
  sorry

end NUMINAMATH_GPT_andrew_paid_1428_l783_78355


namespace NUMINAMATH_GPT_frankie_pets_l783_78391

variable {C S P D : ℕ}

theorem frankie_pets (h1 : S = C + 6) (h2 : P = C - 1) (h3 : C + D = 6) (h4 : C + S + P + D = 19) : 
  C + S + P + D = 19 :=
  by sorry

end NUMINAMATH_GPT_frankie_pets_l783_78391


namespace NUMINAMATH_GPT_find_number_l783_78337

theorem find_number (x : ℕ) (h1 : x - 13 = 31) : x + 11 = 55 :=
  sorry

end NUMINAMATH_GPT_find_number_l783_78337


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_even_integers_l783_78318

theorem sum_of_squares_of_consecutive_even_integers (n : ℤ) (h : (2 * n - 2) * (2 * n) * (2 * n + 2) = 12 * ((2 * n - 2) + (2 * n) + (2 * n + 2))) :
  (2 * n - 2) ^ 2 + (2 * n) ^ 2 + (2 * n + 2) ^ 2 = 440 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_even_integers_l783_78318


namespace NUMINAMATH_GPT_Bobby_ate_5_pancakes_l783_78378

theorem Bobby_ate_5_pancakes
  (total_pancakes : ℕ := 21)
  (dog_eaten : ℕ := 7)
  (leftover : ℕ := 9) :
  (total_pancakes - dog_eaten - leftover = 5) := by
  sorry

end NUMINAMATH_GPT_Bobby_ate_5_pancakes_l783_78378


namespace NUMINAMATH_GPT_exists_solution_in_interval_l783_78336

theorem exists_solution_in_interval : ∃ x ∈ (Set.Ioo (3: ℝ) (4: ℝ)), Real.log x / Real.log 2 + x - 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_solution_in_interval_l783_78336


namespace NUMINAMATH_GPT_total_books_l783_78349

-- Define the number of books Victor originally had and the number he bought
def original_books : ℕ := 9
def bought_books : ℕ := 3

-- The proof problem statement: Prove Victor has a total of original_books + bought_books books
theorem total_books : original_books + bought_books = 12 := by
  -- proof will go here, using sorry to indicate it's omitted
  sorry

end NUMINAMATH_GPT_total_books_l783_78349


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_q_l783_78394

variable (p q r : Prop)

theorem sufficient_but_not_necessary_condition_for_q (hp : p → r) (hq1 : r → q) (hq2 : ¬(q → r)) : 
  (p → q) ∧ ¬(q → p) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_q_l783_78394


namespace NUMINAMATH_GPT_shark_sightings_relationship_l783_78356

theorem shark_sightings_relationship (C D R : ℕ) (h₁ : C + D = 40) (h₂ : C = R - 8) (h₃ : C = 24) :
  R = 32 :=
by
  sorry

end NUMINAMATH_GPT_shark_sightings_relationship_l783_78356


namespace NUMINAMATH_GPT_jiaqi_grade_is_95_3_l783_78350

def extracurricular_score : ℝ := 96
def mid_term_score : ℝ := 92
def final_exam_score : ℝ := 97

def extracurricular_weight : ℝ := 0.2
def mid_term_weight : ℝ := 0.3
def final_exam_weight : ℝ := 0.5

def total_grade : ℝ :=
  extracurricular_score * extracurricular_weight +
  mid_term_score * mid_term_weight +
  final_exam_score * final_exam_weight

theorem jiaqi_grade_is_95_3 : total_grade = 95.3 :=
by
  simp [total_grade, extracurricular_score, mid_term_score, final_exam_score,
    extracurricular_weight, mid_term_weight, final_exam_weight]
  sorry

end NUMINAMATH_GPT_jiaqi_grade_is_95_3_l783_78350


namespace NUMINAMATH_GPT_length_of_notebook_is_24_l783_78373

-- Definitions
def span_of_hand : ℕ := 12
def length_of_notebook (span : ℕ) : ℕ := 2 * span

-- Theorem statement that proves the question == answer given conditions
theorem length_of_notebook_is_24 :
  length_of_notebook span_of_hand = 24 :=
sorry

end NUMINAMATH_GPT_length_of_notebook_is_24_l783_78373


namespace NUMINAMATH_GPT_polar_to_rectangular_l783_78385

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 7) (h_θ : θ = π / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l783_78385


namespace NUMINAMATH_GPT_proof_problem_l783_78304

def label_sum_of_domains_specified (labels: List Nat) (domains: List Nat) : Nat :=
  let relevant_labels := labels.filter (fun l => domains.contains l)
  relevant_labels.foldl (· + ·) 0

def label_product_of_continuous_and_invertible (labels: List Nat) (properties: List Bool) : Nat :=
  let relevant_labels := labels.zip properties |>.filter (fun (_, p) => p) |>.map (·.fst)
  relevant_labels.foldl (· * ·) 1

theorem proof_problem :
  label_sum_of_domains_specified [1, 2, 3, 4] [4] = 4 ∧ label_product_of_continuous_and_invertible [1, 2, 3, 4] [true, false, true, false] = 3 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l783_78304


namespace NUMINAMATH_GPT_union_of_complements_eq_l783_78313

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_of_complements_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 4, 5, 7} →
  B = {3, 4, 5} →
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 6, 7}) :=
by
  intros hU hA hB
  sorry

end NUMINAMATH_GPT_union_of_complements_eq_l783_78313


namespace NUMINAMATH_GPT_largest_inscribed_triangle_area_l783_78323

theorem largest_inscribed_triangle_area (r : ℝ) (h_r : r = 12) : ∃ A : ℝ, A = 144 :=
by
  sorry

end NUMINAMATH_GPT_largest_inscribed_triangle_area_l783_78323


namespace NUMINAMATH_GPT_triangle_acute_angles_integer_solution_l783_78353

theorem triangle_acute_angles_integer_solution :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), (20 < x ∧ x < 27) ∧ (12 < x ∧ x < 36) ↔ (x = 21 ∨ x = 22 ∨ x = 23 ∨ x = 24 ∨ x = 25 ∨ x = 26) :=
by
  sorry

end NUMINAMATH_GPT_triangle_acute_angles_integer_solution_l783_78353


namespace NUMINAMATH_GPT_part1_part2_l783_78311

-- Definitions based on the conditions
def original_sales : ℕ := 30
def profit_per_shirt_initial : ℕ := 40

-- Additional shirts sold for each 1 yuan price reduction
def additional_shirts_per_yuan : ℕ := 2

-- Price reduction example of 3 yuan
def price_reduction_example : ℕ := 3

-- New sales quantity after 3 yuan reduction
def new_sales_quantity_example := 
  original_sales + (price_reduction_example * additional_shirts_per_yuan)

-- Prove that the sales quantity is 36 shirts for a reduction of 3 yuan
theorem part1 : new_sales_quantity_example = 36 := by
  sorry

-- General price reduction variable
def price_reduction_per_item (x : ℕ) : ℕ := x
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt_initial - x
def new_sales_quantity (x : ℕ) : ℕ := original_sales + (additional_shirts_per_yuan * x)
def daily_sales_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (new_sales_quantity x)

-- Goal for daily sales profit of 1200 yuan
def goal_profit : ℕ := 1200

-- Prove that a price reduction of 25 yuan per shirt achieves a daily sales profit of 1200 yuan
theorem part2 : daily_sales_profit 25 = goal_profit := by
  sorry

end NUMINAMATH_GPT_part1_part2_l783_78311


namespace NUMINAMATH_GPT_albert_number_l783_78335

theorem albert_number :
  ∃ (n : ℕ), (1 / (n : ℝ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) ∧ 
             ∃ m : ℕ, (1 / (m : ℝ) + 1 / 2 = 1 / 3 + 2 / (m + 1)) ∧ m ≠ n :=
sorry

end NUMINAMATH_GPT_albert_number_l783_78335


namespace NUMINAMATH_GPT_jo_age_l783_78366

theorem jo_age (j d g : ℕ) (even_j : 2 * j = j * 2) (even_d : 2 * d = d * 2) (even_g : 2 * g = g * 2)
    (h : 8 * j * d * g = 2024) : 2 * j = 46 :=
sorry

end NUMINAMATH_GPT_jo_age_l783_78366


namespace NUMINAMATH_GPT_exists_n_ge_1_le_2020_l783_78310

theorem exists_n_ge_1_le_2020
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j : ℕ, 1 ≤ i → i ≤ 2020 → 1 ≤ j → j ≤ 2020 → i ≠ j → a i ≠ a j)
  (h_periodic1 : a 2021 = a 1)
  (h_periodic2 : a 2022 = a 2) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ a n ^ 2 + a (n + 1) ^ 2 ≥ a (n + 2) ^ 2 + n ^ 2 + 3 := 
sorry

end NUMINAMATH_GPT_exists_n_ge_1_le_2020_l783_78310


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l783_78377

theorem isosceles_triangle_angles (α β γ : ℝ) (h_iso : α = β ∨ α = γ ∨ β = γ) (h_angle : α + β + γ = 180) (h_40 : α = 40 ∨ β = 40 ∨ γ = 40) :
  (α = 70 ∧ β = 70 ∧ γ = 40) ∨ (α = 40 ∧ β = 100 ∧ γ = 40) ∨ (α = 40 ∧ β = 40 ∧ γ = 100) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l783_78377


namespace NUMINAMATH_GPT_sum_of_triangle_angles_is_540_l783_78362

theorem sum_of_triangle_angles_is_540
  (A1 A3 A5 B2 B4 B6 C7 C8 C9 : ℝ)
  (H1 : A1 + A3 + A5 = 180)
  (H2 : B2 + B4 + B6 = 180)
  (H3 : C7 + C8 + C9 = 180) :
  A1 + A3 + A5 + B2 + B4 + B6 + C7 + C8 + C9 = 540 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_triangle_angles_is_540_l783_78362


namespace NUMINAMATH_GPT_lollipops_given_l783_78357

theorem lollipops_given (initial_people later_people : ℕ) (total_people groups_of_five : ℕ) :
  initial_people = 45 →
  later_people = 15 →
  total_people = initial_people + later_people →
  groups_of_five = total_people / 5 →
  total_people = 60 →
  groups_of_five = 12 :=
by intros; sorry

end NUMINAMATH_GPT_lollipops_given_l783_78357


namespace NUMINAMATH_GPT_initial_sand_amount_l783_78359

theorem initial_sand_amount (lost_sand : ℝ) (arrived_sand : ℝ)
  (h1 : lost_sand = 2.4) (h2 : arrived_sand = 1.7) :
  lost_sand + arrived_sand = 4.1 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_initial_sand_amount_l783_78359


namespace NUMINAMATH_GPT_Yasmin_children_count_l783_78397

theorem Yasmin_children_count (Y : ℕ) (h1 : 2 * Y + Y = 6) : Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_Yasmin_children_count_l783_78397


namespace NUMINAMATH_GPT_train_speed_on_time_l783_78303

theorem train_speed_on_time (v : ℕ) (t : ℕ) :
  (15 / v + 1 / 4 = 15 / 50) ∧ (t = 15) → v = 300 := by
  sorry

end NUMINAMATH_GPT_train_speed_on_time_l783_78303


namespace NUMINAMATH_GPT_green_sweets_count_l783_78384

def total_sweets := 285
def red_sweets := 49
def neither_red_nor_green_sweets := 177

theorem green_sweets_count : 
  (total_sweets - red_sweets - neither_red_nor_green_sweets) = 59 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_green_sweets_count_l783_78384


namespace NUMINAMATH_GPT_fraction_auto_installment_credit_extended_by_finance_companies_l783_78382

def total_consumer_installment_credit : ℝ := 291.6666666666667
def auto_instalment_percentage : ℝ := 0.36
def auto_finance_companies_credit_extended : ℝ := 35

theorem fraction_auto_installment_credit_extended_by_finance_companies :
  auto_finance_companies_credit_extended / (auto_instalment_percentage * total_consumer_installment_credit) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_auto_installment_credit_extended_by_finance_companies_l783_78382


namespace NUMINAMATH_GPT_area_of_playground_l783_78330

variable (l w : ℝ)

-- Conditions:
def perimeter_eq : Prop := 2 * l + 2 * w = 90
def length_three_times_width : Prop := l = 3 * w

-- Theorem:
theorem area_of_playground (h1 : perimeter_eq l w) (h2 : length_three_times_width l w) : l * w = 379.6875 :=
  sorry

end NUMINAMATH_GPT_area_of_playground_l783_78330


namespace NUMINAMATH_GPT_abcdefg_defghij_value_l783_78334

variable (a b c d e f g h i : ℚ)

theorem abcdefg_defghij_value :
  (a / b = -7 / 3) →
  (b / c = -5 / 2) →
  (c / d = 2) →
  (d / e = -3 / 2) →
  (e / f = 4 / 3) →
  (f / g = -1 / 4) →
  (g / h = 3 / -5) →
  (abcdefg / defghij = (-21 / 16) * (c / i)) :=
by
  sorry

end NUMINAMATH_GPT_abcdefg_defghij_value_l783_78334


namespace NUMINAMATH_GPT_problem_statement_l783_78395

variables {a b c : ℝ}

theorem problem_statement 
  (h1 : a^2 + a * b + b^2 = 9)
  (h2 : b^2 + b * c + c^2 = 52)
  (h3 : c^2 + c * a + a^2 = 49) : 
  (49 * b^2 - 33 * b * c + 9 * c^2) / a^2 = 52 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l783_78395


namespace NUMINAMATH_GPT_malachi_selfies_total_l783_78392

theorem malachi_selfies_total (x y : ℕ) 
  (h_ratio : 10 * y = 17 * x)
  (h_diff : y = x + 630) : 
  x + y = 2430 :=
sorry

end NUMINAMATH_GPT_malachi_selfies_total_l783_78392


namespace NUMINAMATH_GPT_find_c_l783_78301

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_c (a b c : ℝ) 
  (h1 : perpendicular (a / 2) (-2 / b))
  (h2 : a = b)
  (h3 : a * 1 - 2 * (-5) = c) 
  (h4 : 2 * 1 + b * (-5) = -c) : 
  c = 13 := by
  sorry

end NUMINAMATH_GPT_find_c_l783_78301


namespace NUMINAMATH_GPT_value_of_x_l783_78347

theorem value_of_x (x : ℝ) (h : x = 12 + (20 / 100) * 12) : x = 14.4 :=
by sorry

end NUMINAMATH_GPT_value_of_x_l783_78347


namespace NUMINAMATH_GPT_count_even_divisors_8_l783_78324

theorem count_even_divisors_8! :
  ∃ (even_divisors total : ℕ),
    even_divisors = 84 ∧
    total = 56 :=
by
  /-
    To formulate the problem in Lean:
    We need to establish two main facts:
    1. The count of even divisors of 8! is 84.
    2. The count of those even divisors that are multiples of both 2 and 3 is 56.
  -/
  sorry

end NUMINAMATH_GPT_count_even_divisors_8_l783_78324


namespace NUMINAMATH_GPT_rope_length_l783_78341

theorem rope_length (h1 : ∃ x : ℝ, 4 * x = 20) : 
  ∃ l : ℝ, l = 35 := by
sorry

end NUMINAMATH_GPT_rope_length_l783_78341


namespace NUMINAMATH_GPT_total_amount_spent_l783_78322

theorem total_amount_spent (num_pigs num_hens avg_price_hen avg_price_pig : ℕ)
                          (h_num_pigs : num_pigs = 3)
                          (h_num_hens : num_hens = 10)
                          (h_avg_price_hen : avg_price_hen = 30)
                          (h_avg_price_pig : avg_price_pig = 300) :
                          num_hens * avg_price_hen + num_pigs * avg_price_pig = 1200 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l783_78322


namespace NUMINAMATH_GPT_remainder_17_pow_45_div_5_l783_78302

theorem remainder_17_pow_45_div_5 : (17 ^ 45) % 5 = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_remainder_17_pow_45_div_5_l783_78302


namespace NUMINAMATH_GPT_julia_tuesday_kids_l783_78332

-- Definitions based on the given conditions in the problem.
def monday_kids : ℕ := 15
def monday_tuesday_kids : ℕ := 33

-- The problem statement to prove the number of kids played with on Tuesday.
theorem julia_tuesday_kids :
  (∃ tuesday_kids : ℕ, tuesday_kids = monday_tuesday_kids - monday_kids) →
  18 = monday_tuesday_kids - monday_kids :=
by
  intro h
  sorry

end NUMINAMATH_GPT_julia_tuesday_kids_l783_78332


namespace NUMINAMATH_GPT_volvox_pentagons_heptagons_diff_l783_78399

-- Given conditions
variables (V E F f_5 f_6 f_7 : ℕ)

-- Euler's polyhedron formula
axiom euler_formula : V - E + F = 2

-- Each edge is shared by two faces
axiom edge_formula : 2 * E = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Each vertex shared by three faces
axiom vertex_formula : 3 * V = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Total number of faces equals sum of individual face types 
def total_faces : ℕ := f_5 + f_6 + f_7

-- Prove that the number of pentagonal cells exceeds the number of heptagonal cells by 12
theorem volvox_pentagons_heptagons_diff : f_5 - f_7 = 12 := 
sorry

end NUMINAMATH_GPT_volvox_pentagons_heptagons_diff_l783_78399


namespace NUMINAMATH_GPT_find_S30_l783_78319

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_S30_l783_78319


namespace NUMINAMATH_GPT_problem_l783_78369

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.log x + (a + 1) * (1 / x - 2)

theorem problem (a x : ℝ) (ha_pos : a > 0) :
  f a x > - (a^2 / (a + 1)) - 2 :=
sorry

end NUMINAMATH_GPT_problem_l783_78369


namespace NUMINAMATH_GPT_function_pass_through_point_l783_78320

theorem function_pass_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), y = a^(x-2) - 1 ∧ (x, y) = (2, 0) := 
by
  use 2
  use 0
  sorry

end NUMINAMATH_GPT_function_pass_through_point_l783_78320


namespace NUMINAMATH_GPT_xiaoli_estimate_larger_l783_78309

theorem xiaoli_estimate_larger (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : 
  (1.1 * x) / (0.9 * y) > x / y :=
by
  sorry

end NUMINAMATH_GPT_xiaoli_estimate_larger_l783_78309


namespace NUMINAMATH_GPT_curve_is_line_segment_l783_78364

noncomputable def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = Real.cos θ ^ 2 ∧ p.2 = Real.sin θ ^ 2}

theorem curve_is_line_segment :
  (∀ p ∈ parametric_curve, p.1 + p.2 = 1 ∧ p.1 ∈ Set.Icc 0 1) :=
by
  sorry

end NUMINAMATH_GPT_curve_is_line_segment_l783_78364


namespace NUMINAMATH_GPT_find_two_digit_number_l783_78345

theorem find_two_digit_number (x : ℕ) (h1 : (x + 3) % 3 = 0) (h2 : (x + 7) % 7 = 0) (h3 : (x - 4) % 4 = 0) : x = 84 := 
by
  -- Place holder for the proof
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l783_78345


namespace NUMINAMATH_GPT_percentage_of_total_money_raised_from_donations_l783_78381

-- Define the conditions
def max_donation := 1200
def num_donors_max := 500
def half_donation := max_donation / 2
def num_donors_half := 3 * num_donors_max
def total_money_raised := 3750000

-- Define the amounts collected from each group
def amount_from_max_donors := num_donors_max * max_donation
def amount_from_half_donors := num_donors_half * half_donation
def total_amount_from_donations := amount_from_max_donors + amount_from_half_donors

-- Define the percentage calculation
def percentage_of_total := (total_amount_from_donations / total_money_raised) * 100

-- State the theorem (but not the proof)
theorem percentage_of_total_money_raised_from_donations : 
  percentage_of_total = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_of_total_money_raised_from_donations_l783_78381


namespace NUMINAMATH_GPT_remainder_when_P_divided_by_DD_l783_78380

noncomputable def remainder (a b : ℕ) : ℕ := a % b

theorem remainder_when_P_divided_by_DD' (P D Q R D' Q'' R'' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  remainder P (D * D') = R :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_P_divided_by_DD_l783_78380


namespace NUMINAMATH_GPT_area_of_region_l783_78321

theorem area_of_region : 
  (∃ x y : ℝ, |5 * x - 10| + |4 * y + 20| ≤ 10) →
  ∃ area : ℝ, 
  area = 10 :=
sorry

end NUMINAMATH_GPT_area_of_region_l783_78321


namespace NUMINAMATH_GPT_table_height_l783_78396

theorem table_height (l w h : ℝ) (h1 : l + h - w = 38) (h2 : w + h - l = 34) : h = 36 :=
by
  sorry

end NUMINAMATH_GPT_table_height_l783_78396


namespace NUMINAMATH_GPT_length_AB_is_4_l783_78333

section HyperbolaProof

/-- Define the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 8) = 1

/-- Define the line l given by x = 2√6 -/
def line_l (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 6

/-- Define the condition for intersection points -/
def intersect_points (x y : ℝ) : Prop :=
  hyperbola x y ∧ line_l x

/-- Prove the length of the line segment AB is 4 -/
theorem length_AB_is_4 :
  ∀ y : ℝ, intersect_points (2 * Real.sqrt 6) y → |y| = 2 → length_AB = 4 :=
sorry

end HyperbolaProof

end NUMINAMATH_GPT_length_AB_is_4_l783_78333


namespace NUMINAMATH_GPT_quadratic_eq_solution_1_quadratic_eq_solution_2_l783_78393

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_eq_solution_1_quadratic_eq_solution_2_l783_78393


namespace NUMINAMATH_GPT_pool_capacity_l783_78361

noncomputable def total_capacity : ℝ := 1000

theorem pool_capacity
    (C : ℝ)
    (H1 : 0.75 * C = 0.45 * C + 300)
    (H2 : 300 / 0.3 = 1000)
    : C = total_capacity :=
by
  -- Solution steps are omitted, proof goes here.
  sorry

end NUMINAMATH_GPT_pool_capacity_l783_78361


namespace NUMINAMATH_GPT_polar_eq_of_circle_product_of_distances_MA_MB_l783_78365

noncomputable def circle_center := (2, Real.pi / 3)
noncomputable def circle_radius := 2

-- Polar equation of the circle
theorem polar_eq_of_circle :
  ∀ (ρ θ : ℝ),
    (circle_center.snd = Real.pi / 3) →
    ρ = 2 * 2 * Real.cos (θ - circle_center.snd) → 
    ρ = 4 * Real.cos (θ - (Real.pi / 3)) :=
by 
  sorry

noncomputable def point_M := (1, -2)

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ := 
  (1 + 1/2 * t, -2 + Real.sqrt 3 / 2 * t)

noncomputable def cartesian_center := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
noncomputable def cartesian_radius := 2

-- Cartesian form of the circle equation from the polar coordinates
noncomputable def cartesian_eq (x y : ℝ) : Prop :=
  (x - cartesian_center.fst)^2 + (y - cartesian_center.snd)^2 = circle_radius^2

-- Product of distances |MA| * |MB|
theorem product_of_distances_MA_MB :
  ∃ (t1 t2 : ℝ),
  (∀ t, parametric_line t ∈ {p : ℝ × ℝ | cartesian_eq p.fst p.snd}) → 
  (point_M.fst, point_M.snd) = (1, -2) →
  t1 * t2 = 3 + 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_polar_eq_of_circle_product_of_distances_MA_MB_l783_78365


namespace NUMINAMATH_GPT_range_of_a_l783_78344

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₀ d : ℝ), ∀ n, a n = a₀ + n * d

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a_seq) 
  (h2 : a_seq 0 = a)
  (h3 : ∀ n, b n = (1 + a_seq n) / a_seq n)
  (h4 : ∀ n : ℕ, 0 < n → b n ≥ b 8) :
  -8 < a ∧ a < -7 :=
sorry

end NUMINAMATH_GPT_range_of_a_l783_78344


namespace NUMINAMATH_GPT_kayak_rental_cost_l783_78306

theorem kayak_rental_cost
    (canoe_cost_per_day : ℕ := 14)
    (total_revenue : ℕ := 288)
    (canoe_kayak_ratio : ℕ × ℕ := (3, 2))
    (canoe_kayak_difference : ℕ := 4)
    (number_of_kayaks : ℕ := 8)
    (number_of_canoes : ℕ := number_of_kayaks + canoe_kayak_difference)
    (canoe_revenue : ℕ := number_of_canoes * canoe_cost_per_day) :
    number_of_kayaks * kayak_cost_per_day = total_revenue - canoe_revenue →
    kayak_cost_per_day = 15 := 
by
  sorry

end NUMINAMATH_GPT_kayak_rental_cost_l783_78306


namespace NUMINAMATH_GPT_sufficient_condition_l783_78376

theorem sufficient_condition 
  (x y z : ℤ)
  (H : x = y ∧ y = z)
  : x * (x - y) + y * (y - z) + z * (z - x) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_condition_l783_78376


namespace NUMINAMATH_GPT_solution_system_solution_rational_l783_78343

-- Definitions for the system of equations
def sys_eq_1 (x y : ℤ) : Prop := 2 * x - y = 3
def sys_eq_2 (x y : ℤ) : Prop := x + y = -12

-- Theorem to prove the solution of the system of equations
theorem solution_system (x y : ℤ) (h1 : sys_eq_1 x y) (h2 : sys_eq_2 x y) : x = -3 ∧ y = -9 :=
by {
  sorry
}

-- Definition for the rational equation
def rational_eq (x : ℤ) : Prop := (2 / (1 - x) : ℚ) + 1 = (x / (1 + x) : ℚ)

-- Theorem to prove the solution of the rational equation
theorem solution_rational (x : ℤ) (h : rational_eq x) : x = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_system_solution_rational_l783_78343


namespace NUMINAMATH_GPT_cost_price_of_radio_l783_78354

-- Define the conditions
def selling_price : ℝ := 1335
def loss_percentage : ℝ := 0.11

-- Define what we need to prove
theorem cost_price_of_radio (C : ℝ) (h1 : selling_price = 0.89 * C) : C = 1500 :=
by
  -- This is where we would put the proof, but we can leave it as a sorry for now.
  sorry

end NUMINAMATH_GPT_cost_price_of_radio_l783_78354


namespace NUMINAMATH_GPT_max_value_of_expression_l783_78367

noncomputable def max_expression_value (x y : ℝ) : ℝ :=
  let expr := x^2 + 6 * y + 2
  14

theorem max_value_of_expression 
  (x y : ℝ) (h : x^2 + y^2 = 4) : ∃ (M : ℝ), M = 14 ∧ ∀ x y, x^2 + y^2 = 4 → x^2 + 6 * y + 2 ≤ M :=
  by
    use 14
    sorry

end NUMINAMATH_GPT_max_value_of_expression_l783_78367


namespace NUMINAMATH_GPT_train_length_is_500_l783_78315

def speed_kmph : ℕ := 360
def time_sec : ℕ := 5

def speed_mps (v_kmph : ℕ) : ℕ :=
  v_kmph * 1000 / 3600

def length_of_train (v_mps : ℕ) (t_sec : ℕ) : ℕ :=
  v_mps * t_sec

theorem train_length_is_500 :
  length_of_train (speed_mps speed_kmph) time_sec = 500 := 
sorry

end NUMINAMATH_GPT_train_length_is_500_l783_78315


namespace NUMINAMATH_GPT_polygon_sides_eight_l783_78327

theorem polygon_sides_eight (n : ℕ) (h : 180 * (n - 2) = 3 * 360) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eight_l783_78327


namespace NUMINAMATH_GPT_tangent_identity_problem_l783_78328

theorem tangent_identity_problem 
    (α β : ℝ) 
    (h1 : Real.tan (α + β) = 1) 
    (h2 : Real.tan (α - π / 3) = 1 / 3) 
    : Real.tan (β + π / 3) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_tangent_identity_problem_l783_78328


namespace NUMINAMATH_GPT_fly_distance_from_ceiling_l783_78390

theorem fly_distance_from_ceiling (x y z : ℝ) (hx : x = 2) (hy : y = 6) (hP : x^2 + y^2 + z^2 = 100) : z = 2 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_fly_distance_from_ceiling_l783_78390


namespace NUMINAMATH_GPT_parabola_c_value_l783_78372

theorem parabola_c_value (b c : ℝ) 
  (h1 : 5 = 2 * 1^2 + b * 1 + c)
  (h2 : 17 = 2 * 3^2 + b * 3 + c) : 
  c = 5 := 
by
  sorry

end NUMINAMATH_GPT_parabola_c_value_l783_78372


namespace NUMINAMATH_GPT_recurrence_relation_l783_78398

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_GPT_recurrence_relation_l783_78398


namespace NUMINAMATH_GPT_common_divisor_l783_78339

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end NUMINAMATH_GPT_common_divisor_l783_78339


namespace NUMINAMATH_GPT_initial_average_mark_l783_78329

theorem initial_average_mark (A : ℝ) (n_total n_excluded remaining_students_avg : ℝ) 
  (h1 : n_total = 25) 
  (h2 : n_excluded = 5) 
  (h3 : remaining_students_avg = 90)
  (excluded_students_avg : ℝ)
  (h_excluded_avg : excluded_students_avg = 40)
  (A_def : (n_total * A) = (n_excluded * excluded_students_avg + (n_total - n_excluded) * remaining_students_avg)) :
  A = 80 := 
by
  sorry

end NUMINAMATH_GPT_initial_average_mark_l783_78329


namespace NUMINAMATH_GPT_smallest_common_term_larger_than_2023_l783_78387

noncomputable def a_seq (n : ℕ) : ℤ :=
  3 * n - 2

noncomputable def b_seq (m : ℕ) : ℤ :=
  10 * m - 8

theorem smallest_common_term_larger_than_2023 :
  ∃ (n m : ℕ), a_seq n = b_seq m ∧ a_seq n > 2023 ∧ a_seq n = 2032 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_common_term_larger_than_2023_l783_78387


namespace NUMINAMATH_GPT_mass_percentage_Al_in_Al2O3_l783_78305

-- Define the atomic masses and formula unit
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

-- Define the statement for the mass percentage of Al in Al2O3
theorem mass_percentage_Al_in_Al2O3 : (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100 = 52.91 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_mass_percentage_Al_in_Al2O3_l783_78305


namespace NUMINAMATH_GPT_fraction_given_to_jerry_l783_78351

-- Define the problem conditions
def initial_apples := 2
def slices_per_apple := 8
def total_slices := initial_apples * slices_per_apple -- 2 * 8 = 16

def remaining_slices_after_eating := 5
def slices_before_eating := remaining_slices_after_eating * 2 -- 5 * 2 = 10
def slices_given_to_jerry := total_slices - slices_before_eating -- 16 - 10 = 6

-- Define the proof statement to verify that the fraction of slices given to Jerry is 3/8
theorem fraction_given_to_jerry : (slices_given_to_jerry : ℚ) / total_slices = 3 / 8 :=
by
  -- skip the actual proof, just outline the goal
  sorry

end NUMINAMATH_GPT_fraction_given_to_jerry_l783_78351


namespace NUMINAMATH_GPT_plane_split_into_four_regions_l783_78368

theorem plane_split_into_four_regions {x y : ℝ} :
  (y = 3 * x) ∨ (y = (1 / 3) * x - (2 / 3)) →
  ∃ r : ℕ, r = 4 :=
by
  intro h
  -- We must show that these lines split the plane into 4 regions
  sorry

end NUMINAMATH_GPT_plane_split_into_four_regions_l783_78368


namespace NUMINAMATH_GPT_annual_interest_rate_l783_78317

-- Definitions based on conditions
def initial_amount : ℝ := 1000
def spent_amount : ℝ := 440
def final_amount : ℝ := 624

-- The main theorem
theorem annual_interest_rate (x : ℝ) : 
  (initial_amount * (1 + x) - spent_amount) * (1 + x) = final_amount →
  x = 0.04 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l783_78317


namespace NUMINAMATH_GPT_solve_quadratic_equation_l783_78331

theorem solve_quadratic_equation (m : ℝ) : 9 * m^2 - (2 * m + 1)^2 = 0 → m = 1 ∨ m = -1/5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l783_78331


namespace NUMINAMATH_GPT_units_digit_of_23_mul_51_squared_l783_78326

theorem units_digit_of_23_mul_51_squared : 
  ∀ n m : ℕ, (n % 10 = 3) ∧ ((m^2 % 10) = 1) → (n * m^2 % 10) = 3 :=
by
  intros n m h
  sorry

end NUMINAMATH_GPT_units_digit_of_23_mul_51_squared_l783_78326


namespace NUMINAMATH_GPT_distance_from_minus_one_is_four_or_minus_six_l783_78308

theorem distance_from_minus_one_is_four_or_minus_six :
  {x : ℝ | abs (x + 1) = 5} = {-6, 4} :=
sorry

end NUMINAMATH_GPT_distance_from_minus_one_is_four_or_minus_six_l783_78308


namespace NUMINAMATH_GPT_paintable_wall_area_l783_78312

theorem paintable_wall_area :
  let bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let doorway_window_area := 70
  let area_one_bedroom := 
    2 * (length * height) + 2 * (width * height) - doorway_window_area
  let total_paintable_area := bedrooms * area_one_bedroom
  total_paintable_area = 1520 := by
  sorry

end NUMINAMATH_GPT_paintable_wall_area_l783_78312


namespace NUMINAMATH_GPT_pipe_a_filling_time_l783_78300

theorem pipe_a_filling_time
  (pipeA_fill_time : ℝ)
  (pipeB_fill_time : ℝ)
  (both_pipes_open : Bool)
  (pipeB_shutoff_time : ℝ)
  (overflow_time : ℝ)
  (pipeB_rate : ℝ)
  (combined_rate : ℝ)
  (a_filling_time : ℝ) :
  pipeA_fill_time = 1 / 2 :=
by
  -- Definitions directly from conditions in a)
  let pipeA_fill_time := a_filling_time
  let pipeB_fill_time := 1  -- Pipe B fills in 1 hour
  let both_pipes_open := True
  let pipeB_shutoff_time := 0.5 -- Pipe B shuts 30 minutes before overflow
  let overflow_time := 0.5  -- Tank overflows in 30 minutes
  let pipeB_rate := 1 / pipeB_fill_time
  
  -- Goal to prove
  sorry

end NUMINAMATH_GPT_pipe_a_filling_time_l783_78300


namespace NUMINAMATH_GPT_fraction_value_is_one_fourth_l783_78342

theorem fraction_value_is_one_fourth (k : Nat) (hk : k ≥ 1) :
  (10^k + 6 * (10^k - 1) / 9) / (60 * (10^k - 1) / 9 + 4) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_is_one_fourth_l783_78342
