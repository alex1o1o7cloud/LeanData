import Mathlib

namespace NUMINAMATH_GPT_seven_books_cost_l1291_129192

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end NUMINAMATH_GPT_seven_books_cost_l1291_129192


namespace NUMINAMATH_GPT_distance_vancouver_calgary_l1291_129199

theorem distance_vancouver_calgary : 
  ∀ (map_distance : ℝ) (scale : ℝ) (terrain_factor : ℝ), 
    map_distance = 12 →
    scale = 35 →
    terrain_factor = 1.1 →
    map_distance * scale * terrain_factor = 462 := by
  intros map_distance scale terrain_factor 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_distance_vancouver_calgary_l1291_129199


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1291_129104

variable {a : ℕ → ℝ} {d : ℝ} -- Declare the sequence and common difference

-- Define the arithmetic sequence property
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 5 + a 10 = 12 ∧ arithmetic_sequence a d

-- Main theorem statement
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h : given_conditions a d) :
  3 * a 7 + a 9 = 24 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1291_129104


namespace NUMINAMATH_GPT_sin_and_tan_alpha_l1291_129150

variable (x : ℝ) (α : ℝ)

-- Conditions
def vertex_is_origin : Prop := true
def initial_side_is_non_negative_half_axis : Prop := true
def terminal_side_passes_through_P : Prop := ∃ (P : ℝ × ℝ), P = (x, -Real.sqrt 2)
def cos_alpha_eq : Prop := x ≠ 0 ∧ Real.cos α = (Real.sqrt 3 / 6) * x

-- Proof Problem Statement
theorem sin_and_tan_alpha (h1 : vertex_is_origin) 
                         (h2 : initial_side_is_non_negative_half_axis) 
                         (h3 : terminal_side_passes_through_P x) 
                         (h4 : cos_alpha_eq x α) 
                         : Real.sin α = -Real.sqrt 6 / 6 ∧ (Real.tan α = Real.sqrt 5 / 5 ∨ Real.tan α = -Real.sqrt 5 / 5) := 
sorry

end NUMINAMATH_GPT_sin_and_tan_alpha_l1291_129150


namespace NUMINAMATH_GPT_kim_spends_time_on_coffee_l1291_129186

noncomputable def time_per_employee_status_update : ℕ := 2
noncomputable def time_per_employee_payroll_update : ℕ := 3
noncomputable def number_of_employees : ℕ := 9
noncomputable def total_morning_routine_time : ℕ := 50

theorem kim_spends_time_on_coffee :
  ∃ C : ℕ, C + (time_per_employee_status_update * number_of_employees) + 
  (time_per_employee_payroll_update * number_of_employees) = total_morning_routine_time ∧
  C = 5 :=
by
  sorry

end NUMINAMATH_GPT_kim_spends_time_on_coffee_l1291_129186


namespace NUMINAMATH_GPT_bills_difference_l1291_129166

noncomputable def Mike_tip : ℝ := 5
noncomputable def Joe_tip : ℝ := 10
noncomputable def Mike_percentage : ℝ := 20
noncomputable def Joe_percentage : ℝ := 25

theorem bills_difference
  (m j : ℝ)
  (Mike_condition : (Mike_percentage / 100) * m = Mike_tip)
  (Joe_condition : (Joe_percentage / 100) * j = Joe_tip) :
  |m - j| = 15 :=
by
  sorry

end NUMINAMATH_GPT_bills_difference_l1291_129166


namespace NUMINAMATH_GPT_number_of_players_l1291_129100

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

end NUMINAMATH_GPT_number_of_players_l1291_129100


namespace NUMINAMATH_GPT_machine_a_production_rate_l1291_129145

/-
Given:
1. Machine p and machine q are each used to manufacture 440 sprockets.
2. Machine q produces 10% more sprockets per hour than machine a.
3. It takes machine p 10 hours longer to produce 440 sprockets than machine q.

Prove that machine a produces 4 sprockets per hour.
-/

theorem machine_a_production_rate (T A : ℝ) (hq : 440 = T * (1.1 * A)) (hp : 440 = (T + 10) * A) : A = 4 := 
by
  sorry

end NUMINAMATH_GPT_machine_a_production_rate_l1291_129145


namespace NUMINAMATH_GPT_intersection_eq_zero_set_l1291_129130

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | x^2 ≤ 0}

theorem intersection_eq_zero_set : M ∩ N = {0} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_zero_set_l1291_129130


namespace NUMINAMATH_GPT_smallest_M_convex_quadrilateral_l1291_129190

section ConvexQuadrilateral

-- Let a, b, c, d be the sides of a convex quadrilateral
variables {a b c d M : ℝ}

-- Condition to ensure that a, b, c, d are the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d < 360

-- The theorem statement
theorem smallest_M_convex_quadrilateral (hconvex : is_convex_quadrilateral a b c d) : ∃ M, (∀ a b c d, is_convex_quadrilateral a b c d → (a^2 + b^2) / (c^2 + d^2) > M) ∧ M = 1/2 :=
by sorry

end ConvexQuadrilateral

end NUMINAMATH_GPT_smallest_M_convex_quadrilateral_l1291_129190


namespace NUMINAMATH_GPT_tenth_term_arithmetic_seq_l1291_129181

theorem tenth_term_arithmetic_seq :
  let a₁ : ℚ := 1 / 2
  let a₂ : ℚ := 5 / 6
  let d : ℚ := a₂ - a₁
  let a₁₀ : ℚ := a₁ + 9 * d
  a₁₀ = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_seq_l1291_129181


namespace NUMINAMATH_GPT_exists_consecutive_non_primes_l1291_129142

theorem exists_consecutive_non_primes (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ i : ℕ, i < k → ¬Nat.Prime (n + i) := 
sorry

end NUMINAMATH_GPT_exists_consecutive_non_primes_l1291_129142


namespace NUMINAMATH_GPT_max_median_soda_cans_l1291_129163

theorem max_median_soda_cans (total_customers total_cans : ℕ) 
    (h_customers : total_customers = 120)
    (h_cans : total_cans = 300) 
    (h_min_cans_per_customer : ∀ (n : ℕ), n < total_customers → 2 ≤ n) :
    ∃ (median : ℝ), median = 3.5 := 
sorry

end NUMINAMATH_GPT_max_median_soda_cans_l1291_129163


namespace NUMINAMATH_GPT_least_number_of_groups_l1291_129141

theorem least_number_of_groups (total_players : ℕ) (max_per_group : ℕ) (h1 : total_players = 30) (h2 : max_per_group = 12) : ∃ (groups : ℕ), groups = 3 := 
by {
  -- Mathematical conditions and solution to be formalized here
  sorry
}

end NUMINAMATH_GPT_least_number_of_groups_l1291_129141


namespace NUMINAMATH_GPT_quadratic_solution_difference_l1291_129111

theorem quadratic_solution_difference (x : ℝ) :
  ∀ x : ℝ, (x^2 - 5*x + 15 = x + 55) → (∃ a b : ℝ, a ≠ b ∧ x^2 - 6*x - 40 = 0 ∧ abs (a - b) = 14) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_difference_l1291_129111


namespace NUMINAMATH_GPT_proof1_proof2_l1291_129108

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

def recurrence_relation : Prop :=
  ∀ n, a (n + 3) - 9 * a (n + 2) + 27 * a (n + 1) - 27 * a n = 0

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n, a n * x^n

def series_evaluation (x : ℝ) : Prop :=
  series_sum x = (1 - 3*x + 18*x^2) / (1 - 3*x)^3

theorem proof1 : recurrence_relation := 
  by sorry

theorem proof2 : ∀ x : ℝ, series_evaluation x := 
  by sorry

end NUMINAMATH_GPT_proof1_proof2_l1291_129108


namespace NUMINAMATH_GPT_cindy_correct_answer_l1291_129151

theorem cindy_correct_answer (x : ℝ) (h : (x - 5) / 7 = 15) :
  (x - 7) / 5 = 20.6 :=
by
  sorry

end NUMINAMATH_GPT_cindy_correct_answer_l1291_129151


namespace NUMINAMATH_GPT_meal_cost_l1291_129109

/-- 
    Define the cost of a meal consisting of one sandwich, one cup of coffee, and one piece of pie 
    given the costs of two different meals.
-/
theorem meal_cost (s c p : ℝ) (h1 : 2 * s + 5 * c + p = 5) (h2 : 3 * s + 8 * c + p = 7) :
    s + c + p = 3 :=
by
  sorry

end NUMINAMATH_GPT_meal_cost_l1291_129109


namespace NUMINAMATH_GPT_jenny_ate_more_than_thrice_mike_l1291_129175

theorem jenny_ate_more_than_thrice_mike :
  let mike_ate := 20
  let jenny_ate := 65
  jenny_ate - 3 * mike_ate = 5 :=
by
  let mike_ate := 20
  let jenny_ate := 65
  have : jenny_ate - 3 * mike_ate = 5 := by
    sorry
  exact this

end NUMINAMATH_GPT_jenny_ate_more_than_thrice_mike_l1291_129175


namespace NUMINAMATH_GPT_statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l1291_129116

-- Definitions of conditions
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℝ)
  (angles : Fin n → ℝ)

def circumscribed (P : Polygon n) : Prop := sorry -- Definition of circumscribed
def inscribed (P : Polygon n) : Prop := sorry -- Definition of inscribed
def equal_sides (P : Polygon n) : Prop := ∀ i j, P.sides i = P.sides j
def equal_angles (P : Polygon n) : Prop := ∀ i j, P.angles i = P.angles j

-- The statements to be proved
theorem statement_I : ∀ P : Polygon n, circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_II : ∃ P : Polygon n, inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_III : ∃ P : Polygon n, circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_IV : ∀ P : Polygon n, inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_V : ∀ (P : Polygon 5), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VI : ∀ (P : Polygon 6), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VII : ∀ (P : Polygon 5), inscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VIII : ∃ (P : Polygon 6), inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_IX : ∀ (P : Polygon 5), circumscribed P → equal_angles P → equal_sides P := sorry

theorem statement_X : ∃ (P : Polygon 6), circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_XI : ∀ (P : Polygon 5), inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_XII : ∀ (P : Polygon 6), inscribed P → equal_angles P → equal_sides P := sorry

end NUMINAMATH_GPT_statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l1291_129116


namespace NUMINAMATH_GPT_least_number_of_plates_needed_l1291_129173

theorem least_number_of_plates_needed
  (cubes : ℕ)
  (cube_dim : ℕ)
  (temp_limit : ℕ)
  (plates_exist : ∀ (n : ℕ), n > temp_limit → ∃ (p : ℕ), p = 21) :
  cubes = 512 ∧ cube_dim = 8 → temp_limit > 0 → 21 = 7 + 7 + 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_number_of_plates_needed_l1291_129173


namespace NUMINAMATH_GPT_union_M_N_is_R_l1291_129118

open Set

/-- Define the sets M and N -/
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

/-- Main goal: prove M ∪ N = ℝ -/
theorem union_M_N_is_R : M ∪ N = univ :=
by
  sorry

end NUMINAMATH_GPT_union_M_N_is_R_l1291_129118


namespace NUMINAMATH_GPT_least_possible_b_l1291_129170

noncomputable def a : ℕ := 8

theorem least_possible_b (b : ℕ) (h1 : ∀ n : ℕ, n > 0 → a.factors.count n = 1 → a = n^3)
  (h2 : b.factors.count a = 1)
  (h3 : b % a = 0) :
  b = 24 :=
sorry

end NUMINAMATH_GPT_least_possible_b_l1291_129170


namespace NUMINAMATH_GPT_circle_locus_l1291_129155

theorem circle_locus (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  13 * a^2 + 49 * b^2 - 12 * a - 1 = 0 := 
sorry

end NUMINAMATH_GPT_circle_locus_l1291_129155


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1291_129168

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x_0 : ℝ, x_0^2 < 0) := sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1291_129168


namespace NUMINAMATH_GPT_total_number_of_pages_l1291_129148

variable (x : ℕ)

-- Conditions
def first_day_remaining : ℕ := x - (x / 6 + 10)
def second_day_remaining : ℕ := first_day_remaining x - (first_day_remaining x / 5 + 20)
def third_day_remaining : ℕ := second_day_remaining x - (second_day_remaining x / 4 + 25)
def final_remaining : Prop := third_day_remaining x = 100

-- Theorem statement
theorem total_number_of_pages : final_remaining x → x = 298 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_total_number_of_pages_l1291_129148


namespace NUMINAMATH_GPT_solve_for_x_l1291_129121

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1291_129121


namespace NUMINAMATH_GPT_profit_per_meter_is_20_l1291_129194

-- Define given conditions
def selling_price_total (n : ℕ) (price : ℕ) : ℕ := n * price
def cost_price_per_meter : ℕ := 85
def selling_price_total_85_meters : ℕ := 8925

-- Define the expected profit per meter
def expected_profit_per_meter : ℕ := 20

-- Rewrite the problem statement: Prove that with given conditions the profit per meter is Rs. 20
theorem profit_per_meter_is_20 
  (n : ℕ := 85)
  (sp : ℕ := selling_price_total_85_meters)
  (cp_pm : ℕ := cost_price_per_meter) 
  (expected_profit : ℕ := expected_profit_per_meter) :
  (sp - n * cp_pm) / n = expected_profit :=
by
  sorry

end NUMINAMATH_GPT_profit_per_meter_is_20_l1291_129194


namespace NUMINAMATH_GPT_f_positive_l1291_129126

variable (f : ℝ → ℝ)

-- f is a differentiable function on ℝ
variable (hf : differentiable ℝ f)

-- Condition: (x+1)f(x) + x f''(x) > 0
variable (H : ∀ x, (x + 1) * f x + x * (deriv^[2]) f x > 0)

-- Prove: ∀ x, f x > 0
theorem f_positive : ∀ x, f x > 0 := 
by
  sorry

end NUMINAMATH_GPT_f_positive_l1291_129126


namespace NUMINAMATH_GPT_original_number_l1291_129189

theorem original_number (N : ℤ) : (∃ k : ℤ, N - 7 = 12 * k) → N = 19 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_original_number_l1291_129189


namespace NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_5_and_6_l1291_129183

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_5_and_6_l1291_129183


namespace NUMINAMATH_GPT_highest_water_level_changes_on_tuesday_l1291_129129

def water_levels : List (String × Float) :=
  [("Monday", 0.03), ("Tuesday", 0.41), ("Wednesday", 0.25), ("Thursday", 0.10),
   ("Friday", 0.0), ("Saturday", -0.13), ("Sunday", -0.2)]

theorem highest_water_level_changes_on_tuesday :
  ∃ d : String, d = "Tuesday" ∧ ∀ d' : String × Float, d' ∈ water_levels → d'.snd ≤ 0.41 := by
  sorry

end NUMINAMATH_GPT_highest_water_level_changes_on_tuesday_l1291_129129


namespace NUMINAMATH_GPT_find_v₃_value_l1291_129172

def f (x : ℕ) : ℕ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def v₃_expr (x : ℕ) : ℕ := (((7 * x + 6) * x + 5) * x + 4)

theorem find_v₃_value : v₃_expr 3 = 262 := by
  sorry

end NUMINAMATH_GPT_find_v₃_value_l1291_129172


namespace NUMINAMATH_GPT_min_value_proof_l1291_129160

noncomputable def min_value (a b : ℝ) : ℝ := (1 : ℝ)/a + (1 : ℝ)/b

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) :
  min_value a b = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l1291_129160


namespace NUMINAMATH_GPT_jose_total_caps_l1291_129124

def initial_caps := 26
def additional_caps := 13
def total_caps := initial_caps + additional_caps

theorem jose_total_caps : total_caps = 39 :=
by
  sorry

end NUMINAMATH_GPT_jose_total_caps_l1291_129124


namespace NUMINAMATH_GPT_misread_number_is_correct_l1291_129133

-- Definitions for the given conditions
def avg_incorrect : ℕ := 19
def incorrect_number : ℕ := 26
def avg_correct : ℕ := 24

-- Statement to prove the actual number that was misread
theorem misread_number_is_correct (x : ℕ) (h : 10 * avg_correct - 10 * avg_incorrect = x - incorrect_number) : x = 76 :=
by {
  sorry
}

end NUMINAMATH_GPT_misread_number_is_correct_l1291_129133


namespace NUMINAMATH_GPT_not_perfect_square_l1291_129147

theorem not_perfect_square (n : ℤ) : ¬ ∃ (m : ℤ), 4*n + 3 = m^2 := 
by 
  sorry

end NUMINAMATH_GPT_not_perfect_square_l1291_129147


namespace NUMINAMATH_GPT_fewer_hours_l1291_129153

noncomputable def distance : ℝ := 300
noncomputable def speed_T : ℝ := 20
noncomputable def speed_A : ℝ := speed_T + 5

theorem fewer_hours (d : ℝ) (V_T : ℝ) (V_A : ℝ) :
    V_T = 20 ∧ V_A = V_T + 5 ∧ d = 300 → (d / V_T) - (d / V_A) = 3 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_fewer_hours_l1291_129153


namespace NUMINAMATH_GPT_math_problem_l1291_129112

theorem math_problem
  (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ)
  (h1 : x₁ + 4 * x₂ + 9 * x₃ + 16 * x₄ + 25 * x₅ + 36 * x₆ + 49 * x₇ = 1)
  (h2 : 4 * x₁ + 9 * x₂ + 16 * x₃ + 25 * x₄ + 36 * x₅ + 49 * x₆ + 64 * x₇ = 12)
  (h3 : 9 * x₁ + 16 * x₂ + 25 * x₃ + 36 * x₄ + 49 * x₅ + 64 * x₆ + 81 * x₇ = 123) :
  16 * x₁ + 25 * x₂ + 36 * x₃ + 49 * x₄ + 64 * x₅ + 81 * x₆ + 100 * x₇ = 334 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1291_129112


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1291_129115

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : U \ A = {2, 4} := 
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1291_129115


namespace NUMINAMATH_GPT_exponential_monotone_l1291_129105

theorem exponential_monotone {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b :=
sorry

end NUMINAMATH_GPT_exponential_monotone_l1291_129105


namespace NUMINAMATH_GPT_earnings_per_weed_is_six_l1291_129117

def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def grass_weeds : ℕ := 32
def grass_weeds_half : ℕ := grass_weeds / 2
def soda_cost : ℕ := 99
def money_left : ℕ := 147
def total_weeds : ℕ := flower_bed_weeds + vegetable_patch_weeds + grass_weeds_half
def total_money : ℕ := money_left + soda_cost

theorem earnings_per_weed_is_six :
  total_money / total_weeds = 6 :=
by
  sorry

end NUMINAMATH_GPT_earnings_per_weed_is_six_l1291_129117


namespace NUMINAMATH_GPT_calculate_expression_l1291_129131

theorem calculate_expression : (3^5 * 4^5) / 6^5 = 32 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1291_129131


namespace NUMINAMATH_GPT_cos_x_plus_2y_is_one_l1291_129162

theorem cos_x_plus_2y_is_one
    (x y : ℝ) (a : ℝ) 
    (hx : x ∈ Set.Icc (-Real.pi) Real.pi)
    (hy : y ∈ Set.Icc (-Real.pi) Real.pi)
    (h_eq : 2 * a = x ^ 3 + Real.sin x ∧ 2 * a = (-2 * y) ^ 3 - Real.sin (-2 * y)) :
    Real.cos (x + 2 * y) = 1 := 
sorry

end NUMINAMATH_GPT_cos_x_plus_2y_is_one_l1291_129162


namespace NUMINAMATH_GPT_faulty_balance_inequality_l1291_129127

variable (m n a b G : ℝ)

theorem faulty_balance_inequality
  (h1 : m * a = n * G)
  (h2 : n * b = m * G) :
  (a + b) / 2 > G :=
sorry

end NUMINAMATH_GPT_faulty_balance_inequality_l1291_129127


namespace NUMINAMATH_GPT_cosA_value_area_of_triangle_l1291_129179

noncomputable def cosA (a b c : ℝ) (cos_C : ℝ) : ℝ :=
  if (a ≠ 0 ∧ cos_C ≠ 0) then (2 * b - c) * cos_C / a else 1 / 2

noncomputable def area_triangle (a b c : ℝ) (cosA_val : ℝ) : ℝ :=
  let S := a * b * (Real.sqrt (1 - cosA_val ^ 2)) / 2
  S

theorem cosA_value (a b c : ℝ) (cos_C : ℝ) : a * cos_C = (2 * b - c) * (cosA a b c cos_C) → cosA a b c cos_C = 1 / 2 :=
by
  sorry

theorem area_of_triangle (a b c : ℝ) (cos_A : ℝ) (cos_A_proof : a * cos_C = (2 * b - c) * cos_A) (h₀ : a = 6) (h₁ : b + c = 8) : area_triangle a b c cos_A = 7 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cosA_value_area_of_triangle_l1291_129179


namespace NUMINAMATH_GPT_number_of_buses_l1291_129136

theorem number_of_buses (x y : ℕ) (h1 : x + y = 40) (h2 : 6 * x + 4 * y = 210) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_buses_l1291_129136


namespace NUMINAMATH_GPT_ratio_a_d_l1291_129128

variables (a b c d : ℕ)

-- Given conditions
def ratio_ab := 8 / 3
def ratio_bc := 1 / 5
def ratio_cd := 3 / 2
def b_value := 27

theorem ratio_a_d (h₁ : a / b = ratio_ab)
                  (h₂ : b / c = ratio_bc)
                  (h₃ : c / d = ratio_cd)
                  (h₄ : b = b_value) :
  a / d = 4 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_a_d_l1291_129128


namespace NUMINAMATH_GPT_sarah_min_correct_l1291_129185

theorem sarah_min_correct (c : ℕ) (hc : c * 8 + 10 ≥ 110) : c ≥ 13 :=
sorry

end NUMINAMATH_GPT_sarah_min_correct_l1291_129185


namespace NUMINAMATH_GPT_union_M_N_eq_U_l1291_129119

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_M_N_eq_U : M ∪ N = U := 
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_union_M_N_eq_U_l1291_129119


namespace NUMINAMATH_GPT_quintuple_sum_not_less_than_l1291_129103

theorem quintuple_sum_not_less_than (a : ℝ) : 5 * (a + 3) ≥ 6 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_quintuple_sum_not_less_than_l1291_129103


namespace NUMINAMATH_GPT_no_prime_roots_l1291_129180

noncomputable def roots_are_prime (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q

theorem no_prime_roots : 
  ∀ k : ℕ, ¬ (∃ p q : ℕ, roots_are_prime p q ∧ p + q = 65 ∧ p * q = k) := 
sorry

end NUMINAMATH_GPT_no_prime_roots_l1291_129180


namespace NUMINAMATH_GPT_probability_of_blank_l1291_129137

-- Definitions based on conditions
def num_prizes : ℕ := 10
def num_blanks : ℕ := 25
def total_outcomes : ℕ := num_prizes + num_blanks

-- Statement of the proof problem
theorem probability_of_blank : (num_blanks / total_outcomes : ℚ) = 5 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_blank_l1291_129137


namespace NUMINAMATH_GPT_smallest_tree_height_l1291_129138

theorem smallest_tree_height (tallest middle smallest : ℝ)
  (h1 : tallest = 108)
  (h2 : middle = (tallest / 2) - 6)
  (h3 : smallest = middle / 4) : smallest = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_tree_height_l1291_129138


namespace NUMINAMATH_GPT_unattainable_y_ne_l1291_129195

theorem unattainable_y_ne : ∀ x : ℝ, x ≠ -5/4 → y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3/4 :=
by
  sorry

end NUMINAMATH_GPT_unattainable_y_ne_l1291_129195


namespace NUMINAMATH_GPT_terminal_side_in_fourth_quadrant_l1291_129132

theorem terminal_side_in_fourth_quadrant 
  (h_sin_half : Real.sin (α / 2) = 3 / 5)
  (h_cos_half : Real.cos (α / 2) = -4 / 5) : 
  (Real.sin α < 0) ∧ (Real.cos α > 0) :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_in_fourth_quadrant_l1291_129132


namespace NUMINAMATH_GPT_problem_number_eq_7_5_l1291_129146

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end NUMINAMATH_GPT_problem_number_eq_7_5_l1291_129146


namespace NUMINAMATH_GPT_find_real_number_a_l1291_129193

variable (U : Set ℕ) (M : Set ℕ) (a : ℕ)

theorem find_real_number_a :
  U = {1, 3, 5, 7} →
  M = {1, a} →
  (U \ M) = {5, 7} →
  a = 3 :=
by
  intros hU hM hCompU
  -- Proof part will be here
  sorry

end NUMINAMATH_GPT_find_real_number_a_l1291_129193


namespace NUMINAMATH_GPT_complex_computation_l1291_129102

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_GPT_complex_computation_l1291_129102


namespace NUMINAMATH_GPT_math_problem_l1291_129157

open Real

theorem math_problem (α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5) :
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (cos (3 * π / 2 + α) ^ 2 + 2 * cos α * cos (π / 2 - α)) / (1 + sin (π / 2 - α) ^ 2) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1291_129157


namespace NUMINAMATH_GPT_obtuse_triangle_l1291_129135

variable (A B C : ℝ)
variable (angle_sum : A + B + C = 180)
variable (cond1 : A + B = 141)
variable (cond2 : B + C = 165)

theorem obtuse_triangle : B > 90 :=
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_l1291_129135


namespace NUMINAMATH_GPT_least_value_of_p_plus_q_l1291_129113

theorem least_value_of_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 28 * (q + 1)) : p + q = 135 :=
  sorry

end NUMINAMATH_GPT_least_value_of_p_plus_q_l1291_129113


namespace NUMINAMATH_GPT_gcd_459_357_l1291_129191

/-- Prove that the greatest common divisor of 459 and 357 is 51. -/
theorem gcd_459_357 : gcd 459 357 = 51 :=
by
  sorry

end NUMINAMATH_GPT_gcd_459_357_l1291_129191


namespace NUMINAMATH_GPT_simplify_expression_l1291_129101

theorem simplify_expression (x y : ℝ) : ((3 * x + 22) + (150 * y + 22)) = (3 * x + 150 * y + 44) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1291_129101


namespace NUMINAMATH_GPT_janele_cats_average_weight_l1291_129140

noncomputable def average_weight_cats (w1 w2 w3 w4 : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) / 4

theorem janele_cats_average_weight :
  average_weight_cats 12 12 14.7 9.3 = 12 :=
by
  sorry

end NUMINAMATH_GPT_janele_cats_average_weight_l1291_129140


namespace NUMINAMATH_GPT_find_19a_20b_21c_l1291_129144

theorem find_19a_20b_21c (a b c : ℕ) (h₁ : 29 * a + 30 * b + 31 * c = 366) 
  (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 19 * a + 20 * b + 21 * c = 246 := 
sorry

end NUMINAMATH_GPT_find_19a_20b_21c_l1291_129144


namespace NUMINAMATH_GPT_directrix_of_parabola_l1291_129134

def parabola_directrix (x_y_eqn : ℝ → ℝ) : ℝ := by
  -- Assuming the parabola equation x = -(1/4) y^2
  sorry

theorem directrix_of_parabola : parabola_directrix (fun y => -(1/4) * y^2) = 1 := by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1291_129134


namespace NUMINAMATH_GPT_total_campers_went_rowing_l1291_129182

-- Definitions based on given conditions
def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def evening_campers : ℕ := 49

-- Theorem statement to be proven
theorem total_campers_went_rowing : morning_campers + afternoon_campers + evening_campers = 98 :=
by sorry

end NUMINAMATH_GPT_total_campers_went_rowing_l1291_129182


namespace NUMINAMATH_GPT_first_step_of_testing_circuit_broken_l1291_129106

-- Definitions based on the problem
def circuit_broken : Prop := true
def binary_search_method : Prop := true
def test_first_step_at_midpoint : Prop := true

-- The theorem stating the first step in testing a broken circuit using the binary search method
theorem first_step_of_testing_circuit_broken (h1 : circuit_broken) (h2 : binary_search_method) :
  test_first_step_at_midpoint :=
sorry

end NUMINAMATH_GPT_first_step_of_testing_circuit_broken_l1291_129106


namespace NUMINAMATH_GPT_find_initial_music_files_l1291_129197

-- Define the initial state before any deletion
def initial_files (music_files : ℕ) (video_files : ℕ) : ℕ := music_files + video_files

-- Define the state after deleting files
def files_after_deletion (initial_files : ℕ) (deleted_files : ℕ) : ℕ := initial_files - deleted_files

-- Theorem to prove that the initial number of music files was 13
theorem find_initial_music_files 
  (video_files : ℕ) (deleted_files : ℕ) (remaining_files : ℕ) 
  (h_videos : video_files = 30) (h_deleted : deleted_files = 10) (h_remaining : remaining_files = 33) : 
  ∃ (music_files : ℕ), initial_files music_files video_files - deleted_files = remaining_files ∧ music_files = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_initial_music_files_l1291_129197


namespace NUMINAMATH_GPT_solve_system_of_equations_l1291_129196

def system_of_equations(x y z: ℝ): Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

theorem solve_system_of_equations :
  ∀ (x y z: ℝ), system_of_equations x y z ↔
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2 ∨ z = -1) ∨
  (x = -3 ∧ y = 2 ∨ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1291_129196


namespace NUMINAMATH_GPT_kite_minimum_area_correct_l1291_129114

noncomputable def minimumKiteAreaAndSum (r : ℕ) (OP : ℕ) (h₁ : r = 60) (h₂ : OP < r) : ℕ × ℝ :=
  let d₁ := 2 * r
  let d₂ := 2 * Real.sqrt (r^2 - OP^2)
  let area := (d₁ * d₂) / 2
  (120 + 119, area)

theorem kite_minimum_area_correct {r OP : ℕ} (h₁ : r = 60) (h₂ : OP < r) :
  minimumKiteAreaAndSum r OP h₁ h₂ = (239, 120 * Real.sqrt 119) :=
by simp [minimumKiteAreaAndSum, h₁, h₂] ; sorry

end NUMINAMATH_GPT_kite_minimum_area_correct_l1291_129114


namespace NUMINAMATH_GPT_bells_ring_together_l1291_129158

open Nat

theorem bells_ring_together :
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  next_ring_time / total_minutes_in_an_hour = 6 :=
by
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  have h_next_ring_time : next_ring_time = 360 := by
    sorry
  have h_hours : next_ring_time / total_minutes_in_an_hour = 6 := by
    sorry
  exact h_hours

end NUMINAMATH_GPT_bells_ring_together_l1291_129158


namespace NUMINAMATH_GPT_florist_has_56_roses_l1291_129125

def initial_roses := 50
def roses_sold := 15
def roses_picked := 21

theorem florist_has_56_roses (r0 rs rp : ℕ) (h1 : r0 = initial_roses) (h2 : rs = roses_sold) (h3 : rp = roses_picked) : 
  r0 - rs + rp = 56 :=
by sorry

end NUMINAMATH_GPT_florist_has_56_roses_l1291_129125


namespace NUMINAMATH_GPT_correct_calculation_l1291_129169

theorem correct_calculation :
  - (1 / 2) - (- (1 / 3)) = - (1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1291_129169


namespace NUMINAMATH_GPT_honey_harvest_this_year_l1291_129188

def last_year_harvest : ℕ := 2479
def increase_this_year : ℕ := 6085

theorem honey_harvest_this_year : last_year_harvest + increase_this_year = 8564 :=
by {
  sorry
}

end NUMINAMATH_GPT_honey_harvest_this_year_l1291_129188


namespace NUMINAMATH_GPT_correct_calculation_l1291_129156

theorem correct_calculation (x : ℝ) : x * x^2 = x^3 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l1291_129156


namespace NUMINAMATH_GPT_sum_of_numbers_l1291_129159

theorem sum_of_numbers :
  145 + 35 + 25 + 5 = 210 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1291_129159


namespace NUMINAMATH_GPT_lily_cups_in_order_l1291_129178

theorem lily_cups_in_order :
  ∀ (rose_rate lily_rate : ℕ) (order_rose_cups total_payment hourly_wage : ℕ),
    rose_rate = 6 →
    lily_rate = 7 →
    order_rose_cups = 6 →
    total_payment = 90 →
    hourly_wage = 30 →
    ∃ lily_cups: ℕ, lily_cups = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lily_cups_in_order_l1291_129178


namespace NUMINAMATH_GPT_mooncake_inspection_random_event_l1291_129164

-- Definition of event categories
inductive Event
| certain
| impossible
| random

-- Definition of the event in question
def mooncakeInspectionEvent (satisfactory: Bool) : Event :=
if satisfactory then Event.random else Event.random

-- Theorem statement to prove that the event is a random event
theorem mooncake_inspection_random_event (satisfactory: Bool) :
  mooncakeInspectionEvent satisfactory = Event.random :=
sorry

end NUMINAMATH_GPT_mooncake_inspection_random_event_l1291_129164


namespace NUMINAMATH_GPT_enemies_left_undefeated_l1291_129110

theorem enemies_left_undefeated (points_per_enemy points_earned total_enemies : ℕ) 
  (h1 : points_per_enemy = 3)
  (h2 : total_enemies = 6)
  (h3 : points_earned = 12) : 
  (total_enemies - points_earned / points_per_enemy) = 2 :=
by
  sorry

end NUMINAMATH_GPT_enemies_left_undefeated_l1291_129110


namespace NUMINAMATH_GPT_greatest_divisor_of_28_l1291_129161

theorem greatest_divisor_of_28 : ∀ d : ℕ, d ∣ 28 → d ≤ 28 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_28_l1291_129161


namespace NUMINAMATH_GPT_problem1_problem2_l1291_129184

-- Definitions
def vec_a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)
def sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Problem 1: Prove the value of m such that vec_a ⊥ (vec_a - b(m))
theorem problem1 (m : ℝ) (h_perp: dot vec_a (sub vec_a (b m)) = 0) : m = -4 := sorry

-- Problem 2: Prove the value of k such that k * vec_a + b(-4) is parallel to vec_a - b(-4)
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def parallel (u v : ℝ × ℝ) := ∃ (k : ℝ), scale k u = v

theorem problem2 (k : ℝ) (h_parallel: parallel (add (scale k vec_a) (b (-4))) (sub vec_a (b (-4)))) : k = -1 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1291_129184


namespace NUMINAMATH_GPT_no_prime_roots_of_quadratic_l1291_129165

open Int Nat

theorem no_prime_roots_of_quadratic (k : ℤ) :
  ¬ (∃ p q : ℤ, Prime p ∧ Prime q ∧ p + q = 107 ∧ p * q = k) :=
by
  sorry

end NUMINAMATH_GPT_no_prime_roots_of_quadratic_l1291_129165


namespace NUMINAMATH_GPT_gcd_example_l1291_129143

theorem gcd_example : Nat.gcd 8675309 7654321 = 36 := sorry

end NUMINAMATH_GPT_gcd_example_l1291_129143


namespace NUMINAMATH_GPT_min_value_of_expr_min_value_achieved_final_statement_l1291_129152

theorem min_value_of_expr (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  1 ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem min_value_achieved (x y z : ℝ) (h1 : x = 1) (h2 : y = 1) (h3 : z = 1) :
  1 = (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem final_statement (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  ∃ (x y z : ℝ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x + y + z = 3) ∧ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_min_value_achieved_final_statement_l1291_129152


namespace NUMINAMATH_GPT_prime_product_solution_l1291_129174

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_product_solution (p_1 p_2 p_3 p_4 : ℕ) :
  is_prime p_1 ∧ is_prime p_2 ∧ is_prime p_3 ∧ is_prime p_4 ∧ 
  p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_1 ≠ p_4 ∧ p_2 ≠ p_3 ∧ p_2 ≠ p_4 ∧ p_3 ≠ p_4 ∧
  2 * p_1 + 3 * p_2 + 5 * p_3 + 7 * p_4 = 162 ∧
  11 * p_1 + 7 * p_2 + 5 * p_3 + 4 * p_4 = 162 
  → p_1 * p_2 * p_3 * p_4 = 570 := 
by
  sorry

end NUMINAMATH_GPT_prime_product_solution_l1291_129174


namespace NUMINAMATH_GPT_ab_multiple_of_7_2010_l1291_129167

theorem ab_multiple_of_7_2010 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 7 ^ 2009 ∣ a^2 + b^2) : 7 ^ 2010 ∣ a * b :=
by
  sorry

end NUMINAMATH_GPT_ab_multiple_of_7_2010_l1291_129167


namespace NUMINAMATH_GPT_contradiction_example_l1291_129123

theorem contradiction_example (a b c : ℕ) : (¬ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) → (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_example_l1291_129123


namespace NUMINAMATH_GPT_intersecting_lines_l1291_129149

theorem intersecting_lines (a b : ℚ) :
  (3 = (1 / 3 : ℚ) * 4 + a) → 
  (4 = (1 / 2 : ℚ) * 3 + b) → 
  a + b = 25 / 6 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_intersecting_lines_l1291_129149


namespace NUMINAMATH_GPT_find_f_m_eq_neg_one_l1291_129176

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^(2 - m)

theorem find_f_m_eq_neg_one (m : ℝ)
  (h1 : ∀ x : ℝ, f x m = - f (-x) m) (h2 : m^2 - m = 3 + m) :
  f m m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_m_eq_neg_one_l1291_129176


namespace NUMINAMATH_GPT_blocks_to_store_l1291_129154

theorem blocks_to_store
  (T : ℕ) (S : ℕ)
  (hT : T = 25)
  (h_total_walk : S + 6 + 8 = T) :
  S = 11 :=
by
  sorry

end NUMINAMATH_GPT_blocks_to_store_l1291_129154


namespace NUMINAMATH_GPT_daily_reading_goal_l1291_129171

-- Define the constants for pages read each day
def pages_on_sunday : ℕ := 43
def pages_on_monday : ℕ := 65
def pages_on_tuesday : ℕ := 28
def pages_on_wednesday : ℕ := 0
def pages_on_thursday : ℕ := 70
def pages_on_friday : ℕ := 56
def pages_on_saturday : ℕ := 88

-- Define the total pages read in the week
def total_pages := pages_on_sunday + pages_on_monday + pages_on_tuesday + pages_on_wednesday 
                    + pages_on_thursday + pages_on_friday + pages_on_saturday

-- The theorem that expresses Berry's daily reading goal
theorem daily_reading_goal : total_pages / 7 = 50 :=
by
  sorry

end NUMINAMATH_GPT_daily_reading_goal_l1291_129171


namespace NUMINAMATH_GPT_given_sequence_find_a_and_b_l1291_129198

-- Define the general pattern of the sequence
def sequence_pattern (n a b : ℕ) : Prop :=
  n + (b / a : ℚ) = (n^2 : ℚ) * (b / a : ℚ)

-- State the specific case for n = 9
def sequence_case_for_9 (a b : ℕ) : Prop :=
  sequence_pattern 9 a b ∧ a + b = 89

-- Now, structure this as a theorem to be proven in Lean
theorem given_sequence_find_a_and_b :
  ∃ (a b : ℕ), sequence_case_for_9 a b :=
sorry

end NUMINAMATH_GPT_given_sequence_find_a_and_b_l1291_129198


namespace NUMINAMATH_GPT_sampling_method_is_systematic_l1291_129107

-- Definition of the conditions
def factory_produces_product := True  -- Assuming the factory is always producing
def uses_conveyor_belt := True  -- Assuming the conveyor belt is always in use
def samples_taken_every_10_minutes := True  -- Sampling at specific intervals

-- Definition corresponding to the systematic sampling
def systematic_sampling := True

-- Theorem: Prove that given the conditions, the sampling method is systematic sampling.
theorem sampling_method_is_systematic :
  factory_produces_product → uses_conveyor_belt → samples_taken_every_10_minutes → systematic_sampling :=
by
  intros _ _ _
  trivial

end NUMINAMATH_GPT_sampling_method_is_systematic_l1291_129107


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_nine_l1291_129139

variable {a : ℕ → ℤ} -- Define a_n sequence as a function from ℕ to ℤ

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n m, a (n + m) = a n + m * d

def fifth_term_is_two (a : ℕ → ℤ) : Prop :=
  a 5 = 2

-- Lean statement to prove the sum of the first 9 terms
theorem arithmetic_sequence_sum_nine (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : fifth_term_is_two a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_nine_l1291_129139


namespace NUMINAMATH_GPT_wrapping_third_roll_l1291_129120

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end NUMINAMATH_GPT_wrapping_third_roll_l1291_129120


namespace NUMINAMATH_GPT_prime_square_condition_no_prime_cube_condition_l1291_129122

-- Part (a): Prove p = 3 given 8*p + 1 = n^2 and p is a prime
theorem prime_square_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 2) : 
  p = 3 :=
sorry

-- Part (b): Prove no p exists given 8*p + 1 = n^3 and p is a prime
theorem no_prime_cube_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 3) : 
  False :=
sorry

end NUMINAMATH_GPT_prime_square_condition_no_prime_cube_condition_l1291_129122


namespace NUMINAMATH_GPT_smallest_divisor_l1291_129177

theorem smallest_divisor (n : ℕ) (h1 : n = 999) :
  ∃ d : ℕ, 2.45 ≤ (999 : ℝ) / d ∧ (999 : ℝ) / d < 2.55 ∧ d = 392 :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisor_l1291_129177


namespace NUMINAMATH_GPT_total_distance_trip_l1291_129187

-- Defining conditions
def time_paved := 2 -- hours
def time_dirt := 3 -- hours
def speed_dirt := 32 -- mph
def speed_paved := speed_dirt + 20 -- mph

-- Defining distances
def distance_dirt := speed_dirt * time_dirt -- miles
def distance_paved := speed_paved * time_paved -- miles

-- Proving total distance
theorem total_distance_trip : distance_dirt + distance_paved = 200 := by
  sorry

end NUMINAMATH_GPT_total_distance_trip_l1291_129187
