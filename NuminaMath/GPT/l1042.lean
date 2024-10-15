import Mathlib

namespace NUMINAMATH_GPT_justin_reads_pages_l1042_104263

theorem justin_reads_pages (x : ℕ) 
  (h1 : 130 = x + 6 * (2 * x)) : x = 10 := 
sorry

end NUMINAMATH_GPT_justin_reads_pages_l1042_104263


namespace NUMINAMATH_GPT_sum_of_solutions_of_quadratic_l1042_104298

theorem sum_of_solutions_of_quadratic :
  ∀ a b c x₁ x₂ : ℝ, a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (∃ s : ℝ, s = x₁ + x₂ ∧ -b / a = s) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_of_quadratic_l1042_104298


namespace NUMINAMATH_GPT_quadrilateral_front_view_iff_cylinder_or_prism_l1042_104248

inductive Solid
| cone : Solid
| cylinder : Solid
| triangular_pyramid : Solid
| quadrangular_prism : Solid

def has_quadrilateral_front_view (s : Solid) : Prop :=
  s = Solid.cylinder ∨ s = Solid.quadrangular_prism

theorem quadrilateral_front_view_iff_cylinder_or_prism (s : Solid) :
  has_quadrilateral_front_view s ↔ s = Solid.cylinder ∨ s = Solid.quadrangular_prism :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_front_view_iff_cylinder_or_prism_l1042_104248


namespace NUMINAMATH_GPT_B_catches_up_with_A_l1042_104226

-- Define the conditions
def speed_A : ℝ := 10 -- A's speed in kmph
def speed_B : ℝ := 20 -- B's speed in kmph
def delay : ℝ := 6 -- Delay in hours after A's start

-- Define the total distance where B catches up with A
def distance_catch_up : ℝ := 120

-- Statement to prove B catches up with A at 120 km from the start
theorem B_catches_up_with_A :
  (speed_A * delay + speed_A * (distance_catch_up / speed_B - delay)) = distance_catch_up :=
by
  sorry

end NUMINAMATH_GPT_B_catches_up_with_A_l1042_104226


namespace NUMINAMATH_GPT_find_sets_l1042_104294

theorem find_sets (A B : Set ℕ) :
  A ∩ B = {1, 2, 3} ∧ A ∪ B = {1, 2, 3, 4, 5} →
    (A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
    (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
    (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
    (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4}) :=
by
  sorry

end NUMINAMATH_GPT_find_sets_l1042_104294


namespace NUMINAMATH_GPT_average_income_Q_and_R_l1042_104236

variable (P Q R: ℝ)

theorem average_income_Q_and_R:
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 :=
by
  sorry

end NUMINAMATH_GPT_average_income_Q_and_R_l1042_104236


namespace NUMINAMATH_GPT_find_tangent_line_l1042_104229

theorem find_tangent_line (k : ℝ) :
  (∃ k : ℝ, ∀ (x y : ℝ), y = k * (x - 1) + 3 ∧ k^2 + 1 = 1) →
  (∃ k : ℝ, k = 4 / 3 ∧ (k * x - y + 3 - k = 0) ∨ (x = 1)) :=
sorry

end NUMINAMATH_GPT_find_tangent_line_l1042_104229


namespace NUMINAMATH_GPT_fraction_value_l1042_104201

theorem fraction_value (a b : ℚ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_fraction_value_l1042_104201


namespace NUMINAMATH_GPT_find_speed_first_car_l1042_104287

noncomputable def speed_first_car (v : ℝ) : Prop :=
  let t := (14 : ℝ) / 3
  let d_total := 490
  let d_second_car := 60 * t
  let d_first_car := v * t
  d_second_car + d_first_car = d_total

theorem find_speed_first_car : ∃ v : ℝ, speed_first_car v ∧ v = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_first_car_l1042_104287


namespace NUMINAMATH_GPT_star_points_number_l1042_104233

-- Let n be the number of points in the star
def n : ℕ := sorry

-- Let A and B be the angles at the star points, with the condition that A_i = B_i - 20
def A (i : ℕ) : ℝ := sorry
def B (i : ℕ) : ℝ := sorry

-- Condition: For all i, A_i = B_i - 20
axiom angle_condition : ∀ i, A i = B i - 20

-- Total sum of angle differences equal to 360 degrees
axiom angle_sum_condition : n * 20 = 360

-- Theorem to prove
theorem star_points_number : n = 18 := by
  sorry

end NUMINAMATH_GPT_star_points_number_l1042_104233


namespace NUMINAMATH_GPT_sum_of_possible_x_values_l1042_104257

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end NUMINAMATH_GPT_sum_of_possible_x_values_l1042_104257


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1042_104297

theorem simplify_and_evaluate_expression (x : ℝ) (h : x^2 - 2 * x - 2 = 0) :
    ( ( (x - 1)/x - (x - 2)/(x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) = 1 / 2 ) :=
by
    -- sorry to skip the proof
    sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1042_104297


namespace NUMINAMATH_GPT_suff_but_not_nec_l1042_104280

theorem suff_but_not_nec (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_suff_but_not_nec_l1042_104280


namespace NUMINAMATH_GPT_joe_marshmallow_ratio_l1042_104289

theorem joe_marshmallow_ratio (J : ℕ) (h1 : 21 / 3 = 7) (h2 : 1 / 2 * J = 49 - 7) : J / 21 = 4 :=
by
  sorry

end NUMINAMATH_GPT_joe_marshmallow_ratio_l1042_104289


namespace NUMINAMATH_GPT_part_a_l1042_104290

theorem part_a : (2^41 + 1) % 83 = 0 :=
  sorry

end NUMINAMATH_GPT_part_a_l1042_104290


namespace NUMINAMATH_GPT_minimal_dominoes_needed_l1042_104243

-- Variables representing the number of dominoes and tetraminoes
variables (d t : ℕ)

-- Definitions related to the problem
def area_rectangle : ℕ := 2008 * 2010 -- Total area of the rectangle
def area_domino : ℕ := 1 * 2 -- Area of a single domino
def area_tetramino : ℕ := 2 * 3 - 2 -- Area of a single tetramino
def total_area_covered : ℕ := 2 * d + 4 * t -- Total area covered by dominoes and tetraminoes

-- The theorem we want to prove
theorem minimal_dominoes_needed :
  total_area_covered d t = area_rectangle → d = 0 :=
sorry

end NUMINAMATH_GPT_minimal_dominoes_needed_l1042_104243


namespace NUMINAMATH_GPT_find_amount_l1042_104228

-- Let A be the certain amount.
variable (A x : ℝ)

-- Given conditions
def condition1 (x : ℝ) := 0.65 * x = 0.20 * A
def condition2 (x : ℝ) := x = 150

-- Goal
theorem find_amount (A x : ℝ) (h1 : condition1 A x) (h2 : condition2 x) : A = 487.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_amount_l1042_104228


namespace NUMINAMATH_GPT_solve_for_y_l1042_104255

theorem solve_for_y (y : ℝ) (h : y + 49 / y = 14) : y = 7 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1042_104255


namespace NUMINAMATH_GPT_tan_alpha_ratio_expression_l1042_104276

variable (α : Real)
variable (h1 : Real.sin α = 3/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : Real.tan α = -3/4 := sorry

theorem ratio_expression {α : Real}
  (h1 : Real.sin α = 3/5)
  (h2 : π/2 < α ∧ α < π)
  : (2 * Real.sin α + 3 * Real.cos α) / (Real.cos α - Real.sin α) = 6/7 := sorry

end NUMINAMATH_GPT_tan_alpha_ratio_expression_l1042_104276


namespace NUMINAMATH_GPT_find_c_l1042_104286

theorem find_c (a b c : ℝ) (k₁ k₂ : ℝ) 
  (h₁ : a * b = k₁) 
  (h₂ : b * c = k₂) 
  (h₃ : 40 * 5 = k₁) 
  (h₄ : 7 * 10 = k₂) 
  (h₅ : a = 16) : 
  c = 5.6 :=
  sorry

end NUMINAMATH_GPT_find_c_l1042_104286


namespace NUMINAMATH_GPT_find_common_ratio_l1042_104244

theorem find_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h₀ : ∀ n, a (n + 1) = q * a n)
  (h₁ : a 0 = 4)
  (h₂ : q ≠ 1)
  (h₃ : 2 * a 4 = 4 * a 0 - 2 * a 2) :
  q = -1 := 
sorry

end NUMINAMATH_GPT_find_common_ratio_l1042_104244


namespace NUMINAMATH_GPT_max_dot_product_OB_OA_l1042_104242

theorem max_dot_product_OB_OA (P A O B : ℝ × ℝ)
  (h₁ : ∃ x y : ℝ, (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1)
  (t : ℝ)
  (h₂ : A = (t - 1) • P)
  (h₃ : P • O = 64)
  (h₄ : B = (0, 1)) :
  ∃ t : ℝ, abs (B • A) ≤ (24/5) := 
sorry

end NUMINAMATH_GPT_max_dot_product_OB_OA_l1042_104242


namespace NUMINAMATH_GPT_answered_both_questions_correctly_l1042_104292

theorem answered_both_questions_correctly (P_A P_B P_A_prime_inter_B_prime : ℝ)
  (h1 : P_A = 70 / 100) (h2 : P_B = 55 / 100) (h3 : P_A_prime_inter_B_prime = 20 / 100) :
  P_A + P_B - (1 - P_A_prime_inter_B_prime) = 45 / 100 := 
by
  sorry

end NUMINAMATH_GPT_answered_both_questions_correctly_l1042_104292


namespace NUMINAMATH_GPT_y_intercept_of_parallel_line_l1042_104283

theorem y_intercept_of_parallel_line (m x1 y1 : ℝ) (h_slope : m = -3) (h_point : (x1, y1) = (3, -1))
  (b : ℝ) (h_line_parallel : ∀ x, b = y1 + m * (x - x1)) :
  b = 8 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_parallel_line_l1042_104283


namespace NUMINAMATH_GPT_initial_balance_l1042_104291

theorem initial_balance (B : ℝ) (payment : ℝ) (new_balance : ℝ)
  (h1 : payment = 50) (h2 : new_balance = 120) (h3 : B - payment = new_balance) :
  B = 170 :=
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_initial_balance_l1042_104291


namespace NUMINAMATH_GPT_percentage_of_16_l1042_104253

theorem percentage_of_16 (p : ℝ) (h : (p / 100) * 16 = 0.04) : p = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_16_l1042_104253


namespace NUMINAMATH_GPT_owl_cost_in_gold_l1042_104215

-- Definitions for conditions
def spellbook_cost_gold := 5
def potionkit_cost_silver := 20
def num_spellbooks := 5
def num_potionkits := 3
def silver_per_gold := 9
def total_payment_silver := 537

-- Function to convert gold to silver
def gold_to_silver (gold : ℕ) : ℕ := gold * silver_per_gold

-- Function to compute total cost in silver for spellbooks and potion kits
def total_spellbook_cost_silver : ℕ :=
  gold_to_silver spellbook_cost_gold * num_spellbooks

def total_potionkit_cost_silver : ℕ :=
  potionkit_cost_silver * num_potionkits

-- Function to calculate the cost of the owl in silver
def owl_cost_silver : ℕ :=
  total_payment_silver - (total_spellbook_cost_silver + total_potionkit_cost_silver)

-- Function to convert the owl's cost from silver to gold
def owl_cost_gold : ℕ :=
  owl_cost_silver / silver_per_gold

-- The proof statement
theorem owl_cost_in_gold : owl_cost_gold = 28 :=
  by
    sorry

end NUMINAMATH_GPT_owl_cost_in_gold_l1042_104215


namespace NUMINAMATH_GPT_units_digit_square_l1042_104295

theorem units_digit_square (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) (h2 : (n % 10 = 2) ∨ (n % 10 = 7)) :
  ∀ (d : ℕ), (d = 2 ∨ d = 6 ∨ d = 3) → (n^2 % 10 ≠ d) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_square_l1042_104295


namespace NUMINAMATH_GPT_exists_pairwise_coprime_product_of_two_consecutive_integers_l1042_104220

theorem exists_pairwise_coprime_product_of_two_consecutive_integers (n : ℕ) (h : 0 < n) :
  ∃ (a : Fin n → ℕ), (∀ i, 2 ≤ a i) ∧ (Pairwise (IsCoprime on fun i => a i)) ∧ (∃ k : ℕ, (Finset.univ.prod a) - 1 = k * (k + 1)) := 
sorry

end NUMINAMATH_GPT_exists_pairwise_coprime_product_of_two_consecutive_integers_l1042_104220


namespace NUMINAMATH_GPT_weight_of_11th_person_l1042_104268

theorem weight_of_11th_person
  (n : ℕ) (avg1 avg2 : ℝ)
  (hn : n = 10)
  (havg1 : avg1 = 165)
  (havg2 : avg2 = 170)
  (W : ℝ) (X : ℝ)
  (hw : W = n * avg1)
  (havg2_eq : (W + X) / (n + 1) = avg2) :
  X = 220 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_11th_person_l1042_104268


namespace NUMINAMATH_GPT_inradius_of_triangle_l1042_104245

variable (A : ℝ) (p : ℝ) (r : ℝ) (s : ℝ)

theorem inradius_of_triangle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l1042_104245


namespace NUMINAMATH_GPT_no_such_triples_l1042_104260

noncomputable def no_triple_satisfy (a b c : ℤ) : Prop :=
  ∀ (x1 x2 x3 : ℤ), 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    Int.gcd x1 x2 = 1 ∧ Int.gcd x2 x3 = 1 ∧ Int.gcd x1 x3 = 1 ∧
    (x1^3 - a^2 * x1^2 + b^2 * x1 - a * b + 3 * c = 0) ∧ 
    (x2^3 - a^2 * x2^2 + b^2 * x2 - a * b + 3 * c = 0) ∧ 
    (x3^3 - a^2 * x3^2 + b^2 * x3 - a * b + 3 * c = 0) →
    False

theorem no_such_triples : ∀ (a b c : ℤ), no_triple_satisfy a b c :=
by
  intros
  sorry

end NUMINAMATH_GPT_no_such_triples_l1042_104260


namespace NUMINAMATH_GPT_problem_l1042_104249

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem problem 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y = 30) 
  (h5 : x * z = 60) 
  (h6 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := 
  sorry

end NUMINAMATH_GPT_problem_l1042_104249


namespace NUMINAMATH_GPT_conference_handshakes_l1042_104270

theorem conference_handshakes (n_leaders n_participants : ℕ) (n_total : ℕ) 
  (h_total : n_total = n_leaders + n_participants) 
  (h_leaders : n_leaders = 5) 
  (h_participants : n_participants = 25) 
  (h_total_people : n_total = 30) : 
  (n_leaders * (n_total - 1) - (n_leaders * (n_leaders - 1) / 2)) = 135 := 
by 
  sorry

end NUMINAMATH_GPT_conference_handshakes_l1042_104270


namespace NUMINAMATH_GPT_sodas_to_take_back_l1042_104277

def num_sodas_brought : ℕ := 50
def num_sodas_drank : ℕ := 38

theorem sodas_to_take_back : (num_sodas_brought - num_sodas_drank) = 12 := by
  sorry

end NUMINAMATH_GPT_sodas_to_take_back_l1042_104277


namespace NUMINAMATH_GPT_prove_a2_minus_b2_l1042_104240

theorem prove_a2_minus_b2 : 
  ∀ (a b : ℚ), 
  a + b = 9 / 17 ∧ a - b = 1 / 51 → a^2 - b^2 = 3 / 289 :=
by
  intros a b h
  cases' h
  sorry

end NUMINAMATH_GPT_prove_a2_minus_b2_l1042_104240


namespace NUMINAMATH_GPT_solve_for_x_l1042_104266

theorem solve_for_x (a r s x : ℝ) (h1 : s > r) (h2 : r * (x + a) = s * (x - a)) :
  x = a * (s + r) / (s - r) :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1042_104266


namespace NUMINAMATH_GPT_product_of_two_numbers_l1042_104235
noncomputable def find_product (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : ℝ :=
x * y

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : find_product x y h1 h2 = 200 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1042_104235


namespace NUMINAMATH_GPT_cars_meet_time_l1042_104278

theorem cars_meet_time (s1 s2 : ℝ) (d : ℝ) (c : s1 = (5 / 4) * s2) 
  (h1 : s1 = 100) (h2 : d = 720) : d / (s1 + s2) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_cars_meet_time_l1042_104278


namespace NUMINAMATH_GPT_sugar_needed_l1042_104282

variable (a b c d : ℝ)
variable (H1 : a = 2)
variable (H2 : b = 1)
variable (H3 : d = 5)

theorem sugar_needed (c : ℝ) : c = 2.5 :=
by
  have H : 2 / 1 = 5 / c := by {
    sorry
  }
  sorry

end NUMINAMATH_GPT_sugar_needed_l1042_104282


namespace NUMINAMATH_GPT_cos_double_angle_l1042_104231

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi - θ) = 1 / 3) : 
  Real.cos (2 * θ) = 7 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1042_104231


namespace NUMINAMATH_GPT_prime_eq_solution_l1042_104296

theorem prime_eq_solution (a b : ℕ) (h1 : Nat.Prime a) (h2 : b > 0)
  (h3 : 9 * (2 * a + b) ^ 2 = 509 * (4 * a + 511 * b)) : 
  (a = 251 ∧ b = 7) :=
sorry

end NUMINAMATH_GPT_prime_eq_solution_l1042_104296


namespace NUMINAMATH_GPT_trigonometric_simplification_l1042_104209

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.cos α ^ 2 - 1) /
  (2 * Real.tan (π / 4 - α) * Real.sin (π / 4 + α) ^ 2) = 1 :=
sorry

end NUMINAMATH_GPT_trigonometric_simplification_l1042_104209


namespace NUMINAMATH_GPT_xy_sum_values_l1042_104225

theorem xy_sum_values (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end NUMINAMATH_GPT_xy_sum_values_l1042_104225


namespace NUMINAMATH_GPT_hockey_games_in_season_l1042_104262

theorem hockey_games_in_season
  (games_per_month : ℤ)
  (months_in_season : ℤ)
  (h1 : games_per_month = 25)
  (h2 : months_in_season = 18) :
  games_per_month * months_in_season = 450 :=
by
  sorry

end NUMINAMATH_GPT_hockey_games_in_season_l1042_104262


namespace NUMINAMATH_GPT_difference_between_Annette_and_Sara_l1042_104224

-- Define the weights of the individuals
variables (A C S B E : ℝ)

-- Conditions given in the problem
def condition1 := A + C = 95
def condition2 := C + S = 87
def condition3 := A + S = 97
def condition4 := C + B = 100
def condition5 := A + C + B = 155
def condition6 := A + S + B + E = 240
def condition7 := E = 1.25 * C

-- The theorem that we want to prove
theorem difference_between_Annette_and_Sara (A C S B E : ℝ)
  (h1 : condition1 A C)
  (h2 : condition2 C S)
  (h3 : condition3 A S)
  (h4 : condition4 C B)
  (h5 : condition5 A C B)
  (h6 : condition6 A S B E)
  (h7 : condition7 C E) :
  A - S = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_between_Annette_and_Sara_l1042_104224


namespace NUMINAMATH_GPT_calculate_exponent_product_l1042_104200

theorem calculate_exponent_product : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end NUMINAMATH_GPT_calculate_exponent_product_l1042_104200


namespace NUMINAMATH_GPT_al_bill_cal_probability_l1042_104293

-- Let's define the conditions and problem setup
def al_bill_cal_prob : ℚ :=
  let total_ways := 12 * 11 * 10
  let valid_ways := 12 -- This represent the summed valid cases as calculated
  valid_ways / total_ways

theorem al_bill_cal_probability :
  al_bill_cal_prob = 1 / 110 :=
  by
  -- Placeholder for calculation and proof
  sorry

end NUMINAMATH_GPT_al_bill_cal_probability_l1042_104293


namespace NUMINAMATH_GPT_terminating_decimal_expansion_l1042_104238

theorem terminating_decimal_expansion : (15 / 625 : ℝ) = 0.024 :=
by
  -- Lean requires a justification for non-trivial facts
  -- Provide math reasoning here if necessary
  sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_l1042_104238


namespace NUMINAMATH_GPT_add_fractions_l1042_104241

theorem add_fractions : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by sorry

end NUMINAMATH_GPT_add_fractions_l1042_104241


namespace NUMINAMATH_GPT_find_initial_solution_liters_l1042_104221

-- Define the conditions
def percentage_initial_solution_alcohol := 0.26
def added_water := 5
def percentage_new_mixture_alcohol := 0.195

-- Define the initial amount of the solution
def initial_solution_liters (x : ℝ) : Prop :=
  0.26 * x = 0.195 * (x + 5)

-- State the proof problem
theorem find_initial_solution_liters : initial_solution_liters 15 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_solution_liters_l1042_104221


namespace NUMINAMATH_GPT_total_wet_surface_area_correct_l1042_104206

namespace Cistern

-- Define the dimensions of the cistern and the depth of the water
def length : ℝ := 10
def width : ℝ := 8
def depth : ℝ := 1.5

-- Calculate the individual surface areas
def bottom_surface_area : ℝ := length * width
def longer_side_surface_area : ℝ := length * depth * 2
def shorter_side_surface_area : ℝ := width * depth * 2

-- The total wet surface area is the sum of all individual wet surface areas
def total_wet_surface_area : ℝ := 
  bottom_surface_area + longer_side_surface_area + shorter_side_surface_area

-- Prove that the total wet surface area is 134 m^2
theorem total_wet_surface_area_correct : 
  total_wet_surface_area = 134 := 
by sorry

end Cistern

end NUMINAMATH_GPT_total_wet_surface_area_correct_l1042_104206


namespace NUMINAMATH_GPT_female_democrats_count_l1042_104264

theorem female_democrats_count 
  (F M D : ℕ)
  (total_participants : F + M = 660)
  (total_democrats : F / 2 + M / 4 = 660 / 3)
  (female_democrats : D = F / 2) : 
  D = 110 := 
by
  sorry

end NUMINAMATH_GPT_female_democrats_count_l1042_104264


namespace NUMINAMATH_GPT_rectangle_square_division_l1042_104272

theorem rectangle_square_division (a b : ℝ) (n : ℕ) (h1 : (∃ (s1 : ℝ), s1^2 * (n : ℝ) = a * b))
                                            (h2 : (∃ (s2 : ℝ), s2^2 * (n + 76 : ℝ) = a * b)) :
    n = 324 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_square_division_l1042_104272


namespace NUMINAMATH_GPT_area_triangle_DEF_l1042_104227

noncomputable def triangleDEF (DE EF DF : ℝ) (angleDEF : ℝ) : ℝ :=
  if angleDEF = 60 ∧ DF = 3 ∧ EF = 6 / Real.sqrt 3 then
    1 / 2 * DE * EF * Real.sin (Real.pi / 3)
  else
    0

theorem area_triangle_DEF :
  triangleDEF (Real.sqrt 3) (6 / Real.sqrt 3) 3 60 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_DEF_l1042_104227


namespace NUMINAMATH_GPT_vacation_cost_split_l1042_104251

theorem vacation_cost_split 
  (john_paid mary_paid lisa_paid : ℕ) 
  (total_amount : ℕ) 
  (share : ℕ)
  (j m : ℤ)
  (h1 : john_paid = 150)
  (h2 : mary_paid = 90)
  (h3 : lisa_paid = 210)
  (h4 : total_amount = 450)
  (h5 : share = total_amount / 3) 
  (h6 : john_paid - share = j) 
  (h7 : mary_paid - share = m) 
  : j - m = -60 :=
by
  sorry

end NUMINAMATH_GPT_vacation_cost_split_l1042_104251


namespace NUMINAMATH_GPT_solve_system_of_equations_l1042_104252

theorem solve_system_of_equations (x y : ℝ) : 
  (x + y = x^2 + 2 * x * y + y^2) ∧ (x - y = x^2 - 2 * x * y + y^2) ↔ 
  (x = 0 ∧ y = 0) ∨ 
  (x = 1/2 ∧ y = 1/2) ∨ 
  (x = 1/2 ∧ y = -1/2) ∨ 
  (x = 1 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1042_104252


namespace NUMINAMATH_GPT_octahedron_tetrahedron_volume_ratio_l1042_104222

theorem octahedron_tetrahedron_volume_ratio (s : ℝ) :
  let V_T := (s^3 * Real.sqrt 2) / 12
  let a := s / 2
  let V_O := (a^3 * Real.sqrt 2) / 3
  V_O / V_T = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_octahedron_tetrahedron_volume_ratio_l1042_104222


namespace NUMINAMATH_GPT_find_weight_of_first_new_player_l1042_104219

variable (weight_of_first_new_player : ℕ)
variable (weight_of_second_new_player : ℕ := 60) -- Second new player's weight is a given constant
variable (num_of_original_players : ℕ := 7)
variable (avg_weight_of_original_players : ℕ := 121)
variable (new_avg_weight : ℕ := 113)
variable (num_of_new_players : ℕ := 2)

def total_weight_of_original_players : ℕ := 
  num_of_original_players * avg_weight_of_original_players

def total_weight_of_new_players : ℕ :=
  num_of_new_players * new_avg_weight

def combined_weight_without_first_new_player : ℕ := 
  total_weight_of_original_players + weight_of_second_new_player

def weight_of_first_new_player_proven : Prop :=
  total_weight_of_new_players - combined_weight_without_first_new_player = weight_of_first_new_player

theorem find_weight_of_first_new_player : weight_of_first_new_player = 110 :=
by 
  sorry

end NUMINAMATH_GPT_find_weight_of_first_new_player_l1042_104219


namespace NUMINAMATH_GPT_choose_lines_intersect_l1042_104223

-- We need to define the proof problem
theorem choose_lines_intersect : 
  ∃ (lines : ℕ → ℝ × ℝ → ℝ), 
    (∀ i j, i < 100 ∧ j < 100 ∧ i ≠ j → (lines i = lines j) → ∃ (p : ℕ), p = 2022) :=
sorry

end NUMINAMATH_GPT_choose_lines_intersect_l1042_104223


namespace NUMINAMATH_GPT_correct_number_of_statements_l1042_104258

noncomputable def number_of_correct_statements := 1

def statement_1 : Prop := false -- Equal angles are not preserved
def statement_2 : Prop := false -- Equal lengths are not preserved
def statement_3 : Prop := false -- The longest segment feature is not preserved
def statement_4 : Prop := true  -- The midpoint feature is preserved

theorem correct_number_of_statements :
  (statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4) = true →
  number_of_correct_statements = 1 :=
by
  sorry

end NUMINAMATH_GPT_correct_number_of_statements_l1042_104258


namespace NUMINAMATH_GPT_correct_option_l1042_104285

theorem correct_option : ∀ (x y : ℝ), 10 * x * y - 10 * y * x = 0 :=
by 
  intros x y
  sorry

end NUMINAMATH_GPT_correct_option_l1042_104285


namespace NUMINAMATH_GPT_relationship_l1042_104261

-- Definitions for the points on the inverse proportion function
def on_inverse_proportion (x : ℝ) (y : ℝ) : Prop :=
  y = -6 / x

-- Given conditions
def A (y1 : ℝ) : Prop :=
  on_inverse_proportion (-3) y1

def B (y2 : ℝ) : Prop :=
  on_inverse_proportion (-1) y2

def C (y3 : ℝ) : Prop :=
  on_inverse_proportion (2) y3

-- The theorem that expresses the relationship
theorem relationship (y1 y2 y3 : ℝ) (hA : A y1) (hB : B y2) (hC : C y3) : y3 < y1 ∧ y1 < y2 :=
by
  -- skeleton of proof
  sorry

end NUMINAMATH_GPT_relationship_l1042_104261


namespace NUMINAMATH_GPT_octopus_legs_l1042_104281

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end NUMINAMATH_GPT_octopus_legs_l1042_104281


namespace NUMINAMATH_GPT_forty_percent_of_number_l1042_104259

variables {N : ℝ}

theorem forty_percent_of_number (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) : 0.40 * N = 120 :=
by sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1042_104259


namespace NUMINAMATH_GPT_even_product_divisible_by_1947_l1042_104256

theorem even_product_divisible_by_1947 (n : ℕ) (h_even : n % 2 = 0) :
  (∃ k: ℕ, 2 ≤ k ∧ k ≤ n / 2 ∧ 1947 ∣ (2 ^ k * k!)) → n ≥ 3894 :=
by
  sorry

end NUMINAMATH_GPT_even_product_divisible_by_1947_l1042_104256


namespace NUMINAMATH_GPT_cats_given_by_Mr_Sheridan_l1042_104247

-- Definitions of the initial state and final state
def initial_cats : Nat := 17
def total_cats : Nat := 31

-- Proof statement that Mr. Sheridan gave her 14 cats
theorem cats_given_by_Mr_Sheridan : total_cats - initial_cats = 14 := by
  sorry

end NUMINAMATH_GPT_cats_given_by_Mr_Sheridan_l1042_104247


namespace NUMINAMATH_GPT_rosa_initial_flowers_l1042_104218

-- Definitions derived from conditions
def initial_flowers (total_flowers : ℕ) (given_flowers : ℕ) : ℕ :=
  total_flowers - given_flowers

-- The theorem stating the proof problem
theorem rosa_initial_flowers : initial_flowers 90 23 = 67 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_rosa_initial_flowers_l1042_104218


namespace NUMINAMATH_GPT_ordered_pair_sol_l1042_104274

noncomputable def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![5, d]]

noncomputable def is_inverse_scalar_mul (d k : ℝ) : Prop :=
  (A d)⁻¹ = k • (A d)

theorem ordered_pair_sol (d k : ℝ) :
  is_inverse_scalar_mul d k → (d = -2 ∧ k = 1 / 19) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ordered_pair_sol_l1042_104274


namespace NUMINAMATH_GPT_least_subtracted_number_l1042_104279

theorem least_subtracted_number (r : ℕ) : r = 10^1000 % 97 := 
sorry

end NUMINAMATH_GPT_least_subtracted_number_l1042_104279


namespace NUMINAMATH_GPT_relationship_between_x_t_G_D_and_x_l1042_104237

-- Definitions
variables {G D : ℝ → ℝ}
variables {t : ℝ}
noncomputable def number_of_boys (x : ℝ) : ℝ := 9000 / x
noncomputable def total_population (x : ℝ) (x_t : ℝ) : Prop := x_t = 15000 / x

-- The proof problem
theorem relationship_between_x_t_G_D_and_x
  (G D : ℝ → ℝ)
  (x : ℝ) (t : ℝ) (x_t : ℝ)
  (h1 : 90 = x / 100 * number_of_boys x)
  (h2 : 0.60 * x_t = number_of_boys x)
  (h3 : 0.40 * x_t > 0)
  (h4 : true) :       -- Placeholder for some condition not used directly
  total_population x x_t :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_relationship_between_x_t_G_D_and_x_l1042_104237


namespace NUMINAMATH_GPT_rebus_solution_l1042_104214

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end NUMINAMATH_GPT_rebus_solution_l1042_104214


namespace NUMINAMATH_GPT_infinite_solutions_a_l1042_104299

theorem infinite_solutions_a (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_a_l1042_104299


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1042_104250

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1042_104250


namespace NUMINAMATH_GPT_average_people_added_each_year_l1042_104208

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end NUMINAMATH_GPT_average_people_added_each_year_l1042_104208


namespace NUMINAMATH_GPT_least_value_r_minus_p_l1042_104204

theorem least_value_r_minus_p (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 5) :
  ∃ r p, r = 5 ∧ p = 1/2 ∧ r - p = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_least_value_r_minus_p_l1042_104204


namespace NUMINAMATH_GPT_binary_to_decimal_and_octal_l1042_104207

theorem binary_to_decimal_and_octal (binary_input : Nat) (h : binary_input = 0b101101110) :
    binary_input == 366 ∧ (366 : Nat) == 0o66 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_and_octal_l1042_104207


namespace NUMINAMATH_GPT_post_height_l1042_104205

-- Conditions
def spiral_path (circuit_per_rise rise_distance : ℝ) := ∀ (total_distance circ_circumference height : ℝ), 
  circuit_per_rise = total_distance / circ_circumference ∧ 
  height = circuit_per_rise * rise_distance

-- Given conditions
def cylinder_post : Prop := 
  ∀ (total_distance circ_circumference rise_distance : ℝ), 
    spiral_path (total_distance / circ_circumference) rise_distance ∧ 
    circ_circumference = 3 ∧ 
    rise_distance = 4 ∧ 
    total_distance = 12

-- Proof problem: Post height
theorem post_height : cylinder_post → ∃ height : ℝ, height = 16 := 
by sorry

end NUMINAMATH_GPT_post_height_l1042_104205


namespace NUMINAMATH_GPT_selling_price_for_given_profit_selling_price_to_maximize_profit_l1042_104254

-- Define the parameters
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_decrement_per_unit_increase : ℝ := 10

-- Define the function for monthly sales based on price increment
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrement_per_unit_increase * x

-- Define the function for selling price based on price increment
def selling_price (x : ℝ) : ℝ := initial_selling_price + x

-- Define the function for monthly profit
def monthly_profit (x : ℝ) : ℝ :=
  let total_revenue := monthly_sales x * selling_price x 
  let total_cost := monthly_sales x * cost_price
  total_revenue - total_cost

-- Problem 1: Prove the selling price when monthly profit is 8750 yuan
theorem selling_price_for_given_profit : 
  ∃ x : ℝ, monthly_profit x = 8750 ∧ (selling_price x = 75 ∨ selling_price x = 65) :=
sorry

-- Problem 2: Prove the selling price that maximizes the monthly profit
theorem selling_price_to_maximize_profit : 
  ∀ x : ℝ, monthly_profit x ≤ monthly_profit 20 ∧ selling_price 20 = 70 :=
sorry

end NUMINAMATH_GPT_selling_price_for_given_profit_selling_price_to_maximize_profit_l1042_104254


namespace NUMINAMATH_GPT_number_of_boxes_l1042_104288

-- Define the given conditions
def total_chocolates : ℕ := 442
def chocolates_per_box : ℕ := 26

-- Prove the number of small boxes in the large box
theorem number_of_boxes : (total_chocolates / chocolates_per_box) = 17 := by
  sorry

end NUMINAMATH_GPT_number_of_boxes_l1042_104288


namespace NUMINAMATH_GPT_correct_factoring_example_l1042_104267

-- Define each option as hypotheses
def optionA (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2
def optionB (a b : ℝ) : Prop := 2 * a ^ 2 - a * b - a = a * (2 * a - b - 1)
def optionC (a b : ℝ) : Prop := 8 * a ^ 5 * b ^ 2 = 4 * a ^ 3 * b * 2 * a ^ 2 * b
def optionD (a : ℝ) : Prop := a ^ 2 - 4 * a + 3 = (a - 1) * (a - 3)

-- The goal is to prove that optionD is the correct example of factoring
theorem correct_factoring_example (a b : ℝ) : optionD a ↔ (∀ a b, ¬ optionA a b) ∧ (∀ a b, ¬ optionB a b) ∧ (∀ a b, ¬ optionC a b) :=
by
  sorry

end NUMINAMATH_GPT_correct_factoring_example_l1042_104267


namespace NUMINAMATH_GPT_teacher_drank_milk_false_l1042_104275

-- Define the condition that the volume of milk a teacher can reasonably drink in a day is more appropriately measured in milliliters rather than liters.
def reasonable_volume_units := "milliliters"

-- Define the statement to be judged
def teacher_milk_intake := 250

-- Define the unit of the statement
def unit_of_statement := "liters"

-- The proof goal is to conclude that the statement "The teacher drank 250 liters of milk today" is false, given the condition on volume units.
theorem teacher_drank_milk_false (vol : ℕ) (unit : String) (reasonable_units : String) :
  vol = 250 ∧ unit = "liters" ∧ reasonable_units = "milliliters" → false :=
by
  sorry

end NUMINAMATH_GPT_teacher_drank_milk_false_l1042_104275


namespace NUMINAMATH_GPT_blake_spent_on_apples_l1042_104273

noncomputable def apples_spending_problem : Prop :=
  let initial_amount := 300
  let change_received := 150
  let oranges_cost := 40
  let mangoes_cost := 60
  let total_spent := initial_amount - change_received
  let other_fruits_cost := oranges_cost + mangoes_cost
  let apples_cost := total_spent - other_fruits_cost
  apples_cost = 50

theorem blake_spent_on_apples : apples_spending_problem :=
by
  sorry

end NUMINAMATH_GPT_blake_spent_on_apples_l1042_104273


namespace NUMINAMATH_GPT_find_floor_l1042_104216

-- Define the total number of floors
def totalFloors : ℕ := 9

-- Define the total number of entrances
def totalEntrances : ℕ := 10

-- Each floor has the same number of apartments
-- The claim we are to prove is that for entrance 10 and apartment 333, Petya needs to go to the 3rd floor.

theorem find_floor (apartment_number : ℕ) (entrance_number : ℕ) (floor : ℕ)
  (h1 : entrance_number = 10)
  (h2 : apartment_number = 333)
  (h3 : ∀ (f : ℕ), 0 < f ∧ f ≤ totalFloors)
  (h4 : ∃ (n : ℕ), totalEntrances * totalFloors * n >= apartment_number)
  : floor = 3 :=
  sorry

end NUMINAMATH_GPT_find_floor_l1042_104216


namespace NUMINAMATH_GPT_trumpet_cost_l1042_104210

variable (total_amount : ℝ) (book_cost : ℝ)

theorem trumpet_cost (h1 : total_amount = 151) (h2 : book_cost = 5.84) :
  (total_amount - book_cost = 145.16) :=
by
  sorry

end NUMINAMATH_GPT_trumpet_cost_l1042_104210


namespace NUMINAMATH_GPT_fraction_savings_spent_on_furniture_l1042_104265

theorem fraction_savings_spent_on_furniture (savings : ℝ) (tv_cost : ℝ) (F : ℝ) 
  (h1 : savings = 840) (h2 : tv_cost = 210) 
  (h3 : F * savings + tv_cost = savings) : F = 3 / 4 :=
sorry

end NUMINAMATH_GPT_fraction_savings_spent_on_furniture_l1042_104265


namespace NUMINAMATH_GPT_determine_x_l1042_104203

theorem determine_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
by
  sorry

end NUMINAMATH_GPT_determine_x_l1042_104203


namespace NUMINAMATH_GPT_min_m_n_sum_l1042_104232

theorem min_m_n_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 108 * m = n^3) : m + n = 8 :=
  sorry

end NUMINAMATH_GPT_min_m_n_sum_l1042_104232


namespace NUMINAMATH_GPT_smallest_k_condition_l1042_104213

theorem smallest_k_condition (n k : ℕ) (h_n : n ≥ 2) (h_k : k = 2 * n) :
  ∀ (f : Fin n → Fin n → Fin k), (∀ i j, f i j < k) →
  (∃ a b c d : Fin n, a ≠ c ∧ b ≠ d ∧ f a b ≠ f a d ∧ f a b ≠ f c b ∧ f a b ≠ f c d ∧ f a d ≠ f c b ∧ f a d ≠ f c d ∧ f c b ≠ f c d) :=
sorry

end NUMINAMATH_GPT_smallest_k_condition_l1042_104213


namespace NUMINAMATH_GPT_initial_sand_in_bucket_A_l1042_104230

theorem initial_sand_in_bucket_A (C : ℝ) : 
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  x / C = 1 / 4 := by
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  show x / C = 1 / 4
  sorry

end NUMINAMATH_GPT_initial_sand_in_bucket_A_l1042_104230


namespace NUMINAMATH_GPT_digit_for_multiple_of_9_l1042_104217

theorem digit_for_multiple_of_9 : 
  -- Condition: Sum of the digits 4, 5, 6, 7, 8, and d must be divisible by 9.
  (∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (4 + 5 + 6 + 7 + 8 + d) % 9 = 0) →
  -- Result: The digit d that makes 45678d a multiple of 9 is 6.
  d = 6 :=
by
  sorry

end NUMINAMATH_GPT_digit_for_multiple_of_9_l1042_104217


namespace NUMINAMATH_GPT_two_bedroom_units_l1042_104212

theorem two_bedroom_units {x y : ℕ} 
  (h1 : x + y = 12) 
  (h2 : 360 * x + 450 * y = 4950) : 
  y = 7 := 
by
  sorry

end NUMINAMATH_GPT_two_bedroom_units_l1042_104212


namespace NUMINAMATH_GPT_OC_eq_l1042_104202

variable {V : Type} [AddCommGroup V]

-- Given vectors a and b
variables (a b : V)

-- Conditions given in the problem
def OA := a + b
def AB := 3 • (a - b)
def CB := 2 • a + b

-- Prove that OC = 2a - 3b
theorem OC_eq : (a + b) + (3 • (a - b)) + (- (2 • a + b)) = 2 • a - 3 • b :=
by
  -- write your proof here
  sorry

end NUMINAMATH_GPT_OC_eq_l1042_104202


namespace NUMINAMATH_GPT_hearty_total_beads_l1042_104234

-- Definition of the problem conditions
def blue_beads_per_package (r : ℕ) : ℕ := 2 * r
def red_beads_per_package : ℕ := 40
def red_packages : ℕ := 5
def blue_packages : ℕ := 3

-- Define the total number of beads Hearty has
def total_beads (r : ℕ) (rp : ℕ) (bp : ℕ) : ℕ :=
  (rp * red_beads_per_package) + (bp * blue_beads_per_package red_beads_per_package)

-- The theorem to be proven
theorem hearty_total_beads : total_beads red_beads_per_package red_packages blue_packages = 440 := by
  sorry

end NUMINAMATH_GPT_hearty_total_beads_l1042_104234


namespace NUMINAMATH_GPT_max_items_with_discount_l1042_104211

theorem max_items_with_discount (total_money items original_price discount : ℕ) 
  (h_orig: original_price = 30)
  (h_discount: discount = 24) 
  (h_limit: items > 5 → (total_money <= 270)) : items ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_max_items_with_discount_l1042_104211


namespace NUMINAMATH_GPT_simplify_complex_number_l1042_104269

theorem simplify_complex_number (i : ℂ) (h : i^2 = -1) : i * (1 - i)^2 = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_complex_number_l1042_104269


namespace NUMINAMATH_GPT_height_of_brick_l1042_104284

-- Definitions of given conditions
def length_brick : ℝ := 125
def width_brick : ℝ := 11.25
def length_wall : ℝ := 800
def height_wall : ℝ := 600
def width_wall : ℝ := 22.5
def number_bricks : ℝ := 1280

-- Prove that the height of each brick is 6.01 cm
theorem height_of_brick :
  ∃ H : ℝ,
    H = 6.01 ∧
    (number_bricks * (length_brick * width_brick * H) = length_wall * height_wall * width_wall) :=
by
  sorry

end NUMINAMATH_GPT_height_of_brick_l1042_104284


namespace NUMINAMATH_GPT_initial_peanuts_count_l1042_104271

def peanuts_initial (P : ℕ) : Prop :=
  P - (1 / 4 : ℝ) * P - 29 = 82

theorem initial_peanuts_count (P : ℕ) (h : peanuts_initial P) : P = 148 :=
by
  -- The complete proof can be constructed here.
  sorry

end NUMINAMATH_GPT_initial_peanuts_count_l1042_104271


namespace NUMINAMATH_GPT_exponentiation_equality_l1042_104246

theorem exponentiation_equality :
  3^12 * 8^12 * 3^3 * 8^8 = 24 ^ 15 * 32768 := by
  sorry

end NUMINAMATH_GPT_exponentiation_equality_l1042_104246


namespace NUMINAMATH_GPT_mul_fraction_eq_l1042_104239

theorem mul_fraction_eq : 7 * (1 / 11) * 33 = 21 :=
by
  sorry

end NUMINAMATH_GPT_mul_fraction_eq_l1042_104239
