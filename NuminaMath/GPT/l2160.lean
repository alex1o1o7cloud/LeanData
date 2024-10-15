import Mathlib

namespace NUMINAMATH_GPT_younger_brother_height_l2160_216064

theorem younger_brother_height
  (O Y : ℕ)
  (h1 : O - Y = 12)
  (h2 : O + Y = 308) :
  Y = 148 :=
by
  sorry

end NUMINAMATH_GPT_younger_brother_height_l2160_216064


namespace NUMINAMATH_GPT_find_number_eq_150_l2160_216053

variable {x : ℝ}

theorem find_number_eq_150 (h : 0.60 * x - 40 = 50) : x = 150 :=
sorry

end NUMINAMATH_GPT_find_number_eq_150_l2160_216053


namespace NUMINAMATH_GPT_shorter_piece_length_l2160_216033

theorem shorter_piece_length (total_len : ℝ) (ratio : ℝ) (shorter_len : ℝ) (longer_len : ℝ) 
  (h1 : total_len = 49) (h2 : ratio = 2/5) (h3 : shorter_len = x) 
  (h4 : longer_len = (5/2) * x) (h5 : shorter_len + longer_len = total_len) : 
  shorter_len = 14 := 
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l2160_216033


namespace NUMINAMATH_GPT_derivative_sum_l2160_216074

theorem derivative_sum (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (hf : ∀ x, deriv f x = f' x)
  (h : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 :=
sorry

end NUMINAMATH_GPT_derivative_sum_l2160_216074


namespace NUMINAMATH_GPT_no_linear_term_l2160_216050

theorem no_linear_term (m : ℝ) (x : ℝ) : 
  (x + m) * (x + 3) - (x * x + 3 * m) = 0 → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_no_linear_term_l2160_216050


namespace NUMINAMATH_GPT_maximal_value_ratio_l2160_216023

theorem maximal_value_ratio (a b c h : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_altitude : h = (a * b) / c) :
  ∃ θ : ℝ, a = c * Real.cos θ ∧ b = c * Real.sin θ ∧ (1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2) ∧
  ( Real.cos θ * Real.sin θ = (1 + 2 * Real.cos θ * Real.sin θ - 1) / 2 ) → 
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_maximal_value_ratio_l2160_216023


namespace NUMINAMATH_GPT_semicircle_circumference_correct_l2160_216051

noncomputable def perimeter_of_rectangle (l b : ℝ) : ℝ := 2 * (l + b)
noncomputable def side_of_square_by_rectangle (l b : ℝ) : ℝ := perimeter_of_rectangle l b / 4
noncomputable def circumference_of_semicircle (d : ℝ) : ℝ := (Real.pi * (d / 2)) + d

theorem semicircle_circumference_correct :
  let l := 16
  let b := 12
  let d := side_of_square_by_rectangle l b
  circumference_of_semicircle d = 35.98 :=
by
  sorry

end NUMINAMATH_GPT_semicircle_circumference_correct_l2160_216051


namespace NUMINAMATH_GPT_total_spending_in_4_years_is_680_l2160_216003

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end NUMINAMATH_GPT_total_spending_in_4_years_is_680_l2160_216003


namespace NUMINAMATH_GPT_find_abc_l2160_216086

theorem find_abc : ∃ (a b c : ℝ), a + b + c = 1 ∧ 4 * a + 2 * b + c = 5 ∧ 9 * a + 3 * b + c = 13 ∧ a - b + c = 5 := by
  sorry

end NUMINAMATH_GPT_find_abc_l2160_216086


namespace NUMINAMATH_GPT_minimum_daily_production_to_avoid_losses_l2160_216018

theorem minimum_daily_production_to_avoid_losses (x : ℕ) :
  (∀ x, (10 * x) ≥ (5 * x + 4000)) → (x ≥ 800) :=
sorry

end NUMINAMATH_GPT_minimum_daily_production_to_avoid_losses_l2160_216018


namespace NUMINAMATH_GPT_compound_interest_semiannual_l2160_216004

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_semiannual :
  compound_interest 150 0.20 2 1 = 181.50 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_semiannual_l2160_216004


namespace NUMINAMATH_GPT_find_a_l2160_216091

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem find_a (a : ℝ) (h : binom_coeff 9 3 * (-a)^3 = -84) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2160_216091


namespace NUMINAMATH_GPT_anya_takes_home_balloons_l2160_216069

theorem anya_takes_home_balloons:
  ∀ (total_balloons : ℕ) (colors : ℕ) (half : ℕ) (balloons_per_color : ℕ),
  total_balloons = 672 →
  colors = 4 →
  balloons_per_color = total_balloons / colors →
  half = balloons_per_color / 2 →
  half = 84 :=
by 
  intros total_balloons colors half balloons_per_color 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_anya_takes_home_balloons_l2160_216069


namespace NUMINAMATH_GPT_find_tangent_lines_l2160_216034

noncomputable def tangent_lines (x y : ℝ) : Prop :=
  (x = 2 ∨ 3 * x - 4 * y + 10 = 0)

theorem find_tangent_lines :
  ∃ (x y : ℝ), tangent_lines x y ∧ (x^2 + y^2 = 4) ∧ ((x, y) ≠ (2, 4)) :=
by
  sorry

end NUMINAMATH_GPT_find_tangent_lines_l2160_216034


namespace NUMINAMATH_GPT_turtles_remaining_l2160_216037

-- Define the initial number of turtles
def initial_turtles : ℕ := 9

-- Define the number of turtles that climbed onto the log
def climbed_turtles : ℕ := 3 * initial_turtles - 2

-- Define the total number of turtles on the log before any jump off
def total_turtles_before_jumping : ℕ := initial_turtles + climbed_turtles

-- Define the number of turtles remaining after half jump off
def remaining_turtles : ℕ := total_turtles_before_jumping / 2

theorem turtles_remaining : remaining_turtles = 17 :=
  by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_turtles_remaining_l2160_216037


namespace NUMINAMATH_GPT_speed_of_stream_l2160_216094

variable (b s : ℝ)

-- Conditions:
def downstream_eq : Prop := 90 = (b + s) * 3
def upstream_eq : Prop := 72 = (b - s) * 3

-- Goal:
theorem speed_of_stream (h1 : downstream_eq b s) (h2 : upstream_eq b s) : s = 3 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l2160_216094


namespace NUMINAMATH_GPT_find_common_ratio_l2160_216098

variable (a : ℕ → ℝ) -- represents the geometric sequence
variable (q : ℝ) -- represents the common ratio

-- conditions given in the problem
def a_3_condition : a 3 = 4 := sorry
def a_6_condition : a 6 = 1 / 2 := sorry

-- the general form of the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * q ^ n

-- the theorem we want to prove
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 4) (h2 : a 6 = 1 / 2) 
  (hg : geometric_sequence a q) : q = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l2160_216098


namespace NUMINAMATH_GPT_degree_g_greater_than_5_l2160_216011

-- Definitions according to the given conditions
variables {f g : Polynomial ℤ}
variables (h : Polynomial ℤ)
variables (r : Fin 81 → ℤ)

-- Condition 1: g(x) divides f(x), meaning there exists an h(x) such that f(x) = g(x) * h(x)
def divides (g f : Polynomial ℤ) := ∃ (h : Polynomial ℤ), f = g * h

-- Condition 2: f(x) - 2008 has at least 81 distinct integer roots
def has_81_distinct_roots (f : Polynomial ℤ) (roots : Fin 81 → ℤ) : Prop :=
  ∀ i : Fin 81, f.eval (roots i) = 2008 ∧ Function.Injective roots

-- The theorem to prove
theorem degree_g_greater_than_5 (nonconst_f : f.degree > 0) (nonconst_g : g.degree > 0) 
  (g_div_f : divides g f) (f_has_roots : has_81_distinct_roots (f - Polynomial.C 2008) r) :
  g.degree > 5 :=
sorry

end NUMINAMATH_GPT_degree_g_greater_than_5_l2160_216011


namespace NUMINAMATH_GPT_find_y_coordinate_l2160_216035

theorem find_y_coordinate (m n : ℝ) 
  (h₁ : m = 2 * n + 5) 
  (h₂ : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := 
sorry

end NUMINAMATH_GPT_find_y_coordinate_l2160_216035


namespace NUMINAMATH_GPT_min_value_of_function_l2160_216083

theorem min_value_of_function :
  ∀ x : ℝ, x > -1 → (y : ℝ) = (x^2 + 7*x + 10) / (x + 1) → y ≥ 9 :=
by
  intros x hx h
  sorry

end NUMINAMATH_GPT_min_value_of_function_l2160_216083


namespace NUMINAMATH_GPT_age_ratio_l2160_216006

theorem age_ratio (Tim_age : ℕ) (John_age : ℕ) (ratio : ℚ) 
  (h1 : Tim_age = 79) 
  (h2 : John_age = 35) 
  (h3 : Tim_age = ratio * John_age - 5) : 
  ratio = 2.4 := 
by sorry

end NUMINAMATH_GPT_age_ratio_l2160_216006


namespace NUMINAMATH_GPT_starfish_arms_l2160_216067

variable (x : ℕ)

theorem starfish_arms :
  (7 * x + 14 = 49) → (x = 5) := by
  sorry

end NUMINAMATH_GPT_starfish_arms_l2160_216067


namespace NUMINAMATH_GPT_inequality_proof_l2160_216055

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  1/a + 1/b + 1/c ≥ 2/(a + b) + 2/(b + c) + 2/(c + a) ∧ 2/(a + b) + 2/(b + c) + 2/(c + a) ≥ 9/(a + b + c) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2160_216055


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l2160_216079

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l2160_216079


namespace NUMINAMATH_GPT_height_of_david_l2160_216014

theorem height_of_david
  (building_height : ℕ)
  (building_shadow : ℕ)
  (david_shadow : ℕ)
  (ratio : ℕ)
  (h1 : building_height = 50)
  (h2 : building_shadow = 25)
  (h3 : david_shadow = 18)
  (h4 : ratio = building_height / building_shadow) :
  david_shadow * ratio = 36 := sorry

end NUMINAMATH_GPT_height_of_david_l2160_216014


namespace NUMINAMATH_GPT_paul_money_left_l2160_216024

-- Conditions
def cost_of_bread : ℕ := 2
def cost_of_butter : ℕ := 3
def cost_of_juice : ℕ := 2 * cost_of_bread
def total_money : ℕ := 15

-- Definition of total cost
def total_cost := cost_of_bread + cost_of_butter + cost_of_juice

-- Statement of the theorem
theorem paul_money_left : total_money - total_cost = 6 := by
  -- Sorry, implementation skipped
  sorry

end NUMINAMATH_GPT_paul_money_left_l2160_216024


namespace NUMINAMATH_GPT_ticTacToe_CarlWins_l2160_216076

def ticTacToeBoard := Fin 3 × Fin 3

noncomputable def countConfigurations : Nat := sorry

theorem ticTacToe_CarlWins :
  countConfigurations = 148 :=
sorry

end NUMINAMATH_GPT_ticTacToe_CarlWins_l2160_216076


namespace NUMINAMATH_GPT_johns_subtraction_l2160_216071

theorem johns_subtraction : 
  ∀ (a : ℕ), 
  a = 40 → 
  (a - 1)^2 = a^2 - 79 := 
by 
  -- The proof is omitted as per instruction
  sorry

end NUMINAMATH_GPT_johns_subtraction_l2160_216071


namespace NUMINAMATH_GPT_yearly_production_target_l2160_216016

-- Definitions for the conditions
def p_current : ℕ := 100
def p_add : ℕ := 50

-- The theorem to be proven
theorem yearly_production_target : (p_current + p_add) * 12 = 1800 := by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_yearly_production_target_l2160_216016


namespace NUMINAMATH_GPT_find_b_l2160_216088

theorem find_b (a b : ℤ) (h1 : 0 ≤ a) (h2 : a < 2^2008) (h3 : 0 ≤ b) (h4 : b < 8) (h5 : 7 * (a + 2^2008 * b) % 2^2011 = 1) :
  b = 3 :=
sorry

end NUMINAMATH_GPT_find_b_l2160_216088


namespace NUMINAMATH_GPT_gym_guest_count_l2160_216059

theorem gym_guest_count (G : ℕ) (H1 : ∀ G, 0 < G → ∀ G, G * 5.7 = 285 ∧ G = 50) : G = 50 :=
by
  sorry

end NUMINAMATH_GPT_gym_guest_count_l2160_216059


namespace NUMINAMATH_GPT_tangent_k_value_one_common_point_range_l2160_216066

namespace Geometry

-- Definitions:
def line (k : ℝ) : ℝ → ℝ := λ x => k * x - 3 * k + 2
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4
def is_tangent (k : ℝ) : Prop := |-2 * k + 3| / (Real.sqrt (k^2 + 1)) = 2
def has_only_one_common_point (k : ℝ) : Prop :=
  (1 / 2 < k ∧ k <= 5 / 2) ∨ (k = 5 / 12)

-- Theorem statements:
theorem tangent_k_value : ∀ k : ℝ, is_tangent k → k = 5 / 12 := sorry

theorem one_common_point_range : ∀ k : ℝ, has_only_one_common_point k → k ∈
  Set.union (Set.Ioc (1 / 2) (5 / 2)) {5 / 12} := sorry

end Geometry

end NUMINAMATH_GPT_tangent_k_value_one_common_point_range_l2160_216066


namespace NUMINAMATH_GPT_eq_has_infinite_solutions_l2160_216005

theorem eq_has_infinite_solutions (b : ℤ) :
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by 
  sorry

end NUMINAMATH_GPT_eq_has_infinite_solutions_l2160_216005


namespace NUMINAMATH_GPT_fraction_of_4d_nails_l2160_216019

variables (fraction2d fraction2d_or_4d fraction4d : ℚ)

theorem fraction_of_4d_nails
  (h1 : fraction2d = 0.25)
  (h2 : fraction2d_or_4d = 0.75) :
  fraction4d = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_4d_nails_l2160_216019


namespace NUMINAMATH_GPT_triangle_side_length_l2160_216028

theorem triangle_side_length (x : ℝ) (h1 : 6 < x) (h2 : x < 14) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l2160_216028


namespace NUMINAMATH_GPT_vector_subtraction_l2160_216077

variables (a b : ℝ × ℝ)

-- Definitions based on conditions
def vector_a : ℝ × ℝ := (1, -2)
def m : ℝ := 2
def vector_b : ℝ × ℝ := (4, m)

-- Prove given question equals answer
theorem vector_subtraction :
  vector_a = (1, -2) →
  vector_b = (4, m) →
  (1 * 4 + (-2) * m = 0) →
  5 • vector_a - vector_b = (1, -12) := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_vector_subtraction_l2160_216077


namespace NUMINAMATH_GPT_vector_AB_equality_l2160_216017

variable {V : Type*} [AddCommGroup V]

variables (a b : V)

theorem vector_AB_equality (BC CA : V) (hBC : BC = a) (hCA : CA = b) :
  CA - BC = b - a :=
by {
  sorry
}

end NUMINAMATH_GPT_vector_AB_equality_l2160_216017


namespace NUMINAMATH_GPT_angles_equal_l2160_216048

variables {A B C M W L T : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace M] [MetricSpace W] [MetricSpace L] [MetricSpace T]

-- A, B, C are points of the triangle ABC with incircle k.
-- Line_segment AC is longer than line segment BC.
-- M is the intersection of median from C.
-- W is the intersection of angle bisector from C.
-- L is the intersection of altitude from C.
-- T is the point where the tangent from M to the incircle k, different from AB, touches k.
def triangle_ABC (A B C : Type*) : Prop := sorry
def incircle_k (A B C : Type*) (k : Type*) : Prop := sorry
def longer_AC (A B C : Type*) : Prop := sorry
def intersection_median_C (M C : Type*) : Prop := sorry
def intersection_angle_bisector_C (W C : Type*) : Prop := sorry
def intersection_altitude_C (L C : Type*) : Prop := sorry
def tangent_through_M (M T k : Type*) : Prop := sorry
def touches_k (T k : Type*) : Prop := sorry
def angle_eq (M T W L : Type*) : Prop := sorry

theorem angles_equal (A B C M W L T k : Type*)
  (h_triangle : triangle_ABC A B C)
  (h_incircle : incircle_k A B C k)
  (h_longer_AC : longer_AC A B C)
  (h_inter_median : intersection_median_C M C)
  (h_inter_bisector : intersection_angle_bisector_C W C)
  (h_inter_altitude : intersection_altitude_C L C)
  (h_tangent : tangent_through_M M T k)
  (h_touches : touches_k T k) :
  angle_eq M T W L := 
sorry


end NUMINAMATH_GPT_angles_equal_l2160_216048


namespace NUMINAMATH_GPT_cost_for_15_pounds_of_apples_l2160_216038

-- Axiom stating the cost of apples per weight
axiom cost_of_apples (pounds : ℕ) : ℕ

-- Condition given in the problem
def rate_apples : Prop := cost_of_apples 5 = 4

-- Statement of the problem
theorem cost_for_15_pounds_of_apples : rate_apples → cost_of_apples 15 = 12 :=
by
  intro h
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_cost_for_15_pounds_of_apples_l2160_216038


namespace NUMINAMATH_GPT_deborah_oranges_zero_l2160_216072

-- Definitions for given conditions.
def initial_oranges : Float := 55.0
def oranges_added_by_susan : Float := 35.0
def total_oranges_after : Float := 90.0

-- Defining Deborah's oranges in her bag.
def oranges_in_bag : Float := total_oranges_after - (initial_oranges + oranges_added_by_susan)

-- The theorem to be proved.
theorem deborah_oranges_zero : oranges_in_bag = 0 := by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_deborah_oranges_zero_l2160_216072


namespace NUMINAMATH_GPT_sequence_a_general_term_sequence_b_sum_of_first_n_terms_l2160_216060

variable {n : ℕ}

def a (n : ℕ) : ℕ := 2 * n

def b (n : ℕ) : ℕ := 3^(n-1) + 2 * n

def T (n : ℕ) : ℕ := (3^n - 1) / 2 + n^2 + n

theorem sequence_a_general_term :
  (∀ n, a n = 2 * n) :=
by
  intro n
  sorry

theorem sequence_b_sum_of_first_n_terms :
  (∀ n, T n = (3^n - 1) / 2 + n^2 + n) :=
by
  intro n
  sorry

end NUMINAMATH_GPT_sequence_a_general_term_sequence_b_sum_of_first_n_terms_l2160_216060


namespace NUMINAMATH_GPT_mixed_doubles_teams_l2160_216052

theorem mixed_doubles_teams (m n : ℕ) (h_m : m = 7) (h_n : n = 5) :
  (∃ (k : ℕ), k = 4) ∧ (m ≥ 2) ∧ (n ≥ 2) →
  ∃ (number_of_combinations : ℕ), number_of_combinations = 2 * Nat.choose 7 2 * Nat.choose 5 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mixed_doubles_teams_l2160_216052


namespace NUMINAMATH_GPT_hyperbola_focus_coordinates_l2160_216025

theorem hyperbola_focus_coordinates :
  let a := 7
  let b := 11
  let h := 5
  let k := -3
  let c := Real.sqrt (a^2 + b^2)
  (∃ x y : ℝ, (x = h + c ∧ y = k) ∧ (∀ x' y', (x' = h + c ∧ y' = k) ↔ (x = x' ∧ y = y'))) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_coordinates_l2160_216025


namespace NUMINAMATH_GPT_expand_expression_l2160_216010

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2160_216010


namespace NUMINAMATH_GPT_tan_of_13pi_over_6_l2160_216056

theorem tan_of_13pi_over_6 : Real.tan (13 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_of_13pi_over_6_l2160_216056


namespace NUMINAMATH_GPT_sum_of_four_squares_eq_20_l2160_216081

variable (x y : ℕ)

-- Conditions based on the provided problem
def condition1 := 2 * x + 2 * y = 16
def condition2 := 2 * x + 3 * y = 19

-- Theorem to be proven
theorem sum_of_four_squares_eq_20 (h1 : condition1 x y) (h2 : condition2 x y) : 4 * x = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_four_squares_eq_20_l2160_216081


namespace NUMINAMATH_GPT_ratio_of_roots_l2160_216027

theorem ratio_of_roots (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) (h₃ : a * x₁^2 + b * x₁ + c = 0) (h₄ : a * x₂^2 + b * x₂ + c = 0) (h₅ : x₁ = 4 * x₂) : (b^2) / (a * c) = 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_roots_l2160_216027


namespace NUMINAMATH_GPT_radian_measure_of_sector_l2160_216078

-- Lean statement for the proof problem
theorem radian_measure_of_sector (R : ℝ) (hR : 0 < R) (h_area : (1 / 2) * (2 : ℝ) * R^2 = R^2) : 
  (2 : ℝ) = 2 :=
by 
  sorry
 
end NUMINAMATH_GPT_radian_measure_of_sector_l2160_216078


namespace NUMINAMATH_GPT_math_equivalent_proof_l2160_216042

-- Define the probabilities given the conditions
def P_A1 := 3 / 4
def P_A2 := 2 / 3
def P_A3 := 1 / 2
def P_B1 := 3 / 5
def P_B2 := 2 / 5

-- Define events
def P_C : ℝ := (P_A1 * P_B1 * (1 - P_A2)) + (P_A1 * P_B1 * P_A2 * P_B2 * (1 - P_A3))

-- Probability distribution of X
def P_X_0 : ℝ := (1 - P_A1) + P_C
def P_X_600 : ℝ := P_A1 * (1 - P_B1)
def P_X_1500 : ℝ := P_A1 * P_B1 * P_A2 * (1 - P_B2)
def P_X_3000 : ℝ := P_A1 * P_B1 * P_A2 * P_B2 * P_A3

-- Expected value of X
def E_X : ℝ := 600 * P_X_600 + 1500 * P_X_1500 + 3000 * P_X_3000

-- Statement to prove P(C) and expected value E(X)
theorem math_equivalent_proof :
  P_C = 21 / 100 ∧ 
  P_X_0 = 23 / 50 ∧
  P_X_600 = 3 / 10 ∧
  P_X_1500 = 9 / 50 ∧
  P_X_3000 = 3 / 50 ∧ 
  E_X = 630 := 
by 
  sorry

end NUMINAMATH_GPT_math_equivalent_proof_l2160_216042


namespace NUMINAMATH_GPT_mean_points_scored_is_48_l2160_216000

def class_points : List ℤ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_points_scored_is_48 : (class_points.sum / class_points.length) = 48 := by
  sorry

end NUMINAMATH_GPT_mean_points_scored_is_48_l2160_216000


namespace NUMINAMATH_GPT_ratio_of_squares_l2160_216022

theorem ratio_of_squares (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + 2 * y + 3 * z = 0) :
    (x^2 + y^2 + z^2) / (x * y + y * z + z * x) = -4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_squares_l2160_216022


namespace NUMINAMATH_GPT_find_valid_pair_l2160_216054

noncomputable def valid_angle (x : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ x = 180 * (n - 2) / n

noncomputable def valid_pair (x k : ℕ) : Prop :=
  valid_angle x ∧ valid_angle (k * x) ∧ 1 < k ∧ k < 5

theorem find_valid_pair : valid_pair 60 2 :=
by
  sorry

end NUMINAMATH_GPT_find_valid_pair_l2160_216054


namespace NUMINAMATH_GPT_range_of_a_l2160_216020

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x - 2 * a) * (a * x - 1) < 0 → (x > 1 / a ∨ x < 2 * a)) → (a ≤ -Real.sqrt 2 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2160_216020


namespace NUMINAMATH_GPT_intersection_S_T_l2160_216058

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end NUMINAMATH_GPT_intersection_S_T_l2160_216058


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2160_216085

theorem simplify_and_evaluate_expression :
  ∀ x : ℤ, -1 ≤ x ∧ x ≤ 2 →
  (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2) →
  ( ( (x^2 - 1) / (x^2 - 2*x + 1) + ((x^2 - 2*x) / (x - 2)) / x ) = 1 ) :=
by
  intros x hx_constraints x_ne_criteria
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2160_216085


namespace NUMINAMATH_GPT_find_other_number_l2160_216061

-- Define LCM and HCF conditions
def lcm_a_b := 2310
def hcf_a_b := 83
def number_a := 210

-- Define the problem to find the other number
def number_b : ℕ :=
  lcm_a_b * hcf_a_b / number_a

-- Statement: Prove that the other number is 913
theorem find_other_number : number_b = 913 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_find_other_number_l2160_216061


namespace NUMINAMATH_GPT_tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l2160_216040

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

noncomputable def tangent_line_p (x y : ℝ) : Prop :=
  2 * x - sqrt 5 * y - 9 = 0

noncomputable def line_q1 (x y : ℝ) : Prop :=
  x = 3

noncomputable def line_q2 (x y : ℝ) : Prop :=
  8 * x - 15 * y + 51 = 0

theorem tangent_line_through_P :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (2, -sqrt 5) →
    tangent_line_p x y := 
sorry

theorem tangent_line_through_Q1 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q1 x y := 
sorry

theorem tangent_line_through_Q2 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q2 x y := 
sorry

end NUMINAMATH_GPT_tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l2160_216040


namespace NUMINAMATH_GPT_total_cards_1750_l2160_216013

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end NUMINAMATH_GPT_total_cards_1750_l2160_216013


namespace NUMINAMATH_GPT_digit_for_divisibility_by_5_l2160_216043

theorem digit_for_divisibility_by_5 (B : ℕ) (h : B < 10) :
  (∃ (n : ℕ), n = 527 * 10 + B ∧ n % 5 = 0) ↔ (B = 0 ∨ B = 5) :=
by sorry

end NUMINAMATH_GPT_digit_for_divisibility_by_5_l2160_216043


namespace NUMINAMATH_GPT_find_days_A_alone_works_l2160_216029

-- Given conditions
def A_is_twice_as_fast_as_B (a b : ℕ) : Prop := a = b / 2
def together_complete_in_12_days (a b : ℕ) : Prop := (1 / b + 1 / a) = 1 / 12

-- We need to prove that A alone can finish the work in 18 days.
def A_alone_in_18_days (a : ℕ) : Prop := a = 18

theorem find_days_A_alone_works :
  ∃ (a b : ℕ), A_is_twice_as_fast_as_B a b ∧ together_complete_in_12_days a b ∧ A_alone_in_18_days a :=
sorry

end NUMINAMATH_GPT_find_days_A_alone_works_l2160_216029


namespace NUMINAMATH_GPT_area_isosceles_right_triangle_l2160_216026

open Real

-- Define the condition that the hypotenuse of an isosceles right triangle is 4√2 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = (4 * sqrt 2)^2

-- State the theorem to prove the area of the triangle is 8 square units
theorem area_isosceles_right_triangle (a b : ℝ) (h : hypotenuse a b) : 
  a = b → 1/2 * a * b = 8 := 
by 
  intros
  -- Proof steps are not required, so we use 'sorry'
  sorry

end NUMINAMATH_GPT_area_isosceles_right_triangle_l2160_216026


namespace NUMINAMATH_GPT_distance_from_x_axis_l2160_216007

theorem distance_from_x_axis (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_GPT_distance_from_x_axis_l2160_216007


namespace NUMINAMATH_GPT_reflex_angle_at_G_correct_l2160_216021

noncomputable def reflex_angle_at_G
    (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80)
    : ℝ :=
  360 - (180 - (180 - angle_BAG) - (180 - angle_GEL))

theorem reflex_angle_at_G_correct :
    (∀ (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80),
    reflex_angle_at_G B A E L G on_line off_line angle_BAG angle_GEL h1 h2 = 340) := sorry

end NUMINAMATH_GPT_reflex_angle_at_G_correct_l2160_216021


namespace NUMINAMATH_GPT_find_highest_score_l2160_216080

-- Define the conditions for the proof
section
  variable {runs_innings : ℕ → ℕ}

  -- Total runs scored in 46 innings
  def total_runs (average num_innings : ℕ) : ℕ := average * num_innings
  def total_runs_46_innings := total_runs 60 46
  def total_runs_excluding_H_L := total_runs 58 44

  -- Evaluated difference and sum of scores
  def diff_H_and_L : ℕ := 180
  def sum_H_and_L : ℕ := total_runs_46_innings - total_runs_excluding_H_L

  -- Define the proof goal
  theorem find_highest_score (H L : ℕ)
    (h1 : H - L = diff_H_and_L)
    (h2 : H + L = sum_H_and_L) :
    H = 194 :=
  by
    sorry

end

end NUMINAMATH_GPT_find_highest_score_l2160_216080


namespace NUMINAMATH_GPT_minimum_socks_to_guarantee_20_pairs_l2160_216095

-- Definitions and conditions
def red_socks := 120
def green_socks := 100
def blue_socks := 80
def black_socks := 50
def number_of_pairs := 20

-- Statement
theorem minimum_socks_to_guarantee_20_pairs 
  (red_socks green_socks blue_socks black_socks number_of_pairs: ℕ) 
  (h1: red_socks = 120) 
  (h2: green_socks = 100) 
  (h3: blue_socks = 80) 
  (h4: black_socks = 50) 
  (h5: number_of_pairs = 20) : 
  ∃ min_socks, min_socks = 43 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_socks_to_guarantee_20_pairs_l2160_216095


namespace NUMINAMATH_GPT_isosceles_triangle_apex_angle_l2160_216030

theorem isosceles_triangle_apex_angle (base_angle : ℝ) (h_base_angle : base_angle = 42) : 
  180 - 2 * base_angle = 96 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_apex_angle_l2160_216030


namespace NUMINAMATH_GPT_find_t_l2160_216075

open Complex Real

theorem find_t (a b : ℂ) (t : ℝ) (h₁ : abs a = 3) (h₂ : abs b = 5) (h₃ : a * b = t - 3 * I) :
  t = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_find_t_l2160_216075


namespace NUMINAMATH_GPT_calc1_calc2_l2160_216049

theorem calc1 : (1 * -11 + 8 + (-14) = -17) := by
  sorry

theorem calc2 : (13 - (-12) + (-21) = 4) := by
  sorry

end NUMINAMATH_GPT_calc1_calc2_l2160_216049


namespace NUMINAMATH_GPT_polynomial_expansion_l2160_216068

noncomputable def poly1 (z : ℝ) : ℝ := 3 * z ^ 3 + 2 * z ^ 2 - 4 * z + 1
noncomputable def poly2 (z : ℝ) : ℝ := 2 * z ^ 4 - 3 * z ^ 2 + z - 5
noncomputable def expanded_poly (z : ℝ) : ℝ := 6 * z ^ 7 + 4 * z ^ 6 - 4 * z ^ 5 - 9 * z ^ 3 + 7 * z ^ 2 + z - 5

theorem polynomial_expansion (z : ℝ) : poly1 z * poly2 z = expanded_poly z := by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l2160_216068


namespace NUMINAMATH_GPT_sum_of_coefficients_is_256_l2160_216097

theorem sum_of_coefficients_is_256 :
  ∀ (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  ((x : ℤ) - a)^8 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 → 
  a5 = 56 →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 256 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_256_l2160_216097


namespace NUMINAMATH_GPT_sides_of_regular_polygon_with_20_diagonals_l2160_216093

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_sides_of_regular_polygon_with_20_diagonals_l2160_216093


namespace NUMINAMATH_GPT_R_depends_on_a_d_n_l2160_216012

-- Definition of sum of an arithmetic progression
def sum_arithmetic_progression (n : ℕ) (a d : ℤ) : ℤ := 
  n * (2 * a + (n - 1) * d) / 2

-- Definitions for s1, s2, and s4
def s1 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression n a d
def s2 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (2 * n) a d
def s4 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (4 * n) a d

-- Definition of R
def R (n : ℕ) (a d : ℤ) : ℤ := s4 n a d - s2 n a d - s1 n a d

-- Theorem stating R depends on a, d, and n
theorem R_depends_on_a_d_n : 
  ∀ (n : ℕ) (a d : ℤ), ∃ (p q r : ℤ), R n a d = p * a + q * d + r := 
by
  sorry

end NUMINAMATH_GPT_R_depends_on_a_d_n_l2160_216012


namespace NUMINAMATH_GPT_original_solution_percentage_l2160_216087

theorem original_solution_percentage (P : ℝ) (h1 : 0.5 * P + 0.5 * 30 = 40) : P = 50 :=
by
  sorry

end NUMINAMATH_GPT_original_solution_percentage_l2160_216087


namespace NUMINAMATH_GPT_part1_minimum_value_part2_zeros_inequality_l2160_216099

noncomputable def f (x a : ℝ) := x * Real.exp x - a * (Real.log x + x)

theorem part1_minimum_value (a : ℝ) :
  (∀ x > 0, f x a > 0) ∨ (∃ x > 0, f x a = a - a * Real.log a) :=
sorry

theorem part2_zeros_inequality (a x₁ x₂ : ℝ) (hx₁ : f x₁ a = 0) (hx₂ : f x₂ a = 0) :
  Real.exp (x₁ + x₂ - 2) > 1 / (x₁ * x₂) :=
sorry

end NUMINAMATH_GPT_part1_minimum_value_part2_zeros_inequality_l2160_216099


namespace NUMINAMATH_GPT_dave_total_rides_l2160_216090

theorem dave_total_rides (rides_first_day rides_second_day : ℕ) (h1 : rides_first_day = 4) (h2 : rides_second_day = 3) :
  rides_first_day + rides_second_day = 7 :=
by
  sorry

end NUMINAMATH_GPT_dave_total_rides_l2160_216090


namespace NUMINAMATH_GPT_find_ending_number_of_range_l2160_216045

theorem find_ending_number_of_range :
  ∃ n : ℕ, (∀ avg_200_400 avg_100_n : ℕ,
    avg_200_400 = (200 + 400) / 2 ∧
    avg_100_n = (100 + n) / 2 ∧
    avg_100_n + 150 = avg_200_400) ∧
    n = 200 :=
sorry

end NUMINAMATH_GPT_find_ending_number_of_range_l2160_216045


namespace NUMINAMATH_GPT_smallest_angle_pentagon_l2160_216044

theorem smallest_angle_pentagon (x : ℝ) (h : 16 * x = 540) : 2 * x = 67.5 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_angle_pentagon_l2160_216044


namespace NUMINAMATH_GPT_smallest_n_congruent_l2160_216065

theorem smallest_n_congruent (n : ℕ) (h : 635 * n ≡ 1251 * n [MOD 30]) : n = 15 :=
sorry

end NUMINAMATH_GPT_smallest_n_congruent_l2160_216065


namespace NUMINAMATH_GPT_cube_split_includes_2015_l2160_216096

theorem cube_split_includes_2015 (m : ℕ) (h1 : m > 1) (h2 : ∃ (k : ℕ), 2 * k + 1 = 2015) : m = 45 :=
by
  sorry

end NUMINAMATH_GPT_cube_split_includes_2015_l2160_216096


namespace NUMINAMATH_GPT_sally_pokemon_cards_l2160_216015

variable (X : ℤ)

theorem sally_pokemon_cards : X + 41 + 20 = 34 → X = -27 :=
by
  sorry

end NUMINAMATH_GPT_sally_pokemon_cards_l2160_216015


namespace NUMINAMATH_GPT_base_conversion_403_base_6_eq_223_base_8_l2160_216031

theorem base_conversion_403_base_6_eq_223_base_8 :
  (6^2 * 4 + 6^1 * 0 + 6^0 * 3 : ℕ) = (8^2 * 2 + 8^1 * 2 + 8^0 * 3 : ℕ) :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_403_base_6_eq_223_base_8_l2160_216031


namespace NUMINAMATH_GPT_div_by_7_of_sum_div_by_7_l2160_216036

theorem div_by_7_of_sum_div_by_7 (x y z : ℤ) (h : 7 ∣ x^3 + y^3 + z^3) : 7 ∣ x * y * z := by
  sorry

end NUMINAMATH_GPT_div_by_7_of_sum_div_by_7_l2160_216036


namespace NUMINAMATH_GPT_area_inside_octagon_outside_semicircles_l2160_216062

theorem area_inside_octagon_outside_semicircles :
  let s := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area := (1/2) * Real.pi * (s / 2)^2
  let total_semicircle_area := 8 * semicircle_area
  octagon_area - total_semicircle_area = 54 + 24 * Real.sqrt 2 - 9 * Real.pi :=
sorry

end NUMINAMATH_GPT_area_inside_octagon_outside_semicircles_l2160_216062


namespace NUMINAMATH_GPT_polynomial_divisibility_l2160_216047

def P (a : ℤ) (x : ℤ) : ℤ := x^1000 + a*x^2 + 9

theorem polynomial_divisibility (a : ℤ) : (P a (-1) = 0) ↔ (a = -10) := by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l2160_216047


namespace NUMINAMATH_GPT_points_on_square_diagonal_l2160_216039

theorem points_on_square_diagonal (a : ℝ) (ha : a > 1) (Q : ℝ × ℝ) (hQ : Q = (a + 1, 4 * a + 1)) 
    (line : ℝ × ℝ → Prop) (hline : ∀ (x y : ℝ), line (x, y) ↔ y = a * x + 3) :
    ∃ (P R : ℝ × ℝ), line Q ∧ P = (6, 3) ∧ R = (-3, 6) :=
by
  sorry

end NUMINAMATH_GPT_points_on_square_diagonal_l2160_216039


namespace NUMINAMATH_GPT_total_cost_meal_l2160_216082

-- Define the initial conditions
variables (x : ℝ) -- x represents the total cost of the meal

-- Initial number of friends
def initial_friends : ℝ := 4

-- New number of friends after additional friends join
def new_friends : ℝ := 7

-- The decrease in cost per friend
def cost_decrease : ℝ := 15

-- Lean statement to assert our proof
theorem total_cost_meal : x / initial_friends - x / new_friends = cost_decrease → x = 140 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_meal_l2160_216082


namespace NUMINAMATH_GPT_ratio_of_speeds_l2160_216092

theorem ratio_of_speeds (v_A v_B : ℝ)
  (h₀ : 4 * v_A = abs (600 - 4 * v_B))
  (h₁ : 9 * v_A = abs (600 - 9 * v_B)) :
  v_A / v_B = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2160_216092


namespace NUMINAMATH_GPT_largest_angle_90_degrees_l2160_216046

def triangle_altitudes (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (9 * a = 12 * b) ∧ (9 * a = 18 * c)

theorem largest_angle_90_degrees (a b c : ℝ) 
  (h : triangle_altitudes a b c) : 
  exists (A B C : ℝ) (hApos : A > 0) (hBpos : B > 0) (hCpos : C > 0),
    (A^2 = B^2 + C^2) ∧ (B * C / 2 = 9 * a / 2 ∨ 
                         B * A / 2 = 12 * b / 2 ∨ 
                         C * A / 2 = 18 * c / 2) :=
sorry

end NUMINAMATH_GPT_largest_angle_90_degrees_l2160_216046


namespace NUMINAMATH_GPT_find_N_is_20_l2160_216057

theorem find_N_is_20 : ∃ (N : ℤ), ∃ (u v : ℤ), (N + 5 = u ^ 2) ∧ (N - 11 = v ^ 2) ∧ (N = 20) :=
by
  sorry

end NUMINAMATH_GPT_find_N_is_20_l2160_216057


namespace NUMINAMATH_GPT_product_of_three_numbers_l2160_216032

theorem product_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 210) (h2 : 5 * a = b - 11) (h3 : 5 * a = c + 11) : a * b * c = 168504 :=
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l2160_216032


namespace NUMINAMATH_GPT_minimum_value_expression_l2160_216089

open Real

theorem minimum_value_expression (x y z : ℝ) (hxyz : x * y * z = 1 / 2) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * sqrt 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l2160_216089


namespace NUMINAMATH_GPT_C_can_complete_work_in_100_days_l2160_216041

-- Definitions for conditions
def A_work_rate : ℚ := 1 / 20
def B_work_rate : ℚ := 1 / 15
def work_done_by_A_and_B : ℚ := 6 * (1 / 20 + 1 / 15)
def remaining_work : ℚ := 1 - work_done_by_A_and_B
def work_done_by_A_in_5_days : ℚ := 5 * (1 / 20)
def work_done_by_C_in_5_days : ℚ := remaining_work - work_done_by_A_in_5_days
def C_work_rate_in_5_days : ℚ := work_done_by_C_in_5_days / 5

-- Statement to prove
theorem C_can_complete_work_in_100_days : 
  work_done_by_C_in_5_days ≠ 0 → 1 / C_work_rate_in_5_days = 100 :=
by
  -- proof of the theorem
  sorry

end NUMINAMATH_GPT_C_can_complete_work_in_100_days_l2160_216041


namespace NUMINAMATH_GPT_find_linear_equation_l2160_216073

def is_linear_eq (eq : String) : Prop :=
  eq = "2x = 0"

theorem find_linear_equation :
  is_linear_eq "2x = 0" :=
by
  sorry

end NUMINAMATH_GPT_find_linear_equation_l2160_216073


namespace NUMINAMATH_GPT_value_of_c7_l2160_216008

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

end NUMINAMATH_GPT_value_of_c7_l2160_216008


namespace NUMINAMATH_GPT_parallelogram_side_length_sum_l2160_216001

theorem parallelogram_side_length_sum (x y z : ℚ) 
  (h1 : 3 * x - 1 = 12)
  (h2 : 4 * z + 2 = 7 * y + 3) :
  x + y + z = 121 / 21 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_side_length_sum_l2160_216001


namespace NUMINAMATH_GPT_inequality_problem_l2160_216070

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := Real.logb (1 / 2) (1 / 3)

theorem inequality_problem :
  c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_inequality_problem_l2160_216070


namespace NUMINAMATH_GPT_syllogism_sequence_l2160_216084

theorem syllogism_sequence (P Q R : Prop)
  (h1 : R)
  (h2 : Q)
  (h3 : P) : 
  (Q ∧ R → P) → (R → P) ∧ (Q → (P ∧ R)) := 
by
  sorry

end NUMINAMATH_GPT_syllogism_sequence_l2160_216084


namespace NUMINAMATH_GPT_xy_inequality_l2160_216002

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) :=
sorry

end NUMINAMATH_GPT_xy_inequality_l2160_216002


namespace NUMINAMATH_GPT_people_in_room_l2160_216009

open Nat

theorem people_in_room (C : ℕ) (P : ℕ) (h1 : 1 / 4 * C = 6) (h2 : 3 / 4 * C = 2 / 3 * P) : P = 27 := by
  sorry

end NUMINAMATH_GPT_people_in_room_l2160_216009


namespace NUMINAMATH_GPT_min_value_l2160_216063

open Real

theorem min_value (x y : ℝ) (h : x + y = 4) : x^2 + y^2 ≥ 8 := by
  sorry

end NUMINAMATH_GPT_min_value_l2160_216063
