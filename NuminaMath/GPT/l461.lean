import Mathlib

namespace NUMINAMATH_GPT_probability_target_hit_l461_46188

theorem probability_target_hit (P_A P_B : ℚ) (h1 : P_A = 1/2) (h2 : P_B = 1/3) : 
  (1 - (1 - P_A) * (1 - P_B)) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_probability_target_hit_l461_46188


namespace NUMINAMATH_GPT_crayons_total_correct_l461_46117

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end NUMINAMATH_GPT_crayons_total_correct_l461_46117


namespace NUMINAMATH_GPT_fraction_power_rule_example_l461_46135

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end NUMINAMATH_GPT_fraction_power_rule_example_l461_46135


namespace NUMINAMATH_GPT_bags_sold_in_afternoon_l461_46196

theorem bags_sold_in_afternoon (bags_morning : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) 
  (h1 : bags_morning = 29) (h2 : weight_per_bag = 7) (h3 : total_weight = 322) : 
  total_weight - bags_morning * weight_per_bag / weight_per_bag = 17 := 
by 
  sorry

end NUMINAMATH_GPT_bags_sold_in_afternoon_l461_46196


namespace NUMINAMATH_GPT_action_figure_ratio_l461_46122

variable (initial : ℕ) (sold : ℕ) (remaining : ℕ) (left : ℕ)
variable (h1 : initial = 24)
variable (h2 : sold = initial / 4)
variable (h3 : remaining = initial - sold)
variable (h4 : remaining - left = left)

theorem action_figure_ratio
  (h1 : initial = 24)
  (h2 : sold = initial / 4)
  (h3 : remaining = initial - sold)
  (h4 : remaining - left = left) :
  (remaining - left) * 3 = left :=
by
  sorry

end NUMINAMATH_GPT_action_figure_ratio_l461_46122


namespace NUMINAMATH_GPT_find_values_of_m_and_n_l461_46123

theorem find_values_of_m_and_n (m n : ℝ) (h : m / (1 + I) = 1 - n * I) : 
  m = 2 ∧ n = 1 :=
sorry

end NUMINAMATH_GPT_find_values_of_m_and_n_l461_46123


namespace NUMINAMATH_GPT_line_equation_l461_46163

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4)) 
  (h_intercept_sum : ∃ b c, b + c = 0 ∧ (∀ x y, A.1 * x + A.2 * y = 1 ∨ A.1 * x + A.2 * y = -1)) :
  (∃ m n, m = 4 ∧ n = -1 ∧ (∀ x y, m * x + n * y = 0)) ∨ 
  (∃ p q r, p = 1 ∧ q = -1 ∧ r = 3 ∧ (∀ x y, p * x + q * y + r = 0)) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l461_46163


namespace NUMINAMATH_GPT_landscape_length_l461_46140

-- Define the conditions from the problem
def breadth (b : ℝ) := b > 0
def length_of_landscape (l b : ℝ) := l = 8 * b
def area_of_playground (pg_area : ℝ) := pg_area = 1200
def playground_fraction (A b : ℝ) := A = 8 * b^2
def fraction_of_landscape (pg_area A : ℝ) := pg_area = (1/6) * A

-- Main theorem statement
theorem landscape_length (b l A pg_area : ℝ) 
  (H_b : breadth b) 
  (H_length : length_of_landscape l b)
  (H_pg_area : area_of_playground pg_area)
  (H_pg_fraction : playground_fraction A b)
  (H_pg_landscape_fraction : fraction_of_landscape pg_area A) :
  l = 240 :=
by
  sorry

end NUMINAMATH_GPT_landscape_length_l461_46140


namespace NUMINAMATH_GPT_factor_expression_l461_46141

theorem factor_expression (y : ℝ) : 
  5 * y * (y + 2) + 8 * (y + 2) + 15 = (5 * y + 8) * (y + 2) + 15 := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l461_46141


namespace NUMINAMATH_GPT_min_value_expr_l461_46115

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x + 2 * y = 5) :
  (1 / (x - 1) + 1 / (y - 1)) = (3 / 2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_min_value_expr_l461_46115


namespace NUMINAMATH_GPT_stage_order_permutations_l461_46145

-- Define the problem in Lean terms
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem stage_order_permutations :
  let total_students := 6
  let predetermined_students := 3
  (permutations total_students) / (permutations predetermined_students) = 120 := by
  sorry

end NUMINAMATH_GPT_stage_order_permutations_l461_46145


namespace NUMINAMATH_GPT_zacharys_bus_ride_length_l461_46128

theorem zacharys_bus_ride_length (Vince Zachary : ℝ) (hV : Vince = 0.62) (hDiff : Vince = Zachary + 0.13) : Zachary = 0.49 :=
by
  sorry

end NUMINAMATH_GPT_zacharys_bus_ride_length_l461_46128


namespace NUMINAMATH_GPT_find_ages_l461_46174

-- Define that f is a polynomial with integer coefficients
noncomputable def f : ℤ → ℤ := sorry

-- Given conditions
axiom f_at_7 : f 7 = 77
axiom f_at_b : ∃ b : ℕ, f b = 85
axiom f_at_c : ∃ c : ℕ, f c = 0

-- Define what we need to prove
theorem find_ages : ∃ b c : ℕ, (b - 7 ∣ 8) ∧ (c - b ∣ 85) ∧ (c - 7 ∣ 77) ∧ (b = 9) ∧ (c = 14) :=
sorry

end NUMINAMATH_GPT_find_ages_l461_46174


namespace NUMINAMATH_GPT_ratio_of_sequences_is_5_over_4_l461_46105

-- Definitions of arithmetic sequences
def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Hypotheses
def sequence_1_sum : ℕ :=
  arithmetic_sum 5 5 16

def sequence_2_sum : ℕ :=
  arithmetic_sum 4 4 16

-- Main statement to be proven
theorem ratio_of_sequences_is_5_over_4 : sequence_1_sum / sequence_2_sum = 5 / 4 := sorry

end NUMINAMATH_GPT_ratio_of_sequences_is_5_over_4_l461_46105


namespace NUMINAMATH_GPT_no_generating_combination_l461_46150

-- Representing Rubik's Cube state as a type (assume a type exists)
axiom CubeState : Type

-- A combination of turns represented as a function on states
axiom A : CubeState → CubeState

-- Simple rotations
axiom P : CubeState → CubeState
axiom Q : CubeState → CubeState

-- Rubik's Cube property of generating combination (assuming generating implies all states achievable)
def is_generating (A : CubeState → CubeState) :=
  ∀ X : CubeState, ∃ m n : ℕ, P X = A^[m] X ∧ Q X = A^[n] X

-- Non-commutativity condition
axiom non_commutativity : ∀ X : CubeState, P (Q X) ≠ Q (P X)

-- Formal statement of the problem
theorem no_generating_combination : ¬ ∃ A : CubeState → CubeState, is_generating A :=
by sorry

end NUMINAMATH_GPT_no_generating_combination_l461_46150


namespace NUMINAMATH_GPT_number_of_real_solutions_l461_46136

noncomputable def f (x : ℝ) : ℝ := 2^(-x) + x^2 - 3

theorem number_of_real_solutions :
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∀ x : ℝ, f x = 0 → (x = x₁ ∨ x = x₂)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_solutions_l461_46136


namespace NUMINAMATH_GPT_number_thought_of_eq_95_l461_46144

theorem number_thought_of_eq_95 (x : ℝ) (h : (x / 5) + 23 = 42) : x = 95 := 
by
  sorry

end NUMINAMATH_GPT_number_thought_of_eq_95_l461_46144


namespace NUMINAMATH_GPT_possible_polynomials_l461_46180

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end NUMINAMATH_GPT_possible_polynomials_l461_46180


namespace NUMINAMATH_GPT_factor_is_three_l461_46127

theorem factor_is_three (x f : ℝ) (h1 : 2 * x + 5 = y) (h2 : f * y = 111) (h3 : x = 16):
  f = 3 :=
by
  sorry

end NUMINAMATH_GPT_factor_is_three_l461_46127


namespace NUMINAMATH_GPT_complex_number_equality_l461_46198

-- Define the conditions a, b ∈ ℝ and a + i = 1 - bi
theorem complex_number_equality (a b : ℝ) (i : ℂ) (h : a + i = 1 - b * i) : (a + b * i) ^ 8 = 16 :=
  sorry

end NUMINAMATH_GPT_complex_number_equality_l461_46198


namespace NUMINAMATH_GPT_union_of_sets_l461_46126

def M := {x : ℝ | -1 < x ∧ x < 1}
def N := {x : ℝ | x^2 - 3 * x ≤ 0}

theorem union_of_sets : M ∪ N = {x : ℝ | -1 < x ∧ x ≤ 3} :=
by sorry

end NUMINAMATH_GPT_union_of_sets_l461_46126


namespace NUMINAMATH_GPT_circle_tangent_line_standard_equation_l461_46166

-- Problem Statement:
-- Prove that the standard equation of the circle with center at (1,1)
-- and tangent to the line x + y = 4 is (x - 1)^2 + (y - 1)^2 = 2
theorem circle_tangent_line_standard_equation :
  (forall (x y : ℝ), (x + y = 4) -> (x - 1)^2 + (y - 1)^2 = 2) := by
  sorry

end NUMINAMATH_GPT_circle_tangent_line_standard_equation_l461_46166


namespace NUMINAMATH_GPT_chomp_game_configurations_l461_46106

/-- Number of valid configurations such that 0 ≤ a_1 ≤ a_2 ≤ ... ≤ a_5 ≤ 7 is 330 -/
theorem chomp_game_configurations :
  let valid_configs := {a : Fin 6 → Fin 8 // (∀ i j, i ≤ j → a i ≤ a j)}
  Fintype.card valid_configs = 330 :=
sorry

end NUMINAMATH_GPT_chomp_game_configurations_l461_46106


namespace NUMINAMATH_GPT_cylinder_surface_area_l461_46104

noncomputable def total_surface_area_cylinder (r h : ℝ) : ℝ :=
  let base_area := 64 * Real.pi
  let lateral_surface_area := 2 * Real.pi * r * h
  let total_surface_area := 2 * base_area + lateral_surface_area
  total_surface_area

theorem cylinder_surface_area (r h : ℝ) (hr : Real.pi * r^2 = 64 * Real.pi) (hh : h = 2 * r) : 
  total_surface_area_cylinder r h = 384 * Real.pi := by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l461_46104


namespace NUMINAMATH_GPT_geometric_sequence_value_l461_46148

theorem geometric_sequence_value (a : ℝ) (h₁ : 280 ≠ 0) (h₂ : 35 ≠ 0) : 
  (∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8 ∧ a > 0) → a = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_value_l461_46148


namespace NUMINAMATH_GPT_correct_expression_l461_46152

theorem correct_expression (a b c m x y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b ≠ 0) (h5 : x ≠ y) : 
  ¬ ( (a + m) / (b + m) = a / b ) ∧
  ¬ ( (a + b) / (a + b) = 0 ) ∧ 
  ¬ ( (a * b - 1) / (a * c - 1) = (b - 1) / (c - 1) ) ∧ 
  ( (x - y) / (x^2 - y^2) = 1 / (x + y) ) :=
by
  sorry

end NUMINAMATH_GPT_correct_expression_l461_46152


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_45_divisible_by_3_l461_46100

theorem smallest_positive_multiple_of_45_divisible_by_3 
  (x : ℕ) (hx: x > 0) : ∃ y : ℕ, y = 45 ∧ 45 ∣ y ∧ 3 ∣ y ∧ ∀ z : ℕ, (45 ∣ z ∧ 3 ∣ z ∧ z > 0) → z ≥ y :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_45_divisible_by_3_l461_46100


namespace NUMINAMATH_GPT_smallest_number_of_cubes_l461_46102

def box_length : ℕ := 49
def box_width : ℕ := 42
def box_depth : ℕ := 14
def gcd_box_dimensions : ℕ := Nat.gcd (Nat.gcd box_length box_width) box_depth

theorem smallest_number_of_cubes :
  (box_length / gcd_box_dimensions) *
  (box_width / gcd_box_dimensions) *
  (box_depth / gcd_box_dimensions) = 84 := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_cubes_l461_46102


namespace NUMINAMATH_GPT_log_base_5_domain_correct_l461_46191

def log_base_5_domain : Set ℝ := {x : ℝ | x > 0}

theorem log_base_5_domain_correct : (∀ x : ℝ, x > 0 ↔ x ∈ log_base_5_domain) :=
by sorry

end NUMINAMATH_GPT_log_base_5_domain_correct_l461_46191


namespace NUMINAMATH_GPT_smallest_natural_b_for_root_exists_l461_46187

-- Define the problem's conditions
def quadratic_eqn (b : ℕ) := ∀ x : ℝ, x^2 + (b : ℝ) * x + 25 = 0

def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Define the main problem statement
theorem smallest_natural_b_for_root_exists :
  ∃ b : ℕ, (discriminant 1 b 25 ≥ 0) ∧ (∀ b' : ℕ, b' < b → discriminant 1 b' 25 < 0) ∧ b = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_b_for_root_exists_l461_46187


namespace NUMINAMATH_GPT_average_first_50_even_numbers_l461_46158

-- Condition: The sequence starts from 2.
-- Condition: The sequence consists of the first 50 even numbers.
def first50EvenNumbers : List ℤ := List.range' 2 100

theorem average_first_50_even_numbers : (first50EvenNumbers.sum / 50 = 51) :=
by
  sorry

end NUMINAMATH_GPT_average_first_50_even_numbers_l461_46158


namespace NUMINAMATH_GPT_lateral_surface_area_ratio_l461_46107

theorem lateral_surface_area_ratio (r h : ℝ) :
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  cylinder_area / cone_area = 2 :=
by
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  sorry

end NUMINAMATH_GPT_lateral_surface_area_ratio_l461_46107


namespace NUMINAMATH_GPT_ratio_of_money_with_Gopal_and_Krishan_l461_46161

theorem ratio_of_money_with_Gopal_and_Krishan 
  (R G K : ℕ) 
  (h1 : R = 735) 
  (h2 : K = 4335) 
  (h3 : R * 17 = G * 7) :
  G * 4335 = 1785 * K :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_money_with_Gopal_and_Krishan_l461_46161


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l461_46153

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 14) : a + b + c = 39 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l461_46153


namespace NUMINAMATH_GPT_factorize_expression_l461_46177

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l461_46177


namespace NUMINAMATH_GPT_predicted_temperature_l461_46156

-- Define the observation data points
def data_points : List (ℕ × ℝ) :=
  [(20, 25), (30, 27.5), (40, 29), (50, 32.5), (60, 36)]

-- Define the linear regression equation with constant k
def regression (x : ℕ) (k : ℝ) : ℝ :=
  0.25 * x + k

-- Proof statement
theorem predicted_temperature (k : ℝ) (h : regression 40 k = 30) : regression 80 k = 40 :=
by
  sorry

end NUMINAMATH_GPT_predicted_temperature_l461_46156


namespace NUMINAMATH_GPT_right_triangle_width_l461_46133

theorem right_triangle_width (height : ℝ) (side_square : ℝ) (width : ℝ) (n_triangles : ℕ) 
  (triangle_right : height = 2)
  (fit_inside_square : side_square = 2)
  (number_triangles : n_triangles = 2) :
  width = 2 :=
sorry

end NUMINAMATH_GPT_right_triangle_width_l461_46133


namespace NUMINAMATH_GPT_total_blocks_l461_46132

-- Conditions
def original_blocks : ℝ := 35.0
def added_blocks : ℝ := 65.0

-- Question and proof goal
theorem total_blocks : original_blocks + added_blocks = 100.0 := 
by
  -- The proof would be provided here
  sorry

end NUMINAMATH_GPT_total_blocks_l461_46132


namespace NUMINAMATH_GPT_find_m_plus_n_l461_46113

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + x

theorem find_m_plus_n (m n : ℝ) (h1 : m < n ∧ n ≤ 1) (h2 : ∀ (x : ℝ), m ≤ x ∧ x ≤ n → 3 * m ≤ f x ∧ f x ≤ 3 * n) : m + n = -4 :=
by
  have H1 : - (1 / 2) * m^2 + m = 3 * m := sorry
  have H2 : - (1 / 2) * n^2 + n = 3 * n := sorry
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l461_46113


namespace NUMINAMATH_GPT_shooter_variance_l461_46134

def scores : List ℝ := [9.7, 9.9, 10.1, 10.2, 10.1] -- Defining the scores

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length -- Calculating the mean

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length -- Defining the variance

theorem shooter_variance :
  variance scores = 0.032 :=
by
  sorry -- Proof to be provided later

end NUMINAMATH_GPT_shooter_variance_l461_46134


namespace NUMINAMATH_GPT_point_in_plane_region_l461_46137

-- Defining the condition that the inequality represents a region on the plane
def plane_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

-- Stating that the point (0, 1) lies within the plane region represented by the inequality
theorem point_in_plane_region : plane_region 0 1 :=
by {
    sorry
}

end NUMINAMATH_GPT_point_in_plane_region_l461_46137


namespace NUMINAMATH_GPT_garden_area_l461_46143

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end NUMINAMATH_GPT_garden_area_l461_46143


namespace NUMINAMATH_GPT_prime_solution_l461_46182

theorem prime_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_prime_solution_l461_46182


namespace NUMINAMATH_GPT_probability_scrapped_l461_46130

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end NUMINAMATH_GPT_probability_scrapped_l461_46130


namespace NUMINAMATH_GPT_negation_statement_l461_46165

theorem negation_statement (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) : x^2 - x ≠ 0 :=
by sorry

end NUMINAMATH_GPT_negation_statement_l461_46165


namespace NUMINAMATH_GPT_team_structure_ways_l461_46169

open Nat

noncomputable def combinatorial_structure (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem team_structure_ways :
  let total_members := 13
  let team_lead_choices := total_members
  let remaining_after_lead := total_members - 1
  let project_manager_choices := combinatorial_structure remaining_after_lead 3
  let remaining_after_pm1 := remaining_after_lead - 3
  let subordinate_choices_pm1 := combinatorial_structure remaining_after_pm1 3
  let remaining_after_pm2 := remaining_after_pm1 - 3
  let subordinate_choices_pm2 := combinatorial_structure remaining_after_pm2 3
  let remaining_after_pm3 := remaining_after_pm2 - 3
  let subordinate_choices_pm3 := combinatorial_structure remaining_after_pm3 3
  let total_ways := team_lead_choices * project_manager_choices * subordinate_choices_pm1 * subordinate_choices_pm2 * subordinate_choices_pm3
  total_ways = 4804800 :=
by
  sorry

end NUMINAMATH_GPT_team_structure_ways_l461_46169


namespace NUMINAMATH_GPT_polar_line_equation_l461_46139

theorem polar_line_equation (r θ: ℝ) (p : r = 3 ∧ θ = 0) : r = 3 := 
by 
  sorry

end NUMINAMATH_GPT_polar_line_equation_l461_46139


namespace NUMINAMATH_GPT_complement_U_P_l461_46121

def U (y : ℝ) : Prop := y > 0
def P (y : ℝ) : Prop := 0 < y ∧ y < 1/3

theorem complement_U_P :
  {y : ℝ | U y} \ {y : ℝ | P y} = {y : ℝ | y ≥ 1/3} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_P_l461_46121


namespace NUMINAMATH_GPT_outdoor_chairs_count_l461_46175

theorem outdoor_chairs_count (indoor_tables outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) 
  (total_chairs : ℕ) (h1: indoor_tables = 9) (h2: outdoor_tables = 11) 
  (h3: chairs_per_indoor_table = 10) (h4: total_chairs = 123) : 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 :=
by 
  sorry

end NUMINAMATH_GPT_outdoor_chairs_count_l461_46175


namespace NUMINAMATH_GPT_first_week_tickets_calc_l461_46110

def total_tickets : ℕ := 90
def second_week_tickets : ℕ := 17
def tickets_left : ℕ := 35

theorem first_week_tickets_calc : total_tickets - (second_week_tickets + tickets_left) = 38 := by
  sorry

end NUMINAMATH_GPT_first_week_tickets_calc_l461_46110


namespace NUMINAMATH_GPT_all_girls_select_same_color_probability_l461_46147

theorem all_girls_select_same_color_probability :
  let white_marbles := 10
  let black_marbles := 10
  let red_marbles := 10
  let girls := 15
  ∀ (total_marbles : ℕ), total_marbles = white_marbles + black_marbles + red_marbles →
  (white_marbles < girls ∧ black_marbles < girls ∧ red_marbles < girls) →
  0 = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_all_girls_select_same_color_probability_l461_46147


namespace NUMINAMATH_GPT_graph_does_not_pass_second_quadrant_l461_46197

noncomputable def y_function (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) : 
  ∀ x y : ℝ, (y = y_function a b x) → ¬(x < 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_GPT_graph_does_not_pass_second_quadrant_l461_46197


namespace NUMINAMATH_GPT_larger_number_is_23_l461_46194

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end NUMINAMATH_GPT_larger_number_is_23_l461_46194


namespace NUMINAMATH_GPT_seokjin_rank_l461_46125

-- Define the ranks and the people between them as given conditions in the problem
def jimin_rank : Nat := 4
def people_between : Nat := 19

-- The goal is to prove that Seokjin's rank is 24
theorem seokjin_rank : jimin_rank + people_between + 1 = 24 := 
by
  sorry

end NUMINAMATH_GPT_seokjin_rank_l461_46125


namespace NUMINAMATH_GPT_solve_quadratic_equation_l461_46111

theorem solve_quadratic_equation : ∀ x : ℝ, x * (x - 14) = 0 ↔ x = 0 ∨ x = 14 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l461_46111


namespace NUMINAMATH_GPT_metallic_sheet_width_l461_46171

theorem metallic_sheet_width 
  (length_of_cut_square : ℝ) (original_length_of_sheet : ℝ) (volume_of_box : ℝ) (w : ℝ)
  (h1 : length_of_cut_square = 5) 
  (h2 : original_length_of_sheet = 48) 
  (h3 : volume_of_box = 4940) : 
  (38 * (w - 10) * 5 = 4940) → w = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_metallic_sheet_width_l461_46171


namespace NUMINAMATH_GPT_candy_store_revenue_l461_46119

/-- A candy store sold 20 pounds of fudge for $2.50 per pound,
    5 dozen chocolate truffles for $1.50 each, 
    and 3 dozen chocolate-covered pretzels at $2.00 each.
    Prove that the total money made by the candy store is $212.00. --/
theorem candy_store_revenue :
  let fudge_pounds := 20
  let fudge_price_per_pound := 2.50
  let truffle_dozen := 5
  let truffle_price_each := 1.50
  let pretzel_dozen := 3
  let pretzel_price_each := 2.00
  (fudge_pounds * fudge_price_per_pound) + 
  (truffle_dozen * 12 * truffle_price_each) + 
  (pretzel_dozen * 12 * pretzel_price_each) = 212 :=
by
  sorry

end NUMINAMATH_GPT_candy_store_revenue_l461_46119


namespace NUMINAMATH_GPT_jerry_pool_depth_l461_46118

theorem jerry_pool_depth :
  ∀ (total_gallons : ℝ) (gallons_drinking_cooking : ℝ) (gallons_per_shower : ℝ)
    (number_of_showers : ℝ) (pool_length : ℝ) (pool_width : ℝ)
    (gallons_per_cubic_foot : ℝ),
    total_gallons = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    number_of_showers = 15 →
    pool_length = 10 →
    pool_width = 10 →
    gallons_per_cubic_foot = 1 →
    (total_gallons - (gallons_drinking_cooking + gallons_per_shower * number_of_showers)) / 
    (pool_length * pool_width) = 6 := 
by
  intros total_gallons gallons_drinking_cooking gallons_per_shower number_of_showers pool_length pool_width gallons_per_cubic_foot
  intros total_gallons_eq drinking_cooking_eq shower_eq showers_eq length_eq width_eq cubic_foot_eq
  sorry

end NUMINAMATH_GPT_jerry_pool_depth_l461_46118


namespace NUMINAMATH_GPT_magical_stack_card_count_l461_46116

theorem magical_stack_card_count :
  ∃ n, n = 157 + 78 ∧ 2 * n = 470 :=
by
  let n := 235
  use n
  have h1: n = 157 + 78 := by sorry
  have h2: 2 * n = 470 := by sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_magical_stack_card_count_l461_46116


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_verify_options_l461_46167

open Real

-- Option A: Prove the maximum value of x(6-x) given 0 < x < 6 is 9.
theorem option_A (x : ℝ) (h1 : 0 < x) (h2 : x < 6) : 
  ∃ (max_value : ℝ), max_value = 9 ∧ ∀(y : ℝ), 0 < y ∧ y < 6 → y * (6 - y) ≤ max_value :=
sorry

-- Option B: Prove the minimum value of x^2 + 1/(x^2 + 3) for x in ℝ is not -1.
theorem option_B (x : ℝ) : ¬(∃ (min_value : ℝ), min_value = -1 ∧ ∀(y : ℝ), (y ^ 2) + 1 / (y ^ 2 + 3) ≥ min_value) :=
sorry

-- Option C: Prove the maximum value of xy given x + 2y + xy = 6 and x, y > 0 is 2.
theorem option_C (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y + x * y = 6) : 
  ∃ (max_value : ℝ), max_value = 2 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 2 * v + u * v = 6 → u * v ≤ max_value :=
sorry

-- Option D: Prove the minimum value of 2x + y given x + 4y + 4 = xy and x, y > 0 is 17.
theorem option_D (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 4 * y + 4 = x * y) : 
  ∃ (min_value : ℝ), min_value = 17 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 4 * v + 4 = u * v → 2 * u + v ≥ min_value :=
sorry

-- Combine to verify which options are correct
theorem verify_options (A_correct B_correct C_correct D_correct : Prop) :
  A_correct = true ∧ B_correct = false ∧ C_correct = true ∧ D_correct = true :=
sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_verify_options_l461_46167


namespace NUMINAMATH_GPT_sin_cos_15_eq_quarter_l461_46120

theorem sin_cos_15_eq_quarter :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_sin_cos_15_eq_quarter_l461_46120


namespace NUMINAMATH_GPT_circle_line_chord_length_l461_46178

theorem circle_line_chord_length :
  ∀ (k m : ℝ), (∀ x y : ℝ, x^2 + y^2 = 4 → y = k * x + m → ∃ (a : ℝ), a = 2) →
    |m| = Real.sqrt 3 :=
by 
  intros k m h
  sorry

end NUMINAMATH_GPT_circle_line_chord_length_l461_46178


namespace NUMINAMATH_GPT_class_heights_mode_median_l461_46159

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℕ := sorry

theorem class_heights_mode_median 
  (A : List ℕ) -- Heights of students from Class A
  (B : List ℕ) -- Heights of students from Class B
  (hA : A = [170, 170, 169, 171, 171, 171])
  (hB : B = [168, 170, 170, 172, 169, 170]) :
  mode A = 171 ∧ median B = 170 := sorry

end NUMINAMATH_GPT_class_heights_mode_median_l461_46159


namespace NUMINAMATH_GPT_union_A_B_intersection_complementA_B_range_of_a_l461_46192

-- Definition of the universal set U, sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Complement of A in the universal set U
def complement_A : Set ℝ := {x | x < 1 ∨ x ≥ 5}

-- Definition of set C parametrized by a
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Prove that A ∪ B is {x | 1 ≤ x < 8}
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 8} :=
sorry

-- Prove that (complement_U A) ∩ B = {x | 5 ≤ x < 8}
theorem intersection_complementA_B : (complement_A ∩ B) = {x | 5 ≤ x ∧ x < 8} :=
sorry

-- Prove the range of values for a if C ∩ A = C
theorem range_of_a (a : ℝ) : (C a ∩ A = C a) → a ≤ -1 :=
sorry

end NUMINAMATH_GPT_union_A_B_intersection_complementA_B_range_of_a_l461_46192


namespace NUMINAMATH_GPT_value_of_expression_l461_46189

theorem value_of_expression : (0.3 : ℝ)^2 + 0.1 = 0.19 := 
by sorry

end NUMINAMATH_GPT_value_of_expression_l461_46189


namespace NUMINAMATH_GPT_gear_ratios_l461_46108

variable (x y z w : ℝ)
variable (ω_A ω_B ω_C ω_D : ℝ)
variable (k : ℝ)

theorem gear_ratios (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D) : 
    ω_A/ω_B = yzw/xzw ∧ ω_B/ω_C = xzw/xyw ∧ ω_C/ω_D = xyw/xyz ∧ ω_A/ω_C = yzw/xyw := 
sorry

end NUMINAMATH_GPT_gear_ratios_l461_46108


namespace NUMINAMATH_GPT_sum_of_squares_l461_46168

def b1 : ℚ := 10 / 32
def b2 : ℚ := 0
def b3 : ℚ := -5 / 32
def b4 : ℚ := 0
def b5 : ℚ := 1 / 32

theorem sum_of_squares : b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 512 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l461_46168


namespace NUMINAMATH_GPT_find_n_l461_46142

theorem find_n (x : ℝ) (hx : x > 0) (h : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_n_l461_46142


namespace NUMINAMATH_GPT_min_value_2a_3b_6c_l461_46172

theorem min_value_2a_3b_6c (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (habc : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 :=
sorry

end NUMINAMATH_GPT_min_value_2a_3b_6c_l461_46172


namespace NUMINAMATH_GPT_quadratic_unique_solution_l461_46164

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end NUMINAMATH_GPT_quadratic_unique_solution_l461_46164


namespace NUMINAMATH_GPT_quadratic_rewrite_constants_l461_46173

theorem quadratic_rewrite_constants (a b c : ℤ) 
    (h1 : -4 * (x - 2) ^ 2 + 144 = -4 * x ^ 2 + 16 * x + 128) 
    (h2 : a = -4)
    (h3 : b = -2)
    (h4 : c = 144) 
    : a + b + c = 138 := by
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_constants_l461_46173


namespace NUMINAMATH_GPT_one_number_greater_than_one_l461_46170

theorem one_number_greater_than_one 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > (1 / a) + (1 / b) + (1 / c)) 
  : (1 < a ∨ 1 < b ∨ 1 < c) ∧ ¬(1 < a ∧ 1 < b ∧ 1 < c) :=
by
  sorry

end NUMINAMATH_GPT_one_number_greater_than_one_l461_46170


namespace NUMINAMATH_GPT_sin_double_angle_l461_46184

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 3 / 4) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l461_46184


namespace NUMINAMATH_GPT_upper_bound_neg_expr_l461_46151

theorem upper_bound_neg_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  - (1 / (2 * a) + 2 / b) ≤ - (9 / 2) := 
sorry

end NUMINAMATH_GPT_upper_bound_neg_expr_l461_46151


namespace NUMINAMATH_GPT_determine_value_of_x_l461_46114

theorem determine_value_of_x {b x : ℝ} (hb : 1 < b) (hx : 0 < x) 
  (h_eq : (4 * x)^(Real.logb b 2) = (5 * x)^(Real.logb b 3)) : 
  x = (4 / 5)^(Real.logb (3 / 2) b) :=
by
  sorry

end NUMINAMATH_GPT_determine_value_of_x_l461_46114


namespace NUMINAMATH_GPT_ball_bounce_height_l461_46160

theorem ball_bounce_height :
  ∃ k : ℕ, 800 * (1 / 2 : ℝ)^k < 2 ∧ k ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_ball_bounce_height_l461_46160


namespace NUMINAMATH_GPT_tangent_slope_at_one_one_l461_46112

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem tangent_slope_at_one_one : (deriv curve 1) = 2 := 
sorry

end NUMINAMATH_GPT_tangent_slope_at_one_one_l461_46112


namespace NUMINAMATH_GPT_constant_term_l461_46131

theorem constant_term (n : ℕ) (h : (Nat.choose n 4 * 2^4) / (Nat.choose n 2 * 2^2) = (56 / 3)) :
  (∃ k : ℕ, k = 2 ∧ n = 10 ∧ Nat.choose 10 k * 2^k = 180) := by
  sorry

end NUMINAMATH_GPT_constant_term_l461_46131


namespace NUMINAMATH_GPT_real_root_exists_l461_46193

theorem real_root_exists (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 :=
sorry

end NUMINAMATH_GPT_real_root_exists_l461_46193


namespace NUMINAMATH_GPT_determine_digits_l461_46109

def product_eq_digits (A B C D x : ℕ) : Prop :=
  x * (x + 1) = 1000 * A + 100 * B + 10 * C + D

def product_minus_3_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D

def product_minus_30_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D

theorem determine_digits :
  ∃ (A B C D x : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  product_eq_digits A B C D x ∧
  product_minus_3_eq_digits A B C D x ∧
  product_minus_30_eq_digits A B C D x ∧
  A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_digits_l461_46109


namespace NUMINAMATH_GPT_work_completion_l461_46185

theorem work_completion (original_men planned_days absent_men remaining_men completion_days : ℕ) :
  original_men = 180 → 
  planned_days = 55 →
  absent_men = 15 →
  remaining_men = original_men - absent_men →
  remaining_men * completion_days = original_men * planned_days →
  completion_days = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_work_completion_l461_46185


namespace NUMINAMATH_GPT_laura_pants_count_l461_46138

def cost_of_pants : ℕ := 54
def cost_of_shirt : ℕ := 33
def number_of_shirts : ℕ := 4
def total_money_given : ℕ := 250
def change_received : ℕ := 10

def laura_spent : ℕ := total_money_given - change_received
def total_cost_shirts : ℕ := cost_of_shirt * number_of_shirts
def spent_on_pants : ℕ := laura_spent - total_cost_shirts
def pairs_of_pants_bought : ℕ := spent_on_pants / cost_of_pants

theorem laura_pants_count : pairs_of_pants_bought = 2 :=
by
  sorry

end NUMINAMATH_GPT_laura_pants_count_l461_46138


namespace NUMINAMATH_GPT_nutty_professor_mixture_weight_l461_46129

/-- The Nutty Professor's problem translated to Lean 4 -/
theorem nutty_professor_mixture_weight :
  let cashews_weight := 20
  let cashews_cost_per_pound := 6.75
  let brazil_nuts_cost_per_pound := 5.00
  let mixture_cost_per_pound := 5.70
  ∃ (brazil_nuts_weight : ℝ), cashews_weight * cashews_cost_per_pound + brazil_nuts_weight * brazil_nuts_cost_per_pound =
                             (cashews_weight + brazil_nuts_weight) * mixture_cost_per_pound ∧
                             (cashews_weight + brazil_nuts_weight = 50) := 
sorry

end NUMINAMATH_GPT_nutty_professor_mixture_weight_l461_46129


namespace NUMINAMATH_GPT_vertex_closest_point_l461_46186

theorem vertex_closest_point (a : ℝ) (x y : ℝ) :
  (x^2 = 2 * y) ∧ (y ≥ 0) ∧ ((y^2 + 2 * (1 - a) * y + a^2) ≤ 0) → a ≤ 1 :=
by 
  sorry

end NUMINAMATH_GPT_vertex_closest_point_l461_46186


namespace NUMINAMATH_GPT_cost_to_paint_cube_l461_46124

theorem cost_to_paint_cube :
  let cost_per_kg := 50
  let coverage_per_kg := 20
  let side_length := 20
  let surface_area := 6 * (side_length * side_length)
  let amount_of_paint := surface_area / coverage_per_kg
  let total_cost := amount_of_paint * cost_per_kg
  total_cost = 6000 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_paint_cube_l461_46124


namespace NUMINAMATH_GPT_goods_train_speed_l461_46195

theorem goods_train_speed (length_train length_platform distance time : ℕ) (conversion_factor : ℚ) : 
  length_train = 250 → 
  length_platform = 270 → 
  distance = length_train + length_platform → 
  time = 26 → 
  conversion_factor = 3.6 →
  (distance / time : ℚ) * conversion_factor = 72 :=
by
  intros h_lt h_lp h_d h_t h_cf
  rw [h_lt, h_lp] at h_d
  rw [h_t, h_cf]
  sorry

end NUMINAMATH_GPT_goods_train_speed_l461_46195


namespace NUMINAMATH_GPT_fraction_value_l461_46190

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l461_46190


namespace NUMINAMATH_GPT_count_whole_numbers_between_cuberoots_l461_46183

theorem count_whole_numbers_between_cuberoots : 
  ∃ (n : ℕ), n = 7 ∧ 
      ∀ x : ℝ, (3 < x ∧ x < 4 → ∃ k : ℕ, k = 4) ∧ 
                (9 < x ∧ x ≤ 10 → ∃ k : ℕ, k = 10) :=
sorry

end NUMINAMATH_GPT_count_whole_numbers_between_cuberoots_l461_46183


namespace NUMINAMATH_GPT_units_digit_of_150_factorial_is_zero_l461_46146

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end NUMINAMATH_GPT_units_digit_of_150_factorial_is_zero_l461_46146


namespace NUMINAMATH_GPT_work_days_l461_46176

theorem work_days (m r d : ℕ) (h : 2 * m * d = 2 * (m + r) * (md / (m + r))) : d = md / (m + r) :=
by
  sorry

end NUMINAMATH_GPT_work_days_l461_46176


namespace NUMINAMATH_GPT_min_packs_to_buy_120_cans_l461_46199

/-- Prove that the minimum number of packs needed to buy exactly 120 cans of soda,
with packs available in sizes of 8, 15, and 30 cans, is 4. -/
theorem min_packs_to_buy_120_cans : 
  ∃ n, n = 4 ∧ ∀ x y z: ℕ, 8 * x + 15 * y + 30 * z = 120 → x + y + z ≥ n :=
sorry

end NUMINAMATH_GPT_min_packs_to_buy_120_cans_l461_46199


namespace NUMINAMATH_GPT_find_f_neg4_l461_46154

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg4 (a b : ℝ) (h : f a b 4 = 0) : f a b (-4) = 2 := by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_f_neg4_l461_46154


namespace NUMINAMATH_GPT_direct_proportion_function_l461_46103

-- Definitions of the functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 3 * x^2 + 2

-- Definition of a direct proportion function
def is_direct_proportion (f : ℝ → ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, f x = k * x)

-- Theorem statement
theorem direct_proportion_function : is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_function_l461_46103


namespace NUMINAMATH_GPT_bags_filled_l461_46157

def bags_filled_on_certain_day (x : ℕ) : Prop :=
  let bags := x + 3
  let total_cans := 8 * bags
  total_cans = 72

theorem bags_filled {x : ℕ} (h : bags_filled_on_certain_day x) : x = 6 :=
  sorry

end NUMINAMATH_GPT_bags_filled_l461_46157


namespace NUMINAMATH_GPT_projectile_height_time_l461_46155

-- Define constants and the height function
def a : ℝ := -4.9
def b : ℝ := 29.75
def c : ℝ := -35
def y (t : ℝ) : ℝ := a * t^2 + b * t

-- Problem statement
theorem projectile_height_time (h : y t = 35) : ∃ t : ℝ, 0 < t ∧ abs (t - 1.598) < 0.001 := by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_projectile_height_time_l461_46155


namespace NUMINAMATH_GPT_certain_number_is_14_l461_46179

theorem certain_number_is_14 
  (a b n : ℕ) 
  (h1 : ∃ k1, a = k1 * n) 
  (h2 : ∃ k2, b = k2 * n) 
  (h3 : b = a + 11 * n) 
  (h4 : b = a + 22 * 7) : n = 14 := 
by 
  sorry

end NUMINAMATH_GPT_certain_number_is_14_l461_46179


namespace NUMINAMATH_GPT_find_triangle_angles_l461_46181

theorem find_triangle_angles (a b h_a h_b : ℝ) (A B C : ℝ) :
  a ≤ h_a → b ≤ h_b →
  h_a ≤ b → h_b ≤ a →
  ∃ x y z : ℝ, (x = 90 ∧ y = 45 ∧ z = 45) ∧ 
  (x + y + z = 180) :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_angles_l461_46181


namespace NUMINAMATH_GPT_ratio_area_perimeter_eq_sqrt3_l461_46101

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_area_perimeter_eq_sqrt3_l461_46101


namespace NUMINAMATH_GPT_fill_buckets_lcm_l461_46149

theorem fill_buckets_lcm :
  (∀ (A B C : ℕ), (2 / 3 : ℚ) * A = 90 ∧ (1 / 2 : ℚ) * B = 120 ∧ (3 / 4 : ℚ) * C = 150 → lcm A (lcm B C) = 1200) :=
by
  sorry

end NUMINAMATH_GPT_fill_buckets_lcm_l461_46149


namespace NUMINAMATH_GPT_final_sale_price_l461_46162

def initial_price : ℝ := 450
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def third_discount : ℝ := 0.05

def price_after_first_discount (initial : ℝ) (discount : ℝ) : ℝ :=
  initial * (1 - discount)
  
def price_after_second_discount (price_first : ℝ) (discount : ℝ) : ℝ :=
  price_first * (1 - discount)
  
def price_after_third_discount (price_second : ℝ) (discount : ℝ) : ℝ :=
  price_second * (1 - discount)

theorem final_sale_price :
  price_after_third_discount
    (price_after_second_discount
      (price_after_first_discount initial_price first_discount)
      second_discount)
    third_discount = 288.5625 := 
sorry

end NUMINAMATH_GPT_final_sale_price_l461_46162
