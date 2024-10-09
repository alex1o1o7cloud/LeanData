import Mathlib

namespace ryan_spanish_hours_l281_28141

theorem ryan_spanish_hours (S : ℕ) (h : 7 = S + 3) : S = 4 :=
sorry

end ryan_spanish_hours_l281_28141


namespace min_equal_area_triangles_l281_28139

theorem min_equal_area_triangles (chessboard_area missing_area : ℕ) (total_area : ℕ := chessboard_area - missing_area) 
(H1 : chessboard_area = 64) (H2 : missing_area = 1) : 
∃ n : ℕ, n = 18 ∧ (total_area = 63) → total_area / ((7:ℕ)/2) = n := 
sorry

end min_equal_area_triangles_l281_28139


namespace math_proof_problem_l281_28143
noncomputable def expr : ℤ := 3000 * (3000 ^ 3000) + 3000 ^ 2

theorem math_proof_problem : expr = 3000 ^ 3001 + 9000000 :=
by
  -- Proof
  sorry

end math_proof_problem_l281_28143


namespace candles_must_be_odd_l281_28122

theorem candles_must_be_odd (n k : ℕ) (h : n * k = (n * (n + 1)) / 2) : n % 2 = 1 :=
by
  -- Given that the total burn time for all n candles = k * n
  -- And the sum of the first n natural numbers = (n * (n + 1)) / 2
  -- We have the hypothesis h: n * k = (n * (n + 1)) / 2
  -- We need to prove that n must be odd
  sorry

end candles_must_be_odd_l281_28122


namespace evaluate_expression_l281_28194

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12)) ^ 2 = 1600 := by
  sorry

end evaluate_expression_l281_28194


namespace intersection_M_N_l281_28187

-- Define the universe U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set M based on the condition x^2 <= x
def M : Set ℤ := {x ∈ U | x^2 ≤ x}

-- Define the set N based on the condition x^3 - 3x^2 + 2x = 0
def N : Set ℤ := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

-- State the theorem to be proven
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l281_28187


namespace range_of_a_l281_28193

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > a then x + 2 else x^2 + 5 * x + 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
f x a - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) ↔ (-1 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l281_28193


namespace charlie_certain_instrument_l281_28158

theorem charlie_certain_instrument :
  ∃ (x : ℕ), (1 + 2 + x) + (2 + 1 + 0) = 7 → x = 1 :=
by
  sorry

end charlie_certain_instrument_l281_28158


namespace houses_before_boom_l281_28107

theorem houses_before_boom (T B H : ℕ) (hT : T = 2000) (hB : B = 574) : H = 1426 := by
  sorry

end houses_before_boom_l281_28107


namespace find_n_l281_28130

theorem find_n (n : ℕ) : (10^n = (10^5)^3) → n = 15 :=
by sorry

end find_n_l281_28130


namespace range_of_a_l281_28134

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 3) ∧ (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≥ 2) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 2) ↔ 1 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l281_28134


namespace matrix_multiplication_correct_l281_28199

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 6]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![17, -3], ![16, -24]]

theorem matrix_multiplication_correct : A * B = C := by 
  sorry

end matrix_multiplication_correct_l281_28199


namespace interest_after_5_years_l281_28153

noncomputable def initial_amount : ℝ := 2000
noncomputable def interest_rate : ℝ := 0.08
noncomputable def duration : ℕ := 5
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ duration
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem interest_after_5_years : interest_earned = 938.66 := by
  sorry

end interest_after_5_years_l281_28153


namespace arithmetic_sequence_a6_l281_28151

-- Definitions representing the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n (n + m) = a_n n + a_n m + n

def sum_of_first_n_terms (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = (n / 2) * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

theorem arithmetic_sequence_a6 (S : ℕ → ℝ) (a_n : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_first_n_terms S a_n)
  (h_cond : S 9 - S 2 = 35) : 
  a_n 6 = 5 :=
by
  sorry

end arithmetic_sequence_a6_l281_28151


namespace problem_1_and_2_problem_1_infinite_solutions_l281_28156

open Nat

theorem problem_1_and_2 (k : ℕ) (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2 = k * a * b * c) →
  (k = 1 ∨ k = 3) :=
sorry

theorem problem_1_infinite_solutions (k : ℕ) (h_k : k = 1 ∨ k = 3) :
  ∃ (a_n b_n c_n : ℕ) (n : ℕ), 
  a_n > 0 ∧ b_n > 0 ∧ c_n > 0 ∧
  (a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n) ∧
  ∀ x y : ℕ, (x = a_n ∧ y = b_n) ∨ (x = a_n ∧ y = c_n) ∨ (x = b_n ∧ y = c_n) →
    ∃ p q : ℕ, x * y = p^2 + q^2 :=
sorry

end problem_1_and_2_problem_1_infinite_solutions_l281_28156


namespace desired_butterfat_percentage_l281_28190

theorem desired_butterfat_percentage (milk1 milk2 : ℝ) (butterfat1 butterfat2 : ℝ) :
  milk1 = 8 →
  butterfat1 = 0.10 →
  milk2 = 8 →
  butterfat2 = 0.30 →
  ((butterfat1 * milk1) + (butterfat2 * milk2)) / (milk1 + milk2) * 100 = 20 := 
by
  intros
  sorry

end desired_butterfat_percentage_l281_28190


namespace perfect_square_trinomial_solution_l281_28166

theorem perfect_square_trinomial_solution (m : ℝ) :
  (∃ a : ℝ, (∀ x : ℝ, x^2 - 2*(m+3)*x + 9 = (x - a)^2))
  → m = 0 ∨ m = -6 :=
by
  sorry

end perfect_square_trinomial_solution_l281_28166


namespace min_value_fraction_l281_28182

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end min_value_fraction_l281_28182


namespace math_problem_l281_28110

theorem math_problem (m : ℝ) (h : m^2 - m = 2) : (m - 1)^2 + (m + 2) * (m - 2) = 1 := 
by sorry

end math_problem_l281_28110


namespace piecewise_function_continuity_l281_28124

theorem piecewise_function_continuity :
  (∃ a c : ℝ, (2 * a * 2 + 4 = 2^2 - 2) ∧ (4 - 2 = 3 * (-2) - c) ∧ a + c = -17 / 2) :=
by
  sorry

end piecewise_function_continuity_l281_28124


namespace union_sets_l281_28168

def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 3 } :=
sorry

end union_sets_l281_28168


namespace table_capacity_l281_28152

theorem table_capacity :
  ∀ (n_invited no_show tables : ℕ), n_invited = 47 → no_show = 7 → tables = 8 → 
  (n_invited - no_show) / tables = 5 := by
  intros n_invited no_show tables h_invited h_no_show h_tables
  sorry

end table_capacity_l281_28152


namespace find_base_b_l281_28161

theorem find_base_b (b : ℕ) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 6) = (4 * b^2 + 1 * b + 1) →
  7 < b →
  b = 10 :=
by
  intro h₁ h₂
  sorry

end find_base_b_l281_28161


namespace ellipse_properties_l281_28105

theorem ellipse_properties 
  (foci1 foci2 : ℝ × ℝ) 
  (point_on_ellipse : ℝ × ℝ) 
  (h k a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (ellipse_condition : foci1 = (-4, 1) ∧ foci2 = (-4, 5) ∧ point_on_ellipse = (1, 3))
  (ellipse_eqn : (x y : ℝ) → ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) :
  a + k = 8 :=
by
  sorry

end ellipse_properties_l281_28105


namespace bucket_full_weight_l281_28144

variable {a b x y : ℝ}

theorem bucket_full_weight (h1 : x + 2/3 * y = a) (h2 : x + 1/2 * y = b) : 
  (x + y) = 3 * a - 2 * b := 
sorry

end bucket_full_weight_l281_28144


namespace height_cylinder_l281_28119

variables (r_c h_c r_cy h_cy : ℝ)
variables (V_cone V_cylinder : ℝ)
variables (r_c_val : r_c = 15)
variables (h_c_val : h_c = 20)
variables (r_cy_val : r_cy = 30)
variables (V_cone_eq : V_cone = (1/3) * π * r_c^2 * h_c)
variables (V_cylinder_eq : V_cylinder = π * r_cy^2 * h_cy)

theorem height_cylinder : h_cy = 1.67 :=
by
  rw [r_c_val, h_c_val, r_cy_val] at *
  have V_cone := V_cone_eq
  have V_cylinder := V_cylinder_eq
  sorry

end height_cylinder_l281_28119


namespace simplify_expression_l281_28112

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l281_28112


namespace joyce_apples_l281_28185

theorem joyce_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 75 → 
    given_apples = 52 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 23 :=
by 
  intros initial_apples given_apples remaining_apples h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end joyce_apples_l281_28185


namespace find_angle_A_find_value_of_c_l281_28108

variable {a b c A B C : ℝ}

-- Define the specific conditions as Lean 'variables' and 'axioms'
-- Condition: In triangle ABC, the sides opposite to angles A, B and C are a, b, and c respectively.
axiom triangle_ABC_sides : b = 2 * (a * Real.cos B - c)

-- Part (1): Prove the value of angle A
theorem find_angle_A (h : b = 2 * (a * Real.cos B - c)) : A = (2 * Real.pi) / 3 :=
by
  sorry

-- Condition: a * cos C = sqrt 3 and b = 1
axiom cos_C_value : a * Real.cos C = Real.sqrt 3
axiom b_value : b = 1

-- Part (2): Prove the value of c
theorem find_value_of_c (h1 : a * Real.cos C = Real.sqrt 3) (h2 : b = 1) : c = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end find_angle_A_find_value_of_c_l281_28108


namespace digit_100th_is_4_digit_1000th_is_3_l281_28167

noncomputable section

def digit_100th_place : Nat :=
  4

def digit_1000th_place : Nat :=
  3

theorem digit_100th_is_4 (n : ℕ) (h1 : n ∈ {m | m = 100}) : digit_100th_place = 4 := by
  sorry

theorem digit_1000th_is_3 (n : ℕ) (h1 : n ∈ {m | m = 1000}) : digit_1000th_place = 3 := by
  sorry

end digit_100th_is_4_digit_1000th_is_3_l281_28167


namespace largest_fraction_proof_l281_28101

theorem largest_fraction_proof 
  (w x y z : ℕ)
  (hw : 0 < w)
  (hx : w < x)
  (hy : x < y)
  (hz : y < z)
  (w_eq : w = 1)
  (x_eq : x = y - 1)
  (z_eq : z = y + 1)
  (y_eq : y = x!) : 
  (max (max (w + z) (w + x)) (max (x + z) (max (x + y) (y + z))) = 5 / 3) := 
sorry

end largest_fraction_proof_l281_28101


namespace cube_paint_same_color_l281_28180

theorem cube_paint_same_color (colors : Fin 6) : ∃ ways : ℕ, ways = 6 :=
sorry

end cube_paint_same_color_l281_28180


namespace rearrange_digits_2552_l281_28160

theorem rearrange_digits_2552 : 
    let digits := [2, 5, 5, 2]
    let factorial := fun n => Nat.factorial n
    let permutations := (factorial 4) / (factorial 2 * factorial 2)
    permutations = 6 :=
by
  sorry

end rearrange_digits_2552_l281_28160


namespace exists_strictly_increasing_sequence_l281_28198

open Nat

-- Definition of strictly increasing sequence of integers a
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

-- Condition i): Every natural number can be written as the sum of two terms from the sequence
def condition_i (a : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j

-- Condition ii): For each positive integer n, a_n > n^2/16
def condition_ii (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > n^2 / 16

-- The main theorem stating the existence of such a sequence
theorem exists_strictly_increasing_sequence :
  ∃ a : ℕ → ℕ, a 0 = 0 ∧ strictly_increasing_sequence a ∧ condition_i a ∧ condition_ii a :=
sorry

end exists_strictly_increasing_sequence_l281_28198


namespace calculate_cubic_sum_roots_l281_28195

noncomputable def α := (27 : ℝ)^(1/3)
noncomputable def β := (64 : ℝ)^(1/3)
noncomputable def γ := (125 : ℝ)^(1/3)

theorem calculate_cubic_sum_roots (u v w : ℝ) :
  (u - α) * (u - β) * (u - γ) = 1/2 ∧
  (v - α) * (v - β) * (v - γ) = 1/2 ∧
  (w - α) * (w - β) * (w - γ) = 1/2 →
  u^3 + v^3 + w^3 = 217.5 :=
by
  sorry

end calculate_cubic_sum_roots_l281_28195


namespace car_miles_per_gallon_l281_28154

-- Define the conditions
def distance_home : ℕ := 220
def additional_distance : ℕ := 100
def total_distance : ℕ := distance_home + additional_distance
def tank_capacity : ℕ := 16 -- in gallons
def miles_per_gallon : ℕ := total_distance / tank_capacity

-- State the goal
theorem car_miles_per_gallon : miles_per_gallon = 20 := by
  sorry

end car_miles_per_gallon_l281_28154


namespace unit_price_ratio_l281_28142

theorem unit_price_ratio (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (1.1 * p / (1.4 * v)) / (0.85 * p / (1.3 * v)) = 13 / 11 :=
by
  sorry

end unit_price_ratio_l281_28142


namespace smallest_positive_period_max_min_value_interval_l281_28155

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin (x + Real.pi / 3))^2 - (Real.cos x)^2 + (Real.sin x)^2

theorem smallest_positive_period : (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

theorem max_min_value_interval :
  (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), 
    f x ≤ 3 / 2 ∧ f x ≥ 0 ∧ 
    (f (-Real.pi / 6) = 0) ∧ 
    (f (Real.pi / 6) = 3 / 2)) :=
by sorry

end smallest_positive_period_max_min_value_interval_l281_28155


namespace correct_operation_l281_28162

theorem correct_operation :
  (∀ (a : ℤ), 3 * a + 2 * a ≠ 5 * a ^ 2) ∧
  (∀ (a : ℤ), a ^ 6 / a ^ 2 ≠ a ^ 3) ∧
  (∀ (a : ℤ), (-3 * a ^ 3) ^ 2 = 9 * a ^ 6) ∧
  (∀ (a : ℤ), (a + 2) ^ 2 ≠ a ^ 2 + 4) := 
by
  sorry

end correct_operation_l281_28162


namespace how_many_leaves_l281_28183

def ladybugs_per_leaf : ℕ := 139
def total_ladybugs : ℕ := 11676

theorem how_many_leaves : total_ladybugs / ladybugs_per_leaf = 84 :=
by
  sorry

end how_many_leaves_l281_28183


namespace last_box_weight_l281_28115

theorem last_box_weight (a b c : ℕ) (h1 : a = 2) (h2 : b = 11) (h3 : a + b + c = 18) : c = 5 :=
by
  sorry

end last_box_weight_l281_28115


namespace tan_45_eq_one_l281_28102

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l281_28102


namespace percentage_x_of_yz_l281_28140

theorem percentage_x_of_yz (x y z w : ℝ) (h1 : x = 0.07 * y) (h2 : y = 0.35 * z) (h3 : z = 0.60 * w) :
  (x / (y + z) * 100) = 1.8148 :=
by
  sorry

end percentage_x_of_yz_l281_28140


namespace relation_among_a_b_c_l281_28129

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relation_among_a_b_c : c > a ∧ a > b :=
by
  -- Prove that c > a and a > b
  sorry

end relation_among_a_b_c_l281_28129


namespace pieces_in_each_package_l281_28179

-- Definitions from conditions
def num_packages : ℕ := 5
def extra_pieces : ℕ := 6
def total_pieces : ℕ := 41

-- Statement to prove
theorem pieces_in_each_package : ∃ x : ℕ, num_packages * x + extra_pieces = total_pieces ∧ x = 7 :=
by
  -- Begin the proof with the given setup
  sorry

end pieces_in_each_package_l281_28179


namespace sin_pi_div_three_l281_28148

theorem sin_pi_div_three : Real.sin (π / 3) = Real.sqrt 3 / 2 := 
sorry

end sin_pi_div_three_l281_28148


namespace Tim_carrots_count_l281_28149

theorem Tim_carrots_count (initial_potatoes new_potatoes initial_carrots final_potatoes final_carrots : ℕ) 
  (h_ratio : 3 * final_potatoes = 4 * final_carrots)
  (h_initial_potatoes : initial_potatoes = 32)
  (h_new_potatoes : new_potatoes = 28)
  (h_final_potatoes : final_potatoes = initial_potatoes + new_potatoes)
  (h_initial_ratio : 3 * 32 = 4 * initial_carrots) : 
  final_carrots = 45 :=
by {
  sorry
}

end Tim_carrots_count_l281_28149


namespace smallest_of_three_integers_l281_28175

theorem smallest_of_three_integers (a b c : ℤ) (h1 : a * b * c = 32) (h2 : a + b + c = 3) : min (min a b) c = -4 := 
sorry

end smallest_of_three_integers_l281_28175


namespace total_handshakes_l281_28126

def gremlins := 30
def pixies := 12
def unfriendly_gremlins := 15
def friendly_gremlins := 15

def handshake_count : Nat :=
  let handshakes_friendly_gremlins := friendly_gremlins * (friendly_gremlins - 1) / 2
  let handshakes_friendly_unfriendly := friendly_gremlins * unfriendly_gremlins
  let handshakes_gremlins_pixies := gremlins * pixies
  handshakes_friendly_gremlins + handshakes_friendly_unfriendly + handshakes_gremlins_pixies

theorem total_handshakes : handshake_count = 690 := by
  sorry

end total_handshakes_l281_28126


namespace y_intercept_3x_minus_4y_eq_12_l281_28146

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l281_28146


namespace height_of_isosceles_triangle_l281_28170

variable (s : ℝ) (h : ℝ) (A : ℝ)
variable (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
variable (rectangle : ∀ (s : ℝ), A = s^2)

theorem height_of_isosceles_triangle (s : ℝ) (h : ℝ) (A : ℝ) (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
  (rectangle : ∀ (s : ℝ), A = s^2) : h = s := by
  sorry

end height_of_isosceles_triangle_l281_28170


namespace price_of_ice_cream_l281_28174

theorem price_of_ice_cream (x : ℝ) :
  (225 * x + 125 * 0.52 = 200) → (x = 0.60) :=
sorry

end price_of_ice_cream_l281_28174


namespace fourth_power_sum_l281_28189

variable (a b c : ℝ)

theorem fourth_power_sum (h1 : a + b + c = 2) 
                         (h2 : a^2 + b^2 + c^2 = 3) 
                         (h3 : a^3 + b^3 + c^3 = 4) : 
                         a^4 + b^4 + c^4 = 41 / 6 := 
by 
  sorry

end fourth_power_sum_l281_28189


namespace find_a_l281_28169

open Real

-- Definition of regression line
def regression_line (x : ℝ) : ℝ := 12.6 * x + 0.6

-- Data points for x and y
def x_values : List ℝ := [2, 3, 3.5, 4.5, 7]
def y_values : List ℝ := [26, 38, 43, 60]

-- Proof statement
theorem find_a (a : ℝ) (hx : x_values = [2, 3, 3.5, 4.5, 7])
  (hy : y_values ++ [a] = [26, 38, 43, 60, a]) : a = 88 :=
  sorry

end find_a_l281_28169


namespace total_population_calculation_l281_28100

theorem total_population_calculation :
  ∀ (total_lions total_leopards adult_lions adult_leopards : ℕ)
  (female_lions male_lions female_leopards male_leopards : ℕ)
  (adult_elephants baby_elephants total_elephants total_zebras : ℕ),
  total_lions = 200 →
  total_lions = 2 * total_leopards →
  adult_lions = 3 * total_lions / 4 →
  adult_leopards = 3 * total_leopards / 5 →
  female_lions = 3 * total_lions / 5 →
  male_lions = 2 * total_lions / 5 →
  female_leopards = 2 * total_leopards / 3 →
  male_leopards = total_leopards / 3 →
  adult_elephants = (adult_lions + adult_leopards) / 2 →
  baby_elephants = 100 →
  total_elephants = adult_elephants + baby_elephants →
  total_zebras = adult_elephants + total_leopards →
  total_lions + total_leopards + total_elephants + total_zebras = 710 :=
by sorry

end total_population_calculation_l281_28100


namespace function_equation_l281_28178

noncomputable def f (n : ℕ) : ℕ := sorry

theorem function_equation (h : ∀ m n : ℕ, m > 0 → n > 0 →
  f (f (f m) + 2 * f (f n)) = m^2 + 2 * n^2) : 
  ∀ n : ℕ, n > 0 → f n = n := 
sorry

end function_equation_l281_28178


namespace triangle_angle_problem_l281_28128

open Real

-- Define degrees to radians conversion (if necessary)
noncomputable def degrees (d : ℝ) : ℝ := d * π / 180

-- Define the problem conditions and goal
theorem triangle_angle_problem
  (x y : ℝ)
  (h1 : degrees 3 * x + degrees y = degrees 90) :
  x = 18 ∧ y = 36 := by
  sorry

end triangle_angle_problem_l281_28128


namespace noodles_initial_l281_28150

-- Definitions of our conditions
def given_away : ℝ := 12.0
def noodles_left : ℝ := 42.0
def initial_noodles : ℝ := 54.0

-- Theorem statement
theorem noodles_initial (a b : ℝ) (x : ℝ) (h₁ : a = 12.0) (h₂ : b = 42.0) (h₃ : x = a + b) : x = initial_noodles :=
by
  -- Placeholder for the proof
  sorry

end noodles_initial_l281_28150


namespace solve_for_x_l281_28113

theorem solve_for_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end solve_for_x_l281_28113


namespace f_2010_plus_f_2011_l281_28138

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions in Lean 4
variables (f : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_1 : f 1 = 2

-- The theorem to be proved
theorem f_2010_plus_f_2011 : f (2010) + f (2011) = -2 :=
by
  sorry

end f_2010_plus_f_2011_l281_28138


namespace sin_double_angle_l281_28121

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 :=
by sorry

end sin_double_angle_l281_28121


namespace smallest_xyz_sum_l281_28181

theorem smallest_xyz_sum (x y z : ℕ) (h1 : (x + y) * (y + z) = 2016) (h2 : (x + y) * (z + x) = 1080) :
  x > 0 → y > 0 → z > 0 → x + y + z = 61 :=
  sorry

end smallest_xyz_sum_l281_28181


namespace book_pages_l281_28114

-- Define the number of pages Sally reads on weekdays and weekends
def pages_on_weekdays : ℕ := 10
def pages_on_weekends : ℕ := 20

-- Define the number of weekdays and weekends in 2 weeks
def weekdays_in_two_weeks : ℕ := 5 * 2
def weekends_in_two_weeks : ℕ := 2 * 2

-- Total number of pages read in 2 weeks
def total_pages_read (pages_on_weekdays : ℕ) (pages_on_weekends : ℕ) (weekdays_in_two_weeks : ℕ) (weekends_in_two_weeks : ℕ) : ℕ :=
  (pages_on_weekdays * weekdays_in_two_weeks) + (pages_on_weekends * weekends_in_two_weeks)

-- Prove the number of pages in the book
theorem book_pages : total_pages_read 10 20 10 4 = 180 := by
  sorry

end book_pages_l281_28114


namespace total_animal_legs_l281_28196

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end total_animal_legs_l281_28196


namespace largest_positive_integer_l281_28131

def binary_operation (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer (n : Int) : (∀ m : Int, m > 0 → n - (n * 5) < -19 → m ≤ n) 
  ↔ n = 5 := 
by
  sorry

end largest_positive_integer_l281_28131


namespace number_of_correct_statements_l281_28117

-- Define the statements
def statement_1 : Prop := ∀ (a : ℚ), |a| < |0| → a = 0
def statement_2 : Prop := ∃ (b : ℚ), ∀ (c : ℚ), b < 0 ∧ b ≥ c → c = b
def statement_3 : Prop := -4^6 = (-4) * (-4) * (-4) * (-4) * (-4) * (-4)
def statement_4 : Prop := ∀ (a b : ℚ), a + b = 0 → a ≠ 0 → b ≠ 0 → (a / b = -1)
def statement_5 : Prop := ∀ (c : ℚ), (0 / c = 0 ↔ c ≠ 0)

-- Define the overall proof problem
theorem number_of_correct_statements : (statement_1 ∧ statement_4) ∧ ¬(statement_2 ∨ statement_3 ∨ statement_5) :=
by
  sorry

end number_of_correct_statements_l281_28117


namespace workers_in_first_group_l281_28125

theorem workers_in_first_group
  (W D : ℕ)
  (h1 : 6 * W * D = 9450)
  (h2 : 95 * D = 9975) :
  W = 15 := 
sorry

end workers_in_first_group_l281_28125


namespace solve_inequality_l281_28159

theorem solve_inequality {x : ℝ} : (x^2 - 5 * x + 6 ≤ 0) → (2 ≤ x ∧ x ≤ 3) :=
by
  intro h
  sorry

end solve_inequality_l281_28159


namespace total_handshakes_l281_28186

-- Define the groups and their properties
def GroupA := 30
def GroupB := 15
def GroupC := 5
def KnowEachOtherA := true -- All 30 people in Group A know each other
def KnowFromB := 10 -- Each person in Group B knows 10 people from Group A
def KnowNoOneC := true -- Each person in Group C knows no one

-- Define the number of handshakes based on the conditions
def handshakes_between_A_and_B : Nat := GroupB * (GroupA - KnowFromB)
def handshakes_between_B_and_C : Nat := GroupB * GroupC
def handshakes_within_C : Nat := (GroupC * (GroupC - 1)) / 2
def handshakes_between_A_and_C : Nat := GroupA * GroupC

-- Prove the total number of handshakes
theorem total_handshakes : 
  handshakes_between_A_and_B +
  handshakes_between_B_and_C +
  handshakes_within_C +
  handshakes_between_A_and_C = 535 :=
by sorry

end total_handshakes_l281_28186


namespace steps_to_11th_floor_l281_28165

theorem steps_to_11th_floor 
  (steps_between_3_and_5 : ℕ) 
  (third_floor : ℕ := 3) 
  (fifth_floor : ℕ := 5) 
  (eleventh_floor : ℕ := 11) 
  (ground_floor : ℕ := 1) 
  (steps_per_floor : ℕ := steps_between_3_and_5 / (fifth_floor - third_floor)) :
  steps_between_3_and_5 = 42 →
  steps_between_3_and_5 / (fifth_floor - third_floor) = 21 →
  (eleventh_floor - ground_floor) = 10 →
  21 * 10 = 210 := 
by
  intros _ _ _
  exact rfl

end steps_to_11th_floor_l281_28165


namespace highest_value_of_a_l281_28120

theorem highest_value_of_a (a : ℕ) (h : 0 ≤ a ∧ a ≤ 9) : (365 * 10 ^ 3 + a * 10 ^ 2 + 16) % 8 = 0 → a = 8 := by
  sorry

end highest_value_of_a_l281_28120


namespace scarlett_oil_amount_l281_28176

theorem scarlett_oil_amount (initial_oil add_oil : ℝ) (h1 : initial_oil = 0.17) (h2 : add_oil = 0.67) :
  initial_oil + add_oil = 0.84 :=
by
  rw [h1, h2]
  -- Proof step goes here
  sorry

end scarlett_oil_amount_l281_28176


namespace general_term_arithmetic_sequence_l281_28133

-- Consider an arithmetic sequence {a_n}
variable (a : ℕ → ℤ)

-- Conditions
def a1 : Prop := a 1 = 1
def a3 : Prop := a 3 = -3
def is_arithmetic_sequence : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Theorem statement
theorem general_term_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 1) (h3 : a 3 = -3) (h_arith : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 3 - 2 * n :=
by
  sorry  -- proof is not required

end general_term_arithmetic_sequence_l281_28133


namespace melanie_more_turnips_l281_28116

theorem melanie_more_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) :
  melanie_turnips - benny_turnips = 26 := by
  sorry

end melanie_more_turnips_l281_28116


namespace sin_cos_half_angle_sum_l281_28111

theorem sin_cos_half_angle_sum 
  (θ : ℝ)
  (hcos : Real.cos θ = -7/25) 
  (hθ : θ ∈ Set.Ioo (-Real.pi) 0) : 
  Real.sin (θ/2) + Real.cos (θ/2) = -1/5 := 
sorry

end sin_cos_half_angle_sum_l281_28111


namespace relationship_between_D_and_A_l281_28172

variable {A B C D : Prop}

theorem relationship_between_D_and_A
  (h1 : A → B)
  (h2 : B → C)
  (h3 : D ↔ C) :
  (A → D) ∧ ¬(D → A) :=
by
sorry

end relationship_between_D_and_A_l281_28172


namespace num_sets_M_l281_28177

theorem num_sets_M (M : Set ℕ) :
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5, 6} → ∃ n : Nat, n = 16 :=
by
  sorry

end num_sets_M_l281_28177


namespace tree_last_tree_height_difference_l281_28135

noncomputable def treeHeightDifference : ℝ :=
  let t1 := 1000
  let t2 := 500
  let t3 := 500
  let avgHeight := 800
  let lastTreeHeight := 4 * avgHeight - (t1 + t2 + t3)
  lastTreeHeight - t1

theorem tree_last_tree_height_difference :
  treeHeightDifference = 200 := sorry

end tree_last_tree_height_difference_l281_28135


namespace cost_of_graveling_per_sq_meter_l281_28163

theorem cost_of_graveling_per_sq_meter
    (length_lawn : ℝ) (breadth_lawn : ℝ)
    (width_road : ℝ) (total_cost_gravel : ℝ)
    (length_road_area : ℝ) (breadth_road_area : ℝ) (intersection_area : ℝ)
    (total_graveled_area : ℝ) (cost_per_sq_meter : ℝ) :
    length_lawn = 55 →
    breadth_lawn = 35 →
    width_road = 4 →
    total_cost_gravel = 258 →
    length_road_area = length_lawn * width_road →
    intersection_area = width_road * width_road →
    breadth_road_area = breadth_lawn * width_road - intersection_area →
    total_graveled_area = length_road_area + breadth_road_area →
    cost_per_sq_meter = total_cost_gravel / total_graveled_area →
    cost_per_sq_meter = 0.75 :=
by
  intros
  sorry

end cost_of_graveling_per_sq_meter_l281_28163


namespace spadesuit_problem_l281_28106

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := abs (a - b)

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := 
sorry

end spadesuit_problem_l281_28106


namespace angle_SVU_l281_28118

theorem angle_SVU (TU SV SU : ℝ) (angle_STU_T : ℝ) (angle_STU_S : ℝ) :
  TU = SV → angle_STU_T = 75 → angle_STU_S = 30 →
  TU = SU → SU = SV → S_V_U = 65 :=
by
  intros H1 H2 H3 H4 H5
  -- skip proof
  sorry

end angle_SVU_l281_28118


namespace work_rate_c_l281_28136

theorem work_rate_c (A B C : ℝ) (h1 : A + B = 1 / 4) (h2 : B + C = 1 / 6) (h3 : C + A = 1 / 3) :
    1 / C = 8 :=
by
  sorry

end work_rate_c_l281_28136


namespace x_eq_1_sufficient_not_necessary_l281_28171

theorem x_eq_1_sufficient_not_necessary (x : ℝ) : 
    (x = 1 → (x^2 - 3 * x + 2 = 0)) ∧ ¬((x^2 - 3 * x + 2 = 0) → (x = 1)) := 
by
  sorry

end x_eq_1_sufficient_not_necessary_l281_28171


namespace pure_imaginary_number_a_l281_28132

theorem pure_imaginary_number_a (a : ℝ) 
  (h1 : a^2 + 2 * a - 3 = 0)
  (h2 : a^2 - 4 * a + 3 ≠ 0) : a = -3 :=
sorry

end pure_imaginary_number_a_l281_28132


namespace largest_common_number_in_arithmetic_sequences_l281_28197

theorem largest_common_number_in_arithmetic_sequences (n : ℕ) :
  (∃ a1 a2 : ℕ, a1 = 5 + 8 * n ∧ a2 = 3 + 9 * n ∧ a1 = a2 ∧ 1 ≤ a1 ∧ a1 ≤ 150) →
  (a1 = 93) :=
by
  sorry

end largest_common_number_in_arithmetic_sequences_l281_28197


namespace valid_passwords_count_l281_28192

-- Define the total number of unrestricted passwords
def total_passwords : ℕ := 10000

-- Define the number of restricted passwords (ending with 6, 3, 9)
def restricted_passwords : ℕ := 10

-- Define the total number of valid passwords
def valid_passwords := total_passwords - restricted_passwords

theorem valid_passwords_count : valid_passwords = 9990 := 
by 
  sorry

end valid_passwords_count_l281_28192


namespace product_approximation_l281_28109

-- Define the approximation condition
def approxProduct (x y : ℕ) (approxX approxY : ℕ) : ℕ :=
  approxX * approxY

-- State the theorem
theorem product_approximation :
  let x := 29
  let y := 32
  let approxX := 30
  let approxY := 30
  approxProduct x y approxX approxY = 900 := by
  sorry

end product_approximation_l281_28109


namespace a_3_value_l281_28127

def arithmetic_seq (a: ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_value :
  ∃ a : ℕ → ℤ, a 1 = 19 ∧ arithmetic_seq a ∧ a 3 = 13 :=
by
  sorry

end a_3_value_l281_28127


namespace cade_marbles_now_l281_28123

def original_marbles : ℝ := 87.0
def added_marbles : ℝ := 8.0
def total_marbles : ℝ := original_marbles + added_marbles

theorem cade_marbles_now : total_marbles = 95.0 :=
by
  sorry

end cade_marbles_now_l281_28123


namespace betty_cookies_brownies_l281_28137

theorem betty_cookies_brownies (cookies_per_day brownies_per_day initial_cookies initial_brownies days : ℕ) :
  cookies_per_day = 3 → brownies_per_day = 1 → initial_cookies = 60 → initial_brownies = 10 → days = 7 →
  initial_cookies - days * cookies_per_day - (initial_brownies - days * brownies_per_day) = 36 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_cookies_brownies_l281_28137


namespace range_of_a_l281_28145

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l281_28145


namespace root_product_minus_sums_l281_28184

variable {b c : ℝ}

theorem root_product_minus_sums
  (h1 : 3 * b^2 + 5 * b - 2 = 0)
  (h2 : 3 * c^2 + 5 * c - 2 = 0)
  : (b - 1) * (c - 1) = 2 := 
by
  sorry

end root_product_minus_sums_l281_28184


namespace sqrt_of_0_09_l281_28173

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end sqrt_of_0_09_l281_28173


namespace find_m_n_l281_28164

theorem find_m_n (m n : ℕ) (hmn : m + 6 < n + 4)
  (median_cond : ((m + 2 + m + 6 + n + 4 + n + 5) / 7) = n + 2)
  (mean_cond : ((m + (m + 2) + (m + 6) + (n + 4) + (n + 5) + (2 * n - 1) + (2 * n + 2)) / 7) = n + 2) :
  m + n = 10 :=
sorry

end find_m_n_l281_28164


namespace product_of_undefined_roots_l281_28188

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end product_of_undefined_roots_l281_28188


namespace train_crossing_time_is_correct_l281_28147

noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_crossing_time_is_correct :
  train_crossing_time 250 180 120 = 12.9 :=
by
  sorry

end train_crossing_time_is_correct_l281_28147


namespace at_least_one_expression_is_leq_neg_two_l281_28157

variable (a b c : ℝ)

theorem at_least_one_expression_is_leq_neg_two 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1 / b ≤ -2) ∨ (b + 1 / c ≤ -2) ∨ (c + 1 / a ≤ -2) :=
sorry

end at_least_one_expression_is_leq_neg_two_l281_28157


namespace count_even_fibonacci_first_2007_l281_28103

def fibonacci (n : Nat) : Nat :=
  if h : n = 0 then 0
  else if h : n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

def fibonacci_parity : List Bool := List.map (fun x => fibonacci x % 2 = 0) (List.range 2008)

def count_even (l : List Bool) : Nat :=
  l.foldl (fun acc x => if x then acc + 1 else acc) 0

theorem count_even_fibonacci_first_2007 : count_even (fibonacci_parity.take 2007) = 669 :=
sorry

end count_even_fibonacci_first_2007_l281_28103


namespace problem_1_problem_2_l281_28191

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a|

theorem problem_1 (x : ℝ) : (∀ x, f x 4 < 8 - |x - 1|) → x ∈ Set.Ioo (-1 : ℝ) (13 / 3) :=
by sorry

theorem problem_2 (a : ℝ) : (∃ x, f x a > 8 + |2 * x - 1|) → a > 9 ∨ a < -7 :=
by sorry

end problem_1_problem_2_l281_28191


namespace allan_balloons_l281_28104

theorem allan_balloons (a j t : ℕ) (h1 : t = 6) (h2 : j = 4) (h3 : t = a + j) : a = 2 := by
  sorry

end allan_balloons_l281_28104
