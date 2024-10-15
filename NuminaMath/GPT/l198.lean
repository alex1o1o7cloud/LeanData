import Mathlib

namespace NUMINAMATH_GPT_determine_m_l198_19844

noncomputable def function_f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

def exists_constant_interval (a b c m : ℝ) : Prop :=
  a < b ∧ ∀ x, a ≤ x ∧ x ≤ b → function_f m x = c

theorem determine_m (m : ℝ) (a b c : ℝ) :
  (a < b ∧ a ≥ -2 ∧ b ≥ -2 ∧ (∀ x, a ≤ x ∧ x ≤ b → function_f m x = c)) →
  m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_GPT_determine_m_l198_19844


namespace NUMINAMATH_GPT_solve_new_system_l198_19816

theorem solve_new_system (a_1 b_1 a_2 b_2 c_1 c_2 x y : ℝ)
(h1 : a_1 * 2 - b_1 * (-1) = c_1)
(h2 : a_2 * 2 + b_2 * (-1) = c_2) :
  (x = -1) ∧ (y = 1) :=
by
  have hx : x + 3 = 2 := by sorry
  have hy : y - 2 = -1 := by sorry
  have hx_sol : x = -1 := by linarith
  have hy_sol : y = 1 := by linarith
  exact ⟨hx_sol, hy_sol⟩

end NUMINAMATH_GPT_solve_new_system_l198_19816


namespace NUMINAMATH_GPT_find_a_perpendicular_line_l198_19815

theorem find_a_perpendicular_line (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 3 * y + 1 = 0) → (2 * x + 2 * y - 3 = 0) → (-(a / 3) * (-1) = -1)) → 
  a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_perpendicular_line_l198_19815


namespace NUMINAMATH_GPT_pythagorean_ratio_l198_19863

variables (a b : ℝ)

theorem pythagorean_ratio (h1 : a > 0) (h2 : b > a) (h3 : b^2 = 13 * (b - a)^2) :
  a / b = 2 / 3 :=
sorry

end NUMINAMATH_GPT_pythagorean_ratio_l198_19863


namespace NUMINAMATH_GPT_candy_box_price_l198_19879

theorem candy_box_price (c s : ℝ) 
  (h1 : 1.50 * s = 6) 
  (h2 : c + s = 16) 
  (h3 : ∀ c, 1.25 * c = 1.25 * 12) : 
  (1.25 * c = 15) :=
by
  sorry

end NUMINAMATH_GPT_candy_box_price_l198_19879


namespace NUMINAMATH_GPT_min_S_l198_19899

-- Define the arithmetic sequence
def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + (n - 1) * d

-- Define the sum of the first n terms
def S (n : ℕ) (a1 : ℤ) (d : ℤ) : ℤ :=
  (n * (a1 + a n a1 d)) / 2

-- Conditions
def a4 : ℤ := -15
def d : ℤ := 3

-- Found a1 from a4 and d
def a1 : ℤ := -24

-- Theorem stating the minimum value of the sum
theorem min_S : ∃ n, S n a1 d = -108 :=
  sorry

end NUMINAMATH_GPT_min_S_l198_19899


namespace NUMINAMATH_GPT_polynomial_value_l198_19838

variable (x : ℝ)

theorem polynomial_value (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l198_19838


namespace NUMINAMATH_GPT_amusement_park_ticket_cost_l198_19846

theorem amusement_park_ticket_cost (T_adult T_child : ℕ) (num_children num_adults : ℕ) 
  (h1 : T_adult = 15) (h2 : T_child = 8) 
  (h3 : num_children = 15) (h4 : num_adults = 25 + num_children) :
  num_adults * T_adult + num_children * T_child = 720 :=
by
  sorry

end NUMINAMATH_GPT_amusement_park_ticket_cost_l198_19846


namespace NUMINAMATH_GPT_find_x_values_l198_19865

open Real

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h₁ : x + 1/y = 5) (h₂ : y + 1/x = 7/4) : 
  x = 4/7 ∨ x = 5 := 
by sorry

end NUMINAMATH_GPT_find_x_values_l198_19865


namespace NUMINAMATH_GPT_number_of_integer_solutions_Q_is_one_l198_19898

def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 13 * x^2 + 3 * x - 19

theorem number_of_integer_solutions_Q_is_one : 
    (∃! x : ℤ, ∃ k : ℤ, Q x = k^2) := 
sorry

end NUMINAMATH_GPT_number_of_integer_solutions_Q_is_one_l198_19898


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l198_19880

-- Conditions and definitions for the arithmetic sequence
def arithmetic_seq (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n+1) = a_n n + d

def sum_seq (a_n S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

def T_plus_K_eq_19 (T K : ℕ) : Prop :=
  T + K = 19

-- The given problem to prove
theorem arithmetic_seq_problem (a_n S_n : ℕ → ℝ) (d : ℝ) (h1 : d > 0)
  (h2 : arithmetic_seq a_n d) (h3 : sum_seq a_n S_n)
  (h4 : ∀ T K, T_plus_K_eq_19 T K → S_n T = S_n K) :
  ∃! n, a_n n - S_n n ≥ 0 := sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l198_19880


namespace NUMINAMATH_GPT_modulusOfComplexNumber_proof_l198_19822

noncomputable def complexNumber {a : ℝ} (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : ℂ :=
  (2 + Real.sqrt 2 * Complex.I) / (a - Complex.I)

theorem modulusOfComplexNumber_proof (a : ℝ) (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : Complex.abs (complexNumber h) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_modulusOfComplexNumber_proof_l198_19822


namespace NUMINAMATH_GPT_question1_question2_l198_19812

noncomputable def A (x : ℝ) : Prop := x^2 - 3 * x + 2 ≤ 0
noncomputable def B_set (x a : ℝ) : ℝ := x^2 - 2 * x + a
def B (y a : ℝ) : Prop := y ≥ a - 1
noncomputable def C (x a : ℝ) : Prop := x^2 - a * x - 4 ≤ 0

def prop_p (a : ℝ) : Prop := ∃ x, A x ∧ B (B_set x a) a
def prop_q (a : ℝ) : Prop := ∀ x, A x → C x a

theorem question1 (a : ℝ) (h : ¬ prop_p a) : a > 3 :=
sorry

theorem question2 (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 0 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_question1_question2_l198_19812


namespace NUMINAMATH_GPT_find_min_a_l198_19856

theorem find_min_a (a : ℕ) (h1 : (3150 * a) = x^2) (h2 : a > 0) :
  a = 14 := by
  sorry

end NUMINAMATH_GPT_find_min_a_l198_19856


namespace NUMINAMATH_GPT_parabola_and_x4_value_l198_19858

theorem parabola_and_x4_value :
  (∀ P, dist P (0, 1/2) = dist P (x, -1/2) → ∃ y, P = (x, y) ∧ x^2 = 2 * y) ∧
  (∀ (x1 x2 : ℝ), x1 = 6 → x2 = 2 → ∃ x4, 1/x4 = 1/((3/2) : ℝ) + 1/x2 ∧ x4 = 6/7) :=
by
  sorry

end NUMINAMATH_GPT_parabola_and_x4_value_l198_19858


namespace NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_nonagon_is_140_l198_19854

-- Define the number of sides for a nonagon
def number_of_sides_nonagon : ℕ := 9

-- Define the formula for the sum of the interior angles of a regular n-gon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- The sum of the interior angles of a nonagon
def sum_of_interior_angles_nonagon : ℕ := sum_of_interior_angles number_of_sides_nonagon

-- The measure of one interior angle of a regular n-gon
def measure_of_one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- The measure of one interior angle of a regular nonagon
def measure_of_one_interior_angle_nonagon : ℕ := measure_of_one_interior_angle number_of_sides_nonagon

-- The final theorem statement
theorem measure_of_one_interior_angle_of_regular_nonagon_is_140 : 
  measure_of_one_interior_angle_nonagon = 140 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_nonagon_is_140_l198_19854


namespace NUMINAMATH_GPT_slope_of_line_det_by_two_solutions_l198_19895

theorem slope_of_line_det_by_two_solutions (x y : ℝ) (h : 3 / x + 4 / y = 0) :
  (y = -4 * x / 3) → 
  ∀ x1 x2 y1 y2, (y1 = -4 * x1 / 3) ∧ (y2 = -4 * x2 / 3) → 
  (y2 - y1) / (x2 - x1) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_slope_of_line_det_by_two_solutions_l198_19895


namespace NUMINAMATH_GPT_geometric_sequence_a6a7_l198_19873

theorem geometric_sequence_a6a7 (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n+1) = q * a n)
  (h1 : a 4 * a 5 = 1)
  (h2 : a 8 * a 9 = 16) : a 6 * a 7 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a6a7_l198_19873


namespace NUMINAMATH_GPT_product_of_two_numbers_l198_19807

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 289)
  (h2 : x + y = 23) : 
  x * y = 120 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l198_19807


namespace NUMINAMATH_GPT_ratio_of_saute_times_l198_19839

-- Definitions
def time_saute_onions : ℕ := 20
def time_saute_garlic_and_peppers : ℕ := 5
def time_knead_dough : ℕ := 30
def time_rest_dough : ℕ := 2 * time_knead_dough
def combined_knead_rest_time : ℕ := time_knead_dough + time_rest_dough
def time_assemble_calzones : ℕ := combined_knead_rest_time / 10
def total_time : ℕ := 124

-- Conditions
axiom saute_time_condition : time_saute_onions + time_saute_garlic_and_peppers + time_knead_dough + time_rest_dough + time_assemble_calzones = total_time

-- Question to be proved as a theorem
theorem ratio_of_saute_times :
  (time_saute_garlic_and_peppers : ℚ) / time_saute_onions = 1 / 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_saute_times_l198_19839


namespace NUMINAMATH_GPT_range_S₁₂_div_d_l198_19855

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a₁ d : α) (n : ℕ) : α :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem range_S₁₂_div_d (a₁ d : α) (h_a₁_pos : a₁ > 0) (h_d_neg : d < 0) 
  (h_max_S_8 : ∀ n, arithmetic_sequence_sum a₁ d n ≤ arithmetic_sequence_sum a₁ d 8) :
  -30 < (arithmetic_sequence_sum a₁ d 12) / d ∧ (arithmetic_sequence_sum a₁ d 12) / d < -18 :=
by
  have h1 : -8 < a₁ / d := by sorry
  have h2 : a₁ / d < -7 := by sorry
  have h3 : (arithmetic_sequence_sum a₁ d 12) / d = 12 * (a₁ / d) + 66 := by sorry
  sorry

end NUMINAMATH_GPT_range_S₁₂_div_d_l198_19855


namespace NUMINAMATH_GPT_triangle_identity_l198_19871

variables (a b c h_a h_b h_c x y z : ℝ)

-- Define the given conditions
def condition1 := a / h_a = x
def condition2 := b / h_b = y
def condition3 := c / h_c = z

-- Statement of the theorem to be proved
theorem triangle_identity 
  (h1 : condition1 a h_a x) 
  (h2 : condition2 b h_b y) 
  (h3 : condition3 c h_c z) : 
  x^2 + y^2 + z^2 - 2 * x * y - 2 * y * z - 2 * z * x + 4 = 0 := 
  by 
    sorry

end NUMINAMATH_GPT_triangle_identity_l198_19871


namespace NUMINAMATH_GPT_greatest_whole_number_lt_100_with_odd_factors_l198_19830

theorem greatest_whole_number_lt_100_with_odd_factors :
  ∃ n, n < 100 ∧ (∃ p : ℕ, n = p * p) ∧ 
    ∀ m, (m < 100 ∧ (∃ q : ℕ, m = q * q)) → m ≤ n :=
sorry

end NUMINAMATH_GPT_greatest_whole_number_lt_100_with_odd_factors_l198_19830


namespace NUMINAMATH_GPT_minimum_odd_numbers_in_A_P_l198_19827

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end NUMINAMATH_GPT_minimum_odd_numbers_in_A_P_l198_19827


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l198_19840

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x * (x - 1) = 0) ∧ ¬(x * (x - 1) = 0 → x = 1) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l198_19840


namespace NUMINAMATH_GPT_floor_e_sub_6_eq_neg_4_l198_19882

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end NUMINAMATH_GPT_floor_e_sub_6_eq_neg_4_l198_19882


namespace NUMINAMATH_GPT_f_x_f_2x_plus_1_l198_19876

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem f_x (x : ℝ) : f x = x^2 - 2 * x - 3 := 
by sorry

theorem f_2x_plus_1 (x : ℝ) : f (2 * x + 1) = 4 * x^2 - 4 := 
by sorry

end NUMINAMATH_GPT_f_x_f_2x_plus_1_l198_19876


namespace NUMINAMATH_GPT_distance_traveled_by_bus_l198_19805

noncomputable def total_distance : ℕ := 900
noncomputable def distance_by_plane : ℕ := total_distance / 3
noncomputable def distance_by_bus : ℕ := 360
noncomputable def distance_by_train : ℕ := (2 * distance_by_bus) / 3

theorem distance_traveled_by_bus :
  distance_by_plane + distance_by_train + distance_by_bus = total_distance :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_by_bus_l198_19805


namespace NUMINAMATH_GPT_total_games_played_l198_19890

theorem total_games_played (months : ℕ) (games_per_month : ℕ) (h1 : months = 17) (h2 : games_per_month = 19) : 
  months * games_per_month = 323 :=
by
  sorry

end NUMINAMATH_GPT_total_games_played_l198_19890


namespace NUMINAMATH_GPT_parity_of_magazines_and_celebrities_l198_19872

-- Define the main problem statement using Lean 4

theorem parity_of_magazines_and_celebrities {m c : ℕ}
  (h1 : ∀ i, i < m → ∃ d_i, d_i % 2 = 1)
  (h2 : ∀ j, j < c → ∃ e_j, e_j % 2 = 1) :
  (m % 2 = c % 2) ∧ (∃ ways, ways = 2 ^ ((m - 1) * (c - 1))) :=
by
  sorry

end NUMINAMATH_GPT_parity_of_magazines_and_celebrities_l198_19872


namespace NUMINAMATH_GPT_infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l198_19892

-- Define x, y, z to be natural numbers
def has_infinitely_many_solutions : Prop :=
  ∃ (x y z : ℕ), x^2 + 2 * y^2 = z^2

-- Prove that there are infinitely many such x, y, z
theorem infinite_solutions_x2_plus_2y2_eq_z2 : has_infinitely_many_solutions :=
  sorry

-- Define x, y, z, t to be integers and non-zero
def no_nontrivial_integer_quadruplets : Prop :=
  ∀ (x y z t : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) → 
    ¬((x^2 + 2 * y^2 = z^2) ∧ (2 * x^2 + y^2 = t^2))

-- Prove that no nontrivial integer quadruplets exist
theorem no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2 : no_nontrivial_integer_quadruplets :=
  sorry

end NUMINAMATH_GPT_infinite_solutions_x2_plus_2y2_eq_z2_no_nontrivial_solns_x2_plus_2y2_eq_z2_and_2x2_plus_y2_eq_t2_l198_19892


namespace NUMINAMATH_GPT_peter_ends_up_with_eleven_erasers_l198_19832

def eraser_problem : Nat :=
  let initial_erasers := 8
  let additional_erasers := 3
  let total_erasers := initial_erasers + additional_erasers
  total_erasers

theorem peter_ends_up_with_eleven_erasers :
  eraser_problem = 11 :=
by
  sorry

end NUMINAMATH_GPT_peter_ends_up_with_eleven_erasers_l198_19832


namespace NUMINAMATH_GPT_alok_total_payment_l198_19868

theorem alok_total_payment :
  let chapatis_cost := 16 * 6
  let rice_cost := 5 * 45
  let mixed_vegetable_cost := 7 * 70
  chapatis_cost + rice_cost + mixed_vegetable_cost = 811 :=
by
  sorry

end NUMINAMATH_GPT_alok_total_payment_l198_19868


namespace NUMINAMATH_GPT_det_A_l198_19836

-- Define the matrix A
noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sin 1, Real.cos 2, Real.sin 3],
   ![Real.sin 4, Real.cos 5, Real.sin 6],
   ![Real.sin 7, Real.cos 8, Real.sin 9]]

-- Define the explicit determinant calculation
theorem det_A :
  Matrix.det A = Real.sin 1 * (Real.cos 5 * Real.sin 9 - Real.sin 6 * Real.cos 8) -
                 Real.cos 2 * (Real.sin 4 * Real.sin 9 - Real.sin 6 * Real.sin 7) +
                 Real.sin 3 * (Real.sin 4 * Real.cos 8 - Real.cos 5 * Real.sin 7) :=
by
  sorry

end NUMINAMATH_GPT_det_A_l198_19836


namespace NUMINAMATH_GPT_number_of_girls_who_left_l198_19835

-- Definitions for initial conditions and event information
def initial_boys : ℕ := 24
def initial_girls : ℕ := 14
def final_students : ℕ := 30

-- Main theorem statement translating the problem question
theorem number_of_girls_who_left (B G : ℕ) (h1 : B = G) 
  (h2 : initial_boys + initial_girls - B - G = final_students) :
  G = 4 := 
sorry

end NUMINAMATH_GPT_number_of_girls_who_left_l198_19835


namespace NUMINAMATH_GPT_find_c_l198_19889

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12 * x + 3 * x^2 - 4 * x^3 + 5 * x^4
def g (x : ℝ) : ℝ := 3 - 2 * x - 6 * x^3 + 7 * x^4

-- Define the main theorem stating that c = -5/7 makes f(x) + c*g(x) have degree 3
theorem find_c (c : ℝ) (h : ∀ x : ℝ, f x + c * g x = 0) : c = -5 / 7 := by
  sorry

end NUMINAMATH_GPT_find_c_l198_19889


namespace NUMINAMATH_GPT_temperature_drop_l198_19885

theorem temperature_drop (initial_temperature drop: ℤ) (h1: initial_temperature = 3) (h2: drop = 5) : initial_temperature - drop = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_temperature_drop_l198_19885


namespace NUMINAMATH_GPT_lake_crystal_frogs_percentage_l198_19825

noncomputable def percentage_fewer_frogs (frogs_in_lassie_lake total_frogs : ℕ) : ℕ :=
  let P := (total_frogs - frogs_in_lassie_lake) * 100 / frogs_in_lassie_lake
  P

theorem lake_crystal_frogs_percentage :
  let frogs_in_lassie_lake := 45
  let total_frogs := 81
  percentage_fewer_frogs frogs_in_lassie_lake total_frogs = 20 :=
by
  sorry

end NUMINAMATH_GPT_lake_crystal_frogs_percentage_l198_19825


namespace NUMINAMATH_GPT_value_of_m_l198_19847

theorem value_of_m (m x : ℝ) (h1 : mx + 1 = 2 * (m - x)) (h2 : |x + 2| = 0) : m = -|3 / 4| :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l198_19847


namespace NUMINAMATH_GPT_norma_cards_left_l198_19800

def initial_cards : ℕ := 88
def lost_cards : ℕ := 70
def remaining_cards (initial lost : ℕ) : ℕ := initial - lost

theorem norma_cards_left : remaining_cards initial_cards lost_cards = 18 := by
  sorry

end NUMINAMATH_GPT_norma_cards_left_l198_19800


namespace NUMINAMATH_GPT_part_one_part_two_l198_19808

def f (x : ℝ) : ℝ := abs (3 * x + 2)

theorem part_one (x : ℝ) : f x < 4 - abs (x - 1) ↔ x ∈ Set.Ioo (-5 / 4) (1 / 2) :=
sorry

noncomputable def g (x a : ℝ) : ℝ :=
if x < -2/3 then 2 * x + 2 + a
else if x ≤ a then -4 * x - 2 + a
else -2 * x - 2 - a

theorem part_two (m n a : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) :
  (∀ (x : ℝ), abs (x - a) - f x ≤ 1 / m + 1 / n) ↔ (0 < a ∧ a ≤ 10 / 3) :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l198_19808


namespace NUMINAMATH_GPT_fractional_equation_solution_l198_19820

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (2 - x) - 1 = (2 * x - 5) / (x - 2) → x = 3 :=
by 
  intro h_eq
  sorry

end NUMINAMATH_GPT_fractional_equation_solution_l198_19820


namespace NUMINAMATH_GPT_box_interior_surface_area_l198_19810

-- Defining the conditions
def original_length := 30
def original_width := 20
def corner_length := 5
def num_corners := 4

-- Defining the area calculations based on given dimensions and removed corners
def original_area := original_length * original_width
def area_one_corner := corner_length * corner_length
def total_area_removed := num_corners * area_one_corner
def remaining_area := original_area - total_area_removed

-- Statement to prove
theorem box_interior_surface_area :
  remaining_area = 500 :=
by 
  sorry

end NUMINAMATH_GPT_box_interior_surface_area_l198_19810


namespace NUMINAMATH_GPT_members_count_l198_19803

theorem members_count
  (n : ℝ)
  (h1 : 191.25 = n / 4) :
  n = 765 :=
by
  sorry

end NUMINAMATH_GPT_members_count_l198_19803


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l198_19870

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ y = k / x) →
  (∀ x : ℝ, x ≠ 0 → ( (x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0) ) ) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l198_19870


namespace NUMINAMATH_GPT_largest_number_from_hcf_factors_l198_19862

/-- This statement checks the largest number derivable from given HCF and factors. -/
theorem largest_number_from_hcf_factors (HCF factor1 factor2 : ℕ) (hHCF : HCF = 52) (hfactor1 : factor1 = 11) (hfactor2 : factor2 = 12) :
  max (HCF * factor1) (HCF * factor2) = 624 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_from_hcf_factors_l198_19862


namespace NUMINAMATH_GPT_volume_frustum_l198_19891

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1/3) * (base_edge ^ 2) * height

theorem volume_frustum (original_base_edge original_height small_base_edge small_height : ℝ)
  (h_orig : original_base_edge = 10) (h_orig_height : original_height = 10)
  (h_small : small_base_edge = 5) (h_small_height : small_height = 5) :
  volume_pyramid original_base_edge original_height - volume_pyramid small_base_edge small_height
  = 875 / 3 := by
    simp [volume_pyramid, h_orig, h_orig_height, h_small, h_small_height]
    sorry

end NUMINAMATH_GPT_volume_frustum_l198_19891


namespace NUMINAMATH_GPT_f_x_plus_f_neg_x_eq_seven_l198_19869

variable (f : ℝ → ℝ)

-- Given conditions: 
axiom cond1 : ∀ x : ℝ, f x + f (1 - x) = 10
axiom cond2 : ∀ x : ℝ, f (1 + x) = 3 + f x

-- Prove statement:
theorem f_x_plus_f_neg_x_eq_seven : ∀ x : ℝ, f x + f (-x) = 7 := 
by
  sorry

end NUMINAMATH_GPT_f_x_plus_f_neg_x_eq_seven_l198_19869


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l198_19811

-- Given an arithmetic sequence 
variable {a : ℕ → ℝ}

-- The conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k, m = n + k → a (m + 1) - a m = a (n + 1) - a n

def condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 9

-- Lean 4 statement for the proof problem
theorem no_real_roots_of_quadratic (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : condition a) :
  let b := a 4 + a 6
  ∃ Δ, Δ = b ^ 2 - 4 * 10 ∧ Δ < 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l198_19811


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l198_19813

theorem sufficient_not_necessary_condition :
  ∀ x : ℝ, (x^2 - 3 * x < 0) → (0 < x ∧ x < 2) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l198_19813


namespace NUMINAMATH_GPT_intersection_A_B_l198_19828

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_A_B : A ∩ B = {1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l198_19828


namespace NUMINAMATH_GPT_black_lambs_count_l198_19842

def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193
def brown_lambs : ℕ := 527

theorem black_lambs_count :
  total_lambs - white_lambs - brown_lambs = 5328 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_black_lambs_count_l198_19842


namespace NUMINAMATH_GPT_problem_l198_19896

theorem problem (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
sorry

end NUMINAMATH_GPT_problem_l198_19896


namespace NUMINAMATH_GPT_total_weight_of_envelopes_l198_19806

theorem total_weight_of_envelopes :
  (8.5 * 880 / 1000) = 7.48 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_envelopes_l198_19806


namespace NUMINAMATH_GPT_functional_eq_unique_solution_l198_19897

theorem functional_eq_unique_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_functional_eq_unique_solution_l198_19897


namespace NUMINAMATH_GPT_at_least_one_even_difference_l198_19826

-- Statement of the problem in Lean 4
theorem at_least_one_even_difference 
  (a b : Fin (2 * n + 1) → ℤ) 
  (hperm : ∃ σ : Equiv.Perm (Fin (2 * n + 1)), ∀ k, a k = (b ∘ σ) k) : 
  ∃ k, (a k - b k) % 2 = 0 := 
sorry

end NUMINAMATH_GPT_at_least_one_even_difference_l198_19826


namespace NUMINAMATH_GPT_min_value_x_y_l198_19802

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 4/y = 1) : x + y ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_l198_19802


namespace NUMINAMATH_GPT_domain_of_g_eq_7_infty_l198_19893

noncomputable def domain_function (x : ℝ) : Prop := (2 * x + 1 ≥ 0) ∧ (x - 7 > 0)

theorem domain_of_g_eq_7_infty : 
  (∀ x : ℝ, domain_function x ↔ x > 7) :=
by 
  -- We declare the structure of our proof problem here.
  -- The detailed proof steps would follow.
  sorry

end NUMINAMATH_GPT_domain_of_g_eq_7_infty_l198_19893


namespace NUMINAMATH_GPT_product_of_two_numbers_l198_19809

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x - y = 1 * k) 
  (h2 : x + y = 2 * k) 
  (h3 : (x * y)^2 = 18 * k) : (x * y = 16) := 
by 
    sorry


end NUMINAMATH_GPT_product_of_two_numbers_l198_19809


namespace NUMINAMATH_GPT_harly_adopts_percentage_l198_19849

/-- Definitions for the conditions -/
def initial_dogs : ℝ := 80
def dogs_taken_back : ℝ := 5
def dogs_left : ℝ := 53

/-- Define the percentage of dogs adopted out -/
def percentage_adopted (P : ℝ) := P

/-- Lean 4 statement where we prove that if the given conditions are met, then the percentage of dogs initially adopted out is 40 -/
theorem harly_adopts_percentage : 
  ∃ P : ℝ, 
    (initial_dogs - (percentage_adopted P / 100 * initial_dogs) + dogs_taken_back = dogs_left) 
    ∧ P = 40 :=
by
  sorry

end NUMINAMATH_GPT_harly_adopts_percentage_l198_19849


namespace NUMINAMATH_GPT_abs_ineq_real_solution_range_l198_19883

theorem abs_ineq_real_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x + 3| < a) ↔ a > 7 :=
sorry

end NUMINAMATH_GPT_abs_ineq_real_solution_range_l198_19883


namespace NUMINAMATH_GPT_percentage_increase_biking_time_l198_19859

theorem percentage_increase_biking_time
  (time_young_hours : ℕ)
  (distance_young_miles : ℕ)
  (time_now_hours : ℕ)
  (distance_now_miles : ℕ)
  (time_young_minutes : ℕ := time_young_hours * 60)
  (time_now_minutes : ℕ := time_now_hours * 60)
  (time_per_mile_young : ℕ := time_young_minutes / distance_young_miles)
  (time_per_mile_now : ℕ := time_now_minutes / distance_now_miles)
  (increase_in_time_per_mile : ℕ := time_per_mile_now - time_per_mile_young)
  (percentage_increase : ℕ := (increase_in_time_per_mile * 100) / time_per_mile_young) :
  percentage_increase = 100 :=
by
  -- substitution of values for conditions
  have time_young_hours := 2
  have distance_young_miles := 20
  have time_now_hours := 3
  have distance_now_miles := 15
  sorry

end NUMINAMATH_GPT_percentage_increase_biking_time_l198_19859


namespace NUMINAMATH_GPT_range_of_a_l198_19881

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (-x)

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) (h_ineq : f a (-2) > f a (-3)) : 0 < a ∧ a < 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l198_19881


namespace NUMINAMATH_GPT_quadratic_roots_max_value_l198_19888

theorem quadratic_roots_max_value (t q u₁ u₂ : ℝ)
  (h1 : u₁ + u₂ = t)
  (h2 : u₁ * u₂ = q)
  (h3 : u₁ + u₂ = u₁^2 + u₂^2)
  (h4 : u₁ + u₂ = u₁^4 + u₂^4) :
  (1 / u₁^2009 + 1 / u₂^2009) ≤ 2 :=
sorry

-- Explaination: 
-- This theorem states that given the conditions on the roots u₁ and u₂ of the quadratic equation, 
-- the maximum possible value of the expression (1 / u₁^2009 + 1 / u₂^2009) is 2.

end NUMINAMATH_GPT_quadratic_roots_max_value_l198_19888


namespace NUMINAMATH_GPT_sarah_pencils_on_tuesday_l198_19894

theorem sarah_pencils_on_tuesday 
    (x : ℤ)
    (h1 : 20 + x + 3 * x = 92) : 
    x = 18 := 
by 
    sorry

end NUMINAMATH_GPT_sarah_pencils_on_tuesday_l198_19894


namespace NUMINAMATH_GPT_harmonic_mean_of_3_6_12_l198_19829

-- Defining the harmonic mean function
def harmonic_mean (a b c : ℕ) : ℚ := 
  3 / ((1 / (a : ℚ)) + (1 / (b : ℚ)) + (1 / (c : ℚ)))

-- Stating the theorem
theorem harmonic_mean_of_3_6_12 : harmonic_mean 3 6 12 = 36 / 7 :=
by
  sorry

end NUMINAMATH_GPT_harmonic_mean_of_3_6_12_l198_19829


namespace NUMINAMATH_GPT_LawOfCosines_triangle_l198_19817

theorem LawOfCosines_triangle {a b C : ℝ} (ha : a = 9) (hb : b = 2 * Real.sqrt 3) (hC : C = Real.pi / 6 * 5) :
  ∃ c, c = 2 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_GPT_LawOfCosines_triangle_l198_19817


namespace NUMINAMATH_GPT_storks_initial_count_l198_19821

theorem storks_initial_count (S : ℕ) 
  (h1 : 6 = (S + 2) + 1) : S = 3 :=
sorry

end NUMINAMATH_GPT_storks_initial_count_l198_19821


namespace NUMINAMATH_GPT_polar_to_rectangular_correct_l198_19884

noncomputable def polar_to_rectangular (rho theta x y : ℝ) : Prop :=
  rho = 4 * Real.sin theta + 2 * Real.cos theta ∧
  rho * Real.sin theta = y ∧
  rho * Real.cos theta = x ∧
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5

theorem polar_to_rectangular_correct {rho theta x y : ℝ} :
  (rho = 4 * Real.sin theta + 2 * Real.cos theta) →
  (rho * Real.sin theta = y) →
  (rho * Real.cos theta = x) →
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_correct_l198_19884


namespace NUMINAMATH_GPT_percentage_of_green_ducks_smaller_pond_l198_19804

-- Definitions of the conditions
def num_ducks_smaller_pond : ℕ := 30
def num_ducks_larger_pond : ℕ := 50
def percentage_green_larger_pond : ℕ := 12
def percentage_green_total : ℕ := 15
def total_ducks : ℕ := num_ducks_smaller_pond + num_ducks_larger_pond

-- Calculation of the number of green ducks
def num_green_larger_pond := percentage_green_larger_pond * num_ducks_larger_pond / 100
def num_green_total := percentage_green_total * total_ducks / 100

-- Define the percentage of green ducks in the smaller pond
def percentage_green_smaller_pond (x : ℕ) :=
  x * num_ducks_smaller_pond / 100 + num_green_larger_pond = num_green_total

-- The theorem to be proven
theorem percentage_of_green_ducks_smaller_pond : percentage_green_smaller_pond 20 :=
  sorry

end NUMINAMATH_GPT_percentage_of_green_ducks_smaller_pond_l198_19804


namespace NUMINAMATH_GPT_friends_total_earnings_l198_19857

def Lauryn_earnings : ℝ := 2000
def Aurelia_fraction : ℝ := 0.7

def Aurelia_earnings : ℝ := Aurelia_fraction * Lauryn_earnings

def total_earnings : ℝ := Lauryn_earnings + Aurelia_earnings

theorem friends_total_earnings : total_earnings = 3400 := by
  -- We defined everything necessary here; the exact proof steps are omitted as per instructions.
  sorry

end NUMINAMATH_GPT_friends_total_earnings_l198_19857


namespace NUMINAMATH_GPT_annual_increase_fraction_l198_19841

theorem annual_increase_fraction (InitAmt FinalAmt : ℝ) (f : ℝ) :
  InitAmt = 51200 ∧ FinalAmt = 64800 ∧ FinalAmt = InitAmt * (1 + f)^2 →
  f = 0.125 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_annual_increase_fraction_l198_19841


namespace NUMINAMATH_GPT_length_of_longest_side_l198_19886

variable (a b c p x l : ℝ)

-- conditions of the original problem
def original_triangle_sides (a b c : ℝ) : Prop := a = 8 ∧ b = 15 ∧ c = 17

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def similar_triangle_perimeter (a b c p x : ℝ) : Prop := (a * x) + (b * x) + (c * x) = p

-- proof target
theorem length_of_longest_side (h1: original_triangle_sides a b c) 
                               (h2: is_right_triangle a b c) 
                               (h3: similar_triangle_perimeter a b c p x) 
                               (h4: x = 4)
                               (h5: p = 160): (c * x) = 68 := by
  -- to complete the proof
  sorry

end NUMINAMATH_GPT_length_of_longest_side_l198_19886


namespace NUMINAMATH_GPT_evaluate_expression_l198_19852

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : (2 * x - b + 5) = (b + 23) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l198_19852


namespace NUMINAMATH_GPT_total_stickers_used_l198_19853

-- Define all the conditions as given in the problem
def initially_water_bottles : ℕ := 20
def lost_at_school : ℕ := 5
def found_at_park : ℕ := 3
def stolen_at_dance : ℕ := 4
def misplaced_at_library : ℕ := 2
def acquired_from_friend : ℕ := 6
def stickers_per_bottle_school : ℕ := 4
def stickers_per_bottle_dance : ℕ := 3
def stickers_per_bottle_library : ℕ := 2

-- Prove the total number of stickers used
theorem total_stickers_used : 
  (lost_at_school * stickers_per_bottle_school)
  + (stolen_at_dance * stickers_per_bottle_dance)
  + (misplaced_at_library * stickers_per_bottle_library)
  = 36 := 
by
  sorry

end NUMINAMATH_GPT_total_stickers_used_l198_19853


namespace NUMINAMATH_GPT_cara_between_pairs_l198_19848

-- Definitions based on the conditions
def friends := 7 -- Cara has 7 friends
def fixed_neighbor : Prop := true -- Alex must always be one of the neighbors

-- Problem statement to be proven
theorem cara_between_pairs (h : fixed_neighbor): 
  ∃ n : ℕ, n = 6 ∧ (1 + (friends - 1)) = n := by
  sorry

end NUMINAMATH_GPT_cara_between_pairs_l198_19848


namespace NUMINAMATH_GPT_gummy_cost_proof_l198_19843

variables (lollipop_cost : ℝ) (num_lollipops : ℕ) (initial_money : ℝ) (remaining_money : ℝ)
variables (num_gummies : ℕ) (cost_per_gummy : ℝ)

-- Conditions
def conditions : Prop :=
  lollipop_cost = 1.50 ∧
  num_lollipops = 4 ∧
  initial_money = 15 ∧
  remaining_money = 5 ∧
  num_gummies = 2 ∧
  initial_money - remaining_money = (num_lollipops * lollipop_cost) + (num_gummies * cost_per_gummy)

-- Proof problem
theorem gummy_cost_proof : conditions lollipop_cost num_lollipops initial_money remaining_money num_gummies cost_per_gummy → cost_per_gummy = 2 :=
by
  sorry  -- Solution steps would be filled in here


end NUMINAMATH_GPT_gummy_cost_proof_l198_19843


namespace NUMINAMATH_GPT_impossible_sequence_l198_19801

theorem impossible_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) (a : ℕ → ℝ) (ha : ∀ n, 0 < a n) :
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) → false :=
by
  sorry

end NUMINAMATH_GPT_impossible_sequence_l198_19801


namespace NUMINAMATH_GPT_value_of_first_equation_l198_19833

variables (x y z w : ℝ)

theorem value_of_first_equation (h1 : xw + yz = 8) (h2 : (2 * x + y) * (2 * z + w) = 20) : xz + yw = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_first_equation_l198_19833


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l198_19824

theorem arithmetic_expression_eval : 2 + 8 * 3 - 4 + 10 * 2 / 5 = 26 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l198_19824


namespace NUMINAMATH_GPT_number_of_pieces_l198_19866

def area_of_pan (length : ℕ) (width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

theorem number_of_pieces (length width side : ℕ) (h_length : length = 24) (h_width : width = 15) (h_side : side = 3) :
  (area_of_pan length width) / (area_of_piece side) = 40 :=
by
  rw [h_length, h_width, h_side]
  sorry

end NUMINAMATH_GPT_number_of_pieces_l198_19866


namespace NUMINAMATH_GPT_investment_amount_l198_19867

theorem investment_amount (P: ℝ) (q_investment: ℝ) (ratio_pq: ℝ) (ratio_qp: ℝ) 
  (h1: ratio_pq = 4) (h2: ratio_qp = 6) (q_investment: ℝ) (h3: q_investment = 90000): 
  P = 60000 :=
by 
  -- Sorry is used here to skip the actual proof
  sorry

end NUMINAMATH_GPT_investment_amount_l198_19867


namespace NUMINAMATH_GPT_digits_same_l198_19877

theorem digits_same (k : ℕ) (hk : k ≥ 2) :
  (∃ n : ℕ, (10^(10^n) - 9^(9^n)) % (10^k) = 0) ↔ (k = 2 ∨ k = 3 ∨ k = 4) :=
sorry

end NUMINAMATH_GPT_digits_same_l198_19877


namespace NUMINAMATH_GPT_sale_second_month_l198_19850

def sale_first_month : ℝ := 5700
def sale_third_month : ℝ := 6855
def sale_fourth_month : ℝ := 3850
def sale_fifth_month : ℝ := 14045
def average_sale : ℝ := 7800

theorem sale_second_month : 
  ∃ x : ℝ, -- there exists a sale in the second month such that...
    (sale_first_month + x + sale_third_month + sale_fourth_month + sale_fifth_month) / 5 = average_sale
    ∧ x = 7550 := 
by
  sorry

end NUMINAMATH_GPT_sale_second_month_l198_19850


namespace NUMINAMATH_GPT_sum_of_possible_values_l198_19874

theorem sum_of_possible_values 
  (x y : ℝ) 
  (h : x * y - x / y^2 - y / x^2 = 3) :
  (x = 0 ∨ y = 0 → False) → 
  ((x - 1) * (y - 1) = 1 ∨ (x - 1) * (y - 1) = 4) → 
  ((x - 1) * (y - 1) = 1 → (x - 1) * (y - 1) = 1) → 
  ((x - 1) * (y - 1) = 4 → (x - 1) * (y - 1) = 4) → 
  (1 + 4 = 5) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l198_19874


namespace NUMINAMATH_GPT_no_non_trivial_solutions_l198_19818

theorem no_non_trivial_solutions (x y z : ℤ) :
  3 * x^2 + 7 * y^2 = z^4 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_no_non_trivial_solutions_l198_19818


namespace NUMINAMATH_GPT_range_of_a_l198_19814

noncomputable def a_n (n : ℕ) (a : ℝ) : ℝ :=
  (-1)^(n + 2018) * a

noncomputable def b_n (n : ℕ) : ℝ :=
  2 + (-1)^(n + 2019) / n

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ, 1 ≤ n → a_n n a < b_n n) ↔ -2 ≤ a ∧ a < 3 / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l198_19814


namespace NUMINAMATH_GPT_determinant_of_matrixA_l198_19837

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, -2],
  ![5, 6, -4],
  ![1, 3, 7]
]

theorem determinant_of_matrixA : Matrix.det matrixA = 144 := by
  sorry

end NUMINAMATH_GPT_determinant_of_matrixA_l198_19837


namespace NUMINAMATH_GPT_find_solutions_in_positive_integers_l198_19834

theorem find_solutions_in_positive_integers :
  ∃ a b c x y z : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
  a + b + c = x * y * z ∧ x + y + z = a * b * c ∧
  ((a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1) ∨
   (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 3 ∧ z = 1) ∨
   (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 5 ∧ y = 2 ∧ z = 1)) :=
sorry

end NUMINAMATH_GPT_find_solutions_in_positive_integers_l198_19834


namespace NUMINAMATH_GPT_minimum_box_value_l198_19875

theorem minimum_box_value :
  ∃ (a b : ℤ), a * b = 36 ∧ (a^2 + b^2 = 72 ∧ ∀ (a' b' : ℤ), a' * b' = 36 → a'^2 + b'^2 ≥ 72) :=
by
  sorry

end NUMINAMATH_GPT_minimum_box_value_l198_19875


namespace NUMINAMATH_GPT_probability_odd_divisor_25_factorial_l198_19823

theorem probability_odd_divisor_25_factorial : 
  let divisors := (22 + 1) * (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  (odd_divisors / divisors = 1 / 23) :=
sorry

end NUMINAMATH_GPT_probability_odd_divisor_25_factorial_l198_19823


namespace NUMINAMATH_GPT_remainder_of_87_pow_88_plus_7_l198_19861

theorem remainder_of_87_pow_88_plus_7 :
  (87^88 + 7) % 88 = 8 :=
by sorry

end NUMINAMATH_GPT_remainder_of_87_pow_88_plus_7_l198_19861


namespace NUMINAMATH_GPT_functional_eq_log_l198_19831

theorem functional_eq_log {f : ℝ → ℝ} (h₁ : f 4 = 2) 
                           (h₂ : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 * x2) = f x1 + f x2) : 
                           (∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2) := 
by
  sorry

end NUMINAMATH_GPT_functional_eq_log_l198_19831


namespace NUMINAMATH_GPT_find_divisor_l198_19864

theorem find_divisor (x d : ℕ) (h1 : x ≡ 7 [MOD d]) (h2 : (x + 11) ≡ 18 [MOD 31]) : d = 31 := 
sorry

end NUMINAMATH_GPT_find_divisor_l198_19864


namespace NUMINAMATH_GPT_remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l198_19878

theorem remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero (x : ℝ) :
  (x + 1) ^ 2025 % (x ^ 2 + 1) = 0 :=
  sorry

end NUMINAMATH_GPT_remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l198_19878


namespace NUMINAMATH_GPT_percentage_boys_playing_soccer_is_correct_l198_19887

-- Definition of conditions 
def total_students := 420
def boys := 312
def soccer_players := 250
def girls_not_playing_soccer := 73

-- Calculated values based on conditions
def girls := total_students - boys
def girls_playing_soccer := girls - girls_not_playing_soccer
def boys_playing_soccer := soccer_players - girls_playing_soccer

-- Percentage of boys playing soccer
def percentage_boys_playing_soccer := (boys_playing_soccer / soccer_players) * 100

-- We assert the percentage of boys playing soccer is 86%
theorem percentage_boys_playing_soccer_is_correct : percentage_boys_playing_soccer = 86 := 
by
  -- Placeholder proof (use sorry as the proof is not required)
  sorry

end NUMINAMATH_GPT_percentage_boys_playing_soccer_is_correct_l198_19887


namespace NUMINAMATH_GPT_smallest_integer_problem_l198_19860

theorem smallest_integer_problem (m : ℕ) (h1 : Nat.lcm 60 m / Nat.gcd 60 m = 28) : m = 105 := sorry

end NUMINAMATH_GPT_smallest_integer_problem_l198_19860


namespace NUMINAMATH_GPT_robin_hair_length_l198_19819

theorem robin_hair_length
  (l d g : ℕ)
  (h₁ : l = 16)
  (h₂ : d = 11)
  (h₃ : g = 12) :
  (l - d + g = 17) :=
by sorry

end NUMINAMATH_GPT_robin_hair_length_l198_19819


namespace NUMINAMATH_GPT_fraction_sum_l198_19845

namespace GeometricSequence

-- Given conditions in the problem
def q : ℕ := 2

-- Definition of the sum of the first n terms (S_n) of a geometric sequence
def S_n (a₁ : ℤ) (n : ℕ) : ℤ := 
  a₁ * (1 - q ^ n) / (1 - q)

-- Specific sum for the first 4 terms (S₄)
def S₄ (a₁ : ℤ) : ℤ := S_n a₁ 4

-- Define the 2nd term of the geometric sequence
def a₂ (a₁ : ℤ) : ℤ := a₁ * q

-- The statement to prove: $\dfrac{S_4}{a_2} = \dfrac{15}{2}$
theorem fraction_sum (a₁ : ℤ) : (S₄ a₁) / (a₂ a₁) = Rat.ofInt 15 / Rat.ofInt 2 :=
  by
  -- Implementation of proof will go here
  sorry

end GeometricSequence

end NUMINAMATH_GPT_fraction_sum_l198_19845


namespace NUMINAMATH_GPT_decreasing_intervals_sin_decreasing_intervals_log_cos_l198_19851

theorem decreasing_intervals_sin (k : ℤ) :
  ∀ x : ℝ, 
    ( (π / 2 + 2 * k * π < x) ∧ (x < 3 * π / 2 + 2 * k * π) ) ↔
    (∃ k : ℤ, (π / 2 + 2 * k * π < x) ∧ (x < 3 * π / 2 + 2 * k * π)) :=
sorry

theorem decreasing_intervals_log_cos (k : ℤ) :
  ∀ x : ℝ, 
    ( (2 * k * π < x) ∧ (x < π / 2 + 2 * k * π) ) ↔
    (∃ k : ℤ, (2 * k * π < x) ∧ (x < π / 2 + 2 * k * π)) :=
sorry

end NUMINAMATH_GPT_decreasing_intervals_sin_decreasing_intervals_log_cos_l198_19851
