import Mathlib

namespace NUMINAMATH_GPT_number_divisible_by_5_l1830_183074

theorem number_divisible_by_5 (A B C : ℕ) :
  (∃ (k1 k2 k3 k4 k5 k6 : ℕ), 3*10^6 + 10^5 + 7*10^4 + A*10^3 + B*10^2 + 4*10 + C = k1 ∧ 5 * k1 = 0 ∧
                          5 * k2 + 10 = 5 * k2 ∧ 5 * k3 + 5 = 5 * k3 ∧ 
                          5 * k4 + 3 = 5 * k4 ∧ 5 * k5 + 1 = 5 * k5 ∧ 
                          5 * k6 + 7 = 5 * k6) → C = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_divisible_by_5_l1830_183074


namespace NUMINAMATH_GPT_alice_has_largest_result_l1830_183023

def initial_number : ℕ := 15

def alice_transformation (x : ℕ) : ℕ := (x * 3 - 2 + 4)
def bob_transformation (x : ℕ) : ℕ := (x * 2 + 3 - 5)
def charlie_transformation (x : ℕ) : ℕ := (x + 5) / 2 * 4

def alice_final := alice_transformation initial_number
def bob_final := bob_transformation initial_number
def charlie_final := charlie_transformation initial_number

theorem alice_has_largest_result :
  alice_final > bob_final ∧ alice_final > charlie_final := by
  sorry

end NUMINAMATH_GPT_alice_has_largest_result_l1830_183023


namespace NUMINAMATH_GPT_vasya_most_points_anya_least_possible_l1830_183075

theorem vasya_most_points_anya_least_possible :
  ∃ (A B V : ℕ) (A_score B_score V_score : ℕ),
  A > B ∧ B > V ∧
  A_score = 9 ∧ B_score = 10 ∧ V_score = 11 ∧
  (∃ (words_common_AB words_common_AV words_only_B words_only_V : ℕ),
  words_common_AB = 6 ∧ words_common_AV = 3 ∧ words_only_B = 2 ∧ words_only_V = 4 ∧
  A = words_common_AB + words_common_AV ∧
  B = words_only_B + words_common_AB ∧
  V = words_only_V + words_common_AV ∧
  A_score = words_common_AB + words_common_AV ∧
  B_score = 2 * words_only_B + words_common_AB ∧
  V_score = 2 * words_only_V + words_common_AV) :=
sorry

end NUMINAMATH_GPT_vasya_most_points_anya_least_possible_l1830_183075


namespace NUMINAMATH_GPT_tangent_line_equation_l1830_183068

theorem tangent_line_equation (x y : ℝ) (h : y = x^3 + 1) (t : x = -1) :
  3*x - y + 3 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l1830_183068


namespace NUMINAMATH_GPT_maximum_sum_of_triplets_l1830_183034

-- Define a list representing a 9-digit number consisting of digits 1 to 9 in some order
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ ∀ n, n ∈ digits → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]
  
def sum_of_triplets (digits : List ℕ) : ℕ :=
  100 * digits[0]! + 10 * digits[1]! + digits[2]! +
  100 * digits[1]! + 10 * digits[2]! + digits[3]! +
  100 * digits[2]! + 10 * digits[3]! + digits[4]! +
  100 * digits[3]! + 10 * digits[4]! + digits[5]! +
  100 * digits[4]! + 10 * digits[5]! + digits[6]! +
  100 * digits[5]! + 10 * digits[6]! + digits[7]! +
  100 * digits[6]! + 10 * digits[7]! + digits[8]!

theorem maximum_sum_of_triplets :
  ∃ digits : List ℕ, valid_digits digits ∧ sum_of_triplets digits = 4648 :=
  sorry

end NUMINAMATH_GPT_maximum_sum_of_triplets_l1830_183034


namespace NUMINAMATH_GPT_angle_complement_l1830_183017

-- Conditions: The complement of angle A is 60 degrees
def complement (α : ℝ) : ℝ := 90 - α 

theorem angle_complement (A : ℝ) : complement A = 60 → A = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_complement_l1830_183017


namespace NUMINAMATH_GPT_rectangular_to_cylindrical_4_neg4_6_l1830_183069

theorem rectangular_to_cylindrical_4_neg4_6 :
  let x := 4
  let y := -4
  let z := 6
  let r := 4 * Real.sqrt 2
  let theta := (7 * Real.pi) / 4
  (r = Real.sqrt (x^2 + y^2)) ∧
  (Real.cos theta = x / r) ∧
  (Real.sin theta = y / r) ∧
  0 ≤ theta ∧ theta < 2 * Real.pi ∧
  z = 6 → 
  (r, theta, z) = (4 * Real.sqrt 2, (7 * Real.pi) / 4, 6) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_to_cylindrical_4_neg4_6_l1830_183069


namespace NUMINAMATH_GPT_history_percentage_l1830_183038

theorem history_percentage (H : ℕ) (math_percentage : ℕ := 72) (third_subject_percentage : ℕ := 69) (overall_average : ℕ := 75) :
  (math_percentage + H + third_subject_percentage) / 3 = overall_average → H = 84 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_history_percentage_l1830_183038


namespace NUMINAMATH_GPT_total_wax_required_l1830_183010

/-- Given conditions: -/
def wax_already_have : ℕ := 331
def wax_needed_more : ℕ := 22

/-- Prove the question (the total amount of wax required) -/
theorem total_wax_required :
  (wax_already_have + wax_needed_more) = 353 := by
  sorry

end NUMINAMATH_GPT_total_wax_required_l1830_183010


namespace NUMINAMATH_GPT_odd_powers_sum_divisible_by_p_l1830_183089

theorem odd_powers_sum_divisible_by_p
  (p : ℕ)
  (hp_prime : Prime p)
  (hp_gt_3 : 3 < p)
  (a b c d : ℕ)
  (h_sum : (a + b + c + d) % p = 0)
  (h_cube_sum : (a^3 + b^3 + c^3 + d^3) % p = 0)
  (n : ℕ)
  (hn_odd : n % 2 = 1 ) :
  (a^n + b^n + c^n + d^n) % p = 0 :=
sorry

end NUMINAMATH_GPT_odd_powers_sum_divisible_by_p_l1830_183089


namespace NUMINAMATH_GPT_value_of_m2_plus_3n2_l1830_183085

noncomputable def real_numbers_with_condition (m n : ℝ) : Prop :=
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0

theorem value_of_m2_plus_3n2 (m n : ℝ) (h : real_numbers_with_condition m n) : m^2 + 3*n^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m2_plus_3n2_l1830_183085


namespace NUMINAMATH_GPT_larger_number_l1830_183084

theorem larger_number (x y : ℕ) (h1 : x + y = 47) (h2 : x - y = 3) : max x y = 25 :=
sorry

end NUMINAMATH_GPT_larger_number_l1830_183084


namespace NUMINAMATH_GPT_sugar_total_more_than_two_l1830_183065

noncomputable def x (p q : ℝ) : ℝ :=
p / q

noncomputable def y (p q : ℝ) : ℝ :=
q / p

theorem sugar_total_more_than_two (p q : ℝ) (hpq : p ≠ q) :
  x p q + y p q > 2 :=
by sorry

end NUMINAMATH_GPT_sugar_total_more_than_two_l1830_183065


namespace NUMINAMATH_GPT_percent_y_of_x_l1830_183019

-- Definitions and assumptions based on the problem conditions
variables (x y : ℝ)
-- Given: 20% of (x - y) = 14% of (x + y)
axiom h : 0.20 * (x - y) = 0.14 * (x + y)

-- Prove that y is 0.1765 (or 17.65%) of x
theorem percent_y_of_x (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) : 
  y = 0.1765 * x :=
sorry

end NUMINAMATH_GPT_percent_y_of_x_l1830_183019


namespace NUMINAMATH_GPT_rabbits_ate_27_watermelons_l1830_183064

theorem rabbits_ate_27_watermelons
  (original_watermelons : ℕ)
  (watermelons_left : ℕ)
  (watermelons_eaten : ℕ)
  (h1 : original_watermelons = 35)
  (h2 : watermelons_left = 8)
  (h3 : original_watermelons - watermelons_left = watermelons_eaten) :
  watermelons_eaten = 27 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_rabbits_ate_27_watermelons_l1830_183064


namespace NUMINAMATH_GPT_max_area_triangle_l1830_183090

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) (h1 : Real.sqrt 2 * Real.sin A = Real.sqrt 3 * Real.cos A) (h2 : a = Real.sqrt 3) :
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / (8 * Real.sqrt 5) := 
sorry

end NUMINAMATH_GPT_max_area_triangle_l1830_183090


namespace NUMINAMATH_GPT_cost_of_tax_free_items_l1830_183078

-- Definitions based on the conditions.
def total_spending : ℝ := 20
def sales_tax_percentage : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- Derived calculations for intermediate variables for clarity
def taxable_items_cost : ℝ := total_spending * (1 - sales_tax_percentage)
def sales_tax_paid : ℝ := taxable_items_cost * tax_rate
def tax_free_items_cost : ℝ := total_spending - taxable_items_cost

-- Lean 4 statement for the problem
theorem cost_of_tax_free_items :
  tax_free_items_cost = 6 := by
    -- The proof would go here, but we are skipping it.
    sorry

end NUMINAMATH_GPT_cost_of_tax_free_items_l1830_183078


namespace NUMINAMATH_GPT_sum_of_abs_of_coefficients_l1830_183004

theorem sum_of_abs_of_coefficients :
  ∃ a_0 a_2 a_4 a_1 a_3 a_5 : ℤ, 
    ((2*x - 1)^5 + (x + 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) ∧
    (|a_0| + |a_2| + |a_4| = 110) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abs_of_coefficients_l1830_183004


namespace NUMINAMATH_GPT_remainder_76_pow_77_mod_7_l1830_183008

theorem remainder_76_pow_77_mod_7 : (76 ^ 77) % 7 = 6 := 
by 
  sorry 

end NUMINAMATH_GPT_remainder_76_pow_77_mod_7_l1830_183008


namespace NUMINAMATH_GPT_algebra_inequality_l1830_183002

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end NUMINAMATH_GPT_algebra_inequality_l1830_183002


namespace NUMINAMATH_GPT_xyz_squared_sum_l1830_183050

theorem xyz_squared_sum (x y z : ℝ)
  (h1 : x^2 + 6 * y = -17)
  (h2 : y^2 + 4 * z = 1)
  (h3 : z^2 + 2 * x = 2) :
  x^2 + y^2 + z^2 = 14 := 
sorry

end NUMINAMATH_GPT_xyz_squared_sum_l1830_183050


namespace NUMINAMATH_GPT_correct_triangle_set_l1830_183066

/-- Definition of triangle inequality -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Sets of lengths for checking the triangle inequality -/
def Set1 : ℝ × ℝ × ℝ := (5, 8, 2)
def Set2 : ℝ × ℝ × ℝ := (5, 8, 13)
def Set3 : ℝ × ℝ × ℝ := (5, 8, 5)
def Set4 : ℝ × ℝ × ℝ := (2, 7, 5)

/-- The correct set of lengths that can form a triangle according to the triangle inequality -/
theorem correct_triangle_set : satisfies_triangle_inequality 5 8 5 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_correct_triangle_set_l1830_183066


namespace NUMINAMATH_GPT_sequence_ratio_proof_l1830_183028

variable {a : ℕ → ℤ}

-- Sequence definition
axiom a₁ : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 4 * a n + 3

-- The theorem to be proved
theorem sequence_ratio_proof (n : ℕ) : (a (n + 1) + 1) / (a n + 1) = 4 := by
  sorry

end NUMINAMATH_GPT_sequence_ratio_proof_l1830_183028


namespace NUMINAMATH_GPT_discount_on_shoes_l1830_183036

theorem discount_on_shoes (x : ℝ) :
  let shoe_price := 200
  let shirt_price := 80
  let total_spent := 285
  let total_shirt_price := 2 * shirt_price
  let initial_total := shoe_price + total_shirt_price
  let disc_shoe_price := shoe_price - (shoe_price * x / 100)
  let pre_final_total := disc_shoe_price + total_shirt_price
  let final_total := pre_final_total * (1 - 0.05)
  final_total = total_spent → x = 30 :=
by
  intros shoe_price shirt_price total_spent total_shirt_price initial_total disc_shoe_price pre_final_total final_total h
  dsimp [shoe_price, shirt_price, total_spent, total_shirt_price, initial_total, disc_shoe_price, pre_final_total, final_total] at h
  -- Here, we would normally continue the proof, but we'll insert 'sorry' for now as instructed.
  sorry

end NUMINAMATH_GPT_discount_on_shoes_l1830_183036


namespace NUMINAMATH_GPT_simplify_expression_l1830_183088

variable (x : ℝ)

theorem simplify_expression : 
  2 * x^3 - (7 * x^2 - 9 * x) - 2 * (x^3 - 3 * x^2 + 4 * x) = -x^2 + x := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1830_183088


namespace NUMINAMATH_GPT_intersect_sets_l1830_183053

open Set

noncomputable def P : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}

theorem intersect_sets (U : Set ℝ) (P : Set ℝ) (Q : Set ℝ) :
  U = univ → P = {x : ℝ | x^2 - 2 * x ≤ 0} → Q = {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x} →
  P ∩ Q = Icc (0 : ℝ) (2 : ℝ) :=
by
  intros
  sorry

end NUMINAMATH_GPT_intersect_sets_l1830_183053


namespace NUMINAMATH_GPT_divides_8x_7y_l1830_183012

theorem divides_8x_7y (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end NUMINAMATH_GPT_divides_8x_7y_l1830_183012


namespace NUMINAMATH_GPT_find_x_l1830_183029

def x_y_conditions (x y : ℝ) : Prop :=
  x > y ∧
  x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40 ∧
  x * y + x + y = 8

theorem find_x (x y : ℝ) (h : x_y_conditions x y) : x = 3 + Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1830_183029


namespace NUMINAMATH_GPT_cannot_determine_number_of_pens_l1830_183094

theorem cannot_determine_number_of_pens 
  (P : ℚ) -- marked price of one pen
  (N : ℕ) -- number of pens = 46
  (discount : ℚ := 0.01) -- 1% discount
  (profit_percent : ℚ := 11.91304347826087) -- given profit percent
  : ¬ ∃ (N : ℕ), 
        profit_percent = ((N * P * (1 - discount) - N * P) / (N * P)) * 100 :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_number_of_pens_l1830_183094


namespace NUMINAMATH_GPT_solve_inequality_l1830_183007

theorem solve_inequality (a x : ℝ) (h : a < 0) :
  (56 * x^2 + a * x - a^2 < 0) ↔ (a / 8 < x ∧ x < -a / 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1830_183007


namespace NUMINAMATH_GPT_crystal_run_final_segment_length_l1830_183051

theorem crystal_run_final_segment_length :
  let north_distance := 2
  let southeast_leg := 1 / Real.sqrt 2
  let southeast_movement_north := -southeast_leg
  let southeast_movement_east := southeast_leg
  let northeast_leg := 2 / Real.sqrt 2
  let northeast_movement_north := northeast_leg
  let northeast_movement_east := northeast_leg
  let total_north_movement := north_distance + northeast_movement_north + southeast_movement_north
  let total_east_movement := southeast_movement_east + northeast_movement_east
  total_north_movement = 2.5 ∧ 
  total_east_movement = 3 * Real.sqrt 2 / 2 ∧ 
  Real.sqrt (total_north_movement^2 + total_east_movement^2) = Real.sqrt 10.75 :=
by
  sorry

end NUMINAMATH_GPT_crystal_run_final_segment_length_l1830_183051


namespace NUMINAMATH_GPT_Karlson_drink_ratio_l1830_183043

noncomputable def conical_glass_volume_ratio (r h : ℝ) : Prop :=
  let V_fuzh := (1 / 3) * Real.pi * r^2 * h
  let V_Mal := (1 / 8) * V_fuzh
  let V_Karlsson := V_fuzh - V_Mal
  (V_Karlsson / V_Mal) = 7

theorem Karlson_drink_ratio (r h : ℝ) : conical_glass_volume_ratio r h := sorry

end NUMINAMATH_GPT_Karlson_drink_ratio_l1830_183043


namespace NUMINAMATH_GPT_range_of_a_l1830_183063

-- Given conditions
def condition1 (x : ℝ) := (4 + x) / 3 > (x + 2) / 2
def condition2 (x : ℝ) (a : ℝ) := (x + a) / 2 < 0

-- The statement to prove
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, condition1 x → condition2 x a → x < 2) → a ≤ -2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1830_183063


namespace NUMINAMATH_GPT_max_xy_l1830_183052

-- Lean statement for the given problem
theorem max_xy (x y : ℝ) (h : x^2 + y^2 = 4) : xy ≤ 2 := sorry

end NUMINAMATH_GPT_max_xy_l1830_183052


namespace NUMINAMATH_GPT_find_real_pairs_l1830_183062

theorem find_real_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_real_pairs_l1830_183062


namespace NUMINAMATH_GPT_charley_initial_pencils_l1830_183098

theorem charley_initial_pencils (P : ℕ) (lost_initially : P - 6 = (P - 1/3 * (P - 6) - 6)) (current_pencils : P - 1/3 * (P - 6) - 6 = 16) : P = 30 := 
sorry

end NUMINAMATH_GPT_charley_initial_pencils_l1830_183098


namespace NUMINAMATH_GPT_solution_l1830_183026

noncomputable def problem_statement : Prop :=
  ∃ (A B C D : ℝ) (a b : ℝ) (x : ℝ), 
    (|A - B| = 3) ∧
    (|A - C| = 1) ∧
    (A = Real.pi / 2) ∧  -- This typically signifies angle A is 90 degrees.
    (a > 0) ∧
    (b > 0) ∧
    (a = 1) ∧
    (|A - D| = x) ∧
    (|B - D| = 3 - x) ∧
    (|C - D| = Real.sqrt (x^2 + 1)) ∧
    (Real.sqrt (x^2 + 1) - (3 - x) = 2) ∧
    (|A - D| / |B - D| = 4)

theorem solution : problem_statement :=
sorry

end NUMINAMATH_GPT_solution_l1830_183026


namespace NUMINAMATH_GPT_product_of_positive_solutions_l1830_183079

theorem product_of_positive_solutions :
  ∃ n : ℕ, ∃ p : ℕ, Prime p ∧ (n^2 - 41*n + 408 = p) ∧ (∀ m : ℕ, (Prime p ∧ (m^2 - 41*m + 408 = p)) → m = n) ∧ (n = 406) := 
sorry

end NUMINAMATH_GPT_product_of_positive_solutions_l1830_183079


namespace NUMINAMATH_GPT_john_total_animals_is_114_l1830_183054

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end NUMINAMATH_GPT_john_total_animals_is_114_l1830_183054


namespace NUMINAMATH_GPT_carrie_bought_tshirts_l1830_183016

variable (cost_per_tshirt : ℝ) (total_spent : ℝ)

theorem carrie_bought_tshirts (h1 : cost_per_tshirt = 9.95) (h2 : total_spent = 248) :
  ⌊total_spent / cost_per_tshirt⌋ = 24 :=
by
  sorry

end NUMINAMATH_GPT_carrie_bought_tshirts_l1830_183016


namespace NUMINAMATH_GPT_range_of_m_l1830_183056

-- Condition p: The solution set of the inequality x² + mx + 1 < 0 is an empty set
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Condition q: The function y = 4x² + 4(m-1)x + 3 has no extreme value
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 12 * x^2 + 4 * (m - 1) ≥ 0

-- Combined condition: "p or q" is true and "p and q" is false
def combined_condition (m : ℝ) : Prop :=
  (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- The range of values for the real number m
theorem range_of_m (m : ℝ) : combined_condition m → (-2 ≤ m ∧ m < 1) ∨ m > 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1830_183056


namespace NUMINAMATH_GPT_find_coordinates_of_point_M_l1830_183032

theorem find_coordinates_of_point_M :
  ∃ (M : ℝ × ℝ), 
    (M.1 > 0) ∧ (M.2 < 0) ∧ 
    abs M.2 = 12 ∧ 
    abs M.1 = 4 ∧ 
    M = (4, -12) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_point_M_l1830_183032


namespace NUMINAMATH_GPT_unique_positive_a_for_one_solution_l1830_183014

theorem unique_positive_a_for_one_solution :
  ∃ (d : ℝ), d ≠ 0 ∧ (∀ a : ℝ, a > 0 → (∀ x : ℝ, x^2 + (a + 1/a) * x + d = 0 ↔ x^2 + (a + 1/a) * x + d = 0)) ∧ d = 1 := 
by
  sorry

end NUMINAMATH_GPT_unique_positive_a_for_one_solution_l1830_183014


namespace NUMINAMATH_GPT_floor_plus_self_eq_l1830_183025

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end NUMINAMATH_GPT_floor_plus_self_eq_l1830_183025


namespace NUMINAMATH_GPT_sqrt_difference_eq_neg_six_sqrt_two_l1830_183020

theorem sqrt_difference_eq_neg_six_sqrt_two :
  (Real.sqrt ((5 - 3 * Real.sqrt 2)^2)) - (Real.sqrt ((5 + 3 * Real.sqrt 2)^2)) = -6 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_sqrt_difference_eq_neg_six_sqrt_two_l1830_183020


namespace NUMINAMATH_GPT_find_m_plus_n_l1830_183031

def probability_no_exact_k_pairs (k n : ℕ) : ℚ :=
  -- A function to calculate the probability
  -- Placeholder definition (details omitted for brevity)
  sorry

theorem find_m_plus_n : ∃ m n : ℕ,
  gcd m n = 1 ∧ 
  (probability_no_exact_k_pairs k n = (97 / 1000) → m + n = 1097) :=
sorry

end NUMINAMATH_GPT_find_m_plus_n_l1830_183031


namespace NUMINAMATH_GPT_percentage_donated_to_orphan_house_l1830_183018

-- Given conditions as definitions in Lean 4
def income : ℝ := 400000
def children_percentage : ℝ := 0.2
def children_count : ℕ := 3
def wife_percentage : ℝ := 0.25
def remaining_after_donation : ℝ := 40000

-- Define the problem as a theorem
theorem percentage_donated_to_orphan_house :
  (children_count * children_percentage + wife_percentage) * income = 0.85 * income →
  (income - 0.85 * income = 60000) →
  remaining_after_donation = 40000 →
  (100 * (60000 - remaining_after_donation) / 60000) = 33.33 := 
by
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_percentage_donated_to_orphan_house_l1830_183018


namespace NUMINAMATH_GPT_complete_the_square_l1830_183027

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x^2 + 10 * x - 3 = 0) → ((x + a)^2 = b) ∧ b = 28) :=
sorry

end NUMINAMATH_GPT_complete_the_square_l1830_183027


namespace NUMINAMATH_GPT_cosine_of_angle_in_convex_quadrilateral_l1830_183040

theorem cosine_of_angle_in_convex_quadrilateral
    (A C : ℝ)
    (AB CD AD BC : ℝ)
    (h1 : A = C)
    (h2 : AB = 150)
    (h3 : CD = 150)
    (h4 : AD = BC)
    (h5 : AB + BC + CD + AD = 580) :
    Real.cos A = 7 / 15 := 
  sorry

end NUMINAMATH_GPT_cosine_of_angle_in_convex_quadrilateral_l1830_183040


namespace NUMINAMATH_GPT_max_sequence_value_l1830_183077

theorem max_sequence_value : 
  ∃ n ∈ (Set.univ : Set ℤ), (∀ m ∈ (Set.univ : Set ℤ), -m^2 + 15 * m + 3 ≤ -n^2 + 15 * n + 3) ∧ (-n^2 + 15 * n + 3 = 59) :=
by
  sorry

end NUMINAMATH_GPT_max_sequence_value_l1830_183077


namespace NUMINAMATH_GPT_eq1_solution_eq2_no_solution_l1830_183095

theorem eq1_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2) :
  (2/x + 1/(x*(x-2)) = 5/(2*x)) ↔ x = 4 :=
by sorry

theorem eq2_no_solution (x : ℝ) (h : x ≠ 2) :
  (5*x - 4)/ (x - 2) = (4*x + 10) / (3*x - 6) - 1 ↔ false :=
by sorry

end NUMINAMATH_GPT_eq1_solution_eq2_no_solution_l1830_183095


namespace NUMINAMATH_GPT_tangent_line_equation_l1830_183003

theorem tangent_line_equation (a : ℝ) (h : a ≠ 0) :
  (∃ b : ℝ, b = 2 ∧ (∀ x : ℝ, y = a * x^2) ∧ y - a = b * (x - 1)) → 
  ∃ (x y : ℝ), 2 * x - y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1830_183003


namespace NUMINAMATH_GPT_intersect_sets_l1830_183076

   variable (P : Set ℕ) (Q : Set ℕ)

   -- Definitions based on given conditions
   def P_def : Set ℕ := {1, 3, 5}
   def Q_def : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

   -- Theorem statement in Lean 4
   theorem intersect_sets :
     P = P_def → Q = Q_def → P ∩ Q = {3, 5} :=
   by
     sorry
   
end NUMINAMATH_GPT_intersect_sets_l1830_183076


namespace NUMINAMATH_GPT_basketball_tournament_l1830_183059

theorem basketball_tournament (x : ℕ) 
  (h1 : ∀ n, ((n * (n - 1)) / 2) = 28 -> n = x) 
  (h2 : (x * (x - 1)) / 2 = 28) : 
  (1 / 2 : ℚ) * x * (x - 1) = 28 :=
by 
  sorry

end NUMINAMATH_GPT_basketball_tournament_l1830_183059


namespace NUMINAMATH_GPT_evaluate_expression_l1830_183081

theorem evaluate_expression : (3^3)^4 = 531441 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1830_183081


namespace NUMINAMATH_GPT_train_length_problem_l1830_183037

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end NUMINAMATH_GPT_train_length_problem_l1830_183037


namespace NUMINAMATH_GPT_towel_bleach_percentage_decrease_l1830_183087

-- Define the problem
theorem towel_bleach_percentage_decrease (L B : ℝ) (x : ℝ) (h_length : 0 < L) (h_breadth : 0 < B) 
  (h1 : 0.64 * L * B = 0.8 * L * (1 - x / 100) * B) :
  x = 20 :=
by
  -- The actual proof is not needed, providing "sorry" as a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_towel_bleach_percentage_decrease_l1830_183087


namespace NUMINAMATH_GPT_min_value_expression_71_l1830_183058

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5)

theorem min_value_expression_71 (x y : ℝ) (hx : x > 4) (hy : y > 5) : 
  min_value_expression x y ≥ 71 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_71_l1830_183058


namespace NUMINAMATH_GPT_original_ratio_white_yellow_l1830_183005

-- Define the given conditions
variables (W Y : ℕ)
axiom total_balls : W + Y = 64
axiom erroneous_dispatch : W = 8 * (Y + 20) / 13

-- The theorem we need to prove
theorem original_ratio_white_yellow (W Y : ℕ) (h1 : W + Y = 64) (h2 : W = 8 * (Y + 20) / 13) : W = Y :=
by sorry

end NUMINAMATH_GPT_original_ratio_white_yellow_l1830_183005


namespace NUMINAMATH_GPT_women_attended_l1830_183046

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end NUMINAMATH_GPT_women_attended_l1830_183046


namespace NUMINAMATH_GPT_greatest_possible_x_l1830_183086

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end NUMINAMATH_GPT_greatest_possible_x_l1830_183086


namespace NUMINAMATH_GPT_select_k_plus_1_nums_divisible_by_n_l1830_183057

theorem select_k_plus_1_nums_divisible_by_n (n k : ℕ) (hn : n > 0) (hk : k > 0) (nums : Fin (n + k) → ℕ) :
  ∃ (indices : Finset (Fin (n + k))), indices.card ≥ k + 1 ∧ (indices.sum (nums ∘ id)) % n = 0 :=
sorry

end NUMINAMATH_GPT_select_k_plus_1_nums_divisible_by_n_l1830_183057


namespace NUMINAMATH_GPT_parking_lot_capacity_l1830_183072

-- Definitions based on the conditions
def levels : ℕ := 5
def parkedCars : ℕ := 23
def moreCars : ℕ := 62
def capacityPerLevel : ℕ := parkedCars + moreCars

-- Proof problem statement
theorem parking_lot_capacity : levels * capacityPerLevel = 425 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_parking_lot_capacity_l1830_183072


namespace NUMINAMATH_GPT_gcd_390_455_546_l1830_183009

theorem gcd_390_455_546 : Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
by
  sorry    -- this indicates the proof is not included

end NUMINAMATH_GPT_gcd_390_455_546_l1830_183009


namespace NUMINAMATH_GPT_inequality_lemma_l1830_183092

theorem inequality_lemma (a b c d : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) (hd : 0 < d ∧ d < 1) :
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d :=
by 
  sorry

end NUMINAMATH_GPT_inequality_lemma_l1830_183092


namespace NUMINAMATH_GPT_point_of_tangency_l1830_183091

noncomputable def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 10 * x + 14
noncomputable def parabola2 (y : ℝ) : ℝ := 4 * y^2 + 16 * y + 68

theorem point_of_tangency : 
  ∃ (x y : ℝ), parabola1 x = y ∧ parabola2 y = x ∧ x = -9/4 ∧ y = -15/8 :=
by
  -- The proof will show that the point of tangency is (-9/4, -15/8)
  sorry

end NUMINAMATH_GPT_point_of_tangency_l1830_183091


namespace NUMINAMATH_GPT_students_neither_math_nor_physics_l1830_183099

theorem students_neither_math_nor_physics :
  let total_students := 150
  let students_math := 80
  let students_physics := 60
  let students_both := 20
  total_students - (students_math - students_both + students_physics - students_both + students_both) = 30 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_math_nor_physics_l1830_183099


namespace NUMINAMATH_GPT_choir_members_max_l1830_183006

theorem choir_members_max (x r m : ℕ) 
  (h1 : r * x + 3 = m)
  (h2 : (r - 3) * (x + 2) = m) 
  (h3 : m < 150) : 
  m = 759 :=
sorry

end NUMINAMATH_GPT_choir_members_max_l1830_183006


namespace NUMINAMATH_GPT_C_D_meeting_time_l1830_183033

-- Defining the conditions.
variables (A B C D : Type) [LinearOrderedField A] (V_A V_B V_C V_D : A)
variables (startTime meet_AC meet_BD meet_AB meet_CD : A)

-- Cars' initial meeting conditions
axiom init_cond : startTime = 0
axiom meet_cond_AC : meet_AC = 7
axiom meet_cond_BD : meet_BD = 7
axiom meet_cond_AB : meet_AB = 53
axiom speed_relation : V_A + V_C = V_B + V_D ∧ V_A - V_B = V_D - V_C

-- The problem asks for the meeting time of C and D
theorem C_D_meeting_time : meet_CD = 53 :=
by sorry

end NUMINAMATH_GPT_C_D_meeting_time_l1830_183033


namespace NUMINAMATH_GPT_ratio_of_diamonds_to_spades_l1830_183013

-- Given conditions
variable (total_cards : Nat := 13)
variable (black_cards : Nat := 7)
variable (red_cards : Nat := 6)
variable (clubs : Nat := 6)
variable (diamonds : Nat)
variable (spades : Nat)
variable (hearts : Nat := 2 * diamonds)
variable (cards_distribution : clubs + diamonds + hearts + spades = total_cards)
variable (black_distribution : clubs + spades = black_cards)

-- Define the proof theorem
theorem ratio_of_diamonds_to_spades : (diamonds / spades : ℝ) = 2 :=
 by
  -- temporarily we insert sorry to skip the proof
  sorry

end NUMINAMATH_GPT_ratio_of_diamonds_to_spades_l1830_183013


namespace NUMINAMATH_GPT_cafeteria_apples_l1830_183044

theorem cafeteria_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ) 
(h1: handed_out = 27) (h2: pies = 5) (h3: apples_per_pie = 4) : handed_out + pies * apples_per_pie = 47 :=
by
  -- The proof will be provided here if needed
  sorry

end NUMINAMATH_GPT_cafeteria_apples_l1830_183044


namespace NUMINAMATH_GPT_max_n_intersection_non_empty_l1830_183021

-- Define the set An
def An (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- State the theorem
theorem max_n_intersection_non_empty : 
  ∃ x, (∀ n, n ≤ 4 → x ∈ An n) ∧ (∀ n, n > 4 → x ∉ An n) :=
by
  sorry

end NUMINAMATH_GPT_max_n_intersection_non_empty_l1830_183021


namespace NUMINAMATH_GPT_avg_cards_removed_until_prime_l1830_183097

theorem avg_cards_removed_until_prime:
  let prime_count := 13
  let cards_count := 42
  let non_prime_count := cards_count - prime_count
  let groups_count := prime_count + 1
  let avg_non_prime_per_group := (non_prime_count: ℚ) / (groups_count: ℚ)
  (groups_count: ℚ) > 0 →
  avg_non_prime_per_group + 1 = (43: ℚ) / (14: ℚ) :=
by
  sorry

end NUMINAMATH_GPT_avg_cards_removed_until_prime_l1830_183097


namespace NUMINAMATH_GPT_smallest_four_digit_in_pascals_triangle_l1830_183039

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_in_pascals_triangle_l1830_183039


namespace NUMINAMATH_GPT_probability_of_fx_leq_zero_is_3_over_10_l1830_183083

noncomputable def fx (x : ℝ) : ℝ := -x + 2

def in_interval (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

def probability_fx_leq_zero : ℚ :=
  let interval_start := -5
  let interval_end := 5
  let fx_leq_zero_start := 2
  let fx_leq_zero_end := 5
  (fx_leq_zero_end - fx_leq_zero_start) / (interval_end - interval_start)

theorem probability_of_fx_leq_zero_is_3_over_10 :
  probability_fx_leq_zero = 3 / 10 :=
sorry

end NUMINAMATH_GPT_probability_of_fx_leq_zero_is_3_over_10_l1830_183083


namespace NUMINAMATH_GPT_five_digit_numbers_last_two_different_l1830_183082

def total_five_digit_numbers : ℕ := 90000

def five_digit_numbers_last_two_same : ℕ := 9000

theorem five_digit_numbers_last_two_different :
  (total_five_digit_numbers - five_digit_numbers_last_two_same) = 81000 := 
by 
  sorry

end NUMINAMATH_GPT_five_digit_numbers_last_two_different_l1830_183082


namespace NUMINAMATH_GPT_melted_ice_cream_depth_l1830_183001

noncomputable def ice_cream_depth : ℝ :=
  let r1 := 3 -- radius of the sphere
  let r2 := 10 -- radius of the cylinder
  let V_sphere := (4/3) * Real.pi * r1^3 -- volume of the sphere
  let V_cylinder h := Real.pi * r2^2 * h -- volume of the cylinder
  V_sphere / (Real.pi * r2^2)

theorem melted_ice_cream_depth :
  ice_cream_depth = 9 / 25 :=
by
  sorry

end NUMINAMATH_GPT_melted_ice_cream_depth_l1830_183001


namespace NUMINAMATH_GPT_evaluate_expression_l1830_183030

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a ^ a - a * (a - 2) ^ a) ^ (a + 1) = 14889702426 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1830_183030


namespace NUMINAMATH_GPT_value_of_a6_l1830_183071

noncomputable def Sn (n : ℕ) : ℕ := n * 2^(n + 1)
noncomputable def an (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem value_of_a6 : an 6 = 448 := by
  sorry

end NUMINAMATH_GPT_value_of_a6_l1830_183071


namespace NUMINAMATH_GPT_find_primes_l1830_183096

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end NUMINAMATH_GPT_find_primes_l1830_183096


namespace NUMINAMATH_GPT_unique_triangle_solution_l1830_183024

noncomputable def triangle_solutions (a b A : ℝ) : ℕ :=
sorry -- Placeholder for actual function calculating number of solutions

theorem unique_triangle_solution : triangle_solutions 30 25 150 = 1 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_unique_triangle_solution_l1830_183024


namespace NUMINAMATH_GPT_smallest_n_with_290_trailing_zeros_in_factorial_l1830_183000

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 5^2) + (n / 5^3) + (n / 5^4) + (n / 5^5) + (n / 5^6) -- sum until the division becomes zero

theorem smallest_n_with_290_trailing_zeros_in_factorial : 
  ∀ (n : ℕ), n >= 1170 ↔ trailing_zeros n >= 290 ∧ trailing_zeros (n-1) < 290 := 
by { sorry }

end NUMINAMATH_GPT_smallest_n_with_290_trailing_zeros_in_factorial_l1830_183000


namespace NUMINAMATH_GPT_correct_transformation_l1830_183035

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end NUMINAMATH_GPT_correct_transformation_l1830_183035


namespace NUMINAMATH_GPT_range_of_t_l1830_183093
noncomputable def f (x : ℝ) (t : ℝ) : ℝ := Real.exp (2 * x) - t
noncomputable def g (x : ℝ) (t : ℝ) : ℝ := t * Real.exp x - 1

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x t ≥ g x t) ↔ t ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

end NUMINAMATH_GPT_range_of_t_l1830_183093


namespace NUMINAMATH_GPT_carrie_profit_l1830_183045

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end NUMINAMATH_GPT_carrie_profit_l1830_183045


namespace NUMINAMATH_GPT_grant_room_proof_l1830_183041

/-- Danielle's apartment has 6 rooms -/
def danielle_rooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment -/
def heidi_rooms : ℕ := 3 * danielle_rooms

/-- Jenny's apartment has 5 more rooms than Danielle's apartment -/
def jenny_rooms : ℕ := danielle_rooms + 5

/-- Lina's apartment has 7 rooms -/
def lina_rooms : ℕ := 7

/-- The total number of rooms from Danielle, Heidi, Jenny,
    and Lina's apartments -/
def total_rooms : ℕ := danielle_rooms + heidi_rooms + jenny_rooms + lina_rooms

/-- Grant's apartment has 1/3 less rooms than 1/9 of the
    combined total of rooms from Danielle's, Heidi's, Jenny's, and Lina's apartments -/
def grant_rooms : ℕ := (total_rooms / 9) - (total_rooms / 9) / 3

/-- Prove that Grant's apartment has 3 rooms -/
theorem grant_room_proof : grant_rooms = 3 :=
by
  sorry

end NUMINAMATH_GPT_grant_room_proof_l1830_183041


namespace NUMINAMATH_GPT_find_ellipse_l1830_183047

-- Define the ellipse and conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the focus points
def focus (a b c : ℝ) : Prop :=
  c^2 = a^2 - b^2

-- Define the range condition
def range_condition (a b c : ℝ) : Prop :=
  let min_val := b^2 - c^2;
  let max_val := a^2 - c^2;
  min_val = -3 ∧ max_val = 3

-- Prove the equation of the ellipse
theorem find_ellipse (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (ellipse a b a_pos b_pos ∧ focus a b c ∧ range_condition a b c) →
  (a^2 = 9 ∧ b^2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_ellipse_l1830_183047


namespace NUMINAMATH_GPT_inscribed_circle_radius_square_l1830_183060

theorem inscribed_circle_radius_square (ER RF GS SH : ℝ) (r : ℝ) 
  (hER : ER = 23) (hRF : RF = 34) (hGS : GS = 42) (hSH : SH = 28)
  (h_tangent : ∀ t, t = r * r * (70 * t - 87953)) :
  r^2 = 87953 / 70 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_square_l1830_183060


namespace NUMINAMATH_GPT_number_of_good_weeks_l1830_183022

-- Definitions from conditions
def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def tough_weeks : ℕ := 3
def total_money_made : ℕ := 10400
def total_tough_week_sales : ℕ := tough_weeks * tough_week_sales
def total_good_week_sales : ℕ := total_money_made - total_tough_week_sales

-- Question to be proven
theorem number_of_good_weeks (G : ℕ) : 
  (total_good_week_sales = G * good_week_sales) → G = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_good_weeks_l1830_183022


namespace NUMINAMATH_GPT_apple_slices_per_group_l1830_183067

-- defining the conditions
variables (a g : ℕ)

-- 1. Equal number of apple slices and grapes in groups
def equal_group (a g : ℕ) : Prop := a = g

-- 2. Grapes packed in groups of 9
def grapes_groups_of_9 (g : ℕ) : Prop := ∃ k : ℕ, g = 9 * k

-- 3. Smallest number of grapes is 18
def smallest_grapes (g : ℕ) : Prop := g = 18

-- theorem stating that the number of apple slices per group is 9
theorem apple_slices_per_group : equal_group a g ∧ grapes_groups_of_9 g ∧ smallest_grapes g → a = 9 := by
  sorry

end NUMINAMATH_GPT_apple_slices_per_group_l1830_183067


namespace NUMINAMATH_GPT_books_arrangement_count_l1830_183042

noncomputable def arrangement_of_books : ℕ :=
  let total_books := 5
  let identical_books := 2
  Nat.factorial total_books / Nat.factorial identical_books

theorem books_arrangement_count : arrangement_of_books = 60 := by
  sorry

end NUMINAMATH_GPT_books_arrangement_count_l1830_183042


namespace NUMINAMATH_GPT_cubed_expression_value_l1830_183073

open Real

theorem cubed_expression_value (a b c : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + 2 * c = 0) :
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) :=
  sorry

end NUMINAMATH_GPT_cubed_expression_value_l1830_183073


namespace NUMINAMATH_GPT_max_stickers_l1830_183011

theorem max_stickers (n_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) 
  (total_players : n_players = 22) 
  (average : avg_stickers = 4) 
  (minimum : ∀ i, i < n_players → min_stickers = 1) :
  ∃ max_sticker : ℕ, max_sticker = 67 :=
by
  sorry

end NUMINAMATH_GPT_max_stickers_l1830_183011


namespace NUMINAMATH_GPT_least_value_x_y_z_l1830_183055

theorem least_value_x_y_z 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h_eq: 2 * x = 5 * y) 
  (h_eq': 5 * y = 8 * z) : 
  x + y + z = 33 :=
by 
  sorry

end NUMINAMATH_GPT_least_value_x_y_z_l1830_183055


namespace NUMINAMATH_GPT_alpha_beta_inequality_l1830_183061

theorem alpha_beta_inequality (α β : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x^α * y^β < k * (x + y)) ↔ (0 ≤ α ∧ 0 ≤ β ∧ α + β = 1) :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_inequality_l1830_183061


namespace NUMINAMATH_GPT_increase_80_by_150_percent_l1830_183015

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end NUMINAMATH_GPT_increase_80_by_150_percent_l1830_183015


namespace NUMINAMATH_GPT_equation_of_line_is_correct_l1830_183049

/-! Given the circle x^2 + y^2 + 2x - 4y + a = 0 with a < 3 and the midpoint of the chord AB as C(-2, 3), prove that the equation of the line l that intersects the circle at points A and B is x - y + 5 = 0. -/

theorem equation_of_line_is_correct (a : ℝ) (h : a < 3) :
  ∃ l : ℝ × ℝ × ℝ, (l = (1, -1, 5)) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0 → 
    (x - y + 5 = 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_line_is_correct_l1830_183049


namespace NUMINAMATH_GPT_coordinates_of_P_l1830_183048

-- Define a structure for a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Define the distance from a point to the x-axis
def distance_to_x_axis (P : Point) : ℝ :=
  |P.y|

-- Define the distance from a point to the y-axis
def distance_to_y_axis (P : Point) : ℝ :=
  |P.x|

-- The main proof statement
theorem coordinates_of_P (P : Point) :
  in_third_quadrant P →
  distance_to_x_axis P = 2 →
  distance_to_y_axis P = 5 →
  P = { x := -5, y := -2 } :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l1830_183048


namespace NUMINAMATH_GPT_width_decrease_percentage_l1830_183070

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end NUMINAMATH_GPT_width_decrease_percentage_l1830_183070


namespace NUMINAMATH_GPT_find_triplets_l1830_183080

theorem find_triplets (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : Nat.Prime p) (h_eq : (a + b)^p = p^a + p^b) : (a = 1 ∧ b = 1 ∧ p = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_l1830_183080
