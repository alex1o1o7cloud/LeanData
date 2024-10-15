import Mathlib

namespace NUMINAMATH_GPT_cars_fell_in_lot_l2320_232046

theorem cars_fell_in_lot (initial_cars went_out_cars came_in_cars final_cars: ℕ) (h1 : initial_cars = 25) 
    (h2 : went_out_cars = 18) (h3 : came_in_cars = 12) (h4 : final_cars = initial_cars - went_out_cars + came_in_cars) :
    initial_cars - final_cars = 6 :=
    sorry

end NUMINAMATH_GPT_cars_fell_in_lot_l2320_232046


namespace NUMINAMATH_GPT_simplify_expression_l2320_232061

theorem simplify_expression (x y : ℝ) (m : ℤ) : 
  ((x + y)^(2 * m + 1) / (x + y)^(m - 1) = (x + y)^(m + 2)) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2320_232061


namespace NUMINAMATH_GPT_least_number_to_subtract_l2320_232030

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (r : ℕ) (h : n = 3674958423) (div : k = 47) (rem : r = 30) :
  (n % k = r) → 3674958423 % 47 = 30 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l2320_232030


namespace NUMINAMATH_GPT_solution_range_for_m_l2320_232026

theorem solution_range_for_m (x m : ℝ) (h₁ : 2 * x - 1 > 3 * (x - 2)) (h₂ : x < m) : m ≥ 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_range_for_m_l2320_232026


namespace NUMINAMATH_GPT_roots_sum_eq_product_l2320_232097

theorem roots_sum_eq_product (m : ℝ) :
  (∀ x : ℝ, 2 * (x - 1) * (x - 3 * m) = x * (m - 4)) →
  (∀ a b : ℝ, 2 * a * b = 2 * (5 * m + 6) / -2 ∧ 2 * a * b = 6 * m / 2) →
  m = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_roots_sum_eq_product_l2320_232097


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l2320_232044

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * Real.pi - 5 * Real.pi / 12 <= x ∧ x <= k * Real.pi + Real.pi / 12) →
    (∃ r : ℝ, f (x + r) > f x ∨ f (x + r) < f x) := by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l2320_232044


namespace NUMINAMATH_GPT_four_digit_multiples_of_5_count_l2320_232015

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end NUMINAMATH_GPT_four_digit_multiples_of_5_count_l2320_232015


namespace NUMINAMATH_GPT_solve_nat_numbers_equation_l2320_232075

theorem solve_nat_numbers_equation (n k l m : ℕ) (h_l : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n = 2) ∧ (k = 1) ∧ (l = 2) ∧ (m = 3) := 
by
  sorry

end NUMINAMATH_GPT_solve_nat_numbers_equation_l2320_232075


namespace NUMINAMATH_GPT_days_in_april_l2320_232031

-- Hannah harvests 5 strawberries daily for the whole month of April.
def harvest_per_day : ℕ := 5
-- She gives away 20 strawberries.
def strawberries_given_away : ℕ := 20
-- 30 strawberries are stolen.
def strawberries_stolen : ℕ := 30
-- She has 100 strawberries by the end of April.
def strawberries_final : ℕ := 100

theorem days_in_april : 
  ∃ (days : ℕ), (days * harvest_per_day = strawberries_final + strawberries_given_away + strawberries_stolen) :=
by
  sorry

end NUMINAMATH_GPT_days_in_april_l2320_232031


namespace NUMINAMATH_GPT_dice_sum_not_22_l2320_232019

theorem dice_sum_not_22 (a b c d e : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 6) (h₁ : 1 ≤ b ∧ b ≤ 6)
  (h₂ : 1 ≤ c ∧ c ≤ 6) (h₃ : 1 ≤ d ∧ d ≤ 6) (h₄ : 1 ≤ e ∧ e ≤ 6) 
  (h₅ : a * b * c * d * e = 432) : a + b + c + d + e ≠ 22 :=
sorry

end NUMINAMATH_GPT_dice_sum_not_22_l2320_232019


namespace NUMINAMATH_GPT_variance_of_data_set_l2320_232092

theorem variance_of_data_set :
  let data_set := [2, 3, 4, 5, 6]
  let mean := (2 + 3 + 4 + 5 + 6) / 5
  let variance := (1 / 5 : Real) * ((2 - mean)^2 + (3 - mean)^2 + (4 - mean)^2 + (5 - mean)^2 + (6 - mean)^2)
  variance = 2 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_data_set_l2320_232092


namespace NUMINAMATH_GPT_largest_on_edge_l2320_232065

/-- On a grid, each cell contains a number which is the arithmetic mean of the four numbers around it 
    and all numbers are different. Prove that the largest number is located on the edge of the grid. -/
theorem largest_on_edge 
    (grid : ℕ → ℕ → ℝ) 
    (h_condition : ∀ (i j : ℕ), grid i j = (grid (i+1) j + grid (i-1) j + grid i (j+1) + grid i (j-1)) / 4)
    (h_unique : ∀ (i1 j1 i2 j2 : ℕ), (i1 ≠ i2 ∨ j1 ≠ j2) → grid i1 j1 ≠ grid i2 j2)
    : ∃ (i j : ℕ), (i = 0 ∨ j = 0 ∨ i = max_i ∨ j = max_j) ∧ ∀ (x y : ℕ), grid x y ≤ grid i j :=
sorry

end NUMINAMATH_GPT_largest_on_edge_l2320_232065


namespace NUMINAMATH_GPT_p_has_49_l2320_232089

theorem p_has_49 (P : ℝ) (h : P = (2/7) * P + 35) : P = 49 :=
by
  sorry

end NUMINAMATH_GPT_p_has_49_l2320_232089


namespace NUMINAMATH_GPT_parabola_hyperbola_tangent_l2320_232023

theorem parabola_hyperbola_tangent : ∃ m : ℝ, 
  (∀ x y : ℝ, y = x^2 - 2 * x + 2 → y^2 - m * x^2 = 1) ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_tangent_l2320_232023


namespace NUMINAMATH_GPT_bran_tuition_fee_l2320_232004

theorem bran_tuition_fee (P : ℝ) (S : ℝ) (M : ℕ) (R : ℝ) (T : ℝ) 
  (h1 : P = 15) (h2 : S = 0.30) (h3 : M = 3) (h4 : R = 18) 
  (h5 : 0.70 * T - (M * P) = R) : T = 90 :=
by
  sorry

end NUMINAMATH_GPT_bran_tuition_fee_l2320_232004


namespace NUMINAMATH_GPT_solution_set_inequality_l2320_232021

theorem solution_set_inequality (x : ℝ) : 
  (-x^2 + 3 * x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l2320_232021


namespace NUMINAMATH_GPT_kabob_cubes_calculation_l2320_232059

-- Define the properties of a slab of beef
def cubes_per_slab := 80
def cost_per_slab := 25

-- Define Simon's usage and expenditure
def simons_budget := 50
def number_of_kabob_sticks := 40

-- Auxiliary calculations for proofs (making noncomputable if necessary)
noncomputable def cost_per_cube := cost_per_slab / cubes_per_slab
noncomputable def cubes_per_kabob_stick := (2 * cubes_per_slab) / number_of_kabob_sticks

-- The theorem we want to prove
theorem kabob_cubes_calculation :
  cubes_per_kabob_stick = 4 := by
  sorry

end NUMINAMATH_GPT_kabob_cubes_calculation_l2320_232059


namespace NUMINAMATH_GPT_find_f_of_2_l2320_232080

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem find_f_of_2 : f 2 = 5 := by
  sorry

end NUMINAMATH_GPT_find_f_of_2_l2320_232080


namespace NUMINAMATH_GPT_gen_term_seq_l2320_232025

open Nat

def seq (a : ℕ → ℕ) : Prop := 
a 1 = 1 ∧ (∀ n : ℕ, n ≠ 0 → a (n + 1) = 2 * a n - 3)

theorem gen_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end NUMINAMATH_GPT_gen_term_seq_l2320_232025


namespace NUMINAMATH_GPT_find_ordered_pair_l2320_232018

theorem find_ordered_pair (s m : ℚ) :
  (∃ t : ℚ, (5 * s - 7 = 2) ∧ 
           ((∃ (t1 : ℚ), (x = s + 3 * t1) ∧  (y = 2 + m * t1)) 
           → (x = 24 / 5) → (y = 5))) →
  (s = 9 / 5 ∧ m = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l2320_232018


namespace NUMINAMATH_GPT_average_number_of_carnations_l2320_232070

-- Define the number of carnations in each bouquet
def n1 : ℤ := 9
def n2 : ℤ := 23
def n3 : ℤ := 13
def n4 : ℤ := 36
def n5 : ℤ := 28
def n6 : ℤ := 45

-- Define the number of bouquets
def number_of_bouquets : ℤ := 6

-- Prove that the average number of carnations in the bouquets is 25.67
theorem average_number_of_carnations :
  ((n1 + n2 + n3 + n4 + n5 + n6) : ℚ) / (number_of_bouquets : ℚ) = 25.67 := 
by
  sorry

end NUMINAMATH_GPT_average_number_of_carnations_l2320_232070


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2320_232045

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x^2 - x - 6) ≥ 0} = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (3 < x)} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2320_232045


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_prove_inequality_l2320_232064

-- Definition for part (I)
def f (x: ℝ) := |x - 2|
def g (x: ℝ) := 4 - |x - 1|

-- Theorem for part (I)
theorem part_I_solution_set :
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≤ -1/2} ∪ {x : ℝ | x ≥ 7/2} :=
by sorry

-- Definition for part (II)
def satisfiable_range (a: ℝ) := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def density_equation (m n a: ℝ) := (1 / m) + (1 / (2 * n)) = a

-- Theorem for part (II)
theorem part_II_prove_inequality (m n: ℝ) (hm: 0 < m) (hn: 0 < n) 
  (a: ℝ) (h_a: satisfiable_range a = {x : ℝ | abs (x - a) ≤ 1}) (h_density: density_equation m n a) :
  m + 2 * n ≥ 4 :=
by sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_prove_inequality_l2320_232064


namespace NUMINAMATH_GPT_max_blocks_in_box_l2320_232047

def volume (l w h : ℕ) : ℕ := l * w * h

-- Define the dimensions of the box and the block
def box_length := 4
def box_width := 3
def box_height := 2
def block_length := 3
def block_width := 1
def block_height := 1

-- Define the volumes of the box and the block using the dimensions
def V_box : ℕ := volume box_length box_width box_height
def V_block : ℕ := volume block_length block_width block_height

theorem max_blocks_in_box : V_box / V_block = 8 :=
  sorry

end NUMINAMATH_GPT_max_blocks_in_box_l2320_232047


namespace NUMINAMATH_GPT_roots_reciprocal_l2320_232073

theorem roots_reciprocal {a b c x y : ℝ} (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (a * x^2 + b * x + c = 0) ↔ (c * y^2 + b * y + a = 0) := by
sorry

end NUMINAMATH_GPT_roots_reciprocal_l2320_232073


namespace NUMINAMATH_GPT_length_BF_l2320_232007

-- Define the geometrical configuration
structure Point :=
  (x : ℝ) (y : ℝ)

def A := Point.mk 0 0
def B := Point.mk 6 4.8
def C := Point.mk 12 0
def D := Point.mk 3 (-6)
def E := Point.mk 3 0
def F := Point.mk 6 0

-- Define given conditions
def AE := (3 : ℝ)
def CE := (9 : ℝ)
def DE := (6 : ℝ)
def AC := AE + CE

theorem length_BF : (BF = (72 / 7 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_length_BF_l2320_232007


namespace NUMINAMATH_GPT_subway_speed_increase_l2320_232068

theorem subway_speed_increase (s : ℝ) (h₀ : 0 ≤ s) (h₁ : s ≤ 7) : 
  (s^2 + 2 * s = 63) ↔ (s = 7) :=
by
  sorry 

end NUMINAMATH_GPT_subway_speed_increase_l2320_232068


namespace NUMINAMATH_GPT_line_through_points_l2320_232096

/-- The line passing through points A(1, 1) and B(2, 3) satisfies the equation 2x - y - 1 = 0. -/
theorem line_through_points (x y : ℝ) :
  (∃ k : ℝ, k * (y - 1) = 2 * (x - 1)) → 2 * x - y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_l2320_232096


namespace NUMINAMATH_GPT_max_zeros_in_product_of_three_natural_numbers_sum_1003_l2320_232071

theorem max_zeros_in_product_of_three_natural_numbers_sum_1003 :
  ∀ (a b c : ℕ), a + b + c = 1003 →
    ∃ N, (a * b * c) % (10^N) = 0 ∧ N = 7 := by
  sorry

end NUMINAMATH_GPT_max_zeros_in_product_of_three_natural_numbers_sum_1003_l2320_232071


namespace NUMINAMATH_GPT_determine_correct_path_l2320_232094

variable (A B C : Type)
variable (truthful : A → Prop)
variable (whimsical : A → Prop)
variable (answers : A → Prop)
variable (path_correct : A → Prop)

-- Conditions
axiom two_truthful_one_whimsical (x y z : A) : (truthful x ∧ truthful y ∧ whimsical z) ∨ 
                                                (truthful x ∧ truthful z ∧ whimsical y) ∨ 
                                                (truthful y ∧ truthful z ∧ whimsical x)

axiom traveler_aware : ∀ x y : A, truthful x → ¬ truthful y
axiom siblings : A → B → C → Prop
axiom ask_sibling : A → B → C → Prop

-- Conditions formalized
axiom ask_about_truthfulness (x y : A) : answers x → (truthful y ↔ ¬truthful y)

theorem determine_correct_path (x y z : A) :
  (truthful x ∧ ¬truthful y ∧ path_correct x) ∨
  (¬truthful x ∧ truthful y ∧ path_correct y) ∨
  (¬truthful x ∧ ¬truthful y ∧ truthful z ∧ path_correct z) :=
sorry

end NUMINAMATH_GPT_determine_correct_path_l2320_232094


namespace NUMINAMATH_GPT_Beth_crayons_proof_l2320_232029

def Beth_packs_of_crayons (packs_crayons : ℕ) (total_crayons extra_crayons : ℕ) : ℕ :=
  total_crayons - extra_crayons

theorem Beth_crayons_proof
  (packs_crayons : ℕ)
  (each_pack_contains total_crayons extra_crayons : ℕ)
  (h_each_pack : each_pack_contains = 10) 
  (h_extra : extra_crayons = 6)
  (h_total : total_crayons = 40) 
  (valid_packs : packs_crayons = (Beth_packs_of_crayons total_crayons extra_crayons / each_pack_contains)) :
  packs_crayons = 3 :=
by
  rw [h_each_pack, h_extra, h_total] at valid_packs
  sorry

end NUMINAMATH_GPT_Beth_crayons_proof_l2320_232029


namespace NUMINAMATH_GPT_more_math_than_reading_l2320_232098

def pages_reading := 4
def pages_math := 7

theorem more_math_than_reading : pages_math - pages_reading = 3 :=
by
  sorry

end NUMINAMATH_GPT_more_math_than_reading_l2320_232098


namespace NUMINAMATH_GPT_range_half_diff_l2320_232084

theorem range_half_diff (α β : ℝ) (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) : 
    -π/2 ≤ (α - β) / 2 ∧ (α - β) / 2 < 0 := 
    sorry

end NUMINAMATH_GPT_range_half_diff_l2320_232084


namespace NUMINAMATH_GPT_quadratic_inequality_l2320_232038

theorem quadratic_inequality (a : ℝ) 
  (x₁ x₂ : ℝ) (h_roots : ∀ x, x^2 + (3 * a - 1) * x + a + 8 = 0) 
  (h_distinct : x₁ ≠ x₂)
  (h_x1_lt_1 : x₁ < 1) (h_x2_gt_1 : x₂ > 1) : 
  a < -2 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l2320_232038


namespace NUMINAMATH_GPT_cos2_minus_sin2_pi_over_12_l2320_232056

theorem cos2_minus_sin2_pi_over_12 : 
  (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2 = Real.cos (Real.pi / 6) := 
by
  sorry

end NUMINAMATH_GPT_cos2_minus_sin2_pi_over_12_l2320_232056


namespace NUMINAMATH_GPT_alex_piles_of_jelly_beans_l2320_232042

theorem alex_piles_of_jelly_beans : 
  ∀ (initial_weight eaten weight_per_pile remaining_weight piles : ℕ),
    initial_weight = 36 →
    eaten = 6 →
    weight_per_pile = 10 →
    remaining_weight = initial_weight - eaten →
    piles = remaining_weight / weight_per_pile →
    piles = 3 :=
by
  intros initial_weight eaten weight_per_pile remaining_weight piles h_init h_eat h_wpile h_remaining h_piles
  sorry

end NUMINAMATH_GPT_alex_piles_of_jelly_beans_l2320_232042


namespace NUMINAMATH_GPT_expression_even_nat_l2320_232049

theorem expression_even_nat (m n : ℕ) : 
  2 ∣ (5 * m + n + 1) * (3 * m - n + 4) := 
sorry

end NUMINAMATH_GPT_expression_even_nat_l2320_232049


namespace NUMINAMATH_GPT_carbon_copies_after_folding_l2320_232091

-- Define the initial condition of sheets and carbon papers
def initial_sheets : ℕ := 3
def initial_carbons : ℕ := 2

-- Define the condition of folding the paper
def fold_paper (sheets carbons : ℕ) : ℕ := sheets * 2

-- Statement of the problem
theorem carbon_copies_after_folding : (fold_paper initial_sheets initial_carbons - initial_sheets + initial_carbons) = 4 :=
by
  sorry

end NUMINAMATH_GPT_carbon_copies_after_folding_l2320_232091


namespace NUMINAMATH_GPT_custom_op_difference_l2320_232037

def custom_op (x y : ℕ) : ℕ := x * y - (x + y)

theorem custom_op_difference : custom_op 7 4 - custom_op 4 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_difference_l2320_232037


namespace NUMINAMATH_GPT_value_a_squared_plus_b_squared_l2320_232087

-- Defining the problem with the given conditions
theorem value_a_squared_plus_b_squared (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_value_a_squared_plus_b_squared_l2320_232087


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l2320_232055

def a_n (n : ℕ) : ℤ := 2 * n + 1

def S_n (n : ℕ) : ℤ := n * (n + 2)

theorem arithmetic_sequence_properties : 
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) :=
by {
  -- Proof to be filled
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_properties_l2320_232055


namespace NUMINAMATH_GPT_largest_digit_M_divisible_by_6_l2320_232060

theorem largest_digit_M_divisible_by_6 (M : ℕ) (h1 : 5172 * 10 + M % 2 = 0) (h2 : (5 + 1 + 7 + 2 + M) % 3 = 0) : M = 6 := by
  sorry

end NUMINAMATH_GPT_largest_digit_M_divisible_by_6_l2320_232060


namespace NUMINAMATH_GPT_leon_total_payment_l2320_232002

-- Define the constants based on the problem conditions
def cost_toy_organizer : ℝ := 78
def num_toy_organizers : ℝ := 3
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def delivery_fee_rate : ℝ := 0.05

-- Calculate the cost for each category and the total cost
def total_cost_toy_organizers : ℝ := num_toy_organizers * cost_toy_organizer
def total_cost_gaming_chairs : ℝ := num_gaming_chairs * cost_gaming_chair
def total_sales : ℝ := total_cost_toy_organizers + total_cost_gaming_chairs
def delivery_fee : ℝ := delivery_fee_rate * total_sales
def total_amount_paid : ℝ := total_sales + delivery_fee

-- State the theorem for the total amount Leon has to pay
theorem leon_total_payment :
  total_amount_paid = 420 := by
  sorry

end NUMINAMATH_GPT_leon_total_payment_l2320_232002


namespace NUMINAMATH_GPT_solve_for_x_l2320_232076

theorem solve_for_x (x : ℤ) (h : -3 * x - 8 = 8 * x + 3) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2320_232076


namespace NUMINAMATH_GPT_m_greater_than_p_l2320_232008

theorem m_greater_than_p (p m n : ℕ) (pp : Nat.Prime p) (pos_m : m > 0) (pos_n : n > 0) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end NUMINAMATH_GPT_m_greater_than_p_l2320_232008


namespace NUMINAMATH_GPT_determinant_calculation_l2320_232039

variable {R : Type*} [CommRing R]

def matrix_example (a b c : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![1, a, b], ![1, a + b, b + c], ![1, a, a + c]]

theorem determinant_calculation (a b c : R) :
  (matrix_example a b c).det = ab + b^2 + bc :=
by sorry

end NUMINAMATH_GPT_determinant_calculation_l2320_232039


namespace NUMINAMATH_GPT_flower_options_l2320_232011

theorem flower_options (x y : ℕ) : 2 * x + 3 * y = 20 → ∃ x1 y1 x2 y2 x3 y3, 
  (2 * x1 + 3 * y1 = 20) ∧ (2 * x2 + 3 * y2 = 20) ∧ (2 * x3 + 3 * y3 = 20) ∧ 
  (((x1, y1) ≠ (x2, y2)) ∧ ((x2, y2) ≠ (x3, y3)) ∧ ((x1, y1) ≠ (x3, y3))) ∧ 
  ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :=
sorry

end NUMINAMATH_GPT_flower_options_l2320_232011


namespace NUMINAMATH_GPT_turquoise_more_green_l2320_232053

-- Definitions based on given conditions
def total_people : ℕ := 150
def more_blue : ℕ := 90
def both_blue_green : ℕ := 40
def neither_blue_green : ℕ := 20

-- Theorem statement to prove the number of people who believe turquoise is more green
theorem turquoise_more_green : (total_people - neither_blue_green - (more_blue - both_blue_green) - both_blue_green) + both_blue_green = 80 := by
  sorry

end NUMINAMATH_GPT_turquoise_more_green_l2320_232053


namespace NUMINAMATH_GPT_multiply_by_12_correct_result_l2320_232009

theorem multiply_by_12_correct_result (x : ℕ) (h : x / 14 = 42) : x * 12 = 7056 :=
by
  sorry

end NUMINAMATH_GPT_multiply_by_12_correct_result_l2320_232009


namespace NUMINAMATH_GPT_find_A_l2320_232082

theorem find_A (A B : ℕ) (h : 632 - (100 * A + 10 * B) = 41) : A = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_A_l2320_232082


namespace NUMINAMATH_GPT_find_third_number_l2320_232006

theorem find_third_number (x : ℝ) (third_number : ℝ) : 
  0.6 / 0.96 = third_number / 8 → x = 0.96 → third_number = 5 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_third_number_l2320_232006


namespace NUMINAMATH_GPT_women_left_room_is_3_l2320_232062

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end NUMINAMATH_GPT_women_left_room_is_3_l2320_232062


namespace NUMINAMATH_GPT_find_added_amount_l2320_232022

theorem find_added_amount (x y : ℕ) (h1 : x = 18) (h2 : 3 * (2 * x + y) = 123) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_added_amount_l2320_232022


namespace NUMINAMATH_GPT_curve_is_line_l2320_232001

def curve_theta (theta : ℝ) : Prop :=
  theta = Real.pi / 4

theorem curve_is_line : curve_theta θ → (curve_type = "line") :=
by
  intros h
  cases h
  -- This is where the proof would go, but we'll use a placeholder for now.
  -- The essence of the proof will show that all points making an angle of π/4 with the x-axis lie on a line.
  exact sorry

end NUMINAMATH_GPT_curve_is_line_l2320_232001


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2320_232040

theorem simplify_and_evaluate_expression :
  let x := -1
  let y := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 :=
by
  let x := -1
  let y := Real.sqrt 2
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2320_232040


namespace NUMINAMATH_GPT_alpha_value_l2320_232069

theorem alpha_value (α : ℝ) (h : 0 ≤ α ∧ α ≤ 2 * Real.pi 
    ∧ ∃β : ℝ, β = 2 * Real.pi / 3 ∧ (Real.sin β, Real.cos β) = (Real.sin α, Real.cos α)) : 
    α = 5 * Real.pi / 3 := 
  by
    sorry

end NUMINAMATH_GPT_alpha_value_l2320_232069


namespace NUMINAMATH_GPT_trig_relationship_l2320_232099

theorem trig_relationship : 
  let a := Real.sin (145 * Real.pi / 180)
  let b := Real.cos (52 * Real.pi / 180)
  let c := Real.tan (47 * Real.pi / 180)
  a < b ∧ b < c :=
by 
  sorry

end NUMINAMATH_GPT_trig_relationship_l2320_232099


namespace NUMINAMATH_GPT_determine_range_of_b_l2320_232077

noncomputable def f (b x : ℝ) : ℝ := (Real.log x + (x - b) ^ 2) / x
noncomputable def f'' (b x : ℝ) : ℝ := (2 * Real.log x - 2) / x ^ 3

theorem determine_range_of_b (b : ℝ) (h : ∃ x ∈ Set.Icc (1 / 2) 2, f b x > -x * f'' b x) :
  b < 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_range_of_b_l2320_232077


namespace NUMINAMATH_GPT_no_such_f_exists_l2320_232013

theorem no_such_f_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x : ℝ), f (f x) = x^2 - 2 := by
  sorry

end NUMINAMATH_GPT_no_such_f_exists_l2320_232013


namespace NUMINAMATH_GPT_product_M1_M2_l2320_232051

theorem product_M1_M2 :
  (∃ M1 M2 : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    (45 * x - 36) / (x^2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3)) ∧
    M1 * M2 = -222.75) :=
sorry

end NUMINAMATH_GPT_product_M1_M2_l2320_232051


namespace NUMINAMATH_GPT_remainder_property_l2320_232048

theorem remainder_property (a : ℤ) (h : ∃ k : ℤ, a = 45 * k + 36) :
  ∃ n : ℤ, a = 45 * n + 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_property_l2320_232048


namespace NUMINAMATH_GPT_mutter_paid_correct_amount_l2320_232074

def total_lagaan_collected : ℝ := 344000
def mutter_land_percentage : ℝ := 0.0023255813953488372
def mutter_lagaan_paid : ℝ := 800

theorem mutter_paid_correct_amount : 
  mutter_lagaan_paid = total_lagaan_collected * mutter_land_percentage := by
  sorry

end NUMINAMATH_GPT_mutter_paid_correct_amount_l2320_232074


namespace NUMINAMATH_GPT_bc_over_ad_l2320_232028

noncomputable def a : ℝ := 32 / 3
noncomputable def b : ℝ := 16 * Real.pi
noncomputable def c : ℝ := 24 * Real.pi
noncomputable def d : ℝ := 16 * Real.pi

theorem bc_over_ad : (b * c) / (a * d) = 9 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_bc_over_ad_l2320_232028


namespace NUMINAMATH_GPT_compute_inverse_10_mod_1729_l2320_232035

def inverse_of_10_mod_1729 : ℕ :=
  1537

theorem compute_inverse_10_mod_1729 :
  (10 * inverse_of_10_mod_1729) % 1729 = 1 :=
by
  sorry

end NUMINAMATH_GPT_compute_inverse_10_mod_1729_l2320_232035


namespace NUMINAMATH_GPT_total_digits_in_book_l2320_232057

open Nat

theorem total_digits_in_book (n : Nat) (h : n = 10000) : 
    let pages_1_9 := 9
    let pages_10_99 := 90 * 2
    let pages_100_999 := 900 * 3
    let pages_1000_9999 := 9000 * 4
    let page_10000 := 5
    pages_1_9 + pages_10_99 + pages_100_999 + pages_1000_9999 + page_10000 = 38894 :=
by
    sorry

end NUMINAMATH_GPT_total_digits_in_book_l2320_232057


namespace NUMINAMATH_GPT_least_subtracted_12702_is_26_l2320_232079

theorem least_subtracted_12702_is_26 : 12702 % 99 = 26 :=
by
  sorry

end NUMINAMATH_GPT_least_subtracted_12702_is_26_l2320_232079


namespace NUMINAMATH_GPT_rectangle_perimeter_l2320_232081

theorem rectangle_perimeter (w : ℝ) (P : ℝ) (l : ℝ) (A : ℝ) 
  (h1 : l = 18)
  (h2 : A = l * w)
  (h3 : P = 2 * l + 2 * w) 
  (h4 : A + P = 2016) : 
  P = 234 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2320_232081


namespace NUMINAMATH_GPT_pretty_number_characterization_l2320_232027

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, k < n → ℓ < n → k > 0 → ℓ > 0 → 
    (n ∣ 2*k - ℓ ∨ n ∣ 2*ℓ - k)

theorem pretty_number_characterization :
  ∀ n : ℕ, is_pretty n ↔ (Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15) :=
by
  sorry

end NUMINAMATH_GPT_pretty_number_characterization_l2320_232027


namespace NUMINAMATH_GPT_smallest_number_l2320_232020

theorem smallest_number (a b c : ℕ) (h1 : b = 29) (h2 : c = b + 7) (h3 : (a + b + c) / 3 = 30) : a = 25 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l2320_232020


namespace NUMINAMATH_GPT_sqrt_sqrt_81_is_9_l2320_232050

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end NUMINAMATH_GPT_sqrt_sqrt_81_is_9_l2320_232050


namespace NUMINAMATH_GPT_ratio_dislikes_to_likes_l2320_232003

theorem ratio_dislikes_to_likes 
  (D : ℕ) 
  (h1 : D + 1000 = 2600) 
  (h2 : 3000 > 0) : 
  D / 3000 = 8 / 15 :=
by sorry

end NUMINAMATH_GPT_ratio_dislikes_to_likes_l2320_232003


namespace NUMINAMATH_GPT_books_ratio_l2320_232000

-- Definitions based on the conditions
def Alyssa_books : Nat := 36
def Nancy_books : Nat := 252

-- Statement to prove
theorem books_ratio :
  (Nancy_books / Alyssa_books) = 7 := 
sorry

end NUMINAMATH_GPT_books_ratio_l2320_232000


namespace NUMINAMATH_GPT_max_value_expression_le_380_l2320_232063

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_le_380 (a b c d : ℝ)
  (ha : -9.5 ≤ a ∧ a ≤ 9.5)
  (hb : -9.5 ≤ b ∧ b ≤ 9.5)
  (hc : -9.5 ≤ c ∧ c ≤ 9.5)
  (hd : -9.5 ≤ d ∧ d ≤ 9.5) :
  max_value_expression a b c d ≤ 380 :=
sorry

end NUMINAMATH_GPT_max_value_expression_le_380_l2320_232063


namespace NUMINAMATH_GPT_power_comparison_l2320_232054

theorem power_comparison :
  2 ^ 16 = 256 * 16 ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_power_comparison_l2320_232054


namespace NUMINAMATH_GPT_area_ratio_l2320_232086

theorem area_ratio (l b r : ℝ) (h1 : l = 2 * b) (h2 : 6 * b = 2 * π * r) :
  (l * b) / (π * r ^ 2) = 2 * π / 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_area_ratio_l2320_232086


namespace NUMINAMATH_GPT_tangent_line_at_zero_l2320_232090

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem tangent_line_at_zero : ∀ x : ℝ, x = 0 → Real.exp x * Real.sin x = 0 ∧ (Real.exp x * (Real.sin x + Real.cos x)) = 1 → (∀ y, y = x) :=
  by
    sorry

end NUMINAMATH_GPT_tangent_line_at_zero_l2320_232090


namespace NUMINAMATH_GPT_monthly_rent_calculation_l2320_232010

-- Definitions based on the problem conditions
def investment_amount : ℝ := 20000
def desired_annual_return_rate : ℝ := 0.06
def annual_property_taxes : ℝ := 650
def maintenance_percentage : ℝ := 0.15

-- Theorem stating the mathematically equivalent problem
theorem monthly_rent_calculation : 
  let required_annual_return := desired_annual_return_rate * investment_amount
  let total_annual_earnings := required_annual_return + annual_property_taxes
  let monthly_earnings_target := total_annual_earnings / 12
  let monthly_rent := monthly_earnings_target / (1 - maintenance_percentage)
  monthly_rent = 181.38 :=
by
  sorry

end NUMINAMATH_GPT_monthly_rent_calculation_l2320_232010


namespace NUMINAMATH_GPT_min_value_of_a_l2320_232012

theorem min_value_of_a :
  ∀ (x y : ℝ), |x| + |y| ≤ 1 → (|2 * x - 3 * y + 3 / 2| + |y - 1| + |2 * y - x - 3| ≤ 23 / 2) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_min_value_of_a_l2320_232012


namespace NUMINAMATH_GPT_fill_time_is_13_seconds_l2320_232036

-- Define the given conditions as constants
def flow_rate_in (t : ℝ) : ℝ := 24 * t -- 24 gallons/second
def leak_rate (t : ℝ) : ℝ := 4 * t -- 4 gallons/second
def basin_capacity : ℝ := 260 -- 260 gallons

-- Main theorem to be proven
theorem fill_time_is_13_seconds : 
  ∀ t : ℝ, (flow_rate_in t - leak_rate t) * (13) = basin_capacity := 
sorry

end NUMINAMATH_GPT_fill_time_is_13_seconds_l2320_232036


namespace NUMINAMATH_GPT_cube_division_l2320_232017

theorem cube_division (n : ℕ) (hn1 : 6 ≤ n) (hn2 : n % 2 = 0) : 
  ∃ m : ℕ, (n = 2 * m) ∧ (∀ a : ℕ, ∀ b : ℕ, ∀ c: ℕ, a = m^3 - (m - 1)^3 + 1 → b = 3 * m * (m - 1) + 2 → a = b) :=
by
  sorry

end NUMINAMATH_GPT_cube_division_l2320_232017


namespace NUMINAMATH_GPT_rooms_with_two_beds_l2320_232041

variable (x y : ℕ)

theorem rooms_with_two_beds:
  x + y = 13 →
  2 * x + 3 * y = 31 →
  x = 8 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_rooms_with_two_beds_l2320_232041


namespace NUMINAMATH_GPT_find_f_of_one_third_l2320_232043

-- Define g function according to given condition
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f function according to given condition, valid for x ≠ 0
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x

-- State the theorem we need to prove
theorem find_f_of_one_third : f (1 / 3) = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_f_of_one_third_l2320_232043


namespace NUMINAMATH_GPT_first_batch_price_is_50_max_number_of_type_a_tools_l2320_232088

-- Define the conditions
def first_batch_cost : Nat := 2000
def second_batch_cost : Nat := 2200
def price_increase : Nat := 5
def max_total_cost : Nat := 2500
def type_b_cost : Nat := 40
def total_third_batch : Nat := 50

-- First batch price per tool
theorem first_batch_price_is_50 (x : Nat) (h1 : first_batch_cost * (x + price_increase) = second_batch_cost * x) :
  x = 50 :=
sorry

-- Second batch price per tool & maximum type A tools in third batch
theorem max_number_of_type_a_tools (y : Nat)
  (h2 : 55 * y + type_b_cost * (total_third_batch - y) ≤ max_total_cost) :
  y ≤ 33 :=
sorry

end NUMINAMATH_GPT_first_batch_price_is_50_max_number_of_type_a_tools_l2320_232088


namespace NUMINAMATH_GPT_determine_p_q_l2320_232072

theorem determine_p_q (r1 r2 p q : ℝ) (h1 : r1 + r2 = 5) (h2 : r1 * r2 = 6) (h3 : r1^2 + r2^2 = -p) (h4 : r1^2 * r2^2 = q) : p = -13 ∧ q = 36 :=
by
  sorry

end NUMINAMATH_GPT_determine_p_q_l2320_232072


namespace NUMINAMATH_GPT_missing_digits_pairs_l2320_232034

theorem missing_digits_pairs (x y : ℕ) : (2 + 4 + 6 + x + y + 8) % 9 = 0 ↔ x + y = 7 := by
  sorry

end NUMINAMATH_GPT_missing_digits_pairs_l2320_232034


namespace NUMINAMATH_GPT_n_leq_1972_l2320_232066

theorem n_leq_1972 (n : ℕ) (h1 : 4 ^ 27 + 4 ^ 1000 + 4 ^ n = k ^ 2) : n ≤ 1972 :=
by
  sorry

end NUMINAMATH_GPT_n_leq_1972_l2320_232066


namespace NUMINAMATH_GPT_people_at_table_l2320_232093

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end NUMINAMATH_GPT_people_at_table_l2320_232093


namespace NUMINAMATH_GPT_rectangular_prism_volume_l2320_232024

theorem rectangular_prism_volume
  (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66) :
  a * b * c = 150 :=
by sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l2320_232024


namespace NUMINAMATH_GPT_ratio_of_students_to_professors_l2320_232052

theorem ratio_of_students_to_professors (total : ℕ) (students : ℕ) (professors : ℕ)
  (h1 : total = 40000) (h2 : students = 37500) (h3 : total = students + professors) :
  students / professors = 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_students_to_professors_l2320_232052


namespace NUMINAMATH_GPT_find_center_of_circle_l2320_232078

-- Condition 1: The circle is tangent to the lines 3x - 4y = 12 and 3x - 4y = -48
def tangent_line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 12
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -48

-- Condition 2: The center of the circle lies on the line x - 2y = 0
def center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- The center of the circle
def circle_center (x y : ℝ) : Prop := 
  tangent_line1 x y ∧ tangent_line2 x y ∧ center_line x y

-- Statement to prove
theorem find_center_of_circle : 
  circle_center (-18) (-9) := 
sorry

end NUMINAMATH_GPT_find_center_of_circle_l2320_232078


namespace NUMINAMATH_GPT_exists_prime_with_composite_sequence_l2320_232033

theorem exists_prime_with_composite_sequence (n : ℕ) (hn : n ≠ 0) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬ Nat.Prime (p + k) :=
sorry

end NUMINAMATH_GPT_exists_prime_with_composite_sequence_l2320_232033


namespace NUMINAMATH_GPT_rain_first_hour_l2320_232058

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_rain_first_hour_l2320_232058


namespace NUMINAMATH_GPT_john_outside_doors_count_l2320_232085

theorem john_outside_doors_count 
  (bedroom_doors : ℕ := 3) 
  (cost_outside_door : ℕ := 20) 
  (total_cost : ℕ := 70) 
  (cost_bedroom_door := cost_outside_door / 2) 
  (total_bedroom_cost := bedroom_doors * cost_bedroom_door) 
  (outside_doors := (total_cost - total_bedroom_cost) / cost_outside_door) : 
  outside_doors = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_outside_doors_count_l2320_232085


namespace NUMINAMATH_GPT_infinite_series_evaluates_to_12_l2320_232016

noncomputable def infinite_series : ℝ :=
  ∑' k, (k^3) / (3^k)

theorem infinite_series_evaluates_to_12 :
  infinite_series = 12 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_evaluates_to_12_l2320_232016


namespace NUMINAMATH_GPT_single_elimination_game_count_l2320_232014

theorem single_elimination_game_count (n : Nat) (h : n = 23) : n - 1 = 22 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_game_count_l2320_232014


namespace NUMINAMATH_GPT_short_answer_question_time_l2320_232067

-- Definitions from the conditions
def minutes_per_paragraph := 15
def minutes_per_essay := 60
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15
def total_minutes := 4 * 60

-- Auxiliary calculations
def total_minutes_essays := num_essays * minutes_per_essay
def total_minutes_paragraphs := num_paragraphs * minutes_per_paragraph
def total_minutes_used := total_minutes_essays + total_minutes_paragraphs

-- The time per short-answer question is 3 minutes
theorem short_answer_question_time (x : ℕ) : (total_minutes - total_minutes_used) / num_short_answer_questions = 3 :=
by
  -- x is defined as the time per short-answer question
  let x := (total_minutes - total_minutes_used) / num_short_answer_questions
  have time_for_short_answer_questions : total_minutes - total_minutes_used = 45 := by sorry
  have time_per_short_answer_question : 45 / num_short_answer_questions = 3 := by sorry
  have x_equals_3 : x = 3 := by sorry
  exact x_equals_3

end NUMINAMATH_GPT_short_answer_question_time_l2320_232067


namespace NUMINAMATH_GPT_intersect_not_A_B_l2320_232083

open Set

-- Define the universal set U
def U := ℝ

-- Define set A
def A := {x : ℝ | x ≤ 3}

-- Define set B
def B := {x : ℝ | x ≤ 6}

-- Define the complement of A in U
def not_A := {x : ℝ | x > 3}

-- The proof problem
theorem intersect_not_A_B :
  (not_A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 6} :=
sorry

end NUMINAMATH_GPT_intersect_not_A_B_l2320_232083


namespace NUMINAMATH_GPT_Kaylee_total_boxes_needed_l2320_232095

-- Defining the conditions
def lemon_biscuits := 12
def chocolate_biscuits := 5
def oatmeal_biscuits := 4
def still_needed := 12

-- Defining the total boxes sold so far
def total_sold := lemon_biscuits + chocolate_biscuits + oatmeal_biscuits

-- Defining the total number of boxes that need to be sold in total
def total_needed := total_sold + still_needed

-- Lean statement to prove the required total number of boxes
theorem Kaylee_total_boxes_needed : total_needed = 33 :=
by
  sorry

end NUMINAMATH_GPT_Kaylee_total_boxes_needed_l2320_232095


namespace NUMINAMATH_GPT_min_a_b_div_1176_l2320_232032

theorem min_a_b_div_1176 (a b : ℕ) (h : b^3 = 1176 * a) : a = 63 :=
by sorry

end NUMINAMATH_GPT_min_a_b_div_1176_l2320_232032


namespace NUMINAMATH_GPT_remainder_when_sum_of_six_primes_divided_by_seventh_prime_l2320_232005

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_of_six_primes_divided_by_seventh_prime_l2320_232005
