import Mathlib

namespace NUMINAMATH_GPT_Xiaofang_English_score_l903_90388

/-- Given the conditions about the average scores of Xiaofang's subjects:
  1. The average score for 4 subjects is 88.
  2. The average score for the first 2 subjects is 93.
  3. The average score for the last 3 subjects is 87.
Prove that Xiaofang's English test score is 95. -/
theorem Xiaofang_English_score
    (L M E S : ℝ)
    (h1 : (L + M + E + S) / 4 = 88)
    (h2 : (L + M) / 2 = 93)
    (h3 : (M + E + S) / 3 = 87) :
    E = 95 :=
by
  sorry

end NUMINAMATH_GPT_Xiaofang_English_score_l903_90388


namespace NUMINAMATH_GPT_sum_of_first_n_odd_integers_eq_169_l903_90396

theorem sum_of_first_n_odd_integers_eq_169 (n : ℕ) 
  (h : n^2 = 169) : n = 13 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_n_odd_integers_eq_169_l903_90396


namespace NUMINAMATH_GPT_adam_initial_money_l903_90310

theorem adam_initial_money :
  let cost_of_airplane := 4.28
  let change_received := 0.72
  cost_of_airplane + change_received = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_adam_initial_money_l903_90310


namespace NUMINAMATH_GPT_carl_additional_marbles_l903_90317

def initial_marbles := 12
def lost_marbles := initial_marbles / 2
def additional_marbles_from_mom := 25
def marbles_in_jar_after_game := 41

theorem carl_additional_marbles :
  (marbles_in_jar_after_game - additional_marbles_from_mom) + lost_marbles - initial_marbles = 10 :=
by
  sorry

end NUMINAMATH_GPT_carl_additional_marbles_l903_90317


namespace NUMINAMATH_GPT_positive_diff_of_supplementary_angles_l903_90384

theorem positive_diff_of_supplementary_angles (x : ℝ) (h : 5 * x + 3 * x = 180) : 
  abs ((5 * x - 3 * x)) = 45 := by
  sorry

end NUMINAMATH_GPT_positive_diff_of_supplementary_angles_l903_90384


namespace NUMINAMATH_GPT_matrix_solution_l903_90356

variable {x : ℝ}

theorem matrix_solution (x: ℝ) :
  let M := (3*x) * (2*x + 1) - (1) * (2*x)
  M = 5 → (x = 5/6) ∨ (x = -1) :=
by
  sorry

end NUMINAMATH_GPT_matrix_solution_l903_90356


namespace NUMINAMATH_GPT_geometric_sequence_term_l903_90343

theorem geometric_sequence_term
  (r a : ℝ)
  (h1 : 180 * r = a)
  (h2 : a * r = 81 / 32)
  (h3 : a > 0) :
  a = 135 / 19 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_term_l903_90343


namespace NUMINAMATH_GPT_books_before_purchase_l903_90334

theorem books_before_purchase (x : ℕ) (h : x + 140 = (27 / 25 : ℚ) * x) : x = 1750 :=
sorry

end NUMINAMATH_GPT_books_before_purchase_l903_90334


namespace NUMINAMATH_GPT_candies_remaining_l903_90377

theorem candies_remaining 
    (red_candies : ℕ)
    (yellow_candies : ℕ)
    (blue_candies : ℕ)
    (yellow_condition : yellow_candies = 3 * red_candies - 20)
    (blue_condition : blue_candies = yellow_candies / 2)
    (initial_red_candies : red_candies = 40) :
    (red_candies + yellow_candies + blue_candies - yellow_candies) = 90 := 
by
  sorry

end NUMINAMATH_GPT_candies_remaining_l903_90377


namespace NUMINAMATH_GPT_number_of_levels_l903_90318

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_levels_l903_90318


namespace NUMINAMATH_GPT_inequality_solution_set_l903_90363

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (3 / 4 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l903_90363


namespace NUMINAMATH_GPT_cube_difference_divisibility_l903_90346

-- Given conditions
variables {m n : ℤ} (h1 : m % 2 = 1) (h2 : n % 2 = 1) (k : ℕ)

-- The equivalent statement to be proven
theorem cube_difference_divisibility (h1 : m % 2 = 1) (h2 : n % 2 = 1) : 
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) :=
sorry

end NUMINAMATH_GPT_cube_difference_divisibility_l903_90346


namespace NUMINAMATH_GPT_gasoline_reduction_l903_90395

theorem gasoline_reduction (P Q : ℝ) :
  let new_price := 1.25 * P
  let new_budget := 1.10 * (P * Q)
  let new_quantity := new_budget / new_price
  let percent_reduction := 1 - (new_quantity / Q)
  percent_reduction = 0.12 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_reduction_l903_90395


namespace NUMINAMATH_GPT_problem_inequality_minimum_value_l903_90370

noncomputable def f (x y z : ℝ) : ℝ := 
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem problem_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z ≥ 0 :=
sorry

theorem minimum_value (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end NUMINAMATH_GPT_problem_inequality_minimum_value_l903_90370


namespace NUMINAMATH_GPT_vegetables_sold_mass_l903_90313

/-- Define the masses of the vegetables --/
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8

/-- Define the total mass of installed vegetables --/
def total_mass : ℕ := mass_carrots + mass_zucchini + mass_broccoli

/-- Define the mass of vegetables sold (half of the total mass) --/
def mass_sold : ℕ := total_mass / 2

/-- Prove that the mass of vegetables sold is 18 kg --/
theorem vegetables_sold_mass : mass_sold = 18 := by
  sorry

end NUMINAMATH_GPT_vegetables_sold_mass_l903_90313


namespace NUMINAMATH_GPT_decomposition_of_x_l903_90352

-- Definitions derived from the conditions
def x : ℝ × ℝ × ℝ := (11, 5, -3)
def p : ℝ × ℝ × ℝ := (1, 0, 2)
def q : ℝ × ℝ × ℝ := (-1, 0, 1)
def r : ℝ × ℝ × ℝ := (2, 5, -3)

-- Theorem statement proving the decomposition
theorem decomposition_of_x : x = (3 : ℝ) • p + (-6 : ℝ) • q + (1 : ℝ) • r := by
  sorry

end NUMINAMATH_GPT_decomposition_of_x_l903_90352


namespace NUMINAMATH_GPT_marlon_keeps_4_lollipops_l903_90329

def initial_lollipops : ℕ := 42
def fraction_given_to_emily : ℚ := 2 / 3
def lollipops_given_to_lou : ℕ := 10

theorem marlon_keeps_4_lollipops :
  let lollipops_given_to_emily := fraction_given_to_emily * initial_lollipops
  let lollipops_after_emily := initial_lollipops - lollipops_given_to_emily
  let marlon_keeps := lollipops_after_emily - lollipops_given_to_lou
  marlon_keeps = 4 :=
by
  sorry

end NUMINAMATH_GPT_marlon_keeps_4_lollipops_l903_90329


namespace NUMINAMATH_GPT_compare_probabilities_l903_90368

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end NUMINAMATH_GPT_compare_probabilities_l903_90368


namespace NUMINAMATH_GPT_intersection_is_correct_l903_90315

-- Conditions definitions
def setA : Set ℝ := {x | 2 < x ∧ x < 8}
def setB : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Intersection definition
def intersection : Set ℝ := {x | 2 < x ∧ x ≤ 6}

-- Theorem statement
theorem intersection_is_correct : setA ∩ setB = intersection := 
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l903_90315


namespace NUMINAMATH_GPT_no_integer_solutions_l903_90328

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l903_90328


namespace NUMINAMATH_GPT_new_sphere_radius_l903_90361

noncomputable def calculateVolume (R r : ℝ) : ℝ :=
  let originalSphereVolume := (4 / 3) * Real.pi * R^3
  let cylinderHeight := 2 * Real.sqrt (R^2 - r^2)
  let cylinderVolume := Real.pi * r^2 * cylinderHeight
  let capHeight := R - Real.sqrt (R^2 - r^2)
  let capVolume := (Real.pi * capHeight^2 * (3 * R - capHeight)) / 3
  let totalCapVolume := 2 * capVolume
  originalSphereVolume - cylinderVolume - totalCapVolume

theorem new_sphere_radius
  (R : ℝ) (r : ℝ) (h : ℝ) (new_sphere_radius : ℝ)
  (h_eq: h = 2 * Real.sqrt (R^2 - r^2))
  (new_sphere_volume_eq: calculateVolume R r = (4 / 3) * Real.pi * new_sphere_radius^3)
  : new_sphere_radius = 16 :=
sorry

end NUMINAMATH_GPT_new_sphere_radius_l903_90361


namespace NUMINAMATH_GPT_train_length_l903_90330

theorem train_length :
  (∃ L : ℕ, (L / 15) = (L + 800) / 45) → L = 400 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l903_90330


namespace NUMINAMATH_GPT_standard_robot_weight_l903_90393

variable (S : ℕ) -- Define the variable for the standard robot's weight
variable (MaxWeight : ℕ := 210) -- Define the variable for the maximum weight of a robot, which is 210 pounds
variable (MinWeight : ℕ) -- Define the variable for the minimum weight of the robot

theorem standard_robot_weight (h1 : 2 * MinWeight ≥ MaxWeight) 
                             (h2 : MinWeight = S + 5) 
                             (h3 : MaxWeight = 210) :
  100 ≤ S ∧ S ≤ 105 := 
by
  sorry

end NUMINAMATH_GPT_standard_robot_weight_l903_90393


namespace NUMINAMATH_GPT_xiao_ming_total_score_l903_90323

theorem xiao_ming_total_score :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5 ∧ 
  a_1 + a_2 = 10 ∧ 
  a_4 + a_5 = 18 ∧ 
  a_1 + a_2 + a_3 + a_4 + a_5 = 35 :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_total_score_l903_90323


namespace NUMINAMATH_GPT_fill_box_with_cubes_l903_90300

-- Define the dimensions of the box
def boxLength : ℕ := 35
def boxWidth : ℕ := 20
def boxDepth : ℕ := 10

-- Define the greatest common divisor of the box dimensions
def gcdBoxDims : ℕ := Nat.gcd (Nat.gcd boxLength boxWidth) boxDepth

-- Define the smallest number of identical cubes that can fill the box
def smallestNumberOfCubes : ℕ := (boxLength / gcdBoxDims) * (boxWidth / gcdBoxDims) * (boxDepth / gcdBoxDims)

theorem fill_box_with_cubes :
  smallestNumberOfCubes = 56 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fill_box_with_cubes_l903_90300


namespace NUMINAMATH_GPT_true_proposition_l903_90373

-- Definitions of propositions
def p := ∃ (x : ℝ), x - x + 1 ≥ 0
def q := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- Theorem statement
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l903_90373


namespace NUMINAMATH_GPT_fraction_difference_l903_90345

theorem fraction_difference : (18 / 42) - (3 / 8) = 3 / 56 := 
by
  sorry

end NUMINAMATH_GPT_fraction_difference_l903_90345


namespace NUMINAMATH_GPT_fraction_identity_l903_90347

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l903_90347


namespace NUMINAMATH_GPT_problem_solution_l903_90333

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_solution (a m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) →
  a = 1 ∧ (∃ n : ℝ, f n 1 ≤ m - f (-n) 1) → 4 ≤ m := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l903_90333


namespace NUMINAMATH_GPT_parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l903_90394

-- Define the first parabola proof problem
theorem parabola_vertex_at_origin_axis_x_passing_point :
  (∃ (m : ℝ), ∀ (x y : ℝ), y^2 = m * x ↔ (y, x) = (0, 0) ∨ (x = 6 ∧ y = -3)) → 
  ∃ m : ℝ, m = 1.5 ∧ (y^2 = m * x) :=
sorry

-- Define the second parabola proof problem
theorem parabola_vertex_at_origin_axis_y_distance_focus :
  (∃ (p : ℝ), ∀ (x y : ℝ), x^2 = 4 * p * y ↔ (y, x) = (0, 0) ∨ (p = 3)) → 
  ∃ q : ℝ, q = 12 ∧ (x^2 = q * y ∨ x^2 = -q * y) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l903_90394


namespace NUMINAMATH_GPT_peter_vacation_saving_l903_90359

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end NUMINAMATH_GPT_peter_vacation_saving_l903_90359


namespace NUMINAMATH_GPT_cube_value_proportional_l903_90324

theorem cube_value_proportional (side_length1 side_length2 : ℝ) (volume1 volume2 : ℝ) (value1 value2 : ℝ) :
  side_length1 = 4 → volume1 = side_length1 ^ 3 → value1 = 500 →
  side_length2 = 6 → volume2 = side_length2 ^ 3 → value2 = value1 * (volume2 / volume1) →
  value2 = 1688 :=
by
  sorry

end NUMINAMATH_GPT_cube_value_proportional_l903_90324


namespace NUMINAMATH_GPT_final_match_l903_90380

-- Definitions of players and conditions
inductive Player
| Antony | Bart | Carl | Damian | Ed | Fred | Glen | Harry

open Player

-- Condition definitions
def beat (p1 p2 : Player) : Prop := sorry

-- Given conditions
axiom Bart_beats_Antony : beat Bart Antony
axiom Carl_beats_Damian : beat Carl Damian
axiom Glen_beats_Harry : beat Glen Harry
axiom Glen_beats_Carl : beat Glen Carl
axiom Carl_beats_Bart : beat Carl Bart
axiom Ed_beats_Fred : beat Ed Fred
axiom Glen_beats_Ed : beat Glen Ed

-- The proof statement
theorem final_match : beat Glen Carl :=
by
  sorry

end NUMINAMATH_GPT_final_match_l903_90380


namespace NUMINAMATH_GPT_total_cost_correct_l903_90302

-- Definitions for the costs of items.
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87

-- Definitions for the quantities.
def num_sandwiches : ℝ := 2
def num_sodas : ℝ := 4

-- The calculation for the total cost.
def total_cost : ℝ := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The claim that needs to be proved.
theorem total_cost_correct : total_cost = 10.46 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l903_90302


namespace NUMINAMATH_GPT_televisions_selection_ways_l903_90362

noncomputable def combination (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem televisions_selection_ways :
  let TypeA := 4
  let TypeB := 5
  let choosen := 3
  (∃ (n m : ℕ), n + m = choosen ∧ 1 ≤ n ∧ n ≤ TypeA ∧ 1 ≤ m ∧ m ≤ TypeB ∧
    combination TypeA n * combination TypeB m = 70) :=
by
  sorry

end NUMINAMATH_GPT_televisions_selection_ways_l903_90362


namespace NUMINAMATH_GPT_least_possible_perimeter_l903_90369

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_perimeter_l903_90369


namespace NUMINAMATH_GPT_tangent_line_parallel_range_a_l903_90307

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log x + 1/2 * x^2 + a * x

theorem tangent_line_parallel_range_a (a : ℝ) :
  (∃ x > 0, deriv (f a) x = 3) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_range_a_l903_90307


namespace NUMINAMATH_GPT_set_condition_implies_union_l903_90386

open Set

variable {α : Type*} {M P : Set α}

theorem set_condition_implies_union 
  (h : M ∩ P = P) : M ∪ P = M := 
sorry

end NUMINAMATH_GPT_set_condition_implies_union_l903_90386


namespace NUMINAMATH_GPT_infinite_quadruples_inequality_quadruple_l903_90382

theorem infinite_quadruples 
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  ∃ (a p q r : ℕ), 
    1 < p ∧ 1 < q ∧ 1 < r ∧
    p ∣ (a * q * r + 1) ∧
    q ∣ (a * p * r + 1) ∧
    r ∣ (a * p * q + 1) :=
sorry

theorem inequality_quadruple
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end NUMINAMATH_GPT_infinite_quadruples_inequality_quadruple_l903_90382


namespace NUMINAMATH_GPT_find_quadruples_l903_90399

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

theorem find_quadruples (a b p n : ℕ) (hp : is_prime p) (h_ab : a + b ≠ 0) :
  a^3 + b^3 = p^n ↔ (a = 1 ∧ b = 1 ∧ p = 2 ∧ n = 1) ∨
               (a = 1 ∧ b = 2 ∧ p = 3 ∧ n = 2) ∨ 
               (a = 2 ∧ b = 1 ∧ p = 3 ∧ n = 2) ∨
               ∃ (k : ℕ), (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨ 
                          (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
                          (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end NUMINAMATH_GPT_find_quadruples_l903_90399


namespace NUMINAMATH_GPT_truck_travel_due_east_distance_l903_90383

theorem truck_travel_due_east_distance :
  ∀ (x : ℕ),
  (20 + 20)^2 + x^2 = 50^2 → x = 30 :=
by
  intro x
  sorry -- proof will be here

end NUMINAMATH_GPT_truck_travel_due_east_distance_l903_90383


namespace NUMINAMATH_GPT_group_contains_2007_l903_90306

theorem group_contains_2007 : 
  ∃ k, 2007 ∈ {a | (k * (k + 1)) / 2 < a ∧ a ≤ ((k + 1) * (k + 2)) / 2} ∧ k = 45 :=
by sorry

end NUMINAMATH_GPT_group_contains_2007_l903_90306


namespace NUMINAMATH_GPT_find_p_plus_s_l903_90354

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem find_p_plus_s (p q r s : ℝ) (h : p * q * r * s ≠ 0) 
  (hg : ∀ x : ℝ, g p q r s (g p q r s x) = x) : p + s = 0 := 
by 
  sorry

end NUMINAMATH_GPT_find_p_plus_s_l903_90354


namespace NUMINAMATH_GPT_total_tickets_sold_correct_l903_90355

theorem total_tickets_sold_correct :
  ∀ (A : ℕ), (21 * A + 15 * 327 = 8748) → (A + 327 = 509) :=
by
  intros A h
  sorry

end NUMINAMATH_GPT_total_tickets_sold_correct_l903_90355


namespace NUMINAMATH_GPT_rectangle_side_ratio_l903_90321

theorem rectangle_side_ratio
  (s : ℝ)  -- the side length of the inner square
  (y x : ℝ) -- the side lengths of the rectangles (y: shorter, x: longer)
  (h1 : 9 * s^2 = (3 * s)^2)  -- the area of the outer square is 9 times that of the inner square
  (h2 : s + 2*y = 3*s)  -- the total side length relation due to geometry
  (h3 : x + y = 3*s)  -- another side length relation
: x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_ratio_l903_90321


namespace NUMINAMATH_GPT_cauchy_inequality_minimum_value_inequality_l903_90316

-- Part 1: Prove Cauchy Inequality
theorem cauchy_inequality (a b x y : ℝ) : 
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

-- Part 2: Find the minimum value under the given conditions
theorem minimum_value_inequality (x y : ℝ) (h₁ : x^2 + y^2 = 2) (h₂ : x ≠ y ∨ x ≠ -y) : 
  ∃ m, m = (1 / (9 * x^2) + 9 / y^2) ∧ m = 50 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cauchy_inequality_minimum_value_inequality_l903_90316


namespace NUMINAMATH_GPT_pirate_treasure_probability_l903_90372

theorem pirate_treasure_probability :
  let p_treasure_no_traps := 1 / 3
  let p_traps_no_treasure := 1 / 6
  let p_neither := 1 / 2
  let choose_4_out_of_8 := 70
  let p_4_treasure_no_traps := (1 / 3) ^ 4
  let p_4_neither := (1 / 2) ^ 4
  choose_4_out_of_8 * p_4_treasure_no_traps * p_4_neither = 35 / 648 :=
by
  sorry

end NUMINAMATH_GPT_pirate_treasure_probability_l903_90372


namespace NUMINAMATH_GPT_Mitch_saved_amount_l903_90391

theorem Mitch_saved_amount :
  let boat_cost_per_foot := 1500
  let license_and_registration := 500
  let docking_fees := 3 * 500
  let longest_boat_length := 12
  let total_license_and_fees := license_and_registration + docking_fees
  let total_boat_cost := boat_cost_per_foot * longest_boat_length
  let total_saved := total_boat_cost + total_license_and_fees
  total_saved = 20000 :=
by
  sorry

end NUMINAMATH_GPT_Mitch_saved_amount_l903_90391


namespace NUMINAMATH_GPT_intersection_eq_l903_90339

def M : Set Real := {x | x^2 < 3 * x}
def N : Set Real := {x | Real.log x < 0}

theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l903_90339


namespace NUMINAMATH_GPT_Jessie_weight_l903_90381

theorem Jessie_weight (c l w : ℝ) (hc : c = 27) (hl : l = 101) : c + l = w ↔ w = 128 := by
  sorry

end NUMINAMATH_GPT_Jessie_weight_l903_90381


namespace NUMINAMATH_GPT_gallons_needed_to_grandmas_house_l903_90385

def car_fuel_efficiency : ℝ := 20
def distance_to_grandmas_house : ℝ := 100

theorem gallons_needed_to_grandmas_house : (distance_to_grandmas_house / car_fuel_efficiency) = 5 :=
by
  sorry

end NUMINAMATH_GPT_gallons_needed_to_grandmas_house_l903_90385


namespace NUMINAMATH_GPT_tan_double_angle_l903_90360

theorem tan_double_angle (α : Real) (h1 : α > π ∧ α < 3 * π / 2) (h2 : Real.sin (π - α) = -3/5) :
  Real.tan (2 * α) = 24/7 := 
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l903_90360


namespace NUMINAMATH_GPT_no_common_points_iff_parallel_l903_90326

-- Definitions based on conditions:
def line (a : Type) : Prop := sorry
def plane (M : Type) : Prop := sorry
def no_common_points (a : Type) (M : Type) : Prop := sorry
def parallel (a : Type) (M : Type) : Prop := sorry

-- Theorem stating the relationship is necessary and sufficient
theorem no_common_points_iff_parallel (a M : Type) :
  no_common_points a M ↔ parallel a M := sorry

end NUMINAMATH_GPT_no_common_points_iff_parallel_l903_90326


namespace NUMINAMATH_GPT_carrie_jellybeans_l903_90344

def volume (a : ℕ) : ℕ := a * a * a

def bert_box_volume : ℕ := 216

def carrie_factor : ℕ := 3

def count_error_factor : ℝ := 1.10

noncomputable def jellybeans_carrie (bert_box_volume carrie_factor count_error_factor : ℝ) : ℝ :=
  count_error_factor * (carrie_factor ^ 3 * bert_box_volume)

theorem carrie_jellybeans (bert_box_volume := 216) (carrie_factor := 3) (count_error_factor := 1.10) :
  jellybeans_carrie bert_box_volume carrie_factor count_error_factor = 6415 :=
sorry

end NUMINAMATH_GPT_carrie_jellybeans_l903_90344


namespace NUMINAMATH_GPT_func_passes_through_fixed_point_l903_90325

theorem func_passes_through_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  a^(2 * (1 / 2) - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_func_passes_through_fixed_point_l903_90325


namespace NUMINAMATH_GPT_lines_of_first_character_l903_90392

-- Definitions for the number of lines each character has
def L3 : Nat := 2

def L2 : Nat := 3 * L3 + 6

def L1 : Nat := L2 + 8

-- The theorem we are proving
theorem lines_of_first_character : L1 = 20 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_lines_of_first_character_l903_90392


namespace NUMINAMATH_GPT_jason_money_l903_90365

theorem jason_money (fred_money_before : ℕ) (jason_money_before : ℕ)
  (fred_money_after : ℕ) (total_earned : ℕ) :
  fred_money_before = 111 →
  jason_money_before = 40 →
  fred_money_after = 115 →
  total_earned = 4 →
  jason_money_before = 40 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jason_money_l903_90365


namespace NUMINAMATH_GPT_two_layers_area_zero_l903_90348

theorem two_layers_area_zero (A X Y Z : ℕ)
  (h1 : A = 212)
  (h2 : X + Y + Z = 140)
  (h3 : Y + Z = 24)
  (h4 : Z = 24) : Y = 0 :=
by
  sorry

end NUMINAMATH_GPT_two_layers_area_zero_l903_90348


namespace NUMINAMATH_GPT_boundary_shadow_function_l903_90303

theorem boundary_shadow_function 
    (r : ℝ) (O P : ℝ × ℝ × ℝ) (f : ℝ → ℝ)
    (h_radius : r = 1)
    (h_center : O = (1, 0, 1))
    (h_light_source : P = (1, -1, 2)) :
  (∀ x, f x = (x - 1) ^ 2 / 4 - 1) := 
by 
  sorry

end NUMINAMATH_GPT_boundary_shadow_function_l903_90303


namespace NUMINAMATH_GPT_team_selection_ways_l903_90332

theorem team_selection_ways :
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose boys team_size_boys * choose girls team_size_girls = 103950 :=
by
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end NUMINAMATH_GPT_team_selection_ways_l903_90332


namespace NUMINAMATH_GPT_calculate_X_l903_90304

theorem calculate_X
  (top_seg1 : ℕ) (top_seg2 : ℕ) (X : ℕ)
  (vert_seg : ℕ)
  (bottom_seg1 : ℕ) (bottom_seg2 : ℕ) (bottom_seg3 : ℕ)
  (h1 : top_seg1 = 3) (h2 : top_seg2 = 2)
  (h3 : vert_seg = 4)
  (h4 : bottom_seg1 = 4) (h5 : bottom_seg2 = 2) (h6 : bottom_seg3 = 5)
  (h_eq : 5 + X = 11) :
  X = 6 :=
by
  -- Proof is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_calculate_X_l903_90304


namespace NUMINAMATH_GPT_compute_moles_of_NaHCO3_l903_90305

def equilibrium_constant : Real := 7.85 * 10^5

def balanced_equation (NaHCO3 HCl H2O CO2 NaCl : ℝ) : Prop :=
  NaHCO3 = HCl ∧ NaHCO3 = H2O ∧ NaHCO3 = CO2 ∧ NaHCO3 = NaCl

theorem compute_moles_of_NaHCO3
  (K : Real)
  (hK : K = 7.85 * 10^5)
  (HCl_required : ℝ)
  (hHCl : HCl_required = 2)
  (Water_formed : ℝ)
  (hWater : Water_formed = 2)
  (CO2_formed : ℝ)
  (hCO2 : CO2_formed = 2)
  (NaCl_formed : ℝ)
  (hNaCl : NaCl_formed = 2) :
  ∃ NaHCO3 : ℝ, NaHCO3 = 2 :=
by
  -- Conditions: equilibrium constant, balanced equation
  have equilibrium_condition := equilibrium_constant
  -- Here you would normally work through the steps of the proof using the given conditions,
  -- but we are setting it up as a theorem without a proof for now.
  existsi 2
  -- Placeholder for the formal proof.
  sorry

end NUMINAMATH_GPT_compute_moles_of_NaHCO3_l903_90305


namespace NUMINAMATH_GPT_sum_of_coordinates_of_B_l903_90375

theorem sum_of_coordinates_of_B (x y : ℕ) (hM : (2 * 6 = x + 10) ∧ (2 * 8 = y + 8)) :
    x + y = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_B_l903_90375


namespace NUMINAMATH_GPT_inequality_y_lt_x_div_4_l903_90340

open Real

/-- Problem statement:
Given x ∈ (0, π / 6) and y ∈ (0, π / 6), and x * tan y = 2 * (1 - cos x),
prove that y < x / 4.
-/
theorem inequality_y_lt_x_div_4
  (x y : ℝ)
  (hx : 0 < x ∧ x < π / 6)
  (hy : 0 < y ∧ y < π / 6)
  (h : x * tan y = 2 * (1 - cos x)) :
  y < x / 4 := sorry

end NUMINAMATH_GPT_inequality_y_lt_x_div_4_l903_90340


namespace NUMINAMATH_GPT_find_a_odd_function_l903_90389

theorem find_a_odd_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, 0 < x → f x = 1 + a^x)
  (h3 : 0 < a)
  (h4 : a ≠ 1)
  (h5 : f (-1) = -3 / 2) :
  a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_odd_function_l903_90389


namespace NUMINAMATH_GPT_consecutive_sum_36_unique_l903_90301

def is_consecutive_sum (a b n : ℕ) :=
  (0 < n) ∧ ((n ≥ 2) ∧ (b = a + n - 1) ∧ (2 * a + n - 1) * n = 72)

theorem consecutive_sum_36_unique :
  ∃! n, ∃ a b, is_consecutive_sum a b n :=
by
  sorry

end NUMINAMATH_GPT_consecutive_sum_36_unique_l903_90301


namespace NUMINAMATH_GPT_value_of_a_l903_90398

theorem value_of_a (a b : ℝ) (h1 : b = 2120) (h2 : a / b = 0.5) : a = 1060 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_l903_90398


namespace NUMINAMATH_GPT_average_age_students_l903_90322

theorem average_age_students 
  (total_students : ℕ)
  (group1 : ℕ)
  (group1_avg_age : ℕ)
  (group2 : ℕ)
  (group2_avg_age : ℕ)
  (student15_age : ℕ)
  (avg_age : ℕ) 
  (h1 : total_students = 15)
  (h2 : group1_avg_age = 14)
  (h3 : group2 = 8)
  (h4 : group2_avg_age = 16)
  (h5 : student15_age = 13)
  (h6 : avg_age = (84 + 128 + 13) / 15)
  (h7 : avg_age = 15) :
  group1 = 6 :=
by sorry

end NUMINAMATH_GPT_average_age_students_l903_90322


namespace NUMINAMATH_GPT_album_cost_l903_90309

-- Definitions for given conditions
def M (X : ℕ) : ℕ := X - 2
def K (X : ℕ) : ℕ := X - 34
def F (X : ℕ) : ℕ := X - 35

-- We need to prove that X = 35
theorem album_cost : ∃ X : ℕ, (M X) + (K X) + (F X) < X ∧ X = 35 :=
by
  sorry -- Proof not required.

end NUMINAMATH_GPT_album_cost_l903_90309


namespace NUMINAMATH_GPT_abc_over_sum_leq_four_thirds_l903_90331

theorem abc_over_sum_leq_four_thirds (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_a_leq_2 : a ≤ 2) (h_b_leq_2 : b ≤ 2) (h_c_leq_2 : c ≤ 2) :
  (abc / (a + b + c) ≤ 4/3) :=
by
  sorry

end NUMINAMATH_GPT_abc_over_sum_leq_four_thirds_l903_90331


namespace NUMINAMATH_GPT_hill_height_l903_90338

theorem hill_height (h : ℝ) (time_up : ℝ := h / 9) (time_down : ℝ := h / 12) (total_time : ℝ := time_up + time_down) (time_cond : total_time = 175) : h = 900 :=
by 
  sorry

end NUMINAMATH_GPT_hill_height_l903_90338


namespace NUMINAMATH_GPT_exists_saddle_point_probability_l903_90314

noncomputable def saddle_point_probability := (3 : ℝ) / 10

theorem exists_saddle_point_probability {A : ℕ → ℕ → ℝ}
  (h : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1 ∧ (∀ k l, (i ≠ k ∨ j ≠ l) → A i j ≠ A k l)) :
  (∃ (p : ℝ), p = saddle_point_probability) :=
by 
  sorry

end NUMINAMATH_GPT_exists_saddle_point_probability_l903_90314


namespace NUMINAMATH_GPT_find_g_at_4_l903_90353

theorem find_g_at_4 (g : ℝ → ℝ) (h : ∀ x, 2 * g x + 3 * g (1 - x) = 4 * x^3 - x) : g 4 = 193.2 :=
sorry

end NUMINAMATH_GPT_find_g_at_4_l903_90353


namespace NUMINAMATH_GPT_months_after_withdrawal_and_advance_eq_eight_l903_90350

-- Define initial conditions
def initial_investment_A : ℝ := 3000
def initial_investment_B : ℝ := 4000
def withdrawal_A : ℝ := 1000
def advancement_B : ℝ := 1000
def total_profit : ℝ := 630
def share_A : ℝ := 240
def share_B : ℝ := total_profit - share_A

-- Define the main proof problem
theorem months_after_withdrawal_and_advance_eq_eight
  (initial_investment_A : ℝ) (initial_investment_B : ℝ)
  (withdrawal_A : ℝ) (advancement_B : ℝ)
  (total_profit : ℝ) (share_A : ℝ) (share_B : ℝ) : 
  ∃ x : ℝ, 
  (3000 * x + 2000 * (12 - x)) / (4000 * x + 5000 * (12 - x)) = 240 / 390 ∧
  x = 8 :=
sorry

end NUMINAMATH_GPT_months_after_withdrawal_and_advance_eq_eight_l903_90350


namespace NUMINAMATH_GPT_least_sum_possible_l903_90312

theorem least_sum_possible (x y z w k : ℕ) (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) 
  (hx : 4 * x = k) (hy : 5 * y = k) (hz : 6 * z = k) (hw : 7 * w = k) :
  x + y + z + w = 319 := 
  sorry

end NUMINAMATH_GPT_least_sum_possible_l903_90312


namespace NUMINAMATH_GPT_sum_of_factors_of_30_is_72_l903_90387

-- Condition: given the number 30
def number := 30

-- Define the positive factors of 30
def factors : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- Statement to prove the sum of the positive factors
theorem sum_of_factors_of_30_is_72 : (factors.sum) = 72 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_factors_of_30_is_72_l903_90387


namespace NUMINAMATH_GPT_cats_not_eating_either_l903_90319

theorem cats_not_eating_either (total_cats : ℕ) (cats_like_apples : ℕ) (cats_like_chicken : ℕ) (cats_like_both : ℕ) 
  (h1 : total_cats = 80)
  (h2 : cats_like_apples = 15)
  (h3 : cats_like_chicken = 60)
  (h4 : cats_like_both = 10) : 
  total_cats - (cats_like_apples + cats_like_chicken - cats_like_both) = 15 :=
by sorry

end NUMINAMATH_GPT_cats_not_eating_either_l903_90319


namespace NUMINAMATH_GPT_solve_expression_l903_90335

theorem solve_expression (a x : ℝ) (h1 : a ≠ 0) (h2 : x ≠ a) : 
  (a / (2 * a + x) - x / (a - x)) / (x / (2 * a + x) + a / (a - x)) = -1 → 
  x = a / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l903_90335


namespace NUMINAMATH_GPT_negative_exp_eq_l903_90371

theorem negative_exp_eq :
  (-2 : ℤ)^3 = (-2 : ℤ)^3 := by
  sorry

end NUMINAMATH_GPT_negative_exp_eq_l903_90371


namespace NUMINAMATH_GPT_olympiad_permutations_l903_90349

theorem olympiad_permutations : 
  let total_permutations := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) 
  let invalid_permutations := 5 * (Nat.factorial 4 / Nat.factorial 2)
  total_permutations - invalid_permutations = 90660 :=
by
  let total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
  let invalid_permutations : ℕ := 5 * (Nat.factorial 4 / Nat.factorial 2)
  show total_permutations - invalid_permutations = 90660
  sorry

end NUMINAMATH_GPT_olympiad_permutations_l903_90349


namespace NUMINAMATH_GPT_x_intercept_of_perpendicular_line_l903_90366

theorem x_intercept_of_perpendicular_line 
  (a : ℝ)
  (l1 : ℝ → ℝ → Prop)
  (l1_eq : ∀ x y, l1 x y ↔ (a+3)*x + y - 4 = 0)
  (l2 : ℝ → ℝ → Prop)
  (l2_eq : ∀ x y, l2 x y ↔ x + (a-1)*y + 4 = 0)
  (perpendicular : ∀ x y, l1 x y → l2 x y → (a+3)*(a-1) = -1) :
  (∃ x : ℝ, l1 x 0 ∧ x = 2) :=
sorry

end NUMINAMATH_GPT_x_intercept_of_perpendicular_line_l903_90366


namespace NUMINAMATH_GPT_sarah_jamie_julien_ratio_l903_90378

theorem sarah_jamie_julien_ratio (S J : ℕ) (R : ℝ) :
  -- Conditions
  (J = S + 20) ∧
  (S = R * 50) ∧
  (7 * (J + S + 50) = 1890) ∧
  -- Prove the ratio
  R = 2 := by
  sorry

end NUMINAMATH_GPT_sarah_jamie_julien_ratio_l903_90378


namespace NUMINAMATH_GPT_suraj_average_after_13th_innings_l903_90342

theorem suraj_average_after_13th_innings
  (A : ℝ)
  (h : (12 * A + 96) / 13 = A + 5) :
  (12 * A + 96) / 13 = 36 :=
by
  sorry

end NUMINAMATH_GPT_suraj_average_after_13th_innings_l903_90342


namespace NUMINAMATH_GPT_inequality_additive_l903_90379

variable {a b c d : ℝ}

theorem inequality_additive (h1 : a > b) (h2 : c > d) : a + c > b + d :=
by
  sorry

end NUMINAMATH_GPT_inequality_additive_l903_90379


namespace NUMINAMATH_GPT_find_length_of_side_c_l903_90320

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

/-- Given that in triangle ABC, sin C = 1 / 2, a = 2 * sqrt 3, b = 2,
we want to prove the length of side c is either 2 or 2 * sqrt 7. -/
theorem find_length_of_side_c (C : Real) (a b c : Real) (h1 : Real.sin C = 1 / 2)
  (h2 : a = 2 * Real.sqrt 3) (h3 : b = 2) :
  c = 2 ∨ c = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_side_c_l903_90320


namespace NUMINAMATH_GPT_math_proof_l903_90336

noncomputable def side_length_of_smaller_square (d e f : ℕ) : ℝ :=
  (d - Real.sqrt e) / f

def are_positive_integers (d e f : ℕ) : Prop := d > 0 ∧ e > 0 ∧ f > 0
def is_not_divisible_by_square_of_any_prime (e : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p * p ∣ e)

def proof_problem : Prop :=
  ∃ (d e f : ℕ),
    are_positive_integers d e f ∧
    is_not_divisible_by_square_of_any_prime e ∧
    side_length_of_smaller_square d e f = (4 - Real.sqrt 10) / 3 ∧
    d + e + f = 17

theorem math_proof : proof_problem := sorry

end NUMINAMATH_GPT_math_proof_l903_90336


namespace NUMINAMATH_GPT_tap_C_fills_in_6_l903_90358

-- Definitions for the rates at which taps fill the tank
def rate_A := 1/10
def rate_B := 1/15
def rate_combined := 1/3

-- Proof problem: Given the conditions, prove that the third tap fills the tank in 6 hours
theorem tap_C_fills_in_6 (rate_A rate_B rate_combined : ℚ) (h : rate_A + rate_B + 1/x = rate_combined) : x = 6 :=
sorry

end NUMINAMATH_GPT_tap_C_fills_in_6_l903_90358


namespace NUMINAMATH_GPT_large_square_min_side_and_R_max_area_l903_90341

-- Define the conditions
variable (s : ℝ) -- the side length of the larger square
variable (rect_1_side1 rect_1_side2 : ℝ) -- sides of the first rectangle
variable (square_side : ℝ) -- side of the inscribed square
variable (R_area : ℝ) -- area of the rectangle R

-- The known dimensions
axiom h1 : rect_1_side1 = 2
axiom h2 : rect_1_side2 = 4
axiom h3 : square_side = 2
axiom h4 : ∀ x y : ℝ, x > 0 → y > 0 → R_area = x * y -- non-overlapping condition

-- Define the result to be proved
theorem large_square_min_side_and_R_max_area 
  (h_r_fit_1 : rect_1_side1 + square_side ≤ s)
  (h_r_fit_2 : rect_1_side2 + square_side ≤ s)
  (h_R_max_area : R_area = 4)
  : s = 4 ∧ R_area = 4 := 
by 
  sorry

end NUMINAMATH_GPT_large_square_min_side_and_R_max_area_l903_90341


namespace NUMINAMATH_GPT_rational_inequality_solution_l903_90351

theorem rational_inequality_solution {x : ℝ} : (4 / (x + 1) ≤ 1) → (x ∈ Set.Iic (-1) ∪ Set.Ici 3) :=
by 
  sorry

end NUMINAMATH_GPT_rational_inequality_solution_l903_90351


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l903_90311

theorem sum_of_squares_of_roots : 
  (∃ r1 r2 : ℝ, r1 + r2 = 11 ∧ r1 * r2 = 12 ∧ (r1 ^ 2 + r2 ^ 2) = 97) := 
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l903_90311


namespace NUMINAMATH_GPT_find_y_l903_90374

theorem find_y (x y: ℝ) (h1: x = 680) (h2: 0.25 * x = 0.20 * y - 30) : y = 1000 :=
by 
  sorry

end NUMINAMATH_GPT_find_y_l903_90374


namespace NUMINAMATH_GPT_helen_baked_more_raisin_cookies_l903_90308

-- Definitions based on conditions
def raisin_cookies_yesterday : ℕ := 300
def raisin_cookies_day_before : ℕ := 280

-- Theorem to prove the answer
theorem helen_baked_more_raisin_cookies : raisin_cookies_yesterday - raisin_cookies_day_before = 20 :=
by
  sorry

end NUMINAMATH_GPT_helen_baked_more_raisin_cookies_l903_90308


namespace NUMINAMATH_GPT_infinite_div_pairs_l903_90357

theorem infinite_div_pairs {a : ℕ → ℕ} (h_seq : ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001) :
  ∃ (s : ℕ → (ℕ × ℕ)), (∀ n, (s n).2 < (s n).1) ∧ (a ((s n).2) ∣ a ((s n).1)) :=
sorry

end NUMINAMATH_GPT_infinite_div_pairs_l903_90357


namespace NUMINAMATH_GPT_area_of_EFGH_l903_90390

def short_side_length : ℕ := 4
def long_side_length : ℕ := short_side_length * 2
def number_of_rectangles : ℕ := 4
def larger_rectangle_length : ℕ := short_side_length
def larger_rectangle_width : ℕ := number_of_rectangles * long_side_length

theorem area_of_EFGH :
  (larger_rectangle_length * larger_rectangle_width) = 128 := 
  by
    sorry

end NUMINAMATH_GPT_area_of_EFGH_l903_90390


namespace NUMINAMATH_GPT_expression_simplifies_to_49_l903_90364

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplifies_to_49_l903_90364


namespace NUMINAMATH_GPT_two_lines_perpendicular_to_same_plane_are_parallel_l903_90327

/- 
Problem: Let a, b be two lines, and α be a plane. Prove that if a ⊥ α and b ⊥ α, then a ∥ b.
-/

variables {Line Plane : Type} 

def is_parallel (l1 l2 : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry
def is_contained_in (l : Line) (p : Plane) : Prop := sorry

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane)
  (ha_perpendicular : is_perpendicular a α)
  (hb_perpendicular : is_perpendicular b α) :
  is_parallel a b :=
by
  sorry

end NUMINAMATH_GPT_two_lines_perpendicular_to_same_plane_are_parallel_l903_90327


namespace NUMINAMATH_GPT_max_value_y_l903_90376

/-- Given x < 0, the maximum value of y = (1 + x^2) / x is -2 -/
theorem max_value_y {x : ℝ} (h : x < 0) : ∃ y, y = 1 + x^2 / x ∧ y ≤ -2 :=
sorry

end NUMINAMATH_GPT_max_value_y_l903_90376


namespace NUMINAMATH_GPT_sum_of_remainders_is_six_l903_90367

theorem sum_of_remainders_is_six (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
  (a + b + c) % 15 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_is_six_l903_90367


namespace NUMINAMATH_GPT_distance_from_B_l903_90337

theorem distance_from_B (s y : ℝ) 
  (h1 : s^2 = 12)
  (h2 : ∀y, (1 / 2) * y^2 = 12 - y^2)
  (h3 : y = 2 * Real.sqrt 2)
: Real.sqrt ((2 * Real.sqrt 2)^2 + (2 * Real.sqrt 2)^2) = 4 := by
  sorry

end NUMINAMATH_GPT_distance_from_B_l903_90337


namespace NUMINAMATH_GPT_film_radius_l903_90397

theorem film_radius 
  (thickness : ℝ)
  (container_volume : ℝ)
  (r : ℝ)
  (H1 : thickness = 0.25)
  (H2 : container_volume = 128) :
  r = Real.sqrt (512 / Real.pi) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_film_radius_l903_90397
