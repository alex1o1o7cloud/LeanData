import Mathlib

namespace NUMINAMATH_GPT_nearest_integer_to_sum_l1269_126908

theorem nearest_integer_to_sum (x y : ℝ) (h1 : |x| - y = 1) (h2 : |x| * y + x^2 = 2) : Int.ceil (x + y) = 2 :=
sorry

end NUMINAMATH_GPT_nearest_integer_to_sum_l1269_126908


namespace NUMINAMATH_GPT_equation1_unique_solutions_equation2_unique_solutions_l1269_126912

noncomputable def solve_equation1 : ℝ → Prop :=
fun x => x ^ 2 - 4 * x + 1 = 0

noncomputable def solve_equation2 : ℝ → Prop :=
fun x => 2 * x ^ 2 - 3 * x + 1 = 0

theorem equation1_unique_solutions :
  ∀ x, solve_equation1 x ↔ (x = 2 + Real.sqrt 3) ∨ (x = 2 - Real.sqrt 3) := by
  sorry

theorem equation2_unique_solutions :
  ∀ x, solve_equation2 x ↔ (x = 1) ∨ (x = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_equation1_unique_solutions_equation2_unique_solutions_l1269_126912


namespace NUMINAMATH_GPT_apple_and_pear_costs_l1269_126928

theorem apple_and_pear_costs (x y : ℝ) (h1 : x + 2 * y = 194) (h2 : 2 * x + 5 * y = 458) : 
  y = 70 ∧ x = 54 := 
by 
  sorry

end NUMINAMATH_GPT_apple_and_pear_costs_l1269_126928


namespace NUMINAMATH_GPT_money_left_after_shopping_l1269_126929

-- Define the initial amount of money Sandy took for shopping
def initial_amount : ℝ := 310

-- Define the percentage of money spent in decimal form
def percentage_spent : ℝ := 0.30

-- Define the remaining money as per the given conditions
def remaining_money : ℝ := initial_amount * (1 - percentage_spent)

-- The statement we need to prove
theorem money_left_after_shopping :
  remaining_money = 217 :=
by
  sorry

end NUMINAMATH_GPT_money_left_after_shopping_l1269_126929


namespace NUMINAMATH_GPT_loss_percentage_l1269_126907

-- Definitions of cost price (C) and selling price (S)
def cost_price : ℤ := sorry
def selling_price : ℤ := sorry

-- Given condition: Cost price of 40 articles equals selling price of 25 articles
axiom condition : 40 * cost_price = 25 * selling_price

-- Statement to prove: The merchant made a loss of 20%
theorem loss_percentage (C S : ℤ) (h : 40 * C = 25 * S) : 
  ((S - C) * 100) / C = -20 := 
sorry

end NUMINAMATH_GPT_loss_percentage_l1269_126907


namespace NUMINAMATH_GPT_image_length_interval_two_at_least_four_l1269_126978

noncomputable def quadratic_function (p q r : ℝ) : ℝ → ℝ :=
  fun x => p * (x - q)^2 + r

theorem image_length_interval_two_at_least_four (p q r : ℝ)
  (h : ∀ I : Set ℝ, (∀ a b : ℝ, I = Set.Icc a b ∨ I = Set.Ioo a b → |b - a| = 1 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 1)) :
  ∀ I' : Set ℝ, (∀ a b : ℝ, I' = Set.Icc a b ∨ I' = Set.Ioo a b → |b - a| = 2 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 4) :=
by
  sorry


end NUMINAMATH_GPT_image_length_interval_two_at_least_four_l1269_126978


namespace NUMINAMATH_GPT_average_water_per_day_l1269_126920

variable (day1 : ℕ)
variable (day2 : ℕ)
variable (day3 : ℕ)

def total_water_over_three_days (d1 d2 d3 : ℕ) := d1 + d2 + d3

theorem average_water_per_day :
  day1 = 215 ->
  day2 = 215 + 76 ->
  day3 = 291 - 53 ->
  (total_water_over_three_days day1 day2 day3) / 3 = 248 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_average_water_per_day_l1269_126920


namespace NUMINAMATH_GPT_donation_to_second_orphanage_l1269_126984

variable (total_donation : ℝ) (first_donation : ℝ) (third_donation : ℝ)

theorem donation_to_second_orphanage :
  total_donation = 650 ∧ first_donation = 175 ∧ third_donation = 250 →
  (total_donation - first_donation - third_donation = 225) := by
  sorry

end NUMINAMATH_GPT_donation_to_second_orphanage_l1269_126984


namespace NUMINAMATH_GPT_inequality_bound_l1269_126990

theorem inequality_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) :
  |x^2 - ax - a^2| ≤ 5 / 4 :=
sorry

end NUMINAMATH_GPT_inequality_bound_l1269_126990


namespace NUMINAMATH_GPT_park_needs_minimum_37_nests_l1269_126937

-- Defining the number of different birds
def num_sparrows : ℕ := 5
def num_pigeons : ℕ := 3
def num_starlings : ℕ := 6
def num_robins : ℕ := 2

-- Defining the nesting requirements for each bird species
def nests_per_sparrow : ℕ := 1
def nests_per_pigeon : ℕ := 2
def nests_per_starling : ℕ := 3
def nests_per_robin : ℕ := 4

-- Definition of total minimum nests required
def min_nests_required : ℕ :=
  (num_sparrows * nests_per_sparrow) +
  (num_pigeons * nests_per_pigeon) +
  (num_starlings * nests_per_starling) +
  (num_robins * nests_per_robin)

-- Proof Statement
theorem park_needs_minimum_37_nests :
  min_nests_required = 37 :=
sorry

end NUMINAMATH_GPT_park_needs_minimum_37_nests_l1269_126937


namespace NUMINAMATH_GPT_area_square_diagonal_l1269_126977

theorem area_square_diagonal (d : ℝ) (k : ℝ) :
  (∀ side : ℝ, d^2 = 2 * side^2 → side^2 = (d^2)/2) →
  (∀ A : ℝ, A = (d^2)/2 → A = k * d^2) →
  k = 1/2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_area_square_diagonal_l1269_126977


namespace NUMINAMATH_GPT_value_of_f_prime_at_1_l1269_126938

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem value_of_f_prime_at_1 : deriv f 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_prime_at_1_l1269_126938


namespace NUMINAMATH_GPT_election_result_l1269_126914

theorem election_result (Vx Vy Vz : ℝ) (Pz : ℝ)
  (h1 : Vx = 3 * (Vx / 3)) (h2 : Vy = 2 * (Vy / 2)) (h3 : Vz = 1 * (Vz / 1))
  (h4 : 0.63 * (Vx + Vy + Vz) = 0.74 * Vx + 0.67 * Vy + Pz * Vz) :
  Pz = 0.22 :=
by
  -- proof steps would go here
  -- sorry to keep the proof incomplete
  sorry

end NUMINAMATH_GPT_election_result_l1269_126914


namespace NUMINAMATH_GPT_prism_volume_l1269_126925

theorem prism_volume (x : ℝ) (L W H : ℝ) (hL : L = 2 * x) (hW : W = x) (hH : H = 1.5 * x) 
  (hedges_sum : 4 * L + 4 * W + 4 * H = 72) : 
  L * W * H = 192 := 
by
  sorry

end NUMINAMATH_GPT_prism_volume_l1269_126925


namespace NUMINAMATH_GPT_all_are_truth_tellers_l1269_126941

-- Define the possible states for Alice, Bob, and Carol
inductive State
| true_teller
| liar

-- Define the predicates for each person's statements
def alice_statement (B C : State) : Prop :=
  B = State.true_teller ∨ C = State.true_teller

def bob_statement (A C : State) : Prop :=
  A = State.true_teller ∧ C = State.true_teller

def carol_statement (A B : State) : Prop :=
  A = State.true_teller → B = State.true_teller

-- The theorem to be proved
theorem all_are_truth_tellers
    (A B C : State)
    (alice: A = State.true_teller → alice_statement B C)
    (bob: B = State.true_teller → bob_statement A C)
    (carol: C = State.true_teller → carol_statement A B)
    : A = State.true_teller ∧ B = State.true_teller ∧ C = State.true_teller :=
by
  sorry

end NUMINAMATH_GPT_all_are_truth_tellers_l1269_126941


namespace NUMINAMATH_GPT_team_total_points_l1269_126968

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end NUMINAMATH_GPT_team_total_points_l1269_126968


namespace NUMINAMATH_GPT_expression_value_l1269_126958

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
    sorry

end NUMINAMATH_GPT_expression_value_l1269_126958


namespace NUMINAMATH_GPT_cookie_cost_l1269_126902

theorem cookie_cost 
    (initial_amount : ℝ := 100)
    (latte_cost : ℝ := 3.75)
    (croissant_cost : ℝ := 3.50)
    (days : ℕ := 7)
    (num_cookies : ℕ := 5)
    (remaining_amount : ℝ := 43) :
    (initial_amount - remaining_amount - (days * (latte_cost + croissant_cost))) / num_cookies = 1.25 := 
by
  sorry

end NUMINAMATH_GPT_cookie_cost_l1269_126902


namespace NUMINAMATH_GPT_abs_x_minus_one_iff_x_in_interval_l1269_126913

theorem abs_x_minus_one_iff_x_in_interval (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_minus_one_iff_x_in_interval_l1269_126913


namespace NUMINAMATH_GPT_Karl_miles_driven_l1269_126952

theorem Karl_miles_driven
  (gas_per_mile : ℝ)
  (tank_capacity : ℝ)
  (initial_gas : ℝ)
  (first_leg_miles : ℝ)
  (refuel_gallons : ℝ)
  (final_gas_fraction : ℝ)
  (total_miles_driven : ℝ) :
  gas_per_mile = 30 →
  tank_capacity = 16 →
  initial_gas = 16 →
  first_leg_miles = 420 →
  refuel_gallons = 10 →
  final_gas_fraction = 3 / 4 →
  total_miles_driven = 420 :=
by
  sorry

end NUMINAMATH_GPT_Karl_miles_driven_l1269_126952


namespace NUMINAMATH_GPT_sum_of_coords_of_circle_center_l1269_126963

theorem sum_of_coords_of_circle_center (x y : ℝ) :
  (x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coords_of_circle_center_l1269_126963


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1269_126904

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_series_common_ratio_l1269_126904


namespace NUMINAMATH_GPT_find_solution_l1269_126992

open Nat

def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

noncomputable def expression (n : ℕ) : ℕ :=
  1 + binomial n 1 + binomial n 2 + binomial n 3

theorem find_solution (n : ℕ) (h : n > 3) :
  expression n ∣ 2 ^ 2000 ↔ n = 7 ∨ n = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_solution_l1269_126992


namespace NUMINAMATH_GPT_number_of_impossible_d_vals_is_infinite_l1269_126916

theorem number_of_impossible_d_vals_is_infinite
  (t_1 t_2 s d : ℕ)
  (h1 : 2 * t_1 + t_2 - 4 * s = 4041)
  (h2 : t_1 = s + 2 * d)
  (h3 : t_2 = s + d)
  (h4 : 4 * s > 0) :
  ∀ n : ℕ, n ≠ 808 * 5 ↔ ∃ d, d > 0 ∧ d ≠ n :=
sorry

end NUMINAMATH_GPT_number_of_impossible_d_vals_is_infinite_l1269_126916


namespace NUMINAMATH_GPT_volume_ratio_octahedron_cube_l1269_126986

theorem volume_ratio_octahedron_cube 
  (s : ℝ) -- edge length of the octahedron
  (h := s * Real.sqrt 2 / 2) -- height of one of the pyramids forming the octahedron
  (volume_O := s^3 * Real.sqrt 2 / 3) -- volume of the octahedron
  (a := (2 * s) / Real.sqrt 3) -- edge length of the cube
  (volume_C := (a ^ 3)) -- volume of the cube
  (diag_C : ℝ := 2 * s) -- diagonal of the cube
  (h_diag : diag_C = (a * Real.sqrt 3)) -- relation of diagonal to edge length of the cube
  (ratio := volume_O / volume_C) -- ratio of the volumes
  (desired_ratio := 3 / 8) -- given ratio in simplified form
  (m := 3) -- first part of the ratio
  (n := 8) -- second part of the ratio
  (rel_prime : Nat.gcd m n = 1) -- m and n are relatively prime
  (correct_ratio : ratio = desired_ratio) -- the ratio is correct
  : m + n = 11 :=
by
  sorry 

end NUMINAMATH_GPT_volume_ratio_octahedron_cube_l1269_126986


namespace NUMINAMATH_GPT_tree_planting_equation_l1269_126927

variables (x : ℝ)

theorem tree_planting_equation (h1 : x > 50) :
  (300 / (x - 50) = 400 / x) ≠ False :=
by
  sorry

end NUMINAMATH_GPT_tree_planting_equation_l1269_126927


namespace NUMINAMATH_GPT_problem_solution_l1269_126965

-- Define the necessary conditions
def f (x : ℤ) : ℤ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Define the main theorem
theorem problem_solution :
  (Nat.gcd 840 1785 = 105) ∧ (f 2 = 62) :=
by {
  -- We include sorry here to indicate that the proof is omitted.
  sorry
}

end NUMINAMATH_GPT_problem_solution_l1269_126965


namespace NUMINAMATH_GPT_reasoning_is_wrong_l1269_126906

-- Definitions of the conditions
def some_rationals_are_proper_fractions := ∃ q : ℚ, ∃ f : ℚ, q = f ∧ f.den ≠ 1
def integers_are_rationals := ∀ z : ℤ, ∃ q : ℚ, q = z

-- Proof that the form of reasoning is wrong given the conditions
theorem reasoning_is_wrong 
  (h₁ : some_rationals_are_proper_fractions) 
  (h₂ : integers_are_rationals) :
  ¬ (∀ z : ℤ, ∃ f : ℚ, z = f ∧ f.den ≠ 1) := 
sorry

end NUMINAMATH_GPT_reasoning_is_wrong_l1269_126906


namespace NUMINAMATH_GPT_inequality_proof_l1269_126933

variable {x y : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hxy : x > y) :
    2 * x + 1 / (x ^ 2 - 2 * x * y + y ^ 2) ≥ 2 * y + 3 := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1269_126933


namespace NUMINAMATH_GPT_existence_of_inf_polynomials_l1269_126919

noncomputable def P_xy_defined (P : ℝ→ℝ) (x y z : ℝ) :=
  P x ^ 2 + P y ^ 2 + P z ^ 2 + 2 * P x * P y * P z = 1

theorem existence_of_inf_polynomials (x y z : ℝ) (P : ℕ → ℝ → ℝ) :
  (x^2 + y^2 + z^2 + 2 * x * y * z = 1) →
  (∀ n, P (n+1) = P n ∘ P n) →
  P_xy_defined (P 0) x y z →
  ∀ n, P_xy_defined (P n) x y z :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_existence_of_inf_polynomials_l1269_126919


namespace NUMINAMATH_GPT_intersection_of_A_and_B_is_5_and_8_l1269_126988

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

theorem intersection_of_A_and_B_is_5_and_8 : A ∩ B = {5, 8} :=
  by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_is_5_and_8_l1269_126988


namespace NUMINAMATH_GPT_sum_divisible_by_49_l1269_126991

theorem sum_divisible_by_49
  {x y z : ℤ} 
  (hx : x % 7 ≠ 0)
  (hy : y % 7 ≠ 0)
  (hz : z % 7 ≠ 0)
  (h : 7 ^ 3 ∣ (x ^ 7 + y ^ 7 + z ^ 7)) : 7^2 ∣ (x + y + z) :=
by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_49_l1269_126991


namespace NUMINAMATH_GPT_handmade_ornaments_l1269_126923

noncomputable def handmade_more_than_1_sixth(O : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * (handmade : ℕ) = 20) : Prop :=
  handmade - (1 / 6 * O) = 20

theorem handmade_ornaments (O handmade : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * handmade = 20) :
  handmade_more_than_1_sixth O h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_handmade_ornaments_l1269_126923


namespace NUMINAMATH_GPT_quadratic_condition_l1269_126995

theorem quadratic_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_quadratic_condition_l1269_126995


namespace NUMINAMATH_GPT_garin_homework_pages_l1269_126997

theorem garin_homework_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
    pages_per_day = 19 → 
    days = 24 → 
    total_pages = pages_per_day * days → 
    total_pages = 456 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_garin_homework_pages_l1269_126997


namespace NUMINAMATH_GPT_exchange_ways_100_yuan_l1269_126975

theorem exchange_ways_100_yuan : ∃ n : ℕ, n = 6 ∧ (∀ (x y : ℕ), 20 * x + 10 * y = 100 ↔ y = 10 - 2 * x):=
by
  sorry

end NUMINAMATH_GPT_exchange_ways_100_yuan_l1269_126975


namespace NUMINAMATH_GPT_area_of_ADFE_l1269_126987

namespace Geometry

open Classical

noncomputable def area_triangle (A B C : Type) [Field A] (area_DBF area_BFC area_FCE : A) : A :=
  let total_area := area_DBF + area_BFC + area_FCE
  let area := (105 : A) / 4
  total_area + area

theorem area_of_ADFE (A B C D E F : Type) [Field A] 
  (area_DBF : A) (area_BFC : A) (area_FCE : A) : 
  area_DBF = 4 → area_BFC = 6 → area_FCE = 5 → 
  area_triangle A B C area_DBF area_BFC area_FCE = (15 : A) + (105 : A) / 4 := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_area_of_ADFE_l1269_126987


namespace NUMINAMATH_GPT_angle_B_value_l1269_126974

theorem angle_B_value (a b c A B : ℝ) (h1 : Real.sqrt 3 * a = 2 * b * Real.sin A) : 
  Real.sin B = Real.sqrt 3 / 2 ↔ (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) :=
by sorry

noncomputable def find_b_value (a : ℝ) (area : ℝ) (A B c : ℝ) (h1 : a = 6) (h2 : area = 6 * Real.sqrt 3) (h3 : c = 4) (h4 : B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) : 
  ℝ := 
if B = Real.pi / 3 then 2 * Real.sqrt 7 else Real.sqrt 76

end NUMINAMATH_GPT_angle_B_value_l1269_126974


namespace NUMINAMATH_GPT_cuboid_volume_l1269_126969

theorem cuboid_volume (x y z : ℝ)
  (h1 : 2 * (x + y) = 20)
  (h2 : 2 * (y + z) = 32)
  (h3 : 2 * (x + z) = 28) : x * y * z = 240 := 
by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l1269_126969


namespace NUMINAMATH_GPT_apples_in_pile_l1269_126945

/-- Assuming an initial pile of 8 apples and adding 5 more apples, there should be 13 apples in total. -/
theorem apples_in_pile (initial_apples added_apples : ℕ) (h1 : initial_apples = 8) (h2 : added_apples = 5) :
  initial_apples + added_apples = 13 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_pile_l1269_126945


namespace NUMINAMATH_GPT_folding_positions_l1269_126921

theorem folding_positions (positions : Finset ℕ) (h_conditions: positions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) : 
  ∃ valid_positions : Finset ℕ, valid_positions = {1, 2, 3, 4, 9, 10, 11, 12} ∧ valid_positions.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_folding_positions_l1269_126921


namespace NUMINAMATH_GPT_question1_question2_l1269_126934

-- Define the function representing the inequality
def inequality (a x : ℝ) : Prop := (a * x - 5) / (x - a) < 0

-- Question 1: Compute the solution set M when a=1
theorem question1 : (setOf (λ x : ℝ => inequality 1 x)) = {x : ℝ | 1 < x ∧ x < 5} :=
by
  sorry

-- Question 2: Determine the range for a such that 3 ∈ M but 5 ∉ M
theorem question2 : (setOf (λ a : ℝ => 3 ∈ (setOf (λ x : ℝ => inequality a x)) ∧ 5 ∉ (setOf (λ x : ℝ => inequality a x)))) = 
  {a : ℝ | (1 ≤ a ∧ a < 5 / 3) ∨ (3 < a ∧ a ≤ 5)} :=
by
  sorry

end NUMINAMATH_GPT_question1_question2_l1269_126934


namespace NUMINAMATH_GPT_ratio_of_ages_l1269_126985

-- Given conditions
def present_age_sum (H J : ℕ) : Prop :=
  H + J = 43

def present_ages (H J : ℕ) : Prop := 
  H = 27 ∧ J = 16

def multiple_of_age (H J k : ℕ) : Prop :=
  H - 5 = k * (J - 5)

-- Prove that the ratio of Henry's age to Jill's age 5 years ago was 2:1
theorem ratio_of_ages (H J k : ℕ) 
  (h_sum : present_age_sum H J)
  (h_present : present_ages H J)
  (h_multiple : multiple_of_age H J k) :
  (H - 5) / (J - 5) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1269_126985


namespace NUMINAMATH_GPT_total_outcomes_l1269_126909

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of events
def num_events : ℕ := 3

-- Theorem statement: asserting the total number of different outcomes
theorem total_outcomes : num_students ^ num_events = 125 :=
by
  sorry

end NUMINAMATH_GPT_total_outcomes_l1269_126909


namespace NUMINAMATH_GPT_quadratic_rewriting_l1269_126931

theorem quadratic_rewriting:
  ∃ (d e f : ℤ), (∀ x : ℝ, 4 * x^2 - 28 * x + 49 = (d * x + e)^2 + f) ∧ d * e = -14 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_rewriting_l1269_126931


namespace NUMINAMATH_GPT_polygon_sides_in_arithmetic_progression_l1269_126976

theorem polygon_sides_in_arithmetic_progression 
  (n : ℕ) 
  (d : ℕ := 3)
  (max_angle : ℕ := 150)
  (sum_of_interior_angles : ℕ := 180 * (n - 2)) 
  (a_n : ℕ := max_angle) : 
  (max_angle - d * (n - 1) + max_angle) * n / 2 = sum_of_interior_angles → 
  n = 28 :=
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_in_arithmetic_progression_l1269_126976


namespace NUMINAMATH_GPT_find_last_year_rate_l1269_126971

-- Define the problem setting with types and values (conditions)
def last_year_rate (r : ℝ) : Prop := 
  -- Let r be the annual interest rate last year
  1.1 * r = 0.09

-- Define the theorem to prove the interest rate last year given this year's rate
theorem find_last_year_rate :
  ∃ r : ℝ, last_year_rate r ∧ r = 0.09 / 1.1 := 
by
  sorry

end NUMINAMATH_GPT_find_last_year_rate_l1269_126971


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1269_126939

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2 * q - 1 = 0) :
  p * (1/2) + 3 * (-1/6) + q = 0 :=
by
  -- placeholders for the actual proof steps
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1269_126939


namespace NUMINAMATH_GPT_route_time_saving_zero_l1269_126979

theorem route_time_saving_zero 
  (distance_X : ℝ) (speed_X : ℝ) 
  (total_distance_Y : ℝ) (construction_distance_Y : ℝ) (construction_speed_Y : ℝ)
  (normal_distance_Y : ℝ) (normal_speed_Y : ℝ)
  (hx1 : distance_X = 7)
  (hx2 : speed_X = 35)
  (hy1 : total_distance_Y = 6)
  (hy2 : construction_distance_Y = 1)
  (hy3 : construction_speed_Y = 10)
  (hy4 : normal_distance_Y = 5)
  (hy5 : normal_speed_Y = 50) :
  (distance_X / speed_X * 60) - 
  ((construction_distance_Y / construction_speed_Y * 60) + 
  (normal_distance_Y / normal_speed_Y * 60)) = 0 := 
sorry

end NUMINAMATH_GPT_route_time_saving_zero_l1269_126979


namespace NUMINAMATH_GPT_smallest_value_of_c_l1269_126973

/-- The polynomial x^3 - cx^2 + dx - 2550 has three positive integer roots,
    and the product of the roots is 2550. Prove that the smallest possible value of c is 42. -/
theorem smallest_value_of_c :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2550 ∧ c = a + b + c) → c = 42 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_c_l1269_126973


namespace NUMINAMATH_GPT_find_ab_l1269_126989

noncomputable def poly (x a b : ℝ) := x^4 + a * x^3 - 5 * x^2 + b * x - 6

theorem find_ab (a b : ℝ) (h : poly 2 a b = 0) : (a = 0 ∧ b = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l1269_126989


namespace NUMINAMATH_GPT_payment_difference_correct_l1269_126999

noncomputable def initial_debt : ℝ := 12000

noncomputable def planA_interest_rate : ℝ := 0.08
noncomputable def planA_compounding_periods : ℕ := 2

noncomputable def planB_interest_rate : ℝ := 0.08

noncomputable def planA_payment_years : ℕ := 4
noncomputable def planA_remaining_years : ℕ := 4

noncomputable def planB_years : ℕ := 8

-- Amount accrued in Plan A after 4 years
noncomputable def planA_amount_after_first_period : ℝ :=
  initial_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_payment_years)

-- Amount paid at the end of first period (two-thirds of total)
noncomputable def planA_first_payment : ℝ :=
  (2/3) * planA_amount_after_first_period

-- Remaining debt after first payment
noncomputable def planA_remaining_debt : ℝ :=
  planA_amount_after_first_period - planA_first_payment

-- Amount accrued on remaining debt after 8 years (second 4-year period)
noncomputable def planA_second_payment : ℝ :=
  planA_remaining_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_remaining_years)

-- Total payment under Plan A
noncomputable def total_payment_planA : ℝ :=
  planA_first_payment + planA_second_payment

-- Total payment under Plan B
noncomputable def total_payment_planB : ℝ :=
  initial_debt * (1 + planB_interest_rate * planB_years)

-- Positive difference between payments
noncomputable def payment_difference : ℝ :=
  total_payment_planB - total_payment_planA

theorem payment_difference_correct :
  payment_difference = 458.52 :=
by
  sorry

end NUMINAMATH_GPT_payment_difference_correct_l1269_126999


namespace NUMINAMATH_GPT_quadratic_real_equal_roots_l1269_126980

theorem quadratic_real_equal_roots (m : ℝ) :
  (∃ x : ℝ, 3*x^2 + (2*m-5)*x + 12 = 0) ↔ (m = 8.5 ∨ m = -3.5) :=
sorry

end NUMINAMATH_GPT_quadratic_real_equal_roots_l1269_126980


namespace NUMINAMATH_GPT_quadratic_k_value_l1269_126940

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_k_value_l1269_126940


namespace NUMINAMATH_GPT_ellipse_hyperbola_eccentricities_l1269_126972

theorem ellipse_hyperbola_eccentricities :
  ∃ x y : ℝ, (2 * x^2 - 5 * x + 2 = 0) ∧ (2 * y^2 - 5 * y + 2 = 0) ∧ 
  ((2 > 1) ∧ (0 < (1/2) ∧ (1/2 < 1))) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_eccentricities_l1269_126972


namespace NUMINAMATH_GPT_permissible_m_values_l1269_126955

theorem permissible_m_values :
  ∀ (m : ℕ) (a : ℝ), 
  (∃ k, 2 ≤ k ∧ k ≤ 4 ∧ (3 / (6 / (2 * m + 1)) ≤ k)) → m = 2 ∨ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_permissible_m_values_l1269_126955


namespace NUMINAMATH_GPT_Debby_jogging_plan_l1269_126932

def Monday_jog : ℝ := 3
def Tuesday_jog : ℝ := Monday_jog * 1.1
def Wednesday_jog : ℝ := 0
def Thursday_jog : ℝ := Tuesday_jog * 1.1
def Saturday_jog : ℝ := Thursday_jog * 2.5
def total_distance : ℝ := Monday_jog + Tuesday_jog + Thursday_jog + Saturday_jog
def weekly_goal : ℝ := 40
def Sunday_jog : ℝ := weekly_goal - total_distance

theorem Debby_jogging_plan :
  Tuesday_jog = 3.3 ∧
  Thursday_jog = 3.63 ∧
  Saturday_jog = 9.075 ∧
  Sunday_jog = 21.995 :=
by
  -- Proof goes here, but is omitted as the problem statement requires only the theorem outline.
  sorry

end NUMINAMATH_GPT_Debby_jogging_plan_l1269_126932


namespace NUMINAMATH_GPT_average_of_xyz_l1269_126951

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_xyz_l1269_126951


namespace NUMINAMATH_GPT_parabola_focus_l1269_126901

theorem parabola_focus (x y : ℝ) (p : ℝ) (h_eq : x^2 = 8 * y) (h_form : x^2 = 4 * p * y) : 
  p = 2 ∧ y = (x^2 / 8) ∧ (0, p) = (0, 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1269_126901


namespace NUMINAMATH_GPT_numbers_square_and_cube_root_l1269_126961

theorem numbers_square_and_cube_root (x : ℝ) : (x^2 = x ∧ x^3 = x) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_numbers_square_and_cube_root_l1269_126961


namespace NUMINAMATH_GPT_M_geq_N_l1269_126950

variable (a b : ℝ)

def M : ℝ := a^2 + 12 * a - 4 * b
def N : ℝ := 4 * a - 20 - b^2

theorem M_geq_N : M a b ≥ N a b := by
  sorry

end NUMINAMATH_GPT_M_geq_N_l1269_126950


namespace NUMINAMATH_GPT_circles_non_intersecting_l1269_126983

def circle1_equation (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def circle2_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem circles_non_intersecting :
    (∀ (x y : ℝ), ¬(circle1_equation x y ∧ circle2_equation x y)) :=
by
  sorry

end NUMINAMATH_GPT_circles_non_intersecting_l1269_126983


namespace NUMINAMATH_GPT_complete_the_square_l1269_126905

theorem complete_the_square (x : ℝ) (h : x^2 + 7 * x - 5 = 0) : (x + 7 / 2) ^ 2 = 69 / 4 :=
sorry

end NUMINAMATH_GPT_complete_the_square_l1269_126905


namespace NUMINAMATH_GPT_temperature_max_time_l1269_126910

theorem temperature_max_time (t : ℝ) (h : 0 ≤ t) : 
  (-t^2 + 10 * t + 60 = 85) → t = 15 := 
sorry

end NUMINAMATH_GPT_temperature_max_time_l1269_126910


namespace NUMINAMATH_GPT_arctan_sum_l1269_126966

theorem arctan_sum (a b : ℝ) (h1 : a = 1/3) (h2 : (a + 1) * (b + 1) = 3) : 
  Real.arctan a + Real.arctan b = Real.arctan (19 / 7) :=
by
  sorry

end NUMINAMATH_GPT_arctan_sum_l1269_126966


namespace NUMINAMATH_GPT_father_l1269_126981

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S) (h2 : F + 15 = 2 * (S + 15)) : F = 45 :=
sorry

end NUMINAMATH_GPT_father_l1269_126981


namespace NUMINAMATH_GPT_sum_mod_9_equal_6_l1269_126903

theorem sum_mod_9_equal_6 :
  ((1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9) = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_9_equal_6_l1269_126903


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1269_126915

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = -1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1269_126915


namespace NUMINAMATH_GPT_value_of_k_through_point_l1269_126967

noncomputable def inverse_proportion_function (x : ℝ) (k : ℝ) : ℝ :=
  k / x

theorem value_of_k_through_point (k : ℝ) (h : k ≠ 0) : inverse_proportion_function 2 k = 3 → k = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_through_point_l1269_126967


namespace NUMINAMATH_GPT_percent_calculation_l1269_126924

theorem percent_calculation (Part Whole : ℝ) (h1 : Part = 120) (h2 : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  sorry

end NUMINAMATH_GPT_percent_calculation_l1269_126924


namespace NUMINAMATH_GPT_option_A_correct_l1269_126936

theorem option_A_correct (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
by sorry

end NUMINAMATH_GPT_option_A_correct_l1269_126936


namespace NUMINAMATH_GPT_solve_triangle_l1269_126911

theorem solve_triangle (a b m₁ m₂ k₃ : ℝ) (h1 : a = m₂ / Real.sin γ) (h2 : b = m₁ / Real.sin γ) : 
  a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := 
  by 
  sorry

end NUMINAMATH_GPT_solve_triangle_l1269_126911


namespace NUMINAMATH_GPT_acme_cheaper_than_beta_l1269_126970

theorem acme_cheaper_than_beta (x : ℕ) :
  (50 + 9 * x < 25 + 15 * x) ↔ (5 ≤ x) :=
by sorry

end NUMINAMATH_GPT_acme_cheaper_than_beta_l1269_126970


namespace NUMINAMATH_GPT_race_positions_l1269_126949

variable (nabeel marzuq arabi rafsan lian rahul : ℕ)

theorem race_positions :
  (arabi = 6) →
  (arabi = rafsan + 1) →
  (rafsan = rahul + 2) →
  (rahul = nabeel + 1) →
  (nabeel = marzuq + 6) →
  (marzuq = 8) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_race_positions_l1269_126949


namespace NUMINAMATH_GPT_odd_function_has_specific_a_l1269_126922

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
x / ((2 * x + 1) * (x - a))

theorem odd_function_has_specific_a :
  ∀ a, is_odd (f a) → a = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_odd_function_has_specific_a_l1269_126922


namespace NUMINAMATH_GPT_arithmetic_sequence_sixth_term_l1269_126996

theorem arithmetic_sequence_sixth_term (a d : ℤ) 
    (sum_first_five : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
    (fourth_term : a + 3 * d = 4) : a + 5 * d = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sixth_term_l1269_126996


namespace NUMINAMATH_GPT_gcd_459_357_l1269_126954

theorem gcd_459_357 : gcd 459 357 = 51 := 
sorry

end NUMINAMATH_GPT_gcd_459_357_l1269_126954


namespace NUMINAMATH_GPT_domain_of_g_eq_l1269_126900

noncomputable def g (x : ℝ) : ℝ := (x + 2) / (Real.sqrt (x^2 - 5 * x + 6))

theorem domain_of_g_eq : 
  {x : ℝ | 0 < x^2 - 5 * x + 6} = {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_g_eq_l1269_126900


namespace NUMINAMATH_GPT_difference_between_oranges_and_apples_l1269_126953

-- Definitions of the conditions
variables (A B P O: ℕ)
variables (h1: O = 6)
variables (h2: B = 3 * A)
variables (h3: P = B / 2)
variables (h4: A + B + P + O = 28)

-- The proof problem statement
theorem difference_between_oranges_and_apples
    (A B P O: ℕ)
    (h1: O = 6)
    (h2: B = 3 * A)
    (h3: P = B / 2)
    (h4: A + B + P + O = 28) :
    O - A = 2 :=
sorry

end NUMINAMATH_GPT_difference_between_oranges_and_apples_l1269_126953


namespace NUMINAMATH_GPT_find_m_if_f_even_l1269_126994

theorem find_m_if_f_even (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = x^4 + (m - 1) * x + 1) ∧ (∀ x : ℝ, f x = f (-x)) → m = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_if_f_even_l1269_126994


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1269_126917

theorem sum_of_x_and_y (x y : ℚ) (h1 : 1/x + 1/y = 3) (h2 : 1/x - 1/y = -7) : x + y = -3/10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1269_126917


namespace NUMINAMATH_GPT_ratio_of_N_to_R_l1269_126930

variables (N T R k : ℝ)

theorem ratio_of_N_to_R (h1 : T = (1 / 4) * N)
                        (h2 : R = 40)
                        (h3 : N = k * R)
                        (h4 : T + R + N = 190) :
    N / R = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_N_to_R_l1269_126930


namespace NUMINAMATH_GPT_find_y_and_y2_l1269_126935

theorem find_y_and_y2 (d y y2 : ℤ) (h1 : 3 ^ 2 = 9) (h2 : 3 ^ 4 = 81)
  (h3 : y = 9 + d) (h4 : y2 = 81 + d) (h5 : 81 = 9 + 3 * d) :
  y = 33 ∧ y2 = 105 :=
by
  sorry

end NUMINAMATH_GPT_find_y_and_y2_l1269_126935


namespace NUMINAMATH_GPT_total_selection_methods_l1269_126947

theorem total_selection_methods (synthetic_students : ℕ) (analytical_students : ℕ)
  (h_synthetic : synthetic_students = 5) (h_analytical : analytical_students = 3) :
  synthetic_students + analytical_students = 8 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_selection_methods_l1269_126947


namespace NUMINAMATH_GPT_julio_salary_l1269_126946

-- Define the conditions
def customers_first_week : ℕ := 35
def customers_second_week : ℕ := 2 * customers_first_week
def customers_third_week : ℕ := 3 * customers_first_week
def commission_per_customer : ℕ := 1
def bonus : ℕ := 50
def total_earnings : ℕ := 760

-- Calculate total commission and total earnings
def commission_first_week : ℕ := customers_first_week * commission_per_customer
def commission_second_week : ℕ := customers_second_week * commission_per_customer
def commission_third_week : ℕ := customers_third_week * commission_per_customer
def total_commission : ℕ := commission_first_week + commission_second_week + commission_third_week
def total_earnings_commission_bonus : ℕ := total_commission + bonus

-- Define the proof problem
theorem julio_salary : total_earnings - total_earnings_commission_bonus = 500 :=
by
  sorry

end NUMINAMATH_GPT_julio_salary_l1269_126946


namespace NUMINAMATH_GPT_cost_price_eq_l1269_126948

variable (SP : Real) (profit_percentage : Real)

theorem cost_price_eq : SP = 100 → profit_percentage = 0.15 → (100 / (1 + profit_percentage)) = 86.96 :=
by
  intros hSP hProfit
  sorry

end NUMINAMATH_GPT_cost_price_eq_l1269_126948


namespace NUMINAMATH_GPT_find_other_num_l1269_126956

variables (a b : ℕ)

theorem find_other_num (h_gcd : Nat.gcd a b = 12) (h_lcm : Nat.lcm a b = 5040) (h_a : a = 240) :
  b = 252 :=
  sorry

end NUMINAMATH_GPT_find_other_num_l1269_126956


namespace NUMINAMATH_GPT_largest_consecutive_sum_55_l1269_126998

theorem largest_consecutive_sum_55 :
  ∃ n a : ℕ, (n * (a + (n - 1) / 2) = 55) ∧ (n = 10) ∧ (∀ m : ℕ, ∀ b : ℕ, (m * (b + (m - 1) / 2) = 55) → (m ≤ 10)) :=
by 
  sorry

end NUMINAMATH_GPT_largest_consecutive_sum_55_l1269_126998


namespace NUMINAMATH_GPT_volleyball_lineup_ways_l1269_126957

def num_ways_lineup (team_size : ℕ) (positions : ℕ) : ℕ :=
  if positions ≤ team_size then
    Nat.descFactorial team_size positions
  else
    0

theorem volleyball_lineup_ways :
  num_ways_lineup 10 5 = 30240 :=
by
  rfl

end NUMINAMATH_GPT_volleyball_lineup_ways_l1269_126957


namespace NUMINAMATH_GPT_find_a_l1269_126959

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end NUMINAMATH_GPT_find_a_l1269_126959


namespace NUMINAMATH_GPT_find_x_l1269_126926

noncomputable def series_sum (x : ℝ) : ℝ :=
∑' n : ℕ, (1 + 6 * n) * x^n

theorem find_x (x : ℝ) (h : series_sum x = 100) (hx : |x| < 1) : x = 3 / 5 := 
sorry

end NUMINAMATH_GPT_find_x_l1269_126926


namespace NUMINAMATH_GPT_michael_and_truck_meet_l1269_126960

/--
Assume:
1. Michael walks at 6 feet per second.
2. Trash pails are every 240 feet.
3. A truck travels at 10 feet per second and stops for 36 seconds at each pail.
4. Initially, when Michael passes a pail, the truck is 240 feet ahead.

Prove:
Michael and the truck meet every 120 seconds starting from 120 seconds.
-/
theorem michael_and_truck_meet (t : ℕ) : t ≥ 120 → (t - 120) % 120 = 0 :=
sorry

end NUMINAMATH_GPT_michael_and_truck_meet_l1269_126960


namespace NUMINAMATH_GPT_intersection_M_N_l1269_126918

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1269_126918


namespace NUMINAMATH_GPT_prime_pairs_divisibility_l1269_126962

theorem prime_pairs_divisibility:
  ∀ (p q : ℕ), (Nat.Prime p ∧ Nat.Prime q ∧ p ≤ q ∧ p * q ∣ ((5 ^ p - 2 ^ p) * (7 ^ q - 2 ^ q))) ↔ 
                (p = 3 ∧ q = 5) ∨ 
                (p = 3 ∧ q = 3) ∨ 
                (p = 5 ∧ q = 37) ∨ 
                (p = 5 ∧ q = 83) := by
  sorry

end NUMINAMATH_GPT_prime_pairs_divisibility_l1269_126962


namespace NUMINAMATH_GPT_overall_loss_is_correct_l1269_126982

-- Define the conditions
def worth_of_stock : ℝ := 17500
def percent_stock_sold_at_profit : ℝ := 0.20
def profit_rate : ℝ := 0.10
def percent_stock_sold_at_loss : ℝ := 0.80
def loss_rate : ℝ := 0.05

-- Define the calculations based on the conditions
def worth_sold_at_profit : ℝ := percent_stock_sold_at_profit * worth_of_stock
def profit_amount : ℝ := profit_rate * worth_sold_at_profit

def worth_sold_at_loss : ℝ := percent_stock_sold_at_loss * worth_of_stock
def loss_amount : ℝ := loss_rate * worth_sold_at_loss

-- Define the overall loss amount
def overall_loss : ℝ := loss_amount - profit_amount

-- Theorem to prove that the calculated overall loss amount matches the expected loss amount
theorem overall_loss_is_correct :
  overall_loss = 350 :=
by
  sorry

end NUMINAMATH_GPT_overall_loss_is_correct_l1269_126982


namespace NUMINAMATH_GPT_find_coeff_a9_l1269_126964

theorem find_coeff_a9 (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (x^3 + x^10 = a + a1 * (x + 1) + a2 * (x + 1)^2 + 
  a3 * (x + 1)^3 + a4 * (x + 1)^4 + a5 * (x + 1)^5 + 
  a6 * (x + 1)^6 + a7 * (x + 1)^7 + a8 * (x + 1)^8 + 
  a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a9 = -10 :=
sorry

end NUMINAMATH_GPT_find_coeff_a9_l1269_126964


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1269_126943

variable {G : Type*} [Field G]

def is_geometric (a : ℕ → G) (q : G) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → G) (q : G)
  (h1 : a 0 + a 1 = 3)
  (h2 : a 1 + a 2 = 6)
  (hq : is_geometric a q) :
  a 6 = 64 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1269_126943


namespace NUMINAMATH_GPT_geom_seq_value_l1269_126944

variable (a_n : ℕ → ℝ)
variable (r : ℝ)
variable (π : ℝ)

-- Define the conditions
axiom geom_seq : ∀ n, a_n (n + 1) = a_n n * r
axiom sum_pi : a_n 3 + a_n 5 = π

-- Statement to prove
theorem geom_seq_value : a_n 4 * (a_n 2 + 2 * a_n 4 + a_n 6) = π^2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_value_l1269_126944


namespace NUMINAMATH_GPT_empty_seats_in_theater_l1269_126993

theorem empty_seats_in_theater :
  let total_seats := 750
  let occupied_seats := 532
  total_seats - occupied_seats = 218 :=
by
  sorry

end NUMINAMATH_GPT_empty_seats_in_theater_l1269_126993


namespace NUMINAMATH_GPT_smallest_solution_l1269_126942

theorem smallest_solution (x : ℝ) (h : x * |x| = 2 * x + 1) : x = -1 := 
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l1269_126942
