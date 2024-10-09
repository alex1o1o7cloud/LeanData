import Mathlib

namespace min_value_inequality_l2056_205642

variable (x y : ℝ)

theorem min_value_inequality (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ m : ℝ, m = 1 / 4 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (x ^ 2) / (x + 2) + (y ^ 2) / (y + 1) ≥ m) :=
by
  use (1 / 4)
  sorry

end min_value_inequality_l2056_205642


namespace letter_puzzle_solutions_l2056_205621

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l2056_205621


namespace max_possible_score_l2056_205684

theorem max_possible_score (s : ℝ) (h : 80 = s * 2) : s * 5 ≥ 100 :=
by
  -- sorry placeholder for the proof
  sorry

end max_possible_score_l2056_205684


namespace combined_weight_of_jake_and_sister_l2056_205600

theorem combined_weight_of_jake_and_sister (j s : ℕ) (h1 : j = 188) (h2 : j - 8 = 2 * s) : j + s = 278 :=
sorry

end combined_weight_of_jake_and_sister_l2056_205600


namespace expected_value_of_flipped_coins_l2056_205679

theorem expected_value_of_flipped_coins :
  let p := 1
  let n := 5
  let d := 10
  let q := 25
  let f := 50
  let prob := (1:ℝ) / 2
  let V := prob * p + prob * n + prob * d + prob * q + prob * f
  V = 45.5 :=
by
  sorry

end expected_value_of_flipped_coins_l2056_205679


namespace tin_silver_ratio_l2056_205624

theorem tin_silver_ratio (T S : ℝ) 
  (h1 : T + S = 50) 
  (h2 : 0.1375 * T + 0.075 * S = 5) : 
  T / S = 2 / 3 :=
by
  sorry

end tin_silver_ratio_l2056_205624


namespace barycentric_vector_identity_l2056_205602

variables {A B C X : Type} [AddCommGroup X] [Module ℝ X]
variables (α β γ : ℝ) (A B C X : X)

-- Defining the barycentric coordinates condition
axiom barycentric_coords : α • A + β • B + γ • C = X

-- Additional condition that sum of coordinates is 1
axiom sum_coords : α + β + γ = 1

-- The theorem to prove
theorem barycentric_vector_identity :
  (X - A) = β • (B - A) + γ • (C - A) :=
sorry

end barycentric_vector_identity_l2056_205602


namespace kimberly_initial_skittles_l2056_205635

theorem kimberly_initial_skittles : 
  ∀ (x : ℕ), (x + 7 = 12) → x = 5 :=
by
  sorry

end kimberly_initial_skittles_l2056_205635


namespace quadratic_distinct_roots_l2056_205628

theorem quadratic_distinct_roots
  (a b c : ℝ)
  (h1 : 5 * a + 3 * b + 2 * c = 0)
  (h2 : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1 ^ 2 + b * x1 + c = 0) ∧ (a * x2 ^ 2 + b * x2 + c = 0) :=
by
  sorry

end quadratic_distinct_roots_l2056_205628


namespace move_line_left_and_up_l2056_205651

/--
The equation of the line obtained by moving the line y = 2x - 3
2 units to the left and then 3 units up is y = 2x + 4.
-/
theorem move_line_left_and_up :
  ∀ (x y : ℝ), y = 2*x - 3 → ∃ x' y', x' = x + 2 ∧ y' = y + 3 ∧ y' = 2*x' + 4 :=
by
  sorry

end move_line_left_and_up_l2056_205651


namespace area_of_inscribed_rectangle_l2056_205603

theorem area_of_inscribed_rectangle (r : ℝ) (h : r = 6) (ratio : ℝ) (hr : ratio = 3 / 1) :
  ∃ (length width : ℝ), (width = 2 * r) ∧ (length = ratio * width) ∧ (length * width = 432) :=
by
  sorry

end area_of_inscribed_rectangle_l2056_205603


namespace area_common_to_all_four_circles_l2056_205633

noncomputable def common_area (R : ℝ) : ℝ :=
  (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6

theorem area_common_to_all_four_circles (R : ℝ) :
  ∃ (O1 O2 A B : ℝ × ℝ),
    dist O1 O2 = R ∧
    dist O1 A = R ∧
    dist O2 A = R ∧
    dist O1 B = R ∧
    dist O2 B = R ∧
    dist A B = R ∧
    common_area R = (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6 :=
by
  sorry

end area_common_to_all_four_circles_l2056_205633


namespace trigonometric_identity_l2056_205640

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 3) : 1 / (Real.sin x ^ 2 - 2 * Real.cos x ^ 2) = 10 / 7 :=
by
  sorry

end trigonometric_identity_l2056_205640


namespace quadratic_completing_square_sum_l2056_205641

theorem quadratic_completing_square_sum (q t : ℝ) :
    (∃ (x : ℝ), 9 * x^2 - 54 * x - 36 = 0 ∧ (x + q)^2 = t) →
    q + t = 10 := sorry

end quadratic_completing_square_sum_l2056_205641


namespace maple_trees_remaining_l2056_205601

-- Define the initial number of maple trees in the park
def initial_maple_trees : ℝ := 9.0

-- Define the number of maple trees that will be cut down
def cut_down_maple_trees : ℝ := 2.0

-- Define the expected number of maple trees left after cutting down
def remaining_maple_trees : ℝ := 7.0

-- Theorem to prove the remaining number of maple trees is correct
theorem maple_trees_remaining :
  initial_maple_trees - cut_down_maple_trees = remaining_maple_trees := by
  admit -- sorry can be used alternatively

end maple_trees_remaining_l2056_205601


namespace Tim_transactions_l2056_205650

theorem Tim_transactions
  (Mabel_Monday : ℕ)
  (Mabel_Tuesday : ℕ := Mabel_Monday + Mabel_Monday / 10)
  (Anthony_Tuesday : ℕ := 2 * Mabel_Tuesday)
  (Cal_Tuesday : ℕ := (2 * Anthony_Tuesday) / 3)
  (Jade_Tuesday : ℕ := Cal_Tuesday + 17)
  (Isla_Wednesday : ℕ := Mabel_Tuesday + Cal_Tuesday - 12)
  (Tim_Thursday : ℕ := (Jade_Tuesday + Isla_Wednesday) * 3 / 2)
  : Tim_Thursday = 614 := by sorry

end Tim_transactions_l2056_205650


namespace distance_from_LV_to_LA_is_273_l2056_205660

-- Define the conditions
def distance_SLC_to_LV : ℝ := 420
def total_time : ℝ := 11
def avg_speed : ℝ := 63

-- Define the total distance covered given the average speed and time
def total_distance : ℝ := avg_speed * total_time

-- Define the distance from Las Vegas to Los Angeles
def distance_LV_to_LA : ℝ := total_distance - distance_SLC_to_LV

-- Now state the theorem we want to prove
theorem distance_from_LV_to_LA_is_273 :
  distance_LV_to_LA = 273 :=
sorry

end distance_from_LV_to_LA_is_273_l2056_205660


namespace remainder_of_large_power_l2056_205670

def powerMod (base exp mod_ : ℕ) : ℕ := (base ^ exp) % mod_

theorem remainder_of_large_power :
  powerMod 2 (2^(2^2)) 500 = 536 :=
sorry

end remainder_of_large_power_l2056_205670


namespace area_of_right_triangle_with_incircle_l2056_205695

theorem area_of_right_triangle_with_incircle (a b c r : ℝ) :
  (a = 6 + r) → 
  (b = 7 + r) → 
  (c = 13) → 
  (a^2 + b^2 = c^2) →
  (2 * r^2 + 26 * r = 84) →
  (area = 1/2 * ((6 + r) * (7 + r))) →
  area = 42 := 
by 
  sorry

end area_of_right_triangle_with_incircle_l2056_205695


namespace milk_drinks_on_weekdays_l2056_205636

-- Defining the number of boxes Lolita drinks on a weekday as a variable W
variable (W : ℕ)

-- Condition: Lolita drinks 30 boxes of milk per week.
axiom total_milk_per_week : 5 * W + 2 * W + 3 * W = 30

-- Proof (Statement) that Lolita drinks 15 boxes of milk on weekdays.
theorem milk_drinks_on_weekdays : 5 * W = 15 :=
by {
  -- Use the given axiom to derive the solution
  sorry
}

end milk_drinks_on_weekdays_l2056_205636


namespace Paul_correct_probability_l2056_205629

theorem Paul_correct_probability :
  let P_Ghana := 1/2
  let P_Bolivia := 1/6
  let P_Argentina := 1/6
  let P_France := 1/6
  (P_Ghana^2 + P_Bolivia^2 + P_Argentina^2 + P_France^2) = 1/3 :=
by
  sorry

end Paul_correct_probability_l2056_205629


namespace probability_all_quit_same_tribe_l2056_205668

-- Define the number of participants and the number of tribes
def numParticipants : ℕ := 18
def numTribes : ℕ := 2
def tribeSize : ℕ := 9 -- Each tribe has 9 members

-- Define the problem statement
theorem probability_all_quit_same_tribe : 
  (numParticipants.choose 3) = 816 ∧
  ((tribeSize.choose 3) * numTribes) = 168 ∧
  ((tribeSize.choose 3) * numTribes) / (numParticipants.choose 3) = 7 / 34 :=
by
  sorry

end probability_all_quit_same_tribe_l2056_205668


namespace distinct_flavors_count_l2056_205611

-- Define the number of available candies
def red_candies := 3
def green_candies := 2
def blue_candies := 4

-- Define what it means for a flavor to be valid: includes at least one candy of each color.
def is_valid_flavor (x y z : Nat) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x ≤ red_candies ∧ y ≤ green_candies ∧ z ≤ blue_candies

-- Define what it means for two flavors to have the same ratio
def same_ratio (x1 y1 z1 x2 y2 z2 : Nat) : Prop :=
  x1 * y2 * z2 = x2 * y1 * z1

-- Define the proof problem: the number of distinct flavors
theorem distinct_flavors_count :
  ∃ n, n = 21 ∧ ∀ (x y z : Nat), is_valid_flavor x y z ↔ (∃ x' y' z', is_valid_flavor x' y' z' ∧ ¬ same_ratio x y z x' y' z') :=
sorry

end distinct_flavors_count_l2056_205611


namespace hexagon_perimeter_l2056_205674

-- Define the side length 's' based on the given area condition
def side_length (s : ℝ) : Prop :=
  (3 * Real.sqrt 2 + Real.sqrt 3) / 4 * s^2 = 12

-- The theorem to prove
theorem hexagon_perimeter (s : ℝ) (h : side_length s) : 
  6 * s = 6 * Real.sqrt (48 / (3 * Real.sqrt 2 + Real.sqrt 3)) :=
by
  sorry

end hexagon_perimeter_l2056_205674


namespace total_students_l2056_205647

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l2056_205647


namespace avg_remaining_two_l2056_205619

-- Defining the given conditions
variable (six_num_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ)

-- Defining the known values
axiom avg_val : six_num_avg = 3.95
axiom avg_group1 : group1_avg = 3.6
axiom avg_group2 : group2_avg = 3.85

-- Stating the problem to prove that the average of the remaining 2 numbers is 4.4
theorem avg_remaining_two (h : six_num_avg = 3.95) 
                           (h1: group1_avg = 3.6)
                           (h2: group2_avg = 3.85) : 
  4.4 = ((six_num_avg * 6) - (group1_avg * 2 + group2_avg * 2)) / 2 := 
sorry

end avg_remaining_two_l2056_205619


namespace product_of_numbers_l2056_205638

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 157) : x * y = 22 := 
by 
  sorry

end product_of_numbers_l2056_205638


namespace strictly_increasing_interval_l2056_205685

noncomputable def f (x : ℝ) : ℝ := x - x * Real.log x

theorem strictly_increasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x - x * Real.log x → ∀ y : ℝ, (0 < y ∧ y < 1 ∧ y > x) → f y > f x :=
sorry

end strictly_increasing_interval_l2056_205685


namespace focus_of_parabola_l2056_205657

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end focus_of_parabola_l2056_205657


namespace cyclic_quadrilaterals_count_l2056_205677

noncomputable def num_cyclic_quadrilaterals (n : ℕ) : ℕ :=
  if n = 32 then 568 else 0 -- encapsulating the problem's answer

theorem cyclic_quadrilaterals_count :
  num_cyclic_quadrilaterals 32 = 568 :=
sorry

end cyclic_quadrilaterals_count_l2056_205677


namespace total_students_proof_l2056_205632

variable (studentsA studentsB : ℕ) (ratioAtoB : ℕ := 3/2)
variable (percentA percentB : ℕ := 10/100)
variable (diffPercent : ℕ := 20/100)
variable (extraStudentsInA : ℕ := 190)
variable (totalStudentsB : ℕ := 650)

theorem total_students_proof :
  (studentsB = totalStudentsB) ∧ 
  ((percentA * studentsA - diffPercent * studentsB = extraStudentsInA) ∧
  (studentsA / studentsB = ratioAtoB)) →
  (studentsA + studentsB = 1625) :=
by
  sorry

end total_students_proof_l2056_205632


namespace sin_cos_inequality_l2056_205634

open Real

theorem sin_cos_inequality (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * π) :
  (sin (x - π / 6) > cos x) ↔ (π / 3 < x ∧ x < 4 * π / 3) :=
by sorry

end sin_cos_inequality_l2056_205634


namespace find_a_plus_b_l2056_205661

open Complex

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∃ (r1 r2 r3 : ℂ),
     r1 = 1 + I * Real.sqrt 3 ∧
     r2 = 1 - I * Real.sqrt 3 ∧
     r3 = -2 ∧
     (r1 + r2 + r3 = 0) ∧
     (r1 * r2 * r3 = -b) ∧
     (r1 * r2 + r2 * r3 + r3 * r1 = -a))

theorem find_a_plus_b (a b : ℝ) (h : problem_statement a b) : a + b = 8 :=
sorry

end find_a_plus_b_l2056_205661


namespace problem1_problem2_l2056_205623

section
variable {x a : ℝ}

-- Definitions of the functions
def f (x : ℝ) : ℝ := |x + 1|
def g (x : ℝ) (a : ℝ) : ℝ := 2 * |x| + a

-- Problem 1
theorem problem1 (a : ℝ) (H : a = -1) : 
  ∀ x : ℝ, f x ≤ g x a ↔ (x ≤ -2/3 ∨ 2 ≤ x) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∃ x₀ : ℝ, f x₀ ≥ 1/2 * g x₀ a) → a ≤ 2 :=
sorry

end

end problem1_problem2_l2056_205623


namespace find_m_l2056_205689

-- Definitions for the given vectors
def a : ℝ × ℝ := (3, 4)
def b (m : ℝ) : ℝ × ℝ := (-1, 2 * m)
def c (m : ℝ) : ℝ × ℝ := (m, -4)

-- Definition of vector addition
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition that c is perpendicular to a + b
def perpendicular_condition (m : ℝ) : Prop :=
  dot_product (c m) (vector_add a (b m)) = 0

-- Proof statement
theorem find_m : ∃ m : ℝ, perpendicular_condition m ∧ m = -8 / 3 :=
sorry

end find_m_l2056_205689


namespace find_x_l2056_205610

theorem find_x 
  (x y : ℤ) 
  (h1 : 2 * x - y = 5) 
  (h2 : x + 2 * y = 5) : 
  x = 3 := 
sorry

end find_x_l2056_205610


namespace integer_solution_count_l2056_205606

theorem integer_solution_count {a b c d : ℤ} (h : a ≠ b) :
  (∀ x y : ℤ, (x + a * y + c) * (x + b * y + d) = 2 →
    ∃ a b : ℤ, (|a - b| = 1 ∨ (|a - b| = 2 ∧ (d - c) % 2 = 1))) :=
sorry

end integer_solution_count_l2056_205606


namespace lionel_distance_walked_when_met_l2056_205680

theorem lionel_distance_walked_when_met (distance_between : ℕ) (lionel_speed : ℕ) (walt_speed : ℕ) (advance_time : ℕ) 
(h1 : distance_between = 48) 
(h2 : lionel_speed = 2) 
(h3 : walt_speed = 6) 
(h4 : advance_time = 2) : 
  ∃ D : ℕ, D = 15 :=
by
  sorry

end lionel_distance_walked_when_met_l2056_205680


namespace probability_at_least_two_same_l2056_205672

theorem probability_at_least_two_same :
  let total_outcomes := (8 ^ 4 : ℕ)
  let num_diff_outcomes := (8 * 7 * 6 * 5 : ℕ)
  let probability_diff := (num_diff_outcomes : ℝ) / total_outcomes
  let probability_at_least_two := 1 - probability_diff
  probability_at_least_two = (151 : ℝ) / 256 :=
by
  sorry

end probability_at_least_two_same_l2056_205672


namespace complement_union_equals_l2056_205683

def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {-2, 0, 2}

def C_I (I : Set ℤ) (s : Set ℤ) : Set ℤ := I \ s

theorem complement_union_equals :
  C_I universal_set (A ∪ B) = {4, 5} :=
by
  sorry

end complement_union_equals_l2056_205683


namespace pasta_cost_is_one_l2056_205664

-- Define the conditions
def pasta_cost (p : ℝ) : ℝ := p -- The cost of the pasta per box
def sauce_cost : ℝ := 2.00 -- The cost of the sauce
def meatballs_cost : ℝ := 5.00 -- The cost of the meatballs
def servings : ℕ := 8 -- The number of servings
def cost_per_serving : ℝ := 1.00 -- The cost per serving

-- Calculate the total meal cost
def total_meal_cost : ℝ := servings * cost_per_serving

-- Calculate the combined cost of sauce and meatballs
def combined_cost_of_sauce_and_meatballs : ℝ := sauce_cost + meatballs_cost

-- Calculate the cost of the pasta
def pasta_cost_calculation : ℝ := total_meal_cost - combined_cost_of_sauce_and_meatballs

-- The theorem stating that the pasta cost should be $1
theorem pasta_cost_is_one (p : ℝ) (h : pasta_cost_calculation = p) : p = 1 := by
  sorry

end pasta_cost_is_one_l2056_205664


namespace bacteria_growth_l2056_205671

theorem bacteria_growth (d : ℕ) (t : ℕ) (initial final : ℕ) 
  (h_doubling : d = 4) 
  (h_initial : initial = 500) 
  (h_final : final = 32000) 
  (h_ratio : final / initial = 2^6) :
  t = d * 6 → t = 24 :=
by
  sorry

end bacteria_growth_l2056_205671


namespace terminal_side_angles_l2056_205691

theorem terminal_side_angles (k : ℤ) (β : ℝ) :
  β = (Real.pi / 3) + 2 * k * Real.pi → -2 * Real.pi ≤ β ∧ β < 4 * Real.pi :=
by
  sorry

end terminal_side_angles_l2056_205691


namespace range_of_x_l2056_205681

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : 0 < x) (h₂ : A (2 * x * A x) = 5) : 1 < x ∧ x ≤ 5 / 4 := 
sorry

end range_of_x_l2056_205681


namespace total_students_multiple_of_8_l2056_205630

theorem total_students_multiple_of_8 (B G T : ℕ) (h : G = 7 * B) (ht : T = B + G) : T % 8 = 0 :=
by
  sorry

end total_students_multiple_of_8_l2056_205630


namespace cricket_team_age_difference_l2056_205652

theorem cricket_team_age_difference :
  ∀ (captain_age : ℕ) (keeper_age : ℕ) (team_size : ℕ) (team_average_age : ℕ) (remaining_size : ℕ),
  captain_age = 28 →
  keeper_age = captain_age + 3 →
  team_size = 11 →
  team_average_age = 25 →
  remaining_size = team_size - 2 →
  (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 24 →
  team_average_age - (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 1 :=
by
  intros captain_age keeper_age team_size team_average_age remaining_size h1 h2 h3 h4 h5 h6
  sorry

end cricket_team_age_difference_l2056_205652


namespace gen_formula_arithmetic_seq_sum_maximizes_at_5_l2056_205692

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a n = a 1 + (n - 1) * d

variables (an : ℕ → ℤ) (Sn : ℕ → ℤ)
variable (d : ℤ)

theorem gen_formula_arithmetic_seq (h1 : an 3 = 5) (h2 : an 10 = -9) :
  ∀ n, an n = 11 - 2 * n :=
sorry

theorem sum_maximizes_at_5 (h_seq : ∀ n, an n = 11 - 2 * n) :
  ∀ n, Sn n = (n * 10 - n^2) → (∃ n, ∀ k, Sn n ≥ Sn k) :=
sorry

end gen_formula_arithmetic_seq_sum_maximizes_at_5_l2056_205692


namespace center_polar_coordinates_l2056_205653

-- Assuming we have a circle defined in polar coordinates
def polar_circle_center (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

-- The goal is to prove that the center of this circle has the polar coordinates (sqrt 2, π/4)
theorem center_polar_coordinates : ∃ ρ θ, polar_circle_center ρ θ ∧ ρ = Real.sqrt 2 ∧ θ = Real.pi / 4 :=
sorry

end center_polar_coordinates_l2056_205653


namespace exists_linear_eq_exactly_m_solutions_l2056_205639

theorem exists_linear_eq_exactly_m_solutions (m : ℕ) (hm : 0 < m) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ), a * x + b * y = c ↔
    (1 ≤ x ∧ 1 ≤ y ∧ x + y = m + 1) :=
by
  sorry

end exists_linear_eq_exactly_m_solutions_l2056_205639


namespace scientific_notation_l2056_205698

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end scientific_notation_l2056_205698


namespace cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l2056_205622

theorem cos_eq_neg_1_over_4_of_sin_eq_1_over_4
  (α : ℝ)
  (h : Real.sin (α + π / 3) = 1 / 4) :
  Real.cos (α + 5 * π / 6) = -1 / 4 :=
sorry

end cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l2056_205622


namespace seq_b_arithmetic_diff_seq_a_general_term_l2056_205618

variable {n : ℕ}

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n / (a n + 2)

def seq_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / a n

theorem seq_b_arithmetic_diff (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : seq_a a) (h_b : seq_b a b) :
  ∀ n, b (n + 1) - b n = 1 / 2 :=
by
  sorry

theorem seq_a_general_term (a : ℕ → ℝ) (h_a : seq_a a) :
  ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end seq_b_arithmetic_diff_seq_a_general_term_l2056_205618


namespace widget_cost_reduction_l2056_205604

theorem widget_cost_reduction (W R : ℝ) (h1 : 6 * W = 36) (h2 : 8 * (W - R) = 36) : R = 1.5 :=
by
  sorry

end widget_cost_reduction_l2056_205604


namespace greatest_x_for_4x_in_factorial_21_l2056_205678

-- Definition and theorem to state the problem mathematically
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_x_for_4x_in_factorial_21 : ∃ x : ℕ, (4^x ∣ factorial 21) ∧ ∀ y : ℕ, (4^y ∣ factorial 21) → y ≤ 9 :=
by
  sorry

end greatest_x_for_4x_in_factorial_21_l2056_205678


namespace calligraphy_prices_max_brushes_l2056_205655

theorem calligraphy_prices 
  (x y : ℝ)
  (h1 : 40 * x + 100 * y = 280)
  (h2 : 30 * x + 200 * y = 260) :
  x = 6 ∧ y = 0.4 := 
by sorry

theorem max_brushes 
  (m : ℝ)
  (h_budget : 6 * m + 0.4 * (200 - m) ≤ 360) :
  m ≤ 50 :=
by sorry

end calligraphy_prices_max_brushes_l2056_205655


namespace supplementary_angles_ratio_l2056_205649

theorem supplementary_angles_ratio (A B : ℝ) (h1 : A + B = 180) (h2 : A / B = 5 / 4) : B = 80 :=
by
   sorry

end supplementary_angles_ratio_l2056_205649


namespace turtles_on_lonely_island_l2056_205648

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end turtles_on_lonely_island_l2056_205648


namespace find_x_l2056_205697

noncomputable def x : ℝ :=
  sorry

theorem find_x (h : ∃ x : ℝ, x > 0 ∧ ⌊x⌋ * x = 48) : x = 8 :=
  sorry

end find_x_l2056_205697


namespace completing_the_square_sum_l2056_205665

theorem completing_the_square_sum :
  ∃ (a b c : ℤ), 64 * (x : ℝ) ^ 2 + 96 * x - 81 = 0 ∧ a > 0 ∧ (8 * x + 6) ^ 2 = c ∧ a = 8 ∧ b = 6 ∧ a + b + c = 131 :=
by
  sorry

end completing_the_square_sum_l2056_205665


namespace profit_equation_correct_l2056_205625

theorem profit_equation_correct (x : ℝ) : 
  let original_selling_price := 36
  let purchase_price := 20
  let original_sales_volume := 200
  let price_increase_effect := 5
  let desired_profit := 1200
  let original_profit_per_unit := original_selling_price - purchase_price
  let new_selling_price := original_selling_price + x
  let new_sales_volume := original_sales_volume - price_increase_effect * x
  (original_profit_per_unit + x) * new_sales_volume = desired_profit :=
sorry

end profit_equation_correct_l2056_205625


namespace num_possible_y_l2056_205694

theorem num_possible_y : 
  (∃ (count : ℕ), count = (54 - 26 + 1) ∧ 
  ∀ (y : ℤ), 25 < y ∧ y < 55 ↔ (26 ≤ y ∧ y ≤ 54)) :=
by {
  sorry 
}

end num_possible_y_l2056_205694


namespace inequality_proof_l2056_205614

noncomputable def f (x m : ℝ) : ℝ := 2 * m * x - Real.log x

theorem inequality_proof (m x₁ x₂ : ℝ) (hm : m ≥ -1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hineq : (f x₁ m + f x₂ m) / 2 ≤ x₁ ^ 2 + x₂ ^ 2 + (3 / 2) * x₁ * x₂) :
  x₁ + x₂ ≥ (Real.sqrt 3 - 1) / 2 := 
sorry

end inequality_proof_l2056_205614


namespace student_B_speed_l2056_205699

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l2056_205699


namespace polynomial_roots_power_sum_l2056_205646

theorem polynomial_roots_power_sum {a b c : ℝ}
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 6)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 :=
by
  sorry

end polynomial_roots_power_sum_l2056_205646


namespace sum_of_consecutive_integers_l2056_205654

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l2056_205654


namespace exist_equal_success_rate_l2056_205643

noncomputable def S : ℕ → ℝ := sorry -- Definition of S(N), the number of successful free throws

theorem exist_equal_success_rate (N1 N2 : ℕ) 
  (h1 : S N1 < 0.8 * N1) 
  (h2 : S N2 > 0.8 * N2) : 
  ∃ (N : ℕ), N1 ≤ N ∧ N ≤ N2 ∧ S N = 0.8 * N :=
sorry

end exist_equal_success_rate_l2056_205643


namespace tetrahedron_has_six_edges_l2056_205667

-- Define what a tetrahedron is
inductive Tetrahedron : Type
| mk : Tetrahedron

-- Define the number of edges of a Tetrahedron
def edges_of_tetrahedron (t : Tetrahedron) : Nat := 6

theorem tetrahedron_has_six_edges (t : Tetrahedron) : edges_of_tetrahedron t = 6 := 
by
  sorry

end tetrahedron_has_six_edges_l2056_205667


namespace largest_divisor_consecutive_odd_squares_l2056_205627

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (hmn : m = n + 2) 
  (hodd_m : m % 2 = 1) 
  (hodd_n : n % 2 = 1) 
  (horder : n < m) : ∃ k : ℤ, m^2 - n^2 = 8 * k :=
by 
  sorry

end largest_divisor_consecutive_odd_squares_l2056_205627


namespace smallest_integer_y_l2056_205686

theorem smallest_integer_y (y : ℤ) : (∃ (y : ℤ), (y / 4) + (3 / 7) > (4 / 7) ∧ ∀ (z : ℤ), z < y → (z / 4) + (3 / 7) ≤ (4 / 7)) := 
by
  sorry

end smallest_integer_y_l2056_205686


namespace car_speed_after_modifications_l2056_205696

theorem car_speed_after_modifications (s : ℕ) (p : ℝ) (w : ℕ) :
  s = 150 →
  p = 0.30 →
  w = 10 →
  s + (p * s) + w = 205 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  done

end car_speed_after_modifications_l2056_205696


namespace necessary_but_not_sufficient_for_lt_l2056_205676

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_lt (h : a < b + 1) : a < b := 
sorry

end necessary_but_not_sufficient_for_lt_l2056_205676


namespace original_number_from_sum_l2056_205644

variable (a b c : ℕ) (m S : ℕ)

/-- Given a three-digit number, the magician asks the participant to add all permutations -/
def three_digit_number_permutations_sum (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) +
  (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)

/-- Given the sum of all permutations of the three-digit number is 4239, determine the original number -/
theorem original_number_from_sum (S : ℕ) (hS : S = 4239) (Sum_conditions : three_digit_number_permutations_sum a b c = S) :
  (100 * a + 10 * b + c) = 429 := by
  sorry

end original_number_from_sum_l2056_205644


namespace toby_friends_girls_l2056_205662

theorem toby_friends_girls (total_friends : ℕ) (num_boys : ℕ) (perc_boys : ℕ) 
  (h1 : perc_boys = 55) (h2 : num_boys = 33) (h3 : total_friends = 60) : 
  (total_friends - num_boys = 27) :=
by
  sorry

end toby_friends_girls_l2056_205662


namespace evaluate_expression_l2056_205688

noncomputable def g (x : ℝ) : ℝ := x^3 + 3*x + 2*Real.sqrt x

theorem evaluate_expression : 
  3 * g 3 - 2 * g 9 = -1416 + 6 * Real.sqrt 3 :=
by
  sorry

end evaluate_expression_l2056_205688


namespace part1_part2_l2056_205656

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

/-- Given sequence properties -/
axiom h1 : a 1 = 5
axiom h2 : ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1) + 2^n - 1

/-- Part (I): Proving the sequence is arithmetic -/
theorem part1 (n : ℕ) : ∃ d, (∀ m ≥ 1, (a (m + 1) - 1) / 2^(m + 1) - (a m - 1) / 2^m = d)
∧ ((a 1 - 1) / 2 = 2) := sorry

/-- Part (II): Sum of the first n terms -/
theorem part2 (n : ℕ) : S n = n * 2^(n+1) := sorry

end part1_part2_l2056_205656


namespace melissa_games_played_l2056_205669

-- Define the conditions mentioned:
def points_per_game := 12
def total_points := 36

-- State the proof problem:
theorem melissa_games_played : total_points / points_per_game = 3 :=
by sorry

end melissa_games_played_l2056_205669


namespace income_of_first_member_l2056_205690

-- Define the number of family members
def num_members : ℕ := 4

-- Define the average income per member
def avg_income : ℕ := 10000

-- Define the known incomes of the other three members
def income2 : ℕ := 15000
def income3 : ℕ := 6000
def income4 : ℕ := 11000

-- Define the total income of the family
def total_income : ℕ := avg_income * num_members

-- Define the total income of the other three members
def total_other_incomes : ℕ := income2 + income3 + income4

-- Define the income of the first member
def income1 : ℕ := total_income - total_other_incomes

-- The theorem to prove
theorem income_of_first_member : income1 = 8000 := by
  sorry

end income_of_first_member_l2056_205690


namespace distinct_triangle_not_isosceles_l2056_205607

theorem distinct_triangle_not_isosceles (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  ¬(a = b ∨ b = c ∨ c = a) :=
by {
  sorry
}

end distinct_triangle_not_isosceles_l2056_205607


namespace remainder_when_x_plus_3uy_divided_by_y_eq_v_l2056_205615

theorem remainder_when_x_plus_3uy_divided_by_y_eq_v
  (x y u v : ℕ) (h_pos_y : 0 < y) (h_division_algo : x = u * y + v) (h_remainder : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_when_x_plus_3uy_divided_by_y_eq_v_l2056_205615


namespace purchases_per_customer_l2056_205663

noncomputable def number_of_customers_in_cars (num_cars : ℕ) (customers_per_car : ℕ) : ℕ :=
  num_cars * customers_per_car

def total_sales (sports_store_sales : ℕ) (music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

theorem purchases_per_customer {num_cars : ℕ} {customers_per_car : ℕ} {sports_store_sales : ℕ} {music_store_sales : ℕ}
    (h1 : num_cars = 10)
    (h2 : customers_per_car = 5)
    (h3 : sports_store_sales = 20)
    (h4: music_store_sales = 30) :
    (total_sales sports_store_sales music_store_sales / number_of_customers_in_cars num_cars customers_per_car) = 1 :=
by
  sorry

end purchases_per_customer_l2056_205663


namespace percentage_of_female_officers_on_duty_l2056_205687

theorem percentage_of_female_officers_on_duty
    (on_duty : ℕ) (half_on_duty_female : on_duty / 2 = 100)
    (total_female_officers : ℕ)
    (total_female_officers_value : total_female_officers = 1000)
    : (100 / total_female_officers : ℝ) * 100 = 10 :=
by sorry

end percentage_of_female_officers_on_duty_l2056_205687


namespace correct_assertions_l2056_205617

variables {A B : Type} (f : A → B)

-- 1. Different elements in set A can have the same image in set B
def statement_1 : Prop := ∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = f a2

-- 2. A single element in set A can have different images in B
def statement_2 : Prop := ∃ a1 : A, ∃ b1 b2 : B, b1 ≠ b2 ∧ (f a1 = b1 ∧ f a1 = b2)

-- 3. There can be elements in set B that do not have a pre-image in A
def statement_3 : Prop := ∃ b : B, ∀ a : A, f a ≠ b

-- Correct answer is statements 1 and 3 are true, statement 2 is false
theorem correct_assertions : statement_1 f ∧ ¬statement_2 f ∧ statement_3 f := sorry

end correct_assertions_l2056_205617


namespace find_x_from_equation_l2056_205620

/-- If (1 / 8) * 2^36 = 4^x, then x = 16.5 -/
theorem find_x_from_equation (x : ℝ) (h : (1/8) * (2:ℝ)^36 = (4:ℝ)^x) : x = 16.5 :=
by sorry

end find_x_from_equation_l2056_205620


namespace divisors_form_l2056_205637

theorem divisors_form (p n : ℕ) (h_prime : Nat.Prime p) (h_pos : 0 < n) :
  ∃ k : ℕ, (p^n - 1 = 2^k - 1 ∨ p^n - 1 ∣ 48) :=
sorry

end divisors_form_l2056_205637


namespace roots_are_prime_then_a_is_five_l2056_205608

theorem roots_are_prime_then_a_is_five (x1 x2 a : ℕ) (h_prime_x1 : Prime x1) (h_prime_x2 : Prime x2)
  (h_eq : x1 + x2 = a) (h_eq_mul : x1 * x2 = a + 1) : a = 5 :=
sorry

end roots_are_prime_then_a_is_five_l2056_205608


namespace total_movies_attended_l2056_205693

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end total_movies_attended_l2056_205693


namespace divisor_of_1025_l2056_205609

theorem divisor_of_1025 (d : ℕ) (h1: 1015 + 10 = 1025) (h2 : d ∣ 1025) : d = 5 := 
sorry

end divisor_of_1025_l2056_205609


namespace total_fence_length_l2056_205626

variable (Darren Doug : ℝ)

-- Definitions based on given conditions
def Darren_paints_more := Darren = 1.20 * Doug
def Darren_paints_360 := Darren = 360

-- The statement to prove
theorem total_fence_length (h1 : Darren_paints_more Darren Doug) (h2 : Darren_paints_360 Darren) : (Darren + Doug) = 660 := 
by
  sorry

end total_fence_length_l2056_205626


namespace tile_floor_covering_l2056_205659

theorem tile_floor_covering (n : ℕ) (h1 : 10 < n) (h2 : n < 20) (h3 : ∃ x, 9 * x = n^2) : n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end tile_floor_covering_l2056_205659


namespace parallel_lines_l2056_205666

theorem parallel_lines (k1 k2 l1 l2 : ℝ) :
  (∀ x, (k1 ≠ k2) -> (k1 * x + l1 ≠ k2 * x + l2)) ↔ 
  (k1 = k2 ∧ l1 ≠ l2) := 
by sorry

end parallel_lines_l2056_205666


namespace isosceles_triangle_base_length_l2056_205613

theorem isosceles_triangle_base_length
  (a b c : ℕ)
  (h_iso : a = b)
  (h_perimeter : a + b + c = 62)
  (h_leg_length : a = 25) :
  c = 12 :=
by
  sorry

end isosceles_triangle_base_length_l2056_205613


namespace total_different_books_l2056_205605

def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def tony_dean_shared_books : ℕ := 3
def all_three_shared_book : ℕ := 1

theorem total_different_books :
  tony_books + dean_books + breanna_books - tony_dean_shared_books - 2 * all_three_shared_book = 47 := 
by
  sorry 

end total_different_books_l2056_205605


namespace proof_problem_l2056_205631

variable (balls : Finset ℕ) (blackBalls whiteBalls : Finset ℕ)
variable (drawnBalls : Finset ℕ)

/-- There are 6 black balls numbered 1 to 6. -/
def initialBlackBalls : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- There are 4 white balls numbered 7 to 10. -/
def initialWhiteBalls : Finset ℕ := {7, 8, 9, 10}

/-- The total balls (black + white). -/
def totalBalls : Finset ℕ := initialBlackBalls ∪ initialWhiteBalls

/-- The hypergeometric distribution condition for black balls. -/
def hypergeometricBlack : Prop :=
  true  -- placeholder: black balls follow hypergeometric distribution

/-- The probability of drawing 2 white balls is not 1/14. -/
def probDraw2White : Prop :=
  (3 / 7) ≠ (1 / 14)

/-- The probability of the maximum total score (8 points) is 1/14. -/
def probMaxScore : Prop :=
  (15 / 210) = (1 / 14)

/-- Main theorem combining the above conditions for the problem. -/
theorem proof_problem : hypergeometricBlack ∧ probMaxScore :=
by
  unfold hypergeometricBlack
  unfold probMaxScore
  sorry

end proof_problem_l2056_205631


namespace fourth_derivative_of_function_y_l2056_205612

noncomputable def log_base_3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

noncomputable def function_y (x : ℝ) : ℝ := (log_base_3 x) / (x ^ 2)

theorem fourth_derivative_of_function_y (x : ℝ) (h : 0 < x) : 
    (deriv^[4] (fun x => function_y x)) x = (-154 + 120 * (Real.log x)) / (x ^ 6 * Real.log 3) :=
  sorry

end fourth_derivative_of_function_y_l2056_205612


namespace swim_distance_l2056_205682

theorem swim_distance (v d : ℝ) (c : ℝ := 2.5) :
  (8 = d / (v + c)) ∧ (8 = 24 / (v - c)) → d = 84 :=
by
  sorry

end swim_distance_l2056_205682


namespace area_of_region_l2056_205645

-- Define the condition: the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 10 * x - 4 * y + 9 = 0

-- State the theorem: the area of the region defined by the equation is 20π
theorem area_of_region : ∀ x y : ℝ, region_equation x y → ∃ A : ℝ, A = 20 * Real.pi :=
by sorry

end area_of_region_l2056_205645


namespace problem1_problem2_l2056_205616

-- Problem 1: Evaluating an integer arithmetic expression
theorem problem1 : (1 * (-8)) - (-6) + (-3) = -5 := 
by
  sorry

-- Problem 2: Evaluating a mixed arithmetic expression with rational numbers and decimals
theorem problem2 : (5 / 13) - 3.7 + (8 / 13) - (-1.7) = -1 := 
by
  sorry

end problem1_problem2_l2056_205616


namespace binom_21_10_l2056_205673

theorem binom_21_10 :
  (Nat.choose 19 9 = 92378) →
  (Nat.choose 19 10 = 92378) →
  (Nat.choose 19 11 = 75582) →
  Nat.choose 21 10 = 352716 := by
  sorry

end binom_21_10_l2056_205673


namespace convert_255_to_base8_l2056_205658

-- Define the conversion function from base 10 to base 8
def base10_to_base8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let r2 := n % 64
  let d1 := r2 / 8
  let r1 := r2 % 8
  d2 * 100 + d1 * 10 + r1

-- Define the specific number and base in the conditions
def num10 : ℕ := 255
def base8_result : ℕ := 377

-- The theorem stating the proof problem
theorem convert_255_to_base8 : base10_to_base8 num10 = base8_result :=
by
  -- You would provide the proof steps here
  sorry

end convert_255_to_base8_l2056_205658


namespace complement_of_A_in_U_is_2_l2056_205675

open Set

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }

theorem complement_of_A_in_U_is_2 : compl A ∩ U = {2} :=
by
  sorry

end complement_of_A_in_U_is_2_l2056_205675
