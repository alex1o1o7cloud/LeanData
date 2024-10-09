import Mathlib

namespace pipe_A_fill_time_l2109_210929

theorem pipe_A_fill_time (x : ℝ) (h1 : ∀ t : ℝ, t = 45) (h2 : ∀ t : ℝ, t = 18) :
  (1/x + 1/45 = 1/18) → x = 30 :=
by {
  -- Proof is omitted
  sorry
}

end pipe_A_fill_time_l2109_210929


namespace cowboy_shortest_distance_l2109_210934

noncomputable def distance : ℝ :=
  let C := (0, 5)
  let B := (-10, 11)
  let C' := (0, -5)
  5 + Real.sqrt ((C'.1 - B.1)^2 + (C'.2 - B.2)^2)

theorem cowboy_shortest_distance :
  distance = 5 + Real.sqrt 356 :=
by
  sorry

end cowboy_shortest_distance_l2109_210934


namespace find_integer_l2109_210910

theorem find_integer
  (x y : ℤ)
  (h1 : 4 * x + y = 34)
  (h2 : 2 * x - y = 20)
  (h3 : y^2 = 4) :
  y = -2 :=
by
  sorry

end find_integer_l2109_210910


namespace greatest_possible_sum_of_visible_numbers_l2109_210984

theorem greatest_possible_sum_of_visible_numbers :
  ∀ (numbers : ℕ → ℕ) (Cubes : Fin 4 → ℤ), 
  (numbers 0 = 1) → (numbers 1 = 3) → (numbers 2 = 9) → (numbers 3 = 27) → (numbers 4 = 81) → (numbers 5 = 243) →
  (Cubes 0 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) → 
  (Cubes 1 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) →
  (Cubes 2 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 3 = 16 * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 0 + Cubes 1 + Cubes 2 + Cubes 3 = 1452) :=
by 
  sorry

end greatest_possible_sum_of_visible_numbers_l2109_210984


namespace range_of_m_l2109_210982

theorem range_of_m (m : ℝ) :
  ¬(1^2 + 2*1 - m > 0) ∧ (2^2 + 2*2 - m > 0) ↔ (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l2109_210982


namespace Tim_total_payment_l2109_210948

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l2109_210948


namespace expand_and_solve_solve_quadratic_l2109_210921

theorem expand_and_solve (x : ℝ) :
  6 * (x - 3) * (x + 5) = 6 * x^2 + 12 * x - 90 :=
by sorry

theorem solve_quadratic (x : ℝ) :
  6 * x^2 + 12 * x - 90 = 0 ↔ x = -5 ∨ x = 3 :=
by sorry

end expand_and_solve_solve_quadratic_l2109_210921


namespace min_value_of_expression_l2109_210971

theorem min_value_of_expression (x y : ℝ) : 
  ∃ m : ℝ, m = (xy - 1)^2 + (x + y)^2 ∧ (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ m) := 
sorry

end min_value_of_expression_l2109_210971


namespace grid_segments_divisible_by_4_l2109_210985

-- Definition: square grid where each cell has a side length of 1
structure SquareGrid (n : ℕ) :=
  (segments : ℕ)

-- Condition: Function to calculate the total length of segments in the grid
def total_length {n : ℕ} (Q : SquareGrid n) : ℕ := Q.segments

-- Lean 4 statement: Prove that for any grid, the total length is divisible by 4
theorem grid_segments_divisible_by_4 {n : ℕ} (Q : SquareGrid n) :
  total_length Q % 4 = 0 :=
sorry

end grid_segments_divisible_by_4_l2109_210985


namespace shaded_area_l2109_210940

theorem shaded_area (side_len : ℕ) (triangle_base : ℕ) (triangle_height : ℕ)
  (h1 : side_len = 40) (h2 : triangle_base = side_len / 2)
  (h3 : triangle_height = side_len / 2) : 
  side_len^2 - 2 * (1/2 * triangle_base * triangle_height) = 1200 := 
  sorry

end shaded_area_l2109_210940


namespace inequality_solution_l2109_210972

noncomputable def solve_inequality (m : ℝ) (m_lt_neg2 : m < -2) : Set ℝ :=
  if h : m = -3 then {x | 1 < x}
  else if h' : -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
  else {x | 1 < x ∧ x < m / (m + 3)}

theorem inequality_solution (m : ℝ) (m_lt_neg2 : m < -2) :
  (solve_inequality m m_lt_neg2) = 
    if m = -3 then {x | 1 < x}
    else if -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
    else {x | 1 < x ∧ x < m / (m + 3)} :=
sorry

end inequality_solution_l2109_210972


namespace tub_volume_ratio_l2109_210973

theorem tub_volume_ratio (C D : ℝ) 
  (h₁ : 0 < C) 
  (h₂ : 0 < D)
  (h₃ : (3/4) * C = (2/3) * D) : 
  C / D = 8 / 9 := 
sorry

end tub_volume_ratio_l2109_210973


namespace arithmetic_sequence_a6_eq_1_l2109_210935

theorem arithmetic_sequence_a6_eq_1
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : S 11 = 11)
  (h2 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h3 : ∃ d, ∀ n, a n = a 1 + (n - 1) * d) :
  a 6 = 1 :=
by
  sorry

end arithmetic_sequence_a6_eq_1_l2109_210935


namespace expression_for_f_minimum_positive_period_of_f_range_of_f_l2109_210950

noncomputable def f (x : ℝ) : ℝ :=
  let A := (2, 0) 
  let B := (0, 2)
  let C := (Real.cos (2 * x), Real.sin (2 * x))
  let AB := (B.1 - A.1, B.2 - A.2) 
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.fst * AC.fst + AB.snd * AC.snd 

theorem expression_for_f (x : ℝ) :
  f x = 2 * Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 4 :=
by sorry

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by sorry

theorem range_of_f (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) :
  2 < f x ∧ f x ≤ 4 + 2 * Real.sqrt 2 :=
by sorry

end expression_for_f_minimum_positive_period_of_f_range_of_f_l2109_210950


namespace sum_of_legs_of_larger_triangle_l2109_210967

theorem sum_of_legs_of_larger_triangle 
  (area_small area_large : ℝ)
  (hypotenuse_small : ℝ)
  (A : area_small = 10)
  (B : area_large = 250)
  (C : hypotenuse_small = 13) : 
  ∃ a b : ℝ, (a + b = 35) := 
sorry

end sum_of_legs_of_larger_triangle_l2109_210967


namespace print_shop_x_charges_l2109_210908

theorem print_shop_x_charges (x : ℝ) (h1 : ∀ y : ℝ, y = 1.70) (h2 : 40 * x + 20 = 40 * 1.70) : x = 1.20 :=
by
  sorry

end print_shop_x_charges_l2109_210908


namespace range_of_m_l2109_210946

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, f y = x) :
  (∀ x, ∃ y, y = f (x + m) - f (x - m)) →
  -1 ≤ m ∧ m ≤ 1 :=
by
  intro hF
  sorry

end range_of_m_l2109_210946


namespace num_integer_solutions_eq_3_l2109_210987

theorem num_integer_solutions_eq_3 :
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((2 * x^2) + (x * y) + (y^2) - x + 2 * y + 1 = 0 ↔ (x, y) ∈ S)) ∧ 
  S.card = 3 :=
sorry

end num_integer_solutions_eq_3_l2109_210987


namespace exists_k_in_octahedron_l2109_210981

theorem exists_k_in_octahedron
  (x0 y0 z0 : ℚ)
  (h : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ 
                 x0 + y0 - z0 ≠ n ∧ 
                 x0 - y0 + z0 ≠ n ∧ 
                 x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, ∃ (xk yk zk : ℚ), 
    k ≠ 0 ∧ 
    xk = k * x0 ∧ 
    yk = k * y0 ∧ 
    zk = k * z0 ∧
    ∀ n : ℤ, 
      (xk + yk + zk < ↑n → xk + yk + zk > ↑(n - 1)) ∧ 
      (xk + yk - zk < ↑n → xk + yk - zk > ↑(n - 1)) ∧ 
      (xk - yk + zk < ↑n → xk - yk + zk > ↑(n - 1)) ∧ 
      (xk - yk - zk < ↑n → xk - yk - zk > ↑(n - 1)) :=
sorry

end exists_k_in_octahedron_l2109_210981


namespace count_original_scissors_l2109_210924

def originalScissors (addedScissors totalScissors : ℕ) : ℕ := totalScissors - addedScissors

theorem count_original_scissors :
  ∃ (originalScissorsCount : ℕ), originalScissorsCount = originalScissors 13 52 := 
  sorry

end count_original_scissors_l2109_210924


namespace required_force_18_inch_wrench_l2109_210944

def inverse_force (l : ℕ) (k : ℕ) : ℕ := k / l

def extra_force : ℕ := 50

def initial_force : ℕ := 300

noncomputable
def handle_length_1 : ℕ := 12

noncomputable
def handle_length_2 : ℕ := 18

noncomputable
def adjusted_force : ℕ := inverse_force handle_length_2 (initial_force * handle_length_1)

theorem required_force_18_inch_wrench : 
  adjusted_force + extra_force = 250 := 
by
  sorry

end required_force_18_inch_wrench_l2109_210944


namespace students_neither_math_physics_l2109_210992

theorem students_neither_math_physics (total_students math_students physics_students both_students : ℕ) 
  (h1 : total_students = 120)
  (h2 : math_students = 80)
  (h3 : physics_students = 50)
  (h4 : both_students = 15) : 
  total_students - (math_students - both_students + physics_students - both_students + both_students) = 5 :=
by
  -- Each of the hypotheses are used exactly as given in the conditions.
  -- We omit the proof as requested.
  sorry

end students_neither_math_physics_l2109_210992


namespace intersection_A_B_l2109_210903

noncomputable def A : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1) ∧ y ≥ 0}

theorem intersection_A_B : A ∩ {x | ∃ y, y = Real.log (x^2 + 1) ∧ y ≥ 0} = {x | 0 < x ∧ x < 2} :=
  sorry

end intersection_A_B_l2109_210903


namespace major_axis_length_l2109_210958

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship given in the problem
def major_axis_ratio : ℝ := 1.6

-- Define the calculation for minor axis
def minor_axis : ℝ := 2 * cylinder_radius

-- Define the calculation for major axis
def major_axis : ℝ := major_axis_ratio * minor_axis

-- The theorem statement
theorem major_axis_length:
  major_axis = 6.4 :=
by 
  sorry -- Proof to be provided later

end major_axis_length_l2109_210958


namespace diff_of_squares_odd_divisible_by_8_l2109_210926

theorem diff_of_squares_odd_divisible_by_8 (m n : ℤ) :
  ((2 * m + 1) ^ 2 - (2 * n + 1) ^ 2) % 8 = 0 :=
by 
  sorry

end diff_of_squares_odd_divisible_by_8_l2109_210926


namespace swimmers_pass_each_other_l2109_210970

/-- Two swimmers in a 100-foot pool, one swimming at 4 feet per second, the other at 3 feet per second,
    continuously for 12 minutes, pass each other exactly 32 times. -/
theorem swimmers_pass_each_other 
  (pool_length : ℕ) 
  (time : ℕ) 
  (rate1 : ℕ)
  (rate2 : ℕ)
  (meet_times : ℕ)
  (hp : pool_length = 100) 
  (ht : time = 720) -- 12 minutes = 720 seconds
  (hr1 : rate1 = 4) 
  (hr2 : rate2 = 3)
  : meet_times = 32 := 
sorry

end swimmers_pass_each_other_l2109_210970


namespace automobile_travel_distance_5_minutes_l2109_210994

variable (a r : ℝ)

theorem automobile_travel_distance_5_minutes (h0 : r ≠ 0) :
  let distance_in_feet := (2 * a) / 5
  let time_in_seconds := 300
  (distance_in_feet / r) * time_in_seconds / 3 = 40 * a / r :=
by
  sorry

end automobile_travel_distance_5_minutes_l2109_210994


namespace larger_pie_crust_flour_l2109_210966

theorem larger_pie_crust_flour
  (p1 p2 : ℕ)
  (f1 f2 c : ℚ)
  (h1 : p1 = 36)
  (h2 : p2 = 24)
  (h3 : f1 = 1 / 8)
  (h4 : p1 * f1 = c)
  (h5 : p2 * f2 = c)
  : f2 = 3 / 16 :=
sorry

end larger_pie_crust_flour_l2109_210966


namespace expected_value_is_one_third_l2109_210955

noncomputable def expected_value_of_winnings : ℚ :=
  let p1 := (1/6 : ℚ)
  let p2 := (1/6 : ℚ)
  let p3 := (1/6 : ℚ)
  let p4 := (1/6 : ℚ)
  let p5 := (1/6 : ℚ)
  let p6 := (1/6 : ℚ)
  let winnings1 := (5 : ℚ)
  let winnings2 := (5 : ℚ)
  let winnings3 := (0 : ℚ)
  let winnings4 := (0 : ℚ)
  let winnings5 := (-4 : ℚ)
  let winnings6 := (-4 : ℚ)
  (p1 * winnings1 + p2 * winnings2 + p3 * winnings3 + p4 * winnings4 + p5 * winnings5 + p6 * winnings6)

theorem expected_value_is_one_third : expected_value_of_winnings = 1 / 3 := by
  sorry

end expected_value_is_one_third_l2109_210955


namespace double_series_evaluation_l2109_210915

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) :=
sorry

end double_series_evaluation_l2109_210915


namespace annual_interest_rate_is_correct_l2109_210927

-- Definitions of the conditions
def true_discount : ℚ := 210
def bill_amount : ℚ := 1960
def time_period_years : ℚ := 3 / 4

-- The present value of the bill
def present_value : ℚ := bill_amount - true_discount

-- The formula for simple interest given principal, rate, and time
def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T / 100

-- Proof statement
theorem annual_interest_rate_is_correct : 
  ∃ (R : ℚ), simple_interest present_value R time_period_years = true_discount ∧ R = 16 :=
by
  use 16
  sorry

end annual_interest_rate_is_correct_l2109_210927


namespace complement_intersection_l2109_210901

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 3}

theorem complement_intersection : (U \ N) ∩ M = {4, 5} :=
by 
  sorry

end complement_intersection_l2109_210901


namespace heloise_total_pets_l2109_210963

-- Define initial data
def ratio_dogs_to_cats := (10, 17)
def dogs_given_away := 10
def dogs_remaining := 60

-- Definition of initial number of dogs based on conditions
def initial_dogs := dogs_remaining + dogs_given_away

-- Definition based on ratio of dogs to cats
def dogs_per_set := ratio_dogs_to_cats.1
def cats_per_set := ratio_dogs_to_cats.2

-- Compute the number of sets of dogs
def sets_of_dogs := initial_dogs / dogs_per_set

-- Compute the number of cats
def initial_cats := sets_of_dogs * cats_per_set

-- Definition of the total number of pets
def total_pets := dogs_remaining + initial_cats

-- Lean statement for the proof
theorem heloise_total_pets :
  initial_dogs = 70 ∧
  sets_of_dogs = 7 ∧
  initial_cats = 119 ∧
  total_pets = 179 :=
by
  -- The statements to be proved are listed as conjunctions (∧)
  sorry

end heloise_total_pets_l2109_210963


namespace gcf_of_lcm_9_15_and_10_21_is_5_l2109_210938

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l2109_210938


namespace largest_lcm_value_is_60_l2109_210911

-- Define the conditions
def lcm_values : List ℕ := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 9, Nat.lcm 15 12, Nat.lcm 15 10, Nat.lcm 15 15]

-- State the proof problem
theorem largest_lcm_value_is_60 : lcm_values.maximum = some 60 :=
by
  repeat { sorry }

end largest_lcm_value_is_60_l2109_210911


namespace maximum_ab_expression_l2109_210988

open Function Real

theorem maximum_ab_expression {a b : ℝ} (h : 0 < a ∧ 0 < b ∧ 5 * a + 6 * b < 110) :
  ab * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 :=
sorry

end maximum_ab_expression_l2109_210988


namespace simplify_and_evaluate_l2109_210945

noncomputable def simplified_expr (x y : ℝ) : ℝ :=
  ((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)

theorem simplify_and_evaluate :
  let x := -1
  let y := 2
  simplified_expr x y = 1 :=
by
  -- Proof will go here
  sorry

end simplify_and_evaluate_l2109_210945


namespace find_q_l2109_210977

theorem find_q (a b m p q : ℚ) 
  (h1 : ∀ x, x^2 - m * x + 3 = (x - a) * (x - b)) 
  (h2 : a * b = 3) 
  (h3 : (x^2 - p * x + q) = (x - (a + 1/b)) * (x - (b + 1/a))) : 
  q = 16 / 3 := 
by sorry

end find_q_l2109_210977


namespace pure_milk_in_final_solution_l2109_210951

noncomputable def final_quantity_of_milk (initial_milk : ℕ) (milk_removed_each_step : ℕ) (steps : ℕ) : ℝ :=
  let remaining_milk_step1 := initial_milk - milk_removed_each_step
  let proportion := (milk_removed_each_step : ℝ) / (initial_milk : ℝ)
  let milk_removed_step2 := proportion * remaining_milk_step1
  remaining_milk_step1 - milk_removed_step2

theorem pure_milk_in_final_solution :
  final_quantity_of_milk 30 9 2 = 14.7 :=
by
  sorry

end pure_milk_in_final_solution_l2109_210951


namespace kite_area_correct_l2109_210996

-- Define the coordinates of the vertices
def vertex1 : (ℤ × ℤ) := (3, 0)
def vertex2 : (ℤ × ℤ) := (0, 5)
def vertex3 : (ℤ × ℤ) := (3, 7)
def vertex4 : (ℤ × ℤ) := (6, 5)

-- Define the area of a kite using the Shoelace formula for a quadrilateral
-- with given vertices
def kite_area (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))) / 2

theorem kite_area_correct : kite_area vertex1 vertex2 vertex3 vertex4 = 21 := 
  sorry

end kite_area_correct_l2109_210996


namespace sum_of_solutions_l2109_210919

theorem sum_of_solutions (x1 x2 : ℝ) (h : ∀ (x : ℝ), x^2 - 10 * x + 14 = 0 → x = x1 ∨ x = x2) :
  x1 + x2 = 10 :=
sorry

end sum_of_solutions_l2109_210919


namespace arithmetic_seq_proof_l2109_210943

theorem arithmetic_seq_proof
  (x : ℕ → ℝ)
  (h : ∀ n ≥ 3, x (n-1) = (x n + x (n-1) + x (n-2)) / 3):
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry

end arithmetic_seq_proof_l2109_210943


namespace solution_set_of_inequality_l2109_210975

theorem solution_set_of_inequality (x : ℝ) : 
  (1 / x ≤ 1 ↔ (0 < x ∧ x < 1) ∨ (1 ≤ x)) :=
  sorry

end solution_set_of_inequality_l2109_210975


namespace hcf_36_84_l2109_210969

def highestCommonFactor (a b : ℕ) : ℕ := Nat.gcd a b

theorem hcf_36_84 : highestCommonFactor 36 84 = 12 := by
  sorry

end hcf_36_84_l2109_210969


namespace complete_square_l2109_210904

-- Definitions based on conditions
def row_sum_piece2 := 2 + 1 + 3 + 1
def total_sum_square := 4 * row_sum_piece2
def sum_piece1 := 7
def sum_piece2 := 8
def sum_piece3 := 8
def total_given_pieces := sum_piece1 + sum_piece2 + sum_piece3
def sum_missing_piece := total_sum_square - total_given_pieces

-- Statement to prove that the missing piece has the correct sum
theorem complete_square : (sum_missing_piece = 5) :=
by 
  -- It is a placeholder for the proof steps, the actual proof steps are not needed
  sorry

end complete_square_l2109_210904


namespace factorize_expression_l2109_210983

variable {a b : ℕ}

theorem factorize_expression (h : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1)) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) :=
by sorry

end factorize_expression_l2109_210983


namespace inequality_solution_l2109_210995

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := 
by 
  sorry

end inequality_solution_l2109_210995


namespace rational_equation_solutions_l2109_210953

open Real

theorem rational_equation_solutions :
  (∃ x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ ((x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0)) →
  ∃ S : Finset ℝ, S.card = 2 ∧ ∀ x ∈ S, (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end rational_equation_solutions_l2109_210953


namespace division_proof_l2109_210900

theorem division_proof :
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1 / 2) = 3.5 :=
by
  -- definitions based on conditions
  let numerator1 := 2 * 4 * 6
  let denominator1 := 1 + 3 + 5 + 7
  let numerator2 := 1 * 3 * 5
  let denominator2 := 2 + 4 + 6
  -- the statement of the theorem
  sorry

end division_proof_l2109_210900


namespace find_r_cubed_l2109_210978

theorem find_r_cubed (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 :=
by
  sorry

end find_r_cubed_l2109_210978


namespace odd_indexed_terms_geometric_sequence_l2109_210902

open Nat

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (2 * n + 3) = r * a (2 * n + 1)

theorem odd_indexed_terms_geometric_sequence (b : ℕ → ℝ) (h : ∀ n, b n * b (n + 1) = 3 ^ n) :
  is_geometric_sequence b 3 :=
by
  sorry

end odd_indexed_terms_geometric_sequence_l2109_210902


namespace dinner_potatoes_l2109_210930

def lunch_potatoes : ℕ := 5
def total_potatoes : ℕ := 7

theorem dinner_potatoes : total_potatoes - lunch_potatoes = 2 :=
by
  sorry

end dinner_potatoes_l2109_210930


namespace rectangle_semicircle_area_split_l2109_210980

open Real

/-- The main problem statement -/
theorem rectangle_semicircle_area_split 
  (A B D C N U T : ℝ)
  (AU_AN_UAlengths : AU = 84 ∧ AN = 126 ∧ UB = 168)
  (area_ratio : ∃ (ℓ : ℝ), ∃ (N U T : ℝ), 1 / 2 = area_differ / (area_left + area_right))
  (DA_calculation : DA = 63 * sqrt 6) :
  63 + 6 = 69
:=
sorry

end rectangle_semicircle_area_split_l2109_210980


namespace exists_increasing_sequence_l2109_210912

theorem exists_increasing_sequence (n : ℕ) : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i → i ≤ n → x i < x (i + 1)) :=
by
  sorry

end exists_increasing_sequence_l2109_210912


namespace compute_expression_l2109_210942

theorem compute_expression (x : ℤ) (h : x = 3) : (x^8 + 24 * x^4 + 144) / (x^4 + 12) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l2109_210942


namespace maximum_value_of_a_l2109_210968

theorem maximum_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) : a ≤ 2924 := 
sorry

end maximum_value_of_a_l2109_210968


namespace greatest_possible_price_per_notebook_l2109_210956

theorem greatest_possible_price_per_notebook (budget entrance_fee : ℝ) (notebooks : ℕ) (tax_rate : ℝ) (price_per_notebook : ℝ) :
  budget = 160 ∧ entrance_fee = 5 ∧ notebooks = 18 ∧ tax_rate = 0.05 ∧ price_per_notebook * notebooks * (1 + tax_rate) ≤ (budget - entrance_fee) →
  price_per_notebook = 8 :=
by
  sorry

end greatest_possible_price_per_notebook_l2109_210956


namespace infinite_series_problem_l2109_210937

noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (2 * (n + 1)^2 - 3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))

theorem infinite_series_problem :
  infinite_series_sum = -4 :=
by sorry

end infinite_series_problem_l2109_210937


namespace granger_bought_12_cans_of_spam_l2109_210928

theorem granger_bought_12_cans_of_spam : 
  ∀ (S : ℕ), 
    (3 * 5 + 4 * 2 + 3 * S = 59) → 
    (S = 12) := 
by
  intro S h
  sorry

end granger_bought_12_cans_of_spam_l2109_210928


namespace depth_of_well_l2109_210976

theorem depth_of_well
  (d : ℝ)
  (h1 : ∃ t1 t2 : ℝ, 18 * t1^2 = d ∧ t2 = d / 1150 ∧ t1 + t2 = 8) :
  d = 33.18 :=
sorry

end depth_of_well_l2109_210976


namespace liked_both_desserts_l2109_210979

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end liked_both_desserts_l2109_210979


namespace geom_sum_eq_six_l2109_210909

variable (a : ℕ → ℝ)
variable (r : ℝ) -- common ratio for geometric sequence

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a (n + 1) > 0
axiom given_eq : a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36

-- Proof statement
theorem geom_sum_eq_six : a 2 + a 5 = 6 :=
sorry

end geom_sum_eq_six_l2109_210909


namespace t_shaped_region_slope_divides_area_in_half_l2109_210906

theorem t_shaped_region_slope_divides_area_in_half :
  ∃ (m : ℚ), (m = 4 / 11) ∧ (
    let area1 := 2 * (m * 2 * 4)
    let area2 := ((4 - m * 2) * 4) + 6
    area1 = area2
  ) :=
by
  sorry

end t_shaped_region_slope_divides_area_in_half_l2109_210906


namespace sqrt_product_simplification_l2109_210939

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end sqrt_product_simplification_l2109_210939


namespace productivity_increase_is_233_33_percent_l2109_210986

noncomputable def productivity_increase :
  Real :=
  let B := 1 -- represents the base number of bears made per week
  let H := 1 -- represents the base number of hours worked per week
  let P := B / H -- base productivity in bears per hour

  let B1 := 1.80 * B -- bears per week with first assistant
  let H1 := 0.90 * H -- hours per week with first assistant
  let P1 := B1 / H1 -- productivity with first assistant

  let B2 := 1.60 * B -- bears per week with second assistant
  let H2 := 0.80 * H -- hours per week with second assistant
  let P2 := B2 / H2 -- productivity with second assistant

  let B_both := B1 + B2 - B -- total bears with both assistants
  let H_both := H1 * H2 / H -- total hours with both assistants
  let P_both := B_both / H_both -- productivity with both assistants

  (P_both / P - 1) * 100

theorem productivity_increase_is_233_33_percent :
  productivity_increase = 233.33 :=
by
  sorry

end productivity_increase_is_233_33_percent_l2109_210986


namespace projection_of_vector_l2109_210998

open Real EuclideanSpace

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b

theorem projection_of_vector : 
  vector_projection (6, -3) (3, 0) = (6, 0) := 
by 
  sorry

end projection_of_vector_l2109_210998


namespace value_of_squared_difference_l2109_210907

theorem value_of_squared_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 :=
by
  sorry

end value_of_squared_difference_l2109_210907


namespace ellipse_focal_length_l2109_210964

theorem ellipse_focal_length (k : ℝ) :
  (∀ x y : ℝ, x^2 / k + y^2 / 2 = 1) →
  (∃ c : ℝ, 2 * c = 2 ∧ (k = 1 ∨ k = 3)) :=
by
  -- Given condition: equation of ellipse and focal length  
  intro h  
  sorry

end ellipse_focal_length_l2109_210964


namespace marco_score_percentage_less_l2109_210931

theorem marco_score_percentage_less
  (average_score : ℕ)
  (margaret_score : ℕ)
  (margaret_more_than_marco : ℕ)
  (h1 : average_score = 90)
  (h2 : margaret_score = 86)
  (h3 : margaret_more_than_marco = 5) :
  (average_score - (margaret_score - margaret_more_than_marco)) * 100 / average_score = 10 :=
by
  sorry

end marco_score_percentage_less_l2109_210931


namespace problem_g3_1_l2109_210999

theorem problem_g3_1 (a : ℝ) : 
  (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = a ↔ a = 2005 := 
sorry

end problem_g3_1_l2109_210999


namespace arnold_danny_age_l2109_210959

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 13) : x = 6 :=
by {
  sorry
}

end arnold_danny_age_l2109_210959


namespace ratio_of_geometric_sequence_sum_l2109_210941

theorem ratio_of_geometric_sequence_sum (a : ℕ → ℕ) 
    (q : ℕ) (h_q_pos : 0 < q) (h_q_ne_one : q ≠ 1)
    (h_geo_seq : ∀ n : ℕ, a (n + 1) = a n * q)
    (h_arith_seq : 2 * a (3 + 2) = a 3 - a (3 + 1)) :
  (a 4 * (1 - q ^ 4) / (1 - q)) / (a 4 * (1 - q ^ 2) / (1 - q)) = 5 / 4 := 
  sorry

end ratio_of_geometric_sequence_sum_l2109_210941


namespace find_unknown_rate_l2109_210989

variable {x : ℝ}

theorem find_unknown_rate (h : (3 * 100 + 1 * 150 + 2 * x) / 6 = 150) : x = 225 :=
by 
  sorry

end find_unknown_rate_l2109_210989


namespace complement_union_l2109_210922

open Set

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 3}

def B : Set ℕ := {3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {0, 2, 4} :=
by
  sorry

end complement_union_l2109_210922


namespace total_frogs_in_pond_l2109_210991

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l2109_210991


namespace scientific_notation_example_l2109_210960

theorem scientific_notation_example :
  110000 = 1.1 * 10^5 :=
by {
  sorry
}

end scientific_notation_example_l2109_210960


namespace circle_diameter_l2109_210925

-- The problem statement in Lean 4

theorem circle_diameter
  (d α β : ℝ) :
  ∃ r: ℝ,
  r * 2 = d * (Real.sin α) * (Real.sin β) / (Real.cos ((α + β) / 2) * (Real.sin ((α - β) / 2))) :=
sorry

end circle_diameter_l2109_210925


namespace suitable_value_for_x_evaluates_to_neg1_l2109_210962

noncomputable def given_expression (x : ℝ) : ℝ :=
  (x^3 + 2 * x^2) / (x^2 - 4 * x + 4) / (4 * x + 8) - 1 / (x - 2)

theorem suitable_value_for_x_evaluates_to_neg1 : 
  given_expression (-6) = -1 :=
by
  sorry

end suitable_value_for_x_evaluates_to_neg1_l2109_210962


namespace rectangle_cut_into_square_l2109_210947

theorem rectangle_cut_into_square (a b : ℝ) (h : a ≤ 4 * b) : 4 * b ≥ a := 
by 
  exact h

end rectangle_cut_into_square_l2109_210947


namespace tetrahedron_volume_lower_bound_l2109_210905

noncomputable def volume_tetrahedron (d1 d2 d3 : ℝ) : ℝ := sorry

theorem tetrahedron_volume_lower_bound {d1 d2 d3 : ℝ} (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d3 > 0) :
  volume_tetrahedron d1 d2 d3 ≥ (1 / 3) * d1 * d2 * d3 :=
sorry

end tetrahedron_volume_lower_bound_l2109_210905


namespace hike_up_days_l2109_210917

theorem hike_up_days (R_up R_down D_down D_up : ℝ) 
  (H1 : R_up = 8) 
  (H2 : R_down = 1.5 * R_up)
  (H3 : D_down = 24)
  (H4 : D_up / R_up = D_down / R_down) : 
  D_up / R_up = 2 :=
by
  sorry

end hike_up_days_l2109_210917


namespace initial_bales_l2109_210990

theorem initial_bales (bales_initially bales_added bales_now : ℕ)
  (h₀ : bales_added = 26)
  (h₁ : bales_now = 54)
  (h₂ : bales_now = bales_initially + bales_added) :
  bales_initially = 28 :=
by
  sorry

end initial_bales_l2109_210990


namespace bees_count_on_fifth_day_l2109_210914

theorem bees_count_on_fifth_day
  (initial_count : ℕ) (h_initial : initial_count = 1)
  (growth_factor : ℕ) (h_growth : growth_factor = 3) :
  let bees_at_day (n : ℕ) : ℕ := initial_count * (growth_factor + 1) ^ n
  bees_at_day 5 = 1024 := 
by {
  sorry
}

end bees_count_on_fifth_day_l2109_210914


namespace observer_height_proof_l2109_210933

noncomputable def height_observer (d m α β : ℝ) : ℝ :=
  let cot_alpha := 1 / Real.tan α
  let cot_beta := 1 / Real.tan β
  let u := (d * (m * cot_beta - d)) / (2 * d - m * (cot_beta - cot_alpha))
  20 + Real.sqrt (400 + u * m * cot_alpha - u^2)

theorem observer_height_proof :
  height_observer 290 40 (11.4 * Real.pi / 180) (4.7 * Real.pi / 180) = 52 := sorry

end observer_height_proof_l2109_210933


namespace anns_age_l2109_210936

theorem anns_age (a b : ℕ)
  (h1 : a + b = 72)
  (h2 : ∃ y, y = a - b)
  (h3 : b = a / 3 + 2 * (a - b)) : a = 36 :=
by
  sorry

end anns_age_l2109_210936


namespace loss_per_metre_is_5_l2109_210932

-- Definitions
def selling_price (total_meters : ℕ) : ℕ := 18000
def cost_price_per_metre : ℕ := 65
def total_meters : ℕ := 300

-- Loss per meter calculation
def loss_per_metre (selling_price : ℕ) (cost_price_per_metre : ℕ) (total_meters : ℕ) : ℕ :=
  ((cost_price_per_metre * total_meters) - selling_price) / total_meters

-- Theorem statement
theorem loss_per_metre_is_5 : loss_per_metre (selling_price total_meters) cost_price_per_metre total_meters = 5 :=
by
  sorry

end loss_per_metre_is_5_l2109_210932


namespace f_of_x_l2109_210957

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end f_of_x_l2109_210957


namespace rhombus_area_l2109_210997

theorem rhombus_area
  (side_length : ℝ)
  (h₀ : side_length = 2 * Real.sqrt 3)
  (tri_a_base : ℝ)
  (tri_b_base : ℝ)
  (h₁ : tri_a_base = side_length)
  (h₂ : tri_b_base = side_length) :
  ∃ rhombus_area : ℝ,
    rhombus_area = 8 * Real.sqrt 3 - 12 :=
by
  sorry

end rhombus_area_l2109_210997


namespace solve_equation_l2109_210923

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l2109_210923


namespace find_a3_a4_a5_l2109_210920

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 2 * a n

noncomputable def sum_first_three (a : ℕ → ℝ) : Prop :=
a 0 + a 1 + a 2 = 21

theorem find_a3_a4_a5 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : sum_first_three a) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end find_a3_a4_a5_l2109_210920


namespace value_multiplied_by_l2109_210913

theorem value_multiplied_by (x : ℝ) (h : (7.5 / 6) * x = 15) : x = 12 :=
by
  sorry

end value_multiplied_by_l2109_210913


namespace binary_to_decimal_l2109_210954

theorem binary_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5) :=
by
  sorry

end binary_to_decimal_l2109_210954


namespace train_crosses_pole_in_l2109_210961

noncomputable def train_crossing_time (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (5.0 / 18.0)
  length / speed_m_s

theorem train_crosses_pole_in : train_crossing_time 175 180 = 3.5 :=
by
  -- Proof would be here, but for now, it is omitted.
  sorry

end train_crosses_pole_in_l2109_210961


namespace norma_found_cards_l2109_210993

/-- Assume Norma originally had 88.0 cards. -/
def original_cards : ℝ := 88.0

/-- Assume Norma now has a total of 158 cards. -/
def total_cards : ℝ := 158

/-- Prove that Norma found 70 cards. -/
theorem norma_found_cards : total_cards - original_cards = 70 := 
by
  sorry

end norma_found_cards_l2109_210993


namespace number_of_possible_ceil_values_l2109_210952

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l2109_210952


namespace compute_105_squared_l2109_210916

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end compute_105_squared_l2109_210916


namespace find_divisor_l2109_210949

theorem find_divisor (x : ℝ) (h : x / n = 0.01 * (x * n)) : n = 10 :=
sorry

end find_divisor_l2109_210949


namespace prob_same_color_is_correct_l2109_210918

noncomputable def prob_same_color : ℚ :=
  let green_prob := (8 : ℚ) / 10
  let red_prob := (2 : ℚ) / 10
  (green_prob)^2 + (red_prob)^2

theorem prob_same_color_is_correct :
  prob_same_color = 17 / 25 := by
  sorry

end prob_same_color_is_correct_l2109_210918


namespace spicy_hot_noodles_plates_l2109_210965

theorem spicy_hot_noodles_plates (total_plates lobster_rolls seafood_noodles spicy_hot_noodles : ℕ) :
  total_plates = 55 →
  lobster_rolls = 25 →
  seafood_noodles = 16 →
  spicy_hot_noodles = total_plates - (lobster_rolls + seafood_noodles) →
  spicy_hot_noodles = 14 := by
  intros h_total h_lobster h_seafood h_eq
  rw [h_total, h_lobster, h_seafood] at h_eq
  exact h_eq

end spicy_hot_noodles_plates_l2109_210965


namespace systematic_sampling_removal_count_l2109_210974

-- Define the conditions
def total_population : Nat := 1252
def sample_size : Nat := 50

-- Define the remainder after division
def remainder := total_population % sample_size

-- Proof statement
theorem systematic_sampling_removal_count :
  remainder = 2 := by
    sorry

end systematic_sampling_removal_count_l2109_210974
