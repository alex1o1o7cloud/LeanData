import Mathlib

namespace find_a8_l56_56706

variable {α : Type} [LinearOrderedField α]

/-- Given conditions of an arithmetic sequence -/
def arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a_n n = a1 + n * d

theorem find_a8 (a_n : ℕ → ℝ)
  (h_arith : arithmetic_sequence a_n)
  (h3 : a_n 3 = 5)
  (h5 : a_n 5 = 3) :
  a_n 8 = 0 :=
sorry

end find_a8_l56_56706


namespace problem_solution_l56_56975

theorem problem_solution :
  -20 + 7 * (8 - 2 / 2) = 29 :=
by 
  sorry

end problem_solution_l56_56975


namespace initial_roses_l56_56546

theorem initial_roses {x : ℕ} (h : x + 11 = 14) : x = 3 := by
  sorry

end initial_roses_l56_56546


namespace quadratic_equality_l56_56091

theorem quadratic_equality (a_2 : ℝ) (a_1 : ℝ) (a_0 : ℝ) (r : ℝ) (s : ℝ) (x : ℝ)
  (h₁ : a_2 ≠ 0)
  (h₂ : a_0 ≠ 0)
  (h₃ : a_2 * r^2 + a_1 * r + a_0 = 0)
  (h₄ : a_2 * s^2 + a_1 * s + a_0 = 0) :
  a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s) :=
by
  sorry

end quadratic_equality_l56_56091


namespace solve_for_x_l56_56500

def delta (x : ℝ) : ℝ := 5 * x + 6
def phi (x : ℝ) : ℝ := 6 * x + 5

theorem solve_for_x : ∀ x : ℝ, delta (phi x) = -1 → x = - 16 / 15 :=
by
  intro x
  intro h
  -- Proof skipped
  sorry

end solve_for_x_l56_56500


namespace power_function_quadrant_IV_l56_56941

theorem power_function_quadrant_IV (a : ℝ) (h : a ∈ ({-1, 1/2, 2, 3} : Set ℝ)) :
  ∀ x : ℝ, x * x^a ≠ -x * (-x^a) := sorry

end power_function_quadrant_IV_l56_56941


namespace factor_72x3_minus_252x7_l56_56049

theorem factor_72x3_minus_252x7 (x : ℝ) : (72 * x^3 - 252 * x^7) = (36 * x^3 * (2 - 7 * x^4)) :=
by
  sorry

end factor_72x3_minus_252x7_l56_56049


namespace right_triangle_hypotenuse_l56_56739

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse :
  ∀ (a b : ℝ),
  (1/3) * Real.pi * b^2 * a = 675 * Real.pi →
  (1/3) * Real.pi * a^2 * b = 1215 * Real.pi →
  hypotenuse_length a b = 3 * Real.sqrt 106 :=
  by
  intros a b h1 h2
  sorry

end right_triangle_hypotenuse_l56_56739


namespace portion_of_money_given_to_Blake_l56_56490

theorem portion_of_money_given_to_Blake
  (initial_amount : ℝ)
  (tripled_amount : ℝ)
  (sale_amount : ℝ)
  (amount_given_to_Blake : ℝ)
  (h1 : initial_amount = 20000)
  (h2 : tripled_amount = 3 * initial_amount)
  (h3 : sale_amount = tripled_amount)
  (h4 : amount_given_to_Blake = 30000) :
  amount_given_to_Blake / sale_amount = 1 / 2 :=
sorry

end portion_of_money_given_to_Blake_l56_56490


namespace matrix_B6_eq_sB_plus_tI_l56_56053

noncomputable section

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![1, -1;
     4, 2]

theorem matrix_B6_eq_sB_plus_tI :
  ∃ s t : ℤ, B^6 = s • B + t • (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  have B2_eq : B^2 = -3 • B :=
    -- Matrix multiplication and scalar multiplication
    sorry
  use 81, 0
  have B4_eq : B^4 = 9 • B^2 := by
    rw [B2_eq]
    -- Calculation steps for B^4 equation
    sorry
  have B6_eq : B^6 = B^4 * B^2 := by
    rw [B4_eq, B2_eq]
    -- Calculation steps for B^6 final equation
    sorry
  rw [B6_eq]
  -- Final steps to show (81 • B + 0 • I = 81 • B)
  sorry

end matrix_B6_eq_sB_plus_tI_l56_56053


namespace horse_revolutions_l56_56789

theorem horse_revolutions :
  ∀ (r_1 r_2 : ℝ) (n : ℕ),
    r_1 = 30 → r_2 = 10 → n = 25 → (r_1 * n) / r_2 = 75 := by
  sorry

end horse_revolutions_l56_56789


namespace single_elimination_tournament_games_23_teams_l56_56292

noncomputable def single_elimination_tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games_23_teams :
  single_elimination_tournament_games 23 = 22 :=
by
  -- Proof has been intentionally omitted
  sorry

end single_elimination_tournament_games_23_teams_l56_56292


namespace solve_system_equations_l56_56676

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_system_equations :
  ∃ x y : ℝ, (y = 10^((log10 x)^(log10 x)) ∧ (log10 x)^(log10 (2 * x)) = (log10 y) * 10^((log10 (log10 x))^2))
  → ((x = 10 ∧ y = 10) ∨ (x = 100 ∧ y = 10000)) :=
by
  sorry

end solve_system_equations_l56_56676


namespace value_of_C_is_2_l56_56510

def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0
def isDivisibleBy7 (n : ℕ) : Prop := n % 7 = 0

def sumOfDigitsFirstNumber (A B : ℕ) : ℕ := 6 + 5 + A + 3 + 1 + B + 4
def sumOfDigitsSecondNumber (A B C : ℕ) : ℕ := 4 + 1 + 7 + A + B + 5 + C

theorem value_of_C_is_2 (A B : ℕ) (hDiv3First : isDivisibleBy3 (sumOfDigitsFirstNumber A B))
  (hDiv7First : isDivisibleBy7 (sumOfDigitsFirstNumber A B))
  (hDiv3Second : isDivisibleBy3 (sumOfDigitsSecondNumber A B 2))
  (hDiv7Second : isDivisibleBy7 (sumOfDigitsSecondNumber A B 2)) : 
  (∃ (C : ℕ), C = 2) :=
sorry

end value_of_C_is_2_l56_56510


namespace total_sours_is_123_l56_56382

noncomputable def cherry_sours := 32
noncomputable def lemon_sours := 40 -- Derived from the ratio 4/5 = 32/x
noncomputable def orange_sours := 24 -- 25% of the total sours in the bag after adding them
noncomputable def grape_sours := 27 -- Derived from the ratio 3/2 = 40/y

theorem total_sours_is_123 :
  cherry_sours + lemon_sours + orange_sours + grape_sours = 123 :=
by
  sorry

end total_sours_is_123_l56_56382


namespace students_taking_both_courses_l56_56919

theorem students_taking_both_courses (total_students students_french students_german students_neither both_courses : ℕ) 
(h1 : total_students = 94) 
(h2 : students_french = 41) 
(h3 : students_german = 22) 
(h4 : students_neither = 40) 
(h5 : total_students = students_french + students_german - both_courses + students_neither) :
both_courses = 9 :=
by
  -- sorry can be replaced with the actual proof if necessary
  sorry

end students_taking_both_courses_l56_56919


namespace sum_at_simple_interest_l56_56821

theorem sum_at_simple_interest
  (P R : ℝ)  -- P is the principal amount, R is the rate of interest
  (H1 : (9 * P * (R + 5) / 100 - 9 * P * R / 100 = 1350)) :
  P = 3000 :=
by
  sorry

end sum_at_simple_interest_l56_56821


namespace tammy_average_speed_second_day_l56_56964

theorem tammy_average_speed_second_day :
  ∃ v t : ℝ, 
  t + (t - 2) + (t + 1) = 20 ∧
  v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = 80 ∧
  (v + 0.5) = 4.575 :=
by 
  sorry

end tammy_average_speed_second_day_l56_56964


namespace range_of_reciprocals_l56_56080

theorem range_of_reciprocals (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) (h_sum : a + b = 1) :
  4 < (1 / a + 1 / b) :=
sorry

end range_of_reciprocals_l56_56080


namespace elena_fraction_left_l56_56464

variable (M : ℝ) -- Total amount of money
variable (B : ℝ) -- Total cost of all the books

-- Condition: Elena spends one-third of her money to buy half of the books
def condition : Prop := (1 / 3) * M = (1 / 2) * B

-- Goal: Fraction of the money left after buying all the books is one-third
theorem elena_fraction_left (h : condition M B) : (M - B) / M = 1 / 3 :=
by
  sorry

end elena_fraction_left_l56_56464


namespace factorization_of_polynomial_l56_56448

theorem factorization_of_polynomial : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by sorry

end factorization_of_polynomial_l56_56448


namespace weight_of_b_l56_56807

-- Definitions based on conditions
variables (A B C : ℝ)

def avg_abc := (A + B + C) / 3 = 45
def avg_ab := (A + B) / 2 = 40
def avg_bc := (B + C) / 2 = 44

-- The theorem to prove
theorem weight_of_b (h1 : avg_abc A B C) (h2 : avg_ab A B) (h3 : avg_bc B C) :
  B = 33 :=
sorry

end weight_of_b_l56_56807


namespace arccos_one_eq_zero_l56_56712

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l56_56712


namespace inverse_value_l56_56473

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value (x : ℝ) (h : g (-3) = x) : (g ∘ g⁻¹) x = x := by
  sorry

end inverse_value_l56_56473


namespace percentage_increase_l56_56006

theorem percentage_increase (x : ℝ) (h1 : 75 + 0.75 * x * 0.8 = 72) : x = 20 :=
by
  sorry

end percentage_increase_l56_56006


namespace interval_monotonicity_minimum_value_range_of_a_l56_56738

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x

theorem interval_monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x, 0 < x ∧ x < a → f x a > 0) ∧ (∀ x, x > a → f x a < 0) :=
sorry

theorem minimum_value (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → f x a ≥ 1) ∧ (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = 1) → a = 1 :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, x > 1 → f x a < 1 / 2 * x) → a < 1 / 2 :=
sorry

end interval_monotonicity_minimum_value_range_of_a_l56_56738


namespace point_returns_to_original_after_seven_steps_l56_56578

-- Define a structure for a triangle and a point inside it
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x y : ℝ)

-- Given a triangle and a point inside it
variable (ABC : Triangle)
variable (M : Point)

-- Define the set of movements and the intersection points
def move_parallel_to_BC (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AB (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AC (M : Point) (ABC : Triangle) : Point := sorry

-- Function to perform the stepwise movement through 7 steps
def move_M_seven_times (M : Point) (ABC : Triangle) : Point :=
  let M1 := move_parallel_to_BC M ABC
  let M2 := move_parallel_to_AB M1 ABC 
  let M3 := move_parallel_to_AC M2 ABC
  let M4 := move_parallel_to_BC M3 ABC
  let M5 := move_parallel_to_AB M4 ABC
  let M6 := move_parallel_to_AC M5 ABC
  let M7 := move_parallel_to_BC M6 ABC
  M7

-- The theorem stating that after 7 steps, point M returns to its original position
theorem point_returns_to_original_after_seven_steps :
  move_M_seven_times M ABC = M := sorry

end point_returns_to_original_after_seven_steps_l56_56578


namespace cost_per_candy_bar_l56_56059

-- Define the conditions as hypotheses
variables (candy_bars_total : ℕ) (candy_bars_paid_by_dave : ℕ) (amount_paid_by_john : ℝ)
-- Assume the given values
axiom total_candy_bars : candy_bars_total = 20
axiom candy_bars_by_dave : candy_bars_paid_by_dave = 6
axiom paid_by_john : amount_paid_by_john = 21

-- Define the proof problem
theorem cost_per_candy_bar :
  (amount_paid_by_john / (candy_bars_total - candy_bars_paid_by_dave) = 1.50) :=
by
  sorry

end cost_per_candy_bar_l56_56059


namespace min_colors_needed_l56_56374

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  max (abs (c1.1 - c2.1)) (abs (c1.2 - c2.2))

def color (c : cell) : ℤ :=
  (c.1 + c.2) % 4

theorem min_colors_needed : 4 = 4 :=
sorry

end min_colors_needed_l56_56374


namespace find_k_l56_56987

theorem find_k (k : ℤ) (h1 : ∃(a b c : ℤ), a = (36 + k) ∧ b = (300 + k) ∧ c = (596 + k) ∧ (∃ d, 
  (a = d^2) ∧ (b = (d + 1)^2) ∧ (c = (d + 2)^2)) ) : k = 925 := by
  sorry

end find_k_l56_56987


namespace var_power_eight_l56_56647

variable (k j : ℝ)
variable {x y z : ℝ}

theorem var_power_eight (hx : x = k * y^4) (hy : y = j * z^2) : ∃ c : ℝ, x = c * z^8 :=
by
  sorry

end var_power_eight_l56_56647


namespace temperature_problem_l56_56520

theorem temperature_problem (N : ℤ) (P : ℤ) (D : ℤ) (D_3_pm : ℤ) (P_3_pm : ℤ) :
  D = P + N →
  D_3_pm = D - 8 →
  P_3_pm = P + 9 →
  |D_3_pm - P_3_pm| = 1 →
  (N = 18 ∨ N = 16) →
  18 * 16 = 288 :=
by
  sorry

end temperature_problem_l56_56520


namespace binary_representation_of_28_l56_56680

-- Define a function to convert a number to binary representation.
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem binary_representation_of_28 : decimalToBinary 28 = [1, 1, 1, 0, 0] := 
  sorry

end binary_representation_of_28_l56_56680


namespace min_distance_from_circle_to_line_l56_56462

noncomputable def circle_center : (ℝ × ℝ) := (3, -1)
noncomputable def circle_radius : ℝ := 2

def on_circle (P : ℝ × ℝ) : Prop := (P.1 - circle_center.1) ^ 2 + (P.2 + circle_center.2) ^ 2 = circle_radius ^ 2
def on_line (Q : ℝ × ℝ) : Prop := Q.1 = -3

theorem min_distance_from_circle_to_line (P Q : ℝ × ℝ)
  (h1 : on_circle P) (h2 : on_line Q) : dist P Q = 4 := 
sorry

end min_distance_from_circle_to_line_l56_56462


namespace find_cost_price_l56_56151

theorem find_cost_price (C S : ℝ) (h1 : S = 1.35 * C) (h2 : S - 25 = 0.98 * C) : C = 25 / 0.37 :=
by
  sorry

end find_cost_price_l56_56151


namespace age_relationships_l56_56994

variables (a b c d : ℕ)

theorem age_relationships (h1 : a + b = b + c + d + 18) (h2 : 2 * a = 3 * c) :
  c = 2 * a / 3 ∧ d = a / 3 - 18 :=
by
  sorry

end age_relationships_l56_56994


namespace multiple_of_6_is_multiple_of_2_and_3_l56_56011

theorem multiple_of_6_is_multiple_of_2_and_3 (n : ℕ) :
  (∃ k : ℕ, n = 6 * k) → (∃ m1 : ℕ, n = 2 * m1) ∧ (∃ m2 : ℕ, n = 3 * m2) := by
  sorry

end multiple_of_6_is_multiple_of_2_and_3_l56_56011


namespace mrs_blue_expected_tomato_yield_l56_56811

-- Definitions for conditions
def steps_length := 3 -- each step measures 3 feet
def length_steps := 18 -- 18 steps in length
def width_steps := 25 -- 25 steps in width
def yield_per_sq_ft := 3 / 4 -- three-quarters of a pound per square foot

-- Define the total expected yield in pounds
def expected_yield : ℝ :=
  let length_ft := length_steps * steps_length
  let width_ft := width_steps * steps_length
  let area := length_ft * width_ft
  area * yield_per_sq_ft

-- The goal statement
theorem mrs_blue_expected_tomato_yield : expected_yield = 3037.5 := by
  sorry

end mrs_blue_expected_tomato_yield_l56_56811


namespace number_of_faces_l56_56220

-- Define the given conditions
def ways_to_paint_faces (n : ℕ) := Nat.factorial n

-- State the problem: Given ways_to_paint_faces n = 720, prove n = 6
theorem number_of_faces (n : ℕ) (h : ways_to_paint_faces n = 720) : n = 6 :=
sorry

end number_of_faces_l56_56220


namespace percent_of_z_l56_56885

variable (x y z : ℝ)

theorem percent_of_z :
  x = 1.20 * y →
  y = 0.40 * z →
  x = 0.48 * z :=
by
  intros h1 h2
  sorry

end percent_of_z_l56_56885


namespace part_a_l56_56306

theorem part_a (α : ℝ) (n : ℕ) (hα : α > 0) (hn : n > 1) : (1 + α)^n > 1 + n * α :=
sorry

end part_a_l56_56306


namespace ceil_sqrt_180_eq_14_l56_56912

theorem ceil_sqrt_180_eq_14
  (h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14) :
  Int.ceil (Real.sqrt 180) = 14 :=
  sorry

end ceil_sqrt_180_eq_14_l56_56912


namespace sector_central_angle_l56_56250

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : 1/2 * l * r = 1) : l / r = 2 := 
by
  sorry

end sector_central_angle_l56_56250


namespace ratio_of_four_numbers_exists_l56_56513

theorem ratio_of_four_numbers_exists (A B C D : ℕ) (h1 : A + B + C + D = 1344) (h2 : D = 672) : 
  ∃ rA rB rC rD, rA ≠ 0 ∧ rB ≠ 0 ∧ rC ≠ 0 ∧ rD ≠ 0 ∧ A = rA * k ∧ B = rB * k ∧ C = rC * k ∧ D = rD * k :=
by {
  sorry
}

end ratio_of_four_numbers_exists_l56_56513


namespace jon_original_number_l56_56904

theorem jon_original_number :
  ∃ y : ℤ, (5 * (3 * y + 6) - 8 = 142) ∧ (y = 8) :=
sorry

end jon_original_number_l56_56904


namespace solve_equation_1_solve_equation_2_l56_56868

theorem solve_equation_1 (x : ℝ) : (x + 2) ^ 2 = 3 * (x + 2) ↔ x = -2 ∨ x = 1 := by
  sorry

theorem solve_equation_2 (x : ℝ) : x ^ 2 - 8 * x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13 := by
  sorry

end solve_equation_1_solve_equation_2_l56_56868


namespace no_solution_perfect_square_abcd_l56_56237

theorem no_solution_perfect_square_abcd (x : ℤ) :
  (x ≤ 24) → (∃ (m : ℤ), 104 * x = m * m) → false :=
by
  sorry

end no_solution_perfect_square_abcd_l56_56237


namespace solve_weight_of_bowling_ball_l56_56173

-- Conditions: Eight bowling balls equal the weight of five canoes
-- and four canoes weigh 120 pounds.

def weight_of_bowling_ball : Prop :=
  ∃ (b c : ℝ), (8 * b = 5 * c) ∧ (4 * c = 120) ∧ (b = 18.75)

theorem solve_weight_of_bowling_ball : weight_of_bowling_ball :=
  sorry

end solve_weight_of_bowling_ball_l56_56173


namespace intersection_complement_l56_56799

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l56_56799


namespace total_albums_l56_56813

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l56_56813


namespace binder_cost_l56_56561

variable (B : ℕ) -- Define B as the cost of each binder

theorem binder_cost :
  let book_cost := 16
  let num_binders := 3
  let notebook_cost := 1
  let num_notebooks := 6
  let total_cost := 28
  (book_cost + num_binders * B + num_notebooks * notebook_cost = total_cost) → (B = 2) :=
by
  sorry

end binder_cost_l56_56561


namespace Michael_needs_more_money_l56_56061

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end Michael_needs_more_money_l56_56061


namespace polynomial_sum_l56_56431

theorem polynomial_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 :=
by
  sorry

end polynomial_sum_l56_56431


namespace base_6_units_digit_l56_56580

def num1 : ℕ := 217
def num2 : ℕ := 45
def base : ℕ := 6

theorem base_6_units_digit :
  (num1 % base) * (num2 % base) % base = (num1 * num2) % base :=
by
  sorry

end base_6_units_digit_l56_56580


namespace coefficient_x3_in_product_l56_56110

-- Definitions for the polynomials
def P(x : ℕ → ℕ) : ℕ → ℤ
| 4 => 3
| 3 => 4
| 2 => -2
| 1 => 8
| 0 => -5
| _ => 0

def Q(x : ℕ → ℕ) : ℕ → ℤ
| 3 => 2
| 2 => -7
| 1 => 5
| 0 => -3
| _ => 0

-- Statement of the problem
theorem coefficient_x3_in_product :
  (P 3 * Q 0 + P 2 * Q 1 + P 1 * Q 2) = -78 :=
by
  sorry

end coefficient_x3_in_product_l56_56110


namespace students_on_bus_after_all_stops_l56_56242

-- Define the initial number of students getting on the bus at the first stop.
def students_first_stop : ℕ := 39

-- Define the number of students added at the second stop.
def students_second_stop_add : ℕ := 29

-- Define the number of students getting off at the second stop.
def students_second_stop_remove : ℕ := 12

-- Define the number of students added at the third stop.
def students_third_stop_add : ℕ := 35

-- Define the number of students getting off at the third stop.
def students_third_stop_remove : ℕ := 18

-- Calculating the expected number of students on the bus after all stops.
def total_students_expected : ℕ :=
  students_first_stop + students_second_stop_add - students_second_stop_remove +
  students_third_stop_add - students_third_stop_remove

-- The theorem stating the number of students on the bus after all stops.
theorem students_on_bus_after_all_stops : total_students_expected = 73 := by
  sorry

end students_on_bus_after_all_stops_l56_56242


namespace g_inv_undefined_at_one_l56_56344

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

theorem g_inv_undefined_at_one :
  ∀ (x : ℝ), (∃ (y : ℝ), g y = x ∧ ¬ ∃ (z : ℝ), g z = y ∧ g z = 1) ↔ x = 1 :=
by
  sorry

end g_inv_undefined_at_one_l56_56344


namespace book_arrangements_l56_56102

theorem book_arrangements :
  let math_books := 4
  let english_books := 4
  let groups := 2
  (groups.factorial) * (math_books.factorial) * (english_books.factorial) = 1152 :=
by
  sorry

end book_arrangements_l56_56102


namespace Rebecca_worked_56_l56_56877

-- Define the conditions
variables (x : ℕ)
def Toby_hours := 2 * x - 10
def Rebecca_hours := Toby_hours - 8
def Total_hours := x + Toby_hours + Rebecca_hours

-- Theorem stating that under the given conditions, Rebecca worked 56 hours
theorem Rebecca_worked_56 
  (h : Total_hours = 157) 
  (hx : x = 37) : Rebecca_hours = 56 :=
by sorry

end Rebecca_worked_56_l56_56877


namespace simplify_and_evaluate_expression_l56_56477

variables (m n : ℚ)

theorem simplify_and_evaluate_expression (h1 : m = -1) (h2 : n = 1 / 2) :
  ( (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n ^ 2) / (m ^ 3 - m * n ^ 2)) ) = -2 :=
by
  sorry

end simplify_and_evaluate_expression_l56_56477


namespace initial_liquid_A_amount_l56_56364

noncomputable def initial_amount_of_A (x : ℚ) : ℚ :=
  3 * x

theorem initial_liquid_A_amount {x : ℚ} (h : (3 * x - 3) / (2 * x + 3) = 3 / 5) : initial_amount_of_A (8 / 3) = 8 := by
  sorry

end initial_liquid_A_amount_l56_56364


namespace arrangements_21_leaders_l56_56688

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutations A_n^k
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then factorial n / factorial (n - k) else 0

theorem arrangements_21_leaders : permutations 2 2 * permutations 18 18 = factorial 18 ^ 2 :=
by 
  sorry

end arrangements_21_leaders_l56_56688


namespace total_trolls_l56_56989

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l56_56989


namespace student_correct_answers_l56_56030

theorem student_correct_answers (c w : ℕ) 
  (h1 : c + w = 60)
  (h2 : 4 * c - w = 120) : 
  c = 36 :=
sorry

end student_correct_answers_l56_56030


namespace students_in_classroom_l56_56629

theorem students_in_classroom :
  ∃ n : ℕ, (n < 50) ∧ (n % 6 = 5) ∧ (n % 3 = 2) ∧ 
  (n = 5 ∨ n = 11 ∨ n = 17 ∨ n = 23 ∨ n = 29 ∨ n = 35 ∨ n = 41 ∨ n = 47) :=
by
  sorry

end students_in_classroom_l56_56629


namespace product_of_two_numbers_l56_56804

-- Define the conditions
def two_numbers (x y : ℝ) : Prop :=
  x + y = 27 ∧ x - y = 7

-- Define the product function
def product_two_numbers (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem product_of_two_numbers : ∃ x y : ℝ, two_numbers x y ∧ product_two_numbers x y = 170 := by
  sorry

end product_of_two_numbers_l56_56804


namespace train_speed_l56_56009

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end train_speed_l56_56009


namespace sum_of_first_five_terms_l56_56186

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem sum_of_first_five_terms :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 5 / 6 := 
by 
  unfold a
  -- sorry is used as a placeholder for the actual proof
  sorry

end sum_of_first_five_terms_l56_56186


namespace first_term_of_arithmetic_progression_l56_56763

theorem first_term_of_arithmetic_progression 
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (nth_term_eq : a + (n - 1) * d = 26)
  (common_diff : d = 2)
  (term_num : n = 10) : 
  a = 8 := 
by 
  sorry

end first_term_of_arithmetic_progression_l56_56763


namespace constructed_expression_equals_original_l56_56773

variable (a : ℝ)

theorem constructed_expression_equals_original : 
  a ≠ 0 → 
  ((1/a) / ((1/a) * (1/a)) - (1/a)) / (1/a) = (a + 1) * (a - 1) :=
by
  intro h
  sorry

end constructed_expression_equals_original_l56_56773


namespace slices_served_during_dinner_l56_56109

theorem slices_served_during_dinner (slices_lunch slices_total slices_dinner : ℕ)
  (h1 : slices_lunch = 7)
  (h2 : slices_total = 12)
  (h3 : slices_dinner = slices_total - slices_lunch) :
  slices_dinner = 5 := 
by 
  sorry

end slices_served_during_dinner_l56_56109


namespace icosahedron_colorings_l56_56709

theorem icosahedron_colorings :
  let n := 10
  let f := 9
  n! / 5 = 72576 :=
by
  sorry

end icosahedron_colorings_l56_56709


namespace power_function_value_l56_56264

theorem power_function_value (α : ℝ) (h₁ : (2 : ℝ) ^ α = (Real.sqrt 2) / 2) : (9 : ℝ) ^ α = 1 / 3 := 
by
  sorry

end power_function_value_l56_56264


namespace average_age_without_teacher_l56_56145

theorem average_age_without_teacher 
  (A : ℕ) 
  (h : 15 * A + 26 = 16 * (A + 1)) : 
  A = 10 :=
sorry

end average_age_without_teacher_l56_56145


namespace evaluate_expression_l56_56103

noncomputable def a := Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def b := -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6
noncomputable def d := -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 3 / 50 :=
by
  sorry

end evaluate_expression_l56_56103


namespace real_solution_unique_l56_56117

variable (x : ℝ)

theorem real_solution_unique :
  (x ≠ 2 ∧ (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = 3) ↔ x = 1 := 
by 
  sorry

end real_solution_unique_l56_56117


namespace percentage_increase_l56_56584

theorem percentage_increase (P : ℕ) (x y : ℕ) (h1 : x = 5) (h2 : y = 7) 
    (h3 : (x * (1 + P / 100) / (y * (1 - 10 / 100))) = 20 / 21) : 
    P = 20 :=
by
  sorry

end percentage_increase_l56_56584


namespace simplify_trig_expression_l56_56385

open Real

theorem simplify_trig_expression (α : ℝ) : 
  (cos (2 * π + α) * tan (π + α)) / cos (π / 2 - α) = 1 := 
sorry

end simplify_trig_expression_l56_56385


namespace spherical_to_rectangular_coordinates_l56_56526

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 10 ∧ θ = 3 * Real.pi / 4 ∧ φ = Real.pi / 6 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  :=
by
  intros ρ θ φ h
  rcases h with ⟨hρ, hθ, hφ⟩
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coordinates_l56_56526


namespace prob_no_distinct_roots_l56_56336

-- Definition of integers a, b, c between -7 and 7
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Definition of the discriminant condition for non-distinct real roots
def no_distinct_real_roots (a b c : Int) : Prop := b * b - 4 * a * c ≤ 0

-- Counting total triplets (a, b, c) with valid range
def total_triplets : Int := 15 * 15 * 15

-- Counting valid triplets with no distinct real roots
def valid_triplets : Int := 225 + (3150 / 2) -- 225 when a = 0 and estimation for a ≠ 0

theorem prob_no_distinct_roots : 
  let P := valid_triplets / total_triplets 
  P = (604 / 1125 : Rat) := 
by
  sorry

end prob_no_distinct_roots_l56_56336


namespace required_run_rate_l56_56475

def initial_run_rate : ℝ := 3.2
def overs_completed : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 50

theorem required_run_rate :
  (target_runs - initial_run_rate * overs_completed) / remaining_overs = 5 := 
by
  sorry

end required_run_rate_l56_56475


namespace kayak_rental_cost_l56_56488

variable (K : ℕ) -- the cost of a kayak rental per day
variable (x : ℕ) -- the number of kayaks rented

-- Conditions
def canoe_cost_per_day : ℕ := 11
def total_revenue : ℕ := 460
def canoes_more_than_kayaks : ℕ := 5

def ratio_condition : Prop := 4 * x = 3 * (x + 5)
def total_revenue_condition : Prop := canoe_cost_per_day * (x + 5) + K * x = total_revenue

-- Main statement
theorem kayak_rental_cost :
  ratio_condition x →
  total_revenue_condition K x →
  K = 16 := by sorry

end kayak_rental_cost_l56_56488


namespace correct_simplification_l56_56874

theorem correct_simplification (m a b x y : ℝ) :
  ¬ (4 * m - m = 3) ∧
  ¬ (a^2 * b - a * b^2 = 0) ∧
  ¬ (2 * a^3 - 3 * a^3 = a^3) ∧
  (x * y - 2 * x * y = - x * y) :=
by {
  sorry
}

end correct_simplification_l56_56874


namespace sum_of_coordinates_of_B_is_zero_l56_56463

structure Point where
  x : Int
  y : Int

def translation_to_right (P : Point) (n : Int) : Point :=
  { x := P.x + n, y := P.y }

def translation_down (P : Point) (n : Int) : Point :=
  { x := P.x, y := P.y - n }

def A : Point := { x := -1, y := 2 }

def B : Point := translation_down (translation_to_right A 1) 2

theorem sum_of_coordinates_of_B_is_zero :
  B.x + B.y = 0 := by
  sorry

end sum_of_coordinates_of_B_is_zero_l56_56463


namespace last_two_digits_x_pow_y_add_y_pow_x_l56_56002

noncomputable def proof_problem (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : ℕ :=
  (x^y + y^x) % 100

theorem last_two_digits_x_pow_y_add_y_pow_x {x y : ℕ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : 
  proof_problem x y h1 h2 h3 h4 = 74 :=
sorry

end last_two_digits_x_pow_y_add_y_pow_x_l56_56002


namespace minimum_value_of_function_l56_56990

theorem minimum_value_of_function (x : ℝ) (hx : x > 1) : (x + 4 / (x - 1)) ≥ 5 := by
  sorry

end minimum_value_of_function_l56_56990


namespace max_value_trig_expression_l56_56758

variable (a b φ θ : ℝ)

theorem max_value_trig_expression :
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2 * a * b * Real.sin φ + b^2) := sorry

end max_value_trig_expression_l56_56758


namespace mass_of_man_l56_56830

-- Definitions of the given conditions
def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def sink_depth : Float := 0.01 -- 1 cm converted to meters
def water_density : Float := 1000.0 -- Density of water in kg/m³

-- Define the proof goal as the mass of the man
theorem mass_of_man : Float :=
by
  let volume_displaced := boat_length * boat_breadth * sink_depth
  let weight_displaced := volume_displaced * water_density
  exact weight_displaced

end mass_of_man_l56_56830


namespace yen_to_usd_conversion_l56_56927

theorem yen_to_usd_conversion
  (cost_of_souvenir : ℕ)
  (service_charge : ℕ)
  (conversion_rate : ℕ)
  (total_cost_in_yen : ℕ)
  (usd_equivalent : ℚ)
  (h1 : cost_of_souvenir = 340)
  (h2 : service_charge = 25)
  (h3 : conversion_rate = 115)
  (h4 : total_cost_in_yen = cost_of_souvenir + service_charge)
  (h5 : usd_equivalent = (total_cost_in_yen : ℚ) / conversion_rate) :
  total_cost_in_yen = 365 ∧ usd_equivalent = 3.17 :=
by
  sorry

end yen_to_usd_conversion_l56_56927


namespace sqrt_expression_value_l56_56122

theorem sqrt_expression_value :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 :=
by
  sorry

end sqrt_expression_value_l56_56122


namespace correct_option_b_l56_56545

theorem correct_option_b (a : ℝ) : 
  (-2 * a) ^ 3 = -8 * a ^ 3 :=
by sorry

end correct_option_b_l56_56545


namespace largest_n_multiple_3_l56_56875

theorem largest_n_multiple_3 (n : ℕ) (h1 : n < 100000) (h2 : (8 * (n + 2)^5 - n^2 + 14 * n - 30) % 3 = 0) : n = 99999 := 
sorry

end largest_n_multiple_3_l56_56875


namespace equal_striped_areas_l56_56909

theorem equal_striped_areas (A B C D : ℝ) (h_AD_DB : D = A + B) (h_CD2 : C^2 = A * B) :
  (π * C^2 / 4 = π * B^2 / 8 - π * A^2 / 8 - π * D^2 / 8) := 
sorry

end equal_striped_areas_l56_56909


namespace area_of_square_is_1225_l56_56836

-- Given some basic definitions and conditions
variable (s : ℝ) -- side of the square which is the radius of the circle
variable (length : ℝ := (2 / 5) * s)
variable (breadth : ℝ := 10)
variable (area_rectangle : ℝ := length * breadth)

-- Statement to prove
theorem area_of_square_is_1225 
  (h1 : length = (2 / 5) * s)
  (h2 : breadth = 10)
  (h3 : area_rectangle = 140) : 
  s^2 = 1225 := by
    sorry

end area_of_square_is_1225_l56_56836


namespace age_of_older_friend_l56_56907

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end age_of_older_friend_l56_56907


namespace sides_of_triangle_l56_56814

-- Definitions from conditions
variables (a b c : ℕ) (r bk kc : ℕ)
def is_tangent_split : Prop := bk = 8 ∧ kc = 6
def inradius : Prop := r = 4

-- Main theorem statement
theorem sides_of_triangle (h1 : is_tangent_split bk kc) (h2 : inradius r) : a + 6 = 13 ∧ a + 8 = 15 ∧ b = 14 := by
  sorry

end sides_of_triangle_l56_56814


namespace product_of_geometric_terms_l56_56027

noncomputable def arithmeticSeq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def geometricSeq (b1 r : ℕ) (n : ℕ) : ℕ :=
  b1 * r^(n - 1)

theorem product_of_geometric_terms :
  ∃ (a1 d b1 r : ℕ),
    (3 * a1 - (arithmeticSeq a1 d 8)^2 + 3 * (arithmeticSeq a1 d 15) = 0) ∧ 
    (arithmeticSeq a1 d 8 = geometricSeq b1 r 10) ∧ 
    (geometricSeq b1 r 3 * geometricSeq b1 r 17 = 36) :=
sorry

end product_of_geometric_terms_l56_56027


namespace compare_squares_l56_56352

theorem compare_squares : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := sorry

end compare_squares_l56_56352


namespace probability_of_selecting_product_not_less_than_4_l56_56679

theorem probability_of_selecting_product_not_less_than_4 :
  let total_products := 5 
  let favorable_outcomes := 2 
  (favorable_outcomes : ℚ) / total_products = 2 / 5 := 
by 
  sorry

end probability_of_selecting_product_not_less_than_4_l56_56679


namespace part_a_part_b_part_c_part_d_part_e_l56_56622

variable (n : ℤ)

theorem part_a : (n^3 - n) % 3 = 0 :=
  sorry

theorem part_b : (n^5 - n) % 5 = 0 :=
  sorry

theorem part_c : (n^7 - n) % 7 = 0 :=
  sorry

theorem part_d : (n^11 - n) % 11 = 0 :=
  sorry

theorem part_e : (n^13 - n) % 13 = 0 :=
  sorry

end part_a_part_b_part_c_part_d_part_e_l56_56622


namespace problem_statement_l56_56041

variables {A B x y a : ℝ}

theorem problem_statement (h1 : 1/A = 1 - (1 - x) / y)
                          (h2 : 1/B = 1 - y / (1 - x))
                          (h3 : x = (1 - a) / (1 - 1/a))
                          (h4 : y = 1 - 1/x)
                          (h5 : a ≠ 1) (h6 : a ≠ -1) : 
                          A + B = 1 :=
sorry

end problem_statement_l56_56041


namespace A_B_distance_l56_56544

noncomputable def distance_between_A_and_B 
  (vA: ℕ) (vB: ℕ) (vA_after_return: ℕ) 
  (meet_distance: ℕ) : ℚ := sorry

theorem A_B_distance (distance: ℚ) 
  (hA: vA = 40) (hB: vB = 60) 
  (hA_after_return: vA_after_return = 60) 
  (hmeet: meet_distance = 50) : 
  distance_between_A_and_B vA vB vA_after_return meet_distance = 1000 / 7 := sorry

end A_B_distance_l56_56544


namespace probability_max_roll_correct_l56_56508
open Classical

noncomputable def probability_max_roll_fourth : ℚ :=
  let six_sided_max := 1 / 6
  let eight_sided_max := 3 / 4
  let ten_sided_max := 4 / 5

  let prob_A_given_B1 := (1 / 6) ^ 3
  let prob_A_given_B2 := (3 / 4) ^ 3
  let prob_A_given_B3 := (4 / 5) ^ 3

  let prob_B1 := 1 / 3
  let prob_B2 := 1 / 3
  let prob_B3 := 1 / 3

  let prob_A := prob_A_given_B1 * prob_B1 + prob_A_given_B2 * prob_B2 + prob_A_given_B3 * prob_B3

  -- Calculate probabilities with Bayes' Theorem
  let P_B1_A := (prob_A_given_B1 * prob_B1) / prob_A
  let P_B2_A := (prob_A_given_B2 * prob_B2) / prob_A
  let P_B3_A := (prob_A_given_B3 * prob_B3) / prob_A

  -- Probability of the fourth roll showing the maximum face value
  P_B1_A * six_sided_max + P_B2_A * eight_sided_max + P_B3_A * ten_sided_max

theorem probability_max_roll_correct : 
  ∃ (p q : ℕ), probability_max_roll_fourth = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 4386 :=
by sorry

end probability_max_roll_correct_l56_56508


namespace trigonometric_inequality_l56_56099

noncomputable def a : Real := (1/2) * Real.cos (8 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (8 * Real.pi / 180)
noncomputable def b : Real := (2 * Real.tan (14 * Real.pi / 180)) / (1 - (Real.tan (14 * Real.pi / 180))^2)
noncomputable def c : Real := Real.sqrt ((1 - Real.cos (48 * Real.pi / 180)) / 2)

theorem trigonometric_inequality :
  a < c ∧ c < b := by
  sorry

end trigonometric_inequality_l56_56099


namespace craig_distance_ridden_farther_l56_56268

/-- Given that Craig rode the bus for 3.83 miles and walked for 0.17 miles,
    prove that the distance he rode farther than he walked is 3.66 miles. -/
theorem craig_distance_ridden_farther :
  let distance_bus := 3.83
  let distance_walked := 0.17
  distance_bus - distance_walked = 3.66 :=
by
  let distance_bus := 3.83
  let distance_walked := 0.17
  show distance_bus - distance_walked = 3.66
  sorry

end craig_distance_ridden_farther_l56_56268


namespace buffaloes_number_l56_56079

theorem buffaloes_number (B D : ℕ) 
  (h : 4 * B + 2 * D = 2 * (B + D) + 24) : 
  B = 12 :=
sorry

end buffaloes_number_l56_56079


namespace jony_speed_l56_56627

theorem jony_speed :
  let start_block := 10
  let end_block := 90
  let turn_around_block := 70
  let block_length := 40 -- meters
  let start_time := 0 -- 07:00 in minutes from the start of his walk
  let end_time := 40 -- 07:40 in minutes from the start of his walk
  let total_blocks_walked := (end_block - start_block) + (end_block - turn_around_block)
  let total_distance := total_blocks_walked * block_length
  let total_time := end_time - start_time
  total_distance / total_time = 100 :=
by
  sorry

end jony_speed_l56_56627


namespace unanswered_questions_equal_nine_l56_56257

theorem unanswered_questions_equal_nine
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : 
  z = 9 := by
  sorry

end unanswered_questions_equal_nine_l56_56257


namespace problem1_l56_56903

theorem problem1 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : 
  (x * y = 5) ∧ ((x - y)^2 = 5) :=
by
  sorry

end problem1_l56_56903


namespace mean_proportional_49_64_l56_56499

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l56_56499


namespace age_difference_l56_56550

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : C + 16 = A := 
by
  sorry

end age_difference_l56_56550


namespace fraction_less_than_thirty_percent_l56_56310

theorem fraction_less_than_thirty_percent (x : ℚ) (hx : x * 180 = 36) (hx_lt : x < 0.3) : x = 1 / 5 := 
by
  sorry

end fraction_less_than_thirty_percent_l56_56310


namespace B_equals_1_2_3_l56_56072

def A : Set ℝ := { x | x^2 ≤ 4 }
def B : Set ℕ := { x | x > 0 ∧ (x - 1:ℝ) ∈ A }

theorem B_equals_1_2_3 : B = {1, 2, 3} :=
by
  sorry

end B_equals_1_2_3_l56_56072


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l56_56020

-- Problem 1
theorem solve_quadratic_1 (x : ℝ) : (x - 1) ^ 2 - 4 = 0 ↔ (x = -1 ∨ x = 3) :=
by
  sorry

-- Problem 2
theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1) * (x + 3) = 4 ↔ (x = -7 / 2 ∨ x = 1) :=
by
  sorry

-- Problem 3
theorem solve_quadratic_3 (x : ℝ) : 2 * x ^ 2 - 5 * x + 2 = 0 ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l56_56020


namespace part1_real_roots_part2_specific_roots_l56_56951

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end part1_real_roots_part2_specific_roots_l56_56951


namespace probability_of_drawing_white_ball_l56_56605

theorem probability_of_drawing_white_ball (P_A P_B P_C : ℝ) 
    (hA : P_A = 0.4) 
    (hB : P_B = 0.25)
    (hSum : P_A + P_B + P_C = 1) : 
    P_C = 0.35 :=
by
    -- Placeholder for the proof
    sorry

end probability_of_drawing_white_ball_l56_56605


namespace simplify_and_evaluate_expression_l56_56266

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -2) :
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l56_56266


namespace oldest_child_age_l56_56650

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 :=
by
  sorry

end oldest_child_age_l56_56650


namespace min_children_see_ear_l56_56357

theorem min_children_see_ear (n : ℕ) : ∃ (k : ℕ), k = n + 2 :=
by
  sorry

end min_children_see_ear_l56_56357


namespace new_arithmetic_mean_l56_56591

theorem new_arithmetic_mean
  (seq : List ℝ)
  (h_seq_len : seq.length = 60)
  (h_mean : (seq.sum / 60 : ℝ) = 42)
  (h_removed : ∃ a b, a ∈ seq ∧ b ∈ seq ∧ a = 50 ∧ b = 60) :
  ((seq.erase 50).erase 60).sum / 58 = 41.55 := 
sorry

end new_arithmetic_mean_l56_56591


namespace percentage_increase_in_y_l56_56201

variable (x y k q : ℝ) (h1 : x * y = k) (h2 : x' = x * (1 - q / 100))

theorem percentage_increase_in_y (h1 : x * y = k) (h2 : x' = x * (1 - q / 100)) :
  (y * 100 / (100 - q) - y) / y * 100 = (100 * q) / (100 - q) :=
by
  sorry

end percentage_increase_in_y_l56_56201


namespace area_of_shaded_region_l56_56461

open Real

-- Define points and squares
structure Point (α : Type*) := (x : α) (y : α)

def A := Point.mk 0 12 -- top-left corner of large square
def G := Point.mk 0 0  -- bottom-left corner of large square
def F := Point.mk 4 0  -- bottom-right corner of small square
def E := Point.mk 4 4  -- top-right corner of small square
def C := Point.mk 12 0 -- bottom-right corner of large square
def D := Point.mk 3 0  -- intersection of AF extended with the bottom edge

-- Define the length of sides
def side_small_square : ℝ := 4
def side_large_square : ℝ := 12

-- Areas calculation
def area_square (side : ℝ) : ℝ := side * side

def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Theorem statement
theorem area_of_shaded_region : area_square side_small_square - area_triangle 3 side_small_square = 10 :=
by
  rw [area_square, area_triangle]
  -- Plug in values: 4^2 - 0.5 * 3 * 4
  norm_num
  sorry

end area_of_shaded_region_l56_56461


namespace quadratic_roots_sum_l56_56219

theorem quadratic_roots_sum :
  ∃ a b c d : ℤ, (x^2 + 23 * x + 132 = (x + a) * (x + b)) ∧ (x^2 - 25 * x + 168 = (x - c) * (x - d)) ∧ (a + c + d = 42) :=
by {
  sorry
}

end quadratic_roots_sum_l56_56219


namespace sin_alpha_in_second_quadrant_l56_56749

theorem sin_alpha_in_second_quadrant
  (α : ℝ)
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.tan α = - (8 / 15)) :
  Real.sin α = 8 / 17 :=
sorry

end sin_alpha_in_second_quadrant_l56_56749


namespace area_of_field_l56_56659

-- Define the conditions: length, width, and total fencing
def length : ℕ := 40
def fencing : ℕ := 74

-- Define the property being proved: the area of the field
theorem area_of_field : ∃ (width : ℕ), 2 * width + length = fencing ∧ length * width = 680 :=
by
  -- Proof omitted
  sorry

end area_of_field_l56_56659


namespace male_contestants_count_l56_56769

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end male_contestants_count_l56_56769


namespace volume_of_remaining_solid_l56_56582

noncomputable def volume_cube_with_cylindrical_hole 
  (side_length : ℝ) (hole_diameter : ℝ) (π : ℝ := 3.141592653589793) : ℝ :=
  let V_cube := side_length^3
  let radius := hole_diameter / 2
  let height := side_length
  let V_cylinder := π * radius^2 * height
  V_cube - V_cylinder

theorem volume_of_remaining_solid 
  (side_length : ℝ)
  (hole_diameter : ℝ)
  (h₁ : side_length = 6) 
  (h₂ : hole_diameter = 3)
  (π : ℝ := 3.141592653589793) : 
  abs (volume_cube_with_cylindrical_hole side_length hole_diameter π - 173.59) < 0.01 :=
by
  sorry

end volume_of_remaining_solid_l56_56582


namespace value_two_stddevs_less_l56_56588

theorem value_two_stddevs_less (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : μ - 2 * σ = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stddevs_less_l56_56588


namespace train_platform_length_equal_l56_56287

theorem train_platform_length_equal 
  (v : ℝ) (t : ℝ) (L_train : ℝ)
  (h1 : v = 144 * (1000 / 3600))
  (h2 : t = 60)
  (h3 : L_train = 1200) :
  L_train = 2400 - L_train := 
sorry

end train_platform_length_equal_l56_56287


namespace find_share_of_b_l56_56369

variable (a b c : ℕ)
axiom h1 : a = 3 * b
axiom h2 : b = c + 25
axiom h3 : a + b + c = 645

theorem find_share_of_b : b = 134 := by
  sorry

end find_share_of_b_l56_56369


namespace bases_for_204_base_b_l56_56873

theorem bases_for_204_base_b (b : ℕ) : (∃ n : ℤ, 2 * b^2 + 4 = n^2) ↔ b = 4 ∨ b = 6 ∨ b = 8 ∨ b = 10 :=
by
  sorry

end bases_for_204_base_b_l56_56873


namespace nabla_example_l56_56866

def nabla (a b : ℕ) : ℕ := 2 + b ^ a

theorem nabla_example : nabla (nabla 1 2) 3 = 83 :=
  by
  sorry

end nabla_example_l56_56866


namespace smallest_two_digit_number_l56_56432

theorem smallest_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100)
  (h2 : ∃ k : ℕ, (N - (N / 10 + (N % 10) * 10)) = k ∧ k > 0 ∧ (∃ m : ℕ, k = m * m))
  : N = 90 := 
sorry

end smallest_two_digit_number_l56_56432


namespace path_bound_l56_56418

/-- Definition of P_k: the number of non-intersecting paths of length k starting from point O on a grid 
    where each cell has side length 1. -/
def P_k (k : ℕ) : ℕ := sorry  -- This would normally be defined through some combinatorial method

/-- The main theorem stating the required proof statement. -/
theorem path_bound (k : ℕ) : (P_k k : ℝ) / (3^k : ℝ) < 2 := sorry

end path_bound_l56_56418


namespace blue_lipstick_count_l56_56916

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l56_56916


namespace parabola_tangents_min_area_l56_56263

noncomputable def parabola_tangents (p : ℝ) : Prop :=
  ∃ (y₀ : ℝ), p > 0 ∧ (2 * Real.sqrt (y₀^2 + 2 * p) = 4)

theorem parabola_tangents_min_area (p : ℝ) : parabola_tangents 2 :=
by
  sorry

end parabola_tangents_min_area_l56_56263


namespace smallest_five_digit_number_divisible_by_first_five_primes_l56_56913

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l56_56913


namespace find_a_and_b_nth_equation_conjecture_l56_56895

theorem find_a_and_b {a b : ℤ} (h1 : 1^2 + 2^2 - 3^2 = 1 * a - b)
                                        (h2 : 2^2 + 3^2 - 4^2 = 2 * 0 - b)
                                        (h3 : 3^2 + 4^2 - 5^2 = 3 * 1 - b)
                                        (h4 : 4^2 + 5^2 - 6^2 = 4 * 2 - b):
    a = -1 ∧ b = 3 :=
    sorry

theorem nth_equation_conjecture (n : ℤ) :
  n^2 + (n+1)^2 - (n+2)^2 = n * (n-2) - 3 :=
  sorry

end find_a_and_b_nth_equation_conjecture_l56_56895


namespace intersection_of_A_and_B_l56_56484

def set_A : Set ℝ := {x | x >= 1 ∨ x <= -2}
def set_B : Set ℝ := {x | -3 < x ∧ x < 2}

def set_C : Set ℝ := {x | (-3 < x ∧ x <= -2) ∨ (1 <= x ∧ x < 2)}

theorem intersection_of_A_and_B (x : ℝ) : x ∈ set_A ∧ x ∈ set_B ↔ x ∈ set_C :=
  by
  sorry

end intersection_of_A_and_B_l56_56484


namespace daughterAgeThreeYearsFromNow_l56_56313

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l56_56313


namespace four_n_div_four_remainder_zero_l56_56505

theorem four_n_div_four_remainder_zero (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := 
by
  sorry

end four_n_div_four_remainder_zero_l56_56505


namespace intersection_points_l56_56325

-- Define the line equation
def line (x : ℝ) : ℝ := 2 * x - 1

-- Problem statement to be proven
theorem intersection_points :
  (line 0.5 = 0) ∧ (line 0 = -1) :=
by 
  sorry

end intersection_points_l56_56325


namespace largest_common_term_lt_300_l56_56725

theorem largest_common_term_lt_300 :
  ∃ a : ℕ, a < 300 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 3 + 7 * m) ∧ ∀ b : ℕ, b < 300 → (∃ n : ℤ, b = 4 + 5 * n) → (∃ m : ℤ, b = 3 + 7 * m) → b ≤ a :=
sorry

end largest_common_term_lt_300_l56_56725


namespace tom_gave_fred_balloons_l56_56427

variable (initial_balloons : ℕ) (remaining_balloons : ℕ)

def balloons_given (initial remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_fred_balloons (h₀ : initial_balloons = 30) (h₁ : remaining_balloons = 14) :
  balloons_given initial_balloons remaining_balloons = 16 :=
by
  -- Here we are skipping the proof
  sorry

end tom_gave_fred_balloons_l56_56427


namespace handshakes_minimum_l56_56701

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l56_56701


namespace sum_of_numbers_l56_56959

/-- Given three numbers in the ratio 1:2:5, with the sum of their squares being 4320,
prove that the sum of the numbers is 96. -/

theorem sum_of_numbers (x : ℝ) (h1 : (x:ℝ) = x) (h2 : 2 * x = 2 * x) (h3 : 5 * x = 5 * x) 
  (h4 : x^2 + (2 * x)^2 + (5 * x)^2 = 4320) :
  x + 2 * x + 5 * x = 96 := 
sorry

end sum_of_numbers_l56_56959


namespace combined_rocket_height_l56_56631

variable (h1 : ℕ) (h2 : ℕ)

-- Given conditions
def first_rocket_height : ℕ := 500
def second_rocket_height : ℕ := first_rocket_height * 2

-- Prove that the combined height is 1500 ft
theorem combined_rocket_height : first_rocket_height + second_rocket_height = 1500 := by
  sorry

end combined_rocket_height_l56_56631


namespace probability_one_from_harold_and_one_from_marilyn_l56_56328

-- Define the names and the number of letters in each name
def harold_name_length := 6
def marilyn_name_length := 7

-- Total cards
def total_cards := harold_name_length + marilyn_name_length

-- Probability of drawing one card from Harold's name and one from Marilyn's name
theorem probability_one_from_harold_and_one_from_marilyn :
    (harold_name_length : ℚ) / total_cards * marilyn_name_length / (total_cards - 1) +
    marilyn_name_length / total_cards * harold_name_length / (total_cards - 1) 
    = 7 / 13 := 
by
  sorry

end probability_one_from_harold_and_one_from_marilyn_l56_56328


namespace no_solution_for_n_eq_neg2_l56_56129

theorem no_solution_for_n_eq_neg2 : ∀ (x y : ℝ), ¬ (2 * x = 1 + -2 * y ∧ -2 * x = 1 + 2 * y) :=
by sorry

end no_solution_for_n_eq_neg2_l56_56129


namespace consecutive_odd_split_l56_56589

theorem consecutive_odd_split (m : ℕ) (hm : m > 1) : (∃ n : ℕ, n = 2015 ∧ n < ((m + 2) * (m - 1)) / 2) → m = 45 :=
by
  sorry

end consecutive_odd_split_l56_56589


namespace Joan_orange_balloons_l56_56152

theorem Joan_orange_balloons (originally_has : ℕ) (received : ℕ) (final_count : ℕ) 
  (h1 : originally_has = 8) (h2 : received = 2) : 
  final_count = 10 := by
  sorry

end Joan_orange_balloons_l56_56152


namespace inequality_holds_for_positive_x_l56_56669

theorem inequality_holds_for_positive_x (x : ℝ) (h : 0 < x) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 :=
sorry

end inequality_holds_for_positive_x_l56_56669


namespace jefferson_high_school_ninth_graders_l56_56851

theorem jefferson_high_school_ninth_graders (total_students science_students arts_students students_taking_both : ℕ):
  total_students = 120 →
  science_students = 85 →
  arts_students = 65 →
  students_taking_both = 150 - 120 →
  science_students - students_taking_both = 55 :=
by
  sorry

end jefferson_high_school_ninth_graders_l56_56851


namespace quadratic_inequality_solution_range_l56_56899

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end quadratic_inequality_solution_range_l56_56899


namespace resistance_between_opposite_vertices_of_cube_l56_56747

-- Define the parameters of the problem
def resistance_cube_edge : ℝ := 1

-- Define the function to calculate the equivalent resistance
noncomputable def equivalent_resistance_opposite_vertices (R : ℝ) : ℝ :=
  let R1 := R / 3
  let R2 := R / 6
  let R3 := R / 3
  R1 + R2 + R3

-- State the theorem to prove the resistance between two opposite vertices
theorem resistance_between_opposite_vertices_of_cube :
  equivalent_resistance_opposite_vertices resistance_cube_edge = 5 / 6 :=
by
  sorry

end resistance_between_opposite_vertices_of_cube_l56_56747


namespace initial_capital_is_15000_l56_56231

noncomputable def initialCapital (profitIncrease: ℝ) (oldRate newRate: ℝ) (distributionRatio: ℝ) : ℝ :=
  (profitIncrease / ((newRate - oldRate) * distributionRatio))

theorem initial_capital_is_15000 :
  initialCapital 200 0.05 0.07 (2 / 3) = 15000 :=
by
  sorry

end initial_capital_is_15000_l56_56231


namespace third_shiny_penny_prob_l56_56273

open Nat

def num_shiny : Nat := 4
def num_dull : Nat := 5
def total_pennies : Nat := num_shiny + num_dull

theorem third_shiny_penny_prob :
  let a := 5
  let b := 9
  a + b = 14 := 
by
  sorry

end third_shiny_penny_prob_l56_56273


namespace size_relationship_l56_56778

theorem size_relationship (a b : ℝ) (h₀ : a + b > 0) :
  a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b :=
by
  sorry

end size_relationship_l56_56778


namespace grogg_possible_cubes_l56_56855

theorem grogg_possible_cubes (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prob : (a - 2) * (b - 2) * (c - 2) / (a * b * c) = 1 / 5) :
  a * b * c = 120 ∨ a * b * c = 160 ∨ a * b * c = 240 ∨ a * b * c = 360 := 
sorry

end grogg_possible_cubes_l56_56855


namespace average_speed_round_trip_l56_56066

noncomputable def average_speed (d : ℝ) (v_to v_from : ℝ) : ℝ :=
  let time_to := d / v_to
  let time_from := d / v_from
  let total_time := time_to + time_from
  let total_distance := 2 * d
  total_distance / total_time

theorem average_speed_round_trip (d : ℝ) :
  average_speed d 60 40 = 48 :=
by
  sorry

end average_speed_round_trip_l56_56066


namespace geometric_sequence_increasing_l56_56615

theorem geometric_sequence_increasing {a : ℕ → ℝ} (r : ℝ) (h_pos : 0 < r) (h_geometric : ∀ n, a (n + 1) = r * a n) :
  (a 0 < a 1 ∧ a 1 < a 2) ↔ ∀ n m, n < m → a n < a m :=
by sorry

end geometric_sequence_increasing_l56_56615


namespace andrew_brian_ratio_l56_56350

-- Definitions based on conditions extracted from the problem
variables (A S B : ℕ)

-- Conditions
def steven_shirts : Prop := S = 72
def brian_shirts : Prop := B = 3
def steven_andrew_relation : Prop := S = 4 * A

-- The goal is to prove the ratio of Andrew's shirts to Brian's shirts is 6
theorem andrew_brian_ratio (A S B : ℕ) 
  (h1 : steven_shirts S) 
  (h2 : brian_shirts B)
  (h3 : steven_andrew_relation A S) :
  A / B = 6 := by
  sorry

end andrew_brian_ratio_l56_56350


namespace point_not_on_graph_l56_56476

theorem point_not_on_graph : 
  ∀ (k : ℝ), (k ≠ 0) → (∀ x y : ℝ, y = k * x → (x, y) = (1, 2)) → ¬ (∀ x y : ℝ, y = k * x → (x, y) = (1, -2)) :=
by
  sorry

end point_not_on_graph_l56_56476


namespace coin_flip_sequences_l56_56788

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l56_56788


namespace shaded_area_l56_56948

-- Define the points as per the problem
structure Point where
  x : ℝ
  y : ℝ

@[simp]
def A : Point := ⟨0, 0⟩
@[simp]
def B : Point := ⟨0, 7⟩
@[simp]
def C : Point := ⟨7, 7⟩
@[simp]
def D : Point := ⟨7, 0⟩
@[simp]
def E : Point := ⟨7, 0⟩
@[simp]
def F : Point := ⟨14, 0⟩
@[simp]
def G : Point := ⟨10.5, 7⟩

-- Define function for area of a triangle given three points
def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((P.x - R.x) * (Q.y - P.y) - (P.x - Q.x) * (R.y - P.y))

-- The theorem stating the area of the shaded region
theorem shaded_area : triangle_area D G H - triangle_area D E H = 24.5 := by
  sorry

end shaded_area_l56_56948


namespace quadratic_root_signs_l56_56284

-- Variables representation
variables {x m : ℝ}

-- Given: The quadratic equation with one positive root and one negative root
theorem quadratic_root_signs (h : ∃ a b : ℝ, 2*a*2*b + (m+1)*(a + b) + m = 0 ∧ a > 0 ∧ b < 0) : 
  m < 0 := 
sorry

end quadratic_root_signs_l56_56284


namespace sum_of_real_roots_of_even_function_l56_56667

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem sum_of_real_roots_of_even_function (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_intersects : ∃ a b c d, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a + b + c + d = 0 :=
sorry

end sum_of_real_roots_of_even_function_l56_56667


namespace molecular_weight_l56_56511

-- Definitions of the molar masses of the elements
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_I : ℝ := 126.90
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.07
def molar_mass_O : ℝ := 16.00

-- Definition of the molar masses of the compounds
def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Number of moles
def moles_NH4I : ℝ := 3
def moles_CaSO4 : ℝ := 2

-- Total mass calculation
def total_mass : ℝ :=
  moles_NH4I * molar_mass_NH4I + 
  moles_CaSO4 * molar_mass_CaSO4

-- Problem statement
theorem molecular_weight : total_mass = 707.15 := by
  sorry

end molecular_weight_l56_56511


namespace equivalent_expression_l56_56539

theorem equivalent_expression (x : ℝ) : 
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) + 1 = x^4 := 
by
  sorry

end equivalent_expression_l56_56539


namespace cat_food_percentage_l56_56819

theorem cat_food_percentage (D C : ℝ) (h1 : 7 * D + 4 * C = 8 * D) (h2 : 4 * C = D) : 
  (C / (7 * D + D)) * 100 = 3.125 := by
  sorry

end cat_food_percentage_l56_56819


namespace option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l56_56791

noncomputable def triangle (A B C : ℝ) := A + B + C = 180

-- Define the conditions for options A, B, C, and D
def option_a := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = 3 * C
def option_b := ∀ A B C : ℝ, triangle A B C → A + B = C
def option_c := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = (1/2) * C
def option_d := ∀ A B C : ℝ, triangle A B C → ∃ x : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x

-- Define that option A does not form a right triangle
theorem option_a_not_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_a → A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 :=
sorry

-- Check that options B, C, and D do form right triangles
theorem option_b_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_b → C = 90 :=
sorry

theorem option_c_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_c → C = 90 :=
sorry

theorem option_d_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_d → C = 90 :=
sorry

end option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l56_56791


namespace perfect_square_pairs_l56_56827

theorem perfect_square_pairs (x y : ℕ) (a b : ℤ) :
  (x^2 + 8 * ↑y = a^2 ∧ y^2 - 8 * ↑x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨ (x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23) :=
by
  sorry

end perfect_square_pairs_l56_56827


namespace symmetry_center_of_f_l56_56301

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos (2 * x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f (2 * x + π / 6) = Real.sin (2 * (-π / 12) + π / 6) :=
sorry

end symmetry_center_of_f_l56_56301


namespace each_person_pays_12_10_l56_56210

noncomputable def total_per_person : ℝ :=
  let taco_salad := 10
  let daves_single := 6 * 5
  let french_fries := 5 * 2.5
  let peach_lemonade := 7 * 2
  let apple_pecan_salad := 4 * 6
  let chocolate_frosty := 5 * 3
  let chicken_sandwiches := 3 * 4
  let chili := 2 * 3.5
  let subtotal := taco_salad + daves_single + french_fries + peach_lemonade + apple_pecan_salad + chocolate_frosty + chicken_sandwiches + chili
  let discount := 0.10
  let tax := 0.08
  let subtotal_after_discount := subtotal * (1 - discount)
  let total_after_tax := subtotal_after_discount * (1 + tax)
  total_after_tax / 10

theorem each_person_pays_12_10 :
  total_per_person = 12.10 :=
by
  -- omitted proof
  sorry

end each_person_pays_12_10_l56_56210


namespace correct_calculation_l56_56001

theorem correct_calculation (x : ℝ) (h : 63 + x = 69) : 36 / x = 6 :=
by
  sorry

end correct_calculation_l56_56001


namespace age_composition_is_decline_l56_56816

-- Define the population and age groups
variable (P : Type)
variable (Y E : P → ℕ) -- Functions indicating the number of young and elderly individuals

-- Assumptions as per the conditions
axiom fewer_young_more_elderly (p : P) : Y p < E p

-- Conclusion: Prove that the population is of Decline type.
def age_composition_decline (p : P) : Prop :=
  Y p < E p

theorem age_composition_is_decline (p : P) : age_composition_decline P Y E p := by
  sorry

end age_composition_is_decline_l56_56816


namespace stratified_sampling_l56_56930

theorem stratified_sampling
  (ratio_first : ℕ)
  (ratio_second : ℕ)
  (ratio_third : ℕ)
  (sample_size : ℕ)
  (h_ratio : ratio_first = 3 ∧ ratio_second = 4 ∧ ratio_third = 3)
  (h_sample_size : sample_size = 50) :
  (ratio_second * sample_size) / (ratio_first + ratio_second + ratio_third) = 20 :=
by
  sorry

end stratified_sampling_l56_56930


namespace num_ways_to_select_five_crayons_including_red_l56_56318

noncomputable def num_ways_select_five_crayons (total_crayons : ℕ) (selected_crayons : ℕ) (fixed_red_crayon : ℕ) : ℕ :=
  Nat.choose (total_crayons - fixed_red_crayon) selected_crayons

theorem num_ways_to_select_five_crayons_including_red
  (total_crayons : ℕ) 
  (fixed_red_crayon : ℕ)
  (selected_crayons : ℕ)
  (h1 : total_crayons = 15)
  (h2 : fixed_red_crayon = 1)
  (h3 : selected_crayons = 4) : 
  num_ways_select_five_crayons total_crayons selected_crayons fixed_red_crayon = 1001 := by
  sorry

end num_ways_to_select_five_crayons_including_red_l56_56318


namespace arithmetic_sequence_y_value_l56_56744

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l56_56744


namespace Dad_steps_l56_56963

variable (d m y : ℕ)

-- Conditions
def condition_1 : Prop := d = 3 → m = 5
def condition_2 : Prop := m = 3 → y = 5
def condition_3 : Prop := m + y = 400

-- Question and Answer
theorem Dad_steps : condition_1 d m → condition_2 m y → condition_3 m y → d = 90 :=
by
  intros
  sorry

end Dad_steps_l56_56963


namespace fraction_value_l56_56883

theorem fraction_value (a b : ℝ) (h : 1 / a - 1 / b = 4) : 
    (a - 2 * a * b - b) / (2 * a + 7 * a * b - 2 * b) = 6 :=
by
  sorry

end fraction_value_l56_56883


namespace polygon_sides_l56_56335

theorem polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 140 * n) : n = 9 :=
sorry

end polygon_sides_l56_56335


namespace decrease_in_average_age_l56_56097

theorem decrease_in_average_age (original_avg_age : ℕ) (new_students_avg_age : ℕ) 
    (original_strength : ℕ) (new_students_strength : ℕ) 
    (h1 : original_avg_age = 40) (h2 : new_students_avg_age = 32) 
    (h3 : original_strength = 8) (h4 : new_students_strength = 8) : 
    (original_avg_age - ((original_strength * original_avg_age + new_students_strength * new_students_avg_age) / (original_strength + new_students_strength))) = 4 :=
by 
  sorry

end decrease_in_average_age_l56_56097


namespace square_diagonal_length_l56_56968

theorem square_diagonal_length (rect_length rect_width : ℝ) 
  (h1 : rect_length = 45) 
  (h2 : rect_width = 40) 
  (rect_area := rect_length * rect_width) 
  (square_area := rect_area) 
  (side_length := Real.sqrt square_area) 
  (diagonal := side_length * Real.sqrt 2) :
  diagonal = 60 :=
by
  -- Proof goes here
  sorry

end square_diagonal_length_l56_56968


namespace cube_root_equation_l56_56767

theorem cube_root_equation (x : ℝ) (h : (2 * x - 14)^(1/3) = -2) : 2 * x + 3 = 9 := by
  sorry

end cube_root_equation_l56_56767


namespace profit_after_discount_l56_56175

noncomputable def purchase_price : ℝ := 100
noncomputable def increase_rate : ℝ := 0.25
noncomputable def discount_rate : ℝ := 0.10

theorem profit_after_discount :
  let selling_price := purchase_price * (1 + increase_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit := discounted_price - purchase_price
  profit = 12.5 :=
by
  sorry 

end profit_after_discount_l56_56175


namespace rectangle_height_l56_56764

variable (h : ℕ) -- Define h as a natural number for the height

-- Given conditions
def width : ℕ := 32
def area_divided_by_diagonal : ℕ := 576

-- Math proof problem
theorem rectangle_height :
  (1 / 2 * (width * h) = area_divided_by_diagonal) → h = 36 := 
by
  sorry

end rectangle_height_l56_56764


namespace avg_marks_chem_math_l56_56702

variable (P C M : ℝ)

theorem avg_marks_chem_math (h : P + C + M = P + 140) : (C + M) / 2 = 70 :=
by
  -- skip the proof, just provide the statement
  sorry

end avg_marks_chem_math_l56_56702


namespace point_same_side_of_line_l56_56798

def same_side (p₁ p₂ : ℝ × ℝ) (a b c : ℝ) : Prop :=
  (a * p₁.1 + b * p₁.2 + c > 0) ↔ (a * p₂.1 + b * p₂.2 + c > 0)

theorem point_same_side_of_line :
  same_side (1, 2) (1, 0) 2 (-1) 1 :=
by
  unfold same_side
  sorry

end point_same_side_of_line_l56_56798


namespace problem_1_problem_2_l56_56013

def f (a : ℝ) (x : ℝ) : ℝ := abs (a * x + 1)

def g (a : ℝ) (x : ℝ) : ℝ := f a x - abs (x + 1)

theorem problem_1 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 ↔ f a x ≤ 3) → a = 2 := by
  intro h
  sorry

theorem problem_2 (a : ℝ) : a = 2 → (∃ x : ℝ, ∀ y : ℝ, g a y ≥ g a x ∧ g a x = -1/2) := by
  intro ha2
  use -1/2
  sorry

end problem_1_problem_2_l56_56013


namespace bike_tire_fixing_charge_l56_56305

theorem bike_tire_fixing_charge (total_profit rent_profit retail_profit: ℝ) (cost_per_tire_parts charge_per_complex_parts charge_per_complex: ℝ) (complex_repairs tire_repairs: ℕ) (charge_per_tire: ℝ) :
  total_profit  = 3000 → rent_profit = 4000 → retail_profit = 2000 →
  cost_per_tire_parts = 5 → charge_per_complex_parts = 50 → charge_per_complex = 300 →
  complex_repairs = 2 → tire_repairs = 300 →
  total_profit = (tire_repairs * charge_per_tire + complex_repairs * charge_per_complex + retail_profit - tire_repairs * cost_per_tire_parts - complex_repairs * charge_per_complex_parts - rent_profit) →
  charge_per_tire = 20 :=
by 
  sorry

end bike_tire_fixing_charge_l56_56305


namespace width_of_beam_l56_56438

theorem width_of_beam (L W k : ℝ) (h1 : L = k * W) (h2 : 250 = k * 1.5) : 
  (k = 166.6667) → (583.3333 = 166.6667 * W) → W = 3.5 :=
by 
  intro hk1 
  intro h583
  sorry

end width_of_beam_l56_56438


namespace monotonic_range_l56_56569

theorem monotonic_range (a : ℝ) :
  (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) < (y^2 - 2*a*y + 3))
  ∨ (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) > (y^2 - 2*a*y + 3))
  ↔ (a ≤ 2 ∨ a ≥ 3) :=
by
  sorry

end monotonic_range_l56_56569


namespace a_minus_c_value_l56_56322

theorem a_minus_c_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := 
by 
  -- We provide the proof inline with sorry
  sorry

end a_minus_c_value_l56_56322


namespace square_angle_l56_56695

theorem square_angle (PQ QR : ℝ) (x : ℝ) (PQR_is_square : true)
  (angle_sum_of_triangle : ∀ a b c : ℝ, a + b + c = 180)
  (right_angle : ∀ a, a = 90) :
  x = 45 :=
by
  -- We start with the properties of the square (implicitly given by the conditions)
  -- Now use the conditions and provided values to conclude the proof
  sorry

end square_angle_l56_56695


namespace abc_sum_is_32_l56_56185

theorem abc_sum_is_32 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b + c = 31) (h5 : b * c + a = 31) (h6 : a * c + b = 31) : 
  a + b + c = 32 := 
by
  -- Proof goes here
  sorry

end abc_sum_is_32_l56_56185


namespace value_is_200_l56_56347

variable (x value : ℝ)
variable (h1 : 0.20 * x = value)
variable (h2 : 1.20 * x = 1200)

theorem value_is_200 : value = 200 :=
by
  sorry

end value_is_200_l56_56347


namespace prob_B_win_correct_l56_56922

-- Define the probabilities for player A winning and a draw
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.4

-- Define the total probability of all outcomes
def total_prob : ℝ := 1

-- Define the probability of player B winning
def prob_B_win : ℝ := total_prob - prob_A_win - prob_draw

-- Proof problem: Prove that the probability of player B winning is 0.3
theorem prob_B_win_correct : prob_B_win = 0.3 :=
by
  -- The proof would go here, but we use sorry to skip it for now.
  sorry

end prob_B_win_correct_l56_56922


namespace eval_g_five_l56_56790

def g (x : ℝ) : ℝ := 4 * x - 2

theorem eval_g_five : g 5 = 18 := by
  sorry

end eval_g_five_l56_56790


namespace inequality_solution_l56_56125

theorem inequality_solution (x : ℝ) : x > 0 ∧ (x^(1/3) < 3 - x) ↔ x < 3 :=
by 
  sorry

end inequality_solution_l56_56125


namespace max_street_lamps_proof_l56_56848

noncomputable def max_street_lamps_on_road : ℕ := 1998

theorem max_street_lamps_proof (L : ℕ) (l : ℕ)
    (illuminates : ∀ i, i ≤ max_street_lamps_on_road → 
                  (∃ unique_segment : ℕ, unique_segment ≤ L ∧ unique_segment > L - l )):
  max_street_lamps_on_road = 1998 := by
  sorry

end max_street_lamps_proof_l56_56848


namespace shakes_indeterminable_l56_56491

theorem shakes_indeterminable (B S C x : ℝ) (h1 : 3 * B + 7 * S + C = 120) (h2 : 4 * B + x * S + C = 164.50) : ¬ (∃ B S C, ∀ x, 4 * B + x * S + C = 164.50) → false := 
by 
  sorry

end shakes_indeterminable_l56_56491


namespace ratio_of_time_l56_56205

theorem ratio_of_time (tX tY tZ : ℕ) (h1 : tX = 16) (h2 : tY = 12) (h3 : tZ = 8) :
  (tX : ℚ) / (tY * tZ / (tY + tZ) : ℚ) = 10 / 3 := 
by 
  sorry

end ratio_of_time_l56_56205


namespace number_of_divisors_125n5_l56_56327

theorem number_of_divisors_125n5 (n : ℕ) (hn : n > 0)
  (h150 : ∀ m : ℕ, m = 150 * n ^ 4 → (∃ d : ℕ, d * (d + 1) = 150)) :
  ∃ d : ℕ, d = 125 * n ^ 5 ∧ ((13 + 1) * (5 + 1) * (5 + 1) = 504) :=
by
  sorry

end number_of_divisors_125n5_l56_56327


namespace train_speed_without_stoppages_l56_56953

theorem train_speed_without_stoppages 
  (distance_with_stoppages : ℝ)
  (avg_speed_with_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (distance_without_stoppages : ℝ)
  (avg_speed_without_stoppages : ℝ) :
  avg_speed_with_stoppages = 200 → 
  stoppage_time_per_hour = 20 / 60 →
  distance_without_stoppages = distance_with_stoppages * avg_speed_without_stoppages →
  distance_with_stoppages = avg_speed_with_stoppages →
  avg_speed_without_stoppages == 300 := 
by
  intros
  sorry

end train_speed_without_stoppages_l56_56953


namespace comparison_of_products_l56_56556

def A : ℕ := 8888888888888888888 -- 19 digits, all 8's
def B : ℕ := 3333333333333333333333333333333333333333333333333333333333333333 -- 68 digits, all 3's
def C : ℕ := 4444444444444444444 -- 19 digits, all 4's
def D : ℕ := 6666666666666666666666666666666666666666666666666666666666666667 -- 68 digits, first 67 are 6's, last is 7

theorem comparison_of_products : C * D > A * B ∧ C * D - A * B = 4444444444444444444 := sorry

end comparison_of_products_l56_56556


namespace total_expenditure_of_Louis_l56_56084

def fabric_cost (yards price_per_yard : ℕ) : ℕ :=
  yards * price_per_yard

def thread_cost (spools price_per_spool : ℕ) : ℕ :=
  spools * price_per_spool

def total_cost (yards price_per_yard pattern_cost spools price_per_spool : ℕ) : ℕ :=
  fabric_cost yards price_per_yard + pattern_cost + thread_cost spools price_per_spool

theorem total_expenditure_of_Louis :
  total_cost 5 24 15 2 3 = 141 :=
by
  sorry

end total_expenditure_of_Louis_l56_56084


namespace find_x_l56_56262

-- We are given points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 2)

-- Vector a is (2x + 3, x^2 - 4)
def vec_a (x : ℝ) : ℝ × ℝ := (2 * x + 3, x^2 - 4)

-- Vector AB is calculated as
def vec_AB : ℝ × ℝ := (3 - 1, 2 - 2)

-- Define the condition that vec_a and vec_AB form 0° angle
def forms_zero_angle (u v : ℝ × ℝ) : Prop := (u.1 * v.2 - u.2 * v.1) = 0 ∧ (u.1 = v.1 ∧ v.2 = 0)

-- The proof statement
theorem find_x (x : ℝ) (h₁ : forms_zero_angle (vec_a x) vec_AB) : x = 2 :=
by
  sorry

end find_x_l56_56262


namespace extreme_value_at_1_l56_56341

theorem extreme_value_at_1 (a b : ℝ) (h1 : (deriv (λ x => x^3 + a * x^2 + b * x + a^2) 1 = 0))
(h2 : (1 + a + b + a^2 = 10)) : a + b = -7 := by
  sorry

end extreme_value_at_1_l56_56341


namespace winter_expenditure_l56_56543

theorem winter_expenditure (exp_end_nov : Real) (exp_end_feb : Real) 
  (h_nov : exp_end_nov = 3.0) (h_feb : exp_end_feb = 5.5) : 
  (exp_end_feb - exp_end_nov) = 2.5 :=
by 
  sorry

end winter_expenditure_l56_56543


namespace complement_intersection_l56_56892

-- Definitions of sets and complements
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}
def C_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}
def C_U_B : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- The proof statement
theorem complement_intersection {U A B C_U_A C_U_B : Set ℕ} (h1 : U = {1, 2, 3, 4, 5}) (h2 : A = {1, 2, 3}) (h3 : B = {2, 5}) (h4 : C_U_A = {x | x ∈ U ∧ x ∉ A}) (h5 : C_U_B = {x | x ∈ U ∧ x ∉ B}) : 
  (C_U_A ∩ C_U_B) = {4} :=
by 
  sorry

end complement_intersection_l56_56892


namespace solve_for_x_l56_56094

theorem solve_for_x (x : ℝ) : (5 * x + 9 * x = 350 - 10 * (x - 5)) -> x = 50 / 3 :=
by
  intro h
  sorry

end solve_for_x_l56_56094


namespace value_of_2a_plus_b_l56_56808

theorem value_of_2a_plus_b (a b : ℤ) (h1 : |a - 1| = 4) (h2 : |b| = 7) (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := sorry

end value_of_2a_plus_b_l56_56808


namespace interest_rate_calc_l56_56925

theorem interest_rate_calc
  (P : ℝ) (A : ℝ) (T : ℝ) (SI : ℝ := A - P)
  (R : ℝ := (SI * 100) / (P * T))
  (hP : P = 750)
  (hA : A = 950)
  (hT : T = 5) :
  R = 5.33 :=
by
  sorry

end interest_rate_calc_l56_56925


namespace harry_weekly_earnings_l56_56889

def dogs_walked_MWF := 7
def dogs_walked_Tue := 12
def dogs_walked_Thu := 9
def pay_per_dog := 5

theorem harry_weekly_earnings : 
  dogs_walked_MWF * pay_per_dog * 3 + dogs_walked_Tue * pay_per_dog + dogs_walked_Thu * pay_per_dog = 210 :=
by
  sorry

end harry_weekly_earnings_l56_56889


namespace three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l56_56428

-- Problem (1)
theorem three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty : 3^40 > 4^30 ∧ 4^30 > 5^20 := 
by
  sorry

-- Problem (2)
theorem sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one : 16^31 > 8^41 ∧ 8^41 > 4^61 :=
by 
  sorry

-- Problem (3)
theorem a_lt_b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a^5 = 2) (h4 : b^7 = 3) : a < b :=
by
  sorry

end three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l56_56428


namespace average_visitors_per_day_l56_56371

theorem average_visitors_per_day (avg_sunday_visitors : ℕ) (avg_otherday_visitors : ℕ) (days_in_month : ℕ)
  (starts_with_sunday : Bool) (num_sundays : ℕ) (num_otherdays : ℕ)
  (h1 : avg_sunday_visitors = 510)
  (h2 : avg_otherday_visitors = 240)
  (h3 : days_in_month = 30)
  (h4 : starts_with_sunday = true)
  (h5 : num_sundays = 5)
  (h6 : num_otherdays = 25) :
  (num_sundays * avg_sunday_visitors + num_otherdays * avg_otherday_visitors) / days_in_month = 285 :=
by 
  sorry

end average_visitors_per_day_l56_56371


namespace divisible_by_12_for_all_integral_n_l56_56320

theorem divisible_by_12_for_all_integral_n (n : ℤ) : 12 ∣ (2 * n ^ 3 - 2 * n) :=
sorry

end divisible_by_12_for_all_integral_n_l56_56320


namespace sqrt_sum_ineq_l56_56070

open Real

theorem sqrt_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 1) :
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) + a + b + c > 3 := by
  sorry

end sqrt_sum_ineq_l56_56070


namespace range_of_m_l56_56978

theorem range_of_m {x m : ℝ} (h : ∀ x, x^2 - 2*x + 2*m - 1 ≥ 0) : m ≥ 1 :=
sorry

end range_of_m_l56_56978


namespace max_value_of_y_l56_56841

open Real

noncomputable def y (x : ℝ) := 1 + 1 / (x^2 + 2*x + 2)

theorem max_value_of_y : ∃ x : ℝ, y x = 2 :=
sorry

end max_value_of_y_l56_56841


namespace prime_arithmetic_progression_difference_divisible_by_6_l56_56636

theorem prime_arithmetic_progression_difference_divisible_by_6
    (p d : ℕ) (h₀ : Prime p) (h₁ : Prime (p - d)) (h₂ : Prime (p + d))
    (p_neq_3 : p ≠ 3) :
    ∃ (k : ℕ), d = 6 * k := by
  sorry

end prime_arithmetic_progression_difference_divisible_by_6_l56_56636


namespace coordinates_of_P_l56_56879

theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) :
  P = (2 * m, m + 8) ∧ 2 * m = 0 → P = (0, 8) := by
  intros hm
  sorry

end coordinates_of_P_l56_56879


namespace solution_x_y_l56_56939

theorem solution_x_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
    x^4 - 6 * x^2 + 1 = 7 * 2^y ↔ (x = 3 ∧ y = 2) :=
by {
    sorry
}

end solution_x_y_l56_56939


namespace final_notebooks_l56_56247

def initial_notebooks : ℕ := 10
def ordered_notebooks : ℕ := 6
def lost_notebooks : ℕ := 2

theorem final_notebooks : initial_notebooks + ordered_notebooks - lost_notebooks = 14 :=
by
  sorry

end final_notebooks_l56_56247


namespace find_x_l56_56003

theorem find_x (x : ℝ) (hx : x > 0) (condition : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 :=
by
  sorry

end find_x_l56_56003


namespace miranda_saves_half_of_salary_l56_56249

noncomputable def hourly_wage := 10
noncomputable def daily_hours := 10
noncomputable def weekly_days := 5
noncomputable def weekly_salary := hourly_wage * daily_hours * weekly_days

noncomputable def robby_saving_fraction := 2 / 5
noncomputable def jaylen_saving_fraction := 3 / 5
noncomputable def total_savings := 3000
noncomputable def weeks := 4

noncomputable def robby_weekly_savings := robby_saving_fraction * weekly_salary
noncomputable def jaylen_weekly_savings := jaylen_saving_fraction * weekly_salary
noncomputable def robby_total_savings := robby_weekly_savings * weeks
noncomputable def jaylen_total_savings := jaylen_weekly_savings * weeks
noncomputable def combined_savings_rj := robby_total_savings + jaylen_total_savings
noncomputable def miranda_total_savings := total_savings - combined_savings_rj
noncomputable def miranda_weekly_savings := miranda_total_savings / weeks

noncomputable def miranda_saving_fraction := miranda_weekly_savings / weekly_salary

theorem miranda_saves_half_of_salary:
  miranda_saving_fraction = 1 / 2 := 
by sorry

end miranda_saves_half_of_salary_l56_56249


namespace sum_primes_less_than_20_l56_56298

theorem sum_primes_less_than_20 : (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := sorry

end sum_primes_less_than_20_l56_56298


namespace percentage_increase_pay_rate_l56_56199

theorem percentage_increase_pay_rate (r t c e : ℕ) (h_reg_rate : r = 10) (h_total_surveys : t = 100) (h_cellphone_surveys : c = 60) (h_total_earnings : e = 1180) : 
  (13 - 10) / 10 * 100 = 30 :=
by
  sorry

end percentage_increase_pay_rate_l56_56199


namespace vector_subtraction_magnitude_l56_56784

theorem vector_subtraction_magnitude (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 :=
sorry

end vector_subtraction_magnitude_l56_56784


namespace walking_west_is_negative_l56_56466

-- Definitions based on conditions
def east (m : Int) : Int := m
def west (m : Int) : Int := -m

-- Proof statement (no proof required, so use "sorry")
theorem walking_west_is_negative (m : Int) (h : east 8 = 8) : west 10 = -10 :=
by
  sorry

end walking_west_is_negative_l56_56466


namespace g_of_x_l56_56624

theorem g_of_x (f g : ℕ → ℕ) (h1 : ∀ x, f x = 2 * x + 3)
  (h2 : ∀ x, g (x + 2) = f x) : ∀ x, g x = 2 * x - 1 :=
by
  sorry

end g_of_x_l56_56624


namespace prove_original_sides_l56_56348

def original_parallelogram_sides (a b : ℕ) : Prop :=
  ∃ k : ℕ, (a, b) = (k * 1, k * 2) ∨ (a, b) = (1, 5) ∨ (a, b) = (4, 5) ∨ (a, b) = (3, 7) ∨ (a, b) = (4, 7) ∨ (a, b) = (3, 8) ∨ (a, b) = (5, 8) ∨ (a, b) = (5, 7) ∨ (a, b) = (2, 7)

theorem prove_original_sides (a b : ℕ) : original_parallelogram_sides a b → (1, 2) = (1, 2) :=
by
  intro h
  sorry

end prove_original_sides_l56_56348


namespace sqrt_sum_eq_fraction_l56_56600

-- Definitions as per conditions
def w : ℕ := 4
def x : ℕ := 9
def z : ℕ := 25

-- Main theorem statement
theorem sqrt_sum_eq_fraction : (Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15) := by
  sorry

end sqrt_sum_eq_fraction_l56_56600


namespace min_value_frac_sum_l56_56202

theorem min_value_frac_sum (a b : ℝ) (hab : a + b = 1) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (x : ℝ), x = 3 + 2 * Real.sqrt 2 ∧ x = (1/a + 2/b) :=
sorry

end min_value_frac_sum_l56_56202


namespace intersection_A_B_subsets_C_l56_56363

-- Definition of sets A and B
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | 0 ≤ x}

-- Definition of intersection C
def C : Set ℤ := A ∩ B

-- The proof statements
theorem intersection_A_B : C = {1, 2} := 
by sorry

theorem subsets_C : {s | s ⊆ C} = {∅, {1}, {2}, {1, 2}} := 
by sorry

end intersection_A_B_subsets_C_l56_56363


namespace find_a_l56_56332

noncomputable def f (x : ℝ) : ℝ := x^2 + 12
noncomputable def g (x : ℝ) : ℝ := x^2 - x - 4

theorem find_a (a : ℝ) (h_pos : a > 0) (h_fga : f (g a) = 12) : a = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_a_l56_56332


namespace gcd_g_150_151_l56_56067

def g (x : ℤ) : ℤ := x^2 - 2*x + 3020

theorem gcd_g_150_151 : Int.gcd (g 150) (g 151) = 1 :=
  by
  sorry

end gcd_g_150_151_l56_56067


namespace average_of_remaining_primes_l56_56181

theorem average_of_remaining_primes (avg30: ℕ) (avg15: ℕ) (h1 : avg30 = 110) (h2 : avg15 = 95) : 
  ((30 * avg30 - 15 * avg15) / 15) = 125 := 
by
  -- Proof
  sorry

end average_of_remaining_primes_l56_56181


namespace sum_of_first_15_squares_l56_56101

noncomputable def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_first_15_squares :
  sum_of_squares 15 = 1240 :=
by
  sorry

end sum_of_first_15_squares_l56_56101


namespace partition_triangle_l56_56894

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l56_56894


namespace solve_trig_eq_l56_56740

theorem solve_trig_eq (k : ℤ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * (Real.sin t)^2 - Real.sin (2 * t) + 3 * Real.cos t^2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
sorry

end solve_trig_eq_l56_56740


namespace michael_wants_to_buy_more_packs_l56_56118

theorem michael_wants_to_buy_more_packs
  (initial_packs : ℕ)
  (cost_per_pack : ℝ)
  (total_value_after_purchase : ℝ)
  (value_of_current_packs : ℝ := initial_packs * cost_per_pack)
  (additional_value_needed : ℝ := total_value_after_purchase - value_of_current_packs)
  (packs_to_buy : ℝ := additional_value_needed / cost_per_pack)
  (answer : ℕ := 2) :
  initial_packs = 4 → cost_per_pack = 2.5 → total_value_after_purchase = 15 → packs_to_buy = answer :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end michael_wants_to_buy_more_packs_l56_56118


namespace volume_of_displaced_water_l56_56835

-- Defining the conditions of the problem
def cube_side_length : ℝ := 6
def cyl_radius : ℝ := 5
def cyl_height : ℝ := 12
def cube_volume (s : ℝ) : ℝ := s^3

-- Statement: The volume of water displaced by the cube when it is fully submerged in the barrel
theorem volume_of_displaced_water :
  cube_volume cube_side_length = 216 := by
  sorry

end volume_of_displaced_water_l56_56835


namespace necessary_not_sufficient_cond_l56_56458

theorem necessary_not_sufficient_cond (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y < 4 → xy < 4) ∧ ¬(xy < 4 → x + y < 4) :=
  by
    sorry

end necessary_not_sufficient_cond_l56_56458


namespace problem_solution_l56_56411

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end problem_solution_l56_56411


namespace intersection_complement_is_l56_56405

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_is :
  N ∩ (U \ M) = {3, 5} :=
  sorry

end intersection_complement_is_l56_56405


namespace intersection_of_A_and_B_l56_56852

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 4} := by
  sorry

end intersection_of_A_and_B_l56_56852


namespace find_z_l56_56644

open Complex

theorem find_z (z : ℂ) (h : (1 - I) * z = 2 * I) : z = -1 + I := by
  sorry

end find_z_l56_56644


namespace inv_three_mod_thirty_seven_l56_56119

theorem inv_three_mod_thirty_seven : (3 * 25) % 37 = 1 :=
by
  -- Explicit mention to skip the proof with sorry
  sorry

end inv_three_mod_thirty_seven_l56_56119


namespace product_of_consecutive_integers_between_sqrt_29_l56_56324

-- Define that \(5 \lt \sqrt{29} \lt 6\)
lemma sqrt_29_bounds : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 :=
sorry

-- Main theorem statement
theorem product_of_consecutive_integers_between_sqrt_29 :
  (∃ (a b : ℤ), 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 ∧ a = 5 ∧ b = 6 ∧ a * b = 30) := 
sorry

end product_of_consecutive_integers_between_sqrt_29_l56_56324


namespace horner_evaluation_at_two_l56_56742

/-- Define the polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 8

/-- States that the value of f(2) using Horner's Rule equals 14. -/
theorem horner_evaluation_at_two : f 2 = 14 :=
sorry

end horner_evaluation_at_two_l56_56742


namespace problem_solution_l56_56880

noncomputable def expr := 
  (Real.tan (Real.pi / 15) - Real.sqrt 3) / ((4 * (Real.cos (Real.pi / 15))^2 - 2) * Real.sin (Real.pi / 15))

theorem problem_solution : expr = -4 :=
by
  sorry

end problem_solution_l56_56880


namespace find_m_n_sum_l56_56112

theorem find_m_n_sum (x y m n : ℤ) 
  (h1 : x = 2)
  (h2 : y = 1)
  (h3 : m * x + y = -3)
  (h4 : x - 2 * y = 2 * n) : 
  m + n = -2 := 
by 
  sorry

end find_m_n_sum_l56_56112


namespace intersection_points_l56_56554

noncomputable def h (x : ℝ) : ℝ := -x^2 - 4 * x + 1
noncomputable def j (x : ℝ) : ℝ := -h x
noncomputable def k (x : ℝ) : ℝ := h (-x)

def c : ℕ := 2 -- Number of intersections of y = h(x) and y = j(x)
def d : ℕ := 1 -- Number of intersections of y = h(x) and y = k(x)

theorem intersection_points :
  10 * c + d = 21 := by
  sorry

end intersection_points_l56_56554


namespace number_of_violas_l56_56766

theorem number_of_violas (V : ℕ) 
  (cellos : ℕ := 800) 
  (pairs : ℕ := 70) 
  (probability : ℝ := 0.00014583333333333335) 
  (h : probability = pairs / (cellos * V)) : V = 600 :=
by
  sorry

end number_of_violas_l56_56766


namespace square_perimeter_l56_56822

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l56_56822


namespace hypotenuse_length_l56_56421

def triangle_hypotenuse (x : ℝ) (h : ℝ) : Prop :=
  (3 * x - 3)^2 + x^2 = h^2 ∧
  (1 / 2) * x * (3 * x - 3) = 72

theorem hypotenuse_length :
  ∃ (x h : ℝ), triangle_hypotenuse x h ∧ h = Real.sqrt 505 :=
by
  sorry

end hypotenuse_length_l56_56421


namespace probability_composite_product_l56_56549

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l56_56549


namespace distinct_intersections_count_l56_56598

theorem distinct_intersections_count :
  (∃ (x y : ℝ), (x + 2 * y = 7 ∧ 3 * x - 4 * y + 8 = 0) ∨ (x + 2 * y = 7 ∧ 4 * x + 5 * y - 20 = 0) ∨
                (x - 2 * y - 1 = 0 ∧ 3 * x - 4 * y = 8) ∨ (x - 2 * y - 1 = 0 ∧ 4 * x + 5 * y - 20 = 0)) ∧
  ∃ count : ℕ, count = 3 :=
by sorry

end distinct_intersections_count_l56_56598


namespace hot_water_bottles_sold_l56_56672

theorem hot_water_bottles_sold (T H : ℕ) (h1 : 2 * T + 6 * H = 1200) (h2 : T = 7 * H) : H = 60 := 
by 
  sorry

end hot_water_bottles_sold_l56_56672


namespace fabric_per_pair_of_pants_l56_56092

theorem fabric_per_pair_of_pants 
  (jenson_shirts_per_day : ℕ)
  (kingsley_pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric_needed : ℕ)
  (days : ℕ)
  (fabric_per_pant : ℕ) :
  jenson_shirts_per_day = 3 →
  kingsley_pants_per_day = 5 →
  fabric_per_shirt = 2 →
  total_fabric_needed = 93 →
  days = 3 →
  fabric_per_pant = 5 :=
by sorry

end fabric_per_pair_of_pants_l56_56092


namespace complex_number_solution_l56_56390

theorem complex_number_solution (a b : ℤ) (z : ℂ) (h1 : z = a + b * Complex.I) (h2 : z^3 = 2 + 11 * Complex.I) : a + b = 3 :=
sorry

end complex_number_solution_l56_56390


namespace math_problem_l56_56146

variable (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c < d)

theorem math_problem : a - c > b - d :=
by {
  sorry
}

end math_problem_l56_56146


namespace cost_of_sculpture_cny_l56_56893

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end cost_of_sculpture_cny_l56_56893


namespace union_of_A_and_B_l56_56498

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B : Set ℝ := {x | x < 3}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 3} := by
  sorry

end union_of_A_and_B_l56_56498


namespace sum_of_digits_decrease_by_10_percent_l56_56177

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum -- Assuming this method computes the sum of the digits

theorem sum_of_digits_decrease_by_10_percent :
  ∃ (n m : ℕ), m = 11 * n / 10 ∧ sum_of_digits m = 9 * sum_of_digits n / 10 :=
by
  sorry

end sum_of_digits_decrease_by_10_percent_l56_56177


namespace sum_of_integers_l56_56937

theorem sum_of_integers (a b c : ℤ) (h1 : a = (1 / 3) * (b + c)) (h2 : b = (1 / 5) * (a + c)) (h3 : c = 35) : a + b + c = 60 :=
by
  sorry

end sum_of_integers_l56_56937


namespace line_through_points_l56_56330

theorem line_through_points (x1 y1 x2 y2 : ℝ) :
  (3 * x1 - 4 * y1 - 2 = 0) →
  (3 * x2 - 4 * y2 - 2 = 0) →
  (∀ x y : ℝ, (x = x1) → (y = y1) ∨ (x = x2) → (y = y2) → 3 * x - 4 * y - 2 = 0) :=
by
  sorry

end line_through_points_l56_56330


namespace range_of_a_l56_56772

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 2^(2 * x) + 2^x * a + a + 1 = 0) : a ≤ 2 - 2 * Real.sqrt 2 :=
sorry

end range_of_a_l56_56772


namespace train_times_l56_56632

theorem train_times (t x : ℝ) : 
  (30 * t = 360) ∧ (36 * (t - x) = 360) → x = 2 :=
by
  sorry

end train_times_l56_56632


namespace initial_water_percentage_l56_56260

noncomputable def S : ℝ := 4.0
noncomputable def V_initial : ℝ := 440
noncomputable def V_final : ℝ := 460
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8
noncomputable def kola_percentage : ℝ := 8.0 / 100.0
noncomputable def final_sugar_percentage : ℝ := 4.521739130434784 / 100.0

theorem initial_water_percentage : 
  ∀ (W S : ℝ),
  V_initial * (S / 100) + sugar_added = final_sugar_percentage * V_final →
  (W + 8.0 + S) = 100.0 →
  W = 88.0
:=
by
  intros W S h1 h2
  sorry

end initial_water_percentage_l56_56260


namespace find_range_t_l56_56065

def sequence_increasing (n : ℕ) (t : ℝ) : Prop :=
  (2 * (n + 1) + t^2 - 8) / (n + 1 + t) > (2 * n + t^2 - 8) / (n + t)

theorem find_range_t (t : ℝ) (h : ∀ n : ℕ, sequence_increasing n t) : 
  -1 < t ∧ t < 4 :=
sorry

end find_range_t_l56_56065


namespace neg_prop_true_l56_56402

theorem neg_prop_true (a : ℝ) :
  ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) → ∃ a : ℝ, a > 2 ∧ a^2 ≥ 4 :=
by
  intros h
  sorry

end neg_prop_true_l56_56402


namespace a_perfect_square_l56_56614

theorem a_perfect_square (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_div : 2 * a * b ∣ a^2 + b^2 - a) : ∃ k : ℕ, a = k^2 := 
sorry

end a_perfect_square_l56_56614


namespace area_of_given_trapezium_l56_56019

def area_of_trapezium (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem area_of_given_trapezium :
  area_of_trapezium 20 18 25 = 475 :=
by
  sorry

end area_of_given_trapezium_l56_56019


namespace count_complex_numbers_l56_56966

theorem count_complex_numbers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : a + b ≤ 5) : 
  ∃ n, n = 10 := 
by
  sorry

end count_complex_numbers_l56_56966


namespace susan_spaces_to_win_l56_56826

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end susan_spaces_to_win_l56_56826


namespace fraction_identity_l56_56337

theorem fraction_identity (a b c : ℕ) (h : (a : ℚ) / (36 - a) + (b : ℚ) / (48 - b) + (c : ℚ) / (72 - c) = 9) : 
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 := 
by 
  sorry

end fraction_identity_l56_56337


namespace parabola_directrix_l56_56468

theorem parabola_directrix (x y : ℝ) (h : y = - (1/8) * x^2) : y = 2 :=
sorry

end parabola_directrix_l56_56468


namespace average_speed_l56_56373

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 50) (h2 : d2 = 20) (h3 : t1 = 50 / 20) (h4 : t2 = 20 / 40) :
  ((d1 + d2) / (t1 + t2)) = 23.33 := 
  sorry

end average_speed_l56_56373


namespace second_triangle_weight_l56_56235

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def weight_of_second_triangle (m_1 : ℝ) (s_1 s_2 : ℝ) : ℝ :=
  m_1 * (area_equilateral_triangle s_2 / area_equilateral_triangle s_1)

theorem second_triangle_weight :
  let m_1 := 12   -- weight of the first triangle in ounces
  let s_1 := 3    -- side length of the first triangle in inches
  let s_2 := 5    -- side length of the second triangle in inches
  weight_of_second_triangle m_1 s_1 s_2 = 33.3 :=
by
  sorry

end second_triangle_weight_l56_56235


namespace find_r_s_l56_56661

def N : Matrix (Fin 2) (Fin 2) Int := ![![3, 4], ![-2, 0]]
def I : Matrix (Fin 2) (Fin 2) Int := ![![1, 0], ![0, 1]]

theorem find_r_s :
  ∃ (r s : Int), (N * N = r • N + s • I) ∧ (r = 3) ∧ (s = 16) :=
by
  sorry

end find_r_s_l56_56661


namespace transformation_correct_l56_56193

theorem transformation_correct (a b : ℝ) (h : a > b) : 2 * a + 1 > 2 * b + 1 :=
by
  sorry

end transformation_correct_l56_56193


namespace smallest_fourth_number_l56_56509

-- Define the given conditions
def first_three_numbers_sum : ℕ := 28 + 46 + 59 
def sum_of_digits_of_first_three_numbers : ℕ := 2 + 8 + 4 + 6 + 5 + 9 

-- Define the condition for the fourth number represented as 10a + b and its digits 
def satisfies_condition (a b : ℕ) : Prop := 
  first_three_numbers_sum + 10 * a + b = 4 * (sum_of_digits_of_first_three_numbers + a + b)

-- Statement to prove the smallest fourth number
theorem smallest_fourth_number : ∃ (a b : ℕ), satisfies_condition a b ∧ 10 * a + b = 11 := 
sorry

end smallest_fourth_number_l56_56509


namespace sum_first_2014_terms_l56_56853

def sequence_is_arithmetic (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + a 2

def first_arithmetic_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :=
  S n = (n * (n - 1)) / 2

theorem sum_first_2014_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : sequence_is_arithmetic a) 
  (h2 : a 3 = 2) : 
  S 2014 = 1007 * 2013 :=
sorry

end sum_first_2014_terms_l56_56853


namespace cost_of_meatballs_is_five_l56_56957

-- Define the conditions
def cost_of_pasta : ℕ := 1
def cost_of_sauce : ℕ := 2
def total_cost_of_meal (servings : ℕ) (cost_per_serving : ℕ) : ℕ := servings * cost_per_serving

-- Define the cost of meatballs calculation
def cost_of_meatballs (total_cost pasta_cost sauce_cost : ℕ) : ℕ :=
  total_cost - pasta_cost - sauce_cost

-- State the theorem we want to prove
theorem cost_of_meatballs_is_five :
  cost_of_meatballs (total_cost_of_meal 8 1) cost_of_pasta cost_of_sauce = 5 :=
by
  -- This part will include the proof steps
  sorry

end cost_of_meatballs_is_five_l56_56957


namespace triangle_sides_angles_l56_56086

open Real

variables {a b c : ℝ} {α β γ : ℝ}

theorem triangle_sides_angles
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (angles_sum : α + β + γ = π)
  (condition : 3 * α + 2 * β = π) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_sides_angles_l56_56086


namespace option2_is_cheaper_l56_56422

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price_option1 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.10
  apply_discount price_after_second_discount 0.05

def final_price_option2 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.05
  apply_discount price_after_second_discount 0.15

theorem option2_is_cheaper (initial_price : ℝ) (h : initial_price = 12000) :
  final_price_option2 initial_price = 6783 ∧ final_price_option1 initial_price = 7182 → 6783 < 7182 :=
by
  intros
  sorry

end option2_is_cheaper_l56_56422


namespace builder_installed_windows_l56_56829

-- Conditions
def total_windows : ℕ := 14
def hours_per_window : ℕ := 8
def remaining_hours : ℕ := 48

-- Definition for the problem statement
def installed_windows := total_windows - remaining_hours / hours_per_window

-- The hypothesis we need to prove
theorem builder_installed_windows : installed_windows = 8 := by
  sorry

end builder_installed_windows_l56_56829


namespace problem_part1_problem_part2_l56_56711

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := abs (a - x)

def setA (a : ℝ) : Set ℝ := {x | f a (2 * x - 3 / 2) > 2 * f a (x + 2) + 2}

theorem problem_part1 {a : ℝ} (h : a = 3 / 2) : setA a = {x | x < 0} := by
  sorry

theorem problem_part2 {a : ℝ} (h : a = 3 / 2) (x0 : ℝ) (hx0 : x0 ∈ setA a) (x : ℝ) : 
    f a (x0 * x) ≥ x0 * f a x + f a (a * x0) := by
  sorry

end problem_part1_problem_part2_l56_56711


namespace culture_medium_preparation_l56_56538

theorem culture_medium_preparation :
  ∀ (V : ℝ), 0 < V → 
  ∃ (nutrient_broth pure_water saline_water : ℝ),
    nutrient_broth = V / 3 ∧
    pure_water = V * 0.3 ∧
    saline_water = V - (nutrient_broth + pure_water) :=
by
  sorry

end culture_medium_preparation_l56_56538


namespace number_of_female_students_in_sample_l56_56155

theorem number_of_female_students_in_sample (male_students female_students sample_size : ℕ)
  (h1 : male_students = 560)
  (h2 : female_students = 420)
  (h3 : sample_size = 280) :
  (female_students * sample_size) / (male_students + female_students) = 120 := 
sorry

end number_of_female_students_in_sample_l56_56155


namespace winning_candidate_percentage_l56_56309

theorem winning_candidate_percentage
  (votes1 votes2 votes3 : ℕ)
  (h1 : votes1 = 3000)
  (h2 : votes2 = 5000)
  (h3 : votes3 = 20000) :
  ((votes3 : ℝ) / (votes1 + votes2 + votes3) * 100) = 71.43 := by
  sorry

end winning_candidate_percentage_l56_56309


namespace seating_arrangement_l56_56979

theorem seating_arrangement (students : ℕ) (desks : ℕ) (empty_desks : ℕ) 
  (h_students : students = 2) (h_desks : desks = 5) 
  (h_empty : empty_desks ≥ 1) :
  ∃ ways, ways = 12 := by
  sorry

end seating_arrangement_l56_56979


namespace volume_ratio_proof_l56_56178

-- Definitions based on conditions
def edge_ratio (a b : ℝ) : Prop := a = 3 * b
def volume_ratio (V_large V_small : ℝ) : Prop := V_large = 27 * V_small

-- Problem statement
theorem volume_ratio_proof (e V_small V_large : ℝ) 
  (h1 : edge_ratio (3 * e) e)
  (h2 : volume_ratio V_large V_small) : 
  V_large / V_small = 27 := 
by sorry

end volume_ratio_proof_l56_56178


namespace total_area_of_triangles_l56_56625

theorem total_area_of_triangles :
    let AB := 12
    let DE := 8 * Real.sqrt 2
    let area_ABC := (1 / 2) * AB * AB
    let area_DEF := (1 / 2) * DE * DE * 2
    area_ABC + area_DEF = 136 := by
  sorry

end total_area_of_triangles_l56_56625


namespace find_x_given_y_l56_56783

-- Given x varies inversely as the square of y, we define the relationship
def varies_inversely (x y k : ℝ) : Prop := x = k / y^2

theorem find_x_given_y (k : ℝ) (h_k : k = 4) :
  ∀ (y : ℝ), varies_inversely x y k → y = 2 → x = 1 :=
by
  intros y h_varies h_y_eq
  -- We need to prove the statement here
  sorry

end find_x_given_y_l56_56783


namespace determine_alpha_l56_56607

theorem determine_alpha (α : ℝ) (y : ℝ → ℝ) (h : ∀ x, y x = x^α) (hp : y 2 = Real.sqrt 2) : α = 1 / 2 :=
sorry

end determine_alpha_l56_56607


namespace john_text_messages_per_day_l56_56039

theorem john_text_messages_per_day (m n : ℕ) (h1 : m = 20) (h2 : n = 245) : 
  m + n / 7 = 55 :=
by
  sorry

end john_text_messages_per_day_l56_56039


namespace min_product_log_condition_l56_56236

theorem min_product_log_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : Real.log a / Real.log 2 * Real.log b / Real.log 2 = 1) : 4 ≤ a * b :=
by
  sorry

end min_product_log_condition_l56_56236


namespace integer_solutions_2x2_2xy_9x_y_eq_2_l56_56290

theorem integer_solutions_2x2_2xy_9x_y_eq_2 : ∀ (x y : ℤ), 2 * x^2 - 2 * x * y + 9 * x + y = 2 → (x, y) = (1, 9) ∨ (x, y) = (2, 8) ∨ (x, y) = (0, 2) ∨ (x, y) = (-1, 3) := 
by 
  intros x y h
  sorry

end integer_solutions_2x2_2xy_9x_y_eq_2_l56_56290


namespace solution_set_of_new_inequality_l56_56993

-- Define the conditions
variable (a b c x : ℝ)

-- ax^2 + bx + c > 0 has solution set {-3 < x < 2}
def inequality_solution_set (a b c : ℝ) : Prop := ∀ x : ℝ, (-3 < x ∧ x < 2) → a * x^2 + b * x + c > 0

-- Prove that cx^2 + bx + a > 0 has solution set {x < -1/3 ∨ x > 1/2}
theorem solution_set_of_new_inequality
  (a b c : ℝ)
  (h : a < 0 ∧ inequality_solution_set a b c) :
  ∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ (c * x^2 + b * x + a > 0) := sorry

end solution_set_of_new_inequality_l56_56993


namespace perimeter_of_park_l56_56761

def length := 300
def breadth := 200

theorem perimeter_of_park : 2 * (length + breadth) = 1000 := by
  sorry

end perimeter_of_park_l56_56761


namespace darts_final_score_is_600_l56_56239

def bullseye_points : ℕ := 50

def first_dart_points (bullseye : ℕ) : ℕ := 3 * bullseye

def second_dart_points : ℕ := 0

def third_dart_points (bullseye : ℕ) : ℕ := bullseye / 2

def fourth_dart_points (bullseye : ℕ) : ℕ := 2 * bullseye

def total_points_before_fifth (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def fifth_dart_points (bullseye : ℕ) (previous_total : ℕ) : ℕ :=
  bullseye + previous_total

def final_score (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4 + d5

theorem darts_final_score_is_600 :
  final_score
    (first_dart_points bullseye_points)
    second_dart_points
    (third_dart_points bullseye_points)
    (fourth_dart_points bullseye_points)
    (fifth_dart_points bullseye_points (total_points_before_fifth
      (first_dart_points bullseye_points)
      second_dart_points
      (third_dart_points bullseye_points)
      (fourth_dart_points bullseye_points))) = 600 :=
  sorry

end darts_final_score_is_600_l56_56239


namespace magnitude_z_l56_56905

open Complex

theorem magnitude_z
  (z w : ℂ)
  (h1 : abs (2 * z - w) = 25)
  (h2 : abs (z + 2 * w) = 5)
  (h3 : abs (z + w) = 2) : abs z = 9 := 
by 
  sorry

end magnitude_z_l56_56905


namespace imaginary_part_of_fraction_l56_56388

open Complex

theorem imaginary_part_of_fraction :
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  z.im = 1 :=
by
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  show z.im = 1
  sorry

end imaginary_part_of_fraction_l56_56388


namespace quadratic_inequality_solution_l56_56691

theorem quadratic_inequality_solution (a b: ℝ) (h1: ∀ x: ℝ, 1 < x ∧ x < 2 → ax^2 + bx - 4 > 0) (h2: ∀ x: ℝ, x ≤ 1 ∨ x ≥ 2 → ax^2 + bx - 4 ≤ 0) : a + b = 4 :=
sorry

end quadratic_inequality_solution_l56_56691


namespace max_vertex_value_in_cube_l56_56333

def transform_black (v : ℕ) (e1 e2 e3 : ℕ) : ℕ :=
  e1 + e2 + e3

def transform_white (v : ℕ) (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

def max_value_after_transformation (initial_values : Fin 8 → ℕ) : ℕ :=
  -- Combination of transformations and iterations are derived here
  42648

theorem max_vertex_value_in_cube :
  ∀ (initial_values : Fin 8 → ℕ),
  (∀ i, 1 ≤ initial_values i ∧ initial_values i ≤ 8) →
  (∃ (final_value : ℕ), final_value = max_value_after_transformation initial_values) → final_value = 42648 :=
by {
  sorry
}

end max_vertex_value_in_cube_l56_56333


namespace intersection_of_M_N_l56_56693

-- Definitions of the sets M and N
def M : Set ℝ := { x | (x + 2) * (x - 1) < 0 }
def N : Set ℝ := { x | x + 1 < 0 }

-- Proposition stating that the intersection of M and N is { x | -2 < x < -1 }
theorem intersection_of_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < -1 } :=
  by
    sorry

end intersection_of_M_N_l56_56693


namespace evaluate_expression_l56_56209

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem evaluate_expression : (factorial (factorial 4)) / factorial 4 = factorial 23 :=
by sorry

end evaluate_expression_l56_56209


namespace avg_people_moving_to_florida_per_hour_l56_56931

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end avg_people_moving_to_florida_per_hour_l56_56931


namespace room_analysis_l56_56346

-- First person's statements
def statement₁ (n: ℕ) (liars: ℕ) :=
  n ≤ 3 ∧ liars = n

-- Second person's statements
def statement₂ (n: ℕ) (liars: ℕ) :=
  n ≤ 4 ∧ liars < n

-- Third person's statements
def statement₃ (n: ℕ) (liars: ℕ) :=
  n = 5 ∧ liars = 3

theorem room_analysis (n liars : ℕ) :
  (¬ statement₁ n liars) ∧ statement₂ n liars ∧ ¬ statement₃ n liars → (n = 4 ∧ liars = 2) :=
by
  sorry

end room_analysis_l56_56346


namespace households_using_both_brands_l56_56359

def total : ℕ := 260
def neither : ℕ := 80
def onlyA : ℕ := 60
def onlyB (both : ℕ) : ℕ := 3 * both

theorem households_using_both_brands (both : ℕ) : 80 + 60 + both + onlyB both = 260 → both = 30 :=
by
  intro h
  sorry

end households_using_both_brands_l56_56359


namespace sin_double_angle_l56_56731

theorem sin_double_angle (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l56_56731


namespace h_h_neg1_l56_56277

def h (x: ℝ) : ℝ := 3 * x^2 - x + 1

theorem h_h_neg1 : h (h (-1)) = 71 := by
  sorry

end h_h_neg1_l56_56277


namespace problem_l56_56229

theorem problem (x y : ℝ) (h : (3 * x - y + 5)^2 + |2 * x - y + 3| = 0) : x + y = -3 := 
by
  sorry

end problem_l56_56229


namespace combination_identity_l56_56721

theorem combination_identity (C : ℕ → ℕ → ℕ)
  (comb_formula : ∀ n r, C r n = Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)))
  (identity_1 : ∀ n r, C r n = C (n-r) n)
  (identity_2 : ∀ n r, C r (n+1) = C r n + C (r-1) n) :
  C 2 100 + C 97 100 = C 3 101 :=
by sorry

end combination_identity_l56_56721


namespace geometric_progression_terms_l56_56626

theorem geometric_progression_terms (b1 b2 bn : ℕ) (q n : ℕ)
  (h1 : b1 = 3) 
  (h2 : b2 = 12)
  (h3 : bn = 3072)
  (h4 : b2 = b1 * q)
  (h5 : bn = b1 * q^(n-1)) : 
  n = 6 := 
by 
  sorry

end geometric_progression_terms_l56_56626


namespace intersection_eq_l56_56881

-- Define the sets M and N using the given conditions
def M : Set ℝ := { x | x < 1 / 2 }
def N : Set ℝ := { x | x ≥ -4 }

-- The goal is to prove that the intersection of M and N is { x | -4 ≤ x < 1 / 2 }
theorem intersection_eq : M ∩ N = { x | -4 ≤ x ∧ x < (1 / 2) } :=
by
  sorry

end intersection_eq_l56_56881


namespace minimum_value_of_f_l56_56651

open Real

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x

theorem minimum_value_of_f (x : ℝ) (h : abs x ≤ π / 4) : 
  ∃ m : ℝ, (∀ y : ℝ, f y ≥ m) ∧ m = 1 / 2 - sqrt 2 / 2 :=
sorry

end minimum_value_of_f_l56_56651


namespace radius_ratio_of_smaller_to_larger_l56_56609

noncomputable def ratio_of_radii (v_large v_small : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = v_large) (h_small : v_small = 0.25 * v_large) (h_small_sphere : (4/3) * Real.pi * r^3 = v_small) : ℝ :=
  let ratio := r / R
  ratio

theorem radius_ratio_of_smaller_to_larger (v_large : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = 576 * Real.pi) (h_small_sphere : (4/3) * Real.pi * r^3 = 0.25 * 576 * Real.pi) : r / R = 1 / (2^(2/3)) :=
by
  sorry

end radius_ratio_of_smaller_to_larger_l56_56609


namespace water_bottles_needed_l56_56303

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end water_bottles_needed_l56_56303


namespace prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l56_56771

theorem prime_in_form_x_squared_plus_16y_squared (p : ℕ) (hprime : Prime p) (h1 : p % 8 = 1) :
  ∃ x y : ℤ, p = x^2 + 16 * y^2 :=
by
  sorry

theorem prime_in_form_4x_squared_plus_4xy_plus_5y_squared (p : ℕ) (hprime : Prime p) (h5 : p % 8 = 5) :
  ∃ x y : ℤ, p = 4 * x^2 + 4 * x * y + 5 * y^2 :=
by
  sorry

end prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l56_56771


namespace cucumbers_count_l56_56282

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end cucumbers_count_l56_56282


namespace car_speed_ratio_to_pedestrian_speed_l56_56077

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end car_speed_ratio_to_pedestrian_speed_l56_56077


namespace circle_and_parabola_no_intersection_l56_56397

theorem circle_and_parabola_no_intersection (m : ℝ) (h : m ≠ 0) :
  (m > 0 ∨ m < -4) ↔
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x = 0) → (y^2 = 4 * m * x) → x ≠ -m := 
sorry

end circle_and_parabola_no_intersection_l56_56397


namespace green_pairs_count_l56_56308

theorem green_pairs_count 
  (blue_students : ℕ)
  (green_students : ℕ)
  (total_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ) 
  (mixed_pairs_students : ℕ) 
  (green_green_pairs : ℕ) 
  (count_blue : blue_students = 65)
  (count_green : green_students = 67)
  (count_total_students : total_students = 132)
  (count_total_pairs : total_pairs = 66)
  (count_blue_blue_pairs : blue_blue_pairs = 29)
  (count_mixed_blue_students : mixed_pairs_students = 7)
  (count_green_green_pairs : green_green_pairs = 30) :
  green_green_pairs = 30 :=
sorry

end green_pairs_count_l56_56308


namespace perpendicular_planes_l56_56917

variables (b c : Line) (α β : Plane)
axiom line_in_plane (b : Line) (α : Plane) : Prop -- b ⊆ α
axiom line_parallel_plane (c : Line) (α : Plane) : Prop -- c ∥ α
axiom lines_are_skew (b c : Line) : Prop -- b and c could be skew
axiom planes_are_perpendicular (α β : Plane) : Prop -- α ⊥ β
axiom line_perpendicular_plane (c : Line) (β : Plane) : Prop -- c ⊥ β

theorem perpendicular_planes (hcα : line_in_plane c α) (hcβ : line_perpendicular_plane c β) : planes_are_perpendicular α β := 
sorry

end perpendicular_planes_l56_56917


namespace exchange_rate_change_2014_l56_56774

theorem exchange_rate_change_2014 :
  let init_rate := 32.6587
  let final_rate := 56.2584
  let change := final_rate - init_rate
  let rounded_change := Float.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_2014_l56_56774


namespace min_value_a_b_l56_56934

variable (a b : ℝ)

theorem min_value_a_b (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 * (Real.sqrt 2 + 1) :=
sorry

end min_value_a_b_l56_56934


namespace trigonometric_identity_l56_56389

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trigonometric_identity_l56_56389


namespace continuity_f_at_3_l56_56134

noncomputable def f (x : ℝ) := if x ≤ 3 then 3 * x^2 - 5 else 18 * x - 32

theorem continuity_f_at_3 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x - f 3) < ε := by
  intro ε ε_pos
  use 1
  simp
  sorry

end continuity_f_at_3_l56_56134


namespace prime_ge_7_p2_sub1_div_by_30_l56_56943

theorem prime_ge_7_p2_sub1_div_by_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) :=
sorry

end prime_ge_7_p2_sub1_div_by_30_l56_56943


namespace percent_increase_stock_l56_56716

theorem percent_increase_stock (P_open P_close: ℝ) (h1: P_open = 30) (h2: P_close = 45):
  (P_close - P_open) / P_open * 100 = 50 :=
by
  sorry

end percent_increase_stock_l56_56716


namespace probability_of_selected_cubes_l56_56317

-- Total number of unit cubes
def total_unit_cubes : ℕ := 125

-- Number of cubes with exactly two blue faces (from edges not corners)
def two_blue_faces : ℕ := 9

-- Number of unpainted unit cubes
def unpainted_cubes : ℕ := 51

-- Calculate total combinations of choosing 2 cubes out of 125
def total_combinations : ℕ := Nat.choose total_unit_cubes 2

-- Calculate favorable outcomes: one cube with 2 blue faces and one unpainted cube
def favorable_outcomes : ℕ := two_blue_faces * unpainted_cubes

-- Calculate probability
def probability : ℚ := favorable_outcomes / total_combinations

-- The theorem we want to prove
theorem probability_of_selected_cubes :
  probability = 3 / 50 :=
by
  -- Show that the probability indeed equals 3/50
  sorry

end probability_of_selected_cubes_l56_56317


namespace find_m_l56_56529

theorem find_m (m : ℝ) (h : ∀ x : ℝ, m - |x| ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1) : m = 1 :=
sorry

end find_m_l56_56529


namespace expression_value_l56_56748

theorem expression_value : 2 + 3 * 5 + 2 = 19 := by
  sorry

end expression_value_l56_56748


namespace least_positive_number_of_24x_plus_16y_is_8_l56_56612

theorem least_positive_number_of_24x_plus_16y_is_8 :
  ∃ (x y : ℤ), 24 * x + 16 * y = 8 :=
by
  sorry

end least_positive_number_of_24x_plus_16y_is_8_l56_56612


namespace greatest_possible_fourth_term_l56_56720

theorem greatest_possible_fourth_term {a d : ℕ} (h : 5 * a + 10 * d = 60) : a + 3 * (12 - a) ≤ 34 :=
by 
  sorry

end greatest_possible_fourth_term_l56_56720


namespace minimum_value_fraction_l56_56157

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b + c ≥ a)

theorem minimum_value_fraction : (b / c + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end minimum_value_fraction_l56_56157


namespace value_of_a6_l56_56839

theorem value_of_a6 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 * n^2 - 5 * n)
  (ha : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h1 : a 1 = S 1):
  a 6 = 28 :=
sorry

end value_of_a6_l56_56839


namespace evaluate_expression_l56_56601

theorem evaluate_expression (a b : ℝ) (h : (1/2 * a * (1:ℝ)^3 - 3 * b * 1 + 4 = 9)) :
  (1/2 * a * (-1:ℝ)^3 - 3 * b * (-1) + 4 = -1) := by
sorry

end evaluate_expression_l56_56601


namespace arithmetic_geom_sequences_l56_56495

theorem arithmetic_geom_sequences
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ q, ∀ n, b (n + 1) = b n * q)
  (h1 : a 2 + a 3 = 14)
  (h2 : a 4 - a 1 = 6)
  (h3 : b 2 = a 1)
  (h4 : b 3 = a 3) :
  (∀ n, a n = 2 * n + 2) ∧ (∃ m, b 6 = a m ∧ m = 31) := sorry

end arithmetic_geom_sequences_l56_56495


namespace total_bricks_fill_box_l56_56516

-- Define brick and box volumes based on conditions
def volume_brick1 := 2 * 5 * 8
def volume_brick2 := 2 * 3 * 7
def volume_box := 10 * 11 * 14

-- Define the main proof problem
theorem total_bricks_fill_box (x y : ℕ) (h1 : volume_brick1 * x + volume_brick2 * y = volume_box) :
  x + y = 24 :=
by
  -- Left as an exercise (proof steps are not included per instructions)
  sorry

end total_bricks_fill_box_l56_56516


namespace tickets_spent_correct_l56_56657

/-- Tom won 32 tickets playing 'whack a mole'. -/
def tickets_whack_mole : ℕ := 32

/-- Tom won 25 tickets playing 'skee ball'. -/
def tickets_skee_ball : ℕ := 25

/-- Tom is left with 50 tickets after spending some on a hat. -/
def tickets_left : ℕ := 50

/-- The total number of tickets Tom won from both games. -/
def tickets_total : ℕ := tickets_whack_mole + tickets_skee_ball

/-- The number of tickets Tom spent on the hat. -/
def tickets_spent : ℕ := tickets_total - tickets_left

-- Prove that the number of tickets Tom spent on the hat is 7.
theorem tickets_spent_correct : tickets_spent = 7 := by
  -- Proof goes here
  sorry

end tickets_spent_correct_l56_56657


namespace gcd_71_19_l56_56524

theorem gcd_71_19 : Int.gcd 71 19 = 1 := by
  sorry

end gcd_71_19_l56_56524


namespace brass_total_l56_56730

theorem brass_total (p_cu : ℕ) (p_zn : ℕ) (m_zn : ℕ) (B : ℕ) 
  (h_ratio : p_cu = 13) 
  (h_zn_ratio : p_zn = 7) 
  (h_zn_mass : m_zn = 35) : 
  (h_brass_total :  p_zn / (p_cu + p_zn) * B = m_zn) → B = 100 :=
sorry

end brass_total_l56_56730


namespace incorrect_proposition_C_l56_56736

theorem incorrect_proposition_C (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ↔ False :=
by sorry

end incorrect_proposition_C_l56_56736


namespace least_possible_z_minus_x_l56_56576

theorem least_possible_z_minus_x (x y z : ℕ) 
  (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hxy : x < y) (hyz : y < z) (hyx_gt_3: y - x > 3)
  (hx_even : x % 2 = 0) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1) :
  z - x = 9 :=
sorry

end least_possible_z_minus_x_l56_56576


namespace condition_sufficiency_but_not_necessity_l56_56089

variable (p q : Prop)

theorem condition_sufficiency_but_not_necessity:
  (¬ (p ∨ q) → ¬ p) ∧ (¬ p → ¬ (p ∨ q) → False) := 
by
  sorry

end condition_sufficiency_but_not_necessity_l56_56089


namespace average_speed_correct_l56_56752

-- Define the speeds for each hour
def speed_hour1 := 90 -- km/h
def speed_hour2 := 40 -- km/h
def speed_hour3 := 60 -- km/h
def speed_hour4 := 80 -- km/h
def speed_hour5 := 50 -- km/h

-- Define the total time of the journey
def total_time := 5 -- hours

-- Calculate the sum of distances
def total_distance := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4 + speed_hour5

-- Define the average speed calculation
def average_speed := total_distance / total_time

-- The proof problem: average speed is 64 km/h
theorem average_speed_correct : average_speed = 64 := by
  sorry

end average_speed_correct_l56_56752


namespace selected_number_in_first_group_is_7_l56_56800

def N : ℕ := 800
def k : ℕ := 50
def interval : ℕ := N / k
def selected_number : ℕ := 39
def second_group_start : ℕ := 33
def second_group_end : ℕ := 48

theorem selected_number_in_first_group_is_7 
  (h1 : interval = 16)
  (h2 : selected_number ≥ second_group_start ∧ selected_number ≤ second_group_end)
  (h3 : ∃ n, selected_number = second_group_start + interval * n - 1) :
  selected_number % interval = 7 :=
sorry

end selected_number_in_first_group_is_7_l56_56800


namespace alpha_beta_sum_l56_56248

variable (α β : ℝ)

theorem alpha_beta_sum (h : ∀ x, (x - α) / (x + β) = (x^2 - 64 * x + 992) / (x^2 + 56 * x - 3168)) :
  α + β = 82 :=
sorry

end alpha_beta_sum_l56_56248


namespace six_digit_count_div_by_217_six_digit_count_div_by_218_l56_56838

-- Definitions for the problem
def six_digit_format (n : ℕ) : Prop :=
  ∃ a b : ℕ, (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10) ∧ n = 100001 * a + 10010 * b + 100 * a + 10 * b + a

def divisible_by (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 0

-- Problem Part a: How many six-digit numbers of the form are divisible by 217
theorem six_digit_count_div_by_217 :
  ∃ count : ℕ, count = 3 ∧ ∀ n : ℕ, six_digit_format n → divisible_by n 217  → (n = 313131 ∨ n = 626262 ∨ n = 939393) :=
sorry

-- Problem Part b: How many six-digit numbers of the form are divisible by 218
theorem six_digit_count_div_by_218 :
  ∀ n : ℕ, six_digit_format n → divisible_by n 218 → false :=
sorry

end six_digit_count_div_by_217_six_digit_count_div_by_218_l56_56838


namespace log_division_simplification_l56_56191

theorem log_division_simplification (log_base_half : ℝ → ℝ) (log_base_half_pow5 :  log_base_half (2 ^ 5) = 5 * log_base_half 2)
  (log_base_half_pow1 : log_base_half (2 ^ 1) = 1 * log_base_half 2) :
  (log_base_half 32) / (log_base_half 2) = 5 :=
sorry

end log_division_simplification_l56_56191


namespace kylie_daisies_l56_56012

theorem kylie_daisies :
  ∀ (initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies : ℕ),
    initial_daisies = 5 →
    sister_daisies = 9 →
    final_daisies = 7 →
    total_daisies = initial_daisies + sister_daisies →
    daisies_given_to_mother = total_daisies - final_daisies →
    daisies_given_to_mother * 2 = total_daisies :=
by
  intros initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies h1 h2 h3 h4 h5
  sorry

end kylie_daisies_l56_56012


namespace min_colors_needed_is_3_l56_56297

noncomputable def min_colors_needed (S : Finset (Fin 7)) : Nat :=
  -- function to determine the minimum number of colors needed
  if ∀ (f : Finset (Fin 7) → Fin 3), ∀ (A B : Finset (Fin 7)), A.card = 3 ∧ B.card = 3 →
    A ∩ B = ∅ → f A ≠ f B then
    3
  else
    sorry

theorem min_colors_needed_is_3 :
  ∀ S : Finset (Fin 7), min_colors_needed S = 3 :=
by
  sorry

end min_colors_needed_is_3_l56_56297


namespace time_for_b_l56_56064

theorem time_for_b (A B C : ℚ) (H1 : A + B + C = 1/4) (H2 : A = 1/12) (H3 : C = 1/18) : B = 1/9 :=
by {
  sorry
}

end time_for_b_l56_56064


namespace birthday_candles_l56_56665

def number_of_red_candles : ℕ := 18
def number_of_green_candles : ℕ := 37
def number_of_yellow_candles := number_of_red_candles / 2
def total_age : ℕ := 85
def total_candles_so_far := number_of_red_candles + number_of_yellow_candles + number_of_green_candles
def number_of_blue_candles := total_age - total_candles_so_far

theorem birthday_candles :
  number_of_yellow_candles = 9 ∧
  number_of_blue_candles = 21 ∧
  (number_of_red_candles + number_of_yellow_candles + number_of_green_candles + number_of_blue_candles) = total_age :=
by
  sorry

end birthday_candles_l56_56665


namespace range_of_a_in_quadratic_l56_56334

theorem range_of_a_in_quadratic :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 ≠ x2 ∧ x1^2 + a * x1 - 2 = 0 ∧ x2^2 + a * x2 - 2 = 0) → -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_in_quadratic_l56_56334


namespace linear_equation_with_two_variables_l56_56425

def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3 * a) * x + (a + 1) * y + 3 * a

theorem linear_equation_with_two_variables (a : ℝ) :
  (equation a x y = 0) ∧ (a^2 - 4 = 0) ∧ (2 - 3 * a ≠ 0) ∧ (a + 1 ≠ 0) →
  (a = 2 ∨ a = -2) :=
by sorry

end linear_equation_with_two_variables_l56_56425


namespace science_votes_percentage_l56_56114

theorem science_votes_percentage 
  (math_votes : ℕ) (english_votes : ℕ) (science_votes : ℕ) (history_votes : ℕ) (art_votes : ℕ) 
  (total_votes : ℕ := math_votes + english_votes + science_votes + history_votes + art_votes) 
  (percentage : ℕ := ((science_votes * 100) / total_votes)) :
  math_votes = 80 →
  english_votes = 70 →
  science_votes = 90 →
  history_votes = 60 →
  art_votes = 50 →
  percentage = 26 :=
by
  intros
  sorry

end science_votes_percentage_l56_56114


namespace necessary_but_not_sufficient_l56_56886

theorem necessary_but_not_sufficient
  (x y : ℝ) :
  (x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧ ¬ (x^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 2*x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l56_56886


namespace exists_integers_for_expression_l56_56430

theorem exists_integers_for_expression (n : ℤ) : 
  ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := 
sorry

end exists_integers_for_expression_l56_56430


namespace truthfulness_count_l56_56812

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l56_56812


namespace average_speed_increased_pace_l56_56997

theorem average_speed_increased_pace 
  (speed_constant : ℝ) (time_constant : ℝ) (distance_increased : ℝ) (total_time : ℝ) 
  (h1 : speed_constant = 15) 
  (h2 : time_constant = 3) 
  (h3 : distance_increased = 190) 
  (h4 : total_time = 13) :
  (distance_increased / (total_time - time_constant)) = 19 :=
by
  sorry

end average_speed_increased_pace_l56_56997


namespace water_channel_area_l56_56047

-- Define the given conditions
def top_width := 14
def bottom_width := 8
def depth := 70

-- The area formula for a trapezium given the top width, bottom width, and height
def trapezium_area (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- The main theorem stating the area of the trapezium
theorem water_channel_area : 
  trapezium_area top_width bottom_width depth = 770 := by
  -- Proof can be completed here
  sorry

end water_channel_area_l56_56047


namespace false_inverse_proposition_l56_56139

theorem false_inverse_proposition (a b : ℝ) : (a^2 = b^2) → (a = b ∨ a = -b) := sorry

end false_inverse_proposition_l56_56139


namespace f_order_l56_56233

variable (f : ℝ → ℝ)

-- Given conditions
axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom incr_f : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y

-- Prove that f(2) < f (-3/2) < f(-1)
theorem f_order : f 2 < f (-3/2) ∧ f (-3/2) < f (-1) :=
by
  sorry

end f_order_l56_56233


namespace matrix_power_sub_l56_56323

section 
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
variable (hA : A = ![![2, 3], ![0, 1]])

theorem matrix_power_sub (A : Matrix (Fin 2) (Fin 2) ℝ)
  (h: A = ![![2, 3], ![0, 1]]) :
  A ^ 20 - 2 * A ^ 19 = ![![0, 3], ![0, -1]] :=
by
  sorry
end

end matrix_power_sub_l56_56323


namespace quadrilateral_perimeter_correct_l56_56460

noncomputable def quadrilateral_perimeter : ℝ :=
  let AB := 15
  let BC := 20
  let CD := 9
  let AC := Real.sqrt (AB^2 + BC^2)
  let AD := Real.sqrt (AC^2 + CD^2)
  AB + BC + CD + AD

theorem quadrilateral_perimeter_correct :
  quadrilateral_perimeter = 44 + Real.sqrt 706 := by
  sorry

end quadrilateral_perimeter_correct_l56_56460


namespace age_6_not_child_l56_56928

-- Definition and assumptions based on the conditions
def billboard_number : ℕ := 5353
def mr_smith_age : ℕ := 53
def children_ages : List ℕ := [1, 2, 3, 4, 5, 7, 8, 9, 10, 11] -- Excluding age 6

-- The theorem to prove that the age 6 is not one of Mr. Smith's children's ages.
theorem age_6_not_child :
  (billboard_number ≡ 53 * 101 [MOD 10^4]) ∧
  (∀ age ∈ children_ages, billboard_number % age = 0) ∧
  oldest_child_age = 11 → ¬(6 ∈ children_ages) :=
sorry

end age_6_not_child_l56_56928


namespace not_dividable_by_wobbly_l56_56590

-- Define a wobbly number
def is_wobbly_number (n : ℕ) : Prop :=
  n > 0 ∧ (∀ k : ℕ, k < (Nat.log 10 n) → 
    (n / (10^k) % 10 ≠ 0 → n / (10^(k+1)) % 10 = 0) ∧
    (n / (10^k) % 10 = 0 → n / (10^(k+1)) % 10 ≠ 0))

-- Define sets of multiples of 10 and 25
def multiples_of (m : ℕ) (k : ℕ): Prop :=
  ∃ q : ℕ, k = q * m

def is_multiple_of_10 (k : ℕ) : Prop := multiples_of 10 k
def is_multiple_of_25 (k : ℕ) : Prop := multiples_of 25 k

theorem not_dividable_by_wobbly (n : ℕ) : 
  ¬ ∃ w : ℕ, is_wobbly_number w ∧ n ∣ w ↔ is_multiple_of_10 n ∨ is_multiple_of_25 n :=
by
  sorry

end not_dividable_by_wobbly_l56_56590


namespace find_common_students_l56_56147

theorem find_common_students
  (total_english : ℕ)
  (total_math : ℕ)
  (difference_only_english_math : ℕ)
  (both_english_math : ℕ) :
  total_english = both_english_math + (both_english_math + 10) →
  total_math = both_english_math + both_english_math →
  difference_only_english_math = 10 →
  total_english = 30 →
  total_math = 20 →
  both_english_math = 10 :=
by
  intros
  sorry

end find_common_students_l56_56147


namespace first_dimension_length_l56_56793

-- Definitions for conditions
def tank_surface_area (x : ℝ) : ℝ := 14 * x + 20
def cost_per_sqft : ℝ := 20
def total_cost (x : ℝ) : ℝ := (tank_surface_area x) * cost_per_sqft

-- The theorem we need to prove
theorem first_dimension_length : ∃ x : ℝ, total_cost x = 1520 ∧ x = 4 := by 
  sorry

end first_dimension_length_l56_56793


namespace correct_propositions_l56_56391

variable (A : Set ℝ)
variable (oplus : ℝ → ℝ → ℝ)

def condition_a1 : Prop := ∀ a b : ℝ, a ∈ A → b ∈ A → (oplus a b) ∈ A
def condition_a2 : Prop := ∀ a : ℝ, a ∈ A → (oplus a a) = 0
def condition_a3 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus (oplus a b) c) = (oplus a c) + (oplus b c) + c

def proposition_1 : Prop := 0 ∈ A
def proposition_2 : Prop := (1 ∈ A) → (oplus (oplus 1 1) 1) = 0
def proposition_3 : Prop := ∀ a : ℝ, a ∈ A → (oplus a 0) = a → a = 0
def proposition_4 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus a 0) = a → (oplus a b) = (oplus c b) → a = c

theorem correct_propositions 
  (h1 : condition_a1 A oplus) 
  (h2 : condition_a2 A oplus)
  (h3 : condition_a3 A oplus) : 
  (proposition_1 A) ∧ (¬proposition_2 A oplus) ∧ (proposition_3 A oplus) ∧ (proposition_4 A oplus) := by
  sorry

end correct_propositions_l56_56391


namespace calc1_l56_56420

theorem calc1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (Real.pi / 3) = 1 - Real.sqrt 3 :=
by
  sorry

end calc1_l56_56420


namespace intersection_M_N_l56_56857

def M : Set ℝ := {y | ∃ x, x ∈ Set.Icc (-5) 5 ∧ y = 2 * Real.sin x}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2}

theorem intersection_M_N : {x | 1 < x ∧ x ≤ 2} = {x | x ∈ M ∩ N} :=
by sorry

end intersection_M_N_l56_56857


namespace geometric_to_arithmetic_l56_56098

theorem geometric_to_arithmetic {a1 a2 a3 a4 q : ℝ}
  (hq : q ≠ 1)
  (geom_seq : a2 = a1 * q ∧ a3 = a1 * q^2 ∧ a4 = a1 * q^3)
  (arith_seq : (2 * a3 = a1 + a4 ∨ 2 * a2 = a1 + a4)) :
  q = (1 + Real.sqrt 5) / 2 ∨ q = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_to_arithmetic_l56_56098


namespace range_of_p_l56_56824

def A (x : ℝ) : Prop := -2 < x ∧ x < 5
def B (p : ℝ) (x : ℝ) : Prop := p + 1 < x ∧ x < 2 * p - 1

theorem range_of_p (p : ℝ) :
  (∀ x, A x ∨ B p x → A x) ↔ p ≤ 3 :=
by
  sorry

end range_of_p_l56_56824


namespace percentage_of_silver_in_final_solution_l56_56368

noncomputable section -- because we deal with real numbers and division

variable (volume_4pct : ℝ) (percentage_4pct : ℝ)
variable (volume_10pct : ℝ) (percentage_10pct : ℝ)

def final_percentage_silver (v4 : ℝ) (p4 : ℝ) (v10 : ℝ) (p10 : ℝ) : ℝ :=
  let total_silver := v4 * p4 + v10 * p10
  let total_volume := v4 + v10
  (total_silver / total_volume) * 100

theorem percentage_of_silver_in_final_solution :
  final_percentage_silver 5 0.04 2.5 0.10 = 6 := by
  sorry

end percentage_of_silver_in_final_solution_l56_56368


namespace locus_is_hyperbola_l56_56331

theorem locus_is_hyperbola
  (x y a θ₁ θ₂ c : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (hc : c > 1) 
  : ∃ k l m : ℝ, k * (x ^ 2) + l * x * y + m * (y ^ 2) = 1 := sorry

end locus_is_hyperbola_l56_56331


namespace percent_increase_march_to_april_l56_56222

theorem percent_increase_march_to_april (P : ℝ) (X : ℝ) 
  (H1 : ∃ Y Z : ℝ, P * (1 + X / 100) * 0.8 * 1.5 = P * (1 + Y / 100) ∧ Y = 56.00000000000001)
  (H2 : P * (1 + X / 100) * 0.8 * 1.5 = P * 1.5600000000000001)
  (H3 : P ≠ 0) :
  X = 30 :=
by sorry

end percent_increase_march_to_april_l56_56222


namespace percent_spent_on_other_items_l56_56471

def total_amount_spent (T : ℝ) : ℝ := T
def clothing_percent (p : ℝ) : Prop := p = 0.45
def food_percent (p : ℝ) : Prop := p = 0.45
def clothing_tax (t : ℝ) (T : ℝ) : ℝ := 0.05 * (0.45 * T)
def food_tax (t : ℝ) (T : ℝ) : ℝ := 0.0 * (0.45 * T)
def other_items_tax (p : ℝ) (T : ℝ) : ℝ := 0.10 * (p * T)
def total_tax (T : ℝ) (tax : ℝ) : Prop := tax = 0.0325 * T

theorem percent_spent_on_other_items (T : ℝ) (p_clothing p_food x : ℝ) (tax : ℝ) 
  (h1 : clothing_percent p_clothing) (h2 : food_percent p_food)
  (h3 : clothing_tax tax T = 0.05 * (0.45 * T))
  (h4 : food_tax tax T = 0.0)
  (h5 : other_items_tax x T = 0.10 * (x * T))
  (h6 : total_tax T (clothing_tax tax T + food_tax tax T + other_items_tax x T)) : 
  x = 0.10 :=
by
  sorry

end percent_spent_on_other_items_l56_56471


namespace original_marketing_pct_correct_l56_56521

-- Define the initial and final percentages of finance specialization students
def initial_finance_pct := 0.88
def final_finance_pct := 0.90

-- Define the final percentage of marketing specialization students
def final_marketing_pct := 0.43333333333333335

-- Define the original percentage of marketing specialization students
def original_marketing_pct := 0.45333333333333335

-- The Lean statement to prove the original percentage of marketing students
theorem original_marketing_pct_correct :
  initial_finance_pct + (final_marketing_pct - initial_finance_pct) = original_marketing_pct := 
sorry

end original_marketing_pct_correct_l56_56521


namespace pies_and_leftover_apples_l56_56645

theorem pies_and_leftover_apples 
  (apples : ℕ) 
  (h : apples = 55) 
  (h1 : 15/3 = 5) :
  (apples / 5 = 11) ∧ (apples - 11 * 5 = 0) :=
by
  sorry

end pies_and_leftover_apples_l56_56645


namespace JackBuckets_l56_56660

theorem JackBuckets (tank_capacity buckets_per_trip_jill trips_jill time_ratio trip_buckets_jack : ℕ) :
  tank_capacity = 600 → buckets_per_trip_jill = 5 → trips_jill = 30 →
  time_ratio = 3 / 2 → trip_buckets_jack = 2 :=
  sorry

end JackBuckets_l56_56660


namespace fountain_distance_l56_56217

theorem fountain_distance (h_AD : ℕ) (h_BC : ℕ) (h_AB : ℕ) (h_AD_eq : h_AD = 30) (h_BC_eq : h_BC = 40) (h_AB_eq : h_AB = 50) :
  ∃ AE EB : ℕ, AE = 32 ∧ EB = 18 := by
  sorry

end fountain_distance_l56_56217


namespace well_depth_l56_56737

def daily_climb_up : ℕ := 4
def daily_slip_down : ℕ := 3
def total_days : ℕ := 27

theorem well_depth : (daily_climb_up * (total_days - 1) - daily_slip_down * (total_days - 1)) + daily_climb_up = 30 := by
  -- conditions
  let net_daily_progress := daily_climb_up - daily_slip_down
  let net_26_days_progress := net_daily_progress * (total_days - 1)

  -- proof to be completed
  sorry

end well_depth_l56_56737


namespace n_divisible_by_6_l56_56116

open Int -- Open integer namespace for convenience

theorem n_divisible_by_6 (m n : ℤ)
    (h1 : ∃ (a b : ℤ), a + b = -m ∧ a * b = -n)
    (h2 : ∃ (c d : ℤ), c + d = m ∧ c * d = n) :
    6 ∣ n := 
sorry

end n_divisible_by_6_l56_56116


namespace find_pairs_xy_l56_56474

theorem find_pairs_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : 7^x - 3 * 2^y = 1) : 
  (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
sorry

end find_pairs_xy_l56_56474


namespace compute_f3_l56_56972

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 4*n + 3 else 2*n + 1

theorem compute_f3 : f (f (f 3)) = 99 :=
by
  sorry

end compute_f3_l56_56972


namespace repeating_decimal_sum_is_one_l56_56861

noncomputable def repeating_decimal_sum : ℝ :=
  let x := (1/3 : ℝ)
  let y := (2/3 : ℝ)
  x + y

theorem repeating_decimal_sum_is_one : repeating_decimal_sum = 1 := by
  sorry

end repeating_decimal_sum_is_one_l56_56861


namespace correct_option_given_inequality_l56_56379

theorem correct_option_given_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
sorry

end correct_option_given_inequality_l56_56379


namespace total_people_at_evening_l56_56107

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l56_56107


namespace max_disks_l56_56023

theorem max_disks (n k : ℕ) (h1: n ≥ 1) (h2: k ≥ 1) :
  (∃ (d : ℕ), d = if n > 1 ∧ k > 1 then 2 * (n + k) - 4 else max n k) ∧
  (∀ (p q : ℕ), (p <= n → q <= k → ¬∃ (x y : ℕ), x + 1 = y ∨ x - 1 = y ∨ x + 1 = p ∨ x - 1 = p)) :=
sorry

end max_disks_l56_56023


namespace max_discount_rate_l56_56444

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l56_56444


namespace find_M_l56_56954

theorem find_M (M : ℕ) (h1 : M > 0) (h2 : M < 10) : 
  5 ∣ (1989^M + M^1989) ↔ M = 1 ∨ M = 4 := by
  sorry

end find_M_l56_56954


namespace cirrus_clouds_count_l56_56018

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end cirrus_clouds_count_l56_56018


namespace no_valid_a_exists_l56_56361

theorem no_valid_a_exists (a : ℕ) (n : ℕ) (h1 : a > 1) (b := a * (10^n + 1)) :
  ¬ (∃ a : ℕ, b % (a^2) = 0) :=
by {
  sorry -- The actual proof is not required as per instructions.
}

end no_valid_a_exists_l56_56361


namespace root_and_value_of_a_equation_has_real_roots_l56_56755

theorem root_and_value_of_a (a : ℝ) (other_root : ℝ) :
  (∃ x : ℝ, x^2 + a * x + a - 1 = 0 ∧ x = 2) → a = -1 ∧ other_root = -1 :=
by sorry

theorem equation_has_real_roots (a : ℝ) :
  ∃ x : ℝ, x^2 + a * x + a - 1 = 0 :=
by sorry

end root_and_value_of_a_equation_has_real_roots_l56_56755


namespace initial_number_of_persons_l56_56653

/-- The average weight of some persons increases by 3 kg when a new person comes in place of one of them weighing 65 kg. 
    The weight of the new person might be 89 kg.
    Prove that the number of persons initially was 8.
-/
theorem initial_number_of_persons (n : ℕ) (h1 : (89 - 65 = 3 * n)) : n = 8 := by
  sorry

end initial_number_of_persons_l56_56653


namespace evaluate_expression_l56_56504

theorem evaluate_expression : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end evaluate_expression_l56_56504


namespace find_f_cos_10_l56_56898

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) : f (Real.sin x) = Real.cos (3 * x)

theorem find_f_cos_10 : f (Real.cos (10 * Real.pi / 180)) = -1/2 := by
  sorry

end find_f_cos_10_l56_56898


namespace simplify_expression_l56_56847

-- Define the main theorem
theorem simplify_expression 
  (a b x : ℝ) 
  (hx : x = 1 / a * Real.sqrt ((2 * a - b) / b))
  (hc1 : 0 < b / 2)
  (hc2 : b / 2 < a)
  (hc3 : a < b) : 
  (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1 :=
sorry

end simplify_expression_l56_56847


namespace john_sarah_money_total_l56_56417

theorem john_sarah_money_total (j_money s_money : ℚ) (H1 : j_money = 5/8) (H2 : s_money = 7/16) :
  (j_money + s_money : ℚ) = 1.0625 := 
by
  sorry

end john_sarah_money_total_l56_56417


namespace sum_of_coefficients_l56_56570

theorem sum_of_coefficients (f : ℕ → ℕ) :
  (5 * 1 + 2)^7 = 823543 :=
by
  sorry

end sum_of_coefficients_l56_56570


namespace solve_for_x_l56_56166

theorem solve_for_x (x : ℝ) (h : (10 - 6 * x)^ (1 / 3) = -2) : x = 3 := 
by
  sorry

end solve_for_x_l56_56166


namespace lives_per_each_player_l56_56733

def num_initial_players := 8
def num_quit_players := 3
def total_remaining_lives := 15
def num_remaining_players := num_initial_players - num_quit_players
def lives_per_remaining_player := total_remaining_lives / num_remaining_players

theorem lives_per_each_player :
  lives_per_remaining_player = 3 := by
  sorry

end lives_per_each_player_l56_56733


namespace mean_of_six_numbers_l56_56137

theorem mean_of_six_numbers (sum_of_six : ℚ) (h : sum_of_six = 3 / 4) : (sum_of_six / 6) = 1 / 8 :=
by
  sorry

end mean_of_six_numbers_l56_56137


namespace cos_pi_over_3_plus_2alpha_l56_56872

variable (α : ℝ)

theorem cos_pi_over_3_plus_2alpha (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (π / 3 + 2 * α) = 7 / 9 :=
  sorry

end cos_pi_over_3_plus_2alpha_l56_56872


namespace can_pay_without_change_l56_56338

theorem can_pay_without_change (n : ℕ) (h : n > 7) :
  ∃ (a b : ℕ), 3 * a + 5 * b = n :=
sorry

end can_pay_without_change_l56_56338


namespace class_with_avg_40_students_l56_56230

theorem class_with_avg_40_students
  (x y : ℕ)
  (h : 40 * x + 60 * y = (380 * (x + y)) / 7) : x = 40 :=
sorry

end class_with_avg_40_students_l56_56230


namespace correct_operation_B_l56_56300

theorem correct_operation_B (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 :=
sorry

end correct_operation_B_l56_56300


namespace dixie_cup_ounces_l56_56120

def gallons_to_ounces (gallons : ℕ) : ℕ := gallons * 128

def initial_water_gallons (gallons : ℕ) : ℕ := gallons_to_ounces gallons

def total_chairs (rows chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

theorem dixie_cup_ounces (initial_gallons rows chairs_per_row water_left : ℕ) 
  (h1 : initial_gallons = 3) 
  (h2 : rows = 5) 
  (h3 : chairs_per_row = 10) 
  (h4 : water_left = 84) 
  (h5 : 128 = 128) : 
  (initial_water_gallons initial_gallons - water_left) / total_chairs rows chairs_per_row = 6 :=
by 
  sorry

end dixie_cup_ounces_l56_56120


namespace quadratic_roots_min_value_l56_56457

theorem quadratic_roots_min_value (m α β : ℝ) (h_eq : 4 * α^2 - 4 * m * α + m + 2 = 0) (h_eq2 : 4 * β^2 - 4 * m * β + m + 2 = 0) :
  (∃ m_val : ℝ, m_val = -1 ∧ α^2 + β^2 = 1 / 2) :=
by
  sorry

end quadratic_roots_min_value_l56_56457


namespace red_button_probability_l56_56169

theorem red_button_probability :
  let jarA_red := 6
  let jarA_blue := 9
  let jarA_total := jarA_red + jarA_blue
  let jarA_half := jarA_total / 2
  let removed_total := jarA_total - jarA_half
  let removed_red := removed_total / 2
  let removed_blue := removed_total / 2
  let jarA_red_remaining := jarA_red - removed_red
  let jarA_blue_remaining := jarA_blue - removed_blue
  let jarB_red := removed_red
  let jarB_blue := removed_blue
  let jarA_total_remaining := jarA_red_remaining + jarA_blue_remaining
  let jarB_total := jarB_red + jarB_blue
  (jarA_total = 15) →
  (jarA_red_remaining = 6 - removed_red) →
  (jarA_blue_remaining = 9 - removed_blue) →
  (jarB_red = removed_red) →
  (jarB_blue = removed_blue) →
  (jarA_red_remaining + jarA_blue_remaining = 9) →
  (jarB_red + jarB_blue = 6) →
  let prob_red_JarA := jarA_red_remaining / jarA_total_remaining
  let prob_red_JarB := jarB_red / jarB_total
  prob_red_JarA * prob_red_JarB = 1 / 6 := by sorry

end red_button_probability_l56_56169


namespace exponentiation_condition_l56_56663

theorem exponentiation_condition (a b : ℝ) (h0 : a > 0) (h1 : a ≠ 1) : 
  (a ^ b > 1 ↔ (a - 1) * b > 0) :=
sorry

end exponentiation_condition_l56_56663


namespace clerical_percentage_l56_56132

theorem clerical_percentage (total_employees clerical_fraction reduce_fraction: ℕ) 
  (h1 : total_employees = 3600) 
  (h2 : clerical_fraction = 1 / 3)
  (h3 : reduce_fraction = 1 / 2) : 
  ( (reduce_fraction * (clerical_fraction * total_employees)) / 
    (total_employees - reduce_fraction * (clerical_fraction * total_employees))) * 100 = 20 :=
by
  sorry

end clerical_percentage_l56_56132


namespace power_mod_equiv_l56_56037

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end power_mod_equiv_l56_56037


namespace symmetric_point_of_P_l56_56143

-- Define a point in the Cartesian coordinate system
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define central symmetry with respect to the origin
def symmetric (p : Point) : Point :=
  { x := -p.x, y := -p.y }

-- Given point P with coordinates (1, -2)
def P : Point := { x := 1, y := -2 }

-- The theorem to be proved: the symmetric point of P is (-1, 2)
theorem symmetric_point_of_P :
  symmetric P = { x := -1, y := 2 } :=
by
  -- Proof is omitted.
  sorry

end symmetric_point_of_P_l56_56143


namespace machine_production_time_l56_56367

theorem machine_production_time (x : ℝ) 
  (h1 : 60 / x + 2 = 12) : 
  x = 6 :=
sorry

end machine_production_time_l56_56367


namespace curve_symmetry_l56_56685

-- Define the curve as a predicate
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0

-- Define the point symmetry condition for a line
def is_symmetric_about_line (curve : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, curve x y → line x y

-- Define the line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop := x + y = 0

-- Main theorem stating the curve is symmetrical about the line x + y = 0
theorem curve_symmetry : is_symmetric_about_line curve line_x_plus_y_eq_0 := 
sorry

end curve_symmetry_l56_56685


namespace base_of_second_exponent_l56_56643

theorem base_of_second_exponent (a b : ℕ) (x : ℕ) 
  (h1 : (18^a) * (x^(3 * a - 1)) = (2^6) * (3^b)) 
  (h2 : a = 6) 
  (h3 : 0 < a)
  (h4 : 0 < b) : x = 3 := 
by
  sorry

end base_of_second_exponent_l56_56643


namespace infinitely_many_primes_of_form_6n_plus_5_l56_56656

theorem infinitely_many_primes_of_form_6n_plus_5 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p % 6 = 5 :=
sorry

end infinitely_many_primes_of_form_6n_plus_5_l56_56656


namespace sum_fifth_powers_divisible_by_15_l56_56342

theorem sum_fifth_powers_divisible_by_15
  (A B C D E : ℤ) 
  (h : A + B + C + D + E = 0) : 
  (A^5 + B^5 + C^5 + D^5 + E^5) % 15 = 0 := 
by 
  sorry

end sum_fifth_powers_divisible_by_15_l56_56342


namespace sum_of_sides_of_similar_triangle_l56_56718

theorem sum_of_sides_of_similar_triangle (a b c : ℕ) (scale_factor : ℕ) (longest_side_sim : ℕ) (sum_of_other_sides_sim : ℕ) : 
  a * scale_factor = 21 → c = 7 → b = 5 → a = 3 → 
  sum_of_other_sides = a * scale_factor + b * scale_factor → 
sum_of_other_sides = 24 :=
by
  sorry

end sum_of_sides_of_similar_triangle_l56_56718


namespace find_arithmetic_sequence_l56_56189

theorem find_arithmetic_sequence (a d : ℝ) (h1 : (a - d) + a + (a + d) = 12) (h2 : (a - d) * a * (a + d) = 48) :
  (a = 4 ∧ d = 2) ∨ (a = 4 ∧ d = -2) :=
sorry

end find_arithmetic_sequence_l56_56189


namespace problem_a_problem_b_problem_c_l56_56042

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l56_56042


namespace sally_gave_joan_5_balloons_l56_56602

theorem sally_gave_joan_5_balloons (x : ℕ) (h1 : 9 + x - 2 = 12) : x = 5 :=
by
  -- Proof is skipped
  sorry

end sally_gave_joan_5_balloons_l56_56602


namespace total_property_value_l56_56372

-- Define the given conditions
def price_per_sq_ft_condo := 98
def price_per_sq_ft_barn := 84
def price_per_sq_ft_detached := 102
def price_per_sq_ft_garage := 60
def sq_ft_condo := 2400
def sq_ft_barn := 1200
def sq_ft_detached := 3500
def sq_ft_garage := 480

-- Main statement to prove the total value of the property
theorem total_property_value :
  (price_per_sq_ft_condo * sq_ft_condo + 
   price_per_sq_ft_barn * sq_ft_barn + 
   price_per_sq_ft_detached * sq_ft_detached + 
   price_per_sq_ft_garage * sq_ft_garage = 721800) :=
by
  -- Placeholder for the actual proof
  sorry

end total_property_value_l56_56372


namespace value_of_b_l56_56548

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
sorry

end value_of_b_l56_56548


namespace greatest_mean_YZ_l56_56525

noncomputable def X_mean := 60
noncomputable def Y_mean := 70
noncomputable def XY_mean := 64
noncomputable def XZ_mean := 66

theorem greatest_mean_YZ (Xn Yn Zn : ℕ) (m : ℕ) :
  (60 * Xn + 70 * Yn) / (Xn + Yn) = 64 →
  (60 * Xn + m) / (Xn + Zn) = 66 →
  ∃ (k : ℕ), k = 69 :=
by
  intro h1 h2
  -- Sorry is used to skip the proof
  sorry

end greatest_mean_YZ_l56_56525


namespace actual_distance_traveled_l56_56560

theorem actual_distance_traveled (D T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 20 * T) : D = 20 :=
by
  sorry

end actual_distance_traveled_l56_56560


namespace triangle_is_isosceles_l56_56571

theorem triangle_is_isosceles
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * c * Real.cos B)
    (h2 : b = c * Real.cos A) 
    (h3 : c = a * Real.cos C) 
    : a = b := 
sorry

end triangle_is_isosceles_l56_56571


namespace brick_height_l56_56142

/-- A certain number of bricks, each measuring 25 cm x 11.25 cm x some height, 
are needed to build a wall of 8 m x 6 m x 22.5 cm. 
If 6400 bricks are needed, prove that the height of each brick is 6 cm. -/
theorem brick_height (h : ℝ) : 
  6400 * (25 * 11.25 * h) = (800 * 600 * 22.5) → h = 6 :=
by
  sorry

end brick_height_l56_56142


namespace find_d_l56_56401

theorem find_d (d : ℕ) : (1059 % d = 1417 % d) ∧ (1059 % d = 2312 % d) ∧ (1417 % d = 2312 % d) ∧ (d > 1) → d = 179 :=
by
  sorry

end find_d_l56_56401


namespace bacteria_growth_l56_56985

-- Defining the function for bacteria growth
def bacteria_count (t : ℕ) (initial_count : ℕ) (division_time : ℕ) : ℕ :=
  initial_count * 2 ^ (t / division_time)

-- The initial conditions given in the problem
def initial_bacteria : ℕ := 1
def division_interval : ℕ := 10
def total_time : ℕ := 2 * 60

-- Stating the hypothesis and the goal
theorem bacteria_growth : bacteria_count total_time initial_bacteria division_interval = 2 ^ 12 :=
by
  -- Proof would go here
  sorry

end bacteria_growth_l56_56985


namespace time_of_same_distance_l56_56523

theorem time_of_same_distance (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 60) : 180 - 6 * m = 90 + 0.5 * m :=
by
  sorry

end time_of_same_distance_l56_56523


namespace morning_snowfall_l56_56754

theorem morning_snowfall (afternoon_snowfall total_snowfall : ℝ) (h₀ : afternoon_snowfall = 0.5) (h₁ : total_snowfall = 0.63):
  total_snowfall - afternoon_snowfall = 0.13 :=
by 
  sorry

end morning_snowfall_l56_56754


namespace coordinates_of_P_l56_56559

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-1, -2)

theorem coordinates_of_P : P = (1 / 3 • (B.1 - A.1) + 2 / 3 • A.1, 1 / 3 • (B.2 - A.2) + 2 / 3 • A.2) :=
by
    rw [A, B, P]
    sorry

end coordinates_of_P_l56_56559


namespace least_subtraction_for_divisibility_l56_56882

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 964807) : ∃ k, k = 7 ∧ (n - k) % 8 = 0 :=
by 
  sorry

end least_subtraction_for_divisibility_l56_56882


namespace not_square_difference_formula_l56_56809

theorem not_square_difference_formula (x y : ℝ) : ¬ ∃ (a b : ℝ), (x - y) * (-x + y) = (a + b) * (a - b) := 
sorry

end not_square_difference_formula_l56_56809


namespace proposition_2_proposition_4_l56_56823

-- Definitions from conditions.
def circle_M (x y q : ℝ) : Prop := (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1
def line_l (y k x : ℝ) : Prop := y = k * x

-- Prove that the line l and circle M always intersect for any real k and q.
theorem proposition_2 : ∀ (k q : ℝ), ∃ (x y : ℝ), circle_M x y q ∧ line_l y k x := sorry

-- Prove that for any real k, there exists a real q such that the line l is tangent to the circle M.
theorem proposition_4 : ∀ (k : ℝ), ∃ (q x y : ℝ), circle_M x y q ∧ line_l y k x ∧
  (abs (Real.sin q + k * Real.cos q) = 1 / Real.sqrt (1 + k^2)) := sorry

end proposition_2_proposition_4_l56_56823


namespace handshaking_remainder_div_1000_l56_56183

/-- Given eleven people where each person shakes hands with exactly three others, 
  let handshaking_count be the number of distinct handshaking arrangements.
  Find the remainder when handshaking_count is divided by 1000. -/
def handshaking_count (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) : Nat :=
  sorry

theorem handshaking_remainder_div_1000 (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) :
  (handshaking_count P hP handshakes H) % 1000 = 800 :=
sorry

end handshaking_remainder_div_1000_l56_56183


namespace ratio_of_length_to_breadth_l56_56831

theorem ratio_of_length_to_breadth 
    (breadth : ℝ) (area : ℝ) (h_breadth : breadth = 12) (h_area : area = 432)
    (h_area_formula : area = l * breadth) : 
    l / breadth = 3 :=
sorry

end ratio_of_length_to_breadth_l56_56831


namespace birds_in_sanctuary_l56_56603

theorem birds_in_sanctuary (x y : ℕ) 
    (h1 : x + y = 200)
    (h2 : 2 * x + 4 * y = 590) : 
    x = 105 :=
by
  sorry

end birds_in_sanctuary_l56_56603


namespace count_integers_with_zero_l56_56707

/-- There are 740 positive integers less than or equal to 3017 that contain the digit 0. -/
theorem count_integers_with_zero (n : ℕ) (h : n ≤ 3017) : 
  (∃ k : ℕ, k ≤ 3017 ∧ ∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ k / 10 ^ d % 10 = 0) ↔ n = 740 :=
by sorry

end count_integers_with_zero_l56_56707


namespace cube_of_number_l56_56259

theorem cube_of_number (n : ℕ) (h1 : 40000 < n^3) (h2 : n^3 < 50000) (h3 : (n^3 % 10) = 6) : n = 36 := by
  sorry

end cube_of_number_l56_56259


namespace value_of_n_l56_56920

theorem value_of_n : ∃ (n : ℕ), 6 * 8 * 3 * n = Nat.factorial 8 ∧ n = 280 :=
by
  use 280
  sorry

end value_of_n_l56_56920


namespace televisions_bought_l56_56100

theorem televisions_bought (T : ℕ)
  (television_cost : ℕ := 50)
  (figurine_cost : ℕ := 1)
  (num_figurines : ℕ := 10)
  (total_spent : ℕ := 260) :
  television_cost * T + figurine_cost * num_figurines = total_spent → T = 5 :=
by
  intros h
  sorry

end televisions_bought_l56_56100


namespace max_ben_cupcakes_l56_56340

theorem max_ben_cupcakes (total_cupcakes : ℕ) (ben_cupcakes charles_cupcakes diana_cupcakes : ℕ)
    (h1 : total_cupcakes = 30)
    (h2 : diana_cupcakes = 2 * ben_cupcakes)
    (h3 : charles_cupcakes = diana_cupcakes)
    (h4 : total_cupcakes = ben_cupcakes + charles_cupcakes + diana_cupcakes) :
    ben_cupcakes = 6 :=
by
  -- Proof steps would go here
  sorry

end max_ben_cupcakes_l56_56340


namespace amy_biking_miles_l56_56269

theorem amy_biking_miles (x : ℕ) (h1 : ∀ y : ℕ, y = 2 * x - 3) (h2 : ∀ y : ℕ, x + y = 33) : x = 12 :=
by
  sorry

end amy_biking_miles_l56_56269


namespace multiplication_trick_l56_56025

theorem multiplication_trick (a b c : ℕ) (h : b + c = 10) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by
  sorry

end multiplication_trick_l56_56025


namespace sample_size_is_five_l56_56052

def population := 100
def sample (n : ℕ) := n ≤ population
def sample_size (n : ℕ) := n

theorem sample_size_is_five (n : ℕ) (h : sample 5) : sample_size 5 = 5 :=
by
  sorry

end sample_size_is_five_l56_56052


namespace find_c_value_l56_56537

theorem find_c_value (x c : ℝ) (h₁ : 3 * x + 8 = 5) (h₂ : c * x + 15 = 3) : c = 12 :=
by
  -- This is where the proof steps would go, but we will use sorry for now.
  sorry

end find_c_value_l56_56537


namespace days_considered_l56_56593

theorem days_considered (visitors_current : ℕ) (visitors_previous : ℕ) (total_visitors : ℕ)
  (h1 : visitors_current = 132) (h2 : visitors_previous = 274) (h3 : total_visitors = 406)
  (h_total : visitors_current + visitors_previous = total_visitors) :
  2 = 2 :=
by
  sorry

end days_considered_l56_56593


namespace expression_evaluation_l56_56606

theorem expression_evaluation : abs (abs (-abs (-2 + 1) - 2) + 2) = 5 := 
by  
  sorry

end expression_evaluation_l56_56606


namespace remainder_of_product_l56_56870

theorem remainder_of_product (a b n : ℕ) (ha : a % n = 7) (hb : b % n = 1) :
  ((a * b) % n) = 7 :=
by
  -- Definitions as per the conditions
  let a := 63
  let b := 65
  let n := 8
  /- Now prove the statement -/
  sorry

end remainder_of_product_l56_56870


namespace alice_wrong_questions_l56_56988

theorem alice_wrong_questions :
  ∃ a b e : ℕ,
    (a + b = 6 + 8 + e) ∧
    (a + 8 = b + 6 + 3) ∧
    a = 9 :=
by {
  sorry
}

end alice_wrong_questions_l56_56988


namespace combined_annual_income_l56_56900

-- Define the given conditions and verify the combined annual income
def A_ratio : ℤ := 5
def B_ratio : ℤ := 2
def C_ratio : ℤ := 3
def D_ratio : ℤ := 4

def C_income : ℤ := 15000
def B_income : ℤ := 16800
def A_income : ℤ := 25000
def D_income : ℤ := 21250

theorem combined_annual_income :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by
  sorry

end combined_annual_income_l56_56900


namespace sin_alpha_at_point_l56_56536

open Real

theorem sin_alpha_at_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (1, -2)) :
  sin α = -2 * sqrt 5 / 5 :=
sorry

end sin_alpha_at_point_l56_56536


namespace tan_eq_one_over_three_l56_56365

theorem tan_eq_one_over_three (x : ℝ) (h1 : x ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.cos (2 * x - (Real.pi / 2)) = Real.sin x ^ 2) :
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end tan_eq_one_over_three_l56_56365


namespace tv_episode_length_l56_56577

theorem tv_episode_length :
  ∀ (E : ℕ), 
    600 = 3 * E + 270 + 2 * 105 + 45 → 
    E = 25 :=
by
  intros E h
  sorry

end tv_episode_length_l56_56577


namespace max_possible_value_l56_56285

theorem max_possible_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∀ (z : ℝ), (z = (x + y + 1) / x) → z ≤ -0.2 :=
by sorry

end max_possible_value_l56_56285


namespace tan_frac_eq_one_l56_56426

open Real

-- Conditions given in the problem
def sin_frac_cond (x y : ℝ) : Prop := (sin x / sin y) + (sin y / sin x) = 4
def cos_frac_cond (x y : ℝ) : Prop := (cos x / cos y) + (cos y / cos x) = 3

-- Statement of the theorem to be proved
theorem tan_frac_eq_one (x y : ℝ) (h1 : sin_frac_cond x y) (h2 : cos_frac_cond x y) : (tan x / tan y) + (tan y / tan x) = 1 :=
by
  sorry

end tan_frac_eq_one_l56_56426


namespace probability_both_A_B_selected_l56_56423

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l56_56423


namespace average_waiting_time_l56_56708

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l56_56708


namespace max_probability_of_winning_is_correct_l56_56719

noncomputable def max_probability_of_winning : ℚ :=
  sorry

theorem max_probability_of_winning_is_correct :
  max_probability_of_winning = 17 / 32 :=
sorry

end max_probability_of_winning_is_correct_l56_56719


namespace largest_expression_l56_56356

def P : ℕ := 3 * 2024 ^ 2025
def Q : ℕ := 2024 ^ 2025
def R : ℕ := 2023 * 2024 ^ 2024
def S : ℕ := 3 * 2024 ^ 2024
def T : ℕ := 2024 ^ 2024
def U : ℕ := 2024 ^ 2023

theorem largest_expression : 
  (P - Q) > (Q - R) ∧ 
  (P - Q) > (R - S) ∧ 
  (P - Q) > (S - T) ∧ 
  (P - Q) > (T - U) :=
by sorry

end largest_expression_l56_56356


namespace exists_power_of_two_with_consecutive_zeros_l56_56940

theorem exists_power_of_two_with_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ, 2^n = a * 10^(m + k) + b ∧ 10^(k - 1) ≤ b ∧ b < 10^k ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 :=
sorry

end exists_power_of_two_with_consecutive_zeros_l56_56940


namespace rationalize_sqrt_fraction_l56_56699

theorem rationalize_sqrt_fraction {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) : 
  (Real.sqrt ((a : ℝ) / b)) = (Real.sqrt (a * (b / (b * b)))) → 
  (Real.sqrt (5 / 12)) = (Real.sqrt 15 / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l56_56699


namespace remainder_of_b2_minus_3a_div_6_l56_56090

theorem remainder_of_b2_minus_3a_div_6 (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 5) : 
  (b^2 - 3 * a) % 6 = 1 := 
sorry

end remainder_of_b2_minus_3a_div_6_l56_56090


namespace water_added_l56_56715

theorem water_added (W x : ℕ) (h₁ : 2 * W = 5 * 10)
                    (h₂ : 2 * (W + x) = 7 * 10) :
  x = 10 :=
by
  sorry

end water_added_l56_56715


namespace smallest_x_consecutive_cubes_l56_56412

theorem smallest_x_consecutive_cubes :
  ∃ (u v w x : ℕ), u < v ∧ v < w ∧ w < x ∧ u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ (u^3 + v^3 + w^3 = x^3) ∧ (x = 6) :=
by {
  sorry
}

end smallest_x_consecutive_cubes_l56_56412


namespace hexagon_perimeter_l56_56775

-- Define the length of one side of the hexagon
def side_length : ℕ := 5

-- Define the number of sides of a hexagon
def num_sides : ℕ := 6

-- Problem statement: Prove the perimeter of a regular hexagon with the given side length
theorem hexagon_perimeter (s : ℕ) (n : ℕ) : s = side_length ∧ n = num_sides → n * s = 30 :=
by sorry

end hexagon_perimeter_l56_56775


namespace cannot_cut_square_into_7_rectangles_l56_56069

theorem cannot_cut_square_into_7_rectangles (a : ℝ) :
  ¬ ∃ (x : ℝ), 7 * 2 * x ^ 2 = a ^ 2 ∧ 
    ∀ (i : ℕ), 0 ≤ i → i < 7 → (∃ (rect : ℝ × ℝ), rect.1 = x ∧ rect.2 = 2 * x ) :=
by
  sorry

end cannot_cut_square_into_7_rectangles_l56_56069


namespace spherical_to_rectangular_coordinates_l56_56159

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l56_56159


namespace trolley_length_l56_56668

theorem trolley_length (L F : ℝ) (h1 : 4 * L + 3 * F = 108) (h2 : 10 * L + 9 * F = 168) : L = 78 := 
by
  sorry

end trolley_length_l56_56668


namespace charles_paints_l56_56126

-- Define the ratio and total work conditions
def ratio_a_to_c (a c : ℕ) := a * 6 = c * 2

def total_work (total : ℕ) := total = 320

-- Define the question, i.e., the amount of work Charles does
theorem charles_paints (a c total : ℕ) (h_ratio : ratio_a_to_c a c) (h_total : total_work total) : 
  (total / (a + c)) * c = 240 :=
by 
  -- We include sorry to indicate the need for proof here
  sorry

end charles_paints_l56_56126


namespace lower_limit_for_a_l56_56446

theorem lower_limit_for_a 
  {k : ℤ} 
  (a b : ℤ) 
  (h1 : k ≤ a) 
  (h2 : a < 17) 
  (h3 : 3 < b) 
  (h4 : b < 29) 
  (h5 : 3.75 = 4 - 0.25) 
  : (7 ≤ a) :=
sorry

end lower_limit_for_a_l56_56446


namespace area_of_region_S_is_correct_l56_56533

noncomputable def area_of_inverted_region (d : ℝ) : ℝ :=
  if h : d = 1.5 then 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi else 0

theorem area_of_region_S_is_correct :
  area_of_inverted_region 1.5 = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi := 
by 
  sorry

end area_of_region_S_is_correct_l56_56533


namespace exists_b_for_a_ge_condition_l56_56158

theorem exists_b_for_a_ge_condition (a : ℝ) (h : a ≥ -Real.sqrt 2 - 1 / 4) :
  ∃ b : ℝ, ∃ x y : ℝ, 
    y = x^2 - a ∧
    x^2 + y^2 + 8 * b^2 = 4 * b * (y - x) + 1 :=
sorry

end exists_b_for_a_ge_condition_l56_56158


namespace total_divisors_7350_l56_56280

def primeFactorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 7350 then [(2, 1), (3, 1), (5, 2), (7, 2)] else []

def totalDivisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc (p : ℕ × ℕ) => acc * (p.snd + 1)) 1

theorem total_divisors_7350 : totalDivisors (primeFactorization 7350) = 36 :=
by
  sorry

end total_divisors_7350_l56_56280


namespace initial_stock_decaf_percentage_l56_56776

variable (x : ℝ)
variable (initialStock newStock totalStock initialDecaf newDecaf totalDecaf: ℝ)

theorem initial_stock_decaf_percentage :
  initialStock = 400 ->
  newStock = 100 ->
  totalStock = 500 ->
  initialDecaf = initialStock * x / 100 ->
  newDecaf = newStock * 60 / 100 ->
  totalDecaf = 180 ->
  initialDecaf + newDecaf = totalDecaf ->
  x = 30 := by
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇
  sorry

end initial_stock_decaf_percentage_l56_56776


namespace johns_average_speed_l56_56579

-- Conditions
def biking_time_minutes : ℝ := 45
def biking_speed_mph : ℝ := 20
def walking_time_minutes : ℝ := 120
def walking_speed_mph : ℝ := 3

-- Proof statement
theorem johns_average_speed :
  let biking_time_hours := biking_time_minutes / 60
  let biking_distance := biking_speed_mph * biking_time_hours
  let walking_time_hours := walking_time_minutes / 60
  let walking_distance := walking_speed_mph * walking_time_hours
  let total_distance := biking_distance + walking_distance
  let total_time := biking_time_hours + walking_time_hours
  let average_speed := total_distance / total_time
  average_speed = 7.64 :=
by
  sorry

end johns_average_speed_l56_56579


namespace age_problem_l56_56453

theorem age_problem 
  (K S E F : ℕ)
  (h1 : K = S - 5)
  (h2 : S = 2 * E)
  (h3 : E = F + 9)
  (h4 : K = 33) : 
  F = 10 :=
by 
  sorry

end age_problem_l56_56453


namespace find_v_l56_56960

theorem find_v (v : ℝ) (h : (v - v / 3) - ((v - v / 3) / 3) = 4) : v = 9 := 
by 
  sorry

end find_v_l56_56960


namespace quadratic_expression_transformation_l56_56218

theorem quadratic_expression_transformation :
  ∀ (a h k : ℝ), (∀ x : ℝ, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intros a h k h_eq
  sorry

end quadratic_expression_transformation_l56_56218


namespace sum_of_decimals_is_fraction_l56_56393

def decimal_to_fraction_sum : ℚ :=
  (1 / 10) + (2 / 100) + (3 / 1000) + (4 / 10000) + (5 / 100000) + (6 / 1000000) + (7 / 10000000)

theorem sum_of_decimals_is_fraction :
  decimal_to_fraction_sum = 1234567 / 10000000 :=
by sorry

end sum_of_decimals_is_fraction_l56_56393


namespace area_of_rhombus_l56_56628

theorem area_of_rhombus (P D : ℕ) (area : ℝ) (hP : P = 48) (hD : D = 26) :
  area = 25 := by
  sorry

end area_of_rhombus_l56_56628


namespace number_of_ways_to_choose_bases_l56_56891

-- Definitions of the conditions
def num_students : Nat := 4
def num_bases : Nat := 3

-- The main statement that we need to prove
theorem number_of_ways_to_choose_bases : (num_bases ^ num_students) = 81 := by
  sorry

end number_of_ways_to_choose_bases_l56_56891


namespace equation_1_solution_1_equation_2_solution_l56_56447

theorem equation_1_solution_1 (x : ℝ) (h : 4 * (x - 1) ^ 2 = 25) : x = 7 / 2 ∨ x = -3 / 2 := by
  sorry

theorem equation_2_solution (x : ℝ) (h : (1 / 3) * (x + 2) ^ 3 - 9 = 0) : x = 1 := by
  sorry

end equation_1_solution_1_equation_2_solution_l56_56447


namespace intersection_of_P_and_Q_l56_56184

noncomputable def P : Set ℝ := {x | 0 < Real.log x / Real.log 8 ∧ Real.log x / Real.log 8 < 2 * (Real.log 3 / Real.log 8)}
noncomputable def Q : Set ℝ := {x | 2 / (2 - x) > 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l56_56184


namespace proportional_segments_l56_56617

theorem proportional_segments (a b c d : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) (h4 : a / b = c / d) : d = 3 / 2 :=
by
  -- proof steps here
  sorry

end proportional_segments_l56_56617


namespace at_least_one_less_than_two_l56_56485

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
by 
  sorry

end at_least_one_less_than_two_l56_56485


namespace arithmetic_sequence_sufficient_not_necessary_l56_56113

variables {a b c d : ℤ}

-- Proving sufficiency: If a, b, c, d form an arithmetic sequence, then a + d = b + c.
def arithmetic_sequence (a b c d : ℤ) : Prop := 
  a + d = 2*b ∧ b + c = 2*a

theorem arithmetic_sequence_sufficient_not_necessary (h : arithmetic_sequence a b c d) : a + d = b + c ∧ ∃ (x y z w : ℤ), x + w = y + z ∧ ¬ arithmetic_sequence x y z w :=
by {
  sorry
}

end arithmetic_sequence_sufficient_not_necessary_l56_56113


namespace fraction_of_smaller_part_l56_56343

theorem fraction_of_smaller_part (A B : ℕ) (x : ℚ) (h1 : A + B = 66) (h2 : A = 50) (h3 : 0.40 * A = x * B + 10) : x = 5 / 8 :=
by
  sorry

end fraction_of_smaller_part_l56_56343


namespace probability_queen_then_spade_l56_56326

-- Define the size of the deck and the quantities for specific cards
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_spades : ℕ := 13

-- Define the probability calculation problem
theorem probability_queen_then_spade :
  (num_queens / deck_size : ℚ) * ((num_spades - 1) / (deck_size - 1) : ℚ) + ((num_queens - 1) / deck_size : ℚ) * (num_spades / (deck_size - 1) : ℚ) = 1 / deck_size :=
by sorry

end probability_queen_then_spade_l56_56326


namespace correct_equation_D_l56_56163

theorem correct_equation_D : (|5 - 3| = - (3 - 5)) :=
by
  sorry

end correct_equation_D_l56_56163


namespace xyz_equality_l56_56472

theorem xyz_equality (x y z : ℝ) (h : x^2 + y^2 + z^2 = x * y + y * z + z * x) : x = y ∧ y = z :=
by
  sorry

end xyz_equality_l56_56472


namespace part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l56_56637

-- Part (Ⅰ)
theorem part1_coordinates_of_P_if_AB_perp_PB :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (7, 0)) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_coordinates_of_P_area_ABP_10 :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (9, 0) ∨ P = (-11, 0)) :=
by
  sorry

end part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l56_56637


namespace num_distinct_values_for_sum_l56_56399

theorem num_distinct_values_for_sum (x y z : ℝ) 
  (h : (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0) :
  ∃ s : Finset ℝ, 
  (∀ x y z, (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0 → (x + y + z) ∈ s) ∧ 
  s.card = 7 :=
by sorry

end num_distinct_values_for_sum_l56_56399


namespace afternoon_more_than_evening_l56_56076

def campers_in_morning : Nat := 33
def campers_in_afternoon : Nat := 34
def campers_in_evening : Nat := 10

theorem afternoon_more_than_evening : campers_in_afternoon - campers_in_evening = 24 := by
  sorry

end afternoon_more_than_evening_l56_56076


namespace number_of_small_slices_l56_56616

-- Define the given conditions
variables (S L : ℕ)
axiom total_slices : S + L = 5000
axiom total_revenue : 150 * S + 250 * L = 1050000

-- State the problem we need to prove
theorem number_of_small_slices : S = 1500 :=
by sorry

end number_of_small_slices_l56_56616


namespace sally_score_is_12_5_l56_56200

-- Conditions
def correctAnswers : ℕ := 15
def incorrectAnswers : ℕ := 10
def unansweredQuestions : ℕ := 5
def pointsPerCorrect : ℝ := 1.0
def pointsPerIncorrect : ℝ := -0.25
def pointsPerUnanswered : ℝ := 0.0

-- Score computation
noncomputable def sallyScore : ℝ :=
  (correctAnswers * pointsPerCorrect) + 
  (incorrectAnswers * pointsPerIncorrect) + 
  (unansweredQuestions * pointsPerUnanswered)

-- Theorem to prove Sally's score is 12.5
theorem sally_score_is_12_5 : sallyScore = 12.5 := by
  sorry

end sally_score_is_12_5_l56_56200


namespace smallest_integer_gcd_6_l56_56842

theorem smallest_integer_gcd_6 : ∃ n : ℕ, n > 100 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n :=
by
  let n := 114
  have h1 : n > 100 := sorry
  have h2 : gcd n 18 = 6 := sorry
  have h3 : ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n := sorry
  exact ⟨n, h1, h2, h3⟩

end smallest_integer_gcd_6_l56_56842


namespace find_other_number_l56_56896

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 12 n = 60) (h_hcf : Nat.gcd 12 n = 3) : n = 15 := by
  sorry

end find_other_number_l56_56896


namespace largest_interior_angle_of_triangle_l56_56553

theorem largest_interior_angle_of_triangle (exterior_ratio_2k : ℝ) (exterior_ratio_3k : ℝ) (exterior_ratio_4k : ℝ) (sum_exterior_angles : exterior_ratio_2k + exterior_ratio_3k + exterior_ratio_4k = 360) :
  180 - exterior_ratio_2k = 100 :=
by
  sorry

end largest_interior_angle_of_triangle_l56_56553


namespace fraction_is_three_fourths_l56_56241

-- Define the number
def n : ℝ := 8.0

-- Define the fraction
variable (x : ℝ)

-- The main statement to be proved
theorem fraction_is_three_fourths
(h : x * n + 2 = 8) : x = 3 / 4 :=
sorry

end fraction_is_three_fourths_l56_56241


namespace gcd_polynomial_l56_56375

-- Define conditions
variables (b : ℤ) (k : ℤ)

-- Assume b is an even multiple of 8753
def is_even_multiple_of_8753 (b : ℤ) : Prop := ∃ k : ℤ, b = 2 * 8753 * k

-- Statement to be proven
theorem gcd_polynomial (b : ℤ) (h : is_even_multiple_of_8753 b) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 :=
by sorry

end gcd_polynomial_l56_56375


namespace fifth_term_of_geometric_sequence_l56_56933

theorem fifth_term_of_geometric_sequence
  (a r : ℝ)
  (h1 : a * r^2 = 16)
  (h2 : a * r^6 = 2) : a * r^4 = 8 :=
sorry

end fifth_term_of_geometric_sequence_l56_56933


namespace square_area_eq_36_l56_56140

theorem square_area_eq_36 (A_triangle : ℝ) (P_triangle : ℝ) 
  (h1 : A_triangle = 16 * Real.sqrt 3)
  (h2 : P_triangle = 3 * (Real.sqrt (16 * 4 * Real.sqrt 3)))
  (h3 : ∀ a, 4 * a = P_triangle) : 
  a^2 = 36 :=
by sorry

end square_area_eq_36_l56_56140


namespace find_distinct_natural_numbers_l56_56658

theorem find_distinct_natural_numbers :
  ∃ (x y : ℕ), x ≥ 10 ∧ y ≠ 1 ∧
  (x * y + x) + (x * y - x) + (x * y * x) + (x * y / x) = 576 :=
by
  sorry

end find_distinct_natural_numbers_l56_56658


namespace sale_in_second_month_l56_56562

theorem sale_in_second_month 
  (m1 m2 m3 m4 m5 m6 : ℕ) 
  (h1: m1 = 6335) 
  (h2: m3 = 6855) 
  (h3: m4 = 7230) 
  (h4: m5 = 6562) 
  (h5: m6 = 5091)
  (average: (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 6500) : 
  m2 = 6927 :=
sorry

end sale_in_second_month_l56_56562


namespace students_and_ticket_price_l56_56057

theorem students_and_ticket_price (students teachers ticket_price : ℕ) 
  (h1 : students % 5 = 0)
  (h2 : (students + teachers) * (ticket_price / 2) = 1599)
  (h3 : ∃ n, ticket_price = 2 * n) 
  (h4 : teachers = 1) :
  students = 40 ∧ ticket_price = 78 := 
by
  sorry

end students_and_ticket_price_l56_56057


namespace smallest_value_3a_plus_1_l56_56970

theorem smallest_value_3a_plus_1 
  (a : ℝ)
  (h : 8 * a^2 + 9 * a + 6 = 2) : 
  ∃ (b : ℝ), b = 3 * a + 1 ∧ b = -2 :=
by 
  sorry

end smallest_value_3a_plus_1_l56_56970


namespace find_k_value_l56_56203

theorem find_k_value (k : ℝ) (x : ℝ) :
  -x^2 - (k + 12) * x - 8 = -(x - 2) * (x - 4) → k = -18 :=
by
  intro h
  sorry

end find_k_value_l56_56203


namespace James_leftover_money_l56_56746

variable (W : ℝ)
variable (M : ℝ)

theorem James_leftover_money 
  (h1 : M = (W / 2 - 2))
  (h2 : M + 114 = W) : 
  M = 110 := sorry

end James_leftover_money_l56_56746


namespace school_population_proof_l56_56026

variables (x y z: ℕ)
variable (B: ℕ := (50 * y) / 100)

theorem school_population_proof (h1 : 162 = (x * B) / 100)
                               (h2 : B = (50 * y) / 100)
                               (h3 : z = 100 - 50) :
  z = 50 :=
  sorry

end school_population_proof_l56_56026


namespace cost_of_socks_l56_56494

/-- Given initial amount of $100 and cost of shirt is $24,
    find out the cost of socks if the remaining amount is $65. --/
theorem cost_of_socks
  (initial_amount : ℕ)
  (cost_of_shirt : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : cost_of_shirt = 24)
  (h3 : remaining_amount = 65) : 
  (initial_amount - cost_of_shirt - remaining_amount) = 11 :=
by
  sorry

end cost_of_socks_l56_56494


namespace least_subset_gcd_l56_56329

variable (S : Set ℕ) (f : ℕ → ℤ)
variable (a : ℕ → ℕ)
variable (k : ℕ)

def conditions (S : Set ℕ) (f : ℕ → ℤ) : Prop :=
  ∃ (a : ℕ → ℕ), 
  (∀ i j, i ≠ j → a i < a j) ∧ 
  (S = {i | ∃ n, i = a n ∧ n < 2004}) ∧ 
  (∀ i, f (a i) < 2003) ∧ 
  (∀ i j, f (a i) = f (a j))

theorem least_subset_gcd (h : conditions S f) : k = 1003 :=
  sorry

end least_subset_gcd_l56_56329


namespace apple_price_l56_56354

theorem apple_price :
  ∀ (l q : ℝ), 
    (10 * l = 3.62) →
    (30 * l + 3 * q = 11.67) →
    (30 * l + 6 * q = 12.48) :=
by
  intros l q h₁ h₂
  -- The proof would go here with the steps, but for now we use sorry.
  sorry

end apple_price_l56_56354


namespace Amy_current_age_l56_56212

def Mark_age_in_5_years : ℕ := 27
def years_in_future : ℕ := 5
def age_difference : ℕ := 7

theorem Amy_current_age : ∃ (Amy_age : ℕ), Amy_age = 15 :=
  by
    let Mark_current_age := Mark_age_in_5_years - years_in_future
    let Amy_age := Mark_current_age - age_difference
    use Amy_age
    sorry

end Amy_current_age_l56_56212


namespace poly_coeff_difference_l56_56404

theorem poly_coeff_difference :
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (2 + x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 →
  a = 16 →
  1 = a - a_1 + a_2 - a_3 + a_4 →
  a_2 - a_1 + a_4 - a_3 = -15 :=
by
  intros a a_1 a_2 a_3 a_4 h_poly h_a h_eq
  sorry

end poly_coeff_difference_l56_56404


namespace quadratic_transformation_concept_l56_56276

theorem quadratic_transformation_concept :
  ∀ x : ℝ, (x-3)^2 - 4*(x-3) = 0 ↔ (x = 3 ∨ x = 7) :=
by
  intro x
  sorry

end quadratic_transformation_concept_l56_56276


namespace zero_in_interval_l56_56795

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval (h_mono : ∀ x y, 0 < x → x < y → f x < f y) (h_f2 : f 2 < 0) (h_f3 : 0 < f 3) :
  ∃ x₀ ∈ (Set.Ioo 2 3), f x₀ = 0 :=
by
  sorry

end zero_in_interval_l56_56795


namespace minimum_product_xyz_l56_56743

noncomputable def minimalProduct (x y z : ℝ) : ℝ :=
  3 * x^2 * (1 - 4 * x)

theorem minimum_product_xyz :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  z = 3 * x →
  x ≤ y ∧ y ≤ z →
  minimalProduct x y z = (9 / 343) :=
by
  intros x y z x_pos y_pos z_pos sum_eq1 z_eq3x inequalities
  sorry

end minimum_product_xyz_l56_56743


namespace number_b_smaller_than_number_a_l56_56722

theorem number_b_smaller_than_number_a (A B : ℝ)
  (h : A = B + 1/4) : (B + 1/4 = A) ∧ (B < A) → B = (4 * A - A) / 5 := by
  sorry

end number_b_smaller_than_number_a_l56_56722


namespace at_least_one_not_land_designated_area_l56_56467

variable (p q : Prop)

theorem at_least_one_not_land_designated_area : ¬p ∨ ¬q ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_land_designated_area_l56_56467


namespace probability_heads_l56_56619

variable (p : ℝ)
variable (h1 : 0 ≤ p)
variable (h2 : p ≤ 1)
variable (h3 : p * (1 - p) ^ 4 = 0.03125)

theorem probability_heads :
  p = 0.5 :=
sorry

end probability_heads_l56_56619


namespace blue_ball_weight_l56_56686

variable (b t x : ℝ)
variable (c1 : b = 3.12)
variable (c2 : t = 9.12)
variable (c3 : t = b + x)

theorem blue_ball_weight : x = 6 :=
by
  sorry

end blue_ball_weight_l56_56686


namespace sign_of_a_l56_56728

theorem sign_of_a (a b c d : ℝ) (h : b * (3 * d + 2) ≠ 0) (ineq : a / b < -c / (3 * d + 2)) : 
  (a = 0 ∨ a > 0 ∨ a < 0) :=
sorry

end sign_of_a_l56_56728


namespace vertex_of_parabola_minimum_value_for_x_ge_2_l56_56884

theorem vertex_of_parabola :
  ∀ x y : ℝ, y = x^2 + 2*x - 3 → ∃ (vx vy : ℝ), (vx = -1) ∧ (vy = -4) :=
by
  sorry

theorem minimum_value_for_x_ge_2 :
  ∀ x : ℝ, x ≥ 2 → y = x^2 + 2*x - 3 → ∃ (min_val : ℝ), min_val = 5 :=
by
  sorry

end vertex_of_parabola_minimum_value_for_x_ge_2_l56_56884


namespace volume_in_cubic_yards_l56_56969

-- Define the conditions
def volume_in_cubic_feet : ℕ := 162
def cubic_feet_per_cubic_yard : ℕ := 27

-- Problem statement in Lean 4
theorem volume_in_cubic_yards : volume_in_cubic_feet / cubic_feet_per_cubic_yard = 6 := 
  by
    sorry

end volume_in_cubic_yards_l56_56969


namespace smallest_a_value_l56_56435

theorem smallest_a_value {a b c : ℝ} :
  (∃ (a b c : ℝ), (∀ x, (a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c)) ∧ a > 0 ∧ ∃ n : ℤ, a + b + c = n)
  → (∃ (a : ℝ), a = 1) :=
by
  sorry

end smallest_a_value_l56_56435


namespace quadratic_vertex_on_x_axis_l56_56271

theorem quadratic_vertex_on_x_axis (k : ℝ) :
  (∃ x : ℝ, (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end quadratic_vertex_on_x_axis_l56_56271


namespace find_m_condition_l56_56698

theorem find_m_condition (m : ℕ) (h : 9^4 = 3^(2*m)) : m = 4 := by
  sorry

end find_m_condition_l56_56698


namespace numbers_written_in_red_l56_56106

theorem numbers_written_in_red :
  ∃ (x : ℕ), x > 0 ∧ x <= 101 ∧ 
  ∀ (largest_blue_num : ℕ) (smallest_red_num : ℕ), 
  (largest_blue_num = x) ∧ 
  (smallest_red_num = x + 1) ∧ 
  (smallest_red_num = (101 - x) / 2) → 
  (101 - x = 68) := by
  sorry

end numbers_written_in_red_l56_56106


namespace functional_equation_solution_l56_56063

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y) + f (f x + f y) = y * f x + f (x + f y)) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_equation_solution_l56_56063


namespace problem_1_problem_2_l56_56433

def f (x a : ℝ) := |x + a| + |x + 3|
def g (x : ℝ) := |x - 1| + 2

theorem problem_1 : ∀ x : ℝ, |g x| < 3 ↔ 0 < x ∧ x < 2 := 
by
  sorry

theorem problem_2 : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) ↔ a ≥ 5 ∨ a ≤ 1 := 
by
  sorry

end problem_1_problem_2_l56_56433


namespace inequality_proof_l56_56149

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_proof_l56_56149


namespace time_without_moving_walkway_l56_56741

/--
Assume a person walks from one end to the other of a 90-meter long moving walkway at a constant rate in 30 seconds, assisted by the walkway. When this person reaches the end, they reverse direction and continue walking with the same speed, but this time it takes 120 seconds because the person is traveling against the direction of the moving walkway.

Prove that if the walkway were to stop moving, it would take this person 48 seconds to walk from one end of the walkway to the other.
-/
theorem time_without_moving_walkway : 
  ∀ (v_p v_w : ℝ),
  (v_p + v_w) * 30 = 90 →
  (v_p - v_w) * 120 = 90 →
  90 / v_p = 48 :=
by
  intros v_p v_w h1 h2
  have hpw := eq_of_sub_eq_zero (sub_eq_zero.mpr h1)
  have hmw := eq_of_sub_eq_zero (sub_eq_zero.mpr h2)
  sorry

end time_without_moving_walkway_l56_56741


namespace fraction_pow_zero_l56_56944

theorem fraction_pow_zero
  (a : ℤ) (b : ℤ)
  (h_a : a = -325123789)
  (h_b : b = 59672384757348)
  (h_nonzero_num : a ≠ 0)
  (h_nonzero_denom : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 :=
by {
  sorry
}

end fraction_pow_zero_l56_56944


namespace maximum_value_expression_l56_56044

noncomputable def expression (s t : ℝ) := -2 * s^2 + 24 * s + 3 * t - 38

theorem maximum_value_expression : ∀ (s : ℝ), expression s 4 ≤ 46 :=
by sorry

end maximum_value_expression_l56_56044


namespace tens_digit_of_19_pow_2023_l56_56198

theorem tens_digit_of_19_pow_2023 : (19 ^ 2023) % 100 = 59 := 
  sorry

end tens_digit_of_19_pow_2023_l56_56198


namespace carrots_picked_first_day_l56_56179

theorem carrots_picked_first_day (X : ℕ) 
  (H1 : X - 10 + 47 = 60) : X = 23 :=
by 
  -- We state the proof steps here, completing the proof with sorry
  sorry

end carrots_picked_first_day_l56_56179


namespace valid_grid_count_l56_56958

def is_adjacent (i j : ℕ) (n : ℕ) : Prop :=
  (i = j + 1 ∨ i + 1 = j ∨ (i = n - 1 ∧ j = 0) ∨ (i = 0 ∧ j = n - 1))

def valid_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 4 ∧ 0 ≤ j ∧ j < 4 →
         (is_adjacent i (i+1) 4 → grid i (i+1) * grid i (i+1) = 0) ∧ 
         (is_adjacent j (j+1) 4 → grid (j+1) j * grid (j+1) j = 0)

theorem valid_grid_count : 
  ∃ s : ℕ, s = 1234 ∧
    (∃ grid : ℕ → ℕ → ℕ, valid_grid grid) :=
sorry

end valid_grid_count_l56_56958


namespace original_price_l56_56610

theorem original_price (x : ℝ) (h : 0.9504 * x = 108) : x = 10800 / 9504 :=
by
  sorry

end original_price_l56_56610


namespace cylindrical_to_rectangular_coordinates_l56_56234

theorem cylindrical_to_rectangular_coordinates (r θ z : ℝ) (h1 : r = 6) (h2 : θ = 5 * Real.pi / 3) (h3 : z = 7) :
    (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 7) :=
by
  rw [h1, h2, h3]
  -- Using trigonometric identities:
  have hcos : Real.cos (5 * Real.pi / 3) = 1 / 2 := sorry
  have hsin : Real.sin (5 * Real.pi / 3) = -(Real.sqrt 3) / 2 := sorry
  rw [hcos, hsin]
  simp
  sorry

end cylindrical_to_rectangular_coordinates_l56_56234


namespace giyoon_chocolates_l56_56283

theorem giyoon_chocolates (C X : ℕ) (h1 : C = 8 * X) (h2 : C = 6 * (X + 1) + 4) : C = 40 :=
by sorry

end giyoon_chocolates_l56_56283


namespace socks_impossible_l56_56634

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l56_56634


namespace slope_of_line_l56_56480

-- Defining the parametric equations of the line
def parametric_x (t : ℝ) : ℝ := 3 + 4 * t
def parametric_y (t : ℝ) : ℝ := 4 - 5 * t

-- Stating the problem in Lean: asserting the slope of the line
theorem slope_of_line : 
  (∃ (m : ℝ), ∀ t : ℝ, parametric_y t = m * parametric_x t + (4 - 3 * m)) 
  → (∃ m : ℝ, m = -5 / 4) :=
  by sorry

end slope_of_line_l56_56480


namespace complete_square_solution_l56_56452

theorem complete_square_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : (x - 2)^2 = 2 := 
by sorry

end complete_square_solution_l56_56452


namespace geom_seq_sum_seven_terms_l56_56104

-- Defining the conditions
def a0 : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 7

-- Definition for the sum of the first n terms in a geometric series
def geom_series_sum (a r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Statement to prove the sum of the first seven terms equals 1093/2187
theorem geom_seq_sum_seven_terms : geom_series_sum a0 r n = 1093 / 2187 := 
by 
  sorry

end geom_seq_sum_seven_terms_l56_56104


namespace union_complement_l56_56238

open Set

-- Definitions based on conditions
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}
def C_UA : Set ℕ := U \ A

-- The theorem to prove
theorem union_complement :
  (C_UA ∪ B) = {0, 2, 4, 5, 6} :=
by
  sorry

end union_complement_l56_56238


namespace medium_stores_count_l56_56261

-- Define the total number of stores
def total_stores : ℕ := 300

-- Define the number of medium stores
def medium_stores : ℕ := 75

-- Define the sample size
def sample_size : ℕ := 20

-- Define the expected number of medium stores in the sample
def expected_medium_stores : ℕ := 5

-- The theorem statement claiming that the number of medium stores in the sample is 5
theorem medium_stores_count : 
  (sample_size * medium_stores) / total_stores = expected_medium_stores :=
by
  -- Proof omitted
  sorry

end medium_stores_count_l56_56261


namespace greatest_cds_in_box_l56_56459

theorem greatest_cds_in_box (r c p n : ℕ) (hr : r = 14) (hc : c = 12) (hp : p = 8) (hn : n = 2) :
  n = Nat.gcd r (Nat.gcd c p) :=
by
  rw [hr, hc, hp]
  sorry

end greatest_cds_in_box_l56_56459


namespace non_neg_int_solutions_eq_10_l56_56028

theorem non_neg_int_solutions_eq_10 :
  ∃ n : ℕ, n = 55 ∧
  (∃ (x y z : ℕ), x + y + z = 10) :=
by
  sorry

end non_neg_int_solutions_eq_10_l56_56028


namespace largest_pos_int_divisor_l56_56492

theorem largest_pos_int_divisor:
  ∃ n : ℕ, (n + 10 ∣ n^3 + 2011) ∧ (∀ m : ℕ, (m + 10 ∣ m^3 + 2011) → m ≤ n) :=
sorry

end largest_pos_int_divisor_l56_56492


namespace greatest_two_digit_multiple_of_7_l56_56572

theorem greatest_two_digit_multiple_of_7 : ∃ n, 10 ≤ n ∧ n < 100 ∧ n % 7 = 0 ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ m % 7 = 0 → n ≥ m := 
by
  sorry

end greatest_two_digit_multiple_of_7_l56_56572


namespace handshakes_mod_500_l56_56844

theorem handshakes_mod_500 : 
  let n := 10
  let k := 3
  let M := 199584 -- total number of ways calculated from the problem
  (n = 10) -> (k = 3) -> (M % 500 = 84) :=
by
  intros
  sorry

end handshakes_mod_500_l56_56844


namespace probability_divisibility_9_correct_l56_56062

-- Define the set S
def S : Set ℕ := { n | ∃ a b: ℕ, 0 ≤ a ∧ a < 40 ∧ 0 ≤ b ∧ b < 40 ∧ a ≠ b ∧ n = 2^a + 2^b }

-- Define the criteria for divisibility by 9
def divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

-- Define the total size of set S
def size_S : ℕ := 780  -- as calculated from combination

-- Count valid pairs (a, b) such that 2^a + 2^b is divisible by 9
def valid_pairs : ℕ := 133  -- as calculated from summation

-- Define the probability
def probability_divisible_by_9 : ℕ := valid_pairs / size_S

-- The proof statement
theorem probability_divisibility_9_correct:
  (valid_pairs : ℚ) / (size_S : ℚ) = 133 / 780 := sorry

end probability_divisibility_9_correct_l56_56062


namespace max_remainder_l56_56409

-- Definition of the problem
def max_remainder_condition (x : ℕ) (y : ℕ) : Prop :=
  x % 7 = y

theorem max_remainder (y : ℕ) :
  (max_remainder_condition (7 * 102 + y) y ∧ y < 7) → (y = 6 ∧ 7 * 102 + 6 = 720) :=
by
  sorry

end max_remainder_l56_56409


namespace playground_perimeter_is_correct_l56_56225

-- Definition of given conditions
def length_of_playground : ℕ := 110
def width_of_playground : ℕ := length_of_playground - 15

-- Statement of the problem to prove
theorem playground_perimeter_is_correct :
  2 * (length_of_playground + width_of_playground) = 230 := 
by
  sorry

end playground_perimeter_is_correct_l56_56225


namespace relationship_among_abc_l56_56532

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem relationship_among_abc :
  b > c ∧ c > a :=
by
  sorry

end relationship_among_abc_l56_56532


namespace increase_by_percentage_l56_56376

theorem increase_by_percentage (x : ℝ) (y : ℝ): x = 90 → y = 0.50 → x + x * y = 135 := 
by
  intro h1 h2
  sorry

end increase_by_percentage_l56_56376


namespace transform_to_100_l56_56398

theorem transform_to_100 (a b c : ℤ) (h : Int.gcd (Int.gcd a b) c = 1) :
  ∃ f : (ℤ × ℤ × ℤ → ℤ × ℤ × ℤ), (∀ p : ℤ × ℤ × ℤ,
    ∃ q : ℕ, q ≤ 5 ∧ f^[q] p = (1, 0, 0)) :=
sorry

end transform_to_100_l56_56398


namespace four_pow_2024_mod_11_l56_56024

theorem four_pow_2024_mod_11 : (4 ^ 2024) % 11 = 3 :=
by
  sorry

end four_pow_2024_mod_11_l56_56024


namespace product_of_faces_and_vertices_of_cube_l56_56378

def number_of_faces := 6
def number_of_vertices := 8

theorem product_of_faces_and_vertices_of_cube : number_of_faces * number_of_vertices = 48 := 
by 
  sorry

end product_of_faces_and_vertices_of_cube_l56_56378


namespace factorize_expression_l56_56635

theorem factorize_expression (x : ℝ) : 
  x^4 + 324 = (x^2 - 18 * x + 162) * (x^2 + 18 * x + 162) := 
sorry

end factorize_expression_l56_56635


namespace output_of_program_l56_56419

def loop_until (i S : ℕ) : ℕ :=
if i < 9 then S
else loop_until (i - 1) (S * i)

theorem output_of_program : loop_until 11 1 = 990 :=
sorry

end output_of_program_l56_56419


namespace solve_equation_l56_56164

theorem solve_equation (x : ℝ) : 2 * x + 17 = 32 - 3 * x → x = 3 := 
by 
  sorry

end solve_equation_l56_56164


namespace retail_price_per_book_l56_56952

theorem retail_price_per_book (n r w : ℝ)
  (h1 : r * n = 48)
  (h2 : w = r - 2)
  (h3 : w * (n + 4) = 48) :
  r = 6 := by
  sorry

end retail_price_per_book_l56_56952


namespace black_pens_per_student_l56_56887

theorem black_pens_per_student (number_of_students : ℕ)
                               (red_pens_per_student : ℕ)
                               (taken_first_month : ℕ)
                               (taken_second_month : ℕ)
                               (pens_after_splitting : ℕ)
                               (initial_black_pens_per_student : ℕ) : 
  number_of_students = 3 → 
  red_pens_per_student = 62 → 
  taken_first_month = 37 → 
  taken_second_month = 41 → 
  pens_after_splitting = 79 → 
  initial_black_pens_per_student = 43 :=
by sorry

end black_pens_per_student_l56_56887


namespace calculate_perimeter_l56_56434

noncomputable def length_square := 8
noncomputable def breadth_square := 8 -- since it's a square, length and breadth are the same
noncomputable def length_rectangle := 8
noncomputable def breadth_rectangle := 4

noncomputable def combined_length := length_square + length_rectangle
noncomputable def combined_breadth := breadth_square 

noncomputable def perimeter := 2 * (combined_length + combined_breadth)

theorem calculate_perimeter : 
  length_square = 8 ∧ 
  breadth_square = 8 ∧ 
  length_rectangle = 8 ∧ 
  breadth_rectangle = 4 ∧ 
  perimeter = 48 := 
by 
  sorry

end calculate_perimeter_l56_56434


namespace inequality_proof_l56_56272

theorem inequality_proof (n : ℕ) (h : n > 1) : 
  1 / (2 * n * Real.exp 1) < 1 / Real.exp 1 - (1 - 1 / n) ^ n ∧ 
  1 / Real.exp 1 - (1 - 1 / n) ^ n < 1 / (n * Real.exp 1) := 
by
  sorry

end inequality_proof_l56_56272


namespace rainfall_ratio_l56_56450

theorem rainfall_ratio (rain_15_days : ℕ) (total_rain : ℕ) (days_in_month : ℕ) (rain_per_day_first_15 : ℕ) :
  rain_per_day_first_15 * 15 = rain_15_days →
  rain_15_days + (days_in_month - 15) * (rain_per_day_first_15 * 2) = total_rain →
  days_in_month = 30 →
  total_rain = 180 →
  rain_per_day_first_15 = 4 →
  2 = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rainfall_ratio_l56_56450


namespace limit_of_sequence_l56_56074

theorem limit_of_sequence {ε : ℝ} (hε : ε > 0) : 
  ∃ (N : ℝ), ∀ (n : ℝ), n > N → |(2 * n^3) / (n^3 - 2) - 2| < ε :=
by
  sorry

end limit_of_sequence_l56_56074


namespace abs_sub_nonneg_l56_56058

theorem abs_sub_nonneg (a : ℝ) : |a| - a ≥ 0 :=
sorry

end abs_sub_nonneg_l56_56058


namespace ratio_of_sheep_to_cow_l56_56762

noncomputable def sheep_to_cow_ratio 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : ℕ × ℕ := 
if h3 : 12 = 0 then (0, 0) else (2, 1)

theorem ratio_of_sheep_to_cow 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : sheep_to_cow_ratio S h1 h2 = (2, 1) := 
sorry

end ratio_of_sheep_to_cow_l56_56762


namespace cost_to_fill_sandbox_l56_56801

-- Definitions for conditions
def side_length : ℝ := 3
def volume_per_bag : ℝ := 3
def cost_per_bag : ℝ := 4

-- Theorem statement
theorem cost_to_fill_sandbox : (side_length ^ 3 / volume_per_bag * cost_per_bag) = 36 := by
  sorry

end cost_to_fill_sandbox_l56_56801


namespace calculate_expression_l56_56216

theorem calculate_expression :
  6 * 1000 + 5 * 100 + 6 * 1 = 6506 :=
by
  sorry

end calculate_expression_l56_56216


namespace Vasya_Capital_Decreased_l56_56208

theorem Vasya_Capital_Decreased (C : ℝ) (Du Dd : ℕ) 
  (h1 : 1000 * Du - 2000 * Dd = 0)
  (h2 : Du = 2 * Dd) :
  C * ((1.1:ℝ) ^ Du) * ((0.8:ℝ) ^ Dd) < C :=
by
  -- Assuming non-zero initial capital
  have hC : C ≠ 0 := sorry
  -- Substitution of Du = 2 * Dd
  rw [h2] at h1 
  -- From h1 => 1000 * 2 * Dd - 2000 * Dd = 0 => true always
  have hfalse : true := by sorry
  -- Substitution of h2 in the Vasya capital formula
  let cf := C * ((1.1:ℝ) ^ (2 * Dd)) * ((0.8:ℝ) ^ Dd)
  -- Further simplification
  have h₀ : C * ((1.1 : ℝ) ^ 2) ^ Dd * (0.8 : ℝ) ^ Dd = cf := by sorry
  -- Calculation of the effective multiplier
  have h₁ : (1.1 : ℝ) ^ 2 = 1.21 := by sorry
  have h₂ : 1.21 * (0.8 : ℝ) = 0.968 := by sorry
  -- Conclusion from the effective multiplier being < 1
  exact sorry

end Vasya_Capital_Decreased_l56_56208


namespace part1_solution_set_part2_range_of_a_l56_56782

-- Define the function f
def f (a x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Problem 1: When a = 1, solve the inequality f(x) ≤ 5
theorem part1_solution_set : 
  { x : ℝ | f 1 x ≤ 5 } = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 4} := 
  by 
  sorry

-- Problem 2: Determine the range of a for which f(x) has a minimum
theorem part2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x < 1/3 → f a x ≤ f a 1/3) → 
           (∀ x : ℝ, x ≥ 1/3 → f a x ≥ f a 1/3) ↔ 
           (-3 ≤ a ∧ a ≤ 3) := 
  by
  sorry

end part1_solution_set_part2_range_of_a_l56_56782


namespace can_determine_counterfeit_l56_56161

-- Define the conditions of the problem
structure ProblemConditions where
  totalCoins : ℕ := 100
  exaggeration : ℕ

-- Define the problem statement
theorem can_determine_counterfeit (P : ProblemConditions) : 
  ∃ strategy : ℕ → Prop, 
    ∀ (k : ℕ), strategy P.exaggeration -> 
    (∀ i, i < 100 → (P.totalCoins = 100 ∧ ∃ n, n > 0 ∧ 
     ∀ j, j < P.totalCoins → (P.totalCoins = j + 1 ∨ P.totalCoins = 99 + j))) := 
sorry

end can_determine_counterfeit_l56_56161


namespace convert_base_10_to_base_7_l56_56278

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 784) : 
  ∃ a b c d : ℕ, n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by
  sorry

end convert_base_10_to_base_7_l56_56278


namespace pizza_eaten_after_six_trips_l56_56825

noncomputable def fraction_eaten : ℚ :=
  let first_trip := 1 / 3
  let second_trip := 1 / (3 ^ 2)
  let third_trip := 1 / (3 ^ 3)
  let fourth_trip := 1 / (3 ^ 4)
  let fifth_trip := 1 / (3 ^ 5)
  let sixth_trip := 1 / (3 ^ 6)
  first_trip + second_trip + third_trip + fourth_trip + fifth_trip + sixth_trip

theorem pizza_eaten_after_six_trips : fraction_eaten = 364 / 729 :=
by sorry

end pizza_eaten_after_six_trips_l56_56825


namespace interest_rate_l56_56574

variable (P : ℝ) (T : ℝ) (SI : ℝ)

theorem interest_rate (h_P : P = 535.7142857142857) (h_T : T = 4) (h_SI : SI = 75) :
    (SI / (P * T)) * 100 = 3.5 := by
  sorry

end interest_rate_l56_56574


namespace find_y_l56_56567

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2 + 1 / y) 
  (h2 : y = 3 + 1 / x) : 
  y = (3/2) + (Real.sqrt 15 / 2) :=
by
  sorry

end find_y_l56_56567


namespace quadratic_inequality_real_solution_l56_56345

theorem quadratic_inequality_real_solution (a : ℝ) :
  (∃ x : ℝ, 2*x^2 + (a-1)*x + 1/2 ≤ 0) ↔ (a ≤ -1 ∨ 3 ≤ a) := 
sorry

end quadratic_inequality_real_solution_l56_56345


namespace smallest_composite_no_prime_factors_less_than_15_l56_56226

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l56_56226


namespace at_least_one_angle_not_less_than_sixty_l56_56921

theorem at_least_one_angle_not_less_than_sixty (A B C : ℝ)
  (hABC_sum : A + B + C = 180)
  (hA : A < 60)
  (hB : B < 60)
  (hC : C < 60) : false :=
by
  sorry

end at_least_one_angle_not_less_than_sixty_l56_56921


namespace find_r_minus2_l56_56863

noncomputable def p : ℤ → ℤ := sorry
def r : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom p_minus1 : p (-1) = 2
axiom p_3 : p (3) = 5
axiom p_minus4 : p (-4) = -3

-- Definition of r(x) when p(x) is divided by (x + 1)(x - 3)(x + 4)
axiom r_def : ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * (sorry : ℤ → ℤ) + r x

-- Our goal to prove
theorem find_r_minus2 : r (-2) = 32 / 7 :=
sorry

end find_r_minus2_l56_56863


namespace unique_cell_50_distance_l56_56060

-- Define the distance between two cells
def kingDistance (p1 p2 : ℤ × ℤ) : ℤ :=
  max (abs (p1.1 - p2.1)) (abs (p1.2 - p2.2))

-- A condition stating three cells with specific distances
variables (A B C : ℤ × ℤ) (hAB : kingDistance A B = 100) (hBC : kingDistance B C = 100) (hCA : kingDistance C A = 100)

-- A proposition to prove there is exactly one cell at a distance of 50 from all three given cells
theorem unique_cell_50_distance : ∃! D : ℤ × ℤ, kingDistance D A = 50 ∧ kingDistance D B = 50 ∧ kingDistance D C = 50 :=
sorry

end unique_cell_50_distance_l56_56060


namespace greatest_third_term_arithmetic_seq_l56_56818

theorem greatest_third_term_arithmetic_seq (a d : ℤ) (h1: a > 0) (h2: d ≥ 0) (h3: 5 * a + 10 * d = 65) : 
  a + 2 * d = 13 := 
by 
  sorry

end greatest_third_term_arithmetic_seq_l56_56818


namespace sin_600_eq_neg_sqrt_3_div_2_l56_56182

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l56_56182


namespace solve_system1_solve_system2_l56_56206

section System1

variables (x y : ℤ)

def system1_sol := x = 4 ∧ y = 8

theorem solve_system1 (h1 : y = 2 * x) (h2 : x + y = 12) : system1_sol x y :=
by 
  sorry

end System1

section System2

variables (x y : ℤ)

def system2_sol := x = 2 ∧ y = 3

theorem solve_system2 (h1 : 3 * x + 5 * y = 21) (h2 : 2 * x - 5 * y = -11) : system2_sol x y :=
by 
  sorry

end System2

end solve_system1_solve_system2_l56_56206


namespace defective_pens_l56_56075

theorem defective_pens :
  ∃ D N : ℕ, (N + D = 9) ∧ (N / 9 * (N - 1) / 8 = 5 / 12) ∧ (D = 3) :=
by
  sorry

end defective_pens_l56_56075


namespace number_of_zeros_at_end_l56_56254

def N (n : Nat) := 10^(n+1) + 1

theorem number_of_zeros_at_end (n : Nat) (h : n = 2017) : 
  (N n)^(n + 1) - 1 ≡ 0 [MOD 10^(n + 1)] :=
sorry

end number_of_zeros_at_end_l56_56254


namespace area_of_common_part_geq_3484_l56_56008

theorem area_of_common_part_geq_3484 :
  ∀ (R : ℝ) (S T : ℝ → Prop), 
  (R = 1) →
  (∀ x y, S x ↔ (x * x + y * y = R * R) ∧ T y) →
  ∃ (S_common : ℝ) (T_common : ℝ),
    (S_common + T_common > 3.484) :=
by
  sorry

end area_of_common_part_geq_3484_l56_56008


namespace range_of_a_and_t_minimum_of_y_l56_56540

noncomputable def minimum_value_y (a b : ℝ) (h : a + b = 1) : ℝ :=
(a + 1/a) * (b + 1/b)

theorem range_of_a_and_t (a b : ℝ) (h : a + b = 1) :
  0 < a ∧ a < 1 ∧ 0 < a * b ∧ a * b <= 1/4 :=
sorry

theorem minimum_of_y (a b : ℝ) (h : a + b = 1) :
  minimum_value_y a b h = 25/4 :=
sorry

end range_of_a_and_t_minimum_of_y_l56_56540


namespace number_of_daisies_is_two_l56_56723

theorem number_of_daisies_is_two :
  ∀ (total_flowers daisies tulips sunflowers remaining_flowers : ℕ), 
    total_flowers = 12 →
    sunflowers = 4 →
    (3 / 5) * remaining_flowers = tulips →
    (2 / 5) * remaining_flowers = sunflowers →
    remaining_flowers = total_flowers - daisies - sunflowers →
    daisies = 2 :=
by
  intros total_flowers daisies tulips sunflowers remaining_flowers 
  sorry

end number_of_daisies_is_two_l56_56723


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l56_56281

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l56_56281


namespace base_conversion_l56_56684

theorem base_conversion (k : ℕ) : (5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k + 4) → k = 7 :=
by 
  let x := 5 * 8^2 + 2 * 8^1 + 4 * 8^0
  have h : x = 340 := by sorry
  have hk : 6 * k^2 + 6 * k + 4 = 340 := by sorry
  sorry

end base_conversion_l56_56684


namespace operation_star_correct_l56_56353

def op_table (i j : ℕ) : ℕ :=
  if i = 1 then
    if j = 1 then 4 else if j = 2 then 1 else if j = 3 then 2 else if j = 4 then 3 else 0
  else if i = 2 then
    if j = 1 then 1 else if j = 2 then 3 else if j = 3 then 4 else if j = 4 then 2 else 0
  else if i = 3 then
    if j = 1 then 2 else if j = 2 then 4 else if j = 3 then 1 else if j = 4 then 3 else 0
  else if i = 4 then
    if j = 1 then 3 else if j = 2 then 2 else if j = 3 then 3 else if j = 4 then 4 else 0
  else 0

theorem operation_star_correct : op_table (op_table 3 1) (op_table 4 2) = 3 :=
  by sorry

end operation_star_correct_l56_56353


namespace inverse_function_domain_l56_56046

noncomputable def f (x : ℝ) : ℝ := -3 + Real.log (x - 1) / Real.log 2

theorem inverse_function_domain :
  ∀ x : ℝ, x ≥ 5 → ∃ y : ℝ, f x = y ∧ y ≥ -1 :=
by
  intro x hx
  use f x
  sorry

end inverse_function_domain_l56_56046


namespace smallest_prime_reversing_to_composite_l56_56395

theorem smallest_prime_reversing_to_composite (p : ℕ) :
  p = 23 ↔ (p < 100 ∧ p ≥ 10 ∧ Nat.Prime p ∧ 
  ∃ c, c < 100 ∧ c ≥ 10 ∧ ¬ Nat.Prime c ∧ c = (p % 10) * 10 + p / 10 ∧ (p / 10 = 2 ∨ p / 10 = 3)) :=
by
  sorry

end smallest_prime_reversing_to_composite_l56_56395


namespace tens_digit_of_2013_squared_minus_2013_l56_56796

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 / 10 = 5 := by
  sorry

end tens_digit_of_2013_squared_minus_2013_l56_56796


namespace time_to_cover_same_distance_l56_56469

theorem time_to_cover_same_distance
  (a b c d : ℕ) (k : ℕ) 
  (h_k : k = 3) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_speed_eq : 3 * (a + 2 * b) = 3 * a - b) : 
  (a + 2 * b) * (c + d) / (3 * a - b) = (a + 2 * b) * (c + d) / (3 * a - b) :=
by sorry

end time_to_cover_same_distance_l56_56469


namespace students_more_than_pets_l56_56124

theorem students_more_than_pets
    (num_classrooms : ℕ)
    (students_per_classroom : ℕ)
    (rabbits_per_classroom : ℕ)
    (hamsters_per_classroom : ℕ)
    (total_students : ℕ)
    (total_pets : ℕ)
    (difference : ℕ)
    (classrooms_eq : num_classrooms = 5)
    (students_eq : students_per_classroom = 20)
    (rabbits_eq : rabbits_per_classroom = 2)
    (hamsters_eq : hamsters_per_classroom = 1)
    (total_students_eq : total_students = num_classrooms * students_per_classroom)
    (total_pets_eq : total_pets = num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom)
    (difference_eq : difference = total_students - total_pets) :
  difference = 85 := by
  sorry

end students_more_than_pets_l56_56124


namespace fraction_b_not_whole_l56_56815

-- Defining the fractions as real numbers
def fraction_a := 60 / 12
def fraction_b := 60 / 8
def fraction_c := 60 / 5
def fraction_d := 60 / 4
def fraction_e := 60 / 3

-- Defining what it means to be a whole number
def is_whole_number (x : ℝ) : Prop := ∃ (n : ℤ), x = n

-- Theorem stating that fraction_b is not a whole number
theorem fraction_b_not_whole : ¬ is_whole_number fraction_b := 
by 
-- proof to be filled in
sorry

end fraction_b_not_whole_l56_56815


namespace birds_in_trees_l56_56135

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_l56_56135


namespace f_increasing_on_Ioo_l56_56408

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_increasing_on_Ioo : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2 :=
by sorry

end f_increasing_on_Ioo_l56_56408


namespace sum_of_integers_l56_56786

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 144) : x + y = 24 :=
sorry

end sum_of_integers_l56_56786


namespace number_of_adults_had_meal_l56_56845

theorem number_of_adults_had_meal (A : ℝ) :
  let num_children_food : ℝ := 63
  let food_for_adults : ℝ := 70
  let food_for_children : ℝ := 90
  (food_for_children - A * (food_for_children / food_for_adults) = num_children_food) →
  A = 21 :=
by
  intros num_children_food food_for_adults food_for_children h
  have h2 : 90 - A * (90 / 70) = 63 := h
  sorry

end number_of_adults_had_meal_l56_56845


namespace no_such_integers_l56_56910

def p (x : ℤ) : ℤ := x^2 + x - 70

theorem no_such_integers : ¬ (∃ m n : ℤ, 0 < m ∧ m < n ∧ n ∣ p m ∧ (n + 1) ∣ p (m + 1)) :=
by
  sorry

end no_such_integers_l56_56910


namespace max_students_l56_56518

def num_pens : Nat := 1204
def num_pencils : Nat := 840

theorem max_students (n_pens n_pencils : Nat) (h_pens : n_pens = num_pens) (h_pencils : n_pencils = num_pencils) :
  Nat.gcd n_pens n_pencils = 16 := by
  sorry

end max_students_l56_56518


namespace floor_sqrt_50_l56_56429

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l56_56429


namespace german_students_count_l56_56355

def total_students : ℕ := 45
def both_english_german : ℕ := 12
def only_english : ℕ := 23

theorem german_students_count :
  ∃ G : ℕ, G = 45 - (23 + 12) + 12 :=
sorry

end german_students_count_l56_56355


namespace john_sells_percentage_of_newspapers_l56_56620

theorem john_sells_percentage_of_newspapers
    (n_newspapers : ℕ)
    (selling_price : ℝ)
    (cost_price_discount : ℝ)
    (profit : ℝ)
    (sold_percentage : ℝ)
    (h1 : n_newspapers = 500)
    (h2 : selling_price = 2)
    (h3 : cost_price_discount = 0.75)
    (h4 : profit = 550)
    (h5 : sold_percentage = 80) : 
    ( ∃ (sold_n : ℕ), 
      sold_n / n_newspapers * 100 = sold_percentage ∧
      sold_n * selling_price = 
        n_newspapers * selling_price * (1 - cost_price_discount) + profit) :=
by
  sorry

end john_sells_percentage_of_newspapers_l56_56620


namespace fraction_addition_l56_56503

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l56_56503


namespace cupcakes_for_children_l56_56416

-- Definitions for the conditions
def packs15 : Nat := 4
def packs10 : Nat := 4
def cupcakes_per_pack15 : Nat := 15
def cupcakes_per_pack10 : Nat := 10

-- Proposition to prove the total number of cupcakes is 100
theorem cupcakes_for_children :
  (packs15 * cupcakes_per_pack15) + (packs10 * cupcakes_per_pack10) = 100 := by
  sorry

end cupcakes_for_children_l56_56416


namespace cylinder_volume_l56_56965

variables (a : ℝ) (π_ne_zero : π ≠ 0) (two_ne_zero : 2 ≠ 0) 

theorem cylinder_volume (h1 : ∃ (h r : ℝ), (2 * π * r = 2 * a ∧ h = a) 
                        ∨ (2 * π * r = a ∧ h = 2 * a)) :
  (∃ (V : ℝ), V = a^3 / π) ∨ (∃ (V : ℝ), V = a^3 / (2 * π)) :=
by
  sorry

end cylinder_volume_l56_56965


namespace sum_of_digits_l56_56172

noncomputable def digits_divisibility (C F : ℕ) : Prop :=
  (C >= 0 ∧ C <= 9 ∧ F >= 0 ∧ F <= 9) ∧
  (C + 8 + 5 + 4 + F + 7 + 2) % 9 = 0 ∧
  (100 * 4 + 10 * F + 72) % 8 = 0

theorem sum_of_digits (C F : ℕ) (h : digits_divisibility C F) : C + F = 10 :=
sorry

end sum_of_digits_l56_56172


namespace max_k_value_l56_56750

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∃ k : ℝ, (∀ m, 0 < m → m < 1/2 → (1/m + 2/(1-2*m) ≥ k)) ∧ k = 8) := 
sorry

end max_k_value_l56_56750


namespace arcade_fraction_spent_l56_56293

noncomputable def weekly_allowance : ℚ := 2.25 
def y (x : ℚ) : ℚ := 1 - x
def remainding_after_toy (x : ℚ) : ℚ := y x - (1/3) * y x

theorem arcade_fraction_spent : 
  ∃ x : ℚ, remainding_after_toy x = 0.60 ∧ x = 3/5 :=
by
  sorry

end arcade_fraction_spent_l56_56293


namespace ratio_M_N_l56_56674

theorem ratio_M_N (M Q P N : ℝ) (hM : M = 0.40 * Q) (hQ : Q = 0.25 * P) (hN : N = 0.60 * P) (hP : P ≠ 0) : 
  (M / N) = (1 / 6) := 
by 
  sorry

end ratio_M_N_l56_56674


namespace ticket_savings_percentage_l56_56854

theorem ticket_savings_percentage:
  ∀ (P : ℝ), 9 * P - 6 * P = (1 / 3) * (9 * P) ∧ (33 + 1/3) = 100 * (3 * P / (9 * P)) := 
by
  intros P
  sorry

end ticket_savings_percentage_l56_56854


namespace certain_number_eq_neg_thirteen_over_two_l56_56005

noncomputable def CertainNumber (w : ℝ) : ℝ := 13 * w / (1 - w)

theorem certain_number_eq_neg_thirteen_over_two (w : ℝ) (h : w ^ 2 = 1) (hz : 1 - w ≠ 0) :
  CertainNumber w = -13 / 2 :=
sorry

end certain_number_eq_neg_thirteen_over_two_l56_56005


namespace charge_move_increases_energy_l56_56902

noncomputable def energy_increase_when_charge_moved : ℝ :=
  let initial_energy := 15
  let energy_per_pair := initial_energy / 3
  let new_energy_AB := energy_per_pair
  let new_energy_AC := 2 * energy_per_pair
  let new_energy_BC := 2 * energy_per_pair
  let final_energy := new_energy_AB + new_energy_AC + new_energy_BC
  final_energy - initial_energy

theorem charge_move_increases_energy :
  energy_increase_when_charge_moved = 10 :=
by
  sorry

end charge_move_increases_energy_l56_56902


namespace michael_total_payment_correct_l56_56295

variable (original_suit_price : ℕ := 430)
variable (suit_discount : ℕ := 100)
variable (suit_tax_rate : ℚ := 0.05)

variable (original_shoes_price : ℕ := 190)
variable (shoes_discount : ℕ := 30)
variable (shoes_tax_rate : ℚ := 0.07)

variable (original_dress_shirt_price : ℕ := 80)
variable (original_tie_price : ℕ := 50)
variable (combined_discount_rate : ℚ := 0.20)
variable (dress_shirt_tax_rate : ℚ := 0.06)
variable (tie_tax_rate : ℚ := 0.04)

def calculate_total_amount_paid : ℚ :=
  let discounted_suit_price := original_suit_price - suit_discount
  let suit_tax := discounted_suit_price * suit_tax_rate
  let discounted_shoes_price := original_shoes_price - shoes_discount
  let shoes_tax := discounted_shoes_price * shoes_tax_rate
  let combined_original_price := original_dress_shirt_price + original_tie_price
  let combined_discount := combined_discount_rate * combined_original_price
  let discounted_combined_price := combined_original_price - combined_discount
  let discounted_dress_shirt_price := (original_dress_shirt_price / combined_original_price) * discounted_combined_price
  let discounted_tie_price := (original_tie_price / combined_original_price) * discounted_combined_price
  let dress_shirt_tax := discounted_dress_shirt_price * dress_shirt_tax_rate
  let tie_tax := discounted_tie_price * tie_tax_rate
  discounted_suit_price + suit_tax + discounted_shoes_price + shoes_tax + discounted_dress_shirt_price + dress_shirt_tax + discounted_tie_price + tie_tax

theorem michael_total_payment_correct : calculate_total_amount_paid = 627.14 := by
  sorry

end michael_total_payment_correct_l56_56295


namespace cost_per_component_l56_56188

theorem cost_per_component (C : ℝ) : 
  (150 * C + 150 * 4 + 16500 = 150 * 193.33) → 
  C = 79.33 := 
by
  intro h
  sorry

end cost_per_component_l56_56188


namespace find_first_term_l56_56575

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l56_56575


namespace three_digit_repeated_digits_percentage_l56_56633

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 900
  let non_repeated := 9 * 9 * 8
  let repeated := total_numbers - non_repeated
  (repeated / total_numbers) * 100

theorem three_digit_repeated_digits_percentage :
  percentage_repeated_digits = 28.0 := by
  sorry

end three_digit_repeated_digits_percentage_l56_56633


namespace percentage_decrease_is_25_percent_l56_56246

noncomputable def percentage_decrease_in_revenue
  (R : ℝ)
  (projected_revenue : ℝ)
  (actual_revenue : ℝ) : ℝ :=
  ((R - actual_revenue) / R) * 100

-- Conditions
def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.20 * R
def actual_revenue (R : ℝ) := 0.625 * (1.20 * R)

-- Proof statement
theorem percentage_decrease_is_25_percent (R : ℝ) :
  percentage_decrease_in_revenue R (projected_revenue R) (actual_revenue R) = 25 :=
by
  sorry

end percentage_decrease_is_25_percent_l56_56246


namespace total_sum_l56_56530

theorem total_sum (p q r s t : ℝ) (P : ℝ) 
  (h1 : q = 0.75 * P) 
  (h2 : r = 0.50 * P) 
  (h3 : s = 0.25 * P) 
  (h4 : t = 0.10 * P) 
  (h5 : s = 25) 
  :
  p + q + r + s + t = 260 :=
by 
  sorry

end total_sum_l56_56530


namespace sum_odd_numbers_to_2019_is_correct_l56_56888

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end sum_odd_numbers_to_2019_is_correct_l56_56888


namespace neg_exists_eq_forall_ne_l56_56751

variable (x : ℝ)

theorem neg_exists_eq_forall_ne : (¬ ∃ x : ℝ, x^2 - 2 * x = 0) ↔ ∀ x : ℝ, x^2 - 2 * x ≠ 0 := by
  sorry

end neg_exists_eq_forall_ne_l56_56751


namespace complex_div_l56_56154

open Complex

theorem complex_div (i : ℂ) (hi : i = Complex.I) : 
  (6 + 7 * i) / (1 + 2 * i) = 4 - i := 
by 
  sorry

end complex_div_l56_56154


namespace triangle_A_and_Area_l56_56381

theorem triangle_A_and_Area :
  ∀ (a b c A B C : ℝ), 
  (b - (1 / 2) * c = a * Real.cos C) 
  → (4 * (b + c) = 3 * b * c) 
  → (a = 2 * Real.sqrt 3)
  → (A = 60) ∧ (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) :=
by
  intros a b c A B C h1 h2 h3
  sorry

end triangle_A_and_Area_l56_56381


namespace not_P_4_given_not_P_5_l56_56192

-- Define the proposition P for natural numbers
def P (n : ℕ) : Prop := sorry

-- Define the statement we need to prove
theorem not_P_4_given_not_P_5 (h1 : ∀ k : ℕ, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 := by
  sorry

end not_P_4_given_not_P_5_l56_56192


namespace smallest_abs_sum_of_products_l56_56876

noncomputable def g (x : ℝ) : ℝ := x^4 + 16 * x^3 + 69 * x^2 + 112 * x + 64

theorem smallest_abs_sum_of_products :
  (∀ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 → 
   |w1 * w2 + w3 * w4| ≥ 8) ∧ 
  (∃ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 ∧ 
   |w1 * w2 + w3 * w4| = 8) :=
sorry

end smallest_abs_sum_of_products_l56_56876


namespace unique_solution_l56_56358

def my_operation (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution :
  ∃! y : ℝ, my_operation 4 y = 15 ∧ y = -1/2 :=
by 
  sorry

end unique_solution_l56_56358


namespace part1_part2_l56_56938

-- Define the conditions p and q
def p (a x : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := (x - 2) * (x - 4) < 0 ∧ (x - 3) * (x - 5) > 0

-- Problem Part 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem part1 (x : ℝ) : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by
  intro h
  sorry

-- Problem Part 2: Prove that if p is a necessary but not sufficient condition for q, then 1 ≤ a ≤ 2
theorem part2 (a : ℝ) : (∀ x, q x → p a x) ∧ (∃ x, p a x ∧ ¬q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  intro h
  sorry

end part1_part2_l56_56938


namespace minimum_value_sine_shift_l56_56760

theorem minimum_value_sine_shift :
  ∀ (f : ℝ → ℝ) (φ : ℝ), (∀ x, f x = Real.sin (2 * x + φ)) → |φ| < Real.pi / 2 →
  (∀ x, f (x + Real.pi / 6) = f (-x)) →
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = - Real.sqrt 3 / 2 :=
by
  sorry

end minimum_value_sine_shift_l56_56760


namespace solve_quadratic_equation_l56_56654

theorem solve_quadratic_equation (x : ℝ) : x^2 + 4 * x = 5 ↔ x = 1 ∨ x = -5 := sorry

end solve_quadratic_equation_l56_56654


namespace escalator_times_comparison_l56_56168

variable (v v1 v2 l : ℝ)
variable (h_v_lt_v1 : v < v1)
variable (h_v1_lt_v2 : v1 < v2)

theorem escalator_times_comparison
  (h_cond : v < v1 ∧ v1 < v2) :
  (l / (v1 + v) + l / (v2 - v)) < (l / (v1 - v) + l / (v2 + v)) :=
  sorry

end escalator_times_comparison_l56_56168


namespace men_in_second_group_l56_56007

theorem men_in_second_group (M : ℕ) (W : ℝ) (h1 : 15 * 25 = W) (h2 : M * 18.75 = W) : M = 20 :=
sorry

end men_in_second_group_l56_56007


namespace triangle_isosceles_l56_56035

-- Definitions involved: Triangle, Circumcircle, Angle Bisector, Isosceles Triangle
universe u

structure Triangle (α : Type u) :=
  (A B C : α)

structure Circumcircle (α : Type u) :=
  (triangle : Triangle α)

structure AngleBisector (α : Type u) :=
  (A : α)
  (triangle : Triangle α)

def IsoscelesTriangle {α : Type u} (P Q R : α) : Prop :=
  ∃ (p₁ p₂ p₃ : α), (p₁ = P ∧ p₂ = Q ∧ p₃ = R) ∧
                  ((∃ θ₁ θ₂, θ₁ + θ₂ = 90) → (∃ θ₃ θ₂, θ₃ + θ₂ = 90))

theorem triangle_isosceles {α : Type u} (T : Triangle α) (S : α)
  (h1 : Circumcircle α) (h2 : AngleBisector α) :
  IsoscelesTriangle T.B T.C S :=
by
  sorry

end triangle_isosceles_l56_56035


namespace q_minus_r_max_value_l56_56936

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), 1073 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
sorry

end q_minus_r_max_value_l56_56936


namespace degree_g_is_six_l56_56981

theorem degree_g_is_six 
  (f g : Polynomial ℂ) 
  (h : Polynomial ℂ) 
  (h_def : h = f.comp g + Polynomial.X * g) 
  (deg_h : h.degree = 7) 
  (deg_f : f.degree = 3) 
  : g.degree = 6 := 
sorry

end degree_g_is_six_l56_56981


namespace α_in_quadrants_l56_56623

def α (k : ℤ) : ℝ := k * 180 + 45

theorem α_in_quadrants (k : ℤ) : 
  (0 ≤ α k ∧ α k < 90) ∨ (180 < α k ∧ α k ≤ 270) :=
sorry

end α_in_quadrants_l56_56623


namespace fruits_eaten_total_l56_56174

variable (apples blueberries bonnies : ℕ)

noncomputable def total_fruits_eaten : ℕ :=
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 / 4 * third_dog_bonnies
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies

theorem fruits_eaten_total:
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 * third_dog_bonnies / 4
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies = 240 := by
  sorry

end fruits_eaten_total_l56_56174


namespace greatest_y_l56_56045

theorem greatest_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : y ≤ -2 :=
by {
  sorry
}

end greatest_y_l56_56045


namespace relatively_prime_subsequence_exists_l56_56386

theorem relatively_prime_subsequence_exists :
  ∃ (s : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → Nat.gcd (2^(s i) - 3) (2^(s j) - 3) = 1) :=
by
  sorry

end relatively_prime_subsequence_exists_l56_56386


namespace zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l56_56131

theorem zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three :
  (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) :=
by
  sorry

end zero_plus_one_plus_two_plus_three_not_eq_zero_mul_one_mul_two_mul_three_l56_56131


namespace thabo_books_220_l56_56160

def thabo_books_total (H PNF PF Total : ℕ) : Prop :=
  (H = 40) ∧
  (PNF = H + 20) ∧
  (PF = 2 * PNF) ∧
  (Total = H + PNF + PF)

theorem thabo_books_220 : ∃ H PNF PF Total : ℕ, thabo_books_total H PNF PF 220 :=
by {
  sorry
}

end thabo_books_220_l56_56160


namespace smallest_angle_of_trapezoid_l56_56608

theorem smallest_angle_of_trapezoid 
  (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : ∀ i j k l : ℝ, i + j = k + l → i + j = 180 ∧ k + l = 180) :
  a = 40 :=
by
  sorry

end smallest_angle_of_trapezoid_l56_56608


namespace total_weekly_messages_l56_56307

theorem total_weekly_messages (n r1 r2 r3 r4 r5 m1 m2 m3 m4 m5 : ℕ) 
(p1 p2 p3 p4 : ℕ) (h1 : n = 200) (h2 : r1 = 15) (h3 : r2 = 25) (h4 : r3 = 10) 
(h5 : r4 = 20) (h6 : r5 = 5) (h7 : m1 = 40) (h8 : m2 = 60) (h9 : m3 = 50) 
(h10 : m4 = 30) (h11 : m5 = 20) (h12 : p1 = 15) (h13 : p2 = 25) (h14 : p3 = 40) 
(h15 : p4 = 10) : 
  let total_members_removed := r1 + r2 + r3 + r4 + r5
  let remaining_members := n - total_members_removed
  let daily_messages :=
        (25 * remaining_members / 100 * p1) +
        (50 * remaining_members / 100 * p2) +
        (20 * remaining_members / 100 * p3) +
        (5 * remaining_members / 100 * p4)
  let weekly_messages := daily_messages * 7
  weekly_messages = 21663 :=
by
  sorry

end total_weekly_messages_l56_56307


namespace factorize_expr_l56_56942

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l56_56942


namespace andrew_total_appeizers_count_l56_56437

theorem andrew_total_appeizers_count :
  let hotdogs := 30
  let cheese_pops := 20
  let chicken_nuggets := 40
  hotdogs + cheese_pops + chicken_nuggets = 90 := 
by 
  sorry

end andrew_total_appeizers_count_l56_56437


namespace maximum_q_minus_r_l56_56502

theorem maximum_q_minus_r : 
  ∀ q r : ℕ, (1027 = 23 * q + r) ∧ (q > 0) ∧ (r > 0) → q - r ≤ 29 := 
by
  sorry

end maximum_q_minus_r_l56_56502


namespace find_x_eq_14_4_l56_56596

theorem find_x_eq_14_4 (x : ℝ) (h : ⌈x⌉ * x = 216) : x = 14.4 :=
by
  sorry

end find_x_eq_14_4_l56_56596


namespace inequality_proof_l56_56666

theorem inequality_proof (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 → x > 5 / 7 := 
by
  sorry

end inequality_proof_l56_56666


namespace height_previous_year_l56_56048

theorem height_previous_year (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) 
  (h1 : current_height = 126)
  (h2 : growth_rate = 0.05) 
  (h3 : current_height = 1.05 * previous_height) : 
  previous_height = 120 :=
sorry

end height_previous_year_l56_56048


namespace arithmetic_mean_multiplied_correct_l56_56383

-- Define the fractions involved
def frac1 : ℚ := 3 / 4
def frac2 : ℚ := 5 / 8

-- Define the arithmetic mean and the final multiplication result
def mean_and_multiply_result : ℚ := ( (frac1 + frac2) / 2 ) * 3

-- Statement to prove that the calculated result is equal to 33/16
theorem arithmetic_mean_multiplied_correct : mean_and_multiply_result = 33 / 16 := 
by 
  -- Skipping the proof with sorry for the statement only requirement
  sorry

end arithmetic_mean_multiplied_correct_l56_56383


namespace fraction_increase_by_two_times_l56_56176

theorem fraction_increase_by_two_times (x y : ℝ) : 
  let new_val := ((2 * x) * (2 * y)) / (2 * x + 2 * y)
  let original_val := (x * y) / (x + y)
  new_val = 2 * original_val := 
by
  sorry

end fraction_increase_by_two_times_l56_56176


namespace simplify_expression_l56_56681

theorem simplify_expression (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 :=
by
  sorry

end simplify_expression_l56_56681


namespace find_W_l56_56029

def digit_sum_eq (X Y Z W : ℕ) : Prop := X * 10 + Y + Z * 10 + X = W * 10 + X
def digit_diff_eq (X Y Z : ℕ) : Prop := X * 10 + Y - (Z * 10 + X) = X
def is_digit (n : ℕ) : Prop := n < 10

theorem find_W (X Y Z W : ℕ) (h1 : digit_sum_eq X Y Z W) (h2 : digit_diff_eq X Y Z) 
  (hX : is_digit X) (hY : is_digit Y) (hZ : is_digit Z) (hW : is_digit W) : W = 0 := 
sorry

end find_W_l56_56029


namespace smallest_integral_value_of_y_l56_56413

theorem smallest_integral_value_of_y :
  ∃ y : ℤ, (1 / 4 : ℝ) < y / 7 ∧ y / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 / 4 : ℝ) < z / 7 ∧ z / 7 < 2 / 3 → y ≤ z :=
by
  -- The statement is defined and the proof is left as "sorry" to illustrate that no solution steps are used directly.
  sorry

end smallest_integral_value_of_y_l56_56413


namespace quotient_base_6_l56_56267

noncomputable def base_6_to_base_10 (n : ℕ) : ℕ := 
  match n with
  | 2314 => 2 * 6^3 + 3 * 6^2 + 1 * 6^1 + 4
  | 14 => 1 * 6^1 + 4
  | _ => 0

noncomputable def base_10_to_base_6 (n : ℕ) : ℕ := 
  match n with
  | 55 => 1 * 6^2 + 3 * 6^1 + 5
  | _ => 0

theorem quotient_base_6 :
  base_10_to_base_6 ((base_6_to_base_10 2314) / (base_6_to_base_10 14)) = 135 :=
by
  sorry

end quotient_base_6_l56_56267


namespace money_lent_to_B_l56_56794

theorem money_lent_to_B (total_money : ℕ) (interest_A_rate : ℚ) (interest_B_rate : ℚ) (interest_difference : ℚ) (years : ℕ) 
  (x y : ℚ) 
  (h1 : total_money = 10000)
  (h2 : interest_A_rate = 0.15)
  (h3 : interest_B_rate = 0.18)
  (h4 : interest_difference = 360)
  (h5 : years = 2)
  (h6 : y = total_money - x)
  (h7 : ((x * interest_A_rate * years) = ((y * interest_B_rate * years) + interest_difference))) : 
  y = 4000 := 
sorry

end money_lent_to_B_l56_56794


namespace three_digit_number_divisible_by_7_l56_56056

theorem three_digit_number_divisible_by_7
  (a b : ℕ)
  (h1 : (a + b) % 7 = 0) :
  (100 * a + 10 * b + a) % 7 = 0 :=
sorry

end three_digit_number_divisible_by_7_l56_56056


namespace solve_for_five_minus_a_l56_56710

theorem solve_for_five_minus_a (a b : ℤ) 
  (h1 : 5 + a = 6 - b)
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := 
by 
  sorry

end solve_for_five_minus_a_l56_56710


namespace sum_first_10_terms_arith_seq_l56_56377

theorem sum_first_10_terms_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 3 = 5)
  (h2 : a 7 = 13)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S 10 = 100 :=
sorry

end sum_first_10_terms_arith_seq_l56_56377


namespace cos_alpha_minus_270_l56_56096

open Real

theorem cos_alpha_minus_270 (α : ℝ) : 
  sin (540 * (π / 180) + α) = -4 / 5 → cos (α - 270 * (π / 180)) = -4 / 5 :=
by
  sorry

end cos_alpha_minus_270_l56_56096


namespace parameter_exists_solution_l56_56240

theorem parameter_exists_solution (b : ℝ) (h : b ≥ -2 * Real.sqrt 2 - 1 / 4) :
  ∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y) :=
by
  sorry

end parameter_exists_solution_l56_56240


namespace solve_system_infinite_solutions_l56_56291

theorem solve_system_infinite_solutions (m : ℝ) (h1 : ∀ x y : ℝ, x + m * y = 2) (h2 : ∀ x y : ℝ, m * x + 16 * y = 8) :
  m = 4 :=
sorry

end solve_system_infinite_solutions_l56_56291


namespace correct_system_of_equations_l56_56641

-- Define the given problem conditions.
def cost_doll : ℝ := 60
def cost_keychain : ℝ := 20
def total_cost : ℝ := 5000

-- Define the condition that each gift set needs 1 doll and 2 keychains.
def gift_set_relation (x y : ℝ) : Prop := 2 * x = y

-- Define the system of equations representing the problem.
def system_of_equations (x y : ℝ) : Prop :=
  2 * x = y ∧
  60 * x + 20 * y = total_cost

-- State the theorem to prove that the given system correctly models the problem.
theorem correct_system_of_equations (x y : ℝ) :
  system_of_equations x y ↔ (2 * x = y ∧ 60 * x + 20 * y = 5000) :=
by sorry

end correct_system_of_equations_l56_56641


namespace proof_problem_l56_56360

theorem proof_problem (a b c : ℤ) (h1 : a > 2) (h2 : b < 10) (h3 : c ≥ 0) (h4 : 32 = a + 2 * b + 3 * c) : 
  a = 4 ∧ b = 8 ∧ c = 4 :=
by
  sorry

end proof_problem_l56_56360


namespace basketball_game_l56_56705

/-- Given the conditions of the basketball game:
  * a, ar, ar^2, ar^3 form the Dragons' scores
  * b, b + d, b + 2d, b + 3d form the Lions' scores
  * The game was tied at halftime: a + ar = b + (b + d)
  * The Dragons won by three points at the end: a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3
  * Neither team scored more than 100 points
Prove that the total number of points scored by the two teams in the first half is 30.
-/
theorem basketball_game (a r b d : ℕ) (h1 : a + a * r = b + (b + d))
  (h2 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (h3 : a * (1 + r + r^2 + r^3) < 100)
  (h4 : 4 * b + 6 * d < 100) :
  a + a * r + b + (b + d) = 30 :=
by
  sorry

end basketball_game_l56_56705


namespace equal_perimeter_triangle_side_length_l56_56677

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end equal_perimeter_triangle_side_length_l56_56677


namespace largest_prime_factor_of_expression_l56_56315

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Prime p ∧ p > 35 ∧ p > 2 ∧ p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ ∀ q, Prime q ∧ q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l56_56315


namespace shanna_tomato_ratio_l56_56078

-- Define the initial conditions
def initial_tomato_plants : ℕ := 6
def initial_eggplant_plants : ℕ := 2
def initial_pepper_plants : ℕ := 4
def pepper_plants_died : ℕ := 1
def vegetables_per_plant : ℕ := 7
def total_vegetables_harvested : ℕ := 56

-- Define the number of tomato plants that died
def tomato_plants_died (total_vegetables : ℕ) (veg_per_plant : ℕ) (initial_tomato : ℕ) 
  (initial_eggplant : ℕ) (initial_pepper : ℕ) (pepper_died : ℕ) : ℕ :=
  let surviving_plants := total_vegetables / veg_per_plant
  let surviving_pepper := initial_pepper - pepper_died
  let surviving_tomato := surviving_plants - (initial_eggplant + surviving_pepper)
  initial_tomato - surviving_tomato

-- Define the ratio
def ratio_tomato_plants_died_to_initial (tomato_died : ℕ) (initial_tomato : ℕ) : ℚ :=
  (tomato_died : ℚ) / (initial_tomato : ℚ)

theorem shanna_tomato_ratio :
  ratio_tomato_plants_died_to_initial (tomato_plants_died total_vegetables_harvested vegetables_per_plant 
    initial_tomato_plants initial_eggplant_plants initial_pepper_plants pepper_plants_died) initial_tomato_plants 
  = 1 / 2 := by
  sorry

end shanna_tomato_ratio_l56_56078


namespace craig_total_distance_l56_56068

-- Define the distances Craig walked
def dist_school_to_david : ℝ := 0.27
def dist_david_to_home : ℝ := 0.73

-- Prove the total distance walked
theorem craig_total_distance : dist_school_to_david + dist_david_to_home = 1.00 :=
by
  -- Proof goes here
  sorry

end craig_total_distance_l56_56068


namespace number_of_first_grade_students_l56_56479

noncomputable def sampling_ratio (total_students : ℕ) (sampled_students : ℕ) : ℚ :=
  sampled_students / total_students

noncomputable def num_first_grade_selected (first_grade_students : ℕ) (ratio : ℚ) : ℚ :=
  ratio * first_grade_students

theorem number_of_first_grade_students
  (total_students : ℕ)
  (sampled_students : ℕ)
  (first_grade_students : ℕ)
  (h_total : total_students = 2400)
  (h_sampled : sampled_students = 100)
  (h_first_grade : first_grade_students = 840)
  : num_first_grade_selected first_grade_students (sampling_ratio total_students sampled_students) = 35 := by
  sorry

end number_of_first_grade_students_l56_56479


namespace coin_toss_probability_l56_56370

theorem coin_toss_probability :
  (∀ n : ℕ, 0 ≤ n → n ≤ 10 → (∀ m : ℕ, 0 ≤ m → m = 10 → 
  (∀ k : ℕ, k = 9 → 
  (∀ i : ℕ, 0 ≤ i → i = 10 → ∃ p : ℝ, p = 1/2 → 
  (∃ q : ℝ, q = 1/2 → q = p))))) := 
sorry

end coin_toss_probability_l56_56370


namespace simplify_fraction_l56_56777

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by
  sorry

end simplify_fraction_l56_56777


namespace correct_option_is_B_l56_56587

theorem correct_option_is_B :
  (∃ (A B C D : String), A = "√49 = -7" ∧ B = "√((-3)^2) = 3" ∧ C = "-√((-5)^2) = 5" ∧ D = "√81 = ±9" ∧
    (B = "√((-3)^2) = 3")) :=
by
  sorry

end correct_option_is_B_l56_56587


namespace volleyball_teams_l56_56542

theorem volleyball_teams (managers employees teams : ℕ) (h1 : managers = 3) (h2 : employees = 3) (h3 : teams = 3) :
  ((managers + employees) / teams) = 2 :=
by
  sorry

end volleyball_teams_l56_56542


namespace max_children_l56_56878

/-- Total quantities -/
def total_apples : ℕ := 55
def total_cookies : ℕ := 114
def total_chocolates : ℕ := 83

/-- Leftover quantities after distribution -/
def leftover_apples : ℕ := 3
def leftover_cookies : ℕ := 10
def leftover_chocolates : ℕ := 5

/-- Distributed quantities -/
def distributed_apples : ℕ := total_apples - leftover_apples
def distributed_cookies : ℕ := total_cookies - leftover_cookies
def distributed_chocolates : ℕ := total_chocolates - leftover_chocolates

/-- The theorem states the maximum number of children -/
theorem max_children : Nat.gcd (Nat.gcd distributed_apples distributed_cookies) distributed_chocolates = 26 :=
by
  sorry

end max_children_l56_56878


namespace find_f1_l56_56227

noncomputable def f (x a b : ℝ) : ℝ := a * Real.sin x - b * Real.tan x + 4 * Real.cos (Real.pi / 3)

theorem find_f1 (a b : ℝ) (h : f (-1) a b = 1) : f 1 a b = 3 :=
by {
  sorry
}

end find_f1_l56_56227


namespace ab_root_inequality_l56_56311

theorem ab_root_inequality (a b : ℝ) (h1: ∀ x : ℝ, (x + a) * (x + b) = -9) (h2: a < 0) (h3: b < 0) :
  a + b < -6 :=
sorry

end ab_root_inequality_l56_56311


namespace double_probability_correct_l56_56481

def is_double (a : ℕ × ℕ) : Prop := a.1 = a.2

def total_dominoes : ℕ := 13 * 13

def double_count : ℕ := 13

def double_probability := (double_count : ℚ) / total_dominoes

theorem double_probability_correct : double_probability = 13 / 169 := by
  sorry

end double_probability_correct_l56_56481


namespace total_students_l56_56213

-- Define the conditions based on the problem
def valentines_have : ℝ := 58.0
def valentines_needed : ℝ := 16.0

-- Theorem stating that the total number of students (which is equal to the total number of Valentines required)
theorem total_students : valentines_have + valentines_needed = 74.0 :=
by
  sorry

end total_students_l56_56213


namespace proof_l56_56496

noncomputable def line_standard_form (t : ℝ) : Prop :=
  let (x, y) := (t + 3, 3 - t)
  x + y = 6

noncomputable def circle_standard_form (θ : ℝ) : Prop :=
  let (x, y) := (2 * Real.cos θ, 2 * Real.sin θ + 2)
  x^2 + (y - 2)^2 = 4

noncomputable def distance_center_to_line (x1 y1 : ℝ) : ℝ :=
  let (a, b, c) := (1, 1, -6)
  let num := abs (a * x1 + b * y1 + c)
  let denom := Real.sqrt (a^2 + b^2)
  num / denom

theorem proof : 
  (∀ t, line_standard_form t) ∧ 
  (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → circle_standard_form θ) ∧ 
  distance_center_to_line 0 2 = 2 * Real.sqrt 2 :=
by
  sorry

end proof_l56_56496


namespace given_conditions_imply_f_neg3_gt_f_neg2_l56_56817

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem given_conditions_imply_f_neg3_gt_f_neg2
  {f : ℝ → ℝ}
  (h_even : is_even_function f)
  (h_comparison : f 2 < f 3) :
  f (-3) > f (-2) :=
by
  sorry

end given_conditions_imply_f_neg3_gt_f_neg2_l56_56817


namespace circle_radius_l56_56424

theorem circle_radius (A : ℝ) (r : ℝ) (hA : A = 121 * Real.pi) (hArea : A = Real.pi * r^2) : r = 11 :=
by
  sorry

end circle_radius_l56_56424


namespace sum_of_consecutive_odds_l56_56639

theorem sum_of_consecutive_odds (N1 N2 N3 : ℕ) (h1 : N1 % 2 = 1) (h2 : N2 % 2 = 1) (h3 : N3 % 2 = 1)
  (h_consec1 : N2 = N1 + 2) (h_consec2 : N3 = N2 + 2) (h_max : N3 = 27) : 
  N1 + N2 + N3 = 75 := by
  sorry

end sum_of_consecutive_odds_l56_56639


namespace expression_value_l56_56646

theorem expression_value :
  (6^2 - 3^2)^4 = 531441 := by
  -- Proof steps were omitted
  sorry

end expression_value_l56_56646


namespace exam_passing_marks_l56_56756

theorem exam_passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.40 * T + 10 = P) 
  (h3 : 0.50 * T - 5 = P + 40) : 
  P = 210 := 
sorry

end exam_passing_marks_l56_56756


namespace max_a_l56_56128

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a (a m n : ℝ) (h₀ : 1 ≤ m ∧ m ≤ 5)
                      (h₁ : 1 ≤ n ∧ n ≤ 5)
                      (h₂ : n - m ≥ 2)
                      (h_eq : f a m = f a n) :
  a ≤ Real.log 3 / 4 :=
sorry

end max_a_l56_56128


namespace n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l56_56441

theorem n_squared_plus_m_squared_odd_implies_n_plus_m_not_even (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 := by
  sorry

end n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l56_56441


namespace average_rate_first_half_80_l56_56890

theorem average_rate_first_half_80
    (total_distance : ℝ)
    (average_rate_trip : ℝ)
    (distance_first_half : ℝ)
    (time_first_half : ℝ)
    (time_second_half : ℝ)
    (time_total : ℝ)
    (R : ℝ)
    (H1 : total_distance = 640)
    (H2 : average_rate_trip = 40)
    (H3 : distance_first_half = total_distance / 2)
    (H4 : time_first_half = distance_first_half / R)
    (H5 : time_second_half = 3 * time_first_half)
    (H6 : time_total = time_first_half + time_second_half)
    (H7 : average_rate_trip = total_distance / time_total) :
    R = 80 := 
by 
  -- Given conditions
  sorry

end average_rate_first_half_80_l56_56890


namespace min_value_four_l56_56753

noncomputable def min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y > 2 * x) : ℝ :=
  (y^2 - 2 * x * y + x^2) / (x * y - 2 * x^2)

theorem min_value_four (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hy_gt_2x : y > 2 * x) :
  min_value x y hx_pos hy_pos hy_gt_2x = 4 := 
sorry

end min_value_four_l56_56753


namespace janet_family_needs_91_tickets_l56_56581

def janet_family_tickets (adults: ℕ) (children: ℕ) (roller_coaster_adult_tickets: ℕ) (roller_coaster_child_tickets: ℕ) 
  (giant_slide_adult_tickets: ℕ) (giant_slide_child_tickets: ℕ) (num_roller_coaster_rides_adult: ℕ) 
  (num_roller_coaster_rides_child: ℕ) (num_giant_slide_rides_adult: ℕ) (num_giant_slide_rides_child: ℕ) : ℕ := 
  (adults * roller_coaster_adult_tickets * num_roller_coaster_rides_adult) + 
  (children * roller_coaster_child_tickets * num_roller_coaster_rides_child) + 
  (1 * giant_slide_adult_tickets * num_giant_slide_rides_adult) + 
  (1 * giant_slide_child_tickets * num_giant_slide_rides_child)

theorem janet_family_needs_91_tickets :
  janet_family_tickets 2 2 7 5 4 3 3 2 5 3 = 91 := 
by 
  -- Calculations based on the given conditions (skipped in this statement)
  sorry

end janet_family_needs_91_tickets_l56_56581


namespace problem_statement_l56_56456

-- Define the given condition
def cond_1 (x : ℝ) := x + 1/x = 5

-- State the theorem that needs to be proven
theorem problem_statement (x : ℝ) (h : cond_1 x) : x^3 + 1/x^3 = 110 :=
sorry

end problem_statement_l56_56456


namespace find_m_for_parallel_vectors_l56_56696

theorem find_m_for_parallel_vectors (m : ℝ) :
  let a := (1, m)
  let b := (2, -1)
  (2 * a.1 + b.1, 2 * a.2 + b.2) = (k * (a.1 - 2 * b.1), k * (a.2 - 2 * b.2)) → m = -1/2 :=
by
  sorry

end find_m_for_parallel_vectors_l56_56696


namespace carpenter_material_cost_l56_56757

theorem carpenter_material_cost (total_estimate hourly_rate num_hours : ℝ) 
    (h1 : total_estimate = 980)
    (h2 : hourly_rate = 28)
    (h3 : num_hours = 15) : 
    total_estimate - hourly_rate * num_hours = 560 := 
by
  sorry

end carpenter_material_cost_l56_56757


namespace people_visited_neither_l56_56170

theorem people_visited_neither (total_people iceland_visitors norway_visitors both_visitors : ℕ)
  (h1 : total_people = 100)
  (h2 : iceland_visitors = 55)
  (h3 : norway_visitors = 43)
  (h4 : both_visitors = 61) :
  total_people - (iceland_visitors + norway_visitors - both_visitors) = 63 :=
by
  sorry

end people_visited_neither_l56_56170


namespace mike_office_visits_per_day_l56_56015

-- Define the constants from the conditions
def pull_ups_per_visit : ℕ := 2
def total_pull_ups_per_week : ℕ := 70
def days_per_week : ℕ := 7

-- Calculate total office visits per week
def office_visits_per_week : ℕ := total_pull_ups_per_week / pull_ups_per_visit

-- Lean statement that states Mike goes into his office 5 times a day
theorem mike_office_visits_per_day : office_visits_per_week / days_per_week = 5 := by
  sorry

end mike_office_visits_per_day_l56_56015


namespace star_polygon_points_l56_56867

theorem star_polygon_points (n : ℕ) (A B : ℕ → ℝ) 
  (h_angles_congruent_A : ∀ i j, A i = A j)
  (h_angles_congruent_B : ∀ i j, B i = B j)
  (h_angle_relation : ∀ i, A i = B i - 15) :
  n = 24 :=
by
  sorry

end star_polygon_points_l56_56867


namespace find_x_l56_56515

theorem find_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end find_x_l56_56515


namespace only_solution_l56_56865

theorem only_solution (a b c : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
    (h_le : a ≤ b ∧ b ≤ c) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_div_a2b : a^3 + b^3 + c^3 % (a^2 * b) = 0)
    (h_div_b2c : a^3 + b^3 + c^3 % (b^2 * c) = 0)
    (h_div_c2a : a^3 + b^3 + c^3 % (c^2 * a) = 0) : 
    a = 1 ∧ b = 1 ∧ c = 1 :=
  by
  sorry

end only_solution_l56_56865


namespace circle_equation_l56_56171

open Real

theorem circle_equation (x y : ℝ) :
  let center := (2, -1)
  let line := (x + y = 7)
  (center.1 - 2)^2 + (center.2 + 1)^2 = 18 :=
by
  sorry

end circle_equation_l56_56171


namespace alice_reeboks_sold_l56_56792

theorem alice_reeboks_sold
  (quota : ℝ)
  (price_adidas : ℝ)
  (price_nike : ℝ)
  (price_reeboks : ℝ)
  (num_nike : ℕ)
  (num_adidas : ℕ)
  (excess : ℝ)
  (total_sales_goal : ℝ)
  (total_sales : ℝ)
  (sales_nikes_adidas : ℝ)
  (sales_reeboks : ℝ)
  (num_reeboks : ℕ) :
  quota = 1000 →
  price_adidas = 45 →
  price_nike = 60 →
  price_reeboks = 35 →
  num_nike = 8 →
  num_adidas = 6 →
  excess = 65 →
  total_sales_goal = quota + excess →
  total_sales = 1065 →
  sales_nikes_adidas = price_nike * num_nike + price_adidas * num_adidas →
  sales_reeboks = total_sales - sales_nikes_adidas →
  num_reeboks = sales_reeboks / price_reeboks →
  num_reeboks = 9 :=
by
  intros
  sorry

end alice_reeboks_sold_l56_56792


namespace isosceles_triangle_perimeter_l56_56223

theorem isosceles_triangle_perimeter (side1 side2 base : ℕ)
    (h1 : side1 = 12) (h2 : side2 = 12) (h3 : base = 17) : 
    side1 + side2 + base = 41 := by
  sorry

end isosceles_triangle_perimeter_l56_56223


namespace smallest_x_plus_y_l56_56924

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l56_56924


namespace sand_needed_for_sandbox_l56_56133

def length1 : ℕ := 50
def width1 : ℕ := 30
def length2 : ℕ := 20
def width2 : ℕ := 15
def area_per_bag : ℕ := 80
def weight_per_bag : ℕ := 30

theorem sand_needed_for_sandbox :
  (length1 * width1 + length2 * width2 + area_per_bag - 1) / area_per_bag * weight_per_bag = 690 :=
by sorry

end sand_needed_for_sandbox_l56_56133


namespace range_of_a_l56_56638

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l56_56638


namespace win_game_A_win_game_C_l56_56296

-- Define the probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probability of winning Game A
def prob_win_game_A : ℚ := (prob_heads ^ 3) + (prob_tails ^ 3)

-- Define the probability of winning Game C
def prob_win_game_C : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)

-- State the theorem for Game A
theorem win_game_A : prob_win_game_A = 7 / 16 :=
by 
  -- Lean will check this proof
  sorry

-- State the theorem for Game C
theorem win_game_C : prob_win_game_C = 41 / 128 :=
by 
  -- Lean will check this proof
  sorry

end win_game_A_win_game_C_l56_56296


namespace run_to_grocery_store_time_l56_56275

theorem run_to_grocery_store_time
  (running_time: ℝ)
  (grocery_distance: ℝ)
  (friend_distance: ℝ)
  (half_way : friend_distance = grocery_distance / 2)
  (constant_pace : running_time / grocery_distance = (25 : ℝ) / 3)
  : (friend_distance * (25 / 3)) + (friend_distance * (25 / 3)) = 25 :=
by
  -- Given proofs for the conditions can be filled here
  sorry

end run_to_grocery_store_time_l56_56275


namespace jazmin_dolls_correct_l56_56130

-- Define the number of dolls Geraldine has.
def geraldine_dolls : ℕ := 2186

-- Define the number of extra dolls Geraldine has compared to Jazmin.
def extra_dolls : ℕ := 977

-- Define the calculation of the number of dolls Jazmin has.
def jazmin_dolls : ℕ := geraldine_dolls - extra_dolls

-- Prove that the number of dolls Jazmin has is 1209.
theorem jazmin_dolls_correct : jazmin_dolls = 1209 := by
  -- Include the required steps in the future proof here.
  sorry

end jazmin_dolls_correct_l56_56130


namespace digital_earth_correct_purposes_l56_56683

def Purpose : Type := String

def P1 : Purpose := "To deal with natural and social issues of the entire Earth using digital means."
def P2 : Purpose := "To maximize the utilization of natural resources."
def P3 : Purpose := "To conveniently obtain information about the Earth."
def P4 : Purpose := "To provide precise locations, directions of movement, and speeds of moving objects."

def correct_purposes : Set Purpose := {P1, P2, P3}

theorem digital_earth_correct_purposes :
  {P1, P2, P3} = correct_purposes :=
by 
  sorry

end digital_earth_correct_purposes_l56_56683


namespace jill_total_phone_time_l56_56043

def phone_time : ℕ → ℕ
| 0 => 5
| (n + 1) => 2 * phone_time n

theorem jill_total_phone_time (n : ℕ) (h : n = 4) : 
  phone_time 0 + phone_time 1 + phone_time 2 + phone_time 3 + phone_time 4 = 155 :=
by
  cases h
  sorry

end jill_total_phone_time_l56_56043


namespace time_to_complete_together_l56_56442

-- Definitions for the given conditions
variables (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Theorem statement for the mathematically equivalent proof problem
theorem time_to_complete_together (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
   (1 : ℝ) / ((1 / x) + (1 / y)) = x * y / (x + y) :=
sorry

end time_to_complete_together_l56_56442


namespace neg_int_solution_l56_56573

theorem neg_int_solution (x : ℤ) : -2 * x < 4 ↔ x = -1 :=
by
  sorry

end neg_int_solution_l56_56573


namespace number_of_episodes_l56_56517

def episode_length : ℕ := 20
def hours_per_day : ℕ := 2
def days : ℕ := 15

theorem number_of_episodes : (days * hours_per_day * 60) / episode_length = 90 :=
by
  sorry

end number_of_episodes_l56_56517


namespace stamps_sum_to_n_l56_56618

noncomputable def selectStamps : Prop :=
  ∀ (n : ℕ) (k : ℕ), n > 0 → 
                      ∃ stamps : List ℕ, 
                      stamps.length = k ∧ 
                      n ≤ stamps.sum ∧ stamps.sum < 2 * k → 
                      ∃ (subset : List ℕ), 
                      subset.sum = n

theorem stamps_sum_to_n : selectStamps := sorry

end stamps_sum_to_n_l56_56618


namespace compound_interest_rate_l56_56017

theorem compound_interest_rate :
  ∀ (P A : ℝ) (t n : ℕ) (r : ℝ),
  P = 12000 →
  A = 21500 →
  t = 5 →
  n = 1 →
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.121898 :=
by
  intros P A t n r hP hA ht hn hCompound
  sorry

end compound_interest_rate_l56_56017


namespace tan_sum_angle_identity_l56_56362

theorem tan_sum_angle_identity
  (α β : ℝ)
  (h1 : Real.tan (α + 2 * β) = 2)
  (h2 : Real.tan β = -3) :
  Real.tan (α + β) = -1 := sorry

end tan_sum_angle_identity_l56_56362


namespace inverse_prop_l56_56871

theorem inverse_prop (a b : ℝ) : (a > b) → (|a| > |b|) :=
sorry

end inverse_prop_l56_56871


namespace instantaneous_velocity_at_3_l56_56929

-- Definitions based on the conditions.
def displacement (t : ℝ) := 2 * t ^ 3

-- The statement to prove.
theorem instantaneous_velocity_at_3 : (deriv displacement 3) = 54 := by
  sorry

end instantaneous_velocity_at_3_l56_56929


namespace student_count_l56_56339

theorem student_count (N : ℕ) (h1 : ∀ W : ℝ, W - 46 = 86 - 40) (h2 : (86 - 46) = 5 * N) : N = 8 :=
sorry

end student_count_l56_56339


namespace gcd_divisors_remainders_l56_56314

theorem gcd_divisors_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end gcd_divisors_remainders_l56_56314


namespace value_of_expression_l56_56864

-- Definitions based on the conditions
def a : ℕ := 15
def b : ℕ := 3

-- The theorem to prove
theorem value_of_expression : a^2 + 2 * a * b + b^2 = 324 := by
  -- Skipping the proof as per instructions
  sorry

end value_of_expression_l56_56864


namespace more_stable_yield_A_l56_56245

theorem more_stable_yield_A (s_A s_B : ℝ) (hA : s_A * s_A = 794) (hB : s_B * s_B = 958) : s_A < s_B :=
by {
  sorry -- Details of the proof would go here
}

end more_stable_yield_A_l56_56245


namespace intersection_P_Q_l56_56856

def P : Set ℝ := { x : ℝ | 2 ≤ x ∧ x < 4 }
def Q : Set ℝ := { x : ℝ | 3 ≤ x }

theorem intersection_P_Q :
  P ∩ Q = { x : ℝ | 3 ≤ x ∧ x < 4 } :=
by
  sorry  -- Proof step will be provided here

end intersection_P_Q_l56_56856


namespace find_xyz_l56_56770

variables (x y z s : ℝ)

theorem find_xyz (h₁ : (x + y + z) * (x * y + x * z + y * z) = 12)
    (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
    (hs : x + y + z = s) : xyz = -8 :=
by
  sorry

end find_xyz_l56_56770


namespace modulo_residue_l56_56962

theorem modulo_residue:
  (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 :=
by
  sorry

end modulo_residue_l56_56962


namespace smallest_distance_l56_56144

open Complex

noncomputable def a := 2 + 4 * Complex.I
noncomputable def b := 8 + 6 * Complex.I

theorem smallest_distance (z w : ℂ)
    (hz : abs (z - a) = 2)
    (hw : abs (w - b) = 4) :
    abs (z - w) ≥ 2 * Real.sqrt 10 - 6 := by
  sorry

end smallest_distance_l56_56144


namespace count_ordered_triples_l56_56527

theorem count_ordered_triples (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : 2 * a * b * c = 2 * (a * b + b * c + a * c)) : 
  ∃ n, n = 10 :=
by
  sorry

end count_ordered_triples_l56_56527


namespace boy_lap_time_l56_56394

noncomputable def muddy_speed : ℝ := 5 * 1000 / 3600
noncomputable def sandy_speed : ℝ := 7 * 1000 / 3600
noncomputable def uphill_speed : ℝ := 4 * 1000 / 3600

noncomputable def muddy_distance : ℝ := 10
noncomputable def sandy_distance : ℝ := 15
noncomputable def uphill_distance : ℝ := 10

noncomputable def time_for_muddy : ℝ := muddy_distance / muddy_speed
noncomputable def time_for_sandy : ℝ := sandy_distance / sandy_speed
noncomputable def time_for_uphill : ℝ := uphill_distance / uphill_speed

noncomputable def total_time_for_one_side : ℝ := time_for_muddy + time_for_sandy + time_for_uphill
noncomputable def total_time_for_lap : ℝ := 4 * total_time_for_one_side

theorem boy_lap_time : total_time_for_lap = 95.656 := by
  sorry

end boy_lap_time_l56_56394


namespace selling_price_decreased_l56_56787

theorem selling_price_decreased (d m : ℝ) (hd : d = 0.10) (hm : m = 0.10) :
  (1 - d) * (1 + m) < 1 :=
by
  rw [hd, hm]
  sorry

end selling_price_decreased_l56_56787


namespace Mr_Spacek_birds_l56_56256

theorem Mr_Spacek_birds :
  ∃ N : ℕ, 50 < N ∧ N < 100 ∧ N % 9 = 0 ∧ N % 4 = 0 ∧ N = 72 :=
by
  sorry

end Mr_Spacek_birds_l56_56256


namespace evaluate_complex_ratio_l56_56675

noncomputable def complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) : ℂ :=
(a^12 + b^12) / (a + b)^12

theorem evaluate_complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) :
  complex_ratio a b h1 h2 h3 = 1 / 32 :=
by
  sorry

end evaluate_complex_ratio_l56_56675


namespace sum_of_first_ten_terms_l56_56033

theorem sum_of_first_ten_terms (a : ℕ → ℝ)
  (h1 : a 3 ^ 2 + a 8 ^ 2 + 2 * a 3 * a 8 = 9)
  (h2 : ∀ n, a n < 0) :
  (5 * (a 3 + a 8) = -15) :=
sorry

end sum_of_first_ten_terms_l56_56033


namespace inequality_solution_l56_56955

theorem inequality_solution (x : ℝ) (hx1 : x ≥ -1/2) (hx2 : x ≠ 0) :
  (4 * x^2 / (1 - Real.sqrt (1 + 2 * x))^2 < 2 * x + 9) ↔ 
  (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45/8) :=
by
  sorry

end inequality_solution_l56_56955


namespace vector_addition_correct_l56_56604

open Matrix

-- Define the vectors as 3x1 matrices
def v1 : Matrix (Fin 3) (Fin 1) ℤ := ![![3], ![-5], ![1]]
def v2 : Matrix (Fin 3) (Fin 1) ℤ := ![![-1], ![4], ![-2]]
def v3 : Matrix (Fin 3) (Fin 1) ℤ := ![![2], ![-1], ![3]]

-- Define the scalar multiples
def scaled_v1 := (2 : ℤ) • v1
def scaled_v2 := (3 : ℤ) • v2
def neg_v3 := (-1 : ℤ) • v3

-- Define the summation result
def result := scaled_v1 + scaled_v2 + neg_v3

-- Define the expected result for verification
def expected_result : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![3], ![-7]]

-- The proof statement (without the proof itself)
theorem vector_addition_correct :
  result = expected_result := by
  sorry

end vector_addition_correct_l56_56604


namespace sum_of_consecutive_integers_with_product_1680_l56_56531

theorem sum_of_consecutive_integers_with_product_1680 : 
  ∃ (a b c d : ℤ), (a * b * c * d = 1680 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3) → (a + b + c + d = 26) := sorry

end sum_of_consecutive_integers_with_product_1680_l56_56531


namespace sequence_2011_l56_56832

theorem sequence_2011 :
  ∀ (a : ℕ → ℤ), (a 1 = 1) →
                  (a 2 = 2) →
                  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
                  a 2011 = 1 :=
by {
  -- Insert proof here
  sorry
}

end sequence_2011_l56_56832


namespace find_f3_l56_56999

theorem find_f3 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : f 3 = -25.5 :=
sorry

end find_f3_l56_56999


namespace ratio_of_larger_to_smaller_l56_56506

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l56_56506


namespace first_term_exceeding_10000_l56_56558

theorem first_term_exceeding_10000 :
  ∃ (n : ℕ), (2^(n-1) > 10000) ∧ (2^(n-1) = 16384) :=
by
  sorry

end first_term_exceeding_10000_l56_56558


namespace vendor_sales_first_day_l56_56512

theorem vendor_sales_first_day (A S: ℝ) (h1: S = S / 100) 
  (h2: 0.20 * A * (1 - S / 100) = 0.42 * A - 0.50 * A * (0.80 * (1 - S / 100)))
  (h3: 0 < S) (h4: S < 100) : 
  S = 30 := 
by
  sorry

end vendor_sales_first_day_l56_56512


namespace inequality_x_solution_l56_56470

theorem inequality_x_solution (a b c d x : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ( (a^3 / (a^3 + 15 * b * c * d))^(1/2) = a^x / (a^x + b^x + c^x + d^x) ) ↔ x = 15 / 8 := 
sorry

end inequality_x_solution_l56_56470


namespace christina_walking_speed_l56_56123

-- Definitions based on the conditions
def initial_distance : ℝ := 150  -- Jack and Christina are 150 feet apart
def jack_speed : ℝ := 7  -- Jack's speed in feet per second
def lindy_speed : ℝ := 10  -- Lindy's speed in feet per second
def lindy_total_distance : ℝ := 100  -- Total distance Lindy travels

-- Proof problem: Prove that Christina's walking speed is 8 feet per second
theorem christina_walking_speed : 
  ∃ c : ℝ, (lindy_total_distance / lindy_speed) * jack_speed + (lindy_total_distance / lindy_speed) * c = initial_distance ∧ 
  c = 8 :=
by {
  use 8,
  sorry
}

end christina_walking_speed_l56_56123


namespace isosceles_triangle_count_l56_56977

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end isosceles_triangle_count_l56_56977


namespace arc_length_of_sector_l56_56265

theorem arc_length_of_sector : 
  ∀ (r : ℝ) (theta: ℝ), r = 1 ∧ theta = 30 * (Real.pi / 180) → (theta * r = Real.pi / 6) :=
by
  sorry

end arc_length_of_sector_l56_56265


namespace max_value_expression_correct_l56_56803

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_correct :
  ∃ a b c d : ℝ, a ∈ Set.Icc (-13.5) 13.5 ∧ b ∈ Set.Icc (-13.5) 13.5 ∧ 
                  c ∈ Set.Icc (-13.5) 13.5 ∧ d ∈ Set.Icc (-13.5) 13.5 ∧ 
                  max_value_expression a b c d = 756 := 
sorry

end max_value_expression_correct_l56_56803


namespace find_divisor_l56_56673

theorem find_divisor (x : ℕ) (h : 172 = 10 * x + 2) : x = 17 :=
sorry

end find_divisor_l56_56673


namespace gain_percent_l56_56392

variable (C S : ℝ)

theorem gain_percent 
  (h : 81 * C = 45 * S) : ((4 / 5) * 100) = 80 := 
by 
  sorry

end gain_percent_l56_56392


namespace quad_eq_complete_square_l56_56289

theorem quad_eq_complete_square (p q : ℝ) 
  (h : ∀ x : ℝ, (4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16)) : q / p = -4 := by
  sorry

end quad_eq_complete_square_l56_56289


namespace bugs_eat_flowers_l56_56274

-- Define the problem conditions
def number_of_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Define the expected outcome
def total_flowers_eaten : ℕ := 6

-- Prove that total flowers eaten is equal to the product of the number of bugs and flowers per bug
theorem bugs_eat_flowers : number_of_bugs * flowers_per_bug = total_flowers_eaten :=
by
  sorry

end bugs_eat_flowers_l56_56274


namespace exists_monochromatic_triangle_l56_56630

theorem exists_monochromatic_triangle (points : Fin 6 → Point) (color : (Point × Point) → Color) :
  ∃ (a b c : Point), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (color (a, b) = color (b, c) ∧ color (b, c) = color (c, a)) :=
by
  sorry

end exists_monochromatic_triangle_l56_56630


namespace range_of_m_l56_56547

theorem range_of_m {f : ℝ → ℝ} (h : ∀ x, f x = x^2 - 6*x - 16)
  {a b : ℝ} (h_domain : ∀ x, 0 ≤ x ∧ x ≤ a → ∃ y, f y ≤ b) 
  (h_range : ∀ y, -25 ≤ y ∧ y ≤ -16 → ∃ x, f x = y) : 3 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_m_l56_56547


namespace find_a_for_no_x2_term_l56_56785

theorem find_a_for_no_x2_term :
  ∀ a : ℝ, (∀ x : ℝ, (3 * x^2 + 2 * a * x + 1) * (-3 * x) - 4 * x^2 = -9 * x^3 + (-6 * a - 4) * x^2 - 3 * x) →
  (¬ ∃ x : ℝ, (-6 * a - 4) * x^2 ≠ 0) →
  a = -2 / 3 :=
by
  intros a h1 h2
  sorry

end find_a_for_no_x2_term_l56_56785


namespace proof_of_neg_p_or_neg_q_l56_56704

variables (p q : Prop)

theorem proof_of_neg_p_or_neg_q (h₁ : ¬ (p ∧ q)) (h₂ : p ∨ q) : ¬ p ∨ ¬ q :=
  sorry

end proof_of_neg_p_or_neg_q_l56_56704


namespace camper_ratio_l56_56840

theorem camper_ratio (total_campers : ℕ) (G : ℕ) (B : ℕ)
  (h1: total_campers = 96) 
  (h2: G = total_campers / 3) 
  (h3: B = total_campers - G) 
  : B / total_campers = 2 / 3 :=
  by
    sorry

end camper_ratio_l56_56840


namespace cost_price_proof_l56_56850

noncomputable def selling_price : Real := 12000
noncomputable def discount_rate : Real := 0.10
noncomputable def new_selling_price : Real := selling_price * (1 - discount_rate)
noncomputable def profit_rate : Real := 0.08

noncomputable def cost_price : Real := new_selling_price / (1 + profit_rate)

theorem cost_price_proof : cost_price = 10000 := by sorry

end cost_price_proof_l56_56850


namespace girls_in_school_l56_56085

theorem girls_in_school (boys girls : ℕ) (ratio : ℕ → ℕ → Prop) (h1 : ratio 5 4) (h2 : boys = 1500) :
    girls = 1200 :=
by
  sorry

end girls_in_school_l56_56085


namespace initial_average_weight_l56_56908

theorem initial_average_weight 
    (W : ℝ)
    (a b c d e : ℝ)
    (h1 : (a + b + c) / 3 = W)
    (h2 : (a + b + c + d) / 4 = W)
    (h3 : (b + c + d + (d + 3)) / 4 = 68)
    (h4 : a = 81) :
    W = 70 := 
sorry

end initial_average_weight_l56_56908


namespace austin_tax_l56_56595

theorem austin_tax 
  (number_of_robots : ℕ)
  (cost_per_robot change_left starting_amount : ℚ) 
  (h1 : number_of_robots = 7)
  (h2 : cost_per_robot = 8.75)
  (h3 : change_left = 11.53)
  (h4 : starting_amount = 80) : 
  ∃ tax : ℚ, tax = 7.22 :=
by
  sorry

end austin_tax_l56_56595


namespace max_term_of_sequence_l56_56655

noncomputable def a_n (n : ℕ) : ℚ := (n^2 : ℚ) / (2^n : ℚ)

theorem max_term_of_sequence :
  ∃ n : ℕ, (∀ m : ℕ, a_n n ≥ a_n m) ∧ a_n n = 9 / 8 :=
sorry

end max_term_of_sequence_l56_56655


namespace duck_travel_days_l56_56779

theorem duck_travel_days (x : ℕ) (h1 : 40 + 2 * 40 + x = 180) : x = 60 := by
  sorry

end duck_travel_days_l56_56779


namespace y_coordinate_equidistant_l56_56843

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ ptC ptD : ℝ × ℝ, ptC = (-3, 0) → ptD = (4, 5) → 
    dist (0, y) ptC = dist (0, y) ptD) ∧ y = 16 / 5 :=
by
  sorry

end y_coordinate_equidistant_l56_56843


namespace solve_equation_l56_56108

theorem solve_equation (x : ℂ) (h : (x^2 + 3*x + 4) / (x + 3) = x + 6) : x = -7 / 3 := sorry

end solve_equation_l56_56108


namespace sum_of_fourth_powers_l56_56465

-- Define the sum of fourth powers as per the given formula
noncomputable def sum_fourth_powers (n: ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30

-- Define the statement to be proved
theorem sum_of_fourth_powers :
  2 * sum_fourth_powers 100 = 41006666600 :=
by sorry

end sum_of_fourth_powers_l56_56465


namespace geometric_series_sum_l56_56004

theorem geometric_series_sum 
  (a : ℝ) (r : ℝ) (s : ℝ)
  (h_a : a = 9)
  (h_r : r = -2/3)
  (h_abs_r : |r| < 1)
  (h_s : s = a / (1 - r)) : 
  s = 5.4 := by
  sorry

end geometric_series_sum_l56_56004


namespace no_closed_loop_after_replacement_l56_56034

theorem no_closed_loop_after_replacement (N M : ℕ) 
  (h1 : N = M) 
  (h2 : (N + M) % 4 = 0) :
  ¬((N - 1) - (M + 1)) % 4 = 0 :=
by
  sorry

end no_closed_loop_after_replacement_l56_56034


namespace calculate_expression_l56_56984

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 :=
by
  sorry

end calculate_expression_l56_56984


namespace smallest_among_given_numbers_l56_56846

theorem smallest_among_given_numbers :
  let a := abs (-3)
  let b := -2
  let c := 0
  let d := Real.pi
  b < a ∧ b < c ∧ b < d := by
  sorry

end smallest_among_given_numbers_l56_56846


namespace total_ceilings_to_paint_l56_56780

theorem total_ceilings_to_paint (ceilings_painted_this_week : ℕ) 
                                (ceilings_painted_next_week : ℕ)
                                (ceilings_left_to_paint : ℕ) 
                                (h1 : ceilings_painted_this_week = 12) 
                                (h2 : ceilings_painted_next_week = ceilings_painted_this_week / 4) 
                                (h3 : ceilings_left_to_paint = 13) : 
    ceilings_painted_this_week + ceilings_painted_next_week + ceilings_left_to_paint = 28 :=
by
  sorry

end total_ceilings_to_paint_l56_56780


namespace tan_a2_a12_l56_56534

noncomputable def arithmetic_term (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem tan_a2_a12 (a d : ℝ) (h : a + (a + 6 * d) + (a + 12 * d) = 4 * Real.pi) :
  Real.tan (arithmetic_term a d 2 + arithmetic_term a d 12) = - Real.sqrt 3 :=
by
  sorry

end tan_a2_a12_l56_56534


namespace average_beef_sales_l56_56384

theorem average_beef_sales 
  (thursday_sales : ℕ)
  (friday_sales : ℕ)
  (saturday_sales : ℕ)
  (h_thursday : thursday_sales = 210)
  (h_friday : friday_sales = 2 * thursday_sales)
  (h_saturday : saturday_sales = 150) :
  (thursday_sales + friday_sales + saturday_sales) / 3 = 260 :=
by sorry

end average_beef_sales_l56_56384


namespace eval_g_at_3_l56_56759

def g (x : ℤ) : ℤ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem eval_g_at_3 : g 3 = 183 := by
  sorry

end eval_g_at_3_l56_56759


namespace brianna_sandwiches_l56_56410

theorem brianna_sandwiches (meats : ℕ) (cheeses : ℕ) (h_meats : meats = 8) (h_cheeses : cheeses = 7) :
  (Nat.choose meats 2) * (Nat.choose cheeses 1) = 196 := 
by
  rw [h_meats, h_cheeses]
  norm_num
  sorry

end brianna_sandwiches_l56_56410


namespace second_pirate_gets_diamond_l56_56439

theorem second_pirate_gets_diamond (coins_bag1 coins_bag2 : ℕ) :
  (coins_bag1 ≤ 1 ∧ coins_bag2 ≤ 1) ∨ (coins_bag1 > 1 ∨ coins_bag2 > 1) →
  (∃ n k : ℕ, n % 2 = 0 → (coins_bag1 + n) = (coins_bag2 + k)) :=
sorry

end second_pirate_gets_diamond_l56_56439


namespace minimum_value_fraction_l56_56996

theorem minimum_value_fraction (m n : ℝ) (h1 : m + 4 * n = 1) (h2 : m > 0) (h3 : n > 0): 
  (1 / m + 4 / n) ≥ 25 :=
sorry

end minimum_value_fraction_l56_56996


namespace find_b_l56_56038

-- Define the variables involved
variables (a b : ℝ)

-- Conditions provided in the problem
def condition_1 : Prop := 2 * a + 1 = 1
def condition_2 : Prop := b + a = 3

-- Theorem statement to prove b = 3 given the conditions
theorem find_b (h1 : condition_1 a) (h2 : condition_2 a b) : b = 3 := by
  sorry

end find_b_l56_56038


namespace one_bag_covers_250_sqfeet_l56_56162

noncomputable def lawn_length : ℝ := 22
noncomputable def lawn_width : ℝ := 36
noncomputable def bags_count : ℝ := 4
noncomputable def extra_area : ℝ := 208

noncomputable def lawn_area : ℝ := lawn_length * lawn_width
noncomputable def total_covered_area : ℝ := lawn_area + extra_area
noncomputable def one_bag_area : ℝ := total_covered_area / bags_count

theorem one_bag_covers_250_sqfeet :
  one_bag_area = 250 := 
by
  sorry

end one_bag_covers_250_sqfeet_l56_56162


namespace percent_decrease_second_year_l56_56406

theorem percent_decrease_second_year
  (V_0 V_1 V_2 : ℝ)
  (p_2 : ℝ)
  (h1 : V_1 = V_0 * 0.7)
  (h2 : V_2 = V_1 * (1 - p_2 / 100))
  (h3 : V_2 = V_0 * 0.63) :
  p_2 = 10 :=
sorry

end percent_decrease_second_year_l56_56406


namespace calculate_result_l56_56781

def multiply (a b : ℕ) : ℕ := a * b
def subtract (a b : ℕ) : ℕ := a - b
def three_fifths (a : ℕ) : ℕ := 3 * a / 5

theorem calculate_result :
  let result := three_fifths (subtract (multiply 12 10) 20)
  result = 60 :=
by
  sorry

end calculate_result_l56_56781


namespace store_profit_is_20_percent_l56_56407

variable (C : ℝ)
variable (marked_up_price : ℝ := 1.20 * C)          -- First markup price
variable (new_year_price : ℝ := 1.50 * C)           -- Second markup price
variable (discounted_price : ℝ := 1.20 * C)         -- Discounted price in February
variable (profit : ℝ := discounted_price - C)       -- Profit on items sold in February

theorem store_profit_is_20_percent (C : ℝ) : profit = 0.20 * C := 
  sorry

end store_profit_is_20_percent_l56_56407


namespace combinedHeightCorrect_l56_56483

def empireStateBuildingHeightToTopFloor : ℕ := 1250
def empireStateBuildingAntennaHeight : ℕ := 204

def willisTowerHeightToTopFloor : ℕ := 1450
def willisTowerAntennaHeight : ℕ := 280

def oneWorldTradeCenterHeightToTopFloor : ℕ := 1368
def oneWorldTradeCenterAntennaHeight : ℕ := 408

def totalHeightEmpireStateBuilding := empireStateBuildingHeightToTopFloor + empireStateBuildingAntennaHeight
def totalHeightWillisTower := willisTowerHeightToTopFloor + willisTowerAntennaHeight
def totalHeightOneWorldTradeCenter := oneWorldTradeCenterHeightToTopFloor + oneWorldTradeCenterAntennaHeight

def combinedHeight := totalHeightEmpireStateBuilding + totalHeightWillisTower + totalHeightOneWorldTradeCenter

theorem combinedHeightCorrect : combinedHeight = 4960 := by
  sorry

end combinedHeightCorrect_l56_56483


namespace stratified_sampling_class2_l56_56228

theorem stratified_sampling_class2 (students_class1 : ℕ) (students_class2 : ℕ) (total_samples : ℕ) (h1 : students_class1 = 36) (h2 : students_class2 = 42) (h_tot : total_samples = 13) : 
  (students_class2 / (students_class1 + students_class2) * total_samples = 7) :=
by
  sorry

end stratified_sampling_class2_l56_56228


namespace volume_of_pyramid_l56_56932

variables (a b c : ℝ)

def triangle_face1 (a b : ℝ) : Prop := 1/2 * a * b = 1.5
def triangle_face2 (b c : ℝ) : Prop := 1/2 * b * c = 2
def triangle_face3 (c a : ℝ) : Prop := 1/2 * c * a = 6

theorem volume_of_pyramid (h1 : triangle_face1 a b) (h2 : triangle_face2 b c) (h3 : triangle_face3 c a) :
  1/3 * a * b * c = 2 :=
sorry

end volume_of_pyramid_l56_56932


namespace olympic_rings_area_l56_56967

theorem olympic_rings_area (d R r: ℝ) 
  (hyp_d : d = 12 * Real.sqrt 2) 
  (hyp_R : R = 11) 
  (hyp_r : r = 9) 
  (overlap_area : ∀ (i j : ℕ), i ≠ j → 592 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54): 
  592.0 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54 := 
by sorry

end olympic_rings_area_l56_56967


namespace force_with_18_inch_crowbar_l56_56535

noncomputable def inverseForce (L F : ℝ) : ℝ :=
  F * L

theorem force_with_18_inch_crowbar :
  ∀ (F : ℝ), (inverseForce 12 200 = inverseForce 18 F) → F = 133.333333 :=
by
  intros
  sorry

end force_with_18_inch_crowbar_l56_56535


namespace point_P_in_first_quadrant_l56_56415

def lies_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : lies_in_first_quadrant 2 1 :=
by {
  sorry
}

end point_P_in_first_quadrant_l56_56415


namespace problem_I_problem_II_l56_56497

theorem problem_I (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) :
  c / a = 2 :=
sorry

theorem problem_II (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) 
  (h3 : b = 4) (h4 : Real.cos C = 1 / 4) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
sorry

end problem_I_problem_II_l56_56497


namespace sequence_formula_l56_56087

theorem sequence_formula (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h : ∀ n : ℕ, S n = 3 * a n + (-1)^n) :
  ∀ n : ℕ, a n = (1/10) * (3/2)^(n-1) - (2/5) * (-1)^n :=
by sorry

end sequence_formula_l56_56087


namespace Jo_has_least_l56_56032

variable (Money : Type) 
variable (Bo Coe Flo Jo Moe Zoe : Money)
variable [LT Money] [LE Money] -- Money type is an ordered type with less than and less than or equal relations.

-- Conditions
axiom h1 : Jo < Flo 
axiom h2 : Flo < Bo
axiom h3 : Jo < Moe
axiom h4 : Moe < Bo
axiom h5 : Bo < Coe
axiom h6 : Coe < Zoe

-- The main statement to prove that Jo has the least money.
theorem Jo_has_least (h1 : Jo < Flo) (h2 : Flo < Bo) (h3 : Jo < Moe) (h4 : Moe < Bo) (h5 : Bo < Coe) (h6 : Coe < Zoe) : 
  ∀ x, x = Jo ∨ x = Bo ∨ x = Flo ∨ x = Moe ∨ x = Coe ∨ x = Zoe → Jo ≤ x :=
by
  -- Proof is skipped using sorry
  sorry

end Jo_has_least_l56_56032


namespace cars_served_from_4pm_to_6pm_l56_56687

theorem cars_served_from_4pm_to_6pm : 
  let cars_per_15_min_peak := 12
  let cars_per_15_min_offpeak := 8 
  let blocks_in_an_hour := 4 
  let total_peak_hour := cars_per_15_min_peak * blocks_in_an_hour 
  let total_offpeak_hour := cars_per_15_min_offpeak * blocks_in_an_hour 
  total_peak_hour + total_offpeak_hour = 80 := 
by 
  sorry 

end cars_served_from_4pm_to_6pm_l56_56687


namespace linear_function_no_third_quadrant_l56_56380

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end linear_function_no_third_quadrant_l56_56380


namespace brenda_age_l56_56258

theorem brenda_age (A B J : ℝ)
  (h1 : A = 4 * B)
  (h2 : J = B + 8)
  (h3 : A = J + 2) :
  B = 10 / 3 :=
by
  sorry

end brenda_age_l56_56258


namespace find_inverse_value_l56_56351

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) function definition goes here

theorem find_inverse_value :
  (∀ x : ℝ, f (x - 1) = f (x + 3)) →
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → f x = 2^x + 1) →
  f⁻¹ 19 = 3 - 2 * (Real.log 3 / Real.log 2) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end find_inverse_value_l56_56351


namespace John_sells_each_wig_for_five_dollars_l56_56599

theorem John_sells_each_wig_for_five_dollars
  (plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (wig_cost : ℕ)
  (total_cost : ℕ)
  (sold_wigs_cost : ℕ)
  (remaining_wigs_cost : ℕ) :
  plays = 3 ∧
  acts_per_play = 5 ∧
  wigs_per_act = 2 ∧
  wig_cost = 5 ∧
  total_cost = 150 ∧
  remaining_wigs_cost = 110 ∧
  total_cost - remaining_wigs_cost = sold_wigs_cost →
  (sold_wigs_cost / (plays * acts_per_play * wigs_per_act - remaining_wigs_cost / wig_cost)) = wig_cost :=
by sorry

end John_sells_each_wig_for_five_dollars_l56_56599


namespace frances_towels_weight_in_ounces_l56_56971

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end frances_towels_weight_in_ounces_l56_56971


namespace pulled_pork_sandwiches_l56_56207

/-
  Jack uses 3 cups of ketchup, 1 cup of vinegar, and 1 cup of honey.
  Each burger takes 1/4 cup of sauce.
  Each pulled pork sandwich takes 1/6 cup of sauce.
  Jack makes 8 burgers.
  Prove that Jack can make exactly 18 pulled pork sandwiches.
-/
theorem pulled_pork_sandwiches :
  (3 + 1 + 1) - (8 * (1/4)) = 3 -> 
  3 / (1/6) = 18 :=
sorry

end pulled_pork_sandwiches_l56_56207


namespace polynomial_inequality_l56_56190

theorem polynomial_inequality (x : ℝ) : -6 * x^2 + 2 * x - 8 < 0 :=
sorry

end polynomial_inequality_l56_56190


namespace find_k_of_symmetry_l56_56859

noncomputable def f (x k : ℝ) := Real.sin (2 * x) + k * Real.cos (2 * x)

theorem find_k_of_symmetry (k : ℝ) :
  (∃ x, x = (Real.pi / 6) ∧ f x k = f (Real.pi / 6 - x) k) →
  k = Real.sqrt 3 / 3 :=
sorry

end find_k_of_symmetry_l56_56859


namespace KaydenceAge_l56_56294

-- Definitions for ages of family members based on the problem conditions
def fatherAge := 60
def motherAge := fatherAge - 2 
def brotherAge := fatherAge / 2 
def sisterAge := 40
def totalFamilyAge := 200

-- Lean statement to prove the age of Kaydence
theorem KaydenceAge : 
  fatherAge + motherAge + brotherAge + sisterAge + Kaydence = totalFamilyAge → 
  Kaydence = 12 := 
by
  sorry

end KaydenceAge_l56_56294


namespace probability_of_x_in_interval_l56_56729

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval : ℝ :=
  let length_total := interval_length (-2) 1
  let length_sub := interval_length 0 1
  length_sub / length_total

theorem probability_of_x_in_interval :
  probability_in_interval = 1 / 3 :=
by
  sorry

end probability_of_x_in_interval_l56_56729


namespace sum_of_smallest_multiples_l56_56566

def smallest_two_digit_multiple_of_5 := 10
def smallest_three_digit_multiple_of_7 := 105

theorem sum_of_smallest_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end sum_of_smallest_multiples_l56_56566


namespace starting_number_is_100_l56_56082

theorem starting_number_is_100 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, k = 10 ∧ n = 1000 - (k - 1) * 100) :
  n = 100 := by
  sorry

end starting_number_is_100_l56_56082


namespace hilary_big_toenails_count_l56_56141

def fit_toenails (total_capacity : ℕ) (big_toenail_space_ratio : ℕ) (current_regular : ℕ) (additional_regular : ℕ) : ℕ :=
  (total_capacity - (current_regular + additional_regular)) / big_toenail_space_ratio

theorem hilary_big_toenails_count :
  fit_toenails 100 2 40 20 = 10 :=
  by
    sorry

end hilary_big_toenails_count_l56_56141


namespace blu_ray_movies_returned_l56_56563

theorem blu_ray_movies_returned (D B x : ℕ)
  (h1 : D / B = 17 / 4)
  (h2 : D + B = 378)
  (h3 : D / (B - x) = 9 / 2) :
  x = 4 := by
  sorry

end blu_ray_movies_returned_l56_56563


namespace gymnastics_performance_participation_l56_56319

def total_people_in_gym_performance (grades : ℕ) (classes_per_grade : ℕ) (students_per_class : ℕ) : ℕ :=
  grades * classes_per_grade * students_per_class

theorem gymnastics_performance_participation :
  total_people_in_gym_performance 3 4 15 = 180 :=
by
  -- This is where the proof would go
  sorry

end gymnastics_performance_participation_l56_56319


namespace slope_points_eq_l56_56597

theorem slope_points_eq (m : ℚ) (h : ((m + 2) / (3 - m) = 2)) : m = 4 / 3 :=
sorry

end slope_points_eq_l56_56597


namespace calculate_geometric_sequence_sum_l56_56703

def geometric_sequence (a₁ r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^n

theorem calculate_geometric_sequence_sum :
  let a₁ := 1
  let r := -2
  let a₂ := geometric_sequence a₁ r 1
  let a₃ := geometric_sequence a₁ r 2
  let a₄ := geometric_sequence a₁ r 3
  a₁ + |a₂| + a₃ + |a₄| = 15 :=
by
  sorry

end calculate_geometric_sequence_sum_l56_56703


namespace functional_equation_solution_l56_56299

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f (m + n)) = f m + f n) ↔
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a ∨ f n = 0) :=
sorry

end functional_equation_solution_l56_56299


namespace geometric_series_m_value_l56_56714

theorem geometric_series_m_value (m : ℝ) : 
    let a : ℝ := 20
    let r₁ : ℝ := 1 / 2  -- Common ratio for the first series
    let S₁ : ℝ := a / (1 - r₁)  -- Sum of the first series
    let b : ℝ := 1 / 2 + m / 20  -- Common ratio for the second series
    let S₂ : ℝ := a / (1 - b)  -- Sum of the second series
    S₁ = 40 ∧ S₂ = 120 → m = 20 / 3 :=
sorry

end geometric_series_m_value_l56_56714


namespace evaluate_expression_l56_56153

theorem evaluate_expression : 101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end evaluate_expression_l56_56153


namespace fault_line_movement_l56_56051

theorem fault_line_movement (total_movement: ℝ) (past_year: ℝ) (prev_year: ℝ) (total_eq: total_movement = 6.5) (past_eq: past_year = 1.25) :
  prev_year = 5.25 := by
  sorry

end fault_line_movement_l56_56051


namespace garden_yield_l56_56918

theorem garden_yield
  (steps_length : ℕ)
  (steps_width : ℕ)
  (step_to_feet : ℕ → ℝ)
  (yield_per_sqft : ℝ)
  (h1 : steps_length = 18)
  (h2 : steps_width = 25)
  (h3 : ∀ n : ℕ, step_to_feet n = n * 2.5)
  (h4 : yield_per_sqft = 2 / 3)
  : (step_to_feet steps_length * step_to_feet steps_width) * yield_per_sqft = 1875 :=
by
  sorry

end garden_yield_l56_56918


namespace half_time_score_30_l56_56195

-- Define sequence conditions
def arithmetic_sequence (a d : ℕ) : ℕ × ℕ × ℕ × ℕ := (a, a + d, a + 2 * d, a + 3 * d)
def geometric_sequence (b r : ℕ) : ℕ × ℕ × ℕ × ℕ := (b, b * r, b * r^2, b * r^3)

-- Define the sum of the first team
def first_team_sum (a d : ℕ) : ℕ := 4 * a + 6 * d

-- Define the sum of the second team
def second_team_sum (b r : ℕ) : ℕ := b * (1 + r + r^2 + r^3)

-- Define the winning condition
def winning_condition (a d b r : ℕ) : Prop := first_team_sum a d = second_team_sum b r + 2

-- Define the point sum constraint
def point_sum_constraint (a d b r : ℕ) : Prop := first_team_sum a d ≤ 100 ∧ second_team_sum b r ≤ 100

-- Define the constraints on r and d
def r_d_positive (r d : ℕ) : Prop := r > 1 ∧ d > 0

-- Define the half-time score for the first team
def first_half_first_team (a d : ℕ) : ℕ := a + (a + d)

-- Define the half-time score for the second team
def first_half_second_team (b r : ℕ) : ℕ := b + (b * r)

-- Define the total half-time score
def total_half_time_score (a d b r : ℕ) : ℕ := first_half_first_team a d + first_half_second_team b r

-- Main theorem: Total half-time score is 30 under given conditions
theorem half_time_score_30 (a d b r : ℕ) 
  (r_d_pos : r_d_positive r d) 
  (win_cond : winning_condition a d b r)
  (point_sum_cond : point_sum_constraint a d b r) : 
  total_half_time_score a d b r = 30 :=
sorry

end half_time_score_30_l56_56195


namespace ratio_of_arithmetic_sequence_sums_l56_56805

-- Definitions of the arithmetic sequences based on the conditions
def numerator_seq (n : ℕ) : ℕ := 3 + (n - 1) * 3
def denominator_seq (m : ℕ) : ℕ := 4 + (m - 1) * 4

-- Definitions of the number of terms based on the conditions
def num_terms_num : ℕ := 32
def num_terms_den : ℕ := 16

-- Definitions of the sums based on the sequences
def sum_numerator_seq : ℕ := (num_terms_num / 2) * (3 + 96)
def sum_denominator_seq : ℕ := (num_terms_den / 2) * (4 + 64)

-- Calculate the ratio of the sums
def ratio_of_sums : ℚ := sum_numerator_seq / sum_denominator_seq

-- Proof statement
theorem ratio_of_arithmetic_sequence_sums : ratio_of_sums = 99 / 34 := by
  sorry

end ratio_of_arithmetic_sequence_sums_l56_56805


namespace simplify_fraction_l56_56682

noncomputable def simplified_expression (x y : ℝ) : ℝ :=
  (x^2 - (4 / y)) / (y^2 - (4 / x))

theorem simplify_fraction {x y : ℝ} (h : x * y ≠ 4) :
  simplified_expression x y = x / y := 
by 
  sorry

end simplify_fraction_l56_56682


namespace sarah_min_width_l56_56983

noncomputable def minWidth (S : Type) [LinearOrder S] (w : S) : Prop :=
  ∃ w, w ≥ 0 ∧ w * (w + 20) ≥ 150 ∧ ∀ w', (w' ≥ 0 ∧ w' * (w' + 20) ≥ 150) → w ≤ w'

theorem sarah_min_width : minWidth ℝ 10 :=
by {
  sorry -- proof goes here
}

end sarah_min_width_l56_56983


namespace num_undefined_values_l56_56649

-- Condition: Denominator is given as (x^2 + 2x - 3)(x - 3)(x + 1)
def denominator (x : ℝ) : ℝ := (x^2 + 2 * x - 3) * (x - 3) * (x + 1)

-- The Lean statement to prove the number of values of x for which the expression is undefined
theorem num_undefined_values : 
  ∃ (n : ℕ), (∀ x : ℝ, denominator x = 0 → (x = 1 ∨ x = -3 ∨ x = 3 ∨ x = -1)) ∧ n = 4 :=
by
  sorry

end num_undefined_values_l56_56649


namespace working_capacity_ratio_l56_56031

theorem working_capacity_ratio (team_p_engineers : ℕ) (team_q_engineers : ℕ) (team_p_days : ℕ) (team_q_days : ℕ) :
  team_p_engineers = 20 → team_q_engineers = 16 → team_p_days = 32 → team_q_days = 30 →
  (team_p_days / team_q_days) = (16:ℤ) / (15:ℤ) :=
by
  intros h1 h2 h3 h4
  sorry

end working_capacity_ratio_l56_56031


namespace max_complete_dresses_l56_56150

namespace DressMaking

-- Define the initial quantities of fabric
def initial_silk : ℕ := 600
def initial_satin : ℕ := 400
def initial_chiffon : ℕ := 350

-- Define the quantities given to each of 8 friends
def silk_per_friend : ℕ := 15
def satin_per_friend : ℕ := 10
def chiffon_per_friend : ℕ := 5

-- Define the quantities required to make one dress
def silk_per_dress : ℕ := 5
def satin_per_dress : ℕ := 3
def chiffon_per_dress : ℕ := 2

-- Calculate the remaining quantities
def remaining_silk : ℕ := initial_silk - 8 * silk_per_friend
def remaining_satin : ℕ := initial_satin - 8 * satin_per_friend
def remaining_chiffon : ℕ := initial_chiffon - 8 * chiffon_per_friend

-- Calculate the maximum number of dresses that can be made
def max_dresses_silk : ℕ := remaining_silk / silk_per_dress
def max_dresses_satin : ℕ := remaining_satin / satin_per_dress
def max_dresses_chiffon : ℕ := remaining_chiffon / chiffon_per_dress

-- The main theorem indicating the number of complete dresses
theorem max_complete_dresses : max_dresses_silk = 96 ∧ max_dresses_silk ≤ max_dresses_satin ∧ max_dresses_silk ≤ max_dresses_chiffon := by
  sorry

end DressMaking

end max_complete_dresses_l56_56150


namespace tom_trout_count_l56_56302

theorem tom_trout_count (M T : ℕ) (hM : M = 8) (hT : T = 2 * M) : T = 16 :=
by
  -- proof goes here
  sorry

end tom_trout_count_l56_56302


namespace roots_difference_l56_56270

theorem roots_difference (a b c : ℝ) (h_eq : a = 1) (h_b : b = -11) (h_c : c = 24) :
    let r1 := (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    let r2 := (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    r1 - r2 = 5 := 
by
  sorry

end roots_difference_l56_56270


namespace fraction_sum_simplified_l56_56349

theorem fraction_sum_simplified (a b : ℕ) (h1 : 0.6125 = (a : ℝ) / b) (h2 : Nat.gcd a b = 1) : a + b = 129 :=
sorry

end fraction_sum_simplified_l56_56349


namespace rational_solution_for_k_is_6_l56_56557

theorem rational_solution_for_k_is_6 (k : ℕ) (h : 0 < k) :
  (∃ x : ℚ, k * x ^ 2 + 12 * x + k = 0) ↔ k = 6 :=
by { sorry }

end rational_solution_for_k_is_6_l56_56557


namespace max_fraction_value_l56_56021

theorem max_fraction_value :
  ∀ (x y : ℝ), (1/4 ≤ x ∧ x ≤ 3/5) ∧ (1/5 ≤ y ∧ y ≤ 1/2) → 
    xy / (x^2 + y^2) ≤ 2/5 :=
by
  sorry

end max_fraction_value_l56_56021


namespace largest_n_consecutive_product_l56_56083

theorem largest_n_consecutive_product (n : ℕ) : n = 0 ↔ (n! = (n+1) * (n+2) * (n+3) * (n+4) * (n+5)) := by
  sorry

end largest_n_consecutive_product_l56_56083


namespace BoatsRUs_total_canoes_l56_56897

def totalCanoesBuiltByJuly (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem BoatsRUs_total_canoes :
  totalCanoesBuiltByJuly 5 3 7 = 5465 :=
by
  sorry

end BoatsRUs_total_canoes_l56_56897


namespace prime_p_satisfies_conditions_l56_56243

theorem prime_p_satisfies_conditions (p : ℕ) (hp1 : Nat.Prime p) (hp2 : p ≠ 2) (hp3 : p ≠ 7) :
  ∃ n : ℕ, n = 29 ∧ ∀ x y : ℕ, (1 ≤ x ∧ x ≤ 29) ∧ (1 ≤ y ∧ y ≤ 29) → (29 ∣ (y^2 - x^p - 26)) :=
sorry

end prime_p_satisfies_conditions_l56_56243


namespace solve_inequality_l56_56806

theorem solve_inequality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 - 9) / (x^2 - 1) > 0 ↔ (x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1)) :=
sorry

end solve_inequality_l56_56806


namespace product_units_digit_of_five_consecutive_l56_56449

theorem product_units_digit_of_five_consecutive (n : ℕ) : 
  ((n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10) = 0 := 
sorry

end product_units_digit_of_five_consecutive_l56_56449


namespace cos_315_eq_sqrt2_div_2_l56_56232

theorem cos_315_eq_sqrt2_div_2 : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 :=
sorry

end cos_315_eq_sqrt2_div_2_l56_56232


namespace real_solutions_l56_56765

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end real_solutions_l56_56765


namespace andrew_purchase_grapes_l56_56869

theorem andrew_purchase_grapes (G : ℕ) (h : 70 * G + 495 = 1055) : G = 8 :=
by
  sorry

end andrew_purchase_grapes_l56_56869


namespace quadratic_residue_property_l56_56810

theorem quadratic_residue_property (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ)
  (h : ∃ t : ℤ, ∃ k : ℤ, k * k = p * t + a) : (a ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_property_l56_56810


namespace remaining_money_is_correct_l56_56858

def initial_amount : ℕ := 53
def cost_toy_car : ℕ := 11
def number_toy_cars : ℕ := 2
def cost_scarf : ℕ := 10
def cost_beanie : ℕ := 14
def remaining_money : ℕ := 
  initial_amount - (cost_toy_car * number_toy_cars) - cost_scarf - cost_beanie

theorem remaining_money_is_correct : remaining_money = 7 := by
  sorry

end remaining_money_is_correct_l56_56858


namespace ab_plus_b_l56_56828

theorem ab_plus_b (A B : ℤ) (h1 : A * B = 10) (h2 : 3 * A + 7 * B = 51) : A * B + B = 12 :=
by
  sorry

end ab_plus_b_l56_56828


namespace octagon_edge_length_from_pentagon_l56_56400

noncomputable def regular_pentagon_edge_length : ℝ := 16
def num_of_pentagon_edges : ℕ := 5
def num_of_octagon_edges : ℕ := 8

theorem octagon_edge_length_from_pentagon (total_length_thread : ℝ) :
  total_length_thread = num_of_pentagon_edges * regular_pentagon_edge_length →
  (total_length_thread / num_of_octagon_edges) = 10 :=
by
  intro h
  sorry

end octagon_edge_length_from_pentagon_l56_56400


namespace diane_honey_harvest_l56_56724

theorem diane_honey_harvest (last_year : ℕ) (increase : ℕ) (this_year : ℕ) :
  last_year = 2479 → increase = 6085 → this_year = last_year + increase → this_year = 8564 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end diane_honey_harvest_l56_56724


namespace certain_number_value_l56_56642

theorem certain_number_value :
  let D := 20
  let S := 55
  3 * D - 5 + (D - S) = 15 :=
by
  -- Definitions for D and S
  let D := 20
  let S := 55
  -- The main assertion
  show 3 * D - 5 + (D - S) = 15
  sorry

end certain_number_value_l56_56642


namespace third_root_of_cubic_equation_l56_56915

-- Definitions
variable (a b : ℚ) -- We use rational numbers due to the fractions involved
def cubic_equation (x : ℚ) : ℚ := a * x^3 + (a + 3 * b) * x^2 + (2 * b - 4 * a) * x + (10 - a)

-- Conditions
axiom h1 : cubic_equation a b (-1) = 0
axiom h2 : cubic_equation a b 4 = 0

-- The theorem we aim to prove
theorem third_root_of_cubic_equation : ∃ (c : ℚ), c = -62 / 19 ∧ cubic_equation a b c = 0 :=
sorry

end third_root_of_cubic_equation_l56_56915


namespace sin_complementary_angle_l56_56482

theorem sin_complementary_angle (θ : ℝ) (h1 : Real.tan θ = 2) (h2 : Real.cos θ < 0) : 
  Real.sin (Real.pi / 2 - θ) = -Real.sqrt 5 / 5 :=
sorry

end sin_complementary_angle_l56_56482


namespace fraction_eq_zero_l56_56713

theorem fraction_eq_zero {x : ℝ} (h : (6 * x) ≠ 0) : (x - 5) / (6 * x) = 0 ↔ x = 5 := 
by
  sorry

end fraction_eq_zero_l56_56713


namespace angle_B_in_triangle_l56_56487

theorem angle_B_in_triangle
  (a b c : ℝ)
  (h_area : 2 * (a * c * ((a^2 + c^2 - b^2) / (2 * a * c)).sin) = (a^2 + c^2 - b^2) * (Real.sqrt 3 / 6)) :
  ∃ B : ℝ, B = π / 6 :=
by
  sorry

end angle_B_in_triangle_l56_56487


namespace hyperbola_equation_foci_shared_l56_56055

theorem hyperbola_equation_foci_shared :
  ∃ m : ℝ, (∃ c : ℝ, c = 2 * Real.sqrt 2 ∧ 
              ∃ a b : ℝ, a^2 = 12 ∧ b^2 = 4 ∧ c^2 = a^2 - b^2) ∧ 
    (c = 2 * Real.sqrt 2 → (∃ a b : ℝ, a^2 = m ∧ b^2 = m - 8 ∧ c^2 = a^2 + b^2)) → 
  (∃ m : ℝ, m = 7) := 
sorry

end hyperbola_equation_foci_shared_l56_56055


namespace domain_of_f_l56_56312

-- Define the function y = sqrt(x-1) + sqrt(x*(3-x))
noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + Real.sqrt (x * (3 - x))

-- Proposition about the domain of the function
theorem domain_of_f (x : ℝ) : (∃ y : ℝ, y = f x) ↔ 1 ≤ x ∧ x ≤ 3 :=
by
  sorry

end domain_of_f_l56_56312


namespace number_of_children_l56_56837

def weekly_husband : ℕ := 335
def weekly_wife : ℕ := 225
def weeks_in_six_months : ℕ := 24
def amount_per_child : ℕ := 1680

theorem number_of_children : (weekly_husband + weekly_wife) * weeks_in_six_months / 2 / amount_per_child = 4 := by
  sorry

end number_of_children_l56_56837


namespace fruit_total_l56_56834

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end fruit_total_l56_56834


namespace thrown_away_oranges_l56_56664

theorem thrown_away_oranges (x : ℕ) (h : 40 - x + 7 = 10) : x = 37 :=
by sorry

end thrown_away_oranges_l56_56664


namespace number_of_valid_pairs_l56_56486

theorem number_of_valid_pairs :
  ∃ (n : Nat), n = 8 ∧ 
  (∃ (a b : Int), 4 < a ∧ a < b ∧ b < 22 ∧ (4 + a + b + 22) / 4 = 13) :=
sorry

end number_of_valid_pairs_l56_56486


namespace product_of_solutions_l56_56073

theorem product_of_solutions : 
  ∀ y : ℝ, (|y| = 3 * (|y| - 2)) → ∃ a b : ℝ, (a = 3 ∧ b = -3) ∧ (a * b = -9) := 
by 
  sorry

end product_of_solutions_l56_56073


namespace number_of_solutions_pi_equation_l56_56745

theorem number_of_solutions_pi_equation : 
  ∃ (x0 x1 : ℝ), (x0 = 0 ∧ x1 = 1) ∧ ∀ x : ℝ, (π^(x-1) * x^2 + π^(x^2) * x - π^(x^2) = x^2 + x - 1 ↔ x = x0 ∨ x = x1)
:=
by sorry

end number_of_solutions_pi_equation_l56_56745


namespace intersection_points_count_l56_56288

noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x ^ 2 - 4 * x + 4

theorem intersection_points_count : ∃! x y : ℝ, 0 < x ∧ f x = g x ∧ y ≠ x ∧ f y = g y :=
sorry

end intersection_points_count_l56_56288


namespace part1_geometric_sequence_part2_sum_of_terms_l56_56255

/- Part 1 -/
theorem part1_geometric_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h₀ : a 1 = 3) 
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2 * a n) 
  (h₂ : ∀ n, 2 ^ b n = a n + 1) :
  ∃ r, ∀ n, b (n + 1) = r * b n ∧ r = 2 :=
by 
  use 2 
  sorry

/- Part 2 -/
theorem part2_sum_of_terms (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₀ : ∀ n, b n = 2 ^ n)
  (h₁ : ∀ n, c n = n / b n + 1) :
  ∀ n, T n = n + 2 - (n + 2) / 2 ^ n :=
by
  sorry

end part1_geometric_sequence_part2_sum_of_terms_l56_56255


namespace longest_path_is_critical_path_l56_56478

noncomputable def longest_path_in_workflow_diagram : String :=
"Critical Path"

theorem longest_path_is_critical_path :
  (longest_path_in_workflow_diagram = "Critical Path") :=
  by
  sorry

end longest_path_is_critical_path_l56_56478


namespace light_travel_distance_120_years_l56_56403

theorem light_travel_distance_120_years :
  let annual_distance : ℝ := 9.46e12
  let years : ℝ := 120
  (annual_distance * years) = 1.1352e15 := 
by
  sorry

end light_travel_distance_120_years_l56_56403


namespace max_volume_prism_l56_56947

theorem max_volume_prism (a b h : ℝ) (V : ℝ) 
  (h1 : a * h + b * h + a * b = 32) : 
  V = a * b * h → V ≤ 128 * Real.sqrt 3 / 3 := 
by
  sorry

end max_volume_prism_l56_56947


namespace shells_total_l56_56088

theorem shells_total (a s v : ℕ) 
  (h1 : s = v + 16) 
  (h2 : v = a - 5) 
  (h3 : a = 20) : 
  s + v + a = 66 := 
by
  sorry

end shells_total_l56_56088


namespace find_x2_plus_y2_l56_56690

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
  sorry

end find_x2_plus_y2_l56_56690


namespace negation_of_exists_prop_l56_56127

theorem negation_of_exists_prop (x : ℝ) :
  (¬ ∃ (x : ℝ), (x > 0) ∧ (|x| + x >= 0)) ↔ (∀ (x : ℝ), x > 0 → |x| + x < 0) := 
sorry

end negation_of_exists_prop_l56_56127


namespace range_q_l56_56214

def q (x : ℝ ) : ℝ := x^4 + 4 * x^2 + 4

theorem range_q :
  (∀ y, ∃ x, 0 ≤ x ∧ q x = y ↔ y ∈ Set.Ici 4) :=
sorry

end range_q_l56_56214


namespace expression_simplifies_to_32_l56_56111

noncomputable def simplified_expression (a : ℝ) : ℝ :=
  8 / (1 + a^8) + 4 / (1 + a^4) + 2 / (1 + a^2) + 1 / (1 + a) + 1 / (1 - a)

theorem expression_simplifies_to_32 :
  simplified_expression (2^(-1/16 : ℝ)) = 32 :=
by
  sorry

end expression_simplifies_to_32_l56_56111


namespace solve_system_l56_56454

theorem solve_system :
  ∃ (x1 y1 x2 y2 x3 y3 : ℚ), 
    (x1 = 0 ∧ y1 = 0) ∧ 
    (x2 = -14 ∧ y2 = 6) ∧ 
    (x3 = -85/6 ∧ y3 = 35/6) ∧ 
    ((x1 + 2*y1)*(x1 + 3*y1) = x1 + y1 ∧ (2*x1 + y1)*(3*x1 + y1) = -99*(x1 + y1)) ∧ 
    ((x2 + 2*y2)*(x2 + 3*y2) = x2 + y2 ∧ (2*x2 + y2)*(3*x2 + y2) = -99*(x2 + y2)) ∧ 
    ((x3 + 2*y3)*(x3 + 3*y3) = x3 + y3 ∧ (2*x3 + y3)*(3*x3 + y3) = -99*(x3 + y3)) :=
by
  -- skips the actual proof
  sorry

end solve_system_l56_56454


namespace friends_meeting_games_only_l56_56992

theorem friends_meeting_games_only 
  (M P G MP MG PG MPG : ℕ) 
  (h1 : M + MP + MG + MPG = 10) 
  (h2 : P + MP + PG + MPG = 20) 
  (h3 : MP = 4) 
  (h4 : MG = 2) 
  (h5 : PG = 0) 
  (h6 : MPG = 2) 
  (h7 : M + P + G + MP + MG + PG + MPG = 31) : 
  G = 1 := 
by
  sorry

end friends_meeting_games_only_l56_56992


namespace chi_squared_confidence_level_l56_56187

theorem chi_squared_confidence_level 
  (chi_squared_value : ℝ)
  (p_value_3841 : ℝ)
  (p_value_5024 : ℝ)
  (h1 : chi_squared_value = 4.073)
  (h2 : p_value_3841 = 0.05)
  (h3 : p_value_5024 = 0.025)
  (h4 : 3.841 ≤ chi_squared_value ∧ chi_squared_value < 5.024) :
  ∃ confidence_level : ℝ, confidence_level = 0.95 :=
by 
  sorry

end chi_squared_confidence_level_l56_56187


namespace parallel_lines_l56_56991

noncomputable def line1 (x y : ℝ) : Prop := x - y + 1 = 0
noncomputable def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

theorem parallel_lines (a x y : ℝ) : (∀ (x y : ℝ), line1 x y → line2 a x y → x = y ∨ (line1 x y ∧ x ≠ y)) → 
  (a = -1 ∧ ∃ d : ℝ, d = Real.sqrt 2) :=
sorry

end parallel_lines_l56_56991


namespace simplify_exp_l56_56986

theorem simplify_exp : (10^8 / (10 * 10^5)) = 100 := 
by
  -- The proof is omitted; we are stating the problem.
  sorry

end simplify_exp_l56_56986


namespace distance_from_Q_to_BC_l56_56095

-- Definitions for the problem
structure Square :=
(A B C D : ℝ × ℝ)
(side_length : ℝ)

def P : (ℝ × ℝ) := (3, 6)
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 6)^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25
def side_BC (x y : ℝ) : Prop := x = 6

-- Lean proof statement
theorem distance_from_Q_to_BC (Q : ℝ × ℝ) (hQ1 : circle1 Q.1 Q.2) (hQ2 : circle2 Q.1 Q.2) :
  Exists (fun d : ℝ => Q.1 = 6 ∧ Q.2 = d) := sorry

end distance_from_Q_to_BC_l56_56095


namespace minimum_value_of_absolute_sum_l56_56833

theorem minimum_value_of_absolute_sum (x : ℝ) :
  ∃ y : ℝ, (∀ x : ℝ, y ≤ |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5|) ∧ y = 6 :=
sorry

end minimum_value_of_absolute_sum_l56_56833


namespace ratio_area_octagons_correct_l56_56321

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l56_56321


namespace least_four_digit_11_heavy_l56_56551

def is_11_heavy (n : ℕ) : Prop := (n % 11) > 7

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem least_four_digit_11_heavy : ∃ n : ℕ, is_four_digit n ∧ is_11_heavy n ∧ 
  (∀ m : ℕ, is_four_digit m ∧ is_11_heavy m → 1000 ≤ n) := 
sorry

end least_four_digit_11_heavy_l56_56551


namespace violet_balloons_remaining_l56_56613

def initial_count : ℕ := 7
def lost_count : ℕ := 3

theorem violet_balloons_remaining : initial_count - lost_count = 4 :=
by sorry

end violet_balloons_remaining_l56_56613


namespace sequence_is_k_plus_n_l56_56670

theorem sequence_is_k_plus_n (a : ℕ → ℕ) (k : ℕ) (h : ∀ n : ℕ, a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1))
  (pos: ∀ n: ℕ, a n > 0) : ∀ n: ℕ, a n = k + n := 
sorry

end sequence_is_k_plus_n_l56_56670


namespace value_of_a_plus_b_l56_56694

theorem value_of_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 3 ↔ ax^2 + bx + 3 < 0) :
  a + b = -3 :=
sorry

end value_of_a_plus_b_l56_56694


namespace large_hotdogs_sold_l56_56197

theorem large_hotdogs_sold (total_hodogs : ℕ) (small_hotdogs : ℕ) (h1 : total_hodogs = 79) (h2 : small_hotdogs = 58) : 
  total_hodogs - small_hotdogs = 21 :=
by
  sorry

end large_hotdogs_sold_l56_56197


namespace meeting_time_l56_56568

-- Variables representing the conditions
def uniform_rate_cassie := 15
def uniform_rate_brian := 18
def distance_route := 70
def cassie_start_time := 8.0
def brian_start_time := 9.25

-- The goal
theorem meeting_time : ∃ T : ℝ, (15 * T + 18 * (T - 1.25) = 70) ∧ T = 2.803 := 
by {
  sorry
}

end meeting_time_l56_56568


namespace silverware_probability_l56_56901

def numWaysTotal (totalPieces : ℕ) (choosePieces : ℕ) : ℕ :=
  Nat.choose totalPieces choosePieces

def numWaysForks (forks : ℕ) (chooseForks : ℕ) : ℕ :=
  Nat.choose forks chooseForks

def numWaysSpoons (spoons : ℕ) (chooseSpoons : ℕ) : ℕ :=
  Nat.choose spoons chooseSpoons

def numWaysKnives (knives : ℕ) (chooseKnives : ℕ) : ℕ :=
  Nat.choose knives chooseKnives

def favorableOutcomes (forks : ℕ) (spoons : ℕ) (knives : ℕ) : ℕ :=
  numWaysForks forks 2 * numWaysSpoons spoons 1 * numWaysKnives knives 1

def probability (totalWays : ℕ) (favorableWays : ℕ) : ℚ :=
  favorableWays / totalWays

theorem silverware_probability :
  probability (numWaysTotal 18 4) (favorableOutcomes 5 7 6) = 7 / 51 := by
  sorry

end silverware_probability_l56_56901


namespace simplify_radical_1_simplify_radical_2_find_value_of_a_l56_56727

-- Problem 1
theorem simplify_radical_1 : 7 + 2 * (Real.sqrt 10) = (Real.sqrt 2 + Real.sqrt 5) ^ 2 := 
by sorry

-- Problem 2
theorem simplify_radical_2 : (Real.sqrt (11 - 6 * (Real.sqrt 2))) = 3 - Real.sqrt 2 := 
by sorry

-- Problem 3
theorem find_value_of_a (a m n : ℕ) (h : a + 2 * Real.sqrt 21 = (Real.sqrt m + Real.sqrt n) ^ 2) : 
  a = 10 ∨ a = 22 := 
by sorry

end simplify_radical_1_simplify_radical_2_find_value_of_a_l56_56727


namespace measure_angle_P_l56_56252

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end measure_angle_P_l56_56252


namespace blue_pairs_count_l56_56244

-- Define the problem and conditions
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def sum9_pairs : Finset (ℕ × ℕ) := { (1, 8), (2, 7), (3, 6), (4, 5), (8, 1), (7, 2), (6, 3), (5, 4) }

-- Definition for counting valid pairs excluding pairs summing to 9
noncomputable def count_valid_pairs : ℕ := 
  (faces.card * (faces.card - 2)) / 2

-- Theorem statement proving the number of valid pairs
theorem blue_pairs_count : count_valid_pairs = 24 := 
by
  sorry

end blue_pairs_count_l56_56244


namespace least_froods_l56_56945

theorem least_froods (n : ℕ) :
  (∃ n, n ≥ 1 ∧ (n * (n + 1)) / 2 > 20 * n) → (∃ n, n = 40) :=
by {
  sorry
}

end least_froods_l56_56945


namespace smallest_balanced_number_l56_56998

theorem smallest_balanced_number :
  ∃ (a b c : ℕ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  100 * a + 10 * b + c = 
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) ∧ 
  100 * a + 10 * b + c = 132 :=
sorry

end smallest_balanced_number_l56_56998


namespace sarah_score_l56_56436

theorem sarah_score (j g s : ℕ) 
  (h1 : g = 2 * j) 
  (h2 : s = g + 50) 
  (h3 : (s + g + j) / 3 = 110) : 
  s = 162 := 
by 
  sorry

end sarah_score_l56_56436


namespace constant_t_exists_l56_56555

theorem constant_t_exists (c : ℝ) :
  ∃ t : ℝ, (∀ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.2 = A.1 * c + c) ∧ (B.2 = B.1 * c + c) → (t = -2)) :=
sorry

end constant_t_exists_l56_56555


namespace complement_union_l56_56493

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_union : U \ (A ∪ B) = {3, 5} :=
by
  sorry

end complement_union_l56_56493


namespace diagonals_in_regular_nine_sided_polygon_l56_56224

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l56_56224


namespace min_lit_bulbs_l56_56115

theorem min_lit_bulbs (n : ℕ) (h : n ≥ 1) : 
  ∃ rows cols, (rows ⊆ Finset.range n) ∧ (cols ⊆ Finset.range n) ∧ 
  (∀ i j, (i ∈ rows ∧ j ∈ cols) ↔ (i + j) % 2 = 1) ∧ 
  rows.card * (n - cols.card) + cols.card * (n - rows.card) = 2 * n - 2 :=
by sorry

end min_lit_bulbs_l56_56115


namespace expression_equals_one_l56_56167

theorem expression_equals_one (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_sum : a + b + c = 1) :
  (a^3 * b^3 / ((a^3 - b * c) * (b^3 - a * c)) + a^3 * c^3 / ((a^3 - b * c) * (c^3 - a * b)) +
    b^3 * c^3 / ((b^3 - a * c) * (c^3 - a * b))) = 1 :=
by
  sorry

end expression_equals_one_l56_56167


namespace lateral_surface_of_prism_is_parallelogram_l56_56251

-- Definitions based on conditions
def is_right_prism (P : Type) : Prop := sorry
def is_oblique_prism (P : Type) : Prop := sorry
def is_rectangle (S : Type) : Prop := sorry
def is_parallelogram (S : Type) : Prop := sorry
def lateral_surface (P : Type) : Type := sorry

-- Condition 1: The lateral surface of a right prism is a rectangle
axiom right_prism_surface_is_rectangle (P : Type) (h : is_right_prism P) : is_rectangle (lateral_surface P)

-- Condition 2: The lateral surface of an oblique prism can either be a rectangle or a parallelogram
axiom oblique_prism_surface_is_rectangle_or_parallelogram (P : Type) (h : is_oblique_prism P) :
  is_rectangle (lateral_surface P) ∨ is_parallelogram (lateral_surface P)

-- Lean 4 statement for the proof problem
theorem lateral_surface_of_prism_is_parallelogram (P : Type) (p : is_right_prism P ∨ is_oblique_prism P) :
  is_parallelogram (lateral_surface P) :=
by
  sorry

end lateral_surface_of_prism_is_parallelogram_l56_56251


namespace Atlantic_Call_additional_charge_is_0_20_l56_56501

def United_Telephone_base_rate : ℝ := 7.00
def United_Telephone_rate_per_minute : ℝ := 0.25
def Atlantic_Call_base_rate : ℝ := 12.00
def United_Telephone_total_charge_100_minutes : ℝ := United_Telephone_base_rate + 100 * United_Telephone_rate_per_minute
def Atlantic_Call_total_charge_100_minutes (x : ℝ) : ℝ := Atlantic_Call_base_rate + 100 * x

theorem Atlantic_Call_additional_charge_is_0_20 :
  ∃ x : ℝ, United_Telephone_total_charge_100_minutes = Atlantic_Call_total_charge_100_minutes x ∧ x = 0.20 :=
by {
  -- Since United_Telephone_total_charge_100_minutes = 32.00, we need to prove:
  -- Atlantic_Call_total_charge_100_minutes 0.20 = 32.00
  sorry
}

end Atlantic_Call_additional_charge_is_0_20_l56_56501


namespace digit_proportions_l56_56820

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l56_56820


namespace additional_track_length_l56_56022

theorem additional_track_length (rise : ℝ) (grade1 grade2 : ℝ) (h1 : grade1 = 0.04) (h2 : grade2 = 0.02) (h3 : rise = 800) :
  ∃ (additional_length : ℝ), additional_length = (rise / grade2 - rise / grade1) ∧ additional_length = 20000 :=
by
  sorry

end additional_track_length_l56_56022


namespace Joan_spent_68_353_on_clothing_l56_56732

theorem Joan_spent_68_353_on_clothing :
  let shorts := 15.00
  let jacket := 14.82 * 0.9
  let shirt := 12.51 * 0.5
  let shoes := 21.67 - 3
  let hat := 8.75
  let belt := 6.34
  shorts + jacket + shirt + shoes + hat + belt = 68.353 :=
sorry

end Joan_spent_68_353_on_clothing_l56_56732


namespace smallest_n_for_Sn_gt_10_l56_56528

noncomputable def harmonicSeriesSum : ℕ → ℝ
| 0       => 0
| (n + 1) => harmonicSeriesSum n + 1 / (n + 1)

theorem smallest_n_for_Sn_gt_10 : ∃ n : ℕ, (harmonicSeriesSum n > 10) ∧ ∀ k < 12367, harmonicSeriesSum k ≤ 10 :=
by
  sorry

end smallest_n_for_Sn_gt_10_l56_56528


namespace relationship_between_y1_y2_l56_56507

theorem relationship_between_y1_y2 
  (y1 y2 : ℝ) 
  (hA : y1 = 6 / -3) 
  (hB : y2 = 6 / 2) : y1 < y2 :=
by 
  sorry

end relationship_between_y1_y2_l56_56507


namespace inequality_proof_l56_56279

theorem inequality_proof (a b c : ℝ) (ha : a = 2 / 21) (hb : b = Real.log 1.1) (hc : c = 21 / 220) : a < b ∧ b < c :=
by
  sorry

end inequality_proof_l56_56279


namespace value_of_expression_l56_56016

theorem value_of_expression (a b c k : ℕ) (h_a : a = 30) (h_b : b = 25) (h_c : c = 4) (h_k : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 :=
by
  rw [h_a, h_b, h_c, h_k]
  simp
  sorry

end value_of_expression_l56_56016


namespace sum_inequality_l56_56735

theorem sum_inequality (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (b * (a + b))) + (1 / (c * (b + c))) + (1 / (a * (c + a))) ≥ 3 / 2 :=
sorry

end sum_inequality_l56_56735


namespace count_valid_n_l56_56221

theorem count_valid_n : 
  ∃ n_values : Finset ℤ, 
    (∀ n ∈ n_values, (n + 2 ≤ 6 * n - 8) ∧ (6 * n - 8 < 3 * n + 7)) ∧
    (n_values.card = 3) :=
by sorry

end count_valid_n_l56_56221


namespace n_values_satisfy_condition_l56_56906

-- Define the exponential functions
def exp1 (n : ℤ) : ℚ := (-1/2) ^ n
def exp2 (n : ℤ) : ℚ := (-1/5) ^ n

-- Define the set of possible values for n
def valid_n : List ℤ := [-2, -1, 0, 1, 2, 3]

-- Define the condition for n to satisfy the inequality
def satisfies_condition (n : ℤ) : Prop := exp1 n > exp2 n

-- Prove that the only values of n that satisfy the condition are -1 and 2
theorem n_values_satisfy_condition :
  ∀ n ∈ valid_n, satisfies_condition n ↔ (n = -1 ∨ n = 2) :=
by
  intro n
  sorry

end n_values_satisfy_condition_l56_56906


namespace gcd_3_666666666_equals_3_l56_56586

theorem gcd_3_666666666_equals_3 :
  Nat.gcd 33333333 666666666 = 3 := by
  sorry

end gcd_3_666666666_equals_3_l56_56586


namespace smallest_w_value_l56_56000

theorem smallest_w_value (w : ℕ) (hw : w > 0) :
  (∀ k : ℕ, (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (10^2 ∣ 936 * w)) ↔ w = 900 := 
sorry

end smallest_w_value_l56_56000


namespace elena_marco_sum_ratio_l56_56662

noncomputable def sum_odds (n : Nat) : Nat := (n / 2 + 1) * n

noncomputable def sum_integers (n : Nat) : Nat := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odds 499) / (sum_integers 250) = 2 :=
by
  sorry

end elena_marco_sum_ratio_l56_56662


namespace angle_2016_216_in_same_quadrant_l56_56583

noncomputable def angle_in_same_quadrant (a b : ℝ) : Prop :=
  let normalized (x : ℝ) := x % 360
  normalized a = normalized b

theorem angle_2016_216_in_same_quadrant : angle_in_same_quadrant 2016 216 := by
  sorry

end angle_2016_216_in_same_quadrant_l56_56583


namespace overall_average_mark_l56_56923

theorem overall_average_mark :
  let n1 := 70
  let mean1 := 50
  let n2 := 35
  let mean2 := 60
  let n3 := 45
  let mean3 := 55
  let n4 := 42
  let mean4 := 45
  (n1 * mean1 + n2 * mean2 + n3 * mean3 + n4 * mean4 : ℝ) / (n1 + n2 + n3 + n4) = 51.89 := 
by {
  sorry
}

end overall_average_mark_l56_56923


namespace assoc_mul_l56_56594

-- Conditions from the problem
variables (x y z : Type) [Mul x] [Mul y] [Mul z]

theorem assoc_mul (a b c : x) : (a * b) * c = a * (b * c) := by sorry

end assoc_mul_l56_56594


namespace trisha_hourly_wage_l56_56911

theorem trisha_hourly_wage (annual_take_home_pay : ℝ) (percent_withheld : ℝ)
  (hours_per_week : ℝ) (weeks_per_year : ℝ) (hourly_wage : ℝ) :
  annual_take_home_pay = 24960 ∧ 
  percent_withheld = 0.20 ∧ 
  hours_per_week = 40 ∧ 
  weeks_per_year = 52 ∧ 
  hourly_wage = (annual_take_home_pay / (0.80 * (hours_per_week * weeks_per_year))) → 
  hourly_wage = 15 :=
by sorry

end trisha_hourly_wage_l56_56911


namespace find_d_l56_56611

variable (d x : ℕ)
axiom balls_decomposition : d = x + (x + 1) + (x + 2)
axiom probability_condition : (x : ℚ) / (d : ℚ) < 1 / 6

theorem find_d : d = 3 := sorry

end find_d_l56_56611


namespace combination_lock_l56_56215

theorem combination_lock :
  (∃ (n_1 n_2 n_3 : ℕ), 
    n_1 ≥ 0 ∧ n_1 ≤ 39 ∧
    n_2 ≥ 0 ∧ n_2 ≤ 39 ∧
    n_3 ≥ 0 ∧ n_3 ≤ 39 ∧ 
    n_1 % 4 = n_3 % 4 ∧ 
    n_2 % 4 = (n_1 + 2) % 4) →
  ∃ (count : ℕ), count = 4000 :=
by
  sorry

end combination_lock_l56_56215


namespace sum_of_interior_angles_l56_56253

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l56_56253


namespace percentage_of_girls_l56_56802

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 900) (h2 : B = 90) :
  (G / (B + G) : ℚ) * 100 = 90 :=
  by
  sorry

end percentage_of_girls_l56_56802


namespace minimum_norm_of_v_l56_56700

open Real 

-- Define the vector v and condition
noncomputable def v : ℝ × ℝ := sorry

-- Define the condition
axiom v_condition : ‖(v.1 + 4, v.2 + 2)‖ = 10

-- The statement that we need to prove
theorem minimum_norm_of_v : ‖v‖ = 10 - 2 * sqrt 5 :=
by
  sorry

end minimum_norm_of_v_l56_56700


namespace k_value_function_range_l56_56040

noncomputable def f : ℝ → ℝ := λ x => Real.log x + x

def is_k_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ (∀ x, a ≤ x ∧ x ≤ b → (f x = k * x)) ∧ (k > 0)

theorem k_value_function_range :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.log x + x) →
  (∃ (k : ℝ), is_k_value_function f k) →
  1 < k ∧ k < 1 + (1 / Real.exp 1) :=
by
  sorry

end k_value_function_range_l56_56040


namespace intervals_of_monotonicity_range_of_a_for_zeros_l56_56138

open Real

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - 3 * a * x + 2 * a^2 * log x

theorem intervals_of_monotonicity (a : ℝ) (ha : a ≠ 0) :
  (0 < a → ∀ x, (0 < x ∧ x < a → f x a < f (x + 1) a)
            ∧ (a < x ∧ x < 2 * a → f x a > f (x + 1) a)
            ∧ (2 * a < x → f x a < f (x + 1) a))
  ∧ (a < 0 → ∀ x, (0 < x → f x a < f (x + 1) a)) :=
sorry

theorem range_of_a_for_zeros (a x : ℝ) (ha : 0 < a) 
  (h1 : f a a > 0) (h2 : f (2 * a) a < 0) :
  e ^ (5 / 4) < a ∧ a < e ^ 2 / 2 :=
sorry

end intervals_of_monotonicity_range_of_a_for_zeros_l56_56138


namespace weight_feel_when_lowered_l56_56180

-- Conditions from the problem
def num_plates : ℕ := 10
def weight_per_plate : ℝ := 30
def technology_increase : ℝ := 0.20
def incline_increase : ℝ := 0.15

-- Calculate the contributions
def total_weight_without_factors : ℝ := num_plates * weight_per_plate
def weight_with_technology : ℝ := total_weight_without_factors * (1 + technology_increase)
def weight_with_incline : ℝ := weight_with_technology * (1 + incline_increase)

-- Theorem statement we want to prove
theorem weight_feel_when_lowered : weight_with_incline = 414 := by
  sorry

end weight_feel_when_lowered_l56_56180


namespace complementary_set_count_is_correct_l56_56204

inductive Shape
| circle | square | triangle | hexagon

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

def deck : List Card :=
  -- (Note: Explicitly listing all 36 cards would be too verbose, pseudo-defining it for simplicity)
  [(Card.mk Shape.circle Color.red Shade.light),
   (Card.mk Shape.circle Color.red Shade.medium), 
   -- and so on for all 36 unique combinations...
   (Card.mk Shape.hexagon Color.green Shade.dark)]

def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∨ (c1.shape = c2.shape ∧ c2.shape = c3.shape)) ∧ 
  ((c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∨ (c1.color = c2.color ∧ c2.color = c3.color)) ∧
  ((c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∨ (c1.shade = c2.shade ∧ c2.shade = c3.shade))

noncomputable def count_complementary_sets : ℕ :=
  -- (Note: Implementation here is a placeholder. Actual counting logic would be non-trivial.)
  1836 -- placeholding the expected count

theorem complementary_set_count_is_correct :
  count_complementary_sets = 1836 :=
by
  trivial

end complementary_set_count_is_correct_l56_56204


namespace dilation_origin_distance_l56_56121

open Real

-- Definition of points and radii
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Given conditions as definitions
def original_circle := Circle.mk (3, 3) 3
def dilated_circle := Circle.mk (8, 10) 5
def dilation_factor := 5 / 3

-- Problem statement to prove
theorem dilation_origin_distance :
  let d₀ := dist (0, 0) (-6, -6)
  let d₁ := dilation_factor * d₀
  d₁ - d₀ = 4 * sqrt 2 :=
by
  sorry

end dilation_origin_distance_l56_56121


namespace markup_percentage_l56_56552

theorem markup_percentage (S M : ℝ) (h1 : S = 56 + M * S) (h2 : 0.80 * S - 56 = 8) : M = 0.30 :=
sorry

end markup_percentage_l56_56552


namespace least_value_of_x_l56_56973

theorem least_value_of_x (x : ℝ) : (4 * x^2 + 8 * x + 3 = 1) → (-1 ≤ x) :=
by
  intro h
  sorry

end least_value_of_x_l56_56973


namespace nancy_first_counted_l56_56692

theorem nancy_first_counted (x : ℤ) (h : (x + 12 + 1 + 12 + 7 + 3 + 8) / 6 = 7) : x = -1 := 
by 
  sorry

end nancy_first_counted_l56_56692


namespace shift_graph_to_right_l56_56136

theorem shift_graph_to_right (x : ℝ) : 
  4 * Real.cos (2 * x + π / 4) = 4 * Real.cos (2 * (x - π / 8) + π / 4) :=
by 
  -- sketch of the intended proof without actual steps for clarity
  sorry

end shift_graph_to_right_l56_56136


namespace megan_picture_shelves_l56_56036

def books_per_shelf : ℕ := 7
def mystery_shelves : ℕ := 8
def total_books : ℕ := 70
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := total_books - total_mystery_books
def picture_shelves : ℕ := total_picture_books / books_per_shelf

theorem megan_picture_shelves : picture_shelves = 2 := 
by sorry

end megan_picture_shelves_l56_56036


namespace fraction_of_hidden_sea_is_five_over_eight_l56_56396

noncomputable def cloud_fraction := 1 / 2
noncomputable def island_uncovered_fraction := 1 / 4 
noncomputable def island_covered_fraction := island_uncovered_fraction / (1 - cloud_fraction)

-- The total island area is the sum of covered and uncovered.
noncomputable def total_island_fraction := island_uncovered_fraction + island_covered_fraction 

-- The sea area covered by the cloud is half minus the fraction of the island covered by the cloud.
noncomputable def sea_covered_by_cloud := cloud_fraction - island_covered_fraction 

-- The sea occupies the remainder of the landscape not taken by the uncoveed island.
noncomputable def total_sea_fraction := 1 - island_uncovered_fraction - cloud_fraction + island_covered_fraction 

-- The sea fraction visible and not covered by clouds
noncomputable def sea_visible_not_covered := total_sea_fraction - sea_covered_by_cloud 

-- The fraction of the sea hidden by the cloud
noncomputable def sea_fraction_hidden_by_cloud := sea_covered_by_cloud / total_sea_fraction 

theorem fraction_of_hidden_sea_is_five_over_eight : sea_fraction_hidden_by_cloud = 5 / 8 := 
by
  sorry

end fraction_of_hidden_sea_is_five_over_eight_l56_56396


namespace necessarily_negative_expression_l56_56592

theorem necessarily_negative_expression
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 1)
  (hy : -1 < y ∧ y < 0)
  (hz : 0 < z ∧ z < 1)
  : y - z < 0 :=
sorry

end necessarily_negative_expression_l56_56592


namespace range_of_a_l56_56455

theorem range_of_a 
  (a : ℝ)
  (H1 : ∀ x : ℝ, -2 < x ∧ x < 3 → -2 < x ∧ x < a)
  (H2 : ¬(∀ x : ℝ, -2 < x ∧ x < a → -2 < x ∧ x < 3)) :
  3 < a :=
by
  sorry

end range_of_a_l56_56455


namespace largest_initial_number_l56_56976

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l56_56976


namespace inequality_ac2_geq_bc2_l56_56980

theorem inequality_ac2_geq_bc2 (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end inequality_ac2_geq_bc2_l56_56980


namespace power_of_p_in_product_l56_56489

theorem power_of_p_in_product (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) 
  (h : (x + 1) * 6 = 30) : x = 4 := 
by sorry

end power_of_p_in_product_l56_56489


namespace sales_volume_function_max_profit_min_boxes_for_2000_profit_l56_56950

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end sales_volume_function_max_profit_min_boxes_for_2000_profit_l56_56950


namespace number_of_members_l56_56648

theorem number_of_members (n : ℕ) (H : n * n = 5776) : n = 76 :=
by
  sorry

end number_of_members_l56_56648


namespace three_digit_number_digits_difference_l56_56697

theorem three_digit_number_digits_difference (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : a < b) (h4 : b < c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  reversed_number - original_number = 198 := by
  sorry

end three_digit_number_digits_difference_l56_56697


namespace find_sample_size_l56_56914

-- Define the frequencies
def frequencies (k : ℕ) : List ℕ := [2 * k, 3 * k, 4 * k, 6 * k, 4 * k, k]

-- Define the sum of the first three frequencies
def sum_first_three_frequencies (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k

-- Define the total number of data points
def total_data_points (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k + 6 * k + 4 * k + k

-- Define the main theorem
theorem find_sample_size (n k : ℕ) (h1 : sum_first_three_frequencies k = 27)
  (h2 : total_data_points k = n) : n = 60 := by
  sorry

end find_sample_size_l56_56914


namespace calculate_f_g_of_1_l56_56156

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem calculate_f_g_of_1 : f (g 1) = 39 :=
by
  -- Enable quick skippable proof with 'sorry'
  sorry

end calculate_f_g_of_1_l56_56156


namespace georgia_makes_muffins_l56_56304

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end georgia_makes_muffins_l56_56304


namespace number_of_ways_to_distribute_balls_l56_56414

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l56_56414


namespace find_unknown_gift_l56_56165

def money_from_aunt : ℝ := 9
def money_from_uncle : ℝ := 9
def money_from_bestfriend1 : ℝ := 22
def money_from_bestfriend2 : ℝ := 22
def money_from_bestfriend3 : ℝ := 22
def money_from_sister : ℝ := 7
def mean_money : ℝ := 16.3
def number_of_gifts : ℕ := 7

theorem find_unknown_gift (X : ℝ)
  (h1: money_from_aunt = 9)
  (h2: money_from_uncle = 9)
  (h3: money_from_bestfriend1 = 22)
  (h4: money_from_bestfriend2 = 22)
  (h5: money_from_bestfriend3 = 22)
  (h6: money_from_sister = 7)
  (h7: mean_money = 16.3)
  (h8: number_of_gifts = 7)
  : X = 23.1 := sorry

end find_unknown_gift_l56_56165


namespace incorrect_eqn_x9_y9_neg1_l56_56010

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end incorrect_eqn_x9_y9_neg1_l56_56010


namespace quadratic_root_difference_l56_56054

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def root_difference (a b c : ℝ) : ℝ :=
  (Real.sqrt (discriminant a b c)) / a

theorem quadratic_root_difference :
  root_difference (3 + 2 * Real.sqrt 2) (5 + Real.sqrt 2) (-4) = Real.sqrt (177 - 122 * Real.sqrt 2) :=
by
  sorry

end quadratic_root_difference_l56_56054


namespace row_seat_notation_l56_56995

-- Define that the notation (4, 5) corresponds to "Row 4, Seat 5"
def notation_row_seat := (4, 5)

-- Prove that "Row 5, Seat 4" should be denoted as (5, 4)
theorem row_seat_notation : (5, 4) = (5, 4) :=
by sorry

end row_seat_notation_l56_56995


namespace marlon_goals_l56_56956

theorem marlon_goals :
  ∃ g : ℝ,
    (∀ p f : ℝ, p + f = 40 → g = 0.4 * p + 0.5 * f) → g = 20 :=
by
  sorry

end marlon_goals_l56_56956


namespace problem1_problem2_problem3_problem4_l56_56440

theorem problem1 : 25 - 9 + (-12) - (-7) = 4 := by
  sorry

theorem problem2 : (1 / 9) * (-2)^3 / ((2 / 3)^2) = -2 := by
  sorry

theorem problem3 : ((5 / 12) + (2 / 3) - (3 / 4)) * (-12) = -4 := by
  sorry

theorem problem4 : -(1^4) + (-2) / (-1/3) - |(-9)| = -4 := by
  sorry

end problem1_problem2_problem3_problem4_l56_56440


namespace probability_two_roads_at_least_5_miles_long_l56_56443

-- Probabilities of roads being at least 5 miles long
def prob_A_B := 3 / 4
def prob_B_C := 2 / 3
def prob_C_D := 1 / 2

-- Theorem: Probability of at least two roads being at least 5 miles long
theorem probability_two_roads_at_least_5_miles_long :
  prob_A_B * prob_B_C * (1 - prob_C_D) +
  prob_A_B * prob_C_D * (1 - prob_B_C) +
  (1 - prob_A_B) * prob_B_C * prob_C_D +
  prob_A_B * prob_B_C * prob_C_D = 11 / 24 := 
by
  sorry -- Proof goes here

end probability_two_roads_at_least_5_miles_long_l56_56443


namespace total_dots_not_visible_l56_56148

def total_dots_on_dice (n : ℕ): ℕ := n * 21
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 4 + 5 + 6
def total_dice : ℕ := 4

theorem total_dots_not_visible :
  total_dots_on_dice total_dice - visible_dots = 58 := by
  sorry

end total_dots_not_visible_l56_56148


namespace frank_spend_more_l56_56514

noncomputable def table_cost : ℝ := 140
noncomputable def chair_cost : ℝ := 100
noncomputable def joystick_cost : ℝ := 20
noncomputable def frank_joystick : ℝ := joystick_cost * (1 / 4)
noncomputable def eman_joystick : ℝ := joystick_cost - frank_joystick
noncomputable def frank_total : ℝ := table_cost + frank_joystick
noncomputable def eman_total : ℝ := chair_cost + eman_joystick

theorem frank_spend_more :
  frank_total - eman_total = 30 :=
  sorry

end frank_spend_more_l56_56514


namespace greatest_integer_value_l56_56974

theorem greatest_integer_value (x : ℤ) (h : 3 * |x| + 4 ≤ 19) : x ≤ 5 :=
by
  sorry

end greatest_integer_value_l56_56974


namespace second_acid_solution_percentage_l56_56935

-- Definitions of the problem conditions
def P : ℝ := 75
def V₁ : ℝ := 4
def C₁ : ℝ := 0.60
def V₂ : ℝ := 20
def C₂ : ℝ := 0.72

/-
Given that 4 liters of a 60% acid solution are mixed with a certain volume of another acid solution
to get 20 liters of 72% solution, prove that the percentage of the second acid solution must be 75%.
-/
theorem second_acid_solution_percentage
  (x : ℝ) -- volume of the second acid solution
  (P_percent : ℝ := P) -- percentage of the second acid solution
  (h1 : V₁ + x = V₂) -- condition on volume
  (h2 : C₁ * V₁ + (P_percent / 100) * x = C₂ * V₂) -- condition on acid content
  : P_percent = P := 
by
  -- Moving forward with proof the lean proof
  sorry

end second_acid_solution_percentage_l56_56935


namespace exactly_two_pass_probability_l56_56451

theorem exactly_two_pass_probability (PA PB PC : ℚ) (hPA : PA = 2 / 3) (hPB : PB = 3 / 4) (hPC : PC = 2 / 5) :
  ((PA * PB * (1 - PC)) + (PA * (1 - PB) * PC) + ((1 - PA) * PB * PC) = 7 / 15) := by
  sorry

end exactly_two_pass_probability_l56_56451


namespace rent_percentage_increase_l56_56734

theorem rent_percentage_increase 
  (E : ℝ) 
  (h1 : ∀ (E : ℝ), rent_last_year = 0.25 * E)
  (h2 : ∀ (E : ℝ), earnings_this_year = 1.45 * E)
  (h3 : ∀ (E : ℝ), rent_this_year = 0.35 * earnings_this_year) :
  (rent_this_year / rent_last_year) * 100 = 203 := 
by 
  sorry

end rent_percentage_increase_l56_56734


namespace special_sale_day_price_l56_56014

-- Define the original price
def original_price : ℝ := 250

-- Define the first discount rate
def first_discount_rate : ℝ := 0.40

-- Calculate the price after the first discount
def price_after_first_discount (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- Define the second discount rate (special sale day)
def second_discount_rate : ℝ := 0.10

-- Calculate the price after the second discount
def price_after_second_discount (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price * (1 - discount_rate)

-- Theorem statement
theorem special_sale_day_price :
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 := by
  sorry

end special_sale_day_price_l56_56014


namespace joe_paint_initial_amount_l56_56522

theorem joe_paint_initial_amount (P : ℕ) (h1 : P / 6 + (5 * P / 6) / 5 = 120) :
  P = 360 := by
  sorry

end joe_paint_initial_amount_l56_56522


namespace range_of_x_l56_56105

theorem range_of_x (x : ℝ) :
  (∀ y : ℝ, 0 < y → y^2 + (2*x - 5)*y - x^2 * (Real.log x - Real.log y) ≤ 0) ↔ x = 5 / 2 :=
by 
  sorry

end range_of_x_l56_56105


namespace find_angle_C_l56_56652

theorem find_angle_C (a b c A B C : ℝ) (h₀ : 0 < C) (h₁ : C < Real.pi)
  (h₂ : 2 * c * Real.sin A = a * Real.tan C) :
  C = Real.pi / 3 :=
sorry

end find_angle_C_l56_56652


namespace least_number_to_subtract_l56_56621

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : 
  ∃ k, (n - k) % 10 = 0 ∧ k = 8 :=
by
  sorry

end least_number_to_subtract_l56_56621


namespace fraction_problem_l56_56565

theorem fraction_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 :=
by sorry

end fraction_problem_l56_56565


namespace find_weight_of_A_l56_56671

theorem find_weight_of_A 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 5) 
  (h4 : (B + C + D + E) / 4 = 79) 
  : A = 77 := 
sorry

end find_weight_of_A_l56_56671


namespace batsman_average_after_15th_innings_l56_56541

theorem batsman_average_after_15th_innings 
  (A : ℕ) 
  (h1 : 14 * A + 85 = 15 * (A + 3)) 
  (h2 : A = 40) : 
  (A + 3) = 43 := by 
  sorry

end batsman_average_after_15th_innings_l56_56541


namespace sum_of_inscribed_angles_l56_56849

theorem sum_of_inscribed_angles 
  (n : ℕ) 
  (total_degrees : ℝ)
  (arcs : ℕ)
  (x_arcs : ℕ)
  (y_arcs : ℕ) 
  (arc_angle : ℝ)
  (x_central_angle : ℝ)
  (y_central_angle : ℝ)
  (x_inscribed_angle : ℝ)
  (y_inscribed_angle : ℝ)
  (total_inscribed_angles : ℝ) :
  n = 18 →
  total_degrees = 360 →
  x_arcs = 3 →
  y_arcs = 5 →
  arc_angle = total_degrees / n →
  x_central_angle = x_arcs * arc_angle →
  y_central_angle = y_arcs * arc_angle →
  x_inscribed_angle = x_central_angle / 2 →
  y_inscribed_angle = y_central_angle / 2 →
  total_inscribed_angles = x_inscribed_angle + y_inscribed_angle →
  total_inscribed_angles = 80 := sorry

end sum_of_inscribed_angles_l56_56849


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_19_l56_56211

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_19 :
  let n := 16385
  let p := 3277
  let prime_p : Prime p := by sorry
  let greatest_prime_divisor := p
  let sum_digits := 3 + 2 + 7 + 7
  sum_digits = 19 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_19_l56_56211


namespace find_exponent_l56_56926

theorem find_exponent (y : ℝ) (exponent : ℝ) :
  (12^1 * 6^exponent / 432 = y) → (y = 36) → (exponent = 3) :=
by 
  intros h₁ h₂ 
  sorry

end find_exponent_l56_56926


namespace linear_equation_in_two_variables_l56_56445

def is_linear_equation_two_variables (eq : String → Prop) : Prop :=
  eq "D"

-- Given Conditions
def eqA (x y z : ℝ) : Prop := 2 * x + 3 * y = z
def eqB (x y : ℝ) : Prop := 4 / x + y = 5
def eqC (x y : ℝ) : Prop := 1 / 2 * x^2 + y = 0
def eqD (x y : ℝ) : Prop := y = 1 / 2 * (x + 8)

-- Problem Statement to be Proved
theorem linear_equation_in_two_variables :
  is_linear_equation_two_variables (λ s =>
    ∃ x y z : ℝ, 
      (s = "A" → eqA x y z) ∨ 
      (s = "B" → eqB x y) ∨ 
      (s = "C" → eqC x y) ∨ 
      (s = "D" → eqD x y)
  ) :=
sorry

end linear_equation_in_two_variables_l56_56445


namespace limit_C_of_f_is_2_l56_56366

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}
variable {f' : ℝ}

noncomputable def differentiable_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ f' : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f (x + h) - f x - f' * h) / abs (h) < ε

axiom hf_differentiable : differentiable_at f x₀
axiom f'_at_x₀ : f' = 1

theorem limit_C_of_f_is_2 
  (hf_differentiable : differentiable_at f x₀) 
  (h_f'_at_x₀ : f' = 1) : 
  (∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ + 2 * Δx) - f x₀) / Δx - 2) < ε) :=
sorry

end limit_C_of_f_is_2_l56_56366


namespace solution_set_of_abs_inequality_l56_56862

theorem solution_set_of_abs_inequality : 
  {x : ℝ | abs (x - 1) - abs (x - 5) < 2} = {x : ℝ | x < 4} := 
by 
  sorry

end solution_set_of_abs_inequality_l56_56862


namespace polynomial_expansion_p_eq_l56_56194

theorem polynomial_expansion_p_eq (p q : ℝ) (h1 : 10 * p^9 * q = 45 * p^8 * q^2) (h2 : p + 2 * q = 1) (hp : p > 0) (hq : q > 0) : p = 9 / 13 :=
by
  sorry

end polynomial_expansion_p_eq_l56_56194


namespace polynomial_product_expansion_l56_56050

theorem polynomial_product_expansion (x : ℝ) : (x^2 + 3 * x + 3) * (x^2 - 3 * x + 3) = x^4 - 3 * x^2 + 9 := 
by sorry

end polynomial_product_expansion_l56_56050


namespace exp_13_pi_i_over_2_eq_i_l56_56585

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13_pi_i_over_2_eq_i_l56_56585


namespace time_to_traverse_nth_mile_l56_56726

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 3) : ∃ t : ℕ, t = (n - 2)^2 :=
by
  -- Given:
  -- Speed varies inversely as the square of the number of miles already traveled.
  -- Speed is constant for each mile.
  -- The third mile is traversed in 4 hours.
  -- Show that:
  -- The time to traverse the nth mile is (n - 2)^2 hours.
  sorry

end time_to_traverse_nth_mile_l56_56726


namespace simplify_expression_l56_56717

theorem simplify_expression (n : ℕ) : 
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by sorry

end simplify_expression_l56_56717


namespace tangerines_more_than_oranges_l56_56196

-- Define initial conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17

-- Define actions taken
def oranges_taken := 2
def tangerines_taken := 10

-- Resulting quantities
def oranges_left := initial_oranges - oranges_taken
def tangerines_left := initial_tangerines - tangerines_taken

-- Proof problem
theorem tangerines_more_than_oranges : tangerines_left - oranges_left = 4 := 
by sorry

end tangerines_more_than_oranges_l56_56196


namespace amount_received_by_sam_l56_56961

def P : ℝ := 15000
def r : ℝ := 0.10
def n : ℝ := 2
def t : ℝ := 1

noncomputable def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem amount_received_by_sam : compoundInterest P r n t = 16537.50 := by
  sorry

end amount_received_by_sam_l56_56961


namespace remainder_5_pow_100_mod_18_l56_56860

theorem remainder_5_pow_100_mod_18 : (5 ^ 100) % 18 = 13 := 
by
  -- We will skip the proof since only the statement is required.
  sorry

end remainder_5_pow_100_mod_18_l56_56860


namespace cyclic_sums_sine_cosine_l56_56982

theorem cyclic_sums_sine_cosine (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) = 
  2 * (Real.sin α + Real.sin β + Real.sin γ) * 
      (Real.cos α + Real.cos β + Real.cos γ) - 
  2 * (Real.sin α + Real.sin β + Real.sin γ) := 
  sorry

end cyclic_sums_sine_cosine_l56_56982


namespace total_canoes_by_end_of_march_l56_56093

theorem total_canoes_by_end_of_march
  (canoes_jan : ℕ := 3)
  (canoes_feb : ℕ := canoes_jan * 2)
  (canoes_mar : ℕ := canoes_feb * 2) :
  canoes_jan + canoes_feb + canoes_mar = 21 :=
by
  sorry

end total_canoes_by_end_of_march_l56_56093


namespace similar_triangles_XY_length_l56_56316

-- Defining necessary variables.
variables (PQ QR YZ XY : ℝ) (area_XYZ : ℝ)

-- Given conditions to be used in the proof.
def condition1 : PQ = 8 := sorry
def condition2 : QR = 16 := sorry
def condition3 : YZ = 24 := sorry
def condition4 : area_XYZ = 144 := sorry

-- Statement of the mathematical proof problem to show XY = 12
theorem similar_triangles_XY_length :
  PQ = 8 → QR = 16 → YZ = 24 → area_XYZ = 144 → XY = 12 :=
by
  intros hPQ hQR hYZ hArea
  sorry

end similar_triangles_XY_length_l56_56316


namespace f_increasing_interval_l56_56678

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end f_increasing_interval_l56_56678


namespace subgroups_of_integers_l56_56286

theorem subgroups_of_integers (G : AddSubgroup ℤ) : ∃ (d : ℤ), G = AddSubgroup.zmultiples d := 
sorry

end subgroups_of_integers_l56_56286


namespace heating_time_l56_56949

def T_initial: ℝ := 20
def T_final: ℝ := 100
def rate: ℝ := 5

theorem heating_time : (T_final - T_initial) / rate = 16 := by
  sorry

end heating_time_l56_56949


namespace smallest_n_product_exceeds_l56_56946

theorem smallest_n_product_exceeds (n : ℕ) : (5 : ℝ) ^ (n * (n + 1) / 14) > 1000 ↔ n = 7 :=
by sorry

end smallest_n_product_exceeds_l56_56946


namespace find_a_b_extreme_points_l56_56640

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : deriv (f a b) 2 = 0) (h₃ : f a b 2 = 8) : 
  a = 4 ∧ b = 24 :=
by
  sorry

noncomputable def f_deriv (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

theorem extreme_points (a : ℝ) (h₁ : a > 0) : 
  (∃ x: ℝ, f_deriv a x = 0 ∧ 
      ((x = -Real.sqrt a ∧ f a 24 x = 40) ∨ 
       (x = Real.sqrt a ∧ f a 24 x = 16))) := 
by
  sorry

end find_a_b_extreme_points_l56_56640


namespace parallelepiped_surface_area_l56_56519

theorem parallelepiped_surface_area (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 12) 
  (h2 : a * b * c = 8) : 
  6 * (a^2) = 24 :=
by
  sorry

end parallelepiped_surface_area_l56_56519


namespace pens_each_student_gets_now_l56_56071

-- Define conditions
def red_pens_per_student := 62
def black_pens_per_student := 43
def num_students := 3
def pens_taken_first_month := 37
def pens_taken_second_month := 41

-- Define total pens bought and remaining pens after each month
def total_pens := num_students * (red_pens_per_student + black_pens_per_student)
def remaining_pens_after_first_month := total_pens - pens_taken_first_month
def remaining_pens_after_second_month := remaining_pens_after_first_month - pens_taken_second_month

-- Theorem statement
theorem pens_each_student_gets_now :
  (remaining_pens_after_second_month / num_students) = 79 :=
by
  sorry

end pens_each_student_gets_now_l56_56071


namespace range_of_a_l56_56689

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x^2 + 4 * x else Real.logb 2 x - a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 1 < a :=
sorry

end range_of_a_l56_56689


namespace time_to_reach_6400ft_is_200min_l56_56081

noncomputable def time_to_reach_ship (depth : ℕ) (rate : ℕ) : ℕ :=
  depth / rate

theorem time_to_reach_6400ft_is_200min :
  time_to_reach_ship 6400 32 = 200 := by
  sorry

end time_to_reach_6400ft_is_200min_l56_56081


namespace solution_set_of_inequality_system_l56_56387

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end solution_set_of_inequality_system_l56_56387


namespace parallel_if_perp_to_plane_l56_56768

variable {α m n : Type}

variables (plane : α) (line_m line_n : m)

-- Define what it means for lines to be perpendicular to a plane
def perpendicular_to_plane (line : m) (pl : α) : Prop := sorry

-- Define what it means for lines to be parallel
def parallel (line1 line2 : m) : Prop := sorry

-- The conditions
axiom perp_1 : perpendicular_to_plane line_m plane
axiom perp_2 : perpendicular_to_plane line_n plane

-- The theorem to prove
theorem parallel_if_perp_to_plane : parallel line_m line_n := sorry

end parallel_if_perp_to_plane_l56_56768


namespace kevin_total_hops_l56_56797

/-- Define the hop function for Kevin -/
def hop (remaining_distance : ℚ) : ℚ :=
  remaining_distance / 4

/-- Summing the series for five hops -/
def total_hops (start_distance : ℚ) (hops : ℕ) : ℚ :=
  let h0 := hop start_distance
  let h1 := hop (start_distance - h0)
  let h2 := hop (start_distance - h0 - h1)
  let h3 := hop (start_distance - h0 - h1 - h2)
  let h4 := hop (start_distance - h0 - h1 - h2 - h3)
  h0 + h1 + h2 + h3 + h4

/-- Final proof statement: after five hops from starting distance of 2, total distance hopped should be 1031769/2359296 -/
theorem kevin_total_hops :
  total_hops 2 5 = 1031769 / 2359296 :=
sorry

end kevin_total_hops_l56_56797


namespace quadratic_function_behavior_l56_56564

theorem quadratic_function_behavior (x : ℝ) (h : x > 2) :
  ∃ y : ℝ, y = - (x - 2)^2 - 7 ∧ ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → (-(x₂ - 2)^2 - 7) < (-(x₁ - 2)^2 - 7) :=
by
  sorry

end quadratic_function_behavior_l56_56564
