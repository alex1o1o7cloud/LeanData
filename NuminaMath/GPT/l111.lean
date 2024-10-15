import Mathlib

namespace NUMINAMATH_GPT_number_of_middle_managers_selected_l111_11115

-- Definitions based on conditions
def total_employees := 1000
def senior_managers := 50
def middle_managers := 150
def general_staff := 800
def survey_size := 200

-- Proposition to state the question and correct answer formally
theorem number_of_middle_managers_selected:
  200 * (150 / 1000) = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_middle_managers_selected_l111_11115


namespace NUMINAMATH_GPT_no_two_digit_multiples_of_3_5_7_l111_11165

theorem no_two_digit_multiples_of_3_5_7 : ∀ n : ℕ, 10 ≤ n ∧ n < 100 → ¬ (3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := 
by
  intro n
  intro h
  intro h_div
  sorry

end NUMINAMATH_GPT_no_two_digit_multiples_of_3_5_7_l111_11165


namespace NUMINAMATH_GPT_find_r_l111_11171

theorem find_r (f g : ℝ → ℝ) (monic_f : ∀x, f x = (x - r - 2) * (x - r - 8) * (x - a))
  (monic_g : ∀x, g x = (x - r - 4) * (x - r - 10) * (x - b)) (h : ∀ x, f x - g x = r):
  r = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l111_11171


namespace NUMINAMATH_GPT_students_play_football_l111_11178

theorem students_play_football (total_students : ℕ) (C : ℕ) (B : ℕ) (neither : ℕ) (F : ℕ)
  (h1 : total_students = 460)
  (h2 : C = 175)
  (h3 : B = 90)
  (h4 : neither = 50)
  (h5 : total_students = neither + F + C - B) : 
  F = 325 :=
by 
  sorry

end NUMINAMATH_GPT_students_play_football_l111_11178


namespace NUMINAMATH_GPT_find_S16_l111_11196

theorem find_S16 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 12 = -8)
  (h2 : S 9 = -9)
  (h_sum : ∀ n, S n = (n * (a 1 + a n) / 2)) :
  S 16 = -72 := 
by
  sorry

end NUMINAMATH_GPT_find_S16_l111_11196


namespace NUMINAMATH_GPT_seahawks_final_score_l111_11172

def num_touchdowns : ℕ := 4
def num_field_goals : ℕ := 3
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3

theorem seahawks_final_score : (num_touchdowns * points_per_touchdown) + (num_field_goals * points_per_fieldgoal) = 37 := by
  sorry

end NUMINAMATH_GPT_seahawks_final_score_l111_11172


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_l111_11109

theorem largest_of_seven_consecutive (n : ℕ) 
  (h1: n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) :
  n + 6 = 433 :=
by 
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_l111_11109


namespace NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l111_11184

noncomputable def arithmetic_sequence_n : ℕ :=
  sorry

theorem find_n_in_arithmetic_sequence (a : ℕ → ℕ) (d n : ℕ) :
  (a 3) + (a 4) = 10 → (a (n-3) + a (n-2)) = 30 → n * (a 1 + a n) / 2 = 100 → n = 10 :=
  sorry

end NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l111_11184


namespace NUMINAMATH_GPT_mod_inverse_13_1728_l111_11110

theorem mod_inverse_13_1728 :
  (13 * 133) % 1728 = 1 := by
  sorry

end NUMINAMATH_GPT_mod_inverse_13_1728_l111_11110


namespace NUMINAMATH_GPT_proof_problem_l111_11146

variable (A B C : ℕ)

-- Defining the conditions
def condition1 : Prop := A + B + C = 700
def condition2 : Prop := B + C = 600
def condition3 : Prop := C = 200

-- Stating the proof problem
theorem proof_problem (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 C) : A + C = 300 :=
sorry

end NUMINAMATH_GPT_proof_problem_l111_11146


namespace NUMINAMATH_GPT_top_cell_pos_cases_l111_11183

-- Define the rule for the cell sign propagation
def cell_sign (a b : ℤ) : ℤ := 
  if a = b then 1 else -1

-- The pyramid height
def pyramid_height : ℕ := 5

-- Define the final condition for the top cell in the pyramid to be "+"
def top_cell_sign (a b c d e : ℤ) : ℤ :=
  a * b * c * d * e

-- Define the proof statement
theorem top_cell_pos_cases :
  (∃ a b c d e : ℤ,
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    top_cell_sign a b c d e = 1) ∧
  (∃ n, n = 11) :=
by
  sorry

end NUMINAMATH_GPT_top_cell_pos_cases_l111_11183


namespace NUMINAMATH_GPT_square_inequality_l111_11163

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end NUMINAMATH_GPT_square_inequality_l111_11163


namespace NUMINAMATH_GPT_passengers_from_other_continents_l111_11192

theorem passengers_from_other_continents :
  (∀ (n NA EU AF AS : ℕ),
     NA = n / 4 →
     EU = n / 8 →
     AF = n / 12 →
     AS = n / 6 →
     96 = n →
     n - (NA + EU + AF + AS) = 36) :=
by
  sorry

end NUMINAMATH_GPT_passengers_from_other_continents_l111_11192


namespace NUMINAMATH_GPT_ensure_nonempty_intersection_l111_11153

def M (x : ℝ) : Prop := x ≤ 1
def N (x : ℝ) (p : ℝ) : Prop := x > p

theorem ensure_nonempty_intersection (p : ℝ) : (∃ x : ℝ, M x ∧ N x p) ↔ p < 1 :=
by
  sorry

end NUMINAMATH_GPT_ensure_nonempty_intersection_l111_11153


namespace NUMINAMATH_GPT_find_7c_plus_7d_l111_11195

noncomputable def f (c d x : ℝ) : ℝ := c * x + d
noncomputable def h (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 1

theorem find_7c_plus_7d (c d : ℝ) (h_def : ∀ x, h x = f_inv x - 5) (f_def : ∀ x, f c d x = c * x + d) (f_inv_def : ∀ x, f_inv x = 7 * x - 1) : 7 * c + 7 * d = 2 := by
  sorry

end NUMINAMATH_GPT_find_7c_plus_7d_l111_11195


namespace NUMINAMATH_GPT_factorize_expression_l111_11133

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l111_11133


namespace NUMINAMATH_GPT_frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l111_11168

-- Part (a): Prove the number of ways to reach vertex C from A in n jumps when n is even
theorem frog_reaches_C_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = (4^n/2 - 1) / 3 := by sorry

-- Part (b): Prove the number of ways to reach vertex C from A in n jumps without jumping to D when n is even
theorem frog_reaches_C_no_D_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = 3^(n/2 - 1) := by sorry

-- Part (c): Prove the probability the frog is alive after n jumps with a mine at D
theorem frog_alive_probability (n : ℕ) (k : ℕ) (h_n : n = 2*k - 1 ∨ n = 2*k) : 
    ∃ p : ℝ, p = (3/4)^(k-1) := by sorry

-- Part (d): Prove the average lifespan of the frog in the presence of a mine at D
theorem frog_average_lifespan : 
    ∃ t : ℝ, t = 9 := by sorry

end NUMINAMATH_GPT_frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l111_11168


namespace NUMINAMATH_GPT_lemonade_glasses_from_fruit_l111_11175

noncomputable def lemons_per_glass : ℕ := 2
noncomputable def oranges_per_glass : ℕ := 1
noncomputable def total_lemons : ℕ := 18
noncomputable def total_oranges : ℕ := 10
noncomputable def grapefruits : ℕ := 6
noncomputable def lemons_per_grapefruit : ℕ := 2
noncomputable def oranges_per_grapefruit : ℕ := 1

theorem lemonade_glasses_from_fruit :
  (total_lemons / lemons_per_glass) = 9 →
  (total_oranges / oranges_per_glass) = 10 →
  min (total_lemons / lemons_per_glass) (total_oranges / oranges_per_glass) = 9 →
  (grapefruits * lemons_per_grapefruit = 12) →
  (grapefruits * oranges_per_grapefruit = 6) →
  (9 + grapefruits) = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_lemonade_glasses_from_fruit_l111_11175


namespace NUMINAMATH_GPT_exponent_m_n_add_l111_11125

variable (a : ℝ) (m n : ℕ)

theorem exponent_m_n_add (h1 : a ^ m = 2) (h2 : a ^ n = 3) : a ^ (m + n) = 6 := by
  sorry

end NUMINAMATH_GPT_exponent_m_n_add_l111_11125


namespace NUMINAMATH_GPT_unit_prices_max_helmets_A_l111_11152

open Nat Real

-- Given conditions
variables (x y : ℝ)
variables (m : ℕ)

def wholesale_price_A := 30
def wholesale_price_B := 20
def price_difference := 15
def revenue_A := 450
def revenue_B := 600
def total_helmets := 100
def budget := 2350

-- Part 1: Prove the unit prices of helmets A and B
theorem unit_prices :
  ∃ (price_A price_B : ℝ), 
    (price_A = price_B + price_difference) ∧ 
    (revenue_B / price_B = 2 * revenue_A / price_A) ∧
    (price_B = 30) ∧
    (price_A = 45) :=
by
  sorry

-- Part 2: Prove the maximum number of helmets of type A that can be purchased
theorem max_helmets_A :
  ∃ (m : ℕ), 
    (30 * m + 20 * (total_helmets - m) ≤ budget) ∧
    (m ≤ 35) :=
by
  sorry

end NUMINAMATH_GPT_unit_prices_max_helmets_A_l111_11152


namespace NUMINAMATH_GPT_midpoint_distance_trapezoid_l111_11129

theorem midpoint_distance_trapezoid (x : ℝ) : 
  let AD := x
  let BC := 5
  PQ = (|x - 5| / 2) :=
sorry

end NUMINAMATH_GPT_midpoint_distance_trapezoid_l111_11129


namespace NUMINAMATH_GPT_range_of_a_l111_11197

theorem range_of_a (a : ℝ) : (0 < a ∧ a ≤ Real.exp 1) ↔ ∀ x : ℝ, 0 < x → a * Real.log (a * x) ≤ Real.exp x := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l111_11197


namespace NUMINAMATH_GPT_calculate_expression_l111_11141

theorem calculate_expression : -1^2021 + 1^2022 = 0 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l111_11141


namespace NUMINAMATH_GPT_find_angle_l111_11144

theorem find_angle (x : ℝ) (h1 : 90 - x = (1/2) * (180 - x)) : x = 90 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l111_11144


namespace NUMINAMATH_GPT_greatest_drop_in_price_l111_11185

theorem greatest_drop_in_price (jan feb mar apr may jun : ℝ)
  (h_jan : jan = -0.50)
  (h_feb : feb = 2.00)
  (h_mar : mar = -2.50)
  (h_apr : apr = 3.00)
  (h_may : may = -0.50)
  (h_jun : jun = -2.00) :
  mar = -2.50 ∧ (mar ≤ jan ∧ mar ≤ may ∧ mar ≤ jun) :=
by
  sorry

end NUMINAMATH_GPT_greatest_drop_in_price_l111_11185


namespace NUMINAMATH_GPT_rectangle_perimeter_l111_11118

theorem rectangle_perimeter (a b : ℤ) (h1 : a ≠ b) (h2 : 2 * (2 * a + 2 * b) - a * b = 12) : 2 * (a + b) = 26 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_l111_11118


namespace NUMINAMATH_GPT_unique_two_digit_integer_l111_11102

theorem unique_two_digit_integer (t : ℕ) (h : 11 * t % 100 = 36) (ht : 10 ≤ t ∧ t ≤ 99) : t = 76 :=
by
  sorry

end NUMINAMATH_GPT_unique_two_digit_integer_l111_11102


namespace NUMINAMATH_GPT_length_of_AB_l111_11149

noncomputable def ratio3to5 (AP PB : ℝ) : Prop := AP / PB = 3 / 5
noncomputable def ratio4to5 (AQ QB : ℝ) : Prop := AQ / QB = 4 / 5
noncomputable def pointDistances (P Q : ℝ) : Prop := P - Q = 3

theorem length_of_AB (A B P Q : ℝ) (P_on_AB : P > A ∧ P < B) (Q_on_AB : Q > A ∧ Q < B)
  (middle_side : P < (A + B) / 2 ∧ Q < (A + B) / 2)
  (h1 : ratio3to5 (P - A) (B - P))
  (h2 : ratio4to5 (Q - A) (B - Q))
  (h3 : pointDistances P Q) : B - A = 43.2 := 
sorry

end NUMINAMATH_GPT_length_of_AB_l111_11149


namespace NUMINAMATH_GPT_misha_current_dollars_l111_11176

variable (x : ℕ)

def misha_needs_more : ℕ := 13
def total_amount : ℕ := 47

theorem misha_current_dollars : x = total_amount - misha_needs_more → x = 34 :=
by
  sorry

end NUMINAMATH_GPT_misha_current_dollars_l111_11176


namespace NUMINAMATH_GPT_find_m_eq_5_l111_11159

-- Definitions for the problem conditions
def f (x m : ℝ) := 2 * x + m

theorem find_m_eq_5 (m : ℝ) (a b : ℝ) :
  (a = f 0 m) ∧ (b = f m m) ∧ ((b - a) = (m - 0 + 5)) → m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_eq_5_l111_11159


namespace NUMINAMATH_GPT_cylinder_volume_l111_11116

theorem cylinder_volume (V_sphere : ℝ) (V_cylinder : ℝ) (R H : ℝ) 
  (h1 : V_sphere = 4 * π / 3) 
  (h2 : (4 * π * R ^ 3) / 3 = V_sphere) 
  (h3 : H = 2 * R) 
  (h4 : R = 1) : V_cylinder = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l111_11116


namespace NUMINAMATH_GPT_scramble_time_is_correct_l111_11187

-- Define the conditions
def sausages : ℕ := 3
def fry_time_per_sausage : ℕ := 5
def eggs : ℕ := 6
def total_time : ℕ := 39

-- Define the time to scramble each egg
def scramble_time_per_egg : ℕ :=
  let frying_time := sausages * fry_time_per_sausage
  let scrambling_time := total_time - frying_time
  scrambling_time / eggs

-- The theorem stating the main question and desired answer
theorem scramble_time_is_correct : scramble_time_per_egg = 4 := by
  sorry

end NUMINAMATH_GPT_scramble_time_is_correct_l111_11187


namespace NUMINAMATH_GPT_total_ways_to_choose_president_and_vice_president_of_same_gender_l111_11198

theorem total_ways_to_choose_president_and_vice_president_of_same_gender :
  let boys := 12
  let girls := 12
  (boys * (boys - 1) + girls * (girls - 1)) = 264 :=
by
  sorry

end NUMINAMATH_GPT_total_ways_to_choose_president_and_vice_president_of_same_gender_l111_11198


namespace NUMINAMATH_GPT_number_of_permutations_l111_11162

theorem number_of_permutations (readers : Fin 8 → Type) : ∃! (n : ℕ), n = 40320 :=
by
  sorry

end NUMINAMATH_GPT_number_of_permutations_l111_11162


namespace NUMINAMATH_GPT_elsa_final_marbles_l111_11130

def start_marbles : ℕ := 40
def lost_breakfast : ℕ := 3
def given_susie : ℕ := 5
def new_marbles : ℕ := 12
def returned_marbles : ℕ := 2 * given_susie

def final_marbles : ℕ :=
  start_marbles - lost_breakfast - given_susie + new_marbles + returned_marbles

theorem elsa_final_marbles : final_marbles = 54 := by
  sorry

end NUMINAMATH_GPT_elsa_final_marbles_l111_11130


namespace NUMINAMATH_GPT_moles_of_NaOH_combined_l111_11123

-- Define the reaction conditions
variable (moles_NH4NO3 : ℕ) (moles_NaNO3 : ℕ)

-- Define a proof problem that asserts the number of moles of NaOH combined
theorem moles_of_NaOH_combined
  (h1 : moles_NH4NO3 = 3)  -- 3 moles of NH4NO3 are combined
  (h2 : moles_NaNO3 = 3)  -- 3 moles of NaNO3 are formed
  : ∃ moles_NaOH : ℕ, moles_NaOH = 3 :=
by {
  -- Proof skeleton to be filled
  sorry
}

end NUMINAMATH_GPT_moles_of_NaOH_combined_l111_11123


namespace NUMINAMATH_GPT_triangle_circumradius_l111_11167

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 10) : 
  ∃ r : ℝ, r = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_circumradius_l111_11167


namespace NUMINAMATH_GPT_roots_are_distinct_l111_11160

theorem roots_are_distinct (a x1 x2 : ℝ) (h : x1 ≠ x2) :
  (∀ x, x^2 - a*x - 2 = 0 → x = x1 ∨ x = x2) → x1 ≠ x2 := sorry

end NUMINAMATH_GPT_roots_are_distinct_l111_11160


namespace NUMINAMATH_GPT_identity_proof_l111_11169

theorem identity_proof (A B C A1 B1 C1 : ℝ) :
  (A^2 + B^2 + C^2) * (A1^2 + B1^2 + C1^2) - (A * A1 + B * B1 + C * C1)^2 =
    (A * B1 + A1 * B)^2 + (A * C1 + A1 * C)^2 + (B * C1 + B1 * C)^2 :=
by
  sorry

end NUMINAMATH_GPT_identity_proof_l111_11169


namespace NUMINAMATH_GPT_expression_of_f_f_increasing_on_interval_inequality_solution_l111_11140

noncomputable def f (x : ℝ) : ℝ := (x / (1 + x^2))

-- 1. Proving f(x) is the given function
theorem expression_of_f (x : ℝ) (h₁ : f x = (a*x + b) / (1 + x^2)) (h₂ : (∀ x, f (-x) = -f x)) (h₃ : f (1/2) = 2/5) :
  f x = x / (1 + x^2) :=
sorry

-- 2. Prove f(x) is increasing on (-1,1)
theorem f_increasing_on_interval {x₁ x₂ : ℝ} (h₁ : -1 < x₁ ∧ x₁ < 1) (h₂ : -1 < x₂ ∧ x₂ < 1) (h₃ : x₁ < x₂) :
  f x₁ < f x₂ :=
sorry

-- 3. Solve the inequality f(t-1) + f(t) < 0 on (0, 1/2)
theorem inequality_solution (t : ℝ) (h₁ : 0 < t) (h₂ : t < 1/2) :
  f (t - 1) + f t < 0 :=
sorry

end NUMINAMATH_GPT_expression_of_f_f_increasing_on_interval_inequality_solution_l111_11140


namespace NUMINAMATH_GPT_tan_theta_eq_1_over_3_l111_11158

noncomputable def unit_circle_point (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := Real.sin θ
  (x^2 + y^2 = 1) ∧ (θ = Real.arccos ((4*x + 3*y) / 5))

theorem tan_theta_eq_1_over_3 (θ : ℝ) (h : unit_circle_point θ) : Real.tan θ = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_theta_eq_1_over_3_l111_11158


namespace NUMINAMATH_GPT_smallest_three_digit_number_l111_11155

theorem smallest_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧
  (x % 2 = 0) ∧
  ((x + 1) % 3 = 0) ∧
  ((x + 2) % 4 = 0) ∧
  ((x + 3) % 5 = 0) ∧
  ((x + 4) % 6 = 0) ∧
  x = 122 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_number_l111_11155


namespace NUMINAMATH_GPT_total_divisions_is_48_l111_11126

-- Definitions based on the conditions
def initial_cells := 1
def final_cells := 1993
def cells_added_division_42 := 41
def cells_added_division_44 := 43

-- The main statement we want to prove
theorem total_divisions_is_48 (a b : ℕ) 
  (h1 : cells_added_division_42 = 41)
  (h2 : cells_added_division_44 = 43)
  (h3 : cells_added_division_42 * a + cells_added_division_44 * b = final_cells - initial_cells) :
  a + b = 48 := 
sorry

end NUMINAMATH_GPT_total_divisions_is_48_l111_11126


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l111_11120

-- Definitions for quadrants
def in_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360
def in_third_quadrant (β : ℝ) : Prop := 180 < β ∧ β < 270

theorem angle_in_third_quadrant (α : ℝ) (h : in_fourth_quadrant α) : in_third_quadrant (180 - α) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l111_11120


namespace NUMINAMATH_GPT_comparison_of_exponential_and_power_l111_11112

theorem comparison_of_exponential_and_power :
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  a > b :=
by
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  sorry

end NUMINAMATH_GPT_comparison_of_exponential_and_power_l111_11112


namespace NUMINAMATH_GPT_find_number_l111_11154

theorem find_number (x: ℝ) (h: (6 * x) / 2 - 5 = 25) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l111_11154


namespace NUMINAMATH_GPT_total_trucks_l111_11108

-- Define the number of trucks Namjoon has
def trucks_namjoon : ℕ := 3

-- Define the number of trucks Taehyung has
def trucks_taehyung : ℕ := 2

-- Prove that together, Namjoon and Taehyung have 5 trucks
theorem total_trucks : trucks_namjoon + trucks_taehyung = 5 := by 
  sorry

end NUMINAMATH_GPT_total_trucks_l111_11108


namespace NUMINAMATH_GPT_distance_between_A_and_B_l111_11174

-- Definitions according to the problem's conditions
def speed_train_A : ℕ := 50
def speed_train_B : ℕ := 60
def distance_difference : ℕ := 100

-- The main theorem statement to prove
theorem distance_between_A_and_B
  (x : ℕ) -- x is the distance traveled by the first train
  (distance_train_A := x)
  (distance_train_B := x + distance_difference)
  (total_distance := distance_train_A + distance_train_B)
  (meet_condition : distance_train_A / speed_train_A = distance_train_B / speed_train_B) :
  total_distance = 1100 := 
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l111_11174


namespace NUMINAMATH_GPT_class_A_scores_more_uniform_l111_11199

-- Define the variances of the test scores for classes A and B
def variance_A := 13.2
def variance_B := 26.26

-- Theorem: Prove that the scores of the 10 students from class A are more uniform than those from class B
theorem class_A_scores_more_uniform :
  variance_A < variance_B :=
  by
    -- Assume the given variances and state the comparison
    have h : 13.2 < 26.26 := by sorry
    exact h

end NUMINAMATH_GPT_class_A_scores_more_uniform_l111_11199


namespace NUMINAMATH_GPT_diophantine_eq_unique_solutions_l111_11137

theorem diophantine_eq_unique_solutions (x y : ℕ) (hx_positive : x > 0) (hy_positive : y > 0) :
  x^y = y^x + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_diophantine_eq_unique_solutions_l111_11137


namespace NUMINAMATH_GPT_monotonically_decreasing_implies_a_geq_3_l111_11122

noncomputable def f (x a : ℝ): ℝ := x^3 - a * x - 1

theorem monotonically_decreasing_implies_a_geq_3 : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x a ≤ f x 3) →
  a ≥ 3 := 
sorry

end NUMINAMATH_GPT_monotonically_decreasing_implies_a_geq_3_l111_11122


namespace NUMINAMATH_GPT_sector_area_maximized_l111_11117

noncomputable def maximize_sector_area (r θ : ℝ) : Prop :=
  2 * r + θ * r = 20 ∧
  (r > 0 ∧ θ > 0) ∧
  ∀ (r' θ' : ℝ), (2 * r' + θ' * r' = 20 ∧ r' > 0 ∧ θ' > 0) → (1/2 * θ' * r'^2 ≤ 1/2 * θ * r^2)

theorem sector_area_maximized : maximize_sector_area 5 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_maximized_l111_11117


namespace NUMINAMATH_GPT_initial_calculated_average_was_23_l111_11164

theorem initial_calculated_average_was_23 (S : ℕ) (incorrect_sum : ℕ) (n : ℕ)
  (correct_sum : ℕ) (correct_average : ℕ) (wrong_read : ℕ) (correct_read : ℕ) :
  (n = 10) →
  (wrong_read = 26) →
  (correct_read = 36) →
  (correct_average = 24) →
  (correct_sum = n * correct_average) →
  (incorrect_sum = correct_sum - correct_read + wrong_read) →
  S = incorrect_sum →
  S / n = 23 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_calculated_average_was_23_l111_11164


namespace NUMINAMATH_GPT_number_of_possible_A2_eq_one_l111_11113

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end NUMINAMATH_GPT_number_of_possible_A2_eq_one_l111_11113


namespace NUMINAMATH_GPT_lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l111_11189

def lassis_per_three_mangoes := 15
def smoothies_per_mango := 1
def bananas_per_smoothie := 2

-- proving the number of lassis Caroline can make with eighteen mangoes
theorem lassis_with_eighteen_mangoes :
  (18 / 3) * lassis_per_three_mangoes = 90 :=
by 
  sorry

-- proving the number of smoothies Caroline can make with eighteen mangoes and thirty-six bananas
theorem smoothies_with_eighteen_mangoes_and_thirtysix_bananas :
  min (18 / smoothies_per_mango) (36 / bananas_per_smoothie) = 18 :=
by 
  sorry

end NUMINAMATH_GPT_lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l111_11189


namespace NUMINAMATH_GPT_sum_of_three_squares_l111_11101

theorem sum_of_three_squares (s t : ℤ) (h1 : 3 * s + 2 * t = 27)
                             (h2 : 2 * s + 3 * t = 23) (h3 : s + 2 * t = 13) :
  3 * s = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_three_squares_l111_11101


namespace NUMINAMATH_GPT_milk_replacement_problem_l111_11127

theorem milk_replacement_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 90)
  (h3 : (90 - x) - ((90 - x) * x / 90) = 72.9) : x = 9 :=
sorry

end NUMINAMATH_GPT_milk_replacement_problem_l111_11127


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l111_11193

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) (h : π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π) :
  ∃ m : ℤ, -π - 2 * m * π < π / 2 - α ∧ (π / 2 - α) < -π / 2 - 2 * m * π :=
by
  -- Lean users note: The proof isn't required here, just setting up the statement as instructed.
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l111_11193


namespace NUMINAMATH_GPT_restore_arithmetic_operations_l111_11166

/--
Given the placeholders \(A, B, C, D, E\) for operations in the equations:
1. \(4 A 2 = 2\)
2. \(8 = 4 C 2\)
3. \(2 D 3 = 5\)
4. \(4 = 5 E 1\)

Prove that:
(a) \(A = ÷\)
(b) \(B = =\)
(c) \(C = ×\)
(d) \(D = +\)
(e) \(E = -\)
-/
theorem restore_arithmetic_operations {A B C D E : String} (h1 : B = "=") 
    (h2 : "4" ++ A  ++ "2" ++ B ++ "2" = "4" ++ "÷" ++ "2" ++ "=" ++ "2")
    (h3 : "8" ++ "=" ++ "4" ++ C ++ "2" = "8" ++ "=" ++ "4" ++ "×" ++ "2")
    (h4 : "2" ++ D ++ "3" ++ "=" ++ "5" = "2" ++ "+" ++ "3" ++ "=" ++ "5")
    (h5 : "4" ++ "=" ++ "5" ++ E ++ "1" = "4" ++ "=" ++ "5" ++ "-" ++ "1") :
  (A = "÷") ∧ (B = "=") ∧ (C = "×") ∧ (D = "+") ∧ (E = "-") := by
    sorry

end NUMINAMATH_GPT_restore_arithmetic_operations_l111_11166


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l111_11128

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 :=
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l111_11128


namespace NUMINAMATH_GPT_single_shot_percentage_decrease_l111_11142

theorem single_shot_percentage_decrease
  (initial_salary : ℝ)
  (final_salary : ℝ := initial_salary * 0.95 * 0.90 * 0.85) :
  ((1 - final_salary / initial_salary) * 100) = 27.325 := by
  sorry

end NUMINAMATH_GPT_single_shot_percentage_decrease_l111_11142


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_triangle_perimeter_l111_11106

theorem smallest_whole_number_larger_than_triangle_perimeter :
  (∀ s : ℝ, 16 < s ∧ s < 30 → ∃ n : ℕ, n = 60) :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_triangle_perimeter_l111_11106


namespace NUMINAMATH_GPT_discount_percentage_l111_11124

theorem discount_percentage (cost_price marked_price : ℝ) (profit_percentage : ℝ) 
  (h_cost_price : cost_price = 47.50)
  (h_marked_price : marked_price = 65)
  (h_profit_percentage : profit_percentage = 0.30) :
  ((marked_price - (cost_price + (profit_percentage * cost_price))) / marked_price) * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l111_11124


namespace NUMINAMATH_GPT_base8_to_base10_conversion_l111_11119

theorem base8_to_base10_conversion : 
  let n := 432
  let base := 8
  let result := 282
  (2 * base^0 + 3 * base^1 + 4 * base^2) = result := 
by
  let n := 2 * 8^0 + 3 * 8^1 + 4 * 8^2
  have h1 : n = 2 + 24 + 256 := by sorry
  have h2 : 2 + 24 + 256 = 282 := by sorry
  exact Eq.trans h1 h2


end NUMINAMATH_GPT_base8_to_base10_conversion_l111_11119


namespace NUMINAMATH_GPT_tabs_per_window_l111_11147

def totalTabs (browsers windowsPerBrowser tabsOpened : Nat) : Nat :=
  tabsOpened / (browsers * windowsPerBrowser)

theorem tabs_per_window : totalTabs 2 3 60 = 10 := by
  sorry

end NUMINAMATH_GPT_tabs_per_window_l111_11147


namespace NUMINAMATH_GPT_loaves_count_l111_11103

theorem loaves_count 
  (init_loaves : ℕ)
  (sold_percent : ℕ) 
  (bulk_purchase : ℕ)
  (bulk_discount_percent : ℕ)
  (evening_purchase : ℕ)
  (evening_discount_percent : ℕ)
  (final_loaves : ℕ)
  (h1 : init_loaves = 2355)
  (h2 : sold_percent = 30)
  (h3 : bulk_purchase = 750)
  (h4 : bulk_discount_percent = 20)
  (h5 : evening_purchase = 489)
  (h6 : evening_discount_percent = 15)
  (h7 : final_loaves = 2888) :
  let mid_morning_sold := init_loaves * sold_percent / 100
  let loaves_after_sale := init_loaves - mid_morning_sold
  let bulk_discount_loaves := bulk_purchase * bulk_discount_percent / 100
  let loaves_after_bulk_purchase := loaves_after_sale + bulk_purchase
  let evening_discount_loaves := evening_purchase * evening_discount_percent / 100
  let loaves_after_evening_purchase := loaves_after_bulk_purchase + evening_purchase
  loaves_after_evening_purchase = final_loaves :=
by
  sorry

end NUMINAMATH_GPT_loaves_count_l111_11103


namespace NUMINAMATH_GPT_teams_played_same_matches_l111_11161

theorem teams_played_same_matches (n : ℕ) (h : n = 30)
  (matches_played : Fin n → ℕ) :
  ∃ (i j : Fin n), i ≠ j ∧ matches_played i = matches_played j :=
by
  sorry

end NUMINAMATH_GPT_teams_played_same_matches_l111_11161


namespace NUMINAMATH_GPT_max_intersections_quadrilateral_l111_11191

-- Define intersection properties
def max_intersections_side : ℕ := 2
def sides_of_quadrilateral : ℕ := 4

theorem max_intersections_quadrilateral : 
  (max_intersections_side * sides_of_quadrilateral) = 8 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_max_intersections_quadrilateral_l111_11191


namespace NUMINAMATH_GPT_number_of_pairs_of_shoes_size_40_to_42_200_pairs_l111_11182

theorem number_of_pairs_of_shoes_size_40_to_42_200_pairs 
  (total_pairs_sample : ℕ)
  (freq_3rd_group : ℝ)
  (freq_1st_group : ℕ)
  (freq_2nd_group : ℕ)
  (freq_4th_group : ℕ)
  (total_pairs_200 : ℕ)
  (scaled_pairs_size_40_42 : ℕ)
: total_pairs_sample = 40 ∧ freq_3rd_group = 0.25 ∧ freq_1st_group = 6 ∧ freq_2nd_group = 7 ∧ freq_4th_group = 9 ∧ total_pairs_200 = 200 ∧ scaled_pairs_size_40_42 = 40 :=
sorry

end NUMINAMATH_GPT_number_of_pairs_of_shoes_size_40_to_42_200_pairs_l111_11182


namespace NUMINAMATH_GPT_books_added_is_10_l111_11194

-- Define initial number of books on the shelf
def initial_books : ℕ := 38

-- Define the final number of books on the shelf
def final_books : ℕ := 48

-- Define the number of books that Marta added
def books_added : ℕ := final_books - initial_books

-- Theorem stating that Marta added 10 books
theorem books_added_is_10 : books_added = 10 :=
by
  sorry

end NUMINAMATH_GPT_books_added_is_10_l111_11194


namespace NUMINAMATH_GPT_average_score_l111_11100

theorem average_score (T : ℝ) (M F : ℝ) (avgM avgF : ℝ) 
  (h1 : M = 0.4 * T) 
  (h2 : M + F = T) 
  (h3 : avgM = 75) 
  (h4 : avgF = 80) : 
  (75 * M + 80 * F) / T = 78 := 
  by 
  sorry

end NUMINAMATH_GPT_average_score_l111_11100


namespace NUMINAMATH_GPT_thread_length_l111_11177

def side_length : ℕ := 13

def perimeter (s : ℕ) : ℕ := 4 * s

theorem thread_length : perimeter side_length = 52 := by
  sorry

end NUMINAMATH_GPT_thread_length_l111_11177


namespace NUMINAMATH_GPT_ellipse_eq_max_area_AEBF_l111_11148

open Real

section ellipse_parabola_problem

variables {a b : ℝ} (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (x y k : ℝ) {M : ℝ × ℝ} {AO BO : ℝ} 
  (b_pos : 0 < b) (a_gt_b : b < a) (MF1_dist : abs (y - 1) = 5 / 3) (M_on_parabola : x^2 = 4 * y)
  (M_on_ellipse : (y / a)^2 + (x / b)^2 = 1) (A : ℝ × ℝ) (B : ℝ × ℝ) (D : ℝ × ℝ)
  (E F : ℝ × ℝ) (A_on_x : A.1 = b ∧ A.2 = 0) (B_on_y : B.1 = 0 ∧ B.2 = a)
  (D_intersect : D.2 = k * D.1) (E_on_ellipse : (E.2 / a)^2 + (E.1 / b)^2 = 1) 
  (F_on_ellipse : (F.2 / a)^2 + (F.1 / b)^2 = 1)
  (k_pos : 0 < k)

theorem ellipse_eq :
  a = 2 ∧ b = sqrt 3 → (y^2 / (2:ℝ)^2 + x^2 / (sqrt 3:ℝ)^2 = 1) :=
sorry

theorem max_area_AEBF :
  (a = 2 ∧ b = sqrt 3) →
  ∃ max_area : ℝ, max_area = 2 * sqrt 6 :=
sorry

end ellipse_parabola_problem

end NUMINAMATH_GPT_ellipse_eq_max_area_AEBF_l111_11148


namespace NUMINAMATH_GPT_six_digit_numbers_with_zero_l111_11170

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end NUMINAMATH_GPT_six_digit_numbers_with_zero_l111_11170


namespace NUMINAMATH_GPT_pauline_bought_2_pounds_of_meat_l111_11181

theorem pauline_bought_2_pounds_of_meat :
  ∀ (cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent : ℝ) 
    (num_bell_peppers : ℕ),
  cost_taco_shells = 5 →
  cost_bell_pepper = 1.5 →
  cost_meat_per_pound = 3 →
  total_spent = 17 →
  num_bell_peppers = 4 →
  (total_spent - (cost_taco_shells + (num_bell_peppers * cost_bell_pepper))) / cost_meat_per_pound = 2 :=
by
  intros cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent num_bell_peppers 
         h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_pauline_bought_2_pounds_of_meat_l111_11181


namespace NUMINAMATH_GPT_xy_difference_l111_11150

theorem xy_difference (x y : ℚ) (h1 : 3 * x - 4 * y = 17) (h2 : x + 3 * y = 5) : x - y = 73 / 13 :=
by
  sorry

end NUMINAMATH_GPT_xy_difference_l111_11150


namespace NUMINAMATH_GPT_sequence_properties_l111_11136

-- Definitions from conditions
def S (n : ℕ) := n^2 - n
def a (n : ℕ) := if n = 1 then 0 else 2 * (n - 1)
def b (n : ℕ) := 2^(n - 1)
def c (n : ℕ) := a n * b n
def T (n : ℕ) := (n - 2) * 2^(n + 1) + 4

-- Theorem statement proving the required identities
theorem sequence_properties {n : ℕ} (hn : n ≠ 0) :
  (a n = (if n = 1 then 0 else 2 * (n - 1))) ∧ 
  (b 2 = a 2) ∧ 
  (b 4 = a 5) ∧ 
  (T n = (n - 2) * 2^(n + 1) + 4) := by
  sorry

end NUMINAMATH_GPT_sequence_properties_l111_11136


namespace NUMINAMATH_GPT_eighth_group_number_correct_stratified_sampling_below_30_correct_l111_11180

noncomputable def systematic_sampling_eighth_group_number 
  (total_employees : ℕ) (sample_size : ℕ) (groups : ℕ) (fifth_group_number : ℕ) : ℕ :=
  let interval := total_employees / groups
  let initial_number := fifth_group_number - 4 * interval
  initial_number + 7 * interval

theorem eighth_group_number_correct :
  systematic_sampling_eighth_group_number 200 40 40 22 = 37 :=
  sorry

noncomputable def stratified_sampling_below_30_persons 
  (total_employees : ℕ) (sample_size : ℕ) (percent_below_30 : ℕ) : ℕ :=
  (percent_below_30 * sample_size) / 100

theorem stratified_sampling_below_30_correct :
  stratified_sampling_below_30_persons 200 40 40 = 16 :=
  sorry

end NUMINAMATH_GPT_eighth_group_number_correct_stratified_sampling_below_30_correct_l111_11180


namespace NUMINAMATH_GPT_average_score_l111_11157

theorem average_score (N : ℕ) (p3 p2 p1 p0 : ℕ) (n : ℕ) 
  (H1 : N = 3)
  (H2 : p3 = 30)
  (H3 : p2 = 50)
  (H4 : p1 = 10)
  (H5 : p0 = 10)
  (H6 : n = 20)
  (H7 : p3 + p2 + p1 + p0 = 100) :
  (3 * (p3 * n / 100) + 2 * (p2 * n / 100) + 1 * (p1 * n / 100) + 0 * (p0 * n / 100)) / n = 2 :=
by 
  sorry

end NUMINAMATH_GPT_average_score_l111_11157


namespace NUMINAMATH_GPT_least_possible_value_of_quadratic_l111_11186

theorem least_possible_value_of_quadratic (p q : ℝ) (hq : ∀ x : ℝ, x^2 + p * x + q ≥ 0) : q = (p^2) / 4 :=
sorry

end NUMINAMATH_GPT_least_possible_value_of_quadratic_l111_11186


namespace NUMINAMATH_GPT_p_p_values_l111_11132

def p (x y : ℤ) : ℤ :=
if 0 ≤ x ∧ 0 ≤ y then x + 2*y
else if x < 0 ∧ y < 0 then x - 3*y
else 4*x + y

theorem p_p_values : p (p 2 (-2)) (p (-3) (-1)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_p_p_values_l111_11132


namespace NUMINAMATH_GPT_find_d_l111_11190

theorem find_d (c d : ℝ) (h1 : c / d = 5) (h2 : c = 18 - 7 * d) : d = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_find_d_l111_11190


namespace NUMINAMATH_GPT_simon_practice_hours_l111_11121

theorem simon_practice_hours (x : ℕ) (h : (12 + 16 + 14 + x) / 4 ≥ 15) : x = 18 := 
by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_simon_practice_hours_l111_11121


namespace NUMINAMATH_GPT_floor_neg_seven_fourths_l111_11135

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end NUMINAMATH_GPT_floor_neg_seven_fourths_l111_11135


namespace NUMINAMATH_GPT_xy_system_l111_11131

theorem xy_system (x y : ℚ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 :=
by
  sorry

end NUMINAMATH_GPT_xy_system_l111_11131


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l111_11138

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 3 * a 2 - 5 * a 1)
  (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, a n < a (n + 1))
  (h4 : ∀ n, a (n + 1) = a n * q) : 
  q = 5 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l111_11138


namespace NUMINAMATH_GPT_tan_alpha_trigonometric_expression_l111_11104

variable (α : ℝ)
variable (h1 : Real.sin (Real.pi + α) = 3 / 5)
variable (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2)

theorem tan_alpha (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : Real.tan α = 3 / 4 := 
sorry

theorem trigonometric_expression (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  (Real.sin ((Real.pi + α) / 2) - Real.cos ((Real.pi + α) / 2)) / 
  (Real.sin ((Real.pi - α) / 2) - Real.cos ((Real.pi - α) / 2)) = -1 / 2 := 
sorry

end NUMINAMATH_GPT_tan_alpha_trigonometric_expression_l111_11104


namespace NUMINAMATH_GPT_find_a_l111_11151

theorem find_a (a : ℝ) (h : a * (1 : ℝ)^2 - 6 * 1 + 3 = 0) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l111_11151


namespace NUMINAMATH_GPT_upper_bound_of_expression_l111_11143

theorem upper_bound_of_expression (n : ℤ) (h1 : ∀ (n : ℤ), 4 * n + 7 > 1 ∧ 4 * n + 7 < 111) :
  ∃ U, (∀ (n : ℤ), 4 * n + 7 < U) ∧ 
       (∀ (n : ℤ), 4 * n + 7 < U ↔ 4 * n + 7 < 111) ∧ 
       U = 111 :=
by
  sorry

end NUMINAMATH_GPT_upper_bound_of_expression_l111_11143


namespace NUMINAMATH_GPT_possible_values_of_a_plus_b_l111_11105

theorem possible_values_of_a_plus_b (a b : ℤ)
  (h1 : ∃ α : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ (∃ (sinα cosα : ℝ), sinα = Real.sin α ∧ cosα = Real.cos α ∧ (sinα + cosα = -a) ∧ (sinα * cosα = 2 * b^2))) :
  a + b = 1 ∨ a + b = -1 := 
sorry

end NUMINAMATH_GPT_possible_values_of_a_plus_b_l111_11105


namespace NUMINAMATH_GPT_sixth_root_of_unity_l111_11179

/- Constants and Variables -/
variable (p q r s t k : ℂ)
variable (nz_p : p ≠ 0) (nz_q : q ≠ 0) (nz_r : r ≠ 0) (nz_s : s ≠ 0) (nz_t : t ≠ 0)
variable (hk1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
variable (hk2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0)

/- Theorem to prove -/
theorem sixth_root_of_unity : k^6 = 1 :=
by sorry

end NUMINAMATH_GPT_sixth_root_of_unity_l111_11179


namespace NUMINAMATH_GPT_first_spade_second_king_prob_l111_11139

-- Definitions and conditions of the problem
def total_cards := 52
def total_spades := 13
def total_kings := 4
def spades_excluding_king := 12 -- Number of spades excluding the king of spades
def remaining_kings_after_king_spade := 3

-- Calculate probabilities for each case
def first_non_king_spade_prob := spades_excluding_king / total_cards
def second_king_after_non_king_spade_prob := total_kings / (total_cards - 1)
def case1_prob := first_non_king_spade_prob * second_king_after_non_king_spade_prob

def first_king_spade_prob := 1 / total_cards
def second_king_after_king_spade_prob := remaining_kings_after_king_spade / (total_cards - 1)
def case2_prob := first_king_spade_prob * second_king_after_king_spade_prob

def combined_prob := case1_prob + case2_prob

-- The proof statement
theorem first_spade_second_king_prob :
  combined_prob = 1 / total_cards := by
  sorry

end NUMINAMATH_GPT_first_spade_second_king_prob_l111_11139


namespace NUMINAMATH_GPT_balloons_remaining_proof_l111_11107

-- The initial number of balloons the clown has
def initial_balloons : ℕ := 3 * 12

-- The number of boys who buy balloons
def boys : ℕ := 3

-- The number of girls who buy balloons
def girls : ℕ := 12

-- The total number of children buying balloons
def total_children : ℕ := boys + girls

-- The remaining number of balloons after sales
def remaining_balloons (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- Problem statement: Proof that the remaining balloons are 21 given the conditions
theorem balloons_remaining_proof : remaining_balloons initial_balloons total_children = 21 := sorry

end NUMINAMATH_GPT_balloons_remaining_proof_l111_11107


namespace NUMINAMATH_GPT_even_square_is_even_l111_11188

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end NUMINAMATH_GPT_even_square_is_even_l111_11188


namespace NUMINAMATH_GPT_proportion_fourth_number_l111_11114

theorem proportion_fourth_number (x y : ℝ) (h₀ : 0.75 * y = 5 * x) (h₁ : x = 1.65) : y = 11 :=
by
  sorry

end NUMINAMATH_GPT_proportion_fourth_number_l111_11114


namespace NUMINAMATH_GPT_desks_built_by_carpenters_l111_11145

theorem desks_built_by_carpenters (h : 2 * 2.5 * r ≥ 2 * r) : 4 * 5 * r ≥ 8 * r :=
by
  sorry

end NUMINAMATH_GPT_desks_built_by_carpenters_l111_11145


namespace NUMINAMATH_GPT_distance_between_bars_l111_11173

theorem distance_between_bars (d V v : ℝ) 
  (h1 : x = 2 * d - 200)
  (h2 : d = P * V)
  (h3 : d - 200 = P * v)
  (h4 : V = (d - 200) / 4)
  (h5 : v = d / 9)
  (h6 : P = 4 * d / (d - 200))
  (h7 : P * (d - 200) = 8)
  (h8 : P * d = 18) :
  x = 1000 := by
  sorry

end NUMINAMATH_GPT_distance_between_bars_l111_11173


namespace NUMINAMATH_GPT_distance_BC_400m_l111_11111

-- Define the hypotheses
variables
  (starting_from_same_time : Prop) -- Sam and Nik start from points A and B respectively at the same time
  (constant_speeds : Prop) -- They travel towards each other at constant speeds along the same route
  (meeting_point_C : Prop) -- They meet at point C, which is 600 m away from starting point A
  (speed_Sam : ℕ) (speed_Sam_value : speed_Sam = 50) -- The speed of Sam is 50 meters per minute
  (time_Sam : ℕ) (time_Sam_value : time_Sam = 20) -- It took Sam 20 minutes to cover the distance between A and B

-- Define the statement to be proven
theorem distance_BC_400m
  (d_AB : ℕ) (d_AB_value : d_AB = speed_Sam * time_Sam)
  (d_AC : ℕ) (d_AC_value : d_AC = 600)
  (d_BC : ℕ) (d_BC_value : d_BC = d_AB - d_AC) :
  d_BC = 400 := by
  sorry

end NUMINAMATH_GPT_distance_BC_400m_l111_11111


namespace NUMINAMATH_GPT_sequence_contradiction_l111_11134

open Classical

variable {α : Type} (a : ℕ → α) [PartialOrder α]

theorem sequence_contradiction {a : ℕ → ℝ} :
  (∀ n, a n < 2) ↔ ¬ ∃ k, a k ≥ 2 := 
by sorry

end NUMINAMATH_GPT_sequence_contradiction_l111_11134


namespace NUMINAMATH_GPT_right_angled_triangle_k_values_l111_11156

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def BC (k : ℝ) : ℝ × ℝ := (1, k - 1)

theorem right_angled_triangle_k_values (k : ℝ) :
  (dot_product AB (AC k) = 0 ∨ dot_product AB (BC k) = 0 ∨ dot_product (BC k) (AC k) = 0) ↔ (k = -6 ∨ k = -1) :=
sorry

end NUMINAMATH_GPT_right_angled_triangle_k_values_l111_11156
