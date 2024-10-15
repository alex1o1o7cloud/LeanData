import Mathlib

namespace NUMINAMATH_GPT_union_of_intervals_l591_59117

theorem union_of_intervals :
  let M := {x : ℝ | x^2 - 3 * x - 4 ≤ 0}
  let N := {x : ℝ | x^2 - 16 ≤ 0}
  M ∪ N = {x : ℝ | -4 ≤ x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_union_of_intervals_l591_59117


namespace NUMINAMATH_GPT_perpendicular_vectors_l591_59123

theorem perpendicular_vectors (b : ℝ) :
  (5 * b - 12 = 0) → b = 12 / 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l591_59123


namespace NUMINAMATH_GPT_power_function_solution_l591_59150

theorem power_function_solution (m : ℝ) 
    (h1 : m^2 - 3 * m + 3 = 1) 
    (h2 : m - 1 ≠ 0) : m = 2 := 
by
  sorry

end NUMINAMATH_GPT_power_function_solution_l591_59150


namespace NUMINAMATH_GPT_card_probability_ratio_l591_59183

theorem card_probability_ratio :
  let total_cards := 40
  let numbers := 10
  let cards_per_number := 4
  let choose (n k : ℕ) := Nat.choose n k
  let p := 10 / choose total_cards 4
  let q := 1440 / choose total_cards 4
  (q / p) = 144 :=
by
  sorry

end NUMINAMATH_GPT_card_probability_ratio_l591_59183


namespace NUMINAMATH_GPT_greatest_divisor_remainders_l591_59132

theorem greatest_divisor_remainders (x : ℕ) (h1 : 1255 % x = 8) (h2 : 1490 % x = 11) : x = 29 :=
by
  -- The proof steps would go here, but for now, we use sorry.
  sorry

end NUMINAMATH_GPT_greatest_divisor_remainders_l591_59132


namespace NUMINAMATH_GPT_tangent_addition_tangent_subtraction_l591_59160

theorem tangent_addition (a b : ℝ) : 
  Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
sorry

theorem tangent_subtraction (a b : ℝ) : 
  Real.tan (a - b) = (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b) :=
sorry

end NUMINAMATH_GPT_tangent_addition_tangent_subtraction_l591_59160


namespace NUMINAMATH_GPT_maximum_value_of_a_l591_59105

theorem maximum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) ↔ a ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_a_l591_59105


namespace NUMINAMATH_GPT_least_integer_square_eq_double_plus_64_l591_59127

theorem least_integer_square_eq_double_plus_64 :
  ∃ x : ℤ, x^2 = 2 * x + 64 ∧ ∀ y : ℤ, y^2 = 2 * y + 64 → y ≥ x → x = -8 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_square_eq_double_plus_64_l591_59127


namespace NUMINAMATH_GPT_shift_parabola_two_units_right_l591_59116

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end NUMINAMATH_GPT_shift_parabola_two_units_right_l591_59116


namespace NUMINAMATH_GPT_is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l591_59155

theorem is_triangle_inequality (a b c: ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem set_B_valid_triangle :
  is_triangle_inequality 5 5 6 := by
  sorry

theorem set_A_not_triangle :
  ¬ is_triangle_inequality 7 4 2 := by
  sorry

theorem set_C_not_triangle :
  ¬ is_triangle_inequality 3 4 8 := by
  sorry

theorem set_D_not_triangle :
  ¬ is_triangle_inequality 2 3 5 := by
  sorry

end NUMINAMATH_GPT_is_triangle_inequality_set_B_valid_triangle_set_A_not_triangle_set_C_not_triangle_set_D_not_triangle_l591_59155


namespace NUMINAMATH_GPT_value_of_c7_l591_59181

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end NUMINAMATH_GPT_value_of_c7_l591_59181


namespace NUMINAMATH_GPT_vikki_worked_42_hours_l591_59106

-- Defining the conditions
def hourly_pay_rate : ℝ := 10
def tax_deduction : ℝ := 0.20 * hourly_pay_rate
def insurance_deduction : ℝ := 0.05 * hourly_pay_rate
def union_dues : ℝ := 5
def take_home_pay : ℝ := 310

-- Equation derived from the given conditions
def total_hours_worked (h : ℝ) : Prop :=
  hourly_pay_rate * h - (tax_deduction * h + insurance_deduction * h + union_dues) = take_home_pay

-- Prove that Vikki worked for 42 hours given the conditions
theorem vikki_worked_42_hours : total_hours_worked 42 := by
  sorry

end NUMINAMATH_GPT_vikki_worked_42_hours_l591_59106


namespace NUMINAMATH_GPT_find_pairs_l591_59171

/-
Define the conditions:
1. The number of three-digit phone numbers consisting of only odd digits.
2. The number of three-digit phone numbers consisting of only even digits excluding 0.
3. Revenue difference is given by a specific equation.
4. \(X\) and \(Y\) are integers less than 250.
-/
def N₁ : ℕ := 5 * 5 * 5  -- Number of combinations with odd digits (1, 3, 5, 7, 9)
def N₂ : ℕ := 4 * 4 * 4  -- Number of combinations with even digits (2, 4, 6, 8)

-- Main theorem: finding pairs (X, Y) that satisfy the given conditions.
theorem find_pairs (X Y : ℕ) (hX : X < 250) (hY : Y < 250) :
  N₁ * X - N₂ * Y = 5 ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) := 
by {
  sorry
}

end NUMINAMATH_GPT_find_pairs_l591_59171


namespace NUMINAMATH_GPT_chess_tournament_participants_l591_59149

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 120) : n = 16 :=
sorry

end NUMINAMATH_GPT_chess_tournament_participants_l591_59149


namespace NUMINAMATH_GPT_min_value_proof_l591_59167

noncomputable def min_value (x y : ℝ) : ℝ :=
x^3 + y^3 - x^2 - y^2

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) : 
min_value x y ≥ 1 := by
  sorry

end NUMINAMATH_GPT_min_value_proof_l591_59167


namespace NUMINAMATH_GPT_cyclic_sequence_u_16_eq_a_l591_59182

-- Sequence definition and recurrence relation
def cyclic_sequence (u : ℕ → ℝ) (a : ℝ) : Prop :=
  u 1 = a ∧ ∀ n : ℕ, u (n + 1) = -1 / (u n + 1)

-- Proof that u_{16} = a under given conditions
theorem cyclic_sequence_u_16_eq_a (a : ℝ) (h : 0 < a) : ∃ (u : ℕ → ℝ), cyclic_sequence u a ∧ u 16 = a :=
by
  sorry

end NUMINAMATH_GPT_cyclic_sequence_u_16_eq_a_l591_59182


namespace NUMINAMATH_GPT_roots_of_Q_are_fifth_powers_of_roots_of_P_l591_59173

def P (x : ℝ) : ℝ := x^3 - 3 * x + 1

noncomputable def Q (y : ℝ) : ℝ := y^3 + 15 * y^2 - 198 * y + 1

theorem roots_of_Q_are_fifth_powers_of_roots_of_P : 
  ∀ α β γ : ℝ, (P α = 0) ∧ (P β = 0) ∧ (P γ = 0) →
  (Q (α^5) = 0) ∧ (Q (β^5) = 0) ∧ (Q (γ^5) = 0) := 
by 
  intros α β γ h
  sorry

end NUMINAMATH_GPT_roots_of_Q_are_fifth_powers_of_roots_of_P_l591_59173


namespace NUMINAMATH_GPT_evaluate_expression_l591_59198

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 240 / 961 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l591_59198


namespace NUMINAMATH_GPT_toys_per_week_l591_59164

-- Define the number of days the workers work in a week
def days_per_week : ℕ := 4

-- Define the number of toys produced each day
def toys_per_day : ℕ := 1140

-- State the proof problem: workers produce 4560 toys per week
theorem toys_per_week : (toys_per_day * days_per_week) = 4560 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_toys_per_week_l591_59164


namespace NUMINAMATH_GPT_microphotonics_budget_allocation_l591_59107

theorem microphotonics_budget_allocation
    (home_electronics : ℕ)
    (food_additives : ℕ)
    (gen_mod_microorg : ℕ)
    (ind_lubricants : ℕ)
    (basic_astrophysics_degrees : ℕ)
    (full_circle_degrees : ℕ := 360)
    (total_budget_percentage : ℕ := 100)
    (basic_astrophysics_percentage : ℕ) :
  home_electronics = 24 →
  food_additives = 15 →
  gen_mod_microorg = 19 →
  ind_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  basic_astrophysics_percentage = (basic_astrophysics_degrees * total_budget_percentage) / full_circle_degrees →
  (total_budget_percentage -
    (home_electronics + food_additives + gen_mod_microorg + ind_lubricants + basic_astrophysics_percentage)) = 14 :=
by
  intros he fa gmm il bad bp
  sorry

end NUMINAMATH_GPT_microphotonics_budget_allocation_l591_59107


namespace NUMINAMATH_GPT_interval_first_bell_l591_59195

theorem interval_first_bell (x : ℕ) : (Nat.lcm (Nat.lcm (Nat.lcm x 10) 14) 18 = 630) → x = 1 := by
  sorry

end NUMINAMATH_GPT_interval_first_bell_l591_59195


namespace NUMINAMATH_GPT_ball_travel_distance_l591_59118

theorem ball_travel_distance 
    (initial_height : ℕ)
    (half : ℕ → ℕ)
    (num_bounces : ℕ)
    (height_after_bounce : ℕ → ℕ)
    (total_distance : ℕ) :
    initial_height = 16 ∧ 
    (∀ n, half n = n / 2) ∧ 
    num_bounces = 4 ∧ 
    (height_after_bounce 0 = initial_height) ∧
    (∀ n, height_after_bounce (n + 1) = half (height_after_bounce n))
→ total_distance = 46 :=
by
  sorry

end NUMINAMATH_GPT_ball_travel_distance_l591_59118


namespace NUMINAMATH_GPT_hash_op_is_100_l591_59179

def hash_op (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_op_is_100 (a b : ℕ) (h1 : a + b = 5) : hash_op a b = 100 :=
sorry

end NUMINAMATH_GPT_hash_op_is_100_l591_59179


namespace NUMINAMATH_GPT_angie_age_l591_59120

variables (age : ℕ)

theorem angie_age (h : 2 * age + 4 = 20) : age = 8 :=
sorry

end NUMINAMATH_GPT_angie_age_l591_59120


namespace NUMINAMATH_GPT_least_number_to_make_divisible_by_9_l591_59194

theorem least_number_to_make_divisible_by_9 (n : ℕ) :
  ∃ m : ℕ, (228712 + m) % 9 = 0 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_make_divisible_by_9_l591_59194


namespace NUMINAMATH_GPT_find_f_neg1_l591_59162

theorem find_f_neg1 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 := 
by 
  -- skipping the proof: 
  sorry

end NUMINAMATH_GPT_find_f_neg1_l591_59162


namespace NUMINAMATH_GPT_cuboid_length_l591_59110

theorem cuboid_length (A b h : ℝ) (A_eq : A = 2400) (b_eq : b = 10) (h_eq : h = 16) :
    ∃ l : ℝ, 2 * (l * b + b * h + h * l) = A ∧ l = 40 := by
  sorry

end NUMINAMATH_GPT_cuboid_length_l591_59110


namespace NUMINAMATH_GPT_find_magnitude_of_z_l591_59177

open Complex

theorem find_magnitude_of_z
    (z : ℂ)
    (h : z^4 = 80 - 96 * I) : abs z = 5^(3/4) :=
by sorry

end NUMINAMATH_GPT_find_magnitude_of_z_l591_59177


namespace NUMINAMATH_GPT_sqrt_number_is_169_l591_59130

theorem sqrt_number_is_169 (a b : ℝ) 
  (h : a^2 + b^2 + (4 * a - 6 * b + 13) = 0) : 
  (a^2 + b^2)^2 = 169 :=
sorry

end NUMINAMATH_GPT_sqrt_number_is_169_l591_59130


namespace NUMINAMATH_GPT_any_nat_representation_as_fraction_l591_59165

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end NUMINAMATH_GPT_any_nat_representation_as_fraction_l591_59165


namespace NUMINAMATH_GPT_part_I_part_II_l591_59196

/-- (I) -/
theorem part_I (x : ℝ) (a : ℝ) (h_a : a = -1) :
  (|2 * x| + |x - 1| ≤ 4) → x ∈ Set.Icc (-1) (5 / 3) :=
by sorry

/-- (II) -/
theorem part_II (x : ℝ) (a : ℝ) (h_eq : |2 * x| + |x + a| = |x - a|) :
  (a > 0 → x ∈ Set.Icc (-a) 0) ∧ (a < 0 → x ∈ Set.Icc 0 (-a)) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l591_59196


namespace NUMINAMATH_GPT_solution_l591_59189

noncomputable def problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : Prop :=
  x + y ≥ 9

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : problem x y h1 h2 h3 :=
  sorry

end NUMINAMATH_GPT_solution_l591_59189


namespace NUMINAMATH_GPT_f_bounds_l591_59119

-- Define the function f with the given properties
def f : ℝ → ℝ :=
sorry 

-- Specify the conditions on f
axiom f_0 : f 0 = 0
axiom f_1 : f 1 = 1
axiom f_ratio (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x < y) (h3 : y < z) (h4 : z ≤ 1) 
  (h5 : z - y = y - x) : 1/2 ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

-- State the theorem to be proven
theorem f_bounds : 1 / 7 ≤ f (1 / 3) ∧ f (1 / 3) ≤ 4 / 7 :=
sorry

end NUMINAMATH_GPT_f_bounds_l591_59119


namespace NUMINAMATH_GPT_largest_two_digit_n_l591_59124

theorem largest_two_digit_n (x : ℕ) (n : ℕ) (hx : x < 10) (hx_nonzero : 0 < x)
  (hn : n = 12 * x * x) (hn_two_digit : n < 100) : n = 48 :=
by sorry

end NUMINAMATH_GPT_largest_two_digit_n_l591_59124


namespace NUMINAMATH_GPT_length_of_scale_parts_l591_59192

theorem length_of_scale_parts (total_length_ft : ℕ) (remaining_inches : ℕ) (parts : ℕ) : 
  total_length_ft = 6 ∧ remaining_inches = 8 ∧ parts = 2 →
  ∃ ft inches, ft = 3 ∧ inches = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_scale_parts_l591_59192


namespace NUMINAMATH_GPT_five_colored_flags_l591_59139

def num_different_flags (colors total_stripes : ℕ) : ℕ :=
  Nat.choose colors total_stripes * Nat.factorial total_stripes

theorem five_colored_flags : num_different_flags 11 5 = 55440 := by
  sorry

end NUMINAMATH_GPT_five_colored_flags_l591_59139


namespace NUMINAMATH_GPT_profit_at_end_of_first_year_l591_59134

theorem profit_at_end_of_first_year :
  let total_amount := 50000
  let part1 := 30000
  let interest_rate1 := 0.10
  let part2 := total_amount - part1
  let interest_rate2 := 0.20
  let time_period := 1
  let interest1 := part1 * interest_rate1 * time_period
  let interest2 := part2 * interest_rate2 * time_period
  let total_profit := interest1 + interest2
  total_profit = 7000 := 
by 
  sorry

end NUMINAMATH_GPT_profit_at_end_of_first_year_l591_59134


namespace NUMINAMATH_GPT_polygon_proof_l591_59121

-- Define the conditions and the final proof problem.
theorem polygon_proof 
  (interior_angle : ℝ) 
  (side_length : ℝ) 
  (h1 : interior_angle = 160) 
  (h2 : side_length = 4) 
  : ∃ n : ℕ, ∃ P : ℝ, (interior_angle = 180 * (n - 2) / n) ∧ (P = n * side_length) ∧ (n = 18) ∧ (P = 72) :=
by
  sorry

end NUMINAMATH_GPT_polygon_proof_l591_59121


namespace NUMINAMATH_GPT_statement_II_must_be_true_l591_59152

-- Define the set of all creatures
variable (Creature : Type)

-- Define properties for being a dragon, mystical, and fire-breathing
variable (Dragon Mystical FireBreathing : Creature → Prop)

-- Given conditions
-- All dragons breathe fire
axiom all_dragons_breathe_fire : ∀ c, Dragon c → FireBreathing c
-- Some mystical creatures are dragons
axiom some_mystical_creatures_are_dragons : ∃ c, Mystical c ∧ Dragon c

-- Questions to prove (we will only formalize the must be true statement)
-- Statement II: Some fire-breathing creatures are mystical creatures

theorem statement_II_must_be_true : ∃ c, FireBreathing c ∧ Mystical c :=
by
  sorry

end NUMINAMATH_GPT_statement_II_must_be_true_l591_59152


namespace NUMINAMATH_GPT_cakes_bought_l591_59199

theorem cakes_bought (initial : ℕ) (left : ℕ) (bought : ℕ) :
  initial = 169 → left = 32 → bought = initial - left → bought = 137 :=
by
  intros h_initial h_left h_bought
  rw [h_initial, h_left] at h_bought
  exact h_bought

end NUMINAMATH_GPT_cakes_bought_l591_59199


namespace NUMINAMATH_GPT_probability_at_least_two_tails_l591_59176

def fair_coin_prob (n : ℕ) : ℚ :=
  (1 / 2 : ℚ)^n

def at_least_two_tails_in_next_three_flips : ℚ :=
  1 - (fair_coin_prob 3 + 3 * fair_coin_prob 3)

theorem probability_at_least_two_tails :
  at_least_two_tails_in_next_three_flips = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_probability_at_least_two_tails_l591_59176


namespace NUMINAMATH_GPT_problem_statement_l591_59145

-- Given: x, y, z are real numbers such that x < 0 and x < y < z
variables {x y z : ℝ} 

-- Conditions
axiom h1 : x < 0
axiom h2 : x < y
axiom h3 : y < z

-- Statement to prove: x + y < y + z
theorem problem_statement : x + y < y + z :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l591_59145


namespace NUMINAMATH_GPT_Jack_hands_in_l591_59131

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end NUMINAMATH_GPT_Jack_hands_in_l591_59131


namespace NUMINAMATH_GPT_proposition_3_correct_l591_59174

open Real

def is_obtuse (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

theorem proposition_3_correct (A B C : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : 0 < C) (h₃ : A + B + C = π)
  (h : sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1) : is_obtuse A B C :=
by
  sorry

end NUMINAMATH_GPT_proposition_3_correct_l591_59174


namespace NUMINAMATH_GPT_normal_time_to_finish_bs_l591_59115

theorem normal_time_to_finish_bs (P : ℕ) (H1 : P = 5) (H2 : ∀ total_time, total_time = 6 → total_time = (3 / 4) * (P + B)) : B = (8 - P) :=
by sorry

end NUMINAMATH_GPT_normal_time_to_finish_bs_l591_59115


namespace NUMINAMATH_GPT_expression_of_quadratic_function_coordinates_of_vertex_l591_59111

def quadratic_function_through_points (a b : ℝ) : Prop :=
  (0 = a * (-3)^2 + b * (-3) + 3) ∧ (-5 = a * 2^2 + b * 2 + 3)

theorem expression_of_quadratic_function :
  ∃ a b : ℝ, quadratic_function_through_points a b ∧ ∀ x : ℝ, -x^2 - 2 * x + 3 = a * x^2 + b * x + 3 :=
by
  sorry

theorem coordinates_of_vertex :
  - (1 : ℝ) * (1 : ℝ) = (-1) / (2 * (-1)) ∧ 4 = -(1 - (-1) + 3) + 4 :=
by
  sorry

end NUMINAMATH_GPT_expression_of_quadratic_function_coordinates_of_vertex_l591_59111


namespace NUMINAMATH_GPT_findPrincipalAmount_l591_59103

noncomputable def principalAmount (r : ℝ) (t : ℝ) (diff : ℝ) : ℝ :=
  let n := 2 -- compounded semi-annually
  let rate_per_period := (1 + r / n)
  let num_periods := n * t
  (diff / (rate_per_period^num_periods - 1 - r * t))

theorem findPrincipalAmount :
  let r := 0.05
  let t := 3
  let diff := 25
  abs (principalAmount r t diff - 2580.39) < 0.01 := 
by 
  sorry

end NUMINAMATH_GPT_findPrincipalAmount_l591_59103


namespace NUMINAMATH_GPT_expression_evaluation_l591_59148

theorem expression_evaluation :
  2 - 3 * (-4) + 5 - (-6) * 7 = 61 :=
sorry

end NUMINAMATH_GPT_expression_evaluation_l591_59148


namespace NUMINAMATH_GPT_freds_total_marbles_l591_59159

theorem freds_total_marbles :
  let red := 38
  let green := red / 2
  let dark_blue := 6
  red + green + dark_blue = 63 := by
  sorry

end NUMINAMATH_GPT_freds_total_marbles_l591_59159


namespace NUMINAMATH_GPT_simplify_sqrt_l591_59128

theorem simplify_sqrt (x : ℝ) (h : x = (Real.sqrt 3) + 1) : Real.sqrt (x^2) = Real.sqrt 3 + 1 :=
by
  -- This will serve as the placeholder for the proof.
  sorry

end NUMINAMATH_GPT_simplify_sqrt_l591_59128


namespace NUMINAMATH_GPT_algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l591_59156

theorem algebraic_simplification (x : ℤ) (h1 : -3 < x) (h2 : x < 3) (h3 : x ≠ 0) (h4 : x ≠ 1) (h5 : x ≠ -1) :
  (x - (x / (x + 1))) / (1 + (1 / (x^2 - 1))) = x - 1 :=
sorry

theorem evaluate_expression_for_x2 (h1 : -3 < 2) (h2 : 2 < 3) (h3 : 2 ≠ 0) (h4 : 2 ≠ 1) (h5 : 2 ≠ -1) :
  (2 - (2 / (2 + 1))) / (1 + (1 / (2^2 - 1))) = 1 :=
sorry

theorem evaluate_expression_for_x_neg2 (h1 : -3 < -2) (h2 : -2 < 3) (h3 : -2 ≠ 0) (h4 : -2 ≠ 1) (h5 : -2 ≠ -1) :
  (-2 - (-2 / (-2 + 1))) / (1 + (1 / ((-2)^2 - 1))) = -3 :=
sorry

end NUMINAMATH_GPT_algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l591_59156


namespace NUMINAMATH_GPT_speed_ratio_l591_59191

theorem speed_ratio (v_A v_B : ℝ) (h : 71 / v_B = 142 / v_A) : v_A / v_B = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l591_59191


namespace NUMINAMATH_GPT_isosceles_triangle_aacute_l591_59184

theorem isosceles_triangle_aacute (a b c : ℝ) (h1 : a = b) (h2 : a + b + c = 180) (h3 : c = 108)
  : ∃ x y z : ℝ, x + y + z = 180 ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_aacute_l591_59184


namespace NUMINAMATH_GPT_correct_location_l591_59122

-- Define the possible options
inductive Location
| A : Location
| B : Location
| C : Location
| D : Location

-- Define the conditions
def option_A : Prop := ¬(∃ d, d ≠ "right")
def option_B : Prop := ¬(∃ d, d ≠ 900)
def option_C : Prop := ¬(∃ d, d ≠ "west")
def option_D : Prop := (∃ d₁ d₂, d₁ = "west" ∧ d₂ = 900)

-- The objective is to prove that option D is the correct description of the location
theorem correct_location : ∃ l, l = Location.D → 
  (option_A ∧ option_B ∧ option_C ∧ option_D) :=
by
  sorry

end NUMINAMATH_GPT_correct_location_l591_59122


namespace NUMINAMATH_GPT_sharpened_off_length_l591_59146

-- Define the conditions
def original_length : ℤ := 31
def length_after_sharpening : ℤ := 14

-- Define the theorem to prove the length sharpened off is 17 inches
theorem sharpened_off_length : original_length - length_after_sharpening = 17 := sorry

end NUMINAMATH_GPT_sharpened_off_length_l591_59146


namespace NUMINAMATH_GPT_quarters_for_soda_l591_59175

def quarters_for_chips := 4
def total_dollars := 4

theorem quarters_for_soda :
  (total_dollars * 4) - quarters_for_chips = 12 :=
by
  sorry

end NUMINAMATH_GPT_quarters_for_soda_l591_59175


namespace NUMINAMATH_GPT_rectangle_area_increase_l591_59102

theorem rectangle_area_increase (a b : ℝ) :
  let new_length := (1 + 1/4) * a
  let new_width := (1 + 1/5) * b
  let original_area := a * b
  let new_area := new_length * new_width
  let area_increase := new_area - original_area
  (area_increase / original_area) = 1/2 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l591_59102


namespace NUMINAMATH_GPT_k_plus_a_equals_three_halves_l591_59112

theorem k_plus_a_equals_three_halves :
  ∃ (k a : ℝ), (2 = k * 4 ^ a) ∧ (k + a = 3 / 2) :=
sorry

end NUMINAMATH_GPT_k_plus_a_equals_three_halves_l591_59112


namespace NUMINAMATH_GPT_days_in_month_l591_59104

theorem days_in_month
  (monthly_production : ℕ)
  (production_per_half_hour : ℚ)
  (hours_per_day : ℕ)
  (daily_production : ℚ)
  (days_in_month : ℚ) :
  monthly_production = 8400 ∧
  production_per_half_hour = 6.25 ∧
  hours_per_day = 24 ∧
  daily_production = production_per_half_hour * 2 * hours_per_day ∧
  days_in_month = monthly_production / daily_production
  → days_in_month = 28 :=
by
  sorry

end NUMINAMATH_GPT_days_in_month_l591_59104


namespace NUMINAMATH_GPT_seashells_count_l591_59197

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end NUMINAMATH_GPT_seashells_count_l591_59197


namespace NUMINAMATH_GPT_probability_of_3_black_face_cards_l591_59137

-- Definitions based on conditions
def total_cards : ℕ := 36
def total_black_face_cards : ℕ := 8
def total_other_cards : ℕ := total_cards - total_black_face_cards
def draw_cards : ℕ := 6
def draw_black_face_cards : ℕ := 3
def draw_other_cards := draw_cards - draw_black_face_cards

-- Calculation using combinations
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_combinations : ℕ := combination total_cards draw_cards
noncomputable def favorable_combinations : ℕ := combination total_black_face_cards draw_black_face_cards * combination total_other_cards draw_other_cards

-- Calculating probability
noncomputable def probability : ℚ := favorable_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_3_black_face_cards : probability = 11466 / 121737 := by
  -- proof
  sorry

end NUMINAMATH_GPT_probability_of_3_black_face_cards_l591_59137


namespace NUMINAMATH_GPT_undefined_integer_count_l591_59170

noncomputable def expression (x : ℤ) : ℚ := (x^2 - 16) / ((x^2 - x - 6) * (x - 4))

theorem undefined_integer_count : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x^2 - x - 6) * (x - 4) = 0) ∧ S.card = 3 :=
  sorry

end NUMINAMATH_GPT_undefined_integer_count_l591_59170


namespace NUMINAMATH_GPT_toms_expense_l591_59166

def cost_per_square_foot : ℝ := 5
def square_feet_per_seat : ℝ := 12
def number_of_seats : ℝ := 500
def partner_coverage : ℝ := 0.40

def total_square_feet : ℝ := square_feet_per_seat * number_of_seats
def land_cost : ℝ := cost_per_square_foot * total_square_feet
def construction_cost : ℝ := 2 * land_cost
def total_cost : ℝ := land_cost + construction_cost
def tom_coverage_percentage : ℝ := 1 - partner_coverage
def toms_share : ℝ := tom_coverage_percentage * total_cost

theorem toms_expense :
  toms_share = 54000 :=
by
  sorry

end NUMINAMATH_GPT_toms_expense_l591_59166


namespace NUMINAMATH_GPT_cannot_make_62_cents_with_five_coins_l591_59129

theorem cannot_make_62_cents_with_five_coins :
  ∀ (p n d q : ℕ), p + n + d + q = 5 ∧ q ≤ 1 →
  1 * p + 5 * n + 10 * d + 25 * q ≠ 62 := by
  intro p n d q h
  sorry

end NUMINAMATH_GPT_cannot_make_62_cents_with_five_coins_l591_59129


namespace NUMINAMATH_GPT_part_one_part_two_l591_59161

namespace ProofProblem

def setA (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def setB := {x : ℝ | 0 < x ∧ x < 1}

theorem part_one (a : ℝ) (h : a = 1/2) : 
  setA a ∩ setB = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

theorem part_two (a : ℝ) (h_subset : setB ⊆ setA a) : 
  0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_part_one_part_two_l591_59161


namespace NUMINAMATH_GPT_croissants_for_breakfast_l591_59108

def total_items (C : ℕ) : Prop :=
  C + 18 + 30 = 110

theorem croissants_for_breakfast (C : ℕ) (h : total_items C) : C = 62 :=
by {
  -- The proof might be here, but since it's not required:
  sorry
}

end NUMINAMATH_GPT_croissants_for_breakfast_l591_59108


namespace NUMINAMATH_GPT_squares_below_16x_144y_1152_l591_59187

noncomputable def count_squares_below_line (a b c : ℝ) (x_max y_max : ℝ) : ℝ :=
  let total_squares := x_max * y_max
  let line_slope := -a/b
  let squares_crossed_by_diagonal := x_max + y_max - 1
  (total_squares - squares_crossed_by_diagonal) / 2

theorem squares_below_16x_144y_1152 : 
  count_squares_below_line 16 144 1152 72 8 = 248.5 := 
by
  sorry

end NUMINAMATH_GPT_squares_below_16x_144y_1152_l591_59187


namespace NUMINAMATH_GPT_simplify_f_value_of_f_l591_59141

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - (5 * Real.pi) / 2) * Real.cos ((3 * Real.pi) / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem simplify_f (α : ℝ) : f α = -Real.cos α := by
  sorry

theorem value_of_f (α : ℝ)
  (h : Real.cos (α + (3 * Real.pi) / 2) = 1 / 5)
  (h2 : α > Real.pi / 2 ∧ α < Real.pi ) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_GPT_simplify_f_value_of_f_l591_59141


namespace NUMINAMATH_GPT_definite_integral_eval_l591_59190

theorem definite_integral_eval :
  ∫ x in (1:ℝ)..(3:ℝ), (2 * x - 1 / x ^ 2) = 22 / 3 :=
by
  sorry

end NUMINAMATH_GPT_definite_integral_eval_l591_59190


namespace NUMINAMATH_GPT_find_multiplier_value_l591_59140

def number : ℤ := 18
def increase : ℤ := 198

theorem find_multiplier_value (x : ℤ) (h : number * x = number + increase) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_value_l591_59140


namespace NUMINAMATH_GPT_number_in_pattern_l591_59109

theorem number_in_pattern (m n : ℕ) (h : 8 * m - 5 = 2023) (hn : n = 5) : m + n = 258 :=
by
  sorry

end NUMINAMATH_GPT_number_in_pattern_l591_59109


namespace NUMINAMATH_GPT_cosine_evaluation_l591_59135

variable (α : ℝ)

theorem cosine_evaluation
  (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (Real.pi / 3 - α) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_cosine_evaluation_l591_59135


namespace NUMINAMATH_GPT_sum_of_maximum_and_minimum_of_u_l591_59138

theorem sum_of_maximum_and_minimum_of_u :
  ∀ (x y z : ℝ),
    0 ≤ x → 0 ≤ y → 0 ≤ z →
    3 * x + 2 * y + z = 5 →
    2 * x + y - 3 * z = 1 →
    3 * x + y - 7 * z = 3 * z - 2 →
    (-5 : ℝ) / 7 + (-1 : ℝ) / 11 = -62 / 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_maximum_and_minimum_of_u_l591_59138


namespace NUMINAMATH_GPT_initial_boys_l591_59113

-- Define the initial condition
def initial_girls : ℕ := 18
def additional_girls : ℕ := 7
def quitting_boys : ℕ := 4
def total_children_after_changes : ℕ := 36

-- Define the initial number of boys
variable (B : ℕ)

-- State the main theorem
theorem initial_boys (h : 25 + (B - 4) = 36) : B = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_boys_l591_59113


namespace NUMINAMATH_GPT_polygon_interior_angle_sum_l591_59158

theorem polygon_interior_angle_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * (n - 2 + 3) = 2880 := by
  sorry

end NUMINAMATH_GPT_polygon_interior_angle_sum_l591_59158


namespace NUMINAMATH_GPT_duration_of_loan_l591_59168

namespace SimpleInterest

variables (P SI R : ℝ) (T : ℝ)

-- Defining the conditions
def principal := P = 1500
def simple_interest := SI = 735
def rate := R = 7 / 100

-- The question: Prove the duration (T) of the loan
theorem duration_of_loan (hP : principal P) (hSI : simple_interest SI) (hR : rate R) :
  T = 7 :=
sorry

end SimpleInterest

end NUMINAMATH_GPT_duration_of_loan_l591_59168


namespace NUMINAMATH_GPT_man_speed_was_5_kmph_l591_59157

theorem man_speed_was_5_kmph (time_in_minutes : ℕ) (distance_in_km : ℝ)
  (h_time : time_in_minutes = 30)
  (h_distance : distance_in_km = 2.5) :
  (distance_in_km / (time_in_minutes / 60 : ℝ) = 5) :=
by
  sorry

end NUMINAMATH_GPT_man_speed_was_5_kmph_l591_59157


namespace NUMINAMATH_GPT_students_qualifying_percentage_l591_59193

theorem students_qualifying_percentage (N B G : ℕ) (boy_percent : ℝ) (girl_percent : ℝ) :
  N = 400 →
  G = 100 →
  B = N - G →
  boy_percent = 0.60 →
  girl_percent = 0.80 →
  (boy_percent * B + girl_percent * G) / N * 100 = 65 :=
by
  intros hN hG hB hBoy hGirl
  simp [hN, hG, hB, hBoy, hGirl]
  sorry

end NUMINAMATH_GPT_students_qualifying_percentage_l591_59193


namespace NUMINAMATH_GPT_triangle_inequality_lt_l591_59154

theorem triangle_inequality_lt {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a < b + c) (h2 : b < a + c) (h3 : c < a + b) : a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := 
sorry

end NUMINAMATH_GPT_triangle_inequality_lt_l591_59154


namespace NUMINAMATH_GPT_problem_statement_l591_59163

open Real

namespace MathProblem

def p₁ := ∃ x : ℝ, x^2 + x + 1 < 0
def p₂ := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_statement : (¬p₁) ∨ (¬p₂) :=
by
  sorry

end MathProblem

end NUMINAMATH_GPT_problem_statement_l591_59163


namespace NUMINAMATH_GPT_max_value_k_l591_59143

theorem max_value_k (x y : ℝ) (k : ℝ) (h₁ : x^2 + y^2 = 1) (h₂ : ∀ x y, x^2 + y^2 = 1 → x + y - k ≥ 0) : 
  k ≤ -Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_k_l591_59143


namespace NUMINAMATH_GPT_inequality_proof_l591_59153

theorem inequality_proof (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n):
  x^n / (1 + x^2) + y^n / (1 + y^2) ≤ (x^n + y^n) / (1 + x * y) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l591_59153


namespace NUMINAMATH_GPT_sum_of_terms_arithmetic_sequence_l591_59147

variable {S : ℕ → ℕ}
variable {k : ℕ}

-- Given conditions
axiom S_k : S k = 2
axiom S_3k : S (3 * k) = 18

-- The statement to prove
theorem sum_of_terms_arithmetic_sequence : S (4 * k) = 32 := by
  sorry

end NUMINAMATH_GPT_sum_of_terms_arithmetic_sequence_l591_59147


namespace NUMINAMATH_GPT_line_equation_l591_59100

theorem line_equation (a : ℝ) (P : ℝ × ℝ) (hx : P = (5, 6)) 
                      (cond : (a ≠ 0) ∧ (2 * a = 17)) : 
  ∃ (m b : ℝ), - (m * (0 : ℝ) + b) = a ∧ (- m * 17 / 2 + b) = 6 ∧ 
               (x + 2 * y - 17 =  0) := sorry

end NUMINAMATH_GPT_line_equation_l591_59100


namespace NUMINAMATH_GPT_member_age_greater_than_zero_l591_59186

def num_members : ℕ := 23
def avg_age : ℤ := 0
def age_range : Set ℤ := {x | x ≥ -20 ∧ x ≤ 20}
def num_negative_members : ℕ := 5

theorem member_age_greater_than_zero :
  ∃ n : ℕ, n ≤ 18 ∧ (avg_age = 0 ∧ num_members = 23 ∧ num_negative_members = 5 ∧ ∀ age ∈ age_range, age ≥ -20 ∧ age ≤ 20) :=
sorry

end NUMINAMATH_GPT_member_age_greater_than_zero_l591_59186


namespace NUMINAMATH_GPT_probability_no_absolute_winner_l591_59169

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end NUMINAMATH_GPT_probability_no_absolute_winner_l591_59169


namespace NUMINAMATH_GPT_investor_amount_after_two_years_l591_59178

noncomputable def compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investor_amount_after_two_years :
  compound_interest 3000 0.10 1 2 = 3630 :=
by
  -- Calculation goes here
  sorry

end NUMINAMATH_GPT_investor_amount_after_two_years_l591_59178


namespace NUMINAMATH_GPT_area_of_circle_r_is_16_percent_of_circle_s_l591_59185

open Real

variables (Ds Dr Rs Rr As Ar : ℝ)

def circle_r_is_40_percent_of_circle_s (Ds Dr : ℝ) := Dr = 0.40 * Ds
def radius_of_circle (D : ℝ) (R : ℝ) := R = D / 2
def area_of_circle (R : ℝ) (A : ℝ) := A = π * R^2
def percentage_area (As Ar : ℝ) (P : ℝ) := P = (Ar / As) * 100

theorem area_of_circle_r_is_16_percent_of_circle_s :
  ∀ (Ds Dr Rs Rr As Ar : ℝ),
    circle_r_is_40_percent_of_circle_s Ds Dr →
    radius_of_circle Ds Rs →
    radius_of_circle Dr Rr →
    area_of_circle Rs As →
    area_of_circle Rr Ar →
    percentage_area As Ar 16 := by
  intros Ds Dr Rs Rr As Ar H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_area_of_circle_r_is_16_percent_of_circle_s_l591_59185


namespace NUMINAMATH_GPT_radius_range_of_circle_l591_59133

theorem radius_range_of_circle (r : ℝ) :
  (∃ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 ∧ 
    (∃ a b : ℝ, 4*a - 3*b - 2 = 0 ∧ ∃ c d : ℝ, 4*c - 3*d - 2 = 0 ∧ 
      (a - x)^2 + (b - y)^2 = 1 ∧ (c - x)^2 + (d - y)^2 = 1 ∧
       a ≠ c ∧ b ≠ d)) ↔ 4 < r ∧ r < 6 :=
by
  sorry

end NUMINAMATH_GPT_radius_range_of_circle_l591_59133


namespace NUMINAMATH_GPT_exponent_of_two_gives_n_l591_59126

theorem exponent_of_two_gives_n (x: ℝ) (n: ℝ) (b: ℝ)
  (h1: n = 2 ^ x)
  (h2: n ^ b = 8)
  (h3: b = 12) : x = 3 / 12 :=
by
  sorry

end NUMINAMATH_GPT_exponent_of_two_gives_n_l591_59126


namespace NUMINAMATH_GPT_negation_example_l591_59125

theorem negation_example : ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ ∃ x : ℝ, x > 1 ∧ x^2 ≤ 1 := by
  sorry

end NUMINAMATH_GPT_negation_example_l591_59125


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l591_59101

theorem necessary_and_sufficient_condition 
  (a : ℕ) 
  (A B : ℝ) 
  (x y z : ℤ) 
  (h1 : (x^2 + y^2 + z^2 : ℝ) = (B * ↑a)^2) 
  (h2 : (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) : ℝ) = (1 / 4) * (2 * A + B) * (B * (↑a)^4)) :
  B = 2 * A :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l591_59101


namespace NUMINAMATH_GPT_apple_crisps_calculation_l591_59151

theorem apple_crisps_calculation (apples crisps : ℕ) (h : crisps = 3 ∧ apples = 12) : 
  (36 / apples) * crisps = 9 := by
  sorry

end NUMINAMATH_GPT_apple_crisps_calculation_l591_59151


namespace NUMINAMATH_GPT_prime_in_range_l591_59142

theorem prime_in_range (p: ℕ) (h_prime: Nat.Prime p) (h_int_roots: ∃ a b: ℤ, a ≠ b ∧ a + b = -p ∧ a * b = -520 * p) : 11 < p ∧ p ≤ 21 := 
by
  sorry

end NUMINAMATH_GPT_prime_in_range_l591_59142


namespace NUMINAMATH_GPT_average_percentage_decrease_l591_59114

theorem average_percentage_decrease (x : ℝ) : 60 * (1 - x) * (1 - x) = 48.6 → x = 0.1 :=
by sorry

end NUMINAMATH_GPT_average_percentage_decrease_l591_59114


namespace NUMINAMATH_GPT_smallest_s_triangle_l591_59188

theorem smallest_s_triangle (s : ℕ) :
  (7 + s > 11) ∧ (7 + 11 > s) ∧ (11 + s > 7) → s = 5 :=
sorry

end NUMINAMATH_GPT_smallest_s_triangle_l591_59188


namespace NUMINAMATH_GPT_books_in_final_category_l591_59180

-- Define the number of initial books
def initial_books : ℕ := 400

-- Define the number of divisions
def num_divisions : ℕ := 4

-- Define the iterative division process
def final_books (initial : ℕ) (divisions : ℕ) : ℕ :=
  initial / (2 ^ divisions)

-- State the theorem
theorem books_in_final_category : final_books initial_books num_divisions = 25 := by
  sorry

end NUMINAMATH_GPT_books_in_final_category_l591_59180


namespace NUMINAMATH_GPT_smallest_integer_CC4_DD6_rep_l591_59136

-- Lean 4 Statement
theorem smallest_integer_CC4_DD6_rep (C D : ℕ) (hC : C < 4) (hD : D < 6) :
  (5 * C = 7 * D) → (5 * C = 35 ∧ 7 * D = 35) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_CC4_DD6_rep_l591_59136


namespace NUMINAMATH_GPT_converse_proposition_inverse_proposition_contrapositive_proposition_l591_59172

theorem converse_proposition (x y : ℝ) : (xy = 0 → x^2 + y^2 = 0) = false :=
sorry

theorem inverse_proposition (x y : ℝ) : (x^2 + y^2 ≠ 0 → xy ≠ 0) = false :=
sorry

theorem contrapositive_proposition (x y : ℝ) : (xy ≠ 0 → x^2 + y^2 ≠ 0) = true :=
sorry

end NUMINAMATH_GPT_converse_proposition_inverse_proposition_contrapositive_proposition_l591_59172


namespace NUMINAMATH_GPT_point_B_represent_l591_59144

-- Given conditions
def point_A := -2
def units_moved := 4

-- Lean statement to prove
theorem point_B_represent : 
  ∃ B : ℤ, (B = point_A - units_moved) ∨ (B = point_A + units_moved) := by
    sorry

end NUMINAMATH_GPT_point_B_represent_l591_59144
