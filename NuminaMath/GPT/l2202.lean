import Mathlib

namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2202_220233

noncomputable def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
noncomputable def B : Set ℝ := { x | 0 ≤ x }

theorem intersection_of_A_and_B :
  { x | x ∈ A ∧ x ∈ B } = { x | 0 ≤ x ∧ x ≤ 3 } :=
  by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2202_220233


namespace NUMINAMATH_GPT_smallest_multiple_of_45_and_75_not_20_l2202_220229

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_45_and_75_not_20_l2202_220229


namespace NUMINAMATH_GPT_Rihanna_money_left_l2202_220213

theorem Rihanna_money_left (initial_money mango_count juice_count mango_price juice_price : ℕ)
  (h_initial : initial_money = 50)
  (h_mango_count : mango_count = 6)
  (h_juice_count : juice_count = 6)
  (h_mango_price : mango_price = 3)
  (h_juice_price : juice_price = 3) :
  initial_money - (mango_count * mango_price + juice_count * juice_price) = 14 :=
sorry

end NUMINAMATH_GPT_Rihanna_money_left_l2202_220213


namespace NUMINAMATH_GPT_fixed_point_is_5_225_l2202_220288

theorem fixed_point_is_5_225 : ∃ a b : ℝ, (∀ k : ℝ, 9 * a^2 + k * a - 5 * k = b) → (a = 5 ∧ b = 225) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_is_5_225_l2202_220288


namespace NUMINAMATH_GPT_add_base8_numbers_l2202_220296

def fromBase8 (n : Nat) : Nat :=
  Nat.digits 8 n |> Nat.ofDigits 8

theorem add_base8_numbers : 
  fromBase8 356 + fromBase8 672 + fromBase8 145 = fromBase8 1477 :=
by
  sorry

end NUMINAMATH_GPT_add_base8_numbers_l2202_220296


namespace NUMINAMATH_GPT_train_speed_correct_l2202_220258

def length_of_train : ℕ := 700
def time_to_cross_pole : ℕ := 20
def expected_speed : ℕ := 35

theorem train_speed_correct : (length_of_train / time_to_cross_pole) = expected_speed := by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l2202_220258


namespace NUMINAMATH_GPT_odd_function_domain_real_l2202_220295

theorem odd_function_domain_real
  (a : ℤ)
  (h_condition : a = -1 ∨ a = 1 ∨ a = 3) :
  (∀ x : ℝ, ∃ y : ℝ, x ≠ 0 → y = x^a) →
  (∀ x : ℝ, x ≠ 0 → (x^a = (-x)^a)) →
  (a = 1 ∨ a = 3) :=
sorry

end NUMINAMATH_GPT_odd_function_domain_real_l2202_220295


namespace NUMINAMATH_GPT_find_a_l2202_220207

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) :
  (2 * x₁ + 1 = 3) →
  (2 - (a - x₂) / 3 = 1) →
  (x₁ = x₂) →
  a = 4 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_find_a_l2202_220207


namespace NUMINAMATH_GPT_units_digit_base6_product_l2202_220281

theorem units_digit_base6_product (a b : ℕ) (h1 : a = 168) (h2 : b = 59) : ((a * b) % 6) = 0 := by
  sorry

end NUMINAMATH_GPT_units_digit_base6_product_l2202_220281


namespace NUMINAMATH_GPT_tangent_curves_line_exists_l2202_220235

theorem tangent_curves_line_exists (a : ℝ) :
  (∃ l : ℝ → ℝ, ∃ x₀ : ℝ, l 1 = 0 ∧ ∀ x, (l x = x₀^3 ∧ l x = a * x^2 + (15 / 4) * x - 9)) →
  a = -25/64 ∨ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_curves_line_exists_l2202_220235


namespace NUMINAMATH_GPT_simplify_expression_l2202_220251

theorem simplify_expression (x : ℝ) (h1 : x^2 - 4*x + 3 ≠ 0) (h2 : x^2 - 6*x + 9 ≠ 0) (h3 : x^2 - 3*x + 2 ≠ 0) (h4 : x^2 - 4*x + 4 ≠ 0) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / (x^2 - 3*x + 2) / (x^2 - 4*x + 4) = (x-2) / (x-3) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l2202_220251


namespace NUMINAMATH_GPT_average_height_corrected_l2202_220272

-- Defining the conditions as functions and constants
def incorrect_average_height : ℝ := 175
def number_of_students : ℕ := 30
def incorrect_height : ℝ := 151
def actual_height : ℝ := 136

-- The target average height to prove
def target_actual_average_height : ℝ := 174.5

-- Main theorem stating the problem
theorem average_height_corrected : 
  (incorrect_average_height * number_of_students - (incorrect_height - actual_height)) / number_of_students = target_actual_average_height :=
by
  sorry

end NUMINAMATH_GPT_average_height_corrected_l2202_220272


namespace NUMINAMATH_GPT_group_count_l2202_220214

theorem group_count (sample_capacity : ℕ) (frequency : ℝ) (h_sample_capacity : sample_capacity = 80) (h_frequency : frequency = 0.125) : sample_capacity * frequency = 10 := 
by
  sorry

end NUMINAMATH_GPT_group_count_l2202_220214


namespace NUMINAMATH_GPT_leak_empty_time_l2202_220279

/-- 
The time taken for a leak to empty a full tank, given that an electric pump can fill a tank in 7 hours and it takes 14 hours to fill the tank with the leak present, is 14 hours.
 -/
theorem leak_empty_time (P L : ℝ) (hP : P = 1 / 7) (hCombined : P - L = 1 / 14) : L = 1 / 14 ∧ 1 / L = 14 :=
by
  sorry

end NUMINAMATH_GPT_leak_empty_time_l2202_220279


namespace NUMINAMATH_GPT_abs_diff_eq_1point5_l2202_220286

theorem abs_diff_eq_1point5 (x y : ℝ)
    (hx : (⌊x⌋ : ℝ) + (y - ⌊y⌋) = 3.7)
    (hy : (x - ⌊x⌋) + (⌊y⌋ : ℝ) = 4.2) :
        |x - y| = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_eq_1point5_l2202_220286


namespace NUMINAMATH_GPT_gcd_polynomial_example_l2202_220271

theorem gcd_polynomial_example (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1177 * k) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_example_l2202_220271


namespace NUMINAMATH_GPT_first_player_wins_l2202_220212

-- Define the initial conditions
def initial_pile_1 : ℕ := 100
def initial_pile_2 : ℕ := 200

-- Define the game rules
def valid_move (pile_1 pile_2 n : ℕ) : Prop :=
  (n > 0) ∧ ((n <= pile_1) ∨ (n <= pile_2))

-- The game state is represented as a pair of natural numbers
def GameState := ℕ × ℕ

-- Define what it means to win the game
def winning_move (s: GameState) : Prop :=
  (s.1 = 0 ∧ s.2 = 1) ∨ (s.1 = 1 ∧ s.2 = 0)

-- Define the main theorem
theorem first_player_wins : 
  ∀ s : GameState, (s = (initial_pile_1, initial_pile_2)) → (∃ move, valid_move s.1 s.2 move ∧ winning_move (s.1 - move, s.2 - move)) :=
sorry

end NUMINAMATH_GPT_first_player_wins_l2202_220212


namespace NUMINAMATH_GPT_operation_addition_l2202_220273

theorem operation_addition (a b c : ℝ) (op : ℝ → ℝ → ℝ)
  (H : ∀ a b c : ℝ, op (op a b) c = a + b + c) :
  ∀ a b : ℝ, op a b = a + b :=
sorry

end NUMINAMATH_GPT_operation_addition_l2202_220273


namespace NUMINAMATH_GPT_sequence_of_numbers_exists_l2202_220260

theorem sequence_of_numbers_exists :
  ∃ (a b : ℤ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) :=
sorry

end NUMINAMATH_GPT_sequence_of_numbers_exists_l2202_220260


namespace NUMINAMATH_GPT_cost_of_child_ticket_l2202_220275

-- Define the conditions
def adult_ticket_cost : ℕ := 60
def total_people : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80
def adults_attended : ℕ := total_people - children_attended
def total_collected_from_adults : ℕ := adults_attended * adult_ticket_cost

-- State the theorem to prove the cost of a child ticket
theorem cost_of_child_ticket (x : ℕ) :
  total_collected_from_adults + children_attended * x = total_collected_cents →
  x = 25 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_child_ticket_l2202_220275


namespace NUMINAMATH_GPT_ellipse_abs_sum_max_min_l2202_220206

theorem ellipse_abs_sum_max_min (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) :
  2 ≤ |x| + |y| ∧ |x| + |y| ≤ 3 :=
sorry

end NUMINAMATH_GPT_ellipse_abs_sum_max_min_l2202_220206


namespace NUMINAMATH_GPT_range_of_a_l2202_220278

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x < 2) : 
  (a ∈ Set.Ioo (Real.sqrt 2 / 2) 1 ∨ a ∈ Set.Ioo 1 (Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2202_220278


namespace NUMINAMATH_GPT_good_function_count_l2202_220211

noncomputable def num_good_functions (n : ℕ) : ℕ :=
  if n < 2 then 0 else
    n * Nat.totient n

theorem good_function_count (n : ℕ) (h : n ≥ 2) :
  ∃ (f : ℤ → Fin (n + 1)), 
  (∀ k, 1 ≤ k ∧ k ≤ n - 1 → ∃ j, ∀ m, (f (m + j) : ℤ) ≡ (f (m + k) - f m : ℤ) [ZMOD (n + 1)]) → 
  num_good_functions n = n * Nat.totient n :=
sorry

end NUMINAMATH_GPT_good_function_count_l2202_220211


namespace NUMINAMATH_GPT_friends_attended_reception_l2202_220246

-- Definition of the given conditions
def total_guests : ℕ := 180
def couples_per_side : ℕ := 20

-- Statement based on the given problem
theorem friends_attended_reception : 
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  let friends := total_guests - family_guests
  friends = 100 :=
by
  -- We define the family_guests calculation
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  -- We define the friends calculation
  let friends := total_guests - family_guests
  -- We state the conclusion
  show friends = 100
  sorry

end NUMINAMATH_GPT_friends_attended_reception_l2202_220246


namespace NUMINAMATH_GPT_parametric_plane_equation_l2202_220297

-- Definitions to translate conditions
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ := (2 + 2 * s - t, 4 - 2 * s, 6 + s - 3 * t)

-- Theorem to prove the equivalence to plane equation
theorem parametric_plane_equation : 
  ∃ A B C D, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧ 
  (∀ s t x y z, parametric_plane s t = (x, y, z) → 6 * x - 5 * y - 2 * z + 20 = 0) := by
  sorry

end NUMINAMATH_GPT_parametric_plane_equation_l2202_220297


namespace NUMINAMATH_GPT_right_triangle_roots_l2202_220248

theorem right_triangle_roots (α β : ℝ) (k : ℕ) (h_triangle : (α^2 + β^2 = 100) ∧ (α + β = 14) ∧ (α * β = 4 * k - 4)) : k = 13 :=
sorry

end NUMINAMATH_GPT_right_triangle_roots_l2202_220248


namespace NUMINAMATH_GPT_domain_of_log_function_l2202_220204

theorem domain_of_log_function :
  {x : ℝ | x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log_function_l2202_220204


namespace NUMINAMATH_GPT_correct_rounded_result_l2202_220222

def round_to_nearest_ten (n : ℤ) : ℤ :=
  (n + 5) / 10 * 10

theorem correct_rounded_result :
  round_to_nearest_ten ((57 + 68) * 2) = 250 :=
by
  sorry

end NUMINAMATH_GPT_correct_rounded_result_l2202_220222


namespace NUMINAMATH_GPT_exists_polynomial_triangle_property_l2202_220203

noncomputable def f (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem exists_polynomial_triangle_property :
  ∀ (x y z : ℝ), (f x y z > 0 ↔ (|x| + |y| > |z| ∧ |y| + |z| > |x| ∧ |z| + |x| > |y|)) :=
sorry

end NUMINAMATH_GPT_exists_polynomial_triangle_property_l2202_220203


namespace NUMINAMATH_GPT_second_solution_lemonade_is_45_l2202_220266

-- Define percentages as real numbers for simplicity
def firstCarbonatedWater : ℝ := 0.80
def firstLemonade : ℝ := 0.20
def secondCarbonatedWater : ℝ := 0.55
def mixturePercentageFirst : ℝ := 0.50
def mixtureCarbonatedWater : ℝ := 0.675

-- The ones that already follow from conditions or trivial definitions:
def secondLemonade : ℝ := 1 - secondCarbonatedWater

-- Define the percentage of carbonated water in mixture, based on given conditions
def mixtureIsCorrect : Prop :=
  mixturePercentageFirst * firstCarbonatedWater + (1 - mixturePercentageFirst) * secondCarbonatedWater = mixtureCarbonatedWater

-- The theorem to prove: second solution's lemonade percentage is 45%
theorem second_solution_lemonade_is_45 :
  mixtureIsCorrect → secondLemonade = 0.45 :=
by
  sorry

end NUMINAMATH_GPT_second_solution_lemonade_is_45_l2202_220266


namespace NUMINAMATH_GPT_jina_total_mascots_l2202_220277

-- Definitions and Conditions
def num_teddies := 5
def num_bunnies := 3 * num_teddies
def num_koala_bears := 1
def additional_teddies := 2 * num_bunnies

-- Total mascots calculation
def total_mascots := num_teddies + num_bunnies + num_koala_bears + additional_teddies

theorem jina_total_mascots : total_mascots = 51 := by
  sorry

end NUMINAMATH_GPT_jina_total_mascots_l2202_220277


namespace NUMINAMATH_GPT_no_real_solution_l2202_220223

theorem no_real_solution (x y : ℝ) (hx : x^2 = 1 + 1 / y^2) (hy : y^2 = 1 + 1 / x^2) : false :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_l2202_220223


namespace NUMINAMATH_GPT_thomas_saves_40_per_month_l2202_220269

variables (T J : ℝ) (months : ℝ := 72) 

theorem thomas_saves_40_per_month 
  (h1 : J = (3/5) * T)
  (h2 : 72 * T + 72 * J = 4608) : 
  T = 40 :=
by sorry

end NUMINAMATH_GPT_thomas_saves_40_per_month_l2202_220269


namespace NUMINAMATH_GPT_students_got_off_l2202_220257

-- Define the number of students originally on the bus
def original_students : ℕ := 10

-- Define the number of students left on the bus after the first stop
def students_left : ℕ := 7

-- Prove that the number of students who got off the bus at the first stop is 3
theorem students_got_off : original_students - students_left = 3 :=
by
  sorry

end NUMINAMATH_GPT_students_got_off_l2202_220257


namespace NUMINAMATH_GPT_mork_tax_rate_l2202_220267

theorem mork_tax_rate (M R : ℝ) (h1 : 0.15 = 0.15) (h2 : 4 * M = Mindy_income) (h3 : (R / 100 * M + 0.15 * 4 * M) = 0.21 * 5 * M):
  R = 45 :=
sorry

end NUMINAMATH_GPT_mork_tax_rate_l2202_220267


namespace NUMINAMATH_GPT_marble_counts_l2202_220268

theorem marble_counts (A B C : ℕ) : 
  (∃ x : ℕ, 
    A = 165 ∧ 
    B = 57 ∧ 
    C = 21 ∧ 
    (A = 55 * x / 27) ∧ 
    (B = 19 * x / 27) ∧ 
    (C = 7 * x / 27) ∧ 
    (7 * x / 9 = x / 9 + 54) ∧ 
    (A + B + C) = 3 * x
  ) :=
sorry

end NUMINAMATH_GPT_marble_counts_l2202_220268


namespace NUMINAMATH_GPT_monotonic_increase_range_of_alpha_l2202_220205

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.cos (ω * x)

theorem monotonic_increase_range_of_alpha
  (ω : ℝ) (hω : ω > 0)
  (zeros_form_ap : ∀ k : ℤ, ∃ x₀ : ℝ, f ω x₀ = 0 ∧ ∀ n : ℤ, f ω (x₀ + n * (π / 2)) = 0) :
  ∃ α : ℝ, 0 < α ∧ α < 5 * π / 12 ∧ ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ α → f ω x ≤ f ω y :=
sorry

end NUMINAMATH_GPT_monotonic_increase_range_of_alpha_l2202_220205


namespace NUMINAMATH_GPT_train_additional_time_l2202_220243

theorem train_additional_time
  (t : ℝ)  -- time the car takes to reach station B
  (x : ℝ)  -- additional time the train takes compared to the car
  (h₁ : t = 4.5)  -- car takes 4.5 hours to reach station B
  (h₂ : t + (t + x) = 11)  -- combined time for both the car and the train to reach station B
  : x = 2 :=
sorry

end NUMINAMATH_GPT_train_additional_time_l2202_220243


namespace NUMINAMATH_GPT_proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l2202_220259

noncomputable def prob_boy_pass_all_rounds : ℚ :=
  (5/6) * (4/5) * (3/4) * (2/3)

noncomputable def prob_girl_pass_all_rounds : ℚ :=
  (4/5) * (3/4) * (2/3) * (1/2)

def prob_xi_distribution : (ℚ × ℚ × ℚ × ℚ × ℚ) :=
  (64/225, 96/225, 52/225, 12/225, 1/225)

def exp_xi : ℚ :=
  (0 * (64/225) + 1 * (96/225) + 2 * (52/225) + 3 * (12/225) + 4 * (1/225))

theorem proof_prob_boy_pass_all_rounds :
  prob_boy_pass_all_rounds = 1/3 :=
by
  sorry

theorem proof_prob_girl_pass_all_rounds :
  prob_girl_pass_all_rounds = 1/5 :=
by
  sorry

theorem proof_xi_distribution :
  prob_xi_distribution = (64/225, 96/225, 52/225, 12/225, 1/225) :=
by
  sorry

theorem proof_exp_xi :
  exp_xi = 16/15 :=
by
  sorry

end NUMINAMATH_GPT_proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l2202_220259


namespace NUMINAMATH_GPT_periodic_even_function_value_l2202_220239

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x - a)

-- Conditions: 
-- 1. f(x) is even 
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- 2. f(x) is periodic with period 6
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

-- Main theorem
theorem periodic_even_function_value 
  (a : ℝ) 
  (f_def : ∀ x, -3 ≤ x ∧ x ≤ 3 → f x a = (x + 1) * (x - a))
  (h_even : is_even_function (f · a))
  (h_periodic : is_periodic_function (f · a) 6) : 
  f (-6) a = -1 := 
sorry

end NUMINAMATH_GPT_periodic_even_function_value_l2202_220239


namespace NUMINAMATH_GPT_not_perfect_square_7p_3p_4_l2202_220227

theorem not_perfect_square_7p_3p_4 (p : ℕ) (hp : Nat.Prime p) : ¬∃ a : ℕ, a^2 = 7 * p + 3^p - 4 := 
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_7p_3p_4_l2202_220227


namespace NUMINAMATH_GPT_minimum_value_of_sum_2_l2202_220249

noncomputable def minimum_value_of_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) : 
  Prop := 
  x + y = 2

theorem minimum_value_of_sum_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) :
  minimum_value_of_sum x y hx hy inequality := 
sorry

end NUMINAMATH_GPT_minimum_value_of_sum_2_l2202_220249


namespace NUMINAMATH_GPT_weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l2202_220216

noncomputable def cost_price : ℝ := 10
noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def w (x : ℝ) : ℝ := -10 * x ^ 2 + 500 * x - 4000

-- Proof Step 1: Show the functional relationship between w and x
theorem weekly_profit_function : ∀ x : ℝ, w x = -10 * x ^ 2 + 500 * x - 4000 := by
  intro x
  -- This is the function definition provided, proof omitted
  sorry

-- Proof Step 2: Find the selling price x that maximizes weekly profit
theorem maximize_weekly_profit : ∃ x : ℝ, x = 25 ∧ (∀ y : ℝ, y ≠ x → w y ≤ w x) := by
  use 25
  -- The details of solving the optimization are omitted
  sorry

-- Proof Step 3: Given weekly profit w = 2000 and constraints on y, find the weekly sales quantity
theorem weekly_sales_quantity (x : ℝ) (H : w x = 2000 ∧ y x ≥ 180) : y x = 200 := by
  have Hy : y x = -10 * x + 400 := by rfl
  have Hconstraint : y x ≥ 180 := H.2
  have Hprofit : w x = 2000 := H.1
  -- The details of solving for x and ensuring constraints are omitted
  sorry

end NUMINAMATH_GPT_weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l2202_220216


namespace NUMINAMATH_GPT_total_apple_weight_proof_l2202_220283

-- Define the weights of each fruit in terms of ounces
def weight_apple : ℕ := 4
def weight_orange : ℕ := 3
def weight_plum : ℕ := 2

-- Define the bag's capacity and the number of bags
def bag_capacity : ℕ := 49
def number_of_bags : ℕ := 5

-- Define the least common multiple (LCM) of the weights
def lcm_weight : ℕ := Nat.lcm weight_apple (Nat.lcm weight_orange weight_plum)

-- Define the largest multiple of LCM that is less than or equal to the bag's capacity
def max_lcm_multiple : ℕ := (bag_capacity / lcm_weight) * lcm_weight

-- Determine the number of each fruit per bag
def sets_per_bag : ℕ := max_lcm_multiple / lcm_weight
def apples_per_bag : ℕ := sets_per_bag * 1  -- 1 apple per set

-- Calculate the weight of apples per bag and total needed in all bags
def apple_weight_per_bag : ℕ := apples_per_bag * weight_apple
def total_apple_weight : ℕ := apple_weight_per_bag * number_of_bags

-- The statement to be proved in Lean
theorem total_apple_weight_proof : total_apple_weight = 80 := by
  sorry

end NUMINAMATH_GPT_total_apple_weight_proof_l2202_220283


namespace NUMINAMATH_GPT_smaller_number_l2202_220242

theorem smaller_number (x y : ℤ) (h1 : x + y = 79) (h2 : x - y = 15) : y = 32 := by
  sorry

end NUMINAMATH_GPT_smaller_number_l2202_220242


namespace NUMINAMATH_GPT_number_of_ostriches_l2202_220280

theorem number_of_ostriches
    (x y : ℕ)
    (h1 : x + y = 150)
    (h2 : 2 * x + 6 * y = 624) :
    x = 69 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_ostriches_l2202_220280


namespace NUMINAMATH_GPT_smallest_possible_n_l2202_220289

theorem smallest_possible_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n > 20) : n = 52 := 
sorry

end NUMINAMATH_GPT_smallest_possible_n_l2202_220289


namespace NUMINAMATH_GPT_JodiMilesFourthWeek_l2202_220255

def JodiMilesFirstWeek := 1 * 6
def JodiMilesSecondWeek := 2 * 6
def JodiMilesThirdWeek := 3 * 6
def TotalMilesFirstThreeWeeks := JodiMilesFirstWeek + JodiMilesSecondWeek + JodiMilesThirdWeek
def TotalMilesFourWeeks := 60

def MilesInFourthWeek := TotalMilesFourWeeks - TotalMilesFirstThreeWeeks
def DaysInWeek := 6

theorem JodiMilesFourthWeek : (MilesInFourthWeek / DaysInWeek) = 4 := by
  sorry

end NUMINAMATH_GPT_JodiMilesFourthWeek_l2202_220255


namespace NUMINAMATH_GPT_equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l2202_220230

theorem equation1_solutions (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

theorem equation2_solutions (x : ℝ) : x * (3 * x + 1) = 2 * (3 * x + 1) ↔ (x = -1 / 3 ∨ x = 2) :=
by sorry

theorem equation3_solutions (x : ℝ) : 2 * x^2 + x - 4 = 0 ↔ (x = (-1 + Real.sqrt 33) / 4 ∨ x = (-1 - Real.sqrt 33) / 4) :=
by sorry

theorem equation4_no_real_solutions (x : ℝ) : ¬ ∃ x, 4 * x^2 - 3 * x + 1 = 0 :=
by sorry

end NUMINAMATH_GPT_equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l2202_220230


namespace NUMINAMATH_GPT_complement_A_l2202_220285

open Set

variable (A : Set ℝ) (x : ℝ)
def A_def : Set ℝ := { x | x ≥ 1 }

theorem complement_A : Aᶜ = { y | y < 1 } :=
by
  sorry

end NUMINAMATH_GPT_complement_A_l2202_220285


namespace NUMINAMATH_GPT_mary_groceries_fitting_l2202_220219

theorem mary_groceries_fitting :
  (∀ bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice,
    bags = 2 →
    wt_green = 4 →
    wt_milk = 6 →
    wt_carrots = 2 * wt_green →
    wt_apples = 3 →
    wt_bread = 1 →
    wt_rice = 5 →
    (wt_green + wt_milk + wt_carrots + wt_apples + wt_bread + wt_rice = 27) →
    (∀ b, b < 20 →
      (b = 6 + 5 ∨ b = 22 - 11) →
      (20 - b = 9))) :=
by
  intros bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice h_bags h_green h_milk h_carrots h_apples h_bread h_rice h_total h_b
  sorry

end NUMINAMATH_GPT_mary_groceries_fitting_l2202_220219


namespace NUMINAMATH_GPT_staircase_steps_eq_twelve_l2202_220263

theorem staircase_steps_eq_twelve (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → (n = 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_staircase_steps_eq_twelve_l2202_220263


namespace NUMINAMATH_GPT_pages_per_brochure_l2202_220252

-- Define the conditions
def single_page_spreads := 20
def double_page_spreads := 2 * single_page_spreads
def pages_per_double_spread := 2
def pages_from_single := single_page_spreads
def pages_from_double := double_page_spreads * pages_per_double_spread
def total_pages_from_spreads := pages_from_single + pages_from_double
def ads_per_4_pages := total_pages_from_spreads / 4
def total_ads_pages := ads_per_4_pages
def total_pages := total_pages_from_spreads + total_ads_pages
def brochures := 25

-- The theorem we want to prove
theorem pages_per_brochure : total_pages / brochures = 5 :=
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_pages_per_brochure_l2202_220252


namespace NUMINAMATH_GPT_ceil_floor_eq_zero_implies_sum_l2202_220291

theorem ceil_floor_eq_zero_implies_sum (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ + ⌊x⌋ = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_ceil_floor_eq_zero_implies_sum_l2202_220291


namespace NUMINAMATH_GPT_B_share_after_tax_l2202_220228

noncomputable def B_share (x : ℝ) : ℝ := 3 * x
noncomputable def salary_proportion (A B C D : ℝ) (x : ℝ) :=
  A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ D = 6 * x
noncomputable def D_more_than_C (D C : ℝ) : Prop :=
  D - C = 700
noncomputable def meets_minimum_wage (B : ℝ) : Prop :=
  B ≥ 1000
noncomputable def tax_deduction (B : ℝ) : ℝ :=
  if B > 1500 then B - 0.15 * (B - 1500) else B

theorem B_share_after_tax (A B C D : ℝ) (x : ℝ) (h1 : salary_proportion A B C D x)
  (h2 : D_more_than_C D C) (h3 : meets_minimum_wage B) :
  tax_deduction B = 1050 :=
by
  sorry

end NUMINAMATH_GPT_B_share_after_tax_l2202_220228


namespace NUMINAMATH_GPT_x_plus_y_value_l2202_220253

def sum_evens_40_to_60 : ℕ :=
  (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)

def num_evens_40_to_60 : ℕ := 11

theorem x_plus_y_value : sum_evens_40_to_60 + num_evens_40_to_60 = 561 := by
  sorry

end NUMINAMATH_GPT_x_plus_y_value_l2202_220253


namespace NUMINAMATH_GPT_problem_l2202_220236

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end NUMINAMATH_GPT_problem_l2202_220236


namespace NUMINAMATH_GPT_problem_statement_l2202_220221

theorem problem_statement : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3) ^ 2 = 4 - 2 * Real.sqrt 6 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2202_220221


namespace NUMINAMATH_GPT_average_of_distinct_numbers_l2202_220282

theorem average_of_distinct_numbers (A B C D : ℕ) (hA : A = 1 ∨ A = 3 ∨ A = 5 ∨ A = 7)
                                   (hB : B = 1 ∨ B = 3 ∨ B = 5 ∨ B = 7)
                                   (hC : C = 1 ∨ C = 3 ∨ C = 5 ∨ C = 7)
                                   (hD : D = 1 ∨ D = 3 ∨ D = 5 ∨ D = 7)
                                   (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
    (A + B + C + D) / 4 = 4 := by
  sorry

end NUMINAMATH_GPT_average_of_distinct_numbers_l2202_220282


namespace NUMINAMATH_GPT_range_of_a_l2202_220265

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ (2 ≤ y ∧ y ≤ 3) → x * y ≤ a * x^2 + 2 * y^2) → a ≥ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2202_220265


namespace NUMINAMATH_GPT_integer_powers_of_reciprocal_sum_l2202_220232

variable (x: ℝ)

theorem integer_powers_of_reciprocal_sum (hx : x ≠ 0) (hx_int : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ k : ℤ, x^n + 1/x^n = k :=
by
  sorry

end NUMINAMATH_GPT_integer_powers_of_reciprocal_sum_l2202_220232


namespace NUMINAMATH_GPT_money_taken_l2202_220294

def total_people : ℕ := 6
def cost_per_soda : ℝ := 0.5
def cost_per_pizza : ℝ := 1.0

theorem money_taken (total_people cost_per_soda cost_per_pizza : ℕ × ℝ × ℝ ) :
  total_people * cost_per_soda + total_people * cost_per_pizza = 9 := by
  sorry

end NUMINAMATH_GPT_money_taken_l2202_220294


namespace NUMINAMATH_GPT_quadratic_expression_result_l2202_220274

theorem quadratic_expression_result (x y : ℚ) 
  (h1 : 4 * x + y = 11) 
  (h2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_expression_result_l2202_220274


namespace NUMINAMATH_GPT_number_of_students_l2202_220224

variables (T S n : ℕ)

-- 1. The teacher's age is 24 years more than the average age of the students.
def condition1 : Prop := T = S / n + 24

-- 2. The teacher's age is 20 years more than the average age of everyone present.
def condition2 : Prop := T = (T + S) / (n + 1) + 20

-- Proving that the number of students in the classroom is 5 given the conditions.
theorem number_of_students (h1 : condition1 T S n) (h2 : condition2 T S n) : n = 5 :=
by sorry

end NUMINAMATH_GPT_number_of_students_l2202_220224


namespace NUMINAMATH_GPT_sum_of_coordinates_of_C_and_D_l2202_220201

structure Point where
  x : ℤ
  y : ℤ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def sum_coordinates (p1 p2 : Point) : ℤ :=
  p1.x + p1.y + p2.x + p2.y

def C : Point := { x := 3, y := -2 }
def D : Point := reflect_y C

theorem sum_of_coordinates_of_C_and_D : sum_coordinates C D = -4 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_C_and_D_l2202_220201


namespace NUMINAMATH_GPT_squirrel_climb_l2202_220264

-- Define the problem conditions and the goal
variable (x : ℝ)

-- net_distance_climbed_every_two_minutes
def net_distance_climbed_every_two_minutes : ℝ := x - 2

-- distance_climbed_in_14_minutes
def distance_climbed_in_14_minutes : ℝ := 7 * (x - 2)

-- distance_climbed_in_15th_minute
def distance_climbed_in_15th_minute : ℝ := x

-- total_distance_climbed_in_15_minutes
def total_distance_climbed_in_15_minutes : ℝ := 26

-- Theorem: proving x based on the conditions
theorem squirrel_climb : 
  7 * (x - 2) + x = 26 -> x = 5 := by
  intros h
  sorry

end NUMINAMATH_GPT_squirrel_climb_l2202_220264


namespace NUMINAMATH_GPT_calculateRequiredMonthlyRent_l2202_220225

noncomputable def requiredMonthlyRent (purchase_price : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (repair_percentage : ℝ) : ℝ :=
  let annual_return := annual_return_rate * purchase_price
  let total_annual_need := annual_return + annual_taxes
  let monthly_requirement := total_annual_need / 12
  let monthly_rent := monthly_requirement / (1 - repair_percentage)
  monthly_rent

theorem calculateRequiredMonthlyRent : requiredMonthlyRent 20000 0.06 450 0.10 = 152.78 := by
  sorry

end NUMINAMATH_GPT_calculateRequiredMonthlyRent_l2202_220225


namespace NUMINAMATH_GPT_directrix_parabola_l2202_220261

theorem directrix_parabola (p : ℝ) (h : 4 * p = 2) : 
  ∃ d : ℝ, d = -p / 2 ∧ d = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_directrix_parabola_l2202_220261


namespace NUMINAMATH_GPT_find_BD_l2202_220284

theorem find_BD 
  (A B C D : Type)
  (AC BC : ℝ) (h₁ : AC = 10) (h₂ : BC = 10)
  (AD CD : ℝ) (h₃ : AD = 12) (h₄ : CD = 5) :
  ∃ (BD : ℝ), BD = 152 / 24 := 
sorry

end NUMINAMATH_GPT_find_BD_l2202_220284


namespace NUMINAMATH_GPT_number_of_players_taking_mathematics_l2202_220276

def total_players : ℕ := 25
def players_taking_physics : ℕ := 12
def players_taking_both : ℕ := 5

theorem number_of_players_taking_mathematics :
  total_players - players_taking_physics + players_taking_both = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_players_taking_mathematics_l2202_220276


namespace NUMINAMATH_GPT_range_of_m_l2202_220208

noncomputable def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2 * x - 8 < 0}
def U : Set ℝ := Set.univ
def CU_M (m : ℝ) : Set ℝ := {x | x < -m}
def empty_intersection (m : ℝ) : Prop := (CU_M m ∩ N = ∅)

theorem range_of_m (m : ℝ) : empty_intersection m → m ≥ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2202_220208


namespace NUMINAMATH_GPT_largest_possible_sum_l2202_220287

-- Define whole numbers
def whole_numbers : Set ℕ := Set.univ

-- Define the given conditions
variables (a b : ℕ)
axiom h1 : a ∈ whole_numbers
axiom h2 : b ∈ whole_numbers
axiom h3 : a * b = 48

-- Prove the largest sum condition
theorem largest_possible_sum : a + b ≤ 49 :=
sorry

end NUMINAMATH_GPT_largest_possible_sum_l2202_220287


namespace NUMINAMATH_GPT_remainder_of_division_987543_12_l2202_220238

theorem remainder_of_division_987543_12 : 987543 % 12 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_division_987543_12_l2202_220238


namespace NUMINAMATH_GPT_probability_eight_distinct_numbers_l2202_220237

theorem probability_eight_distinct_numbers :
  let total_ways := 10^8
  let ways_distinct := (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3)
  (ways_distinct / total_ways : ℚ) = 18144 / 500000 := 
by
  sorry

end NUMINAMATH_GPT_probability_eight_distinct_numbers_l2202_220237


namespace NUMINAMATH_GPT_james_has_43_oreos_l2202_220200

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end NUMINAMATH_GPT_james_has_43_oreos_l2202_220200


namespace NUMINAMATH_GPT_rectangular_to_polar_coordinates_l2202_220210

noncomputable def polar_coordinates_of_point (x y : ℝ) : ℝ × ℝ := sorry

theorem rectangular_to_polar_coordinates :
  polar_coordinates_of_point 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := sorry

end NUMINAMATH_GPT_rectangular_to_polar_coordinates_l2202_220210


namespace NUMINAMATH_GPT_walk_to_bus_stop_time_l2202_220250

theorem walk_to_bus_stop_time 
  (S T : ℝ)   -- Usual speed and time
  (D : ℝ)        -- Distance to bus stop
  (T'_delay : ℝ := 9)   -- Additional delay in minutes
  (T_coffee : ℝ := 6)   -- Coffee shop time in minutes
  (reduced_speed_factor : ℝ := 4/5)  -- Reduced speed factor
  (h1 : D = S * T)
  (h2 : D = reduced_speed_factor * S * (T + T'_delay - T_coffee)) :
  T = 12 :=
by
  sorry

end NUMINAMATH_GPT_walk_to_bus_stop_time_l2202_220250


namespace NUMINAMATH_GPT_base5_first_digit_of_1024_l2202_220245

theorem base5_first_digit_of_1024: 
  ∀ (d : ℕ), (d * 5^4 ≤ 1024) ∧ (1024 < (d+1) * 5^4) → d = 1 :=
by
  sorry

end NUMINAMATH_GPT_base5_first_digit_of_1024_l2202_220245


namespace NUMINAMATH_GPT_complex_fraction_evaluation_l2202_220292

open Complex

theorem complex_fraction_evaluation (c d : ℂ) (hz : c ≠ 0) (hz' : d ≠ 0) (h : c^2 + c * d + d^2 = 0) :
  (c^12 + d^12) / (c^3 + d^3)^4 = 1 / 8 := 
by sorry

end NUMINAMATH_GPT_complex_fraction_evaluation_l2202_220292


namespace NUMINAMATH_GPT_total_cost_of_car_rental_l2202_220247

theorem total_cost_of_car_rental :
  ∀ (rental_cost_per_day mileage_cost_per_mile : ℝ) (days rented : ℕ) (miles_driven : ℕ),
  rental_cost_per_day = 30 →
  mileage_cost_per_mile = 0.25 →
  rented = 5 →
  miles_driven = 500 →
  rental_cost_per_day * rented + mileage_cost_per_mile * miles_driven = 275 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_car_rental_l2202_220247


namespace NUMINAMATH_GPT_find_A_and_height_l2202_220262

noncomputable def triangle_properties (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ) :=
  a = 7 ∧ b = 8 ∧ cos_B = -1 / 7 ∧ 
  h = (a : ℝ) * (Real.sqrt (1 - (cos_B)^2)) * (1 : ℝ) / b / 2

theorem find_A_and_height : 
  ∀ (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ), 
  triangle_properties a b B cos_B h → 
  ∃ A h1, A = Real.pi / 3 ∧ h1 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_A_and_height_l2202_220262


namespace NUMINAMATH_GPT_polynomial_roots_l2202_220293

theorem polynomial_roots (α : ℝ) : 
  (α^2 + α - 1 = 0) → (α^3 - 2 * α + 1 = 0) :=
by sorry

end NUMINAMATH_GPT_polynomial_roots_l2202_220293


namespace NUMINAMATH_GPT_unique_solution_l2202_220270

noncomputable def f : ℝ → ℝ :=
sorry

theorem unique_solution (x : ℝ) (hx : 0 ≤ x) : 
  (f : ℝ → ℝ) (2 * x + 1) = 3 * (f x) + 5 ↔ f x = -5 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_l2202_220270


namespace NUMINAMATH_GPT_calc_expression_l2202_220202

theorem calc_expression : (900^2) / (264^2 - 256^2) = 194.711 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l2202_220202


namespace NUMINAMATH_GPT_spinner_probability_C_l2202_220254

theorem spinner_probability_C 
  (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
  (hA : P_A = 1/3)
  (hB : P_B = 1/4)
  (hD : P_D = 1/6)
  (hSum : P_A + P_B + P_C + P_D = 1) :
  P_C = 1 / 4 := 
sorry

end NUMINAMATH_GPT_spinner_probability_C_l2202_220254


namespace NUMINAMATH_GPT_mean_temperature_correct_l2202_220241

-- Define the condition (temperatures)
def temperatures : List Int :=
  [-6, -3, -3, -4, 2, 4, 1]

-- Define the total number of days
def num_days : ℕ := 7

-- Define the expected mean temperature
def expected_mean : Rat := (-6 : Int) / (7 : Int)

-- State the theorem that we need to prove
theorem mean_temperature_correct :
  (temperatures.sum : Rat) / (num_days : Rat) = expected_mean := 
by
  sorry

end NUMINAMATH_GPT_mean_temperature_correct_l2202_220241


namespace NUMINAMATH_GPT_length_AD_l2202_220299

theorem length_AD (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) (h3 : x * (13 - x) = 36) : x = 4 ∨ x = 9 :=
by sorry

end NUMINAMATH_GPT_length_AD_l2202_220299


namespace NUMINAMATH_GPT_n_is_prime_or_power_of_2_l2202_220231

noncomputable def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

noncomputable def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 ^ k

noncomputable def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem n_is_prime_or_power_of_2 {n : ℕ} (h1 : n > 6)
  (h2 : ∃ (a : ℕ → ℕ) (k : ℕ), 
    (∀ i : ℕ, i < k → a i < n ∧ coprime (a i) n) ∧ 
    (∀ i : ℕ, 1 ≤ i → i < k → a (i + 1) - a i = a 2 - a 1)) 
  : is_prime n ∨ is_power_of_2 n := 
sorry

end NUMINAMATH_GPT_n_is_prime_or_power_of_2_l2202_220231


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l2202_220256

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : (S 9) / 9 - (S 5) / 5 = 4)
  (hSn : ∀ n, S n = n * (2 + (n - 1) / 2 * (a 2 - a 1) )) : 
  a 10 = 20 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l2202_220256


namespace NUMINAMATH_GPT_proof_F_2_f_3_l2202_220217

def f (a : ℕ) : ℕ := a ^ 2 - 1

def F (a : ℕ) (b : ℕ) : ℕ := 3 * b ^ 2 + 2 * a

theorem proof_F_2_f_3 : F 2 (f 3) = 196 := by
  have h1 : f 3 = 3 ^ 2 - 1 := rfl
  rw [h1]
  have h2 : 3 ^ 2 - 1 = 8 := by norm_num
  rw [h2]
  exact rfl

end NUMINAMATH_GPT_proof_F_2_f_3_l2202_220217


namespace NUMINAMATH_GPT_coin_problem_l2202_220220

theorem coin_problem (n d q : ℕ) 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 25 * q = 410)
  (h3 : d = n + 4) : q - n = 2 :=
by
  sorry

end NUMINAMATH_GPT_coin_problem_l2202_220220


namespace NUMINAMATH_GPT_intersect_at_two_points_l2202_220215

theorem intersect_at_two_points (a : ℝ) :
  (∃ p q : ℝ × ℝ, 
    (p.1 - p.2 + 1 = 0) ∧ (2 * p.1 + p.2 - 4 = 0) ∧ (a * p.1 - p.2 + 2 = 0) ∧
    (q.1 - q.2 + 1 = 0) ∧ (2 * q.1 + q.2 - 4 = 0) ∧ (a * q.1 - q.2 + 2 = 0) ∧ p ≠ q) →
  (a = 1 ∨ a = -2) :=
by 
  sorry

end NUMINAMATH_GPT_intersect_at_two_points_l2202_220215


namespace NUMINAMATH_GPT_simplify_fraction_l2202_220244

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2202_220244


namespace NUMINAMATH_GPT_moving_circle_trajectory_l2202_220298

-- Define the two given circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- The theorem statement
theorem moving_circle_trajectory :
  (∀ x y : ℝ, (exists r : ℝ, r > 0 ∧ ∃ M : ℝ × ℝ, 
  (C₁ M.1 M.2 ∧ ((M.1 - 4)^2 + M.2^2 = (13 - r)^2) ∧
  C₂ M.1 M.2 ∧ ((M.1 + 4)^2 + M.2^2 = (r + 3)^2)) ∧
  ((x = M.1) ∧ (y = M.2))) ↔ (x^2 / 64 + y^2 / 48 = 1)) := sorry

end NUMINAMATH_GPT_moving_circle_trajectory_l2202_220298


namespace NUMINAMATH_GPT_motel_total_rent_l2202_220209

theorem motel_total_rent (R40 R60 : ℕ) (total_rent : ℕ) 
  (h1 : total_rent = 40 * R40 + 60 * R60) 
  (h2 : 40 * (R40 + 10) + 60 * (R60 - 10) = total_rent - total_rent / 10) 
  (h3 : total_rent / 10 = 200) : 
  total_rent = 2000 := 
sorry

end NUMINAMATH_GPT_motel_total_rent_l2202_220209


namespace NUMINAMATH_GPT_flower_garden_width_l2202_220226

-- Define the conditions
def gardenArea : ℝ := 143.2
def gardenLength : ℝ := 4
def gardenWidth : ℝ := 35.8

-- The proof statement (question to answer)
theorem flower_garden_width :
    gardenWidth = gardenArea / gardenLength :=
by 
  sorry

end NUMINAMATH_GPT_flower_garden_width_l2202_220226


namespace NUMINAMATH_GPT_james_worked_41_hours_l2202_220234

theorem james_worked_41_hours (x : ℝ) :
  ∃ (J : ℕ), 
    (24 * x + 12 * 1.5 * x = 40 * x + (J - 40) * 2 * x) ∧ 
    J = 41 := 
by 
  sorry

end NUMINAMATH_GPT_james_worked_41_hours_l2202_220234


namespace NUMINAMATH_GPT_number_times_half_squared_eq_eight_l2202_220240

theorem number_times_half_squared_eq_eight : 
  ∃ n : ℝ, n * (1/2)^2 = 2^3 := 
sorry

end NUMINAMATH_GPT_number_times_half_squared_eq_eight_l2202_220240


namespace NUMINAMATH_GPT_probability_two_white_balls_l2202_220218

-- Definitions based on the conditions provided
def total_balls := 17        -- 8 white + 9 black
def white_balls := 8
def drawn_without_replacement := true

-- Proposition: Probability of drawing two white balls successively
theorem probability_two_white_balls:
  drawn_without_replacement → 
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 7 / 34 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_two_white_balls_l2202_220218


namespace NUMINAMATH_GPT_alice_spent_19_percent_l2202_220290

variable (A : ℝ) (x : ℝ)
variable (h1 : ∃ (B : ℝ), B = 0.9 * A) -- Bob's initial amount in terms of Alice's initial amount
variable (h2 : A - x = 0.81 * A) -- Alice's remaining amount after spending x

theorem alice_spent_19_percent (h1 : ∃ (B : ℝ), B = 0.9 * A) (h2 : A - x = 0.81 * A) : (x / A) * 100 = 19 := by
  sorry

end NUMINAMATH_GPT_alice_spent_19_percent_l2202_220290
