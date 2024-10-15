import Mathlib

namespace NUMINAMATH_GPT_sum_of_a_b_c_l136_13633

theorem sum_of_a_b_c (a b c : ℝ) (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_sum_of_a_b_c_l136_13633


namespace NUMINAMATH_GPT_average_score_girls_cedar_drake_l136_13680

theorem average_score_girls_cedar_drake
  (C c D d : ℕ)
  (cedar_boys_score cedar_girls_score cedar_combined_score
   drake_boys_score drake_girls_score drake_combined_score combined_boys_score : ℝ)
  (h1 : cedar_boys_score = 68)
  (h2 : cedar_girls_score = 80)
  (h3 : cedar_combined_score = 73)
  (h4 : drake_boys_score = 75)
  (h5 : drake_girls_score = 88)
  (h6 : drake_combined_score = 83)
  (h7 : combined_boys_score = 74)
  (h8 : (68 * C + 80 * c) / (C + c) = 73)
  (h9 : (75 * D + 88 * d) / (D + d) = 83)
  (h10 : (68 * C + 75 * D) / (C + D) = 74) :
  (80 * c + 88 * d) / (c + d) = 87 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_average_score_girls_cedar_drake_l136_13680


namespace NUMINAMATH_GPT_problem1_problem2_l136_13621

-- Problem 1
theorem problem1 (x : ℝ) (h1 : 2 * x > 1 - x) (h2 : x + 2 < 4 * x - 1) : x > 1 := 
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ)
  (h1 : (2 / 3) * x + 5 > 1 - x)
  (h2 : x - 1 ≤ (3 / 4) * x - 1 / 8) :
  -12 / 5 < x ∧ x ≤ 7 / 2 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l136_13621


namespace NUMINAMATH_GPT_total_beads_sue_necklace_l136_13643

theorem total_beads_sue_necklace (purple blue green : ℕ) (h1 : purple = 7)
  (h2 : blue = 2 * purple) (h3 : green = blue + 11) : 
  purple + blue + green = 46 := 
by 
  sorry

end NUMINAMATH_GPT_total_beads_sue_necklace_l136_13643


namespace NUMINAMATH_GPT_original_total_movies_is_293_l136_13600

noncomputable def original_movies (dvd_to_bluray_ratio : ℕ × ℕ) (additional_blurays : ℕ) (new_ratio : ℕ × ℕ) : ℕ :=
  let original_dvds := dvd_to_bluray_ratio.1
  let original_blurays := dvd_to_bluray_ratio.2
  let added_blurays := additional_blurays
  let new_dvds := new_ratio.1
  let new_blurays := new_ratio.2
  let x := (new_dvds * original_blurays - new_blurays * original_dvds) / (new_blurays * original_dvds - added_blurays * original_blurays)
  let total_movies := (original_dvds * x + original_blurays * x)
  let blurays_after_purchase := original_blurays * x + added_blurays

  if (new_dvds * (original_blurays * x + added_blurays) = new_blurays * (original_dvds * x))
  then 
    (original_dvds * x + original_blurays * x)
  else
    0 -- This case should theoretically never happen if the input ratios are consistent.

theorem original_total_movies_is_293 : original_movies (7, 2) 5 (13, 4) = 293 :=
by sorry

end NUMINAMATH_GPT_original_total_movies_is_293_l136_13600


namespace NUMINAMATH_GPT_corresponding_angles_equal_l136_13625

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end NUMINAMATH_GPT_corresponding_angles_equal_l136_13625


namespace NUMINAMATH_GPT_max_value_of_f_l136_13649

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : ∃ x, f x ≤ 6 / 5 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l136_13649


namespace NUMINAMATH_GPT_exponential_sum_sequence_l136_13651

noncomputable def Sn (n : ℕ) : ℝ :=
  Real.log (1 + 1 / n)

theorem exponential_sum_sequence : 
  e^(Sn 9 - Sn 6) = (20 : ℝ) / 21 := by
  sorry

end NUMINAMATH_GPT_exponential_sum_sequence_l136_13651


namespace NUMINAMATH_GPT_otimes_2_1_equals_3_l136_13644

namespace MathProof

-- Define the operation
def otimes (a b : ℝ) : ℝ := a^2 - b

-- The main theorem to prove
theorem otimes_2_1_equals_3 : otimes 2 1 = 3 :=
by
  -- Proof content not needed
  sorry

end MathProof

end NUMINAMATH_GPT_otimes_2_1_equals_3_l136_13644


namespace NUMINAMATH_GPT_gcd_12a_18b_l136_13619

theorem gcd_12a_18b (a b : ℕ) (h : Nat.gcd a b = 12) : Nat.gcd (12 * a) (18 * b) = 72 :=
sorry

end NUMINAMATH_GPT_gcd_12a_18b_l136_13619


namespace NUMINAMATH_GPT_find_positive_Y_for_nine_triangle_l136_13616

def triangle_relation (X Y : ℝ) : ℝ := X^2 + 3 * Y^2

theorem find_positive_Y_for_nine_triangle (Y : ℝ) : (9^2 + 3 * Y^2 = 360) → Y = Real.sqrt 93 := 
by
  sorry

end NUMINAMATH_GPT_find_positive_Y_for_nine_triangle_l136_13616


namespace NUMINAMATH_GPT_other_asymptote_of_hyperbola_l136_13664

theorem other_asymptote_of_hyperbola (a b : ℝ) :
  (∀ x : ℝ, a * x + b = 2 * x) →
  (∀ p : ℝ × ℝ, (p.1 = 3)) →
  ∀ (c : ℝ × ℝ), (c.1 = 3 ∧ c.2 = 6) ->
  ∃ (m : ℝ), m = -1/2 ∧ (∀ x, c.2 = -1/2 * x + 15/2) :=
by
  sorry

end NUMINAMATH_GPT_other_asymptote_of_hyperbola_l136_13664


namespace NUMINAMATH_GPT_sqrt_div_add_l136_13637

theorem sqrt_div_add :
  let sqrt_0_81 := 0.9
  let sqrt_1_44 := 1.2
  let sqrt_0_49 := 0.7
  (Real.sqrt 1.1 / sqrt_0_81) + (sqrt_1_44 / sqrt_0_49) = 2.8793 :=
by
  -- Prove equality using the given conditions
  sorry

end NUMINAMATH_GPT_sqrt_div_add_l136_13637


namespace NUMINAMATH_GPT_melanie_balloons_l136_13606

theorem melanie_balloons (joan_balloons melanie_balloons total_balloons : ℕ)
  (h_joan : joan_balloons = 40)
  (h_total : total_balloons = 81) :
  melanie_balloons = total_balloons - joan_balloons :=
by
  sorry

end NUMINAMATH_GPT_melanie_balloons_l136_13606


namespace NUMINAMATH_GPT_slope_proof_l136_13695

noncomputable def slope_between_midpoints : ℚ :=
  let p1 := (2, 3)
  let p2 := (4, 5)
  let q1 := (7, 3)
  let q2 := (8, 7)

  let midpoint (a b : ℚ × ℚ) : ℚ × ℚ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

  let m1 := midpoint p1 p2
  let m2 := midpoint q1 q2

  (m2.2 - m1.2) / (m2.1 - m1.1)

theorem slope_proof : slope_between_midpoints = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_slope_proof_l136_13695


namespace NUMINAMATH_GPT_rectangle_side_lengths_l136_13639

theorem rectangle_side_lengths:
  ∃ x : ℝ, ∃ y : ℝ, (2 * (x + y) * 2 = x * y) ∧ (y = x + 3) ∧ (x > 0) ∧ (y > 0) ∧ x = 8 ∧ y = 11 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_lengths_l136_13639


namespace NUMINAMATH_GPT_tyler_remaining_money_l136_13681

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end NUMINAMATH_GPT_tyler_remaining_money_l136_13681


namespace NUMINAMATH_GPT_max_quotient_l136_13602

theorem max_quotient (a b : ℝ) (h1 : 300 ≤ a) (h2 : a ≤ 500) (h3 : 900 ≤ b) (h4 : b ≤ 1800) :
  ∃ (q : ℝ), q = 5 / 9 ∧ (∀ (x y : ℝ), (300 ≤ x ∧ x ≤ 500) ∧ (900 ≤ y ∧ y ≤ 1800) → (x / y ≤ q)) :=
by
  use 5 / 9
  sorry

end NUMINAMATH_GPT_max_quotient_l136_13602


namespace NUMINAMATH_GPT_triangle_centroid_eq_l136_13636

-- Define the proof problem
theorem triangle_centroid_eq
  (P Q R G : ℝ × ℝ) -- Points P, Q, R, and G (the centroid of the triangle PQR)
  (centroid_eq : G = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)) -- Condition that G is the centroid
  (gp_sq_gq_sq_gr_sq_eq : dist G P ^ 2 + dist G Q ^ 2 + dist G R ^ 2 = 22) -- Given GP^2 + GQ^2 + GR^2 = 22
  : dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2 = 66 := -- Prove PQ^2 + PR^2 + QR^2 = 66
sorry -- Proof is omitted

end NUMINAMATH_GPT_triangle_centroid_eq_l136_13636


namespace NUMINAMATH_GPT_determine_quarters_given_l136_13682

def total_initial_coins (dimes quarters nickels : ℕ) : ℕ :=
  dimes + quarters + nickels

def updated_dimes (original_dimes added_dimes : ℕ) : ℕ :=
  original_dimes + added_dimes

def updated_nickels (original_nickels factor : ℕ) : ℕ :=
  original_nickels + original_nickels * factor

def total_coins_after_addition (dimes quarters nickels : ℕ) (added_dimes added_quarters added_nickels_factor : ℕ) : ℕ :=
  updated_dimes dimes added_dimes +
  (quarters + added_quarters) +
  updated_nickels nickels added_nickels_factor

def quarters_given_by_mother (total_coins initial_dimes initial_quarters initial_nickels added_dimes added_nickels_factor : ℕ) : ℕ :=
  total_coins - total_initial_coins initial_dimes initial_quarters initial_nickels - added_dimes - initial_nickels * added_nickels_factor

theorem determine_quarters_given :
  quarters_given_by_mother 35 2 6 5 2 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_determine_quarters_given_l136_13682


namespace NUMINAMATH_GPT_probability_same_color_shoes_l136_13609

theorem probability_same_color_shoes (pairs : ℕ) (total_shoes : ℕ)
  (each_pair_diff_color : pairs * 2 = total_shoes)
  (select_2_without_replacement : total_shoes = 10 ∧ pairs = 5) :
  let successful_outcomes := pairs
  let total_outcomes := (total_shoes * (total_shoes - 1)) / 2
  successful_outcomes / total_outcomes = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_shoes_l136_13609


namespace NUMINAMATH_GPT_when_to_sell_goods_l136_13627

variable (a : ℝ) (currentMonthProfit nextMonthProfitWithStorage : ℝ) 
          (interestRate storageFee thisMonthProfit nextMonthProfit : ℝ)
          (hm1 : interestRate = 0.005)
          (hm2 : storageFee = 5)
          (hm3 : thisMonthProfit = 100)
          (hm4 : nextMonthProfit = 120)
          (hm5 : currentMonthProfit = thisMonthProfit + (a + thisMonthProfit) * interestRate)
          (hm6 : nextMonthProfitWithStorage = nextMonthProfit - storageFee)

theorem when_to_sell_goods :
  (a > 2900 → currentMonthProfit > nextMonthProfitWithStorage) ∧
  (a = 2900 → currentMonthProfit = nextMonthProfitWithStorage) ∧
  (a < 2900 → currentMonthProfit < nextMonthProfitWithStorage) := by
  sorry

end NUMINAMATH_GPT_when_to_sell_goods_l136_13627


namespace NUMINAMATH_GPT_original_faculty_size_l136_13653

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end NUMINAMATH_GPT_original_faculty_size_l136_13653


namespace NUMINAMATH_GPT_find_x_l136_13623

-- Define the digits used
def digits : List ℕ := [1, 4, 5]

-- Define the sum of all four-digit numbers formed
def sum_of_digits (x : ℕ) : ℕ :=
  24 * (1 + 4 + 5 + x)

-- State the theorem
theorem find_x (x : ℕ) (h : sum_of_digits x = 288) : x = 2 :=
  by
    sorry

end NUMINAMATH_GPT_find_x_l136_13623


namespace NUMINAMATH_GPT_max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l136_13665

theorem max_value_y_eq_x_mul_2_minus_x (x : ℝ) (h : 0 < x ∧ x < 3 / 2) : ∃ y : ℝ, y = x * (2 - x) ∧ y ≤ 1 :=
sorry

theorem min_value_y_eq_x_plus_4_div_x_minus_3 (x : ℝ) (h : x > 3) : ∃ y : ℝ, y = x + 4 / (x - 3) ∧ y ≥ 7 :=
sorry

end NUMINAMATH_GPT_max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l136_13665


namespace NUMINAMATH_GPT_smallest_integer_sum_consecutive_l136_13666

theorem smallest_integer_sum_consecutive
  (l m n a : ℤ)
  (h1 : a = 9 * l + 36)
  (h2 : a = 10 * m + 45)
  (h3 : a = 11 * n + 55)
  : a = 495 :=
sorry

end NUMINAMATH_GPT_smallest_integer_sum_consecutive_l136_13666


namespace NUMINAMATH_GPT_B_more_cost_effective_l136_13674

variable (x y : ℝ)
variable (hx : x ≠ y)

theorem B_more_cost_effective (x y : ℝ) (hx : x ≠ y) :
  (1/2 * x + 1/2 * y) > (2 * x * y / (x + y)) :=
by
  sorry

end NUMINAMATH_GPT_B_more_cost_effective_l136_13674


namespace NUMINAMATH_GPT_find_expression_l136_13615

variable (a b E : ℝ)

-- Conditions
def condition1 := a / b = 4 / 3
def condition2 := E / (3 * a - 2 * b) = 3

-- Conclusion we want to prove
theorem find_expression : condition1 a b → condition2 a b E → E = 6 * b :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_expression_l136_13615


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l136_13603

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : a 6 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l136_13603


namespace NUMINAMATH_GPT_angle_B_area_of_triangle_l136_13629

/-
Given a triangle ABC with angle A, B, C and sides a, b, c opposite to these angles respectively.
Consider the conditions:
- A = π/6
- b = (4 + 2 * sqrt 3) * a * cos B
- b = 1

Prove:
1. B = 5 * π / 12
2. The area of triangle ABC = 1 / 4
-/

namespace TriangleProof

open Real

def triangle_conditions (A B C a b c : ℝ) : Prop :=
  A = π / 6 ∧
  b = (4 + 2 * sqrt 3) * a * cos B ∧
  b = 1

theorem angle_B (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  B = 5 * π / 12 :=
sorry

theorem area_of_triangle (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  1 / 2 * b * c * sin A = 1 / 4 :=
sorry

end TriangleProof

end NUMINAMATH_GPT_angle_B_area_of_triangle_l136_13629


namespace NUMINAMATH_GPT_gambler_initial_games_l136_13641

theorem gambler_initial_games (x : ℕ)
  (h1 : ∀ x, ∃ (wins : ℝ), wins = 0.40 * x) 
  (h2 : ∀ x, ∃ (total_games : ℕ), total_games = x + 30)
  (h3 : ∀ x, ∃ (total_wins : ℝ), total_wins = 0.40 * x + 24)
  (h4 : ∀ x, ∃ (final_win_rate : ℝ), final_win_rate = (0.40 * x + 24) / (x + 30))
  (h5 : ∃ (final_win_rate_target : ℝ), final_win_rate_target = 0.60) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_gambler_initial_games_l136_13641


namespace NUMINAMATH_GPT_log_eqn_proof_l136_13675

theorem log_eqn_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 4 = 8)
  (h2 : Real.log a / Real.log 4 + Real.log b / Real.log 8 = 2) :
  Real.log a / Real.log 8 + Real.log b / Real.log 2 = -52 / 3 := 
by
  sorry

end NUMINAMATH_GPT_log_eqn_proof_l136_13675


namespace NUMINAMATH_GPT_which_set_can_form_triangle_l136_13692

-- Definition of the triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for each set of line segments
def setA := (2, 6, 8)
def setB := (4, 6, 7)
def setC := (5, 6, 12)
def setD := (2, 3, 6)

-- Proof problem statement
theorem which_set_can_form_triangle : 
  triangle_inequality 2 6 8 = false ∧
  triangle_inequality 4 6 7 = true ∧
  triangle_inequality 5 6 12 = false ∧
  triangle_inequality 2 3 6 = false := 
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_which_set_can_form_triangle_l136_13692


namespace NUMINAMATH_GPT_solve_quadratic_equation_l136_13634

theorem solve_quadratic_equation (x : ℝ) : 2 * (x + 1) ^ 2 - 49 = 1 ↔ (x = 4 ∨ x = -6) := 
sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l136_13634


namespace NUMINAMATH_GPT_perimeter_of_plot_l136_13652

variable (length breadth : ℝ)
variable (h_ratio : length / breadth = 7 / 5)
variable (h_area : length * breadth = 5040)

theorem perimeter_of_plot (h_ratio : length / breadth = 7 / 5) (h_area : length * breadth = 5040) : 
  (2 * length + 2 * breadth = 288) :=
sorry

end NUMINAMATH_GPT_perimeter_of_plot_l136_13652


namespace NUMINAMATH_GPT_solve_x_squared_plus_y_squared_l136_13699

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end NUMINAMATH_GPT_solve_x_squared_plus_y_squared_l136_13699


namespace NUMINAMATH_GPT_looms_employed_l136_13656

def sales_value := 500000
def manufacturing_expenses := 150000
def establishment_charges := 75000
def profit_decrease := 5000

def profit_per_loom (L : ℕ) : ℕ := (sales_value / L) - (manufacturing_expenses / L)

theorem looms_employed (L : ℕ) (h : profit_per_loom L = profit_decrease) : L = 70 :=
by
  have h_eq : profit_per_loom L = (sales_value - manufacturing_expenses) / L := by
    sorry
  have profit_expression : profit_per_loom L = profit_decrease := by
    sorry
  have L_value : L = (sales_value - manufacturing_expenses) / profit_decrease := by
    sorry
  have L_is_70 : L = 70 := by
    sorry
  exact L_is_70

end NUMINAMATH_GPT_looms_employed_l136_13656


namespace NUMINAMATH_GPT_number_of_nickels_l136_13607

-- Define the conditions
variable (m : ℕ) -- Total number of coins initially
variable (v : ℕ) -- Total value of coins initially in cents
variable (n : ℕ) -- Number of nickels

-- State the conditions in terms of mathematical equations
-- Condition 1: Average value is 25 cents
axiom avg_value_initial : v = 25 * m

-- Condition 2: Adding one half-dollar (50 cents) results in average of 26 cents
axiom avg_value_after_half_dollar : v + 50 = 26 * (m + 1)

-- Define the relationship between the number of each type of coin and the total value
-- We sum the individual products of the count of each type and their respective values
axiom total_value_definition : v = 5 * n  -- since the problem already validates with total_value == 25m

-- Question to prove
theorem number_of_nickels : n = 30 :=
by
  -- Since we are not providing proof, we will use sorry to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_nickels_l136_13607


namespace NUMINAMATH_GPT_tangent_line_eqn_l136_13684

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_eqn (h : f' x = 2) : 2 * x - y - Real.exp 1 + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eqn_l136_13684


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l136_13631

theorem hyperbola_eccentricity : 
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 6 / 2 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l136_13631


namespace NUMINAMATH_GPT_buffalo_weight_rounding_l136_13687

theorem buffalo_weight_rounding
  (weight_kg : ℝ) (conversion_factor : ℝ) (expected_weight_lb : ℕ) :
  weight_kg = 850 →
  conversion_factor = 0.454 →
  expected_weight_lb = 1872 →
  Nat.floor (weight_kg / conversion_factor + 0.5) = expected_weight_lb :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_buffalo_weight_rounding_l136_13687


namespace NUMINAMATH_GPT_weight_loss_l136_13638

def initial_weight : ℕ := 69
def current_weight : ℕ := 34

theorem weight_loss :
  initial_weight - current_weight = 35 :=
by
  sorry

end NUMINAMATH_GPT_weight_loss_l136_13638


namespace NUMINAMATH_GPT_least_pos_int_solution_l136_13632

theorem least_pos_int_solution (x : ℤ) : x + 4609 ≡ 2104 [ZMOD 12] → x = 3 := by
  sorry

end NUMINAMATH_GPT_least_pos_int_solution_l136_13632


namespace NUMINAMATH_GPT_max_experiments_fibonacci_search_l136_13694

-- Define the conditions and the theorem
def is_unimodal (f : ℕ → ℕ) : Prop :=
  ∃ k, ∀ n m, (n < k ∧ k ≤ m) → f n < f k ∧ f k > f m

def fibonacci_search_experiments (n : ℕ) : ℕ :=
  -- Placeholder function representing the steps of Fibonacci search
  if n <= 1 then n else fibonacci_search_experiments (n - 1) + fibonacci_search_experiments (n - 2)

theorem max_experiments_fibonacci_search (f : ℕ → ℕ) (n : ℕ) (hn : n = 33) (hf : is_unimodal f) : fibonacci_search_experiments n ≤ 7 :=
  sorry

end NUMINAMATH_GPT_max_experiments_fibonacci_search_l136_13694


namespace NUMINAMATH_GPT_simplify_expression_l136_13604

theorem simplify_expression (x : ℝ) :
  ((3 * x^2 + 2 * x - 1) + 2 * x^2) * 4 + (5 - 2 / 2) * (3 * x^2 + 6 * x - 8) = 32 * x^2 + 32 * x - 36 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l136_13604


namespace NUMINAMATH_GPT_parabola_intersections_l136_13662

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ :=
  2 * x^2 - 10 * x - 10

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ :=
  x^2 - 4 * x + 6

-- Define the theorem stating the points of intersection
theorem parabola_intersections :
  ∀ (p : ℝ × ℝ), (parabola1 p.1 = p.2) ∧ (parabola2 p.1 = p.2) ↔ (p = (-2, 18) ∨ p = (8, 38)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersections_l136_13662


namespace NUMINAMATH_GPT_inequality_sum_geq_three_l136_13617

theorem inequality_sum_geq_three
  (a b c : ℝ)
  (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) + 
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := 
sorry

end NUMINAMATH_GPT_inequality_sum_geq_three_l136_13617


namespace NUMINAMATH_GPT_value_division_l136_13659

theorem value_division (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) 
                       (h3 : x / y = n) : n = 4 := 
by 
sorry

end NUMINAMATH_GPT_value_division_l136_13659


namespace NUMINAMATH_GPT_inequality_solution_l136_13679

theorem inequality_solution (x : ℝ) (hx : x > 0) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l136_13679


namespace NUMINAMATH_GPT_find_f_1_l136_13672

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_1 : (∀ x : ℝ, f x + 3 * f (-x) = Real.logb 2 (x + 3)) → f 1 = 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_1_l136_13672


namespace NUMINAMATH_GPT_quadratic_roots_l136_13677

theorem quadratic_roots (x : ℝ) : (x^2 - 8 * x - 2 = 0) ↔ (x = 4 + 3 * Real.sqrt 2) ∨ (x = 4 - 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l136_13677


namespace NUMINAMATH_GPT_f_sum_positive_l136_13640

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x₁ x₂ x₃ : ℝ) (h₁₂ : x₁ + x₂ > 0) (h₂₃ : x₂ + x₃ > 0) (h₃₁ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := 
sorry

end NUMINAMATH_GPT_f_sum_positive_l136_13640


namespace NUMINAMATH_GPT_find_ordered_pair_l136_13642

noncomputable def discriminant_eq_zero (a c : ℝ) : Prop :=
  a * c = 9

def sum_eq_14 (a c : ℝ) : Prop :=
  a + c = 14

def a_greater_than_c (a c : ℝ) : Prop :=
  a > c

theorem find_ordered_pair : 
  ∃ (a c : ℝ), 
    sum_eq_14 a c ∧ 
    discriminant_eq_zero a c ∧ 
    a_greater_than_c a c ∧ 
    a = 7 + 2 * Real.sqrt 10 ∧ 
    c = 7 - 2 * Real.sqrt 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_ordered_pair_l136_13642


namespace NUMINAMATH_GPT_alex_final_bill_l136_13678

def original_bill : ℝ := 500
def first_late_charge (bill : ℝ) : ℝ := bill * 1.02
def final_bill (bill : ℝ) : ℝ := first_late_charge bill * 1.03

theorem alex_final_bill : final_bill original_bill = 525.30 :=
by sorry

end NUMINAMATH_GPT_alex_final_bill_l136_13678


namespace NUMINAMATH_GPT_female_salmon_returned_l136_13614

theorem female_salmon_returned :
  let total_salmon : ℕ := 971639
  let male_salmon : ℕ := 712261
  total_salmon - male_salmon = 259378 :=
by
  let total_salmon := 971639
  let male_salmon := 712261
  calc
    971639 - 712261 = 259378 := by norm_num

end NUMINAMATH_GPT_female_salmon_returned_l136_13614


namespace NUMINAMATH_GPT_product_of_numbers_l136_13647

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 30 * k) : 
  x * y = 400 / 7 := 
sorry

end NUMINAMATH_GPT_product_of_numbers_l136_13647


namespace NUMINAMATH_GPT_additional_men_required_l136_13624

variables (W_r : ℚ) (W : ℚ) (D : ℚ) (M : ℚ) (E : ℚ)

-- Given variables
def initial_work_rate := (2.5 : ℚ) / (50 * 100)
def remaining_work_length := (12.5 : ℚ)
def remaining_days := (200 : ℚ)
def initial_men := (50 : ℚ)
def additional_men_needed := (75 : ℚ)

-- Calculating the additional men required
theorem additional_men_required
  (calc_wr : W_r = initial_work_rate)
  (calc_wr_remain : W = remaining_work_length)
  (calc_days_remain : D = remaining_days)
  (calc_initial_men : M = initial_men)
  (calc_additional_men : M + E = (125 : ℚ)) :
  E = additional_men_needed :=
sorry

end NUMINAMATH_GPT_additional_men_required_l136_13624


namespace NUMINAMATH_GPT_Sheila_attend_probability_l136_13671

noncomputable def prob_rain := 0.3
noncomputable def prob_sunny := 0.4
noncomputable def prob_cloudy := 0.3

noncomputable def prob_attend_if_rain := 0.25
noncomputable def prob_attend_if_sunny := 0.9
noncomputable def prob_attend_if_cloudy := 0.5

noncomputable def prob_attend :=
  prob_rain * prob_attend_if_rain +
  prob_sunny * prob_attend_if_sunny +
  prob_cloudy * prob_attend_if_cloudy

theorem Sheila_attend_probability : prob_attend = 0.585 := by
  sorry

end NUMINAMATH_GPT_Sheila_attend_probability_l136_13671


namespace NUMINAMATH_GPT_maximum_S_n_l136_13658

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem maximum_S_n (a_1 : ℝ) (h : a_1 > 0)
  (h_sequence : 3 * a_n a_1 (2 * a_1 / 39) 8 = 5 * a_n a_1 (2 * a_1 / 39) 13)
  : ∀ n : ℕ, S_n a_1 (2 * a_1 / 39) n ≤ S_n a_1 (2 * a_1 / 39) 20 :=
sorry

end NUMINAMATH_GPT_maximum_S_n_l136_13658


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l136_13648

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + abs (2*x - a)

-- Proof statements
theorem problem_part1 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) : a = 0 := sorry

theorem problem_part2 (a : ℝ) (h_a_gt_two : a > 2) : 
  ∃ x : ℝ, ∀ y : ℝ, f x a ≤ f y a ∧ f x a = a - 1 := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l136_13648


namespace NUMINAMATH_GPT_initial_kittens_count_l136_13688

-- Let's define the initial conditions first.
def kittens_given_away : ℕ := 2
def kittens_remaining : ℕ := 6

-- The main theorem to prove the initial number of kittens.
theorem initial_kittens_count : (kittens_given_away + kittens_remaining) = 8 :=
by sorry

end NUMINAMATH_GPT_initial_kittens_count_l136_13688


namespace NUMINAMATH_GPT_negation_of_implication_l136_13630

variable (a b c : ℝ)

theorem negation_of_implication :
  (¬(a + b + c = 3) → a^2 + b^2 + c^2 < 3) ↔
  ¬((a + b + c = 3) → a^2 + b^2 + c^2 ≥ 3) := by
sorry

end NUMINAMATH_GPT_negation_of_implication_l136_13630


namespace NUMINAMATH_GPT_triangle_side_lengths_expression_neg_l136_13605

theorem triangle_side_lengths_expression_neg {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * a^2 * b^2 - 2 * b^2 * c^2 - 2 * c^2 * a^2 < 0 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_expression_neg_l136_13605


namespace NUMINAMATH_GPT_most_reasonable_sampling_method_l136_13669

-- Define the conditions
axiom significant_differences_in_educational_stages : Prop
axiom insignificant_differences_between_genders : Prop

-- Define the options
inductive SamplingMethod
| SimpleRandomSampling
| StratifiedSamplingByGender
| StratifiedSamplingByEducationalStage
| SystematicSampling

-- State the problem as a theorem
theorem most_reasonable_sampling_method
  (H1 : significant_differences_in_educational_stages)
  (H2 : insignificant_differences_between_genders) :
  SamplingMethod.StratifiedSamplingByEducationalStage = SamplingMethod.StratifiedSamplingByEducationalStage :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_most_reasonable_sampling_method_l136_13669


namespace NUMINAMATH_GPT_product_of_digits_l136_13626

theorem product_of_digits (A B : ℕ) (h1 : A + B = 14) (h2 : (10 * A + B) % 4 = 0) : A * B = 48 :=
by
  sorry

end NUMINAMATH_GPT_product_of_digits_l136_13626


namespace NUMINAMATH_GPT_circle_equation_range_of_k_l136_13683

theorem circle_equation_range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 4*k*x - 2*y + 5*k = 0) ↔ (k > 1 ∨ k < 1/4) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_range_of_k_l136_13683


namespace NUMINAMATH_GPT_tangent_values_l136_13613

theorem tangent_values (A : ℝ) (h : A < π) (cos_A : Real.cos A = 3 / 5) :
  Real.tan A = 4 / 3 ∧ Real.tan (A + π / 4) = -7 := 
by
  sorry

end NUMINAMATH_GPT_tangent_values_l136_13613


namespace NUMINAMATH_GPT_age_of_25th_student_l136_13635

theorem age_of_25th_student 
(A : ℤ) (B : ℤ) (C : ℤ) (D : ℤ)
(total_students : ℤ)
(total_age : ℤ)
(age_all_students : ℤ)
(avg_age_all_students : ℤ)
(avg_age_7_students : ℤ)
(avg_age_12_students : ℤ)
(avg_age_5_students : ℤ)
:
total_students = 25 →
avg_age_all_students = 18 →
avg_age_7_students = 20 →
avg_age_12_students = 16 →
avg_age_5_students = 19 →
total_age = total_students * avg_age_all_students →
age_all_students = total_age - (7 * avg_age_7_students + 12 * avg_age_12_students + 5 * avg_age_5_students) →
A = 7 * avg_age_7_students →
B = 12 * avg_age_12_students →
C = 5 * avg_age_5_students →
D = total_age - (A + B + C) →
D = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_of_25th_student_l136_13635


namespace NUMINAMATH_GPT_tree_growth_rate_l136_13698

noncomputable def growth_rate_per_week (initial_height final_height : ℝ) (months weeks_per_month : ℕ) : ℝ :=
  (final_height - initial_height) / (months * weeks_per_month)

theorem tree_growth_rate :
  growth_rate_per_week 10 42 4 4 = 2 := 
by
  sorry

end NUMINAMATH_GPT_tree_growth_rate_l136_13698


namespace NUMINAMATH_GPT_sin_neg_45_l136_13690

theorem sin_neg_45 :
  Real.sin (-45 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_sin_neg_45_l136_13690


namespace NUMINAMATH_GPT_tan_value_l136_13668

open Real

theorem tan_value (α : ℝ) 
  (h1 : sin (α + π / 6) = -3 / 5)
  (h2 : -2 * π / 3 < α ∧ α < -π / 6) : 
  tan (4 * π / 3 - α) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_value_l136_13668


namespace NUMINAMATH_GPT_customers_in_other_countries_l136_13618

-- Given 
def total_customers : ℕ := 7422
def customers_in_us : ℕ := 723

-- To Prove
theorem customers_in_other_countries : (total_customers - customers_in_us) = 6699 := 
by
  sorry

end NUMINAMATH_GPT_customers_in_other_countries_l136_13618


namespace NUMINAMATH_GPT_biology_vs_reading_diff_l136_13655

def math_hw_pages : ℕ := 2
def reading_hw_pages : ℕ := 3
def total_hw_pages : ℕ := 15

def biology_hw_pages : ℕ := total_hw_pages - (math_hw_pages + reading_hw_pages)

theorem biology_vs_reading_diff : (biology_hw_pages - reading_hw_pages) = 7 := by
  sorry

end NUMINAMATH_GPT_biology_vs_reading_diff_l136_13655


namespace NUMINAMATH_GPT_find_k_l136_13670

variable {x y k : ℝ}

theorem find_k (h1 : 3 * x + 4 * y = k + 2) 
             (h2 : 2 * x + y = 4) 
             (h3 : x + y = 2) :
  k = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l136_13670


namespace NUMINAMATH_GPT_area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l136_13693

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (cosA : ℝ) (sinA : ℝ)
variable (area : ℝ)
variable (tanA tanB tanC : ℝ)

-- Given conditions
axiom angle_identity : b^2 + c^2 = 3 * b * c * cosA
axiom sin_cos_identity : sinA^2 + cosA^2 = 1
axiom law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cosA

-- Part (1) statement
theorem area_of_triangle_is_sqrt_5 (B_eq_C : B = C) (a_eq_2 : a = 2) 
    (cosA_eq_2_3 : cosA = 2/3) 
    (b_eq_sqrt6 : b = Real.sqrt 6) 
    (sinA_eq_sqrt5_3 : sinA = Real.sqrt 5 / 3) 
    : area = Real.sqrt 5 := sorry

-- Part (2) statement
theorem sum_of_tangents_eq_1 (tanA_eq : tanA = sinA / cosA)
    (tanB_eq : tanB = sinA * sinA / (cosA * cosA))
    (tanC_eq : tanC = sinA * sinA / (cosA * cosA))
    : (tanA / tanB) + (tanA / tanC) = 1 := sorry

end NUMINAMATH_GPT_area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l136_13693


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l136_13622

theorem necessary_but_not_sufficient (a : ℝ) (ha : a > 1) : a^2 > a :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l136_13622


namespace NUMINAMATH_GPT_no_integer_solution_for_equation_l136_13611

theorem no_integer_solution_for_equation :
    ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x * y * z - 1 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_equation_l136_13611


namespace NUMINAMATH_GPT_lab_preparation_is_correct_l136_13691

def correct_operation (m_CuSO4 : ℝ) (m_CuSO4_5H2O : ℝ) (V_solution : ℝ) : Prop :=
  let molar_mass_CuSO4 := 160 -- g/mol
  let molar_mass_CuSO4_5H2O := 250 -- g/mol
  let desired_concentration := 0.1 -- mol/L
  let desired_volume := 0.480 -- L
  let prepared_volume := 0.500 -- L
  (m_CuSO4 = 8.0 ∧ V_solution = 0.500 ∧ m_CuSO4_5H2O = 12.5 ∧ desired_concentration * prepared_volume * molar_mass_CuSO4_5H2O = 12.5)

-- Example proof statement to show the problem with "sorry"
theorem lab_preparation_is_correct : correct_operation 8.0 12.5 0.500 :=
by
  sorry

end NUMINAMATH_GPT_lab_preparation_is_correct_l136_13691


namespace NUMINAMATH_GPT_inequality_solution_l136_13676

-- Condition definitions in lean
def numerator (x : ℝ) : ℝ := (x^5 - 13 * x^3 + 36 * x) * (x^4 - 17 * x^2 + 16)
def denominator (y : ℝ) : ℝ := (y^5 - 13 * y^3 + 36 * y) * (y^4 - 17 * y^2 + 16)

-- Given the critical conditions
def is_zero_or_pm1_pm2_pm3_pm4 (y : ℝ) : Prop := 
  y = 0 ∨ y = 1 ∨ y = -1 ∨ y = 2 ∨ y = -2 ∨ y = 3 ∨ y = -3 ∨ y = 4 ∨ y = -4

-- The theorem statement
theorem inequality_solution (x y : ℝ) : 
  (numerator x / denominator y) ≥ 0 ↔ ¬ (is_zero_or_pm1_pm2_pm3_pm4 y) :=
sorry -- proof to be filled in later

end NUMINAMATH_GPT_inequality_solution_l136_13676


namespace NUMINAMATH_GPT_cross_prod_correct_l136_13650

open Matrix

def vec1 : ℝ × ℝ × ℝ := (3, -1, 4)
def vec2 : ℝ × ℝ × ℝ := (-4, 6, 2)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
  a.2.2 * b.1 - a.1 * b.2.2,
  a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_prod_correct :
  cross_product vec1 vec2 = (-26, -22, 14) := by
  -- sorry is used to simplify the proof.
  sorry

end NUMINAMATH_GPT_cross_prod_correct_l136_13650


namespace NUMINAMATH_GPT_largest_number_is_56_l136_13696

-- Definitions based on the conditions
def ratio_three_five_seven (a b c : ℕ) : Prop :=
  3 * c = a ∧ 5 * c = b ∧ 7 * c = c

def difference_is_32 (a c : ℕ) : Prop :=
  c - a = 32

-- Statement of the proof
theorem largest_number_is_56 (a b c : ℕ) (h1 : ratio_three_five_seven a b c) (h2 : difference_is_32 a c) : c = 56 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_is_56_l136_13696


namespace NUMINAMATH_GPT_problem_statement_l136_13601

theorem problem_statement : 2456 + 144 / 12 * 5 - 256 = 2260 := 
by
  -- statements and proof steps would go here
  sorry

end NUMINAMATH_GPT_problem_statement_l136_13601


namespace NUMINAMATH_GPT_smallest_n_conditions_l136_13685

theorem smallest_n_conditions :
  ∃ n : ℕ, 0 < n ∧ (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^4) ∧ n = 54 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_conditions_l136_13685


namespace NUMINAMATH_GPT_all_sets_form_right_angled_triangle_l136_13645

theorem all_sets_form_right_angled_triangle :
    (6 * 6 + 8 * 8 = 10 * 10) ∧
    (7 * 7 + 24 * 24 = 25 * 25) ∧
    (3 * 3 + 4 * 4 = 5 * 5) ∧
    (Real.sqrt 2 * Real.sqrt 2 + Real.sqrt 3 * Real.sqrt 3 = Real.sqrt 5 * Real.sqrt 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_all_sets_form_right_angled_triangle_l136_13645


namespace NUMINAMATH_GPT_find_side_length_a_l136_13654

variable {a b c : ℝ}
variable {B : ℝ}

theorem find_side_length_a (h_b : b = 7) (h_c : c = 5) (h_B : B = 2 * Real.pi / 3) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_find_side_length_a_l136_13654


namespace NUMINAMATH_GPT_complement_intersection_l136_13628

open Set

-- Definitions of the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

-- The theorem we want to prove
theorem complement_intersection :
  (compl M ∩ N) = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l136_13628


namespace NUMINAMATH_GPT_arrange_in_order_l136_13620

noncomputable def a := (Real.sqrt 2 / 2) * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c := Real.sqrt 3 / 2

theorem arrange_in_order : c < a ∧ a < b := 
by
  sorry

end NUMINAMATH_GPT_arrange_in_order_l136_13620


namespace NUMINAMATH_GPT_length_error_probability_l136_13660

theorem length_error_probability
  (μ σ : ℝ)
  (X : ℝ → ℝ)
  (h_norm_dist : ∀ x : ℝ, X x = (Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) / (σ * Real.sqrt (2 * Real.pi))))
  (h_max_density : X 0 = 1 / (3 * Real.sqrt (2 * Real.pi)))
  (P : Set ℝ → ℝ)
  (h_prop1 : P {x | μ - σ < x ∧ x < μ + σ} = 0.6826)
  (h_prop2 : P {x | μ - 2 * σ < x ∧ x < μ + 2 * σ} = 0.9544) :
  P {x | 3 < x ∧ x < 6} = 0.1359 :=
sorry

end NUMINAMATH_GPT_length_error_probability_l136_13660


namespace NUMINAMATH_GPT_student_solved_correctly_l136_13663

theorem student_solved_correctly (c e : ℕ) (h1 : c + e = 80) (h2 : 5 * c - 3 * e = 8) : c = 31 :=
sorry

end NUMINAMATH_GPT_student_solved_correctly_l136_13663


namespace NUMINAMATH_GPT_point_Q_in_first_quadrant_l136_13610

theorem point_Q_in_first_quadrant (a b : ℝ) (h : a < 0 ∧ b < 0) : (0 < -a) ∧ (0 < -b) :=
by
  have ha : -a > 0 := by linarith
  have hb : -b > 0 := by linarith
  exact ⟨ha, hb⟩

end NUMINAMATH_GPT_point_Q_in_first_quadrant_l136_13610


namespace NUMINAMATH_GPT_eval_expression_l136_13697

theorem eval_expression : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l136_13697


namespace NUMINAMATH_GPT_total_school_population_220_l136_13667

theorem total_school_population_220 (x B : ℕ) 
  (h1 : 242 = (x * B) / 100) 
  (h2 : B = (50 * x) / 100) : x = 220 := by
  sorry

end NUMINAMATH_GPT_total_school_population_220_l136_13667


namespace NUMINAMATH_GPT_complex_power_difference_l136_13673

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 16 - (1 - i) ^ 16 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_power_difference_l136_13673


namespace NUMINAMATH_GPT_find_a_value_l136_13657

theorem find_a_value (a a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) → 
  a = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l136_13657


namespace NUMINAMATH_GPT_paint_ratio_l136_13661

theorem paint_ratio
  (blue yellow white : ℕ)
  (ratio_b : ℕ := 4)
  (ratio_y : ℕ := 3)
  (ratio_w : ℕ := 5)
  (total_white : ℕ := 15)
  : yellow = 9 := by
  have ratio := ratio_b + ratio_y + ratio_w
  have white_parts := total_white * ratio_w / ratio_w
  have yellow_parts := white_parts * ratio_y / ratio_w
  exact sorry

end NUMINAMATH_GPT_paint_ratio_l136_13661


namespace NUMINAMATH_GPT_tangent_line_at_P_l136_13646

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 1

def P : ℝ × ℝ := (-1, 3)

theorem tangent_line_at_P :
    ∀ (x y : ℝ), (y = 2*x^2 + 1) →
    (x, y) = P →
    ∃ m b : ℝ, b = -1 ∧ m = -4 ∧ (y = m*x + b) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_P_l136_13646


namespace NUMINAMATH_GPT_smallest_value_of_x_l136_13612

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_x_l136_13612


namespace NUMINAMATH_GPT_triangle_perimeter_l136_13689

theorem triangle_perimeter (x : ℕ) (hx1 : x % 2 = 1) (hx2 : 5 < x) (hx3 : x < 11) : 
  (3 + 8 + x = 18) ∨ (3 + 8 + x = 20) :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l136_13689


namespace NUMINAMATH_GPT_smallest_possible_value_l136_13608

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end NUMINAMATH_GPT_smallest_possible_value_l136_13608


namespace NUMINAMATH_GPT_find_f_at_4_l136_13686

noncomputable def f : ℝ → ℝ := sorry -- We assume such a function exists

theorem find_f_at_4:
  (∀ x : ℝ, f (4^x) + x * f (4^(-x)) = 3) → f (4) = 0 := by
  intro h
  -- Proof would go here, but is omitted as per instructions
  sorry

end NUMINAMATH_GPT_find_f_at_4_l136_13686
