import Mathlib

namespace binomial_12_6_eq_924_l252_252956

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l252_252956


namespace tangent_circles_locus_l252_252100

noncomputable def locus_condition (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

theorem tangent_circles_locus (a b : ℝ) (r : ℝ) (h : locus_condition a b r) :
  a^2 + 7 * b^2 - 34 * a - 57 = 0 :=
sorry

end tangent_circles_locus_l252_252100


namespace min_total_cost_of_tank_l252_252433

theorem min_total_cost_of_tank (V D c₁ c₂ : ℝ) (hV : V = 0.18) (hD : D = 0.5)
  (hc₁ : c₁ = 400) (hc₂ : c₂ = 100) : 
  ∃ x : ℝ, x > 0 ∧ (y = c₂*D*(2*x + 0.72/x) + c₁*0.36) ∧ y = 264 := 
sorry

end min_total_cost_of_tank_l252_252433


namespace calculation_expression_solve_system_of_equations_l252_252140

-- Part 1: Prove the calculation
theorem calculation_expression :
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2) ^ 2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 :=
by
  -- proof will be here
  sorry

-- Part 2: Prove the solution of the system of equations
theorem solve_system_of_equations (x y : ℝ) :
  (5 * x - y = -9) ∧ (3 * x + y = 1) → (x = -1 ∧ y = 4) :=
by
  -- proof will be here
  sorry

end calculation_expression_solve_system_of_equations_l252_252140


namespace sum_of_remainders_l252_252569

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end sum_of_remainders_l252_252569


namespace num_comics_bought_l252_252085

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end num_comics_bought_l252_252085


namespace binomial_evaluation_l252_252949

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l252_252949


namespace faye_pencils_l252_252639

theorem faye_pencils (rows crayons : ℕ) (pencils_per_row : ℕ) (h1 : rows = 7) (h2 : pencils_per_row = 5) : 
  (rows * pencils_per_row) = 35 :=
by {
  sorry
}

end faye_pencils_l252_252639


namespace part_a_not_divisible_by_29_part_b_divisible_by_11_l252_252736
open Nat

-- Part (a): Checking divisibility of 5641713 by 29
def is_divisible_by_29 (n : ℕ) : Prop :=
  n % 29 = 0

theorem part_a_not_divisible_by_29 : ¬is_divisible_by_29 5641713 :=
  by sorry

-- Part (b): Checking divisibility of 1379235 by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem part_b_divisible_by_11 : is_divisible_by_11 1379235 :=
  by sorry

end part_a_not_divisible_by_29_part_b_divisible_by_11_l252_252736


namespace difference_of_squares_example_l252_252730

theorem difference_of_squares_example (a b : ℕ) (h₁ : a = 650) (h₂ : b = 350) :
  a^2 - b^2 = 300000 :=
by
  sorry

end difference_of_squares_example_l252_252730


namespace bottles_left_on_shelf_l252_252118

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l252_252118


namespace avg_licks_l252_252622

theorem avg_licks (Dan Michael Sam David Lance : ℕ) 
  (hDan : Dan = 58) 
  (hMichael : Michael = 63) 
  (hSam : Sam = 70) 
  (hDavid : David = 70) 
  (hLance : Lance = 39) : 
  (Dan + Michael + Sam + David + Lance) / 5 = 60 :=
by 
  sorry

end avg_licks_l252_252622


namespace fraction_paint_remaining_l252_252604

theorem fraction_paint_remaining :
  let original_paint := 1
  let first_day_usage := original_paint / 4
  let paint_remaining_after_first_day := original_paint - first_day_usage
  let second_day_usage := paint_remaining_after_first_day / 2
  let paint_remaining_after_second_day := paint_remaining_after_first_day - second_day_usage
  let third_day_usage := paint_remaining_after_second_day / 3
  let paint_remaining_after_third_day := paint_remaining_after_second_day - third_day_usage
  paint_remaining_after_third_day = original_paint / 4 := 
by
  sorry

end fraction_paint_remaining_l252_252604


namespace discount_percentage_l252_252912

theorem discount_percentage (CP MP SP D : ℝ) (cp_value : CP = 100) 
(markup : MP = CP + 0.5 * CP) (profit : SP = CP + 0.35 * CP) 
(discount : D = MP - SP) : (D / MP) * 100 = 10 := 
by 
  sorry

end discount_percentage_l252_252912


namespace basketball_total_points_l252_252800

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end basketball_total_points_l252_252800


namespace camel_cost_is_5200_l252_252600

-- Definitions of costs in terms of Rs.
variable (C H O E : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : ∃ X : ℕ, X * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 130000

-- Theorem to prove
theorem camel_cost_is_5200 (hC : C = 5200) : C = 5200 :=
by sorry

end camel_cost_is_5200_l252_252600


namespace range_alpha_minus_beta_over_2_l252_252179

theorem range_alpha_minus_beta_over_2 (α β : ℝ) (h1 : -π / 2 ≤ α) (h2 : α < β) (h3 : β ≤ π / 2) :
  Set.Ico (-π / 2) 0 = {x : ℝ | ∃ α β : ℝ, -π / 2 ≤ α ∧ α < β ∧ β ≤ π / 2 ∧ x = (α - β) / 2} :=
by
  sorry

end range_alpha_minus_beta_over_2_l252_252179


namespace shorter_leg_of_right_triangle_l252_252804

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252804


namespace min_balls_to_draw_l252_252144

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end min_balls_to_draw_l252_252144


namespace problem_1_problem_2_l252_252761

-- Problem 1: Prove that (\frac{1}{5} - \frac{2}{3} - \frac{3}{10}) × (-60) = 46
theorem problem_1 : (1/5 - 2/3 - 3/10) * -60 = 46 := by
  sorry

-- Problem 2: Prove that (-1)^{2024} + 24 ÷ (-2)^3 - 15^2 × (1/15)^2 = -3
theorem problem_2 : (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 := by
  sorry

end problem_1_problem_2_l252_252761


namespace circle_equation_l252_252185

open Real

variable {x y : ℝ}

theorem circle_equation (a : ℝ) (h_a_positive : a > 0) 
    (h_tangent : abs (3 * a + 4) / sqrt (3^2 + 4^2) = 2) :
    (∀ x y : ℝ, (x - a)^2 + y^2 = 4) := sorry

end circle_equation_l252_252185


namespace age_ratio_rahul_deepak_l252_252556

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end age_ratio_rahul_deepak_l252_252556


namespace expand_expression_l252_252169

theorem expand_expression (x y : ℝ) : 
  (16 * x + 18 - 7 * y) * (3 * x) = 48 * x^2 + 54 * x - 21 * x * y :=
by
  sorry

end expand_expression_l252_252169


namespace chameleons_changed_color_l252_252229

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l252_252229


namespace chameleon_color_change_l252_252218

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l252_252218


namespace students_selected_milk_is_54_l252_252234

-- Define the parameters.
variable (total_students : ℕ)
variable (students_selected_soda students_selected_milk : ℕ)

-- Given conditions.
axiom h1 : students_selected_soda = 90
axiom h2 : students_selected_soda = (1 / 2) * total_students
axiom h3 : students_selected_milk = (3 / 5) * students_selected_soda

-- Prove that the number of students who selected milk is equal to 54.
theorem students_selected_milk_is_54 : students_selected_milk = 54 :=
by
  sorry

end students_selected_milk_is_54_l252_252234


namespace inequality_solution_l252_252017

theorem inequality_solution (x : ℝ) :
  (2 * x^2 - 4 * x - 70 > 0) ∧ (x ≠ -2) ∧ (x ≠ 0) ↔ (x < -5 ∨ x > 7) :=
by
  sorry

end inequality_solution_l252_252017


namespace kenya_peanuts_l252_252519

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l252_252519


namespace valid_digit_cancel_fractions_l252_252895

def digit_cancel_fraction (a b c d : ℕ) : Prop :=
  10 * a + b == 0 ∧ 10 * c + d == 0 ∧ 
  (b == d ∨ b == c ∨ a == d ∨ a == c) ∧
  (b ≠ a ∨ d ≠ c) ∧
  ((10 * a + b) ≠ (10 * c + d)) ∧
  ((10 * a + b) * d == (10 * c + d) * a)

theorem valid_digit_cancel_fractions :
  ∀ (a b c d : ℕ), 
  digit_cancel_fraction a b c d → 
  (10 * a + b == 26 ∧ 10 * c + d == 65) ∨
  (10 * a + b == 16 ∧ 10 * c + d == 64) ∨
  (10 * a + b == 19 ∧ 10 * c + d == 95) ∨
  (10 * a + b == 49 ∧ 10 * c + d == 98) :=
by {sorry}

end valid_digit_cancel_fractions_l252_252895


namespace value_of_a_l252_252782

theorem value_of_a (a : ℝ) (A B : ℝ × ℝ) (hA : A = (a - 2, 2 * a + 7)) (hB : B = (1, 5)) (h_parallel : (A.1 = B.1)) : a = 3 :=
by {
  sorry
}

end value_of_a_l252_252782


namespace axis_of_symmetry_cosine_l252_252545

theorem axis_of_symmetry_cosine (x : ℝ) : 
  (∃ k : ℤ, 2 * x + π / 3 = k * π) → x = -π / 6 :=
sorry

end axis_of_symmetry_cosine_l252_252545


namespace xyz_poly_identity_l252_252845

theorem xyz_poly_identity (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
  (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
  (x^6 + y^6 + z^6) / (xyz * (xy + xz + yz)) = 6 :=
by
  sorry

end xyz_poly_identity_l252_252845


namespace intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l252_252347

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.sqrt (x + 2))) + Real.log (3 - x)
def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 1 - m < x ∧ x < 3 * m - 1 }

theorem intersection_of_A_and_B_is_B_implies_m_leq_4_over_3 (m : ℝ) 
    (h : A ∩ B m = B m) : m ≤ 4 / 3 := by
  sorry

end intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l252_252347


namespace compare_combined_sums_l252_252008

def numeral1 := 7524258
def numeral2 := 523625072

def place_value_2_numeral1 := 200000 + 20
def place_value_5_numeral1 := 50000 + 500
def combined_sum_numeral1 := place_value_2_numeral1 + place_value_5_numeral1

def place_value_2_numeral2 := 200000000 + 20
def place_value_5_numeral2 := 500000 + 50
def combined_sum_numeral2 := place_value_2_numeral2 + place_value_5_numeral2

def difference := combined_sum_numeral2 - combined_sum_numeral1

theorem compare_combined_sums :
  difference = 200249550 := by
  sorry

end compare_combined_sums_l252_252008


namespace solve_sqrt_eq_l252_252022

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 3) ↔ (x = 2 ∨ x = -2) := 
by sorry

end solve_sqrt_eq_l252_252022


namespace determine_y_l252_252016

theorem determine_y : 
  ∀ y : ℝ, 
    (2 * Real.arctan (1 / 5) + Real.arctan (1 / 25) + Real.arctan (1 / y) = Real.pi / 4) -> 
    y = -121 / 60 :=
by
  sorry

end determine_y_l252_252016


namespace greatest_even_integer_leq_z_l252_252970

theorem greatest_even_integer_leq_z (z : ℝ) (z_star : ℝ → ℝ)
  (h1 : ∀ z, z_star z = z_star (z - (z - z_star z))) -- (This is to match the definition given)
  (h2 : 6.30 - z_star 6.30 = 0.2999999999999998) : z_star 6.30 ≤ 6.30 := by
sorry

end greatest_even_integer_leq_z_l252_252970


namespace binom_12_6_l252_252931

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l252_252931


namespace arithmetic_sequence_n_l252_252781

theorem arithmetic_sequence_n (a1 d an n : ℕ) (h1 : a1 = 1) (h2 : d = 3) (h3 : an = 298) (h4 : an = a1 + (n - 1) * d) : n = 100 :=
by
  sorry

end arithmetic_sequence_n_l252_252781


namespace find_rate_percent_l252_252293

-- Define the conditions based on the problem statement
def principal : ℝ := 800
def simpleInterest : ℝ := 160
def time : ℝ := 5

-- Create the statement to prove the rate percent
theorem find_rate_percent : ∃ (rate : ℝ), simpleInterest = (principal * rate * time) / 100 := sorry

end find_rate_percent_l252_252293


namespace number_of_bushes_l252_252239

theorem number_of_bushes (T B x y : ℕ) (h1 : B = T - 6) (h2 : x ≥ y + 10) (h3 : T * x = 128) (hT_pos : T > 0) (hx_pos : x > 0) : B = 2 :=
sorry

end number_of_bushes_l252_252239


namespace pies_left_l252_252251

theorem pies_left (pies_per_batch : ℕ) (batches : ℕ) (dropped : ℕ) (total_pies : ℕ) (pies_left : ℕ)
  (h1 : pies_per_batch = 5)
  (h2 : batches = 7)
  (h3 : dropped = 8)
  (h4 : total_pies = pies_per_batch * batches)
  (h5 : pies_left = total_pies - dropped) :
  pies_left = 27 := by
  sorry

end pies_left_l252_252251


namespace fraction_product_l252_252314

theorem fraction_product :
  (2 / 3) * (5 / 7) * (9 / 11) * (4 / 13) = 360 / 3003 := by
  sorry

end fraction_product_l252_252314


namespace average_height_corrected_l252_252422

theorem average_height_corrected (students : ℕ) (incorrect_avg_height : ℝ) (incorrect_height : ℝ) (actual_height : ℝ)
  (h1 : students = 20)
  (h2 : incorrect_avg_height = 175)
  (h3 : incorrect_height = 151)
  (h4 : actual_height = 111) :
  (incorrect_avg_height * students - incorrect_height + actual_height) / students = 173 :=
by
  sorry

end average_height_corrected_l252_252422


namespace division_quotient_proof_l252_252705

theorem division_quotient_proof (x : ℕ) (larger_number : ℕ) (h1 : larger_number - x = 1365)
    (h2 : larger_number = 1620) (h3 : larger_number % x = 15) : larger_number / x = 6 :=
by
  sorry

end division_quotient_proof_l252_252705


namespace maximum_distinct_numbers_l252_252488

theorem maximum_distinct_numbers (n : ℕ) (hsum : n = 250) : 
  ∃ k ≤ 21, k = 21 :=
by
  sorry

end maximum_distinct_numbers_l252_252488


namespace tan_theta_expr_l252_252983

theorem tan_theta_expr (θ : ℝ) (h : Real.tan θ = 4) : 
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
by sorry

end tan_theta_expr_l252_252983


namespace wheat_flour_one_third_l252_252910

theorem wheat_flour_one_third (recipe_cups: ℚ) (third_recipe: ℚ) 
  (h1: recipe_cups = 5 + 2 / 3) (h2: third_recipe = recipe_cups / 3) :
  third_recipe = 1 + 8 / 9 :=
by
  sorry

end wheat_flour_one_third_l252_252910


namespace edric_hours_per_day_l252_252767

/--
Edric's monthly salary is $576. He works 6 days a week for 4 weeks in a month and 
his hourly rate is $3. Prove that Edric works 8 hours in a day.
-/
theorem edric_hours_per_day (m : ℕ) (r : ℕ) (d : ℕ) (w : ℕ)
  (h_m : m = 576) (h_r : r = 3) (h_d : d = 6) (h_w : w = 4) :
  (m / r) / (d * w) = 8 := by
    sorry

end edric_hours_per_day_l252_252767


namespace unique_n_for_prime_p_l252_252538

theorem unique_n_for_prime_p (p : ℕ) (hp1 : p > 2) (hp2 : Nat.Prime p) :
  ∃! (n : ℕ), (∃ (k : ℕ), n^2 + n * p = k^2) ∧ n = (p - 1) / 2 ^ 2 :=
sorry

end unique_n_for_prime_p_l252_252538


namespace smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l252_252737

theorem smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450 :
  ∃ n : ℕ, (n - 10) % 12 = 0 ∧
           (n - 10) % 16 = 0 ∧
           (n - 10) % 18 = 0 ∧
           (n - 10) % 21 = 0 ∧
           (n - 10) % 28 = 0 ∧
           (n - 10) % 35 = 0 ∧
           (n - 10) % 40 = 0 ∧
           (n - 10) % 45 = 0 ∧
           (n - 10) % 55 = 0 ∧
           n = 55450 :=
by
  sorry

end smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l252_252737


namespace totalPayment_l252_252564

def totalNumberOfTrees : Nat := 850
def pricePerDouglasFir : Nat := 300
def pricePerPonderosaPine : Nat := 225
def numberOfDouglasFirPurchased : Nat := 350
def numberOfPonderosaPinePurchased := totalNumberOfTrees - numberOfDouglasFirPurchased

def costDouglasFir := numberOfDouglasFirPurchased * pricePerDouglasFir
def costPonderosaPine := numberOfPonderosaPinePurchased * pricePerPonderosaPine

def totalCost := costDouglasFir + costPonderosaPine

theorem totalPayment : totalCost = 217500 := by
  sorry

end totalPayment_l252_252564


namespace rectangular_prism_length_l252_252718

theorem rectangular_prism_length (w l h : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : h = 3 * w) 
  (h3 : 4 * l + 4 * w + 4 * h = 256) : 
  l = 32 :=
by
  sorry

end rectangular_prism_length_l252_252718


namespace shorter_leg_of_right_triangle_l252_252808

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252808


namespace wire_length_ratio_l252_252001

open Real

noncomputable def bonnie_wire_length : ℝ := 12 * 8
noncomputable def bonnie_cube_volume : ℝ := 8^3
noncomputable def roark_unit_cube_volume : ℝ := 2^3
noncomputable def roark_number_of_cubes : ℝ := bonnie_cube_volume / roark_unit_cube_volume
noncomputable def roark_wire_length_per_cube : ℝ := 12 * 2
noncomputable def roark_total_wire_length : ℝ := roark_number_of_cubes * roark_wire_length_per_cube
noncomputable def bonnie_to_roark_wire_ratio := bonnie_wire_length / roark_total_wire_length

theorem wire_length_ratio : bonnie_to_roark_wire_ratio = (1 : ℝ) / 16 :=
by
  sorry

end wire_length_ratio_l252_252001


namespace third_box_weight_l252_252909

def box1_height := 1 -- inches
def box1_width := 2 -- inches
def box1_length := 4 -- inches
def box1_weight := 30 -- grams

def box2_height := 3 * box1_height
def box2_width := 2 * box1_width
def box2_length := box1_length

def box3_height := box2_height
def box3_width := box2_width / 2
def box3_length := box2_length

def volume (height : ℕ) (width : ℕ) (length : ℕ) : ℕ := height * width * length

def weight (box1_weight : ℕ) (box1_volume : ℕ) (box3_volume : ℕ) : ℕ := 
  box3_volume / box1_volume * box1_weight

theorem third_box_weight :
  weight box1_weight (volume box1_height box1_width box1_length) 
  (volume box3_height box3_width box3_length) = 90 :=
by
  sorry

end third_box_weight_l252_252909


namespace crayons_left_l252_252397

theorem crayons_left (start_crayons lost_crayons left_crayons : ℕ) 
  (h1 : start_crayons = 479) 
  (h2 : lost_crayons = 345) 
  (h3 : left_crayons = start_crayons - lost_crayons) : 
  left_crayons = 134 :=
sorry

end crayons_left_l252_252397


namespace profit_amount_l252_252748

-- Conditions: Selling Price and Profit Percentage
def SP : ℝ := 850
def P_percent : ℝ := 37.096774193548384

-- Theorem: The profit amount is $230
theorem profit_amount : (SP / (1 + P_percent / 100)) * P_percent / 100 = 230 := by
  -- sorry will be replaced with the proof
  sorry

end profit_amount_l252_252748


namespace ab2_plus_bc2_plus_ca2_le_27_div_8_l252_252660

theorem ab2_plus_bc2_plus_ca2_le_27_div_8 (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end ab2_plus_bc2_plus_ca2_le_27_div_8_l252_252660


namespace people_lineup_l252_252061

/-- Theorem: Given five people, where the youngest person cannot be on the first or last position, 
we want to prove that there are exactly 72 ways to arrange them in a straight line. -/
theorem people_lineup (p : Fin 5 → ℕ) 
  (hy : ∃ i : Fin 5, ∀ j : Fin 5, i ≠ j → p i < p j) 
  (h_pos : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4)
  : (∑ x in ({1, 2, 3, 4} : Finset (Fin 5)), 4 * 3 * 2 * 1) = 72 := by
  -- The proof is omitted.
  sorry

end people_lineup_l252_252061


namespace correct_calculation_is_d_l252_252416

theorem correct_calculation_is_d :
  (-7) + (-7) ≠ 0 ∧
  ((-1 / 10) - (1 / 10)) ≠ 0 ∧
  (0 + (-101)) ≠ 101 ∧
  (1 / 3 + -1 / 2 = -1 / 6) :=
by
  sorry

end correct_calculation_is_d_l252_252416


namespace closed_broken_line_impossible_l252_252241

theorem closed_broken_line_impossible (n : ℕ) (h : n = 1989) : ¬ (∃ a b : ℕ, 2 * (a + b) = n) :=
by {
  sorry
}

end closed_broken_line_impossible_l252_252241


namespace simplify_expression_1_simplify_expression_2_l252_252453

theorem simplify_expression_1 (x y : ℝ) :
  x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y :=
sorry

theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b :=
sorry

end simplify_expression_1_simplify_expression_2_l252_252453


namespace determine_a_l252_252540

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem determine_a (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end determine_a_l252_252540


namespace kenya_peanuts_correct_l252_252514

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l252_252514


namespace chameleon_color_change_l252_252224

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l252_252224


namespace bottles_left_on_shelf_l252_252112

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l252_252112


namespace francine_leave_time_earlier_l252_252658

-- Definitions for the conditions in the problem
def leave_time := "noon"  -- Francine and her father leave at noon every day.
def father_meet_time_shorten := 10  -- They arrived home 10 minutes earlier than usual.
def francine_walk_duration := 15  -- Francine walked for 15 minutes.

-- Premises based on the conditions
def usual_meet_time := 12 * 60  -- Meeting time in minutes from midnight (noon = 720 minutes)
def special_day_meet_time := usual_meet_time - father_meet_time_shorten / 2  -- 5 minutes earlier

-- The main theorem to prove: Francine leaves at 11:40 AM (700 minutes from midnight)
theorem francine_leave_time_earlier :
  usual_meet_time - (father_meet_time_shorten / 2 + francine_walk_duration) = (11 * 60 + 40) := by
  sorry

end francine_leave_time_earlier_l252_252658


namespace three_digit_numbers_containing_2_and_exclude_6_l252_252361

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end three_digit_numbers_containing_2_and_exclude_6_l252_252361


namespace shorter_leg_of_right_triangle_l252_252809

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252809


namespace find_d_l252_252863

theorem find_d (a b c d : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (hd : 1 < d) 
  (h_eq : ∀ M : ℝ, M ≠ 1 → (M^(1/a)) * (M^(1/(a * b))) * (M^(1/(a * b * c))) * (M^(1/(a * b * c * d))) = M^(17/24)) : d = 8 :=
sorry

end find_d_l252_252863


namespace shorter_leg_of_right_triangle_l252_252812

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252812


namespace complex_expression_evaluation_l252_252387

theorem complex_expression_evaluation (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := 
sorry

end complex_expression_evaluation_l252_252387


namespace max_value_x2_y3_z_l252_252846

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  if x + y + z = 3 then x^2 * y^3 * z else 0

theorem max_value_x2_y3_z
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x + y + z = 3) :
  maximum_value x y z ≤ 9 / 16 := sorry

end max_value_x2_y3_z_l252_252846


namespace no_solution_equation_l252_252599

theorem no_solution_equation (x : ℝ) : (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1 :=
  sorry

end no_solution_equation_l252_252599


namespace both_dice_3_given_one_dice_3_l252_252491

noncomputable def both_dice_3_probability_given_one_is_3 : ℚ := 1 / 11

theorem both_dice_3_given_one_dice_3 :
  let S := ({1, 2, 3, 4, 5, 6} : set ℕ × {1, 2, 3, 4, 5, 6}) in
  let outcomes := { (d1, d2) ∈ S | d1 = 3 ∨ d2 = 3 } in
  let favorable := { (3, 3) } in
  outcomes ≠ ∅ → 
  (favorable.finite.to_finset.card : ℚ) / (outcomes.finite.to_finset.card : ℚ) = both_dice_3_probability_given_one_is_3 :=
by
  sorry

end both_dice_3_given_one_dice_3_l252_252491


namespace binomial_inequality_l252_252184

theorem binomial_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end binomial_inequality_l252_252184


namespace part1_part2_l252_252590

-- Part 1
theorem part1 : real.cbrt 8 + abs (-5 : ℤ) + (-1 : ℤ) ^ 2023 = 6 := by
  sorry

-- Part 2
theorem part2 : ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (y = 1) ∧ (y = 2 * 2 + 1) := by
  sorry

end part1_part2_l252_252590


namespace powerjet_pumps_250_gallons_in_30_minutes_l252_252871

theorem powerjet_pumps_250_gallons_in_30_minutes :
  let rate : ℝ := 500
  let time_in_hours : ℝ := 1 / 2
  rate * time_in_hours = 250 :=
by
  sorry

end powerjet_pumps_250_gallons_in_30_minutes_l252_252871


namespace find_k_for_perfect_square_trinomial_l252_252714

noncomputable def perfect_square_trinomial (k : ℝ) : Prop :=
∀ x : ℝ, (x^2 - 8*x + k) = (x - 4)^2

theorem find_k_for_perfect_square_trinomial :
  ∃ k : ℝ, perfect_square_trinomial k ∧ k = 16 :=
by
  use 16
  sorry

end find_k_for_perfect_square_trinomial_l252_252714


namespace sum_of_digits_l252_252676

-- Conditions setup
variables (a b c d : ℕ)
variables (h1 : a + c = 10) 
variables (h2 : b + c = 9) 
variables (h3 : a + d = 10)
variables (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

theorem sum_of_digits : a + b + c + d = 19 :=
sorry

end sum_of_digits_l252_252676


namespace playground_width_l252_252568

open Nat

theorem playground_width (garden_width playground_length perimeter_garden : ℕ) (garden_area_eq_playground_area : Bool) :
  garden_width = 8 →
  playground_length = 16 →
  perimeter_garden = 64 →
  garden_area_eq_playground_area →
  ∃ (W : ℕ), W = 12 :=
by
  intros h_t1 h_t2 h_t3 h_t4
  sorry

end playground_width_l252_252568


namespace problem1_problem2_l252_252597

-- Define the conditions for Problem (1)
def cuberoot_8 := real.cbrt 8
def abs_neg_5 := abs (-5)
def pow_neg_1 := (-1 : ℤ) ^ 2023

-- Define the proof problem for Problem (1)
theorem problem1 : cuberoot_8 + abs_neg_5 + (pow_neg_1 : ℝ) = 6 := 
by sorry

-- Define the conditions for Problem (2)
def point1 := (0, 1)
def point2 := (2, 5)

-- Define the equation of the line
def line (k b : ℝ) (x : ℝ) := k * x + b

-- Prove that the line passing through the given points has the form y = 2x + 1
theorem problem2 (k b : ℝ) : 
  line k b 0 = (point1.snd : ℝ) ∧ line k b 2 = (point2.snd : ℝ) → 
  line 2 1 = 2 * point2.fst + 1 := 
by sorry

end problem1_problem2_l252_252597


namespace marked_price_l252_252151

theorem marked_price (original_price : ℝ) 
                     (discount1_rate : ℝ) 
                     (profit_rate : ℝ) 
                     (discount2_rate : ℝ)
                     (marked_price : ℝ) : 
                     original_price = 40 → 
                     discount1_rate = 0.15 → 
                     profit_rate = 0.25 → 
                     discount2_rate = 0.10 → 
                     marked_price = 47.20 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end marked_price_l252_252151


namespace half_angle_in_second_quadrant_l252_252388

theorem half_angle_in_second_quadrant (α : Real) (h1 : 180 < α ∧ α < 270)
        (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
        90 < α / 2 ∧ α / 2 < 180 :=
sorry

end half_angle_in_second_quadrant_l252_252388


namespace trivia_team_students_per_group_l252_252281

theorem trivia_team_students_per_group (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58) (h2 : not_picked = 10) (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 :=
by
  sorry

end trivia_team_students_per_group_l252_252281


namespace lemonade_water_quarts_l252_252647

theorem lemonade_water_quarts :
  let ratioWaterLemon := (4 : ℕ) / (1 : ℕ)
  let totalParts := 4 + 1
  let totalVolumeInGallons := 3
  let quartsPerGallon := 4
  let totalVolumeInQuarts := totalVolumeInGallons * quartsPerGallon
  let volumePerPart := totalVolumeInQuarts / totalParts
  let volumeWater := 4 * volumePerPart
  volumeWater = 9.6 :=
by
  -- placeholder for actual proof
  sorry

end lemonade_water_quarts_l252_252647


namespace line_symmetric_about_y_eq_x_l252_252343

-- Define the line equation types and the condition for symmetry
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Conditions given
variable (a b c : ℝ)
variable (h_ab_pos : a * b > 0)

-- Definition of the problem in Lean
theorem line_symmetric_about_y_eq_x (h_bisector : ∀ x y : ℝ, line_equation a b c x y ↔ line_equation b a c y x) : 
  ∀ x y : ℝ, line_equation b a c x y := by
  sorry

end line_symmetric_about_y_eq_x_l252_252343


namespace binomial_12_6_l252_252926

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l252_252926


namespace max_base_angle_is_7_l252_252375

-- Define the conditions and the problem statement
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isosceles_triangle (x : ℕ) : Prop :=
  is_prime x ∧ ∃ y : ℕ, 2 * x + y = 180 ∧ is_prime y

theorem max_base_angle_is_7 :
  ∃ (x : ℕ), isosceles_triangle x ∧ x = 7 :=
by
  sorry

end max_base_angle_is_7_l252_252375


namespace parallel_lines_slope_eq_l252_252629

theorem parallel_lines_slope_eq (k : ℝ) : (∀ x : ℝ, 3 = 6 * k) → k = 1 / 2 :=
by
  intro h
  sorry

end parallel_lines_slope_eq_l252_252629


namespace total_food_pounds_l252_252655

theorem total_food_pounds (chicken hamburger hot_dogs sides : ℕ) 
  (h1 : chicken = 16) 
  (h2 : hamburger = chicken / 2) 
  (h3 : hot_dogs = hamburger + 2) 
  (h4 : sides = hot_dogs / 2) : 
  chicken + hamburger + hot_dogs + sides = 39 := 
  by 
    sorry

end total_food_pounds_l252_252655


namespace priyas_fathers_age_l252_252537

-- Define Priya's age P and her father's age F
variables (P F : ℕ)

-- Define the conditions
def conditions : Prop :=
  F - P = 31 ∧ P + F = 53

-- Define the theorem to be proved
theorem priyas_fathers_age (h : conditions P F) : F = 42 :=
sorry

end priyas_fathers_age_l252_252537


namespace a_greater_than_b_c_less_than_a_l252_252130

-- Condition 1: Definition of box dimensions
def Box := (Nat × Nat × Nat)

-- Condition 2: Dimension comparisons
def le_box (a b : Box) : Prop :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  (a1 ≤ b1 ∨ a1 ≤ b2 ∨ a1 ≤ b3) ∧ (a2 ≤ b1 ∨ a2 ≤ b2 ∨ a2 ≤ b3) ∧ (a3 ≤ b1 ∨ a3 ≤ b2 ∨ a3 ≤ b3)

def lt_box (a b : Box) : Prop := le_box a b ∧ ¬(a = b)

-- Condition 3: Box dimensions
def A : Box := (6, 5, 3)
def B : Box := (5, 4, 1)
def C : Box := (3, 2, 2)

-- Equivalent Problem 1: Prove A > B
theorem a_greater_than_b : lt_box B A :=
by
  -- theorem proof here
  sorry

-- Equivalent Problem 2: Prove C < A
theorem c_less_than_a : lt_box C A :=
by
  -- theorem proof here
  sorry

end a_greater_than_b_c_less_than_a_l252_252130


namespace find_cos_A_l252_252999

theorem find_cos_A
  (A C : ℝ)
  (AB CD : ℝ)
  (AD BC : ℝ)
  (α : ℝ)
  (h1 : A = C)
  (h2 : AB = 150)
  (h3 : CD = 150)
  (h4 : AD ≠ BC)
  (h5 : AB + BC + CD + AD = 560)
  (h6 : A = α)
  (h7 : C = α)
  (BD₁ BD₂ : ℝ)
  (h8 : BD₁^2 = AD^2 + 150^2 - 2 * 150 * AD * Real.cos α)
  (h9 : BD₂^2 = BC^2 + 150^2 - 2 * 150 * BC * Real.cos α)
  (h10 : BD₁ = BD₂) :
  Real.cos A = 13 / 15 := 
sorry

end find_cos_A_l252_252999


namespace product_of_roots_eq_neg30_l252_252366

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end product_of_roots_eq_neg30_l252_252366


namespace dryer_runtime_per_dryer_l252_252308

-- Definitions for the given conditions
def washer_cost : ℝ := 4
def dryer_cost_per_10min : ℝ := 0.25
def loads_of_laundry : ℕ := 2
def num_dryers : ℕ := 3
def total_spent : ℝ := 11

-- Statement to prove
theorem dryer_runtime_per_dryer : 
  (2 * washer_cost + ((total_spent - 2 * washer_cost) / dryer_cost_per_10min) * 10) / num_dryers = 40 :=
by
  sorry

end dryer_runtime_per_dryer_l252_252308


namespace find_certain_number_l252_252701

def certain_number (x : ℤ) : Prop := x - 9 = 5

theorem find_certain_number (x : ℤ) (h : certain_number x) : x = 14 :=
by
  sorry

end find_certain_number_l252_252701


namespace common_speed_is_10_l252_252835

noncomputable def speed_jack (x : ℝ) : ℝ := x^2 - 11 * x - 22
noncomputable def speed_jill (x : ℝ) : ℝ := 
  if x = -6 then 0 else (x^2 - 4 * x - 12) / (x + 6)

theorem common_speed_is_10 (x : ℝ) (h : speed_jack x = speed_jill x) (hx : x = 16) : 
  speed_jack x = 10 :=
by
  sorry

end common_speed_is_10_l252_252835


namespace binomial_12_6_eq_924_l252_252938

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l252_252938


namespace four_digit_multiples_of_7_l252_252352

theorem four_digit_multiples_of_7 : 
  (card { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 7 = 0 }) = 1286 :=
sorry

end four_digit_multiples_of_7_l252_252352


namespace shorter_leg_of_right_triangle_l252_252807

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252807


namespace min_value_of_x_plus_y_l252_252987

theorem min_value_of_x_plus_y (x y : ℝ) (h1: y ≠ 0) (h2: 1 / y = (x - 1) / 2) : x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_of_x_plus_y_l252_252987


namespace number_plus_273_l252_252259

theorem number_plus_273 (x : ℤ) (h : x - 477 = 273) : x + 273 = 1023 := by
  sorry

end number_plus_273_l252_252259


namespace minimum_value_of_2a5_a4_l252_252058

variable {a : ℕ → ℝ} {q : ℝ}

-- Defining that the given sequence is geometric, i.e., a_{n+1} = a_n * q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

-- The condition given in the problem is
def condition (a : ℕ → ℝ) : Prop :=
2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

-- The sequence is positive
def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

theorem minimum_value_of_2a5_a4 (h_geom : is_geometric_sequence a q) (h_cond : condition a) (h_pos : positive_sequence a) (h_q : q > 0) :
  2 * a 5 + a 4 = 12 * Real.sqrt 3 :=
sorry

end minimum_value_of_2a5_a4_l252_252058


namespace ellipse_foci_distance_l252_252326

noncomputable def distance_between_foci : ℝ := 2 * Real.sqrt 29

theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 
  (Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25) → 
  distance_between_foci = 2 * Real.sqrt 29 := 
by
  intros x y h
  -- proof goes here (skipped)
  sorry

end ellipse_foci_distance_l252_252326


namespace shifted_parabola_sum_l252_252733

theorem shifted_parabola_sum :
  let f (x : ℝ) := 3 * x^2 - 2 * x + 5
  let g (x : ℝ) := 3 * (x - 3)^2 - 2 * (x - 3) + 5
  let a := 3
  let b := -20
  let c := 38
  a + b + c = 21 :=
by
  sorry

end shifted_parabola_sum_l252_252733


namespace correct_factorization_l252_252638

theorem correct_factorization (a b : ℝ) : 
  ((x + 6) * (x - 1) = x^2 + 5 * x - 6) →
  ((x - 2) * (x + 1) = x^2 - x - 2) →
  (a = 1 ∧ b = -6) →
  (x^2 - x - 6 = (x + 2) * (x - 3)) :=
sorry

end correct_factorization_l252_252638


namespace smallest_digit_to_make_divisible_by_9_l252_252032

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l252_252032


namespace equal_sharing_l252_252573

theorem equal_sharing (total_cards friends : ℕ) (h1 : total_cards = 455) (h2 : friends = 5) : total_cards / friends = 91 := by
  sorry

end equal_sharing_l252_252573


namespace chameleon_color_change_l252_252226

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l252_252226


namespace min_value_a_l252_252502

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_a_l252_252502


namespace right_triangle_shorter_leg_l252_252830

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l252_252830


namespace Thabo_books_problem_l252_252870

theorem Thabo_books_problem 
  (P F : ℕ)
  (H1 : 180 = F + P + 30)
  (H2 : F = 2 * P)
  (H3 : P > 30) :
  P - 30 = 20 := 
sorry

end Thabo_books_problem_l252_252870


namespace large_integer_value_l252_252628

theorem large_integer_value :
  (2 + 3) * (2^2 + 3^2) * (2^4 - 3^4) * (2^8 + 3^8) * (2^16 - 3^16) * (2^32 + 3^32) * (2^64 - 3^64)
  > 0 := 
by
  sorry

end large_integer_value_l252_252628


namespace initial_black_beads_l252_252763

theorem initial_black_beads (B : ℕ) : 
  let white_beads := 51
  let black_beads_removed := 1 / 6 * B
  let white_beads_removed := 1 / 3 * white_beads
  let total_beads_removed := 32
  white_beads_removed + black_beads_removed = total_beads_removed →
  B = 90 :=
by
  sorry

end initial_black_beads_l252_252763


namespace trinomial_identity_l252_252731

theorem trinomial_identity :
  let a := 23
  let b := 15
  let c := 7
  (a + b + c)^2 - (a^2 + b^2 + c^2) = 1222 :=
by
  let a := 23
  let b := 15
  let c := 7
  sorry

end trinomial_identity_l252_252731


namespace unit_circle_inequality_l252_252295

theorem unit_circle_inequality 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (habcd : a * b + c * d = 1) 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (hx1 : x1^2 + y1^2 = 1)
  (hx2 : x2^2 + y2^2 = 1)
  (hx3 : x3^2 + y3^2 = 1)
  (hx4 : x4^2 + y4^2 = 1) :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2 ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
sorry

end unit_circle_inequality_l252_252295


namespace problem_conditions_l252_252494

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end problem_conditions_l252_252494


namespace cos_alpha_minus_2pi_l252_252778

open Real

noncomputable def problem_statement (alpha : ℝ) : Prop :=
  (sin (π + alpha) = 4 / 5) ∧ (cos (alpha - 2 * π) = 3 / 5)

theorem cos_alpha_minus_2pi (alpha : ℝ) (h1 : sin (π + alpha) = 4 / 5) (quad4 : cos alpha > 0 ∧ sin alpha < 0) :
  cos (alpha - 2 * π) = 3 / 5 :=
sorry

end cos_alpha_minus_2pi_l252_252778


namespace binom_12_6_eq_924_l252_252943

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l252_252943


namespace cups_of_oil_used_l252_252686

-- Define the required amounts
def total_liquid : ℝ := 1.33
def water_used : ℝ := 1.17

-- The statement we want to prove
theorem cups_of_oil_used : total_liquid - water_used = 0.16 := by
sorry

end cups_of_oil_used_l252_252686


namespace sum_of_coefficients_l252_252673

-- Define the polynomial expansion and the target question
theorem sum_of_coefficients
  (x : ℝ)
  (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℝ)
  (h : (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + 
                        b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0) :
  (b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 729 :=
by {
  -- We substitute x = 1 and show that the polynomial equals 729
  sorry
}

end sum_of_coefficients_l252_252673


namespace infinite_solutions_c_l252_252630

theorem infinite_solutions_c (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 :=
sorry

end infinite_solutions_c_l252_252630


namespace question_correctness_l252_252415

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l252_252415


namespace class_boys_count_l252_252122

theorem class_boys_count
    (x y : ℕ)
    (h1 : x + y = 20)
    (h2 : (1 / 3 : ℚ) * x = (1 / 2 : ℚ) * y) :
    x = 12 :=
by
  sorry

end class_boys_count_l252_252122


namespace find_a_find_m_l252_252667

-- Definition of the odd function condition
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- The first proof problem
theorem find_a (a : ℝ) (h_odd : odd_function (fun x => Real.log (Real.exp x + a + 1))) : a = -1 :=
sorry

-- Definitions of the two functions involved in the second proof problem
noncomputable def f1 (x : ℝ) : ℝ :=
if x = 0 then 0 else Real.log x / x

noncomputable def f2 (x m : ℝ) : ℝ :=
x^2 - 2 * Real.exp 1 * x + m

-- The second proof problem
theorem find_m (m : ℝ) (h_root : ∃! x, f1 x = f2 x m) : m = Real.exp 2 + 1 / Real.exp 1 :=
sorry

end find_a_find_m_l252_252667


namespace calculate_total_cost_l252_252445

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end calculate_total_cost_l252_252445


namespace binomial_12_6_eq_924_l252_252937

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l252_252937


namespace poly_has_integer_roots_iff_a_eq_one_l252_252841

-- Definition: a positive real number
def pos_real (a : ℝ) : Prop := a > 0

-- The polynomial
def p (a : ℝ) (x : ℝ) : ℝ := a^3 * x^3 + a^2 * x^2 + a * x + a

-- The main theorem
theorem poly_has_integer_roots_iff_a_eq_one (a : ℝ) (x : ℤ) :
  (pos_real a ∧ ∃ x : ℤ, p a x = 0) ↔ a = 1 :=
by sorry

end poly_has_integer_roots_iff_a_eq_one_l252_252841


namespace number_of_multiples_of_4_between_100_and_350_l252_252989

theorem number_of_multiples_of_4_between_100_and_350 :
  (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≥ 104 ∧ (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≤ 348 →
  (set.filter (λ x, x % 4 = 0) (finset.Icc 100 350).to_set).card = 62 :=
by
  sorry

end number_of_multiples_of_4_between_100_and_350_l252_252989


namespace sequence_increasing_range_of_a_l252_252268

theorem sequence_increasing_range_of_a :
  ∀ {a : ℝ}, (∀ n : ℕ, 
    (n ≤ 7 → (4 - a) * n - 10 ≤ (4 - a) * (n + 1) - 10) ∧ 
    (7 < n → a^(n - 6) ≤ a^(n - 5))
  ) → 2 < a ∧ a < 4 :=
by
  sorry

end sequence_increasing_range_of_a_l252_252268


namespace diff_of_squares_l252_252618

variable {x y : ℝ}

theorem diff_of_squares : (x + y) * (x - y) = x^2 - y^2 := 
sorry

end diff_of_squares_l252_252618


namespace no_constant_term_l252_252666

theorem no_constant_term (n : ℕ) (hn : ∀ r : ℕ, ¬(n = (4 * r) / 3)) : n ≠ 8 :=
by 
  intro h
  sorry

end no_constant_term_l252_252666


namespace inequality_solution_exists_l252_252789

theorem inequality_solution_exists (a : ℝ) : 
  ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a := 
by
  sorry

end inequality_solution_exists_l252_252789


namespace sequence_divisible_by_13_l252_252359

theorem sequence_divisible_by_13 (n : ℕ) (h : n ≤ 1000) : 
  ∃ m, m = 165 ∧ ∀ k, 1 ≤ k ∧ k ≤ m → (10^(6*k) + 1) % 13 = 0 := 
sorry

end sequence_divisible_by_13_l252_252359


namespace binomial_12_6_l252_252927

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l252_252927


namespace smallest_positive_integer_l252_252729

def smallest_x (x : ℕ) : Prop :=
  (540 * x) % 800 = 0

theorem smallest_positive_integer (x : ℕ) : smallest_x x → x = 80 :=
by {
  sorry
}

end smallest_positive_integer_l252_252729


namespace determine_h_l252_252625

def h (x : ℝ) := -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1

theorem determine_h (x : ℝ) : 
  (12 * x^4 + 9 * x^3 - 3 * x + 1 + h x = 5 * x^3 - 8 * x^2 + 3) →
  h x = -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1 :=
by
  sorry

end determine_h_l252_252625


namespace julien_swims_50_meters_per_day_l252_252068

-- Definitions based on given conditions
def distance_julien_swims_per_day : ℕ := 50
def distance_sarah_swims_per_day (J : ℕ) : ℕ := 2 * J
def distance_jamir_swims_per_day (J : ℕ) : ℕ := distance_sarah_swims_per_day J + 20
def combined_distance_per_day (J : ℕ) : ℕ := J + distance_sarah_swims_per_day J + distance_jamir_swims_per_day J
def combined_distance_per_week (J : ℕ) : ℕ := 7 * combined_distance_per_day J

-- Proof statement 
theorem julien_swims_50_meters_per_day :
  combined_distance_per_week distance_julien_swims_per_day = 1890 :=
by
  -- We are formulating the proof without solving it, to be proven formally in Lean
  sorry

end julien_swims_50_meters_per_day_l252_252068


namespace find_a₁_l252_252187

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ n

noncomputable def sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

variables (a₁ q : ℝ)
-- Condition: The common ratio should not be 1.
axiom hq : q ≠ 1
-- Condition: Second term of the sequence a₂ = 1
axiom ha₂ : geometric_sequence a₁ q 1 = 1
-- Condition: 9S₃ = S₆
axiom hsum : 9 * sequence_sum a₁ q 3 = sequence_sum a₁ q 6

theorem find_a₁ : a₁ = 1 / 2 :=
  sorry

end find_a₁_l252_252187


namespace intersection_of_sets_l252_252195

def A := { x : ℝ | x^2 - 2 * x - 8 < 0 }
def B := { x : ℝ | x >= 0 }
def intersection := { x : ℝ | 0 <= x ∧ x < 4 }

theorem intersection_of_sets : (A ∩ B) = intersection := 
sorry

end intersection_of_sets_l252_252195


namespace find_divided_number_l252_252395

theorem find_divided_number :
  ∃ (Number : ℕ), ∃ (q r d : ℕ), q = 8 ∧ r = 3 ∧ d = 21 ∧ Number = d * q + r ∧ Number = 171 :=
by
  sorry

end find_divided_number_l252_252395


namespace garden_dimensions_l252_252751

theorem garden_dimensions (w l : ℕ) (h₁ : l = w + 3) (h₂ : 2 * (l + w) = 26) : w = 5 ∧ l = 8 :=
by
  sorry

end garden_dimensions_l252_252751


namespace A_completes_job_alone_l252_252605

theorem A_completes_job_alone (efficiency_B efficiency_A total_work days_A : ℝ) :
  efficiency_A = 1.3 * efficiency_B → 
  total_work = (efficiency_A + efficiency_B) * 13 → 
  days_A = total_work / efficiency_A → 
  days_A = 23 :=
by
  intros h1 h2 h3
  sorry

end A_completes_job_alone_l252_252605


namespace prime_cond_l252_252163

theorem prime_cond (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1) : 
  (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) → (p = 2 ∧ q = 5 ∧ n = 2) :=
  sorry

end prime_cond_l252_252163


namespace max_possible_median_l252_252543

/-- 
Given:
1. The Beverage Barn sold 300 cans of soda to 120 customers.
2. Every customer bought at least 1 can of soda but no more than 5 cans.
Prove that the maximum possible median number of cans of soda bought per customer is 5.
-/
theorem max_possible_median (total_cans : ℕ) (customers : ℕ) (min_can_per_customer : ℕ) (max_can_per_customer : ℕ) :
  total_cans = 300 ∧ customers = 120 ∧ min_can_per_customer = 1 ∧ max_can_per_customer = 5 →
  (∃ median : ℕ, median = 5) :=
by
  sorry

end max_possible_median_l252_252543


namespace range_of_abs_function_l252_252764

theorem range_of_abs_function:
  (∀ y, ∃ x : ℝ, y = |x + 3| - |x - 5|) → ∀ y, y ≤ 8 :=
by
  sorry

end range_of_abs_function_l252_252764


namespace min_sum_of_squares_l252_252844

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3 * b + 5 * c + 7 * d = 14) : 
  a^2 + b^2 + c^2 + d^2 ≥ 7 / 3 :=
sorry

end min_sum_of_squares_l252_252844


namespace middle_even_integer_l252_252275

theorem middle_even_integer (a b c : ℤ) (ha : even a) (hb : even b) (hc : even c) 
(h1 : a < b) (h2 : b < c) (h3 : 0 < a) (h4 : a < 10) (h5 : a + b + c = (1/8) * a * b * c) : b = 4 := 
sorry

end middle_even_integer_l252_252275


namespace num_comics_bought_l252_252086

def initial_comic_books : ℕ := 14
def current_comic_books : ℕ := 13
def comic_books_sold (initial : ℕ) : ℕ := initial / 2
def comics_bought (initial current : ℕ) : ℕ :=
  current - (initial - comic_books_sold initial)

theorem num_comics_bought :
  comics_bought initial_comic_books current_comic_books = 6 :=
by
  sorry

end num_comics_bought_l252_252086


namespace correct_model_is_pakistan_traditional_l252_252417

-- Given definitions
def hasPrimitiveModel (country : String) : Prop := country = "Nigeria"
def hasTraditionalModel (country : String) : Prop := country = "India" ∨ country = "Pakistan" ∨ country = "Nigeria"
def hasModernModel (country : String) : Prop := country = "China"

-- The proposition to prove
theorem correct_model_is_pakistan_traditional :
  (hasPrimitiveModel "Nigeria")
  ∧ (hasModernModel "China")
  ∧ (hasTraditionalModel "India")
  ∧ (hasTraditionalModel "Pakistan") →
  (hasTraditionalModel "Pakistan") := by
  intros h
  exact (h.right.right.right)

end correct_model_is_pakistan_traditional_l252_252417


namespace limit_of_derivative_l252_252344

variable {𝕜 : Type*} [NormedField 𝕜] {E : Type*} [NormedSpace 𝕜 E] {f : 𝕜 → E} {a A : 𝕜}

theorem limit_of_derivative (h : HasDerivAt f A a) : 
  filter.tendsto (λ Δx, (f (a + Δx) - f (a - Δx)) / Δx) (nhds_within 0 (set.Ioo (-(1 : 𝕜)) (1 : 𝕜))) (𝓝 (2 * A)) :=
sorry

end limit_of_derivative_l252_252344


namespace calculate_total_cost_l252_252444

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end calculate_total_cost_l252_252444


namespace bottles_left_on_shelf_l252_252119

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l252_252119


namespace probability_correct_l252_252204

open Finset

def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def pairs_with_sum_divisible_by_3 (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.bUnion (λ x, s.filter (λ y, x < y ∧ (x + y) % 3 = 0).image (λ y, (x, y)))

noncomputable def probability_sum_divisible_by_3 : ℚ :=
  (pairs_with_sum_divisible_by_3 first_ten_primes).card / (first_ten_primes.card.choose 2)

theorem probability_correct : probability_sum_divisible_by_3 = 1 / 5 := 
sorry

end probability_correct_l252_252204


namespace zero_point_in_interval_l252_252407

noncomputable def f (x a : ℝ) : ℝ := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : (∃ x ∈ Ioo 1 2, f x a = 0) ↔ 0 < a ∧ a < 3 := by
  sorry

end zero_point_in_interval_l252_252407


namespace complement_intersection_l252_252043

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2 * x > 0}

-- Define complement of A in U
def C_U_A : Set ℝ := U \ A

-- Define set B
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_intersection (x : ℝ) : x ∈ C_U_A ∩ B ↔ 1 < x ∧ x ≤ 2 :=
by
   sorry

end complement_intersection_l252_252043


namespace psychiatrist_problem_l252_252914

theorem psychiatrist_problem 
  (x : ℕ)
  (h_total : 4 * 8 + x + (x + 5) = 25)
  : x = 2 := by
  sorry

end psychiatrist_problem_l252_252914


namespace goldbach_conjecture_2024_l252_252765

-- Definitions for the problem
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement for the proof problem
theorem goldbach_conjecture_2024 :
  is_even 2024 ∧ 2024 > 2 → ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 2024 = p1 + p2 :=
by
  sorry

end goldbach_conjecture_2024_l252_252765


namespace gcd_lcm_sum_l252_252288

-- Define the given numbers
def a1 := 54
def b1 := 24
def a2 := 48
def b2 := 18

-- Define the GCD and LCM functions in Lean
def gcd_ab := Nat.gcd a1 b1
def lcm_cd := Nat.lcm a2 b2

-- Define the final sum
def final_sum := gcd_ab + lcm_cd

-- State the equality that represents the problem
theorem gcd_lcm_sum : final_sum = 150 := by
  sorry

end gcd_lcm_sum_l252_252288


namespace magnitude_of_linear_combination_is_sqrt_65_l252_252486

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, 3 * m - 2)
noncomputable def perpendicular (u v : ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 = 0)

theorem magnitude_of_linear_combination_is_sqrt_65 (m : ℝ) 
  (h_perpendicular : perpendicular (vector_a m) (vector_b m)) : 
  ‖((2 : ℝ) • (vector_a 1) - (3 : ℝ) • (vector_b 1))‖ = Real.sqrt 65 := 
by
  sorry

end magnitude_of_linear_combination_is_sqrt_65_l252_252486


namespace highest_score_not_necessarily_12_l252_252139

-- Define the structure of the round-robin tournament setup
structure RoundRobinTournament :=
  (teams : ℕ)
  (matches_per_team : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (points_draw : ℕ)

-- Tournament conditions
def tournament : RoundRobinTournament :=
  { teams := 12,
    matches_per_team := 11,
    points_win := 2,
    points_loss := 0,
    points_draw := 1 }

-- The statement we want to prove
theorem highest_score_not_necessarily_12 (T : RoundRobinTournament) :
  ∃ team_highest_score : ℕ, team_highest_score < 12 :=
by
  -- Provide a proof here
  sorry

end highest_score_not_necessarily_12_l252_252139


namespace dan_time_second_hour_tshirts_l252_252306

-- Definition of conditions
def t_shirts_in_first_hour (rate1 : ℕ) (time : ℕ) : ℕ := time / rate1
def total_t_shirts (hour1_ts hour2_ts : ℕ) : ℕ := hour1_ts + hour2_ts
def time_per_t_shirt_in_second_hour (time : ℕ) (hour2_ts : ℕ) : ℕ := time / hour2_ts

-- Main theorem statement (without proof)
theorem dan_time_second_hour_tshirts
  (rate1 : ℕ) (hour1_time : ℕ) (total_ts : ℕ) (hour_time : ℕ)
  (hour1_ts := t_shirts_in_first_hour rate1 hour1_time)
  (hour2_ts := total_ts - hour1_ts) :
  rate1 = 12 → 
  hour1_time = 60 → 
  total_ts = 15 → 
  hour_time = 60 →
  time_per_t_shirt_in_second_hour hour_time hour2_ts = 6 :=
by
  intros rate1_eq hour1_time_eq total_ts_eq hour_time_eq
  sorry

end dan_time_second_hour_tshirts_l252_252306


namespace binom_12_6_l252_252945

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l252_252945


namespace p_is_prime_and_gt_3_l252_252498

theorem p_is_prime_and_gt_3 (p : ℤ) (h1 : p > 3)
  (h2 : (p^2 + 15) % 12 = 4) : p.prime := 
by
  sorry

end p_is_prime_and_gt_3_l252_252498


namespace intersection_complement_l252_252044

open Set

-- Definitions from the problem
def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {y | 0 < y}

-- The proof statement
theorem intersection_complement : A ∩ (compl B) = Ioc (-1 : ℝ) 0 := by
  sorry

end intersection_complement_l252_252044


namespace tournament_committee_count_l252_252506

theorem tournament_committee_count :
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  total_choices = 11568055296 := 
by {
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  have h_total_choices_eq : total_choices = 11568055296 := sorry
  exact h_total_choices_eq
}

end tournament_committee_count_l252_252506


namespace inequality_solution_l252_252110

theorem inequality_solution (x : ℝ) : (1 - 3 * (x - 1) < x) ↔ (x > 1) :=
by sorry

end inequality_solution_l252_252110


namespace lisa_additional_marbles_l252_252393

theorem lisa_additional_marbles (n_friends : ℕ) (initial_marbles : ℕ) (h_friends : n_friends = 12) (h_marbles : initial_marbles = 50) :
  let total_marbles_needed := (n_friends * (n_friends + 1)) / 2 in
  total_marbles_needed - initial_marbles = 28 :=
by
  sorry

end lisa_additional_marbles_l252_252393


namespace minimum_b_value_l252_252039

theorem minimum_b_value (k : ℕ) (x y z b : ℕ) (h1 : x = 3 * k) (h2 : y = 4 * k)
  (h3 : z = 7 * k) (h4 : y = 15 * b - 5) (h5 : ∀ n : ℕ, n = 4 * k + 5 → n % 15 = 0) : 
  b = 3 :=
by
  sorry

end minimum_b_value_l252_252039


namespace necessary_but_not_sufficient_l252_252186

noncomputable def is_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 1 < f 2) → (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∨ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry

end necessary_but_not_sufficient_l252_252186


namespace chameleons_changed_color_l252_252233

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l252_252233


namespace ratio_of_still_lifes_to_portraits_l252_252757

noncomputable def total_paintings : ℕ := 80
noncomputable def portraits : ℕ := 16
noncomputable def still_lifes : ℕ := total_paintings - portraits
axiom still_lifes_is_multiple_of_portraits : ∃ k : ℕ, still_lifes = k * portraits

theorem ratio_of_still_lifes_to_portraits : still_lifes / portraits = 4 := by
  -- proof would go here
  sorry

end ratio_of_still_lifes_to_portraits_l252_252757


namespace middle_card_number_is_6_l252_252721

noncomputable def middle_card_number : ℕ :=
  6

theorem middle_card_number_is_6 (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 17)
  (casey_cannot_determine : ∀ (x : ℕ), (a = x) → ∃ (y z : ℕ), y ≠ z ∧ a + y + z = 17 ∧ a < y ∧ y < z)
  (tracy_cannot_determine : ∀ (x : ℕ), (c = x) → ∃ (y z : ℕ), y ≠ z ∧ y + z + c = 17 ∧ y < z ∧ z < c)
  (stacy_cannot_determine : ∀ (x : ℕ), (b = x) → ∃ (y z : ℕ), y ≠ z ∧ y + b + z = 17 ∧ y < b ∧ b < z) : 
  b = middle_card_number :=
sorry

end middle_card_number_is_6_l252_252721


namespace divide_5000_among_x_and_y_l252_252631

theorem divide_5000_among_x_and_y (total_amount : ℝ) (ratio_x : ℝ) (ratio_y : ℝ) (parts : ℝ) :
  total_amount = 5000 → ratio_x = 2 → ratio_y = 8 → parts = ratio_x + ratio_y → 
  (total_amount / parts) * ratio_x = 1000 := 
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end divide_5000_among_x_and_y_l252_252631


namespace two_point_two_five_as_fraction_l252_252896

theorem two_point_two_five_as_fraction : (2.25 : ℚ) = 9 / 4 := 
by 
  -- Proof steps would be added here
  sorry

end two_point_two_five_as_fraction_l252_252896


namespace probability_of_symmetry_line_l252_252374

-- Define the conditions of the problem.
def is_on_symmetry_line (P Q : (ℤ × ℤ)) :=
  (Q.fst = P.fst) ∨ (Q.snd = P.snd) ∨ (Q.fst - P.fst = Q.snd - P.snd) ∨ (Q.fst - P.fst = P.snd - Q.snd)

-- Define the main statement of the theorem to be proved.
theorem probability_of_symmetry_line :
  let grid_size := 11
  let total_points := grid_size * grid_size
  let center : (ℤ × ℤ) := (grid_size / 2, grid_size / 2)
  let other_points := total_points - 1
  let symmetric_points := 40
  /- Here we need to calculate the probability, which is the ratio of symmetric points to other points,
     and this should equal 1/3 -/
  (symmetric_points : ℚ) / other_points = 1 / 3 :=
by sorry

end probability_of_symmetry_line_l252_252374


namespace compare_xyz_l252_252305

theorem compare_xyz
  (a b c d : ℝ) (h : a < b ∧ b < c ∧ c < d)
  (x : ℝ) (hx : x = (a + b) * (c + d))
  (y : ℝ) (hy : y = (a + c) * (b + d))
  (z : ℝ) (hz : z = (a + d) * (b + c)) :
  x < y ∧ y < z :=
by sorry

end compare_xyz_l252_252305


namespace intersection_of_A_and_B_l252_252691

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def setB (x : ℝ) : Prop := 0 < x ∧ x ≤ 2
def setIntersection (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

theorem intersection_of_A_and_B :
  ∀ x, (setA x ∧ setB x) ↔ setIntersection x := 
by sorry

end intersection_of_A_and_B_l252_252691


namespace chess_tournament_participants_l252_252997

-- Define the number of grandmasters
variables (x : ℕ)

-- Define the number of masters as three times the number of grandmasters
def num_masters : ℕ := 3 * x

-- Condition on total points scored: Master's points is 1.2 times the Grandmaster's points
def points_condition (g m : ℕ) : Prop := m = 12 * g / 10

-- Proposition that the total number of participants is 12
theorem chess_tournament_participants (x_nonnegative: 0 < x) (g m : ℕ)
  (masters_points: points_condition g m) : 
  4 * x = 12 := 
sorry

end chess_tournament_participants_l252_252997


namespace binomial_12_6_eq_924_l252_252934

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l252_252934


namespace abc_inequality_l252_252688

theorem abc_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (a * (a^2 + b * c)) / (b + c) + (b * (b^2 + c * a)) / (c + a) + (c * (c^2 + a * b)) / (a + b) ≥ a * b + b * c + c * a := 
by 
  sorry

end abc_inequality_l252_252688


namespace arithmetic_sequence_min_sum_l252_252981

theorem arithmetic_sequence_min_sum (x : ℝ) (d : ℝ) (h₁ : d > 0) :
  (∃ n : ℕ, n > 0 ∧ (n^2 - 4 * n < 0) ∧ (n = 6 ∨ n = 7)) :=
by
  sorry

end arithmetic_sequence_min_sum_l252_252981


namespace smallest_digit_divisible_by_9_l252_252029

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l252_252029


namespace ellipse_sum_a_k_l252_252441

theorem ellipse_sum_a_k {a b h k : ℝ}
  (foci1 foci2 : ℝ × ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : h = (foci1.1 + foci2.1) / 2)
  (k_center : k = (foci1.2 + foci2.2) / 2)
  (distance1 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci1.1)^2 + (point_on_ellipse.2 - foci1.2)^2))
  (distance2 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci2.1)^2 + (point_on_ellipse.2 - foci2.2)^2))
  (major_axis_length : ℝ := distance1 + distance2)
  (h_a : a = major_axis_length / 2)
  (c := Real.sqrt ((foci2.1 - foci1.1)^2 + (foci2.2 - foci1.2)^2) / 2)
  (h_b : b^2 = a^2 - c^2) :
  a + k = (7 + Real.sqrt 13) / 2 := 
by
  sorry

end ellipse_sum_a_k_l252_252441


namespace avg_licks_l252_252621

theorem avg_licks (Dan Michael Sam David Lance : ℕ) 
  (hDan : Dan = 58) 
  (hMichael : Michael = 63) 
  (hSam : Sam = 70) 
  (hDavid : David = 70) 
  (hLance : Lance = 39) : 
  (Dan + Michael + Sam + David + Lance) / 5 = 60 :=
by 
  sorry

end avg_licks_l252_252621


namespace spiral_grid_third_row_sum_l252_252995

theorem spiral_grid_third_row_sum :
  let n := 12 
  let grid := array n (array n ℕ)
  -- Assume a function to generate the spiral grid
  let generate_spiral_grid : ℕ → array n (array n ℕ) := sorry
  -- Fill the grid with numbers 1 to n*n in a spiral order
  let spiral := generate_spiral_grid n
  -- Extract the third row
  let third_row := spiral[2]
  -- Find the least and greatest numbers in the third row
  let least_number_in_third_row := min third_row
  let greatest_number_in_third_row := max third_row
  -- Calculate the sum of these two numbers
  let sum := least_number_in_third_row + greatest_number_in_third_row
  sum = 55 :=
by
  sorry

end spiral_grid_third_row_sum_l252_252995


namespace solution_of_equation_l252_252495

theorem solution_of_equation (a : ℝ) : (∃ x : ℝ, x = 4 ∧ (a * x - 3 = 4 * x + 1)) → a = 5 :=
by
  sorry

end solution_of_equation_l252_252495


namespace slope_of_arithmetic_sequence_l252_252980

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (a_1 d n : α) : α := n * a_1 + n * (n-1) / 2 * d

theorem slope_of_arithmetic_sequence (a_1 d n : α) 
  (hS2 : S a_1 d 2 = 10)
  (hS5 : S a_1 d 5 = 55)
  : (a_1 + 2 * d - a_1) / 2 = 4 :=
by
  sorry

end slope_of_arithmetic_sequence_l252_252980


namespace remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l252_252727

theorem remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero (x : ℝ) :
  (x + 1) ^ 2025 % (x ^ 2 + 1) = 0 :=
  sorry

end remainder_x_plus_1_pow_2025_mod_x_square_plus_1_eq_zero_l252_252727


namespace smallest_square_contains_five_disks_l252_252465

noncomputable def smallest_side_length := 2 + 2 * Real.sqrt 2

theorem smallest_square_contains_five_disks :
  ∃ (a : ℝ), a = smallest_side_length ∧ (∃ (d : ℕ → ℝ × ℝ), 
    (∀ i, 0 ≤ i ∧ i < 5 → (d i).fst ^ 2 + (d i).snd ^ 2 < (a / 2 - 1) ^ 2) ∧ 
    (∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 ∧ i ≠ j → 
      (d i).fst ^ 2 + (d i).snd ^ 2 + (d j).fst ^ 2 + (d j).snd ^ 2 ≥ 4)) :=
sorry

end smallest_square_contains_five_disks_l252_252465


namespace calc_expression_find_linear_function_l252_252592

-- Part 1
theorem calc_expression : real.cbrt 8 + abs (-5) + (-1:ℤ) ^ 2023 = 6 :=
by
  sorry

-- Part 2
theorem find_linear_function :
  ∃ k b : ℝ, (y = k * x + b ↔ y = 2 * x + 1) ∧
  (b = 1) ∧
  (2 * k + b = 5) :=
by
  sorry

end calc_expression_find_linear_function_l252_252592


namespace root_conditions_l252_252548

theorem root_conditions (m : ℝ) : (∃ a b : ℝ, a < 2 ∧ b > 2 ∧ a * b = -1 ∧ a + b = m) ↔ m > 3 / 2 := sorry

end root_conditions_l252_252548


namespace radius_of_large_circle_l252_252177

theorem radius_of_large_circle : 
  ∃ (R : ℝ), R = 2 + 2 * Real.sqrt 2 ∧ 
  ∀ (r : ℝ) (n : ℕ), 
    (∀ (i j : ℕ), i ≠ j → i < n → j < n → 
    dist (r * cos (2 * i * π / n), r * sin (2 * i * π / n)) 
         (r * cos (2 * j * π / n), r * sin (2 * j * π / n)) = 2 * r) ∧ 
    (∀ (i : ℕ), 
      i < n → 
      dist (r * cos (2 * i * π / n), r * sin (2 * i * π / n)) 
         (0, 0) = R - r) → r = 2 ∧ n = 4 :=
by
  sorry

end radius_of_large_circle_l252_252177


namespace sum_of_remainders_l252_252442

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  -- sorry is here to skip the actual proof as per instructions
  sorry

end sum_of_remainders_l252_252442


namespace asha_borrowed_from_mother_l252_252307

def total_money (M : ℕ) : ℕ := 20 + 40 + 70 + 100 + M

def remaining_money_after_spending_3_4 (total : ℕ) : ℕ := total * 1 / 4

theorem asha_borrowed_from_mother : ∃ M : ℕ, total_money M = 260 ∧ remaining_money_after_spending_3_4 (total_money M) = 65 :=
by
  sorry

end asha_borrowed_from_mother_l252_252307


namespace shorter_leg_of_right_triangle_l252_252820

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l252_252820


namespace triangle_count_l252_252574

-- Define the function to compute the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the number of points on each side
def pointsAB : ℕ := 6
def pointsBC : ℕ := 7

-- Compute the number of triangles that can be formed
theorem triangle_count (h₁ : pointsAB = 6) (h₂ : pointsBC = 7) : 
  (binom pointsAB 2) * (binom pointsBC 1) + (binom pointsBC 2) * (binom pointsAB 1) = 231 := by
  sorry

end triangle_count_l252_252574


namespace prob_dist_and_expectation_prob_non_negative_l252_252435

-- Definitions based on problem conditions
def total_score (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * 100 - (3 - correct_answers) * 100

def prob_correct : ℝ := 0.8
def prob_incorrect : ℝ := 0.2

def prob_dist (s : ℤ) : ℝ :=
  if s = -300 then prob_incorrect ^ 3 else
  if s = -100 then 3 * (prob_incorrect ^ 2) * prob_correct else
  if s = 100 then 3 * prob_incorrect * (prob_correct ^ 2) else
  if s = 300 then prob_correct ^ 3 else 0

-- Part Ⅰ: Probability distribution and mathematical expectation
theorem prob_dist_and_expectation :
  (prob_dist (-300) = 0.008) ∧
  (prob_dist (-100) = 0.096) ∧
  (prob_dist (100) = 0.384) ∧
  (prob_dist (300) = 0.512) ∧
  (∑ s in {-300, -100, 100, 300}, s * prob_dist s) = 180 := sorry

-- Part Ⅱ: Probability of a non-negative score
theorem prob_non_negative :
  (∑ s in {100, 300}, prob_dist s) = 0.896 := sorry

end prob_dist_and_expectation_prob_non_negative_l252_252435


namespace find_x_for_which_ffx_eq_fx_l252_252689

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_for_which_ffx_eq_fx :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end find_x_for_which_ffx_eq_fx_l252_252689


namespace balance_scale_equation_l252_252533

theorem balance_scale_equation 
  (G Y B W : ℝ)
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 6 * B)
  (h3 : 2 * B = 3 * W) : 
  3 * G + 4 * Y + 3 * W = 16 * B :=
by
  sorry

end balance_scale_equation_l252_252533


namespace range_of_c_for_two_distinct_roots_l252_252240

theorem range_of_c_for_two_distinct_roots (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 3 * x1 + c = x1 + 2) ∧ (x2^2 - 3 * x2 + c = x2 + 2)) ↔ (c < 6) :=
sorry

end range_of_c_for_two_distinct_roots_l252_252240


namespace find_shop_width_l252_252403

def shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_square_foot : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_area := annual_rent / annual_rent_per_square_foot
  total_area / length

theorem find_shop_width :
  shop_width 3600 20 144 = 15 :=
by 
  -- Here would go the proof, but we add sorry to skip it
  sorry

end find_shop_width_l252_252403


namespace min_sum_of_factors_l252_252267

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1176) :
  a + b + c ≥ 59 :=
sorry

end min_sum_of_factors_l252_252267


namespace radii_of_cylinder_and_cone_are_equal_l252_252911

theorem radii_of_cylinder_and_cone_are_equal
  (h : ℝ)
  (r : ℝ)
  (V_cylinder : ℝ := π * r^2 * h)
  (V_cone : ℝ := (1/3) * π * r^2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
  r = r :=
by
  sorry

end radii_of_cylinder_and_cone_are_equal_l252_252911


namespace min_dist_sum_l252_252664

theorem min_dist_sum (x y : ℝ) :
  let M := (1, 3)
  let N := (7, 5)
  let P_on_M := (x - 1)^2 + (y - 3)^2 = 1
  let Q_on_N := (x - 7)^2 + (y - 5)^2 = 4
  let A_on_x_axis := y = 0
  ∃ (P Q : ℝ × ℝ), P_on_M ∧ Q_on_N ∧ ∀ A : ℝ × ℝ, A_on_x_axis → (|dist A P| + |dist A Q|) = 7 := 
sorry

end min_dist_sum_l252_252664


namespace num_four_digit_multiples_of_7_l252_252356

theorem num_four_digit_multiples_of_7 : 
  let smallest_k := Int.ceil (1000 / 7) in
  let largest_k := Int.floor (9999 / 7) in
  largest_k - smallest_k + 1 = 1286 := 
by
  sorry

end num_four_digit_multiples_of_7_l252_252356


namespace rational_numbers_property_l252_252472

theorem rational_numbers_property (n : ℕ) (h : n > 0) :
  ∃ (a b : ℚ), a ≠ b ∧ (∀ k, 1 ≤ k ∧ k ≤ n → ∃ m : ℤ, a^k - b^k = m) ∧ 
  ∀ i, (a : ℝ) ≠ i ∧ (b : ℝ) ≠ i :=
sorry

end rational_numbers_property_l252_252472


namespace total_pounds_of_food_l252_252653

-- Conditions
def chicken := 16
def hamburgers := chicken / 2
def hot_dogs := hamburgers + 2
def sides := hot_dogs / 2

-- Define the total pounds of food
def total_food := chicken + hamburgers + hot_dogs + sides

-- Theorem statement that corresponds to the problem, showing the final result
theorem total_pounds_of_food : total_food = 39 := 
by
  -- Placeholder for the proof
  sorry

end total_pounds_of_food_l252_252653


namespace vector_subtraction_identity_l252_252003

variables (a b : ℝ)

theorem vector_subtraction_identity (a b : ℝ) :
  ((1 / 2) * a - b) - ((3 / 2) * a - 2 * b) = b - a :=
by
  sorry

end vector_subtraction_identity_l252_252003


namespace num_partitions_l252_252072

open Finset

theorem num_partitions (s : Finset ℕ) (h : s = (range 15)) :
  let p := s.filter (λ x, x ≠ 7) in
  ∑ x in p, choose 12 (x-1) = 3172 := 
by
  let n := ∑ x in range 13, if x ≠ 7 then 1 else 0;
  let m := choose 12 6;
  have h1 : ∑ x in p, choose 12 (x-1) = 2 ^ 12 - m :=
    by simp [p, sum_filter, choose, h];
  rw [h1];
  simp [nat.choose, nat.sub, tsub, nat.pow];
  sorry

end num_partitions_l252_252072


namespace binom_12_6_eq_924_l252_252941

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l252_252941


namespace fraction_to_decimal_l252_252459

theorem fraction_to_decimal :
  (7 : ℝ) / (16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l252_252459


namespace inequality_not_always_true_l252_252193

theorem inequality_not_always_true {a b c : ℝ}
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : c ≠ 0) : ¬ ∀ c : ℝ, (a / c > b / c) :=
by
  sorry

end inequality_not_always_true_l252_252193


namespace arithmetic_sequence_a4_is_5_l252_252832

variable (a : ℕ → ℕ)

-- Arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m k : ℕ, n < m ∧ m < k → 2 * a m = a n + a k

-- Given condition
axiom sum_third_and_fifth : a 3 + a 5 = 10

-- Prove that a_4 = 5
theorem arithmetic_sequence_a4_is_5
  (h : is_arithmetic_sequence a) : a 4 = 5 := by
  sorry

end arithmetic_sequence_a4_is_5_l252_252832


namespace Jessica_paid_1000_for_rent_each_month_last_year_l252_252243

/--
Jessica paid $200 for food each month last year.
Jessica paid $100 for car insurance each month last year.
This year her rent goes up by 30%.
This year food costs increase by 50%.
This year the cost of her car insurance triples.
Jessica pays $7200 more for her expenses over the whole year compared to last year.
-/
theorem Jessica_paid_1000_for_rent_each_month_last_year
  (R : ℝ) -- monthly rent last year
  (h1 : 12 * (0.30 * R + 100 + 200) = 7200) :
  R = 1000 :=
sorry

end Jessica_paid_1000_for_rent_each_month_last_year_l252_252243


namespace ratio_platform_to_pole_l252_252438

variables (l t T v : ℝ)
-- Conditions
axiom constant_velocity : ∀ t l, l = v * t
axiom pass_pole : l = v * t
axiom pass_platform : 6 * l = v * T 

theorem ratio_platform_to_pole (h1 : l = v * t) (h2 : 6 * l = v * T) : T / t = 6 := 
  by sorry

end ratio_platform_to_pole_l252_252438


namespace six_divisors_third_seven_times_second_fourth_ten_more_than_third_l252_252551

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end six_divisors_third_seven_times_second_fourth_ten_more_than_third_l252_252551


namespace six_divisors_third_seven_times_second_fourth_ten_more_than_third_l252_252550

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end six_divisors_third_seven_times_second_fourth_ten_more_than_third_l252_252550


namespace chameleon_color_change_l252_252217

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l252_252217


namespace eggs_for_husband_is_correct_l252_252848

-- Define the conditions
def eggs_per_child : Nat := 2
def num_children : Nat := 4
def eggs_for_herself : Nat := 2
def total_eggs_per_year : Nat := 3380
def days_per_week : Nat := 5
def weeks_per_year : Nat := 52

-- Define the total number of eggs Lisa makes for her husband per year
def eggs_for_husband : Nat :=
  total_eggs_per_year - 
  (num_children * eggs_per_child + eggs_for_herself) * (days_per_week * weeks_per_year)

-- Prove the main statement
theorem eggs_for_husband_is_correct : eggs_for_husband = 780 := by
  sorry

end eggs_for_husband_is_correct_l252_252848


namespace chameleons_color_change_l252_252219

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l252_252219


namespace solution_of_equation_l252_252626

theorem solution_of_equation (a b c : ℕ) :
    a^(b + 20) * (c - 1) = c^(b + 21) - 1 ↔ 
    (∃ b' : ℕ, b = b' ∧ a = 1 ∧ c = 0) ∨ 
    (∃ a' b' : ℕ, a = a' ∧ b = b' ∧ c = 1) :=
by sorry

end solution_of_equation_l252_252626


namespace product_with_a_equals_3_l252_252961

theorem product_with_a_equals_3 (a : ℤ) (h : a = 3) : 
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 :=
by
  sorry

end product_with_a_equals_3_l252_252961


namespace count_four_digit_multiples_of_7_l252_252354

theorem count_four_digit_multiples_of_7 : 
  let smallest := 1000
  let largest := 9999
  let first_multiple := Nat.least (λ n => n % 7 = 0) smallest 1001
  let last_multiple := largest - (largest % 7)
  let count := (last_multiple - first_multiple) / 7 + 1 in
  count = 1286 := 
by
  sorry

end count_four_digit_multiples_of_7_l252_252354


namespace least_integer_value_l252_252463

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end least_integer_value_l252_252463


namespace line_slope_angle_y_intercept_l252_252559

theorem line_slope_angle_y_intercept :
  ∀ (x y : ℝ), x - y - 1 = 0 → 
    (∃ k b : ℝ, y = x - 1 ∧ k = 1 ∧ b = -1 ∧ θ = 45 ∧ θ = Real.arctan k) := 
    by
      sorry

end line_slope_angle_y_intercept_l252_252559


namespace remaining_credit_l252_252382

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end remaining_credit_l252_252382


namespace isosceles_triangle_of_condition_l252_252796

theorem isosceles_triangle_of_condition (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 2 * b * Real.cos C)
  (h2 : A + B + C = Real.pi) :
  (B = C) ∨ (A = C) ∨ (A = B) := 
sorry

end isosceles_triangle_of_condition_l252_252796


namespace peter_total_food_l252_252649

-- Given conditions
variables (chicken hamburgers hot_dogs sides total_food : ℕ)
  (h1 : chicken = 16)
  (h2 : hamburgers = chicken / 2)
  (h3 : hot_dogs = hamburgers + 2)
  (h4 : sides = hot_dogs / 2)
  (total_food : ℕ := chicken + hamburgers + hot_dogs + sides)

theorem peter_total_food (h1 : chicken = 16)
                          (h2 : hamburgers = chicken / 2)
                          (h3 : hot_dogs = hamburgers + 2)
                          (h4 : sides = hot_dogs / 2) :
                          total_food = 39 :=
by sorry

end peter_total_food_l252_252649


namespace circumcircle_diameter_l252_252713

-- Given that the perimeter of triangle ABC is equal to 3 times the sum of the sines of its angles
-- and the Law of Sines holds for this triangle, we need to prove the diameter of the circumcircle is 3.
theorem circumcircle_diameter (a b c : ℝ) (A B C : ℝ) (R : ℝ)
  (h_perimeter : a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C))
  (h_law_of_sines : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R) :
  2 * R = 3 := 
by
  sorry

end circumcircle_diameter_l252_252713


namespace conic_section_focus_l252_252481

theorem conic_section_focus {m : ℝ} (h_non_zero : m ≠ 0) (h_non_five : m ≠ 5)
  (h_focus : ∃ (x_focus y_focus : ℝ), (x_focus, y_focus) = (2, 0) 
  ∧ (x_focus = c ∧ x_focus^2 / 4 = 5 * (1 - c^2 / m))) : m = 9 := 
by
  sorry

end conic_section_focus_l252_252481


namespace find_y_satisfies_equation_l252_252775

theorem find_y_satisfies_equation :
  ∃ y : ℝ, 3 * y + 6 = |(-20 + 2)| :=
by
  sorry

end find_y_satisfies_equation_l252_252775


namespace problem_l252_252468

theorem problem
  (x y : ℝ)
  (h₁ : x - 2 * y = -5)
  (h₂ : x * y = -2) :
  2 * x^2 * y - 4 * x * y^2 = 20 := 
by
  sorry

end problem_l252_252468


namespace cannot_be_zero_l252_252877

-- Define polynomial Q(x)
def Q (x : ℝ) (f g h i j : ℝ) : ℝ := x^5 + f * x^4 + g * x^3 + h * x^2 + i * x + j

-- Define the hypotheses for the proof
def distinct_roots (a b c d e : ℝ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def one_root_is_one (f g h i j : ℝ) := Q 1 f g h i j = 0

-- Statement to prove
theorem cannot_be_zero (f g h i j a b c d : ℝ)
  (h1 : Q 1 f g h i j = 0)
  (h2 : distinct_roots 1 a b c d)
  (h3 : Q 1 f g h i j = (1-a)*(1-b)*(1-c)*(1-d)) :
  i ≠ 0 :=
by
  sorry

end cannot_be_zero_l252_252877


namespace exists_strictly_increasing_sequence_l252_252700

open Nat

-- Definition of strictly increasing sequence of integers a
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

-- Condition i): Every natural number can be written as the sum of two terms from the sequence
def condition_i (a : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j

-- Condition ii): For each positive integer n, a_n > n^2/16
def condition_ii (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > n^2 / 16

-- The main theorem stating the existence of such a sequence
theorem exists_strictly_increasing_sequence :
  ∃ a : ℕ → ℕ, a 0 = 0 ∧ strictly_increasing_sequence a ∧ condition_i a ∧ condition_ii a :=
sorry

end exists_strictly_increasing_sequence_l252_252700


namespace problems_per_worksheet_l252_252612

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) (h1 : total_worksheets = 15) (h2 : graded_worksheets = 7) (h3 : remaining_problems = 24) : (remaining_problems / (total_worksheets - graded_worksheets)) = 3 :=
by {
  sorry
}

end problems_per_worksheet_l252_252612


namespace max_checkers_on_chessboard_l252_252135

open Finset

variable (V : Finset (Fin 8 × Fin 8))
variable [hboard : card V ≤ 64]

def convex_polygon (S : Finset (Fin 8 × Fin 8)) : Prop :=
  ∀ p1 p2 p3 ∈ S, ∠ p1 p2 p3 ≤ 180 ∧
    ∀ p q ∈ S, segment p q ⊆ S

theorem max_checkers_on_chessboard : ∃ S : Finset (Fin 8 × Fin 8), convex_polygon S ∧ card S = 13 :=
by
  sorry

end max_checkers_on_chessboard_l252_252135


namespace candies_per_person_l252_252636

def clowns : ℕ := 4
def children : ℕ := 30
def initial_candies : ℕ := 700
def candies_left : ℕ := 20

def total_people : ℕ := clowns + children
def candies_sold : ℕ := initial_candies - candies_left

theorem candies_per_person : candies_sold / total_people = 20 := by
  sorry

end candies_per_person_l252_252636


namespace sum_of_distances_l252_252384

theorem sum_of_distances (AB A'B' AD A'D' x y : ℝ) 
  (h1 : AB = 8)
  (h2 : A'B' = 6)
  (h3 : AD = 3)
  (h4 : A'D' = 1)
  (h5 : x = 2)
  (h6 : x / y = 3 / 2) : 
  x + y = 10 / 3 :=
by
  sorry

end sum_of_distances_l252_252384


namespace bikes_added_per_week_l252_252908

variables (x : ℕ) -- bikes added per week

-- Conditions
def original_bikes := 51
def bikes_sold := 18
def end_month_bikes := 45
def weeks_in_month := 4

-- Prove that the number of bikes added per week is 3
theorem bikes_added_per_week : 
  (original_bikes - bikes_sold + weeks_in_month * x = end_month_bikes) → 
  x = 3 := by
  sorry

end bikes_added_per_week_l252_252908


namespace moon_speed_conversion_l252_252265

theorem moon_speed_conversion
  (speed_kps : ℝ)
  (seconds_per_hour : ℝ)
  (h1 : speed_kps = 0.2)
  (h2 : seconds_per_hour = 3600) :
  speed_kps * seconds_per_hour = 720 := by
  sorry

end moon_speed_conversion_l252_252265


namespace age_ratio_rahul_deepak_l252_252555

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end age_ratio_rahul_deepak_l252_252555


namespace total_dog_food_amount_l252_252019

def initial_dog_food : ℝ := 15
def first_purchase : ℝ := 15
def second_purchase : ℝ := 10

theorem total_dog_food_amount : initial_dog_food + first_purchase + second_purchase = 40 := 
by 
  sorry

end total_dog_food_amount_l252_252019


namespace problem_A_inter_complement_B_l252_252192

noncomputable def A : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}
noncomputable def B : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}
noncomputable def complement_B : Set ℝ := {x : ℝ | ¬((x - 2) / (x - 1) ≥ 0)}

theorem problem_A_inter_complement_B : 
  (A ∩ complement_B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end problem_A_inter_complement_B_l252_252192


namespace cars_meet_time_l252_252738

theorem cars_meet_time (s1 s2 : ℝ) (d : ℝ) (c : s1 = (5 / 4) * s2) 
  (h1 : s1 = 100) (h2 : d = 720) : d / (s1 + s2) = 4 :=
by 
  sorry

end cars_meet_time_l252_252738


namespace largest_prime_divisor_of_36_sq_plus_49_sq_l252_252330

theorem largest_prime_divisor_of_36_sq_plus_49_sq : ∃ (p : ℕ), p = 36^2 + 49^2 ∧ Prime p := 
by
  let n := 36^2 + 49^2
  have h : n = 3697 := by norm_num
  use 3697
  split
  . exact h
  . exact sorry

end largest_prime_divisor_of_36_sq_plus_49_sq_l252_252330


namespace min_balls_for_color_15_l252_252142

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end min_balls_for_color_15_l252_252142


namespace percentage_of_volume_occupied_l252_252601

-- Define the dimensions of the block
def block_length : ℕ := 9
def block_width : ℕ := 7
def block_height : ℕ := 12

-- Define the dimension of the cube
def cube_side : ℕ := 4

-- Define the volumes
def block_volume : ℕ := block_length * block_width * block_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the count of cubes along each dimension
def cubes_along_length : ℕ := block_length / cube_side
def cubes_along_width : ℕ := block_width / cube_side
def cubes_along_height : ℕ := block_height / cube_side

-- Define the total number of cubes that fit into the block
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height

-- Define the total volume occupied by the cubes
def occupied_volume : ℕ := total_cubes * cube_volume

-- Define the percentage of the block's volume occupied by the cubes (as a float for precision)
def volume_percentage : Float := (Float.ofNat occupied_volume / Float.ofNat block_volume) * 100

-- Statement to prove
theorem percentage_of_volume_occupied :
  volume_percentage = 50.79 := by
  sorry

end percentage_of_volume_occupied_l252_252601


namespace binomial_12_6_eq_924_l252_252955

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l252_252955


namespace curve_is_hyperbola_l252_252968

-- We define the given polar equation as a function.
noncomputable def polar_curve (r θ : ℝ) : ℝ :=
  3 * (Real.cot θ) * (Real.csc θ)

-- We state that the curve defined by 'polar_curve' is a hyperbola.
theorem curve_is_hyperbola (r θ : ℝ) : 
  polar_curve r θ = r 
  → -- Proving that this curve corresponds to a hyperbola in Cartesian coordinates
  is_hyperbola (Some_transform_to_Cartesian r θ) := 
sorry

end curve_is_hyperbola_l252_252968


namespace gcd_2197_2209_l252_252724

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := 
by
  sorry

end gcd_2197_2209_l252_252724


namespace average_last_three_l252_252703

theorem average_last_three (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 65) 
  (h2 : (a + b + c + d) / 4 = 60) : 
  (e + f + g) / 3 = 71.67 :=
by
  sorry

end average_last_three_l252_252703


namespace question_correctness_l252_252413

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l252_252413


namespace find_second_number_l252_252097

variable (A B : ℕ)

def is_LCM (a b lcm : ℕ) := Nat.lcm a b = lcm
def is_HCF (a b hcf : ℕ) := Nat.gcd a b = hcf

theorem find_second_number (h_lcm : is_LCM 330 B 2310) (h_hcf : is_HCF 330 B 30) : B = 210 := by
  sorry

end find_second_number_l252_252097


namespace distance_to_post_office_l252_252420

theorem distance_to_post_office
  (D : ℝ)
  (travel_rate : ℝ) (walk_rate : ℝ)
  (total_time_hours : ℝ)
  (h1 : travel_rate = 25)
  (h2 : walk_rate = 4)
  (h3 : total_time_hours = 5 + 48 / 60) :
  D = 20 :=
by
  sorry

end distance_to_post_office_l252_252420


namespace at_least_30_cents_probability_l252_252866

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l252_252866


namespace find_p_q_l252_252467

def op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

theorem find_p_q :
  (∀ (a b c d : ℝ), (a = c ∧ b = d) ↔ (a, b) = (c, d)) →
  (op 1 2 p q = (5, 0)) →
  (p, q) = (1, -2) :=
by
  intro h
  intro eq_op
  sorry

end find_p_q_l252_252467


namespace conjecture_a_n_l252_252077

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

noncomputable def S_n (n : ℕ) : ℚ := 2 * n - a_n n

theorem conjecture_a_n (n : ℕ) (h : n > 0) : a_n n = (2^n - 1) / 2^(n-1) :=
by 
  sorry

end conjecture_a_n_l252_252077


namespace find_f_at_one_l252_252346

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4 * x ^ 2 - m * x + 5

theorem find_f_at_one :
  (∀ x : ℝ, x ≥ -2 → f x (-16) ≥ f (-2) (-16)) ∧
  (∀ x : ℝ, x ≤ -2 → f x (-16) ≤ f (-2) (-16)) →
  f 1 (-16) = 25 :=
sorry

end find_f_at_one_l252_252346


namespace count_squares_parallel_to_r_or_s_count_isosceles_right_triangles_parallel_to_r_or_s_l252_252064

-- Definitions
def square_grid : finset (ℕ × ℕ) := 
  {(i, j) | i ∈ range 4 ∧ j ∈ range 4 }.to_finset

def is_parallel_to_lines (v1 v2 : (ℕ × ℕ)) (r s : (ℕ × ℕ) → bool) : Prop := 
  r v1 || s v2

-- Theorem statements
theorem count_squares_parallel_to_r_or_s :
  (∀ r s, ∃ count, count = 6 ∧
    ∀ (sq : finset (ℕ × ℕ)), sq ⊆ square_grid → 
    (∃ v1 v2 v3 v4 ∈ sq, 
      (¬ is_parallel_to_lines v1 v2 r s) ∧ 
      (¬ is_parallel_to_lines v2 v3 r s) ∧ 
      (¬ is_parallel_to_lines v3 v4 r s) ∧ 
      (¬ is_parallel_to_lines v4 v1 r s))) := sorry

theorem count_isosceles_right_triangles_parallel_to_r_or_s :
  (∀ r s, ∃ count, count = 16 ∧ 
    ∀ (tri : finset (ℕ × ℕ)), tri ⊆ square_grid →
    (∃ v1 v2 v3 ∈ tri, 
      (¬ is_parallel_to_lines v1 v2 r s) ∧ 
      (¬ is_parallel_to_lines v2 v3 r s) ∧ 
      (∃ hyp, hyp = (v1 - v3) /\ |hyp| = |v1 - v2| * sqrt2))) := sorry

end count_squares_parallel_to_r_or_s_count_isosceles_right_triangles_parallel_to_r_or_s_l252_252064


namespace probability_blue_prime_and_yellow_divisible_by_3_l252_252125

-- Define the set of outcomes for each die roll
def outcomes := Finset.range 8

-- Define the set of prime numbers ≤ 8 for the blue die
def blue_prime_outcomes := {2, 3, 5, 7}

-- Define the set of numbers divisible by 3 ≤ 8 for the yellow die
def yellow_divisible_by_3_outcomes := {3, 6}

-- Define the total number of outcomes when two 8-sided dice are rolled
def total_outcomes := (outcomes.card) * (outcomes.card)

-- Define the number of successful outcomes for our condition
def successful_outcomes := (blue_prime_outcomes.card) * (yellow_divisible_by_3_outcomes.card)

-- Define the probability calculation
def probability := (successful_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem to be proved
theorem probability_blue_prime_and_yellow_divisible_by_3 :
  probability = 1 / 8 :=
by
  -- Proof is not required, so we use sorry
  sorry

end probability_blue_prime_and_yellow_divisible_by_3_l252_252125


namespace count_three_digit_numbers_with_2_without_6_l252_252360

theorem count_three_digit_numbers_with_2_without_6 : 
  let total_without_6 : ℕ := 648
  let total_without_6_and_2 : ℕ := 448
  total_without_6 - total_without_6_and_2 = 200 :=
by 
  have total_without_6 := 8 * 9 * 9
  have total_without_6_and_2 := 7 * 8 * 8
  rw total_without_6
  rw total_without_6_and_2
  exact calc
    8 * 9 * 9 - 7 * 8 * 8 = 648 - 448 := by simp
    ... = 200 := by norm_num

end count_three_digit_numbers_with_2_without_6_l252_252360


namespace mul_add_distrib_l252_252333

theorem mul_add_distrib :
  15 * 36 + 15 * 24 = 900 := by
  sorry

end mul_add_distrib_l252_252333


namespace olivine_more_stones_l252_252439

theorem olivine_more_stones (x O D : ℕ) (h1 : O = 30 + x) (h2 : D = O + 11)
  (h3 : 30 + O + D = 111) : x = 5 :=
by
  sorry

end olivine_more_stones_l252_252439


namespace find_pairs_eq_l252_252772

theorem find_pairs_eq : 
  { (m, n) : ℕ × ℕ | 0 < m ∧ 0 < n ∧ m ^ 2 + 2 * n ^ 2 = 3 * (m + 2 * n) } = {(3, 3), (4, 2)} :=
by sorry

end find_pairs_eq_l252_252772


namespace rank_values_l252_252073

noncomputable def a : ℝ := Real.log 7 / Real.log 3
noncomputable def b : ℝ := 2^1.1
noncomputable def c : ℝ := 0.8^3.1

theorem rank_values : c < a ∧ a < b := 
by
  have ha: 1 < a := sorry
  have hb: a < 2 := sorry
  have hc: b > 2 := sorry
  have hd: c < 1 := sorry
  exact ⟨sorry, sorry⟩

end rank_values_l252_252073


namespace smallest_digit_divisible_by_9_l252_252027

theorem smallest_digit_divisible_by_9 : 
  ∃ (d : ℕ), 0 ≤ d ∧ d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ d = 2 :=
by
  use 2
  split
  { exact nat.zero_le _ }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end smallest_digit_divisible_by_9_l252_252027


namespace tagged_fish_in_second_catch_l252_252208

theorem tagged_fish_in_second_catch
  (N : ℕ)
  (initial_catch tagged_returned : ℕ)
  (second_catch : ℕ)
  (approximate_pond_fish : ℕ)
  (condition_1 : initial_catch = 60)
  (condition_2 : tagged_returned = 60)
  (condition_3 : second_catch = 60)
  (condition_4 : approximate_pond_fish = 1800) :
  (tagged_returned * second_catch) / approximate_pond_fish = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l252_252208


namespace binomial_evaluation_l252_252952

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l252_252952


namespace problem1_l252_252576

theorem problem1 : (Nat.cbrt 8 : Int) + abs (-5 : Int) + (-1 : Int) ^ 2023 = 6 := by
  -- Proof goes here
  sorry

end problem1_l252_252576


namespace average_calculation_l252_252542

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 2 0) (average_two 1 2) 1 = 23 / 18 :=
by sorry

end average_calculation_l252_252542


namespace problem_1_problem_2_l252_252470

variable {a : ℕ → ℝ}
variable (n : ℕ)

-- Conditions of the problem
def seq_positive : ∀ (k : ℕ), a k > 0 := sorry
def a1 : a 1 = 1 := sorry
def recurrence (n : ℕ) : a (n + 1) = (a n + 1) / (12 * a n) := sorry

-- Proofs to be provided
theorem problem_1 : ∀ n : ℕ, a (2 * n + 1) < a (2 * n - 1) := 
by 
  apply sorry 

theorem problem_2 : ∀ n : ℕ, 1 / 6 ≤ a n ∧ a n ≤ 1 := 
by 
  apply sorry 

end problem_1_problem_2_l252_252470


namespace water_tower_excess_consumption_l252_252756

def water_tower_problem : Prop :=
  let initial_water := 2700
  let first_neighborhood := 300
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let fourth_neighborhood := 3 * first_neighborhood
  let fifth_neighborhood := third_neighborhood / 2
  let leakage := 50
  let first_neighborhood_final := first_neighborhood + 0.10 * first_neighborhood
  let second_neighborhood_final := second_neighborhood - 0.05 * second_neighborhood
  let third_neighborhood_final := third_neighborhood + 0.10 * third_neighborhood
  let fifth_neighborhood_final := fifth_neighborhood - 0.05 * fifth_neighborhood
  let total_consumption := 
    first_neighborhood_final + second_neighborhood_final + third_neighborhood_final +
    fourth_neighborhood + fifth_neighborhood_final + leakage
  let excess_consumption := total_consumption - initial_water
  excess_consumption = 252.5

theorem water_tower_excess_consumption : water_tower_problem := by
  sorry

end water_tower_excess_consumption_l252_252756


namespace sarah_must_solve_at_least_16_l252_252702

theorem sarah_must_solve_at_least_16
  (total_problems : ℕ)
  (problems_attempted : ℕ)
  (problems_unanswered : ℕ)
  (points_per_correct : ℕ)
  (points_per_unanswered : ℕ)
  (target_score : ℕ)
  (h1 : total_problems = 30)
  (h2 : points_per_correct = 7)
  (h3 : points_per_unanswered = 2)
  (h4 : problems_unanswered = 5)
  (h5 : problems_attempted = 25)
  (h6 : target_score = 120) :
  ∃ (correct_solved : ℕ), correct_solved ≥ 16 ∧ correct_solved ≤ problems_attempted ∧
    (correct_solved * points_per_correct) + (problems_unanswered * points_per_unanswered) ≥ target_score :=
by {
  sorry
}

end sarah_must_solve_at_least_16_l252_252702


namespace retail_price_percentage_l252_252915

variable (P : ℝ)
variable (wholesale_cost : ℝ)
variable (employee_price : ℝ)

axiom wholesale_cost_def : wholesale_cost = 200
axiom employee_price_def : employee_price = 192
axiom employee_discount_def : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))

theorem retail_price_percentage (P : ℝ) (wholesale_cost : ℝ) (employee_price : ℝ)
    (H1 : wholesale_cost = 200)
    (H2 : employee_price = 192)
    (H3 : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))) :
    P = 20 :=
  sorry

end retail_price_percentage_l252_252915


namespace sum_2016_eq_1008_l252_252663

-- Define the arithmetic sequence {a_n} and the sum of the first n terms S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
variable (h_arith_seq : ∀ n m, a (n+1) - a n = a (m+1) - a m)
variable (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)

-- Additional conditions from the problem
variable (h_vector : a 4 + a 2013 = 1)

-- Goal: Prove that the sum of the first 2016 terms equals 1008
theorem sum_2016_eq_1008 : S 2016 = 1008 := by
  sorry

end sum_2016_eq_1008_l252_252663


namespace length_of_train_l252_252105

theorem length_of_train (v : ℝ) (t : ℝ) (L : ℝ) 
  (h₁ : v = 36) 
  (h₂ : t = 1) 
  (h_eq_lengths : true) -- assuming the equality of lengths tacitly without naming
  : L = 300 := 
by 
  -- proof steps would go here
  sorry

end length_of_train_l252_252105


namespace problem_solution_l252_252487

variable {a b x y : ℝ}

-- Define the conditions as Lean assumptions
axiom cond1 : a * x + b * y = 3
axiom cond2 : a * x^2 + b * y^2 = 7
axiom cond3 : a * x^3 + b * y^3 = 16
axiom cond4 : a * x^4 + b * y^4 = 42

-- The main theorem statement: under these conditions, prove a * x^5 + b * y^5 = 99
theorem problem_solution : a * x^5 + b * y^5 = 99 := 
sorry -- proof omitted

end problem_solution_l252_252487


namespace decipher_numbers_l252_252834

variable (K I S : Nat)

theorem decipher_numbers
  (h1: 1 ≤ K ∧ K < 5)
  (h2: I ≠ 0)
  (h3: I ≠ K)
  (h_eq: K * 100 + I * 10 + S + K * 10 + S * 10 + I = I * 100 + S * 10 + K):
  (K, I, S) = (4, 9, 5) :=
by sorry

end decipher_numbers_l252_252834


namespace find_f_of_fraction_l252_252189

noncomputable def f (t : ℝ) : ℝ := sorry

theorem find_f_of_fraction (x : ℝ) (h : f ((1-x^2)/(1+x^2)) = x) :
  f ((2*x)/(1+x^2)) = (1 - x) / (1 + x) ∨ f ((2*x)/(1+x^2)) = (x - 1) / (1 + x) :=
sorry

end find_f_of_fraction_l252_252189


namespace binomial_evaluation_l252_252950

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l252_252950


namespace ticket_prices_count_l252_252998

theorem ticket_prices_count :
  let y := 30
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  ∀ (k : ℕ), (k ∈ divisors) ↔ (60 % k = 0 ∧ 90 % k = 0) → 
  (∃ n : ℕ, n = 8) :=
by
  sorry

end ticket_prices_count_l252_252998


namespace domain_of_f_l252_252706

-- Define the function domain transformation
theorem domain_of_f (f : ℝ → ℝ) : 
  (∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → -7 ≤ 2*x - 3 ∧ 2*x - 3 ≤ 1) ↔ (∀ (y : ℝ), -7 ≤ y ∧ y ≤ 1) :=
sorry

end domain_of_f_l252_252706


namespace binomial_12_6_l252_252925

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l252_252925


namespace rhombus_area_l252_252904

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 150 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l252_252904


namespace first_number_is_38_l252_252272

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end first_number_is_38_l252_252272


namespace smallest_number_in_sample_l252_252776

theorem smallest_number_in_sample :
  ∀ (N : ℕ) (k : ℕ) (n : ℕ), 
  0 < k → 
  N = 80 → 
  k = 5 →
  n = 42 →
  ∃ (a : ℕ), (0 ≤ a ∧ a < k) ∧
  42 = (N / k) * (42 / (N / k)) + a ∧
  ∀ (m : ℕ), (0 ≤ m ∧ m < k) → 
    (∀ (j : ℕ), (j = (N / k) * m + 10)) → 
    m = 0 → a = 10 := 
by
  sorry

end smallest_number_in_sample_l252_252776


namespace no_real_roots_of_quadratic_l252_252175

theorem no_real_roots_of_quadratic (k : ℝ) (hk : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l252_252175


namespace range_of_eccentricity_l252_252471

open Real

theorem range_of_eccentricity (a b c : ℝ) (α : ℝ) (e : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a^2 * b^2 > 0)
  (h4 : α ∈ Ioo (π/4) (π/3))
  (h5 : e = c / a)
  (h6 : (2 * c) / (sin α + cos α) = 2 * a)
  : e ∈ Ioo (sqrt 2 / 2) (sqrt 3 - 1) :=
sorry

end range_of_eccentricity_l252_252471


namespace solution_set_inequality_l252_252560

-- Statement of the problem
theorem solution_set_inequality :
  {x : ℝ | 1 / x < 1 / 2} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_inequality_l252_252560


namespace eight_digit_number_min_max_l252_252614

theorem eight_digit_number_min_max (Amin Amax B : ℕ) 
  (hAmin: Amin = 14444446) 
  (hAmax: Amax = 99999998) 
  (hB_coprime: Nat.gcd B 12 = 1) 
  (hB_length: 44444444 < B) 
  (h_digits: ∀ (b : ℕ), b < 10 → ∃ (A : ℕ), A = 10^7 * b + (B - b) / 10 ∧ A < 100000000) :
  (∃ b, Amin = 10^7 * b + (44444461 - b) / 10 ∧ Nat.gcd 44444461 12 = 1 ∧ 44444444 < 44444461) ∧
  (∃ b, Amax = 10^7 * b + (999999989 - b) / 10 ∧ Nat.gcd 999999989 12 = 1 ∧ 44444444 < 999999989) :=
  sorry

end eight_digit_number_min_max_l252_252614


namespace yang_hui_problem_l252_252007

theorem yang_hui_problem (x : ℝ) :
  x * (x + 12) = 864 :=
sorry

end yang_hui_problem_l252_252007


namespace kenya_peanuts_eq_133_l252_252522

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l252_252522


namespace fraction_of_suitable_dishes_l252_252194

theorem fraction_of_suitable_dishes {T : Type} (total_menu: ℕ) (vegan_dishes: ℕ) (vegan_fraction: ℚ) (gluten_inclusive_vegan_dishes: ℕ) (low_sugar_gluten_free_vegan_dishes: ℕ) 
(h1: vegan_dishes = 6)
(h2: vegan_fraction = 1/4)
(h3: gluten_inclusive_vegan_dishes = 4)
(h4: low_sugar_gluten_free_vegan_dishes = 1)
(h5: total_menu = vegan_dishes / vegan_fraction) :
(1 : ℚ) / (total_menu : ℚ) = (1 : ℚ) / 24 := 
by
  sorry

end fraction_of_suitable_dishes_l252_252194


namespace radius_large_circle_l252_252176

-- Definitions for the conditions
def radius_small_circle : ℝ := 2

def is_tangent_externally (r1 r2 : ℝ) : Prop := -- Definition of external tangency
  r1 + r2 = 4

def is_tangent_internally (R r : ℝ) : Prop := -- Definition of internal tangency
  R - r = 4

-- Setting up the property we need to prove: large circle radius
theorem radius_large_circle
  (R r : ℝ)
  (h1 : r = radius_small_circle)
  (h2 : is_tangent_externally r r)
  (h3 : is_tangent_externally r r)
  (h4 : is_tangent_externally r r)
  (h5 : is_tangent_externally r r)
  (h6 : is_tangent_internally R r) :
  R = 4 :=
by sorry

end radius_large_circle_l252_252176


namespace unique_valid_number_l252_252899

-- Define the form of the three-digit number.
def is_form_sixb5 (n : ℕ) : Prop :=
  ∃ b : ℕ, b < 10 ∧ n = 600 + 10 * b + 5

-- Define the condition for divisibility by 11.
def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

-- Define the alternating sum property for our specific number format.
def alternating_sum_cond (b : ℕ) : Prop :=
  (11 - b) % 11 = 0

-- The final proposition to be proved.
theorem unique_valid_number : ∃ n, is_form_sixb5 n ∧ is_divisible_by_11 n ∧ n = 605 :=
by {
  sorry
}

end unique_valid_number_l252_252899


namespace prob_negative_product_of_three_elems_from_set_l252_252124

open Finset

theorem prob_negative_product_of_three_elems_from_set : 
  let s := ({-3, -1, 0, 2, 4, 5} : Finset ℤ)
  let subsets_of_three := s.powerset.filter (λ t, t.card = 3)
  let total_subsets := subsets_of_three.card
  let negative_product_subsets := subsets_of_three.filter (λ t, (t.prod id < 0)).card
  (negative_product_subsets : ℚ) / total_subsets = 3 / 10 :=
by
  sorry

end prob_negative_product_of_three_elems_from_set_l252_252124


namespace chameleon_color_change_l252_252216

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l252_252216


namespace gcd_150_m_l252_252566

theorem gcd_150_m (m : ℕ)
  (h : ∃ d : ℕ, d ∣ 150 ∧ d ∣ m ∧ (∀ x, x ∣ 150 → x ∣ m → x = 1 ∨ x = 5 ∨ x = 25)) :
  gcd 150 m = 25 :=
sorry

end gcd_150_m_l252_252566


namespace binom_12_6_l252_252933

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l252_252933


namespace hyperbola_asymptotes_l252_252917

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 - (y^2 / 4) = 1) ↔ (y = 2 * x ∨ y = -2 * x) := by
  sorry

end hyperbola_asymptotes_l252_252917


namespace eval_expression_l252_252769

theorem eval_expression (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := 
by 
  sorry

end eval_expression_l252_252769


namespace probability_sum_divisible_by_3_l252_252206

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

noncomputable def num_pairs_divisible_by_3 (primes : List ℕ) : ℕ :=
  (primes.toFinset.powerset.toList.filter 
    (λ s, s.card = 2 ∧ (s.sum % 3 = 0))).length

theorem probability_sum_divisible_by_3 :
  (num_pairs_divisible_by_3 first_ten_primes : ℚ) / (10.choose 2) = 2 / 15 :=
by
  sorry

end probability_sum_divisible_by_3_l252_252206


namespace find_n_with_divisors_conditions_l252_252552

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end find_n_with_divisors_conditions_l252_252552


namespace smallest_angle_in_convex_polygon_l252_252875

theorem smallest_angle_in_convex_polygon :
  ∀ (n : ℕ) (angles : ℕ → ℕ) (d : ℕ), n = 25 → (∀ i, 1 ≤ i ∧ i ≤ n → angles i = 166 - 1 * (13 - i)) 
  → 1 ≤ d ∧ d ≤ 1 → (angles 1 = 154) := 
by
  sorry

end smallest_angle_in_convex_polygon_l252_252875


namespace imaginary_part_of_fraction_l252_252711

open Complex

theorem imaginary_part_of_fraction :
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  z.im = 1 :=
by
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  show z.im = 1
  sorry

end imaginary_part_of_fraction_l252_252711


namespace prob_correct_l252_252712

noncomputable def r : ℝ := (4.5 : ℝ)  -- derived from solving area and line equations
noncomputable def s : ℝ := (7.5 : ℝ)  -- derived from solving area and line equations

theorem prob_correct (P Q T : ℝ × ℝ)
  (hP : P = (9, 0))
  (hQ : Q = (0, 15))
  (hT : T = (r, s))
  (hline : s = -5/3 * r + 15)
  (harea : 2 * (1/2 * 9 * 15) = (1/2 * 9 * s) * 4) :
  r + s = 12 := by
  sorry

end prob_correct_l252_252712


namespace unique_solutions_of_system_l252_252642

theorem unique_solutions_of_system (a : ℝ) :
  (∃! (x y : ℝ), a^2 - 2 * a * x - 6 * y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  (a ∈ Set.union (Set.Ioo (-12) (-6)) (Set.union {0} (Set.Ioo 6 12))) :=
by
  sorry

end unique_solutions_of_system_l252_252642


namespace xz_squared_value_l252_252527

theorem xz_squared_value (x y z : ℝ) (h₁ : 3 * x * 5 * z = (4 * y)^2) (h₂ : (y^2 : ℝ) = (x^2 + z^2) / 2) :
  x^2 + z^2 = 16 := 
sorry

end xz_squared_value_l252_252527


namespace school_year_days_l252_252889

theorem school_year_days :
  ∀ (D : ℕ),
  (9 = 5 * D / 100) →
  D = 180 := by
  intro D
  sorry

end school_year_days_l252_252889


namespace count_divisibles_by_8_in_range_100_250_l252_252790

theorem count_divisibles_by_8_in_range_100_250 : 
  let lower_bound := 100
  let upper_bound := 250
  let divisor := 8
  ∃ n : ℕ, (∀ x : ℕ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ x % divisor = 0 ↔ (n = 19)) :=
begin
  let lower_bound := 100,
  let upper_bound := 250,
  let divisor := 8,
  let first_multiple := ((lower_bound + divisor - 1) / divisor) * divisor,
  let last_multiple := (upper_bound / divisor) * divisor,
  let first_index := first_multiple / divisor,
  let last_index := last_multiple / divisor,
  let n := (last_index - first_index + 1),
  use n,
  intros x,
  split,
  { intro hx,
    exact ⟨nat.exists_eq_add_of_le hx.1, nat.exists_eq_add_of_le hx.2.1, nat.exists_eq_of_divisible hx.2.2⟩ },
  { intro hn,
    rw hn,
    refine ⟨_, _, _⟩,
    sorry
  }
end

end count_divisibles_by_8_in_range_100_250_l252_252790


namespace cocktail_cost_per_litre_l252_252106

theorem cocktail_cost_per_litre :
  let mixed_fruit_cost := 262.85
  let acai_berry_cost := 3104.35
  let mixed_fruit_volume := 37
  let acai_berry_volume := 24.666666666666668
  let total_cost := mixed_fruit_volume * mixed_fruit_cost + acai_berry_volume * acai_berry_cost
  let total_volume := mixed_fruit_volume + acai_berry_volume
  total_cost / total_volume = 1400 :=
by
  sorry

end cocktail_cost_per_litre_l252_252106


namespace sector_area_l252_252040

-- Define the given parameters
def central_angle : ℝ := 2
def radius : ℝ := 3

-- Define the statement about the area of the sector
theorem sector_area (α r : ℝ) (hα : α = 2) (hr : r = 3) :
  let l := α * r
  let A := 0.5 * l * r
  A = 9 :=
by
  -- The proof is not required
  sorry

end sector_area_l252_252040


namespace mario_age_difference_l252_252269

variable (Mario_age Maria_age : ℕ)

def age_conditions (Mario_age Maria_age difference : ℕ) : Prop :=
  Mario_age + Maria_age = 7 ∧
  Mario_age = 4 ∧
  Mario_age - Maria_age = difference

theorem mario_age_difference : ∃ (difference : ℕ), age_conditions 4 (4 - difference) difference ∧ difference = 1 := by
  sorry

end mario_age_difference_l252_252269


namespace quadrilateral_is_parallelogram_l252_252148

theorem quadrilateral_is_parallelogram
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2 * a * c - 2 * b * d = 0) :
  (a = c) ∧ (b = d) :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l252_252148


namespace number_of_schools_is_8_l252_252720

-- Define the number of students trying out and not picked per school
def students_trying_out := 65.0
def students_not_picked := 17.0
def students_picked := students_trying_out - students_not_picked

-- Define the total number of students who made the teams
def total_students_made_teams := 384.0

-- Define the number of schools
def number_of_schools := total_students_made_teams / students_picked

theorem number_of_schools_is_8 : number_of_schools = 8 := by
  -- Proof omitted
  sorry

end number_of_schools_is_8_l252_252720


namespace smallest_positive_multiple_l252_252287

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end smallest_positive_multiple_l252_252287


namespace number_is_450064_l252_252432

theorem number_is_450064 : (45 * 10000 + 64) = 450064 :=
by
  sorry

end number_is_450064_l252_252432


namespace find_first_number_l252_252271

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end find_first_number_l252_252271


namespace rectangle_area_perimeter_eq_l252_252783

theorem rectangle_area_perimeter_eq (x : ℝ) (h : 4 * x * (x + 4) = 2 * 4 * x + 2 * (x + 4)) : x = 1 / 2 :=
sorry

end rectangle_area_perimeter_eq_l252_252783


namespace solution_l252_252320

noncomputable def polynomial (x m : ℝ) := 3 * x^2 - 5 * x + m

theorem solution (m : ℝ) : (∃ a : ℝ, a = 2 ∧ polynomial a m = 0) -> m = -2 := by
  sorry

end solution_l252_252320


namespace remainder_of_polynomial_division_l252_252960

-- Define the polynomial f(r)
def f (r : ℝ) : ℝ := r ^ 15 + 1

-- Define the polynomial divisor g(r)
def g (r : ℝ) : ℝ := r + 1

-- State the theorem about the remainder when f(r) is divided by g(r)
theorem remainder_of_polynomial_division : 
  (f (-1)) = 0 := by
  -- Skipping the proof for now
  sorry

end remainder_of_polynomial_division_l252_252960


namespace child_admission_charge_l252_252872

-- Given conditions
variables (A C : ℝ) (T : ℝ := 3.25) (n : ℕ := 3)

-- Admission charge for an adult
def admission_charge_adult : ℝ := 1

-- Admission charge for a child
def admission_charge_child (C : ℝ) : ℝ := C

-- Total cost paid by adult with 3 children
def total_cost (A C : ℝ) (n : ℕ) : ℝ := A + n * C

-- The proof statement
theorem child_admission_charge (C : ℝ) : total_cost 1 C 3 = 3.25 -> C = 0.75 :=
by
  sorry

end child_admission_charge_l252_252872


namespace binomial_12_6_l252_252924

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l252_252924


namespace range_of_a_l252_252191
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 1| + |2 * x - a|

theorem range_of_a (a : ℝ)
  (h : ∀ x : ℝ, f x a ≥ (1 / 4) * a ^ 2 + 1) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l252_252191


namespace smallest_positive_sum_l252_252060

structure ArithmeticSequence :=
  (a_n : ℕ → ℤ)  -- The sequence is an integer sequence
  (d : ℤ)        -- The common difference of the sequence

def sum_of_first_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a_n 1 + seq.a_n n)) / 2  -- Sum of first n terms

def condition (seq : ArithmeticSequence) : Prop :=
  (seq.a_n 11 < -1 * seq.a_n 10)

theorem smallest_positive_sum (seq : ArithmeticSequence) (H : condition seq) :
  ∃ n, sum_of_first_n seq n > 0 ∧ ∀ m < n, sum_of_first_n seq m ≤ 0 → n = 19 :=
sorry

end smallest_positive_sum_l252_252060


namespace product_of_a_l252_252014

theorem product_of_a : 
  (∃ a b : ℝ, (3 * a - 5)^2 + (a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2 ∧ 
    (a * b = -8.32)) :=
by 
  sorry

end product_of_a_l252_252014


namespace measure_angle_A_l252_252510

theorem measure_angle_A (a b c : ℝ) (A B C : ℝ)
  (h1 : ∀ (Δ : Type), Δ → Δ → Δ)
  (h2 : a / Real.cos A = b / (2 * Real.cos B) ∧ 
        a / Real.cos A = c / (3 * Real.cos C))
  (h3 : A + B + C = Real.pi) : 
  A = Real.pi / 4 :=
sorry

end measure_angle_A_l252_252510


namespace connie_initial_marbles_l252_252959

theorem connie_initial_marbles (marbles_given : ℝ) (marbles_left : ℝ) : 
  marbles_given = 183 → marbles_left = 593 → marbles_given + marbles_left = 776 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end connie_initial_marbles_l252_252959


namespace coefficients_divisible_by_7_l252_252511

theorem coefficients_divisible_by_7 
  {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  ∃ k l m n o : ℤ, a = 7*k ∧ b = 7*l ∧ c = 7*m ∧ d = 7*n ∧ e = 7*o :=
by
  sorry

end coefficients_divisible_by_7_l252_252511


namespace tan_theta_sqrt3_l252_252973

theorem tan_theta_sqrt3 (θ : ℝ) 
  (h : Real.cos (40 * (π / 180) - θ) 
     + Real.cos (40 * (π / 180) + θ) 
     + Real.cos (80 * (π / 180) - θ) = 0) 
  : Real.tan θ = -Real.sqrt 3 := 
by
  sorry

end tan_theta_sqrt3_l252_252973


namespace lisa_needs_28_more_marbles_l252_252391

theorem lisa_needs_28_more_marbles :
  ∀ (friends : ℕ) (initial_marbles : ℕ),
  friends = 12 → 
  initial_marbles = 50 →
  (∀ n, 1 ≤ n ∧ n ≤ friends → ∃ (marbles : ℕ), marbles ≥ 1 ∧ ∀ i j, (i ≠ j ∧ i ≠ 0 ∧ j ≠ 0) → (marbles i ≠ marbles j)) →
  ( ∑ k in finset.range (friends + 1), k ) - initial_marbles = 28 :=
by
  intros friends initial_marbles h_friends h_initial_marbles _,
  rw [h_friends, h_initial_marbles],
  sorry

end lisa_needs_28_more_marbles_l252_252391


namespace subway_length_in_meters_l252_252717

noncomputable def subway_speed : ℝ := 1.6 -- km per minute
noncomputable def crossing_time : ℝ := 3 + 15 / 60 -- minutes
noncomputable def bridge_length : ℝ := 4.85 -- km

theorem subway_length_in_meters :
  let total_distance_traveled := subway_speed * crossing_time
  let subway_length_km := total_distance_traveled - bridge_length
  let subway_length_m := subway_length_km * 1000
  subway_length_m = 350 :=
by
  sorry

end subway_length_in_meters_l252_252717


namespace ingrid_income_l252_252838

theorem ingrid_income (combined_tax_rate : ℝ)
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_tax_rate : ℝ)
  (combined_income : ℝ)
  (combined_tax : ℝ) :
  combined_tax_rate = 0.35581395348837205 →
  john_income = 57000 →
  john_tax_rate = 0.3 →
  ingrid_tax_rate = 0.4 →
  combined_income = john_income + (combined_income - john_income) →
  combined_tax = (john_tax_rate * john_income) + (ingrid_tax_rate * (combined_income - john_income)) →
  combined_tax_rate = combined_tax / combined_income →
  combined_income = 57000 + 72000 :=
by
  sorry

end ingrid_income_l252_252838


namespace amanda_more_than_average_l252_252836

-- Conditions
def jill_peaches : ℕ := 12
def steven_peaches : ℕ := jill_peaches + 15
def jake_peaches : ℕ := steven_peaches - 16
def amanda_peaches : ℕ := jill_peaches * 2
def total_peaches : ℕ := jake_peaches + steven_peaches + jill_peaches
def average_peaches : ℚ := total_peaches / 3

-- Question: Prove that Amanda has 7.33 more peaches than the average peaches Jake, Steven, and Jill have
theorem amanda_more_than_average : amanda_peaches - average_peaches = 22 / 3 := by
  sorry

end amanda_more_than_average_l252_252836


namespace count_multiples_of_4_between_100_and_350_l252_252988

theorem count_multiples_of_4_between_100_and_350 : 
  (∃ n : ℕ, 104 + (n - 1) * 4 = 348) ∧ (∀ k : ℕ, (104 + k * 4 ∈ set.Icc 100 350) ↔ (k ≤ 61)) → 
  n = 62 :=
by
  sorry

end count_multiples_of_4_between_100_and_350_l252_252988


namespace determine_n_l252_252524

theorem determine_n (x n : ℝ) : 
  (∃ c d : ℝ, G = (c * x + d) ^ 2) ∧ (G = (8 * x^2 + 24 * x + 3 * n) / 8) → n = 6 :=
by {
  sorry
}

end determine_n_l252_252524


namespace unique_value_expression_l252_252536

theorem unique_value_expression (m n : ℤ) : 
  (mn + 13 * m + 13 * n - m^2 - n^2 = 169) → 
  ∃! (m n : ℤ), mn + 13 * m + 13 * n - m^2 - n^2 = 169 := 
by
  sorry

end unique_value_expression_l252_252536


namespace movie_theater_ticket_cost_l252_252309

theorem movie_theater_ticket_cost
  (adult_ticket_cost : ℝ)
  (child_ticket_cost : ℝ)
  (total_moviegoers : ℝ)
  (total_amount_paid : ℝ)
  (number_of_adults : ℝ)
  (H_child_ticket_cost : child_ticket_cost = 6.50)
  (H_total_moviegoers : total_moviegoers = 7)
  (H_total_amount_paid : total_amount_paid = 54.50)
  (H_number_of_adults : number_of_adults = 3)
  (H_number_of_children : total_moviegoers - number_of_adults = 4) :
  adult_ticket_cost = 9.50 :=
sorry

end movie_theater_ticket_cost_l252_252309


namespace agency_comparison_l252_252546

variable (days m : ℝ)

theorem agency_comparison (h : 20.25 * days + 0.14 * m < 18.25 * days + 0.22 * m) : m > 25 * days :=
by
  sorry

end agency_comparison_l252_252546


namespace football_team_total_players_l252_252565

/-- Let's denote the total number of players on the football team as P.
    We know that there are 31 throwers, and all of them are right-handed.
    The rest of the team is divided so one third are left-handed and the rest are right-handed.
    There are a total of 57 right-handed players on the team.
    Prove that the total number of players on the football team is 70. -/
theorem football_team_total_players 
  (P : ℕ) -- total number of players
  (T : ℕ := 31) -- number of throwers
  (L : ℕ) -- number of left-handed players
  (R : ℕ := 57) -- total number of right-handed players
  (H_all_throwers_rhs: ∀ x : ℕ, (x < P) → (x < T) → (x = T → x < R)) -- all throwers are right-handed
  (H_rest_division: ∀ x : ℕ, (x < P - T) → (x = L) → (x = 2 * L))
  : P = 70 :=
  sorry

end football_team_total_players_l252_252565


namespace solve_for_x_l252_252256
noncomputable theory

theorem solve_for_x (x : ℝ) (h : 5^(x + 6) = (5^4)^x) : x = 2 :=
by
  sorry

end solve_for_x_l252_252256


namespace find_C_probability_within_r_l252_252509

noncomputable def probability_density (x y R : ℝ) (C : ℝ) : ℝ :=
if x^2 + y^2 <= R^2 then C * (R - Real.sqrt (x^2 + y^2)) else 0

noncomputable def total_integral (R : ℝ) (C : ℝ) : ℝ :=
∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C

theorem find_C (R : ℝ) (hR : 0 < R) : 
  (∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C) = 1 ↔ 
  C = 3 / (π * R^3) := 
by 
  sorry

theorem probability_within_r (R r : ℝ) 
  (hR : 0 < R) (hr : 0 < r) (hrR : r <= R) (P : ℝ) : 
  (∫ (x : ℝ) in -r..r, ∫ (y : ℝ) in -r..r, probability_density x y R (3 / (π * R^3))) = P ↔ 
  (R = 2 ∧ r = 1 → P = 1 / 2) := 
by 
  sorry

end find_C_probability_within_r_l252_252509


namespace unique_solution_xy_l252_252640

theorem unique_solution_xy
  (x y : ℕ)
  (h1 : (x^3 + y) % (x^2 + y^2) = 0)
  (h2 : (y^3 + x) % (x^2 + y^2) = 0) :
  x = 1 ∧ y = 1 := sorry

end unique_solution_xy_l252_252640


namespace part_I_part_II_l252_252076

section problem_1

def f (x : ℝ) (a : ℝ) := |x - 3| - |x + a|

theorem part_I (x : ℝ) (hx : f x 2 < 1) : 0 < x :=
by
  sorry

theorem part_II (a : ℝ) (h : ∀ (x : ℝ), f x a ≤ 2 * a) : 3 ≤ a :=
by
  sorry

end problem_1

end part_I_part_II_l252_252076


namespace smallest_sum_of_four_consecutive_primes_divisible_by_five_l252_252325

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    b = a + 2 ∧ c = b + 4 ∧ d = c + 2 ∧
    (a + b + c + d) % 5 = 0 ∧ (a + b + c + d = 60) := sorry

end smallest_sum_of_four_consecutive_primes_divisible_by_five_l252_252325


namespace find_m_l252_252190

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x + α / x + Real.log x

theorem find_m (α : ℝ) (m : ℝ) (l e : ℝ) (hα_range : α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 2))
(h1 : f 1 α < m) (he : f (Real.exp 1) α < m) :
m > 1 + 2 * Real.exp 2 := by
  sorry

end find_m_l252_252190


namespace basketball_total_points_l252_252799

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end basketball_total_points_l252_252799


namespace length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l252_252331

def hexagon_vertex_to_center_length (a : ℝ) (h : a = 16) (regular_hexagon : Prop) : Prop :=
∃ (O A : ℝ), (a = 16) → (regular_hexagon = true) → (O = 0) ∧ (A = a) ∧ (dist O A = 16)

theorem length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16 :
  hexagon_vertex_to_center_length 16 (by rfl) true :=
sorry

end length_of_segment_from_vertex_to_center_of_regular_hexagon_is_16_l252_252331


namespace binom_12_6_l252_252929

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l252_252929


namespace relationship_of_a_and_b_l252_252181

theorem relationship_of_a_and_b (a b : ℝ) (h_b_nonzero: b ≠ 0)
  (m n : ℤ) (h_intersection : ∃ (m n : ℤ), n = m^3 - a * m^2 - b * m ∧ n = a * m + b) :
  2 * a - b + 8 = 0 :=
  sorry

end relationship_of_a_and_b_l252_252181


namespace marble_boxes_l252_252692

theorem marble_boxes (m : ℕ) : 
  (720 % m = 0) ∧ (m > 1) ∧ (720 / m > 1) ↔ m = 28 := 
sorry

end marble_boxes_l252_252692


namespace afternoon_snack_calories_l252_252323

def ellen_daily_calories : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def dinner_remaining_calories : ℕ := 832

theorem afternoon_snack_calories :
  ellen_daily_calories - (breakfast_calories + lunch_calories + dinner_remaining_calories) = 130 :=
by sorry

end afternoon_snack_calories_l252_252323


namespace Mira_trips_to_fill_tank_l252_252079

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cube (a : ℝ) : ℝ :=
  a^3

noncomputable def number_of_trips (cube_side : ℝ) (sphere_diameter : ℝ) : ℕ :=
  let r := sphere_diameter / 2
  let sphere_volume := volume_of_sphere r
  let cube_volume := volume_of_cube cube_side
  Nat.ceil (cube_volume / sphere_volume)

theorem Mira_trips_to_fill_tank : number_of_trips 8 6 = 5 :=
by
  sorry

end Mira_trips_to_fill_tank_l252_252079


namespace swap_tens_units_digits_l252_252699

theorem swap_tens_units_digits (x a b : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : a = x / 10) (h4 : b = x % 10) :
  10 * b + a = (x % 10) * 10 + (x / 10) :=
by
  sorry

end swap_tens_units_digits_l252_252699


namespace decimal_to_binary_18_l252_252162

theorem decimal_to_binary_18 : (18: ℕ) = 0b10010 := by
  sorry

end decimal_to_binary_18_l252_252162


namespace john_avg_increase_l252_252070

theorem john_avg_increase (a b c d : ℝ) (h₁ : a = 90) (h₂ : b = 85) (h₃ : c = 92) (h₄ : d = 95) :
    let initial_avg := (a + b + c) / 3
    let new_avg := (a + b + c + d) / 4
    new_avg - initial_avg = 1.5 :=
by
  sorry

end john_avg_increase_l252_252070


namespace integer_solutions_to_quadratic_inequality_l252_252881

theorem integer_solutions_to_quadratic_inequality :
  {x : ℤ | (x^2 + 6 * x + 8) * (x^2 - 4 * x + 3) < 0} = {-3, 2} :=
by
  sorry

end integer_solutions_to_quadratic_inequality_l252_252881


namespace find_n_with_divisors_conditions_l252_252553

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end find_n_with_divisors_conditions_l252_252553


namespace quadratic_expression_always_positive_l252_252971

theorem quadratic_expression_always_positive (x y : ℝ) : 
  x^2 - 4 * x * y + 6 * y^2 - 4 * y + 3 > 0 :=
by 
  sorry

end quadratic_expression_always_positive_l252_252971


namespace problem_part1_problem_part2_problem_part3_l252_252036

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem problem_part1 : f 1 = 5 / 2 ∧ f 2 = 17 / 4 := 
by
  sorry

theorem problem_part2 : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem problem_part3 : ∀ x1 x2 : ℝ, x1 < x2 → x1 < 0 → x2 < 0 → f x1 > f x2 :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l252_252036


namespace integral_result_l252_252316

open Real

theorem integral_result :
  (∫ x in (0:ℝ)..(π/2), (x^2 - 5 * x + 6) * sin (3 * x)) = (67 - 3 * π) / 27 := by
  sorry

end integral_result_l252_252316


namespace bottles_left_on_shelf_l252_252113

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l252_252113


namespace petya_wins_in_two_moves_l252_252723

theorem petya_wins_in_two_moves (chosen_positions : Fin 8 → Fin 8) :
  ∃ (moves : List (Fin 8 → Fin 8)), moves.length ≤ 2 ∧
  ∀ move ∈ moves, (rooks_placed move).length = 8 ∧
  (∀ i j, i ≠ j → move i ≠ move j) ∧  -- No two rooks in the same row or column
  win_condition chosen_positions move :=
by {
  sorry
}

end petya_wins_in_two_moves_l252_252723


namespace problem_statement_l252_252386

def line : Type := sorry
def plane : Type := sorry

def perpendicular (l : line) (p : plane) : Prop := sorry
def parallel (l1 l2 : line) : Prop := sorry

variable (m n : line)
variable (α β : plane)

theorem problem_statement (h1 : perpendicular m α) 
                          (h2 : parallel m n) 
                          (h3 : parallel n β) : 
                          perpendicular α β := 
sorry

end problem_statement_l252_252386


namespace charity_distribution_l252_252609

theorem charity_distribution 
  (X : ℝ) (Y : ℝ) (Z : ℝ) (W : ℝ) (A : ℝ)
  (h1 : X > 0) (h2 : Y > 0) (h3 : Y < 100) (h4 : Z > 0) (h5 : W > 0) (h6 : A > 0)
  (h7 : W * A = X * (100 - Y) / 100) :
  (Y * X) / (100 * Z) = A * W * Y / (100 * Z) :=
by 
  sorry

end charity_distribution_l252_252609


namespace max_median_cans_per_customer_l252_252697

theorem max_median_cans_per_customer : 
    ∀ (total_cans : ℕ) (total_customers : ℕ), 
    total_cans = 252 → total_customers = 100 →
    (∀ (cans_per_customer : ℕ),
    1 ≤ cans_per_customer) →
    (∃ (max_median : ℝ),
    max_median = 3.5) :=
by
  sorry

end max_median_cans_per_customer_l252_252697


namespace value_of_y_at_64_l252_252496

theorem value_of_y_at_64 (x y k : ℝ) (h1 : y = k * x^(1/3)) (h2 : 8^(1/3) = 2) (h3 : y = 4 ∧ x = 8):
  y = 8 :=
by {
  sorry
}

end value_of_y_at_64_l252_252496


namespace rons_chocolate_cost_l252_252425

theorem rons_chocolate_cost :
  let cost_per_bar := 1.5
  let sections_per_bar := 3
  let scouts := 15
  let smores_per_scout := 2
  let total_smores := scouts * smores_per_scout
  let bars_needed := total_smores / sections_per_bar in
  bars_needed * cost_per_bar = 15.0 := by
  sorry

end rons_chocolate_cost_l252_252425


namespace binom_12_6_eq_924_l252_252942

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l252_252942


namespace inequality_for_positive_reals_l252_252084

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) :=
by
  sorry

end inequality_for_positive_reals_l252_252084


namespace determine_a_l252_252499

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem determine_a (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = 1 / 2 :=
by
  sorry

end determine_a_l252_252499


namespace rose_price_vs_carnation_price_l252_252282

variable (x y : ℝ)

theorem rose_price_vs_carnation_price
  (h1 : 3 * x + 2 * y > 8)
  (h2 : 2 * x + 3 * y < 7) :
  x > 2 * y :=
sorry

end rose_price_vs_carnation_price_l252_252282


namespace fraction_representation_correct_l252_252483

theorem fraction_representation_correct (h : ∀ (x y z w: ℕ), 9*x = y ∧ 47*z = w ∧ 2*47*5 = 235):
  (18: ℚ) / (9 * 47 * 5) = (2: ℚ) / 235 :=
by
  sorry

end fraction_representation_correct_l252_252483


namespace base_729_base8_l252_252011

theorem base_729_base8 (b : ℕ) (X Y : ℕ) (h_distinct : X ≠ Y)
  (h_range : b^3 ≤ 729 ∧ 729 < b^4)
  (h_form : 729 = X * b^3 + Y * b^2 + X * b + Y) : b = 8 :=
sorry

end base_729_base8_l252_252011


namespace value_of_x_plus_y_pow_2023_l252_252477

theorem value_of_x_plus_y_pow_2023 (x y : ℝ) (h : abs (x - 2) + abs (y + 3) = 0) : 
  (x + y) ^ 2023 = -1 := 
sorry

end value_of_x_plus_y_pow_2023_l252_252477


namespace sum_of_remainders_l252_252570

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end sum_of_remainders_l252_252570


namespace kenya_peanuts_l252_252518

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l252_252518


namespace cos_sin_gt_sin_cos_l252_252661

theorem cos_sin_gt_sin_cos (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi) : Real.cos (Real.sin x) > Real.sin (Real.cos x) :=
by
  sorry

end cos_sin_gt_sin_cos_l252_252661


namespace tolu_pencils_l252_252861

theorem tolu_pencils (price_per_pencil : ℝ) (robert_pencils : ℕ) (melissa_pencils : ℕ) (total_money_spent : ℝ) (tolu_pencils : ℕ) :
  price_per_pencil = 0.20 →
  robert_pencils = 5 →
  melissa_pencils = 2 →
  total_money_spent = 2.00 →
  tolu_pencils * price_per_pencil = 2.00 - (5 * 0.20 + 2 * 0.20) →
  tolu_pencils = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end tolu_pencils_l252_252861


namespace william_napkins_l252_252853

-- Define the given conditions
variables (O A C G W : ℕ)
variables (ho: O = 10)
variables (ha: A = 2 * O)
variables (hc: C = A / 2)
variables (hg: G = 3 * C)
variables (hw: W = 15)

-- Prove the total number of napkins William has now
theorem william_napkins (O A C G W : ℕ) (ho: O = 10) (ha: A = 2 * O)
  (hc: C = A / 2) (hg: G = 3 * C) (hw: W = 15) : W + (O + A + C + G) = 85 :=
by {
  sorry
}

end william_napkins_l252_252853


namespace domain_range_of_p_l252_252750

variable (h : ℝ → ℝ)
variable (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3)
variable (h_range : ∀ x, 0 ≤ h x ∧ h x ≤ 2)

def p (x : ℝ) : ℝ := 2 - h (x - 1)

theorem domain_range_of_p :
  (∀ x, 0 ≤ x ∧ x ≤ 4) ∧ (∀ y, 0 ≤ y ∧ y ≤ 2) :=
by
  -- Proof to show that the domain of p(x) is [0, 4] and the range is [0, 2]
  sorry

end domain_range_of_p_l252_252750


namespace candy_last_days_l252_252740

theorem candy_last_days (candy_neighbors candy_sister candy_per_day : ℕ)
  (h1 : candy_neighbors = 5)
  (h2 : candy_sister = 13)
  (h3 : candy_per_day = 9):
  (candy_neighbors + candy_sister) / candy_per_day = 2 :=
by
  sorry

end candy_last_days_l252_252740


namespace amber_total_cost_l252_252603

/-
Conditions:
1. Base cost of the plan: $25.
2. Cost for text messages with different rates for the first 120 messages and additional messages.
3. Cost for additional talk time.
4. Given specific usage data for Amber in January.

Objective:
Prove that the total monthly cost for Amber is $47.
-/
noncomputable def base_cost : ℕ := 25
noncomputable def text_message_cost (total_messages : ℕ) : ℕ :=
  if total_messages <= 120 then
    3 * total_messages
  else
    3 * 120 + 2 * (total_messages - 120)

noncomputable def talk_time_cost (talk_hours : ℕ) : ℕ :=
  if talk_hours <= 25 then
    0
  else
    15 * 60 * (talk_hours - 25)

noncomputable def total_monthly_cost (total_messages : ℕ) (talk_hours : ℕ) : ℕ :=
  base_cost + ((text_message_cost total_messages) / 100) + ((talk_time_cost talk_hours) / 100)

theorem amber_total_cost : total_monthly_cost 140 27 = 47 := by
  sorry

end amber_total_cost_l252_252603


namespace time_ratio_l252_252437

theorem time_ratio (A : ℝ) (B : ℝ) (h1 : B = 18) (h2 : 1 / A + 1 / B = 1 / 3) : A / B = 1 / 5 :=
by
  sorry

end time_ratio_l252_252437


namespace geometric_seq_relation_l252_252786

variables {α : Type*} [Field α]

-- Conditions for the arithmetic sequence (for reference)
def arithmetic_seq_sum (S : ℕ → α) (d : α) : Prop :=
∀ m n : ℕ, S (m + n) = S m + S n + (m * n) * d

-- Conditions for the geometric sequence
def geometric_seq_prod (T : ℕ → α) (q : α) : Prop :=
∀ m n : ℕ, T (m + n) = T m * T n * (q ^ (m * n))

-- Proving the desired relationship
theorem geometric_seq_relation {T : ℕ → α} {q : α} (h : geometric_seq_prod T q) (m n : ℕ) :
  T (m + n) = T m * T n * (q ^ (m * n)) :=
by
  apply h m n

end geometric_seq_relation_l252_252786


namespace plot_length_l252_252401

variable (b length : ℝ)

theorem plot_length (h1 : length = b + 10)
  (fence_N_cost : ℝ := 26.50 * (b + 10))
  (fence_E_cost : ℝ := 32 * b)
  (fence_S_cost : ℝ := 22 * (b + 10))
  (fence_W_cost : ℝ := 30 * b)
  (total_cost : ℝ := fence_N_cost + fence_E_cost + fence_S_cost + fence_W_cost)
  (h2 : 1.05 * total_cost = 7500) :
  length = 70.25 := by
  sorry

end plot_length_l252_252401


namespace union_of_A_and_B_l252_252777

def A : Set ℤ := {-1, 0, 2}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l252_252777


namespace range_of_a_l252_252675

theorem range_of_a (a : Real) : 
  (∀ x y : Real, (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0 → x < 0 ∧ y > 0)) ↔ (a > 2) := 
sorry

end range_of_a_l252_252675


namespace right_triangle_shorter_leg_l252_252813

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l252_252813


namespace meal_serving_problem_l252_252893

/-
Twelve people sit down for dinner where there are three choices of meals: beef, chicken, and fish.
Four people order beef, four people order chicken, and four people order fish.
The waiter serves the twelve meals in random order.
We need to find the number of ways in which the waiter could serve the meals so that exactly two people receive the type of meal ordered by them.
-/
theorem meal_serving_problem :
    ∃ (n : ℕ), n = 12210 ∧
    (∃ (people : Fin 12 → char), 
        (∀ i : Fin 4, people i = 'B') ∧ 
        (∀ i : Fin 4, people (i + 4) = 'C') ∧ 
        (∀ i : Fin 4, people (i + 8) = 'F') ∧ 
        (∃ (served : Fin 12 → char), 
            (∃ (correct : Fin 12), set.range correct ⊆ {0, 1} ∧
            (∀ i : Fin 12, (served i = people correct i) ↔ (i ∈ {0, 1}) = true)) ∧
            (related_permutations served people))
    )
    sorry

end meal_serving_problem_l252_252893


namespace roots_of_poly_l252_252026

noncomputable def poly : Polynomial ℝ := 8 * (Polynomial.monomial 4 1) + 14 * (Polynomial.monomial 3 1) - 66 * (Polynomial.monomial 2 1) + 40 * (Polynomial.monomial 1 1)

theorem roots_of_poly : {0, 1 / 2, 2, -5} = {x : ℝ | poly.eval x = 0} :=
by {
  sorry
}

end roots_of_poly_l252_252026


namespace average_weight_increase_l252_252505

theorem average_weight_increase (W_new : ℝ) (W_old : ℝ) (num_persons : ℝ): 
  W_new = 94 ∧ W_old = 70 ∧ num_persons = 8 → 
  (W_new - W_old) / num_persons = 3 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end average_weight_increase_l252_252505


namespace incident_ray_slope_in_circle_problem_l252_252668

noncomputable def slope_of_incident_ray : ℚ := sorry

theorem incident_ray_slope_in_circle_problem :
  ∃ (P : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ),
  P = (-1, -3) ∧
  C = (2, -1) ∧
  (D = (C.1, -C.2)) ∧
  (D = (2, 1)) ∧
  ∀ (m : ℚ), (m = (D.2 - P.2) / (D.1 - P.1)) → m = 4 / 3 := 
sorry

end incident_ray_slope_in_circle_problem_l252_252668


namespace exist_tangent_circles_l252_252099

noncomputable def locus_centers_of_tangent_circles (a b : ℝ) : Prop :=
  40 * a ^ 2 + 49 * b ^ 2 - 48 * a - 64 = 0

theorem exist_tangent_circles (a b : ℝ) :
  (∀ r ≥ 0, ∃ C, (C.center = ⟨a, b⟩ ∧ C.radius = r ∧ externally_tangent C C1 ∧ internally_tangent C C3)) → 
  locus_centers_of_tangent_circles a b := 
sorry

end exist_tangent_circles_l252_252099


namespace total_amount_spent_l252_252922

def cost_of_tshirt : ℕ := 100
def cost_of_pants : ℕ := 250
def num_of_tshirts : ℕ := 5
def num_of_pants : ℕ := 4

theorem total_amount_spent : (num_of_tshirts * cost_of_tshirt) + (num_of_pants * cost_of_pants) = 1500 := by
  sorry

end total_amount_spent_l252_252922


namespace count_multiples_of_four_between_100_and_350_l252_252990

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end count_multiples_of_four_between_100_and_350_l252_252990


namespace chameleons_color_change_l252_252211

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l252_252211


namespace tank_fewer_eggs_in_second_round_l252_252207

variables (T E_total T_r2_diff : ℕ)

theorem tank_fewer_eggs_in_second_round
  (h1 : E_total = 400)
  (h2 : E_total = (T + (T - 10)) + (30 + 60))
  (h3 : T_r2_diff = T - 30) :
  T_r2_diff = 130 := by
    sorry

end tank_fewer_eggs_in_second_round_l252_252207


namespace Shiela_drawings_l252_252090

theorem Shiela_drawings (n_neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
    (h1 : n_neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
by 
  sorry

end Shiela_drawings_l252_252090


namespace constant_term_expansion_l252_252794

theorem constant_term_expansion (n : ℕ) (hn : n = 9) :
  y^3 * (x + 1 / (x^2 * y))^n = 84 :=
by sorry

end constant_term_expansion_l252_252794


namespace trig_identity_sum_l252_252741

-- Define the trigonometric functions and their properties
def sin_210_eq : Real.sin (210 * Real.pi / 180) = - Real.sin (30 * Real.pi / 180) := by
  sorry

def cos_60_eq : Real.cos (60 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) := by
  sorry

-- The goal is to prove that the sum of these specific trigonometric values is 0
theorem trig_identity_sum : Real.sin (210 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) = 0 := by
  rw [sin_210_eq, cos_60_eq]
  sorry

end trig_identity_sum_l252_252741


namespace problem_conditions_l252_252493

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end problem_conditions_l252_252493


namespace find_coefficients_l252_252554

noncomputable def polynomial_h (x : ℚ) : ℚ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def polynomial_j (b c d x : ℚ) : ℚ := x^3 + b * x^2 + c * x + d

theorem find_coefficients :
  (∃ b c d : ℚ,
     (∀ s : ℚ, polynomial_h s = 0 → polynomial_j b c d (s^3) = 0) ∧
     (b, c, d) = (6, 12, 8)) :=
sorry

end find_coefficients_l252_252554


namespace quadratic_inequality_solution_l252_252795

theorem quadratic_inequality_solution (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : a * 2^2 + b * 2 + c = 0) 
  (h3 : a * (-1)^2 + b * (-1) + c = 0) :
  ∀ x, ax^2 + bx + c ≥ 0 ↔ (-1 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end quadratic_inequality_solution_l252_252795


namespace problem_1_problem_2_l252_252237

noncomputable def O := (0, 0)
noncomputable def A := (1, 2)
noncomputable def B := (-3, 4)

noncomputable def vector_AB := (B.1 - A.1, B.2 - A.2)
noncomputable def magnitude_AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def dot_OA_OB := A.1 * B.1 + A.2 * B.2
noncomputable def magnitude_OA := Real.sqrt (A.1^2 + A.2^2)
noncomputable def magnitude_OB := Real.sqrt (B.1^2 + B.2^2)
noncomputable def cosine_angle := dot_OA_OB / (magnitude_OA * magnitude_OB)

theorem problem_1 : vector_AB = (-4, 2) ∧ magnitude_AB = 2 * Real.sqrt 5 := sorry

theorem problem_2 : cosine_angle = Real.sqrt 5 / 5 := sorry

end problem_1_problem_2_l252_252237


namespace Sam_has_most_pages_l252_252532

theorem Sam_has_most_pages :
  let pages_per_inch_miles := 5
  let inches_miles := 240
  let pages_per_inch_daphne := 50
  let inches_daphne := 25
  let pages_per_inch_sam := 30
  let inches_sam := 60

  let pages_miles := inches_miles * pages_per_inch_miles
  let pages_daphne := inches_daphne * pages_per_inch_daphne
  let pages_sam := inches_sam * pages_per_inch_sam
  pages_sam = 1800 ∧ pages_sam > pages_miles ∧ pages_sam > pages_daphne :=
by
  sorry

end Sam_has_most_pages_l252_252532


namespace neither_sufficient_nor_necessary_l252_252178

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0) ↔ (ab < ((a + b) / 2)^2)) :=
sorry

end neither_sufficient_nor_necessary_l252_252178


namespace final_number_independent_of_order_l252_252503

theorem final_number_independent_of_order 
  (p q r : ℕ) : 
  ∃ k : ℕ, 
    (p % 2 ≠ 0 ∨ q % 2 ≠ 0 ∨ r % 2 ≠ 0) ∧ 
    (∀ (p' q' r' : ℕ), 
       p' + q' + r' = p + q + r → 
       p' % 2 = p % 2 ∧ q' % 2 = q % 2 ∧ r' % 2 = r % 2 → 
       (p' = 1 ∧ q' = 0 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 1 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 0 ∧ r' = 1) → 
       k = p ∨ k = q ∨ k = r) := 
sorry

end final_number_independent_of_order_l252_252503


namespace work_done_correct_l252_252567

open Real

noncomputable def work_done (a b g : ℝ) : ℝ :=
  -- Define potential function based on gravitational field
  let ϕ := fun (z : ℝ) => -g * z
  -- Calculate potential at point A and point B
  let ϕ_A := ϕ (2 * π * b)
  let ϕ_B := ϕ 0
  ϕ_B - ϕ_A

theorem work_done_correct (a b g : ℝ) : work_done a b g = 2 * π * g * b :=
by
  -- Simplify potential difference
  simp [work_done]
  unfold ϕ
  norm_num

end work_done_correct_l252_252567


namespace A_gt_B_and_C_lt_A_l252_252128

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end A_gt_B_and_C_lt_A_l252_252128


namespace ron_chocolate_bar_cost_l252_252427

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l252_252427


namespace chameleons_changed_color_l252_252231

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l252_252231


namespace four_digit_multiples_of_7_l252_252351

theorem four_digit_multiples_of_7 : 
  ∃ n : ℕ, n = (9999 / 7).toNat - (1000 / 7).toNat + 1 ∧ n = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l252_252351


namespace min_expression_value_l252_252474

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end min_expression_value_l252_252474


namespace mr_thompson_third_score_is_78_l252_252080

theorem mr_thompson_third_score_is_78 :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
                   (a = 58 ∧ b = 65 ∧ c = 70 ∧ d = 78) ∧ 
                   (a + b + c + d) % 4 = 3 ∧ 
                   (∀ i j k, (a + i + j + k) % 4 = 0) ∧ -- This checks that average is integer
                   c = 78 := sorry

end mr_thompson_third_score_is_78_l252_252080


namespace annalise_spending_l252_252446

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end annalise_spending_l252_252446


namespace g_g_g_25_l252_252528

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 18

theorem g_g_g_25 :
  g (g (g 25)) = 22 :=
by
  sorry

end g_g_g_25_l252_252528


namespace probability_at_least_one_hit_l252_252745

variable (P₁ P₂ : ℝ)

theorem probability_at_least_one_hit (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
by
  sorry

end probability_at_least_one_hit_l252_252745


namespace count_four_digit_multiples_of_7_l252_252358

theorem count_four_digit_multiples_of_7 : 
    (card {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}) = 1286 :=
sorry

end count_four_digit_multiples_of_7_l252_252358


namespace product_of_possible_values_of_x_l252_252365

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end product_of_possible_values_of_x_l252_252365


namespace find_ratio_of_sums_l252_252180

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = n * (a 1 + a n) / 2

def ratio_condition (a : ℕ → ℝ) :=
  a 6 / a 5 = 9 / 11

theorem find_ratio_of_sums (seq : ∃ d, arithmetic_sequence a d)
    (sum_prop : sum_first_n_terms S a)
    (ratio_prop : ratio_condition a) :
  S 11 / S 9 = 1 :=
sorry

end find_ratio_of_sums_l252_252180


namespace radius_relation_l252_252160

-- Define the conditions under which the spheres exist
variable {R r : ℝ}

-- The problem statement
theorem radius_relation (h : r = R * (2 - Real.sqrt 2)) : r = R * (2 - Real.sqrt 2) :=
sorry

end radius_relation_l252_252160


namespace ellen_legos_final_count_l252_252634

-- Definitions based on conditions
def initial_legos : ℕ := 380
def lost_legos_first_week : ℕ := 57
def additional_legos_second_week (remaining_legos : ℕ) : ℕ := 32
def borrowed_legos_third_week (total_legos : ℕ) : ℕ := 88

-- Computed values based on conditions
def legos_after_first_week (initial : ℕ) (lost : ℕ) : ℕ := initial - lost
def legos_after_second_week (remaining : ℕ) (additional : ℕ) : ℕ := remaining + additional
def legos_after_third_week (total : ℕ) (borrowed : ℕ) : ℕ := total - borrowed

-- Proof statement
theorem ellen_legos_final_count : 
  legos_after_third_week 
    (legos_after_second_week 
      (legos_after_first_week initial_legos lost_legos_first_week)
      (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))
    (borrowed_legos_third_week (legos_after_second_week 
                                  (legos_after_first_week initial_legos lost_legos_first_week)
                                  (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))) 
  = 267 :=
by 
  sorry

end ellen_legos_final_count_l252_252634


namespace figure_100_squares_l252_252249

theorem figure_100_squares :
  ∀ (f : ℕ → ℕ),
    (f 0 = 1) →
    (f 1 = 6) →
    (f 2 = 17) →
    (f 3 = 34) →
    f 100 = 30201 :=
by
  intros f h0 h1 h2 h3
  sorry

end figure_100_squares_l252_252249


namespace Jasmine_gets_off_work_at_4pm_l252_252376

-- Conditions
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_clean_time : ℕ := 10
def groomer_time : ℕ := 20
def cook_time : ℕ := 90
def dinner_time : ℕ := 19 * 60  -- 7:00 pm in minutes

-- Question to prove
theorem Jasmine_gets_off_work_at_4pm : 
  (dinner_time - cook_time - groomer_time - dry_clean_time - grocery_time - commute_time = 16 * 60) := sorry

end Jasmine_gets_off_work_at_4pm_l252_252376


namespace product_base9_conversion_l252_252645

noncomputable def base_9_to_base_10 (n : ℕ) : ℕ :=
match n with
| 237 => 2 * 9^2 + 3 * 9^1 + 7
| 17 => 9 + 7
| _ => 0

noncomputable def base_10_to_base_9 (n : ℕ) : ℕ :=
match n with
-- Step of conversion from example: 3136 => 4*9^3 + 2*9^2 + 6*9^1 + 4*9^0
| 3136 => 4 * 1000 + 2 * 100 + 6 * 10 + 4 -- representing 4264 in base 9
| _ => 0

theorem product_base9_conversion :
  base_10_to_base_9 ((base_9_to_base_10 237) * (base_9_to_base_10 17)) = 4264 := by
  sorry

end product_base9_conversion_l252_252645


namespace chameleons_color_change_l252_252221

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l252_252221


namespace general_admission_tickets_l252_252722

-- Define the number of student tickets and general admission tickets
variables {S G : ℕ}

-- Define the conditions
def tickets_sold (S G : ℕ) : Prop := S + G = 525
def amount_collected (S G : ℕ) : Prop := 4 * S + 6 * G = 2876

-- The theorem to prove that the number of general admission tickets is 388
theorem general_admission_tickets : 
  ∀ (S G : ℕ), tickets_sold S G → amount_collected S G → G = 388 :=
by
  sorry -- Proof to be provided

end general_admission_tickets_l252_252722


namespace solve_system_l252_252042

theorem solve_system :
  ∃ (x y z : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y - 4 * z = 10 ∧ 2 * x + y + 3 * z = 1.25 :=
by
  sorry

end solve_system_l252_252042


namespace binomial_12_6_eq_924_l252_252936

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l252_252936


namespace cylinder_height_l252_252708

   theorem cylinder_height (r h : ℝ) (SA : ℝ) (π : ℝ) :
     r = 3 → SA = 30 * π → SA = 2 * π * r^2 + 2 * π * r * h → h = 2 :=
   by
     intros hr hSA hSA_formula
     rw [hr] at hSA_formula
     rw [hSA] at hSA_formula
     sorry
   
end cylinder_height_l252_252708


namespace reinforcement_correct_l252_252419

-- Conditions
def initial_men : ℕ := 2000
def initial_days : ℕ := 54
def days_before_reinforcement : ℕ := 18
def days_after_reinforcement : ℕ := 20

-- Define the remaining provisions after 18 days
def provisions_left : ℕ := initial_men * (initial_days - days_before_reinforcement)

-- Define reinforcement
def reinforcement : ℕ := 
  sorry -- placeholder for the definition

-- Theorem to prove
theorem reinforcement_correct :
  reinforcement = 1600 :=
by
  -- Use the given conditions to derive the reinforcement value
  let total_provision := initial_men * initial_days
  let remaining_provision := provisions_left
  let men_after_reinforcement := initial_men + reinforcement
  have h := remaining_provision = men_after_reinforcement * days_after_reinforcement
  sorry -- placeholder for the proof

end reinforcement_correct_l252_252419


namespace kerosene_cost_l252_252292

/-- Given that:
    - A dozen eggs cost as much as a pound of rice.
    - A half-liter of kerosene costs as much as 8 eggs.
    - The cost of each pound of rice is $0.33.
    - One dollar has 100 cents.
Prove that a liter of kerosene costs 44 cents.
-/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12  -- Cost per egg in dollars
  let kerosene_half_liter_cost := egg_cost * 8  -- Half-liter of kerosene cost in dollars
  let kerosene_liter_cost := kerosene_half_liter_cost * 2  -- Liter of kerosene cost in dollars
  let kerosene_liter_cost_cents := kerosene_liter_cost * 100  -- Liter of kerosene cost in cents
  kerosene_liter_cost_cents = 44 :=
by
  sorry

end kerosene_cost_l252_252292


namespace shorter_leg_of_right_triangle_l252_252805

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252805


namespace single_cakes_needed_l252_252253

theorem single_cakes_needed :
  ∀ (layer_cake_frosting single_cake_frosting cupcakes_frosting brownies_frosting : ℝ)
  (layer_cakes cupcakes brownies total_frosting : ℕ)
  (single_cakes_needed : ℝ),
  layer_cake_frosting = 1 →
  single_cake_frosting = 0.5 →
  cupcakes_frosting = 0.5 →
  brownies_frosting = 0.5 →
  layer_cakes = 3 →
  cupcakes = 6 →
  brownies = 18 →
  total_frosting = 21 →
  single_cakes_needed = (total_frosting - (layer_cakes * layer_cake_frosting + cupcakes * cupcakes_frosting + brownies * brownies_frosting)) / single_cake_frosting →
  single_cakes_needed = 12 :=
by
  intros
  sorry

end single_cakes_needed_l252_252253


namespace percentage_increase_l252_252157

theorem percentage_increase (M N : ℝ) (h : M ≠ N) : 
  (200 * (M - N) / (M + N) = ((200 : ℝ) * (M - N) / (M + N))) :=
by
  -- Translate the problem conditions into Lean definitions
  let average := (M + N) / 2
  let increase := (M - N)
  let fraction_of_increase_over_average := (increase / average) * 100

  -- Additional annotations and calculations to construct the proof would go here
  sorry

end percentage_increase_l252_252157


namespace raspberry_pie_degrees_l252_252371

def total_students : ℕ := 48
def chocolate_preference : ℕ := 18
def apple_preference : ℕ := 10
def blueberry_preference : ℕ := 8
def remaining_students : ℕ := total_students - chocolate_preference - apple_preference - blueberry_preference
def raspberry_preference : ℕ := remaining_students / 2
def pie_chart_degrees : ℕ := (raspberry_preference * 360) / total_students

theorem raspberry_pie_degrees :
  pie_chart_degrees = 45 := by
  sorry

end raspberry_pie_degrees_l252_252371


namespace triangle_inequality_l252_252855

theorem triangle_inequality
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l252_252855


namespace smallest_b_for_fraction_eq_l252_252424

theorem smallest_b_for_fraction_eq (a b : ℕ) (h1 : 1000 ≤ a ∧ a < 10000) (h2 : 100000 ≤ b ∧ b < 1000000)
(h3 : 1/2006 = 1/a + 1/b) : b = 120360 := sorry

end smallest_b_for_fraction_eq_l252_252424


namespace min_sum_a_b_l252_252339

-- The conditions
variables {a b : ℝ}
variables (h₁ : a > 1) (h₂ : b > 1) (h₃ : ab - (a + b) = 1)

-- The theorem statement
theorem min_sum_a_b : a + b = 2 + 2 * Real.sqrt 2 :=
sorry

end min_sum_a_b_l252_252339


namespace common_root_l252_252859

def f (x : ℝ) : ℝ := x^4 - x^3 - 22 * x^2 + 16 * x + 96
def g (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 10

theorem common_root :
  f (-2) = 0 ∧ g (-2) = 0 := by
  sorry

end common_root_l252_252859


namespace train_crossing_time_l252_252903

theorem train_crossing_time :
  ∀ (length_train1 length_train2 : ℕ) 
    (speed_train1_kmph speed_train2_kmph : ℝ), 
  length_train1 = 420 →
  speed_train1_kmph = 72 →
  length_train2 = 640 →
  speed_train2_kmph = 36 →
  (length_train1 + length_train2) / ((speed_train1_kmph - speed_train2_kmph) * (1000 / 3600)) = 106 :=
by
  intros
  sorry

end train_crossing_time_l252_252903


namespace k_value_for_polynomial_l252_252991

theorem k_value_for_polynomial (k : ℤ) :
  (3 : ℤ)^3 + k * (3 : ℤ) - 18 = 0 → k = -3 :=
by
  sorry

end k_value_for_polynomial_l252_252991


namespace line_through_two_points_l252_252263

-- Define the points
def p1 : ℝ × ℝ := (1, 0)
def p2 : ℝ × ℝ := (0, -2)

-- Define the equation of the line passing through the points
def line_equation (x y : ℝ) : Prop :=
  2 * x - y - 2 = 0

-- The main theorem
theorem line_through_two_points : ∀ x y, p1 = (1, 0) ∧ p2 = (0, -2) → line_equation x y :=
  by sorry

end line_through_two_points_l252_252263


namespace spadesuit_example_l252_252010

-- Define the operation spadesuit
def spadesuit (a b : ℤ) : ℤ := abs (a - b)

-- Define the specific instance to prove
theorem spadesuit_example : spadesuit 2 (spadesuit 4 7) = 1 :=
by
  sorry

end spadesuit_example_l252_252010


namespace linear_eq_m_value_l252_252657

theorem linear_eq_m_value (x m : ℝ) (h : 2 * x + m = 5) (hx : x = 1) : m = 3 :=
by
  -- Here we would carry out the proof steps
  sorry

end linear_eq_m_value_l252_252657


namespace solve_for_x_l252_252255

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(x + 6) = (625 : ℝ)^x → x = 2 :=
by
  sorry

end solve_for_x_l252_252255


namespace zombies_count_decrease_l252_252887

theorem zombies_count_decrease (z : ℕ) (d : ℕ) : z = 480 → (∀ n, d = 2^n * z) → ∃ t, d / t < 50 :=
by
  intros hz hdz
  let initial_count := 480
  have := 480 / (2 ^ 4)
  sorry

end zombies_count_decrease_l252_252887


namespace min_squared_distance_l252_252985

open Real

theorem min_squared_distance : ∀ (x y : ℝ), (3 * x + y = 10) → (x^2 + y^2) ≥ 10 :=
by
  intros x y hxy
  -- Insert the necessary steps or key elements here
  sorry

end min_squared_distance_l252_252985


namespace beverage_price_l252_252561

theorem beverage_price (P : ℝ) :
  (3 * 2.25 + 4 * P + 4 * 1.00) / 6 = 2.79 → P = 1.50 :=
by
  intro h -- Introduce the hypothesis.
  sorry  -- Proof is omitted.

end beverage_price_l252_252561


namespace smallest_digit_divisible_by_9_l252_252028

theorem smallest_digit_divisible_by_9 : 
  ∃ (d : ℕ), 0 ≤ d ∧ d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ d = 2 :=
by
  use 2
  split
  { exact nat.zero_le _ }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end smallest_digit_divisible_by_9_l252_252028


namespace matrix_multiplication_correct_l252_252617

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 6]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![17, -3], ![16, -24]]

theorem matrix_multiplication_correct : A * B = C := by 
  sorry

end matrix_multiplication_correct_l252_252617


namespace find_m_l252_252643

-- Define the condition for m to be within the specified range
def valid_range (m : ℤ) : Prop := -180 < m ∧ m < 180

-- Define the relationship with the trigonometric equation to be proven
def tan_eq (m : ℤ) : Prop := Real.tan (m * Real.pi / 180) = Real.tan (1500 * Real.pi / 180)

-- State the main theorem to be proved
theorem find_m (m : ℤ) (h1 : valid_range m) (h2 : tan_eq m) : m = 60 :=
sorry

end find_m_l252_252643


namespace chameleon_color_change_l252_252227

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l252_252227


namespace min_value_a_squared_plus_b_squared_l252_252065

theorem min_value_a_squared_plus_b_squared :
  ∃ (a b : ℝ), (b = 3 * a - 6) → (a^2 + b^2 = 18 / 5) :=
by
  sorry

end min_value_a_squared_plus_b_squared_l252_252065


namespace tan_diff_l252_252792

theorem tan_diff (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) : Real.tan (α - β) = 1/3 := 
sorry

end tan_diff_l252_252792


namespace fraction_of_b_equals_4_15_of_a_is_0_4_l252_252136

variable (A B : ℤ)
variable (X : ℚ)

def a_and_b_together_have_1210 : Prop := A + B = 1210
def b_has_484 : Prop := B = 484
def fraction_of_b_equals_4_15_of_a : Prop := (4 / 15 : ℚ) * A = X * B

theorem fraction_of_b_equals_4_15_of_a_is_0_4
  (h1 : a_and_b_together_have_1210 A B)
  (h2 : b_has_484 B)
  (h3 : fraction_of_b_equals_4_15_of_a A B X) :
  X = 0.4 := sorry

end fraction_of_b_equals_4_15_of_a_is_0_4_l252_252136


namespace fourth_is_20_fewer_than_third_l252_252299

-- Definitions of the number of road signs at each intersection
def first_intersection := 40
def second_intersection := first_intersection + first_intersection / 4
def third_intersection := 2 * second_intersection
def total_signs := 270
def fourth_intersection := total_signs - (first_intersection + second_intersection + third_intersection)

-- Proving the fourth intersection has 20 fewer signs than the third intersection
theorem fourth_is_20_fewer_than_third : third_intersection - fourth_intersection = 20 :=
by
  -- This is a placeholder for the proof
  sorry

end fourth_is_20_fewer_than_third_l252_252299


namespace tetrahedron_sphere_surface_area_l252_252150

-- Define the conditions
variables (a : ℝ) (mid_AB_C : ℝ → Prop) (S : ℝ)
variables (h1 : a > 0)
variables (h2 : mid_AB_C a)
variables (h3 : S = 3 * Real.sqrt 2)

-- Theorem statement
theorem tetrahedron_sphere_surface_area (h1 : a = 2 * Real.sqrt 3) : 
  4 * Real.pi * ( (Real.sqrt 6 / 4) * a )^2 = 18 * Real.pi := by
  sorry

end tetrahedron_sphere_surface_area_l252_252150


namespace find_value_of_E_l252_252322

variables (Q U I E T Z : ℤ)

theorem find_value_of_E (hZ : Z = 15) (hQUIZ : Q + U + I + Z = 60) (hQUIET : Q + U + I + E + T = 75) (hQUIT : Q + U + I + T = 50) : E = 25 :=
by
  have hQUIZ_val : Q + U + I = 45 := by linarith [hZ, hQUIZ]
  have hQUIET_val : E + T = 30 := by linarith [hQUIZ_val, hQUIET]
  have hQUIT_val : T = 5 := by linarith [hQUIZ_val, hQUIT]
  linarith [hQUIET_val, hQUIT_val]

end find_value_of_E_l252_252322


namespace kangaroo_mob_has_6_l252_252372

-- Define the problem conditions
def mob_of_kangaroos (W : ℝ) (k : ℕ) : Prop :=
  ∃ (two_lightest three_heaviest remaining : ℝ) (n_two n_three n_rem : ℕ),
    two_lightest = 0.25 * W ∧
    three_heaviest = 0.60 * W ∧
    remaining = 0.15 * W ∧
    n_two = 2 ∧
    n_three = 3 ∧
    n_rem = 1 ∧
    k = n_two + n_three + n_rem

-- The theorem to be proven
theorem kangaroo_mob_has_6 (W : ℝ) : ∃ k, mob_of_kangaroos W k ∧ k = 6 :=
by
  sorry

end kangaroo_mob_has_6_l252_252372


namespace book_original_price_l252_252746

-- Definitions for conditions
def selling_price := 56
def profit_percentage := 75

-- Statement of the theorem
theorem book_original_price : ∃ CP : ℝ, selling_price = CP * (1 + profit_percentage / 100) ∧ CP = 32 :=
by
  sorry

end book_original_price_l252_252746


namespace even_sum_probability_l252_252409

-- Conditions
def first_wheel_total_sections := 5
def first_wheel_even_sections := 2
def second_wheel_total_sections := 4
def second_wheel_even_sections := 2

-- Definitions derived from conditions
def first_wheel_even_prob := first_wheel_even_sections / first_wheel_total_sections
def first_wheel_odd_prob := 1 - first_wheel_even_prob
def second_wheel_even_prob := second_wheel_even_sections / second_wheel_total_sections
def second_wheel_odd_prob := 1 - second_wheel_even_prob

-- Compute the probabilities of getting an even sum
def even_sum_prob := (first_wheel_even_prob * second_wheel_even_prob) +
                     (first_wheel_odd_prob * second_wheel_odd_prob)

-- Assertion that the probability of an even sum is 1/2
theorem even_sum_probability :
  even_sum_prob = 1 / 2 :=
sorry

end even_sum_probability_l252_252409


namespace binom_12_6_l252_252944

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l252_252944


namespace trig_identity_simplified_l252_252923

open Real

theorem trig_identity_simplified :
  (sin (15 * π / 180) + cos (15 * π / 180)) * (sin (15 * π / 180) - cos (15 * π / 180)) = - (sqrt 3 / 2) :=
by
  sorry

end trig_identity_simplified_l252_252923


namespace Ron_spends_15_dollars_l252_252429

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l252_252429


namespace binom_12_6_l252_252932

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l252_252932


namespace total_pencils_l252_252167

def pencils_in_rainbow_box : ℕ := 7
def total_people : ℕ := 8

theorem total_pencils : pencils_in_rainbow_box * total_people = 56 := by
  sorry

end total_pencils_l252_252167


namespace smallest_digit_to_make_divisible_by_9_l252_252031

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l252_252031


namespace sides_of_second_polygon_l252_252127

theorem sides_of_second_polygon (s : ℝ) (n : ℕ) 
  (perimeter1_is_perimeter2 : 38 * (2 * s) = n * s) : 
  n = 76 := by
  sorry

end sides_of_second_polygon_l252_252127


namespace pat_interest_rate_l252_252051

noncomputable def interest_rate (t : ℝ) : ℝ := 70 / t

theorem pat_interest_rate (r : ℝ) (t : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (years : ℝ) : 
  initial_amount * 2^((years / t)) = final_amount ∧ 
  years = 18 ∧ 
  final_amount = 28000 ∧ 
  initial_amount = 7000 →    
  r = interest_rate 9 := 
by
  sorry

end pat_interest_rate_l252_252051


namespace kenya_peanuts_correct_l252_252516

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l252_252516


namespace shorter_leg_of_right_triangle_l252_252811

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252811


namespace proof_problem_l252_252348

theorem proof_problem (a b c : ℤ)
  (h1 : a + b + c = 6)
  (h2 : a - b + c = 4)
  (h3 : c = 3) : 3 * a - 2 * b + c = 7 := by
  sorry

end proof_problem_l252_252348


namespace range_of_a_l252_252972

noncomputable def setA (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (setA a ∩ setB = ∅) ↔ (a > 3 ∨ (-1 / 2 ≤ a ∧ a ≤ 2)) := 
  sorry

end range_of_a_l252_252972


namespace arithmetic_sequence_1005th_term_l252_252707

theorem arithmetic_sequence_1005th_term (p r : ℤ) 
  (h1 : 11 = p + 2 * r)
  (h2 : 11 + 2 * r = 4 * p - r) :
  (5 + 1004 * 6) = 6029 :=
by
  sorry

end arithmetic_sequence_1005th_term_l252_252707


namespace find_number_l252_252108

theorem find_number (n : ℝ) :
  (n + 2 * 1.5)^5 = (1 + 3 * 1.5)^4 → n = 0.72 :=
sorry

end find_number_l252_252108


namespace condition_sufficient_not_necessary_l252_252101

theorem condition_sufficient_not_necessary (x : ℝ) :
  (0 < x ∧ x < 2) → (x < 2) ∧ ¬((x < 2) → (0 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l252_252101


namespace total_pounds_of_food_l252_252652

-- Conditions
def chicken := 16
def hamburgers := chicken / 2
def hot_dogs := hamburgers + 2
def sides := hot_dogs / 2

-- Define the total pounds of food
def total_food := chicken + hamburgers + hot_dogs + sides

-- Theorem statement that corresponds to the problem, showing the final result
theorem total_pounds_of_food : total_food = 39 := 
by
  -- Placeholder for the proof
  sorry

end total_pounds_of_food_l252_252652


namespace problem1_problem2_l252_252578

-- Proof problem (1)
theorem problem1 : real.cbrt 8 + abs (-5) + (-1 : ℝ) ^ 2023 = 6 := by
  sorry

-- Proof problem (2)
theorem problem2 (k b : ℝ) 
  (h1 : 1 = b)
  (h2 : 5 = 2 * k + b) :
  (∀ x, ∃ y, y = k * x + b) → (∀ x, ∃ y, y = 2 * x + 1) :=
by 
  sorry

end problem1_problem2_l252_252578


namespace correct_op_l252_252336

-- Declare variables and conditions
variables {a b : ℝ} {m n : ℤ}
variable (ha : a > 0)
variable (hb : b ≠ 0)

-- Define and state the theorem
theorem correct_op (ha : a > 0) (hb : b ≠ 0) : (b / a)^m = a^(-m) * b^m :=
sorry  -- Proof omitted

end correct_op_l252_252336


namespace right_triangle_shorter_leg_l252_252817

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l252_252817


namespace shift_quadratic_function_left_l252_252501

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the shifted quadratic function
def shifted_function (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem shift_quadratic_function_left :
  ∀ x : ℝ, shifted_function x = original_function (x + 1) := by
  sorry

end shift_quadratic_function_left_l252_252501


namespace min_expression_value_l252_252475

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end min_expression_value_l252_252475


namespace abs_eq_solutions_l252_252965

theorem abs_eq_solutions (x : ℝ) (hx : |x - 5| = 3 * x + 6) :
  x = -11 / 2 ∨ x = -1 / 4 :=
sorry

end abs_eq_solutions_l252_252965


namespace least_tiles_needed_l252_252443

-- Define the conditions
def hallway_length_ft : ℕ := 18
def hallway_width_ft : ℕ := 6
def tile_side_in : ℕ := 6
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Translate conditions
def hallway_length_in := feet_to_inches hallway_length_ft
def hallway_width_in := feet_to_inches hallway_width_ft

-- Define the areas
def hallway_area : ℕ := hallway_length_in * hallway_width_in
def tile_area : ℕ := tile_side_in * tile_side_in

-- State the theorem to be proved
theorem least_tiles_needed :
  hallway_area / tile_area = 432 := 
sorry

end least_tiles_needed_l252_252443


namespace value_of_seventh_observation_l252_252261

-- Given conditions
def sum_of_first_six_observations : ℕ := 90
def new_total_sum : ℕ := 98

-- Problem: prove the value of the seventh observation
theorem value_of_seventh_observation : new_total_sum - sum_of_first_six_observations = 8 :=
by
  sorry

end value_of_seventh_observation_l252_252261


namespace at_least_30_cents_prob_l252_252865

def coin := {penny, nickel, dime, quarter, half_dollar}
def value (c : coin) : ℕ := 
  match c with
  | penny => 1
  | nickel => 5
  | dime => 10
  | quarter => 25
  | half_dollar => 50

def coin_positions : List (coin × Bool) := 
  [(penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, false)]

def count_successful_outcomes : ℕ :=
  List.length (List.filter (λ positions, List.foldl (λ acc (c, h) => if h then acc + value c else acc) 0 positions >= 30) coin_positions)

def total_outcomes : ℕ := 32

def probability_of_success : ℚ :=
  ⟨count_successful_outcomes, total_outcomes⟩

theorem at_least_30_cents_prob : probability_of_success = 3 / 4 :=
by sorry

end at_least_30_cents_prob_l252_252865


namespace inequalities_hold_l252_252966

theorem inequalities_hold (b : ℝ) :
  (b ∈ Set.Ioo (-(1 : ℝ) - Real.sqrt 2 / 4) (0 : ℝ) ∨ b < -(1 : ℝ) - Real.sqrt 2 / 4) →
  (∀ x y : ℝ, 2 * b * Real.cos (2 * (x - y)) + 8 * b^2 * Real.cos (x - y) + 8 * b^2 * (b + 1) + 5 * b < 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 1 > 2 * b * x + 2 * y + b - b^2) :=
by 
  intro h
  sorry

end inequalities_hold_l252_252966


namespace max_page_number_with_given_fives_l252_252535

theorem max_page_number_with_given_fives (plenty_digit_except_five : ℕ → ℕ) 
  (H0 : ∀ d ≠ 5, ∀ n, plenty_digit_except_five d = n)
  (H5 : plenty_digit_except_five 5 = 30) : ∃ (n : ℕ), n = 154 :=
by {
  sorry
}

end max_page_number_with_given_fives_l252_252535


namespace number_of_pairs_l252_252670

def f (n k : ℕ) : ℕ :=
  n + k - Nat.gcd n k

theorem number_of_pairs (N : ℕ) : 
  N = (number of pairs (n k : ℕ) where n ≥ k and f(n, k) = 2018) ↔ N = 874 :=
by
  sorry

end number_of_pairs_l252_252670


namespace expression_equals_4096_l252_252313

noncomputable def calculate_expression : ℕ :=
  ((16^15 / 16^14)^3 * 8^3) / 2^9

theorem expression_equals_4096 : calculate_expression = 4096 :=
by {
  -- proof would go here
  sorry
}

end expression_equals_4096_l252_252313


namespace total_points_l252_252797

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end total_points_l252_252797


namespace sum_of_monomials_is_monomial_l252_252053

variable (a b : ℕ)

theorem sum_of_monomials_is_monomial (m n : ℕ) (h : ∃ k : ℕ, 2 * a^m * b^n + a * b^3 = k * a^1 * b^3) :
  m = 1 ∧ n = 3 :=
sorry

end sum_of_monomials_is_monomial_l252_252053


namespace g_is_odd_function_l252_252067

noncomputable def g (x : ℝ) := 5 / (3 * x^5 - 7 * x)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  unfold g
  sorry

end g_is_odd_function_l252_252067


namespace polygon_sides_eight_l252_252054

theorem polygon_sides_eight (n : ℕ) (h : 180 * (n - 2) = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_eight_l252_252054


namespace common_ratio_is_2_l252_252480

noncomputable def arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1)

theorem common_ratio_is_2 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 > 0)
  (h3 : arithmetic_sequence_common_ratio a q) :
  q = 2 :=
sorry

end common_ratio_is_2_l252_252480


namespace count_T_diff_S_l252_252362

-- Define a function to check if a digit is in a given number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ i, i < 3 ∧ (n / 10^i) % 10 = d

-- Define a function to check if a three-digit number is valid
def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the set T' of three digit numbers that do not contain a 6
def T_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6

-- Define the set S' of three digit numbers that neither contain a 2 nor a 6
def S_prime : {n // is_valid_three_digit n} → Prop :=
  λ n, ¬ contains_digit n 6 ∧ ¬ contains_digit n 2

-- Define the set of numbers we are interested in, has 2 but not 6
def T_diff_S : {n // is_valid_three_digit n} → Prop := 
  λ n, contains_digit n 2 ∧ ¬ contains_digit n 6

-- Statement to prove
theorem count_T_diff_S : ∃ n, n = 200 ∧ (∀ (x : {n // is_valid_three_digit n}), T_diff_S x) :=
sorry

end count_T_diff_S_l252_252362


namespace min_value_of_f_solve_inequality_l252_252484

noncomputable def f (x : ℝ) : ℝ := abs (x - 5/2) + abs (x - 1/2)

theorem min_value_of_f : (∀ x : ℝ, f x ≥ 2) ∧ (∃ x : ℝ, f x = 2) := by
  sorry

theorem solve_inequality (x : ℝ) : (f x ≤ x + 4) ↔ (-1/3 ≤ x ∧ x ≤ 7) := by
  sorry

end min_value_of_f_solve_inequality_l252_252484


namespace first_number_is_38_l252_252273

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end first_number_is_38_l252_252273


namespace intersection_points_l252_252901

theorem intersection_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2) ↔ (y = x^2 - 2 * a)) ↔ (0 < a ∧ a < 1) :=
sorry

end intersection_points_l252_252901


namespace problem1_l252_252575

theorem problem1 : (Nat.cbrt 8 : Int) + abs (-5 : Int) + (-1 : Int) ^ 2023 = 6 := by
  -- Proof goes here
  sorry

end problem1_l252_252575


namespace subtraction_makes_divisible_l252_252294

theorem subtraction_makes_divisible :
  ∃ n : Nat, 9671 - n % 2 = 0 ∧ n = 1 :=
by
  sorry

end subtraction_makes_divisible_l252_252294


namespace basketball_match_scores_l252_252057

theorem basketball_match_scores :
  ∃ (a r b d : ℝ), (a = b) ∧ (a * (1 + r + r^2 + r^3) < 120) ∧
  (4 * b + 6 * d < 120) ∧ ((a * (1 + r + r^2 + r^3) - (4 * b + 6 * d)) = 3) ∧
  a + b + (a * r + (b + d)) = 35.5 :=
sorry

end basketball_match_scores_l252_252057


namespace polygon_with_120_degree_interior_angle_has_6_sides_l252_252198

theorem polygon_with_120_degree_interior_angle_has_6_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (sum_interior_angles : ℕ) = (n-2) * 180 / n ∧ (each_angle : ℕ) = 120) : n = 6 :=
by
  sorry

end polygon_with_120_degree_interior_angle_has_6_sides_l252_252198


namespace ethanol_percentage_in_fuel_A_l252_252153

variable {capacity_A fuel_A : ℝ}
variable (ethanol_A ethanol_B total_ethanol : ℝ)
variable (E : ℝ)

def fuelTank (capacity_A fuel_A ethanol_A ethanol_B total_ethanol : ℝ) (E : ℝ) : Prop := 
  (ethanol_A / fuel_A = E) ∧
  (capacity_A - fuel_A = 200 - 99.99999999999999) ∧
  (ethanol_B = 0.16 * (200 - 99.99999999999999)) ∧
  (total_ethanol = ethanol_A + ethanol_B) ∧
  (total_ethanol = 28)

theorem ethanol_percentage_in_fuel_A : 
  ∃ E, fuelTank 99.99999999999999 99.99999999999999 ethanol_A ethanol_B 28 E ∧ E = 0.12 := 
sorry

end ethanol_percentage_in_fuel_A_l252_252153


namespace tan_2x_abs_properties_l252_252104

open Real

theorem tan_2x_abs_properties :
  (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (-x))|) ∧ (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (x + π / 2))|) :=
by
  sorry

end tan_2x_abs_properties_l252_252104


namespace water_level_decrease_l252_252370

theorem water_level_decrease (increase_notation : ℝ) (h : increase_notation = 2) :
  -increase_notation = -2 :=
by
  sorry

end water_level_decrease_l252_252370


namespace probability_red_or_white_ball_l252_252907

theorem probability_red_or_white_ball :
  let red_balls := 3
  let yellow_balls := 2
  let white_balls := 1
  let total_balls := red_balls + yellow_balls + white_balls
  let favorable_outcomes := red_balls + white_balls
  (favorable_outcomes / total_balls : ℚ) = 2 / 3 := by
  sorry

end probability_red_or_white_ball_l252_252907


namespace sandy_bought_6_books_l252_252087

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_bought_6_books_l252_252087


namespace probability_of_at_least_30_cents_l252_252869

def coin := fin 5

def value (c : coin) : ℤ :=
match c with
| 0 => 1   -- penny
| 1 => 5   -- nickel
| 2 => 10  -- dime
| 3 => 25  -- quarter
| 4 => 50  -- half-dollar
| _ => 0

def coin_flip : coin -> bool := λ c => true -- Placeholder for whether heads or tails

def total_value (flips : coin -> bool) : ℤ :=
  finset.univ.sum (λ c, if flips c then value c else 0)

noncomputable def probability_at_least_30_cents : ℚ :=
  let coin_flips := (finset.pi finset.univ (λ _, finset.univ : finset (coin -> bool))).val in
  let successful_flips := coin_flips.filter (λ flips, total_value flips >= 30) in
  successful_flips.card / coin_flips.card

theorem probability_of_at_least_30_cents :
  probability_at_least_30_cents = 9 / 16 :=
by
  sorry

end probability_of_at_least_30_cents_l252_252869


namespace bottles_left_l252_252115

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l252_252115


namespace find_a3_plus_a5_l252_252665

variable {a : ℕ → ℝ}

-- Condition 1: The sequence {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition 2: All terms in the sequence are negative
def all_negative (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < 0

-- Condition 3: The given equation
def given_equation (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25

-- The problem statement
theorem find_a3_plus_a5 (h_geo : is_geometric_sequence a) (h_neg : all_negative a) (h_eq : given_equation a) :
  a 3 + a 5 = -5 :=
sorry

end find_a3_plus_a5_l252_252665


namespace smallest_value_of_n_l252_252457

/-- Given that Casper has exactly enough money to buy either 
  18 pieces of red candy, 20 pieces of green candy, 
  25 pieces of blue candy, or n pieces of purple candy where 
  each purple candy costs 30 cents, prove that the smallest 
  possible value of n is 30.
-/
theorem smallest_value_of_n
  (r g b n : ℕ)
  (h : 18 * r = 20 * g ∧ 20 * g = 25 * b ∧ 25 * b = 30 * n) : 
  n = 30 :=
sorry

end smallest_value_of_n_l252_252457


namespace chameleons_color_change_l252_252220

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l252_252220


namespace chameleons_color_change_l252_252210

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l252_252210


namespace number_of_days_worked_l252_252695

-- Definitions based on the given conditions and question
def total_hours_worked : ℕ := 15
def hours_worked_each_day : ℕ := 3

-- The statement we need to prove:
theorem number_of_days_worked : 
  (total_hours_worked / hours_worked_each_day) = 5 :=
by
  sorry

end number_of_days_worked_l252_252695


namespace percentage_both_colors_l252_252418

theorem percentage_both_colors
  (total_flags : ℕ)
  (even_flags : total_flags % 2 = 0)
  (C : ℕ)
  (total_flags_eq : total_flags = 2 * C)
  (blue_percent : ℕ)
  (blue_percent_eq : blue_percent = 60)
  (red_percent : ℕ)
  (red_percent_eq : red_percent = 65) :
  ∃ both_colors_percent : ℕ, both_colors_percent = 25 :=
by
  sorry

end percentage_both_colors_l252_252418


namespace micrometer_conversion_l252_252850

theorem micrometer_conversion :
  (0.01 * (1 * 10 ^ (-6))) = (1 * 10 ^ (-8)) :=
by 
  -- sorry is used to skip the actual proof but ensure the theorem is recognized
  sorry

end micrometer_conversion_l252_252850


namespace Euclid1976_PartA_Problem8_l252_252093

theorem  Euclid1976_PartA_Problem8 (a b c m n : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h4 : Polynomial.eval (-a) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0)
  (h5 : Polynomial.eval (-b) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0) :
  n = -1 :=
sorry

end Euclid1976_PartA_Problem8_l252_252093


namespace perfect_cube_factors_count_l252_252109

-- Define the given prime factorization
def prime_factorization_8820 : Prop :=
  ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧
  (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) = 8820

-- Prove the statement about positive integer factors that are perfect cubes
theorem perfect_cube_factors_count : prime_factorization_8820 → (∃ n : ℕ, n = 1) :=
by
  sorry

end perfect_cube_factors_count_l252_252109


namespace cube_paint_problem_l252_252431

theorem cube_paint_problem : 
  ∀ (n : ℕ),
  n = 6 →
  (∃ k : ℕ, 216 = k^3 ∧ k = n) →
  ∀ (faces inner_faces total_cubelets : ℕ),
  faces = 6 →
  inner_faces = 4 →
  total_cubelets = faces * (inner_faces * inner_faces) →
  total_cubelets = 96 :=
by 
  intros n hn hc faces hfaces inner_faces hinner_faces total_cubelets htotal_cubelets
  sorry

end cube_paint_problem_l252_252431


namespace problem1_problem2_l252_252580

-- Statement for Problem 1
theorem problem1 : real.cbrt 8 + abs (-5) + (-1 : ℤ) ^ 2023 = 6 := 
by sorry

-- Statement for Problem 2
theorem problem2 (k b : ℝ) (h1 : (0 : ℝ) * k + b = 1)
  (h2 : (2 : ℝ) * k + b = 5) : k = 2 ∧ b = 1 ∧ ∀ x, (k * x + b) = 2 * x + 1 :=
by sorry

end problem1_problem2_l252_252580


namespace gcd_of_45_75_90_l252_252725

def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_of_45_75_90 : gcd_three_numbers 45 75 90 = 15 := by
  sorry

end gcd_of_45_75_90_l252_252725


namespace four_digit_multiples_of_7_count_l252_252355

theorem four_digit_multiples_of_7_count : 
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  num_multiples = 1286 := 
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_multiple := (1000 / 7 + 1) * 7
  let largest_multiple := (9999 / 7) * 7
  let num_multiples := (largest_multiple / 7 - smallest_multiple / 7 + 1)
  have h1: smallest_multiple = 1001, by sorry
  have h2: largest_multiple = 9996, by sorry
  have h3: num_multiples = 1286, by sorry
  exact h3

end four_digit_multiples_of_7_count_l252_252355


namespace trees_left_after_typhoon_l252_252302

-- Define the initial count of trees and the number of trees that died
def initial_trees := 150
def trees_died := 24

-- Define the expected number of trees left
def expected_trees_left := 126

-- The statement to be proven: after trees died, the number of trees left is as expected
theorem trees_left_after_typhoon : (initial_trees - trees_died) = expected_trees_left := by
  sorry

end trees_left_after_typhoon_l252_252302


namespace units_digit_17_pow_39_l252_252897

theorem units_digit_17_pow_39 : 
  ∃ d : ℕ, d < 10 ∧ (17^39 % 10 = d) ∧ d = 3 :=
by
  sorry

end units_digit_17_pow_39_l252_252897


namespace shorter_leg_of_right_triangle_l252_252810

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252810


namespace find_coordinates_M_l252_252337

open Real

theorem find_coordinates_M (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℝ) :
  ∃ (xM yM zM : ℝ), 
  xM = (x1 + x2 + x3 + x4) / 4 ∧
  yM = (y1 + y2 + y3 + y4) / 4 ∧
  zM = (z1 + z2 + z3 + z4) / 4 ∧
  (x1 - xM) + (x2 - xM) + (x3 - xM) + (x4 - xM) = 0 ∧
  (y1 - yM) + (y2 - yM) + (y3 - yM) + (y4 - yM) = 0 ∧
  (z1 - zM) + (z2 - zM) + (z3 - zM) + (z4 - zM) = 0 := by
  sorry

end find_coordinates_M_l252_252337


namespace find_a_l252_252324

theorem find_a (a r s : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 24) (h3 : s^2 = 9) : a = 16 :=
sorry

end find_a_l252_252324


namespace ratio_of_areas_l252_252572

theorem ratio_of_areas (s : ℝ) (h1 : s > 0) : 
  let small_square_area := s^2
  let total_small_squares_area := 4 * s^2
  let large_square_side_length := 4 * s
  let large_square_area := (4 * s)^2
  total_small_squares_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l252_252572


namespace triangle_shape_and_maximum_tan_B_minus_C_l252_252066

open Real

variable (A B C : ℝ)
variable (sin cos tan : ℝ → ℝ)

-- Given conditions
axiom sin2A_plus_3sin2C_equals_3sin2B : sin A ^ 2 + 3 * sin C ^ 2 = 3 * sin B ^ 2
axiom sinB_cosC_equals_2div3 : sin B * cos C = 2 / 3

-- Prove
theorem triangle_shape_and_maximum_tan_B_minus_C :
  (A = π / 2) ∧ (∀ x y : ℝ, (x = B - C) → tan x ≤ sqrt 2 / 4) :=
by sorry

end triangle_shape_and_maximum_tan_B_minus_C_l252_252066


namespace correct_option_l252_252905

-- Definitions based on conditions
def sentence_structure : String := "He’s never interested in what ______ is doing."

def option_A : String := "no one else"
def option_B : String := "anyone else"
def option_C : String := "someone else"
def option_D : String := "nobody else"

-- The proof statement
theorem correct_option : option_B = "anyone else" := by
  sorry

end correct_option_l252_252905


namespace find_x_y_sum_l252_252009

def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem find_x_y_sum (n x y : ℕ) (hn : n = 450) (hx : x > 0) (hy : y > 0)
  (hxsq : is_perfect_square (n * x))
  (hycube : is_perfect_cube (n * y)) :
  x + y = 62 :=
  sorry

end find_x_y_sum_l252_252009


namespace scientific_notation_of_11_million_l252_252682

theorem scientific_notation_of_11_million :
  (11_000_000 : ℝ) = 1.1 * (10 : ℝ) ^ 7 :=
by
  sorry

end scientific_notation_of_11_million_l252_252682


namespace calculate_expression_l252_252582

theorem calculate_expression : real.cbrt 8 + abs (-5) + (-1 : ℤ)^2023 = 6 := 
by
  have h1 : real.cbrt 8 = 2 := real.cbrt_of_cube 8
  have h2 : abs (-5) = 5 := abs_of_neg (lt_add_of_le (le_refl 0) (show (0 : ℤ) < 5 by norm_num))
  have h3 : (-1 : ℤ)^2023 = -1 := pow_odd_neg 1 (by norm_num)
  rw [h1, h2, h3]
  norm_num
  sorry -- Omitted detailed proof steps here

end calculate_expression_l252_252582


namespace simplify_and_evaluate_expression_l252_252539

theorem simplify_and_evaluate_expression (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -2) (hx3 : x ≠ 2) :
  ( ( (x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = x - 2 ) ∧ 
  ( (x = 1) → ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = -1 ) :=
by
  sorry

end simplify_and_evaluate_expression_l252_252539


namespace find_y_l252_252289

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : x / y = 86 ∧ ((x % y : ℚ) / y = 0.12)) : y = 75 :=
by
  sorry

end find_y_l252_252289


namespace product_sum_l252_252978

theorem product_sum (y x z: ℕ) 
  (h1: 2014 + y = 2015 + x) 
  (h2: 2015 + x = 2016 + z) 
  (h3: y * x * z = 504): 
  y * x + x * z = 128 := 
by 
  sorry

end product_sum_l252_252978


namespace largest_and_smallest_A_l252_252613

def is_coprime_with_12 (n : ℕ) : Prop := Nat.coprime n 12

noncomputable def last_digit_to_first (B : ℕ) : ℕ :=
let b := B % 10 in
10^7 * b + (B - b) / 10

def is_valid_A (A B : ℕ) : Prop :=
A = last_digit_to_first B ∧ is_coprime_with_12 B ∧ B > 44444444

theorem largest_and_smallest_A (Amin Amax Bmin Bmax : ℕ) :
  Amin = 14444446 ∧ Amax = 99999998 ∧ is_valid_A Amin Bmin ∧ is_valid_A Amax Bmax := sorry

end largest_and_smallest_A_l252_252613


namespace find_a9_l252_252245

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {d a₁ : ℤ}

-- Conditions
def arithmetic_sequence := ∀ n : ℕ, a_n n = a₁ + n * d
def sum_first_n_terms := ∀ n : ℕ, S n = (n * (2 * a₁ + (n - 1) * d)) / 2

-- Specific Conditions for the problem
axiom condition1 : S 8 = 4 * a₁
axiom condition2 : a_n 6 = -2 -- Note that a_n is 0-indexed here.

theorem find_a9 : a_n 8 = 2 :=
by
  sorry

end find_a9_l252_252245


namespace no_right_triangle_with_sqrt_2016_side_l252_252632

theorem no_right_triangle_with_sqrt_2016_side :
  ¬ ∃ (a b : ℤ), (a * a + b * b = 2016) ∨ (a * a + 2016 = b * b) :=
by
  sorry

end no_right_triangle_with_sqrt_2016_side_l252_252632


namespace total_amount_shared_l252_252303

theorem total_amount_shared (A B C : ℕ) (h1 : 3 * B = 5 * A) (h2 : B = 25) (h3 : 5 * C = 8 * B) : A + B + C = 80 := by
  sorry

end total_amount_shared_l252_252303


namespace roots_opposite_eq_minus_one_l252_252203

theorem roots_opposite_eq_minus_one (k : ℝ) 
  (h_real_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 0 ∧ x₁ * x₂ = k + 1) :
  k = -1 :=
by
  sorry

end roots_opposite_eq_minus_one_l252_252203


namespace tangent_line_length_l252_252236

noncomputable def curve_C (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def problem_conditions : Prop :=
  curve_C 0 = 4 ∧ cartesian 4 0 = (4, 0)

theorem tangent_line_length :
  problem_conditions → 
  ∃ l : ℝ, l = 2 :=
by
  sorry

end tangent_line_length_l252_252236


namespace term_15_of_sequence_l252_252633

theorem term_15_of_sequence : 
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ a 2 = 7 ∧ (∀ n, a (n + 1) = 21 / a n) ∧ a 15 = 3 :=
sorry

end term_15_of_sequence_l252_252633


namespace shorter_leg_of_right_triangle_l252_252802

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252802


namespace problem_l252_252780

def seq (a : ℕ → ℝ) := a 0 = 1 / 2 ∧ ∀ n > 0, a n = a (n - 1) + (1 / n^2) * (a (n - 1))^2

theorem problem (a : ℕ → ℝ) (n : ℕ) (h_seq : seq a) (h_n_pos : n > 0) :
  (1 / a (n - 1) - 1 / a n < 1 / n^2) ∧
  (∀ n > 0, a n < n) ∧
  (∀ n > 0, 1 / a n < 5 / 6 + 1 / (n + 1)) :=
by
  sorry

end problem_l252_252780


namespace dot_product_computation_l252_252479

open Real

variables (a b : ℝ) (θ : ℝ)

noncomputable def dot_product (u v : ℝ) : ℝ :=
  u * v * cos θ

noncomputable def magnitude (v : ℝ) : ℝ :=
  abs v

theorem dot_product_computation (a b : ℝ) (h1 : θ = 120) (h2 : magnitude a = 4) (h3 : magnitude b = 4) :
  dot_product b (3 * a + b) = -8 :=
by
  sorry

end dot_product_computation_l252_252479


namespace no_such_function_exists_l252_252856

theorem no_such_function_exists (f : ℤ → ℤ) (h : ∀ m n : ℤ, f (m + f n) = f m - n) : false :=
sorry

end no_such_function_exists_l252_252856


namespace calculate_expression_l252_252581

theorem calculate_expression : real.cbrt 8 + abs (-5) + (-1 : ℤ)^2023 = 6 := 
by
  have h1 : real.cbrt 8 = 2 := real.cbrt_of_cube 8
  have h2 : abs (-5) = 5 := abs_of_neg (lt_add_of_le (le_refl 0) (show (0 : ℤ) < 5 by norm_num))
  have h3 : (-1 : ℤ)^2023 = -1 := pow_odd_neg 1 (by norm_num)
  rw [h1, h2, h3]
  norm_num
  sorry -- Omitted detailed proof steps here

end calculate_expression_l252_252581


namespace shorter_leg_of_right_triangle_l252_252824

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l252_252824


namespace mark_more_hours_than_kate_l252_252396

theorem mark_more_hours_than_kate {K : ℕ} (h1 : K + 2 * K + 6 * K = 117) :
  6 * K - K = 65 :=
by
  sorry

end mark_more_hours_than_kate_l252_252396


namespace student_B_more_consistent_l252_252156

noncomputable def standard_deviation_A := 5.09
noncomputable def standard_deviation_B := 3.72
def games_played := 7
noncomputable def average_score_A := 16
noncomputable def average_score_B := 16

theorem student_B_more_consistent :
  standard_deviation_B < standard_deviation_A :=
sorry

end student_B_more_consistent_l252_252156


namespace rons_chocolate_cost_l252_252426

theorem rons_chocolate_cost :
  let cost_per_bar := 1.5
  let sections_per_bar := 3
  let scouts := 15
  let smores_per_scout := 2
  let total_smores := scouts * smores_per_scout
  let bars_needed := total_smores / sections_per_bar in
  bars_needed * cost_per_bar = 15.0 := by
  sorry

end rons_chocolate_cost_l252_252426


namespace Ron_spends_15_dollars_l252_252430

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l252_252430


namespace sum_of_constants_eq_zero_l252_252627

theorem sum_of_constants_eq_zero (A B C D E : ℝ) :
  (∀ (x : ℝ), (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
              A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 :=
by
  sorry

end sum_of_constants_eq_zero_l252_252627


namespace average_salary_correct_l252_252880

/-- The salaries of A, B, C, D, and E. -/
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

/-- The number of people. -/
def number_of_people : ℕ := 5

/-- The total salary is the sum of the salaries. -/
def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

/-- The average salary is the total salary divided by the number of people. -/
def average_salary : ℕ := total_salary / number_of_people

/-- The average salary of A, B, C, D, and E is Rs. 8000. -/
theorem average_salary_correct : average_salary = 8000 := by
  sorry

end average_salary_correct_l252_252880


namespace binomial_12_6_eq_924_l252_252935

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l252_252935


namespace percent_alcohol_new_solution_l252_252744

theorem percent_alcohol_new_solution :
  let original_volume := 40
  let original_percent_alcohol := 5
  let added_alcohol := 2.5
  let added_water := 7.5
  let original_alcohol := original_volume * (original_percent_alcohol / 100)
  let total_alcohol := original_alcohol + added_alcohol
  let new_total_volume := original_volume + added_alcohol + added_water
  (total_alcohol / new_total_volume) * 100 = 9 :=
by
  sorry

end percent_alcohol_new_solution_l252_252744


namespace product_586645_9999_l252_252286

theorem product_586645_9999 :
  586645 * 9999 = 5865885355 :=
by
  sorry

end product_586645_9999_l252_252286


namespace sequence_last_number_is_one_l252_252123

theorem sequence_last_number_is_one :
  ∃ (a : ℕ → ℤ), (a 1 = 1) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → a (n + 1) = a n + a (n + 2)) ∧ (a 1999 = 1) := sorry

end sequence_last_number_is_one_l252_252123


namespace carla_sheep_l252_252762

theorem carla_sheep (T : ℝ) (pen_sheep wilderness_sheep : ℝ) 
(h1: 0.90 * T = 81) (h2: pen_sheep = 81) 
(h3: wilderness_sheep = 0.10 * T) : wilderness_sheep = 9 :=
sorry

end carla_sheep_l252_252762


namespace half_ears_kernels_l252_252046

theorem half_ears_kernels (stalks ears_per_stalk total_kernels : ℕ) (X : ℕ)
  (half_ears : ℕ := stalks * ears_per_stalk / 2)
  (total_ears : ℕ := stalks * ears_per_stalk)
  (condition_e1 : stalks = 108)
  (condition_e2 : ears_per_stalk = 4)
  (condition_e3 : total_kernels = 237600)
  (condition_kernel_sum : total_kernels = 216 * X + 216 * (X + 100)) :
  X = 500 := by
  have condition_eq : 432 * X + 21600 = 237600 := by sorry
  have X_value : X = 216000 / 432 := by sorry
  have X_result : X = 500 := by sorry
  exact X_result

end half_ears_kernels_l252_252046


namespace factor_expression_l252_252962

theorem factor_expression (a b c : ℝ) : 
  ( (a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4 ) / 
  ( (a - b)^4 + (b - c)^4 + (c - a)^4 ) = 1 := 
by sorry

end factor_expression_l252_252962


namespace pasha_game_solvable_l252_252906

def pasha_game : Prop :=
∃ (a : Fin 2017 → ℕ), 
  (∀ i, a i > 0) ∧
  (∃ (moves : ℕ), moves = 43 ∧
   (∀ (box_contents : Fin 2017 → ℕ), 
    (∀ j, box_contents j = 0) →
    (∃ (equal_count : ℕ),
      (∀ j, box_contents j = equal_count)
      ∧
      (∀ m < 43,
        ∃ j, box_contents j ≠ equal_count))))

theorem pasha_game_solvable : pasha_game :=
by
  sorry

end pasha_game_solvable_l252_252906


namespace product_of_possible_values_of_x_l252_252364

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end product_of_possible_values_of_x_l252_252364


namespace percentage_of_acid_in_original_mixture_l252_252126

theorem percentage_of_acid_in_original_mixture
  (a w : ℚ)
  (h1 : a / (a + w + 2) = 18 / 100)
  (h2 : (a + 2) / (a + w + 4) = 30 / 100) :
  (a / (a + w)) * 100 = 29 := 
sorry

end percentage_of_acid_in_original_mixture_l252_252126


namespace peter_total_food_l252_252648

-- Given conditions
variables (chicken hamburgers hot_dogs sides total_food : ℕ)
  (h1 : chicken = 16)
  (h2 : hamburgers = chicken / 2)
  (h3 : hot_dogs = hamburgers + 2)
  (h4 : sides = hot_dogs / 2)
  (total_food : ℕ := chicken + hamburgers + hot_dogs + sides)

theorem peter_total_food (h1 : chicken = 16)
                          (h2 : hamburgers = chicken / 2)
                          (h3 : hot_dogs = hamburgers + 2)
                          (h4 : sides = hot_dogs / 2) :
                          total_food = 39 :=
by sorry

end peter_total_food_l252_252648


namespace sum_divisible_by_3_probability_l252_252466

-- The 12 prime numbers
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Function to calculate the residue of a number modulo 3
def residue_mod_3 (n : ℕ) : ℕ := n % 3

-- List of residues of the first 12 primes
def residues : List ℕ := primes.map residue_mod_3

-- Assume residues
def prime_residues := [2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1]

-- Number of valid combinations where the sum of residues modulo 3 is zero
noncomputable def valid_combinations : ℕ := 150

-- Total combinations to choose 5 out of 12 prime numbers
noncomputable def total_combinations : ℕ := Nat.choose 12 5

-- Final probability
noncomputable def probability := Rat.mk valid_combinations total_combinations

theorem sum_divisible_by_3_probability :
  probability = Rat.mk 25 132 :=
by sorry

end sum_divisible_by_3_probability_l252_252466


namespace kenya_peanuts_eq_133_l252_252521

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l252_252521


namespace problem1_l252_252583

theorem problem1 : real.cbrt 8 + abs (-5) + (-1:ℝ)^2023 = 6 := 
by 
  sorry

end problem1_l252_252583


namespace opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l252_252879

theorem opposite_number_of_neg_two (a : Int) (h : a = -2) :
  -a = 2 := by
  sorry

theorem reciprocal_of_three (x y : Real) (hx : x = 3) (hy : y = 1 / 3) : 
  x * y = 1 := by
  sorry

theorem abs_val_three_eq (x : Real) (hx : abs x = 3) :
  x = -3 ∨ x = 3 := by
  sorry

end opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l252_252879


namespace polygon_sides_eight_l252_252055

theorem polygon_sides_eight (n : ℕ) 
  (h₀ : ∑ (exterior_angles : 360)) 
  (h₁ : ∑ (interior_angles = 180 * (n - 2)) = 3 * ∑ (exterior_angles)) 
  : n = 8 := 
by 
  sorry

end polygon_sides_eight_l252_252055


namespace num_people_end_race_l252_252886

-- Define the conditions
def num_cars : ℕ := 20
def initial_passengers_per_car : ℕ := 2
def drivers_per_car : ℕ := 1
def additional_passengers_per_car : ℕ := 1

-- Define the total number of people in a car at the start
def total_people_per_car_initial := initial_passengers_per_car + drivers_per_car

-- Define the total number of people in a car after halfway point
def total_people_per_car_end := total_people_per_car_initial + additional_passengers_per_car

-- Define the total number of people in all cars at the end
def total_people_end := num_cars * total_people_per_car_end

-- Theorem statement
theorem num_people_end_race : total_people_end = 80 := by
  sorry

end num_people_end_race_l252_252886


namespace chameleon_color_change_l252_252215

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l252_252215


namespace flower_combinations_count_l252_252747

/-- Prove that there are exactly 3 combinations of tulips and sunflowers that sum up to $60,
    where tulips cost $4 each and sunflowers cost $3 each, and the number of sunflowers is greater than the number 
    of tulips. -/
theorem flower_combinations_count :
  ∃ n : ℕ, n = 3 ∧
    ∃ t s : ℕ, 4 * t + 3 * s = 60 ∧ s > t :=
by {
  sorry
}

end flower_combinations_count_l252_252747


namespace base8_minus_base7_base10_eq_l252_252637

-- Definitions of the two numbers in their respective bases
def n1_base8 : ℕ := 305
def n2_base7 : ℕ := 165

-- Conversion of these numbers to base 10
def n1_base10 : ℕ := 3 * 8^2 + 0 * 8^1 + 5 * 8^0
def n2_base10 : ℕ := 1 * 7^2 + 6 * 7^1 + 5 * 7^0

-- Statement of the theorem to be proven
theorem base8_minus_base7_base10_eq :
  (n1_base10 - n2_base10 = 101) :=
  by
    -- The proof would go here
    sorry

end base8_minus_base7_base10_eq_l252_252637


namespace music_class_uncool_parents_l252_252121

theorem music_class_uncool_parents:
  ∀ (total students coolDads coolMoms bothCool : ℕ),
  total = 40 →
  coolDads = 25 →
  coolMoms = 19 →
  bothCool = 8 →
  (total - (bothCool + (coolDads - bothCool) + (coolMoms - bothCool))) = 4 :=
by
  intros total coolDads coolMoms bothCool h_total h_dads h_moms h_both
  sorry

end music_class_uncool_parents_l252_252121


namespace range_of_a_l252_252993

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l252_252993


namespace least_integer_value_of_x_l252_252462

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end least_integer_value_of_x_l252_252462


namespace find_m_l252_252791

theorem find_m (x y m : ℝ) (hx : x = 1) (hy : y = 2) (h : m * x + 2 * y = 6) : m = 2 :=
by sorry

end find_m_l252_252791


namespace sum_last_two_digits_of_x2012_l252_252390

def sequence_defined (x : ℕ → ℕ) : Prop :=
  (x 1 = 5 ∨ x 1 = 7) ∧ ∀ k ≥ 1, (x (k+1) = 5^(x k) ∨ x (k+1) = 7^(x k))

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def possible_values : List ℕ :=
  [25, 7, 43]

theorem sum_last_two_digits_of_x2012 {x : ℕ → ℕ} (h : sequence_defined x) :
  List.sum (List.map last_two_digits [25, 7, 43]) = 75 :=
  by
    sorry

end sum_last_two_digits_of_x2012_l252_252390


namespace find_first_number_l252_252270

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end find_first_number_l252_252270


namespace kenya_peanuts_correct_l252_252515

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l252_252515


namespace log_inequality_l252_252182

noncomputable def log3_2 : ℝ := Real.log 2 / Real.log 3
noncomputable def log2_3 : ℝ := Real.log 3 / Real.log 2
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem log_inequality :
  let a := log3_2;
  let b := log2_3;
  let c := log2_5;
  a < b ∧ b < c :=
  by
  sorry

end log_inequality_l252_252182


namespace right_triangle_exists_l252_252902

theorem right_triangle_exists :
  (3^2 + 4^2 = 5^2) ∧ ¬(2^2 + 3^2 = 4^2) ∧ ¬(4^2 + 6^2 = 7^2) ∧ ¬(5^2 + 11^2 = 12^2) :=
by
  sorry

end right_triangle_exists_l252_252902


namespace annalise_spending_l252_252447

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end annalise_spending_l252_252447


namespace rectangular_cube_length_l252_252753

theorem rectangular_cube_length (L : ℝ) (h1 : 2 * (L * 2) + 2 * (L * 0.5) + 2 * (2 * 0.5) = 24) : L = 4.6 := 
by {
  sorry
}

end rectangular_cube_length_l252_252753


namespace A_gt_B_and_C_lt_A_l252_252129

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end A_gt_B_and_C_lt_A_l252_252129


namespace max_y_difference_intersection_l252_252774

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference_intersection :
  let x1 := 1
  let y1 := g x1
  let x2 := -1
  let y2 := g x2
  y1 - y2 = 2 :=
by
  sorry

end max_y_difference_intersection_l252_252774


namespace smallest_non_multiple_of_5_abundant_l252_252918

def proper_divisors (n : ℕ) : List ℕ := List.filter (fun d => d ∣ n ∧ d < n) (List.range (n + 1))

def is_abundant (n : ℕ) : Prop := (proper_divisors n).sum > n

def is_not_multiple_of_5 (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_non_multiple_of_5_abundant : ∃ n, is_abundant n ∧ is_not_multiple_of_5 n ∧ 
  ∀ m, is_abundant m ∧ is_not_multiple_of_5 m → n ≤ m :=
  sorry

end smallest_non_multiple_of_5_abundant_l252_252918


namespace tracy_initial_candies_l252_252283

theorem tracy_initial_candies (x y : ℕ) (h₁ : x = 108) (h₂ : 2 ≤ y ∧ y ≤ 6) : 
  let remaining_after_eating := (3 / 4) * x 
  let remaining_after_giving := (2 / 3) * remaining_after_eating
  let remaining_after_mom := remaining_after_giving - 40
  remaining_after_mom - y = 10 :=
by 
  sorry

end tracy_initial_candies_l252_252283


namespace carl_olivia_cookie_difference_l252_252768

-- Defining the various conditions
def Carl_cookies : ℕ := 7
def Olivia_cookies : ℕ := 2

-- Stating the theorem we need to prove
theorem carl_olivia_cookie_difference : Carl_cookies - Olivia_cookies = 5 :=
by sorry

end carl_olivia_cookie_difference_l252_252768


namespace television_hours_watched_l252_252758

theorem television_hours_watched (minutes_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)
  (h1 : minutes_per_day = 45) (h2 : days_per_week = 4) (h3 : weeks = 2):
  (minutes_per_day * days_per_week / 60) * weeks = 6 :=
by
  sorry

end television_hours_watched_l252_252758


namespace binomial_12_6_eq_924_l252_252954

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l252_252954


namespace part1_part2_l252_252389

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1 : {x : ℝ | f x ≤ 4} = {x : ℝ | -5 / 3 ≤ x ∧ x ≤ 1} :=
by
  sorry

theorem part2 {a : ℝ} :
  ({x : ℝ | f x ≤ 4} ⊆ {x : ℝ | |x + 3| + |x + a| < x + 6}) ↔ (-4 / 3 < a ∧ a < 2) :=
by
  sorry

end part1_part2_l252_252389


namespace sum_of_first_100_positive_odd_integers_is_correct_l252_252679

def sum_first_100_positive_odd_integers : ℕ :=
  10000

theorem sum_of_first_100_positive_odd_integers_is_correct :
  sum_first_100_positive_odd_integers = 10000 :=
by
  sorry

end sum_of_first_100_positive_odd_integers_is_correct_l252_252679


namespace david_cups_consumed_l252_252082

noncomputable def cups_of_water (time_in_minutes : ℕ) : ℝ :=
  time_in_minutes / 20

theorem david_cups_consumed : cups_of_water 225 = 11.25 := by
  sorry

end david_cups_consumed_l252_252082


namespace store_loss_l252_252434

theorem store_loss (x y : ℝ) (hx : x + 0.25 * x = 135) (hy : y - 0.25 * y = 135) : 
  (135 * 2) - (x + y) = -18 := 
by
  sorry

end store_loss_l252_252434


namespace password_probability_l252_252921

def is_prime_single_digit : Fin 10 → Prop
| 2 | 3 | 5 | 7 => true
| _ => false

def is_vowel : Char → Prop
| 'A' | 'E' | 'I' | 'O' | 'U' => true
| _ => false

def is_positive_even_single_digit : Fin 9 → Prop
| 2 | 4 | 6 | 8 => true
| _ => false

def prime_probability : ℚ := 4 / 10
def vowel_probability : ℚ := 5 / 26
def even_pos_digit_probability : ℚ := 4 / 9

theorem password_probability :
  prime_probability * vowel_probability * even_pos_digit_probability = 8 / 117 := by
  sorry

end password_probability_l252_252921


namespace problem_inequality_l252_252669

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem problem_inequality (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2) :
  (f x2 - f x1) / (x2 - x1) < (1 + Real.log ((x1 + x2) / 2)) :=
sorry

end problem_inequality_l252_252669


namespace chameleons_color_change_l252_252213

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l252_252213


namespace proof_problem_l252_252246

theorem proof_problem (a1 a2 a3 : ℕ) (h1 : a1 = a2 - 1) (h2 : a3 = a2 + 1) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by sorry

end proof_problem_l252_252246


namespace sum_of_first_6033_terms_l252_252884

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms (a r : ℝ) (h1 : geometric_sum a r 2011 = 200) 
  (h2 : geometric_sum a r 4022 = 380) : 
  geometric_sum a r 6033 = 542 :=
sorry

end sum_of_first_6033_terms_l252_252884


namespace four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l252_252244

theorem four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime
  (N : ℕ) (hN : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬ ∃ k : ℕ, k^2 = 4 * n * (N - n) + 1) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l252_252244


namespace total_spokes_in_garage_l252_252451

theorem total_spokes_in_garage :
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114 :=
by
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  show bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114
  sorry

end total_spokes_in_garage_l252_252451


namespace general_eq_line_BC_std_eq_circumscribed_circle_ABC_l252_252063

-- Define the points A, B, and C
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-4, 1)

-- Prove the general equation of line BC is x + 1 = 0
theorem general_eq_line_BC : ∀ x y : ℝ, (x = -1) → y = 2 ∧ (x = -4) → y = 1 → x + 1 = 0 :=
by
  sorry

-- Prove the standard equation of the circumscribed circle of triangle ABC is (x + 5/2)^2 + (y - 3/2)^2 = 5/2
theorem std_eq_circumscribed_circle_ABC :
  ∀ x y : ℝ,
  (x, y) = (A : ℝ × ℝ) ∨ (x, y) = (B : ℝ × ℝ) ∨ (x, y) = (C : ℝ × ℝ) →
  (x + 5/2)^2 + (y - 3/2)^2 = 5/2 :=
by
  sorry

end general_eq_line_BC_std_eq_circumscribed_circle_ABC_l252_252063


namespace merry_boxes_on_sunday_l252_252694

theorem merry_boxes_on_sunday
  (num_boxes_saturday : ℕ := 50)
  (apples_per_box : ℕ := 10)
  (total_apples_sold : ℕ := 720)
  (remaining_boxes : ℕ := 3) :
  num_boxes_saturday * apples_per_box ≤ total_apples_sold →
  (total_apples_sold - num_boxes_saturday * apples_per_box) / apples_per_box + remaining_boxes = 25 := by
  intros
  sorry

end merry_boxes_on_sunday_l252_252694


namespace tilly_total_profit_l252_252891

theorem tilly_total_profit :
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  total_profit = 300 :=
by
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  sorry

end tilly_total_profit_l252_252891


namespace min_quadratic_expression_l252_252412

theorem min_quadratic_expression:
  ∀ x : ℝ, x = 3 → (x^2 - 6 * x + 5 = -4) :=
by
  sorry

end min_quadratic_expression_l252_252412


namespace box_width_l252_252571

theorem box_width (W : ℕ) (h₁ : 15 * W * 13 = 3120) : W = 16 := by
  sorry

end box_width_l252_252571


namespace product_of_roots_eq_neg30_l252_252367

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end product_of_roots_eq_neg30_l252_252367


namespace quadratic_function_inequality_l252_252038

theorem quadratic_function_inequality
  (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hx1_pos : 0 < x1)
  (hx2_pos : x1 < x2)
  (hy1 : y1 = x1^2 - 1)
  (hy2 : y2 = x2^2 - 1) :
  y1 < y2 := 
sorry

end quadratic_function_inequality_l252_252038


namespace unknown_number_is_six_l252_252335

theorem unknown_number_is_six (n : ℝ) (h : 12 * n^4 / 432 = 36) : n = 6 :=
by 
  -- This will be the placeholder for the proof
  sorry

end unknown_number_is_six_l252_252335


namespace binom_12_6_l252_252930

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l252_252930


namespace num_solutions_g_l252_252455

noncomputable def g : ℝ → ℝ
| x := if -5 ≤ x ∧ x ≤ -1 then -(x + 3) ^ 2 + 4
       else if -1 < x ∧ x ≤ 3 then x - 1
       else if 3 < x ∧ x ≤ 5 then (x - 4) ^ 2 + 1
       else 0

theorem num_solutions_g (h : ∀ x, -5 ≤ x ∧ x ≤ 5) :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ g(g(x₁)) = 3 ∧ g(g(x₂)) = 3 ∧
  ∀ y, g(g(y)) = 3 → (y = x₁ ∨ y = x₂) := by
  sorry

end num_solutions_g_l252_252455


namespace remaining_balance_on_phone_card_l252_252380

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end remaining_balance_on_phone_card_l252_252380


namespace intersection_of_M_and_N_l252_252847

open Set

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}
def I : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = I := by
  sorry

end intersection_of_M_and_N_l252_252847


namespace find_m_and_equation_of_l2_l252_252986

theorem find_m_and_equation_of_l2 (a : ℝ) (M: ℝ × ℝ) (m : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (hM : M = (-5, 1)) 
  (hl1 : ∀ {x y : ℝ}, 2 * x - y + 2 = 0) 
  (hl : ∀ {x y : ℝ}, x + y + m = 0) 
  (hl2 : ∀ {x y : ℝ}, (∃ p : ℝ × ℝ, p = M → x - 2 * y + 7 = 0)) : 
  m = -5 ∧ ∀ {x y : ℝ}, x - 2 * y + 7 = 0 :=
by
  sorry

end find_m_and_equation_of_l2_l252_252986


namespace rulers_left_l252_252719

variable (rulers_in_drawer : Nat)
variable (rulers_taken : Nat)

theorem rulers_left (h1 : rulers_in_drawer = 46) (h2 : rulers_taken = 25) : 
  rulers_in_drawer - rulers_taken = 21 := by
  sorry

end rulers_left_l252_252719


namespace white_balls_in_bag_l252_252297

theorem white_balls_in_bag:
  ∀ (total balls green yellow red purple : Nat),
  total = 60 →
  green = 18 →
  yellow = 8 →
  red = 5 →
  purple = 7 →
  (1 - 0.8) = (red + purple : ℚ) / total →
  (W + green + yellow = total - (red + purple : ℚ)) →
  W = 22 :=
by
  intros total balls green yellow red purple ht hg hy hr hp hprob heqn
  sorry

end white_balls_in_bag_l252_252297


namespace other_train_length_l252_252743

-- Define a theorem to prove that the length of the other train (L) is 413.95 meters
theorem other_train_length (length_first_train : ℝ) (speed_first_train_kmph : ℝ) 
                           (speed_second_train_kmph: ℝ) (time_crossing_seconds : ℝ) : 
                           length_first_train = 350 → 
                           speed_first_train_kmph = 150 →
                           speed_second_train_kmph = 100 →
                           time_crossing_seconds = 11 →
                           ∃ (L : ℝ), L = 413.95 :=
by
  intros h1 h2 h3 h4
  sorry

end other_train_length_l252_252743


namespace largest_y_coordinate_ellipse_l252_252161

theorem largest_y_coordinate_ellipse:
  (∀ x y : ℝ, (x^2 / 49) + ((y + 3)^2 / 25) = 1 → y ≤ 2)  ∧ 
  (∃ x : ℝ, (x^2 / 49) + ((2 + 3)^2 / 25) = 1) := sorry

end largest_y_coordinate_ellipse_l252_252161


namespace largest_divisible_n_l252_252134

theorem largest_divisible_n (n : ℕ) :
  (n^3 + 2006) % (n + 26) = 0 → n = 15544 :=
sorry

end largest_divisible_n_l252_252134


namespace average_licks_l252_252623

theorem average_licks 
  (Dan_licks : ℕ := 58)
  (Michael_licks : ℕ := 63)
  (Sam_licks : ℕ := 70)
  (David_licks : ℕ := 70)
  (Lance_licks : ℕ := 39) :
  (Dan_licks + Michael_licks + Sam_licks + David_licks + Lance_licks) / 5 = 60 := 
sorry

end average_licks_l252_252623


namespace at_least_30_cents_probability_l252_252867

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l252_252867


namespace positive_difference_of_numbers_l252_252874

theorem positive_difference_of_numbers (x : ℝ) (h : (30 + x) / 2 = 34) : abs (x - 30) = 8 :=
by
  sorry

end positive_difference_of_numbers_l252_252874


namespace total_pounds_of_food_l252_252651

-- Conditions
def chicken := 16
def hamburgers := chicken / 2
def hot_dogs := hamburgers + 2
def sides := hot_dogs / 2

-- Define the total pounds of food
def total_food := chicken + hamburgers + hot_dogs + sides

-- Theorem statement that corresponds to the problem, showing the final result
theorem total_pounds_of_food : total_food = 39 := 
by
  -- Placeholder for the proof
  sorry

end total_pounds_of_food_l252_252651


namespace total_cats_in_academy_is_100_l252_252310

open Finset

def CleverCatAcademy :=
  let J := {1, 2, ..., 60}     -- cats that can jump
  let PD := {61, 62, ..., 95}  -- cats that can play dead
  let F := {96, 97, ..., 135}  -- cats that can fetch
  let JP := {136, 137, ..., 155} -- cats that can jump and play dead
  let PF := {156, 157, ..., 170} -- cats that can play dead and fetch
  let JF := {171, 172, ..., 192} -- cats that can jump and fetch
  let AllThree := {193, ..., 202} -- cats that can do all three tricks
  let None := {203, ..., 214} -- cats that can do none of the tricks
  J ∪ PD ∪ F ∪ JP ∪ PF ∪ JF ∪ AllThree ∪ None

noncomputable def cleverCatAcademySet : Finset ℕ :=
  {1, ..., 214}.erase (197+205) -- removing index to define cats only once across sets

theorem total_cats_in_academy_is_100 : (cleverCatAcademySet.card = 100) := by
  let J := {1, ..., 60} -- 60 cats can jump
  let PD := {61, ..., 95} -- 35 cats can play dead
  let F := {96, ..., 135} -- 40 cats can fetch
  let JP := {136, ..., 155} -- 20 cats can jump and play dead
  let PF := {156, ..., 170} -- 15 cats can play dead and fetch
  let JF := {171, ..., 192} -- 22 cats can jump and fetch
  let AllThree := {193, ..., 202} -- 10 cats can do all three tricks
  let None := {203, ..., 214} -- 12 cats can do none of the tricks
  let Academy := J ∪ PD ∪ F ∪ JP ∪ PF ∪ JF ∪ AllThree ∪ None
  have h1: 60 cats can jump := rfl
  have h2: 35 cats can play dead := rfl
  have h3: 40 cats can fetch := rfl
  have h4: 20 cats can jump and play dead := rfl
  have h5: 15 cats can play dead and fetch := rfl
  have h6: 22 cats can jump and fetch := rfl
  have h7: 10 cats can do all three tricks := rfl
  have h8: 12 cats can do none of the tricks := rfl
  show_finite Academy
  sorry -- Proof skipped

end total_cats_in_academy_is_100_l252_252310


namespace right_triangle_shorter_leg_l252_252818

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l252_252818


namespace find_f_four_thirds_l252_252200

def f (y: ℝ) : ℝ := sorry  -- Placeholder for the function definition

theorem find_f_four_thirds : f (4 / 3) = - (7 / 2) := sorry

end find_f_four_thirds_l252_252200


namespace find_x_l252_252279

-- Define the problem conditions.
def workers := ℕ
def gadgets := ℕ
def gizmos := ℕ
def hours := ℕ

-- Given conditions
def condition1 (g h : ℝ) := (1 / g = 2) ∧ (1 / h = 3)
def condition2 (g h : ℝ) := (100 * 3 / g = 900) ∧ (100 * 3 / h = 600)
def condition3 (x : ℕ) (g h : ℝ) := (40 * 4 / g = x) ∧ (40 * 4 / h = 480)

-- Proof problem statement
theorem find_x (g h : ℝ) (x : ℕ) : 
  condition1 g h → condition2 g h → condition3 x g h → x = 320 :=
by 
  intros h1 h2 h3
  sorry

end find_x_l252_252279


namespace max_z_under_D_le_1_l252_252843

noncomputable def f (x a b : ℝ) : ℝ := x - a * x^2 + b
noncomputable def f0 (x b0 : ℝ) : ℝ := x^2 + b0
noncomputable def g (x a b b0 : ℝ) : ℝ := f x a b - f0 x b0

theorem max_z_under_D_le_1 
  (a b b0 : ℝ) (D : ℝ)
  (h_a : a = 0) 
  (h_b0 : b0 = 0) 
  (h_D : D ≤ 1)
  (h_maxD : ∀ x : ℝ, - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 → g (Real.sin x) a b b0 ≤ D) :
  ∃ z : ℝ, z = b - a^2 / 4 ∧ z = 1 :=
by
  sorry

end max_z_under_D_le_1_l252_252843


namespace problem1_problem2_l252_252593

-- Problem 1: Calculate \(\sqrt[3]{8} + |-5| + (-1)^{2023} = 6\)
theorem problem1 : (2 + 5 + (-1)).toReal = 6 := by
  sorry

-- Problem 2: Find the expression of the linear function passing through (0, 1) and (2, 5) as y = 2x + 1
theorem problem2 (k b : ℝ) (h1 : b = 1) (h2 : 5 = 2 * k + b) : 
  (∀ x, k * x + b = 2 * x + 1) := by
  sorry

end problem1_problem2_l252_252593


namespace shorter_leg_of_right_triangle_l252_252822

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l252_252822


namespace solve_system_eq_l252_252860

theorem solve_system_eq (a b c x y z : ℝ) (h1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (h2 : x / a + y / b + z / c = a + b + c) (h3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c :=
by
  sorry

end solve_system_eq_l252_252860


namespace angles_congruence_mod_360_l252_252440

theorem angles_congruence_mod_360 (a b c d : ℤ) : 
  (a = 30) → (b = -30) → (c = 630) → (d = -630) →
  (b % 360 = 330 % 360) ∧ 
  (a % 360 ≠ 330 % 360) ∧ (c % 360 ≠ 330 % 360) ∧ (d % 360 ≠ 330 % 360) :=
by
  intros
  sorry

end angles_congruence_mod_360_l252_252440


namespace total_food_pounds_l252_252654

theorem total_food_pounds (chicken hamburger hot_dogs sides : ℕ) 
  (h1 : chicken = 16) 
  (h2 : hamburger = chicken / 2) 
  (h3 : hot_dogs = hamburger + 2) 
  (h4 : sides = hot_dogs / 2) : 
  chicken + hamburger + hot_dogs + sides = 39 := 
  by 
    sorry

end total_food_pounds_l252_252654


namespace triangle_side_lengths_consecutive_l252_252402

theorem triangle_side_lengths_consecutive (n : ℕ) (a b c A : ℕ) 
  (h1 : a = n - 1) (h2 : b = n) (h3 : c = n + 1) (h4 : A = n + 2)
  (h5 : 2 * A * A = 3 * n^2 * (n^2 - 4)) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
sorry

end triangle_side_lengths_consecutive_l252_252402


namespace shorter_leg_of_right_triangle_l252_252819

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l252_252819


namespace total_weight_30_l252_252278

-- Definitions of initial weights and ratio conditions
variables (a b : ℕ)
def initial_weights (h1 : a = 4 * b) : Prop := True

-- Definitions of transferred weights
def transferred_weights (a' b' : ℕ) (h2 : a' = a - 10) (h3 : b' = b + 10) : Prop := True

-- Definition of the new ratio condition
def new_ratio (a' b' : ℕ) (h4 : 8 * a' = 7 * b') : Prop := True

-- The final proof statement
theorem total_weight_30 (a b a' b' : ℕ)
    (h1 : a = 4 * b) 
    (h2 : a' = a - 10) 
    (h3 : b' = b + 10)
    (h4 : 8 * a' = 7 * b') : a + b = 30 := 
    sorry

end total_weight_30_l252_252278


namespace binomial_12_6_eq_924_l252_252958

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l252_252958


namespace rectangle_measurement_error_l252_252508

theorem rectangle_measurement_error
  (L W : ℝ)
  (x : ℝ)
  (h1 : ∀ x, L' = L * (1 + x / 100))
  (h2 : W' = W * 0.9)
  (h3 : A = L * W)
  (h4 : A' = A * 1.08) :
  x = 20 :=
by
  sorry

end rectangle_measurement_error_l252_252508


namespace part1_part2_l252_252589

-- Part 1
theorem part1 : real.cbrt 8 + abs (-5 : ℤ) + (-1 : ℤ) ^ 2023 = 6 := by
  sorry

-- Part 2
theorem part2 : ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (y = 1) ∧ (y = 2 * 2 + 1) := by
  sorry

end part1_part2_l252_252589


namespace range_of_m_l252_252041

noncomputable def f (m x : ℝ) : ℝ :=
  1 - m * (Real.exp x) / (x^2 + x + 1)

theorem range_of_m (m : ℝ) :
  (∃ x : ℕ, 0 < x ∧ f m x ≥ 0 ∧ (∀ y : ℕ, (0 < y ∧ y ≠ x) → f m y < 0)) →
  (∃ a b : ℝ, a = 7 / Real.exp 2 ∧ b = 3 / Real.exp 1 ∧ (a < m ∧ m ≤ b)) :=
sorry

end range_of_m_l252_252041


namespace certain_number_value_l252_252646

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

end certain_number_value_l252_252646


namespace find_fraction_divide_equal_l252_252456

theorem find_fraction_divide_equal (x : ℚ) : 
  (3 * x = (1 / (5 / 2))) → (x = 2 / 15) :=
by
  intro h
  sorry

end find_fraction_divide_equal_l252_252456


namespace at_least_one_not_less_than_two_l252_252526

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l252_252526


namespace black_pens_per_student_l252_252684

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

end black_pens_per_student_l252_252684


namespace integer_value_of_K_l252_252012

theorem integer_value_of_K (K : ℤ) : 
  (1000 < K^4 ∧ K^4 < 5000) ∧ K > 1 → K = 6 ∨ K = 7 ∨ K = 8 :=
by sorry

end integer_value_of_K_l252_252012


namespace abs_h_eq_one_l252_252274

theorem abs_h_eq_one (h : ℝ) (roots_square_sum_eq : ∀ x : ℝ, x^2 + 6 * h * x + 8 = 0 → x^2 + (x + 6 * h)^2 = 20) : |h| = 1 :=
by
  sorry

end abs_h_eq_one_l252_252274


namespace chameleons_changed_color_l252_252232

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l252_252232


namespace Benjie_is_older_by_5_l252_252311

def BenjieAge : ℕ := 6
def MargoFutureAge : ℕ := 4
def YearsToFuture : ℕ := 3

theorem Benjie_is_older_by_5 :
  BenjieAge - (MargoFutureAge - YearsToFuture) = 5 :=
by
  sorry

end Benjie_is_older_by_5_l252_252311


namespace find_five_dollar_bills_l252_252602

-- Define the number of bills
def total_bills (x y : ℕ) : Prop := x + y = 126

-- Define the total value of the bills
def total_value (x y : ℕ) : Prop := 5 * x + 10 * y = 840

-- Now we state the theorem
theorem find_five_dollar_bills (x y : ℕ) (h1 : total_bills x y) (h2 : total_value x y) : x = 84 :=
by sorry

end find_five_dollar_bills_l252_252602


namespace smallest_digit_divisible_by_9_l252_252030

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l252_252030


namespace solve_for_x_l252_252482

theorem solve_for_x (x y : ℝ) (h : 3 * x - 4 * y = 5) : x = (1 / 3) * (5 + 4 * y) :=
  sorry

end solve_for_x_l252_252482


namespace compute_problem_l252_252159

theorem compute_problem : (19^12 / 19^8)^2 = 130321 := by
  sorry

end compute_problem_l252_252159


namespace tan_alpha_eq_neg_four_thirds_l252_252779

theorem tan_alpha_eq_neg_four_thirds
  (α : ℝ) (hα1 : 0 < α ∧ α < π) 
  (hα2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = - 4 / 3 := 
  sorry

end tan_alpha_eq_neg_four_thirds_l252_252779


namespace chameleons_color_change_l252_252212

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l252_252212


namespace infinite_sequence_domain_l252_252201

def seq_domain (f : ℕ → ℕ) : Set ℕ := {n | 0 < n}

theorem infinite_sequence_domain (f : ℕ → ℕ) (a_n : ℕ → ℕ)
   (h : ∀ (n : ℕ), a_n n = f n) : 
   seq_domain f = {n | 0 < n} :=
sorry

end infinite_sequence_domain_l252_252201


namespace relationship_between_a_and_b_l252_252035

def a : ℤ := (-12) * (-23) * (-34) * (-45)
def b : ℤ := (-123) * (-234) * (-345)

theorem relationship_between_a_and_b : a > b := by
  sorry

end relationship_between_a_and_b_l252_252035


namespace blocks_used_for_fenced_area_l252_252685

theorem blocks_used_for_fenced_area
  (initial_blocks : ℕ) (building_blocks : ℕ) (farmhouse_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 344 →
  building_blocks = 80 →
  farmhouse_blocks = 123 →
  remaining_blocks = 84 →
  initial_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 :=
by
  intros h1 h2 h3 h4
  sorry

end blocks_used_for_fenced_area_l252_252685


namespace smallest_divisor_of_2880_that_gives_perfect_square_is_5_l252_252734

theorem smallest_divisor_of_2880_that_gives_perfect_square_is_5 :
  (∃ x : ℕ, x ≠ 0 ∧ 2880 % x = 0 ∧ (∃ y : ℕ, 2880 / x = y * y) ∧ x = 5) := by
  sorry

end smallest_divisor_of_2880_that_gives_perfect_square_is_5_l252_252734


namespace negation_of_universal_l252_252107

theorem negation_of_universal (h : ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) : ∃ x : ℝ, x^2 + 2 * x + 5 = 0 :=
sorry

end negation_of_universal_l252_252107


namespace right_triangle_shorter_leg_l252_252828

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l252_252828


namespace unique_cell_50_distance_l252_252262

-- Define the distance between two cells
def kingDistance (p1 p2 : ℤ × ℤ) : ℤ :=
  max (abs (p1.1 - p2.1)) (abs (p1.2 - p2.2))

-- A condition stating three cells with specific distances
variables (A B C : ℤ × ℤ) (hAB : kingDistance A B = 100) (hBC : kingDistance B C = 100) (hCA : kingDistance C A = 100)

-- A proposition to prove there is exactly one cell at a distance of 50 from all three given cells
theorem unique_cell_50_distance : ∃! D : ℤ × ℤ, kingDistance D A = 50 ∧ kingDistance D B = 50 ∧ kingDistance D C = 50 :=
sorry

end unique_cell_50_distance_l252_252262


namespace island_solution_l252_252300

-- Definitions based on conditions
def is_liar (n : ℕ) (m : ℕ) : Prop := n = m + 2 ∨ n = m - 2
def is_truth_teller (n : ℕ) (m : ℕ) : Prop := n = m

-- Residents' statements
def first_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1001 ∧ is_truth_teller truth_tellers 1002 ∨
  is_liar liars 1001 ∧ is_liar truth_tellers 1002

def second_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1000 ∧ is_truth_teller truth_tellers 999 ∨
  is_liar liars 1000 ∧ is_liar truth_tellers 999

-- Proving the correct number of liars and truth-tellers, and identifying the residents
theorem island_solution :
  ∃ (liars : ℕ) (truth_tellers : ℕ),
    first_resident_statement (liars + 1) (truth_tellers + 1) ∧
    second_resident_statement (liars + 1) (truth_tellers + 1) ∧
    liars = 1000 ∧ truth_tellers = 1000 ∧
    first_resident_statement liars truth_tellers ∧ second_resident_statement liars truth_tellers :=
by
  sorry

end island_solution_l252_252300


namespace shorter_leg_of_right_triangle_l252_252806

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252806


namespace minimum_bottles_needed_l252_252298

theorem minimum_bottles_needed :
  (∃ n : ℕ, n * 45 ≥ 720 - 20 ∧ (n - 1) * 45 < 720 - 20) ∧ 720 - 20 = 700 :=
by
  sorry

end minimum_bottles_needed_l252_252298


namespace sum_of_digits_l252_252075

def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits :
  (Finset.range 2013).sum S = 28077 :=
by 
  sorry

end sum_of_digits_l252_252075


namespace trig_inequality_l252_252842

theorem trig_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.cos β)^2 * (Real.sin β)^2) ≥ 9) := by
  sorry

end trig_inequality_l252_252842


namespace n_greater_than_7_l252_252454

theorem n_greater_than_7 (m n : ℕ) (hmn : m > n) (h : ∃k:ℕ, 22220038^m - 22220038^n = 10^8 * k) : n > 7 :=
sorry

end n_greater_than_7_l252_252454


namespace find_missing_term_l252_252315

theorem find_missing_term (a b : ℕ) : ∃ x, (2 * a - b) * x = 4 * a^2 - b^2 :=
by
  use (2 * a + b)
  sorry

end find_missing_term_l252_252315


namespace exists_n_sum_digits_n3_eq_million_l252_252883

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem exists_n_sum_digits_n3_eq_million :
  ∃ n : ℕ, sum_digits n = 100 ∧ sum_digits (n ^ 3) = 1000000 := sorry

end exists_n_sum_digits_n3_eq_million_l252_252883


namespace correctProduct_l252_252235

-- Define the digits reverse function
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- Main theorem statement
theorem correctProduct (a b : ℕ) (h1 : 9 < a ∧ a < 100) (h2 : reverseDigits a * b = 143) : a * b = 341 :=
  sorry -- proof to be provided

end correctProduct_l252_252235


namespace total_games_to_determine_winner_l252_252394

-- Conditions: Initial number of teams in the preliminary round
def initial_teams : ℕ := 24

-- Condition: Preliminary round eliminates 50% of the teams
def preliminary_round_elimination (n : ℕ) : ℕ := n / 2

-- Function to compute the required games for any single elimination tournament
def single_elimination_games (teams : ℕ) : ℕ :=
  if teams = 0 then 0
  else teams - 1

-- Proof Statement: Total number of games to determine the winner
theorem total_games_to_determine_winner (n : ℕ) (h : n = 24) :
  preliminary_round_elimination n + single_elimination_games (preliminary_round_elimination n) = 23 :=
by
  sorry

end total_games_to_determine_winner_l252_252394


namespace math_problem_l252_252497

theorem math_problem
  (x y z : ℕ)
  (h1 : z = 4)
  (h2 : x + y = 7)
  (h3 : x + z = 8) :
  x + y + z = 11 := 
by
  sorry

end math_problem_l252_252497


namespace problem_statement_l252_252369

theorem problem_statement : ∀ (x y : ℝ), |x - 2| + (y + 3)^2 = 0 → (x + y)^2023 = -1 :=
by
  intros x y h
  sorry

end problem_statement_l252_252369


namespace simple_interest_fraction_l252_252882

theorem simple_interest_fraction (P : ℝ) (R T : ℝ) (hR: R = 4) (hT: T = 5) :
  (P * R * T / 100) / P = 1 / 5 := 
by
  sorry

end simple_interest_fraction_l252_252882


namespace find_h_l252_252406

theorem find_h (h : ℝ) (r s : ℝ) (h_eq : ∀ x : ℝ, x^2 - 4 * h * x - 8 = 0)
  (sum_of_squares : r^2 + s^2 = 20) (roots_eq : x^2 - 4 * h * x - 8 = (x - r) * (x - s)) :
  h = 1 / 2 ∨ h = -1 / 2 := 
sorry

end find_h_l252_252406


namespace middle_integer_is_six_l252_252276

def valid_even_integer (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = x ∧ x = n - 2 ∧ y = n ∧ z = n + 2 ∧ x < y ∧ y < z ∧
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9

theorem middle_integer_is_six (n : ℕ) (h : valid_even_integer n) :
  n = 6 :=
by
  sorry

end middle_integer_is_six_l252_252276


namespace a_75_eq_24_l252_252368

variable {a : ℕ → ℤ}

-- Conditions for the problem
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def a_15_eq_8 : a 15 = 8 := sorry

def a_60_eq_20 : a 60 = 20 := sorry

-- The theorem we want to prove
theorem a_75_eq_24 (d : ℤ) (h_seq : is_arithmetic_sequence a d) (h15 : a 15 = 8) (h60 : a 60 = 20) : a 75 = 24 :=
  by
    sorry

end a_75_eq_24_l252_252368


namespace grandmother_age_l252_252098

theorem grandmother_age 
  (avg_age : ℝ)
  (age1 age2 age3 grandma_age : ℝ)
  (h_avg_age : avg_age = 20)
  (h_ages : age1 = 5)
  (h_ages2 : age2 = 10)
  (h_ages3 : age3 = 13)
  (h_eq : (age1 + age2 + age3 + grandma_age) / 4 = avg_age) : 
  grandma_age = 52 := 
by
  sorry

end grandmother_age_l252_252098


namespace complex_number_addition_identity_l252_252984

-- Definitions of the conditions
def imaginary_unit (i : ℂ) := i^2 = -1

def complex_fraction_decomposition (a b : ℝ) (i : ℂ) := 
  (1 + i) / (1 - i) = a + b * i

-- The statement of the problem
theorem complex_number_addition_identity :
  ∃ (a b : ℝ) (i : ℂ), imaginary_unit i ∧ complex_fraction_decomposition a b i ∧ (a + b = 1) :=
sorry

end complex_number_addition_identity_l252_252984


namespace right_triangles_not_1000_l252_252563

-- Definitions based on the conditions
def numPoints := 100
def numDiametricallyOppositePairs := numPoints / 2
def rightTrianglesPerPair := numPoints - 2
def totalRightTriangles := numDiametricallyOppositePairs * rightTrianglesPerPair

-- Theorem stating the final evaluation of the problem
theorem right_triangles_not_1000 :
  totalRightTriangles ≠ 1000 :=
by
  -- calculation shows it's impossible
  sorry

end right_triangles_not_1000_l252_252563


namespace area_difference_l252_252363

theorem area_difference (r1 d2 : ℝ) (h1 : r1 = 30) (h2 : d2 = 15) : 
  π * r1^2 - π * (d2 / 2)^2 = 843.75 * π :=
by
  sorry

end area_difference_l252_252363


namespace problem1_problem2_l252_252595

-- Definition for Problem 1
def cube_root_8 : ℝ := 2
def abs_neg5 : ℝ := 5
def pow_neg1_2023 : ℝ := -1

theorem problem1 : cube_root_8 + abs_neg5 + pow_neg1_2023 = 6 := by
  sorry

-- Definitions for Problem 2
structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := { x := 0, y := 1 }
def point2 : Point := { x := 2, y := 5 }

def linear_function (k b x : ℝ) : ℝ := k * x + b

def is_linear_function (p1 p2 : Point) (k b : ℝ) : Prop :=
  p1.y = linear_function k b p1.x ∧ p2.y = linear_function k b p2.x

theorem problem2 : ∃ k b : ℝ, is_linear_function point1 point2 k b ∧ (linear_function k b x = 2 * x + 1) := by
  sorry

end problem1_problem2_l252_252595


namespace original_price_of_article_l252_252607

theorem original_price_of_article
  (P S : ℝ) 
  (h1 : S = 1.4 * P) 
  (h2 : S - P = 560) 
  : P = 1400 :=
by
  sorry

end original_price_of_article_l252_252607


namespace system_of_equations_unique_solution_l252_252015

theorem system_of_equations_unique_solution :
  (∃ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7) →
  (∀ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7 →
    x = 26 / 5 ∧ y = 9 / 5) := 
by {
  -- Proof to be provided
  sorry
}

end system_of_equations_unique_solution_l252_252015


namespace watch_hands_angle_120_l252_252608

theorem watch_hands_angle_120 (n : ℝ) (h₁ : 0 ≤ n ∧ n ≤ 60) 
    (h₂ : abs ((210 + n / 2) - 6 * n) = 120) : n = 43.64 := sorry

end watch_hands_angle_120_l252_252608


namespace songs_before_camp_l252_252284

theorem songs_before_camp (total_songs : ℕ) (learned_at_camp : ℕ) (songs_before_camp : ℕ) (h1 : total_songs = 74) (h2 : learned_at_camp = 18) : songs_before_camp = 56 :=
by
  sorry

end songs_before_camp_l252_252284


namespace polynomial_difference_l252_252759

theorem polynomial_difference (a : ℝ) :
  (6 * a^2 - 5 * a + 3) - (5 * a^2 + 2 * a - 1) = a^2 - 7 * a + 4 :=
by
  sorry

end polynomial_difference_l252_252759


namespace point_M_first_quadrant_distances_length_of_segment_MN_l252_252037

-- Proof problem 1
theorem point_M_first_quadrant_distances (m : ℝ) (h1 : 2 * m + 1 > 0) (h2 : m + 3 > 0) (h3 : m + 3 = 2 * (2 * m + 1)) :
  m = 1 / 3 :=
by
  sorry

-- Proof problem 2
theorem length_of_segment_MN (m : ℝ) (h4 : m + 3 = 1) :
  let Mx := 2 * m + 1
  let My := m + 3
  let Nx := 2
  let Ny := 1
  let distMN := abs (Nx - Mx)
  distMN = 5 :=
by
  sorry

end point_M_first_quadrant_distances_length_of_segment_MN_l252_252037


namespace problem1_l252_252584

theorem problem1 : real.cbrt 8 + abs (-5) + (-1:ℝ)^2023 = 6 := 
by 
  sorry

end problem1_l252_252584


namespace Carter_cards_l252_252530

variable (C : ℕ) -- Let C be the number of baseball cards Carter has.

-- Condition 1: Marcus has 210 baseball cards.
def Marcus_cards : ℕ := 210

-- Condition 2: Marcus has 58 more cards than Carter.
def Marcus_has_more (C : ℕ) : Prop := Marcus_cards = C + 58

theorem Carter_cards (C : ℕ) (h : Marcus_has_more C) : C = 152 :=
by
  -- Expand the condition
  unfold Marcus_has_more at h
  -- Simplify the given equation
  rw [Marcus_cards] at h
  -- Solve for C
  linarith

end Carter_cards_l252_252530


namespace find_c_l252_252399

theorem find_c (a b c : ℝ) (h1 : ∃ a, ∃ b, ∃ c, 
              ∀ y, (∀ x, (x = a * (y-1)^2 + 4) ↔ (x = -2 → y = 3)) ∧
              (∀ y, x = a * y^2 + b * y + c)) : c = 1 / 2 :=
sorry

end find_c_l252_252399


namespace largest_cube_volume_l252_252732

theorem largest_cube_volume (width length height : ℕ) (h₁ : width = 15) (h₂ : length = 12) (h₃ : height = 8) :
  ∃ V, V = 512 := by
  use 8^3
  sorry

end largest_cube_volume_l252_252732


namespace markup_rate_correct_l252_252377

noncomputable def selling_price : ℝ := 10.00
noncomputable def profit_percentage : ℝ := 0.20
noncomputable def expenses_percentage : ℝ := 0.15
noncomputable def cost (S : ℝ) : ℝ := S - (profit_percentage * S + expenses_percentage * S)
noncomputable def markup_rate (S C : ℝ) : ℝ := (S - C) / C * 100

theorem markup_rate_correct :
  markup_rate selling_price (cost selling_price) = 53.85 := 
by
  sorry

end markup_rate_correct_l252_252377


namespace least_large_groups_l252_252141

theorem least_large_groups (total_members : ℕ) (members_large_group : ℕ) (members_small_group : ℕ) (L : ℕ) (S : ℕ)
  (H_total : total_members = 90)
  (H_large : members_large_group = 7)
  (H_small : members_small_group = 3)
  (H_eq : total_members = L * members_large_group + S * members_small_group) :
  L = 12 :=
by
  have h1 : total_members = 90 := by exact H_total
  have h2 : members_large_group = 7 := by exact H_large
  have h3 : members_small_group = 3 := by exact H_small
  rw [h1, h2, h3] at H_eq
  -- The proof is skipped here
  sorry

end least_large_groups_l252_252141


namespace chameleon_color_change_l252_252228

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l252_252228


namespace max_intersection_points_circle_sine_l252_252147

def circle_eq (x y h k : ℝ) : ℝ := (x - h)^2 + (y - k)^2

theorem max_intersection_points_circle_sine :
  ∀ (h : ℝ), ∃ k ∈ Icc (-2 : ℝ) (2 : ℝ),
    ∀ (x : ℝ), circle_eq x (Real.sin x) h k = 4 → (∃! x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ, 
    circle_eq x₁ (Real.sin x₁) h k = 4 ∧
    circle_eq x₂ (Real.sin x₂) h k = 4 ∧
    circle_eq x₃ (Real.sin x₃) h k = 4 ∧
    circle_eq x₄ (Real.sin x₄) h k = 4 ∧
    circle_eq x₅ (Real.sin x₅) h k = 4 ∧
    circle_eq x₆ (Real.sin x₆) h k = 4 ∧
    circle_eq x₇ (Real.sin x₇) h k = 4 ∧
    circle_eq x₈ (Real.sin x₈) h k = 4) := 
begin
  sorry
end

end max_intersection_points_circle_sine_l252_252147


namespace tv_interest_rate_zero_l252_252715

theorem tv_interest_rate_zero (price_installment first_installment last_installment : ℕ) 
  (installment_count : ℕ) (total_price : ℕ) : 
  total_price = 60000 ∧  
  price_installment = 1000 ∧ 
  first_installment = price_installment ∧ 
  last_installment = 59000 ∧ 
  installment_count = 20 ∧  
  (20 * price_installment = 20000) ∧
  (total_price - first_installment = 59000) →
  0 = 0 :=
by 
  sorry

end tv_interest_rate_zero_l252_252715


namespace customer_bought_29_eggs_l252_252021

-- Defining the conditions
def baskets : List ℕ := [4, 6, 12, 13, 22, 29]
def total_eggs : ℕ := 86
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Stating the problem
theorem customer_bought_29_eggs :
  ∃ eggs_in_basket,
    eggs_in_basket ∈ baskets ∧
    is_multiple_of_three (total_eggs - eggs_in_basket) ∧
    eggs_in_basket = 29 :=
by sorry

end customer_bought_29_eggs_l252_252021


namespace function_properties_l252_252979

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x > 0) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  f 1 = 3 →
  Set.range f = Set.Icc (-6) 6 :=
sorry

end function_properties_l252_252979


namespace age_of_25th_student_l252_252873

variable (total_students : ℕ) (total_average : ℕ)
variable (group1_students : ℕ) (group1_average : ℕ)
variable (group2_students : ℕ) (group2_average : ℕ)

theorem age_of_25th_student 
  (h1 : total_students = 25) 
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28) : 
  (total_students * total_average) =
  (group1_students * group1_average) + (group2_students * group2_average) + 13 :=
by sorry

end age_of_25th_student_l252_252873


namespace find_series_sum_l252_252074

noncomputable def series_sum (s : ℝ) : ℝ := ∑' n : ℕ, (n+1) * s^(4*n + 3)

theorem find_series_sum (s : ℝ) (h : s^4 - s - 1/2 = 0) : series_sum s = -4 := by
  sorry

end find_series_sum_l252_252074


namespace exists_four_numbers_product_fourth_power_l252_252854

theorem exists_four_numbers_product_fourth_power :
  ∃ (numbers : Fin 81 → ℕ),
    (∀ i, ∃ a b c : ℕ, numbers i = 2^a * 3^b * 5^c) ∧
    ∃ (i j k l : Fin 81), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ∃ m : ℕ, m^4 = numbers i * numbers j * numbers k * numbers l :=
by
  sorry

end exists_four_numbers_product_fourth_power_l252_252854


namespace price_per_litre_mixed_oil_l252_252890

-- Define the given conditions
def cost_oil1 : ℝ := 100 * 45
def cost_oil2 : ℝ := 30 * 57.50
def cost_oil3 : ℝ := 20 * 72
def total_cost : ℝ := cost_oil1 + cost_oil2 + cost_oil3
def total_volume : ℝ := 100 + 30 + 20

-- Define the statement to be proved
theorem price_per_litre_mixed_oil : (total_cost / total_volume) = 51.10 :=
by
  sorry

end price_per_litre_mixed_oil_l252_252890


namespace solution_set_of_inequality_l252_252969

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x + 1) * (1 - 2 * x) > 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} := 
by 
  sorry

end solution_set_of_inequality_l252_252969


namespace ratio_of_points_l252_252252

def Noa_points : ℕ := 30
def total_points : ℕ := 90

theorem ratio_of_points (Phillip_points : ℕ) (h1 : Phillip_points = 2 * Noa_points) (h2 : Noa_points + Phillip_points = total_points) : Phillip_points / Noa_points = 2 := 
by
  intros
  sorry

end ratio_of_points_l252_252252


namespace no_solution_system_l252_252258

theorem no_solution_system : ¬ ∃ (x y z : ℝ), 
  x^2 - 2*y + 2 = 0 ∧ 
  y^2 - 4*z + 3 = 0 ∧ 
  z^2 + 4*x + 4 = 0 := 
by
  sorry

end no_solution_system_l252_252258


namespace question_correctness_l252_252414

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l252_252414


namespace integer_solutions_to_inequality_l252_252672

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
1 + 2 * n^2 + 2 * n

theorem integer_solutions_to_inequality (n : ℕ) :
  ∃ (count : ℕ), count = count_integer_solutions n ∧ 
  ∀ (x y : ℤ), |x| + |y| ≤ n → (∃ (k : ℕ), k = count) :=
by
  sorry

end integer_solutions_to_inequality_l252_252672


namespace even_odd_difference_l252_252285

def even_sum_n (n : ℕ) : ℕ := (n * (n + 1))
def odd_sum_n (n : ℕ) : ℕ := n * n

theorem even_odd_difference : even_sum_n 100 - odd_sum_n 100 = 100 := by
  -- The proof goes here
  sorry

end even_odd_difference_l252_252285


namespace right_triangle_shorter_leg_l252_252814

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l252_252814


namespace sandy_bought_6_books_l252_252088

variable (initialBooks soldBooks boughtBooks remainingBooks : ℕ)

def half (n : ℕ) : ℕ := n / 2

theorem sandy_bought_6_books :
  initialBooks = 14 →
  soldBooks = half initialBooks →
  remainingBooks = initialBooks - soldBooks →
  remainingBooks + boughtBooks = 13 →
  boughtBooks = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_bought_6_books_l252_252088


namespace counterexamples_count_l252_252013

def sum_of_digits (n : Nat) : Nat :=
  -- Function to calculate the sum of digits of n
  sorry

def no_zeros (n : Nat) : Prop :=
  -- Function to check that there are no zeros in the digits of n
  sorry

def is_prime (n : Nat) : Prop :=
  -- Function to check if a number is prime
  sorry

theorem counterexamples_count : 
  ∃ (M : List Nat), 
  (∀ m ∈ M, sum_of_digits m = 5 ∧ no_zeros m) ∧ 
  (∀ m ∈ M, ¬ is_prime m) ∧
  M.length = 9 := 
sorry

end counterexamples_count_l252_252013


namespace part_I_part_II_l252_252469

-- Part I: Inequality solution
theorem part_I (x : ℝ) : 
  (abs (x - 1) ≥ 4 - abs (x - 3)) ↔ (x ≤ 0 ∨ x ≥ 4) := 
sorry

-- Part II: Minimum value of mn
theorem part_II (m n : ℕ) (h1 : (1:ℝ)/m + (1:ℝ)/(2*n) = 1) (hm : 0 < m) (hn : 0 < n) :
  (mn : ℕ) = 2 :=
sorry

end part_I_part_II_l252_252469


namespace polio_cases_in_1990_l252_252056

theorem polio_cases_in_1990 (c_1970 c_2000 : ℕ) (T : ℕ) (linear_decrease : ∀ t, c_1970 - (c_2000 * t) / T > 0):
  (c_1970 = 300000) → (c_2000 = 600) → (T = 30) → ∃ c_1990, c_1990 = 100400 :=
by
  intros
  sorry

end polio_cases_in_1990_l252_252056


namespace find_R_when_S_is_five_l252_252411

theorem find_R_when_S_is_five (g : ℚ) :
  (∀ (S : ℚ), R = g * S^2 - 5) →
  (R = 25 ∧ S = 3) →
  R = (250 / 3) - 5 :=
by 
  sorry

end find_R_when_S_is_five_l252_252411


namespace smallest_student_count_l252_252920

theorem smallest_student_count (x y z w : ℕ) 
  (ratio12to10 : x / y = 3 / 2) 
  (ratio12to11 : x / z = 7 / 4) 
  (ratio12to9 : x / w = 5 / 3) : 
  x + y + z + w = 298 :=
by
  sorry

end smallest_student_count_l252_252920


namespace tangent_line_equation_at_1_range_of_a_l252_252345

noncomputable def f (x a : ℝ) : ℝ := (x+1) * Real.log x - a * (x-1)

-- (I) Tangent line equation when a = 4
theorem tangent_line_equation_at_1 (x : ℝ) (hx : x = 1) :
  let a := 4
  2*x + f 1 a - 2 = 0 :=
sorry

-- (II) Range of values for a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_equation_at_1_range_of_a_l252_252345


namespace player_A_wins_l252_252687

theorem player_A_wins (n : ℕ) : ∃ m, (m > 2 * n^2) ∧ (∀ S : Finset (ℕ × ℕ), S.card = m → ∃ (r c : Finset ℕ), r.card = n ∧ c.card = n ∧ ∀ rc ∈ r.product c, rc ∈ S → false) :=
by sorry

end player_A_wins_l252_252687


namespace at_least_30_cents_prob_l252_252864

def coin := {penny, nickel, dime, quarter, half_dollar}
def value (c : coin) : ℕ := 
  match c with
  | penny => 1
  | nickel => 5
  | dime => 10
  | quarter => 25
  | half_dollar => 50

def coin_positions : List (coin × Bool) := 
  [(penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, false)]

def count_successful_outcomes : ℕ :=
  List.length (List.filter (λ positions, List.foldl (λ acc (c, h) => if h then acc + value c else acc) 0 positions >= 30) coin_positions)

def total_outcomes : ℕ := 32

def probability_of_success : ℚ :=
  ⟨count_successful_outcomes, total_outcomes⟩

theorem at_least_30_cents_prob : probability_of_success = 3 / 4 :=
by sorry

end at_least_30_cents_prob_l252_252864


namespace max_correct_answers_l252_252059

-- Definitions based on the conditions
def total_problems : ℕ := 12
def points_per_correct : ℕ := 6
def points_per_incorrect : ℕ := 3
def max_score : ℤ := 37 -- Final score, using ℤ to handle potential negatives in deducting points

-- The statement to prove
theorem max_correct_answers :
  ∃ (c w : ℕ), c + w = total_problems ∧ points_per_correct * c - points_per_incorrect * (total_problems - c) = max_score ∧ c = 8 :=
by
  sorry

end max_correct_answers_l252_252059


namespace initial_tomatoes_l252_252749

def t_picked : ℕ := 83
def t_left : ℕ := 14
def t_total : ℕ := t_picked + t_left

theorem initial_tomatoes : t_total = 97 := by
  rw [t_total]
  rfl

end initial_tomatoes_l252_252749


namespace martha_initial_juice_pantry_l252_252152

theorem martha_initial_juice_pantry (P : ℕ) : 
  4 + P + 5 - 3 = 10 → P = 4 := 
by
  intro h
  sorry

end martha_initial_juice_pantry_l252_252152


namespace five_person_lineup_l252_252062

theorem five_person_lineup : 
  let total_ways := Nat.factorial 5
  let invalid_first := Nat.factorial 4
  let invalid_last := Nat.factorial 4
  let valid_ways := total_ways - (invalid_first + invalid_last)
  valid_ways = 72 :=
by
  sorry

end five_person_lineup_l252_252062


namespace problem1_problem2_l252_252317

theorem problem1 : (Real.sqrt 24 - Real.sqrt 18) - Real.sqrt 6 = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : 2 * Real.sqrt 12 * Real.sqrt (1 / 8) + 5 * Real.sqrt 2 = Real.sqrt 6 + 5 * Real.sqrt 2 := by
  sorry

end problem1_problem2_l252_252317


namespace area_ratio_l252_252383

-- Definitions corresponding to the conditions
variable {A B C P Q R : Type}
variable (t : ℝ)
variable (h_pos : 0 < t) (h_lt_one : t < 1)

-- Define the areas in terms of provided conditions
noncomputable def area_AP : ℝ := sorry
noncomputable def area_BQ : ℝ := sorry
noncomputable def area_CR : ℝ := sorry
noncomputable def K : ℝ := area_AP * area_BQ * area_CR
noncomputable def L : ℝ := sorry -- Area of triangle ABC

-- The statement to be proved
theorem area_ratio (h_pos : 0 < t) (h_lt_one : t < 1) :
  (K / L) = (1 - t + t^2)^2 :=
sorry

end area_ratio_l252_252383


namespace bottles_left_on_shelf_l252_252120

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l252_252120


namespace problem1_problem2_l252_252579

-- Statement for Problem 1
theorem problem1 : real.cbrt 8 + abs (-5) + (-1 : ℤ) ^ 2023 = 6 := 
by sorry

-- Statement for Problem 2
theorem problem2 (k b : ℝ) (h1 : (0 : ℝ) * k + b = 1)
  (h2 : (2 : ℝ) * k + b = 5) : k = 2 ∧ b = 1 ∧ ∀ x, (k * x + b) = 2 * x + 1 :=
by sorry

end problem1_problem2_l252_252579


namespace parallel_lines_a_values_l252_252785

theorem parallel_lines_a_values (a : Real) : 
  (∃ k : Real, 2 = k * a ∧ -a = k * (-8)) ↔ (a = 4 ∨ a = -4) := sorry

end parallel_lines_a_values_l252_252785


namespace binom_12_6_l252_252948

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l252_252948


namespace find_natural_pairs_l252_252771

-- Definitions
def is_natural (n : ℕ) : Prop := n > 0
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def satisfies_equation (x y : ℕ) : Prop := 2 * x^2 + 5 * x * y + 3 * y^2 = 41 * x + 62 * y + 21

-- Problem statement
theorem find_natural_pairs (x y : ℕ) (hx : is_natural x) (hy : is_natural y) (hrel : relatively_prime x y) :
  satisfies_equation x y ↔ (x = 2 ∧ y = 19) ∨ (x = 19 ∧ y = 2) :=
by
  sorry

end find_natural_pairs_l252_252771


namespace orchid_bushes_planted_tomorrow_l252_252280

theorem orchid_bushes_planted_tomorrow 
  (initial : ℕ) (planted_today : ℕ) (final : ℕ) (planted_tomorrow : ℕ) :
  initial = 47 →
  planted_today = 37 →
  final = 109 →
  planted_tomorrow = final - (initial + planted_today) →
  planted_tomorrow = 25 :=
by
  intros h_initial h_planted_today h_final h_planted_tomorrow
  rw [h_initial, h_planted_today, h_final] at h_planted_tomorrow
  exact h_planted_tomorrow


end orchid_bushes_planted_tomorrow_l252_252280


namespace shorter_leg_of_right_triangle_l252_252821

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l252_252821


namespace bridge_length_l252_252264

def train_length : ℕ := 120
def train_speed : ℕ := 45
def crossing_time : ℕ := 30

theorem bridge_length :
  let speed_m_per_s := (train_speed * 1000) / 3600
  let total_distance := speed_m_per_s * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 255 := by
  sorry

end bridge_length_l252_252264


namespace boat_speed_l252_252111

theorem boat_speed (v : ℝ) : 
  let rate_current := 7
  let distance := 35.93
  let time := 44 / 60
  (v + rate_current) * time = distance → v = 42 :=
by
  intro h
  sorry

end boat_speed_l252_252111


namespace calvin_buys_chips_days_per_week_l252_252318

-- Define the constants based on the problem conditions
def cost_per_pack : ℝ := 0.50
def total_amount_spent : ℝ := 10
def number_of_weeks : ℕ := 4

-- Define the proof statement
theorem calvin_buys_chips_days_per_week : 
  (total_amount_spent / cost_per_pack) / number_of_weeks = 5 := 
by
  -- Placeholder proof
  sorry

end calvin_buys_chips_days_per_week_l252_252318


namespace zero_point_interval_l252_252327

noncomputable def f (x : ℝ) : ℝ := log 10 x + x - 2

theorem zero_point_interval : ∃ x ∈ set.Ioo 1 2, f x = 0 := 
by
  -- We need to prove that the zero point of function f lies in the interval (1, 2).
  sorry

end zero_point_interval_l252_252327


namespace rationalize_denominator_sum_l252_252254

theorem rationalize_denominator_sum :
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  A + B + C + D + E + F = 210 :=
by
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  show 3 + -9 + -9 + 9 + 165 + 51 = 210
  sorry

end rationalize_denominator_sum_l252_252254


namespace product_of_last_two_digits_l252_252992

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 11) (h2 : ∃ (n : ℕ), 10 * A + B = 6 * n) : A * B = 24 :=
sorry

end product_of_last_two_digits_l252_252992


namespace calligraphy_prices_max_brushes_l252_252158

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

end calligraphy_prices_max_brushes_l252_252158


namespace same_terminal_side_l252_252716

theorem same_terminal_side (k : ℤ) : 
  {α | ∃ k : ℤ, α = k * 360 + (-263 : ℤ)} = 
  {α | ∃ k : ℤ, α = k * 360 - 263} := 
by sorry

end same_terminal_side_l252_252716


namespace average_licks_l252_252624

theorem average_licks 
  (Dan_licks : ℕ := 58)
  (Michael_licks : ℕ := 63)
  (Sam_licks : ℕ := 70)
  (David_licks : ℕ := 70)
  (Lance_licks : ℕ := 39) :
  (Dan_licks + Michael_licks + Sam_licks + David_licks + Lance_licks) / 5 = 60 := 
sorry

end average_licks_l252_252624


namespace shorter_leg_of_right_triangle_l252_252801

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252801


namespace rectangle_side_greater_than_12_l252_252188

theorem rectangle_side_greater_than_12 
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 := 
by
  sorry

end rectangle_side_greater_than_12_l252_252188


namespace tan_of_alpha_l252_252476

theorem tan_of_alpha
  (α : ℝ)
  (h1 : Real.sin (α + Real.pi / 2) = 1 / 3)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α = 2 * Real.sqrt 2 := 
sorry

end tan_of_alpha_l252_252476


namespace find_number_l252_252450

theorem find_number (x : ℚ) (h : x / 11 + 156 = 178) : x = 242 :=
sorry

end find_number_l252_252450


namespace simplify_and_rationalize_denominator_l252_252091

theorem simplify_and_rationalize_denominator :
  ( (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) = 3 * Real.sqrt 420 / 42 ) := 
by {
  sorry
}

end simplify_and_rationalize_denominator_l252_252091


namespace find_8th_result_l252_252398

theorem find_8th_result 
  (S_17 : ℕ := 17 * 24) 
  (S_7 : ℕ := 7 * 18) 
  (S_5_1 : ℕ := 5 * 23) 
  (S_5_2 : ℕ := 5 * 32) : 
  S_17 - S_7 - S_5_1 - S_5_2 = 7 := 
by
  sorry

end find_8th_result_l252_252398


namespace percentage_reduction_l252_252752

variable (P R : ℝ)
variable (ReducedPrice : R = 15)
variable (AmountMore : 900 / 15 - 900 / P = 6)

theorem percentage_reduction (ReducedPrice : R = 15) (AmountMore : 900 / 15 - 900 / P = 6) :
  (P - R) / P * 100 = 10 :=
by
  sorry

end percentage_reduction_l252_252752


namespace customers_who_did_not_tip_l252_252155

def total_customers := 10
def total_tips := 15
def tip_per_customer := 3

theorem customers_who_did_not_tip : total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_who_did_not_tip_l252_252155


namespace binomial_12_6_eq_924_l252_252957

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l252_252957


namespace right_triangle_shorter_leg_l252_252826

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l252_252826


namespace maximum_n_l252_252174

/-- Definition of condition (a): For any three people, there exist at least two who know each other. -/
def condition_a (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 3 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), G.Adj a b

/-- Definition of condition (b): For any four people, there exist at least two who do not know each other. -/
def condition_b (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 4 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), ¬ G.Adj a b

theorem maximum_n (G : SimpleGraph V) [Fintype V] (h1 : condition_a G) (h2 : condition_b G) : 
  Fintype.card V ≤ 8 :=
by
  sorry

end maximum_n_l252_252174


namespace algebraic_expression_value_l252_252183

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 :=
by
  sorry

end algebraic_expression_value_l252_252183


namespace lisa_needs_additional_marbles_l252_252392

theorem lisa_needs_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 50) :
  ∃ additional_marbles : ℕ, additional_marbles = 78 - marbles ∧ additional_marbles = 28 :=
by
  -- The sum of the first 12 natural numbers is calculated as:
  have h_sum : (∑ i in finset.range (friends + 1), i) = 78 := by sorry
  -- The additional marbles needed:
  use 78 - marbles
  -- It should equal to 28:
  split
  . exact rfl
  . sorry

end lisa_needs_additional_marbles_l252_252392


namespace probability_sum_divisible_by_3_l252_252205

-- Define the first ten prime numbers
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate to check divisibility by 3
def divisibleBy3 (n : ℕ) : Prop := n % 3 = 0

-- Define the main theorem statement
theorem probability_sum_divisible_by_3 :
  (let pairs := (firstTenPrimes.product firstTenPrimes).filter (λ (x : ℕ × ℕ), x.1 < x.2) in
    let totalPairs := pairs.length in
    let divisiblePairs := pairs.count (λ (x : ℕ × ℕ), divisibleBy3 (x.1 + x.2)) in
    (divisiblePairs.to_rat / totalPairs.to_rat) = (1 : ℚ) / 3) :=
begin
  sorry -- Proof is not required.
end

end probability_sum_divisible_by_3_l252_252205


namespace sum_of_roots_l252_252248

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (3 + x) = f (3 - x)) →
  (∃ (S : Finset ℝ), S.card = 6 ∧ ∀ x ∈ S, f x = 0) →
  (∃ (S : Finset ℝ), S.sum id = 18) :=
by
  sorry

end sum_of_roots_l252_252248


namespace binom_12_6_l252_252947

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l252_252947


namespace dinner_serving_problem_l252_252892

theorem dinner_serving_problem : 
  let orders := ["B", "B", "B", "B", "C", "C", "C", "C", "F", "F", "F", "F"].to_finset in
  let possible_serving_count := choose 12 2 * 160 in
  ∃ (serving : set (fin 12)), 
    (serving : cardinal) = 2 ∧
    (orders = serving) →
    possible_serving_count = 211200
:= 
begin
  sorry
end

end dinner_serving_problem_l252_252892


namespace find_number_l252_252197

-- Define the hypothesis/condition
def condition (x : ℤ) : Prop := 2 * x + 20 = 8 * x - 4

-- Define the statement to prove
theorem find_number (x : ℤ) (h : condition x) : x = 4 := 
by
  sorry

end find_number_l252_252197


namespace smallest_repunit_divisible_by_97_l252_252513

theorem smallest_repunit_divisible_by_97 :
  ∃ n : ℕ, (∃ d : ℤ, 10^n - 1 = 97 * 9 * d) ∧ (∀ m : ℕ, (∃ d : ℤ, 10^m - 1 = 97 * 9 * d) → n ≤ m) :=
by
  sorry

end smallest_repunit_divisible_by_97_l252_252513


namespace count_four_digit_multiples_of_7_l252_252350

theorem count_four_digit_multiples_of_7 : 
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ 7 ∣ n}.to_finset.card = 1285 := 
by
  sorry

end count_four_digit_multiples_of_7_l252_252350


namespace votes_for_winner_is_744_l252_252423

variable (V : ℝ) -- Total number of votes cast

-- Conditions
axiom two_candidates : True
axiom winner_received_62_percent : True
axiom winner_won_by_288_votes : 0.62 * V - 0.38 * V = 288

-- Theorem to prove
theorem votes_for_winner_is_744 :
  0.62 * V = 744 :=
by
  sorry

end votes_for_winner_is_744_l252_252423


namespace total_food_pounds_l252_252656

theorem total_food_pounds (chicken hamburger hot_dogs sides : ℕ) 
  (h1 : chicken = 16) 
  (h2 : hamburger = chicken / 2) 
  (h3 : hot_dogs = hamburger + 2) 
  (h4 : sides = hot_dogs / 2) : 
  chicken + hamburger + hot_dogs + sides = 39 := 
  by 
    sorry

end total_food_pounds_l252_252656


namespace jill_total_tax_percentage_l252_252081

theorem jill_total_tax_percentage (spent_clothing_percent spent_food_percent spent_other_percent tax_clothing_percent tax_food_percent tax_other_percent : ℝ)
  (h1 : spent_clothing_percent = 0.5)
  (h2 : spent_food_percent = 0.25)
  (h3 : spent_other_percent = 0.25)
  (h4 : tax_clothing_percent = 0.1)
  (h5 : tax_food_percent = 0)
  (h6 : tax_other_percent = 0.2) :
  ((spent_clothing_percent * tax_clothing_percent + spent_food_percent * tax_food_percent + spent_other_percent * tax_other_percent) * 100) = 10 :=
by
  sorry

end jill_total_tax_percentage_l252_252081


namespace inequality_inequation_l252_252857

theorem inequality_inequation (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x + y + z = 1) :
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 :=
by
  sorry

end inequality_inequation_l252_252857


namespace right_triangle_shorter_leg_l252_252825

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l252_252825


namespace exists_radius_for_marked_points_l252_252138

theorem exists_radius_for_marked_points :
  ∃ R : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ < 2 * π) →
    (∃ n : ℕ, (θ ≤ (n * 2 * π * R) % (2 * π * R) + 1 / R ∧ (n * 2 * π * R) % (2 * π * R) < θ + 1))) :=
sorry

end exists_radius_for_marked_points_l252_252138


namespace average_points_per_player_l252_252839

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end average_points_per_player_l252_252839


namespace degree_of_k_l252_252048

open Polynomial

theorem degree_of_k (h k : Polynomial ℝ) 
  (h_def : h = -5 * X^5 + 4 * X^3 - 2 * X^2 + C 8)
  (deg_sum : (h + k).degree = 2) : k.degree = 5 :=
sorry

end degree_of_k_l252_252048


namespace range_of_m_l252_252788

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l252_252788


namespace bananas_in_collection_l252_252083

theorem bananas_in_collection
  (groups : ℕ)
  (bananas_per_group : ℕ)
  (h1 : groups = 11)
  (h2 : bananas_per_group = 37) :
  (groups * bananas_per_group) = 407 :=
by sorry

end bananas_in_collection_l252_252083


namespace tangent_line_to_parabola_l252_252334

theorem tangent_line_to_parabola : ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 6 * y + k = 0) ∧ (∀ y : ℝ, ∃ x : ℝ, y^2 = 32 * x) ∧ (48^2 - 4 * (1 : ℝ) * 8 * k = 0) := by
  use 72
  sorry

end tangent_line_to_parabola_l252_252334


namespace number_of_towers_l252_252146

noncomputable def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem number_of_towers :
  (multinomial 10 3 3 4 = 4200) :=
by
  sorry

end number_of_towers_l252_252146


namespace cubic_root_abs_power_linear_function_points_l252_252586

theorem cubic_root_abs_power : real.sqrt3 8 + abs (-5) + (-1 : real)^2023 = 6 := by
  sorry

theorem linear_function_points : 
  ∃ k b : real, (0, 1) ∈ set_of (λ p : real × real, p.snd = k * p.fst + b) ∧
                (2, 5) ∈ set_of (λ p : real × real, p.snd = k * p.fst + b) ∧
                (∀ x : real, k * x + b = 2 * x + 1) := by
  sorry

end cubic_root_abs_power_linear_function_points_l252_252586


namespace calc_expression_find_linear_function_l252_252591

-- Part 1
theorem calc_expression : real.cbrt 8 + abs (-5) + (-1:ℤ) ^ 2023 = 6 :=
by
  sorry

-- Part 2
theorem find_linear_function :
  ∃ k b : ℝ, (y = k * x + b ↔ y = 2 * x + 1) ∧
  (b = 1) ∧
  (2 * k + b = 5) :=
by
  sorry

end calc_expression_find_linear_function_l252_252591


namespace wizard_concoction_valid_combinations_l252_252301

structure WizardConcoction :=
(herbs : Nat)
(crystals : Nat)
(single_incompatible : Nat)
(double_incompatible : Nat)

def valid_combinations (concoction : WizardConcoction) : Nat :=
  concoction.herbs * concoction.crystals - (concoction.single_incompatible + concoction.double_incompatible)

theorem wizard_concoction_valid_combinations (c : WizardConcoction)
  (h_herbs : c.herbs = 4)
  (h_crystals : c.crystals = 6)
  (h_single_incompatible : c.single_incompatible = 1)
  (h_double_incompatible : c.double_incompatible = 2) :
  valid_combinations c = 21 :=
by
  sorry

end wizard_concoction_valid_combinations_l252_252301


namespace product_of_possible_values_N_l252_252615

theorem product_of_possible_values_N 
  (L M : ℤ) 
  (h1 : M = L + N) 
  (h2 : M - 7 = L + N - 7)
  (h3 : L + 5 = L + 5)
  (h4 : |(L + N - 7) - (L + 5)| = 4) : 
  N = 128 := 
  sorry

end product_of_possible_values_N_l252_252615


namespace age_ratio_is_4_over_3_l252_252557

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end age_ratio_is_4_over_3_l252_252557


namespace bobby_initial_blocks_l252_252000

variable (b : ℕ)

theorem bobby_initial_blocks
  (h : b + 6 = 8) : b = 2 := by
  sorry

end bobby_initial_blocks_l252_252000


namespace negation_of_proposition_l252_252982

theorem negation_of_proposition:
  (∀ x : ℝ, x ≥ 0 → x - 2 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) := 
sorry

end negation_of_proposition_l252_252982


namespace quadratic_negative_roots_prob_eq_two_thirds_l252_252089

noncomputable def quadratic_negative_roots_probability : ℝ :=
∫ p in set.Icc (0 : ℝ) 5, if (p ∈ set.Icc (2/3) 1 ∪ set.Ici 2) then 1 else 0 / (5 - 0)

theorem quadratic_negative_roots_prob_eq_two_thirds :
  quadratic_negative_roots_probability = 2 / 3 := 
sorry

end quadratic_negative_roots_prob_eq_two_thirds_l252_252089


namespace rounding_estimate_lt_exact_l252_252862

variable (a b c a' b' c' : ℕ)

theorem rounding_estimate_lt_exact (ha : a' ≤ a) (hb : b' ≥ b) (hc : c' ≤ c) (hb_pos : b > 0) (hb'_pos : b' > 0) :
  (a':ℚ) / (b':ℚ) + (c':ℚ) < (a:ℚ) / (b:ℚ) + (c:ℚ) :=
sorry

end rounding_estimate_lt_exact_l252_252862


namespace other_asymptote_of_hyperbola_l252_252534

theorem other_asymptote_of_hyperbola (a b : ℝ) :
  (∀ x : ℝ, a * x + b = 2 * x) →
  (∀ p : ℝ × ℝ, (p.1 = 3)) →
  ∀ (c : ℝ × ℝ), (c.1 = 3 ∧ c.2 = 6) ->
  ∃ (m : ℝ), m = -1/2 ∧ (∀ x, c.2 = -1/2 * x + 15/2) :=
by
  sorry

end other_asymptote_of_hyperbola_l252_252534


namespace find_special_numbers_l252_252963

theorem find_special_numbers (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) :=
by
  sorry

end find_special_numbers_l252_252963


namespace min_value_of_expression_l252_252340

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) :
  2 * a + b ≥ 4 * Real.sqrt 2 - 3 := 
sorry

end min_value_of_expression_l252_252340


namespace complex_quadrant_l252_252342

open Complex

-- Let complex number i be the imaginary unit
noncomputable def purely_imaginary (z : ℂ) : Prop := 
  z.re = 0

theorem complex_quadrant (z : ℂ) (a : ℂ) (hz : purely_imaginary z) (h : (2 + I) * z = 1 + a * I ^ 3) :
  (a + z).re > 0 ∧ (a + z).im < 0 :=
by 
  sorry

end complex_quadrant_l252_252342


namespace determine_a_l252_252202

theorem determine_a (a : ℝ) (h : 2 * (-1) + a = 3) : a = 5 := sorry

end determine_a_l252_252202


namespace binomial_identity_l252_252247

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) :=
sorry

end binomial_identity_l252_252247


namespace expression_divisible_by_7_l252_252033

theorem expression_divisible_by_7 (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ m : ℤ, 3^(6*n-1) - k * 2^(3*n-2) + 1 = 7 * m) ↔ ∃ m' : ℤ, k = 7 * m' + 3 := 
by
  sorry

end expression_divisible_by_7_l252_252033


namespace product_of_repeating_decimal_l252_252025

theorem product_of_repeating_decimal (x : ℚ) (h : x = 1 / 3) : (x * 8) = 8 / 3 := by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l252_252025


namespace complex_modulus_product_l252_252452

noncomputable def z1 : ℂ := 4 - 3 * Complex.I
noncomputable def z2 : ℂ := 4 + 3 * Complex.I

theorem complex_modulus_product : Complex.abs z1 * Complex.abs z2 = 25 := by 
  sorry

end complex_modulus_product_l252_252452


namespace home_run_difference_l252_252304

def hank_aaron_home_runs : ℕ := 755
def dave_winfield_home_runs : ℕ := 465

theorem home_run_difference :
  2 * dave_winfield_home_runs - hank_aaron_home_runs = 175 := by
  sorry

end home_run_difference_l252_252304


namespace minimum_bailing_rate_l252_252735

-- Conditions
def distance_to_shore : ℝ := 2 -- miles
def rowing_speed : ℝ := 3 -- miles per hour
def water_intake_rate : ℝ := 15 -- gallons per minute
def max_water_capacity : ℝ := 50 -- gallons

-- Result to prove
theorem minimum_bailing_rate (r : ℝ) : 
  (distance_to_shore / rowing_speed * 60 * water_intake_rate - distance_to_shore / rowing_speed * 60 * r) ≤ max_water_capacity →
  r ≥ 13.75 :=
by
  sorry

end minimum_bailing_rate_l252_252735


namespace largest_prime_divisor_of_36_squared_plus_49_squared_l252_252329

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end largest_prime_divisor_of_36_squared_plus_49_squared_l252_252329


namespace chameleons_changed_color_l252_252230

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end chameleons_changed_color_l252_252230


namespace decreasing_power_function_l252_252103

theorem decreasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(m^2 + m - 1) < (m^2 - m - 1) * (x + 1) ^ (m^2 + m - 1)) →
  m = -1 :=
sorry

end decreasing_power_function_l252_252103


namespace find_values_of_a_and_c_l252_252405

theorem find_values_of_a_and_c
  (a c : ℝ)
  (h1 : ∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ a * x^2 + 5 * x + c > 0) :
  a = -6 ∧ c = -1 :=
by
  sorry

end find_values_of_a_and_c_l252_252405


namespace atomic_weight_of_Calcium_l252_252644

/-- Given definitions -/
def molecular_weight_CaOH₂ : ℕ := 74
def atomic_weight_O : ℕ := 16
def atomic_weight_H : ℕ := 1

/-- Given conditions -/
def total_weight_O_H : ℕ := 2 * atomic_weight_O + 2 * atomic_weight_H

/-- Problem statement -/
theorem atomic_weight_of_Calcium (H1 : molecular_weight_CaOH₂ = 74)
                                   (H2 : atomic_weight_O = 16)
                                   (H3 : atomic_weight_H = 1)
                                   (H4 : total_weight_O_H = 2 * atomic_weight_O + 2 * atomic_weight_H) :
  74 - (2 * 16 + 2 * 1) = 40 :=
by {
  sorry
}

end atomic_weight_of_Calcium_l252_252644


namespace smallest_number_four_solutions_sum_four_squares_l252_252728

def is_sum_of_four_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2

theorem smallest_number_four_solutions_sum_four_squares :
  ∃ n : ℕ,
    is_sum_of_four_squares n ∧
    (∃ (a1 b1 c1 d1 a2 b2 c2 d2 a3 b3 c3 d3 a4 b4 c4 d4 : ℕ),
      n = a1^2 + b1^2 + c1^2 + d1^2 ∧
      n = a2^2 + b2^2 + c2^2 + d2^2 ∧
      n = a3^2 + b3^2 + c3^2 + d3^2 ∧
      n = a4^2 + b4^2 + c4^2 + d4^2 ∧
      (a1, b1, c1, d1) ≠ (a2, b2, c2, d2) ∧
      (a1, b1, c1, d1) ≠ (a3, b3, c3, d3) ∧
      (a1, b1, c1, d1) ≠ (a4, b4, c4, d4) ∧
      (a2, b2, c2, d2) ≠ (a3, b3, c3, d3) ∧
      (a2, b2, c2, d2) ≠ (a4, b4, c4, d4) ∧
      (a3, b3, c3, d3) ≠ (a4, b4, c4, d4)) ∧
    (∀ m : ℕ,
      m < 635318657 →
      ¬ (∃ (a5 b5 c5 d5 a6 b6 c6 d6 a7 b7 c7 d7 a8 b8 c8 d8 : ℕ),
        m = a5^2 + b5^2 + c5^2 + d5^2 ∧
        m = a6^2 + b6^2 + c6^2 + d6^2 ∧
        m = a7^2 + b7^2 + c7^2 + d7^2 ∧
        m = a8^2 + b8^2 + c8^2 + d8^2 ∧
        (a5, b5, c5, d5) ≠ (a6, b6, c6, d6) ∧
        (a5, b5, c5, d5) ≠ (a7, b7, c7, d7) ∧
        (a5, b5, c5, d5) ≠ (a8, b8, c8, d8) ∧
        (a6, b6, c6, d6) ≠ (a7, b7, c7, d7) ∧
        (a6, b6, c6, d6) ≠ (a8, b8, c8, d8) ∧
        (a7, b7, c7, d7) ≠ (a8, b8, c8, d8))) :=
  sorry

end smallest_number_four_solutions_sum_four_squares_l252_252728


namespace find_pairs_l252_252773

theorem find_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) ↔ (a = b) := by
  sorry

end find_pairs_l252_252773


namespace rick_savings_ratio_proof_l252_252635

-- Define the conditions
def erika_savings : ℤ := 155
def cost_of_gift : ℤ := 250
def cost_of_cake : ℤ := 25
def amount_left : ℤ := 5

-- Define the total amount they have together
def total_amount : ℤ := cost_of_gift + cost_of_cake - amount_left

-- Define Rick's savings based on the conditions
def rick_savings : ℤ := total_amount - erika_savings

-- Define the ratio of Rick's savings to the cost of the gift
def rick_gift_ratio : ℚ := rick_savings / cost_of_gift

-- Prove the ratio is 23/50
theorem rick_savings_ratio_proof : rick_gift_ratio = 23 / 50 :=
  by
    have h1 : total_amount = 270 := by sorry
    have h2 : rick_savings = 115 := by sorry
    have h3 : rick_gift_ratio = 23 / 50 := by sorry
    exact h3

end rick_savings_ratio_proof_l252_252635


namespace binomial_evaluation_l252_252953

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l252_252953


namespace proof_problem_l252_252831

noncomputable def M : ℕ := 50
noncomputable def T : ℕ := M + Nat.div M 10
noncomputable def W : ℕ := 2 * (M + T)
noncomputable def Th : ℕ := W / 2
noncomputable def total_T_T_W_Th : ℕ := T + W + Th
noncomputable def total_M_T_W_Th : ℕ := M + total_T_T_W_Th
noncomputable def F_S_sun : ℕ := Nat.div (450 - total_M_T_W_Th) 3
noncomputable def car_tolls : ℕ := 150 * 2
noncomputable def bus_tolls : ℕ := 150 * 5
noncomputable def truck_tolls : ℕ := 150 * 10
noncomputable def total_toll : ℕ := car_tolls + bus_tolls + truck_tolls

theorem proof_problem :
  (total_T_T_W_Th = 370) ∧
  (F_S_sun = 10) ∧
  (total_toll = 2550) := by
  sorry

end proof_problem_l252_252831


namespace arithmetic_expression_evaluation_l252_252833

theorem arithmetic_expression_evaluation : 
  ∃ (a b c d e f : Float),
  a - b * c / d + e = 0 ∧
  a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 ∧ e = 1 := sorry

end arithmetic_expression_evaluation_l252_252833


namespace age_ratio_is_4_over_3_l252_252558

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end age_ratio_is_4_over_3_l252_252558


namespace min_balls_for_color_15_l252_252143

theorem min_balls_for_color_15
  (red green yellow blue white black : ℕ)
  (h_red : red = 28)
  (h_green : green = 20)
  (h_yellow : yellow = 19)
  (h_blue : blue = 13)
  (h_white : white = 11)
  (h_black : black = 9) :
  ∃ n, n = 76 ∧ ∀ balls_drawn, balls_drawn = n →
  ∃ color, 
    (color = "red" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= red) ∨
    (color = "green" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= green) ∨
    (color = "yellow" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= yellow) ∨
    (color = "blue" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= blue) ∨
    (color = "white" ∧ balls_drawn >= 15 ∧ balls_drawn <= white) ∨
    (color = "black" ∧ balls_drawn ≥ 15 ∧ balls_drawn <= black) := 
sorry

end min_balls_for_color_15_l252_252143


namespace find_triplet_x_y_z_l252_252460

theorem find_triplet_x_y_z :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + 1 / (y + 1 / z : ℝ) = (10 : ℝ) / 7) ∧ (x = 1 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_triplet_x_y_z_l252_252460


namespace arithmetic_progression_sum_15_terms_l252_252562

def arithmetic_progression_sum (a₁ d : ℚ) : ℚ :=
  15 * (2 * a₁ + (15 - 1) * d) / 2

def am_prog3_and_9_sum_and_product (a₁ d : ℚ) : Prop :=
  (a₁ + 2 * d) + (a₁ + 8 * d) = 6 ∧ (a₁ + 2 * d) * (a₁ + 8 * d) = 135 / 16

theorem arithmetic_progression_sum_15_terms (a₁ d : ℚ)
  (h : am_prog3_and_9_sum_and_product a₁ d) :
  arithmetic_progression_sum a₁ d = 37.5 ∨ arithmetic_progression_sum a₁ d = 52.5 :=
sorry

end arithmetic_progression_sum_15_terms_l252_252562


namespace maria_should_buy_more_l252_252693

-- Define the conditions as assumptions.
variables (needs total_cartons : ℕ) (strawberries blueberries : ℕ)

-- Specify the given conditions.
def maria_conditions (needs total_cartons strawberries blueberries : ℕ) : Prop :=
  needs = 21 ∧ strawberries = 4 ∧ blueberries = 8 ∧ total_cartons = strawberries + blueberries

-- State the theorem to be proven.
theorem maria_should_buy_more
  (needs total_cartons : ℕ) (strawberries blueberries : ℕ)
  (h : maria_conditions needs total_cartons strawberries blueberries) :
  needs - total_cartons = 9 :=
sorry

end maria_should_buy_more_l252_252693


namespace derivative_at_x₀_l252_252704

-- Define the function y = (x - 2)^2
def f (x : ℝ) : ℝ := (x - 2) ^ 2

-- Define the point of interest
def x₀ : ℝ := 1

-- State the problem and the correct answer
theorem derivative_at_x₀ : (deriv f x₀) = -2 := by
  sorry

end derivative_at_x₀_l252_252704


namespace chameleon_color_change_l252_252225

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l252_252225


namespace silverware_probability_l252_252681

-- Defining the number of each type of silverware
def num_forks : ℕ := 8
def num_spoons : ℕ := 10
def num_knives : ℕ := 4
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def num_remove : ℕ := 4

-- Proving the probability calculation
theorem silverware_probability :
  -- Calculation of the total number of ways to choose 4 pieces from 22
  let total_ways := Nat.choose total_silverware num_remove
  -- Calculation of ways to choose 2 forks from 8
  let ways_to_choose_forks := Nat.choose num_forks 2
  -- Calculation of ways to choose 1 spoon from 10
  let ways_to_choose_spoon := Nat.choose num_spoons 1
  -- Calculation of ways to choose 1 knife from 4
  let ways_to_choose_knife := Nat.choose num_knives 1
  -- Calculation of the number of favorable outcomes
  let favorable_outcomes := ways_to_choose_forks * ways_to_choose_spoon * ways_to_choose_knife
  -- Probability in simplified form
  let probability := (favorable_outcomes : ℚ) / total_ways
  probability = (32 : ℚ) / 209 :=
by
  sorry

end silverware_probability_l252_252681


namespace group1_calculation_group2_calculation_l252_252004

theorem group1_calculation : 9 / 3 * (9 - 1) = 24 := by
  sorry

theorem group2_calculation : 7 * (3 + 3 / 7) = 24 := by
  sorry

end group1_calculation_group2_calculation_l252_252004


namespace maria_bottles_count_l252_252849

-- Definitions from the given conditions
def b_initial : ℕ := 23
def d : ℕ := 12
def g : ℕ := 5
def b : ℕ := 65

-- Definition of the question based on conditions
def b_final : ℕ := b_initial - d - g + b

-- The statement to prove the correctness of the answer
theorem maria_bottles_count : b_final = 71 := by
  -- We skip the proof for this statement
  sorry

end maria_bottles_count_l252_252849


namespace meal_serving_count_correct_l252_252894

def meals_served_correctly (total_people : ℕ) (meal_type : Type*)
  (orders : meal_type → ℕ) (correct_meals : ℕ) : ℕ :=
  -- function to count the number of ways to serve meals correctly
  sorry

theorem meal_serving_count_correct (total_people : ℕ) (meal_type : fin 3) 
  [decidable_eq meal_type]
  (orders : fin 3 → ℕ) (h_orders : orders = (λ x, 4)) :
  meals_served_correctly total_people meal_type orders 2 = 22572 :=
  begin
    have orders_correct: ∀ x, orders x = 4 := by rw h_orders,
    -- Further steps and usage of derangements would be here, 
    -- but for now we will skip to the final count.
    sorry
  end

end meal_serving_count_correct_l252_252894


namespace remaining_credit_l252_252381

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end remaining_credit_l252_252381


namespace binom_12_6_l252_252946

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l252_252946


namespace angle_between_AB_CD_l252_252967

def point := (ℝ × ℝ × ℝ)

def A : point := (-3, 0, 1)
def B : point := (2, 1, -1)
def C : point := (-2, 2, 0)
def D : point := (1, 3, 2)

noncomputable def angle_between_lines (p1 p2 p3 p4 : point) : ℝ := sorry

theorem angle_between_AB_CD :
  angle_between_lines A B C D = Real.arccos (2 * Real.sqrt 105 / 35) :=
sorry

end angle_between_AB_CD_l252_252967


namespace sufficient_but_not_necessary_condition_l252_252047

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : ∀ x : ℝ, x > a → x > 2 ∧ ¬(x > 2 → x > a)) : a > 2 :=
sorry

end sufficient_but_not_necessary_condition_l252_252047


namespace cube_volume_is_27_l252_252710

noncomputable def original_volume (s : ℝ) : ℝ := s^3
noncomputable def new_solid_volume (s : ℝ) : ℝ := (s + 2) * (s + 2) * (s - 2)

theorem cube_volume_is_27 (s : ℝ) (h : original_volume s - new_solid_volume s = 10) :
  original_volume s = 27 :=
by
  sorry

end cube_volume_is_27_l252_252710


namespace smallest_possible_S_l252_252900

/-- Define the maximum possible sum for n dice --/
def max_sum (n : ℕ) : ℕ := 6 * n

/-- Define the transformation of the dice sum when each result is transformed to 7 - d_i --/
def transformed_sum (n R : ℕ) : ℕ := 7 * n - R

/-- Determine the smallest possible S under given conditions --/
theorem smallest_possible_S :
  ∃ n : ℕ, max_sum n ≥ 2001 ∧ transformed_sum n 2001 = 337 :=
by
  -- TODO: Complete the proof
  sorry

end smallest_possible_S_l252_252900


namespace min_combined_number_of_horses_and_ponies_l252_252291

theorem min_combined_number_of_horses_and_ponies :
  ∃ P H : ℕ, H = P + 4 ∧ (∃ k : ℕ, k = (3 * P) / 10 ∧ k = 16 * (3 * P) / (16 * 10) ∧ H + P = 36) :=
sorry

end min_combined_number_of_horses_and_ponies_l252_252291


namespace team_A_minimum_workers_l252_252784

-- Define the variables and conditions for the problem.
variables (A B c : ℕ)

-- Condition 1: If team A lends 90 workers to team B, Team B will have twice as many workers as Team A.
def condition1 : Prop :=
  2 * (A - 90) = B + 90

-- Condition 2: If team B lends c workers to team A, Team A will have six times as many workers as Team B.
def condition2 : Prop :=
  A + c = 6 * (B - c)

-- Define the proof goal.
theorem team_A_minimum_workers (h1 : condition1 A B) (h2 : condition2 A B c) : 
  153 ≤ A :=
sorry

end team_A_minimum_workers_l252_252784


namespace binomial_evaluation_l252_252951

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l252_252951


namespace jack_needs_more_money_l252_252683

/--
Jack is a soccer player. He needs to buy two pairs of socks, a pair of soccer shoes, a soccer ball, and a sports bag.
Each pair of socks costs $12.75, the shoes cost $145, the soccer ball costs $38, and the sports bag costs $47.
Jack has a 5% discount coupon for the shoes and a 10% discount coupon for the sports bag.
He currently has $25. How much more money does Jack need to buy all the items?
-/
theorem jack_needs_more_money :
  let socks_cost : ℝ := 12.75
  let shoes_cost : ℝ := 145
  let ball_cost : ℝ := 38
  let bag_cost : ℝ := 47
  let shoes_discount : ℝ := 0.05
  let bag_discount : ℝ := 0.10
  let money_jack_has : ℝ := 25
  let total_cost := 2 * socks_cost + (shoes_cost - shoes_cost * shoes_discount) + ball_cost + (bag_cost - bag_cost * bag_discount)
  total_cost - money_jack_has = 218.55 :=
by
  sorry

end jack_needs_more_money_l252_252683


namespace sales_tax_difference_l252_252876

theorem sales_tax_difference:
  let original_price := 50 
  let discount_rate := 0.10 
  let sales_tax_rate_1 := 0.08
  let sales_tax_rate_2 := 0.075 
  let discounted_price := original_price * (1 - discount_rate) 
  let sales_tax_1 := discounted_price * sales_tax_rate_1 
  let sales_tax_2 := discounted_price * sales_tax_rate_2 
  sales_tax_1 - sales_tax_2 = 0.225 := by
  sorry

end sales_tax_difference_l252_252876


namespace smallest_n_l252_252541

noncomputable def smallest_positive_integer (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : ℕ :=
  if 3 % 7 = 0 then 7 else 7

theorem smallest_n (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : smallest_positive_integer x y h1 h2 = 7 := 
  by
  admit

end smallest_n_l252_252541


namespace hendricks_payment_l252_252045

variable (Hendricks Gerald : ℝ)
variable (less_percent : ℝ) (amount_paid : ℝ)

theorem hendricks_payment (h g : ℝ) (h_less_g : h = g * (1 - less_percent)) (g_val : g = amount_paid) (less_percent_val : less_percent = 0.2) (amount_paid_val: amount_paid = 250) :
h = 200 :=
by
  sorry

end hendricks_payment_l252_252045


namespace cannot_have_N_less_than_K_l252_252662

theorem cannot_have_N_less_than_K (K N : ℕ) (hK : K > 2) (cards : Fin N → ℕ) (h_cards : ∀ i, cards i > 0) :
  ¬ (N < K) :=
sorry

end cannot_have_N_less_than_K_l252_252662


namespace functional_eq_solutions_l252_252170

-- Define the conditions for the problem
def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

-- Define the two solutions to be proven correct
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := |x|

-- State the main theorem to be proven
theorem functional_eq_solutions (f : ℝ → ℝ) (h : func_equation f) : f = f1 ∨ f = f2 :=
sorry

end functional_eq_solutions_l252_252170


namespace problem1_problem2_l252_252587

-- Problem 1: Calculate the expression 
theorem problem1 :
  (∛8) + | -5 | + (-1 : ℤ)^2023 = 6 :=
  sorry

-- Problem 2: Find the equation of a line passing through two points
theorem problem2 (k b : ℚ) :
  (∀ x, y = k*x + b) →
  (0, 1) ∈ λ x, y = k*x + b ∧ 
  (2, 5) ∈ λ x, y = k*x + b →
  k = 2 ∧ b = 1 :=
  sorry

end problem1_problem2_l252_252587


namespace ab_bc_ca_lt_quarter_l252_252473

theorem ab_bc_ca_lt_quarter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) :
  (a * b)^(5/4) + (b * c)^(5/4) + (c * a)^(5/4) < 1/4 :=
sorry

end ab_bc_ca_lt_quarter_l252_252473


namespace perfect_cubes_between_200_and_1600_l252_252490

theorem perfect_cubes_between_200_and_1600 : 
  ∃ (count : ℕ), count = (finset.filter (λ n, 200 ≤ n^3 ∧ n^3 ≤ 1600) (finset.range 50)).card := 
begin
  use 6,
  sorry,
end

end perfect_cubes_between_200_and_1600_l252_252490


namespace find_side_length_b_of_triangle_l252_252787

/-- Given a triangle ABC with angles satisfying A:B:C = 1:2:3, opposite sides a, b, and c, and
    given a = 1, c = 2, find the length of side b. -/
noncomputable def triangle_side_length_b : ℝ := 
  let A := 1 * Real.pi / 6 in
  let B := 2 * Real.pi / 6 in
  let C := 3 * Real.pi / 6 in
  let a := 1 in 
  let c := 2 in
  Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)
  
theorem find_side_length_b_of_triangle : triangle_side_length_b = Real.sqrt 3 := 
by {
  sorry
}

end find_side_length_b_of_triangle_l252_252787


namespace chameleons_color_change_l252_252222

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l252_252222


namespace equilibrium_temperature_l252_252606

theorem equilibrium_temperature 
  (c_B : ℝ) (c_m : ℝ)
  (m_B : ℝ) (m_m : ℝ)
  (T₁ : ℝ) (T_eq₁ : ℝ) (T_metal : ℝ) 
  (T_eq₂ : ℝ)
  (h₁ : T₁ = 80)
  (h₂ : T_eq₁ = 60)
  (h₃ : T_metal = 20)
  (h₄ : T₂ = 50)
  (h_ratio : c_B * m_B = 2 * c_m * m_m) :
  T_eq₂ = 50 :=
by
  sorry

end equilibrium_temperature_l252_252606


namespace solve_equation_l252_252257

theorem solve_equation : ∀ x : ℝ, (x + 1 - 2 * (x - 1) = 1 - 3 * x) → x = 0 := 
by
  intros x h
  sorry

end solve_equation_l252_252257


namespace binom_12_6_eq_924_l252_252940

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l252_252940


namespace identity_function_uniq_l252_252766

theorem identity_function_uniq (f g h : ℝ → ℝ)
    (hg : ∀ x, g x = x + 1)
    (hh : ∀ x, h x = x^2)
    (H1 : ∀ x, f (g x) = g (f x))
    (H2 : ∀ x, f (h x) = h (f x)) :
  ∀ x, f x = x :=
by
  sorry

end identity_function_uniq_l252_252766


namespace find_person_age_l252_252290

theorem find_person_age : ∃ x : ℕ, 4 * (x + 4) - 4 * (x - 4) = x ∧ x = 32 := by
  sorry

end find_person_age_l252_252290


namespace geometric_series_sum_l252_252164

theorem geometric_series_sum :
  let a := (1/2 : ℚ)
  let r := (-1/3 : ℚ)
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 547 / 1458 :=
by
  sorry

end geometric_series_sum_l252_252164


namespace xiao_ding_distance_l252_252277

variable (x y z w : ℕ)

theorem xiao_ding_distance (h1 : x = 4 * y)
                          (h2 : z = x / 2 + 20)
                          (h3 : w = 2 * z - 15)
                          (h4 : x + y + z + w = 705) : 
                          y = 60 := 
sorry

end xiao_ding_distance_l252_252277


namespace total_cost_of_cultivating_field_l252_252544

theorem total_cost_of_cultivating_field 
  (base height : ℕ) 
  (cost_per_hectare : ℝ) 
  (base_eq: base = 3 * height) 
  (height_eq: height = 300) 
  (cost_eq: cost_per_hectare = 24.68) 
  : (1/2 : ℝ) * base * height / 10000 * cost_per_hectare = 333.18 :=
by
  sorry

end total_cost_of_cultivating_field_l252_252544


namespace stratified_sampling_probability_l252_252996

open Finset Nat

noncomputable def combin (n k : ℕ) : ℕ := choose n k

theorem stratified_sampling_probability :
  let total_balls := 40
  let red_balls := 16
  let blue_balls := 12
  let white_balls := 8
  let yellow_balls := 4
  let n_draw := 10
  let red_draw := 4
  let blue_draw := 3
  let white_draw := 2
  let yellow_draw := 1
  
  combin yellow_balls yellow_draw * combin white_balls white_draw * combin blue_balls blue_draw * combin red_balls red_draw = combin total_balls n_draw :=
sorry

end stratified_sampling_probability_l252_252996


namespace water_consumption_150_litres_per_household_4_months_6000_litres_l252_252504

def number_of_households (household_water_use_per_month : ℕ) (water_supply : ℕ) (duration_months : ℕ) : ℕ :=
  water_supply / (household_water_use_per_month * duration_months)

theorem water_consumption_150_litres_per_household_4_months_6000_litres : 
  number_of_households 150 6000 4 = 10 :=
by
  sorry

end water_consumption_150_litres_per_household_4_months_6000_litres_l252_252504


namespace common_difference_is_two_l252_252238

-- Define the properties and conditions.
variables {a : ℕ → ℝ} {d : ℝ}

-- An arithmetic sequence definition.
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement to be proved.
theorem common_difference_is_two (h1 : a 1 + a 5 = 10) (h2 : a 4 = 7) (h3 : arithmetic_sequence a d) : 
  d = 2 :=
sorry

end common_difference_is_two_l252_252238


namespace find_number_l252_252260

-- Definitions based on the given conditions
def area (s : ℝ) := s^2
def perimeter (s : ℝ) := 4 * s
def given_perimeter : ℝ := 36
def equation (s : ℝ) (n : ℝ) := 5 * area s = 10 * perimeter s + n

-- Statement of the problem
theorem find_number :
  ∃ n : ℝ, equation (given_perimeter / 4) n ∧ n = 45 :=
by
  sorry

end find_number_l252_252260


namespace student_made_mistake_l252_252755

theorem student_made_mistake (AB CD MLNKT : ℕ) (h1 : 10 ≤ AB ∧ AB ≤ 99) (h2 : 10 ≤ CD ∧ CD ≤ 99) (h3 : 10000 ≤ MLNKT ∧ MLNKT < 100000) : AB * CD ≠ MLNKT :=
by {
  sorry
}

end student_made_mistake_l252_252755


namespace sum_of_first_10_bn_l252_252709

def a (n : ℕ) : ℚ :=
  (2 / 5) * n + (3 / 5)

def b (n : ℕ) : ℤ :=
  ⌊a n⌋

def sum_first_10_b : ℤ :=
  (b 1) + (b 2) + (b 3) + (b 4) + (b 5) + (b 6) + (b 7) + (b 8) + (b 9) + (b 10)

theorem sum_of_first_10_bn : sum_first_10_b = 24 :=
  by sorry

end sum_of_first_10_bn_l252_252709


namespace square_sum_zero_real_variables_l252_252410

theorem square_sum_zero_real_variables (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end square_sum_zero_real_variables_l252_252410


namespace point_B_l252_252512

-- Define constants for perimeter and speed factor
def perimeter : ℕ := 24
def speed_factor : ℕ := 2

-- Define the speeds of Jane and Hector
def hector_speed (s : ℕ) : ℕ := s
def jane_speed (s : ℕ) : ℕ := speed_factor * s

-- Define the times until they meet
def time_until_meeting (s : ℕ) : ℚ := perimeter / (hector_speed s + jane_speed s)

-- Distances walked by Hector and Jane upon meeting
noncomputable def hector_distance (s : ℕ) : ℚ := hector_speed s * time_until_meeting s
noncomputable def jane_distance (s : ℕ) : ℚ := jane_speed s * time_until_meeting s

-- Map the perimeter position to a point
def position_on_track (d : ℚ) : ℚ := d % perimeter

-- When they meet
theorem point_B (s : ℕ) (h₀ : 0 < s) : position_on_track (hector_distance s) = position_on_track (jane_distance s) → 
                          position_on_track (hector_distance s) = 8 := 
by 
  sorry

end point_B_l252_252512


namespace expected_yield_of_carrots_l252_252852

def steps_to_feet (steps : ℕ) (step_size : ℕ) : ℕ :=
  steps * step_size

def garden_area (length width : ℕ) : ℕ :=
  length * width

def yield_of_carrots (area : ℕ) (yield_rate : ℚ) : ℚ :=
  area * yield_rate

theorem expected_yield_of_carrots :
  steps_to_feet 18 3 * steps_to_feet 25 3 = 4050 →
  yield_of_carrots 4050 (3 / 4) = 3037.5 :=
by
  sorry

end expected_yield_of_carrots_l252_252852


namespace kenya_peanuts_eq_133_l252_252520

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l252_252520


namespace incorrect_step_l252_252976

-- Given conditions
variables {a b : ℝ} (hab : a < b)

-- Proof statement of the incorrect step ③
theorem incorrect_step : ¬ (2 * (a - b) ^ 2 < (a - b) ^ 2) :=
by sorry

end incorrect_step_l252_252976


namespace inscribed_square_neq_five_l252_252373

theorem inscribed_square_neq_five (a b : ℝ) 
  (h1 : a - b = 1)
  (h2 : a * b = 1)
  (h3 : a + b = Real.sqrt 5) : a^2 + b^2 ≠ 5 :=
by sorry

end inscribed_square_neq_five_l252_252373


namespace problem1_problem2_l252_252596

-- Definition for Problem 1
def cube_root_8 : ℝ := 2
def abs_neg5 : ℝ := 5
def pow_neg1_2023 : ℝ := -1

theorem problem1 : cube_root_8 + abs_neg5 + pow_neg1_2023 = 6 := by
  sorry

-- Definitions for Problem 2
structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := { x := 0, y := 1 }
def point2 : Point := { x := 2, y := 5 }

def linear_function (k b x : ℝ) : ℝ := k * x + b

def is_linear_function (p1 p2 : Point) (k b : ℝ) : Prop :=
  p1.y = linear_function k b p1.x ∧ p2.y = linear_function k b p2.x

theorem problem2 : ∃ k b : ℝ, is_linear_function point1 point2 k b ∧ (linear_function k b x = 2 * x + 1) := by
  sorry

end problem1_problem2_l252_252596


namespace katherine_savings_multiple_l252_252448

variable (A K : ℕ)

theorem katherine_savings_multiple
  (h1 : A + K = 750)
  (h2 : A - 150 = 1 / 3 * K) :
  2 * K / A = 3 :=
sorry

end katherine_savings_multiple_l252_252448


namespace vertex_difference_l252_252400

theorem vertex_difference (n m : ℝ) : 
  ∀ x : ℝ, (∀ x, -x^2 + 2*x + n = -((x - m)^2) + 1) → m - n = 1 := 
by 
  sorry

end vertex_difference_l252_252400


namespace problem1_problem2_l252_252598

-- Define the conditions for Problem (1)
def cuberoot_8 := real.cbrt 8
def abs_neg_5 := abs (-5)
def pow_neg_1 := (-1 : ℤ) ^ 2023

-- Define the proof problem for Problem (1)
theorem problem1 : cuberoot_8 + abs_neg_5 + (pow_neg_1 : ℝ) = 6 := 
by sorry

-- Define the conditions for Problem (2)
def point1 := (0, 1)
def point2 := (2, 5)

-- Define the equation of the line
def line (k b : ℝ) (x : ℝ) := k * x + b

-- Prove that the line passing through the given points has the form y = 2x + 1
theorem problem2 (k b : ℝ) : 
  line k b 0 = (point1.snd : ℝ) ∧ line k b 2 = (point2.snd : ℝ) → 
  line 2 1 = 2 * point2.fst + 1 := 
by sorry

end problem1_problem2_l252_252598


namespace reciprocal_inequality_l252_252196

theorem reciprocal_inequality {a b c : ℝ} (hab : a < b) (hbc : b < c) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_l252_252196


namespace lowest_possible_price_l252_252137

theorem lowest_possible_price
  (regular_discount_rate : ℚ)
  (sale_discount_rate : ℚ)
  (manufacturer_price : ℚ)
  (H1 : regular_discount_rate = 0.30)
  (H2 : sale_discount_rate = 0.20)
  (H3 : manufacturer_price = 35) :
  (manufacturer_price * (1 - regular_discount_rate) * (1 - sale_discount_rate)) = 19.60 := by
  sorry

end lowest_possible_price_l252_252137


namespace bottles_left_l252_252116

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l252_252116


namespace exists_x_gt_zero_negation_l252_252549

theorem exists_x_gt_zero_negation :
  (∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry  -- Proof goes here

end exists_x_gt_zero_negation_l252_252549


namespace remainder_when_xy_div_by_22_l252_252529

theorem remainder_when_xy_div_by_22
  (x y : ℤ)
  (h1 : x % 126 = 37)
  (h2 : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
  sorry

end remainder_when_xy_div_by_22_l252_252529


namespace find_number_l252_252049

theorem find_number (x : ℤ) (h : 45 - (28 - (37 - (x - 18))) = 57) : x = 15 :=
by
  sorry

end find_number_l252_252049


namespace valuing_fraction_l252_252974

variable {x y : ℚ}

theorem valuing_fraction (h : x / y = 1 / 2) : (x - y) / (x + y) = -1 / 3 :=
by
  sorry

end valuing_fraction_l252_252974


namespace total_tickets_spent_l252_252616

def tickets_spent_on_hat : ℕ := 2
def tickets_spent_on_stuffed_animal : ℕ := 10
def tickets_spent_on_yoyo : ℕ := 2

theorem total_tickets_spent :
  tickets_spent_on_hat + tickets_spent_on_stuffed_animal + tickets_spent_on_yoyo = 14 := by
  sorry

end total_tickets_spent_l252_252616


namespace race_speed_ratio_l252_252421

theorem race_speed_ratio (L v_a v_b : ℝ) (h1 : v_a = v_b / 0.84375) :
  v_a / v_b = 32 / 27 :=
by sorry

end race_speed_ratio_l252_252421


namespace kenya_peanuts_l252_252517

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l252_252517


namespace inequality_correct_l252_252492

theorem inequality_correct {a b : ℝ} (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : a < a * b ^ 2 ∧ a * b ^ 2 < a * b := 
sorry

end inequality_correct_l252_252492


namespace ratio_of_unit_prices_l252_252002

def volume_y (v : ℝ) : ℝ := v
def price_y (p : ℝ) : ℝ := p
def volume_x (v : ℝ) : ℝ := 1.3 * v
def price_x (p : ℝ) : ℝ := 0.8 * p

theorem ratio_of_unit_prices (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (0.8 * p / (1.3 * v)) / (p / v) = 8 / 13 :=
by 
  sorry

end ratio_of_unit_prices_l252_252002


namespace right_triangle_shorter_leg_l252_252827

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l252_252827


namespace remaining_balance_on_phone_card_l252_252379

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end remaining_balance_on_phone_card_l252_252379


namespace min_value_fraction_sum_l252_252341

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_eq : 1 = 2 * a + b) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_sum_l252_252341


namespace james_take_home_pay_l252_252837

theorem james_take_home_pay :
  let main_hourly_rate := 20
  let second_hourly_rate := main_hourly_rate - (main_hourly_rate * 0.20)
  let main_hours := 30
  let second_hours := main_hours / 2
  let side_gig_earnings := 100 * 2
  let overtime_hours := 5
  let overtime_rate := main_hourly_rate * 1.5
  let irs_tax_rate := 0.18
  let state_tax_rate := 0.05
  
  -- Main job earnings
  let main_regular_earnings := main_hours * main_hourly_rate
  let main_overtime_earnings := overtime_hours * overtime_rate
  let main_total_earnings := main_regular_earnings + main_overtime_earnings
  
  -- Second job earnings
  let second_total_earnings := second_hours * second_hourly_rate
  
  -- Total earnings before taxes
  let total_earnings := main_total_earnings + second_total_earnings + side_gig_earnings
  
  -- Tax calculations
  let federal_tax := total_earnings * irs_tax_rate
  let state_tax := total_earnings * state_tax_rate
  let total_taxes := federal_tax + state_tax

  -- Total take home pay after taxes
  let take_home_pay := total_earnings - total_taxes

  take_home_pay = 916.30 := 
sorry

end james_take_home_pay_l252_252837


namespace problem1_problem2_l252_252588

-- Problem 1: Calculate the expression 
theorem problem1 :
  (∛8) + | -5 | + (-1 : ℤ)^2023 = 6 :=
  sorry

-- Problem 2: Find the equation of a line passing through two points
theorem problem2 (k b : ℚ) :
  (∀ x, y = k*x + b) →
  (0, 1) ∈ λ x, y = k*x + b ∧ 
  (2, 5) ∈ λ x, y = k*x + b →
  k = 2 ∧ b = 1 :=
  sorry

end problem1_problem2_l252_252588


namespace workers_are_280_women_l252_252680

variables (W : ℕ) 
          (workers_without_retirement_plan : ℕ := W / 3)
          (women_without_retirement_plan : ℕ := (workers_without_retirement_plan * 1) / 10)
          (workers_with_retirement_plan : ℕ := W * 2 / 3)
          (men_with_retirement_plan : ℕ := (workers_with_retirement_plan * 4) / 10)
          (total_men : ℕ := (workers_without_retirement_plan * 9) / 30)
          (total_workers := total_men / (9 / 30))
          (number_of_women : ℕ := total_workers - 120)

theorem workers_are_280_women : total_workers = 400 ∧ number_of_women = 280 :=
by sorry

end workers_are_280_women_l252_252680


namespace product_of_repeating_decimal_l252_252024

-- Define the repeating decimal 0.3
def repeating_decimal : ℚ := 1 / 3
-- Define the question
def product (a b : ℚ) := a * b

-- State the theorem to be proved
theorem product_of_repeating_decimal :
  product repeating_decimal 8 = 8 / 3 :=
sorry

end product_of_repeating_decimal_l252_252024


namespace ratio_of_speeds_l252_252458

-- Definitions based on the conditions provided
def timeEddy : ℝ := 3 -- hours
def distanceEddy : ℝ := 510 -- km
def timeFreddy : ℝ := 4 -- hours
def distanceFreddy : ℝ := 300 -- km

-- Helper definitions to compute average speeds
def averageSpeed (distance : ℝ) (time : ℝ) : ℝ := distance / time
def ratio (a : ℝ) (b : ℝ) : Rat := Rat.mk a b

theorem ratio_of_speeds :
  ratio (averageSpeed distanceEddy timeEddy) (averageSpeed distanceFreddy timeFreddy) = Rat.mk 34 15 :=
by
  sorry

end ratio_of_speeds_l252_252458


namespace right_triangle_shorter_leg_l252_252829

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l252_252829


namespace problem1_problem2_l252_252577

-- Proof problem (1)
theorem problem1 : real.cbrt 8 + abs (-5) + (-1 : ℝ) ^ 2023 = 6 := by
  sorry

-- Proof problem (2)
theorem problem2 (k b : ℝ) 
  (h1 : 1 = b)
  (h2 : 5 = 2 * k + b) :
  (∀ x, ∃ y, y = k * x + b) → (∀ x, ∃ y, y = 2 * x + 1) :=
by 
  sorry

end problem1_problem2_l252_252577


namespace existence_of_solution_largest_unsolvable_n_l252_252321

-- Definitions based on the conditions provided in the problem
def equation (x y z n : ℕ) : Prop := 28 * x + 30 * y + 31 * z = n

-- There exist positive integers x, y, z such that 28x + 30y + 31z = 365
theorem existence_of_solution : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z 365 :=
by
  sorry

-- The largest positive integer n such that 28x + 30y + 31z = n cannot be solved in positive integers x, y, z is 370
theorem largest_unsolvable_n : ∀ (n : ℕ), (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → n ≠ 370) → ∀ (n' : ℕ), n' > 370 → (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n') :=
by
  sorry

end existence_of_solution_largest_unsolvable_n_l252_252321


namespace greg_savings_l252_252349

-- Definitions based on the conditions
def scooter_cost : ℕ := 90
def money_needed : ℕ := 33

-- The theorem to prove
theorem greg_savings : scooter_cost - money_needed = 57 := 
by
  -- sorry is used to skip the actual mathematical proof steps
  sorry

end greg_savings_l252_252349


namespace infinitely_many_perfect_squares_of_form_l252_252071

theorem infinitely_many_perfect_squares_of_form (k : ℕ) (h : k > 0) : 
  ∃ (n : ℕ), ∃ m : ℕ, n * 2^k - 7 = m^2 :=
by
  sorry

end infinitely_many_perfect_squares_of_form_l252_252071


namespace find_a_l252_252641

theorem find_a (a : ℝ) (h_pos : a > 0) :
  (∀ x y : ℤ, x^2 - a * (x : ℝ) + 4 * a = 0) →
  a = 25 ∨ a = 18 ∨ a = 16 :=
by
  sorry

end find_a_l252_252641


namespace upper_bound_for_k_squared_l252_252677

theorem upper_bound_for_k_squared :
  (∃ (k : ℤ), k^2 > 121 ∧ ∀ m : ℤ, (m^2 > 121 ∧ m^2 < 323 → m = k + 1)) →
  (k ≤ 17) → (18^2 > 323) := 
by 
  sorry

end upper_bound_for_k_squared_l252_252677


namespace suraj_next_innings_runs_l252_252096

variable (A R : ℕ)

def suraj_average_eq (A : ℕ) : Prop :=
  A + 8 = 128

def total_runs_eq (A R : ℕ) : Prop :=
  9 * A + R = 10 * 128

theorem suraj_next_innings_runs :
  ∃ A : ℕ, suraj_average_eq A ∧ ∃ R : ℕ, total_runs_eq A R ∧ R = 200 := 
by
  sorry

end suraj_next_innings_runs_l252_252096


namespace claire_hours_cleaning_l252_252319

-- Definitions of given conditions
def total_hours_in_day : ℕ := 24
def hours_sleeping : ℕ := 8
def hours_cooking : ℕ := 2
def hours_crafting : ℕ := 5
def total_working_hours : ℕ := total_hours_in_day - hours_sleeping

-- Definition of the question
def hours_cleaning := total_working_hours - (hours_cooking + hours_crafting + hours_crafting)

-- The proof goal
theorem claire_hours_cleaning : hours_cleaning = 4 := by
  sorry

end claire_hours_cleaning_l252_252319


namespace smallest_possible_n_l252_252878

theorem smallest_possible_n (x n : ℤ) (hx : 0 < x) (m : ℤ) (hm : m = 30) (h1 : m.gcd n = x + 1) (h2 : m.lcm n = x * (x + 1)) : n = 6 := sorry

end smallest_possible_n_l252_252878


namespace right_triangle_max_value_l252_252611

theorem right_triangle_max_value (a b c : ℝ) (h : a^2 + b^2 = c^2) :
    (a + b) / (ab / c) ≤ 2 * Real.sqrt 2 := sorry

end right_triangle_max_value_l252_252611


namespace range_of_a_l252_252678

variable (a x : ℝ)

theorem range_of_a (h : x - 5 = -3 * a) (hx_neg : x < 0) : a > 5 / 3 :=
by {
  sorry
}

end range_of_a_l252_252678


namespace least_integer_value_of_x_l252_252461

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end least_integer_value_of_x_l252_252461


namespace find_number_of_10_bills_from_mother_l252_252250

variable (m10 : ℕ)  -- number of $10 bills given by Luke's mother

def mother_total : ℕ := 50 + 2*20 + 10*m10
def father_total : ℕ := 4*50 + 20 + 10
def total : ℕ := mother_total m10 + father_total

theorem find_number_of_10_bills_from_mother
  (fee : ℕ := 350)
  (m10 : ℕ) :
  total m10 = fee → m10 = 3 := 
by
  sorry

end find_number_of_10_bills_from_mother_l252_252250


namespace rectangle_perimeter_l252_252149

theorem rectangle_perimeter (a b c d e f g : ℕ)
  (h1 : a + b + c = d)
  (h2 : d + e = g)
  (h3 : b + c = f)
  (h4 : c + f = g)
  (h5 : Nat.gcd (a + b + g) (d + e) = 1)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g) :
  2 * (a + b + g + d + e) = 40 :=
sorry

end rectangle_perimeter_l252_252149


namespace bottles_left_on_shelf_l252_252114

variable (initial_bottles : ℕ)
variable (bottles_jason : ℕ)
variable (bottles_harry : ℕ)

theorem bottles_left_on_shelf (h₁ : initial_bottles = 35) (h₂ : bottles_jason = 5) (h₃ : bottles_harry = bottles_jason + 6) :
  initial_bottles - (bottles_jason + bottles_harry) = 24 := by
  sorry

end bottles_left_on_shelf_l252_252114


namespace sum_of_two_digit_reversible_primes_l252_252005

noncomputable def is_two_digit_prime (n : ℕ) : Prop :=
  (10 ≤ n) ∧ (n < 50) ∧ (Nat.Prime n)

noncomputable def is_reverse_prime (n : ℕ) : Prop :=
  let reversed_digits : ℕ := (n % 10) * 10 + (n / 10) in
  (reversed_digits < 50) ∧ (Nat.Prime reversed_digits)

theorem sum_of_two_digit_reversible_primes : 
  ∑ n in {n | is_two_digit_prime n ∧ is_reverse_prime n}, n = 55 := by
  sorry

end sum_of_two_digit_reversible_primes_l252_252005


namespace inequality_solution_l252_252964

theorem inequality_solution 
  (x : ℝ) : 
  (x^2 / (x+2)^2 ≥ 0) ↔ x ≠ -2 := 
by
  sorry

end inequality_solution_l252_252964


namespace least_number_l252_252172

theorem least_number (n : ℕ) : 
  (n % 45 = 2) ∧ (n % 59 = 2) ∧ (n % 77 = 2) → n = 205517 :=
by
  sorry

end least_number_l252_252172


namespace repeating_decimal_product_as_fraction_l252_252770

theorem repeating_decimal_product_as_fraction :
  let x := 37 / 999
  let y := 7 / 9
  x * y = 259 / 8991 := by {
    sorry
  }

end repeating_decimal_product_as_fraction_l252_252770


namespace meaningful_fraction_l252_252052

theorem meaningful_fraction (x : ℝ) : (x ≠ -2) ↔ (∃ y : ℝ, y = 1 / (x + 2)) :=
by sorry

end meaningful_fraction_l252_252052


namespace right_triangle_shorter_leg_l252_252815

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l252_252815


namespace pentagon_area_l252_252619

-- Define the lengths of the sides of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 25

-- Define the sides of the rectangle and triangle
def rectangle_length := side4
def rectangle_width := side2
def triangle_base := side1
def triangle_height := rectangle_width

-- Define areas of rectangle and right triangle
def area_rectangle := rectangle_length * rectangle_width
def area_triangle := (triangle_base * triangle_height) / 2

-- Define the total area of the pentagon
def total_area_pentagon := area_rectangle + area_triangle

theorem pentagon_area : total_area_pentagon = 925 := by
  sorry

end pentagon_area_l252_252619


namespace ice_cream_depth_l252_252754

noncomputable def volume_sphere (r : ℝ) := (4/3) * Real.pi * r^3
noncomputable def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h

theorem ice_cream_depth
  (radius_sphere : ℝ)
  (radius_cylinder : ℝ)
  (density_constancy : volume_sphere radius_sphere = volume_cylinder radius_cylinder (h : ℝ)) :
  h = 9 / 25 := by
  sorry

end ice_cream_depth_l252_252754


namespace chameleons_color_change_l252_252223

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l252_252223


namespace joohyeon_snack_count_l252_252378

theorem joohyeon_snack_count
  (c s : ℕ)
  (h1 : 300 * c + 500 * s = 3000)
  (h2 : c + s = 8) :
  s = 3 :=
sorry

end joohyeon_snack_count_l252_252378


namespace a_greater_than_b_c_less_than_a_l252_252131

-- Condition 1: Definition of box dimensions
def Box := (Nat × Nat × Nat)

-- Condition 2: Dimension comparisons
def le_box (a b : Box) : Prop :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  (a1 ≤ b1 ∨ a1 ≤ b2 ∨ a1 ≤ b3) ∧ (a2 ≤ b1 ∨ a2 ≤ b2 ∨ a2 ≤ b3) ∧ (a3 ≤ b1 ∨ a3 ≤ b2 ∨ a3 ≤ b3)

def lt_box (a b : Box) : Prop := le_box a b ∧ ¬(a = b)

-- Condition 3: Box dimensions
def A : Box := (6, 5, 3)
def B : Box := (5, 4, 1)
def C : Box := (3, 2, 2)

-- Equivalent Problem 1: Prove A > B
theorem a_greater_than_b : lt_box B A :=
by
  -- theorem proof here
  sorry

-- Equivalent Problem 2: Prove C < A
theorem c_less_than_a : lt_box C A :=
by
  -- theorem proof here
  sorry

end a_greater_than_b_c_less_than_a_l252_252131


namespace root_in_interval_l252_252690

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 < 0 → ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end root_in_interval_l252_252690


namespace friend1_reading_time_friend2_reading_time_l252_252696

theorem friend1_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time / 2) : 
  ∃ t1 : ℕ, t1 = 90 := by
  sorry

theorem friend2_reading_time (my_reading_time : ℕ) (h1 : my_reading_time = 180) (h2 : ∀ t : ℕ, t = my_reading_time * 2) : 
  ∃ t2 : ℕ, t2 = 360 := by
  sorry

end friend1_reading_time_friend2_reading_time_l252_252696


namespace largest_even_number_in_sequence_of_six_l252_252328

-- Definitions and conditions
def smallest_even_number (x : ℤ) : Prop :=
  x + (x + 2) + (x+4) + (x+6) + (x + 8) + (x + 10) = 540

def sum_of_squares_of_sequence (x : ℤ) : Prop :=
  x^2 + (x + 2)^2 + (x + 4)^2 + (x + 6)^2 + (x + 8)^2 + (x + 10)^2 = 97920

-- Statement to prove
theorem largest_even_number_in_sequence_of_six (x : ℤ) (h1 : smallest_even_number x) (h2 : sum_of_squares_of_sequence x) : x + 10 = 95 :=
  sorry

end largest_even_number_in_sequence_of_six_l252_252328


namespace cos_double_angle_l252_252975

open Real

theorem cos_double_angle {α β : ℝ} (h1 : sin α = sqrt 5 / 5)
                         (h2 : sin (α - β) = - sqrt 10 / 10)
                         (h3 : 0 < α ∧ α < π / 2)
                         (h4 : 0 < β ∧ β < π / 2) :
  cos (2 * β) = 0 :=
  sorry

end cos_double_angle_l252_252975


namespace number_of_four_digit_multiples_of_7_l252_252353

theorem number_of_four_digit_multiples_of_7 :
  let first_digit := 1001,
      last_digit := 9996
  in (last_digit - first_digit) / 7 + 1 = 1286 := by {
  -- Skipping the proof
  sorry 
}

end number_of_four_digit_multiples_of_7_l252_252353


namespace find_ten_x_l252_252523

theorem find_ten_x (x : ℝ) 
  (h : 4^(2*x) + 2^(-x) + 1 = (129 + 8 * Real.sqrt 2) * (4^x + 2^(- x) - 2^x)) : 
  10 * x = 35 := 
sorry

end find_ten_x_l252_252523


namespace find_three_numbers_l252_252913

-- Define the conditions
def condition1 (X : ℝ) : Prop := X = 0.35 * X + 60
def condition2 (X Y : ℝ) : Prop := X = 0.7 * (1 / 2) * Y + (1 / 2) * Y
def condition3 (Y Z : ℝ) : Prop := Y = 2 * Z ^ 2

-- Define the final result that we need to prove
def final_result (X Y Z : ℝ) : Prop := X = 92 ∧ Y = 108 ∧ Z = 7

-- The main theorem statement
theorem find_three_numbers :
  ∃ (X Y Z : ℝ), condition1 X ∧ condition2 X Y ∧ condition3 Y Z ∧ final_result X Y Z :=
by
  sorry

end find_three_numbers_l252_252913


namespace compute_expression_l252_252050

theorem compute_expression (x : ℝ) (h : x + (1 / x) = 7) :
  (x - 3)^2 + (49 / (x - 3)^2) = 23 :=
by
  sorry

end compute_expression_l252_252050


namespace cubic_root_abs_power_linear_function_points_l252_252585

theorem cubic_root_abs_power : real.sqrt3 8 + abs (-5) + (-1 : real)^2023 = 6 := by
  sorry

theorem linear_function_points : 
  ∃ k b : real, (0, 1) ∈ set_of (λ p : real × real, p.snd = k * p.fst + b) ∧
                (2, 5) ∈ set_of (λ p : real × real, p.snd = k * p.fst + b) ∧
                (∀ x : real, k * x + b = 2 * x + 1) := by
  sorry

end cubic_root_abs_power_linear_function_points_l252_252585


namespace cos_neg_13pi_div_4_l252_252885

theorem cos_neg_13pi_div_4 : (Real.cos (-13 * Real.pi / 4)) = -Real.sqrt 2 / 2 := 
by sorry

end cos_neg_13pi_div_4_l252_252885


namespace solution_set_of_inequality_l252_252404

theorem solution_set_of_inequality:
  {x : ℝ | x^2 - |x-1| - 1 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end solution_set_of_inequality_l252_252404


namespace betty_cookies_and_brownies_difference_l252_252312

-- Definitions based on the conditions
def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10
def cookies_per_day : ℕ := 3
def brownies_per_day : ℕ := 1
def days : ℕ := 7

-- The proof statement
theorem betty_cookies_and_brownies_difference :
  initial_cookies - (cookies_per_day * days) - (initial_brownies - (brownies_per_day * days)) = 36 :=
by
  sorry

end betty_cookies_and_brownies_difference_l252_252312


namespace smallest_integer_condition_l252_252332

theorem smallest_integer_condition :
  ∃ (x : ℕ) (d : ℕ) (n : ℕ) (p : ℕ), x = 1350 ∧ d = 1 ∧ n = 450 ∧ p = 2 ∧
  x = 10^p * d + n ∧
  n = x / 19 ∧
  (1 ≤ d ∧ d ≤ 9 ∧ 10^p * d % 18 = 0) :=
sorry

end smallest_integer_condition_l252_252332


namespace total_points_l252_252798

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end total_points_l252_252798


namespace remainder_of_sum_division_l252_252739

theorem remainder_of_sum_division (f y : ℤ) (a b : ℤ) (h_f : f = 5 * a + 3) (h_y : y = 5 * b + 4) :  
  (f + y) % 5 = 2 :=
by
  sorry

end remainder_of_sum_division_l252_252739


namespace product_of_areas_square_of_volume_l252_252095

-- Declare the original dimensions and volume
variables (a b c : ℝ)
def V := a * b * c

-- Declare the areas of the new box
def area_bottom := (a + 2) * (b + 2)
def area_side := (b + 2) * (c + 2)
def area_front := (c + 2) * (a + 2)

-- Final theorem to prove
theorem product_of_areas_square_of_volume :
  (area_bottom a b) * (area_side b c) * (area_front c a) = V a b c ^ 2 :=
sorry

end product_of_areas_square_of_volume_l252_252095


namespace equation_for_number_l252_252266

variable (a : ℤ)

theorem equation_for_number : 3 * a + 5 = 9 :=
sorry

end equation_for_number_l252_252266


namespace players_taking_physics_l252_252919

-- Definitions based on the conditions
def total_players : ℕ := 30
def players_taking_math : ℕ := 15
def players_taking_both : ℕ := 6

-- The main theorem to prove
theorem players_taking_physics : total_players - players_taking_math + players_taking_both = 21 := by
  sorry

end players_taking_physics_l252_252919


namespace average_points_per_player_l252_252840

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end average_points_per_player_l252_252840


namespace three_digit_number_div_by_11_l252_252898

theorem three_digit_number_div_by_11 (x : ℕ) (h : x < 10) : 
  ∃ n : ℕ, n = 605 ∧ n < 1000 ∧ 
  (n % 10 = 5 ∧ (n / 100) % 10 = 6 ∧ n % 11 = 0) :=
begin
  use 605,
  split,
  { refl, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  norm_num,
end

end three_digit_number_div_by_11_l252_252898


namespace probability_of_at_least_30_cents_l252_252868

def coin := fin 5

def value (c : coin) : ℤ :=
match c with
| 0 => 1   -- penny
| 1 => 5   -- nickel
| 2 => 10  -- dime
| 3 => 25  -- quarter
| 4 => 50  -- half-dollar
| _ => 0

def coin_flip : coin -> bool := λ c => true -- Placeholder for whether heads or tails

def total_value (flips : coin -> bool) : ℤ :=
  finset.univ.sum (λ c, if flips c then value c else 0)

noncomputable def probability_at_least_30_cents : ℚ :=
  let coin_flips := (finset.pi finset.univ (λ _, finset.univ : finset (coin -> bool))).val in
  let successful_flips := coin_flips.filter (λ flips, total_value flips >= 30) in
  successful_flips.card / coin_flips.card

theorem probability_of_at_least_30_cents :
  probability_at_least_30_cents = 9 / 16 :=
by
  sorry

end probability_of_at_least_30_cents_l252_252868


namespace team_B_task_alone_optimal_scheduling_l252_252659

-- Condition definitions
def task_completed_in_18_months (A : Nat → Prop) : Prop := A 18
def work_together_complete_task_in_10_months (A B : Nat → Prop) : Prop := 
  ∃ n m : ℕ, n = 2 ∧ A n ∧ B m ∧ m = 10 ∧ ∀ x y : ℕ, (x / y = 1 / 18 + 1 / (n + 10))

-- Question 1
theorem team_B_task_alone (B : Nat → Prop) : ∃ x : ℕ, x = 27 := sorry

-- Conditions for the second theorem
def team_a_max_time (a : ℕ) : Prop := a ≤ 6
def team_b_max_time (b : ℕ) : Prop := b ≤ 24
def positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 
def total_work_done (a b : ℕ) : Prop := (a / 18) + (b / 27) = 1

-- Question 2
theorem optimal_scheduling (A B : Nat → Prop) : 
  ∃ a b : ℕ, team_a_max_time a ∧ team_b_max_time b ∧ positive_integers a b ∧
             (a / 18 + b / 27 = 1) → min_cost := sorry

end team_B_task_alone_optimal_scheduling_l252_252659


namespace chameleons_color_change_l252_252209

theorem chameleons_color_change (total_chameleons : ℕ) (initial_blue_red_ratio : ℕ) 
  (final_blue_ratio : ℕ) (final_red_ratio : ℕ) (initial_total : ℕ) (final_total : ℕ) 
  (initial_red : ℕ) (final_red : ℕ) (blues_decrease_by : ℕ) (reds_increase_by : ℕ) 
  (x : ℕ) :
  total_chameleons = 140 →
  initial_blue_red_ratio = 5 →
  final_blue_ratio = 1 →
  final_red_ratio = 3 →
  initial_total = 140 →
  final_total = 140 →
  initial_red = 140 - 5 * x →
  final_red = 3 * (140 - 5 * x) →
  total_chameleons = initial_total →
  (total_chameleons = final_total) →
  initial_red + x = 3 * (140 - 5 * x) →
  blues_decrease_by = (initial_blue_red_ratio - final_blue_ratio) * x →
  reds_increase_by = final_red_ratio * (140 - initial_blue_red_ratio * x) →
  blues_decrease_by = 4 * x →
  x = 20 →
  blues_decrease_by = 80 :=
by
  sorry

end chameleons_color_change_l252_252209


namespace problem1_problem2_l252_252594

-- Problem 1: Calculate \(\sqrt[3]{8} + |-5| + (-1)^{2023} = 6\)
theorem problem1 : (2 + 5 + (-1)).toReal = 6 := by
  sorry

-- Problem 2: Find the expression of the linear function passing through (0, 1) and (2, 5) as y = 2x + 1
theorem problem2 (k b : ℝ) (h1 : b = 1) (h2 : 5 = 2 * k + b) : 
  (∀ x, k * x + b = 2 * x + 1) := by
  sorry

end problem1_problem2_l252_252594


namespace table_tennis_matches_l252_252507

theorem table_tennis_matches (n : ℕ) :
  ∃ x : ℕ, 3 * 2 - x + n * (n - 1) / 2 = 50 ∧ x = 1 :=
by
  sorry

end table_tennis_matches_l252_252507


namespace right_triangle_shorter_leg_l252_252816

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l252_252816


namespace air_conditioner_sales_l252_252166

-- Definitions based on conditions
def ratio_air_conditioners_refrigerators : ℕ := 5
def ratio_refrigerators_air_conditioners : ℕ := 3
def difference_in_sales : ℕ := 54

-- The property to be proven: 
def number_of_air_conditioners : ℕ := 135

theorem air_conditioner_sales
  (r_ac : ℕ := ratio_air_conditioners_refrigerators) 
  (r_ref : ℕ := ratio_refrigerators_air_conditioners) 
  (diff : ℕ := difference_in_sales) 
  : number_of_air_conditioners = 135 := sorry

end air_conditioner_sales_l252_252166


namespace range_m_of_inequality_l252_252977

noncomputable def f (x : ℝ) : ℝ := x / (4 - x^2)

theorem range_m_of_inequality :
  (∀ x ∈ Ioo (-2 : ℝ) 2, f (-x) = -f x) →
  (∀ x₁ x₂ ∈ Ioo (-2 : ℝ) 2, x₁ < x₂ → f x₁ < f x₂) →
  (∀ m : ℝ, f (1 + m) + f (1 - m^2) < 0 ↔ m ∈ Ioo (-Real.sqrt 3) (-1)) :=
by
  sorry

end range_m_of_inequality_l252_252977


namespace complement_of_angle_l252_252793

def complement_angle (deg : ℕ) (min : ℕ) : ℕ × ℕ :=
  if deg < 90 then 
    let total_min := (90 * 60)
    let angle_min := (deg * 60) + min
    let comp_min := total_min - angle_min
    (comp_min / 60, comp_min % 60) -- degrees and remaining minutes
  else 
    (0, 0) -- this case handles if the angle is not less than complement allowable range

-- Definitions based on the problem
def given_angle_deg : ℕ := 57
def given_angle_min : ℕ := 13

-- Complement calculation
def comp (deg : ℕ) (min : ℕ) : ℕ × ℕ := complement_angle deg min

-- Expected result of the complement
def expected_comp : ℕ × ℕ := (32, 47)

-- Theorem to prove the complement of 57°13' is 32°47'
theorem complement_of_angle : comp given_angle_deg given_angle_min = expected_comp := by
  sorry

end complement_of_angle_l252_252793


namespace evaluate_expression_l252_252168

theorem evaluate_expression :
  ((3^1 - 2 + 7^3 + 1 : ℚ)⁻¹ * 6) = (2 / 115) := by
  sorry

end evaluate_expression_l252_252168


namespace largest_m_dividing_factorials_l252_252154

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

theorem largest_m_dividing_factorials (m : ℕ) :
  (∀ k : ℕ, k ≤ m → factorial k ∣ (factorial 100 + factorial 99 + factorial 98)) ↔ m = 98 :=
by
  sorry

end largest_m_dividing_factorials_l252_252154


namespace both_hit_exactly_one_hits_at_least_one_hits_l252_252408

noncomputable def prob_A : ℝ := 0.8
noncomputable def prob_B : ℝ := 0.9

theorem both_hit : prob_A * prob_B = 0.72 := by
  sorry

theorem exactly_one_hits : prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = 0.26 := by
  sorry

theorem at_least_one_hits : 1 - (1 - prob_A) * (1 - prob_B) = 0.98 := by
  sorry

end both_hit_exactly_one_hits_at_least_one_hits_l252_252408


namespace sin_double_angle_l252_252338

theorem sin_double_angle {x : ℝ} (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_l252_252338


namespace integers_square_less_than_three_times_l252_252132

theorem integers_square_less_than_three_times (x : ℤ) : x^2 < 3 * x ↔ x = 1 ∨ x = 2 :=
by
  sorry

end integers_square_less_than_three_times_l252_252132


namespace d_share_l252_252436

theorem d_share (T : ℝ) (A B C D E : ℝ) 
  (h1 : A = 5 / 15 * T) 
  (h2 : B = 2 / 15 * T) 
  (h3 : C = 4 / 15 * T)
  (h4 : D = 3 / 15 * T)
  (h5 : E = 1 / 15 * T)
  (combined_AC : A + C = 3 / 5 * T)
  (diff_BE : B - E = 250) : 
  D = 750 :=
by
  sorry

end d_share_l252_252436


namespace ellipse_foci_l252_252102

noncomputable def focal_coordinates (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

-- Given the equation of the ellipse: x^2 / a^2 + y^2 / b^2 = 1
def ellipse_equation (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

-- Proposition stating that if the ellipse equation holds for a=√5 and b=2, then the foci are at (± c, 0)
theorem ellipse_foci (x y : ℝ) (h : ellipse_equation x y (Real.sqrt 5) 2) :
  y = 0 ∧ (x = 1 ∨ x = -1) :=
sorry

end ellipse_foci_l252_252102


namespace shorter_leg_of_right_triangle_l252_252803

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l252_252803


namespace max_vouchers_with_680_l252_252018

def spend_to_voucher (spent : ℕ) : ℕ := (spent / 100) * 20

theorem max_vouchers_with_680 : spend_to_voucher 680 = 160 := by
  sorry

end max_vouchers_with_680_l252_252018


namespace exists_pentagon_from_midpoints_l252_252620

noncomputable def pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) : Prop :=
  ∃ (A B C D E : ℝ × ℝ), 
    (A1 = (A + B) / 2) ∧ 
    (B1 = (B + C) / 2) ∧ 
    (C1 = (C + D) / 2) ∧ 
    (D1 = (D + E) / 2) ∧ 
    (E1 = (E + A) / 2)

-- statement of the theorem
theorem exists_pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) :
  pentagon_from_midpoints A1 B1 C1 D1 E1 :=
sorry

end exists_pentagon_from_midpoints_l252_252620


namespace rabbits_in_cage_l252_252199

theorem rabbits_in_cage (rabbits_in_cage : ℕ) (rabbits_park : ℕ) : 
  rabbits_in_cage = 13 ∧ rabbits_park = 60 → (1/3 * rabbits_park - rabbits_in_cage) = 7 :=
by
  sorry

end rabbits_in_cage_l252_252199


namespace count_perfect_cubes_l252_252489

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1600) :
  ∃ (n : ℕ), n = 6 :=
by
  sorry

end count_perfect_cubes_l252_252489


namespace average_of_roots_l252_252610

theorem average_of_roots (c : ℝ) (h : ∃ x1 x2 : ℝ, 2 * x1^2 - 6 * x1 + c = 0 ∧ 2 * x2^2 - 6 * x2 + c = 0 ∧ x1 ≠ x2) :
    (∃ p q : ℝ, (2 : ℝ) * (p : ℝ)^2 + (-6 : ℝ) * p + c = 0 ∧ (2 : ℝ) * (q : ℝ)^2 + (-6 : ℝ) * q + c = 0 ∧ p ≠ q) →
    (p + q) / 2 = 3 / 2 := 
sorry

end average_of_roots_l252_252610


namespace bottles_left_l252_252117

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l252_252117


namespace product_sequence_eq_l252_252760

theorem product_sequence_eq :
  let seq := [ (1 : ℚ) / 2, 4 / 1, 1 / 8, 16 / 1, 1 / 32, 64 / 1,
               1 / 128, 256 / 1, 1 / 512, 1024 / 1, 1 / 2048, 4096 / 1 ]
  (seq.prod) * (3 / 4) = 1536 := by 
  -- expand and simplify the series of products
  sorry 

end product_sequence_eq_l252_252760


namespace total_students_calculation_score_cutoff_calculation_l252_252165

-- Definitions based on conditions
def mu : ℝ := 60
def sigma2 : ℝ := 100
def sigma : ℝ := Real.sqrt sigma2
def z_score (x : ℝ) : ℝ := (x - mu) / sigma

-- Constants
def students_scored_90_or_above : ℕ := 13
def total_students : ℕ := 10000
def top_rewarded_students : ℕ := 228

-- Proving the total number of students given the conditions
theorem total_students_calculation :
  (1 - Real.cdf (z_score 90)) * total_students = students_scored_90_or_above :=
sorry

-- Proving the score cutoff for top 228 students
theorem score_cutoff_calculation :
  let p_top := (top_rewarded_students : ℝ) / (total_students : ℝ)
  let z := Real.invCdf (1 - p_top)
  Real.toFixed (z * sigma + mu) = 80 :=
sorry

end total_students_calculation_score_cutoff_calculation_l252_252165


namespace domain_of_g_eq_l252_252133

noncomputable def g (x : ℝ) : ℝ := (x + 2) / (Real.sqrt (x^2 - 5 * x + 6))

theorem domain_of_g_eq : 
  {x : ℝ | 0 < x^2 - 5 * x + 6} = {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_g_eq_l252_252133


namespace length_of_second_offset_l252_252023

theorem length_of_second_offset (d₁ d₂ h₁ A : ℝ) (h_d₁ : d₁ = 30) (h_h₁ : h₁ = 9) (h_A : A = 225):
  ∃ h₂, (A = (1/2) * d₁ * h₁ + (1/2) * d₁ * h₂) → h₂ = 6 := by
  sorry

end length_of_second_offset_l252_252023


namespace min_value_inequality_l252_252385

theorem min_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 := 
by
  sorry

end min_value_inequality_l252_252385


namespace four_digit_multiples_of_7_l252_252357

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l252_252357


namespace projectiles_meet_in_90_minutes_l252_252994

theorem projectiles_meet_in_90_minutes
  (d : ℝ) (v1 : ℝ) (v2 : ℝ) (time_in_minutes : ℝ)
  (h_d : d = 1455)
  (h_v1 : v1 = 470)
  (h_v2 : v2 = 500)
  (h_time : time_in_minutes = 90) :
  d / (v1 + v2) * 60 = time_in_minutes :=
by
  sorry

end projectiles_meet_in_90_minutes_l252_252994


namespace binomial_12_6_l252_252928

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l252_252928


namespace jerry_age_l252_252531

theorem jerry_age (M J : ℤ) (h1 : M = 16) (h2 : M = 2 * J - 8) : J = 12 :=
by
  sorry

end jerry_age_l252_252531


namespace petya_cannot_have_equal_coins_l252_252547

theorem petya_cannot_have_equal_coins
  (transact : ℕ → ℕ)
  (initial_two_kopeck : ℕ)
  (total_operations : ℕ)
  (insertion_machine : ℕ)
  (by_insert_two : ℕ)
  (by_insert_ten : ℕ)
  (odd : ℕ)
  :
  (initial_two_kopeck = 1) ∧ 
  (by_insert_two = 5) ∧ 
  (by_insert_ten = 5) ∧
  (∀ n, transact n = 1 + 4 * n) →
  (odd % 2 = 1) →
  (total_operations = transact insertion_machine) →
  (total_operations % 2 = 1) →
  (∀ x y, (x + y = total_operations) → (x = y) → False) :=
sorry

end petya_cannot_have_equal_coins_l252_252547


namespace ron_chocolate_bar_cost_l252_252428

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l252_252428


namespace coeff_x3y2z5_in_expansion_l252_252171

def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3y2z5_in_expansion :
  let x := 1
  let y := 1
  let z := 1
  let x_term := 2 * x
  let y_term := y
  let z_term := z
  let target_term := x_term ^ 3 * y_term ^ 2 * z_term ^ 5
  let coeff := 2^3 * binomialCoeff 10 3 * binomialCoeff 7 2 * binomialCoeff 5 5
  coeff = 20160 :=
by
  sorry

end coeff_x3y2z5_in_expansion_l252_252171


namespace necessary_french_woman_l252_252449

structure MeetingConditions where
  total_money_women : ℝ
  total_money_men : ℝ
  total_money_french : ℝ
  total_money_russian : ℝ

axiom no_other_representatives : Prop
axiom money_french_vs_russian (conditions : MeetingConditions) : conditions.total_money_french > conditions.total_money_russian
axiom money_women_vs_men (conditions : MeetingConditions) : conditions.total_money_women > conditions.total_money_men

theorem necessary_french_woman (conditions : MeetingConditions) :
  ∃ w_f : ℝ, w_f > 0 ∧ conditions.total_money_french > w_f ∧ w_f + conditions.total_money_men > conditions.total_money_women :=
by
  sorry

end necessary_french_woman_l252_252449


namespace product_of_fractions_l252_252006

theorem product_of_fractions :
  (2 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) * (5 / 6 : ℚ) * (6 / 7 : ℚ) * (7 / 8 : ℚ) = 1 / 4 :=
by
  sorry

end product_of_fractions_l252_252006


namespace min_balls_to_draw_l252_252145

theorem min_balls_to_draw {red green yellow blue white black : ℕ} 
    (h_red : red = 28) 
    (h_green : green = 20) 
    (h_yellow : yellow = 19) 
    (h_blue : blue = 13) 
    (h_white : white = 11) 
    (h_black : black = 9) :
    ∃ n, n = 76 ∧ 
    (∀ drawn, (drawn < n → (drawn ≤ 14 + 14 + 14 + 13 + 11 + 9)) ∧ (drawn >= n → (∃ c, c ≥ 15))) :=
sorry

end min_balls_to_draw_l252_252145


namespace plane_eq_passing_A_perpendicular_BC_l252_252296

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def subtract_points (P Q : Point3D) : Point3D :=
  { x := P.x - Q.x, y := P.y - Q.y, z := P.z - Q.z }

-- Points A, B, and C given in the conditions
def A : Point3D := { x := 1, y := -5, z := -2 }
def B : Point3D := { x := 6, y := -2, z := 1 }
def C : Point3D := { x := 2, y := -2, z := -2 }

-- Vector BC
def BC : Point3D := subtract_points C B

theorem plane_eq_passing_A_perpendicular_BC :
  (-4 : ℝ) * (A.x - 1) + (0 : ℝ) * (A.y + 5) + (-3 : ℝ) * (A.z + 2) = 0 :=
  sorry

end plane_eq_passing_A_perpendicular_BC_l252_252296


namespace part1_part2_l252_252485

open Real

def f (x a : ℝ) := abs (x + 2 * a) + abs (x - 1)

section part1

variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

end part1

section part2

noncomputable def g (a : ℝ) := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part2 {a : ℝ} (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end part2

end part1_part2_l252_252485


namespace tourist_tax_l252_252916

theorem tourist_tax (total_value : ℝ) (non_taxable_amount : ℝ) (tax_rate : ℝ) 
  (h1 : total_value = 1720) (h2 : non_taxable_amount = 600) (h3 : tax_rate = 0.08) : 
  ((total_value - non_taxable_amount) * tax_rate = 89.60) :=
by 
  sorry

end tourist_tax_l252_252916


namespace value_is_correct_l252_252742

-- Define the number
def initial_number : ℝ := 4400

-- Define the value calculation in Lean
def value : ℝ := 0.15 * (0.30 * (0.50 * initial_number))

-- The theorem statement
theorem value_is_correct : value = 99 := by
  sorry

end value_is_correct_l252_252742


namespace binom_12_6_eq_924_l252_252939

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l252_252939


namespace total_books_sum_l252_252069

-- Given conditions
def Joan_books := 10
def Tom_books := 38
def Lisa_books := 27
def Steve_books := 45
def Kim_books := 14
def Alex_books := 48

-- Define the total number of books
def total_books := Joan_books + Tom_books + Lisa_books + Steve_books + Kim_books + Alex_books

-- Proof statement
theorem total_books_sum : total_books = 182 := by
  sorry

end total_books_sum_l252_252069


namespace shifted_sine_function_l252_252500

theorem shifted_sine_function :
  ∀ x : ℝ, (2 * Real.sin (2 * x + π / 6)) = (2 * Real.sin (2 * (x - π / 4) + π / 6)) ↔
            (2 * Real.sin (2 * x - π / 3)) :=
by
  sorry

end shifted_sine_function_l252_252500


namespace log_addition_closed_l252_252525

def is_log_of_nat (n : ℝ) : Prop := ∃ k : ℕ, k > 0 ∧ n = Real.log k

theorem log_addition_closed (a b : ℝ) (ha : is_log_of_nat a) (hb : is_log_of_nat b) : is_log_of_nat (a + b) :=
by
  sorry

end log_addition_closed_l252_252525


namespace average_halfway_l252_252173

theorem average_halfway (a b : ℚ) (h_a : a = 1/8) (h_b : b = 1/3) : (a + b) / 2 = 11 / 48 := by
  sorry

end average_halfway_l252_252173


namespace chameleon_color_change_l252_252214

theorem chameleon_color_change :
  ∃ (x : ℕ), (∃ (blue_initial : ℕ) (red_initial : ℕ), blue_initial = 5 * x ∧ red_initial = 140 - 5 * x) →
  (∃ (blue_final : ℕ) (red_final : ℕ), blue_final = x ∧ red_final = 3 * (140 - 5 * x)) →
  (5 * x - x = 80) :=
begin
  sorry
end

end chameleon_color_change_l252_252214


namespace trajectory_of_moving_point_l252_252671

theorem trajectory_of_moving_point (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
  (hF1 : F1 = (-2, 0)) (hF2 : F2 = (2, 0))
  (h_arith_mean : dist F1 F2 = (dist P F1 + dist P F2) / 2) :
  ∃ a b : ℝ, a = 4 ∧ b^2 = 12 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1) :=
sorry

end trajectory_of_moving_point_l252_252671


namespace probability_at_least_one_female_is_five_sixths_l252_252034

-- Declare the total number of male and female students
def total_male_students := 6
def total_female_students := 4
def total_students := total_male_students + total_female_students
def selected_students := 3

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 3 students from 10 students
def total_ways_to_select_3 := binomial_coefficient total_students selected_students

-- Ways to select 3 male students from 6 male students
def ways_to_select_3_males := binomial_coefficient total_male_students selected_students

-- Probability of selecting at least one female student
def probability_of_at_least_one_female : ℚ := 1 - (ways_to_select_3_males / total_ways_to_select_3)

-- The theorem statement to be proved
theorem probability_at_least_one_female_is_five_sixths :
  probability_of_at_least_one_female = 5/6 := by
  sorry

end probability_at_least_one_female_is_five_sixths_l252_252034


namespace polynomial_expansion_l252_252020

noncomputable def poly1 (z : ℝ) : ℝ := 3 * z ^ 3 + 2 * z ^ 2 - 4 * z + 1
noncomputable def poly2 (z : ℝ) : ℝ := 2 * z ^ 4 - 3 * z ^ 2 + z - 5
noncomputable def expanded_poly (z : ℝ) : ℝ := 6 * z ^ 7 + 4 * z ^ 6 - 4 * z ^ 5 - 9 * z ^ 3 + 7 * z ^ 2 + z - 5

theorem polynomial_expansion (z : ℝ) : poly1 z * poly2 z = expanded_poly z := by
  sorry

end polynomial_expansion_l252_252020


namespace pyramid_on_pentagonal_prism_l252_252698

-- Define the structure of a pentagonal prism
structure PentagonalPrism where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

-- Initial pentagonal prism properties
def initialPrism : PentagonalPrism := {
  faces := 7,
  vertices := 10,
  edges := 15
}

-- Assume we add a pyramid on top of one pentagonal face
def addPyramid (prism : PentagonalPrism) : PentagonalPrism := {
  faces := prism.faces - 1 + 5, -- 1 face covered, 5 new faces
  vertices := prism.vertices + 1, -- 1 new vertex
  edges := prism.edges + 5 -- 5 new edges
}

-- The resulting shape after adding the pyramid
def resultingShape : PentagonalPrism := addPyramid initialPrism

-- Calculating the sum of faces, vertices, and edges
def sumFacesVerticesEdges (shape : PentagonalPrism) : ℕ :=
  shape.faces + shape.vertices + shape.edges

-- Statement of the problem in Lean 4
theorem pyramid_on_pentagonal_prism : sumFacesVerticesEdges resultingShape = 42 := by
  sorry

end pyramid_on_pentagonal_prism_l252_252698


namespace peter_total_food_l252_252650

-- Given conditions
variables (chicken hamburgers hot_dogs sides total_food : ℕ)
  (h1 : chicken = 16)
  (h2 : hamburgers = chicken / 2)
  (h3 : hot_dogs = hamburgers + 2)
  (h4 : sides = hot_dogs / 2)
  (total_food : ℕ := chicken + hamburgers + hot_dogs + sides)

theorem peter_total_food (h1 : chicken = 16)
                          (h2 : hamburgers = chicken / 2)
                          (h3 : hot_dogs = hamburgers + 2)
                          (h4 : sides = hot_dogs / 2) :
                          total_food = 39 :=
by sorry

end peter_total_food_l252_252650


namespace solve_phi_eq_l252_252092

noncomputable def φ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat := (1 - Real.sqrt 5) / 2
noncomputable def F : ℕ → ℤ
| n =>
  if n = 0 then 0
  else if n = 1 then 1
  else F (n - 1) + F (n - 2)

theorem solve_phi_eq (n : ℕ) :
  ∃ x y : ℤ, x * φ ^ (n + 1) + y * φ^n = 1 ∧ 
    x = (-1 : ℤ)^(n+1) * F n ∧ y = (-1 : ℤ)^n * F (n + 1) := by
  sorry

end solve_phi_eq_l252_252092


namespace find_slope_l3_l252_252078

/-- Conditions --/
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def line2 (x y : ℝ) : Prop := y = 2
def A : Prod ℝ ℝ := (0, -3)
def area_ABC : ℝ := 5

noncomputable def B : Prod ℝ ℝ := (2, 2)  -- Simultaneous solution of line1 and line2

theorem find_slope_l3 (C : ℝ × ℝ) (slope_l3 : ℝ) :
  line2 C.1 C.2 ∧
  ((0 : ℝ), -3) ∈ {p : ℝ × ℝ | line1 p.1 p.2 → line2 p.1 p.2 } ∧
  C.2 = 2 ∧
  0 ≤ slope_l3 ∧
  area_ABC = 5 →
  slope_l3 = 5 / 4 :=
sorry

end find_slope_l3_l252_252078


namespace geometric_sequence_sum_l252_252478

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_a1 : a 1 = 3)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 4 + a 5 + a 6 = 168 := 
sorry

end geometric_sequence_sum_l252_252478


namespace probability_reroll_two_dice_eq_5_over_36_l252_252242

/-- Jason rolls three fair six-sided dice. He can choose a subset of the dice to reroll. 
  Jason wins if and only if the sum of the dice is 9.
  The probability that Jason's optimal strategy rerolls exactly two dice is 5/36. -/
theorem probability_reroll_two_dice_eq_5_over_36 :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let favorable_outcomes := 
    outcomes.filter (λ t, 
      let (a,(b,c)) := t in
      (a + b + c = 9 ∧ 
       ((a + (6.choose 2) = 9 ∨ b + (6.choose 2) = 9 ∨ c + (6.choose 2) = 9) ∨ 
        (a + b = 9 - c) ∨ (a + c = 9 - b) ∨ (b + c = 9 - a)))
    )
  in (favorable_outcomes.card: ℚ) / (outcomes.card : ℚ) = 5/36 :=
sorry

end probability_reroll_two_dice_eq_5_over_36_l252_252242


namespace miss_adamson_num_classes_l252_252851

theorem miss_adamson_num_classes
  (students_per_class : ℕ)
  (sheets_per_student : ℕ)
  (total_sheets : ℕ)
  (h1 : students_per_class = 20)
  (h2 : sheets_per_student = 5)
  (h3 : total_sheets = 400) :
  let sheets_per_class := sheets_per_student * students_per_class
  let num_classes := total_sheets / sheets_per_class
  num_classes = 4 :=
by
  sorry

end miss_adamson_num_classes_l252_252851


namespace zombies_less_than_50_four_days_ago_l252_252888

theorem zombies_less_than_50_four_days_ago
  (curr_zombies : ℕ)
  (days_ago : ℕ)
  (half_rate : ℕ)
  (initial_zombies : ℕ)
  (h_initial : curr_zombies = 480)
  (h_half : half_rate = 2)
  (h_days : days_ago = 4)
  : (curr_zombies / half_rate^days_ago) < 50 :=
by
  have h1 : curr_zombies / half_rate^1 = 480 / 2 := sorry
  have h2 : curr_zombies / half_rate^2 = 480 / 2^2 := sorry
  have h3 : curr_zombies / half_rate^3 = 480 / 2^3 := sorry
  have h4 : curr_zombies / half_rate^4 = 480 / 2^4 := sorry
  show 30 < 50 from sorry
  rw h_initial at *
  sorry

end zombies_less_than_50_four_days_ago_l252_252888


namespace no_2018_zero_on_curve_l252_252674

theorem no_2018_zero_on_curve (a c d : ℝ) (hac : a * c > 0) : ¬∃(d : ℝ), (2018 : ℝ) ^ 2 * a + 2018 * c + d = 0 := 
by {
  sorry
}

end no_2018_zero_on_curve_l252_252674


namespace shorter_leg_of_right_triangle_l252_252823

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l252_252823


namespace connected_graph_edges_ge_verts_minus_one_l252_252858

variables {V : Type*} {E : Type*}

-- Define a connected simple graph G = (V, E) with vertices V and edges E
structure ConnectedGraph (V : Type*) :=
(verts : Type*)
(edges : set (verts × verts))
(is_connected : ∀ v1 v2 : verts, ∃ (path : list verts), v1 ∈ path ∧ v2 ∈ path ∧ (∀ i ∈ list.zip (path.init) (path.tail), (i.fst, i.snd) ∈ edges ∨ (i.snd, i.fst) ∈ edges))

-- Define the predicate for the number of edges
def num_edges {V : Type*} {E : set (V × V)} : ℕ := set.card E
def num_verts {V : Type*} (G : ConnectedGraph V) := set.card G.verts

-- Statement of the theorem
theorem connected_graph_edges_ge_verts_minus_one (G : ConnectedGraph V) :
  num_edges G.edges ≥ num_verts G - 1 := sorry

end connected_graph_edges_ge_verts_minus_one_l252_252858


namespace molecular_weight_of_Aluminium_hydroxide_l252_252726

-- Given conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Definition of molecular weight of Aluminium hydroxide
def molecular_weight_Al_OH_3 : ℝ := 
  atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

-- Proof statement
theorem molecular_weight_of_Aluminium_hydroxide : molecular_weight_Al_OH_3 = 78.01 :=
  by sorry

end molecular_weight_of_Aluminium_hydroxide_l252_252726


namespace sum_five_consecutive_l252_252094

theorem sum_five_consecutive (n : ℤ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) = 5 * n + 10 := by
  sorry

end sum_five_consecutive_l252_252094


namespace least_integer_value_l252_252464

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end least_integer_value_l252_252464
