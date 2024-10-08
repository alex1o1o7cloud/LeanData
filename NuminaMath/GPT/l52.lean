import Mathlib

namespace three_consecutive_odds_l52_52549

theorem three_consecutive_odds (x : ℤ) (h3 : x + 4 = 133) : 
  x + (x + 4) = 3 * (x + 2) - 131 := 
by {
  sorry
}

end three_consecutive_odds_l52_52549


namespace contrapositive_proof_l52_52196

theorem contrapositive_proof (x : ℝ) : (x^2 < 1 → -1 < x ∧ x < 1) → (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by
  sorry

end contrapositive_proof_l52_52196


namespace problem_equivalent_statement_l52_52031

-- Conditions as Lean definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def periodic_property (f : ℝ → ℝ) := ∀ x, x ≥ 0 → f (x + 2) = -f x
def specific_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 8

-- The main theorem
theorem problem_equivalent_statement (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_periodic : periodic_property f) 
  (hf_specific : specific_interval f) :
  f (-2013) + f 2014 = 1 / 3 := 
sorry

end problem_equivalent_statement_l52_52031


namespace count_angles_l52_52783

open Real

noncomputable def isGeometricSequence (a b c : ℝ) : Prop :=
(a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a / b = b / c ∨ b / a = a / c ∨ c / a = a / b)

theorem count_angles (h1 : ∀ θ : ℝ, 0 < θ ∧ θ < 2 * π → (sin θ * cos θ = tan θ) ∨ (sin θ ^ 3 = cos θ ^ 2)) :
  ∃ n : ℕ, 
    (∀ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ (θ % (π/2) ≠ 0) → isGeometricSequence (sin θ) (cos θ) (tan θ) ) → 
    n = 6 := 
sorry

end count_angles_l52_52783


namespace ellipse_major_minor_axes_product_l52_52908

-- Definitions based on conditions
def OF : ℝ := 8
def inradius_triangle_OCF : ℝ := 2  -- diameter / 2

-- Define a and b based on the ellipse properties and conditions
def a : ℝ := 10  -- Solved from the given conditions and steps
def b : ℝ := 6   -- Solved from the given conditions and steps

-- Defining the axes of the ellipse in terms of a and b
def AB : ℝ := 2 * a
def CD : ℝ := 2 * b

-- The product (AB)(CD) we are interested in
def product_AB_CD := AB * CD

-- The main proof statement
theorem ellipse_major_minor_axes_product : product_AB_CD = 240 :=
by
  sorry

end ellipse_major_minor_axes_product_l52_52908


namespace right_triangle_inequality_l52_52614

-- Definition of a right-angled triangle with given legs a, b, hypotenuse c, and altitude h_c to the hypotenuse
variables {a b c h_c : ℝ}

-- Right-angled triangle condition definition with angle at C is right
def right_angled_triangle (a b c : ℝ) : Prop :=
  ∃ (a b c : ℝ), c^2 = a^2 + b^2

-- Definition of the altitude to the hypotenuse
def altitude_to_hypotenuse (a b c h_c : ℝ) : Prop :=
  h_c = (a * b) / c

-- Theorem statement to prove the inequality for any right-angled triangle
theorem right_triangle_inequality (a b c h_c : ℝ) (h1 : right_angled_triangle a b c) (h2 : altitude_to_hypotenuse a b c h_c) : 
  a + b < c + h_c :=
by
  sorry

end right_triangle_inequality_l52_52614


namespace isosceles_triangle_perimeter_l52_52499

theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h1 : (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) 
  (h2 : a + b > c ∧ a + c > b ∧ b + c > a) : a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l52_52499


namespace calculate_total_prime_dates_l52_52264

-- Define the prime months
def prime_months : List Nat := [2, 3, 5, 7, 11, 13]

-- Define the number of days in each month for a non-leap year
def days_in_month (month : Nat) : Nat :=
  if month = 2 then 28
  else if month = 3 then 31
  else if month = 5 then 31
  else if month = 7 then 31
  else if month = 11 then 30
  else if month = 13 then 31
  else 0

-- Define the prime days in a month
def prime_days : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Calculate the number of prime dates in a given month
def prime_dates_in_month (month : Nat) : Nat :=
  (prime_days.filter (λ d => d <= days_in_month month)).length

-- Calculate the total number of prime dates for the year
def total_prime_dates : Nat :=
  (prime_months.map prime_dates_in_month).sum

theorem calculate_total_prime_dates : total_prime_dates = 62 := by
  sorry

end calculate_total_prime_dates_l52_52264


namespace symmetric_line_equation_l52_52184

theorem symmetric_line_equation (x y : ℝ) (h₁ : x + y + 1 = 0) : (2 - x) + (4 - y) - 7 = 0 :=
by
  sorry

end symmetric_line_equation_l52_52184


namespace arc_length_calculation_l52_52961

theorem arc_length_calculation (C θ : ℝ) (hC : C = 72) (hθ : θ = 45) :
  (θ / 360) * C = 9 :=
by
  sorry

end arc_length_calculation_l52_52961


namespace solve_for_n_l52_52469

theorem solve_for_n (n : ℕ) (h : 9^n * 9^n * 9^n * 9^n = 81^n) : n = 0 :=
by
  sorry

end solve_for_n_l52_52469


namespace solve_fractional_equation_l52_52540

theorem solve_fractional_equation (x : ℝ) (h₀ : 2 = 3 * (x + 1) / (4 - x)) : x = 1 :=
sorry

end solve_fractional_equation_l52_52540


namespace find_c2_given_d4_l52_52032

theorem find_c2_given_d4 (c d k : ℝ) (h : c^2 * d^4 = k) (hc8 : c = 8) (hd2 : d = 2) (hd4 : d = 4):
  c^2 = 4 :=
by
  sorry

end find_c2_given_d4_l52_52032


namespace parabola_vertex_l52_52178

theorem parabola_vertex :
  ∃ a k : ℝ, (∀ x y : ℝ, y^2 - 4*y + 2*x + 7 = 0 ↔ y = k ∧ x = a - (1/2)*(y - k)^2) ∧ a = -3/2 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l52_52178


namespace range_of_a_l52_52295

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + 3 < 0 ∧ 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0 ) ↔ (-4 ≤ a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l52_52295


namespace correct_equation_l52_52697

theorem correct_equation :
  ¬ (7^3 * 7^3 = 7^9) ∧ 
  (-3^7 / 3^2 = -3^5) ∧ 
  ¬ (2^6 + (-2)^6 = 0) ∧ 
  ¬ ((-3)^5 / (-3)^3 = -3^2) :=
by 
  sorry

end correct_equation_l52_52697


namespace new_problem_l52_52428

theorem new_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := 
by
  sorry

end new_problem_l52_52428


namespace identity_is_only_sum_free_preserving_surjection_l52_52514

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, f m = n

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ x y : ℕ, x ∈ A → y ∈ A → x + y ∉ A

noncomputable def identity_function_property : Prop :=
  ∀ f : ℕ → ℕ, is_surjective f →
  (∀ A : Set ℕ, is_sum_free A → is_sum_free (Set.image f A)) →
  ∀ n : ℕ, f n = n

theorem identity_is_only_sum_free_preserving_surjection : identity_function_property := sorry

end identity_is_only_sum_free_preserving_surjection_l52_52514


namespace cube_side_length_of_paint_cost_l52_52067

theorem cube_side_length_of_paint_cost (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  cost_per_kg = 20 ∧ coverage_per_kg = 15 ∧ total_cost = 200 →
  6 * side_length ^ 2 = (total_cost / cost_per_kg) * coverage_per_kg →
  side_length = 5 :=
by
  intros h1 h2
  sorry

end cube_side_length_of_paint_cost_l52_52067


namespace integer_a_for_factoring_l52_52000

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end integer_a_for_factoring_l52_52000


namespace two_digit_remainder_one_when_divided_by_4_and_17_l52_52663

-- Given the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def yields_remainder (n d r : ℕ) : Prop := n % d = r

-- Define the main problem that checks if there is only one such number
theorem two_digit_remainder_one_when_divided_by_4_and_17 :
  ∃! n : ℕ, is_two_digit n ∧ yields_remainder n 4 1 ∧ yields_remainder n 17 1 :=
sorry

end two_digit_remainder_one_when_divided_by_4_and_17_l52_52663


namespace fifth_coordinate_is_14_l52_52203

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end fifth_coordinate_is_14_l52_52203


namespace negation_of_proposition_l52_52204

-- Definitions and conditions from the problem
def original_proposition (x : ℝ) : Prop := x^3 - x^2 + 1 > 0

-- The proof problem: Prove the negation
theorem negation_of_proposition : (¬ ∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, ¬original_proposition x := 
by
  -- here we insert our proof later
  sorry

end negation_of_proposition_l52_52204


namespace smallest_angle_in_convex_20_gon_seq_l52_52902

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l52_52902


namespace solve_cubic_eq_with_geo_prog_coeff_l52_52544

variables {a q x : ℝ}

theorem solve_cubic_eq_with_geo_prog_coeff (h_a_nonzero : a ≠ 0) 
    (h_b : b = a * q) (h_c : c = a * q^2) (h_d : d = a * q^3) :
    (a * x^3 + b * x^2 + c * x + d = 0) → (x = -q) :=
by
  intros h_cubic_eq
  have h_b' : b = a * q := h_b
  have h_c' : c = a * q^2 := h_c
  have h_d' : d = a * q^3 := h_d
  sorry

end solve_cubic_eq_with_geo_prog_coeff_l52_52544


namespace Shell_Ratio_l52_52303

-- Definitions of the number of shells collected by Alan, Ben, and Laurie.
variable (A B L : ℕ)

-- Hypotheses based on the given conditions:
-- 1. Alan collected four times as many shells as Ben did.
-- 2. Laurie collected 36 shells.
-- 3. Alan collected 48 shells.
theorem Shell_Ratio (h1 : A = 4 * B) (h2 : L = 36) (h3 : A = 48) : B / Nat.gcd B L = 1 ∧ L / Nat.gcd B L = 3 :=
by
  sorry

end Shell_Ratio_l52_52303


namespace inequality_true_l52_52063

theorem inequality_true (a b : ℝ) (hab : a < b) (hb : b < 0) (ha : a < 0) : (b / a) < 1 :=
by
  sorry

end inequality_true_l52_52063


namespace find_slope_l52_52650

theorem find_slope (k b x y y2 : ℝ) (h1 : y = k * x + b) (h2 : y2 = k * (x + 3) + b) (h3 : y2 - y = -2) : k = -2 / 3 := by
  sorry

end find_slope_l52_52650


namespace russian_dolls_initial_purchase_l52_52223

theorem russian_dolls_initial_purchase (cost_initial cost_discount : ℕ) (num_discount : ℕ) (savings : ℕ) :
  cost_initial = 4 → cost_discount = 3 → num_discount = 20 → savings = num_discount * cost_discount → 
  (savings / cost_initial) = 15 := 
by {
sorry
}

end russian_dolls_initial_purchase_l52_52223


namespace amount_allocated_to_food_l52_52313

theorem amount_allocated_to_food (total_amount : ℝ) (household_ratio food_ratio misc_ratio : ℝ) 
  (h₁ : total_amount = 1800) (h₂ : household_ratio = 5) (h₃ : food_ratio = 4) (h₄ : misc_ratio = 1) :
  food_ratio / (household_ratio + food_ratio + misc_ratio) * total_amount = 720 :=
by
  sorry

end amount_allocated_to_food_l52_52313


namespace time_after_2051_hours_l52_52568

theorem time_after_2051_hours (h₀ : 9 ≤ 11): 
  (9 + 2051 % 12) % 12 = 8 :=
by {
  -- proving the statement here
  sorry
}

end time_after_2051_hours_l52_52568


namespace range_of_function_x_l52_52366

theorem range_of_function_x (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := sorry

end range_of_function_x_l52_52366


namespace remainder_127_14_l52_52994

theorem remainder_127_14 : ∃ r : ℤ, r = 127 - (14 * 9) ∧ r = 1 := by
  sorry

end remainder_127_14_l52_52994


namespace find_k_l52_52982

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k

theorem find_k (k : ℝ) : 
  (f 3 - g 3 k = 6) → k = -23/3 := 
by
  sorry

end find_k_l52_52982


namespace sports_club_membership_l52_52820

theorem sports_club_membership (B T Both Neither : ℕ) (hB : B = 17) (hT : T = 19) (hBoth : Both = 11) (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end sports_club_membership_l52_52820


namespace tensor_value_l52_52346

variables (h : ℝ)

def tensor (x y : ℝ) : ℝ := x^2 - y^2

theorem tensor_value : tensor h (tensor h h) = h^2 :=
by 
-- Complete proof body not required, 'sorry' is used for omitted proof
sorry

end tensor_value_l52_52346


namespace expand_product_l52_52569

theorem expand_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7 * x^3 + 12 := 
  sorry

end expand_product_l52_52569


namespace inflation_over_two_years_real_yield_deposit_second_year_l52_52672

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l52_52672


namespace find_f_f_neg1_l52_52901

def f (x : Int) : Int :=
  if x >= 0 then x + 2 else 1

theorem find_f_f_neg1 : f (f (-1)) = 3 :=
by
  sorry

end find_f_f_neg1_l52_52901


namespace sam_money_left_l52_52370

-- Assuming the cost per dime and quarter
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Given conditions
def dimes : ℕ := 19
def quarters : ℕ := 6
def cost_per_candy_bar_in_dimes : ℕ := 3
def candy_bars : ℕ := 4
def lollipops : ℕ := 1

-- Calculate the initial money in cents
def initial_money : ℕ := (dimes * dime_value) + (quarters * quarter_value)

-- Calculate the cost of candy bars in cents
def candy_bars_cost : ℕ := candy_bars * cost_per_candy_bar_in_dimes * dime_value

-- Calculate the cost of lollipops in cents
def lollipop_cost : ℕ := lollipops * quarter_value

-- Calculate the total cost of purchases in cents
def total_cost : ℕ := candy_bars_cost + lollipop_cost

-- Calculate the final money left in cents
def final_money : ℕ := initial_money - total_cost

-- Theorem to prove
theorem sam_money_left : final_money = 195 := by
  sorry

end sam_money_left_l52_52370


namespace percentage_of_students_owning_birds_l52_52712

theorem percentage_of_students_owning_birds
    (total_students : ℕ) 
    (students_owning_birds : ℕ) 
    (h_total_students : total_students = 500) 
    (h_students_owning_birds : students_owning_birds = 75) : 
    (students_owning_birds * 100) / total_students = 15 := 
by 
    sorry

end percentage_of_students_owning_birds_l52_52712


namespace popsicles_eaten_l52_52157

theorem popsicles_eaten (total_time : ℕ) (interval : ℕ) (p : ℕ)
  (h_total_time : total_time = 6 * 60)
  (h_interval : interval = 20) :
  p = total_time / interval :=
sorry

end popsicles_eaten_l52_52157


namespace complex_imaginary_part_l52_52956

theorem complex_imaginary_part : 
  Complex.im ((1 : ℂ) / (-2 + Complex.I) + (1 : ℂ) / (1 - 2 * Complex.I)) = 1/5 := 
  sorry

end complex_imaginary_part_l52_52956


namespace abby_damon_weight_l52_52068

theorem abby_damon_weight (a' b' c' d' : ℕ) (h1 : a' + b' = 265) (h2 : b' + c' = 250) (h3 : c' + d' = 280) :
  a' + d' = 295 :=
  sorry -- Proof goes here

end abby_damon_weight_l52_52068


namespace erased_length_l52_52197

def original_length := 100 -- in cm
def final_length := 76 -- in cm

theorem erased_length : original_length - final_length = 24 :=
by
    sorry

end erased_length_l52_52197


namespace find_e_l52_52396

theorem find_e (x y e : ℝ) (h1 : x / (2 * y) = 5 / e) (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) : e = 2 := 
by
  sorry

end find_e_l52_52396


namespace factorize_expression_l52_52101

theorem factorize_expression (a b x y : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l52_52101


namespace g_at_9_l52_52523

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l52_52523


namespace find_naturals_divisibility_l52_52968

theorem find_naturals_divisibility :
  {n : ℕ | (2^n + n) ∣ (8^n + n)} = {1, 2, 4, 6} :=
by sorry

end find_naturals_divisibility_l52_52968


namespace arithmetic_sqrt_of_9_l52_52286

def arithmetic_sqrt (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem arithmetic_sqrt_of_9 : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_9_l52_52286


namespace y_coordinate_of_third_vertex_eq_l52_52520

theorem y_coordinate_of_third_vertex_eq (x1 x2 y1 y2 : ℝ)
    (h1 : x1 = 0) 
    (h2 : y1 = 3) 
    (h3 : x2 = 10) 
    (h4 : y2 = 3) 
    (h5 : x1 ≠ x2) 
    (h6 : y1 = y2) 
    : ∃ y3 : ℝ, y3 = 3 + 5 * Real.sqrt 3 := 
by
  sorry

end y_coordinate_of_third_vertex_eq_l52_52520


namespace plates_arrangement_l52_52622

theorem plates_arrangement : 
  let blue := 6
  let red := 3
  let green := 2
  let yellow := 1
  let total_ways_without_rest := Nat.factorial (blue + red + green + yellow - 1) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)
  let green_adj_ways := Nat.factorial (blue + red + green + yellow - 2) / (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * Nat.factorial yellow)
  total_ways_without_rest - green_adj_ways = 22680 
:= sorry

end plates_arrangement_l52_52622


namespace simplify_expression_l52_52150

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end simplify_expression_l52_52150


namespace correct_operations_result_l52_52398

theorem correct_operations_result {n : ℕ} (h₁ : n / 8 - 20 = 12) :
  (n * 8 + 20) = 2068 ∧ 1800 < 2068 ∧ 2068 < 2200 :=
by
  sorry

end correct_operations_result_l52_52398


namespace value_of_stocks_l52_52746

def initial_investment (bonus : ℕ) (stocks : ℕ) : ℕ := bonus / stocks
def final_value_stock_A (initial : ℕ) : ℕ := initial * 2
def final_value_stock_B (initial : ℕ) : ℕ := initial * 2
def final_value_stock_C (initial : ℕ) : ℕ := initial / 2

theorem value_of_stocks 
    (bonus : ℕ) (stocks : ℕ) (h_bonus : bonus = 900) (h_stocks : stocks = 3) : 
    initial_investment bonus stocks * 2 + initial_investment bonus stocks * 2 + initial_investment bonus stocks / 2 = 1350 :=
by
    sorry

end value_of_stocks_l52_52746


namespace weight_of_3_moles_of_BaF2_is_correct_l52_52576

-- Definitions for the conditions
def atomic_weight_Ba : ℝ := 137.33 -- g/mol
def atomic_weight_F : ℝ := 19.00 -- g/mol

-- Definition of the molecular weight of BaF2
def molecular_weight_BaF2 : ℝ := (1 * atomic_weight_Ba) + (2 * atomic_weight_F)

-- The statement to prove
theorem weight_of_3_moles_of_BaF2_is_correct : (3 * molecular_weight_BaF2) = 525.99 :=
by
  -- Proof omitted
  sorry

end weight_of_3_moles_of_BaF2_is_correct_l52_52576


namespace nancy_history_books_l52_52064

/-- Nancy started with 46 books in total on the cart.
    She shelved 8 romance books and 4 poetry books from the top section.
    She shelved 5 Western novels and 6 biographies from the bottom section.
    Half the books on the bottom section were mystery books.
    Prove that Nancy shelved 12 history books.
-/
theorem nancy_history_books 
  (total_books : ℕ)
  (romance_books : ℕ)
  (poetry_books : ℕ)
  (western_novels : ℕ)
  (biographies : ℕ)
  (bottom_books_half_mystery : ℕ)
  (history_books : ℕ) :
  (total_books = 46) →
  (romance_books = 8) →
  (poetry_books = 4) →
  (western_novels = 5) →
  (biographies = 6) →
  (bottom_books_half_mystery = 11) →
  (history_books = total_books - ((romance_books + poetry_books) + (2 * (western_novels + biographies)))) →
  history_books = 12 :=
by
  intros
  sorry

end nancy_history_books_l52_52064


namespace jason_age_at_end_of_2004_l52_52958

noncomputable def jason_age_in_1997 (y : ℚ) (g : ℚ) : Prop :=
  y = g / 3 

noncomputable def birth_years_sum (y : ℚ) (g : ℚ) : Prop :=
  (1997 - y) + (1997 - g) = 3852

theorem jason_age_at_end_of_2004
  (y g : ℚ)
  (h1 : jason_age_in_1997 y g)
  (h2 : birth_years_sum y g) :
  y + 7 = 42.5 :=
by
  sorry

end jason_age_at_end_of_2004_l52_52958


namespace evaluate_expression_l52_52484

variable {x y : ℝ}

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y ^ 2) :
  (x - 1 / x ^ 2) * (y + 2 / y) = 2 * x ^ (5 / 2) - 1 / x := 
by
  sorry

end evaluate_expression_l52_52484


namespace mixed_number_division_l52_52770

theorem mixed_number_division :
  (5 + 1 / 2) / (2 / 11) = 121 / 4 :=
by sorry

end mixed_number_division_l52_52770


namespace smallest_lcm_of_4digit_integers_with_gcd_5_l52_52136

theorem smallest_lcm_of_4digit_integers_with_gcd_5 :
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 1000 ≤ b ∧ b < 10000 ∧ gcd a b = 5 ∧ lcm a b = 201000 :=
by
  sorry

end smallest_lcm_of_4digit_integers_with_gcd_5_l52_52136


namespace range_of_abscissa_l52_52333

/--
Given three points A, F1, F2 in the Cartesian plane and a point P satisfying the given conditions,
prove that the range of the abscissa of point P is [0, 3].

Conditions:
- A = (1, 0)
- F1 = (-2, 0)
- F2 = (2, 0)
- \| overrightarrow{PF1} \| + \| overrightarrow{PF2} \| = 6
- \| overrightarrow{PA} \| ≤ sqrt(6)
-/
theorem range_of_abscissa :
  ∀ (P : ℝ × ℝ),
    (|P.1 + 2| + |P.1 - 2| = 6) →
    ((P.1 - 1)^2 + P.2^2 ≤ 6) →
    (0 ≤ P.1 ∧ P.1 ≤ 3) :=
by
  intros P H1 H2
  sorry

end range_of_abscissa_l52_52333


namespace sum_of_perimeters_l52_52017

theorem sum_of_perimeters (x y z : ℝ) 
    (h_large_triangle_perimeter : 3 * 20 = 60)
    (h_hexagon_perimeter : 60 - (x + y + z) = 40) :
    3 * (x + y + z) = 60 := by
  sorry

end sum_of_perimeters_l52_52017


namespace tangent_line_hyperbola_l52_52858

variable {a b x x₀ y y₀ : ℝ}
variable (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (he : x₀^2 / a^2 + y₀^2 / b^2 = 1)
variable (hh : x₀^2 / a^2 - y₀^2 / b^2 = 1)

theorem tangent_line_hyperbola
  (h_tangent_ellipse : (x₀ * x / a^2 + y₀ * y / b^2 = 1)) :
  (x₀ * x / a^2 - y₀ * y / b^2 = 1) :=
sorry

end tangent_line_hyperbola_l52_52858


namespace total_weight_of_load_l52_52086

def weight_of_crate : ℕ := 4
def weight_of_carton : ℕ := 3
def number_of_crates : ℕ := 12
def number_of_cartons : ℕ := 16

theorem total_weight_of_load :
  number_of_crates * weight_of_crate + number_of_cartons * weight_of_carton = 96 :=
by sorry

end total_weight_of_load_l52_52086


namespace zero_ordered_triples_non_zero_satisfy_conditions_l52_52242

theorem zero_ordered_triples_non_zero_satisfy_conditions :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → a = b + c → b = c + a → c = a + b → a + b + c ≠ 0 :=
by
  sorry

end zero_ordered_triples_non_zero_satisfy_conditions_l52_52242


namespace value_of_x_l52_52884

theorem value_of_x (x : ℕ) : (8^4 + 8^4 + 8^4 = 2^x) → x = 13 :=
by
  sorry

end value_of_x_l52_52884


namespace amount_of_salmon_sold_first_week_l52_52706

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end amount_of_salmon_sold_first_week_l52_52706


namespace distance_from_point_to_y_axis_l52_52407

/-- Proof that the distance from point P(-4, 3) to the y-axis is 4. -/
theorem distance_from_point_to_y_axis {P : ℝ × ℝ} (hP : P = (-4, 3)) : |P.1| = 4 :=
by {
   -- The proof will depend on the properties of absolute value
   -- and the given condition about the coordinates of P.
   sorry
}

end distance_from_point_to_y_axis_l52_52407


namespace prize_behind_door_4_eq_a_l52_52405

theorem prize_behind_door_4_eq_a :
  ∀ (prize : ℕ → ℕ)
    (h_prizes : ∀ i j, 1 ≤ prize i ∧ prize i ≤ 4 ∧ prize i = prize j → i = j)
    (hA1 : prize 1 = 2)
    (hA2 : prize 3 = 3)
    (hB1 : prize 2 = 2)
    (hB2 : prize 3 = 4)
    (hC1 : prize 4 = 2)
    (hC2 : prize 2 = 3)
    (hD1 : prize 4 = 1)
    (hD2 : prize 3 = 3),
    prize 4 = 1 :=
by
  intro prize h_prizes hA1 hA2 hB1 hB2 hC1 hC2 hD1 hD2
  sorry

end prize_behind_door_4_eq_a_l52_52405


namespace gain_percent_l52_52003

theorem gain_percent (CP SP : ℕ) (h1 : CP = 20) (h2 : SP = 25) : 
  (SP - CP) * 100 / CP = 25 := by
  sorry

end gain_percent_l52_52003


namespace cost_per_pack_is_correct_l52_52732

def total_amount_spent : ℝ := 120
def num_packs_bought : ℕ := 6
def expected_cost_per_pack : ℝ := 20

theorem cost_per_pack_is_correct :
  total_amount_spent / num_packs_bought = expected_cost_per_pack :=
  by 
    -- here would be the proof
    sorry

end cost_per_pack_is_correct_l52_52732


namespace find_sum_invested_l52_52535

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end find_sum_invested_l52_52535


namespace fraction_of_remaining_paint_used_l52_52951

theorem fraction_of_remaining_paint_used (total_paint : ℕ) (first_week_fraction : ℚ) (total_used : ℕ) :
  total_paint = 360 ∧ first_week_fraction = 1/6 ∧ total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
  by
    sorry

end fraction_of_remaining_paint_used_l52_52951


namespace doris_hourly_wage_l52_52669

-- Defining the conditions from the problem
def money_needed : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturday_hours_per_day : ℕ := 5
def weeks_needed : ℕ := 3
def weekdays_per_week : ℕ := 5
def saturdays_per_week : ℕ := 1

-- Calculating total hours worked by Doris in 3 weeks
def total_hours (w_hours: ℕ) (s_hours: ℕ) 
    (w_days : ℕ) (s_days : ℕ) (weeks : ℕ) : ℕ := 
    (w_days * w_hours + s_days * s_hours) * weeks

-- Defining the weekly work hours
def weekly_hours := total_hours weekday_hours_per_day saturday_hours_per_day weekdays_per_week saturdays_per_week 1

-- Result of hours worked in 3 weeks
def hours_worked_in_3_weeks := weekly_hours * weeks_needed

-- Define the proof task
theorem doris_hourly_wage : 
  (money_needed : ℕ) / (hours_worked_in_3_weeks : ℕ) = 20 := by 
  sorry

end doris_hourly_wage_l52_52669


namespace find_police_stations_in_pittsburgh_l52_52686

-- Conditions
def stores_in_pittsburgh : ℕ := 2000
def hospitals_in_pittsburgh : ℕ := 500
def schools_in_pittsburgh : ℕ := 200
def total_buildings_in_new_city : ℕ := 2175

-- Define the problem statement and the target proof
theorem find_police_stations_in_pittsburgh (P : ℕ) :
  1000 + 1000 + 150 + (P + 5) = total_buildings_in_new_city → P = 20 :=
by
  sorry

end find_police_stations_in_pittsburgh_l52_52686


namespace nonzero_fraction_power_zero_l52_52256

theorem nonzero_fraction_power_zero (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0) : ((a : ℚ) / b)^0 = 1 := 
by
  -- proof goes here
  sorry

end nonzero_fraction_power_zero_l52_52256


namespace f_sqrt_2_l52_52883

noncomputable def f : ℝ → ℝ :=
sorry

axiom domain_f : ∀ x, 0 < x → 0 < f x
axiom add_property : ∀ x y, f (x * y) = f x + f y
axiom f_at_8 : f 8 = 6

theorem f_sqrt_2 : f (Real.sqrt 2) = 1 :=
by
  have sqrt2pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  sorry

end f_sqrt_2_l52_52883


namespace points_on_line_any_real_n_l52_52473

theorem points_on_line_any_real_n (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  True :=
by
  sorry

end points_on_line_any_real_n_l52_52473


namespace find_a_l52_52399

open Real

def is_chord_length_correct (a : ℝ) : Prop :=
  let x_line := fun t : ℝ => 1 + t
  let y_line := fun t : ℝ => a - t
  let x_circle := fun α : ℝ => 2 + 2 * cos α
  let y_circle := fun α : ℝ => 2 + 2 * sin α
  let distance_from_center := abs (3 - a) / sqrt 2
  let chord_length := 2 * sqrt (4 - distance_from_center ^ 2)
  chord_length = 2 * sqrt 2 

theorem find_a (a : ℝ) : is_chord_length_correct a → a = 1 ∨ a = 5 :=
by
  sorry

end find_a_l52_52399


namespace sqrt_nested_l52_52009

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l52_52009


namespace polygon_sides_from_diagonals_l52_52882

/-- A theorem to prove that a regular polygon with 740 diagonals has 40 sides. -/
theorem polygon_sides_from_diagonals (n : ℕ) (h : (n * (n - 3)) / 2 = 740) : n = 40 := sorry

end polygon_sides_from_diagonals_l52_52882


namespace setC_not_basis_l52_52380

-- Definitions based on the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e₁ e₂ : V)
variables (v₁ v₂ : V)

-- Assuming e₁ and e₂ are non-collinear
axiom non_collinear : ¬Collinear ℝ {e₁, e₂}

-- The vectors in the set C
def setC_v1 : V := 3 • e₁ - 2 • e₂
def setC_v2 : V := 4 • e₂ - 6 • e₁

-- The proof problem statement
theorem setC_not_basis : Collinear ℝ {setC_v1 e₁ e₂, setC_v2 e₁ e₂} :=
sorry

end setC_not_basis_l52_52380


namespace max_radius_of_circle_l52_52529

theorem max_radius_of_circle (c : ℝ × ℝ → Prop) (h1 : c (16, 0)) (h2 : c (-16, 0)) :
  ∃ r : ℝ, r = 16 :=
by
  sorry

end max_radius_of_circle_l52_52529


namespace flat_terrain_length_l52_52301

noncomputable def terrain_distance_equation (x y z : ℝ) : Prop :=
  (x + y + z = 11.5) ∧
  (x / 3 + y / 4 + z / 5 = 2.9) ∧
  (z / 3 + y / 4 + x / 5 = 3.1)

theorem flat_terrain_length (x y z : ℝ) 
  (h : terrain_distance_equation x y z) :
  y = 4 :=
sorry

end flat_terrain_length_l52_52301


namespace find_b_l52_52895

noncomputable def circle1 (x y a : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 5 - a^2 = 0
noncomputable def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - (2*b - 10)*x - 2*b*y + 2*b^2 - 10*b + 16 = 0
def is_intersection (x1 y1 x2 y2 : ℝ) : Prop := x1^2 + y1^2 = x2^2 + y2^2

theorem find_b (a x1 y1 x2 y2 : ℝ) (b : ℝ) :
  (circle1 x1 y1 a) ∧ (circle1 x2 y2 a) ∧ 
  (circle2 x1 y1 b) ∧ (circle2 x2 y2 b) ∧ 
  is_intersection x1 y1 x2 y2 →
  b = 5 / 3 :=
sorry

end find_b_l52_52895


namespace expansion_terms_count_l52_52092

-- Define the number of terms in the first polynomial
def first_polynomial_terms : ℕ := 3

-- Define the number of terms in the second polynomial
def second_polynomial_terms : ℕ := 4

-- Prove that the number of terms in the expansion is 12
theorem expansion_terms_count : first_polynomial_terms * second_polynomial_terms = 12 :=
by
  sorry

end expansion_terms_count_l52_52092


namespace contradiction_proof_l52_52456

theorem contradiction_proof (a b : ℝ) : a + b = 12 → ¬ (a < 6 ∧ b < 6) :=
by
  intro h
  intro h_contra
  sorry

end contradiction_proof_l52_52456


namespace solutions_count_l52_52093

noncomputable def number_of_solutions (a : ℝ) : ℕ :=
if a < 0 then 1
else if 0 ≤ a ∧ a < Real.exp 1 then 0
else if a = Real.exp 1 then 1
else if a > Real.exp 1 then 2
else 0

theorem solutions_count (a : ℝ) :
  (a < 0 ∧ number_of_solutions a = 1) ∨
  (0 ≤ a ∧ a < Real.exp 1 ∧ number_of_solutions a = 0) ∨
  (a = Real.exp 1 ∧ number_of_solutions a = 1) ∨
  (a > Real.exp 1 ∧ number_of_solutions a = 2) :=
by {
  sorry
}

end solutions_count_l52_52093


namespace number_of_elements_l52_52632

def average_incorrect (N : ℕ) := 21
def correction (incorrect : ℕ) (correct : ℕ) := correct - incorrect
def average_correct (N : ℕ) := 22

theorem number_of_elements (N : ℕ) (incorrect : ℕ) (correct : ℕ) :
  average_incorrect N = 21 ∧ incorrect = 26 ∧ correct = 36 ∧ average_correct N = 22 →
  N = 10 :=
by
  sorry

end number_of_elements_l52_52632


namespace sophie_hours_needed_l52_52451

-- Sophie needs 206 hours to finish the analysis of all bones.
theorem sophie_hours_needed (num_bones : ℕ) (time_per_bone : ℕ) (total_hours : ℕ) (h1 : num_bones = 206) (h2 : time_per_bone = 1) : 
  total_hours = num_bones * time_per_bone :=
by
  rw [h1, h2]
  norm_num
  sorry

end sophie_hours_needed_l52_52451


namespace tom_purchases_mangoes_l52_52445

theorem tom_purchases_mangoes (m : ℕ) (h1 : 8 * 70 + m * 65 = 1145) : m = 9 :=
by
  sorry

end tom_purchases_mangoes_l52_52445


namespace quadrilateral_area_proof_l52_52606

-- Definitions of points
def A : (ℝ × ℝ) := (1, 3)
def B : (ℝ × ℝ) := (1, 1)
def C : (ℝ × ℝ) := (3, 1)
def D : (ℝ × ℝ) := (2010, 2011)

-- Function to calculate the area of the quadrilateral
def area_of_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ := 
  let area_triangle (P Q R : (ℝ × ℝ)) : ℝ := 
    0.5 * (P.1 * Q.2 + Q.1 * R.2 + R.1 * P.2 - P.2 * Q.1 - Q.2 * R.1 - R.2 * P.1)
  area_triangle A B C + area_triangle A C D

-- Lean statement to prove the desired area
theorem quadrilateral_area_proof : area_of_quadrilateral A B C D = 7 := 
  sorry

end quadrilateral_area_proof_l52_52606


namespace find_a_equidistant_l52_52558

theorem find_a_equidistant :
  ∀ a : ℝ, (abs (a - 2) = abs (6 - 2 * a)) →
    (a = 8 / 3 ∨ a = 4) :=
by
  intro a h
  sorry

end find_a_equidistant_l52_52558


namespace line_eq_l52_52890

theorem line_eq (x y : ℝ) (point eq_direction_vector) (h₀ : point = (3, -2))
    (h₁ : eq_direction_vector = (-5, 3)) :
    3 * x + 5 * y + 1 = 0 := by sorry

end line_eq_l52_52890


namespace tylenol_mg_per_tablet_l52_52430

noncomputable def dose_intervals : ℕ := 3  -- Mark takes Tylenol 3 times
noncomputable def total_mg : ℕ := 3000     -- Total intake in milligrams
noncomputable def tablets_per_dose : ℕ := 2  -- Number of tablets per dose

noncomputable def tablet_mg : ℕ :=
  total_mg / dose_intervals / tablets_per_dose

theorem tylenol_mg_per_tablet : tablet_mg = 500 := by
  sorry

end tylenol_mg_per_tablet_l52_52430


namespace rental_days_l52_52851

-- Definitions based on conditions
def daily_rate := 30
def weekly_rate := 190
def total_payment := 310

-- Prove that Jennie rented the car for 11 days
theorem rental_days : ∃ d : ℕ, d = 11 ∧ (total_payment = weekly_rate + (d - 7) * daily_rate) ∨ (d < 7 ∧ total_payment = d * daily_rate) :=
by
  sorry

end rental_days_l52_52851


namespace total_boxes_l52_52853
namespace AppleBoxes

theorem total_boxes (initial_boxes : ℕ) (apples_per_box : ℕ) (rotten_apples : ℕ)
  (apples_per_bag : ℕ) (bags_per_box : ℕ) (good_apples : ℕ) (final_boxes : ℕ) :
  initial_boxes = 14 →
  apples_per_box = 105 →
  rotten_apples = 84 →
  apples_per_bag = 6 →
  bags_per_box = 7 →
  final_boxes = (initial_boxes * apples_per_box - rotten_apples) / (apples_per_bag * bags_per_box) →
  final_boxes = 33 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  exact h6

end AppleBoxes

end total_boxes_l52_52853


namespace max_marks_l52_52478

theorem max_marks (M : ℝ) : 0.33 * M = 59 + 40 → M = 300 :=
by
  sorry

end max_marks_l52_52478


namespace quadratic_has_one_real_root_l52_52574

theorem quadratic_has_one_real_root (k : ℝ) : 
  (∃ (x : ℝ), -2 * x^2 + 8 * x + k = 0 ∧ ∀ y, -2 * y^2 + 8 * y + k = 0 → y = x) ↔ k = -8 := 
by
  sorry

end quadratic_has_one_real_root_l52_52574


namespace daily_production_n_l52_52482

theorem daily_production_n (n : ℕ) 
  (h1 : (60 * n) / n = 60)
  (h2 : (60 * n + 90) / (n + 1) = 65) : 
  n = 5 :=
by
  -- Proof goes here
  sorry

end daily_production_n_l52_52482


namespace max_days_for_C_l52_52123

-- Define the durations of the processes and the total project duration
def A := 2
def B := 5
def D := 4
def T := 9

-- Define the condition to prove the maximum days required for process C
theorem max_days_for_C (x : ℕ) (h : 2 + x + 4 = 9) : x = 3 := by
  sorry

end max_days_for_C_l52_52123


namespace tan_B_eq_one_third_l52_52860

theorem tan_B_eq_one_third
  (A B : ℝ)
  (h1 : Real.cos A = 4 / 5)
  (h2 : Real.tan (A - B) = 1 / 3) :
  Real.tan B = 1 / 3 := by
  sorry

end tan_B_eq_one_third_l52_52860


namespace original_number_is_fraction_l52_52449

theorem original_number_is_fraction (x : ℚ) (h : 1 + (1 / x) = 9 / 4) : x = 4 / 5 :=
by
  sorry

end original_number_is_fraction_l52_52449


namespace expression_not_defined_at_x_l52_52244

theorem expression_not_defined_at_x :
  ∃ (x : ℝ), x = 10 ∧ (x^3 - 30 * x^2 + 300 * x - 1000) = 0 := 
sorry

end expression_not_defined_at_x_l52_52244


namespace value_of_expression_l52_52345

variable (a b : ℝ)

theorem value_of_expression : 
  let x := a + b 
  let y := a - b 
  (x - y) * (x + y) = 4 * a * b := 
by
  sorry

end value_of_expression_l52_52345


namespace erasers_per_box_l52_52255

theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (erasers_per_box : ℕ) : total_erasers = 40 → num_boxes = 4 → erasers_per_box = total_erasers / num_boxes → erasers_per_box = 10 :=
by
  intros h_total h_boxes h_div
  rw [h_total, h_boxes] at h_div
  norm_num at h_div
  exact h_div

end erasers_per_box_l52_52255


namespace q_range_l52_52395

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  ∀ y : ℝ, y ∈ Set.range q ↔ 0 ≤ y :=
by sorry

end q_range_l52_52395


namespace garden_length_to_width_ratio_l52_52356

theorem garden_length_to_width_ratio (area : ℕ) (width : ℕ) (h_area : area = 432) (h_width : width = 12) :
  ∃ length : ℕ, length = area / width ∧ (length / width = 3) := 
by
  sorry

end garden_length_to_width_ratio_l52_52356


namespace triangle_sides_inequality_l52_52742

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : a + b + c ≤ 2) :
  -3 < (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) ∧ 
  (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) < 3 :=
by sorry

end triangle_sides_inequality_l52_52742


namespace ages_of_boys_l52_52541

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end ages_of_boys_l52_52541


namespace matroskin_milk_amount_l52_52611

theorem matroskin_milk_amount :
  ∃ S M x : ℝ, S + M = 10 ∧ (S - x) = (1 / 3) * S ∧ (M + x) = 3 * M ∧ (M + x) = 7.5 := 
sorry

end matroskin_milk_amount_l52_52611


namespace find_n_from_ratio_l52_52053

theorem find_n_from_ratio (a b n : ℕ) (h : (a + 3 * b) ^ n = 4 ^ n)
  (h_ratio : 4 ^ n / 2 ^ n = 64) : 
  n = 6 := 
by
  sorry

end find_n_from_ratio_l52_52053


namespace min_value_expression_l52_52809

theorem min_value_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 :=
by
  sorry

end min_value_expression_l52_52809


namespace lowest_temperature_at_noon_l52_52642

theorem lowest_temperature_at_noon
  (L : ℤ) -- Denote lowest temperature as L
  (avg_temp : ℤ) -- Average temperature from Monday to Friday
  (max_range : ℤ) -- Maximum possible range of the temperature
  (h1 : avg_temp = 50) -- Condition 1: average temperature is 50
  (h2 : max_range = 50) -- Condition 2: maximum range is 50
  (total_temp : ℤ) -- Sum of temperatures from Monday to Friday
  (h3 : total_temp = 250) -- Sum of temperatures equals 5 * 50
  (h4 : total_temp = L + (L + 50) + (L + 50) + (L + 50) + (L + 50)) -- Sum represented in terms of L
  : L = 10 := -- Prove that L equals 10
sorry

end lowest_temperature_at_noon_l52_52642


namespace john_labor_cost_l52_52096

def plank_per_tree : ℕ := 25
def table_cost : ℕ := 300
def profit : ℕ := 12000
def trees_chopped : ℕ := 30
def planks_per_table : ℕ := 15
def total_table_revenue := (trees_chopped * plank_per_tree / planks_per_table) * table_cost
def labor_cost := total_table_revenue - profit

theorem john_labor_cost :
  labor_cost = 3000 :=
by
  sorry

end john_labor_cost_l52_52096


namespace min_possible_value_of_box_l52_52938

theorem min_possible_value_of_box
  (c d : ℤ)
  (distinct : c ≠ d)
  (h_cd : c * d = 29) :
  ∃ (box : ℤ), c^2 + d^2 = box ∧ box = 842 :=
by
  sorry

end min_possible_value_of_box_l52_52938


namespace real_solutions_count_is_two_l52_52458

def equation_has_two_real_solutions (a b c : ℝ) : Prop :=
  (3*a^2 - 8*b + 2 = c) → (∀ x : ℝ, 3*x^2 - 8*x + 2 = 0) → ∃! x₁ x₂ : ℝ, (3*x₁^2 - 8*x₁ + 2 = 0) ∧ (3*x₂^2 - 8*x₂ + 2 = 0)

theorem real_solutions_count_is_two : equation_has_two_real_solutions (3 : ℝ) (-8 : ℝ) (2 : ℝ) := by
  sorry

end real_solutions_count_is_two_l52_52458


namespace total_tissues_brought_l52_52074

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l52_52074


namespace ceil_floor_diff_l52_52443

theorem ceil_floor_diff (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ - ⌊x⌋ = 1 := 
by 
  sorry

end ceil_floor_diff_l52_52443


namespace distance_qr_eq_b_l52_52208

theorem distance_qr_eq_b
  (a b c : ℝ)
  (hP : b = c * Real.cosh (a / c))
  (hQ : ∃ Q : ℝ × ℝ, Q = (0, c) ∧ Q.2 = c * Real.cosh (Q.1 / c))
  : QR = b := by
  sorry

end distance_qr_eq_b_l52_52208


namespace number_of_students_like_photography_l52_52302

variable (n_dislike n_like n_neutral : ℕ)

theorem number_of_students_like_photography :
  (3 * n_dislike = n_dislike + 12) →
  (5 * n_dislike = n_like) →
  n_like = 30 :=
by
  sorry

end number_of_students_like_photography_l52_52302


namespace find_x_l52_52505

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (8, 1/2 * x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : vector_a x = (8, 1/2 * x)) 
(h3 : vector_b x = (x, 1)) 
(h4 : ∀ k : ℝ, (vector_a x).1 = k * (vector_b x).1 ∧ 
                       (vector_a x).2 = k * (vector_b x).2) : 
                       x = 4 := sorry

end find_x_l52_52505


namespace investment_time_p_l52_52023

theorem investment_time_p (p_investment q_investment p_profit q_profit : ℝ) (p_invest_time : ℝ) (investment_ratio_pq : p_investment / q_investment = 7 / 5.00001) (profit_ratio_pq : p_profit / q_profit = 7.00001 / 10) (q_invest_time : q_invest_time = 9.999965714374696) : p_invest_time = 50 :=
sorry

end investment_time_p_l52_52023


namespace prove_f_2_eq_3_l52_52942

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then 3 * a ^ x else Real.log (2 * x + 4) / Real.log a

theorem prove_f_2_eq_3 (a : ℝ) (h1 : f 1 a = 6) : f 2 a = 3 :=
by
  -- Define the conditions
  have h1 : 3 * a = 6 := by simp [f] at h1; assumption
  -- Two subcases: x <= 1 and x > 1
  have : a = 2 := by linarith
  simp [f, this]
  sorry

end prove_f_2_eq_3_l52_52942


namespace quadratic_point_value_l52_52936

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h_min : ∀ x : ℝ, a * x^2 + b * x + c ≥ a * (-1)^2 + b * (-1) + c) 
  (h_at_min : a * (-1)^2 + b * (-1) + c = -3)
  (h_point : a * (1)^2 + b * (1) + c = 7) : 
  a * (3)^2 + b * (3) + c = 37 :=
sorry

end quadratic_point_value_l52_52936


namespace fraction_mango_sold_l52_52522

theorem fraction_mango_sold :
  ∀ (choco_total mango_total choco_sold unsold: ℕ) (x : ℚ),
    choco_total = 50 →
    mango_total = 54 →
    choco_sold = (3 * 50) / 5 →
    unsold = 38 →
    (choco_total + mango_total) - (choco_sold + x * mango_total) = unsold →
    x = 4 / 27 :=
by
  intros choco_total mango_total choco_sold unsold x
  sorry

end fraction_mango_sold_l52_52522


namespace minimum_value_w_l52_52542

theorem minimum_value_w : 
  ∀ x y : ℝ, ∃ (w : ℝ), w = 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 → w ≥ 26.25 :=
by
  intro x y
  use 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30
  sorry

end minimum_value_w_l52_52542


namespace percentage_of_salt_in_second_solution_l52_52999

-- Define the data and initial conditions
def original_solution_salt_percentage := 0.15
def replaced_solution_salt_percentage (x: ℝ) := x
def resulting_solution_salt_percentage := 0.16

-- State the question as a theorem
theorem percentage_of_salt_in_second_solution (S : ℝ) (x : ℝ) :
  0.15 * S - 0.0375 * S + x * (S / 4) = 0.16 * S → x = 0.19 :=
by 
  sorry

end percentage_of_salt_in_second_solution_l52_52999


namespace unique_functional_equation_l52_52051

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
sorry

end unique_functional_equation_l52_52051


namespace sum_a_b_c_l52_52349

theorem sum_a_b_c (a b c : ℕ) (h : a = 5 ∧ b = 10 ∧ c = 14) : a + b + c = 29 :=
by
  sorry

end sum_a_b_c_l52_52349


namespace find_cost_price_per_meter_l52_52693

/-- Given that a shopkeeper sells 200 meters of cloth for Rs. 12000 at a loss of Rs. 6 per meter,
we want to find the cost price per meter of cloth. Specifically, we need to prove that the
cost price per meter is Rs. 66. -/
theorem find_cost_price_per_meter
  (total_meters : ℕ := 200)
  (selling_price : ℕ := 12000)
  (loss_per_meter : ℕ := 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 :=
sorry

end find_cost_price_per_meter_l52_52693


namespace range_of_k_decreasing_l52_52914

theorem range_of_k_decreasing (k b : ℝ) (h : ∀ x₁ x₂, x₁ < x₂ → (k^2 - 3*k + 2) * x₁ + b > (k^2 - 3*k + 2) * x₂ + b) : 1 < k ∧ k < 2 :=
by
  -- Proof 
  sorry

end range_of_k_decreasing_l52_52914


namespace common_roots_of_cubic_polynomials_l52_52604

/-- The polynomials \( x^3 + 6x^2 + 11x + 6 \) and \( x^3 + 7x^2 + 14x + 8 \) have two distinct roots in common. -/
theorem common_roots_of_cubic_polynomials :
  ∃ r s : ℝ, r ≠ s ∧ (r^3 + 6 * r^2 + 11 * r + 6 = 0) ∧ (s^3 + 6 * s^2 + 11 * s + 6 = 0)
  ∧ (r^3 + 7 * r^2 + 14 * r + 8 = 0) ∧ (s^3 + 7 * s^2 + 14 * s + 8 = 0) :=
sorry

end common_roots_of_cubic_polynomials_l52_52604


namespace new_average_score_l52_52738

theorem new_average_score (avg_score : ℝ) (num_students : ℕ) (dropped_score : ℝ) (new_num_students : ℕ) :
  num_students = 16 →
  avg_score = 61.5 →
  dropped_score = 24 →
  new_num_students = num_students - 1 →
  (avg_score * num_students - dropped_score) / new_num_students = 64 :=
by
  sorry

end new_average_score_l52_52738


namespace wholesale_price_l52_52911

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l52_52911


namespace parallel_case_perpendicular_case_l52_52277

variables (m : ℝ)
def a := (2, -1)
def b := (-1, m)
def c := (-1, 2)
def sum_ab := (1, m - 1)

-- Parallel case (dot product is zero)
theorem parallel_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = -1 :=
by
  sorry

-- Perpendicular case (dot product is zero)
theorem perpendicular_case : (sum_ab m).fst * c.fst + (sum_ab m).snd * c.snd = 0 ↔ m = 3 / 2 :=
by
  sorry

end parallel_case_perpendicular_case_l52_52277


namespace max_consecutive_integers_sum_lt_1000_l52_52109

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l52_52109


namespace Jimin_scabs_l52_52350

theorem Jimin_scabs (total_scabs : ℕ) (days_in_week : ℕ) (daily_scabs: ℕ)
  (h₁ : total_scabs = 220) (h₂ : days_in_week = 7) 
  (h₃ : daily_scabs = (total_scabs + days_in_week - 1) / days_in_week) : 
  daily_scabs ≥ 32 := by
  sorry

end Jimin_scabs_l52_52350


namespace smallest_positive_integer_n_l52_52550

def contains_digit_9 (n : ℕ) : Prop := 
  ∃ m : ℕ, (10^m) ∣ n ∧ (n / 10^m) % 10 = 9

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (∀ k : ℕ, k > 0 ∧ k < n → 
  (∃ a b : ℕ, k = 2^a * 5^b * 3) ∧ contains_digit_9 k ∧ (k % 3 = 0))
  → n = 90 :=
sorry

end smallest_positive_integer_n_l52_52550


namespace total_distance_of_trail_l52_52417

theorem total_distance_of_trail (a b c d e : ℕ) 
    (h1 : a + b + c = 30) 
    (h2 : b + d = 30) 
    (h3 : d + e = 28) 
    (h4 : a + d = 34) : 
    a + b + c + d + e = 58 := 
sorry

end total_distance_of_trail_l52_52417


namespace value_of_expression_l52_52103

variable (x y : ℝ)

theorem value_of_expression 
  (h1 : x + Real.sqrt (x * y) + y = 9)
  (h2 : x^2 + x * y + y^2 = 27) :
  x - Real.sqrt (x * y) + y = 3 :=
sorry

end value_of_expression_l52_52103


namespace trig_inequality_l52_52900

theorem trig_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.cos β)^2 * (Real.sin β)^2) ≥ 9) := by
  sorry

end trig_inequality_l52_52900


namespace log_fraction_property_l52_52249

noncomputable def log_base (a N : ℝ) : ℝ := Real.log N / Real.log a

theorem log_fraction_property :
  (log_base 3 4 / log_base 9 8) = 4 / 3 :=
by
  sorry

end log_fraction_property_l52_52249


namespace problem_statement_l52_52733

/-!
The problem states:
If |a-2| and |m+n+3| are opposite numbers, then a + m + n = -1.
-/

theorem problem_statement (a m n : ℤ) (h : |a - 2| = -|m + n + 3|) : a + m + n = -1 :=
by {
  sorry
}

end problem_statement_l52_52733


namespace sufficient_but_not_necessary_condition_l52_52759

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end sufficient_but_not_necessary_condition_l52_52759


namespace find_2008_star_2010_l52_52076

-- Define the operation
def operation_star (x y : ℕ) : ℕ := sorry  -- We insert a sorry here because the precise definition is given by the conditions

-- The properties given in the problem
axiom property1 : operation_star 2 2010 = 1
axiom property2 : ∀ n : ℕ, operation_star (2 * (n + 1)) 2010 = 3 * operation_star (2 * n) 2010

-- The main proof statement
theorem find_2008_star_2010 : operation_star 2008 2010 = 3 ^ 1003 :=
by
  -- Here we would provide the proof, but it's omitted.
  sorry

end find_2008_star_2010_l52_52076


namespace remainder_T10_mod_5_l52_52795

noncomputable def T : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => T (n+1) + T n + T n

theorem remainder_T10_mod_5 :
  (T 10) % 5 = 4 :=
sorry

end remainder_T10_mod_5_l52_52795


namespace gain_percent_l52_52913

theorem gain_percent (CP SP : ℝ) (hCP : CP = 110) (hSP : SP = 125) : 
  (SP - CP) / CP * 100 = 13.64 := by
  sorry

end gain_percent_l52_52913


namespace angle_D_measure_l52_52024

theorem angle_D_measure (A B C D : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 35) :
  D = 120 :=
  sorry

end angle_D_measure_l52_52024


namespace find_a2_a3_sequence_constant_general_formula_l52_52474

-- Definition of the sequence and its sum Sn
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom a1_eq : a 1 = 2
axiom S_eq : ∀ n, S (n + 1) = 4 * a n - 2

-- Prove that a_2 = 4 and a_3 = 8
theorem find_a2_a3 : a 2 = 4 ∧ a 3 = 8 :=
sorry

-- Prove that the sequence {a_n - 2a_{n-1}} is constant
theorem sequence_constant {n : ℕ} (hn : n ≥ 2) :
  ∃ c, ∀ k ≥ 2, a k - 2 * a (k - 1) = c :=
sorry

-- Find the general formula for the sequence
theorem general_formula :
  ∀ n, a n = 2^n :=
sorry

end find_a2_a3_sequence_constant_general_formula_l52_52474


namespace area_ratio_greater_than_two_ninths_l52_52557

variable {α : Type*} [LinearOrder α] [LinearOrderedField α]

def area_triangle (A B C : α) : α := sorry -- Placeholder for the area function
noncomputable def triangle_division (A B C P Q R : α) : Prop :=
  -- Placeholder for division condition
  -- Here you would check that P, Q, and R divide the perimeter of triangle ABC into three equal parts
  sorry

theorem area_ratio_greater_than_two_ninths (A B C P Q R : α) :
  triangle_division A B C P Q R → area_triangle P Q R > (2 / 9) * area_triangle A B C :=
by
  sorry -- The proof goes here

end area_ratio_greater_than_two_ninths_l52_52557


namespace coordinate_plane_points_l52_52836

theorem coordinate_plane_points (x y : ℝ) :
    4 * x^2 * y^2 = 4 * x * y + 3 ↔ (x * y = 3 / 2 ∨ x * y = -1 / 2) :=
by 
  sorry

end coordinate_plane_points_l52_52836


namespace quadratic_function_range_l52_52205

theorem quadratic_function_range (x : ℝ) (y : ℝ) (h1 : y = x^2 - 2*x - 3) (h2 : -2 ≤ x ∧ x ≤ 2) :
  -4 ≤ y ∧ y ≤ 5 :=
sorry

end quadratic_function_range_l52_52205


namespace problem_min_value_problem_inequality_range_l52_52965

theorem problem_min_value (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

theorem problem_inequality_range (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) (x : ℝ) :
  (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| ↔ -7 ≤ x ∧ x ≤ 11 :=
sorry

end problem_min_value_problem_inequality_range_l52_52965


namespace sum_of_ages_l52_52819

/-- Given a woman's age is three years more than twice her son's age, 
and the son is 27 years old, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ)
  (h1 : son_age = 27)
  (h2 : woman_age = 3 + 2 * son_age) :
  son_age + woman_age = 84 := 
sorry

end sum_of_ages_l52_52819


namespace number_of_zero_points_l52_52781

theorem number_of_zero_points (f : ℝ → ℝ) (h_odd : ∀ x, f x = -f (-x)) (h_period : ∀ x, f (x - π) = f (x + π)) :
  ∃ (points : Finset ℝ), (∀ x ∈ points, 0 ≤ x ∧ x ≤ 8 ∧ f x = 0) ∧ points.card = 7 :=
by
  sorry

end number_of_zero_points_l52_52781


namespace digit_makes_divisible_by_nine_l52_52464

theorem digit_makes_divisible_by_nine (A : ℕ) : (7 + A + 4 + 6) % 9 = 0 ↔ A = 1 :=
by
  sorry

end digit_makes_divisible_by_nine_l52_52464


namespace floor_add_self_eq_14_5_iff_r_eq_7_5_l52_52050

theorem floor_add_self_eq_14_5_iff_r_eq_7_5 (r : ℝ) : 
  (⌊r⌋ + r = 14.5) ↔ r = 7.5 :=
by
  sorry

end floor_add_self_eq_14_5_iff_r_eq_7_5_l52_52050


namespace find_ordered_pair_l52_52751

theorem find_ordered_pair (x y : ℤ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x - y = (x - 2) + (y - 2))
  : (x, y) = (5, 2) := 
sorry

end find_ordered_pair_l52_52751


namespace train_speed_is_30_kmh_l52_52254

noncomputable def speed_of_train (train_length : ℝ) (cross_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_is_30_kmh :
  speed_of_train 400 59.99520038396929 6 = 30 :=
by
  -- Using the approximation mentioned in the solution, hence no computation proof required.
  sorry

end train_speed_is_30_kmh_l52_52254


namespace range_of_expression_l52_52219

variable (a b c : ℝ)

theorem range_of_expression (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 :=
sorry

end range_of_expression_l52_52219


namespace ratio_of_A_to_B_l52_52713

theorem ratio_of_A_to_B (A B C : ℝ) (hB : B = 270) (hBC : B = (1 / 4) * C) (hSum : A + B + C = 1440) : A / B = 1 / 3 :=
by
  -- The proof is omitted for this example
  sorry

end ratio_of_A_to_B_l52_52713


namespace apples_left_l52_52065

theorem apples_left (initial_apples : ℕ) (ricki_removes : ℕ) (samson_removes : ℕ) 
  (h1 : initial_apples = 74) 
  (h2 : ricki_removes = 14) 
  (h3 : samson_removes = 2 * ricki_removes) : 
  initial_apples - (ricki_removes + samson_removes) = 32 := 
by
  sorry

end apples_left_l52_52065


namespace nails_needed_for_house_wall_l52_52862

theorem nails_needed_for_house_wall
    (large_planks : ℕ)
    (small_planks : ℕ)
    (nails_for_large_planks : ℕ)
    (nails_for_small_planks : ℕ)
    (H1 : large_planks = 12)
    (H2 : small_planks = 10)
    (H3 : nails_for_large_planks = 15)
    (H4 : nails_for_small_planks = 5) :
    (nails_for_large_planks + nails_for_small_planks) = 20 := by
  sorry

end nails_needed_for_house_wall_l52_52862


namespace chess_pieces_missing_l52_52573

theorem chess_pieces_missing 
  (total_pieces : ℕ) (pieces_present : ℕ) (h1 : total_pieces = 32) (h2 : pieces_present = 28) : 
  total_pieces - pieces_present = 4 := 
by
  -- Sorry proof
  sorry

end chess_pieces_missing_l52_52573


namespace inequality_example_l52_52083

theorem inequality_example (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) (h4 : b < 0) : a + b < b + c := 
by sorry

end inequality_example_l52_52083


namespace probability_either_A1_or_B1_not_both_is_half_l52_52133

-- Definitions of the students
inductive Student
| A : ℕ → Student
| B : ℕ → Student
| C : ℕ → Student

-- Excellent grades students
def math_students := [Student.A 1, Student.A 2, Student.A 3]
def physics_students := [Student.B 1, Student.B 2]
def chemistry_students := [Student.C 1, Student.C 2]

-- Total number of ways to select one student from each category
def total_ways : ℕ := 3 * 2 * 2

-- Number of ways either A_1 or B_1 is selected but not both
def special_ways : ℕ := 1 * 1 * 2 + 2 * 1 * 2

-- Probability calculation
def probability := (special_ways : ℚ) / total_ways

-- Theorem to be proven
theorem probability_either_A1_or_B1_not_both_is_half :
  probability = 1 / 2 := by
  sorry

end probability_either_A1_or_B1_not_both_is_half_l52_52133


namespace no_right_obtuse_triangle_l52_52871

theorem no_right_obtuse_triangle :
  ∀ (α β γ : ℝ),
  (α + β + γ = 180) →
  (α = 90 ∨ β = 90 ∨ γ = 90) →
  (α > 90 ∨ β > 90 ∨ γ > 90) →
  false :=
by
  sorry

end no_right_obtuse_triangle_l52_52871


namespace least_subtracted_divisible_by_5_l52_52206

theorem least_subtracted_divisible_by_5 :
  ∃ n : ℕ, (568219 - n) % 5 = 0 ∧ n ≤ 4 ∧ (∀ m : ℕ, m < 4 → (568219 - m) % 5 ≠ 0) :=
sorry

end least_subtracted_divisible_by_5_l52_52206


namespace profit_calculation_l52_52528

theorem profit_calculation (cost_price_per_card_yuan : ℚ) (total_sales_yuan : ℚ)
  (n : ℕ) (sales_price_per_card_yuan : ℚ)
  (h1 : cost_price_per_card_yuan = 0.21)
  (h2 : total_sales_yuan = 14.57)
  (h3 : total_sales_yuan = n * sales_price_per_card_yuan)
  (h4 : sales_price_per_card_yuan ≤ 2 * cost_price_per_card_yuan) :
  (total_sales_yuan - n * cost_price_per_card_yuan = 4.7) :=
by
  sorry

end profit_calculation_l52_52528


namespace students_neither_cs_nor_elec_l52_52662

theorem students_neither_cs_nor_elec
  (total_students : ℕ)
  (cs_students : ℕ)
  (elec_students : ℕ)
  (both_cs_and_elec : ℕ)
  (h_total : total_students = 150)
  (h_cs : cs_students = 90)
  (h_elec : elec_students = 60)
  (h_both : both_cs_and_elec = 20) :
  (total_students - (cs_students + elec_students - both_cs_and_elec) = 20) :=
by
  sorry

end students_neither_cs_nor_elec_l52_52662


namespace logs_left_after_3_hours_l52_52117

theorem logs_left_after_3_hours : 
  ∀ (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (time : ℕ),
  initial_logs = 6 →
  burn_rate = 3 →
  add_rate = 2 →
  time = 3 →
  initial_logs + (add_rate * time) - (burn_rate * time) = 3 := 
by
  intros initial_logs burn_rate add_rate time h1 h2 h3 h4
  sorry

end logs_left_after_3_hours_l52_52117


namespace binom_12_10_eq_66_l52_52252

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l52_52252


namespace additional_width_is_25cm_l52_52247

-- Definitions
def length_of_room_cm := 5000
def width_of_room_cm := 1100
def additional_width_cm := 25
def number_of_tiles := 9000
def side_length_of_tile_cm := 25

-- Statement to prove
theorem additional_width_is_25cm : additional_width_cm = 25 :=
by
  -- The proof is omitted, we assume the proof steps here
  sorry

end additional_width_is_25cm_l52_52247


namespace find_number_of_girls_l52_52889

variable (B G : ℕ)

theorem find_number_of_girls
  (h1 : B = G / 2)
  (h2 : B + G = 90)
  : G = 60 :=
sorry

end find_number_of_girls_l52_52889


namespace inequality_proof_l52_52833

variable (x1 x2 y1 y2 z1 z2 : ℝ)
variable (h0 : 0 < x1)
variable (h1 : 0 < x2)
variable (h2 : x1 * y1 > z1^2)
variable (h3 : x2 * y2 > z2^2)

theorem inequality_proof :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l52_52833


namespace find_R_l52_52087

theorem find_R (R : ℝ) (h_diff : ∃ a b : ℝ, a ≠ b ∧ (a - b = 12 ∨ b - a = 12) ∧ a + b = 2 ∧ a * b = -R) : R = 35 :=
by
  obtain ⟨a, b, h_neq, h_diff_12, h_sum, h_prod⟩ := h_diff
  sorry

end find_R_l52_52087


namespace arithmetic_sequence_sum_condition_l52_52502

variable (a : ℕ → ℤ)

theorem arithmetic_sequence_sum_condition (h1 : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) : 
  a 2 + a 10 = 120 :=
sorry

end arithmetic_sequence_sum_condition_l52_52502


namespace margaret_speed_on_time_l52_52623
-- Import the necessary libraries from Mathlib

-- Define the problem conditions and state the theorem
theorem margaret_speed_on_time :
  ∃ r : ℝ, (∀ d t : ℝ,
    d = 50 * (t - 1/12) ∧
    d = 30 * (t + 1/12) →
    r = d / t) ∧
  r = 37.5 := 
sorry

end margaret_speed_on_time_l52_52623


namespace equivalent_form_l52_52992

theorem equivalent_form (x y : ℝ) (h : y = x + 1/x) :
  (x^4 + x^3 - 3*x^2 + x + 2 = 0) ↔ (x^2 * (y^2 + y - 5) = 0) :=
sorry

end equivalent_form_l52_52992


namespace candy_bars_per_bag_l52_52835

theorem candy_bars_per_bag (total_candy_bars : ℕ) (number_of_bags : ℕ) (h1 : total_candy_bars = 15) (h2 : number_of_bags = 5) : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l52_52835


namespace find_number_l52_52954

theorem find_number (x : ℚ) (h : 0.15 * 0.30 * 0.50 * x = 108) : x = 4800 :=
by
  sorry

end find_number_l52_52954


namespace scientific_notation_of_213_million_l52_52481

theorem scientific_notation_of_213_million : ∃ (n : ℝ), (213000000 : ℝ) = 2.13 * 10^8 :=
by
  sorry

end scientific_notation_of_213_million_l52_52481


namespace average_weight_l52_52270

theorem average_weight (w : ℕ) : 
  (64 < w ∧ w ≤ 67) → w = 66 :=
by sorry

end average_weight_l52_52270


namespace arithmetic_geometric_sequences_l52_52598

noncomputable def geometric_sequence_sum (a q n : ℝ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequences (a : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  S 5 = geometric_sequence_sum a q 5 →
  2 * a * q = 6 + a * q^4 →
  S 5 = -31 / 2 :=
by
  intros hq1 hS5 hAR
  sorry

end arithmetic_geometric_sequences_l52_52598


namespace greatest_integer_third_side_l52_52389

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l52_52389


namespace original_price_of_books_l52_52935

theorem original_price_of_books (purchase_cost : ℝ) (original_price : ℝ) :
  (purchase_cost = 162) →
  (original_price ≤ 100) ∨ 
  (100 < original_price ∧ original_price ≤ 200 ∧ purchase_cost = original_price * 0.9) ∨ 
  (original_price > 200 ∧ purchase_cost = original_price * 0.8) →
  (original_price = 180 ∨ original_price = 202.5) :=
by
  sorry

end original_price_of_books_l52_52935


namespace average_weight_of_three_l52_52002

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l52_52002


namespace two_pow_gt_square_for_n_ge_5_l52_52433

theorem two_pow_gt_square_for_n_ge_5 (n : ℕ) (hn : n ≥ 5) : 2^n > n^2 :=
sorry

end two_pow_gt_square_for_n_ge_5_l52_52433


namespace jerry_clock_reading_l52_52934

noncomputable def clock_reading_after_pills (pills : ℕ) (start_time : ℕ) (interval : ℕ) : ℕ :=
(start_time + (pills - 1) * interval) % 12

theorem jerry_clock_reading :
  clock_reading_after_pills 150 12 5 = 1 :=
by
  sorry

end jerry_clock_reading_l52_52934


namespace trig_problem_l52_52691

-- Translate the conditions and problems into Lean 4:
theorem trig_problem (α : ℝ) (h1 : Real.tan α = 2) :
    (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end trig_problem_l52_52691


namespace derivative_at_neg_one_l52_52730

noncomputable def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_neg_one (a b c : ℝ) (h : (4 * a * 1^3 + 2 * b * 1) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := 
sorry

end derivative_at_neg_one_l52_52730


namespace initial_time_is_11_55_l52_52020

-- Definitions for the conditions
variable (X : ℕ) (Y : ℕ)

def initial_time_shown_by_clock (X Y : ℕ) : Prop :=
  (5 * (18 - X) = 35) ∧ (Y = 60 - 5)

theorem initial_time_is_11_55 (h : initial_time_shown_by_clock X Y) : (X = 11) ∧ (Y = 55) :=
sorry

end initial_time_is_11_55_l52_52020


namespace person_walk_rate_l52_52239

theorem person_walk_rate (v : ℝ) (elevator_speed : ℝ) (length : ℝ) (time : ℝ) 
  (h1 : elevator_speed = 10) 
  (h2 : length = 112) 
  (h3 : time = 8) 
  (h4 : length = (v + elevator_speed) * time) 
  : v = 4 :=
by 
  sorry

end person_walk_rate_l52_52239


namespace parabola_maximum_value_l52_52964

noncomputable def maximum_parabola (a b c : ℝ) (h := -b / (2*a)) (k := a * h^2 + b * h + c) : Prop :=
  ∀ (x : ℝ), a ≠ 0 → b = 12 → c = 4 → a = -3 → k = 16

theorem parabola_maximum_value : maximum_parabola (-3) 12 4 :=
by
  sorry

end parabola_maximum_value_l52_52964


namespace correct_expression_l52_52671

theorem correct_expression (x : ℝ) :
  (x^3 / x^2 = x) :=
by sorry

end correct_expression_l52_52671


namespace age_product_difference_l52_52476

theorem age_product_difference 
  (age_today : ℕ) 
  (Arnold_age : age_today = 6) 
  (Danny_age : age_today = 6) : 
  (7 * 7) - (6 * 6) = 13 := 
by
  sorry

end age_product_difference_l52_52476


namespace area_of_perpendicular_triangle_l52_52375

theorem area_of_perpendicular_triangle 
  (S R d : ℝ) (S' : ℝ) -- defining the variables and constants
  (h1 : S > 0) (h2 : R > 0) (h3 : d ≥ 0) :
  S' = (S / 4) * |1 - (d^2 / R^2)| := 
sorry

end area_of_perpendicular_triangle_l52_52375


namespace min_score_needed_l52_52040

/-- 
Given the list of scores and the targeted increase in the average score,
ascertain that the minimum score required on the next test to achieve the
new average is 110.
 -/
theorem min_score_needed 
  (scores : List ℝ) 
  (target_increase : ℝ) 
  (new_score : ℝ) 
  (total_scores : ℝ)
  (current_average : ℝ) 
  (target_average : ℝ) 
  (needed_score : ℝ) :
  (total_scores = 86 + 92 + 75 + 68 + 88 + 84) ∧
  (current_average = total_scores / 6) ∧
  (target_average = current_average + target_increase) ∧
  (new_score = total_scores + needed_score) ∧
  (target_average = new_score / 7) ->
  needed_score = 110 :=
by
  sorry

end min_score_needed_l52_52040


namespace leading_coefficient_of_f_l52_52794

noncomputable def polynomial : Type := ℕ → ℝ

def satisfies_condition (f : polynomial) : Prop :=
  ∀ (x : ℕ), f (x + 1) - f x = 6 * x + 4

theorem leading_coefficient_of_f (f : polynomial) (h : satisfies_condition f) : 
  ∃ a b c : ℝ, (∀ (x : ℕ), f x = a * (x^2) + b * x + c) ∧ a = 3 := 
by
  sorry

end leading_coefficient_of_f_l52_52794


namespace periodic_sequences_zero_at_two_l52_52402

variable {R : Type*} [AddGroup R]

def seq_a (a b : ℕ → R) (n : ℕ) : Prop := a (n + 1) = a n + b n
def seq_b (b c : ℕ → R) (n : ℕ) : Prop := b (n + 1) = b n + c n
def seq_c (c d : ℕ → R) (n : ℕ) : Prop := c (n + 1) = c n + d n
def seq_d (d a : ℕ → R) (n : ℕ) : Prop := d (n + 1) = d n + a n

theorem periodic_sequences_zero_at_two
  (a b c d : ℕ → R)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (ha : ∀ n, seq_a a b n)
  (hb : ∀ n, seq_b b c n)
  (hc : ∀ n, seq_c c d n)
  (hd : ∀ n, seq_d d a n)
  (kra : a (k + m) = a m)
  (krb : b (k + m) = b m)
  (krc : c (k + m) = c m)
  (krd : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := sorry

end periodic_sequences_zero_at_two_l52_52402


namespace find_tangent_points_l52_52796

-- Step a: Define the curve and the condition for the tangent line parallel to y = 4x.
def curve (x : ℝ) : ℝ := x^3 + x - 2
def tangent_slope : ℝ := 4

-- Step d: Provide the statement that the coordinates of P₀ are (1, 0) and (-1, -4).
theorem find_tangent_points : 
  ∃ (P₀ : ℝ × ℝ), (curve P₀.1 = P₀.2) ∧ 
                 ((P₀ = (1, 0)) ∨ (P₀ = (-1, -4))) := 
by
  sorry

end find_tangent_points_l52_52796


namespace haniMoreSitupsPerMinute_l52_52052

-- Define the conditions given in the problem
def totalSitups : Nat := 110
def situpsByDiana : Nat := 40
def rateDianaPerMinute : Nat := 4

-- Define the derived conditions from the solution steps
def timeDianaMinutes := situpsByDiana / rateDianaPerMinute -- 10 minutes
def situpsByHani := totalSitups - situpsByDiana -- 70 situps
def rateHaniPerMinute := situpsByHani / timeDianaMinutes -- 7 situps per minute

-- The theorem we need to prove
theorem haniMoreSitupsPerMinute : rateHaniPerMinute - rateDianaPerMinute = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end haniMoreSitupsPerMinute_l52_52052


namespace union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l52_52496

def U : Set ℕ := { x | 1 ≤ x ∧ x < 9 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}
def C (S : Set ℕ) : Set ℕ := U \ S

theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6} := 
by {
  -- proof here
  sorry
}

theorem inter_A_B : A ∩ B = {3} := 
by {
  -- proof here
  sorry
}

theorem C_U_union_A_B : C (A ∪ B) = {7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_inter_A_B : C (A ∩ B) = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_A : C A = {4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_B : C B = {1, 2, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem union_C_U_A_C_U_B : C A ∪ C B = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem inter_C_U_A_C_U_B : C A ∩ C B = {7, 8} := 
by {
  -- proof here
  sorry
}

end union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l52_52496


namespace ratio_of_investments_l52_52089

variable (A B C : ℝ) (k : ℝ)

-- Conditions
def investments_ratio := (6 * k + 5 * k + 4 * k = 7250) ∧ (5 * k - 6 * k = 250)

-- Theorem we need to prove
theorem ratio_of_investments (h : investments_ratio k) : (A / B = 6 / 5) ∧ (B / C = 5 / 4) := 
  sorry

end ratio_of_investments_l52_52089


namespace quadratic_function_incorrect_statement_l52_52849

theorem quadratic_function_incorrect_statement (x : ℝ) : 
  ∀ y : ℝ, y = -(x + 2)^2 - 1 → ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y = 0 ∧ -(x1 + 2)^2 - 1 = 0 ∧ -(x2 + 2)^2 - 1 = 0) :=
by 
sorry

end quadratic_function_incorrect_statement_l52_52849


namespace hens_ratio_l52_52463

theorem hens_ratio
  (total_chickens : ℕ)
  (fraction_roosters : ℚ)
  (chickens_not_laying : ℕ)
  (h : total_chickens = 80)
  (fr : fraction_roosters = 1/4)
  (cnl : chickens_not_laying = 35) :
  (total_chickens * (1 - fraction_roosters) - chickens_not_laying) / (total_chickens * (1 - fraction_roosters)) = 5 / 12 :=
by
  sorry

end hens_ratio_l52_52463


namespace integer_divisibility_l52_52727

theorem integer_divisibility
  (x y z : ℤ)
  (h : 11 ∣ (7 * x + 2 * y - 5 * z)) :
  11 ∣ (3 * x - 7 * y + 12 * z) :=
sorry

end integer_divisibility_l52_52727


namespace carpet_needed_in_sq_yards_l52_52091

theorem carpet_needed_in_sq_yards :
  let length := 15
  let width := 10
  let area_sq_feet := length * width
  let conversion_factor := 9
  let area_sq_yards := area_sq_feet / conversion_factor
  area_sq_yards = 16.67 := by
  sorry

end carpet_needed_in_sq_yards_l52_52091


namespace isosceles_triangle_perimeter_l52_52331

theorem isosceles_triangle_perimeter 
  (a b c : ℝ)  (h_iso : a = b ∨ b = c ∨ c = a)
  (h_len1 : a = 4 ∨ b = 4 ∨ c = 4)
  (h_len2 : a = 9 ∨ b = 9 ∨ c = 9)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 22 :=
sorry

end isosceles_triangle_perimeter_l52_52331


namespace solve_for_x_opposites_l52_52145

theorem solve_for_x_opposites (x : ℝ) (h : -2 * x = -(3 * x - 1)) : x = 1 :=
by {
  sorry
}

end solve_for_x_opposites_l52_52145


namespace problem1_part1_problem1_part2_l52_52414

open Set Real

theorem problem1_part1 (a : ℝ) (h1: a = 5) :
  let A := { x : ℝ | (x - 6) * (x - 2 * a - 5) > 0 }
  let B := { x : ℝ | (a ^ 2 + 2 - x) * (2 * a - x) < 0 }
  A ∩ B = { x | 15 < x ∧ x < 27 } := sorry

theorem problem1_part2 (a : ℝ) (h2: a > 1 / 2) :
  let A := { x : ℝ | x < 6 ∨ x > 2 * a + 5 }
  let B := { x : ℝ | 2 * a < x ∧ x < a ^ 2 + 2 }
  (∀ x, x ∈ A → x ∈ B) ∧ ¬ (∀ x, x ∈ B → x ∈ A) → (1 / 2 < a ∧ a ≤ 2) := sorry

end problem1_part1_problem1_part2_l52_52414


namespace only_solution_l52_52609

theorem only_solution (x : ℝ) : (3 / (x - 3) = 5 / (x - 5)) ↔ (x = 0) := 
sorry

end only_solution_l52_52609


namespace find_M_l52_52703

variable (M : ℕ)

theorem find_M (h : (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M) : M = 1003 :=
sorry

end find_M_l52_52703


namespace calculate_expression_l52_52435

theorem calculate_expression :
  |-2*Real.sqrt 3| - (1 - Real.pi)^0 + 2*Real.cos (Real.pi / 6) + (1 / 4)^(-1 : ℤ) = 3 * Real.sqrt 3 + 3 :=
by
  sorry

end calculate_expression_l52_52435


namespace horizontal_distance_l52_52401

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - x^2 - x - 6

-- Condition: y-coordinate of point P is 8
def P_y : ℝ := 8

-- Condition: y-coordinate of point Q is -8
def Q_y : ℝ := -8

-- x-coordinates of points P and Q solve these equations respectively
def P_satisfies (x : ℝ) : Prop := curve x = P_y
def Q_satisfies (x : ℝ) : Prop := curve x = Q_y

-- The horizontal distance between P and Q is 1
theorem horizontal_distance : ∃ (Px Qx : ℝ), P_satisfies Px ∧ Q_satisfies Qx ∧ |Px - Qx| = 1 :=
by
  sorry

end horizontal_distance_l52_52401


namespace student_comprehensive_score_l52_52358

def comprehensive_score (t_score i_score d_score : ℕ) (t_ratio i_ratio d_ratio : ℕ) :=
  (t_score * t_ratio + i_score * i_ratio + d_score * d_ratio) / (t_ratio + i_ratio + d_ratio)

theorem student_comprehensive_score :
  comprehensive_score 95 88 90 2 5 3 = 90 :=
by
  -- The proof goes here
  sorry

end student_comprehensive_score_l52_52358


namespace cos_double_angle_l52_52165

open Real

theorem cos_double_angle (α : ℝ) (h0 : 0 < α ∧ α < π) (h1 : sin α + cos α = 1 / 2) : cos (2 * α) = -sqrt 7 / 4 :=
by
  sorry

end cos_double_angle_l52_52165


namespace circle_convex_polygons_count_l52_52608

theorem circle_convex_polygons_count : 
  let total_subsets := (2^15 - 1) - (15 + 105 + 455 + 255)
  let final_count := total_subsets - 500
  final_count = 31437 :=
by
  sorry

end circle_convex_polygons_count_l52_52608


namespace distance_traveled_in_20_seconds_l52_52258

-- Define the initial distance, common difference, and total time
def initial_distance : ℕ := 8
def common_difference : ℕ := 9
def total_time : ℕ := 20

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := initial_distance + (n - 1) * common_difference

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_terms (n : ℕ) : ℕ := n * (initial_distance + nth_term n) / 2

-- The main theorem to be proven
theorem distance_traveled_in_20_seconds : sum_of_terms 20 = 1870 := 
by sorry

end distance_traveled_in_20_seconds_l52_52258


namespace largest_possible_integer_smallest_possible_integer_l52_52216

theorem largest_possible_integer : 3 * (15 + 20 / 4 + 1) = 63 := by
  sorry

theorem smallest_possible_integer : (3 * 15 + 20) / (4 + 1) = 13 := by
  sorry

end largest_possible_integer_smallest_possible_integer_l52_52216


namespace angela_problems_l52_52220

theorem angela_problems (total_problems martha_problems : ℕ) (jenna_problems mark_problems : ℕ) 
    (h1 : total_problems = 20) 
    (h2 : martha_problems = 2)
    (h3 : jenna_problems = 4 * martha_problems - 2)
    (h4 : mark_problems = jenna_problems / 2) :
    total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := 
sorry

end angela_problems_l52_52220


namespace abs_value_product_l52_52714

theorem abs_value_product (x : ℝ) (h : |x - 5| - 4 = 0) : ∃ y z, (y - 5 = 4 ∨ y - 5 = -4) ∧ (z - 5 = 4 ∨ z - 5 = -4) ∧ y * z = 9 :=
by 
  sorry

end abs_value_product_l52_52714


namespace tank_fill_time_l52_52944

theorem tank_fill_time (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1/30) (hB : B_rate = 1/20) (hC : C_rate = -1/40) : 
  1 / (A_rate + B_rate + C_rate) = 120 / 7 :=
by
  -- proof goes here
  sorry

end tank_fill_time_l52_52944


namespace sum_of_tens_and_units_digit_l52_52236

theorem sum_of_tens_and_units_digit (n : ℕ) (h : n = 11^2004 - 5) : 
  (n % 100 / 10) + (n % 10) = 9 :=
by
  sorry

end sum_of_tens_and_units_digit_l52_52236


namespace positive_quadratic_expression_l52_52222

theorem positive_quadratic_expression (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + 4 + m > 0) ↔ (- (Real.sqrt 55) / 2 < m ∧ m < (Real.sqrt 55) / 2) := 
sorry

end positive_quadratic_expression_l52_52222


namespace number_of_ordered_tuples_l52_52831

noncomputable def count_tuples 
  (a1 a2 a3 a4 : ℕ) 
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2): ℕ :=
40

theorem number_of_ordered_tuples 
  (a1 a2 a3 a4 : ℕ)
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2) : 
  count_tuples a1 a2 a3 a4 H_distinct H_range H_eqn = 40 :=
sorry

end number_of_ordered_tuples_l52_52831


namespace functional_relationship_max_daily_profit_price_reduction_1200_profit_l52_52409

noncomputable def y : ℝ → ℝ := λ x => -2 * x^2 + 60 * x + 800

theorem functional_relationship :
  ∀ x : ℝ, y x = (40 - x) * (20 + 2 * x) := 
by
  intro x
  sorry

theorem max_daily_profit :
  y 15 = 1250 :=
by
  sorry

theorem price_reduction_1200_profit :
  ∀ x : ℝ, y x = 1200 → x = 10 ∨ x = 20 :=
by
  intro x
  sorry

end functional_relationship_max_daily_profit_price_reduction_1200_profit_l52_52409


namespace expression_bounds_l52_52363

theorem expression_bounds (a b c d : ℝ) (h0a : 0 ≤ a) (h1a : a ≤ 1) (h0b : 0 ≤ b) (h1b : b ≤ 1)
  (h0c : 0 ≤ c) (h1c : c ≤ 1) (h0d : 0 ≤ d) (h1d : d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ∧
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end expression_bounds_l52_52363


namespace circle_area_from_diameter_endpoints_l52_52230

theorem circle_area_from_diameter_endpoints :
  let C := (-2, 3)
  let D := (4, -1)
  let diameter := Real.sqrt ((4 - (-2))^2 + ((-1) - 3)^2)
  let radius := diameter / 2
  let area := Real.pi * radius^2
  C = (-2, 3) ∧ D = (4, -1) → area = 13 * Real.pi := by
    sorry

end circle_area_from_diameter_endpoints_l52_52230


namespace prove_b_eq_d_and_c_eq_e_l52_52765

variable (a b c d e f : ℕ)

-- Define the expressions for A and B as per the problem statement
def A := 10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f
def B := 10^5 * f + 10^4 * d + 10^3 * e + 10^2 * b + 10 * c + a

-- Define the condition that A - B is divisible by 271
def divisible_by_271 (n : ℕ) : Prop := ∃ k : ℕ, n = 271 * k

-- Define the main theorem to prove b = d and c = e under the given conditions
theorem prove_b_eq_d_and_c_eq_e
    (h1 : divisible_by_271 (A a b c d e f - B a b c d e f)) :
    b = d ∧ c = e :=
sorry

end prove_b_eq_d_and_c_eq_e_l52_52765


namespace smallest_circle_tangent_to_line_and_circle_l52_52680

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the original circle equation as a condition
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 2 * y = 0

-- Define the smallest circle equation as a condition
def smallest_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- The main lemma to prove that the smallest circle's equation matches the expected result
theorem smallest_circle_tangent_to_line_and_circle :
  (∀ x y, line_eq x y → smallest_circle_eq x y) ∧ (∀ x y, circle_eq x y → smallest_circle_eq x y) :=
by
  sorry -- Proof is omitted, as instructed

end smallest_circle_tangent_to_line_and_circle_l52_52680


namespace max_product_two_integers_sum_300_l52_52952

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l52_52952


namespace seahawks_field_goals_l52_52779

-- Defining the conditions as hypotheses
def final_score_seahawks : ℕ := 37
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3
def touchdowns_seahawks : ℕ := 4

-- Stating the goal to prove
theorem seahawks_field_goals : 
  (final_score_seahawks - touchdowns_seahawks * points_per_touchdown) / points_per_fieldgoal = 3 := 
by 
  sorry

end seahawks_field_goals_l52_52779


namespace problem_2011_Mentougou_l52_52274

theorem problem_2011_Mentougou 
  (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (H2 : ∀ x : ℝ, 0 < x → 0 < f x) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
sorry

end problem_2011_Mentougou_l52_52274


namespace distance_between_points_is_sqrt_5_l52_52976

noncomputable def distance_between_polar_points : ℝ :=
  let xA := 1 * Real.cos (3/4 * Real.pi)
  let yA := 1 * Real.sin (3/4 * Real.pi)
  let xB := 2 * Real.cos (Real.pi / 4)
  let yB := 2 * Real.sin (Real.pi / 4)
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

theorem distance_between_points_is_sqrt_5 :
  distance_between_polar_points = Real.sqrt 5 :=
by
  sorry

end distance_between_points_is_sqrt_5_l52_52976


namespace school_students_count_l52_52415

def students_in_school (c n : ℕ) : ℕ := n * c

theorem school_students_count
  (c n : ℕ)
  (h1 : n * c = (n - 6) * (c + 5))
  (h2 : n * c = (n - 16) * (c + 20)) :
  students_in_school c n = 900 :=
by
  sorry

end school_students_count_l52_52415


namespace ratio_hours_per_day_l52_52539

theorem ratio_hours_per_day 
  (h₁ : ∀ h : ℕ, h * 30 = 1200 + (h - 40) * 45 → 40 ≤ h ∧ 6 * 3 ≤ 40)
  (h₂ : 6 * 3 + (x - 6 * 3) / 2 = 24)
  (h₃ : x = 1290) :
  (24 / 2) / 6 = 2 := 
by
  sorry

end ratio_hours_per_day_l52_52539


namespace exists_m_n_for_d_l52_52378

theorem exists_m_n_for_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) := 
sorry

end exists_m_n_for_d_l52_52378


namespace common_ratio_geom_series_l52_52740

theorem common_ratio_geom_series 
  (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 4 / 7) 
  (h₂ : a₂ = 20 / 21) :
  ∃ r : ℚ, r = 5 / 3 ∧ a₂ / a₁ = r := 
sorry

end common_ratio_geom_series_l52_52740


namespace age_difference_l52_52865

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l52_52865


namespace solve_system_of_equations_l52_52924

theorem solve_system_of_equations :
  ∃ (x y: ℝ), (x - y - 1 = 0) ∧ (4 * (x - y) - y = 0) ∧ (x = 5) ∧ (y = 4) :=
by
  sorry

end solve_system_of_equations_l52_52924


namespace inequality_solution_set_l52_52156

theorem inequality_solution_set :
  ∀ x : ℝ, 8 * x^3 + 9 * x^2 + 7 * x - 6 < 0 ↔ (( -6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)) :=
sorry

end inequality_solution_set_l52_52156


namespace time_after_12345_seconds_is_13_45_45_l52_52049

def seconds_in_a_minute := 60
def minutes_in_an_hour := 60
def initial_hour := 10
def initial_minute := 45
def initial_second := 0
def total_seconds := 12345

def time_after_seconds (hour minute second : Nat) (elapsed_seconds : Nat) : (Nat × Nat × Nat) :=
  let total_initial_seconds := hour * 3600 + minute * 60 + second
  let total_final_seconds := total_initial_seconds + elapsed_seconds
  let final_hour := total_final_seconds / 3600
  let remaining_seconds_after_hour := total_final_seconds % 3600
  let final_minute := remaining_seconds_after_hour / 60
  let final_second := remaining_seconds_after_hour % 60
  (final_hour, final_minute, final_second)

theorem time_after_12345_seconds_is_13_45_45 :
  time_after_seconds initial_hour initial_minute initial_second total_seconds = (13, 45, 45) :=
by
  sorry

end time_after_12345_seconds_is_13_45_45_l52_52049


namespace Nancy_needs_5_loads_l52_52351

/-- Definition of the given problem conditions. -/
def pieces_of_clothing (shirts sweaters socks jeans : ℕ) : ℕ :=
  shirts + sweaters + socks + jeans

def washing_machine_capacity : ℕ := 12

def loads_required (total_clothing capacity : ℕ) : ℕ :=
  (total_clothing + capacity - 1) / capacity -- integer division with rounding up

/-- Theorem statement. -/
theorem Nancy_needs_5_loads :
  loads_required (pieces_of_clothing 19 8 15 10) washing_machine_capacity = 5 :=
by
  -- Insert proof here when needed.
  sorry

end Nancy_needs_5_loads_l52_52351


namespace find_x_l52_52080

theorem find_x :
  ∀ x : ℝ, (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
  8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) →
  x = 1486 / 225 :=
by
  sorry

end find_x_l52_52080


namespace john_change_proof_l52_52602

def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

def cost_of_candy_bar : ℕ := 131
def quarters_paid : ℕ := 4
def dimes_paid : ℕ := 3
def nickels_paid : ℕ := 1

def total_payment : ℕ := (quarters_paid * quarter_value) + (dimes_paid * dime_value) + (nickels_paid * nickel_value)
def change_received : ℕ := total_payment - cost_of_candy_bar

theorem john_change_proof : change_received = 4 :=
by
  -- Proof will be provided here
  sorry

end john_change_proof_l52_52602


namespace cousins_initial_money_l52_52955

theorem cousins_initial_money (x : ℕ) :
  let Carmela_initial := 7
  let num_cousins := 4
  let gift_each := 1
  Carmela_initial - num_cousins * gift_each = x + gift_each →
  x = 2 :=
by
  intro h
  sorry

end cousins_initial_money_l52_52955


namespace cody_initial_money_l52_52012

variable (x : ℤ)

theorem cody_initial_money :
  (x + 9 - 19 = 35) → (x = 45) :=
by
  intro h
  sorry

end cody_initial_money_l52_52012


namespace neg_p_equiv_l52_52559

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x / (x - 1) > 0

theorem neg_p_equiv :
  ¬p I ↔ ∃ x ∈ I, x / (x - 1) ≤ 0 ∨ x - 1 = 0 :=
by
  sorry

end neg_p_equiv_l52_52559


namespace gallons_per_hour_l52_52629

-- Define conditions
def total_runoff : ℕ := 240000
def days : ℕ := 10
def hours_per_day : ℕ := 24

-- Define the goal: proving the sewers handle 1000 gallons of run-off per hour
theorem gallons_per_hour : (total_runoff / (days * hours_per_day)) = 1000 :=
by
  -- Proof can be inserted here
  sorry

end gallons_per_hour_l52_52629


namespace largest_multiple_of_7_negation_greater_than_neg_150_l52_52875

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l52_52875


namespace lambda_property_l52_52947
open Int

noncomputable def lambda : ℝ := 1 + Real.sqrt 2

theorem lambda_property (n : ℕ) (hn : n > 0) :
  2 * ⌊lambda * n⌋ = 1 - n + ⌊lambda * ⌊lambda * n⌋⌋ :=
sorry

end lambda_property_l52_52947


namespace solution_set_of_inequality_l52_52647

theorem solution_set_of_inequality (x: ℝ) : 
  (1 / x ≤ 1) ↔ (x < 0 ∨ x ≥ 1) :=
sorry

end solution_set_of_inequality_l52_52647


namespace complex_expression_identity_l52_52800

noncomputable section

variable (x y : ℂ) 

theorem complex_expression_identity (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x^2 + x*y + y^2 = 0) : 
  (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 := 
by 
  sorry

end complex_expression_identity_l52_52800


namespace sum_over_positive_reals_nonnegative_l52_52371

theorem sum_over_positive_reals_nonnegative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (b + c - 2 * a) / (a^2 + b * c) + 
  (c + a - 2 * b) / (b^2 + c * a) + 
  (a + b - 2 * c) / (c^2 + a * b) ≥ 0 :=
sorry

end sum_over_positive_reals_nonnegative_l52_52371


namespace true_proposition_l52_52237

variable (p q : Prop)
variable (hp : p = true)
variable (hq : q = false)

theorem true_proposition : (¬p ∨ ¬q) = true := by
  sorry

end true_proposition_l52_52237


namespace bus_distance_time_relation_l52_52592

theorem bus_distance_time_relation (t : ℝ) :
    (0 ≤ t ∧ t ≤ 1 → s = 60 * t) ∧
    (1 < t ∧ t ≤ 1.5 → s = 60) ∧
    (1.5 < t ∧ t ≤ 2.5 → s = 80 * (t - 1.5) + 60) :=
sorry

end bus_distance_time_relation_l52_52592


namespace bc_fraction_ad_l52_52465

theorem bc_fraction_ad
  (B C E A D : Type)
  (on_AD : ∀ P : Type, P = B ∨ P = C ∨ P = E)
  (AB BD AC CD DE EA: ℝ)
  (h1 : AB = 3 * BD)
  (h2 : AC = 5 * CD)
  (h3 : DE = 2 * EA)

  : ∃ BC AD: ℝ, BC = 1 / 12 * AD := 
sorry -- Proof is omitted

end bc_fraction_ad_l52_52465


namespace speed_conversion_l52_52177

theorem speed_conversion (speed_kmh : ℝ) (conversion_factor : ℝ) :
  speed_kmh = 1.3 → conversion_factor = (1000 / 3600) → speed_kmh * conversion_factor = 0.3611 :=
by
  intros h_speed h_factor
  rw [h_speed, h_factor]
  norm_num
  sorry

end speed_conversion_l52_52177


namespace minimum_discount_l52_52718

open Real

theorem minimum_discount (CP MP SP_min : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  CP = 800 ∧ MP = 1200 ∧ SP_min = 960 ∧ profit_margin = 0.20 ∧
  MP * (1 - discount / 100) ≥ SP_min → discount = 20 :=
by
  intros h
  rcases h with ⟨h_cp, h_mp, h_sp_min, h_profit_margin, h_selling_price⟩
  simp [h_cp, h_mp, h_sp_min, h_profit_margin, sub_eq_self, div_eq_self] at *
  sorry

end minimum_discount_l52_52718


namespace achieve_target_ratio_l52_52633

-- Initial volume and ratio
def initial_volume : ℕ := 20
def initial_milk_ratio : ℕ := 3
def initial_water_ratio : ℕ := 2

-- Mixture removal and addition
def removal_volume : ℕ := 10
def added_milk : ℕ := 10

-- Target ratio of milk to water
def target_milk_ratio : ℕ := 9
def target_water_ratio : ℕ := 1

-- Number of operations required
def operations_needed: ℕ := 2

-- Statement of proof problem
theorem achieve_target_ratio :
  (initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) + added_milk * operations_needed) / 
  (initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) = target_milk_ratio :=
sorry

end achieve_target_ratio_l52_52633


namespace cos_neg_pi_over_3_l52_52572

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l52_52572


namespace least_sum_exponents_of_520_l52_52582

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l52_52582


namespace y_coord_of_third_vertex_of_equilateral_l52_52160

/-- Given two vertices of an equilateral triangle at (0, 6) and (10, 6), and the third vertex in the first quadrant,
    prove that the y-coordinate of the third vertex is 6 + 5 * sqrt 3. -/
theorem y_coord_of_third_vertex_of_equilateral (A B C : ℝ × ℝ)
  (hA : A = (0, 6)) (hB : B = (10, 6)) (hAB : dist A B = 10) (hC : C.2 > 6):
  C.2 = 6 + 5 * Real.sqrt 3 :=
sorry

end y_coord_of_third_vertex_of_equilateral_l52_52160


namespace race_track_radius_l52_52376

theorem race_track_radius (C_inner : ℝ) (width : ℝ) (r_outer : ℝ) : 
  C_inner = 440 ∧ width = 14 ∧ r_outer = (440 / (2 * Real.pi) + 14) → r_outer = 84 :=
by
  intros
  sorry

end race_track_radius_l52_52376


namespace slope_of_line_l52_52702

noncomputable def line_equation (x y : ℝ) : Prop := 4 * y + 2 * x = 10

theorem slope_of_line (x y : ℝ) (h : line_equation x y) : -1 / 2 = -1 / 2 :=
by
  sorry

end slope_of_line_l52_52702


namespace correct_answer_l52_52891

theorem correct_answer (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := 
by
  sorry

end correct_answer_l52_52891


namespace evaluate_expression_l52_52854

theorem evaluate_expression : 3 ^ 123 + 9 ^ 5 / 9 ^ 3 = 3 ^ 123 + 81 :=
by
  -- we add sorry as the proof is not required
  sorry

end evaluate_expression_l52_52854


namespace cake_volume_l52_52815

theorem cake_volume :
  let thickness := 1 / 2
  let diameter := 16
  let radius := diameter / 2
  let total_volume := Real.pi * radius^2 * thickness
  total_volume / 16 = 2 * Real.pi := by
    sorry

end cake_volume_l52_52815


namespace parabola_through_point_l52_52937

theorem parabola_through_point (x y : ℝ) (hx : x = 2) (hy : y = 4) : 
  (∃ a : ℝ, y^2 = a * x ∧ a = 8) ∨ (∃ b : ℝ, x^2 = b * y ∧ b = 1) :=
sorry

end parabola_through_point_l52_52937


namespace abs_less_than_zero_impossible_l52_52027

theorem abs_less_than_zero_impossible (x : ℝ) : |x| < 0 → false :=
by
  sorry

end abs_less_than_zero_impossible_l52_52027


namespace arithmetic_sequence_ratios_l52_52756

noncomputable def a_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {a_n}
noncomputable def b_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {b_n}
noncomputable def S_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {a_n}
noncomputable def T_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {b_n}

theorem arithmetic_sequence_ratios :
  (∀ n : ℕ, 0 < n → S_n n / T_n n = (7 * n + 1) / (4 * n + 27)) →
  (a_n 7 / b_n 7 = 92 / 79) :=
by
  intros h
  sorry

end arithmetic_sequence_ratios_l52_52756


namespace ab_range_l52_52922

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + (1 / a) + (1 / b) = 5) :
  1 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end ab_range_l52_52922


namespace smallest_positive_omega_l52_52172

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * (x + Real.pi / 4) - Real.pi / 6)

theorem smallest_positive_omega (ω : ℝ) :
  (∀ x : ℝ, g (ω) x = g (ω) (-x)) → (ω = 4 / 3) := sorry

end smallest_positive_omega_l52_52172


namespace bisection_method_termination_condition_l52_52354

theorem bisection_method_termination_condition (x1 x2 e : ℝ) (h : e > 0) :
  |x1 - x2| < e → true :=
sorry

end bisection_method_termination_condition_l52_52354


namespace length_of_room_l52_52343

theorem length_of_room {L : ℝ} (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : width = 4)
  (h2 : cost_per_sqm = 750)
  (h3 : total_cost = 16500) :
  L = 5.5 ↔ (L * width) * cost_per_sqm = total_cost := 
by
  sorry

end length_of_room_l52_52343


namespace n_squared_sum_of_squares_l52_52259

theorem n_squared_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) : 
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 :=
by 
  sorry

end n_squared_sum_of_squares_l52_52259


namespace gcd_lcm_product_l52_52142

theorem gcd_lcm_product (a b : ℕ) (ha : a = 18) (hb : b = 42) :
  Nat.gcd a b * Nat.lcm a b = 756 :=
by
  rw [ha, hb]
  sorry

end gcd_lcm_product_l52_52142


namespace gcd_84_126_l52_52169

-- Conditions
def a : ℕ := 84
def b : ℕ := 126

-- Theorem to prove gcd(a, b) = 42
theorem gcd_84_126 : Nat.gcd a b = 42 := by
  sorry

end gcd_84_126_l52_52169


namespace solve_for_C_l52_52022

theorem solve_for_C : 
  ∃ C : ℝ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 :=
by
  sorry

end solve_for_C_l52_52022


namespace rotational_transform_preserves_expression_l52_52805

theorem rotational_transform_preserves_expression
  (a b c : ℝ)
  (ϕ : ℝ)
  (a1 b1 c1 : ℝ)
  (x' y' x'' y'' : ℝ)
  (h1 : x'' = x' * Real.cos ϕ + y' * Real.sin ϕ)
  (h2 : y'' = -x' * Real.sin ϕ + y' * Real.cos ϕ)
  (def_a1 : a1 = a * (Real.cos ϕ)^2 - 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.sin ϕ)^2)
  (def_b1 : b1 = a * (Real.cos ϕ) * (Real.sin ϕ) + b * ((Real.cos ϕ)^2 - (Real.sin ϕ)^2) - c * (Real.cos ϕ) * (Real.sin ϕ))
  (def_c1 : c1 = a * (Real.sin ϕ)^2 + 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.cos ϕ)^2) :
  a1 * c1 - b1^2 = a * c - b^2 := sorry

end rotational_transform_preserves_expression_l52_52805


namespace problem_1_problem_2_l52_52575

def p (x : ℝ) : Prop := -x^2 + 6*x + 16 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0 ∧ m > 0

theorem problem_1 (x : ℝ) : p x → -2 ≤ x ∧ x ≤ 8 :=
by
  -- Proof goes here
  sorry

theorem problem_2 (m : ℝ) : (∀ x, p x → q x m) ∧ (∃ x, ¬ p x ∧ q x m) → m ≥ 6 :=
by
  -- Proof goes here
  sorry

end problem_1_problem_2_l52_52575


namespace age_of_b_l52_52692

variable {a b c d Y : ℝ}

-- Conditions
def condition1 (a b : ℝ) := a = b + 2
def condition2 (b c : ℝ) := b = 2 * c
def condition3 (a d : ℝ) := d = a / 2
def condition4 (a b c d Y : ℝ) := a + b + c + d = Y

-- Theorem to prove
theorem age_of_b (a b c d Y : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 b c) 
  (h3 : condition3 a d) 
  (h4 : condition4 a b c d Y) : 
  b = Y / 3 - 1 := 
sorry

end age_of_b_l52_52692


namespace kevin_prizes_l52_52329

theorem kevin_prizes (total_prizes stuffed_animals yo_yos frisbees : ℕ)
  (h1 : total_prizes = 50) (h2 : stuffed_animals = 14) (h3 : yo_yos = 18) :
  frisbees = total_prizes - (stuffed_animals + yo_yos) → frisbees = 18 :=
by
  intro h4
  sorry

end kevin_prizes_l52_52329


namespace range_of_a_l52_52276

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) ↔ a < 1006 :=
sorry

end range_of_a_l52_52276


namespace smallest_number_of_ten_consecutive_natural_numbers_l52_52778

theorem smallest_number_of_ten_consecutive_natural_numbers 
  (x : ℕ) 
  (h : 6 * x + 39 = 2 * (4 * x + 6) + 15) : 
  x = 6 := 
by 
  sorry

end smallest_number_of_ten_consecutive_natural_numbers_l52_52778


namespace analogical_reasoning_correct_l52_52132

variable (a b c : Real)

theorem analogical_reasoning_correct (h : c ≠ 0) (h_eq : (a + b) * c = a * c + b * c) : 
  (a + b) / c = a / c + b / c :=
  sorry

end analogical_reasoning_correct_l52_52132


namespace sister_height_on_birthday_l52_52400

theorem sister_height_on_birthday (previous_height : ℝ) (growth_rate : ℝ)
    (h_previous_height : previous_height = 139.65)
    (h_growth_rate : growth_rate = 0.05) :
    previous_height * (1 + growth_rate) = 146.6325 :=
by
  -- Proof omitted
  sorry

end sister_height_on_birthday_l52_52400


namespace minuend_is_12_point_5_l52_52852

theorem minuend_is_12_point_5 (x y : ℝ) (h : x + y + (x - y) = 25) : x = 12.5 := by
  sorry

end minuend_is_12_point_5_l52_52852


namespace trapezoid_perimeter_is_183_l52_52750

-- Declare the lengths of the sides of the trapezoid
def EG : ℕ := 35
def FH : ℕ := 40
def GH : ℕ := 36

-- Declare the relation between the bases EF and GH
def EF : ℕ := 2 * GH

-- The statement of the problem
theorem trapezoid_perimeter_is_183 : EF = 72 ∧ (EG + GH + FH + EF) = 183 := by
  sorry

end trapezoid_perimeter_is_183_l52_52750


namespace consecutive_number_other_17_l52_52211

theorem consecutive_number_other_17 (a b : ℕ) (h1 : b = 17) (h2 : a + b = 35) (h3 : a + b % 5 = 0) : a = 18 :=
sorry

end consecutive_number_other_17_l52_52211


namespace squareInPentagon_l52_52048

-- Definitions pertinent to the problem
structure Pentagon (α : Type) [AddCommGroup α] :=
(A B C D E : α) 

def isRegularPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α) : Prop :=
  -- Conditions for a regular pentagon (typically involving equal side lengths and equal angles)
  sorry

def inscribedSquareExists {α : Type} [AddCommGroup α] (P : Pentagon α) : Prop :=
  -- There exists a square inscribed in the pentagon P with vertices on four different sides
  sorry

-- The main theorem to state the proof problem
theorem squareInPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α)
  (hP : isRegularPentagon P) : inscribedSquareExists P :=
sorry

end squareInPentagon_l52_52048


namespace putnam_inequality_l52_52689

variable (a x : ℝ)

theorem putnam_inequality (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3 * a * (a - x)^5 +
  5 / 2 * a^2 * (a - x)^4 -
  1 / 2 * a^4 * (a - x)^2 < 0 :=
by
  sorry

end putnam_inequality_l52_52689


namespace randy_trips_l52_52082

def trips_per_month
  (initial : ℕ) -- Randy initially had $200 in his piggy bank
  (final : ℕ)   -- Randy had $104 left in his piggy bank after a year
  (spend_per_trip : ℕ) -- Randy spends $2 every time he goes to the store
  (months_in_year : ℕ) -- Number of months in a year, which is 12
  (total_trips_per_year : ℕ) -- Total trips he makes in a year
  (trips_per_month : ℕ) -- Trips to the store every month
  : Prop :=
  initial = 200 ∧ final = 104 ∧ spend_per_trip = 2 ∧ months_in_year = 12 ∧
  total_trips_per_year = (initial - final) / spend_per_trip ∧ 
  trips_per_month = total_trips_per_year / months_in_year ∧
  trips_per_month = 4

theorem randy_trips :
  trips_per_month 200 104 2 12 ((200 - 104) / 2) (48 / 12) :=
by 
  sorry

end randy_trips_l52_52082


namespace complex_fraction_eval_l52_52110

theorem complex_fraction_eval (i : ℂ) (hi : i^2 = -1) : (3 + i) / (1 + i) = 2 - i := 
by 
  sorry

end complex_fraction_eval_l52_52110


namespace prime_roots_quadratic_l52_52827

theorem prime_roots_quadratic (p q : ℕ) (x1 x2 : ℕ) 
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (h_prime_x1 : Nat.Prime x1)
  (h_prime_x2 : Nat.Prime x2)
  (h_eq : p * x1 * x1 + p * x2 * x2 - q * x1 * x2 + 1985 = 0) :
  12 * p * p + q = 414 :=
sorry

end prime_roots_quadratic_l52_52827


namespace lily_milk_quantity_l52_52077

theorem lily_milk_quantity :
  let init_gallons := (5 : ℝ)
  let given_away := (18 / 4 : ℝ)
  let received_back := (7 / 4 : ℝ)
  init_gallons - given_away + received_back = 2 + 1 / 4 :=
by
  sorry

end lily_milk_quantity_l52_52077


namespace general_term_of_sequence_l52_52987

theorem general_term_of_sequence (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 4 * n) : 
  a n = 2 * n - 5 :=
by
  -- Proof can be completed here
  sorry

end general_term_of_sequence_l52_52987


namespace intersection_M_N_l52_52186

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N:
  M ∩ N = {-1} := by
  sorry

end intersection_M_N_l52_52186


namespace reciprocal_of_neg_three_l52_52320

theorem reciprocal_of_neg_three : ∃ x : ℚ, (-3) * x = 1 ∧ x = (-1) / 3 := sorry

end reciprocal_of_neg_three_l52_52320


namespace mechanical_pencils_and_pens_price_l52_52357

theorem mechanical_pencils_and_pens_price
    (x y : ℝ)
    (h₁ : 7 * x + 6 * y = 46.8)
    (h₂ : 3 * x + 5 * y = 32.2) :
  x = 2.4 ∧ y = 5 :=
sorry

end mechanical_pencils_and_pens_price_l52_52357


namespace population_Lake_Bright_l52_52127

-- Definition of total population
def T := 80000

-- Definition of population of Gordonia
def G := (1 / 2) * T

-- Definition of population of Toadon
def Td := (60 / 100) * G

-- Proof that the population of Lake Bright is 16000
theorem population_Lake_Bright : T - (G + Td) = 16000 :=
by {
    -- Leaving the proof as sorry
    sorry
}

end population_Lake_Bright_l52_52127


namespace triangle_side_difference_l52_52916

theorem triangle_side_difference (y : ℝ) (h : y > 6) :
  max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3 :=
by
  sorry

end triangle_side_difference_l52_52916


namespace work_completion_times_l52_52338

-- Definitions based on conditions
def condition1 (x y : ℝ) : Prop := 2 * (1 / x) + 5 * (1 / y) = 1 / 2
def condition2 (x y : ℝ) : Prop := 3 * (1 / x + 1 / y) = 0.45

-- Main theorem stating the solution
theorem work_completion_times :
  ∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ x = 12 ∧ y = 15 := 
sorry

end work_completion_times_l52_52338


namespace John_Anna_total_eBooks_l52_52108

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l52_52108


namespace complement_A_inter_B_l52_52688

def U : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }
def A : Set ℤ := { x | x * (x - 1) = 0 }
def B : Set ℤ := { x | -1 < x ∧ x < 2 }

theorem complement_A_inter_B {U A B : Set ℤ} :
  A ⊆ U → B ⊆ U → 
  (A ∩ B) ⊆ (U ∩ A ∩ B) → 
  (U \ (A ∩ B)) = { -1, 2 } :=
by 
  sorry

end complement_A_inter_B_l52_52688


namespace find_retail_price_l52_52639

-- Define the conditions
def wholesale_price : ℝ := 90
def discount_rate : ℝ := 0.10
def profit_rate : ℝ := 0.20

-- Calculate the necessary values from conditions
def profit : ℝ := profit_rate * wholesale_price
def selling_price : ℝ := wholesale_price + profit
def discount_factor : ℝ := 1 - discount_rate

-- Rewrite the main theorem statement
theorem find_retail_price : ∃ w : ℝ, discount_factor * w = selling_price → w = 120 :=
by sorry

end find_retail_price_l52_52639


namespace payment_to_C_l52_52842

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℚ := 3360

def work_done (rate : ℚ) (days : ℕ) : ℚ := rate * days

-- Conditions
def person_A_work_rate := work_rate 6
def person_B_work_rate := work_rate 8
def combined_work_rate := person_A_work_rate + person_B_work_rate
def work_by_A_and_B_in_3_days := work_done combined_work_rate 3
def total_work : ℚ := 1
def work_done_by_C := total_work - work_by_A_and_B_in_3_days

-- Proof problem statement
theorem payment_to_C :
  (work_done_by_C / total_work) * total_payment = 420 := 
sorry

end payment_to_C_l52_52842


namespace count_valid_choices_l52_52361

open Nat

def base4_representation (N : ℕ) : ℕ := 
  let a3 := N / 64 % 4
  let a2 := N / 16 % 4
  let a1 := N / 4 % 4
  let a0 := N % 4
  64 * a3 + 16 * a2 + 4 * a1 + a0

def base7_representation (N : ℕ) : ℕ := 
  let b3 := N / 343 % 7
  let b2 := N / 49 % 7
  let b1 := N / 7 % 7
  let b0 := N % 7
  343 * b3 + 49 * b2 + 7 * b1 + b0

def S (N : ℕ) : ℕ := base4_representation N + base7_representation N

def valid_choices (N : ℕ) : Prop := 
  (S N % 100) = (2 * N % 100)

theorem count_valid_choices : 
  ∃ (count : ℕ), count = 20 ∧ ∀ (N : ℕ), (N >= 1000 ∧ N < 10000) → valid_choices N ↔ (count = 20) :=
sorry

end count_valid_choices_l52_52361


namespace smallest_positive_z_l52_52641

theorem smallest_positive_z (x z : ℝ) (hx : Real.sin x = 1) (hz : Real.sin (x + z) = -1/2) : z = 2 * Real.pi / 3 :=
by
  sorry

end smallest_positive_z_l52_52641


namespace mass_percentage_O_in_CaO_l52_52846

theorem mass_percentage_O_in_CaO :
  let molar_mass_Ca := 40.08
  let molar_mass_O := 16.00
  let molar_mass_CaO := molar_mass_Ca + molar_mass_O
  let mass_percentage_O := (molar_mass_O / molar_mass_CaO) * 100
  mass_percentage_O = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l52_52846


namespace regular_15gon_symmetry_l52_52013

theorem regular_15gon_symmetry :
  ∀ (L R : ℕ),
  (L = 15) →
  (R = 24) →
  L + R = 39 :=
by
  intros L R hL hR
  exact sorry

end regular_15gon_symmetry_l52_52013


namespace total_cats_in_academy_l52_52450

theorem total_cats_in_academy (cats_jump cats_jump_fetch cats_fetch cats_fetch_spin cats_spin cats_jump_spin cats_all_three cats_none: ℕ)
  (h_jump: cats_jump = 60)
  (h_jump_fetch: cats_jump_fetch = 20)
  (h_fetch: cats_fetch = 35)
  (h_fetch_spin: cats_fetch_spin = 15)
  (h_spin: cats_spin = 40)
  (h_jump_spin: cats_jump_spin = 22)
  (h_all_three: cats_all_three = 11)
  (h_none: cats_none = 10) :
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none = 99 :=
by
  calc 
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none 
  = 11 + (20 - 11) + (15 - 11) + (22 - 11) + (60 - (9 + 11 + 11)) + (35 - (9 + 4 + 11)) + (40 - (11 + 4 + 11)) + 10 
  := by sorry
  _ = 99 := by sorry

end total_cats_in_academy_l52_52450


namespace gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l52_52627

theorem gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1 :
  Int.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
  -- proof goes here
  sorry

end gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l52_52627


namespace determine_m_range_l52_52250

-- Define propositions P and Q
def P (t : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1)
def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define negation of propositions
def notP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) ≠ 1)
def notQ (t m : ℝ) : Prop := ¬ (1 - m < t ∧ t < 1 + m)

-- Main problem: Determine the range of m where notP -> notQ is a sufficient but not necessary condition
theorem determine_m_range {m : ℝ} : (∃ t : ℝ, notP t → notQ t m) ↔ (0 < m ∧ m ≤ 3) := by
  sorry

end determine_m_range_l52_52250


namespace find_x_l52_52870

variable (x : ℤ)
def A : Set ℤ := {x^2, x + 1, -3}
def B : Set ℤ := {x - 5, 2 * x - 1, x^2 + 1}

theorem find_x (h : A x ∩ B x = {-3}) : x = -1 :=
sorry

end find_x_l52_52870


namespace vessel_reaches_boat_in_shortest_time_l52_52950

-- Define the given conditions as hypotheses
variable (dist_AC : ℝ) (angle_C : ℝ) (speed_CB : ℝ) (angle_B : ℝ) (speed_A : ℝ)

-- Assign values to variables based on the problem statement
def vessel_distress_boat_condition : Prop :=
  dist_AC = 10 ∧ angle_C = 45 ∧ speed_CB = 9 ∧ angle_B = 105 ∧ speed_A = 21

-- Define the time (in minutes) for the vessel to reach the fishing boat
noncomputable def shortest_time_to_reach_boat : ℝ :=
  25

-- The theorem that we need to prove given the conditions
theorem vessel_reaches_boat_in_shortest_time :
  vessel_distress_boat_condition dist_AC angle_C speed_CB angle_B speed_A → 
  shortest_time_to_reach_boat = 25 := by
    intros
    sorry

end vessel_reaches_boat_in_shortest_time_l52_52950


namespace production_increase_percentage_l52_52386

variable (T : ℝ) -- Initial production
variable (T1 T2 T5 : ℝ) -- Productions at different years
variable (x : ℝ) -- Unknown percentage increase for last three years

-- Conditions
def condition1 : Prop := T1 = T * 1.06
def condition2 : Prop := T2 = T1 * 1.08
def condition3 : Prop := T5 = T * (1.1 ^ 5)

-- Statement to prove
theorem production_increase_percentage :
  condition1 T T1 →
  condition2 T1 T2 →
  (T5 = T2 * (1 + x / 100) ^ 3) →
  x = 12.1 :=
by
  sorry

end production_increase_percentage_l52_52386


namespace three_topping_pizzas_l52_52817

theorem three_topping_pizzas : Nat.choose 8 3 = 56 := by
  sorry

end three_topping_pizzas_l52_52817


namespace cost_to_paint_cube_l52_52195

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) 
  (h1 : cost_per_kg = 40) 
  (h2 : coverage_per_kg = 20) 
  (h3 : side_length = 10) 
  : (6 * side_length^2 / coverage_per_kg) * cost_per_kg = 1200 :=
by
  sorry

end cost_to_paint_cube_l52_52195


namespace complementary_angles_difference_l52_52556

theorem complementary_angles_difference :
  ∃ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 90 ∧ 5 * θ₁ = 3 * θ₂ ∧ abs (θ₁ - θ₂) = 22.5 :=
by
  sorry

end complementary_angles_difference_l52_52556


namespace tens_digit_of_binary_result_l52_52782

def digits_tens_digit_subtraction (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) : ℕ :=
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let difference := original_number - reversed_number
  (difference % 100) / 10

theorem tens_digit_of_binary_result (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) :
  digits_tens_digit_subtraction a b c h1 h2 = 9 :=
sorry

end tens_digit_of_binary_result_l52_52782


namespace acute_angle_range_l52_52719

theorem acute_angle_range (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : Real.sin α < Real.cos α) : 0 < α ∧ α < π / 4 :=
sorry

end acute_angle_range_l52_52719


namespace multiply_123_32_125_l52_52212

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end multiply_123_32_125_l52_52212


namespace angle_value_l52_52497

theorem angle_value (x : ℝ) (h₁ : (90 : ℝ) = 44 + x) : x = 46 :=
by
  sorry

end angle_value_l52_52497


namespace each_spider_eats_seven_bugs_l52_52998

theorem each_spider_eats_seven_bugs (initial_bugs : ℕ) (reduction_rate : ℚ) (spiders_introduced : ℕ) (bugs_left : ℕ) (result : ℕ)
  (h1 : initial_bugs = 400)
  (h2 : reduction_rate = 0.80)
  (h3 : spiders_introduced = 12)
  (h4 : bugs_left = 236)
  (h5 : result = initial_bugs * (4 / 5) - bugs_left) :
  (result / spiders_introduced) = 7 :=
by
  sorry

end each_spider_eats_seven_bugs_l52_52998


namespace bobs_password_probability_l52_52477

theorem bobs_password_probability :
  (5 / 10) * (5 / 10) * 1 * (9 / 10) = 9 / 40 :=
by
  sorry

end bobs_password_probability_l52_52477


namespace class_overall_score_l52_52570

def max_score : ℝ := 100
def percentage_study : ℝ := 0.4
def percentage_hygiene : ℝ := 0.25
def percentage_discipline : ℝ := 0.25
def percentage_activity : ℝ := 0.1

def score_study : ℝ := 85
def score_hygiene : ℝ := 90
def score_discipline : ℝ := 80
def score_activity : ℝ := 75

theorem class_overall_score :
  (score_study * percentage_study) +
  (score_hygiene * percentage_hygiene) +
  (score_discipline * percentage_discipline) +
  (score_activity * percentage_activity) = 84 :=
  by sorry

end class_overall_score_l52_52570


namespace range_of_k_l52_52787

open Set

variable {k : ℝ}

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

theorem range_of_k (h : (compl A) ∩ B k ≠ ∅) : 0 < k ∧ k < 3 := sorry

end range_of_k_l52_52787


namespace total_distance_is_1095_l52_52266

noncomputable def totalDistanceCovered : ℕ :=
  let running_first_3_months := 3 * 3 * 10
  let running_next_3_months := 3 * 3 * 20
  let running_last_6_months := 3 * 6 * 30
  let total_running := running_first_3_months + running_next_3_months + running_last_6_months

  let swimming_first_6_months := 3 * 6 * 5
  let total_swimming := swimming_first_6_months

  let total_hiking := 13 * 15

  total_running + total_swimming + total_hiking

theorem total_distance_is_1095 : totalDistanceCovered = 1095 := by
  sorry

end total_distance_is_1095_l52_52266


namespace mail_distribution_l52_52016

-- Define the number of houses
def num_houses : ℕ := 10

-- Define the pieces of junk mail per house
def mail_per_house : ℕ := 35

-- Define total pieces of junk mail delivered
def total_pieces_of_junk_mail : ℕ := num_houses * mail_per_house

-- Main theorem statement
theorem mail_distribution : total_pieces_of_junk_mail = 350 := by
  sorry

end mail_distribution_l52_52016


namespace quadratic_inverse_condition_l52_52653

theorem quadratic_inverse_condition : 
  (∀ x₁ x₂ : ℝ, (x₁ ≥ 2 ∧ x₂ ≥ 2 ∧ x₁ ≠ x₂) → (x₁^2 - 4*x₁ + 5 ≠ x₂^2 - 4*x₂ + 5)) :=
sorry

end quadratic_inverse_condition_l52_52653


namespace solve_for_y_l52_52004

theorem solve_for_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 8) : y = 1 / 3 := 
by
  sorry

end solve_for_y_l52_52004


namespace average_of_second_set_l52_52818

open Real

theorem average_of_second_set 
  (avg6 : ℝ)
  (n1 n2 n3 n4 n5 n6 : ℝ)
  (avg1_set : ℝ)
  (avg3_set : ℝ)
  (h1 : avg6 = 3.95)
  (h2 : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = avg6)
  (h3 : (n1 + n2) / 2 = 3.6)
  (h4 : (n5 + n6) / 2 = 4.400000000000001) :
  (n3 + n4) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_l52_52818


namespace exists_indices_l52_52213

theorem exists_indices (a : ℕ → ℕ) 
  (h_seq_perm : ∀ n, ∃ m, a m = n) : 
  ∃ ℓ m, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ :=
by
  sorry

end exists_indices_l52_52213


namespace true_inverse_of_opposites_true_contrapositive_of_real_roots_l52_52112

theorem true_inverse_of_opposites (X Y : Int) :
  (X = -Y) → (X + Y = 0) :=
by 
  sorry

theorem true_contrapositive_of_real_roots (q : Real) :
  (¬ ∃ x : Real, x^2 + 2*x + q = 0) → (q > 1) :=
by
  sorry

end true_inverse_of_opposites_true_contrapositive_of_real_roots_l52_52112


namespace avg_divisible_by_4_between_15_and_55_eq_34_l52_52100

theorem avg_divisible_by_4_between_15_and_55_eq_34 :
  let numbers := (List.filter (λ x => x % 4 = 0) (List.range' 16 37))
  (List.sum numbers) / (numbers.length) = 34 := by
  sorry

end avg_divisible_by_4_between_15_and_55_eq_34_l52_52100


namespace complement_union_complement_intersection_l52_52607

open Set

noncomputable def universal_set : Set ℝ := univ

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem complement_union :
  compl (A ∪ B) = {x : ℝ | x ≤ 2 ∨ 7 ≤ x} := by
  sorry

theorem complement_intersection :
  (compl A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end complement_union_complement_intersection_l52_52607


namespace fraction_unchanged_when_increased_by_ten_l52_52182

variable {x y : ℝ}

theorem fraction_unchanged_when_increased_by_ten (x y : ℝ) :
  (5 * (10 * x)) / (10 * x + 10 * y) = 5 * x / (x + y) :=
by
  sorry

end fraction_unchanged_when_increased_by_ten_l52_52182


namespace sum_minimum_values_l52_52335

def P (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem sum_minimum_values (a b c d e f : ℝ)
  (hPQ : ∀ x, P (Q x d e f) a b c = 0 → x = -4 ∨ x = -2 ∨ x = 0 ∨ x = 2 ∨ x = 4)
  (hQP : ∀ x, Q (P x a b c) d e f = 0 → x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 3) :
  P 0 a b c + Q 0 d e f = -20 := sorry

end sum_minimum_values_l52_52335


namespace poly_a_roots_poly_b_roots_l52_52887

-- Define the polynomials
def poly_a (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 - 11 * x + 6
def poly_b (x : ℤ) : ℤ := x ^ 4 + 4 * x ^ 3 - 9 * x ^ 2 - 16 * x + 20

-- Assert the integer roots for poly_a
theorem poly_a_roots : {x : ℤ | poly_a x = 0} = {-2, 3} := sorry

-- Assert the integer roots for poly_b
theorem poly_b_roots : {x : ℤ | poly_b x = 0} = {1, 2, -2, -5} := sorry

end poly_a_roots_poly_b_roots_l52_52887


namespace similarity_ratio_of_polygons_l52_52837

theorem similarity_ratio_of_polygons (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : a / (b : ℚ) = 3 / 5 :=
by 
  sorry

end similarity_ratio_of_polygons_l52_52837


namespace ab_squared_non_positive_l52_52774

theorem ab_squared_non_positive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 :=
sorry

end ab_squared_non_positive_l52_52774


namespace bars_not_sold_l52_52812

-- Definitions for the conditions
def cost_per_bar : ℕ := 3
def total_bars : ℕ := 9
def money_made : ℕ := 18

-- The theorem we need to prove
theorem bars_not_sold : total_bars - (money_made / cost_per_bar) = 3 := sorry

end bars_not_sold_l52_52812


namespace sum_div_by_24_l52_52294

theorem sum_div_by_24 (m n : ℕ) (h : ∃ k : ℤ, mn + 1 = 24 * k): (m + n) % 24 = 0 := 
by
  sorry

end sum_div_by_24_l52_52294


namespace correct_operation_l52_52803
variable (a x y: ℝ)

theorem correct_operation : 
  ¬ (5 * a - 2 * a = 3) ∧
  ¬ ((x + 2 * y)^2 = x^2 + 4 * y^2) ∧
  ¬ (x^8 / x^4 = x^2) ∧
  ((2 * a)^3 = 8 * a^3) :=
by
  sorry

end correct_operation_l52_52803


namespace Avery_builds_in_4_hours_l52_52373

variable (A : ℝ) (TomTime : ℝ := 2) (TogetherTime : ℝ := 1) (RemainingTomTime : ℝ := 0.5)

-- Conditions:
axiom Tom_builds_in_2_hours : TomTime = 2
axiom Work_together_for_1_hour : TogetherTime = 1
axiom Tom_finishes_in_0_5_hours : RemainingTomTime = 0.5

-- Question:
theorem Avery_builds_in_4_hours : A = 4 :=
by
  sorry

end Avery_builds_in_4_hours_l52_52373


namespace world_expo_visitors_l52_52928

noncomputable def per_person_cost (x : ℕ) : ℕ :=
  if x <= 30 then 120 else max (120 - 2 * (x - 30)) 90

theorem world_expo_visitors (x : ℕ) (h_cost : x * per_person_cost x = 4000) : x = 40 :=
by
  sorry

end world_expo_visitors_l52_52928


namespace unique_reconstruction_l52_52753

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end unique_reconstruction_l52_52753


namespace yoongi_age_l52_52695

theorem yoongi_age (Y H : ℕ) (h1 : Y + H = 16) (h2 : Y = H + 2) : Y = 9 :=
by
  sorry

end yoongi_age_l52_52695


namespace problem1_problem2_l52_52119

variable (a : ℝ)

def quadratic_roots (a x : ℝ) : Prop := a*x^2 + 2*x + 1 = 0

-- Problem 1: If 1/2 is a root, find the set A
theorem problem1 (h : quadratic_roots a (1/2)) : 
  {x : ℝ | quadratic_roots (a) x } = { -1/4, 1/2 } :=
sorry

-- Problem 2: If A contains exactly one element, find the set B consisting of such a
theorem problem2 (h : ∃! (x : ℝ), quadratic_roots a x ) : 
  {a : ℝ | ∃! (x : ℝ), quadratic_roots a x} = { 0, 1 } :=
sorry

end problem1_problem2_l52_52119


namespace initial_percentage_alcohol_l52_52471

variables (P : ℝ) (initial_volume : ℝ) (added_volume : ℝ) (total_volume : ℝ) (final_percentage : ℝ) (init_percentage : ℝ)

theorem initial_percentage_alcohol (h1 : initial_volume = 6)
                                  (h2 : added_volume = 3)
                                  (h3 : total_volume = initial_volume + added_volume)
                                  (h4 : final_percentage = 50)
                                  (h5 : init_percentage = 100 * (initial_volume * P / 100 + added_volume) / total_volume)
                                  : P = 25 :=
by {
  sorry
}

end initial_percentage_alcohol_l52_52471


namespace sum_of_all_four_numbers_is_zero_l52_52071

theorem sum_of_all_four_numbers_is_zero 
  {a b c d : ℝ}
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b = c + d)
  (h_prod : a * c = b * d) 
  : a + b + c + d = 0 := 
by
  sorry

end sum_of_all_four_numbers_is_zero_l52_52071


namespace solve_for_percentage_l52_52069

-- Define the constants and variables
variables (P : ℝ)

-- Define the given conditions
def condition : Prop := (P / 100 * 1600 = P / 100 * 650 + 190)

-- Formalize the conjecture: if the conditions hold, then P = 20
theorem solve_for_percentage (h : condition P) : P = 20 :=
sorry

end solve_for_percentage_l52_52069


namespace problem_2014_minus_4102_l52_52526

theorem problem_2014_minus_4102 : 2014 - 4102 = -2088 := 
by
  -- The proof is omitted as per the requirement
  sorry

end problem_2014_minus_4102_l52_52526


namespace common_region_area_of_triangles_l52_52233

noncomputable def area_of_common_region (a : ℝ) : ℝ :=
  (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3

theorem common_region_area_of_triangles (a : ℝ) (h : 0 < a) : 
  area_of_common_region a = (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3 :=
by
  sorry

end common_region_area_of_triangles_l52_52233


namespace find_positive_integer_solutions_l52_52503

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2^x + 3^y = z^2 ↔ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 4 ∧ y = 2 ∧ z = 5) := 
sorry

end find_positive_integer_solutions_l52_52503


namespace no_pair_of_primes_l52_52527

theorem no_pair_of_primes (p q : ℕ) (hp_prime : Prime p) (hq_prime : Prime q) (h_gt : p > q) :
  ¬ (∃ (h : ℤ), 2 * (p^2 - q^2) = 8 * h + 4) :=
by
  sorry

end no_pair_of_primes_l52_52527


namespace carbon_neutrality_l52_52461

theorem carbon_neutrality (a b : ℝ) (t : ℕ) (ha : a > 0)
  (h1 : S = a * b ^ t)
  (h2 : a * b ^ 7 = 4 * a / 5)
  (h3 : a / 4 = S) :
  t = 42 := 
sorry

end carbon_neutrality_l52_52461


namespace isolate_y_l52_52485

theorem isolate_y (x y : ℝ) (h : 3 * x - 2 * y = 6) : y = 3 * x / 2 - 3 :=
sorry

end isolate_y_l52_52485


namespace power_mod_congruence_l52_52521

theorem power_mod_congruence (h : 3^400 ≡ 1 [MOD 500]) : 3^800 ≡ 1 [MOD 500] :=
by {
  sorry
}

end power_mod_congruence_l52_52521


namespace tom_cost_cheaper_than_jane_l52_52267

def store_A_full_price : ℝ := 125
def store_A_discount_single : ℝ := 0.08
def store_A_discount_bulk : ℝ := 0.12
def store_A_tax_rate : ℝ := 0.07
def store_A_shipping_fee : ℝ := 10
def store_A_club_discount : ℝ := 0.05

def store_B_full_price : ℝ := 130
def store_B_discount_single : ℝ := 0.10
def store_B_discount_bulk : ℝ := 0.15
def store_B_tax_rate : ℝ := 0.05
def store_B_free_shipping_threshold : ℝ := 250
def store_B_club_discount : ℝ := 0.03

def tom_smartphones_qty : ℕ := 2
def jane_smartphones_qty : ℕ := 3

theorem tom_cost_cheaper_than_jane :
  let tom_cost := 
    let total := store_A_full_price * tom_smartphones_qty
    let discount := if tom_smartphones_qty ≥ 2 then store_A_discount_bulk else store_A_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_A_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_A_tax_rate) 
    price_after_tax + store_A_shipping_fee

  let jane_cost := 
    let total := store_B_full_price * jane_smartphones_qty
    let discount := if jane_smartphones_qty ≥ 3 then store_B_discount_bulk else store_B_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_B_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_B_tax_rate)
    let shipping_fee := if total > store_B_free_shipping_threshold then 0 else 0
    price_after_tax + shipping_fee
  
  jane_cost - tom_cost = 104.01 := 
by 
  sorry

end tom_cost_cheaper_than_jane_l52_52267


namespace juliet_older_than_maggie_l52_52102

-- Definitions from the given conditions
def Juliet_age : ℕ := 10
def Ralph_age (J : ℕ) : ℕ := J + 2
def Maggie_age (R : ℕ) : ℕ := 19 - R

-- Theorem statement
theorem juliet_older_than_maggie :
  Juliet_age - Maggie_age (Ralph_age Juliet_age) = 3 :=
by
  sorry

end juliet_older_than_maggie_l52_52102


namespace mary_needs_10_charges_to_vacuum_house_l52_52886

theorem mary_needs_10_charges_to_vacuum_house :
  (let bedroom_time := 10
   let kitchen_time := 12
   let living_room_time := 8
   let dining_room_time := 6
   let office_time := 9
   let bathroom_time := 5
   let battery_duration := 8
   3 * bedroom_time + kitchen_time + living_room_time + dining_room_time + office_time + 2 * bathroom_time) / battery_duration = 10 :=
by sorry

end mary_needs_10_charges_to_vacuum_house_l52_52886


namespace factor_expression_l52_52536

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l52_52536


namespace prism_volume_l52_52268

-- Define the right triangular prism conditions

variables (AB BC AC : ℝ)
variable (S : ℝ)
variable (volume : ℝ)

-- Given conditions
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2
axiom AC_eq_2sqrt3 : AC = 2 * Real.sqrt 3
axiom circumscribed_sphere_surface_area : S = 32 * Real.pi

-- Statement to prove
theorem prism_volume : volume = 4 * Real.sqrt 3 :=
sorry

end prism_volume_l52_52268


namespace carl_profit_l52_52207

-- Define the conditions
def price_per_watermelon : ℕ := 3
def watermelons_start : ℕ := 53
def watermelons_end : ℕ := 18

-- Define the number of watermelons sold
def watermelons_sold : ℕ := watermelons_start - watermelons_end

-- Define the profit
def profit : ℕ := watermelons_sold * price_per_watermelon

-- State the theorem about Carl's profit
theorem carl_profit : profit = 105 :=
by
  -- Proof can be filled in later
  sorry

end carl_profit_l52_52207


namespace trajectory_of_P_eqn_l52_52548

noncomputable def point_A : ℝ × ℝ := (1, 0)

def curve_C (x : ℝ) : ℝ := x^2 - 2

def symmetric_point (Qx Qy Px Py : ℝ) : Prop :=
  Qx = 2 - Px ∧ Qy = -Py

theorem trajectory_of_P_eqn (Qx Qy Px Py : ℝ) (hQ_on_C : Qy = curve_C Qx)
  (h_symm : symmetric_point Qx Qy Px Py) :
  Py = -Px^2 + 4 * Px - 2 :=
by
  sorry

end trajectory_of_P_eqn_l52_52548


namespace five_x_plus_four_is_25_over_7_l52_52387

theorem five_x_plus_four_is_25_over_7 (x : ℚ) (h : 5 * x - 8 = 12 * x + 15) : 5 * (x + 4) = 25 / 7 := by
  sorry

end five_x_plus_four_is_25_over_7_l52_52387


namespace combination_simplify_l52_52081

theorem combination_simplify : (Nat.choose 6 2) + 3 = 18 := by
  sorry

end combination_simplify_l52_52081


namespace midpoint_product_zero_l52_52681

theorem midpoint_product_zero (x y : ℝ)
  (h_midpoint_x : (2 + x) / 2 = 4)
  (h_midpoint_y : (6 + y) / 2 = 3) :
  x * y = 0 :=
by
  sorry

end midpoint_product_zero_l52_52681


namespace length_of_flat_terrain_l52_52790

theorem length_of_flat_terrain (total_time : ℚ)
  (total_distance : ℕ)
  (speed_uphill speed_flat speed_downhill : ℚ)
  (distance_uphill distance_flat : ℕ) :
  total_time = 116 / 60 ∧
  total_distance = distance_uphill + distance_flat + (total_distance - distance_uphill - distance_flat) ∧
  speed_uphill = 4 ∧
  speed_flat = 5 ∧
  speed_downhill = 6 ∧
  distance_uphill ≥ 0 ∧
  distance_flat ≥ 0 ∧
  distance_uphill + distance_flat ≤ total_distance →
  distance_flat = 3 := 
by 
  sorry

end length_of_flat_terrain_l52_52790


namespace volunteers_allocation_scheme_count_l52_52963

theorem volunteers_allocation_scheme_count :
  let volunteers := 6
  let groups_of_two := 2
  let groups_of_one := 2
  let pavilions := 4
  let calculate_combinations (n k : ℕ) := Nat.choose n k
  calculate_combinations volunteers 2 * calculate_combinations (volunteers - 2) 2 * 
  calculate_combinations pavilions 2 * Nat.factorial pavilions = 1080 := by
sorry

end volunteers_allocation_scheme_count_l52_52963


namespace minFuseLength_l52_52620

namespace EarthquakeRelief

def fuseLengthRequired (distanceToSafety : ℕ) (speedOperator : ℕ) (burningSpeed : ℕ) (lengthFuse : ℕ) : Prop :=
  (lengthFuse : ℝ) / (burningSpeed : ℝ) > (distanceToSafety : ℝ) / (speedOperator : ℝ)

theorem minFuseLength 
  (distanceToSafety : ℕ := 400) 
  (speedOperator : ℕ := 5) 
  (burningSpeed : ℕ := 12) : 
  ∀ lengthFuse: ℕ, 
  fuseLengthRequired distanceToSafety speedOperator burningSpeed lengthFuse → lengthFuse > 96 := 
by
  sorry

end EarthquakeRelief

end minFuseLength_l52_52620


namespace intersection_point_at_neg4_l52_52494

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end intersection_point_at_neg4_l52_52494


namespace sasha_remainder_20_l52_52308

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l52_52308


namespace complement_A_intersect_B_eq_l52_52098

def setA : Set ℝ := { x : ℝ | |x - 2| ≤ 2 }

def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }

def A_intersect_B := setA ∩ setB

def complement (A : Set ℝ) : Set ℝ := { x : ℝ | x ∉ A }

theorem complement_A_intersect_B_eq {A : Set ℝ} {B : Set ℝ} 
  (hA : A = { x : ℝ | |x - 2| ≤ 2 })
  (hB : B = { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }) :
  complement (A ∩ B) = { x : ℝ | x ≠ 0 } :=
by
  sorry

end complement_A_intersect_B_eq_l52_52098


namespace girl_needs_120_oranges_l52_52760

-- Define the cost and selling prices per pack
def cost_per_pack : ℤ := 15   -- cents
def oranges_per_pack_cost : ℤ := 4
def sell_per_pack : ℤ := 30   -- cents
def oranges_per_pack_sell : ℤ := 6

-- Define the target profit
def target_profit : ℤ := 150  -- cents

-- Calculate the cost price per orange
def cost_per_orange : ℚ := cost_per_pack / oranges_per_pack_cost

-- Calculate the selling price per orange
def sell_per_orange : ℚ := sell_per_pack / oranges_per_pack_sell

-- Calculate the profit per orange
def profit_per_orange : ℚ := sell_per_orange - cost_per_orange

-- Calculate the number of oranges needed to achieve the target profit
def oranges_needed : ℚ := target_profit / profit_per_orange

-- Lean theorem statement
theorem girl_needs_120_oranges :
  oranges_needed = 120 :=
  sorry

end girl_needs_120_oranges_l52_52760


namespace incorrect_table_value_l52_52594

theorem incorrect_table_value (a b c : ℕ) (values : List ℕ) (correct : values = [2051, 2197, 2401, 2601, 2809, 3025, 3249, 3481]) : 
  (2401 ∉ [2051, 2197, 2399, 2601, 2809, 3025, 3249, 3481]) :=
sorry

end incorrect_table_value_l52_52594


namespace find_vanilla_cookies_l52_52838

variable (V : ℕ)

def num_vanilla_cookies_sold (choc_cookies: ℕ) (vanilla_cookies: ℕ) (total_revenue: ℕ) : Prop :=
  choc_cookies * 1 + vanilla_cookies * 2 = total_revenue

theorem find_vanilla_cookies (h : num_vanilla_cookies_sold 220 V 360) : V = 70 :=
by
  sorry

end find_vanilla_cookies_l52_52838


namespace train_speed_correct_l52_52773

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end train_speed_correct_l52_52773


namespace miles_left_l52_52918

theorem miles_left (d_total d_covered d_left : ℕ) 
  (h₁ : d_total = 78) 
  (h₂ : d_covered = 32) 
  (h₃ : d_left = d_total - d_covered):
  d_left = 46 := 
by {
  sorry 
}

end miles_left_l52_52918


namespace exists_n_divisible_by_5_l52_52567

open Int

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h1 : 5 ∣ (a * m^3 + b * m^2 + c * m + d)) 
  (h2 : ¬ (5 ∣ d)) :
  ∃ n : ℤ, 5 ∣ (d * n^3 + c * n^2 + b * n + a) :=
by
  sorry

end exists_n_divisible_by_5_l52_52567


namespace unique_root_exists_maximum_value_lnx_l52_52974

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x

theorem unique_root_exists (k : ℝ) :
  ∃ a, a = 1 ∧ (∃ x ∈ Set.Ioo k (k+1), f x = g x) :=
sorry

theorem maximum_value_lnx (p q : ℝ) :
  (∃ x, (x = min p q) ∧ Real.log x = ( 4 / Real.exp 2 )) :=
sorry

end unique_root_exists_maximum_value_lnx_l52_52974


namespace samantha_routes_l52_52039

-- Define the positions relative to the grid
structure Position where
  x : Int
  y : Int

-- Define the initial conditions and path constraints
def house : Position := ⟨-3, -2⟩
def sw_corner_of_park : Position := ⟨0, 0⟩
def ne_corner_of_park : Position := ⟨8, 5⟩
def school : Position := ⟨11, 8⟩

-- Define the combinatorial function for calculating number of ways
def binom (n k : Nat) : Nat := Nat.choose n k

-- Route segments based on the constraints
def ways_house_to_sw_corner : Nat := binom 5 2
def ways_through_park : Nat := 1
def ways_ne_corner_to_school : Nat := binom 6 3

-- Total number of routes
def total_routes : Nat := ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school

-- The statement to be proven
theorem samantha_routes : total_routes = 200 := by
  sorry

end samantha_routes_l52_52039


namespace max_value_of_linear_combination_of_m_n_k_l52_52518

-- The style grants us maximum flexibility for definitions.
theorem max_value_of_linear_combination_of_m_n_k 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (m n k : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ m → a i % 3 = 1)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → b i % 3 = 2)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ k → c i % 3 = 0)
  (h4 : Function.Injective a)
  (h5 : Function.Injective b)
  (h6 : Function.Injective c)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)
  (h_sum : (Finset.range m).sum a + (Finset.range n).sum b + (Finset.range k).sum c = 2007)
  : 4 * m + 3 * n + 5 * k ≤ 256 := by
  sorry

end max_value_of_linear_combination_of_m_n_k_l52_52518


namespace one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l52_52028

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem one_zero_implies_a_eq_pm2 (a : ℝ) : (∃! x, f a x = 0) → (a = 2 ∨ a = -2) := by
  sorry

theorem zero_in_interval_implies_a_in_open_interval (a : ℝ) : (∃ x, f a x = 0 ∧ 0 < x ∧ x < 1) → 2 < a := by
  sorry

end one_zero_implies_a_eq_pm2_zero_in_interval_implies_a_in_open_interval_l52_52028


namespace range_of_a_l52_52444

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x + 3

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (a-1) (a+1), 4*x - 1/x = 0) ↔ 1 ≤ a ∧ a < 3/2 :=
sorry

end range_of_a_l52_52444


namespace spend_on_video_games_l52_52201

/-- Given the total allowance and the fractions of spending on various categories,
prove the amount spent on video games. -/
theorem spend_on_video_games (total_allowance : ℝ)
  (fraction_books fraction_snacks fraction_crafts : ℝ)
  (h_total : total_allowance = 50)
  (h_fraction_books : fraction_books = 1 / 4)
  (h_fraction_snacks : fraction_snacks = 1 / 5)
  (h_fraction_crafts : fraction_crafts = 3 / 10) :
  total_allowance - (fraction_books * total_allowance + fraction_snacks * total_allowance + fraction_crafts * total_allowance) = 12.5 :=
by
  sorry

end spend_on_video_games_l52_52201


namespace inequality_solution_range_l52_52652

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end inequality_solution_range_l52_52652


namespace problem_statement_l52_52155

open Real

theorem problem_statement (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (a : ℝ := x + x⁻¹) (b : ℝ := y + y⁻¹) (c : ℝ := z + z⁻¹) :
  a > 2 ∧ b > 2 ∧ c > 2 :=
by sorry

end problem_statement_l52_52155


namespace exists_nat_number_reduce_by_57_l52_52191

theorem exists_nat_number_reduce_by_57 :
  ∃ (N : ℕ), ∃ (k : ℕ) (a x : ℕ),
    N = 10^k * a + x ∧
    10^k * a + x = 57 * x ∧
    N = 7125 :=
sorry

end exists_nat_number_reduce_by_57_l52_52191


namespace merchants_tea_cups_l52_52005

theorem merchants_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11)
  (h2 : b + c = 15)
  (h3 : a + c = 14) : 
  a + b + c = 20 :=
by
  sorry

end merchants_tea_cups_l52_52005


namespace intersection_A_B_l52_52894

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def setA : Set ℝ := { x | Real.log x > 0 }
def setB : Set ℝ := { x | Real.exp x * Real.exp x < 3 }

theorem intersection_A_B : setA ∩ setB = { x | 1 < x ∧ x < log2 3 } :=
by
  sorry

end intersection_A_B_l52_52894


namespace students_standing_count_l52_52953

def students_seated : ℕ := 300
def teachers_seated : ℕ := 30
def total_attendees : ℕ := 355

theorem students_standing_count : total_attendees - (students_seated + teachers_seated) = 25 :=
by
  sorry

end students_standing_count_l52_52953


namespace quadrilateral_is_trapezoid_or_parallelogram_l52_52921

noncomputable def quadrilateral_property (s1 s2 s3 s4 : ℝ) : Prop :=
  (s1 + s2) * (s3 + s4) = (s1 + s4) * (s2 + s3)

theorem quadrilateral_is_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ) (h : quadrilateral_property s1 s2 s3 s4) :
  (s1 = s3) ∨ (s2 = s4) ∨ -- Trapezoid conditions
  ∃ (p : ℝ), (p * s1 = s3 * (s1 + s4)) := -- Add necessary conditions to represent a parallelogram
sorry

end quadrilateral_is_trapezoid_or_parallelogram_l52_52921


namespace sin_405_eq_sqrt2_div2_l52_52966

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l52_52966


namespace range_of_k_l52_52876

def P (x k : ℝ) : Prop := x^2 + k*x + 1 > 0
def Q (x k : ℝ) : Prop := k*x^2 + x + 2 < 0

theorem range_of_k (k : ℝ) : (¬ (P 2 k ∧ Q 2 k)) ↔ k ∈ (Set.Iic (-5/2) ∪ Set.Ici (-1)) := 
by
  sorry

end range_of_k_l52_52876


namespace probability_log_value_l52_52257

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 - 1

theorem probability_log_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ 10) :
  (4 / 9 : ℝ) = 
    ((8 - 4) / (10 - 1) : ℝ) := by
  sorry

end probability_log_value_l52_52257


namespace mean_temperature_is_correct_l52_52797

def temperatures : List ℤ := [-8, -6, -3, -3, 0, 4, -1]
def mean_temperature (temps : List ℤ) : ℚ := (temps.sum : ℚ) / temps.length

theorem mean_temperature_is_correct :
  mean_temperature temperatures = -17 / 7 :=
by
  sorry

end mean_temperature_is_correct_l52_52797


namespace solve_quadratic_l52_52336

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 5 * x ^ 2 + 9 * x - 18 = 0) : x = 6 / 5 :=
by
  sorry

end solve_quadratic_l52_52336


namespace solution_set_of_inequality_l52_52084

variable (a x : ℝ)

theorem solution_set_of_inequality (h : 0 < a ∧ a < 1) :
  (a - x) * (x - (1/a)) > 0 ↔ a < x ∧ x < 1/a :=
sorry

end solution_set_of_inequality_l52_52084


namespace range_g_l52_52927

noncomputable def g (x : Real) : Real := (Real.sin x)^6 + (Real.cos x)^4

theorem range_g :
  ∃ (a : Real), 
    (∀ x : Real, g x ≥ a ∧ g x ≤ 1) ∧
    (∀ y : Real, y < a → ¬∃ x : Real, g x = y) :=
sorry

end range_g_l52_52927


namespace tim_total_expenditure_l52_52734

def apple_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 6
def chocolate_price : ℕ := 10

def apple_quantity : ℕ := 8
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 3
def flour_quantity : ℕ := 3
def chocolate_quantity : ℕ := 1

def discounted_pineapple_price : ℕ := pineapple_price / 2
def discounted_milk_price : ℕ := milk_price - 1
def coupon_discount : ℕ := 10
def discount_threshold : ℕ := 50

def total_cost_before_coupon : ℕ :=
  (apple_quantity * apple_price) +
  (milk_quantity * discounted_milk_price) +
  (pineapple_quantity * discounted_pineapple_price) +
  (flour_quantity * flour_price) +
  chocolate_price

def final_price : ℕ :=
  if total_cost_before_coupon >= discount_threshold
  then total_cost_before_coupon - coupon_discount
  else total_cost_before_coupon

theorem tim_total_expenditure : final_price = 40 := by
  sorry

end tim_total_expenditure_l52_52734


namespace min_value_of_c_l52_52893

noncomputable def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

theorem min_value_of_c (c : ℕ) (n m : ℕ) (h1 : 5 * c = n^3) (h2 : 3 * c = m^2) : c = 675 := by
  sorry

end min_value_of_c_l52_52893


namespace matchstick_problem_l52_52455

theorem matchstick_problem (n : ℕ) (T : ℕ → ℕ) :
  (∀ n, T n = 4 + 9 * (n - 1)) ∧ n = 15 → T n = 151 :=
by
  sorry

end matchstick_problem_l52_52455


namespace triangle_angle_sum_33_75_l52_52452

theorem triangle_angle_sum_33_75 (x : ℝ) 
  (h₁ : 45 + 3 * x + x = 180) : 
  x = 33.75 :=
  sorry

end triangle_angle_sum_33_75_l52_52452


namespace factorize1_factorize2_factorize3_factorize4_l52_52319

-- Statement for the first equation
theorem factorize1 (a x : ℝ) : 
  a * x^2 - 7 * a * x + 6 * a = a * (x - 6) * (x - 1) :=
sorry

-- Statement for the second equation
theorem factorize2 (x y : ℝ) : 
  x * y^2 - 9 * x = x * (y + 3) * (y - 3) :=
sorry

-- Statement for the third equation
theorem factorize3 (x y : ℝ) : 
  1 - x^2 + 2 * x * y - y^2 = (1 + x - y) * (1 - x + y) :=
sorry

-- Statement for the fourth equation
theorem factorize4 (x y : ℝ) : 
  8 * (x^2 - 2 * y^2) - x * (7 * x + y) + x * y = (x + 4 * y) * (x - 4 * y) :=
sorry

end factorize1_factorize2_factorize3_factorize4_l52_52319


namespace polygon_area_l52_52828

-- Definitions and conditions
def side_length (n : ℕ) (p : ℕ) := p / n
def rectangle_area (s : ℕ) := 2 * s * s
def total_area (r : ℕ) (area : ℕ) := r * area

-- Theorem statement with conditions and conclusion
theorem polygon_area (n r p : ℕ) (h1 : n = 24) (h2 : r = 4) (h3 : p = 48) :
  total_area r (rectangle_area (side_length n p)) = 32 := by
  sorry

end polygon_area_l52_52828


namespace corrected_average_l52_52646

theorem corrected_average (incorrect_avg : ℕ) (correct_val incorrect_val number_of_values : ℕ) (avg := 17) (n := 10) (inc := 26) (cor := 56) :
  incorrect_avg = 17 →
  number_of_values = 10 →
  correct_val = 56 →
  incorrect_val = 26 →
  correct_avg = (incorrect_avg * number_of_values + (correct_val - incorrect_val)) / number_of_values →
  correct_avg = 20 := by
  sorry

end corrected_average_l52_52646


namespace inequality_proof_l52_52388

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom abc_eq_one : a * b * c = 1

theorem inequality_proof :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 :=
by
  sorry

end inequality_proof_l52_52388


namespace largest_consecutive_odd_sum_l52_52709

theorem largest_consecutive_odd_sum (x : ℤ) (h : 20 * (x + 19) = 8000) : x + 38 = 419 := 
by
  sorry

end largest_consecutive_odd_sum_l52_52709


namespace geometric_series_first_term_l52_52397

noncomputable def first_term_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) : Prop :=
  S = a / (1 - r)

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (hr : r = 1/6)
  (hS : S = 54) :
  first_term_geometric_series r S a →
  a = 45 :=
by
  intros h
  -- The proof goes here
  sorry

end geometric_series_first_term_l52_52397


namespace lean_proof_problem_l52_52816

section

variable {R : Type*} [AddCommGroup R]

def is_odd_function (f : ℝ → R) : Prop :=
  ∀ x, f (-x) = -f x

theorem lean_proof_problem (f: ℝ → ℝ) (h_odd: is_odd_function f)
    (h_cond: f 3 + f (-2) = 2) : f 2 - f 3 = -2 :=
by
  sorry

end

end lean_proof_problem_l52_52816


namespace find_x_l52_52152

-- Definitions of binomial coefficients as conditions
def binomial (n k : ℕ) : ℕ := n.choose k

-- The specific conditions given
def C65_eq_6 : Prop := binomial 6 5 = 6
def C64_eq_15 : Prop := binomial 6 4 = 15

-- The theorem we need to prove: ∃ x, binomial 7 x = 21
theorem find_x (h1 : C65_eq_6) (h2 : C64_eq_15) : ∃ x, binomial 7 x = 21 :=
by
  -- Proof will go here
  sorry

end find_x_l52_52152


namespace syllogism_correct_l52_52369

-- Hypotheses for each condition
def OptionA := "The first section, the second section, the third section"
def OptionB := "Major premise, minor premise, conclusion"
def OptionC := "Induction, conjecture, proof"
def OptionD := "Dividing the discussion into three sections"

-- Definition of a syllogism in deductive reasoning
def syllogism_def := "A logical argument that applies deductive reasoning to arrive at a conclusion based on two propositions assumed to be true"

-- Theorem stating that a syllogism corresponds to Option B
theorem syllogism_correct :
  syllogism_def = OptionB :=
by
  sorry

end syllogism_correct_l52_52369


namespace J_3_15_10_eq_68_over_15_l52_52171

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_15_10_eq_68_over_15 : J 3 15 10 = 68 / 15 := by
  sorry

end J_3_15_10_eq_68_over_15_l52_52171


namespace cows_and_goats_sum_l52_52829

theorem cows_and_goats_sum (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 4 * x + 2 * y + 4 * z = 18 + 2 * (x + y + z)) 
  : x + z = 9 := by 
  sorry

end cows_and_goats_sum_l52_52829


namespace exp_mono_increasing_of_gt_l52_52139

variable {a b : ℝ}

theorem exp_mono_increasing_of_gt (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b :=
by sorry

end exp_mono_increasing_of_gt_l52_52139


namespace find_square_tiles_l52_52991

variable {s p : ℕ}

theorem find_square_tiles (h1 : s + p = 30) (h2 : 4 * s + 5 * p = 110) : s = 20 :=
by
  sorry

end find_square_tiles_l52_52991


namespace hyperbola_vertex_distance_l52_52736

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

-- Statement: The distance between the vertices of the hyperbola is 1
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 2 * (1 / 2) = 1 :=
by
  intros x y H
  sorry

end hyperbola_vertex_distance_l52_52736


namespace min_students_wearing_both_glasses_and_watches_l52_52603

theorem min_students_wearing_both_glasses_and_watches
  (n : ℕ)
  (H_glasses : n * 3 / 5 = 18)
  (H_watches : n * 5 / 6 = 25)
  (H_neither : n * 1 / 10 = 3):
  ∃ (x : ℕ), x = 16 := 
by
  sorry

end min_students_wearing_both_glasses_and_watches_l52_52603


namespace find_q_l52_52189

def f (q : ℝ) : ℝ := 3 * q - 3

theorem find_q (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end find_q_l52_52189


namespace points_on_parabola_l52_52907

theorem points_on_parabola (a : ℝ) (y1 y2 y3 : ℝ) 
  (h_a : a < -1) 
  (h1 : y1 = (a - 1)^2) 
  (h2 : y2 = a^2) 
  (h3 : y3 = (a + 1)^2) : 
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end points_on_parabola_l52_52907


namespace value_two_sd_below_mean_l52_52423

theorem value_two_sd_below_mean (mean : ℝ) (std_dev : ℝ) (h_mean : mean = 17.5) (h_std_dev : std_dev = 2.5) : 
  mean - 2 * std_dev = 12.5 := by
  -- proof omitted
  sorry

end value_two_sd_below_mean_l52_52423


namespace Albaszu_machine_productivity_l52_52661

theorem Albaszu_machine_productivity (x : ℝ) 
  (h1 : 1.5 * x = 25) : x = 16 := 
by 
  sorry

end Albaszu_machine_productivity_l52_52661


namespace necessary_but_not_sufficient_l52_52416

theorem necessary_but_not_sufficient (a c : ℝ) : 
  (c ≠ 0) → (∀ (x y : ℝ), ax^2 + y^2 = c → (c = 0 → false) ∧ (c ≠ 0 → (∃ x y : ℝ, ax^2 + y^2 = c))) :=
by
  sorry

end necessary_but_not_sufficient_l52_52416


namespace determine_k_and_solution_l52_52315

theorem determine_k_and_solution :
  ∃ (k : ℚ), (5 * k * x^2 + 30 * x + 10 = 0 → k = 9/2) ∧
    (∃ (x : ℚ), (5 * (9/2) * x^2 + 30 * x + 10 = 0) ∧ x = -2/3) := by
  sorry

end determine_k_and_solution_l52_52315


namespace initial_girls_count_l52_52822

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end initial_girls_count_l52_52822


namespace no_solution_exists_l52_52066

theorem no_solution_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x ^ 2 + f y) = 2 * x - f y :=
by
  sorry

end no_solution_exists_l52_52066


namespace local_min_at_neg_one_l52_52426

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_min_at_neg_one : 
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≥ f (-1) := by
  sorry

end local_min_at_neg_one_l52_52426


namespace net_gain_is_88837_50_l52_52436

def initial_home_value : ℝ := 500000
def first_sale_price : ℝ := 1.15 * initial_home_value
def first_purchase_price : ℝ := 0.95 * first_sale_price
def second_sale_price : ℝ := 1.1 * first_purchase_price
def second_purchase_price : ℝ := 0.9 * second_sale_price

def total_sales : ℝ := first_sale_price + second_sale_price
def total_purchases : ℝ := first_purchase_price + second_purchase_price
def net_gain_for_A : ℝ := total_sales - total_purchases

theorem net_gain_is_88837_50 : net_gain_for_A = 88837.50 := by
  -- proof steps would go here, but they are omitted per instructions
  sorry

end net_gain_is_88837_50_l52_52436


namespace problem_r_minus_s_l52_52660

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l52_52660


namespace find_age_of_older_friend_l52_52555

theorem find_age_of_older_friend (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) : 
  A = 104.25 :=
by
  sorry

end find_age_of_older_friend_l52_52555


namespace largest_fraction_l52_52411

variable {a b c d e f g h : ℝ}
variable {w x y z : ℝ}

/-- Given real numbers w, x, y, z such that w < x < y < z,
    the fraction z/w represents the largest value among the given fractions. -/
theorem largest_fraction (hwx : w < x) (hxy : x < y) (hyz : y < z) :
  (z / w) > (x / w) ∧ (z / w) > (y / x) ∧ (z / w) > (y / w) ∧ (z / w) > (z / x) :=
by
  sorry

end largest_fraction_l52_52411


namespace value_of_f_1_plus_g_4_l52_52342

def f (x : Int) : Int := 2 * x - 1
def g (x : Int) : Int := x + 1

theorem value_of_f_1_plus_g_4 : f (1 + g 4) = 11 := by
  sorry

end value_of_f_1_plus_g_4_l52_52342


namespace common_fraction_difference_l52_52130

def repeating_decimal := 23 / 99
def non_repeating_decimal := 23 / 100
def fraction_difference := 23 / 9900

theorem common_fraction_difference : repeating_decimal - non_repeating_decimal = fraction_difference := 
by
  sorry

end common_fraction_difference_l52_52130


namespace quadratic_equation_real_roots_l52_52940

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_equation_real_roots_l52_52940


namespace tips_fraction_l52_52126

theorem tips_fraction (S T : ℝ) (h : T / (S + T) = 0.6363636363636364) : T / S = 1.75 :=
sorry

end tips_fraction_l52_52126


namespace triangle_altitude_l52_52872

theorem triangle_altitude
  (base : ℝ) (height : ℝ) (side : ℝ)
  (h_base : base = 6)
  (h_side : side = 6)
  (area_triangle : ℝ) (area_square : ℝ)
  (h_area_square : area_square = side ^ 2)
  (h_area_equal : area_triangle = area_square)
  (h_area_triangle : area_triangle = (base * height) / 2) :
  height = 12 := 
by
  sorry

end triangle_altitude_l52_52872


namespace find_t_value_l52_52929

theorem find_t_value (x y z t : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z) :
  x + y + z + t = 10 → t = 4 :=
by
  -- Proof goes here
  sorry

end find_t_value_l52_52929


namespace car_catch_truck_l52_52878

theorem car_catch_truck (truck_speed car_speed : ℕ) (time_head_start : ℕ) (t : ℕ)
  (h1 : truck_speed = 45) (h2 : car_speed = 60) (h3 : time_head_start = 1) :
  45 * t + 45 = 60 * t → t = 3 := by
  intro h
  sorry

end car_catch_truck_l52_52878


namespace triangle_area_l52_52768

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : C = π / 3) : 
  (1/2 * a * b * Real.sin C) = (3 * Real.sqrt 3 / 2) :=
by
  sorry

end triangle_area_l52_52768


namespace solve_system_of_equations_l52_52073

theorem solve_system_of_equations (m b : ℤ) 
  (h1 : 3 * m + b = 11)
  (h2 : -4 * m - b = 11) : 
  m = -22 ∧ b = 77 :=
  sorry

end solve_system_of_equations_l52_52073


namespace smallest_positive_and_largest_negative_l52_52488

theorem smallest_positive_and_largest_negative:
  (∃ (a : ℤ), a > 0 ∧ ∀ (b : ℤ), b > 0 → b ≥ a ∧ a = 1) ∧
  (∃ (c : ℤ), c < 0 ∧ ∀ (d : ℤ), d < 0 → d ≤ c ∧ c = -1) :=
by
  sorry

end smallest_positive_and_largest_negative_l52_52488


namespace appleJuicePercentageIsCorrect_l52_52631

-- Define the initial conditions
def MikiHas : ℕ × ℕ := (15, 10) -- Miki has 15 apples and 10 bananas

-- Define the juice extraction rates
def appleJuicePerApple : ℚ := 9 / 3 -- 9 ounces from 3 apples
def bananaJuicePerBanana : ℚ := 10 / 2 -- 10 ounces from 2 bananas

-- Define the number of apples and bananas used for the blend
def applesUsed : ℕ := 5
def bananasUsed : ℕ := 4

-- Calculate the total juice extracted
def appleJuice : ℚ := applesUsed * appleJuicePerApple
def bananaJuice : ℚ := bananasUsed * bananaJuicePerBanana

-- Calculate the total juice and percentage of apple juice
def totalJuice : ℚ := appleJuice + bananaJuice
def percentageAppleJuice : ℚ := (appleJuice / totalJuice) * 100

theorem appleJuicePercentageIsCorrect : percentageAppleJuice = 42.86 := by
  sorry

end appleJuicePercentageIsCorrect_l52_52631


namespace find_value_divide_subtract_l52_52314

theorem find_value_divide_subtract :
  (Number = 8 * 156 + 2) → 
  (CorrectQuotient = Number / 5) → 
  (Value = CorrectQuotient - 3) → 
  Value = 247 :=
by
  intros h1 h2 h3
  sorry

end find_value_divide_subtract_l52_52314


namespace lioness_age_l52_52176

theorem lioness_age (H L : ℕ) 
  (h1 : L = 2 * H) 
  (h2 : (H / 2 + 5) + (L / 2 + 5) = 19) : 
  L = 12 :=
sorry

end lioness_age_l52_52176


namespace production_rate_equation_l52_52701

theorem production_rate_equation (x : ℝ) (h : x > 0) :
  3000 / x - 3000 / (2 * x) = 5 :=
sorry

end production_rate_equation_l52_52701


namespace book_distribution_methods_l52_52078

theorem book_distribution_methods :
  let novels := 2
  let picture_books := 2
  let students := 3
  (number_ways : ℕ) = 12 :=
by
  sorry

end book_distribution_methods_l52_52078


namespace lifting_to_bodyweight_ratio_l52_52969

variable (t : ℕ) (w : ℕ) (p : ℕ) (delta_w : ℕ)

def lifting_total_after_increase (t : ℕ) (p : ℕ) : ℕ :=
  t + (t * p / 100)

def bodyweight_after_increase (w : ℕ) (delta_w : ℕ) : ℕ :=
  w + delta_w

theorem lifting_to_bodyweight_ratio (h_t : t = 2200) (h_w : w = 245) (h_p : p = 15) (h_delta_w : delta_w = 8) :
  lifting_total_after_increase t p / bodyweight_after_increase w delta_w = 10 :=
  by
    -- Use the given conditions
    rw [h_t, h_w, h_p, h_delta_w]
    -- Calculation steps are omitted, directly providing the final assertion
    sorry

end lifting_to_bodyweight_ratio_l52_52969


namespace smooth_transition_l52_52925

theorem smooth_transition (R : ℝ) (x₀ y₀ : ℝ) :
  ∃ m : ℝ, ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = R^2 → y - y₀ = m * (x - x₀) :=
sorry

end smooth_transition_l52_52925


namespace find_f_neg2_l52_52802

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Define the conditions and statement to be proved
theorem find_f_neg2 (a b : ℝ) (h1 : f a b 2 = 9) : f a b (-2) = 13 :=
by
  -- Conditions lead to the conclusion to be proved
  sorry

end find_f_neg2_l52_52802


namespace y_sum_equals_three_l52_52532

noncomputable def sum_of_y_values (solutions : List (ℝ × ℝ × ℝ)) : ℝ :=
  solutions.foldl (fun acc (_, y, _) => acc + y) 0

theorem y_sum_equals_three (solutions : List (ℝ × ℝ × ℝ))
  (h1 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → x + y * z = 5)
  (h2 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → y + x * z = 8)
  (h3 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → z + x * y = 12) :
  sum_of_y_values solutions = 3 := sorry

end y_sum_equals_three_l52_52532


namespace sum_of_a_and_b_l52_52179

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l52_52179


namespace playground_dimensions_l52_52224

theorem playground_dimensions 
  (a b : ℕ) 
  (h1 : (a - 2) * (b - 2) = 4) : a * b = 2 * a + 2 * b :=
by
  sorry

end playground_dimensions_l52_52224


namespace valid_pairs_for_area_18_l52_52158

theorem valid_pairs_for_area_18 (w l : ℕ) (hw : 0 < w) (hl : 0 < l) (h_area : w * l = 18) (h_lt : w < l) :
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) :=
sorry

end valid_pairs_for_area_18_l52_52158


namespace probability_adjacent_vertices_of_octagon_l52_52636

theorem probability_adjacent_vertices_of_octagon :
  let num_vertices := 8;
  let adjacent_vertices (v1 v2 : Fin num_vertices) : Prop := 
    (v2 = (v1 + 1) % num_vertices) ∨ (v2 = (v1 - 1 + num_vertices) % num_vertices);
  let total_vertices := num_vertices - 1;
  (2 : ℚ) / total_vertices = (2 / 7 : ℚ) :=
by
  -- Proof goes here
  sorry

end probability_adjacent_vertices_of_octagon_l52_52636


namespace find_B_l52_52874

variable {A B C a b c : Real}

noncomputable def B_value (A B C a b c : Real) : Prop :=
  B = 2 * Real.pi / 3

theorem find_B 
  (h_triangle: a^2 + b^2 + c^2 = 2*a*b*Real.cos C)
  (h_cos_eq: (2 * a + c) * Real.cos B + b * Real.cos C = 0) : 
  B_value A B C a b c :=
by
  sorry

end find_B_l52_52874


namespace eggs_problem_solution_l52_52981

theorem eggs_problem_solution :
  ∃ (n x : ℕ), 
  (120 * n = 206 * x) ∧
  (n = 103) ∧
  (x = 60) :=
by sorry

end eggs_problem_solution_l52_52981


namespace inequality_of_fractions_l52_52297

theorem inequality_of_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (x + y)) + (y / (y + z)) + (z / (z + x)) ≤ 2 := 
by 
  sorry

end inequality_of_fractions_l52_52297


namespace hexagon_ratio_identity_l52_52360

theorem hexagon_ratio_identity
  (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (AB BC CD DE EF FA : ℝ)
  (angle_B angle_D angle_F : ℝ)
  (h1 : AB / BC * CD / DE * EF / FA = 1)
  (h2 : angle_B + angle_D + angle_F = 360) :
  (BC / AC * AE / EF * FD / DB = 1) := sorry

end hexagon_ratio_identity_l52_52360


namespace equation_of_line_l52_52180

theorem equation_of_line (l : ℝ → ℝ) :
  (∀ (P : ℝ × ℝ), P = (4, 2) → 
    ∃ (a b : ℝ), ((P = ( (4 - a), (2 - b)) ∨ P = ( (4 + a), (2 + b))) ∧ 
    ((4 - a)^2 / 36 + (2 - b)^2 / 9 = 1) ∧ ((4 + a)^2 / 36 + (2 + b)^2 / 9 = 1)) ∧
    (P.2 = l P.1)) →
  (∀ (x y : ℝ), y = l x ↔ 2 * x + 3 * y - 16 = 0) :=
by
  intros h P hp
  sorry -- Placeholder for the proof

end equation_of_line_l52_52180


namespace quadratic_real_roots_l52_52635

theorem quadratic_real_roots (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 - 1 + m = 0 ∧ x2^2 + 2 * x2 - 1 + m = 0) ↔ m ≤ 2 :=
by
  sorry

end quadratic_real_roots_l52_52635


namespace fan_airflow_weekly_l52_52183

def fan_airflow_per_second : ℕ := 10
def fan_work_minutes_per_day : ℕ := 10
def minutes_to_seconds (m : ℕ) : ℕ := m * 60
def days_per_week : ℕ := 7

theorem fan_airflow_weekly : 
  (fan_airflow_per_second * (minutes_to_seconds fan_work_minutes_per_day) * days_per_week) = 42000 := 
by
  sorry

end fan_airflow_weekly_l52_52183


namespace find_x_value_l52_52121

theorem find_x_value : (8 = 2^3) ∧ (8 * 8^32 = 8^33) ∧ (8^33 = 2^99) → ∃ x, 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 = 2^x ∧ x = 99 :=
by
  intros h
  sorry

end find_x_value_l52_52121


namespace binomial_square_value_l52_52561

theorem binomial_square_value (c : ℝ) : (∃ d : ℝ, 16 * x^2 + 40 * x + c = (4 * x + d) ^ 2) → c = 25 :=
by
  sorry

end binomial_square_value_l52_52561


namespace solve_system_l52_52316

noncomputable def sqrt_cond (x y : ℝ) : Prop :=
  Real.sqrt ((3 * x - 2 * y) / (2 * x)) + Real.sqrt ((2 * x) / (3 * x - 2 * y)) = 2

noncomputable def quad_cond (x y : ℝ) : Prop :=
  x^2 - 18 = 2 * y * (4 * y - 9)

theorem solve_system (x y : ℝ) : sqrt_cond x y ∧ quad_cond x y ↔ (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 1.5) :=
by
  sorry

end solve_system_l52_52316


namespace initial_amount_of_milk_l52_52716

theorem initial_amount_of_milk (M : ℝ) (h : 0 < M) (h2 : 0.10 * M = 0.05 * (M + 20)) : M = 20 := 
sorry

end initial_amount_of_milk_l52_52716


namespace shell_placements_l52_52161

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem shell_placements : factorial 14 / 7 = 10480142147302400 := by
  sorry

end shell_placements_l52_52161


namespace simplify_expression_l52_52771

theorem simplify_expression :
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by sorry

end simplify_expression_l52_52771


namespace john_needs_packs_l52_52001

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end john_needs_packs_l52_52001


namespace smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l52_52217

theorem smallest_prime_factor_of_5_pow_5_minus_5_pow_3 : Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p ∧ p ∣ (5^5 - 5^3) → p ≥ 2) := by
  sorry

end smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l52_52217


namespace clea_ride_escalator_time_l52_52134

theorem clea_ride_escalator_time (x y k : ℝ) (h1 : 80 * x = y) (h2 : 30 * (x + k) = y) : (y / k) + 5 = 53 :=
by {
  sorry
}

end clea_ride_escalator_time_l52_52134


namespace A_scores_2_points_B_scores_at_least_2_points_l52_52285

-- Define the probabilities of outcomes.
def prob_A_win := 0.5
def prob_A_lose := 0.3
def prob_A_draw := 0.2

-- Calculate the probability of A scoring 2 points.
theorem A_scores_2_points : 
    (prob_A_win * prob_A_lose + prob_A_lose * prob_A_win + prob_A_draw * prob_A_draw) = 0.34 :=
by
  sorry

-- Calculate the probability of B scoring at least 2 points.
theorem B_scores_at_least_2_points : 
    (1 - (prob_A_win * prob_A_win + (prob_A_win * prob_A_draw + prob_A_draw * prob_A_win))) = 0.55 :=
by
  sorry

end A_scores_2_points_B_scores_at_least_2_points_l52_52285


namespace div_product_four_consecutive_integers_l52_52581

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l52_52581


namespace monotonically_increasing_a_range_l52_52296

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) * Real.exp x

theorem monotonically_increasing_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≥ 0) ↔ 1 ≤ a  :=
by
  sorry

end monotonically_increasing_a_range_l52_52296


namespace find_k_l52_52075

open Real

variables (a b : ℝ × ℝ) (k : ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

theorem find_k (ha : a = (1, 2)) (hb : b = (-2, 4)) (perpendicular : dot_product (k • a + b) b = 0) :
  k = - (10 / 3) :=
by
  sorry

end find_k_l52_52075


namespace geometric_sequence_product_l52_52261

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h : a 4 = 4) :
  a 2 * a 6 = 16 := by
  -- Definition of geomtric sequence
  -- a_n = a_0 * r^n
  -- Using the fact that the product of corresponding terms equidistant from two ends is constant
  sorry

end geometric_sequence_product_l52_52261


namespace find_page_added_twice_l52_52427

theorem find_page_added_twice (m p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ m) (h3 : (m * (m + 1)) / 2 + p = 2550) : p = 6 :=
sorry

end find_page_added_twice_l52_52427


namespace snack_eaters_left_l52_52147

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end snack_eaters_left_l52_52147


namespace correct_mark_l52_52475

theorem correct_mark (x : ℝ) (n : ℝ) (avg_increase : ℝ) :
  n = 40 → avg_increase = 1 / 2 → (83 - x) / n = avg_increase → x = 63 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l52_52475


namespace largest_n_for_factorization_l52_52025

theorem largest_n_for_factorization :
  ∃ (n : ℤ), (∀ (A B : ℤ), AB = 96 → n = 4 * B + A) ∧ (n = 385) := by
  sorry

end largest_n_for_factorization_l52_52025


namespace total_sum_of_rupees_l52_52318

theorem total_sum_of_rupees :
  ∃ (total_coins : ℕ) (paise20_coins : ℕ) (paise25_coins : ℕ),
    total_coins = 344 ∧ paise20_coins = 300 ∧ paise25_coins = total_coins - paise20_coins ∧
    (60 + (44 * 0.25)) = 71 :=
by
  sorry

end total_sum_of_rupees_l52_52318


namespace treehouse_total_planks_l52_52293

theorem treehouse_total_planks (T : ℕ) 
    (h1 : T / 4 + T / 2 + 20 + 30 = T) : T = 200 :=
sorry

end treehouse_total_planks_l52_52293


namespace capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l52_52420

-- Step 1: Define the capacities of type A and B cars
def typeACarCapacity := 3
def typeBCarCapacity := 4

-- Step 2: Prove transportation capacities x and y
theorem capacities_correct (x y: ℕ) (h1 : 3 * x + 2 * y = 17) (h2 : 2 * x + 3 * y = 18) :
    x = typeACarCapacity ∧ y = typeBCarCapacity :=
by
  sorry

-- Step 3: Define a rental plan to transport 35 tons
theorem rental_plan_exists (a b : ℕ) : 3 * a + 4 * b = 35 :=
by
  sorry

-- Step 4: Prove the minimal cost solution
def typeACarCost := 300
def typeBCarCost := 320

def rentalCost (a b : ℕ) : ℕ := a * typeACarCost + b * typeBCarCost

theorem minimal_rental_cost_exists :
    ∃ a b, 3 * a + 4 * b = 35 ∧ rentalCost a b = 2860 :=
by
  sorry

end capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l52_52420


namespace minimum_meals_needed_l52_52035

theorem minimum_meals_needed (total_jam : ℝ) (max_per_meal : ℝ) (jars : ℕ) (max_jar_weight : ℝ):
  (total_jam = 50) → (max_per_meal = 5) → (jars ≥ 50) → (max_jar_weight ≤ 1) →
  (jars * max_jar_weight = total_jam) →
  jars ≥ 12 := sorry

end minimum_meals_needed_l52_52035


namespace rationalize_denominator_l52_52010

theorem rationalize_denominator :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let A := 25
  let B := 20
  let C := 16
  let D := 1
  (1 / (a - b)) = ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / D ∧ (A + B + C + D = 62) := by
  sorry

end rationalize_denominator_l52_52010


namespace meaningful_expression_l52_52480

theorem meaningful_expression (x : ℝ) : (∃ y, y = 5 / (Real.sqrt (x + 1))) ↔ x > -1 :=
by
  sorry

end meaningful_expression_l52_52480


namespace jen_ducks_l52_52332

theorem jen_ducks (c d : ℕ) (h1 : d = 4 * c + 10) (h2 : c + d = 185) : d = 150 := by
  sorry

end jen_ducks_l52_52332


namespace find_B_l52_52560

noncomputable def A : ℝ := 1 / 49
noncomputable def C : ℝ := -(1 / 7)

theorem find_B :
  (∀ x : ℝ, 1 / (x^3 + 2 * x^2 - 25 * x - 50) 
            = (A / (x - 2)) + (B / (x + 5)) + (C / ((x + 5)^2))) 
    → B = - (11 / 490) :=
sorry

end find_B_l52_52560


namespace andy_cavity_per_candy_cane_l52_52990

theorem andy_cavity_per_candy_cane 
  (cavities_per_candy_cane : ℝ)
  (candy_caned_from_parents : ℝ := 2)
  (candy_caned_each_teacher : ℝ := 3)
  (num_teachers : ℝ := 4)
  (allowance_factor : ℝ := 1/7)
  (total_cavities : ℝ := 16) :
  let total_given_candy : ℝ := candy_caned_from_parents + candy_caned_each_teacher * num_teachers
  let total_bought_candy : ℝ := allowance_factor * total_given_candy
  let total_candy : ℝ := total_given_candy + total_bought_candy
  total_candy / total_cavities = cavities_per_candy_cane :=
by
  sorry

end andy_cavity_per_candy_cane_l52_52990


namespace toy_store_shelves_l52_52105

theorem toy_store_shelves (initial_bears : ℕ) (shipment_bears : ℕ) (bears_per_shelf : ℕ)
                          (h_initial : initial_bears = 5) (h_shipment : shipment_bears = 7) 
                          (h_per_shelf : bears_per_shelf = 6) : 
                          (initial_bears + shipment_bears) / bears_per_shelf = 2 :=
by
  sorry

end toy_store_shelves_l52_52105


namespace weeks_per_mouse_correct_l52_52813

def years_in_decade : ℕ := 10
def weeks_per_year : ℕ := 52
def total_mice : ℕ := 130

def total_weeks_in_decade : ℕ := years_in_decade * weeks_per_year
def weeks_per_mouse : ℕ := total_weeks_in_decade / total_mice

theorem weeks_per_mouse_correct : weeks_per_mouse = 4 := 
sorry

end weeks_per_mouse_correct_l52_52813


namespace find_p_l52_52943

variables {m n p : ℚ}

theorem find_p (h1 : m = 3 * n + 5) (h2 : (m + 2) = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end find_p_l52_52943


namespace find_principal_sum_l52_52618

theorem find_principal_sum (R P : ℝ) 
  (h1 : (3 * P * (R + 1) / 100 - 3 * P * R / 100) = 72) : 
  P = 2400 := 
by 
  sorry

end find_principal_sum_l52_52618


namespace smallest_n_divisible_by_2009_l52_52148

theorem smallest_n_divisible_by_2009 : ∃ n : ℕ, n > 1 ∧ (n^2 * (n - 1)) % 2009 = 0 ∧ (∀ m : ℕ, m > 1 → (m^2 * (m - 1)) % 2009 = 0 → m ≥ n) :=
by
  sorry

end smallest_n_divisible_by_2009_l52_52148


namespace students_taking_german_l52_52021

theorem students_taking_german
  (total_students : ℕ)
  (french_students : ℕ)
  (both_courses_students : ℕ)
  (no_course_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : both_courses_students = 9)
  (h4 : no_course_students = 33)
  : ∃ german_students : ℕ, german_students = 22 := 
by
  -- proof can be filled in here
  sorry

end students_taking_german_l52_52021


namespace sqrt_equation_solution_l52_52546

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_equation_solution_l52_52546


namespace largest_quantity_l52_52088

noncomputable def D := (2007 / 2006) + (2007 / 2008)
noncomputable def E := (2007 / 2008) + (2009 / 2008)
noncomputable def F := (2008 / 2007) + (2008 / 2009)

theorem largest_quantity : D > E ∧ D > F :=
by { sorry }

end largest_quantity_l52_52088


namespace quadratic_equation_has_real_root_l52_52698

theorem quadratic_equation_has_real_root
  (a c m n : ℝ) :
  ∃ x : ℝ, c * x^2 + m * x - a = 0 ∨ ∃ y : ℝ, a * y^2 + n * y + c = 0 :=
by
  -- Proof omitted
  sorry

end quadratic_equation_has_real_root_l52_52698


namespace total_selling_price_l52_52630

def original_price : ℝ := 120
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.15

def sale_price (original_price discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent)

def final_price (sale_price tax_percent : ℝ) : ℝ :=
  sale_price * (1 + tax_percent)

theorem total_selling_price :
  final_price (sale_price original_price discount_percent) tax_percent = 96.6 :=
sorry

end total_selling_price_l52_52630


namespace algebraic_expression_value_l52_52807

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b = 2) : 
  (-a * (-2) ^ 2 + b * (-2) + 1) = -1 :=
by
  sorry

end algebraic_expression_value_l52_52807


namespace num_diamonds_F10_l52_52265

def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

theorem num_diamonds_F10 : num_diamonds 10 = 112 := by
  sorry

end num_diamonds_F10_l52_52265


namespace intersection_of_A_and_B_l52_52167

def U := Set ℝ
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := {x : ℝ | x < -1}
def C := {x : ℝ | -2 ≤ x ∧ x < -1}

theorem intersection_of_A_and_B : A ∩ B = C :=
by sorry

end intersection_of_A_and_B_l52_52167


namespace convert_binary_to_decimal_l52_52008

theorem convert_binary_to_decimal : (1 * 2^2 + 1 * 2^1 + 1 * 2^0) = 7 := by
  sorry

end convert_binary_to_decimal_l52_52008


namespace problem_l52_52748

-- Define i as the imaginary unit
def i : ℂ := Complex.I

-- The statement to be proved
theorem problem : i * (1 - i) ^ 2 = 2 := by
  sorry

end problem_l52_52748


namespace product_of_two_numbers_l52_52628

theorem product_of_two_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 560) (h_hcf : Nat.gcd a b = 75) :
  a * b = 42000 :=
by
  sorry

end product_of_two_numbers_l52_52628


namespace find_a1_plus_a2_l52_52985

theorem find_a1_plus_a2 (x : ℝ) (a0 a1 a2 a3 : ℝ) 
  (h : (1 - 2/x)^3 = a0 + a1 * (1/x) + a2 * (1/x)^2 + a3 * (1/x)^3) : 
  a1 + a2 = 6 :=
by
  sorry

end find_a1_plus_a2_l52_52985


namespace smallest_positive_integer_remainder_conditions_l52_52240

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l52_52240


namespace problem_l52_52880

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def max_value_in (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v ∧ (∃ z, a ≤ z ∧ z ≤ b ∧ f z = v)

theorem problem
  (h_even : even_function f)
  (h_decreasing : decreasing_on f (-5) (-2))
  (h_max : max_value_in f (-5) (-2) 7) :
  increasing_on f 2 5 ∧ max_value_in f 2 5 7 :=
by
  sorry

end problem_l52_52880


namespace fraction_sum_product_roots_of_quadratic_l52_52328

theorem fraction_sum_product_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) →
  (x1 ≠ x2) →
  (x1 + x2) / (x1 * x2) = -1 / 4 := 
by
  sorry

end fraction_sum_product_roots_of_quadratic_l52_52328


namespace a_plus_b_in_D_l52_52115

def setA : Set ℤ := {x | ∃ k : ℤ, x = 4 * k}
def setB : Set ℤ := {x | ∃ m : ℤ, x = 4 * m + 1}
def setC : Set ℤ := {x | ∃ n : ℤ, x = 4 * n + 2}
def setD : Set ℤ := {x | ∃ t : ℤ, x = 4 * t + 3}

theorem a_plus_b_in_D (a b : ℤ) (ha : a ∈ setB) (hb : b ∈ setC) : a + b ∈ setD := by
  sorry

end a_plus_b_in_D_l52_52115


namespace glass_panels_in_neighborhood_l52_52209

def total_glass_panels_in_neighborhood := 
  let double_windows_downstairs : ℕ := 6
  let glass_panels_per_double_window_downstairs : ℕ := 4
  let single_windows_upstairs : ℕ := 8
  let glass_panels_per_single_window_upstairs : ℕ := 3
  let bay_windows : ℕ := 2
  let glass_panels_per_bay_window : ℕ := 6
  let houses : ℕ := 10

  let glass_panels_in_one_house : ℕ := 
    (double_windows_downstairs * glass_panels_per_double_window_downstairs) +
    (single_windows_upstairs * glass_panels_per_single_window_upstairs) +
    (bay_windows * glass_panels_per_bay_window)

  houses * glass_panels_in_one_house

theorem glass_panels_in_neighborhood : total_glass_panels_in_neighborhood = 600 := by
  -- Calculation steps skipped
  sorry

end glass_panels_in_neighborhood_l52_52209


namespace vegetarian_count_l52_52543

theorem vegetarian_count (only_veg only_non_veg both_veg_non_veg : ℕ) 
  (h1 : only_veg = 19) (h2 : only_non_veg = 9) (h3 : both_veg_non_veg = 12) : 
  (only_veg + both_veg_non_veg = 31) :=
by
  -- We leave the proof here
  sorry

end vegetarian_count_l52_52543


namespace sasha_remainder_l52_52097

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l52_52097


namespace sqrt_three_pow_divisible_l52_52393

/-- For any non-negative integer n, (1 + sqrt 3)^(2*n + 1) is divisible by 2^(n + 1) -/
theorem sqrt_three_pow_divisible (n : ℕ) :
  ∃ k : ℕ, (⌊(1 + Real.sqrt 3)^(2 * n + 1)⌋ : ℝ) = k * 2^(n + 1) :=
sorry

end sqrt_three_pow_divisible_l52_52393


namespace polynomial_value_at_3_l52_52235

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem polynomial_value_at_3 : f 3 = 1209.4 := 
by
  sorry

end polynomial_value_at_3_l52_52235


namespace distinct_nonzero_digits_sum_l52_52725

theorem distinct_nonzero_digits_sum
  (x y z w : Nat)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hxw : x ≠ w)
  (hyz : y ≠ z)
  (hyw : y ≠ w)
  (hzw : z ≠ w)
  (h1 : w + x = 10)
  (h2 : y + w = 9)
  (h3 : z + x = 9) :
  x + y + z + w = 18 :=
sorry

end distinct_nonzero_digits_sum_l52_52725


namespace saree_blue_stripes_l52_52762

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end saree_blue_stripes_l52_52762


namespace greatest_third_term_of_arithmetic_sequence_l52_52391

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l52_52391


namespace triangle_area_l52_52624

/-- 
In a triangle ABC, given that ∠B=30°, AB=2√3, and AC=2, 
prove that the area of the triangle ABC is either √3 or 2√3.
 -/
theorem triangle_area (B : Real) (AB AC : Real) 
  (h_B : B = 30) (h_AB : AB = 2 * Real.sqrt 3) (h_AC : AC = 2) :
  ∃ S : Real, (S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3) := 
by 
  sorry

end triangle_area_l52_52624


namespace midpoint_condition_l52_52905

theorem midpoint_condition (c : ℝ) :
  (∃ A B : ℝ × ℝ,
    A ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    B ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    A ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    B ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2) = 2017
  ) ↔
  c = 4031 := sorry

end midpoint_condition_l52_52905


namespace set_intersection_complement_l52_52166

variable (U : Set ℕ)
variable (P Q : Set ℕ)

theorem set_intersection_complement {U : Set ℕ} {P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5, 6}) 
  (hP : P = {1, 2, 3, 4}) 
  (hQ : Q = {3, 4, 5, 6}) : 
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l52_52166


namespace renne_savings_ratio_l52_52425

theorem renne_savings_ratio (ME CV N : ℕ) (h_ME : ME = 4000) (h_CV : CV = 16000) (h_N : N = 8) :
  (CV / N : ℕ) / ME = 1 / 2 :=
by
  sorry

end renne_savings_ratio_l52_52425


namespace tetrahedron_labeling_count_l52_52367

def is_valid_tetrahedron_labeling (labeling : Fin 4 → ℕ) : Prop :=
  let f1 := labeling 0 + labeling 1 + labeling 2
  let f2 := labeling 0 + labeling 1 + labeling 3
  let f3 := labeling 0 + labeling 2 + labeling 3
  let f4 := labeling 1 + labeling 2 + labeling 3
  labeling 0 + labeling 1 + labeling 2 + labeling 3 = 10 ∧ 
  f1 = f2 ∧ f2 = f3 ∧ f3 = f4

theorem tetrahedron_labeling_count : 
  ∃ (n : ℕ), n = 3 ∧ (∃ (labelings: Finset (Fin 4 → ℕ)), 
  ∀ labeling ∈ labelings, is_valid_tetrahedron_labeling labeling) :=
sorry

end tetrahedron_labeling_count_l52_52367


namespace A_investment_l52_52960

-- Conditions as definitions
def B_investment := 72000
def C_investment := 81000
def C_profit := 36000
def Total_profit := 80000

-- Statement to prove
theorem A_investment : 
  ∃ (x : ℕ), x = 27000 ∧
  (C_profit / Total_profit = (9 : ℕ) / 20) ∧
  (C_investment / (x + B_investment + C_investment) = (9 : ℕ) / 20) :=
by sorry

end A_investment_l52_52960


namespace solve_for_t_l52_52418

theorem solve_for_t (s t : ℝ) (h1 : 12 * s + 8 * t = 160) (h2 : s = t^2 + 2) :
  t = (Real.sqrt 103 - 1) / 3 :=
sorry

end solve_for_t_l52_52418


namespace kelly_spends_correct_amount_l52_52128

noncomputable def total_cost_with_discount : ℝ :=
  let mango_cost_per_pound := (0.60 : ℝ) * 2
  let orange_cost_per_pound := (0.40 : ℝ) * 4
  let mango_total_cost := 5 * mango_cost_per_pound
  let orange_total_cost := 5 * orange_cost_per_pound
  let total_cost_without_discount := mango_total_cost + orange_total_cost
  let discount := 0.10 * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount
  total_cost_with_discount

theorem kelly_spends_correct_amount :
  total_cost_with_discount = 12.60 := by
  sorry

end kelly_spends_correct_amount_l52_52128


namespace find_x_l52_52190

theorem find_x (x : ℤ) :
  3 < x ∧ x < 10 →
  5 < x ∧ x < 18 →
  -2 < x ∧ x < 9 →
  0 < x ∧ x < 8 →
  x + 1 < 9 →
  x = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_x_l52_52190


namespace severe_flood_probability_next_10_years_l52_52640

variable (A B C : Prop)
variable (P : Prop → ℝ)
variable (P_A : P A = 0.8)
variable (P_B : P B = 0.85)
variable (thirty_years_no_flood : ¬A)

theorem severe_flood_probability_next_10_years :
  P C = (P B - P A) / (1 - P A) := by
  sorry

end severe_flood_probability_next_10_years_l52_52640


namespace cars_15th_time_l52_52970

noncomputable def minutes_since_8am (hour : ℕ) (minute : ℕ) : ℕ :=
  hour * 60 + minute

theorem cars_15th_time :
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  total_time = expected_time :=
by
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  show total_time = expected_time
  sorry

end cars_15th_time_l52_52970


namespace monotonicity_f_a_eq_1_domain_condition_inequality_condition_l52_52898

noncomputable def f (x a : ℝ) := (Real.log (x^2 - 2 * x + a)) / (x - 1)

theorem monotonicity_f_a_eq_1 :
  ∀ x : ℝ, 1 < x → 
  (f x 1 < f (e + 1) 1 → 
   ∀ y, 1 < y ∧ y < e + 1 → f y 1 < f (e + 1) 1) ∧ 
  (f (e + 1) 1 < f x 1 → 
   ∀ z, e + 1 < z → f (e + 1) 1 < f z 1) :=
sorry

theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 1) → x^2 - 2 * x + a > 0) ↔ a ≥ 1 :=
sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f x a < (x - 1) * Real.exp x)) ↔ (1 + 1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

end monotonicity_f_a_eq_1_domain_condition_inequality_condition_l52_52898


namespace hexagon_area_l52_52231

theorem hexagon_area (s t_height : ℕ) (tri_area rect_area : ℕ) :
    s = 2 →
    t_height = 4 →
    tri_area = 1 / 2 * s * t_height →
    rect_area = (s + s + s) * (t_height + t_height) →
    rect_area - 4 * tri_area = 32 :=
by
  sorry

end hexagon_area_l52_52231


namespace sum_of_three_numbers_l52_52113

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a <= 10) (h2 : 10 <= c)
  (h3 : (a + 10 + c) / 3 = a + 8)
  (h4 : (a + 10 + c) / 3 = c - 20) :
  a + 10 + c = 66 :=
by
  sorry

end sum_of_three_numbers_l52_52113


namespace final_game_deficit_l52_52873

-- Define the points for each scoring action
def free_throw_points := 1
def three_pointer_points := 3
def jump_shot_points := 2
def layup_points := 2
def and_one_points := layup_points + free_throw_points

-- Define the points scored by Liz
def liz_free_throws := 5 * free_throw_points
def liz_three_pointers := 4 * three_pointer_points
def liz_jump_shots := 5 * jump_shot_points
def liz_and_one := and_one_points

def liz_points := liz_free_throws + liz_three_pointers + liz_jump_shots + liz_and_one

-- Define the points scored by Taylor
def taylor_three_pointers := 2 * three_pointer_points
def taylor_jump_shots := 3 * jump_shot_points

def taylor_points := taylor_three_pointers + taylor_jump_shots

-- Define the points for Liz's team
def team_points := liz_points + taylor_points

-- Define the points scored by the opposing team players
def opponent_player1_points := 4 * three_pointer_points

def opponent_player2_jump_shots := 4 * jump_shot_points
def opponent_player2_free_throws := 2 * free_throw_points
def opponent_player2_points := opponent_player2_jump_shots + opponent_player2_free_throws

def opponent_player3_jump_shots := 2 * jump_shot_points
def opponent_player3_three_pointer := 1 * three_pointer_points
def opponent_player3_points := opponent_player3_jump_shots + opponent_player3_three_pointer

-- Define the points for the opposing team
def opponent_team_points := opponent_player1_points + opponent_player2_points + opponent_player3_points

-- Initial deficit
def initial_deficit := 25

-- Final net scoring in the final quarter
def net_quarter_scoring := team_points - opponent_team_points

-- Final deficit
def final_deficit := initial_deficit - net_quarter_scoring

theorem final_game_deficit : final_deficit = 12 := by
  sorry

end final_game_deficit_l52_52873


namespace loan_principal_and_repayment_amount_l52_52834

theorem loan_principal_and_repayment_amount (P R : ℝ) (r : ℝ) (years : ℕ) (total_interest : ℝ)
    (h1: r = 0.12)
    (h2: years = 3)
    (h3: total_interest = 5400)
    (h4: total_interest / years = R)
    (h5: R = P * r) :
    P = 15000 ∧ R = 1800 :=
sorry

end loan_principal_and_repayment_amount_l52_52834


namespace smallest_sum_l52_52626

theorem smallest_sum (x y : ℕ) (h : (2010 / 2011 : ℚ) < x / y ∧ x / y < (2011 / 2012 : ℚ)) : x + y = 8044 :=
sorry

end smallest_sum_l52_52626


namespace Neil_candy_collected_l52_52516

variable (M H N : ℕ)

-- Conditions
def Maggie_collected := M = 50
def Harper_collected := H = M + (30 * M) / 100
def Neil_collected := N = H + (40 * H) / 100

-- Theorem statement 
theorem Neil_candy_collected
  (hM : Maggie_collected M)
  (hH : Harper_collected M H)
  (hN : Neil_collected H N) :
  N = 91 := by
  sorry

end Neil_candy_collected_l52_52516


namespace f_g_evaluation_l52_52596

-- Definitions of the functions g and f
def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3 * x - 2

-- Goal: Prove that f(g(2)) = 22
theorem f_g_evaluation : f (g 2) = 22 :=
by
  sorry

end f_g_evaluation_l52_52596


namespace quadratic_completeness_l52_52988

noncomputable def quad_eqn : Prop :=
  ∃ b c : ℤ, (∀ x : ℝ, (x^2 - 10 * x + 15 = 0) ↔ ((x + b)^2 = c)) ∧ b + c = 5

theorem quadratic_completeness : quad_eqn :=
sorry

end quadratic_completeness_l52_52988


namespace mixture_price_l52_52651

-- Define constants
noncomputable def V1 (X : ℝ) : ℝ := 3.50 * X
noncomputable def V2 : ℝ := 4.30 * 6.25
noncomputable def W2 : ℝ := 6.25
noncomputable def W1 (X : ℝ) : ℝ := X

-- Define the total mixture weight condition
theorem mixture_price (X : ℝ) (P : ℝ) (h1 : W1 X + W2 = 10) (h2 : 10 * P = V1 X + V2) :
  P = 4 := by
  sorry

end mixture_price_l52_52651


namespace distance_on_dirt_section_distance_on_muddy_section_l52_52046

section RaceProblem

variables {v_h v_d v_m : ℕ} (initial_gap : ℕ)

-- Problem conditions
def highway_speed := 150 -- km/h
def dirt_road_speed := 60 -- km/h
def muddy_section_speed := 18 -- km/h
def initial_gap_start := 300 -- meters

-- Convert km/h to m/s
def to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Speeds in m/s
def highway_speed_mps := to_m_per_s highway_speed
def dirt_road_speed_mps := to_m_per_s dirt_road_speed
def muddy_section_speed_mps := to_m_per_s muddy_section_speed

-- Questions
theorem distance_on_dirt_section :
  ∃ (d : ℕ), (d = 120) :=
sorry

theorem distance_on_muddy_section :
  ∃ (d : ℕ), (d = 36) :=
sorry

end RaceProblem

end distance_on_dirt_section_distance_on_muddy_section_l52_52046


namespace radius_inner_circle_l52_52106

theorem radius_inner_circle (s : ℝ) (n : ℕ) (d : ℝ) (r : ℝ) :
  s = 4 ∧ n = 16 ∧ d = s / 4 ∧ ∀ k, k = d / 2 → r = (Real.sqrt (s^2 / 4 + k^2) - k) / 2 
  → r = Real.sqrt 4.25 / 2 :=
by
  sorry

end radius_inner_circle_l52_52106


namespace Zhenya_Venya_are_truth_tellers_l52_52045

-- Definitions
def is_truth_teller(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = true

def is_liar(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = false

noncomputable def BenyaStatement := "V is a liar"
noncomputable def ZhenyaStatement := "B is a liar"
noncomputable def SenyaStatement1 := "B and V are liars"
noncomputable def SenyaStatement2 := "Zh is a liar"

-- Conditions and proving the statement
theorem Zhenya_Venya_are_truth_tellers (truth_teller : String → Bool) :
  (∀ dwarf, truth_teller dwarf = true ∨ truth_teller dwarf = false) →
  (is_truth_teller "Benya" truth_teller → is_liar "Venya" truth_teller) →
  (is_truth_teller "Zhenya" truth_teller → is_liar "Benya" truth_teller) →
  (is_truth_teller "Senya" truth_teller → 
    is_liar "Benya" truth_teller ∧ is_liar "Venya" truth_teller ∧ is_liar "Zhenya" truth_teller) →
  is_truth_teller "Zhenya" truth_teller ∧ is_truth_teller "Venya" truth_teller :=
by
  sorry

end Zhenya_Venya_are_truth_tellers_l52_52045


namespace area_of_gray_region_l52_52248

theorem area_of_gray_region (r R : ℝ) (hr : r = 2) (hR : R = 3 * r) : 
  π * R ^ 2 - π * r ^ 2 = 32 * π :=
by
  have hr : r = 2 := hr
  have hR : R = 3 * r := hR
  sorry

end area_of_gray_region_l52_52248


namespace find_multiplier_l52_52330

theorem find_multiplier (n m : ℕ) (h1 : 2 * n = (26 - n) + 19) (h2 : n = 15) : m = 2 :=
by
  sorry

end find_multiplier_l52_52330


namespace solution_set_of_inequality_l52_52545

theorem solution_set_of_inequality:
  {x : ℝ | 1 < abs (2 * x - 1) ∧ abs (2 * x - 1) < 3} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ 
  {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l52_52545


namespace no_intersection_points_l52_52804

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := -x^2 + 6 * x - 8

-- The statement asserting that the parabolas do not intersect
theorem no_intersection_points :
  ∀ (x y : ℝ), parabola1 x = y → parabola2 x = y → false :=
by
  -- Introducing x and y as elements of the real numbers
  intros x y h1 h2
  
  -- Since this is only the statement, we use sorry to skip the actual proof
  sorry

end no_intersection_points_l52_52804


namespace distinct_shell_arrangements_l52_52888

/--
John draws a regular five pointed star and places one of ten different sea shells at each of the 5 outward-pointing points and 5 inward-pointing points. 
Considering rotations and reflections of an arrangement as equivalent, prove that the number of ways he can place the shells is 362880.
-/
theorem distinct_shell_arrangements : 
  let total_arrangements := Nat.factorial 10
  let symmetries := 10
  total_arrangements / symmetries = 362880 :=
by
  sorry

end distinct_shell_arrangements_l52_52888


namespace train_A_distance_travelled_l52_52634

/-- Let Train A and Train B start from opposite ends of a 200-mile route at the same time.
Train A has a constant speed of 20 miles per hour, and Train B has a constant speed of 200 miles / 6 hours (which is approximately 33.33 miles per hour).
Prove that Train A had traveled 75 miles when it met Train B. --/
theorem train_A_distance_travelled:
  ∀ (T : Type) (start_time : T) (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (meeting_time : ℝ),
  distance = 200 ∧ speed_A = 20 ∧ speed_B = 33.33 ∧ meeting_time = 200 / (speed_A + speed_B) → 
  (speed_A * meeting_time = 75) :=
by
  sorry

end train_A_distance_travelled_l52_52634


namespace find_base_b_l52_52767

theorem find_base_b :
  ∃ b : ℕ, (b > 7) ∧ (b > 10) ∧ (b > 8) ∧ (b > 12) ∧ 
    (4 + 3 = 7) ∧ ((2 + 7 + 1) % b = 3) ∧ ((3 + 4 + 1) % b = 5) ∧ 
    ((5 + 6 + 1) % b = 2) ∧ (1 + 1 = 2)
    ∧ b = 13 :=
by
  sorry

end find_base_b_l52_52767


namespace value_of_b_plus_a_l52_52280

theorem value_of_b_plus_a (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 2) (h3 : |a - b| = |b - a|) : b + a = -6 ∨ b + a = -10 :=
by
  sorry

end value_of_b_plus_a_l52_52280


namespace estate_value_l52_52941

theorem estate_value (E : ℝ) (x : ℝ) (hx : 5 * x = 0.6 * E) (charity_share : ℝ)
  (hcharity : charity_share = 800) (hwife : 3 * x * 4 = 12 * x)
  (htotal : E = 17 * x + charity_share) : E = 1923 :=
by
  sorry

end estate_value_l52_52941


namespace radian_measure_of_200_degrees_l52_52826

theorem radian_measure_of_200_degrees :
  (200 : ℝ) * (Real.pi / 180) = (10 / 9) * Real.pi :=
sorry

end radian_measure_of_200_degrees_l52_52826


namespace sum_of_digits_625_base5_l52_52769

def sum_of_digits_base_5 (n : ℕ) : ℕ :=
  let rec sum_digits n :=
    if n = 0 then 0
    else (n % 5) + sum_digits (n / 5)
  sum_digits n

theorem sum_of_digits_625_base5 : sum_of_digits_base_5 625 = 5 := by
  sorry

end sum_of_digits_625_base5_l52_52769


namespace decimal_to_base7_conversion_l52_52288

theorem decimal_to_base7_conversion :
  (2023 : ℕ) = 5 * (7^3) + 6 * (7^2) + 2 * (7^1) + 0 * (7^0) :=
by
  sorry

end decimal_to_base7_conversion_l52_52288


namespace car_speed_l52_52984

theorem car_speed (t_60 : ℝ := 60) (t_12 : ℝ := 12) (t_dist : ℝ := 1) :
  ∃ v : ℝ, v = 50 ∧ (t_60 / 60 + t_12 = 3600 / v) := 
by
  sorry

end car_speed_l52_52984


namespace Ethan_uses_8_ounces_each_l52_52326

def Ethan (b: ℕ): Prop :=
  let number_of_candles := 10 - 3
  let total_coconut_oil := number_of_candles * 1
  let total_beeswax := 63 - total_coconut_oil
  let beeswax_per_candle := total_beeswax / number_of_candles
  beeswax_per_candle = b

theorem Ethan_uses_8_ounces_each (b: ℕ) (hb: Ethan b): b = 8 :=
  sorry

end Ethan_uses_8_ounces_each_l52_52326


namespace problem_inequality_l52_52408

theorem problem_inequality (a b x y : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end problem_inequality_l52_52408


namespace malcolm_joshua_time_difference_l52_52904

-- Define the constants
def malcolm_speed : ℕ := 5 -- minutes per mile
def joshua_speed : ℕ := 8 -- minutes per mile
def race_distance : ℕ := 12 -- miles

-- Define the times it takes each runner to finish
def malcolm_time : ℕ := malcolm_speed * race_distance
def joshua_time : ℕ := joshua_speed * race_distance

-- Define the time difference and the proof statement
def time_difference : ℕ := joshua_time - malcolm_time

theorem malcolm_joshua_time_difference : time_difference = 36 := by
  sorry

end malcolm_joshua_time_difference_l52_52904


namespace team_sports_competed_l52_52973

theorem team_sports_competed (x : ℕ) (n : ℕ) 
  (h1 : (97 + n) / x = 90) 
  (h2 : (73 + n) / x = 87) : 
  x = 8 := 
by sorry

end team_sports_competed_l52_52973


namespace opposite_of_half_l52_52946

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l52_52946


namespace sqrt_xyz_sum_l52_52129

theorem sqrt_xyz_sum {x y z : ℝ} (h₁ : y + z = 24) (h₂ : z + x = 26) (h₃ : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end sqrt_xyz_sum_l52_52129


namespace total_legs_correct_l52_52352

def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goats : ℕ := 1
def legs_per_animal : ℕ := 4

theorem total_legs_correct :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = 72 :=
by
  sorry

end total_legs_correct_l52_52352


namespace correct_representations_l52_52447

open Set

theorem correct_representations : 
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  (¬S1 ∧ ¬S2 ∧ S3 ∧ S4) :=
by
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  exact sorry

end correct_representations_l52_52447


namespace ratio_n_over_p_l52_52215

theorem ratio_n_over_p (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0) 
  (h4 : ∃ r1 r2 : ℝ, r1 + r2 = -p ∧ r1 * r2 = m ∧ 3 * r1 + 3 * r2 = -m ∧ 9 * r1 * r2 = n) :
  n / p = -27 := 
by
  sorry

end ratio_n_over_p_l52_52215


namespace calculate_sum_l52_52094

theorem calculate_sum : (-2) + 1 = -1 :=
by 
  sorry

end calculate_sum_l52_52094


namespace price_increase_after_reduction_l52_52188

theorem price_increase_after_reduction (P : ℝ) (h : P > 0) : 
  let reduced_price := P * 0.85
  let increase_factor := 1 / 0.85
  let percentage_increase := (increase_factor - 1) * 100
  percentage_increase = 17.65 := by
  sorry

end price_increase_after_reduction_l52_52188


namespace unique_solution_of_quadratic_l52_52814

theorem unique_solution_of_quadratic :
  ∀ (b : ℝ), b ≠ 0 → (∃ x : ℝ, 3 * x^2 + b * x + 12 = 0 ∧ ∀ y : ℝ, 3 * y^2 + b * y + 12 = 0 → y = x) → 
  (b = 12 ∧ ∃ x : ℝ, x = -2 ∧ 3 * x^2 + 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 + 12 * y + 12 = 0 → y = x)) ∨ 
  (b = -12 ∧ ∃ x : ℝ, x = 2 ∧ 3 * x^2 - 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 - 12 * y + 12 = 0 → y = x)) :=
by 
  sorry

end unique_solution_of_quadratic_l52_52814


namespace max_value_at_x0_l52_52948

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_at_x0 {x0 : ℝ} (h : ∃ x0, ∀ x, f x ≤ f x0) : 
  f x0 = x0 :=
sorry

end max_value_at_x0_l52_52948


namespace m_1_sufficient_but_not_necessary_l52_52379

def lines_parallel (m : ℝ) : Prop :=
  let l1_slope := -m
  let l2_slope := (2 - 3 * m) / m
  l1_slope = l2_slope

theorem m_1_sufficient_but_not_necessary (m : ℝ) (h₁ : lines_parallel m) : 
  (m = 1) → (∃ m': ℝ, lines_parallel m' ∧ m' ≠ 1) :=
sorry

end m_1_sufficient_but_not_necessary_l52_52379


namespace find_general_formula_l52_52877

section sequence

variables {R : Type*} [LinearOrderedField R]
variable (c : R)
variable (h_c : c ≠ 0)

def seq (a : Nat → R) : Prop :=
  a 1 = 1 ∧ ∀ n : Nat, n > 0 → a (n + 1) = c * a n + c^(n + 1) * (2 * n + 1)

def general_formula (a : Nat → R) : Prop :=
  ∀ n : Nat, n > 0 → a n = (n^2 - 1) * c^n + c^(n - 1)

theorem find_general_formula :
  ∃ a : Nat → R, seq c a ∧ general_formula c a :=
by
  sorry

end sequence

end find_general_formula_l52_52877


namespace exists_special_integer_l52_52243

-- Define the mathematical conditions and the proof
theorem exists_special_integer (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) : 
  ∃ x : ℕ, 
    (∀ p ∈ P, ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) ∧
    (∀ p ∉ P, ¬∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) :=
sorry

end exists_special_integer_l52_52243


namespace fraction_representing_repeating_decimal_l52_52798

theorem fraction_representing_repeating_decimal (x a b : ℕ) (h : x = 35) (h1 : 100 * x - x = 35) 
(h2 : ∃ (a b : ℕ), x = a / b ∧ gcd a b = 1 ∧ a + b = 134) : a + b = 134 := 
sorry

end fraction_representing_repeating_decimal_l52_52798


namespace updated_mean_of_observations_l52_52057

theorem updated_mean_of_observations
    (number_of_observations : ℕ)
    (initial_mean : ℝ)
    (decrement_per_observation : ℝ)
    (h1 : number_of_observations = 50)
    (h2 : initial_mean = 200)
    (h3 : decrement_per_observation = 15) :
    (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 185 :=
by {
    sorry
}

end updated_mean_of_observations_l52_52057


namespace cow_cost_calculation_l52_52337

theorem cow_cost_calculation (C cow calf : ℝ) 
  (h1 : cow = 8 * calf) 
  (h2 : cow + calf = 990) : 
  cow = 880 :=
by
  sorry

end cow_cost_calculation_l52_52337


namespace large_buckets_needed_l52_52468

def capacity_large_bucket (S: ℚ) : ℚ := 2 * S + 3

theorem large_buckets_needed (n : ℕ) (L S : ℚ) (h1 : L = capacity_large_bucket S) (h2 : L = 4) (h3 : 2 * S + n * L = 63)
: n = 16 := sorry

end large_buckets_needed_l52_52468


namespace inequality_solution_l52_52699

theorem inequality_solution (x : ℝ) : 3 * x^2 - 8 * x + 3 < 0 ↔ (1 / 3 < x ∧ x < 3) := by
  sorry

end inequality_solution_l52_52699


namespace find_x_l52_52879

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end find_x_l52_52879


namespace relative_errors_are_equal_l52_52677

theorem relative_errors_are_equal :
  let e1 := 0.04
  let l1 := 20.0
  let e2 := 0.3
  let l2 := 150.0
  (e1 / l1) = (e2 / l2) :=
by
  sorry

end relative_errors_are_equal_l52_52677


namespace find_largest_integer_solution_l52_52824

theorem find_largest_integer_solution:
  ∃ x: ℤ, (1/4 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < (7/9 : ℝ) ∧ (x = 4) := by
  sorry

end find_largest_integer_solution_l52_52824


namespace regular_polygon_sides_l52_52221

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ n : ℕ, n = 12 := by
  sorry

end regular_polygon_sides_l52_52221


namespace unique_k_for_equal_power_l52_52062

theorem unique_k_for_equal_power (k : ℕ) (hk : 0 < k) (h : ∃ m n : ℕ, n > 1 ∧ (3 ^ k + 5 ^ k = m ^ n)) : k = 1 :=
by
  sorry

end unique_k_for_equal_power_l52_52062


namespace regina_total_cost_l52_52513

-- Definitions
def daily_cost : ℝ := 30
def mileage_cost : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 450
def fixed_fee : ℝ := 15

-- Proposition for total cost
noncomputable def total_cost : ℝ := daily_cost * days_rented + mileage_cost * miles_driven + fixed_fee

-- Theorem statement
theorem regina_total_cost : total_cost = 217.5 := by
  sorry

end regina_total_cost_l52_52513


namespace geometric_series_second_term_l52_52763

theorem geometric_series_second_term (a : ℝ) (r : ℝ) (sum : ℝ) 
  (h1 : r = 1/4) 
  (h2 : sum = 40) 
  (sum_formula : sum = a / (1 - r)) : a * r = 7.5 :=
by {
  -- Proof to be filled in later
  sorry
}

end geometric_series_second_term_l52_52763


namespace train_departure_at_10am_l52_52368

noncomputable def train_departure_time (distance travel_rate : ℕ) (arrival_time_chicago : ℕ) (time_difference : ℤ) : ℕ :=
  let travel_time := distance / travel_rate
  let arrival_time_ny := arrival_time_chicago + 1
  arrival_time_ny - travel_time

theorem train_departure_at_10am :
  train_departure_time 480 60 17 1 = 10 :=
by
  -- implementation of the proof will go here
  -- but we skip the proof as per the instructions
  sorry

end train_departure_at_10am_l52_52368


namespace quadratic_coeff_sum_l52_52472

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l52_52472


namespace chess_competition_l52_52777

theorem chess_competition (W M : ℕ) 
  (hW : W * (W - 1) / 2 = 45) 
  (hM : M * 10 = 200) :
  M * (M - 1) / 2 = 190 :=
by
  sorry

end chess_competition_l52_52777


namespace degree_of_monomial_neg2x2y_l52_52226

def monomial_degree (coeff : ℤ) (exp_x exp_y : ℕ) : ℕ :=
  exp_x + exp_y

theorem degree_of_monomial_neg2x2y :
  monomial_degree (-2) 2 1 = 3 :=
by
  -- Definition matching conditions given
  sorry

end degree_of_monomial_neg2x2y_l52_52226


namespace solve_trig_eq_l52_52079

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end solve_trig_eq_l52_52079


namespace find_x_l52_52583

theorem find_x 
  (AB AC BC : ℝ) 
  (x : ℝ)
  (hO : π * (AB / 2)^2 = 12 + 2 * x)
  (hP : π * (AC / 2)^2 = 24 + x)
  (hQ : π * (BC / 2)^2 = 108 - x)
  : AC^2 + BC^2 = AB^2 → x = 60 :=
by {
   sorry
}

end find_x_l52_52583


namespace number_of_men_in_first_group_l52_52659

/-- The number of men in the first group that can complete a piece of work in 5 days alongside 16 boys,
    given that 13 men and 24 boys can complete the same work in 4 days, and the ratio of daily work done 
    by a man to a boy is 2:1, is 12. -/
theorem number_of_men_in_first_group
  (x : ℕ)  -- define x as the amount of work a boy can do in a day
  (m : ℕ)  -- define m as the number of men in the first group
  (h1 : ∀ (x : ℕ), 5 * (m * 2 * x + 16 * x) = 4 * (13 * 2 * x + 24 * x))
  (h2 : 2 * x = x + x) : m = 12 :=
sorry

end number_of_men_in_first_group_l52_52659


namespace matrix_problem_l52_52385

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![6, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 8], ![3, -5]]
def RHS : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![15, -3]]

theorem matrix_problem : 
  2 • A + B = RHS :=
by
  sorry

end matrix_problem_l52_52385


namespace expression_evaluation_l52_52323

theorem expression_evaluation : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := 
by 
  sorry

end expression_evaluation_l52_52323


namespace liquid_level_ratio_l52_52149

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end liquid_level_ratio_l52_52149


namespace percent_both_correct_proof_l52_52225

-- Define the problem parameters
def totalTestTakers := 100
def percentFirstCorrect := 80
def percentSecondCorrect := 75
def percentNeitherCorrect := 5

-- Define the target proof statement
theorem percent_both_correct_proof :
  percentFirstCorrect + percentSecondCorrect - percentFirstCorrect + percentNeitherCorrect = 60 := 
by 
  sorry

end percent_both_correct_proof_l52_52225


namespace geom_seq_inc_condition_l52_52431

theorem geom_seq_inc_condition (a₁ a₂ q : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ = a₁ * q) :
  (a₁^2 < a₂^2) ↔ 
  (∀ n m : ℕ, n < m → (a₁ * q^n) < (a₁ * q^m) ∨ ((a₁ * q^n) = (a₁ * q^m) ∧ q = 1)) :=
by
  sorry

end geom_seq_inc_condition_l52_52431


namespace solution_exists_unique_l52_52095

variable (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)

theorem solution_exists_unique (x y z : ℝ)
  (hx : x = (b + c) / 2)
  (hy : y = (c + a) / 2)
  (hz : z = (a + b) / 2)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by
  sorry

end solution_exists_unique_l52_52095


namespace solve_fraction_eq_l52_52589

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4 / 3 :=
by 
  sorry

end solve_fraction_eq_l52_52589


namespace mixed_fraction_product_example_l52_52403

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l52_52403


namespace sampling_method_systematic_l52_52229

theorem sampling_method_systematic 
  (inspect_interval : ℕ := 10)
  (products_interval : ℕ := 10)
  (position : ℕ) :
  inspect_interval = 10 ∧ products_interval = 10 → 
  (sampling_method = "Systematic Sampling") :=
by
  sorry

end sampling_method_systematic_l52_52229


namespace value_of_expression_l52_52341

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : 3 * m^2 + 3 * m + 2006 = 2009 :=
by
  sorry

end value_of_expression_l52_52341


namespace hearts_per_card_l52_52120

-- Definitions of the given conditions
def num_suits := 4
def num_cards_total := 52
def num_cards_per_suit := num_cards_total / num_suits
def cost_per_cow := 200
def total_cost := 83200
def num_cows := total_cost / cost_per_cow

-- The mathematical proof problem translated to Lean 4:
theorem hearts_per_card :
    (2 * (num_cards_total / num_suits) = num_cows) → (num_cows = 416) → (num_cards_total / num_suits = 208) :=
by
  intros h1 h2
  sorry

end hearts_per_card_l52_52120


namespace find_a_l52_52707

theorem find_a : (a : ℕ) = 103 * 97 * 10009 → a = 99999919 := by
  intro h
  sorry

end find_a_l52_52707


namespace greatest_x_l52_52347

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end greatest_x_l52_52347


namespace problem_equivalence_l52_52717

theorem problem_equivalence :
  (∃ a a1 a2 a3 a4 a5 : ℝ, ((1 - x)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5)) → 
  ∀ (a a1 a2 a3 a4 a5 : ℝ), (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5 →
  (1 + 1)^5 = a - a1 + a2 - a3 + a4 - a5 →
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by
  intros h a a1 a2 a3 a4 a5 e1 e2
  sorry

end problem_equivalence_l52_52717


namespace problem_1_problem_2_problem_3_problem_4_l52_52508

-- Given conditions
variable {T : Type} -- Type representing teachers
variable {S : Type} -- Type representing students

def arrangements_ends (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_not_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_two_between (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

-- Statements to prove

-- 1. Prove that if teachers A and B must stand at the two ends, there are 48 different arrangements
theorem problem_1 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_ends teachers students = 48 :=
  sorry

-- 2. Prove that if teachers A and B must stand next to each other, there are 240 different arrangements
theorem problem_2 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_next_to_each_other teachers students = 240 :=
  sorry 

-- 3. Prove that if teachers A and B cannot stand next to each other, there are 480 different arrangements
theorem problem_3 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_not_next_to_each_other teachers students = 480 :=
  sorry 

-- 4. Prove that if there must be two students standing between teachers A and B, there are 144 different arrangements
theorem problem_4 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_two_between teachers students = 144 :=
  sorry

end problem_1_problem_2_problem_3_problem_4_l52_52508


namespace m_minus_n_eq_six_l52_52441

theorem m_minus_n_eq_six (m n : ℝ) (h : ∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) : m - n = 6 := by
  sorry

end m_minus_n_eq_six_l52_52441


namespace square_D_perimeter_l52_52584

theorem square_D_perimeter 
(C_perimeter: Real) 
(D_area_ratio : Real) 
(hC : C_perimeter = 32) 
(hD : D_area_ratio = 1/3) : 
    ∃ D_perimeter, D_perimeter = (32 * Real.sqrt 3) / 3 := 
by 
    sorry

end square_D_perimeter_l52_52584


namespace angelina_speed_l52_52861

theorem angelina_speed (v : ℝ) (h1 : 200 / v - 50 = 300 / (2 * v)) : 2 * v = 2 := 
by
  sorry

end angelina_speed_l52_52861


namespace same_color_probability_l52_52118

theorem same_color_probability 
  (B R : ℕ)
  (hB : B = 5)
  (hR : R = 5)
  : (B + R = 10) → (1/2 * 4/9 + 1/2 * 4/9 = 4/9) := by
  intros
  sorry

end same_color_probability_l52_52118


namespace pos_solution_sum_l52_52251

theorem pos_solution_sum (c d : ℕ) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (∃ x : ℝ, x ^ 2 + 16 * x = 100 ∧ x = Real.sqrt c - d) → c + d = 172 :=
by
  intro h
  sorry

end pos_solution_sum_l52_52251


namespace age_difference_is_12_l52_52643

noncomputable def age_difference (x : ℕ) : ℕ :=
  let older := 3 * x
  let younger := 2 * x
  older - younger

theorem age_difference_is_12 :
  ∃ x : ℕ, 3 * x + 2 * x = 60 ∧ age_difference x = 12 :=
by
  sorry

end age_difference_is_12_l52_52643


namespace distinct_arith_prog_triangles_l52_52801

theorem distinct_arith_prog_triangles (n : ℕ) (h10 : n % 10 = 0) : 
  (3 * n = 180 → ∃ d : ℕ, ∀ a b c, a = n - d ∧ b = n ∧ c = n + d 
  →  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 60) :=
by
  sorry

end distinct_arith_prog_triangles_l52_52801


namespace quadratic_solution_l52_52752

noncomputable def g (x : ℝ) : ℝ := x^2 + 2021 * x + 18

theorem quadratic_solution : ∀ x : ℝ, g (g x + x + 1) / g x = x^2 + 2023 * x + 2040 :=
by
  intros
  sorry

end quadratic_solution_l52_52752


namespace cyclist_return_trip_average_speed_l52_52232

theorem cyclist_return_trip_average_speed :
  let first_leg_distance := 12
  let second_leg_distance := 24
  let first_leg_speed := 8
  let second_leg_speed := 12
  let round_trip_time := 7.5
  let distance_to_destination := first_leg_distance + second_leg_distance
  let time_to_destination := (first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)
  let return_trip_time := round_trip_time - time_to_destination
  let return_trip_distance := distance_to_destination
  (return_trip_distance / return_trip_time) = 9 := 
by
  sorry

end cyclist_return_trip_average_speed_l52_52232


namespace bisection_method_root_interval_l52_52279

def f (x : ℝ) : ℝ := x^3 + x - 8

theorem bisection_method_root_interval :
  f 1 < 0 → f 1.5 < 0 → f 1.75 < 0 → f 2 > 0 → ∃ x, (1.75 < x ∧ x < 2 ∧ f x = 0) :=
by
  intros h1 h15 h175 h2
  sorry

end bisection_method_root_interval_l52_52279


namespace fractional_inequality_solution_set_l52_52612

theorem fractional_inequality_solution_set (x : ℝ) :
  (x / (x + 1) < 0) ↔ (-1 < x) ∧ (x < 0) :=
sorry

end fractional_inequality_solution_set_l52_52612


namespace john_money_left_l52_52104

def cost_of_drink (q : ℝ) : ℝ := q
def cost_of_small_pizza (q : ℝ) : ℝ := cost_of_drink q
def cost_of_large_pizza (q : ℝ) : ℝ := 4 * cost_of_drink q
def total_cost (q : ℝ) : ℝ := 2 * cost_of_drink q + 2 * cost_of_small_pizza q + cost_of_large_pizza q
def initial_money : ℝ := 50
def remaining_money (q : ℝ) : ℝ := initial_money - total_cost q

theorem john_money_left (q : ℝ) : remaining_money q = 50 - 8 * q :=
by
  sorry

end john_money_left_l52_52104


namespace bullet_speed_difference_l52_52359

theorem bullet_speed_difference
  (horse_speed : ℝ := 20) 
  (bullet_speed : ℝ := 400) : 
  ((bullet_speed + horse_speed) - (bullet_speed - horse_speed) = 40) := by
  sorry

end bullet_speed_difference_l52_52359


namespace meals_neither_vegan_kosher_nor_gluten_free_l52_52007

def total_clients : ℕ := 50
def n_vegan : ℕ := 10
def n_kosher : ℕ := 12
def n_gluten_free : ℕ := 6
def n_both_vegan_kosher : ℕ := 3
def n_both_vegan_gluten_free : ℕ := 4
def n_both_kosher_gluten_free : ℕ := 2
def n_all_three : ℕ := 1

/-- The number of clients who need a meal that is neither vegan, kosher, nor gluten-free. --/
theorem meals_neither_vegan_kosher_nor_gluten_free :
  total_clients - (n_vegan + n_kosher + n_gluten_free - n_both_vegan_kosher - n_both_vegan_gluten_free - n_both_kosher_gluten_free + n_all_three) = 30 :=
by
  sorry

end meals_neither_vegan_kosher_nor_gluten_free_l52_52007


namespace multiple_of_10_and_12_within_100_l52_52601

theorem multiple_of_10_and_12_within_100 :
  ∀ (n : ℕ), n ≤ 100 → (∃ k₁ k₂ : ℕ, n = 10 * k₁ ∧ n = 12 * k₂) ↔ n = 60 :=
by
  sorry

end multiple_of_10_and_12_within_100_l52_52601


namespace min_value_eq_9_l52_52724

-- Defining the conditions
variable (a b : ℝ)
variable (ha : a > 0) (hb : b > 0)
variable (h_eq : a - 2 * b = 0)

-- The goal is to prove the minimum value of (1/a) + (4/b) is 9
theorem min_value_eq_9 (ha : a > 0) (hb : b > 0) (h_eq : a - 2 * b = 0) 
  : ∃ (m : ℝ), m = 9 ∧ (∀ x, x = 1/a + 4/b → x ≥ m) :=
sorry

end min_value_eq_9_l52_52724


namespace f_is_periodic_l52_52238

noncomputable def f (x : ℝ) : ℝ := x - ⌊x⌋

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x := by
  intro x
  sorry

end f_is_periodic_l52_52238


namespace committee_count_is_correct_l52_52489

-- Definitions of the problem conditions
def total_people : ℕ := 10
def committee_size : ℕ := 5
def remaining_people := total_people - 1
def members_to_choose := committee_size - 1

-- The combinatorial function for selecting committee members
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def number_of_ways_to_form_committee : ℕ :=
  binomial remaining_people members_to_choose

-- Statement of the problem to prove the number of ways is 126
theorem committee_count_is_correct :
  number_of_ways_to_form_committee = 126 :=
by
  sorry

end committee_count_is_correct_l52_52489


namespace whole_milk_fat_percentage_l52_52654

def fat_in_some_milk : ℝ := 4
def percentage_less : ℝ := 0.5

theorem whole_milk_fat_percentage : ∃ (x : ℝ), fat_in_some_milk = percentage_less * x ∧ x = 8 :=
sorry

end whole_milk_fat_percentage_l52_52654


namespace third_root_of_polynomial_l52_52312

theorem third_root_of_polynomial (a b : ℚ) 
  (h₁ : a*(-1)^3 + (a + 3*b)*(-1)^2 + (2*b - 4*a)*(-1) + (10 - a) = 0)
  (h₂ : a*(4)^3 + (a + 3*b)*(4)^2 + (2*b - 4*a)*(4) + (10 - a) = 0) :
  ∃ (r : ℚ), r = -24 / 19 :=
by
  sorry

end third_root_of_polynomial_l52_52312


namespace nonoverlapping_unit_squares_in_figure_100_l52_52364

theorem nonoverlapping_unit_squares_in_figure_100 :
  ∃ f : ℕ → ℕ, (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 15 ∧ f 3 = 27) ∧ f 100 = 20203 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_100_l52_52364


namespace cylinder_radius_in_cone_l52_52459

-- Define the conditions
def cone_diameter := 18
def cone_height := 20
def cylinder_height_eq_diameter {r : ℝ} := 2 * r

-- Define the theorem to prove
theorem cylinder_radius_in_cone : ∃ r : ℝ, r = 90 / 19 ∧ (20 - 2 * r) / r = 20 / 9 :=
by
  sorry

end cylinder_radius_in_cone_l52_52459


namespace positive_difference_prime_factors_159137_l52_52365

-- Lean 4 Statement Following the Instructions
theorem positive_difference_prime_factors_159137 :
  (159137 = 11 * 17 * 23 * 37) → (37 - 23 = 14) :=
by
  intro h
  sorry -- Proof will be written here

end positive_difference_prime_factors_159137_l52_52365


namespace green_beans_count_l52_52675

def total_beans := 572
def red_beans := (1 / 4) * total_beans
def remaining_after_red := total_beans - red_beans
def white_beans := (1 / 3) * remaining_after_red
def remaining_after_white := remaining_after_red - white_beans
def green_beans := (1 / 2) * remaining_after_white

theorem green_beans_count : green_beans = 143 := by
  sorry

end green_beans_count_l52_52675


namespace find_stickers_before_birthday_l52_52246

variable (stickers_received : ℕ) (total_stickers : ℕ)

def stickers_before_birthday (stickers_received total_stickers : ℕ) : ℕ :=
  total_stickers - stickers_received

theorem find_stickers_before_birthday (h1 : stickers_received = 22) (h2 : total_stickers = 61) : 
  stickers_before_birthday stickers_received total_stickers = 39 :=
by 
  have h1 : stickers_received = 22 := h1
  have h2 : total_stickers = 61 := h2
  rw [h1, h2]
  rfl

end find_stickers_before_birthday_l52_52246


namespace bicycle_stock_decrease_l52_52977

-- Define the conditions and the problem
theorem bicycle_stock_decrease (m : ℕ) (jan_to_oct_decrease june_to_oct_decrease monthly_decrease : ℕ) 
  (h1: monthly_decrease = 4)
  (h2: jan_to_oct_decrease = 36)
  (h3: june_to_oct_decrease = 4 * monthly_decrease):
  m * monthly_decrease = jan_to_oct_decrease - june_to_oct_decrease → m = 5 := 
by
  sorry

end bicycle_stock_decrease_l52_52977


namespace quadratic_value_at_point_a_l52_52986

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

open Real

theorem quadratic_value_at_point_a
  (a b c : ℝ)
  (axis : ℝ)
  (sym : ∀ x, quadratic a b c (2 * axis - x) = quadratic a b c x)
  (at_zero : quadratic a b c 0 = -3) :
  quadratic a b c 20 = -3 := by
  -- proof steps would go here
  sorry

end quadratic_value_at_point_a_l52_52986


namespace other_root_of_quadratic_eq_l52_52030

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l52_52030


namespace solve_equation_l52_52384

theorem solve_equation : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end solve_equation_l52_52384


namespace find_z_solutions_l52_52552

open Real

noncomputable def is_solution (z : ℝ) : Prop :=
  sin z + sin (2 * z) + sin (3 * z) = cos z + cos (2 * z) + cos (3 * z)

theorem find_z_solutions (z : ℝ) : 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k - 1)) ∨ 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k + 1)) ∨ 
  (∃ k : ℤ, z = π / 8 * (4 * k + 1)) ↔
  is_solution z :=
by
  sorry

end find_z_solutions_l52_52552


namespace sum_of_specific_terms_l52_52453

theorem sum_of_specific_terms 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h1 : S 3 = 9) 
  (h2 : S 6 = 36) 
  (h3 : ∀ n, S n = n * (a 1) + d * n * (n - 1) / 2) :
  a 7 + a 8 + a 9 = 45 := 
sorry

end sum_of_specific_terms_l52_52453


namespace find_num_officers_l52_52038

noncomputable def num_officers (O : ℕ) : Prop :=
  let avg_salary_all := 120
  let avg_salary_officers := 440
  let avg_salary_non_officers := 110
  let num_non_officers := 480
  let total_salary :=
    avg_salary_all * (O + num_non_officers)
  let salary_officers :=
    avg_salary_officers * O
  let salary_non_officers :=
    avg_salary_non_officers * num_non_officers
  total_salary = salary_officers + salary_non_officers

theorem find_num_officers : num_officers 15 :=
sorry

end find_num_officers_l52_52038


namespace rectangle_area_unchanged_l52_52729

theorem rectangle_area_unchanged
  (x y : ℝ)
  (h1 : x * y = (x + 3) * (y - 1))
  (h2 : x * y = (x - 3) * (y + 1.5)) :
  x * y = 31.5 :=
sorry

end rectangle_area_unchanged_l52_52729


namespace range_of_a_l52_52989

theorem range_of_a (a m : ℝ) (hp : 3 * a < m ∧ m < 4 * a) 
  (hq : 1 < m ∧ m < 3 / 2) :
  1 / 3 ≤ a ∧ a ≤ 3 / 8 :=
by
  sorry

end range_of_a_l52_52989


namespace people_left_line_l52_52310

theorem people_left_line (initial new final L : ℕ) 
  (h1 : initial = 30) 
  (h2 : new = 5) 
  (h3 : final = 25) 
  (h4 : initial - L + new = final) : L = 10 := by
  sorry

end people_left_line_l52_52310


namespace increasing_interval_m_range_l52_52185

def y (x m : ℝ) : ℝ := x^2 + 2 * m * x + 10

theorem increasing_interval_m_range (m : ℝ) : (∀ x, 2 ≤ x → ∀ x', x' ≥ x → y x m ≤ y x' m) → (-2 : ℝ) ≤ m :=
sorry

end increasing_interval_m_range_l52_52185


namespace number_of_intersection_points_l52_52664

theorem number_of_intersection_points : 
  ∃! (P : ℝ × ℝ), 
    (P.1 ^ 2 + P.2 ^ 2 = 16) ∧ (P.1 = 4) := 
by
  sorry

end number_of_intersection_points_l52_52664


namespace middle_number_l52_52511

theorem middle_number (x y z : ℤ) 
  (h1 : x + y = 21)
  (h2 : x + z = 25)
  (h3 : y + z = 28)
  (h4 : x < y)
  (h5 : y < z) : 
  y = 12 :=
sorry

end middle_number_l52_52511


namespace travel_agency_choice_l52_52202

noncomputable def cost_A (x : ℕ) : ℝ :=
  350 * x + 1000

noncomputable def cost_B (x : ℕ) : ℝ :=
  400 * x + 800

theorem travel_agency_choice (x : ℕ) :
  if x < 4 then cost_A x > cost_B x
  else if x = 4 then cost_A x = cost_B x
  else cost_A x < cost_B x :=
by sorry

end travel_agency_choice_l52_52202


namespace sum_of_geometric_sequence_l52_52262

noncomputable def geometric_sequence_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence
  (a_1 q : ℝ) 
  (h1 : a_1^2 * q^6 = 2 * a_1 * q^2)
  (h2 : (a_1 * q^3 + 2 * a_1 * q^6) / 2 = 5 / 4)
  : geometric_sequence_sum a_1 q 4 = 30 :=
by
  sorry

end sum_of_geometric_sequence_l52_52262


namespace converse_proposition_l52_52855

-- Define the condition: The equation x^2 + x - m = 0 has real roots
def has_real_roots (a b c : ℝ) : Prop :=
  let Δ := b * b - 4 * a * c
  Δ ≥ 0

theorem converse_proposition (m : ℝ) :
  has_real_roots 1 1 (-m) → m > 0 :=
by
  sorry

end converse_proposition_l52_52855


namespace remaining_lemon_heads_after_eating_l52_52979

-- Assume initial number of lemon heads is given
variables (initial_lemon_heads : ℕ)

-- Patricia eats 15 lemon heads
def remaining_lemon_heads (initial_lemon_heads : ℕ) : ℕ :=
  initial_lemon_heads - 15

theorem remaining_lemon_heads_after_eating :
  ∀ (initial_lemon_heads : ℕ), remaining_lemon_heads initial_lemon_heads = initial_lemon_heads - 15 :=
by
  intros
  rfl

end remaining_lemon_heads_after_eating_l52_52979


namespace tiffany_total_bags_l52_52227

theorem tiffany_total_bags (monday_bags next_day_bags : ℕ) (h1 : monday_bags = 4) (h2 : next_day_bags = 8) :
  monday_bags + next_day_bags = 12 :=
by
  sorry

end tiffany_total_bags_l52_52227


namespace smallest_number_of_three_l52_52446

theorem smallest_number_of_three (x : ℕ) (h1 : x = 18)
  (h2 : ∀ y z : ℕ, y = 4 * x ∧ z = 2 * y)
  (h3 : (x + 4 * x + 8 * x) / 3 = 78)
  : x = 18 := by
  sorry

end smallest_number_of_three_l52_52446


namespace power_function_m_l52_52524

theorem power_function_m (m : ℝ) 
  (h_even : ∀ x : ℝ, x^m = (-x)^m) 
  (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → x^m > y^m) : m = -2 :=
sorry

end power_function_m_l52_52524


namespace sum_of_arithmetic_sequence_l52_52551

theorem sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) 
  (h1 : ∀ n, S n = n * a₁ + (n - 1) * n / 2 * d)
  (h2 : S 1 / S 4 = 1 / 10) :
  S 3 / S 5 = 2 / 5 := 
sorry

end sum_of_arithmetic_sequence_l52_52551


namespace range_of_a_l52_52043

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := 
sorry

end range_of_a_l52_52043


namespace range_of_a_l52_52290

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x
noncomputable def g (x a : ℝ) : ℝ := x + 1 / (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, x1 ∈ Set.Icc 0 2 → ∃ x2 : ℝ, x2 ∈ Set.Ioi a ∧ f x1 ≥ g x2 a) →
  a ≤ -1 :=
by
  intro h
  sorry

end range_of_a_l52_52290


namespace arithmetic_sequence_a_eq_zero_l52_52260

theorem arithmetic_sequence_a_eq_zero (a : ℝ) :
  (∀ n : ℕ, n > 0 → ∃ S : ℕ → ℝ, S n = (n^2 : ℝ) + 2 * n + a) →
  a = 0 :=
by
  sorry

end arithmetic_sequence_a_eq_zero_l52_52260


namespace part1_part2_l52_52722

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 2 * a + 3)

theorem part1 (x : ℝ) : f x 2 ≤ 9 ↔ -2 ≤ x ∧ x ≤ 4 :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : a ∈ Set.Iic (-2 / 3) ∪ Set.Ici (14 / 3) :=
by sorry

end part1_part2_l52_52722


namespace smallest_omega_l52_52737

theorem smallest_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∃ k : ℤ, (2 / 3) * ω = 2 * k) -> ω = 3 :=
by
  sorry

end smallest_omega_l52_52737


namespace mia_min_stamps_l52_52676

theorem mia_min_stamps (x y : ℕ) (hx : 5 * x + 7 * y = 37) : x + y = 7 :=
sorry

end mia_min_stamps_l52_52676


namespace inequality_solution_set_l52_52728

variable {f : ℝ → ℝ}

-- Conditions
def neg_domain : Set ℝ := {x | x < 0}
def pos_domain : Set ℝ := {x | x > 0}
def f_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f_property_P (f : ℝ → ℝ) := ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)

-- Translate question and correct answer into a proposition in Lean
theorem inequality_solution_set (h1 : ∀ x, f (-x) = -f x)
                                (h2 : ∀ x1 x2, (0 < x1) → (0 < x2) → (x1 ≠ x1) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)) :
  {x | f (x - 2) < f (x^2 - 4) / (x + 2)} = {x | x < -3} ∪ {x | -1 < x ∧ x < 2} := 
sorry

end inequality_solution_set_l52_52728


namespace strawberries_in_each_handful_l52_52454

theorem strawberries_in_each_handful (x : ℕ) (h : (x - 1) * (75 / x) = 60) : x = 5 :=
sorry

end strawberries_in_each_handful_l52_52454


namespace compare_game_A_and_C_l52_52670

-- Probability definitions for coin toss
def p_heads := 2/3
def p_tails := 1/3

-- Probability of winning Game A
def prob_win_A := (p_heads^3) + (p_tails^3)

-- Probability of winning Game C
def prob_win_C := (p_heads^3 + p_tails^3)^2

-- Theorem statement to compare chances of winning Game A to Game C
theorem compare_game_A_and_C : prob_win_A - prob_win_C = 2/9 := by sorry

end compare_game_A_and_C_l52_52670


namespace cos_diff_half_l52_52410

theorem cos_diff_half (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1 / 2)
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) :
  Real.cos (α - β) = -1 / 2 :=
by
  sorry

end cos_diff_half_l52_52410


namespace jenny_run_distance_l52_52163

theorem jenny_run_distance (walk_distance : ℝ) (ran_walk_diff : ℝ) (h_walk : walk_distance = 0.4) (h_diff : ran_walk_diff = 0.2) :
  (walk_distance + ran_walk_diff) = 0.6 :=
sorry

end jenny_run_distance_l52_52163


namespace unique_solution_cond_l52_52761

open Real

theorem unique_solution_cond (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 :=
by sorry

end unique_solution_cond_l52_52761


namespace range_of_a_l52_52479

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l52_52479


namespace Montoya_budget_spent_on_food_l52_52448

-- Define the fractions spent on groceries and going out to eat
def groceries_fraction : ℝ := 0.6
def eating_out_fraction : ℝ := 0.2

-- Define the total fraction spent on food
def total_food_fraction (g : ℝ) (e : ℝ) : ℝ := g + e

-- The theorem to prove
theorem Montoya_budget_spent_on_food : total_food_fraction groceries_fraction eating_out_fraction = 0.8 := 
by
  -- the proof will go here
  sorry

end Montoya_budget_spent_on_food_l52_52448


namespace drainage_capacity_per_day_l52_52011

theorem drainage_capacity_per_day
  (capacity : ℝ)
  (rain_1 : ℝ)
  (rain_2 : ℝ)
  (rain_3 : ℝ)
  (rain_4_min : ℝ)
  (total_days : ℕ) 
  (days_to_drain : ℕ)
  (feet_to_inches : ℝ := 12)
  (required_rain_capacity : ℝ) 
  (drain_capacity_per_day : ℝ)

  (h1: capacity = 6 * feet_to_inches)
  (h2: rain_1 = 10)
  (h3: rain_2 = 2 * rain_1)
  (h4: rain_3 = 1.5 * rain_2)
  (h5: rain_4_min = 21)
  (h6: total_days = 4)
  (h7: days_to_drain = 3)
  (h8: required_rain_capacity = capacity - (rain_1 + rain_2 + rain_3))

  : drain_capacity_per_day = (rain_1 + rain_2 + rain_3 - required_rain_capacity + rain_4_min) / days_to_drain :=
sorry

end drainage_capacity_per_day_l52_52011


namespace unique_solution_condition_l52_52125

theorem unique_solution_condition (a b : ℝ) : (4 * x - 6 + a = (b + 1) * x + 2) → b ≠ 3 :=
by
  intro h
  -- Given the condition equation
  have eq1 : 4 * x - 6 + a = (b + 1) * x + 2 := h
  -- Simplify to the form (3 - b) * x = 8 - a
  sorry

end unique_solution_condition_l52_52125


namespace additional_hours_to_travel_l52_52525

theorem additional_hours_to_travel (distance1 time1 rate distance2 : ℝ)
  (H1 : distance1 = 360)
  (H2 : time1 = 3)
  (H3 : rate = distance1 / time1)
  (H4 : distance2 = 240)
  :
  distance2 / rate = 2 := 
sorry

end additional_hours_to_travel_l52_52525


namespace prob_red_or_blue_l52_52355

open Nat

noncomputable def total_marbles : Nat := 90
noncomputable def prob_white : (ℚ) := 1 / 6
noncomputable def prob_green : (ℚ) := 1 / 5

theorem prob_red_or_blue :
  let prob_total := 1
  let prob_white_or_green := prob_white + prob_green
  let prob_red_blue := prob_total - prob_white_or_green
  prob_red_blue = 19 / 30 := by
    sorry

end prob_red_or_blue_l52_52355


namespace slope_of_decreasing_linear_function_l52_52263

theorem slope_of_decreasing_linear_function (m b : ℝ) :
  (∀ x y : ℝ, x < y → mx + b > my + b) → m < 0 :=
by
  intro h
  sorry

end slope_of_decreasing_linear_function_l52_52263


namespace sin_2alpha_plus_sin_squared_l52_52298

theorem sin_2alpha_plus_sin_squared (α : ℝ) (h : Real.tan α = 1 / 2) : Real.sin (2 * α) + Real.sin α ^ 2 = 1 :=
sorry

end sin_2alpha_plus_sin_squared_l52_52298


namespace proof_expression_equals_60_times_10_power_1501_l52_52657

noncomputable def expression_equals_60_times_10_power_1501 : Prop :=
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501

theorem proof_expression_equals_60_times_10_power_1501 :
  expression_equals_60_times_10_power_1501 :=
by 
  sorry

end proof_expression_equals_60_times_10_power_1501_l52_52657


namespace division_correct_l52_52844

-- Definitions based on conditions
def expr1 : ℕ := 12 + 15 * 3
def expr2 : ℚ := 180 / expr1

-- Theorem statement using the question and correct answer
theorem division_correct : expr2 = 180 / 57 := by
  sorry

end division_correct_l52_52844


namespace solve_fractional_equation_l52_52085

theorem solve_fractional_equation (x : ℚ) (h: x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 2) ↔ (x = 7 / 6) := 
by
  sorry

end solve_fractional_equation_l52_52085


namespace CALI_area_is_180_l52_52493

-- all the conditions used in Lean definitions
def is_square (s : ℕ) : Prop := (s > 0)

def are_midpoints (T O W N B E R K : ℕ) : Prop := 
  (T = (B + E) / 2) ∧ (O = (E + R) / 2) ∧ (W = (R + K) / 2) ∧ (N = (K + B) / 2)

def is_parallel (CA BO : ℕ) : Prop :=
  CA = BO 

-- the condition indicates the length of each side of the square BERK is 10
def side_length_of_BERK : ℕ := 10

-- definition of lengths and condition
def BERK_lengths (BERK_side_length : ℕ) (BERK_diag_length : ℕ): Prop :=
  BERK_side_length = side_length_of_BERK ∧ BERK_diag_length = BERK_side_length * (2^(1/2))

def CALI_area_of_length (length: ℕ): ℕ := length^2

theorem CALI_area_is_180 
(BERK_side_length BERK_diag_length : ℕ)
(CALI_length : ℕ)
(T O W N B E R K CA BO : ℕ)
(h1 : is_square BERK_side_length)
(h2 : are_midpoints T O W N B E R K)
(h3 : is_parallel CA BO)
(h4 : BERK_lengths BERK_side_length BERK_diag_length)
(h5 : CA = CA)
: CALI_area_of_length 15 = 180 :=
sorry

end CALI_area_is_180_l52_52493


namespace gamma_start_time_correct_l52_52823

noncomputable def trisection_points (AB : ℕ) : Prop := AB ≥ 3

structure Walkers :=
  (d : ℕ) -- Total distance AB
  (Vα : ℕ) -- Speed of person α
  (Vβ : ℕ) -- Speed of person β
  (Vγ : ℕ) -- Speed of person γ

def meeting_times (w : Walkers) := 
  w.Vα = w.d / 72 ∧ 
  w.Vβ = w.d / 36 ∧ 
  w.Vγ = w.Vβ

def start_times_correct (startA timeA_meetC : ℕ) (startB timeB_reachesA: ℕ) (startC_latest: ℕ): Prop :=
  startA = 0 ∧ 
  startB = 12 ∧
  timeA_meetC = 24 ∧ 
  timeB_reachesA = 30 ∧
  startC_latest = 16

theorem gamma_start_time_correct (AB : ℕ) (w : Walkers) (t : Walkers → Prop) : 
  trisection_points AB → 
  meeting_times w →
  start_times_correct 0 24 12 30 16 → 
  ∃ tγ_start, tγ_start = 16 :=
sorry

end gamma_start_time_correct_l52_52823


namespace smallest_divisor_of_7614_l52_52340

theorem smallest_divisor_of_7614 (h : Nat) (H_h_eq : h = 1) (n : Nat) (H_n_eq : n = (7600 + 10 * h + 4)) :
  ∃ d, d > 1 ∧ d ∣ n ∧ ∀ x, x > 1 ∧ x ∣ n → d ≤ x :=
by
  sorry

end smallest_divisor_of_7614_l52_52340


namespace eight_digit_increasing_numbers_mod_1000_l52_52044

theorem eight_digit_increasing_numbers_mod_1000 : 
  ((Nat.choose 17 8) % 1000) = 310 := 
by 
  sorry -- Proof not required as per instructions

end eight_digit_increasing_numbers_mod_1000_l52_52044


namespace dice_surface_sum_l52_52644

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end dice_surface_sum_l52_52644


namespace inequality_proof_l52_52362

-- Define the main theorem with the conditions
theorem inequality_proof 
  (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a = b ∧ b = c ∧ c = d) ↔ (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a)) := 
sorry

end inequality_proof_l52_52362


namespace alicia_read_more_books_than_ian_l52_52241

def books_read : List Nat := [3, 5, 8, 6, 7, 4, 2, 1]

def alicia_books (books : List Nat) : Nat :=
  books.maximum?.getD 0

def ian_books (books : List Nat) : Nat :=
  books.minimum?.getD 0

theorem alicia_read_more_books_than_ian :
  alicia_books books_read - ian_books books_read = 7 :=
by
  -- By reviewing the given list of books read [3, 5, 8, 6, 7, 4, 2, 1]
  -- We find that alicia_books books_read = 8 and ian_books books_read = 1
  -- Thus, 8 - 1 = 7
  sorry

end alicia_read_more_books_than_ian_l52_52241


namespace percentage_increase_of_base_l52_52930

theorem percentage_increase_of_base
  (h b : ℝ) -- Original height and base
  (h_new : ℝ) -- New height
  (b_new : ℝ) -- New base
  (A_original A_new : ℝ) -- Original and new areas
  (p : ℝ) -- Percentage increase in the base
  (h_new_def : h_new = 0.60 * h)
  (b_new_def : b_new = b * (1 + p / 100))
  (A_original_def : A_original = 0.5 * b * h)
  (A_new_def : A_new = 0.5 * b_new * h_new)
  (area_decrease : A_new = 0.84 * A_original) :
  p = 40 := by
  sorry

end percentage_increase_of_base_l52_52930


namespace system_has_real_solution_l52_52810

theorem system_has_real_solution (k : ℝ) : 
  (∃ x y : ℝ, y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by
  sorry

end system_has_real_solution_l52_52810


namespace area_of_shaded_trapezoid_l52_52501

-- Definitions of conditions:
def side_lengths : List ℕ := [1, 3, 5, 7]
def total_base : ℕ := side_lengths.sum
def height_largest_square : ℕ := 7
def ratio : ℚ := height_largest_square / total_base

def height_at_end (n : ℕ) : ℚ := ratio * n
def lower_base_height : ℚ := height_at_end 4
def upper_base_height : ℚ := height_at_end 9
def trapezoid_height : ℕ := 2

-- Main theorem:
theorem area_of_shaded_trapezoid :
  (1 / 2) * (lower_base_height + upper_base_height) * trapezoid_height = 91 / 8 :=
by
  sorry

end area_of_shaded_trapezoid_l52_52501


namespace triangle_area_l52_52272

def vec2 := ℝ × ℝ

def area_of_triangle (a b : vec2) : ℝ :=
  0.5 * |a.1 * b.2 - a.2 * b.1|

def a : vec2 := (2, -3)
def b : vec2 := (4, -1)

theorem triangle_area : area_of_triangle a b = 5 := by
  sorry

end triangle_area_l52_52272


namespace solve_for_x_l52_52971

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = x / 0.0144) : x = 14.4 :=
by
  sorry

end solve_for_x_l52_52971


namespace oranges_thrown_away_l52_52656

theorem oranges_thrown_away (initial_oranges old_oranges_thrown new_oranges final_oranges : ℕ) 
    (h1 : initial_oranges = 34)
    (h2 : new_oranges = 13)
    (h3 : final_oranges = 27)
    (h4 : initial_oranges - old_oranges_thrown + new_oranges = final_oranges) :
    old_oranges_thrown = 20 :=
by
  sorry

end oranges_thrown_away_l52_52656


namespace roots_abs_lt_one_l52_52885

theorem roots_abs_lt_one
  (a b : ℝ)
  (h1 : |a| + |b| < 1)
  (h2 : a^2 - 4 * b ≥ 0) :
  ∀ (x : ℝ), x^2 + a * x + b = 0 → |x| < 1 :=
sorry

end roots_abs_lt_one_l52_52885


namespace necessary_but_not_sufficient_for_inequality_l52_52291

theorem necessary_but_not_sufficient_for_inequality : 
  ∀ x : ℝ, (-2 < x ∧ x < 4) → (x < 5) ∧ (¬(x < 5) → (-2 < x ∧ x < 4) ) :=
by 
  sorry

end necessary_but_not_sufficient_for_inequality_l52_52291


namespace albrecht_correct_substitution_l52_52962

theorem albrecht_correct_substitution (a b : ℕ) (h : (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9) :
  (a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2) :=
by
  -- The proof will be filled in here
  sorry

end albrecht_correct_substitution_l52_52962


namespace maximum_value_of_function_l52_52613

theorem maximum_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ M, (∀ y, y = x * (1 - 2 * x) → y ≤ M) ∧ M = 1/8 :=
sorry

end maximum_value_of_function_l52_52613


namespace trapezoidal_field_perimeter_l52_52533

-- Definitions derived from the conditions
def length_of_longer_parallel_side : ℕ := 15
def length_of_shorter_parallel_side : ℕ := 9
def total_perimeter_of_rectangle : ℕ := 52

-- Correct Answer
def correct_perimeter_of_trapezoidal_field : ℕ := 46

-- Theorem statement
theorem trapezoidal_field_perimeter 
  (a b w : ℕ)
  (h1 : a = length_of_longer_parallel_side)
  (h2 : b = length_of_shorter_parallel_side)
  (h3 : 2 * (a + w) = total_perimeter_of_rectangle)
  (h4 : w = 11) -- from the solution calculation
  : a + b + 2 * w = correct_perimeter_of_trapezoidal_field :=
by
  sorry

end trapezoidal_field_perimeter_l52_52533


namespace math_problem_example_l52_52381

theorem math_problem_example (m n : ℤ) (h0 : m > 0) (h1 : n > 0)
    (h2 : 3 * m + 2 * n = 225) (h3 : Int.gcd m n = 15) : m + n = 105 :=
sorry

end math_problem_example_l52_52381


namespace least_positive_integer_x_l52_52638

theorem least_positive_integer_x (x : ℕ) (h : x + 5683 ≡ 420 [MOD 17]) : x = 7 :=
sorry

end least_positive_integer_x_l52_52638


namespace initial_winning_percentage_calc_l52_52673

variable (W : ℝ)
variable (initial_matches : ℝ := 120)
variable (additional_wins : ℝ := 70)
variable (final_matches : ℝ := 190)
variable (final_average : ℝ := 0.52)
variable (initial_wins : ℝ := 29)

noncomputable def winning_percentage_initial :=
  (initial_wins / initial_matches) * 100

theorem initial_winning_percentage_calc :
  (W = initial_wins) →
  ((W + additional_wins) / final_matches = final_average) →
  winning_percentage_initial = 24.17 :=
by
  intros
  sorry

end initial_winning_percentage_calc_l52_52673


namespace sin_15_add_sin_75_l52_52424

theorem sin_15_add_sin_75 : 
  Real.sin (15 * Real.pi / 180) + Real.sin (75 * Real.pi / 180) = Real.sqrt 6 / 2 :=
by
  sorry

end sin_15_add_sin_75_l52_52424


namespace tangent_line_through_origin_to_circle_in_third_quadrant_l52_52159

theorem tangent_line_through_origin_to_circle_in_third_quadrant :
  ∃ m : ℝ, (∀ x y : ℝ, y = m * x) ∧ (∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0) ∧ (x < 0 ∧ y < 0) ∧ y = -3 * x :=
sorry

end tangent_line_through_origin_to_circle_in_third_quadrant_l52_52159


namespace solution_set_a_eq_half_l52_52616

theorem solution_set_a_eq_half (a : ℝ) : (∀ x : ℝ, (ax / (x - 1) < 1 ↔ (x < 1 ∨ x > 2))) → a = 1 / 2 :=
by
sorry

end solution_set_a_eq_half_l52_52616


namespace max_ab_value_l52_52857

theorem max_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_perpendicular : (2 * a - 1) * b = -1) : ab <= 1 / 8 := by
  sorry

end max_ab_value_l52_52857


namespace f_neg_1_l52_52743

-- Define the functions
variable (f : ℝ → ℝ) -- f is a real-valued function
variable (g : ℝ → ℝ) -- g is a real-valued function

-- Given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_def : ∀ x, g x = f x + 4
axiom g_at_1 : g 1 = 2

-- Define the theorem to prove
theorem f_neg_1 : f (-1) = 2 :=
by
  -- Proof goes here
  sorry

end f_neg_1_l52_52743


namespace evaluate_expression_l52_52234

theorem evaluate_expression :
  (1 / (-5^3)^4) * (-5)^15 * 5^2 = -3125 :=
by
  sorry

end evaluate_expression_l52_52234


namespace find_fourth_student_in_sample_l52_52307

theorem find_fourth_student_in_sample :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 48 ∧ 
           (∀ (k : ℕ), k = 29 → 1 ≤ k ∧ k ≤ 48 ∧ ((k = 5 + 2 * 12) ∨ (k = 41 - 12)) ∧ n = 17) :=
sorry

end find_fourth_student_in_sample_l52_52307


namespace doug_marbles_l52_52273

theorem doug_marbles (e_0 d_0 : ℕ) (h1 : e_0 = d_0 + 12) (h2 : e_0 - 20 = 17) : d_0 = 25 :=
by
  sorry

end doug_marbles_l52_52273


namespace taxi_fare_distance_l52_52173

theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (initial_distance : ℝ) (total_fare : ℝ) : 
  initial_fare = 2.0 →
  subsequent_fare = 0.60 →
  initial_distance = 1 / 5 →
  total_fare = 25.4 →
  ∃ d : ℝ, d = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end taxi_fare_distance_l52_52173


namespace mike_total_spending_is_correct_l52_52708

-- Definitions for the costs of the items
def cost_marbles : ℝ := 9.05
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52
def cost_toy_car : ℝ := 3.75
def cost_puzzle : ℝ := 8.99
def cost_stickers : ℝ := 1.25

-- Definitions for the discounts
def discount_puzzle : ℝ := 0.15
def discount_toy_car : ℝ := 0.10

-- Definition for the coupon
def coupon_amount : ℝ := 5.00

-- Total spent by Mike on toys
def total_spent : ℝ :=
  cost_marbles + 
  cost_football + 
  cost_baseball + 
  (cost_toy_car - cost_toy_car * discount_toy_car) + 
  (cost_puzzle - cost_puzzle * discount_puzzle) + 
  cost_stickers - 
  coupon_amount

-- Proof statement
theorem mike_total_spending_is_correct : 
  total_spent = 27.7865 :=
by
  sorry

end mike_total_spending_is_correct_l52_52708


namespace original_number_of_boys_l52_52915

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 135 = (n + 3) * 36) : 
  n = 27 := 
by 
  sorry

end original_number_of_boys_l52_52915


namespace evaluate_expression_l52_52193

theorem evaluate_expression : (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end evaluate_expression_l52_52193


namespace g_five_eq_one_l52_52419

noncomputable def g : ℝ → ℝ := sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_nonzero : ∀ x : ℝ, g x ≠ 0

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l52_52419


namespace sum_of_digits_divisible_by_45_l52_52392

theorem sum_of_digits_divisible_by_45 (a b : ℕ) (h1 : b = 0 ∨ b = 5) (h2 : (21 + a + b) % 9 = 0) : a + b = 6 :=
by
  sorry

end sum_of_digits_divisible_by_45_l52_52392


namespace manolo_face_mask_time_l52_52439
variable (x : ℕ)
def time_to_make_mask_first_hour := x
def face_masks_made_first_hour := 60 / x
def face_masks_made_next_three_hours := 180 / 6
def total_face_masks_in_four_hours := face_masks_made_first_hour + face_masks_made_next_three_hours

theorem manolo_face_mask_time : 
  total_face_masks_in_four_hours x = 45 ↔ x = 4 := sorry

end manolo_face_mask_time_l52_52439


namespace proportion_first_number_l52_52090

theorem proportion_first_number (x : ℝ) (h : x / 5 = 0.96 / 8) : x = 0.6 :=
by
  sorry

end proportion_first_number_l52_52090


namespace can_weigh_1kg_with_300g_and_650g_weights_l52_52850

-- Definitions based on conditions
def balance_scale (a b : ℕ) (w₁ w₂ : ℕ) : Prop :=
  a * w₁ + b * w₂ = 1000

-- Statement to prove based on the problem and solution
theorem can_weigh_1kg_with_300g_and_650g_weights (w₁ : ℕ) (w₂ : ℕ) (a b : ℕ)
  (h_w1 : w₁ = 300) (h_w2 : w₂ = 650) (h_a : a = 1) (h_b : b = 1) :
  balance_scale a b w₁ w₂ :=
by 
  -- We are given:
  -- - w1 = 300 g
  -- - w2 = 650 g
  -- - we want to measure 1000 g using these weights
  -- - a = 1
  -- - b = 1
  -- Prove that:
  --   a * w1 + b * w2 = 1000
  -- Which is:
  --   1 * 300 + 1 * 650 = 1000
  sorry

end can_weigh_1kg_with_300g_and_650g_weights_l52_52850


namespace visitors_surveyed_l52_52995

-- Given definitions
def total_visitors : ℕ := 400
def visitors_not_enjoyed_nor_understood : ℕ := 100
def E := total_visitors / 2
def U := total_visitors / 2

-- Using condition that 3/4th visitors enjoyed and understood
def enjoys_and_understands := (3 * total_visitors) / 4

-- Assert the equivalence of total number of visitors calculation
theorem visitors_surveyed:
  total_visitors = enjoys_and_understands + visitors_not_enjoyed_nor_understood :=
by
  sorry

end visitors_surveyed_l52_52995


namespace sequence_b_n_l52_52486

theorem sequence_b_n (b : ℕ → ℝ) 
  (h1 : b 1 = 3)
  (h2 : ∀ n ≥ 1, (b (n + 1))^3 = 27 * (b n)^3) :
  b 50 = 3^50 :=
sorry

end sequence_b_n_l52_52486


namespace g_five_l52_52404

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_one : g 1 = 2

theorem g_five : g 5 = 10 :=
by sorry

end g_five_l52_52404


namespace joe_average_speed_l52_52412

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem joe_average_speed :
  let distance1 := 420
  let speed1 := 60
  let distance2 := 120
  let speed2 := 40
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  average_speed total_distance total_time = 54 := by
sorry

end joe_average_speed_l52_52412


namespace coordinates_of_points_l52_52283

theorem coordinates_of_points
  (R : ℝ) (a b : ℝ)
  (hR : R = 10)
  (h_area : 1/2 * a * b = 600)
  (h_a_gt_b : a > b) :
  (a, 0) = (40, 0) ∧ (0, b) = (0, 30) ∧ (16, 18) = (16, 18) :=
  sorry

end coordinates_of_points_l52_52283


namespace math_problem_l52_52840

theorem math_problem (x : ℝ) : 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1/2 ∧ (x^2 + x^3 - 2 * x^4) / (x + x^2 - 2 * x^3) ≥ -1 ↔ 
  x ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ioc (-1/2 : ℝ) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 := 
by 
  sorry

end math_problem_l52_52840


namespace find_x_l52_52996

theorem find_x (x : ℝ) : x * 2.25 - (5 * 0.85) / 2.5 = 5.5 → x = 3.2 :=
by
  sorry

end find_x_l52_52996


namespace maximum_elements_in_A_l52_52041

theorem maximum_elements_in_A (n : ℕ) (h : n > 0)
  (A : Finset (Finset (Fin n))) 
  (hA : ∀ a ∈ A, ∀ b ∈ A, a ≠ b → ¬ a ⊆ b) :  
  A.card ≤ Nat.choose n (n / 2) :=
sorry

end maximum_elements_in_A_l52_52041


namespace bob_weekly_income_increase_l52_52269

theorem bob_weekly_income_increase
  (raise_per_hour : ℝ)
  (hours_per_week : ℝ)
  (benefit_reduction_per_month : ℝ)
  (weeks_per_month : ℝ)
  (h_raise : raise_per_hour = 0.50)
  (h_hours : hours_per_week = 40)
  (h_reduction : benefit_reduction_per_month = 60)
  (h_weeks : weeks_per_month = 4.33) :
  (raise_per_hour * hours_per_week - benefit_reduction_per_month / weeks_per_month) = 6.14 :=
by
  simp [h_raise, h_hours, h_reduction, h_weeks]
  norm_num
  sorry

end bob_weekly_income_increase_l52_52269


namespace john_quiz_goal_l52_52591

theorem john_quiz_goal
  (total_quizzes : ℕ)
  (goal_percentage : ℕ)
  (quizzes_completed : ℕ)
  (quizzes_remaining : ℕ)
  (quizzes_with_A_completed : ℕ)
  (total_quizzes_with_A_needed : ℕ)
  (additional_A_needed : ℕ)
  (quizzes_below_A_allowed : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 75)
  (h3 : quizzes_completed = 40)
  (h4 : quizzes_remaining = total_quizzes - quizzes_completed)
  (h5 : quizzes_with_A_completed = 27)
  (h6 : total_quizzes_with_A_needed = total_quizzes * goal_percentage / 100)
  (h7 : additional_A_needed = total_quizzes_with_A_needed - quizzes_with_A_completed)
  (h8 : quizzes_below_A_allowed = quizzes_remaining - additional_A_needed)
  (h_goal : quizzes_below_A_allowed ≤ 2) : quizzes_below_A_allowed = 2 :=
by
  sorry

end john_quiz_goal_l52_52591


namespace hyperbola_focal_coordinates_l52_52334

theorem hyperbola_focal_coordinates:
  ∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1 → ∃ c : ℝ, c = 5 ∧ (x = -c ∨ x = c) ∧ y = 0 :=
by
  intro x y
  sorry

end hyperbola_focal_coordinates_l52_52334


namespace fraction_eq_l52_52587

def f(x : ℤ) : ℤ := 3 * x + 2
def g(x : ℤ) : ℤ := 2 * x - 3

theorem fraction_eq : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by 
  sorry

end fraction_eq_l52_52587


namespace graph_fixed_point_l52_52926

theorem graph_fixed_point (f : ℝ → ℝ) (h : f 1 = 1) : f 1 = 1 :=
by
  sorry

end graph_fixed_point_l52_52926


namespace domain_of_g_l52_52070

-- Define the function f and specify the domain of f(x+1)
def f : ℝ → ℝ := sorry
def domain_f_x_plus_1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3} -- Domain of f(x+1) is [-1, 3]

-- Define the definition of the function g where g(x) = f(x^2)
def g (x : ℝ) : ℝ := f (x^2)

-- Prove that the domain of g(x) is [-2, 2]
theorem domain_of_g : {x | -2 ≤ x ∧ x ≤ 2} = {x | ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 4) ∧ (x = y ∨ x = -y)} :=
by 
  sorry

end domain_of_g_l52_52070


namespace find_integer_pairs_l52_52579

theorem find_integer_pairs (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) :=
by
  sorry

end find_integer_pairs_l52_52579


namespace valid_fractions_l52_52841

theorem valid_fractions :
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (1 ≤ z ∧ z ≤ 9) ∧
  (10 * x + y) % (10 * y + z) = 0 ∧ (10 * x + y) / (10 * y + z) = x / z :=
sorry

end valid_fractions_l52_52841


namespace population_increase_l52_52595

theorem population_increase (birth_rate : ℝ) (death_rate : ℝ) (initial_population : ℝ) :
  initial_population = 1000 →
  birth_rate = 32 / 1000 →
  death_rate = 11 / 1000 →
  ((birth_rate - death_rate) / initial_population) * 100 = 2.1 :=
by
  sorry

end population_increase_l52_52595


namespace simplify_polynomial_expression_l52_52715

theorem simplify_polynomial_expression (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) = r^3 - 4 * r^2 + 2 * r + 3 :=
by
  sorry

end simplify_polynomial_expression_l52_52715


namespace part_I_part_II_l52_52735

noncomputable def f (x : ℝ) : ℝ :=
  |x - (1/2)| + |x + (1/2)|

def solutionSetM : Set ℝ :=
  { x : ℝ | -1 < x ∧ x < 1 }

theorem part_I :
  { x : ℝ | f x < 2 } = solutionSetM := 
sorry

theorem part_II (a b : ℝ) (ha : a ∈ solutionSetM) (hb : b ∈ solutionSetM) :
  |a + b| < |1 + a * b| :=
sorry

end part_I_part_II_l52_52735


namespace reinforcement_size_l52_52311

theorem reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (days_remaining : ℕ) (reinforcement : ℕ) : 
  initial_men = 150 → initial_days = 31 → days_before_reinforcement = 16 → days_remaining = 5 → (150 * 15) = (150 + reinforcement) * 5 → reinforcement = 300 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end reinforcement_size_l52_52311


namespace chessboard_max_squares_l52_52438

def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

theorem chessboard_max_squares (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) : max_squares 1000 1000 = 1998 := 
by
  -- This is the theorem statement representing the maximum number of squares chosen
  -- in a 1000 x 1000 chessboard without having exactly three of them with two in the same row
  -- and two in the same column.
  sorry

end chessboard_max_squares_l52_52438


namespace annual_interest_payment_l52_52058

noncomputable def principal : ℝ := 9000
noncomputable def rate : ℝ := 9 / 100
noncomputable def time : ℝ := 1
noncomputable def interest : ℝ := principal * rate * time

theorem annual_interest_payment : interest = 810 := by
  sorry

end annual_interest_payment_l52_52058


namespace evaluate_expression_l52_52678

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end evaluate_expression_l52_52678


namespace tank_fill_time_l52_52144

-- Define the conditions
def capacity := 800
def rate_A := 40
def rate_B := 30
def rate_C := -20

def net_rate_per_cycle := rate_A + rate_B + rate_C
def cycle_duration := 3
def total_cycles := capacity / net_rate_per_cycle
def total_time := total_cycles * cycle_duration

-- The proof that tank will be full after 48 minutes
theorem tank_fill_time : total_time = 48 := by
  sorry

end tank_fill_time_l52_52144


namespace geometric_progression_solution_l52_52649

theorem geometric_progression_solution 
  (b₁ q : ℝ)
  (h₁ : b₁^3 * q^3 = 1728)
  (h₂ : b₁ * (1 + q + q^2) = 63) :
  (b₁ = 3 ∧ q = 4) ∨ (b₁ = 48 ∧ q = 1/4) :=
  sorry

end geometric_progression_solution_l52_52649


namespace blender_customers_l52_52949

variable (p_t p_b : ℕ) (c_t c_b : ℕ) (k : ℕ)

-- Define the conditions
def condition_toaster_popularity : p_t = 20 := sorry
def condition_toaster_cost : c_t = 300 := sorry
def condition_blender_cost : c_b = 450 := sorry
def condition_inverse_proportionality : p_t * c_t = k := sorry

-- Proof goal: number of customers who would buy the blender
theorem blender_customers : p_b = 13 :=
by
  have h1 : p_t * c_t = 6000 := by sorry -- Using the given conditions
  have h2 : p_b * c_b = 6000 := by sorry -- Assumption for the same constant k
  have h3 : c_b = 450 := sorry
  have h4 : p_b = 6000 / 450 := by sorry
  have h5 : p_b = 13 := by sorry
  exact h5

end blender_customers_l52_52949


namespace solution_to_problem_l52_52154

def f (x : ℝ) : ℝ := sorry

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem solution_to_problem
  (f : ℝ → ℝ)
  (h : functional_equation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end solution_to_problem_l52_52154


namespace books_in_school_libraries_correct_l52_52668

noncomputable def booksInSchoolLibraries : ℕ :=
  let booksInPublicLibrary := 1986
  let totalBooks := 7092
  totalBooks - booksInPublicLibrary

-- Now we create a theorem to check the correctness of our definition
theorem books_in_school_libraries_correct :
  booksInSchoolLibraries = 5106 := by
  sorry -- We skip the proof, as instructed

end books_in_school_libraries_correct_l52_52668


namespace quadratic_equation_reciprocal_integer_roots_l52_52897

noncomputable def quadratic_equation_conditions (a b c : ℝ) : Prop :=
  (∃ r : ℝ, (r * (1/r) = 1) ∧ (r + (1/r) = 4)) ∧ 
  (c = a) ∧ 
  (b = -4 * a)

theorem quadratic_equation_reciprocal_integer_roots (a b c : ℝ) (h1 : quadratic_equation_conditions a b c) : 
  c = a ∧ b = -4 * a :=
by
  obtain ⟨r, hr₁, hr₂⟩ := h1.1
  sorry

end quadratic_equation_reciprocal_integer_roots_l52_52897


namespace sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l52_52299

-- (1)
theorem sqrt_S_n_arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d) (h3 : S n = (n * (2 * a 1 + (n - 1) * (2 : ℝ))) / 2) :
  ∃ d, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d :=
by sorry

-- (2)
theorem seq_sqrt_S_n_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ) :
  (∃ d, ∀ n, S n / 2 = n * (a1 + (n - 1) * d)) ↔ (∀ n, S n = a1 * n^2) :=
by sorry

end sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l52_52299


namespace neg_p_is_necessary_but_not_sufficient_for_neg_q_l52_52562

variables (p q : Prop)

-- Given conditions: (p → q) and ¬(q → p)
theorem neg_p_is_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) :
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q) :=
sorry

end neg_p_is_necessary_but_not_sufficient_for_neg_q_l52_52562


namespace number_of_8_digit_integers_l52_52406

theorem number_of_8_digit_integers : 
  ∃ n, n = 90000000 ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
     d1 ≠ 0 → 0 ≤ d1 ∧ d1 ≤ 9 ∧ 
     0 ≤ d2 ∧ d2 ≤ 9 ∧ 
     0 ≤ d3 ∧ d3 ≤ 9 ∧ 
     0 ≤ d4 ∧ d4 ≤ 9 ∧ 
     0 ≤ d5 ∧ d5 ≤ 9 ∧ 
     0 ≤ d6 ∧ d6 ≤ 9 ∧ 
     0 ≤ d7 ∧ d7 ≤ 9 ∧ 
     0 ≤ d8 ∧ d8 ≤ 9 →
     ∀ count, count = (if d1 ≠ 0 then 9 * 10^7 else 0)) :=
sorry

end number_of_8_digit_integers_l52_52406


namespace ratio_of_sheep_to_horses_l52_52755

theorem ratio_of_sheep_to_horses (H : ℕ) (hH : 230 * H = 12880) (n_sheep : ℕ) (h_sheep : n_sheep = 56) :
  (n_sheep / H) = 1 := by
  sorry

end ratio_of_sheep_to_horses_l52_52755


namespace percentage_problem_l52_52674

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by
  sorry

end percentage_problem_l52_52674


namespace exists_integer_n_l52_52933

theorem exists_integer_n (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℤ, (n + 1981^k)^(1/2 : ℝ) + (n : ℝ)^(1/2 : ℝ) = (1982^(1/2 : ℝ) + 1) ^ k :=
sorry

end exists_integer_n_l52_52933


namespace addition_of_decimals_l52_52793

theorem addition_of_decimals : (0.3 + 0.03 : ℝ) = 0.33 := by
  sorry

end addition_of_decimals_l52_52793


namespace find_equation_of_line_l52_52619

theorem find_equation_of_line
  (m b : ℝ) 
  (h1 : ∃ k : ℝ, (k^2 - 2*k + 3 = k*m + b ∧ ∃ d : ℝ, d = 4) 
        ∧ (4*m - k^2 + 2*m*k - 3 + b = 0)) 
  (h2 : 8 = 2*m + b)
  (h3 : b ≠ 0) 
  : y = 8 :=
by 
  sorry

end find_equation_of_line_l52_52619


namespace probability_different_colors_l52_52114

def total_chips := 7 + 5 + 4

def probability_blue_draw : ℚ := 7 / total_chips
def probability_red_draw : ℚ := 5 / total_chips
def probability_yellow_draw : ℚ := 4 / total_chips
def probability_different_color (color1_prob color2_prob : ℚ) : ℚ := color1_prob * (1 - color2_prob)

theorem probability_different_colors :
  (probability_blue_draw * probability_different_color 7 (7 / total_chips)) +
  (probability_red_draw * probability_different_color 5 (5 / total_chips)) +
  (probability_yellow_draw * probability_different_color 4 (4 / total_chips)) 
  = 83 / 128 := 
by 
  sorry

end probability_different_colors_l52_52114


namespace time_after_9876_seconds_l52_52491

-- Define the initial time in seconds
def initial_seconds : ℕ := 6 * 3600

-- Define the elapsed time in seconds
def elapsed_seconds : ℕ := 9876

-- Convert given time in seconds to hours, minutes, and seconds
def time_in_hms (total_seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let hours := total_seconds / 3600
  let minutes := (total_seconds % 3600) / 60
  let seconds := total_seconds % 60
  (hours, minutes, seconds)

-- Define the final time in 24-hour format (08:44:36)
def final_time : (ℕ × ℕ × ℕ) := (8, 44, 36)

-- The question's proof statement
theorem time_after_9876_seconds : 
  time_in_hms (initial_seconds + elapsed_seconds) = final_time :=
sorry

end time_after_9876_seconds_l52_52491


namespace evaluate_g_at_3_l52_52980

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem evaluate_g_at_3 : g 3 = 79 := by
  sorry

end evaluate_g_at_3_l52_52980


namespace fraction_of_seats_sold_l52_52683

theorem fraction_of_seats_sold
  (ticket_price : ℕ) (number_of_rows : ℕ) (seats_per_row : ℕ) (total_earnings : ℕ)
  (h1 : ticket_price = 10)
  (h2 : number_of_rows = 20)
  (h3 : seats_per_row = 10)
  (h4 : total_earnings = 1500) :
  (total_earnings / ticket_price : ℕ) / (number_of_rows * seats_per_row : ℕ) = 3 / 4 := by
  sorry

end fraction_of_seats_sold_l52_52683


namespace find_original_wage_l52_52137

theorem find_original_wage (W : ℝ) (h : 1.50 * W = 51) : W = 34 :=
sorry

end find_original_wage_l52_52137


namespace horner_polynomial_rewrite_polynomial_value_at_5_l52_52610

def polynomial (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 6 * x^3 - 2 * x^2 - 5 * x - 2

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_polynomial_rewrite :
  polynomial = horner_polynomial := 
sorry

theorem polynomial_value_at_5 :
  polynomial 5 = 7548 := 
sorry

end horner_polynomial_rewrite_polynomial_value_at_5_l52_52610


namespace area_of_rectangle_l52_52784

theorem area_of_rectangle (w d : ℝ) (h_w : w = 4) (h_d : d = 5) : ∃ l : ℝ, (w^2 + l^2 = d^2) ∧ (w * l = 12) :=
by
  sorry

end area_of_rectangle_l52_52784


namespace sum_of_edges_l52_52565

theorem sum_of_edges (n : ℕ) (total_length large_edge small_edge : ℤ) : 
  n = 27 → 
  total_length = 828 → -- convert to millimeters
  large_edge = total_length / 12 → 
  small_edge = large_edge / 3 → 
  (large_edge + small_edge) / 10 = 92 :=
by
  intros
  sorry

end sum_of_edges_l52_52565


namespace fraction_of_n_is_80_l52_52744

-- Definitions from conditions
def n := (5 / 6) * 240

-- The theorem we want to prove
theorem fraction_of_n_is_80 : (2 / 5) * n = 80 :=
by
  -- This is just a placeholder to complete the statement, 
  -- actual proof logic is not included based on the prompt instructions
  sorry

end fraction_of_n_is_80_l52_52744


namespace minvalue_expression_l52_52910

theorem minvalue_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
    9 * z / (3 * x + y) + 9 * x / (y + 3 * z) + 4 * y / (x + z) ≥ 3 := 
by
  sorry

end minvalue_expression_l52_52910


namespace circle_equation_center_at_1_2_passing_through_origin_l52_52253

theorem circle_equation_center_at_1_2_passing_through_origin :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ∧
                (0 - 1)^2 + (0 - 2)^2 = 5 :=
by
  sorry

end circle_equation_center_at_1_2_passing_through_origin_l52_52253


namespace positive_solution_sqrt_a_sub_b_l52_52434

theorem positive_solution_sqrt_a_sub_b (a b : ℕ) (x : ℝ) 
  (h_eq : x^2 + 14 * x = 32) 
  (h_form : x = Real.sqrt a - b) 
  (h_pos_nat : a > 0 ∧ b > 0) : 
  a + b = 88 := 
by
  sorry

end positive_solution_sqrt_a_sub_b_l52_52434


namespace part1_part2_l52_52939

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Statement 1: If f(x) is an odd function, then a = 1.
theorem part1 (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) → a = 1 :=
sorry

-- Statement 2: If f(x) is defined on [-4, +∞), and for all x in the domain, 
-- f(cos(x) + b + 1/4) ≥ f(sin^2(x) - b - 3), then b ∈ [-1,1].
theorem part2 (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, f a (Real.cos x + b + 1/4) ≥ f a (Real.sin x ^ 2 - b - 3)) ∧
  (∀ x : ℝ, -4 ≤ x) ∧ -4 ≤ a ∧ a = 1 → -1 ≤ b ∧ b ≤ 1 :=
sorry

end part1_part2_l52_52939


namespace ram_account_balance_increase_l52_52754

theorem ram_account_balance_increase 
  (initial_deposit : ℕ := 500)
  (first_year_balance : ℕ := 600)
  (second_year_percentage_increase : ℕ := 32)
  (second_year_balance : ℕ := initial_deposit + initial_deposit * second_year_percentage_increase / 100) 
  (second_year_increase : ℕ := second_year_balance - first_year_balance) 
  : (second_year_increase * 100 / first_year_balance) = 10 := 
sorry

end ram_account_balance_increase_l52_52754


namespace P_plus_Q_l52_52413

theorem P_plus_Q (P Q : ℝ) (h : ∀ x, x ≠ 3 → (P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3))) : P + Q = 46 :=
sorry

end P_plus_Q_l52_52413


namespace MrKishoreSavings_l52_52162

noncomputable def TotalExpenses : ℕ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

noncomputable def MonthlySalary : ℕ :=
  (TotalExpenses * 10) / 9

noncomputable def Savings : ℕ :=
  (MonthlySalary * 1) / 10

theorem MrKishoreSavings :
  Savings = 2300 :=
by
  sorry

end MrKishoreSavings_l52_52162


namespace weight_of_triangular_piece_l52_52645

noncomputable def density_factor (weight : ℝ) (area : ℝ) : ℝ :=
  weight / area

noncomputable def square_weight (side_length : ℝ) (weight : ℝ) : ℝ := weight

noncomputable def triangle_area (side_length : ℝ) : ℝ :=
  (side_length ^ 2 * Real.sqrt 3) / 4

theorem weight_of_triangular_piece :
  let side_square := 4
  let weight_square := 16
  let side_triangle := 6
  let area_square := side_square ^ 2
  let area_triangle := triangle_area side_triangle
  let density_square := density_factor weight_square area_square
  let weight_triangle := area_triangle * density_square
  abs weight_triangle - 15.59 < 0.01 :=
by
  sorry

end weight_of_triangular_piece_l52_52645


namespace at_least_one_zero_l52_52099

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) : 
  a = 0 ∨ b = 0 ∨ c = 0 := 
sorry

end at_least_one_zero_l52_52099


namespace fraction_replaced_l52_52881

theorem fraction_replaced (x : ℝ) (h₁ : 0.15 * (1 - x) + 0.19000000000000007 * x = 0.16) : x = 0.25 :=
by
  sorry

end fraction_replaced_l52_52881


namespace sequence_monotonically_decreasing_l52_52322

theorem sequence_monotonically_decreasing (t : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = -↑n^2 + t * ↑n) →
  (∀ n : ℕ, a (n + 1) < a n) →
  t < 3 :=
by
  intros h1 h2
  sorry

end sequence_monotonically_decreasing_l52_52322


namespace jerry_stickers_l52_52764

variable (G F J : ℕ)

theorem jerry_stickers (h1 : F = 18) (h2 : G = F - 6) (h3 : J = 3 * G) : J = 36 :=
by {
  sorry
}

end jerry_stickers_l52_52764


namespace distance_between_A_and_B_l52_52590

noncomputable def distance_between_points (v_A v_B : ℝ) (t_meet t_A_to_B_after_meet : ℝ) : ℝ :=
  let t_total_A := t_meet + t_A_to_B_after_meet
  let t_total_B := t_meet + (t_meet - t_A_to_B_after_meet)
  let D := v_A * t_total_A + v_B * t_total_B
  D

-- Given conditions
def t_meet : ℝ := 4
def t_A_to_B_after_meet : ℝ := 3
def speed_difference : ℝ := 20

-- Function to calculate speeds based on given conditions
noncomputable def calculate_speeds (v_B : ℝ) : ℝ × ℝ :=
  let v_A := v_B + speed_difference
  (v_A, v_B)

-- Statement of the problem in Lean 4
theorem distance_between_A_and_B : ∃ (v_B v_A : ℝ), 
  v_A = v_B + speed_difference ∧
  distance_between_points v_A v_B t_meet t_A_to_B_after_meet = 240 :=
by 
  sorry

end distance_between_A_and_B_l52_52590


namespace exists_n0_find_N_l52_52745

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)

-- Definition of the sequence {a_n}
def seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = f (a n)

-- Problem (1): Existence of n0
theorem exists_n0 (a : ℕ → ℝ) (h_seq : seq a) (h_a1 : a 1 = 3) : 
  ∃ n0 : ℕ, ∀ n ≥ n0, a (n + 1) > a n :=
  sorry

-- Problem (2): Smallest N
theorem find_N (a : ℕ → ℝ) (h_seq : seq a) (m : ℕ) (h_m : m > 1) 
  (h_a1 : 1 + 1 / (m : ℝ) < a 1 ∧ a 1 < m / (m - 1)) : 
  ∃ N : ℕ, ∀ n ≥ N, 0 < a n ∧ a n < 1 :=
  sorry

end exists_n0_find_N_l52_52745


namespace functional_equation_solution_l52_52304

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x) + f (f y) = 2 * y + f (x - y)) ↔ (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l52_52304


namespace solve_number_puzzle_l52_52867

def number_puzzle (N : ℕ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14 → (40/100) * N = 168

theorem solve_number_puzzle : ∃ N, number_puzzle N := by
  sorry

end solve_number_puzzle_l52_52867


namespace maisy_new_job_hours_l52_52566

-- Define the conditions
def current_job_earnings : ℚ := 80
def new_job_wage_per_hour : ℚ := 15
def new_job_bonus : ℚ := 35
def earnings_difference : ℚ := 15

-- Define the problem
theorem maisy_new_job_hours (h : ℚ) 
  (h1 : current_job_earnings = 80) 
  (h2 : new_job_wage_per_hour * h + new_job_bonus = current_job_earnings + earnings_difference) :
  h = 4 :=
  sorry

end maisy_new_job_hours_l52_52566


namespace sum_of_two_numbers_l52_52906

theorem sum_of_two_numbers (x : ℤ) (sum certain value : ℤ) (h₁ : 25 - x = 5) : 25 + x = 45 := by
  sorry

end sum_of_two_numbers_l52_52906


namespace total_tax_in_cents_l52_52512

-- Declare the main variables and constants
def wage_per_hour_cents : ℕ := 2500
def local_tax_rate : ℝ := 0.02
def state_tax_rate : ℝ := 0.005

-- Define the total tax calculation as a proof statement
theorem total_tax_in_cents :
  local_tax_rate * wage_per_hour_cents + state_tax_rate * wage_per_hour_cents = 62.5 :=
by sorry

end total_tax_in_cents_l52_52512


namespace sequence_inequality_for_k_l52_52218

theorem sequence_inequality_for_k (k : ℝ) : 
  (∀ n : ℕ, 0 < n → (n + 1)^2 + k * (n + 1) + 2 > n^2 + k * n + 2) ↔ k > -3 :=
sorry

end sequence_inequality_for_k_l52_52218


namespace sum_g_eq_half_l52_52534

noncomputable def g (n : ℕ) : ℝ := ∑' k, if h : k ≥ 3 then (1 / (k : ℝ) ^ n) else 0

theorem sum_g_eq_half : (∑' n, if h : n ≥ 3 then g n else 0) = 1 / 2 := by
  sorry

end sum_g_eq_half_l52_52534


namespace area_between_chords_is_correct_l52_52866

noncomputable def circle_radius : ℝ := 10
noncomputable def chord_distance_apart : ℝ := 12
noncomputable def area_between_chords : ℝ := 44.73

theorem area_between_chords_is_correct 
    (r : ℝ) (d : ℝ) (A : ℝ) 
    (hr : r = circle_radius) 
    (hd : d = chord_distance_apart) 
    (hA : A = area_between_chords) : 
    ∃ area : ℝ, area = A := by 
  sorry

end area_between_chords_is_correct_l52_52866


namespace percent_of_y_l52_52309

theorem percent_of_y (y : ℝ) (hy : y > 0) : (8 * y) / 20 + (3 * y) / 10 = 0.7 * y :=
by
  sorry

end percent_of_y_l52_52309


namespace four_digit_unique_count_l52_52843

theorem four_digit_unique_count : 
  (∃ k : ℕ, k = 14 ∧ ∃ lst : List ℕ, lst.length = 4 ∧ 
    (∀ d ∈ lst, d = 2 ∨ d = 3) ∧ (2 ∈ lst) ∧ (3 ∈ lst)) :=
by
  sorry

end four_digit_unique_count_l52_52843


namespace relatively_prime_2n_plus_1_4n2_plus_1_l52_52723

theorem relatively_prime_2n_plus_1_4n2_plus_1 (n : ℕ) (h : n > 0) : 
  Nat.gcd (2 * n + 1) (4 * n^2 + 1) = 1 := 
by
  sorry

end relatively_prime_2n_plus_1_4n2_plus_1_l52_52723


namespace max_sum_of_digits_l52_52383

theorem max_sum_of_digits (a b c : ℕ) (x : ℕ) (N : ℕ) :
  N = 100 * a + 10 * b + c →
  100 <= N →
  N < 1000 →
  a ≠ 0 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1730 + x →
  a + b + c = 20 :=
by
  intros hN hN_ge_100 hN_lt_1000 ha_ne_0 hsum
  sorry

end max_sum_of_digits_l52_52383


namespace solve_fraction_l52_52621

open Real

theorem solve_fraction (x : ℝ) (hx : 1 - 4 / x + 4 / x^2 = 0) : 2 / x = 1 :=
by
  -- We'll include the necessary steps of the proof here, but for now we leave it as sorry.
  sorry

end solve_fraction_l52_52621


namespace Jerry_needs_72_dollars_l52_52863

def action_figures_current : ℕ := 7
def action_figures_total : ℕ := 16
def cost_per_figure : ℕ := 8
def money_needed : ℕ := 72

theorem Jerry_needs_72_dollars : 
  (action_figures_total - action_figures_current) * cost_per_figure = money_needed :=
by
  sorry

end Jerry_needs_72_dollars_l52_52863


namespace find_k_inv_h_of_10_l52_52460

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end find_k_inv_h_of_10_l52_52460


namespace tangent_slope_at_one_l52_52122

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x + Real.sqrt x

theorem tangent_slope_at_one :
  (deriv f 1) = 3 / 2 :=
by
  sorry

end tangent_slope_at_one_l52_52122


namespace cara_optimal_reroll_two_dice_probability_l52_52504

def probability_reroll_two_dice : ℚ :=
  -- Probability derived from Cara's optimal reroll decisions
  5 / 27

theorem cara_optimal_reroll_two_dice_probability :
  cara_probability_optimal_reroll_two_dice = 5 / 27 := by sorry

end cara_optimal_reroll_two_dice_probability_l52_52504


namespace find_multiple_l52_52321

variables (total_questions correct_answers score : ℕ)
variable (m : ℕ)
variable (incorrect_answers : ℕ := total_questions - correct_answers)

-- Given conditions
axiom total_questions_eq : total_questions = 100
axiom correct_answers_eq : correct_answers = 92
axiom score_eq : score = 76

-- Define the scoring method
def score_formula : ℕ := correct_answers - m * incorrect_answers

-- Statement to prove
theorem find_multiple : score = 76 → correct_answers = 92 → total_questions = 100 → score_formula total_questions correct_answers m = score → m = 2 := by
  intros h1 h2 h3 h4
  sorry

end find_multiple_l52_52321


namespace true_and_false_propositions_l52_52466

theorem true_and_false_propositions (p q : Prop) 
  (hp : p = true) (hq : q = false) : (¬q) = true :=
by
  sorry

end true_and_false_propositions_l52_52466


namespace remainder_of_division_l52_52993

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l52_52993


namespace solve_inequality_l52_52749

theorem solve_inequality (x : ℝ) : 
  (-9 * x^2 + 6 * x + 15 > 0) ↔ (x > -1 ∧ x < 5/3) := 
sorry

end solve_inequality_l52_52749


namespace find_m_l52_52151

def f (x m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

theorem find_m :
  let m := 10 / 7
  3 * f 5 m = 2 * g 5 m :=
by
  sorry

end find_m_l52_52151


namespace initial_amount_correct_l52_52282

-- Definitions
def spent_on_fruits : ℝ := 15.00
def left_to_spend : ℝ := 85.00
def initial_amount_given (spent: ℝ) (left: ℝ) : ℝ := spent + left

-- Theorem stating the problem
theorem initial_amount_correct :
  initial_amount_given spent_on_fruits left_to_spend = 100.00 :=
by
  sorry

end initial_amount_correct_l52_52282


namespace isosceles_triangle_perimeter_l52_52658

-- Define an isosceles triangle structure
structure IsoscelesTriangle where
  (a b c : ℝ) 
  (isosceles : a = b ∨ a = c ∨ b = c)
  (side_lengths : (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧ (c = 2 ∨ c = 3))
  (valid_triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem to prove the perimeter
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.a + t.b + t.c = 7 ∨ t.a + t.b + t.c = 8 :=
sorry

end isosceles_triangle_perimeter_l52_52658


namespace min_value_fraction_l52_52830

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  (∃ y, y > 9 ∧ (∀ z, z > 9 → y ≤ (z^3 / (z - 9)))) ∧ (∀ z, z > 9 → (∃ w, w > 9 ∧ z^3 / (z - 9) = 325)) := 
  sorry

end min_value_fraction_l52_52830


namespace alley_width_l52_52055

noncomputable def calculate_width (l k h : ℝ) : ℝ :=
  l / 2

theorem alley_width (k h l w : ℝ) (h1 : k = (l * (Real.sin (Real.pi / 3)))) (h2 : h = (l * (Real.sin (Real.pi / 6)))) :
  w = calculate_width l k h :=
by
  sorry

end alley_width_l52_52055


namespace percentage_of_singles_l52_52325

/-- In a baseball season, Lisa had 50 hits. Among her hits were 2 home runs, 
2 triples, 8 doubles, and 1 quadruple. The rest of her hits were singles. 
What percent of her hits were singles? --/
theorem percentage_of_singles
  (total_hits : ℕ := 50)
  (home_runs : ℕ := 2)
  (triples : ℕ := 2)
  (doubles : ℕ := 8)
  (quadruples : ℕ := 1)
  (non_singles := home_runs + triples + doubles + quadruples)
  (singles := total_hits - non_singles) :
  (singles : ℚ) / (total_hits : ℚ) * 100 = 74 := by
  sorry

end percentage_of_singles_l52_52325


namespace problem_1_problem_2_l52_52138

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem problem_1 (x : ℝ) : f x ≥ 2 ↔ (x ≤ -7 ∨ x ≥ 5 / 3) :=
sorry

theorem problem_2 : ∃ x : ℝ, f x = -9 / 2 :=
sorry

end problem_1_problem_2_l52_52138


namespace third_cyclist_speed_l52_52141

theorem third_cyclist_speed (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : 
  ∃ V : ℝ, V = (a + 3 * b + Real.sqrt (a^2 - 10 * a * b + 9 * b^2)) / 4 :=
by
  sorry

end third_cyclist_speed_l52_52141


namespace hyperbola_eccentricity_l52_52808

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b = -4 * a / 3)
  (hc : c = (Real.sqrt (a ^ 2 + b ^ 2)))
  (point_on_asymptote : ∃ x y : ℝ, x = 3 ∧ y = -4 ∧ (y = b / a * x ∨ y = -b / a * x)) :
  (c / a) = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l52_52808


namespace sarahs_loan_amount_l52_52945

theorem sarahs_loan_amount 
  (down_payment : ℕ := 10000)
  (monthly_payment : ℕ := 600)
  (repayment_years : ℕ := 5)
  (interest_rate : ℚ := 0) : down_payment + (monthly_payment * (12 * repayment_years)) = 46000 :=
by
  sorry

end sarahs_loan_amount_l52_52945


namespace johns_earnings_without_bonus_l52_52578
-- Import the Mathlib library to access all necessary functions and definitions

-- Define the conditions of the problem
def hours_without_bonus : ℕ := 8
def bonus_amount : ℕ := 20
def extra_hours_for_bonus : ℕ := 2
def hours_with_bonus : ℕ := hours_without_bonus + extra_hours_for_bonus
def hourly_wage_with_bonus : ℕ := 10

-- Define the total earnings with the performance bonus
def total_earnings_with_bonus : ℕ := hours_with_bonus * hourly_wage_with_bonus

-- Statement to prove the earnings without the bonus
theorem johns_earnings_without_bonus :
  total_earnings_with_bonus - bonus_amount = 80 :=
by
  -- Placeholder for the proof
  sorry

end johns_earnings_without_bonus_l52_52578


namespace remainder_of_76_pow_k_mod_7_is_6_l52_52997

theorem remainder_of_76_pow_k_mod_7_is_6 (k : ℕ) (hk : k % 2 = 1) : (76 ^ k) % 7 = 6 :=
sorry

end remainder_of_76_pow_k_mod_7_is_6_l52_52997


namespace player2_wins_l52_52586

-- Definitions for the initial conditions and game rules
def initial_piles := [10, 15, 20]
def split_rule (piles : List ℕ) (move : ℕ → ℕ × ℕ) : List ℕ :=
  let (pile1, pile2) := move (piles.head!)
  (pile1 :: pile2 :: piles.tail!)

-- Winning condition proof
theorem player2_wins :
  ∀ piles : List ℕ, piles = [10, 15, 20] →
  (∀ move_count : ℕ, move_count = 42 →
  (move_count > 0 ∧ ¬ ∃ split : ℕ → ℕ × ℕ, move_count % 2 = 1)) :=
by
  intro piles hpiles
  intro move_count hmove_count
  sorry

end player2_wins_l52_52586


namespace winter_spending_l52_52547

-- Define the total spending by the end of November
def total_spending_end_november : ℝ := 3.3

-- Define the total spending by the end of February
def total_spending_end_february : ℝ := 7.0

-- Formalize the problem: prove that the spending during December, January, and February is 3.7 million dollars
theorem winter_spending : total_spending_end_february - total_spending_end_november = 3.7 := by
  sorry

end winter_spending_l52_52547


namespace no_positive_integer_solution_l52_52599

theorem no_positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ¬ (x^2 * y^4 - x^4 * y^2 + 4 * x^2 * y^2 * z^2 + x^2 * z^4 - y^2 * z^4 = 0) :=
sorry

end no_positive_integer_solution_l52_52599


namespace tv_sales_value_increase_l52_52353

theorem tv_sales_value_increase (P V : ℝ) :
    let P1 := 0.82 * P
    let V1 := 1.72 * V
    let P2 := 0.75 * P1
    let V2 := 1.90 * V1
    let initial_sales := P * V
    let final_sales := P2 * V2
    final_sales = 2.00967 * initial_sales :=
by
  sorry

end tv_sales_value_increase_l52_52353


namespace distinct_solutions_difference_l52_52655

theorem distinct_solutions_difference (r s : ℝ) (hr : (r - 5) * (r + 5) = 25 * r - 125)
  (hs : (s - 5) * (s + 5) = 25 * s - 125) (neq : r ≠ s) (hgt : r > s) : r - s = 15 := by
  sorry

end distinct_solutions_difference_l52_52655


namespace empty_subset_of_A_l52_52593

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A :=
by
  sorry

end empty_subset_of_A_l52_52593


namespace no_solution_exists_l52_52018

   theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
     ¬ (3 / a + 4 / b = 12 / (a + b)) := 
   sorry
   
end no_solution_exists_l52_52018


namespace common_measure_largest_l52_52174

theorem common_measure_largest {a b : ℕ} (h_a : a = 15) (h_b : b = 12): 
  (∀ c : ℕ, c ∣ a ∧ c ∣ b → c ≤ Nat.gcd a b) ∧ Nat.gcd a b = 3 := 
by
  sorry

end common_measure_largest_l52_52174


namespace exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l52_52538

theorem exists_n_such_that_5_pow_n_has_six_consecutive_zeros :
  ∃ n : ℕ, n < 1000000 ∧ ∃ k : ℕ, k = 20 ∧ 5 ^ n % (10 ^ k) < (10 ^ (k - 6)) :=
by
  -- proof goes here
  sorry

end exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l52_52538


namespace kim_shoes_l52_52785

variable (n : ℕ)

theorem kim_shoes : 
  (∀ n, 2 * n = 6 → (1 : ℚ) / (2 * n - 1) = (1 : ℚ) / 5 → n = 3) := 
sorry

end kim_shoes_l52_52785


namespace relationship_of_AT_l52_52429

def S : ℝ := 300
def PC : ℝ := S + 500
def total_cost : ℝ := 2200

theorem relationship_of_AT (AT : ℝ) 
  (h1: S + PC + AT = total_cost) : 
  AT = S + PC - 400 :=
by
  sorry

end relationship_of_AT_l52_52429


namespace find_y_value_l52_52228

theorem find_y_value (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 3 - 2 * t)
  (h2 : y = 3 * t + 6)
  (h3 : x = -6)
  : y = 19.5 :=
by {
  sorry
}

end find_y_value_l52_52228


namespace monotonic_intervals_find_f_max_l52_52637

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem monotonic_intervals :
  (∀ x, 0 < x → x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
  (∀ x, x > Real.exp 1 → (1 - Real.log x) / x^2 < 0) :=
sorry

theorem find_f_max (m : ℝ) (h : m > 0) :
  if 0 < 2 * m ∧ 2 * m ≤ Real.exp 1 then f (2 * m) = Real.log (2 * m) / (2 * m)
  else if m ≥ Real.exp 1 then f m = Real.log m / m
  else f (Real.exp 1) = 1 / Real.exp 1 :=
sorry

end monotonic_intervals_find_f_max_l52_52637


namespace height_relationship_l52_52625

theorem height_relationship (B V G : ℝ) (h1 : B = 2 * V) (h2 : V = (2 / 3) * G) : B = (4 / 3) * G :=
sorry

end height_relationship_l52_52625


namespace bs_sequence_bounded_iff_f_null_l52_52245

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = abs (a (n + 1) - a (n + 2))

def f_null (a : ℕ → ℝ) : Prop :=
  ∀ n k, a n * a k * (a n - a k) = 0

def bs_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, abs (a n) ≤ M

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (bs_bounded a ↔ f_null a) := by
  sorry

end bs_sequence_bounded_iff_f_null_l52_52245


namespace remainder_3x_minus_6_divides_P_l52_52972

def P(x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 8 * x^4 + 3 * x^3 - 5
def D(x : ℝ) : ℝ := 3 * x - 6

theorem remainder_3x_minus_6_divides_P :
  P 2 = 915 :=
by
  sorry

end remainder_3x_minus_6_divides_P_l52_52972


namespace maximum_value_of_f_l52_52531

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem maximum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = Real.exp 1 :=
sorry

end maximum_value_of_f_l52_52531


namespace manfred_average_paycheck_l52_52287

def average_paycheck : ℕ → ℕ → ℕ → ℕ := fun total_paychecks first_paychecks_value num_first_paychecks =>
  let remaining_paychecks_value := first_paychecks_value + 20
  let total_payment := (num_first_paychecks * first_paychecks_value) + ((total_paychecks - num_first_paychecks) * remaining_paychecks_value)
  let average_payment := total_payment / total_paychecks
  average_payment

theorem manfred_average_paycheck :
  average_paycheck 26 750 6 = 765 := by
  sorry

end manfred_average_paycheck_l52_52287


namespace area_ratio_of_squares_l52_52131

theorem area_ratio_of_squares (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a * a) = 16 * (b * b) :=
by
  sorry

end area_ratio_of_squares_l52_52131


namespace captivating_quadruples_count_l52_52271

theorem captivating_quadruples_count :
  (∃ n : ℕ, n = 682) ↔ 
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d < b + c :=
sorry

end captivating_quadruples_count_l52_52271


namespace mary_friends_count_l52_52305

-- Definitions based on conditions
def total_stickers := 50
def stickers_left := 8
def total_students := 17
def classmates := total_students - 1 -- excluding Mary

-- Defining the proof problem
theorem mary_friends_count (F : ℕ) (h1 : 4 * F + 2 * (classmates - F) = total_stickers - stickers_left) :
  F = 5 :=
by sorry

end mary_friends_count_l52_52305


namespace original_selling_price_is_1100_l52_52300

-- Let P be the original purchase price.
variable (P : ℝ)

-- Condition 1: Bill made a profit of 10% on the original purchase price.
def original_selling_price := 1.10 * P

-- Condition 2: If he had purchased that product for 10% less 
-- and sold it at a profit of 30%, he would have received $70 more.
def new_purchase_price := 0.90 * P
def new_selling_price := 1.17 * P
def price_difference := new_selling_price - original_selling_price

-- Theorem: The original selling price was $1100.
theorem original_selling_price_is_1100 (h : price_difference P = 70) : 
  original_selling_price P = 1100 :=
sorry

end original_selling_price_is_1100_l52_52300


namespace target_hit_probability_l52_52847

/-- 
The probabilities for two shooters to hit a target are 1/2 and 1/3, respectively.
If both shooters fire at the target simultaneously, the probability that the target 
will be hit is 2/3.
-/
theorem target_hit_probability (P₁ P₂ : ℚ) (h₁ : P₁ = 1/2) (h₂ : P₂ = 1/3) :
  1 - ((1 - P₁) * (1 - P₂)) = 2/3 :=
by
  sorry

end target_hit_probability_l52_52847


namespace rectangle_vertex_x_coordinate_l52_52679

theorem rectangle_vertex_x_coordinate
  (x : ℝ)
  (y1 y2 : ℝ)
  (slope : ℝ)
  (h1 : x = 1)
  (h2 : 9 = 9)
  (h3 : slope = 0.2)
  (h4 : y1 = 0)
  (h5 : y2 = 2)
  (h6 : ∀ (x : ℝ), (0.2 * x : ℝ) = 1 → x = 1) :
  x = 1 := 
by sorry

end rectangle_vertex_x_coordinate_l52_52679


namespace transmitted_word_is_PAROHOD_l52_52339

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end transmitted_word_is_PAROHOD_l52_52339


namespace longest_side_length_l52_52509

-- Define the sides of the triangle
def side_a : ℕ := 9
def side_b (x : ℕ) : ℕ := 2 * x + 3
def side_c (x : ℕ) : ℕ := 3 * x - 2

-- Define the perimeter condition
def perimeter_condition (x : ℕ) : Prop := side_a + side_b x + side_c x = 45

-- Main theorem statement: Length of the longest side is 19
theorem longest_side_length (x : ℕ) (h : perimeter_condition x) : side_b x = 19 ∨ side_c x = 19 :=
sorry

end longest_side_length_l52_52509


namespace max_area_rectangle_l52_52168

theorem max_area_rectangle :
  ∃ (l w : ℕ), (2 * (l + w) = 40) ∧ (l ≥ w + 3) ∧ (l * w = 91) :=
by
  sorry

end max_area_rectangle_l52_52168


namespace total_time_to_row_l52_52278

theorem total_time_to_row (boat_speed_in_still_water : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed_in_still_water = 9 → stream_speed = 1.5 → distance = 105 → 
  (distance / (boat_speed_in_still_water + stream_speed)) + (distance / (boat_speed_in_still_water - stream_speed)) = 24 :=
by
  intro h_boat_speed h_stream_speed h_distance
  rw [h_boat_speed, h_stream_speed, h_distance]
  sorry

end total_time_to_row_l52_52278


namespace negative_three_degrees_below_zero_l52_52772

-- Definitions based on conditions
def positive_temperature (t : ℤ) : Prop := t > 0
def negative_temperature (t : ℤ) : Prop := t < 0
def above_zero (t : ℤ) : Prop := positive_temperature t
def below_zero (t : ℤ) : Prop := negative_temperature t

-- Example given in conditions
def ten_degrees_above_zero := above_zero 10

-- Lean 4 statement for the proof
theorem negative_three_degrees_below_zero : below_zero (-3) :=
by
  sorry

end negative_three_degrees_below_zero_l52_52772


namespace quadratic_eq_roots_quadratic_eq_range_l52_52422

theorem quadratic_eq_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0 ∧ x1 + 3 * x2 = 2 * m + 8) →
  (m = -1 ∨ m = -2) :=
sorry

theorem quadratic_eq_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0) →
  m ≤ 0 :=
sorry

end quadratic_eq_roots_quadratic_eq_range_l52_52422


namespace simplify_expr1_simplify_expr2_l52_52515

variable (x y : ℝ)

theorem simplify_expr1 : 
  3 * x^2 - 2 * x * y + y^2 - 3 * x^2 + 3 * x * y = x * y + y^2 :=
by
  sorry

theorem simplify_expr2 : 
  (7 * x^2 - 3 * x * y) - 6 * (x^2 - 1/3 * x * y) = x^2 - x * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l52_52515


namespace becky_to_aliyah_ratio_l52_52848

def total_school_days : ℕ := 180
def days_aliyah_packs_lunch : ℕ := total_school_days / 2
def days_becky_packs_lunch : ℕ := 45

theorem becky_to_aliyah_ratio :
  (days_becky_packs_lunch : ℚ) / days_aliyah_packs_lunch = 1 / 2 := by
  sorry

end becky_to_aliyah_ratio_l52_52848


namespace square_diagonal_l52_52498

theorem square_diagonal (P : ℝ) (d : ℝ) (hP : P = 200 * Real.sqrt 2) :
  d = 100 :=
by
  sorry

end square_diagonal_l52_52498


namespace part_I_part_II_l52_52665

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1))

theorem part_I (a : ℝ) (h_a_pos : a > 0) : (∀ x > 0, (1 / ((2 * x + 1) * (a * (2 * x + 1) - (2 * (a * x + 1) / 2))) ≥ 0) ↔ a ≥ 2) :=
sorry

theorem part_II : ∃ a : ℝ, (∀ x > 0, (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1)) ≥ 1) ∧ (Real.log (a * (Real.sqrt ((2 - a) / (4 * a))) + 1 / 2) + (2 / (2 * (Real.sqrt ((2 - a) / (4 * a))) + 1)) = 1) ∧ a = 1 :=
sorry

end part_I_part_II_l52_52665


namespace rectangular_plot_breadth_l52_52015

theorem rectangular_plot_breadth (b : ℝ) 
    (h1 : ∃ l : ℝ, l = 3 * b)
    (h2 : 432 = 3 * b * b) : b = 12 :=
by
  sorry

end rectangular_plot_breadth_l52_52015


namespace smallest_possible_value_of_other_number_l52_52920

theorem smallest_possible_value_of_other_number (x n : ℕ) (h_pos : x > 0) 
  (h_gcd : Nat.gcd 72 n = x + 6) (h_lcm : Nat.lcm 72 n = x * (x + 6)) : n = 12 := by
  sorry

end smallest_possible_value_of_other_number_l52_52920


namespace triangle_angle_contradiction_l52_52019

theorem triangle_angle_contradiction (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) :
  false :=
by
  have h : α + β + γ > 180 := by
  { linarith }
  linarith

end triangle_angle_contradiction_l52_52019


namespace simplify_expression_l52_52327

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (yz + xz + xy) / (xyz * (x + y + z)) :=
by
  sorry

end simplify_expression_l52_52327


namespace tip_percentage_is_20_l52_52705

noncomputable def total_bill : ℕ := 16 + 14
noncomputable def james_share : ℕ := total_bill / 2
noncomputable def james_paid : ℕ := 21
noncomputable def tip_amount : ℕ := james_paid - james_share
noncomputable def tip_percentage : ℕ := (tip_amount * 100) / total_bill 

theorem tip_percentage_is_20 :
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_is_20_l52_52705


namespace triangle_is_isosceles_l52_52788

/-- Given triangle ABC with angles A, B, and C, where C = π - (A + B),
    if 2 * sin A * cos B = sin C, then triangle ABC is an isosceles triangle -/
theorem triangle_is_isosceles
  (A B C : ℝ)
  (hC : C = π - (A + B))
  (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B :=
by
  sorry

end triangle_is_isosceles_l52_52788


namespace evaluate_Y_l52_52199

def Y (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2 + 3

theorem evaluate_Y : Y 2 5 = 2 :=
by
  sorry

end evaluate_Y_l52_52199


namespace product_4_6_7_14_l52_52164

theorem product_4_6_7_14 : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end product_4_6_7_14_l52_52164


namespace original_price_of_bag_l52_52210

theorem original_price_of_bag (P : ℝ) 
  (h1 : ∀ x, 0 < x → x < 1 → x * 100 = 75)
  (h2 : 2 * (0.25 * P) = 3)
  : P = 6 :=
sorry

end original_price_of_bag_l52_52210


namespace patients_per_doctor_l52_52194

theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) (h_patients : total_patients = 400) (h_doctors : total_doctors = 16) : 
  (total_patients / total_doctors) = 25 :=
by
  sorry

end patients_per_doctor_l52_52194


namespace false_conjunction_l52_52437

theorem false_conjunction (p q : Prop) (h : ¬(p ∧ q)) : ¬ (¬p ∧ ¬q) := sorry

end false_conjunction_l52_52437


namespace find_k_l52_52832

theorem find_k (x y k : ℝ) (h1 : x + 2 * y = k + 1) (h2 : 2 * x + y = 1) (h3 : x + y = 3) : k = 7 :=
by
  sorry

end find_k_l52_52832


namespace triangle_perimeter_l52_52014

-- Define the conditions of the problem
def a := 4
def b := 8
def quadratic_eq (x : ℝ) : Prop := x^2 - 14 * x + 40 = 0

-- Define the perimeter calculation, ensuring triangle inequality and correct side length
def valid_triangle (x : ℝ) : Prop :=
  x ≠ a ∧ x ≠ b ∧ quadratic_eq x ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)

-- Define the problem statement as a theorem
theorem triangle_perimeter : ∃ x : ℝ, valid_triangle x ∧ (a + b + x = 22) :=
by {
  -- Placeholder for the proof
  sorry
}

end triangle_perimeter_l52_52014


namespace min_fraction_value_l52_52666

theorem min_fraction_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_tangent : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 :=
by
  sorry

end min_fraction_value_l52_52666


namespace part_a_part_b_l52_52585

namespace ProofProblem

def number_set := {n : ℕ | ∃ k : ℕ, n = (10^k - 1)}

noncomputable def special_structure (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2 * m + 1 ∨ n = 2 * m + 2

theorem part_a :
  ∃ (a b c : ℕ) (ha : a ∈ number_set) (hb : b ∈ number_set) (hc : c ∈ number_set),
    special_structure (a + b + c) :=
by
  sorry

theorem part_b (cards : List ℕ) (h : ∀ x ∈ cards, x ∈ number_set)
    (hs : special_structure (cards.sum)) :
  ∃ (d : ℕ), d ≠ 2 ∧ (d = 0 ∨ d = 1) :=
by
  sorry

end ProofProblem

end part_a_part_b_l52_52585


namespace first_machine_rate_l52_52554

theorem first_machine_rate (x : ℕ) (h1 : 30 * x + 30 * 65 = 3000) : x = 35 := sorry

end first_machine_rate_l52_52554


namespace mn_value_l52_52153

theorem mn_value (m n : ℤ) (h1 : m = n + 2) (h2 : 2 * m + n = 4) : m * n = 0 := by
  sorry

end mn_value_l52_52153


namespace num_boys_in_class_l52_52377

-- Definitions based on conditions
def num_positions (p1 p2 : Nat) (total : Nat) : Nat :=
  if h : p1 < p2 then p2 - p1
  else total - (p1 - p2)

theorem num_boys_in_class (p1 p2 : Nat) (total : Nat) :
  p1 = 6 ∧ p2 = 16 ∧ num_positions p1 p2 total = 10 → total = 22 :=
by
  intros h
  sorry

end num_boys_in_class_l52_52377


namespace max_third_side_l52_52869

open Real

variables {A B C : ℝ} {a b c : ℝ} 

theorem max_third_side (h : cos (4 * A) + cos (4 * B) + cos (4 * C) = 1) 
                       (ha : a = 8) (hb : b = 15) : c = 17 :=
 by
  sorry 

end max_third_side_l52_52869


namespace prob_sum_divisible_by_4_l52_52687

-- Defining the set and its properties
def set : Finset ℕ := {1, 2, 3, 4, 5}

def isDivBy4 (n : ℕ) : Prop := n % 4 = 0

-- Defining a function to calculate combinations
def combinations (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Defining the successful outcomes and the total combinations
def successfulOutcomes : ℕ := 3
def totalOutcomes : ℕ := combinations 5 3

-- Defining the probability
def probability : ℚ := successfulOutcomes / ↑totalOutcomes

-- The proof problem
theorem prob_sum_divisible_by_4 : probability = 3 / 10 := by
  sorry

end prob_sum_divisible_by_4_l52_52687


namespace pure_imaginary_m_value_l52_52696

theorem pure_imaginary_m_value (m : ℝ) (h₁ : m ^ 2 + m - 2 = 0) (h₂ : m ^ 2 - 1 ≠ 0) : m = -2 := by
  sorry

end pure_imaginary_m_value_l52_52696


namespace lcm_18_45_l52_52859

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l52_52859


namespace hyogeun_weight_l52_52140

noncomputable def weights_are_correct : Prop :=
  ∃ H S G : ℝ, 
    H + S + G = 106.6 ∧
    G = S - 7.7 ∧
    S = H - 4.8 ∧
    H = 41.3

theorem hyogeun_weight : weights_are_correct :=
by
  sorry

end hyogeun_weight_l52_52140


namespace domain_of_tan_l52_52694

theorem domain_of_tan :
    ∀ k : ℤ, ∀ x : ℝ,
    (x > (k * π / 2 - π / 8) ∧ x < (k * π / 2 + 3 * π / 8)) ↔
    2 * x - π / 4 ≠ k * π + π / 2 :=
by
  intro k x
  sorry

end domain_of_tan_l52_52694


namespace trigonometric_expression_l52_52868

variable (α : Real)
open Real

theorem trigonometric_expression (h : tan α = 3) : 
  (2 * sin α - cos α) / (sin α + 3 * cos α) = 5 / 6 := 
by
  sorry

end trigonometric_expression_l52_52868


namespace matrix_mult_3I_l52_52896

variable (N : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_mult_3I (w : Fin 3 → ℝ):
  (∀ (w : Fin 3 → ℝ), N.mulVec w = 3 * w) ↔ (N = 3 • (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_mult_3I_l52_52896


namespace beau_age_today_l52_52111

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l52_52111


namespace polynomial_coefficients_sum_and_difference_l52_52821

theorem polynomial_coefficients_sum_and_difference :
  ∀ (a_0 a_1 a_2 a_3 a_4 : ℤ),
  (∀ (x : ℤ), (2 * x - 3)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_1 + a_2 + a_3 + a_4 = -80) ∧ ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625) :=
by
  intros a_0 a_1 a_2 a_3 a_4 h
  sorry

end polynomial_coefficients_sum_and_difference_l52_52821


namespace cone_cube_volume_ratio_l52_52741

noncomputable def volumeRatio (s : ℝ) : ℝ :=
  let r := s / 2
  let h := s
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  let volume_cube := s^3
  volume_cone / volume_cube

theorem cone_cube_volume_ratio (s : ℝ) (h_cube_eq_s : s > 0) :
  volumeRatio s = Real.pi / 12 :=
by
  sorry

end cone_cube_volume_ratio_l52_52741


namespace linear_equation_solution_l52_52275

theorem linear_equation_solution (x y b : ℝ) (h1 : x - 2*y + b = 0) (h2 : y = (1/2)*x + b - 1) :
  b = 2 :=
by
  sorry

end linear_equation_solution_l52_52275


namespace number_of_women_attended_l52_52390

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l52_52390


namespace slope_of_line_eq_neg_four_thirds_l52_52432

variable {x y : ℝ}
variable (p₁ p₂ : ℝ × ℝ) (h₁ : 3 / p₁.1 + 4 / p₁.2 = 0) (h₂ : 3 / p₂.1 + 4 / p₂.2 = 0)

theorem slope_of_line_eq_neg_four_thirds 
  (hneq : p₁.1 ≠ p₂.1):
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1) = -4 / 3 := 
sorry

end slope_of_line_eq_neg_four_thirds_l52_52432


namespace Durakavalyanie_last_lesson_class_1C_l52_52648

theorem Durakavalyanie_last_lesson_class_1C :
  ∃ (class_lesson : String × Nat → String), 
  class_lesson ("1B", 1) = "Kurashenie" ∧
  (∃ (k m n : Nat), class_lesson ("1A", k) = "Durakavalyanie" ∧ class_lesson ("1B", m) = "Durakavalyanie" ∧ m > k) ∧
  class_lesson ("1A", 2) ≠ "Nizvedenie" ∧
  class_lesson ("1C", 3) = "Durakavalyanie" :=
sorry

end Durakavalyanie_last_lesson_class_1C_l52_52648


namespace no_real_sol_l52_52006

open Complex

theorem no_real_sol (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (↑(x.re) ≠ x ∨ ↑(y.re) ≠ y) → (x + y) / y ≠ x / (y + x) := by
  sorry

end no_real_sol_l52_52006


namespace train_stops_one_minute_per_hour_l52_52864

theorem train_stops_one_minute_per_hour (D : ℝ) (h1 : D / 400 = T₁) (h2 : D / 360 = T₂) : 
  (T₂ - T₁) * 60 = 1 :=
by
  sorry

end train_stops_one_minute_per_hour_l52_52864


namespace find_sum_abc_l52_52124

-- Define the real numbers a, b, c
variables {a b c : ℝ}

-- Define the conditions that a, b, c are positive reals.
axiom ha_pos : 0 < a
axiom hb_pos : 0 < b
axiom hc_pos : 0 < c

-- Define the condition that a^2 + b^2 + c^2 = 989
axiom habc_sq : a^2 + b^2 + c^2 = 989

-- Define the condition that (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013
axiom habc_sq_sum : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013

-- The proposition to be proven
theorem find_sum_abc : a + b + c = 32 :=
by
  -- ...(proof goes here)
  sorry

end find_sum_abc_l52_52124


namespace exists_min_a_l52_52919

open Real

theorem exists_min_a (x y z : ℝ) : 
  (∃ x y z : ℝ, (sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) = (11/2 - 1)) ∧ 
  (sqrt (x + 1) + sqrt (y + 1) + sqrt (z + 1) = (11/2 + 1))) :=
sorry

end exists_min_a_l52_52919


namespace part1_part2_l52_52710

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part (1) 
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ a) → -2 ≤ a ∧ a ≤ 1 := by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l52_52710


namespace find_min_value_l52_52615

-- Define a structure to represent vectors in 2D space
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

-- Define the dot product of two vectors
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the condition for perpendicular vectors (dot product is zero)
def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

-- Define the problem: given vectors a = (m, 1) and b = (1, n - 2)
-- with conditions m > 0, n > 0, and a ⊥ b, then prove the minimum value of 1/m + 2/n
theorem find_min_value (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0)
  (h₂ : perpendicular ⟨m, 1⟩ ⟨1, n - 2⟩) :
  (1 / m + 2 / n) = (3 + 2 * Real.sqrt 2) / 2 :=
  sorry

end find_min_value_l52_52615


namespace set_A_range_l52_52530

def A := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ (-1 ≤ x ∧ x ≤ 2)}

theorem set_A_range :
  A = {y : ℝ | -4 ≤ y ∧ y ≤ 0} :=
sorry

end set_A_range_l52_52530


namespace ratio_of_longer_side_to_square_l52_52047

theorem ratio_of_longer_side_to_square (s a b : ℝ) (h1 : a * b = 2 * s^2) (h2 : a = 2 * b) : a / s = 2 :=
by
  sorry

end ratio_of_longer_side_to_square_l52_52047


namespace football_combinations_l52_52959

theorem football_combinations : 
  ∃ (W D L : ℕ), W + D + L = 15 ∧ 3 * W + D = 33 ∧ 
  (9 ≤ W ∧ W ≤ 11) ∧
  (W = 9 → D = 6 ∧ L = 0) ∧
  (W = 10 → D = 3 ∧ L = 2) ∧
  (W = 11 → D = 0 ∧ L = 4) :=
sorry

end football_combinations_l52_52959


namespace arithmetic_sequence_sum_l52_52281

variable {a_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → (n - m) = k → a_n n = a_n m + k * (a_n 1 - a_n 0)

theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a_n →
  a_n 2 = 5 →
  a_n 6 = 33 →
  a_n 3 + a_n 5 = 38 :=
by
  intros h_seq h_a2 h_a6
  sorry

end arithmetic_sequence_sum_l52_52281


namespace consistent_price_per_kg_l52_52856

theorem consistent_price_per_kg (m₁ m₂ : ℝ) (p₁ p₂ : ℝ)
  (h₁ : p₁ = 6) (h₂ : m₁ = 2)
  (h₃ : p₂ = 36) (h₄ : m₂ = 12) :
  (p₁ / m₁ = p₂ / m₂) := 
by 
  sorry

end consistent_price_per_kg_l52_52856


namespace length_of_shortest_side_30_60_90_l52_52440

theorem length_of_shortest_side_30_60_90 (x : ℝ) : 
  (∃ x : ℝ, (2 * x = 15)) → x = 15 / 2 :=
by
  sorry

end length_of_shortest_side_30_60_90_l52_52440


namespace dickens_birth_day_l52_52470

def is_leap_year (year : ℕ) : Prop :=
  (year % 400 = 0) ∨ (year % 4 = 0 ∧ year % 100 ≠ 0)

theorem dickens_birth_day :
  let day_of_week_2012 := 2 -- 0: Sunday, 1: Monday, ..., 2: Tuesday
  let years := 200
  let regular_years := 151
  let leap_years := 49
  let days_shift := regular_years + 2 * leap_years
  let day_of_week_birth := (day_of_week_2012 + days_shift) % 7
  day_of_week_birth = 5 -- 5: Friday
:= 
sorry -- proof not supplied

end dickens_birth_day_l52_52470


namespace student_chose_number_l52_52374

theorem student_chose_number : ∃ x : ℤ, 2 * x - 152 = 102 ∧ x = 127 :=
by
  sorry

end student_chose_number_l52_52374


namespace calculate_discount_percentage_l52_52292

theorem calculate_discount_percentage :
  ∃ (x : ℝ), (∀ (P S : ℝ),
    (S = 439.99999999999966) →
    (S = 1.10 * P) →
    (1.30 * (1 - x / 100) * P = S + 28) →
    x = 10) :=
sorry

end calculate_discount_percentage_l52_52292


namespace product_of_numbers_l52_52721

theorem product_of_numbers (x y z : ℤ) 
  (h1 : x + y + z = 30) 
  (h2 : x = 3 * ((y + z) - 2))
  (h3 : y = 4 * z - 1) : 
  x * y * z = 294 := 
  sorry

end product_of_numbers_l52_52721


namespace extra_yellow_balls_dispatched_l52_52344

theorem extra_yellow_balls_dispatched : 
  ∀ (W Y E : ℕ), -- Declare natural numbers W, Y, E
  W = Y →      -- Condition that the number of white balls equals the number of yellow balls
  W + Y = 64 → -- Condition that the total number of originally ordered balls is 64
  W / (Y + E) = 8 / 13 → -- The given ratio involving the extra yellow balls
  E = 20 :=               -- Prove that the extra yellow balls E equals 20
by
  intros W Y E h1 h2 h3
  -- Proof mechanism here
  sorry

end extra_yellow_balls_dispatched_l52_52344


namespace find_linear_function_l52_52792

theorem find_linear_function (a : ℝ) (a_pos : 0 < a) :
  ∃ (b : ℝ), ∀ (f : ℕ → ℝ),
  (∀ (k m : ℕ), (a * m ≤ k ∧ k < (a + 1) * m) → f (k + m) = f k + f m) →
  ∀ n : ℕ, f n = b * n :=
sorry

end find_linear_function_l52_52792


namespace must_be_true_l52_52780

noncomputable def f (x : ℝ) := |Real.log x|

theorem must_be_true (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) 
                     (h3 : f b < f a) (h4 : f a < f c) :
                     (c > 1) ∧ (1 / c < a) ∧ (a < 1) ∧ (a < b) ∧ (b < 1 / a) :=
by
  sorry

end must_be_true_l52_52780


namespace max_value_of_f_l52_52072

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x - 1/2

theorem max_value_of_f : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ (∀ y, (0 ≤ y ∧ y ≤ 2) → f y ≤ f x) ∧ f x = -3 :=
by
  sorry

end max_value_of_f_l52_52072


namespace union_M_N_equals_set_x_ge_1_l52_52036

-- Definitions of M and N based on the conditions from step a)
def M : Set ℝ := { x | x - 2 > 0 }

def N : Set ℝ := { y | ∃ x : ℝ, y = Real.sqrt (x^2 + 1) }

-- Statement of the theorem
theorem union_M_N_equals_set_x_ge_1 : (M ∪ N) = { x : ℝ | x ≥ 1 } := 
sorry

end union_M_N_equals_set_x_ge_1_l52_52036


namespace necessarily_positive_l52_52056

theorem necessarily_positive (x y z : ℝ) (hx : -1 < x ∧ x < 1) 
                      (hy : -1 < y ∧ y < 0) 
                      (hz : 1 < z ∧ z < 2) : 
    y + z > 0 := 
by
  sorry

end necessarily_positive_l52_52056


namespace f_even_l52_52192

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero : ∃ x : ℝ, f x ≠ 0

axiom f_functional_eqn : ∀ a b : ℝ, 
  f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_even (x : ℝ) : f (-x) = f x :=
  sorry

end f_even_l52_52192


namespace tan_alpha_expression_value_l52_52580

-- (I) Prove that tan(α) = 4/3 under the given conditions
theorem tan_alpha (O A B C P : ℝ × ℝ) (α : ℝ)
  (hO : O = (0, 0))
  (hA : A = (Real.sin α, 1))
  (hB : B = (Real.cos α, 0))
  (hC : C = (-Real.sin α, 2))
  (hP : P = (2 * Real.cos α - Real.sin α, 1))
  (h_collinear : ∃ t : ℝ, C = t • (P.1, P.2)) :
  Real.tan α = 4 / 3 := sorry

-- (II) Prove the given expression under the condition tan(α) = 4/3
theorem expression_value (α : ℝ)
  (h_tan : Real.tan α = 4 / 3) :
  (Real.sin (2 * α) + Real.sin α) / (2 * Real.cos (2 * α) + 2 * Real.sin α * Real.sin α + Real.cos α) + Real.sin (2 * α) = 
  172 / 75 := sorry

end tan_alpha_expression_value_l52_52580


namespace proof_ab_value_l52_52571

theorem proof_ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by
  sorry

end proof_ab_value_l52_52571


namespace P_and_Q_equivalent_l52_52700

def P (x : ℝ) : Prop := 3 * x - x^2 ≤ 0
def Q (x : ℝ) : Prop := |x| ≤ 2
def P_intersection_Q (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

theorem P_and_Q_equivalent : ∀ x, (P x ∧ Q x) ↔ P_intersection_Q x :=
by {
  sorry
}

end P_and_Q_equivalent_l52_52700


namespace average_of_roots_l52_52317

theorem average_of_roots (c : ℝ) (h : ∃ x1 x2 : ℝ, 2 * x1^2 - 6 * x1 + c = 0 ∧ 2 * x2^2 - 6 * x2 + c = 0 ∧ x1 ≠ x2) :
    (∃ p q : ℝ, (2 : ℝ) * (p : ℝ)^2 + (-6 : ℝ) * p + c = 0 ∧ (2 : ℝ) * (q : ℝ)^2 + (-6 : ℝ) * q + c = 0 ∧ p ≠ q) →
    (p + q) / 2 = 3 / 2 := 
sorry

end average_of_roots_l52_52317


namespace pizza_topping_combinations_l52_52799

theorem pizza_topping_combinations :
  (Nat.choose 7 3) = 35 :=
sorry

end pizza_topping_combinations_l52_52799


namespace ronalds_egg_sharing_l52_52135

theorem ronalds_egg_sharing (total_eggs : ℕ) (eggs_per_friend : ℕ) (num_friends : ℕ) 
  (h1 : total_eggs = 16) (h2 : eggs_per_friend = 2) 
  (h3 : num_friends = total_eggs / eggs_per_friend) : 
  num_friends = 8 := 
by 
  sorry

end ronalds_egg_sharing_l52_52135


namespace arithmetic_mean_probability_l52_52909

theorem arithmetic_mean_probability
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : b = (a + c) / 2) :
  b = 1 / 3 :=
by
  sorry

end arithmetic_mean_probability_l52_52909


namespace grade_on_second_test_l52_52037

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end grade_on_second_test_l52_52037


namespace interest_after_4_years_l52_52492
-- Importing the necessary library

-- Definitions based on the conditions
def initial_amount : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def number_of_years : ℕ := 4

-- Calculating the total amount after 4 years using compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Calculating the interest earned
def interest_earned : ℝ :=
  compound_interest initial_amount annual_interest_rate number_of_years - initial_amount

-- The Lean statement to prove the interest earned is $859.25
theorem interest_after_4_years : interest_earned = 859.25 :=
by
  sorry

end interest_after_4_years_l52_52492


namespace right_triangle_lengths_l52_52029

theorem right_triangle_lengths (a b c : ℝ) (h1 : c + b = 2 * a) (h2 : c^2 = a^2 + b^2) : 
  b = 3 / 4 * a ∧ c = 5 / 4 * a := 
by
  sorry

end right_triangle_lengths_l52_52029


namespace length_of_adult_bed_is_20_decimeters_l52_52811

-- Define the length of an adult bed as per question context
def length_of_adult_bed := 20

-- Prove that the length of an adult bed in decimeters equals 20
theorem length_of_adult_bed_is_20_decimeters : length_of_adult_bed = 20 :=
by
  -- Proof goes here
  sorry

end length_of_adult_bed_is_20_decimeters_l52_52811


namespace inequality1_solution_inequality2_solution_l52_52617

-- Definitions for the conditions
def cond1 (x : ℝ) : Prop := abs (1 - (2 * x - 1) / 3) ≤ 2
def cond2 (x : ℝ) : Prop := (2 - x) * (x + 3) < 2 - x

-- Lean 4 statement for the proof problem
theorem inequality1_solution (x : ℝ) : cond1 x → -1 ≤ x ∧ x ≤ 5 := by
  sorry

theorem inequality2_solution (x : ℝ) : cond2 x → x > 2 ∨ x < -2 := by
  sorry

end inequality1_solution_inequality2_solution_l52_52617


namespace system_of_equations_unique_solution_l52_52421

theorem system_of_equations_unique_solution :
  (∃ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7) →
  (∀ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7 →
    x = 26 / 5 ∧ y = 9 / 5) := 
by {
  -- Proof to be provided
  sorry
}

end system_of_equations_unique_solution_l52_52421


namespace heather_biked_per_day_l52_52553

def total_kilometers_biked : ℝ := 320
def days_biked : ℝ := 8
def kilometers_per_day : ℝ := 40

theorem heather_biked_per_day : total_kilometers_biked / days_biked = kilometers_per_day := 
by
  -- Proof will be inserted here
  sorry

end heather_biked_per_day_l52_52553


namespace remainder_8547_div_9_l52_52187

theorem remainder_8547_div_9 : 8547 % 9 = 6 :=
by
  sorry

end remainder_8547_div_9_l52_52187


namespace science_club_election_l52_52704

theorem science_club_election :
  let total_candidates := 20
  let past_officers := 10
  let non_past_officers := total_candidates - past_officers
  let positions := 6
  let total_ways := Nat.choose total_candidates positions
  let no_past_officer_ways := Nat.choose non_past_officers positions
  let exactly_one_past_officer_ways := past_officers * Nat.choose non_past_officers (positions - 1)
  total_ways - no_past_officer_ways - exactly_one_past_officer_ways = 36030 := by
    sorry

end science_club_election_l52_52704


namespace Ben_ate_25_percent_of_cake_l52_52510

theorem Ben_ate_25_percent_of_cake (R B : ℕ) (h_ratio : R / B = 3 / 1) : B / (R + B) * 100 = 25 := by
  sorry

end Ben_ate_25_percent_of_cake_l52_52510


namespace compare_numbers_l52_52711

theorem compare_numbers :
  3 * 10^5 < 2 * 10^6 ∧ -2 - 1 / 3 > -3 - 1 / 2 := by
  sorry

end compare_numbers_l52_52711


namespace a6_minus_b6_divisible_by_9_l52_52682

theorem a6_minus_b6_divisible_by_9 {a b : ℤ} (h₁ : a % 3 ≠ 0) (h₂ : b % 3 ≠ 0) : (a ^ 6 - b ^ 6) % 9 = 0 := 
sorry

end a6_minus_b6_divisible_by_9_l52_52682


namespace value_of_a_l52_52284

theorem value_of_a (a : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → (a * x + 6 ≤ 10)) :
  a = 2 ∨ a = -4 ∨ a = 0 :=
sorry

end value_of_a_l52_52284


namespace angle_between_vectors_with_offset_l52_52839

noncomputable def vector_angle_with_offset : ℝ :=
  let v1 := (4, -1)
  let v2 := (6, 8)
  let dot_product := 4 * 6 + (-1) * 8
  let magnitude_v1 := Real.sqrt (4 ^ 2 + (-1) ^ 2)
  let magnitude_v2 := Real.sqrt (6 ^ 2 + 8 ^ 2)
  let cos_theta := dot_product / (magnitude_v1 * magnitude_v2)
  Real.arccos cos_theta + 30

theorem angle_between_vectors_with_offset :
  vector_angle_with_offset = Real.arccos (8 / (5 * Real.sqrt 17)) + 30 := 
sorry

end angle_between_vectors_with_offset_l52_52839


namespace n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l52_52500

theorem n_to_power_eight_plus_n_to_power_seven_plus_one_prime (n : ℕ) (hn_pos : n > 0) :
  (Nat.Prime (n^8 + n^7 + 1)) → (n = 1) :=
by
  sorry

end n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l52_52500


namespace checkout_speed_ratio_l52_52214

theorem checkout_speed_ratio (n x y : ℝ) 
  (h1 : 40 * x = 20 * y + n)
  (h2 : 36 * x = 12 * y + n) : 
  x = 2 * y := 
sorry

end checkout_speed_ratio_l52_52214


namespace partI_inequality_partII_inequality_l52_52490

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Part (Ⅰ): Prove f(x) ≤ x + 1 for 1 ≤ x ≤ 5
theorem partI_inequality (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) : f x ≤ x + 1 := by
  sorry

-- Part (Ⅱ): Prove (a^2)/(a+1) + (b^2)/(b+1) ≥ 1 when a + b = 2 and a > 0, b > 0
theorem partII_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
    (a^2) / (a + 1) + (b^2) / (b + 1) ≥ 1 := by
  sorry

end partI_inequality_partII_inequality_l52_52490


namespace lars_bakes_for_six_hours_l52_52107

variable (h : ℕ)

-- Conditions
def bakes_loaves : ℕ := 10 * h
def bakes_baguettes : ℕ := 15 * h
def total_breads : ℕ := bakes_loaves h + bakes_baguettes h

-- Proof goal
theorem lars_bakes_for_six_hours (h : ℕ) (H : total_breads h = 150) : h = 6 :=
sorry

end lars_bakes_for_six_hours_l52_52107


namespace intersecting_lines_a_plus_b_l52_52931

theorem intersecting_lines_a_plus_b :
  ∃ (a b : ℝ), (∀ x y : ℝ, (x = 1 / 3 * y + a) ∧ (y = 1 / 3 * x + b) → (x = 3 ∧ y = 4)) ∧ a + b = 14 / 3 :=
sorry

end intersecting_lines_a_plus_b_l52_52931


namespace remaining_watermelons_l52_52306

-- Define the given conditions
def initial_watermelons : ℕ := 35
def watermelons_eaten : ℕ := 27

-- Define the question as a theorem
theorem remaining_watermelons : 
  initial_watermelons - watermelons_eaten = 8 :=
by
  sorry

end remaining_watermelons_l52_52306


namespace find_factor_l52_52791

-- Definitions based on the conditions
def n : ℤ := 155
def result : ℤ := 110
def constant : ℤ := 200

-- Statement to be proved
theorem find_factor (f : ℤ) (h : n * f - constant = result) : f = 2 := by
  sorry

end find_factor_l52_52791


namespace marching_band_members_l52_52467

theorem marching_band_members (B W P : ℕ) (h1 : P = 4 * W) (h2 : W = 2 * B) (h3 : B = 10) : B + W + P = 110 :=
by
  sorry

end marching_band_members_l52_52467


namespace find_principal_l52_52034

theorem find_principal
  (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ)
  (hA : A = 896)
  (hr : r = 0.05)
  (ht : t = 12 / 5) :
  P = 800 ↔ A = P * (1 + r * t) :=
by {
  sorry
}

end find_principal_l52_52034


namespace solve_equation1_solve_equation2_l52_52507

-- Statement for the first equation: x^2 - 16 = 0
theorem solve_equation1 (x : ℝ) : x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Statement for the second equation: (x + 10)^3 + 27 = 0
theorem solve_equation2 (x : ℝ) : (x + 10)^3 + 27 = 0 ↔ x = -13 :=
by sorry

end solve_equation1_solve_equation2_l52_52507


namespace A_roster_method_l52_52517

open Set

def A : Set ℤ := {x : ℤ | (∃ (n : ℤ), n > 0 ∧ 6 / (5 - x) = n) }

theorem A_roster_method :
  A = {-1, 2, 3, 4} :=
  sorry

end A_roster_method_l52_52517


namespace trapezoid_base_count_l52_52060

theorem trapezoid_base_count (A h : ℕ) (multiple : ℕ) (bases_sum pairs_count : ℕ) : 
  A = 1800 ∧ h = 60 ∧ multiple = 10 ∧ pairs_count = 4 ∧ 
  bases_sum = (A / (1/2 * h)) / multiple → pairs_count > 3 := 
by 
  sorry

end trapezoid_base_count_l52_52060


namespace Tile_in_rectangle_R_l52_52786

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def X : Tile := ⟨5, 3, 6, 2⟩
def Y : Tile := ⟨3, 6, 2, 5⟩
def Z : Tile := ⟨6, 0, 1, 5⟩
def W : Tile := ⟨2, 5, 3, 0⟩

theorem Tile_in_rectangle_R : 
  X.top = 5 ∧ X.right = 3 ∧ X.bottom = 6 ∧ X.left = 2 ∧ 
  Y.top = 3 ∧ Y.right = 6 ∧ Y.bottom = 2 ∧ Y.left = 5 ∧ 
  Z.top = 6 ∧ Z.right = 0 ∧ Z.bottom = 1 ∧ Z.left = 5 ∧ 
  W.top = 2 ∧ W.right = 5 ∧ W.bottom = 3 ∧ W.left = 0 → 
  (∀ rectangle_R : Tile, rectangle_R = W) :=
by sorry

end Tile_in_rectangle_R_l52_52786


namespace product_of_roots_l52_52775

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 :=
sorry

end product_of_roots_l52_52775


namespace local_value_of_7_in_diff_l52_52903

-- Definitions based on conditions
def local_value (n : ℕ) (d : ℕ) : ℕ :=
  if h : d < 10 ∧ (n / Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)) % 10 = d then
    d * Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)
  else
    0

def diff (a b : ℕ) : ℕ := a - b

-- Question translated to Lean 4 statement
theorem local_value_of_7_in_diff :
  local_value (diff 100889 (local_value 28943712 3)) 7 = 70000 :=
by sorry

end local_value_of_7_in_diff_l52_52903


namespace pies_sold_each_day_l52_52720

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end pies_sold_each_day_l52_52720


namespace no_such_convex_polyhedron_exists_l52_52932

-- Definitions of convex polyhedron and the properties related to its faces and vertices.
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  -- Additional properties and constraints can be added if necessary

-- Definition that captures the condition where each face has more than 5 sides.
def each_face_has_more_than_five_sides (P : ConvexPolyhedron) : Prop :=
  ∀ f, f > 5 -- Simplified assumption

-- Definition that captures the condition where more than five edges meet at each vertex.
def more_than_five_edges_meet_each_vertex (P : ConvexPolyhedron) : Prop :=
  ∀ v, v > 5 -- Simplified assumption

-- The statement to be proven
theorem no_such_convex_polyhedron_exists :
  ¬ ∃ (P : ConvexPolyhedron), (each_face_has_more_than_five_sides P) ∨ (more_than_five_edges_meet_each_vertex P) := by
  -- Proof of this theorem is omitted with "sorry"
  sorry

end no_such_convex_polyhedron_exists_l52_52932


namespace line_intersect_yaxis_at_l52_52912

theorem line_intersect_yaxis_at
  (x1 y1 x2 y2 : ℝ) : (x1 = 3) → (y1 = 19) → (x2 = -7) → (y2 = -1) →
  ∃ y : ℝ, (0, y) = (0, 13) :=
by
  intros h1 h2 h3 h4
  sorry

end line_intersect_yaxis_at_l52_52912


namespace m_value_for_positive_root_eq_l52_52899

-- We start by defining the problem:
-- Given the condition that the equation (3x - 1)/(x + 1) - m/(x + 1) = 1 has a positive root,
-- we need to prove that m = -4.

theorem m_value_for_positive_root_eq (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 :=
by
  sorry

end m_value_for_positive_root_eq_l52_52899


namespace circle_radius_l52_52033

theorem circle_radius (D : ℝ) (h : D = 14) : (D / 2) = 7 :=
by
  sorry

end circle_radius_l52_52033


namespace divide_value_l52_52766

def divide (a b c : ℝ) : ℝ := |b^2 - 5 * a * c|

theorem divide_value : divide 2 (-3) 1 = 1 :=
by
  sorry

end divide_value_l52_52766


namespace deepak_present_age_l52_52978

theorem deepak_present_age (x : ℕ) (rahul deepak rohan : ℕ) 
  (h_ratio : rahul = 5 * x ∧ deepak = 2 * x ∧ rohan = 3 * x)
  (h_rahul_future_age : rahul + 8 = 28) :
  deepak = 8 := 
by
  sorry

end deepak_present_age_l52_52978


namespace triangle_height_l52_52747

theorem triangle_height (area base : ℝ) (h : ℝ) (h_area : area = 46) (h_base : base = 10) 
  (h_formula : area = (base * h) / 2) : 
  h = 9.2 :=
by
  sorry

end triangle_height_l52_52747


namespace sum_of_money_invested_l52_52967

noncomputable def principal_sum_of_money (R : ℝ) (T : ℝ) (CI_minus_SI : ℝ) : ℝ :=
  let SI := (625 * R * T / 100)
  let CI := 625 * ((1 + R / 100)^(T : ℝ) - 1)
  if (CI - SI = CI_minus_SI)
  then 625
  else 0

theorem sum_of_money_invested : 
  (principal_sum_of_money 4 2 1) = 625 :=
by
  unfold principal_sum_of_money
  sorry

end sum_of_money_invested_l52_52967


namespace school_population_l52_52690

variable (b g t a : ℕ)

theorem school_population (h1 : b = 2 * g) (h2 : g = 4 * t) (h3 : a = t / 2) : 
  b + g + t + a = 27 * b / 16 := by
  sorry

end school_population_l52_52690


namespace incorrect_judgment_l52_52442

variable (p q : Prop)
variable (hyp_p : p = (3 + 3 = 5))
variable (hyp_q : q = (5 > 2))

theorem incorrect_judgment : 
  (¬ (p ∧ q) ∧ ¬p) = false :=
by
  sorry

end incorrect_judgment_l52_52442


namespace sequence_non_positive_l52_52198

theorem sequence_non_positive
  (a : ℕ → ℝ) (n : ℕ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) :
  ∀ k, k ≤ n → a k ≤ 0 := 
sorry

end sequence_non_positive_l52_52198


namespace displacement_during_interval_l52_52519

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem displacement_during_interval :
  (∫ t in (0 : ℝ)..3, velocity t) = 36 :=
by
  sorry

end displacement_during_interval_l52_52519


namespace common_difference_is_4_l52_52483

variable (a : ℕ → ℤ) (d : ℤ)

-- Conditions of the problem
def arithmetic_sequence := ∀ n m : ℕ, a n = a m + (n - m) * d

axiom a7_eq_25 : a 7 = 25
axiom a4_eq_13 : a 4 = 13

-- The theorem to prove
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l52_52483


namespace velocity_zero_times_l52_52577

noncomputable def s (t : ℝ) : ℝ := (1 / 4) * t^4 - (5 / 3) * t^3 + 2 * t^2

theorem velocity_zero_times :
  {t : ℝ | deriv s t = 0} = {0, 1, 4} :=
by 
  sorry

end velocity_zero_times_l52_52577


namespace apples_remain_correct_l52_52825

def total_apples : ℕ := 15
def apples_eaten : ℕ := 7
def apples_remaining : ℕ := total_apples - apples_eaten

theorem apples_remain_correct : apples_remaining = 8 :=
by
  -- Initial number of apples
  let total := total_apples
  -- Number of apples eaten
  let eaten := apples_eaten
  -- Remaining apples
  let remain := total - eaten
  -- Assertion
  have h : remain = 8 := by
      sorry
  exact h

end apples_remain_correct_l52_52825


namespace find_erased_number_l52_52923

/-- Define the variables used in the conditions -/
def n : ℕ := 69
def erased_number_mean : ℚ := 35 + 7 / 17
def sequence_sum : ℕ := n * (n + 1) / 2

/-- State the condition for the erased number -/
noncomputable def erased_number (x : ℕ) : Prop :=
  (sequence_sum - x) / (n - 1) = erased_number_mean

/-- The main theorem stating that the erased number is 7 -/
theorem find_erased_number : ∃ x : ℕ, erased_number x ∧ x = 7 :=
by
  use 7
  unfold erased_number sequence_sum
  -- Sum of first 69 natural numbers is 69 * (69 + 1) / 2
  -- Hence,
  -- (69 * 70 / 2 - 7) / 68 = 35 + 7 / 17
  -- which simplifies to true under these conditions
  -- Detailed proof skipped here as per instructions
  sorry

end find_erased_number_l52_52923


namespace quadrilateral_centroid_perimeter_l52_52506

-- Definition for the side length of the square and distances for points Q
def side_length : ℝ := 40
def EQ_dist : ℝ := 18
def FQ_dist : ℝ := 34

-- Theorem statement: Perimeter of the quadrilateral formed by centroids
theorem quadrilateral_centroid_perimeter :
  let centroid_perimeter := (4 * ((2 / 3) * side_length))
  centroid_perimeter = (320 / 3) := by
  sorry

end quadrilateral_centroid_perimeter_l52_52506


namespace mathematicians_correctness_l52_52597

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l52_52597


namespace matrix_projection_ratios_l52_52170

theorem matrix_projection_ratios (x y z : ℚ) (h : 
  (1 / 14 : ℚ) * x - (5 / 14 : ℚ) * y = x ∧
  - (5 / 14 : ℚ) * x + (24 / 14 : ℚ) * y = y ∧
  0 * x + 0 * y + 1 * z = z)
  : y / x = 13 / 5 ∧ z / x = 1 := 
by 
  sorry

end matrix_projection_ratios_l52_52170


namespace fourth_place_points_l52_52372

variables (x : ℕ)

def points_awarded (place : ℕ) : ℕ :=
  if place = 1 then 11
  else if place = 2 then 7
  else if place = 3 then 5
  else if place = 4 then x
  else 0

theorem fourth_place_points:
  (∃ a b c y u : ℕ, a + b + c + y + u = 7 ∧ points_awarded x 1 ^ a * points_awarded x 2 ^ b * points_awarded x 3 ^ c * points_awarded x 4 ^ y * 1 ^ u = 38500) →
  x = 4 :=
sorry

end fourth_place_points_l52_52372


namespace new_salary_after_increase_l52_52200

theorem new_salary_after_increase : 
  ∀ (previous_salary : ℝ) (percentage_increase : ℝ), 
    previous_salary = 2000 → percentage_increase = 0.05 → 
    previous_salary + (previous_salary * percentage_increase) = 2100 :=
by
  intros previous_salary percentage_increase h1 h2
  sorry

end new_salary_after_increase_l52_52200


namespace find_polynomial_coefficients_l52_52143

-- Define the quadratic polynomial q(x) = ax^2 + bx + c
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions for polynomial
axiom condition1 (a b c : ℝ) : polynomial a b c (-2) = 9
axiom condition2 (a b c : ℝ) : polynomial a b c 1 = 2
axiom condition3 (a b c : ℝ) : polynomial a b c 3 = 10

-- Conjecture for the polynomial q(x)
theorem find_polynomial_coefficients : 
  ∃ (a b c : ℝ), 
    polynomial a b c (-2) = 9 ∧
    polynomial a b c 1 = 2 ∧
    polynomial a b c 3 = 10 ∧
    a = 19 / 15 ∧
    b = -2 / 15 ∧
    c = 13 / 15 :=
by {
  -- Placeholder proof
  sorry
}

end find_polynomial_coefficients_l52_52143


namespace find_pairs_l52_52026

theorem find_pairs (m n: ℕ) (h: m > 0 ∧ n > 0 ∧ m + n - (3 * m * n) / (m + n) = 2011 / 3) : (m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144) :=
by sorry

end find_pairs_l52_52026


namespace maximal_difference_of_areas_l52_52495

-- Given:
-- A circle of radius R
-- A chord of length 2x is drawn perpendicular to the diameter of the circle
-- The endpoints of this chord are connected to the endpoints of the diameter
-- We need to prove that under these conditions, the length of the chord 2x that maximizes the difference in areas of the triangles is R √ 2

theorem maximal_difference_of_areas (R x : ℝ) (h : 2 * x = R * Real.sqrt 2) :
  2 * x = R * Real.sqrt 2 :=
by
  sorry

end maximal_difference_of_areas_l52_52495


namespace chinese_chess_sets_l52_52776

theorem chinese_chess_sets (x y : ℕ) 
  (h1 : 24 * x + 18 * y = 300) 
  (h2 : x + y = 14) : 
  y = 6 := 
sorry

end chinese_chess_sets_l52_52776


namespace increase_speed_to_pass_correctly_l52_52462

theorem increase_speed_to_pass_correctly
  (x a : ℝ)
  (ha1 : 50 < a)
  (hx1 : (a - 40) * x = 30)
  (hx2 : (a + 50) * x = 210) :
  a - 50 = 5 :=
by
  sorry

end increase_speed_to_pass_correctly_l52_52462


namespace α_eq_β_plus_two_l52_52054

-- Definitions based on the given conditions:
-- α(n): number of ways n can be expressed as a sum of the integers 1 and 2, considering different orders as distinct ways.
-- β(n): number of ways n can be expressed as a sum of integers greater than 1, considering different orders as distinct ways.

def α (n : ℕ) : ℕ := sorry
def β (n : ℕ) : ℕ := sorry

-- The proof statement that needs to be proved.
theorem α_eq_β_plus_two (n : ℕ) (h : 0 < n) : α n = β (n + 2) := 
  sorry

end α_eq_β_plus_two_l52_52054


namespace dorothy_money_left_l52_52537

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end dorothy_money_left_l52_52537


namespace mr_klinker_twice_as_old_l52_52588

theorem mr_klinker_twice_as_old (x : ℕ) (current_age_klinker : ℕ) (current_age_daughter : ℕ)
  (h1 : current_age_klinker = 35) (h2 : current_age_daughter = 10) 
  (h3 : current_age_klinker + x = 2 * (current_age_daughter + x)) : 
  x = 15 :=
by 
  -- We include sorry to indicate where the proof should be
  sorry

end mr_klinker_twice_as_old_l52_52588


namespace overall_percentage_change_is_113_point_4_l52_52605

-- Define the conditions
def total_customers_survey_1 := 100
def male_percentage_survey_1 := 60
def respondents_survey_1 := 10
def male_respondents_survey_1 := 5

def total_customers_survey_2 := 80
def male_percentage_survey_2 := 70
def respondents_survey_2 := 16
def male_respondents_survey_2 := 12

def total_customers_survey_3 := 70
def male_percentage_survey_3 := 40
def respondents_survey_3 := 21
def male_respondents_survey_3 := 13

def total_customers_survey_4 := 90
def male_percentage_survey_4 := 50
def respondents_survey_4 := 27
def male_respondents_survey_4 := 8

-- Define the calculated response rates
def original_male_response_rate := (male_respondents_survey_1.toFloat / (total_customers_survey_1 * male_percentage_survey_1 / 100).toFloat) * 100
def final_male_response_rate := (male_respondents_survey_4.toFloat / (total_customers_survey_4 * male_percentage_survey_4 / 100).toFloat) * 100

-- Calculate the percentage change in response rate
def percentage_change := ((final_male_response_rate - original_male_response_rate) / original_male_response_rate) * 100

-- The target theorem 
theorem overall_percentage_change_is_113_point_4 : percentage_change = 113.4 := sorry

end overall_percentage_change_is_113_point_4_l52_52605


namespace brother_to_madeline_ratio_l52_52739

theorem brother_to_madeline_ratio (M B T : ℕ) (hM : M = 48) (hT : T = 72) (hSum : M + B = T) : B / M = 1 / 2 := by
  sorry

end brother_to_madeline_ratio_l52_52739


namespace percentage_less_than_l52_52667

theorem percentage_less_than (x y : ℝ) (P : ℝ) (h1 : y = 1.6667 * x) (h2 : x = (1 - P / 100) * y) : P = 66.67 :=
sorry

end percentage_less_than_l52_52667


namespace angle_405_eq_45_l52_52600

def same_terminal_side (angle1 angle2 : ℝ) : Prop :=
  ∃ k : ℤ, angle1 = angle2 + k * 360

theorem angle_405_eq_45 (k : ℤ) : same_terminal_side 405 45 := 
sorry

end angle_405_eq_45_l52_52600


namespace relationship_between_roots_l52_52061

-- Define the number of real roots of the equations
def number_real_roots_lg_eq_sin : ℕ := 3
def number_real_roots_x_eq_sin : ℕ := 1
def number_real_roots_x4_eq_sin : ℕ := 2

-- Define the variables
def a : ℕ := number_real_roots_lg_eq_sin
def b : ℕ := number_real_roots_x_eq_sin
def c : ℕ := number_real_roots_x4_eq_sin

-- State the theorem
theorem relationship_between_roots : a > c ∧ c > b :=
by
  -- the proof is skipped
  sorry

end relationship_between_roots_l52_52061


namespace shells_needed_l52_52146

theorem shells_needed (current_shells : ℕ) (total_shells : ℕ) (difference : ℕ) :
  current_shells = 5 → total_shells = 17 → difference = total_shells - current_shells → difference = 12 :=
by
  intros h1 h2 h3
  sorry

end shells_needed_l52_52146


namespace valid_three_digit_numbers_count_l52_52983

def count_three_digit_numbers : ℕ := 900

def count_invalid_numbers : ℕ := (90 + 90 - 9)

def count_valid_three_digit_numbers : ℕ := 900 - (90 + 90 - 9)

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 729 :=
by
  show 900 - (90 + 90 - 9) = 729
  sorry

end valid_three_digit_numbers_count_l52_52983


namespace sum_of_numbers_l52_52917

-- Define the given conditions.
def S : ℕ := 30
def F : ℕ := 2 * S
def T : ℕ := F / 3

-- State the proof problem.
theorem sum_of_numbers : F + S + T = 110 :=
by
  -- Assume the proof here.
  sorry

end sum_of_numbers_l52_52917


namespace Holly_throws_5_times_l52_52892

def Bess.throw_distance := 20
def Bess.throw_times := 4
def Holly.throw_distance := 8
def total_distance := 200

theorem Holly_throws_5_times : 
  (total_distance - Bess.throw_times * 2 * Bess.throw_distance) / Holly.throw_distance = 5 :=
by 
  sorry

end Holly_throws_5_times_l52_52892


namespace radius_of_cone_l52_52059

theorem radius_of_cone (A : ℝ) (g : ℝ) (R : ℝ) (hA : A = 15 * Real.pi) (hg : g = 5) : R = 3 :=
sorry

end radius_of_cone_l52_52059


namespace stephanie_bills_l52_52042

theorem stephanie_bills :
  let electricity_bill := 120
  let electricity_paid := 0.80 * electricity_bill
  let gas_bill := 80
  let gas_paid := (3 / 4) * gas_bill
  let additional_gas_payment := 10
  let water_bill := 60
  let water_paid := 0.65 * water_bill
  let internet_bill := 50
  let internet_paid := 6 * 5
  let internet_remaining_before_discount := internet_bill - internet_paid
  let internet_discount := 0.10 * internet_remaining_before_discount
  let phone_bill := 45
  let phone_paid := 0.20 * phone_bill
  let remaining_electricity := electricity_bill - electricity_paid
  let remaining_gas := gas_bill - (gas_paid + additional_gas_payment)
  let remaining_water := water_bill - water_paid
  let remaining_internet := internet_remaining_before_discount - internet_discount
  let remaining_phone := phone_bill - phone_paid
  (remaining_electricity + remaining_gas + remaining_water + remaining_internet + remaining_phone) = 109 :=
by
  sorry

end stephanie_bills_l52_52042


namespace polynomial_remainder_l52_52324

theorem polynomial_remainder :
  let f := X^2023 + 1
  let g := X^6 - X^4 + X^2 - 1
  ∃ (r : Polynomial ℤ), (r = -X^3 + 1) ∧ (∃ q : Polynomial ℤ, f = q * g + r) :=
by
  sorry

end polynomial_remainder_l52_52324


namespace third_candidate_votes_l52_52394

theorem third_candidate_votes
  (total_votes : ℝ)
  (votes_for_two_candidates : ℝ)
  (winning_percentage : ℝ)
  (H1 : votes_for_two_candidates = 4636 + 11628)
  (H2 : winning_percentage = 67.21387283236994 / 100)
  (H3 : total_votes = votes_for_two_candidates / (1 - winning_percentage)) :
  (total_votes - votes_for_two_candidates) = 33336 :=
by
  sorry

end third_candidate_votes_l52_52394


namespace problem_l52_52564

def f (x a b : ℝ) : ℝ := a * x ^ 3 - b * x + 1

theorem problem (a b : ℝ) (h : f 2 a b = -1) : f (-2) a b = 3 :=
by {
  sorry
}

end problem_l52_52564


namespace total_number_of_orders_l52_52289

-- Define the conditions
def num_original_programs : Nat := 6
def num_added_programs : Nat := 3

-- State the theorem
theorem total_number_of_orders : ∃ n : ℕ, n = 210 :=
by
  -- This is where the proof would go
  sorry

end total_number_of_orders_l52_52289


namespace rex_cards_remaining_l52_52757

theorem rex_cards_remaining
  (nicole_cards : ℕ)
  (cindy_cards : ℕ)
  (rex_cards : ℕ)
  (cards_per_person : ℕ)
  (h1 : nicole_cards = 400)
  (h2 : cindy_cards = 2 * nicole_cards)
  (h3 : rex_cards = (nicole_cards + cindy_cards) / 2)
  (h4 : cards_per_person = rex_cards / 4) :
  cards_per_person = 150 :=
by
  sorry

end rex_cards_remaining_l52_52757


namespace infinitely_many_n_squared_plus_one_no_special_divisor_l52_52726

theorem infinitely_many_n_squared_plus_one_no_special_divisor :
  ∃ (f : ℕ → ℕ), (∀ n, f n ≠ 0) ∧ ∀ n, ∀ k, f n^2 + 1 ≠ k^2 + 1 ∨ k^2 + 1 = 1 :=
by
  sorry

end infinitely_many_n_squared_plus_one_no_special_divisor_l52_52726


namespace g_at_3_eq_19_l52_52789

def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem g_at_3_eq_19 : g 3 = 19 := by
  sorry

end g_at_3_eq_19_l52_52789


namespace jim_gold_per_hour_l52_52181

theorem jim_gold_per_hour :
  ∀ (hours: ℕ) (treasure_chest: ℕ) (num_small_bags: ℕ)
    (each_small_bag_has: ℕ),
    hours = 8 →
    treasure_chest = 100 →
    num_small_bags = 2 →
    each_small_bag_has = (treasure_chest / 2) →
    (treasure_chest + num_small_bags * each_small_bag_has) / hours = 25 :=
by
  intros hours treasure_chest num_small_bags each_small_bag_has
  intros hours_eq treasure_chest_eq num_small_bags_eq small_bag_eq
  have total_gold : ℕ := treasure_chest + num_small_bags * each_small_bag_has
  have per_hour : ℕ := total_gold / hours
  sorry

end jim_gold_per_hour_l52_52181


namespace hansel_album_duration_l52_52975

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_l52_52975


namespace problem_solution_l52_52487

-- Define the problem
noncomputable def a_b_sum : ℝ := 
  let a := 5
  let b := 3
  a + b

-- Theorem statement
theorem problem_solution (a b i : ℝ) (h1 : a + b * i = (11 - 7 * i) / (1 - 2 * i)) (hi : i * i = -1) :
  a + b = 8 :=
by sorry

end problem_solution_l52_52487


namespace eight_and_five_l52_52758

def my_and (a b : ℕ) : ℕ := (a + b) ^ 2 * (a - b)

theorem eight_and_five : my_and 8 5 = 507 := 
  by sorry

end eight_and_five_l52_52758


namespace largest_value_of_x_l52_52348

theorem largest_value_of_x : 
  ∃ x, ( (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ) ∧ x = (19 + Real.sqrt 229) / 22 :=
sorry

end largest_value_of_x_l52_52348


namespace max_quarters_in_wallet_l52_52175

theorem max_quarters_in_wallet:
  ∃ (q n : ℕ), 
    (30 * n) + 50 = 31 * (n + 1) ∧ 
    q = 22 :=
by
  sorry

end max_quarters_in_wallet_l52_52175


namespace probability_page_multiple_of_7_l52_52845

theorem probability_page_multiple_of_7 (total_pages : ℕ) (probability : ℚ)
  (h_total_pages : total_pages = 500) 
  (h_probability : probability = 71 / 500) :
  probability = 0.142 := 
sorry

end probability_page_multiple_of_7_l52_52845


namespace solve_for_x_l52_52116

theorem solve_for_x (x : ℝ) (hx_pos : x > 0) (h_eq : 3 * x^2 + 13 * x - 10 = 0) : x = 2 / 3 :=
sorry

end solve_for_x_l52_52116


namespace slope_of_line_passes_through_points_l52_52382

theorem slope_of_line_passes_through_points :
  let k := (2 + Real.sqrt 3 - 2) / (4 - 1)
  k = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_passes_through_points_l52_52382


namespace limit_r_l52_52457

noncomputable def L (m : ℝ) : ℝ := (m - Real.sqrt (m^2 + 24)) / 2

noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem limit_r (h : ∀ m : ℝ, m ≠ 0) : Filter.Tendsto r (nhds 0) (nhds (-1)) :=
sorry

end limit_r_l52_52457


namespace gcd_1260_924_l52_52957

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 :=
by
  sorry

end gcd_1260_924_l52_52957


namespace exists_divisor_between_l52_52685

theorem exists_divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) 
  (h_div1 : a ∣ n) (h_div2 : b ∣ n) (h_neq : a ≠ b) 
  (h_lt : a < b) (h_eq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end exists_divisor_between_l52_52685


namespace sum_of_arith_seq_l52_52684

noncomputable def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arith_seq (a : ℕ → ℝ) (h_a : is_arith_seq a)
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 21 :=
sorry

end sum_of_arith_seq_l52_52684


namespace positive_slope_asymptote_l52_52731

-- Define the foci points A and B and the given equation of the hyperbola
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-3, 1)
def hyperbola_eqn (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y - 1)^2) - Real.sqrt ((x + 3)^2 + (y - 1)^2) = 4

-- State the theorem about the positive slope of the asymptote
theorem positive_slope_asymptote (x y : ℝ) (h : hyperbola_eqn x y) : 
  ∃ b a : ℝ, b = Real.sqrt 5 ∧ a = 2 ∧ (b / a) = Real.sqrt 5 / 2 :=
by
  sorry

end positive_slope_asymptote_l52_52731


namespace yellow_balls_count_l52_52563

theorem yellow_balls_count (r y : ℕ) (h1 : r = 9) (h2 : (r : ℚ) / (r + y) = 1 / 3) : y = 18 := 
by
  sorry

end yellow_balls_count_l52_52563


namespace books_checked_out_on_Thursday_l52_52806

theorem books_checked_out_on_Thursday (initial_books : ℕ) (wednesday_checked_out : ℕ) 
                                      (thursday_returned : ℕ) (friday_returned : ℕ) (final_books : ℕ) 
                                      (thursday_checked_out : ℕ) : 
  (initial_books = 98) → 
  (wednesday_checked_out = 43) → 
  (thursday_returned = 23) → 
  (friday_returned = 7) → 
  (final_books = 80) → 
  (initial_books - wednesday_checked_out + thursday_returned - thursday_checked_out + friday_returned = final_books) → 
  (thursday_checked_out = 5) :=
by
  intros
  sorry

end books_checked_out_on_Thursday_l52_52806
