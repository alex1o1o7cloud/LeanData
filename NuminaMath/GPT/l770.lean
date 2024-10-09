import Mathlib

namespace minimum_value_of_f_l770_77056

noncomputable def f (x y z : ℝ) : ℝ := (1 / (x + y)) + (1 / (x + z)) + (1 / (y + z)) - (x * y * z)

theorem minimum_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → f x y z = 1 / 2 :=
by
  sorry

end minimum_value_of_f_l770_77056


namespace total_books_count_l770_77036

theorem total_books_count (books_read : ℕ) (books_unread : ℕ) (h1 : books_read = 13) (h2 : books_unread = 8) : books_read + books_unread = 21 := 
by
  -- Proof omitted
  sorry

end total_books_count_l770_77036


namespace mooncake_packaging_problem_l770_77066

theorem mooncake_packaging_problem
  (x y : ℕ)
  (L : ℕ := 9)
  (S : ℕ := 4)
  (M : ℕ := 35)
  (h1 : L = 9)
  (h2 : S = 4)
  (h3 : M = 35) :
  9 * x + 4 * y = 35 ∧ x + y = 5 := 
by
  sorry

end mooncake_packaging_problem_l770_77066


namespace complex_fraction_expression_equals_half_l770_77037

theorem complex_fraction_expression_equals_half :
  ((2 / (3 + 1/5)) + (((3 + 1/4) / 13) / (2 / 3)) + (((2 + 5/18) - (17/36)) * (18 / 65))) * (1 / 3) = 0.5 :=
by
  sorry

end complex_fraction_expression_equals_half_l770_77037


namespace wendy_furniture_time_l770_77060

theorem wendy_furniture_time (chairs tables minutes_per_piece : ℕ) 
    (h_chairs : chairs = 4) 
    (h_tables : tables = 4) 
    (h_minutes_per_piece : minutes_per_piece = 6) : 
    chairs + tables * minutes_per_piece = 48 := 
by 
    sorry

end wendy_furniture_time_l770_77060


namespace symmetric_circle_l770_77003

variable (x y : ℝ)

def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem symmetric_circle :
  (∃ x y, original_circle x y) → (x^2 + (y + 2)^2 = 5) :=
sorry

end symmetric_circle_l770_77003


namespace compute_value_l770_77041

noncomputable def repeating_decimal_31 : ℝ := 31 / 100000
noncomputable def repeating_decimal_47 : ℝ := 47 / 100000
def term : ℝ := 10^5 - 10^3

theorem compute_value : (term * repeating_decimal_31 + term * repeating_decimal_47) = 77.22 := 
by
  sorry

end compute_value_l770_77041


namespace parabola_equation_l770_77015

theorem parabola_equation (h k : ℝ) (p : ℝ × ℝ) (a b c : ℝ) :
  h = 3 ∧ k = -2 ∧ p = (4, -5) ∧
  (∀ x y : ℝ, y = a * (x - h) ^ 2 + k → p.2 = a * (p.1 - h) ^ 2 + k) →
  -(3:ℝ) = a ∧ 18 = b ∧ -29 = c :=
by sorry

end parabola_equation_l770_77015


namespace binomial_sum_of_coefficients_l770_77087

theorem binomial_sum_of_coefficients (n : ℕ) (h₀ : (1 - 2)^n = 8) :
  (1 - 2)^n = -1 :=
sorry

end binomial_sum_of_coefficients_l770_77087


namespace find_x_if_alpha_beta_eq_4_l770_77020

def alpha (x : ℝ) : ℝ := 4 * x + 9
def beta (x : ℝ) : ℝ := 9 * x + 6

theorem find_x_if_alpha_beta_eq_4 :
  (∃ x : ℝ, alpha (beta x) = 4 ∧ x = -29 / 36) :=
by
  sorry

end find_x_if_alpha_beta_eq_4_l770_77020


namespace lemons_per_glass_l770_77051

theorem lemons_per_glass (lemons glasses : ℕ) (h : lemons = 18 ∧ glasses = 9) : lemons / glasses = 2 :=
by
  sorry

end lemons_per_glass_l770_77051


namespace simplified_expression_is_one_l770_77067

-- Define the specific mathematical expressions
def expr1 := -1 ^ 2023
def expr2 := (-2) ^ 3
def expr3 := (-2) * (-3)

-- Construct the full expression
def full_expr := expr1 - expr2 - expr3

-- State the theorem that this full expression equals 1
theorem simplified_expression_is_one : full_expr = 1 := by
  sorry

end simplified_expression_is_one_l770_77067


namespace sin_45_eq_sqrt2_div_2_l770_77052

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = (Real.sqrt 2) / 2 := 
  sorry

end sin_45_eq_sqrt2_div_2_l770_77052


namespace loan_amount_l770_77047

theorem loan_amount (R T SI : ℕ) (hR : R = 7) (hT : T = 7) (hSI : SI = 735) : 
  ∃ P : ℕ, P = 1500 := 
by 
  sorry

end loan_amount_l770_77047


namespace housewife_spent_on_oil_l770_77074

-- Define the conditions
variables (P A : ℝ)
variables (h_price_reduced : 0.7 * P = 70)
variables (h_more_oil : A / 70 = A / P + 3)

-- Define the theorem to be proven
theorem housewife_spent_on_oil : A = 700 :=
by
  sorry

end housewife_spent_on_oil_l770_77074


namespace sum_of_first_45_natural_numbers_l770_77099

theorem sum_of_first_45_natural_numbers : (45 * (45 + 1)) / 2 = 1035 := by
  sorry

end sum_of_first_45_natural_numbers_l770_77099


namespace final_share_approx_equal_l770_77094

noncomputable def total_bill : ℝ := 211.0
noncomputable def number_of_people : ℝ := 6.0
noncomputable def tip_percentage : ℝ := 0.15
noncomputable def tip_amount : ℝ := tip_percentage * total_bill
noncomputable def total_amount : ℝ := total_bill + tip_amount
noncomputable def each_person_share : ℝ := total_amount / number_of_people

theorem final_share_approx_equal :
  abs (each_person_share - 40.44) < 0.01 :=
by
  sorry

end final_share_approx_equal_l770_77094


namespace solve_equation_l770_77035

theorem solve_equation (x : ℝ) (h1 : x + 1 ≠ 0) (h2 : 2 * x - 1 ≠ 0) :
  (2 / (x + 1) = 3 / (2 * x - 1)) ↔ (x = 5) := 
sorry

end solve_equation_l770_77035


namespace exists_prime_q_not_div_n_p_minus_p_l770_77053

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem exists_prime_q_not_div_n_p_minus_p :
  ∃ q : ℕ, Nat.Prime q ∧ q ≠ p ∧ ∀ n : ℕ, ¬ q ∣ (n ^ p - p) :=
sorry

end exists_prime_q_not_div_n_p_minus_p_l770_77053


namespace divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l770_77024

open Nat

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (factors n).eraseDups.length

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

variables {p q m n : ℕ}
variables (hp : is_prime p) (hq : is_prime q) (hdist : p ≠ q) (hm : 0 ≤ m) (hn : 0 ≤ n)

-- a) Prove the number of divisors of pq is 4
theorem divisors_pq : num_divisors (p * q) = 4 :=
sorry

-- b) Prove the number of divisors of p^2 q is 6
theorem divisors_p2q : num_divisors (p^2 * q) = 6 :=
sorry

-- c) Prove the number of divisors of p^2 q^2 is 9
theorem divisors_p2q2 : num_divisors (p^2 * q^2) = 9 :=
sorry

-- d) Prove the number of divisors of p^m q^n is (m + 1)(n + 1)
theorem divisors_pmqn : num_divisors (p^m * q^n) = (m + 1) * (n + 1) :=
sorry

end divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l770_77024


namespace longer_side_of_rectangle_l770_77011

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l770_77011


namespace player_A_min_score_l770_77068

theorem player_A_min_score (A B : ℕ) (hA_first_move : A = 1) (hB_next_move : B = 2) : 
  ∃ k : ℕ, k = 64 :=
by
  sorry

end player_A_min_score_l770_77068


namespace math_problem_l770_77005

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0):
  ((2^(2^n) + 1) * (2^(2^m) + 1)) % (m * n) = 0 →
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) :=
by
  sorry

end math_problem_l770_77005


namespace ad_lt_bc_l770_77055

theorem ad_lt_bc (a b c d : ℝ ) (h1a : a > 0) (h1b : b > 0) (h1c : c > 0) (h1d : d > 0)
  (h2 : a + d = b + c) (h3 : |a - d| < |b - c|) : a * d < b * c :=
  sorry

end ad_lt_bc_l770_77055


namespace cos_alpha_value_l770_77071

noncomputable def cos_alpha (α : ℝ) : ℝ :=
  (3 - 4 * Real.sqrt 3) / 10

theorem cos_alpha_value (α : ℝ) (h1 : Real.sin (Real.pi / 6 + α) = 3 / 5) (h2 : Real.pi / 3 < α ∧ α < 5 * Real.pi / 6) :
  Real.cos α = cos_alpha α :=
by
sorry

end cos_alpha_value_l770_77071


namespace linear_dependent_vectors_l770_77008

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l770_77008


namespace maximum_daily_sales_revenue_l770_77032

noncomputable def P (t : ℕ) : ℤ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

noncomputable def Q (t : ℕ) : ℤ :=
  if 0 < t ∧ t ≤ 30 then -t + 40 else 0

noncomputable def y (t : ℕ) : ℤ := P t * Q t

theorem maximum_daily_sales_revenue : 
  ∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ y t = 1125 :=
by
  sorry

end maximum_daily_sales_revenue_l770_77032


namespace stratified_sampling_elderly_employees_l770_77073

-- Definitions for the conditions
def total_employees : ℕ := 430
def young_employees : ℕ := 160
def middle_aged_employees : ℕ := 180
def elderly_employees : ℕ := 90
def sample_young_employees : ℕ := 32

-- The property we want to prove
theorem stratified_sampling_elderly_employees :
  (sample_young_employees / young_employees) * elderly_employees = 18 :=
by
  sorry

end stratified_sampling_elderly_employees_l770_77073


namespace tax_diminished_percentage_l770_77098

theorem tax_diminished_percentage (T C : ℝ) (x : ℝ) (h : (T * (1 - x / 100)) * (C * 1.10) = T * C * 0.88) :
  x = 20 :=
sorry

end tax_diminished_percentage_l770_77098


namespace triangle_right_angle_l770_77078

theorem triangle_right_angle
  (a b m : ℝ)
  (h1 : 0 < b)
  (h2 : b < m)
  (h3 : a^2 + b^2 = m^2) :
  a^2 + b^2 = m^2 :=
by sorry

end triangle_right_angle_l770_77078


namespace modulus_remainder_l770_77014

namespace Proof

def a (n : ℕ) : ℕ := 88134 + n

theorem modulus_remainder :
  (2 * ((a 0)^2 + (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2)) % 11 = 3 := by
  sorry

end Proof

end modulus_remainder_l770_77014


namespace distributor_cost_l770_77001

variable (C : ℝ) -- Cost of the item for the distributor
variable (P_observed : ℝ) -- Observed price
variable (commission_rate : ℝ) -- Commission rate
variable (profit_rate : ℝ) -- Desired profit rate

-- Conditions
def is_observed_price_correct (C : ℝ) (P_observed : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) : Prop :=
  let SP := C * (1 + profit_rate)
  let observed := SP * (1 - commission_rate)
  observed = P_observed

-- The proof goal
theorem distributor_cost (h : is_observed_price_correct C 30 0.20 0.20) : C = 31.25 := sorry

end distributor_cost_l770_77001


namespace circles_radii_divide_regions_l770_77097

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l770_77097


namespace fireworks_display_l770_77086

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end fireworks_display_l770_77086


namespace find_d_in_polynomial_l770_77069

theorem find_d_in_polynomial 
  (a b c d : ℤ) 
  (x1 x2 x3 x4 : ℤ)
  (roots_neg : x1 < 0 ∧ x2 < 0 ∧ x3 < 0 ∧ x4 < 0)
  (h_poly : ∀ x, 
    (x + x1) * (x + x2) * (x + x3) * (x + x4) = 
    x^4 + a * x^3 + b * x^2 + c * x + d)
  (h_sum_eq : a + b + c + d = 2009) :
  d = (x1 * x2 * x3 * x4) :=
by
  sorry

end find_d_in_polynomial_l770_77069


namespace siamese_cats_initial_l770_77083

theorem siamese_cats_initial (S : ℕ) (h1 : 20 + S - 20 = 12) : S = 12 :=
by
  sorry

end siamese_cats_initial_l770_77083


namespace problem_statement_l770_77022

open Nat

theorem problem_statement (k : ℕ) (hk : k > 0) : 
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * (factorial (k*n) / factorial n) ≤ (factorial (k*n) / factorial n))
  ↔ ∃ a : ℕ, k = 2^a := 
sorry

end problem_statement_l770_77022


namespace expected_volunteers_2008_l770_77002

theorem expected_volunteers_2008 (initial_volunteers: ℕ) (annual_increase: ℚ) (h1: initial_volunteers = 500) (h2: annual_increase = 1.2) : 
  let volunteers_2006 := initial_volunteers * annual_increase
  let volunteers_2007 := volunteers_2006 * annual_increase
  let volunteers_2008 := volunteers_2007 * annual_increase
  volunteers_2008 = 864 := 
by
  sorry

end expected_volunteers_2008_l770_77002


namespace math_proof_problem_l770_77064

noncomputable def proof_problem : Prop :=
  ∃ (p : ℝ) (k m : ℝ), 
    (∀ (x y : ℝ), y^2 = 2 * p * x) ∧
    (p > 0) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (y1 * y2 = -8) ∧
      (x1 = 4 ∧ y1 = 0 ∨ x2 = 4 ∧ y2 = 0)) ∧
    (p = 1) ∧ 
    (∀ x0 : ℝ, 
      (2 * k * m = 1) ∧
      (∀ (x y : ℝ), y = k * x + m) ∧ 
      (∃ (r : ℝ), 
        ((x0 - r + 1 = 0) ∧
         (x0 - r * x0 + r^2 = 0))) ∧ 
       x0 = -1 / 2 )

theorem math_proof_problem : proof_problem := 
  sorry

end math_proof_problem_l770_77064


namespace laura_five_dollar_bills_l770_77080

theorem laura_five_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 40) 
  (h2 : x + 2 * y + 5 * z = 120) 
  (h3 : y = 2 * x) : 
  z = 16 := 
by
  sorry

end laura_five_dollar_bills_l770_77080


namespace mo_rainy_days_last_week_l770_77019

theorem mo_rainy_days_last_week (R NR n : ℕ) (h1 : n * R + 4 * NR = 26) (h2 : 4 * NR - n * R = 14) (h3 : R + NR = 7) : R = 2 :=
sorry

end mo_rainy_days_last_week_l770_77019


namespace find_speed_of_stream_l770_77029

-- Definitions of the conditions:
def downstream_equation (b s : ℝ) : Prop := b + s = 60
def upstream_equation (b s : ℝ) : Prop := b - s = 30

-- Theorem stating the speed of the stream given the conditions:
theorem find_speed_of_stream (b s : ℝ) (h1 : downstream_equation b s) (h2 : upstream_equation b s) : s = 15 := 
sorry

end find_speed_of_stream_l770_77029


namespace solve_combinations_l770_77063

-- This function calculates combinations
noncomputable def C (n k : ℕ) : ℕ := if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem solve_combinations (x : ℤ) :
  C 16 (x^2 - x).natAbs = C 16 (5*x - 5).natAbs → x = 1 ∨ x = 3 :=
by
  sorry

end solve_combinations_l770_77063


namespace total_pieces_of_gum_l770_77062

theorem total_pieces_of_gum (packages pieces_per_package : ℕ) 
  (h_packages : packages = 9)
  (h_pieces_per_package : pieces_per_package = 15) : 
  packages * pieces_per_package = 135 := by
  subst h_packages
  subst h_pieces_per_package
  exact Nat.mul_comm 9 15 ▸ rfl

end total_pieces_of_gum_l770_77062


namespace wizard_achievable_for_odd_n_l770_77057

-- Define what it means for the wizard to achieve his goal
def wizard_goal_achievable (n : ℕ) : Prop :=
  ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = 2 * n ∧ 
    ∀ (sorcerer_breaks : Finset (ℕ × ℕ)), sorcerer_breaks.card = n → 
      ∃ (dwarves : Finset ℕ), dwarves.card = 2 * n ∧
      ∀ k ∈ dwarves, ((k, (k + 1) % n) ∈ pairs ∨ ((k + 1) % n, k) ∈ pairs) ∧
                     (∀ i j, (i, j) ∈ sorcerer_breaks → ¬((i, j) ∈ pairs ∨ (j, i) ∈ pairs))

theorem wizard_achievable_for_odd_n (n : ℕ) (h : Odd n) : wizard_goal_achievable n := sorry

end wizard_achievable_for_odd_n_l770_77057


namespace age_sum_l770_77021

variables (A B C : ℕ)

theorem age_sum (h1 : A = 20 + B + C) (h2 : A^2 = 2000 + (B + C)^2) : A + B + C = 100 :=
by
  -- Assume the subsequent proof follows here
  sorry

end age_sum_l770_77021


namespace checkerboard_contains_5_black_squares_l770_77033

def is_checkerboard (x y : ℕ) : Prop := 
  x < 8 ∧ y < 8 ∧ (x + y) % 2 = 0

def contains_5_black_squares (x y n : ℕ) : Prop :=
  ∃ k l : ℕ, k ≤ n ∧ l ≤ n ∧ (x + k + y + l) % 2 = 0 ∧ k * l >= 5

theorem checkerboard_contains_5_black_squares :
  ∃ num, num = 73 ∧
  (∀ x y n, contains_5_black_squares x y n → num = 73) :=
by
  sorry

end checkerboard_contains_5_black_squares_l770_77033


namespace select_4_blocks_no_same_row_column_l770_77081

theorem select_4_blocks_no_same_row_column :
  ∃ (n : ℕ), n = (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4) ∧ n = 5400 :=
by
  sorry

end select_4_blocks_no_same_row_column_l770_77081


namespace therapy_charge_l770_77092

-- Defining the conditions
variables (A F : ℝ)
variables (h1 : F = A + 25)
variables (h2 : F + 4*A = 250)

-- The statement we need to prove
theorem therapy_charge : F + A = 115 := 
by
  -- proof would go here
  sorry

end therapy_charge_l770_77092


namespace definite_integral_cos_exp_l770_77044

open Real

theorem definite_integral_cos_exp :
  ∫ x in -π..0, (cos x + exp x) = 1 - (1 / exp π) :=
by
  sorry

end definite_integral_cos_exp_l770_77044


namespace simplify_polynomial_subtraction_l770_77091

variable (x : ℝ)

def P1 : ℝ := 2*x^6 + x^5 + 3*x^4 + x^3 + 5
def P2 : ℝ := x^6 + 2*x^5 + x^4 - x^3 + 7
def P3 : ℝ := x^6 - x^5 + 2*x^4 + 2*x^3 - 2

theorem simplify_polynomial_subtraction : (P1 x - P2 x) = P3 x :=
by
  sorry

end simplify_polynomial_subtraction_l770_77091


namespace apples_needed_l770_77084

-- Define a simple equivalence relation between the weights of oranges and apples.
def weight_equivalent (oranges apples : ℕ) : Prop :=
  8 * apples = 6 * oranges
  
-- State the main theorem based on the given conditions
theorem apples_needed (oranges_count : ℕ) (h : weight_equivalent 1 1) : oranges_count = 32 → ∃ apples_count, apples_count = 24 :=
by
  sorry

end apples_needed_l770_77084


namespace total_miles_l770_77010

-- Define the variables and equations as given in the conditions
variables (a b c d e : ℝ)
axiom h1 : a + b = 36
axiom h2 : b + c + d = 45
axiom h3 : c + d + e = 45
axiom h4 : a + c + e = 38

-- The conjecture we aim to prove
theorem total_miles : a + b + c + d + e = 83 :=
sorry

end total_miles_l770_77010


namespace valid_raise_percentage_l770_77009

-- Define the conditions
def raise_between (x : ℝ) : Prop :=
  0.05 ≤ x ∧ x ≤ 0.10

def salary_increase_by_fraction (x : ℝ) : Prop :=
  x = 0.06

-- Define the main theorem 
theorem valid_raise_percentage (x : ℝ) (hx_between : raise_between x) (hx_fraction : salary_increase_by_fraction x) :
  x = 0.06 :=
sorry

end valid_raise_percentage_l770_77009


namespace sufficient_condition_l770_77039

variable (x : ℝ) (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, |x| + |x - 1| ≥ 1) : a < 1 → ∀ x : ℝ, a ≤ |x| + |x - 1| :=
by
  sorry

end sufficient_condition_l770_77039


namespace find_second_projection_l770_77049

noncomputable def second_projection (plane : Prop) (first_proj : Prop) (distance : ℝ) : Prop :=
∃ second_proj : Prop, true

theorem find_second_projection 
  (plane : Prop) 
  (first_proj : Prop) 
  (distance : ℝ) :
  ∃ second_proj : Prop, true :=
sorry

end find_second_projection_l770_77049


namespace girls_more_than_boys_l770_77076

theorem girls_more_than_boys (total_students boys : ℕ) (h1 : total_students = 650) (h2 : boys = 272) :
  (total_students - boys) - boys = 106 :=
by
  sorry

end girls_more_than_boys_l770_77076


namespace quadratic_value_at_5_l770_77030

-- Define the conditions provided in the problem
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Create a theorem that states that if a quadratic with given conditions has its vertex at (2, 7) and passes through (0, -7), then passing through (5, n) means n = -24.5
theorem quadratic_value_at_5 (a b c n : ℝ)
  (h1 : quadratic a b c 2 = 7)
  (h2 : quadratic a b c 0 = -7)
  (h3 : quadratic a b c 5 = n) :
  n = -24.5 :=
by
  sorry

end quadratic_value_at_5_l770_77030


namespace g_88_value_l770_77012

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n m : ℕ) (h : n < m) : g n < g m
axiom g_multiplicative (m n : ℕ) : g (m * n) = g m * g n
axiom g_exponential_condition (m n : ℕ) (h : m ≠ n ∧ m ^ n = n ^ m) : g m = n ∨ g n = m

theorem g_88_value : g 88 = 7744 :=
sorry

end g_88_value_l770_77012


namespace high_school_total_students_l770_77093

theorem high_school_total_students (N_seniors N_sample N_freshmen_sample N_sophomores_sample N_total : ℕ)
  (h_seniors : N_seniors = 1000)
  (h_sample : N_sample = 185)
  (h_freshmen_sample : N_freshmen_sample = 75)
  (h_sophomores_sample : N_sophomores_sample = 60)
  (h_proportion : N_seniors * (N_sample - (N_freshmen_sample + N_sophomores_sample)) = N_total * (N_sample - N_freshmen_sample - N_sophomores_sample)) :
  N_total = 3700 :=
by
  sorry

end high_school_total_students_l770_77093


namespace positivity_of_xyz_l770_77079

variable {x y z : ℝ}

theorem positivity_of_xyz
  (h1 : x + y + z > 0)
  (h2 : xy + yz + zx > 0)
  (h3 : xyz > 0) :
  x > 0 ∧ y > 0 ∧ z > 0 := 
sorry

end positivity_of_xyz_l770_77079


namespace smallest_positive_leading_coefficient_l770_77034

variable {a b c : ℚ} -- Define variables a, b, c that are rational numbers
variable (P : ℤ → ℚ) -- Define the polynomial P as a function from integers to rationals

-- State that P(x) is in the form of ax^2 + bx + c
def is_quadratic_polynomial (P : ℤ → ℚ) (a b c : ℚ) :=
  ∀ x : ℤ, P x = a * x^2 + b * x + c

-- State that P(x) takes integer values for all integer x
def takes_integer_values (P : ℤ → ℚ) :=
  ∀ x : ℤ, ∃ k : ℤ, P x = k

-- The statement we want to prove
theorem smallest_positive_leading_coefficient (h1 : is_quadratic_polynomial P a b c)
                                              (h2 : takes_integer_values P) :
  ∃ a : ℚ, 0 < a ∧ ∀ b c : ℚ, is_quadratic_polynomial P a b c → takes_integer_values P → a = 1/2 :=
sorry

end smallest_positive_leading_coefficient_l770_77034


namespace imo_inequality_l770_77025

variable {a b c : ℝ}

theorem imo_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : (a + b) * (b + c) * (c + a) = 1) :
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ (1 / 2) := 
sorry

end imo_inequality_l770_77025


namespace part1_part2_l770_77090

variable (x : ℝ)

def A := {x : ℝ | 1 < x ∧ x < 3}
def B := {x : ℝ | x < -3 ∨ 2 < x}

theorem part1 : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

theorem part2 (a b : ℝ) : (∀ x, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) → a = -5 ∧ b = 6 := by
  sorry

end part1_part2_l770_77090


namespace smallest_number_of_people_l770_77065

open Nat

theorem smallest_number_of_people (x : ℕ) :
  (∃ x, x % 18 = 0 ∧ x % 50 = 0 ∧
  (∀ y, y % 18 = 0 ∧ y % 50 = 0 → x ≤ y)) → x = 450 :=
by
  sorry

end smallest_number_of_people_l770_77065


namespace ninety_times_ninety_l770_77059

theorem ninety_times_ninety : (90 * 90) = 8100 := by
  let a := 100
  let b := 10
  have h1 : (90 * 90) = (a - b) * (a - b) := by decide
  have h2 : (a - b) * (a - b) = a^2 - 2 * a * b + b^2 := by decide
  have h3 : a = 100 := rfl
  have h4 : b = 10 := rfl
  have h5 : 100^2 - 2 * 100 * 10 + 10^2 = 8100 := by decide
  sorry

end ninety_times_ninety_l770_77059


namespace maximize_product_minimize_product_l770_77017

-- Define lists of the digits to be used
def digits : List ℕ := [2, 4, 6, 8]

-- Function to calculate the number from a list of digits
def toNumber (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * 10 + d) 0

-- Function to calculate the product given two numbers represented as lists of digits
def product (digits1 digits2 : List ℕ) : ℕ :=
  toNumber digits1 * toNumber digits2

-- Definitions of specific permutations to be used
def maxDigits1 : List ℕ := [8, 6, 4]
def maxDigit2 : List ℕ := [2]
def minDigits1 : List ℕ := [2, 4, 6]
def minDigit2 : List ℕ := [8]

-- Theorem statements
theorem maximize_product : product maxDigits1 maxDigit2 = 864 * 2 := by
  sorry

theorem minimize_product : product minDigits1 minDigit2 = 246 * 8 := by
  sorry

end maximize_product_minimize_product_l770_77017


namespace length_of_CD_l770_77023

theorem length_of_CD (x y : ℝ) (h1 : x = (1/5) * (4 + y))
  (h2 : (x + 4) / y = 2 / 3) (h3 : 4 = 4) : x + y + 4 = 17.143 :=
sorry

end length_of_CD_l770_77023


namespace equation_solution_count_l770_77007

theorem equation_solution_count (n : ℕ) (h_pos : n > 0)
    (h_solutions : ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 28 ∧ ∀ (x y z : ℕ), (x, y, z) ∈ s → 2 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) :
    n = 17 ∨ n = 18 :=
sorry

end equation_solution_count_l770_77007


namespace find_a_l770_77085

theorem find_a (a : ℝ) : (∃ (p : ℝ × ℝ), p = (3, -9) ∧ (3 * a * p.1 + (2 * a + 1) * p.2 = 3 * a + 3)) → a = -1 :=
by
  sorry

end find_a_l770_77085


namespace complement_of_beta_l770_77072

theorem complement_of_beta (α β : ℝ) (h₀ : α + β = 180) (h₁ : α > β) : 
  90 - β = 1/2 * (α - β) :=
by
  sorry

end complement_of_beta_l770_77072


namespace exists_x_y_mod_p_l770_77058

theorem exists_x_y_mod_p (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : ∃ x y : ℤ, (x^2 + y^3) % p = a % p :=
by
  sorry

end exists_x_y_mod_p_l770_77058


namespace total_number_of_girls_in_school_l770_77096

theorem total_number_of_girls_in_school 
  (students_sampled : ℕ) 
  (students_total : ℕ) 
  (sample_girls : ℕ) 
  (sample_boys : ℕ)
  (h_sample_size : students_sampled = 200)
  (h_total_students : students_total = 2000)
  (h_diff_girls_boys : sample_boys = sample_girls + 6)
  (h_stratified_sampling : students_sampled / students_total = 200 / 2000) :
  sample_girls * (students_total / students_sampled) = 970 :=
by
  sorry

end total_number_of_girls_in_school_l770_77096


namespace ratio_evaluation_l770_77042

theorem ratio_evaluation : (5^3003 * 2^3005) / (10^3004) = 2 / 5 := by
  sorry

end ratio_evaluation_l770_77042


namespace range_of_m_l770_77095

theorem range_of_m (m : ℝ) : (∃ x1 x2 x3 : ℝ, 
    (x1 - 1) * (x1^2 - 2*x1 + m) = 0 ∧ 
    (x2 - 1) * (x2^2 - 2*x2 + m) = 0 ∧ 
    (x3 - 1) * (x3^2 - 2*x3 + m) = 0 ∧ 
    x1 = 1 ∧ 
    x2^2 - 2*x2 + m = 0 ∧ 
    x3^2 - 2*x3 + m = 0 ∧ 
    x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1 ∧ 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0) ↔ 3 / 4 < m ∧ m ≤ 1 := 
by
  sorry

end range_of_m_l770_77095


namespace abs_pi_sub_abs_pi_sub_three_l770_77075

theorem abs_pi_sub_abs_pi_sub_three (h : Real.pi > 3) : 
  abs (Real.pi - abs (Real.pi - 3)) = 2 * Real.pi - 3 := 
by
  sorry

end abs_pi_sub_abs_pi_sub_three_l770_77075


namespace probability_black_ball_l770_77028

variable (total_balls : ℕ)
variable (red_balls : ℕ)
variable (white_probability : ℝ)

def number_of_balls : Prop := total_balls = 100
def red_ball_count : Prop := red_balls = 45
def white_ball_probability : Prop := white_probability = 0.23

theorem probability_black_ball 
  (h1 : number_of_balls total_balls)
  (h2 : red_ball_count red_balls)
  (h3 : white_ball_probability white_probability) :
  let white_balls := white_probability * total_balls 
  let black_balls := total_balls - red_balls - white_balls
  let black_ball_prob := black_balls / total_balls
  black_ball_prob = 0.32 :=
sorry

end probability_black_ball_l770_77028


namespace bounded_sequence_range_l770_77089

theorem bounded_sequence_range (a : ℝ) (a_n : ℕ → ℝ) (h1 : a_n 1 = a)
    (hrec : ∀ n : ℕ, a_n (n + 1) = 3 * (a_n n)^3 - 7 * (a_n n)^2 + 5 * (a_n n))
    (bounded : ∃ M : ℝ, ∀ n : ℕ, abs (a_n n) ≤ M) :
    0 ≤ a ∧ a ≤ 4/3 :=
by
  sorry

end bounded_sequence_range_l770_77089


namespace sufficient_but_not_necessary_l770_77000

theorem sufficient_but_not_necessary (x : ℝ) : 
  (1 < x ∧ x < 2) → (x > 0) ∧ ¬((x > 0) → (1 < x ∧ x < 2)) := 
by 
  sorry

end sufficient_but_not_necessary_l770_77000


namespace crayons_problem_l770_77004

theorem crayons_problem
  (S M L : ℕ)
  (hS_condition : (3 / 5 : ℚ) * S = 60)
  (hM_condition : (1 / 4 : ℚ) * M = 98)
  (hL_condition : (4 / 7 : ℚ) * L = 168) :
  S = 100 ∧ M = 392 ∧ L = 294 ∧ ((2 / 5 : ℚ) * S + (3 / 4 : ℚ) * M + (3 / 7 : ℚ) * L = 460) := 
by
  sorry

end crayons_problem_l770_77004


namespace bottles_of_regular_soda_l770_77026

theorem bottles_of_regular_soda
  (diet_soda : ℕ)
  (apples : ℕ)
  (more_bottles_than_apples : ℕ)
  (R : ℕ)
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : more_bottles_than_apples = 26)
  (h4 : R + diet_soda = apples + more_bottles_than_apples) :
  R = 72 := 
by sorry

end bottles_of_regular_soda_l770_77026


namespace work_completion_time_l770_77016

/-
Conditions:
1. A man alone can do the work in 6 days.
2. A woman alone can do the work in 18 days.
3. A boy alone can do the work in 9 days.

Question:
How long will they take to complete the work together?

Correct Answer:
3 days
-/

theorem work_completion_time (M W B : ℕ) (hM : M = 6) (hW : W = 18) (hB : B = 9) : 1 / (1/M + 1/W + 1/B) = 3 := 
by
  sorry

end work_completion_time_l770_77016


namespace general_term_of_arithmetic_seq_l770_77048

variable {a : ℕ → ℕ} 
variable {S : ℕ → ℕ}

/-- Definition of sum of first n terms of an arithmetic sequence -/
def sum_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n, a (n + 1) = a n + d

theorem general_term_of_arithmetic_seq
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 6 = 12)
  (h3 : S 3 = 12)
  (h4 : sum_of_arithmetic_sequence S a) :
  ∀ n, a n = 2 * n := 
sorry

end general_term_of_arithmetic_seq_l770_77048


namespace a_lt_sqrt3b_l770_77045

open Int

theorem a_lt_sqrt3b (a b : ℤ) (h1 : a > b) (h2 : b > 1) 
    (h3 : a + b ∣ a * b + 1) (h4 : a - b ∣ a * b - 1) : a < sqrt 3 * b :=
  sorry

end a_lt_sqrt3b_l770_77045


namespace sum_slope_y_intercept_eq_l770_77040

noncomputable def J : ℝ × ℝ := (0, 8)
noncomputable def K : ℝ × ℝ := (0, 0)
noncomputable def L : ℝ × ℝ := (10, 0)
noncomputable def G : ℝ × ℝ := ((J.1 + K.1) / 2, (J.2 + K.2) / 2)

theorem sum_slope_y_intercept_eq :
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  slope + y_intercept = 18 / 5 :=
by
  -- Place the conditions and setup here
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  -- Proof will be provided here eventually
  sorry

end sum_slope_y_intercept_eq_l770_77040


namespace peter_money_l770_77031

theorem peter_money (cost_per_ounce : ℝ) (amount_bought : ℝ) (leftover_money : ℝ) (total_money : ℝ) :
  cost_per_ounce = 0.25 ∧ amount_bought = 6 ∧ leftover_money = 0.50 → total_money = 2 :=
by
  intros h
  let h1 := h.1
  let h2 := h.2.1
  let h3 := h.2.2
  sorry

end peter_money_l770_77031


namespace fixed_cost_to_break_even_l770_77027

def cost_per_handle : ℝ := 0.6
def selling_price_per_handle : ℝ := 4.6
def num_handles_to_break_even : ℕ := 1910

theorem fixed_cost_to_break_even (F : ℝ) (h : F = num_handles_to_break_even * (selling_price_per_handle - cost_per_handle)) :
  F = 7640 := by
  sorry

end fixed_cost_to_break_even_l770_77027


namespace coconut_grove_problem_l770_77018

variable (x : ℝ)

-- Conditions
def trees_yield_40_nuts_per_year : ℝ := 40 * (x + 2)
def trees_yield_120_nuts_per_year : ℝ := 120 * x
def trees_yield_180_nuts_per_year : ℝ := 180 * (x - 2)
def average_yield_per_tree_per_year : ℝ := 100

-- Problem Statement
theorem coconut_grove_problem
  (yield_40_trees : trees_yield_40_nuts_per_year x = 40 * (x + 2))
  (yield_120_trees : trees_yield_120_nuts_per_year x = 120 * x)
  (yield_180_trees : trees_yield_180_nuts_per_year x = 180 * (x - 2))
  (average_yield : average_yield_per_tree_per_year = 100) :
  x = 7 :=
by
  sorry

end coconut_grove_problem_l770_77018


namespace soccer_field_solution_l770_77038

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l770_77038


namespace repeating_block_length_7_div_13_l770_77050

-- Definitions for the conditions
def decimal_expansion_period (n d : ℕ) : ℕ := sorry

-- The corresponding Lean statement
theorem repeating_block_length_7_div_13 : decimal_expansion_period 7 13 = 6 := 
sorry

end repeating_block_length_7_div_13_l770_77050


namespace total_weight_of_packages_l770_77088

theorem total_weight_of_packages (x y z w : ℕ) (h1 : x + y + z = 150) (h2 : y + z + w = 160) (h3 : z + w + x = 170) :
  x + y + z + w = 160 :=
by sorry

end total_weight_of_packages_l770_77088


namespace find_alpha_l770_77054

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = - (Real.sqrt 3 / 2)) (h_range : 0 < α ∧ α < Real.pi) : α = 5 * Real.pi / 6 :=
sorry

end find_alpha_l770_77054


namespace find_extreme_value_number_of_zeros_l770_77082

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + (a - 2) * x - Real.log x

-- Math proof problem I
theorem find_extreme_value (a : ℝ) (h : (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f a x > f a 1)) : a = 1 := 
sorry

-- Math proof problem II
theorem number_of_zeros (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := 
sorry

end find_extreme_value_number_of_zeros_l770_77082


namespace division_correct_l770_77013

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end division_correct_l770_77013


namespace number_of_boxes_ordered_l770_77006

-- Definitions based on the conditions
def boxes_contain_matchboxes : Nat := 20
def matchboxes_contain_sticks : Nat := 300
def total_match_sticks : Nat := 24000

-- Statement of the proof problem
theorem number_of_boxes_ordered :
  (total_match_sticks / matchboxes_contain_sticks) / boxes_contain_matchboxes = 4 := 
sorry

end number_of_boxes_ordered_l770_77006


namespace quadratic_eq_l770_77070

noncomputable def roots (r s : ℝ): Prop := r + s = 12 ∧ r * s = 27 ∧ (r = 2 * s ∨ s = 2 * r)

theorem quadratic_eq (r s : ℝ) (h : roots r s) : 
   Polynomial.C 1 * (X^2 - Polynomial.C (r + s) * X + Polynomial.C (r * s)) = X ^ 2 - 12 * X + 27 := 
sorry

end quadratic_eq_l770_77070


namespace charcoal_amount_l770_77077

theorem charcoal_amount (water_per_charcoal : ℕ) (charcoal_ratio : ℕ) (water_added : ℕ) (charcoal_needed : ℕ) 
  (h1 : water_per_charcoal = 30) (h2 : charcoal_ratio = 2) (h3 : water_added = 900) : charcoal_needed = 60 :=
by
  sorry

end charcoal_amount_l770_77077


namespace total_birds_correct_l770_77043

def numPairs : Nat := 3
def birdsPerPair : Nat := 2
def totalBirds : Nat := numPairs * birdsPerPair

theorem total_birds_correct : totalBirds = 6 :=
by
  -- proof goes here
  sorry

end total_birds_correct_l770_77043


namespace total_fat_l770_77046

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end total_fat_l770_77046


namespace positive_iff_triangle_l770_77061

def is_triangle_inequality (x y z : ℝ) : Prop :=
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x)

noncomputable def poly (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem positive_iff_triangle (x y z : ℝ) : 
  poly |x| |y| |z| > 0 ↔ is_triangle_inequality |x| |y| |z| :=
sorry

end positive_iff_triangle_l770_77061
