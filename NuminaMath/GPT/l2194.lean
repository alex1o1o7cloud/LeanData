import Mathlib

namespace problem_l2194_219424

def f (x: ℝ) := 3 * x - 4
def g (x: ℝ) := 2 * x + 3

theorem problem (x : ℝ) : f (2 + g 3) = 29 :=
by
  sorry

end problem_l2194_219424


namespace imaginary_part_of_fraction_l2194_219446

theorem imaginary_part_of_fraction (i : ℂ) (hi : i * i = -1) : (1 + i) / (1 - i) = 1 :=
by
  -- Skipping the proof
  sorry

end imaginary_part_of_fraction_l2194_219446


namespace hundreds_digit_even_l2194_219439

-- Define the given conditions
def units_digit (n : ℕ) : ℕ := n % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- The main theorem to prove
theorem hundreds_digit_even (x : ℕ) 
  (h1 : units_digit (x*x) = 9) 
  (h2 : tens_digit (x*x) = 0) : ((x*x) / 100) % 2 = 0 :=
  sorry

end hundreds_digit_even_l2194_219439


namespace probability_of_same_color_is_correct_l2194_219434

-- Define the parameters for balls in the bag
def green_balls : ℕ := 8
def red_balls : ℕ := 6
def blue_balls : ℕ := 1
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probabilities of drawing each color
def prob_green : ℚ := green_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_blue : ℚ := blue_balls / total_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ :=
  prob_green^2 + prob_red^2 + prob_blue^2

theorem probability_of_same_color_is_correct :
  prob_same_color = 101 / 225 :=
by
  sorry

end probability_of_same_color_is_correct_l2194_219434


namespace min_value_of_a_plus_b_l2194_219471

-- Definitions based on the conditions
variables (a b : ℝ)
def roots_real (a b : ℝ) : Prop := a^2 ≥ 8 * b ∧ b^2 ≥ a
def positive_vars (a b : ℝ) : Prop := a > 0 ∧ b > 0
def min_a_plus_b (a b : ℝ) : Prop := a + b = 6

-- Lean theorem statement
theorem min_value_of_a_plus_b (a b : ℝ) (hr : roots_real a b) (pv : positive_vars a b) : min_a_plus_b a b :=
sorry

end min_value_of_a_plus_b_l2194_219471


namespace thirty_times_multiple_of_every_integer_is_zero_l2194_219435

theorem thirty_times_multiple_of_every_integer_is_zero (n : ℤ) (h : ∀ x : ℤ, n = 30 * x ∧ x = 0 → n = 0) : n = 0 :=
by
  sorry

end thirty_times_multiple_of_every_integer_is_zero_l2194_219435


namespace problem_solution_l2194_219474

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n % 100) / 10) * 8 + (n % 10)

def base3_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 3 + (n % 10)

def base7_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 49 + ((n % 100) / 10) * 7 + (n % 10)

def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

def expression_in_base10 : ℕ :=
  (base8_to_base10 254) / (base3_to_base10 13) + (base7_to_base10 232) / (base5_to_base10 32)

theorem problem_solution : expression_in_base10 = 35 :=
by
  sorry

end problem_solution_l2194_219474


namespace original_triangle_angles_determined_l2194_219422

-- Define the angles of the formed triangle
def formed_triangle_angles : Prop := 
  52 + 61 + 67 = 180

-- Define the angles of the original triangle
def original_triangle_angles (α β γ : ℝ) : Prop := 
  α + β + γ = 180

theorem original_triangle_angles_determined :
  formed_triangle_angles → 
  ∃ α β γ : ℝ, 
    original_triangle_angles α β γ ∧
    α = 76 ∧ β = 58 ∧ γ = 46 :=
by
  sorry

end original_triangle_angles_determined_l2194_219422


namespace factorize_expression_l2194_219489

theorem factorize_expression (x y : ℝ) : 4 * x^2 - 2 * x * y = 2 * x * (2 * x - y) := 
by
  sorry

end factorize_expression_l2194_219489


namespace downstream_rate_l2194_219481

/--  
A man's rowing conditions and rates:
- The man's upstream rate is U = 12 kmph.
- The man's rate in still water is S = 7 kmph.
- We need to prove that the man's downstream rate D is 14 kmph.
-/
theorem downstream_rate (U S D : ℝ) (hU : U = 12) (hS : S = 7) : D = 14 :=
by
  -- Proof to be filled here
  sorry

end downstream_rate_l2194_219481


namespace martha_initial_marbles_l2194_219490

-- Definition of the conditions
def initial_marbles_dilan : ℕ := 14
def initial_marbles_phillip : ℕ := 19
def initial_marbles_veronica : ℕ := 7
def marbles_after_redistribution_each : ℕ := 15
def number_of_people : ℕ := 4

-- Total marbles after redistribution
def total_marbles_after_redistribution : ℕ := marbles_after_redistribution_each * number_of_people

-- Total initial marbles of Dilan, Phillip, and Veronica
def total_initial_marbles_dilan_phillip_veronica : ℕ := initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica

-- Prove the number of marbles Martha initially had
theorem martha_initial_marbles : initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica + x = number_of_people * marbles_after_redistribution →
  x = 20 := by
  sorry

end martha_initial_marbles_l2194_219490


namespace children_tickets_count_l2194_219469

theorem children_tickets_count (A C : ℕ) (h1 : 8 * A + 5 * C = 201) (h2 : A + C = 33) : C = 21 :=
by
  sorry

end children_tickets_count_l2194_219469


namespace ratio_sum_2_or_4_l2194_219454

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end ratio_sum_2_or_4_l2194_219454


namespace sum_max_min_ratios_l2194_219476

theorem sum_max_min_ratios
  (c d : ℚ)
  (h1 : ∀ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 → y / x = c ∨ y / x = d)
  (h2 : ∀ r : ℚ, (∃ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 ∧ y / x = r) → (r = c ∨ r = d))
  : c + d = 63 / 43 :=
sorry

end sum_max_min_ratios_l2194_219476


namespace geometric_sequence_condition_l2194_219427

variable {a : ℕ → ℝ}

-- Definitions based on conditions in the problem
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The statement translating the problem
theorem geometric_sequence_condition (q : ℝ) (a : ℕ → ℝ) (h : is_geometric_sequence a q) : ¬((q > 1) ↔ is_increasing_sequence a) :=
  sorry

end geometric_sequence_condition_l2194_219427


namespace binomial_coeff_sum_l2194_219421

theorem binomial_coeff_sum : 
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) + (Nat.choose 7 2) + (Nat.choose 8 2) = 83 := by
  sorry

end binomial_coeff_sum_l2194_219421


namespace investments_ratio_l2194_219449

theorem investments_ratio (P Q : ℝ) (hpq : 7 / 10 = (P * 2) / (Q * 4)) : P / Q = 7 / 5 :=
by 
  sorry

end investments_ratio_l2194_219449


namespace min_value_is_neg2032188_l2194_219428

noncomputable def min_expression_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) : ℝ :=
(x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016)

theorem min_value_is_neg2032188 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) :
  min_expression_value x y h_pos_x h_pos_y h_neq h_cond = -2032188 := 
sorry

end min_value_is_neg2032188_l2194_219428


namespace w12_plus_inv_w12_l2194_219411

open Complex

-- Given conditions
def w_plus_inv_w_eq_two_cos_45 (w : ℂ) : Prop :=
  w + (1 / w) = 2 * Real.cos (Real.pi / 4)

-- Statement of the theorem to prove
theorem w12_plus_inv_w12 {w : ℂ} (h : w_plus_inv_w_eq_two_cos_45 w) : 
  w^12 + (1 / (w^12)) = -2 :=
sorry

end w12_plus_inv_w12_l2194_219411


namespace part_I_part_II_l2194_219496

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

theorem part_I (m : ℝ) (h : ∀ x : ℝ, f x ≠ m) : m < -2 ∨ m > 2 :=
sorry

theorem part_II (P : ℝ × ℝ) (hP : P = (2, -6)) :
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b = 0 ∧ m = 3 ∧ b = 0) ∨ 
                 (m * x + b = 24 * x - 54 ∧ P.2 = 24 * P.1 - 54)) :=
sorry

end part_I_part_II_l2194_219496


namespace train_distance_in_2_hours_l2194_219420

theorem train_distance_in_2_hours :
  (∀ (t : ℕ), t = 90 → (1 / ↑t) * 7200 = 80) :=
by
  sorry

end train_distance_in_2_hours_l2194_219420


namespace surface_area_after_removal_l2194_219480

theorem surface_area_after_removal :
  let cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (cube_side * cube_side)
  (original_surface_area = 96) ->
  (6 * (cube_side * cube_side) - 8 * 3 * (corner_cube_side * corner_cube_side) + 8 * 3 * (corner_cube_side * corner_cube_side) = 96) :=
by
  intros
  sorry

end surface_area_after_removal_l2194_219480


namespace square_angle_l2194_219407

theorem square_angle (PQ QR : ℝ) (x : ℝ) (PQR_is_square : true)
  (angle_sum_of_triangle : ∀ a b c : ℝ, a + b + c = 180)
  (right_angle : ∀ a, a = 90) :
  x = 45 :=
by
  -- We start with the properties of the square (implicitly given by the conditions)
  -- Now use the conditions and provided values to conclude the proof
  sorry

end square_angle_l2194_219407


namespace find_larger_number_l2194_219488

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 8 * S + 15) : L = 1557 := 
sorry

end find_larger_number_l2194_219488


namespace cube_difference_l2194_219441

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : a^3 - b^3 = 124 :=
by sorry

end cube_difference_l2194_219441


namespace Ben_more_new_shirts_than_Joe_l2194_219419

theorem Ben_more_new_shirts_than_Joe :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
    alex_shirts = 4 →
    joe_shirts = alex_shirts + 3 →
    ben_shirts = 15 →
    ben_shirts - joe_shirts = 8 :=
by
  intros alex_shirts joe_shirts ben_shirts
  intros h_alex h_joe h_ben
  sorry

end Ben_more_new_shirts_than_Joe_l2194_219419


namespace bus_capacity_l2194_219429

def seats_available_on_left := 15
def seats_available_diff := 3
def people_per_seat := 3
def back_seat_capacity := 7

theorem bus_capacity : 
  (seats_available_on_left * people_per_seat) + 
  ((seats_available_on_left - seats_available_diff) * people_per_seat) + 
  back_seat_capacity = 88 := 
by 
  sorry

end bus_capacity_l2194_219429


namespace exponentiation_distributes_over_multiplication_l2194_219438

theorem exponentiation_distributes_over_multiplication (a b c : ℝ) : (a * b) ^ c = a ^ c * b ^ c := 
sorry

end exponentiation_distributes_over_multiplication_l2194_219438


namespace nearly_tricky_7_tiny_count_l2194_219410

-- Define a tricky polynomial
def is_tricky (P : Polynomial ℤ) : Prop :=
  Polynomial.eval 4 P = 0

-- Define a k-tiny polynomial
def is_k_tiny (k : ℤ) (P : Polynomial ℤ) : Prop :=
  P.degree ≤ 7 ∧ ∀ i, abs (Polynomial.coeff P i) ≤ k

-- Define a 1-tiny polynomial
def is_1_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 1 P

-- Define a nearly tricky polynomial as the sum of a tricky polynomial and a 1-tiny polynomial
def is_nearly_tricky (P : Polynomial ℤ) : Prop :=
  ∃ Q T : Polynomial ℤ, is_tricky Q ∧ is_1_tiny T ∧ P = Q + T

-- Define a 7-tiny polynomial
def is_7_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 7 P

-- Count the number of nearly tricky 7-tiny polynomials
def count_nearly_tricky_7_tiny : ℕ :=
  -- Simplification: hypothetical function counting the number of polynomials
  sorry

-- The main theorem statement
theorem nearly_tricky_7_tiny_count :
  count_nearly_tricky_7_tiny = 64912347 :=
sorry

end nearly_tricky_7_tiny_count_l2194_219410


namespace minimum_dimes_l2194_219462

-- Given amounts in dollars
def value_of_dimes (n : ℕ) : ℝ := 0.10 * n
def value_of_nickels : ℝ := 0.50
def value_of_one_dollar_bill : ℝ := 1.0
def value_of_four_tens : ℝ := 40.0
def price_of_scarf : ℝ := 42.85

-- Prove the total value of the money is at least the price of the scarf implies n >= 14
theorem minimum_dimes (n : ℕ) :
  value_of_four_tens + value_of_one_dollar_bill + value_of_nickels + value_of_dimes n ≥ price_of_scarf → n ≥ 14 :=
by
  sorry

end minimum_dimes_l2194_219462


namespace incorrect_proposition_C_l2194_219401

theorem incorrect_proposition_C (p q : Prop) :
  (¬(p ∨ q) → ¬p ∧ ¬q) ↔ False :=
by sorry

end incorrect_proposition_C_l2194_219401


namespace barefoot_kids_count_l2194_219461

def kidsInClassroom : Nat := 35
def kidsWearingSocks : Nat := 18
def kidsWearingShoes : Nat := 15
def kidsWearingBoth : Nat := 8

def barefootKids : Nat := kidsInClassroom - (kidsWearingSocks - kidsWearingBoth + kidsWearingShoes - kidsWearingBoth + kidsWearingBoth)

theorem barefoot_kids_count : barefootKids = 10 := by
  sorry

end barefoot_kids_count_l2194_219461


namespace nancy_first_counted_l2194_219402

theorem nancy_first_counted (x : ℤ) (h : (x + 12 + 1 + 12 + 7 + 3 + 8) / 6 = 7) : x = -1 := 
by 
  sorry

end nancy_first_counted_l2194_219402


namespace find_alpha_l2194_219459

-- Given conditions
variables (α β : ℝ)
axiom h1 : α + β = 11
axiom h2 : α * β = 24
axiom h3 : α > β

-- Theorems to prove
theorem find_alpha : α = 8 :=
  sorry

end find_alpha_l2194_219459


namespace decrease_in_average_salary_l2194_219447

-- Define the conditions
variable (I : ℕ := 20)
variable (L : ℕ := 10)
variable (initial_wage_illiterate : ℕ := 25)
variable (new_wage_illiterate : ℕ := 10)

-- Define the theorem statement
theorem decrease_in_average_salary :
  (I * (initial_wage_illiterate - new_wage_illiterate)) / (I + L) = 10 := by
  sorry

end decrease_in_average_salary_l2194_219447


namespace find_vector_l2194_219458

def line_r (t : ℝ) : ℝ × ℝ :=
  (2 + 5 * t, 3 - 2 * t)

def line_s (u : ℝ) : ℝ × ℝ :=
  (1 + 5 * u, -2 - 2 * u)

def is_projection (w1 w2 : ℝ) : Prop :=
  w1 - w2 = 3

theorem find_vector (w1 w2 : ℝ) (h_proj : is_projection w1 w2) :
  (w1, w2) = (-2, -5) :=
sorry

end find_vector_l2194_219458


namespace ellipse_with_foci_on_y_axis_l2194_219494

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1) ↔ (m > n ∧ n > 0) := 
sorry

end ellipse_with_foci_on_y_axis_l2194_219494


namespace diagonals_in_polygon_with_150_sides_l2194_219486

-- (a) Definitions for conditions
def sides : ℕ := 150

def diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- (c) Statement of the problem in Lean 4
theorem diagonals_in_polygon_with_150_sides :
  diagonals sides = 11025 :=
by
  sorry

end diagonals_in_polygon_with_150_sides_l2194_219486


namespace range_of_a_l2194_219409

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x^2 + 4 * x else Real.logb 2 x - a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 1 < a :=
sorry

end range_of_a_l2194_219409


namespace increased_area_l2194_219497

variable (r : ℝ)

theorem increased_area (r : ℝ) : 
  let initial_area : ℝ := π * r^2
  let final_area : ℝ := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π := by
sorry

end increased_area_l2194_219497


namespace complement_of_A_in_U_l2194_219491

open Set

def U : Set ℕ := {x | x < 8}
def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : (U \ A) = {0, 2, 5, 6} := by
  sorry

end complement_of_A_in_U_l2194_219491


namespace total_distance_is_27_l2194_219413

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end total_distance_is_27_l2194_219413


namespace total_spent_two_years_l2194_219436

def home_game_price : ℕ := 60
def away_game_price : ℕ := 75
def home_playoff_price : ℕ := 120
def away_playoff_price : ℕ := 100

def this_year_home_games : ℕ := 2
def this_year_away_games : ℕ := 2
def this_year_home_playoff_games : ℕ := 1
def this_year_away_playoff_games : ℕ := 0

def last_year_home_games : ℕ := 6
def last_year_away_games : ℕ := 3
def last_year_home_playoff_games : ℕ := 1
def last_year_away_playoff_games : ℕ := 1

def calculate_total_cost : ℕ :=
  let this_year_cost := this_year_home_games * home_game_price + this_year_away_games * away_game_price + this_year_home_playoff_games * home_playoff_price + this_year_away_playoff_games * away_playoff_price
  let last_year_cost := last_year_home_games * home_game_price + last_year_away_games * away_game_price + last_year_home_playoff_games * home_playoff_price + last_year_away_playoff_games * away_playoff_price
  this_year_cost + last_year_cost

theorem total_spent_two_years : calculate_total_cost = 1195 :=
by
  sorry

end total_spent_two_years_l2194_219436


namespace am_gm_inequality_l2194_219484

theorem am_gm_inequality {a b : ℝ} (n : ℕ) (h₁ : n ≠ 1) (h₂ : a > b) (h₃ : b > 0) : 
  ( (a + b) / 2 )^n < (a^n + b^n) / 2 := 
sorry

end am_gm_inequality_l2194_219484


namespace minimize_f_l2194_219475

noncomputable def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f (a : ℝ) : a = 82 / 43 :=
by
  sorry

end minimize_f_l2194_219475


namespace bottles_per_case_l2194_219483

theorem bottles_per_case (days: ℕ) (daily_intake: ℚ) (total_spent: ℚ) (case_cost: ℚ) (total_cases: ℕ) (total_bottles: ℕ) (B: ℕ) 
    (H1 : days = 240)
    (H2 : daily_intake = 1/2)
    (H3 : total_spent = 60)
    (H4 : case_cost = 12)
    (H5 : total_cases = total_spent / case_cost)
    (H6 : total_bottles = days * daily_intake)
    (H7 : B = total_bottles / total_cases) :
    B = 24 :=
by
    sorry

end bottles_per_case_l2194_219483


namespace total_amount_of_money_if_all_cookies_sold_equals_1255_50_l2194_219442

-- Define the conditions
def number_cookies_Clementine : ℕ := 72
def number_cookies_Jake : ℕ := 5 * number_cookies_Clementine / 2
def number_cookies_Tory : ℕ := (number_cookies_Jake + number_cookies_Clementine) / 2
def number_cookies_Spencer : ℕ := 3 * (number_cookies_Jake + number_cookies_Tory) / 2
def price_per_cookie : ℝ := 1.50

-- Total number of cookies
def total_cookies : ℕ :=
  number_cookies_Clementine + number_cookies_Jake + number_cookies_Tory + number_cookies_Spencer

-- Proof statement
theorem total_amount_of_money_if_all_cookies_sold_equals_1255_50 :
  (total_cookies * price_per_cookie : ℝ) = 1255.50 := by
  sorry

end total_amount_of_money_if_all_cookies_sold_equals_1255_50_l2194_219442


namespace remainder_when_160_divided_by_k_l2194_219455

-- Define k to be a positive integer
def positive_integer (n : ℕ) := n > 0

-- Given conditions in the problem
def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def problem_condition (k : ℕ) := positive_integer k ∧ (120 % (k * k) = 12)

-- Prove the main statement
theorem remainder_when_160_divided_by_k (k : ℕ) (h : problem_condition k) : 160 % k = 4 := 
sorry  -- Proof here

end remainder_when_160_divided_by_k_l2194_219455


namespace print_pages_l2194_219465

theorem print_pages (pages_per_cost : ℕ) (cost_cents : ℕ) (dollars : ℕ)
                    (h1 : pages_per_cost = 7) (h2 : cost_cents = 9) (h3 : dollars = 50) :
  (dollars * 100 * pages_per_cost) / cost_cents = 3888 :=
by
  sorry

end print_pages_l2194_219465


namespace solution_interval_for_x_l2194_219468

theorem solution_interval_for_x (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (48 / 7 ≤ x ∧ x < 49 / 7) :=
by sorry

end solution_interval_for_x_l2194_219468


namespace cars_served_from_4pm_to_6pm_l2194_219408

theorem cars_served_from_4pm_to_6pm : 
  let cars_per_15_min_peak := 12
  let cars_per_15_min_offpeak := 8 
  let blocks_in_an_hour := 4 
  let total_peak_hour := cars_per_15_min_peak * blocks_in_an_hour 
  let total_offpeak_hour := cars_per_15_min_offpeak * blocks_in_an_hour 
  total_peak_hour + total_offpeak_hour = 80 := 
by 
  sorry 

end cars_served_from_4pm_to_6pm_l2194_219408


namespace perpendicular_angles_l2194_219479

theorem perpendicular_angles (α : ℝ) 
  (h1 : 4 * Real.pi < α) 
  (h2 : α < 6 * Real.pi)
  (h3 : ∃ (k : ℤ), α = -2 * Real.pi / 3 + Real.pi / 2 + k * Real.pi) :
  α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6 :=
by
  sorry

end perpendicular_angles_l2194_219479


namespace isosceles_triangle_area_l2194_219464

theorem isosceles_triangle_area 
  (x y : ℝ)
  (h_perimeter : 2*y + 2*x = 32)
  (h_height : ∃ h : ℝ, h = 8 ∧ y^2 = x^2 + h^2) :
  ∃ area : ℝ, area = 48 :=
by
  sorry

end isosceles_triangle_area_l2194_219464


namespace range_of_a12_l2194_219492

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end range_of_a12_l2194_219492


namespace arrangements_21_leaders_l2194_219405

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutations A_n^k
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then factorial n / factorial (n - k) else 0

theorem arrangements_21_leaders : permutations 2 2 * permutations 18 18 = factorial 18 ^ 2 :=
by 
  sorry

end arrangements_21_leaders_l2194_219405


namespace cole_time_to_work_is_90_minutes_l2194_219487

noncomputable def cole_drive_time_to_work (D : ℝ) : ℝ := D / 30

def cole_trip_proof : Prop :=
  ∃ (D : ℝ), (D / 30) + (D / 90) = 2 ∧ cole_drive_time_to_work D * 60 = 90

theorem cole_time_to_work_is_90_minutes : cole_trip_proof :=
  sorry

end cole_time_to_work_is_90_minutes_l2194_219487


namespace range_of_a_l2194_219433

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), a_seq n = if n < 6 then (1 / 2 - a) * n + 1 else a ^ (n - 5))
  (h2 : ∀ (n : ℕ), n > 0 → a_seq n > a_seq (n + 1)) :
  (1 / 2 : ℝ) < a ∧ a < (7 / 12 : ℝ) :=
sorry

end range_of_a_l2194_219433


namespace smallest_k_for_distinct_real_roots_l2194_219440

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem smallest_k_for_distinct_real_roots :
  ∃ k : ℤ, (k > 0) ∧ discriminant (k : ℝ) (-3) (-9/4) > 0 ∧ (∀ m : ℤ, discriminant (m : ℝ) (-3) (-9/4) > 0 → m ≥ k) := 
by
  sorry

end smallest_k_for_distinct_real_roots_l2194_219440


namespace remainder_of_4000th_term_l2194_219448

def sequence_term_position (n : ℕ) : ℕ :=
  n^2

def sum_of_squares_up_to (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem remainder_of_4000th_term : 
  ∃ n : ℕ, sum_of_squares_up_to n ≥ 4000 ∧ (n-1) * n * (2 * (n-1) + 1) / 6 < 4000 ∧ (n % 7) = 1 :=
by 
  sorry

end remainder_of_4000th_term_l2194_219448


namespace mixed_groups_count_l2194_219418

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l2194_219418


namespace min_value_x2_y2_z2_l2194_219432

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 3 :=
sorry

end min_value_x2_y2_z2_l2194_219432


namespace largest_even_sum_1988_is_290_l2194_219453

theorem largest_even_sum_1988_is_290 (n : ℕ) 
  (h : 14 * n = 1988) : 2 * n + 6 = 290 :=
sorry

end largest_even_sum_1988_is_290_l2194_219453


namespace sequence_formula_l2194_219412

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_sum: ∀ n : ℕ, n ≥ 2 → S n = n^2 * a n)
  (h_a1 : a 1 = 1) : ∀ n : ℕ, n ≥ 2 → a n = 2 / (n * (n + 1)) :=
by {
  sorry
}

end sequence_formula_l2194_219412


namespace f_decreasing_f_odd_l2194_219444

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b

axiom negativity (x : ℝ) (h_pos : 0 < x) : f x < 0

theorem f_decreasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intros x
  sorry

end f_decreasing_f_odd_l2194_219444


namespace sum_of_ages_l2194_219498

theorem sum_of_ages (A B C : ℕ)
  (h1 : A = C + 8)
  (h2 : A + 10 = 3 * (C - 6))
  (h3 : B = 2 * C) :
  A + B + C = 80 := 
by 
  sorry

end sum_of_ages_l2194_219498


namespace tim_fewer_apples_l2194_219451

theorem tim_fewer_apples (martha_apples : ℕ) (harry_apples : ℕ) (tim_apples : ℕ) (H1 : martha_apples = 68) (H2 : harry_apples = 19) (H3 : harry_apples * 2 = tim_apples) : martha_apples - tim_apples = 30 :=
by
  sorry

end tim_fewer_apples_l2194_219451


namespace determine_d_l2194_219470

theorem determine_d (u v d c : ℝ) (p q : ℝ → ℝ)
  (hp : ∀ x, p x = x^3 + c * x + d)
  (hq : ∀ x, q x = x^3 + c * x + d + 300)
  (huv : p u = 0 ∧ p v = 0)  
  (hu5_v4 : q (u + 5) = 0 ∧ q (v - 4) = 0)
  (sum_roots_p : u + v + (-u - v) = 0)
  (sum_roots_q : (u + 5) + (v - 4) + (-u - v - 1) = 0)
  : d = -4 ∨ d = 6 :=
sorry

end determine_d_l2194_219470


namespace rectangle_area_l2194_219485

theorem rectangle_area :
  ∃ (x y : ℝ), (x + 3.5) * (y - 1.5) = x * y ∧
               (x - 3.5) * (y + 2.5) = x * y ∧
               2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y ∧
               x * y = 196 :=
by
  sorry

end rectangle_area_l2194_219485


namespace pages_in_first_issue_l2194_219437

-- Define variables for the number of pages in the issues and total pages
variables (P : ℕ) (total_pages : ℕ) (eqn : total_pages = 3 * P + 4)

-- State the theorem using the given conditions and question
theorem pages_in_first_issue (h : total_pages = 220) : P = 72 :=
by
  -- Use the given equation
  have h_eqn : total_pages = 3 * P + 4 := eqn
  sorry

end pages_in_first_issue_l2194_219437


namespace man_is_older_by_16_l2194_219445

variable (M S : ℕ)

-- Condition: The present age of the son is 14.
def son_age := S = 14

-- Condition: In two years, the man's age will be twice the son's age.
def age_relation := M + 2 = 2 * (S + 2)

-- Theorem: Prove that the man is 16 years older than his son.
theorem man_is_older_by_16 (h1 : son_age S) (h2 : age_relation M S) : M - S = 16 := 
sorry

end man_is_older_by_16_l2194_219445


namespace soap_last_duration_l2194_219423

-- Definitions of the given conditions
def cost_per_bar := 8 -- cost in dollars
def total_spent := 48 -- total spent in dollars
def months_in_year := 12

-- Definition of the query statement/proof goal
theorem soap_last_duration (h₁ : total_spent = 48) (h₂ : cost_per_bar = 8) (h₃ : months_in_year = 12) : months_in_year / (total_spent / cost_per_bar) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end soap_last_duration_l2194_219423


namespace zain_has_80_coins_l2194_219456

theorem zain_has_80_coins (emerie_quarters emerie_dimes emerie_nickels emerie_pennies emerie_half_dollars : ℕ)
  (h_quarters : emerie_quarters = 6) 
  (h_dimes : emerie_dimes = 7)
  (h_nickels : emerie_nickels = 5)
  (h_pennies : emerie_pennies = 10) 
  (h_half_dollars : emerie_half_dollars = 2) : 
  10 + emerie_quarters + 10 + emerie_dimes + 10 + emerie_nickels + 10 + emerie_pennies + 10 + emerie_half_dollars = 80 :=
by
  sorry

end zain_has_80_coins_l2194_219456


namespace inheritance_problem_l2194_219426

theorem inheritance_problem
    (A B C : ℕ)
    (h1 : A + B + C = 30000)
    (h2 : A - B = B - C)
    (h3 : A = B + C) :
    A = 15000 ∧ B = 10000 ∧ C = 5000 := by
  sorry

end inheritance_problem_l2194_219426


namespace solve_system_equations_l2194_219414

theorem solve_system_equations (a b c x y z : ℝ) (h1 : x + y + z = 0)
(h2 : c * x + a * y + b * z = 0)
(h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
(x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = a - b ∧ y = b - c ∧ z = c - a) := 
sorry

end solve_system_equations_l2194_219414


namespace part1_part2_l2194_219450

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : f x ≤ x^2 :=
sorry

theorem part2 (x : ℝ) (hx : x > 0) (c : ℝ) (hc : c ≥ -1) : f x ≤ 2 * x + c :=
sorry

end part1_part2_l2194_219450


namespace charged_amount_is_35_l2194_219493

-- Definitions based on conditions
def annual_interest_rate : ℝ := 0.05
def owed_amount : ℝ := 36.75
def time_in_years : ℝ := 1

-- The amount charged on the account in January
def charged_amount (P : ℝ) : Prop :=
  owed_amount = P + (P * annual_interest_rate * time_in_years)

-- The proof statement
theorem charged_amount_is_35 : charged_amount 35 := by
  sorry

end charged_amount_is_35_l2194_219493


namespace sign_of_a_l2194_219406

theorem sign_of_a (a b c d : ℝ) (h : b * (3 * d + 2) ≠ 0) (ineq : a / b < -c / (3 * d + 2)) : 
  (a = 0 ∨ a > 0 ∨ a < 0) :=
sorry

end sign_of_a_l2194_219406


namespace functional_equation_to_odd_function_l2194_219416

variables (f : ℝ → ℝ)

theorem functional_equation_to_odd_function (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  f 0 = 0 ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end functional_equation_to_odd_function_l2194_219416


namespace min_x_y_l2194_219415

theorem min_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : x + y ≥ 4 :=
by
  sorry

end min_x_y_l2194_219415


namespace woman_waits_time_until_man_catches_up_l2194_219472

theorem woman_waits_time_until_man_catches_up
  (woman_speed : ℝ)
  (man_speed : ℝ)
  (wait_time : ℝ)
  (woman_slows_after : ℝ)
  (h_man_speed : man_speed = 5 / 60) -- man's speed in miles per minute
  (h_woman_speed : woman_speed = 25 / 60) -- woman's speed in miles per minute
  (h_wait_time : woman_slows_after = 5) -- the time in minutes after which the woman waits for man
  (h_woman_waits : wait_time = 25) : wait_time = (woman_slows_after * woman_speed) / man_speed :=
sorry

end woman_waits_time_until_man_catches_up_l2194_219472


namespace largest_by_changing_first_digit_l2194_219457

-- Define the original number
def original_number : ℝ := 0.7162534

-- Define the transformation that changes a specific digit to 8
def transform_to_8 (n : ℕ) (d : ℝ) : ℝ :=
  match n with
  | 1 => 0.8162534
  | 2 => 0.7862534
  | 3 => 0.7182534
  | 4 => 0.7168534
  | 5 => 0.7162834
  | 6 => 0.7162584
  | 7 => 0.7162538
  | _ => d

-- State the theorem
theorem largest_by_changing_first_digit :
  ∀ (n : ℕ), transform_to_8 1 original_number ≥ transform_to_8 n original_number :=
by
  sorry

end largest_by_changing_first_digit_l2194_219457


namespace janice_walk_dog_more_than_homework_l2194_219478

theorem janice_walk_dog_more_than_homework 
  (H C T: Nat) 
  (W: Nat) 
  (total_time remaining_time spent_time: Nat) 
  (hw_time room_time trash_time extra_time: Nat)
  (H_eq : H = 30)
  (C_eq : C = H / 2)
  (T_eq : T = H / 6)
  (remaining_time_eq : remaining_time = 35)
  (total_time_eq : total_time = 120)
  (spent_time_eq : spent_time = total_time - remaining_time)
  (task_time_sum_eq : task_time_sum = H + C + T)
  (W_eq : W = spent_time - task_time_sum)
  : W - H = 5 := 
sorry

end janice_walk_dog_more_than_homework_l2194_219478


namespace notebooks_have_50_pages_l2194_219417

theorem notebooks_have_50_pages (notebooks : ℕ) (total_dollars : ℕ) (page_cost_cents : ℕ) 
  (total_cents : ℕ) (total_pages : ℕ) (pages_per_notebook : ℕ)
  (h1 : notebooks = 2) 
  (h2 : total_dollars = 5) 
  (h3 : page_cost_cents = 5) 
  (h4 : total_cents = total_dollars * 100) 
  (h5 : total_pages = total_cents / page_cost_cents) 
  (h6 : pages_per_notebook = total_pages / notebooks) 
  : pages_per_notebook = 50 :=
by
  sorry

end notebooks_have_50_pages_l2194_219417


namespace part1_inequality_part2_inequality_l2194_219452

theorem part1_inequality (x : ℝ) : 
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 ∨ x < 1 / 2 := 
by sorry

theorem part2_inequality (x a : ℝ) : 
  x^2 - a * x - 2 * a^2 < 0 ↔ 
  (a = 0 → False) ∧ 
  (a > 0 → -a < x ∧ x < 2 * a) ∧ 
  (a < 0 → 2 * a < x ∧ x < -a) := 
by sorry

end part1_inequality_part2_inequality_l2194_219452


namespace sum_inequality_l2194_219404

theorem sum_inequality (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (b * (a + b))) + (1 / (c * (b + c))) + (1 / (a * (c + a))) ≥ 3 / 2 :=
sorry

end sum_inequality_l2194_219404


namespace sum_of_geometric_sequence_first_9000_terms_l2194_219430

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l2194_219430


namespace find_x_value_l2194_219466

theorem find_x_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 :=
sorry

end find_x_value_l2194_219466


namespace total_amount_given_away_l2194_219431

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end total_amount_given_away_l2194_219431


namespace expression_evaluate_l2194_219482

theorem expression_evaluate :
  50 * (50 - 5) - (50 * 50 - 5) = -245 :=
by
  sorry

end expression_evaluate_l2194_219482


namespace percentage_of_males_l2194_219473

noncomputable def total_employees : ℝ := 1800
noncomputable def males_below_50_years_old : ℝ := 756
noncomputable def percentage_below_50 : ℝ := 0.70

theorem percentage_of_males : (males_below_50_years_old / percentage_below_50 / total_employees) * 100 = 60 :=
by
  sorry

end percentage_of_males_l2194_219473


namespace proof_problem_l2194_219467

theorem proof_problem (x : ℝ) : (0 < x ∧ x < 5) → (x^2 - 5 * x < 0) ∧ (|x - 2| < 3) :=
by
  sorry

end proof_problem_l2194_219467


namespace max_probability_of_winning_is_correct_l2194_219400

noncomputable def max_probability_of_winning : ℚ :=
  sorry

theorem max_probability_of_winning_is_correct :
  max_probability_of_winning = 17 / 32 :=
sorry

end max_probability_of_winning_is_correct_l2194_219400


namespace find_original_number_of_men_l2194_219443

theorem find_original_number_of_men (x : ℕ) (h1 : x * 12 = (x - 6) * 14) : x = 42 :=
  sorry

end find_original_number_of_men_l2194_219443


namespace Joan_spent_68_353_on_clothing_l2194_219403

theorem Joan_spent_68_353_on_clothing :
  let shorts := 15.00
  let jacket := 14.82 * 0.9
  let shirt := 12.51 * 0.5
  let shoes := 21.67 - 3
  let hat := 8.75
  let belt := 6.34
  shorts + jacket + shirt + shoes + hat + belt = 68.353 :=
sorry

end Joan_spent_68_353_on_clothing_l2194_219403


namespace ratio_area_rect_sq_l2194_219463

/-- 
  Given:
  1. The longer side of rectangle R is 1.2 times the length of a side of square S.
  2. The shorter side of rectangle R is 0.85 times the length of a side of square S.
  Prove that the ratio of the area of rectangle R to the area of square S is 51/50.
-/
theorem ratio_area_rect_sq (s : ℝ) 
  (h1 : ∃ r1, r1 = 1.2 * s) 
  (h2 : ∃ r2, r2 = 0.85 * s) : 
  (1.2 * s * 0.85 * s) / (s * s) = 51 / 50 := 
by
  sorry

end ratio_area_rect_sq_l2194_219463


namespace households_3_houses_proportion_l2194_219495

noncomputable def total_households : ℕ := 100000
noncomputable def ordinary_households : ℕ := 99000
noncomputable def high_income_households : ℕ := 1000

noncomputable def sampled_ordinary_households : ℕ := 990
noncomputable def sampled_high_income_households : ℕ := 100

noncomputable def sampled_ordinary_3_houses : ℕ := 40
noncomputable def sampled_high_income_3_houses : ℕ := 80

noncomputable def proportion_3_houses : ℝ := (sampled_ordinary_3_houses / sampled_ordinary_households * ordinary_households + sampled_high_income_3_houses / sampled_high_income_households * high_income_households) / total_households

theorem households_3_houses_proportion : proportion_3_houses = 0.048 := 
by
  sorry

end households_3_houses_proportion_l2194_219495


namespace largest_four_digit_number_property_l2194_219477

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l2194_219477


namespace maximum_sum_S6_l2194_219425

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

def sum_arithmetic_sequence (a d : α) (n : ℕ) : α :=
  (n : α) / 2 * (2 * a + (n - 1) * d)

theorem maximum_sum_S6 (a d : α)
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 10 < 0)
  (h2 : sum_arithmetic_sequence a d 11 > 0) :
  ∀ n : ℕ, sum_arithmetic_sequence a d n ≤ sum_arithmetic_sequence a d 6 :=
by sorry

end maximum_sum_S6_l2194_219425


namespace ratio_of_age_differences_l2194_219499

variable (R J K : ℕ)

-- conditions
axiom h1 : R = J + 6
axiom h2 : R + 2 = 2 * (J + 2)
axiom h3 : (R + 2) * (K + 2) = 108

-- statement to prove
theorem ratio_of_age_differences : (R - J) = 2 * (R - K) := 
sorry

end ratio_of_age_differences_l2194_219499


namespace initial_pennies_l2194_219460

theorem initial_pennies (initial: ℕ) (h : initial + 93 = 191) : initial = 98 := by
  sorry

end initial_pennies_l2194_219460
