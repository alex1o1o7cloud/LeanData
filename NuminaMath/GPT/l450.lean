import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_sum_l450_45061

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, (r > 0) ∧ (∀ n : ℕ, a (n + 1) = a n * r)

theorem geometric_sequence_sum
  (a_seq_geometric : is_geometric_sequence a)
  (a_pos : ∀ n : ℕ, a n > 0)
  (eqn : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l450_45061


namespace NUMINAMATH_GPT_zero_sum_of_squares_eq_zero_l450_45053

theorem zero_sum_of_squares_eq_zero {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end NUMINAMATH_GPT_zero_sum_of_squares_eq_zero_l450_45053


namespace NUMINAMATH_GPT_cost_price_is_92_percent_l450_45093

noncomputable def cost_price_percentage_of_selling_price (profit_percentage : ℝ) : ℝ :=
  let CP := (1 / ((profit_percentage / 100) + 1))
  CP * 100

theorem cost_price_is_92_percent (profit_percentage : ℝ) (h : profit_percentage = 8.695652173913043) :
  cost_price_percentage_of_selling_price profit_percentage = 92 :=
by
  rw [h]
  -- now we need to show that cost_price_percentage_of_selling_price 8.695652173913043 = 92
  -- by definition, cost_price_percentage_of_selling_price 8.695652173913043 is:
  -- let CP := 1 / (8.695652173913043 / 100 + 1)
  -- CP * 100 = (1 / (8.695652173913043 / 100 + 1)) * 100
  sorry

end NUMINAMATH_GPT_cost_price_is_92_percent_l450_45093


namespace NUMINAMATH_GPT_union_M_N_l450_45085

noncomputable def M : Set ℝ := { x | x^2 - 3 * x = 0 }
noncomputable def N : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }

theorem union_M_N : M ∪ N = {0, 2, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_union_M_N_l450_45085


namespace NUMINAMATH_GPT_abs_eq_2_iff_l450_45034

theorem abs_eq_2_iff (a : ℚ) : abs a = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_2_iff_l450_45034


namespace NUMINAMATH_GPT_sum_of_cubes_ratio_l450_45072

theorem sum_of_cubes_ratio (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) (h2 : d + e + f = 0) :
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = (a * b * c) / (d * e * f) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_ratio_l450_45072


namespace NUMINAMATH_GPT_ratio_of_socks_l450_45086

theorem ratio_of_socks (y p : ℝ) (h1 : 5 * p + y * 2 * p = 5 * p + 4 * y * p / 3) :
  (5 : ℝ) / y = 11 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_socks_l450_45086


namespace NUMINAMATH_GPT_negation_statement_l450_45024

theorem negation_statement (h : ∀ x : ℝ, |x - 2| + |x - 4| > 3) : 
  ∃ x0 : ℝ, |x0 - 2| + |x0 - 4| ≤ 3 :=
sorry

end NUMINAMATH_GPT_negation_statement_l450_45024


namespace NUMINAMATH_GPT_ben_fewer_pints_than_kathryn_l450_45079

-- Define the conditions
def annie_picked := 8
def kathryn_picked := annie_picked + 2
def total_picked := 25

-- Add noncomputable because constants are involved
noncomputable def ben_picked : ℕ := total_picked - (annie_picked + kathryn_picked)

theorem ben_fewer_pints_than_kathryn : ben_picked = kathryn_picked - 3 := 
by 
  -- The problem statement does not require proof body
  sorry

end NUMINAMATH_GPT_ben_fewer_pints_than_kathryn_l450_45079


namespace NUMINAMATH_GPT_rahim_average_price_per_book_l450_45069

noncomputable section

open BigOperators

def store_A_price_per_book : ℝ := 
  let original_total := 1600
  let discount := original_total * 0.15
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.05
  let final_total := discounted_total + sales_tax
  final_total / 25

def store_B_price_per_book : ℝ := 
  let original_total := 3200
  let effective_books_paid := 35 - (35 / 4)
  original_total / effective_books_paid

def store_C_price_per_book : ℝ := 
  let original_total := 3800
  let discount := 0.10 * (4 * (original_total / 40))
  let discounted_total := original_total - discount
  let service_charge := discounted_total * 0.07
  let final_total := discounted_total + service_charge
  final_total / 40

def store_D_price_per_book : ℝ := 
  let original_total := 2400
  let discount := 0.50 * (original_total / 30)
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.06
  let final_total := discounted_total + sales_tax
  final_total / 30

def store_E_price_per_book : ℝ := 
  let original_total := 1800
  let discount := original_total * 0.08
  let discounted_total := original_total - discount
  let additional_fee := discounted_total * 0.04
  let final_total := discounted_total + additional_fee
  final_total / 20

def total_books : ℝ := 25 + 35 + 40 + 30 + 20

def total_amount : ℝ := 
  store_A_price_per_book * 25 + 
  store_B_price_per_book * 35 + 
  store_C_price_per_book * 40 + 
  store_D_price_per_book * 30 + 
  store_E_price_per_book * 20

def average_price_per_book : ℝ := total_amount / total_books

theorem rahim_average_price_per_book : average_price_per_book = 85.85 :=
sorry

end NUMINAMATH_GPT_rahim_average_price_per_book_l450_45069


namespace NUMINAMATH_GPT_Lacy_correct_percent_l450_45095

theorem Lacy_correct_percent (x : ℝ) (h1 : 7 * x > 0) : ((5 * 100) / 7) = 71.43 :=
by
  sorry

end NUMINAMATH_GPT_Lacy_correct_percent_l450_45095


namespace NUMINAMATH_GPT_kaleb_cherries_left_l450_45021

theorem kaleb_cherries_left (initial_cherries eaten_cherries remaining_cherries : ℕ) (h1 : initial_cherries = 67) (h2 : eaten_cherries = 25) : remaining_cherries = initial_cherries - eaten_cherries → remaining_cherries = 42 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_kaleb_cherries_left_l450_45021


namespace NUMINAMATH_GPT_brandon_skittles_loss_l450_45027

theorem brandon_skittles_loss (original final : ℕ) (H1 : original = 96) (H2 : final = 87) : original - final = 9 :=
by sorry

end NUMINAMATH_GPT_brandon_skittles_loss_l450_45027


namespace NUMINAMATH_GPT_problem_solution_l450_45044

theorem problem_solution (a b : ℝ) (h1 : a^3 - 15 * a^2 + 25 * a - 75 = 0) (h2 : 8 * b^3 - 60 * b^2 - 310 * b + 2675 = 0) :
  a + b = 15 / 2 :=
sorry

end NUMINAMATH_GPT_problem_solution_l450_45044


namespace NUMINAMATH_GPT_mary_pays_fifteen_l450_45047

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_per_5_fruits : ℕ := 1

def apples_bought : ℕ := 5
def oranges_bought : ℕ := 3
def bananas_bought : ℕ := 2

def total_cost_before_discount : ℕ :=
  apples_bought * apple_cost +
  oranges_bought * orange_cost +
  bananas_bought * banana_cost

def total_fruits : ℕ :=
  apples_bought + oranges_bought + bananas_bought

def total_discount : ℕ :=
  (total_fruits / 5) * discount_per_5_fruits

def final_amount_to_pay : ℕ :=
  total_cost_before_discount - total_discount

theorem mary_pays_fifteen : final_amount_to_pay = 15 := by
  sorry

end NUMINAMATH_GPT_mary_pays_fifteen_l450_45047


namespace NUMINAMATH_GPT_glycerin_solution_l450_45062

theorem glycerin_solution (x : ℝ) :
    let total_volume := 100
    let final_glycerin_percentage := 0.75
    let volume_first_solution := 75
    let volume_second_solution := 75
    let second_solution_percentage := 0.90
    let final_glycerin_volume := final_glycerin_percentage * total_volume
    let glycerin_second_solution := second_solution_percentage * volume_second_solution
    let glycerin_first_solution := x * volume_first_solution / 100
    glycerin_first_solution + glycerin_second_solution = final_glycerin_volume →
    x = 10 :=
by
    sorry

end NUMINAMATH_GPT_glycerin_solution_l450_45062


namespace NUMINAMATH_GPT_polygon_area_l450_45035

-- Define the vertices of the polygon
def x1 : ℝ := 0
def y1 : ℝ := 0

def x2 : ℝ := 4
def y2 : ℝ := 0

def x3 : ℝ := 2
def y3 : ℝ := 3

def x4 : ℝ := 4
def y4 : ℝ := 6

-- Define the expression for the Shoelace Theorem
def shoelace_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  0.5 * abs (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

-- The theorem statement proving the area of the polygon
theorem polygon_area :
  shoelace_area x1 y1 x2 y2 x3 y3 x4 y4 = 6 := 
  by
  sorry

end NUMINAMATH_GPT_polygon_area_l450_45035


namespace NUMINAMATH_GPT_table_price_l450_45048

theorem table_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : C + T = 60) :
  T = 52.5 :=
by
  sorry

end NUMINAMATH_GPT_table_price_l450_45048


namespace NUMINAMATH_GPT_factor_difference_of_squares_l450_45002

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l450_45002


namespace NUMINAMATH_GPT_problem1_problem2_l450_45073

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- Problem 1: Find x such that f(x) < -2 when a = 1
theorem problem1 : 
  {x : ℝ | f x 1 < -2} = {x | x > 3 / 2} :=
sorry

-- Problem 2: Find the range of values for 'a' when -2 + f(y) ≤ f(x) ≤ 2 + f(y) for all x, y ∈ ℝ
theorem problem2 : 
  (∀ x y : ℝ, -2 + f y a ≤ f x a ∧ f x a ≤ 2 + f y a) ↔ (-3 ≤ a ∧ a ≤ -1) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l450_45073


namespace NUMINAMATH_GPT_minibuses_not_enough_l450_45078

def num_students : ℕ := 300
def minibus_capacity : ℕ := 23
def num_minibuses : ℕ := 13

theorem minibuses_not_enough :
  num_minibuses * minibus_capacity < num_students :=
by
  sorry

end NUMINAMATH_GPT_minibuses_not_enough_l450_45078


namespace NUMINAMATH_GPT_boys_neither_happy_nor_sad_correct_l450_45003

def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def total_boys : ℕ := 16
def total_girls : ℕ := 44
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- The number of boys who are neither happy nor sad
def boys_neither_happy_nor_sad : ℕ :=
  total_boys - happy_boys - (sad_children - sad_girls)

theorem boys_neither_happy_nor_sad_correct : boys_neither_happy_nor_sad = 4 := by
  sorry

end NUMINAMATH_GPT_boys_neither_happy_nor_sad_correct_l450_45003


namespace NUMINAMATH_GPT_evaluate_expr_l450_45075

-- Define the imaginary unit i
def i := Complex.I

-- Define the expressions for the proof
def expr1 := (1 + 2 * i) * i ^ 3
def expr2 := 2 * i ^ 2

-- The main statement we need to prove
theorem evaluate_expr : expr1 + expr2 = -i :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expr_l450_45075


namespace NUMINAMATH_GPT_find_q_l450_45071

variable {a d q : ℝ}
variables (M N : Set ℝ)

theorem find_q (hM : M = {a, a + d, a + 2 * d}) 
              (hN : N = {a, a * q, a * q^2})
              (ha : a ≠ 0)
              (heq : M = N) :
  q = -1 / 2 :=
sorry

end NUMINAMATH_GPT_find_q_l450_45071


namespace NUMINAMATH_GPT_tino_jellybeans_l450_45031

theorem tino_jellybeans (Tino Lee Arnold Joshua : ℕ)
  (h1 : Tino = Lee + 24)
  (h2 : Arnold = Lee / 2)
  (h3 : Joshua = 3 * Arnold)
  (h4 : Arnold = 5) : Tino = 34 := by
sorry

end NUMINAMATH_GPT_tino_jellybeans_l450_45031


namespace NUMINAMATH_GPT_new_person_weight_l450_45087

/-- Conditions: The average weight of 8 persons increases by 6 kg when a new person replaces one of them weighing 45 kg -/
theorem new_person_weight (W : ℝ) (new_person_wt : ℝ) (avg_increase : ℝ) (replaced_person_wt : ℝ) 
  (h1 : avg_increase = 6) (h2 : replaced_person_wt = 45) (weight_increase : 8 * avg_increase = new_person_wt - replaced_person_wt) :
  new_person_wt = 93 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l450_45087


namespace NUMINAMATH_GPT_min_value_of_expression_l450_45022

theorem min_value_of_expression (x y : ℝ) : 
  ∃ x y, 2 * x^2 + 3 * y^2 - 8 * x + 12 * y + 40 = 20 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l450_45022


namespace NUMINAMATH_GPT_amount_received_is_500_l450_45089

-- Define the conditions
def books_per_month : ℕ := 3
def months_per_year : ℕ := 12
def price_per_book : ℕ := 20
def loss : ℕ := 220

-- Calculate number of books bought in a year
def books_per_year : ℕ := books_per_month * months_per_year

-- Calculate total amount spent on books in a year
def total_spent : ℕ := books_per_year * price_per_book

-- Calculate the amount Jack got from selling the books based on the given loss
def amount_received : ℕ := total_spent - loss

-- Proving the amount received is $500
theorem amount_received_is_500 : amount_received = 500 := by
  sorry

end NUMINAMATH_GPT_amount_received_is_500_l450_45089


namespace NUMINAMATH_GPT_class_has_24_students_l450_45029

theorem class_has_24_students (n S : ℕ) 
  (h1 : (S - 91 + 19) / n = 87)
  (h2 : S / n = 90) : 
  n = 24 :=
by sorry

end NUMINAMATH_GPT_class_has_24_students_l450_45029


namespace NUMINAMATH_GPT_solve_eq_l450_45099

theorem solve_eq {x : ℝ} (h : x * (x - 1) = x) : x = 0 ∨ x = 2 := 
by {
    sorry
}

end NUMINAMATH_GPT_solve_eq_l450_45099


namespace NUMINAMATH_GPT_binom_eq_fraction_l450_45042

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_binom_eq_fraction_l450_45042


namespace NUMINAMATH_GPT_simplify_expr_correct_l450_45006

-- Define the expression
def simplify_expr (z : ℝ) : ℝ := (3 - 5 * z^2) - (5 + 7 * z^2)

-- Prove the simplified form
theorem simplify_expr_correct (z : ℝ) : simplify_expr z = -2 - 12 * z^2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_correct_l450_45006


namespace NUMINAMATH_GPT_sufficient_not_necessary_example_l450_45050

lemma sufficient_but_not_necessary_condition (x y : ℝ) (hx : x >= 2) (hy : y >= 2) : x^2 + y^2 >= 4 :=
by
  -- We only need to state the lemma, so the proof is omitted.
  sorry

theorem sufficient_not_necessary_example :
  ¬(∀ x y : ℝ, (x^2 + y^2 >= 4) -> (x >= 2) ∧ (y >= 2)) :=
by 
  -- We only need to state the theorem, so the proof is omitted.
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_example_l450_45050


namespace NUMINAMATH_GPT_parity_of_expression_l450_45065

theorem parity_of_expression (a b c : ℕ) (ha : a % 2 = 1) (hb : b % 2 = 0) :
  (3 ^ a + (b - 1) ^ 2 * (c + 1)) % 2 = if c % 2 = 0 then 1 else 0 :=
by
  sorry

end NUMINAMATH_GPT_parity_of_expression_l450_45065


namespace NUMINAMATH_GPT_simplify_fraction_mul_l450_45088

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : a = 210) (h2 : b = 7350) (h3 : c = 1) (h4 : d = 35) (h5 : 210 / gcd 210 7350 = 1) (h6: 7350 / gcd 210 7350 = 35) :
  (a / b) * 14 = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_mul_l450_45088


namespace NUMINAMATH_GPT_complex_square_l450_45092

theorem complex_square (a b : ℤ) (i : ℂ) (h1: a = 5) (h2: b = 3) (h3: i^2 = -1) :
  ((↑a) + (↑b) * i)^2 = 16 + 30 * i := by
  sorry

end NUMINAMATH_GPT_complex_square_l450_45092


namespace NUMINAMATH_GPT_pow_modulus_l450_45055

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end NUMINAMATH_GPT_pow_modulus_l450_45055


namespace NUMINAMATH_GPT_number_of_members_is_44_l450_45058

-- Define necessary parameters and conditions
def paise_per_rupee : Nat := 100

def total_collection_in_paise : Nat := 1936

def number_of_members_in_group (n : Nat) : Prop :=
  n * n = total_collection_in_paise

-- Proposition to prove
theorem number_of_members_is_44 : number_of_members_in_group 44 :=
by
  sorry

end NUMINAMATH_GPT_number_of_members_is_44_l450_45058


namespace NUMINAMATH_GPT_dictionary_prices_and_max_A_l450_45005

-- Definitions for the problem
def price_A := 70
def price_B := 50

-- Conditions from the problem
def condition1 := (price_A + 2 * price_B = 170)
def condition2 := (2 * price_A + 3 * price_B = 290)

-- The proof problem statement
theorem dictionary_prices_and_max_A (h1 : price_A + 2 * price_B = 170) (h2 : 2 * price_A + 3 * price_B = 290) :
  price_A = 70 ∧ price_B = 50 ∧ (∀ (x y : ℕ), x + y = 30 → 70 * x + 50 * y ≤ 1600 → x ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_dictionary_prices_and_max_A_l450_45005


namespace NUMINAMATH_GPT_sum_primes_less_than_20_l450_45012

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end NUMINAMATH_GPT_sum_primes_less_than_20_l450_45012


namespace NUMINAMATH_GPT_roots_eq_solution_l450_45043

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end NUMINAMATH_GPT_roots_eq_solution_l450_45043


namespace NUMINAMATH_GPT_discount_difference_l450_45059

open Real

noncomputable def single_discount (B : ℝ) (d1 : ℝ) : ℝ :=
  B * (1 - d1)

noncomputable def successive_discounts (B : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (B * (1 - d2)) * (1 - d3)

theorem discount_difference (B : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  B = 12000 →
  d1 = 0.30 →
  d2 = 0.25 →
  d3 = 0.05 →
  abs (single_discount B d1 - successive_discounts B d2 d3) = 150 := by
  intros h_B h_d1 h_d2 h_d3
  rw [h_B, h_d1, h_d2, h_d3]
  rw [single_discount, successive_discounts]
  sorry

end NUMINAMATH_GPT_discount_difference_l450_45059


namespace NUMINAMATH_GPT_smallest_factorization_c_l450_45030

theorem smallest_factorization_c : ∃ (c : ℤ), (∀ (r s : ℤ), r * s = 2016 → r + s = c) ∧ c > 0 ∧ c = 108 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_factorization_c_l450_45030


namespace NUMINAMATH_GPT_ten_times_average_letters_l450_45036

-- Define the number of letters Elida has
def letters_Elida : ℕ := 5

-- Define the number of letters Adrianna has
def letters_Adrianna : ℕ := 2 * letters_Elida - 2

-- Define the average number of letters in both names
def average_letters : ℕ := (letters_Elida + letters_Adrianna) / 2

-- Define the final statement for 10 times the average number of letters
theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end NUMINAMATH_GPT_ten_times_average_letters_l450_45036


namespace NUMINAMATH_GPT_max_rubles_earned_l450_45067

theorem max_rubles_earned :
  ∀ (cards_with_1 cards_with_2 : ℕ), 
  cards_with_1 = 2013 ∧ cards_with_2 = 2013 →
  ∃ (max_moves : ℕ), max_moves = 5 :=
by
  intros cards_with_1 cards_with_2 h
  sorry

end NUMINAMATH_GPT_max_rubles_earned_l450_45067


namespace NUMINAMATH_GPT_four_times_sum_of_squares_gt_sum_squared_l450_45070

open Real

theorem four_times_sum_of_squares_gt_sum_squared
  {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 :=
sorry

end NUMINAMATH_GPT_four_times_sum_of_squares_gt_sum_squared_l450_45070


namespace NUMINAMATH_GPT_hyperbola_foci_product_l450_45076

theorem hyperbola_foci_product
  (F1 F2 P : ℝ × ℝ)
  (hF1 : F1 = (-Real.sqrt 5, 0))
  (hF2 : F2 = (Real.sqrt 5, 0))
  (hP : P.1 ^ 2 / 4 - P.2 ^ 2 = 1)
  (hDot : (P.1 + Real.sqrt 5) * (P.1 - Real.sqrt 5) + P.2 ^ 2 = 0) :
  (Real.sqrt ((P.1 + Real.sqrt 5) ^ 2 + P.2 ^ 2)) * (Real.sqrt ((P.1 - Real.sqrt 5) ^ 2 + P.2 ^ 2)) = 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_product_l450_45076


namespace NUMINAMATH_GPT_xy_sum_l450_45051

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_xy_sum_l450_45051


namespace NUMINAMATH_GPT_find_numbers_l450_45041

theorem find_numbers :
  ∃ a b : ℕ, a + b = 60 ∧ Nat.gcd a b + Nat.lcm a b = 84 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l450_45041


namespace NUMINAMATH_GPT_find_num_terms_in_AP_l450_45052

-- Define the necessary conditions and prove the final result
theorem find_num_terms_in_AP
  (a d : ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h_last_term_difference : (n - 1 : ℝ) * d = 7.5)
  (h_sum_odd_terms : n * (a + (n - 2 : ℝ) / 2 * d) = 60)
  (h_sum_even_terms : n * (a + ((n - 1 : ℝ) / 2) * d + d) = 90) :
  n = 12 := 
sorry

end NUMINAMATH_GPT_find_num_terms_in_AP_l450_45052


namespace NUMINAMATH_GPT_negation_proof_l450_45032

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 2 * x)) ↔ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l450_45032


namespace NUMINAMATH_GPT_jim_age_l450_45090

variable (J F S : ℕ)

theorem jim_age (h1 : J = 2 * F) (h2 : F = S + 9) (h3 : J - 6 = 5 * (S - 6)) : J = 46 := 
by
  sorry

end NUMINAMATH_GPT_jim_age_l450_45090


namespace NUMINAMATH_GPT_work_days_l450_45011

theorem work_days (A B C : ℝ)
  (h1 : A + B = 1 / 20)
  (h2 : B + C = 1 / 30)
  (h3 : A + C = 1 / 30) :
  (1 / (A + B + C)) = 120 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_work_days_l450_45011


namespace NUMINAMATH_GPT_find_y_l450_45023

-- Suppose C > A > B > 0
-- and A is y% smaller than C.
-- Also, C = 2B.
-- We need to show that y = 100 - 50 * (A / B).

variable (A B C : ℝ)
variable (y : ℝ)

-- Conditions
axiom h1 : C > A
axiom h2 : A > B
axiom h3 : B > 0
axiom h4 : C = 2 * B
axiom h5 : A = (1 - y / 100) * C

-- Goal
theorem find_y : y = 100 - 50 * (A / B) :=
by
  sorry

end NUMINAMATH_GPT_find_y_l450_45023


namespace NUMINAMATH_GPT_keiko_walking_speed_l450_45001

theorem keiko_walking_speed (r : ℝ) (t : ℝ) (width : ℝ) 
   (time_diff : ℝ) (h0 : width = 8) (h1 : time_diff = 48) 
   (h2 : t = (2 * (2 * (r + 8) * Real.pi) / (r + 8) + 2 * (0 * Real.pi))) 
   (h3 : 2 * (2 * r * Real.pi) / r + 2 * (0 * Real.pi) = t - time_diff) :
   t = 48 -> 
   (v : ℝ) →
   v = (16 * Real.pi) / time_diff →
   v = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_keiko_walking_speed_l450_45001


namespace NUMINAMATH_GPT_find_parallel_line_through_P_l450_45068

noncomputable def line_parallel_passing_through (p : (ℝ × ℝ)) (line : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, _) := line
  let (x, y) := p
  (a, b, - (a * x + b * y))

theorem find_parallel_line_through_P :
  line_parallel_passing_through (4, -1) (3, -4, 6) = (3, -4, -16) :=
by 
  sorry

end NUMINAMATH_GPT_find_parallel_line_through_P_l450_45068


namespace NUMINAMATH_GPT_m_and_n_must_have_same_parity_l450_45080

-- Define the problem conditions
def square_has_four_colored_edges (square : Type) : Prop :=
  ∃ (colors : Fin 4 → square), true

def m_and_n_same_parity (m n : ℕ) : Prop :=
  (m % 2 = n % 2)

-- Formalize the proof statement based on the conditions
theorem m_and_n_must_have_same_parity (m n : ℕ) (square : Type)
  (H : square_has_four_colored_edges square) : 
  m_and_n_same_parity m n :=
by 
  sorry

end NUMINAMATH_GPT_m_and_n_must_have_same_parity_l450_45080


namespace NUMINAMATH_GPT_masha_more_cakes_l450_45045

theorem masha_more_cakes (S : ℝ) (m n : ℝ) (H1 : S > 0) (H2 : m > 0) (H3 : n > 0) 
  (H4 : 2 * S * (m + n) ≤ S * m + (1/3) * S * n) :
  m > n := 
by 
  sorry

end NUMINAMATH_GPT_masha_more_cakes_l450_45045


namespace NUMINAMATH_GPT_trajectory_of_M_ellipse_trajectory_l450_45013

variable {x y : ℝ}

theorem trajectory_of_M (hx : x ≠ 5) (hnx : x ≠ -5)
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (2 * x^2 + y^2 = 50) :=
by
  -- Proof is omitted.
  sorry

theorem ellipse_trajectory (hx : x ≠ 5) (hnx : x ≠ -5) 
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (x^2 / 25 + y^2 / 50 = 1) :=
by
  -- Using the previous theorem to derive.
  have h1 : (2 * x^2 + y^2 = 50) := trajectory_of_M hx hnx h
  -- Proof of transformation is omitted.
  sorry

end NUMINAMATH_GPT_trajectory_of_M_ellipse_trajectory_l450_45013


namespace NUMINAMATH_GPT_length_squared_t_graph_interval_l450_45066

noncomputable def p (x : ℝ) : ℝ := -x + 2
noncomputable def q (x : ℝ) : ℝ := x + 2
noncomputable def r (x : ℝ) : ℝ := 2
noncomputable def t (x : ℝ) : ℝ :=
  if x ≤ -2 then p x
  else if x ≤ 2 then r x
  else q x

theorem length_squared_t_graph_interval :
  let segment_length (f : ℝ → ℝ) (a b : ℝ) : ℝ := Real.sqrt ((f b - f a)^2 + (b - a)^2)
  segment_length t (-4) (-2) + segment_length t (-2) 2 + segment_length t 2 4 = 4 + 2 * Real.sqrt 32 →
  (4 + 2 * Real.sqrt 32)^2 = 80 :=
sorry

end NUMINAMATH_GPT_length_squared_t_graph_interval_l450_45066


namespace NUMINAMATH_GPT_root_exists_in_interval_l450_45007

noncomputable def f (x : ℝ) := (1 / 2) ^ x - x + 1

theorem root_exists_in_interval :
  (0 < f 1) ∧ (f 1.5 < 0) ∧ (f 2 < 0) ∧ (f 3 < 0) → ∃ x, 1 < x ∧ x < 1.5 ∧ f x = 0 :=
by
  -- use the intermediate value theorem and bisection method here
  sorry

end NUMINAMATH_GPT_root_exists_in_interval_l450_45007


namespace NUMINAMATH_GPT_minimum_a_l450_45039

theorem minimum_a (x : ℝ) (h : ∀ x ≥ 0, x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a) : 
    a ≥ -1 := by
  sorry

end NUMINAMATH_GPT_minimum_a_l450_45039


namespace NUMINAMATH_GPT_quadratic_points_relationship_l450_45014

theorem quadratic_points_relationship (c y1 y2 y3 : ℝ) 
  (hA : y1 = (-3)^2 + 2*(-3) + c)
  (hB : y2 = (1/2)^2 + 2*(1/2) + c)
  (hC : y3 = 2^2 + 2*2 + c) : y2 < y1 ∧ y1 < y3 := 
sorry

end NUMINAMATH_GPT_quadratic_points_relationship_l450_45014


namespace NUMINAMATH_GPT_books_got_rid_of_l450_45081

-- Define the number of books they originally had
def original_books : ℕ := 87

-- Define the number of shelves used
def shelves_used : ℕ := 9

-- Define the number of books per shelf
def books_per_shelf : ℕ := 6

-- Define the number of books left after placing them on shelves
def remaining_books : ℕ := shelves_used * books_per_shelf

-- The statement to prove
theorem books_got_rid_of : original_books - remaining_books = 33 := 
by 
-- here is proof body you need to fill in 
  sorry

end NUMINAMATH_GPT_books_got_rid_of_l450_45081


namespace NUMINAMATH_GPT_magnitude_of_power_l450_45083

-- Given conditions
def z : ℂ := 3 + 2 * Complex.I
def n : ℕ := 6

-- Mathematical statement to prove
theorem magnitude_of_power :
  Complex.abs (z ^ n) = 2197 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_power_l450_45083


namespace NUMINAMATH_GPT_cube_point_problem_l450_45020
open Int

theorem cube_point_problem (n : ℤ) (x y z u : ℤ)
  (hx : x = 0 ∨ x = 8)
  (hy : y = 0 ∨ y = 12)
  (hz : z = 0 ∨ z = 6)
  (hu : 24 ∣ u)
  (hn : n = x + y + z + u) :
  (n ≠ 100) ∧ (n = 200) ↔ (n % 6 = 0 ∨ (n - 8) % 6 = 0) :=
by sorry

end NUMINAMATH_GPT_cube_point_problem_l450_45020


namespace NUMINAMATH_GPT_value_of_m_l450_45082

-- Define the condition of the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m

-- State the equivalence to be proved
theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 1 ∧ quadratic_equation x m = 0) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l450_45082


namespace NUMINAMATH_GPT_min_detectors_correct_l450_45025

noncomputable def min_detectors (M N : ℕ) : ℕ :=
  ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊

theorem min_detectors_correct (M N : ℕ) (hM : 2 ≤ M) (hN : 2 ≤ N) :
  min_detectors M N = ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊ :=
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_min_detectors_correct_l450_45025


namespace NUMINAMATH_GPT_determinant_real_root_unique_l450_45015

theorem determinant_real_root_unique {a b c : ℝ} (ha : 0 < a ∧ a ≠ 1) (hb : 0 < b ∧ b ≠ 1) (hc : 0 < c ∧ c ≠ 1) :
  ∃! x : ℝ, (Matrix.det ![
    ![x - 1, c - 1, -(b - 1)],
    ![-(c - 1), x - 1, a - 1],
    ![b - 1, -(a - 1), x - 1]
  ]) = 0 :=
by
  sorry

end NUMINAMATH_GPT_determinant_real_root_unique_l450_45015


namespace NUMINAMATH_GPT_volume_of_circumscribed_sphere_of_cube_l450_45074

theorem volume_of_circumscribed_sphere_of_cube (a : ℝ) (h : a = 1) : 
  (4 / 3) * Real.pi * ((Real.sqrt 3 / 2) ^ 3) = (Real.sqrt 3 / 2) * Real.pi :=
by sorry

end NUMINAMATH_GPT_volume_of_circumscribed_sphere_of_cube_l450_45074


namespace NUMINAMATH_GPT_no_distinct_roots_exist_l450_45098

theorem no_distinct_roots_exist :
  ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a^2 - 2 * b * a + c^2 = 0) ∧
  (b^2 - 2 * c * b + a^2 = 0) ∧ 
  (c^2 - 2 * a * c + b^2 = 0) := 
sorry

end NUMINAMATH_GPT_no_distinct_roots_exist_l450_45098


namespace NUMINAMATH_GPT_man_speed_approx_l450_45046

noncomputable def speed_of_man : ℝ :=
  let L := 700    -- Length of the train in meters
  let u := 63 / 3.6  -- Speed of the train in meters per second (converted)
  let t := 41.9966402687785 -- Time taken to cross the man in seconds
  let v := (u * t - L) / t  -- Speed of the man
  v

-- The main theorem to prove that the speed of the man is approximately 0.834 m/s.
theorem man_speed_approx : abs (speed_of_man - 0.834) < 1e-3 :=
by
  -- Simplification and exact calculations will be handled by the Lean prover or could be manually done.
  sorry

end NUMINAMATH_GPT_man_speed_approx_l450_45046


namespace NUMINAMATH_GPT_find_x_l450_45028

-- Define the percentages and multipliers as constants
def percent_47 := 47.0 / 100.0
def percent_36 := 36.0 / 100.0

-- Define the given quantities
def quantity1 := 1442.0
def quantity2 := 1412.0

-- Calculate the percentages of the quantities
def part1 := percent_47 * quantity1
def part2 := percent_36 * quantity2

-- Calculate the expression
def expression := (part1 - part2) + 63.0

-- Define the value of x given
def x := 232.42

-- Theorem stating the proof problem
theorem find_x : expression = x := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_x_l450_45028


namespace NUMINAMATH_GPT_count_primes_5p2p1_minus_1_perfect_square_l450_45004

-- Define the predicate for a prime number
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate for perfect square
def is_perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

-- The main theorem statement
theorem count_primes_5p2p1_minus_1_perfect_square :
  (∀ p : ℕ, is_prime p → is_perfect_square (5 * p * (2^(p + 1) - 1))) → ∃! p : ℕ, is_prime p ∧ is_perfect_square (5 * p * (2^(p + 1) - 1)) :=
sorry

end NUMINAMATH_GPT_count_primes_5p2p1_minus_1_perfect_square_l450_45004


namespace NUMINAMATH_GPT_bill_can_buy_donuts_in_35_ways_l450_45016

def different_ways_to_buy_donuts : ℕ :=
  5 + 20 + 10  -- Number of ways to satisfy the conditions

theorem bill_can_buy_donuts_in_35_ways :
  different_ways_to_buy_donuts = 35 :=
by
  -- Proof steps
  -- The problem statement and the solution show the calculation to be correct.
  sorry

end NUMINAMATH_GPT_bill_can_buy_donuts_in_35_ways_l450_45016


namespace NUMINAMATH_GPT_towel_area_decrease_l450_45049

theorem towel_area_decrease (L B : ℝ) :
  let A_original := L * B
  let L_new := 0.8 * L
  let B_new := 0.9 * B
  let A_new := L_new * B_new
  let percentage_decrease := ((A_original - A_new) / A_original) * 100
  percentage_decrease = 28 := 
by
  sorry

end NUMINAMATH_GPT_towel_area_decrease_l450_45049


namespace NUMINAMATH_GPT_number_of_second_graders_l450_45063

-- Define the number of kindergartners, first graders, and total students
def k : ℕ := 14
def f : ℕ := 24
def t : ℕ := 42

-- Define the number of second graders
def s : ℕ := t - (k + f)

-- The theorem to prove
theorem number_of_second_graders : s = 4 := by
  -- We can use sorry here since we are not required to provide the proof
  sorry

end NUMINAMATH_GPT_number_of_second_graders_l450_45063


namespace NUMINAMATH_GPT_estimate_fish_number_l450_45038

noncomputable def numFishInLake (marked: ℕ) (caughtSecond: ℕ) (markedSecond: ℕ) : ℕ :=
  let totalFish := (caughtSecond * marked) / markedSecond
  totalFish

theorem estimate_fish_number (marked caughtSecond markedSecond : ℕ) :
  marked = 100 ∧ caughtSecond = 200 ∧ markedSecond = 25 → numFishInLake marked caughtSecond markedSecond = 800 :=
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_estimate_fish_number_l450_45038


namespace NUMINAMATH_GPT_find_people_and_carriages_l450_45057

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end NUMINAMATH_GPT_find_people_and_carriages_l450_45057


namespace NUMINAMATH_GPT_polynomial_remainder_l450_45084

theorem polynomial_remainder (y : ℝ) : 
  let a := 3 ^ 50 - 2 ^ 50
  let b := 2 ^ 50 - 2 * 3 ^ 50 + 2 ^ 51
  (y ^ 50) % (y ^ 2 - 5 * y + 6) = a * y + b :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l450_45084


namespace NUMINAMATH_GPT_mass_percentage_O_in_mixture_l450_45096

/-- Mass percentage of oxygen in a mixture of Acetone and Methanol -/
theorem mass_percentage_O_in_mixture 
  (mass_acetone: ℝ)
  (mass_methanol: ℝ)
  (mass_O_acetone: ℝ)
  (mass_O_methanol: ℝ) 
  (total_mass: ℝ) : 
  mass_acetone = 30 → 
  mass_methanol = 20 → 
  mass_O_acetone = (16 / 58.08) * 30 →
  mass_O_methanol = (16 / 32.04) * 20 →
  total_mass = mass_acetone + mass_methanol →
  ((mass_O_acetone + mass_O_methanol) / total_mass) * 100 = 36.52 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_mixture_l450_45096


namespace NUMINAMATH_GPT_percentage_decrease_l450_45026

theorem percentage_decrease (x y : ℝ) : 
  (xy^2 - (0.7 * x) * (0.6 * y)^2) / xy^2 = 0.748 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l450_45026


namespace NUMINAMATH_GPT_ripe_oranges_l450_45008

theorem ripe_oranges (U : ℕ) (hU : U = 25) (hR : R = U + 19) : R = 44 := by
  sorry

end NUMINAMATH_GPT_ripe_oranges_l450_45008


namespace NUMINAMATH_GPT_fraction_of_total_money_l450_45060

variable (Max Leevi Nolan Ollie : ℚ)

-- Condition: Each of Max, Leevi, and Nolan gave Ollie the same amount of money
variable (x : ℚ) (h1 : Max / 6 = x) (h2 : Leevi / 3 = x) (h3 : Nolan / 2 = x)

-- Proving that the fraction of the group's (Max, Leevi, Nolan, Ollie) total money possessed by Ollie is 3/11.
theorem fraction_of_total_money (h4 : Max + Leevi + Nolan + Ollie = Max + Leevi + Nolan + 3 * x) : 
  x / (Max + Leevi + Nolan + x) = 3 / 11 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_total_money_l450_45060


namespace NUMINAMATH_GPT_Adam_spent_21_dollars_l450_45009

-- Define the conditions as given in the problem
def initial_money : ℕ := 91
def spent_money (x : ℕ) : Prop := (initial_money - x) * 3 = 10 * x

-- The theorem we want to prove: Adam spent 21 dollars on new books
theorem Adam_spent_21_dollars : spent_money 21 :=
by sorry

end NUMINAMATH_GPT_Adam_spent_21_dollars_l450_45009


namespace NUMINAMATH_GPT_hair_cut_length_l450_45064

-- Definitions corresponding to the conditions in the problem
def initial_length : ℕ := 18
def current_length : ℕ := 9

-- Statement to prove
theorem hair_cut_length : initial_length - current_length = 9 :=
by
  sorry

end NUMINAMATH_GPT_hair_cut_length_l450_45064


namespace NUMINAMATH_GPT_distance_ratio_gt_9_l450_45017

theorem distance_ratio_gt_9 (points : Fin 1997 → ℝ × ℝ × ℝ) (M m : ℝ) :
  (∀ i j, i ≠ j → dist (points i) (points j) ≤ M) →
  (∀ i j, i ≠ j → dist (points i) (points j) ≥ m) →
  m ≠ 0 →
  M / m > 9 :=
by
  sorry

end NUMINAMATH_GPT_distance_ratio_gt_9_l450_45017


namespace NUMINAMATH_GPT_parallel_lines_d_l450_45054

theorem parallel_lines_d (d : ℝ) : (∀ x : ℝ, -3 * x + 5 = (-6 * d) * x + 10) → d = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_d_l450_45054


namespace NUMINAMATH_GPT_determine_a_l450_45056

theorem determine_a (a b c : ℤ)
  (vertex_condition : ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = -3 → y = a * (x - 2) ^ 2 - 3)
  (point_condition : ∀ x : ℝ, x = 1 → ∀ y : ℝ, y = -2 → y = a * (x - 2) ^ 2 - 3) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l450_45056


namespace NUMINAMATH_GPT_molecular_weight_CaO_is_56_l450_45033

def atomic_weight_Ca : ℕ := 40
def atomic_weight_O : ℕ := 16
def molecular_weight_CaO : ℕ := atomic_weight_Ca + atomic_weight_O

theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = 56 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_CaO_is_56_l450_45033


namespace NUMINAMATH_GPT_unique_solution_sin_tan_eq_l450_45094

noncomputable def S (x : ℝ) : ℝ := Real.tan (Real.sin x) - Real.sin x

theorem unique_solution_sin_tan_eq (h : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.arcsin (1/2) → S x < S y) :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) := by
sorry

end NUMINAMATH_GPT_unique_solution_sin_tan_eq_l450_45094


namespace NUMINAMATH_GPT_min_value_of_quadratic_l450_45097

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 2 ∧ (∀ y : ℝ, quadratic y ≥ 2) :=
by
  use 4
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l450_45097


namespace NUMINAMATH_GPT_set_membership_l450_45018

theorem set_membership :
  {m : ℤ | ∃ k : ℤ, 10 = k * (m + 1)} = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by sorry

end NUMINAMATH_GPT_set_membership_l450_45018


namespace NUMINAMATH_GPT_tangent_parabola_points_l450_45019

theorem tangent_parabola_points (a b : ℝ) (h_circle : a^2 + b^2 = 1) (h_discriminant : a^2 - 4 * b * (b - 1) = 0) :
    (a = 0 ∧ b = 1) ∨ 
    (a = 2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) ∨ 
    (a = -2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) := sorry

end NUMINAMATH_GPT_tangent_parabola_points_l450_45019


namespace NUMINAMATH_GPT_olivia_paper_count_l450_45010

-- State the problem conditions and the final proof statement.
theorem olivia_paper_count :
  let math_initial := 220
  let science_initial := 150
  let math_used := 95
  let science_used := 68
  let math_received := 30
  let science_given := 15
  let math_remaining := math_initial - math_used + math_received
  let science_remaining := science_initial - science_used - science_given
  let total_pieces := math_remaining + science_remaining
  total_pieces = 222 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_olivia_paper_count_l450_45010


namespace NUMINAMATH_GPT_sets_equality_l450_45037

variables {α : Type*} (A B C : Set α)

theorem sets_equality (h1 : A ∪ B ⊆ C) (h2 : A ∪ C ⊆ B) (h3 : B ∪ C ⊆ A) : A = B ∧ B = C :=
by
  sorry

end NUMINAMATH_GPT_sets_equality_l450_45037


namespace NUMINAMATH_GPT_library_visitors_total_l450_45000

theorem library_visitors_total
  (visitors_monday : ℕ)
  (visitors_tuesday : ℕ)
  (average_visitors_remaining_days : ℕ)
  (remaining_days : ℕ)
  (total_visitors : ℕ)
  (hmonday : visitors_monday = 50)
  (htuesday : visitors_tuesday = 2 * visitors_monday)
  (haverage : average_visitors_remaining_days = 20)
  (hremaining_days : remaining_days = 5)
  (htotal : total_visitors =
    visitors_monday + visitors_tuesday + remaining_days * average_visitors_remaining_days) :
  total_visitors = 250 :=
by
  -- here goes the proof, marked as sorry for now
  sorry

end NUMINAMATH_GPT_library_visitors_total_l450_45000


namespace NUMINAMATH_GPT_maggie_fraction_caught_l450_45077

theorem maggie_fraction_caught :
  let total_goldfish := 100
  let allowed_to_take_home := total_goldfish / 2
  let remaining_goldfish_to_catch := 20
  let goldfish_caught := allowed_to_take_home - remaining_goldfish_to_catch
  (goldfish_caught / allowed_to_take_home : ℚ) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_maggie_fraction_caught_l450_45077


namespace NUMINAMATH_GPT_train_speed_kmph_l450_45091

/-- Define the lengths of the train and bridge, as well as the time taken to cross the bridge. --/
def train_length : ℝ := 150
def bridge_length : ℝ := 150
def crossing_time_seconds : ℝ := 29.997600191984642

/-- Calculate the speed of the train in km/h. --/
theorem train_speed_kmph : 
  let total_distance := train_length + bridge_length
  let time_in_hours := crossing_time_seconds / 3600
  let speed_mph := total_distance / time_in_hours
  let speed_kmph := speed_mph / 1000
  speed_kmph = 36 := by
  /- Proof omitted -/
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l450_45091


namespace NUMINAMATH_GPT_income_left_at_end_of_year_l450_45040

variable (I : ℝ) -- Monthly income at the beginning of the year
variable (food_expense : ℝ := 0.35 * I) 
variable (education_expense : ℝ := 0.25 * I)
variable (transportation_expense : ℝ := 0.15 * I)
variable (medical_expense : ℝ := 0.10 * I)
variable (initial_expenses : ℝ := food_expense + education_expense + transportation_expense + medical_expense)
variable (remaining_income : ℝ := I - initial_expenses)
variable (house_rent : ℝ := 0.80 * remaining_income)

variable (annual_income : ℝ := 12 * I)
variable (annual_expenses : ℝ := 12 * (initial_expenses + house_rent))

variable (increased_food_expense : ℝ := food_expense * 1.05)
variable (increased_education_expense : ℝ := education_expense * 1.05)
variable (increased_transportation_expense : ℝ := transportation_expense * 1.05)
variable (increased_medical_expense : ℝ := medical_expense * 1.05)
variable (total_increased_expenses : ℝ := increased_food_expense + increased_education_expense + increased_transportation_expense + increased_medical_expense)

variable (new_income : ℝ := 1.10 * I)
variable (new_remaining_income : ℝ := new_income - total_increased_expenses)

variable (new_house_rent : ℝ := 0.80 * new_remaining_income)

variable (final_remaining_income : ℝ := new_income - (total_increased_expenses + new_house_rent))

theorem income_left_at_end_of_year : 
  final_remaining_income / new_income * 100 = 2.15 := 
  sorry

end NUMINAMATH_GPT_income_left_at_end_of_year_l450_45040
