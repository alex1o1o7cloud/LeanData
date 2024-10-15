import Mathlib

namespace NUMINAMATH_GPT_mike_spent_total_l2134_213448

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end NUMINAMATH_GPT_mike_spent_total_l2134_213448


namespace NUMINAMATH_GPT_wine_count_l2134_213404

theorem wine_count (S B total W : ℕ) (hS : S = 22) (hB : B = 17) (htotal : S - B + W = total) (htotal_val : total = 31) : W = 26 :=
by
  sorry

end NUMINAMATH_GPT_wine_count_l2134_213404


namespace NUMINAMATH_GPT_find_n_l2134_213481

noncomputable def arithmetic_sequence (a : ℕ → ℕ) := 
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_n (a : ℕ → ℕ) (n d : ℕ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : a 2 + a 5 = 12)
  (h4 : a n = 25) : 
  n = 13 := 
sorry

end NUMINAMATH_GPT_find_n_l2134_213481


namespace NUMINAMATH_GPT_max_a_value_l2134_213412

def f (a x : ℝ) : ℝ := x^3 - a*x^2 + (a^2 - 2)*x + 1

theorem max_a_value (a : ℝ) :
  (∃ m : ℝ, m > 0 ∧ f a m ≤ 0) → a ≤ 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_max_a_value_l2134_213412


namespace NUMINAMATH_GPT_truck_wheels_l2134_213440

theorem truck_wheels (t x : ℝ) (wheels_front : ℕ) (wheels_other : ℕ) :
  (t = 1.50 + 1.50 * (x - 2)) → (t = 6) → (wheels_front = 2) → (wheels_other = 4) → x = 5 → 
  (wheels_front + wheels_other * (x - 1) = 18) :=
by
  intros h1 h2 h3 h4 h5
  rw [h5] at *
  sorry

end NUMINAMATH_GPT_truck_wheels_l2134_213440


namespace NUMINAMATH_GPT_fraction_of_teeth_removed_l2134_213416

theorem fraction_of_teeth_removed
  (total_teeth : ℕ)
  (initial_teeth : ℕ)
  (second_fraction : ℚ)
  (third_fraction : ℚ)
  (second_removed : ℕ)
  (third_removed : ℕ)
  (fourth_removed : ℕ)
  (total_removed : ℕ)
  (first_removed : ℕ)
  (fraction_first_removed : ℚ) :
  total_teeth = 32 →
  initial_teeth = 32 →
  second_fraction = 3 / 8 →
  third_fraction = 1 / 2 →
  second_removed = 12 →
  third_removed = 16 →
  fourth_removed = 4 →
  total_removed = 40 →
  first_removed + second_removed + third_removed + fourth_removed = total_removed →
  first_removed = 8 →
  fraction_first_removed = first_removed / initial_teeth →
  fraction_first_removed = 1 / 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end NUMINAMATH_GPT_fraction_of_teeth_removed_l2134_213416


namespace NUMINAMATH_GPT_max_ab_min_fraction_l2134_213490

-- Question 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : ab ≤ 25/21 := sorry

-- Question 2: Minimum value of (3/a + 7/b)
theorem min_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : 3/a + 7/b ≥ 10 := sorry

end NUMINAMATH_GPT_max_ab_min_fraction_l2134_213490


namespace NUMINAMATH_GPT_geometric_sequence_eighth_term_l2134_213459

theorem geometric_sequence_eighth_term (a r : ℝ) (h1 : a * r ^ 3 = 12) (h2 : a * r ^ 11 = 3) : 
  a * r ^ 7 = 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_eighth_term_l2134_213459


namespace NUMINAMATH_GPT_draw_contains_chinese_book_l2134_213493

theorem draw_contains_chinese_book
  (total_books : ℕ)
  (chinese_books : ℕ)
  (math_books : ℕ)
  (drawn_books : ℕ)
  (h_total : total_books = 12)
  (h_chinese : chinese_books = 10)
  (h_math : math_books = 2)
  (h_drawn : drawn_books = 3) :
  ∃ n, n ≥ 1 ∧ n ≤ drawn_books ∧ n * (chinese_books/total_books) > 1 := 
  sorry

end NUMINAMATH_GPT_draw_contains_chinese_book_l2134_213493


namespace NUMINAMATH_GPT_no_linear_term_in_product_l2134_213468

theorem no_linear_term_in_product (a : ℝ) (h : ∀ x : ℝ, (x + 4) * (x + a) - x^2 - 4 * a = 0) : a = -4 :=
sorry

end NUMINAMATH_GPT_no_linear_term_in_product_l2134_213468


namespace NUMINAMATH_GPT_three_layer_rug_area_l2134_213442

theorem three_layer_rug_area 
  (A B C D : ℕ) 
  (hA : A = 350) 
  (hB : B = 250) 
  (hC : C = 45) 
  (h_formula : A = B + C + D) : 
  D = 55 :=
by
  sorry

end NUMINAMATH_GPT_three_layer_rug_area_l2134_213442


namespace NUMINAMATH_GPT_more_valley_than_humpy_l2134_213400

def is_humpy (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 > d4 ∧ d4 > d5

def is_valley (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 > d2 ∧ d2 > d3 ∧ d3 < d4 ∧ d4 < d5

def starts_with_5 (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  d1 = 5

theorem more_valley_than_humpy :
  (∃ m, starts_with_5 m ∧ is_humpy m) → (∃ n, starts_with_5 n ∧ is_valley n) ∧ 
  (∀ x, starts_with_5 x → is_humpy x → ∃ y, starts_with_5 y ∧ is_valley y ∧ y ≠ x) :=
by sorry

end NUMINAMATH_GPT_more_valley_than_humpy_l2134_213400


namespace NUMINAMATH_GPT_find_solutions_l2134_213487

def system_solutions (x y z : ℝ) : Prop :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solutions :
  ∃ (x y z : ℝ), system_solutions x y z ∧ ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_l2134_213487


namespace NUMINAMATH_GPT_total_apples_l2134_213410

def green_apples : ℕ := 2
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

theorem total_apples : green_apples + red_apples + yellow_apples = 19 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_apples_l2134_213410


namespace NUMINAMATH_GPT_provisions_last_for_girls_l2134_213451

theorem provisions_last_for_girls (P : ℝ) (G : ℝ) (h1 : P / (50 * G) = P / (250 * (G + 20))) : G = 25 := 
by
  sorry

end NUMINAMATH_GPT_provisions_last_for_girls_l2134_213451


namespace NUMINAMATH_GPT_sqrt_seven_to_six_power_eq_343_l2134_213463

theorem sqrt_seven_to_six_power_eq_343 : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_GPT_sqrt_seven_to_six_power_eq_343_l2134_213463


namespace NUMINAMATH_GPT_maximum_area_l2134_213469

-- Define necessary variables and conditions
variables (x y : ℝ)
variable (A : ℝ)
variable (peri : ℝ := 30)

-- Provide the premise that defines the perimeter condition
axiom perimeter_condition : 2 * x + 2 * y = peri

-- Define y in terms of x based on the perimeter condition
def y_in_terms_of_x (x : ℝ) : ℝ := 15 - x

-- Define the area of the rectangle in terms of x
def area (x : ℝ) : ℝ := x * (y_in_terms_of_x x)

-- The statement that needs to be proved
theorem maximum_area : A = 56.25 :=
by sorry

end NUMINAMATH_GPT_maximum_area_l2134_213469


namespace NUMINAMATH_GPT_minimal_degree_of_g_l2134_213420

noncomputable def g_degree_minimal (f g h : Polynomial ℝ) (deg_f : ℕ) (deg_h : ℕ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h) : Prop :=
  Polynomial.degree f = deg_f ∧ Polynomial.degree h = deg_h → Polynomial.degree g = 12

theorem minimal_degree_of_g (f g h : Polynomial ℝ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h)
    (deg_f : Polynomial.degree f = 5) (deg_h : Polynomial.degree h = 12) :
    Polynomial.degree g = 12 := by
  sorry

end NUMINAMATH_GPT_minimal_degree_of_g_l2134_213420


namespace NUMINAMATH_GPT_expressions_equality_l2134_213413

-- Assumptions that expressions (1) and (2) are well-defined (denominators are non-zero)
variable {a b c m n p : ℝ}
variable (h1 : m ≠ 0)
variable (h2 : bp + cn ≠ 0)
variable (h3 : n ≠ 0)
variable (h4 : ap + cm ≠ 0)

-- Main theorem statement
theorem expressions_equality
  (hS : (a / m) + (bc + np) / (bp + cn) = 0) :
  (b / n) + (ac + mp) / (ap + cm) = 0 :=
  sorry

end NUMINAMATH_GPT_expressions_equality_l2134_213413


namespace NUMINAMATH_GPT_speed_of_man_is_approx_4_99_l2134_213479

noncomputable def train_length : ℝ := 110  -- meters
noncomputable def train_speed : ℝ := 50  -- km/h
noncomputable def time_to_pass_man : ℝ := 7.2  -- seconds

def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

noncomputable def relative_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def relative_speed_kmph : ℝ :=
  mps_to_kmph (relative_speed train_length time_to_pass_man)

noncomputable def speed_of_man (relative_speed_kmph : ℝ) (train_speed : ℝ) : ℝ :=
  relative_speed_kmph - train_speed

theorem speed_of_man_is_approx_4_99 :
  abs (speed_of_man relative_speed_kmph train_speed - 4.99) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_is_approx_4_99_l2134_213479


namespace NUMINAMATH_GPT_excircle_opposite_side_b_l2134_213464

-- Definition of the terms and assumptions
variables {a b c : ℝ} -- sides of the triangle
variables {r r1 : ℝ}  -- radii of the circles

-- Given conditions
def touches_side_c_and_extensions_of_a_b (r : ℝ) (a b c : ℝ) : Prop :=
  r = (a + b + c) / 2

-- The goal to be proved
theorem excircle_opposite_side_b (a b c : ℝ) (r1 : ℝ) (h1 : touches_side_c_and_extensions_of_a_b r a b c) :
  r1 = (a + c - b) / 2 := 
by
  sorry

end NUMINAMATH_GPT_excircle_opposite_side_b_l2134_213464


namespace NUMINAMATH_GPT_unique_parallel_line_in_beta_l2134_213461

-- Define the basic geometrical entities.
axiom Plane : Type
axiom Line : Type
axiom Point : Type

-- Definitions relating entities.
def contains (P : Plane) (l : Line) : Prop := sorry
def parallel (A B : Plane) : Prop := sorry
def in_plane (p : Point) (P : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry

-- Statements derived from the conditions in problem.
variables (α β : Plane) (a : Line) (B : Point)
-- Given conditions
axiom plane_parallel : parallel α β
axiom line_in_plane : contains α a
axiom point_in_plane : in_plane B β

-- The ultimate goal derived from the question.
theorem unique_parallel_line_in_beta : 
  ∃! b : Line, (in_plane B β) ∧ (parallel_lines a b) :=
sorry

end NUMINAMATH_GPT_unique_parallel_line_in_beta_l2134_213461


namespace NUMINAMATH_GPT_mean_cat_weights_l2134_213483

-- Define a list representing the weights of the cats from the stem-and-leaf plot
def cat_weights : List ℕ := [12, 13, 14, 20, 21, 21, 25, 25, 28, 30, 31, 32, 32, 36, 38, 39, 39]

-- Function to calculate the sum of elements in a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Function to calculate the mean of a list of natural numbers
def mean_list (l : List ℕ) : ℚ := (sum_list l : ℚ) / l.length

-- The theorem we need to prove
theorem mean_cat_weights : mean_list cat_weights = 27 := by 
  sorry

end NUMINAMATH_GPT_mean_cat_weights_l2134_213483


namespace NUMINAMATH_GPT_square_perimeter_l2134_213441

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end NUMINAMATH_GPT_square_perimeter_l2134_213441


namespace NUMINAMATH_GPT_negation_of_p_l2134_213424

-- Define the proposition p
def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 > 0

-- State the theorem: the negation of proposition p
theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_p_l2134_213424


namespace NUMINAMATH_GPT_probability_of_winning_quiz_l2134_213485

theorem probability_of_winning_quiz :
  let n := 4 -- number of questions
  let choices := 3 -- number of choices per question
  let probability_correct := 1 / choices -- probability of answering correctly
  let probability_incorrect := 1 - probability_correct -- probability of answering incorrectly
  let probability_all_correct := probability_correct^n -- probability of getting all questions correct
  let probability_exactly_three_correct := 4 * probability_correct^3 * probability_incorrect -- probability of getting exactly 3 questions correct
  probability_all_correct + probability_exactly_three_correct = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_quiz_l2134_213485


namespace NUMINAMATH_GPT_female_democrats_count_l2134_213460

-- Define the parameters and conditions
variables (F M D_f D_m D_total : ℕ)
variables (h1 : F + M = 840)
variables (h2 : D_total = 1/3 * (F + M))
variables (h3 : D_f = 1/2 * F)
variables (h4 : D_m = 1/4 * M)
variables (h5 : D_total = D_f + D_m)

-- State the theorem
theorem female_democrats_count : D_f = 140 :=
by
  sorry

end NUMINAMATH_GPT_female_democrats_count_l2134_213460


namespace NUMINAMATH_GPT_part1_l2134_213472

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end NUMINAMATH_GPT_part1_l2134_213472


namespace NUMINAMATH_GPT_find_b_l2134_213443

noncomputable def point (x y : Float) : Float × Float := (x, y)

def line_y_eq_b_plus_x (b x : Float) : Float := b + x

def intersects_y_axis (b : Float) : Float × Float := (0, b)

def intersects_x_axis (b : Float) : Float × Float := (-b, 0)

def intersects_x_eq_5 (b : Float) : Float × Float := (5, b + 5)

def area_triangle_qrs (b : Float) : Float :=
  0.5 * (5 + b) * (b + 5)

def area_triangle_qop (b : Float) : Float :=
  0.5 * b * b

theorem find_b (b : Float) (h : b > 0) (h_area_ratio : area_triangle_qrs b / area_triangle_qop b = 4 / 9) : b = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2134_213443


namespace NUMINAMATH_GPT_prob_four_children_at_least_one_boy_one_girl_l2134_213402

-- Define the probability of a single birth being a boy or a girl
def prob_boy_or_girl : ℚ := 1/2

-- Calculate the probability of all children being boys or all girls
def prob_all_boys : ℚ := (prob_boy_or_girl)^4
def prob_all_girls : ℚ := (prob_boy_or_girl)^4

-- Calculate the probability of having neither all boys nor all girls
def prob_at_least_one_boy_one_girl : ℚ := 1 - (prob_all_boys + prob_all_girls)

-- The theorem to prove
theorem prob_four_children_at_least_one_boy_one_girl : 
  prob_at_least_one_boy_one_girl = 7/8 := 
by 
  sorry

end NUMINAMATH_GPT_prob_four_children_at_least_one_boy_one_girl_l2134_213402


namespace NUMINAMATH_GPT_number_of_ways_to_place_letters_l2134_213499

-- Define the number of letters and mailboxes
def num_letters : Nat := 3
def num_mailboxes : Nat := 5

-- Define the function to calculate the number of ways to place the letters into mailboxes
def count_ways (n : Nat) (m : Nat) : Nat := m ^ n

-- The theorem to prove
theorem number_of_ways_to_place_letters :
  count_ways num_letters num_mailboxes = 5 ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_place_letters_l2134_213499


namespace NUMINAMATH_GPT_vehicle_value_last_year_l2134_213439

theorem vehicle_value_last_year (value_this_year : ℝ) (ratio : ℝ) (value_this_year_cond : value_this_year = 16000) (ratio_cond : ratio = 0.8) :
  ∃ (value_last_year : ℝ), value_this_year = ratio * value_last_year ∧ value_last_year = 20000 :=
by
  use 20000
  sorry

end NUMINAMATH_GPT_vehicle_value_last_year_l2134_213439


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2134_213433

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 3 * x - 18 > 0) ↔ (x < -6 ∨ x > 3) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2134_213433


namespace NUMINAMATH_GPT_calculate_expression_l2134_213432

variable (y : ℝ) (π : ℝ) (Q : ℝ)

theorem calculate_expression (h : 5 * (3 * y - 7 * π) = Q) : 
  10 * (6 * y - 14 * π) = 4 * Q := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2134_213432


namespace NUMINAMATH_GPT_payment_n_amount_l2134_213494

def payment_m_n (m n : ℝ) : Prop :=
  m + n = 550 ∧ m = 1.2 * n

theorem payment_n_amount : ∃ n : ℝ, ∀ m : ℝ, payment_m_n m n → n = 250 :=
by
  sorry

end NUMINAMATH_GPT_payment_n_amount_l2134_213494


namespace NUMINAMATH_GPT_correct_operation_l2134_213484

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ (2 * a^3 / a = 2 * a^2) ∧ ¬((a * b)^2 = a * b^2) ∧ ¬((-a^3)^3 = -a^6) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2134_213484


namespace NUMINAMATH_GPT_percentage_problem_l2134_213409

theorem percentage_problem
    (x : ℕ) (h1 : (x:ℝ) / 100 * 20 = 8) :
    x = 40 :=
by
    sorry

end NUMINAMATH_GPT_percentage_problem_l2134_213409


namespace NUMINAMATH_GPT_single_burger_cost_l2134_213488

theorem single_burger_cost
  (total_cost : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (cost_double_burger : ℝ)
  (remaining_cost : ℝ)
  (single_burgers : ℕ)
  (cost_single_burger : ℝ) :
  total_cost = 64.50 ∧
  total_hamburgers = 50 ∧
  double_burgers = 29 ∧
  cost_double_burger = 1.50 ∧
  remaining_cost = total_cost - (double_burgers * cost_double_burger) ∧
  single_burgers = total_hamburgers - double_burgers ∧
  cost_single_burger = remaining_cost / single_burgers →
  cost_single_burger = 1.00 :=
by
  sorry

end NUMINAMATH_GPT_single_burger_cost_l2134_213488


namespace NUMINAMATH_GPT_minimize_function_l2134_213470

noncomputable def f (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

theorem minimize_function : 
  (∀ x : ℝ, x > -1 → f x ≥ 1) ∧ (f 2 = 1) :=
by 
  sorry

end NUMINAMATH_GPT_minimize_function_l2134_213470


namespace NUMINAMATH_GPT_max_area_square_pen_l2134_213431

theorem max_area_square_pen (P : ℝ) (h1 : P = 64) : ∃ A : ℝ, A = 256 := 
by
  sorry

end NUMINAMATH_GPT_max_area_square_pen_l2134_213431


namespace NUMINAMATH_GPT_chosen_number_is_121_l2134_213489

theorem chosen_number_is_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := 
by 
  sorry

end NUMINAMATH_GPT_chosen_number_is_121_l2134_213489


namespace NUMINAMATH_GPT_problem_statement_l2134_213437

def f (x : ℝ) : ℝ := x^5 - x^3 + 1
def g (x : ℝ) : ℝ := x^2 - 2

theorem problem_statement (x1 x2 x3 x4 x5 : ℝ) 
  (h_roots : ∀ x, f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) :
  g x1 * g x2 * g x3 * g x4 * g x5 = -7 := 
sorry

end NUMINAMATH_GPT_problem_statement_l2134_213437


namespace NUMINAMATH_GPT_spinner_prob_l2134_213450

theorem spinner_prob:
  let sections := 4
  let prob := 1 / sections
  let prob_not_e := 1 - prob
  (prob_not_e * prob_not_e) = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_spinner_prob_l2134_213450


namespace NUMINAMATH_GPT_cube_sum_l2134_213407

theorem cube_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end NUMINAMATH_GPT_cube_sum_l2134_213407


namespace NUMINAMATH_GPT_sum_of_first_9_terms_arithmetic_sequence_l2134_213436

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem sum_of_first_9_terms_arithmetic_sequence
  (h_arith_seq : is_arithmetic_sequence a)
  (h_condition : a 2 + a 8 = 8) :
  (Finset.range 9).sum a = 36 :=
sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_arithmetic_sequence_l2134_213436


namespace NUMINAMATH_GPT_fill_tank_in_12_minutes_l2134_213497

theorem fill_tank_in_12_minutes (rate1 rate2 rate_out : ℝ) 
  (h1 : rate1 = 1 / 18) (h2 : rate2 = 1 / 20) (h_out : rate_out = 1 / 45) : 
  12 = 1 / (rate1 + rate2 - rate_out) :=
by
  -- sorry will be replaced with the actual proof.
  sorry

end NUMINAMATH_GPT_fill_tank_in_12_minutes_l2134_213497


namespace NUMINAMATH_GPT_min_val_of_q_l2134_213466

theorem min_val_of_q (p q : ℕ) (h1 : 72 / 487 < p / q) (h2 : p / q < 18 / 121) : 
  ∃ p q : ℕ, (72 / 487 < p / q) ∧ (p / q < 18 / 121) ∧ q = 27 :=
sorry

end NUMINAMATH_GPT_min_val_of_q_l2134_213466


namespace NUMINAMATH_GPT_fermats_little_theorem_l2134_213428

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) (hgcd : gcd a p = 1) : (a^(p-1) - 1) % p = 0 := by
  sorry

end NUMINAMATH_GPT_fermats_little_theorem_l2134_213428


namespace NUMINAMATH_GPT_move_line_upwards_l2134_213447

theorem move_line_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_move_line_upwards_l2134_213447


namespace NUMINAMATH_GPT_find_a_of_tangent_area_l2134_213458

theorem find_a_of_tangent_area (a : ℝ) (h : a > 0) (h_area : (a^3 / 4) = 2) : a = 2 :=
by
  -- Proof is omitted as it's not required.
  sorry

end NUMINAMATH_GPT_find_a_of_tangent_area_l2134_213458


namespace NUMINAMATH_GPT_maximum_third_height_l2134_213452

theorem maximum_third_height 
  (A B C : Type)
  (h1 h2 : ℕ)
  (h1_pos : h1 = 4) 
  (h2_pos : h2 = 12) 
  (h3_pos : ℕ)
  (triangle_inequality : ∀ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a)
  (scalene : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c)
  : (3 < h3_pos ∧ h3_pos < 6) → h3_pos = 5 := 
sorry

end NUMINAMATH_GPT_maximum_third_height_l2134_213452


namespace NUMINAMATH_GPT_terrier_to_poodle_grooming_ratio_l2134_213430

-- Definitions and conditions
def time_to_groom_poodle : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_grooming_time : ℕ := 210
def time_to_groom_terrier := total_grooming_time - (num_poodles * time_to_groom_poodle) / num_terriers

-- Theorem statement
theorem terrier_to_poodle_grooming_ratio :
  time_to_groom_terrier / time_to_groom_poodle = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_terrier_to_poodle_grooming_ratio_l2134_213430


namespace NUMINAMATH_GPT_non_rent_extra_expenses_is_3000_l2134_213405

-- Define the constants
def cost_parts : ℕ := 800
def markup : ℝ := 1.4
def num_computers : ℕ := 60
def rent : ℕ := 5000
def profit : ℕ := 11200

-- Calculate the selling price per computer
def selling_price : ℝ := cost_parts * markup

-- Calculate the total revenue from selling 60 computers
def total_revenue : ℝ := selling_price * num_computers

-- Calculate the total cost of components for 60 computers
def total_cost_components : ℕ := cost_parts * num_computers

-- Calculate the total expenses
def total_expenses : ℝ := total_revenue - profit

-- Define the non-rent extra expenses
def non_rent_extra_expenses : ℝ := total_expenses - rent - total_cost_components

-- Prove that the non-rent extra expenses equal to $3000
theorem non_rent_extra_expenses_is_3000 : non_rent_extra_expenses = 3000 := sorry

end NUMINAMATH_GPT_non_rent_extra_expenses_is_3000_l2134_213405


namespace NUMINAMATH_GPT_min_abs_value_sum_l2134_213401

theorem min_abs_value_sum (x : ℚ) : (min (|x - 1| + |x + 3|) = 4) :=
sorry

end NUMINAMATH_GPT_min_abs_value_sum_l2134_213401


namespace NUMINAMATH_GPT_order_of_a_b_c_l2134_213411

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem order_of_a_b_c : b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l2134_213411


namespace NUMINAMATH_GPT_find_square_number_divisible_by_three_between_90_and_150_l2134_213406

theorem find_square_number_divisible_by_three_between_90_and_150 :
  ∃ x : ℕ, 90 < x ∧ x < 150 ∧ ∃ y : ℕ, x = y * y ∧ 3 ∣ x ∧ x = 144 := 
by 
  sorry

end NUMINAMATH_GPT_find_square_number_divisible_by_three_between_90_and_150_l2134_213406


namespace NUMINAMATH_GPT_range_of_a_l2134_213427

variable (A B : Set ℝ) (a : ℝ)

def setA : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def setB : Set ℝ := {x | (2^(1 - x) + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

theorem range_of_a :
  A ⊆ B ↔ (-4 ≤ a) ∧ (a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2134_213427


namespace NUMINAMATH_GPT_combined_annual_income_eq_correct_value_l2134_213473

theorem combined_annual_income_eq_correct_value :
  let A_income := 5 / 2 * 17000
  let B_income := 1.12 * 17000
  let C_income := 17000
  let D_income := 0.85 * A_income
  (A_income + B_income + C_income + D_income) * 12 = 1375980 :=
by
  sorry

end NUMINAMATH_GPT_combined_annual_income_eq_correct_value_l2134_213473


namespace NUMINAMATH_GPT_baker_sold_more_pastries_l2134_213456

theorem baker_sold_more_pastries {cakes_made pastries_made pastries_sold cakes_sold : ℕ}
    (h1 : cakes_made = 105)
    (h2 : pastries_made = 275)
    (h3 : pastries_sold = 214)
    (h4 : cakes_sold = 163) :
    pastries_sold - cakes_sold = 51 := by
  sorry

end NUMINAMATH_GPT_baker_sold_more_pastries_l2134_213456


namespace NUMINAMATH_GPT_middle_digit_is_3_l2134_213446

theorem middle_digit_is_3 (d e f : ℕ) (hd : 0 ≤ d ∧ d ≤ 7) (he : 0 ≤ e ∧ e ≤ 7) (hf : 0 ≤ f ∧ f ≤ 7)
    (h_eq : 64 * d + 8 * e + f = 100 * f + 10 * e + d) : e = 3 :=
sorry

end NUMINAMATH_GPT_middle_digit_is_3_l2134_213446


namespace NUMINAMATH_GPT_range_of_m_l2134_213474

def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 9 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2134_213474


namespace NUMINAMATH_GPT_sum_of_circle_areas_l2134_213492

theorem sum_of_circle_areas (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) : 
  π * r^2 + π * s^2 + π * t^2 = 56 * π := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_circle_areas_l2134_213492


namespace NUMINAMATH_GPT_profit_without_discount_l2134_213449

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 0.05
noncomputable def profit_with_discount_percentage : ℝ := 0.387
noncomputable def selling_price_with_discount : ℝ := cost_price * (1 + profit_with_discount_percentage)

noncomputable def profit_without_discount_percentage : ℝ :=
  let selling_price_without_discount := selling_price_with_discount / (1 - discount_percentage)
  ((selling_price_without_discount - cost_price) / cost_price) * 100

theorem profit_without_discount :
  profit_without_discount_percentage = 45.635 := by
  sorry

end NUMINAMATH_GPT_profit_without_discount_l2134_213449


namespace NUMINAMATH_GPT_cost_price_of_toy_l2134_213415

-- Define the conditions
def sold_toys := 18
def selling_price := 23100
def gain_toys := 3

-- Define the cost price of one toy 
noncomputable def C := 1100

-- Lean 4 statement to prove the cost price
theorem cost_price_of_toy (C : ℝ) (sold_toys selling_price gain_toys : ℕ) (h1 : selling_price = (sold_toys + gain_toys) * C) : 
  C = 1100 := 
by
  sorry


end NUMINAMATH_GPT_cost_price_of_toy_l2134_213415


namespace NUMINAMATH_GPT_inverse_sum_l2134_213434

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 2 * x - x^2

theorem inverse_sum :
  let f_inv_2 := (1 + Real.sqrt 3)
  let f_inv_1 := 2
  let f_inv_4 := -1
  f_inv_2 + f_inv_1 + f_inv_4 = 2 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_inverse_sum_l2134_213434


namespace NUMINAMATH_GPT_gasoline_needed_l2134_213457

theorem gasoline_needed (D : ℕ) 
    (fuel_efficiency : ℕ) 
    (fuel_efficiency_proof : fuel_efficiency = 20)
    (gallons_for_130km : ℕ) 
    (gallons_for_130km_proof : gallons_for_130km = 130 / 20) :
    (D : ℕ) / fuel_efficiency = (D : ℕ) / 20 :=
by
  -- The proof is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_gasoline_needed_l2134_213457


namespace NUMINAMATH_GPT_ellipse_equation_l2134_213438

theorem ellipse_equation (a b c : ℝ) 
  (h1 : 0 < b) (h2 : b < a) 
  (h3 : c = 3 * Real.sqrt 3) 
  (h4 : a = 6) 
  (h5 : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l2134_213438


namespace NUMINAMATH_GPT_Jeff_total_ounces_of_peanut_butter_l2134_213421

theorem Jeff_total_ounces_of_peanut_butter
    (jars : ℕ)
    (equal_count : ℕ)
    (total_jars : jars = 9)
    (j16 : equal_count = 3) 
    (j28 : equal_count = 3)
    (j40 : equal_count = 3) :
    (3 * 16 + 3 * 28 + 3 * 40 = 252) :=
by
  sorry

end NUMINAMATH_GPT_Jeff_total_ounces_of_peanut_butter_l2134_213421


namespace NUMINAMATH_GPT_sufficient_condition_m_ge_4_range_of_x_for_m5_l2134_213480

variable (x m : ℝ)

-- Problem (1)
theorem sufficient_condition_m_ge_4 (h : m > 0)
  (hpq : ∀ x, ((x + 2) * (x - 6) ≤ 0) → (2 - m ≤ x ∧ x ≤ 2 + m)) : m ≥ 4 := by
  sorry

-- Problem (2)
theorem range_of_x_for_m5 (h : m = 5)
  (hp_or_q : ∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) )
  (hp_and_not_q : ∀ x, ¬(((x + 2) * (x - 6) ≤ 0) ∧ (-3 ≤ x ∧ x ≤ 7))):
  ∀ x, x ∈ Set.Ico (-3) (-2) ∨ x ∈ Set.Ioc (6) (7) := by
  sorry

end NUMINAMATH_GPT_sufficient_condition_m_ge_4_range_of_x_for_m5_l2134_213480


namespace NUMINAMATH_GPT_compare_A_B_l2134_213454

-- Definitions based on conditions from part a)
def A (n : ℕ) : ℕ := 2 * n^2
def B (n : ℕ) : ℕ := 3^n

-- The theorem that needs to be proven
theorem compare_A_B (n : ℕ) (h : n > 0) : A n < B n := 
by sorry

end NUMINAMATH_GPT_compare_A_B_l2134_213454


namespace NUMINAMATH_GPT_sushi_downstream_distance_l2134_213422

variable (sushi_speed : ℕ)
variable (stream_speed : ℕ := 12)
variable (upstream_distance : ℕ := 27)
variable (upstream_time : ℕ := 9)
variable (downstream_time : ℕ := 9)

theorem sushi_downstream_distance (h : upstream_distance = (sushi_speed - stream_speed) * upstream_time) : 
  ∃ (D_d : ℕ), D_d = (sushi_speed + stream_speed) * downstream_time ∧ D_d = 243 :=
by {
  -- We assume the given condition for upstream_distance
  sorry
}

end NUMINAMATH_GPT_sushi_downstream_distance_l2134_213422


namespace NUMINAMATH_GPT_smallest_class_size_l2134_213445

theorem smallest_class_size (n : ℕ) 
  (eight_students_scored_120 : 8 * 120 ≤ n * 92)
  (three_students_scored_115 : 3 * 115 ≤ n * 92)
  (min_score_70 : 70 * n ≤ n * 92)
  (mean_score_92 : (8 * 120 + 3 * 115 + 70 * (n - 11)) / n = 92) :
  n = 25 :=
by
  sorry

end NUMINAMATH_GPT_smallest_class_size_l2134_213445


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2134_213462

variable (a₁ q : ℝ)

def geometric_sequence (n : ℕ) := a₁ * q^n

theorem common_ratio_of_geometric_sequence
  (h_sum : geometric_sequence a₁ q 0 + geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 = 3 * a₁) :
  q = 1 ∨ q = -2 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2134_213462


namespace NUMINAMATH_GPT_kelly_single_shot_decrease_l2134_213414

def kelly_salary_decrease (s : ℝ) : ℝ :=
  let first_cut := s * 0.92
  let second_cut := first_cut * 0.86
  let third_cut := second_cut * 0.82
  third_cut

theorem kelly_single_shot_decrease :
  let original_salary := 1.0 -- Assume original salary is 1 for percentage calculation
  let final_salary := kelly_salary_decrease original_salary
  (100 : ℝ) - (final_salary * 100) = 34.8056 :=
by
  sorry

end NUMINAMATH_GPT_kelly_single_shot_decrease_l2134_213414


namespace NUMINAMATH_GPT_find_x_value_l2134_213419

theorem find_x_value (x : ℝ) 
  (h₁ : 1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) :
  x = (-1 + Real.sqrt 97) / 6 :=
sorry

end NUMINAMATH_GPT_find_x_value_l2134_213419


namespace NUMINAMATH_GPT_miranda_heels_cost_l2134_213496

theorem miranda_heels_cost (months_saved : ℕ) (savings_per_month : ℕ) (gift_from_sister : ℕ) 
  (h1 : months_saved = 3) (h2 : savings_per_month = 70) (h3 : gift_from_sister = 50) : 
  months_saved * savings_per_month + gift_from_sister = 260 := 
by
  sorry

end NUMINAMATH_GPT_miranda_heels_cost_l2134_213496


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_n_ge_52_l2134_213423

theorem sum_arithmetic_sequence_n_ge_52 (n : ℕ) : 
  (∃ k, k = n) → 22 - 3 * (n - 1) = 22 - 3 * (n - 1) ∧ n ∈ { k | 3 ≤ k ∧ k ≤ 13 } :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_n_ge_52_l2134_213423


namespace NUMINAMATH_GPT_part_a_part_b_l2134_213471

variable {p q n : ℕ}

-- Conditions
def coprime (a b : ℕ) : Prop := gcd a b = 1
def differ_by_more_than_one (p q : ℕ) : Prop := (q > p + 1) ∨ (p > q + 1)

-- Part (a): Prove there exists a natural number n such that p + n and q + n are not coprime
theorem part_a (coprime_pq : coprime p q) (diff : differ_by_more_than_one p q) : 
  ∃ n : ℕ, ¬ coprime (p + n) (q + n) :=
sorry

-- Part (b): Prove the smallest such n is 41 for p = 2 and q = 2023
theorem part_b (h : p = 2) (h1 : q = 2023) : 
  ∃ n : ℕ, (n = 41) ∧ (¬ coprime (2 + n) (2023 + n)) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l2134_213471


namespace NUMINAMATH_GPT_average_height_31_students_l2134_213444

theorem average_height_31_students (avg1 avg2 : ℝ) (n1 n2 : ℕ) (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) : ((avg1 * n1 + avg2 * n2) / (n1 + n2)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_height_31_students_l2134_213444


namespace NUMINAMATH_GPT_consecutive_integers_product_l2134_213467

theorem consecutive_integers_product (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 336) : a + b + c = 21 :=
sorry

end NUMINAMATH_GPT_consecutive_integers_product_l2134_213467


namespace NUMINAMATH_GPT_total_cost_l2134_213417

-- Define the given conditions.
def coffee_pounds : ℕ := 4
def coffee_cost_per_pound : ℝ := 2.50
def milk_gallons : ℕ := 2
def milk_cost_per_gallon : ℝ := 3.50

-- The total cost Jasmine will pay is $17.00
theorem total_cost : coffee_pounds * coffee_cost_per_pound + milk_gallons * milk_cost_per_gallon = 17.00 := by
  sorry

end NUMINAMATH_GPT_total_cost_l2134_213417


namespace NUMINAMATH_GPT_saved_percentage_is_correct_l2134_213491

def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 5200
def amount_saved : ℝ := 2300

noncomputable def total_expenses : ℝ :=
  rent + milk + groceries + education + petrol + miscellaneous

noncomputable def total_salary : ℝ :=
  total_expenses + amount_saved

noncomputable def percentage_saved : ℝ :=
  (amount_saved / total_salary) * 100

theorem saved_percentage_is_correct :
  percentage_saved = 8.846 := by
  sorry

end NUMINAMATH_GPT_saved_percentage_is_correct_l2134_213491


namespace NUMINAMATH_GPT_john_sublets_to_3_people_l2134_213455

def monthly_income (n : ℕ) : ℕ := 400 * n
def monthly_cost : ℕ := 900
def annual_profit (n : ℕ) : ℕ := 12 * (monthly_income n - monthly_cost)

theorem john_sublets_to_3_people
  (h1 : forall n : ℕ, monthly_income n - monthly_cost > 0)
  (h2 : annual_profit 3 = 3600) :
  3 = 3 := by
  sorry

end NUMINAMATH_GPT_john_sublets_to_3_people_l2134_213455


namespace NUMINAMATH_GPT_part1_part2_l2134_213429

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≥ 3 * x + 2) : x ≥ 3 ∨ x ≤ -1 :=
sorry

-- Part (2)
theorem part2 (h : ∀ x, f x a ≤ 0 → x ≤ -1) : a = 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2134_213429


namespace NUMINAMATH_GPT_problem_statement_l2134_213403

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then 4^x
  else if (-1 < x ∧ x < 0) then -4^(-x)
  else if (-2 < x ∧ x < -1) then -4^(x + 2)
  else if (1 < x ∧ x < 2) then 4^(x - 2)
  else 0

theorem problem_statement :
  (f (-5 / 2) + f 1) = -2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2134_213403


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_not_necessary_condition_l2134_213477

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  a^2 + b^2 = 1 → (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) :=
by
  sorry

theorem not_necessary_condition (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) → ¬(a^2 + b^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_not_necessary_condition_l2134_213477


namespace NUMINAMATH_GPT_complement_of_angle_l2134_213475

theorem complement_of_angle (supplement : ℝ) (h_supp : supplement = 130) (original_angle : ℝ) (h_orig : original_angle = 180 - supplement) : 
  (90 - original_angle) = 40 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_complement_of_angle_l2134_213475


namespace NUMINAMATH_GPT_johnson_potatoes_left_l2134_213408

theorem johnson_potatoes_left :
  ∀ (initial gina tom anne remaining : Nat),
  initial = 300 →
  gina = 69 →
  tom = 2 * gina →
  anne = tom / 3 →
  remaining = initial - (gina + tom + anne) →
  remaining = 47 := by
sorry

end NUMINAMATH_GPT_johnson_potatoes_left_l2134_213408


namespace NUMINAMATH_GPT_axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l2134_213418

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt 3 * Real.sin (Real.pi - x) + 5 * Real.sin (Real.pi / 2 + x) + 5

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = f (Real.pi / 3 + k * Real.pi) :=
sorry

theorem center_of_symmetry :
  ∃ k : ℤ, f (k * Real.pi - Real.pi / 6) = 5 :=
sorry

noncomputable def g (x : ℝ) : ℝ := 10 * Real.sin (2 * x) - 8

theorem g_max_value :
  ∀ x : ℝ, g x ≤ 2 :=
sorry

theorem g_increasing_intervals :
  ∀ k : ℤ, -Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≤ g (x + 1) :=
sorry

theorem g_decreasing_intervals :
  ∀ k : ℤ, Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 4 + k * Real.pi → ∀ x : ℝ, g x ≥ g (x + 1) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_center_of_symmetry_g_max_value_g_increasing_intervals_g_decreasing_intervals_l2134_213418


namespace NUMINAMATH_GPT_minnie_lucy_time_difference_is_66_minutes_l2134_213465

noncomputable def minnie_time_uphill : ℚ := 12 / 6
noncomputable def minnie_time_downhill : ℚ := 18 / 25
noncomputable def minnie_time_flat : ℚ := 15 / 15

noncomputable def minnie_total_time : ℚ := minnie_time_uphill + minnie_time_downhill + minnie_time_flat

noncomputable def lucy_time_flat : ℚ := 15 / 25
noncomputable def lucy_time_uphill : ℚ := 12 / 8
noncomputable def lucy_time_downhill : ℚ := 18 / 35

noncomputable def lucy_total_time : ℚ := lucy_time_flat + lucy_time_uphill + lucy_time_downhill

-- Convert hours to minutes
noncomputable def minnie_total_time_minutes : ℚ := minnie_total_time * 60
noncomputable def lucy_total_time_minutes : ℚ := lucy_total_time * 60

-- Difference in minutes
noncomputable def time_difference : ℚ := minnie_total_time_minutes - lucy_total_time_minutes

theorem minnie_lucy_time_difference_is_66_minutes : time_difference = 66 := by
  sorry

end NUMINAMATH_GPT_minnie_lucy_time_difference_is_66_minutes_l2134_213465


namespace NUMINAMATH_GPT_three_layer_carpet_area_l2134_213486

-- Define the dimensions of the carpets and the hall
structure Carpet := (width : ℕ) (height : ℕ)

def principal_carpet : Carpet := ⟨6, 8⟩
def caretaker_carpet : Carpet := ⟨6, 6⟩
def parent_committee_carpet : Carpet := ⟨5, 7⟩
def hall : Carpet := ⟨10, 10⟩

-- Define the area function
def area (c : Carpet) : ℕ := c.width * c.height

-- Prove the area of the part of the hall covered by all three carpets
theorem three_layer_carpet_area : area ⟨3, 2⟩ = 6 :=
by
  sorry

end NUMINAMATH_GPT_three_layer_carpet_area_l2134_213486


namespace NUMINAMATH_GPT_minimize_expression_l2134_213476

theorem minimize_expression (n : ℕ) (h : 0 < n) : 
  (n = 10) ↔ (∀ m : ℕ, 0 < m → ((n / 2) + (50 / n) ≤ (m / 2) + (50 / m))) :=
sorry

end NUMINAMATH_GPT_minimize_expression_l2134_213476


namespace NUMINAMATH_GPT_intersection_product_l2134_213482

noncomputable def line_l (t : ℝ) := (1 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

def curve_C (x y : ℝ) : Prop := y^2 = 8 * x

theorem intersection_product :
  ∀ (t1 t2 : ℝ), 
  (1 + (1/2) * t1)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t1) →
  (1 + (1/2) * t2)^2 = 8 * (1 + (Real.sqrt 3 / 2) * t2) →
  (1 + (1/2) * t1) * (1 + (1/2) * t2) = 28 := 
  sorry

end NUMINAMATH_GPT_intersection_product_l2134_213482


namespace NUMINAMATH_GPT_inequality_not_always_true_l2134_213453

theorem inequality_not_always_true (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬(∀ a > 0, ∀ b > 0, (2 / ((1 / a) + (1 / b)) ≥ Real.sqrt (a * b))) :=
sorry

end NUMINAMATH_GPT_inequality_not_always_true_l2134_213453


namespace NUMINAMATH_GPT_calculate_c_from_law_of_cosines_l2134_213435

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

theorem calculate_c_from_law_of_cosines 
  (a b c : ℝ) (B : ℝ)
  (ha : a = 8) (hb : b = 7) (hB : B = Real.pi / 3) : 
  (c = 3) ∨ (c = 5) :=
sorry

end NUMINAMATH_GPT_calculate_c_from_law_of_cosines_l2134_213435


namespace NUMINAMATH_GPT_value_of_m_over_q_l2134_213425

-- Definitions for the given conditions
variables (n m p q : ℤ) 

-- Main theorem statement
theorem value_of_m_over_q (h1 : m = 10 * n) (h2 : p = 2 * n) (h3 : p = q / 5) :
  m / q = 1 :=
sorry

end NUMINAMATH_GPT_value_of_m_over_q_l2134_213425


namespace NUMINAMATH_GPT_mean_equality_l2134_213495

theorem mean_equality (x : ℝ) 
  (h : (7 + 9 + 23) / 3 = (16 + x) / 2) : 
  x = 10 := 
sorry

end NUMINAMATH_GPT_mean_equality_l2134_213495


namespace NUMINAMATH_GPT_middle_of_three_consecutive_integers_is_60_l2134_213498

theorem middle_of_three_consecutive_integers_is_60 (n : ℤ)
    (h : (n - 1) + n + (n + 1) = 180) : n = 60 := by
  sorry

end NUMINAMATH_GPT_middle_of_three_consecutive_integers_is_60_l2134_213498


namespace NUMINAMATH_GPT_total_screens_sold_l2134_213426

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end NUMINAMATH_GPT_total_screens_sold_l2134_213426


namespace NUMINAMATH_GPT_value_of_expression_l2134_213478

theorem value_of_expression (x : ℝ) : 
  let a := 2000 * x + 2001
  let b := 2000 * x + 2002
  let c := 2000 * x + 2003
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2134_213478
