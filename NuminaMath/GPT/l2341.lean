import Mathlib

namespace NUMINAMATH_GPT_mutually_exclusive_pairs_l2341_234122

-- Define the events based on the conditions
def event_two_red_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 2 ∧ drawn.count "white" = 1)

def event_one_red_two_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 1 ∧ drawn.count "white" = 2)

def event_three_red (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "red" = 3

def event_at_least_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ 1 ≤ drawn.count "white"

def event_three_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "white" = 3

-- Define mutually exclusive property
def mutually_exclusive (A B : List String → List String → Prop) (bag : List String) : Prop :=
  ∀ drawn, A bag drawn → ¬ B bag drawn

-- Define the main theorem statement
theorem mutually_exclusive_pairs (bag : List String) (condition : bag = ["red", "red", "red", "red", "red", "white", "white", "white", "white", "white"]) :
  mutually_exclusive event_three_red event_at_least_one_white bag ∧
  mutually_exclusive event_three_red event_three_white bag :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_pairs_l2341_234122


namespace NUMINAMATH_GPT_weight_loss_comparison_l2341_234169

-- Define the conditions
def weight_loss_Barbi : ℝ := 1.5 * 24
def weight_loss_Luca : ℝ := 9 * 15
def weight_loss_Kim : ℝ := (2 * 12) + (3 * 60)

-- Define the combined weight loss of Luca and Kim
def combined_weight_loss_Luca_Kim : ℝ := weight_loss_Luca + weight_loss_Kim

-- Define the difference in weight loss between Luca and Kim combined and Barbi
def weight_loss_difference : ℝ := combined_weight_loss_Luca_Kim - weight_loss_Barbi

-- State the theorem to be proved
theorem weight_loss_comparison : weight_loss_difference = 303 := by
  sorry

end NUMINAMATH_GPT_weight_loss_comparison_l2341_234169


namespace NUMINAMATH_GPT_negation_of_one_even_is_all_odd_or_at_least_two_even_l2341_234153

-- Definitions based on the problem conditions
def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

def all_odd (a b c : ℕ) : Prop :=
  ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c

def at_least_two_even (a b c : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c)

-- The proposition to prove
theorem negation_of_one_even_is_all_odd_or_at_least_two_even (a b c : ℕ) :
  ¬ exactly_one_even a b c ↔ all_odd a b c ∨ at_least_two_even a b c :=
by sorry

end NUMINAMATH_GPT_negation_of_one_even_is_all_odd_or_at_least_two_even_l2341_234153


namespace NUMINAMATH_GPT_find_y_in_terms_of_x_l2341_234197

theorem find_y_in_terms_of_x (p : ℝ) (x y : ℝ) (h1 : x = 1 + 3^p) (h2 : y = 1 + 3^(-p)) : y = x / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_y_in_terms_of_x_l2341_234197


namespace NUMINAMATH_GPT_contradiction_example_l2341_234191

theorem contradiction_example (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_contradiction_example_l2341_234191


namespace NUMINAMATH_GPT_jimmy_exams_l2341_234195

theorem jimmy_exams (p l a : ℕ) (h_p : p = 50) (h_l : l = 5) (h_a : a = 5) (x : ℕ) :
  (20 * x - (l + a) ≥ p) ↔ (x ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_jimmy_exams_l2341_234195


namespace NUMINAMATH_GPT_sum_6n_is_correct_l2341_234137

theorem sum_6n_is_correct {n : ℕ} (h : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by sorry

end NUMINAMATH_GPT_sum_6n_is_correct_l2341_234137


namespace NUMINAMATH_GPT_locus_of_midpoint_of_square_l2341_234100

theorem locus_of_midpoint_of_square (a : ℝ) (x y : ℝ) (h1 : x^2 + y^2 = 4 * a^2) :
  (∃ X Y : ℝ, 2 * X = x ∧ 2 * Y = y ∧ X^2 + Y^2 = a^2) :=
by {
  -- No proof is required, so we use 'sorry' here
  sorry
}

end NUMINAMATH_GPT_locus_of_midpoint_of_square_l2341_234100


namespace NUMINAMATH_GPT_jerry_won_47_tickets_l2341_234126

open Nat

-- Define the initial number of tickets
def initial_tickets : Nat := 4

-- Define the number of tickets spent on the beanie
def tickets_spent_on_beanie : Nat := 2

-- Define the current total number of tickets Jerry has
def current_tickets : Nat := 49

-- Define the number of tickets Jerry won later
def tickets_won_later : Nat := current_tickets - (initial_tickets - tickets_spent_on_beanie)

-- The theorem to prove
theorem jerry_won_47_tickets :
  tickets_won_later = 47 :=
by sorry

end NUMINAMATH_GPT_jerry_won_47_tickets_l2341_234126


namespace NUMINAMATH_GPT_xiao_wang_program_output_l2341_234162

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end NUMINAMATH_GPT_xiao_wang_program_output_l2341_234162


namespace NUMINAMATH_GPT_puzzle_pieces_count_l2341_234164

variable (border_pieces : ℕ) (trevor_pieces : ℕ) (joe_pieces : ℕ) (missing_pieces : ℕ)

def total_puzzle_pieces (border_pieces trevor_pieces joe_pieces missing_pieces : ℕ) : ℕ :=
  border_pieces + trevor_pieces + joe_pieces + missing_pieces

theorem puzzle_pieces_count :
  border_pieces = 75 → 
  trevor_pieces = 105 → 
  joe_pieces = 3 * trevor_pieces → 
  missing_pieces = 5 → 
  total_puzzle_pieces border_pieces trevor_pieces joe_pieces missing_pieces = 500 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  -- proof step to get total_number_pieces = 75 + 105 + (3 * 105) + 5
  -- hence total_puzzle_pieces = 500
  sorry

end NUMINAMATH_GPT_puzzle_pieces_count_l2341_234164


namespace NUMINAMATH_GPT_factorization_problem_l2341_234104

theorem factorization_problem 
    (a m n b : ℝ)
    (h1 : (x + 2) * (x + 4) = x^2 + a * x + m)
    (h2 : (x + 1) * (x + 9) = x^2 + n * x + b) :
    (x + 3) * (x + 3) = x^2 + a * x + b :=
by
  sorry

end NUMINAMATH_GPT_factorization_problem_l2341_234104


namespace NUMINAMATH_GPT_no_consecutive_integer_sum_to_36_l2341_234194

theorem no_consecutive_integer_sum_to_36 :
  ∀ (a n : ℕ), n ≥ 2 → (n * a + n * (n - 1) / 2) = 36 → false :=
by
  sorry

end NUMINAMATH_GPT_no_consecutive_integer_sum_to_36_l2341_234194


namespace NUMINAMATH_GPT_find_tax_rate_l2341_234131

variable (total_spent : ℝ) (sales_tax : ℝ) (tax_free_cost : ℝ) (taxable_items_cost : ℝ) 
variable (T : ℝ)

theorem find_tax_rate (h1 : total_spent = 25) 
                      (h2 : sales_tax = 0.30)
                      (h3 : tax_free_cost = 21.7)
                      (h4 : taxable_items_cost = total_spent - tax_free_cost - sales_tax)
                      (h5 : sales_tax = (T / 100) * taxable_items_cost) :
  T = 10 := 
sorry

end NUMINAMATH_GPT_find_tax_rate_l2341_234131


namespace NUMINAMATH_GPT_alyosha_cube_problem_l2341_234165

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end NUMINAMATH_GPT_alyosha_cube_problem_l2341_234165


namespace NUMINAMATH_GPT_largest_possible_distance_between_spheres_l2341_234109

noncomputable def largest_distance_between_spheres : ℝ :=
  110 + Real.sqrt 1818

theorem largest_possible_distance_between_spheres :
  let center1 := (3, -5, 7)
  let radius1 := 15
  let center2 := (-10, 20, -25)
  let radius2 := 95
  ∀ A B : ℝ × ℝ × ℝ,
    (dist A center1 = radius1) →
    (dist B center2 = radius2) →
    dist A B ≤ largest_distance_between_spheres :=
  sorry

end NUMINAMATH_GPT_largest_possible_distance_between_spheres_l2341_234109


namespace NUMINAMATH_GPT_books_sold_l2341_234183

theorem books_sold (initial_books sold_books remaining_books : ℕ) 
  (h_initial : initial_books = 242) 
  (h_remaining : remaining_books = 105)
  (h_relation : sold_books = initial_books - remaining_books) :
  sold_books = 137 := 
by
  sorry

end NUMINAMATH_GPT_books_sold_l2341_234183


namespace NUMINAMATH_GPT_max_A_plus_B_l2341_234182

theorem max_A_plus_B:
  ∃ A B C D : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  A + B + C + D = 17 ∧ ∃ k : ℕ, C + D ≠ 0 ∧ A + B = k * (C + D) ∧
  A + B = 16 :=
by sorry

end NUMINAMATH_GPT_max_A_plus_B_l2341_234182


namespace NUMINAMATH_GPT_solution_set_of_inequality_af_neg2x_pos_l2341_234168

-- Given conditions:
-- f(x) = x^2 + ax + b has roots -1 and 2
-- We need to prove that the solution set for af(-2x) > 0 is -1 < x < 1/2
theorem solution_set_of_inequality_af_neg2x_pos (a b : ℝ) (x : ℝ) 
  (h1 : -1 + 2 = -a) 
  (h2 : -1 * 2 = b) : 
  (a * ((-2 * x)^2 + a * (-2 * x) + b) > 0) = (-1 < x ∧ x < 1/2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_af_neg2x_pos_l2341_234168


namespace NUMINAMATH_GPT_no_real_solutions_for_equation_l2341_234143

theorem no_real_solutions_for_equation (x : ℝ) :
  y = 3 * x ∧ y = (x^3 - 8) / (x - 2) → false :=
by {
  sorry
}

end NUMINAMATH_GPT_no_real_solutions_for_equation_l2341_234143


namespace NUMINAMATH_GPT_xy_value_l2341_234193

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := 
by
  sorry

end NUMINAMATH_GPT_xy_value_l2341_234193


namespace NUMINAMATH_GPT_solve_for_x_l2341_234128

theorem solve_for_x (x : ℝ) (h : 1 - 2 * (1 / (1 + x)) = 1 / (1 + x)) : x = 2 := 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2341_234128


namespace NUMINAMATH_GPT_gcd_420_135_l2341_234199

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_420_135_l2341_234199


namespace NUMINAMATH_GPT_polygon_sides_l2341_234107

theorem polygon_sides (a : ℝ) (n : ℕ) (h1 : a = 140) (h2 : 180 * (n-2) = n * a) : n = 9 := 
by sorry

end NUMINAMATH_GPT_polygon_sides_l2341_234107


namespace NUMINAMATH_GPT_second_candy_cost_l2341_234156

theorem second_candy_cost 
  (C : ℝ) 
  (hp := 25 * 8 + 50 * C = 75 * 6) : 
  C = 5 := 
  sorry

end NUMINAMATH_GPT_second_candy_cost_l2341_234156


namespace NUMINAMATH_GPT_evaluate_seventy_two_square_minus_twenty_four_square_l2341_234127

theorem evaluate_seventy_two_square_minus_twenty_four_square :
  72 ^ 2 - 24 ^ 2 = 4608 := 
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_seventy_two_square_minus_twenty_four_square_l2341_234127


namespace NUMINAMATH_GPT_minimum_number_of_rooks_l2341_234188

theorem minimum_number_of_rooks (n : ℕ) : 
  ∃ (num_rooks : ℕ), (∀ (cells_colored : ℕ), cells_colored = n^2 → num_rooks = n) :=
by sorry

end NUMINAMATH_GPT_minimum_number_of_rooks_l2341_234188


namespace NUMINAMATH_GPT_product_of_possible_values_of_x_l2341_234112

theorem product_of_possible_values_of_x :
  (∃ x, |x - 7| - 3 = -2) → ∃ y z, |y - 7| - 3 = -2 ∧ |z - 7| - 3 = -2 ∧ y * z = 48 :=
by
  sorry

end NUMINAMATH_GPT_product_of_possible_values_of_x_l2341_234112


namespace NUMINAMATH_GPT_surface_area_of_sphere_containing_prism_l2341_234135

-- Assume the necessary geometric context and definitions are available.
def rightSquarePrism (a h : ℝ) (V : ℝ) := 
  a^2 * h = V

theorem surface_area_of_sphere_containing_prism 
  (a h V : ℝ) (S : ℝ) (π := Real.pi)
  (prism_on_sphere : ∀ (prism : rightSquarePrism a h V), True)
  (height_eq_4 : h = 4) 
  (volume_eq_16 : V = 16) :
  S = 4 * π * 24 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_containing_prism_l2341_234135


namespace NUMINAMATH_GPT_locus_of_circle_center_l2341_234196

theorem locus_of_circle_center (x y : ℝ) : 
    (exists C : ℝ × ℝ, (C.1, C.2) = (x,y)) ∧ 
    ((x - 0)^2 + (y - 3)^2 = r^2) ∧ 
    (y + 3 = 0) → x^2 = 12 * y :=
sorry

end NUMINAMATH_GPT_locus_of_circle_center_l2341_234196


namespace NUMINAMATH_GPT_solve_recurrence_relation_l2341_234130

def recurrence_relation (a : ℕ → ℤ) : Prop :=
  ∀ n ≥ 3, a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6

def initial_conditions (a : ℕ → ℤ) : Prop :=
  a 0 = -4 ∧ a 1 = -2 ∧ a 2 = 2

def explicit_solution (n : ℕ) : ℤ :=
  -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem solve_recurrence_relation :
  ∀ (a : ℕ → ℤ),
    recurrence_relation a →
    initial_conditions a →
    ∀ n, a n = explicit_solution n := by
  intros a h_recur h_init n
  sorry

end NUMINAMATH_GPT_solve_recurrence_relation_l2341_234130


namespace NUMINAMATH_GPT_find_values_l2341_234113

theorem find_values (a b: ℝ) (h1: a > b) (h2: b > 1)
  (h3: Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h4: a^b = b^a) :
  a = 4 ∧ b = 2 := 
sorry

end NUMINAMATH_GPT_find_values_l2341_234113


namespace NUMINAMATH_GPT_find_minimum_n_l2341_234138

noncomputable def a_seq (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def S_n (n : ℕ) : ℕ := 1 / 2 * (3 ^ n - 1)

theorem find_minimum_n (S_n : ℕ → ℕ) (n : ℕ) :
  (3^n - 1) / 2 > 1000 → n = 7 := 
sorry

end NUMINAMATH_GPT_find_minimum_n_l2341_234138


namespace NUMINAMATH_GPT_ratio_cost_to_marked_price_l2341_234144

variables (x : ℝ) (marked_price : ℝ) (selling_price : ℝ) (cost_price : ℝ)

theorem ratio_cost_to_marked_price :
  (selling_price = marked_price - 1/4 * marked_price) →
  (cost_price = 2/3 * selling_price) →
  (cost_price / marked_price = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_cost_to_marked_price_l2341_234144


namespace NUMINAMATH_GPT_max_value_g_f_less_than_e_x_div_x_sq_l2341_234172

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end NUMINAMATH_GPT_max_value_g_f_less_than_e_x_div_x_sq_l2341_234172


namespace NUMINAMATH_GPT_e_exp_ax1_ax2_gt_two_l2341_234154

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem e_exp_ax1_ax2_gt_two {a x1 x2 : ℝ} (h : a ≠ 0) (h1 : f a x1 = 0) (h2 : f a x2 = 0) (hx : x1 < x2) : 
  Real.exp (a * x1) + Real.exp (a * x2) > 2 :=
sorry

end NUMINAMATH_GPT_e_exp_ax1_ax2_gt_two_l2341_234154


namespace NUMINAMATH_GPT_chocolate_chip_cookie_count_l2341_234133

-- Let cookies_per_bag be the number of cookies in each bag
def cookies_per_bag : ℕ := 5

-- Let oatmeal_cookies be the number of oatmeal cookies
def oatmeal_cookies : ℕ := 2

-- Let num_baggies be the number of baggies
def num_baggies : ℕ := 7

-- Define the total number of cookies as num_baggies * cookies_per_bag
def total_cookies : ℕ := num_baggies * cookies_per_bag

-- Define the number of chocolate chip cookies as total_cookies - oatmeal_cookies
def chocolate_chip_cookies : ℕ := total_cookies - oatmeal_cookies

-- Prove that the number of chocolate chip cookies is 33
theorem chocolate_chip_cookie_count : chocolate_chip_cookies = 33 := by
  sorry

end NUMINAMATH_GPT_chocolate_chip_cookie_count_l2341_234133


namespace NUMINAMATH_GPT_linear_eq_must_be_one_l2341_234177

theorem linear_eq_must_be_one (m : ℝ) : (∀ x y : ℝ, (m + 1) * x + 3 * y ^ m = 5 → (m = 1)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_linear_eq_must_be_one_l2341_234177


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l2341_234176

variable (a : ℕ → ℤ) (d : ℤ)
variable (h1 : a 1 + a 4 + a 7 = 39)
variable (h2 : a 2 + a 5 + a 8 = 33)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_value : a 5 + a 8 + a 11 = 15 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l2341_234176


namespace NUMINAMATH_GPT_smaller_sphere_radius_l2341_234101

theorem smaller_sphere_radius (R x : ℝ) (h1 : (4/3) * Real.pi * R^3 = (4/3) * Real.pi * x^3 + (4/3) * Real.pi * (2 * x)^3) 
  (h2 : ∀ r₁ r₂ : ℝ, r₁ / r₂ = 1 / 2 → r₁ = x ∧ r₂ = 2 * x) : x = R / 3 :=
by 
  sorry

end NUMINAMATH_GPT_smaller_sphere_radius_l2341_234101


namespace NUMINAMATH_GPT_arithmetic_mean_equality_l2341_234114

variable (x y a b : ℝ)

theorem arithmetic_mean_equality (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / 2 * ((x + a) / y + (y - b) / x)) = (x^2 + a * x + y^2 - b * y) / (2 * x * y) :=
  sorry

end NUMINAMATH_GPT_arithmetic_mean_equality_l2341_234114


namespace NUMINAMATH_GPT_problem_1_problem_2_l2341_234179

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (x + 1)

theorem problem_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x ≥ 1 - x + x^2 := 
sorry

theorem problem_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 1 - x + x^2) : f x > 3 / 4 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2341_234179


namespace NUMINAMATH_GPT_birds_count_is_30_l2341_234116

def total_animals : ℕ := 77
def number_of_kittens : ℕ := 32
def number_of_hamsters : ℕ := 15

def number_of_birds : ℕ := total_animals - number_of_kittens - number_of_hamsters

theorem birds_count_is_30 : number_of_birds = 30 := by
  sorry

end NUMINAMATH_GPT_birds_count_is_30_l2341_234116


namespace NUMINAMATH_GPT_factor_polynomial_l2341_234186

theorem factor_polynomial (a b : ℕ) : 
  2 * a^3 - 3 * a^2 * b - 3 * a * b^2 + 2 * b^3 = (a + b) * (a - 2 * b) * (2 * a - b) :=
by sorry

end NUMINAMATH_GPT_factor_polynomial_l2341_234186


namespace NUMINAMATH_GPT_distinct_orders_scoops_l2341_234132

-- Conditions
def total_scoops : ℕ := 4
def chocolate_scoops : ℕ := 2
def vanilla_scoops : ℕ := 1
def strawberry_scoops : ℕ := 1

-- Problem statement
theorem distinct_orders_scoops :
  (Nat.factorial total_scoops) / ((Nat.factorial chocolate_scoops) * (Nat.factorial vanilla_scoops) * (Nat.factorial strawberry_scoops)) = 12 := by
  sorry

end NUMINAMATH_GPT_distinct_orders_scoops_l2341_234132


namespace NUMINAMATH_GPT_odds_against_C_win_l2341_234118

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C_win (pA pB : ℚ) (hA : pA = 1/5) (hB : pB = 2/3) :
  odds_against_winning (1 - pA - pB) = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_odds_against_C_win_l2341_234118


namespace NUMINAMATH_GPT_pyramid_circumscribed_sphere_volume_l2341_234180

theorem pyramid_circumscribed_sphere_volume 
  (PA ABCD : ℝ) 
  (square_base : Prop)
  (perpendicular_PA_base : Prop)
  (AB : ℝ)
  (PA_val : PA = 1)
  (AB_val : AB = 2) 
  : (∃ (volume : ℝ), volume = (4/3) * π * (3/2)^3 ∧ volume = 9 * π / 2) := 
by
  -- Provided the conditions, we need to prove that the volume of the circumscribed sphere is 9π/2
  sorry

end NUMINAMATH_GPT_pyramid_circumscribed_sphere_volume_l2341_234180


namespace NUMINAMATH_GPT_josh_marbles_l2341_234146

theorem josh_marbles (initial_marbles lost_marbles remaining_marbles : ℤ) 
  (h1 : initial_marbles = 19) 
  (h2 : lost_marbles = 11) 
  (h3 : remaining_marbles = initial_marbles - lost_marbles) : 
  remaining_marbles = 8 := 
by
  sorry

end NUMINAMATH_GPT_josh_marbles_l2341_234146


namespace NUMINAMATH_GPT_proof_by_contradiction_conditions_l2341_234173

theorem proof_by_contradiction_conditions :
  ∀ (P Q : Prop),
    (∃ R : Prop, (R = ¬Q) ∧ (P → R) ∧ (R → P) ∧ (∀ T : Prop, (T = Q) → false)) →
    (∃ S : Prop, (S = ¬Q) ∧ P ∧ (∃ U : Prop, U) ∧ ¬Q) :=
by
  sorry

end NUMINAMATH_GPT_proof_by_contradiction_conditions_l2341_234173


namespace NUMINAMATH_GPT_composite_for_positive_integers_l2341_234174

def is_composite (n : ℤ) : Prop :=
  ∃ a b : ℤ, 1 < a ∧ 1 < b ∧ n = a * b

theorem composite_for_positive_integers (n : ℕ) (h_pos : 1 < n) :
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) := 
sorry

end NUMINAMATH_GPT_composite_for_positive_integers_l2341_234174


namespace NUMINAMATH_GPT_intersect_x_axis_iff_k_le_4_l2341_234175

theorem intersect_x_axis_iff_k_le_4 (k : ℝ) :
  (∃ x : ℝ, (k-3) * x^2 + 2 * x + 1 = 0) ↔ k ≤ 4 :=
sorry

end NUMINAMATH_GPT_intersect_x_axis_iff_k_le_4_l2341_234175


namespace NUMINAMATH_GPT_trajectory_ellipse_l2341_234187

/--
Given two fixed points A(-2,0) and B(2,0) in the Cartesian coordinate system, 
if a moving point P satisfies |PA| + |PB| = 6, 
then prove that the equation of the trajectory for point P is (x^2) / 9 + (y^2) / 5 = 1.
-/
theorem trajectory_ellipse (P : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hPA_PB : dist P A + dist P B = 6) :
  (P.1 ^ 2) / 9 + (P.2 ^ 2) / 5 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_ellipse_l2341_234187


namespace NUMINAMATH_GPT_find_a_value_l2341_234123

theorem find_a_value : 
  (∀ x, (3 * (x - 2) - 4 * (x - 5 / 4) = 0) ↔ ( ∃ a, ((2 * x - a) / 3 - (x - a) / 2 = x - 1) ∧ a = -11 )) := sorry

end NUMINAMATH_GPT_find_a_value_l2341_234123


namespace NUMINAMATH_GPT_range_of_m_for_function_l2341_234148

noncomputable def isFunctionDefinedForAllReal (f : ℝ → ℝ) := ∀ x : ℝ, true

theorem range_of_m_for_function :
  (∀ x : ℝ, x^2 - 2 * m * x + m + 2 > 0) ↔ (-1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_for_function_l2341_234148


namespace NUMINAMATH_GPT_triple_apply_l2341_234189

def f (x : ℝ) : ℝ := 5 * x - 4

theorem triple_apply : f (f (f 2)) = 126 :=
by
  rw [f, f, f]
  sorry

end NUMINAMATH_GPT_triple_apply_l2341_234189


namespace NUMINAMATH_GPT_part1_factorization_part2_factorization_l2341_234125

-- Part 1
theorem part1_factorization (x : ℝ) :
  (x - 1) * (6 * x + 5) = 6 * x^2 - x - 5 :=
by {
  sorry
}

-- Part 2
theorem part2_factorization (x : ℝ) :
  (x - 1) * (x + 3) * (x - 2) = x^3 - 7 * x + 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_factorization_part2_factorization_l2341_234125


namespace NUMINAMATH_GPT_statement_B_statement_D_l2341_234106

variable {a b c d : ℝ}

theorem statement_B (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : (c / a) > (c / b) := 
by sorry

theorem statement_D (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : (a * c) < (b * d) := 
by sorry

end NUMINAMATH_GPT_statement_B_statement_D_l2341_234106


namespace NUMINAMATH_GPT_exactly_one_divisible_by_4_l2341_234157

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_exactly_one_divisible_by_4_l2341_234157


namespace NUMINAMATH_GPT_fraction_subtraction_simplify_l2341_234152

noncomputable def fraction_subtraction : ℚ :=
  (12 / 25) - (3 / 75)

theorem fraction_subtraction_simplify : fraction_subtraction = (11 / 25) :=
  by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_fraction_subtraction_simplify_l2341_234152


namespace NUMINAMATH_GPT_triangle_value_l2341_234160

-- Define the operation \(\triangle\)
def triangle (m n p q : ℕ) : ℕ := (m * m) * p * q / n

-- Define the problem statement
theorem triangle_value : triangle 5 6 9 4 = 150 := by
  sorry

end NUMINAMATH_GPT_triangle_value_l2341_234160


namespace NUMINAMATH_GPT_sqrt_product_simplify_l2341_234185

theorem sqrt_product_simplify (x : ℝ) (hx : 0 ≤ x):
  Real.sqrt (48*x) * Real.sqrt (3*x) * Real.sqrt (50*x) = 60 * x * Real.sqrt x := 
by
  sorry

end NUMINAMATH_GPT_sqrt_product_simplify_l2341_234185


namespace NUMINAMATH_GPT_hamburger_combinations_l2341_234139

def number_of_condiments := 8
def condiment_combinations := 2 ^ number_of_condiments
def number_of_meat_patties := 4
def total_hamburgers := number_of_meat_patties * condiment_combinations

theorem hamburger_combinations :
  total_hamburgers = 1024 :=
by
  sorry

end NUMINAMATH_GPT_hamburger_combinations_l2341_234139


namespace NUMINAMATH_GPT_rate_downstream_l2341_234171

-- Define the man's rate in still water
def rate_still_water : ℝ := 24.5

-- Define the rate of the current
def rate_current : ℝ := 7.5

-- Define the man's rate upstream (unused in the proof but given in the problem)
def rate_upstream : ℝ := 17.0

-- Prove that the man's rate when rowing downstream is as stated given the conditions
theorem rate_downstream : rate_still_water + rate_current = 32 := by
  simp [rate_still_water, rate_current]
  norm_num

end NUMINAMATH_GPT_rate_downstream_l2341_234171


namespace NUMINAMATH_GPT_cost_price_of_article_l2341_234192

variable (C : ℝ)
variable (h1 : (0.18 * C - 0.09 * C = 72))

theorem cost_price_of_article : C = 800 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l2341_234192


namespace NUMINAMATH_GPT_three_term_arithmetic_seq_l2341_234163

noncomputable def arithmetic_sequence_squares (x y z : ℤ) : Prop :=
  x^2 + z^2 = 2 * y^2

theorem three_term_arithmetic_seq (x y z : ℤ) :
  (∃ a b : ℤ, a = (x + z) / 2 ∧ b = (x - z) / 2 ∧ x^2 + z^2 = 2 * y^2) ↔
  arithmetic_sequence_squares x y z :=
by
  sorry

end NUMINAMATH_GPT_three_term_arithmetic_seq_l2341_234163


namespace NUMINAMATH_GPT_arithmetic_prog_leq_l2341_234149

def t3 (s : List ℤ) : ℕ := 
  sorry -- Placeholder for function calculating number of 3-term arithmetic progressions

theorem arithmetic_prog_leq (a : List ℤ) (k : ℕ) (h_sorted : a = List.range k)
  : t3 a ≤ t3 (List.range k) :=
sorry -- Proof here

end NUMINAMATH_GPT_arithmetic_prog_leq_l2341_234149


namespace NUMINAMATH_GPT_intersecting_points_of_curves_l2341_234159

theorem intersecting_points_of_curves :
  (∀ x y, (y = 2 * x^3 + x^2 - 5 * x + 2) ∧ (y = 3 * x^2 + 6 * x - 4) → 
   (x = -1 ∧ y = -7) ∨ (x = 3 ∧ y = 41)) := sorry

end NUMINAMATH_GPT_intersecting_points_of_curves_l2341_234159


namespace NUMINAMATH_GPT_smallest_possible_value_of_M_l2341_234102

theorem smallest_possible_value_of_M (a b c d e : ℕ) (h1 : a + b + c + d + e = 3060) 
    (h2 : a + e ≥ 1300) :
    ∃ M : ℕ, M = max (max (a + b) (max (b + c) (max (c + d) (d + e)))) ∧ M = 1174 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_M_l2341_234102


namespace NUMINAMATH_GPT_clock_hands_overlap_l2341_234147

theorem clock_hands_overlap (t : ℝ) :
  (∀ (h_angle m_angle : ℝ), h_angle = 30 + 0.5 * t ∧ m_angle = 6 * t ∧ h_angle = m_angle ∧ h_angle = 45) → t = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_clock_hands_overlap_l2341_234147


namespace NUMINAMATH_GPT_monotonic_increasing_interval_of_f_l2341_234141

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.logb (1/2) (x^2))

theorem monotonic_increasing_interval_of_f : 
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 0 ∧ -1 ≤ x₂ ∧ x₂ < 0 ∧ x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧ 
  (∀ x : ℝ, f x ≥ 0) := sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_of_f_l2341_234141


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2341_234105

def p (a : ℝ) : Prop := (a - 1) * (a - 2) = 0
def q (a : ℝ) : Prop := a = 1

theorem necessary_but_not_sufficient (a : ℝ) : 
  (q a → p a) ∧ (p a → q a → False) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2341_234105


namespace NUMINAMATH_GPT_inequality_solution_l2341_234110

theorem inequality_solution (x : ℝ) : 2 * (3 * x - 2) > x + 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2341_234110


namespace NUMINAMATH_GPT_proof_problem1_proof_problem2_proof_problem3_l2341_234145

-- Definition of the three mathematical problems
def problem1 : Prop := 8 / (-2) - (-4) * (-3) = -16

def problem2 : Prop := -2^3 + (-3) * ((-2)^3 + 5) = 1

def problem3 (x : ℝ) : Prop := (2 * x^2)^3 * x^2 - x^10 / x^2 = 7 * x^8

-- Statements of the proofs required
theorem proof_problem1 : problem1 :=
by sorry

theorem proof_problem2 : problem2 :=
by sorry

theorem proof_problem3 (x : ℝ) : problem3 x :=
by sorry

end NUMINAMATH_GPT_proof_problem1_proof_problem2_proof_problem3_l2341_234145


namespace NUMINAMATH_GPT_solve_for_x0_l2341_234136

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 2 then x^2 + 2 else 2 * x

theorem solve_for_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 ∨ x0 = - Real.sqrt 6 :=
  by
  sorry

end NUMINAMATH_GPT_solve_for_x0_l2341_234136


namespace NUMINAMATH_GPT_B_is_subset_of_A_l2341_234178
open Set

def A := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def B := {y : ℤ | ∃ k : ℤ, y = 4 * k}

theorem B_is_subset_of_A : B ⊆ A :=
by sorry

end NUMINAMATH_GPT_B_is_subset_of_A_l2341_234178


namespace NUMINAMATH_GPT_max_discount_rate_l2341_234111

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end NUMINAMATH_GPT_max_discount_rate_l2341_234111


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l2341_234124

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 2
  let r := (1 : ℝ) / 2
  (a + a * r + a * r^2 + a * r^3 + ∑' n : ℕ, a * r^n) = 1 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l2341_234124


namespace NUMINAMATH_GPT_max_visible_unit_cubes_l2341_234119

def cube_size := 11
def total_unit_cubes := cube_size ^ 3

def visible_unit_cubes (n : ℕ) : ℕ :=
  (n * n) + (n * (n - 1)) + ((n - 1) * (n - 1))

theorem max_visible_unit_cubes : 
  visible_unit_cubes cube_size = 331 := by
  sorry

end NUMINAMATH_GPT_max_visible_unit_cubes_l2341_234119


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l2341_234103

theorem arithmetic_sequence_a5 (a_n : ℕ → ℝ) 
  (h_arith : ∀ n, a_n (n+1) - a_n n = a_n (n+2) - a_n (n+1))
  (h_condition : a_n 1 + a_n 9 = 10) :
  a_n 5 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l2341_234103


namespace NUMINAMATH_GPT_find_m_l2341_234151

theorem find_m (m : ℝ) :
  (∀ x y, x + (m^2 - m) * y = 4 * m - 1 → ∀ x y, 2 * x - y - 5 = 0 → (-1 / (m^2 - m)) = -1 / 2) → 
  (m = -1 ∨ m = 2) :=
sorry

end NUMINAMATH_GPT_find_m_l2341_234151


namespace NUMINAMATH_GPT_verify_salary_problem_l2341_234120

def salary_problem (W : ℕ) (S_old : ℕ) (S_new : ℕ := 780) (n : ℕ := 9) : Prop :=
  (W + S_old) / n = 430 ∧ (W + S_new) / n = 420 → S_old = 870

theorem verify_salary_problem (W S_old : ℕ) (h1 : (W + S_old) / 9 = 430) (h2 : (W + 780) / 9 = 420) : S_old = 870 :=
by {
  sorry
}

end NUMINAMATH_GPT_verify_salary_problem_l2341_234120


namespace NUMINAMATH_GPT_boat_speed_l2341_234150

theorem boat_speed (v : ℝ) (h1 : 5 + v = 30) : v = 25 :=
by 
  -- Solve for the speed of the second boat
  sorry

end NUMINAMATH_GPT_boat_speed_l2341_234150


namespace NUMINAMATH_GPT_factorization_of_polynomial_l2341_234161

noncomputable def p (x : ℤ) : ℤ := x^15 + x^10 + x^5 + 1
noncomputable def f (x : ℤ) : ℤ := x^3 + x^2 + x + 1
noncomputable def g (x : ℤ) : ℤ := x^12 - x^11 + x^9 - x^8 + x^6 - x^5 + x^3 - x^2 + x - 1

theorem factorization_of_polynomial : ∀ x : ℤ, p x = f x * g x :=
by sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l2341_234161


namespace NUMINAMATH_GPT_tetrahedron_surface_area_l2341_234181

theorem tetrahedron_surface_area (a : ℝ) (h : a = Real.sqrt 2) :
  let R := (a * Real.sqrt 6) / 4
  let S := 4 * Real.pi * R^2
  S = 3 * Real.pi := by
  /- Proof here -/
  sorry

end NUMINAMATH_GPT_tetrahedron_surface_area_l2341_234181


namespace NUMINAMATH_GPT_john_drove_total_distance_l2341_234117

-- Define different rates and times for John's trip
def rate1 := 45 -- mph
def rate2 := 55 -- mph
def time1 := 2 -- hours
def time2 := 3 -- hours

-- Define the distances for each segment of the trip
def distance1 := rate1 * time1
def distance2 := rate2 * time2

-- Define the total distance
def total_distance := distance1 + distance2

-- The theorem to prove that John drove 255 miles in total
theorem john_drove_total_distance : total_distance = 255 :=
by
  sorry

end NUMINAMATH_GPT_john_drove_total_distance_l2341_234117


namespace NUMINAMATH_GPT_yoga_studio_women_count_l2341_234115

theorem yoga_studio_women_count :
  ∃ W : ℕ, 
  (8 * 190) + (W * 120) = 14 * 160 ∧ W = 6 :=
by 
  existsi (6);
  sorry

end NUMINAMATH_GPT_yoga_studio_women_count_l2341_234115


namespace NUMINAMATH_GPT_zero_of_function_l2341_234134

theorem zero_of_function : ∃ x : ℝ, (x + 1)^2 = 0 :=
by
  use -1
  sorry

end NUMINAMATH_GPT_zero_of_function_l2341_234134


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2341_234140

theorem problem1 (h : Real.cos 75 * Real.sin 75 = 1 / 2) : False :=
by
  sorry

theorem problem2 : (1 + Real.tan 15) / (1 - Real.tan 15) = Real.sqrt 3 :=
by
  sorry

theorem problem3 : Real.tan 20 + Real.tan 25 + Real.tan 20 * Real.tan 25 = 1 :=
by
  sorry

theorem problem4 (θ : Real) (h1 : Real.sin (2 * θ) ≠ 0) : (1 / Real.tan θ - 1 / Real.tan (2 * θ) = 1 / Real.sin (2 * θ)) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2341_234140


namespace NUMINAMATH_GPT_garment_industry_initial_men_l2341_234108

theorem garment_industry_initial_men (M : ℕ) :
  (M * 8 * 10 = 6 * 20 * 8) → M = 12 :=
by
  sorry

end NUMINAMATH_GPT_garment_industry_initial_men_l2341_234108


namespace NUMINAMATH_GPT_avg_equivalence_l2341_234155

-- Definition of binary average [a, b]
def avg2 (a b : ℤ) : ℤ := (a + b) / 2

-- Definition of ternary average {a, b, c}
def avg3 (a b c : ℤ) : ℤ := (a + b + c) / 3

-- Lean statement for proving the given problem
theorem avg_equivalence : avg3 (avg3 2 2 (-1)) (avg2 3 (-1)) 1 = 1 := by
  sorry

end NUMINAMATH_GPT_avg_equivalence_l2341_234155


namespace NUMINAMATH_GPT_factorize_expression_l2341_234167

theorem factorize_expression (x y : ℝ) : 
  x^3 - x*y^2 = x * (x + y) * (x - y) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l2341_234167


namespace NUMINAMATH_GPT_find_f_x_minus_1_l2341_234129

theorem find_f_x_minus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x ^ 2 + 2 * x) :
  ∀ x : ℤ, f (x - 1) = x ^ 2 - 2 * x :=
by
  sorry

end NUMINAMATH_GPT_find_f_x_minus_1_l2341_234129


namespace NUMINAMATH_GPT_marty_combinations_l2341_234198

theorem marty_combinations : 
  let C := 5
  let P := 4
  C * P = 20 :=
by
  sorry

end NUMINAMATH_GPT_marty_combinations_l2341_234198


namespace NUMINAMATH_GPT_total_volume_of_cubes_l2341_234158

theorem total_volume_of_cubes :
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  sarah_volume + tom_volume = 472 := by
  -- Definitions coming from conditions
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  -- Total volume of all cubes
  have h : sarah_volume + tom_volume = 472 := by sorry

  exact h

end NUMINAMATH_GPT_total_volume_of_cubes_l2341_234158


namespace NUMINAMATH_GPT_special_numbers_count_l2341_234184

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_zero (n : ℕ) : Prop := n % 10 = 0
def divisible_by_30 (n : ℕ) : Prop := n % 30 = 0

-- Define the count of numbers with the specified conditions
noncomputable def count_special_numbers : ℕ :=
  (9990 - 1020) / 30 + 1

-- The proof problem
theorem special_numbers_count : count_special_numbers = 300 := sorry

end NUMINAMATH_GPT_special_numbers_count_l2341_234184


namespace NUMINAMATH_GPT_sum_due_l2341_234190

theorem sum_due (BD TD S : ℝ) (hBD : BD = 18) (hTD : TD = 15) (hRel : BD = TD + (TD^2 / S)) : S = 75 :=
by
  sorry

end NUMINAMATH_GPT_sum_due_l2341_234190


namespace NUMINAMATH_GPT_tan_two_beta_l2341_234142

variables {α β : Real}

theorem tan_two_beta (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 7) : Real.tan (2 * β) = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_two_beta_l2341_234142


namespace NUMINAMATH_GPT_time_ratio_xiao_ming_schools_l2341_234166

theorem time_ratio_xiao_ming_schools
  (AB BC CD : ℝ) 
  (flat_speed uphill_speed downhill_speed : ℝ)
  (h1 : AB + BC + CD = 1) 
  (h2 : AB / BC = 1 / 2)
  (h3 : BC / CD = 2 / 1)
  (h4 : flat_speed / uphill_speed = 3 / 2)
  (h5 : uphill_speed / downhill_speed = 2 / 4) :
  (AB / flat_speed + BC / uphill_speed + CD / downhill_speed) / 
  (AB / flat_speed + BC / downhill_speed + CD / uphill_speed) = 19 / 16 :=
by
  sorry

end NUMINAMATH_GPT_time_ratio_xiao_ming_schools_l2341_234166


namespace NUMINAMATH_GPT_christina_walking_speed_l2341_234121

noncomputable def christina_speed : ℕ :=
  let distance_between := 270
  let jack_speed := 4
  let lindy_total_distance := 240
  let lindy_speed := 8
  let meeting_time := lindy_total_distance / lindy_speed
  let jack_covered := jack_speed * meeting_time
  let remaining_distance := distance_between - jack_covered
  remaining_distance / meeting_time

theorem christina_walking_speed : christina_speed = 5 := by
  -- Proof will be provided here to verify the theorem, but for now, we use sorry to skip it
  sorry

end NUMINAMATH_GPT_christina_walking_speed_l2341_234121


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l2341_234170

def p (x : ℕ) : ℕ := x^5 - 2 * x^3 + 4 * x + 5

theorem remainder_when_divided_by_x_minus_2 : p 2 = 29 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l2341_234170
