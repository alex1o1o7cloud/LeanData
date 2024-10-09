import Mathlib

namespace determine_suit_cost_l2144_214400

def cost_of_suit (J B V : ℕ) : Prop :=
  (J + B + V = 150)

theorem determine_suit_cost
  (J B V : ℕ)
  (h1 : J = B + V)
  (h2 : J + 2 * B = 175)
  (h3 : B + 2 * V = 100) :
  cost_of_suit J B V :=
by
  sorry

end determine_suit_cost_l2144_214400


namespace infinite_geometric_series_sum_l2144_214435

theorem infinite_geometric_series_sum :
  ∑' (n : ℕ), (1 : ℚ) * (-1 / 4 : ℚ) ^ n = 4 / 5 :=
by
  sorry

end infinite_geometric_series_sum_l2144_214435


namespace total_profit_l2144_214418

theorem total_profit (C_profit : ℝ) (x : ℝ) (h1 : 4 * x = 48000) : 12 * x = 144000 :=
by
  sorry

end total_profit_l2144_214418


namespace nat_square_not_div_factorial_l2144_214425

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end nat_square_not_div_factorial_l2144_214425


namespace intersection_A_B_l2144_214429

def set_A : Set ℝ := { x | abs (x - 1) < 2 }
def set_B : Set ℝ := { x | Real.log x / Real.log 2 > Real.log x / Real.log 3 }

theorem intersection_A_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l2144_214429


namespace h_at_2_l2144_214416

noncomputable def h (x : ℝ) : ℝ := 
(x + 2) * (x - 1) * (x + 4) * (x - 3) - x^2

theorem h_at_2 : 
  h (-2) = -4 ∧ h (1) = -1 ∧ h (-4) = -16 ∧ h (3) = -9 → h (2) = -28 := 
by
  intro H
  sorry

end h_at_2_l2144_214416


namespace binomial_expansion_coefficients_equal_l2144_214461

theorem binomial_expansion_coefficients_equal (n : ℕ) (h : n ≥ 6)
  (h_eq : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : n = 7 := by
  sorry

end binomial_expansion_coefficients_equal_l2144_214461


namespace apples_per_box_l2144_214432

variable (A : ℕ) -- Number of apples packed in a box

-- Conditions
def normal_boxes_per_day := 50
def days_per_week := 7
def boxes_first_week := normal_boxes_per_day * days_per_week * A
def boxes_second_week := (normal_boxes_per_day * A - 500) * days_per_week
def total_apples := 24500

-- Theorem
theorem apples_per_box : boxes_first_week + boxes_second_week = total_apples → A = 40 :=
by
  sorry

end apples_per_box_l2144_214432


namespace quadratic_properties_l2144_214444

theorem quadratic_properties (d e f : ℝ)
  (h1 : d * 1^2 + e * 1 + f = 3)
  (h2 : d * 2^2 + e * 2 + f = 0)
  (h3 : d * 9 + e * 3 + f = -3) :
  d + e + 2 * f = 19.5 :=
sorry

end quadratic_properties_l2144_214444


namespace problem_statement_l2144_214409

-- Definitions of propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x = 1
def q : Prop := ∀ x : ℝ, x^2 > 0

-- The proof problem
theorem problem_statement : ¬ (¬ p ∧ ¬ q) :=
by 
  -- sorry here indicates that actual proof is omitted
  sorry

end problem_statement_l2144_214409


namespace total_number_of_values_l2144_214440

theorem total_number_of_values (S n : ℕ) (h1 : (S - 165 + 135) / n = 150) (h2 : S / n = 151) : n = 30 :=
by {
  sorry
}

end total_number_of_values_l2144_214440


namespace no_n_such_that_n_times_s_is_20222022_l2144_214422

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem
theorem no_n_such_that_n_times_s_is_20222022 :
  ∀ n : ℕ, n * sum_of_digits n ≠ 20222022 :=
by
  sorry

end no_n_such_that_n_times_s_is_20222022_l2144_214422


namespace max_volumes_on_fedor_shelf_l2144_214479

theorem max_volumes_on_fedor_shelf 
  (S s1 s2 n : ℕ) 
  (h1 : S + s1 ≥ (n - 2) / 2) 
  (h2 : S + s2 < (n - 2) / 3) 
  : n = 12 := 
sorry

end max_volumes_on_fedor_shelf_l2144_214479


namespace spend_on_laundry_detergent_l2144_214492

def budget : ℕ := 60
def price_shower_gel : ℕ := 4
def num_shower_gels : ℕ := 4
def price_toothpaste : ℕ := 3
def remaining_budget : ℕ := 30

theorem spend_on_laundry_detergent : 
  (budget - remaining_budget) = (num_shower_gels * price_shower_gel + price_toothpaste) + 11 := 
by
  sorry

end spend_on_laundry_detergent_l2144_214492


namespace last_four_digits_5_2011_l2144_214415

theorem last_four_digits_5_2011 :
  (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_5_2011_l2144_214415


namespace arithmetic_sequence_sum_first_three_terms_l2144_214427

theorem arithmetic_sequence_sum_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 4) (h5 : a 5 = 7) (h6 : a 6 = 10) : a 1 + a 2 + a 3 = -6 :=
sorry

end arithmetic_sequence_sum_first_three_terms_l2144_214427


namespace smallest_positive_x_for_maximum_sine_sum_l2144_214448

theorem smallest_positive_x_for_maximum_sine_sum :
  ∃ x : ℝ, (0 < x) ∧ (∃ k m : ℕ, x = 450 + 1800 * k ∧ x = 630 + 2520 * m ∧ x = 12690) := by
  sorry

end smallest_positive_x_for_maximum_sine_sum_l2144_214448


namespace max_value_of_a_l2144_214498

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - a * x + 1

theorem max_value_of_a :
  ∃ (a : ℝ), (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) → |f a x| ≤ 1) ∧ a = 8 := by
  sorry

end max_value_of_a_l2144_214498


namespace quadratic_functions_count_correct_even_functions_count_correct_l2144_214464

def num_coefficients := 4
def valid_coefficients := [-1, 0, 1, 2]

def count_quadratic_functions : ℕ :=
  num_coefficients * num_coefficients * (num_coefficients - 1)

def count_even_functions : ℕ :=
  (num_coefficients - 1) * (num_coefficients - 2)

def total_quad_functions_correct : Prop := count_quadratic_functions = 18
def total_even_functions_correct : Prop := count_even_functions = 6

theorem quadratic_functions_count_correct : total_quad_functions_correct :=
by sorry

theorem even_functions_count_correct : total_even_functions_correct :=
by sorry

end quadratic_functions_count_correct_even_functions_count_correct_l2144_214464


namespace arithmetic_sequence_term_l2144_214412

theorem arithmetic_sequence_term (a : ℕ → ℤ) (d : ℤ) (n : ℕ) :
  a 5 = 33 ∧ a 45 = 153 ∧ (∀ n, a n = a 1 + (n - 1) * d) ∧ a n = 201 → n = 61 :=
by
  sorry

end arithmetic_sequence_term_l2144_214412


namespace find_b_of_quadratic_eq_l2144_214453

theorem find_b_of_quadratic_eq (a b c y1 y2 : ℝ) 
    (h1 : y1 = a * (2:ℝ)^2 + b * (2:ℝ) + c) 
    (h2 : y2 = a * (-2:ℝ)^2 + b * (-2:ℝ) + c) 
    (h_diff : y1 - y2 = 4) : b = 1 :=
by
  sorry

end find_b_of_quadratic_eq_l2144_214453


namespace ratio_total_length_to_perimeter_l2144_214402

noncomputable def length_initial : ℝ := 25
noncomputable def width_initial : ℝ := 15
noncomputable def extension : ℝ := 10
noncomputable def length_total : ℝ := length_initial + extension
noncomputable def perimeter_new : ℝ := 2 * (length_total + width_initial)
noncomputable def ratio : ℝ := length_total / perimeter_new

theorem ratio_total_length_to_perimeter : ratio = 35 / 100 := by
  sorry

end ratio_total_length_to_perimeter_l2144_214402


namespace quadratic_has_two_distinct_real_roots_l2144_214410

variable {R : Type} [LinearOrderedField R]

theorem quadratic_has_two_distinct_real_roots (c d : R) :
  ∀ x : R, (x + c) * (x + d) - (2 * x + c + d) = 0 → 
  (x + c)^2 + 4 > 0 :=
by
  intros x h
  -- Proof (skipped)
  sorry

end quadratic_has_two_distinct_real_roots_l2144_214410


namespace percentage_for_x_plus_y_l2144_214465

theorem percentage_for_x_plus_y (x y : Real) (P : Real) 
  (h1 : 0.60 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := 
by 
  sorry

end percentage_for_x_plus_y_l2144_214465


namespace employees_use_public_transportation_l2144_214472

theorem employees_use_public_transportation
    (total_employees : ℕ)
    (drive_percentage : ℝ)
    (public_transportation_fraction : ℝ)
    (h1 : total_employees = 100)
    (h2 : drive_percentage = 0.60)
    (h3 : public_transportation_fraction = 0.50) :
    ((total_employees * (1 - drive_percentage)) * public_transportation_fraction) = 20 :=
by
    sorry

end employees_use_public_transportation_l2144_214472


namespace algebraic_expression_evaluation_l2144_214458

theorem algebraic_expression_evaluation (x m : ℝ) (h1 : 5 * (2 - 1) + 3 * m * 2 = -7) (h2 : m = -2) :
  5 * (x - 1) + 3 * m * x = -1 ↔ x = -4 :=
by
  sorry

end algebraic_expression_evaluation_l2144_214458


namespace sum_three_ways_l2144_214436

theorem sum_three_ways (n : ℕ) (h : n > 0) : 
  ∃ k, k = (n^2) / 12 ∧ k = (n^2) / 12 :=
sorry

end sum_three_ways_l2144_214436


namespace sum_of_squares_l2144_214496

theorem sum_of_squares (a b c : ℝ)
  (h1 : a + b + c = 19)
  (h2 : a * b + b * c + c * a = 131) :
  a^2 + b^2 + c^2 = 99 :=
by
  sorry

end sum_of_squares_l2144_214496


namespace find_xyz_l2144_214445

variable (x y z : ℝ)

theorem find_xyz (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 192) : x * y * z = 842 :=
sorry

end find_xyz_l2144_214445


namespace stuart_initial_marbles_l2144_214417

theorem stuart_initial_marbles (B S : ℝ) (h1 : B = 60) (h2 : 0.40 * B = 24) (h3 : S + 24 = 80) : S = 56 :=
by
  sorry

end stuart_initial_marbles_l2144_214417


namespace abcd_product_l2144_214431

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

axiom a_eq : a = Real.sqrt (4 - Real.sqrt (5 - a))
axiom b_eq : b = Real.sqrt (4 + Real.sqrt (5 - b))
axiom c_eq : c = Real.sqrt (4 - Real.sqrt (5 + c))
axiom d_eq : d = Real.sqrt (4 + Real.sqrt (5 + d))

theorem abcd_product : a * b * c * d = 11 := sorry

end abcd_product_l2144_214431


namespace expression_evaluation_l2144_214406

theorem expression_evaluation (a : ℕ) (h : a = 1580) : 
  2 * a - ((2 * a - 3) / (a + 1) - (a + 1) / (2 - 2 * a) - (a^2 + 3) / 2) * ((a^3 + 1) / (a^2 - a)) + 2 / a = 2 := 
sorry

end expression_evaluation_l2144_214406


namespace lillian_candies_addition_l2144_214467

noncomputable def lillian_initial_candies : ℕ := 88
noncomputable def lillian_father_candies : ℕ := 5
noncomputable def lillian_total_candies : ℕ := 93

theorem lillian_candies_addition : lillian_initial_candies + lillian_father_candies = lillian_total_candies := by
  sorry

end lillian_candies_addition_l2144_214467


namespace evaluation_result_l2144_214473

noncomputable def evaluate_expression : ℝ :=
  let a := 210
  let b := 206
  let numerator := 980 ^ 2
  let denominator := a^2 - b^2
  numerator / denominator

theorem evaluation_result : evaluate_expression = 577.5 := 
  sorry  -- Placeholder for the proof

end evaluation_result_l2144_214473


namespace find_sin_minus_cos_l2144_214414

variable {a : ℝ}
variable {α : ℝ}

def point_of_angle (a : ℝ) (h : a < 0) := (3 * a, -4 * a)

theorem find_sin_minus_cos (a : ℝ) (h : a < 0) (ha : point_of_angle a h = (3 * a, -4 * a)) (sinα : ℝ) (cosα : ℝ) :
  sinα = 4 / 5 → cosα = -3 / 5 → sinα - cosα = 7 / 5 :=
by sorry

end find_sin_minus_cos_l2144_214414


namespace tim_total_score_l2144_214474

-- Definitions from conditions
def single_line_points : ℕ := 1000
def tetris_points : ℕ := 8 * single_line_points
def doubled_tetris_points : ℕ := 2 * tetris_points
def num_singles : ℕ := 6
def num_tetrises : ℕ := 4
def consecutive_tetrises : ℕ := 2
def regular_tetrises : ℕ := num_tetrises - consecutive_tetrises

-- Total score calculation
def total_score : ℕ :=
  num_singles * single_line_points +
  regular_tetrises * tetris_points +
  consecutive_tetrises * doubled_tetris_points

-- Prove that Tim's total score is 54000
theorem tim_total_score : total_score = 54000 :=
by 
  sorry

end tim_total_score_l2144_214474


namespace arithmetic_example_l2144_214419

theorem arithmetic_example : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end arithmetic_example_l2144_214419


namespace one_fourth_way_from_x1_to_x2_l2144_214493

-- Definitions of the points
def x1 : ℚ := 1 / 5
def x2 : ℚ := 4 / 5

-- Problem statement: Prove that one fourth of the way from x1 to x2 is 7/20
theorem one_fourth_way_from_x1_to_x2 : (3 * x1 + 1 * x2) / 4 = 7 / 20 := by
  sorry

end one_fourth_way_from_x1_to_x2_l2144_214493


namespace find_m_l2144_214478

theorem find_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := 
by
  sorry

end find_m_l2144_214478


namespace surface_area_parallelepiped_l2144_214494

theorem surface_area_parallelepiped (a b : ℝ) :
  ∃ S : ℝ, (S = 3 * a * b) :=
sorry

end surface_area_parallelepiped_l2144_214494


namespace negative_integer_reciprocal_of_d_l2144_214403

def a : ℚ := 3
def b : ℚ := |1 / 3|
def c : ℚ := -2
def d : ℚ := -1 / 2

theorem negative_integer_reciprocal_of_d (h : d ≠ 0) : ∃ k : ℤ, (d⁻¹ : ℚ) = ↑k ∧ k < 0 :=
by
  sorry

end negative_integer_reciprocal_of_d_l2144_214403


namespace solution_set_l2144_214463

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end solution_set_l2144_214463


namespace elise_spent_on_comic_book_l2144_214483

-- Define the initial amount of money Elise had
def initial_amount : ℤ := 8

-- Define the amount saved from allowance
def saved_amount : ℤ := 13

-- Define the amount spent on puzzle
def spent_on_puzzle : ℤ := 18

-- Define the amount left after all expenditures
def amount_left : ℤ := 1

-- Define the total amount of money Elise had after saving
def total_amount : ℤ := initial_amount + saved_amount

-- Define the total amount spent which equals
-- the sum of amount spent on the comic book and the puzzle
def total_spent : ℤ := total_amount - amount_left

-- Define the amount spent on the comic book as the proposition to be proved
def spent_on_comic_book : ℤ := total_spent - spent_on_puzzle

-- State the theorem to prove how much Elise spent on the comic book
theorem elise_spent_on_comic_book : spent_on_comic_book = 2 :=
by
  sorry

end elise_spent_on_comic_book_l2144_214483


namespace Problem_l2144_214446

theorem Problem (N : ℕ) (hn : N = 16) :
  (Nat.choose N 5) = 2002 := 
by 
  rw [hn] 
  sorry

end Problem_l2144_214446


namespace pyramid_volume_pyramid_surface_area_l2144_214434

noncomputable def volume_of_pyramid (l : ℝ) := (l^3 * Real.sqrt 2) / 12

noncomputable def surface_area_of_pyramid (l : ℝ) := (l^2 * (2 + Real.sqrt 2)) / 2

theorem pyramid_volume (l : ℝ) :
  volume_of_pyramid l = (l^3 * Real.sqrt 2) / 12 :=
sorry

theorem pyramid_surface_area (l : ℝ) :
  surface_area_of_pyramid l = (l^2 * (2 + Real.sqrt 2)) / 2 :=
sorry

end pyramid_volume_pyramid_surface_area_l2144_214434


namespace gcd_lcm_sum_l2144_214447

theorem gcd_lcm_sum (a b : ℕ) (h : a = 1999 * b) : Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end gcd_lcm_sum_l2144_214447


namespace find_g_of_3_l2144_214466

theorem find_g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 5 * g (1 / x) = 2 * x) : g 3 = -32 / 63 :=
by sorry

end find_g_of_3_l2144_214466


namespace sum_arith_seq_elems_l2144_214404

noncomputable def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem sum_arith_seq_elems (a d : ℝ) 
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 5 + arithmetic_seq a d 8 + arithmetic_seq a d 11 = 48) :
  arithmetic_seq a d 6 + arithmetic_seq a d 7 = 24 := 
by 
  sorry

end sum_arith_seq_elems_l2144_214404


namespace region_to_the_upper_left_of_line_l2144_214497

variable (x y : ℝ)

def line_eqn := 3 * x - 2 * y - 6 = 0

def region := 3 * x - 2 * y - 6 < 0

theorem region_to_the_upper_left_of_line :
  ∃ rect_upper_left, (rect_upper_left = region) := 
sorry

end region_to_the_upper_left_of_line_l2144_214497


namespace largest_divisible_by_digits_sum_l2144_214420

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_divisible_by_digits_sum : ∃ n, n < 900 ∧ n % digits_sum n = 0 ∧ ∀ m, m < 900 ∧ m % digits_sum m = 0 → m ≤ 888 :=
by
  sorry

end largest_divisible_by_digits_sum_l2144_214420


namespace vector_subtraction_magnitude_l2144_214470

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition1 : Real := 3 -- |a|
def condition2 : Real := 2 -- |b|
def condition3 : Real := 4 -- |a + b|

-- Proving the statement
theorem vector_subtraction_magnitude (h1 : ‖a‖ = condition1) (h2 : ‖b‖ = condition2) (h3 : ‖a + b‖ = condition3) :
  ‖a - b‖ = Real.sqrt 10 :=
by
  sorry

end vector_subtraction_magnitude_l2144_214470


namespace determine_clothes_l2144_214426

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l2144_214426


namespace value_of_a_squared_plus_b_squared_l2144_214487

theorem value_of_a_squared_plus_b_squared (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) :
  a^2 + b^2 = 68 :=
sorry

end value_of_a_squared_plus_b_squared_l2144_214487


namespace find_x_l2144_214443

noncomputable section

open Real

theorem find_x (x : ℝ) (hx : 0 < x ∧ x < 180) : 
  tan (120 * π / 180 - x * π / 180) = (sin (120 * π / 180) - sin (x * π / 180)) / (cos (120 * π / 180) - cos (x * π / 180)) →
  x = 100 :=
by
  sorry

end find_x_l2144_214443


namespace floor_sqrt_equality_l2144_214484

theorem floor_sqrt_equality (n : ℕ) : 
  (Int.floor (Real.sqrt (4 * n + 1))) = (Int.floor (Real.sqrt (4 * n + 3))) := 
by 
  sorry

end floor_sqrt_equality_l2144_214484


namespace solve_for_x_l2144_214488

theorem solve_for_x (x : ℝ) (h : (3 * x - 15) / 4 = (x + 9) / 5) : x = 10 :=
by {
  sorry
}

end solve_for_x_l2144_214488


namespace carly_butterfly_days_l2144_214482

-- Define the conditions
variable (x : ℕ) -- number of days Carly practices her butterfly stroke
def butterfly_hours_per_day := 3  -- hours per day for butterfly stroke
def backstroke_hours_per_day := 2  -- hours per day for backstroke stroke
def backstroke_days_per_week := 6  -- days per week for backstroke stroke
def total_hours_per_month := 96  -- total hours practicing swimming in a month
def weeks_in_month := 4  -- number of weeks in a month

-- The proof problem
theorem carly_butterfly_days :
  (butterfly_hours_per_day * x + backstroke_hours_per_day * backstroke_days_per_week) * weeks_in_month = total_hours_per_month
  → x = 4 := 
by
  sorry

end carly_butterfly_days_l2144_214482


namespace total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l2144_214468

def chocolate_sold : ℕ := 6 + 7 + 4 + 8 + 9 + 10 + 5
def vanilla_sold : ℕ := 4 + 5 + 3 + 7 + 6 + 8 + 4
def strawberry_sold : ℕ := 3 + 2 + 6 + 4 + 5 + 7 + 4

theorem total_chocolate_sold : chocolate_sold = 49 :=
by
  unfold chocolate_sold
  rfl

theorem total_vanilla_sold : vanilla_sold = 37 :=
by
  unfold vanilla_sold
  rfl

theorem total_strawberry_sold : strawberry_sold = 31 :=
by
  unfold strawberry_sold
  rfl

end total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l2144_214468


namespace q_can_complete_work_in_25_days_l2144_214450

-- Define work rates for p, q, and r
variables (W_p W_q W_r : ℝ)

-- Define total work
variable (W : ℝ)

-- Prove that q can complete the work in 25 days under given conditions
theorem q_can_complete_work_in_25_days
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 50) :
  W_q = W / 25 :=
by
  -- Given: W_p = W_q + W_r
  -- Given: W_p + W_q = W / 10
  -- Given: W_r = W / 50
  -- We need to prove: W_q = W / 25
  sorry

end q_can_complete_work_in_25_days_l2144_214450


namespace quadratic_inequality_solution_l2144_214452

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} :=
sorry

end quadratic_inequality_solution_l2144_214452


namespace tammy_speed_proof_l2144_214451

noncomputable def tammy_average_speed_second_day (v t : ℝ) :=
  v + 0.5

theorem tammy_speed_proof :
  ∃ v t : ℝ, 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    tammy_average_speed_second_day v t = 4 :=
by
  sorry

end tammy_speed_proof_l2144_214451


namespace product_equals_16896_l2144_214486

theorem product_equals_16896 (A B C D : ℕ) (h1 : A + B + C + D = 70)
  (h2 : A = 3 * C + 1) (h3 : B = 3 * C + 5) (h4 : C = C) (h5 : D = 3 * C^2) :
  A * B * C * D = 16896 :=
by
  sorry

end product_equals_16896_l2144_214486


namespace initial_number_of_girls_l2144_214456

theorem initial_number_of_girls (p : ℝ) (h : (0.4 * p - 2) / p = 0.3) : 0.4 * p = 8 := 
by
  sorry

end initial_number_of_girls_l2144_214456


namespace find_m_l2144_214471

variable {m : ℝ}

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vector_diff (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_m (hm: dot_product vector_a (vector_diff vector_a (vector_b m)) = 0) : m = 3 :=
  by
  sorry

end find_m_l2144_214471


namespace max_S_value_l2144_214485

noncomputable def maximize_S (a b c : ℝ) : ℝ :=
  (a^2 - a * b + b^2) * (b^2 - b * c + c^2) * (c^2 - c * a + a^2)

theorem max_S_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 3 →
  maximize_S a b c ≤ 12 :=
by sorry

end max_S_value_l2144_214485


namespace tan_sum_identity_l2144_214401

theorem tan_sum_identity (α β : ℝ)
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l2144_214401


namespace max_ratio_l2144_214439

theorem max_ratio {a b c d : ℝ} 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0) 
  (h2 : a^2 + b^2 + c^2 + d^2 = (a + b + c + d)^2 / 3) : 
  ∃ x, x = (7 + 2 * Real.sqrt 6) / 5 ∧ x = (a + c) / (b + d) :=
by
  sorry

end max_ratio_l2144_214439


namespace min_m_plus_n_l2144_214454

open Nat

theorem min_m_plus_n (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 45 * m = n^3) (h_mult_of_five : 5 ∣ n) :
  m + n = 90 :=
sorry

end min_m_plus_n_l2144_214454


namespace smallest_divisible_by_2022_l2144_214423

theorem smallest_divisible_by_2022 (n : ℕ) (N : ℕ) :
  (N = 20230110) ∧ (∃ k : ℕ, N = 2023 * 10^n + k) ∧ N % 2022 = 0 → 
  ∀ M: ℕ, (∃ m : ℕ, M = 2023 * 10^n + m) ∧ M % 2022 = 0 → N ≤ M :=
sorry

end smallest_divisible_by_2022_l2144_214423


namespace cos_of_double_angles_l2144_214433

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l2144_214433


namespace area_of_square_l2144_214457

noncomputable def square_area (s : ℝ) : ℝ := s ^ 2

theorem area_of_square
  {E F G H : Type}
  (ABCD : Type)
  (on_segments : E → F → G → H → Prop)
  (EG FH : ℝ)
  (angle_intersection : ℝ)
  (hEG : EG = 7)
  (hFH : FH = 8)
  (hangle : angle_intersection = 30) :
  ∃ s : ℝ, square_area s = 147 / 4 :=
sorry

end area_of_square_l2144_214457


namespace tan_alpha_sqrt3_l2144_214476

theorem tan_alpha_sqrt3 (α : ℝ) (h : Real.sin (α + 20 * Real.pi / 180) = Real.cos (α + 10 * Real.pi / 180) + Real.cos (α - 10 * Real.pi / 180)) :
  Real.tan α = Real.sqrt 3 := 
  sorry

end tan_alpha_sqrt3_l2144_214476


namespace no_perfect_powers_in_sequence_l2144_214462

noncomputable def nth_triplet (n : Nat) : Nat × Nat × Nat :=
  Nat.recOn n (2, 3, 5) (λ _ ⟨a, b, c⟩ => (a + c, a + b, b + c))

def is_perfect_power (x : Nat) : Prop :=
  ∃ (m : Nat) (k : Nat), k ≥ 2 ∧ m^k = x

theorem no_perfect_powers_in_sequence : ∀ (n : Nat), ∀ (a b c : Nat),
  nth_triplet n = (a, b, c) →
  ¬(is_perfect_power a ∨ is_perfect_power b ∨ is_perfect_power c) :=
by
  intros
  sorry

end no_perfect_powers_in_sequence_l2144_214462


namespace find_m_given_solution_l2144_214459

theorem find_m_given_solution (m x y : ℚ) (h₁ : x = 4) (h₂ : y = 3) (h₃ : m * x - y = 4) : m = 7 / 4 :=
by
  sorry

end find_m_given_solution_l2144_214459


namespace mn_values_l2144_214441

theorem mn_values (m n : ℤ) (h : m^2 * n^2 + m^2 + n^2 + 10 * m * n + 16 = 0) : 
  (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
  sorry

end mn_values_l2144_214441


namespace solve_eq_l2144_214424

theorem solve_eq {x : ℝ} (h : x + 2 * Real.sqrt x - 8 = 0) : x = 4 :=
by
  sorry

end solve_eq_l2144_214424


namespace probability_of_being_closer_to_origin_l2144_214411

noncomputable def probability_closer_to_origin 
  (rect : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2})
  (origin : ℝ × ℝ := (0, 0))
  (point : ℝ × ℝ := (4, 2))
  : ℚ :=
1/3

theorem probability_of_being_closer_to_origin :
  probability_closer_to_origin = 1/3 :=
by sorry

end probability_of_being_closer_to_origin_l2144_214411


namespace mike_spent_on_speakers_l2144_214455

-- Definitions of the conditions:
def total_car_parts_cost : ℝ := 224.87
def new_tires_cost : ℝ := 106.33

-- Statement of the proof problem:
theorem mike_spent_on_speakers : total_car_parts_cost - new_tires_cost = 118.54 :=
by
  sorry

end mike_spent_on_speakers_l2144_214455


namespace sin_double_angle_l2144_214491

theorem sin_double_angle (θ : ℝ) 
    (h : Real.sin (Real.pi / 4 + θ) = 1 / 3) : 
    Real.sin (2 * θ) = -7 * Real.sqrt 2 / 9 :=
by
  sorry

end sin_double_angle_l2144_214491


namespace prism_faces_l2144_214469

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end prism_faces_l2144_214469


namespace max_a_condition_l2144_214438

theorem max_a_condition (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) → a = -1 :=
by
  sorry

end max_a_condition_l2144_214438


namespace hats_per_yard_of_velvet_l2144_214421

theorem hats_per_yard_of_velvet
  (H : ℕ)
  (velvet_for_cloak : ℕ := 3)
  (total_velvet : ℕ := 21)
  (number_of_cloaks : ℕ := 6)
  (number_of_hats : ℕ := 12)
  (yards_for_6_cloaks : ℕ := number_of_cloaks * velvet_for_cloak)
  (remaining_yards_for_hats : ℕ := total_velvet - yards_for_6_cloaks)
  (hats_per_remaining_yard : ℕ := number_of_hats / remaining_yards_for_hats)
  : H = hats_per_remaining_yard :=
  by
  sorry

end hats_per_yard_of_velvet_l2144_214421


namespace ab_bc_ca_fraction_l2144_214449

theorem ab_bc_ca_fraction (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 10) (h3 : a * b * c = 12) :
    (a * b / c) + (b * c / a) + (c * a / b) = -17 / 3 := 
    sorry

end ab_bc_ca_fraction_l2144_214449


namespace ferrisWheelPeopleCount_l2144_214407

/-!
# Problem Description

We are given the following conditions:
- The ferris wheel has 6.0 seats.
- It has to run 2.333333333 times for everyone to get a turn.

We need to prove that the total number of people who want to ride the ferris wheel is 14.
-/

def ferrisWheelSeats : ℕ := 6
def ferrisWheelRuns : ℚ := 2333333333 / 1000000000

theorem ferrisWheelPeopleCount :
  (ferrisWheelSeats : ℚ) * ferrisWheelRuns = 14 :=
by
  sorry

end ferrisWheelPeopleCount_l2144_214407


namespace tom_age_is_19_l2144_214499

-- Define the ages of Carla, Tom, Dave, and Emily
variable (C : ℕ) -- Carla's age

-- Conditions
def tom_age := 2 * C - 1
def dave_age := C + 3
def emily_age := C / 2

-- Sum of their ages equating to 48
def total_age := C + tom_age C + dave_age C + emily_age C

-- Theorem to be proven
theorem tom_age_is_19 (h : total_age C = 48) : tom_age C = 19 := 
by {
  sorry
}

end tom_age_is_19_l2144_214499


namespace min_sum_abc_l2144_214408

theorem min_sum_abc (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hprod : a * b * c = 2550) : a + b + c ≥ 48 :=
by sorry

end min_sum_abc_l2144_214408


namespace distance_of_each_race_l2144_214437

theorem distance_of_each_race (d : ℝ) : 
  (∃ (d : ℝ), 
    let lake_speed := 3 
    let ocean_speed := 2.5 
    let num_races := 10 
    let total_time := 11
    let num_lake_races := num_races / 2
    let num_ocean_races := num_races / 2
    (num_lake_races * (d / lake_speed) + num_ocean_races * (d / ocean_speed) = total_time)) →
  d = 3 :=
sorry

end distance_of_each_race_l2144_214437


namespace rice_weight_per_container_l2144_214489

-- Given total weight of rice in pounds
def totalWeightPounds : ℚ := 25 / 2

-- Conversion factor from pounds to ounces
def poundsToOunces : ℚ := 16

-- Number of containers
def numberOfContainers : ℕ := 4

-- Total weight in ounces
def totalWeightOunces : ℚ := totalWeightPounds * poundsToOunces

-- Weight per container in ounces
def weightPerContainer : ℚ := totalWeightOunces / numberOfContainers

theorem rice_weight_per_container :
  weightPerContainer = 50 := 
sorry

end rice_weight_per_container_l2144_214489


namespace reggie_books_l2144_214495

/-- 
Reggie's father gave him $48. Reggie bought some books, each of which cost $2, 
and now he has $38 left. How many books did Reggie buy?
-/
theorem reggie_books (initial_amount spent_amount remaining_amount book_cost books_bought : ℤ)
  (h_initial : initial_amount = 48)
  (h_remaining : remaining_amount = 38)
  (h_book_cost : book_cost = 2)
  (h_spent : spent_amount = initial_amount - remaining_amount)
  (h_books_bought : books_bought = spent_amount / book_cost) :
  books_bought = 5 :=
by
  sorry

end reggie_books_l2144_214495


namespace number_of_sides_of_polygon_l2144_214428

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)
noncomputable def sum_known_angles : ℝ := 3780

theorem number_of_sides_of_polygon
  (n : ℕ)
  (h1 : sum_known_angles + missing_angle = sum_of_interior_angles n)
  (h2 : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a = 3 * c ∧ b = 3 * c ∧ a + b + c ≤ sum_known_angles) :
  n = 23 :=
sorry

end number_of_sides_of_polygon_l2144_214428


namespace cos_13pi_over_4_eq_neg_one_div_sqrt_two_l2144_214405

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end cos_13pi_over_4_eq_neg_one_div_sqrt_two_l2144_214405


namespace system_of_equations_solution_system_of_inequalities_no_solution_l2144_214430

-- Problem 1: Solving system of linear equations
theorem system_of_equations_solution :
  ∃ x y : ℝ, x - 3*y = -5 ∧ 2*x + 2*y = 6 ∧ x = 1 ∧ y = 2 := by
  sorry

-- Problem 2: Solving the system of inequalities
theorem system_of_inequalities_no_solution :
  ¬ (∃ x : ℝ, 2*x < -4 ∧ (1/2)*x - 5 > 1 - (3/2)*x) := by
  sorry

end system_of_equations_solution_system_of_inequalities_no_solution_l2144_214430


namespace inheritance_amount_l2144_214475

theorem inheritance_amount (x : ℝ) (hx1 : 0.25 * x + 0.1 * x = 15000) : x = 42857 := 
by
  -- Proof omitted
  sorry

end inheritance_amount_l2144_214475


namespace initial_average_mark_of_class_l2144_214480

theorem initial_average_mark_of_class
  (avg_excluded : ℝ) (n_excluded : ℕ) (avg_remaining : ℝ)
  (n_total : ℕ) : 
  avg_excluded = 70 → 
  n_excluded = 5 → 
  avg_remaining = 90 → 
  n_total = 10 → 
  (10 * (10 / n_total + avg_excluded - avg_remaining) / 10) = 80 :=
by 
  intros 
  sorry

end initial_average_mark_of_class_l2144_214480


namespace parabola_equation_standard_form_l2144_214490

theorem parabola_equation_standard_form (p : ℝ) (x y : ℝ)
    (h₁ : y^2 = 2 * p * x)
    (h₂ : y = -4)
    (h₃ : x = -2) : y^2 = -8 * x := by
  sorry

end parabola_equation_standard_form_l2144_214490


namespace sequence_total_sum_is_correct_l2144_214442

-- Define the sequence pattern
def sequence_sum : ℕ → ℤ
| 0       => 1
| 1       => -2
| 2       => -4
| 3       => 8
| (n + 4) => sequence_sum n + 4

-- Define the number of groups in the sequence
def num_groups : ℕ := 319

-- Define the sum of each individual group
def group_sum : ℤ := 3

-- Define the total sum of the sequence
def total_sum : ℤ := num_groups * group_sum

theorem sequence_total_sum_is_correct : total_sum = 957 := by
  sorry

end sequence_total_sum_is_correct_l2144_214442


namespace acute_triangle_on_perpendicular_lines_l2144_214460

theorem acute_triangle_on_perpendicular_lines :
  ∀ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) →
  ∃ (x y z : ℝ), (x^2 = (b^2 + c^2 - a^2) / 2) ∧ (y^2 = (a^2 + c^2 - b^2) / 2) ∧ (z^2 = (a^2 + b^2 - c^2) / 2) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) :=
by
  sorry

end acute_triangle_on_perpendicular_lines_l2144_214460


namespace bowl_weight_after_refill_l2144_214481

-- Define the problem conditions
def empty_bowl_weight : ℕ := 420
def day1_consumption : ℕ := 53
def day2_consumption : ℕ := 76
def day3_consumption : ℕ := 65
def day4_consumption : ℕ := 14

-- Define the total consumption over 4 days
def total_consumption : ℕ :=
  day1_consumption + day2_consumption + day3_consumption + day4_consumption

-- Define the final weight of the bowl after refilling
def final_bowl_weight : ℕ :=
  empty_bowl_weight + total_consumption

-- Statement to prove
theorem bowl_weight_after_refill : final_bowl_weight = 628 := by
  sorry

end bowl_weight_after_refill_l2144_214481


namespace max_value_f_l2144_214413

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.sin (2 * x) + (a - 2) * Real.cos (2 * x)

theorem max_value_f (a : ℝ) (h : a < 0)
  (symm : ∀ x, f a (x - π / 4) = f a (-x - π / 4)) :
  ∃ x, f a x = 4 * Real.sqrt 2 :=
sorry

end max_value_f_l2144_214413


namespace smallest_n_l2144_214477

theorem smallest_n
  (n : ℕ)
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : n % 7 = 1)
  (h7 : 8 ∣ n) :
  n = 1681 :=
  sorry

end smallest_n_l2144_214477
