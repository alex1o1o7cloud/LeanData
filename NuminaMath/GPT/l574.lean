import Mathlib

namespace NUMINAMATH_GPT_find_t_l574_57438

-- Define sets M and N
def M (t : ℝ) : Set ℝ := {1, t^2}
def N (t : ℝ) : Set ℝ := {-2, t + 2}

-- Goal: prove that t = 2 given M ∩ N ≠ ∅
theorem find_t (t : ℝ) (h : (M t ∩ N t).Nonempty) : t = 2 :=
sorry

end NUMINAMATH_GPT_find_t_l574_57438


namespace NUMINAMATH_GPT_range_of_x_l574_57464

theorem range_of_x (a b x : ℝ) (h1 : a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) → (-7 ≤ x ∧ x ≤ 11) :=
by
  -- we provide the exact statement we aim to prove.
  sorry

end NUMINAMATH_GPT_range_of_x_l574_57464


namespace NUMINAMATH_GPT_deg_d_eq_6_l574_57477

theorem deg_d_eq_6
  (f d q : Polynomial ℝ)
  (r : Polynomial ℝ)
  (hf : f.degree = 15)
  (hdq : (d * q + r) = f)
  (hq : q.degree = 9)
  (hr : r.degree = 4) :
  d.degree = 6 :=
by sorry

end NUMINAMATH_GPT_deg_d_eq_6_l574_57477


namespace NUMINAMATH_GPT_face_value_of_share_l574_57435

-- Let FV be the face value of each share.
-- Given conditions:
-- Dividend rate is 9%
-- Market value of each share is Rs. 42
-- Desired interest rate is 12%

theorem face_value_of_share (market_value : ℝ) (dividend_rate : ℝ) (interest_rate : ℝ) (FV : ℝ) :
  market_value = 42 ∧ dividend_rate = 0.09 ∧ interest_rate = 0.12 →
  0.09 * FV = 0.12 * market_value →
  FV = 56 :=
by
  sorry

end NUMINAMATH_GPT_face_value_of_share_l574_57435


namespace NUMINAMATH_GPT_percent_cities_less_than_50000_l574_57439

-- Definitions of the conditions
def percent_cities_50000_to_149999 := 40
def percent_cities_less_than_10000 := 35
def percent_cities_10000_to_49999 := 10
def percent_cities_150000_or_more := 15

-- Prove that the total percentage of cities with fewer than 50,000 residents is 45%
theorem percent_cities_less_than_50000 :
  percent_cities_less_than_10000 + percent_cities_10000_to_49999 = 45 :=
by
  sorry

end NUMINAMATH_GPT_percent_cities_less_than_50000_l574_57439


namespace NUMINAMATH_GPT_domain_of_function_l574_57491

open Real

theorem domain_of_function : 
  ∀ x, 
    (x + 1 ≠ 0) ∧ 
    (-x^2 - 3 * x + 4 > 0) ↔ 
    (-4 < x ∧ x < -1) ∨ ( -1 < x ∧ x < 1) := 
by 
  sorry

end NUMINAMATH_GPT_domain_of_function_l574_57491


namespace NUMINAMATH_GPT_fraction_of_air_conditioned_rooms_rented_l574_57499

variable (R : ℚ)
variable (h1 : R > 0)
variable (rented_rooms : ℚ := (3/4) * R)
variable (air_conditioned_rooms : ℚ := (3/5) * R)
variable (not_rented_rooms : ℚ := (1/4) * R)
variable (air_conditioned_not_rented_rooms : ℚ := (4/5) * not_rented_rooms)
variable (air_conditioned_rented_rooms : ℚ := air_conditioned_rooms - air_conditioned_not_rented_rooms)
variable (fraction_air_conditioned_rented : ℚ := air_conditioned_rented_rooms / air_conditioned_rooms)

theorem fraction_of_air_conditioned_rooms_rented :
  fraction_air_conditioned_rented = (2/3) := by
  sorry

end NUMINAMATH_GPT_fraction_of_air_conditioned_rooms_rented_l574_57499


namespace NUMINAMATH_GPT_cuboid_first_edge_length_l574_57441

theorem cuboid_first_edge_length (x : ℝ) (hx : 180 = x * 5 * 6) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_first_edge_length_l574_57441


namespace NUMINAMATH_GPT_score_on_fourth_board_l574_57423

theorem score_on_fourth_board 
  (score1 score2 score3 score4 : ℕ)
  (h1 : score1 = 30)
  (h2 : score2 = 38)
  (h3 : score3 = 41)
  (total_score : score1 + score2 = 2 * score4) :
  score4 = 34 := by
  sorry

end NUMINAMATH_GPT_score_on_fourth_board_l574_57423


namespace NUMINAMATH_GPT_nesting_doll_height_l574_57442

variable (H₀ : ℝ) (n : ℕ)

theorem nesting_doll_height (H₀ : ℝ) (Hₙ : ℝ) (H₁ : H₀ = 243) (H₂ : ∀ n : ℕ, Hₙ = H₀ * (2 / 3) ^ n) (H₃ : Hₙ = 32) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_nesting_doll_height_l574_57442


namespace NUMINAMATH_GPT_kimberly_peanuts_per_visit_l574_57470

theorem kimberly_peanuts_per_visit 
  (trips : ℕ) (total_peanuts : ℕ) 
  (h1 : trips = 3) 
  (h2 : total_peanuts = 21) : 
  total_peanuts / trips = 7 :=
by
  sorry

end NUMINAMATH_GPT_kimberly_peanuts_per_visit_l574_57470


namespace NUMINAMATH_GPT_bob_work_days_per_week_l574_57497

theorem bob_work_days_per_week (daily_hours : ℕ) (monthly_hours : ℕ) (average_days_per_month : ℕ) (days_per_week : ℕ)
  (h1 : daily_hours = 10)
  (h2 : monthly_hours = 200)
  (h3 : average_days_per_month = 30)
  (h4 : days_per_week = 7) :
  (monthly_hours / daily_hours) / (average_days_per_month / days_per_week) = 5 := by
  -- Now we will skip the proof itself. The focus here is on the structure.
  sorry

end NUMINAMATH_GPT_bob_work_days_per_week_l574_57497


namespace NUMINAMATH_GPT_value_of_business_l574_57445

theorem value_of_business (V : ℝ) (h₁ : (3/5) * (1/3) * V = 2000) : V = 10000 :=
by
  sorry

end NUMINAMATH_GPT_value_of_business_l574_57445


namespace NUMINAMATH_GPT_max_period_initial_phase_function_l574_57466

theorem max_period_initial_phase_function 
  (A ω ϕ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : A = 1/2) 
  (h2 : ω = 6) 
  (h3 : ϕ = π/4) 
  (h4 : ∀ x, f x = A * Real.sin (ω * x + ϕ)) : 
  ∀ x, f x = (1/2) * Real.sin (6 * x + (π/4)) :=
by
  sorry

end NUMINAMATH_GPT_max_period_initial_phase_function_l574_57466


namespace NUMINAMATH_GPT_problem_one_problem_two_l574_57419

theorem problem_one (α : ℝ) (h : Real.tan α = 2) : (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

theorem problem_two (α : ℝ) (h : Real.tan α = 2) (h_quadrant : α > π ∧ α < 3 * π / 2) : Real.cos α = - (Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_problem_one_problem_two_l574_57419


namespace NUMINAMATH_GPT_cartons_per_stack_l574_57498

-- Declare the variables and conditions
def total_cartons := 799
def stacks := 133

-- State the theorem
theorem cartons_per_stack : (total_cartons / stacks) = 6 := by
  sorry

end NUMINAMATH_GPT_cartons_per_stack_l574_57498


namespace NUMINAMATH_GPT_adjacent_zero_point_range_l574_57433

def f (x : ℝ) : ℝ := x - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_range (a : ℝ) :
  (∀ β, (∃ x, g x a = 0) → (|1 - β| ≤ 1 → (∃ x, f x = 0 → |x - β| ≤ 1))) →
  (2 ≤ a ∧ a ≤ 7 / 3) :=
sorry

end NUMINAMATH_GPT_adjacent_zero_point_range_l574_57433


namespace NUMINAMATH_GPT_number_of_performances_l574_57432

theorem number_of_performances (hanna_songs : ℕ) (mary_songs : ℕ) (alina_songs : ℕ) (tina_songs : ℕ)
    (hanna_cond : hanna_songs = 4)
    (mary_cond : mary_songs = 7)
    (alina_cond : 4 < alina_songs ∧ alina_songs < 7)
    (tina_cond : 4 < tina_songs ∧ tina_songs < 7) :
    ((hanna_songs + mary_songs + alina_songs + tina_songs) / 3) = 7 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_number_of_performances_l574_57432


namespace NUMINAMATH_GPT_find_circle_parameter_l574_57494

theorem find_circle_parameter (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y + c = 0 ∧ ((x + 4)^2 + (y - 1)^2 = 25)) → c = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_circle_parameter_l574_57494


namespace NUMINAMATH_GPT_shares_proportion_l574_57449

theorem shares_proportion (C D : ℕ) (h1 : D = 1500) (h2 : C = D + 500) : C / Nat.gcd C D = 4 ∧ D / Nat.gcd C D = 3 := by
  sorry

end NUMINAMATH_GPT_shares_proportion_l574_57449


namespace NUMINAMATH_GPT_factorial_ratio_l574_57480

theorem factorial_ratio : Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 5120 := by
  sorry

end NUMINAMATH_GPT_factorial_ratio_l574_57480


namespace NUMINAMATH_GPT_fraction_books_left_l574_57425

theorem fraction_books_left (initial_books sold_books remaining_books : ℕ)
  (h1 : initial_books = 9900) (h2 : sold_books = 3300) (h3 : remaining_books = initial_books - sold_books) :
  (remaining_books : ℚ) / initial_books = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_books_left_l574_57425


namespace NUMINAMATH_GPT_integer_solutions_l574_57483

theorem integer_solutions (m n : ℤ) (h1 : m * (m + n) = n * 12) (h2 : n * (m + n) = m * 3) :
  (m = 4 ∧ n = 2) :=
by sorry

end NUMINAMATH_GPT_integer_solutions_l574_57483


namespace NUMINAMATH_GPT_no_integer_n_gte_1_where_9_divides_7n_plus_n3_l574_57436

theorem no_integer_n_gte_1_where_9_divides_7n_plus_n3 :
  ∀ n : ℕ, 1 ≤ n → ¬ (7^n + n^3) % 9 = 0 := 
by
  intros n hn
  sorry

end NUMINAMATH_GPT_no_integer_n_gte_1_where_9_divides_7n_plus_n3_l574_57436


namespace NUMINAMATH_GPT_length_QR_l574_57443

-- Let's define the given conditions and the theorem to prove

-- Define the lengths of the sides of the triangle
def PQ : ℝ := 4
def PR : ℝ := 7
def PM : ℝ := 3.5

-- Define the median formula
def median_formula (PQ PR QR PM : ℝ) := PM = 0.5 * Real.sqrt (2 * PQ^2 + 2 * PR^2 - QR^2)

-- The theorem to prove: QR = 9
theorem length_QR 
  (hPQ : PQ = 4) 
  (hPR : PR = 7) 
  (hPM : PM = 3.5) 
  (hMedian : median_formula PQ PR QR PM) : 
  QR = 9 :=
sorry  -- proof will be here

end NUMINAMATH_GPT_length_QR_l574_57443


namespace NUMINAMATH_GPT_polynomial_divisibility_l574_57454

theorem polynomial_divisibility (P : Polynomial ℝ) (h_nonconstant : ∃ n : ℕ, P.degree = n ∧ n ≥ 1)
  (h_div : ∀ x : ℝ, P.eval (x^3 + 8) = 0 → P.eval (x^2 - 2*x + 4) = 0) :
  ∃ a : ℝ, ∃ n : ℕ, a ≠ 0 ∧ P = Polynomial.C a * Polynomial.X ^ n :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l574_57454


namespace NUMINAMATH_GPT_sin_cos_product_l574_57485

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_GPT_sin_cos_product_l574_57485


namespace NUMINAMATH_GPT_simplify_fraction_l574_57451

theorem simplify_fraction : (45 / (7 - 3 / 4)) = (36 / 5) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l574_57451


namespace NUMINAMATH_GPT_correct_word_is_tradition_l574_57424

-- Definitions of the words according to the problem conditions
def tradition : String := "custom, traditional practice"
def balance : String := "equilibrium"
def concern : String := "worry, care about"
def relationship : String := "relation"

-- The sentence to be filled
def sentence (word : String) : String :=
"There’s a " ++ word ++ " in our office that when it’s somebody’s birthday, they bring in a cake for us all to share."

-- The proof problem statement
theorem correct_word_is_tradition :
  ∀ word, (word ≠ tradition) → (sentence word ≠ "There’s a tradition in our office that when it’s somebody’s birthday, they bring in a cake for us all to share.") :=
by sorry

end NUMINAMATH_GPT_correct_word_is_tradition_l574_57424


namespace NUMINAMATH_GPT_find_f_of_1_over_3_l574_57431

theorem find_f_of_1_over_3
  (g : ℝ → ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, g x = 1 - x^2)
  (h2 : ∀ x, x ≠ 0 → f (g x) = (1 - x^2) / x^2) :
  f (1 / 3) = 1 / 2 := by
  sorry -- Proof goes here

end NUMINAMATH_GPT_find_f_of_1_over_3_l574_57431


namespace NUMINAMATH_GPT_product_modulo_l574_57469

theorem product_modulo : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m := 
  sorry

end NUMINAMATH_GPT_product_modulo_l574_57469


namespace NUMINAMATH_GPT_contrapositive_equivalence_l574_57410

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1)) ↔ (∀ x : ℝ, (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_contrapositive_equivalence_l574_57410


namespace NUMINAMATH_GPT_quadratic_non_negative_iff_a_in_range_l574_57407

theorem quadratic_non_negative_iff_a_in_range :
  (∀ x : ℝ, x^2 + (a - 2) * x + 1/4 ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_quadratic_non_negative_iff_a_in_range_l574_57407


namespace NUMINAMATH_GPT_initial_pencils_count_l574_57452

-- Define the conditions
def students : ℕ := 25
def pencils_per_student : ℕ := 5

-- Statement of the proof problem
theorem initial_pencils_count : students * pencils_per_student = 125 :=
by
  sorry

end NUMINAMATH_GPT_initial_pencils_count_l574_57452


namespace NUMINAMATH_GPT_triangle_to_initial_position_l574_57456

-- Definitions for triangle vertices
structure Point where
  x : Int
  y : Int

def p1 : Point := { x := 0, y := 0 }
def p2 : Point := { x := 6, y := 0 }
def p3 : Point := { x := 0, y := 4 }

-- Definitions for transformations
def rotate90 (p : Point) : Point := { x := -p.y, y := p.x }
def rotate180 (p : Point) : Point := { x := -p.x, y := -p.y }
def rotate270 (p : Point) : Point := { x := p.y, y := -p.x }
def reflect_y_eq_x (p : Point) : Point := { x := p.y, y := p.x }
def reflect_y_eq_neg_x (p : Point) : Point := { x := -p.y, y := -p.x }

-- Definitions for combination of transformations
-- This part defines how to combine transformations, e.g., as a sequence of three transformations.
def transform (fs : List (Point → Point)) (p : Point) : Point :=
  fs.foldl (fun acc f => f acc) p

-- The total number of valid sequences that return the triangle to its original position
def valid_sequences_count : Int := 6

-- Lean 4 statement
theorem triangle_to_initial_position : valid_sequences_count = 6 := by
  sorry

end NUMINAMATH_GPT_triangle_to_initial_position_l574_57456


namespace NUMINAMATH_GPT_tablecloth_covers_table_l574_57475

theorem tablecloth_covers_table
(length_ellipse : ℝ) (width_ellipse : ℝ) (length_tablecloth : ℝ) (width_tablecloth : ℝ)
(h1 : length_ellipse = 160)
(h2 : width_ellipse = 100)
(h3 : length_tablecloth = 140)
(h4 : width_tablecloth = 130) :
length_tablecloth >= width_ellipse ∧ width_tablecloth >= width_ellipse ∧
(length_tablecloth ^ 2 + width_tablecloth ^ 2) >= (length_ellipse ^ 2 + width_ellipse ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_tablecloth_covers_table_l574_57475


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l574_57471

theorem arithmetic_expression_evaluation : 
  ∃ (a b c d e f : Float),
  a - b * c / d + e = 0 ∧
  a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 ∧ e = 1 := sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l574_57471


namespace NUMINAMATH_GPT_weight_of_each_bag_is_7_l574_57444

-- Defining the conditions
def morning_bags : ℕ := 29
def afternoon_bags : ℕ := 17
def total_weight : ℕ := 322

-- Defining the question in terms of proving a specific weight per bag
def bags_sold := morning_bags + afternoon_bags
def weight_per_bag (w : ℕ) := total_weight = bags_sold * w

-- Proving the question == answer under the given conditions
theorem weight_of_each_bag_is_7 :
  ∃ w : ℕ, weight_per_bag w ∧ w = 7 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_each_bag_is_7_l574_57444


namespace NUMINAMATH_GPT_smallest_n_for_coloring_l574_57484

theorem smallest_n_for_coloring (n : ℕ) : n = 4 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_coloring_l574_57484


namespace NUMINAMATH_GPT_salary_increase_is_57point35_percent_l574_57486

variable (S : ℝ)

-- Assume Mr. Blue receives a 12% raise every year.
def annualRaise : ℝ := 1.12

-- After four years
theorem salary_increase_is_57point35_percent (h : annualRaise ^ 4 = 1.5735):
  ((annualRaise ^ 4 - 1) * S) / S = 0.5735 :=
by
  sorry

end NUMINAMATH_GPT_salary_increase_is_57point35_percent_l574_57486


namespace NUMINAMATH_GPT_parallelepiped_length_l574_57406

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_length_l574_57406


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l574_57461

theorem solve_equation_1 (x : ℝ) : (2 * x - 1) ^ 2 - 25 = 0 ↔ x = 3 ∨ x = -2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (1 / 3) * (x + 3) ^ 3 - 9 = 0 ↔ x = 0 := 
sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l574_57461


namespace NUMINAMATH_GPT_evaluate_expressions_l574_57455

theorem evaluate_expressions : (∀ (a b c d : ℤ), a = -(-3) → b = -(|-3|) → c = -(-(3^2)) → d = ((-3)^2) → b < 0) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expressions_l574_57455


namespace NUMINAMATH_GPT_simplify_expression_l574_57401

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l574_57401


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l574_57473

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) := 
sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l574_57473


namespace NUMINAMATH_GPT_sum_of_factors_eq_l574_57428

theorem sum_of_factors_eq :
  ∃ (d e f : ℤ), (∀ (x : ℤ), x^2 + 21 * x + 110 = (x + d) * (x + e)) ∧
                 (∀ (x : ℤ), x^2 - 19 * x + 88 = (x - e) * (x - f)) ∧
                 (d + e + f = 30) :=
sorry

end NUMINAMATH_GPT_sum_of_factors_eq_l574_57428


namespace NUMINAMATH_GPT_film_cost_eq_five_l574_57411

variable (F : ℕ)

theorem film_cost_eq_five (H1 : 9 * F + 4 * 4 + 6 * 3 = 79) : F = 5 :=
by
  -- This is a placeholder for your proof
  sorry

end NUMINAMATH_GPT_film_cost_eq_five_l574_57411


namespace NUMINAMATH_GPT_vegetarian_gluten_free_fraction_l574_57422

theorem vegetarian_gluten_free_fraction :
  ∀ (total_dishes meatless_dishes gluten_free_meatless_dishes : ℕ),
  meatless_dishes = 4 →
  meatless_dishes = total_dishes / 5 →
  gluten_free_meatless_dishes = meatless_dishes - 3 →
  gluten_free_meatless_dishes / total_dishes = 1 / 20 :=
by sorry

end NUMINAMATH_GPT_vegetarian_gluten_free_fraction_l574_57422


namespace NUMINAMATH_GPT_evaluate_expression_l574_57487

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem evaluate_expression : (nabla (nabla 2 3) 4) = 16777219 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l574_57487


namespace NUMINAMATH_GPT_find_ordered_pairs_of_b_c_l574_57457

theorem find_ordered_pairs_of_b_c : 
  ∃! (pairs : ℕ × ℕ), 
    (pairs.1 > 0 ∧ pairs.2 > 0) ∧ 
    (pairs.1 * pairs.1 = 4 * pairs.2) ∧ 
    (pairs.2 * pairs.2 = 4 * pairs.1) :=
sorry

end NUMINAMATH_GPT_find_ordered_pairs_of_b_c_l574_57457


namespace NUMINAMATH_GPT_graphene_scientific_notation_l574_57421

theorem graphene_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (0.00000000034 : ℝ) = a * 10^n ∧ a = 3.4 ∧ n = -10 :=
sorry

end NUMINAMATH_GPT_graphene_scientific_notation_l574_57421


namespace NUMINAMATH_GPT_ratio_of_c_and_d_l574_57450

theorem ratio_of_c_and_d (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 3 * x + 2 * y = c) 
  (h2 : 4 * y - 6 * x = d) : c / d = -1 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_c_and_d_l574_57450


namespace NUMINAMATH_GPT_initial_birds_was_one_l574_57476

def initial_birds (b : Nat) : Prop :=
  b + 4 = 5

theorem initial_birds_was_one : ∃ b, initial_birds b ∧ b = 1 :=
by
  use 1
  unfold initial_birds
  sorry

end NUMINAMATH_GPT_initial_birds_was_one_l574_57476


namespace NUMINAMATH_GPT_students_play_neither_l574_57418

-- Define the given conditions
def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_sports_players : ℕ := 17

-- The goal is to prove the number of students playing neither sport is 7
theorem students_play_neither :
  total_students - (football_players + long_tennis_players - both_sports_players) = 7 :=
by
  sorry

end NUMINAMATH_GPT_students_play_neither_l574_57418


namespace NUMINAMATH_GPT_quadratic_roots_real_and_equal_l574_57427

open Real

theorem quadratic_roots_real_and_equal :
  ∀ (x : ℝ), x^2 - 4 * x * sqrt 2 + 8 = 0 → ∃ r : ℝ, x = r :=
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_roots_real_and_equal_l574_57427


namespace NUMINAMATH_GPT_trader_profit_l574_57478

theorem trader_profit (P : ℝ) :
  let buy_price := 0.80 * P
  let sell_price := 1.20 * P
  sell_price - P = 0.20 * P := 
by
  sorry

end NUMINAMATH_GPT_trader_profit_l574_57478


namespace NUMINAMATH_GPT_ezekiel_painted_faces_l574_57493

noncomputable def cuboid_faces_painted (num_cuboids : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
num_cuboids * faces_per_cuboid

theorem ezekiel_painted_faces :
  cuboid_faces_painted 8 6 = 48 := 
by
  sorry

end NUMINAMATH_GPT_ezekiel_painted_faces_l574_57493


namespace NUMINAMATH_GPT_min_value_abs_plus_one_l574_57429

theorem min_value_abs_plus_one : ∃ x : ℝ, |x| + 1 = 1 :=
by
  use 0
  sorry

end NUMINAMATH_GPT_min_value_abs_plus_one_l574_57429


namespace NUMINAMATH_GPT_sphere_in_cube_volume_unreachable_l574_57403

noncomputable def volume_unreachable_space (cube_side : ℝ) (sphere_radius : ℝ) : ℝ :=
  let corner_volume := 64 - (32/3) * Real.pi
  let edge_volume := 288 - 72 * Real.pi
  corner_volume + edge_volume

theorem sphere_in_cube_volume_unreachable : 
  (volume_unreachable_space 6 1 = 352 - (248 * Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_GPT_sphere_in_cube_volume_unreachable_l574_57403


namespace NUMINAMATH_GPT_average_fixed_points_of_permutation_l574_57472

open Finset

noncomputable def average_fixed_points (n : ℕ) : ℕ :=
  1

theorem average_fixed_points_of_permutation (n : ℕ) :
  ∀ (σ : (Fin n) → (Fin n)), 
  (1: ℚ) = (1: ℕ) :=
by
  sorry

end NUMINAMATH_GPT_average_fixed_points_of_permutation_l574_57472


namespace NUMINAMATH_GPT_factorization_correct_l574_57458

theorem factorization_correct (x : ℤ) :
  (3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2) =
  ((3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6)) :=
by sorry

end NUMINAMATH_GPT_factorization_correct_l574_57458


namespace NUMINAMATH_GPT_convert_base_five_to_ten_l574_57413

theorem convert_base_five_to_ten : ∃ n : ℕ, n = 38 ∧ (1 * 5^2 + 2 * 5^1 + 3 * 5^0 = n) :=
by
  sorry

end NUMINAMATH_GPT_convert_base_five_to_ten_l574_57413


namespace NUMINAMATH_GPT_num_dimes_is_3_l574_57453

noncomputable def num_dimes (pennies nickels dimes quarters : ℕ) : ℕ :=
  dimes

theorem num_dimes_is_3 (h_total_coins : pennies + nickels + dimes + quarters = 11)
  (h_total_value : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 118)
  (h_at_least_one_each : 0 < pennies ∧ 0 < nickels ∧ 0 < dimes ∧ 0 < quarters) :
  num_dimes pennies nickels dimes quarters = 3 :=
sorry

end NUMINAMATH_GPT_num_dimes_is_3_l574_57453


namespace NUMINAMATH_GPT_range_of_a_l574_57400

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) * x^3

theorem range_of_a (a : ℝ) :
  f (Real.logb 2 a) + f (Real.logb 0.5 a) ≤ 2 * f 1 → (1/2 : ℝ) ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l574_57400


namespace NUMINAMATH_GPT_triangle_altitude_sum_l574_57430

-- Problem Conditions
def line_eq (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Altitudes Length Sum
theorem triangle_altitude_sum :
  ∀ x y : ℝ, line_eq x y → 
  ∀ (a b c: ℝ), a = 8 → b = 10 → c = 40 / Real.sqrt 41 →
  a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_GPT_triangle_altitude_sum_l574_57430


namespace NUMINAMATH_GPT_area_of_rhombus_l574_57488

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 4) (h2 : d2 = 4) :
    (d1 * d2) / 2 = 8 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l574_57488


namespace NUMINAMATH_GPT_tara_road_trip_cost_l574_57408

theorem tara_road_trip_cost :
  let tank_capacity := 12
  let price1 := 3
  let price2 := 3.50
  let price3 := 4
  let price4 := 4.50
  (price1 * tank_capacity) + (price2 * tank_capacity) + (price3 * tank_capacity) + (price4 * tank_capacity) = 180 :=
by
  sorry

end NUMINAMATH_GPT_tara_road_trip_cost_l574_57408


namespace NUMINAMATH_GPT_last_five_digits_l574_57416

theorem last_five_digits : (99 * 10101 * 111 * 1001) % 100000 = 88889 :=
by
  sorry

end NUMINAMATH_GPT_last_five_digits_l574_57416


namespace NUMINAMATH_GPT_correct_average_l574_57492

theorem correct_average (n : Nat) (incorrect_avg correct_mark incorrect_mark : ℝ) 
  (h1 : n = 30) (h2 : incorrect_avg = 60) (h3 : correct_mark = 15) (h4 : incorrect_mark = 90) :
  (incorrect_avg * n - incorrect_mark + correct_mark) / n = 57.5 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_l574_57492


namespace NUMINAMATH_GPT_simplify_expression_l574_57405

noncomputable def simplify_expr (a b : ℝ) : ℝ :=
  (3 * a^5 * b^3 + a^4 * b^2) / (-(a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b)

theorem simplify_expression (a b : ℝ) :
  simplify_expr a b = 8 * a * b - 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l574_57405


namespace NUMINAMATH_GPT_problem_solution_l574_57415

noncomputable def M (a b c : ℝ) : ℝ := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1) :
  M a b c ≤ -8 :=
sorry

end NUMINAMATH_GPT_problem_solution_l574_57415


namespace NUMINAMATH_GPT_total_ages_is_32_l574_57459

variable (a b c : ℕ)
variable (h_b : b = 12)
variable (h_a : a = b + 2)
variable (h_c : b = 2 * c)

theorem total_ages_is_32 (h_b : b = 12) (h_a : a = b + 2) (h_c : b = 2 * c) : a + b + c = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_ages_is_32_l574_57459


namespace NUMINAMATH_GPT_work_completion_l574_57412

theorem work_completion (A B : ℝ → ℝ) (h1 : ∀ t, A t = B t) (h3 : A 4 + B 4 = 1) : B 1 = 1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_work_completion_l574_57412


namespace NUMINAMATH_GPT_donna_fully_loaded_truck_weight_l574_57437

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end NUMINAMATH_GPT_donna_fully_loaded_truck_weight_l574_57437


namespace NUMINAMATH_GPT_percent_difference_l574_57468

theorem percent_difference :
  (0.90 * 40) - ((4 / 5) * 25) = 16 :=
by sorry

end NUMINAMATH_GPT_percent_difference_l574_57468


namespace NUMINAMATH_GPT_find_negative_integer_l574_57467

theorem find_negative_integer (M : ℤ) (h_neg : M < 0) (h_eq : M^2 + M = 12) : M = -4 :=
sorry

end NUMINAMATH_GPT_find_negative_integer_l574_57467


namespace NUMINAMATH_GPT_probability_divisor_of_8_is_half_l574_57402

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_divisor_of_8_is_half_l574_57402


namespace NUMINAMATH_GPT_remainder_divisible_by_4_l574_57482

theorem remainder_divisible_by_4 (z : ℕ) (h : z % 4 = 0) : ((z * (2 + 4 + z) + 3) % 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_divisible_by_4_l574_57482


namespace NUMINAMATH_GPT_evaluate_expression_l574_57409

theorem evaluate_expression : (1 / (5^2)^4) * 5^15 = 5^7 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l574_57409


namespace NUMINAMATH_GPT_minimum_area_of_rectangle_l574_57460

theorem minimum_area_of_rectangle (x y : ℝ) (h1 : x = 3) (h2 : y = 4) : 
  (min_area : ℝ) = (2.3 * 3.3) :=
by
  have length_min := x - 0.7
  have width_min := y - 0.7
  have min_area := length_min * width_min
  sorry

end NUMINAMATH_GPT_minimum_area_of_rectangle_l574_57460


namespace NUMINAMATH_GPT_project_completion_l574_57420

theorem project_completion (a b : ℕ) (h1 : 3 * (1 / b : ℚ) + (1 / a : ℚ) + (1 / b : ℚ) = 1) : 
  a + b = 9 ∨ a + b = 10 :=
sorry

end NUMINAMATH_GPT_project_completion_l574_57420


namespace NUMINAMATH_GPT_distance_between_closest_points_l574_57448

noncomputable def distance_closest_points (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  (Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - r1 - r2)

theorem distance_between_closest_points :
  distance_closest_points (4, 4) (20, 12) 4 12 = 4 * Real.sqrt 20 - 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_closest_points_l574_57448


namespace NUMINAMATH_GPT_least_number_to_add_l574_57462

theorem least_number_to_add (x : ℕ) (h : 1056 % 23 = 21) : (1056 + x) % 23 = 0 ↔ x = 2 :=
by {
    sorry
}

end NUMINAMATH_GPT_least_number_to_add_l574_57462


namespace NUMINAMATH_GPT_garden_contains_53_33_percent_tulips_l574_57490

theorem garden_contains_53_33_percent_tulips :
  (∃ (flowers : ℕ) (yellow tulips flowers_in_garden : ℕ) (yellow_flowers blue_flowers yellow_tulips blue_tulips : ℕ),
    flowers_in_garden = yellow_flowers + blue_flowers ∧
    yellow_flowers = 4 * flowers / 5 ∧
    blue_flowers = 1 * flowers / 5 ∧
    yellow_tulips = yellow_flowers / 2 ∧
    blue_tulips = 2 * blue_flowers / 3 ∧
    (yellow_tulips + blue_tulips) = 8 * flowers / 15) →
    0.5333 ∈ ([46.67, 53.33, 60, 75, 80] : List ℝ) := sorry

end NUMINAMATH_GPT_garden_contains_53_33_percent_tulips_l574_57490


namespace NUMINAMATH_GPT_product_not_divisible_by_201_l574_57417

theorem product_not_divisible_by_201 (a b : ℕ) (h₁ : a + b = 201) : ¬ (201 ∣ a * b) := sorry

end NUMINAMATH_GPT_product_not_divisible_by_201_l574_57417


namespace NUMINAMATH_GPT_intersection_eq_interval_l574_57489

def P : Set ℝ := {x | x * (x - 3) < 0}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_eq_interval : P ∩ Q = {x | 0 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_interval_l574_57489


namespace NUMINAMATH_GPT_cases_in_1990_l574_57495

theorem cases_in_1990 (cases_1970 cases_2000 : ℕ) (linear_decrease : ℕ → ℝ) :
  cases_1970 = 300000 →
  cases_2000 = 600 →
  (∀ t, linear_decrease t = cases_1970 - (cases_1970 - cases_2000) * t / 30) →
  linear_decrease 20 = 100400 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cases_in_1990_l574_57495


namespace NUMINAMATH_GPT_integer_part_sqrt_sum_l574_57465

theorem integer_part_sqrt_sum {a b c : ℤ} 
  (h_a : |a| = 4) 
  (h_b_sqrt : b^2 = 9) 
  (h_c_cubert : c^3 = -8) 
  (h_order : a > b ∧ b > c) 
  : (⌊ Real.sqrt (a + b + c) ⌋) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_integer_part_sqrt_sum_l574_57465


namespace NUMINAMATH_GPT_total_length_of_segments_in_new_figure_l574_57479

-- Defining the given conditions.
def left_side := 10
def top_side := 3
def right_side := 8
def segments_removed_from_bottom := [2, 1, 2] -- List of removed segments from the bottom.

-- This is the theorem statement that confirms the total length of the new figure's sides.
theorem total_length_of_segments_in_new_figure :
  (left_side + top_side + right_side) = 21 :=
by
  -- This is where the proof would be written.
  sorry

end NUMINAMATH_GPT_total_length_of_segments_in_new_figure_l574_57479


namespace NUMINAMATH_GPT_symmetric_line_equation_l574_57496

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l574_57496


namespace NUMINAMATH_GPT_angles_equal_or_cofunctions_equal_l574_57434

def cofunction (θ : ℝ) : ℝ := sorry -- Define the co-function (e.g., sine and cosine)

theorem angles_equal_or_cofunctions_equal (θ₁ θ₂ : ℝ) :
  θ₁ = θ₂ ∨ cofunction θ₁ = cofunction θ₂ → θ₁ = θ₂ :=
sorry

end NUMINAMATH_GPT_angles_equal_or_cofunctions_equal_l574_57434


namespace NUMINAMATH_GPT_no_such_f_exists_l574_57426

theorem no_such_f_exists (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
  (h2 : ∀ x y, 0 < x → 0 < y → f x ^ 2 ≥ f (x + y) * (f x + y)) : false :=
sorry

end NUMINAMATH_GPT_no_such_f_exists_l574_57426


namespace NUMINAMATH_GPT_point_on_parabola_distance_l574_57404

theorem point_on_parabola_distance (a b : ℝ) (h1 : a^2 = 20 * b) (h2 : |b + 5| = 25) : |a * b| = 400 :=
sorry

end NUMINAMATH_GPT_point_on_parabola_distance_l574_57404


namespace NUMINAMATH_GPT_range_of_a_l574_57463

theorem range_of_a (a : ℝ) (h1 : a > 0)
  (h2 : ∃ x : ℝ, abs (Real.sin x) > a)
  (h3 : ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ico (Real.sqrt 2 / 2) 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l574_57463


namespace NUMINAMATH_GPT_LCM_of_two_numbers_l574_57446

theorem LCM_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 11) (h_product : a * b = 1991) : Nat.lcm a b = 181 :=
by
  sorry

end NUMINAMATH_GPT_LCM_of_two_numbers_l574_57446


namespace NUMINAMATH_GPT_min_value_one_over_a_plus_nine_over_b_l574_57447

theorem min_value_one_over_a_plus_nine_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  16 ≤ (1 / a) + (9 / b) :=
sorry

end NUMINAMATH_GPT_min_value_one_over_a_plus_nine_over_b_l574_57447


namespace NUMINAMATH_GPT_johns_drawings_l574_57481

theorem johns_drawings (total_pictures : ℕ) (back_pictures : ℕ) 
  (h1 : total_pictures = 15) (h2 : back_pictures = 9) : total_pictures - back_pictures = 6 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_johns_drawings_l574_57481


namespace NUMINAMATH_GPT_total_tickets_l574_57440

theorem total_tickets (O B : ℕ) (h1 : 12 * O + 8 * B = 3320) (h2 : B = O + 90) : O + B = 350 := by
  sorry

end NUMINAMATH_GPT_total_tickets_l574_57440


namespace NUMINAMATH_GPT_eddie_games_l574_57474

-- Define the study block duration in minutes
def study_block_duration : ℕ := 60

-- Define the homework time in minutes
def homework_time : ℕ := 25

-- Define the time for one game in minutes
def game_time : ℕ := 5

-- Define the total time Eddie can spend playing games
noncomputable def time_for_games : ℕ := study_block_duration - homework_time

-- Define the number of games Eddie can play
noncomputable def number_of_games : ℕ := time_for_games / game_time

-- Theorem stating the number of games Eddie can play while completing his homework
theorem eddie_games : number_of_games = 7 := by
  sorry

end NUMINAMATH_GPT_eddie_games_l574_57474


namespace NUMINAMATH_GPT_seq_solution_l574_57414

theorem seq_solution {a b : ℝ} (h1 : a - b = 8) (h2 : a + b = 11) : 2 * a = 19 ∧ 2 * b = 3 := by
  sorry

end NUMINAMATH_GPT_seq_solution_l574_57414
