import Mathlib

namespace NUMINAMATH_GPT_xy_eq_one_l968_96869

theorem xy_eq_one (x y : ℝ) (h : x + y = (1 / x) + (1 / y) ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end NUMINAMATH_GPT_xy_eq_one_l968_96869


namespace NUMINAMATH_GPT_total_cookies_l968_96849

-- Define the number of bags and the number of cookies per bag
def bags : ℕ := 37
def cookies_per_bag : ℕ := 19

-- State the theorem
theorem total_cookies : bags * cookies_per_bag = 703 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_l968_96849


namespace NUMINAMATH_GPT_circle_area_l968_96826

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l968_96826


namespace NUMINAMATH_GPT_distribution_problem_distribution_problem_variable_distribution_problem_equal_l968_96894

def books_distribution_fixed (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then n.factorial / (a.factorial * b.factorial * c.factorial) else 0

theorem distribution_problem (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_fixed n a b c = 1260 :=
sorry

def books_distribution_variable (n : ℕ) (a b c : ℕ) : ℕ :=
  if h : a + b + c = n then (n.factorial / (a.factorial * b.factorial * c.factorial)) * 6 else 0

theorem distribution_problem_variable (n a b c : ℕ) (h : a + b + c = n) : 
  books_distribution_variable n a b c = 7560 :=
sorry

def books_distribution_equal (n : ℕ) (k : ℕ) : ℕ :=
  if h : 3 * k = n then n.factorial / (k.factorial * k.factorial * k.factorial) else 0

theorem distribution_problem_equal (n k : ℕ) (h : 3 * k = n) : 
  books_distribution_equal n k = 1680 :=
sorry

end NUMINAMATH_GPT_distribution_problem_distribution_problem_variable_distribution_problem_equal_l968_96894


namespace NUMINAMATH_GPT_groceries_spent_l968_96878

/-- Defining parameters from the conditions provided -/
def rent : ℝ := 5000
def milk : ℝ := 1500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 700
def savings_rate : ℝ := 0.10
def savings : ℝ := 1800

/-- Adding an assertion for the total spent on groceries -/
def groceries : ℝ := 4500

theorem groceries_spent (total_salary total_expenses : ℝ) :
  total_salary = savings / savings_rate →
  total_expenses = rent + milk + education + petrol + miscellaneous →
  groceries = total_salary - (total_expenses + savings) :=
by
  intros h_salary h_expenses
  sorry

end NUMINAMATH_GPT_groceries_spent_l968_96878


namespace NUMINAMATH_GPT_solve_for_y_l968_96856

theorem solve_for_y : ∀ (y : ℝ), (3 / 4 - 5 / 8 = 1 / y) → y = 8 :=
by
  intros y h
  sorry

end NUMINAMATH_GPT_solve_for_y_l968_96856


namespace NUMINAMATH_GPT_map_area_ratio_l968_96898

theorem map_area_ratio (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ¬ ((l * w) / ((500 * l) * (500 * w)) = 1 / 500) :=
by
  -- The proof will involve calculations showing the true ratio is 1/250000
  sorry

end NUMINAMATH_GPT_map_area_ratio_l968_96898


namespace NUMINAMATH_GPT_approx_log_base_5_10_l968_96829

noncomputable def log_base (b a : ℝ) : ℝ := (Real.log a) / (Real.log b)

theorem approx_log_base_5_10 :
  let lg2 := 0.301
  let lg3 := 0.477
  let lg10 := 1
  let lg5 := lg10 - lg2
  log_base 5 10 = 10 / 7 :=
  sorry

end NUMINAMATH_GPT_approx_log_base_5_10_l968_96829


namespace NUMINAMATH_GPT_inequality_x_n_l968_96806

theorem inequality_x_n (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : n ≥ 2) : (1 - x)^n + (1 + x)^n < 2^n := 
sorry

end NUMINAMATH_GPT_inequality_x_n_l968_96806


namespace NUMINAMATH_GPT_radius_ratio_of_circumscribed_truncated_cone_l968_96890

theorem radius_ratio_of_circumscribed_truncated_cone 
  (R r ρ : ℝ) 
  (h : ℝ) 
  (Vcs Vg : ℝ) 
  (h_eq : h = 2 * ρ)
  (Vcs_eq : Vcs = (π / 3) * h * (R^2 + r^2 + R * r))
  (Vg_eq : Vg = (4 * π * (ρ^3)) / 3)
  (Vcs_Vg_eq : Vcs = 2 * Vg) :
  (R / r) = (3 + Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_radius_ratio_of_circumscribed_truncated_cone_l968_96890


namespace NUMINAMATH_GPT_Sam_has_most_pages_l968_96807

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

end NUMINAMATH_GPT_Sam_has_most_pages_l968_96807


namespace NUMINAMATH_GPT_exists_convex_polygon_diagonals_l968_96864

theorem exists_convex_polygon_diagonals :
  ∃ n : ℕ, n * (n - 3) / 2 = 54 :=
by
  sorry

end NUMINAMATH_GPT_exists_convex_polygon_diagonals_l968_96864


namespace NUMINAMATH_GPT_three_f_x_expression_l968_96863

variable (f : ℝ → ℝ)
variable (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x))

theorem three_f_x_expression (x : ℝ) (hx : x > 0) : 3 * f x = 27 / (9 + 2 * x) :=
by sorry

end NUMINAMATH_GPT_three_f_x_expression_l968_96863


namespace NUMINAMATH_GPT_elixir_concentration_l968_96825

theorem elixir_concentration (x a : ℝ) 
  (h1 : (x * 100) / (100 + a) = 9) 
  (h2 : (x * 100 + a * 100) / (100 + 2 * a) = 23) : 
  x = 11 :=
by 
  sorry

end NUMINAMATH_GPT_elixir_concentration_l968_96825


namespace NUMINAMATH_GPT_cubs_more_home_runs_than_cardinals_l968_96824

theorem cubs_more_home_runs_than_cardinals 
(h1 : 2 + 1 + 2 = 5) 
(h2 : 1 + 1 = 2) : 
5 - 2 = 3 :=
by sorry

end NUMINAMATH_GPT_cubs_more_home_runs_than_cardinals_l968_96824


namespace NUMINAMATH_GPT_drug_price_reduction_l968_96848

theorem drug_price_reduction :
  ∃ x : ℝ, 56 * (1 - x)^2 = 31.5 :=
by
  sorry

end NUMINAMATH_GPT_drug_price_reduction_l968_96848


namespace NUMINAMATH_GPT_ending_number_l968_96846

theorem ending_number (h : ∃ n, 3 * n = 99 ∧ n = 33) : ∃ m, m = 99 :=
by
  sorry

end NUMINAMATH_GPT_ending_number_l968_96846


namespace NUMINAMATH_GPT_tangent_line_equation_l968_96801

theorem tangent_line_equation :
  ∃ (P : ℝ × ℝ) (m : ℝ), 
  P = (-2, 15) ∧ m = 2 ∧ 
  (∀ (x y : ℝ), (y = x^3 - 10 * x + 3) → (y - 15 = 2 * (x + 2))) :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l968_96801


namespace NUMINAMATH_GPT_each_person_pays_l968_96891

def numPeople : ℕ := 6
def rentalDays : ℕ := 4
def weekdayRate : ℕ := 420
def weekendRate : ℕ := 540
def numWeekdays : ℕ := 2
def numWeekends : ℕ := 2

theorem each_person_pays : 
  (numWeekdays * weekdayRate + numWeekends * weekendRate) / numPeople = 320 :=
by
  sorry

end NUMINAMATH_GPT_each_person_pays_l968_96891


namespace NUMINAMATH_GPT_total_widgets_sold_after_15_days_l968_96844

def widgets_sold_day_n (n : ℕ) : ℕ :=
  2 + (n - 1) * 3

def sum_of_widgets (n : ℕ) : ℕ :=
  n * (2 + widgets_sold_day_n n) / 2

theorem total_widgets_sold_after_15_days : 
  sum_of_widgets 15 = 345 :=
by
  -- Prove the arithmetic sequence properties and sum.
  sorry

end NUMINAMATH_GPT_total_widgets_sold_after_15_days_l968_96844


namespace NUMINAMATH_GPT_find_amplitude_l968_96870

theorem find_amplitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : ∀ x, a * Real.cos (b * x - c) ≤ 3) 
  (h5 : ∀ x, abs (a * Real.cos (b * x - c) - a * Real.cos (b * (x + 2 * π / b) - c)) = 0) :
  a = 3 := 
sorry

end NUMINAMATH_GPT_find_amplitude_l968_96870


namespace NUMINAMATH_GPT_smallest_possible_b_l968_96803

-- Definition of the polynomial Q(x)
def Q (x : ℤ) : ℤ := sorry -- Polynomial with integer coefficients

-- Initial conditions for b and Q
variable (b : ℤ) (hb : b > 0)
variable (hQ1 : Q 2 = b)
variable (hQ2 : Q 4 = b)
variable (hQ3 : Q 6 = b)
variable (hQ4 : Q 8 = b)
variable (hQ5 : Q 1 = -b)
variable (hQ6 : Q 3 = -b)
variable (hQ7 : Q 5 = -b)
variable (hQ8 : Q 7 = -b)

theorem smallest_possible_b : b = 315 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_b_l968_96803


namespace NUMINAMATH_GPT_code_transformation_l968_96835

def old_to_new_encoding (s : String) : String := sorry

theorem code_transformation :
  old_to_new_encoding "011011010011" = "211221121" := sorry

end NUMINAMATH_GPT_code_transformation_l968_96835


namespace NUMINAMATH_GPT_total_people_veg_l968_96837

-- Definitions based on the conditions
def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 6

-- The statement we need to prove
theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 19 :=
by
  sorry

end NUMINAMATH_GPT_total_people_veg_l968_96837


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l968_96886

variable {α : Type*} [Field α]

def geometric_sequence (a_1 q : α) (n : ℕ) : α :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_ratio (a1 q a4 a14 a5 a13 : α)
  (h_seq : ∀ n, geometric_sequence a1 q (n + 1) = a_5) 
  (h0 : geometric_sequence a1 q 5 * geometric_sequence a1 q 13 = 6) 
  (h1 : geometric_sequence a1 q 4 + geometric_sequence a1 q 14 = 5) :
  (∃ (k : α), k = 2 / 3 ∨ k = 3 / 2) → 
  geometric_sequence a1 q 80 / geometric_sequence a1 q 90 = k :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l968_96886


namespace NUMINAMATH_GPT_simplify_expression_l968_96866

def a : ℕ := 1050
def p : ℕ := 2101
def q : ℕ := 1050 * 1051

theorem simplify_expression : 
  (1051 / 1050) - (1050 / 1051) = (p : ℚ) / (q : ℚ) ∧ Nat.gcd p a = 1 ∧ Nat.gcd p (a + 1) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l968_96866


namespace NUMINAMATH_GPT_tommy_profit_l968_96821

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end NUMINAMATH_GPT_tommy_profit_l968_96821


namespace NUMINAMATH_GPT_age_of_B_l968_96854

theorem age_of_B (A B C : ℕ) 
  (h1 : (A + B + C) / 3 = 22)
  (h2 : (A + B) / 2 = 18)
  (h3 : (B + C) / 2 = 25) : 
  B = 20 := 
by
  sorry

end NUMINAMATH_GPT_age_of_B_l968_96854


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l968_96808

theorem rectangular_solid_surface_area
  (a b c : ℕ)
  (h_prime_a : Prime a)
  (h_prime_b : Prime b)
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 143) :
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l968_96808


namespace NUMINAMATH_GPT_sum_product_of_pairs_l968_96875

theorem sum_product_of_pairs (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x^2 + y^2 + z^2 = 200) :
  x * y + x * z + y * z = 100 := 
by
  sorry

end NUMINAMATH_GPT_sum_product_of_pairs_l968_96875


namespace NUMINAMATH_GPT_largest_and_smallest_value_of_expression_l968_96839

theorem largest_and_smallest_value_of_expression
  (w x y z : ℝ)
  (h1 : w + x + y + z = 0)
  (h2 : w^7 + x^7 + y^7 + z^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
sorry

end NUMINAMATH_GPT_largest_and_smallest_value_of_expression_l968_96839


namespace NUMINAMATH_GPT_sum_first_twelve_arithmetic_divisible_by_6_l968_96833

theorem sum_first_twelve_arithmetic_divisible_by_6 
  (a d : ℕ) (h1 : a > 0) (h2 : d > 0) : 
  6 ∣ (12 * a + 66 * d) := 
by
  sorry

end NUMINAMATH_GPT_sum_first_twelve_arithmetic_divisible_by_6_l968_96833


namespace NUMINAMATH_GPT_not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l968_96818

def vector_a : ℝ × ℝ := (3, 2)
def vector_vA : ℝ × ℝ := (3, -2)
def vector_vB : ℝ × ℝ := (2, 3)
def vector_vD : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem not_perpendicular_to_vA : dot_product vector_a vector_vA ≠ 0 := by sorry
theorem not_perpendicular_to_vB : dot_product vector_a vector_vB ≠ 0 := by sorry
theorem not_perpendicular_to_vD : dot_product vector_a vector_vD ≠ 0 := by sorry

end NUMINAMATH_GPT_not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l968_96818


namespace NUMINAMATH_GPT_sunset_time_l968_96811

theorem sunset_time (length_of_daylight : Nat := 11 * 60 + 18) -- length of daylight in minutes
    (sunrise : Nat := 6 * 60 + 32) -- sunrise time in minutes after midnight
    : (sunrise + length_of_daylight) % (24 * 60) = 17 * 60 + 50 := -- sunset time calculation
by
  sorry

end NUMINAMATH_GPT_sunset_time_l968_96811


namespace NUMINAMATH_GPT_chess_tournament_winner_l968_96858

theorem chess_tournament_winner :
  ∀ (x : ℕ) (P₉ P₁₀ : ℕ),
  (x > 0) →
  (9 * x) = 4 * P₃ →
  P₉ = (x * (x - 1)) / 2 + 9 * x^2 →
  P₁₀ = (9 * x * (9 * x - 1)) / 2 →
  (9 * x^2 - x) * 2 ≥ 81 * x^2 - 9 * x →
  x = 1 →
  P₃ = 9 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_winner_l968_96858


namespace NUMINAMATH_GPT_solution_set_of_inequality_l968_96867

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality :
  (∀ x, f (x) = f (-x)) →               -- f(x) is even
  (∀ x y, 0 < x → x < y → f y ≤ f x) →   -- f(x) is monotonically decreasing on (0, +∞)
  f 2 = 0 →                              -- f(2) = 0
  {x : ℝ | (f x + f (-x)) / (3 * x) < 0} = 
    {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l968_96867


namespace NUMINAMATH_GPT_cos_a2_plus_a8_eq_neg_half_l968_96832

noncomputable def a_n (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem cos_a2_plus_a8_eq_neg_half 
  (a₁ d : ℝ) 
  (h : a₁ + a_n 5 a₁ d + a_n 9 a₁ d = 5 * Real.pi)
  : Real.cos (a_n 2 a₁ d + a_n 8 a₁ d) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_a2_plus_a8_eq_neg_half_l968_96832


namespace NUMINAMATH_GPT_probability_no_adjacent_birch_l968_96873

theorem probability_no_adjacent_birch (m n : ℕ):
  let maple_trees := 5
  let oak_trees := 4
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  (∀ (prob : ℚ), prob = (2 : ℚ) / 45) → (m + n = 47) := by
  sorry

end NUMINAMATH_GPT_probability_no_adjacent_birch_l968_96873


namespace NUMINAMATH_GPT_determine_x_l968_96885

theorem determine_x (p q : ℝ) (hpq : p ≠ q) : 
  ∃ (c d : ℝ), (x = c*p + d*q) ∧ c = 2 ∧ d = -2 :=
by 
  sorry

end NUMINAMATH_GPT_determine_x_l968_96885


namespace NUMINAMATH_GPT_general_formula_and_arithmetic_sequence_l968_96834

noncomputable def S_n (n : ℕ) : ℕ := 3 * n ^ 2 - 2 * n
noncomputable def a_n (n : ℕ) : ℕ := S_n n - S_n (n - 1)

theorem general_formula_and_arithmetic_sequence :
  (∀ n : ℕ, a_n n = 6 * n - 5) ∧
  (∀ n : ℕ, (n ≥ 2 → a_n n - a_n (n - 1) = 6) ∧ (a_n 1 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_general_formula_and_arithmetic_sequence_l968_96834


namespace NUMINAMATH_GPT_determine_a_l968_96802

theorem determine_a (x : ℝ) (n : ℕ) (h : x > 0) (h_ineq : x + a / x^n ≥ n + 1) : a = n^n := by
  sorry

end NUMINAMATH_GPT_determine_a_l968_96802


namespace NUMINAMATH_GPT_hands_straight_line_time_l968_96828

noncomputable def time_when_hands_straight_line : List (ℕ × ℚ) :=
  let x₁ := 21 + 9 / 11
  let x₂ := 54 + 6 / 11
  [(4, x₁), (4, x₂)]

theorem hands_straight_line_time :
  time_when_hands_straight_line = [(4, 21 + 9 / 11), (4, 54 + 6 / 11)] :=
by
  sorry

end NUMINAMATH_GPT_hands_straight_line_time_l968_96828


namespace NUMINAMATH_GPT_log_expression_is_zero_l968_96860

noncomputable def log_expr : ℝ := (Real.logb 2 3 + Real.logb 2 27) * (Real.logb 4 4 + Real.logb 4 (1/4))

theorem log_expression_is_zero : log_expr = 0 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_is_zero_l968_96860


namespace NUMINAMATH_GPT_ratio_of_sister_to_Aaron_l968_96852

noncomputable def Aaron_age := 15
variable (H S : ℕ)
axiom Henry_age_relation : H = 4 * S
axiom combined_age : H + S + Aaron_age = 240

theorem ratio_of_sister_to_Aaron : (S : ℚ) / Aaron_age = 3 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ratio_of_sister_to_Aaron_l968_96852


namespace NUMINAMATH_GPT_area_enclosed_by_graph_l968_96805

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_graph_l968_96805


namespace NUMINAMATH_GPT_gift_combinations_l968_96823

theorem gift_combinations (wrapping_paper_count ribbon_count card_count : ℕ)
  (restricted_wrapping : ℕ)
  (restricted_ribbon : ℕ)
  (total_combinations := wrapping_paper_count * ribbon_count * card_count)
  (invalid_combinations := card_count)
  (valid_combinations := total_combinations - invalid_combinations) :
  wrapping_paper_count = 10 →
  ribbon_count = 4 →
  card_count = 5 →
  restricted_wrapping = 10 →
  restricted_ribbon = 1 →
  valid_combinations = 195 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gift_combinations_l968_96823


namespace NUMINAMATH_GPT_union_is_correct_l968_96884

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}
def union_set : Set ℤ := {-1, 0, 1, 2}

theorem union_is_correct : M ∪ N = union_set :=
  by sorry

end NUMINAMATH_GPT_union_is_correct_l968_96884


namespace NUMINAMATH_GPT_product_xyz_equals_one_l968_96880

theorem product_xyz_equals_one (x y z : ℝ) (h1 : x + (1/y) = 2) (h2 : y + (1/z) = 2) : x * y * z = 1 := 
by
  sorry

end NUMINAMATH_GPT_product_xyz_equals_one_l968_96880


namespace NUMINAMATH_GPT_analytic_expression_on_1_2_l968_96804

noncomputable def f : ℝ → ℝ :=
  sorry

theorem analytic_expression_on_1_2 (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  f x = Real.logb (1 / 2) (x - 1) :=
sorry

end NUMINAMATH_GPT_analytic_expression_on_1_2_l968_96804


namespace NUMINAMATH_GPT_number_of_pipes_used_l968_96874

-- Definitions
def T1 : ℝ := 15
def T2 : ℝ := T1 - 5
def T3 : ℝ := T2 - 4
def condition : Prop := 1 / T1 + 1 / T2 = 1 / T3

-- Proof Statement
theorem number_of_pipes_used : condition → 3 = 3 :=
by intros h; sorry

end NUMINAMATH_GPT_number_of_pipes_used_l968_96874


namespace NUMINAMATH_GPT_cost_of_filling_all_pots_l968_96871

def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny_per_plant : ℝ := 4.00
def num_creeping_jennies : ℝ := 4
def cost_geranium_per_plant : ℝ := 3.50
def num_geraniums : ℝ := 4
def cost_elephant_ear_per_plant : ℝ := 7.00
def num_elephant_ears : ℝ := 2
def cost_purple_fountain_grass_per_plant : ℝ := 6.00
def num_purple_fountain_grasses : ℝ := 3
def num_pots : ℝ := 4

def total_cost_per_pot : ℝ := 
  cost_palm_fern +
  (num_creeping_jennies * cost_creeping_jenny_per_plant) +
  (num_geraniums * cost_geranium_per_plant) +
  (num_elephant_ears * cost_elephant_ear_per_plant) +
  (num_purple_fountain_grasses * cost_purple_fountain_grass_per_plant)

def total_cost : ℝ := total_cost_per_pot * num_pots

theorem cost_of_filling_all_pots : total_cost = 308.00 := by
  sorry

end NUMINAMATH_GPT_cost_of_filling_all_pots_l968_96871


namespace NUMINAMATH_GPT_product_representation_count_l968_96841

theorem product_representation_count :
  let n := 1000000
  let distinct_ways := 139
  (∃ (a b c d e f : ℕ), 2^(a+b+c) * 5^(d+e+f) = n ∧ 
    a + b + c = 6 ∧ d + e + f = 6 ) → 
    139 = distinct_ways := 
by {
  sorry
}

end NUMINAMATH_GPT_product_representation_count_l968_96841


namespace NUMINAMATH_GPT_lines_intersect_l968_96847

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
(1 + 2 * t, 2 - 3 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
(-1 + 3 * u, 4 + u)

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = (-5 / 11, 46 / 11) ∧ line2 u = (-5 / 11, 46 / 11) :=
sorry

end NUMINAMATH_GPT_lines_intersect_l968_96847


namespace NUMINAMATH_GPT_abs_frac_sqrt_l968_96892

theorem abs_frac_sqrt (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 + b^2 = 9 * a * b) : 
  abs ((a + b) / (a - b)) = Real.sqrt (11 / 7) :=
by
  sorry

end NUMINAMATH_GPT_abs_frac_sqrt_l968_96892


namespace NUMINAMATH_GPT_circle_through_three_points_l968_96896

open Real

structure Point where
  x : ℝ
  y : ℝ

def circle_equation (D E F : ℝ) (P : Point) : Prop :=
  P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0

theorem circle_through_three_points :
  ∃ (D E F : ℝ), 
    (circle_equation D E F ⟨1, 12⟩) ∧ 
    (circle_equation D E F ⟨7, 10⟩) ∧ 
    (circle_equation D E F ⟨-9, 2⟩) ∧
    (D = -2) ∧ (E = -4) ∧ (F = -95) :=
by
  sorry

end NUMINAMATH_GPT_circle_through_three_points_l968_96896


namespace NUMINAMATH_GPT_mean_points_scored_l968_96813

def Mrs_Williams_points : ℝ := 50
def Mr_Adams_points : ℝ := 57
def Mrs_Browns_points : ℝ := 49
def Mrs_Daniels_points : ℝ := 57

def total_points : ℝ := Mrs_Williams_points + Mr_Adams_points + Mrs_Browns_points + Mrs_Daniels_points
def number_of_classes : ℝ := 4

theorem mean_points_scored :
  (total_points / number_of_classes) = 53.25 :=
by
  sorry

end NUMINAMATH_GPT_mean_points_scored_l968_96813


namespace NUMINAMATH_GPT_series_sum_equals_9_over_4_l968_96816

noncomputable def series_sum : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

theorem series_sum_equals_9_over_4 :
  series_sum = 9 / 4 :=
sorry

end NUMINAMATH_GPT_series_sum_equals_9_over_4_l968_96816


namespace NUMINAMATH_GPT_find_m_value_l968_96882

theorem find_m_value :
  62519 * 9999 = 625127481 :=
  by sorry

end NUMINAMATH_GPT_find_m_value_l968_96882


namespace NUMINAMATH_GPT_scientific_notation_of_3300000000_l968_96881

theorem scientific_notation_of_3300000000 :
  3300000000 = 3.3 * 10^9 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_3300000000_l968_96881


namespace NUMINAMATH_GPT_sum_is_composite_l968_96827

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (h : a^2 - a * b + b^2 = c^2 - c * d + d^2) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ k * l = a + b + c + d :=
by sorry

end NUMINAMATH_GPT_sum_is_composite_l968_96827


namespace NUMINAMATH_GPT_range_of_a_over_b_l968_96888

variable (a b : ℝ)

theorem range_of_a_over_b (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  -2 < a / b ∧ a / b < -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_over_b_l968_96888


namespace NUMINAMATH_GPT_total_elephants_in_two_parks_is_280_l968_96862

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end NUMINAMATH_GPT_total_elephants_in_two_parks_is_280_l968_96862


namespace NUMINAMATH_GPT_symmetry_axis_of_transformed_function_l968_96810

theorem symmetry_axis_of_transformed_function :
  let initial_func (x : ℝ) := Real.sin (4 * x - π / 6)
  let stretched_func (x : ℝ) := Real.sin (8 * x - π / 3)
  let transformed_func (x : ℝ) := Real.sin (8 * (x + π / 4) - π / 3)
  let ω := 8
  let φ := 5 * π / 3
  x = π / 12 :=
  sorry

end NUMINAMATH_GPT_symmetry_axis_of_transformed_function_l968_96810


namespace NUMINAMATH_GPT_grain_output_l968_96851

-- Define the condition regarding grain output.
def premier_goal (x : ℝ) : Prop :=
  x > 1.3

-- The mathematical statement that needs to be proved, given the condition.
theorem grain_output (x : ℝ) (h : premier_goal x) : x > 1.3 :=
by
  sorry

end NUMINAMATH_GPT_grain_output_l968_96851


namespace NUMINAMATH_GPT_interval_of_increase_inequality_for_large_x_l968_96814

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + log x

theorem interval_of_increase :
  ∀ x > 0, ∀ y > x, f y > f x :=
by
  sorry

theorem inequality_for_large_x (x : ℝ) (hx : x > 1) :
  (1/2) * x^2 + log x < (2/3) * x^3 :=
by
  sorry

end NUMINAMATH_GPT_interval_of_increase_inequality_for_large_x_l968_96814


namespace NUMINAMATH_GPT_sufficient_not_necessary_l968_96809

variable (x : ℝ)
def p := x^2 > 4
def q := x > 2

theorem sufficient_not_necessary : (∀ x, q x -> p x) ∧ ¬ (∀ x, p x -> q x) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l968_96809


namespace NUMINAMATH_GPT_pyramid_volume_l968_96817

theorem pyramid_volume
  (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5)
  (angle_lateral : ℝ) (h4 : angle_lateral = 45) :
  ∃ (V : ℝ), V = 6 :=
by
  -- the proof steps would be included here
  sorry

end NUMINAMATH_GPT_pyramid_volume_l968_96817


namespace NUMINAMATH_GPT_xy_computation_l968_96877

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end NUMINAMATH_GPT_xy_computation_l968_96877


namespace NUMINAMATH_GPT_rachel_baked_brownies_l968_96879

theorem rachel_baked_brownies (b : ℕ) (h : 3 * b / 5 = 18) : b = 30 :=
by
  sorry

end NUMINAMATH_GPT_rachel_baked_brownies_l968_96879


namespace NUMINAMATH_GPT_initial_red_balls_l968_96861

-- Define all the conditions as given in part (a)
variables (R : ℕ)  -- Initial number of red balls
variables (B : ℕ)  -- Number of blue balls
variables (Y : ℕ)  -- Number of yellow balls

-- The conditions
def conditions (R B Y total : ℕ) : Prop :=
  B = 2 * R ∧
  Y = 32 ∧
  total = (R - 6) + B + Y

-- The target statement proving R = 16 given the conditions
theorem initial_red_balls (R: ℕ) (B: ℕ) (Y: ℕ) (total: ℕ) 
  (h : conditions R B Y total): 
  total = 74 → R = 16 :=
by 
  sorry

end NUMINAMATH_GPT_initial_red_balls_l968_96861


namespace NUMINAMATH_GPT_beta_max_success_ratio_l968_96843

theorem beta_max_success_ratio :
  ∃ (a b c d : ℕ),
    0 < a ∧ a < b ∧
    0 < c ∧ c < d ∧
    b + d ≤ 550 ∧
    (15 * a < 8 * b) ∧ (10 * c < 7 * d) ∧
    (21 * a + 16 * c < 4400) ∧
    ((a + c) / (b + d : ℚ) = 274 / 550) :=
sorry

end NUMINAMATH_GPT_beta_max_success_ratio_l968_96843


namespace NUMINAMATH_GPT_remainder_3042_div_29_l968_96836

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end NUMINAMATH_GPT_remainder_3042_div_29_l968_96836


namespace NUMINAMATH_GPT_polygon_side_count_l968_96815

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_side_count_l968_96815


namespace NUMINAMATH_GPT_sector_area_l968_96830

theorem sector_area (theta r : ℝ) (h1 : theta = 2 * Real.pi / 3) (h2 : r = 2) :
  (1 / 2 * r ^ 2 * theta) = 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_sector_area_l968_96830


namespace NUMINAMATH_GPT_decreasing_interval_of_logarithm_derived_function_l968_96868

theorem decreasing_interval_of_logarithm_derived_function :
  ∀ (x : ℝ), 1 < x → ∃ (f : ℝ → ℝ), (f x = x / (x - 1)) ∧ (∀ (h : x ≠ 1), deriv f x < 0) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_of_logarithm_derived_function_l968_96868


namespace NUMINAMATH_GPT_pipe_q_fills_cistern_in_15_minutes_l968_96819

theorem pipe_q_fills_cistern_in_15_minutes :
  ∃ T : ℝ, 
    (1/12 * 2 + 1/T * 2 + 1/T * 10.5 = 1) → 
    T = 15 :=
by {
  -- Assume the conditions and derive T = 15
  sorry
}

end NUMINAMATH_GPT_pipe_q_fills_cistern_in_15_minutes_l968_96819


namespace NUMINAMATH_GPT_correct_option_l968_96865

variable (a : ℝ)

theorem correct_option (h1 : 5 * a^2 - 4 * a^2 = a^2)
                       (h2 : a^7 / a^4 = a^3)
                       (h3 : (a^3)^2 = a^6)
                       (h4 : a^2 * a^3 = a^5) : 
                       a^7 / a^4 = a^3 := 
by
  exact h2

end NUMINAMATH_GPT_correct_option_l968_96865


namespace NUMINAMATH_GPT_lines_intersect_at_point_l968_96838

/-
Given two lines parameterized as:
Line 1: (x, y) = (2, 0) + s * (3, -4)
Line 2: (x, y) = (6, -10) + v * (5, 3)
Prove that these lines intersect at (242/29, -248/29).
-/

def parametric_line_1 (s : ℚ) : ℚ × ℚ :=
  (2 + 3 * s, -4 * s)

def parametric_line_2 (v : ℚ) : ℚ × ℚ :=
  (6 + 5 * v, -10 + 3 * v)

theorem lines_intersect_at_point :
  ∃ (s v : ℚ), parametric_line_1 s = parametric_line_2 v ∧ parametric_line_1 s = (242 / 29, -248 / 29) :=
sorry

end NUMINAMATH_GPT_lines_intersect_at_point_l968_96838


namespace NUMINAMATH_GPT_abs_sum_lt_abs_l968_96883

theorem abs_sum_lt_abs (a b : ℝ) (h : a * b < 0) : |a + b| < |a| + |b| :=
sorry

end NUMINAMATH_GPT_abs_sum_lt_abs_l968_96883


namespace NUMINAMATH_GPT_range_of_a_if_odd_symmetric_points_l968_96897

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a

theorem range_of_a_if_odd_symmetric_points (a : ℝ): 
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ f x₀ a = -f (-x₀) a) → (1 < a) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_if_odd_symmetric_points_l968_96897


namespace NUMINAMATH_GPT_maxwell_walking_speed_l968_96845

theorem maxwell_walking_speed :
  ∀ (distance_between_homes : ℕ)
    (brad_speed : ℕ)
    (middle_travel_maxwell : ℕ)
    (middle_distance : ℕ),
    distance_between_homes = 36 →
    brad_speed = 4 →
    middle_travel_maxwell = 12 →
    middle_distance = 18 →
    (middle_travel_maxwell : ℕ) / (8 : ℕ) = (middle_distance - middle_travel_maxwell) / brad_speed :=
  sorry

end NUMINAMATH_GPT_maxwell_walking_speed_l968_96845


namespace NUMINAMATH_GPT_roots_of_x2_eq_x_l968_96840

theorem roots_of_x2_eq_x : ∀ x : ℝ, x^2 = x ↔ (x = 0 ∨ x = 1) := 
by
  sorry

end NUMINAMATH_GPT_roots_of_x2_eq_x_l968_96840


namespace NUMINAMATH_GPT_Connie_savings_l968_96889

theorem Connie_savings (cost_of_watch : ℕ) (extra_needed : ℕ) (saved_amount : ℕ) : 
  cost_of_watch = 55 → 
  extra_needed = 16 → 
  saved_amount = cost_of_watch - extra_needed → 
  saved_amount = 39 := 
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end NUMINAMATH_GPT_Connie_savings_l968_96889


namespace NUMINAMATH_GPT_trajectory_of_A_l968_96820

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end NUMINAMATH_GPT_trajectory_of_A_l968_96820


namespace NUMINAMATH_GPT_value_of_x_l968_96842

theorem value_of_x (x y : ℕ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
sorry

end NUMINAMATH_GPT_value_of_x_l968_96842


namespace NUMINAMATH_GPT_find_natural_number_l968_96812

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_natural_number (n : ℕ) : sum_of_digits (2 ^ n) = 5 ↔ n = 5 := by
  sorry

end NUMINAMATH_GPT_find_natural_number_l968_96812


namespace NUMINAMATH_GPT_forgot_to_take_capsules_l968_96893

theorem forgot_to_take_capsules (total_days : ℕ) (days_taken : ℕ) 
  (h1 : total_days = 31) 
  (h2 : days_taken = 29) : 
  total_days - days_taken = 2 := 
by 
  sorry

end NUMINAMATH_GPT_forgot_to_take_capsules_l968_96893


namespace NUMINAMATH_GPT_power_relationship_l968_96822

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end NUMINAMATH_GPT_power_relationship_l968_96822


namespace NUMINAMATH_GPT_total_bill_l968_96850

theorem total_bill (total_friends : ℕ) (extra_payment : ℝ) (total_bill : ℝ) (paid_by_friends : ℝ) :
  total_friends = 8 → extra_payment = 2.50 →
  (7 * ((total_bill / total_friends) + extra_payment)) = total_bill →
  total_bill = 140 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_bill_l968_96850


namespace NUMINAMATH_GPT_simplify_fraction_l968_96859

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l968_96859


namespace NUMINAMATH_GPT_fruit_salad_cherries_l968_96887

theorem fruit_salad_cherries (b r g c : ℕ) 
(h1 : b + r + g + c = 360)
(h2 : r = 3 * b) 
(h3 : g = 4 * c)
(h4 : c = 5 * r) :
c = 68 := 
sorry

end NUMINAMATH_GPT_fruit_salad_cherries_l968_96887


namespace NUMINAMATH_GPT_sum_of_coefficients_proof_l968_96899

-- Problem statement: Define the expressions and prove the sum of the coefficients
def expr1 (c : ℝ) : ℝ := -(3 - c) * (c + 2 * (3 - c))
def expanded_form (c : ℝ) : ℝ := -c^2 + 9 * c - 18
def sum_of_coefficients (p : ℝ) := -1 + 9 - 18

theorem sum_of_coefficients_proof (c : ℝ) : sum_of_coefficients (expr1 c) = -10 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_proof_l968_96899


namespace NUMINAMATH_GPT_percent_of_ducks_among_non_swans_l968_96876

theorem percent_of_ducks_among_non_swans
  (total_birds : ℕ) 
  (percent_ducks percent_swans percent_eagles percent_sparrows : ℕ)
  (h1 : percent_ducks = 40) 
  (h2 : percent_swans = 20) 
  (h3 : percent_eagles = 15) 
  (h4 : percent_sparrows = 25)
  (h_sum : percent_ducks + percent_swans + percent_eagles + percent_sparrows = 100) :
  (percent_ducks * 100) / (100 - percent_swans) = 50 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_ducks_among_non_swans_l968_96876


namespace NUMINAMATH_GPT_line_circle_separate_l968_96853

def point_inside_circle (x0 y0 a : ℝ) : Prop :=
  x0^2 + y0^2 < a^2

def not_center_of_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 ≠ 0

theorem line_circle_separate (x0 y0 a : ℝ) (h1 : point_inside_circle x0 y0 a) (h2 : a > 0) (h3 : not_center_of_circle x0 y0) :
  ∀ (x y : ℝ), ¬ (x0 * x + y0 * y = a^2 ∧ x^2 + y^2 = a^2) :=
by
  sorry

end NUMINAMATH_GPT_line_circle_separate_l968_96853


namespace NUMINAMATH_GPT_additional_spending_required_l968_96800

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end NUMINAMATH_GPT_additional_spending_required_l968_96800


namespace NUMINAMATH_GPT_nancy_marks_home_economics_l968_96872

-- Definitions from conditions
def marks_american_lit := 66
def marks_history := 75
def marks_physical_ed := 68
def marks_art := 89
def average_marks := 70
def num_subjects := 5
def total_marks := average_marks * num_subjects
def marks_other_subjects := marks_american_lit + marks_history + marks_physical_ed + marks_art

-- Statement to prove
theorem nancy_marks_home_economics : 
  (total_marks - marks_other_subjects = 52) := by 
  sorry

end NUMINAMATH_GPT_nancy_marks_home_economics_l968_96872


namespace NUMINAMATH_GPT_fliers_left_l968_96857

theorem fliers_left (initial_fliers : ℕ) (fraction_morning : ℕ) (fraction_afternoon : ℕ) :
  initial_fliers = 2000 → 
  fraction_morning = 1 / 10 → 
  fraction_afternoon = 1 / 4 → 
  (initial_fliers - initial_fliers * fraction_morning - 
  (initial_fliers - initial_fliers * fraction_morning) * fraction_afternoon) = 1350 := by
  intros initial_fliers_eq fraction_morning_eq fraction_afternoon_eq
  sorry

end NUMINAMATH_GPT_fliers_left_l968_96857


namespace NUMINAMATH_GPT_solution_verification_l968_96831

-- Define the differential equation
def diff_eq (y y' y'': ℝ → ℝ) : Prop :=
  ∀ x, y'' x - 4 * y' x + 5 * y x = 2 * Real.cos x + 6 * Real.sin x

-- General solution form
def general_solution (C₁ C₂ : ℝ) (y: ℝ → ℝ) : Prop :=
  ∀ x, y x = Real.exp (2 * x) * (C₁ * Real.cos x + C₂ * Real.sin x) + Real.cos x + 1/2 * Real.sin x

-- Proof problem statement
theorem solution_verification (C₁ C₂ : ℝ) (y y' y'': ℝ → ℝ) :
  (∀ x, y' x = deriv y x) →
  (∀ x, y'' x = deriv (deriv y) x) →
  diff_eq y y' y'' →
  general_solution C₁ C₂ y :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_solution_verification_l968_96831


namespace NUMINAMATH_GPT_equidistant_divisors_multiple_of_6_l968_96855

open Nat

theorem equidistant_divisors_multiple_of_6 (n : ℕ) :
  (∃ a b : ℕ, a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
    (a + b = 2 * (n / 3))) → 
  (∃ k : ℕ, n = 6 * k) := 
by
  sorry

end NUMINAMATH_GPT_equidistant_divisors_multiple_of_6_l968_96855


namespace NUMINAMATH_GPT_remainders_sum_l968_96895

theorem remainders_sum (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 20) 
  (h3 : c % 30 = 10) : 
  (a + b + c) % 30 = 15 := 
by
  sorry

end NUMINAMATH_GPT_remainders_sum_l968_96895
