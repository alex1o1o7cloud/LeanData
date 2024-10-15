import Mathlib

namespace NUMINAMATH_GPT_tan_10pi_minus_theta_l1049_104944

open Real

theorem tan_10pi_minus_theta (θ : ℝ) (h1 : π < θ) (h2 : θ < 2 * π) (h3 : cos (θ - 9 * π) = -3 / 5) : 
  tan (10 * π - θ) = -4 / 3 := 
sorry

end NUMINAMATH_GPT_tan_10pi_minus_theta_l1049_104944


namespace NUMINAMATH_GPT_range_of_a_l1049_104943

theorem range_of_a :
  (∀ x : ℝ, abs (x - a) < 1 ↔ (1 / 2 < x ∧ x < 3 / 2)) → (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1049_104943


namespace NUMINAMATH_GPT_ratio_length_to_breadth_l1049_104921

-- Definitions of the given conditions
def length_landscape : ℕ := 120
def area_playground : ℕ := 1200
def ratio_playground_to_landscape : ℕ := 3

-- Property that the area of the playground is 1/3 of the area of the landscape
def total_area_landscape (area_playground : ℕ) (ratio_playground_to_landscape : ℕ) : ℕ :=
  area_playground * ratio_playground_to_landscape

-- Calculation that breadth of the landscape
def breadth_landscape (length_landscape total_area_landscape : ℕ) : ℕ :=
  total_area_landscape / length_landscape

-- The proof statement for the ratio of length to breadth
theorem ratio_length_to_breadth (length_landscape area_playground : ℕ) (ratio_playground_to_landscape : ℕ)
  (h1 : length_landscape = 120)
  (h2 : area_playground = 1200)
  (h3 : ratio_playground_to_landscape = 3)
  (h4 : total_area_landscape area_playground ratio_playground_to_landscape = 3600)
  (h5 : breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 30) :
  length_landscape / breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 4 :=
by
  sorry


end NUMINAMATH_GPT_ratio_length_to_breadth_l1049_104921


namespace NUMINAMATH_GPT_final_tree_count_l1049_104997

def current_trees : ℕ := 7
def monday_trees : ℕ := 3
def tuesday_trees : ℕ := 2
def wednesday_trees : ℕ := 5
def thursday_trees : ℕ := 1
def friday_trees : ℕ := 6
def saturday_trees : ℕ := 4
def sunday_trees : ℕ := 3

def total_trees_planted : ℕ := monday_trees + tuesday_trees + wednesday_trees + thursday_trees + friday_trees + saturday_trees + sunday_trees

theorem final_tree_count :
  current_trees + total_trees_planted = 31 :=
by
  sorry

end NUMINAMATH_GPT_final_tree_count_l1049_104997


namespace NUMINAMATH_GPT_polynomial_factor_implies_a_minus_b_l1049_104929

theorem polynomial_factor_implies_a_minus_b (a b : ℝ) :
  (∀ x y : ℝ, (x + y - 2) ∣ (x^2 + a * x * y + b * y^2 - 5 * x + y + 6))
  → a - b = 1 :=
by
  intro h
  -- Proof needs to be filled in
  sorry

end NUMINAMATH_GPT_polynomial_factor_implies_a_minus_b_l1049_104929


namespace NUMINAMATH_GPT_circle_diameter_l1049_104974

theorem circle_diameter (r : ℝ) (h : r = 4) : 2 * r = 8 := sorry

end NUMINAMATH_GPT_circle_diameter_l1049_104974


namespace NUMINAMATH_GPT_range_of_m_l1049_104913

open Real

def vector_a (m : ℝ) : ℝ × ℝ := (m, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (-2 * m, m)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def not_parallel (m : ℝ) : Prop :=
  m^2 + 2 * m ≠ 0

theorem range_of_m (m : ℝ) (h1 : dot_product (vector_a m) (vector_b m) < 0) (h2 : not_parallel m) :
  m < 0 ∨ (m > (1 / 2) ∧ m ≠ -2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1049_104913


namespace NUMINAMATH_GPT_smallest_base10_integer_l1049_104940

theorem smallest_base10_integer (X Y : ℕ) (hX : X < 6) (hY : Y < 8) (h : 7 * X = 9 * Y) :
  63 = 7 * X ∧ 63 = 9 * Y :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_smallest_base10_integer_l1049_104940


namespace NUMINAMATH_GPT_find_X_l1049_104966

def operation (X Y : Int) : Int := X + 2 * Y 

lemma property_1 (X : Int) : operation X 0 = X := 
by simp [operation]

lemma property_2 (X Y : Int) : operation X (Y - 1) = (operation X Y) - 2 := 
by simp [operation]; linarith

lemma property_3 (X Y : Int) : operation X (Y + 1) = (operation X Y) + 2 := 
by simp [operation]; linarith

theorem find_X (X : Int) : operation X X = -2019 ↔ X = -673 :=
by sorry

end NUMINAMATH_GPT_find_X_l1049_104966


namespace NUMINAMATH_GPT_division_multiplication_l1049_104979

-- Given a number x, we want to prove that (x / 6) * 12 = 2 * x under basic arithmetic operations.

theorem division_multiplication (x : ℝ) : (x / 6) * 12 = 2 * x := 
by
  sorry

end NUMINAMATH_GPT_division_multiplication_l1049_104979


namespace NUMINAMATH_GPT_Maria_soap_cost_l1049_104938
-- Import the entire Mathlib library
  
theorem Maria_soap_cost (soap_last_months : ℕ) (cost_per_bar : ℝ) (months_in_year : ℕ):
  (soap_last_months = 2) -> 
  (cost_per_bar = 8.00) ->
  (months_in_year = 12) -> 
  (months_in_year / soap_last_months * cost_per_bar = 48.00) := 
by
  intros h_soap_last h_cost h_year
  sorry

end NUMINAMATH_GPT_Maria_soap_cost_l1049_104938


namespace NUMINAMATH_GPT_average_GPA_school_l1049_104995

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end NUMINAMATH_GPT_average_GPA_school_l1049_104995


namespace NUMINAMATH_GPT_repeating_decimal_base4_sum_l1049_104945

theorem repeating_decimal_base4_sum (a b : ℕ) (hrelprime : Int.gcd a b = 1)
  (h4_rep : ((12 : ℚ) / (44 : ℚ)) = (a : ℚ) / (b : ℚ)) : a + b = 7 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_base4_sum_l1049_104945


namespace NUMINAMATH_GPT_andy_wrong_questions_l1049_104939

/-- Andy, Beth, Charlie, and Daniel take a test. Andy and Beth together get the same number of 
    questions wrong as Charlie and Daniel together. Andy and Daniel together get four more 
    questions wrong than Beth and Charlie do together. Charlie gets five questions wrong. 
    Prove that Andy gets seven questions wrong. -/
theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 4) (h3 : c = 5) :
  a = 7 :=
by
  sorry

end NUMINAMATH_GPT_andy_wrong_questions_l1049_104939


namespace NUMINAMATH_GPT_roots_quadratic_expr_l1049_104978

theorem roots_quadratic_expr (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0)
    (h2 : Polynomial.eval n (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0) :
  m^2 + m * n + 2 * m = 0 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_expr_l1049_104978


namespace NUMINAMATH_GPT_complex_sum_zero_l1049_104955

noncomputable def complexSum {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^(15) + ω^(18) + ω^(21) + ω^(24) + ω^(27) + ω^(30) +
  ω^(33) + ω^(36) + ω^(39) + ω^(42) + ω^(45)

theorem complex_sum_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : complexSum h1 h2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_sum_zero_l1049_104955


namespace NUMINAMATH_GPT_f_at_1_l1049_104932

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom fg_eq : ∀ x : ℝ, f x + g x = x^3 - x^2 + 1

theorem f_at_1 : f 1 = 1 := by
  sorry

end NUMINAMATH_GPT_f_at_1_l1049_104932


namespace NUMINAMATH_GPT_incorrect_inequality_l1049_104919

theorem incorrect_inequality (m n : ℝ) (a : ℝ) (hmn : m > n) (hm1 : m > 1) (hn1 : n > 1) (ha0 : 0 < a) (ha1 : a < 1) : 
  ¬ (a^m > a^n) :=
sorry

end NUMINAMATH_GPT_incorrect_inequality_l1049_104919


namespace NUMINAMATH_GPT_imaginary_part_of_complex_number_l1049_104917

theorem imaginary_part_of_complex_number :
  let z := (1 + Complex.I)^2 * (2 + Complex.I)
  Complex.im z = 4 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_number_l1049_104917


namespace NUMINAMATH_GPT_flower_shop_options_l1049_104911

theorem flower_shop_options:
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, 2 * p.1 + 3 * p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) ∧ S.card = 4 :=
sorry

end NUMINAMATH_GPT_flower_shop_options_l1049_104911


namespace NUMINAMATH_GPT_liquid_x_percentage_l1049_104922

theorem liquid_x_percentage 
  (percentage_a : ℝ) (percentage_b : ℝ)
  (weight_a : ℝ) (weight_b : ℝ)
  (h1 : percentage_a = 0.8)
  (h2 : percentage_b = 1.8)
  (h3 : weight_a = 400)
  (h4 : weight_b = 700) :
  (weight_a * (percentage_a / 100) + weight_b * (percentage_b / 100)) / (weight_a + weight_b) * 100 = 1.44 := 
by
  sorry

end NUMINAMATH_GPT_liquid_x_percentage_l1049_104922


namespace NUMINAMATH_GPT_tan_value_l1049_104926

theorem tan_value (θ : ℝ) (h : Real.sin (12 * Real.pi / 5 + θ) + 2 * Real.sin (11 * Real.pi / 10 - θ) = 0) :
  Real.tan (2 * Real.pi / 5 + θ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_value_l1049_104926


namespace NUMINAMATH_GPT_original_agreed_amount_l1049_104925

theorem original_agreed_amount (months: ℕ) (cash: ℚ) (uniform_price: ℚ) (received_total: ℚ) (full_year: ℚ) :
  months = 9 →
  cash = 300 →
  uniform_price = 300 →
  received_total = 600 →
  full_year = (12: ℚ) →
  ((months / full_year) * (cash + uniform_price) = received_total) →
  cash + uniform_price = 800 := 
by
  intros h_months h_cash h_uniform h_received h_year h_proportion
  sorry

end NUMINAMATH_GPT_original_agreed_amount_l1049_104925


namespace NUMINAMATH_GPT_prob_rain_at_least_one_day_l1049_104969

noncomputable def prob_rain_saturday := 0.35
noncomputable def prob_rain_sunday := 0.45

theorem prob_rain_at_least_one_day : 
  (1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)) * 100 = 64.25 := 
by 
  sorry

end NUMINAMATH_GPT_prob_rain_at_least_one_day_l1049_104969


namespace NUMINAMATH_GPT_smallest_possible_N_l1049_104934

theorem smallest_possible_N :
  ∀ (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0),
  p + q + r + s + t = 4020 →
  (∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1342) :=
by
  intros p q r s t hp hq hr hs ht h
  use 1342
  sorry

end NUMINAMATH_GPT_smallest_possible_N_l1049_104934


namespace NUMINAMATH_GPT_l_shape_area_is_42_l1049_104920

-- Defining the dimensions of the larger rectangle
def large_rect_length : ℕ := 10
def large_rect_width : ℕ := 7

-- Defining the smaller rectangle dimensions based on the given conditions
def small_rect_length : ℕ := large_rect_length - 3
def small_rect_width : ℕ := large_rect_width - 3

-- Defining the areas of the rectangles
def large_rect_area : ℕ := large_rect_length * large_rect_width
def small_rect_area : ℕ := small_rect_length * small_rect_width

-- Defining the area of the "L" shape
def l_shape_area : ℕ := large_rect_area - small_rect_area

-- The theorem to prove
theorem l_shape_area_is_42 : l_shape_area = 42 :=
by
  sorry

end NUMINAMATH_GPT_l_shape_area_is_42_l1049_104920


namespace NUMINAMATH_GPT_average_next_seven_l1049_104994

variable (c : ℕ) (h : c > 0)

theorem average_next_seven (d : ℕ) (h1 : d = (2 * c + 3)) 
  : (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6 := by
  sorry

end NUMINAMATH_GPT_average_next_seven_l1049_104994


namespace NUMINAMATH_GPT_find_z_proportional_l1049_104915

theorem find_z_proportional (k : ℝ) (y x z : ℝ) 
  (h₁ : y = 8) (h₂ : x = 2) (h₃ : z = 4) (relationship : y = (k * x^2) / z)
  (y' x' z' : ℝ) (h₄ : y' = 72) (h₅ : x' = 4) : 
  z' = 16 / 9 := by
  sorry

end NUMINAMATH_GPT_find_z_proportional_l1049_104915


namespace NUMINAMATH_GPT_leila_savings_l1049_104952

theorem leila_savings (S : ℝ) (h : (1 / 4) * S = 20) : S = 80 :=
by
  sorry

end NUMINAMATH_GPT_leila_savings_l1049_104952


namespace NUMINAMATH_GPT_original_number_eq_9999876_l1049_104985

theorem original_number_eq_9999876 (x : ℕ) (h : x + 9876 = 10 * x + 9 + 876) : x = 999 :=
by {
  -- Simplify the equation and solve for x
  sorry
}

end NUMINAMATH_GPT_original_number_eq_9999876_l1049_104985


namespace NUMINAMATH_GPT_total_length_of_table_free_sides_l1049_104904

theorem total_length_of_table_free_sides
  (L W : ℕ) -- Define lengths of the sides
  (h1 : L = 2 * W) -- The side opposite the wall is twice the length of each of the other two free sides
  (h2 : L * W = 128) -- The area of the rectangular table is 128 square feet
  : L + 2 * W = 32 -- Prove the total length of the table's free sides is 32 feet
  :=
sorry -- proof omitted

end NUMINAMATH_GPT_total_length_of_table_free_sides_l1049_104904


namespace NUMINAMATH_GPT_maximum_bugs_on_board_l1049_104949

-- Definition of the problem board size, bug movement directions, and non-collision rule
def board_size := 10
inductive Direction
| up | down | left | right

-- The main theorem stating the maximum number of bugs on the board
theorem maximum_bugs_on_board (bugs : List (Nat × Nat × Direction)) :
  (∀ (x y : Nat) (d : Direction) (bug : Nat × Nat × Direction), 
    bug = (x, y, d) → 
    x < board_size ∧ y < board_size ∧ 
    (∀ (c : Nat × Nat × Direction), 
      c ∈ bugs → bug ≠ c → bug.1 ≠ c.1 ∨ bug.2 ≠ c.2)) →
  List.length bugs <= 40 :=
sorry

end NUMINAMATH_GPT_maximum_bugs_on_board_l1049_104949


namespace NUMINAMATH_GPT_validate_expression_l1049_104990

-- Define the expression components
def a := 100
def b := 6
def c := 7
def d := 52
def e := 8
def f := 9

-- Define the expression using the given numbers and operations
def expression := (a - b) * c - d + e + f

-- The theorem statement asserting that the expression evaluates to 623
theorem validate_expression : expression = 623 := 
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_validate_expression_l1049_104990


namespace NUMINAMATH_GPT_sandy_potatoes_l1049_104998

theorem sandy_potatoes (n_total n_nancy n_sandy : ℕ) 
  (h_total : n_total = 13) 
  (h_nancy : n_nancy = 6) 
  (h_sum : n_total = n_nancy + n_sandy) : 
  n_sandy = 7 :=
by
  sorry

end NUMINAMATH_GPT_sandy_potatoes_l1049_104998


namespace NUMINAMATH_GPT_pq_false_l1049_104912

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x > 3 ↔ x^2 > 9
def q (a b : ℝ) : Prop := a^2 > b^2 ↔ a > b

-- Theorem to prove that p ∨ q is false given the conditions
theorem pq_false (x a b : ℝ) (hp : ¬ p x) (hq : ¬ q a b) : ¬ (p x ∨ q a b) :=
by
  sorry

end NUMINAMATH_GPT_pq_false_l1049_104912


namespace NUMINAMATH_GPT_necessary_and_sufficient_conditions_l1049_104928

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - x^2

-- Define the domain of x
def dom_x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

theorem necessary_and_sufficient_conditions {a : ℝ} (ha : a > 0) :
  (∀ x : ℝ, dom_x x → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_conditions_l1049_104928


namespace NUMINAMATH_GPT_maxRegions_formula_l1049_104963

-- Define the maximum number of regions in the plane given by n lines
def maxRegions (n: ℕ) : ℕ := (n^2 + n + 2) / 2

-- Main theorem to prove
theorem maxRegions_formula (n : ℕ) : maxRegions n = (n^2 + n + 2) / 2 := by 
  sorry

end NUMINAMATH_GPT_maxRegions_formula_l1049_104963


namespace NUMINAMATH_GPT_largest_n_divisible_103_l1049_104958

theorem largest_n_divisible_103 (n : ℕ) (h1 : n < 103) (h2 : 103 ∣ (n^3 - 1)) : n = 52 :=
sorry

end NUMINAMATH_GPT_largest_n_divisible_103_l1049_104958


namespace NUMINAMATH_GPT_calculate_entire_surface_area_l1049_104989

-- Define the problem parameters
def cube_edge_length : ℝ := 4
def hole_side_length : ℝ := 2

-- Define the function to compute the total surface area
noncomputable def entire_surface_area : ℝ :=
  let original_surface_area := 6 * (cube_edge_length ^ 2)
  let hole_area := 6 * (hole_side_length ^ 2)
  let exposed_internal_area := 6 * 4 * (hole_side_length ^ 2)
  original_surface_area - hole_area + exposed_internal_area

-- Statement of the problem to prove the given conditions
theorem calculate_entire_surface_area : entire_surface_area = 168 := by
  sorry

end NUMINAMATH_GPT_calculate_entire_surface_area_l1049_104989


namespace NUMINAMATH_GPT_people_ratio_l1049_104910

theorem people_ratio (pounds_coal : ℕ) (days1 : ℕ) (people1 : ℕ) (pounds_goal : ℕ) (days2 : ℕ) :
  pounds_coal = 10000 → days1 = 10 → people1 = 10 → pounds_goal = 40000 → days2 = 80 →
  (people1 * pounds_goal * days1) / (pounds_coal * days2) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_people_ratio_l1049_104910


namespace NUMINAMATH_GPT_geometric_progression_fourth_term_l1049_104923

theorem geometric_progression_fourth_term :
  let a1 := 3^(1/2)
  let a2 := 3^(1/3)
  let a3 := 3^(1/6)
  let r  := a3 / a2    -- Common ratio of the geometric sequence
  let a4 := a3 * r     -- Fourth term in the geometric sequence
  a4 = 1 := by
  sorry

end NUMINAMATH_GPT_geometric_progression_fourth_term_l1049_104923


namespace NUMINAMATH_GPT_Jack_minimum_cars_per_hour_l1049_104976

theorem Jack_minimum_cars_per_hour (J : ℕ) (h1 : 2 * 8 + 8 * J ≥ 40) : J ≥ 3 :=
by {
  -- The statement of the theorem directly follows
  sorry
}

end NUMINAMATH_GPT_Jack_minimum_cars_per_hour_l1049_104976


namespace NUMINAMATH_GPT_function_is_monotonic_and_odd_l1049_104907

   variable (a : ℝ) (x : ℝ)

   noncomputable def f : ℝ := (a^x - a^(-x))

   theorem function_is_monotonic_and_odd (h1 : a > 0) (h2 : a ≠ 1) : 
     (∀ x : ℝ, f (-x) = -f (x)) ∧ ((a > 1 → ∀ x y : ℝ, x < y → f x < f y) ∧ (0 < a ∧ a < 1 → ∀ x y : ℝ, x < y → f x > f y)) :=
   by
         sorry
   
end NUMINAMATH_GPT_function_is_monotonic_and_odd_l1049_104907


namespace NUMINAMATH_GPT_difference_in_ages_l1049_104970

/-- Definitions: --/
def sum_of_ages (B J : ℕ) := B + J = 70
def jennis_age (J : ℕ) := J = 19

/-- Theorem: --/
theorem difference_in_ages : ∀ (B J : ℕ), sum_of_ages B J → jennis_age J → B - J = 32 :=
by
  intros B J hsum hJ
  rw [jennis_age] at hJ
  rw [sum_of_ages] at hsum
  sorry

end NUMINAMATH_GPT_difference_in_ages_l1049_104970


namespace NUMINAMATH_GPT_g_five_eq_one_l1049_104948

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one (hx : ∀ x y : ℝ, g (x * y) = g x * g y) (h1 : g 1 ≠ 0) : g 5 = 1 :=
sorry

end NUMINAMATH_GPT_g_five_eq_one_l1049_104948


namespace NUMINAMATH_GPT_beth_comic_books_percentage_l1049_104931

/-- Definition of total books Beth owns -/
def total_books : ℕ := 120

/-- Definition of percentage novels in her collection -/
def percentage_novels : ℝ := 0.65

/-- Definition of number of graphic novels in her collection -/
def graphic_novels : ℕ := 18

/-- Calculation of the percentage of comic books she owns -/
theorem beth_comic_books_percentage (total_books : ℕ) (percentage_novels : ℝ) (graphic_novels : ℕ) : 
  (100 * ((total_books * (1 - percentage_novels) - graphic_novels) / total_books) = 20) :=
by
  let non_novel_books := total_books * (1 - percentage_novels)
  let comic_books := non_novel_books - graphic_novels
  let percentage_comic_books := 100 * (comic_books / total_books)
  have h : percentage_comic_books = 20 := sorry
  assumption

end NUMINAMATH_GPT_beth_comic_books_percentage_l1049_104931


namespace NUMINAMATH_GPT_fill_pool_time_l1049_104954

theorem fill_pool_time 
  (pool_volume : ℕ) (num_hoses : ℕ) (flow_rate_per_hose : ℕ)
  (H_pool_volume : pool_volume = 36000)
  (H_num_hoses : num_hoses = 6)
  (H_flow_rate_per_hose : flow_rate_per_hose = 3) :
  (pool_volume : ℚ) / (num_hoses * flow_rate_per_hose * 60) = 100 / 3 :=
by sorry

end NUMINAMATH_GPT_fill_pool_time_l1049_104954


namespace NUMINAMATH_GPT_dealer_pricing_l1049_104986

theorem dealer_pricing
  (cost_price : ℝ)
  (discount : ℝ := 0.10)
  (profit : ℝ := 0.20)
  (num_articles_sold : ℕ := 45)
  (num_articles_cost : ℕ := 40)
  (selling_price_per_article : ℝ := (num_articles_cost : ℝ) / num_articles_sold)
  (actual_cost_price_per_article : ℝ := selling_price_per_article / (1 + profit))
  (listed_price_per_article : ℝ := selling_price_per_article / (1 - discount)) :
  100 * ((listed_price_per_article - actual_cost_price_per_article) / actual_cost_price_per_article) = 33.33 := by
  sorry

end NUMINAMATH_GPT_dealer_pricing_l1049_104986


namespace NUMINAMATH_GPT_barber_loss_is_25_l1049_104947

-- Definition of conditions
structure BarberScenario where
  haircut_cost : ℕ
  counterfeit_bill : ℕ
  real_change : ℕ
  change_given : ℕ
  real_bill_given : ℕ

def barberScenario_example : BarberScenario :=
  { haircut_cost := 15,
    counterfeit_bill := 20,
    real_change := 20,
    change_given := 5,
    real_bill_given := 20 }

-- Lean 4 problem statement
theorem barber_loss_is_25 (b : BarberScenario) : 
  b.haircut_cost = 15 ∧
  b.counterfeit_bill = 20 ∧
  b.real_change = 20 ∧
  b.change_given = 5 ∧
  b.real_bill_given = 20 → (15 + 5 + 20 - 20 + 5 = 25) :=
by
  intro h
  cases' h with h1 h23
  sorry

end NUMINAMATH_GPT_barber_loss_is_25_l1049_104947


namespace NUMINAMATH_GPT_wings_area_l1049_104930

-- Define the areas of the two cut triangles
def A1 : ℕ := 4
def A2 : ℕ := 9

-- Define the area of the wings (remaining two triangles)
def W : ℕ := 12

-- The proof goal
theorem wings_area (A1 A2 : ℕ) (W : ℕ) : A1 = 4 → A2 = 9 → W = 12 → A1 + A2 = 13 → W = 12 :=
by
  intros hA1 hA2 hW hTotal
  -- Sorry is used as a placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_wings_area_l1049_104930


namespace NUMINAMATH_GPT_complex_expression_eq_l1049_104987

-- Define the complex numbers
def c1 : ℂ := 6 - 3 * Complex.I
def c2 : ℂ := 2 - 7 * Complex.I

-- Define the scale
def scale : ℂ := 3

-- State the theorem
theorem complex_expression_eq : (c1 + scale * c2) = 12 - 24 * Complex.I :=
by
  -- This is the statement only; the proof is omitted with sorry.
  sorry

end NUMINAMATH_GPT_complex_expression_eq_l1049_104987


namespace NUMINAMATH_GPT_age_solution_l1049_104967

noncomputable def age_problem : Prop :=
  ∃ (m s x : ℕ),
  (m - 3 = 2 * (s - 3)) ∧
  (m - 5 = 3 * (s - 5)) ∧
  (m + x) * 2 = 3 * (s + x) ∧
  x = 1

theorem age_solution : age_problem :=
  by
    sorry

end NUMINAMATH_GPT_age_solution_l1049_104967


namespace NUMINAMATH_GPT_marbles_game_winning_strategy_l1049_104946

theorem marbles_game_winning_strategy :
  ∃ k : ℕ, 1 < k ∧ k < 1024 ∧ (k = 4 ∨ k = 24 ∨ k = 40) := sorry

end NUMINAMATH_GPT_marbles_game_winning_strategy_l1049_104946


namespace NUMINAMATH_GPT_problem_statement_l1049_104968

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 2)
    (h3 : 0 < b) (h4 : b < 2) (h5 : 0 < c) (h6 : c < 2) :
    ¬ ((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1049_104968


namespace NUMINAMATH_GPT_girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l1049_104914

-- Definition of the primary condition
def girls := 3
def boys := 5

-- Statement for each part of the problem
theorem girls_together (A : ℕ → ℕ → ℕ) : 
  A (girls + boys - 1) girls * A girls girls = 4320 := 
sorry

theorem girls_separated (A : ℕ → ℕ → ℕ) : 
  A boys boys * A (girls + boys - 1) girls = 14400 := 
sorry

theorem girls_not_both_ends (A : ℕ → ℕ → ℕ) : 
  A boys 2 * A (girls + boys - 2) (girls + boys - 2) = 14400 := 
sorry

theorem girls_not_both_ends_simul (P : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ) : 
  P (girls + boys) (girls + boys) - A girls 2 * A (girls + boys - 2) (girls + boys - 2) = 36000 := 
sorry

end NUMINAMATH_GPT_girls_together_girls_separated_girls_not_both_ends_girls_not_both_ends_simul_l1049_104914


namespace NUMINAMATH_GPT_range_of_x_range_of_a_l1049_104908

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end NUMINAMATH_GPT_range_of_x_range_of_a_l1049_104908


namespace NUMINAMATH_GPT_symmetric_intersection_points_eq_y_axis_l1049_104999

theorem symmetric_intersection_points_eq_y_axis (k : ℝ) :
  (∀ x y : ℝ, (y = k * x + 1) ∧ (x^2 + y^2 + k * x - y - 9 = 0) → (∃ x' : ℝ, y = k * (-x') + 1 ∧ (x'^2 + y^2 + k * x' - y - 9 = 0) ∧ x' = -x)) →
  k = 0 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_intersection_points_eq_y_axis_l1049_104999


namespace NUMINAMATH_GPT_discount_percentage_l1049_104993

theorem discount_percentage (P D : ℝ) 
  (h1 : P > 0)
  (h2 : D = (1 - 0.28000000000000004 / 0.60)) :
  D = 0.5333333333333333 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1049_104993


namespace NUMINAMATH_GPT_sum_of_digits_in_rectangle_l1049_104961

theorem sum_of_digits_in_rectangle :
  ∃ A B C : ℕ,
    (4 + A + 1 + B = 12) ∧
    (4 + A + 1 + B = 6 + 6) ∧
    (C + 1 + 6 + C = 11) ∧
    (1 + B + 2 + C = 9) ∧
    (A + 8 + 8 = 8) ∧
    (A + 8 + B + 2 + C = 13) ∧
    (9 = 4 + A + 1 + B) ∧
    (B + 2 + C = 9) ∧    
    B = 5 ∧ A = 2 ∧ C = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_in_rectangle_l1049_104961


namespace NUMINAMATH_GPT_c_share_l1049_104950

theorem c_share (A B C : ℕ) (h1 : A + B + C = 364) (h2 : A = B / 2) (h3 : B = C / 2) : 
  C = 208 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_c_share_l1049_104950


namespace NUMINAMATH_GPT_railway_tunnel_construction_days_l1049_104906

theorem railway_tunnel_construction_days
  (a b t : ℝ)
  (h1 : a = 1/3)
  (h2 : b = 20/100)
  (h3 : t = 4/5 ∨ t = 0.8)
  (total_days : ℝ)
  (h_total_days : total_days = 185)
  : total_days = 180 := 
sorry

end NUMINAMATH_GPT_railway_tunnel_construction_days_l1049_104906


namespace NUMINAMATH_GPT_quadrilateral_area_proof_l1049_104992

noncomputable def quadrilateral_area_statement : Prop :=
  ∀ (a b : ℤ), a > b ∧ b > 0 ∧ 8 * (a - b) * (a - b) = 32 → a + b = 4

theorem quadrilateral_area_proof : quadrilateral_area_statement :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_proof_l1049_104992


namespace NUMINAMATH_GPT_triangle_area_l1049_104916

theorem triangle_area (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) 
  (h₄ : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * a * b = 30 := 
by 
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_triangle_area_l1049_104916


namespace NUMINAMATH_GPT_polynomial_evaluation_l1049_104951

theorem polynomial_evaluation (x : ℤ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1049_104951


namespace NUMINAMATH_GPT_skittles_taken_away_l1049_104953

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away (C_initial C_remaining : ℕ) (h1 : C_initial = 25) (h2 : C_remaining = 18) :
  (C_initial - C_remaining = 7) :=
by
  sorry

end NUMINAMATH_GPT_skittles_taken_away_l1049_104953


namespace NUMINAMATH_GPT_birds_initial_count_l1049_104918

theorem birds_initial_count (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end NUMINAMATH_GPT_birds_initial_count_l1049_104918


namespace NUMINAMATH_GPT_trays_needed_to_refill_l1049_104996

theorem trays_needed_to_refill (initial_ice_cubes used_ice_cubes tray_capacity : ℕ)
  (h_initial: initial_ice_cubes = 130)
  (h_used: used_ice_cubes = (initial_ice_cubes * 8 / 10))
  (h_tray_capacity: tray_capacity = 14) :
  (initial_ice_cubes + tray_capacity - 1) / tray_capacity = 10 :=
by
  sorry

end NUMINAMATH_GPT_trays_needed_to_refill_l1049_104996


namespace NUMINAMATH_GPT_binomial_expansion_constant_term_l1049_104982

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∃ c : ℝ, (3 * x^2 - (1 / (2 * x^3)))^5 = c ∧ c = 135 / 2) :=
by
  sorry

end NUMINAMATH_GPT_binomial_expansion_constant_term_l1049_104982


namespace NUMINAMATH_GPT_find_k_value_l1049_104957

theorem find_k_value (k : ℝ) (hx : ∃ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0) :
  k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_value_l1049_104957


namespace NUMINAMATH_GPT_equation_solution_l1049_104977

theorem equation_solution :
  ∃ x : ℝ, (3 * (x + 2) = x * (x + 2)) ↔ (x = -2 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l1049_104977


namespace NUMINAMATH_GPT_x_coordinate_of_tangent_point_l1049_104959

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem x_coordinate_of_tangent_point 
  (a : ℝ) 
  (h_even : ∀ x : ℝ, f x a = f (-x) a)
  (h_slope : ∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) : 
  ∃ m : ℝ, m = Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_x_coordinate_of_tangent_point_l1049_104959


namespace NUMINAMATH_GPT_range_of_m_l1049_104980

-- Definition of propositions p and q
def p (m : ℝ) : Prop := (2 * m - 3)^2 - 4 > 0
def q (m : ℝ) : Prop := m > 2

-- The main theorem stating the range of values for m
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1049_104980


namespace NUMINAMATH_GPT_marked_price_percentage_l1049_104984

variable (L P M S : ℝ)

-- Conditions
def original_list_price := 100               -- L = 100
def purchase_price := 70                     -- P = 70
def required_profit_price := 91              -- S = 91
def final_selling_price (M : ℝ) := 0.85 * M  -- S = 0.85M

-- Question: What percentage of the original list price should the marked price be?
theorem marked_price_percentage :
  L = original_list_price →
  P = purchase_price →
  S = required_profit_price →
  final_selling_price M = S →
  M = 107.06 := sorry

end NUMINAMATH_GPT_marked_price_percentage_l1049_104984


namespace NUMINAMATH_GPT_composite_quotient_is_one_over_49_l1049_104942

/-- Define the first twelve positive composite integers --/
def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def next_six_composites : List ℕ := [14, 15, 16, 18, 20, 21]

/-- Define the product of a list of integers --/
def product (l : List ℕ) : ℕ := l.foldl (λ acc x => acc * x) 1

/-- This defines the quotient of the products of the first six and the next six composite numbers --/
def composite_quotient : ℚ := (↑(product first_six_composites)) / (↑(product next_six_composites))

/-- Main theorem stating that the composite quotient equals to 1/49 --/
theorem composite_quotient_is_one_over_49 : composite_quotient = 1 / 49 := 
  by
    sorry

end NUMINAMATH_GPT_composite_quotient_is_one_over_49_l1049_104942


namespace NUMINAMATH_GPT_sqrt2_minus1_mul_sqrt2_plus1_eq1_l1049_104903

theorem sqrt2_minus1_mul_sqrt2_plus1_eq1 : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 :=
  sorry

end NUMINAMATH_GPT_sqrt2_minus1_mul_sqrt2_plus1_eq1_l1049_104903


namespace NUMINAMATH_GPT_quadrilateral_angles_combinations_pentagon_angles_combination_l1049_104975

-- Define angle types
inductive AngleType
| acute
| right
| obtuse

open AngleType

-- Define predicates for sum of angles in a quadrilateral and pentagon
def quadrilateral_sum (angles : List AngleType) : Bool :=
  match angles with
  | [right, right, right, right] => true
  | [right, right, acute, obtuse] => true
  | [right, acute, obtuse, obtuse] => true
  | [right, acute, acute, obtuse] => true
  | [acute, obtuse, obtuse, obtuse] => true
  | [acute, acute, obtuse, obtuse] => true
  | [acute, acute, acute, obtuse] => true
  | _ => false

def pentagon_sum (angles : List AngleType) : Prop :=
  -- Broad statement, more complex combinations possible
  ∃ a b c d e : ℕ, (a + b + c + d + e = 540) ∧
    (a < 90 ∨ a = 90 ∨ a > 90) ∧
    (b < 90 ∨ b = 90 ∨ b > 90) ∧
    (c < 90 ∨ c = 90 ∨ c > 90) ∧
    (d < 90 ∨ d = 90 ∨ d > 90) ∧
    (e < 90 ∨ e = 90 ∨ e > 90)

-- Prove the possible combinations for a quadrilateral and a pentagon
theorem quadrilateral_angles_combinations {angles : List AngleType} :
  quadrilateral_sum angles = true :=
sorry

theorem pentagon_angles_combination :
  ∃ angles : List AngleType, pentagon_sum angles :=
sorry

end NUMINAMATH_GPT_quadrilateral_angles_combinations_pentagon_angles_combination_l1049_104975


namespace NUMINAMATH_GPT_tan_alpha_add_pi_over_3_l1049_104956

theorem tan_alpha_add_pi_over_3 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5) 
  (h2 : Real.tan (β - π / 3) = 1 / 4) : 
  Real.tan (α + π / 3) = 7 / 23 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_add_pi_over_3_l1049_104956


namespace NUMINAMATH_GPT_find_cost_price_of_radio_l1049_104981

def cost_price_of_radio
  (profit_percent: ℝ) (overhead_expenses: ℝ) (selling_price: ℝ) (C: ℝ) : Prop :=
  profit_percent = ((selling_price - (C + overhead_expenses)) / C) * 100

theorem find_cost_price_of_radio :
  cost_price_of_radio 21.457489878542503 15 300 234.65 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_of_radio_l1049_104981


namespace NUMINAMATH_GPT_coordinates_of_A_l1049_104909

/-- The initial point A and the transformations applied to it -/
def initial_point : Prod ℤ ℤ := (-3, 2)

def translate_right (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1 + units, p.2)

def translate_down (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1, p.2 - units)

/-- Proof that the point A' has coordinates (1, -1) -/
theorem coordinates_of_A' : 
  translate_down (translate_right initial_point 4) 3 = (1, -1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l1049_104909


namespace NUMINAMATH_GPT_value_of_a_for_perfect_square_trinomial_l1049_104972

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 2 * a * x + 9 = (x + b)^2) → (a = 3 ∨ a = -3) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_for_perfect_square_trinomial_l1049_104972


namespace NUMINAMATH_GPT_alpha_plus_beta_eq_l1049_104901

variable {α β : ℝ}
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin (α - β) = 5 / 6)
variable (h4 : Real.tan α / Real.tan β = -1 / 4)

theorem alpha_plus_beta_eq : α + β = 7 * Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_alpha_plus_beta_eq_l1049_104901


namespace NUMINAMATH_GPT_lg_eight_plus_three_lg_five_l1049_104900

theorem lg_eight_plus_three_lg_five : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  sorry

end NUMINAMATH_GPT_lg_eight_plus_three_lg_five_l1049_104900


namespace NUMINAMATH_GPT_find_b_l1049_104960

theorem find_b (h1 : 2.236 = 1 + (b - 1) * 0.618) 
               (h2 : 2.236 = b - (b - 1) * 0.618) : 
               b = 3 ∨ b = 4.236 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l1049_104960


namespace NUMINAMATH_GPT_greatest_perimeter_among_four_pieces_l1049_104988

/--
Given an isosceles triangle with a base of 12 inches and a height of 15 inches,
the greatest perimeter among the four pieces of equal area obtained by cutting
the triangle into four smaller triangles is approximately 33.43 inches.
-/
theorem greatest_perimeter_among_four_pieces :
  let base : ℝ := 12
  let height : ℝ := 15
  ∃ (P : ℝ), P = (3 + Real.sqrt (225 + 4) + Real.sqrt (225 + 9)) ∧ abs (P - 33.43) < 0.01 := sorry

end NUMINAMATH_GPT_greatest_perimeter_among_four_pieces_l1049_104988


namespace NUMINAMATH_GPT_flowchart_structure_correct_l1049_104936

-- Definitions based on conditions
def flowchart_typically_has_one_start : Prop :=
  ∃ (start : Nat), start = 1

def flowchart_typically_has_one_or_more_ends : Prop :=
  ∃ (ends : Nat), ends ≥ 1

-- Theorem for the correct statement
theorem flowchart_structure_correct :
  (flowchart_typically_has_one_start ∧ flowchart_typically_has_one_or_more_ends) →
  (∃ (start : Nat) (ends : Nat), start = 1 ∧ ends ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_flowchart_structure_correct_l1049_104936


namespace NUMINAMATH_GPT_connie_initial_marbles_l1049_104905

theorem connie_initial_marbles (marbles_given : ℝ) (marbles_left : ℝ) : 
  marbles_given = 183 → marbles_left = 593 → marbles_given + marbles_left = 776 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_connie_initial_marbles_l1049_104905


namespace NUMINAMATH_GPT_knicks_equal_knocks_l1049_104937

theorem knicks_equal_knocks :
  (8 : ℝ) / (3 : ℝ) * (5 : ℝ) / (6 : ℝ) * (30 : ℝ) = 66 + 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_knicks_equal_knocks_l1049_104937


namespace NUMINAMATH_GPT_divisible_by_11_l1049_104933

theorem divisible_by_11 (k : ℕ) (h : 0 ≤ k ∧ k ≤ 9) :
  (9 + 4 + 5 + k + 3 + 1 + 7) - 2 * (4 + k + 1) ≡ 0 [MOD 11] → k = 8 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_11_l1049_104933


namespace NUMINAMATH_GPT_pam_bags_l1049_104971

-- Definitions
def gerald_bag_apples : ℕ := 40
def pam_bag_apples : ℕ := 3 * gerald_bag_apples
def pam_total_apples : ℕ := 1200

-- Theorem stating that the number of Pam's bags is 10
theorem pam_bags : pam_total_apples / pam_bag_apples = 10 := by
  sorry

end NUMINAMATH_GPT_pam_bags_l1049_104971


namespace NUMINAMATH_GPT_a_greater_than_b_c_less_than_a_l1049_104935

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

end NUMINAMATH_GPT_a_greater_than_b_c_less_than_a_l1049_104935


namespace NUMINAMATH_GPT_min_distance_is_18_l1049_104973

noncomputable def minimize_distance (a b c d : ℝ) : ℝ := (a - c) ^ 2 + (b - d) ^ 2

theorem min_distance_is_18 (a b c d : ℝ) (h1 : b = a - 2 * Real.exp a) (h2 : c + d = 4) :
  minimize_distance a b c d = 18 :=
sorry

end NUMINAMATH_GPT_min_distance_is_18_l1049_104973


namespace NUMINAMATH_GPT_calculate_down_payment_l1049_104964

def loan_period_years : ℕ := 5
def monthly_payment : ℝ := 250.0
def car_price : ℝ := 20000.0
def months_in_year : ℕ := 12

def total_loan_period_months : ℕ := loan_period_years * months_in_year
def total_amount_paid : ℝ := monthly_payment * total_loan_period_months
def down_payment : ℝ := car_price - total_amount_paid

theorem calculate_down_payment : down_payment = 5000 :=
by 
  simp [loan_period_years, monthly_payment, car_price, months_in_year, total_loan_period_months, total_amount_paid, down_payment]
  sorry

end NUMINAMATH_GPT_calculate_down_payment_l1049_104964


namespace NUMINAMATH_GPT_determine_a_l1049_104902

theorem determine_a : ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1049_104902


namespace NUMINAMATH_GPT_square_area_l1049_104924

theorem square_area (x : ℝ) 
  (h1 : 5 * x - 18 = 27 - 4 * x) 
  (side_length : ℝ := 5 * x - 18) : 
  side_length ^ 2 = 49 := 
by 
  sorry

end NUMINAMATH_GPT_square_area_l1049_104924


namespace NUMINAMATH_GPT_percent_non_condiments_l1049_104962

def sandwich_weight : ℕ := 150
def condiment_weight : ℕ := 45
def non_condiment_weight (total: ℕ) (condiments: ℕ) : ℕ := total - condiments
def percentage (num denom: ℕ) : ℕ := (num * 100) / denom

theorem percent_non_condiments : 
  percentage (non_condiment_weight sandwich_weight condiment_weight) sandwich_weight = 70 :=
by
  sorry

end NUMINAMATH_GPT_percent_non_condiments_l1049_104962


namespace NUMINAMATH_GPT_total_area_correct_at_stage_5_l1049_104965

def initial_side_length := 3

def side_length (n : ℕ) : ℕ := initial_side_length + n

def area (side : ℕ) : ℕ := side * side

noncomputable def total_area_at_stage_5 : ℕ :=
  (area (side_length 0)) + (area (side_length 1)) + (area (side_length 2)) + (area (side_length 3)) + (area (side_length 4))

theorem total_area_correct_at_stage_5 : total_area_at_stage_5 = 135 :=
by
  sorry

end NUMINAMATH_GPT_total_area_correct_at_stage_5_l1049_104965


namespace NUMINAMATH_GPT_original_number_is_two_l1049_104941

theorem original_number_is_two (x : ℝ) (hx : 0 < x) (h : x^2 = 8 * (1 / x)) : x = 2 :=
  sorry

end NUMINAMATH_GPT_original_number_is_two_l1049_104941


namespace NUMINAMATH_GPT_vector_add_sub_eq_l1049_104927

-- Define the vectors involved in the problem
def v1 : ℝ×ℝ×ℝ := (4, -3, 7)
def v2 : ℝ×ℝ×ℝ := (-1, 5, 2)
def v3 : ℝ×ℝ×ℝ := (2, -4, 9)

-- Define the result of the given vector operations
def result : ℝ×ℝ×ℝ := (1, 6, 0)

-- State the theorem we want to prove
theorem vector_add_sub_eq :
  v1 + v2 - v3 = result :=
sorry

end NUMINAMATH_GPT_vector_add_sub_eq_l1049_104927


namespace NUMINAMATH_GPT_sum_of_five_consecutive_integers_l1049_104983

theorem sum_of_five_consecutive_integers (n : ℤ) :
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 5 * n :=
by
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_integers_l1049_104983


namespace NUMINAMATH_GPT_probability_A1_selected_probability_neither_A2_B2_selected_l1049_104991

-- Define the set of students
structure Student := (id : String) (gender : String)

def students : List Student :=
  [⟨"A1", "M"⟩, ⟨"A2", "M"⟩, ⟨"A3", "M"⟩, ⟨"A4", "M"⟩, ⟨"B1", "F"⟩, ⟨"B2", "F"⟩, ⟨"B3", "F"⟩]

-- Define the conditions
def males := students.filter (λ s => s.gender = "M")
def females := students.filter (λ s => s.gender = "F")

def possible_pairs : List (Student × Student) :=
  List.product males females

-- Prove the probability of selecting A1
theorem probability_A1_selected : (3 : ℚ) / (12 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by
  sorry

-- Prove the probability that neither A2 nor B2 are selected
theorem probability_neither_A2_B2_selected : (11 : ℚ) / (12 : ℚ) = (11 : ℚ) / (12 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_A1_selected_probability_neither_A2_B2_selected_l1049_104991
