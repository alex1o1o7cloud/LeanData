import Mathlib

namespace multiplication_correct_l209_209889

theorem multiplication_correct :
  23 * 195 = 4485 :=
by
  sorry

end multiplication_correct_l209_209889


namespace books_number_in_series_l209_209908

-- Definitions and conditions from the problem
def number_books (B : ℕ) := B
def number_movies (M : ℕ) := M
def movies_watched := 61
def books_read := 19
def diff_movies_books := 2

-- The main statement to prove
theorem books_number_in_series (B M: ℕ) 
  (h1 : M = movies_watched)
  (h2 : M - B = diff_movies_books) :
  B = 59 :=
by
  sorry

end books_number_in_series_l209_209908


namespace range_of_m_l209_209221

variable (m : ℝ)

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1 }
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (h : A m ∪ B = B) : m ≤ 11 / 3 := by
  sorry

end range_of_m_l209_209221


namespace fraction_product_l209_209048

theorem fraction_product :
  ((5/4) * (8/16) * (20/12) * (32/64) * (50/20) * (40/80) * (70/28) * (48/96) : ℚ) = 625/768 := 
by
  sorry

end fraction_product_l209_209048


namespace martin_big_bell_rings_l209_209076

theorem martin_big_bell_rings (B S : ℚ) (h1 : S = B / 3 + B^2 / 4) (h2 : S + B = 52) : B = 12 :=
by
  sorry

end martin_big_bell_rings_l209_209076


namespace evaluate_expression_zero_l209_209601

-- Define the variables and conditions
def x : ℕ := 4
def z : ℕ := 0

-- State the property to be proved
theorem evaluate_expression_zero : z * (2 * z - 5 * x) = 0 := by
  sorry

end evaluate_expression_zero_l209_209601


namespace find_a_for_quadratic_roots_l209_209923

theorem find_a_for_quadratic_roots :
  ∀ (a x₁ x₂ : ℝ), 
    (x₁ ≠ x₂) →
    (x₁ * x₁ + a * x₁ + 6 = 0) →
    (x₂ * x₂ + a * x₂ + 6 = 0) →
    (x₁ - (72 / (25 * x₂^3)) = x₂ - (72 / (25 * x₁^3))) →
    (a = 9 ∨ a = -9) :=
by
  sorry

end find_a_for_quadratic_roots_l209_209923


namespace dihedral_angle_of_equilateral_triangle_l209_209402

theorem dihedral_angle_of_equilateral_triangle (a : ℝ) 
(ABC_eq : ∀ {A B C : ℝ}, (B - A) ^ 2 + (C - A) ^ 2 = a^2 ∧ (C - B) ^ 2 + (A - B) ^ 2 = a^2 ∧ (A - C) ^ 2 + (B - C) ^ 2 = a^2) 
(perpendicular : ∀ A B C D : ℝ, D = (B + C)/2 ∧ (B - D) * (C - D) = 0) : 
∃ θ : ℝ, θ = 60 := 
  sorry

end dihedral_angle_of_equilateral_triangle_l209_209402


namespace paint_gallons_needed_l209_209926

theorem paint_gallons_needed (n : ℕ) (h : n = 16) (h_col_height : ℝ) (h_col_height_val : h_col_height = 24)
  (h_col_diameter : ℝ) (h_col_diameter_val : h_col_diameter = 8) (cover_area : ℝ) 
  (cover_area_val : cover_area = 350) : 
  ∃ (gallons : ℤ), gallons = 33 := 
by
  sorry

end paint_gallons_needed_l209_209926


namespace jasmine_laps_l209_209099

theorem jasmine_laps (x : ℕ) :
  (∀ (x : ℕ), ∃ (y : ℕ), y = 60 * x) :=
by
  sorry

end jasmine_laps_l209_209099


namespace find_greater_number_l209_209740

theorem find_greater_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x > y) : x = 25 := 
sorry

end find_greater_number_l209_209740


namespace part1_solution_part2_solution_part3_solution_l209_209843

-- Define the basic conditions
variables (x y m : ℕ)

-- Part 1: Number of pieces of each type purchased (Proof for 10 pieces of A, 20 pieces of B)
theorem part1_solution (h1 : x + y = 30) (h2 : 28 * x + 22 * y = 720) :
  (x = 10) ∧ (y = 20) :=
sorry

-- Part 2: Maximize sales profit for the second purchase
theorem part2_solution (h1 : 28 * m + 22 * (80 - m) ≤ 2000) :
  m = 40 ∧ (max_profit = 1040) :=
sorry

-- Variables for Part 3
variables (a : ℕ)
-- Profit equation for type B apples with adjusted selling price
theorem part3_solution (h : (4 + 2 * a) * (34 - a - 22) = 90) :
  (a = 7) ∧ (selling_price = 27) :=
sorry

end part1_solution_part2_solution_part3_solution_l209_209843


namespace equilateral_triangle_area_decrease_l209_209686

theorem equilateral_triangle_area_decrease (s : ℝ) (A : ℝ) (s_new : ℝ) (A_new : ℝ)
    (hA : A = 100 * Real.sqrt 3)
    (hs : s^2 = 400)
    (hs_new : s_new = s - 6)
    (hA_new : A_new = (Real.sqrt 3 / 4) * s_new^2) :
    (A - A_new) / A * 100 = 51 := by
  sorry

end equilateral_triangle_area_decrease_l209_209686


namespace point_D_in_fourth_quadrant_l209_209088

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (-1, -2)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (1, -2)

theorem point_D_in_fourth_quadrant : is_in_fourth_quadrant (point_D.1) (point_D.2) :=
by
  sorry

end point_D_in_fourth_quadrant_l209_209088


namespace samantha_exam_score_l209_209465

theorem samantha_exam_score :
  ∀ (q1 q2 q3 : ℕ) (s1 s2 s3 : ℚ),
  q1 = 30 → q2 = 50 → q3 = 20 →
  s1 = 0.75 → s2 = 0.8 → s3 = 0.65 →
  (22.5 + 40 + 2 * (0.65 * 20)) / (30 + 50 + 2 * 20) = 0.7375 :=
by
  intros q1 q2 q3 s1 s2 s3 hq1 hq2 hq3 hs1 hs2 hs3
  sorry

end samantha_exam_score_l209_209465


namespace decimal_to_binary_51_l209_209443

theorem decimal_to_binary_51 : (51 : ℕ) = 0b110011 := by sorry

end decimal_to_binary_51_l209_209443


namespace inequality_holds_l209_209847

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
axiom symmetric_property : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom increasing_property : ∀ x y : ℝ, (1 ≤ x) → (x ≤ y) → f x ≤ f y

-- The statement of the theorem
theorem inequality_holds (m : ℝ) (h : m < 1 / 2) : f (1 - m) < f m :=
by sorry

end inequality_holds_l209_209847


namespace find_n_l209_209224

/-- In the expansion of (1 + 3x)^n, where n is a positive integer and n >= 6, 
    if the coefficients of x^5 and x^6 are equal, then n is 7. -/
theorem find_n (n : ℕ) (h₀ : 0 < n) (h₁ : 6 ≤ n)
  (h₂ : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : 
  n = 7 := 
sorry

end find_n_l209_209224


namespace irreducible_fraction_l209_209823

theorem irreducible_fraction {n : ℕ} : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l209_209823


namespace greatest_five_consecutive_odd_integers_l209_209679

theorem greatest_five_consecutive_odd_integers (A B C D E : ℤ) (x : ℤ) 
  (h1 : B = x + 2) 
  (h2 : C = x + 4)
  (h3 : D = x + 6)
  (h4 : E = x + 8)
  (h5 : A + B + C + D + E = 148) :
  E = 33 :=
by {
  sorry -- proof not required
}

end greatest_five_consecutive_odd_integers_l209_209679


namespace max_value_of_expression_l209_209204

theorem max_value_of_expression (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_product : a * b * c = 16) : 
  a^b - b^c + c^a ≤ 263 :=
sorry

end max_value_of_expression_l209_209204


namespace percentage_decrease_l209_209008

theorem percentage_decrease 
  (P0 : ℕ) (P2 : ℕ) (H0 : P0 = 10000) (H2 : P2 = 9600) 
  (P1 : ℕ) (H1 : P1 = P0 + (20 * P0) / 100) :
  ∃ (D : ℕ), P2 = P1 - (D * P1) / 100 ∧ D = 20 :=
by
  sorry

end percentage_decrease_l209_209008


namespace range_of_3x_minus_2y_l209_209767

variable (x y : ℝ)

theorem range_of_3x_minus_2y (h1 : -1 ≤ x + y ∧ x + y ≤ 1) (h2 : 1 ≤ x - y ∧ x - y ≤ 5) :
  ∃ (a b : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ 13 ∧ (3 * x - 2 * y = a ∨ 3 * x - 2 * y = b) :=
by
  sorry

end range_of_3x_minus_2y_l209_209767


namespace range_of_a_l209_209461

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by {
  sorry -- Proof is not required as per instructions.
}

end range_of_a_l209_209461


namespace average_marks_is_75_l209_209390

-- Define the scores for the four tests based on the given conditions.
def first_test : ℕ := 80
def second_test : ℕ := first_test + 10
def third_test : ℕ := 65
def fourth_test : ℕ := third_test

-- Define the total marks scored in the four tests.
def total_marks : ℕ := first_test + second_test + third_test + fourth_test

-- Number of tests.
def num_tests : ℕ := 4

-- Define the average marks scored in the four tests.
def average_marks : ℕ := total_marks / num_tests

-- Prove that the average marks scored in the four tests is 75.
theorem average_marks_is_75 : average_marks = 75 :=
by
  sorry

end average_marks_is_75_l209_209390


namespace fraction_value_l209_209798

variable (u v w x : ℝ)

-- Conditions
def cond1 : Prop := u / v = 5
def cond2 : Prop := w / v = 3
def cond3 : Prop := w / x = 2 / 3

theorem fraction_value (h1 : cond1 u v) (h2 : cond2 w v) (h3 : cond3 w x) : x / u = 9 / 10 := 
by
  sorry

end fraction_value_l209_209798


namespace number_added_at_end_l209_209372

theorem number_added_at_end :
  (26.3 * 12 * 20) / 3 + 125 = 2229 := sorry

end number_added_at_end_l209_209372


namespace solve_equation_in_nat_l209_209395

theorem solve_equation_in_nat {x y : ℕ} :
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) →
  x = 2 ∧ y = 2 :=
by
  sorry

end solve_equation_in_nat_l209_209395


namespace oldest_bride_age_l209_209685

theorem oldest_bride_age (G B : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) : B = 102 :=
by
  sorry

end oldest_bride_age_l209_209685


namespace maximize_revenue_l209_209072

def revenue_function (p : ℝ) : ℝ :=
  p * (200 - 6 * p)

theorem maximize_revenue :
  ∃ (p : ℝ), (p ≤ 30) ∧ (∀ q : ℝ, (q ≤ 30) → revenue_function p ≥ revenue_function q) ∧ p = 50 / 3 :=
by
  sorry

end maximize_revenue_l209_209072


namespace min_flight_routes_l209_209413

-- Defining a problem of connecting cities with flight routes such that 
-- every city can be reached from any other city with no more than two layovers.
theorem min_flight_routes (n : ℕ) (h : n = 50) : ∃ (r : ℕ), (r = 49) ∧
  (∀ (c1 c2 : ℕ), c1 ≠ c2 → c1 < n → c2 < n → ∃ (a b : ℕ),
    a < n ∧ b < n ∧ (a = c1 ∨ a = c2) ∧ (b = c1 ∨ b = c2) ∧
    ((c1 = a ∧ c2 = b) ∨ (c1 = a ∧ b = c2) ∨ (a = c2 ∧ b = c1))) :=
by {
  sorry
}

end min_flight_routes_l209_209413


namespace max_value_of_expression_l209_209374

theorem max_value_of_expression (x y : ℝ) (h : x + y = 4) :
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 ≤ 7225 / 28 :=
sorry

end max_value_of_expression_l209_209374


namespace sqrt_four_eq_plus_minus_two_l209_209524

theorem sqrt_four_eq_plus_minus_two : ∃ y : ℤ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  -- Proof goes here
  sorry

end sqrt_four_eq_plus_minus_two_l209_209524


namespace range_of_a_l209_209422

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 :=
by 
  intro h
  sorry

end range_of_a_l209_209422


namespace abc_not_all_positive_l209_209553

theorem abc_not_all_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ac > 0) (h3 : abc > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := 
by 
sorry

end abc_not_all_positive_l209_209553


namespace area_of_bounded_region_l209_209857

theorem area_of_bounded_region (x y : ℝ) (h : y^2 + 2 * x * y + 50 * abs x = 500) : 
  ∃ A, A = 1250 :=
sorry

end area_of_bounded_region_l209_209857


namespace average_speed_rest_of_trip_l209_209190

variable (v : ℝ) -- The average speed for the rest of the trip
variable (d1 : ℝ := 30 * 5) -- Distance for the first part of the trip
variable (t1 : ℝ := 5) -- Time for the first part of the trip
variable (t_total : ℝ := 7.5) -- Total time for the trip
variable (avg_total : ℝ := 34) -- Average speed for the entire trip

def total_distance := avg_total * t_total
def d2 := total_distance - d1
def t2 := t_total - t1

theorem average_speed_rest_of_trip : 
  v = 42 :=
by
  let distance_rest := d2
  let time_rest := t2
  have v_def : v = distance_rest / time_rest := by sorry
  have v_value : v = 42 := by sorry
  exact v_value

end average_speed_rest_of_trip_l209_209190


namespace megan_initial_strawberry_jelly_beans_l209_209109

variables (s g : ℕ)

theorem megan_initial_strawberry_jelly_beans :
  (s = 3 * g) ∧ (s - 15 = 4 * (g - 15)) → s = 135 :=
by
  sorry

end megan_initial_strawberry_jelly_beans_l209_209109


namespace emerson_row_distance_l209_209364

theorem emerson_row_distance (d1 d2 total : ℕ) (h1 : d1 = 6) (h2 : d2 = 18) (h3 : total = 39) :
  15 = total - (d1 + d2) :=
by sorry

end emerson_row_distance_l209_209364


namespace units_digit_of_expression_l209_209321

theorem units_digit_of_expression :
  (4 ^ 101 * 5 ^ 204 * 9 ^ 303 * 11 ^ 404) % 10 = 0 := 
sorry

end units_digit_of_expression_l209_209321


namespace tan_105_eq_minus_2_minus_sqrt_3_l209_209053

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l209_209053


namespace correct_answers_l209_209381

-- Definitions
variable (C W : ℕ)
variable (h1 : C + W = 120)
variable (h2 : 3 * C - W = 180)

-- Goal statement
theorem correct_answers : C = 75 :=
by
  sorry

end correct_answers_l209_209381


namespace tom_marbles_l209_209444

def jason_marbles := 44
def marbles_difference := 20

theorem tom_marbles : (jason_marbles - marbles_difference = 24) :=
by
  sorry

end tom_marbles_l209_209444


namespace correct_statements_l209_209114

-- Given the values of x and y on the parabola
def parabola (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the points on the parabola
def points_on_parabola (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 3 ∧
  parabola a b c 0 = 0 ∧
  parabola a b c 1 = -1 ∧
  parabola a b c 2 = 0 ∧
  parabola a b c 3 = 3

-- Prove the correct statements
theorem correct_statements (a b c : ℝ) (h : points_on_parabola a b c) : 
  ¬(∃ x, parabola a b c x < 0 ∧ x < 0) ∧
  parabola a b c 2 = 0 :=
by 
  sorry

end correct_statements_l209_209114


namespace weighted_avg_sales_increase_l209_209475

section SalesIncrease

/-- Define the weightages for each category last year. -/
def w_e : ℝ := 0.4
def w_c : ℝ := 0.3
def w_g : ℝ := 0.3

/-- Define the percent increases for each category this year. -/
def p_e : ℝ := 0.15
def p_c : ℝ := 0.25
def p_g : ℝ := 0.35

/-- Prove that the weighted average percent increase in sales this year is 0.24 or 24%. -/
theorem weighted_avg_sales_increase :
  ((w_e * p_e) + (w_c * p_c) + (w_g * p_g)) / (w_e + w_c + w_g) = 0.24 := 
by
  sorry

end SalesIncrease

end weighted_avg_sales_increase_l209_209475


namespace count_polynomials_l209_209287

def is_polynomial (expr : String) : Bool :=
  match expr with
  | "-7"            => true
  | "x"             => true
  | "m^2 + 1/m"     => false
  | "x^2*y + 5"     => true
  | "(x + y)/2"     => true
  | "-5ab^3c^2"     => true
  | "1/y"           => false
  | _               => false

theorem count_polynomials :
  let expressions := ["-7", "x", "m^2 + 1/m", "x^2*y + 5", "(x + y)/2", "-5ab^3c^2", "1/y"]
  List.filter is_polynomial expressions |>.length = 5 :=
by
  sorry

end count_polynomials_l209_209287


namespace range_of_a_l209_209379

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1) → f a x1 ≥ g x2) →
  a ≥ -2 :=
sorry

end range_of_a_l209_209379


namespace calculate_treatment_received_l209_209302

variable (drip_rate : ℕ) (duration_hours : ℕ) (drops_convert : ℕ) (ml_convert : ℕ)

theorem calculate_treatment_received (h1 : drip_rate = 20) (h2 : duration_hours = 2) 
    (h3 : drops_convert = 100) (h4 : ml_convert = 5) : 
    (drip_rate * (duration_hours * 60) * ml_convert) / drops_convert = 120 := 
by
  sorry

end calculate_treatment_received_l209_209302


namespace six_digit_number_theorem_l209_209256

noncomputable def six_digit_number (a b c d e f : ℕ) : ℕ :=
  10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f

noncomputable def rearranged_number (a b c d e f : ℕ) : ℕ :=
  10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + a

theorem six_digit_number_theorem (a b c d e f : ℕ) (h_a : a ≠ 0) 
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  (h4 : 0 ≤ d ∧ d ≤ 9) (h5 : 0 ≤ e ∧ e ≤ 9) (h6 : 0 ≤ f ∧ f ≤ 9) 
  : six_digit_number a b c d e f = 142857 ∨ six_digit_number a b c d e f = 285714 :=
by
  sorry

end six_digit_number_theorem_l209_209256


namespace find_vector_v1_v2_l209_209898

noncomputable def point_on_line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 5 + 2 * t)

noncomputable def point_on_line_m (s : ℝ) : ℝ × ℝ :=
  (3 + 5 * s, 7 + 2 * s)

noncomputable def P_foot_of_perpendicular (B : ℝ × ℝ) : ℝ × ℝ :=
  (4, 8)  -- As derived from the given solution

noncomputable def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def vector_PB (P B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - P.1, B.2 - P.2)

theorem find_vector_v1_v2 :
  ∃ (v1 v2 : ℝ), (v1 + v2 = 1) ∧ (vector_PB (P_foot_of_perpendicular (3,7)) (3,7) = (v1, v2)) :=
  sorry

end find_vector_v1_v2_l209_209898


namespace total_pay_of_two_employees_l209_209157

theorem total_pay_of_two_employees
  (Y_pay : ℝ)
  (X_pay : ℝ)
  (h1 : Y_pay = 280)
  (h2 : X_pay = 1.2 * Y_pay) :
  X_pay + Y_pay = 616 :=
by
  sorry

end total_pay_of_two_employees_l209_209157


namespace speed_ratio_l209_209486

-- Definitions of the conditions in the problem
variables (v_A v_B : ℝ) -- speeds of A and B

-- Condition 1: positions after 3 minutes are equidistant from O
def equidistant_3min : Prop := 3 * v_A = |(-300 + 3 * v_B)|

-- Condition 2: positions after 12 minutes are equidistant from O
def equidistant_12min : Prop := 12 * v_A = |(-300 + 12 * v_B)|

-- Statement to prove
theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_12min v_A v_B) :
  v_A / v_B = 4 / 5 := sorry

end speed_ratio_l209_209486


namespace simplify_expression_l209_209414

theorem simplify_expression :
  (Real.sqrt 600 / Real.sqrt 75 - Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end simplify_expression_l209_209414


namespace continuous_stripe_probability_l209_209366

open ProbabilityTheory

noncomputable def total_stripe_combinations : ℕ := 4 ^ 6

noncomputable def favorable_stripe_outcomes : ℕ := 3 * 4

theorem continuous_stripe_probability :
  (favorable_stripe_outcomes : ℚ) / (total_stripe_combinations : ℚ) = 3 / 1024 := by
  sorry

end continuous_stripe_probability_l209_209366


namespace Carissa_ran_at_10_feet_per_second_l209_209291

theorem Carissa_ran_at_10_feet_per_second :
  ∀ (n : ℕ), 
  (∃ (a : ℕ), 
    (2 * a + 2 * n^2 * a = 260) ∧ -- Total distance
    (a + n * a = 30)) → -- Total time spent
  (2 * n = 10) :=
by
  intro n
  intro h
  sorry

end Carissa_ran_at_10_feet_per_second_l209_209291


namespace factorize_expression_l209_209732

theorem factorize_expression (a x y : ℝ) : 2 * x * (a - 2) - y * (2 - a) = (a - 2) * (2 * x + y) := 
by 
  sorry

end factorize_expression_l209_209732


namespace find_playground_side_length_l209_209863

-- Define the conditions
def playground_side_length (x : ℝ) : Prop :=
  let perimeter_square := 4 * x
  let perimeter_garden := 2 * (12 + 9)
  let total_perimeter := perimeter_square + perimeter_garden
  total_perimeter = 150

-- State the main theorem to prove that the side length of the square fence around the playground is 27 yards
theorem find_playground_side_length : ∃ x : ℝ, playground_side_length x ∧ x = 27 :=
by
  exists 27
  sorry

end find_playground_side_length_l209_209863


namespace relationship_abc_l209_209681

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 15 - Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 11 - Real.sqrt 3

theorem relationship_abc : a > c ∧ c > b := 
by
  unfold a b c
  sorry

end relationship_abc_l209_209681


namespace pi_sub_alpha_in_first_quadrant_l209_209912

theorem pi_sub_alpha_in_first_quadrant (α : ℝ) (h : π / 2 < α ∧ α < π) : 0 < π - α ∧ π - α < π / 2 :=
by
  sorry

end pi_sub_alpha_in_first_quadrant_l209_209912


namespace maximum_value_of_f_intervals_of_monotonic_increase_l209_209674

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let a1 := a x
  let b1 := b x
  a1.1 * (a1.1 + b1.1) + a1.2 * (a1.2 + b1.2)

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = 3 / 2 + Real.sqrt 2 / 2 := sorry

theorem intervals_of_monotonic_increase :
  ∃ I1 I2 : Set ℝ, 
  I1 = Set.Icc 0 (Real.pi / 8) ∧ 
  I2 = Set.Icc (5 * Real.pi / 8) Real.pi ∧ 
  (∀ x ∈ I1, ∀ y ∈ I2, x ≤ y ∧ f x ≤ f y) ∧
  (∀ x y, x ∈ I1 → y ∈ I1 → x < y → f x < f y) ∧
  (∀ x y, x ∈ I2 → y ∈ I2 → x < y → f x < f y) := sorry

end maximum_value_of_f_intervals_of_monotonic_increase_l209_209674


namespace pencil_length_l209_209789

theorem pencil_length (L : ℝ) 
  (h1 : 1 / 8 * L = b) 
  (h2 : 1 / 2 * (L - 1 / 8 * L) = w) 
  (h3 : (L - 1 / 8 * L - 1 / 2 * (L - 1 / 8 * L)) = 7 / 2) :
  L = 8 :=
sorry

end pencil_length_l209_209789


namespace b_is_arithmetic_sequence_l209_209028

theorem b_is_arithmetic_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a 1 = 1 →
  a 2 = 2 →
  (∀ n, a (n + 2) = 2 * a (n + 1) - a n + 2) →
  (∀ n, b n = a (n + 1) - a n) →
  ∃ d, ∀ n, b (n + 1) = b n + d :=
by
  intros h1 h2 h3 h4
  use 2
  sorry

end b_is_arithmetic_sequence_l209_209028


namespace hyperbola_asymptotes_l209_209350

theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  (x ^ 2 / 4 - y ^ 2 / 16 = 1) → (y = 2 * x) ∨ (y = -2 * x) :=
sorry

end hyperbola_asymptotes_l209_209350


namespace missing_number_approximately_1400_l209_209106

theorem missing_number_approximately_1400 :
  ∃ x : ℤ, x * 54 = 75625 ∧ abs (x - Int.ofNat (75625 / 54)) ≤ 1 :=
by
  sorry

end missing_number_approximately_1400_l209_209106


namespace zeros_of_f_is_pm3_l209_209344

def f (x : ℝ) : ℝ := x^2 - 9

theorem zeros_of_f_is_pm3 :
  ∃ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by sorry

end zeros_of_f_is_pm3_l209_209344


namespace f_increasing_l209_209288

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end f_increasing_l209_209288


namespace solve_for_x_l209_209568

-- Define the given equation as a hypothesis
def equation (x : ℝ) : Prop :=
  0.05 * x - 0.09 * (25 - x) = 5.4

-- State the theorem that x = 54.6428571 satisfies the given equation
theorem solve_for_x : (x : ℝ) → equation x → x = 54.6428571 :=
by
  sorry

end solve_for_x_l209_209568


namespace volume_of_S_l209_209998

-- Define the region S in terms of the conditions
def region_S (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1.5 ∧ 
  abs x + abs y ≤ 1 ∧ 
  abs z ≤ 0.5

-- Define the volume calculation function
noncomputable def volume_S : ℝ :=
  sorry -- This is where the computation/theorem proving for volume would go

-- The theorem stating the volume of S
theorem volume_of_S : volume_S = 2 / 3 :=
  sorry

end volume_of_S_l209_209998


namespace largest_number_of_square_plots_l209_209714

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end largest_number_of_square_plots_l209_209714


namespace profit_percentage_l209_209597

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 550) (hSP : SP = 715) : 
  ((SP - CP) / CP) * 100 = 30 := sorry

end profit_percentage_l209_209597


namespace price_of_basketball_l209_209541

-- Problem definitions based on conditions
def price_of_soccer_ball (x : ℝ) : Prop :=
  let price_of_basketball := 2 * x
  x + price_of_basketball = 186

theorem price_of_basketball (x : ℝ) (h : price_of_soccer_ball x) : 2 * x = 124 :=
by
  sorry

end price_of_basketball_l209_209541


namespace radius_of_circle_l209_209431

theorem radius_of_circle (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ a) : 
  ∃ R, R = (b - a) / 2 ∨ R = (b + a) / 2 :=
by {
  sorry
}

end radius_of_circle_l209_209431


namespace minutes_sean_played_each_day_l209_209417

-- Define the given conditions
def t : ℕ := 1512                               -- Total minutes played by Sean and Indira
def i : ℕ := 812                                -- Total minutes played by Indira
def d : ℕ := 14                                 -- Number of days Sean played

-- Define the to-be-proved statement
theorem minutes_sean_played_each_day : (t - i) / d = 50 :=
by
  sorry

end minutes_sean_played_each_day_l209_209417


namespace percentage_of_pushups_l209_209412

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end percentage_of_pushups_l209_209412


namespace corrected_mean_l209_209633

theorem corrected_mean (mean : ℝ) (n : ℕ) (wrong_ob : ℝ) (correct_ob : ℝ) 
(h1 : mean = 36) (h2 : n = 50) (h3 : wrong_ob = 23) (h4 : correct_ob = 34) : 
(mean * n + (correct_ob - wrong_ob)) / n = 36.22 :=
by
  sorry

end corrected_mean_l209_209633


namespace simplify_expression_l209_209419

variable (x : ℝ)

theorem simplify_expression : 
  (3 * x + 6 - 5 * x) / 3 = (-2 / 3) * x + 2 := by
  sorry

end simplify_expression_l209_209419


namespace egyptian_method_percentage_error_l209_209771

theorem egyptian_method_percentage_error :
  let a := 6
  let b := 4
  let c := 20
  let h := Real.sqrt (c^2 - ((a - b) / 2)^2)
  let S := ((a + b) / 2) * h
  let S1 := ((a + b) * c) / 2
  let percentage_error := abs ((20 / Real.sqrt 399) - 1) * 100
  percentage_error = abs ((20 / Real.sqrt 399) - 1) * 100 := by
  sorry

end egyptian_method_percentage_error_l209_209771


namespace am_gm_example_l209_209083

theorem am_gm_example {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 :=
sorry

end am_gm_example_l209_209083


namespace total_number_of_toys_l209_209163

theorem total_number_of_toys (average_cost_Dhoni_toys : ℕ) (number_Dhoni_toys : ℕ) 
    (price_David_toy : ℕ) (new_avg_cost : ℕ) 
    (h1 : average_cost_Dhoni_toys = 10) (h2 : number_Dhoni_toys = 5) 
    (h3 : price_David_toy = 16) (h4 : new_avg_cost = 11) : 
    (number_Dhoni_toys + 1) = 6 := 
by
  sorry

end total_number_of_toys_l209_209163


namespace standard_equation_of_hyperbola_l209_209293

noncomputable def ellipse_eccentricity_problem
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ) : Prop :=
  e = 5 / 13 ∧
  a_maj = 26 ∧
  f_1 = (-5, 0) ∧
  f_2 = (5, 0) ∧
  d = 8 →
  ∃ b, (2 * b = 3) ∧ (2 * b ≠ 0) ∧
  ∃ h k : ℝ, (0 ≤  h) ∧ (0 ≤ k) ∧
  ((h^2)/(4^2)) - ((k^2)/(3^2)) = 1

-- problem statement: 
theorem standard_equation_of_hyperbola
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ)
  (h : e = 5 / 13)
  (a_maj_length : a_maj = 26)
  (f1_coords : f_1 = (-5, 0))
  (f2_coords : f_2 = (5, 0))
  (distance_diff : d = 8) :
  ellipse_eccentricity_problem e a_maj f_1 f_2 d :=
sorry

end standard_equation_of_hyperbola_l209_209293


namespace inequality_proof_l209_209996

variable (b c : ℝ)
variable (hb : b > 0) (hc : c > 0)

theorem inequality_proof :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) :=
  sorry

end inequality_proof_l209_209996


namespace quadratic_expression_positive_intervals_l209_209207

noncomputable def quadratic_expression (x : ℝ) : ℝ := (x + 3) * (x - 1)
def interval_1 (x : ℝ) : Prop := x < (1 - Real.sqrt 13) / 2
def interval_2 (x : ℝ) : Prop := x > (1 + Real.sqrt 13) / 2

theorem quadratic_expression_positive_intervals (x : ℝ) :
  quadratic_expression x > 0 ↔ interval_1 x ∨ interval_2 x :=
by {
  sorry
}

end quadratic_expression_positive_intervals_l209_209207


namespace sum_of_squares_of_roots_l209_209556

theorem sum_of_squares_of_roots (a b : ℝ) (x₁ x₂ : ℝ)
  (h₁ : x₁^2 - (3 * a + b) * x₁ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0)
  (h₂ : x₂^2 - (3 * a + b) * x₂ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0) :
  x₁^2 + x₂^2 = 5 * (a^2 + b^2) := 
by
  sorry

end sum_of_squares_of_roots_l209_209556


namespace greatest_value_of_4a_l209_209945

-- Definitions of the given conditions
def hundreds_digit (x : ℕ) : ℕ := x / 100
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (a b c x : ℕ) : Prop :=
  hundreds_digit x = a ∧
  tens_digit x = b ∧
  units_digit x = c ∧
  4 * a = 2 * b ∧
  2 * b = c ∧
  a > 0

def difference_of_two_greatest_x : ℕ := 124

theorem greatest_value_of_4a (x1 x2 a1 a2 b1 b2 c1 c2 : ℕ) :
  satisfies_conditions a1 b1 c1 x1 →
  satisfies_conditions a2 b2 c2 x2 →
  x1 - x2 = difference_of_two_greatest_x →
  4 * a1 = 8 :=
by
  sorry

end greatest_value_of_4a_l209_209945


namespace max_watched_hours_l209_209235

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end max_watched_hours_l209_209235


namespace cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l209_209318

-- Part 1: Prove the cost of one box of brushes and one canvas each.
theorem cost_of_brushes_and_canvas (x y : ℕ) 
    (h₁ : 2 * x + 4 * y = 94) (h₂ : 4 * x + 2 * y = 98) :
    x = 17 ∧ y = 15 := by
  sorry

-- Part 2: Prove the minimum number of canvases.
theorem minimum_canvases (m : ℕ) 
    (h₃ : m + (10 - m) = 10) (h₄ : 17 * (10 - m) + 15 * m ≤ 157) :
    m ≥ 7 := by
  sorry

-- Part 3: Prove the cost-effective purchasing plan.
theorem cost_effectiveness (m n : ℕ) 
    (h₃ : m + n = 10) (h₄ : 17 * n + 15 * m ≤ 157) (h₅ : m ≤ 8) :
    (m = 8 ∧ n = 2) := by
  sorry

end cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l209_209318


namespace min_value_of_fraction_l209_209183

theorem min_value_of_fraction (n : ℕ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_of_fraction_l209_209183


namespace integral_problem1_integral_problem2_integral_problem3_l209_209664

open Real

noncomputable def integral1 := ∫ x in (0 : ℝ)..1, x * exp (-x) = 1 - 2 / exp 1
noncomputable def integral2 := ∫ x in (1 : ℝ)..2, x * log x / log 2 = 2 - 3 / (4 * log 2)
noncomputable def integral3 := ∫ x in (1 : ℝ)..Real.exp 1, (log x) ^ 2 = exp 1 - 2

theorem integral_problem1 : integral1 := sorry
theorem integral_problem2 : integral2 := sorry
theorem integral_problem3 : integral3 := sorry

end integral_problem1_integral_problem2_integral_problem3_l209_209664


namespace grace_age_is_60_l209_209172

def Grace : ℕ := 60
def motherAge : ℕ := 80
def grandmotherAge : ℕ := 2 * motherAge
def graceAge : ℕ := (3 / 8) * grandmotherAge

theorem grace_age_is_60 : graceAge = Grace := by
  sorry

end grace_age_is_60_l209_209172


namespace fg_of_2_l209_209718

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 10 :=
by
  sorry

end fg_of_2_l209_209718


namespace binary_addition_l209_209659

theorem binary_addition (M : ℕ) (hM : M = 0b101110) :
  let M_plus_five := M + 5 
  let M_plus_five_binary := 0b110011
  let M_plus_five_predecessor := 0b110010
  M_plus_five = M_plus_five_binary ∧ M_plus_five - 1 = M_plus_five_predecessor :=
by
  sorry

end binary_addition_l209_209659


namespace time_per_page_l209_209429

theorem time_per_page 
    (planning_time : ℝ := 3) 
    (fraction : ℝ := 3/4) 
    (pages_read : ℕ := 9) 
    (minutes_per_hour : ℕ := 60) : 
    (fraction * planning_time * minutes_per_hour) / pages_read = 15 := 
by
  sorry

end time_per_page_l209_209429


namespace union_setA_setB_l209_209227

noncomputable def setA : Set ℝ := { x : ℝ | 2 / (x + 1) ≥ 1 }
noncomputable def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0 }

theorem union_setA_setB : setA ∪ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end union_setA_setB_l209_209227


namespace range_of_t_l209_209801

theorem range_of_t (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : x + y > 1) (h4 : x + 1 > y) (h5 : y + 1 > x) :
    1 ≤ max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) ∧
    max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) < (1 + Real.sqrt 5) / 2 := 
sorry

end range_of_t_l209_209801


namespace find_pairs_l209_209837

theorem find_pairs (m n : ℕ) : 
  ∃ x : ℤ, x * x = 2^m * 3^n + 1 ↔ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
by
  sorry

end find_pairs_l209_209837


namespace algebra_problem_l209_209476

variable (a : ℝ)

-- Condition: Given (a + 1/a)^3 = 4
def condition : Prop := (a + 1/a)^3 = 4

-- Statement: Prove a^4 + 1/a^4 = -158/81
theorem algebra_problem (h : condition a) : a^4 + 1/a^4 = -158/81 := 
sorry

end algebra_problem_l209_209476


namespace train_stop_time_l209_209446

theorem train_stop_time : 
  let speed_exc_stoppages := 45.0
  let speed_inc_stoppages := 31.0
  let speed_diff := speed_exc_stoppages - speed_inc_stoppages
  let km_per_minute := speed_exc_stoppages / 60.0
  let stop_time := speed_diff / km_per_minute
  stop_time = 18.67 :=
  by
    sorry

end train_stop_time_l209_209446


namespace range_of_a_l209_209357

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a - x > 1 → x < 2 * a - 1)) ∧
  (∀ x : ℝ, (2 * x + 5 > 3 * a → x > (3 * a - 5) / 2)) ∧
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 6 →
    (x < 2 * a - 1 ∧ x > (3 * a - 5) / 2))) →
  7 / 3 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end range_of_a_l209_209357


namespace expression_takes_many_values_l209_209750

theorem expression_takes_many_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) :
  (∃ y : ℝ, y ≠ 0 ∧ y ≠ (y + 1) ∧ 
    (3 * x ^ 2 + 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 7) / ((x - 3) * (x + 2)) = y) :=
by
  sorry

end expression_takes_many_values_l209_209750


namespace round_trip_time_l209_209141

def boat_speed := 9 -- speed of the boat in standing water (kmph)
def stream_speed := 6 -- speed of the stream (kmph)
def distance := 210 -- distance to the place (km)

def upstream_speed := boat_speed - stream_speed
def downstream_speed := boat_speed + stream_speed

def time_upstream := distance / upstream_speed
def time_downstream := distance / downstream_speed
def total_time := time_upstream + time_downstream

theorem round_trip_time : total_time = 84 := by
  sorry

end round_trip_time_l209_209141


namespace determine_age_l209_209607

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end determine_age_l209_209607


namespace contrapositive_l209_209530

theorem contrapositive (p q : Prop) : (p → q) → (¬q → ¬p) :=
by
  sorry

end contrapositive_l209_209530


namespace veggies_count_l209_209969

def initial_tomatoes := 500
def picked_tomatoes := 325
def initial_potatoes := 400
def picked_potatoes := 270
def initial_cucumbers := 300
def planted_cucumber_plants := 200
def cucumbers_per_plant := 2
def initial_cabbages := 100
def picked_cabbages := 50
def planted_cabbage_plants := 80
def cabbages_per_cabbage_plant := 3

noncomputable def remaining_tomatoes : Nat :=
  initial_tomatoes - picked_tomatoes

noncomputable def remaining_potatoes : Nat :=
  initial_potatoes - picked_potatoes

noncomputable def remaining_cucumbers : Nat :=
  initial_cucumbers + planted_cucumber_plants * cucumbers_per_plant

noncomputable def remaining_cabbages : Nat :=
  (initial_cabbages - picked_cabbages) + planted_cabbage_plants * cabbages_per_cabbage_plant

theorem veggies_count :
  remaining_tomatoes = 175 ∧
  remaining_potatoes = 130 ∧
  remaining_cucumbers = 700 ∧
  remaining_cabbages = 290 :=
by
  sorry

end veggies_count_l209_209969


namespace selling_price_of_cycle_l209_209159

theorem selling_price_of_cycle (cp : ℝ) (loss_percentage : ℝ) (sp : ℝ) : 
  cp = 1400 → loss_percentage = 20 → sp = cp - (loss_percentage / 100) * cp → sp = 1120 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end selling_price_of_cycle_l209_209159


namespace clive_can_correct_time_l209_209220

def can_show_correct_time (hour_hand_angle minute_hand_angle : ℝ) :=
  ∃ θ : ℝ, θ ∈ [0, 360] ∧ hour_hand_angle + θ % 360 = minute_hand_angle + θ % 360

theorem clive_can_correct_time (hour_hand_angle minute_hand_angle : ℝ) :
  can_show_correct_time hour_hand_angle minute_hand_angle :=
sorry

end clive_can_correct_time_l209_209220


namespace no_solution_for_system_l209_209661

theorem no_solution_for_system (x y z : ℝ) 
  (h1 : |x| < |y - z|) 
  (h2 : |y| < |z - x|) 
  (h3 : |z| < |x - y|) : 
  false :=
sorry

end no_solution_for_system_l209_209661


namespace find_7th_term_l209_209282

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem find_7th_term 
    (a d : ℤ) 
    (h3 : a + 2 * d = 17) 
    (h5 : a + 4 * d = 39) : 
    arithmetic_sequence a d 7 = 61 := 
sorry

end find_7th_term_l209_209282


namespace solve_equation1_solve_equation2_l209_209380

open Real

theorem solve_equation1 (x : ℝ) : (x^2 - 4 * x + 3 = 0) ↔ (x = 1 ∨ x = 3) := by
  sorry

theorem solve_equation2 (x : ℝ) : (x * (x - 2) = 2 * (2 - x)) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l209_209380


namespace find_c_l209_209197

noncomputable def f (c x : ℝ) : ℝ :=
  c * x^3 + 17 * x^2 - 4 * c * x + 45

theorem find_c (h : f c (-5) = 0) : c = 94 / 21 :=
by sorry

end find_c_l209_209197


namespace union_of_sets_l209_209564

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_sets : A ∪ B = {1, 2, 3, 5, 6} :=
by sorry

end union_of_sets_l209_209564


namespace problem_statement_l209_209540

def A : ℕ := 9 * 10 * 10 * 5
def B : ℕ := 9 * 10 * 10 * 2 / 3

theorem problem_statement : A + B = 5100 := by
  sorry

end problem_statement_l209_209540


namespace purchase_price_l209_209510

theorem purchase_price (marked_price : ℝ) (discount_rate profit_rate x : ℝ)
  (h1 : marked_price = 126)
  (h2 : discount_rate = 0.05)
  (h3 : profit_rate = 0.05)
  (h4 : marked_price * (1 - discount_rate) - x = x * profit_rate) : 
  x = 114 :=
by 
  sorry

end purchase_price_l209_209510


namespace females_in_town_l209_209987

theorem females_in_town (population : ℕ) (ratio : ℕ × ℕ) (H : population = 480) (H_ratio : ratio = (3, 5)) : 
  let m := ratio.1
  let f := ratio.2
  f * (population / (m + f)) = 300 := by
  sorry

end females_in_town_l209_209987


namespace sum_of_squares_l209_209409

theorem sum_of_squares (n : ℕ) (h : n * (n + 1) * (n + 2) = 12 * (3 * n + 3)) :
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := 
sorry

end sum_of_squares_l209_209409


namespace milk_production_l209_209539

theorem milk_production (y : ℕ) (hcows : y > 0) (hcans : y + 2 > 0) (hdays : y + 3 > 0) :
  let daily_production_per_cow := (y + 2 : ℕ) / (y * (y + 3) : ℕ)
  let total_daily_production := (y + 4 : ℕ) * daily_production_per_cow
  let required_days := (y + 6 : ℕ) / total_daily_production
  required_days = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by
  sorry

end milk_production_l209_209539


namespace projection_of_vector_a_on_b_l209_209518

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / norm_b

theorem projection_of_vector_a_on_b
  (a b : ℝ × ℝ) 
  (ha : Real.sqrt (a.1^2 + a.2^2) = 1)
  (hb : Real.sqrt (b.1^2 + b.2^2) = 2)
  (theta : ℝ)
  (h_theta : theta = Real.pi * (5/6)) -- 150 degrees in radians
  (h_cos_theta : Real.cos theta = -(Real.sqrt 3 / 2)) :
  vector_projection a b = -Real.sqrt 3 / 2 := 
by
  sorry

end projection_of_vector_a_on_b_l209_209518


namespace sum_of_roots_of_quadratic_l209_209216

theorem sum_of_roots_of_quadratic :
  let a := 2
  let b := -8
  let c := 6
  let sum_of_roots := (-b / a)
  2 * (sum_of_roots) * sum_of_roots - 8 * sum_of_roots + 6 = 0 :=
by
  sorry

end sum_of_roots_of_quadratic_l209_209216


namespace inequality_always_true_l209_209015

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_always_true_l209_209015


namespace no_point_in_common_l209_209591

theorem no_point_in_common (b : ℝ) :
  (∀ (x y : ℝ), y = 2 * x + b → (x^2 / 4) + y^2 ≠ 1) ↔ (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) :=
by
  sorry

end no_point_in_common_l209_209591


namespace compute_binom_12_6_eq_1848_l209_209907

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l209_209907


namespace prob_green_second_given_first_green_l209_209492

def total_balls : Nat := 14
def green_balls : Nat := 8
def red_balls : Nat := 6

def prob_green_first_draw : ℚ := green_balls / total_balls

theorem prob_green_second_given_first_green :
  prob_green_first_draw = (8 / 14) → (green_balls / total_balls) = (4 / 7) :=
by
  sorry

end prob_green_second_given_first_green_l209_209492


namespace circles_intersect_in_two_points_l209_209002

def circle1 (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = (3/2)^2
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

theorem circles_intersect_in_two_points :
  ∃! (p : ℝ × ℝ), (circle1 p.1 p.2) ∧ (circle2 p.1 p.2) := 
sorry

end circles_intersect_in_two_points_l209_209002


namespace hotdog_cost_l209_209643

theorem hotdog_cost
  (h s : ℕ) -- Make sure to assume that the cost in cents is a natural number 
  (h1 : 3 * h + 2 * s = 360)
  (h2 : 2 * h + 3 * s = 390) :
  h = 60 :=

sorry

end hotdog_cost_l209_209643


namespace first_number_remainder_one_l209_209471

theorem first_number_remainder_one (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 2023) :
  (∀ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = a + 2 → (a % 3 ≠ b % 3 ∧ a % 3 ≠ c % 3 ∧ b % 3 ≠ c % 3))
  → (n % 3 = 1) :=
sorry

end first_number_remainder_one_l209_209471


namespace lattice_point_exists_l209_209179

noncomputable def exists_distant_lattice_point : Prop :=
∃ (X Y : ℤ), ∀ (x y : ℤ), gcd x y = 1 → (X - x) ^ 2 + (Y - y) ^ 2 ≥ 1995 ^ 2

theorem lattice_point_exists : exists_distant_lattice_point :=
sorry

end lattice_point_exists_l209_209179


namespace derivative_of_exp_sin_l209_209400

theorem derivative_of_exp_sin (x : ℝ) : 
  (deriv (fun x => Real.exp x * Real.sin x)) x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
sorry

end derivative_of_exp_sin_l209_209400


namespace daily_evaporation_rate_l209_209265

/-- A statement that verifies the daily water evaporation rate -/
theorem daily_evaporation_rate
  (initial_water : ℝ)
  (evaporation_percentage : ℝ)
  (evaporation_period : ℕ) :
  initial_water = 15 →
  evaporation_percentage = 0.05 →
  evaporation_period = 15 →
  (evaporation_percentage * initial_water / evaporation_period) = 0.05 :=
by
  intros h_water h_percentage h_period
  sorry

end daily_evaporation_rate_l209_209265


namespace value_of_a5_l209_209054

theorem value_of_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = 2 * n * (n + 1)) (ha : ∀ n, a n = S n - S (n - 1)) :
  a 5 = 20 :=
by
  sorry

end value_of_a5_l209_209054


namespace symmetric_points_addition_l209_209421

theorem symmetric_points_addition 
  (m n : ℝ)
  (A : (ℝ × ℝ)) (B : (ℝ × ℝ))
  (hA : A = (2, m)) 
  (hB : B = (n, -1))
  (symmetry : A.1 = B.1 ∧ A.2 = -B.2) : 
  m + n = 3 :=
by
  sorry

end symmetric_points_addition_l209_209421


namespace greatest_possible_value_of_y_l209_209063

theorem greatest_possible_value_of_y 
  (x y : ℤ) 
  (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ (exists x, x * y + 7 * x + 6 * y = -8) := 
sorry

end greatest_possible_value_of_y_l209_209063


namespace correct_option_C_l209_209950

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complements of sets A and B in U
def complA : Set ℕ := {2, 4}
def complB : Set ℕ := {3, 4}

-- Define sets A and B using the complements
def A : Set ℕ := U \ complA
def B : Set ℕ := U \ complB

-- Mathematical proof problem statement
theorem correct_option_C : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end correct_option_C_l209_209950


namespace molecular_weight_of_compound_l209_209225

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def num_C : ℕ := 4
def num_H : ℕ := 1
def num_O : ℕ := 1

theorem molecular_weight_of_compound : 
  (num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O) = 65.048 := 
  by 
  -- proof skipped
  sorry

end molecular_weight_of_compound_l209_209225


namespace hockey_league_games_l209_209487

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l209_209487


namespace powers_of_2_not_powers_of_4_l209_209654

theorem powers_of_2_not_powers_of_4 (n : ℕ) (h1 : n < 500000) (h2 : ∃ k : ℕ, n = 2^k) (h3 : ∀ m : ℕ, n ≠ 4^m) : n = 9 := 
by
  sorry

end powers_of_2_not_powers_of_4_l209_209654


namespace solve_for_k_l209_209805

theorem solve_for_k (k : ℝ) : (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 :=
  sorry

end solve_for_k_l209_209805


namespace max_value_of_seq_diff_l209_209405

theorem max_value_of_seq_diff :
  ∀ (a : Fin 2017 → ℝ),
    a 0 = a 2016 →
    (∀ i : Fin 2015, |a i + a (i+2) - 2 * a (i+1)| ≤ 1) →
    ∃ b : ℝ, b = 508032 ∧ ∀ i j, 1 ≤ i → i < j → j ≤ 2017 → |a i - a j| ≤ b :=
  sorry

end max_value_of_seq_diff_l209_209405


namespace common_chord_eq_l209_209858

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the statement to prove
theorem common_chord_eq (x y : ℝ) : circle1 x y ∧ circle2 x y → 2*x + y - 5 = 0 :=
sorry

end common_chord_eq_l209_209858


namespace all_selected_prob_l209_209210

def probability_of_selection (P_ram P_ravi P_raj : ℚ) : ℚ :=
  P_ram * P_ravi * P_raj

theorem all_selected_prob :
  let P_ram := 2/7
  let P_ravi := 1/5
  let P_raj := 3/8
  probability_of_selection P_ram P_ravi P_raj = 3/140 := by
  sorry

end all_selected_prob_l209_209210


namespace total_area_of_storage_units_l209_209300

theorem total_area_of_storage_units (total_units remaining_units : ℕ) 
    (size_8_by_4 length width unit_area_200 : ℕ)
    (h1 : total_units = 42)
    (h2 : remaining_units = 22)
    (h3 : length = 8)
    (h4 : width = 4)
    (h5 : unit_area_200 = 200) 
    (h6 : ∀ i : ℕ, i < 20 → unit_area_8_by_4 = length * width) 
    (h7 : ∀ j : ℕ, j < 22 → unit_area_200 = 200) :
    total_area_of_all_units = 5040 :=
by
  let unit_area_8_by_4 := length * width
  let total_area_20_units := 20 * unit_area_8_by_4
  let total_area_22_units := 22 * unit_area_200
  let total_area_of_all_units := total_area_20_units + total_area_22_units
  sorry

end total_area_of_storage_units_l209_209300


namespace arithmetic_geometric_sum_l209_209893

theorem arithmetic_geometric_sum (S : ℕ → ℕ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60)
  (h3 : (S (2 * n) - S n) ^ 2 = S n * (S (3 * n) - S (2 * n))) : 
  S (3 * n) = 63 := by
  sorry

end arithmetic_geometric_sum_l209_209893


namespace find_g6_minus_g2_div_g3_l209_209610

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition (a c : ℝ) : c^3 * g a = a^3 * g c
axiom g_nonzero : g 3 ≠ 0

theorem find_g6_minus_g2_div_g3 : (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end find_g6_minus_g2_div_g3_l209_209610


namespace factorize_quadratic_l209_209351

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by
  sorry

end factorize_quadratic_l209_209351


namespace relationship_of_products_l209_209625

theorem relationship_of_products
  {a1 a2 b1 b2 : ℝ}
  (h1 : a1 < a2)
  (h2 : b1 < b2) :
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 :=
sorry

end relationship_of_products_l209_209625


namespace no_integer_solutions_to_system_l209_209201

theorem no_integer_solutions_to_system :
  ¬ ∃ (x y z : ℤ),
    x^2 - 2 * x * y + y^2 - z^2 = 17 ∧
    -x^2 + 3 * y * z + 3 * z^2 = 27 ∧
    x^2 - x * y + 5 * z^2 = 50 :=
by
  sorry

end no_integer_solutions_to_system_l209_209201


namespace area_of_triangle_l209_209624

noncomputable def segment_length_AB : ℝ := 10
noncomputable def point_AP : ℝ := 2
noncomputable def point_PB : ℝ := segment_length_AB - point_AP -- PB = AB - AP 
noncomputable def radius_omega1 : ℝ := point_AP / 2 -- radius of ω1
noncomputable def radius_omega2 : ℝ := point_PB / 2 -- radius of ω2
noncomputable def distance_centers : ℝ := 5 -- given directly
noncomputable def length_XY : ℝ := 4 -- given directly
noncomputable def altitude_PZ : ℝ := 8 / 5 -- given directly
noncomputable def area_triangle_XPY : ℝ := (1 / 2) * length_XY * altitude_PZ

theorem area_of_triangle : area_triangle_XPY = 16 / 5 := by
  sorry

end area_of_triangle_l209_209624


namespace together_work_days_l209_209934

/-- 
  X does the work in 10 days and Y does the same work in 15 days.
  Together, they will complete the work in 6 days.
 -/
theorem together_work_days (hx : ℝ) (hy : ℝ) : 
  (hx = 10) → (hy = 15) → (1 / (1 / hx + 1 / hy) = 6) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end together_work_days_l209_209934


namespace runway_trip_time_l209_209557

-- Define the conditions
def num_models := 6
def num_bathing_suit_outfits := 2
def num_evening_wear_outfits := 3
def total_time_minutes := 60

-- Calculate the total number of outfits per model
def total_outfits_per_model := num_bathing_suit_outfits + num_evening_wear_outfits

-- Calculate the total number of runway trips
def total_runway_trips := num_models * total_outfits_per_model

-- State the goal: Time per runway trip
def time_per_runway_trip := total_time_minutes / total_runway_trips

theorem runway_trip_time : time_per_runway_trip = 2 := by
  sorry

end runway_trip_time_l209_209557


namespace frog_eggs_ratio_l209_209678

theorem frog_eggs_ratio
    (first_day : ℕ)
    (second_day : ℕ)
    (third_day : ℕ)
    (total_eggs : ℕ)
    (h1 : first_day = 50)
    (h2 : second_day = first_day * 2)
    (h3 : third_day = second_day + 20)
    (h4 : total_eggs = 810) :
    (total_eggs - (first_day + second_day + third_day)) / (first_day + second_day + third_day) = 2 :=
by
    sorry

end frog_eggs_ratio_l209_209678


namespace chosen_number_is_reconstructed_l209_209367

theorem chosen_number_is_reconstructed (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 26) :
  ∃ (a0 a1 a2 : ℤ), (a0 = 0 ∨ a0 = 1 ∨ a0 = 2) ∧ 
                     (a1 = 0 ∨ a1 = 1 ∨ a1 = 2) ∧ 
                     (a2 = 0 ∨ a2 = 1 ∨ a2 = 2) ∧ 
                     n = a0 * 3^0 + a1 * 3^1 + a2 * 3^2 ∧ 
                     n = (if a0 = 1 then 1 else 0) + (if a0 = 2 then 2 else 0) +
                         (if a1 = 1 then 3 else 0) + (if a1 = 2 then 6 else 0) +
                         (if a2 = 1 then 9 else 0) + (if a2 = 2 then 18 else 0) := 
sorry

end chosen_number_is_reconstructed_l209_209367


namespace min_third_side_length_l209_209604

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l209_209604


namespace notebooks_cost_l209_209699

theorem notebooks_cost 
  (P N : ℝ)
  (h1 : 96 * P + 24 * N = 520)
  (h2 : ∃ x : ℝ, 3 * P + x * N = 60)
  (h3 : P + N = 15.512820512820513) :
  ∃ x : ℕ, x = 4 :=
by
  sorry

end notebooks_cost_l209_209699


namespace tangent_of_inclination_of_OP_l209_209507

noncomputable def point_P_x (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def point_P_y (φ : ℝ) : ℝ := 2 * Real.sin φ

theorem tangent_of_inclination_of_OP (φ : ℝ) (h: φ = Real.pi / 6) :
  (point_P_y φ / point_P_x φ) = 2 * Real.sqrt 3 / 9 :=
by
  have h1 : point_P_x φ = 3 * (Real.sqrt 3 / 2) := by sorry
  have h2 : point_P_y φ = 1 := by sorry
  sorry

end tangent_of_inclination_of_OP_l209_209507


namespace brenda_initial_points_l209_209338

theorem brenda_initial_points
  (b : ℕ)  -- points scored by Brenda in her play
  (initial_advantage :ℕ := 22)  -- Brenda is initially 22 points ahead
  (david_score : ℕ := 32)  -- David scores 32 points
  (final_advantage : ℕ := 5)  -- Brenda is 5 points ahead after both plays
  (h : initial_advantage + b - david_score = final_advantage) :
  b = 15 :=
by
  sorry

end brenda_initial_points_l209_209338


namespace min_M_value_l209_209045

noncomputable def max_pq (p q : ℝ) : ℝ := if p ≥ q then p else q

noncomputable def M (x y : ℝ) : ℝ := max_pq (|x^2 + y + 1|) (|y^2 - x + 1|)

theorem min_M_value : (∀ x y : ℝ, M x y ≥ (3 : ℚ) / 4) ∧ (∃ x y : ℝ, M x y = (3 : ℚ) / 4) :=
sorry

end min_M_value_l209_209045


namespace skirt_price_is_13_l209_209322

-- Definitions based on conditions
def skirts_cost (S : ℝ) : ℝ := 2 * S
def blouses_cost : ℝ := 3 * 6
def total_cost (S : ℝ) : ℝ := skirts_cost S + blouses_cost
def amount_spent : ℝ := 100 - 56

-- The statement we want to prove
theorem skirt_price_is_13 (S : ℝ) (h : total_cost S = amount_spent) : S = 13 :=
by sorry

end skirt_price_is_13_l209_209322


namespace quadratic_function_analysis_l209_209637

theorem quadratic_function_analysis (a b c : ℝ) :
  (a - b + c = -1) →
  (c = 2) →
  (4 * a + 2 * b + c = 2) →
  (16 * a + 4 * b + c = -6) →
  (¬ ∃ x > 3, a * x^2 + b * x + c = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end quadratic_function_analysis_l209_209637


namespace function_is_increasing_on_interval_l209_209274

noncomputable def f (m x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3

theorem function_is_increasing_on_interval {m : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3 ≥ (1/3) * (x - dx)^3 - (1/2) * m * (x - dx)^2 + 4 * (x - dx) - 3)
  ↔ m ≤ 4 :=
sorry

end function_is_increasing_on_interval_l209_209274


namespace solve_for_x_l209_209653

theorem solve_for_x (x : ℕ) 
  (h : 225 + 2 * 15 * 4 + 16 = x) : x = 361 := 
by 
  sorry

end solve_for_x_l209_209653


namespace sector_central_angle_l209_209537

theorem sector_central_angle (r l α : ℝ) 
  (h1 : 2 * r + l = 6) 
  (h2 : 0.5 * l * r = 2) :
  α = l / r → α = 4 ∨ α = 1 :=
sorry

end sector_central_angle_l209_209537


namespace number_of_students_l209_209743

theorem number_of_students (n : ℕ) (A : ℕ) 
  (h1 : A = 10 * n)
  (h2 : (A - 11 + 41) / n = 11) :
  n = 30 := 
sorry

end number_of_students_l209_209743


namespace candidates_appeared_l209_209181

-- Define the number of appeared candidates in state A and state B
variables (X : ℝ)

-- The conditions given in the problem
def condition1 : Prop := (0.07 * X = 0.06 * X + 83)

-- The claim that needs to be proved
def claim : Prop := (X = 8300)

-- The theorem statement in Lean 4
theorem candidates_appeared (X : ℝ) (h1 : condition1 X) : claim X := by
  -- Proof is omitted
  sorry

end candidates_appeared_l209_209181


namespace total_animals_on_farm_l209_209831

theorem total_animals_on_farm :
  let coop1 := 60
  let coop2 := 45
  let coop3 := 55
  let coop4 := 40
  let coop5 := 35
  let coop6 := 20
  let coop7 := 50
  let coop8 := 10
  let coop9 := 10
  let first_shed := 2 * 10
  let second_shed := 10
  let third_shed := 6
  let section1 := 15
  let section2 := 25
  let section3 := 2 * 15
  coop1 + coop2 + coop3 + coop4 + coop5 + coop6 + coop7 + coop8 + coop9 + first_shed + second_shed + third_shed + section1 + section2 + section3 = 431 :=
by
  sorry

end total_animals_on_farm_l209_209831


namespace initial_salt_percentage_l209_209203

theorem initial_salt_percentage (initial_mass : ℝ) (added_salt_mass : ℝ) (final_solution_percentage : ℝ) (final_mass : ℝ) 
  (h1 : initial_mass = 100) 
  (h2 : added_salt_mass = 38.46153846153846) 
  (h3 : final_solution_percentage = 0.35) 
  (h4 : final_mass = 138.46153846153846) : 
  ((10 / 100) * 100) = 10 := 
sorry

end initial_salt_percentage_l209_209203


namespace no_integer_solutions_l209_209003

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 := 
by {
  sorry
}

end no_integer_solutions_l209_209003


namespace min_value_of_inverse_sum_l209_209150

theorem min_value_of_inverse_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : (1/x) + (1/y) ≥ 4 :=
by
  sorry

end min_value_of_inverse_sum_l209_209150


namespace mila_calculator_sum_l209_209118

theorem mila_calculator_sum :
  let n := 60
  let calc1_start := 2
  let calc2_start := 0
  let calc3_start := -1
  calc1_start^(3^n) + calc2_start^2^(n) + (-calc3_start)^n = 2^(3^60) + 1 :=
by {
  sorry
}

end mila_calculator_sum_l209_209118


namespace total_pages_written_l209_209295

-- Define the conditions
def timeMon : ℕ := 60  -- Minutes on Monday
def rateMon : ℕ := 30  -- Minutes per page on Monday

def timeTue : ℕ := 45  -- Minutes on Tuesday
def rateTue : ℕ := 15  -- Minutes per page on Tuesday

def pagesWed : ℕ := 5  -- Pages written on Wednesday

-- Function to compute pages written based on time and rate
def pages_written (time rate : ℕ) : ℕ := time / rate

-- Define the theorem to be proved
theorem total_pages_written :
  pages_written timeMon rateMon + pages_written timeTue rateTue + pagesWed = 10 :=
sorry

end total_pages_written_l209_209295


namespace lesser_solution_quadratic_l209_209855

theorem lesser_solution_quadratic (x : ℝ) :
  x^2 + 9 * x - 22 = 0 → x = -11 ∨ x = 2 :=
sorry

end lesser_solution_quadratic_l209_209855


namespace first_person_work_days_l209_209281

theorem first_person_work_days (x : ℝ) (h1 : 0 < x) :
  (1/x + 1/40 = 1/15) → x = 24 :=
by
  intro h
  sorry

end first_person_work_days_l209_209281


namespace find_principal_l209_209252

theorem find_principal (R : ℝ) (P : ℝ) (h : (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56) : P = 700 := 
sorry

end find_principal_l209_209252


namespace red_blue_pencil_difference_l209_209127

theorem red_blue_pencil_difference :
  let total_pencils := 36
  let red_fraction := 5 / 9
  let blue_fraction := 5 / 12
  let red_pencils := red_fraction * total_pencils
  let blue_pencils := blue_fraction * total_pencils
  red_pencils - blue_pencils = 5 :=
by
  -- placeholder proof
  sorry

end red_blue_pencil_difference_l209_209127


namespace three_digit_numbers_not_multiples_of_3_or_11_l209_209582

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l209_209582


namespace Mike_can_play_300_minutes_l209_209811

-- Define the weekly earnings, spending, and costs as conditions
def weekly_earnings : ℕ := 100
def half_spent_at_arcade : ℕ := weekly_earnings / 2
def food_cost : ℕ := 10
def token_cost_per_hour : ℕ := 8
def hour_in_minutes : ℕ := 60

-- Define the remaining money after buying food
def money_for_tokens : ℕ := half_spent_at_arcade - food_cost

-- Define the hours he can play
def hours_playable : ℕ := money_for_tokens / token_cost_per_hour

-- Define the total minutes he can play
def total_minutes_playable : ℕ := hours_playable * hour_in_minutes

-- Prove that with his expenditure, Mike can play for 300 minutes
theorem Mike_can_play_300_minutes : total_minutes_playable = 300 := 
by
  sorry -- Proof will be filled here

end Mike_can_play_300_minutes_l209_209811


namespace bookstore_price_change_l209_209154

theorem bookstore_price_change (P : ℝ) (x : ℝ) (h : P > 0) : 
  (P * (1 + x / 100) * (1 - x / 100)) = 0.75 * P → x = 50 :=
by
  sorry

end bookstore_price_change_l209_209154


namespace prove_fraction_l209_209764

noncomputable def michael_brothers_problem (M O Y : ℕ) :=
  Y = 5 ∧
  M + O + Y = 28 ∧
  O = 2 * (M - 1) + 1 →
  Y / O = 1 / 3

theorem prove_fraction (M O Y : ℕ) : michael_brothers_problem M O Y :=
  sorry

end prove_fraction_l209_209764


namespace possible_last_digits_count_l209_209294

theorem possible_last_digits_count : 
  ∃ s : Finset Nat, s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ ∀ n ∈ s, ∃ m, (m % 10 = n) ∧ (m % 3 = 0) := 
sorry

end possible_last_digits_count_l209_209294


namespace min_shoeing_time_l209_209317

theorem min_shoeing_time
  (num_blacksmiths : ℕ) (num_horses : ℕ) (hooves_per_horse : ℕ) (minutes_per_hoof : ℕ)
  (h_blacksmiths : num_blacksmiths = 48)
  (h_horses : num_horses = 60)
  (h_hooves_per_horse : hooves_per_horse = 4)
  (h_minutes_per_hoof : minutes_per_hoof = 5) :
  (num_horses * hooves_per_horse * minutes_per_hoof) / num_blacksmiths = 25 := 
by
  sorry

end min_shoeing_time_l209_209317


namespace find_time_interval_l209_209585

-- Definitions for conditions
def birthRate : ℕ := 4
def deathRate : ℕ := 2
def netIncreaseInPopulationPerInterval (T : ℕ) : ℕ := birthRate - deathRate
def totalTimeInOneDay : ℕ := 86400
def netIncreaseInOneDay (T : ℕ) : ℕ := (totalTimeInOneDay / T) * (netIncreaseInPopulationPerInterval T)

-- Theorem statement
theorem find_time_interval (T : ℕ) (h1 : netIncreaseInPopulationPerInterval T = 2) (h2 : netIncreaseInOneDay T = 86400) : T = 2 :=
sorry

end find_time_interval_l209_209585


namespace option_C_correct_l209_209838

theorem option_C_correct (a b : ℝ) : 
  (1 / (b / a) * (a / b) = a^2 / b^2) :=
sorry

end option_C_correct_l209_209838


namespace probability_miss_at_least_once_l209_209451
-- Importing the entirety of Mathlib

-- Defining the conditions and question
variable (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1)

-- The main statement for the proof problem
theorem probability_miss_at_least_once (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) : P ≤ 1 → 0 ≤ P ∧ 1 - P^3 ≥ 0 := 
by
  sorry

end probability_miss_at_least_once_l209_209451


namespace average_age_of_town_population_l209_209378

theorem average_age_of_town_population
  (children adults : ℕ)
  (ratio_condition : 3 * adults = 2 * children)
  (avg_age_children : ℕ := 10)
  (avg_age_adults : ℕ := 40) :
  ((10 * children + 40 * adults) / (children + adults) = 22) :=
by
  sorry

end average_age_of_town_population_l209_209378


namespace total_time_spent_l209_209623

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end total_time_spent_l209_209623


namespace initial_rulers_calculation_l209_209754

variable {initial_rulers taken_rulers left_rulers : ℕ}

theorem initial_rulers_calculation 
  (h1 : taken_rulers = 25) 
  (h2 : left_rulers = 21) 
  (h3 : initial_rulers = taken_rulers + left_rulers) : 
  initial_rulers = 46 := 
by 
  sorry

end initial_rulers_calculation_l209_209754


namespace mashed_potatoes_vs_tomatoes_l209_209716

theorem mashed_potatoes_vs_tomatoes :
  let m := 144
  let t := 79
  m - t = 65 :=
by 
  repeat { sorry }

end mashed_potatoes_vs_tomatoes_l209_209716


namespace ratio_of_men_to_women_l209_209105

-- Define conditions
def avg_height_students := 180
def avg_height_female := 170
def avg_height_male := 185

-- This is the math proof problem statement
theorem ratio_of_men_to_women (M W : ℕ) (h1 : (M * avg_height_male + W * avg_height_female) = (M + W) * avg_height_students) : 
  M / W = 2 :=
sorry

end ratio_of_men_to_women_l209_209105


namespace no_prime_divisible_by_91_l209_209125

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end no_prime_divisible_by_91_l209_209125


namespace polynomial_inequality_solution_l209_209416

theorem polynomial_inequality_solution :
  {x : ℝ | x^3 - 4*x^2 - x + 20 > 0} = {x | x < -4} ∪ {x | 1 < x ∧ x < 5} ∪ {x | x > 5} :=
sorry

end polynomial_inequality_solution_l209_209416


namespace find_wall_width_l209_209354

noncomputable def wall_width (painting_width : ℝ) (painting_height : ℝ) (wall_height : ℝ) (painting_coverage : ℝ) : ℝ :=
  (painting_width * painting_height) / (painting_coverage * wall_height)

-- Given constants
def painting_width : ℝ := 2
def painting_height : ℝ := 4
def wall_height : ℝ := 5
def painting_coverage : ℝ := 0.16
def expected_width : ℝ := 10

theorem find_wall_width : wall_width painting_width painting_height wall_height painting_coverage = expected_width := 
by
  sorry

end find_wall_width_l209_209354


namespace widget_production_difference_l209_209636

variable (w t : ℕ)
variable (h_wt : w = 2 * t)

theorem widget_production_difference (w t : ℕ)
    (h_wt : w = 2 * t) :
  (w * t) - ((w + 5) * (t - 3)) = t + 15 :=
by 
  sorry

end widget_production_difference_l209_209636


namespace solution_exists_l209_209760

noncomputable def verify_triples (a b c : ℝ) : Prop :=
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ b = -2 * a ∧ c = 4 * a

theorem solution_exists (a b c : ℝ) : verify_triples a b c :=
by
  sorry

end solution_exists_l209_209760


namespace three_point_three_six_as_fraction_l209_209244

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l209_209244


namespace least_time_for_4_horses_sum_of_digits_S_is_6_l209_209886

-- Definition of horse run intervals
def horse_intervals : List Nat := List.range' 1 9 |>.map (λ k => 2 * k)

-- Function to compute LCM of a set of numbers
def lcm_set (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

-- Proving that 4 of the horse intervals have an LCM of 24
theorem least_time_for_4_horses : 
  ∃ S > 0, (S = 24 ∧ (lcm_set [2, 4, 6, 8] = S)) ∧
  (List.length (horse_intervals.filter (λ t => S % t = 0)) ≥ 4) := 
by
  sorry

-- Proving the sum of the digits of S (24) is 6
theorem sum_of_digits_S_is_6 : 
  let S := 24
  (S / 10 + S % 10 = 6) :=
by
  sorry

end least_time_for_4_horses_sum_of_digits_S_is_6_l209_209886


namespace Soyun_distance_l209_209846

theorem Soyun_distance
  (perimeter : ℕ)
  (Soyun_speed : ℕ)
  (Jia_speed : ℕ)
  (meeting_time : ℕ)
  (time_to_meet : perimeter = (Soyun_speed + Jia_speed) * meeting_time) :
  Soyun_speed * meeting_time = 10 :=
by
  sorry

end Soyun_distance_l209_209846


namespace evaluate_expression_l209_209149

variable {a b c : ℝ}

theorem evaluate_expression
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end evaluate_expression_l209_209149


namespace opposite_of_neg_two_is_two_l209_209423

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l209_209423


namespace first_even_number_l209_209963

theorem first_even_number (x : ℤ) (h : x + (x + 2) + (x + 4) = 1194) : x = 396 :=
by
  -- the proof is skipped as per instructions
  sorry

end first_even_number_l209_209963


namespace smaller_angle_in_parallelogram_l209_209091

theorem smaller_angle_in_parallelogram (a b : ℝ) (h1 : a + b = 180)
  (h2 : b = a + 70) : a = 55 :=
by sorry

end smaller_angle_in_parallelogram_l209_209091


namespace part_I_part_II_l209_209376

-- Define the sets A and B for the given conditions
def setA : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def setB (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part (Ⅰ) When a = 1, find A ∩ B
theorem part_I (a : ℝ) (ha : a = 1) :
  (setA ∩ setB a) = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

-- Part (Ⅱ) If A ∪ B = A, find the range of real number a
theorem part_II : 
  (∀ a : ℝ, setA ∪ setB a = setA → 0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l209_209376


namespace find_complementary_angle_l209_209845

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end find_complementary_angle_l209_209845


namespace product_remainder_mod_5_l209_209426

theorem product_remainder_mod_5 : (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end product_remainder_mod_5_l209_209426


namespace rectangle_area_divisible_by_12_l209_209329

theorem rectangle_area_divisible_by_12
  (x y z : ℤ)
  (h : x^2 + y^2 = z^2) :
  12 ∣ (x * y) :=
sorry

end rectangle_area_divisible_by_12_l209_209329


namespace minimum_blocks_l209_209915

-- Assume we have the following conditions encoded:
-- 
-- 1) Each block is a cube with a snap on one side and receptacle holes on the other five sides.
-- 2) Blocks can connect on the sides, top, and bottom.
-- 3) All snaps must be covered by other blocks' receptacle holes.
-- 
-- Define a formal statement of this requirement.

def block : Type := sorry -- to model the block with snap and holes
def connects (b1 b2 : block) : Prop := sorry -- to model block connectivity

def snap_covered (b : block) : Prop := sorry -- True if and only if the snap is covered by another block’s receptacle hole

theorem minimum_blocks (blocks : List block) : 
  (∀ b ∈ blocks, snap_covered b) → blocks.length ≥ 4 :=
sorry

end minimum_blocks_l209_209915


namespace num_bikes_l209_209388

variable (C B : ℕ)

-- The given conditions
def num_cars : ℕ := 10
def num_wheels_total : ℕ := 44
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- The mathematical proof problem statement
theorem num_bikes :
  C = num_cars →
  B = ((num_wheels_total - (C * wheels_per_car)) / wheels_per_bike) →
  B = 2 :=
by
  intros hC hB
  rw [hC] at hB
  sorry

end num_bikes_l209_209388


namespace sheep_to_horses_ratio_l209_209315

-- Define the known quantities
def number_of_sheep := 32
def total_horse_food := 12880
def food_per_horse := 230

-- Calculate number of horses
def number_of_horses := total_horse_food / food_per_horse

-- Calculate and simplify the ratio of sheep to horses
def ratio_of_sheep_to_horses := (number_of_sheep : ℚ) / (number_of_horses : ℚ)

-- Define the expected simplified ratio
def expected_ratio_of_sheep_to_horses := (4 : ℚ) / (7 : ℚ)

-- The statement we want to prove
theorem sheep_to_horses_ratio : ratio_of_sheep_to_horses = expected_ratio_of_sheep_to_horses :=
by
  -- Proof will be here
  sorry

end sheep_to_horses_ratio_l209_209315


namespace cost_price_per_meter_l209_209710

theorem cost_price_per_meter
  (S : ℝ) (L : ℝ) (C : ℝ) (total_meters : ℝ) (total_price : ℝ)
  (h1 : total_meters = 400) (h2 : total_price = 18000)
  (h3 : L = 5) (h4 : S = total_price / total_meters) 
  (h5 : C = S + L) :
  C = 50 :=
by
  sorry

end cost_price_per_meter_l209_209710


namespace math_problem_proof_l209_209198

-- Define the base conversion functions
def base11_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2471 => 1 * 11^0 + 7 * 11^1 + 4 * 11^2 + 2 * 11^3
  | _    => 0

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 1 * 5^0 + 2 * 5^1 + 1 * 5^2
  | _   => 0

def base7_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3654 => 4 * 7^0 + 5 * 7^1 + 6 * 7^2 + 3 * 7^3
  | _    => 0

def base8_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 5680 => 0 * 8^0 + 8 * 8^1 + 6 * 8^2 + 5 * 8^3
  | _    => 0

theorem math_problem_proof :
  let x := base11_to_base10 2471
  let y := base5_to_base10 121
  let z := base7_to_base10 3654
  let w := base8_to_base10 5680
  x / y - z + w = 1736 :=
by
  sorry

end math_problem_proof_l209_209198


namespace simplified_expr_eval_l209_209656

theorem simplified_expr_eval
  (x : ℚ) (y : ℚ) (h_x : x = -1/2) (h_y : y = 1) :
  (5*x^2 - 10*y^2) = -35/4 := 
by
  subst h_x
  subst h_y
  sorry

end simplified_expr_eval_l209_209656


namespace relationship_m_n_l209_209991

theorem relationship_m_n (b : ℝ) (m : ℝ) (n : ℝ) (h1 : m = 2 * b + 2022) (h2 : n = b^2 + 2023) : m ≤ n :=
by
  sorry

end relationship_m_n_l209_209991


namespace problem1_problem2_l209_209793

variable {a b : ℝ}

theorem problem1 (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 :=
by 
  sorry

theorem problem2 (ha : a ≠ b) :
  (a^2 / (a - b)) + (b^2 / (a - b)) - (2 * a * b / (a - b)) = a - b :=
by 
  sorry

end problem1_problem2_l209_209793


namespace identify_true_statements_l209_209049

-- Definitions of the given statements
def statement1 (a x y : ℝ) : Prop := a * (x + y) = a * x + a * y
def statement2 (a x y : ℝ) : Prop := a ^ (x + y) = a ^ x + a ^ y
def statement3 (x y : ℝ) : Prop := (x + y) ^ 2 = x ^ 2 + y ^ 2
def statement4 (a b : ℝ) : Prop := Real.sqrt (a ^ 2 + b ^ 2) = a + b
def statement5 (a b c : ℝ) : Prop := a * (b / c) = (a * b) / c

-- The statement to prove
theorem identify_true_statements (a x y b c : ℝ) :
  statement1 a x y ∧ statement5 a b c ∧
  ¬ statement2 a x y ∧ ¬ statement3 x y ∧ ¬ statement4 a b :=
sorry

end identify_true_statements_l209_209049


namespace exists_integer_square_with_three_identical_digits_l209_209250

theorem exists_integer_square_with_three_identical_digits:
  ∃ x: ℤ, (x^2 % 1000 = 444) := by
  sorry

end exists_integer_square_with_three_identical_digits_l209_209250


namespace veronica_loss_more_than_seth_l209_209355

noncomputable def seth_loss : ℝ := 17.5
noncomputable def jerome_loss : ℝ := 3 * seth_loss
noncomputable def total_loss : ℝ := 89
noncomputable def veronica_loss : ℝ := total_loss - (seth_loss + jerome_loss)

theorem veronica_loss_more_than_seth :
  veronica_loss - seth_loss = 1.5 :=
by
  have h_seth_loss : seth_loss = 17.5 := rfl
  have h_jerome_loss : jerome_loss = 3 * seth_loss := rfl
  have h_total_loss : total_loss = 89 := rfl
  have h_veronica_loss : veronica_loss = total_loss - (seth_loss + jerome_loss) := rfl
  sorry

end veronica_loss_more_than_seth_l209_209355


namespace michael_and_emma_dig_time_correct_l209_209816

noncomputable def michael_and_emma_digging_time : ℝ :=
let father_rate := 4
let father_time := 450
let father_depth := father_rate * father_time
let mother_rate := 5
let mother_time := 300
let mother_depth := mother_rate * mother_time
let michael_desired_depth := 3 * father_depth - 600
let emma_desired_depth := 2 * mother_depth + 300
let desired_depth := max michael_desired_depth emma_desired_depth
let michael_rate := 3
let emma_rate := 6
let combined_rate := michael_rate + emma_rate
desired_depth / combined_rate

theorem michael_and_emma_dig_time_correct :
  michael_and_emma_digging_time = 533.33 := 
sorry

end michael_and_emma_dig_time_correct_l209_209816


namespace polynomial_coeff_sum_l209_209258

noncomputable def polynomial_expansion (x : ℝ) :=
  (2 * x + 3) * (4 * x^3 - 2 * x^2 + x - 7)

theorem polynomial_coeff_sum :
  let A := 8
  let B := 8
  let C := -4
  let D := -11
  let E := -21
  A + B + C + D + E = -20 :=
by
  -- The following proof steps are skipped
  sorry

end polynomial_coeff_sum_l209_209258


namespace value_of_2_pow_a_l209_209701

theorem value_of_2_pow_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h1 : (2^a)^b = 2^2) (h2 : 2^a * 2^b = 8): 2^a = 2 := 
by
  sorry

end value_of_2_pow_a_l209_209701


namespace lateral_surface_area_of_rotated_triangle_l209_209304

theorem lateral_surface_area_of_rotated_triangle :
  let AC := 3
  let BC := 4
  let AB := Real.sqrt (AC ^ 2 + BC ^ 2)
  let radius := BC
  let slant_height := AB
  let lateral_surface_area := Real.pi * radius * slant_height
  lateral_surface_area = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_rotated_triangle_l209_209304


namespace factorization_correct_l209_209171

theorem factorization_correct (c d : ℤ) (h : 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) : c + 2 * d = -2 := 
sorry

end factorization_correct_l209_209171


namespace overtime_pay_rate_ratio_l209_209565

noncomputable def regular_pay_rate : ℕ := 3
noncomputable def regular_hours : ℕ := 40
noncomputable def total_pay : ℕ := 180
noncomputable def overtime_hours : ℕ := 10

theorem overtime_pay_rate_ratio : 
  (total_pay - (regular_hours * regular_pay_rate)) / overtime_hours / regular_pay_rate = 2 := by
  sorry

end overtime_pay_rate_ratio_l209_209565


namespace cost_of_fencing_theorem_l209_209470

noncomputable def cost_of_fencing (area : ℝ) (ratio_length_width : ℝ) (cost_per_meter_paise : ℝ) : ℝ :=
  let width := (area / (ratio_length_width * 2 * ratio_length_width * 3)).sqrt
  let length := ratio_length_width * 3 * width
  let perimeter := 2 * (length + width)
  let cost_per_meter_rupees := cost_per_meter_paise / 100
  perimeter * cost_per_meter_rupees

theorem cost_of_fencing_theorem :
  cost_of_fencing 3750 3 50 = 125 :=
by
  sorry

end cost_of_fencing_theorem_l209_209470


namespace base4_to_base10_conversion_l209_209427

theorem base4_to_base10_conversion : 
  (1 * 4^3 + 2 * 4^2 + 1 * 4^1 + 2 * 4^0) = 102 :=
by
  sorry

end base4_to_base10_conversion_l209_209427


namespace new_tax_rate_l209_209238

theorem new_tax_rate
  (old_rate : ℝ) (income : ℝ) (savings : ℝ) (new_rate : ℝ)
  (h1 : old_rate = 0.46)
  (h2 : income = 36000)
  (h3 : savings = 5040)
  (h4 : new_rate = (income * old_rate - savings) / income) :
  new_rate = 0.32 :=
by {
  sorry
}

end new_tax_rate_l209_209238


namespace find_b_when_a_equals_neg10_l209_209441

theorem find_b_when_a_equals_neg10 
  (ab_k : ∀ a b : ℝ, (a * b) = 675) 
  (sum_60 : ∀ a b : ℝ, (a + b = 60 → a = 3 * b)) 
  (a_eq_neg10 : ∀ a : ℝ, a = -10) : 
  ∃ b : ℝ, b = -67.5 := 
by 
  sorry

end find_b_when_a_equals_neg10_l209_209441


namespace find_xyz_l209_209193

theorem find_xyz (x y z : ℝ) (h₁ : x + 1 / y = 5) (h₂ : y + 1 / z = 2) (h₃ : z + 2 / x = 10 / 3) : x * y * z = (21 + Real.sqrt 433) / 2 :=
by
  sorry

end find_xyz_l209_209193


namespace calculate_perimeter_of_staircase_region_l209_209929

-- Define the properties and dimensions of the staircase-shaped region
def is_right_angle (angle : ℝ) : Prop := angle = 90

def congruent_side_length : ℝ := 1

def bottom_base_length : ℝ := 12

def total_area : ℝ := 78

def perimeter_region : ℝ := 34.5

theorem calculate_perimeter_of_staircase_region
  (is_right_angle : ∀ angle, is_right_angle angle)
  (congruent_sides_count : ℕ := 12)
  (total_congruent_side_length : ℝ := congruent_sides_count * congruent_side_length)
  (bottom_base_length : ℝ)
  (total_area : ℝ)
  : bottom_base_length = 12 ∧ total_area = 78 → 
    ∃ perimeter : ℝ, perimeter = 34.5 :=
by
  admit -- Proof goes here

end calculate_perimeter_of_staircase_region_l209_209929


namespace savings_after_expense_increase_l209_209208

-- Define constants and initial conditions
def salary : ℝ := 7272.727272727273
def savings_rate : ℝ := 0.10
def expense_increase_rate : ℝ := 0.05

-- Define initial savings, expenses, and new expenses
def initial_savings : ℝ := savings_rate * salary
def initial_expenses : ℝ := salary - initial_savings
def new_expenses : ℝ := initial_expenses * (1 + expense_increase_rate)
def new_savings : ℝ := salary - new_expenses

-- The theorem statement
theorem savings_after_expense_increase : new_savings = 400 := by
  sorry

end savings_after_expense_increase_l209_209208


namespace tip_percentage_is_20_l209_209286

theorem tip_percentage_is_20 (total_spent price_before_tax_and_tip : ℝ) (sales_tax_rate : ℝ) (h1 : total_spent = 158.40) (h2 : price_before_tax_and_tip = 120) (h3 : sales_tax_rate = 0.10) :
  ((total_spent - (price_before_tax_and_tip * (1 + sales_tax_rate))) / (price_before_tax_and_tip * (1 + sales_tax_rate))) * 100 = 20 :=
by
  sorry

end tip_percentage_is_20_l209_209286


namespace initial_population_l209_209810

theorem initial_population (P : ℝ) : 
  (P * 1.2 * 0.8 = 9600) → P = 10000 :=
by
  sorry

end initial_population_l209_209810


namespace greatest_of_consecutive_integers_sum_18_l209_209611

theorem greatest_of_consecutive_integers_sum_18 
  (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := 
sorry

end greatest_of_consecutive_integers_sum_18_l209_209611


namespace arithmetic_sum_S8_proof_l209_209592

-- Definitions of variables and constants
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def a1_condition : a 1 = -40 := sorry
def a6_a10_condition : a 6 + a 10 = -10 := sorry

-- Theorem to prove
theorem arithmetic_sum_S8_proof (a : ℕ → ℝ) (S : ℕ → ℝ)
  (a1 : a 1 = -40)
  (a6a10 : a 6 + a 10 = -10)
  : S 8 = -180 := 
sorry

end arithmetic_sum_S8_proof_l209_209592


namespace rainfall_second_week_january_l209_209191

-- Define the conditions
def total_rainfall_2_weeks (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_first_week + rainfall_second_week = 20

def rainfall_second_week_is_1_5_times_first (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_second_week = 1.5 * rainfall_first_week

-- Define the statement to prove
theorem rainfall_second_week_january (rainfall_first_week rainfall_second_week : ℝ) :
  total_rainfall_2_weeks rainfall_first_week rainfall_second_week →
  rainfall_second_week_is_1_5_times_first rainfall_first_week rainfall_second_week →
  rainfall_second_week = 12 :=
by
  sorry

end rainfall_second_week_january_l209_209191


namespace alternating_sum_l209_209804

theorem alternating_sum : 
  (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19 + 21 - 23 + 25 - 27 + 29 - 31 + 33 - 35 + 37 - 39 + 41 = 21) :=
by
  sorry

end alternating_sum_l209_209804


namespace a10_b10_l209_209662

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a10_b10 : a^10 + b^10 = 123 :=
by
  sorry

end a10_b10_l209_209662


namespace patty_weeks_without_chores_correct_l209_209176

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l209_209176


namespace coin_flips_probability_l209_209500

section 

-- Definition for the probability of heads in a single flip
def prob_heads : ℚ := 1 / 2

-- Definition for flipping the coin 5 times and getting heads on the first 4 flips and tails on the last flip
def prob_specific_sequence (n : ℕ) (k : ℕ) : ℚ := (prob_heads) ^ k * (prob_heads) ^ (n - k)

-- The main theorem which states the probability of the desired outcome
theorem coin_flips_probability : 
  prob_specific_sequence 5 4 = 1 / 32 :=
sorry

end

end coin_flips_probability_l209_209500


namespace total_is_twenty_l209_209748

def num_blue := 5
def num_red := 7
def prob_red_or_white : ℚ := 0.75

noncomputable def total_marbles (T : ℕ) (W : ℕ) :=
  5 + 7 + W = T ∧ (7 + W) / T = prob_red_or_white

theorem total_is_twenty : ∃ (T : ℕ) (W : ℕ), total_marbles T W ∧ T = 20 :=
by
  sorry

end total_is_twenty_l209_209748


namespace prime_constraint_unique_solution_l209_209895

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end prime_constraint_unique_solution_l209_209895


namespace employee_salary_l209_209997

theorem employee_salary (x y : ℝ) (h1 : x + y = 770) (h2 : x = 1.2 * y) : y = 350 :=
by
  sorry

end employee_salary_l209_209997


namespace man_l209_209279

theorem man's_speed_kmph (length_train : ℝ) (time_seconds : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (5/18)
  let rel_speed_mps := length_train / time_seconds
  let man_speed_mps := rel_speed_mps - speed_train_mps
  man_speed_mps * (18/5)

example : man's_speed_kmph 120 6 65.99424046076315 = 6.00735873483709 := by
  sorry

end man_l209_209279


namespace total_combined_rainfall_l209_209861

def mondayRainfall := 7 * 1
def tuesdayRainfall := 4 * 2
def wednesdayRate := 2 * 2
def wednesdayRainfall := 2 * wednesdayRate
def totalRainfall := mondayRainfall + tuesdayRainfall + wednesdayRainfall

theorem total_combined_rainfall : totalRainfall = 23 :=
by
  unfold totalRainfall mondayRainfall tuesdayRainfall wednesdayRainfall wednesdayRate
  sorry

end total_combined_rainfall_l209_209861


namespace distance_from_dormitory_to_city_l209_209813

theorem distance_from_dormitory_to_city (D : ℝ) :
  (1 / 4) * D + (1 / 2) * D + 10 = D → D = 40 :=
by
  intro h
  sorry

end distance_from_dormitory_to_city_l209_209813


namespace Paul_seashells_l209_209881

namespace SeashellProblem

variables (P L : ℕ)

def initial_total_seashells (H P L : ℕ) : Prop := H + P + L = 59

def final_total_seashells (H P L : ℕ) : Prop := H + P + L - L / 4 = 53

theorem Paul_seashells : 
  (initial_total_seashells 11 P L) → (final_total_seashells 11 P L) → P = 24 :=
by
  intros h_initial h_final
  sorry

end SeashellProblem

end Paul_seashells_l209_209881


namespace find_a3_l209_209722

-- Define the sequence sum S_n
def S (n : ℕ) : ℚ := (n + 1) / (n + 2)

-- Define the sequence term a_n using S_n
def a (n : ℕ) : ℚ :=
  if h : n = 1 then S 1 else S n - S (n - 1)

-- State the theorem to find the value of a_3
theorem find_a3 : a 3 = 1 / 20 :=
by
  -- The proof is omitted, use sorry to skip it
  sorry

end find_a3_l209_209722


namespace cafeteria_extra_fruits_l209_209280

def extra_fruits (ordered wanted : Nat) : Nat :=
  ordered - wanted

theorem cafeteria_extra_fruits :
  let red_apples_ordered := 6
  let red_apples_wanted := 5
  let green_apples_ordered := 15
  let green_apples_wanted := 8
  let oranges_ordered := 10
  let oranges_wanted := 6
  let bananas_ordered := 8
  let bananas_wanted := 7
  extra_fruits red_apples_ordered red_apples_wanted = 1 ∧
  extra_fruits green_apples_ordered green_apples_wanted = 7 ∧
  extra_fruits oranges_ordered oranges_wanted = 4 ∧
  extra_fruits bananas_ordered bananas_wanted = 1 := 
by
  sorry

end cafeteria_extra_fruits_l209_209280


namespace range_of_t_l209_209818

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + t ≤ 0 ∧ x ≤ t) ↔ (0 ≤ t ∧ t ≤ 9 / 4) := 
sorry

end range_of_t_l209_209818


namespace square_not_covered_by_circles_l209_209517

noncomputable def area_uncovered_by_circles : Real :=
  let side_length := 2
  let square_area := (side_length^2 : Real)
  let radius := 1
  let circle_area := Real.pi * radius^2
  let quarter_circle_area := circle_area / 4
  let total_circles_area := 4 * quarter_circle_area
  square_area - total_circles_area

theorem square_not_covered_by_circles :
  area_uncovered_by_circles = 4 - Real.pi := sorry

end square_not_covered_by_circles_l209_209517


namespace correct_statement_about_meiosis_and_fertilization_l209_209248

def statement_A : Prop := 
  ∃ oogonia spermatogonia zygotes : ℕ, 
    oogonia = 20 ∧ spermatogonia = 8 ∧ zygotes = 32 ∧ 
    (oogonia + spermatogonia = zygotes)

def statement_B : Prop := 
  ∀ zygote_dna mother_half father_half : ℕ,
    zygote_dna = mother_half + father_half ∧ 
    mother_half = father_half

def statement_C : Prop := 
  ∀ (meiosis stabilizes : Prop) (chromosome_count : ℕ),
    (meiosis → stabilizes) ∧ 
    (stabilizes → chromosome_count = (chromosome_count / 2 + chromosome_count / 2))

def statement_D : Prop := 
  ∀ (diversity : Prop) (gene_mutations chromosomal_variations : Prop),
    (diversity → ¬ (gene_mutations ∨ chromosomal_variations))

theorem correct_statement_about_meiosis_and_fertilization :
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D :=
by
  sorry

end correct_statement_about_meiosis_and_fertilization_l209_209248


namespace problem_statement_l209_209832

def permutations (n r : ℕ) : ℕ := n.factorial / (n - r).factorial
def combinations (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : permutations 4 2 - combinations 4 3 = 8 := 
by 
  sorry

end problem_statement_l209_209832


namespace estimate_red_balls_l209_209822

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l209_209822


namespace angle_sum_and_relation_l209_209970

variable {A B : ℝ}

theorem angle_sum_and_relation (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end angle_sum_and_relation_l209_209970


namespace car_b_speed_l209_209363

theorem car_b_speed :
  ∀ (v : ℕ),
    (232 - 4 * v = 32) →
    v = 50 :=
  by
  sorry

end car_b_speed_l209_209363


namespace sin_alpha_beta_l209_209361

theorem sin_alpha_beta (a b c α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
    (h3 : a * Real.cos α + b * Real.sin α + c = 0) (h4 : a * Real.cos β + b * Real.sin β + c = 0) 
    (h5 : α ≠ β) : Real.sin (α + β) = (2 * a * b) / (a ^ 2 + b ^ 2) := 
sorry

end sin_alpha_beta_l209_209361


namespace hh3_eq_6582_l209_209595

def h (x : ℤ) : ℤ := 3 * x^2 + 5 * x + 4

theorem hh3_eq_6582 : h (h 3) = 6582 :=
by
  sorry

end hh3_eq_6582_l209_209595


namespace purely_imaginary_sol_l209_209299

theorem purely_imaginary_sol {m : ℝ} (h : (m^2 - 3 * m) = 0) (h2 : (m^2 - 5 * m + 6) ≠ 0) : m = 0 :=
sorry

end purely_imaginary_sol_l209_209299


namespace xiaoyu_reading_days_l209_209000

theorem xiaoyu_reading_days
  (h1 : ∀ (p d : ℕ), p = 15 → d = 24 → p * d = 360)
  (h2 : ∀ (p t : ℕ), t = 360 → p = 18 → t / p = 20) :
  ∀ d : ℕ, d = 20 :=
by
  sorry

end xiaoyu_reading_days_l209_209000


namespace tan_half_angle_product_l209_209947

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0) :
  (Real.tan (a / 2)) * (Real.tan (b / 2)) = 5 ∨ (Real.tan (a / 2)) * (Real.tan (b / 2)) = -5 :=
by 
  sorry

end tan_half_angle_product_l209_209947


namespace find_m_l209_209840

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l209_209840


namespace negation_example_l209_209497

theorem negation_example :
  (¬ ∀ x y : ℝ, |x + y| > 3) ↔ (∃ x y : ℝ, |x + y| ≤ 3) :=
by
  sorry

end negation_example_l209_209497


namespace experimental_fertilizer_height_is_correct_l209_209498

/-- Define the static heights and percentages for each plant's growth conditions. -/
def control_plant_height : ℝ := 36
def bone_meal_multiplier : ℝ := 1.25
def cow_manure_multiplier : ℝ := 2
def experimental_fertilizer_multiplier : ℝ := 1.5

/-- Define each plant's height based on the given multipliers and conditions. -/
def bone_meal_plant_height : ℝ := bone_meal_multiplier * control_plant_height
def cow_manure_plant_height : ℝ := cow_manure_multiplier * bone_meal_plant_height
def experimental_fertilizer_plant_height : ℝ := experimental_fertilizer_multiplier * cow_manure_plant_height

/-- Proof that the height of the experimental fertilizer plant is 135 inches. -/
theorem experimental_fertilizer_height_is_correct :
  experimental_fertilizer_plant_height = 135 := by
    sorry

end experimental_fertilizer_height_is_correct_l209_209498


namespace sum_of_squares_xy_l209_209952

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end sum_of_squares_xy_l209_209952


namespace probability_red_is_two_fifths_l209_209403

-- Define the durations
def red_light_duration : ℕ := 30
def yellow_light_duration : ℕ := 5
def green_light_duration : ℕ := 40

-- Define total cycle duration
def total_cycle_duration : ℕ :=
  red_light_duration + yellow_light_duration + green_light_duration

-- Define the probability function
def probability_of_red_light : ℚ :=
  red_light_duration / total_cycle_duration

-- The theorem statement to prove
theorem probability_red_is_two_fifths :
  probability_of_red_light = 2/5 := sorry

end probability_red_is_two_fifths_l209_209403


namespace shoe_cost_l209_209688

def initial_amount : ℕ := 91
def cost_sweater : ℕ := 24
def cost_tshirt : ℕ := 6
def amount_left : ℕ := 50
def cost_shoes : ℕ := 11

theorem shoe_cost :
  initial_amount - (cost_sweater + cost_tshirt) - amount_left = cost_shoes :=
by
  sorry

end shoe_cost_l209_209688


namespace cost_price_as_percentage_l209_209331

theorem cost_price_as_percentage (SP CP : ℝ) 
  (profit_percentage : ℝ := 4.166666666666666) 
  (P : ℝ := SP - CP)
  (profit_eq : P = (profit_percentage / 100) * SP) :
  CP = (95.83333333333334 / 100) * SP := 
by
  sorry

end cost_price_as_percentage_l209_209331


namespace swap_values_l209_209696

theorem swap_values (A B : ℕ) (h₁ : A = 10) (h₂ : B = 20) : 
    let C := A 
    let A := B 
    let B := C
    A = 20 ∧ B = 10 := by
  let C := A
  let A := B
  let B := C
  have h₃ : C = 10 := h₁
  have h₄ : A = 20 := h₂
  have h₅ : B = 10 := h₃
  exact And.intro h₄ h₅

end swap_values_l209_209696


namespace ratio_of_ages_is_six_l209_209209

-- Definitions of ages
def Cody_age : ℕ := 14
def Grandmother_age : ℕ := 84

-- The ratio we want to prove
def age_ratio : ℕ := Grandmother_age / Cody_age

-- The theorem stating the ratio is 6
theorem ratio_of_ages_is_six : age_ratio = 6 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_ages_is_six_l209_209209


namespace algebraic_expression_value_l209_209887

theorem algebraic_expression_value (x y : ℝ) (h : x = 2 * y + 3) : 4 * x - 8 * y + 9 = 21 := by
  sorry

end algebraic_expression_value_l209_209887


namespace solution_set_of_inequality_l209_209877

theorem solution_set_of_inequality :
  ∀ x : ℝ, |2 * x^2 - 1| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end solution_set_of_inequality_l209_209877


namespace cost_per_serving_is_3_62_l209_209151

noncomputable def cost_per_serving : ℝ :=
  let beef_cost := 4 * 6
  let chicken_cost := (2.2 * 5) * 0.85
  let carrots_cost := 2 * 1.50
  let potatoes_cost := (1.5 * 1.80) * 0.85
  let onions_cost := 1 * 3
  let discounted_carrots := carrots_cost * 0.80
  let discounted_potatoes := potatoes_cost * 0.80
  let total_cost_before_tax := beef_cost + chicken_cost + discounted_carrots + discounted_potatoes + onions_cost
  let sales_tax := total_cost_before_tax * 0.07
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax / 12

theorem cost_per_serving_is_3_62 : cost_per_serving = 3.62 :=
by
  sorry

end cost_per_serving_is_3_62_l209_209151


namespace ball_hits_ground_time_l209_209993

noncomputable def h (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 180

theorem ball_hits_ground_time :
  ∃ t : ℝ, h t = 0 ∧ t = 2.545 :=
by
  sorry

end ball_hits_ground_time_l209_209993


namespace smallest_K_222_multiple_of_198_l209_209115

theorem smallest_K_222_multiple_of_198 :
  ∀ K : ℕ, (∃ x : ℕ, x = 2 * (10^K - 1) / 9 ∧ x % 198 = 0) → K = 18 :=
by
  sorry

end smallest_K_222_multiple_of_198_l209_209115


namespace axes_positioning_l209_209919

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem axes_positioning (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c < 0) :
  ∃ x_vertex y_intercept, x_vertex < 0 ∧ y_intercept < 0 ∧ (∀ x, f a b c x > f a b c x) :=
by
  sorry

end axes_positioning_l209_209919


namespace standard_deviations_below_mean_l209_209011

theorem standard_deviations_below_mean (μ σ x : ℝ) (hμ : μ = 14.5) (hσ : σ = 1.7) (hx : x = 11.1) :
    (μ - x) / σ = 2 := by
  sorry

end standard_deviations_below_mean_l209_209011


namespace pens_count_l209_209489

theorem pens_count (N P : ℕ) (h1 : N = 40) (h2 : P / N = 5 / 4) : P = 50 :=
by
  sorry

end pens_count_l209_209489


namespace rate_of_simple_interest_l209_209776

theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (P_nonzero : P ≠ 0) : 
  (P * R * T = P / 6) → R = 1 / 42 :=
by
  intro h
  sorry

end rate_of_simple_interest_l209_209776


namespace sum_of_roots_l209_209829

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 2 * x1^2 - k * x1 = 2 * c) 
  (h2 : 2 * x2^2 - k * x2 = 2 * c) (h3 : x1 ≠ x2) : x1 + x2 = k / 2 := 
sorry

end sum_of_roots_l209_209829


namespace people_in_first_group_l209_209951

theorem people_in_first_group (P : ℕ) (work_done_by_P : 60 = 1 / (P * (1/60))) (work_done_by_16 : 30 = 1 / (16 * (1/30))) : P = 8 :=
by
  sorry

end people_in_first_group_l209_209951


namespace find_A_minus_C_l209_209146

theorem find_A_minus_C (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 :=
by
  sorry

end find_A_minus_C_l209_209146


namespace correct_option_D_l209_209124

theorem correct_option_D (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2 * x :=
by sorry

end correct_option_D_l209_209124


namespace vectors_parallel_eq_l209_209090

-- Defining the problem
variables {m : ℝ}

-- Main statement
theorem vectors_parallel_eq (h : ∃ k : ℝ, (k ≠ 0) ∧ (k * 1 = m) ∧ (k * m = 2)) :
  m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
sorry

end vectors_parallel_eq_l209_209090


namespace eval_expression_l209_209903

theorem eval_expression :
  (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := 
by
  sorry

end eval_expression_l209_209903


namespace complement_of_M_l209_209495

-- Definitions:
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Assertion:
theorem complement_of_M :
  (U \ M) = {x | x ≤ -1} ∪ {x | 2 < x} :=
by sorry

end complement_of_M_l209_209495


namespace total_distance_traveled_l209_209061

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end total_distance_traveled_l209_209061


namespace balance_increase_second_year_l209_209590

variable (initial_deposit : ℝ) (balance_first_year : ℝ) 
variable (total_percentage_increase : ℝ)

theorem balance_increase_second_year
  (h1 : initial_deposit = 1000)
  (h2 : balance_first_year = 1100)
  (h3 : total_percentage_increase = 0.32) : 
  (balance_first_year + (initial_deposit * total_percentage_increase) - balance_first_year) / balance_first_year * 100 = 20 :=
by
  sorry

end balance_increase_second_year_l209_209590


namespace find_number_l209_209949

theorem find_number (x : ℝ) : (45 * x = 0.45 * 900) → (x = 9) :=
by sorry

end find_number_l209_209949


namespace radius_of_circle_eq_l209_209687

-- Define the given quadratic equation representing the circle
noncomputable def circle_eq (x y : ℝ) : ℝ :=
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68

-- State that the radius of the circle given by the equation is 1
theorem radius_of_circle_eq : ∃ r, (∀ x y, circle_eq x y = 0 ↔ (x - 1)^2 + (y - 1.5)^2 = r^2) ∧ r = 1 :=
by 
  use 1
  sorry

end radius_of_circle_eq_l209_209687


namespace ryan_correct_percentage_l209_209341

theorem ryan_correct_percentage :
  let problems1 := 25
  let correct1 := 0.8 * problems1
  let problems2 := 40
  let correct2 := 0.9 * problems2
  let problems3 := 10
  let correct3 := 0.7 * problems3
  let total_problems := problems1 + problems2 + problems3
  let total_correct := correct1 + correct2 + correct3
  (total_correct / total_problems) = 0.84 :=
by 
  sorry

end ryan_correct_percentage_l209_209341


namespace flour_already_put_in_l209_209988

theorem flour_already_put_in (total_flour flour_still_needed: ℕ) (h1: total_flour = 9) (h2: flour_still_needed = 6) : total_flour - flour_still_needed = 3 := 
by
  -- Here we will state the proof
  sorry

end flour_already_put_in_l209_209988


namespace part1_proof_part2_proof_part3_proof_l209_209453

-- Definitions and conditions for part 1
def P (a : ℤ) : ℤ × ℤ := (-3 * a - 4, 2 + a)
def part1_condition (a : ℤ) : Prop := (2 + a = 0)
def part1_answer : ℤ × ℤ := (2, 0)

-- Definitions and conditions for part 2
def Q : ℤ × ℤ := (5, 8)
def part2_condition (a : ℤ) : Prop := (-3 * a - 4 = 5)
def part2_answer : ℤ × ℤ := (5, -1)

-- Definitions and conditions for part 3
def part3_condition (a : ℤ) : Prop := 
  (-3 * a - 4 + 2 + a = 0) ∧ (-3 * a - 4 < 0 ∧ 2 + a > 0) -- Second quadrant
def part3_answer (a : ℤ) : ℤ := (a ^ 2023 + 2023)

-- Lean statements for proofs

theorem part1_proof (a : ℤ) (h : part1_condition a) : P a = part1_answer :=
by sorry

theorem part2_proof (a : ℤ) (h : part2_condition a) : P a = part2_answer :=
by sorry

theorem part3_proof (a : ℤ) (h : part3_condition a) : part3_answer a = 2022 :=
by sorry

end part1_proof_part2_proof_part3_proof_l209_209453


namespace wash_time_difference_l209_209523

def C := 30
def T := 2 * C
def total_time := 135

theorem wash_time_difference :
  ∃ S, C + T + S = total_time ∧ T - S = 15 :=
by
  sorry

end wash_time_difference_l209_209523


namespace diminished_radius_10_percent_l209_209245

theorem diminished_radius_10_percent
  (r r' : ℝ) 
  (h₁ : r > 0)
  (h₂ : r' > 0)
  (h₃ : (π * r'^2) = 0.8100000000000001 * (π * r^2)) :
  r' = 0.9 * r :=
by sorry

end diminished_radius_10_percent_l209_209245


namespace triangle_equilateral_of_angle_and_side_sequences_l209_209792

theorem triangle_equilateral_of_angle_and_side_sequences 
  (A B C : ℝ) (a b c : ℝ) 
  (h_angles_arith_seq: B = (A + C) / 2)
  (h_sides_geom_seq : b^2 = a * c) 
  (h_sum_angles : A + B + C = 180) 
  (h_pos_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_of_angle_and_side_sequences_l209_209792


namespace min_path_length_l209_209001

noncomputable def problem_statement : Prop :=
  let XY := 12
  let XZ := 8
  let angle_XYZ := 30
  let YP_PQ_QZ := by {
    -- Reflect Z across XY to get Z' and Y across XZ to get Y'.
    -- Use the Law of cosines in triangle XY'Z'.
    let cos_150 := -Real.sqrt 3 / 2
    let Y_prime_Z_prime := Real.sqrt (8^2 + 12^2 + 2 * 8 * 12 * cos_150)
    exact Y_prime_Z_prime
  }
  ∃ (P Q : Type), (YP_PQ_QZ = Real.sqrt (208 + 96 * Real.sqrt 3))

-- Goal is to prove the problem statement
theorem min_path_length : problem_statement := sorry

end min_path_length_l209_209001


namespace find_a_of_normal_vector_l209_209733

theorem find_a_of_normal_vector (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y + 5 = 0) ∧ (∃ n : ℝ × ℝ, n = (a, a - 2)) → a = 6 := by
  sorry

end find_a_of_normal_vector_l209_209733


namespace find_a_l209_209546

variable (a x y : ℝ)

theorem find_a (h1 : x / (2 * y) = 3 / 2) (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) : a = 7 :=
sorry

end find_a_l209_209546


namespace roots_of_polynomial_l209_209706

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^2 - 5*x + 6)*(x)*(x-5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 :=
by
  sorry

end roots_of_polynomial_l209_209706


namespace moles_of_water_produced_l209_209260

-- Definitions for the chemical reaction
def moles_NaOH := 4
def moles_H₂SO₄ := 2

-- The balanced chemical equation tells us the ratio of NaOH to H₂O
def chemical_equation (moles_NaOH moles_H₂SO₄ moles_H₂O moles_Na₂SO₄: ℕ) : Prop :=
  2 * moles_NaOH = 2 * moles_H₂O ∧ moles_H₂SO₄ = 1 ∧ moles_Na₂SO₄ = 1

-- The actual proof statement
theorem moles_of_water_produced : 
  ∀ (m_NaOH m_H₂SO₄ m_Na₂SO₄ : ℕ), 
  chemical_equation m_NaOH m_H₂SO₄ 4 m_Na₂SO₄ → moles_H₂O = 4 :=
by
  intros m_NaOH m_H₂SO₄ m_Na₂SO₄ chem_eq
  -- Placeholder for the actual proof.
  sorry

end moles_of_water_produced_l209_209260


namespace find_f_values_l209_209965

noncomputable def f : ℕ → ℕ := sorry

axiom condition1 : ∀ (a b : ℕ), a ≠ b → (a * f a + b * f b > a * f b + b * f a)
axiom condition2 : ∀ (n : ℕ), f (f n) = 3 * n

theorem find_f_values : f 1 + f 6 + f 28 = 66 := 
by
  sorry

end find_f_values_l209_209965


namespace total_birds_l209_209283

theorem total_birds (g d : Nat) (h₁ : g = 58) (h₂ : d = 37) : g + d = 95 :=
by
  sorry

end total_birds_l209_209283


namespace total_percentage_failed_exam_l209_209436

theorem total_percentage_failed_exam :
  let total_candidates := 2000
  let general_candidates := 1000
  let obc_candidates := 600
  let sc_candidates := 300
  let st_candidates := total_candidates - (general_candidates + obc_candidates + sc_candidates)
  let general_pass_percentage := 0.35
  let obc_pass_percentage := 0.50
  let sc_pass_percentage := 0.25
  let st_pass_percentage := 0.30
  let general_failed := general_candidates - (general_candidates * general_pass_percentage)
  let obc_failed := obc_candidates - (obc_candidates * obc_pass_percentage)
  let sc_failed := sc_candidates - (sc_candidates * sc_pass_percentage)
  let st_failed := st_candidates - (st_candidates * st_pass_percentage)
  let total_failed := general_failed + obc_failed + sc_failed + st_failed
  let failed_percentage := (total_failed / total_candidates) * 100
  failed_percentage = 62.25 :=
by
  sorry

end total_percentage_failed_exam_l209_209436


namespace problem_1_problem_2_l209_209668

section Problem1

variable (x a : ℝ)

-- Proposition p
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 1
theorem problem_1 : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by { sorry }

end Problem1

section Problem2

variable (a : ℝ)

-- Proposition p with a as a variable
def p_a (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q with x as a variable
def q_x (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 2
theorem problem_2 : (∀ (x : ℝ), ¬p_a a x → ¬q_x x) → (1 < a ∧ a ≤ 2) :=
by { sorry }

end Problem2

end problem_1_problem_2_l209_209668


namespace systematic_sampling_sequence_l209_209042

theorem systematic_sampling_sequence :
  ∃ k : ℕ, ∃ b : ℕ, (∀ n : ℕ, n < 6 → (3 + n * k = b + n * 10)) ∧ (b = 3 ∨ b = 13 ∨ b = 23 ∨ b = 33 ∨ b = 43 ∨ b = 53) :=
sorry

end systematic_sampling_sequence_l209_209042


namespace area_triangle_l209_209019

noncomputable def area_of_triangle_ABC (AB BC : ℝ) : ℝ := 
    (1 / 2) * AB * BC 

theorem area_triangle (AC : ℝ) (h1 : AC = 40)
    (h2 : ∃ B C : ℝ, B = (1/2) * AC ∧ C = B * Real.sqrt 3) :
    area_of_triangle_ABC ((1 / 2) * AC) (((1 / 2) * AC) * Real.sqrt 3) = 200 * Real.sqrt 3 := 
sorry

end area_triangle_l209_209019


namespace sum_of_a_and_b_l209_209014

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l209_209014


namespace intersection_A_B_l209_209408

def set_A (x : ℝ) : Prop := (x + 1 / 2 ≥ 3 / 2) ∨ (x + 1 / 2 ≤ -3 / 2)
def set_B (x : ℝ) : Prop := x^2 + x < 6
def A_cap_B := { x : ℝ | set_A x ∧ set_B x }

theorem intersection_A_B : A_cap_B = { x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2) } :=
sorry

end intersection_A_B_l209_209408


namespace sum_of_roots_of_cubic_eq_l209_209005

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 72 * x + 6

-- Define the statement to prove
theorem sum_of_roots_of_cubic_eq : 
  ∀ (r p q : ℝ), (cubic_eq r = 0) ∧ (cubic_eq p = 0) ∧ (cubic_eq q = 0) → 
  (r + p + q) = 3 :=
sorry

end sum_of_roots_of_cubic_eq_l209_209005


namespace remainder_when_divided_by_29_l209_209738

theorem remainder_when_divided_by_29 (N : ℤ) (k : ℤ) (h : N = 751 * k + 53) : 
  N % 29 = 24 := 
by 
  sorry

end remainder_when_divided_by_29_l209_209738


namespace inscribed_circle_radius_in_sector_l209_209924

theorem inscribed_circle_radius_in_sector
  (radius : ℝ)
  (sector_fraction : ℝ)
  (r : ℝ) :
  radius = 4 →
  sector_fraction = 1/3 →
  r = 2 * Real.sqrt 3 - 2 →
  true := by
sorry

end inscribed_circle_radius_in_sector_l209_209924


namespace part1_monotonic_intervals_part2_range_of_a_l209_209614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x + 0.5

theorem part1_monotonic_intervals (x : ℝ) : 
  (f 1 x < (f 1 (x + 1)) ↔ x < 1) ∧ 
  (f 1 x > (f 1 (x - 1)) ↔ x > 1) :=
by sorry

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 1 < x ∧ x ≤ Real.exp 1) 
  (h : (f a x / x) + (1 / (2 * x)) < 0) : 
  a < 1 - (1 / Real.exp 1) :=
by sorry

end part1_monotonic_intervals_part2_range_of_a_l209_209614


namespace find_lengths_of_segments_l209_209968

variable (b c : ℝ)

theorem find_lengths_of_segments (CK AK AB CT AC AT : ℝ)
  (h1 : CK = AK + AB)
  (h2 : CK = (b + c) / 2)
  (h3 : CT = AC - AT)
  (h4 : AC = b) :
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := 
sorry

end find_lengths_of_segments_l209_209968


namespace chickens_problem_l209_209900

theorem chickens_problem 
    (john_took_more_mary : ∀ (john mary : ℕ), john = mary + 5)
    (ray_took : ℕ := 10)
    (john_took_more_ray : ∀ (john ray : ℕ), john = ray + 11) :
    ∃ mary : ℕ, ray = mary - 6 :=
by
    sorry

end chickens_problem_l209_209900


namespace wire_pieces_difference_l209_209454

theorem wire_pieces_difference (L1 L2 : ℝ) (H1 : L1 = 14) (H2 : L2 = 16) : L2 - L1 = 2 :=
by
  rw [H1, H2]
  norm_num

end wire_pieces_difference_l209_209454


namespace cookies_recipes_count_l209_209973

theorem cookies_recipes_count 
  (total_students : ℕ)
  (attending_percentage : ℚ)
  (cookies_per_student : ℕ)
  (cookies_per_batch : ℕ) : 
  (total_students = 150) →
  (attending_percentage = 0.60) →
  (cookies_per_student = 3) →
  (cookies_per_batch = 18) →
  (total_students * attending_percentage * cookies_per_student / cookies_per_batch = 15) :=
by
  intros h1 h2 h3 h4
  sorry

end cookies_recipes_count_l209_209973


namespace intersection_M_N_l209_209536

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | x > 0 ∧ x < 2}

theorem intersection_M_N : M ∩ N = {1} :=
by {
  sorry
}

end intersection_M_N_l209_209536


namespace parity_of_pq_l209_209691

theorem parity_of_pq (x y m n p q : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 0)
    (hx : x = p) (hy : y = q) (h1 : x - 1998 * y = n) (h2 : 1999 * x + 3 * y = m) :
    p % 2 = 0 ∧ q % 2 = 1 :=
by
  sorry

end parity_of_pq_l209_209691


namespace hunter_saw_32_frogs_l209_209529

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l209_209529


namespace values_of_m_and_n_l209_209196

theorem values_of_m_and_n (m n : ℕ) (h_cond1 : 2 * m + 3 = 5 * n - 2) (h_cond2 : 5 * n - 2 < 15) : m = 5 ∧ n = 3 :=
by
  sorry

end values_of_m_and_n_l209_209196


namespace distance_between_intersections_l209_209622

-- Given conditions
def line_eq (x : ℝ) : ℝ := 5
def quad_eq (x : ℝ) : ℝ := 5 * x^2 + 2 * x - 2

-- The proof statement
theorem distance_between_intersections : 
  ∃ (C D : ℝ), line_eq C = quad_eq C ∧ line_eq D = quad_eq D ∧ abs (C - D) = 2.4 :=
by
  -- We will later fill in the proof here
  sorry

end distance_between_intersections_l209_209622


namespace mark_money_l209_209856

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end mark_money_l209_209856


namespace num_divisors_630_l209_209700

theorem num_divisors_630 : ∃ d : ℕ, (d = 24) ∧ ∀ n : ℕ, (∃ (a b c d : ℕ), (n = 2^a * 3^b * 5^c * 7^d) ∧ a ≤ 1 ∧ b ≤ 2 ∧ c ≤ 1 ∧ d ≤ 1) ↔ (n ∣ 630) := sorry

end num_divisors_630_l209_209700


namespace elder_person_age_l209_209512

-- Definitions based on conditions
variables (y e : ℕ) 

-- Given conditions
def condition1 : Prop := e = y + 20
def condition2 : Prop := e - 5 = 5 * (y - 5)

-- Theorem stating the required proof problem
theorem elder_person_age (h1 : condition1 y e) (h2 : condition2 y e) : e = 30 :=
by
  sorry

end elder_person_age_l209_209512


namespace calculate_expression_l209_209070

theorem calculate_expression : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end calculate_expression_l209_209070


namespace problem_l209_209652

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end problem_l209_209652


namespace fraction_division_l209_209628

theorem fraction_division :
  (5 / 4) / (8 / 15) = 75 / 32 :=
sorry

end fraction_division_l209_209628


namespace barbara_candies_l209_209337

theorem barbara_candies :
  ∀ (initial left used : ℝ), initial = 18 ∧ left = 9 → initial - left = used → used = 9 :=
by
  intros initial left used h1 h2
  sorry

end barbara_candies_l209_209337


namespace number_of_candidates_l209_209158

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
sorry

end number_of_candidates_l209_209158


namespace find_average_income_of_M_and_O_l209_209946

def average_income_of_M_and_O (M N O : ℕ) : Prop :=
  M + N = 10100 ∧
  N + O = 12500 ∧
  M = 4000 ∧
  (M + O) / 2 = 5200

theorem find_average_income_of_M_and_O (M N O : ℕ):
  average_income_of_M_and_O M N O → 
  (M + O) / 2 = 5200 :=
by
  intro h
  exact h.2.2.2

end find_average_income_of_M_and_O_l209_209946


namespace invalid_votes_percentage_l209_209520

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_candidate2 : ℕ) (valid_votes_percentage_candidate1 : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_valid_votes_candidate2 : valid_votes_candidate2 = 2700)
  (h_valid_votes_percentage_candidate1 : valid_votes_percentage_candidate1 = 55) :
  ((total_votes - (valid_votes_candidate2 * 100 / (100 - valid_votes_percentage_candidate1))) * 100 / total_votes) = 20 :=
by sorry

end invalid_votes_percentage_l209_209520


namespace sine_double_angle_l209_209850

theorem sine_double_angle (theta : ℝ) (h : Real.tan (theta + Real.pi / 4) = 2) : Real.sin (2 * theta) = 3 / 5 :=
sorry

end sine_double_angle_l209_209850


namespace sequence_properties_l209_209169

-- Define the arithmetic-geometric sequence and its sum
def a_n (n : ℕ) : ℕ := 2^(n-1)
def S_n (n : ℕ) : ℕ := 2^n - 1
def T_n (n : ℕ) : ℕ := 2^(n+1) - n - 2

theorem sequence_properties : 
(S_n 3 = 7) ∧ (S_n 6 = 63) → 
(∀ n: ℕ, a_n n = 2^(n-1)) ∧ 
(∀ n: ℕ, S_n n = 2^n - 1) ∧ 
(∀ n: ℕ, T_n n = 2^(n+1) - n - 2) :=
by
  sorry

end sequence_properties_l209_209169


namespace tile_coverage_fraction_l209_209134

structure Room where
  rect_length : ℝ
  rect_width : ℝ
  tri_base : ℝ
  tri_height : ℝ
  
structure Tiles where
  square_tiles : ℕ
  triangular_tiles : ℕ
  triangle_base : ℝ
  triangle_height : ℝ
  tile_area : ℝ
  triangular_tile_area : ℝ
  
noncomputable def fractionalTileCoverage (room : Room) (tiles : Tiles) : ℝ :=
  let rect_area := room.rect_length * room.rect_width
  let tri_area := (room.tri_base * room.tri_height) / 2
  let total_room_area := rect_area + tri_area
  let total_tile_area := (tiles.square_tiles * tiles.tile_area) + (tiles.triangular_tiles * tiles.triangular_tile_area)
  total_tile_area / total_room_area

theorem tile_coverage_fraction
  (room : Room) (tiles : Tiles)
  (h1 : room.rect_length = 12)
  (h2 : room.rect_width = 20)
  (h3 : room.tri_base = 10)
  (h4 : room.tri_height = 8)
  (h5 : tiles.square_tiles = 40)
  (h6 : tiles.triangular_tiles = 4)
  (h7 : tiles.tile_area = 1)
  (h8 : tiles.triangular_tile_area = (1 * 1) / 2) :
  fractionalTileCoverage room tiles = 3 / 20 :=
by 
  sorry

end tile_coverage_fraction_l209_209134


namespace exists_nat_numbers_except_two_three_l209_209037

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end exists_nat_numbers_except_two_three_l209_209037


namespace probability_at_least_one_8_l209_209990

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end probability_at_least_one_8_l209_209990


namespace det_A_is_2_l209_209284

-- Define the matrix A
def A (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 2], ![-3, d]]

-- Define the inverse of matrix A 
noncomputable def A_inv (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / (a * d + 6)) • ![![d, -2], ![3, a]]

-- Condition: A + A_inv = 0
def condition (a d : ℝ) : Prop := A a d + A_inv a d = 0

-- Main theorem: determinant of A under the given condition
theorem det_A_is_2 (a d : ℝ) (h : condition a d) : Matrix.det (A a d) = 2 :=
by sorry

end det_A_is_2_l209_209284


namespace math_problem_l209_209503

theorem math_problem 
  (m n : ℕ) 
  (h1 : (m^2 - n) ∣ (m + n^2))
  (h2 : (n^2 - m) ∣ (m^2 + n)) : 
  (m, n) = (2, 2) ∨ (m, n) = (3, 3) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 1) ∨ (m, n) = (2, 3) ∨ (m, n) = (3, 2) := 
sorry

end math_problem_l209_209503


namespace percentage_increase_equiv_l209_209333

theorem percentage_increase_equiv {P : ℝ} : 
  (P * (1 + 0.08) * (1 + 0.08)) = (P * 1.1664) :=
by
  sorry

end percentage_increase_equiv_l209_209333


namespace find_s_l209_209493

theorem find_s (s : ℝ) (m : ℤ) (d : ℝ) (h_floor : ⌊s⌋ = m) (h_decompose : s = m + d) (h_fractional : 0 ≤ d ∧ d < 1) (h_equation : ⌊s⌋ - s = -10.3) : s = -9.7 :=
by
  sorry

end find_s_l209_209493


namespace quadratic_has_two_distinct_real_roots_l209_209349

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 + 2 * k * r1 + (k - 1) = 0 ∧ r2^2 + 2 * k * r2 + (k - 1) = 0 := 
by 
  sorry

end quadratic_has_two_distinct_real_roots_l209_209349


namespace intersection_M_N_l209_209561

def M : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}
def intersection : Set ℝ := {-1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l209_209561


namespace functional_equation_solution_l209_209735

noncomputable def f : ℚ → ℚ := sorry

theorem functional_equation_solution :
  (∀ x y : ℚ, f (f x + x * f y) = x + f x * y) →
  (∀ x : ℚ, f x = x) :=
by
  intro h
  sorry

end functional_equation_solution_l209_209735


namespace vectors_parallel_l209_209761

-- Let s and n be the direction vector and normal vector respectively
def s : ℝ × ℝ × ℝ := (2, 1, 1)
def n : ℝ × ℝ × ℝ := (-4, -2, -2)

-- Statement that vectors s and n are parallel
theorem vectors_parallel : ∃ (k : ℝ), n = (k • s) := by
  use -2
  simp [s, n]
  sorry

end vectors_parallel_l209_209761


namespace quadratic_no_real_roots_l209_209958

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_no_real_roots
  (a b c: ℝ)
  (h1: ((b - 1)^2 - 4 * a * (c + 1) = 0))
  (h2: ((b + 2)^2 - 4 * a * (c - 2) = 0)) :
  ∀ x : ℝ, f a b c x ≠ 0 := 
sorry

end quadratic_no_real_roots_l209_209958


namespace ratio_of_blue_to_purple_beads_l209_209779

theorem ratio_of_blue_to_purple_beads :
  ∃ (B G : ℕ), 
    7 + B + G = 46 ∧ 
    G = B + 11 ∧ 
    B / 7 = 2 :=
by
  sorry

end ratio_of_blue_to_purple_beads_l209_209779


namespace min_troublemakers_in_class_l209_209298

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l209_209298


namespace gcd_of_g_and_y_l209_209555

noncomputable def g (y : ℕ) := (3 * y + 5) * (8 * y + 3) * (16 * y + 9) * (y + 16)

theorem gcd_of_g_and_y (y : ℕ) (hy : y % 46896 = 0) : Nat.gcd (g y) y = 2160 :=
by
  -- Proof to be written here
  sorry

end gcd_of_g_and_y_l209_209555


namespace combination_square_octagon_tiles_l209_209032

-- Define the internal angles of the polygons
def internal_angle (shape : String) : Float :=
  match shape with
  | "Square"   => 90.0
  | "Pentagon" => 108.0
  | "Hexagon"  => 120.0
  | "Octagon"  => 135.0
  | _          => 0.0

-- Define the condition for the combination of two regular polygons to tile seamlessly
def can_tile (shape1 shape2 : String) : Bool :=
  let angle1 := internal_angle shape1
  let angle2 := internal_angle shape2
  angle1 + 2 * angle2 == 360.0

-- Define the tiling problem
theorem combination_square_octagon_tiles : can_tile "Square" "Octagon" = true :=
by {
  -- The proof of this theorem should show that Square and Octagon can indeed tile seamlessly
  sorry
}

end combination_square_octagon_tiles_l209_209032


namespace sum_geometric_sequence_first_eight_terms_l209_209690

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l209_209690


namespace football_goal_average_increase_l209_209327

theorem football_goal_average_increase :
  ∀ (A : ℝ), 4 * A + 2 = 8 → (8 / 5) - A = 0.1 :=
by
  intro A
  intro h
  sorry -- Proof to be filled in

end football_goal_average_increase_l209_209327


namespace equilateral_triangle_l209_209971

theorem equilateral_triangle (a b c : ℝ) (h1 : a^4 = b^4 + c^4 - b^2 * c^2) (h2 : b^4 = a^4 + c^4 - a^2 * c^2) : 
  a = b ∧ b = c ∧ c = a :=
by sorry

end equilateral_triangle_l209_209971


namespace least_number_to_subtract_l209_209021

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 42398) (h2 : d = 15) (h3 : r = 8) : 
  ∃ k, n - r = k * d :=
by
  sorry

end least_number_to_subtract_l209_209021


namespace union_sets_eq_l209_209483

-- Definitions of the given sets
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

-- The theorem to prove the union of sets A and B equals \{0, 1, 2\}
theorem union_sets_eq : (A ∪ B) = {0, 1, 2} := by
  sorry

end union_sets_eq_l209_209483


namespace max_abs_x_y_l209_209056

theorem max_abs_x_y (x y : ℝ) (h : 4 * x^2 + y^2 = 4) : |x| + |y| ≤ 2 :=
by sorry

end max_abs_x_y_l209_209056


namespace total_pears_picked_is_correct_l209_209586

-- Define the number of pears picked by Sara and Sally
def pears_picked_by_Sara : ℕ := 45
def pears_picked_by_Sally : ℕ := 11

-- The total number of pears picked
def total_pears_picked := pears_picked_by_Sara + pears_picked_by_Sally

-- The theorem statement: prove that the total number of pears picked is 56
theorem total_pears_picked_is_correct : total_pears_picked = 56 := by
  sorry

end total_pears_picked_is_correct_l209_209586


namespace triangle_area_proof_l209_209023

noncomputable def area_of_triangle_ABC : ℝ :=
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  let area := 3 / 11
  area

theorem triangle_area_proof :
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  area_of_triangle_ABC = 3 / 11 :=
by
  sorry

end triangle_area_proof_l209_209023


namespace find_x_plus_y_l209_209353

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 7 * y = 17) (h2 : 3 * x + 5 * y = 11) : x + y = 83 / 23 :=
sorry

end find_x_plus_y_l209_209353


namespace range_of_m_solve_inequality_l209_209853

open Real Set

noncomputable def f (x: ℝ) := -abs (x - 2)
noncomputable def g (x: ℝ) (m: ℝ) := -abs (x - 3) + m

-- Problem 1: Prove the range of m given the condition
theorem range_of_m (h : ∀ x : ℝ, f x > g x m) : m < 1 :=
  sorry

-- Problem 2: Prove the set of solutions for f(x) + a - 1 > 0
theorem solve_inequality (a : ℝ) :
  (if a = 1 then {x : ℝ | x ≠ 2}
   else if a > 1 then univ
   else {x : ℝ | x < 1 + a} ∪ {x : ℝ | x > 3 - a}) = {x : ℝ | f x + a - 1 > 0} :=
  sorry

end range_of_m_solve_inequality_l209_209853


namespace count_marble_pairs_l209_209703

-- Define conditions:
structure Marbles :=
(red : ℕ) (green : ℕ) (blue : ℕ) (yellow : ℕ) (white : ℕ)

def tomsMarbles : Marbles :=
  { red := 1, green := 1, blue := 1, yellow := 3, white := 2 }

-- Define a function to count pairs of marbles:
def count_pairs (m : Marbles) : ℕ :=
  -- Count pairs of identical marbles:
  (if m.yellow >= 2 then 1 else 0) + 
  (if m.white >= 2 then 1 else 0) +
  -- Count pairs of different colored marbles:
  (Nat.choose 5 2)

-- Theorem statement:
theorem count_marble_pairs : count_pairs tomsMarbles = 12 :=
  by
    sorry

end count_marble_pairs_l209_209703


namespace second_less_than_first_l209_209264

-- Define the given conditions
def third_number : ℝ := sorry
def first_number : ℝ := 0.65 * third_number
def second_number : ℝ := 0.58 * third_number

-- Problem statement: Prove that the second number is approximately 10.77% less than the first number
theorem second_less_than_first : 
  (first_number - second_number) / first_number * 100 = 10.77 := 
sorry

end second_less_than_first_l209_209264


namespace completing_the_square_l209_209705

theorem completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 ↔ (x - 2)^2 = 6 :=
by
  sorry

end completing_the_square_l209_209705


namespace day_in_43_days_is_wednesday_l209_209069

-- Define a function to represent the day of the week after a certain number of days
def day_of_week (n : ℕ) : ℕ := n % 7

-- Use an enum or some notation to represent the days of the week, but this is implicit in our setup.
-- We assume the days are numbered from 0 to 6 with 0 representing Tuesday.
def Tuesday : ℕ := 0
def Wednesday : ℕ := 1

-- Theorem to prove that 43 days after Tuesday is a Wednesday
theorem day_in_43_days_is_wednesday : day_of_week (Tuesday + 43) = Wednesday :=
by
  sorry

end day_in_43_days_is_wednesday_l209_209069


namespace f_strictly_decreasing_intervals_f_max_min_on_interval_l209_209621

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 6 * x^2 - 9 * x + 3

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := -3 * x^2 - 12 * x - 9

-- Statement for part (I)
theorem f_strictly_decreasing_intervals :
  (∀ x : ℝ, x < -3 → f_deriv x < 0) ∧ (∀ x : ℝ, x > -1 → f_deriv x < 0) := by
  sorry

-- Statement for part (II)
theorem f_max_min_on_interval :
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≥ -47) :=
  sorry

end f_strictly_decreasing_intervals_f_max_min_on_interval_l209_209621


namespace largest_sum_is_1173_l209_209826

def largest_sum_of_two_3digit_numbers : Prop :=
  ∃ a b c d e f : ℕ, 
  (a = 6 ∧ b = 5 ∧ c = 4 ∧ d = 3 ∧ e = 2 ∧ f = 1) ∧
  100 * (a + b) + 10 * (c + d) + (e + f) = 1173

theorem largest_sum_is_1173 : largest_sum_of_two_3digit_numbers :=
  by
  sorry

end largest_sum_is_1173_l209_209826


namespace sequences_properties_l209_209739

-- Definition of sequences and their properties
variable {n : ℕ}

noncomputable def S (n : ℕ) : ℕ := n^2 - n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else 2 * n - 2
noncomputable def b (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c (n : ℕ) : ℕ := (2 * (n - 1)) / 3^(n - 1)
noncomputable def T (n : ℕ) : ℕ := 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))

-- Main theorem
theorem sequences_properties (n : ℕ) (hn : n > 0) :
  S n = n^2 - n ∧
  (∀ n, a n = if n = 1 then 0 else 2 * n - 2) ∧
  (∀ n, b n = 3^(n-1)) ∧
  (∀ n, T n = 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))) :=
by sorry

end sequences_properties_l209_209739


namespace smallest_3a_plus_1_l209_209989

theorem smallest_3a_plus_1 (a : ℝ) (h : 8 * a ^ 2 + 6 * a + 2 = 4) : 
  ∃ a, (8 * a ^ 2 + 6 * a + 2 = 4) ∧ min (3 * (-1) + 1) (3 * (1 / 4) + 1) = -2 :=
by {
  sorry
}

end smallest_3a_plus_1_l209_209989


namespace preimage_exists_l209_209167

-- Define the mapping function f
def f (x y : ℚ) : ℚ × ℚ :=
  (x + 2 * y, 2 * x - y)

-- Define the statement
theorem preimage_exists (x y : ℚ) :
  f x y = (3, 1) → (x, y) = (-1/3, 5/3) :=
by
  sorry

end preimage_exists_l209_209167


namespace wrenches_in_comparison_group_l209_209038

theorem wrenches_in_comparison_group (H W : ℝ) (x : ℕ) 
  (h1 : W = 2 * H)
  (h2 : 2 * H + 2 * W = (1 / 3) * (8 * H + x * W)) : x = 5 :=
by
  sorry

end wrenches_in_comparison_group_l209_209038


namespace sum_of_squares_of_two_numbers_l209_209593

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) :
  x^2 + y^2 = 840 :=
by
  sorry

end sum_of_squares_of_two_numbers_l209_209593


namespace regular_polygon_with_12_degree_exterior_angle_has_30_sides_l209_209752

def regular_polygon_sides (e : ℤ) : ℤ :=
  360 / e

theorem regular_polygon_with_12_degree_exterior_angle_has_30_sides :
  regular_polygon_sides 12 = 30 :=
by
  -- Proof is omitted
  sorry

end regular_polygon_with_12_degree_exterior_angle_has_30_sides_l209_209752


namespace polynomial_difference_l209_209563

theorem polynomial_difference (a : ℝ) :
  (6 * a^2 - 5 * a + 3) - (5 * a^2 + 2 * a - 1) = a^2 - 7 * a + 4 :=
by
  sorry

end polynomial_difference_l209_209563


namespace find_A_l209_209567

theorem find_A (A B : ℕ) (h1 : 15 = 3 * A) (h2 : 15 = 5 * B) : A = 5 := 
by 
  sorry

end find_A_l209_209567


namespace denis_neighbors_l209_209876

-- Define positions
inductive Position
| P1 | P2 | P3 | P4 | P5

open Position

-- Declare the children
inductive Child
| Anya | Borya | Vera | Gena | Denis

open Child

def next_to (p1 p2 : Position) : Prop := 
  (p1 = P1 ∧ p2 = P2) ∨ (p1 = P2 ∧ p2 = P1) ∨
  (p1 = P2 ∧ p2 = P3) ∨ (p1 = P3 ∧ p2 = P2) ∨
  (p1 = P3 ∧ p2 = P4) ∨ (p1 = P4 ∧ p2 = P3) ∨
  (p1 = P4 ∧ p2 = P5) ∨ (p1 = P5 ∧ p2 = P4)

variables (pos : Child → Position)

-- Given conditions
axiom borya_beginning : pos Borya = P1
axiom vera_next_to_anya : next_to (pos Vera) (pos Anya)
axiom vera_not_next_to_gena : ¬ next_to (pos Vera) (pos Gena)
axiom no_two_next_to : ∀ (c1 c2 : Child), 
  c1 ∈ [Anya, Borya, Gena] → c2 ∈ [Anya, Borya, Gena] → c1 ≠ c2 → ¬ next_to (pos c1) (pos c2)

-- Prove the result
theorem denis_neighbors : next_to (pos Denis) (pos Anya) ∧ next_to (pos Denis) (pos Gena) :=
sorry

end denis_neighbors_l209_209876


namespace arithmetic_sequence_problem_l209_209268

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)
  (h1 : a 6 + a 9 = 16)
  (h2 : a 4 = 1)
  (h_arith : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) :
  a 11 = 15 :=
by
  sorry

end arithmetic_sequence_problem_l209_209268


namespace car_A_speed_l209_209726

theorem car_A_speed (s_A s_B : ℝ) (d_AB d_extra t : ℝ) (h_s_B : s_B = 50) (h_d_AB : d_AB = 40) (h_d_extra : d_extra = 8) (h_time : t = 6) 
(h_distance_traveled_by_car_B : s_B * t = 300) 
(h_distance_difference : d_AB + d_extra = 48) :
  s_A = 58 :=
by
  sorry

end car_A_speed_l209_209726


namespace good_jars_l209_209359

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end good_jars_l209_209359


namespace total_amount_received_l209_209449

theorem total_amount_received (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ) (A : ℝ) 
  (hCI : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (hCI_value : CI = 370.80)
  (hr : r = 0.06)
  (hn : n = 1)
  (ht : t = 2)
  (hP : P = 3000)
  (hP_value : P = CI / 0.1236) :
  A = P + CI := 
by 
sorry

end total_amount_received_l209_209449


namespace min_value_of_Box_l209_209117

theorem min_value_of_Box (c d : ℤ) (hcd : c * d = 42) (distinct_values : c ≠ d ∧ c ≠ 85 ∧ d ≠ 85) :
  ∃ (Box : ℤ), (c^2 + d^2 = Box) ∧ (Box = 85) :=
by
  sorry

end min_value_of_Box_l209_209117


namespace inscribed_sphere_l209_209232

theorem inscribed_sphere (r_base height : ℝ) (r_sphere b d : ℝ)
  (h_base : r_base = 15)
  (h_height : height = 20)
  (h_sphere : r_sphere = b * Real.sqrt d - b)
  (h_rsphere_eq : r_sphere = 120 / 11) : 
  b + d = 12 := 
sorry

end inscribed_sphere_l209_209232


namespace large_hexagon_toothpicks_l209_209059

theorem large_hexagon_toothpicks (n : Nat) (h : n = 1001) : 
  let T_half := (n * (n + 1)) / 2
  let T_total := 2 * T_half + n
  let boundary_toothpicks := 6 * T_half
  let total_toothpicks := 3 * T_total - boundary_toothpicks
  total_toothpicks = 3006003 :=
by
  sorry

end large_hexagon_toothpicks_l209_209059


namespace rate_percent_simple_interest_l209_209296

theorem rate_percent_simple_interest (SI P T R : ℝ) (h₁ : SI = 500) (h₂ : P = 2000) (h₃ : T = 2)
  (h₄ : SI = (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for the proof
  sorry

end rate_percent_simple_interest_l209_209296


namespace polynomial_A_l209_209869

theorem polynomial_A (A a : ℝ) (h : A * (a + 1) = a^2 - 1) : A = a - 1 :=
sorry

end polynomial_A_l209_209869


namespace poll_total_l209_209415

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l209_209415


namespace seq_arithmetic_l209_209922

def seq (n : ℕ) : ℤ := 2 * n + 5

theorem seq_arithmetic :
  ∀ n : ℕ, seq (n + 1) - seq n = 2 :=
by
  intro n
  have h1 : seq (n + 1) = 2 * (n + 1) + 5 := rfl
  have h2 : seq n = 2 * n + 5 := rfl
  rw [h1, h2]
  linarith

end seq_arithmetic_l209_209922


namespace zero_ending_of_A_l209_209693

theorem zero_ending_of_A (A : ℕ) (h : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c ∣ A ∧ a + b + c = 8 → a * b * c = 10) : 
  (10 ∣ A) ∧ ¬(100 ∣ A) :=
by
  sorry

end zero_ending_of_A_l209_209693


namespace largest_four_digit_sum_23_l209_209188

theorem largest_four_digit_sum_23 : ∃ (n : ℕ), (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 23 ∧ 1000 ≤ n ∧ n < 10000) ∧ n = 9950 :=
  sorry

end largest_four_digit_sum_23_l209_209188


namespace fraction_pow_zero_is_one_l209_209774

theorem fraction_pow_zero_is_one (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 := by
  sorry

end fraction_pow_zero_is_one_l209_209774


namespace problem_l209_209708

-- Define the variable
variable (x : ℝ)

-- Define the condition
def condition := 3 * x - 1 = 8

-- Define the statement to be proven
theorem problem (h : condition x) : 150 * (1 / x) + 2 = 52 :=
  sorry

end problem_l209_209708


namespace john_sales_percentage_l209_209672

noncomputable def percentage_buyers (houses_visited_per_day : ℕ) (work_days_per_week : ℕ) (weekly_sales : ℝ) (low_price : ℝ) (high_price : ℝ) : ℝ :=
  let total_houses_per_week := houses_visited_per_day * work_days_per_week
  let average_sale_per_customer := (low_price + high_price) / 2
  let total_customers := weekly_sales / average_sale_per_customer
  (total_customers / total_houses_per_week) * 100

theorem john_sales_percentage :
  percentage_buyers 50 5 5000 50 150 = 20 := 
by 
  sorry

end john_sales_percentage_l209_209672


namespace cookies_in_jar_l209_209184

-- Let C be the total number of cookies in the jar.
def C : ℕ := sorry

-- Conditions
def adults_eat_one_third (C : ℕ) : ℕ := C / 3
def children_get_each (C : ℕ) : ℕ := 20
def num_children : ℕ := 4

-- Proof statement
theorem cookies_in_jar (C : ℕ) (h1 : C / 3 = adults_eat_one_third C)
  (h2 : children_get_each C * num_children = 80)
  (h3 : 2 * (C / 3) = 80) :
  C = 120 :=
sorry

end cookies_in_jar_l209_209184


namespace linda_spent_amount_l209_209791

theorem linda_spent_amount :
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  total_cost = 6.80 :=
by
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  show total_cost = 6.80
  sorry

end linda_spent_amount_l209_209791


namespace determine_divisors_l209_209650

theorem determine_divisors (n : ℕ) (h_pos : n > 0) (d : ℕ) (h_div : d ∣ 3 * n^2) (h_exists : ∃ k : ℤ, n^2 + d = k^2) : d = 3 * n^2 := 
sorry

end determine_divisors_l209_209650


namespace length_of_chord_AB_l209_209386

noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)
noncomputable def line_eq (x : ℝ) := x - Real.sqrt 3
noncomputable def ellipse_eq (x y : ℝ)  := x^2 / 4 + y^2 = 1

theorem length_of_chord_AB :
  ∀ (A B : ℝ × ℝ), 
  (line_eq A.1 = A.2) → 
  (line_eq B.1 = B.2) → 
  (ellipse_eq A.1 A.2) → 
  (ellipse_eq B.1 B.2) → 
  ∃ d : ℝ, d = 8 / 5 ∧ 
  dist A B = d := 
sorry

end length_of_chord_AB_l209_209386


namespace problem1_problem2_l209_209213

def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

-- Proof Problem 1
theorem problem1 (x : ℝ) (h : f x > 2) : x < -2 := sorry

-- Proof Problem 2
theorem problem2 (k : ℝ) (h : ∀ x : ℝ, -3 ≤ x ∧ x ≤ -1 → f x ≤ k * x + 1) : k ≤ -1 := sorry

end problem1_problem2_l209_209213


namespace integer_solutions_of_prime_equation_l209_209995

theorem integer_solutions_of_prime_equation (p : ℕ) (hp : Prime p) :
  ∃ x y : ℤ, (p * (x + y) = x * y) ↔ 
    (x = (p * (p + 1)) ∧ y = (p + 1)) ∨ 
    (x = 2 * p ∧ y = 2 * p) ∨ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p * (1 - p) ∧ y = (p - 1)) := 
sorry

end integer_solutions_of_prime_equation_l209_209995


namespace photos_per_album_l209_209828

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end photos_per_album_l209_209828


namespace min_value_of_fraction_l209_209522

theorem min_value_of_fraction (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 := 
sorry

end min_value_of_fraction_l209_209522


namespace product_xyz_l209_209957

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 5) : 
  x * y * z = 1 / 9 := 
by
  sorry

end product_xyz_l209_209957


namespace count_ball_box_arrangements_l209_209647

theorem count_ball_box_arrangements :
  ∃ (arrangements : ℕ), arrangements = 20 ∧
  (∃ f : Fin 5 → Fin 5,
    (∃! i1, f i1 = i1) ∧ (∃! i2, f i2 = i2) ∧
    ∀ i, ∃! j, f i = j) :=
sorry

end count_ball_box_arrangements_l209_209647


namespace trapezium_height_l209_209077

theorem trapezium_height (a b A h : ℝ) (ha : a = 12) (hb : b = 16) (ha_area : A = 196) :
  (A = 0.5 * (a + b) * h) → h = 14 :=
by
  intros h_eq
  rw [ha, hb, ha_area] at h_eq
  sorry

end trapezium_height_l209_209077


namespace minimum_value_expression_l209_209873

theorem minimum_value_expression (x y : ℝ) : 
  ∃ m : ℝ, ∀ x y : ℝ, 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ m ∧ m = 3 :=
sorry

end minimum_value_expression_l209_209873


namespace smaug_copper_coins_l209_209620

def copper_value_of_silver (silver_coins silver_to_copper : ℕ) : ℕ :=
  silver_coins * silver_to_copper

def copper_value_of_gold (gold_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  gold_coins * gold_to_silver * silver_to_copper

def total_copper_value (gold_coins silver_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  copper_value_of_gold gold_coins gold_to_silver silver_to_copper +
  copper_value_of_silver silver_coins silver_to_copper

def actual_copper_coins (total_value gold_value silver_value : ℕ) : ℕ :=
  total_value - (gold_value + silver_value)

theorem smaug_copper_coins :
  let gold_coins := 100
  let silver_coins := 60
  let silver_to_copper := 8
  let gold_to_silver := 3
  let total_copper_value := 2913
  let gold_value := copper_value_of_gold gold_coins gold_to_silver silver_to_copper
  let silver_value := copper_value_of_silver silver_coins silver_to_copper
  actual_copper_coins total_copper_value gold_value silver_value = 33 :=
by
  sorry

end smaug_copper_coins_l209_209620


namespace determine_coefficients_l209_209854

variable {α : Type} [Field α]
variables (a a1 a2 a3 : α)

theorem determine_coefficients (h : ∀ x : α, a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 = x^3) :
  a = 1 ∧ a2 = 3 :=
by
  -- To be proven
  sorry

end determine_coefficients_l209_209854


namespace circle_tangent_radius_l209_209772

-- Define the radii of the three given circles
def radius1 : ℝ := 1.0
def radius2 : ℝ := 2.0
def radius3 : ℝ := 3.0

-- Define the problem statement: finding the radius of the fourth circle externally tangent to the given three circles
theorem circle_tangent_radius (r1 r2 r3 : ℝ) (cond1 : r1 = 1) (cond2 : r2 = 2) (cond3 : r3 = 3) : 
  ∃ R : ℝ, R = 6 := by
  sorry

end circle_tangent_radius_l209_209772


namespace irrational_sum_root_l209_209098

theorem irrational_sum_root
  (α : ℝ) (hα : Irrational α)
  (n : ℕ) (hn : 0 < n) :
  Irrational ((α + (α^2 - 1).sqrt)^(1/n : ℝ) + (α - (α^2 - 1).sqrt)^(1/n : ℝ)) := sorry

end irrational_sum_root_l209_209098


namespace max_S_2017_l209_209954

noncomputable def max_S (a b c : ℕ) : ℕ := a + b + c

theorem max_S_2017 :
  ∀ (a b c : ℕ),
  a + b = 1014 →
  c - b = 497 →
  a > b →
  max_S a b c = 2017 :=
by
  intros a b c h1 h2 h3
  sorry

end max_S_2017_l209_209954


namespace vehicle_speeds_l209_209576

theorem vehicle_speeds (V_A V_B V_C : ℝ) (d_AB d_AC : ℝ) (decel_A : ℝ)
  (V_A_eff : ℝ) (delta_V_A : ℝ) :
  V_A = 70 → V_B = 50 → V_C = 65 →
  decel_A = 5 → V_A_eff = V_A - decel_A → 
  d_AB = 40 → d_AC = 250 →
  delta_V_A = 10 →
  (d_AB / (V_A_eff + delta_V_A - V_B) < d_AC / (V_A_eff + delta_V_A + V_C)) :=
by
  intros hVA hVB hVC hdecel hV_A_eff hdAB hdAC hdelta_V_A
  -- the proof would be filled in here
  sorry

end vehicle_speeds_l209_209576


namespace count_colorings_l209_209488

-- Define the number of disks
def num_disks : ℕ := 6

-- Define colorings with constraints: 2 black, 2 white, 2 blue considering rotations and reflections as equivalent
def valid_colorings : ℕ :=
  18  -- This is the result obtained using Burnside's Lemma as shown in the solution

theorem count_colorings : valid_colorings = 18 := by
  sorry

end count_colorings_l209_209488


namespace reflect_y_axis_correct_l209_209123

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end reflect_y_axis_correct_l209_209123


namespace average_gpa_difference_2_l209_209589

def avg_gpa_6th_grader := 93
def avg_gpa_8th_grader := 91
def school_avg_gpa := 93

noncomputable def gpa_diff (gpa_7th_grader diff : ℝ) (avg6 avg8 school_avg : ℝ) := 
  gpa_7th_grader = avg6 + diff ∧ 
  (avg6 + gpa_7th_grader + avg8) / 3 = school_avg

theorem average_gpa_difference_2 (x : ℝ) : 
  (∃ G : ℝ, gpa_diff G x avg_gpa_6th_grader avg_gpa_8th_grader school_avg_gpa) → x = 2 :=
by
  sorry

end average_gpa_difference_2_l209_209589


namespace simplified_evaluation_eq_half_l209_209640

theorem simplified_evaluation_eq_half :
  ∃ x y : ℝ, (|x - 2| + (y + 1)^2 = 0) → 
             (3 * x - 2 * (x^2 - (1/2) * y^2) + (x - (1/2) * y^2) = 1/2) :=
by
  sorry

end simplified_evaluation_eq_half_l209_209640


namespace volume_pyramid_l209_209136

theorem volume_pyramid (V : ℝ) : 
  ∃ V_P : ℝ, V_P = V / 6 :=
by
  sorry

end volume_pyramid_l209_209136


namespace find_particular_number_l209_209839

theorem find_particular_number (A B : ℤ) (x : ℤ) (hA : A = 14) (hB : B = 24)
  (h : (((A + x) * A - B) / B = 13)) : x = 10 :=
by {
  -- You can add an appropriate lemma or proof here if necessary
  sorry
}

end find_particular_number_l209_209839


namespace united_airlines_discount_l209_209940

theorem united_airlines_discount :
  ∀ (delta_price original_price_u discount_delta discount_u saved_amount cheapest_price: ℝ),
    delta_price = 850 →
    original_price_u = 1100 →
    discount_delta = 0.20 →
    saved_amount = 90 →
    cheapest_price = delta_price * (1 - discount_delta) - saved_amount →
    discount_u = (original_price_u - cheapest_price) / original_price_u →
    discount_u = 0.4636363636 :=
by
  intros delta_price original_price_u discount_delta discount_u saved_amount cheapest_price δeq ueq deq saeq cpeq dueq
  -- Placeholder for the actual proof steps
  sorry

end united_airlines_discount_l209_209940


namespace sqrt_sin_cos_expression_l209_209572

theorem sqrt_sin_cos_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = |Real.sin α - Real.sin β| :=
sorry

end sqrt_sin_cos_expression_l209_209572


namespace ice_cream_weekend_total_l209_209897

theorem ice_cream_weekend_total 
  (f : ℝ) (r : ℝ) (n : ℕ)
  (h_friday : f = 3.25)
  (h_saturday_reduction : r = 0.25)
  (h_num_people : n = 4)
  (h_saturday : (f - r * n) = 2.25)
  (h_sunday : 2 * ((f - r * n) / n) * n = 4.5) :
  f + (f - r * n) + (2 * ((f - r * n) / n) * n) = 10 := sorry

end ice_cream_weekend_total_l209_209897


namespace product_of_integers_P_Q_R_S_l209_209039

theorem product_of_integers_P_Q_R_S (P Q R S : ℤ)
  (h1 : 0 < P) (h2 : 0 < Q) (h3 : 0 < R) (h4 : 0 < S)
  (h_sum : P + Q + R + S = 50)
  (h_rel : P + 4 = Q - 4 ∧ P + 4 = R * 3 ∧ P + 4 = S / 3) :
  P * Q * R * S = 43 * 107 * 75 * 225 / 1536 := 
by { sorry }

end product_of_integers_P_Q_R_S_l209_209039


namespace emily_card_sequence_l209_209515

/--
Emily orders her playing cards continuously in the following sequence:
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, 3, ...

Prove that the 58th card in this sequence is 6.
-/
theorem emily_card_sequence :
  (58 % 13 = 6) := by
  -- The modulo operation determines the position of the card in the cycle
  sorry

end emily_card_sequence_l209_209515


namespace exists_triangle_with_sides_l2_l3_l4_l209_209219

theorem exists_triangle_with_sides_l2_l3_l4
  (a1 a2 a3 a4 d : ℝ)
  (h_arith_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_pos : a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0)
  (h_d_pos : d > 0) :
  a2 + a3 > a4 ∧ a3 + a4 > a2 ∧ a4 + a2 > a3 :=
by
  sorry

end exists_triangle_with_sides_l2_l3_l4_l209_209219


namespace james_missing_legos_l209_209215

theorem james_missing_legos  (h1 : 500 > 0) (h2 : 500 % 2 = 0) (h3 : 245 < 500)  :
  let total_legos := 500
  let used_legos := total_legos / 2
  let leftover_legos := total_legos - used_legos
  let legos_in_box := 245
  leftover_legos - legos_in_box = 5 := by
{
  sorry
}

end james_missing_legos_l209_209215


namespace problem_statement_l209_209411

theorem problem_statement (a b c x : ℝ) (h1 : a + x^2 = 2015) (h2 : b + x^2 = 2016)
    (h3 : c + x^2 = 2017) (h4 : a * b * c = 24) :
    (a / (b * c) + b / (a * c) + c / (a * b) - (1 / a) - (1 / b) - (1 / c) = 1 / 8) :=
by
  sorry

end problem_statement_l209_209411


namespace Karen_sold_boxes_l209_209736

theorem Karen_sold_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by
  sorry

end Karen_sold_boxes_l209_209736


namespace simplify_expression_l209_209966

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
    a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 :=
by
  sorry

end simplify_expression_l209_209966


namespace parallel_lines_l209_209613

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end parallel_lines_l209_209613


namespace sum_of_products_of_three_numbers_l209_209932

theorem sum_of_products_of_three_numbers
    (a b c : ℝ)
    (h1 : a^2 + b^2 + c^2 = 179)
    (h2 : a + b + c = 21) :
  ab + bc + ac = 131 :=
by
  -- Proof goes here
  sorry

end sum_of_products_of_three_numbers_l209_209932


namespace M_inter_N_eq_l209_209717

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem M_inter_N_eq : {x | -2 < x ∧ x < 2} = M ∩ N := by
  sorry

end M_inter_N_eq_l209_209717


namespace sum_first_15_terms_l209_209786

noncomputable def sum_of_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

noncomputable def fourth_term (a d : ℝ) : ℝ := a + 3 * d
noncomputable def twelfth_term (a d : ℝ) : ℝ := a + 11 * d

theorem sum_first_15_terms (a d : ℝ) 
  (h : fourth_term a d + twelfth_term a d = 10) : sum_of_terms a d 15 = 75 :=
by
  sorry

end sum_first_15_terms_l209_209786


namespace min_convex_cover_area_l209_209566

-- Define the dimensions of the box and the hole
def box_side := 5
def hole_side := 1

-- Define a function to represent the minimum area convex cover
def min_area_convex_cover (box_side hole_side : ℕ) : ℕ :=
  5 -- As given in the problem, the minimum area is concluded to be 5.

-- Theorem to state that the minimum area of the convex cover is 5
theorem min_convex_cover_area : min_area_convex_cover box_side hole_side = 5 :=
by
  -- Proof of the theorem
  sorry

end min_convex_cover_area_l209_209566


namespace sqrt_x_minus_2_meaningful_l209_209089

theorem sqrt_x_minus_2_meaningful (x : ℝ) (hx : x = 0 ∨ x = -1 ∨ x = -2 ∨ x = 2) : (x = 2) ↔ (x - 2 ≥ 0) :=
by
  sorry

end sqrt_x_minus_2_meaningful_l209_209089


namespace maximum_n_value_l209_209043

theorem maximum_n_value (a b c d : ℝ) (n : ℕ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > d) 
(h₃ : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end maximum_n_value_l209_209043


namespace inequality_solution_min_value_of_a2_b2_c2_min_achieved_l209_209844

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (x - 1)

theorem inequality_solution :
  ∀ x : ℝ, (f x ≥ 3) ↔ (x ≤ -1 ∨ x ≥ 1) :=
by sorry

theorem min_value_of_a2_b2_c2 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  a^2 + b^2 + c^2 ≥ 3/7 :=
by sorry

theorem min_achieved (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  (2*a = b) ∧ (b = c/2) ∧ (a^2 + b^2 + c^2 = 3/7) :=
by sorry

end inequality_solution_min_value_of_a2_b2_c2_min_achieved_l209_209844


namespace tiles_needed_l209_209152

def tile_area : ℕ := 3 * 4
def floor_area : ℕ := 36 * 60

theorem tiles_needed : floor_area / tile_area = 180 := by
  sorry

end tiles_needed_l209_209152


namespace smallest_possible_product_l209_209247

theorem smallest_possible_product : 
  ∃ (x : ℕ) (y : ℕ), (x = 56 ∧ y = 78 ∨ x = 57 ∧ y = 68) ∧ x * y = 3876 :=
by
  sorry

end smallest_possible_product_l209_209247


namespace train_ticket_product_l209_209229

theorem train_ticket_product
  (a b c d e : ℕ)
  (h1 : b = a + 1)
  (h2 : c = a + 2)
  (h3 : d = a + 3)
  (h4 : e = a + 4)
  (h_sum : a + b + c + d + e = 120) :
  a * b * c * d * e = 7893600 :=
sorry

end train_ticket_product_l209_209229


namespace fraction_of_time_spent_covering_initial_distance_l209_209531

variables (D T : ℝ) (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40)

theorem fraction_of_time_spent_covering_initial_distance (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40) :
  ((2 / 3) * D / 80) / T = 1 / 2 :=
by
  sorry

end fraction_of_time_spent_covering_initial_distance_l209_209531


namespace cut_wood_into_5_pieces_l209_209062

-- Definitions
def pieces_to_cuts (pieces : ℕ) : ℕ := pieces - 1
def time_per_cut (total_time : ℕ) (cuts : ℕ) : ℕ := total_time / cuts
def total_time_for_pieces (pieces : ℕ) (time_per_cut : ℕ) : ℕ := (pieces_to_cuts pieces) * time_per_cut

-- Given conditions
def conditions : Prop :=
  pieces_to_cuts 4 = 3 ∧
  time_per_cut 24 (pieces_to_cuts 4) = 8

-- Problem statement
theorem cut_wood_into_5_pieces (h : conditions) : total_time_for_pieces 5 8 = 32 :=
by sorry

end cut_wood_into_5_pieces_l209_209062


namespace unique_solution_implies_a_eq_pm_b_l209_209999

theorem unique_solution_implies_a_eq_pm_b 
  (a b : ℝ) 
  (h_nonzero_a : a ≠ 0) 
  (h_nonzero_b : b ≠ 0) 
  (h_unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) : 
  a = b ∨ a = -b :=
sorry

end unique_solution_implies_a_eq_pm_b_l209_209999


namespace correct_statement_B_l209_209544

def flowchart_start_points : Nat := 1
def flowchart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

def program_flowchart_start_points : Nat := 1
def program_flowchart_end_points : Nat := 1

def structure_chart_start_points : Nat := 1
def structure_chart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

theorem correct_statement_B :
  (program_flowchart_start_points = 1 ∧ program_flowchart_end_points = 1) :=
by 
  sorry

end correct_statement_B_l209_209544


namespace find_x_l209_209290

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_x (x : ℝ) (h : star 4 x = 52) : x = 8 :=
by
  sorry

end find_x_l209_209290


namespace opposite_of_five_l209_209490

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l209_209490


namespace find_c_plus_d_l209_209616

theorem find_c_plus_d (c d : ℝ) :
  (∀ x y, (x = (1 / 3) * y + c) → (y = (1 / 3) * x + d) → (x, y) = (3, 3)) → 
  c + d = 4 :=
by
  -- ahead declaration to meet the context requirements in Lean 4
  intros h
  -- Proof steps would go here, but they are omitted
  sorry

end find_c_plus_d_l209_209616


namespace truth_prob_l209_209516

-- Define the probabilities
def prob_A := 0.80
def prob_B := 0.60
def prob_C := 0.75

-- The problem statement
theorem truth_prob :
  prob_A * prob_B * prob_C = 0.27 :=
by
  -- Proof would go here
  sorry

end truth_prob_l209_209516


namespace harry_worked_32_hours_l209_209502

variable (x y : ℝ)
variable (harry_pay james_pay : ℝ)

-- Definitions based on conditions
def harry_weekly_pay (h : ℝ) := 30*x + (h - 30)*y
def james_weekly_pay := 40*x + 1*y

-- Condition: Harry and James were paid the same last week
axiom harry_james_same_pay : ∀ (h : ℝ), harry_weekly_pay x y h = james_weekly_pay x y

-- Prove: Harry worked 32 hours
theorem harry_worked_32_hours : ∃ h : ℝ, h = 32 ∧ harry_weekly_pay x y h = james_weekly_pay x y := by
  sorry

end harry_worked_32_hours_l209_209502


namespace scientific_notation_570_million_l209_209596

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end scientific_notation_570_million_l209_209596


namespace find_n_modulo_l209_209438

theorem find_n_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 11) (h3 : n ≡ 15827 [ZMOD 12]) : n = 11 :=
by
  sorry

end find_n_modulo_l209_209438


namespace three_digit_integers_count_l209_209323

theorem three_digit_integers_count (N : ℕ) :
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
            n % 7 = 4 ∧ 
            n % 8 = 3 ∧ 
            n % 10 = 2) → N = 3 :=
by
  sorry

end three_digit_integers_count_l209_209323


namespace band_member_earnings_l209_209548

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end band_member_earnings_l209_209548


namespace initial_paintings_l209_209657

theorem initial_paintings (x : ℕ) (h : x - 3 = 95) : x = 98 :=
sorry

end initial_paintings_l209_209657


namespace minimum_value_ab_l209_209270

theorem minimum_value_ab (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a * b - 2 * a - b = 0) :
  8 ≤ a * b :=
by sorry

end minimum_value_ab_l209_209270


namespace number_of_digits_in_sum_l209_209266

theorem number_of_digits_in_sum (C D : ℕ) (hC : C ≠ 0 ∧ C < 10) (hD : D % 2 = 0 ∧ D < 10) : 
  (Nat.digits 10 (8765 + (C * 100 + 43) + (D * 10 + 2))).length = 4 := 
by
  sorry

end number_of_digits_in_sum_l209_209266


namespace polar_curve_is_circle_l209_209177

theorem polar_curve_is_circle (θ ρ : ℝ) (h : 4 * Real.sin θ = 5 * ρ) : 
  ∃ c : ℝ×ℝ, ∀ (x y : ℝ), x^2 + y^2 = c.1^2 + c.2^2 :=
by
  sorry

end polar_curve_is_circle_l209_209177


namespace number_of_valid_house_numbers_l209_209324

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def digit_sum_odd (n : ℕ) : Prop :=
  (n / 10 + n % 10) % 2 = 1

def valid_house_number (W X Y Z : ℕ) : Prop :=
  W ≠ 0 ∧ X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0 ∧
  is_two_digit_prime (10 * W + X) ∧ is_two_digit_prime (10 * Y + Z) ∧
  10 * W + X ≠ 10 * Y + Z ∧
  10 * W + X < 60 ∧ 10 * Y + Z < 60 ∧
  digit_sum_odd (10 * W + X)

theorem number_of_valid_house_numbers : ∃ n, n = 108 ∧
  (∀ W X Y Z, valid_house_number W X Y Z → valid_house_number_count = 108) :=
sorry

end number_of_valid_house_numbers_l209_209324


namespace compute_expression_l209_209975

theorem compute_expression :
  (3 + 3 / 8) ^ (2 / 3) - (5 + 4 / 9) ^ (1 / 2) + 0.008 ^ (2 / 3) / 0.02 ^ (1 / 2) * 0.32 ^ (1 / 2) / 0.0625 ^ (1 / 4) = 43 / 150 := 
sorry

end compute_expression_l209_209975


namespace greendale_points_l209_209418

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l209_209418


namespace log_simplification_l209_209310

theorem log_simplification :
  (1 / (Real.log 3 / Real.log 12 + 2))
  + (1 / (Real.log 2 / Real.log 8 + 2))
  + (1 / (Real.log 3 / Real.log 9 + 2)) = 2 :=
  sorry

end log_simplification_l209_209310


namespace triangle_area_l209_209387

theorem triangle_area :
  let line1 (x : ℝ) := 2 * x + 1
  let line2 (x : ℝ) := (16 + x) / 4
  ∃ (base height : ℝ), height = (16 + 2 * base) / 7 ∧ base * height / 2 = 18 / 7 :=
  by
    sorry

end triangle_area_l209_209387


namespace a_b_c_relationship_l209_209978

noncomputable def a (f : ℝ → ℝ) : ℝ := 25 * f (0.2^2)
noncomputable def b (f : ℝ → ℝ) : ℝ := f 1
noncomputable def c (f : ℝ → ℝ) : ℝ := - (Real.log 3 / Real.log 5) * f (Real.log 5 / Real.log 3)

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom decreasing_g (f : ℝ → ℝ) : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)

theorem a_b_c_relationship (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)) :
  a f > b f ∧ b f > c f :=
sorry

end a_b_c_relationship_l209_209978


namespace amy_total_distance_equals_168_l209_209251

def amy_biked_monday := 12

def amy_biked_tuesday (monday: ℕ) := 2 * monday - 3

def amy_biked_other_day (previous_day: ℕ) := previous_day + 2

def total_distance_bike_week := 
  let monday := amy_biked_monday
  let tuesday := amy_biked_tuesday monday
  let wednesday := amy_biked_other_day tuesday
  let thursday := amy_biked_other_day wednesday
  let friday := amy_biked_other_day thursday
  let saturday := amy_biked_other_day friday
  let sunday := amy_biked_other_day saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem amy_total_distance_equals_168 : 
  total_distance_bike_week = 168 := by
  sorry

end amy_total_distance_equals_168_l209_209251


namespace units_digit_of_n_l209_209362

-- Definitions
def units_digit (x : ℕ) : ℕ := x % 10

-- Conditions
variables (m n : ℕ)
axiom condition1 : m * n = 23^5
axiom condition2 : units_digit m = 4

-- Theorem statement
theorem units_digit_of_n : units_digit n = 8 :=
sorry

end units_digit_of_n_l209_209362


namespace exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l209_209297

theorem exists_two_numbers_with_gcd_quotient_ge_p_plus_one (p : ℕ) (hp : Nat.Prime p)
  (l : List ℕ) (hl_len : l.length = p + 1) (hl_distinct : l.Nodup) :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ l ∧ b ∈ l ∧ a > b ∧ a / (Nat.gcd a b) ≥ p + 1 := sorry

end exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l209_209297


namespace option_C_correct_l209_209917

variable {a b c d : ℝ}

theorem option_C_correct (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end option_C_correct_l209_209917


namespace general_term_formula_l209_209017

-- Conditions: sequence \(\frac{1}{2}\), \(\frac{1}{3}\), \(\frac{1}{4}\), \(\frac{1}{5}, \ldots\)
-- Let seq be the sequence in question.

def seq (n : ℕ) : ℚ := 1 / (n + 1)

-- Question: prove the general term formula is \(\frac{1}{n+1}\)
theorem general_term_formula (n : ℕ) : seq n = 1 / (n + 1) :=
by
  -- Proof goes here
  sorry

end general_term_formula_l209_209017


namespace initial_velocity_calculation_l209_209316

-- Define conditions
def acceleration_due_to_gravity := 10 -- m/s^2
def time_to_highest_point := 2 -- s
def velocity_at_highest_point := 0 -- m/s
def initial_observed_acceleration := 15 -- m/s^2

-- Theorem to prove the initial velocity
theorem initial_velocity_calculation
  (a_gravity : ℝ := acceleration_due_to_gravity)
  (t_highest : ℝ := time_to_highest_point)
  (v_highest : ℝ := velocity_at_highest_point)
  (a_initial : ℝ := initial_observed_acceleration) :
  ∃ (v_initial : ℝ), v_initial = 30 := 
sorry

end initial_velocity_calculation_l209_209316


namespace point_distance_l209_209399

theorem point_distance (x y n : ℝ) 
    (h1 : abs x = 8) 
    (h2 : (x - 3)^2 + (y - 10)^2 = 225) 
    (h3 : y > 10) 
    (hn : n = Real.sqrt (x^2 + y^2)) : 
    n = Real.sqrt (364 + 200 * Real.sqrt 2) := 
sorry

end point_distance_l209_209399


namespace fraction_to_decimal_l209_209634

theorem fraction_to_decimal : (53 : ℚ) / (4 * 5^7) = 1325 / 10^7 := sorry

end fraction_to_decimal_l209_209634


namespace box_width_l209_209751

theorem box_width (W : ℕ) (h₁ : 15 * W * 13 = 3120) : W = 16 := by
  sorry

end box_width_l209_209751


namespace combinations_of_eight_choose_three_is_fifty_six_l209_209551

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end combinations_of_eight_choose_three_is_fifty_six_l209_209551


namespace average_weight_of_abc_l209_209619

theorem average_weight_of_abc 
  (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 46)
  (h3 : B = 37) :
  (A + B + C) / 3 = 45 := 
by
  sorry

end average_weight_of_abc_l209_209619


namespace total_cost_of_vitamins_l209_209120

-- Definitions based on the conditions
def original_price : ℝ := 15.00
def discount_percentage : ℝ := 0.20
def coupon_value : ℝ := 2.00
def num_coupons : ℕ := 3
def num_bottles : ℕ := 3

-- Lean statement to prove the final cost
theorem total_cost_of_vitamins
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (coupon_value : ℝ)
  (num_coupons : ℕ)
  (num_bottles : ℕ)
  (discounted_price_per_bottle : ℝ := original_price * (1 - discount_percentage))
  (total_coupon_value : ℝ := coupon_value * num_coupons)
  (total_cost_before_coupons : ℝ := discounted_price_per_bottle * num_bottles) :
  (total_cost_before_coupons - total_coupon_value) = 30.00 :=
by
  sorry

end total_cost_of_vitamins_l209_209120


namespace total_flour_used_l209_209757

theorem total_flour_used :
  let wheat_flour := 0.2
  let white_flour := 0.1
  let rye_flour := 0.15
  let almond_flour := 0.05
  let oat_flour := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + oat_flour = 0.6 :=
by
  sorry

end total_flour_used_l209_209757


namespace quadratic_function_coefficient_not_zero_l209_209095

theorem quadratic_function_coefficient_not_zero (m : ℝ) : (∀ x : ℝ, (m-2)*x^2 + 2*x - 3 ≠ 0) → m ≠ 2 :=
by
  intro h
  by_contra h1
  exact sorry

end quadratic_function_coefficient_not_zero_l209_209095


namespace andrea_rhinestones_needed_l209_209113

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end andrea_rhinestones_needed_l209_209113


namespace tan_alpha_in_third_quadrant_l209_209226

theorem tan_alpha_in_third_quadrant (α : Real) (h1 : Real.sin α = -5/13) (h2 : ∃ k : ℕ, π < α + k * 2 * π ∧ α + k * 2 * π < 3 * π) : 
  Real.tan α = 5/12 :=
sorry

end tan_alpha_in_third_quadrant_l209_209226


namespace sturdy_square_impossible_l209_209562

def size : ℕ := 6
def dominos_used : ℕ := 18
def cells_per_domino : ℕ := 2
def total_cells : ℕ := size * size
def dividing_lines : ℕ := 10

def is_sturdy_square (grid_size : ℕ) (domino_count : ℕ) : Prop :=
  grid_size * grid_size = domino_count * cells_per_domino ∧ 
  ∀ line : ℕ, line < dividing_lines → ∃ domino : ℕ, domino < domino_count

theorem sturdy_square_impossible 
    (grid_size : ℕ) (domino_count : ℕ)
    (h1 : grid_size = size) (h2 : domino_count = dominos_used)
    (h3 : cells_per_domino = 2) (h4 : dividing_lines = 10) : 
  ¬ is_sturdy_square grid_size domino_count :=
by
  cases h1
  cases h2
  cases h3
  cases h4
  sorry

end sturdy_square_impossible_l209_209562


namespace express_y_in_terms_of_x_l209_209583

-- Defining the parameters and assumptions
variables (x y : ℝ)
variables (h : x * y = 30)

-- Stating the theorem
theorem express_y_in_terms_of_x (h : x * y = 30) : y = 30 / x :=
sorry

end express_y_in_terms_of_x_l209_209583


namespace greatest_a_l209_209799

theorem greatest_a (a : ℤ) (h_pos : a > 0) : 
  (∀ x : ℤ, (x^2 + a * x = -30) → (a = 31)) :=
by {
  sorry
}

end greatest_a_l209_209799


namespace x_add_y_eq_neg_one_l209_209241

theorem x_add_y_eq_neg_one (x y : ℝ) (h : |x + 3| + (y - 2)^2 = 0) : x + y = -1 :=
by sorry

end x_add_y_eq_neg_one_l209_209241


namespace one_fifth_of_five_times_nine_l209_209694

theorem one_fifth_of_five_times_nine (a b : ℕ) (h1 : a = 5) (h2 : b = 9) : (1 / 5 : ℚ) * (a * b) = 9 := by
  sorry

end one_fifth_of_five_times_nine_l209_209694


namespace even_sum_probability_l209_209715

-- Define the probabilities of even and odd outcomes for each wheel
def probability_even_first_wheel : ℚ := 2 / 3
def probability_odd_first_wheel : ℚ := 1 / 3
def probability_even_second_wheel : ℚ := 3 / 5
def probability_odd_second_wheel : ℚ := 2 / 5

-- Define the probabilities of the scenarios that result in an even sum
def probability_both_even : ℚ := probability_even_first_wheel * probability_even_second_wheel
def probability_both_odd : ℚ := probability_odd_first_wheel * probability_odd_second_wheel

-- Define the total probability of an even sum
def probability_even_sum : ℚ := probability_both_even + probability_both_odd

-- The theorem statement to be proven
theorem even_sum_probability :
  probability_even_sum = 8 / 15 :=
by
  sorry

end even_sum_probability_l209_209715


namespace geometric_product_l209_209346

theorem geometric_product (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 10) 
  (h2 : 1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 + 1 / a 6 = 5) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end geometric_product_l209_209346


namespace remainder_of_polynomial_division_l209_209343

-- Define the polynomial f(r)
def f (r : ℝ) : ℝ := r ^ 15 + 1

-- Define the polynomial divisor g(r)
def g (r : ℝ) : ℝ := r + 1

-- State the theorem about the remainder when f(r) is divided by g(r)
theorem remainder_of_polynomial_division : 
  (f (-1)) = 0 := by
  -- Skipping the proof for now
  sorry

end remainder_of_polynomial_division_l209_209343


namespace calculate_expr_eq_two_l209_209584

def calculate_expr : ℕ :=
  3^(0^(2^8)) + (3^0^2)^8

theorem calculate_expr_eq_two : calculate_expr = 2 := 
by
  sorry

end calculate_expr_eq_two_l209_209584


namespace cosine_of_negative_three_pi_over_two_l209_209455

theorem cosine_of_negative_three_pi_over_two : 
  Real.cos (-3 * Real.pi / 2) = 0 := 
by sorry

end cosine_of_negative_three_pi_over_two_l209_209455


namespace find_natural_numbers_l209_209145

theorem find_natural_numbers (a b : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (h : a^3 - b^3 = 633 * p) : a = 16 ∧ b = 13 :=
by
  sorry

end find_natural_numbers_l209_209145


namespace range_of_w_l209_209217

noncomputable def f (w x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

theorem range_of_w (w : ℝ) (h_w : 0 < w) :
  (∀ f_zeros : Finset ℝ, ∀ x ∈ f_zeros, (0 < x ∧ x < Real.pi) → f w x = 0 → f_zeros.card = 2) ↔
  (4 / 3 < w ∧ w ≤ 7 / 3) :=
by sorry

end range_of_w_l209_209217


namespace max_value_of_cubes_l209_209737

theorem max_value_of_cubes 
  (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  x^3 + y^3 + z^3 ≤ 27 :=
  sorry

end max_value_of_cubes_l209_209737


namespace find_Roe_speed_l209_209976

-- Definitions from the conditions
def Teena_speed : ℝ := 55
def time_in_hours : ℝ := 1.5
def initial_distance_difference : ℝ := 7.5
def final_distance_difference : ℝ := 15

-- Main theorem statement
theorem find_Roe_speed (R : ℝ) (h1 : R * time_in_hours + final_distance_difference = Teena_speed * time_in_hours - initial_distance_difference) :
  R = 40 :=
  sorry

end find_Roe_speed_l209_209976


namespace gazprom_rd_expense_l209_209606

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end gazprom_rd_expense_l209_209606


namespace prop_C_prop_D_l209_209885

theorem prop_C (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

theorem prop_D (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end prop_C_prop_D_l209_209885


namespace remainder_polynomial_division_l209_209335

theorem remainder_polynomial_division :
  ∀ (x : ℝ), (2 * x^2 - 21 * x + 55) % (x + 3) = 136 := 
sorry

end remainder_polynomial_division_l209_209335


namespace previous_monthly_income_l209_209795

variable (I : ℝ)

-- Conditions from the problem
def condition1 (I : ℝ) : Prop := 0.40 * I = 0.25 * (I + 600)

theorem previous_monthly_income (h : condition1 I) : I = 1000 := by
  sorry

end previous_monthly_income_l209_209795


namespace oranges_for_price_of_apples_l209_209573

-- Given definitions based on the conditions provided
def cost_of_apples_same_as_bananas (a b : ℕ) : Prop := 12 * a = 6 * b
def cost_of_bananas_same_as_cucumbers (b c : ℕ) : Prop := 3 * b = 5 * c
def cost_of_cucumbers_same_as_oranges (c o : ℕ) : Prop := 2 * c = 1 * o

-- The theorem to prove
theorem oranges_for_price_of_apples (a b c o : ℕ) 
  (hab : cost_of_apples_same_as_bananas a b)
  (hbc : cost_of_bananas_same_as_cucumbers b c)
  (hco : cost_of_cucumbers_same_as_oranges c o) : 
  24 * a = 10 * o :=
sorry

end oranges_for_price_of_apples_l209_209573


namespace beetle_total_distance_l209_209865

theorem beetle_total_distance 
  (r_outer : ℝ) (r_middle : ℝ) (r_inner : ℝ)
  (r_outer_eq : r_outer = 25)
  (r_middle_eq : r_middle = 15)
  (r_inner_eq : r_inner = 5)
  : (1/3 * 2 * Real.pi * r_middle + (r_outer - r_middle) + 1/2 * 2 * Real.pi * r_inner + 2 * r_outer + (r_middle - r_inner)) = (15 * Real.pi + 70) :=
by
  rw [r_outer_eq, r_middle_eq, r_inner_eq]
  have := Real.pi
  sorry

end beetle_total_distance_l209_209865


namespace sum_of_tens_and_ones_digit_of_7_pow_25_l209_209928

theorem sum_of_tens_and_ones_digit_of_7_pow_25 : 
  let n := 7 ^ 25 
  let ones_digit := n % 10 
  let tens_digit := (n / 10) % 10 
  ones_digit + tens_digit = 11 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_25_l209_209928


namespace al_original_portion_l209_209128

variables (a b c d : ℝ)

theorem al_original_portion :
  a + b + c + d = 1200 →
  a - 150 + 2 * b + 2 * c + 3 * d = 1800 →
  a = 450 :=
by
  intros h1 h2
  sorry

end al_original_portion_l209_209128


namespace augmented_matrix_solution_l209_209496

theorem augmented_matrix_solution (m n : ℝ) (x y : ℝ)
  (h1 : m * x = 6) (h2 : 3 * y = n) (hx : x = -3) (hy : y = 4) :
  m + n = 10 :=
by
  sorry

end augmented_matrix_solution_l209_209496


namespace isabella_houses_problem_l209_209982

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end isabella_houses_problem_l209_209982


namespace function_is_linear_l209_209041

noncomputable def f : ℕ → ℕ :=
  λ n => n + 1

axiom f_at_0 : f 0 = 1
axiom f_at_2016 : f 2016 = 2017
axiom f_equation : ∀ n : ℕ, f (f n) + f n = 2 * n + 3

theorem function_is_linear : ∀ n : ℕ, f n = n + 1 :=
by
  intro n
  sorry

end function_is_linear_l209_209041


namespace second_set_length_is_correct_l209_209262

variables (first_set_length second_set_length : ℝ)

theorem second_set_length_is_correct 
  (h1 : first_set_length = 4)
  (h2 : second_set_length = 5 * first_set_length) : 
  second_set_length = 20 := 
by 
  sorry

end second_set_length_is_correct_l209_209262


namespace interest_rate_of_additional_investment_l209_209770

section
variable (r : ℝ)

theorem interest_rate_of_additional_investment
  (h : 2800 * 0.05 + 1400 * r = 0.06 * (2800 + 1400)) :
  r = 0.08 := by
  sorry
end

end interest_rate_of_additional_investment_l209_209770


namespace probability_more_ones_than_sixes_l209_209410

theorem probability_more_ones_than_sixes :
  (∃ (p : ℚ), p = 1673 / 3888 ∧ 
  (∃ (d : Fin 6 → ℕ), 
  (∀ i, d i ≤ 4) ∧ 
  (∃ d1 d6 : ℕ, (1 ≤ d1 + d6 ∧ d1 + d6 ≤ 5 ∧ d1 > d6)))) :=
sorry

end probability_more_ones_than_sixes_l209_209410


namespace simplify_and_evaluate_l209_209271

theorem simplify_and_evaluate (a : ℕ) (h : a = 2022) :
  (a - 1) / a / (a - 1 / a) = 1 / 2023 :=
by
  sorry

end simplify_and_evaluate_l209_209271


namespace sum_of_cube_faces_l209_209550

-- Define the cube numbers as consecutive integers starting from 15.
def cube_faces (faces : List ℕ) : Prop :=
  faces = [15, 16, 17, 18, 19, 20]

-- Define the condition that the sum of numbers on opposite faces is the same.
def opposite_faces_condition (pairs : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) = 35

theorem sum_of_cube_faces : ∃ faces : List ℕ, cube_faces faces ∧ (∃ pairs : List (ℕ × ℕ), opposite_faces_condition pairs ∧ faces.sum = 105) :=
by
  sorry

end sum_of_cube_faces_l209_209550


namespace find_B_l209_209452

def A (a : ℝ) : Set ℝ := {3, Real.log a / Real.log 2}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem find_B (a b : ℝ) (hA : A a = {3, 2}) (hB : B a b = {a, b}) (h : (A a) ∩ (B a b) = {2}) :
  B a b = {2, 4} :=
sorry

end find_B_l209_209452


namespace correct_number_for_question_mark_l209_209439

def first_row := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row_no_quest := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
def question_mark (x : ℕ) := first_row.sum = second_row_no_quest.sum + x

theorem correct_number_for_question_mark : question_mark 155 := 
by sorry -- proof to be completed

end correct_number_for_question_mark_l209_209439


namespace chocolate_bars_in_large_box_l209_209800

theorem chocolate_bars_in_large_box
  (small_box_count : ℕ) (chocolate_per_small_box : ℕ)
  (h1 : small_box_count = 20)
  (h2 : chocolate_per_small_box = 25) :
  (small_box_count * chocolate_per_small_box) = 500 :=
by
  sorry

end chocolate_bars_in_large_box_l209_209800


namespace students_passed_both_l209_209148

noncomputable def F_H : ℝ := 32
noncomputable def F_E : ℝ := 56
noncomputable def F_HE : ℝ := 12
noncomputable def total_percentage : ℝ := 100

theorem students_passed_both : (total_percentage - (F_H + F_E - F_HE)) = 24 := by
  sorry

end students_passed_both_l209_209148


namespace sector_to_cone_volume_l209_209914

theorem sector_to_cone_volume (θ : ℝ) (A : ℝ) (V : ℝ) (l r h : ℝ) :
  θ = (2 * Real.pi / 3) →
  A = (3 * Real.pi) →
  A = (1 / 2 * l^2 * θ) →
  θ = (r / l * 2 * Real.pi) →
  h = Real.sqrt (l^2 - r^2) →
  V = (1 / 3 * Real.pi * r^2 * h) →
  V = (2 * Real.sqrt 2 * Real.pi / 3) :=
by
  intros hθ hA hAeq hθeq hh hVeq
  sorry

end sector_to_cone_volume_l209_209914


namespace manicure_cost_per_person_l209_209135

-- Definitions based on given conditions
def fingers_per_person : ℕ := 10
def total_fingers : ℕ := 210
def total_revenue : ℕ := 200  -- in dollars
def non_clients : ℕ := 11

-- Statement we want to prove
theorem manicure_cost_per_person :
  (total_revenue : ℚ) / (total_fingers / fingers_per_person - non_clients) = 9.52 :=
by
  sorry

end manicure_cost_per_person_l209_209135


namespace sum_of_coefficients_l209_209368

theorem sum_of_coefficients :
  (Nat.choose 50 3 + Nat.choose 50 5) = 2138360 := 
by 
  sorry

end sum_of_coefficients_l209_209368


namespace sum_of_coordinates_of_point_B_l209_209955

theorem sum_of_coordinates_of_point_B
  (x y : ℝ)
  (A : (ℝ × ℝ) := (2, 1))
  (B : (ℝ × ℝ) := (x, y))
  (h_line : y = 6)
  (h_slope : (y - 1) / (x - 2) = 4 / 5) :
  x + y = 14.25 :=
by {
  -- convert hypotheses to Lean terms and finish the proof
  sorry
}

end sum_of_coordinates_of_point_B_l209_209955


namespace range_of_a_l209_209074

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x = 1 → a * x^2 + 2 * x + 1 < 0) ↔ a < -3 :=
by
  sorry

end range_of_a_l209_209074


namespace basketball_cards_per_box_l209_209082

-- Given conditions
def num_basketball_boxes : ℕ := 9
def num_football_boxes := num_basketball_boxes - 3
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255
def total_football_cards := num_football_boxes * cards_per_football_box

-- We want to prove that the number of cards in each basketball card box is 15
theorem basketball_cards_per_box :
  (total_cards - total_football_cards) / num_basketball_boxes = 15 := by
  sorry

end basketball_cards_per_box_l209_209082


namespace tan_sum_identity_l209_209689

theorem tan_sum_identity (a b : ℝ) (h₁ : Real.tan a = 1/2) (h₂ : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_identity_l209_209689


namespace reciprocal_of_neg_seven_l209_209044

theorem reciprocal_of_neg_seven : (1 : ℚ) / (-7 : ℚ) = -1 / 7 :=
by
  sorry

end reciprocal_of_neg_seven_l209_209044


namespace distinct_elements_in_T_l209_209731

def sequence1 (k : ℕ) : ℤ := 3 * k - 1
def sequence2 (m : ℕ) : ℤ := 8 * m + 2

def setC : Finset ℤ := Finset.image sequence1 (Finset.range 3000)
def setD : Finset ℤ := Finset.image sequence2 (Finset.range 3000)
def setT : Finset ℤ := setC ∪ setD

theorem distinct_elements_in_T : setT.card = 3000 := by
  sorry

end distinct_elements_in_T_l209_209731


namespace solve_for_y_l209_209547

-- Define the condition
def condition (y : ℤ) : Prop := 7 - y = 13

-- Prove that if the condition is met, then y = -6
theorem solve_for_y (y : ℤ) (h : condition y) : y = -6 :=
by {
  sorry
}

end solve_for_y_l209_209547


namespace law_firm_more_than_two_years_l209_209864

theorem law_firm_more_than_two_years (p_second p_not_first : ℝ) : 
  p_second = 0.30 →
  p_not_first = 0.60 →
  ∃ p_more_than_two_years : ℝ, p_more_than_two_years = 0.30 :=
by
  intros h1 h2
  use (p_not_first - p_second)
  rw [h1, h2]
  norm_num
  done

end law_firm_more_than_two_years_l209_209864


namespace fifteenth_odd_multiple_of_5_is_145_l209_209972

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l209_209972


namespace smaller_acute_angle_l209_209162

theorem smaller_acute_angle (x : ℝ) (h : 5 * x + 4 * x = 90) : 4 * x = 40 :=
by 
  -- proof steps can be added here, but are omitted as per the instructions
  sorry

end smaller_acute_angle_l209_209162


namespace blake_bought_six_chocolate_packs_l209_209370

-- Defining the conditions as hypotheses
variables (lollipops : ℕ) (lollipopCost : ℕ) (packCost : ℕ)
          (cashGiven : ℕ) (changeReceived : ℕ)
          (totalSpent : ℕ) (totalLollipopCost : ℕ) (amountSpentOnChocolates : ℕ)

-- Assertion of the values based on the conditions
axiom h1 : lollipops = 4
axiom h2 : lollipopCost = 2
axiom h3 : packCost = lollipops * lollipopCost
axiom h4 : cashGiven = 6 * 10
axiom h5 : changeReceived = 4
axiom h6 : totalSpent = cashGiven - changeReceived
axiom h7 : totalLollipopCost = lollipops * lollipopCost
axiom h8 : amountSpentOnChocolates = totalSpent - totalLollipopCost
axiom chocolatePacks : ℕ
axiom h9 : chocolatePacks = amountSpentOnChocolates / packCost

-- The statement to be proved
theorem blake_bought_six_chocolate_packs :
    chocolatePacks = 6 :=
by
  subst_vars
  sorry

end blake_bought_six_chocolate_packs_l209_209370


namespace total_legs_on_farm_l209_209066

-- Define the number of each type of animal
def num_ducks : Nat := 6
def num_dogs : Nat := 5
def num_spiders : Nat := 3
def num_three_legged_dogs : Nat := 1

-- Define the number of legs for each type of animal
def legs_per_duck : Nat := 2
def legs_per_dog : Nat := 4
def legs_per_spider : Nat := 8
def legs_per_three_legged_dog : Nat := 3

-- Calculate the total number of legs
def total_duck_legs : Nat := num_ducks * legs_per_duck
def total_dog_legs : Nat := (num_dogs * legs_per_dog) - (num_three_legged_dogs * (legs_per_dog - legs_per_three_legged_dog))
def total_spider_legs : Nat := num_spiders * legs_per_spider

-- The total number of legs on the farm
def total_animal_legs : Nat := total_duck_legs + total_dog_legs + total_spider_legs

-- State the theorem to be proved
theorem total_legs_on_farm : total_animal_legs = 55 :=
by
  -- Assuming conditions and computing as per them
  sorry

end total_legs_on_farm_l209_209066


namespace additional_grassy_ground_l209_209306

theorem additional_grassy_ground (r1 r2 : ℝ) (π : ℝ) :
  r1 = 12 → r2 = 18 → π = Real.pi →
  (π * r2^2 - π * r1^2) = 180 * π := by
sorry

end additional_grassy_ground_l209_209306


namespace proof_statement_l209_209723

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end proof_statement_l209_209723


namespace find_abs_diff_of_average_and_variance_l209_209185

noncomputable def absolute_difference (x y : ℝ) (a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  |x - y|

theorem find_abs_diff_of_average_and_variance (x y : ℝ) (h1 : (x + y + 30 + 29 + 31) / 5 = 30)
  (h2 : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) :
  absolute_difference x y 30 30 29 31 = 4 :=
by
  sorry

end find_abs_diff_of_average_and_variance_l209_209185


namespace lowest_sale_price_is_30_percent_l209_209981

-- Definitions and conditions
def list_price : ℝ := 80
def max_initial_discount : ℝ := 0.50
def additional_sale_discount : ℝ := 0.20

-- Calculations
def initial_discount_amount : ℝ := list_price * max_initial_discount
def initial_discounted_price : ℝ := list_price - initial_discount_amount
def additional_discount_amount : ℝ := list_price * additional_sale_discount
def lowest_sale_price : ℝ := initial_discounted_price - additional_discount_amount

-- Proof statement (with correct answer)
theorem lowest_sale_price_is_30_percent :
  lowest_sale_price = 0.30 * list_price := 
by
  sorry

end lowest_sale_price_is_30_percent_l209_209981


namespace candidate_percentage_l209_209698

variables (M T : ℝ)

theorem candidate_percentage (h1 : (P / 100) * T = M - 30) 
                             (h2 : (45 / 100) * T = M + 15)
                             (h3 : M = 120) : 
                             P = 30 := 
by 
  sorry

end candidate_percentage_l209_209698


namespace travel_time_l209_209827

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l209_209827


namespace find_initial_amount_l209_209071

-- Let x be the initial amount Mark paid for the Magic card
variable {x : ℝ}

-- Condition 1: The card triples in value, resulting in 3x
-- Condition 2: Mark makes a profit of 200
def initial_amount (x : ℝ) : Prop := (3 * x - x = 200)

-- Theorem: Prove that the initial amount x equals 100 given the conditions
theorem find_initial_amount (h : initial_amount x) : x = 100 := by
  sorry

end find_initial_amount_l209_209071


namespace oatmeal_cookies_l209_209888

theorem oatmeal_cookies (total_cookies chocolate_chip_cookies : ℕ)
  (h1 : total_cookies = 6 * 9)
  (h2 : chocolate_chip_cookies = 13) :
  total_cookies - chocolate_chip_cookies = 41 := by
  sorry

end oatmeal_cookies_l209_209888


namespace range_of_m_for_basis_l209_209833

open Real

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 3 * m - 2)

theorem range_of_m_for_basis (m : ℝ) :
  vector_a ≠ vector_b m → m ≠ 2 :=
sorry

end range_of_m_for_basis_l209_209833


namespace possible_values_of_x_l209_209332

theorem possible_values_of_x (x : ℝ) (h : (x^2 - 1) / x = 0) (hx : x ≠ 0) : x = 1 ∨ x = -1 :=
  sorry

end possible_values_of_x_l209_209332


namespace ratio_of_areas_ratio_of_perimeters_l209_209809

-- Define side lengths
def side_length_A : ℕ := 48
def side_length_B : ℕ := 60

-- Define the area of squares
def area_square (side_length : ℕ) : ℕ := side_length * side_length

-- Define the perimeter of squares
def perimeter_square (side_length : ℕ) : ℕ := 4 * side_length

-- Theorem for the ratio of areas
theorem ratio_of_areas : (area_square side_length_A) / (area_square side_length_B) = 16 / 25 :=
by
  sorry

-- Theorem for the ratio of perimeters
theorem ratio_of_perimeters : (perimeter_square side_length_A) / (perimeter_square side_length_B) = 4 / 5 :=
by
  sorry

end ratio_of_areas_ratio_of_perimeters_l209_209809


namespace YZ_length_l209_209697

theorem YZ_length : 
  ∀ (X Y Z : Type) 
  (angle_Y angle_Z angle_X : ℝ)
  (XZ YZ : ℝ),
  angle_Y = 45 ∧ angle_Z = 60 ∧ XZ = 6 →
  angle_X = 180 - angle_Y - angle_Z →
  YZ = XZ * (Real.sin angle_X / Real.sin angle_Y) →
  YZ = 3 * (Real.sqrt 6 + Real.sqrt 2) :=
by
  intros X Y Z angle_Y angle_Z angle_X XZ YZ
  intro h1 h2 h3
  sorry

end YZ_length_l209_209697


namespace false_implies_exists_nonpositive_l209_209139

variable (f : ℝ → ℝ)

theorem false_implies_exists_nonpositive (h : ¬ ∀ x > 0, f x > 0) : ∃ x > 0, f x ≤ 0 :=
by sorry

end false_implies_exists_nonpositive_l209_209139


namespace problem_statement_l209_209933

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geom_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

theorem problem_statement (a₁ q : ℝ) (h : geom_seq a₁ q 6 = 8 * geom_seq a₁ q 3) :
  geom_sum a₁ q 6 / geom_sum a₁ q 3 = 9 :=
by
  -- proof goes here
  sorry

end problem_statement_l209_209933


namespace p_of_neg3_equals_14_l209_209851

-- Functions definitions
def u (x : ℝ) : ℝ := 4 * x + 5
def p (y : ℝ) : ℝ := y^2 - 2 * y + 6

-- Theorem statement
theorem p_of_neg3_equals_14 : p (-3) = 14 := by
  sorry

end p_of_neg3_equals_14_l209_209851


namespace misha_total_students_l209_209012

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l209_209012


namespace find_x_collinear_l209_209992

theorem find_x_collinear (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (x, 1)) 
  (h_collinear : ∃ k : ℝ, (2 * 2 + x) = k * x ∧ (2 * -1 + 1) = k * 1) : x = -2 :=
by
  sorry

end find_x_collinear_l209_209992


namespace fraction_proof_l209_209835

theorem fraction_proof (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) : 
  (x + y) / (y + z) = 26 / 53 := 
by
  sorry

end fraction_proof_l209_209835


namespace clock_hands_angle_120_between_7_and_8_l209_209874

theorem clock_hands_angle_120_between_7_and_8 :
  ∃ (t₁ t₂ : ℕ), (t₁ = 5) ∧ (t₂ = 16) ∧ 
  (∃ (h₀ m₀ : ℕ → ℝ), 
    h₀ 7 = 210 ∧ 
    m₀ 7 = 0 ∧
    (∀ t : ℕ, h₀ (7 + t / 60) = 210 + t * (30 / 60)) ∧
    (∀ t : ℕ, m₀ (7 + t / 60) = t * (360 / 60)) ∧
    ((h₀ (7 + t₁ / 60) - m₀ (7 + t₁ / 60)) % 360 = 120) ∧ 
    ((h₀ (7 + t₂ / 60) - m₀ (7 + t₂ / 60)) % 360 = 120)) := by
  sorry

end clock_hands_angle_120_between_7_and_8_l209_209874


namespace next_ten_winners_each_receive_160_l209_209533

def total_prize_money : ℕ := 2400

def first_winner_amount : ℕ := total_prize_money / 3

def remaining_amount : ℕ := total_prize_money - first_winner_amount

def each_of_ten_winners_receive : ℕ := remaining_amount / 10

theorem next_ten_winners_each_receive_160 : each_of_ten_winners_receive = 160 := by
  sorry

end next_ten_winners_each_receive_160_l209_209533


namespace sum_of_values_l209_209852

def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_values (z₁ z₂ : ℝ) (h₁ : f (3 * z₁) = 10) (h₂ : f (3 * z₂) = 10) :
  z₁ + z₂ = - (2 / 9) :=
by
  sorry

end sum_of_values_l209_209852


namespace duration_of_investment_l209_209944

-- Define the constants as given in the conditions
def Principal : ℝ := 7200
def Rate : ℝ := 17.5
def SimpleInterest : ℝ := 3150

-- Define the time variable we want to prove
def Time : ℝ := 2.5

-- Prove that the calculated time matches the expected value
theorem duration_of_investment :
  SimpleInterest = (Principal * Rate * Time) / 100 :=
sorry

end duration_of_investment_l209_209944


namespace probability_of_snow_during_holiday_l209_209112

theorem probability_of_snow_during_holiday
  (P_snow_Friday : ℝ)
  (P_snow_Monday : ℝ)
  (P_snow_independent : true) -- Placeholder since we assume independence
  (h_Friday: P_snow_Friday = 0.30)
  (h_Monday: P_snow_Monday = 0.45) :
  ∃ P_snow_holiday, P_snow_holiday = 0.615 :=
by
  sorry

end probability_of_snow_during_holiday_l209_209112


namespace cheburashkas_erased_l209_209956

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l209_209956


namespace gribblean_words_count_l209_209648

universe u

-- Define the Gribblean alphabet size
def alphabet_size : Nat := 3

-- Words of length 1 to 4
def words_of_length (n : Nat) : Nat :=
  alphabet_size ^ n

-- All possible words count
def total_words : Nat :=
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3) + (words_of_length 4)

-- Theorem statement
theorem gribblean_words_count : total_words = 120 :=
by
  sorry

end gribblean_words_count_l209_209648


namespace sin_3theta_over_sin_theta_l209_209078

theorem sin_3theta_over_sin_theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 2) : 
  Real.sin (3 * θ) / Real.sin θ = 1 / 3 :=
by
  sorry

end sin_3theta_over_sin_theta_l209_209078


namespace sum_is_zero_l209_209307

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end sum_is_zero_l209_209307


namespace ellipse_hyperbola_foci_l209_209602

theorem ellipse_hyperbola_foci {a b : ℝ} (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) :
  a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 :=
by sorry

end ellipse_hyperbola_foci_l209_209602


namespace rearranged_number_divisible_by_27_l209_209384

theorem rearranged_number_divisible_by_27 (n m : ℕ) (hn : m = 3 * n) 
  (hdigits : ∀ a b : ℕ, (a ∈ n.digits 10 ↔ b ∈ m.digits 10)) : 27 ∣ m :=
sorry

end rearranged_number_divisible_by_27_l209_209384


namespace stories_in_building_l209_209312

-- Definitions of the conditions
def apartments_per_floor := 4
def people_per_apartment := 2
def total_people := 200

-- Definition of people per floor
def people_per_floor := apartments_per_floor * people_per_apartment

-- The theorem stating the desired conclusion
theorem stories_in_building :
  total_people / people_per_floor = 25 :=
by
  -- Insert the proof here
  sorry

end stories_in_building_l209_209312


namespace planeThroughPointAndLine_l209_209527

theorem planeThroughPointAndLine :
  ∃ A B C D : ℤ, (A = -3 ∧ B = -4 ∧ C = -4 ∧ D = 14) ∧ 
  (∀ x y z : ℝ, x = 2 ∧ y = -3 ∧ z = 5 ∨ (∃ t : ℝ, x = 4 * t + 2 ∧ y = -5 * t - 1 ∧ z = 2 * t + 3) → A * x + B * y + C * z + D = 0) :=
sorry

end planeThroughPointAndLine_l209_209527


namespace canoe_kayak_rental_l209_209630

theorem canoe_kayak_rental:
  ∀ (C K : ℕ), 
    12 * C + 18 * K = 504 → 
    C = (3 * K) / 2 → 
    C - K = 7 :=
  by
    intro C K
    intros h1 h2
    sorry

end canoe_kayak_rental_l209_209630


namespace Jake_initial_balloons_l209_209759

theorem Jake_initial_balloons (J : ℕ) 
  (h1 : 6 = (J + 3) + 1) : 
  J = 2 :=
by
  sorry

end Jake_initial_balloons_l209_209759


namespace weight_difference_at_end_of_year_l209_209742

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end weight_difference_at_end_of_year_l209_209742


namespace yaw_yaw_age_in_2016_l209_209901

def is_lucky_double_year (y : Nat) : Prop :=
  let d₁ := y / 1000 % 10
  let d₂ := y / 100 % 10
  let d₃ := y / 10 % 10
  let last_digit := y % 10
  last_digit = 2 * (d₁ + d₂ + d₃)

theorem yaw_yaw_age_in_2016 (next_lucky_year : Nat) (yaw_yaw_age_in_next_lucky_year : Nat)
  (h1 : is_lucky_double_year 2016)
  (h2 : ∀ y, y > 2016 → is_lucky_double_year y → y = next_lucky_year)
  (h3 : yaw_yaw_age_in_next_lucky_year = 17) :
  (17 - (next_lucky_year - 2016)) = 5 := sorry

end yaw_yaw_age_in_2016_l209_209901


namespace seventh_monomial_l209_209935

noncomputable def sequence_monomial (n : ℕ) (x : ℝ) : ℝ :=
  (-1)^n * 2^(n-1) * x^(n-1)

theorem seventh_monomial (x : ℝ) : sequence_monomial 7 x = -64 * x^6 := by
  sorry

end seventh_monomial_l209_209935


namespace complement_of_A_in_U_l209_209458

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set A
def A : Set ℤ := {x | x ∈ Set.univ ∧ x^2 + x - 2 < 0}

-- State the theorem about the complement of A in U
theorem complement_of_A_in_U :
  (U \ A) = {-2, 1, 2} :=
sorry

end complement_of_A_in_U_l209_209458


namespace exists_k_undecided_l209_209677

def tournament (n : ℕ) : Type :=
  { T : Fin n → Fin n → Prop // ∀ i j, T i j = ¬T j i }

def k_undecided (n k : ℕ) (T : tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k → ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → T.1 p a

theorem exists_k_undecided (k : ℕ) (hk : 0 < k) : ∃ (n : ℕ), n > k ∧ ∃ (T : tournament n), k_undecided n k T :=
by
  sorry

end exists_k_undecided_l209_209677


namespace f_at_one_f_decreasing_f_min_on_interval_l209_209096

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_defined : ∀ x, 0 < x → ∃ y, f y = y
axiom f_eq : ∀ x1 x2, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2
axiom f_neg : ∀ x, 1 < x → f x < 0

-- Proof statements
theorem f_at_one : f 1 = 0 := sorry

theorem f_decreasing : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2 := sorry

axiom f_at_three : f 3 = -1

theorem f_min_on_interval : ∀ x, 2 ≤ x ∧ x ≤ 9 → f x ≥ -2 := sorry

end f_at_one_f_decreasing_f_min_on_interval_l209_209096


namespace megan_final_balance_same_as_starting_balance_l209_209182

theorem megan_final_balance_same_as_starting_balance :
  let starting_balance : ℝ := 125
  let increased_balance := starting_balance * (1 + 0.25)
  let final_balance := increased_balance * (1 - 0.20)
  final_balance = starting_balance :=
by
  sorry

end megan_final_balance_same_as_starting_balance_l209_209182


namespace complement_union_M_N_correct_l209_209528

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define the set N
def N : Set ℕ := {5, 6, 7}

-- Define the union of M and N
def union_M_N : Set ℕ := M ∪ N

-- Define the complement of the union of M and N in U
def complement_union_M_N : Set ℕ := U \ union_M_N

-- Main theorem statement to prove
theorem complement_union_M_N_correct : complement_union_M_N = {2, 4, 8} :=
by
  sorry

end complement_union_M_N_correct_l209_209528


namespace intersection_M_N_eq_set_l209_209448

universe u

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {y | ∃ x, x ∈ M ∧ y = 2 * x + 1}

-- Prove the intersection M ∩ N = {-1, 1}
theorem intersection_M_N_eq_set : M ∩ N = {-1, 1} :=
by
  simp [Set.ext_iff, M, N]
  sorry

end intersection_M_N_eq_set_l209_209448


namespace solve_for_n_l209_209484

theorem solve_for_n (n : ℤ) (h : (5/4 : ℚ) * n + (5/4 : ℚ) = n) : n = -5 := by
    sorry

end solve_for_n_l209_209484


namespace continuous_at_5_l209_209581

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x - 2 else 3 * x + b

theorem continuous_at_5 {b : ℝ} : ContinuousAt (fun x => f x b) 5 ↔ b = -12 := by
  sorry

end continuous_at_5_l209_209581


namespace geometric_sequence_fourth_term_l209_209457

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℕ)
  (r : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₅ = 2187)
  (h₃ : a₅ = a₁ * r ^ 4) :
  a₁ * r ^ 3 = 2187 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l209_209457


namespace calculate_expression_l209_209024

theorem calculate_expression : -2 - 2 * Real.sin (Real.pi / 4) + (Real.pi - 3.14) * 0 + (-1) ^ 3 = -3 - Real.sqrt 2 := by 
sorry

end calculate_expression_l209_209024


namespace cost_of_fencing_per_meter_l209_209891

theorem cost_of_fencing_per_meter (x : ℝ) (length width : ℝ) (area : ℝ) (total_cost : ℝ) :
  length = 3 * x ∧ width = 2 * x ∧ area = 3750 ∧ area = length * width ∧ total_cost = 125 →
  (total_cost / (2 * (length + width)) = 0.5) :=
by
  sorry

end cost_of_fencing_per_meter_l209_209891


namespace term_sequence_10th_l209_209683

theorem term_sequence_10th :
  let a (n : ℕ) := (-1:ℚ)^(n+1) * (2*n)/(2*n + 1)
  a 10 = -20/21 := 
by
  sorry

end term_sequence_10th_l209_209683


namespace abel_overtake_kelly_chris_overtake_both_l209_209006

-- Given conditions and variables
variable (d : ℝ)  -- distance at which Abel overtakes Kelly
variable (d_c : ℝ)  -- distance at which Chris overtakes both Kelly and Abel
variable (t_k : ℝ)  -- time taken by Kelly to run d meters
variable (t_a : ℝ)  -- time taken by Abel to run (d + 3) meters
variable (t_c : ℝ)  -- time taken by Chris to run the required distance
variable (k_speed : ℝ := 9)  -- Kelly's speed
variable (a_speed : ℝ := 9.5)  -- Abel's speed
variable (c_speed : ℝ := 10)  -- Chris's speed
variable (head_start_k : ℝ := 3)  -- Kelly's head start over Abel
variable (head_start_c : ℝ := 2)  -- Chris's head start behind Abel
variable (lost_by : ℝ := 0.75)  -- Abel lost by distance

-- Proof problem for Abel overtaking Kelly
theorem abel_overtake_kelly 
  (hk : t_k = d / k_speed) 
  (ha : t_a = (d + head_start_k) / a_speed) 
  (h_lost : lost_by = 0.75):
  d + lost_by = 54.75 := 
sorry

-- Proof problem for Chris overtaking both Kelly and Abel
theorem chris_overtake_both 
  (hc : t_c = (d_c + 5) / c_speed)
  (h_56 : d_c = 56):
  d_c = c_speed * (56 / c_speed) :=
sorry

end abel_overtake_kelly_chris_overtake_both_l209_209006


namespace intersection_point_on_y_eq_neg_x_l209_209016

theorem intersection_point_on_y_eq_neg_x 
  (α β : ℝ)
  (h1 : ∃ x y : ℝ, (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧ 
                   (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧ 
                   (y = -x)) :
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 :=
sorry

end intersection_point_on_y_eq_neg_x_l209_209016


namespace sector_arc_length_l209_209815

noncomputable def arc_length (R : ℝ) (θ : ℝ) : ℝ :=
  θ / 180 * Real.pi * R

theorem sector_arc_length
  (central_angle : ℝ) (area : ℝ) (arc_length_answer : ℝ)
  (h1 : central_angle = 120)
  (h2 : area = 300 * Real.pi) :
  arc_length_answer = 20 * Real.pi :=
by
  sorry

end sector_arc_length_l209_209815


namespace find_x_l209_209202

theorem find_x (x y : ℤ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
by
  sorry

end find_x_l209_209202


namespace expand_product_l209_209313

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 :=
by
  sorry

end expand_product_l209_209313


namespace find_divisor_l209_209102

-- Definitions from the conditions
def remainder : ℤ := 8
def quotient : ℤ := 43
def dividend : ℤ := 997
def is_prime (n : ℤ) : Prop := n ≠ 1 ∧ (∀ d : ℤ, d ∣ n → d = 1 ∨ d = n)

-- The proof problem statement
theorem find_divisor (d : ℤ) 
  (hd : is_prime d) 
  (hdiv : dividend = (d * quotient) + remainder) : 
  d = 23 := 
sorry

end find_divisor_l209_209102


namespace sequence_a_n_is_n_l209_209756

-- Definitions and statements based on the conditions
def sequence_cond (a : ℕ → ℕ) (n : ℕ) : ℕ := 
1 / 2 * (a n) ^ 2 + n / 2

theorem sequence_a_n_is_n :
  ∀ (a : ℕ → ℕ), (∀ n, n > 0 → ∃ (S_n : ℕ), S_n = sequence_cond a n) → 
  (∀ n, n > 0 → a n = n) :=
by
  sorry

end sequence_a_n_is_n_l209_209756


namespace tammy_speed_second_day_l209_209445

theorem tammy_speed_second_day :
  ∀ (v1 t1 v2 t2 : ℝ), 
    t1 + t2 = 14 →
    t2 = t1 - 2 →
    v2 = v1 + 0.5 →
    v1 * t1 + v2 * t2 = 52 →
    v2 = 4 :=
by
  intros v1 t1 v2 t2 h1 h2 h3 h4
  sorry

end tammy_speed_second_day_l209_209445


namespace larger_of_two_numbers_with_hcf_25_l209_209025

theorem larger_of_two_numbers_with_hcf_25 (a b : ℕ) (h_hcf: Nat.gcd a b = 25)
  (h_lcm_factors: 13 * 14 = (25 * 13 * 14) / (Nat.gcd a b)) :
  max a b = 350 :=
sorry

end larger_of_two_numbers_with_hcf_25_l209_209025


namespace isosceles_triangle_base_length_l209_209554

def is_isosceles (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_base_length
  (x y : ℝ)
  (h1 : 2 * x + 2 * y = 16)
  (h2 : 4^2 + y^2 = x^2)
  (h3 : is_isosceles x x (2 * y) ) :
  2 * y = 6 := 
by
  sorry

end isosceles_triangle_base_length_l209_209554


namespace number_satisfies_equation_l209_209442

theorem number_satisfies_equation :
  ∃ x : ℝ, (0.6667 * x - 10 = 0.25 * x) ∧ (x = 23.9936) :=
by
  sorry

end number_satisfies_equation_l209_209442


namespace max_initial_value_seq_l209_209961

theorem max_initial_value_seq :
  ∀ (x : Fin 1996 → ℝ),
    (∀ i : Fin 1996, 1 ≤ x i) →
    (x 0 = x 1995) →
    (∀ i : Fin 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1)) →
    x 0 ≤ 2 ^ 997 :=
sorry

end max_initial_value_seq_l209_209961


namespace percentage_increase_l209_209450

theorem percentage_increase (original final : ℝ) (h1 : original = 90) (h2 : final = 135) : ((final - original) / original) * 100 = 50 := 
by
  sorry

end percentage_increase_l209_209450


namespace num_palindromes_is_correct_l209_209242

section Palindromes

def num_alphanumeric_chars : ℕ := 10 + 26

def num_four_char_palindromes : ℕ := num_alphanumeric_chars * num_alphanumeric_chars

theorem num_palindromes_is_correct : num_four_char_palindromes = 1296 :=
by
  sorry

end Palindromes

end num_palindromes_is_correct_l209_209242


namespace divide_square_into_equal_parts_l209_209430

-- Given a square with four shaded smaller squares inside
structure SquareWithShaded (n : ℕ) :=
  (squares : Fin n → Fin n → Prop) -- this models the presence of shaded squares
  (shaded : (Fin 2) → (Fin 2) → Prop)

-- To prove: we can divide the square into four equal parts with each containing one shaded square
theorem divide_square_into_equal_parts :
  ∀ (sq : SquareWithShaded 4),
  ∃ (parts : Fin 2 → Fin 2 → Prop),
  (∀ i j, parts i j ↔ 
    ((i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 0) ∨ 
    (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1)) ∧
    (∃! k l, sq.shaded k l ∧ parts i j)) :=
sorry

end divide_square_into_equal_parts_l209_209430


namespace min_value_expression_l209_209393

theorem min_value_expression : ∃ x y z : ℝ, (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10) = -7 / 2 :=
by sorry

end min_value_expression_l209_209393


namespace even_numbers_with_specific_square_properties_l209_209129

theorem even_numbers_with_specific_square_properties (n : ℕ) :
  (10^13 ≤ n^2 ∧ n^2 < 10^14 ∧ (n^2 % 100) / 10 = 5) → 
  (2 ∣ n ∧ 273512 > 10^5) := 
sorry

end even_numbers_with_specific_square_properties_l209_209129


namespace perry_more_games_than_phil_l209_209110

theorem perry_more_games_than_phil (dana_wins charlie_wins perry_wins : ℕ) :
  perry_wins = dana_wins + 5 →
  charlie_wins = dana_wins - 2 →
  charlie_wins + 3 = 12 →
  perry_wins - 12 = 4 :=
by
  sorry

end perry_more_games_than_phil_l209_209110


namespace maximum_area_right_triangle_in_rectangle_l209_209645

theorem maximum_area_right_triangle_in_rectangle :
  ∃ (area : ℕ), 
  (∀ (a b : ℕ), a = 12 ∧ b = 5 → area = 1 / 2 * a * b) :=
by
  use 30
  sorry

end maximum_area_right_triangle_in_rectangle_l209_209645


namespace library_width_l209_209967

theorem library_width 
  (num_libraries : ℕ) 
  (length_per_library : ℕ) 
  (total_area_km2 : ℝ) 
  (conversion_factor : ℝ) 
  (total_area : ℝ) 
  (area_of_one_library : ℝ) 
  (width_of_library : ℝ) :

  num_libraries = 8 →
  length_per_library = 300 →
  total_area_km2 = 0.6 →
  conversion_factor = 1000000 →
  total_area = total_area_km2 * conversion_factor →
  area_of_one_library = total_area / num_libraries →
  width_of_library = area_of_one_library / length_per_library →
  width_of_library = 250 :=
by
  intros;
  sorry

end library_width_l209_209967


namespace part1_part2_l209_209320

variable (x k : ℝ)

-- Part (1)
theorem part1 (h1 : x = 3) : ∀ k : ℝ, (1 + k) * 3 ≤ k^2 + k + 4 := sorry

-- Part (2)
theorem part2 (h2 : ∀ k : ℝ, -4 ≤ k → (1 + k) * x ≤ k^2 + k + 4) : -5 ≤ x ∧ x ≤ 3 := sorry

end part1_part2_l209_209320


namespace find_value_of_a_plus_b_l209_209709

noncomputable def A (a b : ℤ) : Set ℤ := {1, a, b}
noncomputable def B (a b : ℤ) : Set ℤ := {a, a^2, a * b}

theorem find_value_of_a_plus_b (a b : ℤ) (h : A a b = B a b) : a + b = -1 :=
by sorry

end find_value_of_a_plus_b_l209_209709


namespace m_zero_sufficient_but_not_necessary_l209_209369

-- Define the sequence a_n
variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the condition for equal difference of squares sequence
def equal_diff_of_squares_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, (a (n+1))^2 - (a n)^2 = d

-- Define the sequence b_n as an arithmetic sequence with common difference m
variable (b : ℕ → ℝ)
variable (m : ℝ)

def arithmetic_sequence (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n+1) - b n = m

-- Prove "m = 0" is a sufficient but not necessary condition for {b_n} to be an equal difference of squares sequence
theorem m_zero_sufficient_but_not_necessary (a b : ℕ → ℝ) (d m : ℝ) :
  equal_diff_of_squares_sequence a d → arithmetic_sequence b m → (m = 0 → equal_diff_of_squares_sequence b d) ∧ (¬(m ≠ 0) → equal_diff_of_squares_sequence b d) :=
sorry


end m_zero_sufficient_but_not_necessary_l209_209369


namespace div_poly_odd_power_l209_209133

theorem div_poly_odd_power (a b : ℤ) (n : ℕ) : (a + b) ∣ (a^(2*n+1) + b^(2*n+1)) :=
sorry

end div_poly_odd_power_l209_209133


namespace no_four_digit_numbers_divisible_by_11_l209_209780

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) :
  (a + b + c + d = 9) ∧ ((a + c) - (b + d)) % 11 = 0 → false :=
by
  sorry

end no_four_digit_numbers_divisible_by_11_l209_209780


namespace increase_fraction_l209_209055

theorem increase_fraction (A F : ℝ) 
  (h₁ : A = 83200) 
  (h₂ : A * (1 + F) ^ 2 = 105300) : 
  F = 0.125 :=
by
  sorry

end increase_fraction_l209_209055


namespace density_is_not_vector_l209_209473

/-- Conditions definition -/
def is_vector (quantity : String) : Prop :=
quantity = "Buoyancy" ∨ quantity = "Wind speed" ∨ quantity = "Displacement"

/-- Problem statement -/
theorem density_is_not_vector : ¬ is_vector "Density" := 
by 
sorry

end density_is_not_vector_l209_209473


namespace point_in_second_quadrant_l209_209921

def point := (ℝ × ℝ)

def second_quadrant (p : point) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : second_quadrant (-1, 2) :=
sorry

end point_in_second_quadrant_l209_209921


namespace min_value_expression_l209_209878

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 20)^2 ≥ 100 :=
sorry

end min_value_expression_l209_209878


namespace rhombus_area_8_cm2_l209_209365

open Real

noncomputable def rhombus_area (side : ℝ) (angle : ℝ) : ℝ :=
  (side * side * sin angle) / 2 * 2

theorem rhombus_area_8_cm2 (side : ℝ) (angle : ℝ) (h1 : side = 4) (h2 : angle = π / 4) : rhombus_area side angle = 8 :=
by
  -- Definitions and calculations are omitted and replaced with 'sorry'
  sorry

end rhombus_area_8_cm2_l209_209365


namespace exist_elem_not_in_union_l209_209424

-- Assume closed sets
def isClosedSet (S : Set ℝ) : Prop :=
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

-- The theorem to prove
theorem exist_elem_not_in_union {S1 S2 : Set ℝ} (hS1 : isClosedSet S1) (hS2 : isClosedSet S2) :
  S1 ⊂ (Set.univ : Set ℝ) → S2 ⊂ (Set.univ : Set ℝ) → ∃ c : ℝ, c ∉ S1 ∪ S2 :=
by
  intro h1 h2
  sorry

end exist_elem_not_in_union_l209_209424


namespace max_sum_of_factors_l209_209814

theorem max_sum_of_factors (heartsuit spadesuit : ℕ) (h : heartsuit * spadesuit = 24) :
  heartsuit + spadesuit ≤ 25 :=
sorry

end max_sum_of_factors_l209_209814


namespace product_of_roots_quadratic_eq_l209_209649

theorem product_of_roots_quadratic_eq : 
  ∀ (x1 x2 : ℝ), 
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 → (x = x1 ∨ x = x2)) → 
  x1 * x2 = -3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_quadratic_eq_l209_209649


namespace combine_monomials_x_plus_y_l209_209939

theorem combine_monomials_x_plus_y : ∀ (x y : ℤ),
  7 * x = 2 - 4 * y →
  y + 7 = 2 * x →
  x + y = -1 :=
by
  intros x y h1 h2
  sorry

end combine_monomials_x_plus_y_l209_209939


namespace Δy_over_Δx_l209_209600

-- Conditions
def f (x : ℝ) : ℝ := 2 * x^2 - 4
def y1 : ℝ := f 1
def y2 (Δx : ℝ) : ℝ := f (1 + Δx)
def Δy (Δx : ℝ) : ℝ := y2 Δx - y1

-- Theorem statement
theorem Δy_over_Δx (Δx : ℝ) : Δy Δx / Δx = 4 + 2 * Δx := by
  sorry

end Δy_over_Δx_l209_209600


namespace find_books_second_shop_l209_209906

def total_books (books_first_shop books_second_shop : ℕ) : ℕ :=
  books_first_shop + books_second_shop

def total_cost (cost_first_shop cost_second_shop : ℕ) : ℕ :=
  cost_first_shop + cost_second_shop

def average_price (total_cost total_books : ℕ) : ℕ :=
  total_cost / total_books

theorem find_books_second_shop : 
  ∀ (books_first_shop cost_first_shop cost_second_shop : ℕ),
    books_first_shop = 65 →
    cost_first_shop = 1480 →
    cost_second_shop = 920 →
    average_price (total_cost cost_first_shop cost_second_shop) (total_books books_first_shop (2400 / 20 - 65)) = 20 →
    2400 / 20 - 65 = 55 := 
by sorry

end find_books_second_shop_l209_209906


namespace lines_intersect_at_l209_209104

theorem lines_intersect_at :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 7 * y = -3 * x - 4 ∧ x = 54 / 5 ∧ y = -26 / 5 := 
by
  sorry

end lines_intersect_at_l209_209104


namespace problem1_l209_209598

theorem problem1 : 2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) = 0 := by
  sorry

end problem1_l209_209598


namespace geometric_sequence_fifth_term_l209_209057

theorem geometric_sequence_fifth_term : 
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  a₅ = 1 / 2048 :=
by
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  sorry

end geometric_sequence_fifth_term_l209_209057


namespace solve_for_x_l209_209097

theorem solve_for_x (x : ℝ) (h : (2 * x - 3) ^ (x + 3) = 1) : 
  x = -3 ∨ x = 2 ∨ x = 1 := 
sorry

end solve_for_x_l209_209097


namespace tax_rate_correct_l209_209389

noncomputable def tax_rate (total_payroll : ℕ) (tax_free_payroll : ℕ) (tax_paid : ℕ) : ℚ :=
  if total_payroll > tax_free_payroll 
  then (tax_paid : ℚ) / (total_payroll - tax_free_payroll) * 100
  else 0

theorem tax_rate_correct :
  tax_rate 400000 200000 400 = 0.2 :=
by
  sorry

end tax_rate_correct_l209_209389


namespace rearrange_squares_into_one_square_l209_209872

theorem rearrange_squares_into_one_square 
  (a b : ℕ) (h_a : a = 3) (h_b : b = 1) 
  (parts : Finset (ℕ × ℕ)) 
  (h_parts1 : parts.card ≤ 3)
  (h_parts2 : ∀ p ∈ parts, p.1 * p.2 = a * a ∨ p.1 * p.2 = b * b)
  : ∃ c : ℕ, (c * c = (a * a) + (b * b)) :=
by
  sorry

end rearrange_squares_into_one_square_l209_209872


namespace savings_account_final_amount_l209_209228

noncomputable def final_amount (P R : ℝ) (t : ℕ) : ℝ :=
  P * (1 + R) ^ t

theorem savings_account_final_amount :
  final_amount 2500 0.06 21 = 8017.84 :=
by
  sorry

end savings_account_final_amount_l209_209228


namespace find_x_l209_209817

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l209_209817


namespace pencils_bought_l209_209599

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end pencils_bought_l209_209599


namespace perimeter_pentagon_ABCD_l209_209631

noncomputable def AB : ℝ := 2
noncomputable def BC : ℝ := Real.sqrt 8
noncomputable def CD : ℝ := Real.sqrt 18
noncomputable def DE : ℝ := Real.sqrt 32
noncomputable def AE : ℝ := Real.sqrt 62

theorem perimeter_pentagon_ABCD : 
  AB + BC + CD + DE + AE = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  -- Note: The proof has been skipped as per instruction.
  sorry

end perimeter_pentagon_ABCD_l209_209631


namespace anthony_initial_pencils_l209_209859

def initial_pencils (given_pencils : ℝ) (remaining_pencils : ℝ) : ℝ :=
  given_pencils + remaining_pencils

theorem anthony_initial_pencils :
  initial_pencils 9.0 47.0 = 56.0 :=
by
  sorry

end anthony_initial_pencils_l209_209859


namespace meaningful_expression_l209_209797

theorem meaningful_expression (m : ℝ) :
  (2 - m ≥ 0) ∧ (m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end meaningful_expression_l209_209797


namespace total_seats_l209_209925

theorem total_seats (s : ℝ) : 
  let first_class := 36
  let business_class := 0.30 * s
  let economy_class := (3/5:ℝ) * s
  let premium_economy := s - (first_class + business_class + economy_class)
  first_class + business_class + economy_class + premium_economy = s := by 
  sorry

end total_seats_l209_209925


namespace largest_c_for_minus3_in_range_of_quadratic_l209_209434

theorem largest_c_for_minus3_in_range_of_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 5*x + c = -3) ↔ c ≤ 13/4 :=
sorry

end largest_c_for_minus3_in_range_of_quadratic_l209_209434


namespace functional_eq_zero_l209_209358

theorem functional_eq_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_l209_209358


namespace fraction_simplification_l209_209867

theorem fraction_simplification : 
  (2222 - 2123) ^ 2 / 121 = 81 :=
by
  sorry

end fraction_simplification_l209_209867


namespace evaluate_expression_l209_209848

-- Define the integers a and b
def a := 2019
def b := 2020

-- The main theorem stating the equivalence
theorem evaluate_expression :
  (a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3 + 6) / (a * b) = 5 / (a * b) := 
by
  sorry

end evaluate_expression_l209_209848


namespace condition_necessary_but_not_sufficient_l209_209330

variable (a : ℝ)

theorem condition_necessary_but_not_sufficient (h : a^2 < 1) : (a < 1) ∧ (¬(a < 1 → a^2 < 1)) := sorry

end condition_necessary_but_not_sufficient_l209_209330


namespace right_angle_triangle_iff_arithmetic_progression_l209_209587

noncomputable def exists_right_angle_triangle_with_rational_sides_and_area (d : ℤ) : Prop :=
  ∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)

noncomputable def rational_squares_in_arithmetic_progression (x y z : ℚ) : Prop :=
  2 * y^2 = x^2 + z^2

theorem right_angle_triangle_iff_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)) ↔ ∃ (x y z : ℚ), rational_squares_in_arithmetic_progression x y z :=
sorry

end right_angle_triangle_iff_arithmetic_progression_l209_209587


namespace right_triangle_area_inscribed_3_4_l209_209794

theorem right_triangle_area_inscribed_3_4 (r1 r2: ℝ) (h1 : r1 = 3) (h2 : r2 = 4) : 
  ∃ (S: ℝ), S = 150 :=
by
  sorry

end right_triangle_area_inscribed_3_4_l209_209794


namespace ratio_josh_to_doug_l209_209155

theorem ratio_josh_to_doug (J D B : ℕ) (h1 : J + D + B = 68) (h2 : J = 2 * B) (h3 : D = 32) : J / D = 3 / 4 := 
by
  sorry

end ratio_josh_to_doug_l209_209155


namespace caroline_citrus_drinks_l209_209116

-- Definitions based on problem conditions
def citrus_drinks (oranges : ℕ) : ℕ := (oranges * 8) / 3

-- Define problem statement
theorem caroline_citrus_drinks : citrus_drinks 21 = 56 :=
by
  sorry

end caroline_citrus_drinks_l209_209116


namespace exists_four_integers_mod_5050_l209_209870

theorem exists_four_integers_mod_5050 (S : Finset ℕ) (hS_card : S.card = 101) (hS_bound : ∀ x ∈ S, x < 5050) : 
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b - c - d) % 5050 = 0 :=
sorry

end exists_four_integers_mod_5050_l209_209870


namespace rhombus_diagonal_l209_209234

theorem rhombus_diagonal (d1 d2 : ℝ) (area_tri : ℝ) (h1 : d1 = 15) (h2 : area_tri = 75) :
  (d1 * d2) / 2 = 2 * area_tri → d2 = 20 :=
by
  sorry

end rhombus_diagonal_l209_209234


namespace find_b_l209_209941

-- Definitions
def quadratic (x b c : ℝ) : ℝ := x^2 + b * x + c

theorem find_b (b c : ℝ) 
  (h_diff : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → (∀ y : ℝ, 1 ≤ y ∧ y ≤ 7 → quadratic x b c - quadratic y b c = 25)) :
  b = -4 ∨ b = -12 :=
by sorry

end find_b_l209_209941


namespace factor_expression_l209_209775

theorem factor_expression (b : ℝ) : 180 * b ^ 2 + 36 * b = 36 * b * (5 * b + 1) :=
by
  -- actual proof is omitted
  sorry

end factor_expression_l209_209775


namespace range_of_m_l209_209334

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

def tangent_points (m : ℝ) (x₀ : ℝ) : Prop := 
  2 * x₀ ^ 3 - 3 * x₀ ^ 2 + m + 3 = 0

theorem range_of_m (m : ℝ) :
  (∀ x₀, tangent_points m x₀) ∧ m ≠ -2 → (-3 < m ∧ m < -2) :=
sorry

end range_of_m_l209_209334


namespace inequality_result_l209_209347

theorem inequality_result
  (a b : ℝ) 
  (x y : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) :
  x + y ≤ 0 :=
sorry

end inequality_result_l209_209347


namespace find_base_c_l209_209195

theorem find_base_c (c : ℕ) : (c^3 - 7*c^2 - 18*c - 8 = 0) → c = 10 :=
by
  sorry

end find_base_c_l209_209195


namespace find_g_l209_209211

variable (x : ℝ)

-- Given condition
def given_condition (g : ℝ → ℝ) : Prop :=
  5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5

-- Goal
def goal (g : ℝ → ℝ) : Prop :=
  g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3

-- The statement combining given condition and goal to prove
theorem find_g (g : ℝ → ℝ) (h : given_condition x g) : goal x g :=
by
  sorry

end find_g_l209_209211


namespace max_earth_to_sun_distance_l209_209763

-- Define the semi-major axis a and semi-focal distance c
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_focal_distance : ℝ := 3 * 10^6

-- Define the maximum distance from the Earth to the Sun
def max_distance (a c : ℝ) : ℝ := a + c

-- Define the Lean statement to be proved
theorem max_earth_to_sun_distance :
  max_distance semi_major_axis semi_focal_distance = 1.53 * 10^8 :=
by
  -- skipping the proof for now
  sorry

end max_earth_to_sun_distance_l209_209763


namespace avg_temp_Brookdale_l209_209684

noncomputable def avg_temp (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

theorem avg_temp_Brookdale : avg_temp [51, 67, 64, 61, 50, 65, 47] = 57.9 :=
by
  sorry

end avg_temp_Brookdale_l209_209684


namespace find_k_l209_209836

theorem find_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h_nz : ∀ n, S n = n ^ 2 - a n) 
  (hSk : 1 < S k ∧ S k < 9) :
  k = 2 := 
sorry

end find_k_l209_209836


namespace find_initial_mangoes_l209_209459

-- Define the initial conditions
def initial_apples : Nat := 7
def initial_oranges : Nat := 8
def apples_taken : Nat := 2
def oranges_taken : Nat := 2 * apples_taken
def remaining_fruits : Nat := 14
def mangoes_remaining (M : Nat) : Nat := M / 3

-- Define the problem statement
theorem find_initial_mangoes (M : Nat) (hM : 7 - apples_taken + 8 - oranges_taken + mangoes_remaining M = remaining_fruits) : M = 15 :=
by
  sorry

end find_initial_mangoes_l209_209459


namespace number_of_valid_grids_l209_209047

-- Define the concept of a grid and the necessary properties
structure Grid (n : ℕ) :=
  (cells: Fin (n * n) → ℕ)
  (unique: Function.Injective cells)
  (ordered_rows: ∀ i j : Fin n, i < j → cells ⟨i * n + j, sorry⟩ > cells ⟨i * n + j - 1, sorry⟩)
  (ordered_columns: ∀ i j : Fin n, i < j → cells ⟨j * n + i, sorry⟩ > cells ⟨(j - 1) * n + i, sorry⟩)

-- Define the 4x4 grid
def grid_4x4 := Grid 4

-- Statement of the problem: prove there are 2 valid grid_4x4 configurations
theorem number_of_valid_grids : ∃ g : grid_4x4, (∃ g1 g2 : grid_4x4, (g1 ≠ g2) ∧ (∀ g3 : grid_4x4, g3 = g1 ∨ g3 = g2)) :=
  sorry

end number_of_valid_grids_l209_209047


namespace probability_at_least_one_defective_is_correct_l209_209768

/-- Define a box containing 21 bulbs, 4 of which are defective -/
def total_bulbs : ℕ := 21
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs

/-- Define probabilities of choosing non-defective bulbs -/
def prob_first_non_defective : ℚ := non_defective_bulbs / total_bulbs
def prob_second_non_defective : ℚ := (non_defective_bulbs - 1) / (total_bulbs - 1)

/-- Calculate the probability of both bulbs being non-defective -/
def prob_both_non_defective : ℚ := prob_first_non_defective * prob_second_non_defective

/-- Calculate the probability of at least one defective bulb -/
def prob_at_least_one_defective : ℚ := 1 - prob_both_non_defective

theorem probability_at_least_one_defective_is_correct :
  prob_at_least_one_defective = 37 / 105 :=
by
  -- Sorry allows us to skip the proof
  sorry

end probability_at_least_one_defective_is_correct_l209_209768


namespace required_additional_coins_l209_209627

-- Summing up to the first 15 natural numbers
def sum_first_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

-- Given: Alex has 15 friends and 90 coins
def number_of_friends := 15
def initial_coins := 90

-- The total number of coins required
def total_coins_required := sum_first_natural_numbers number_of_friends

-- Calculate the additional coins needed
theorem required_additional_coins : total_coins_required - initial_coins = 30 :=
by
  -- Placeholder for proof
  sorry

end required_additional_coins_l209_209627


namespace initial_average_weight_l209_209494

theorem initial_average_weight (a b c d e : ℝ) (A : ℝ) 
    (h1 : (a + b + c) / 3 = A) 
    (h2 : (a + b + c + d) / 4 = 80) 
    (h3 : e = d + 3) 
    (h4 : (b + c + d + e) / 4 = 79) 
    (h5 : a = 75) : A = 84 :=
sorry

end initial_average_weight_l209_209494


namespace total_votes_l209_209373

theorem total_votes (A B C D E : ℕ)
  (votes_A : ℕ) (votes_B : ℕ) (votes_C : ℕ) (votes_D : ℕ) (votes_E : ℕ)
  (dist_A : votes_A = 38 * A / 100)
  (dist_B : votes_B = 28 * B / 100)
  (dist_C : votes_C = 11 * C / 100)
  (dist_D : votes_D = 15 * D / 100)
  (dist_E : votes_E = 8 * E / 100)
  (redistrib_A : votes_A' = votes_A + 5 * A / 100)
  (redistrib_B : votes_B' = votes_B + 5 * B / 100)
  (redistrib_D : votes_D' = votes_D + 2 * D / 100)
  (total_A : votes_A' = 7320) :
  A = 17023 := 
sorry

end total_votes_l209_209373


namespace cyclic_identity_l209_209468

theorem cyclic_identity (a b c : ℝ) :
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) =
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) =
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) := by
sorry

end cyclic_identity_l209_209468


namespace eat_cereal_in_time_l209_209964

noncomputable def time_to_eat_pounds (pounds : ℕ) (rate1 rate2 : ℚ) :=
  pounds / (rate1 + rate2)

theorem eat_cereal_in_time :
  time_to_eat_pounds 5 ((1:ℚ)/15) ((1:ℚ)/40) = 600/11 := 
by 
  sorry

end eat_cereal_in_time_l209_209964


namespace functional_eq_solution_l209_209231

-- Define the conditions
variables (f g : ℕ → ℕ)

-- Define the main theorem
theorem functional_eq_solution :
  (∀ n : ℕ, f n + f (n + g n) = f (n + 1)) →
  ( (∀ n, f n = 0) ∨ 
    (∃ (n₀ c : ℕ), 
      (∀ n < n₀, f n = 0) ∧ 
      (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
      (∀ n < n₀ - 1, ∃ ck : ℕ, g n = ck) ∧
      g (n₀ - 1) = 1 ∧
      ∀ n ≥ n₀, g n = 0 ) ) := 
by
  intro h
  /- Proof goes here -/
  sorry

end functional_eq_solution_l209_209231


namespace solve_problem_l209_209073

open Real

noncomputable def problem_statement : ℝ :=
  2 * log (sqrt 2) + (log 5 / log 2) * log 2

theorem solve_problem : problem_statement = 1 := by
  sorry

end solve_problem_l209_209073


namespace water_temp_increase_per_minute_l209_209504

theorem water_temp_increase_per_minute :
  ∀ (initial_temp final_temp total_time pasta_time mixing_ratio : ℝ),
    initial_temp = 41 →
    final_temp = 212 →
    total_time = 73 →
    pasta_time = 12 →
    mixing_ratio = (1 / 3) →
    ((final_temp - initial_temp) / (total_time - pasta_time - (mixing_ratio * pasta_time)) = 3) :=
by
  intros initial_temp final_temp total_time pasta_time mixing_ratio
  sorry

end water_temp_increase_per_minute_l209_209504


namespace Oliver_9th_l209_209741

def person := ℕ → Prop

axiom Ruby : person
axiom Oliver : person
axiom Quinn : person
axiom Pedro : person
axiom Nina : person
axiom Samuel : person
axiom place : person → ℕ → Prop

-- Conditions given in the problem
axiom Ruby_Oliver : ∀ n, place Ruby n → place Oliver (n + 7)
axiom Quinn_Pedro : ∀ n, place Quinn n → place Pedro (n - 2)
axiom Nina_Oliver : ∀ n, place Nina n → place Oliver (n + 3)
axiom Pedro_Samuel : ∀ n, place Pedro n → place Samuel (n - 3)
axiom Samuel_Ruby : ∀ n, place Samuel n → place Ruby (n + 2)
axiom Quinn_5th : place Quinn 5

-- Question: Prove that Oliver finished in 9th place
theorem Oliver_9th : place Oliver 9 :=
sorry

end Oliver_9th_l209_209741


namespace option_b_is_same_type_l209_209107

def polynomial_same_type (p1 p2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y, (p1 x y = 1 → p2 x y = 1) ∧ (p2 x y = 1 → p1 x y = 1)

def ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0

def a_squared_b (a b : ℕ) := if a = 2 ∧ b = 1 then 1 else 0
def negative_two_ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0
def ab (a b : ℕ) := if a = 1 ∧ b = 1 then 1 else 0
def ab_squared_c (a b c : ℕ) := if a = 1 ∧ b = 2 ∧ c = 1 then 1 else 0

theorem option_b_is_same_type : polynomial_same_type ab_squared negative_two_ab_squared :=
by
  sorry

end option_b_is_same_type_l209_209107


namespace volume_invariant_l209_209984

noncomputable def volume_of_common_region (a b c : ℝ) : ℝ := (5/6) * a * b * c

theorem volume_invariant (a b c : ℝ) (P : ℝ × ℝ × ℝ) (hP : ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧ 0 ≤ z ∧ z ≤ c) :
  volume_of_common_region a b c = (5/6) * a * b * c :=
by sorry

end volume_invariant_l209_209984


namespace trajectory_of_C_l209_209552

-- Definitions of points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 3)

-- Definition of point C as a linear combination of points A and B
def C (α β : ℝ) : ℝ × ℝ := (α * A.1 + β * B.1, α * A.2 + β * B.2)

-- The main theorem statement to prove the equation of the trajectory of point C
theorem trajectory_of_C (x y α β : ℝ)
  (h_cond : α + β = 1)
  (h_C : (x, y) = C α β) : 
  x + 2*y = 5 := 
sorry -- Proof to be skipped

end trajectory_of_C_l209_209552


namespace bob_distance_when_meet_l209_209360

theorem bob_distance_when_meet (total_distance : ℕ) (yolanda_speed : ℕ) (bob_speed : ℕ) 
    (yolanda_additional_distance : ℕ) (t : ℕ) :
    total_distance = 31 ∧ yolanda_speed = 3 ∧ bob_speed = 4 ∧ yolanda_additional_distance = 3 
    ∧ 7 * t = 28 → 4 * t = 16 := by
    sorry

end bob_distance_when_meet_l209_209360


namespace coeff_x5_of_expansion_l209_209156

theorem coeff_x5_of_expansion : 
  (Polynomial.coeff ((Polynomial.C (1 : ℤ)) * (Polynomial.X ^ 2 - Polynomial.X - Polynomial.C 2) ^ 3) 5) = -3 := 
by sorry

end coeff_x5_of_expansion_l209_209156


namespace gas_pressure_inversely_proportional_l209_209513

variable {T : Type} [Nonempty T]

theorem gas_pressure_inversely_proportional
  (P : T → ℝ) (V : T → ℝ)
  (h_inv : ∀ t, P t * V t = 24) -- Given that pressure * volume = k where k = 24
  (t₀ t₁ : T)
  (hV₀ : V t₀ = 3) (hP₀ : P t₀ = 8) -- Initial condition: volume = 3 liters, pressure = 8 kPa
  (hV₁ : V t₁ = 6) -- New condition: volume = 6 liters
  : P t₁ = 4 := -- We need to prove that the new pressure is 4 kPa
by 
  sorry

end gas_pressure_inversely_proportional_l209_209513


namespace trig_identity_l209_209398

theorem trig_identity (α : ℝ) :
  1 - Real.cos (2 * α - Real.pi) + Real.cos (4 * α - 2 * Real.pi) =
  4 * Real.cos (2 * α) * Real.cos (Real.pi / 6 + α) * Real.cos (Real.pi / 6 - α) :=
by
  sorry

end trig_identity_l209_209398


namespace find_analytical_expression_and_a_l209_209658

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

theorem find_analytical_expression_and_a :
  (A > 0) → (ω > 0) → (0 < φ ∧ φ < π / 2) →
  (∀ x, ∃ k : ℤ, f (x + k * π / 2) = f (x)) →
  (∃ A, ∀ x, A * sin (ω * x + φ) ≤ 2) →
  ((∀ x, f (x - π / 6) = -f (-x + π / 6)) ∨ f 0 = sqrt 3 ∨ (∃ x, 2 * x + φ = k * π + π / 2)) →
  (∀ x, f x = 2 * sin (2 * x + π / 3)) ∧
  (∀ (A : ℝ), (0 < A ∧ A < π) → (f A = sqrt 3) →
  (c = 3 ∧ S = 3 * sqrt 3) →
  (a ^ 2 = ((4 * sqrt 3) ^ 2 + 3 ^ 2 - 2 * (4 * sqrt 3) * 3 * cos (π / 6))) → a = sqrt 21) :=
  sorry

end find_analytical_expression_and_a_l209_209658


namespace complex_pure_imaginary_is_x_eq_2_l209_209542

theorem complex_pure_imaginary_is_x_eq_2
  (x : ℝ)
  (z : ℂ)
  (h : z = ⟨x^2 - 3 * x + 2, x - 1⟩)
  (pure_imaginary : z.re = 0) :
  x = 2 :=
by
  sorry

end complex_pure_imaginary_is_x_eq_2_l209_209542


namespace gcd_30_45_is_15_l209_209514

theorem gcd_30_45_is_15 : Nat.gcd 30 45 = 15 := by
  sorry

end gcd_30_45_is_15_l209_209514


namespace B_pow_five_l209_209472

def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![2, 3], ![4, 6]]
  
theorem B_pow_five : 
  B^5 = (4096 : ℝ) • B + (0 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end B_pow_five_l209_209472


namespace larger_number_of_two_integers_l209_209462

theorem larger_number_of_two_integers (x y : ℤ) (h1 : x * y = 30) (h2 : x + y = 13) : (max x y = 10) :=
by
  sorry

end larger_number_of_two_integers_l209_209462


namespace point_on_coordinate_axes_l209_209013

theorem point_on_coordinate_axes (x y : ℝ) (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by sorry

end point_on_coordinate_axes_l209_209013


namespace no_such_function_l209_209580

theorem no_such_function : ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := 
sorry

end no_such_function_l209_209580


namespace otto_knives_l209_209569

theorem otto_knives (n : ℕ) (cost : ℕ) : 
  cost = 32 → 
  (n ≥ 1 → cost = 5 + ((min (n - 1) 3) * 4) + ((max 0 (n - 4)) * 3)) → 
  n = 9 :=
by
  intros h_cost h_structure
  sorry

end otto_knives_l209_209569


namespace product_of_cubes_91_l209_209918

theorem product_of_cubes_91 :
  ∃ (a b : ℤ), (a = 3 ∨ a = 4) ∧ (b = 3 ∨ b = 4) ∧ (a^3 + b^3 = 91) ∧ (a * b = 12) :=
by
  sorry

end product_of_cubes_91_l209_209918


namespace quadratic_equals_binomial_square_l209_209977

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ b : ℝ, (x^2 + 60 * x + d) = (x + b)^2) → d = 900 :=
by
  sorry

end quadratic_equals_binomial_square_l209_209977


namespace technicians_count_l209_209594

def avg_salary_all := 9500
def avg_salary_technicians := 12000
def avg_salary_rest := 6000
def total_workers := 12

theorem technicians_count : 
  ∃ (T R : ℕ), 
  (T + R = total_workers) ∧ 
  ((T * avg_salary_technicians + R * avg_salary_rest) / total_workers = avg_salary_all) ∧ 
  (T = 7) :=
by sorry

end technicians_count_l209_209594


namespace correct_average_l209_209808

-- Define the conditions given in the problem
def avg_incorrect : ℕ := 46 -- incorrect average
def n : ℕ := 10 -- number of values
def incorrect_num : ℕ := 25
def correct_num : ℕ := 75
def diff : ℕ := correct_num - incorrect_num

-- Define the total sums
def total_incorrect : ℕ := avg_incorrect * n
def total_correct : ℕ := total_incorrect + diff

-- Define the correct average
def avg_correct : ℕ := total_correct / n

-- Statement in Lean 4
theorem correct_average :
  avg_correct = 51 :=
by
  -- We expect users to fill the proof here
  sorry

end correct_average_l209_209808


namespace sum_of_gcd_and_lcm_l209_209632

-- Definitions of gcd and lcm for the conditions
def gcd_of_42_and_56 : ℕ := Nat.gcd 42 56
def lcm_of_24_and_18 : ℕ := Nat.lcm 24 18

-- Lean statement that the sum of the gcd and lcm is 86
theorem sum_of_gcd_and_lcm : gcd_of_42_and_56 + lcm_of_24_and_18 = 86 := by
  sorry

end sum_of_gcd_and_lcm_l209_209632


namespace number_of_points_l209_209994

theorem number_of_points (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 :=
by
  -- Proof to be done here
  sorry

end number_of_points_l209_209994


namespace percentage_reduction_l209_209259

theorem percentage_reduction (P : ℝ) (h1 : 700 / P + 3 = 700 / 70) : 
  ((P - 70) / P) * 100 = 30 :=
by
  sorry

end percentage_reduction_l209_209259


namespace max_value_of_expression_l209_209371

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 8) : 
  (1 + x) * (1 + y) ≤ 25 :=
by
  sorry

end max_value_of_expression_l209_209371


namespace largest_quantity_l209_209308

theorem largest_quantity 
  (A := (2010 / 2009) + (2010 / 2011))
  (B := (2012 / 2011) + (2010 / 2011))
  (C := (2011 / 2010) + (2011 / 2012)) : C > A ∧ C > B := 
by {
  sorry
}

end largest_quantity_l209_209308


namespace perpendicular_lines_condition_l209_209692

theorem perpendicular_lines_condition (k : ℝ) : 
  (k = 5 → (∃ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 ∧ x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 → (k = 5 ∨ k = -1)) :=
sorry

end perpendicular_lines_condition_l209_209692


namespace first_term_geometric_sequence_l209_209031

theorem first_term_geometric_sequence (a r : ℚ) 
  (h3 : a * r^(3-1) = 24)
  (h4 : a * r^(4-1) = 36) :
  a = 32 / 3 :=
by
  sorry

end first_term_geometric_sequence_l209_209031


namespace largest_integer_condition_l209_209535

theorem largest_integer_condition (x : ℤ) : (x/3 + 3/4 : ℚ) < 7/3 → x ≤ 4 :=
by
  sorry

end largest_integer_condition_l209_209535


namespace race_time_A_l209_209397

theorem race_time_A (v t : ℝ) (h1 : 1000 = v * t) (h2 : 950 = v * (t - 10)) : t = 200 :=
by
  sorry

end race_time_A_l209_209397


namespace find_B_and_C_l209_209802

def values_of_B_and_C (B C : ℤ) : Prop :=
  5 * B - 3 = 32 ∧ 2 * B + 2 * C = 18

theorem find_B_and_C : ∃ B C : ℤ, values_of_B_and_C B C ∧ B = 7 ∧ C = 2 := by
  sorry

end find_B_and_C_l209_209802


namespace triangle_inradius_l209_209675

theorem triangle_inradius (A s r : ℝ) (h₁ : A = 3 * s) (h₂ : A = r * s) (h₃ : s ≠ 0) : r = 3 :=
by
  -- Proof omitted
  sorry

end triangle_inradius_l209_209675


namespace mike_average_points_per_game_l209_209420

theorem mike_average_points_per_game (total_points games_played points_per_game : ℕ) 
  (h1 : games_played = 6) 
  (h2 : total_points = 24) 
  (h3 : total_points = games_played * points_per_game) : 
  points_per_game = 4 :=
by
  rw [h1, h2] at h3  -- Substitute conditions h1 and h2 into the equation
  sorry  -- the proof goes here

end mike_average_points_per_game_l209_209420


namespace find_train_probability_l209_209236

-- Define the time range and parameters
def start_time : ℕ := 120
def end_time : ℕ := 240
def wait_time : ℕ := 30

-- Define the conditions
def is_in_range (t : ℕ) : Prop := start_time ≤ t ∧ t ≤ end_time

-- Define the probability function
def probability_of_finding_train : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 30 * 30
  let area_parallelogram : ℚ := 90 * 30
  let shaded_area : ℚ := area_triangle + area_parallelogram
  let total_area : ℚ := (end_time - start_time) * (end_time - start_time)
  shaded_area / total_area

-- The theorem to prove
theorem find_train_probability :
  probability_of_finding_train = 7 / 32 :=
by
  sorry

end find_train_probability_l209_209236


namespace sin_alpha_minus_beta_l209_209734

variables (α β : ℝ)

theorem sin_alpha_minus_beta (h1 : (Real.tan α / Real.tan β) = 7 / 13) 
    (h2 : Real.sin (α + β) = 2 / 3) :
    Real.sin (α - β) = -1 / 5 := 
sorry

end sin_alpha_minus_beta_l209_209734


namespace range_of_x_l209_209240

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) : x ≤ -2 ∨ x ≥ 3 :=
sorry

end range_of_x_l209_209240


namespace smallest_k_l209_209532

-- Define p as the largest prime number with 2023 digits
def p : ℕ := sorry -- This represents the largest prime number with 2023 digits

-- Define the target k
def k : ℕ := 1

-- The theorem stating that k is the smallest positive integer such that p^2 - k is divisible by 30
theorem smallest_k (p_largest_prime : ∀ m : ℕ, m ≤ p → Nat.Prime m → m = p) 
  (p_digits : 10^2022 ≤ p ∧ p < 10^2023) : 
  ∀ n : ℕ, n > 0 → (p^2 - n) % 30 = 0 → n = k :=
by 
  sorry

end smallest_k_l209_209532


namespace total_squares_in_4x4_grid_l209_209432

-- Define the grid size
def grid_size : ℕ := 4

-- Define a function to count the number of k x k squares in an n x n grid
def count_squares (n k : ℕ) : ℕ :=
  (n - k + 1) * (n - k + 1)

-- Total number of squares in a 4 x 4 grid
def total_squares (n : ℕ) : ℕ :=
  count_squares n 1 + count_squares n 2 + count_squares n 3 + count_squares n 4

-- The main theorem asserting the total number of squares in a 4 x 4 grid is 30
theorem total_squares_in_4x4_grid : total_squares grid_size = 30 := by
  sorry

end total_squares_in_4x4_grid_l209_209432


namespace days_kept_first_book_l209_209953

def cost_per_day : ℝ := 0.50
def total_days_in_may : ℝ := 31
def total_cost_paid : ℝ := 41

theorem days_kept_first_book (x : ℝ) : 0.50 * x + 2 * (0.50 * 31) = 41 → x = 20 :=
by sorry

end days_kept_first_book_l209_209953


namespace number_called_2009th_position_l209_209948

theorem number_called_2009th_position :
  let sequence := [1, 2, 3, 4, 3, 2]
  ∃ n, n = 2009 → sequence[(2009 % 6) - 1] = 3 := 
by
  -- let sequence := [1, 2, 3, 4, 3, 2]
  -- 2009 % 6 = 5
  -- sequence[4] = 3
  sorry

end number_called_2009th_position_l209_209948


namespace sum_of_x_and_y_l209_209506

theorem sum_of_x_and_y (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hprod : x * y = 555) : x + y = 52 :=
by
  sorry

end sum_of_x_and_y_l209_209506


namespace keiko_speed_calc_l209_209543

noncomputable def keiko_speed (r : ℝ) (time_diff : ℝ) : ℝ :=
  let circumference_diff := 2 * Real.pi * 8
  circumference_diff / time_diff

theorem keiko_speed_calc (r : ℝ) (time_diff : ℝ) :
  keiko_speed r 48 = Real.pi / 3 := by
  sorry

end keiko_speed_calc_l209_209543


namespace hydras_never_die_l209_209309

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l209_209309


namespace polynomial_roots_l209_209469

theorem polynomial_roots :
  ∀ (x : ℝ), (x^3 - x^2 - 6 * x + 8 = 0) ↔ (x = 2 ∨ x = (-1 + Real.sqrt 17) / 2 ∨ x = (-1 - Real.sqrt 17) / 2) :=
by
  sorry

end polynomial_roots_l209_209469


namespace range_of_t_l209_209214

theorem range_of_t (a b t : ℝ) (h1 : a * (-1)^2 + b * (-1) + 1 / 2 = 0)
    (h2 : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x^2 + b * x + 1 / 2))
    (h3 : t = 2 * a + b) : 
    -1 < t ∧ t < 1 / 2 :=
  sorry

end range_of_t_l209_209214


namespace both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l209_209905

variable (p q : Prop)

-- 1. Both shots were unsuccessful
theorem both_shots_unsuccessful : ¬p ∧ ¬q := sorry

-- 2. Both shots were successful
theorem both_shots_successful : p ∧ q := sorry

-- 3. Exactly one shot was successful
theorem exactly_one_shot_successful : (¬p ∧ q) ∨ (p ∧ ¬q) := sorry

-- 4. At least one shot was successful
theorem at_least_one_shot_successful : p ∨ q := sorry

-- 5. At most one shot was successful
theorem at_most_one_shot_successful : ¬(p ∧ q) := sorry

end both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l209_209905


namespace find_f_x_l209_209246

def f (x : ℝ) : ℝ := sorry

theorem find_f_x (x : ℝ) (h : 2 * f x - f (-x) = 3 * x) : f x = x := 
by sorry

end find_f_x_l209_209246


namespace total_time_spent_l209_209348

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end total_time_spent_l209_209348


namespace pens_given_to_sharon_l209_209060

def initial_pens : Nat := 20
def mikes_pens : Nat := 22
def final_pens : Nat := 65

def total_pens_after_mike : Nat := initial_pens + mikes_pens
def total_pens_after_cindy : Nat := total_pens_after_mike * 2

theorem pens_given_to_sharon :
  total_pens_after_cindy - final_pens = 19 :=
by
  sorry

end pens_given_to_sharon_l209_209060


namespace find_N_l209_209467

variable (N : ℚ)
variable (p : ℚ)

def ball_probability_same_color 
  (green1 : ℚ) (total1 : ℚ) 
  (green2 : ℚ) (blue2 : ℚ) 
  (p : ℚ) : Prop :=
  (green1/total1) * (green2 / (green2 + blue2)) + 
  ((total1 - green1) / total1) * (blue2 / (green2 + blue2)) = p

theorem find_N :
  p = 0.65 → 
  ball_probability_same_color 5 12 20 N p → 
  N = 280 / 311 := 
by
  sorry

end find_N_l209_209467


namespace gcd_consecutive_triplets_l209_209665

theorem gcd_consecutive_triplets : ∀ i : ℕ, 1 ≤ i → gcd (i * (i + 1) * (i + 2)) 6 = 6 :=
by
  sorry

end gcd_consecutive_triplets_l209_209665


namespace sum_x_y_m_l209_209702

theorem sum_x_y_m (a b x y m : ℕ) (ha : a - b = 3) (hx : x = 10 * a + b) (hy : y = 10 * b + a) (hxy : x^2 - y^2 = m^2) : x + y + m = 178 := sorry

end sum_x_y_m_l209_209702


namespace problem_I_l209_209666

theorem problem_I (x m : ℝ) (h1 : |x - m| < 1) (h2 : (1/3 : ℝ) < x ∧ x < (1/2 : ℝ)) : (-1/2 : ℝ) ≤ m ∧ m ≤ (4/3 : ℝ) :=
sorry

end problem_I_l209_209666


namespace hypotenuse_length_l209_209092

-- Define the conditions
def right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- State the theorem using the conditions and correct answer
theorem hypotenuse_length : right_triangle 20 21 29 :=
by
  -- To be filled in by proof steps
  sorry

end hypotenuse_length_l209_209092


namespace max_product_ge_993_squared_l209_209180

theorem max_product_ge_993_squared (a : Fin 1985 → Fin 1985) (hperm : ∀ n : Fin 1985, ∃ k : Fin 1985, a k = n ∧ ∃ m : Fin 1985, a m = n) :
  ∃ k : Fin 1985, a k * k ≥ 993^2 :=
sorry

end max_product_ge_993_squared_l209_209180


namespace product_divisible_by_60_l209_209340

theorem product_divisible_by_60 {a : ℤ} : 
  60 ∣ ((a^2 - 1) * a^2 * (a^2 + 1)) := 
by sorry

end product_divisible_by_60_l209_209340


namespace walkways_area_l209_209199

-- Define the conditions and prove the total walkway area is 416 square feet
theorem walkways_area (rows : ℕ) (columns : ℕ) (bed_width : ℝ) (bed_height : ℝ) (walkway_width : ℝ) 
  (h_rows : rows = 4) (h_columns : columns = 3) (h_bed_width : bed_width = 8) (h_bed_height : bed_height = 3) (h_walkway_width : walkway_width = 2) : 
  (rows * (bed_height + walkway_width) + walkway_width) * (columns * (bed_width + walkway_width) + walkway_width) - rows * columns * bed_width * bed_height = 416 := 
by 
  sorry

end walkways_area_l209_209199


namespace largest_last_digit_in_string_l209_209825

theorem largest_last_digit_in_string :
  ∃ (s : Nat → Fin 10), 
    (s 0 = 1) ∧ 
    (∀ k, k < 99 → (∃ m, (s k * 10 + s (k + 1)) = 17 * m ∨ (s k * 10 + s (k + 1)) = 23 * m)) ∧
    (∃ l, l < 10 ∧ (s 99 = l)) ∧
    (forall last, (last < 10 ∧ (s 99 = last))) ∧
    (∀ m n, s 99 = m → s 99 = n → m ≤ n → n = 9) :=
sorry

end largest_last_digit_in_string_l209_209825


namespace total_first_tier_college_applicants_l209_209545

theorem total_first_tier_college_applicants
  (total_students : ℕ)
  (sample_size : ℕ)
  (sample_applicants : ℕ)
  (total_applicants : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 150)
  (h3 : sample_applicants = 60)
  : total_applicants = 400 :=
sorry

end total_first_tier_college_applicants_l209_209545


namespace checkerboard_corners_sum_l209_209985

theorem checkerboard_corners_sum : 
  let N : ℕ := 9 
  let corners := [1, 9, 73, 81]
  (corners.sum = 164) := by
  sorry

end checkerboard_corners_sum_l209_209985


namespace range_of_z_l209_209050

theorem range_of_z (x y : ℝ) (hx1 : x - 2 * y + 1 ≥ 0) (hx2 : y ≥ x) (hx3 : x ≥ 0) :
  ∃ z, z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 2 :=
by
  sorry

end range_of_z_l209_209050


namespace solve_expression_l209_209478

theorem solve_expression :
  2^3 + 2 * 5 - 3 + 6 = 21 :=
by
  sorry

end solve_expression_l209_209478


namespace july_savings_l209_209765

theorem july_savings (january: ℕ := 100) (total_savings: ℕ := 12700) :
  let february := 2 * january
  let march := 2 * february
  let april := 2 * march
  let may := 2 * april
  let june := 2 * may
  let july := 2 * june
  let total := january + february + march + april + may + june + july
  total = total_savings → july = 6400 := 
by
  sorry

end july_savings_l209_209765


namespace desired_depth_l209_209205

-- Define the given conditions
def men_hours_30m (d : ℕ) : ℕ := 18 * 8 * d
def men_hours_Dm (d1 : ℕ) (D : ℕ) : ℕ := 40 * 6 * d1

-- Define the proportion
def proportion (d d1 : ℕ) (D : ℕ) : Prop :=
  (men_hours_30m d) / 30 = (men_hours_Dm d1 D) / D

-- The main theorem to prove the desired depth
theorem desired_depth (d d1 : ℕ) (H : proportion d d1 50) : 50 = 50 :=
by sorry

end desired_depth_l209_209205


namespace instantaneous_speed_at_3_l209_209034

noncomputable def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

theorem instantaneous_speed_at_3 : deriv s 3 = 11 :=
by
  sorry

end instantaneous_speed_at_3_l209_209034


namespace birth_death_rate_interval_l209_209778

theorem birth_death_rate_interval
  (b_rate : ℕ) (d_rate : ℕ) (population_increase_one_day : ℕ) (seconds_in_one_day : ℕ)
  (net_increase_per_t_seconds : ℕ) (t : ℕ)
  (h1 : b_rate = 5)
  (h2 : d_rate = 3)
  (h3 : population_increase_one_day = 86400)
  (h4 : seconds_in_one_day = 86400)
  (h5 : net_increase_per_t_seconds = b_rate - d_rate)
  (h6 : population_increase_one_day = net_increase_per_t_seconds * (seconds_in_one_day / t)) :
  t = 2 :=
by
  sorry

end birth_death_rate_interval_l209_209778


namespace test_methods_first_last_test_methods_within_six_l209_209943

open Classical

def perms (n k : ℕ) : ℕ := sorry -- placeholder for permutation function

theorem test_methods_first_last
  (prod_total : ℕ) (defective : ℕ) (first_test : ℕ) (last_test : ℕ) 
  (A4_2 : ℕ) (A5_2 : ℕ) (A6_4 : ℕ) : first_test = 2 → last_test = 8 → 
  perms 4 2 * perms 5 2 * perms 6 4 = A4_2 * A5_2 * A6_4 :=
by
  intro h_first_test h_last_test
  simp [perms]
  sorry

theorem test_methods_within_six
  (prod_total : ℕ) (defective : ℕ) 
  (A4_4 : ℕ) (A4_3_A6_1 : ℕ) (A5_3_A6_2 : ℕ) (A6_6 : ℕ)
  : perms 4 4 + 4 * perms 4 3 * perms 6 1 + 4 * perms 5 3 * perms 6 2 + perms 6 6 
  = A4_4 + 4 * A4_3_A6_1 + 4 * A5_3_A6_2 + A6_6 :=
by
  simp [perms]
  sorry

end test_methods_first_last_test_methods_within_six_l209_209943


namespace opposite_numbers_l209_209391

theorem opposite_numbers
  (odot otimes : ℝ)
  (x y : ℝ)
  (h1 : 6 * x + odot * y = 3)
  (h2 : 2 * x + otimes * y = -1)
  (h_add : 6 * x + odot * y + (2 * x + otimes * y) = 2) :
  odot + otimes = 0 := by
  sorry

end opposite_numbers_l209_209391


namespace sarah_score_is_122_l209_209276

-- Define the problem parameters and state the theorem
theorem sarah_score_is_122 (s g : ℝ)
  (h1 : s = g + 40)
  (h2 : (s + g) / 2 = 102) :
  s = 122 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end sarah_score_is_122_l209_209276


namespace range_of_a_l209_209746

open Set

noncomputable def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1}

theorem range_of_a :
  (∀ x, x ∈ B a → x ∈ A) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l209_209746


namespace transport_cost_expression_and_min_cost_l209_209916

noncomputable def total_transport_cost (x : ℕ) (a : ℕ) : ℕ :=
if 2 ≤ a ∧ a ≤ 6 then (5 - a) * x + 23200 else 0

theorem transport_cost_expression_and_min_cost :
  ∀ x : ℕ, ∀ a : ℕ,
  (100 ≤ x ∧ x ≤ 800) →
  (2 ≤ a ∧ a ≤ 6) →
  (total_transport_cost x a = 5 * x + 23200) ∧ 
  (a = 6 → total_transport_cost 800 a = 22400) :=
by
  intros
  -- Provide the detailed proof here.
  sorry

end transport_cost_expression_and_min_cost_l209_209916


namespace math_team_count_l209_209267

open Nat

theorem math_team_count :
  let girls := 7
  let boys := 12
  let total_team := 16
  let count_ways (n k : ℕ) := choose n k
  (count_ways girls 3) * (count_ways boys 5) * (count_ways (girls - 3 + boys - 5) 8) = 456660 :=
by
  sorry

end math_team_count_l209_209267


namespace triangle_inequality_from_condition_l209_209942

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end triangle_inequality_from_condition_l209_209942


namespace complete_square_monomials_l209_209638

theorem complete_square_monomials (x : ℝ) :
  ∃ (m : ℝ), (m = 4 * x ^ 4 ∨ m = 4 * x ∨ m = -4 * x ∨ m = -1 ∨ m = -4 * x ^ 2) ∧
              (∃ (a b : ℝ), (4 * x ^ 2 + 1 + m = a ^ 2 + b ^ 2)) :=
sorry

-- Note: The exact formulation of the problem might vary based on the definition
-- of perfect squares and corresponding polynomials in the Lean environment.

end complete_square_monomials_l209_209638


namespace find_distance_PF2_l209_209892

-- Define the properties of the hyperbola
def is_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- Define the property that P lies on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  is_hyperbola P.1 P.2

-- Define foci of the hyperbola
structure foci (F1 F2 : ℝ × ℝ) : Prop :=
(F1_prop : F1 = (2, 0))
(F2_prop : F2 = (-2, 0))

-- Given distance from P to F1
def distance_PF1 (P F1 : ℝ × ℝ) (d : ℝ) : Prop :=
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = d^2

-- The goal is to find the distance |PF2|
theorem find_distance_PF2 (P F1 F2 : ℝ × ℝ) (D1 D2 : ℝ) :
  point_on_hyperbola P →
  foci F1 F2 →
  distance_PF1 P F1 3 →
  D2 - 3 = 4 →
  D2 = 7 :=
by
  intros hP hFoci hDIST hEQ
  -- Proof can be provided here
  sorry

end find_distance_PF2_l209_209892


namespace problem_proof_l209_209806

-- Define I, J, and K respectively to be 9^20, 3^41, 3
def I : ℕ := 9^20
def J : ℕ := 3^41
def K : ℕ := 3

theorem problem_proof : I + I + I = J := by
  -- Lean structure placeholder
  sorry

end problem_proof_l209_209806


namespace value_of_f_neg2011_l209_209707

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem value_of_f_neg2011 (a b : ℝ) (h : f 2011 a b = 10) : f (-2011) a b = -14 := by
  sorry

end value_of_f_neg2011_l209_209707


namespace least_non_lucky_multiple_of_10_l209_209841

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

theorem least_non_lucky_multiple_of_10 : 
  ∃ n : ℕ, n % 10 = 0 ∧ ¬is_lucky n ∧ (∀ m : ℕ, m % 10 = 0 ∧ ¬is_lucky m → m ≥ n) ∧ n = 110 :=
by
  sorry

end least_non_lucky_multiple_of_10_l209_209841


namespace parabola_standard_eq_l209_209303

theorem parabola_standard_eq (p : ℝ) (x y : ℝ) :
  (∃ x y, 3 * x - 4 * y - 12 = 0) →
  ( (p = 6 ∧ x^2 = -12 * y ∧ y = -3) ∨ (p = 8 ∧ y^2 = 16 * x ∧ x = 4)) :=
sorry

end parabola_standard_eq_l209_209303


namespace sample_size_is_correct_l209_209143

-- Define the conditions
def num_classes := 40
def students_per_class := 50
def selected_students := 150

-- Define the statement to prove the sample size
theorem sample_size_is_correct : selected_students = 150 := by 
  -- Proof is skipped with sorry
  sorry

end sample_size_is_correct_l209_209143


namespace log_order_preservation_l209_209345

theorem log_order_preservation {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (Real.log a > Real.log b) → (a > b) :=
by
  sorry

end log_order_preservation_l209_209345


namespace zachary_seventh_day_cans_l209_209278

-- Define the number of cans found by Zachary every day.
def cans_found_on (day : ℕ) : ℕ :=
  if day = 1 then 4
  else if day = 2 then 9
  else if day = 3 then 14
  else 5 * (day - 1) - 1

-- The theorem to prove the number of cans found on the seventh day.
theorem zachary_seventh_day_cans : cans_found_on 7 = 34 :=
by 
  sorry

end zachary_seventh_day_cans_l209_209278


namespace walnuts_left_in_burrow_l209_209192

-- Define the initial quantities
def boy_initial_walnuts : Nat := 6
def boy_dropped_walnuts : Nat := 1
def initial_burrow_walnuts : Nat := 12
def girl_added_walnuts : Nat := 5
def girl_eaten_walnuts : Nat := 2

-- Define the resulting quantity and the proof goal
theorem walnuts_left_in_burrow : boy_initial_walnuts - boy_dropped_walnuts + initial_burrow_walnuts + girl_added_walnuts - girl_eaten_walnuts = 20 :=
by
  sorry

end walnuts_left_in_burrow_l209_209192


namespace probability_of_drawing_red_ball_l209_209721

theorem probability_of_drawing_red_ball (total_balls red_balls white_balls: ℕ) 
    (h1 : total_balls = 5) 
    (h2 : red_balls = 2) 
    (h3 : white_balls = 3) : 
    (red_balls : ℚ) / total_balls = 2 / 5 := 
by 
    sorry

end probability_of_drawing_red_ball_l209_209721


namespace woman_born_1892_l209_209824

theorem woman_born_1892 (y : ℕ) (hy : 1850 ≤ y^2 - y ∧ y^2 - y < 1900) : y = 44 :=
by
  sorry

end woman_born_1892_l209_209824


namespace probability_of_earning_2400_l209_209785

noncomputable def spinner_labels := ["Bankrupt", "$700", "$900", "$200", "$3000", "$800"]
noncomputable def total_possibilities := (spinner_labels.length : ℕ) ^ 3
noncomputable def favorable_outcomes := 6

theorem probability_of_earning_2400 :
  (favorable_outcomes : ℚ) / total_possibilities = 1 / 36 := by
  sorry

end probability_of_earning_2400_l209_209785


namespace probability_and_relationship_l209_209460

noncomputable def companyA_total : ℕ := 240 + 20
noncomputable def companyA_ontime : ℕ := 240
noncomputable def companyA_ontime_prob : ℚ := companyA_ontime / companyA_total

noncomputable def companyB_total : ℕ := 210 + 30
noncomputable def companyB_ontime : ℕ := 210
noncomputable def companyB_ontime_prob : ℚ := companyB_ontime / companyB_total

noncomputable def total_buses_surveyed : ℕ := 500
noncomputable def total_ontime_buses : ℕ := 450
noncomputable def total_not_ontime_buses : ℕ := 50
noncomputable def K2 : ℚ := (total_buses_surveyed * ((240 * 30 - 210 * 20)^2)) / (260 * 240 * 450 * 50)

theorem probability_and_relationship :
  companyA_ontime_prob = 12 / 13 ∧
  companyB_ontime_prob = 7 / 8 ∧
  K2 > 2.706 :=
by 
  sorry

end probability_and_relationship_l209_209460


namespace even_function_expression_l209_209646

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x then 2*x + 1 else -2*x + 1

theorem even_function_expression (x : ℝ) (hx : x < 0) :
  f x = -2*x + 1 :=
by sorry

end even_function_expression_l209_209646


namespace log_2_bounds_l209_209435

theorem log_2_bounds:
  (2^9 = 512) → (2^8 = 256) → (10^2 = 100) → (10^3 = 1000) → 
  (2 / 9 < Real.log 2 / Real.log 10) ∧ (Real.log 2 / Real.log 10 < 3 / 8) :=
by
  intros h1 h2 h3 h4
  sorry

end log_2_bounds_l209_209435


namespace tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l209_209730

open Real

axiom sin_add_half_pi_div_4_eq_zero (α : ℝ) : 
  sin (α + π / 4) + 2 * sin (α - π / 4) = 0

axiom tan_sub_half_pi_div_4_eq_inv_3 (β : ℝ) : 
  tan (π / 4 - β) = 1 / 3

theorem tan_alpha_eq_inv_3 (α : ℝ) (h : sin (α + π / 4) + 2 * sin (α - π / 4) = 0) : 
  tan α = 1 / 3 := sorry

theorem tan_alpha_add_beta_eq_1 (α β : ℝ) 
  (h1 : tan α = 1 / 3) (h2 : tan (π / 4 - β) = 1 / 3) : 
  tan (α + β) = 1 := sorry

end tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l209_209730


namespace newspaper_spending_over_8_weeks_l209_209663

theorem newspaper_spending_over_8_weeks :
  (3 * 0.50 + 2.00) * 8 = 28 := by
  sorry

end newspaper_spending_over_8_weeks_l209_209663


namespace calculate_amount_after_two_years_l209_209325

noncomputable def amount_after_years (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + rate) ^ years

theorem calculate_amount_after_two_years :
  amount_after_years 51200 0.125 2 = 64800 :=
by
  sorry

end calculate_amount_after_two_years_l209_209325


namespace percentage_of_children_who_speak_only_english_l209_209485

theorem percentage_of_children_who_speak_only_english :
  (∃ (total_children both_languages hindi_speaking only_english : ℝ),
    total_children = 60 ∧
    both_languages = 0.20 * total_children ∧
    hindi_speaking = 42 ∧
    only_english = total_children - (hindi_speaking - both_languages + both_languages) ∧
    (only_english / total_children) * 100 = 30) :=
  sorry

end percentage_of_children_who_speak_only_english_l209_209485


namespace find_c_l209_209868

theorem find_c (c : ℝ) :
  (∀ x y : ℝ, 2*x^2 - 4*c*x*y + (2*c^2 + 1)*y^2 - 2*x - 6*y + 9 ≥ 0) ↔ c = 1/6 :=
by
  sorry

end find_c_l209_209868


namespace intersection_A_B_l209_209480

/-- Definitions for the sets A and B --/
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

-- Theorem statement regarding the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {1} :=
by sorry

end intersection_A_B_l209_209480


namespace linear_eq_represents_plane_l209_209085

theorem linear_eq_represents_plane (A B C : ℝ) (h : ¬ (A = 0 ∧ B = 0 ∧ C = 0)) :
  ∃ (P : ℝ × ℝ × ℝ → Prop), (∀ (x y z : ℝ), P (x, y, z) ↔ A * x + B * y + C * z = 0) ∧ 
  (P (0, 0, 0)) :=
by
  -- To be filled in with the proof steps
  sorry

end linear_eq_represents_plane_l209_209085


namespace find_larger_number_l209_209033

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 10) : L = 1636 := 
by
  sorry

end find_larger_number_l209_209033


namespace unique_zero_function_l209_209747

variable (f : ℝ → ℝ)

theorem unique_zero_function (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l209_209747


namespace quadratic_has_real_roots_l209_209578

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l209_209578


namespace Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l209_209166

section
  -- Define the types of participants and the colors
  inductive Participant
  | Ana | Beto | Carolina

  inductive Color
  | blue | green

  -- Define the strategies for each participant
  inductive Strategy
  | guessBlue | guessGreen | pass

  -- Probability calculations for each strategy
  def carolinaStrategyProbability : ℚ := 1 / 8
  def betoStrategyProbability : ℚ := 1 / 2
  def anaStrategyProbability : ℚ := 3 / 4

  -- Statements to prove the probabilities
  theorem Carolina_Winning_Probability :
    carolinaStrategyProbability = 1 / 8 :=
  sorry

  theorem Beto_Winning_Probability :
    betoStrategyProbability = 1 / 2 :=
  sorry

  theorem Ana_Winning_Probability :
    anaStrategyProbability = 3 / 4 :=
  sorry
end

end Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l209_209166


namespace max_value_l209_209712

theorem max_value (y : ℝ) (h : y ≠ 0) : 
  ∃ M, M = 1 / 25 ∧ 
       ∀ y ≠ 0,  ∀ value, value = y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) 
       → value ≤ M :=
sorry

end max_value_l209_209712


namespace traveler_arrangements_l209_209937

theorem traveler_arrangements :
  let travelers := 6
  let rooms := 3
  ∃ (arrangements : Nat), arrangements = 240 := by
  sorry

end traveler_arrangements_l209_209937


namespace feeding_times_per_day_l209_209081

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end feeding_times_per_day_l209_209081


namespace candy_cases_total_l209_209094

theorem candy_cases_total
  (choco_cases lolli_cases : ℕ)
  (h1 : choco_cases = 25)
  (h2 : lolli_cases = 55) : 
  (choco_cases + lolli_cases) = 80 := by
-- The proof is omitted as requested.
sorry

end candy_cases_total_l209_209094


namespace k_is_3_l209_209986

noncomputable def k_solution (k : ℝ) : Prop :=
  k > 1 ∧ (∑' n : ℕ, (n^2 + 3 * n - 2) / k^n = 2)

theorem k_is_3 : ∃ k : ℝ, k_solution k ∧ k = 3 :=
by
  sorry

end k_is_3_l209_209986


namespace different_routes_calculation_l209_209261

-- Definitions for the conditions
def west_blocks := 3
def south_blocks := 2
def east_blocks := 3
def north_blocks := 3

-- Calculation of combinations for the number of sequences
def house_to_sw_corner_routes := Nat.choose (west_blocks + south_blocks) south_blocks
def ne_corner_to_school_routes := Nat.choose (east_blocks + north_blocks) east_blocks

-- Proving the total number of routes
theorem different_routes_calculation : 
  house_to_sw_corner_routes * 1 * ne_corner_to_school_routes = 200 :=
by
  -- Mathematical proof steps (to be filled)
  sorry

end different_routes_calculation_l209_209261


namespace inequality_solution_l209_209783

theorem inequality_solution (x : ℝ) :
  x + 1 ≥ -3 ∧ -2 * (x + 3) > 0 ↔ -4 ≤ x ∧ x < -3 :=
by sorry

end inequality_solution_l209_209783


namespace class_B_has_more_stable_grades_l209_209725

-- Definitions based on conditions
def avg_score_class_A : ℝ := 85
def avg_score_class_B : ℝ := 85
def var_score_class_A : ℝ := 120
def var_score_class_B : ℝ := 90

-- Proving which class has more stable grades (lower variance indicates more stability)
theorem class_B_has_more_stable_grades :
  var_score_class_B < var_score_class_A :=
by
  -- The proof will need to show the given condition and establish the inequality
  sorry

end class_B_has_more_stable_grades_l209_209725


namespace vendor_apples_sold_l209_209004

theorem vendor_apples_sold (x : ℝ) (h : 0.15 * (1 - x / 100) + 0.50 * (1 - x / 100) * 0.85 = 0.23) : x = 60 :=
sorry

end vendor_apples_sold_l209_209004


namespace find_irrational_satisfying_conditions_l209_209651

-- Define a real number x which is irrational
def is_irrational (x : ℝ) : Prop := ¬∃ (q : ℚ), (x : ℝ) = q

-- Define that x satisfies the given conditions
def rational_conditions (x : ℝ) : Prop :=
  (∃ (r1 : ℚ), x^3 - 17 * x = r1) ∧ (∃ (r2 : ℚ), x^2 + 4 * x = r2)

-- The main theorem statement
theorem find_irrational_satisfying_conditions (x : ℝ) 
  (hx_irr : is_irrational x) 
  (hx_cond : rational_conditions x) : x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
by
  sorry

end find_irrational_satisfying_conditions_l209_209651


namespace second_number_is_correct_l209_209788

theorem second_number_is_correct (x : Real) (h : 108^2 + x^2 = 19928) : x = Real.sqrt 8264 :=
by
  sorry

end second_number_is_correct_l209_209788


namespace evaluate_expression_l209_209477

theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2 * x + 2) / x) * ((y^2 + 2 * y + 2) / y) + ((x^2 - 3 * x + 2) / y) * ((y^2 - 3 * y + 2) / x) 
  = 2 * x * y - (x / y) - (y / x) + 13 + 10 / x + 4 / y + 8 / (x * y) :=
by
  sorry

end evaluate_expression_l209_209477


namespace volume_of_one_pizza_piece_l209_209212

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end volume_of_one_pizza_piece_l209_209212


namespace signs_of_x_and_y_l209_209896

theorem signs_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 :=
sorry

end signs_of_x_and_y_l209_209896


namespace sugar_more_than_flour_l209_209132

def flour_needed : Nat := 9
def sugar_needed : Nat := 11
def flour_added : Nat := 4
def sugar_added : Nat := 0

def flour_remaining : Nat := flour_needed - flour_added
def sugar_remaining : Nat := sugar_needed - sugar_added

theorem sugar_more_than_flour : sugar_remaining - flour_remaining = 6 :=
by
  sorry

end sugar_more_than_flour_l209_209132


namespace radius_range_l209_209481

-- Conditions:
-- r1 is the radius of circle O1
-- r2 is the radius of circle O2
-- d is the distance between centers of circles O1 and O2
-- PO1 is the distance from a point P on circle O2 to the center of circle O1

variables (r1 r2 d PO1 : ℝ)

-- Given r1 = 1, d = 5, PO1 = 2
axiom r1_def : r1 = 1
axiom d_def : d = 5
axiom PO1_def : PO1 = 2

-- To prove: 3 ≤ r2 ≤ 7
theorem radius_range (r2 : ℝ) (h : d = 5 ∧ r1 = 1 ∧ PO1 = 2 ∧ (∃ P : ℝ, P = r2)) : 3 ≤ r2 ∧ r2 ≤ 7 :=
by {
  sorry
}

end radius_range_l209_209481


namespace oldest_son_cookies_l209_209126

def youngest_son_cookies : Nat := 2
def total_cookies : Nat := 54
def days : Nat := 9

theorem oldest_son_cookies : ∃ x : Nat, 9 * (x + youngest_son_cookies) = total_cookies ∧ x = 4 := by
  sorry

end oldest_son_cookies_l209_209126


namespace no_positive_integer_solution_l209_209140

theorem no_positive_integer_solution (a b c d : ℕ) (h1 : a^2 + b^2 = c^2 - d^2) (h2 : a * b = c * d) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : false := 
by 
  sorry

end no_positive_integer_solution_l209_209140


namespace smallest_five_sequential_number_greater_than_2000_is_2004_l209_209230

def fiveSequentialNumber (N : ℕ) : Prop :=
  (if 1 ∣ N then 1 else 0) + 
  (if 2 ∣ N then 1 else 0) + 
  (if 3 ∣ N then 1 else 0) + 
  (if 4 ∣ N then 1 else 0) + 
  (if 5 ∣ N then 1 else 0) + 
  (if 6 ∣ N then 1 else 0) + 
  (if 7 ∣ N then 1 else 0) + 
  (if 8 ∣ N then 1 else 0) + 
  (if 9 ∣ N then 1 else 0) ≥ 5

theorem smallest_five_sequential_number_greater_than_2000_is_2004 :
  ∀ N > 2000, fiveSequentialNumber N → N = 2004 :=
by
  intros N hn hfsn
  have hN : N = 2004 := sorry
  exact hN

end smallest_five_sequential_number_greater_than_2000_is_2004_l209_209230


namespace river_flow_rate_l209_209667

theorem river_flow_rate
  (h : ℝ) (h_eq : h = 3)
  (w : ℝ) (w_eq : w = 36)
  (V : ℝ) (V_eq : V = 3600)
  (conversion_factor : ℝ) (conversion_factor_eq : conversion_factor = 3.6) :
  (60 / (w * h)) * conversion_factor = 2 := by
  sorry

end river_flow_rate_l209_209667


namespace quadratic_function_min_value_l209_209501

theorem quadratic_function_min_value (x : ℝ) (y : ℝ) :
  (y = x^2 - 2 * x + 6) →
  (∃ x_min, x_min = 1 ∧ y = (1 : ℝ)^2 - 2 * (1 : ℝ) + 6 ∧ (∀ x, y ≥ x^2 - 2 * x + 6)) :=
by
  sorry

end quadratic_function_min_value_l209_209501


namespace smallest_sum_of_squares_l209_209605

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l209_209605


namespace purchase_price_is_600_l209_209612

open Real

def daily_food_cost : ℝ := 20
def num_days : ℝ := 40
def vaccination_cost : ℝ := 500
def selling_price : ℝ := 2500
def profit : ℝ := 600

def total_food_cost : ℝ := daily_food_cost * num_days
def total_expenses : ℝ := total_food_cost + vaccination_cost
def total_cost : ℝ := selling_price - profit
def purchase_price : ℝ := total_cost - total_expenses

theorem purchase_price_is_600 : purchase_price = 600 := by
  sorry

end purchase_price_is_600_l209_209612


namespace debate_schedule_ways_l209_209036

-- Definitions based on the problem conditions
def east_debaters : Fin 4 := 4
def west_debaters : Fin 4 := 4
def total_debates := east_debaters.val * west_debaters.val
def debates_per_session := 3
def sessions := 5
def rest_debates := total_debates - sessions * debates_per_session

-- Claim that the number of scheduling ways is the given number
theorem debate_schedule_ways : (Nat.factorial total_debates) / ((Nat.factorial debates_per_session) ^ sessions * Nat.factorial rest_debates) = 20922789888000 :=
by
  -- Proof is skipped with sorry
  sorry

end debate_schedule_ways_l209_209036


namespace race_distance_l209_209920

theorem race_distance (a b c : ℝ) (s_A s_B s_C : ℝ) :
  s_A * a = 100 → 
  s_B * a = 95 → 
  s_C * a = 90 → 
  s_B = s_A - 5 → 
  s_C = s_A - 10 → 
  s_C * (s_B / s_A) = 100 → 
  (100 - s_C) = 5 * (5 / 19) :=
sorry

end race_distance_l209_209920


namespace age_of_other_man_l209_209511

variables (A M : ℝ)

theorem age_of_other_man 
  (avg_age_of_men : ℝ)
  (replaced_man_age : ℝ)
  (avg_age_of_women : ℝ)
  (total_age_6_men : 6 * avg_age_of_men = 6 * (avg_age_of_men + 3) - replaced_man_age - M + 2 * avg_age_of_women) :
  M = 44 :=
by
  sorry

end age_of_other_man_l209_209511


namespace M_inter_N_eq_l209_209781

open Set

def M : Set ℝ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 < n ∧ n ≤ 3 }

theorem M_inter_N_eq : M ∩ (coe '' N) = {0, 1} :=
by sorry

end M_inter_N_eq_l209_209781


namespace prism_is_five_sided_l209_209022

-- Definitions based on problem conditions
def prism_faces (total_faces base_faces : Nat) := total_faces = 7 ∧ base_faces = 2

-- Theorem to prove based on the conditions
theorem prism_is_five_sided (total_faces base_faces : Nat) (h : prism_faces total_faces base_faces) : total_faces - base_faces = 5 :=
sorry

end prism_is_five_sided_l209_209022


namespace image_of_center_after_transformations_l209_209803

-- Define the initial center of circle C
def initial_center : ℝ × ℝ := (3, -4)

-- Define a function to reflect a point across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define a function to translate a point by some units left
def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Define the final coordinates after transformations
def final_center : ℝ × ℝ :=
  translate_left (reflect_x_axis initial_center) 5

-- The theorem to prove
theorem image_of_center_after_transformations :
  final_center = (-2, 4) :=
by
  sorry

end image_of_center_after_transformations_l209_209803


namespace volume_of_right_triangle_pyramid_l209_209142

noncomputable def pyramid_volume (H α β : ℝ) : ℝ :=
  (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2)

theorem volume_of_right_triangle_pyramid (H α β : ℝ) (alpha_acute : 0 < α ∧ α < π / 2) (H_pos : 0 < H) (beta_acute : 0 < β ∧ β < π / 2) :
  pyramid_volume H α β = (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2) := 
sorry

end volume_of_right_triangle_pyramid_l209_209142


namespace milk_rate_proof_l209_209796

theorem milk_rate_proof
  (initial_milk : ℕ := 30000)
  (time_pumped_out : ℕ := 4)
  (rate_pumped_out : ℕ := 2880)
  (time_adding_milk : ℕ := 7)
  (final_milk : ℕ := 28980) :
  ((final_milk - (initial_milk - time_pumped_out * rate_pumped_out)) / time_adding_milk = 1500) :=
by {
  sorry
}

end milk_rate_proof_l209_209796


namespace student_thought_six_is_seven_l209_209641

theorem student_thought_six_is_seven
  (n : ℕ → ℕ)
  (h1 : (n 1 + n 3) / 2 = 2)
  (h2 : (n 2 + n 4) / 2 = 3)
  (h3 : (n 3 + n 5) / 2 = 4)
  (h4 : (n 4 + n 6) / 2 = 5)
  (h5 : (n 5 + n 7) / 2 = 6)
  (h6 : (n 6 + n 8) / 2 = 7)
  (h7 : (n 7 + n 9) / 2 = 8)
  (h8 : (n 8 + n 10) / 2 = 9)
  (h9 : (n 9 + n 1) / 2 = 10)
  (h10 : (n 10 + n 2) / 2 = 1) : 
  n 6 = 7 := 
  sorry

end student_thought_six_is_seven_l209_209641


namespace find_constants_eq_l209_209394

theorem find_constants_eq (P Q R : ℚ)
  (h : ∀ x, (x^2 - 5) = P * (x - 4) * (x - 6) + Q * (x - 1) * (x - 6) + R * (x - 1) * (x - 4)) :
  (P = -4 / 15) ∧ (Q = -11 / 6) ∧ (R = 31 / 10) :=
by
  sorry

end find_constants_eq_l209_209394


namespace profit_share_difference_l209_209588

noncomputable def ratio (x y : ℕ) : ℕ := x / Nat.gcd x y

def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1900
def total_parts : ℕ := 15  -- Sum of the ratio parts (4 for A, 5 for B, 6 for C)
def part_amount : ℕ := profit_share_B / 5  -- 5 parts of B

def profit_share_A : ℕ := 4 * part_amount
def profit_share_C : ℕ := 6 * part_amount

theorem profit_share_difference :
  (profit_share_C - profit_share_A) = 760 := by
  sorry

end profit_share_difference_l209_209588


namespace complex_number_solution_l209_209447

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i * z = 1) : z = -i :=
by sorry

end complex_number_solution_l209_209447


namespace find_f1_l209_209440

variable {R : Type*} [LinearOrderedField R]

-- Define function f of the form px + q
def f (p q x : R) : R := p * x + q

-- Given conditions
variables (p q : R)

-- Define the equations from given conditions
def cond1 : Prop := (f p q 3) = 5
def cond2 : Prop := (f p q 5) = 9

theorem find_f1 (hpq1 : cond1 p q) (hpq2 : cond2 p q) : f p q 1 = 1 := sorry

end find_f1_l209_209440


namespace cone_lateral_surface_area_l209_209959

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end cone_lateral_surface_area_l209_209959


namespace alok_paid_rs_811_l209_209974

/-
 Assume Alok ordered the following items at the given prices:
 - 16 chapatis, each costing Rs. 6
 - 5 plates of rice, each costing Rs. 45
 - 7 plates of mixed vegetable, each costing Rs. 70
 - 6 ice-cream cups

 Prove that the total cost Alok paid is Rs. 811.
-/
theorem alok_paid_rs_811 :
  let chapati_cost := 6
  let rice_plate_cost := 45
  let mixed_vegetable_plate_cost := 70
  let chapatis := 16 * chapati_cost
  let rice_plates := 5 * rice_plate_cost
  let mixed_vegetable_plates := 7 * mixed_vegetable_plate_cost
  chapatis + rice_plates + mixed_vegetable_plates = 811 := by
  sorry

end alok_paid_rs_811_l209_209974


namespace complement_of_A_in_U_l209_209560

def U : Set ℕ := {1, 2, 3, 4}

def satisfies_inequality (x : ℕ) : Prop := x^2 - 5 * x + 4 < 0

def A : Set ℕ := {x | satisfies_inequality x}

theorem complement_of_A_in_U : U \ A = {1, 4} :=
by
  -- Proof omitted.
  sorry

end complement_of_A_in_U_l209_209560


namespace odd_checkerboard_cannot_be_covered_by_dominoes_l209_209558

theorem odd_checkerboard_cannot_be_covered_by_dominoes 
    (m n : ℕ) (h : (m * n) % 2 = 1) :
    ¬ ∃ (dominos : Finset (Fin 2 × Fin 2)),
    ∀ {i j : Fin 2}, (i, j) ∈ dominos → 
    ((i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0)) ∧ 
    dominos.card = (m * n) / 2 := sorry

end odd_checkerboard_cannot_be_covered_by_dominoes_l209_209558


namespace father_gave_8_candies_to_Billy_l209_209382

theorem father_gave_8_candies_to_Billy (candies_Billy : ℕ) (candies_Caleb : ℕ) (candies_Andy : ℕ) (candies_father : ℕ) 
  (candies_given_to_Caleb : ℕ) (candies_more_than_Caleb : ℕ) (candies_given_by_father_total : ℕ) :
  (candies_given_to_Caleb = 11) →
  (candies_Caleb = 11) →
  (candies_Andy = 9) →
  (candies_father = 36) →
  (candies_Andy = candies_Caleb + 4) →
  (candies_given_by_father_total = candies_given_to_Caleb + (candies_Andy - 9)) →
  (candies_father - candies_given_by_father_total = 8) →
  candies_Billy = 8 := 
by
  intros
  sorry

end father_gave_8_candies_to_Billy_l209_209382


namespace horizontal_asymptote_degree_l209_209263

noncomputable def degree (p : Polynomial ℝ) : ℕ := Polynomial.natDegree p

theorem horizontal_asymptote_degree (p : Polynomial ℝ) :
  (∃ l : ℝ, ∀ ε > 0, ∃ N, ∀ x > N, |(p.eval x / (3 * x^7 - 2 * x^3 + x - 4)) - l| < ε) →
  degree p ≤ 7 :=
sorry

end horizontal_asymptote_degree_l209_209263


namespace hyperbola_foci_distance_l209_209108

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - 4 * x - 9 * y^2 - 18 * y = 56

-- Define the distance between the foci of the hyperbola
def distance_between_foci (d : ℝ) : Prop :=
  d = 2 * Real.sqrt (170 / 3)

-- The theorem stating that the distance between the foci of the given hyperbola
theorem hyperbola_foci_distance :
  ∃ d, hyperbola_eq x y → distance_between_foci d :=
by { sorry }

end hyperbola_foci_distance_l209_209108


namespace sum_of_fractions_l209_209673

theorem sum_of_fractions : 
  (1/12 + 2/12 + 3/12 + 4/12 + 5/12 + 6/12 + 7/12 + 8/12 + 9/12 + 65/12 + 3/4) = 119 / 12 :=
by
  sorry

end sum_of_fractions_l209_209673


namespace probability_both_selected_l209_209931

-- Given conditions
def jamie_probability : ℚ := 2 / 3
def tom_probability : ℚ := 5 / 7

-- Statement to prove
theorem probability_both_selected :
  jamie_probability * tom_probability = 10 / 21 :=
by
  sorry

end probability_both_selected_l209_209931


namespace no_bijective_function_l209_209704

open Set

def is_bijective {α β : Type*} (f : α → β) : Prop :=
  Function.Bijective f

def are_collinear {P : Type*} (A B C : P) : Prop :=
  sorry -- placeholder for the collinearity predicate on points

def are_parallel_or_concurrent {L : Type*} (l₁ l₂ l₃ : L) : Prop :=
  sorry -- placeholder for the condition that lines are parallel or concurrent

theorem no_bijective_function (P : Type*) (D : Type*) :
  ¬ ∃ (f : P → D), is_bijective f ∧
    ∀ A B C : P, are_collinear A B C → are_parallel_or_concurrent (f A) (f B) (f C) :=
by
  sorry

end no_bijective_function_l209_209704


namespace shari_effective_distance_l209_209301

-- Define the given conditions
def constant_rate : ℝ := 4 -- miles per hour
def wind_resistance : ℝ := 0.5 -- miles per hour
def walking_time : ℝ := 2 -- hours

-- Define the effective walking speed considering wind resistance
def effective_speed : ℝ := constant_rate - wind_resistance

-- Define the effective walking distance
def effective_distance : ℝ := effective_speed * walking_time

-- State that Shari effectively walks 7.0 miles
theorem shari_effective_distance :
  effective_distance = 7.0 :=
by
  sorry

end shari_effective_distance_l209_209301


namespace range_of_m_l209_209222

-- Definition of the quadratic function
def quadratic_function (m x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + 1

-- Statement of the proof problem in Lean
theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, 0 ≤ x ∧ x ≤ 5 → quadratic_function m x ≥ quadratic_function m (x + 1)) ↔ m ≤ -8 :=
by
  sorry

end range_of_m_l209_209222


namespace value_of_expression_l209_209052

theorem value_of_expression (m n : ℝ) (h : m + n = -2) : 5 * m^2 + 5 * n^2 + 10 * m * n = 20 := 
by
  sorry

end value_of_expression_l209_209052


namespace f_800_l209_209755

noncomputable def f : ℕ → ℕ := sorry

axiom axiom1 : ∀ x y : ℕ, 0 < x → 0 < y → f (x * y) = f x + f y
axiom axiom2 : f 10 = 10
axiom axiom3 : f 40 = 14

theorem f_800 : f 800 = 26 :=
by
  -- Apply the conditions here
  sorry

end f_800_l209_209755


namespace cubing_identity_l209_209790

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l209_209790


namespace ratio_of_female_contestants_l209_209960

theorem ratio_of_female_contestants (T M F : ℕ) (hT : T = 18) (hM : M = 12) (hF : F = T - M) :
  F / T = 1 / 3 :=
by
  sorry

end ratio_of_female_contestants_l209_209960


namespace greatest_is_B_l209_209983

def A : ℕ := 95 - 35
def B : ℕ := A + 12
def C : ℕ := B - 19

theorem greatest_is_B : B = 72 ∧ (B > A ∧ B > C) :=
by {
  -- Proof steps would be written here to prove the theorem.
  sorry
}

end greatest_is_B_l209_209983


namespace rectangle_length_is_16_l209_209237

noncomputable def rectangle_length (b : ℝ) (c : ℝ) : ℝ :=
  let pi := Real.pi
  let full_circle_circumference := 2 * c
  let radius := full_circle_circumference / (2 * pi)
  let diameter := 2 * radius
  let side_length_of_square := diameter
  let perimeter_of_square := 4 * side_length_of_square
  let perimeter_of_rectangle := perimeter_of_square
  let length_of_rectangle := (perimeter_of_rectangle / 2) - b
  length_of_rectangle

theorem rectangle_length_is_16 :
  rectangle_length 14 23.56 = 16 :=
by
  sorry

end rectangle_length_is_16_l209_209237


namespace remaining_amount_l209_209342

def initial_amount : ℕ := 18
def spent_amount : ℕ := 16

theorem remaining_amount : initial_amount - spent_amount = 2 := 
by sorry

end remaining_amount_l209_209342


namespace flight_duration_l209_209509

theorem flight_duration (takeoff landing : ℕ) (h : ℕ) (m : ℕ)
  (h0 : takeoff = 11 * 60 + 7)
  (h1 : landing = 2 * 60 + 49 + 12 * 60)
  (h2 : 0 < m) (h3 : m < 60) :
  h + m = 45 := 
sorry

end flight_duration_l209_209509


namespace number_of_zeros_of_quadratic_function_l209_209482

-- Given the quadratic function y = x^2 + x - 1
def quadratic_function (x : ℝ) : ℝ := x^2 + x - 1

-- Prove that the number of zeros of the quadratic function y = x^2 + x - 1 is 2
theorem number_of_zeros_of_quadratic_function : 
  ∃ x1 x2 : ℝ, quadratic_function x1 = 0 ∧ quadratic_function x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

end number_of_zeros_of_quadratic_function_l209_209482


namespace remainder_sum_abc_mod5_l209_209075

theorem remainder_sum_abc_mod5 (a b c : ℕ) (h1 : a < 5) (h2 : b < 5) (h3 : c < 5)
  (h4 : a * b * c ≡ 1 [MOD 5])
  (h5 : 4 * c ≡ 3 [MOD 5])
  (h6 : 3 * b ≡ 2 + b [MOD 5]) :
  (a + b + c) % 5 = 1 :=
  sorry

end remainder_sum_abc_mod5_l209_209075


namespace solve_for_y_l209_209018

theorem solve_for_y :
  ∃ (y : ℝ), 
    (∑' n : ℕ, (4 * (n + 1) - 2) * y^n) = 100 ∧ |y| < 1 ∧ y = 0.6036 :=
sorry

end solve_for_y_l209_209018


namespace printing_presses_equivalence_l209_209153

theorem printing_presses_equivalence :
  ∃ P : ℕ, (500000 / 12) / P = (500000 / 14) / 30 ∧ P = 26 :=
by
  sorry

end printing_presses_equivalence_l209_209153


namespace fraction_problem_l209_209087

theorem fraction_problem : 
  (  (1/4 - 1/5) / (1/3 - 1/4)  ) = 3/5 :=
by
  sorry

end fraction_problem_l209_209087


namespace number_of_item_B_l209_209749

theorem number_of_item_B
    (x y z : ℕ)
    (total_items total_cost : ℕ)
    (hx_price : 1 ≤ x ∧ x ≤ 100)
    (hy_price : 1 ≤ y ∧ y ≤ 100)
    (hz_price : 1 ≤ z ∧ z ≤ 100)
    (h_total_items : total_items = 100)
    (h_total_cost : total_cost = 100)
    (h_price_equation : (x / 8) + 10 * y = z)
    (h_item_equation : x + y + (total_items - (x + y)) = total_items)
    : total_items - (x + y) = 21 :=
sorry

end number_of_item_B_l209_209749


namespace prime_roots_range_l209_209762

theorem prime_roots_range (p : ℕ) (hp : Prime p) (h : ∃ x₁ x₂ : ℤ, x₁ + x₂ = -p ∧ x₁ * x₂ = -444 * p) : 31 < p ∧ p ≤ 41 :=
by sorry

end prime_roots_range_l209_209762


namespace quadratic_transform_l209_209137

theorem quadratic_transform (x : ℝ) : x^2 - 6 * x - 5 = 0 → (x - 3)^2 = 14 :=
by
  intro h
  sorry

end quadratic_transform_l209_209137


namespace tangent_line_at_one_e_l209_209505

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one_e : ∀ (x y : ℝ), (x, y) = (1, Real.exp 1) → (y = 2 * Real.exp x * x - Real.exp 1) :=
by
  intro x y h
  sorry

end tangent_line_at_one_e_l209_209505


namespace other_investment_interest_rate_l209_209980

open Real

-- Definitions of the given conditions
def total_investment : ℝ := 22000
def investment_at_8_percent : ℝ := 17000
def total_interest : ℝ := 1710
def interest_rate_8_percent : ℝ := 0.08

-- Derived definitions from the conditions
def other_investment_amount : ℝ := total_investment - investment_at_8_percent
def interest_from_8_percent : ℝ := investment_at_8_percent * interest_rate_8_percent
def interest_from_other : ℝ := total_interest - interest_from_8_percent

-- Proof problem: Prove that the percentage of the other investment is 0.07 (or 7%).
theorem other_investment_interest_rate :
  interest_from_other / other_investment_amount = 0.07 := by
  sorry

end other_investment_interest_rate_l209_209980


namespace quick_calc_formula_l209_209130

variables (a b A B C : ℤ)

theorem quick_calc_formula (h1 : (100 - a) * (100 - b) = (A + B - 100) * 100 + C)
                           (h2 : (100 + a) * (100 + b) = (A + B - 100) * 100 + C) :
  A = 100 ∨ A = 100 ∧ B = 100 ∨ B = 100 ∧ C = a * b :=
sorry

end quick_calc_formula_l209_209130


namespace problem_1_problem_2_l209_209406

-- Definitions for problem (1)
def p (x a : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0

-- Statement for problem (1)
theorem problem_1 (a : ℝ) (h : p 1 a ∧ q x) : 2 < x ∧ x < 3 :=
by 
  sorry

-- Definitions for problem (2)
def neg_p (x a : ℝ) := ¬ (x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0)
def neg_q (x : ℝ) := ¬ (x^2 - x - 6 ≤ 0 ∧ x^2 + 3 * x - 10 > 0)

-- Statement for problem (2)
theorem problem_2 (a : ℝ) (h : ∀ x, neg_p x a → neg_q x ∧ ¬ (neg_q x → neg_p x a)) : 1 < a ∧ a ≤ 2 :=
by 
  sorry

end problem_1_problem_2_l209_209406


namespace arithmetic_sequence_first_term_l209_209525

theorem arithmetic_sequence_first_term (d : ℤ) (a_n a_2 a_9 a_11 : ℤ) 
  (h1 : a_2 = 7) 
  (h2 : a_11 = a_9 + 6)
  (h3 : a_11 = a_n + 10 * d)
  (h4 : a_9 = a_n + 8 * d)
  (h5 : a_2 = a_n + d) :
  a_n = 4 := by
  sorry

end arithmetic_sequence_first_term_l209_209525


namespace hash_nesting_example_l209_209173

def hash (N : ℝ) : ℝ :=
  0.5 * N + 2

theorem hash_nesting_example : hash (hash (hash (hash 20))) = 5 :=
by
  sorry

end hash_nesting_example_l209_209173


namespace not_a_solution_set4_l209_209821

def set1 : ℝ × ℝ := (1, 2)
def set2 : ℝ × ℝ := (2, 0)
def set3 : ℝ × ℝ := (0.5, 3)
def set4 : ℝ × ℝ := (-2, 4)

noncomputable def is_solution (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 = 4

theorem not_a_solution_set4 : ¬ is_solution set4 := 
by 
  sorry

end not_a_solution_set4_l209_209821


namespace total_eggs_michael_has_l209_209425

-- Define the initial number of crates
def initial_crates : ℕ := 6

-- Define the number of crates given to Susan
def crates_given_to_susan : ℕ := 2

-- Define the number of crates bought on Thursday
def crates_bought_thursday : ℕ := 5

-- Define the number of eggs per crate
def eggs_per_crate : ℕ := 30

-- Theorem stating the total number of eggs Michael has now
theorem total_eggs_michael_has :
  (initial_crates - crates_given_to_susan + crates_bought_thursday) * eggs_per_crate = 270 :=
sorry

end total_eggs_michael_has_l209_209425


namespace nine_pow_div_eighty_one_pow_l209_209254

theorem nine_pow_div_eighty_one_pow (a b : ℕ) (h1 : a = 9^2) (h2 : b = a^4) :
  (9^10 / b = 81) := by
  sorry

end nine_pow_div_eighty_one_pow_l209_209254


namespace license_plates_count_l209_209401

theorem license_plates_count :
  let vowels := 5 -- choices for the first vowel
  let other_letters := 25 -- choices for the second and third letters
  let digits := 10 -- choices for each digit
  (vowels * other_letters * other_letters * (digits * digits * digits)) = 3125000 :=
by
  -- proof steps will go here
  sorry

end license_plates_count_l209_209401


namespace range_of_a_l209_209902

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9 / 8 :=
by
  sorry

end range_of_a_l209_209902


namespace rita_money_left_l209_209549

theorem rita_money_left :
  let initial_amount : ℝ := 400
  let cost_short_dresses : ℝ := 5 * (20 - 0.1 * 20)
  let cost_pants : ℝ := 2 * 15
  let cost_jackets : ℝ := 2 * (30 - 0.15 * 30) + 2 * 30
  let cost_skirts : ℝ := 2 * 18 * 0.8
  let cost_tshirts : ℝ := 2 * 8
  let cost_transportation : ℝ := 5
  let total_spent : ℝ := cost_short_dresses + cost_pants + cost_jackets + cost_skirts + cost_tshirts + cost_transportation
  let money_left : ℝ := initial_amount - total_spent
  money_left = 119.2 :=
by 
  sorry

end rita_money_left_l209_209549


namespace pencils_total_l209_209603

/-- The students in class 5A had a total of 2015 pencils. One of them lost a box containing five pencils and replaced it with a box containing 50 pencils. Prove the final number of pencils is 2060. -/
theorem pencils_total {initial_pencils lost_pencils gained_pencils final_pencils : ℕ} 
  (h1 : initial_pencils = 2015) 
  (h2 : lost_pencils = 5) 
  (h3 : gained_pencils = 50) 
  (h4 : final_pencils = (initial_pencils - lost_pencils + gained_pencils)) 
  : final_pencils = 2060 :=
sorry

end pencils_total_l209_209603


namespace length_of_platform_l209_209880

noncomputable def len_train : ℝ := 120
noncomputable def speed_train : ℝ := 60 * (1000 / 3600) -- kmph to m/s
noncomputable def time_cross : ℝ := 15

theorem length_of_platform (L_train : ℝ) (S_train : ℝ) (T_cross : ℝ) (H_train : L_train = len_train)
  (H_speed : S_train = speed_train) (H_time : T_cross = time_cross) : 
  ∃ (L_platform : ℝ), L_platform = (S_train * T_cross) - L_train ∧ L_platform = 130.05 :=
by
  rw [H_train, H_speed, H_time]
  sorry

end length_of_platform_l209_209880


namespace trailing_zeroes_500_fact_l209_209068

theorem trailing_zeroes_500_fact : 
  let count_multiples (n m : ℕ) := n / m 
  let count_5 := count_multiples 500 5
  let count_25 := count_multiples 500 25
  let count_125 := count_multiples 500 125
-- We don't count multiples of 625 because 625 > 500, thus its count is 0. 
-- Therefore: total trailing zeroes = count_5 + count_25 + count_125
  count_5 + count_25 + count_125 = 124 := sorry

end trailing_zeroes_500_fact_l209_209068


namespace hyperbola_equation_l209_209239

noncomputable def focal_distance : ℝ := 10
noncomputable def c : ℝ := 5
noncomputable def point_P : (ℝ × ℝ) := (2, 1)
noncomputable def eq1 : Prop := ∀ (x y : ℝ), (x^2) / 20 - (y^2) / 5 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1
noncomputable def eq2 : Prop := ∀ (x y : ℝ), (y^2) / 5 - (x^2) / 20 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1

theorem hyperbola_equation :
  (∃ a b : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
    (∀ x y : ℝ, (x^2) / a^2 - (y^2) / b^2 = 1) ∨ 
    (∃ a' b' : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
      (∀ x y : ℝ, (y^2) / a'^2 - (x^2) / b'^2 = 1))) :=
by sorry

end hyperbola_equation_l209_209239


namespace box_office_scientific_notation_l209_209084

def billion : ℝ := 10^9
def box_office_revenue : ℝ := 57.44 * billion
def scientific_notation (n : ℝ) : ℝ × ℝ := (5.744, 10^10)

theorem box_office_scientific_notation :
  scientific_notation box_office_revenue = (5.744, 10^10) :=
by
  sorry

end box_office_scientific_notation_l209_209084


namespace probability_hit_10_or_7_ring_probability_below_7_ring_l209_209164

noncomputable def P_hit_10_ring : ℝ := 0.21
noncomputable def P_hit_9_ring : ℝ := 0.23
noncomputable def P_hit_8_ring : ℝ := 0.25
noncomputable def P_hit_7_ring : ℝ := 0.28
noncomputable def P_below_7_ring : ℝ := 0.03

theorem probability_hit_10_or_7_ring :
  P_hit_10_ring + P_hit_7_ring = 0.49 :=
  by sorry

theorem probability_below_7_ring :
  P_below_7_ring = 0.03 :=
  by sorry

end probability_hit_10_or_7_ring_probability_below_7_ring_l209_209164


namespace number_of_restaurants_l209_209029

def first_restaurant_meals_per_day := 20
def second_restaurant_meals_per_day := 40
def third_restaurant_meals_per_day := 50
def total_meals_per_week := 770

theorem number_of_restaurants :
  (first_restaurant_meals_per_day * 7) + 
  (second_restaurant_meals_per_day * 7) + 
  (third_restaurant_meals_per_day * 7) = total_meals_per_week → 
  3 = 3 :=
by 
  intros h
  sorry

end number_of_restaurants_l209_209029


namespace positive_integer_solution_eq_l209_209618

theorem positive_integer_solution_eq :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (xyz + 2 * x + 3 * y + 6 * z = xy + 2 * xz + 3 * yz) ∧ (x, y, z) = (4, 3, 1) := 
by
  sorry

end positive_integer_solution_eq_l209_209618


namespace smallest_n_inequality_l209_209644

theorem smallest_n_inequality :
  ∃ (n : ℕ), ∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧ n = 4 :=
by
  -- Proof steps would go here
  sorry

end smallest_n_inequality_l209_209644


namespace total_fruit_cost_is_173_l209_209194

-- Define the cost of a single orange and a single apple
def orange_cost := 2
def apple_cost := 3
def banana_cost := 1

-- Define the number of fruits each person has
def louis_oranges := 5
def louis_apples := 3

def samantha_oranges := 8
def samantha_apples := 7

def marley_oranges := 2 * louis_oranges
def marley_apples := 3 * samantha_apples

def edward_oranges := 3 * louis_oranges
def edward_bananas := 4

-- Define the cost of fruits for each person
def louis_cost := (louis_oranges * orange_cost) + (louis_apples * apple_cost)
def samantha_cost := (samantha_oranges * orange_cost) + (samantha_apples * apple_cost)
def marley_cost := (marley_oranges * orange_cost) + (marley_apples * apple_cost)
def edward_cost := (edward_oranges * orange_cost) + (edward_bananas * banana_cost)

-- Define the total cost for all four people
def total_cost := louis_cost + samantha_cost + marley_cost + edward_cost

-- Statement to prove that the total cost is $173
theorem total_fruit_cost_is_173 : total_cost = 173 :=
by
  sorry

end total_fruit_cost_is_173_l209_209194


namespace flower_nectar_water_content_l209_209910

/-- Given that to yield 1 kg of honey, 1.6 kg of flower-nectar must be processed,
    and the honey obtained from this nectar contains 20% water,
    prove that the flower-nectar contains 50% water. --/
theorem flower_nectar_water_content :
  (1.6 : ℝ) * (0.2 / 1) = (50 / 100) * (1.6 : ℝ) := by
  sorry

end flower_nectar_water_content_l209_209910


namespace prob_same_gender_eq_two_fifths_l209_209175

-- Define the number of male and female students
def num_male_students : ℕ := 3
def num_female_students : ℕ := 2

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability calculation
def probability_same_gender := (num_male_students * (num_male_students - 1) / 2 + num_female_students * (num_female_students - 1) / 2) / (total_students * (total_students - 1) / 2)

theorem prob_same_gender_eq_two_fifths :
  probability_same_gender = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end prob_same_gender_eq_two_fifths_l209_209175


namespace sin4x_eq_sin2x_solution_set_l209_209385

noncomputable def solution_set (x : ℝ) : Prop :=
  0 < x ∧ x < (3 / 2) * Real.pi ∧ Real.sin (4 * x) = Real.sin (2 * x)

theorem sin4x_eq_sin2x_solution_set :
  { x : ℝ | solution_set x } =
  { (Real.pi / 6), (Real.pi / 2), Real.pi, (5 * Real.pi / 6), (7 * Real.pi / 6) } :=
by
  sorry

end sin4x_eq_sin2x_solution_set_l209_209385


namespace distance_to_place_l209_209508

theorem distance_to_place 
  (row_speed_still_water : ℝ) 
  (current_speed : ℝ) 
  (headwind_speed : ℝ) 
  (tailwind_speed : ℝ) 
  (total_trip_time : ℝ) 
  (htotal_trip_time : total_trip_time = 15) 
  (hrow_speed_still_water : row_speed_still_water = 10) 
  (hcurrent_speed : current_speed = 2) 
  (hheadwind_speed : headwind_speed = 4) 
  (htailwind_speed : tailwind_speed = 4) :
  ∃ (D : ℝ), D = 48 :=
by
  sorry

end distance_to_place_l209_209508


namespace total_money_l209_209526

theorem total_money (n : ℕ) (h1 : n * 3 = 36) :
  let one_rupee := n * 1
  let five_rupee := n * 5
  let ten_rupee := n * 10
  (one_rupee + five_rupee + ten_rupee) = 192 :=
by
  -- Note: The detailed calculations would go here in the proof
  -- Since we don't need to provide the proof, we add sorry to indicate the omitted part
  sorry

end total_money_l209_209526


namespace paint_cost_is_624_rs_l209_209200

-- Given conditions:
-- Length of floor is 21.633307652783934 meters.
-- Length is 200% more than the breadth (i.e., length = 3 * breadth).
-- Cost to paint the floor is Rs. 4 per square meter.

noncomputable def length : ℝ := 21.633307652783934
noncomputable def cost_per_sq_meter : ℝ := 4
noncomputable def breadth : ℝ := length / 3
noncomputable def area : ℝ := length * breadth
noncomputable def total_cost : ℝ := area * cost_per_sq_meter

theorem paint_cost_is_624_rs : total_cost = 624 := by
  sorry

end paint_cost_is_624_rs_l209_209200


namespace divisible_by_9_l209_209787

theorem divisible_by_9 (n : ℕ) : 9 ∣ (4^n + 15 * n - 1) :=
by
  sorry

end divisible_by_9_l209_209787


namespace combinedAverageAge_l209_209758

-- Definitions
def numFifthGraders : ℕ := 50
def avgAgeFifthGraders : ℕ := 10
def numParents : ℕ := 75
def avgAgeParents : ℕ := 40

-- Calculation of total ages
def totalAgeFifthGraders := numFifthGraders * avgAgeFifthGraders
def totalAgeParents := numParents * avgAgeParents
def combinedTotalAge := totalAgeFifthGraders + totalAgeParents

-- Calculation of total number of individuals
def totalIndividuals := numFifthGraders + numParents

-- The claim to prove
theorem combinedAverageAge : 
  combinedTotalAge / totalIndividuals = 28 := by
  -- Skipping the proof details.
  sorry

end combinedAverageAge_l209_209758


namespace initial_caterpillars_l209_209519

theorem initial_caterpillars (C : ℕ) 
    (hatch_eggs : C + 4 - 8 = 10) : C = 14 :=
by
  sorry

end initial_caterpillars_l209_209519


namespace sum_of_coefficients_l209_209396

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (∀ x : ℝ, (3 * x - 2)^6 = a_0 + a_1 * (2 * x - 1) + a_2 * (2 * x - 1)^2 + a_3 * (2 * x - 1)^3 + a_4 * (2 * x - 1)^4 + a_5 * (2 * x - 1)^5 + a_6 * (2 * x - 1)^6) ->
  a_1 + a_3 + a_5 = -63 / 2 := by
  sorry

end sum_of_coefficients_l209_209396


namespace pyramid_edge_length_correct_l209_209812

-- Definitions for the conditions
def total_length (sum_of_edges : ℝ) := sum_of_edges = 14.8
def edges_count (num_of_edges : ℕ) := num_of_edges = 8

-- Definition for the question and corresponding answer to prove
def length_of_one_edge (sum_of_edges : ℝ) (num_of_edges : ℕ) (one_edge_length : ℝ) :=
  sum_of_edges / num_of_edges = one_edge_length

-- The statement that needs to be proven
theorem pyramid_edge_length_correct : total_length 14.8 → edges_count 8 → length_of_one_edge 14.8 8 1.85 :=
by
  intros h1 h2
  sorry

end pyramid_edge_length_correct_l209_209812


namespace value_of_a1_a3_a5_l209_209879

theorem value_of_a1_a3_a5 (a a1 a2 a3 a4 a5 : ℤ) (h : (2 * x + 1) ^ 5 = a + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5) :
  a1 + a3 + a5 = 122 :=
by
  sorry

end value_of_a1_a3_a5_l209_209879


namespace fraction_of_married_men_l209_209883

theorem fraction_of_married_men (num_women : ℕ) (num_single_women : ℕ) (num_married_women : ℕ)
  (num_married_men : ℕ) (total_people : ℕ) 
  (h1 : num_single_women = num_women / 4) 
  (h2 : num_married_women = num_women - num_single_women)
  (h3 : num_married_men = num_married_women) 
  (h4 : total_people = num_women + num_married_men) :
  (num_married_men : ℚ) / (total_people : ℚ) = 3 / 7 := 
by 
  sorry

end fraction_of_married_men_l209_209883


namespace frank_cookies_l209_209782

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end frank_cookies_l209_209782


namespace problem_solution_l209_209773

def grid_side : ℕ := 4
def square_size : ℝ := 2
def ellipse_major_axis : ℝ := 4
def ellipse_minor_axis : ℝ := 2
def circle_radius : ℝ := 1
def num_circles : ℕ := 3

noncomputable def grid_area : ℝ :=
  (grid_side * grid_side) * (square_size * square_size)

noncomputable def circle_area : ℝ :=
  num_circles * (Real.pi * (circle_radius ^ 2))

noncomputable def ellipse_area : ℝ :=
  Real.pi * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)

noncomputable def visible_shaded_area (A B : ℝ) : Prop :=
  grid_area = A - B * Real.pi

theorem problem_solution : ∃ A B, visible_shaded_area A B ∧ (A + B = 69) :=
by
  sorry

end problem_solution_l209_209773


namespace range_of_a_l209_209911

def valid_real_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a

theorem range_of_a :
  (∀ a : ℝ, (¬ valid_real_a a)) ↔ (a < 1 ∨ a > 3) :=
sorry

end range_of_a_l209_209911


namespace playerA_winning_moves_l209_209777

-- Definitions of the game
-- Circles are labeled from 1 to 9
inductive Circle
| A | B | C1 | C2 | C3 | C4 | C5 | C6 | C7

inductive Player
| A | B

def StraightLine (c1 c2 c3 : Circle) : Prop := sorry
-- The straight line property between circles is specified by the game rules

-- Initial conditions
def initial_conditions (playerA_move playerB_move : Circle) : Prop :=
  playerA_move = Circle.A ∧ playerB_move = Circle.B

-- Winning condition
def winning_move (move : Circle) : Prop := sorry
-- This will check if a move leads to a win for Player A

-- Equivalent proof problem
theorem playerA_winning_moves : ∀ (move : Circle), initial_conditions Circle.A Circle.B → 
  (move = Circle.C2 ∨ move = Circle.C3 ∨ move = Circle.C4) → winning_move move :=
by
  sorry

end playerA_winning_moves_l209_209777


namespace geometric_series_sum_l209_209111

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l209_209111


namespace simplify_expression_l209_209144

theorem simplify_expression (a: ℤ) (h₁: a ≠ 0) (h₂: a ≠ 1) (h₃: a ≠ -3) :
  (2 * a = 4) → a = 2 :=
by
  sorry

end simplify_expression_l209_209144


namespace find_a8_l209_209168

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^(n-1)

noncomputable def sum_geom (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem find_a8 (a_1 q a_2 a_5 a_8 : ℝ) (S : ℕ → ℝ) 
  (Hsum : ∀ n, S n = sum_geom a_1 q n)
  (H1 : 2 * S 9 = S 3 + S 6)
  (H2 : a_2 = geometric_sequence a_1 q 2)
  (H3 : a_5 = geometric_sequence a_1 q 5)
  (H4 : a_2 + a_5 = 4)
  (H5 : a_8 = geometric_sequence a_1 q 8) :
  a_8 = 2 :=
sorry

end find_a8_l209_209168


namespace area_of_smaller_circle_l209_209769

/-
  Variables and assumptions:
  r: Radius of the smaller circle
  R: Radius of the larger circle which is three times the smaller circle. Hence, R = 3 * r.
  PA = AB = 6: Lengths of the tangent segments
  Area: Calculated area of the smaller circle
-/

theorem area_of_smaller_circle (r : ℝ) (h1 : 6 = r) (h2 : 3 * 6 = R) (h3 : 6 = r) : 
  ∃ (area : ℝ), area = (36 * Real.pi) / 7 :=
by
  sorry 

end area_of_smaller_circle_l209_209769


namespace company_picnic_l209_209884

theorem company_picnic :
  (20 / 100 * (30 / 100 * 100) + 40 / 100 * (70 / 100 * 100)) / 100 * 100 = 34 := by
  sorry

end company_picnic_l209_209884


namespace sum_of_distances_l209_209100

theorem sum_of_distances (A B C D M P : ℝ × ℝ) 
    (hA : A = (0, 0))
    (hB : B = (4, 0))
    (hC : C = (4, 4))
    (hD : D = (0, 4))
    (hM : M = (2, 0))
    (hP : P = (0, 2)) :
    dist A M + dist A P = 4 :=
by
  sorry

end sum_of_distances_l209_209100


namespace find_fourth_number_l209_209206

theorem find_fourth_number 
  (average : ℝ) 
  (a1 a2 a3 : ℝ) 
  (x : ℝ) 
  (n : ℝ) 
  (h1 : average = 20) 
  (h2 : a1 = 3) 
  (h3 : a2 = 16) 
  (h4 : a3 = 33) 
  (h5 : n = 27) 
  (h_avg : (a1 + a2 + a3 + x) / 4 = average) :
  x = n + 1 :=
by
  sorry

end find_fourth_number_l209_209206


namespace g_at_50_l209_209058

variable (g : ℝ → ℝ)

axiom g_functional_eq (x y : ℝ) : g (x * y) = x * g y
axiom g_at_1 : g 1 = 40

theorem g_at_50 : g 50 = 2000 :=
by
  -- Placeholder for proof
  sorry

end g_at_50_l209_209058


namespace cosine_in_third_quadrant_l209_209642

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l209_209642


namespace xiaoming_pens_l209_209030

theorem xiaoming_pens (P M : ℝ) (hP : P > 0) (hM : M > 0) :
  (M / (7 / 8 * P) - M / P = 13) → (M / P = 91) := 
by
  sorry

end xiaoming_pens_l209_209030


namespace number_of_digits_in_expression_l209_209521

theorem number_of_digits_in_expression : 
  (Nat.digits 10 (2^12 * 5^8)).length = 10 := 
by
  sorry

end number_of_digits_in_expression_l209_209521


namespace each_player_gets_seven_l209_209375

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end each_player_gets_seven_l209_209375


namespace river_flow_rate_l209_209711

-- Define the conditions
def depth : ℝ := 8
def width : ℝ := 25
def volume_per_min : ℝ := 26666.666666666668

-- The main theorem proving the rate at which the river is flowing
theorem river_flow_rate : (volume_per_min / (depth * width)) = 133.33333333333334 := by
  -- Express the area of the river's cross-section
  let area := depth * width
  -- Define the velocity based on the given volume and calculated area
  let velocity := volume_per_min / area
  -- Simplify and derive the result
  show velocity = 133.33333333333334
  sorry

end river_flow_rate_l209_209711


namespace monomial_2023_l209_209728

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^n * (n + 1), n)

theorem monomial_2023 :
  monomial 2023 = (-2024, 2023) :=
by
  sorry

end monomial_2023_l209_209728


namespace orange_pyramid_total_l209_209079

theorem orange_pyramid_total :
  let base_length := 7
  let base_width := 9
  -- layer 1 -> dimensions (7, 9)
  -- layer 2 -> dimensions (6, 8)
  -- layer 3 -> dimensions (5, 6)
  -- layer 4 -> dimensions (4, 5)
  -- layer 5 -> dimensions (3, 3)
  -- layer 6 -> dimensions (2, 2)
  -- layer 7 -> dimensions (1, 1)
  (base_length * base_width) + ((base_length - 1) * (base_width - 1))
  + ((base_length - 2) * (base_width - 3)) + ((base_length - 3) * (base_width - 4))
  + ((base_length - 4) * (base_width - 6)) + ((base_length - 5) * (base_width - 7))
  + ((base_length - 6) * (base_width - 8)) = 175 := sorry

end orange_pyramid_total_l209_209079


namespace fountains_for_m_4_fountains_for_m_3_l209_209027

noncomputable def ceil_div (a b : ℕ) : ℕ :=
  (a + b - 1) / b

-- Problem for m = 4
theorem fountains_for_m_4 (n : ℕ) : ∃ f : ℕ, f = 2 * ceil_div n 3 := 
sorry

-- Problem for m = 3
theorem fountains_for_m_3 (n : ℕ) : ∃ f : ℕ, f = 3 * ceil_div n 3 :=
sorry

end fountains_for_m_4_fountains_for_m_3_l209_209027


namespace total_wolves_l209_209138

theorem total_wolves (x y : ℕ) :
  (x + 2 * y = 20) →
  (4 * x + 3 * y = 55) →
  (x + y = 15) :=
by
  intro h1 h2
  sorry

end total_wolves_l209_209138


namespace first_group_people_count_l209_209026

theorem first_group_people_count (P : ℕ) (W : ℕ) 
  (h1 : P * 3 * W = 3 * W) 
  (h2 : 8 * 3 * W = 8 * W) : 
  P = 3 :=
by
  sorry

end first_group_people_count_l209_209026


namespace first_three_digits_of_quotient_are_239_l209_209339

noncomputable def a : ℝ := 0.12345678910114748495051
noncomputable def b_lower_bound : ℝ := 0.515
noncomputable def b_upper_bound : ℝ := 0.516

theorem first_three_digits_of_quotient_are_239 (b : ℝ) (hb : b_lower_bound < b ∧ b < b_upper_bound) :
    0.239 * b < a ∧ a < 0.24 * b := 
sorry

end first_three_digits_of_quotient_are_239_l209_209339


namespace average_age_of_inhabitants_l209_209784

theorem average_age_of_inhabitants (H M : ℕ) (avg_age_men avg_age_women : ℕ)
  (ratio_condition : 2 * M = 3 * H)
  (men_avg_age_condition : avg_age_men = 37)
  (women_avg_age_condition : avg_age_women = 42) :
  ((H * 37) + (M * 42)) / (H + M) = 40 :=
by
  sorry

end average_age_of_inhabitants_l209_209784


namespace find_m_n_l209_209695

theorem find_m_n (a b : ℝ) (m n : ℤ) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l209_209695


namespace multiply_increase_by_196_l209_209626

theorem multiply_increase_by_196 (x : ℕ) (h : 14 * x = 14 + 196) : x = 15 :=
sorry

end multiply_increase_by_196_l209_209626


namespace geometric_sequence_sum_l209_209007

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n = (a 0) * q^n)
  (h2 : ∀ n, a n > a (n + 1))
  (h3 : a 2 + a 3 + a 4 = 28)
  (h4 : a 3 + 2 = (a 2 + a 4) / 2) :
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 63 :=
by {
  sorry
}

end geometric_sequence_sum_l209_209007


namespace cards_given_to_Jeff_l209_209579

theorem cards_given_to_Jeff
  (initial_cards : ℕ)
  (cards_given_to_John : ℕ)
  (remaining_cards : ℕ)
  (cards_left : ℕ)
  (h_initial : initial_cards = 573)
  (h_given_John : cards_given_to_John = 195)
  (h_left_before_Jeff : remaining_cards = initial_cards - cards_given_to_John)
  (h_final : cards_left = 210)
  (h_given_Jeff : remaining_cards - cards_left = 168) :
  (initial_cards - cards_given_to_John - cards_left = 168) :=
by
  sorry

end cards_given_to_Jeff_l209_209579


namespace three_digit_number_satisfies_conditions_l209_209285

-- Definitions for the digits of the number
def x := 9
def y := 6
def z := 4

-- Define the three-digit number
def number := 100 * x + 10 * y + z

-- Define the conditions
def geometric_progression := y * y = x * z

def reverse_order_condition := (number - 495) = 100 * z + 10 * y + x

def arithmetic_progression := (z - 1) + (x - 2) = 2 * (y - 1)

-- The theorem to prove
theorem three_digit_number_satisfies_conditions :
  geometric_progression ∧ reverse_order_condition ∧ arithmetic_progression :=
by {
  sorry
}

end three_digit_number_satisfies_conditions_l209_209285


namespace min_cost_to_fence_land_l209_209745

theorem min_cost_to_fence_land (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * w ^ 2 ≥ 500) : 
  5 * (2 * (l + w)) = 150 * Real.sqrt 10 := 
by
  sorry

end min_cost_to_fence_land_l209_209745


namespace least_k_for_divisibility_l209_209170

theorem least_k_for_divisibility (k : ℕ) : (k ^ 4) % 1260 = 0 ↔ k ≥ 210 :=
sorry

end least_k_for_divisibility_l209_209170


namespace sharon_highway_speed_l209_209067

theorem sharon_highway_speed:
  ∀ (total_distance : ℝ) (highway_time : ℝ) (city_time: ℝ) (city_speed : ℝ),
  total_distance = 59 → highway_time = 1 / 3 → city_time = 2 / 3 → city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 :=
by
  intro total_distance highway_time city_time city_speed
  intro h_total_distance h_highway_time h_city_time h_city_speed
  rw [h_total_distance, h_highway_time, h_city_time, h_city_speed]
  sorry

end sharon_highway_speed_l209_209067


namespace compose_f_g_f_l209_209719

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

theorem compose_f_g_f (x : ℝ) : f (g (f 3)) = 79 := by
  sorry

end compose_f_g_f_l209_209719


namespace sections_in_orchard_l209_209009

-- Conditions: Farmers harvest 45 sacks from each section daily, 360 sacks are harvested daily
def harvest_sacks_per_section : ℕ := 45
def total_sacks_harvested_daily : ℕ := 360

-- Statement: Prove that the number of sections is 8 given the conditions
theorem sections_in_orchard (h1 : harvest_sacks_per_section = 45) (h2 : total_sacks_harvested_daily = 360) :
  total_sacks_harvested_daily / harvest_sacks_per_section = 8 :=
sorry

end sections_in_orchard_l209_209009


namespace last_digit_of_product_of_consecutive_numbers_l209_209189

theorem last_digit_of_product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h1 : k > 5)
    (h2 : n = (k + 1) * (k + 2) * (k + 3) * (k + 4))
    (h3 : n % 10 ≠ 0) : n % 10 = 4 :=
sorry -- Proof not provided as per instructions.

end last_digit_of_product_of_consecutive_numbers_l209_209189


namespace domain_of_y_l209_209035

noncomputable def domain_of_function (x : ℝ) : Bool :=
  x < 0 ∧ x ≠ -1

theorem domain_of_y :
  {x : ℝ | (∃ y, y = (x + 1) ^ 0 / Real.sqrt (|x| - x)) } =
  {x : ℝ | domain_of_function x} :=
by
  sorry

end domain_of_y_l209_209035


namespace number_of_boys_l209_209629

theorem number_of_boys (n : ℕ) (h1 : (n * 182 - 60) / n = 180): n = 30 :=
by
  sorry

end number_of_boys_l209_209629


namespace sum_of_differences_l209_209020

theorem sum_of_differences (x : ℝ) (h : (45 + x) / 2 = 38) : abs (x - 45) + abs (x - 30) = 15 := by
  sorry

end sum_of_differences_l209_209020


namespace problem_one_problem_two_l209_209639

variables (a₁ a₂ a₃ : ℤ) (n : ℕ)
def arith_sequence : Prop :=
  a₁ + a₂ + a₃ = 21 ∧ a₁ * a₂ * a₃ = 231

theorem problem_one (h : arith_sequence a₁ a₂ a₃) : a₂ = 7 :=
sorry

theorem problem_two (h : arith_sequence a₁ a₂ a₃) :
  (∃ d : ℤ, (d = -4 ∨ d = 4) ∧ (a_n = a₁ + (n - 1) * d ∨ a_n = a₃ + (n - 1) * d)) :=
sorry

end problem_one_problem_two_l209_209639


namespace additional_trams_proof_l209_209187

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l209_209187


namespace square_diff_l209_209682

-- Definitions and conditions from the problem
def three_times_sum_eq (a b : ℝ) : Prop := 3 * (a + b) = 18
def diff_eq (a b : ℝ) : Prop := a - b = 4

-- Goal to prove that a^2 - b^2 = 24 under the given conditions
theorem square_diff (a b : ℝ) (h₁ : three_times_sum_eq a b) (h₂ : diff_eq a b) : a^2 - b^2 = 24 :=
sorry

end square_diff_l209_209682


namespace no_such_triangle_exists_l209_209311

theorem no_such_triangle_exists (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : b = 0.25 * (a + b + c)) :
  ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end no_such_triangle_exists_l209_209311


namespace sufficient_balance_after_29_months_l209_209849

noncomputable def accumulated_sum (S0 : ℕ) (D : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  S0 * (1 + r)^n + D * ((1 + r)^n - 1) / r

theorem sufficient_balance_after_29_months :
  let S0 := 300000
  let D := 15000
  let r := (1 / 100 : ℚ) -- interest rate of 1%
  accumulated_sum S0 D r 29 ≥ 900000 :=
by
  sorry -- The proof will be elaborated later

end sufficient_balance_after_29_months_l209_209849


namespace function_minimum_value_no_maximum_l209_209820

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.sin x + a) / Real.sin x

theorem function_minimum_value_no_maximum (a : ℝ) (h_a : 0 < a) : 
  ∃ x_min, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≥ x_min ∧ 
           (∀ x ∈ Set.Ioo 0 Real.pi, f a x ≠ x_min) ∧ 
           ¬ (∃ x_max, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≤ x_max) :=
by
  let t := Real.sin
  have h : ∀ x ∈ Set.Ioo 0 Real.pi, t x ∈ Set.Ioo 0 1 := sorry -- Simple property of sine function in (0, π)
  -- Exact details skipped to align with the conditions from the problem, leveraging the property
  sorry -- Full proof not required as per instructions

end function_minimum_value_no_maximum_l209_209820


namespace angle_ABC_bisector_l209_209491

theorem angle_ABC_bisector (θ : ℝ) (h : θ / 2 = (1 / 3) * (180 - θ)) : θ = 72 :=
by
  sorry

end angle_ABC_bisector_l209_209491


namespace total_daily_cost_correct_l209_209930

/-- Definition of the daily wages of each type of worker -/
def daily_wage_worker : ℕ := 100
def daily_wage_electrician : ℕ := 2 * daily_wage_worker
def daily_wage_plumber : ℕ := (5 * daily_wage_worker) / 2 -- 2.5 times daily_wage_worker
def daily_wage_architect : ℕ := 7 * daily_wage_worker / 2 -- 3.5 times daily_wage_worker

/-- Definition of the total daily cost for one project -/
def daily_cost_one_project : ℕ :=
  2 * daily_wage_worker +
  daily_wage_electrician +
  daily_wage_plumber +
  daily_wage_architect

/-- Definition of the total daily cost for three projects -/
def total_daily_cost_three_projects : ℕ :=
  3 * daily_cost_one_project

/-- Theorem stating the overall labor costs for one day for all three projects -/
theorem total_daily_cost_correct :
  total_daily_cost_three_projects = 3000 :=
by
  -- Proof omitted
  sorry

end total_daily_cost_correct_l209_209930


namespace mary_puts_back_correct_number_of_oranges_l209_209010

namespace FruitProblem

def price_apple := 40
def price_orange := 60
def total_fruits := 10
def average_price_all := 56
def average_price_kept := 50

theorem mary_puts_back_correct_number_of_oranges :
  ∀ (A O O' T: ℕ),
  A + O = total_fruits →
  A * price_apple + O * price_orange = total_fruits * average_price_all →
  A = 2 →
  T = A + O' →
  A * price_apple + O' * price_orange = T * average_price_kept →
  O - O' = 6 :=
by
  sorry

end FruitProblem

end mary_puts_back_correct_number_of_oranges_l209_209010


namespace maximum_teams_tied_for_most_wins_l209_209328

/-- In a round-robin tournament with 8 teams, each team plays one game
    against each other team, and each game results in one team winning
    and one team losing. -/
theorem maximum_teams_tied_for_most_wins :
  ∀ (teams games wins : ℕ), 
    teams = 8 → 
    games = (teams * (teams - 1)) / 2 →
    wins = 28 →
    ∃ (max_tied_teams : ℕ), max_tied_teams = 5 :=
by
  sorry

end maximum_teams_tied_for_most_wins_l209_209328


namespace parabola_vertex_l209_209670

noncomputable def is_vertex (x y : ℝ) : Prop :=
  y^2 + 8 * y + 4 * x + 5 = 0 ∧ (∀ y₀, y₀^2 + 8 * y₀ + 4 * x + 5 ≥ 0)

theorem parabola_vertex : is_vertex (11 / 4) (-4) :=
by
  sorry

end parabola_vertex_l209_209670


namespace distance_to_pinedale_mall_l209_209927

-- Define the conditions given in the problem
def average_speed : ℕ := 60  -- km/h
def stops_interval : ℕ := 5   -- minutes
def number_of_stops : ℕ := 8

-- The distance from Yahya's house to Pinedale Mall
theorem distance_to_pinedale_mall : 
  (average_speed * (number_of_stops * stops_interval / 60) = 40) :=
by
  sorry

end distance_to_pinedale_mall_l209_209927


namespace total_sales_l209_209866

noncomputable def sales_in_june : ℕ := 96
noncomputable def sales_in_july : ℕ := sales_in_june * 4 / 3

theorem total_sales (june_sales : ℕ) (july_sales : ℕ) (h1 : june_sales = 96)
                    (h2 : july_sales = june_sales * 4 / 3) :
                    june_sales + july_sales = 224 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_sales_l209_209866


namespace difference_of_two_numbers_l209_209538

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l209_209538


namespace mean_of_set_is_12_point_8_l209_209729

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l209_209729


namespace no_such_class_exists_l209_209463

theorem no_such_class_exists : ¬ ∃ (b g : ℕ), (3 * b = 5 * g) ∧ (32 < b + g) ∧ (b + g < 40) :=
by {
  -- Proof goes here
  sorry
}

end no_such_class_exists_l209_209463


namespace train_speed_is_correct_l209_209574

-- Definitions of the given conditions.
def train_length : ℕ := 250
def bridge_length : ℕ := 150
def time_taken : ℕ := 20

-- Definition of the total distance covered by the train.
def total_distance : ℕ := train_length + bridge_length

-- The speed calculation.
def speed : ℕ := total_distance / time_taken

-- The theorem that we need to prove.
theorem train_speed_is_correct : speed = 20 := by
  -- proof steps go here
  sorry

end train_speed_is_correct_l209_209574


namespace alex_pen_difference_l209_209860

theorem alex_pen_difference 
  (alex_initial_pens : Nat) 
  (doubling_rate : Nat) 
  (weeks : Nat) 
  (jane_pens_month : Nat) :
  alex_initial_pens = 4 →
  doubling_rate = 2 →
  weeks = 4 →
  jane_pens_month = 16 →
  (alex_initial_pens * doubling_rate ^ weeks) - jane_pens_month = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end alex_pen_difference_l209_209860


namespace square_side_length_l209_209608

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 2 :=
by 
  sorry

end square_side_length_l209_209608


namespace smallest_N_l209_209635

theorem smallest_N (N : ℕ) : (N * 3 ≥ 75) ∧ (N * 2 < 75) → N = 25 :=
by {
  sorry
}

end smallest_N_l209_209635


namespace rationalize_denominator_l209_209336

theorem rationalize_denominator : 
  let A := -13 
  let B := -9
  let C := 3
  let D := 2
  let E := 165
  let F := 51
  A + B + C + D + E + F = 199 := by
sorry

end rationalize_denominator_l209_209336


namespace range_of_m_l209_209904

open Real

noncomputable def f (x m : ℝ) : ℝ := log x / log 2 + x - m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x m = 0) → 1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l209_209904


namespace contrapositive_true_l209_209161

theorem contrapositive_true (x : ℝ) : (x^2 - 2*x - 8 ≤ 0 → x ≥ -3) :=
by
  -- Proof omitted
  sorry

end contrapositive_true_l209_209161


namespace broker_investment_increase_l209_209352

noncomputable def final_value_stock_A := 
  let initial := 100.0
  let year1 := initial * (1 + 0.80)
  let year2 := year1 * (1 - 0.30)
  year2 * (1 + 0.10)

noncomputable def final_value_stock_B := 
  let initial := 100.0
  let year1 := initial * (1 + 0.50)
  let year2 := year1 * (1 - 0.10)
  year2 * (1 - 0.25)

noncomputable def final_value_stock_C := 
  let initial := 100.0
  let year1 := initial * (1 - 0.30)
  let year2 := year1 * (1 - 0.40)
  year2 * (1 + 0.80)

noncomputable def final_value_stock_D := 
  let initial := 100.0
  let year1 := initial * (1 + 0.40)
  let year2 := year1 * (1 + 0.20)
  year2 * (1 - 0.15)

noncomputable def total_final_value := 
  final_value_stock_A + final_value_stock_B + final_value_stock_C + final_value_stock_D

noncomputable def initial_total_value := 4 * 100.0

noncomputable def net_increase := total_final_value - initial_total_value

noncomputable def net_increase_percentage := (net_increase / initial_total_value) * 100

theorem broker_investment_increase : net_increase_percentage = 14.5625 := 
by
  sorry

end broker_investment_increase_l209_209352


namespace cube_volume_l209_209570

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l209_209570


namespace min_value_is_144_l209_209269

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2

theorem min_value_is_144 (x y z : ℝ) (hxyz : x * y * z = 48) : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ xyz = 48 ∧ min_value_expression x y z = 144 :=
by 
  sorry

end min_value_is_144_l209_209269


namespace divisors_not_multiples_of_14_l209_209913

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 2
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 3
def is_perfect_fifth (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 5
def is_perfect_seventh (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 7

def n : ℕ := 2^2 * 3^3 * 5^5 * 7^7

theorem divisors_not_multiples_of_14 :
  is_perfect_square (n / 2) →
  is_perfect_cube (n / 3) →
  is_perfect_fifth (n / 5) →
  is_perfect_seventh (n / 7) →
  (∃ d : ℕ, d = 240) :=
by
  sorry

end divisors_not_multiples_of_14_l209_209913


namespace find_a_l209_209875

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end find_a_l209_209875


namespace infection_equation_correct_l209_209559

theorem infection_equation_correct (x : ℝ) :
  1 + x + x * (x + 1) = 196 :=
sorry

end infection_equation_correct_l209_209559


namespace choose_president_and_committee_l209_209609

-- Define the condition of the problem
def total_people := 10
def committee_size := 3

-- Define the function to calculate the number of combinations
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Proving the number of ways to choose the president and the committee
theorem choose_president_and_committee :
  (total_people * comb (total_people - 1) committee_size) = 840 :=
by
  sorry

end choose_president_and_committee_l209_209609


namespace units_digit_fraction_l209_209617

theorem units_digit_fraction (h1 : 30 = 2 * 3 * 5) (h2 : 31 = 31) (h3 : 32 = 2^5) 
    (h4 : 33 = 3 * 11) (h5 : 34 = 2 * 17) (h6 : 35 = 5 * 7) (h7 : 7200 = 2^4 * 3^2 * 5^2) :
    ((30 * 31 * 32 * 33 * 34 * 35) / 7200) % 10 = 2 :=
by
  sorry

end units_digit_fraction_l209_209617


namespace number_of_players_taking_mathematics_l209_209319

-- Define the conditions
def total_players := 15
def players_physics := 10
def players_both := 4

-- Define the conclusion to be proven
theorem number_of_players_taking_mathematics : (total_players - players_physics + players_both) = 9 :=
by
  -- Placeholder for proof
  sorry

end number_of_players_taking_mathematics_l209_209319


namespace interest_rate_is_12_percent_l209_209437

-- Definitions
def SI : ℝ := 5400
def P : ℝ := 15000
def T : ℝ := 3

-- Theorem to prove the interest rate
theorem interest_rate_is_12_percent :
  SI = (P * 12 * T) / 100 :=
by
  sorry

end interest_rate_is_12_percent_l209_209437


namespace solution_set_for_inequality_l209_209218

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + (a - b) * x + 1

theorem solution_set_for_inequality (a b : ℝ) (h1 : 2*a + 4 = -(a-1)) :
  ∀ x : ℝ, (f x a b > f b a b) ↔ ((x ∈ Set.Icc (-2 : ℝ) (2 : ℝ)) ∧ ((x < -1 ∨ 1 < x))) :=
by
  sorry

end solution_set_for_inequality_l209_209218


namespace inequality_general_l209_209086

theorem inequality_general {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 :=
by
  sorry

end inequality_general_l209_209086


namespace duty_person_C_l209_209456

/-- Given amounts of money held by three persons and a total custom duty,
    prove that the duty person C should pay is 17 when payments are proportional. -/
theorem duty_person_C (money_A money_B money_C total_duty : ℕ) (total_money : ℕ)
  (hA : money_A = 560) (hB : money_B = 350) (hC : money_C = 180) (hD : total_duty = 100)
  (hT : total_money = money_A + money_B + money_C) :
  total_duty * money_C / total_money = 17 :=
by
  -- proof goes here
  sorry

end duty_person_C_l209_209456


namespace geometric_sequence_first_term_l209_209065

theorem geometric_sequence_first_term (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 * a 2 * a 3 = 27) (h3 : a 6 = 27) : a 0 = 1 :=
by
  sorry

end geometric_sequence_first_term_l209_209065


namespace distance_travel_l209_209979

-- Definition of the parameters and the proof problem
variable (W_t : ℕ)
variable (R_c : ℕ)
variable (remaining_coal : ℕ)

-- Conditions
def rate_of_coal_consumption : Prop := R_c = 4 * W_t / 1000
def remaining_coal_amount : Prop := remaining_coal = 160

-- Theorem statement
theorem distance_travel (W_t : ℕ) (R_c : ℕ) (remaining_coal : ℕ) 
  (h1 : rate_of_coal_consumption W_t R_c) 
  (h2 : remaining_coal_amount remaining_coal) : 
  (remaining_coal * 1000 / 4 / W_t) = 40000 / W_t := 
by
  sorry

end distance_travel_l209_209979


namespace fill_blank_1_fill_blank_2_l209_209466

theorem fill_blank_1 (x : ℤ) (h : 1 + x = -10) : x = -11 := sorry

theorem fill_blank_2 (y : ℝ) (h : y - 4.5 = -4.5) : y = 0 := sorry

end fill_blank_1_fill_blank_2_l209_209466


namespace value_of_a4_l209_209871

variables {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers.

-- Conditions: The sequence is geometric, positive and satisfies the given product condition.
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = (a n) * (a k)

-- Condition: All terms are positive.
def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

-- Given product condition:
axiom a1_a7_product : a 1 * a 7 = 36

-- The theorem to prove:
theorem value_of_a4 (h_geo : is_geometric_sequence a) (h_pos : all_terms_positive a) : 
  a 4 = 6 :=
sorry

end value_of_a4_l209_209871


namespace find_n_l209_209830

noncomputable def f (n : ℝ) : ℝ :=
  n ^ (n / 2)

example : f 2 = 2 := sorry

theorem find_n : ∃ n : ℝ, f n = 12 ∧ abs (n - 3.4641) < 0.0001 := sorry

end find_n_l209_209830


namespace find_y_l209_209464

theorem find_y (y : ℚ) (h : ⌊y⌋ + y = 5) : y = 7 / 3 :=
sorry

end find_y_l209_209464


namespace amount_y_gets_each_rupee_x_gets_l209_209253

-- Given conditions
variables (x y z a : ℝ)
variables (h_y_share : y = 36) (h_total : x + y + z = 156) (h_z : z = 0.50 * x)

-- Proof problem
theorem amount_y_gets_each_rupee_x_gets (h : 36 / x = a) : a = 9 / 20 :=
by {
  -- The proof is omitted and replaced with 'sorry'.
  sorry
}

end amount_y_gets_each_rupee_x_gets_l209_209253


namespace number_of_TVs_in_shop_c_l209_209122

theorem number_of_TVs_in_shop_c 
  (a b d e : ℕ) 
  (avg : ℕ) 
  (num_shops : ℕ) 
  (total_TVs_in_other_shops : ℕ) 
  (total_TVs : ℕ) 
  (sum_shops : a + b + d + e = total_TVs_in_other_shops) 
  (avg_sets : avg = total_TVs / num_shops) 
  (number_shops : num_shops = 5)
  (avg_value : avg = 48)
  (T_a : a = 20) 
  (T_b : b = 30) 
  (T_d : d = 80) 
  (T_e : e = 50) 
  : (total_TVs - total_TVs_in_other_shops = 60) := 
by 
  sorry

end number_of_TVs_in_shop_c_l209_209122


namespace barbed_wire_cost_l209_209273

noncomputable def total_cost_barbed_wire (area : ℕ) (cost_per_meter : ℝ) (gate_width : ℕ) : ℝ :=
  let s := Real.sqrt area
  let perimeter := 4 * s - 2 * gate_width
  perimeter * cost_per_meter

theorem barbed_wire_cost :
  total_cost_barbed_wire 3136 3.5 1 = 777 := by
  sorry

end barbed_wire_cost_l209_209273


namespace total_pictures_l209_209727

noncomputable def RandyPics : ℕ := 5
noncomputable def PeterPics : ℕ := RandyPics + 3
noncomputable def QuincyPics : ℕ := PeterPics + 20

theorem total_pictures :
  RandyPics + PeterPics + QuincyPics = 41 :=
by
  sorry

end total_pictures_l209_209727


namespace largest_angle_in_pentagon_l209_209962

theorem largest_angle_in_pentagon {R S : ℝ} (h₁: R = S) 
  (h₂: (75 : ℝ) + 110 + R + S + (3 * R - 20) = 540) : 
  (3 * R - 20) = 217 :=
by {
  -- Given conditions are assigned and now we need to prove the theorem, the proof is omitted
  sorry
}

end largest_angle_in_pentagon_l209_209962


namespace greatest_integer_l209_209862

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℤ, n = 9 * k - 2) (h3 : ∃ l : ℤ, n = 8 * l - 4) : n = 124 := 
sorry

end greatest_integer_l209_209862


namespace proof_a_eq_x_and_b_eq_x_pow_x_l209_209807

theorem proof_a_eq_x_and_b_eq_x_pow_x
  {a b x : ℕ}
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_x : 0 < x)
  (h : x^(a + b) = a^b * b) :
  a = x ∧ b = x^x := 
by
  sorry

end proof_a_eq_x_and_b_eq_x_pow_x_l209_209807


namespace calc_op_l209_209174

def op (a b : ℕ) := (a + b) * (a - b)

theorem calc_op : (op 5 2)^2 = 441 := 
by 
  sorry

end calc_op_l209_209174


namespace marbles_given_by_Joan_l209_209383

def initial_yellow_marbles : ℝ := 86.0
def final_yellow_marbles : ℝ := 111.0

theorem marbles_given_by_Joan :
  final_yellow_marbles - initial_yellow_marbles = 25 := by
  sorry

end marbles_given_by_Joan_l209_209383


namespace cube_root_expression_l209_209720

theorem cube_root_expression (x : ℝ) (hx : x ≥ 0) : (x * Real.sqrt (x * x^(1/3)))^(1/3) = x^(5/9) :=
by
  sorry

end cube_root_expression_l209_209720


namespace total_savings_correct_l209_209842

-- Define the savings of Sam, Victory and Alex according to the given conditions
def sam_savings : ℕ := 1200
def victory_savings : ℕ := sam_savings - 200
def alex_savings : ℕ := 2 * victory_savings

-- Define the total savings
def total_savings : ℕ := sam_savings + victory_savings + alex_savings

-- The theorem to prove the total savings
theorem total_savings_correct : total_savings = 4200 :=
by
  sorry

end total_savings_correct_l209_209842


namespace horse_revolutions_l209_209936

-- Defining the problem conditions
def radius_outer : ℝ := 30
def radius_inner : ℝ := 10
def revolutions_outer : ℕ := 25

-- The question we need to prove
theorem horse_revolutions :
  (revolutions_outer : ℝ) * (radius_outer / radius_inner) = 75 := 
by
  sorry

end horse_revolutions_l209_209936


namespace notebooks_to_sell_to_earn_profit_l209_209093

-- Define the given conditions
def notebooks_purchased : ℕ := 2000
def cost_per_notebook : ℚ := 0.15
def selling_price_per_notebook : ℚ := 0.30
def desired_profit : ℚ := 120

-- Define the total cost
def total_cost := notebooks_purchased * cost_per_notebook

-- Define the total revenue needed
def total_revenue_needed := total_cost + desired_profit

-- Define the number of notebooks to be sold to achieve the total revenue
def notebooks_to_sell := total_revenue_needed / selling_price_per_notebook

-- Prove that the number of notebooks to be sold is 1400 to make a profit of $120
theorem notebooks_to_sell_to_earn_profit : notebooks_to_sell = 1400 := 
by {
  sorry
}

end notebooks_to_sell_to_earn_profit_l209_209093


namespace find_acute_angle_of_parallel_vectors_l209_209080

open Real

theorem find_acute_angle_of_parallel_vectors (x : ℝ) (hx1 : (sin x) * (1 / 2 * cos x) = 1 / 4) (hx2 : 0 < x ∧ x < π / 2) : x = π / 4 :=
by
  sorry

end find_acute_angle_of_parallel_vectors_l209_209080


namespace floor_sum_min_value_l209_209671

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l209_209671


namespace compacted_space_of_all_cans_l209_209479

def compacted_space_per_can (original_space: ℕ) (compaction_rate: ℕ) : ℕ :=
  original_space * compaction_rate / 100

def total_compacted_space (num_cans: ℕ) (compacted_space: ℕ) : ℕ :=
  num_cans * compacted_space

theorem compacted_space_of_all_cans :
  ∀ (num_cans original_space compaction_rate : ℕ),
  num_cans = 100 →
  original_space = 30 →
  compaction_rate = 35 →
  total_compacted_space num_cans (compacted_space_per_can original_space compaction_rate) = 1050 :=
by
  intros num_cans original_space compaction_rate h1 h2 h3
  rw [h1, h2, h3]
  dsimp [compacted_space_per_can, total_compacted_space]
  norm_num
  sorry

end compacted_space_of_all_cans_l209_209479


namespace triangle_angle_A_l209_209433

variable {a b c : ℝ} {A : ℝ}

theorem triangle_angle_A (h : a^2 = b^2 + c^2 - b * c) : A = 2 * Real.pi / 3 :=
by
  sorry

end triangle_angle_A_l209_209433


namespace intersection_A_B_l209_209119

open Set

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def B : Set ℝ := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x} :=
by
  sorry

end intersection_A_B_l209_209119


namespace altitudes_sum_eq_l209_209577

variables {α : Type*} [LinearOrderedField α]

structure Triangle (α) :=
(A B C : α)
(R : α)   -- circumradius
(r : α)   -- inradius

variables (T : Triangle α)
(A B C : α)
(m n p : α)  -- points on respective arcs
(h1 h2 h3 : α)  -- altitudes of the segments

theorem altitudes_sum_eq (T : Triangle α) (A B C m n p h1 h2 h3 : α) :
  h1 + h2 + h3 = 2 * T.R - T.r :=
sorry

end altitudes_sum_eq_l209_209577


namespace trajectory_of_P_is_line_l209_209894

noncomputable def P_trajectory_is_line (a m : ℝ) (P : ℝ × ℝ) : Prop :=
  let A := (-a, 0)
  let B := (a, 0)
  let PA := (P.1 + a) ^ 2 + P.2 ^ 2
  let PB := (P.1 - a) ^ 2 + P.2 ^ 2
  PA - PB = m → P.1 = m / (4 * a)

theorem trajectory_of_P_is_line (a m : ℝ) (h : a ≠ 0) :
  ∀ (P : ℝ × ℝ), (P_trajectory_is_line a m P) := sorry

end trajectory_of_P_is_line_l209_209894


namespace minimum_toothpicks_to_remove_l209_209101

-- Conditions
def number_of_toothpicks : ℕ := 60
def largest_triangle_side : ℕ := 3
def smallest_triangle_side : ℕ := 1

-- Problem Statement
theorem minimum_toothpicks_to_remove (toothpicks_total : ℕ) (largest_side : ℕ) (smallest_side : ℕ) 
  (h1 : toothpicks_total = 60) 
  (h2 : largest_side = 3) 
  (h3 : smallest_side = 1) : 
  ∃ n : ℕ, n = 20 := by
  sorry

end minimum_toothpicks_to_remove_l209_209101


namespace women_in_third_group_l209_209103

variables (m w : ℝ)

theorem women_in_third_group (h1 : 3 * m + 8 * w = 6 * m + 2 * w) (x : ℝ) (h2 : 2 * m + x * w = 0.5 * (3 * m + 8 * w)) :
  x = 4 :=
sorry

end women_in_third_group_l209_209103


namespace tickets_per_candy_l209_209669

theorem tickets_per_candy (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) (candies_bought : ℕ)
    (h1 : tickets_whack_a_mole = 26) (h2 : tickets_skee_ball = 19) (h3 : candies_bought = 5) :
    (tickets_whack_a_mole + tickets_skee_ball) / candies_bought = 9 := by
  sorry

end tickets_per_candy_l209_209669


namespace chord_bisection_l209_209305

theorem chord_bisection {r : ℝ} (PQ RS : Set (ℝ × ℝ)) (O T P Q R S M : ℝ × ℝ)
  (radius_OP : dist O P = 6) (radius_OQ : dist O Q = 6)
  (radius_OR : dist O R = 6) (radius_OS : dist O S = 6) (radius_OT : dist O T = 6)
  (radius_OM : dist O M = 2 * Real.sqrt 13) 
  (PT_eq_8 : dist P T = 8) (TQ_eq_8 : dist T Q = 8)
  (sin_theta_eq_4_5 : Real.sin (Real.arcsin (8 / 10)) = 4 / 5) :
  4 * 5 = 20 :=
by
  sorry

end chord_bisection_l209_209305


namespace romeo_total_profit_is_55_l209_209356

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end romeo_total_profit_is_55_l209_209356


namespace min_time_to_cover_distance_l209_209890

variable (distance : ℝ := 3)
variable (vasya_speed_run : ℝ := 4)
variable (vasya_speed_skate : ℝ := 8)
variable (petya_speed_run : ℝ := 5)
variable (petya_speed_skate : ℝ := 10)

theorem min_time_to_cover_distance :
  ∃ (t : ℝ), t = 0.5 ∧
    ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ distance ∧ 
    (distance - x) / vasya_speed_run + x / vasya_speed_skate = t ∧
    x / petya_speed_run + (distance - x) / petya_speed_skate = t :=
by
  sorry

end min_time_to_cover_distance_l209_209890


namespace sum_lent_borrowed_l209_209655

-- Define the given conditions and the sum lent
def sum_lent (P r t : ℝ) (I : ℝ) : Prop :=
  I = P * r * t / 100 ∧ I = P - 1540

-- Define the main theorem to be proven
theorem sum_lent_borrowed : 
  ∃ P : ℝ, sum_lent P 8 10 ((4 * P) / 5) ∧ P = 7700 :=
by
  sorry

end sum_lent_borrowed_l209_209655


namespace find_y_of_pentagon_l209_209938

def y_coordinate (y : ℝ) : Prop :=
  let area_ABDE := 12
  let area_BCD := 2 * (y - 3)
  let total_area := area_ABDE + area_BCD
  total_area = 35

theorem find_y_of_pentagon :
  ∃ y : ℝ, y_coordinate y ∧ y = 14.5 :=
by
  sorry

end find_y_of_pentagon_l209_209938


namespace denote_below_warning_level_l209_209233

-- Conditions
def warning_water_level : ℝ := 905.7
def exceed_by_10 : ℝ := 10
def below_by_5 : ℝ := -5

-- Problem statement
theorem denote_below_warning_level : below_by_5 = -5 := 
by
  sorry

end denote_below_warning_level_l209_209233


namespace combined_work_rate_l209_209571

theorem combined_work_rate (x_rate y_rate : ℚ) (h1 : x_rate = 1 / 15) (h2 : y_rate = 1 / 45) :
    1 / (x_rate + y_rate) = 11.25 :=
by
  -- Proof goes here
  sorry

end combined_work_rate_l209_209571


namespace symmetric_points_x_axis_l209_209186

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ := (a, 1)) (Q : ℝ × ℝ := (-4, b)) :
  (Q.1 = -P.1 ∧ Q.2 = -P.2) → (a = -4 ∧ b = -1) :=
by {
  sorry
}

end symmetric_points_x_axis_l209_209186


namespace money_problem_l209_209255

-- Define the conditions and the required proof
theorem money_problem (B S : ℕ) 
  (h1 : B = 2 * S) -- Condition 1: Brother brought twice as much money as the sister
  (h2 : B - 180 = S - 30) -- Condition 3: Remaining money of brother and sister are equal
  : B = 300 ∧ S = 150 := -- Correct answer to prove
  
sorry -- Placeholder for proof

end money_problem_l209_209255


namespace olivia_wallet_l209_209834

theorem olivia_wallet (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 78)
  (h2 : spent_amount = 15):
  remaining_amount = initial_amount - spent_amount →
  remaining_amount = 63 :=
sorry

end olivia_wallet_l209_209834


namespace problem_f_sum_zero_l209_209744

variable (f : ℝ → ℝ)

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def symmetrical (f : ℝ → ℝ) : Prop := ∀ x, f (1 - x) = f x

-- Prove the required sum is zero given the conditions.
theorem problem_f_sum_zero (hf_odd : odd f) (hf_symm : symmetrical f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end problem_f_sum_zero_l209_209744


namespace solve_for_M_l209_209147

theorem solve_for_M (a b M : ℝ) (h : (a + 2 * b) ^ 2 = (a - 2 * b) ^ 2 + M) : M = 8 * a * b :=
by sorry

end solve_for_M_l209_209147


namespace num_geography_books_l209_209277

theorem num_geography_books
  (total_books : ℕ)
  (history_books : ℕ)
  (math_books : ℕ)
  (h1 : total_books = 100)
  (h2 : history_books = 32)
  (h3 : math_books = 43) :
  total_books - history_books - math_books = 25 :=
by
  sorry

end num_geography_books_l209_209277


namespace find_f_of_2_l209_209165

-- Definitions based on problem conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x) + 9

-- The main statement to proof that f(2) = 6 under the given conditions
theorem find_f_of_2 (f : ℝ → ℝ)
  (hf : is_odd_function f)
  (hg : ∀ x, g f x = f x + 9)
  (h : g f (-2) = 3) :
  f 2 = 6 := 
sorry

end find_f_of_2_l209_209165


namespace units_digit_N_l209_209615

def P (n : ℕ) : ℕ := (n / 10) * (n % 10)
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem units_digit_N (N : ℕ) (h1 : 10 ≤ N ∧ N ≤ 99) (h2 : N = P N + S N) : N % 10 = 9 :=
by
  sorry

end units_digit_N_l209_209615


namespace ratio_doctors_lawyers_l209_209534

theorem ratio_doctors_lawyers (d l : ℕ) (h1 : (45 * d + 60 * l) / (d + l) = 50) (h2 : d + l = 50) : d = 2 * l :=
by
  sorry

end ratio_doctors_lawyers_l209_209534


namespace find_first_factor_of_LCM_l209_209766

-- Conditions
def HCF : ℕ := 23
def Y : ℕ := 14
def largest_number : ℕ := 322

-- Statement
theorem find_first_factor_of_LCM
  (A B : ℕ)
  (H : Nat.gcd A B = HCF)
  (max_num : max A B = largest_number)
  (lcm_eq : Nat.lcm A B = HCF * X * Y) :
  X = 23 :=
sorry

end find_first_factor_of_LCM_l209_209766


namespace initial_people_lifting_weights_l209_209051

theorem initial_people_lifting_weights (x : ℕ) (h : x + 3 = 19) : x = 16 :=
by
  sorry

end initial_people_lifting_weights_l209_209051


namespace min_words_to_learn_l209_209575

theorem min_words_to_learn (n : ℕ) (p_guess : ℝ) (required_score : ℝ)
  (h_n : n = 600) (h_p : p_guess = 0.1) (h_score : required_score = 0.9) :
  ∃ x : ℕ, (x + p_guess * (n - x)) / n ≥ required_score ∧ x = 534 :=
by
  sorry

end min_words_to_learn_l209_209575


namespace card_S_l209_209243

def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 5 * n - 1

def S : Finset ℕ := 
  (Finset.range 2016).image a ∩ (Finset.range (a 2015 + 1)).image b

theorem card_S : S.card = 504 := 
  sorry

end card_S_l209_209243


namespace barbara_removed_total_sheets_l209_209474

theorem barbara_removed_total_sheets :
  let bundles_colored := 3
  let bunches_white := 2
  let heaps_scrap := 5
  let sheets_per_bunch := 4
  let sheets_per_bundle := 2
  let sheets_per_heap := 20
  bundles_colored * sheets_per_bundle + bunches_white * sheets_per_bunch + heaps_scrap * sheets_per_heap = 114 :=
by
  sorry

end barbara_removed_total_sheets_l209_209474


namespace jane_waiting_time_l209_209676

-- Given conditions as constants for readability
def base_coat_drying_time := 2
def first_color_coat_drying_time := 3
def second_color_coat_drying_time := 3
def top_coat_drying_time := 5

-- Total drying time calculation
def total_drying_time := base_coat_drying_time 
                       + first_color_coat_drying_time 
                       + second_color_coat_drying_time 
                       + top_coat_drying_time

-- The theorem to prove
theorem jane_waiting_time : total_drying_time = 13 := 
by
  sorry

end jane_waiting_time_l209_209676


namespace sum_exterior_angles_triangle_and_dodecagon_l209_209899

-- Definitions derived from conditions
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3

-- Proof problem statement
theorem sum_exterior_angles_triangle_and_dodecagon :
  is_polygon 3 ∧ is_polygon 12 → sum_exterior_angles 3 + sum_exterior_angles 12 = 720 :=
by
  sorry

end sum_exterior_angles_triangle_and_dodecagon_l209_209899


namespace unique_chair_arrangement_l209_209407

theorem unique_chair_arrangement (n : ℕ) (h : n = 49)
  (h1 : ∀ i j : ℕ, (n = i * j) → (i ≥ 2) ∧ (j ≥ 2)) :
  ∃! i j : ℕ, (n = i * j) ∧ (i ≥ 2) ∧ (j ≥ 2) :=
by
  sorry

end unique_chair_arrangement_l209_209407


namespace number_of_sequences_l209_209724

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end number_of_sequences_l209_209724


namespace probability_of_staying_in_dark_l209_209178

theorem probability_of_staying_in_dark (revolutions_per_minute : ℕ) (time_in_seconds : ℕ) (dark_time : ℕ) :
  revolutions_per_minute = 2 →
  time_in_seconds = 60 →
  dark_time = 5 →
  (5 / 6 : ℝ) = 5 / 6 :=
by
  intros
  sorry

end probability_of_staying_in_dark_l209_209178


namespace bakery_ratio_l209_209064

theorem bakery_ratio (F B : ℕ) 
    (h1 : F = 10 * B)
    (h2 : F = 8 * (B + 60))
    (sugar : ℕ)
    (h3 : sugar = 3000) :
    sugar / F = 5 / 4 :=
by sorry

end bakery_ratio_l209_209064


namespace number_of_ordered_triples_l209_209223

theorem number_of_ordered_triples (a b c : ℤ) : 
  ∃ (n : ℕ), -31 <= a ∧ a <= 31 ∧ -31 <= b ∧ b <= 31 ∧ -31 <= c ∧ c <= 31 ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a + b + c > 0) ∧ n = 117690 :=
by sorry

end number_of_ordered_triples_l209_209223


namespace inequality_D_holds_l209_209272

theorem inequality_D_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
sorry

end inequality_D_holds_l209_209272


namespace custom_op_identity_l209_209882

def custom_op (x y : ℕ) : ℕ := x * y + 3 * x - 4 * y

theorem custom_op_identity : custom_op 7 5 - custom_op 5 7 = 14 :=
by
  sorry

end custom_op_identity_l209_209882


namespace cuboid_edge_length_l209_209131

theorem cuboid_edge_length (x : ℝ) (h1 : (2 * (x * 5 + x * 6 + 5 * 6)) = 148) : x = 4 :=
by 
  sorry

end cuboid_edge_length_l209_209131


namespace students_only_one_activity_l209_209377

theorem students_only_one_activity 
  (total : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 317) 
  (h_both : both = 30) 
  (h_neither : neither = 20) : 
  (total - both - neither) = 267 :=
by 
  sorry

end students_only_one_activity_l209_209377


namespace ellipse_focal_distance_m_value_l209_209040

-- Define the given conditions 
def focal_distance := 2
def ellipse_equation (x y : ℝ) (m : ℝ) := (x^2 / m) + (y^2 / 4) = 1

-- The proof statement
theorem ellipse_focal_distance_m_value :
  ∀ (m : ℝ), 
    (∃ c : ℝ, (2 * c = focal_distance) ∧ (m = 4 + c^2)) →
      m = 5 := by
  sorry

end ellipse_focal_distance_m_value_l209_209040


namespace police_arrangements_l209_209289

theorem police_arrangements (officers : Fin 5) (A B : Fin 5) (intersections : Fin 3) :
  A ≠ B →
  (∃ arrangement : Fin 5 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ off : Fin 5, arrangement off = i ∧ arrangement off = j) ∧
    arrangement A = arrangement B) →
  ∃ arrangements_count : Nat, arrangements_count = 36 :=
by
  sorry

end police_arrangements_l209_209289


namespace arithmetic_sequence_value_of_n_l209_209314

theorem arithmetic_sequence_value_of_n :
  ∀ (a n d : ℕ), a = 1 → d = 3 → (a + (n - 1) * d = 2005) → n = 669 :=
by
  intros a n d h_a1 h_d ha_n
  sorry

end arithmetic_sequence_value_of_n_l209_209314


namespace weight_of_each_soda_crate_l209_209660

-- Definitions based on conditions
def bridge_weight_limit := 20000
def empty_truck_weight := 12000
def number_of_soda_crates := 20
def dryer_weight := 3000
def number_of_dryers := 3
def fully_loaded_truck_weight := 24000
def soda_weight := 1000
def produce_weight := 2 * soda_weight
def total_cargo_weight := fully_loaded_truck_weight - empty_truck_weight

-- Lean statement to prove the weight of each soda crate
theorem weight_of_each_soda_crate :
  number_of_soda_crates * ((total_cargo_weight - (number_of_dryers * dryer_weight)) / 3) / number_of_soda_crates = 50 :=
by
  sorry

end weight_of_each_soda_crate_l209_209660


namespace max_value_x2_plus_2xy_l209_209326

open Real

theorem max_value_x2_plus_2xy (x y : ℝ) (h : x + y = 5) : 
  ∃ (M : ℝ), (M = x^2 + 2 * x * y) ∧ (∀ z w : ℝ, z + w = 5 → z^2 + 2 * z * w ≤ M) :=
by
  sorry

end max_value_x2_plus_2xy_l209_209326


namespace find_k_value_l209_209404

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + 3 * x + 7
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 3 * x^3 - k * x^2 + 4

theorem find_k_value : (f 5 - g 5 k = 45) → k = 27 / 25 :=
by
  intro h
  sorry

end find_k_value_l209_209404


namespace arithmetic_sequence_a3_l209_209121

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : a 2 + a 4 = 8) (h_seq : a 2 + a 4 = 2 * a 3) :
  a 3 = 4 :=
by
  sorry

end arithmetic_sequence_a3_l209_209121


namespace nonstudent_ticket_cost_l209_209819

theorem nonstudent_ticket_cost :
  ∃ x : ℝ, (530 * 2 + (821 - 530) * x = 1933) ∧ x = 3 :=
by 
  sorry

end nonstudent_ticket_cost_l209_209819


namespace part1_part2_l209_209680

variable (x y : ℤ) (A B : ℤ)

def A_def : ℤ := 3 * x^2 - 5 * x * y - 2 * y^2
def B_def : ℤ := x^2 - 3 * y

theorem part1 : A_def x y - 2 * B_def x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by
  sorry

theorem part2 : A_def 2 (-1) - 2 * B_def 2 (-1) = 6 := by
  sorry

end part1_part2_l209_209680


namespace total_students_in_college_l209_209713

theorem total_students_in_college 
  (girls : ℕ) 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (h_ratio : ratio_boys = 8) 
  (h_ratio_girls : ratio_girls = 5) 
  (h_girls : girls = 400) 
  : (ratio_boys * (girls / ratio_girls) + girls = 1040) := 
by 
  sorry

end total_students_in_college_l209_209713


namespace sheets_of_paper_l209_209909

theorem sheets_of_paper (x : ℕ) (sheets : ℕ) 
  (h1 : sheets = 3 * x + 31)
  (h2 : sheets = 4 * x + 8) : 
  sheets = 100 := by
  sorry

end sheets_of_paper_l209_209909


namespace value_of_a_in_terms_of_b_l209_209046

noncomputable def value_of_a (b : ℝ) : ℝ :=
  b * (38.1966 / 61.8034)

theorem value_of_a_in_terms_of_b (b a : ℝ) :
  (∀ x : ℝ, (b / x = 61.80339887498949 / 100) ∧ (x = (a + b) * (61.80339887498949 / 100)))
  → a = value_of_a b :=
by
  sorry

end value_of_a_in_terms_of_b_l209_209046


namespace roll_four_fair_dice_l209_209275
noncomputable def roll_four_fair_dice_prob : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 6
  let prob_all_same : ℚ := favorable_outcomes / total_outcomes
  let prob_not_all_same : ℚ := 1 - prob_all_same
  prob_not_all_same

theorem roll_four_fair_dice :
  roll_four_fair_dice_prob = 215 / 216 :=
by
  sorry

end roll_four_fair_dice_l209_209275


namespace paving_cost_l209_209392

theorem paving_cost (length width rate : ℝ) (h_length : length = 8) (h_width : width = 4.75) (h_rate : rate = 900) :
  length * width * rate = 34200 :=
by
  rw [h_length, h_width, h_rate]
  norm_num

end paving_cost_l209_209392


namespace consecutive_integer_sets_l209_209499

theorem consecutive_integer_sets (S : ℕ) (hS : S = 180) : 
  ∃ n_values : Finset ℕ, 
  (∀ n ∈ n_values, (∃ a : ℕ, n * (2 * a + n - 1) = 2 * S) ∧ n >= 2) ∧ 
  n_values.card = 4 :=
by
  sorry

end consecutive_integer_sets_l209_209499


namespace max_S_R_squared_l209_209257

theorem max_S_R_squared (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) :
  (∃ a b c, DA = a ∧ DB = b ∧ DC = c ∧ S = 2 * (a * b + b * c + c * a) ∧
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 ∧ (∃ max_val, max_val = (2 / 3) * (3 + Real.sqrt 3))) :=
sorry

end max_S_R_squared_l209_209257


namespace students_not_enrolled_in_any_classes_l209_209753

/--
  At a particular college, 27.5% of the 1050 students are enrolled in biology,
  32.9% of the students are enrolled in mathematics, and 15% of the students are enrolled in literature classes.
  Assuming that no student is taking more than one of these specific subjects,
  the number of students at the college who are not enrolled in biology, mathematics, or literature classes is 260.

  We want to prove the statement:
    number_students_not_enrolled_in_any_classes = 260
-/
theorem students_not_enrolled_in_any_classes 
  (total_students : ℕ) 
  (biology_percent : ℝ) 
  (mathematics_percent : ℝ) 
  (literature_percent : ℝ) 
  (no_student_in_multiple : Prop) : 
  total_students = 1050 →
  biology_percent = 27.5 →
  mathematics_percent = 32.9 →
  literature_percent = 15 →
  (total_students - (⌊biology_percent / 100 * total_students⌋ + ⌊mathematics_percent / 100 * total_students⌋ + ⌊literature_percent / 100 * total_students⌋)) = 260 :=
by {
  sorry
}

end students_not_enrolled_in_any_classes_l209_209753


namespace proof_standard_deviation_l209_209160

noncomputable def standard_deviation (average_age : ℝ) (max_diff_ages : ℕ) : ℝ := sorry

theorem proof_standard_deviation :
  let average_age := 31
  let max_diff_ages := 19
  standard_deviation average_age max_diff_ages = 9 := 
by
  sorry

end proof_standard_deviation_l209_209160


namespace peter_son_is_nikolay_l209_209428

variable (x y : ℕ)

/-- Within the stated scenarios of Nikolai/Peter paired fishes caught -/
theorem peter_son_is_nikolay :
  (∀ n p ns ps : ℕ, (
    n = ns ∧              -- Nikolai caught as many fish as his son
    p = 3 * ps ∧          -- Peter caught three times more fish than his son
    n + ns + p + ps = 25  -- A total of 25 fish were caught
  ) → ("Nikolay" = "Peter's son")) := 
sorry

end peter_son_is_nikolay_l209_209428


namespace nina_widgets_after_reduction_is_approx_8_l209_209249

noncomputable def nina_total_money : ℝ := 16.67
noncomputable def widgets_before_reduction : ℝ := 5
noncomputable def cost_reduction_per_widget : ℝ := 1.25

noncomputable def cost_per_widget_before_reduction : ℝ := nina_total_money / widgets_before_reduction
noncomputable def cost_per_widget_after_reduction : ℝ := cost_per_widget_before_reduction - cost_reduction_per_widget
noncomputable def widgets_after_reduction : ℝ := nina_total_money / cost_per_widget_after_reduction

-- Prove that Nina can purchase approximately 8 widgets after the cost reduction
theorem nina_widgets_after_reduction_is_approx_8 : abs (widgets_after_reduction - 8) < 1 :=
by
  sorry

end nina_widgets_after_reduction_is_approx_8_l209_209249


namespace max_value_sqrt43_l209_209292

noncomputable def max_value_expr (x y z : ℝ) : ℝ :=
  3 * x * z * Real.sqrt 2 + 5 * x * y

theorem max_value_sqrt43 (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  max_value_expr x y z ≤ Real.sqrt 43 :=
sorry

end max_value_sqrt43_l209_209292
