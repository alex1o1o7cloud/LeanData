import Mathlib

namespace second_grade_students_sampled_l1730_173023

-- Definitions corresponding to conditions in a)
def total_students := 2000
def mountain_climbing_fraction := 2 / 5
def running_ratios := (2, 3, 5)
def sample_size := 200

-- Calculation of total running participants based on ratio
def total_running_students :=
  total_students * (1 - mountain_climbing_fraction)

def a := 2 * (total_running_students / (2 + 3 + 5))
def b := 3 * (total_running_students / (2 + 3 + 5))
def c := 5 * (total_running_students / (2 + 3 + 5))

def running_sample_size := sample_size * (3 / 5) --since the ratio is 3:5

-- The statement to prove
theorem second_grade_students_sampled : running_sample_size * (3 / (2+3+5)) = 36 :=
by
  sorry

end second_grade_students_sampled_l1730_173023


namespace aqua_park_earnings_l1730_173017

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end aqua_park_earnings_l1730_173017


namespace min_students_l1730_173085

theorem min_students (b g : ℕ) (hb : 1 ≤ b) (hg : 1 ≤ g)
    (h1 : b = (4/3) * g) 
    (h2 : (1/2) * b = 2 * ((1/3) * g)) 
    : b + g = 7 :=
by sorry

end min_students_l1730_173085


namespace markers_per_box_l1730_173099

theorem markers_per_box
  (students : ℕ) (boxes : ℕ) (group1_students : ℕ) (group1_markers : ℕ)
  (group2_students : ℕ) (group2_markers : ℕ) (last_group_markers : ℕ)
  (h_students : students = 30)
  (h_boxes : boxes = 22)
  (h_group1_students : group1_students = 10)
  (h_group1_markers : group1_markers = 2)
  (h_group2_students : group2_students = 15)
  (h_group2_markers : group2_markers = 4)
  (h_last_group_markers : last_group_markers = 6) :
  (110 = students * ((group1_students * group1_markers + group2_students * group2_markers + (students - group1_students - group2_students) * last_group_markers)) / boxes) :=
by
  sorry

end markers_per_box_l1730_173099


namespace example_calculation_l1730_173036

theorem example_calculation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end example_calculation_l1730_173036


namespace commute_times_absolute_difference_l1730_173083

theorem commute_times_absolute_difference
  (x y : ℝ)
  (H_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (H_var : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  abs (x - y) = 4 :=
by
  -- proof steps are omitted
  sorry

end commute_times_absolute_difference_l1730_173083


namespace hyperbola_eccentricity_l1730_173089

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes_intersect : ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 = -1 ∧ B.1 = -1 ∧
    ∀ (A B : ℝ × ℝ), ∃ x y : ℝ, (A.2 = y ∧ B.2 = y ∧ x^2 / a^2 - y^2 / b^2 = 1))
  (triangle_area : ∃ A B : ℝ × ℝ, 1 / 2 * abs (A.1 * B.2 - A.2 * B.1) = 2 * Real.sqrt 3) :
  ∃ e : ℝ, e = Real.sqrt 13 :=
by {
  sorry
}

end hyperbola_eccentricity_l1730_173089


namespace iron_needed_for_hydrogen_l1730_173010

-- Conditions of the problem
def reaction (Fe H₂SO₄ FeSO₄ H₂ : ℕ) : Prop :=
  Fe + H₂SO₄ = FeSO₄ + H₂

-- Given data
def balanced_equation : Prop :=
  reaction 1 1 1 1
 
def produced_hydrogen : ℕ := 2
def produced_from_sulfuric_acid : ℕ := 2
def needed_iron : ℕ := 2

-- Problem statement to be proved
theorem iron_needed_for_hydrogen (H₂SO₄ H₂ : ℕ) (h1 : produced_hydrogen = H₂) (h2 : produced_from_sulfuric_acid = H₂SO₄) (balanced_eq : balanced_equation) :
  needed_iron = 2 := by
sorry

end iron_needed_for_hydrogen_l1730_173010


namespace floor_sqrt_72_l1730_173090

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 :=
by
  -- Proof required here
  sorry

end floor_sqrt_72_l1730_173090


namespace find_b_if_lines_parallel_l1730_173025

-- Definitions of the line equations and parallel condition
def first_line (x y : ℝ) (b : ℝ) : Prop := 3 * y - b = -9 * x + 1
def second_line (x y : ℝ) (b : ℝ) : Prop := 2 * y + 8 = (b - 3) * x - 2

-- Definition of parallel lines (their slopes are equal)
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

-- Given conditions and the conclusion to prove
theorem find_b_if_lines_parallel :
  ∃ b : ℝ, (∀ x y : ℝ, first_line x y b → ∃ m1 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m1 * x + c) ∧ 
           (∀ x y : ℝ, second_line x y b → ∃ m2 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m2 * x + c) ∧ 
           parallel_lines (-3) ((b - 3) / 2) →
           b = -3 :=
by {
  sorry
}

end find_b_if_lines_parallel_l1730_173025


namespace people_at_first_table_l1730_173043

theorem people_at_first_table (N x : ℕ) 
  (h1 : 20 < N) 
  (h2 : N < 50)
  (h3 : (N - x) % 42 = 0)
  (h4 : N % 8 = 7) : 
  x = 5 :=
sorry

end people_at_first_table_l1730_173043


namespace characterization_of_points_l1730_173088

def satisfies_eq (x : ℝ) (y : ℝ) : Prop :=
  max x (x^2) + min y (y^2) = 1

theorem characterization_of_points :
  ∀ x y : ℝ,
  satisfies_eq x y ↔
  ((x < 0 ∨ x > 1) ∧ (y < 0 ∨ y > 1) ∧ y ≤ 0 ∧ y = 1 - x^2) ∨
  ((x < 0 ∨ x > 1) ∧ (0 < y ∧ y < 1) ∧ x^2 + y^2 = 1 ∧ x ≤ -1 ∧ x > 0) ∨
  ((0 < x ∧ x < 1) ∧ (y < 0 ∨ y > 1) ∧ false) ∨
  ((0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y^2 = 1 - x) :=
sorry

end characterization_of_points_l1730_173088


namespace problem_statement_l1730_173031

variable (a b c d : ℝ)

-- Definitions for the conditions
def condition1 := a + b + c + d = 100
def condition2 := (a / (b + c + d)) + (b / (a + c + d)) + (c / (a + b + d)) + (d / (a + b + c)) = 95

-- The theorem which needs to be proved
theorem problem_statement (h1 : condition1 a b c d) (h2 : condition2 a b c d) :
  (1 / (b + c + d)) + (1 / (a + c + d)) + (1 / (a + b + d)) + (1 / (a + b + c)) = 99 / 100 := by
  sorry

end problem_statement_l1730_173031


namespace soccer_ball_purchase_l1730_173030

theorem soccer_ball_purchase (wholesale_price retail_price profit remaining_balls final_profit : ℕ)
  (h1 : wholesale_price = 30)
  (h2 : retail_price = 45)
  (h3 : profit = retail_price - wholesale_price)
  (h4 : remaining_balls = 30)
  (h5 : final_profit = 1500) :
  ∃ (initial_balls : ℕ), (initial_balls - remaining_balls) * profit = final_profit ∧ initial_balls = 130 :=
by
  sorry

end soccer_ball_purchase_l1730_173030


namespace black_area_remaining_after_changes_l1730_173019

theorem black_area_remaining_after_changes :
  let initial_fraction_black := 1
  let change_factor := 8 / 9
  let num_changes := 4
  let final_fraction_black := (change_factor ^ num_changes)
  final_fraction_black = 4096 / 6561 :=
by
  sorry

end black_area_remaining_after_changes_l1730_173019


namespace equality_of_coefficients_l1730_173058

open Real

theorem equality_of_coefficients (a b c x : ℝ)
  (h1 : a * x^2 - b * x - c = b * x^2 - c * x - a)
  (h2 : b * x^2 - c * x - a = c * x^2 - a * x - b)
  (h3 : c * x^2 - a * x - b = a * x^2 - b * x - c):
  a = b ∧ b = c :=
sorry

end equality_of_coefficients_l1730_173058


namespace sufficient_but_not_necessary_l1730_173018

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > |b|) → (a^3 > b^3) ∧ ¬((a^3 > b^3) → (a > |b|)) :=
by
  sorry

end sufficient_but_not_necessary_l1730_173018


namespace product_d_e_l1730_173092

-- Define the problem: roots of the polynomial x^2 + x - 2
def roots_of_quadratic : Prop :=
  ∃ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0)

-- Define the condition that both roots are also roots of another polynomial
def roots_of_higher_poly (α β : ℚ) : Prop :=
  (α^7 - 7 * α^3 - 10 = 0 ) ∧ (β^7 - 7 * β^3 - 10 = 0)

-- The final proposition to prove
theorem product_d_e :
  ∀ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0) → (α^7 - 7 * α^3 - 10 = 0) ∧ (β^7 - 7 * β^3 - 10 = 0) → 7 * 10 = 70 := 
by sorry

end product_d_e_l1730_173092


namespace equation_of_line_l1730_173008

theorem equation_of_line 
  (slope : ℝ)
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h_slope : slope = 2)
  (h_line1 : a1 = 3 ∧ b1 = 4 ∧ c1 = -5)
  (h_line2 : a2 = 3 ∧ b2 = -4 ∧ c2 = -13) 
  : ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = -7) ∧ 
    (∀ x y : ℝ, (a1 * x + b1 * y + c1 = 0) ∧ (a2 * x + b2 * y + c2 = 0) → (a * x + b * y + c = 0)) :=
by
  sorry

end equation_of_line_l1730_173008


namespace solve_inequality_l1730_173056

-- Defining the inequality
def inequality (x : ℝ) : Prop := 1 / (x - 1) ≤ 1

-- Stating the theorem
theorem solve_inequality :
  { x : ℝ | inequality x } = { x : ℝ | x < 1 } ∪ { x : ℝ | 2 ≤ x } :=
by
  sorry

end solve_inequality_l1730_173056


namespace total_boys_in_camp_l1730_173062

theorem total_boys_in_camp (T : ℝ) 
  (h1 : 0.20 * T = number_of_boys_from_school_A)
  (h2 : 0.30 * number_of_boys_from_school_A = number_of_boys_study_science_from_school_A)
  (h3 : number_of_boys_from_school_A - number_of_boys_study_science_from_school_A = 42) :
  T = 300 := 
sorry

end total_boys_in_camp_l1730_173062


namespace rectangle_area_l1730_173015

theorem rectangle_area (area_square : ℝ) 
  (width_rectangle : ℝ) (length_rectangle : ℝ)
  (h1 : area_square = 16)
  (h2 : width_rectangle^2 = area_square)
  (h3 : length_rectangle = 3 * width_rectangle) :
  width_rectangle * length_rectangle = 48 := by sorry

end rectangle_area_l1730_173015


namespace workman_problem_l1730_173055

theorem workman_problem (A B : ℝ) (h1 : A = B / 2) (h2 : (A + B) * 10 = 1) : B = 1 / 15 := by
  sorry

end workman_problem_l1730_173055


namespace remainder_of_fractions_l1730_173087

theorem remainder_of_fractions : 
  ∀ (x y : ℚ), x = 5/7 → y = 3/4 → (x - y * ⌊x / y⌋) = 5/7 :=
by
  intros x y hx hy
  rw [hx, hy]
  -- Additional steps can be filled in here, if continuing with the proof.
  sorry

end remainder_of_fractions_l1730_173087


namespace prob1_part1_prob1_part2_l1730_173038

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2 * a}

theorem prob1_part1 (a : ℝ) (ha : a = 3) :
  A ∪ B a = {x | -2 < x ∧ x < 7} ∧ A ∩ B a = {x | -1 < x ∧ x < 5} :=
by {
  sorry
}

theorem prob1_part2 (h : ∀ x, x ∈ A → x ∈ B a) :
  ∀ a : ℝ, a ≤ 2 :=
by {
  sorry
}

end prob1_part1_prob1_part2_l1730_173038


namespace min_distance_origin_to_line_l1730_173049

theorem min_distance_origin_to_line 
  (x y : ℝ) 
  (h : x + y = 4) : 
  ∃ P : ℝ, P = 2 * Real.sqrt 2 ∧ 
    (∀ Q : ℝ, Q = Real.sqrt (x^2 + y^2) → P ≤ Q) :=
by
  sorry

end min_distance_origin_to_line_l1730_173049


namespace functional_eq_solution_l1730_173093

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (x) ^ 2 + f (y)) = x * f (x) + y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := sorry

end functional_eq_solution_l1730_173093


namespace num_possible_bases_l1730_173051

theorem num_possible_bases (b : ℕ) (h1 : b ≥ 2) (h2 : b^3 ≤ 256) (h3 : 256 < b^4) : ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_bases_l1730_173051


namespace final_toy_count_correct_l1730_173021

def initial_toy_count : ℝ := 5.3
def tuesday_toys_left (initial: ℝ) : ℝ := initial * 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_toys_left (tuesday_total: ℝ) : ℝ := tuesday_total * 0.498
def wednesday_new_toys : ℝ := 2.4
def thursday_toys_left (wednesday_total: ℝ) : ℝ := wednesday_total * 0.692
def thursday_new_toys : ℝ := 4.5

def total_toys (initial: ℝ) : ℝ :=
  let after_tuesday := tuesday_toys_left initial + tuesday_new_toys
  let after_wednesday := wednesday_toys_left after_tuesday + wednesday_new_toys
  let after_thursday := thursday_toys_left after_wednesday + thursday_new_toys
  after_thursday

def toys_lost_tuesday (initial: ℝ) (left: ℝ) : ℝ := initial - left
def toys_lost_wednesday (tuesday_total: ℝ) (left: ℝ) : ℝ := tuesday_total - left
def toys_lost_thursday (wednesday_total: ℝ) (left: ℝ) : ℝ := wednesday_total - left
def total_lost_toys (initial: ℝ) : ℝ :=
  let tuesday_left := tuesday_toys_left initial
  let tuesday_total := tuesday_left + tuesday_new_toys
  let wednesday_left := wednesday_toys_left tuesday_total
  let wednesday_total := wednesday_left + wednesday_new_toys
  let thursday_left := thursday_toys_left wednesday_total
  let lost_tuesday := toys_lost_tuesday initial tuesday_left
  let lost_wednesday := toys_lost_wednesday tuesday_total wednesday_left
  let lost_thursday := toys_lost_thursday wednesday_total thursday_left
  lost_tuesday + lost_wednesday + lost_thursday

def final_toy_count (initial: ℝ) : ℝ :=
  let current_toys := total_toys initial
  let lost_toys := total_lost_toys initial
  current_toys + lost_toys

theorem final_toy_count_correct :
  final_toy_count initial_toy_count = 15.8 := sorry

end final_toy_count_correct_l1730_173021


namespace wheel_sum_even_and_greater_than_10_l1730_173014

-- Definitions based on conditions
def prob_even_A : ℚ := 3 / 8
def prob_odd_A : ℚ := 5 / 8
def prob_even_B : ℚ := 1 / 4
def prob_odd_B : ℚ := 3 / 4

-- Event probabilities from solution steps
def prob_both_even : ℚ := prob_even_A * prob_even_B
def prob_both_odd : ℚ := prob_odd_A * prob_odd_B
def prob_even_sum : ℚ := prob_both_even + prob_both_odd
def prob_even_sum_greater_10 : ℚ := 1 / 3

-- Compute final probability
def final_probability : ℚ := prob_even_sum * prob_even_sum_greater_10

-- The statement that needs proving
theorem wheel_sum_even_and_greater_than_10 : final_probability = 3 / 16 := by
  sorry

end wheel_sum_even_and_greater_than_10_l1730_173014


namespace pizza_slices_l1730_173091

theorem pizza_slices (P T S : ℕ) (h1 : P = 2) (h2 : T = 16) : S = 8 :=
by
  -- to be filled in
  sorry

end pizza_slices_l1730_173091


namespace find_science_books_l1730_173064

theorem find_science_books
  (S : ℕ)
  (h1 : 2 * 3 + 3 * 2 + 3 * S = 30) :
  S = 6 :=
by
  sorry

end find_science_books_l1730_173064


namespace students_first_day_l1730_173013

-- Definitions based on conditions
def total_books : ℕ := 120
def books_per_student : ℕ := 5
def students_second_day : ℕ := 5
def students_third_day : ℕ := 6
def students_fourth_day : ℕ := 9

-- Main goal
theorem students_first_day (total_books_eq : total_books = 120)
                           (books_per_student_eq : books_per_student = 5)
                           (students_second_day_eq : students_second_day = 5)
                           (students_third_day_eq : students_third_day = 6)
                           (students_fourth_day_eq : students_fourth_day = 9) :
  let books_given_second_day := students_second_day * books_per_student
  let books_given_third_day := students_third_day * books_per_student
  let books_given_fourth_day := students_fourth_day * books_per_student
  let total_books_given_after_first_day := books_given_second_day + books_given_third_day + books_given_fourth_day
  let books_first_day := total_books - total_books_given_after_first_day
  let students_first_day := books_first_day / books_per_student
  students_first_day = 4 :=
by sorry

end students_first_day_l1730_173013


namespace triangle_area_l1730_173039

open Real

def line1 (x y : ℝ) : Prop := y = 6
def line2 (x y : ℝ) : Prop := y = 2 + x
def line3 (x y : ℝ) : Prop := y = 2 - x

def is_vertex (x y : ℝ) (l1 l2 : ℝ → ℝ → Prop) : Prop := l1 x y ∧ l2 x y

def vertices (v1 v2 v3 : ℝ × ℝ) : Prop :=
  is_vertex v1.1 v1.2 line1 line2 ∧
  is_vertex v2.1 v2.2 line1 line3 ∧
  is_vertex v3.1 v3.2 line2 line3

def area_triangle (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2) -
             (v2.1 * v1.2 + v3.1 * v2.2 + v1.1 * v3.2))

theorem triangle_area : vertices (4, 6) (-4, 6) (0, 2) → area_triangle (4, 6) (-4, 6) (0, 2) = 8 :=
by
  sorry

end triangle_area_l1730_173039


namespace area_of_large_rectangle_l1730_173048

-- Define the given areas for the sub-shapes
def shaded_square_area : ℝ := 4
def bottom_rectangle_area : ℝ := 2
def right_rectangle_area : ℝ := 6

-- Prove the total area of the large rectangle EFGH is 12 square inches
theorem area_of_large_rectangle : shaded_square_area + bottom_rectangle_area + right_rectangle_area = 12 := 
by 
sorry

end area_of_large_rectangle_l1730_173048


namespace binom_30_3_eq_4060_l1730_173040

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l1730_173040


namespace determine_fake_coin_weight_l1730_173035

theorem determine_fake_coin_weight
  (coins : Fin 25 → ℤ) 
  (fake_coin : Fin 25) 
  (all_same_weight : ∀ (i j : Fin 25), i ≠ fake_coin → j ≠ fake_coin → coins i = coins j)
  (fake_diff_weight : ∃ (x : Fin 25), (coins x ≠ coins fake_coin)) :
  ∃ (is_heavy : Bool), 
    (is_heavy = true ↔ coins fake_coin > coins (Fin.ofNat 0)) ∨ 
    (is_heavy = false ↔ coins fake_coin < coins (Fin.ofNat 0)) :=
  sorry

end determine_fake_coin_weight_l1730_173035


namespace largest_multiple_of_7_less_than_100_l1730_173080

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end largest_multiple_of_7_less_than_100_l1730_173080


namespace correct_range_of_x_l1730_173020

variable {x : ℝ}

noncomputable def isosceles_triangle (x y : ℝ) : Prop :=
  let perimeter := 2 * y + x
  let relationship := y = - (1/2) * x + 8
  perimeter = 16 ∧ relationship

theorem correct_range_of_x (x y : ℝ) (h : isosceles_triangle x y) : 0 < x ∧ x < 8 :=
by
  -- The proof of the theorem is omitted
  sorry

end correct_range_of_x_l1730_173020


namespace scientific_notation_1300000_l1730_173047

theorem scientific_notation_1300000 :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_1300000_l1730_173047


namespace find_width_of_lot_l1730_173027

noncomputable def volume_of_rectangular_prism (l w h : ℝ) : ℝ := l * w * h

theorem find_width_of_lot
  (l h v : ℝ)
  (h_len : l = 40)
  (h_height : h = 2)
  (h_volume : v = 1600)
  : ∃ w : ℝ, volume_of_rectangular_prism l w h = v ∧ w = 20 := by
  use 20
  simp [volume_of_rectangular_prism, h_len, h_height, h_volume]
  sorry

end find_width_of_lot_l1730_173027


namespace ratio_fifth_terms_l1730_173042

-- Define the arithmetic sequences and their sums
variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}

-- Assume conditions of the problem
axiom sum_condition (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom sum_condition2 (n : ℕ) : T n = n * (b 1 + b n) / 2
axiom ratio_condition : ∀ n, S n / T n = (2 * n - 3) / (3 * n - 2)

-- Prove the ratio of fifth terms a_5 / b_5
theorem ratio_fifth_terms : (a 5 : ℚ) / b 5 = 3 / 5 := by
  sorry

end ratio_fifth_terms_l1730_173042


namespace difference_of_squares_divisibility_l1730_173032

theorem difference_of_squares_divisibility (a b : ℤ) :
  ∃ m : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 8 * m ∧ 
           ¬∃ n : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 16 * n :=
by
  sorry

end difference_of_squares_divisibility_l1730_173032


namespace arithmetic_sequence_geometric_condition_l1730_173050

theorem arithmetic_sequence_geometric_condition :
  ∃ d : ℝ, d ≠ 0 ∧ (∀ (a_n : ℕ → ℝ), (a_n 1 = 1) ∧ 
    (a_n 3 = a_n 1 + 2 * d) ∧ (a_n 13 = a_n 1 + 12 * d) ∧ 
    (a_n 3 ^ 2 = a_n 1 * a_n 13) ↔ d = 2) :=
by 
  sorry

end arithmetic_sequence_geometric_condition_l1730_173050


namespace water_height_in_tank_l1730_173081

noncomputable def cone_radius := 10 -- in cm
noncomputable def cone_height := 15 -- in cm
noncomputable def tank_width := 20 -- in cm
noncomputable def tank_length := 30 -- in cm
noncomputable def cone_volume := (1/3:ℝ) * Real.pi * (cone_radius^2) * cone_height
noncomputable def tank_volume (h:ℝ) := tank_width * tank_length * h

theorem water_height_in_tank : ∃ h : ℝ, tank_volume h = cone_volume ∧ h = 5 * Real.pi / 6 := 
by 
  sorry

end water_height_in_tank_l1730_173081


namespace initial_deadline_in_days_l1730_173077

theorem initial_deadline_in_days
  (men_initial : ℕ)
  (days_initial : ℕ)
  (hours_per_day_initial : ℕ)
  (fraction_work_initial : ℚ)
  (additional_men : ℕ)
  (hours_per_day_additional : ℕ)
  (fraction_work_additional : ℚ)
  (total_work : ℚ := men_initial * days_initial * hours_per_day_initial)
  (remaining_days : ℚ := (men_initial * days_initial * hours_per_day_initial) / (additional_men * hours_per_day_additional * fraction_work_additional))
  (total_days : ℚ := days_initial + remaining_days) :
  men_initial = 100 →
  days_initial = 25 →
  hours_per_day_initial = 8 →
  fraction_work_initial = 1 / 3 →
  additional_men = 160 →
  hours_per_day_additional = 10 →
  fraction_work_additional = 2 / 3 →
  total_days = 37.5 :=
by
  intros
  sorry

end initial_deadline_in_days_l1730_173077


namespace count_valid_age_pairs_l1730_173095

theorem count_valid_age_pairs :
  ∃ (d n : ℕ) (a b : ℕ), 10 * a + b ≥ 30 ∧
                       10 * b + a ≥ 35 ∧
                       b > a ∧
                       ∃ k : ℕ, k = 10 := 
sorry

end count_valid_age_pairs_l1730_173095


namespace packs_of_yellow_bouncy_balls_l1730_173003

/-- Maggie bought 4 packs of red bouncy balls, some packs of yellow bouncy balls (denoted as Y), and 4 packs of green bouncy balls. -/
theorem packs_of_yellow_bouncy_balls (Y : ℕ) : 
  (4 + Y + 4) * 10 = 160 -> Y = 8 := 
by 
  sorry

end packs_of_yellow_bouncy_balls_l1730_173003


namespace compute_expression_l1730_173009

theorem compute_expression : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end compute_expression_l1730_173009


namespace outfits_count_l1730_173026

theorem outfits_count (shirts ties : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 6) : 
  (shirts * (ties + 1) = 49) :=
by
  sorry

end outfits_count_l1730_173026


namespace first_donor_amount_l1730_173096

theorem first_donor_amount
  (x second third fourth : ℝ)
  (h1 : second = 2 * x)
  (h2 : third = 3 * second)
  (h3 : fourth = 4 * third)
  (h4 : x + second + third + fourth = 132)
  : x = 4 := 
by 
  -- Simply add this line to make the theorem complete without proof.
  sorry

end first_donor_amount_l1730_173096


namespace find_a_l1730_173000

-- Definitions given in the conditions
def f (x : ℝ) : ℝ := x^2 - 2
def g (x : ℝ) : ℝ := x^2 + 6

-- The main theorem to show
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 18) : a = Real.sqrt 14 := sorry

end find_a_l1730_173000


namespace fan_rotation_is_not_translation_l1730_173011

def phenomenon := Type

def is_translation (p : phenomenon) : Prop := sorry

axiom elevator_translation : phenomenon
axiom drawer_translation : phenomenon
axiom fan_rotation : phenomenon
axiom car_translation : phenomenon

axiom elevator_is_translation : is_translation elevator_translation
axiom drawer_is_translation : is_translation drawer_translation
axiom car_is_translation : is_translation car_translation

theorem fan_rotation_is_not_translation : ¬ is_translation fan_rotation := sorry

end fan_rotation_is_not_translation_l1730_173011


namespace det_B_eq_2_l1730_173078

theorem det_B_eq_2 {x y : ℝ}
  (hB : ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), B = ![![x, 2], ![-3, y]])
  (h_eqn : ∃ (B_inv : Matrix (Fin 2) (Fin 2) ℝ),
    B_inv = (1 / (x * y + 6)) • ![![y, -2], ![3, x]] ∧
    ![![x, 2], ![-3, y]] + 2 • B_inv = 0) : 
  Matrix.det ![![x, 2], ![-3, y]] = 2 :=
by
  sorry

end det_B_eq_2_l1730_173078


namespace min_value_exists_max_value_exists_l1730_173045

noncomputable def y (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem min_value_exists :
  (∃ k : ℤ, y (π / 6 + 2 * k * π) = -2) ∧ (∃ k : ℤ, y (5 * π / 6 + 2 * k * π) = -2) :=
by 
  sorry

theorem max_value_exists :
  ∃ k : ℤ, y (-π / 2 + 2 * k * π) = 7 :=
by 
  sorry

end min_value_exists_max_value_exists_l1730_173045


namespace triangle_area_l1730_173028

theorem triangle_area (l1 l2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, l1 x y ↔ 3 * x - y + 12 = 0)
  (h2 : ∀ x y, l2 x y ↔ 3 * x + 2 * y - 6 = 0) :
  ∃ A : ℝ, A = 9 :=
by
  sorry

end triangle_area_l1730_173028


namespace asymptotic_lines_of_hyperbola_l1730_173034

open Real

-- Given: Hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- To Prove: Asymptotic lines equation
theorem asymptotic_lines_of_hyperbola : 
  ∀ x y : ℝ, hyperbola x y → (y = x ∨ y = -x) :=
by
  intros x y h
  sorry

end asymptotic_lines_of_hyperbola_l1730_173034


namespace plane_equation_correct_l1730_173057

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def plane_eq (n : Point3D) (A : Point3D) : Point3D → ℝ :=
  fun P => n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

def is_perpendicular_plane (A B C : Point3D) (D : Point3D → ℝ) : Prop :=
  let BC := vector_sub C B
  D = plane_eq BC A

theorem plane_equation_correct :
  let A := { x := 7, y := -5, z := 1 }
  let B := { x := 5, y := -1, z := -3 }
  let C := { x := 3, y := 0, z := -4 }
  is_perpendicular_plane A B C (fun P => -2 * P.x + P.y - P.z + 20) :=
by
  sorry

end plane_equation_correct_l1730_173057


namespace exists_quad_root_l1730_173067

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l1730_173067


namespace rose_paid_after_discount_l1730_173007

-- Define the conditions as given in the problem statement
def original_price : ℕ := 10
def discount_rate : ℕ := 10

-- Define the theorem that needs to be proved
theorem rose_paid_after_discount : 
  original_price - (original_price * discount_rate / 100) = 9 :=
by
  -- Here we skip the proof with sorry
  sorry

end rose_paid_after_discount_l1730_173007


namespace find_quotient_l1730_173065

-- Constants representing the given conditions
def dividend : ℕ := 690
def divisor : ℕ := 36
def remainder : ℕ := 6

-- Theorem statement
theorem find_quotient : ∃ (quotient : ℕ), dividend = (divisor * quotient) + remainder ∧ quotient = 19 := 
by
  sorry

end find_quotient_l1730_173065


namespace blue_notebook_cost_l1730_173069

theorem blue_notebook_cost
    (total_spent : ℕ)
    (total_notebooks : ℕ)
    (red_notebooks : ℕ) (red_cost : ℕ)
    (green_notebooks : ℕ) (green_cost : ℕ)
    (blue_notebooks : ℕ) (blue_total_cost : ℕ) 
    (blue_cost : ℕ) :
    total_spent = 37 →
    total_notebooks = 12 →
    red_notebooks = 3 →
    red_cost = 4 →
    green_notebooks = 2 →
    green_cost = 2 →
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks →
    blue_total_cost = total_spent - red_notebooks * red_cost - green_notebooks * green_cost →
    blue_cost = blue_total_cost / blue_notebooks →
    blue_cost = 3 := 
    by sorry

end blue_notebook_cost_l1730_173069


namespace beef_weight_after_processing_l1730_173070

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end beef_weight_after_processing_l1730_173070


namespace joan_kittens_total_l1730_173044

-- Definition of the initial conditions
def joan_original_kittens : ℕ := 8
def neighbor_original_kittens : ℕ := 6
def joan_gave_away : ℕ := 2
def neighbor_gave_away : ℕ := 4
def joan_adopted_from_neighbor : ℕ := 3

-- The final number of kittens Joan has
def joan_final_kittens : ℕ :=
  let joan_remaining := joan_original_kittens - joan_gave_away
  let neighbor_remaining := neighbor_original_kittens - neighbor_gave_away
  let adopted := min joan_adopted_from_neighbor neighbor_remaining
  joan_remaining + adopted

theorem joan_kittens_total : joan_final_kittens = 8 := 
by 
  -- Lean proof would go here, but adding sorry for now
  sorry

end joan_kittens_total_l1730_173044


namespace votes_for_eliot_l1730_173012

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end votes_for_eliot_l1730_173012


namespace cindy_first_to_get_five_l1730_173054

def probability_of_five : ℚ := 1 / 6

def anne_turn (p: ℚ) : ℚ := 1 - p
def cindy_turn (p: ℚ) : ℚ := p
def none_get_five (p: ℚ) : ℚ := (1 - p)^3

theorem cindy_first_to_get_five : 
    (∑' n, (anne_turn probability_of_five * none_get_five probability_of_five ^ n) * 
                cindy_turn probability_of_five) = 30 / 91 := by 
    sorry

end cindy_first_to_get_five_l1730_173054


namespace common_difference_minimum_sum_value_l1730_173016

variable {α : Type}
variables (a : ℕ → ℤ) (d : ℤ)
variables (S : ℕ → ℚ)

-- Conditions: Arithmetic sequence property and specific initial values
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

axiom a1_eq_neg3 : a 1 = -3
axiom condition : 11 * a 5 = 5 * a 8 - 13

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (d : ℤ) : ℚ :=
  (↑n / 2) * (2 * a 1 + ↑((n - 1) * d))

-- Prove the common difference and the minimum sum value
theorem common_difference : d = 31 / 9 :=
sorry

theorem minimum_sum_value : S 1 = -2401 / 840 :=
sorry

end common_difference_minimum_sum_value_l1730_173016


namespace shorter_piece_length_l1730_173053

-- Definitions according to conditions in a)
variables (x : ℝ) (total_length : ℝ := 140)
variables (ratio : ℝ := 5 / 2)

-- Statement to be proved
theorem shorter_piece_length : x + ratio * x = total_length → x = 40 := 
by
  intros h
  sorry

end shorter_piece_length_l1730_173053


namespace original_houses_count_l1730_173098

namespace LincolnCounty

-- Define the constants based on the conditions
def houses_built_during_boom : ℕ := 97741
def houses_now : ℕ := 118558

-- Statement of the theorem
theorem original_houses_count : houses_now - houses_built_during_boom = 20817 := 
by sorry

end LincolnCounty

end original_houses_count_l1730_173098


namespace monomial_sum_l1730_173037

theorem monomial_sum (m n : ℤ) (h1 : n = 2) (h2 : m + 2 = 1) : m + n = 1 := by
  sorry

end monomial_sum_l1730_173037


namespace number_of_bicycles_l1730_173072

theorem number_of_bicycles (B T : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 := by
  sorry

end number_of_bicycles_l1730_173072


namespace max_omega_value_l1730_173001

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

def center_of_symmetry (ω φ : ℝ) := 
  ∃ n : ℤ, ω * (-Real.pi / 4) + φ = n * Real.pi

def extremum_point (ω φ : ℝ) :=
  ∃ n' : ℤ, ω * (Real.pi / 4) + φ = n' * Real.pi + Real.pi / 2

def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x ≤ y → f x ≤ f y

theorem max_omega_value (ω : ℝ) (φ : ℝ) : 
  (ω > 0) →
  (|φ| ≤ Real.pi / 2) →
  center_of_symmetry ω φ →
  extremum_point ω φ →
  monotonic_in_interval (f ω φ) (5 * Real.pi / 18) (2 * Real.pi / 5) →
  ω = 5 :=
by
  sorry

end max_omega_value_l1730_173001


namespace recorder_price_new_l1730_173097

theorem recorder_price_new (a b : ℕ) (h1 : 10 * a + b < 50) (h2 : 10 * b + a = (10 * a + b) * 12 / 10) :
  10 * b + a = 54 :=
by
  sorry

end recorder_price_new_l1730_173097


namespace set_intersection_A_B_l1730_173066

def A := {x : ℝ | 2 * x - x^2 > 0}
def B := {x : ℝ | x > 1}
def I := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_A_B :
  A ∩ B = I :=
sorry

end set_intersection_A_B_l1730_173066


namespace tetrahedron_cube_volume_ratio_l1730_173046

theorem tetrahedron_cube_volume_ratio (s : ℝ) (h_s : s > 0):
    let V_cube := s ^ 3
    let a := s * Real.sqrt 3
    let V_tetrahedron := (Real.sqrt 2 / 12) * a ^ 3
    (V_tetrahedron / V_cube) = (Real.sqrt 6 / 4) := by
    sorry

end tetrahedron_cube_volume_ratio_l1730_173046


namespace largest_divisor_of_n4_minus_n2_is_12_l1730_173073

theorem largest_divisor_of_n4_minus_n2_is_12 : ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  intro n
  -- Placeholder for proof; the detailed steps of the proof go here
  sorry

end largest_divisor_of_n4_minus_n2_is_12_l1730_173073


namespace percentage_le_29_l1730_173029

def sample_size : ℕ := 100
def freq_17_19 : ℕ := 1
def freq_19_21 : ℕ := 1
def freq_21_23 : ℕ := 3
def freq_23_25 : ℕ := 3
def freq_25_27 : ℕ := 18
def freq_27_29 : ℕ := 16
def freq_29_31 : ℕ := 28
def freq_31_33 : ℕ := 30

theorem percentage_le_29 : (freq_17_19 + freq_19_21 + freq_21_23 + freq_23_25 + freq_25_27 + freq_27_29) * 100 / sample_size = 42 :=
by
  sorry

end percentage_le_29_l1730_173029


namespace combined_resistance_parallel_l1730_173074

theorem combined_resistance_parallel (x y : ℝ) (r : ℝ) (hx : x = 3) (hy : y = 5) 
  (h : 1 / r = 1 / x + 1 / y) : r = 15 / 8 :=
by
  sorry

end combined_resistance_parallel_l1730_173074


namespace tan_alpha_half_l1730_173060

theorem tan_alpha_half (α: ℝ) (h: Real.tan α = 1/2) :
  (1 + 2 * Real.sin (Real.pi - α) * Real.cos (-2 * Real.pi - α)) / (Real.sin (-α)^2 - Real.sin (5 * Real.pi / 2 - α)^2) = -3 := 
by
  sorry

end tan_alpha_half_l1730_173060


namespace range_of_a_l1730_173071

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a + 3

theorem range_of_a :
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ g x2 a = 0 ∧ |x1 - x2| ≤ 1) ↔ (a ∈ Set.Icc 2 3) := sorry

end range_of_a_l1730_173071


namespace find_magnitude_a_l1730_173052

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c (m : ℝ) : ℝ × ℝ := (2, m)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem find_magnitude_a (m : ℝ) (h : dot_product (vector_add (vector_a m) (vector_c m)) (vector_b m) = 0) :
  magnitude (vector_a (-1 / 2)) = Real.sqrt 2 :=
by
  sorry

end find_magnitude_a_l1730_173052


namespace domain_ln_x_squared_minus_2_l1730_173076

theorem domain_ln_x_squared_minus_2 (x : ℝ) : 
  x^2 - 2 > 0 ↔ (x < -Real.sqrt 2 ∨ x > Real.sqrt 2) := 
by 
  sorry

end domain_ln_x_squared_minus_2_l1730_173076


namespace nonneg_or_nonpos_l1730_173094

theorem nonneg_or_nonpos (n : ℕ) (h : n ≥ 2) (c : Fin n → ℝ)
  (h_eq : (n - 1) * (Finset.univ.sum (fun i => c i ^ 2)) = (Finset.univ.sum c) ^ 2) :
  (∀ i, c i ≥ 0) ∨ (∀ i, c i ≤ 0) := 
  sorry

end nonneg_or_nonpos_l1730_173094


namespace correct_statements_for_sequence_l1730_173084

theorem correct_statements_for_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  -- Statement 1
  (S_n = n^2 + n → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 2
  (S_n = 2^n - 1 → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1)) ∧
  -- Statement 3
  (∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1) → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 4
  (¬(∀ n, n ≥ 2 → a n^2 = a (n + 1) * a (n - 1) → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1))) :=
sorry

end correct_statements_for_sequence_l1730_173084


namespace arithmetic_sequence_a4_l1730_173075

theorem arithmetic_sequence_a4 (a : ℕ → ℤ) (a2 a4 a3 : ℤ) (S5 : ℤ)
  (h₁ : S5 = 25)
  (h₂ : a 2 = 3)
  (h₃ : S5 = a 1 + a 2 + a 3 + a 4 + a 5)
  (h₄ : a 3 = (a 1 + a 5) / 2)
  (h₅ : ∀ n : ℕ, (a (n+1) - a n) = (a 2 - a 1)) :
  a 4 = 7 := by
  sorry

end arithmetic_sequence_a4_l1730_173075


namespace allison_total_supply_items_is_28_l1730_173079

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l1730_173079


namespace triangle_lattice_points_l1730_173068

theorem triangle_lattice_points :
  ∀ (A B C : ℕ) (AB AC BC : ℕ), 
    AB = 2016 → AC = 1533 → BC = 1533 → 
    ∃ lattice_points: ℕ, lattice_points = 1165322 := 
by
  sorry

end triangle_lattice_points_l1730_173068


namespace range_of_a_ineq_l1730_173004

noncomputable def range_of_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ x₁ * x₁ + (a * a - 1) * x₁ + (a - 2) = 0 ∧
                x₂ * x₂ + (a * a - 1) * x₂ + (a - 2) = 0

theorem range_of_a_ineq (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧
    x₁^2 + (a^2 - 1) * x₁ + (a - 2) = 0 ∧
    x₂^2 + (a^2 - 1) * x₂ + (a - 2) = 0) → -2 < a ∧ a < 1 :=
sorry

end range_of_a_ineq_l1730_173004


namespace tan_half_sum_l1730_173061

variable (p q : ℝ)

-- Given conditions
def cos_condition : Prop := (Real.cos p + Real.cos q = 1 / 3)
def sin_condition : Prop := (Real.sin p + Real.sin q = 4 / 9)

-- Prove the target expression
theorem tan_half_sum (h1 : cos_condition p q) (h2 : sin_condition p q) : 
  Real.tan ((p + q) / 2) = 4 / 3 :=
sorry

-- For better readability, I included variable declarations and definitions separately

end tan_half_sum_l1730_173061


namespace greatest_power_of_two_factor_l1730_173024

theorem greatest_power_of_two_factor (a b c d : ℕ) (h1 : a = 10) (h2 : b = 1006) (h3 : c = 6) (h4 : d = 503) :
  ∃ k : ℕ, 2^k ∣ (a^b - c^d) ∧ ∀ j : ℕ, 2^j ∣ (a^b - c^d) → j ≤ 503 :=
sorry

end greatest_power_of_two_factor_l1730_173024


namespace fraction_evaluation_l1730_173063

theorem fraction_evaluation : (20 + 24) / (20 - 24) = -11 := by
  sorry

end fraction_evaluation_l1730_173063


namespace allowable_rectangular_formations_count_l1730_173005

theorem allowable_rectangular_formations_count (s t f : ℕ) 
  (h1 : s * t = 240)
  (h2 : Nat.Prime s)
  (h3 : 8 ≤ t ∧ t ≤ 30)
  (h4 : f ≤ 8)
  : f = 0 :=
sorry

end allowable_rectangular_formations_count_l1730_173005


namespace elena_deductions_in_cents_l1730_173022

-- Definitions based on the conditions
def cents_per_dollar : ℕ := 100
def hourly_wage_in_dollars : ℕ := 25
def hourly_wage_in_cents : ℕ := hourly_wage_in_dollars * cents_per_dollar
def tax_rate : ℚ := 0.02
def health_benefit_rate : ℚ := 0.015

-- The problem to prove
theorem elena_deductions_in_cents:
  (tax_rate * hourly_wage_in_cents) + (health_benefit_rate * hourly_wage_in_cents) = 87.5 := 
by
  sorry

end elena_deductions_in_cents_l1730_173022


namespace repeating_decimal_as_fraction_l1730_173006

theorem repeating_decimal_as_fraction :
  ∃ x : ℝ, x = 7.45 ∧ (100 * x - x = 738) → x = 82 / 11 :=
by
  sorry

end repeating_decimal_as_fraction_l1730_173006


namespace Jessica_victory_l1730_173059

def bullseye_points : ℕ := 10
def other_possible_scores : Set ℕ := {0, 2, 5, 8, 10}
def minimum_score_per_shot : ℕ := 2
def shots_taken : ℕ := 40
def remaining_shots : ℕ := 40
def jessica_advantage : ℕ := 30

def victory_condition (n : ℕ) : Prop :=
  8 * n + 80 > 370

theorem Jessica_victory :
  ∃ n, victory_condition n ∧ n = 37 :=
by
  use 37
  sorry

end Jessica_victory_l1730_173059


namespace ball_distribution_l1730_173033

theorem ball_distribution (basketballs volleyballs classes balls : ℕ) 
  (h1 : basketballs = 2) 
  (h2 : volleyballs = 3) 
  (h3 : classes = 4) 
  (h4 : balls = 4) :
  (classes.choose 3) + (classes.choose 2) = 10 :=
by
  sorry

end ball_distribution_l1730_173033


namespace isosceles_right_triangle_side_length_l1730_173082

theorem isosceles_right_triangle_side_length
  (a b : ℝ)
  (h_triangle : a = b ∨ b = a)
  (h_hypotenuse : xy > yz)
  (h_area : (1 / 2) * a * b = 9) :
  xy = 6 :=
by
  -- proof will go here
  sorry

end isosceles_right_triangle_side_length_l1730_173082


namespace weight_of_dry_grapes_l1730_173041

def fresh_grapes : ℝ := 10 -- weight of fresh grapes in kg
def fresh_water_content : ℝ := 0.90 -- fresh grapes contain 90% water by weight
def dried_water_content : ℝ := 0.20 -- dried grapes contain 20% water by weight

theorem weight_of_dry_grapes : 
  (fresh_grapes * (1 - fresh_water_content)) / (1 - dried_water_content) = 1.25 := 
by 
  sorry

end weight_of_dry_grapes_l1730_173041


namespace cone_volume_l1730_173002

theorem cone_volume (l : ℝ) (θ : ℝ) (h r V : ℝ)
  (h_l : l = 5)
  (h_θ : θ = (8 * Real.pi) / 5)
  (h_arc_length : 2 * Real.pi * r = l * θ)
  (h_radius: r = 4)
  (h_height : h = Real.sqrt (l^2 - r^2))
  (h_volume_eq : V = (1 / 3) * Real.pi * r^2 * h) :
  V = 16 * Real.pi :=
by
  -- proof goes here
  sorry

end cone_volume_l1730_173002


namespace two_a_minus_two_d_eq_zero_l1730_173086

noncomputable def g (a b c d x : ℝ) : ℝ := (2 * a * x - b) / (c * x - 2 * d)

theorem two_a_minus_two_d_eq_zero (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : ∀ x : ℝ, (g a a c d (g a b c d x)) = x) : 2 * a - 2 * d = 0 :=
sorry

end two_a_minus_two_d_eq_zero_l1730_173086
