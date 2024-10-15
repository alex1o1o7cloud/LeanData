import Mathlib

namespace NUMINAMATH_GPT_points_lie_on_line_l2161_216112

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t
  let y := (t - 1) / t
  x + y = 2 := by
  sorry

end NUMINAMATH_GPT_points_lie_on_line_l2161_216112


namespace NUMINAMATH_GPT_parallel_lines_a_eq_2_l2161_216100

theorem parallel_lines_a_eq_2 {a : ℝ} :
  (∀ x y : ℝ, a * x + (a + 2) * y + 2 = 0 ∧ x + a * y - 2 = 0 → False) ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_eq_2_l2161_216100


namespace NUMINAMATH_GPT_num_two_digit_primes_with_ones_digit_3_l2161_216123

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end NUMINAMATH_GPT_num_two_digit_primes_with_ones_digit_3_l2161_216123


namespace NUMINAMATH_GPT_number_of_new_books_l2161_216182

-- Defining the given conditions
def adventure_books : ℕ := 24
def mystery_books : ℕ := 37
def used_books : ℕ := 18

-- Defining the total books and new books
def total_books : ℕ := adventure_books + mystery_books
def new_books : ℕ := total_books - used_books

-- Proving the number of new books
theorem number_of_new_books : new_books = 43 := by
  -- Here we need to show that the calculated number of new books equals 43
  sorry

end NUMINAMATH_GPT_number_of_new_books_l2161_216182


namespace NUMINAMATH_GPT_min_cells_marked_l2161_216103

theorem min_cells_marked (grid_size : ℕ) (triomino_size : ℕ) (total_cells : ℕ) : 
  grid_size = 5 ∧ triomino_size = 3 ∧ total_cells = grid_size * grid_size → ∃ m, m = 9 :=
by
  intros h
  -- Placeholder for detailed proof steps
  sorry

end NUMINAMATH_GPT_min_cells_marked_l2161_216103


namespace NUMINAMATH_GPT_emerson_rowed_last_part_l2161_216128

-- Define the given conditions
def emerson_initial_distance: ℝ := 6
def emerson_continued_distance: ℝ := 15
def total_trip_distance: ℝ := 39

-- Define the distance Emerson covered before the last part
def distance_before_last_part := emerson_initial_distance + emerson_continued_distance

-- Define the distance Emerson rowed in the last part of his trip
def distance_last_part := total_trip_distance - distance_before_last_part

-- The theorem we need to prove
theorem emerson_rowed_last_part : distance_last_part = 18 := by
  sorry

end NUMINAMATH_GPT_emerson_rowed_last_part_l2161_216128


namespace NUMINAMATH_GPT_minimize_y_l2161_216197

variables (a b k : ℝ)

def y (x : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y : ∃ x : ℝ, y a b k x = y a b k ( (6 * a + 2 * b - k) / 8 ) :=
  sorry

end NUMINAMATH_GPT_minimize_y_l2161_216197


namespace NUMINAMATH_GPT_Ruth_math_class_percentage_l2161_216135

theorem Ruth_math_class_percentage :
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  (hours_math_week / total_school_hours_week) * 100 = 25 := 
by 
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  -- skip the proof here
  sorry

end NUMINAMATH_GPT_Ruth_math_class_percentage_l2161_216135


namespace NUMINAMATH_GPT_opposite_numbers_abs_eq_l2161_216157

theorem opposite_numbers_abs_eq (a : ℚ) : abs a = abs (-a) :=
by
  sorry

end NUMINAMATH_GPT_opposite_numbers_abs_eq_l2161_216157


namespace NUMINAMATH_GPT_james_second_hour_distance_l2161_216129

theorem james_second_hour_distance :
  ∃ x : ℝ, 
    x + 1.20 * x + 1.50 * x = 37 ∧ 
    1.20 * x = 12 :=
by
  sorry

end NUMINAMATH_GPT_james_second_hour_distance_l2161_216129


namespace NUMINAMATH_GPT_tan_cos_sin_fraction_l2161_216156

theorem tan_cos_sin_fraction (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_tan_cos_sin_fraction_l2161_216156


namespace NUMINAMATH_GPT_lucy_fish_count_l2161_216117

theorem lucy_fish_count (initial_fish : ℕ) (additional_fish : ℕ) (final_fish : ℕ) : 
  initial_fish = 212 ∧ additional_fish = 68 → final_fish = 280 :=
by
  sorry

end NUMINAMATH_GPT_lucy_fish_count_l2161_216117


namespace NUMINAMATH_GPT_area_of_figure_l2161_216174

noncomputable def area_enclosed : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi / 3), 2 * Real.sin x

theorem area_of_figure :
  area_enclosed = 3 := by
  sorry

end NUMINAMATH_GPT_area_of_figure_l2161_216174


namespace NUMINAMATH_GPT_max_product_300_l2161_216107

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end NUMINAMATH_GPT_max_product_300_l2161_216107


namespace NUMINAMATH_GPT_Danai_can_buy_more_decorations_l2161_216155

theorem Danai_can_buy_more_decorations :
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let pumpkins := 24 -- 2 times the number of spiderwebs
  let cauldron := 1
  let planned_total := 83
  let budget_left := 10
  let current_decorations := skulls + broomsticks + spiderwebs + pumpkins + cauldron
  current_decorations = 53 → -- 12 + 4 + 12 + 24 + 1
  let additional_decorations_needed := planned_total - current_decorations
  additional_decorations_needed = 30 → -- 83 - 53
  (additional_decorations_needed - budget_left) = 20 → -- 30 - 10
  True := -- proving the statement
sorry

end NUMINAMATH_GPT_Danai_can_buy_more_decorations_l2161_216155


namespace NUMINAMATH_GPT_find_largest_integer_l2161_216188

theorem find_largest_integer : ∃ (x : ℤ), x < 120 ∧ x % 8 = 7 ∧ x = 119 := 
by
  use 119
  sorry

end NUMINAMATH_GPT_find_largest_integer_l2161_216188


namespace NUMINAMATH_GPT_greatest_product_of_two_integers_with_sum_300_l2161_216193

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end NUMINAMATH_GPT_greatest_product_of_two_integers_with_sum_300_l2161_216193


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2161_216162

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) / (x + 3) ≥ 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2161_216162


namespace NUMINAMATH_GPT_maria_initial_carrots_l2161_216173

theorem maria_initial_carrots (C : ℕ) (h : C - 11 + 15 = 52) : C = 48 :=
by
  sorry

end NUMINAMATH_GPT_maria_initial_carrots_l2161_216173


namespace NUMINAMATH_GPT_remainder_problem_l2161_216124

theorem remainder_problem
  (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : 196 % x = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l2161_216124


namespace NUMINAMATH_GPT_derivative_at_2_l2161_216152

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_2 : (deriv f 2) = -120 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_2_l2161_216152


namespace NUMINAMATH_GPT_eq1_eq2_eq3_eq4_l2161_216180

theorem eq1 (x : ℚ) : 3 * x^2 - 32 * x - 48 = 0 ↔ (x = 12 ∨ x = -4/3) := sorry

theorem eq2 (x : ℚ) : 4 * x^2 + x - 3 = 0 ↔ (x = 3/4 ∨ x = -1) := sorry

theorem eq3 (x : ℚ) : (3 * x + 1)^2 - 4 = 0 ↔ (x = 1/3 ∨ x = -1) := sorry

theorem eq4 (x : ℚ) : 9 * (x - 2)^2 = 4 * (x + 1)^2 ↔ (x = 8 ∨ x = 4/5) := sorry

end NUMINAMATH_GPT_eq1_eq2_eq3_eq4_l2161_216180


namespace NUMINAMATH_GPT_area_triangle_sum_l2161_216161

theorem area_triangle_sum (AB : ℝ) (angle_BAC angle_ABC angle_ACB angle_EDC : ℝ) 
  (h_AB : AB = 1) (h_angle_BAC : angle_BAC = 70) (h_angle_ABC : angle_ABC = 50) 
  (h_angle_ACB : angle_ACB = 60) (h_angle_EDC : angle_EDC = 80) :
  let area_triangle := (1/2) * AB * (Real.sin angle_70 / Real.sin angle_60) * (Real.sin angle_60) 
  let area_CDE := (1/2) * (Real.sin angle_80)
  area_triangle + 2 * area_CDE = (Real.sin angle_70 + Real.sin angle_80) / 2 :=
sorry

end NUMINAMATH_GPT_area_triangle_sum_l2161_216161


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l2161_216105

theorem molecular_weight_of_one_mole (molecular_weight_8_moles : ℝ) (h : molecular_weight_8_moles = 992) : 
  molecular_weight_8_moles / 8 = 124 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l2161_216105


namespace NUMINAMATH_GPT_find_angle_C_find_max_perimeter_l2161_216118

-- Define the first part of the problem
theorem find_angle_C 
  (a b c A B C : ℝ) (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  C = (2 * Real.pi) / 3 :=
sorry

-- Define the second part of the problem
theorem find_max_perimeter 
  (a b A B : ℝ)
  (C : ℝ := (2 * Real.pi) / 3)
  (c : ℝ := Real.sqrt 3)
  (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  (2 * Real.sqrt 3 < a + b + c) ∧ (a + b + c <= 2 + Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_max_perimeter_l2161_216118


namespace NUMINAMATH_GPT_geom_series_sum_l2161_216171

def a : ℚ := 1 / 3
def r : ℚ := 2 / 3
def n : ℕ := 9

def S_n (a r : ℚ) (n : ℕ) := a * (1 - r^n) / (1 - r)

theorem geom_series_sum :
  S_n a r n = 19171 / 19683 := by
    sorry

end NUMINAMATH_GPT_geom_series_sum_l2161_216171


namespace NUMINAMATH_GPT_sequence_values_l2161_216191

theorem sequence_values (x y z : ℚ) :
  (∀ n : ℕ, x = 1 ∧ y = 9 / 8 ∧ z = 5 / 4) :=
by
  sorry

end NUMINAMATH_GPT_sequence_values_l2161_216191


namespace NUMINAMATH_GPT_joy_sixth_time_is_87_seconds_l2161_216179

def sixth_time (times : List ℝ) (new_median : ℝ) : ℝ :=
  let sorted_times := times |>.insertNth 2 (2 * new_median - times.nthLe 2 sorry)
  2 * new_median - times.nthLe 2 sorry

theorem joy_sixth_time_is_87_seconds (times : List ℝ) (new_median : ℝ) :
  times = [82, 85, 93, 95, 99] → new_median = 90 →
  sixth_time times new_median = 87 :=
by
  intros h_times h_median
  rw [h_times]
  rw [h_median]
  sorry

end NUMINAMATH_GPT_joy_sixth_time_is_87_seconds_l2161_216179


namespace NUMINAMATH_GPT_ELMO_value_l2161_216143

def digits := {n : ℕ // n < 10}

variables (L E T M O : digits)

-- Conditions
axiom h1 : L.val ≠ 0
axiom h2 : O.val = 0
axiom h3 : (1000 * L.val + 100 * E.val + 10 * E.val + T.val) + (100 * L.val + 10 * M.val + T.val) = 1000 * T.val + L.val

-- Conclusion
theorem ELMO_value : E.val * 1000 + L.val * 100 + M.val * 10 + O.val = 1880 :=
sorry

end NUMINAMATH_GPT_ELMO_value_l2161_216143


namespace NUMINAMATH_GPT_circle_equation_bisects_l2161_216138

-- Define the given conditions
def circle1_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def circle2_eq (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define the goal equation
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 = 81

-- The statement of the problem
theorem circle_equation_bisects (a r : ℝ) (h1 : ∀ x y, circle1_eq x y → circleC_eq x y) (h2 : ∀ x y, circle2_eq x y → circleC_eq x y):
  circleC_eq (a * r) 0 := sorry

end NUMINAMATH_GPT_circle_equation_bisects_l2161_216138


namespace NUMINAMATH_GPT_fill_tank_time_l2161_216166

-- Define the rates of filling and draining
def rateA : ℕ := 200 -- Pipe A fills at 200 liters per minute
def rateB : ℕ := 50  -- Pipe B fills at 50 liters per minute
def rateC : ℕ := 25  -- Pipe C drains at 25 liters per minute

-- Define the times each pipe is open
def timeA : ℕ := 1   -- Pipe A is open for 1 minute
def timeB : ℕ := 2   -- Pipe B is open for 2 minutes
def timeC : ℕ := 2   -- Pipe C is open for 2 minutes

-- Define the capacity of the tank
def tankCapacity : ℕ := 1000

-- Prove the total time to fill the tank is 20 minutes
theorem fill_tank_time : 
  (tankCapacity * ((timeA * rateA + timeB * rateB) - (timeC * rateC)) * 5) = 20 :=
sorry

end NUMINAMATH_GPT_fill_tank_time_l2161_216166


namespace NUMINAMATH_GPT_proof_5x_plus_4_l2161_216153

variable (x : ℝ)

-- Given condition
def condition := 5 * x - 8 = 15 * x + 12

-- Required proof
theorem proof_5x_plus_4 (h : condition x) : 5 * (x + 4) = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_5x_plus_4_l2161_216153


namespace NUMINAMATH_GPT_n_sum_of_two_squares_l2161_216148

theorem n_sum_of_two_squares (n : ℤ) (m : ℤ) (hn_gt_2 : n > 2) (hn2_eq_diff_cubes : n^2 = (m+1)^3 - m^3) : 
  ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_n_sum_of_two_squares_l2161_216148


namespace NUMINAMATH_GPT_find_P_Q_R_l2161_216139

theorem find_P_Q_R :
  ∃ P Q R : ℝ, (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → 
    (5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2)) 
    ∧ P = 5 ∧ Q = -5 ∧ R = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_P_Q_R_l2161_216139


namespace NUMINAMATH_GPT_reduce_fraction_l2161_216165

-- Defining a structure for a fraction
structure Fraction where
  num : ℕ
  denom : ℕ
  deriving Repr

-- The original fraction
def originalFraction : Fraction :=
  { num := 368, denom := 598 }

-- The reduced fraction
def reducedFraction : Fraction :=
  { num := 184, denom := 299 }

-- The statement of our theorem
theorem reduce_fraction :
  ∃ (d : ℕ), d > 0 ∧ (originalFraction.num / d = reducedFraction.num) ∧ (originalFraction.denom / d = reducedFraction.denom) := by
  sorry

end NUMINAMATH_GPT_reduce_fraction_l2161_216165


namespace NUMINAMATH_GPT_find_sum_of_natural_numbers_l2161_216121

theorem find_sum_of_natural_numbers :
  ∃ (square triangle : ℕ), square^2 + 12 = triangle^2 ∧ square + triangle = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_natural_numbers_l2161_216121


namespace NUMINAMATH_GPT_base_length_first_tri_sail_l2161_216187

-- Define the areas of the sails
def area_rect_sail : ℕ := 5 * 8
def area_second_tri_sail : ℕ := (4 * 6) / 2

-- Total canvas area needed
def total_canvas_area_needed : ℕ := 58

-- Calculate the total area so far (rectangular sail + second triangular sail)
def total_area_so_far : ℕ := area_rect_sail + area_second_tri_sail

-- Define the height of the first triangular sail
def height_first_tri_sail : ℕ := 4

-- Define the area needed for the first triangular sail
def area_first_tri_sail : ℕ := total_canvas_area_needed - total_area_so_far

-- Prove that the base length of the first triangular sail is 3 inches
theorem base_length_first_tri_sail : ∃ base : ℕ, base = 3 ∧ (base * height_first_tri_sail) / 2 = area_first_tri_sail := by
  use 3
  have h1 : (3 * 4) / 2 = 6 := by sorry -- This is a placeholder for actual calculation
  exact ⟨rfl, h1⟩

end NUMINAMATH_GPT_base_length_first_tri_sail_l2161_216187


namespace NUMINAMATH_GPT_sin_2theta_in_third_quadrant_l2161_216183

open Real

variables (θ : ℝ)

/-- \theta is an angle in the third quadrant.
Given that \(\sin^{4}\theta + \cos^{4}\theta = \frac{5}{9}\), 
prove that \(\sin 2\theta = \frac{2\sqrt{2}}{3}\). --/
theorem sin_2theta_in_third_quadrant (h_theta_third_quadrant : π < θ ∧ θ < 3 * π / 2)
(h_cond : sin θ ^ 4 + cos θ ^ 4 = 5 / 9) : sin (2 * θ) = 2 * sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_sin_2theta_in_third_quadrant_l2161_216183


namespace NUMINAMATH_GPT_arithmetic_series_sum_l2161_216164

theorem arithmetic_series_sum : 
  ∀ (a d a_n : ℤ), 
  a = -48 → d = 2 → a_n = 0 → 
  ∃ n S : ℤ, 
  a + (n - 1) * d = a_n ∧ 
  S = n * (a + a_n) / 2 ∧ 
  S = -600 :=
by
  intros a d a_n ha hd han
  have h₁ : a = -48 := ha
  have h₂ : d = 2 := hd
  have h₃ : a_n = 0 := han
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l2161_216164


namespace NUMINAMATH_GPT_find_triangle_height_l2161_216194

-- Given conditions
def triangle_area : ℝ := 960
def base : ℝ := 48

-- The problem is to find the height such that 960 = (1/2) * 48 * height
theorem find_triangle_height (height : ℝ) 
  (h_area : triangle_area = (1/2) * base * height) : height = 40 := by
  sorry

end NUMINAMATH_GPT_find_triangle_height_l2161_216194


namespace NUMINAMATH_GPT_problem_statement_l2161_216199

theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) = Real.sqrt m - Real.sqrt n) →
  m + n = 2011 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2161_216199


namespace NUMINAMATH_GPT_employee_n_salary_l2161_216134

theorem employee_n_salary (m n : ℝ) (h1 : m = 1.2 * n) (h2 : m + n = 594) :
  n = 270 :=
sorry

end NUMINAMATH_GPT_employee_n_salary_l2161_216134


namespace NUMINAMATH_GPT_find_x_plus_y_l2161_216169

theorem find_x_plus_y (x y : Real) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 2009 + Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l2161_216169


namespace NUMINAMATH_GPT_average_of_measurements_l2161_216132

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80.0, 79.6, 80.5]

theorem average_of_measurements : (measurements.sum / measurements.length) = 80 := by sorry

end NUMINAMATH_GPT_average_of_measurements_l2161_216132


namespace NUMINAMATH_GPT_max_a_for_no_lattice_point_l2161_216145

theorem max_a_for_no_lattice_point (a : ℝ) (hm : ∀ m : ℝ, 1 / 2 < m ∧ m < a → ¬ ∃ x y : ℤ, 0 < x ∧ x ≤ 200 ∧ y = m * x + 3) : 
  a = 101 / 201 :=
sorry

end NUMINAMATH_GPT_max_a_for_no_lattice_point_l2161_216145


namespace NUMINAMATH_GPT_infinite_series_sum_l2161_216147

noncomputable def inf_series (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), if n = 1 then 1 / (b * a)
  else if n % 2 = 0 then 1 / ((↑(n - 1) * a - b) * (↑n * a - b))
  else 1 / ((↑(n - 1) * a + b) * (↑n * a - b))

theorem infinite_series_sum (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  inf_series a b = 1 / (a * b) :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l2161_216147


namespace NUMINAMATH_GPT_Pablo_puzzle_completion_l2161_216111

theorem Pablo_puzzle_completion :
  let pieces_per_hour := 100
  let puzzles_400 := 15
  let pieces_per_puzzle_400 := 400
  let puzzles_700 := 10
  let pieces_per_puzzle_700 := 700
  let daily_work_hours := 6
  let daily_work_400_hours := 4
  let daily_work_700_hours := 2
  let break_every_hours := 2
  let break_time := 30 / 60   -- 30 minutes break in hours

  let total_pieces_400 := puzzles_400 * pieces_per_puzzle_400
  let total_pieces_700 := puzzles_700 * pieces_per_puzzle_700
  let total_pieces := total_pieces_400 + total_pieces_700

  let effective_daily_hours := daily_work_hours - (daily_work_hours / break_every_hours * break_time)
  let pieces_400_per_day := daily_work_400_hours * pieces_per_hour
  let pieces_700_per_day := (effective_daily_hours - daily_work_400_hours) * pieces_per_hour
  let total_pieces_per_day := pieces_400_per_day + pieces_700_per_day
  
  total_pieces / total_pieces_per_day = 26 := by
sorry

end NUMINAMATH_GPT_Pablo_puzzle_completion_l2161_216111


namespace NUMINAMATH_GPT_rationalize_sqrt_l2161_216163

theorem rationalize_sqrt (h : Real.sqrt 35 ≠ 0) : 35 / Real.sqrt 35 = Real.sqrt 35 := 
by 
sorry

end NUMINAMATH_GPT_rationalize_sqrt_l2161_216163


namespace NUMINAMATH_GPT_simplify_expression_evaluate_expression_l2161_216122

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3 * a / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1)) = a / (a - 2) :=
by sorry

theorem evaluate_expression :
  (-2 - 3 * (-2) / (-2 + 1)) / (((-2)^2 - 4 * (-2) + 4) / (-2 + 1)) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_evaluate_expression_l2161_216122


namespace NUMINAMATH_GPT_find_shirt_cost_l2161_216137

def cost_each_shirt (x : ℝ) : Prop :=
  let total_purchase_price := x + 5 + 30 + 14
  let shipping_cost := if total_purchase_price > 50 then 0.2 * total_purchase_price else 5
  let total_bill := total_purchase_price + shipping_cost
  total_bill = 102

theorem find_shirt_cost (x : ℝ) (h : cost_each_shirt x) : x = 36 :=
sorry

end NUMINAMATH_GPT_find_shirt_cost_l2161_216137


namespace NUMINAMATH_GPT_systematic_sampling_first_segment_l2161_216178

theorem systematic_sampling_first_segment:
  ∀ (total_students sample_size segment_size 
     drawn_16th drawn_first : ℕ),
  total_students = 160 →
  sample_size = 20 →
  segment_size = 8 →
  drawn_16th = 125 →
  drawn_16th = drawn_first + segment_size * (16 - 1) →
  drawn_first = 5 :=
by
  intros total_students sample_size segment_size drawn_16th drawn_first
         htots hsamp hseg hdrw16 heq
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_segment_l2161_216178


namespace NUMINAMATH_GPT_maria_ends_up_with_22_towels_l2161_216195

-- Define the number of green towels Maria bought
def green_towels : Nat := 35

-- Define the number of white towels Maria bought
def white_towels : Nat := 21

-- Define the number of towels Maria gave to her mother
def given_towels : Nat := 34

-- Total towels Maria initially bought
def total_towels := green_towels + white_towels

-- Towels Maria ended up with
def remaining_towels := total_towels - given_towels

theorem maria_ends_up_with_22_towels :
  remaining_towels = 22 :=
by
  sorry

end NUMINAMATH_GPT_maria_ends_up_with_22_towels_l2161_216195


namespace NUMINAMATH_GPT_village_population_decrease_rate_l2161_216170

theorem village_population_decrease_rate :
  ∃ (R : ℝ), 15 * R = 18000 :=
by
  sorry

end NUMINAMATH_GPT_village_population_decrease_rate_l2161_216170


namespace NUMINAMATH_GPT_value_of_expression_l2161_216175

theorem value_of_expression (x y : ℝ) (h₀ : x = Real.sqrt 2 + 1) (h₁ : y = Real.sqrt 2 - 1) : 
  (x + y) * (x - y) = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2161_216175


namespace NUMINAMATH_GPT_range_of_f_l2161_216167

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * Real.sin x + 6

theorem range_of_f :
  ∀ (x : ℝ), Real.sin x ≠ 2 → 
  (1 ≤ f x ∧ f x ≤ 11) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_f_l2161_216167


namespace NUMINAMATH_GPT_sandy_distance_l2161_216102

theorem sandy_distance :
  ∃ d : ℝ, d = 18 * (1000 / 3600) * 99.9920006399488 := sorry

end NUMINAMATH_GPT_sandy_distance_l2161_216102


namespace NUMINAMATH_GPT_range_of_m_for_nonempty_solution_set_l2161_216177

theorem range_of_m_for_nonempty_solution_set :
  {m : ℝ | ∃ x : ℝ, m * x^2 - m * x + 1 < 0} = {m : ℝ | m < 0} ∪ {m : ℝ | m > 4} :=
by sorry

end NUMINAMATH_GPT_range_of_m_for_nonempty_solution_set_l2161_216177


namespace NUMINAMATH_GPT_correct_choice_D_l2161_216196

variable (a b : Line) (α : Plane)

-- Definitions for the conditions
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry  -- Definition of perpendicular
def is_parallel_line (l1 l2 : Line) : Prop := sorry  -- Definition of parallel lines
def is_parallel_plane (l : Line) (p : Plane) : Prop := sorry  -- Definition of line parallel to plane
def is_subset (l : Line) (p : Plane) : Prop := sorry  -- Definition of line being in a plane

-- The statement of the problem
theorem correct_choice_D :
  (is_parallel_plane a α) ∧ (is_subset b α) → (is_parallel_plane a α) := 
by 
  sorry

end NUMINAMATH_GPT_correct_choice_D_l2161_216196


namespace NUMINAMATH_GPT_find_somu_age_l2161_216159

noncomputable def somu_age (S F : ℕ) : Prop :=
  S = (1/3 : ℝ) * F ∧ S - 6 = (1/5 : ℝ) * (F - 6)

theorem find_somu_age {S F : ℕ} (h : somu_age S F) : S = 12 :=
by sorry

end NUMINAMATH_GPT_find_somu_age_l2161_216159


namespace NUMINAMATH_GPT_general_term_formula_for_sequence_l2161_216146

theorem general_term_formula_for_sequence (a b : ℕ → ℝ) 
  (h1 : ∀ n, 2 * b n = a n + a (n + 1)) 
  (h2 : ∀ n, (a (n + 1))^2 = b n * b (n + 1)) 
  (h3 : a 1 = 1) 
  (h4 : a 2 = 3) :
  ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_for_sequence_l2161_216146


namespace NUMINAMATH_GPT_find_a_b_c_l2161_216125

theorem find_a_b_c (a b c : ℝ) 
  (h_min : ∀ x, -9 * x^2 + 54 * x - 45 ≥ 36) 
  (h1 : 0 = a * (1 - 1) * (1 - 5)) 
  (h2 : 0 = a * (5 - 1) * (5 - 5)) :
  a + b + c = 36 :=
sorry

end NUMINAMATH_GPT_find_a_b_c_l2161_216125


namespace NUMINAMATH_GPT_black_cards_taken_out_l2161_216131

theorem black_cards_taken_out (initial_black : ℕ) (remaining_black : ℕ) (total_cards : ℕ) (black_cards_per_deck : ℕ) :
  total_cards = 52 → black_cards_per_deck = 26 →
  initial_black = black_cards_per_deck → remaining_black = 22 →
  initial_black - remaining_black = 4 := by
  intros
  sorry

end NUMINAMATH_GPT_black_cards_taken_out_l2161_216131


namespace NUMINAMATH_GPT_percentage_puppies_greater_profit_l2161_216151

/-- A dog breeder wants to know what percentage of puppies he can sell for a greater profit.
    Puppies with more than 4 spots sell for more money. The last litter had 10 puppies; 
    6 had 5 spots, 3 had 4 spots, and 1 had 2 spots.
    We need to prove that the percentage of puppies that can be sold for more profit is 60%. -/
theorem percentage_puppies_greater_profit
  (total_puppies : ℕ := 10)
  (puppies_with_5_spots : ℕ := 6)
  (puppies_with_4_spots : ℕ := 3)
  (puppies_with_2_spots : ℕ := 1)
  (puppies_with_more_than_4_spots := puppies_with_5_spots) :
  (puppies_with_more_than_4_spots : ℝ) / (total_puppies : ℝ) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_puppies_greater_profit_l2161_216151


namespace NUMINAMATH_GPT_margaret_mean_score_l2161_216133

noncomputable def cyprian_scores : List ℕ := [82, 85, 89, 91, 95, 97]
noncomputable def cyprian_mean : ℕ := 88

theorem margaret_mean_score :
  let total_sum := List.sum cyprian_scores
  let cyprian_sum := cyprian_mean * 3
  let margaret_sum := total_sum - cyprian_sum
  let margaret_mean := (margaret_sum : ℚ) / 3
  margaret_mean = 91.66666666666667 := 
by 
  -- Definitions used in conditions, skipping steps.
  sorry

end NUMINAMATH_GPT_margaret_mean_score_l2161_216133


namespace NUMINAMATH_GPT_job_time_relation_l2161_216108

theorem job_time_relation (a b c m n x : ℝ) 
  (h1 : m / a = 1 / b + 1 / c)
  (h2 : n / b = 1 / a + 1 / c)
  (h3 : x / c = 1 / a + 1 / b) :
  x = (m + n + 2) / (m * n - 1) := 
sorry

end NUMINAMATH_GPT_job_time_relation_l2161_216108


namespace NUMINAMATH_GPT_determine_min_k_l2161_216101

open Nat

theorem determine_min_k (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → ℕ) (b : Fin (choose n 2) → ℕ) : 
  ∃ k, k = (n - 1) * (n - 2) / 2 + 1 := 
sorry

end NUMINAMATH_GPT_determine_min_k_l2161_216101


namespace NUMINAMATH_GPT_kati_age_l2161_216160

/-- Define the age of Kati using the given conditions -/
theorem kati_age (kati_age : ℕ) (brother_age kati_birthdays : ℕ) 
  (h1 : kati_age = kati_birthdays) 
  (h2 : kati_age + brother_age = 111) 
  (h3 : kati_birthdays = kati_age) : 
  kati_age = 18 :=
by
  sorry

end NUMINAMATH_GPT_kati_age_l2161_216160


namespace NUMINAMATH_GPT_find_natural_number_with_common_divisor_l2161_216186

def commonDivisor (a b : ℕ) (d : ℕ) : Prop :=
  d > 1 ∧ d ∣ a ∧ d ∣ b

theorem find_natural_number_with_common_divisor :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k ≤ 20 →
    ∃ d : ℕ, commonDivisor (n + k) 30030 d) ∧ n = 9440 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_number_with_common_divisor_l2161_216186


namespace NUMINAMATH_GPT_friend_spent_11_l2161_216114

-- Definitions of the conditions
def total_lunch_cost (you friend : ℝ) : Prop := you + friend = 19
def friend_spent_more (you friend : ℝ) : Prop := friend = you + 3

-- The theorem to prove
theorem friend_spent_11 (you friend : ℝ) 
  (h1 : total_lunch_cost you friend) 
  (h2 : friend_spent_more you friend) : 
  friend = 11 := 
by 
  sorry

end NUMINAMATH_GPT_friend_spent_11_l2161_216114


namespace NUMINAMATH_GPT_woman_weaves_ten_day_units_l2161_216119

theorem woman_weaves_ten_day_units 
  (a₁ d : ℕ)
  (h₁ : 4 * a₁ + 6 * d = 24)
  (h₂ : a₁ + 6 * d = a₁ * (a₁ + d)) :
  a₁ + 9 * d = 21 := 
by
  sorry

end NUMINAMATH_GPT_woman_weaves_ten_day_units_l2161_216119


namespace NUMINAMATH_GPT_triangle_inequality_area_equality_condition_l2161_216150

theorem triangle_inequality_area (a b c S : ℝ) (h_area : S = (a * b * Real.sin (Real.arccos ((a*a + b*b - c*c) / (2*a*b)))) / 2) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by
  sorry

theorem equality_condition (a b c : ℝ) (h_eq : a = b ∧ b = c) : 
  a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * (a^2 * (Real.sqrt 3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_area_equality_condition_l2161_216150


namespace NUMINAMATH_GPT_reach_one_from_any_non_zero_l2161_216154

-- Define the game rules as functions
def remove_units_digit (n : ℕ) : ℕ :=
  n / 10

def multiply_by_two (n : ℕ) : ℕ :=
  n * 2

-- Lemma: Prove that starting from 45, we can reach 1 using the game rules.
lemma reach_one_from_45 : ∃ f : ℕ → ℕ, f 45 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Lemma: Prove that starting from 345, we can reach 1 using the game rules.
lemma reach_one_from_345 : ∃ f : ℕ → ℕ, f 345 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Theorem: Prove that any non-zero natural number can be reduced to 1 using the game rules.
theorem reach_one_from_any_non_zero (n : ℕ) (h : n ≠ 0) : ∃ f : ℕ → ℕ, f n = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_reach_one_from_any_non_zero_l2161_216154


namespace NUMINAMATH_GPT_quotient_of_division_l2161_216190

theorem quotient_of_division (Q : ℤ) (h1 : 172 = (17 * Q) + 2) : Q = 10 :=
sorry

end NUMINAMATH_GPT_quotient_of_division_l2161_216190


namespace NUMINAMATH_GPT_solve_equation_l2161_216142

variable (a b c : ℝ)

theorem solve_equation (h : (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1) : 
  a * c = 36 * b :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solve_equation_l2161_216142


namespace NUMINAMATH_GPT_baoh2_formation_l2161_216110

noncomputable def moles_of_baoh2_formed (moles_bao : ℕ) (moles_h2o : ℕ) : ℕ :=
  if moles_bao = moles_h2o then moles_bao else sorry

theorem baoh2_formation :
  moles_of_baoh2_formed 3 3 = 3 :=
by sorry

end NUMINAMATH_GPT_baoh2_formation_l2161_216110


namespace NUMINAMATH_GPT_dans_age_l2161_216116

variable {x : ℤ}

theorem dans_age (h : x + 20 = 7 * (x - 4)) : x = 8 := by
  sorry

end NUMINAMATH_GPT_dans_age_l2161_216116


namespace NUMINAMATH_GPT_gcd_1987_2025_l2161_216184

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_1987_2025_l2161_216184


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2161_216176

noncomputable def a_n (n : ℕ) : ℚ := 1 + (n - 1) / 2

noncomputable def S_n (n : ℕ) : ℚ := n * (n + 3) / 4

theorem arithmetic_sequence_problem :
  -- Given
  (∀ n, ∃ d, a_n n = a_1 + (n - 1) * d) →
  (a_n 7 = 4) →
  (a_n 19 = 2 * a_n 9) →
  -- Prove
  (∀ n, a_n n = (n + 1) / 2) ∧ (∀ n, S_n n = n * (n + 3) / 4) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2161_216176


namespace NUMINAMATH_GPT_sqrt_computation_l2161_216192

theorem sqrt_computation : 
  Real.sqrt ((35 * 34 * 33 * 32) + Nat.factorial 4) = 1114 := by
sorry

end NUMINAMATH_GPT_sqrt_computation_l2161_216192


namespace NUMINAMATH_GPT_total_cost_of_stickers_l2161_216126

-- Definitions based on given conditions
def initial_funds_per_person := 9
def cost_of_deck_of_cards := 10
def Dora_packs_of_stickers := 2

-- Calculate the total amount of money collectively after buying the deck of cards
def remaining_funds := 2 * initial_funds_per_person - cost_of_deck_of_cards

-- Calculate the total packs of stickers if split evenly
def total_packs_of_stickers := 2 * Dora_packs_of_stickers

-- Prove the total cost of the boxes of stickers
theorem total_cost_of_stickers : remaining_funds = 8 := by
  -- Given initial funds per person, cost of deck of cards, and packs of stickers for Dora, the theorem should hold.
  sorry

end NUMINAMATH_GPT_total_cost_of_stickers_l2161_216126


namespace NUMINAMATH_GPT_length_width_difference_l2161_216158

theorem length_width_difference
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 768) :
  l - w = 24 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_length_width_difference_l2161_216158


namespace NUMINAMATH_GPT_range_of_a_minus_abs_b_l2161_216104

theorem range_of_a_minus_abs_b {a b : ℝ} (h1 : 1 < a ∧ a < 3) (h2 : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_abs_b_l2161_216104


namespace NUMINAMATH_GPT_problem_solution_l2161_216113

noncomputable def negThreePower25 : Real := (-3) ^ 25
noncomputable def twoPowerExpression : Real := 2 ^ (4^2 + 5^2 - 7^2)
noncomputable def threeCubed : Real := 3^3

theorem problem_solution :
  negThreePower25 + twoPowerExpression + threeCubed = -3^25 + 27 + (1 / 256) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_problem_solution_l2161_216113


namespace NUMINAMATH_GPT_sleeves_add_correct_weight_l2161_216130

variable (R W_r W_s S : ℝ)

-- Conditions
def raw_squat : Prop := R = 600
def wraps_add_25_percent : Prop := W_r = R + 0.25 * R
def wraps_vs_sleeves_difference : Prop := W_r = W_s + 120

-- To Prove
theorem sleeves_add_correct_weight (h1 : raw_squat R) (h2 : wraps_add_25_percent R W_r) (h3 : wraps_vs_sleeves_difference W_r W_s) : S = 30 :=
by
  sorry

end NUMINAMATH_GPT_sleeves_add_correct_weight_l2161_216130


namespace NUMINAMATH_GPT_curve_crossing_point_l2161_216140

theorem curve_crossing_point :
  (∃ t : ℝ, (t^2 - 4 = 2) ∧ (t^3 - 6 * t + 4 = 4)) ∧
  (∃ t' : ℝ, t ≠ t' ∧ (t'^2 - 4 = 2) ∧ (t'^3 - 6 * t' + 4 = 4)) :=
sorry

end NUMINAMATH_GPT_curve_crossing_point_l2161_216140


namespace NUMINAMATH_GPT_units_digit_of_product_l2161_216181

/-
Problem: What is the units digit of the product of the first three even positive composite numbers?
Conditions: 
- The first three even positive composite numbers are 4, 6, and 8.
Proof: Prove that the units digit of their product is 2.
-/

def even_positive_composite_numbers := [4, 6, 8]
def product := List.foldl (· * ·) 1 even_positive_composite_numbers
def units_digit (n : Nat) := n % 10

theorem units_digit_of_product : units_digit product = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l2161_216181


namespace NUMINAMATH_GPT_coins_fit_in_new_box_l2161_216185

-- Definitions
def diameters_bound (d : ℕ) : Prop :=
  d ≤ 10

def box_fits (length width : ℕ) (fits : Prop) : Prop :=
  fits

-- Conditions
axiom coins_diameter_bound : ∀ (d : ℕ), diameters_bound d
axiom original_box_fits : box_fits 30 70 True

-- Proof statement
theorem coins_fit_in_new_box : box_fits 40 60 True :=
sorry

end NUMINAMATH_GPT_coins_fit_in_new_box_l2161_216185


namespace NUMINAMATH_GPT_average_daily_production_correct_l2161_216106

noncomputable def average_daily_production : ℝ :=
  let jan_production := 3000
  let monthly_increase := 100
  let total_days := 365
  let total_production := jan_production + (11 * jan_production + (100 * (1 + 11))/2)
  (total_production / total_days : ℝ)

theorem average_daily_production_correct :
  average_daily_production = 121.1 :=
sorry

end NUMINAMATH_GPT_average_daily_production_correct_l2161_216106


namespace NUMINAMATH_GPT_find_fraction_l2161_216136

-- Define the initial amount, the amount spent on pads, and the remaining amount
def initial_amount := 150
def spent_on_pads := 50
def remaining := 25

-- Define the fraction she spent on hockey skates
def fraction_spent_on_skates (f : ℚ) : Prop :=
  let spent_on_skates := initial_amount - remaining - spent_on_pads
  (spent_on_skates / initial_amount) = f

theorem find_fraction : fraction_spent_on_skates (1 / 2) :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_find_fraction_l2161_216136


namespace NUMINAMATH_GPT_solve_for_p_l2161_216120

theorem solve_for_p (a b c p t : ℝ) (h1 : a + b + c + p = 360) (h2 : t = 180 - c) : 
  p = 180 - a - b + t :=
by
  sorry

end NUMINAMATH_GPT_solve_for_p_l2161_216120


namespace NUMINAMATH_GPT_eccentricity_of_given_hyperbola_l2161_216189

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : b = 2 * a) : ℝ :=
  Real.sqrt (1 + (b * b) / (a * a))

theorem eccentricity_of_given_hyperbola (a b : ℝ) 
  (h_hyperbola : b = 2 * a)
  (h_asymptote : ∃ k, k = 2 ∧ ∀ x, y = k * x → ((y * a) = (b * x))) :
  hyperbola_eccentricity a b h_hyperbola = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_given_hyperbola_l2161_216189


namespace NUMINAMATH_GPT_age_difference_l2161_216198

variable (a b c d : ℕ)
variable (h1 : a + b = b + c + 11)
variable (h2 : a + c = c + d + 15)
variable (h3 : b + d = 36)
variable (h4 : a * 2 = 3 * d)

theorem age_difference :
  a - b = 39 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l2161_216198


namespace NUMINAMATH_GPT_least_milk_l2161_216115

theorem least_milk (seokjin jungkook yoongi : ℚ) (h_seokjin : seokjin = 11 / 10)
  (h_jungkook : jungkook = 1.3) (h_yoongi : yoongi = 7 / 6) :
  seokjin < jungkook ∧ seokjin < yoongi :=
by
  sorry

end NUMINAMATH_GPT_least_milk_l2161_216115


namespace NUMINAMATH_GPT_range_of_k_l2161_216168

theorem range_of_k {k : ℝ} : (∀ x : ℝ, x < 0 → (k - 2)/x > 0) ∧ (∀ x : ℝ, x > 0 → (k - 2)/x < 0) → k < 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2161_216168


namespace NUMINAMATH_GPT_find_constants_l2161_216127

open Nat

variables {n : ℕ} (b c : ℤ)
def S (n : ℕ) := n^2 + b * n + c
def a (n : ℕ) := S n - S (n - 1)

theorem find_constants (a2a3_sum_eq_4 : a 2 + a 3 = 4) : 
  c = 0 ∧ b = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_constants_l2161_216127


namespace NUMINAMATH_GPT_increasing_on_interval_l2161_216144

open Real

noncomputable def f (x a b : ℝ) := abs (x^2 - 2*a*x + b)

theorem increasing_on_interval {a b : ℝ} (h : a^2 - b ≤ 0) :
  ∀ ⦃x1 x2⦄, a ≤ x1 → x1 ≤ x2 → f x1 a b ≤ f x2 a b := sorry

end NUMINAMATH_GPT_increasing_on_interval_l2161_216144


namespace NUMINAMATH_GPT_find_n_in_range_and_modulus_l2161_216149

theorem find_n_in_range_and_modulus :
  ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ (-200) % 21 = n % 21 → n = 10 := by
  sorry

end NUMINAMATH_GPT_find_n_in_range_and_modulus_l2161_216149


namespace NUMINAMATH_GPT_equivalent_proof_problem_l2161_216109

variable {x : ℝ}

theorem equivalent_proof_problem (h : x + 1/x = Real.sqrt 7) :
  x^12 - 5 * x^8 + 2 * x^6 = 1944 * Real.sqrt 7 * x - 2494 :=
sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l2161_216109


namespace NUMINAMATH_GPT_pure_imaginary_real_part_zero_l2161_216141

-- Define the condition that the complex number a + i is a pure imaginary number.
def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.I * b

-- Define the complex number a + i.
def z (a : ℝ) : ℂ := a + Complex.I

-- The theorem states that if z is pure imaginary, then a = 0.
theorem pure_imaginary_real_part_zero (a : ℝ) (h : isPureImaginary (z a)) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_real_part_zero_l2161_216141


namespace NUMINAMATH_GPT_sum_tens_ones_digit_of_7_pow_11_l2161_216172

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_tens_ones_digit_of_7_pow_11_l2161_216172
