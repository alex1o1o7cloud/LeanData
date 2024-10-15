import Mathlib

namespace NUMINAMATH_GPT_solution_interval_for_x_l581_58199

theorem solution_interval_for_x (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (48 / 7 ≤ x ∧ x < 49 / 7) :=
by sorry

end NUMINAMATH_GPT_solution_interval_for_x_l581_58199


namespace NUMINAMATH_GPT_woman_waits_time_until_man_catches_up_l581_58178

theorem woman_waits_time_until_man_catches_up
  (woman_speed : ℝ)
  (man_speed : ℝ)
  (wait_time : ℝ)
  (woman_slows_after : ℝ)
  (h_man_speed : man_speed = 5 / 60) -- man's speed in miles per minute
  (h_woman_speed : woman_speed = 25 / 60) -- woman's speed in miles per minute
  (h_wait_time : woman_slows_after = 5) -- the time in minutes after which the woman waits for man
  (h_woman_waits : wait_time = 25) : wait_time = (woman_slows_after * woman_speed) / man_speed :=
sorry

end NUMINAMATH_GPT_woman_waits_time_until_man_catches_up_l581_58178


namespace NUMINAMATH_GPT_distance_after_second_sign_l581_58162

-- Define the known conditions
def total_distance_ridden : ℕ := 1000
def distance_to_first_sign : ℕ := 350
def distance_between_signs : ℕ := 375

-- The distance Matt rode after passing the second sign
theorem distance_after_second_sign :
  total_distance_ridden - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end NUMINAMATH_GPT_distance_after_second_sign_l581_58162


namespace NUMINAMATH_GPT_negation_of_rectangular_parallelepipeds_have_12_edges_l581_58117

-- Define a structure for Rectangular Parallelepiped and the property of having edges
structure RectangularParallelepiped where
  hasEdges : ℕ → Prop

-- Problem statement
theorem negation_of_rectangular_parallelepipeds_have_12_edges :
  (∀ rect_p : RectangularParallelepiped, rect_p.hasEdges 12) →
  ∃ rect_p : RectangularParallelepiped, ¬ rect_p.hasEdges 12 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_rectangular_parallelepipeds_have_12_edges_l581_58117


namespace NUMINAMATH_GPT_Eugene_buys_four_t_shirts_l581_58169

noncomputable def t_shirt_price : ℝ := 20
noncomputable def pants_price : ℝ := 80
noncomputable def shoes_price : ℝ := 150
noncomputable def discount : ℝ := 0.10

noncomputable def discounted_t_shirt_price : ℝ := t_shirt_price - (t_shirt_price * discount)
noncomputable def discounted_pants_price : ℝ := pants_price - (pants_price * discount)
noncomputable def discounted_shoes_price : ℝ := shoes_price - (shoes_price * discount)

noncomputable def num_pants : ℝ := 3
noncomputable def num_shoes : ℝ := 2
noncomputable def total_paid : ℝ := 558

noncomputable def total_cost_of_pants_and_shoes : ℝ := (num_pants * discounted_pants_price) + (num_shoes * discounted_shoes_price)
noncomputable def remaining_cost_for_t_shirts : ℝ := total_paid - total_cost_of_pants_and_shoes

noncomputable def num_t_shirts : ℝ := remaining_cost_for_t_shirts / discounted_t_shirt_price

theorem Eugene_buys_four_t_shirts : num_t_shirts = 4 := by
  sorry

end NUMINAMATH_GPT_Eugene_buys_four_t_shirts_l581_58169


namespace NUMINAMATH_GPT_population_growth_rate_l581_58181

-- Define initial and final population
def initial_population : ℕ := 240
def final_population : ℕ := 264

-- Define the formula for calculating population increase rate
def population_increase_rate (P_i P_f : ℕ) : ℕ :=
  ((P_f - P_i) * 100) / P_i

-- State the theorem
theorem population_growth_rate :
  population_increase_rate initial_population final_population = 10 := by
  sorry

end NUMINAMATH_GPT_population_growth_rate_l581_58181


namespace NUMINAMATH_GPT_cryptarithm_base_solution_l581_58130

theorem cryptarithm_base_solution :
  ∃ (K I T : ℕ) (d : ℕ), 
    O = 0 ∧
    2 * T = I ∧
    T + 1 = K ∧
    K + I = d ∧ 
    d = 7 ∧ 
    K ≠ I ∧ K ≠ T ∧ K ≠ O ∧
    I ≠ T ∧ I ≠ O ∧
    T ≠ O :=
sorry

end NUMINAMATH_GPT_cryptarithm_base_solution_l581_58130


namespace NUMINAMATH_GPT_circles_disjoint_l581_58165

theorem circles_disjoint (a : ℝ) : ((x - 1)^2 + (y - 1)^2 = 4) ∧ (x^2 + (y - a)^2 = 1) → (a < 1 - 2 * Real.sqrt 2 ∨ a > 1 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_circles_disjoint_l581_58165


namespace NUMINAMATH_GPT_lilies_per_centerpiece_correct_l581_58151

-- Definitions based on the conditions
def num_centerpieces : ℕ := 6
def roses_per_centerpiece : ℕ := 8
def cost_per_flower : ℕ := 15
def total_budget : ℕ := 2700

-- Definition of the number of orchids per centerpiece using given condition
def orchids_per_centerpiece : ℕ := 2 * roses_per_centerpiece

-- Definition of the total cost for roses and orchids before calculating lilies
def total_rose_cost : ℕ := num_centerpieces * roses_per_centerpiece * cost_per_flower
def total_orchid_cost : ℕ := num_centerpieces * orchids_per_centerpiece * cost_per_flower
def total_rose_and_orchid_cost : ℕ := total_rose_cost + total_orchid_cost

-- Definition for the remaining budget for lilies
def remaining_budget_for_lilies : ℕ := total_budget - total_rose_and_orchid_cost

-- Number of lilies in total and per centerpiece
def total_lilies : ℕ := remaining_budget_for_lilies / cost_per_flower
def lilies_per_centerpiece : ℕ := total_lilies / num_centerpieces

-- The proof statement we want to assert
theorem lilies_per_centerpiece_correct : lilies_per_centerpiece = 6 :=
by
  sorry

end NUMINAMATH_GPT_lilies_per_centerpiece_correct_l581_58151


namespace NUMINAMATH_GPT_proof_problem_l581_58198

theorem proof_problem (x : ℝ) : (0 < x ∧ x < 5) → (x^2 - 5 * x < 0) ∧ (|x - 2| < 3) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l581_58198


namespace NUMINAMATH_GPT_solution_set_x_plus_3_f_x_plus_4_l581_58143

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom even_f_x_plus_1 : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom deriv_negative_f : ∀ x : ℝ, x > 1 → f' x < 0
axiom f_at_4_equals_zero : f 4 = 0

-- To prove
theorem solution_set_x_plus_3_f_x_plus_4 :
  {x : ℝ | (x + 3) * f (x + 4) < 0} = {x : ℝ | -6 < x ∧ x < -3} ∪ {x : ℝ | x > 0} := sorry

end NUMINAMATH_GPT_solution_set_x_plus_3_f_x_plus_4_l581_58143


namespace NUMINAMATH_GPT_solution_l581_58104

def solve_for_x (x : ℝ) : Prop :=
  7 + 3.5 * x = 2.1 * x - 25

theorem solution (x : ℝ) (h : solve_for_x x) : x = -22.857 :=
by
  sorry

end NUMINAMATH_GPT_solution_l581_58104


namespace NUMINAMATH_GPT_max_num_pieces_l581_58166

-- Definition of areas
def largeCake_area : ℕ := 21 * 21
def smallPiece_area : ℕ := 3 * 3

-- Problem Statement
theorem max_num_pieces : largeCake_area / smallPiece_area = 49 := by
  sorry

end NUMINAMATH_GPT_max_num_pieces_l581_58166


namespace NUMINAMATH_GPT_market_value_of_stock_l581_58164

variable (face_value : ℝ) (annual_dividend yield : ℝ)

-- Given conditions:
def stock_four_percent := annual_dividend = 0.04 * face_value
def stock_yield_five_percent := yield = 0.05

-- Problem statement:
theorem market_value_of_stock (face_value := 100) (annual_dividend := 4) (yield := 0.05) 
  (h1 : stock_four_percent face_value annual_dividend) 
  (h2 : stock_yield_five_percent yield) : 
  (4 / 0.05) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_market_value_of_stock_l581_58164


namespace NUMINAMATH_GPT_martha_initial_marbles_l581_58189

-- Definition of the conditions
def initial_marbles_dilan : ℕ := 14
def initial_marbles_phillip : ℕ := 19
def initial_marbles_veronica : ℕ := 7
def marbles_after_redistribution_each : ℕ := 15
def number_of_people : ℕ := 4

-- Total marbles after redistribution
def total_marbles_after_redistribution : ℕ := marbles_after_redistribution_each * number_of_people

-- Total initial marbles of Dilan, Phillip, and Veronica
def total_initial_marbles_dilan_phillip_veronica : ℕ := initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica

-- Prove the number of marbles Martha initially had
theorem martha_initial_marbles : initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica + x = number_of_people * marbles_after_redistribution →
  x = 20 := by
  sorry

end NUMINAMATH_GPT_martha_initial_marbles_l581_58189


namespace NUMINAMATH_GPT_sweet_treats_distribution_l581_58122

-- Define the number of cookies, cupcakes, brownies, and students
def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

-- Define the total number of sweet treats
def total_sweet_treats : ℕ := cookies + cupcakes + brownies

-- Define the number of sweet treats each student will receive
def sweet_treats_per_student : ℕ := total_sweet_treats / students

-- Prove that each student will receive 4 sweet treats
theorem sweet_treats_distribution : sweet_treats_per_student = 4 := 
by sorry

end NUMINAMATH_GPT_sweet_treats_distribution_l581_58122


namespace NUMINAMATH_GPT_find_x_minus_y_l581_58120

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l581_58120


namespace NUMINAMATH_GPT_number_of_dolls_l581_58141

theorem number_of_dolls (total_toys : ℕ) (fraction_action_figures : ℚ) 
  (remaining_fraction_action_figures : fraction_action_figures = 1 / 4) 
  (remaining_fraction_dolls : 1 - fraction_action_figures = 3 / 4) 
  (total_toys_eq : total_toys = 24) : 
  (total_toys - total_toys * fraction_action_figures) = 18 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_dolls_l581_58141


namespace NUMINAMATH_GPT_hyperbola_sufficient_but_not_necessary_l581_58119

theorem hyperbola_sufficient_but_not_necessary :
  (∀ (C : Type) (x y : ℝ), C = {p : ℝ × ℝ | ((p.1)^2 / 16) - ((p.2)^2 / 9) = 1} →
  (∀ x, y = 3 * (x / 4) ∨ y = -3 * (x / 4)) →
  ∃ (C' : Type) (x' y' : ℝ), C' = {p : ℝ × ℝ | ((p.1)^2 / 64) - ((p.2)^2 / 36) = 1} ∧
  (∀ x', y' = 3 * (x' / 4) ∨ y' = -3 * (x' / 4))) :=
sorry

end NUMINAMATH_GPT_hyperbola_sufficient_but_not_necessary_l581_58119


namespace NUMINAMATH_GPT_factorial_quotient_l581_58150

/-- Prove that the quotient of the factorial of 4! divided by 4! simplifies to 23!. -/
theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := 
by
  sorry

end NUMINAMATH_GPT_factorial_quotient_l581_58150


namespace NUMINAMATH_GPT_diagonals_in_polygon_with_150_sides_l581_58180

-- (a) Definitions for conditions
def sides : ℕ := 150

def diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- (c) Statement of the problem in Lean 4
theorem diagonals_in_polygon_with_150_sides :
  diagonals sides = 11025 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_in_polygon_with_150_sides_l581_58180


namespace NUMINAMATH_GPT_height_of_second_triangle_l581_58134

theorem height_of_second_triangle
  (base1 : ℝ) (height1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : (base2 * height2) / 2 = 2 * (base1 * height1) / 2) :
  height2 = 18 :=
sorry

end NUMINAMATH_GPT_height_of_second_triangle_l581_58134


namespace NUMINAMATH_GPT_find_x_from_average_l581_58160

theorem find_x_from_average :
  let sum_series := 5151
  let n := 102
  let known_average := 50 * (x + 1)
  (sum_series + x) / n = known_average → 
  x = 51 / 5099 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_x_from_average_l581_58160


namespace NUMINAMATH_GPT_inequality_sqrt_sum_ge_one_l581_58153

variable (a b c : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variable (prod_abc : a * b * c = 1)

theorem inequality_sqrt_sum_ge_one :
  (Real.sqrt (a / (8 + a)) + Real.sqrt (b / (8 + b)) + Real.sqrt (c / (8 + c)) ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_sum_ge_one_l581_58153


namespace NUMINAMATH_GPT_find_a_l581_58115

theorem find_a (a : ℝ) :
  (∀ x y, x + y = a → x^2 + y^2 = 4) →
  (∀ A B : ℝ × ℝ, (A.1 + A.2 = a ∧ B.1 + B.2 = a ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4) →
      ‖(A.1, A.2) + (B.1, B.2)‖ = ‖(A.1, A.2) - (B.1, B.2)‖) →
  a = 2 ∨ a = -2 :=
by
  intros line_circle_intersect vector_eq_magnitude
  sorry

end NUMINAMATH_GPT_find_a_l581_58115


namespace NUMINAMATH_GPT_sum_of_digits_N_l581_58197

-- Define a function to compute the least common multiple (LCM) of a list of numbers
def lcm_list (xs : List ℕ) : ℕ :=
  xs.foldr Nat.lcm 1

-- The set of numbers less than 8
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- The LCM of numbers less than 8
def N_lcm : ℕ := lcm_list nums

-- The second smallest positive integer that is divisible by every positive integer less than 8
def N : ℕ := 2 * N_lcm

-- Function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Prove that the sum of the digits of N is 12
theorem sum_of_digits_N : sum_of_digits N = 12 :=
by
  -- Necessary proof steps will be filled here
  sorry

end NUMINAMATH_GPT_sum_of_digits_N_l581_58197


namespace NUMINAMATH_GPT_max_rational_sums_is_1250_l581_58136

/-- We define a structure to represent the problem's conditions. -/
structure GridConfiguration where
  grid_rows : Nat
  grid_cols : Nat
  total_numbers : Nat
  rational_count : Nat
  irrational_count : Nat
  (h_grid : grid_rows = 50)
  (h_grid_col : grid_cols = 50)
  (h_total_numbers : total_numbers = 100)
  (h_rational_count : rational_count = 50)
  (h_irrational_count : irrational_count = 50)

/-- We define a function to calculate the number of rational sums in the grid. -/
def max_rational_sums (config : GridConfiguration) : Nat :=
  let x := config.rational_count / 2 -- rational numbers to the left
  let ni := 2 * x * x - 100 * x + 2500
  let rational_sums := 2500 - ni
  rational_sums

/-- The theorem stating the maximum number of rational sums is 1250. -/
theorem max_rational_sums_is_1250 (config : GridConfiguration) : max_rational_sums config = 1250 :=
  sorry

end NUMINAMATH_GPT_max_rational_sums_is_1250_l581_58136


namespace NUMINAMATH_GPT_total_percent_decrease_is_19_l581_58152

noncomputable def original_value : ℝ := 100
noncomputable def first_year_decrease : ℝ := 0.10
noncomputable def second_year_decrease : ℝ := 0.10
noncomputable def value_after_first_year : ℝ := original_value * (1 - first_year_decrease)
noncomputable def value_after_second_year : ℝ := value_after_first_year * (1 - second_year_decrease)
noncomputable def total_decrease_in_dollars : ℝ := original_value - value_after_second_year
noncomputable def total_percent_decrease : ℝ := (total_decrease_in_dollars / original_value) * 100

theorem total_percent_decrease_is_19 :
  total_percent_decrease = 19 := by
  sorry

end NUMINAMATH_GPT_total_percent_decrease_is_19_l581_58152


namespace NUMINAMATH_GPT_rectangle_area_l581_58187

theorem rectangle_area :
  ∃ (x y : ℝ), (x + 3.5) * (y - 1.5) = x * y ∧
               (x - 3.5) * (y + 2.5) = x * y ∧
               2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y ∧
               x * y = 196 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l581_58187


namespace NUMINAMATH_GPT_households_3_houses_proportion_l581_58175

noncomputable def total_households : ℕ := 100000
noncomputable def ordinary_households : ℕ := 99000
noncomputable def high_income_households : ℕ := 1000

noncomputable def sampled_ordinary_households : ℕ := 990
noncomputable def sampled_high_income_households : ℕ := 100

noncomputable def sampled_ordinary_3_houses : ℕ := 40
noncomputable def sampled_high_income_3_houses : ℕ := 80

noncomputable def proportion_3_houses : ℝ := (sampled_ordinary_3_houses / sampled_ordinary_households * ordinary_households + sampled_high_income_3_houses / sampled_high_income_households * high_income_households) / total_households

theorem households_3_houses_proportion : proportion_3_houses = 0.048 := 
by
  sorry

end NUMINAMATH_GPT_households_3_houses_proportion_l581_58175


namespace NUMINAMATH_GPT_solve_for_square_solve_for_cube_l581_58188

variable (x : ℂ)

-- Given condition
def condition := x + 1/x = 8

-- Prove that x^2 + 1/x^2 = 62 given the condition
theorem solve_for_square (h : condition x) : x^2 + 1/x^2 = 62 := 
  sorry

-- Prove that x^3 + 1/x^3 = 488 given the condition
theorem solve_for_cube (h : condition x) : x^3 + 1/x^3 = 488 :=
  sorry

end NUMINAMATH_GPT_solve_for_square_solve_for_cube_l581_58188


namespace NUMINAMATH_GPT_inequality_solution_l581_58173

theorem inequality_solution (x y : ℝ) : 
  (x^2 - 4 * x * y + 4 * x^2 < x^2) ↔ (x < y ∧ y < 3 * x ∧ x > 0) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l581_58173


namespace NUMINAMATH_GPT_print_pages_l581_58185

theorem print_pages (pages_per_cost : ℕ) (cost_cents : ℕ) (dollars : ℕ)
                    (h1 : pages_per_cost = 7) (h2 : cost_cents = 9) (h3 : dollars = 50) :
  (dollars * 100 * pages_per_cost) / cost_cents = 3888 :=
by
  sorry

end NUMINAMATH_GPT_print_pages_l581_58185


namespace NUMINAMATH_GPT_max_n_m_sum_l581_58137

-- Definition of the function f
def f (x : ℝ) : ℝ := -x^2 + 4 * x

-- Statement of the problem
theorem max_n_m_sum {m n : ℝ} (h : n > m) (h_range : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4) : n + m = 7 :=
sorry

end NUMINAMATH_GPT_max_n_m_sum_l581_58137


namespace NUMINAMATH_GPT_number_of_paths_from_A_to_D_l581_58126

-- Definitions based on conditions
def paths_A_to_B : ℕ := 2
def paths_B_to_C : ℕ := 2
def paths_A_to_C : ℕ := 1
def paths_C_to_D : ℕ := 2
def paths_B_to_D : ℕ := 2

-- Theorem statement
theorem number_of_paths_from_A_to_D : 
  paths_A_to_B * paths_B_to_C * paths_C_to_D + 
  paths_A_to_C * paths_C_to_D + 
  paths_A_to_B * paths_B_to_D = 14 :=
by {
  -- proof steps will go here
  sorry
}

end NUMINAMATH_GPT_number_of_paths_from_A_to_D_l581_58126


namespace NUMINAMATH_GPT_largest_four_digit_number_property_l581_58193

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end NUMINAMATH_GPT_largest_four_digit_number_property_l581_58193


namespace NUMINAMATH_GPT_quadratic_form_and_sum_l581_58100

theorem quadratic_form_and_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
  (15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := 
sorry

end NUMINAMATH_GPT_quadratic_form_and_sum_l581_58100


namespace NUMINAMATH_GPT_specimen_exchange_l581_58148

theorem specimen_exchange (x : ℕ) (h : x * (x - 1) = 110) : x * (x - 1) = 110 := by
  exact h

end NUMINAMATH_GPT_specimen_exchange_l581_58148


namespace NUMINAMATH_GPT_max_area_of_triangle_ABC_l581_58171

-- Definitions for the problem conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (5, 4)
def parabola (x : ℝ) : ℝ := x^2 - 3 * x
def C (r : ℝ) : ℝ × ℝ := (r, parabola r)

-- Function to compute the Shoelace Theorem area of ABC
def shoelace_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Proof statement
theorem max_area_of_triangle_ABC : ∃ (r : ℝ), -2 ≤ r ∧ r ≤ 5 ∧ shoelace_area A B (C r) = 39 := 
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_ABC_l581_58171


namespace NUMINAMATH_GPT_probability_of_rolling_2_4_6_on_8_sided_die_l581_58131

theorem probability_of_rolling_2_4_6_on_8_sided_die : 
  ∀ (ω : Fin 8), 
  (1 / 8) * (ite (ω = 1 ∨ ω = 3 ∨ ω = 5) 1 0) = 3 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_rolling_2_4_6_on_8_sided_die_l581_58131


namespace NUMINAMATH_GPT_charged_amount_is_35_l581_58186

-- Definitions based on conditions
def annual_interest_rate : ℝ := 0.05
def owed_amount : ℝ := 36.75
def time_in_years : ℝ := 1

-- The amount charged on the account in January
def charged_amount (P : ℝ) : Prop :=
  owed_amount = P + (P * annual_interest_rate * time_in_years)

-- The proof statement
theorem charged_amount_is_35 : charged_amount 35 := by
  sorry

end NUMINAMATH_GPT_charged_amount_is_35_l581_58186


namespace NUMINAMATH_GPT_right_triangle_inradius_l581_58161

theorem right_triangle_inradius (a b c : ℕ) (h : a = 6) (h2 : b = 8) (h3 : c = 10) :
  ((a^2 + b^2 = c^2) ∧ (1/2 * ↑a * ↑b = 24) ∧ ((a + b + c) / 2 = 12) ∧ (24 = 12 * 2)) :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_inradius_l581_58161


namespace NUMINAMATH_GPT_range_of_a_l581_58154

def f (x a : ℝ) := |x - 2| + |x + a|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 3) → a ≤ -5 ∨ a ≥ 1 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l581_58154


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l581_58139

-- Defining the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Condition for vectors a and b to be parallel
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value : ∃ x, are_parallel a (b x) ∧ x = 6 := by
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l581_58139


namespace NUMINAMATH_GPT_original_amount_charged_l581_58108

variables (P : ℝ) (interest_rate : ℝ) (total_owed : ℝ)

theorem original_amount_charged :
  interest_rate = 0.09 →
  total_owed = 38.15 →
  (P + P * interest_rate = total_owed) →
  P = 35 :=
by
  intros h_interest_rate h_total_owed h_equation
  sorry

end NUMINAMATH_GPT_original_amount_charged_l581_58108


namespace NUMINAMATH_GPT_perimeter_large_star_l581_58125

theorem perimeter_large_star (n m : ℕ) (P : ℕ)
  (triangle_perimeter : ℕ) (quad_perimeter : ℕ) (small_star_perimeter : ℕ)
  (hn : n = 5) (hm : m = 5)
  (h_triangle_perimeter : triangle_perimeter = 7)
  (h_quad_perimeter : quad_perimeter = 18)
  (h_small_star_perimeter : small_star_perimeter = 3) :
  m * quad_perimeter + small_star_perimeter = n * triangle_perimeter + P → P = 58 :=
by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_perimeter_large_star_l581_58125


namespace NUMINAMATH_GPT_find_larger_number_l581_58192

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 8 * S + 15) : L = 1557 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l581_58192


namespace NUMINAMATH_GPT_total_amount_l581_58103

theorem total_amount (x_share : ℝ) (y_share : ℝ) (w_share : ℝ) (hx : x_share = 0.30) (hy : y_share = 0.20) (hw : w_share = 10) :
  (w_share * (1 + x_share + y_share)) = 15 := by
  sorry

end NUMINAMATH_GPT_total_amount_l581_58103


namespace NUMINAMATH_GPT_ratio_of_age_differences_l581_58190

variable (R J K : ℕ)

-- conditions
axiom h1 : R = J + 6
axiom h2 : R + 2 = 2 * (J + 2)
axiom h3 : (R + 2) * (K + 2) = 108

-- statement to prove
theorem ratio_of_age_differences : (R - J) = 2 * (R - K) := 
sorry

end NUMINAMATH_GPT_ratio_of_age_differences_l581_58190


namespace NUMINAMATH_GPT_question1_geometric_sequence_question2_minimum_term_l581_58167

theorem question1_geometric_sequence (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  q = 0 →
  (a 1 = 1 / 2) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * (r ^ n)) →
  (p = 0 ∨ p = 1) :=
by sorry

theorem question2_minimum_term (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  p = 1 →
  (a 1 = 1 / 2) →
  (a 4 = min (min (a 1) (a 2)) (a 3)) →
  3 ≤ q ∧ q ≤ 27 / 4 :=
by sorry

end NUMINAMATH_GPT_question1_geometric_sequence_question2_minimum_term_l581_58167


namespace NUMINAMATH_GPT_final_selling_price_l581_58109

def actual_price : ℝ := 9356.725146198829
def price_after_first_discount (P : ℝ) : ℝ := P * 0.80
def price_after_second_discount (P1 : ℝ) : ℝ := P1 * 0.90
def price_after_third_discount (P2 : ℝ) : ℝ := P2 * 0.95

theorem final_selling_price :
  (price_after_third_discount (price_after_second_discount (price_after_first_discount actual_price))) = 6400 :=
by 
  -- Here we would need to provide the proof, but it is skipped with sorry
  sorry

end NUMINAMATH_GPT_final_selling_price_l581_58109


namespace NUMINAMATH_GPT_find_tangent_line_l581_58111

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let d := abs c / (Real.sqrt (a^2 + b^2))
  d = 1

def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem find_tangent_line :
  ∀ (k b : ℝ),
    is_perpendicular k 1 →
    is_tangent_to_circle 1 1 b →
    ∃ (x y : ℝ), in_first_quadrant x y ∧ x + y - b = 0 →
    b = Real.sqrt 2 := sorry

end NUMINAMATH_GPT_find_tangent_line_l581_58111


namespace NUMINAMATH_GPT_evaluate_x_squared_plus_y_squared_l581_58177

theorem evaluate_x_squared_plus_y_squared (x y : ℝ) (h₁ : 3 * x + y = 20) (h₂ : 4 * x + y = 25) :
  x^2 + y^2 = 50 :=
sorry

end NUMINAMATH_GPT_evaluate_x_squared_plus_y_squared_l581_58177


namespace NUMINAMATH_GPT_train_passes_man_in_12_seconds_l581_58113

noncomputable def time_to_pass_man (train_length: ℝ) (train_speed_kmph: ℝ) (man_speed_kmph: ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18)
  train_length / relative_speed_mps

theorem train_passes_man_in_12_seconds :
  time_to_pass_man 220 60 6 = 12 := by
 sorry

end NUMINAMATH_GPT_train_passes_man_in_12_seconds_l581_58113


namespace NUMINAMATH_GPT_first_class_seat_count_l581_58123

theorem first_class_seat_count :
  let seats_first_class := 10
  let seats_business_class := 30
  let seats_economy_class := 50
  let people_economy_class := seats_economy_class / 2
  let people_business_and_first := people_economy_class
  let unoccupied_business := 8
  let people_business_class := seats_business_class - unoccupied_business
  people_business_and_first - people_business_class = 3 := by
  sorry

end NUMINAMATH_GPT_first_class_seat_count_l581_58123


namespace NUMINAMATH_GPT_calculate_expression_l581_58174

theorem calculate_expression : (7^2 - 5^2)^3 = 13824 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l581_58174


namespace NUMINAMATH_GPT_total_mass_of_individuals_l581_58118

def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def initial_sink_depth : Float := 0.018
def density_of_water : Float := 1000.0
def mass_of_second_person : Float := 75.0

theorem total_mass_of_individuals :
  let V1 := boat_length * boat_breadth * initial_sink_depth
  let m1 := V1 * density_of_water
  let total_mass := m1 + mass_of_second_person
  total_mass = 183 :=
by
  sorry

end NUMINAMATH_GPT_total_mass_of_individuals_l581_58118


namespace NUMINAMATH_GPT_fgf_of_3_l581_58149

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 4 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

-- The statement we need to prove
theorem fgf_of_3 : f (g (f 3)) = 332 := by
  sorry

end NUMINAMATH_GPT_fgf_of_3_l581_58149


namespace NUMINAMATH_GPT_soda_price_increase_l581_58102

theorem soda_price_increase (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_soda_price_increase_l581_58102


namespace NUMINAMATH_GPT_sqrt_x_minus_2_range_l581_58121

theorem sqrt_x_minus_2_range (x : ℝ) : (↑0 ≤ (x - 2)) ↔ (x ≥ 2) := sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_range_l581_58121


namespace NUMINAMATH_GPT_geraldo_drank_7_pints_l581_58114

-- Conditions
def total_gallons : ℝ := 20
def num_containers : ℕ := 80
def gallons_to_pints : ℝ := 8
def containers_drank : ℝ := 3.5

-- Problem statement
theorem geraldo_drank_7_pints :
  let total_pints : ℝ := total_gallons * gallons_to_pints
  let pints_per_container : ℝ := total_pints / num_containers
  let pints_drank : ℝ := containers_drank * pints_per_container
  pints_drank = 7 :=
by
  sorry

end NUMINAMATH_GPT_geraldo_drank_7_pints_l581_58114


namespace NUMINAMATH_GPT_part_I_part_II_l581_58176

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

theorem part_I (m : ℝ) (h : ∀ x : ℝ, f x ≠ m) : m < -2 ∨ m > 2 :=
sorry

theorem part_II (P : ℝ × ℝ) (hP : P = (2, -6)) :
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b = 0 ∧ m = 3 ∧ b = 0) ∨ 
                 (m * x + b = 24 * x - 54 ∧ P.2 = 24 * P.1 - 54)) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l581_58176


namespace NUMINAMATH_GPT_triangle_tan_A_and_area_l581_58132

theorem triangle_tan_A_and_area {A B C a b c : ℝ} (hB : B = Real.pi / 3)
  (h1 : (Real.cos A - 3 * Real.cos C) * b = (3 * c - a) * Real.cos B)
  (hb : b = Real.sqrt 14) : 
  ∃ tan_A : ℝ, tan_A = Real.sqrt 3 / 5 ∧  -- First part: the value of tan A
  ∃ S : ℝ, S = (3 * Real.sqrt 3) / 2 :=  -- Second part: the area of triangle ABC
by
  sorry

end NUMINAMATH_GPT_triangle_tan_A_and_area_l581_58132


namespace NUMINAMATH_GPT_twin_primes_solution_l581_58112

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p = q + 2 ∨ q = p + 2)

theorem twin_primes_solution (p q : ℕ) :
  are_twin_primes p q ∧ is_prime (p^2 - p * q + q^2) ↔ (p, q) = (5, 3) ∨ (p, q) = (3, 5) := by
  sorry

end NUMINAMATH_GPT_twin_primes_solution_l581_58112


namespace NUMINAMATH_GPT_find_G_14_l581_58183

noncomputable def G (x : ℝ) : ℝ := sorry

lemma G_at_7 : G 7 = 20 := sorry

lemma functional_equation (x : ℝ) (hx: x ^ 2 + 8 * x + 16 ≠ 0) : 
  G (4 * x) / G (x + 4) = 16 - (96 * x + 128) / (x^2 + 8 * x + 16) := sorry

theorem find_G_14 : G 14 = 96 := sorry

end NUMINAMATH_GPT_find_G_14_l581_58183


namespace NUMINAMATH_GPT_weaving_increase_l581_58138

theorem weaving_increase (a₁ : ℕ) (S₃₀ : ℕ) (d : ℚ) (hₐ₁ : a₁ = 5) (hₛ₃₀ : S₃₀ = 390)
  (h_sum : S₃₀ = 30 * (a₁ + (a₁ + 29 * d)) / 2) : d = 16 / 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_weaving_increase_l581_58138


namespace NUMINAMATH_GPT_range_of_a_l581_58101

theorem range_of_a (x : ℝ) (a : ℝ) (h1 : 2 < x) (h2 : a ≤ x + 1 / (x - 2)) : a ≤ 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l581_58101


namespace NUMINAMATH_GPT_range_of_k_l581_58128

noncomputable def operation (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

theorem range_of_k (k : ℝ) (h : operation 1 (k^2) < 3) : -1 < k ∧ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l581_58128


namespace NUMINAMATH_GPT_negation_of_proposition_l581_58146

theorem negation_of_proposition (a b : ℝ) : 
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l581_58146


namespace NUMINAMATH_GPT_david_initial_money_l581_58196

-- Given conditions as definitions
def spent (S : ℝ) : Prop := S - 800 = 500
def has_left (H : ℝ) : Prop := H = 500

-- The main theorem to prove
theorem david_initial_money (S : ℝ) (X : ℝ) (H : ℝ)
  (h1 : spent S) 
  (h2 : has_left H) 
  : X = S + H → X = 1800 :=
by
  sorry

end NUMINAMATH_GPT_david_initial_money_l581_58196


namespace NUMINAMATH_GPT_nonempty_solution_set_iff_a_gt_2_l581_58155

theorem nonempty_solution_set_iff_a_gt_2 (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) ↔ a > 2 :=
sorry

end NUMINAMATH_GPT_nonempty_solution_set_iff_a_gt_2_l581_58155


namespace NUMINAMATH_GPT_pupils_in_program_l581_58116

theorem pupils_in_program {total_people parents : ℕ} (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end NUMINAMATH_GPT_pupils_in_program_l581_58116


namespace NUMINAMATH_GPT_jill_marathon_time_l581_58145

def jack_marathon_distance : ℝ := 42
def jack_marathon_time : ℝ := 6
def speed_ratio : ℝ := 0.7

theorem jill_marathon_time :
  ∃ t_jill : ℝ, (t_jill = jack_marathon_distance / (jack_marathon_distance / jack_marathon_time / speed_ratio)) ∧
  t_jill = 4.2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_jill_marathon_time_l581_58145


namespace NUMINAMATH_GPT_max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l581_58168

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f_at_0 :
  f 0 = 5 := by
  sorry

theorem min_value_of_f_on_neg_inf_to_0 :
  f (-3) = -Real.exp 3 := by
  sorry

theorem range_of_a_for_ineq :
  ∀ x : ℝ, x^2 + 5*x + 5 - a * Real.exp x ≥ 0 ↔ a ≤ -Real.exp 3 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l581_58168


namespace NUMINAMATH_GPT_number_of_paths_K_to_L_l581_58157

-- Definition of the problem structure
def K : Type := Unit
def A : Type := Unit
def R : Type := Unit
def L : Type := Unit

-- Defining the number of paths between each stage
def paths_from_K_to_A := 2
def paths_from_A_to_R := 4
def paths_from_R_to_L := 8

-- The main theorem stating the number of paths from K to L
theorem number_of_paths_K_to_L : paths_from_K_to_A * 2 * 2 = 8 := by 
  sorry

end NUMINAMATH_GPT_number_of_paths_K_to_L_l581_58157


namespace NUMINAMATH_GPT_am_gm_inequality_l581_58194

theorem am_gm_inequality {a b : ℝ} (n : ℕ) (h₁ : n ≠ 1) (h₂ : a > b) (h₃ : b > 0) : 
  ( (a + b) / 2 )^n < (a^n + b^n) / 2 := 
sorry

end NUMINAMATH_GPT_am_gm_inequality_l581_58194


namespace NUMINAMATH_GPT_log_product_zero_l581_58124

theorem log_product_zero :
  (Real.log 3 / Real.log 2 + Real.log 27 / Real.log 2) *
  (Real.log 4 / Real.log 4 + Real.log (1 / 4) / Real.log 4) = 0 := by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_log_product_zero_l581_58124


namespace NUMINAMATH_GPT_sin_sum_cos_product_l581_58105

theorem sin_sum_cos_product (A B C : Real) (h : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_sum_cos_product_l581_58105


namespace NUMINAMATH_GPT_greatest_number_divides_with_remainders_l581_58129

theorem greatest_number_divides_with_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_divides_with_remainders_l581_58129


namespace NUMINAMATH_GPT_weight_of_mixture_l581_58133

variable (A B : ℝ)
variable (ratio_A_B : A / B = 9 / 11)
variable (consumed_A : A = 26.1)

theorem weight_of_mixture (A B : ℝ) (ratio_A_B : A / B = 9 / 11) (consumed_A : A = 26.1) : 
  A + B = 58 :=
sorry

end NUMINAMATH_GPT_weight_of_mixture_l581_58133


namespace NUMINAMATH_GPT_angle_AMC_is_70_l581_58107

theorem angle_AMC_is_70 (A B C M : Type) (angle_MBA angle_MAB angle_ACB : ℝ) (AC BC : ℝ) :
  AC = BC → 
  angle_MBA = 30 → 
  angle_MAB = 10 → 
  angle_ACB = 80 → 
  ∃ angle_AMC : ℝ, angle_AMC = 70 :=
by
  sorry

end NUMINAMATH_GPT_angle_AMC_is_70_l581_58107


namespace NUMINAMATH_GPT_circle_center_is_neg4_2_l581_58163

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 4 * y = 16

theorem circle_center_is_neg4_2 :
  ∃ (h k : ℝ), (h = -4 ∧ k = 2) ∧
  ∀ (x y : ℝ), circle_center x y ↔ (x + 4)^2 + (y - 2)^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_is_neg4_2_l581_58163


namespace NUMINAMATH_GPT_haley_trees_initially_grew_l581_58144

-- Given conditions
def num_trees_died : ℕ := 2
def num_trees_survived : ℕ := num_trees_died + 7

-- Prove the total number of trees initially grown
theorem haley_trees_initially_grew : num_trees_died + num_trees_survived = 11 :=
by
  -- here we would provide the proof eventually
  sorry

end NUMINAMATH_GPT_haley_trees_initially_grew_l581_58144


namespace NUMINAMATH_GPT_circle_radius_inscribed_l581_58127

noncomputable def a : ℝ := 6
noncomputable def b : ℝ := 12
noncomputable def c : ℝ := 18

noncomputable def r : ℝ :=
  let term1 := 1/a
  let term2 := 1/b
  let term3 := 1/c
  let sqrt_term := Real.sqrt ((1/(a * b)) + (1/(a * c)) + (1/(b * c)))
  1 / ((term1 + term2 + term3) + 2 * sqrt_term)

theorem circle_radius_inscribed :
  r = 36 / 17 := 
by
  sorry

end NUMINAMATH_GPT_circle_radius_inscribed_l581_58127


namespace NUMINAMATH_GPT_smallest_number_starting_with_five_l581_58156

theorem smallest_number_starting_with_five :
  ∃ n : ℕ, ∃ m : ℕ, m = (5 * m + 5) / 4 ∧ 5 * n + m = 512820 ∧ m < 10^6 := sorry

end NUMINAMATH_GPT_smallest_number_starting_with_five_l581_58156


namespace NUMINAMATH_GPT_range_of_a_l581_58135

variable (a x : ℝ)

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def M (a : ℝ) : Set ℝ := if a = 2 then {2} else {x | 2 ≤ x ∧ x ≤ a}

theorem range_of_a (a : ℝ) (p : x ∈ M a) (h : a ≥ 2) (hpq : Set.Subset (M a) A) : 2 ≤ a ∧ a ≤ 4 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l581_58135


namespace NUMINAMATH_GPT_children_tickets_count_l581_58195

theorem children_tickets_count (A C : ℕ) (h1 : 8 * A + 5 * C = 201) (h2 : A + C = 33) : C = 21 :=
by
  sorry

end NUMINAMATH_GPT_children_tickets_count_l581_58195


namespace NUMINAMATH_GPT_area_of_circle_l581_58159

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end NUMINAMATH_GPT_area_of_circle_l581_58159


namespace NUMINAMATH_GPT_point_coordinates_l581_58158

def point_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0 

theorem point_coordinates (m : ℝ) 
  (h1 : point_in_second_quadrant (-m-1) (2*m+1))
  (h2 : |2*m + 1| = 5) : (-m-1, 2*m+1) = (-3, 5) :=
sorry

end NUMINAMATH_GPT_point_coordinates_l581_58158


namespace NUMINAMATH_GPT_max_value_problem1_l581_58182

theorem max_value_problem1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ t, t = (1 / 2) * x * (1 - 2 * x) ∧ t ≤ 1 / 16 := sorry

end NUMINAMATH_GPT_max_value_problem1_l581_58182


namespace NUMINAMATH_GPT_hyperbola_focus_distance_l581_58142
open Real

theorem hyperbola_focus_distance
  (a b : ℝ)
  (ha : a = 5)
  (hb : b = 3)
  (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (∃ M : ℝ × ℝ, M = (x, y)))
  (M : ℝ × ℝ)
  (hM_on_hyperbola : ∃ x y : ℝ, M = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1)
  (F1_pos : ℝ)
  (h_dist_F1 : dist M (F1_pos, 0) = 18) :
  (∃ (F2_dist : ℝ), (F2_dist = 8 ∨ F2_dist = 28) ∧ dist M (F2_dist, 0) = F2_dist) := 
sorry

end NUMINAMATH_GPT_hyperbola_focus_distance_l581_58142


namespace NUMINAMATH_GPT_percentage_of_males_l581_58179

noncomputable def total_employees : ℝ := 1800
noncomputable def males_below_50_years_old : ℝ := 756
noncomputable def percentage_below_50 : ℝ := 0.70

theorem percentage_of_males : (males_below_50_years_old / percentage_below_50 / total_employees) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_males_l581_58179


namespace NUMINAMATH_GPT_sqrt_meaningful_iff_l581_58106

theorem sqrt_meaningful_iff (x : ℝ) : (3 - x ≥ 0) ↔ (x ≤ 3) := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_iff_l581_58106


namespace NUMINAMATH_GPT_surface_area_after_removal_l581_58184

theorem surface_area_after_removal :
  let cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (cube_side * cube_side)
  (original_surface_area = 96) ->
  (6 * (cube_side * cube_side) - 8 * 3 * (corner_cube_side * corner_cube_side) + 8 * 3 * (corner_cube_side * corner_cube_side) = 96) :=
by
  intros
  sorry

end NUMINAMATH_GPT_surface_area_after_removal_l581_58184


namespace NUMINAMATH_GPT_total_distance_biked_two_days_l581_58170

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end NUMINAMATH_GPT_total_distance_biked_two_days_l581_58170


namespace NUMINAMATH_GPT_increased_area_l581_58191

variable (r : ℝ)

theorem increased_area (r : ℝ) : 
  let initial_area : ℝ := π * r^2
  let final_area : ℝ := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π := by
sorry

end NUMINAMATH_GPT_increased_area_l581_58191


namespace NUMINAMATH_GPT_unique_integer_solution_quad_eqns_l581_58110

def is_single_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem unique_integer_solution_quad_eqns : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ is_single_digit_prime a ∧ is_single_digit_prime b ∧ is_single_digit_prime c ∧ 
                     ∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ S.card = 7 :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_quad_eqns_l581_58110


namespace NUMINAMATH_GPT_problem_l581_58147

variable {w z : ℝ}

theorem problem (hw : w = 8) (hz : z = 3) (h : ∀ z w, z * (w^(1/3)) = 6) : w = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_l581_58147


namespace NUMINAMATH_GPT_number_of_older_females_l581_58172

theorem number_of_older_females (total_population : ℕ) (num_groups : ℕ) (one_group_population : ℕ) :
  total_population = 1000 → num_groups = 5 → total_population = num_groups * one_group_population →
  one_group_population = 200 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_older_females_l581_58172


namespace NUMINAMATH_GPT_karens_class_fund_l581_58140

noncomputable def ratio_of_bills (T W : ℕ) : ℕ × ℕ := (T / Nat.gcd T W, W / Nat.gcd T W)

theorem karens_class_fund (T W : ℕ) (hW : W = 3) (hfund : 10 * T + 20 * W = 120) :
  ratio_of_bills T W = (2, 1) :=
by
  sorry

end NUMINAMATH_GPT_karens_class_fund_l581_58140
