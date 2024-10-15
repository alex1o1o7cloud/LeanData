import Mathlib

namespace NUMINAMATH_GPT_square_side_length_exists_l1661_166148

-- Define the dimensions of the tile
structure Tile where
  width : Nat
  length : Nat

-- Define the specific tile used in the problem
def given_tile : Tile :=
  { width := 16, length := 24 }

-- Define the condition of forming a square using 6 tiles
def forms_square_with_6_tiles (tile : Tile) (side_length : Nat) : Prop :=
  (2 * tile.length = side_length) ∧ (3 * tile.width = side_length)

-- Problem statement requiring proof
theorem square_side_length_exists : forms_square_with_6_tiles given_tile 48 :=
  sorry

end NUMINAMATH_GPT_square_side_length_exists_l1661_166148


namespace NUMINAMATH_GPT_third_row_number_l1661_166178

-- Define the conditions to fill the grid
def grid (n : Nat) := Fin 4 → Fin 4 → Fin n

-- Ensure each number 1-4 in each cell such that numbers do not repeat
def unique_in_row (g : grid 4) : Prop :=
  ∀ i j1 j2, j1 ≠ j2 → g i j1 ≠ g i j2

def unique_in_col (g : grid 4) : Prop :=
  ∀ j i1 i2, i1 ≠ i2 → g i1 j ≠ g i1 j

-- Define the external hints condition, encapsulating the provided hints.
def hints_condition (g : grid 4) : Prop :=
  -- Example placeholders for hint conditions that would be expanded accordingly.
  g 0 0 = 3 ∨ g 0 1 = 2 -- First row hints interpreted constraints
  -- Additional hint conditions to be added accordingly

-- Prove the correct number formed by the numbers in the third row is 4213
theorem third_row_number (g : grid 4) :
  unique_in_row g ∧ unique_in_col g ∧ hints_condition g →
  (g 2 0 = 4 ∧ g 2 1 = 2 ∧ g 2 2 = 1 ∧ g 2 3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_third_row_number_l1661_166178


namespace NUMINAMATH_GPT_average_height_of_trees_l1661_166143

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end NUMINAMATH_GPT_average_height_of_trees_l1661_166143


namespace NUMINAMATH_GPT_find_numerator_l1661_166195

theorem find_numerator (n : ℕ) : 
  (n : ℚ) / 22 = 9545 / 10000 → 
  n = 9545 * 22 / 10000 :=
by sorry

end NUMINAMATH_GPT_find_numerator_l1661_166195


namespace NUMINAMATH_GPT_find_number_250_l1661_166168

theorem find_number_250 (N : ℤ)
  (h1 : 5 * N = 8 * 156 + 2): N = 250 :=
sorry

end NUMINAMATH_GPT_find_number_250_l1661_166168


namespace NUMINAMATH_GPT_delta_value_l1661_166166

-- Define the variables and the hypothesis
variable (Δ : Int)
variable (h : 5 * (-3) = Δ - 3)

-- State the theorem
theorem delta_value : Δ = -12 := by
  sorry

end NUMINAMATH_GPT_delta_value_l1661_166166


namespace NUMINAMATH_GPT_n_times_s_eq_2023_l1661_166111

noncomputable def S := { x : ℝ | x > 0 }

-- Function f: S → ℝ
def f (x : ℝ) : ℝ := sorry

-- Condition: f(x) f(y) = f(xy) + 2023 * (2/x + 2/y + 2022) for all x, y > 0
axiom f_property (x y : ℝ) (hx : x > 0) (hy : y > 0) : f x * f y = f (x * y) + 2023 * (2 / x + 2 / y + 2022)

-- Theorem: Prove n × s = 2023 where n is the number of possible values of f(2) and s is the sum of all possible values of f(2)
theorem n_times_s_eq_2023 (n s : ℕ) : n * s = 2023 :=
sorry

end NUMINAMATH_GPT_n_times_s_eq_2023_l1661_166111


namespace NUMINAMATH_GPT_dictionary_cost_l1661_166110

def dinosaur_book_cost : ℕ := 19
def children_cookbook_cost : ℕ := 7
def saved_amount : ℕ := 8
def needed_amount : ℕ := 29

def total_amount_needed := saved_amount + needed_amount
def combined_books_cost := dinosaur_book_cost + children_cookbook_cost

theorem dictionary_cost : total_amount_needed - combined_books_cost = 11 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_dictionary_cost_l1661_166110


namespace NUMINAMATH_GPT_vector_parallel_unique_solution_l1661_166183

def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

theorem vector_parallel_unique_solution (m : ℝ) :
  let a := (m^2 - 1, m + 1)
  let b := (1, -2)
  a ≠ (0, 0) → is_parallel a b → m = 1/2 := by
  sorry

end NUMINAMATH_GPT_vector_parallel_unique_solution_l1661_166183


namespace NUMINAMATH_GPT_evaluate_expression_l1661_166194

variable (a b c d e : ℝ)

-- The equivalent proof problem statement
theorem evaluate_expression 
  (h : (a / b * c - d + e = a / (b * c - d - e))) : 
  a / b * c - d + e = a / (b * c - d - e) :=
by 
  exact h

-- Placeholder for the proof
#check evaluate_expression

end NUMINAMATH_GPT_evaluate_expression_l1661_166194


namespace NUMINAMATH_GPT_greatest_possible_integer_radius_l1661_166130

theorem greatest_possible_integer_radius (r : ℕ) (h : ∀ (A : ℝ), A = Real.pi * (r : ℝ)^2 → A < 75 * Real.pi) : r ≤ 8 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_integer_radius_l1661_166130


namespace NUMINAMATH_GPT_range_of_m_l1661_166129

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

theorem range_of_m (G_is_square : ∃ c d, ∀ x, G x m = (c * x + d) ^ 2) : 3 < m ∧ m < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1661_166129


namespace NUMINAMATH_GPT_total_arrangements_excluding_zhang_for_shooting_event_l1661_166191

theorem total_arrangements_excluding_zhang_for_shooting_event
  (students : Fin 5) 
  (events : Fin 3)
  (shooting : events ≠ 0) : 
  ∃ arrangements, arrangements = 48 := 
sorry

end NUMINAMATH_GPT_total_arrangements_excluding_zhang_for_shooting_event_l1661_166191


namespace NUMINAMATH_GPT_find_x_l1661_166180

theorem find_x (x : ℝ) (h : 40 * x - 138 = 102) : x = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1661_166180


namespace NUMINAMATH_GPT_overtime_hours_l1661_166156

theorem overtime_hours (x y : ℕ) 
  (h1 : 60 * x + 90 * y = 3240) 
  (h2 : x + y = 50) : 
  y = 8 :=
by
  sorry

end NUMINAMATH_GPT_overtime_hours_l1661_166156


namespace NUMINAMATH_GPT_least_value_a_plus_b_l1661_166197

theorem least_value_a_plus_b (a b : ℕ) (h : 20 / 19 = 1 + 1 / (1 + a / b)) : a + b = 19 :=
sorry

end NUMINAMATH_GPT_least_value_a_plus_b_l1661_166197


namespace NUMINAMATH_GPT_travel_time_reduction_impossible_proof_l1661_166174

noncomputable def travel_time_reduction_impossible : Prop :=
  ∀ (x : ℝ), x > 60 → ¬ (1 / x * 60 = 1 - 1)

theorem travel_time_reduction_impossible_proof : travel_time_reduction_impossible :=
sorry

end NUMINAMATH_GPT_travel_time_reduction_impossible_proof_l1661_166174


namespace NUMINAMATH_GPT_inequality_sqrt_l1661_166137

open Real

theorem inequality_sqrt (x y : ℝ) :
  (sqrt (x^2 - 2*x*y) > sqrt (1 - y^2)) ↔ 
    ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_l1661_166137


namespace NUMINAMATH_GPT_angle_bisector_form_l1661_166169

noncomputable def P : ℝ × ℝ := (-8, 5)
noncomputable def Q : ℝ × ℝ := (-15, -19)
noncomputable def R : ℝ × ℝ := (1, -7)

-- Function to check if the given equation can be in the form ax + 2y + c = 0
-- and that a + c equals 89.
theorem angle_bisector_form (a c : ℝ) : a + c = 89 :=
by
   sorry

end NUMINAMATH_GPT_angle_bisector_form_l1661_166169


namespace NUMINAMATH_GPT_factory_correct_decision_prob_l1661_166189

def prob_correct_decision (p : ℝ) : ℝ :=
  let prob_all_correct := p * p * p
  let prob_two_correct_one_incorrect := 3 * p * p * (1 - p)
  prob_all_correct + prob_two_correct_one_incorrect

theorem factory_correct_decision_prob : prob_correct_decision 0.8 = 0.896 :=
by
  sorry

end NUMINAMATH_GPT_factory_correct_decision_prob_l1661_166189


namespace NUMINAMATH_GPT_p_neither_sufficient_nor_necessary_l1661_166159

theorem p_neither_sufficient_nor_necessary (x y : ℝ) :
  (x > 1 ∧ y > 1) ↔ ¬((x > 1 ∧ y > 1) → (x + y > 3)) ∧ ¬((x + y > 3) → (x > 1 ∧ y > 1)) :=
by
  sorry

end NUMINAMATH_GPT_p_neither_sufficient_nor_necessary_l1661_166159


namespace NUMINAMATH_GPT_maximum_price_for_360_skewers_price_for_1920_profit_l1661_166102

-- Define the number of skewers sold as a function of the price
def skewers_sold (price : ℝ) : ℝ := 300 + 60 * (10 - price)

-- Define the profit as a function of the price
def profit (price : ℝ) : ℝ := (skewers_sold price) * (price - 3)

-- Maximum price for selling at least 360 skewers per day
theorem maximum_price_for_360_skewers (price : ℝ) (h : skewers_sold price ≥ 360) : price ≤ 9 :=
by {
    sorry
}

-- Price to achieve a profit of 1920 yuan per day with price constraint
theorem price_for_1920_profit (price : ℝ) (h₁ : profit price = 1920) (h₂ : price ≤ 8) : price = 7 :=
by {
    sorry
}

end NUMINAMATH_GPT_maximum_price_for_360_skewers_price_for_1920_profit_l1661_166102


namespace NUMINAMATH_GPT_unique_solution_l1661_166186

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (f x)) + f (f y) = f y + x

-- Define the main theorem
theorem unique_solution (f : ℝ → ℝ) :
  (∀ x y, functional_eq f x y) → (∀ x, f x = x) :=
by
  intros h x
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_unique_solution_l1661_166186


namespace NUMINAMATH_GPT_max_profit_30000_l1661_166101

noncomputable def max_profit (type_A : ℕ) (type_B : ℕ) : ℝ := 
  10000 * type_A + 5000 * type_B

theorem max_profit_30000 :
  ∃ (type_A type_B : ℕ), 
  (4 * type_A + 1 * type_B ≤ 10) ∧
  (18 * type_A + 15 * type_B ≤ 66) ∧
  max_profit type_A type_B = 30000 :=
sorry

end NUMINAMATH_GPT_max_profit_30000_l1661_166101


namespace NUMINAMATH_GPT_nine_div_repeating_decimal_l1661_166119

noncomputable def repeating_decimal := 1 / 3

theorem nine_div_repeating_decimal : 9 / repeating_decimal = 27 := by
  sorry

end NUMINAMATH_GPT_nine_div_repeating_decimal_l1661_166119


namespace NUMINAMATH_GPT_coefficient_a2_in_expansion_l1661_166170

theorem coefficient_a2_in_expansion:
  let a := (x - 1)^4
  let expansion := a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4
  a2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_a2_in_expansion_l1661_166170


namespace NUMINAMATH_GPT_smallest_next_smallest_sum_l1661_166173

-- Defining the set of numbers as constants
def nums : Set ℕ := {10, 11, 12, 13}

-- Define the smallest number in the set
def smallest : ℕ := 10

-- Define the next smallest number in the set
def next_smallest : ℕ := 11

-- The main theorem statement
theorem smallest_next_smallest_sum : smallest + next_smallest = 21 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_next_smallest_sum_l1661_166173


namespace NUMINAMATH_GPT_part1_part2_part3_l1661_166162

-- Conditions
def A : Set ℝ := { x : ℝ | 2 < x ∧ x < 6 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m }

-- Proof statements
theorem part1 : A ∪ B 2 = { x : ℝ | 2 < x ∧ x < 6 } := by
  sorry

theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) → m ≤ 3 := by
  sorry

theorem part3 (m : ℝ) : (∃ x, x ∈ B m) ∧ (∀ x, x ∉ A ∩ B m) → m ≥ 5 := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l1661_166162


namespace NUMINAMATH_GPT_sum_of_numbers_l1661_166117

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1661_166117


namespace NUMINAMATH_GPT_determine_k_l1661_166164

theorem determine_k (k : ℝ) : 
  (2 * k * (-1/2) - 3 = -7 * 3) → k = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_k_l1661_166164


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l1661_166181

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : 5 * (v_m + v_s) = 45) (h2 : 5 * (v_m - v_s) = 25) : v_m = 7 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l1661_166181


namespace NUMINAMATH_GPT_find_f_2000_l1661_166108

variable (f : ℕ → ℕ)
variable (x : ℕ)

axiom initial_condition : f 0 = 1
axiom recurrence_relation : ∀ x, f (x + 2) = f x + 4 * x + 2

theorem find_f_2000 : f 2000 = 3998001 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2000_l1661_166108


namespace NUMINAMATH_GPT_division_multiplication_relation_l1661_166155

theorem division_multiplication_relation (h: 7650 / 306 = 25) :
  25 * 306 = 7650 ∧ 7650 / 25 = 306 := 
by 
  sorry

end NUMINAMATH_GPT_division_multiplication_relation_l1661_166155


namespace NUMINAMATH_GPT_correct_calculation_l1661_166182

theorem correct_calculation : (6 + (-13)) = -7 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1661_166182


namespace NUMINAMATH_GPT_find_point_C_on_z_axis_l1661_166125

noncomputable def point_c_condition (C : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) : Prop :=
  dist C A = dist C B

theorem find_point_C_on_z_axis :
  ∃ C : ℝ × ℝ × ℝ, C = (0, 0, 1) ∧ point_c_condition C (1, 0, 2) (1, 1, 1) :=
by
  use (0, 0, 1)
  simp [point_c_condition]
  sorry

end NUMINAMATH_GPT_find_point_C_on_z_axis_l1661_166125


namespace NUMINAMATH_GPT_ratio_of_lengths_l1661_166146

theorem ratio_of_lengths (total_length short_length : ℕ)
  (h1 : total_length = 35)
  (h2 : short_length = 10) :
  short_length / (total_length - short_length) = 2 / 5 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_ratio_of_lengths_l1661_166146


namespace NUMINAMATH_GPT_incorrect_statement_C_l1661_166152

theorem incorrect_statement_C (a b : ℤ) (h : |a| = |b|) : (a ≠ b ∧ a = -b) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l1661_166152


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l1661_166120

theorem batsman_average_after_17th_inning :
  ∃ x : ℤ, (63 + (16 * x) = 17 * (x + 3)) ∧ (x + 3 = 17) :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l1661_166120


namespace NUMINAMATH_GPT_equilibrium_force_l1661_166115

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def expected_f4 : ℝ × ℝ := (1, 2)

theorem equilibrium_force :
  (1, 2) = -(f1 + f2 + f3) := 
by
  sorry

end NUMINAMATH_GPT_equilibrium_force_l1661_166115


namespace NUMINAMATH_GPT_evaluate_expression_l1661_166193

variable (m n p : ℝ)

theorem evaluate_expression 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1661_166193


namespace NUMINAMATH_GPT_top_width_is_76_l1661_166142

-- Definitions of the conditions
def bottom_width : ℝ := 4
def area : ℝ := 10290
def depth : ℝ := 257.25

-- The main theorem to prove that the top width equals 76 meters
theorem top_width_is_76 (x : ℝ) (h : 10290 = 1/2 * (x + 4) * 257.25) : x = 76 :=
by {
  sorry
}

end NUMINAMATH_GPT_top_width_is_76_l1661_166142


namespace NUMINAMATH_GPT_number_of_folds_l1661_166176

theorem number_of_folds (n : ℕ) :
  (3 * (8 * 8)) / n = 48 → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_folds_l1661_166176


namespace NUMINAMATH_GPT_max_profit_l1661_166185

noncomputable def profit (x : ℝ) : ℝ :=
  20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem max_profit :
  ∃ x : ℝ, 4 ≤ x ∧ x ≤ 12 ∧ 
  (∀ y : ℝ, 4 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧ profit x = 96 * Real.log 6 - 78 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l1661_166185


namespace NUMINAMATH_GPT_number_of_groups_l1661_166118

-- Define constants
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6

-- Define the theorem to be proven
theorem number_of_groups :
  (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_groups_l1661_166118


namespace NUMINAMATH_GPT_collinear_points_on_curve_sum_zero_l1661_166161

theorem collinear_points_on_curve_sum_zero
  {x1 y1 x2 y2 x3 y3 : ℝ}
  (h_curve1 : y1^2 = x1^3)
  (h_curve2 : y2^2 = x2^3)
  (h_curve3 : y3^2 = x3^3)
  (h_collinear : ∃ (a b c k : ℝ), k ≠ 0 ∧ 
    a * x1 + b * y1 + c = 0 ∧
    a * x2 + b * y2 + c = 0 ∧
    a * x3 + b * y3 + c = 0) :
  x1 / y1 + x2 / y2 + x3 / y3 = 0 :=
sorry

end NUMINAMATH_GPT_collinear_points_on_curve_sum_zero_l1661_166161


namespace NUMINAMATH_GPT_product_of_numbers_l1661_166192

theorem product_of_numbers (x y : ℝ) 
  (h₁ : x + y = 8 * (x - y)) 
  (h₂ : x * y = 40 * (x - y)) : x * y = 4032 := 
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1661_166192


namespace NUMINAMATH_GPT_cosine_function_range_l1661_166145

theorem cosine_function_range : 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), -1/2 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧
  (∃ a ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos a = 1) ∧
  (∃ b ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos b = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_cosine_function_range_l1661_166145


namespace NUMINAMATH_GPT_find_ABC_l1661_166139

-- Define the angles as real numbers in degrees
variables (ABC CBD DBC DBE ABE : ℝ)

-- Assert the given conditions
axiom horz_angle: CBD = 90
axiom DBC_ABC_relation : DBC = ABC + 30
axiom straight_angle: DBE = 180
axiom measure_abe: ABE = 145

-- State the proof problem
theorem find_ABC : ABC = 30 :=
by
  -- Include all steps required to derive the conclusion in the proof
  sorry

end NUMINAMATH_GPT_find_ABC_l1661_166139


namespace NUMINAMATH_GPT_ellipse_foci_cond_l1661_166167

theorem ellipse_foci_cond (m n : ℝ) (h_cond : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → (m > n ∧ n > 0)) ∧ ((m > n ∧ n > 0) → ∀ x y : ℝ, mx^2 + ny^2 = 1) :=
sorry

end NUMINAMATH_GPT_ellipse_foci_cond_l1661_166167


namespace NUMINAMATH_GPT_maximum_value_expression_l1661_166140

theorem maximum_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
sorry

end NUMINAMATH_GPT_maximum_value_expression_l1661_166140


namespace NUMINAMATH_GPT_ratio_expression_l1661_166144

-- Given conditions: X : Y : Z = 3 : 2 : 6
def ratio (X Y Z : ℚ) : Prop := X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- The expression to be evaluated
def expr (X Y Z : ℚ) : ℚ := (4 * X + 3 * Y) / (5 * Z - 2 * X)

-- The proof problem itself
theorem ratio_expression (X Y Z : ℚ) (h : ratio X Y Z) : expr X Y Z = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_expression_l1661_166144


namespace NUMINAMATH_GPT_johns_photo_world_sitting_fee_l1661_166175

variable (J : ℝ)

theorem johns_photo_world_sitting_fee
  (h1 : ∀ n : ℝ, n = 12 → 2.75 * n + J = 1.50 * n + 140) : J = 125 :=
by
  -- We will skip the proof since it is not required by the problem statement.
  sorry

end NUMINAMATH_GPT_johns_photo_world_sitting_fee_l1661_166175


namespace NUMINAMATH_GPT_tyler_eggs_in_fridge_l1661_166126

def recipe_eggs_for_four : Nat := 2
def people_multiplier : Nat := 2
def eggs_needed : Nat := recipe_eggs_for_four * people_multiplier
def eggs_to_buy : Nat := 1
def eggs_in_fridge : Nat := eggs_needed - eggs_to_buy

theorem tyler_eggs_in_fridge : eggs_in_fridge = 3 := by
  sorry

end NUMINAMATH_GPT_tyler_eggs_in_fridge_l1661_166126


namespace NUMINAMATH_GPT_line_direction_vector_correct_l1661_166109

theorem line_direction_vector_correct :
  ∃ (A B C : ℝ), (A = 2 ∧ B = -3 ∧ C = 1) ∧ 
  ∃ (v w : ℝ), (v = A ∧ w = B) :=
by
  sorry

end NUMINAMATH_GPT_line_direction_vector_correct_l1661_166109


namespace NUMINAMATH_GPT_min_value_expr_l1661_166160

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l1661_166160


namespace NUMINAMATH_GPT_problem_solution_l1661_166198

theorem problem_solution {a b : ℝ} (h : a * b + b^2 = 12) : (a + b)^2 - (a + b) * (a - b) = 24 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1661_166198


namespace NUMINAMATH_GPT_probability_of_correct_digit_in_two_attempts_l1661_166190

theorem probability_of_correct_digit_in_two_attempts : 
  let num_possible_digits := 10
  let num_attempts := 2
  let total_possible_outcomes := num_possible_digits * (num_possible_digits - 1)
  let total_favorable_outcomes := (num_possible_digits - 1) + (num_possible_digits - 1)
  let probability := (total_favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)
  probability = (1 / 5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_correct_digit_in_two_attempts_l1661_166190


namespace NUMINAMATH_GPT_arrangement_count_correct_l1661_166131

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_correct_l1661_166131


namespace NUMINAMATH_GPT_percentage_of_motorists_speeding_l1661_166157

-- Definitions based on the conditions
def total_motorists : Nat := 100
def percent_motorists_receive_tickets : Real := 0.20
def percent_speeders_no_tickets : Real := 0.20

-- Define the variables for the number of speeders
variable (x : Real) -- the percentage of total motorists who speed 

-- Lean statement to formalize the problem
theorem percentage_of_motorists_speeding 
  (h1 : 20 = (0.80 * x) * (total_motorists / 100)) : 
  x = 25 :=
sorry

end NUMINAMATH_GPT_percentage_of_motorists_speeding_l1661_166157


namespace NUMINAMATH_GPT_phil_books_remaining_pages_l1661_166112

/-- We define the initial number of books and the number of pages per book. -/
def initial_books : Nat := 10
def pages_per_book : Nat := 100
def lost_books : Nat := 2

/-- The goal is to find the total number of pages Phil has left after losing 2 books. -/
theorem phil_books_remaining_pages : (initial_books - lost_books) * pages_per_book = 800 := by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_phil_books_remaining_pages_l1661_166112


namespace NUMINAMATH_GPT_inequality_holds_l1661_166104

theorem inequality_holds (a b : ℝ) : (6 * a - 3 * b - 3) * (a ^ 2 + a ^ 2 * b - 2 * a ^ 3) ≤ 0 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1661_166104


namespace NUMINAMATH_GPT_inverse_geometric_sequence_l1661_166106

-- Define that a, b, c form a geometric sequence
def geometric_sequence (a b c : ℝ) := b^2 = a * c

-- Define the theorem: if b^2 = a * c, then a, b, c form a geometric sequence
theorem inverse_geometric_sequence (a b c : ℝ) (h : b^2 = a * c) : geometric_sequence a b c :=
by
  sorry

end NUMINAMATH_GPT_inverse_geometric_sequence_l1661_166106


namespace NUMINAMATH_GPT_Theresa_helper_hours_l1661_166179

theorem Theresa_helper_hours :
  ∃ x : ℕ, (7 + 10 + 8 + 11 + 9 + 7 + x) / 7 = 9 ∧ x ≥ 10 := by
  sorry

end NUMINAMATH_GPT_Theresa_helper_hours_l1661_166179


namespace NUMINAMATH_GPT_find_N_l1661_166107

theorem find_N : ∃ N : ℕ, 36^2 * 72^2 = 12^2 * N^2 ∧ N = 216 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1661_166107


namespace NUMINAMATH_GPT_sum_of_21st_set_l1661_166177

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

def first_element_of_set (n : ℕ) : ℕ := triangular_number n - n + 1

def sum_of_elements_in_set (n : ℕ) : ℕ := 
  n * ((first_element_of_set n + triangular_number n) / 2)

theorem sum_of_21st_set : sum_of_elements_in_set 21 = 4641 := by 
  sorry

end NUMINAMATH_GPT_sum_of_21st_set_l1661_166177


namespace NUMINAMATH_GPT_prime_pairs_l1661_166133

-- Define the predicate to check whether a number is a prime.
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the main theorem.
theorem prime_pairs (p q : Nat) (hp : is_prime p) (hq : is_prime q) : 
  (p^3 - q^5 = (p + q)^2) → (p = 7 ∧ q = 3) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_l1661_166133


namespace NUMINAMATH_GPT_cost_per_meter_l1661_166127

-- Definitions of the conditions
def length_of_plot : ℕ := 63
def breadth_of_plot : ℕ := length_of_plot - 26
def perimeter_of_plot := 2 * length_of_plot + 2 * breadth_of_plot
def total_cost : ℕ := 5300

-- Statement to prove
theorem cost_per_meter : (total_cost : ℚ) / perimeter_of_plot = 26.5 :=
by sorry

end NUMINAMATH_GPT_cost_per_meter_l1661_166127


namespace NUMINAMATH_GPT_rose_price_vs_carnation_price_l1661_166147

variable (x y : ℝ)

theorem rose_price_vs_carnation_price
  (h1 : 3 * x + 2 * y > 8)
  (h2 : 2 * x + 3 * y < 7) :
  x > 2 * y :=
sorry

end NUMINAMATH_GPT_rose_price_vs_carnation_price_l1661_166147


namespace NUMINAMATH_GPT_common_chord_length_l1661_166172

theorem common_chord_length (r d : ℝ) (hr : r = 12) (hd : d = 16) : 
  ∃ l : ℝ, l = 8 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_common_chord_length_l1661_166172


namespace NUMINAMATH_GPT_sum_of_all_possible_k_values_l1661_166123

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_k_values_l1661_166123


namespace NUMINAMATH_GPT_min_fence_length_l1661_166136

theorem min_fence_length (w l F: ℝ) (h1: l = 2 * w) (h2: 2 * w^2 ≥ 500) : F = 96 :=
by sorry

end NUMINAMATH_GPT_min_fence_length_l1661_166136


namespace NUMINAMATH_GPT_solve_for_s_l1661_166134

theorem solve_for_s (s t : ℚ) (h1 : 7 * s + 8 * t = 150) (h2 : s = 2 * t + 3) : s = 162 / 11 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_s_l1661_166134


namespace NUMINAMATH_GPT_lcm_of_numbers_l1661_166158

theorem lcm_of_numbers (a b lcm hcf : ℕ) (h_prod : a * b = 45276) (h_hcf : hcf = 22) (h_relation : a * b = hcf * lcm) : lcm = 2058 :=
by sorry

end NUMINAMATH_GPT_lcm_of_numbers_l1661_166158


namespace NUMINAMATH_GPT_john_reading_time_l1661_166114

theorem john_reading_time:
  let weekday_hours_moses := 1.5
  let weekday_rate_moses := 30
  let saturday_hours_moses := 2
  let saturday_rate_moses := 40
  let pages_moses := 450
  let weekday_hours_rest := 1.5
  let weekday_rate_rest := 45
  let saturday_hours_rest := 2.5
  let saturday_rate_rest := 60
  let pages_rest := 2350
  let weekdays_per_week := 5
  let saturdays_per_week := 1
  let total_pages_per_week_moses := (weekday_hours_moses * weekday_rate_moses * weekdays_per_week) + 
                                    (saturday_hours_moses * saturday_rate_moses * saturdays_per_week)
  let total_pages_per_week_rest := (weekday_hours_rest * weekday_rate_rest * weekdays_per_week) + 
                                   (saturday_hours_rest * saturday_rate_rest * saturdays_per_week)
  let weeks_moses := (pages_moses / total_pages_per_week_moses).ceil
  let weeks_rest := (pages_rest / total_pages_per_week_rest).ceil
  let total_weeks := weeks_moses + weeks_rest
  total_weeks = 7 :=
by
  -- placeholders for the proof steps.
  sorry

end NUMINAMATH_GPT_john_reading_time_l1661_166114


namespace NUMINAMATH_GPT_parking_lot_length_l1661_166121

theorem parking_lot_length (W : ℝ) (U : ℝ) (A_car : ℝ) (N_cars : ℕ) (H_w : W = 400) (H_u : U = 0.80) (H_Acar : A_car = 10) (H_Ncars : N_cars = 16000) :
  (U * (W * L) = N_cars * A_car) → (L = 500) :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_length_l1661_166121


namespace NUMINAMATH_GPT_inequality_bounds_l1661_166171

noncomputable def f (a b A B : ℝ) (θ : ℝ) : ℝ :=
  1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)

theorem inequality_bounds (a b A B : ℝ) (h : ∀ θ : ℝ, f a b A B θ ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_bounds_l1661_166171


namespace NUMINAMATH_GPT_gabriel_forgot_days_l1661_166165

def days_in_july : ℕ := 31
def days_taken : ℕ := 28

theorem gabriel_forgot_days : days_in_july - days_taken = 3 := by
  sorry

end NUMINAMATH_GPT_gabriel_forgot_days_l1661_166165


namespace NUMINAMATH_GPT_total_money_l1661_166100

variable (A B C : ℕ)

theorem total_money
  (h1 : A + C = 250)
  (h2 : B + C = 450)
  (h3 : C = 100) :
  A + B + C = 600 := by
  sorry

end NUMINAMATH_GPT_total_money_l1661_166100


namespace NUMINAMATH_GPT_monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l1661_166163

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin (2 * x + (Real.pi / 4)))

theorem monotonic_intervals_increasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + Real.pi / 8) → f x ≤ f y :=
sorry

theorem monotonic_intervals_decreasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + 5 * Real.pi / 8) → f x ≥ f y :=
sorry

theorem maximum_value (k : ℤ) :
  f (k * Real.pi + Real.pi / 8) = 3 :=
sorry

theorem minimum_value (k : ℤ) :
  f (k * Real.pi - 3 * Real.pi / 8) = -3 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l1661_166163


namespace NUMINAMATH_GPT_min_diff_f_l1661_166141

def f (x : ℝ) := 2017 * x ^ 2 - 2018 * x + 2019 * 2020

theorem min_diff_f (t : ℝ) : 
  let f_max := max (f t) (f (t + 2))
  let f_min := min (f t) (f (t + 2))
  (f_max - f_min) ≥ 2017 :=
sorry

end NUMINAMATH_GPT_min_diff_f_l1661_166141


namespace NUMINAMATH_GPT_incorrect_statement_D_l1661_166103

theorem incorrect_statement_D (a b r : ℝ) (hr : r > 0) :
  ¬ ∀ b < r, ∃ x, (x - a)^2 + (0 - b)^2 = r^2 :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1661_166103


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1661_166150

def M : Set ℤ := { x | -3 < x ∧ x < 3 }
def N : Set ℤ := { x | x < 1 }

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1661_166150


namespace NUMINAMATH_GPT_brayan_hourly_coffee_l1661_166153

theorem brayan_hourly_coffee (I B : ℕ) (h1 : B = 2 * I) (h2 : I + B = 30) : B / 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_brayan_hourly_coffee_l1661_166153


namespace NUMINAMATH_GPT_maximum_angle_B_in_triangle_l1661_166196

theorem maximum_angle_B_in_triangle
  (A B C M : ℝ × ℝ)
  (hM : midpoint ℝ A B = M)
  (h_angle_MAC : ∃ angle_MAC : ℝ, angle_MAC = 15) :
  ∃ angle_B : ℝ, angle_B = 105 := 
by
  sorry

end NUMINAMATH_GPT_maximum_angle_B_in_triangle_l1661_166196


namespace NUMINAMATH_GPT_max_value_a_l1661_166184

theorem max_value_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) ↔ a ≤ 6 := by
  sorry

end NUMINAMATH_GPT_max_value_a_l1661_166184


namespace NUMINAMATH_GPT_complex_number_solution_l1661_166116

theorem complex_number_solution (z : ℂ) (i : ℂ) (h1 : i * z = (1 - 2 * i) ^ 2) (h2 : i * i = -1) : z = -4 + 3 * i := by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l1661_166116


namespace NUMINAMATH_GPT_googoo_total_buttons_l1661_166199

noncomputable def button_count_shirt_1 : ℕ := 3
noncomputable def button_count_shirt_2 : ℕ := 5
noncomputable def quantity_shirt_1 : ℕ := 200
noncomputable def quantity_shirt_2 : ℕ := 200

theorem googoo_total_buttons :
  (quantity_shirt_1 * button_count_shirt_1) + (quantity_shirt_2 * button_count_shirt_2) = 1600 := by
  sorry

end NUMINAMATH_GPT_googoo_total_buttons_l1661_166199


namespace NUMINAMATH_GPT_complex_sum_magnitude_eq_three_l1661_166132

open Complex

theorem complex_sum_magnitude_eq_three (a b c : ℂ) 
    (h1 : abs a = 1) 
    (h2 : abs b = 1) 
    (h3 : abs c = 1) 
    (h4 : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3) : 
    abs (a + b + c) = 3 := 
sorry

end NUMINAMATH_GPT_complex_sum_magnitude_eq_three_l1661_166132


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l1661_166113

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + m * x + 2 > 0) ↔ m > -3 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l1661_166113


namespace NUMINAMATH_GPT_count_arithmetic_sequence_terms_l1661_166128

theorem count_arithmetic_sequence_terms : 
  ∃ n : ℕ, 
  (∀ k : ℕ, k ≥ 1 → 6 + (k - 1) * 4 = 202 → n = k) ∧ n = 50 :=
by
  sorry

end NUMINAMATH_GPT_count_arithmetic_sequence_terms_l1661_166128


namespace NUMINAMATH_GPT_meals_given_away_l1661_166135

def initial_meals_colt_and_curt : ℕ := 113
def additional_meals_sole_mart : ℕ := 50
def remaining_meals : ℕ := 78
def total_initial_meals : ℕ := initial_meals_colt_and_curt + additional_meals_sole_mart
def given_away_meals (total : ℕ) (remaining : ℕ) : ℕ := total - remaining

theorem meals_given_away : given_away_meals total_initial_meals remaining_meals = 85 :=
by
  sorry

end NUMINAMATH_GPT_meals_given_away_l1661_166135


namespace NUMINAMATH_GPT_xyz_problem_l1661_166154

/-- Given x = 36^2 + 48^2 + 64^3 + 81^2, prove the following:
    - x is a multiple of 3. 
    - x is a multiple of 4.
    - x is a multiple of 9.
    - x is not a multiple of 16. 
-/
theorem xyz_problem (x : ℕ) (h_x : x = 36^2 + 48^2 + 64^3 + 81^2) :
  (x % 3 = 0) ∧ (x % 4 = 0) ∧ (x % 9 = 0) ∧ ¬(x % 16 = 0) := 
by
  have h1 : 36^2 = 1296 := by norm_num
  have h2 : 48^2 = 2304 := by norm_num
  have h3 : 64^3 = 262144 := by norm_num
  have h4 : 81^2 = 6561 := by norm_num
  have hx : x = 1296 + 2304 + 262144 + 6561 := by rw [h_x, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_xyz_problem_l1661_166154


namespace NUMINAMATH_GPT_stock_worth_l1661_166122

theorem stock_worth (X : ℝ)
  (H1 : 0.2 * X * 0.1 = 0.02 * X)  -- 20% of stock at 10% profit given in condition.
  (H2 : 0.8 * X * 0.05 = 0.04 * X) -- Remaining 80% of stock at 5% loss given in condition.
  (H3 : 0.04 * X - 0.02 * X = 400) -- Overall loss incurred is Rs. 400.
  : X = 20000 := 
sorry

end NUMINAMATH_GPT_stock_worth_l1661_166122


namespace NUMINAMATH_GPT_number_of_birds_flew_up_correct_l1661_166187

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end NUMINAMATH_GPT_number_of_birds_flew_up_correct_l1661_166187


namespace NUMINAMATH_GPT_number_of_teachers_students_possible_rental_plans_economical_plan_l1661_166188

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end NUMINAMATH_GPT_number_of_teachers_students_possible_rental_plans_economical_plan_l1661_166188


namespace NUMINAMATH_GPT_shape_area_is_36_l1661_166151

def side_length : ℝ := 3
def num_squares : ℕ := 4
def area_square : ℝ := side_length ^ 2
def total_area : ℝ := num_squares * area_square

theorem shape_area_is_36 :
  total_area = 36 := by
  sorry

end NUMINAMATH_GPT_shape_area_is_36_l1661_166151


namespace NUMINAMATH_GPT_max_length_third_side_l1661_166124

open Real

theorem max_length_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : cos (2 * A) + cos (2 * B) + cos (2 * C) = 1)
  (h2 : a = 9) 
  (h3 : b = 12)
  (h4 : a^2 + b^2 = c^2) : 
  c = 15 := 
sorry

end NUMINAMATH_GPT_max_length_third_side_l1661_166124


namespace NUMINAMATH_GPT_a_investment_l1661_166138

theorem a_investment (B C total_profit A_share: ℝ) (hB: B = 7200) (hC: C = 9600) (htotal_profit: total_profit = 9000) 
  (hA_share: A_share = 1125) : ∃ x : ℝ, (A_share / total_profit) = (x / (x + B + C)) ∧ x = 2400 := 
by
  use 2400
  sorry

end NUMINAMATH_GPT_a_investment_l1661_166138


namespace NUMINAMATH_GPT_equation_squares_l1661_166149

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_equation_squares_l1661_166149


namespace NUMINAMATH_GPT_distance_between_points_l1661_166105

theorem distance_between_points (A B : ℝ) (dA : |A| = 2) (dB : |B| = 7) : |A - B| = 5 ∨ |A - B| = 9 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1661_166105
