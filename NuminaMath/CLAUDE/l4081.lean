import Mathlib

namespace NUMINAMATH_CALUDE_equal_squares_on_8x7_board_l4081_408187

/-- Represents a rectangular board with alternating light and dark squares. -/
structure AlternatingBoard :=
  (rows : Nat)
  (columns : Nat)

/-- Counts the number of dark squares on the board. -/
def count_dark_squares (board : AlternatingBoard) : Nat :=
  (board.rows / 2) * ((board.columns + 1) / 2) + 
  ((board.rows + 1) / 2) * (board.columns / 2)

/-- Counts the number of light squares on the board. -/
def count_light_squares (board : AlternatingBoard) : Nat :=
  ((board.rows + 1) / 2) * ((board.columns + 1) / 2) + 
  (board.rows / 2) * (board.columns / 2)

/-- Theorem stating that for an 8x7 alternating board, the number of dark squares equals the number of light squares. -/
theorem equal_squares_on_8x7_board :
  let board : AlternatingBoard := ⟨8, 7⟩
  count_dark_squares board = count_light_squares board := by
  sorry

#eval count_dark_squares ⟨8, 7⟩
#eval count_light_squares ⟨8, 7⟩

end NUMINAMATH_CALUDE_equal_squares_on_8x7_board_l4081_408187


namespace NUMINAMATH_CALUDE_square_division_reversible_l4081_408181

/-- A square of cells can be divided into equal figures -/
structure CellSquare where
  side : ℕ
  total_cells : ℕ
  total_cells_eq : total_cells = side * side

/-- A division of a cell square into equal figures -/
structure SquareDivision (square : CellSquare) where
  num_figures : ℕ
  cells_per_figure : ℕ
  division_valid : square.total_cells = num_figures * cells_per_figure

theorem square_division_reversible (square : CellSquare) 
  (div1 : SquareDivision square) :
  ∃ (div2 : SquareDivision square), 
    div2.num_figures = div1.cells_per_figure ∧ 
    div2.cells_per_figure = div1.num_figures :=
sorry

end NUMINAMATH_CALUDE_square_division_reversible_l4081_408181


namespace NUMINAMATH_CALUDE_bryce_raisins_l4081_408103

/-- Proves that Bryce received 12 raisins given the conditions of the problem -/
theorem bryce_raisins : 
  ∀ (bryce carter emma : ℕ), 
    bryce = carter + 8 →
    carter = bryce / 3 →
    emma = 2 * carter →
    bryce = 12 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l4081_408103


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l4081_408109

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  (180 * (n - 2) : ℚ) / n = 150 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l4081_408109


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_quotients_l4081_408125

theorem no_simultaneous_integer_quotients : ¬ ∃ (n : ℤ), (∃ (k : ℤ), n - 5 = 6 * k) ∧ (∃ (m : ℤ), n - 1 = 21 * m) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_quotients_l4081_408125


namespace NUMINAMATH_CALUDE_quadratic_root_property_l4081_408158

theorem quadratic_root_property (m : ℝ) : 
  m^2 - m - 3 = 0 → 2023 - m^2 + m = 2020 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l4081_408158


namespace NUMINAMATH_CALUDE_second_char_lines_relation_l4081_408155

/-- Represents a character in a script with a certain number of lines. -/
structure Character where
  lines : ℕ

/-- Represents a script with three characters. -/
structure Script where
  char1 : Character
  char2 : Character
  char3 : Character
  first_has_more : char1.lines = char2.lines + 8
  third_has_two : char3.lines = 2
  first_has_twenty : char1.lines = 20

/-- The theorem stating the relationship between the lines of the second and third characters. -/
theorem second_char_lines_relation (script : Script) : 
  script.char2.lines = 3 * script.char3.lines + 6 := by
  sorry

end NUMINAMATH_CALUDE_second_char_lines_relation_l4081_408155


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l4081_408190

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l4081_408190


namespace NUMINAMATH_CALUDE_money_distribution_l4081_408150

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_value : C = 50) :
  B + C = 350 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l4081_408150


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l4081_408163

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l4081_408163


namespace NUMINAMATH_CALUDE_raul_initial_money_l4081_408100

def initial_money (comics_bought : ℕ) (comic_price : ℕ) (money_left : ℕ) : ℕ :=
  comics_bought * comic_price + money_left

theorem raul_initial_money :
  initial_money 8 4 55 = 87 := by
  sorry

end NUMINAMATH_CALUDE_raul_initial_money_l4081_408100


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l4081_408139

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l4081_408139


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l4081_408126

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of vertices adjacent to any vertex in a decagon -/
def AdjacentVertices : ℕ := 2

/-- The probability of selecting two adjacent vertices in a decagon -/
def ProbAdjacentVertices : ℚ := 2 / 9

theorem adjacent_vertices_probability (d : ℕ) (av : ℕ) (p : ℚ) 
  (h1 : d = Decagon) 
  (h2 : av = AdjacentVertices) 
  (h3 : p = ProbAdjacentVertices) : 
  p = av / (d - 1) := by
  sorry

#check adjacent_vertices_probability

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l4081_408126


namespace NUMINAMATH_CALUDE_one_true_proposition_l4081_408147

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Define the converse
def converse (x : ℝ) : Prop := x^2 > 0 → x > 0

-- Define the negation
def negation (x : ℝ) : Prop := x > 0 → x^2 ≤ 0

-- Define the inverse negation
def inverse_negation (x : ℝ) : Prop := x^2 ≤ 0 → x ≤ 0

-- Theorem stating that exactly one of these is true
theorem one_true_proposition :
  ∃! p : (ℝ → Prop), p = converse ∨ p = negation ∨ p = inverse_negation ∧ ∀ x, p x :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l4081_408147


namespace NUMINAMATH_CALUDE_water_remaining_in_bucket_l4081_408116

theorem water_remaining_in_bucket (initial_water : ℚ) (poured_out : ℚ) : 
  initial_water = 3/4 → poured_out = 1/3 → initial_water - poured_out = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_in_bucket_l4081_408116


namespace NUMINAMATH_CALUDE_systematic_sample_41_l4081_408151

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sample_size : ℕ
  interval : ℕ
  first_selected : ℕ

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_selected + k * s.interval ∧ k < s.sample_size

theorem systematic_sample_41 :
  ∀ s : SystematicSample,
    s.total = 60 →
    s.sample_size = 5 →
    s.interval = s.total / s.sample_size →
    in_sample s 17 →
    in_sample s 41 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_41_l4081_408151


namespace NUMINAMATH_CALUDE_work_completion_time_l4081_408192

/-- The number of days it takes y to complete the work -/
def y_days : ℝ := 40

/-- The number of days it takes x and y together to complete the work -/
def combined_days : ℝ := 13.333333333333332

/-- The number of days it takes x to complete the work -/
def x_days : ℝ := 20

theorem work_completion_time :
  1 / x_days + 1 / y_days = 1 / combined_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4081_408192


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l4081_408142

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)) + (y^2 / (x - 2)) ≥ 12 ∧
  ((x^2 / (y - 2)) + (y^2 / (x - 2)) = 12 ↔ x = 4 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l4081_408142


namespace NUMINAMATH_CALUDE_shooter_probabilities_l4081_408195

/-- The probability of hitting the target in a single shot -/
def hit_probability : ℝ := 0.9

/-- The number of shots -/
def num_shots : ℕ := 4

/-- The probability of hitting the target on the third shot -/
def third_shot_probability : ℝ := hit_probability

/-- The probability of hitting the target at least once in four shots -/
def at_least_one_hit_probability : ℝ := 1 - (1 - hit_probability) ^ num_shots

/-- The number of correct statements -/
def correct_statements : ℕ := 2

theorem shooter_probabilities :
  (third_shot_probability = hit_probability) ∧
  (at_least_one_hit_probability = 1 - (1 - hit_probability) ^ num_shots) ∧
  (correct_statements = 2) := by sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l4081_408195


namespace NUMINAMATH_CALUDE_f_composition_value_l4081_408105

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.sin x

theorem f_composition_value : f (f (7 * Real.pi / 6)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l4081_408105


namespace NUMINAMATH_CALUDE_equation_solution_l4081_408134

theorem equation_solution : 
  ∃ y : ℝ, (y^2 - 3*y - 10)/(y + 2) + (4*y^2 + 17*y - 15)/(4*y - 1) = 5 ∧ y = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4081_408134


namespace NUMINAMATH_CALUDE_sin_240_degrees_l4081_408108

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l4081_408108


namespace NUMINAMATH_CALUDE_rect_to_polar_equiv_l4081_408153

/-- Proves that the point (-1, √3) in rectangular coordinates 
    is equivalent to (2, 2π/3) in polar coordinates. -/
theorem rect_to_polar_equiv : 
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let r : ℝ := 2
  let θ : ℝ := 2 * Real.pi / 3
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → 
  (x = r * Real.cos θ ∧ y = r * Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_equiv_l4081_408153


namespace NUMINAMATH_CALUDE_inequality_not_always_correct_l4081_408191

theorem inequality_not_always_correct
  (x y z w : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hxy : x > y)
  (hz : z ≠ 0)
  (hw : w ≠ 0) :
  ∃ (x' y' z' w' : ℝ),
    x' > 0 ∧ y' > 0 ∧ x' > y' ∧ z' ≠ 0 ∧ w' ≠ 0 ∧
    x' * z' ≤ y' * w' * z' :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_correct_l4081_408191


namespace NUMINAMATH_CALUDE_mikes_games_l4081_408128

theorem mikes_games (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 101 →
  spent_amount = 47 →
  game_cost = 6 →
  (initial_amount - spent_amount) / game_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_mikes_games_l4081_408128


namespace NUMINAMATH_CALUDE_parallelogram_projection_sum_l4081_408152

/-- Parallelogram structure -/
structure Parallelogram where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of adjacent side
  e : ℝ  -- Length of longer diagonal
  pa : ℝ  -- Projection of diagonal on side a
  pb : ℝ  -- Projection of diagonal on side b
  a_pos : 0 < a
  b_pos : 0 < b
  e_pos : 0 < e
  pa_pos : 0 < pa
  pb_pos : 0 < pb

/-- Theorem: In a parallelogram, a * pa + b * pb = e^2 -/
theorem parallelogram_projection_sum (p : Parallelogram) : p.a * p.pa + p.b * p.pb = p.e^2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_projection_sum_l4081_408152


namespace NUMINAMATH_CALUDE_vector_subtraction_l4081_408154

/-- Given two complex numbers representing vectors OA and OB, 
    prove that the complex number representing vector AB is their difference. -/
theorem vector_subtraction (OA OB : ℂ) : 
  OA = 5 + 10*I → OB = 3 - 4*I → (OB - OA) = -2 - 14*I := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l4081_408154


namespace NUMINAMATH_CALUDE_ginger_cakes_l4081_408117

/-- The number of cakes Ginger bakes in 10 years --/
def cakes_in_ten_years : ℕ :=
  let children := 2
  let children_holidays := 4
  let husband_holidays := 6
  let parents := 2
  let years := 10
  let cakes_per_year := children * children_holidays + husband_holidays + parents
  cakes_per_year * years

theorem ginger_cakes : cakes_in_ten_years = 160 := by
  sorry

end NUMINAMATH_CALUDE_ginger_cakes_l4081_408117


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l4081_408167

theorem cube_root_of_negative_eight :
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l4081_408167


namespace NUMINAMATH_CALUDE_tourist_distribution_count_l4081_408122

/-- The number of tour guides -/
def num_guides : ℕ := 3

/-- The number of tourists -/
def num_tourists : ℕ := 8

/-- The number of ways to distribute tourists among guides -/
def distribute_tourists : ℕ := 3^8

/-- The number of ways where at least one guide has no tourists -/
def at_least_one_empty : ℕ := 3 * 2^8

/-- The number of ways where exactly two guides have no tourists -/
def two_empty : ℕ := 3

/-- The number of valid distributions where each guide has at least one tourist -/
def valid_distributions : ℕ := distribute_tourists - at_least_one_empty + two_empty

theorem tourist_distribution_count :
  valid_distributions = 5796 :=
sorry

end NUMINAMATH_CALUDE_tourist_distribution_count_l4081_408122


namespace NUMINAMATH_CALUDE_f_minimum_value_l4081_408173

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 5/2) ∧ (∃ x > 0, f x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l4081_408173


namespace NUMINAMATH_CALUDE_circle_equation_proof_l4081_408171

/-- The circle with center on y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 4)^2 = 8

/-- The line y = -4x -/
def center_line (x y : ℝ) : Prop :=
  y = -4 * x

/-- The line x + y - 1 = 0 -/
def tangent_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- The point P(3, -2) -/
def point_P : ℝ × ℝ :=
  (3, -2)

theorem circle_equation_proof :
  ∃ (c : ℝ × ℝ), 
    (center_line c.1 c.2) ∧ 
    (∀ (x y : ℝ), tangent_line x y → 
      ((x - c.1)^2 + (y - c.2)^2 = (c.1 - point_P.1)^2 + (c.2 - point_P.2)^2)) ↔
    (∀ (x y : ℝ), special_circle x y ↔ 
      ((x - 1)^2 + (y + 4)^2 = (1 - point_P.1)^2 + (-4 - point_P.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l4081_408171


namespace NUMINAMATH_CALUDE_staples_remaining_after_stapling_l4081_408159

/-- Calculates the number of staples left in a stapler after stapling reports. -/
def staples_left (initial_staples : ℕ) (reports_stapled : ℕ) : ℕ :=
  initial_staples - reports_stapled

/-- Converts dozens to individual units. -/
def dozens_to_units (dozens : ℕ) : ℕ :=
  dozens * 12

theorem staples_remaining_after_stapling :
  let initial_staples := 50
  let reports_in_dozens := 3
  let reports_stapled := dozens_to_units reports_in_dozens
  staples_left initial_staples reports_stapled = 14 := by
sorry

end NUMINAMATH_CALUDE_staples_remaining_after_stapling_l4081_408159


namespace NUMINAMATH_CALUDE_papaya_tree_growth_ratio_l4081_408123

/-- Papaya tree growth problem -/
theorem papaya_tree_growth_ratio : 
  ∀ (growth_1 growth_2 growth_3 growth_4 growth_5 : ℝ),
  growth_1 = 2 →
  growth_2 = growth_1 * 1.5 →
  growth_3 = growth_2 * 1.5 →
  growth_5 = growth_4 / 2 →
  growth_1 + growth_2 + growth_3 + growth_4 + growth_5 = 23 →
  growth_4 / growth_3 = 2 := by
sorry


end NUMINAMATH_CALUDE_papaya_tree_growth_ratio_l4081_408123


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l4081_408115

-- Define the set difference operation
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetric_difference (M N : Set ℝ) : Set ℝ := 
  set_difference M N ∪ set_difference N M

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 3^x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -(x-1)^2 + 2}

-- State the theorem
theorem symmetric_difference_A_B : 
  symmetric_difference A B = {y | y ≤ 0 ∨ y > 2} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l4081_408115


namespace NUMINAMATH_CALUDE_marks_animals_legs_l4081_408169

/-- The number of legs of all animals owned by Mark -/
def total_legs (num_kangaroos : ℕ) (num_goats : ℕ) : ℕ :=
  2 * num_kangaroos + 4 * num_goats

/-- Theorem stating the total number of legs of Mark's animals -/
theorem marks_animals_legs : 
  let num_kangaroos : ℕ := 23
  let num_goats : ℕ := 3 * num_kangaroos
  total_legs num_kangaroos num_goats = 322 := by
  sorry

#check marks_animals_legs

end NUMINAMATH_CALUDE_marks_animals_legs_l4081_408169


namespace NUMINAMATH_CALUDE_car_distance_l4081_408102

theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time : ℝ) :
  train_speed = 100 →
  car_speed_ratio = 5 / 8 →
  time = 45 / 60 →
  car_speed_ratio * train_speed * time = 46.875 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l4081_408102


namespace NUMINAMATH_CALUDE_at_least_one_red_certain_l4081_408136

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of balls drawn
def drawn_balls : ℕ := 4

-- Theorem statement
theorem at_least_one_red_certain :
  ∀ (draw : Finset ℕ),
  draw.card = drawn_balls →
  draw ⊆ Finset.range total_balls →
  ∃ (x : ℕ), x ∈ draw ∧ x < red_balls :=
sorry

end NUMINAMATH_CALUDE_at_least_one_red_certain_l4081_408136


namespace NUMINAMATH_CALUDE_sequence_not_periodic_l4081_408119

theorem sequence_not_periodic (x : ℝ) (h1 : x > 1) (h2 : ¬ ∃ n : ℤ, x = n) : 
  ¬ ∃ p : ℕ, ∀ n : ℕ, (⌊x^(n+1)⌋ - x * ⌊x^n⌋) = (⌊x^(n+1+p)⌋ - x * ⌊x^(n+p)⌋) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_periodic_l4081_408119


namespace NUMINAMATH_CALUDE_check_cashing_mistake_l4081_408179

theorem check_cashing_mistake (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ y ∧ y ≤ 99) →
  (100 * y + x) - (100 * x + y) = 1820 →
  ∃ x y, y = x + 18 ∧ y = 2 * x :=
sorry

end NUMINAMATH_CALUDE_check_cashing_mistake_l4081_408179


namespace NUMINAMATH_CALUDE_number_equation_solution_l4081_408164

theorem number_equation_solution : 
  ∃ x : ℝ, 0.4 * x + 60 = x ∧ x = 100 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l4081_408164


namespace NUMINAMATH_CALUDE_complex_square_i_positive_l4081_408124

theorem complex_square_i_positive (a : ℝ) :
  (((a : ℂ) + Complex.I)^2 * Complex.I).re > 0 → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_square_i_positive_l4081_408124


namespace NUMINAMATH_CALUDE_total_tickets_sold_l4081_408111

/-- Represents the ticket sales scenario -/
structure TicketSales where
  student_price : ℝ
  adult_price : ℝ
  total_income : ℝ
  student_tickets : ℕ

/-- Theorem stating the total number of tickets sold -/
theorem total_tickets_sold (sale : TicketSales)
  (h1 : sale.student_price = 2)
  (h2 : sale.adult_price = 4.5)
  (h3 : sale.total_income = 60)
  (h4 : sale.student_tickets = 12) :
  ∃ (adult_tickets : ℕ), sale.student_tickets + adult_tickets = 20 :=
by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l4081_408111


namespace NUMINAMATH_CALUDE_side_ratio_not_imply_right_triangle_l4081_408144

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

/-- Definition of a right triangle --/
def IsRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

/-- The condition a:b:c = 1:2:3 --/
def SideRatio (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.a = k ∧ t.b = 2*k ∧ t.c = 3*k

/-- Theorem: The condition a:b:c = 1:2:3 does not imply a right triangle --/
theorem side_ratio_not_imply_right_triangle :
  ∃ (t : Triangle), SideRatio t ∧ ¬IsRightTriangle t :=
sorry

end NUMINAMATH_CALUDE_side_ratio_not_imply_right_triangle_l4081_408144


namespace NUMINAMATH_CALUDE_inequality_solution_l4081_408137

theorem inequality_solution (x : ℝ) : 
  (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -1) ∨ x > 5 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4081_408137


namespace NUMINAMATH_CALUDE_paint_usage_l4081_408184

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (total_used : ℝ)
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1 / 4)
  (h3 : total_used = 225) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_after_first_week := total_paint - first_week_usage
  let second_week_usage := total_used - first_week_usage
  second_week_usage / remaining_after_first_week = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l4081_408184


namespace NUMINAMATH_CALUDE_twenty_one_plates_for_8x8x8_twenty_one_plates_is_minimum_l4081_408183

/-- Represents a cubic arrangement of boxes -/
structure CubeArrangement where
  size : Nat
  total_boxes : Nat

/-- Calculates the number of plates needed to separate boxes in a cube arrangement -/
def plates_needed (arrangement : CubeArrangement) : Nat :=
  3 * (arrangement.size - 1)

/-- Theorem stating that 21 plates are needed for an 8x8x8 arrangement -/
theorem twenty_one_plates_for_8x8x8 :
  ∃ (arr : CubeArrangement), arr.size = 8 ∧ arr.total_boxes = 512 ∧ plates_needed arr = 21 :=
by
  sorry

/-- Theorem stating that 21 is the minimum number of plates needed -/
theorem twenty_one_plates_is_minimum (arr : CubeArrangement) 
  (h1 : arr.size = 8) (h2 : arr.total_boxes = 512) :
  ∀ n : Nat, n ≥ plates_needed arr → n ≥ 21 :=
by
  sorry

end NUMINAMATH_CALUDE_twenty_one_plates_for_8x8x8_twenty_one_plates_is_minimum_l4081_408183


namespace NUMINAMATH_CALUDE_inequality_range_l4081_408146

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x + 1 > 2*x + m) → m < -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l4081_408146


namespace NUMINAMATH_CALUDE_sum_of_squares_l4081_408188

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4081_408188


namespace NUMINAMATH_CALUDE_integer_fraction_theorem_l4081_408140

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧
  (∃ (k₁ k₂ : ℤ), (a^2 + b : ℤ) = k₁ * (b^2 - a) ∧ (b^2 + a : ℤ) = k₂ * (a^2 - b))

def solution_set : Set (ℕ × ℕ) :=
  {(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)}

theorem integer_fraction_theorem :
  ∀ (a b : ℕ), is_valid_pair a b ↔ (a, b) ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_integer_fraction_theorem_l4081_408140


namespace NUMINAMATH_CALUDE_elder_age_proof_l4081_408106

theorem elder_age_proof (younger elder : ℕ) : 
  (elder = younger + 16) →
  (elder - 6 = 3 * (younger - 6)) →
  elder = 30 := by sorry

end NUMINAMATH_CALUDE_elder_age_proof_l4081_408106


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l4081_408110

/-- The probability of selecting a ticket with a number that is a multiple of 3
    from a set of tickets numbered 1 to 27 is equal to 1/3. -/
theorem probability_multiple_of_three (n : ℕ) (h : n = 27) :
  (Finset.filter (fun x => x % 3 = 0) (Finset.range n)).card / n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l4081_408110


namespace NUMINAMATH_CALUDE_simplification_problems_l4081_408174

theorem simplification_problems :
  ((-1/2 + 2/3 - 1/4) / (-1/24) = 2) ∧
  (7/2 * (-5/7) - (-5/7) * 5/2 - 5/7 * (-1/2) = -5/14) := by
  sorry

end NUMINAMATH_CALUDE_simplification_problems_l4081_408174


namespace NUMINAMATH_CALUDE_park_area_calculation_l4081_408175

/-- Represents a rectangular park with cycling path -/
structure Park where
  length : ℝ
  breadth : ℝ
  avg_speed : ℝ
  time : ℝ
  downhill_speed : ℝ
  uphill_speed : ℝ

/-- Calculates the area of the park given the conditions -/
def park_area (p : Park) : ℝ :=
  p.length * p.breadth

/-- Theorem stating the area of the park under given conditions -/
theorem park_area_calculation (p : Park)
  (h1 : p.length = 3 * p.breadth)
  (h2 : p.avg_speed = 12)
  (h3 : p.time = 4 / 60)
  (h4 : p.downhill_speed = 15)
  (h5 : p.uphill_speed = 10)
  (h6 : 2 * (p.length + p.breadth) = p.avg_speed * p.time * 1000) :
  park_area p = 30000 := by
  sorry

#check park_area_calculation

end NUMINAMATH_CALUDE_park_area_calculation_l4081_408175


namespace NUMINAMATH_CALUDE_fifty_three_is_perfect_sum_x_y_is_one_k_equals_36_max_x_minus_2y_is_two_l4081_408107

-- Definition of a perfect number
def isPerfectNumber (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Statement 1
theorem fifty_three_is_perfect : isPerfectNumber 53 := by sorry

-- Statement 2
theorem sum_x_y_is_one (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 5 = 0) : 
  x + y = 1 := by sorry

-- Statement 3
theorem k_equals_36 (k : ℤ) : 
  (∀ x y : ℤ, isPerfectNumber (2*x^2 + y^2 + 2*x*y + 12*x + k)) → k = 36 := by sorry

-- Statement 4
theorem max_x_minus_2y_is_two (x y : ℝ) (h : -x^2 + (7/2)*x + y - 3 = 0) :
  x - 2*y ≤ 2 := by sorry

end NUMINAMATH_CALUDE_fifty_three_is_perfect_sum_x_y_is_one_k_equals_36_max_x_minus_2y_is_two_l4081_408107


namespace NUMINAMATH_CALUDE_cube_root_and_seventh_root_sum_l4081_408113

theorem cube_root_and_seventh_root_sum (m n : ℤ) 
  (hm : m ^ 3 = 61629875)
  (hn : n ^ 7 = 170859375) :
  100 * m + n = 39515 := by
sorry

end NUMINAMATH_CALUDE_cube_root_and_seventh_root_sum_l4081_408113


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l4081_408182

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge (haley michael brandon sofia : ℕ) : 
  haley = 8 →
  michael = 3 * haley →
  brandon = michael / 2 →
  sofia = 2 * (haley + brandon) →
  haley + michael + brandon + sofia = 84 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l4081_408182


namespace NUMINAMATH_CALUDE_inequality_proof_l4081_408130

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4081_408130


namespace NUMINAMATH_CALUDE_fib_120_mod_5_l4081_408112

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the property that the Fibonacci sequence modulo 5 repeats every 20 terms
axiom fib_mod_5_period_20 : ∀ n : ℕ, fib n % 5 = fib (n % 20) % 5

-- Theorem statement
theorem fib_120_mod_5 : fib 120 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_120_mod_5_l4081_408112


namespace NUMINAMATH_CALUDE_total_apples_in_basket_l4081_408189

def initial_apples : Nat := 8
def added_apples : Nat := 7

theorem total_apples_in_basket : initial_apples + added_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_in_basket_l4081_408189


namespace NUMINAMATH_CALUDE_square_root_of_four_l4081_408160

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l4081_408160


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l4081_408149

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l4081_408149


namespace NUMINAMATH_CALUDE_point_below_left_of_line_l4081_408166

-- Define the dice outcomes
def dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the lines
def l1 (a b : ℕ) (x y : ℝ) : Prop := a * x + b * y = 2
def l2 (x y : ℝ) : Prop := x + 2 * y = 2

-- Define the probabilities
def p1 : ℚ := 1 / 18
def p2 : ℚ := 11 / 12

-- Define the point P
def P : ℝ × ℝ := (p1, p2)

-- Theorem statement
theorem point_below_left_of_line :
  (P.1 : ℝ) + 2 * (P.2 : ℝ) < 2 := by sorry

end NUMINAMATH_CALUDE_point_below_left_of_line_l4081_408166


namespace NUMINAMATH_CALUDE_text_plan_cost_per_message_l4081_408176

/-- Represents a text messaging plan with a monthly fee and per-message cost -/
structure TextPlan where
  monthlyFee : ℝ
  costPerMessage : ℝ

/-- Calculates the total cost for a given number of messages under a plan -/
def totalCost (plan : TextPlan) (messages : ℝ) : ℝ :=
  plan.monthlyFee + plan.costPerMessage * messages

theorem text_plan_cost_per_message :
  ∃ (plan1 : TextPlan) (plan2 : TextPlan),
    plan1.monthlyFee = 9 ∧
    plan2.monthlyFee = 0 ∧
    plan2.costPerMessage = 0.4 ∧
    totalCost plan1 60 = totalCost plan2 60 ∧
    plan1.costPerMessage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_text_plan_cost_per_message_l4081_408176


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l4081_408157

/-- The number of distinct convex polygons with 4 or more sides that can be drawn
    using some or all of 15 points marked on a circle as vertices -/
def num_polygons : ℕ := 32192

/-- The total number of points marked on the circle -/
def num_points : ℕ := 15

/-- A function that calculates the number of subsets of size k from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of subsets of 15 points -/
def total_subsets : ℕ := 2^num_points

theorem distinct_polygons_count :
  num_polygons = total_subsets - (choose num_points 0 + choose num_points 1 + 
                                  choose num_points 2 + choose num_points 3) :=
sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l4081_408157


namespace NUMINAMATH_CALUDE_flower_count_l4081_408198

/-- The number of pots -/
def num_pots : ℕ := 141

/-- The number of flowers in each pot -/
def flowers_per_pot : ℕ := 71

/-- The total number of flowers -/
def total_flowers : ℕ := num_pots * flowers_per_pot

theorem flower_count : total_flowers = 10011 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l4081_408198


namespace NUMINAMATH_CALUDE_line_slope_is_two_l4081_408101

-- Define the polar equation of the line
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ - 2 * ρ * Real.cos θ + 3 = 0

-- Theorem: The slope of the line defined by the polar equation is 2
theorem line_slope_is_two :
  ∃ (m : ℝ), m = 2 ∧
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y = m * x - 3 :=
sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l4081_408101


namespace NUMINAMATH_CALUDE_total_sundaes_l4081_408135

def num_flavors : ℕ := 8

def sundae_combinations (n : ℕ) : ℕ := Nat.choose num_flavors n

theorem total_sundaes : 
  sundae_combinations 1 + sundae_combinations 2 + sundae_combinations 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_sundaes_l4081_408135


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l4081_408131

/-- 
Given a quadratic equation x^2 - mx + m - 1 = 0 with two equal real roots,
prove that m = 2 and the roots are x = 1
-/
theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - m*x + m - 1 = 0 ∧ 
   ∀ y : ℝ, y^2 - m*y + m - 1 = 0 → y = x) →
  m = 2 ∧ ∃ x : ℝ, x^2 - m*x + m - 1 = 0 ∧ x = 1 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equal_roots_l4081_408131


namespace NUMINAMATH_CALUDE_strawberry_picking_l4081_408197

theorem strawberry_picking (total strawberries_JM strawberries_Z : ℕ) 
  (h1 : total = 550)
  (h2 : strawberries_JM = 350)
  (h3 : strawberries_Z = 200) :
  total - (strawberries_JM - strawberries_Z) = 400 := by
  sorry

#check strawberry_picking

end NUMINAMATH_CALUDE_strawberry_picking_l4081_408197


namespace NUMINAMATH_CALUDE_polynomial_coefficient_l4081_408121

theorem polynomial_coefficient (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                         a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                         a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_l4081_408121


namespace NUMINAMATH_CALUDE_digit_58_is_4_l4081_408120

/-- The repeating part of the decimal representation of 1/17 -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating part -/
def repeat_length : Nat := decimal_rep_1_17.length

/-- The 58th digit after the decimal point in the decimal representation of 1/17 -/
def digit_58 : Nat :=
  decimal_rep_1_17[(58 - 1) % repeat_length]

theorem digit_58_is_4 : digit_58 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_58_is_4_l4081_408120


namespace NUMINAMATH_CALUDE_ball_max_height_l4081_408114

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 55

-- State the theorem
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 135 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l4081_408114


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l4081_408186

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 2*x + m = 0

-- Theorem statement
theorem quadratic_roots_theorem (m : ℝ) (h : m < 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m) ∧
  (quadratic_equation (-1) m → m = -3 ∧ quadratic_equation 3 m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l4081_408186


namespace NUMINAMATH_CALUDE_max_ab_bisecting_line_l4081_408133

/-- A line that bisects the circumference of a circle --/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects : ∀ (x y : ℝ), a * x + b * y - 1 = 0 → x^2 + y^2 - 4*x - 4*y - 8 = 0

/-- The maximum value of ab for a bisecting line --/
theorem max_ab_bisecting_line (l : BisectingLine) : 
  ∃ (max : ℝ), (∀ (l' : BisectingLine), l'.a * l'.b ≤ max) ∧ max = 1/16 := by
sorry

end NUMINAMATH_CALUDE_max_ab_bisecting_line_l4081_408133


namespace NUMINAMATH_CALUDE_correct_pairing_l4081_408180

structure Couple where
  wife : String
  husband : String
  wife_bottles : Nat
  husband_bottles : Nat

def total_bottles : Nat := 44

def couples : List Couple := [
  ⟨"Anna", "Smith", 2, 8⟩,
  ⟨"Betty", "White", 3, 9⟩,
  ⟨"Carol", "Green", 4, 8⟩,
  ⟨"Dorothy", "Brown", 5, 5⟩
]

theorem correct_pairing : 
  (couples.map (λ c => c.wife_bottles + c.husband_bottles)).sum = total_bottles ∧
  (∃ c ∈ couples, c.husband = "Brown" ∧ c.wife_bottles = c.husband_bottles) ∧
  (∃ c ∈ couples, c.husband = "Green" ∧ c.husband_bottles = 2 * c.wife_bottles) ∧
  (∃ c ∈ couples, c.husband = "White" ∧ c.husband_bottles = 3 * c.wife_bottles) ∧
  (∃ c ∈ couples, c.husband = "Smith" ∧ c.husband_bottles = 4 * c.wife_bottles) :=
by sorry

end NUMINAMATH_CALUDE_correct_pairing_l4081_408180


namespace NUMINAMATH_CALUDE_police_officers_on_duty_l4081_408193

theorem police_officers_on_duty 
  (total_female_officers : ℕ) 
  (female_duty_percentage : ℚ)
  (female_duty_ratio : ℚ) :
  total_female_officers = 400 →
  female_duty_percentage = 19 / 100 →
  female_duty_ratio = 1 / 2 →
  ∃ (officers_on_duty : ℕ), 
    officers_on_duty = 152 ∧ 
    (female_duty_percentage * total_female_officers : ℚ) = (female_duty_ratio * officers_on_duty : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_police_officers_on_duty_l4081_408193


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l4081_408118

/-- Determinant of a 2x2 matrix --/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Geometric sequence --/
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geometric : isGeometric a)
  (h_third : a 3 = 1)
  (h_det : det (a 6) 8 8 (a 8) = 0) :
  a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l4081_408118


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l4081_408162

theorem factorization_of_2m_squared_minus_8 (m : ℝ) :
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l4081_408162


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_101110111_base_5_l4081_408145

def base_five_to_decimal (n : ℕ) : ℕ := 
  5^8 + 5^6 + 5^5 + 5^4 + 5^3 + 5^2 + 5^1 + 5^0

theorem largest_prime_divisor_of_101110111_base_5 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ base_five_to_decimal 101110111 ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ base_five_to_decimal 101110111 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_101110111_base_5_l4081_408145


namespace NUMINAMATH_CALUDE_square_sum_equals_two_l4081_408129

theorem square_sum_equals_two (a b : ℝ) 
  (h1 : (a + b)^2 = 4) 
  (h2 : a * b = 1) : 
  a^2 + b^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_two_l4081_408129


namespace NUMINAMATH_CALUDE_pencil_distribution_l4081_408185

theorem pencil_distribution (num_students : ℕ) (total_pencils : ℕ) (pencils_per_dozen : ℕ) : 
  num_students = 46 → 
  total_pencils = 2208 → 
  pencils_per_dozen = 12 →
  (total_pencils / num_students) / pencils_per_dozen = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4081_408185


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l4081_408161

theorem triangle_angle_problem (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ 2*x < 180 ∧ 
  (x + 2*x + 30 = 180) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l4081_408161


namespace NUMINAMATH_CALUDE_marbles_given_to_mary_l4081_408143

/-- Given that Dan initially had 64 marbles and now has 50 marbles,
    prove that he gave 14 marbles to Mary. -/
theorem marbles_given_to_mary (initial_marbles : ℕ) (current_marbles : ℕ)
    (h1 : initial_marbles = 64)
    (h2 : current_marbles = 50) :
    initial_marbles - current_marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_mary_l4081_408143


namespace NUMINAMATH_CALUDE_continued_fraction_sqrt_15_l4081_408165

theorem continued_fraction_sqrt_15 (y : ℝ) : y = 3 + 5 / (2 + 5 / y) → y = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sqrt_15_l4081_408165


namespace NUMINAMATH_CALUDE_soldier_count_l4081_408141

/-- The number of soldiers in a group forming a hollow square formation -/
def number_of_soldiers (A : ℕ) : ℕ := ((A + 2 * 3) - 3) * 3 * 4 + 9

/-- The side length of the hollow square formation -/
def side_length : ℕ := 5

theorem soldier_count :
  let A := side_length
  (A - 2 * 2)^2 * 3 + 9 = number_of_soldiers A ∧
  (A - 4) * 4 * 4 + 7 = number_of_soldiers A ∧
  number_of_soldiers A = 105 := by sorry

end NUMINAMATH_CALUDE_soldier_count_l4081_408141


namespace NUMINAMATH_CALUDE_jake_weight_loss_l4081_408170

theorem jake_weight_loss (total_weight : ℕ) (jake_weight : ℕ) (weight_loss : ℕ) : 
  total_weight = 290 → 
  jake_weight = 196 → 
  jake_weight - weight_loss = 2 * (total_weight - jake_weight) → 
  weight_loss = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l4081_408170


namespace NUMINAMATH_CALUDE_a_44_mod_45_l4081_408132

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that the remainder when a_44 is divided by 45 is 9 -/
theorem a_44_mod_45 : a 44 % 45 = 9 := by sorry

end NUMINAMATH_CALUDE_a_44_mod_45_l4081_408132


namespace NUMINAMATH_CALUDE_gcf_of_3150_and_9800_l4081_408177

theorem gcf_of_3150_and_9800 : Nat.gcd 3150 9800 = 350 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_3150_and_9800_l4081_408177


namespace NUMINAMATH_CALUDE_sector_area_l4081_408138

/-- Given a circular sector with perimeter 8 cm and central angle 2 radians, its area is 4 cm² -/
theorem sector_area (r : ℝ) (l : ℝ) : 
  l + 2 * r = 8 →  -- Perimeter condition
  l = 2 * r →      -- Arc length condition (derived from central angle)
  (1 / 2) * 2 * r^2 = 4 := by  -- Area calculation
sorry

end NUMINAMATH_CALUDE_sector_area_l4081_408138


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l4081_408127

theorem parabola_fixed_point :
  ∀ t : ℝ, 3 * (2 : ℝ)^2 + t * 2 - 2 * t = 12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l4081_408127


namespace NUMINAMATH_CALUDE_taqeeshas_grade_l4081_408178

theorem taqeeshas_grade (total_students : Nat) (initial_students : Nat) (initial_average : Nat) (new_average : Nat) :
  total_students = 17 →
  initial_students = 16 →
  initial_average = 77 →
  new_average = 78 →
  (initial_students * initial_average + (total_students - initial_students) * 94) / total_students = new_average :=
by sorry

end NUMINAMATH_CALUDE_taqeeshas_grade_l4081_408178


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l4081_408196

theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) →
  (∀ x, x^2 - b*x - a < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l4081_408196


namespace NUMINAMATH_CALUDE_dinner_time_l4081_408148

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_def : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

/-- The starting time (4:00 pm) -/
def startTime : Time := ⟨16, 0, sorry⟩

/-- The total duration of tasks in minutes -/
def totalTaskDuration : ℕ := 30 + 30 + 10 + 20 + 90

/-- Theorem: Adding the total task duration to the start time results in 7:00 pm -/
theorem dinner_time : addMinutes startTime totalTaskDuration = ⟨19, 0, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_dinner_time_l4081_408148


namespace NUMINAMATH_CALUDE_complex_sum_reciprocals_l4081_408194

theorem complex_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocals_l4081_408194


namespace NUMINAMATH_CALUDE_watch_sale_price_l4081_408156

/-- The final sale price of a watch after two consecutive discounts --/
theorem watch_sale_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.20 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_watch_sale_price_l4081_408156


namespace NUMINAMATH_CALUDE_bus_trip_distance_l4081_408168

theorem bus_trip_distance (speed : ℝ) (distance : ℝ) : 
  speed = 55 →
  distance / speed - 1 = distance / (speed + 5) →
  distance = 660 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l4081_408168


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l4081_408199

theorem incorrect_observation_value (n : ℕ) (original_mean corrected_mean correct_value : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : corrected_mean = 36.5)
  (h4 : correct_value = 43) :
  ∃ x : ℝ, 
    (n : ℝ) * original_mean = (n : ℝ) * corrected_mean - correct_value + x :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l4081_408199


namespace NUMINAMATH_CALUDE_palindrome_count_is_60_l4081_408172

/-- Represents a time on a 24-hour digital clock --/
structure DigitalTime where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Checks if a given DigitalTime is a palindrome --/
def is_palindrome (t : DigitalTime) : Bool :=
  sorry

/-- Counts the number of palindromes on a 24-hour digital clock --/
def count_palindromes : Nat :=
  sorry

/-- Theorem stating that the number of palindromes on a 24-hour digital clock is 60 --/
theorem palindrome_count_is_60 : count_palindromes = 60 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_count_is_60_l4081_408172


namespace NUMINAMATH_CALUDE_curve_intersects_median_l4081_408104

/-- Given non-collinear points A, B, C in the complex plane corresponding to 
    z₀ = ai, z₁ = 1/2 + bi, z₂ = 1 + ci respectively, where a, b, c are real numbers,
    prove that the curve z = z₀cos⁴t + 2z₁cos²tsin²t + z₂sin⁴t intersects the median 
    of triangle ABC parallel to AC at exactly one point (1/2, (a+c+2b)/4). -/
theorem curve_intersects_median (a b c : ℝ) 
  (h_non_collinear : a + c - 2*b ≠ 0) : 
  ∃! p : ℂ, 
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
      2 * (1/2 + Complex.I * b) * (Real.cos t)^2 * (Real.sin t)^2 + 
      (1 + Complex.I * c) * (Real.sin t)^4) ∧ 
    p.im = (c - a) * p.re + (3*a + 2*b - c)/4 ∧ 
    p = Complex.mk (1/2) ((a + c + 2*b)/4) := by 
  sorry

end NUMINAMATH_CALUDE_curve_intersects_median_l4081_408104
