import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l4110_411029

theorem root_sum_reciprocals (a b c d : ℂ) : 
  (a^4 + 8*a^3 + 9*a^2 + 5*a + 4 = 0) →
  (b^4 + 8*b^3 + 9*b^2 + 5*b + 4 = 0) →
  (c^4 + 8*c^3 + 9*c^2 + 5*c + 4 = 0) →
  (d^4 + 8*d^3 + 9*d^2 + 5*d + 4 = 0) →
  (1/(a*b) + 1/(a*c) + 1/(a*d) + 1/(b*c) + 1/(b*d) + 1/(c*d) = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l4110_411029


namespace NUMINAMATH_CALUDE_a_bounds_circle_D_equation_l4110_411067

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line L
def line_L (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the function a
def a (x y : ℝ) : ℝ := y - x

-- Theorem for the maximum and minimum values of a on circle C
theorem a_bounds :
  (∃ x y : ℝ, circle_C x y ∧ a x y = 2 * Real.sqrt 2 + 1) ∧
  (∃ x y : ℝ, circle_C x y ∧ a x y = 1 - 2 * Real.sqrt 2) ∧
  (∀ x y : ℝ, circle_C x y → 1 - 2 * Real.sqrt 2 ≤ a x y ∧ a x y ≤ 2 * Real.sqrt 2 + 1) :=
sorry

-- Define circle D
def circle_D (center_x center_y x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 9

-- Theorem for the equation of circle D
theorem circle_D_equation :
  ∃ center_x center_y : ℝ,
    line_L center_x center_y ∧
    ((∀ x y : ℝ, circle_D center_x center_y x y ↔ (x - 3)^2 + (y + 1)^2 = 9) ∨
     (∀ x y : ℝ, circle_D center_x center_y x y ↔ (x + 2)^2 + (y - 4)^2 = 9)) ∧
    (∃ x y : ℝ, circle_C x y ∧ (x - center_x)^2 + (y - center_y)^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_a_bounds_circle_D_equation_l4110_411067


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l4110_411044

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 :
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l4110_411044


namespace NUMINAMATH_CALUDE_factor_calculation_l4110_411002

theorem factor_calculation (f : ℝ) : f * (2 * 20 + 5) = 135 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l4110_411002


namespace NUMINAMATH_CALUDE_derivative_reciprocal_l4110_411095

theorem derivative_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv (fun x => 1 / x) x = -(1 / x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_reciprocal_l4110_411095


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l4110_411008

/-- The equation of the directrix of a parabola given specific conditions -/
theorem parabola_directrix_equation (M : ℝ × ℝ) (p : ℝ) :
  M = (-1, 2) →
  p > 0 →
  (∃ (x y : ℝ), x^2 = 2*p*y) →
  (∃ (x y : ℝ), 2*x - 4*y + 5 = 0 ∧ x = p/2 ∧ y = 0) →
  (∃ (y : ℝ), ∀ (x : ℝ), y = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l4110_411008


namespace NUMINAMATH_CALUDE_equal_squares_exist_l4110_411092

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 10
  col : Fin 10

/-- Represents a square in the grid -/
structure Square where
  cell : Cell
  size : ℕ

/-- The theorem to be proved -/
theorem equal_squares_exist (squares : Finset Square) 
  (h1 : squares.card = 9)
  (h2 : ∀ s ∈ squares, s.cell.row < 10 ∧ s.cell.col < 10) :
  ∃ s1 s2 : Square, s1 ∈ squares ∧ s2 ∈ squares ∧ s1 ≠ s2 ∧ s1.size = s2.size :=
sorry

end NUMINAMATH_CALUDE_equal_squares_exist_l4110_411092


namespace NUMINAMATH_CALUDE_fifty_third_odd_positive_integer_l4110_411048

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem: The 53rd odd positive integer is 105 -/
theorem fifty_third_odd_positive_integer : nthOddPositiveInteger 53 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_odd_positive_integer_l4110_411048


namespace NUMINAMATH_CALUDE_spinner_direction_l4110_411025

-- Define the directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation
def rotate (initial : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction (initial : Direction) 
  (clockwise : ℚ) (counterclockwise : ℚ) :
  initial = Direction.North ∧ 
  clockwise = 7/2 ∧ 
  counterclockwise = 17/4 →
  rotate (rotate initial clockwise) (-counterclockwise) = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_direction_l4110_411025


namespace NUMINAMATH_CALUDE_height_area_ratio_not_always_equal_l4110_411086

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  side : ℝ
  perimeter : ℝ
  area : ℝ

/-- The theorem states that the ratio of heights is not always equal to the ratio of areas for two isosceles triangles with different heights -/
theorem height_area_ratio_not_always_equal (t1 t2 : IsoscelesTriangle) 
  (h_diff : t1.height ≠ t2.height) :
  ¬ ∀ (t1 t2 : IsoscelesTriangle), t1.height / t2.height = t1.area / t2.area :=
by sorry

end NUMINAMATH_CALUDE_height_area_ratio_not_always_equal_l4110_411086


namespace NUMINAMATH_CALUDE_circle_parallel_lines_distance_l4110_411021

-- Define the circle
variable (r : ℝ) -- radius of the circle

-- Define the chords
def chord1 : ℝ := 45
def chord2 : ℝ := 49
def chord3 : ℝ := 49
def chord4 : ℝ := 45

-- Define the distance between adjacent parallel lines
def d : ℝ := 2.8

-- State the theorem
theorem circle_parallel_lines_distance :
  ∃ (r : ℝ), 
    r > 0 ∧
    chord1 = 45 ∧
    chord2 = 49 ∧
    chord3 = 49 ∧
    chord4 = 45 ∧
    d = 2.8 ∧
    r^2 = 506.25 + (1/4) * d^2 ∧
    r^2 = 600.25 + (49/4) * d^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_parallel_lines_distance_l4110_411021


namespace NUMINAMATH_CALUDE_congruence_problem_l4110_411080

theorem congruence_problem (x : ℤ) :
  (4 * x + 9) % 20 = 3 → (3 * x + 15) % 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4110_411080


namespace NUMINAMATH_CALUDE_log_101600_equals_2x_l4110_411063

theorem log_101600_equals_2x (x : ℝ) (h : Real.log 102 = x) : Real.log 101600 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_log_101600_equals_2x_l4110_411063


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4110_411098

theorem absolute_value_inequality (x : ℝ) : 
  |x + 1| - |x - 4| > 3 ↔ x > 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4110_411098


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_eight_l4110_411074

theorem negative_five_greater_than_negative_eight :
  -5 > -8 :=
by sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_eight_l4110_411074


namespace NUMINAMATH_CALUDE_cell_count_after_3_hours_l4110_411022

/-- The number of cells after a given number of half-hour intervals, starting with one cell -/
def cell_count (n : ℕ) : ℕ := 2^n

/-- The number of half-hour intervals in 3 hours -/
def intervals_in_3_hours : ℕ := 6

theorem cell_count_after_3_hours :
  cell_count intervals_in_3_hours = 64 := by
  sorry

end NUMINAMATH_CALUDE_cell_count_after_3_hours_l4110_411022


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4110_411096

/-- The lateral surface area of a cylinder, given the diagonal length and intersection angle of its rectangular lateral surface. -/
theorem cylinder_lateral_surface_area 
  (d : ℝ) 
  (α : ℝ) 
  (h_d_pos : d > 0) 
  (h_α_pos : α > 0) 
  (h_α_lt_pi : α < π) : 
  ∃ (S : ℝ), S = (1/2) * d^2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4110_411096


namespace NUMINAMATH_CALUDE_property_P_theorems_seq_012_has_property_P_l4110_411050

/-- Definition of a sequence with property P -/
def has_property_P (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 3 ∧
  (∀ i j, 1 ≤ i → i ≤ j → j ≤ k → (∃ n ≤ k, a n = a j + a i ∨ a n = a j - a i)) ∧
  (∀ i, 1 ≤ i → i < k → a i < a (i + 1)) ∧
  0 ≤ a 1

theorem property_P_theorems (a : ℕ → ℝ) (k : ℕ) (h : has_property_P a k) :
  (∀ i ≤ k, a k - a i ∈ Set.range (fun n => a n)) ∧
  (k ≥ 5 → ∃ d : ℝ, ∀ i < k, a (i + 1) - a i = d) :=
by sorry

/-- The sequence 0, 1, 2 has property P -/
theorem seq_012_has_property_P :
  has_property_P (fun n => if n = 1 then 0 else if n = 2 then 1 else 2) 3 :=
by sorry

end NUMINAMATH_CALUDE_property_P_theorems_seq_012_has_property_P_l4110_411050


namespace NUMINAMATH_CALUDE_hash_six_two_l4110_411073

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + a / b

-- Theorem statement
theorem hash_six_two : hash 6 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hash_six_two_l4110_411073


namespace NUMINAMATH_CALUDE_geometric_concept_word_counts_l4110_411014

/-- A type representing geometric concepts -/
def GeometricConcept : Type := String

/-- A function that counts the number of words in a string -/
def wordCount (s : String) : Nat :=
  s.split (· == ' ') |>.length

/-- Theorem stating that there exist geometric concepts expressible in 1, 2, 3, and 4 words -/
theorem geometric_concept_word_counts :
  ∃ (a b c d : GeometricConcept),
    wordCount a = 1 ∧
    wordCount b = 2 ∧
    wordCount c = 3 ∧
    wordCount d = 4 :=
by sorry


end NUMINAMATH_CALUDE_geometric_concept_word_counts_l4110_411014


namespace NUMINAMATH_CALUDE_square_semicircle_perimeter_l4110_411043

theorem square_semicircle_perimeter (π : Real) (h : π > 0) : 
  let square_side : Real := 4 / π
  let semicircle_radius : Real := square_side / 2
  let num_semicircles : Nat := 4
  num_semicircles * (π * semicircle_radius) = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_semicircle_perimeter_l4110_411043


namespace NUMINAMATH_CALUDE_f_is_even_l4110_411036

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^4

theorem f_is_even : is_even_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l4110_411036


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l4110_411015

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by
  sorry

theorem negation_of_exponential_equation : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) ↔ (∀ x : ℝ, Real.exp x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l4110_411015


namespace NUMINAMATH_CALUDE_odd_function_value_l4110_411081

theorem odd_function_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^(2-m)
  (∀ x ∈ Set.Icc (-3-m) (m^2-m), f (-x) = -f x) →
  f m = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l4110_411081


namespace NUMINAMATH_CALUDE_cafe_order_combinations_l4110_411065

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- The number of distinct meal combinations for two people choosing from a menu with a given number of items, where order matters and repetition is allowed -/
def meal_combinations (items : ℕ) : ℕ := items ^ num_people

theorem cafe_order_combinations :
  meal_combinations menu_items = 225 := by
  sorry

end NUMINAMATH_CALUDE_cafe_order_combinations_l4110_411065


namespace NUMINAMATH_CALUDE_no_solution_iff_a_less_than_one_l4110_411087

theorem no_solution_iff_a_less_than_one (a : ℝ) :
  (∀ x : ℝ, |x - 1| + x > a) ↔ a < 1 := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_less_than_one_l4110_411087


namespace NUMINAMATH_CALUDE_problem_solution_l4110_411069

theorem problem_solution (a b c : ℝ) (h1 : |a| = 2) (h2 : a < 1) (h3 : b * c = 1) :
  a^3 + 3 - 4*b*c = -9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4110_411069


namespace NUMINAMATH_CALUDE_prob_not_six_l4110_411019

/-- Given a die where the odds of rolling a six are 1:5, 
    the probability of not rolling a six is 5/6 -/
theorem prob_not_six (favorable : ℕ) (unfavorable : ℕ) :
  favorable = 1 →
  unfavorable = 5 →
  (unfavorable : ℚ) / (favorable + unfavorable : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_six_l4110_411019


namespace NUMINAMATH_CALUDE_book_pages_calculation_l4110_411094

/-- The number of pages Steve reads per day -/
def pages_per_day : ℕ := 100

/-- The number of days per week Steve reads -/
def reading_days_per_week : ℕ := 3

/-- The number of weeks Steve takes to read the book -/
def total_weeks : ℕ := 7

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * reading_days_per_week * total_weeks

theorem book_pages_calculation :
  total_pages = 2100 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l4110_411094


namespace NUMINAMATH_CALUDE_calculation_proof_l4110_411083

theorem calculation_proof :
  ((-5/6 + 2/3) / (-7/12) * (7/2) = 1) ∧
  ((1 - 1/6) * (-3) - (-11/6) / (-22/3) = -11/4) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4110_411083


namespace NUMINAMATH_CALUDE_workers_wage_increase_l4110_411071

theorem workers_wage_increase (original_wage new_wage : ℝ) : 
  (original_wage * 1.5 = new_wage) → 
  (new_wage = 51) → 
  (original_wage = 34) := by
sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l4110_411071


namespace NUMINAMATH_CALUDE_weight_of_b_l4110_411099

/-- Given the average weights of different combinations of people a, b, c, and d,
    prove that the weight of b is 31 kg. -/
theorem weight_of_b (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 48 →
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  (c + d) / 2 = 46 →
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l4110_411099


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l4110_411058

theorem absolute_value_simplification (x : ℝ) (h : x < -1) :
  |x - 2 * Real.sqrt ((x + 1)^2)| = -3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l4110_411058


namespace NUMINAMATH_CALUDE_ball_passing_game_l4110_411062

/-- Probability of the ball returning to player A after n passes in a three-player game --/
def P (n : ℕ) : ℚ :=
  1/3 - 1/3 * (-1/2)^(n-1)

theorem ball_passing_game :
  (P 2 = 1/2) ∧
  (∀ n : ℕ, P (n+1) = 1/2 * (1 - P n)) ∧
  (∀ n : ℕ, P n = 1/3 - 1/3 * (-1/2)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_ball_passing_game_l4110_411062


namespace NUMINAMATH_CALUDE_xyz_value_l4110_411049

-- Define the complex numbers x, y, and z
variable (x y z : ℂ)

-- Define the conditions
def condition1 : Prop := x * y + 5 * y = -20
def condition2 : Prop := y * z + 5 * z = -20
def condition3 : Prop := z * x + 5 * x = -20
def condition4 : Prop := x + y + z = 3

-- Theorem statement
theorem xyz_value (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) (h4 : condition4 x y z) :
  x * y * z = 105 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l4110_411049


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4110_411027

theorem imaginary_part_of_z (z : ℂ) (h : (Complex.I - 1) * z = Complex.I) : 
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4110_411027


namespace NUMINAMATH_CALUDE_binary_to_base5_conversion_l4110_411059

-- Define the binary number 1101₂
def binary_num : ℕ := 13

-- Define the base-5 number 23₅
def base5_num : ℕ := 2 * 5 + 3

-- Theorem stating the equality of the two representations
theorem binary_to_base5_conversion :
  binary_num = base5_num := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base5_conversion_l4110_411059


namespace NUMINAMATH_CALUDE_derived_sequence_general_term_l4110_411005

/-- An arithmetic sequence {a_n} with specific terms and a derived sequence {b_n} -/
def arithmetic_and_derived_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) ∧  -- arithmetic sequence condition
  (a 2 = 8) ∧
  (a 8 = 26) ∧
  (∀ n : ℕ, b n = a (3^n))  -- definition of b_n

/-- The general term of the derived sequence b_n -/
def b_general_term (n : ℕ) : ℝ := 3 * 3^n + 2

/-- Theorem stating that b_n equals the derived general term -/
theorem derived_sequence_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_and_derived_sequence a b →
  ∀ n : ℕ, b n = b_general_term n :=
by
  sorry

end NUMINAMATH_CALUDE_derived_sequence_general_term_l4110_411005


namespace NUMINAMATH_CALUDE_first_job_wages_proof_l4110_411010

/-- Calculates the amount received from the first job given total wages and second job details. -/
def first_job_wages (total_wages : ℕ) (second_job_hours : ℕ) (second_job_rate : ℕ) : ℕ :=
  total_wages - second_job_hours * second_job_rate

/-- Proves that given the specified conditions, the amount received from the first job is $52. -/
theorem first_job_wages_proof :
  first_job_wages 160 12 9 = 52 := by
  sorry

end NUMINAMATH_CALUDE_first_job_wages_proof_l4110_411010


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l4110_411070

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x - 1 = 0

-- Define the completed square form
def completed_square (x : ℝ) : Prop := (x - 2)^2 = 5

-- Theorem stating that the completed square form is equivalent to the original equation
theorem completing_square_equivalence :
  ∀ x : ℝ, original_equation x ↔ completed_square x :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l4110_411070


namespace NUMINAMATH_CALUDE_victor_percentage_l4110_411051

def max_marks : ℕ := 500
def victor_marks : ℕ := 460

theorem victor_percentage : 
  (victor_marks : ℚ) / max_marks * 100 = 92 := by sorry

end NUMINAMATH_CALUDE_victor_percentage_l4110_411051


namespace NUMINAMATH_CALUDE_company_research_development_l4110_411009

/-- Success probability of Team A -/
def p_a : ℚ := 2/3

/-- Success probability of Team B -/
def p_b : ℚ := 3/5

/-- Profit from successful development of product A (in thousands of dollars) -/
def profit_a : ℕ := 120

/-- Profit from successful development of product B (in thousands of dollars) -/
def profit_b : ℕ := 100

/-- The probability of at least one new product being successfully developed -/
def prob_at_least_one : ℚ := 1 - (1 - p_a) * (1 - p_b)

/-- The expected profit of the company (in thousands of dollars) -/
def expected_profit : ℚ := 
  0 * (1 - p_a) * (1 - p_b) + 
  profit_a * p_a * (1 - p_b) + 
  profit_b * (1 - p_a) * p_b + 
  (profit_a + profit_b) * p_a * p_b

theorem company_research_development :
  (prob_at_least_one = 13/15) ∧ (expected_profit = 140) := by
  sorry

end NUMINAMATH_CALUDE_company_research_development_l4110_411009


namespace NUMINAMATH_CALUDE_robert_birth_year_l4110_411066

def first_amc8_year : ℕ := 1985

def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

def robert_age_at_tenth_amc8 : ℕ := 15

theorem robert_birth_year :
  ∃ (birth_year : ℕ),
    birth_year = amc8_year 10 - robert_age_at_tenth_amc8 ∧
    birth_year = 1979 :=
by sorry

end NUMINAMATH_CALUDE_robert_birth_year_l4110_411066


namespace NUMINAMATH_CALUDE_workshop_average_salary_l4110_411000

def total_workers : ℕ := 7
def technicians_salary : ℕ := 8000
def rest_salary : ℕ := 6000

theorem workshop_average_salary :
  (total_workers * technicians_salary) / total_workers = technicians_salary :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l4110_411000


namespace NUMINAMATH_CALUDE_hexagon_longest_side_range_l4110_411053

/-- Given a hexagon formed by wrapping a line segment of length 20,
    the length of its longest side is between 10/3 and 10 (exclusive). -/
theorem hexagon_longest_side_range :
  ∀ x : ℝ,
    (∃ a b c d e f : ℝ,
      a + b + c + d + e + f = 20 ∧
      x = max a (max b (max c (max d (max e f)))) ∧
      a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0) →
    (10 / 3 : ℝ) ≤ x ∧ x < 10 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_longest_side_range_l4110_411053


namespace NUMINAMATH_CALUDE_triangle_folding_angle_range_l4110_411024

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  valid : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def pointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define perpendicularity of two line segments
def perpendicular (AB CD : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem triangle_folding_angle_range 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_angle_C : angle (B - C) (A - C) = π / 3) 
  (θ : ℝ) 
  (h_angle_BAC : angle (C - A) (B - A) = θ) :
  (∃ M : ℝ × ℝ, 
    pointOnSegment M B C ∧ 
    (∃ B' : ℝ × ℝ, perpendicular (A, B') (C, M))) →
  π / 6 < θ ∧ θ < 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_folding_angle_range_l4110_411024


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l4110_411045

theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) 
  (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - jogger_speed
  let distance_traveled := relative_speed * passing_time
  let train_length := distance_traveled - initial_distance
  train_length

theorem train_length_proof :
  train_length_calculation 2.5 12.5 240 36 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l4110_411045


namespace NUMINAMATH_CALUDE_gcd_eight_factorial_seven_factorial_l4110_411016

theorem gcd_eight_factorial_seven_factorial :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 7) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_eight_factorial_seven_factorial_l4110_411016


namespace NUMINAMATH_CALUDE_max_value_of_d_l4110_411033

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (prod_sum_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ 5 + (5 * Real.sqrt 34) / 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_d_l4110_411033


namespace NUMINAMATH_CALUDE_selections_with_former_eq_2850_l4110_411060

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of coordinators to be selected -/
def num_coordinators : ℕ := 4

/-- The total number of members -/
def total_members : ℕ := 18

/-- The number of former coordinators -/
def former_coordinators : ℕ := 8

/-- The number of selections including at least one former coordinator -/
def selections_with_former : ℕ :=
  choose total_members num_coordinators - choose (total_members - former_coordinators) num_coordinators

theorem selections_with_former_eq_2850 : selections_with_former = 2850 := by sorry

end NUMINAMATH_CALUDE_selections_with_former_eq_2850_l4110_411060


namespace NUMINAMATH_CALUDE_rental_ratio_l4110_411011

def comedies_rented : ℕ := 15
def action_movies_rented : ℕ := 5

theorem rental_ratio : 
  (comedies_rented : ℚ) / (action_movies_rented : ℚ) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_rental_ratio_l4110_411011


namespace NUMINAMATH_CALUDE_curve_equation_represents_quadrants_l4110_411088

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the right quadrant of the circle
def right_quadrant (x y : ℝ) : Prop := x = Real.sqrt (1 - y^2) ∧ x ≥ 0

-- Define the lower quadrant of the circle
def lower_quadrant (x y : ℝ) : Prop := y = -Real.sqrt (1 - x^2) ∧ y ≤ 0

-- Theorem stating the equation represents the right and lower quadrants of the unit circle
theorem curve_equation_represents_quadrants :
  ∀ x y : ℝ, unit_circle x y →
  ((x - Real.sqrt (1 - y^2)) * (y + Real.sqrt (1 - x^2)) = 0) ↔
  (right_quadrant x y ∨ lower_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_curve_equation_represents_quadrants_l4110_411088


namespace NUMINAMATH_CALUDE_front_wheel_perimeter_front_wheel_perimeter_is_30_l4110_411038

/-- The perimeter of the front wheel of a bicycle, given the perimeter of the back wheel
    and the number of revolutions each wheel makes to cover the same distance. -/
theorem front_wheel_perimeter (back_wheel_perimeter : ℝ) 
    (front_wheel_revolutions : ℝ) (back_wheel_revolutions : ℝ) : ℝ :=
  let front_wheel_perimeter := (back_wheel_perimeter * back_wheel_revolutions) / front_wheel_revolutions
  have back_wheel_perimeter_eq : back_wheel_perimeter = 20 := by sorry
  have front_wheel_revolutions_eq : front_wheel_revolutions = 240 := by sorry
  have back_wheel_revolutions_eq : back_wheel_revolutions = 360 := by sorry
  have equal_distance : front_wheel_perimeter * front_wheel_revolutions = 
                        back_wheel_perimeter * back_wheel_revolutions := by sorry
  30

theorem front_wheel_perimeter_is_30 : front_wheel_perimeter 20 240 360 = 30 := by sorry

end NUMINAMATH_CALUDE_front_wheel_perimeter_front_wheel_perimeter_is_30_l4110_411038


namespace NUMINAMATH_CALUDE_diamond_ruby_difference_l4110_411041

theorem diamond_ruby_difference (d r : ℕ) (h1 : d = 3 * r) : d - r = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_diamond_ruby_difference_l4110_411041


namespace NUMINAMATH_CALUDE_arithmetic_problem_l4110_411023

theorem arithmetic_problem : 300 + 5 * 8 = 340 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l4110_411023


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_25_seconds_l4110_411046

/-- Time taken for a train to pass a jogger -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (train_length initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time taken for the train to pass the jogger is 25 seconds -/
theorem train_passes_jogger_in_25_seconds : 
  train_passing_jogger_time 9 45 100 150 = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_25_seconds_l4110_411046


namespace NUMINAMATH_CALUDE_circle_area_vs_circumference_probability_l4110_411031

theorem circle_area_vs_circumference_probability : 
  let die_roll : Set ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Set ℕ := {n ∈ die_roll | n > 1}
  let probability := (Finset.card favorable_outcomes.toFinset) / (Finset.card die_roll.toFinset)
  let area (r : ℝ) := π * r^2
  let circumference (r : ℝ) := 2 * π * r
  (∀ r ∈ die_roll, area r > (1/2) * circumference r ↔ r > 1) →
  probability = 5/6 := by
sorry

end NUMINAMATH_CALUDE_circle_area_vs_circumference_probability_l4110_411031


namespace NUMINAMATH_CALUDE_function_decomposition_even_odd_l4110_411030

theorem function_decomposition_even_odd (f : ℝ → ℝ) :
  ∃! (f₀ f₁ : ℝ → ℝ),
    (∀ x, f x = f₀ x + f₁ x) ∧
    (∀ x, f₀ (-x) = f₀ x) ∧
    (∀ x, f₁ (-x) = -f₁ x) ∧
    (∀ x, f₀ x = (1/2) * (f x + f (-x))) ∧
    (∀ x, f₁ x = (1/2) * (f x - f (-x))) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_even_odd_l4110_411030


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_zeros_l4110_411076

/-- The quadratic function y = x^2 + 2mx + (m + 2) has two distinct zeros if and only if m ∈ (-∞, -1) ∪ (2, +∞) -/
theorem quadratic_two_distinct_zeros (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*m*x₁ + (m + 2) = 0 ∧ x₂^2 + 2*m*x₂ + (m + 2) = 0) ↔
  (m < -1 ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_zeros_l4110_411076


namespace NUMINAMATH_CALUDE_power_ranger_stickers_l4110_411026

theorem power_ranger_stickers (total : ℕ) (difference : ℕ) (first_box : ℕ) : 
  total = 58 → difference = 12 → first_box + (first_box + difference) = total → first_box = 23 := by
  sorry

end NUMINAMATH_CALUDE_power_ranger_stickers_l4110_411026


namespace NUMINAMATH_CALUDE_sum_of_squares_power_l4110_411055

theorem sum_of_squares_power (a b n : ℕ+) : ∃ x y : ℤ, (a.val ^ 2 + b.val ^ 2) ^ n.val = x ^ 2 + y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_power_l4110_411055


namespace NUMINAMATH_CALUDE_ship_distance_theorem_l4110_411034

/-- Represents the ship's position relative to Island X -/
structure ShipPosition where
  angle : ℝ  -- angle in radians for circular motion
  distance : ℝ -- distance from Island X

/-- Represents the ship's path -/
inductive ShipPath
  | Circle (t : ℝ) -- t represents time spent on circular path
  | StraightLine (t : ℝ) -- t represents time spent on straight line

/-- Function to calculate the ship's distance from Island X -/
def shipDistance (r : ℝ) (path : ShipPath) : ℝ :=
  match path with
  | ShipPath.Circle _ => r
  | ShipPath.StraightLine t => r + t

theorem ship_distance_theorem (r : ℝ) (h : r > 0) :
  ∃ (t₁ t₂ : ℝ), t₁ > 0 ∧ t₂ > 0 ∧
    (∀ t, 0 ≤ t ∧ t ≤ t₁ → shipDistance r (ShipPath.Circle t) = r) ∧
    (∀ t, t > t₁ ∧ t ≤ t₁ + t₂ → shipDistance r (ShipPath.StraightLine (t - t₁)) > r ∧
      (shipDistance r (ShipPath.StraightLine (t - t₁)) - r) = t - t₁) :=
  sorry

end NUMINAMATH_CALUDE_ship_distance_theorem_l4110_411034


namespace NUMINAMATH_CALUDE_min_entries_to_four_coins_l4110_411091

/-- Represents the state of coins and last entry -/
structure CoinState :=
  (coins : ℕ)
  (lastEntry : ℕ)

/-- Defines the coin machine rules -/
def coinMachine (entry : ℕ) : ℕ :=
  match entry with
  | 7 => 3
  | 8 => 11
  | 9 => 4
  | _ => 0

/-- Checks if an entry is valid -/
def isValidEntry (state : CoinState) (entry : ℕ) : Bool :=
  state.coins ≥ entry ∧ entry ≠ state.lastEntry ∧ (entry = 7 ∨ entry = 8 ∨ entry = 9)

/-- Makes an entry and returns the new state -/
def makeEntry (state : CoinState) (entry : ℕ) : CoinState :=
  { coins := state.coins - entry + coinMachine entry,
    lastEntry := entry }

/-- Defines the minimum number of entries to reach the target -/
def minEntries (start : ℕ) (target : ℕ) : ℕ := sorry

/-- Theorem stating the minimum number of entries to reach 4 coins from 15 coins is 4 -/
theorem min_entries_to_four_coins :
  minEntries 15 4 = 4 := by sorry

end NUMINAMATH_CALUDE_min_entries_to_four_coins_l4110_411091


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l4110_411017

theorem quadratic_roots_product (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 1 →
  x₁ * x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l4110_411017


namespace NUMINAMATH_CALUDE_shirt_price_l4110_411037

/-- Given a shirt and sweater with a total cost of $80.34, where the shirt costs $7.43 less than the sweater, the price of the shirt is $36.455. -/
theorem shirt_price (total_cost sweater_price shirt_price : ℝ) : 
  total_cost = 80.34 →
  sweater_price = shirt_price + 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l4110_411037


namespace NUMINAMATH_CALUDE_sin_decreasing_omega_range_l4110_411057

theorem sin_decreasing_omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Icc (π / 4) (π / 2), 
    ∀ y ∈ Set.Icc (π / 4) (π / 2), 
    x < y → Real.sin (ω * x) > Real.sin (ω * y)) →
  ω ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_decreasing_omega_range_l4110_411057


namespace NUMINAMATH_CALUDE_skateboard_cost_l4110_411064

theorem skateboard_cost (total_toys : ℝ) (toy_cars : ℝ) (toy_trucks : ℝ) 
  (h1 : total_toys = 25.62)
  (h2 : toy_cars = 14.88)
  (h3 : toy_trucks = 5.86) :
  total_toys - toy_cars - toy_trucks = 4.88 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_cost_l4110_411064


namespace NUMINAMATH_CALUDE_f_derivative_f_extrema_log_inequality_l4110_411090

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

-- State the theorems
theorem f_derivative (a : ℝ) (x : ℝ) (h : x ≠ 0) :
  deriv (f a) x = (a * x - 1) / (a * x^2) :=
sorry

theorem f_extrema (e : ℝ) (h_e : e > 0) :
  let f_1 := f 1
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc (1/e) e, f_1 x ≤ max_val) ∧
    (∃ x ∈ Set.Icc (1/e) e, f_1 x = max_val) ∧
    (∀ x ∈ Set.Icc (1/e) e, f_1 x ≥ min_val) ∧
    (∃ x ∈ Set.Icc (1/e) e, f_1 x = min_val) ∧
    max_val = e - 2 ∧ min_val = 0 :=
sorry

theorem log_inequality (n : ℕ) (h : n > 1) :
  Real.log (n / (n - 1)) > 1 / n :=
sorry

end NUMINAMATH_CALUDE_f_derivative_f_extrema_log_inequality_l4110_411090


namespace NUMINAMATH_CALUDE_cuboid_volume_l4110_411072

/-- 
Theorem: Volume of a specific cuboid

Given a cuboid with the following properties:
1. The length and width are equal.
2. The length is 2 cm more than the height.
3. When the height is increased by 2 cm (making it equal to the length and width),
   the surface area increases by 56 square centimeters.

This theorem proves that the volume of the original cuboid is 245 cubic centimeters.
-/
theorem cuboid_volume (l w h : ℝ) : 
  l = w → -- length equals width
  l = h + 2 → -- length is 2 more than height
  6 * l^2 - 2 * (l^2 - (l-2)^2) = 56 → -- surface area increase condition
  l * w * h = 245 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l4110_411072


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_equals_seven_l4110_411075

/-- Given vectors a, b, c in ℝ², prove that if a - c is perpendicular to b, then m = 7 -/
theorem perpendicular_vectors_imply_m_equals_seven
  (m : ℝ)
  (a b c : ℝ × ℝ)
  (ha : a = (3, -2*m))
  (hb : b = (m - 1, 2))
  (hc : c = (-2, 1))
  (h_perp : (a.1 - c.1) * b.1 + (a.2 - c.2) * b.2 = 0) :
  m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_equals_seven_l4110_411075


namespace NUMINAMATH_CALUDE_a_range_l4110_411084

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0

theorem a_range (a : ℝ) (h : p a ∧ q a) : a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l4110_411084


namespace NUMINAMATH_CALUDE_skirt_cost_l4110_411040

/-- Calculates the cost of each skirt in Marcia's wardrobe purchase --/
theorem skirt_cost (num_skirts num_pants num_blouses : ℕ)
  (blouse_price pant_price total_spend : ℚ) :
  num_skirts = 3 →
  num_pants = 2 →
  num_blouses = 5 →
  blouse_price = 15 →
  pant_price = 30 →
  total_spend = 180 →
  (total_spend - (num_blouses * blouse_price + pant_price * 1.5)) / num_skirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_skirt_cost_l4110_411040


namespace NUMINAMATH_CALUDE_combined_height_is_twelve_l4110_411082

/-- The height of Chiquita in feet -/
def chiquita_height : ℝ := 5

/-- The height difference between Mr. Martinez and Chiquita in feet -/
def height_difference : ℝ := 2

/-- The height of Mr. Martinez in feet -/
def martinez_height : ℝ := chiquita_height + height_difference

/-- The combined height of Mr. Martinez and Chiquita in feet -/
def combined_height : ℝ := chiquita_height + martinez_height

theorem combined_height_is_twelve : combined_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_combined_height_is_twelve_l4110_411082


namespace NUMINAMATH_CALUDE_binomial_sum_theorem_l4110_411052

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the left-hand side of the first equation
def lhs1 (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the right-hand side of the first equation
def rhs1 (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the left-hand side of the second equation
def lhs2 (n : ℕ) : ℕ := sorry

-- Define the right-hand side of the second equation
def rhs2 (n : ℕ) : ℕ := sorry

-- State the theorem
theorem binomial_sum_theorem (n : ℕ) (hn : n ≥ 1) :
  (∀ x : ℝ, lhs1 n x = rhs1 n x) ∧ (lhs2 n = rhs2 n) := by sorry

end NUMINAMATH_CALUDE_binomial_sum_theorem_l4110_411052


namespace NUMINAMATH_CALUDE_max_d_is_15_l4110_411018

/-- Represents a 6-digit number of the form x5d,33e -/
structure SixDigitNumber where
  x : Nat
  d : Nat
  e : Nat
  h_x : x < 10
  h_d : d < 10
  h_e : e < 10

/-- Checks if a SixDigitNumber is divisible by 33 -/
def isDivisibleBy33 (n : SixDigitNumber) : Prop :=
  (n.x + n.d + n.e + 11) % 3 = 0 ∧ (n.x + n.d - n.e - 5) % 11 = 0

/-- The maximum value of d in a SixDigitNumber divisible by 33 is 15 -/
theorem max_d_is_15 : 
  ∀ n : SixDigitNumber, isDivisibleBy33 n → n.d ≤ 15 ∧ 
  ∃ m : SixDigitNumber, isDivisibleBy33 m ∧ m.d = 15 := by sorry

end NUMINAMATH_CALUDE_max_d_is_15_l4110_411018


namespace NUMINAMATH_CALUDE_order_of_fractions_l4110_411020

theorem order_of_fractions (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hab : a > b) : 
  b / a < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < a / b := by
  sorry

end NUMINAMATH_CALUDE_order_of_fractions_l4110_411020


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l4110_411004

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l4110_411004


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_ab_range_when_c_is_1_l4110_411079

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < Real.pi / 2
  acute_B : 0 < B ∧ B < Real.pi / 2
  acute_C : 0 < C ∧ C < Real.pi / 2
  tan_C_eq : Real.tan C = (a * b) / (a^2 + b^2 - c^2)

/-- Theorem stating that if tan C = (ab) / (a² + b² - c²) in an acute triangle, then C = 30° -/
theorem angle_C_is_30_degrees (t : AcuteTriangle) : t.C = Real.pi / 6 := by
  sorry

/-- Theorem stating that if c = 1 and tan C = (ab) / (a² + b² - 1) in an acute triangle, 
    then 2√3 < ab ≤ 2 + √3 -/
theorem ab_range_when_c_is_1 (t : AcuteTriangle) (h : t.c = 1) : 
  2 * Real.sqrt 3 < t.a * t.b ∧ t.a * t.b ≤ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_ab_range_when_c_is_1_l4110_411079


namespace NUMINAMATH_CALUDE_three_digit_rounding_l4110_411068

theorem three_digit_rounding (A : ℕ) : 
  (100 ≤ A * 100 + 76) ∧ (A * 100 + 76 < 1000) ∧ 
  ((A * 100 + 76) / 100 * 100 = 700) → A = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_rounding_l4110_411068


namespace NUMINAMATH_CALUDE_prob_sum_15_equals_11_663_l4110_411061

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each rank (2 through 10) in a standard deck -/
def cards_per_rank : ℕ := 4

/-- The set of possible ranks that can sum to 15 -/
def valid_ranks : Finset ℕ := {6, 7, 8}

/-- The probability of selecting two number cards that sum to 15 from a standard deck -/
def prob_sum_15 : ℚ :=
  (cards_per_rank * cards_per_rank * 2 + cards_per_rank * (cards_per_rank - 1)) / (deck_size * (deck_size - 1))

theorem prob_sum_15_equals_11_663 : prob_sum_15 = 11 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_15_equals_11_663_l4110_411061


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l4110_411054

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m^2 ≥ 1/20}

/-- Theorem stating the condition for the line to intersect the ellipse -/
theorem line_intersects_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ possible_slopes := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l4110_411054


namespace NUMINAMATH_CALUDE_angle_ratio_MBQ_ABQ_l4110_411006

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- State the conditions
axiom BP_bisects_ABC : angle A B P = angle P B C
axiom BQ_bisects_ABC : angle A B Q = angle Q B C
axiom BM_bisects_PBQ : angle P B M = angle M B Q

-- State the theorem
theorem angle_ratio_MBQ_ABQ : 
  (angle M B Q) / (angle A B Q) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_MBQ_ABQ_l4110_411006


namespace NUMINAMATH_CALUDE_loss_per_meter_is_five_l4110_411028

/-- Calculates the loss per meter of cloth given the total cloth sold, total selling price, and cost price per meter. -/
def loss_per_meter (total_cloth : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  let total_cost_price := total_cloth * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_cloth

/-- Proves that the loss per meter of cloth is 5 rupees given the specific conditions. -/
theorem loss_per_meter_is_five :
  loss_per_meter 450 18000 45 = 5 := by
  sorry

#eval loss_per_meter 450 18000 45

end NUMINAMATH_CALUDE_loss_per_meter_is_five_l4110_411028


namespace NUMINAMATH_CALUDE_birds_left_in_cage_l4110_411047

/-- The number of birds initially in the cage -/
def initial_birds : ℕ := 19

/-- The number of birds taken out of the cage -/
def birds_taken_out : ℕ := 10

/-- Theorem stating that the number of birds left in the cage is 9 -/
theorem birds_left_in_cage : initial_birds - birds_taken_out = 9 := by
  sorry

end NUMINAMATH_CALUDE_birds_left_in_cage_l4110_411047


namespace NUMINAMATH_CALUDE_possible_values_of_a_l4110_411001

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = 1}
def N (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- Define the set of possible values for a
def A : Set ℝ := {-1, 1, 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : N a ⊆ M → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l4110_411001


namespace NUMINAMATH_CALUDE_exponent_sum_l4110_411077

theorem exponent_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l4110_411077


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l4110_411039

theorem repeating_decimal_subtraction : 
  let a : ℚ := 234 / 999
  let b : ℚ := 567 / 999
  let c : ℚ := 891 / 999
  a - b - c = -1224 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l4110_411039


namespace NUMINAMATH_CALUDE_birth_year_digit_sum_difference_l4110_411032

theorem birth_year_digit_sum_difference (m c d u : Nat) 
  (hm : m < 10) (hc : c < 10) (hd : d < 10) (hu : u < 10) :
  ∃ k : Int, (1000 * m + 100 * c + 10 * d + u) - (m + c + d + u) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_birth_year_digit_sum_difference_l4110_411032


namespace NUMINAMATH_CALUDE_phi_equality_iff_in_solution_set_l4110_411056

/-- Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- The set of solutions to the equation φ(2019n) = φ(n²) -/
def solution_set : Set ℕ+ := {1346, 2016, 2019}

/-- Theorem stating that n satisfies φ(2019n) = φ(n²) if and only if n is in the solution set -/
theorem phi_equality_iff_in_solution_set (n : ℕ+) : 
  phi (2019 * n) = phi (n * n) ↔ n ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_phi_equality_iff_in_solution_set_l4110_411056


namespace NUMINAMATH_CALUDE_inequality_proof_l4110_411013

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4110_411013


namespace NUMINAMATH_CALUDE_inverse_proportional_problem_l4110_411093

/-- Given that a and b are inversely proportional, their sum is 24, and their difference is 6,
    prove that when a = 5, b = 27. -/
theorem inverse_proportional_problem (a b : ℝ) (h1 : ∃ k : ℝ, a * b = k) 
  (h2 : a + b = 24) (h3 : a - b = 6) : a = 5 → b = 27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportional_problem_l4110_411093


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l4110_411078

theorem units_digit_of_fraction (n : ℕ) : n = 1994 → (5^n + 6^n) % 7 = 5 → (5^n + 6^n) % 10 = 1 → (5^n + 6^n) / 7 % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l4110_411078


namespace NUMINAMATH_CALUDE_two_minus_repeating_decimal_l4110_411003

/-- The value of the repeating decimal 1.888... -/
def repeating_decimal : ℚ := 17 / 9

/-- Theorem stating that 2 minus the repeating decimal 1.888... equals 1/9 -/
theorem two_minus_repeating_decimal :
  2 - repeating_decimal = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_two_minus_repeating_decimal_l4110_411003


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l4110_411085

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 5040 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 504 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l4110_411085


namespace NUMINAMATH_CALUDE_solution_value_l4110_411035

theorem solution_value (x a : ℝ) (h : x = 5 ∧ a * x - 8 = 20 + a) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l4110_411035


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l4110_411097

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l4110_411097


namespace NUMINAMATH_CALUDE_synthetic_analytic_properties_l4110_411012

/-- Represents a reasoning approach in mathematics or logic -/
inductive ReasoningApproach
| Synthetic
| Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| Forward
| Backward

/-- Represents the relationship between cause and effect in reasoning -/
inductive CauseEffectRelation
| CauseToEffect
| EffectToCause

/-- Properties of a reasoning approach -/
structure ApproachProperties where
  direction : ReasoningDirection
  causeEffect : CauseEffectRelation

/-- Define properties of synthetic and analytic approaches -/
def approachProperties : ReasoningApproach → ApproachProperties
| ReasoningApproach.Synthetic => ⟨ReasoningDirection.Forward, CauseEffectRelation.CauseToEffect⟩
| ReasoningApproach.Analytic => ⟨ReasoningDirection.Backward, CauseEffectRelation.EffectToCause⟩

theorem synthetic_analytic_properties :
  (approachProperties ReasoningApproach.Synthetic).direction = ReasoningDirection.Forward ∧
  (approachProperties ReasoningApproach.Synthetic).causeEffect = CauseEffectRelation.CauseToEffect ∧
  (approachProperties ReasoningApproach.Analytic).direction = ReasoningDirection.Backward ∧
  (approachProperties ReasoningApproach.Analytic).causeEffect = CauseEffectRelation.EffectToCause :=
by sorry

end NUMINAMATH_CALUDE_synthetic_analytic_properties_l4110_411012


namespace NUMINAMATH_CALUDE_yi_jianlian_shots_l4110_411089

/-- Given the basketball game statistics of Yi Jianlian, prove the number of two-point shots and free throws --/
theorem yi_jianlian_shots (total_shots : ℕ) (total_points : ℕ) (three_pointers : ℕ) 
  (h1 : total_shots = 16)
  (h2 : total_points = 28)
  (h3 : three_pointers = 3) :
  ∃ (two_pointers free_throws : ℕ),
    two_pointers + free_throws + three_pointers = total_shots ∧
    2 * two_pointers + free_throws + 3 * three_pointers = total_points ∧
    two_pointers = 6 ∧
    free_throws = 7 := by
  sorry

end NUMINAMATH_CALUDE_yi_jianlian_shots_l4110_411089


namespace NUMINAMATH_CALUDE_complex_trajectory_l4110_411042

theorem complex_trajectory (x y : ℝ) (z : ℂ) :
  z = x + y * I ∧ Complex.abs (z - 1) = x →
  y^2 = 2 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_trajectory_l4110_411042


namespace NUMINAMATH_CALUDE_sum_of_number_and_five_is_nine_l4110_411007

theorem sum_of_number_and_five_is_nine (x : ℤ) : x + 5 = 9 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_five_is_nine_l4110_411007
