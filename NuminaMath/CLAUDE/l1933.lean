import Mathlib

namespace equal_probability_sums_l1933_193345

/-- A standard die with faces labeled 1 to 6 -/
def StandardDie := Fin 6

/-- The number of dice being rolled -/
def numDice : ℕ := 9

/-- The sum we're comparing to -/
def compareSum : ℕ := 15

/-- The sum we're proving has the same probability -/
def targetSum : ℕ := 48

/-- A function to calculate the probability of a specific sum occurring when rolling n dice -/
noncomputable def probabilityOfSum (n : ℕ) (sum : ℕ) : ℝ := sorry

theorem equal_probability_sums :
  probabilityOfSum numDice compareSum = probabilityOfSum numDice targetSum :=
sorry

end equal_probability_sums_l1933_193345


namespace arithmetic_sequences_difference_l1933_193338

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) : ℕ :=
  let n := aₙ - a₁ + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_difference : 
  arithmetic_sum 2001 2093 - arithmetic_sum 201 293 - arithmetic_sum 1 93 = 165044 := by
  sorry

end arithmetic_sequences_difference_l1933_193338


namespace book_sale_profit_l1933_193376

/-- Calculates the difference between total selling price (including tax) and total purchase price (after discount) for books --/
theorem book_sale_profit (num_books : ℕ) (original_price discount_rate desired_price tax_rate : ℚ) : 
  num_books = 15 → 
  original_price = 11 → 
  discount_rate = 1/5 → 
  desired_price = 25 → 
  tax_rate = 1/10 → 
  (num_books * (desired_price * (1 + tax_rate))) - (num_books * (original_price * (1 - discount_rate))) = 280.5 := by
sorry

end book_sale_profit_l1933_193376


namespace infinite_solutions_equation_l1933_193331

theorem infinite_solutions_equation :
  ∀ n : ℕ+, ∃ a b c : ℕ+,
    (a : ℝ) ^ 2 + (b : ℝ) ^ 5 = (c : ℝ) ^ 3 :=
by
  sorry

end infinite_solutions_equation_l1933_193331


namespace original_average_age_l1933_193365

/-- Proves that the original average age of a class was 40 years, given the conditions of the problem -/
theorem original_average_age (original_count : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (avg_decrease : ℕ) :
  original_count = 10 →
  new_count = 10 →
  new_avg_age = 32 →
  avg_decrease = 4 →
  ∃ (original_avg_age : ℕ),
    (original_avg_age * original_count + new_avg_age * new_count) / (original_count + new_count) 
    = original_avg_age - avg_decrease ∧
    original_avg_age = 40 :=
by sorry

end original_average_age_l1933_193365


namespace even_odd_sum_difference_l1933_193325

theorem even_odd_sum_difference : 
  (Finset.sum (Finset.range 100) (fun i => 2 * (i + 1))) - 
  (Finset.sum (Finset.range 100) (fun i => 2 * i + 1)) = 100 := by
  sorry

end even_odd_sum_difference_l1933_193325


namespace max_product_of_ranged_functions_l1933_193355

/-- Given two functions f and g defined on ℝ with specific ranges, 
    prove that the maximum value of their product is -1 -/
theorem max_product_of_ranged_functions 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, 1 ≤ f x ∧ f x ≤ 6) 
  (hg : ∀ x, -4 ≤ g x ∧ g x ≤ -1) : 
  (∀ x, f x * g x ≤ -1) ∧ (∃ x, f x * g x = -1) :=
sorry

end max_product_of_ranged_functions_l1933_193355


namespace factorial_square_root_squared_l1933_193398

theorem factorial_square_root_squared : (((4 * 3 * 2 * 1) * (3 * 2 * 1) : ℕ).sqrt ^ 2 : ℝ) = 144 := by
  sorry

end factorial_square_root_squared_l1933_193398


namespace fraction_equality_l1933_193317

theorem fraction_equality (a b : ℝ) (h : a ≠ 0) : b / a = (a * b) / (a * a) := by
  sorry

end fraction_equality_l1933_193317


namespace equation_B_not_symmetric_l1933_193334

-- Define the equations
def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x + y^2 = -1

-- Define symmetry about x-axis
def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f x (-y)

-- Theorem statement
theorem equation_B_not_symmetric :
  ¬(symmetric_about_x_axis equation_B) ∧
  (symmetric_about_x_axis equation_A) ∧
  (symmetric_about_x_axis equation_C) ∧
  (symmetric_about_x_axis equation_D) :=
sorry

end equation_B_not_symmetric_l1933_193334


namespace perpendicular_vectors_magnitude_l1933_193377

theorem perpendicular_vectors_magnitude (a b : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⟂ b
  (a.1^2 + a.2^2 = 4) →          -- |a| = 2
  (b.1^2 + b.2^2 = 4) →          -- |b| = 2
  ((2*a.1 - b.1)^2 + (2*a.2 - b.2)^2 = 20) := by  -- |2a - b| = 2√5
sorry

end perpendicular_vectors_magnitude_l1933_193377


namespace x_power_4374_minus_reciprocal_l1933_193304

theorem x_power_4374_minus_reciprocal (x : ℂ) : 
  x - (1 / x) = -Complex.I * Real.sqrt 6 → x^4374 - (1 / x^4374) = 0 := by
  sorry

end x_power_4374_minus_reciprocal_l1933_193304


namespace rectangle_coverage_l1933_193379

/-- An L-shaped figure made of 4 unit squares -/
structure LShape :=
  (size : Nat)
  (h_size : size = 4)

/-- Represents a rectangle with dimensions m × n -/
structure Rectangle (m n : Nat) :=
  (width : Nat)
  (height : Nat)
  (h_width : width = m)
  (h_height : height = n)
  (h_positive : m > 1 ∧ n > 1)

/-- Predicate to check if a number is a multiple of 8 -/
def IsMultipleOf8 (n : Nat) : Prop := ∃ k, n = 8 * k

/-- Predicate to check if a rectangle can be covered by L-shaped figures -/
def CanBeCovered (r : Rectangle m n) (l : LShape) : Prop :=
  ∃ (arrangement : Nat), True  -- We don't define the specific arrangement here

/-- The main theorem -/
theorem rectangle_coverage (m n : Nat) (r : Rectangle m n) (l : LShape) :
  (CanBeCovered r l) ↔ (IsMultipleOf8 (m * n)) :=
sorry

end rectangle_coverage_l1933_193379


namespace pool_filling_time_l1933_193303

theorem pool_filling_time (pipe_a pipe_b pipe_c : ℝ) 
  (ha : pipe_a = 1 / 8)
  (hb : pipe_b = 1 / 12)
  (hc : pipe_c = 1 / 16) :
  1 / (pipe_a + pipe_b + pipe_c) = 48 / 13 :=
by sorry

end pool_filling_time_l1933_193303


namespace interest_rate_calculation_l1933_193374

/-- Proves that the interest rate A lends to B is 10% given the specified conditions -/
theorem interest_rate_calculation (principal : ℝ) (rate_c : ℝ) (time : ℝ) (b_gain : ℝ) 
  (h1 : principal = 3500)
  (h2 : rate_c = 0.115)
  (h3 : time = 3)
  (h4 : b_gain = 157.5)
  (h5 : b_gain = principal * rate_c * time - principal * rate_a * time) : 
  rate_a = 0.1 := by
  sorry

#check interest_rate_calculation

end interest_rate_calculation_l1933_193374


namespace nina_savings_weeks_l1933_193300

/-- The number of weeks Nina needs to save to buy a video game -/
def weeks_to_save (game_cost : ℚ) (tax_rate : ℚ) (weekly_allowance : ℚ) (savings_rate : ℚ) : ℚ :=
  let total_cost := game_cost * (1 + tax_rate)
  let weekly_savings := weekly_allowance * savings_rate
  total_cost / weekly_savings

/-- Theorem: Nina needs 11 weeks to save for the video game -/
theorem nina_savings_weeks :
  weeks_to_save 50 0.1 10 0.5 = 11 := by
  sorry

end nina_savings_weeks_l1933_193300


namespace max_subjects_per_teacher_l1933_193375

/-- Proves that the maximum number of subjects a teacher can teach is 4 -/
theorem max_subjects_per_teacher (total_subjects : ℕ) (min_teachers : ℕ) : 
  total_subjects = 16 → min_teachers = 4 → (total_subjects / min_teachers : ℕ) = 4 := by
  sorry

end max_subjects_per_teacher_l1933_193375


namespace perfect_square_condition_l1933_193329

theorem perfect_square_condition (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := by
sorry

end perfect_square_condition_l1933_193329


namespace decimal_expansion_prime_modulo_l1933_193312

theorem decimal_expansion_prime_modulo
  (p : ℕ) (r : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5)
  (hr : ∃ (a : ℕ → ℕ), (∀ i, a i < 10) ∧
    (1 : ℚ) / p = ∑' i, (a i : ℚ) / (10 ^ (i + 1)) - ∑' i, (a (i % r) : ℚ) / (10 ^ (i + r + 1)))
  : 10 ^ r ≡ 1 [MOD p] :=
sorry

end decimal_expansion_prime_modulo_l1933_193312


namespace cost_for_23_days_l1933_193385

/-- Calculates the cost of staying in a student youth hostel for a given number of days. -/
def hostel_cost (days : ℕ) : ℚ :=
  let first_week_cost := 7 * 18
  let remaining_days := days - 7
  let additional_cost := remaining_days * 14
  first_week_cost + additional_cost

/-- Theorem stating that the cost for a 23-day stay is $350.00 -/
theorem cost_for_23_days : hostel_cost 23 = 350 := by
  sorry

end cost_for_23_days_l1933_193385


namespace zoo_animal_count_l1933_193383

/-- Given a zoo with penguins and polar bears, calculate the total number of animals -/
theorem zoo_animal_count (num_penguins : ℕ) (h1 : num_penguins = 21) 
  (h2 : ∃ (num_polar_bears : ℕ), num_polar_bears = 2 * num_penguins) : 
  ∃ (total_animals : ℕ), total_animals = num_penguins + 2 * num_penguins :=
by sorry

end zoo_animal_count_l1933_193383


namespace salesman_pear_sales_l1933_193324

/-- Represents the amount of pears sold by a salesman -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ

/-- The total amount of pears sold in a day -/
def total_sales (sales : PearSales) : ℕ :=
  sales.morning + sales.afternoon

/-- Theorem stating the total sales of pears given the conditions -/
theorem salesman_pear_sales :
  ∃ (sales : PearSales),
    sales.afternoon = 340 ∧
    sales.afternoon = 2 * sales.morning ∧
    total_sales sales = 510 :=
by
  sorry

end salesman_pear_sales_l1933_193324


namespace parabolas_intersection_l1933_193350

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x_coords : Set ℝ :=
  {x | 2 * x^2 - 7 * x + 1 = 5 * x^2 - 2 * x - 2}

/-- The intersection points of two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | p.1 ∈ intersection_x_coords ∧ p.2 = 2 * p.1^2 - 7 * p.1 + 1}

theorem parabolas_intersection :
  intersection_points = {((5 - Real.sqrt 61) / 6, 2 * ((5 - Real.sqrt 61) / 6)^2 - 7 * ((5 - Real.sqrt 61) / 6) + 1),
                         ((5 + Real.sqrt 61) / 6, 2 * ((5 + Real.sqrt 61) / 6)^2 - 7 * ((5 + Real.sqrt 61) / 6) + 1)} :=
by sorry

end parabolas_intersection_l1933_193350


namespace total_hired_is_31_l1933_193337

/-- Represents the daily pay for heavy operators -/
def heavy_operator_pay : ℕ := 129

/-- Represents the daily pay for general laborers -/
def general_laborer_pay : ℕ := 82

/-- Represents the total payroll -/
def total_payroll : ℕ := 3952

/-- Represents the number of general laborers hired -/
def laborers_hired : ℕ := 1

/-- Theorem stating that the total number of people hired is 31 -/
theorem total_hired_is_31 : 
  ∃ (heavy_operators : ℕ), 
    heavy_operator_pay * heavy_operators + general_laborer_pay * laborers_hired = total_payroll ∧
    heavy_operators + laborers_hired = 31 := by
  sorry

end total_hired_is_31_l1933_193337


namespace distance_not_equal_addition_l1933_193364

theorem distance_not_equal_addition : ∀ (a b : ℤ), 
  a = -3 → b = 10 → (abs (b - a) ≠ -3 + 10) :=
by
  sorry

end distance_not_equal_addition_l1933_193364


namespace sin_theta_value_l1933_193301

theorem sin_theta_value (θ : Real) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
  sorry

end sin_theta_value_l1933_193301


namespace base4_sequence_implies_bcd_52_l1933_193368

/-- Represents a digit in base-4 --/
inductive Base4Digit
| A
| B
| C
| D

/-- Converts a Base4Digit to its numerical value --/
def base4DigitToInt (d : Base4Digit) : Nat :=
  match d with
  | Base4Digit.A => 0
  | Base4Digit.B => 1
  | Base4Digit.C => 2
  | Base4Digit.D => 3

/-- Represents a three-digit number in base-4 --/
structure Base4Number :=
  (hundreds : Base4Digit)
  (tens : Base4Digit)
  (ones : Base4Digit)

/-- Converts a Base4Number to its base-10 representation --/
def toBase10 (n : Base4Number) : Nat :=
  4 * 4 * (base4DigitToInt n.hundreds) + 4 * (base4DigitToInt n.tens) + (base4DigitToInt n.ones)

theorem base4_sequence_implies_bcd_52 
  (n1 n2 n3 : Base4Number)
  (h1 : toBase10 n2 = toBase10 n1 + 1)
  (h2 : toBase10 n3 = toBase10 n2 + 1)
  (h3 : n1.hundreds = n2.hundreds ∧ n1.tens = n2.tens)
  (h4 : n2.hundreds = n3.hundreds ∧ n3.tens = Base4Digit.C)
  (h5 : n1.hundreds = Base4Digit.A ∧ n2.hundreds = Base4Digit.A ∧ n3.hundreds = Base4Digit.A)
  (h6 : n1.tens = Base4Digit.B ∧ n2.tens = Base4Digit.B)
  (h7 : n1.ones = Base4Digit.C ∧ n2.ones = Base4Digit.D ∧ n3.ones = Base4Digit.A) :
  toBase10 { hundreds := Base4Digit.B, tens := Base4Digit.C, ones := Base4Digit.D } = 52 := by
  sorry

end base4_sequence_implies_bcd_52_l1933_193368


namespace probability_different_digits_l1933_193332

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def count_valid_numbers : ℕ := 999 - 100 + 1

def count_numbers_with_different_digits : ℕ := 9 * 9 * 8

theorem probability_different_digits :
  (count_numbers_with_different_digits : ℚ) / count_valid_numbers = 18 / 25 := by
  sorry

end probability_different_digits_l1933_193332


namespace multiply_times_theorem_l1933_193330

theorem multiply_times_theorem (n : ℝ) (x : ℝ) (h1 : n = 1) :
  x * n - 1 = 2 * n → x = 3 := by
  sorry

end multiply_times_theorem_l1933_193330


namespace simplify_expression_l1933_193311

theorem simplify_expression (x : ℝ) : 3 * x - 5 * x^2 + 7 + (2 - 3 * x + 5 * x^2) = 9 := by
  sorry

end simplify_expression_l1933_193311


namespace chess_competition_participants_l1933_193305

def is_valid_num_high_school_students (n : ℕ) : Prop :=
  let total_players := n + 2
  let total_games := total_players * (total_players - 1) / 2
  let remaining_points := total_games - 8
  remaining_points % n = 0

theorem chess_competition_participants : 
  ∀ n : ℕ, n > 2 → (is_valid_num_high_school_students n ↔ n = 7 ∨ n = 14) :=
by sorry

end chess_competition_participants_l1933_193305


namespace stating_lens_screen_distance_l1933_193393

/-- Represents the focal length of a thin lens in centimeters -/
def focal_length : ℝ := 150

/-- Represents the distance the screen is moved in centimeters -/
def screen_movement : ℝ := 40

/-- Represents the possible initial distances from the lens to the screen in centimeters -/
def initial_distances : Set ℝ := {130, 170}

/-- 
Theorem stating that given a thin lens with focal length of 150 cm and a screen
that produces the same diameter spot when moved 40 cm, the initial distance
from the lens to the screen is either 130 cm or 170 cm.
-/
theorem lens_screen_distance 
  (s : ℝ) 
  (h1 : s ∈ initial_distances) 
  (h2 : s = focal_length + screen_movement / 2 ∨ s = focal_length - screen_movement / 2) : 
  s ∈ initial_distances :=
sorry

end stating_lens_screen_distance_l1933_193393


namespace pencils_per_row_l1933_193382

theorem pencils_per_row (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) 
  (h1 : packs = 35) 
  (h2 : pencils_per_pack = 4) 
  (h3 : rows = 70) : 
  (packs * pencils_per_pack) / rows = 2 := by
  sorry

end pencils_per_row_l1933_193382


namespace min_cells_in_square_sheet_exists_min_square_sheet_l1933_193309

/-- Represents a rectangular shape with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square shape with side length -/
structure Square where
  side : ℕ

/-- The ship cut out from the paper -/
def ship : Rectangle :=
  { width := 10, height := 11 }

/-- Theorem: The minimum number of cells in the square sheet of paper is 121 -/
theorem min_cells_in_square_sheet : 
  ∀ (s : Square), s.side ≥ max ship.width ship.height → s.side * s.side ≥ 121 :=
by
  sorry

/-- Corollary: There exists a square sheet with exactly 121 cells that can fit the ship -/
theorem exists_min_square_sheet :
  ∃ (s : Square), s.side * s.side = 121 ∧ s.side ≥ max ship.width ship.height :=
by
  sorry

end min_cells_in_square_sheet_exists_min_square_sheet_l1933_193309


namespace rational_solutions_quadratic_l1933_193321

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) := by
sorry

end rational_solutions_quadratic_l1933_193321


namespace impossibility_of_1x3_rectangle_l1933_193323

/-- Represents a cell on the grid -/
structure Cell :=
  (x : Fin 8)
  (y : Fin 8)

/-- Represents a 1x2 rectangle on the grid -/
structure Rectangle :=
  (topLeft : Cell)
  (isVertical : Bool)

/-- Checks if a cell is covered by a rectangle -/
def isCovered (c : Cell) (r : Rectangle) : Prop :=
  (c.x = r.topLeft.x ∧ c.y = r.topLeft.y) ∨
  (r.isVertical ∧ c.x = r.topLeft.x ∧ c.y = r.topLeft.y + 1) ∨
  (¬r.isVertical ∧ c.x = r.topLeft.x + 1 ∧ c.y = r.topLeft.y)

/-- Checks if three consecutive cells form a 1x3 rectangle -/
def is1x3Rectangle (c1 c2 c3 : Cell) : Prop :=
  (c1.x = c2.x ∧ c2.x = c3.x ∧ c2.y = c1.y + 1 ∧ c3.y = c2.y + 1) ∨
  (c1.y = c2.y ∧ c2.y = c3.y ∧ c2.x = c1.x + 1 ∧ c3.x = c2.x + 1)

/-- The main theorem -/
theorem impossibility_of_1x3_rectangle :
  ∃ (configuration : Finset Rectangle),
    configuration.card = 12 ∧
    (∀ c1 c2 c3 : Cell,
      is1x3Rectangle c1 c2 c3 →
      ∃ r ∈ configuration, isCovered c1 r ∨ isCovered c2 r ∨ isCovered c3 r) :=
by
  sorry

end impossibility_of_1x3_rectangle_l1933_193323


namespace largest_difference_l1933_193322

def A : ℕ := 3 * 2010^2011
def B : ℕ := 2010^2011
def C : ℕ := 2009 * 2010^2010
def D : ℕ := 3 * 2010^2010
def E : ℕ := 2010^2010
def F : ℕ := 2010^2009

theorem largest_difference : 
  (A - B) > (B - C) ∧ 
  (A - B) > (C - D) ∧ 
  (A - B) > (D - E) ∧ 
  (A - B) > (E - F) :=
sorry

end largest_difference_l1933_193322


namespace max_value_quadratic_expression_l1933_193356

theorem max_value_quadratic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 - 2*a*b + 3*b^2 = 9 → 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 9*Real.sqrt 3 ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c^2 - 2*c*d + 3*d^2 = 9 ∧ 
  c^2 + 2*c*d + 3*d^2 = 18 + 9*Real.sqrt 3 :=
sorry

end max_value_quadratic_expression_l1933_193356


namespace sufficient_condition_range_l1933_193396

theorem sufficient_condition_range (p : ℝ) : 
  (∀ x : ℝ, 4*x + p < 0 → x^2 - x - 2 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ 4*x + p ≥ 0) →
  p ≥ 4 :=
by sorry

end sufficient_condition_range_l1933_193396


namespace correct_factorization_l1933_193358

theorem correct_factorization (a b : ℝ) : a^2 - 4*a*b + 4*b^2 = (a - 2*b)^2 := by
  sorry

end correct_factorization_l1933_193358


namespace unique_parallel_line_l1933_193315

-- Define a type for points in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a type for lines in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define what it means for a point to be on a line
def PointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define what it means for two lines to be parallel
def ParallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- State the theorem
theorem unique_parallel_line (L : Line) (P : Point) 
  (h : ¬ PointOnLine P L) : 
  ∃! M : Line, ParallelLines M L ∧ PointOnLine P M :=
sorry

end unique_parallel_line_l1933_193315


namespace distance_between_vertices_l1933_193386

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) :
  let f1 := fun x : ℝ => x^2 + a*x + b
  let f2 := fun x : ℝ => x^2 + d*x + e
  let vertex1 := (-a/2, f1 (-a/2))
  let vertex2 := (-d/2, f2 (-d/2))
  a = -4 ∧ b = 7 ∧ d = 6 ∧ e = 20 →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = Real.sqrt 89 :=
by sorry

end distance_between_vertices_l1933_193386


namespace bus_driver_distance_to_destination_l1933_193394

theorem bus_driver_distance_to_destination :
  ∀ (distance_to_destination : ℝ),
    (distance_to_destination / 30 + (distance_to_destination + 10) / 30 + 2 = 6) →
    distance_to_destination = 55 := by
  sorry

end bus_driver_distance_to_destination_l1933_193394


namespace inequality_proof_l1933_193362

theorem inequality_proof (a b c : ℝ) (h : a * b < 0) :
  a^2 + b^2 + c^2 > 2*a*b + 2*b*c + 2*c*a := by
sorry

end inequality_proof_l1933_193362


namespace circle_lines_theorem_l1933_193347

/-- The number of points on the circle -/
def n : ℕ := 5

/-- The total number of lines between any two points -/
def total_lines (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The number of lines between immediate neighbors -/
def neighbor_lines (m : ℕ) : ℕ := m

/-- The number of valid lines (excluding immediate neighbors) -/
def valid_lines (m : ℕ) : ℕ := total_lines m - neighbor_lines m

theorem circle_lines_theorem : valid_lines n = 5 := by sorry

end circle_lines_theorem_l1933_193347


namespace percentage_knives_after_trade_l1933_193349

/-- Represents Carolyn's silverware set -/
structure Silverware :=
  (knives : ℕ)
  (forks : ℕ)
  (spoons : ℕ)

/-- Calculates the total number of silverware pieces -/
def total_silverware (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Calculates the percentage of knives in the silverware set -/
def percentage_knives (s : Silverware) : ℚ :=
  (s.knives : ℚ) / (total_silverware s : ℚ) * 100

/-- The initial silverware set -/
def initial_set : Silverware :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

/-- The silverware set after the trade -/
def after_trade_set : Silverware :=
  { knives := initial_set.knives + 10
  , forks := initial_set.forks
  , spoons := initial_set.spoons - 6 }

theorem percentage_knives_after_trade :
  percentage_knives after_trade_set = 40 := by
  sorry


end percentage_knives_after_trade_l1933_193349


namespace base_conversion_725_9_l1933_193366

def base_9_to_base_3 (n : Nat) : Nat :=
  -- Definition of conversion from base 9 to base 3
  sorry

theorem base_conversion_725_9 :
  base_9_to_base_3 725 = 210212 :=
sorry

end base_conversion_725_9_l1933_193366


namespace roses_stolen_l1933_193335

/-- Given the initial number of roses, number of people, and roses per person,
    prove that the number of roses stolen is equal to the initial number of roses
    minus the product of the number of people and roses per person. -/
theorem roses_stolen (initial_roses : ℕ) (num_people : ℕ) (roses_per_person : ℕ) :
  initial_roses - (num_people * roses_per_person) =
  initial_roses - num_people * roses_per_person :=
by sorry

end roses_stolen_l1933_193335


namespace polygon_with_1080_degrees_is_octagon_l1933_193320

/-- A polygon with interior angles summing to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_is_octagon :
  ∀ n : ℕ, (n - 2) * 180 = 1080 → n = 8 := by
  sorry

end polygon_with_1080_degrees_is_octagon_l1933_193320


namespace part1_solution_set_part2_m_range_l1933_193381

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2 * x - 1|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
sorry

-- Part 2
theorem part2_m_range (m : ℝ) (hm : m > 0) :
  (∀ x ∈ Set.Icc m (2 * m^2), (1/2) * f m x ≤ |x + 1|) →
  1/2 < m ∧ m ≤ 1 :=
sorry

end part1_solution_set_part2_m_range_l1933_193381


namespace inequality_solution_range_l1933_193363

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
sorry

end inequality_solution_range_l1933_193363


namespace distance_to_reflection_l1933_193310

/-- The distance between a point (3, 1) and its reflection over the y-axis is 6 units. -/
theorem distance_to_reflection : ∃ (X X' : ℝ × ℝ),
  X = (3, 1) ∧
  X'.1 = -X.1 ∧
  X'.2 = X.2 ∧
  Real.sqrt ((X'.1 - X.1)^2 + (X'.2 - X.2)^2) = 6 :=
by
  sorry

end distance_to_reflection_l1933_193310


namespace instrument_probability_l1933_193357

/-- The probability of selecting a cello and a viola made from the same tree -/
theorem instrument_probability (total_cellos : ℕ) (total_violas : ℕ) (same_tree_pairs : ℕ) :
  total_cellos = 800 →
  total_violas = 600 →
  same_tree_pairs = 100 →
  (same_tree_pairs : ℚ) / (total_cellos * total_violas) = 1 / 4800 := by
  sorry

end instrument_probability_l1933_193357


namespace texas_california_plates_equal_l1933_193380

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The number of possible California license plates -/
def california_plates : ℕ := num_digits * num_letters^3 * num_digits^3

/-- Theorem stating that Texas and California can issue the same number of license plates -/
theorem texas_california_plates_equal : texas_plates = california_plates := by
  sorry

end texas_california_plates_equal_l1933_193380


namespace tangent_line_y_intercept_l1933_193397

/-- The y-coordinate of the intersection point between the tangent line
    to the curve y = x^3 + 11 at point P(1, 12) and the y-axis is 9. -/
theorem tangent_line_y_intercept : 
  let f (x : ℝ) := x^3 + 11
  let P : ℝ × ℝ := (1, 12)
  let m : ℝ := deriv f 1
  let tangent_line (x : ℝ) := m * (x - P.1) + P.2
  tangent_line 0 = 9 := by sorry

end tangent_line_y_intercept_l1933_193397


namespace officer_selection_count_l1933_193361

def club_members : ℕ := 12
def officer_positions : ℕ := 5

theorem officer_selection_count :
  (club_members.factorial) / ((club_members - officer_positions).factorial) = 95040 := by
  sorry

end officer_selection_count_l1933_193361


namespace square_tile_side_length_l1933_193352

theorem square_tile_side_length (side : ℝ) (area : ℝ) : 
  area = 49 ∧ area = side * side → side = 7 := by
  sorry

end square_tile_side_length_l1933_193352


namespace soda_discount_percentage_l1933_193392

/-- The discount percentage for soda cans purchased in 24-can cases -/
def discount_percentage (regular_price : ℚ) (discounted_price : ℚ) : ℚ :=
  (1 - discounted_price / (100 * regular_price)) * 100

/-- Theorem stating that the discount percentage is 15% -/
theorem soda_discount_percentage :
  let regular_price : ℚ := 40 / 100  -- $0.40 per can
  let discounted_price : ℚ := 34     -- $34 for 100 cans
  discount_percentage regular_price discounted_price = 15 := by
  sorry

#eval discount_percentage (40/100) 34

end soda_discount_percentage_l1933_193392


namespace jesse_pages_left_to_read_l1933_193336

/-- Represents a book with a given number of pages in the first 5 chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ
  chapter4 : ℕ
  chapter5 : ℕ

/-- The number of pages left to read in the book -/
def pagesLeftToRead (b : Book) : ℕ :=
  let pagesRead := b.chapter1 + b.chapter2 + b.chapter3 + b.chapter4 + b.chapter5
  let totalPages := pagesRead * 3
  totalPages - pagesRead

/-- Theorem stating the number of pages left to read in Jesse's book -/
theorem jesse_pages_left_to_read :
  let jessesBook : Book := {
    chapter1 := 10,
    chapter2 := 15,
    chapter3 := 27,
    chapter4 := 12,
    chapter5 := 19
  }
  pagesLeftToRead jessesBook = 166 := by
  sorry

end jesse_pages_left_to_read_l1933_193336


namespace stratified_sample_properties_l1933_193340

/-- Represents the grades of parts in a batch -/
inductive Grade
  | First
  | Second
  | Third

/-- Structure representing a batch of parts -/
structure Batch :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Structure representing a sample drawn from a batch -/
structure Sample :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Function to check if a sample is valid for a given batch -/
def isValidSample (b : Batch) (s : Sample) : Prop :=
  s.first + s.second + s.third = 20 ∧
  s.first ≤ b.first ∧
  s.second ≤ b.second ∧
  s.third ≤ b.third

/-- Theorem stating the properties of the stratified sample -/
theorem stratified_sample_properties (b : Batch) (s : Sample) :
  b.first = 24 →
  b.second = 36 →
  s.third = 10 →
  isValidSample b s →
  b.third = 60 ∧ s.second = 6 := by sorry

end stratified_sample_properties_l1933_193340


namespace expansion_distinct_terms_l1933_193314

/-- The number of distinct terms in the expansion of (x+y)(a+b+c)(d+e+f) -/
def num_distinct_terms : ℕ := 18

/-- The first factor has 2 terms -/
def num_terms_factor1 : ℕ := 2

/-- The second factor has 3 terms -/
def num_terms_factor2 : ℕ := 3

/-- The third factor has 3 terms -/
def num_terms_factor3 : ℕ := 3

theorem expansion_distinct_terms :
  num_distinct_terms = num_terms_factor1 * num_terms_factor2 * num_terms_factor3 := by
  sorry

end expansion_distinct_terms_l1933_193314


namespace triangle_area_l1933_193371

-- Define the triangle ABC and point K
variable (A B C K : ℝ × ℝ)

-- Define the conditions
def is_on_line (P Q R : ℝ × ℝ) : Prop := sorry
def is_altitude (P Q R S : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_area :
  is_on_line K B C →
  is_altitude A K B C →
  distance A C = 12 →
  distance B K = 9 →
  distance B C = 18 →
  area A B C = 27 * Real.sqrt 7 := by sorry

end triangle_area_l1933_193371


namespace curve_is_semicircle_l1933_193384

-- Define the curve
def curve (x y : ℝ) : Prop := x - 1 = Real.sqrt (1 - (y - 1)^2)

-- Define a semicircle
def semicircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧ x ≥ center.1

theorem curve_is_semicircle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (x y : ℝ), curve x y ↔ semicircle center radius x y :=
sorry

end curve_is_semicircle_l1933_193384


namespace satisfying_digits_characterization_l1933_193387

/-- A digit is a natural number less than 10. -/
def Digit : Type := { d : ℕ // d < 10 }

/-- The set of digits that satisfy the given property. -/
def SatisfyingDigits : Set Digit :=
  { z : Digit | ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ n^9 % 10^k = z.val^k % 10^k }

/-- The theorem stating that the satisfying digits are exactly 0, 1, 5, and 6. -/
theorem satisfying_digits_characterization :
  SatisfyingDigits = {⟨0, by norm_num⟩, ⟨1, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩} :=
by sorry

end satisfying_digits_characterization_l1933_193387


namespace office_gender_ratio_l1933_193373

/-- Given an office with 60 employees, if a meeting of 4 men and 6 women
    reduces the number of women on the office floor by 20%,
    then the ratio of men to women in the office is 1:1. -/
theorem office_gender_ratio
  (total_employees : ℕ)
  (meeting_men : ℕ)
  (meeting_women : ℕ)
  (women_reduction_percent : ℚ)
  (h1 : total_employees = 60)
  (h2 : meeting_men = 4)
  (h3 : meeting_women = 6)
  (h4 : women_reduction_percent = 1/5)
  : (total_employees / 2 : ℚ) = (total_employees - (total_employees / 2) : ℚ) :=
by sorry

end office_gender_ratio_l1933_193373


namespace max_consecutive_integers_sum_largest_n_not_exceeding_500_l1933_193353

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by
  sorry

theorem largest_n_not_exceeding_500 : 
  ∀ k : ℕ, (k * (k + 1) ≤ 1000 → k ≤ 31) ∧ 
           (31 * 32 ≤ 1000) ∧ 
           (32 * 33 > 1000) := by
  sorry

end max_consecutive_integers_sum_largest_n_not_exceeding_500_l1933_193353


namespace consecutive_prime_even_triangular_product_l1933_193343

/-- A number is triangular if it can be represented as n * (n + 1) / 2 for some natural number n. -/
def IsTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

theorem consecutive_prime_even_triangular_product : ∃ a b c : ℕ,
  (a < 20 ∧ b < 20 ∧ c < 20) ∧
  (b = a + 1 ∧ c = b + 1) ∧
  Nat.Prime a ∧
  Even b ∧
  IsTriangular c ∧
  a * b * c = 2730 :=
sorry

end consecutive_prime_even_triangular_product_l1933_193343


namespace min_value_of_function_min_value_achievable_l1933_193399

theorem min_value_of_function (x : ℝ) : 3 * x^2 + 6 / (x^2 + 1) ≥ 6 * Real.sqrt 2 - 3 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, 3 * x^2 + 6 / (x^2 + 1) = 6 * Real.sqrt 2 - 3 := by
  sorry

end min_value_of_function_min_value_achievable_l1933_193399


namespace defective_draws_count_l1933_193342

/-- The number of ways to draw at least 2 defective products from a batch of 100 products
    containing 3 defective ones, when drawing 5 products. -/
def defectiveDraws : ℕ := sorry

/-- The total number of products -/
def totalProducts : ℕ := 100

/-- The number of defective products -/
def defectiveProducts : ℕ := 3

/-- The number of products drawn -/
def drawnProducts : ℕ := 5

theorem defective_draws_count :
  defectiveDraws = Nat.choose 3 2 * Nat.choose 97 3 + Nat.choose 3 3 * Nat.choose 97 2 := by
  sorry

end defective_draws_count_l1933_193342


namespace ball_max_height_l1933_193307

-- Define the height function
def h (t : ℝ) : ℝ := -4 * t^2 + 40 * t + 20

-- State the theorem
theorem ball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 120 :=
sorry

end ball_max_height_l1933_193307


namespace polynomial_factorization_l1933_193367

theorem polynomial_factorization (a b : ℝ) : a^3*b - 9*a*b = a*b*(a+3)*(a-3) := by
  sorry

end polynomial_factorization_l1933_193367


namespace inequality_range_l1933_193344

theorem inequality_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ k > 0, Real.log (Real.sin θ)^2 - Real.log (Real.cos θ)^2 ≤ k * Real.cos (2 * θ)) ↔
  θ ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Icc (3 * Real.pi / 4) Real.pi ∪ 
      Set.Ioo Real.pi (5 * Real.pi / 4) ∪ Set.Icc (7 * Real.pi / 4) (2 * Real.pi) :=
by sorry

end inequality_range_l1933_193344


namespace lorelai_jellybeans_count_l1933_193370

/-- The number of jellybeans Gigi has -/
def gigi_jellybeans : ℕ := 15

/-- The number of extra jellybeans Rory has compared to Gigi -/
def rory_extra_jellybeans : ℕ := 30

/-- The number of jellybeans Rory has -/
def rory_jellybeans : ℕ := gigi_jellybeans + rory_extra_jellybeans

/-- The total number of jellybeans both girls have -/
def total_girls_jellybeans : ℕ := gigi_jellybeans + rory_jellybeans

/-- The number of times Lorelai has eaten compared to both girls -/
def lorelai_multiplier : ℕ := 3

/-- The number of jellybeans Lorelai has eaten -/
def lorelai_jellybeans : ℕ := total_girls_jellybeans * lorelai_multiplier

theorem lorelai_jellybeans_count : lorelai_jellybeans = 180 := by
  sorry

end lorelai_jellybeans_count_l1933_193370


namespace factorization_x_squared_plus_2x_l1933_193339

theorem factorization_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x+2) := by
  sorry

end factorization_x_squared_plus_2x_l1933_193339


namespace seashells_given_to_joan_l1933_193395

theorem seashells_given_to_joan (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 8) 
  (h2 : remaining_seashells = 2) : 
  initial_seashells - remaining_seashells = 6 := by
  sorry

end seashells_given_to_joan_l1933_193395


namespace total_cost_four_games_l1933_193389

def batman_price : ℝ := 13.60
def superman_price : ℝ := 5.06
def batman_discount : ℝ := 0.10
def superman_discount : ℝ := 0.05
def sales_tax : ℝ := 0.08
def owned_game1 : ℝ := 7.25
def owned_game2 : ℝ := 12.50

theorem total_cost_four_games :
  let batman_discounted := batman_price * (1 - batman_discount)
  let superman_discounted := superman_price * (1 - superman_discount)
  let batman_with_tax := batman_discounted * (1 + sales_tax)
  let superman_with_tax := superman_discounted * (1 + sales_tax)
  let total_cost := batman_with_tax + superman_with_tax + owned_game1 + owned_game2
  total_cost = 38.16 := by
  sorry

end total_cost_four_games_l1933_193389


namespace cos_420_plus_sin_330_eq_zero_l1933_193351

theorem cos_420_plus_sin_330_eq_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end cos_420_plus_sin_330_eq_zero_l1933_193351


namespace necessary_but_not_sufficient_condition_l1933_193308

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
sorry

end necessary_but_not_sufficient_condition_l1933_193308


namespace albums_needed_for_xiao_hong_l1933_193346

/-- Calculates the minimum number of complete photo albums needed to store a given number of photos. -/
def minimum_albums_needed (pages_per_album : ℕ) (photos_per_page : ℕ) (total_photos : ℕ) : ℕ :=
  (total_photos + pages_per_album * photos_per_page - 1) / (pages_per_album * photos_per_page)

/-- Proves that 6 albums are needed for the given conditions. -/
theorem albums_needed_for_xiao_hong : minimum_albums_needed 32 5 900 = 6 := by
  sorry

#eval minimum_albums_needed 32 5 900

end albums_needed_for_xiao_hong_l1933_193346


namespace factorial_division_l1933_193372

theorem factorial_division : 8 / 3 = 6720 :=
by
  -- Define 8! as given in the problem
  have h1 : 8 = 40320 := by sorry
  
  -- Define 3! (not given in the problem, but necessary for the proof)
  have h2 : 3 = 6 := by sorry
  
  -- Prove that 8! ÷ 3! = 6720
  sorry

end factorial_division_l1933_193372


namespace arithmetic_geometric_ratio_l1933_193316

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the problem
theorem arithmetic_geometric_ratio
  (d : ℝ) (h_d : d ≠ 0)
  (h_geom : ∃ a₁ : ℝ, (arithmetic_sequence a₁ d 3)^2 = 
    (arithmetic_sequence a₁ d 2) * (arithmetic_sequence a₁ d 9)) :
  ∃ a₁ : ℝ, (arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 4) /
            (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6) = 3/8 :=
by sorry

end arithmetic_geometric_ratio_l1933_193316


namespace circle_square_area_difference_l1933_193348

/-- The difference between the areas of the non-overlapping portions of a circle and a square -/
theorem circle_square_area_difference (r c s : ℝ) (h1 : r = 3) (h2 : s = 2) : 
  (π * r^2 - s^2) = 9 * π - 4 := by
  sorry

end circle_square_area_difference_l1933_193348


namespace bakery_flour_usage_l1933_193391

theorem bakery_flour_usage (wheat_flour : ℝ) (white_flour : ℝ) 
  (h1 : wheat_flour = 0.2)
  (h2 : white_flour = 0.1) :
  wheat_flour + white_flour = 0.3 := by
  sorry

end bakery_flour_usage_l1933_193391


namespace equation_has_seven_solutions_l1933_193306

/-- The function f(x) = |x² - 2x - 3| -/
def f (x : ℝ) : ℝ := |x^2 - 2*x - 3|

/-- The equation f³(x) - 4f²(x) - f(x) + 4 = 0 -/
def equation (x : ℝ) : Prop :=
  f x ^ 3 - 4 * (f x)^2 - f x + 4 = 0

/-- Theorem stating that the equation has exactly 7 solutions -/
theorem equation_has_seven_solutions :
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x, x ∈ s ↔ equation x :=
sorry

end equation_has_seven_solutions_l1933_193306


namespace complex_power_difference_l1933_193327

/-- Given that i^2 = -1, prove that (1+2i)^24 - (1-2i)^24 = 0 -/
theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i)^24 - (1 - 2*i)^24 = 0 := by
  sorry

end complex_power_difference_l1933_193327


namespace cube_frame_problem_solution_l1933_193390

/-- Represents the cube frame construction problem. -/
structure CubeFrameProblem where
  bonnie_wire_length : ℕ        -- Length of each wire piece Bonnie uses
  bonnie_wire_count : ℕ         -- Number of wire pieces Bonnie uses
  roark_wire_length : ℕ         -- Length of each wire piece Roark uses
  roark_cube_edge_length : ℕ    -- Edge length of Roark's unit cubes

/-- The solution to the cube frame problem. -/
def cubeProblemSolution (p : CubeFrameProblem) : ℚ :=
  let bonnie_total_length := p.bonnie_wire_length * p.bonnie_wire_count
  let bonnie_cube_volume := p.bonnie_wire_length ^ 3
  let roark_cube_count := bonnie_cube_volume
  let roark_wire_per_cube := 12 * p.roark_wire_length
  let roark_total_length := roark_cube_count * roark_wire_per_cube
  bonnie_total_length / roark_total_length

/-- Theorem stating the solution to the cube frame problem. -/
theorem cube_frame_problem_solution (p : CubeFrameProblem) 
  (h1 : p.bonnie_wire_length = 8)
  (h2 : p.bonnie_wire_count = 12)
  (h3 : p.roark_wire_length = 2)
  (h4 : p.roark_cube_edge_length = 1) :
  cubeProblemSolution p = 1 / 128 := by
  sorry

end cube_frame_problem_solution_l1933_193390


namespace ratio_problem_l1933_193369

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 3) :
  a / c = 15 / 8 := by
  sorry

end ratio_problem_l1933_193369


namespace cube_volume_l1933_193318

theorem cube_volume (s : ℝ) : 
  (s + 2) * (s - 2) * s = s^3 - 12 → s^3 = 27 := by
  sorry

end cube_volume_l1933_193318


namespace product_of_squares_l1933_193302

theorem product_of_squares : 
  (1 + 1 / 1^2) * (1 + 1 / 2^2) * (1 + 1 / 3^2) * (1 + 1 / 4^2) * (1 + 1 / 5^2) * (1 + 1 / 6^2) = 16661 / 3240 := by
  sorry

end product_of_squares_l1933_193302


namespace solve_for_x_l1933_193326

def star_op (x y : ℤ) : ℤ := x * y - 2 * (x + y)

theorem solve_for_x : ∃ x : ℤ, (∀ y : ℤ, star_op x y = x * y - 2 * (x + y)) ∧ star_op x (-3) = 1 → x = 1 := by
  sorry

end solve_for_x_l1933_193326


namespace line_tangent_to_parabola_l1933_193354

/-- A line y = 2x + c is tangent to the parabola y^2 = 8x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = 2 * p.1 + c ∧ p.2^2 = 8 * p.1) ↔ c = 1 := by
  sorry

end line_tangent_to_parabola_l1933_193354


namespace f_properties_l1933_193378

noncomputable def f (x : ℝ) := Real.exp (-x) * Real.sin x

theorem f_properties :
  let a := -Real.pi
  let b := Real.pi
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc a b, f x = max_val) ∧
    (∀ x ∈ Set.Icc a b, min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min_val) ∧
    (StrictMonoOn f (Set.Ioo (-3*Real.pi/4) (Real.pi/4))) ∧
    (StrictAntiOn f (Set.Ioc a (-3*Real.pi/4))) ∧
    (StrictAntiOn f (Set.Ico (Real.pi/4) b)) ∧
    max_val = (Real.sqrt 2 / 2) * Real.exp (-Real.pi/4) ∧
    min_val = -(Real.sqrt 2 / 2) * Real.exp (3*Real.pi/4) :=
by sorry

end f_properties_l1933_193378


namespace part_I_part_II_l1933_193359

-- Define propositions p and q
def p (a : ℝ) : Prop := a > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → a - x ≥ 0

-- Part I
theorem part_I (a : ℝ) (hq : q a) : a ∈ {a : ℝ | a ≥ -1} := by sorry

-- Part II
theorem part_II (a : ℝ) (h_or : p a ∨ q a) (h_not_and : ¬(p a ∧ q a)) : 
  a ∈ Set.Icc (-1) 0 := by sorry

end part_I_part_II_l1933_193359


namespace paint_cost_per_square_meter_l1933_193341

/-- Calculates the paint cost per square meter for a mural --/
theorem paint_cost_per_square_meter
  (mural_length : ℝ)
  (mural_width : ℝ)
  (painting_rate : ℝ)
  (labor_charge : ℝ)
  (total_cost : ℝ)
  (h1 : mural_length = 6)
  (h2 : mural_width = 3)
  (h3 : painting_rate = 1.5)
  (h4 : labor_charge = 10)
  (h5 : total_cost = 192) :
  (total_cost - (mural_length * mural_width / painting_rate) * labor_charge) / (mural_length * mural_width) = 4 :=
by sorry

end paint_cost_per_square_meter_l1933_193341


namespace fraction_change_l1933_193333

theorem fraction_change (n d x : ℚ) : 
  d = 2 * n - 1 →
  n / d = 5 / 9 →
  (n + x) / (d + x) = 3 / 5 →
  x = 1 := by sorry

end fraction_change_l1933_193333


namespace alcohol_water_ratio_l1933_193313

theorem alcohol_water_ratio (alcohol_volume water_volume : ℚ) : 
  alcohol_volume = 2/7 → water_volume = 3/7 → 
  alcohol_volume / water_volume = 2/3 := by
  sorry

end alcohol_water_ratio_l1933_193313


namespace final_cafeteria_count_l1933_193328

def total_students : ℕ := 300

def initial_cafeteria : ℕ := (2 * total_students) / 5
def initial_outside : ℕ := (3 * total_students) / 10
def initial_classroom : ℕ := total_students - initial_cafeteria - initial_outside

def outside_to_cafeteria : ℕ := (40 * initial_outside) / 100
def cafeteria_to_outside : ℕ := 5
def classroom_to_cafeteria : ℕ := (15 * initial_classroom + 50) / 100  -- Rounded up
def outside_to_classroom : ℕ := 2

theorem final_cafeteria_count :
  initial_cafeteria + outside_to_cafeteria - cafeteria_to_outside + classroom_to_cafeteria = 165 :=
sorry

end final_cafeteria_count_l1933_193328


namespace turtle_difference_is_nine_l1933_193319

/-- Given the number of turtles Kristen has, calculate the difference between Trey's and Kristen's turtle counts. -/
def turtle_difference (kristen_turtles : ℕ) : ℕ :=
  let kris_turtles := kristen_turtles / 4
  let trey_turtles := 7 * kris_turtles
  trey_turtles - kristen_turtles

/-- Theorem stating that the difference between Trey's and Kristen's turtle counts is 9 when Kristen has 12 turtles. -/
theorem turtle_difference_is_nine :
  turtle_difference 12 = 9 := by
  sorry

end turtle_difference_is_nine_l1933_193319


namespace line_intersection_x_axis_l1933_193360

/-- Given a line passing through (2, 6) and (5, c) that intersects the x-axis at (d, 0), prove that d = -16 -/
theorem line_intersection_x_axis (c : ℝ) (d : ℝ) : 
  (∃ (m : ℝ), (6 - 0) = m * (2 - d) ∧ (c - 6) = m * (5 - 2)) → d = -16 := by
  sorry

end line_intersection_x_axis_l1933_193360


namespace pyramid_height_theorem_l1933_193388

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  /-- The distance from the apex to the larger section -/
  height : ℝ
  /-- The area of the larger section -/
  larger_section_area : ℝ
  /-- The area of the smaller section -/
  smaller_section_area : ℝ
  /-- The distance between the two sections -/
  section_distance : ℝ

/-- Theorem stating the relationship between the sections and the height of the pyramid -/
theorem pyramid_height_theorem (pyramid : RightOctagonalPyramid) 
    (h1 : pyramid.larger_section_area = 810)
    (h2 : pyramid.smaller_section_area = 360)
    (h3 : pyramid.section_distance = 10) : 
    pyramid.height = 30 := by
  sorry

end pyramid_height_theorem_l1933_193388
