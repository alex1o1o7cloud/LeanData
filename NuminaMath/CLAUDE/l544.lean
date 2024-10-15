import Mathlib

namespace NUMINAMATH_CALUDE_cos_equality_solution_l544_54473

theorem cos_equality_solution (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 360) 
  (h3 : Real.cos (m * π / 180) = Real.cos (970 * π / 180)) : 
  m = 110 ∨ m = 250 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solution_l544_54473


namespace NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l544_54485

-- Define the original expression
def original_expression (a b c : ℝ) : ℝ := (48 * a^5 * b^8 * c^14)^(1/4)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^3 * (3 * a * c^2)^(1/4)

-- Theorem statement
theorem sum_of_exponents_outside_radical (a b c : ℝ) : 
  original_expression a b c = simplified_expression a b c → 
  (1 : ℕ) + 2 + 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_outside_radical_l544_54485


namespace NUMINAMATH_CALUDE_symmetry_correctness_l544_54434

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

def symmetryYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

def symmetryYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

def symmetryOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Theorem statement
theorem symmetry_correctness (p : Point3D) :
  (symmetryXAxis p ≠ p) ∧
  (symmetryYOzPlane p ≠ p) ∧
  (symmetryYAxis p ≠ p) ∧
  (symmetryOrigin p = { x := -p.x, y := -p.y, z := -p.z }) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_correctness_l544_54434


namespace NUMINAMATH_CALUDE_not_right_triangle_l544_54445

theorem not_right_triangle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 2 * B) (h3 : A = 3 * C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
sorry

end NUMINAMATH_CALUDE_not_right_triangle_l544_54445


namespace NUMINAMATH_CALUDE_fib_100_mod_5_l544_54478

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Periodicity of Fibonacci sequence modulo 5 -/
axiom fib_mod_5_periodic (n : ℕ) : fib (n + 5) % 5 = fib n % 5

/-- Theorem: The 100th Fibonacci number modulo 5 is 0 -/
theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_5_l544_54478


namespace NUMINAMATH_CALUDE_kyle_paper_delivery_l544_54462

/-- The number of papers Kyle delivers in a week -/
def weekly_papers (weekday_houses : ℕ) (sunday_skip : ℕ) (sunday_extra : ℕ) : ℕ :=
  (weekday_houses * 6) + (weekday_houses - sunday_skip + sunday_extra)

/-- Theorem stating the total number of papers Kyle delivers in a week -/
theorem kyle_paper_delivery :
  weekly_papers 100 10 30 = 720 := by
  sorry

end NUMINAMATH_CALUDE_kyle_paper_delivery_l544_54462


namespace NUMINAMATH_CALUDE_simplify_fraction_l544_54402

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l544_54402


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l544_54438

/-- Given two points P and Q symmetric with respect to the origin, prove that a + b = -11 --/
theorem symmetric_points_sum (a b : ℝ) :
  let P : ℝ × ℝ := (a + 3*b, 3)
  let Q : ℝ × ℝ := (-5, a + 2*b)
  (P.1 = -Q.1 ∧ P.2 = -Q.2) →
  a + b = -11 := by
sorry


end NUMINAMATH_CALUDE_symmetric_points_sum_l544_54438


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l544_54466

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l544_54466


namespace NUMINAMATH_CALUDE_no_double_apply_function_exists_l544_54482

theorem no_double_apply_function_exists : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_double_apply_function_exists_l544_54482


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l544_54440

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - x * y = 0) :
  x + 2 * y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y - x * y = 0 ∧ x + 2 * y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l544_54440


namespace NUMINAMATH_CALUDE_total_amount_proof_l544_54427

def calculate_total_amount (plant_price tool_price soil_price : ℝ)
  (plant_discount tool_discount : ℝ) (tax_rate : ℝ) (surcharge : ℝ) : ℝ :=
  let discounted_plant := plant_price * (1 - plant_discount)
  let discounted_tool := tool_price * (1 - tool_discount)
  let subtotal := discounted_plant + discounted_tool + soil_price
  let total_with_tax := subtotal * (1 + tax_rate)
  total_with_tax + surcharge

theorem total_amount_proof :
  calculate_total_amount 467 85 38 0.15 0.10 0.08 12 = 564.37 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l544_54427


namespace NUMINAMATH_CALUDE_equal_distribution_of_boxes_l544_54469

theorem equal_distribution_of_boxes (total_boxes : ℕ) (num_stops : ℕ) 
  (h1 : total_boxes = 27) (h2 : num_stops = 3) :
  total_boxes / num_stops = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_boxes_l544_54469


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_exists_l544_54476

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem 1: Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, f (2 * x) ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem 2: Minimum value
theorem minimum_value :
  ∀ a b : ℝ, a + b = 2 → f (a^2) + f (b^2) ≥ 2 :=
sorry

-- Theorem 3: Existence of minimum
theorem minimum_exists :
  ∃ a b : ℝ, a + b = 2 ∧ f (a^2) + f (b^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_exists_l544_54476


namespace NUMINAMATH_CALUDE_resulting_polygon_sides_l544_54495

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  is_regular : sides ≥ 3

/-- Represents the sequence of polygons in the construction. -/
def polygon_sequence : List RegularPolygon :=
  [⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, 
   ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Calculates the number of exposed sides in the resulting polygon. -/
def exposed_sides (seq : List RegularPolygon) : ℕ :=
  (seq.map (·.sides)).sum - 2 * (seq.length - 1)

/-- Theorem stating that the resulting polygon has 30 sides. -/
theorem resulting_polygon_sides : exposed_sides polygon_sequence = 30 := by
  sorry


end NUMINAMATH_CALUDE_resulting_polygon_sides_l544_54495


namespace NUMINAMATH_CALUDE_savings_increase_l544_54405

theorem savings_increase (income expenditure savings new_income new_expenditure new_savings : ℝ)
  (h1 : expenditure = 0.75 * income)
  (h2 : savings = income - expenditure)
  (h3 : new_income = 1.2 * income)
  (h4 : new_expenditure = 1.1 * expenditure)
  (h5 : new_savings = new_income - new_expenditure) :
  (new_savings - savings) / savings * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_savings_increase_l544_54405


namespace NUMINAMATH_CALUDE_equation_solution_l544_54484

theorem equation_solution :
  ∀ x y : ℝ, x^2 - y^4 = Real.sqrt (18*x - x^2 - 81) ↔ (x = 9 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l544_54484


namespace NUMINAMATH_CALUDE_equation_has_six_roots_l544_54472

/-- The number of roots of the equation √(14-x²)(sin x - cos 2x) = 0 in the interval [-√14, √14] -/
def num_roots : ℕ := 6

/-- The equation √(14-x²)(sin x - cos 2x) = 0 -/
def equation (x : ℝ) : Prop :=
  Real.sqrt (14 - x^2) * (Real.sin x - Real.cos (2 * x)) = 0

/-- The domain of the equation -/
def domain (x : ℝ) : Prop :=
  x ≥ -Real.sqrt 14 ∧ x ≤ Real.sqrt 14

/-- Theorem stating that the equation has exactly 6 roots in the given domain -/
theorem equation_has_six_roots :
  ∃! (s : Finset ℝ), s.card = num_roots ∧ 
  (∀ x ∈ s, domain x ∧ equation x) ∧
  (∀ x, domain x → equation x → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_equation_has_six_roots_l544_54472


namespace NUMINAMATH_CALUDE_solve_candy_problem_l544_54423

def candy_problem (candy_from_neighbors : ℝ) : Prop :=
  let candy_from_sister : ℝ := 5.0
  let candy_per_day : ℝ := 8.0
  let days_lasted : ℝ := 2.0
  let total_candy_eaten : ℝ := candy_per_day * days_lasted
  candy_from_neighbors = total_candy_eaten - candy_from_sister

theorem solve_candy_problem :
  ∃ (candy_from_neighbors : ℝ), candy_problem candy_from_neighbors ∧ candy_from_neighbors = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l544_54423


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l544_54424

/-- 
Given a point P with coordinates (x, -8) where the distance from the x-axis to P 
is half the distance from the y-axis to P, prove that P is 16 units from the y-axis.
-/
theorem distance_to_y_axis (x : ℝ) :
  let p : ℝ × ℝ := (x, -8)
  let dist_to_x_axis := |p.2|
  let dist_to_y_axis := |p.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis →
  dist_to_y_axis = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l544_54424


namespace NUMINAMATH_CALUDE_regression_difference_is_residual_sum_of_squares_l544_54497

/-- In regression analysis, the term representing the difference between a data point
    and its corresponding position on the regression line -/
def regression_difference_term : String := "residual sum of squares"

/-- The residual sum of squares represents the difference between data points
    and their corresponding positions on the regression line -/
axiom residual_sum_of_squares_def :
  regression_difference_term = "residual sum of squares"

theorem regression_difference_is_residual_sum_of_squares :
  regression_difference_term = "residual sum of squares" := by
  sorry

end NUMINAMATH_CALUDE_regression_difference_is_residual_sum_of_squares_l544_54497


namespace NUMINAMATH_CALUDE_jane_earnings_l544_54474

/-- Represents the number of bulbs planted for each flower type -/
structure BulbCounts where
  tulip : ℕ
  iris : ℕ
  hyacinth : ℕ
  daffodil : ℕ
  crocus : ℕ
  gladiolus : ℕ

/-- Represents the price per bulb for each flower type -/
structure BulbPrices where
  tulip : ℚ
  iris : ℚ
  hyacinth : ℚ
  daffodil : ℚ
  crocus : ℚ
  gladiolus : ℚ

def calculateEarnings (counts : BulbCounts) (prices : BulbPrices) : ℚ :=
  counts.tulip * prices.tulip +
  counts.iris * prices.iris +
  counts.hyacinth * prices.hyacinth +
  counts.daffodil * prices.daffodil +
  counts.crocus * prices.crocus +
  counts.gladiolus * prices.gladiolus

theorem jane_earnings (counts : BulbCounts) (prices : BulbPrices) :
  counts.tulip = 20 ∧
  counts.iris = counts.tulip / 2 ∧
  counts.hyacinth = counts.iris + counts.iris / 3 ∧
  counts.daffodil = 30 ∧
  counts.crocus = 3 * counts.daffodil ∧
  counts.gladiolus = 2 * (counts.crocus - counts.daffodil) + (15 * counts.daffodil / 100) ∧
  prices.tulip = 1/2 ∧
  prices.iris = 2/5 ∧
  prices.hyacinth = 3/4 ∧
  prices.daffodil = 1/4 ∧
  prices.crocus = 3/5 ∧
  prices.gladiolus = 3/10
  →
  calculateEarnings counts prices = 12245/100 := by
  sorry


end NUMINAMATH_CALUDE_jane_earnings_l544_54474


namespace NUMINAMATH_CALUDE_fluffy_carrots_l544_54419

def carrot_sequence (first_day : ℕ) : ℕ → ℕ
  | 0 => first_day
  | n + 1 => 2 * carrot_sequence first_day n

def total_carrots (first_day : ℕ) : ℕ :=
  (carrot_sequence first_day 0) + (carrot_sequence first_day 1) + (carrot_sequence first_day 2)

theorem fluffy_carrots (first_day : ℕ) :
  total_carrots first_day = 84 → carrot_sequence first_day 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fluffy_carrots_l544_54419


namespace NUMINAMATH_CALUDE_action_figures_added_l544_54408

theorem action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : 
  initial = 3 → removed = 1 → final = 6 → final - (initial - removed) = 4 :=
by sorry

end NUMINAMATH_CALUDE_action_figures_added_l544_54408


namespace NUMINAMATH_CALUDE_vector_operation_l544_54457

/-- Given vectors a and b, prove that -3a - 2b equals the expected result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, -1)) (h2 : b = (-1, 2)) :
  (-3 : ℝ) • a - (2 : ℝ) • b = (-7, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l544_54457


namespace NUMINAMATH_CALUDE_unique_x_value_l544_54421

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a, b) (c, d) ↦ (a + c, b - d)

/-- The theorem stating the unique value of x -/
theorem unique_x_value : ∃! x : ℤ, ∃ y : ℤ, 
  star (x, y) (3, 3) = star (5, 4) (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l544_54421


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_l544_54453

theorem sqrt_difference_equals_two :
  Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_l544_54453


namespace NUMINAMATH_CALUDE_john_thrice_tom_age_l544_54444

/-- Proves that John was thrice as old as Tom 6 years ago, given the conditions -/
theorem john_thrice_tom_age (tom_current_age john_current_age x : ℕ) : 
  tom_current_age = 16 →
  john_current_age + 4 = 2 * (tom_current_age + 4) →
  john_current_age - x = 3 * (tom_current_age - x) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_thrice_tom_age_l544_54444


namespace NUMINAMATH_CALUDE_solve_sues_library_problem_l544_54475

/-- Represents the number of books and movies Sue has --/
structure LibraryItems where
  books : ℕ
  movies : ℕ

/-- The problem statement about Sue's library items --/
def sues_library_problem (initial_items : LibraryItems) 
  (books_checked_out : ℕ) (final_total : ℕ) : Prop :=
  let movies_returned := initial_items.movies / 3
  let final_movies := initial_items.movies - movies_returned
  let total_books_before_return := initial_items.books + books_checked_out
  let final_books := final_total - final_movies
  let books_returned := total_books_before_return - final_books
  books_returned = 8

/-- Theorem stating the solution to Sue's library problem --/
theorem solve_sues_library_problem :
  sues_library_problem ⟨15, 6⟩ 9 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_sues_library_problem_l544_54475


namespace NUMINAMATH_CALUDE_coin_array_final_row_sum_of_digits_l544_54496

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_array_final_row_sum_of_digits :
  ∃ (n : ℕ), triangular_sum n = 5050 ∧ sum_of_digits n = 1 :=
sorry

end NUMINAMATH_CALUDE_coin_array_final_row_sum_of_digits_l544_54496


namespace NUMINAMATH_CALUDE_john_driving_distance_john_driving_distance_proof_l544_54465

theorem john_driving_distance : ℝ → Prop :=
  fun total_distance =>
    let speed1 : ℝ := 45
    let time1 : ℝ := 2
    let speed2 : ℝ := 50
    let time2 : ℝ := 3
    let distance1 := speed1 * time1
    let distance2 := speed2 * time2
    total_distance = distance1 + distance2 ∧ total_distance = 240

-- Proof
theorem john_driving_distance_proof : ∃ d : ℝ, john_driving_distance d := by
  sorry

end NUMINAMATH_CALUDE_john_driving_distance_john_driving_distance_proof_l544_54465


namespace NUMINAMATH_CALUDE_product_modulo_l544_54463

theorem product_modulo : (1582 * 2031) % 600 = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_l544_54463


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l544_54428

/-- The number of ways to assign volunteers to tasks -/
def assignment_count (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_assignment_count :
  assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l544_54428


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l544_54436

theorem cube_less_than_triple (x : ℤ) : x^3 < 3*x ↔ x = 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l544_54436


namespace NUMINAMATH_CALUDE_milk_water_ratio_l544_54447

theorem milk_water_ratio (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk > 0 ∧ initial_water > 0 →
  initial_milk + initial_water + 8 = 72 →
  (initial_milk + 8) / initial_water = 2 →
  initial_milk / initial_water = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l544_54447


namespace NUMINAMATH_CALUDE_mean_temperature_l544_54486

def temperatures : List ℚ := [75, 80, 78, 82, 85, 90, 87, 84, 88, 93]

theorem mean_temperature : (temperatures.sum / temperatures.length : ℚ) = 421/5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l544_54486


namespace NUMINAMATH_CALUDE_texas_tech_profit_calculation_l544_54459

/-- The amount of money made per t-shirt sold -/
def profit_per_shirt : ℕ := 78

/-- The total number of t-shirts sold during both games -/
def total_shirts : ℕ := 186

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts : ℕ := 172

/-- The money made from selling t-shirts during the Texas Tech game -/
def texas_tech_profit : ℕ := (total_shirts - arkansas_shirts) * profit_per_shirt

theorem texas_tech_profit_calculation : texas_tech_profit = 1092 := by
  sorry

end NUMINAMATH_CALUDE_texas_tech_profit_calculation_l544_54459


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l544_54451

theorem price_ratio_theorem (cost_price : ℝ) (price1 price2 : ℝ) 
  (h1 : price1 = cost_price * (1 + 0.32))
  (h2 : price2 = cost_price * (1 - 0.12)) :
  price2 / price1 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l544_54451


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_two_l544_54429

theorem arithmetic_expression_equals_two :
  10 - 9 + 8 * 7 / 2 - 6 * 5 + 4 - 3 + 2 / 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_two_l544_54429


namespace NUMINAMATH_CALUDE_xyz_sum_product_range_l544_54415

theorem xyz_sum_product_range :
  ∀ x y z : ℝ,
  0 < x ∧ x < 1 →
  0 < y ∧ y < 1 →
  0 < z ∧ z < 1 →
  x + y + z = 2 →
  ∃ S : ℝ, S = x*y + y*z + z*x ∧ 1 < S ∧ S ≤ 4/3 ∧
  ∀ T : ℝ, (∃ a b c : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧
                        a + b + c = 2 ∧ T = a*b + b*c + c*a) →
            1 < T ∧ T ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sum_product_range_l544_54415


namespace NUMINAMATH_CALUDE_unique_solution_l544_54407

/-- The system of equations --/
def system (x y z : ℝ) : Prop :=
  y^2 = 4*x^3 + x - 4 ∧
  z^2 = 4*y^3 + y - 4 ∧
  x^2 = 4*z^3 + z - 4

/-- The theorem stating that (1, 1, 1) is the only solution to the system --/
theorem unique_solution :
  ∀ x y z : ℝ, system x y z → x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l544_54407


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l544_54409

theorem trigonometric_equation_solution :
  ∀ x : Real,
    0 < x →
    x < 180 →
    Real.tan ((150 : Real) * Real.pi / 180 - x * Real.pi / 180) = 
      (Real.sin ((150 : Real) * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
      (Real.cos ((150 : Real) * Real.pi / 180) - Real.cos (x * Real.pi / 180)) →
    x = 120 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l544_54409


namespace NUMINAMATH_CALUDE_least_time_four_horses_meet_l544_54467

def horse_lap_time (k : ℕ) : ℕ := k

def all_horses_lcm : ℕ := 840

theorem least_time_four_horses_meet (T : ℕ) : T = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_time_four_horses_meet_l544_54467


namespace NUMINAMATH_CALUDE_cubic_function_property_l544_54411

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the theorem
theorem cubic_function_property (a b : ℝ) :
  f a b 4 = 0 → f a b (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l544_54411


namespace NUMINAMATH_CALUDE_integer_fraction_property_l544_54464

theorem integer_fraction_property (x y : ℤ) (h : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) :
  ∃ m : ℤ, 4 * x - 3 * y = 5 * m := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_property_l544_54464


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_and_discriminant_multiple_of_four_l544_54420

def polynomial (x : ℂ) : ℂ := x^2 + 6*x + 13

theorem monic_quadratic_with_complex_root_and_discriminant_multiple_of_four :
  (∀ x : ℂ, polynomial x = x^2 + 6*x + 13) ∧
  (polynomial (-3 + 2*I) = 0) ∧
  (∃ k : ℤ, 6^2 - 4*(1:ℝ)*13 = 4*k) :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_and_discriminant_multiple_of_four_l544_54420


namespace NUMINAMATH_CALUDE_three_team_soccer_game_total_score_l544_54493

/-- Represents the score of a team in a half of the game -/
structure HalfScore where
  regular : ℕ
  penalties : ℕ

/-- Represents the score of a team for the whole game -/
structure GameScore where
  first_half : HalfScore
  second_half : HalfScore

/-- Calculate the total score for a team -/
def total_score (score : GameScore) : ℕ :=
  score.first_half.regular + score.first_half.penalties +
  score.second_half.regular + score.second_half.penalties

theorem three_team_soccer_game_total_score :
  let team_a : GameScore := {
    first_half := { regular := 8, penalties := 0 },
    second_half := { regular := 8, penalties := 1 }
  }
  let team_b : GameScore := {
    first_half := { regular := 4, penalties := 0 },
    second_half := { regular := 8, penalties := 2 }
  }
  let team_c : GameScore := {
    first_half := { regular := 8, penalties := 0 },
    second_half := { regular := 11, penalties := 0 }
  }
  team_b.first_half.regular = team_a.first_half.regular / 2 →
  team_c.first_half.regular = 2 * team_b.first_half.regular →
  team_a.second_half.regular = team_c.first_half.regular →
  team_b.second_half.regular = team_a.first_half.regular →
  team_c.second_half.regular = team_b.second_half.regular + 3 →
  total_score team_a + total_score team_b + total_score team_c = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_three_team_soccer_game_total_score_l544_54493


namespace NUMINAMATH_CALUDE_fraction_division_equality_l544_54446

theorem fraction_division_equality : (3 / 8) / (5 / 9) = 27 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l544_54446


namespace NUMINAMATH_CALUDE_system_solution_l544_54470

theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 12 ∧
  x^2 + y^2 + z^2 = 230 ∧
  x * y = -15 →
  ((x = 15 ∧ y = -1 ∧ z = -2) ∨
   (x = -1 ∧ y = 15 ∧ z = -2) ∨
   (x = 3 ∧ y = -5 ∧ z = 14) ∨
   (x = -5 ∧ y = 3 ∧ z = 14)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l544_54470


namespace NUMINAMATH_CALUDE_equation_solution_l544_54431

theorem equation_solution :
  ∀ x : ℚ, (x ≠ 4 ∧ x ≠ -6) →
  ((x + 8) / (x - 4) = (x - 3) / (x + 6) ↔ x = -12 / 7) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l544_54431


namespace NUMINAMATH_CALUDE_perpendicular_solution_parallel_solution_l544_54426

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (x : ℝ) : ℝ × ℝ := (x^2 - 1, x + 1)

-- Define perpendicularity condition
def perpendicular (x : ℝ) : Prop := a.1 * (b x).1 + a.2 * (b x).2 = 0

-- Define parallelism condition
def parallel (x : ℝ) : Prop := a.1 * (b x).2 = a.2 * (b x).1

-- Theorem for perpendicular case
theorem perpendicular_solution :
  ∀ x : ℝ, perpendicular x → x = -1 ∨ x = -2 := by sorry

-- Theorem for parallel case
theorem parallel_solution :
  ∀ x : ℝ, parallel x → 
    ‖(a.1 - (b x).1, a.2 - (b x).2)‖ = Real.sqrt 10 ∨
    ‖(a.1 - (b x).1, a.2 - (b x).2)‖ = 2 * Real.sqrt 10 / 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_solution_parallel_solution_l544_54426


namespace NUMINAMATH_CALUDE_inverse_97_mod_98_l544_54452

theorem inverse_97_mod_98 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 97 ∧ (97 * x) % 98 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_97_mod_98_l544_54452


namespace NUMINAMATH_CALUDE_laser_beam_distance_laser_beam_distance_is_ten_l544_54433

/-- The total distance traveled by a laser beam with given conditions -/
theorem laser_beam_distance : ℝ :=
  let start : ℝ × ℝ := (2, 3)
  let end_point : ℝ × ℝ := (6, 3)
  let reflected_end : ℝ × ℝ := (-6, -3)
  Real.sqrt ((start.1 - reflected_end.1)^2 + (start.2 - reflected_end.2)^2)

/-- Proof that the laser beam distance is 10 -/
theorem laser_beam_distance_is_ten : laser_beam_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_laser_beam_distance_laser_beam_distance_is_ten_l544_54433


namespace NUMINAMATH_CALUDE_insulation_cost_example_l544_54481

/-- Calculates the total cost of insulating a rectangular tank with two layers -/
def insulation_cost (length width height : ℝ) (cost1 cost2 : ℝ) : ℝ :=
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area * (cost1 + cost2)

/-- Theorem: The cost of insulating a 4x5x2 tank with $20 and $15 per sq ft layers is $2660 -/
theorem insulation_cost_example : insulation_cost 4 5 2 20 15 = 2660 := by
  sorry

end NUMINAMATH_CALUDE_insulation_cost_example_l544_54481


namespace NUMINAMATH_CALUDE_max_marked_cells_theorem_marked_cells_property_l544_54471

/-- Represents an equilateral triangle divided into n^2 cells -/
structure DividedTriangle where
  n : ℕ
  cells : ℕ := n^2

/-- Represents the maximum number of cells that can be marked -/
def max_marked_cells (t : DividedTriangle) : ℕ :=
  if t.n = 10 then 7
  else if t.n = 9 then 6
  else 0  -- undefined for other values of n

/-- Theorem stating the maximum number of marked cells for n = 10 and n = 9 -/
theorem max_marked_cells_theorem (t : DividedTriangle) :
  (t.n = 10 → max_marked_cells t = 7) ∧
  (t.n = 9 → max_marked_cells t = 6) := by
  sorry

/-- Represents a strip in the divided triangle -/
structure Strip where
  cells : Finset ℕ

/-- Function to check if two cells are in the same strip -/
def in_same_strip (c1 c2 : ℕ) (s : Strip) : Prop :=
  c1 ∈ s.cells ∧ c2 ∈ s.cells

/-- The main theorem to be proved -/
theorem marked_cells_property (t : DividedTriangle) (marked_cells : Finset ℕ) :
  (∀ (s : Strip), ∀ (c1 c2 : ℕ), c1 ∈ marked_cells → c2 ∈ marked_cells →
    in_same_strip c1 c2 s → c1 = c2) →
  (t.n = 10 → marked_cells.card ≤ 7) ∧
  (t.n = 9 → marked_cells.card ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_marked_cells_theorem_marked_cells_property_l544_54471


namespace NUMINAMATH_CALUDE_circle_distance_bounds_specific_circle_distances_l544_54401

/-- Given a circle with radius r and a point M at distance d from the center,
    returns a pair of the minimum and maximum distances from M to any point on the circle -/
def minMaxDistances (r d : ℝ) : ℝ × ℝ :=
  (r - d, r + d)

theorem circle_distance_bounds (r d : ℝ) (hr : r > 0) (hd : 0 ≤ d ∧ d < r) :
  let (min, max) := minMaxDistances r d
  ∀ p : ℝ × ℝ, (p.1 - r)^2 + p.2^2 = r^2 →
    min^2 ≤ (p.1 - d)^2 + p.2^2 ∧ (p.1 - d)^2 + p.2^2 ≤ max^2 :=
by sorry

theorem specific_circle_distances :
  minMaxDistances 10 3 = (7, 13) :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_bounds_specific_circle_distances_l544_54401


namespace NUMINAMATH_CALUDE_conditional_probability_fair_die_l544_54479

-- Define the sample space
def S : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
def P (X : Finset Nat) : ℚ := (X.card : ℚ) / (S.card : ℚ)

-- Define the intersection of events
def AB : Finset Nat := A ∩ B

-- Theorem statement
theorem conditional_probability_fair_die :
  P AB / P B = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_fair_die_l544_54479


namespace NUMINAMATH_CALUDE_seventh_oblong_number_l544_54412

/-- Defines an oblong number for a given positive integer n -/
def oblong_number (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that the 7th oblong number is 56 -/
theorem seventh_oblong_number : oblong_number 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seventh_oblong_number_l544_54412


namespace NUMINAMATH_CALUDE_nonzero_digits_count_l544_54441

-- Define the fraction
def f : ℚ := 80 / (2^4 * 5^9)

-- Define a function to count non-zero digits after decimal point
noncomputable def count_nonzero_digits_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem nonzero_digits_count :
  count_nonzero_digits_after_decimal f = 3 := by sorry

end NUMINAMATH_CALUDE_nonzero_digits_count_l544_54441


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l544_54480

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 1 = 0) → (x₂^2 + 2*x₂ - 1 = 0) → x₁ + x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l544_54480


namespace NUMINAMATH_CALUDE_red_light_probability_l544_54483

theorem red_light_probability (n : ℕ) (p : ℝ) (h1 : n = 4) (h2 : p = 1/3) :
  let q := 1 - p
  (q * q * p : ℝ) = 4/27 :=
by sorry

end NUMINAMATH_CALUDE_red_light_probability_l544_54483


namespace NUMINAMATH_CALUDE_least_multiple_25_over_500_l544_54416

theorem least_multiple_25_over_500 : 
  ∀ n : ℕ, n > 0 → 25 * n > 500 → 525 ≤ 25 * n :=
sorry

end NUMINAMATH_CALUDE_least_multiple_25_over_500_l544_54416


namespace NUMINAMATH_CALUDE_equation_solution_l544_54432

theorem equation_solution : ∃ (x : ℝ), x^2 - 2*x - 8 = -(x + 2)*(x - 6) ↔ x = 5 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l544_54432


namespace NUMINAMATH_CALUDE_trevor_ride_cost_l544_54448

/-- The total cost of Trevor's taxi ride downtown including the tip -/
def total_cost (uber_cost lyft_cost taxi_cost : ℚ) : ℚ :=
  taxi_cost + 0.2 * taxi_cost

theorem trevor_ride_cost :
  ∀ (uber_cost lyft_cost taxi_cost : ℚ),
    uber_cost = lyft_cost + 3 →
    lyft_cost = taxi_cost + 4 →
    uber_cost = 22 →
    total_cost uber_cost lyft_cost taxi_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_trevor_ride_cost_l544_54448


namespace NUMINAMATH_CALUDE_equilateral_triangle_cosine_l544_54403

/-- An acute angle in degrees -/
def AcuteAngle (x : ℝ) : Prop := 0 < x ∧ x < 90

/-- Cosine function for angles in degrees -/
noncomputable def cosDeg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

/-- Theorem: The only acute angle x (in degrees) that satisfies the conditions for an equilateral triangle with sides cos x, cos x, and cos 5x is 60° -/
theorem equilateral_triangle_cosine (x : ℝ) :
  AcuteAngle x ∧ cosDeg x = cosDeg (5 * x) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cosine_l544_54403


namespace NUMINAMATH_CALUDE_count_numbers_l544_54406

/-- The number of digits available for creating numbers -/
def num_digits : ℕ := 5

/-- The set of digits available for creating numbers -/
def digit_set : Finset ℕ := {0, 1, 2, 3, 4}

/-- The number of digits required for each number -/
def num_places : ℕ := 4

/-- Function to calculate the number of four-digit numbers -/
def four_digit_numbers : ℕ := sorry

/-- Function to calculate the number of four-digit even numbers -/
def four_digit_even_numbers : ℕ := sorry

/-- Function to calculate the number of four-digit numbers without repeating digits -/
def four_digit_no_repeat : ℕ := sorry

/-- Function to calculate the number of four-digit even numbers without repeating digits -/
def four_digit_even_no_repeat : ℕ := sorry

theorem count_numbers :
  four_digit_numbers = 500 ∧
  four_digit_even_numbers = 300 ∧
  four_digit_no_repeat = 96 ∧
  four_digit_even_no_repeat = 60 := by sorry

end NUMINAMATH_CALUDE_count_numbers_l544_54406


namespace NUMINAMATH_CALUDE_system_solution_ratio_l544_54417

theorem system_solution_ratio (a b x y : ℝ) :
  8 * x - 6 * y = a →
  12 * y - 18 * x = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = -4 / 9 := by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l544_54417


namespace NUMINAMATH_CALUDE_stamps_on_last_page_is_four_l544_54499

/-- The number of stamps on the last page of Jenny's seventh book after reorganization --/
def stamps_on_last_page (
  initial_books : ℕ)
  (pages_per_book : ℕ)
  (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ)
  (full_books_after_reorg : ℕ)
  (full_pages_in_last_book : ℕ) : ℕ :=
  let total_stamps := initial_books * pages_per_book * initial_stamps_per_page
  let stamps_in_full_books := full_books_after_reorg * pages_per_book * new_stamps_per_page
  let stamps_in_full_pages_of_last_book := full_pages_in_last_book * new_stamps_per_page
  total_stamps - stamps_in_full_books - stamps_in_full_pages_of_last_book

/-- Theorem stating that under the given conditions, there are 4 stamps on the last page --/
theorem stamps_on_last_page_is_four :
  stamps_on_last_page 10 50 8 12 6 37 = 4 := by
  sorry

end NUMINAMATH_CALUDE_stamps_on_last_page_is_four_l544_54499


namespace NUMINAMATH_CALUDE_kathleen_remaining_money_l544_54442

def kathleen_problem (june_savings july_savings august_savings : ℕ)
                     (school_supplies_cost clothes_cost : ℕ)
                     (aunt_bonus_threshold aunt_bonus : ℕ) : ℕ :=
  let total_savings := june_savings + july_savings + august_savings
  let total_expenses := school_supplies_cost + clothes_cost
  let bonus := if total_savings > aunt_bonus_threshold then aunt_bonus else 0
  total_savings + bonus - total_expenses

theorem kathleen_remaining_money :
  kathleen_problem 21 46 45 12 54 125 25 = 46 := by sorry

end NUMINAMATH_CALUDE_kathleen_remaining_money_l544_54442


namespace NUMINAMATH_CALUDE_sin_cos_transformation_cos_transformation_sin_cos_equiv_transformation_l544_54468

theorem sin_cos_transformation (x : Real) : 
  Real.sqrt 2 * Real.sin x = Real.sqrt 2 * Real.cos (x - Real.pi / 2) :=
by sorry

theorem cos_transformation (x : Real) : 
  Real.sqrt 2 * Real.cos (x - Real.pi / 2) = 
  Real.sqrt 2 * Real.cos ((1/2) * (2*x - Real.pi/4) + Real.pi/4) :=
by sorry

theorem sin_cos_equiv_transformation (x : Real) : 
  Real.sqrt 2 * Real.sin x = 
  Real.sqrt 2 * Real.cos ((1/2) * (2*x - Real.pi/4) + Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_cos_transformation_sin_cos_equiv_transformation_l544_54468


namespace NUMINAMATH_CALUDE_find_set_B_l544_54449

theorem find_set_B (a b : ℝ) : 
  let P : Set ℝ := {1, a/b, b}
  let B : Set ℝ := {0, a+b, b^2}
  P = B → B = {0, -1, 1} := by
sorry

end NUMINAMATH_CALUDE_find_set_B_l544_54449


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_largest_l544_54404

theorem triangle_angle_ratio_largest (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  b = 2 * a →              -- second angle is twice the first
  c = 3 * a →              -- third angle is thrice the first
  max a (max b c) = 90     -- the largest angle is 90°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_largest_l544_54404


namespace NUMINAMATH_CALUDE_prob_at_most_one_for_given_probabilities_l544_54490

/-- The probability that at most one of two independent events occurs, given their individual probabilities -/
def prob_at_most_one (p_a p_b : ℝ) : ℝ :=
  1 - p_a * p_b

theorem prob_at_most_one_for_given_probabilities :
  let p_a := 0.6
  let p_b := 0.7
  prob_at_most_one p_a p_b = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_for_given_probabilities_l544_54490


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l544_54400

/-- Given a mixture where the volume fraction of alcohol is 2/7 and the volume fraction of water is 3/7,
    the ratio of the volume of alcohol to the volume of water is 2:3. -/
theorem alcohol_water_ratio (mixture : ℚ → ℚ) (h1 : mixture 1 = 2/7) (h2 : mixture 2 = 3/7) :
  (mixture 1) / (mixture 2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l544_54400


namespace NUMINAMATH_CALUDE_expression_evaluation_l544_54487

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1
  let expr := ((2 * x - 1/2 * y)^2 - (-y + 2*x) * (2*x + y) + y * (x^2 * y - 5/4 * y)) / x
  expr = -4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l544_54487


namespace NUMINAMATH_CALUDE_inequality_solution_l544_54425

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 4 ∧ x < 3 → 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l544_54425


namespace NUMINAMATH_CALUDE_participant_count_2019_l544_54477

/-- The number of participants in the Science Quiz Bowl for different years --/
structure ParticipantCount where
  y2018 : ℕ
  y2019 : ℕ
  y2020 : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (p : ParticipantCount) : Prop :=
  p.y2018 = 150 ∧
  p.y2020 = p.y2019 / 2 - 40 ∧
  p.y2019 = p.y2020 + 200

/-- The theorem to be proved --/
theorem participant_count_2019 (p : ParticipantCount) 
  (h : satisfiesConditions p) : 
  p.y2019 = 320 ∧ p.y2019 - p.y2018 = 170 :=
by
  sorry

end NUMINAMATH_CALUDE_participant_count_2019_l544_54477


namespace NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l544_54450

/-- Given a quadratic function f(x) = 3x^2 - x + 7, shifting it 5 units to the right
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that the sum of the coefficients a + b + c equals 59. -/
theorem shifted_quadratic_coefficient_sum :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - x + 7
  let g : ℝ → ℝ := λ x ↦ f (x - 5)
  ∃ a b c : ℝ, (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 59) :=
by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l544_54450


namespace NUMINAMATH_CALUDE_constant_zero_function_l544_54422

theorem constant_zero_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f (x + y))^2 = (f x)^2 + (f y)^2) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_constant_zero_function_l544_54422


namespace NUMINAMATH_CALUDE_negation_of_implication_l544_54410

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a - 1 > b - 1) ↔ (a ≤ b → a - 1 ≤ b - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l544_54410


namespace NUMINAMATH_CALUDE_valid_purchase_has_two_notebooks_l544_54455

/-- Represents the purchase of notebooks and books -/
structure Purchase where
  notebooks : ℕ
  books : ℕ
  notebook_cost : ℕ
  book_cost : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : Purchase) : Prop :=
  p.books = p.notebooks + 4 ∧
  p.notebooks * p.notebook_cost = 72 ∧
  p.books * p.book_cost = 660 ∧
  p.notebooks * p.book_cost + p.books * p.notebook_cost < 444

/-- The theorem stating that the valid purchase has 2 notebooks -/
theorem valid_purchase_has_two_notebooks :
  ∃ (p : Purchase), is_valid_purchase p ∧ p.notebooks = 2 :=
sorry


end NUMINAMATH_CALUDE_valid_purchase_has_two_notebooks_l544_54455


namespace NUMINAMATH_CALUDE_eel_length_problem_l544_54492

theorem eel_length_problem (jenna_length bill_length : ℝ) : 
  jenna_length = 16 ∧ jenna_length = (1/3) * bill_length → 
  jenna_length + bill_length = 64 := by
  sorry

end NUMINAMATH_CALUDE_eel_length_problem_l544_54492


namespace NUMINAMATH_CALUDE_fruits_left_after_selling_l544_54491

def initial_oranges : ℕ := 40
def initial_apples : ℕ := 70
def orange_sold_fraction : ℚ := 1/4
def apple_sold_fraction : ℚ := 1/2

theorem fruits_left_after_selling :
  (initial_oranges - orange_sold_fraction * initial_oranges) +
  (initial_apples - apple_sold_fraction * initial_apples) = 65 :=
by sorry

end NUMINAMATH_CALUDE_fruits_left_after_selling_l544_54491


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_l544_54494

/-- Given a rectangular prism with dimensions 16, 4, and 24 inches,
    prove that a cube with the same volume has a surface area of
    approximately 798 square inches. -/
theorem cube_surface_area_from_prism_volume :
  let prism_length : ℝ := 16
  let prism_width : ℝ := 4
  let prism_height : ℝ := 24
  let prism_volume : ℝ := prism_length * prism_width * prism_height
  let cube_edge : ℝ := prism_volume ^ (1/3)
  let cube_surface_area : ℝ := 6 * cube_edge ^ 2
  ∃ ε > 0, |cube_surface_area - 798| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_l544_54494


namespace NUMINAMATH_CALUDE_sweetsies_leftover_l544_54454

theorem sweetsies_leftover (n : ℕ) (h : n % 8 = 5) :
  (3 * n) % 8 = 7 :=
sorry

end NUMINAMATH_CALUDE_sweetsies_leftover_l544_54454


namespace NUMINAMATH_CALUDE_island_navigation_time_l544_54414

/-- The time to navigate around the island once, in minutes -/
def navigation_time : ℕ := 30

/-- The total number of rounds completed over the weekend -/
def total_rounds : ℕ := 26

/-- The total time spent circling the island over the weekend, in minutes -/
def total_time : ℕ := 780

/-- Proof that the navigation time around the island is 30 minutes -/
theorem island_navigation_time :
  navigation_time * total_rounds = total_time :=
by sorry

end NUMINAMATH_CALUDE_island_navigation_time_l544_54414


namespace NUMINAMATH_CALUDE_triangle_max_area_l544_54456

theorem triangle_max_area (a b c : ℝ) (h1 : a + b = 12) (h2 : c = 8) :
  let p := (a + b + c) / 2
  let area := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  ∀ a' b' c', a' + b' = 12 → c' = 8 →
    let p' := (a' + b' + c') / 2
    let area' := Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c'))
    area ≤ area' →
  area = 8 * Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_max_area_l544_54456


namespace NUMINAMATH_CALUDE_optimal_apps_l544_54413

/-- The maximum number of apps Roger can have on his phone for optimal function -/
def max_apps : ℕ := 50

/-- The recommended number of apps -/
def recommended_apps : ℕ := 35

/-- The number of apps Roger currently has -/
def rogers_current_apps : ℕ := 2 * recommended_apps

/-- The number of apps Roger needs to delete -/
def apps_to_delete : ℕ := 20

/-- Theorem stating the maximum number of apps Roger can have for optimal function -/
theorem optimal_apps : max_apps = rogers_current_apps - apps_to_delete := by
  sorry

end NUMINAMATH_CALUDE_optimal_apps_l544_54413


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l544_54489

/-- Given arithmetic sequences a and b with sums S and T, prove the ratio of a_6 to b_8 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  (∀ n, S n = (n + 2) * (T n / (n + 1))) →  -- Condition: S_n / T_n = (n + 2) / (n + 1)
  (∀ n, S (n + 1) - S n = a (n + 1)) →      -- Definition of S as sum of a
  (∀ n, T (n + 1) - T n = b (n + 1)) →      -- Definition of T as sum of b
  (∀ n, a (n + 1) - a n = a 2 - a 1) →      -- a is arithmetic sequence
  (∀ n, b (n + 1) - b n = b 2 - b 1) →      -- b is arithmetic sequence
  a 6 / b 8 = 13 / 16 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l544_54489


namespace NUMINAMATH_CALUDE_window_width_theorem_l544_54488

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions and properties of a window -/
structure Window where
  rows : ℕ
  columns : ℕ
  pane : Pane
  border_width : ℝ

/-- Calculates the total width of the window -/
def total_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating the total width of the window with given conditions -/
theorem window_width_theorem (x : ℝ) : 
  let w : Window := {
    rows := 3,
    columns := 4,
    pane := { width := 4 * x, height := 3 * x },
    border_width := 3
  }
  total_width w = 16 * x + 15 := by sorry

end NUMINAMATH_CALUDE_window_width_theorem_l544_54488


namespace NUMINAMATH_CALUDE_difference_of_squares_l544_54443

theorem difference_of_squares (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l544_54443


namespace NUMINAMATH_CALUDE_ratio_equality_l544_54458

theorem ratio_equality : (2^3001 * 5^3003) / 10^3002 = 5/2 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l544_54458


namespace NUMINAMATH_CALUDE_fair_ticket_cost_amy_ticket_spending_l544_54461

/-- The total cost of tickets at a fair with regular and discounted prices -/
theorem fair_ticket_cost (initial_tickets : ℕ) (additional_tickets : ℕ) 
  (regular_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let initial_cost := initial_tickets * regular_price
  let discount := regular_price * discount_rate
  let discounted_price := regular_price - discount
  let additional_cost := additional_tickets * discounted_price
  initial_cost + additional_cost

/-- Amy's total spending on fair tickets -/
theorem amy_ticket_spending : 
  fair_ticket_cost 33 21 (3/2) (1/4) = 73125/1000 := by
  sorry

end NUMINAMATH_CALUDE_fair_ticket_cost_amy_ticket_spending_l544_54461


namespace NUMINAMATH_CALUDE_base_side_from_sphere_volume_l544_54498

/-- Regular triangular prism with inscribed sphere -/
structure RegularTriangularPrism :=
  (base_side : ℝ)
  (height : ℝ)
  (sphere_volume : ℝ)

/-- The theorem stating the relationship between the inscribed sphere volume
    and the base side length of a regular triangular prism -/
theorem base_side_from_sphere_volume
  (prism : RegularTriangularPrism)
  (h_positive : prism.base_side > 0)
  (h_sphere_volume : prism.sphere_volume = 36 * Real.pi)
  (h_height_eq_diameter : prism.height = 2 * (prism.base_side * Real.sqrt 3 / 6)) :
  prism.base_side = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_base_side_from_sphere_volume_l544_54498


namespace NUMINAMATH_CALUDE_polynomial_sum_l544_54435

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 - x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 81 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l544_54435


namespace NUMINAMATH_CALUDE_pencil_gain_percent_l544_54439

/-- 
Proves that if the cost price of 12 pencils equals the selling price of 8 pencils, 
then the gain percent is 50%.
-/
theorem pencil_gain_percent 
  (cost_price selling_price : ℝ) 
  (h : 12 * cost_price = 8 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_pencil_gain_percent_l544_54439


namespace NUMINAMATH_CALUDE_simplify_expression_l544_54418

theorem simplify_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (16 * x^2 * y^3) / (8 * x * y^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l544_54418


namespace NUMINAMATH_CALUDE_unique_divisible_power_of_two_l544_54460

theorem unique_divisible_power_of_two (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 1997) ∧ (∃ k : ℕ, 2^n + 2 = n * k) ↔ n = 946 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_power_of_two_l544_54460


namespace NUMINAMATH_CALUDE_f_of_five_equals_102_l544_54437

/-- Given a function f(x) = 2x^2 + y where f(2) = 60, prove that f(5) = 102 -/
theorem f_of_five_equals_102 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 60) : 
  f 5 = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_102_l544_54437


namespace NUMINAMATH_CALUDE_trailing_zeros_factorial_product_l544_54430

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The sum of trailing zeros in factorials from 1! to n! -/
def sumTrailingZeros (n : ℕ) : ℕ := sorry

/-- The theorem stating that the number of trailing zeros in the product of factorials 
    from 1! to 50!, when divided by 100, yields a remainder of 14 -/
theorem trailing_zeros_factorial_product : 
  (sumTrailingZeros 50) % 100 = 14 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_factorial_product_l544_54430
