import Mathlib

namespace NUMINAMATH_CALUDE_part1_part2_l95_9556

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (1 / (2^x - 1) + a / 2)

-- Part 1: If f(x) is even and f(1) = 3/2, then f(-1) = 3/2
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → 
  f a 1 = 3/2 → 
  f a (-1) = 3/2 := by sorry

-- Part 2: If a = 1, then f(x) is an even function
theorem part2 : 
  ∀ x : ℝ, x ≠ 0 → f 1 x = f 1 (-x) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l95_9556


namespace NUMINAMATH_CALUDE_sum_25_36_in_base3_l95_9545

/-- Converts a natural number from base 10 to base 3 -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def fromBase3 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_25_36_in_base3 :
  toBase3 (25 + 36) = [2, 0, 2, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_25_36_in_base3_l95_9545


namespace NUMINAMATH_CALUDE_daily_rental_cost_satisfies_conditions_l95_9554

/-- Represents the daily rental cost of a car in dollars -/
def daily_rental_cost : ℝ := 30

/-- Represents the cost per mile in dollars -/
def cost_per_mile : ℝ := 0.18

/-- Represents the total budget in dollars -/
def total_budget : ℝ := 75

/-- Represents the number of miles that can be driven -/
def miles_driven : ℝ := 250

/-- Theorem stating that the daily rental cost satisfies the given conditions -/
theorem daily_rental_cost_satisfies_conditions :
  daily_rental_cost + (cost_per_mile * miles_driven) = total_budget :=
by sorry

end NUMINAMATH_CALUDE_daily_rental_cost_satisfies_conditions_l95_9554


namespace NUMINAMATH_CALUDE_hyperbola_and_asymptotes_l95_9582

/-- Given an ellipse and a hyperbola with the same foci, prove the equation of the hyperbola and its asymptotes -/
theorem hyperbola_and_asymptotes (x y : ℝ) : 
  (∃ (a b c : ℝ), 
    -- Ellipse equation
    (x^2 / 36 + y^2 / 27 = 1) ∧ 
    -- Hyperbola has same foci as ellipse
    (c^2 = a^2 + b^2) ∧ 
    -- Length of conjugate axis of hyperbola
    (2 * b = 4) ∧ 
    -- Foci on x-axis
    (c = 3)) →
  -- Equation of hyperbola
  (x^2 / 5 - y^2 / 4 = 1) ∧
  -- Equations of asymptotes
  (y = (2 * Real.sqrt 5 / 5) * x ∨ y = -(2 * Real.sqrt 5 / 5) * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_asymptotes_l95_9582


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l95_9573

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ (N : ℕ), triangular_sum N = 2211 ∧ sum_of_digits N = 12 :=
sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l95_9573


namespace NUMINAMATH_CALUDE_sum_of_roots_l95_9588

theorem sum_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x / Real.sqrt y - y / Real.sqrt x = 7 / 12)
  (h2 : x - y = 7) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l95_9588


namespace NUMINAMATH_CALUDE_no_common_integer_solutions_l95_9505

theorem no_common_integer_solutions : ¬∃ y : ℤ, (-3 * y ≥ y + 9) ∧ (2 * y ≥ 14) ∧ (-4 * y ≥ 2 * y + 21) := by
  sorry

end NUMINAMATH_CALUDE_no_common_integer_solutions_l95_9505


namespace NUMINAMATH_CALUDE_quadratic_roots_l95_9524

/-- A quadratic function f(x) = ax² - 12ax + 36a - 5 has roots at x = 4 and x = 8 -/
theorem quadratic_roots (a : ℝ) : 
  (∀ x ∈ Set.Ioo 4 5, a * x^2 - 12 * a * x + 36 * a - 5 < 0) →
  (∀ x ∈ Set.Ioo 8 9, a * x^2 - 12 * a * x + 36 * a - 5 > 0) →
  a = 5/4 := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_l95_9524


namespace NUMINAMATH_CALUDE_negation_equivalence_l95_9570

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l95_9570


namespace NUMINAMATH_CALUDE_building_height_problem_l95_9572

theorem building_height_problem (h_taller h_shorter : ℝ) : 
  h_taller - h_shorter = 36 →
  h_shorter / h_taller = 5 / 7 →
  h_taller = 126 := by
  sorry

end NUMINAMATH_CALUDE_building_height_problem_l95_9572


namespace NUMINAMATH_CALUDE_squares_below_line_l95_9551

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer points strictly below a line in the first quadrant -/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem -/
def problemLine : Line :=
  { a := 12, b := 180, c := 2160 }

/-- The theorem statement -/
theorem squares_below_line :
  countPointsBelowLine problemLine = 1969 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_l95_9551


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_2016_l95_9522

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the expression
def expression : ℚ :=
  (factorial 2017 + factorial 2014) / (factorial 2016 + factorial 2015)

-- Theorem statement
theorem floor_of_expression_equals_2016 :
  ⌊expression⌋ = 2016 := by sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_2016_l95_9522


namespace NUMINAMATH_CALUDE_expand_expression_l95_9581

theorem expand_expression (x : ℝ) : (20 * x - 25) * (3 * x) = 60 * x^2 - 75 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l95_9581


namespace NUMINAMATH_CALUDE_vertex_to_center_equals_side_length_l95_9568

/-- A regular hexagon with side length 16 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : Bool)
  (side_length_eq_16 : side_length = 16)

/-- The length of a segment from a vertex to the center of a regular hexagon -/
def vertex_to_center_length (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The length of a segment from a vertex to the center of a regular hexagon
    with side length 16 units is equal to 16 units -/
theorem vertex_to_center_equals_side_length (h : RegularHexagon) :
  vertex_to_center_length h = h.side_length :=
sorry

end NUMINAMATH_CALUDE_vertex_to_center_equals_side_length_l95_9568


namespace NUMINAMATH_CALUDE_field_trip_students_l95_9504

theorem field_trip_students (van_capacity : Nat) (num_vans : Nat) (num_adults : Nat) :
  van_capacity = 8 →
  num_vans = 3 →
  num_adults = 2 →
  (van_capacity * num_vans) - num_adults = 22 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_students_l95_9504


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l95_9509

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7cm, one side is 3cm
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 17 :=  -- Perimeter is 17cm
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l95_9509


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l95_9599

theorem inequality_system_solution_set :
  let S := {x : ℝ | 3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0}
  S = {x : ℝ | -1/3 ≤ x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l95_9599


namespace NUMINAMATH_CALUDE_parabola_directrix_l95_9528

/-- The directrix of a parabola y = ax^2 where a < 0 -/
def directrix_equation (a : ℝ) : ℝ → Prop :=
  fun y => y = -1 / (4 * a)

/-- The parabola equation y = ax^2 -/
def parabola_equation (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y = a * x^2

theorem parabola_directrix (a : ℝ) (h : a < 0) :
  ∃ y, directrix_equation a y ∧
    ∀ x, parabola_equation a x y →
      ∃ p, p > 0 ∧ (x^2 = 4 * p * y) ∧ (y = -p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l95_9528


namespace NUMINAMATH_CALUDE_angle_sum_pi_over_two_l95_9564

theorem angle_sum_pi_over_two (a b : Real) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (eq1 : 2 * Real.sin a ^ 3 + 3 * Real.sin b ^ 2 = 1)
  (eq2 : 2 * Real.sin (3 * a) - 3 * Real.sin (3 * b) = 0) :
  a + 3 * b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_pi_over_two_l95_9564


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l95_9585

/-- Given two concentric circles with radii a and b (a > b), 
    if the area of the ring between them is 16π,
    then the length of a chord of the larger circle 
    that is tangent to the smaller circle is 8. -/
theorem chord_length_concentric_circles 
  (a b : ℝ) (h1 : a > b) (h2 : a^2 - b^2 = 16) : 
  ∃ c : ℝ, c = 8 ∧ c^2 = 4 * (a^2 - b^2) := by
sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l95_9585


namespace NUMINAMATH_CALUDE_sum_binary_digits_365_l95_9534

/-- Sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The sum of the digits in the binary representation of 365 is 6 -/
theorem sum_binary_digits_365 : sumBinaryDigits 365 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_365_l95_9534


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l95_9530

def arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j ≤ n → a j - a i = (j - i : ℝ) * (a 1 - a 0)

theorem sum_of_x_and_y (a : ℕ → ℝ) (n : ℕ) :
  arithmetic_sequence a n ∧ a 0 = 3 ∧ a n = 33 →
  a (n - 1) + a (n - 2) = 48 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l95_9530


namespace NUMINAMATH_CALUDE_average_difference_l95_9529

theorem average_difference (x : ℚ) : 
  (10 + 80 + x) / 3 = (20 + 40 + 60) / 3 - 5 ↔ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l95_9529


namespace NUMINAMATH_CALUDE_log_simplification_l95_9561

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_simplification : (log10 2)^2 + log10 2 * log10 5 + log10 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l95_9561


namespace NUMINAMATH_CALUDE_polynomial_coefficient_values_l95_9543

theorem polynomial_coefficient_values (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₄ = 16 ∧ a₅ = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_values_l95_9543


namespace NUMINAMATH_CALUDE_equiangular_and_equilateral_implies_regular_polygon_l95_9590

/-- A figure is equiangular if all its angles are equal. -/
def IsEquiangular (figure : Type) : Prop := sorry

/-- A figure is equilateral if all its sides are equal. -/
def IsEquilateral (figure : Type) : Prop := sorry

/-- A figure is a regular polygon if it is both equiangular and equilateral. -/
def IsRegularPolygon (figure : Type) : Prop := 
  IsEquiangular figure ∧ IsEquilateral figure

/-- Theorem: If a figure is both equiangular and equilateral, then it is a regular polygon. -/
theorem equiangular_and_equilateral_implies_regular_polygon 
  (figure : Type) 
  (h1 : IsEquiangular figure) 
  (h2 : IsEquilateral figure) : 
  IsRegularPolygon figure := by
  sorry


end NUMINAMATH_CALUDE_equiangular_and_equilateral_implies_regular_polygon_l95_9590


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l95_9520

/-- Proves that y = 3 given the conditions of the problem -/
theorem consecutive_numbers_problem (x y z : ℤ) 
  (h1 : x = z + 2) 
  (h2 : y = z + 1) 
  (h3 : x > y ∧ y > z) 
  (h4 : 2*x + 3*y + 3*z = 5*y + 8) 
  (h5 : z = 2) : 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l95_9520


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_product_of_sum_products_eq_27_l95_9591

theorem sum_geq_sqrt_product_of_sum_products_eq_27
  (x y z : ℝ)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z)
  (sum_products : x * y + y * z + z * x = 27) :
  x + y + z ≥ Real.sqrt (3 * x * y * z) ∧
  (x + y + z = Real.sqrt (3 * x * y * z) ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_product_of_sum_products_eq_27_l95_9591


namespace NUMINAMATH_CALUDE_player1_can_win_l95_9560

/-- Represents a square on the game board -/
structure Square where
  x : Fin 2021
  y : Fin 2021

/-- Represents a domino placement on the game board -/
structure Domino where
  square1 : Square
  square2 : Square

/-- The game state -/
structure GameState where
  board : Fin 2021 → Fin 2021 → Bool
  dominoes : List Domino

/-- A player's strategy -/
def Strategy := GameState → Domino

/-- The game play function -/
def play (player1Strategy player2Strategy : Strategy) : GameState :=
  sorry

theorem player1_can_win :
  ∃ (player1Strategy : Strategy),
    ∀ (player2Strategy : Strategy),
      let finalState := play player1Strategy player2Strategy
      ∃ (s1 s2 : Square), s1 ≠ s2 ∧ finalState.board s1.x s1.y = false ∧ finalState.board s2.x s2.y = false :=
  sorry


end NUMINAMATH_CALUDE_player1_can_win_l95_9560


namespace NUMINAMATH_CALUDE_unique_fixed_point_of_f_and_f_inv_l95_9500

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_fixed_point_of_f_and_f_inv :
  ∃! x : ℝ, f x = f_inv x :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_of_f_and_f_inv_l95_9500


namespace NUMINAMATH_CALUDE_pen_purchase_theorem_l95_9547

def budget : ℕ := 31
def price1 : ℕ := 2
def price2 : ℕ := 3
def price3 : ℕ := 4

def max_pens (b p1 p2 p3 : ℕ) : ℕ :=
  (b - p2 - p3) / p1 + 2

def min_pens (b p1 p2 p3 : ℕ) : ℕ :=
  (b - p1 - p2) / p3 + 3

theorem pen_purchase_theorem :
  max_pens budget price1 price2 price3 = 14 ∧
  min_pens budget price1 price2 price3 = 9 :=
by sorry

end NUMINAMATH_CALUDE_pen_purchase_theorem_l95_9547


namespace NUMINAMATH_CALUDE_sandal_price_proof_l95_9583

/-- Proves that the price of each pair of sandals is $3 given the conditions of Yanna's purchase. -/
theorem sandal_price_proof (num_shirts : ℕ) (shirt_price : ℕ) (num_sandals : ℕ) (bill_paid : ℕ) (change_received : ℕ) :
  num_shirts = 10 →
  shirt_price = 5 →
  num_sandals = 3 →
  bill_paid = 100 →
  change_received = 41 →
  (bill_paid - change_received - num_shirts * shirt_price) / num_sandals = 3 :=
by sorry

end NUMINAMATH_CALUDE_sandal_price_proof_l95_9583


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l95_9546

theorem negation_of_universal_quantifier :
  (¬ (∀ x : ℝ, x^2 - x + 1/4 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l95_9546


namespace NUMINAMATH_CALUDE_wang_processing_time_l95_9562

/-- Given that Master Wang processes 92 parts in 4 days, 
    prove that it takes 9 days to process 207 parts using proportion. -/
theorem wang_processing_time 
  (parts_per_four_days : ℕ) 
  (h_parts : parts_per_four_days = 92) 
  (new_parts : ℕ) 
  (h_new_parts : new_parts = 207) : 
  (4 : ℚ) * new_parts / parts_per_four_days = 9 := by
  sorry

end NUMINAMATH_CALUDE_wang_processing_time_l95_9562


namespace NUMINAMATH_CALUDE_first_class_males_count_l95_9548

/-- Represents the number of male students in the first class -/
def first_class_males : ℕ := sorry

/-- Represents the number of female students in the first class -/
def first_class_females : ℕ := 13

/-- Represents the number of male students in the second class -/
def second_class_males : ℕ := 14

/-- Represents the number of female students in the second class -/
def second_class_females : ℕ := 18

/-- Represents the number of male students in the third class -/
def third_class_males : ℕ := 15

/-- Represents the number of female students in the third class -/
def third_class_females : ℕ := 17

/-- Represents the number of students unable to partner with the opposite gender -/
def unpartnered_students : ℕ := 2

theorem first_class_males_count : first_class_males = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_class_males_count_l95_9548


namespace NUMINAMATH_CALUDE_chinese_space_station_altitude_l95_9502

theorem chinese_space_station_altitude :
  ∃ (n : ℝ), n = 389000 ∧ n = 3.89 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_chinese_space_station_altitude_l95_9502


namespace NUMINAMATH_CALUDE_newlandia_density_l95_9593

/-- Represents the population and area data for a country -/
structure CountryData where
  population : ℕ
  area_sq_miles : ℕ

/-- Calculates the average square feet per person in a country -/
def avg_sq_feet_per_person (country : CountryData) : ℚ :=
  (country.area_sq_miles * (5280 * 5280) : ℚ) / country.population

/-- Theorem stating the properties of Newlandia's population density -/
theorem newlandia_density (newlandia : CountryData) 
  (h1 : newlandia.population = 350000000)
  (h2 : newlandia.area_sq_miles = 4500000) :
  let density := avg_sq_feet_per_person newlandia
  (358000 : ℚ) < density ∧ density < (359000 : ℚ) ∧ density > 700 := by
  sorry

#eval avg_sq_feet_per_person ⟨350000000, 4500000⟩

end NUMINAMATH_CALUDE_newlandia_density_l95_9593


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l95_9553

/-- Given an arithmetic sequence {a_n} with a_2 = 7 and a_11 = a_9 + 6, prove a_1 = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) → -- arithmetic sequence condition
  a 2 = 7 →
  a 11 = a 9 + 6 →
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l95_9553


namespace NUMINAMATH_CALUDE_fraction_multiplication_l95_9550

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 * 5 / 8 = 15 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l95_9550


namespace NUMINAMATH_CALUDE_not_prime_n_squared_plus_75_l95_9517

theorem not_prime_n_squared_plus_75 (n : ℕ) (h : Nat.Prime n) : ¬ Nat.Prime (n^2 + 75) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n_squared_plus_75_l95_9517


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l95_9596

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_3 : a 3 = 16)
  (h_9 : a 9 = 80) :
  a 6 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l95_9596


namespace NUMINAMATH_CALUDE_tangent_line_equation_l95_9592

theorem tangent_line_equation (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let slope : ℝ := -a * Real.sin (π / 6)
  slope = 1 / 2 →
  let x₀ : ℝ := π / 6
  let y₀ : ℝ := f x₀
  ∀ x y : ℝ, (y - y₀ = slope * (x - x₀)) ↔ (x - 2 * y - Real.sqrt 3 - π / 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l95_9592


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l95_9532

/-- Given a triangle with specific properties, prove its smallest side length -/
theorem smallest_side_of_triangle (S : ℝ) (p : ℝ) (d : ℝ) :
  S = 6 * Real.sqrt 6 →
  p = 18 →
  d = (2 * Real.sqrt 42) / 3 →
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = p ∧
    S = Real.sqrt (p/2 * (p/2 - a) * (p/2 - b) * (p/2 - c)) ∧
    d^2 = ((p/2 - b) * (p/2 - c) / (p/2))^2 + (S / p)^2 ∧
    min a (min b c) = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l95_9532


namespace NUMINAMATH_CALUDE_dragon_invincible_l95_9523

-- Define the possible head-cutting operations
inductive CutOperation
| cut13 : CutOperation
| cut17 : CutOperation
| cut6 : CutOperation

-- Define the state of the dragon
structure DragonState :=
  (heads : ℕ)

-- Define the rules for head regeneration
def regenerate (s : DragonState) : DragonState :=
  match s.heads with
  | 1 => ⟨8⟩  -- 1 + 7 regenerated
  | 2 => ⟨13⟩ -- 2 + 11 regenerated
  | 3 => ⟨12⟩ -- 3 + 9 regenerated
  | n => s

-- Define a single step of the process (cutting and potential regeneration)
def step (s : DragonState) (op : CutOperation) : DragonState :=
  let s' := match op with
    | CutOperation.cut13 => ⟨s.heads - min s.heads 13⟩
    | CutOperation.cut17 => ⟨s.heads - min s.heads 17⟩
    | CutOperation.cut6 => ⟨s.heads - min s.heads 6⟩
  regenerate s'

-- Define the theorem
theorem dragon_invincible :
  ∀ (ops : List CutOperation),
    let final_state := ops.foldl step ⟨100⟩
    final_state.heads > 0 ∨ final_state.heads ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_dragon_invincible_l95_9523


namespace NUMINAMATH_CALUDE_captain_birth_year_is_1938_l95_9511

/-- Represents the ages of the crew members -/
structure CrewAges where
  sailor : ℕ
  cook : ℕ
  engineer : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : CrewAges) : Prop :=
  Odd ages.cook ∧
  ¬Odd ages.sailor ∧
  ¬Odd ages.engineer ∧
  ages.engineer = ages.sailor + 4 ∧
  ages.cook = 3 * (ages.sailor / 2) ∧
  ages.sailor = 2 * (ages.sailor / 2)

/-- The captain's birth year is the LCM of the crew's ages -/
def captainBirthYear (ages : CrewAges) : ℕ :=
  Nat.lcm ages.sailor (Nat.lcm ages.cook ages.engineer)

/-- The main theorem stating that the captain's birth year is 1938 -/
theorem captain_birth_year_is_1938 :
  ∃ ages : CrewAges, satisfiesConditions ages ∧ captainBirthYear ages = 1938 :=
sorry

end NUMINAMATH_CALUDE_captain_birth_year_is_1938_l95_9511


namespace NUMINAMATH_CALUDE_largest_divisor_of_Q_l95_9518

/-- Q is the product of two consecutive even numbers and their immediate preceding odd integer -/
def Q (n : ℕ) : ℕ := (2*n - 1) * (2*n) * (2*n + 2)

/-- 8 is the largest integer that divides Q for all n -/
theorem largest_divisor_of_Q :
  ∀ k : ℕ, (∀ n : ℕ, k ∣ Q n) → k ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_Q_l95_9518


namespace NUMINAMATH_CALUDE_simplify_expression_l95_9563

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 + b*(2*b+1) = 9*b^3 + 6*b^2 + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l95_9563


namespace NUMINAMATH_CALUDE_evaluate_expression_l95_9544

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 2) : y * (y - 3 * x) = -14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l95_9544


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l95_9571

-- Define the parabola function
def f (x : ℝ) (m : ℝ) : ℝ := -3 * x^2 - 12 * x + m

-- Define the theorem
theorem parabola_y_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-3) m = y₁)
  (h₂ : f (-2) m = y₂)
  (h₃ : f 1 m = y₃) :
  y₂ > y₁ ∧ y₁ > y₃ :=
sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l95_9571


namespace NUMINAMATH_CALUDE_NaHCO3_moles_equal_H2O_moles_NaHCO3_moles_proof_l95_9574

-- Define the molar masses and quantities
def molar_mass_H2O : ℝ := 18
def HNO3_moles : ℝ := 2
def H2O_grams : ℝ := 36

-- Define the reaction stoichiometry
def HNO3_to_H2O_ratio : ℝ := 1
def NaHCO3_to_H2O_ratio : ℝ := 1

-- Theorem statement
theorem NaHCO3_moles_equal_H2O_moles : ℝ → Prop :=
  fun NaHCO3_moles =>
    let H2O_moles := H2O_grams / molar_mass_H2O
    NaHCO3_moles = H2O_moles ∧ NaHCO3_moles = HNO3_moles

-- Proof (skipped)
theorem NaHCO3_moles_proof : ∃ (x : ℝ), NaHCO3_moles_equal_H2O_moles x :=
sorry

end NUMINAMATH_CALUDE_NaHCO3_moles_equal_H2O_moles_NaHCO3_moles_proof_l95_9574


namespace NUMINAMATH_CALUDE_alpha_minus_beta_eq_pi_fourth_l95_9514

theorem alpha_minus_beta_eq_pi_fourth 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (π/4) (π/2)) 
  (h2 : β ∈ Set.Icc (π/4) (π/2)) 
  (h3 : Real.sin α + Real.cos α = Real.sqrt 2 * Real.cos β) : 
  α - β = π/4 := by
sorry

end NUMINAMATH_CALUDE_alpha_minus_beta_eq_pi_fourth_l95_9514


namespace NUMINAMATH_CALUDE_product_and_power_constraint_l95_9519

theorem product_and_power_constraint (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1)
  (h_product : a * b * c = 10)
  (h_power : a^(Real.log a) * b^(Real.log b) * c^(Real.log c) ≥ 10) :
  (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) :=
by sorry

end NUMINAMATH_CALUDE_product_and_power_constraint_l95_9519


namespace NUMINAMATH_CALUDE_mary_garden_potatoes_l95_9533

/-- The number of potatoes left in Mary's garden after planting and rabbit eating -/
def potatoes_left (initial : ℕ) (added : ℕ) (eaten : ℕ) : ℕ :=
  let rows := initial
  let per_row := 1 + added
  max (rows * per_row - rows * eaten) 0

theorem mary_garden_potatoes :
  potatoes_left 8 2 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mary_garden_potatoes_l95_9533


namespace NUMINAMATH_CALUDE_yellow_face_probability_l95_9569

/-- The probability of rolling a yellow face on a 12-sided die with 4 yellow faces is 1/3 -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) 
  (h1 : total_faces = 12) (h2 : yellow_faces = 4) : 
  (yellow_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l95_9569


namespace NUMINAMATH_CALUDE_evaluate_expression_l95_9577

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l95_9577


namespace NUMINAMATH_CALUDE_number_equation_solution_l95_9584

theorem number_equation_solution : 
  ∃ x : ℝ, (5020 - (x / 20.08) = 4970) ∧ (x = 1004) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l95_9584


namespace NUMINAMATH_CALUDE_max_gcd_2015_l95_9540

theorem max_gcd_2015 (x y : ℤ) (h : Int.gcd x y = 1) :
  (∃ a b : ℤ, Int.gcd (a + 2015 * b) (b + 2015 * a) = 4060224) ∧
  (∀ c d : ℤ, Int.gcd (c + 2015 * d) (d + 2015 * c) ≤ 4060224) := by
sorry

end NUMINAMATH_CALUDE_max_gcd_2015_l95_9540


namespace NUMINAMATH_CALUDE_quartic_root_product_l95_9580

theorem quartic_root_product (k : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^4 - 18*a^3 + k*a^2 + 200*a - 1984 = 0) ∧
    (b^4 - 18*b^3 + k*b^2 + 200*b - 1984 = 0) ∧
    (c^4 - 18*c^3 + k*c^2 + 200*c - 1984 = 0) ∧
    (d^4 - 18*d^3 + k*d^2 + 200*d - 1984 = 0) ∧
    (a * b = -32 ∨ a * c = -32 ∨ a * d = -32 ∨ b * c = -32 ∨ b * d = -32 ∨ c * d = -32)) →
  k = 86 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_product_l95_9580


namespace NUMINAMATH_CALUDE_a_left_after_three_days_l95_9586

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 21

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 28

/-- The number of days B worked alone to complete the remaining work -/
def b_remaining_days : ℝ := 21

/-- The number of days A worked before leaving -/
def x : ℝ := 3

theorem a_left_after_three_days :
  (x / (a_days⁻¹ + b_days⁻¹)⁻¹) + (b_remaining_days / b_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_a_left_after_three_days_l95_9586


namespace NUMINAMATH_CALUDE_geraldine_dolls_count_l95_9575

theorem geraldine_dolls_count (jazmin_dolls total_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209)
  (h2 : total_dolls = 3395) :
  total_dolls - jazmin_dolls = 2186 :=
by sorry

end NUMINAMATH_CALUDE_geraldine_dolls_count_l95_9575


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l95_9559

/-- Represents the ages of three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem david_scott_age_difference (ages : BrothersAges) 
  (h : satisfiesConditions ages) : ages.david - ages.scott = 8 := by
  sorry

end NUMINAMATH_CALUDE_david_scott_age_difference_l95_9559


namespace NUMINAMATH_CALUDE_bill_john_score_difference_l95_9587

-- Define the scores as natural numbers
def bill_score : ℕ := 45
def sue_score : ℕ := bill_score * 2
def john_score : ℕ := 160 - bill_score - sue_score

-- Theorem statement
theorem bill_john_score_difference :
  bill_score > john_score ∧
  bill_score = sue_score / 2 ∧
  bill_score + john_score + sue_score = 160 →
  bill_score - john_score = 20 := by
sorry

end NUMINAMATH_CALUDE_bill_john_score_difference_l95_9587


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l95_9594

theorem least_positive_integer_satisfying_congruences : ∃ x : ℕ, x > 0 ∧
  ((x : ℤ) + 127 ≡ 53 [ZMOD 15]) ∧
  ((x : ℤ) + 104 ≡ 76 [ZMOD 7]) ∧
  (∀ y : ℕ, y > 0 →
    ((y : ℤ) + 127 ≡ 53 [ZMOD 15]) →
    ((y : ℤ) + 104 ≡ 76 [ZMOD 7]) →
    x ≤ y) ∧
  x = 91 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l95_9594


namespace NUMINAMATH_CALUDE_quadratic_solution_l95_9589

theorem quadratic_solution (b : ℚ) : 
  ((-4 : ℚ)^2 + b * (-4) - 45 = 0) → b = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l95_9589


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l95_9535

theorem triangle_angle_measure (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : Y = 2 * X) : Z = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l95_9535


namespace NUMINAMATH_CALUDE_grocery_solution_l95_9579

/-- Represents the grocery shopping problem --/
def grocery_problem (initial_money : ℝ) (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
  (pasta_price : ℝ) (pasta_quantity : ℝ) (sauce_price : ℝ) (money_left : ℝ) : Prop :=
  let total_spent := initial_money - money_left
  let mustard_oil_cost := mustard_oil_price * mustard_oil_quantity
  let pasta_cost := pasta_price * pasta_quantity
  let sauce_cost := total_spent - mustard_oil_cost - pasta_cost
  sauce_cost / sauce_price = 1

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution :
  grocery_problem 50 13 2 4 3 5 7 := by
  sorry

#check grocery_solution

end NUMINAMATH_CALUDE_grocery_solution_l95_9579


namespace NUMINAMATH_CALUDE_max_pencils_purchased_l95_9512

/-- Given a pencil price of 25 cents and $50 available, 
    prove that the maximum number of pencils that can be purchased is 200. -/
theorem max_pencils_purchased (pencil_price : ℕ) (available_money : ℕ) :
  pencil_price = 25 →
  available_money = 5000 →
  (∀ n : ℕ, n * pencil_price ≤ available_money → n ≤ 200) ∧
  200 * pencil_price ≤ available_money :=
by
  sorry

#check max_pencils_purchased

end NUMINAMATH_CALUDE_max_pencils_purchased_l95_9512


namespace NUMINAMATH_CALUDE_jimmys_father_emails_l95_9515

/-- The number of emails Jimmy's father received per day before subscribing to the news channel -/
def initial_emails_per_day : ℕ := 20

/-- The number of additional emails per day after subscribing to the news channel -/
def additional_emails : ℕ := 5

/-- The total number of days in April -/
def total_days : ℕ := 30

/-- The day Jimmy's father subscribed to the news channel -/
def subscription_day : ℕ := 15

/-- The total number of emails Jimmy's father received in April -/
def total_emails : ℕ := 675

theorem jimmys_father_emails :
  initial_emails_per_day * subscription_day +
  (initial_emails_per_day + additional_emails) * (total_days - subscription_day) =
  total_emails :=
by
  sorry

#check jimmys_father_emails

end NUMINAMATH_CALUDE_jimmys_father_emails_l95_9515


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l95_9503

/-- Given a cake recipe that requires 12 cups of flour, and knowing that Mary still needs 2 more cups,
    prove that Mary has already put in 10 cups of flour. -/
theorem mary_flour_calculation (recipe_flour : ℕ) (flour_needed : ℕ) (flour_put_in : ℕ) : 
  recipe_flour = 12 → flour_needed = 2 → flour_put_in = recipe_flour - flour_needed := by
  sorry

#check mary_flour_calculation

end NUMINAMATH_CALUDE_mary_flour_calculation_l95_9503


namespace NUMINAMATH_CALUDE_roots_product_equation_l95_9536

theorem roots_product_equation (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 4 = 0) 
  (h2 : β^2 + p*β + 4 = 0)
  (h3 : γ^2 + q*γ + 4 = 0)
  (h4 : δ^2 + q*δ + 4 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -4 * (p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_product_equation_l95_9536


namespace NUMINAMATH_CALUDE_factorial_sum_ratio_l95_9565

theorem factorial_sum_ratio (N : ℕ) (h : N > 0) : 
  (Nat.factorial (N + 1) + Nat.factorial (N - 1)) / Nat.factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3*N^2 + 2*N) := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_ratio_l95_9565


namespace NUMINAMATH_CALUDE_car_rental_problem_l95_9508

/-- Represents a car rental company -/
structure Company where
  totalCars : Nat
  baseRent : ℝ
  rentIncrease : ℝ
  maintenanceFee : ℝ → ℝ

/-- Calculates the profit for a company given the number of cars rented -/
def profit (c : Company) (rented : ℝ) : ℝ :=
  (c.baseRent + (c.totalCars - rented) * c.rentIncrease) * rented - c.maintenanceFee rented

/-- Company A as described in the problem -/
def companyA : Company :=
  { totalCars := 50
  , baseRent := 3000
  , rentIncrease := 50
  , maintenanceFee := λ x => 200 * x }

/-- Company B as described in the problem -/
def companyB : Company :=
  { totalCars := 50
  , baseRent := 3500
  , rentIncrease := 0
  , maintenanceFee := λ _ => 1850 }

theorem car_rental_problem :
  (profit companyA 10 = 48000) ∧
  (∃ x : ℝ, x = 37 ∧ profit companyA x = profit companyB x) ∧
  (∃ max : ℝ, max = 33150 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → |profit companyA x - profit companyB x| ≤ max) ∧
  (∀ a : ℝ, 50 < a ∧ a < 150 ↔
    (let f := λ x => profit companyA x - a * x - profit companyB x
     ∃ max : ℝ, max = f 17 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → f x ≤ max ∧ f x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_car_rental_problem_l95_9508


namespace NUMINAMATH_CALUDE_larger_number_proof_l95_9552

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 4 * S + 15) : L = 1815 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l95_9552


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l95_9538

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + 3 = 2 * x + 1

/-- If the parabola y = ax^2 + 3 is tangent to the line y = 2x + 1, then a = 1/2. -/
theorem parabola_tangent_line (a : ℝ) : is_tangent a → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l95_9538


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l95_9513

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l95_9513


namespace NUMINAMATH_CALUDE_function_comparison_l95_9597

theorem function_comparison (a b : ℝ) (f g : ℝ → ℝ) 
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_deriv : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (h_eq : f a = g a)
  (h_le : a ≤ b) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x :=
sorry

end NUMINAMATH_CALUDE_function_comparison_l95_9597


namespace NUMINAMATH_CALUDE_probability_of_different_colors_l95_9501

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 3

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := 4

/-- The number of white marbles in the jar -/
def white_marbles : ℕ := 5

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := 3

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + green_marbles + white_marbles + blue_marbles

/-- The number of ways to choose 2 marbles from the total number of marbles -/
def total_ways : ℕ := total_marbles.choose 2

/-- The number of ways to choose 2 marbles of different colors -/
def different_color_ways : ℕ :=
  red_marbles * green_marbles +
  red_marbles * white_marbles +
  red_marbles * blue_marbles +
  green_marbles * white_marbles +
  green_marbles * blue_marbles +
  white_marbles * blue_marbles

/-- The probability of drawing two marbles of different colors -/
def probability : ℚ := different_color_ways / total_ways

theorem probability_of_different_colors :
  probability = 83 / 105 :=
sorry

end NUMINAMATH_CALUDE_probability_of_different_colors_l95_9501


namespace NUMINAMATH_CALUDE_simplify_expression_l95_9510

theorem simplify_expression : 0.3 * 0.8 + 0.1 * 0.5 = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l95_9510


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l95_9557

/-- Represents the bus driver's compensation structure and work hours -/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation based on the given rates and hours -/
def calculateTotalCompensation (c : BusDriverCompensation) : ℝ :=
  c.regularRate * c.regularHours + c.overtimeRate * c.overtimeHours

/-- Theorem stating that the bus driver's regular rate is $16 per hour -/
theorem bus_driver_regular_rate :
  ∃ (c : BusDriverCompensation),
    c.regularHours = 40 ∧
    c.overtimeHours = 8 ∧
    c.overtimeRate = 1.75 * c.regularRate ∧
    c.totalCompensation = 864 ∧
    calculateTotalCompensation c = c.totalCompensation ∧
    c.regularRate = 16 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_regular_rate_l95_9557


namespace NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l95_9542

theorem scientific_notation_of_nine_billion :
  9000000000 = 9 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l95_9542


namespace NUMINAMATH_CALUDE_curve_represents_two_points_l95_9506

theorem curve_represents_two_points :
  ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
  (∀ (x y : ℝ), ((x - y)^2 + (x*y - 1)^2 = 0) ↔ (x, y) = p1 ∨ (x, y) = p2) :=
sorry

end NUMINAMATH_CALUDE_curve_represents_two_points_l95_9506


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_l95_9595

theorem triangle_angle_cosine (A B C : ℝ) (a b c : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) 
  (h4 : B = 2 * C) (h5 : A + B + C = π) : Real.cos C = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_l95_9595


namespace NUMINAMATH_CALUDE_complex_simplification_l95_9537

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  6 * (4 - 2*i) + 2*i * (7 - 3*i) = 30 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l95_9537


namespace NUMINAMATH_CALUDE_continuous_third_derivative_product_nonnegative_l95_9549

/-- A real function with continuous third derivative has a point where the product of
    the function value and its first three derivatives is non-negative. -/
theorem continuous_third_derivative_product_nonnegative (f : ℝ → ℝ) 
  (hf : ContDiff ℝ 3 f) :
  ∃ a : ℝ, f a * (deriv f a) * (deriv^[2] f a) * (deriv^[3] f a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_continuous_third_derivative_product_nonnegative_l95_9549


namespace NUMINAMATH_CALUDE_probability_less_than_4_is_7_9_l95_9541

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies x + y < 4 --/
def probabilityLessThan4 (s : Square) : ℝ :=
  sorry

/-- Our specific square with vertices (0,0), (0,3), (3,3), and (3,0) --/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

theorem probability_less_than_4_is_7_9 :
  probabilityLessThan4 specificSquare = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_4_is_7_9_l95_9541


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_l95_9527

-- Define a monic polynomial of degree 2
def monicQuadratic (b c : ℝ) : ℝ → ℝ := λ x => x^2 + b*x + c

-- State the theorem
theorem unique_monic_quadratic (g : ℝ → ℝ) :
  (∃ b c : ℝ, ∀ x, g x = monicQuadratic b c x) →  -- g is a monic quadratic polynomial
  g 0 = 6 →                                       -- g(0) = 6
  g 1 = 8 →                                       -- g(1) = 8
  ∀ x, g x = x^2 + x + 6 :=                       -- Conclusion: g(x) = x^2 + x + 6
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_unique_monic_quadratic_l95_9527


namespace NUMINAMATH_CALUDE_cosine_shift_equals_sine_l95_9576

open Real

theorem cosine_shift_equals_sine (m : ℝ) : (∀ x, cos (x + m) = sin x) → m = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_shift_equals_sine_l95_9576


namespace NUMINAMATH_CALUDE_jovana_shell_weight_l95_9531

/-- Proves that the total weight of shells in Jovana's bucket is approximately 11.29 pounds -/
theorem jovana_shell_weight (initial_weight : ℝ) (large_shell_weight : ℝ) (additional_weight : ℝ) 
  (conversion_rate : ℝ) (h1 : initial_weight = 5.25) (h2 : large_shell_weight = 700) 
  (h3 : additional_weight = 4.5) (h4 : conversion_rate = 453.592) : 
  ∃ (total_weight : ℝ), abs (total_weight - 11.29) < 0.01 ∧ 
  total_weight = initial_weight + (large_shell_weight / conversion_rate) + additional_weight :=
sorry

end NUMINAMATH_CALUDE_jovana_shell_weight_l95_9531


namespace NUMINAMATH_CALUDE_door_purchase_savings_l95_9555

/-- Calculates the cost of purchasing doors with the "buy 3 get 1 free" offer -/
def cost_with_offer (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  ((num_doors + 3) / 4) * 3 * price_per_door

/-- Calculates the regular cost of purchasing doors without any offer -/
def regular_cost (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  num_doors * price_per_door

/-- Calculates the savings when purchasing doors with the offer -/
def savings (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  regular_cost num_doors price_per_door - cost_with_offer num_doors price_per_door

theorem door_purchase_savings :
  let alice_doors := 6
  let bob_doors := 9
  let price_per_door := 120
  let total_doors := alice_doors + bob_doors
  savings total_doors price_per_door = 600 :=
by sorry

end NUMINAMATH_CALUDE_door_purchase_savings_l95_9555


namespace NUMINAMATH_CALUDE_chessboard_color_swap_theorem_l95_9578

/-- A color is represented by a natural number -/
def Color := ℕ

/-- A chessboard is represented by a function from coordinates to colors -/
def Chessboard (n : ℕ) := Fin (2*n) → Fin (2*n) → Color

/-- A rectangle on the chessboard is defined by its corner coordinates -/
structure Rectangle (n : ℕ) where
  i1 : Fin (2*n)
  j1 : Fin (2*n)
  i2 : Fin (2*n)
  j2 : Fin (2*n)

/-- Predicate to check if all corners of a rectangle have the same color -/
def same_color_corners (board : Chessboard n) (rect : Rectangle n) : Prop :=
  board rect.i1 rect.j1 = board rect.i1 rect.j2 ∧
  board rect.i1 rect.j1 = board rect.i2 rect.j1 ∧
  board rect.i1 rect.j1 = board rect.i2 rect.j2

/-- Main theorem: There exist two tiles in the same column such that swapping
    their colors creates a rectangle with all four corners of the same color -/
theorem chessboard_color_swap_theorem (n : ℕ) (board : Chessboard n) :
  ∃ (i1 i2 j : Fin (2*n)) (rect : Rectangle n),
    i1 ≠ i2 ∧
    (∀ (i : Fin (2*n)), board i j ≠ board i1 j → board i j = board i2 j) →
    same_color_corners board rect :=
  sorry

end NUMINAMATH_CALUDE_chessboard_color_swap_theorem_l95_9578


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l95_9525

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem cube_of_square_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 2 ^ 3 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l95_9525


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l95_9521

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l95_9521


namespace NUMINAMATH_CALUDE_total_shoes_l95_9507

/-- Given that Ellie has 8 pairs of shoes and Riley has 3 fewer pairs than Ellie,
    prove that they have 13 pairs of shoes in total. -/
theorem total_shoes (ellie_shoes : ℕ) (riley_difference : ℕ) :
  ellie_shoes = 8 →
  riley_difference = 3 →
  ellie_shoes + (ellie_shoes - riley_difference) = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l95_9507


namespace NUMINAMATH_CALUDE_green_marble_probability_l95_9566

/-- The probability of selecting a green marble from a basket -/
theorem green_marble_probability :
  let total_marbles : ℕ := 4 + 9 + 5 + 10
  let green_marbles : ℕ := 9
  (green_marbles : ℚ) / total_marbles = 9 / 28 :=
by
  sorry

end NUMINAMATH_CALUDE_green_marble_probability_l95_9566


namespace NUMINAMATH_CALUDE_find_n_l95_9558

/-- Given that P = s / ((1 + k)^n + m), prove that n = (log((s/P) - m)) / (log(1 + k)) -/
theorem find_n (P s k m n : ℝ) (h : P = s / ((1 + k)^n + m)) (h1 : k > -1) (h2 : P > 0) (h3 : s > 0) :
  n = (Real.log ((s/P) - m)) / (Real.log (1 + k)) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l95_9558


namespace NUMINAMATH_CALUDE_billy_total_tickets_l95_9539

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def ticket_cost_per_ride : ℕ := 5

/-- Theorem: The total number of tickets Billy used is 50 -/
theorem billy_total_tickets : 
  (ferris_rides + bumper_rides) * ticket_cost_per_ride = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_total_tickets_l95_9539


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l95_9598

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 5/4}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l95_9598


namespace NUMINAMATH_CALUDE_somu_age_problem_l95_9516

theorem somu_age_problem (somu_age father_age : ℕ) : 
  (somu_age = father_age / 3) →
  (somu_age - 10 = (father_age - 10) / 5) →
  somu_age = 20 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l95_9516


namespace NUMINAMATH_CALUDE_alyssa_cherries_cost_l95_9526

/-- The amount Alyssa paid for cherries -/
def cherries_cost (total_spent grapes_cost : ℚ) : ℚ :=
  total_spent - grapes_cost

/-- Proof that Alyssa paid $9.85 for cherries -/
theorem alyssa_cherries_cost :
  let total_spent : ℚ := 21.93
  let grapes_cost : ℚ := 12.08
  cherries_cost total_spent grapes_cost = 9.85 := by
  sorry

#eval cherries_cost 21.93 12.08

end NUMINAMATH_CALUDE_alyssa_cherries_cost_l95_9526


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_16_l95_9567

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the vertex of the original parabola
def original_vertex : ℝ × ℝ := (3, 4)

-- Define the transformed parabola after all operations
def transformed_parabola (x : ℝ) : ℝ := (x - 8)^2

-- Define the new vertex after transformations
def new_vertex : ℝ × ℝ := (8, 8)

-- Define the zeros of the transformed parabola
def zeros : Set ℝ := {x : ℝ | transformed_parabola x = 0}

-- Theorem statement
theorem sum_of_zeros_is_16 : ∀ p q : ℝ, p ∈ zeros ∧ q ∈ zeros → p + q = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_16_l95_9567
