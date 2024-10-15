import Mathlib

namespace NUMINAMATH_CALUDE_even_decreasing_nonpos_ordering_l519_51901

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonpos (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_nonpos_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) (h_decr : decreasing_on_nonpos f) : 
  f 1 < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_nonpos_ordering_l519_51901


namespace NUMINAMATH_CALUDE_no_special_numbers_l519_51928

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_special_numbers :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ n % 3 = 0 ∧ Even n :=
by sorry

end NUMINAMATH_CALUDE_no_special_numbers_l519_51928


namespace NUMINAMATH_CALUDE_sqrt_20_less_than_5_l519_51917

theorem sqrt_20_less_than_5 : Real.sqrt 20 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_20_less_than_5_l519_51917


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l519_51931

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l519_51931


namespace NUMINAMATH_CALUDE_chicken_cage_problem_l519_51927

theorem chicken_cage_problem :
  ∃ (chickens cages : ℕ),
    chickens = 25 ∧ cages = 6 ∧
    (4 * cages + 1 = chickens) ∧
    (5 * (cages - 1) = chickens) :=
by sorry

end NUMINAMATH_CALUDE_chicken_cage_problem_l519_51927


namespace NUMINAMATH_CALUDE_cone_volume_l519_51995

/-- A cone with surface area π and lateral surface that unfolds into a semicircle has volume π/9 -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = 2 * r →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = π →  -- surface area is π
  h^2 + r^2 = l^2 →  -- Pythagorean theorem for cone
  (1/3) * π * r^2 * h = π/9 := by
sorry


end NUMINAMATH_CALUDE_cone_volume_l519_51995


namespace NUMINAMATH_CALUDE_fourth_derivative_y_l519_51936

noncomputable def y (x : ℝ) : ℝ := (x^2 + 3) * Real.log (x - 3)

theorem fourth_derivative_y (x : ℝ) (h : x ≠ 3) :
  (deriv^[4] y) x = (-2 * x^2 + 24 * x - 126) / (x - 3)^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_derivative_y_l519_51936


namespace NUMINAMATH_CALUDE_complex_equation_solution_l519_51981

theorem complex_equation_solution (z : ℂ) : (3 - 4*I + z)*I = 2 + I → z = -2 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l519_51981


namespace NUMINAMATH_CALUDE_equal_cost_at_45_l519_51977

/-- Represents the number of students in a class -/
def num_students : ℕ := 45

/-- Represents the original ticket price in yuan -/
def ticket_price : ℕ := 30

/-- Calculates the cost of Option 1 (20% discount for all) -/
def option1_cost (n : ℕ) : ℚ :=
  n * ticket_price * (4 / 5)

/-- Calculates the cost of Option 2 (10% discount and 5 free tickets) -/
def option2_cost (n : ℕ) : ℚ :=
  (n - 5) * ticket_price * (9 / 10)

/-- Theorem stating that for 45 students, the costs of both options are equal -/
theorem equal_cost_at_45 :
  option1_cost num_students = option2_cost num_students :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_45_l519_51977


namespace NUMINAMATH_CALUDE_smallest_terminating_n_is_correct_l519_51932

/-- A fraction a/b is a terminating decimal if b has only 2 and 5 as prime factors -/
def IsTerminatingDecimal (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ b → p = 2 ∨ p = 5

/-- The smallest positive integer n such that n/(n+150) is a terminating decimal -/
def SmallestTerminatingN : ℕ := 10

theorem smallest_terminating_n_is_correct :
  IsTerminatingDecimal SmallestTerminatingN (SmallestTerminatingN + 150) ∧
  ∀ m : ℕ, 0 < m → m < SmallestTerminatingN →
    ¬IsTerminatingDecimal m (m + 150) := by
  sorry

end NUMINAMATH_CALUDE_smallest_terminating_n_is_correct_l519_51932


namespace NUMINAMATH_CALUDE_prob_no_adjacent_three_of_ten_l519_51930

/-- The number of chairs in a row -/
def n : ℕ := 10

/-- The number of people choosing seats -/
def k : ℕ := 3

/-- The probability of k people choosing seats from n chairs such that none sit next to each other -/
def prob_no_adjacent (n k : ℕ) : ℚ :=
  sorry

/-- Theorem stating that the probability of 3 people choosing seats from 10 chairs 
    such that none sit next to each other is 1/3 -/
theorem prob_no_adjacent_three_of_ten : prob_no_adjacent n k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_three_of_ten_l519_51930


namespace NUMINAMATH_CALUDE_max_area_rectangle_l519_51970

/-- The perimeter of a rectangle -/
def perimeter (x y : ℝ) : ℝ := 2 * (x + y)

/-- The area of a rectangle -/
def area (x y : ℝ) : ℝ := x * y

/-- Theorem: For a rectangle with a fixed perimeter, the area is maximized when length equals width -/
theorem max_area_rectangle (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ perimeter x y = p ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → perimeter a b = p → area a b ≤ area x y) ∧
  x = p / 4 ∧ y = p / 4 := by
  sorry

#check max_area_rectangle

end NUMINAMATH_CALUDE_max_area_rectangle_l519_51970


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l519_51911

/-- The number of knights seated at the round table -/
def num_knights : ℕ := 25

/-- The number of knights chosen -/
def chosen_knights : ℕ := 3

/-- The probability of choosing at least two adjacent knights -/
def P : ℚ := 21/92

/-- Theorem stating the probability of choosing at least two adjacent knights -/
theorem adjacent_knights_probability :
  (
    let total_choices := Nat.choose num_knights chosen_knights
    let adjacent_triples := num_knights
    let adjacent_pairs := num_knights * (num_knights - 2 * chosen_knights + 1)
    let favorable_outcomes := adjacent_triples + adjacent_pairs
    (favorable_outcomes : ℚ) / total_choices
  ) = P := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l519_51911


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l519_51916

theorem unique_two_digit_integer (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (13 * t) % 100 = 52 ↔ t = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l519_51916


namespace NUMINAMATH_CALUDE_simplify_expression_l519_51961

theorem simplify_expression (a b : ℝ) : a + (3*a - 3*b) - (a - 2*b) = 3*a - b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l519_51961


namespace NUMINAMATH_CALUDE_q_polynomial_form_l519_51943

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^6 + 5x^4 + 10x^2) = (9x^4 + 30x^3 + 50x^2 + 4),
    prove that q(x) = -2x^6 + 4x^4 + 30x^3 + 40x^2 + 4 -/
theorem q_polynomial_form (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2 * x^6 + 5 * x^4 + 10 * x^2) = 9 * x^4 + 30 * x^3 + 50 * x^2 + 4) :
  ∀ x, q x = -2 * x^6 + 4 * x^4 + 30 * x^3 + 40 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l519_51943


namespace NUMINAMATH_CALUDE_test_composition_l519_51986

theorem test_composition (total_points total_questions : ℕ) 
  (h1 : total_points = 100) 
  (h2 : total_questions = 40) : 
  ∃ (two_point_questions four_point_questions : ℕ),
    two_point_questions + four_point_questions = total_questions ∧
    2 * two_point_questions + 4 * four_point_questions = total_points ∧
    two_point_questions = 30 := by
  sorry

end NUMINAMATH_CALUDE_test_composition_l519_51986


namespace NUMINAMATH_CALUDE_closest_sum_to_zero_l519_51976

def S : Finset Int := {5, 19, -6, 0, -4}

theorem closest_sum_to_zero (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
  |a + b + c| ≤ |x + y + z| ∧ (∃ p q r, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ |p + q + r| = 1) :=
by sorry

end NUMINAMATH_CALUDE_closest_sum_to_zero_l519_51976


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l519_51902

/-- Given a total number of books and books per bookshelf, calculates the number of bookshelves -/
def calculate_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

/-- Theorem stating that with 504 total books and 56 books per shelf, there are 9 bookshelves -/
theorem bryan_bookshelves :
  calculate_bookshelves 504 56 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l519_51902


namespace NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_eq_one_l519_51975

def abcd_plus_dcba (a : ℤ) : ℤ :=
  let b := a^2 + 1
  let c := a^2 + 2
  let d := a^2 + 3
  2111 * a^2 + 1001 * a + 3333

theorem gcd_abcd_plus_dcba_eq_one :
  ∃ (a : ℤ), ∀ (x : ℤ), x ∣ abcd_plus_dcba a → x = 1 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_eq_one_l519_51975


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_even_numbers_l519_51945

theorem largest_of_three_consecutive_even_numbers (x : ℤ) : 
  (∃ (a b c : ℤ), 
    (a + b + c = 312) ∧ 
    (b = a + 2) ∧ 
    (c = b + 2) ∧ 
    (Even a) ∧ (Even b) ∧ (Even c)) →
  (max a (max b c) = 106) :=
sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_even_numbers_l519_51945


namespace NUMINAMATH_CALUDE_odd_numbers_property_l519_51940

theorem odd_numbers_property (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_odd_numbers_property_l519_51940


namespace NUMINAMATH_CALUDE_sara_marble_count_l519_51960

/-- The number of black marbles Sara has after receiving a gift from Fred -/
def saras_marbles (initial : Float) (gift : Float) : Float :=
  initial + gift

/-- Theorem stating that Sara has 1025.0 black marbles after receiving Fred's gift -/
theorem sara_marble_count : saras_marbles 792.0 233.0 = 1025.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_marble_count_l519_51960


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l519_51944

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 30 * x + 10 = 0) : 
  ∃ x, b * x^2 + 30 * x + 10 = 0 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l519_51944


namespace NUMINAMATH_CALUDE_linear_function_inequality_l519_51942

theorem linear_function_inequality (a b : ℝ) (h1 : a > 0) (h2 : -2 * a + b = 0) :
  ∀ x : ℝ, a * x > b ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_inequality_l519_51942


namespace NUMINAMATH_CALUDE_right_triangle_with_reversed_digits_l519_51939

theorem right_triangle_with_reversed_digits : ∀ a b c : ℕ,
  a = 56 ∧ c = 65 ∧ 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 →
  a^2 + b^2 = c^2 →
  b = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_reversed_digits_l519_51939


namespace NUMINAMATH_CALUDE_course_selection_plans_l519_51910

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of course selection plans -/
def coursePlans (totalCourses restrictedCourses coursesToChoose : ℕ) : ℕ :=
  choose (totalCourses - restrictedCourses) coursesToChoose + 
  restrictedCourses * choose (totalCourses - restrictedCourses) (coursesToChoose - 1)

theorem course_selection_plans :
  coursePlans 8 2 5 = 36 := by sorry

end NUMINAMATH_CALUDE_course_selection_plans_l519_51910


namespace NUMINAMATH_CALUDE_power_division_equality_l519_51988

theorem power_division_equality : (3 : ℕ)^16 / (81 : ℕ)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l519_51988


namespace NUMINAMATH_CALUDE_sequence_term_40_l519_51934

theorem sequence_term_40 (n : ℕ+) (a : ℕ+ → ℕ) : 
  (∀ k : ℕ+, a k = 3 * k + 1) → a 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_40_l519_51934


namespace NUMINAMATH_CALUDE_pie_weight_l519_51926

theorem pie_weight (total_weight : ℝ) (eaten_fraction : ℝ) (eaten_weight : ℝ) : 
  eaten_fraction = 1/6 →
  eaten_weight = 240 →
  total_weight = eaten_weight / eaten_fraction →
  (1 - eaten_fraction) * total_weight = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_pie_weight_l519_51926


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l519_51941

/-- Given two similar right triangles, where the first triangle has a side of 18 units
    and a hypotenuse of 30 units, and the second triangle has a hypotenuse of 60 units,
    the side in the second triangle corresponding to the 18-unit side in the first triangle
    is 36 units long. -/
theorem similar_triangles_side_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + 18^2 = 30^2 →
  c^2 + d^2 = 60^2 →
  30 / 60 = 18 / d →
  d = 36 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l519_51941


namespace NUMINAMATH_CALUDE_nth_odd_multiple_of_three_l519_51991

theorem nth_odd_multiple_of_three (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, k > 0 ∧ k = 6 * n - 3 ∧ k % 2 = 1 ∧ k % 3 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m < k ∧ m % 2 = 1 ∧ m % 3 = 0 → 
   ∃ i : ℕ, i < n ∧ m = 6 * i - 3) :=
by sorry

end NUMINAMATH_CALUDE_nth_odd_multiple_of_three_l519_51991


namespace NUMINAMATH_CALUDE_james_vehicle_count_l519_51974

theorem james_vehicle_count :
  let trucks : ℕ := 12
  let buses : ℕ := 2
  let taxis : ℕ := 4
  let cars : ℕ := 30
  let truck_capacity : ℕ := 2
  let bus_capacity : ℕ := 15
  let taxi_capacity : ℕ := 2
  let motorbike_capacity : ℕ := 1
  let car_capacity : ℕ := 3
  let total_passengers : ℕ := 156
  let motorbikes : ℕ := total_passengers - 
    (trucks * truck_capacity + buses * bus_capacity + 
     taxis * taxi_capacity + cars * car_capacity)
  trucks + buses + taxis + motorbikes + cars = 52 := by
sorry

end NUMINAMATH_CALUDE_james_vehicle_count_l519_51974


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l519_51938

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l519_51938


namespace NUMINAMATH_CALUDE_card_drawing_probabilities_l519_51968

def num_cards : ℕ := 5
def num_odd_cards : ℕ := 3
def num_even_cards : ℕ := 2

def prob_not_both_odd_or_even : ℚ := 3 / 5

def prob_two_even_in_three_draws : ℚ := 36 / 125

theorem card_drawing_probabilities :
  (prob_not_both_odd_or_even = (num_odd_cards * num_even_cards : ℚ) / (num_cards.choose 2)) ∧
  (prob_two_even_in_three_draws = 3 * (num_even_cards / num_cards)^2 * (1 - num_even_cards / num_cards)) :=
by sorry

end NUMINAMATH_CALUDE_card_drawing_probabilities_l519_51968


namespace NUMINAMATH_CALUDE_total_toys_l519_51987

theorem total_toys (jaxon_toys gabriel_toys jerry_toys : ℕ) : 
  jaxon_toys = 15 →
  gabriel_toys = 2 * jaxon_toys →
  jerry_toys = gabriel_toys + 8 →
  jaxon_toys + gabriel_toys + jerry_toys = 83 := by
sorry

end NUMINAMATH_CALUDE_total_toys_l519_51987


namespace NUMINAMATH_CALUDE_distance_on_number_line_l519_51920

theorem distance_on_number_line : 
  let point_a : ℝ := 3
  let point_b : ℝ := -2
  |point_a - point_b| = 5 := by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l519_51920


namespace NUMINAMATH_CALUDE_largest_garden_difference_l519_51957

/-- Represents a rectangular garden with length and width in feet. -/
structure Garden where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden in square feet. -/
def gardenArea (g : Garden) : ℕ := g.length * g.width

/-- Alice's garden -/
def aliceGarden : Garden := { length := 30, width := 50 }

/-- Bob's garden -/
def bobGarden : Garden := { length := 35, width := 45 }

/-- Candace's garden -/
def candaceGarden : Garden := { length := 40, width := 40 }

theorem largest_garden_difference :
  let gardens := [aliceGarden, bobGarden, candaceGarden]
  let areas := gardens.map gardenArea
  let maxArea := areas.maximum?
  let minArea := areas.minimum?
  ∀ max min, maxArea = some max → minArea = some min →
    max - min = 100 := by sorry

end NUMINAMATH_CALUDE_largest_garden_difference_l519_51957


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l519_51915

-- System (1)
theorem system_one_solution (x y : ℚ) :
  (3 * x + 4 * y = 16) ∧ (5 * x - 8 * y = 34) → x = 6 ∧ y = -1/2 := by sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  ((x - 1) / 2 + (y + 1) / 3 = 1) ∧ (x + y = 4) → x = -1 ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l519_51915


namespace NUMINAMATH_CALUDE_triangle_side_length_l519_51969

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = Real.sqrt 7 →
  B = π / 3 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l519_51969


namespace NUMINAMATH_CALUDE_total_wall_area_to_paint_l519_51923

def living_room_width : ℝ := 40
def living_room_length : ℝ := 40
def bedroom_width : ℝ := 10
def bedroom_length : ℝ := 12
def wall_height : ℝ := 10
def living_room_walls_to_paint : ℕ := 3
def bedroom_walls_to_paint : ℕ := 4

theorem total_wall_area_to_paint :
  (living_room_walls_to_paint * living_room_width * wall_height) +
  (bedroom_walls_to_paint * bedroom_width * wall_height) +
  (bedroom_walls_to_paint * bedroom_length * wall_height) -
  (2 * bedroom_width * wall_height) = 1640 := by sorry

end NUMINAMATH_CALUDE_total_wall_area_to_paint_l519_51923


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l519_51984

/-- Represents a painted cube that can be cut into smaller cubes -/
structure PaintedCube where
  edge : ℕ  -- Edge length of the large cube
  small_edge : ℕ  -- Edge length of the smaller cubes

/-- Counts the number of smaller cubes with exactly one painted face -/
def count_one_face_painted (cube : PaintedCube) : ℕ :=
  6 * (cube.edge - 2) * (cube.edge - 2)

/-- Counts the number of smaller cubes with exactly two painted faces -/
def count_two_faces_painted (cube : PaintedCube) : ℕ :=
  12 * (cube.edge - 2)

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.edge = 10) 
  (h2 : cube.small_edge = 1) : 
  count_one_face_painted cube = 384 ∧ count_two_faces_painted cube = 96 := by
  sorry

#eval count_one_face_painted ⟨10, 1⟩
#eval count_two_faces_painted ⟨10, 1⟩

end NUMINAMATH_CALUDE_painted_cube_theorem_l519_51984


namespace NUMINAMATH_CALUDE_specific_rhombus_area_l519_51935

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular_bisectors : Bool

/-- Calculates the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    side_length := Real.sqrt 113,
    diagonal_difference := 8,
    diagonals_perpendicular_bisectors := true
  }
  rhombus_area r = 97 := by sorry

end NUMINAMATH_CALUDE_specific_rhombus_area_l519_51935


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l519_51985

theorem triangle_angle_calculation (A B C a b c : Real) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  a * Real.cos B - b * Real.cos A = c →
  -- Given angle C
  C = π / 5 →
  -- Conclusion
  B = 3 * π / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l519_51985


namespace NUMINAMATH_CALUDE_total_roses_theorem_l519_51965

/-- The number of bouquets to be made -/
def num_bouquets : ℕ := 5

/-- The number of table decorations to be made -/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The number of white roses used in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed for all bouquets and table decorations -/
def total_roses_needed : ℕ := num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

theorem total_roses_theorem : total_roses_needed = 109 := by
  sorry

end NUMINAMATH_CALUDE_total_roses_theorem_l519_51965


namespace NUMINAMATH_CALUDE_area_outside_squares_inside_triangle_l519_51948

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the problem setup -/
structure SquareProblem where
  bigSquare : Square
  smallSquare1 : Square
  smallSquare2 : Square

/-- The main theorem stating the area of the region -/
theorem area_outside_squares_inside_triangle (p : SquareProblem) : 
  p.bigSquare.sideLength = 6 ∧ 
  p.smallSquare1.sideLength = 2 ∧ 
  p.smallSquare2.sideLength = 3 →
  let triangleArea := (p.bigSquare.sideLength ^ 2) / 2
  let smallSquaresArea := p.smallSquare1.sideLength ^ 2 + p.smallSquare2.sideLength ^ 2
  triangleArea - smallSquaresArea = 5 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_squares_inside_triangle_l519_51948


namespace NUMINAMATH_CALUDE_employed_female_parttime_ratio_is_60_percent_l519_51919

/-- Represents the population statistics of Town P -/
structure TownP where
  total_population : ℝ
  employed_percentage : ℝ
  employed_female_percentage : ℝ
  employed_female_parttime_percentage : ℝ
  employed_male_percentage : ℝ

/-- Calculates the percentage of employed females who are part-time workers in Town P -/
def employed_female_parttime_ratio (town : TownP) : ℝ :=
  town.employed_female_parttime_percentage

/-- Theorem stating that 60% of employed females in Town P are part-time workers -/
theorem employed_female_parttime_ratio_is_60_percent (town : TownP) 
  (h1 : town.employed_percentage = 0.6)
  (h2 : town.employed_female_percentage = 0.4)
  (h3 : town.employed_female_parttime_percentage = 0.6)
  (h4 : town.employed_male_percentage = 0.48) :
  employed_female_parttime_ratio town = 0.6 := by
  sorry

#check employed_female_parttime_ratio_is_60_percent

end NUMINAMATH_CALUDE_employed_female_parttime_ratio_is_60_percent_l519_51919


namespace NUMINAMATH_CALUDE_quadratic_shared_root_property_l519_51924

/-- A quadratic polynomial P(x) = x^2 + bx + c -/
def P (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The theorem stating that if P(x) and P(P(P(x))) share a root, then P(0) * P(1) = 0 -/
theorem quadratic_shared_root_property (b c : ℝ) :
  (∃ r : ℝ, P b c r = 0 ∧ P b c (P b c (P b c r)) = 0) →
  P b c 0 * P b c 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shared_root_property_l519_51924


namespace NUMINAMATH_CALUDE_total_birds_caught_l519_51967

-- Define the number of birds hunted during the day
def day_hunt : ℕ := 15

-- Define the success rate during the day
def day_success_rate : ℚ := 3/5

-- Define the number of birds hunted at night
def night_hunt : ℕ := 25

-- Define the success rate at night
def night_success_rate : ℚ := 4/5

-- Define the relationship between day and night catches
def night_day_ratio : ℕ := 2

-- Theorem statement
theorem total_birds_caught :
  ⌊(day_hunt : ℚ) * day_success_rate⌋ +
  ⌊(night_hunt : ℚ) * night_success_rate⌋ = 29 := by
  sorry


end NUMINAMATH_CALUDE_total_birds_caught_l519_51967


namespace NUMINAMATH_CALUDE_room_tiles_count_l519_51979

/-- Calculates the number of tiles needed for a room with given specifications -/
def calculate_tiles (room_length room_width border_width tile_size column_size : ℕ) : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let border_tiles := 2 * ((room_length / tile_size) + (room_width / tile_size) - 4)
  let inner_tiles := (inner_length * inner_width) / (tile_size * tile_size)
  let column_tiles := (column_size * column_size + tile_size * tile_size - 1) / (tile_size * tile_size)
  border_tiles + inner_tiles + column_tiles

/-- Theorem stating that the number of tiles for the given room specification is 78 -/
theorem room_tiles_count : calculate_tiles 15 20 2 2 1 = 78 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_count_l519_51979


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l519_51996

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - I) / (1 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l519_51996


namespace NUMINAMATH_CALUDE_sons_age_l519_51990

theorem sons_age (son_age man_age : ℕ) : 
  man_age = 3 * son_age →
  man_age + 12 = 2 * (son_age + 12) →
  son_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l519_51990


namespace NUMINAMATH_CALUDE_largest_expression_l519_51952

theorem largest_expression :
  let a := 3 + 1 + 2 + 8
  let b := 3 * 1 + 2 + 8
  let c := 3 + 1 * 2 + 8
  let d := 3 + 1 + 2 * 8
  let e := 3 * 1 * 2 * 8
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l519_51952


namespace NUMINAMATH_CALUDE_sin_beta_value_l519_51972

theorem sin_beta_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin β = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l519_51972


namespace NUMINAMATH_CALUDE_smallest_odd_minimizer_l519_51963

/-- The number of positive integer divisors of n, including 1 and n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The function g(n) = d(n) / n^(1/4) -/
noncomputable def g (n : ℕ) : ℝ := (d n : ℝ) / n^(1/4 : ℝ)

/-- n is an odd positive integer -/
def isOddPositive (n : ℕ) : Prop := n > 0 ∧ n % 2 = 1

/-- 9 is the smallest odd positive integer N such that g(N) < g(n) for all odd positive integers n ≠ N -/
theorem smallest_odd_minimizer :
  isOddPositive 9 ∧
  (∀ n : ℕ, isOddPositive n → n ≠ 9 → g 9 < g n) ∧
  (∀ N : ℕ, isOddPositive N → N < 9 → ∃ n : ℕ, isOddPositive n ∧ n ≠ N ∧ g N ≥ g n) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_minimizer_l519_51963


namespace NUMINAMATH_CALUDE_triangle_area_l519_51951

/-- Given a triangle with perimeter 32 and inradius 2.5, prove its area is 40 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h1 : perimeter = 32) 
  (h2 : inradius = 2.5) 
  (h3 : area = inradius * (perimeter / 2)) : 
  area = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l519_51951


namespace NUMINAMATH_CALUDE_total_pages_read_l519_51937

-- Define the reading rates (pages per 60 minutes)
def rene_rate : ℕ := 30
def lulu_rate : ℕ := 27
def cherry_rate : ℕ := 25

-- Define the total reading time in minutes
def total_time : ℕ := 240

-- Define the function to calculate pages read given rate and time
def pages_read (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem statement
theorem total_pages_read :
  pages_read rene_rate total_time +
  pages_read lulu_rate total_time +
  pages_read cherry_rate total_time = 328 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_read_l519_51937


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l519_51999

theorem gcd_powers_of_two : Nat.gcd (2^2024 - 1) (2^2007 - 1) = 2^17 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l519_51999


namespace NUMINAMATH_CALUDE_limits_of_f_l519_51947

noncomputable def f (x : ℝ) : ℝ := 2^(1/x)

theorem limits_of_f :
  (∀ ε > 0, ∃ δ > 0, ∀ x < 0, |x| < δ → |f x| < ε) ∧
  (∀ M > 0, ∃ δ > 0, ∀ x > 0, x < δ → f x > M) ∧
  ¬ (∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |f x - L| < ε) :=
by sorry

end NUMINAMATH_CALUDE_limits_of_f_l519_51947


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l519_51997

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5*x - 2

-- Define the solution set of the original inequality
def solution_set (a : ℝ) := {x : ℝ | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) := a * x^2 - 5*x + a^2 - 1

-- Theorem statement
theorem quadratic_inequality_problem 
  (a : ℝ) 
  (h : ∀ x, f a x > 0 ↔ x ∈ solution_set a) :
  a = -2 ∧ 
  (∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l519_51997


namespace NUMINAMATH_CALUDE_inequality_solution_l519_51929

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  (x / (x + 1) + (x - 3) / (3 * x) ≥ 4) ↔ 
  (x > -1.5 ∧ x < -1) ∨ (x > -0.25) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l519_51929


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l519_51966

theorem cubic_equation_properties :
  (∀ x y : ℕ, x^3 + y = y^3 + x → x = y) ∧
  (∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l519_51966


namespace NUMINAMATH_CALUDE_hard_drive_cost_l519_51958

/-- The cost of seven hard drives with a bulk discount -/
theorem hard_drive_cost : 
  -- Two hard drives cost $50
  (∃ (single_cost : ℝ), 2 * single_cost = 50) →
  -- There's a 10% discount for buying more than 4
  (∀ (n : ℕ), n > 4 → ∃ (discount_factor : ℝ), discount_factor = 0.9) →
  -- The cost of 7 hard drives with the discount is $157.5
  ∃ (total_cost : ℝ), total_cost = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_hard_drive_cost_l519_51958


namespace NUMINAMATH_CALUDE_a0_value_sum_of_all_coefficients_sum_of_odd_coefficients_l519_51921

-- Define the polynomial coefficients
variable (a : Fin 8 → ℤ)

-- Define the equality condition
axiom expansion_equality : ∀ x : ℝ, (2*x - 1)^7 = (Finset.range 8).sum (λ i => a i * x^i)

-- Theorem statements
theorem a0_value : a 0 = -1 := by sorry

theorem sum_of_all_coefficients : (Finset.range 8).sum (λ i => a i) - a 0 = 2 := by sorry

theorem sum_of_odd_coefficients : a 1 + a 3 + a 5 + a 7 = -126 := by sorry

end NUMINAMATH_CALUDE_a0_value_sum_of_all_coefficients_sum_of_odd_coefficients_l519_51921


namespace NUMINAMATH_CALUDE_tuesday_texts_l519_51900

/-- The number of texts sent by Sydney to each person on Monday -/
def monday_texts_per_person : ℕ := 5

/-- The total number of texts sent by Sydney over both days -/
def total_texts : ℕ := 40

/-- The number of people Sydney sent texts to -/
def num_people : ℕ := 2

/-- Theorem: The number of texts sent on Tuesday is 30 -/
theorem tuesday_texts :
  total_texts - (monday_texts_per_person * num_people) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_texts_l519_51900


namespace NUMINAMATH_CALUDE_boulder_splash_width_l519_51908

/-- The width of a boulder's splash given the number of pebbles, rocks, and boulders thrown,
    and the total width of all splashes. -/
theorem boulder_splash_width
  (num_pebbles : ℕ)
  (num_rocks : ℕ)
  (num_boulders : ℕ)
  (total_width : ℝ)
  (pebble_splash : ℝ)
  (rock_splash : ℝ)
  (h1 : num_pebbles = 6)
  (h2 : num_rocks = 3)
  (h3 : num_boulders = 2)
  (h4 : total_width = 7)
  (h5 : pebble_splash = 1/4)
  (h6 : rock_splash = 1/2)
  : (total_width - (num_pebbles * pebble_splash + num_rocks * rock_splash)) / num_boulders = 2 :=
sorry

end NUMINAMATH_CALUDE_boulder_splash_width_l519_51908


namespace NUMINAMATH_CALUDE_tickets_spent_on_beanie_l519_51993

theorem tickets_spent_on_beanie (initial_tickets : Real) (lost_tickets : Real) (remaining_tickets : Real)
  (h1 : initial_tickets = 49.0)
  (h2 : lost_tickets = 6.0)
  (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by sorry

end NUMINAMATH_CALUDE_tickets_spent_on_beanie_l519_51993


namespace NUMINAMATH_CALUDE_max_point_of_f_l519_51956

def f (x : ℝ) := 3 * x - x^3

theorem max_point_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≤ f x₀ ∧ x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_point_of_f_l519_51956


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l519_51933

/-- The degree of a polynomial -/
def degree (p : Polynomial ℂ) : ℕ := sorry

theorem polynomial_remainder_theorem :
  ∃ (Q R : Polynomial ℂ),
    (X : Polynomial ℂ)^2023 + 1 = (X^2 + X + 1) * Q + R ∧
    degree R < 2 ∧
    R = -X + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l519_51933


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_ten_is_three_sixteenths_l519_51953

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- A fair 8-sided die -/
def eight_sided_die : Finset ℕ := Finset.range 8

/-- The product space of rolling both dice -/
def dice_product : Finset (ℕ × ℕ) := six_sided_die.product eight_sided_die

/-- The subset of outcomes where the sum is greater than 10 -/
def sum_greater_than_ten : Finset (ℕ × ℕ) :=
  dice_product.filter (fun p => p.1 + p.2 + 2 > 10)

/-- The probability of the sum being greater than 10 -/
def prob_sum_greater_than_ten : ℚ :=
  (sum_greater_than_ten.card : ℚ) / (dice_product.card : ℚ)

theorem prob_sum_greater_than_ten_is_three_sixteenths :
  prob_sum_greater_than_ten = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_ten_is_three_sixteenths_l519_51953


namespace NUMINAMATH_CALUDE_constant_term_expansion_l519_51925

/-- The constant term in the expansion of (x + 1/x + 1)^4 -/
def constant_term : ℕ := 19

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_expansion :
  constant_term = 1 + binomial 4 2 * binomial 2 1 + binomial 4 4 * binomial 4 2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l519_51925


namespace NUMINAMATH_CALUDE_pepper_plants_died_l519_51913

/-- Represents the garden with its plants and vegetables --/
structure Garden where
  tomato_plants : ℕ
  eggplant_plants : ℕ
  pepper_plants : ℕ
  dead_tomato_plants : ℕ
  dead_pepper_plants : ℕ
  vegetables_per_plant : ℕ
  total_vegetables : ℕ

/-- Theorem representing the problem and its solution --/
theorem pepper_plants_died (g : Garden) : g.dead_pepper_plants = 1 :=
  by
  have h1 : g.tomato_plants = 6 := by sorry
  have h2 : g.eggplant_plants = 2 := by sorry
  have h3 : g.pepper_plants = 4 := by sorry
  have h4 : g.dead_tomato_plants = g.tomato_plants / 2 := by sorry
  have h5 : g.vegetables_per_plant = 7 := by sorry
  have h6 : g.total_vegetables = 56 := by sorry
  
  sorry

end NUMINAMATH_CALUDE_pepper_plants_died_l519_51913


namespace NUMINAMATH_CALUDE_elephants_viewing_time_l519_51912

def zoo_visit (total_time seals_time penguins_multiplier : ℕ) : ℕ :=
  total_time - (seals_time + seals_time * penguins_multiplier)

theorem elephants_viewing_time :
  zoo_visit 130 13 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_elephants_viewing_time_l519_51912


namespace NUMINAMATH_CALUDE_locus_of_centers_l519_51992

-- Define the circles C₃ and C₄
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₄ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being externally tangent to C₃ and internally tangent to C₄
def is_tangent_to_C₃_C₄ (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)

-- State the theorem
theorem locus_of_centers :
  ∀ a b : ℝ, (∃ r : ℝ, is_tangent_to_C₃_C₄ a b r) → a^2 + 18*b^2 - 6*a - 440 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l519_51992


namespace NUMINAMATH_CALUDE_function_inequality_and_sum_product_l519_51906

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 2|

-- State the theorem
theorem function_inequality_and_sum_product (m M a b : ℝ) :
  (∀ x, f x ≥ |m - 1|) →
  (-2 ≤ m ∧ m ≤ 4) ∧
  (M = 4 →
   a > 0 →
   b > 0 →
   a^2 + b^2 = M/2 →
   a + b ≥ 2*a*b) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_sum_product_l519_51906


namespace NUMINAMATH_CALUDE_cone_volume_with_special_surface_area_l519_51946

/-- 
Given a cone with base radius R, if its lateral surface area is equal to the sum of 
the areas of its base and axial section, then its volume is (2π²R³) / (3(π² - 1)).
-/
theorem cone_volume_with_special_surface_area (R : ℝ) (h : R > 0) : 
  let lateral_surface_area := π * R * (R^2 + (2 * π * R / (π^2 - 1))^2).sqrt
  let base_area := π * R^2
  let axial_section_area := R * (2 * π * R / (π^2 - 1))
  lateral_surface_area = base_area + axial_section_area →
  (1/3) * π * R^2 * (2 * π * R / (π^2 - 1)) = 2 * π^2 * R^3 / (3 * (π^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_cone_volume_with_special_surface_area_l519_51946


namespace NUMINAMATH_CALUDE_probability_roots_different_signs_l519_51980

def S : Set ℕ := {1, 2, 3, 4, 5, 6}

def quadratic_equation (a b x : ℝ) : Prop :=
  x^2 - 2*(a-3)*x + 9 - b^2 = 0

def roots_different_signs (a b : ℝ) : Prop :=
  (9 - b^2 < 0) ∧ (4*(a-3)^2 - 4*(9-b^2) > 0)

def count_valid_pairs : ℕ := 18

def total_pairs : ℕ := 36

theorem probability_roots_different_signs :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_probability_roots_different_signs_l519_51980


namespace NUMINAMATH_CALUDE_vanessa_birthday_money_l519_51982

theorem vanessa_birthday_money (money : ℕ) : 
  (∃ k : ℕ, money = 9 * k + 1) ↔ 
  (∃ n : ℕ, money = 9 * n + 1 ∧ 
    ∀ m : ℕ, m < n → money ≥ 9 * m + 1) :=
by sorry

end NUMINAMATH_CALUDE_vanessa_birthday_money_l519_51982


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l519_51918

theorem sqrt_expression_equality : 
  (Real.sqrt 2 - Real.sqrt 3) ^ 2020 * (Real.sqrt 2 + Real.sqrt 3) ^ 2021 = Real.sqrt 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l519_51918


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l519_51955

/-- A tree that triples its height each year --/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height each year and reaches 81 feet after 4 years
    will have a height of 9 feet after 2 years --/
theorem tree_height_after_two_years
  (h : tree_height (tree_height h₀ 2) 2 = 81)
  (h₀ : ℝ) :
  tree_height h₀ 2 = 9 :=
sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l519_51955


namespace NUMINAMATH_CALUDE_complex_imaginary_x_value_l519_51994

/-- A complex number z is imaginary if its real part is zero -/
def IsImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_imaginary_x_value (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 1) (x + 1)
  IsImaginary z → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_x_value_l519_51994


namespace NUMINAMATH_CALUDE_not_parabola_l519_51964

theorem not_parabola (α x y : ℝ) : 
  ∃ (a b c : ℝ), ∀ (x y : ℝ), x^2 * Real.sin α + y^2 * Real.cos α = 1 → y ≠ a*x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l519_51964


namespace NUMINAMATH_CALUDE_smallest_m_value_l519_51954

theorem smallest_m_value (m : ℕ) : 
  (∃! quad : (ℕ × ℕ × ℕ × ℕ) → Prop, 
    (∃ (n : ℕ), n = 80000 ∧ 
      (∀ a b c d : ℕ, quad (a, b, c, d) → 
        Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 100 ∧
        Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m))) →
  m = 2250000 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_value_l519_51954


namespace NUMINAMATH_CALUDE_sachin_age_l519_51971

/-- Proves that Sachin's age is 49 given the conditions -/
theorem sachin_age :
  ∀ (s r : ℕ),
  r = s + 14 →
  s * 9 = r * 7 →
  s = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l519_51971


namespace NUMINAMATH_CALUDE_binary_to_hexadecimal_conversion_l519_51989

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  sorry

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hexadecimal (decimal : ℕ) : List ℕ :=
  sorry

theorem binary_to_hexadecimal_conversion :
  let binary : List Bool := [true, false, true, true, false, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let hexadecimal : List ℕ := decimal_to_hexadecimal decimal
  hexadecimal = [2, 2, 5] := by sorry

end NUMINAMATH_CALUDE_binary_to_hexadecimal_conversion_l519_51989


namespace NUMINAMATH_CALUDE_lipstick_ratio_l519_51903

/-- Proves that the ratio of students wearing blue lipstick to those wearing red lipstick is 1:5 -/
theorem lipstick_ratio (total_students : ℕ) (blue_lipstick : ℕ) 
  (h1 : total_students = 200)
  (h2 : blue_lipstick = 5)
  (h3 : 2 * (total_students / 2) = total_students)  -- Half of students wore lipstick
  (h4 : 4 * (total_students / 2 / 4) = total_students / 2)  -- Quarter of lipstick wearers wore red
  (h5 : blue_lipstick = total_students / 2 / 4)  -- Same number wore blue as red
  : (blue_lipstick : ℚ) / (total_students / 2 / 4 : ℚ) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_lipstick_ratio_l519_51903


namespace NUMINAMATH_CALUDE_means_inequality_l519_51959

theorem means_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Real.sqrt ((a^2 + b^2) / 2) > (a + b) / 2 ∧
  (a + b) / 2 > Real.sqrt (a * b) ∧
  Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_means_inequality_l519_51959


namespace NUMINAMATH_CALUDE_used_car_selections_l519_51962

/-- Proves that given 16 cars, 24 clients, and each client selecting 2 cars, 
    each car must be selected 3 times. -/
theorem used_car_selections (cars : ℕ) (clients : ℕ) (selections_per_client : ℕ) 
    (h1 : cars = 16) 
    (h2 : clients = 24) 
    (h3 : selections_per_client = 2) : 
  (clients * selections_per_client) / cars = 3 := by
  sorry

#check used_car_selections

end NUMINAMATH_CALUDE_used_car_selections_l519_51962


namespace NUMINAMATH_CALUDE_one_story_height_l519_51973

-- Define the parameters
def stories : ℕ := 6
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def num_ropes : ℕ := 4

-- Define the theorem
theorem one_story_height :
  let total_usable_length := (1 - loss_percentage) * rope_length * num_ropes
  let story_height := total_usable_length / stories
  story_height = 10 := by sorry

end NUMINAMATH_CALUDE_one_story_height_l519_51973


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l519_51904

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 120 → m = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l519_51904


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l519_51909

theorem bakery_flour_usage (wheat_flour : Real) (white_flour : Real)
  (h1 : wheat_flour = 0.2)
  (h2 : white_flour = 0.1) :
  wheat_flour + white_flour = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l519_51909


namespace NUMINAMATH_CALUDE_point_inside_circle_l519_51914

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    eccentricity e = √2, right focus F(c, 0), and an equation ax² - bx - c = 0
    with roots x₁ and x₂, prove that the point P(x₁, x₂) is inside the circle x² + y² = 8 -/
theorem point_inside_circle (a b c : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : c / a = Real.sqrt 2)
  (h_focus : c > 0)
  (h_roots : a * x₁^2 - b * x₁ - c = 0 ∧ a * x₂^2 - b * x₂ - c = 0) :
  x₁^2 + x₂^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l519_51914


namespace NUMINAMATH_CALUDE_airport_distance_is_130_l519_51983

/-- Represents the problem of calculating the distance to the airport --/
def AirportDistance (initial_speed : ℝ) (speed_increase : ℝ) (initial_delay : ℝ) (actual_early : ℝ) : Prop :=
  ∃ (distance : ℝ) (time : ℝ),
    distance = initial_speed * (time + 1) ∧
    distance - initial_speed = (initial_speed + speed_increase) * (time - actual_early) ∧
    distance = 130

/-- The theorem stating that the distance to the airport is 130 miles --/
theorem airport_distance_is_130 :
  AirportDistance 40 20 1 0.25 := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_is_130_l519_51983


namespace NUMINAMATH_CALUDE_tree_height_from_shadows_l519_51978

/-- Given a person and a tree casting shadows, calculates the height of the tree -/
theorem tree_height_from_shadows 
  (h s S : ℝ) 
  (h_pos : h > 0) 
  (s_pos : s > 0) 
  (S_pos : S > 0) 
  (h_val : h = 1.5) 
  (s_val : s = 0.5) 
  (S_val : S = 10) : 
  h / s * S = 30 := by
sorry

end NUMINAMATH_CALUDE_tree_height_from_shadows_l519_51978


namespace NUMINAMATH_CALUDE_operation_problem_l519_51998

-- Define the set of operations
inductive Operation
| Add
| Sub
| Mul
| Div

-- Define the function that applies the operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (star mul : Operation) (h : apply_op star 16 4 / apply_op mul 8 2 = 4) :
  apply_op star 9 3 / apply_op mul 18 6 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_operation_problem_l519_51998


namespace NUMINAMATH_CALUDE_probability_of_exact_tails_l519_51907

noncomputable def probability_of_tails : ℚ := 2/3
noncomputable def number_of_flips : ℕ := 10
noncomputable def number_of_tails : ℕ := 4

theorem probability_of_exact_tails :
  (Nat.choose number_of_flips number_of_tails) *
  (probability_of_tails ^ number_of_tails) *
  ((1 - probability_of_tails) ^ (number_of_flips - number_of_tails)) =
  3360/6561 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_exact_tails_l519_51907


namespace NUMINAMATH_CALUDE_inequality_implication_l519_51905

theorem inequality_implication (x y : ℝ) (h : x < y) : -x + 3 > -y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l519_51905


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l519_51949

/-- Given 5 persons, if replacing one person with a new person weighing 95.5 kg
    increases the average weight by 5.5 kg, then the weight of the replaced person was 68 kg. -/
theorem weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 5 →
  new_person_weight = 95.5 →
  avg_increase = 5.5 →
  (new_person_weight - initial_count * avg_increase : ℝ) = 68 := by
sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l519_51949


namespace NUMINAMATH_CALUDE_binomial_15_4_l519_51922

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l519_51922


namespace NUMINAMATH_CALUDE_shelves_count_l519_51950

/-- The number of shelves in a library --/
def number_of_shelves (books_per_shelf : ℕ) (total_round_trip_distance : ℕ) : ℕ :=
  (total_round_trip_distance / 2) / books_per_shelf

/-- Theorem: The number of shelves is 4 --/
theorem shelves_count :
  number_of_shelves 400 3200 = 4 := by
  sorry

end NUMINAMATH_CALUDE_shelves_count_l519_51950
