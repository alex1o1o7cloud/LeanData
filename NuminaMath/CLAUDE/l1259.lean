import Mathlib

namespace line_intersects_circle_twice_min_chord_line_equation_l1259_125930

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem 1: L always intersects C at two points for any real m
theorem line_intersects_circle_twice :
  ∀ m : ℝ, ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ C p1.1 p1.2 ∧ C p2.1 p2.2 ∧ L m p1.1 p1.2 ∧ L m p2.1 p2.2 :=
sorry

-- Theorem 2: When chord length is minimum, L has equation y = 2x - 5
theorem min_chord_line_equation :
  ∃! m : ℝ, (∀ x y : ℝ, L m x y ↔ y = 2*x - 5) ∧
  (∀ m' : ℝ, m' ≠ m → 
    ∃ p1 p2 q1 q2 : ℝ × ℝ, p1 ≠ p2 ∧ q1 ≠ q2 ∧
    C p1.1 p1.2 ∧ C p2.1 p2.2 ∧ L m p1.1 p1.2 ∧ L m p2.1 p2.2 ∧
    C q1.1 q1.2 ∧ C q2.1 q2.2 ∧ L m' q1.1 q1.2 ∧ L m' q2.1 q2.2 ∧
    (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 < (q1.1 - q2.1)^2 + (q1.2 - q2.2)^2) :=
sorry

end line_intersects_circle_twice_min_chord_line_equation_l1259_125930


namespace min_squares_to_remove_202x202_l1259_125984

/-- Represents a T-tetromino -/
structure TTetromino :=
  (shape : List (Int × Int))

/-- Represents a grid -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a tiling of a grid with T-tetrominoes -/
def Tiling (g : Grid) := List TTetromino

/-- The number of squares that need to be removed for a valid tiling -/
def SquaresToRemove (g : Grid) (t : Tiling g) : Nat :=
  g.width * g.height - 4 * t.length

/-- Theorem: The minimum number of squares to remove from a 202x202 grid for T-tetromino tiling is 4 -/
theorem min_squares_to_remove_202x202 :
  ∀ (g : Grid) (t : Tiling g), g.width = 202 → g.height = 202 →
  SquaresToRemove g t ≥ 4 ∧ ∃ (t' : Tiling g), SquaresToRemove g t' = 4 :=
sorry

end min_squares_to_remove_202x202_l1259_125984


namespace tenth_term_of_sequence_l1259_125971

def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a₁ d : ℚ) (h₁ : a₁ = 1/2) (h₂ : d = 1/2) :
  arithmeticSequence a₁ d 10 = 5 := by
  sorry

end tenth_term_of_sequence_l1259_125971


namespace mode_and_median_of_data_set_l1259_125959

def data_set : List ℝ := [1, 1, 4, 5, 5, 5]

/-- The mode of a list of real numbers -/
def mode (l : List ℝ) : ℝ := sorry

/-- The median of a list of real numbers -/
def median (l : List ℝ) : ℝ := sorry

theorem mode_and_median_of_data_set :
  mode data_set = 5 ∧ median data_set = 4.5 := by sorry

end mode_and_median_of_data_set_l1259_125959


namespace unique_matrix_solution_l1259_125933

open Matrix

theorem unique_matrix_solution {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A ^ 3 = 0) : 
  ∃! X : Matrix (Fin n) (Fin n) ℝ, X + A * X + X * A ^ 2 = A ∧ 
    X = A * (1 + A + A ^ 2)⁻¹ := by
  sorry

end unique_matrix_solution_l1259_125933


namespace sphere_surface_area_in_cube_l1259_125994

theorem sphere_surface_area_in_cube (edge_length : Real) (surface_area : Real) :
  edge_length = 2 →
  surface_area = 4 * Real.pi →
  ∃ (r : Real),
    r = edge_length / 2 ∧
    surface_area = 4 * Real.pi * r^2 :=
by
  sorry

end sphere_surface_area_in_cube_l1259_125994


namespace jester_count_l1259_125960

theorem jester_count (total_legs total_heads : ℕ) 
  (jester_legs jester_heads elephant_legs elephant_heads : ℕ) : 
  total_legs = 50 → 
  total_heads = 18 → 
  jester_legs = 3 → 
  jester_heads = 1 → 
  elephant_legs = 4 → 
  elephant_heads = 1 → 
  ∃ (num_jesters num_elephants : ℕ), 
    num_jesters * jester_legs + num_elephants * elephant_legs = total_legs ∧
    num_jesters * jester_heads + num_elephants * elephant_heads = total_heads ∧
    num_jesters = 22 :=
by sorry

end jester_count_l1259_125960


namespace halfway_between_one_third_and_one_fifth_l1259_125950

theorem halfway_between_one_third_and_one_fifth :
  (1 / 3 : ℚ) + (1 / 5 : ℚ) = 2 * (4 / 15 : ℚ) :=
by sorry

end halfway_between_one_third_and_one_fifth_l1259_125950


namespace f_minimum_value_l1259_125980

def f (x : ℝ) := |2*x + 1| + |x - 1|

theorem f_minimum_value :
  (∀ x, f x ≥ 3/2) ∧ (∃ x, f x = 3/2) := by sorry

end f_minimum_value_l1259_125980


namespace sum_of_binary_digits_435_l1259_125911

/-- The sum of the digits in the binary representation of 435 is 6 -/
theorem sum_of_binary_digits_435 : 
  (Nat.digits 2 435).sum = 6 := by sorry

end sum_of_binary_digits_435_l1259_125911


namespace right_triangle_side_c_l1259_125966

theorem right_triangle_side_c (a b c : ℝ) : 
  a = 3 → b = 4 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 5 ∨ c = Real.sqrt 7 := by
  sorry

end right_triangle_side_c_l1259_125966


namespace base_k_representation_of_fraction_l1259_125999

theorem base_k_representation_of_fraction (k : ℕ) (h : k = 18) :
  let series_sum := (1 / k + 6 / k^2) / (1 - 1 / k^2)
  series_sum = 8 / 63 := by
  sorry

end base_k_representation_of_fraction_l1259_125999


namespace cost_of_5_spoons_l1259_125920

-- Define the cost of a set of 7 spoons
def cost_7_spoons : ℝ := 21

-- Define the number of spoons in a set
def spoons_in_set : ℕ := 7

-- Define the number of spoons we want to buy
def spoons_to_buy : ℕ := 5

-- Theorem: The cost of 5 spoons is $15
theorem cost_of_5_spoons :
  (cost_7_spoons / spoons_in_set) * spoons_to_buy = 15 := by
  sorry

end cost_of_5_spoons_l1259_125920


namespace peaches_left_l1259_125995

def total_peaches : ℕ := 250
def fresh_percentage : ℚ := 60 / 100
def small_peaches : ℕ := 15

theorem peaches_left : 
  (total_peaches * fresh_percentage).floor - small_peaches = 135 := by
  sorry

end peaches_left_l1259_125995


namespace simplify_inverse_sum_l1259_125990

theorem simplify_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) / (x * y * z * (x + y + z)) := by
  sorry

end simplify_inverse_sum_l1259_125990


namespace factorization_2x_squared_minus_8_l1259_125926

theorem factorization_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_2x_squared_minus_8_l1259_125926


namespace rhea_and_husband_eggs_per_night_l1259_125969

/-- The number of egg trays Rhea buys every week -/
def trays_per_week : ℕ := 2

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 24

/-- The number of eggs eaten by each child every morning -/
def eggs_per_child_per_morning : ℕ := 2

/-- The number of children -/
def number_of_children : ℕ := 2

/-- The number of eggs not eaten every week -/
def uneaten_eggs_per_week : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that Rhea and her husband eat 2 eggs every night -/
theorem rhea_and_husband_eggs_per_night :
  (trays_per_week * eggs_per_tray - 
   eggs_per_child_per_morning * number_of_children * days_per_week - 
   uneaten_eggs_per_week) / days_per_week = 2 := by
  sorry

end rhea_and_husband_eggs_per_night_l1259_125969


namespace point_distance_to_line_l1259_125958

theorem point_distance_to_line (a : ℝ) (h1 : a > 0) : 
  (|a - 2 + 3| / Real.sqrt 2 = 1) → a = Real.sqrt 2 - 1 := by
  sorry

end point_distance_to_line_l1259_125958


namespace bumper_car_cost_correct_l1259_125968

/-- The number of tickets required for one bumper car ride -/
def bumper_car_cost : ℕ := 5

/-- The number of times Paula rides the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The cost of riding go-karts once -/
def go_kart_cost : ℕ := 4

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := 24

/-- Theorem stating that the bumper car cost satisfies the given conditions -/
theorem bumper_car_cost_correct :
  bumper_car_cost * bumper_car_rides + go_kart_cost = total_tickets :=
by sorry

end bumper_car_cost_correct_l1259_125968


namespace one_meter_per_minute_not_implies_uniform_speed_l1259_125917

/-- A snail's movement over time -/
structure SnailMovement where
  /-- The distance traveled by the snail in meters -/
  distance : ℝ → ℝ
  /-- The property that the snail travels 1 meter every minute -/
  travels_one_meter_per_minute : ∀ t : ℝ, distance (t + 1) - distance t = 1

/-- Definition of uniform speed -/
def UniformSpeed (s : SnailMovement) : Prop :=
  ∃ v : ℝ, ∀ t₁ t₂ : ℝ, s.distance t₂ - s.distance t₁ = v * (t₂ - t₁)

/-- Theorem stating that traveling 1 meter per minute does not imply uniform speed -/
theorem one_meter_per_minute_not_implies_uniform_speed :
  ¬(∀ s : SnailMovement, UniformSpeed s) :=
sorry

end one_meter_per_minute_not_implies_uniform_speed_l1259_125917


namespace tan_alpha_three_implies_expression_equals_two_l1259_125982

theorem tan_alpha_three_implies_expression_equals_two (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin (α - π) + Real.cos (π - α)) / 
  (Real.sin (π / 2 - α) + Real.cos (π / 2 + α)) = 2 := by
  sorry

end tan_alpha_three_implies_expression_equals_two_l1259_125982


namespace repeating_decimal_equals_fraction_l1259_125983

/-- The repeating decimal 4.6̄ -/
def repeating_decimal : ℚ := 4 + 6/9

/-- The fraction 14/3 -/
def fraction : ℚ := 14/3

/-- Theorem: The repeating decimal 4.6̄ is equal to the fraction 14/3 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l1259_125983


namespace weight_difference_proof_l1259_125963

/-- Proves the difference between the average weight of two departing students and Joe's weight --/
theorem weight_difference_proof 
  (n : ℕ) -- number of students in the original group
  (initial_avg : ℝ) -- initial average weight
  (joe_weight : ℝ) -- Joe's weight
  (new_avg : ℝ) -- new average weight after Joe joins
  (final_avg : ℝ) -- final average weight after two students leave
  (h1 : initial_avg = 30)
  (h2 : joe_weight = 43)
  (h3 : new_avg = initial_avg + 1)
  (h4 : final_avg = initial_avg)
  (h5 : (n * initial_avg + joe_weight) / (n + 1) = new_avg)
  (h6 : ((n + 1) * new_avg - 2 * final_avg) / (n - 1) = final_avg) :
  (((n + 1) * new_avg - n * final_avg) / 2) - joe_weight = -6.5 := by
  sorry

end weight_difference_proof_l1259_125963


namespace min_value_theorem_l1259_125981

theorem min_value_theorem (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0)
  (h5 : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + 4 * y^2 - c = 0 →
    (2 * x + y)^2 ≤ (2 * a + b)^2) :
  ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + 4 * y^2 - z = 0 →
    3 / a - 4 / b + 5 / c ≤ 3 / x - 4 / y + 5 / z :=
sorry

end min_value_theorem_l1259_125981


namespace sum_of_coefficients_l1259_125937

theorem sum_of_coefficients (a b : ℝ) : 
  (∃ x y : ℝ, a * x + b * y = 3 ∧ b * x + a * y = 2) →
  (3 * a + 2 * b = 3 ∧ 3 * b + 2 * a = 2) →
  a + b = 1 := by
sorry

end sum_of_coefficients_l1259_125937


namespace egyptian_fraction_sum_l1259_125928

theorem egyptian_fraction_sum : ∃! (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (11 : ℚ) / 13 = b₂ / 6 + b₃ / 24 + b₄ / 120 + b₅ / 720 + b₆ / 5040 ∧
  b₂ < 3 ∧ b₃ < 4 ∧ b₄ < 5 ∧ b₅ < 6 ∧ b₆ < 7 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 1751 := by
  sorry

end egyptian_fraction_sum_l1259_125928


namespace negation_of_all_divisible_by_two_are_even_l1259_125934

theorem negation_of_all_divisible_by_two_are_even :
  (¬ ∀ n : ℤ, n % 2 = 0 → Even n) ↔ (∃ n : ℤ, n % 2 = 0 ∧ ¬ Even n) :=
by sorry

end negation_of_all_divisible_by_two_are_even_l1259_125934


namespace problem_solution_l1259_125947

theorem problem_solution (a b c : ℝ) (h1 : a - b = 2) (h2 : a + c = 6) :
  (2*a + b + c) - 2*(a - b - c) = 12 := by sorry

end problem_solution_l1259_125947


namespace triangle_side_length_l1259_125914

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, angle B is 60°, and a² + c² = 3ac, then the length of side b is 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- Angle B is 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given equation
  (b = 2 * Real.sqrt 2) := by
  sorry

end triangle_side_length_l1259_125914


namespace derivative_at_negative_third_l1259_125924

theorem derivative_at_negative_third (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 + 2 * (deriv f (-1/3)) * x) : 
  deriv f (-1/3) = 2/3 := by
  sorry

end derivative_at_negative_third_l1259_125924


namespace binomial_8_3_l1259_125903

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end binomial_8_3_l1259_125903


namespace log_equality_implies_y_value_l1259_125922

theorem log_equality_implies_y_value (m y : ℝ) (hm : m > 0) (hy : y > 0) :
  (Real.log y / Real.log m) * (Real.log m / Real.log 7) = 4 → y = 2401 := by
  sorry

end log_equality_implies_y_value_l1259_125922


namespace min_value_of_f_inequality_abc_l1259_125962

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m :=
sorry

-- Theorem for the inequality
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_6 : a + b + c = 6) : 
  a^2 + b^2 + c^2 ≥ 12 :=
sorry

end min_value_of_f_inequality_abc_l1259_125962


namespace rectangular_field_equation_l1259_125952

theorem rectangular_field_equation (x : ℝ) : 
  (((60 - x) / 2) * ((60 + x) / 2) = 864) ↔ 
  (∃ (length width : ℝ), 
    length * width = 864 ∧ 
    length + width = 60 ∧ 
    length = width + x) :=
by sorry

end rectangular_field_equation_l1259_125952


namespace two_valid_positions_l1259_125912

/-- Represents a square in the polygon arrangement -/
structure Square :=
  (id : Char)

/-- Represents the flat arrangement of squares -/
def FlatArrangement := List Square

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
  | Right : Square → AttachmentPosition
  | Left : Square → AttachmentPosition

/-- Checks if a given attachment position allows folding into a cube with two opposite faces missing -/
def allows_cube_folding (arrangement : FlatArrangement) (pos : AttachmentPosition) : Prop :=
  sorry

/-- The main theorem stating that there are exactly two valid attachment positions -/
theorem two_valid_positions (arrangement : FlatArrangement) :
  (arrangement.length = 4) →
  (∃ A B C D : Square, arrangement = [A, B, C, D]) →
  (∃! (pos1 pos2 : AttachmentPosition),
    pos1 ≠ pos2 ∧
    allows_cube_folding arrangement pos1 ∧
    allows_cube_folding arrangement pos2 ∧
    (∀ pos, allows_cube_folding arrangement pos → (pos = pos1 ∨ pos = pos2))) :=
  sorry

end two_valid_positions_l1259_125912


namespace complex_power_difference_l1259_125927

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference : (2 + i)^12 - (2 - i)^12 = 503 * i := by
  sorry

end complex_power_difference_l1259_125927


namespace function_characterization_l1259_125946

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) :
  ∀ x : ℝ, f x = 3^x - 2^x := by
sorry

end function_characterization_l1259_125946


namespace population_increase_theorem_l1259_125978

/-- Calculates the average percent increase of population per year given initial and final populations over a specified number of years. -/
def avgPercentIncrease (initialPop finalPop : ℕ) (years : ℕ) : ℚ :=
  ((finalPop - initialPop) : ℚ) / (initialPop * years) * 100

/-- Theorem stating that the average percent increase of population per year is 5% given the specified conditions. -/
theorem population_increase_theorem :
  avgPercentIncrease 175000 262500 10 = 5 := by
  sorry

#eval avgPercentIncrease 175000 262500 10

end population_increase_theorem_l1259_125978


namespace x_plus_q_equals_five_l1259_125977

theorem x_plus_q_equals_five (x q : ℝ) (h1 : |x - 5| = q) (h2 : x < 5) : x + q = 5 := by
  sorry

end x_plus_q_equals_five_l1259_125977


namespace shaded_area_is_correct_l1259_125996

/-- A square and a right triangle with equal height -/
structure GeometricSetup where
  /-- Height of both the square and the triangle -/
  height : ℝ
  /-- Base length of both the square and the triangle -/
  base : ℝ
  /-- The lower right vertex of the square and lower left vertex of the triangle -/
  intersection : ℝ × ℝ
  /-- Assertion that the height equals the base -/
  height_eq_base : height = base
  /-- Assertion that the intersection point is at (15, 0) -/
  intersection_is_fifteen : intersection = (15, 0)
  /-- Assertion that the base length is 15 -/
  base_is_fifteen : base = 15

/-- The area of the shaded region -/
def shaded_area (setup : GeometricSetup) : ℝ := 168.75

/-- Theorem stating that the shaded area is 168.75 square units -/
theorem shaded_area_is_correct (setup : GeometricSetup) : 
  shaded_area setup = 168.75 := by
  sorry

end shaded_area_is_correct_l1259_125996


namespace video_game_players_l1259_125972

theorem video_game_players (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : 
  players_quit = 8 →
  lives_per_player = 6 →
  total_lives = 30 →
  players_quit + (total_lives / lives_per_player) = 13 := by
sorry

end video_game_players_l1259_125972


namespace book_organizing_group_size_l1259_125943

/-- Represents the number of hours of work for one person to complete the task -/
def total_hours : ℕ := 40

/-- Represents the number of hours worked by the initial group -/
def initial_hours : ℕ := 2

/-- Represents the number of hours worked by the remaining group -/
def remaining_hours : ℕ := 4

/-- Represents the number of people who left the group -/
def people_left : ℕ := 2

theorem book_organizing_group_size :
  ∃ (initial_group : ℕ),
    (initial_hours : ℚ) / total_hours * initial_group + 
    (remaining_hours : ℚ) / total_hours * (initial_group - people_left) = 1 ∧
    initial_group = 8 := by
  sorry

end book_organizing_group_size_l1259_125943


namespace probability_two_green_marbles_l1259_125901

/-- The probability of drawing two green marbles without replacement from a jar -/
theorem probability_two_green_marbles (red green white blue : ℕ) 
  (h_red : red = 3)
  (h_green : green = 4)
  (h_white : white = 10)
  (h_blue : blue = 5) :
  let total := red + green + white + blue
  let prob_first := green / total
  let prob_second := (green - 1) / (total - 1)
  prob_first * prob_second = 2 / 77 := by
  sorry

end probability_two_green_marbles_l1259_125901


namespace green_pill_cost_l1259_125923

/-- The cost of Al's pills for three weeks of treatment --/
def total_cost : ℚ := 1092

/-- The number of days in the treatment period --/
def treatment_days : ℕ := 21

/-- The number of times Al takes a blue pill --/
def blue_pill_count : ℕ := 10

/-- The cost difference between a green pill and a pink pill --/
def green_pink_diff : ℚ := 2

/-- The cost of a pink pill --/
def pink_cost : ℚ := 1050 / 62

/-- The cost of a green pill --/
def green_cost : ℚ := pink_cost + green_pink_diff

/-- Theorem stating the cost of a green pill --/
theorem green_pill_cost : green_cost = 587 / 31 := by sorry

end green_pill_cost_l1259_125923


namespace four_digit_with_five_or_seven_l1259_125961

theorem four_digit_with_five_or_seven (total_four_digit : Nat) (without_five_or_seven : Nat) :
  total_four_digit = 9000 →
  without_five_or_seven = 3584 →
  total_four_digit - without_five_or_seven = 5416 := by
  sorry

end four_digit_with_five_or_seven_l1259_125961


namespace power_division_equals_27_l1259_125976

theorem power_division_equals_27 : 3^12 / 27^3 = 27 := by
  sorry

end power_division_equals_27_l1259_125976


namespace midpoint_triangle_area_ratio_l1259_125936

/-- Given a triangle with area S, N is the area of the triangle formed by connecting
    the midpoints of its sides, and P is the area of the triangle formed by connecting
    the midpoints of the sides of the triangle with area N. -/
theorem midpoint_triangle_area_ratio (S N P : ℝ) (hS : S > 0) (hN : N > 0) (hP : P > 0)
  (hN_def : N = S / 4) (hP_def : P = N / 4) : P / S = 1 / 16 := by
  sorry

end midpoint_triangle_area_ratio_l1259_125936


namespace mi_gu_li_fen_problem_l1259_125919

/-- The "Mi-Gu-Li-Fen" problem from the "Mathematical Treatise in Nine Sections" -/
theorem mi_gu_li_fen_problem (total_mixture : ℚ) (sample_size : ℕ) (wheat_in_sample : ℕ) 
  (h1 : total_mixture = 1512)
  (h2 : sample_size = 216)
  (h3 : wheat_in_sample = 27) :
  (total_mixture * (wheat_in_sample : ℚ) / (sample_size : ℚ)) = 189 := by
  sorry

end mi_gu_li_fen_problem_l1259_125919


namespace max_value_sqrt_sum_l1259_125954

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-36) 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 := by
  sorry

end max_value_sqrt_sum_l1259_125954


namespace magical_elixir_combinations_l1259_125944

/-- The number of magical herbs. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals. -/
def num_crystals : ℕ := 6

/-- The number of incompatible herb-crystal pairs. -/
def num_incompatible : ℕ := 3

/-- The number of valid combinations for the magical elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible

theorem magical_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end magical_elixir_combinations_l1259_125944


namespace bisected_areas_correct_l1259_125913

/-- A rectangle with sides 2 meters and 4 meters, divided by angle bisectors -/
structure BisectedRectangle where
  /-- The length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- The length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The short side is 2 meters -/
  short_side_eq : short_side = 2
  /-- The long side is 4 meters -/
  long_side_eq : long_side = 4
  /-- The angle bisectors are drawn from angles adjacent to the longer side -/
  bisectors_from_long_side : Bool

/-- The areas into which the rectangle is divided by the angle bisectors -/
def bisected_areas (rect : BisectedRectangle) : List ℝ :=
  [2, 2, 4]

/-- Theorem stating that the bisected areas are correct -/
theorem bisected_areas_correct (rect : BisectedRectangle) :
  bisected_areas rect = [2, 2, 4] := by
  sorry

end bisected_areas_correct_l1259_125913


namespace triangular_prism_volume_l1259_125916

/-- The volume of a triangular prism with a right triangle base and specific conditions -/
theorem triangular_prism_volume (PQ PR h θ : ℝ) : 
  PQ = Real.sqrt 5 →
  PR = Real.sqrt 5 →
  Real.tan θ = h / Real.sqrt 5 →
  Real.sin θ = 3 / 5 →
  (1 / 2 * PQ * PR) * h = 15 * Real.sqrt 5 / 8 := by
  sorry

#check triangular_prism_volume

end triangular_prism_volume_l1259_125916


namespace fraction_of_states_1790s_l1259_125909

/-- The number of states that joined the union during 1790-1799 -/
def states_joined_1790s : ℕ := 7

/-- The total number of states considered -/
def total_states : ℕ := 30

/-- The fraction of states that joined during 1790-1799 out of the first 30 states -/
theorem fraction_of_states_1790s :
  (states_joined_1790s : ℚ) / total_states = 7 / 30 := by sorry

end fraction_of_states_1790s_l1259_125909


namespace sqrt_expression_simplification_l1259_125915

theorem sqrt_expression_simplification :
  (Real.sqrt 48 + Real.sqrt 20) - (Real.sqrt 12 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 := by
  sorry

end sqrt_expression_simplification_l1259_125915


namespace largest_prime_factor_l1259_125931

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (18^4 + 3 * 18^2 + 1 - 17^4) ∧ 
   ∀ q : ℕ, Nat.Prime q → q ∣ (18^4 + 3 * 18^2 + 1 - 17^4) → q ≤ p) →
  (∃ p : ℕ, p = 307 ∧ Nat.Prime p ∧ p ∣ (18^4 + 3 * 18^2 + 1 - 17^4) ∧ 
   ∀ q : ℕ, Nat.Prime q → q ∣ (18^4 + 3 * 18^2 + 1 - 17^4) → q ≤ p) :=
by sorry

end largest_prime_factor_l1259_125931


namespace solution_set_real_iff_k_less_than_neg_three_l1259_125905

theorem solution_set_real_iff_k_less_than_neg_three (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) ↔ k < -3 := by
  sorry

end solution_set_real_iff_k_less_than_neg_three_l1259_125905


namespace root_of_polynomial_l1259_125992

theorem root_of_polynomial (b : ℝ) (h : b^5 = 2 - Real.sqrt 3) :
  (b + (2 + Real.sqrt 3)^(1/5 : ℝ))^5 - 5*(b + (2 + Real.sqrt 3)^(1/5 : ℝ))^3 + 
  5*(b + (2 + Real.sqrt 3)^(1/5 : ℝ)) - 4 = 0 :=
sorry

end root_of_polynomial_l1259_125992


namespace valid_partition_exists_l1259_125932

/-- Represents a person with their country and position on the circle. -/
structure Person where
  country : Fin 50
  position : Fin 100

/-- Represents a partition of people into two groups. -/
def Partition := Fin 100 → Fin 2

/-- Checks if two people are from the same country. -/
def sameCountry (p1 p2 : Person) : Prop := p1.country = p2.country

/-- Checks if two people are consecutive on the circle. -/
def consecutive (p1 p2 : Person) : Prop :=
  (p1.position + 1) % 100 = p2.position ∨ (p2.position + 1) % 100 = p1.position

/-- The main theorem stating the existence of a valid partition. -/
theorem valid_partition_exists :
  ∃ (people : Fin 100 → Person) (partition : Partition),
    (∀ i : Fin 100, ∃! j : Fin 100, i ≠ j ∧ sameCountry (people i) (people j)) ∧
    (∀ i j : Fin 100, sameCountry (people i) (people j) → partition i ≠ partition j) ∧
    (∀ i j k : Fin 100, consecutive (people i) (people j) ∧ consecutive (people j) (people k) →
      ¬(partition i = partition j ∧ partition j = partition k)) :=
  sorry

end valid_partition_exists_l1259_125932


namespace triangle_abc_is_right_angled_l1259_125904

theorem triangle_abc_is_right_angled (A B C : ℝ) (h1 : A = 60) (h2 : B = 3 * C) 
  (h3 : A + B + C = 180) : B = 90 :=
by sorry

end triangle_abc_is_right_angled_l1259_125904


namespace geometric_sequence_sum_l1259_125938

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 324 and a_3 + a_4 = 36,
    prove that a_5 + a_6 = 4 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 1 + a 2 = 324 →
  a 3 + a 4 = 36 →
  a 5 + a 6 = 4 := by
sorry

end geometric_sequence_sum_l1259_125938


namespace work_completion_proof_l1259_125925

/-- The number of days it takes x to complete the work -/
def x_total_days : ℝ := 40

/-- The number of days it takes y to complete the work -/
def y_total_days : ℝ := 35

/-- The number of days y worked to finish the work after x stopped -/
def y_actual_days : ℝ := 28

/-- The number of days x worked before y took over -/
def x_worked_days : ℝ := 8

theorem work_completion_proof :
  x_worked_days / x_total_days + y_actual_days / y_total_days = 1 := by
  sorry


end work_completion_proof_l1259_125925


namespace sufficient_not_necessary_l1259_125948

/-- The quadratic function f(x) = x^2 + 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- f has a root -/
def has_root (m : ℝ) : Prop := ∃ x, f m x = 0

/-- m < 1 is sufficient but not necessary for f to have a root -/
theorem sufficient_not_necessary :
  (∀ m, m < 1 → has_root m) ∧ 
  (∃ m, ¬(m < 1) ∧ has_root m) :=
sorry

end sufficient_not_necessary_l1259_125948


namespace ball_box_arrangement_l1259_125955

/-- The number of ways to place n different balls into k boxes -/
def total_arrangements (n k : ℕ) : ℕ := k^n

/-- The number of ways to place n different balls into k boxes, 
    with at least one ball in a specific box -/
def arrangements_with_specific_box (n k : ℕ) : ℕ := 
  total_arrangements n k - total_arrangements n (k-1)

theorem ball_box_arrangement : 
  arrangements_with_specific_box 3 6 = 91 := by sorry

end ball_box_arrangement_l1259_125955


namespace rectangle_length_l1259_125993

/-- Proves that a rectangle with perimeter to width ratio of 5:1 and area 150 has length 15 -/
theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2 * l + 2 * w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end rectangle_length_l1259_125993


namespace find_divisor_l1259_125907

theorem find_divisor (x y n : ℕ+) : 
  x = n * y + 4 →
  2 * x = 14 * y + 1 →
  5 * y - x = 3 →
  n = 4 := by
sorry

end find_divisor_l1259_125907


namespace more_girls_than_boys_l1259_125921

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 42 →
  3 * girls = 4 * boys →
  total_students = boys + girls →
  girls - boys = 6 := by
sorry

end more_girls_than_boys_l1259_125921


namespace broken_glass_problem_l1259_125906

/-- The number of broken glass pieces during transportation --/
def broken_glass (total : ℕ) (safe_fee : ℕ) (compensation : ℕ) (total_fee : ℕ) : ℕ :=
  total - (total_fee + total * safe_fee) / (safe_fee + compensation)

theorem broken_glass_problem :
  broken_glass 100 3 5 260 = 5 := by
  sorry

end broken_glass_problem_l1259_125906


namespace blue_jellybean_probability_l1259_125953

/-- The probability of drawing 3 blue jellybeans in succession from a bag 
    containing 10 red and 10 blue jellybeans, without replacement. -/
theorem blue_jellybean_probability : 
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  
  -- The probability is calculated as the product of individual probabilities
  (blue_jellybeans / total_jellybeans) * 
  ((blue_jellybeans - 1) / (total_jellybeans - 1)) * 
  ((blue_jellybeans - 2) / (total_jellybeans - 2)) = 2 / 19 := by
sorry


end blue_jellybean_probability_l1259_125953


namespace time_conversion_not_100_l1259_125929

/-- Represents the conversion rate between adjacent time units -/
def time_conversion_rate : ℕ := 60

/-- The set of standard time units -/
inductive TimeUnit
| Hour
| Minute
| Second

theorem time_conversion_not_100 : time_conversion_rate ≠ 100 := by
  sorry

end time_conversion_not_100_l1259_125929


namespace product_equals_48_l1259_125967

theorem product_equals_48 : 12 * (1 / 7) * 14 * 2 = 48 := by
  sorry

end product_equals_48_l1259_125967


namespace line_curve_intersection_l1259_125974

-- Define the line
def line (x : ℝ) : ℝ := x + 3

-- Define the curve
def curve (x y : ℝ) : Prop := y^2 / 9 - x * abs x / 4 = 1

-- State the theorem
theorem line_curve_intersection :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = 3 ∧ 
    (∀ p ∈ points, curve p.1 p.2 ∧ p.2 = line p.1) ∧
    (∀ x y, curve x y ∧ y = line x → (x, y) ∈ points) :=
sorry

end line_curve_intersection_l1259_125974


namespace second_quarter_profit_l1259_125918

def annual_profit : ℕ := 8000
def first_quarter_profit : ℕ := 1500
def third_quarter_profit : ℕ := 3000
def fourth_quarter_profit : ℕ := 2000

theorem second_quarter_profit :
  annual_profit - (first_quarter_profit + third_quarter_profit + fourth_quarter_profit) = 1500 :=
by sorry

end second_quarter_profit_l1259_125918


namespace pizza_slices_theorem_l1259_125965

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 16

/-- Represents the number of people eating pizza -/
def num_people : ℕ := 4

/-- Represents the number of people eating both types of pizza -/
def num_people_both_types : ℕ := 3

/-- Represents the number of cheese slices left -/
def cheese_slices_left : ℕ := 7

/-- Represents the number of pepperoni slices left -/
def pepperoni_slices_left : ℕ := 1

/-- Represents the total number of slices each person eats -/
def slices_per_person : ℕ := 6

theorem pizza_slices_theorem :
  slices_per_person * num_people = 
    2 * slices_per_pizza - cheese_slices_left - pepperoni_slices_left :=
by sorry

end pizza_slices_theorem_l1259_125965


namespace sum_of_ages_is_twelve_l1259_125941

/-- The sum of ages of four children born at one-year intervals -/
def sum_of_ages (youngest_age : ℝ) : ℝ :=
  youngest_age + (youngest_age + 1) + (youngest_age + 2) + (youngest_age + 3)

/-- Theorem: The sum of ages of four children, where the youngest is 1.5 years old
    and each subsequent child is 1 year older, is 12 years. -/
theorem sum_of_ages_is_twelve :
  sum_of_ages 1.5 = 12 := by sorry

end sum_of_ages_is_twelve_l1259_125941


namespace angle_y_value_l1259_125979

-- Define the angles in the diagram
variable (x y : ℝ)

-- Define the conditions given in the problem
axiom AB_parallel_CD : True  -- We can't directly represent parallel lines, so we use this as a placeholder
axiom angle_BMN : x = 2 * x
axiom angle_MND : x = 70
axiom angle_NMP : x = 70

-- Theorem to prove
theorem angle_y_value : y = 55 := by
  sorry

end angle_y_value_l1259_125979


namespace regular_polygon_perimeter_l1259_125964

/-- The perimeter of a regular polygon with side length 8 and exterior angle 72 degrees is 40 units. -/
theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) : 
  side_length = 8 → 
  exterior_angle = 72 → 
  (360 / exterior_angle) * side_length = 40 := by
sorry

end regular_polygon_perimeter_l1259_125964


namespace museum_ticket_fraction_l1259_125945

theorem museum_ticket_fraction (total_money : ℚ) (sandwich_fraction : ℚ) (book_fraction : ℚ) (leftover : ℚ) : 
  total_money = 120 →
  sandwich_fraction = 1/5 →
  book_fraction = 1/2 →
  leftover = 16 →
  (total_money - (sandwich_fraction * total_money + book_fraction * total_money + leftover)) / total_money = 1/6 := by
  sorry

end museum_ticket_fraction_l1259_125945


namespace pipe_problem_l1259_125989

theorem pipe_problem (fill_time : ℕ → ℝ) (h1 : fill_time 2 = 18) (h2 : ∃ n : ℕ, fill_time n = 12) : 
  ∃ n : ℕ, n = 3 ∧ fill_time n = 12 := by
  sorry

end pipe_problem_l1259_125989


namespace correct_substitution_proof_l1259_125975

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 5
def equation2 (x y : ℝ) : Prop := y = 1 + x

-- Define the correct substitution
def correct_substitution (x : ℝ) : Prop := 2 * x - 1 - x = 5

-- Theorem statement
theorem correct_substitution_proof :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → correct_substitution x :=
by sorry

end correct_substitution_proof_l1259_125975


namespace binomial_congruence_l1259_125956

theorem binomial_congruence (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  (Nat.choose (p - 1) k) ≡ ((-1 : ℤ) ^ k) [ZMOD p] := by
  sorry

end binomial_congruence_l1259_125956


namespace class_average_after_exclusion_l1259_125900

theorem class_average_after_exclusion (total_students : ℕ) (initial_avg : ℚ) 
  (excluded_students : ℕ) (excluded_avg : ℚ) :
  total_students = 30 →
  initial_avg = 80 →
  excluded_students = 5 →
  excluded_avg = 30 →
  let remaining_students := total_students - excluded_students
  let total_marks := initial_avg * total_students
  let excluded_marks := excluded_avg * excluded_students
  let remaining_marks := total_marks - excluded_marks
  (remaining_marks / remaining_students) = 90 := by
  sorry

end class_average_after_exclusion_l1259_125900


namespace matrix19_sum_nonzero_l1259_125957

def Matrix19 := Fin 19 → Fin 19 → Int

def isValidMatrix (A : Matrix19) : Prop :=
  ∀ i j, A i j = 1 ∨ A i j = -1

def rowProduct (A : Matrix19) (i : Fin 19) : Int :=
  (Finset.univ.prod fun j => A i j)

def colProduct (A : Matrix19) (j : Fin 19) : Int :=
  (Finset.univ.prod fun i => A i j)

theorem matrix19_sum_nonzero (A : Matrix19) (h : isValidMatrix A) :
  (Finset.univ.sum fun i => rowProduct A i) + (Finset.univ.sum fun j => colProduct A j) ≠ 0 := by
  sorry

end matrix19_sum_nonzero_l1259_125957


namespace brand_z_percentage_l1259_125973

theorem brand_z_percentage (tank_capacity : ℝ) (brand_z_amount : ℝ) (brand_x_amount : ℝ)
  (h1 : tank_capacity > 0)
  (h2 : brand_z_amount = 1/8 * tank_capacity)
  (h3 : brand_x_amount = 7/8 * tank_capacity)
  (h4 : brand_z_amount + brand_x_amount = tank_capacity) :
  (brand_z_amount / tank_capacity) * 100 = 12.5 := by
sorry

end brand_z_percentage_l1259_125973


namespace remainder_problem_l1259_125902

theorem remainder_problem : 2851 * 7347 * 419^2 % 10 = 7 := by
  sorry

end remainder_problem_l1259_125902


namespace inheritance_tax_problem_l1259_125998

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 12000) → 
  (round x : ℤ) = 33097 := by
  sorry

end inheritance_tax_problem_l1259_125998


namespace least_number_with_remainders_l1259_125910

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 56 = 3 ∧
  n % 78 = 3 ∧
  n % 9 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 56 = 3 ∧ m % 78 = 3 ∧ m % 9 = 0 → n ≤ m :=
by
  -- Proof goes here
  sorry

end least_number_with_remainders_l1259_125910


namespace integral_sqrt_one_minus_x_squared_plus_x_cos_x_l1259_125985

open Set
open MeasureTheory
open Real

theorem integral_sqrt_one_minus_x_squared_plus_x_cos_x (f : ℝ → ℝ) :
  (∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x * Real.cos x)) = π / 2 := by
  sorry

end integral_sqrt_one_minus_x_squared_plus_x_cos_x_l1259_125985


namespace linear_function_value_l1259_125951

theorem linear_function_value (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : f = fun x ↦ a * x + b)
    (h2 : f 1 = 2017)
    (h3 : f 2 = 2018) : 
  f 2019 = 4035 := by
sorry

end linear_function_value_l1259_125951


namespace tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half_l1259_125991

theorem tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half :
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half_l1259_125991


namespace fraction_simplification_l1259_125986

theorem fraction_simplification : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256) / 
  (2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512) = 1 / 2 := by
  sorry

end fraction_simplification_l1259_125986


namespace system_no_solution_l1259_125970

theorem system_no_solution (n : ℝ) : 
  (∀ x y z : ℝ, n^2 * x + y ≠ 1 ∨ n * y + z ≠ 1 ∨ x + n^2 * z ≠ 1) ↔ n = -1 := by
  sorry

end system_no_solution_l1259_125970


namespace investment_percentage_problem_l1259_125942

theorem investment_percentage_problem (x y : ℝ) (P : ℝ) : 
  x + y = 2000 →
  y = 600 →
  0.1 * x - (P / 100) * y = 92 →
  P = 8 := by
sorry

end investment_percentage_problem_l1259_125942


namespace surprise_combinations_for_week_l1259_125940

/-- The number of combinations for surprise gift placement --/
def surprise_combinations (monday tuesday wednesday thursday friday : ℕ) : ℕ :=
  monday * tuesday * wednesday * thursday * friday

/-- Theorem stating the total number of combinations for the given week --/
theorem surprise_combinations_for_week :
  surprise_combinations 2 1 1 4 1 = 8 := by
  sorry

end surprise_combinations_for_week_l1259_125940


namespace sports_lottery_winners_l1259_125908

theorem sports_lottery_winners
  (win : Prop → Prop)
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B → (C ∨ ¬A))
  (h3 : ¬D → (A ∧ ¬C))
  (h4 : D → A) :
  A ∧ B ∧ C ∧ D :=
by sorry

end sports_lottery_winners_l1259_125908


namespace boat_distance_along_stream_l1259_125949

/-- The distance traveled by a boat along a stream in one hour -/
def distance_along_stream (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  boat_speed + (boat_speed - against_stream_distance)

/-- Theorem: The distance traveled by the boat along the stream in one hour is 8 km -/
theorem boat_distance_along_stream :
  distance_along_stream 5 2 = 8 := by
  sorry

end boat_distance_along_stream_l1259_125949


namespace circle_center_and_equation_l1259_125939

/-- A circle passing through two points with a given radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passesThrough : (ℝ × ℝ) → Prop

/-- The line passing through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

theorem circle_center_and_equation 
  (C : Circle) 
  (h1 : C.passesThrough (1, 0)) 
  (h2 : C.passesThrough (0, 1)) 
  (h3 : C.radius = 1) : 
  (∃ t : ℝ, C.center = (t, t)) ∧ 
  (∀ x y : ℝ, C.passesThrough (x, y) ↔ x^2 + y^2 = 1) :=
sorry

end circle_center_and_equation_l1259_125939


namespace min_value_theorem_l1259_125935

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1 / x + 2 / y = 1 → 
  2 / (x - 1) + 1 / (y - 2) ≥ m := by
  sorry

end min_value_theorem_l1259_125935


namespace sugar_amount_in_new_recipe_l1259_125988

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of an ingredient based on a given amount of another ingredient -/
def calculate_ingredient (ratio : RecipeRatio) (known_amount : ℚ) (known_part : ℚ) (target_part : ℚ) : ℚ :=
  (target_part / known_part) * known_amount

theorem sugar_amount_in_new_recipe 
  (original_ratio : RecipeRatio)
  (h_original : original_ratio = ⟨11, 5, 2⟩)
  (new_ratio : RecipeRatio)
  (h_double_flour_water : new_ratio.flour / new_ratio.water = 2 * (original_ratio.flour / original_ratio.water))
  (h_half_flour_sugar : new_ratio.flour / new_ratio.sugar = (1/2) * (original_ratio.flour / original_ratio.sugar))
  (h_water_amount : calculate_ingredient new_ratio 7.5 new_ratio.water new_ratio.sugar = 6) :
  calculate_ingredient new_ratio 7.5 new_ratio.water new_ratio.sugar = 6 := by
  sorry

end sugar_amount_in_new_recipe_l1259_125988


namespace shelf_filling_problem_l1259_125997

/-- Represents the shelf filling problem with biology and geography books -/
theorem shelf_filling_problem 
  (B G P Q K : ℕ) 
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ K ∧ 
                G ≠ P ∧ G ≠ Q ∧ G ≠ K ∧ 
                P ≠ Q ∧ P ≠ K ∧ 
                Q ≠ K)
  (h_positive : B > 0 ∧ G > 0 ∧ P > 0 ∧ Q > 0 ∧ K > 0)
  (h_fill1 : ∃ (a : ℚ), a > 0 ∧ B * a + G * (2 * a) = K * a)
  (h_fill2 : ∃ (a : ℚ), a > 0 ∧ P * a + Q * (2 * a) = K * a) :
  K = B + 2 * G :=
sorry

end shelf_filling_problem_l1259_125997


namespace profit_percent_calculation_l1259_125987

/-- Proves that the profit percent is 140% when selling an article at a certain price,
    given that selling it at 1/3 of that price results in a 20% loss. -/
theorem profit_percent_calculation (C S : ℝ) (h : (1/3) * S = 0.8 * C) :
  (S - C) / C * 100 = 140 := by
  sorry

end profit_percent_calculation_l1259_125987
