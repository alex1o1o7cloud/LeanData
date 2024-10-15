import Mathlib

namespace NUMINAMATH_CALUDE_inconsistent_weight_problem_l2621_262199

theorem inconsistent_weight_problem :
  ∀ (initial_students : ℕ) (initial_avg_weight : ℝ) 
    (new_students : ℕ) (new_avg_weight : ℝ) 
    (first_new_student_weight : ℝ) (second_new_student_min_weight : ℝ),
  initial_students = 19 →
  initial_avg_weight = 15 →
  new_students = 2 →
  new_avg_weight = 14.6 →
  first_new_student_weight = 12 →
  second_new_student_min_weight = 14 →
  ¬∃ (second_new_student_weight : ℝ),
    (initial_students * initial_avg_weight + first_new_student_weight + second_new_student_weight) / 
      (initial_students + new_students) = new_avg_weight ∧
    second_new_student_weight ≥ second_new_student_min_weight :=
by sorry

end NUMINAMATH_CALUDE_inconsistent_weight_problem_l2621_262199


namespace NUMINAMATH_CALUDE_column_for_2023_l2621_262165

def column_sequence : Fin 8 → Char
  | 0 => 'B'
  | 1 => 'C'
  | 2 => 'D'
  | 3 => 'E'
  | 4 => 'D'
  | 5 => 'C'
  | 6 => 'B'
  | 7 => 'A'

def column_for_number (n : ℕ) : Char :=
  column_sequence ((n - 2) % 8)

theorem column_for_2023 : column_for_number 2023 = 'C' := by
  sorry

end NUMINAMATH_CALUDE_column_for_2023_l2621_262165


namespace NUMINAMATH_CALUDE_green_ball_count_l2621_262152

theorem green_ball_count (blue_count : ℕ) (ratio_blue : ℕ) (ratio_green : ℕ) 
  (h1 : blue_count = 16)
  (h2 : ratio_blue = 4)
  (h3 : ratio_green = 3) :
  (blue_count * ratio_green) / ratio_blue = 12 :=
by sorry

end NUMINAMATH_CALUDE_green_ball_count_l2621_262152


namespace NUMINAMATH_CALUDE_fresh_driving_hours_l2621_262191

/-- Calculates the number of hours driving fresh given total distance, total time, and speeds -/
theorem fresh_driving_hours (total_distance : ℝ) (total_time : ℝ) (fresh_speed : ℝ) (fatigued_speed : ℝ) 
  (h1 : total_distance = 152)
  (h2 : total_time = 9)
  (h3 : fresh_speed = 25)
  (h4 : fatigued_speed = 15) :
  ∃ x : ℝ, x = 17 / 10 ∧ fresh_speed * x + fatigued_speed * (total_time - x) = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_fresh_driving_hours_l2621_262191


namespace NUMINAMATH_CALUDE_initial_marbles_count_initial_marbles_proof_l2621_262104

def marbles_to_juan : ℕ := 1835
def marbles_to_lisa : ℕ := 985
def marbles_left : ℕ := 5930

theorem initial_marbles_count : ℕ :=
  marbles_to_juan + marbles_to_lisa + marbles_left

#check initial_marbles_count

theorem initial_marbles_proof : initial_marbles_count = 8750 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_count_initial_marbles_proof_l2621_262104


namespace NUMINAMATH_CALUDE_unique_solution_system_l2621_262177

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    x^2 - 2*x - 4*z = 3 ∧
    y^2 - 2*y - 2*x = -14 ∧
    z^2 - 4*y - 4*z = -18 ∧
    x = 2 ∧ y = 3 ∧ z = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2621_262177


namespace NUMINAMATH_CALUDE_first_course_cost_proof_l2621_262167

/-- The cost of Amelia's dinner --/
def dinner_cost : ℝ := 60

/-- The amount Amelia has left after buying all meals --/
def remaining_amount : ℝ := 20

/-- The additional cost of the second course compared to the first --/
def second_course_additional_cost : ℝ := 5

/-- The ratio of the dessert cost to the second course cost --/
def dessert_ratio : ℝ := 0.25

/-- The cost of the first course --/
def first_course_cost : ℝ := 15

theorem first_course_cost_proof :
  ∃ (x : ℝ),
    x = first_course_cost ∧
    dinner_cost - remaining_amount = x + (x + second_course_additional_cost) + dessert_ratio * (x + second_course_additional_cost) :=
by sorry

end NUMINAMATH_CALUDE_first_course_cost_proof_l2621_262167


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2621_262180

/-- The range of k for which the quadratic equation (k+1)x^2 - 2x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k + 1) * x₁^2 - 2 * x₁ + 1 = 0 ∧ 
    (k + 1) * x₂^2 - 2 * x₂ + 1 = 0) ↔ 
  (k ≤ 0 ∧ k ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2621_262180


namespace NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l2621_262112

theorem series_sum_equals_three_fourths : 
  ∑' k, (k : ℝ) / (3 : ℝ) ^ k = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l2621_262112


namespace NUMINAMATH_CALUDE_tax_fraction_proof_l2621_262160

theorem tax_fraction_proof (gross_income : ℝ) (car_payment : ℝ) (car_payment_percentage : ℝ) :
  gross_income = 3000 →
  car_payment = 400 →
  car_payment_percentage = 0.20 →
  car_payment = car_payment_percentage * (gross_income * (1 - (1/3))) →
  1/3 = (gross_income - (car_payment / car_payment_percentage)) / gross_income :=
by sorry

end NUMINAMATH_CALUDE_tax_fraction_proof_l2621_262160


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2621_262121

theorem fraction_equation_solution (x : ℝ) : 
  (3 - x) / (2 - x) - 1 / (x - 2) = 3 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2621_262121


namespace NUMINAMATH_CALUDE_bug_probability_l2621_262148

def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

theorem bug_probability : P 12 = 683/2048 := by sorry

end NUMINAMATH_CALUDE_bug_probability_l2621_262148


namespace NUMINAMATH_CALUDE_average_velocity_proof_l2621_262185

/-- The average velocity of a particle with motion equation s(t) = 4 - 2t² 
    over the time interval [1, 1+Δt] is equal to -4 - 2Δt. -/
theorem average_velocity_proof (Δt : ℝ) : 
  let s (t : ℝ) := 4 - 2 * t^2
  let v_avg := (s (1 + Δt) - s 1) / Δt
  v_avg = -4 - 2 * Δt :=
by sorry

end NUMINAMATH_CALUDE_average_velocity_proof_l2621_262185


namespace NUMINAMATH_CALUDE_runner_speeds_l2621_262144

/-- The speed of runner A in meters per second -/
def speed_A : ℝ := 9

/-- The speed of runner B in meters per second -/
def speed_B : ℝ := 7

/-- The length of the circular track in meters -/
def track_length : ℝ := 400

/-- The time in seconds it takes for A and B to meet when running in opposite directions -/
def opposite_meeting_time : ℝ := 25

/-- The time in seconds it takes for A to catch up with B when running in the same direction -/
def same_direction_catchup_time : ℝ := 200

theorem runner_speeds :
  speed_A * opposite_meeting_time + speed_B * opposite_meeting_time = track_length ∧
  speed_A * same_direction_catchup_time - speed_B * same_direction_catchup_time = track_length :=
by sorry

end NUMINAMATH_CALUDE_runner_speeds_l2621_262144


namespace NUMINAMATH_CALUDE_students_left_l2621_262102

theorem students_left (initial_boys initial_girls boys_dropout girls_dropout : ℕ) 
  (h1 : initial_boys = 14)
  (h2 : initial_girls = 10)
  (h3 : boys_dropout = 4)
  (h4 : girls_dropout = 3) :
  initial_boys - boys_dropout + (initial_girls - girls_dropout) = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_left_l2621_262102


namespace NUMINAMATH_CALUDE_total_blue_balloons_l2621_262186

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 9

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 5

/-- The number of blue balloons Jessica has -/
def jessica_balloons : ℕ := 2

/-- The total number of blue balloons -/
def total_balloons : ℕ := joan_balloons + sally_balloons + jessica_balloons

theorem total_blue_balloons : total_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l2621_262186


namespace NUMINAMATH_CALUDE_pencil_packing_problem_l2621_262106

theorem pencil_packing_problem :
  ∃ a : ℕ, 200 ≤ a ∧ a ≤ 300 ∧ 
    a % 10 = 7 ∧ 
    a % 12 = 9 ∧
    (a = 237 ∨ a = 297) := by
  sorry

end NUMINAMATH_CALUDE_pencil_packing_problem_l2621_262106


namespace NUMINAMATH_CALUDE_oil_tank_capacity_oil_tank_capacity_proof_l2621_262190

theorem oil_tank_capacity : ℝ → Prop :=
  fun t => 
    (∃ o : ℝ, o / t = 1 / 6 ∧ (o + 4) / t = 1 / 3) → t = 24

-- The proof is omitted
theorem oil_tank_capacity_proof : oil_tank_capacity 24 :=
  sorry

end NUMINAMATH_CALUDE_oil_tank_capacity_oil_tank_capacity_proof_l2621_262190


namespace NUMINAMATH_CALUDE_polygon_sides_l2621_262109

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2621_262109


namespace NUMINAMATH_CALUDE_w_squared_value_l2621_262172

theorem w_squared_value (w : ℝ) (h : (w + 13)^2 = (3*w + 7)*(2*w + 4)) : w^2 = 141/5 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l2621_262172


namespace NUMINAMATH_CALUDE_bus_problem_l2621_262184

/-- Calculates the number of students remaining on a bus after a given number of stops,
    where half of the students get off at each stop. -/
def studentsRemaining (initial : ℕ) (stops : ℕ) : ℕ :=
  initial / (2 ^ stops)

/-- Theorem: If a bus starts with 48 students and half of the remaining students get off
    at each of three consecutive stops, then 6 students will remain on the bus after the third stop. -/
theorem bus_problem : studentsRemaining 48 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2621_262184


namespace NUMINAMATH_CALUDE_sine_shift_left_l2621_262168

/-- Shifting a sine function to the left -/
theorem sine_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let shift : ℝ := π / 6
  let g (t : ℝ) := f (t + shift)
  g x = Real.sin (x + π / 6) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_left_l2621_262168


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2621_262196

/-- Given an isosceles right triangle with a square inscribed as described in Figure 1
    with an area of 256 cm², prove that the area of the square inscribed as described
    in Figure 2 is 576 - 256√2 cm². -/
theorem inscribed_square_area (s : ℝ) (h1 : s^2 = 256) : ∃ S : ℝ,
  S^2 = 576 - 256 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2621_262196


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2621_262140

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2621_262140


namespace NUMINAMATH_CALUDE_intersection_distance_l2621_262198

theorem intersection_distance : ∃ (p₁ p₂ : ℝ × ℝ),
  (p₁.1^2 + p₁.2 = 10 ∧ p₁.1 + p₁.2 = 10) ∧
  (p₂.1^2 + p₂.2 = 10 ∧ p₂.1 + p₂.2 = 10) ∧
  p₁ ≠ p₂ ∧
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2621_262198


namespace NUMINAMATH_CALUDE_cube_sum_digits_eq_square_self_l2621_262192

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The set of solutions to the problem -/
def solutionSet : Set ℕ := {1, 27}

/-- The main theorem -/
theorem cube_sum_digits_eq_square_self :
  ∀ n : ℕ, n < 1000 → (sumOfDigits n)^3 = n^2 ↔ n ∈ solutionSet := by sorry

end NUMINAMATH_CALUDE_cube_sum_digits_eq_square_self_l2621_262192


namespace NUMINAMATH_CALUDE_range_of_a_l2621_262105

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2621_262105


namespace NUMINAMATH_CALUDE_factorization_a_squared_plus_2a_l2621_262197

theorem factorization_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a*(a+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_plus_2a_l2621_262197


namespace NUMINAMATH_CALUDE_sqrt_three_solution_l2621_262166

theorem sqrt_three_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 3*a) : a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_solution_l2621_262166


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2621_262178

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 1000) → (∀ m : ℕ, m * (m + 1) < 1000 → n + (n + 1) ≥ m + (m + 1)) → 
  n + (n + 1) = 63 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2621_262178


namespace NUMINAMATH_CALUDE_parabola_shift_l2621_262111

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h + p.b, c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The main theorem stating that shifting y = 3x^2 right by 1 and down by 2 results in y = 3(x-1)^2 - 2 -/
theorem parabola_shift :
  let p := Parabola.mk 3 0 0
  let p_shifted := shift_vertical (shift_horizontal p 1) (-2)
  p_shifted = Parabola.mk 3 (-6) 1 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2621_262111


namespace NUMINAMATH_CALUDE_mark_fish_problem_l2621_262159

/-- Given the number of tanks, pregnant fish per tank, and young per fish, 
    calculate the total number of young fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Theorem stating that with 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is 240. -/
theorem mark_fish_problem : 
  total_young_fish 3 4 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mark_fish_problem_l2621_262159


namespace NUMINAMATH_CALUDE_money_saved_calculation_marcus_shopping_savings_l2621_262116

/-- Calculates the money saved when buying discounted items with sales tax --/
theorem money_saved_calculation (max_budget : ℝ) 
  (shoe_price shoe_discount : ℝ) 
  (sock_price sock_discount : ℝ) 
  (shirt_price shirt_discount : ℝ) 
  (sales_tax : ℝ) : ℝ :=
  let discounted_shoe := shoe_price * (1 - shoe_discount)
  let discounted_sock := sock_price * (1 - sock_discount)
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let total_before_tax := discounted_shoe + discounted_sock + discounted_shirt
  let final_cost := total_before_tax * (1 + sales_tax)
  let money_saved := max_budget - final_cost
  money_saved

/-- Proves that the money saved is approximately $34.22 --/
theorem marcus_shopping_savings : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |money_saved_calculation 200 120 0.3 25 0.2 55 0.1 0.08 - 34.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_money_saved_calculation_marcus_shopping_savings_l2621_262116


namespace NUMINAMATH_CALUDE_factorization_equality_l2621_262142

theorem factorization_equality (a b : ℝ) :
  276 * a^2 * b^2 + 69 * a * b - 138 * a * b^3 = 69 * a * b * (4 * a * b + 1 - 2 * b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2621_262142


namespace NUMINAMATH_CALUDE_factor_expression_l2621_262149

theorem factor_expression (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) / ((a + b)^3 + (b + c)^3 + (c + a)^3)
  = (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) / ((a + b) * (b + c) * (c + a)) :=
by sorry

end NUMINAMATH_CALUDE_factor_expression_l2621_262149


namespace NUMINAMATH_CALUDE_margo_travel_distance_l2621_262135

/-- The total distance Margo traveled given her jogging and walking times and average speed -/
theorem margo_travel_distance (jog_time walk_time avg_speed : ℝ) : 
  jog_time = 12 / 60 →
  walk_time = 25 / 60 →
  avg_speed = 5 →
  avg_speed * (jog_time + walk_time) = 3.085 :=
by sorry

end NUMINAMATH_CALUDE_margo_travel_distance_l2621_262135


namespace NUMINAMATH_CALUDE_time_to_cover_distance_l2621_262154

/-- Given a constant rate of movement and a remaining distance, prove that the time to cover the remaining distance can be calculated by dividing the remaining distance by the rate. -/
theorem time_to_cover_distance (rate : ℝ) (distance : ℝ) (time : ℝ) : 
  rate > 0 → distance > 0 → time = distance / rate → time * rate = distance := by sorry

end NUMINAMATH_CALUDE_time_to_cover_distance_l2621_262154


namespace NUMINAMATH_CALUDE_unique_right_triangle_completion_l2621_262188

/-- A function that checks if three side lengths form a right triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that there is exactly one integer side length 
    that can complete a right triangle with sides 8 and 15 -/
theorem unique_right_triangle_completion :
  ∃! x : ℕ, is_right_triangle 8 15 x :=
sorry

end NUMINAMATH_CALUDE_unique_right_triangle_completion_l2621_262188


namespace NUMINAMATH_CALUDE_athletes_arrangement_count_l2621_262114

/-- Represents the number of athletes in each team --/
def team_sizes : List Nat := [3, 3, 2, 4]

/-- The total number of athletes --/
def total_athletes : Nat := team_sizes.sum

/-- Calculates the number of ways to arrange the athletes --/
def arrangement_count : Nat :=
  (Nat.factorial team_sizes.length) * (team_sizes.map Nat.factorial).prod

theorem athletes_arrangement_count :
  total_athletes = 12 →
  team_sizes = [3, 3, 2, 4] →
  arrangement_count = 41472 := by
  sorry

end NUMINAMATH_CALUDE_athletes_arrangement_count_l2621_262114


namespace NUMINAMATH_CALUDE_rectangular_box_area_product_l2621_262189

/-- Given a rectangular box with dimensions length, width, and height,
    prove that the product of the areas of its base, side, and front
    is equal to the square of its volume. -/
theorem rectangular_box_area_product (length width height : ℝ) :
  (length * width) * (width * height) * (height * length) = (length * width * height) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_area_product_l2621_262189


namespace NUMINAMATH_CALUDE_green_then_blue_probability_l2621_262133

/-- The probability of drawing a green marble first and a blue marble second from a bag -/
theorem green_then_blue_probability 
  (total_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (green_marbles : ℕ) 
  (h1 : total_marbles = blue_marbles + green_marbles)
  (h2 : blue_marbles = 4)
  (h3 : green_marbles = 6) :
  (green_marbles : ℚ) / total_marbles * (blue_marbles : ℚ) / (total_marbles - 1) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_then_blue_probability_l2621_262133


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2621_262113

/-- Given that quantities a and b vary inversely, this function represents their relationship -/
def inverse_variation (k : ℝ) (a b : ℝ) : Prop := a * b = k

theorem inverse_variation_problem (k : ℝ) :
  inverse_variation k 800 0.5 →
  inverse_variation k 1600 0.25 :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2621_262113


namespace NUMINAMATH_CALUDE_third_side_is_fifteen_l2621_262179

/-- A triangle with two known sides and perimeter -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  perimeter : ℝ

/-- Calculate the third side of a triangle given two sides and the perimeter -/
def thirdSide (t : Triangle) : ℝ :=
  t.perimeter - t.side1 - t.side2

/-- Theorem: The third side of the specific triangle is 15 -/
theorem third_side_is_fifteen : 
  let t : Triangle := { side1 := 7, side2 := 10, perimeter := 32 }
  thirdSide t = 15 := by
  sorry

end NUMINAMATH_CALUDE_third_side_is_fifteen_l2621_262179


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2621_262170

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → ∃ r : ℝ, a (n + 1) = a n * r

-- Define the problem statement
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n > 0 → a n > 0) →
  a 1 * a 99 = 16 →
  a 1 + a 99 = 10 →
  a 40 * a 50 * a 60 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2621_262170


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2621_262176

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_middle_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 2)
  (h_a8 : a 8 = 32) :
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2621_262176


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2621_262119

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 5 * x - 3
  ∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2621_262119


namespace NUMINAMATH_CALUDE_people_in_room_l2621_262173

theorem people_in_room (empty_chairs : ℕ) 
  (h1 : empty_chairs = 5)
  (h2 : ∃ (total_chairs : ℕ), empty_chairs = total_chairs / 5)
  (h3 : ∃ (seated_people : ℕ) (total_people : ℕ), 
    seated_people = 4 * total_chairs / 5 ∧
    seated_people = 5 * total_people / 8) :
  ∃ (total_people : ℕ), total_people = 32 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l2621_262173


namespace NUMINAMATH_CALUDE_concert_ticket_cost_is_181_l2621_262174

/-- Calculates the cost of a concert ticket given hourly wage, weekly hours, percentage of monthly salary for outing, drink ticket cost, and number of drink tickets. -/
def concert_ticket_cost (hourly_wage : ℚ) (weekly_hours : ℚ) (outing_percentage : ℚ) (drink_ticket_cost : ℚ) (num_drink_tickets : ℕ) : ℚ :=
  let monthly_salary := hourly_wage * weekly_hours * 4
  let outing_budget := monthly_salary * outing_percentage
  let drink_tickets_cost := drink_ticket_cost * num_drink_tickets
  outing_budget - drink_tickets_cost

/-- Theorem stating that the cost of the concert ticket is $181 given the specified conditions. -/
theorem concert_ticket_cost_is_181 :
  concert_ticket_cost 18 30 (1/10) 7 5 = 181 := by
  sorry

#eval concert_ticket_cost 18 30 (1/10) 7 5

end NUMINAMATH_CALUDE_concert_ticket_cost_is_181_l2621_262174


namespace NUMINAMATH_CALUDE_rental_van_cost_increase_l2621_262128

theorem rental_van_cost_increase 
  (total_cost : ℝ) 
  (initial_people : ℕ) 
  (withdrawing_people : ℕ) 
  (h1 : total_cost = 450) 
  (h2 : initial_people = 15) 
  (h3 : withdrawing_people = 3) : 
  let remaining_people := initial_people - withdrawing_people
  let initial_share := total_cost / initial_people
  let new_share := total_cost / remaining_people
  new_share - initial_share = 7.5 := by
sorry

end NUMINAMATH_CALUDE_rental_van_cost_increase_l2621_262128


namespace NUMINAMATH_CALUDE_bracelets_count_l2621_262182

def total_stones : ℕ := 140
def stones_per_bracelet : ℕ := 14

theorem bracelets_count : total_stones / stones_per_bracelet = 10 := by
  sorry

end NUMINAMATH_CALUDE_bracelets_count_l2621_262182


namespace NUMINAMATH_CALUDE_calculator_squaring_min_presses_1000_eq_3_l2621_262146

def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => (repeated_square x m) ^ 2

theorem calculator_squaring (target : ℕ) : 
  ∃ (n : ℕ), repeated_square 3 n > target ∧ 
  ∀ (m : ℕ), m < n → repeated_square 3 m ≤ target := by
  sorry

def min_presses (target : ℕ) : ℕ :=
  Nat.find (calculator_squaring target)

theorem min_presses_1000_eq_3 : min_presses 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculator_squaring_min_presses_1000_eq_3_l2621_262146


namespace NUMINAMATH_CALUDE_count_polygons_l2621_262126

/-- The number of points placed on the circle -/
def n : ℕ := 15

/-- The number of distinct convex polygons with at least three sides -/
def num_polygons : ℕ := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)

/-- Theorem stating that the number of distinct convex polygons is 32647 -/
theorem count_polygons : num_polygons = 32647 := by
  sorry

end NUMINAMATH_CALUDE_count_polygons_l2621_262126


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2621_262107

theorem partial_fraction_decomposition (x P Q R : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2 ↔ P = 5 ∧ Q = -5 ∧ R = -5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2621_262107


namespace NUMINAMATH_CALUDE_two_translations_result_l2621_262139

def complex_translation (z w : ℂ) : ℂ → ℂ := fun x ↦ x + w - z

theorem two_translations_result (t₁ t₂ : ℂ → ℂ) :
  t₁ (-3 + 2*I) = -7 - I →
  t₂ (-7 - I) = -10 →
  t₁ = complex_translation (-3 + 2*I) (-7 - I) →
  t₂ = complex_translation (-7 - I) (-10) →
  (t₂ ∘ t₁) (-4 + 5*I) = -11 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_two_translations_result_l2621_262139


namespace NUMINAMATH_CALUDE_triangle_shape_l2621_262137

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if (a² + b²)sin(A - B) = (a² - b²)sin(A + B),
    then the triangle is either isosceles (A = B) or right-angled (2A + 2B = 180°). -/
theorem triangle_shape (a b c A B C : ℝ) (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B)) :
  A = B ∨ 2*A + 2*B = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l2621_262137


namespace NUMINAMATH_CALUDE_triangle_angle_c_value_l2621_262115

theorem triangle_angle_c_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a^2 = 3*b^2 + 3*c^2 - 2*Real.sqrt 3*b*c*Real.sin A →
  C = π/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_value_l2621_262115


namespace NUMINAMATH_CALUDE_b_95_mod_121_l2621_262131

/-- Calculate b₉₅ modulo 121 where bₙ = 5ⁿ + 11ⁿ -/
theorem b_95_mod_121 : (5^95 + 11^95) % 121 = 16 := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_121_l2621_262131


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l2621_262143

/-- The distance from the origin to the line x + √3y - 2 = 0 is 1 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y - 2 = 0}
  ∃ d : ℝ, d = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l2621_262143


namespace NUMINAMATH_CALUDE_det_B_is_one_l2621_262110

theorem det_B_is_one (b e : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![b, 2; -3, e]
  B + B⁻¹ = 1 → Matrix.det B = 1 := by
sorry

end NUMINAMATH_CALUDE_det_B_is_one_l2621_262110


namespace NUMINAMATH_CALUDE_F_is_even_T_is_even_l2621_262171

variable (f : ℝ → ℝ)

def F (x : ℝ) : ℝ := f x * f (-x)

def T (x : ℝ) : ℝ := f x + f (-x)

theorem F_is_even : ∀ x : ℝ, F f x = F f (-x) := by sorry

theorem T_is_even : ∀ x : ℝ, T f x = T f (-x) := by sorry

end NUMINAMATH_CALUDE_F_is_even_T_is_even_l2621_262171


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l2621_262117

theorem sqrt_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l2621_262117


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2621_262136

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ  -- The sequence
  d : ℝ        -- Common difference
  S : ℕ+ → ℝ  -- Sum function
  sum_def : ∀ n : ℕ+, S n = n * a 1 + n * (n - 1) / 2 * d
  seq_def : ∀ n : ℕ+, a n = a 1 + (n - 1) * d

/-- Theorem about properties of an arithmetic sequence given certain conditions -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h : seq.S 6 > seq.S 7 ∧ seq.S 7 > seq.S 5) :
    seq.d < 0 ∧ 
    seq.S 11 > 0 ∧ 
    seq.S 12 > 0 ∧ 
    seq.S 13 < 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2621_262136


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2621_262163

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ x ∈ Set.Icc (-2) 1 ∧ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2621_262163


namespace NUMINAMATH_CALUDE_banana_cost_l2621_262100

/-- Given that 5 dozen bananas cost $24.00, prove that 4 dozen bananas at the same rate will cost $19.20 -/
theorem banana_cost (total_cost : ℝ) (total_dozens : ℕ) (target_dozens : ℕ) 
  (h1 : total_cost = 24)
  (h2 : total_dozens = 5)
  (h3 : target_dozens = 4) :
  (target_dozens : ℝ) * (total_cost / total_dozens) = 19.2 :=
by sorry

end NUMINAMATH_CALUDE_banana_cost_l2621_262100


namespace NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l2621_262183

/-- Two lines in the plane -/
structure Lines (m : ℝ) where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  eq1 : ∀ x y, l1 x y ↔ x + m * y + 6 = 0
  eq2 : ∀ x y, l2 x y ↔ (m - 2) * x + 3 * y + 2 * m = 0

/-- The lines are perpendicular -/
def Perpendicular (m : ℝ) (lines : Lines m) : Prop :=
  (-1 / m) * ((m - 2) / 3) = -1

/-- The lines are parallel -/
def Parallel (m : ℝ) (lines : Lines m) : Prop :=
  -1 / m = (m - 2) / 3

theorem perpendicular_implies_m_eq_half (m : ℝ) (lines : Lines m) :
  Perpendicular m lines → m = 1 / 2 := by
  sorry

theorem parallel_implies_m_eq_neg_one (m : ℝ) (lines : Lines m) :
  Parallel m lines → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l2621_262183


namespace NUMINAMATH_CALUDE_jelly_mold_radius_l2621_262193

theorem jelly_mold_radius :
  let original_radius : ℝ := 1.5
  let num_molds : ℕ := 64
  let hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
  let original_volume := hemisphere_volume original_radius
  let small_mold_radius := (3 : ℝ) / 8
  original_volume = num_molds * hemisphere_volume small_mold_radius :=
by sorry

end NUMINAMATH_CALUDE_jelly_mold_radius_l2621_262193


namespace NUMINAMATH_CALUDE_ellipse_equation_l2621_262162

/-- Given an ellipse with center at the origin, foci on the x-axis, 
    major axis length of 4, and minor axis length of 2, 
    its equation is x²/4 + y² = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let center := (0 : ℝ × ℝ)
  let major_axis := 4
  let minor_axis := 2
  let foci_on_x_axis := true
  x^2 / 4 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2621_262162


namespace NUMINAMATH_CALUDE_square_side_length_l2621_262181

theorem square_side_length (perimeter area : ℝ) (h_perimeter : perimeter = 48) (h_area : area = 144) :
  ∃ (side : ℝ), side * 4 = perimeter ∧ side * side = area ∧ side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2621_262181


namespace NUMINAMATH_CALUDE_max_area_triangle_abc_l2621_262153

theorem max_area_triangle_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  let angle_C : ℝ := π / 3
  let area := (1 / 2) * a * b * Real.sin angle_C
  3 * a * b = 25 - c^2 →
  ∀ (a' b' c' : ℝ), 
    a' > 0 → b' > 0 → c' > 0 →
    3 * a' * b' = 25 - c'^2 →
    area ≤ ((25 : ℝ) / 16) * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_abc_l2621_262153


namespace NUMINAMATH_CALUDE_perpendicular_tangent_line_l2621_262194

/-- Given a line L1 with equation x + 3y - 10 = 0 and a circle C with equation x^2 + y^2 = 4,
    prove that a line L2 perpendicular to L1 and tangent to C has the equation 3x - y ± 2√10 = 0 -/
theorem perpendicular_tangent_line 
  (L1 : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop)
  (h1 : ∀ x y, L1 x y ↔ x + 3*y - 10 = 0)
  (h2 : ∀ x y, C x y ↔ x^2 + y^2 = 4) :
  ∃ L2 : ℝ → ℝ → Prop,
    (∀ x y, L2 x y ↔ (3*x - y = 2*Real.sqrt 10 ∨ 3*x - y = -2*Real.sqrt 10)) ∧
    (∀ x y, L1 x y → ∀ u v, L2 u v → (x - u) * (3 * (y - v)) = -(y - v) * (x - u)) ∧
    (∃ p q, L2 p q ∧ C p q ∧ ∀ x y, C x y → (x - p)^2 + (y - q)^2 ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangent_line_l2621_262194


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l2621_262132

-- Define the fuel tank capacity
def C : ℝ := 200

-- Define the volume of fuel A added
def fuel_A_volume : ℝ := 349.99999999999994

-- Define the ethanol percentage in fuel A
def ethanol_A_percent : ℝ := 0.12

-- Define the ethanol percentage in fuel B
def ethanol_B_percent : ℝ := 0.16

-- Define the total ethanol volume in the full tank
def total_ethanol_volume : ℝ := 18

-- Theorem statement
theorem fuel_tank_capacity :
  C = 200 ∧
  fuel_A_volume = 349.99999999999994 ∧
  ethanol_A_percent = 0.12 ∧
  ethanol_B_percent = 0.16 ∧
  total_ethanol_volume = 18 →
  ethanol_A_percent * fuel_A_volume + ethanol_B_percent * (C - fuel_A_volume) = total_ethanol_volume :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l2621_262132


namespace NUMINAMATH_CALUDE_circle_angle_problem_l2621_262187

theorem circle_angle_problem (x y : ℝ) : 
  3 * x + 2 * y + 5 * x + 7 * x = 360 →
  x = y →
  x = 360 / 17 ∧ y = 360 / 17 := by
sorry

end NUMINAMATH_CALUDE_circle_angle_problem_l2621_262187


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l2621_262108

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l2621_262108


namespace NUMINAMATH_CALUDE_floor_sum_equals_n_l2621_262120

theorem floor_sum_equals_n (N : ℕ+) :
  N = ∑' n : ℕ, ⌊(N : ℝ) / (2 ^ n : ℝ)⌋ := by sorry

end NUMINAMATH_CALUDE_floor_sum_equals_n_l2621_262120


namespace NUMINAMATH_CALUDE_inequality_existence_l2621_262138

variable (a : ℝ)

theorem inequality_existence (h1 : a > 1) (h2 : a ≠ 2) :
  (¬ ∀ x : ℝ, (1 < x ∧ x < a) → (a < 2*x ∧ 2*x < a^2)) ∧
  (∃ x : ℝ, (a < 2*x ∧ 2*x < a^2) ∧ ¬(1 < x ∧ x < a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_existence_l2621_262138


namespace NUMINAMATH_CALUDE_debbys_friend_photos_l2621_262151

theorem debbys_friend_photos (total_photos family_photos : ℕ) 
  (h1 : total_photos = 86) 
  (h2 : family_photos = 23) : 
  total_photos - family_photos = 63 := by
  sorry

end NUMINAMATH_CALUDE_debbys_friend_photos_l2621_262151


namespace NUMINAMATH_CALUDE_chloe_profit_l2621_262150

/-- Calculates Chloe's profit from selling chocolate-dipped strawberries during a 3-day Mother's Day celebration. -/
theorem chloe_profit (buy_price : ℝ) (sell_price : ℝ) (bulk_discount : ℝ) 
  (min_production_cost : ℝ) (max_production_cost : ℝ) 
  (day1_price_factor : ℝ) (day2_price_factor : ℝ) (day3_price_factor : ℝ)
  (total_dozens : ℕ) (day1_dozens : ℕ) (day2_dozens : ℕ) (day3_dozens : ℕ) :
  buy_price = 50 →
  sell_price = 60 →
  bulk_discount = 0.1 →
  min_production_cost = 40 →
  max_production_cost = 45 →
  day1_price_factor = 1 →
  day2_price_factor = 1.2 →
  day3_price_factor = 0.85 →
  total_dozens = 50 →
  day1_dozens = 12 →
  day2_dozens = 18 →
  day3_dozens = 20 →
  total_dozens ≥ 10 →
  day1_dozens + day2_dozens + day3_dozens = total_dozens →
  ∃ profit : ℝ, profit = 152 ∧ 
    profit = (day1_dozens * sell_price * day1_price_factor +
              day2_dozens * sell_price * day2_price_factor +
              day3_dozens * sell_price * day3_price_factor) * (1 - bulk_discount) -
             total_dozens * (min_production_cost + max_production_cost) / 2 :=
by sorry

end NUMINAMATH_CALUDE_chloe_profit_l2621_262150


namespace NUMINAMATH_CALUDE_trapezoid_minimum_distance_l2621_262161

-- Define the trapezoid ABCD
def Trapezoid (A B C D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧
  B.1 = 0 ∧ B.2 = 12 ∧
  C.1 = 10 ∧ C.2 = 12 ∧
  D.1 = 10 ∧ D.2 = 6

-- Define the circle centered at C with radius 8
def Circle (C F : ℝ × ℝ) : Prop :=
  (F.1 - C.1)^2 + (F.2 - C.2)^2 = 64

-- Define point E on AB
def PointOnAB (A B E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Define the theorem
theorem trapezoid_minimum_distance (A B C D E F : ℝ × ℝ) :
  Trapezoid A B C D →
  Circle C F →
  PointOnAB A B E →
  (∀ E' F', PointOnAB A B E' → Circle C F' →
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) + Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) ≤
    Real.sqrt ((D.1 - E'.1)^2 + (D.2 - E'.2)^2) + Real.sqrt ((E'.1 - F'.1)^2 + (E'.2 - F'.2)^2)) →
  E.2 - A.2 = 4.5 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_minimum_distance_l2621_262161


namespace NUMINAMATH_CALUDE_inequality_proof_l2621_262134

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2621_262134


namespace NUMINAMATH_CALUDE_smallest_among_three_l2621_262103

theorem smallest_among_three : ∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 4 → c ≤ a ∧ c ≤ b := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_three_l2621_262103


namespace NUMINAMATH_CALUDE_largest_inscribed_sphere_surface_area_l2621_262145

/-- The surface area of the largest sphere inscribed in a cone -/
theorem largest_inscribed_sphere_surface_area
  (base_radius : ℝ)
  (slant_height : ℝ)
  (h_base_radius : base_radius = 1)
  (h_slant_height : slant_height = 3) :
  ∃ (sphere_surface_area : ℝ),
    sphere_surface_area = 2 * Real.pi ∧
    ∀ (other_sphere_surface_area : ℝ),
      other_sphere_surface_area ≤ sphere_surface_area :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_sphere_surface_area_l2621_262145


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2621_262141

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  is_convex : Bool
  right_angles : ℕ

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex 9-sided polygon with three right angles has 27 diagonals -/
theorem nonagon_diagonals (P : ConvexPolygon 9) (h1 : P.is_convex = true) (h2 : P.right_angles = 3) :
  num_diagonals P.sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2621_262141


namespace NUMINAMATH_CALUDE_expression_simplification_l2621_262129

theorem expression_simplification (m : ℝ) (h : m = 10) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2621_262129


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2621_262155

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 2) :
  1 - (a - 2) / a / ((a^2 - 4) / (a^2 + a)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2621_262155


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2621_262122

-- Problem 1
theorem problem_1 : (-2023)^0 + Real.sqrt 12 + 2 * (-1/2) = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) : (2*m + 1) * (2*m - 1) - 4*m*(m - 1) = 4*m - 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2621_262122


namespace NUMINAMATH_CALUDE_product_less_than_60_probability_l2621_262125

def paco_range : Finset ℕ := Finset.range 5
def manu_range : Finset ℕ := Finset.range 20

def total_outcomes : ℕ := paco_range.card * manu_range.card

def favorable_outcomes : ℕ :=
  (paco_range.filter (fun p => p + 1 ≤ 2)).sum (fun p =>
    (manu_range.filter (fun m => (p + 1) * (m + 1) < 60)).card)
  +
  (paco_range.filter (fun p => p + 1 > 2)).sum (fun p =>
    (manu_range.filter (fun m => (p + 1) * (m + 1) < 60)).card)

theorem product_less_than_60_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 21 / 25 := by sorry

end NUMINAMATH_CALUDE_product_less_than_60_probability_l2621_262125


namespace NUMINAMATH_CALUDE_tan_equation_solution_l2621_262156

theorem tan_equation_solution (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 6)
  (h3 : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l2621_262156


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2621_262195

theorem no_integer_solutions (c : ℕ) (hc_pos : c > 0) (hc_odd : Odd c) :
  ¬∃ (x y : ℤ), x^2 - y^3 = (2*c)^3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2621_262195


namespace NUMINAMATH_CALUDE_bug_path_theorem_l2621_262158

/-- Represents a rectangular garden paved with square pavers -/
structure PavedGarden where
  width : ℕ  -- width in feet
  length : ℕ  -- length in feet
  paver_size : ℕ  -- size of square paver in feet

/-- Calculates the number of pavers a bug visits when walking diagonally across the garden -/
def pavers_visited (garden : PavedGarden) : ℕ :=
  let width_pavers := garden.width / garden.paver_size
  let length_pavers := (garden.length + garden.paver_size - 1) / garden.paver_size
  width_pavers + length_pavers - Nat.gcd width_pavers length_pavers

/-- Theorem stating that a bug walking diagonally across a 14x19 garden with 2-foot pavers visits 16 pavers -/
theorem bug_path_theorem :
  let garden : PavedGarden := { width := 14, length := 19, paver_size := 2 }
  pavers_visited garden = 16 := by sorry

end NUMINAMATH_CALUDE_bug_path_theorem_l2621_262158


namespace NUMINAMATH_CALUDE_circle_radius_and_diameter_l2621_262175

theorem circle_radius_and_diameter 
  (M N : ℝ) 
  (h_area : M = π * r^2) 
  (h_circumference : N = 2 * π * r) 
  (h_ratio : M / N = 15) : 
  r = 30 ∧ 2 * r = 60 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_and_diameter_l2621_262175


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2621_262147

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 + 4 * x + 1 < 0) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2621_262147


namespace NUMINAMATH_CALUDE_sphere_volume_l2621_262124

theorem sphere_volume (R : ℝ) (x y : ℝ) : 
  R > 0 ∧ 
  x ≠ y ∧
  R^2 = x^2 + 5 ∧ 
  R^2 = y^2 + 8 ∧ 
  |x - y| = 1 →
  (4/3) * Real.pi * R^3 = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_l2621_262124


namespace NUMINAMATH_CALUDE_sum_of_squares_divided_by_365_l2621_262157

theorem sum_of_squares_divided_by_365 : (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divided_by_365_l2621_262157


namespace NUMINAMATH_CALUDE_round_31083_58_to_two_sig_figs_l2621_262130

/-- Rounds a number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Theorem: Rounding 31,083.58 to two significant figures results in 3.1 × 10^4 -/
theorem round_31083_58_to_two_sig_figs :
  roundToSignificantFigures 31083.58 2 = 3.1 * 10^4 := by sorry

end NUMINAMATH_CALUDE_round_31083_58_to_two_sig_figs_l2621_262130


namespace NUMINAMATH_CALUDE_solution_to_system_l2621_262118

theorem solution_to_system (x y z : ℝ) 
  (eq1 : x = 1 + Real.sqrt (y - z^2))
  (eq2 : y = 1 + Real.sqrt (z - x^2))
  (eq3 : z = 1 + Real.sqrt (x - y^2)) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_l2621_262118


namespace NUMINAMATH_CALUDE_beth_wins_743_l2621_262123

/-- Represents a configuration of brick walls -/
def Configuration := List Nat

/-- Calculates the nim-value of a single wall -/
noncomputable def nimValue (n : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a configuration is a winning position for the current player -/
def isWinningPosition (config : Configuration) : Prop :=
  nimSum (config.map nimValue) ≠ 0

/-- The game of brick removal -/
theorem beth_wins_743 (config : Configuration) :
  config = [7, 4, 4] → ¬isWinningPosition config :=
  sorry

end NUMINAMATH_CALUDE_beth_wins_743_l2621_262123


namespace NUMINAMATH_CALUDE_building_painting_cost_l2621_262127

theorem building_painting_cost (room1_area room2_area room3_area : ℝ)
  (paint_price1 paint_price2 paint_price3 : ℝ)
  (labor_cost : ℝ) (tax_rate : ℝ) :
  room1_area = 196 →
  room2_area = 150 →
  room3_area = 250 →
  paint_price1 = 15 →
  paint_price2 = 18 →
  paint_price3 = 20 →
  labor_cost = 800 →
  tax_rate = 0.05 →
  let room1_cost := room1_area * paint_price1
  let room2_cost := room2_area * paint_price2
  let room3_cost := room3_area * paint_price3
  let total_painting_cost := room1_cost + room2_cost + room3_cost
  let total_cost_before_tax := total_painting_cost + labor_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + tax
  total_cost_after_tax = 12012 :=
by sorry

end NUMINAMATH_CALUDE_building_painting_cost_l2621_262127


namespace NUMINAMATH_CALUDE_audrey_peaches_l2621_262169

def paul_peaches : ℕ := 48
def peach_difference : ℤ := 22

theorem audrey_peaches :
  ∃ (audrey : ℕ), (audrey : ℤ) - paul_peaches = peach_difference ∧ audrey = 70 := by
  sorry

end NUMINAMATH_CALUDE_audrey_peaches_l2621_262169


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2621_262101

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^11 + i^111 = -2 * i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2621_262101


namespace NUMINAMATH_CALUDE_decorative_gravel_cost_l2621_262164

/-- The cost of decorative gravel in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of gravel -/
def cubic_yards : ℝ := 8

/-- The total cost of the decorative gravel -/
def total_cost : ℝ := cubic_yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem decorative_gravel_cost : total_cost = 1728 := by
  sorry

end NUMINAMATH_CALUDE_decorative_gravel_cost_l2621_262164
