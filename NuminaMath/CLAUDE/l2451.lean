import Mathlib

namespace NUMINAMATH_CALUDE_round_trip_distance_boy_school_distance_l2451_245123

/-- Calculates the distance between two points given the speeds and total time of a round trip -/
theorem round_trip_distance (outbound_speed return_speed : ℝ) (total_time : ℝ) : 
  outbound_speed > 0 → return_speed > 0 → total_time > 0 →
  (1 / outbound_speed + 1 / return_speed) * (outbound_speed * return_speed * total_time / (outbound_speed + return_speed)) = total_time := by
  sorry

/-- The distance between the boy's house and school -/
theorem boy_school_distance : 
  let outbound_speed : ℝ := 3
  let return_speed : ℝ := 2
  let total_time : ℝ := 5
  (outbound_speed * return_speed * total_time) / (outbound_speed + return_speed) = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_boy_school_distance_l2451_245123


namespace NUMINAMATH_CALUDE_correct_average_weight_l2451_245172

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 ∧ 
  initial_average = 58.4 ∧ 
  misread_weight = 56 ∧ 
  correct_weight = 68 →
  (n : ℝ) * initial_average + (correct_weight - misread_weight) = n * 59 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2451_245172


namespace NUMINAMATH_CALUDE_used_car_selection_l2451_245135

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 18 →
  num_clients = 18 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 3 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selection_l2451_245135


namespace NUMINAMATH_CALUDE_chessboard_coloring_count_l2451_245101

/-- The number of ways to paint an N × N chessboard with 4 colors such that:
    1) Squares with a common side are painted with distinct colors
    2) Every 2 × 2 square is painted with the four colors -/
def chessboardColorings (N : ℕ) : ℕ := 24 * (2^(N-1) - 1)

/-- Theorem stating the number of valid colorings for an N × N chessboard -/
theorem chessboard_coloring_count (N : ℕ) (h : N > 1) : 
  chessboardColorings N = 24 * (2^(N-1) - 1) := by
  sorry


end NUMINAMATH_CALUDE_chessboard_coloring_count_l2451_245101


namespace NUMINAMATH_CALUDE_multiplication_puzzle_solution_l2451_245132

theorem multiplication_puzzle_solution : 
  (78346 * 346 = 235038) ∧ (9374 * 82 = 768668) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_solution_l2451_245132


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2451_245144

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 130) (h2 : b = 95) (h3 : c = 115) (h4 : d = 110) (h5 : e = 87) : 
  720 - (a + b + c + d + e) = 183 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2451_245144


namespace NUMINAMATH_CALUDE_museum_time_per_student_l2451_245133

theorem museum_time_per_student 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (time_per_group : ℕ) 
  (h1 : total_students = 18)
  (h2 : num_groups = 3)
  (h3 : time_per_group = 24)
  (h4 : total_students % num_groups = 0) -- Ensures equal division
  : (time_per_group * num_groups) / total_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_museum_time_per_student_l2451_245133


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2451_245195

theorem circular_garden_radius (r : ℝ) : r > 0 → 2 * Real.pi * r = (1 / 5) * Real.pi * r^2 → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2451_245195


namespace NUMINAMATH_CALUDE_speech_competition_probability_l2451_245109

theorem speech_competition_probability (n : ℕ) (h : n = 5) : 
  let total_arrangements := n.factorial
  let favorable_arrangements := (n - 1).factorial
  let prob_A_before_B := (total_arrangements / 2 : ℚ) / total_arrangements
  let prob_adjacent_and_A_before_B := (favorable_arrangements : ℚ) / total_arrangements
  (prob_adjacent_and_A_before_B / prob_A_before_B) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_probability_l2451_245109


namespace NUMINAMATH_CALUDE_negative_deviation_notation_l2451_245118

/-- Represents a height deviation from the average. -/
structure HeightDeviation where
  value : ℝ

/-- The average height of the team. -/
def averageHeight : ℝ := 175

/-- Notation for height deviations. -/
def denoteDeviation (d : HeightDeviation) : ℝ := d.value

/-- Axiom: Positive deviation is denoted by a positive number. -/
axiom positive_deviation_notation (d : HeightDeviation) :
  d.value > 0 → denoteDeviation d > 0

/-- Theorem: Negative deviation should be denoted by a negative number. -/
theorem negative_deviation_notation (d : HeightDeviation) :
  d.value < 0 → denoteDeviation d < 0 :=
sorry

end NUMINAMATH_CALUDE_negative_deviation_notation_l2451_245118


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l2451_245198

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ n : ℕ, n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l2451_245198


namespace NUMINAMATH_CALUDE_jackie_has_more_fruits_than_adam_l2451_245183

/-- Represents the number of fruits a person has -/
structure FruitCount where
  apples : ℕ
  oranges : ℕ
  bananas : ℚ

/-- Calculates the difference in total apples and oranges between two FruitCounts -/
def applePlusOrangeDifference (a b : FruitCount) : ℤ :=
  (b.apples + b.oranges : ℤ) - (a.apples + a.oranges)

theorem jackie_has_more_fruits_than_adam :
  let adam : FruitCount := { apples := 25, oranges := 34, bananas := 18.5 }
  let jackie : FruitCount := { apples := 43, oranges := 29, bananas := 16.5 }
  applePlusOrangeDifference adam jackie = 13 := by
  sorry

end NUMINAMATH_CALUDE_jackie_has_more_fruits_than_adam_l2451_245183


namespace NUMINAMATH_CALUDE_evaluate_expression_l2451_245121

theorem evaluate_expression : 16^3 + 3*(16^2) + 3*16 + 1 = 4913 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2451_245121


namespace NUMINAMATH_CALUDE_marble_selection_probability_l2451_245169

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 2

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 2

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 2

/-- The number of yellow marbles in the bag -/
def yellow_marbles : ℕ := 1

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles + yellow_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 3

/-- The probability of selecting one red, one blue, and one green marble -/
def probability_red_blue_green : ℚ := 8 / 35

theorem marble_selection_probability :
  probability_red_blue_green = (red_marbles * blue_marbles * green_marbles : ℚ) / (total_marbles.choose selected_marbles) :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l2451_245169


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_108_l2451_245137

/-- Represents a triangle with sides a, b, c and incenter radius r -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  r : ℕ+
  isIsosceles : a = b
  incenterRadius : r = 8

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ := t.a.val + t.b.val + t.c.val

/-- Theorem: The smallest possible perimeter of a triangle satisfying the given conditions is 108 -/
theorem smallest_perimeter_is_108 :
  ∀ t : Triangle, perimeter t ≥ 108 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_108_l2451_245137


namespace NUMINAMATH_CALUDE_infinitely_many_wrappers_l2451_245174

/-- A wrapper for a 1 × 1 painting is a rectangle with area 2 that can cover the painting on both sides. -/
def IsWrapper (width height : ℝ) : Prop :=
  width > 0 ∧ height > 0 ∧ width * height = 2 ∧ width ≥ 1 ∧ height ≥ 1

/-- There exist infinitely many wrappers for a 1 × 1 painting. -/
theorem infinitely_many_wrappers :
  ∃ f : ℕ → ℝ × ℝ, ∀ n : ℕ, IsWrapper (f n).1 (f n).2 ∧
    ∀ m : ℕ, m ≠ n → f m ≠ f n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_wrappers_l2451_245174


namespace NUMINAMATH_CALUDE_point_move_left_l2451_245153

def number_line_move (initial_position : ℝ) (move_distance : ℝ) : ℝ :=
  initial_position - move_distance

theorem point_move_left :
  let initial_position : ℝ := -4
  let move_distance : ℝ := 2
  number_line_move initial_position move_distance = -6 := by
sorry

end NUMINAMATH_CALUDE_point_move_left_l2451_245153


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l2451_245128

def number : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor number) = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l2451_245128


namespace NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l2451_245152

theorem quadratic_equation_nonnegative_solutions :
  ∃! x : ℝ, x ≥ 0 ∧ x^2 = -6*x :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l2451_245152


namespace NUMINAMATH_CALUDE_sphere_surface_area_equal_volume_cone_l2451_245131

/-- Given a cone with radius 2 inches and height 6 inches, 
    prove that the surface area of a sphere with the same volume 
    is 4π(6^(2/3)) square inches. -/
theorem sphere_surface_area_equal_volume_cone (π : ℝ) : 
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 6
  let cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
  let sphere_radius : ℝ := (3 * cone_volume / (4 * π))^(1/3)
  let sphere_surface_area : ℝ := 4 * π * sphere_radius^2
  sphere_surface_area = 4 * π * 6^(2/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_equal_volume_cone_l2451_245131


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_one_l2451_245106

/-- A structure representing a set of sample points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  n_ge_2 : n ≥ 2
  not_all_x_equal : ∃ i j, i ≠ j ∧ x i ≠ x j

/-- The sample correlation coefficient -/
def sampleCorrelationCoefficient (data : SampleData) : ℝ := sorry

/-- All points lie on the line y = 2x + 1 -/
def allPointsOnLine (data : SampleData) : Prop :=
  ∀ i, data.y i = 2 * data.x i + 1

/-- Theorem stating that if all points lie on y = 2x + 1, then the correlation coefficient is 1 -/
theorem correlation_coefficient_is_one (data : SampleData) 
  (h : allPointsOnLine data) : sampleCorrelationCoefficient data = 1 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_one_l2451_245106


namespace NUMINAMATH_CALUDE_problem_solution_l2451_245104

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.cos y = 1010)
  (h2 : x + 1010 * Real.sin y = 1009)
  (h3 : π / 4 ≤ y ∧ y ≤ π / 2) :
  x + y = 1010 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2451_245104


namespace NUMINAMATH_CALUDE_sales_tax_difference_specific_sales_tax_difference_l2451_245125

/-- The difference between state and local sales taxes on a discounted sweater --/
theorem sales_tax_difference (original_price : ℝ) (discount_rate : ℝ) 
  (state_tax_rate : ℝ) (local_tax_rate : ℝ) : ℝ :=
by
  -- Define the discounted price
  let discounted_price := original_price * (1 - discount_rate)
  
  -- Calculate state and local taxes
  let state_tax := discounted_price * state_tax_rate
  let local_tax := discounted_price * local_tax_rate
  
  -- Calculate the difference
  exact state_tax - local_tax

/-- The specific case for the given problem --/
theorem specific_sales_tax_difference : 
  sales_tax_difference 50 0.1 0.075 0.07 = 0.225 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_specific_sales_tax_difference_l2451_245125


namespace NUMINAMATH_CALUDE_cos_sin_gt_sin_cos_l2451_245108

theorem cos_sin_gt_sin_cos :
  ∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → Real.cos (Real.sin x) > Real.sin (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_gt_sin_cos_l2451_245108


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2451_245171

-- Define set A
def A : Set ℝ := {x | |x| > 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2451_245171


namespace NUMINAMATH_CALUDE_quadratic_sum_l2451_245194

/-- A quadratic function passing through (1,0) and (-5,0) with minimum value 25 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≥ 25) ∧
  QuadraticFunction a b c 1 = 0 ∧
  QuadraticFunction a b c (-5) = 0 →
  a + b + c = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2451_245194


namespace NUMINAMATH_CALUDE_otimes_nested_l2451_245173

/-- Definition of the ⊗ operation -/
def otimes (g y : ℝ) : ℝ := g^2 + 2*y

/-- Theorem stating the result of g ⊗ (g ⊗ g) -/
theorem otimes_nested (g : ℝ) : otimes g (otimes g g) = g^4 + 4*g^3 + 6*g^2 + 4*g := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_l2451_245173


namespace NUMINAMATH_CALUDE_circle_area_8m_diameter_circle_area_8m_diameter_proof_l2451_245117

/-- The area of a circle with diameter 8 meters, in square centimeters -/
theorem circle_area_8m_diameter (π : ℝ) : ℝ :=
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area_sq_meters : ℝ := π * radius ^ 2
  let sq_cm_per_sq_meter : ℝ := 10000
  160000 * π

/-- Proof that the area of a circle with diameter 8 meters is 160000π square centimeters -/
theorem circle_area_8m_diameter_proof (π : ℝ) :
  circle_area_8m_diameter π = 160000 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_8m_diameter_circle_area_8m_diameter_proof_l2451_245117


namespace NUMINAMATH_CALUDE_square_root_expression_l2451_245122

theorem square_root_expression (m n : ℝ) : 
  Real.sqrt ((m - 2*n - 3) * (m - 2*n + 3) + 9) = 
    if m ≥ 2*n then m - 2*n else 2*n - m := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_l2451_245122


namespace NUMINAMATH_CALUDE_complex_power_sum_l2451_245129

theorem complex_power_sum (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2451_245129


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2451_245176

/-- Given a geometric sequence {a_n} with all positive terms, where 3a_1, (1/2)a_3, 2a_2 form an arithmetic sequence, (a_11 + a_13) / (a_8 + a_10) = 27. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  (3 * a 1 - (1/2) * a 3 = (1/2) * a 3 - 2 * a 2) →  -- arithmetic sequence condition
  (a 11 + a 13) / (a 8 + a 10) = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2451_245176


namespace NUMINAMATH_CALUDE_initial_cars_count_l2451_245199

/-- The initial number of cars on the lot -/
def initial_cars : ℕ := sorry

/-- The percentage of initial cars that are silver -/
def initial_silver_percent : ℚ := 1/5

/-- The number of cars in the new shipment -/
def new_shipment : ℕ := 80

/-- The percentage of new cars that are silver -/
def new_silver_percent : ℚ := 1/2

/-- The percentage of total cars that are silver after the new shipment -/
def total_silver_percent : ℚ := 2/5

theorem initial_cars_count : initial_cars = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_cars_count_l2451_245199


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2451_245191

/-- An isosceles triangle with a median dividing its perimeter -/
structure IsoscelesTriangleWithMedian where
  /-- Length of each leg of the isosceles triangle -/
  leg : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : leg > 0
  /-- The median on one leg divides the perimeter into parts of 6cm and 12cm -/
  medianDivision : leg + leg / 2 = 12 ∧ leg / 2 + base = 6
  /-- Triangle inequality -/
  triangleInequality : 2 * leg > base ∧ base > 0

/-- Theorem: The base of the isosceles triangle is 2cm -/
theorem isosceles_triangle_base_length
  (t : IsoscelesTriangleWithMedian) : t.base = 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2451_245191


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2451_245189

theorem complex_power_magnitude (z w : ℂ) (n : ℕ) :
  z = w ^ n → Complex.abs z ^ 2 = Complex.abs w ^ (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2451_245189


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_150_500_l2451_245178

theorem lcm_gcf_ratio_150_500 : Nat.lcm 150 500 / Nat.gcd 150 500 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_150_500_l2451_245178


namespace NUMINAMATH_CALUDE_two_digit_integer_less_than_multiple_l2451_245154

theorem two_digit_integer_less_than_multiple (n : ℕ) : n = 83 ↔ 
  (10 ≤ n ∧ n < 100) ∧ 
  (∃ k : ℕ, n + 1 = 3 * k) ∧ 
  (∃ k : ℕ, n + 1 = 4 * k) ∧ 
  (∃ k : ℕ, n + 1 = 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_less_than_multiple_l2451_245154


namespace NUMINAMATH_CALUDE_eight_mile_taxi_cost_l2451_245145

/-- Calculates the cost of a taxi ride given the base fare, per-mile charge, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (per_mile_charge : ℝ) (distance : ℝ) : ℝ :=
  base_fare + per_mile_charge * distance

/-- Proves that the cost of an 8-mile taxi ride with a base fare of $2.00 and a per-mile charge of $0.30 is equal to $4.40. -/
theorem eight_mile_taxi_cost :
  taxi_cost 2.00 0.30 8 = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_eight_mile_taxi_cost_l2451_245145


namespace NUMINAMATH_CALUDE_sum_of_roots_when_product_is_24_l2451_245196

theorem sum_of_roots_when_product_is_24 (x₁ x₂ : ℝ) :
  (x₁ + 3) * (x₁ - 4) = 24 →
  (x₂ + 3) * (x₂ - 4) = 24 →
  x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_when_product_is_24_l2451_245196


namespace NUMINAMATH_CALUDE_problem_solution_l2451_245165

theorem problem_solution (x y : ℝ) 
  (hx : 2 < x ∧ x < 3) 
  (hy : -2 < y ∧ y < -1) 
  (hxy : x < y ∧ y < 0) : 
  (0 < x + y ∧ x + y < 2) ∧ 
  (3 < x - y ∧ x - y < 5) ∧ 
  (-6 < x * y ∧ x * y < -2) ∧
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2451_245165


namespace NUMINAMATH_CALUDE_jack_morning_emails_l2451_245143

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 16

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 2

theorem jack_morning_emails : 
  morning_emails = afternoon_emails + email_difference := by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l2451_245143


namespace NUMINAMATH_CALUDE_f_minimum_l2451_245166

def f (x : ℝ) : ℝ := (x^2 + 4*x + 5)*(x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem f_minimum :
  (∀ x : ℝ, f x ≥ -9) ∧ (∃ x : ℝ, f x = -9) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_l2451_245166


namespace NUMINAMATH_CALUDE_cloth_loss_per_metre_l2451_245110

def cloth_problem (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : Prop :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  let loss_per_metre := total_loss / total_metres
  total_metres = 300 ∧ 
  total_selling_price = 18000 ∧ 
  cost_price_per_metre = 65 ∧
  loss_per_metre = 5

theorem cloth_loss_per_metre :
  ∃ (total_metres total_selling_price cost_price_per_metre : ℕ),
    cloth_problem total_metres total_selling_price cost_price_per_metre :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_loss_per_metre_l2451_245110


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2451_245124

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n
  prop1 : a 7 * a 11 = 6
  prop2 : a 4 + a 14 = 5

/-- The main theorem stating the possible values of a_20 / a_10 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 10 = 2/3 ∨ seq.a 20 / seq.a 10 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2451_245124


namespace NUMINAMATH_CALUDE_clubsuit_computation_l2451_245148

-- Define the ♣ operation
def clubsuit (a c b : ℚ) : ℚ := (2 * a + c) / b

-- State the theorem
theorem clubsuit_computation : 
  clubsuit (clubsuit 6 1 (clubsuit 4 2 3)) 2 2 = 49 / 10 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_computation_l2451_245148


namespace NUMINAMATH_CALUDE_cookie_sale_total_l2451_245134

/-- Represents the number of cookies sold for each type -/
structure CookieSales where
  raisin : ℕ
  oatmeal : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Defines the conditions of the cookie sale -/
def cookie_sale_conditions (sales : CookieSales) : Prop :=
  sales.raisin = 42 ∧
  sales.raisin = 6 * sales.oatmeal ∧
  6 * sales.oatmeal = sales.oatmeal + 3 * sales.oatmeal + 2 * sales.oatmeal

theorem cookie_sale_total (sales : CookieSales) :
  cookie_sale_conditions sales →
  sales.raisin + sales.oatmeal + sales.chocolate_chip + sales.peanut_butter = 84 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sale_total_l2451_245134


namespace NUMINAMATH_CALUDE_parabola_equation_l2451_245168

/-- A parabola with its focus and a line passing through it -/
structure ParabolaWithLine where
  p : ℝ
  focus : ℝ × ℝ
  line : Set (ℝ × ℝ)
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The conditions of the problem -/
def problem_conditions (P : ParabolaWithLine) : Prop :=
  P.p > 0 ∧
  P.focus = (P.p / 2, 0) ∧
  P.focus ∈ P.line ∧
  P.A ∈ P.line ∧ P.B ∈ P.line ∧
  P.A.1 ^ 2 = 2 * P.p * P.A.2 ∧
  P.B.1 ^ 2 = 2 * P.p * P.B.2 ∧
  ((P.A.1 + P.B.1) / 2, (P.A.2 + P.B.2) / 2) = (3, 2)

/-- The theorem statement -/
theorem parabola_equation (P : ParabolaWithLine) :
  problem_conditions P →
  P.p = 2 ∨ P.p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2451_245168


namespace NUMINAMATH_CALUDE_volume_63_ounces_l2451_245127

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assertion that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_63_ounces (s : Substance) 
  (h : volume s 112 = 48) : volume s 63 = 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_63_ounces_l2451_245127


namespace NUMINAMATH_CALUDE_mountain_temperature_l2451_245180

theorem mountain_temperature (T : ℝ) 
  (h1 : T * (3/4) = T - 21) : T = 84 := by
  sorry

end NUMINAMATH_CALUDE_mountain_temperature_l2451_245180


namespace NUMINAMATH_CALUDE_no_number_decreases_58_times_when_first_digit_removed_l2451_245142

theorem no_number_decreases_58_times_when_first_digit_removed :
  ¬ ∃ (n : ℕ) (x y : ℕ), 
    n ≥ 2 ∧ 
    x > 0 ∧ x < 10 ∧
    y > 0 ∧
    x * 10^(n-1) + y = 58 * y :=
by sorry

end NUMINAMATH_CALUDE_no_number_decreases_58_times_when_first_digit_removed_l2451_245142


namespace NUMINAMATH_CALUDE_equation_solution_property_l2451_245130

theorem equation_solution_property (m n : ℝ) : 
  (∃ x : ℝ, m * x + n - 2 = 0 ∧ x = 2) → 2 * m + n + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_property_l2451_245130


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2451_245151

-- Define the types for line and plane
variable (m : Line) (α : Plane)

-- Define the property of having no common points
def noCommonPoints (l : Line) (p : Plane) : Prop := sorry

-- Define the property of being parallel
def isParallel (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_plane (h : noCommonPoints m α) : isParallel m α := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2451_245151


namespace NUMINAMATH_CALUDE_bicycle_car_speed_problem_l2451_245190

theorem bicycle_car_speed_problem (distance : ℝ) (delay : ℝ) 
  (h_distance : distance = 10) 
  (h_delay : delay = 1/3) : 
  ∃ (x : ℝ), x > 0 ∧ distance / x = distance / (2 * x) + delay → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_car_speed_problem_l2451_245190


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l2451_245156

/-- Represents the cost of a text messaging plan -/
structure PlanCost where
  perMessage : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages -/
def totalCost (plan : PlanCost) (messages : ℕ) : ℚ :=
  plan.perMessage * messages + plan.monthlyFee

/-- The two text messaging plans offered by the cell phone company -/
def planA : PlanCost := { perMessage := 0.25, monthlyFee := 9 }
def planB : PlanCost := { perMessage := 0.40, monthlyFee := 0 }

theorem equal_cost_at_60_messages :
  ∃ (messages : ℕ), messages = 60 ∧ totalCost planA messages = totalCost planB messages :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l2451_245156


namespace NUMINAMATH_CALUDE_g_25_l2451_245159

-- Define the function g
variable (g : ℝ → ℝ)

-- State the conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = y * g x
axiom g_50 : g 50 = 10

-- State the theorem to be proved
theorem g_25 : g 25 = 20 := by sorry

end NUMINAMATH_CALUDE_g_25_l2451_245159


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_16_l2451_245102

/-- Represents a sequence of C's and D's -/
inductive CDSequence
  | C : CDSequence → CDSequence
  | D : CDSequence → CDSequence
  | empty : CDSequence

/-- Returns true if the given sequence satisfies the conditions -/
def isValidSequence (s : CDSequence) : Bool :=
  sorry

/-- Returns the length of the given sequence -/
def sequenceLength (s : CDSequence) : Nat :=
  sorry

/-- Returns the number of valid sequences of a given length -/
def countValidSequences (n : Nat) : Nat :=
  sorry

theorem valid_sequences_of_length_16 :
  countValidSequences 16 = 55 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_16_l2451_245102


namespace NUMINAMATH_CALUDE_selected_students_in_range_l2451_245150

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  totalPopulation : ℕ
  sampleSize : ℕ
  initialSelection : ℕ
  rangeStart : ℕ
  rangeEnd : ℕ

/-- Calculates the number of selected items in a given range for a systematic sample -/
def selectedInRange (s : SystematicSample) : ℕ :=
  sorry

/-- Theorem stating the number of selected students in the given range -/
theorem selected_students_in_range :
  let s : SystematicSample := {
    totalPopulation := 100,
    sampleSize := 25,
    initialSelection := 4,
    rangeStart := 46,
    rangeEnd := 78
  }
  selectedInRange s = 8 := by sorry

end NUMINAMATH_CALUDE_selected_students_in_range_l2451_245150


namespace NUMINAMATH_CALUDE_combined_weight_is_63_l2451_245160

/-- The combined weight of candles made by Ethan -/
def combined_weight : ℕ :=
  let beeswax_per_candle : ℕ := 8
  let coconut_oil_per_candle : ℕ := 1
  let total_candles : ℕ := 10 - 3
  let weight_per_candle : ℕ := beeswax_per_candle + coconut_oil_per_candle
  total_candles * weight_per_candle

/-- Theorem stating that the combined weight of candles is 63 ounces -/
theorem combined_weight_is_63 : combined_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_is_63_l2451_245160


namespace NUMINAMATH_CALUDE_product_without_linear_term_l2451_245197

theorem product_without_linear_term (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 8) = a * x^2 + b) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_product_without_linear_term_l2451_245197


namespace NUMINAMATH_CALUDE_multiplication_of_powers_of_ten_l2451_245170

theorem multiplication_of_powers_of_ten : (2 * 10^3) * (8 * 10^3) = 1.6 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_of_ten_l2451_245170


namespace NUMINAMATH_CALUDE_probability_at_least_one_grade_12_l2451_245163

def total_sample_size : ℕ := 6
def grade_10_size : ℕ := 54
def grade_11_size : ℕ := 18
def grade_12_size : ℕ := 36

def grade_10_selected : ℕ := 3
def grade_11_selected : ℕ := 1
def grade_12_selected : ℕ := 2

def selected_size : ℕ := 3

theorem probability_at_least_one_grade_12 :
  let total_combinations := Nat.choose total_sample_size selected_size
  let favorable_combinations := total_combinations - Nat.choose (total_sample_size - grade_12_selected) selected_size
  (favorable_combinations : ℚ) / total_combinations = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_grade_12_l2451_245163


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1512_l2451_245164

/-- The largest perfect square factor of 1512 is 36 -/
theorem largest_perfect_square_factor_of_1512 :
  ∃ (n : ℕ), n * n = 36 ∧ n * n ∣ 1512 ∧ ∀ (m : ℕ), m * m ∣ 1512 → m * m ≤ n * n :=
by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1512_l2451_245164


namespace NUMINAMATH_CALUDE_john_rejection_percentage_l2451_245139

theorem john_rejection_percentage
  (jane_rejection_rate : ℝ)
  (total_rejection_rate : ℝ)
  (jane_inspection_fraction : ℝ)
  (h1 : jane_rejection_rate = 0.009)
  (h2 : total_rejection_rate = 0.0075)
  (h3 : jane_inspection_fraction = 0.625)
  : ∃ (john_rejection_rate : ℝ),
    john_rejection_rate = 0.005 ∧
    jane_rejection_rate * jane_inspection_fraction +
    john_rejection_rate * (1 - jane_inspection_fraction) =
    total_rejection_rate :=
by sorry

end NUMINAMATH_CALUDE_john_rejection_percentage_l2451_245139


namespace NUMINAMATH_CALUDE_express_vector_as_linear_combination_l2451_245138

/-- Given two vectors a and b in ℝ², express vector c as a linear combination of a and b -/
theorem express_vector_as_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) (hb : b = (1, -1)) (hc : c = (2, 3)) :
  ∃ x y : ℝ, c = x • a + y • b ∧ x = (5 : ℝ) / 2 ∧ y = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_express_vector_as_linear_combination_l2451_245138


namespace NUMINAMATH_CALUDE_a_range_l2451_245146

theorem a_range (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_a_range_l2451_245146


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2451_245140

theorem quadratic_equation_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, x^2 + a*x + b = 0 ↔ x = -2*a ∨ x = b) →
  (b = -2*(-2*a) ∨ -2*a = -2*b) →
  a = -1/2 ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2451_245140


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2451_245100

/-- A point P on the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P with coordinates (a^2 - 1, a + 1) -/
def P (a : ℝ) : Point := ⟨a^2 - 1, a + 1⟩

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If P(a^2 - 1, a + 1) is on the y-axis, then its coordinates are (0, 2) or (0, 0) -/
theorem point_on_y_axis (a : ℝ) : 
  on_y_axis (P a) → (P a = ⟨0, 2⟩ ∨ P a = ⟨0, 0⟩) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2451_245100


namespace NUMINAMATH_CALUDE_route_time_difference_l2451_245177

theorem route_time_difference (x : ℝ) (h : x > 0) : 
  10 / x - 7 / ((1 + 0.4) * x) = 10 / 60 :=
by
  sorry

#check route_time_difference

end NUMINAMATH_CALUDE_route_time_difference_l2451_245177


namespace NUMINAMATH_CALUDE_sum_of_h_at_x_values_l2451_245185

def f (x : ℝ) : ℝ := |x| - 3

def g (x : ℝ) : ℝ := -x

def h (x : ℝ) : ℝ := f (g (f x))

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_h_at_x_values :
  (x_values.map h).sum = -17 := by sorry

end NUMINAMATH_CALUDE_sum_of_h_at_x_values_l2451_245185


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2451_245167

theorem units_digit_of_product (a b c : ℕ) : 
  (2^104 * 5^205 * 11^302) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2451_245167


namespace NUMINAMATH_CALUDE_inequality_proof_l2451_245126

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a * b + b * c + c * a ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c ∧ Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2451_245126


namespace NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l2451_245182

theorem odd_prime_fifth_power_difference (p : ℕ) (h_prime : Prime p) (h_odd : Odd p)
  (h_fifth_power_diff : ∃ (a b : ℕ), p = a^5 - b^5) :
  ∃ (n : ℕ), Odd n ∧ (((4 * p + 1) : ℚ) / 5).sqrt = ((n^2 + 1) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l2451_245182


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2451_245157

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : S seq 8 = 4 * seq.a 3)
  (h2 : seq.a 7 = -2) :
  seq.a 9 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2451_245157


namespace NUMINAMATH_CALUDE_blue_segments_count_l2451_245136

/-- Set A of points (x, y) where x and y are natural numbers between 1 and 20 inclusive -/
def A : Set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20}

/-- Set B of points (x, y) where x and y are natural numbers between 2 and 19 inclusive -/
def B : Set (ℕ × ℕ) := {p | 2 ≤ p.1 ∧ p.1 ≤ 19 ∧ 2 ≤ p.2 ∧ p.2 ≤ 19}

/-- Color of a point in A -/
inductive Color
| Red
| Blue

/-- Coloring function for points in A -/
def coloring : A → Color := sorry

/-- Total number of red points in A -/
def total_red_points : ℕ := 219

/-- Number of red points in B -/
def red_points_in_B : ℕ := 180

/-- Corner points are blue -/
axiom corner_points_blue :
  coloring ⟨(1, 1), sorry⟩ = Color.Blue ∧
  coloring ⟨(1, 20), sorry⟩ = Color.Blue ∧
  coloring ⟨(20, 1), sorry⟩ = Color.Blue ∧
  coloring ⟨(20, 20), sorry⟩ = Color.Blue

/-- Number of black line segments of length 1 -/
def black_segments : ℕ := 237

/-- Theorem: The number of blue line segments of length 1 is 233 -/
theorem blue_segments_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_blue_segments_count_l2451_245136


namespace NUMINAMATH_CALUDE_pascal_triangle_45th_number_51_entries_l2451_245187

theorem pascal_triangle_45th_number_51_entries : 
  let n : ℕ := 50  -- The row number (0-indexed) with 51 entries
  let k : ℕ := 44  -- The position (0-indexed) of the 45th number
  Nat.choose n k = 19380000 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_45th_number_51_entries_l2451_245187


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2451_245120

theorem exactly_one_greater_than_one (a b c : ℝ) : 
  a * b * c = 1 → a + b + c > 1/a + 1/b + 1/c → 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2451_245120


namespace NUMINAMATH_CALUDE_square_perimeter_l2451_245115

/-- The perimeter of a square with side length 19 cm is 76 cm. -/
theorem square_perimeter : 
  ∀ (s : ℝ), s = 19 → 4 * s = 76 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2451_245115


namespace NUMINAMATH_CALUDE_infinitely_many_primes_l2451_245149

theorem infinitely_many_primes : ∀ S : Finset Nat, (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_l2451_245149


namespace NUMINAMATH_CALUDE_sqrt_36_div_6_l2451_245193

theorem sqrt_36_div_6 : Real.sqrt 36 / 6 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_36_div_6_l2451_245193


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l2451_245113

theorem sum_of_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 4) (sum_sq_eq : x^2 + y^2 + z^2 = 6) :
  ∃ (m M : ℝ), (∀ x', (∃ y' z', x' + y' + z' = 4 ∧ x'^2 + y'^2 + z'^2 = 6) → m ≤ x' ∧ x' ≤ M) ∧
  m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l2451_245113


namespace NUMINAMATH_CALUDE_mode_is_80_l2451_245107

/-- Represents the frequency of each score in the test results -/
def score_frequency : List (Nat × Nat) := [
  (61, 1), (61, 1), (62, 1),
  (75, 1), (77, 1),
  (80, 3), (81, 1), (83, 2),
  (92, 2), (94, 1), (96, 1), (97, 2),
  (105, 2), (109, 1),
  (110, 2)
]

/-- The maximum score possible on the test -/
def max_score : Nat := 120

/-- Definition of mode: the value that appears most frequently in a dataset -/
def is_mode (scores : List (Nat × Nat)) (m : Nat) : Prop :=
  ∀ x, (x ∈ scores.map Prod.fst) → 
    (scores.filter (fun p => p.fst = m)).length ≥ (scores.filter (fun p => p.fst = x)).length

/-- Theorem: The mode of the given set of scores is 80 -/
theorem mode_is_80 : is_mode score_frequency 80 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_80_l2451_245107


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2451_245112

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 1| = 3

/-- The first parabola equation when y ≥ 1 -/
def parabola1 (x y : ℝ) : Prop :=
  y = -1/8 * x^2 + 2

/-- The second parabola equation when y < 1 -/
def parabola2 (x y : ℝ) : Prop :=
  y = 1/4 * x^2 - 1

/-- The vertex of the first parabola -/
def vertex1 : ℝ × ℝ := (0, 2)

/-- The vertex of the second parabola -/
def vertex2 : ℝ × ℝ := (0, -1)

theorem distance_between_vertices :
  |vertex1.2 - vertex2.2| = 3 :=
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2451_245112


namespace NUMINAMATH_CALUDE_train_speed_l2451_245192

/-- Calculate the speed of a train given its length and time to pass an observer -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 9) :
  length / time = 20 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2451_245192


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2451_245175

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 5 * x^3 + x + 20) - (x^6 + 4 * x^5 - 2 * x^4 + x^3 + 15) =
  x^6 - x^5 + 3 * x^4 + 4 * x^3 + x + 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2451_245175


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2451_245186

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2451_245186


namespace NUMINAMATH_CALUDE_meeting_point_distance_l2451_245111

/-- 
Given two people starting at opposite ends of a path, prove that they meet
when the slower person has traveled a specific distance.
-/
theorem meeting_point_distance 
  (total_distance : ℝ) 
  (speed_slow : ℝ) 
  (speed_fast : ℝ) 
  (h1 : total_distance = 36)
  (h2 : speed_slow = 3)
  (h3 : speed_fast = 6)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (meeting_distance : ℝ), 
    meeting_distance = total_distance * speed_slow / (speed_slow + speed_fast) ∧ 
    meeting_distance = 12 := by
sorry


end NUMINAMATH_CALUDE_meeting_point_distance_l2451_245111


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2451_245188

theorem largest_prime_factor_of_1729 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 1729 → q ≤ p ∧ p = 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2451_245188


namespace NUMINAMATH_CALUDE_men_left_hostel_l2451_245116

/-- Proves that 50 men left the hostel given the initial and final conditions -/
theorem men_left_hostel (initial_men : ℕ) (initial_days : ℕ) (final_days : ℕ) 
  (h1 : initial_men = 250)
  (h2 : initial_days = 40)
  (h3 : final_days = 50)
  (h4 : initial_men * initial_days = (initial_men - men_left) * final_days) :
  men_left = 50 := by
  sorry

#check men_left_hostel

end NUMINAMATH_CALUDE_men_left_hostel_l2451_245116


namespace NUMINAMATH_CALUDE_sqrt_factorial_fraction_l2451_245105

theorem sqrt_factorial_fraction : 
  let factorial_10 : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let denominator : ℕ := 2 * 3 * 7 * 7
  Real.sqrt (factorial_10 / denominator) = 120 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sqrt_factorial_fraction_l2451_245105


namespace NUMINAMATH_CALUDE_power_five_mod_seven_l2451_245114

theorem power_five_mod_seven : 5^2010 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_seven_l2451_245114


namespace NUMINAMATH_CALUDE_f_properties_l2451_245158

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x + a) / x

def g : ℝ → ℝ := λ x => 1

theorem f_properties (a : ℝ) :
  (∀ x > 0, f a x ≤ Real.exp (a - 1)) ∧
  (∃ x > 0, x ≤ Real.exp 2 ∧ f a x = g x) ↔ a ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_properties_l2451_245158


namespace NUMINAMATH_CALUDE_probability_one_second_class_l2451_245181

/-- The probability of drawing exactly one second-class product from a batch of products -/
theorem probability_one_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (drawn : ℕ) :
  total = first_class + second_class →
  total = 100 →
  first_class = 90 →
  second_class = 10 →
  drawn = 4 →
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose total drawn =
    Nat.choose second_class 1 * Nat.choose first_class 3 / Nat.choose total drawn :=
by sorry

end NUMINAMATH_CALUDE_probability_one_second_class_l2451_245181


namespace NUMINAMATH_CALUDE_f_difference_l2451_245119

/-- The function f(x) = x^4 + 3x^3 + x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 3*x^3 + x^2 + 7*x

/-- Theorem: f(3) - f(-3) = 204 -/
theorem f_difference : f 3 - f (-3) = 204 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2451_245119


namespace NUMINAMATH_CALUDE_exists_lcm_sum_for_non_power_of_two_l2451_245162

/-- Represents the least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ := (a * b) / Nat.gcd a b

/-- Theorem: For any natural number n that is not a power of 2,
    there exist positive integers a, b, and c such that
    n = lcm a b + lcm b c + lcm c a -/
theorem exists_lcm_sum_for_non_power_of_two (n : ℕ) 
    (h : ∀ k : ℕ, n ≠ 2^k) :
    ∃ (a b c : ℕ+), n = lcm a b + lcm b c + lcm c a := by
  sorry

end NUMINAMATH_CALUDE_exists_lcm_sum_for_non_power_of_two_l2451_245162


namespace NUMINAMATH_CALUDE_min_units_for_nonnegative_profit_l2451_245184

/-- Represents the profit function for ice powder sales -/
def profit : ℕ → ℤ
| 0 => -120
| 10 => -80
| 20 => -40
| 30 => 0
| 40 => 40
| 50 => 80
| _ => 0  -- Default case, not used in the proof

/-- Theorem: The minimum number of units to be sold for non-negative profit is 30 -/
theorem min_units_for_nonnegative_profit :
  (∀ x : ℕ, x < 30 → profit x < 0) ∧
  profit 30 = 0 ∧
  (∀ x : ℕ, x > 30 → profit x > 0) :=
by sorry


end NUMINAMATH_CALUDE_min_units_for_nonnegative_profit_l2451_245184


namespace NUMINAMATH_CALUDE_equation_solution_l2451_245179

theorem equation_solution : ∃ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2451_245179


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l2451_245103

theorem pentagon_angle_sum (A B C x y : ℝ) : 
  A = 35 → B = 65 → C = 40 → 
  (A + B + C + x + y + 180 = 540) →
  x + y = 140 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l2451_245103


namespace NUMINAMATH_CALUDE_units_digit_of_product_division_l2451_245141

theorem units_digit_of_product_division : 
  (30 * 31 * 32 * 33 * 34 * 35) / 2500 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_division_l2451_245141


namespace NUMINAMATH_CALUDE_min_plates_matching_pair_l2451_245161

/-- Represents the colors of plates -/
inductive PlateColor
  | White
  | Green
  | Red
  | Pink
  | Purple

/-- The minimum number of plates needed to guarantee a matching pair -/
def min_plates_for_match : ℕ := 6

/-- Theorem stating that given at least one plate of each of 5 colors,
    the minimum number of plates needed to guarantee a matching pair is 6 -/
theorem min_plates_matching_pair
  (white_count : ℕ) (green_count : ℕ) (red_count : ℕ) (pink_count : ℕ) (purple_count : ℕ)
  (h_white : white_count ≥ 1)
  (h_green : green_count ≥ 1)
  (h_red : red_count ≥ 1)
  (h_pink : pink_count ≥ 1)
  (h_purple : purple_count ≥ 1) :
  min_plates_for_match = 6 := by
  sorry

#check min_plates_matching_pair

end NUMINAMATH_CALUDE_min_plates_matching_pair_l2451_245161


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2451_245155

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2451_245155


namespace NUMINAMATH_CALUDE_range_of_x_l2451_245147

-- Define the sets
def S1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def S2 : Set ℝ := {x | x < 1 ∨ x > 4}

-- Define the condition
def condition (x : ℝ) : Prop := ¬(x ∈ S1 ∨ x ∈ S2)

-- State the theorem
theorem range_of_x : 
  ∀ x : ℝ, condition x → x ∈ {y : ℝ | 1 ≤ y ∧ y < 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l2451_245147
