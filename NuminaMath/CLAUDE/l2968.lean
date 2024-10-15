import Mathlib

namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2968_296846

/-- Given that the total marks in physics, chemistry, and mathematics is 140 more than 
    the marks in physics, prove that the average mark in chemistry and mathematics is 70. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) 
  (h : P + C + M = P + 140) : (C + M) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2968_296846


namespace NUMINAMATH_CALUDE_reverse_product_92565_l2968_296849

def is_reverse (a b : ℕ) : Prop :=
  (Nat.digits 10 a).reverse = Nat.digits 10 b

theorem reverse_product_92565 :
  ∃! (a b : ℕ), a < b ∧ is_reverse a b ∧ a * b = 92565 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_reverse_product_92565_l2968_296849


namespace NUMINAMATH_CALUDE_log_sum_fifty_twenty_l2968_296866

theorem log_sum_fifty_twenty : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_fifty_twenty_l2968_296866


namespace NUMINAMATH_CALUDE_overlapping_squares_theorem_l2968_296886

/-- Represents a rectangle with numbers placed inside it -/
structure NumberedRectangle where
  width : ℕ
  height : ℕ
  numbers : List ℕ

/-- Represents the result of rotating a NumberedRectangle by 180° -/
def rotate180 (nr : NumberedRectangle) : NumberedRectangle :=
  { width := nr.width,
    height := nr.height,
    numbers := [6, 1, 2, 1] }

/-- Calculates the number of overlapping shaded squares when a NumberedRectangle is overlaid with its 180° rotation -/
def overlappingSquares (nr : NumberedRectangle) : ℕ :=
  nr.width * nr.height - 10

/-- The main theorem to be proved -/
theorem overlapping_squares_theorem (nr : NumberedRectangle) :
  nr.width = 8 ∧ nr.height = 5 ∧ nr.numbers = [1, 2, 1, 9] →
  rotate180 nr = { width := 8, height := 5, numbers := [6, 1, 2, 1] } →
  overlappingSquares nr = 30 := by
  sorry

#check overlapping_squares_theorem

end NUMINAMATH_CALUDE_overlapping_squares_theorem_l2968_296886


namespace NUMINAMATH_CALUDE_probability_different_colors_eq_137_162_l2968_296837

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probabilityDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.yellow + counts.green
  let pBlue := counts.blue / total
  let pRed := counts.red / total
  let pYellow := counts.yellow / total
  let pGreen := counts.green / total
  pBlue * (1 - pBlue) + pRed * (1 - pRed) + pYellow * (1 - pYellow) + pGreen * (1 - pGreen)

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_eq_137_162 :
  probabilityDifferentColors ⟨6, 5, 4, 3⟩ = 137 / 162 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_eq_137_162_l2968_296837


namespace NUMINAMATH_CALUDE_inscribed_hexagon_properties_l2968_296859

/-- A regular hexagon inscribed in a circle -/
structure InscribedHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : side_length > 0

/-- Properties of the inscribed hexagon -/
def InscribedHexagon.properties (h : InscribedHexagon) : Prop :=
  let r := h.side_length  -- radius of the circle is equal to side length
  let C := 2 * Real.pi * r  -- circumference of the circle
  let arc_length := C / 6  -- arc length for one side of the hexagon
  let P := 6 * h.side_length  -- perimeter of the hexagon
  C = 10 * Real.pi ∧ 
  arc_length = 5 * Real.pi / 3 ∧ 
  P = 30

/-- Theorem stating the properties of a regular hexagon with side length 5 inscribed in a circle -/
theorem inscribed_hexagon_properties :
  ∀ (h : InscribedHexagon), h.side_length = 5 → h.properties := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_properties_l2968_296859


namespace NUMINAMATH_CALUDE_arc_length_calculation_l2968_296882

theorem arc_length_calculation (r : ℝ) (θ_central : ℝ) (θ_peripheral : ℝ) :
  r = 5 →
  θ_central = (2/3) * θ_peripheral →
  θ_peripheral = 2 * π →
  r * θ_central = (20 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l2968_296882


namespace NUMINAMATH_CALUDE_sum_of_X_and_Y_l2968_296827

/-- X is defined as 2 groups of 10 plus 6 units -/
def X : ℕ := 2 * 10 + 6

/-- Y is defined as 4 groups of 10 plus 1 unit -/
def Y : ℕ := 4 * 10 + 1

/-- The sum of X and Y is 67 -/
theorem sum_of_X_and_Y : X + Y = 67 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_X_and_Y_l2968_296827


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l2968_296839

/-- Calculates the distance traveled against the stream in one hour -/
def distance_against_stream (boat_speed : ℝ) (downstream_distance : ℝ) : ℝ :=
  let stream_speed := downstream_distance - boat_speed
  boat_speed - stream_speed

/-- Theorem: Given a boat with speed 4 km/hr in still water that travels 6 km
    downstream in one hour, it will travel 2 km upstream in one hour -/
theorem boat_upstream_distance :
  distance_against_stream 4 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l2968_296839


namespace NUMINAMATH_CALUDE_ryan_sandwich_slices_l2968_296894

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of slices of bread needed for one sandwich -/
def slices_per_sandwich : ℕ := 3

/-- The total number of slices needed for all sandwiches -/
def total_slices : ℕ := num_sandwiches * slices_per_sandwich

theorem ryan_sandwich_slices : total_slices = 15 := by
  sorry

end NUMINAMATH_CALUDE_ryan_sandwich_slices_l2968_296894


namespace NUMINAMATH_CALUDE_rectangle_area_l2968_296831

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

/-- Represents a rectangle defined by four vertices -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Theorem: Area of a specific rectangle -/
theorem rectangle_area (rect : Rectangle)
  (h1 : rect.P = ⟨1, 1⟩)
  (h2 : rect.Q = ⟨-3, 2⟩)
  (h3 : rect.R = ⟨-1, 6⟩)
  (h4 : rect.S = ⟨3, 5⟩)
  (h5 : squaredDistance rect.P rect.Q = squaredDistance rect.R rect.S) -- PQ is one side
  (h6 : squaredDistance rect.P rect.R = squaredDistance rect.Q rect.S) -- PR is a diagonal
  : (squaredDistance rect.P rect.Q * squaredDistance rect.P rect.R : ℝ) = 4 * 51 :=
sorry


end NUMINAMATH_CALUDE_rectangle_area_l2968_296831


namespace NUMINAMATH_CALUDE_eight_students_in_neither_l2968_296855

/-- Represents the number of students in various categories of a science club. -/
structure ScienceClub where
  total : ℕ
  biology : ℕ
  chemistry : ℕ
  both : ℕ

/-- Calculates the number of students taking neither biology nor chemistry. -/
def studentsInNeither (club : ScienceClub) : ℕ :=
  club.total - (club.biology + club.chemistry - club.both)

/-- Theorem stating that for the given science club configuration, 
    8 students take neither biology nor chemistry. -/
theorem eight_students_in_neither (club : ScienceClub) 
  (h1 : club.total = 60)
  (h2 : club.biology = 42)
  (h3 : club.chemistry = 35)
  (h4 : club.both = 25) : 
  studentsInNeither club = 8 := by
  sorry

#eval studentsInNeither { total := 60, biology := 42, chemistry := 35, both := 25 }

end NUMINAMATH_CALUDE_eight_students_in_neither_l2968_296855


namespace NUMINAMATH_CALUDE_paving_cost_specific_room_l2968_296840

/-- Calculates the cost of paving a floor consisting of two rectangles -/
def paving_cost (length1 width1 length2 width2 cost_per_sqm : ℝ) : ℝ :=
  ((length1 * width1 + length2 * width2) * cost_per_sqm)

/-- Theorem: The cost of paving the specific room is Rs. 26,100 -/
theorem paving_cost_specific_room :
  paving_cost 5.5 3.75 4 3 800 = 26100 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_specific_room_l2968_296840


namespace NUMINAMATH_CALUDE_sum_1000th_to_1010th_term_l2968_296824

def arithmeticSequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

def sumArithmeticSequence (a₁ d m n : ℕ) : ℕ :=
  ((n - m + 1) * (arithmeticSequence a₁ d m + arithmeticSequence a₁ d n)) / 2

theorem sum_1000th_to_1010th_term :
  sumArithmeticSequence 3 7 1000 1010 = 77341 := by
  sorry

end NUMINAMATH_CALUDE_sum_1000th_to_1010th_term_l2968_296824


namespace NUMINAMATH_CALUDE_sum_of_unit_vector_magnitudes_l2968_296812

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given two unit vectors, prove that the sum of their magnitudes is 2 -/
theorem sum_of_unit_vector_magnitudes
  (a₀ b₀ : E) 
  (ha : ‖a₀‖ = 1) 
  (hb : ‖b₀‖ = 1) : 
  ‖a₀‖ + ‖b₀‖ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_unit_vector_magnitudes_l2968_296812


namespace NUMINAMATH_CALUDE_total_lamps_is_147_l2968_296872

/-- The number of lamps per room -/
def lamps_per_room : ℕ := 7

/-- The number of rooms in the hotel -/
def rooms : ℕ := 21

/-- The total number of lamps bought for the hotel -/
def total_lamps : ℕ := lamps_per_room * rooms

/-- Theorem stating that the total number of lamps bought is 147 -/
theorem total_lamps_is_147 : total_lamps = 147 := by sorry

end NUMINAMATH_CALUDE_total_lamps_is_147_l2968_296872


namespace NUMINAMATH_CALUDE_sieve_of_eratosthenes_complexity_l2968_296847

/-- The Sieve of Eratosthenes algorithm for finding prime numbers up to n. -/
def sieve_of_eratosthenes (n : ℕ) : List ℕ := sorry

/-- The time complexity function for the Sieve of Eratosthenes algorithm. -/
def time_complexity (n : ℕ) : ℝ := sorry

/-- Big O notation for comparing functions. -/
def big_o (f g : ℕ → ℝ) : Prop := 
  ∃ c k : ℝ, c > 0 ∧ ∀ n : ℕ, n ≥ k → f n ≤ c * g n

/-- Theorem stating that the time complexity of the Sieve of Eratosthenes is O(n log(n)^2). -/
theorem sieve_of_eratosthenes_complexity :
  big_o time_complexity (λ n => n * (Real.log n)^2) :=
sorry

end NUMINAMATH_CALUDE_sieve_of_eratosthenes_complexity_l2968_296847


namespace NUMINAMATH_CALUDE_train_meeting_distance_l2968_296878

theorem train_meeting_distance (route_length : ℝ) (time_x time_y : ℝ) 
  (h1 : route_length = 160)
  (h2 : time_x = 5)
  (h3 : time_y = 3)
  : let speed_x := route_length / time_x
    let speed_y := route_length / time_y
    let meeting_time := route_length / (speed_x + speed_y)
    speed_x * meeting_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l2968_296878


namespace NUMINAMATH_CALUDE_election_result_count_l2968_296822

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of positions to be filled -/
def num_positions : ℕ := 2

/-- The number of ways to select students for positions with at least one female student -/
def ways_with_female : ℕ := total_students.choose num_positions * num_positions.factorial - num_male.choose num_positions * num_positions.factorial

theorem election_result_count : ways_with_female = 14 := by
  sorry

end NUMINAMATH_CALUDE_election_result_count_l2968_296822


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2968_296865

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5,
    and two sides with distinct integer lengths has an area of 12. -/
theorem quadrilateral_area : ∀ (A B C D : ℝ × ℝ),
  -- Right angles at B and D
  (B.2 - A.2) * (C.1 - B.1) + (B.1 - A.1) * (C.2 - B.2) = 0 →
  (D.2 - C.2) * (A.1 - D.1) + (D.1 - C.1) * (A.2 - D.2) = 0 →
  -- Diagonal AC = 5
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 25 →
  -- Two sides with distinct integer lengths
  ∃ (a b : ℕ), a ≠ b ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
     (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2) ∨
    ((A.1 - D.1)^2 + (A.2 - D.2)^2 = a^2 ∧
     (D.1 - C.1)^2 + (D.2 - C.2)^2 = b^2) →
  -- Area of ABCD is 12
  abs ((A.1 - C.1) * (B.2 - D.2) - (A.2 - C.2) * (B.1 - D.1)) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2968_296865


namespace NUMINAMATH_CALUDE_triamoeba_population_after_one_week_l2968_296835

/-- Represents the population of Triamoebas after a given number of days -/
def triamoeba_population (initial_population : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_rate ^ days

/-- Theorem stating that the Triamoeba population after 7 days is 2187 -/
theorem triamoeba_population_after_one_week :
  triamoeba_population 1 3 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_triamoeba_population_after_one_week_l2968_296835


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2968_296883

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2968_296883


namespace NUMINAMATH_CALUDE_fish_cost_is_80_l2968_296841

/-- The cost of fish per kilogram in pesos -/
def fish_cost : ℝ := sorry

/-- The cost of pork per kilogram in pesos -/
def pork_cost : ℝ := sorry

/-- First condition: 530 pesos can buy 4 kg of fish and 2 kg of pork -/
axiom condition1 : 4 * fish_cost + 2 * pork_cost = 530

/-- Second condition: 875 pesos can buy 7 kg of fish and 3 kg of pork -/
axiom condition2 : 7 * fish_cost + 3 * pork_cost = 875

/-- Theorem: The cost of a kilogram of fish is 80 pesos -/
theorem fish_cost_is_80 : fish_cost = 80 := by sorry

end NUMINAMATH_CALUDE_fish_cost_is_80_l2968_296841


namespace NUMINAMATH_CALUDE_boat_trip_theorem_l2968_296802

/-- Represents the boat trip scenario -/
structure BoatTrip where
  total_time : ℝ
  stream_velocity : ℝ
  boat_speed : ℝ
  distance : ℝ

/-- The specific boat trip instance from the problem -/
def problem_trip : BoatTrip where
  total_time := 38
  stream_velocity := 4
  boat_speed := 14
  distance := 360

/-- Theorem stating that the given boat trip satisfies the problem conditions -/
theorem boat_trip_theorem (trip : BoatTrip) : 
  trip.total_time = 38 ∧ 
  trip.stream_velocity = 4 ∧ 
  trip.boat_speed = 14 ∧
  trip.distance / (trip.boat_speed + trip.stream_velocity) + 
    (trip.distance / 2) / (trip.boat_speed - trip.stream_velocity) = trip.total_time →
  trip.distance = 360 := by
  sorry

#check boat_trip_theorem problem_trip

end NUMINAMATH_CALUDE_boat_trip_theorem_l2968_296802


namespace NUMINAMATH_CALUDE_function_properties_and_inequality_l2968_296864

/-- Given a function f(x) = ax / (x^2 + b) with specific properties, 
    prove its exact form and a related inequality. -/
theorem function_properties_and_inequality 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 1) 
  (h_def : ∀ x, f x = a * x / (x^2 + b)) 
  (h_f1 : f 1 = 1) 
  (h_max : ∀ x, f x ≤ 3 * Real.sqrt 2 / 4) 
  (h_attains_max : ∃ x, f x = 3 * Real.sqrt 2 / 4) :
  (∀ x, f x = 3 * x / (x^2 + 2)) ∧ 
  (∀ m, (2 < m ∧ m ≤ 4) ↔ 
    (∀ x ∈ Set.Icc 1 2, f x ≤ 3 * m / ((x^2 + 2) * |x - m|))) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_inequality_l2968_296864


namespace NUMINAMATH_CALUDE_max_sum_problem_l2968_296832

theorem max_sum_problem (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_cubes : x^3 + y^3 + z^3 + v^3 + w^3 = 2024) : 
  ∃ (M x_M y_M z_M v_M w_M : ℝ),
    (∀ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ 
      a^3 + b^3 + c^3 + d^3 + e^3 = 2024 → 
      a*c + 3*b*c + 4*c*d + 8*c*e ≤ M) ∧
    x_M > 0 ∧ y_M > 0 ∧ z_M > 0 ∧ v_M > 0 ∧ w_M > 0 ∧
    x_M^3 + y_M^3 + z_M^3 + v_M^3 + w_M^3 = 2024 ∧
    x_M*z_M + 3*y_M*z_M + 4*z_M*v_M + 8*z_M*w_M = M ∧
    M + x_M + y_M + z_M + v_M + w_M = 3055 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_problem_l2968_296832


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2968_296823

/-- Calculates the difference between the principal and the simple interest --/
def interestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal - (principal * rate * time)

/-- Theorem stating that the difference between the principal and the simple interest
    is 340 for the given conditions --/
theorem interest_difference_theorem :
  interestDifference 500 0.04 8 = 340 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2968_296823


namespace NUMINAMATH_CALUDE_sales_tax_calculation_l2968_296862

def total_cost : ℝ := 25
def tax_rate : ℝ := 0.05
def tax_free_cost : ℝ := 18.7

theorem sales_tax_calculation :
  ∃ (taxable_cost : ℝ),
    taxable_cost + tax_free_cost + taxable_cost * tax_rate = total_cost ∧
    taxable_cost * tax_rate = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_calculation_l2968_296862


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2968_296817

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2968_296817


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l2968_296873

/-- Proves that in 5 classrooms, where each classroom has 24 students and 3 rabbits,
    the difference between the total number of students and the total number of rabbits is 105. -/
theorem student_rabbit_difference :
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 24
  let rabbits_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 105 := by
sorry


end NUMINAMATH_CALUDE_student_rabbit_difference_l2968_296873


namespace NUMINAMATH_CALUDE_positive_Y_value_l2968_296885

-- Define the ∆ relation
def triangle (X Y : ℝ) : ℝ := X^2 + 3*Y^2

-- Theorem statement
theorem positive_Y_value :
  ∃ Y : ℝ, Y > 0 ∧ triangle 9 Y = 360 ∧ Y = Real.sqrt 93 := by
  sorry

end NUMINAMATH_CALUDE_positive_Y_value_l2968_296885


namespace NUMINAMATH_CALUDE_jills_hair_braiding_l2968_296893

/-- Given the conditions of Jill's hair braiding for the dance team, 
    prove that each dancer has 5 braids. -/
theorem jills_hair_braiding 
  (num_dancers : ℕ) 
  (time_per_braid : ℕ) 
  (total_time_minutes : ℕ) 
  (h1 : num_dancers = 8)
  (h2 : time_per_braid = 30)
  (h3 : total_time_minutes = 20) :
  (total_time_minutes * 60) / (time_per_braid * num_dancers) = 5 :=
sorry

end NUMINAMATH_CALUDE_jills_hair_braiding_l2968_296893


namespace NUMINAMATH_CALUDE_pet_ownership_l2968_296869

theorem pet_ownership (total : ℕ) (dogs cats other_pets no_pets : ℕ) 
  (dogs_cats : ℕ) (dogs_other : ℕ) (cats_other : ℕ) :
  total = 32 →
  dogs = total / 2 →
  cats = total * 3 / 8 →
  other_pets = 6 →
  no_pets = 5 →
  dogs_cats = 10 →
  dogs_other = 2 →
  cats_other = 9 →
  ∃ (all_three : ℕ),
    all_three = 1 ∧
    dogs + cats + other_pets - dogs_cats - dogs_other - cats_other + all_three = total - no_pets :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_l2968_296869


namespace NUMINAMATH_CALUDE_family_weight_ratio_l2968_296863

/-- Given the weights of a family, prove the ratio of child's weight to grandmother's weight -/
theorem family_weight_ratio 
  (total_weight : ℝ) 
  (daughter_child_weight : ℝ) 
  (daughter_weight : ℝ) 
  (h1 : total_weight = 150) 
  (h2 : daughter_child_weight = 60) 
  (h3 : daughter_weight = 42) : 
  ∃ (child_weight grandmother_weight : ℝ), 
    total_weight = grandmother_weight + daughter_weight + child_weight ∧ 
    daughter_child_weight = daughter_weight + child_weight ∧
    child_weight / grandmother_weight = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_family_weight_ratio_l2968_296863


namespace NUMINAMATH_CALUDE_apple_basket_count_l2968_296810

theorem apple_basket_count : 
  ∀ (total : ℕ) (rotten : ℕ) (good : ℕ),
  rotten = (12 * total) / 100 →
  good = 66 →
  good = total - rotten →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_apple_basket_count_l2968_296810


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_million_l2968_296809

def T (n : ℕ) : ℕ := n * 2^(n-1)

theorem smallest_n_exceeding_million :
  (∀ k < 20, T k ≤ 10^6) ∧ T 20 > 10^6 := by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_million_l2968_296809


namespace NUMINAMATH_CALUDE_equality_check_l2968_296843

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-1)^3 = -1^3) ∧ 
  ((2/3)^2 ≠ 2^2/3) ∧ 
  ((-2)^2 ≠ -2^2) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l2968_296843


namespace NUMINAMATH_CALUDE_square_minus_four_equals_negative_three_l2968_296853

theorem square_minus_four_equals_negative_three (a : ℤ) (h : a = -1) : a^2 - 4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_four_equals_negative_three_l2968_296853


namespace NUMINAMATH_CALUDE_money_ratio_l2968_296851

def money_problem (total : ℚ) (rene : ℚ) : Prop :=
  ∃ (isha florence : ℚ) (k : ℕ),
    isha = (1/3) * total ∧
    florence = (1/2) * isha ∧
    florence = k * rene ∧
    total = isha + florence + rene ∧
    rene = 300 ∧
    total = 1650 ∧
    florence / rene = 3/2

theorem money_ratio :
  money_problem 1650 300 := by sorry

end NUMINAMATH_CALUDE_money_ratio_l2968_296851


namespace NUMINAMATH_CALUDE_arithmetic_sequence_201_l2968_296868

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_201 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_5 : a 5 = 33) 
  (h_45 : a 45 = 153) : 
  a 61 = 201 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_201_l2968_296868


namespace NUMINAMATH_CALUDE_unique_intersection_l2968_296875

/-- Parabola C defined by x²=4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line MH passing through points M(t,0) and H(2t,t²) -/
def line_MH (t x y : ℝ) : Prop := y = t*(x - t)

/-- Point H on parabola C -/
def point_H (t : ℝ) : ℝ × ℝ := (2*t, t^2)

theorem unique_intersection (t : ℝ) (h : t ≠ 0) :
  ∀ x y : ℝ, parabola_C x y ∧ line_MH t x y → (x, y) = point_H t :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l2968_296875


namespace NUMINAMATH_CALUDE_original_number_proof_l2968_296897

theorem original_number_proof (N : ℤ) : (N + 1) % 25 = 0 → N = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2968_296897


namespace NUMINAMATH_CALUDE_final_x_value_l2968_296833

/-- Represents the state of the program at each iteration -/
structure ProgramState where
  x : ℕ
  y : ℕ

/-- Updates the program state according to the given rules -/
def updateState (state : ProgramState) : ProgramState :=
  { x := state.x + 2,
    y := state.y + state.x + 2 }

/-- Checks if the program should continue running -/
def shouldContinue (state : ProgramState) : Bool :=
  state.y < 10000

/-- Computes the final state of the program -/
def finalState : ProgramState :=
  sorry

/-- Proves that the final value of x is 201 -/
theorem final_x_value :
  finalState.x = 201 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l2968_296833


namespace NUMINAMATH_CALUDE_chocolate_cost_proof_l2968_296836

/-- The cost of the chocolate given the total spent and the cost of the candy bar -/
def chocolate_cost (total_spent : ℝ) (candy_bar_cost : ℝ) : ℝ :=
  total_spent - candy_bar_cost

theorem chocolate_cost_proof (total_spent candy_bar_cost : ℝ) 
  (h1 : total_spent = 13)
  (h2 : candy_bar_cost = 7) :
  chocolate_cost total_spent candy_bar_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_proof_l2968_296836


namespace NUMINAMATH_CALUDE_investment_growth_l2968_296857

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem: Initial investment of $5000 at 10% p.a. for 2 years yields $6050.000000000001 -/
theorem investment_growth :
  let principal : ℝ := 5000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 6050.000000000001 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l2968_296857


namespace NUMINAMATH_CALUDE_tim_marbles_l2968_296884

/-- Given that Fred has 110 blue marbles and 22 times more blue marbles than Tim,
    prove that Tim has 5 blue marbles. -/
theorem tim_marbles (fred_marbles : ℕ) (ratio : ℕ) (h1 : fred_marbles = 110) (h2 : ratio = 22) :
  fred_marbles / ratio = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_marbles_l2968_296884


namespace NUMINAMATH_CALUDE_all_propositions_false_l2968_296861

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations between lines and planes
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular (p q : Plane) : Prop := sorry

-- Define the given lines and planes
variable (a b l : Line)
variable (α β γ : Plane)

-- Axioms for different objects
axiom different_lines : a ≠ b ∧ b ≠ l ∧ a ≠ l
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- The four propositions
def proposition1 : Prop := 
  ∀ a b α, parallel a b → contained_in b α → parallel a α

def proposition2 : Prop := 
  ∀ a b α, perpendicular a b → perpendicular b α → parallel a α

def proposition3 : Prop := 
  ∀ l α β, plane_perpendicular α β → contained_in l α → perpendicular l β

def proposition4 : Prop := 
  ∀ l a b α, perpendicular l a → perpendicular l b → 
    contained_in a α → contained_in b α → perpendicular l α

-- Theorem stating all propositions are false
theorem all_propositions_false : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2968_296861


namespace NUMINAMATH_CALUDE_noah_sales_this_month_l2968_296830

/-- Represents Noah's painting sales --/
structure NoahSales where
  large_price : ℕ
  small_price : ℕ
  last_month_large : ℕ
  last_month_small : ℕ

/-- Calculates Noah's sales for this month --/
def this_month_sales (s : NoahSales) : ℕ :=
  2 * (s.large_price * s.last_month_large + s.small_price * s.last_month_small)

/-- Theorem: Noah's sales for this month equal $1200 --/
theorem noah_sales_this_month (s : NoahSales) 
  (h1 : s.large_price = 60)
  (h2 : s.small_price = 30)
  (h3 : s.last_month_large = 8)
  (h4 : s.last_month_small = 4) :
  this_month_sales s = 1200 := by
  sorry

end NUMINAMATH_CALUDE_noah_sales_this_month_l2968_296830


namespace NUMINAMATH_CALUDE_car_train_distance_difference_l2968_296870

theorem car_train_distance_difference :
  let train_speed : ℝ := 60
  let car_speed : ℝ := 2 * train_speed
  let travel_time : ℝ := 3
  let train_distance : ℝ := train_speed * travel_time
  let car_distance : ℝ := car_speed * travel_time
  car_distance - train_distance = 180 := by
  sorry

end NUMINAMATH_CALUDE_car_train_distance_difference_l2968_296870


namespace NUMINAMATH_CALUDE_A_power_50_l2968_296806

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 1; -12, -3]

theorem A_power_50 : A^50 = !![301, 50; -900, -301] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l2968_296806


namespace NUMINAMATH_CALUDE_f_inequality_implies_m_bound_l2968_296858

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

theorem f_inequality_implies_m_bound :
  (∀ x : ℝ, f (2^x - 4^x) + f (m * 2^x - 3) < 0) →
  m < 2 * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_implies_m_bound_l2968_296858


namespace NUMINAMATH_CALUDE_haley_flash_drive_files_l2968_296874

/-- Calculates the number of files remaining on a flash drive after compression and deletion. -/
def files_remaining (music_files : ℕ) (video_files : ℕ) (document_files : ℕ) 
                    (music_compression : ℕ) (video_compression : ℕ) 
                    (deleted_files : ℕ) : ℕ :=
  music_files * music_compression + video_files * video_compression + document_files - deleted_files

/-- Theorem stating the number of files remaining on Haley's flash drive -/
theorem haley_flash_drive_files : 
  files_remaining 27 42 12 2 3 11 = 181 := by
  sorry

end NUMINAMATH_CALUDE_haley_flash_drive_files_l2968_296874


namespace NUMINAMATH_CALUDE_geometric_product_and_quotient_l2968_296876

/-- A sequence is geometric if the ratio of consecutive terms is constant. -/
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_product_and_quotient
  (a b : ℕ → ℝ)
  (ha : IsGeometric a)
  (hb : IsGeometric b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  IsGeometric (fun n ↦ a n * b n) ∧
  IsGeometric (fun n ↦ a n / b n) :=
sorry

end NUMINAMATH_CALUDE_geometric_product_and_quotient_l2968_296876


namespace NUMINAMATH_CALUDE_min_integer_value_of_fraction_l2968_296852

theorem min_integer_value_of_fraction (x : ℝ) : 
  ⌊(4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 3)⌋ ≥ -15 ∧ 
  ∃ y : ℝ, ⌊(4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 3)⌋ = -15 := by
  sorry

end NUMINAMATH_CALUDE_min_integer_value_of_fraction_l2968_296852


namespace NUMINAMATH_CALUDE_sum_equals_42_l2968_296842

/-- An increasing geometric sequence with specific properties -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  geometric : ∃ r : ℝ, r > 1 ∧ ∀ n, a (n + 1) = r * a n
  sum_condition : a 1 + a 3 + a 5 = 21
  a3_value : a 3 = 6

/-- The sum of specific terms in the sequence equals 42 -/
theorem sum_equals_42 (seq : IncreasingGeometricSequence) : seq.a 5 + seq.a 3 + seq.a 9 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_42_l2968_296842


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2968_296844

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_employees : ℕ
  num_groups : ℕ
  group_size : ℕ
  sample_interval : ℕ

/-- Calculates the number to be drawn from a specific group -/
def number_from_group (s : SystematicSampling) (group : ℕ) (position : ℕ) : ℕ :=
  (group - 1) * s.sample_interval + position

/-- The main theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.num_groups = 40)
  (h3 : s.group_size = 5)
  (h4 : s.sample_interval = 5)
  (h5 : number_from_group s 5 3 = 22) :
  number_from_group s 8 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2968_296844


namespace NUMINAMATH_CALUDE_plot_width_l2968_296838

/-- Given a rectangular plot with length 90 meters and perimeter that can be enclosed
    by 52 poles placed 5 meters apart, the width of the plot is 40 meters. -/
theorem plot_width (length : ℝ) (num_poles : ℕ) (pole_distance : ℝ) :
  length = 90 ∧ num_poles = 52 ∧ pole_distance = 5 →
  2 * (length + (num_poles * pole_distance / 2 - length) / 2) = num_poles * pole_distance →
  (num_poles * pole_distance / 2 - length) / 2 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_plot_width_l2968_296838


namespace NUMINAMATH_CALUDE_green_balls_removal_l2968_296845

theorem green_balls_removal (total : ℕ) (green_percent : ℚ) (target_percent : ℚ) 
  (h_total : total = 600)
  (h_green_percent : green_percent = 70/100)
  (h_target_percent : target_percent = 60/100) :
  ∃ x : ℕ, 
    (↑x ≤ green_percent * ↑total) ∧ 
    ((green_percent * ↑total - ↑x) / (↑total - ↑x) = target_percent) ∧
    x = 150 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_removal_l2968_296845


namespace NUMINAMATH_CALUDE_tetromino_properties_l2968_296816

/-- A tetromino is a shape made up of 4 squares. -/
structure Tetromino where
  squares : Finset (ℤ × ℤ)
  card_eq_four : squares.card = 4

/-- Two tetrominos are considered identical if they can be superimposed by rotating but not by flipping. -/
def are_identical (t1 t2 : Tetromino) : Prop := sorry

/-- The set of all distinct tetrominos. -/
def distinct_tetrominos : Finset Tetromino := sorry

/-- A 4 × 7 rectangle. -/
def rectangle : Finset (ℤ × ℤ) := sorry

/-- Tiling a rectangle with tetrominos. -/
def tiling (r : Finset (ℤ × ℤ)) (ts : Finset Tetromino) : Prop := sorry

theorem tetromino_properties :
  (distinct_tetrominos.card = 7) ∧
  ¬ (tiling rectangle distinct_tetrominos) := by sorry

end NUMINAMATH_CALUDE_tetromino_properties_l2968_296816


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2968_296890

theorem simplify_polynomial (x : ℝ) : (3*x)^4 + (3*x)*(x^3) + 2*x^5 = 84*x^4 + 2*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2968_296890


namespace NUMINAMATH_CALUDE_cosine_amplitude_l2968_296881

/-- Given a cosine function y = a cos(bx) where a > 0 and b > 0,
    prove that a equals the maximum y-value of the graph. -/
theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x)) →
  (∃ M, M > 0 ∧ ∀ x, a * Real.cos (b * x) ≤ M) →
  (∀ ε > 0, ∃ x, a * Real.cos (b * x) > M - ε) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l2968_296881


namespace NUMINAMATH_CALUDE_at_least_half_eligible_l2968_296889

/-- Represents a team of sailors --/
structure Team where
  heights : List ℝ
  nonempty : heights ≠ []

/-- The median of a list of real numbers --/
def median (l : List ℝ) : ℝ := sorry

/-- The count of elements in a list satisfying a predicate --/
def count_if (l : List ℝ) (p : ℝ → Bool) : ℕ := sorry

theorem at_least_half_eligible (t : Team) (h_median : median t.heights = 167) :
  2 * (count_if t.heights (λ x => x ≤ 168)) ≥ t.heights.length := by sorry

end NUMINAMATH_CALUDE_at_least_half_eligible_l2968_296889


namespace NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l2968_296819

-- Define a quadratic polynomial
def QuadraticPolynomial (α : Type*) [Field α] := α → α

-- Define the property of having exactly three solutions for (f(x))^3 - 4f(x) = 0
def HasThreeSolutionsCubicMinusFour (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, f x ^ 3 - 4 * f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define the property of having exactly two solutions for (f(x))^2 = 1
def HasTwoSolutionsSquaredEqualsOne (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∀ x : ℝ, f x ^ 2 = 1 ↔ x = y₁ ∨ x = y₂) ∧ y₁ ≠ y₂

-- The theorem statement
theorem quadratic_polynomial_solutions (f : QuadraticPolynomial ℝ) :
  HasThreeSolutionsCubicMinusFour f → HasTwoSolutionsSquaredEqualsOne f := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l2968_296819


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l2968_296808

theorem students_playing_neither_sport 
  (total : ℕ) 
  (football : ℕ) 
  (tennis : ℕ) 
  (both : ℕ) 
  (h1 : total = 60) 
  (h2 : football = 36) 
  (h3 : tennis = 30) 
  (h4 : both = 22) : 
  total - (football + tennis - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l2968_296808


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l2968_296888

theorem quadratic_equation_single_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + b₁ * x + 12 * x + 11 = 0 → (∀ y : ℝ, 3 * y^2 + b₁ * y + 12 * y + 11 = 0 → x = y)) ∧
  (∀ x : ℝ, 3 * x^2 + b₂ * x + 12 * x + 11 = 0 → (∀ y : ℝ, 3 * y^2 + b₂ * y + 12 * y + 11 = 0 → x = y)) ∧
  (∃ x : ℝ, 3 * x^2 + b₁ * x + 12 * x + 11 = 0) ∧
  (∃ x : ℝ, 3 * x^2 + b₂ * x + 12 * x + 11 = 0) ∧
  (b₁ ≠ b₂) →
  b₁ + b₂ = -24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l2968_296888


namespace NUMINAMATH_CALUDE_inverse_g_at_113_l2968_296891

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_g_at_113 : g⁻¹ 113 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_113_l2968_296891


namespace NUMINAMATH_CALUDE_modular_exponentiation_l2968_296892

theorem modular_exponentiation (m : ℕ) : 
  0 ≤ m ∧ m < 29 ∧ (4 * m) % 29 = 1 → (5^m)^4 % 29 - 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_modular_exponentiation_l2968_296892


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l2968_296800

theorem number_of_divisors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l2968_296800


namespace NUMINAMATH_CALUDE_triangle_sum_proof_l2968_296879

/-- Triangle operation: a + b - 2c --/
def triangle_op (a b c : ℤ) : ℤ := a + b - 2*c

theorem triangle_sum_proof :
  let t1 := triangle_op 3 4 5
  let t2 := triangle_op 6 8 2
  2 * t1 + 3 * t2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_proof_l2968_296879


namespace NUMINAMATH_CALUDE_highest_power_prime_factorial_l2968_296848

def highest_power_of_prime (p n : ℕ) : ℕ := sorry

def sum_of_floor_divisions (p n : ℕ) : ℕ := sorry

theorem highest_power_prime_factorial (p n : ℕ) (h_prime : Nat.Prime p) :
  ∃ k : ℕ, p ^ k ≤ n ∧ n < p ^ (k + 1) ∧
  highest_power_of_prime p n = sum_of_floor_divisions p n :=
sorry

end NUMINAMATH_CALUDE_highest_power_prime_factorial_l2968_296848


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l2968_296834

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2b cos C, then the triangle is isosceles with B = C -/
theorem triangle_isosceles_condition (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a = 2 * b * Real.cos C   -- Given condition
  → B = C := by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_condition_l2968_296834


namespace NUMINAMATH_CALUDE_ascending_order_proof_l2968_296818

theorem ascending_order_proof : 222^2 < 22^22 ∧ 22^22 < 2^222 := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_proof_l2968_296818


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2968_296860

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (a + Complex.I) / (1 + Complex.I)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2968_296860


namespace NUMINAMATH_CALUDE_function_nonnegative_iff_a_in_range_l2968_296856

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x + 2

-- Define the theorem
theorem function_nonnegative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ [-1, 1] → f a x ≥ 0) ↔ a ∈ [1, 5] := by sorry

end NUMINAMATH_CALUDE_function_nonnegative_iff_a_in_range_l2968_296856


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l2968_296829

/-- Represents an unfair eight-sided die -/
structure UnfairDie where
  prob_8 : ℚ
  prob_others : ℚ
  sum_to_one : prob_8 + 7 * prob_others = 1
  prob_8_is_3_8 : prob_8 = 3/8

/-- Expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.prob_8 * 8

/-- Theorem stating the expected value of the unfair die is 77/14 -/
theorem unfair_die_expected_value :
  ∀ (d : UnfairDie), expected_value d = 77/14 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l2968_296829


namespace NUMINAMATH_CALUDE_english_only_students_l2968_296805

theorem english_only_students (total : ℕ) (both : ℕ) (french : ℕ) (english : ℕ) : 
  total = 30 ∧ 
  both = 2 ∧ 
  english = 3 * french ∧ 
  total = french + english - both → 
  english - both = 20 := by
sorry

end NUMINAMATH_CALUDE_english_only_students_l2968_296805


namespace NUMINAMATH_CALUDE_max_product_bound_l2968_296820

theorem max_product_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_max_product_bound_l2968_296820


namespace NUMINAMATH_CALUDE_x_value_l2968_296887

theorem x_value : ∃ x : ℚ, (3 * x + 4) / 5 = 15 ∧ x = 71 / 3 := by sorry

end NUMINAMATH_CALUDE_x_value_l2968_296887


namespace NUMINAMATH_CALUDE_total_puppies_l2968_296801

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

theorem total_puppies : 
  puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4 = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_puppies_l2968_296801


namespace NUMINAMATH_CALUDE_jack_sugar_calculation_l2968_296811

/-- Given Jack's sugar operations, prove the final amount is correct. -/
theorem jack_sugar_calculation (initial : ℕ) (used : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 65 → used = 18 → bought = 50 → final = 97 → 
  final = initial - used + bought :=
by
  sorry

end NUMINAMATH_CALUDE_jack_sugar_calculation_l2968_296811


namespace NUMINAMATH_CALUDE_parabola_intersection_l2968_296825

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

theorem parabola_intersection :
  ∃ (x1 x2 : ℝ),
    x1 = (3 + Real.sqrt 129) / 4 ∧
    x2 = (3 - Real.sqrt 129) / 4 ∧
    parabola1 x1 = parabola2 x1 ∧
    parabola1 x2 = parabola2 x2 ∧
    ∀ (x : ℝ), parabola1 x = parabola2 x → x = x1 ∨ x = x2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2968_296825


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2968_296867

/-- Given a train of length 1200 m that crosses a tree in 120 seconds,
    prove that the time required for the train to pass a platform of length 400 m is 160 seconds. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1200
  let tree_crossing_time : ℝ := 120
  let platform_length : ℝ := 400
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  total_distance / train_speed = 160 := by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2968_296867


namespace NUMINAMATH_CALUDE_purple_position_correct_l2968_296899

/-- The position of "PURPLE" in the alphabetized list of all its distinguishable rearrangements -/
def purple_position : ℕ := 226

/-- The word to be rearranged -/
def word : String := "PURPLE"

/-- The theorem stating that the position of "PURPLE" in the alphabetized list of all its distinguishable rearrangements is 226 -/
theorem purple_position_correct : 
  purple_position = 226 ∧ 
  word = "PURPLE" ∧
  purple_position = (List.filter (· ≤ word) (List.map String.mk (List.permutations word.data))).length :=
by sorry

end NUMINAMATH_CALUDE_purple_position_correct_l2968_296899


namespace NUMINAMATH_CALUDE_set_relations_imply_a_and_m_ranges_l2968_296896

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 1 = 0}

-- State the theorem
theorem set_relations_imply_a_and_m_ranges :
  ∀ a m : ℝ,
  (A ∪ B a = A) →
  (A ∩ C m = C m) →
  ((a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_set_relations_imply_a_and_m_ranges_l2968_296896


namespace NUMINAMATH_CALUDE_zeros_order_l2968_296826

open Real

noncomputable def f (x : ℝ) := x + log x
noncomputable def g (x : ℝ) := x * log x - 1
noncomputable def h (x : ℝ) := 1 - 1/x + x/2 + x^2/3

theorem zeros_order (a b c : ℝ) 
  (ha : a > 0 ∧ f a = 0)
  (hb : b > 0 ∧ g b = 0)
  (hc : c > 0 ∧ h c = 0)
  (hf : ∀ x, x > 0 → x ≠ a → f x ≠ 0)
  (hg : ∀ x, x > 0 → x ≠ b → g x ≠ 0)
  (hh : ∀ x, x > 0 → x ≠ c → h x ≠ 0) :
  b > c ∧ c > a :=
sorry

end NUMINAMATH_CALUDE_zeros_order_l2968_296826


namespace NUMINAMATH_CALUDE_consecutive_binomial_coefficients_l2968_296877

theorem consecutive_binomial_coefficients (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 2 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 4 →
  n + k = 47 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_binomial_coefficients_l2968_296877


namespace NUMINAMATH_CALUDE_max_shaded_area_achievable_max_area_l2968_296898

/-- Represents a rectangular picture frame made of eight identical trapezoids -/
structure PictureFrame where
  length : ℕ+
  width : ℕ+
  trapezoidArea : ℕ
  isPrime : Nat.Prime trapezoidArea

/-- Calculates the area of the shaded region in the picture frame -/
def shadedArea (frame : PictureFrame) : ℕ :=
  (frame.trapezoidArea - 1) * (3 * frame.trapezoidArea - 1)

/-- Theorem stating the maximum possible area of the shaded region -/
theorem max_shaded_area (frame : PictureFrame) :
  shadedArea frame < 2000 → shadedArea frame ≤ 1496 :=
by
  sorry

/-- Theorem proving that 1496 is achievable -/
theorem achievable_max_area :
  ∃ frame : PictureFrame, shadedArea frame = 1496 ∧ shadedArea frame < 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_shaded_area_achievable_max_area_l2968_296898


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l2968_296850

-- Define the conditions
def condition_p (t : ℝ) : Prop := ∀ x : ℝ, (1/2) * x^2 - t*x + 1/2 > 0

def condition_q (t a : ℝ) : Prop := t^2 - (a-1)*t - a < 0

-- Theorem 1
theorem theorem_1 (t : ℝ) : condition_p t → -1 < t ∧ t < 1 := by sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) : 
  (∀ t : ℝ, condition_p t → condition_q t a) ∧ 
  (∃ t : ℝ, condition_q t a ∧ ¬condition_p t) → 
  a > 1 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l2968_296850


namespace NUMINAMATH_CALUDE_coefficient_x9_eq_240_l2968_296880

/-- The coefficient of x^9 in the expansion of (1+3x-2x^2)^5 -/
def coefficient_x9 : ℤ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^9 in (1+3x-2x^2)^5 is 240 -/
theorem coefficient_x9_eq_240 : coefficient_x9 = 240 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x9_eq_240_l2968_296880


namespace NUMINAMATH_CALUDE_rational_sqrt_two_equation_l2968_296804

theorem rational_sqrt_two_equation (x y : ℚ) (h : x + Real.sqrt 2 * y = 0) : x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_two_equation_l2968_296804


namespace NUMINAMATH_CALUDE_constant_function_integral_equals_one_l2968_296814

theorem constant_function_integral_equals_one : 
  ∫ x in (0 : ℝ)..1, (1 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_constant_function_integral_equals_one_l2968_296814


namespace NUMINAMATH_CALUDE_find_number_l2968_296854

theorem find_number : ∃ x : ℝ, 3 * (x + 8) = 36 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_find_number_l2968_296854


namespace NUMINAMATH_CALUDE_meeting_time_proof_l2968_296803

/-- 
Given two people traveling towards each other on a 600 km route, 
one at 70 km/hr and the other at 80 km/hr, prove that they meet 
after traveling for 4 hours.
-/
theorem meeting_time_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (t : ℝ) : 
  total_distance = 600 →
  speed1 = 70 →
  speed2 = 80 →
  speed1 * t + speed2 * t = total_distance →
  t = 4 := by
sorry

end NUMINAMATH_CALUDE_meeting_time_proof_l2968_296803


namespace NUMINAMATH_CALUDE_fred_savings_period_l2968_296895

/-- The number of weeks Fred needs to save to buy the mountain bike -/
def weeks_to_save (bike_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  (bike_cost - birthday_money) / weekly_earnings

theorem fred_savings_period :
  weeks_to_save 600 150 18 = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_savings_period_l2968_296895


namespace NUMINAMATH_CALUDE_share_face_value_l2968_296828

/-- Given shares with a certain dividend rate and market value, 
    calculate the face value that yields a desired interest rate. -/
theorem share_face_value 
  (dividend_rate : ℚ) 
  (desired_interest_rate : ℚ) 
  (market_value : ℚ) 
  (h1 : dividend_rate = 9 / 100)
  (h2 : desired_interest_rate = 12 / 100)
  (h3 : market_value = 45) : 
  ∃ (face_value : ℚ), 
    face_value * dividend_rate = market_value * desired_interest_rate ∧ 
    face_value = 60 := by
  sorry

#check share_face_value

end NUMINAMATH_CALUDE_share_face_value_l2968_296828


namespace NUMINAMATH_CALUDE_dishes_bananas_difference_is_ten_l2968_296807

/-- The number of pears Charles picked -/
def pears_picked : ℕ := 50

/-- The number of dishes Sandrine washed -/
def dishes_washed : ℕ := 160

/-- The number of bananas Charles cooked -/
def bananas_cooked : ℕ := 3 * pears_picked

/-- The difference between dishes washed and bananas cooked -/
def dishes_bananas_difference : ℕ := dishes_washed - bananas_cooked

theorem dishes_bananas_difference_is_ten :
  dishes_bananas_difference = 10 := by sorry

end NUMINAMATH_CALUDE_dishes_bananas_difference_is_ten_l2968_296807


namespace NUMINAMATH_CALUDE_initial_men_correct_l2968_296813

/-- The initial number of men working on a project -/
def initial_men : ℕ := 15

/-- The number of days to complete the work with the initial group -/
def initial_days : ℕ := 40

/-- The number of men who leave the project -/
def men_leaving : ℕ := 14

/-- The number of days worked before some men leave -/
def days_before_leaving : ℕ := 16

/-- The number of days to complete the remaining work after some men leave -/
def remaining_days : ℕ := 40

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct : 
  (initial_men : ℚ) * initial_days * (initial_days - days_before_leaving) = 
  (initial_men - men_leaving) * initial_days * remaining_days :=
sorry

end NUMINAMATH_CALUDE_initial_men_correct_l2968_296813


namespace NUMINAMATH_CALUDE_inequality_condition_l2968_296871

theorem inequality_condition (x y : ℝ) : (x > y ∧ 1 / x > 1 / y) ↔ x * y < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2968_296871


namespace NUMINAMATH_CALUDE_equation_equivalence_l2968_296821

theorem equation_equivalence (x y : ℝ) : 2 * y - 4 * x + 5 = 0 ↔ y = 2 * x - 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2968_296821


namespace NUMINAMATH_CALUDE_dot_product_implies_t_l2968_296815

/-- Given vectors a and b in R^2, if their dot product is -2, then the second component of b is -4 -/
theorem dot_product_implies_t (a b : Fin 2 → ℝ) (h : a 0 = 5 ∧ a 1 = -7 ∧ b 0 = -6) :
  (a 0 * b 0 + a 1 * b 1 = -2) → b 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_implies_t_l2968_296815
