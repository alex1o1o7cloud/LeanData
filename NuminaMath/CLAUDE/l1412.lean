import Mathlib

namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1412_141205

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) (min_per_box : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 20 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes
    with at least one ball in each box -/
theorem seven_balls_four_boxes : distribute_balls 7 4 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1412_141205


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l1412_141208

/-- The number of days considered in the weather forecast -/
def days : ℕ := 5

/-- The probability of no rain on a given day -/
def prob_no_rain : ℝ := 0.3

/-- The probability of 3 inches of rain on a given day -/
def prob_3_inches : ℝ := 0.5

/-- The probability of 8 inches of rain on a given day -/
def prob_8_inches : ℝ := 0.2

/-- The amount of rainfall (in inches) for the "no rain" scenario -/
def rain_0 : ℝ := 0

/-- The amount of rainfall (in inches) for the "3 inches" scenario -/
def rain_3 : ℝ := 3

/-- The amount of rainfall (in inches) for the "8 inches" scenario -/
def rain_8 : ℝ := 8

/-- The expected total rainfall over the given number of days -/
def expected_total_rainfall : ℝ := days * (prob_no_rain * rain_0 + prob_3_inches * rain_3 + prob_8_inches * rain_8)

theorem expected_rainfall_theorem : expected_total_rainfall = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_rainfall_theorem_l1412_141208


namespace NUMINAMATH_CALUDE_fraction_division_problem_solution_l1412_141275

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem problem_solution : (3 : ℚ) / 4 / ((2 : ℚ) / 5) = 15 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_division_problem_solution_l1412_141275


namespace NUMINAMATH_CALUDE_intersection_and_coefficients_l1412_141204

def A : Set ℝ := {x | x^2 < 9}
def B : Set ℝ := {x | (x-2)*(x+4) < 0}

theorem intersection_and_coefficients :
  (A ∩ B = {x | -3 < x ∧ x < 2}) ∧
  (∃ a b : ℝ, ∀ x : ℝ, (x ∈ A ∪ B) ↔ (2*x^2 + a*x + b < 0) ∧ a = 2 ∧ b = -24) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_coefficients_l1412_141204


namespace NUMINAMATH_CALUDE_triangle_angle_expression_minimum_l1412_141249

theorem triangle_angle_expression_minimum (A B C : ℝ) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) 
  (h_sum : A + B + C = π) : 
  (4 / A) + (1 / (B + C)) ≥ 9 / π ∧ 
  ∃ (A' B' C' : ℝ), A' > 0 ∧ B' > 0 ∧ C' > 0 ∧ A' + B' + C' = π ∧ 
    (4 / A') + (1 / (B' + C')) = 9 / π :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_expression_minimum_l1412_141249


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1412_141272

/-- Represents the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 1216 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1412_141272


namespace NUMINAMATH_CALUDE_man_speed_against_current_l1412_141257

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem,
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l1412_141257


namespace NUMINAMATH_CALUDE_unique_real_solution_and_two_imaginary_l1412_141202

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x - 1)^2
def equation2 (x y : ℂ) : Prop := x * y + y = 2

-- Define a solution pair
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

-- Define the set of all solution pairs
def solution_set : Set (ℂ × ℂ) := {p | is_solution p.1 p.2}

-- State the theorem
theorem unique_real_solution_and_two_imaginary :
  ∃! (x y : ℝ), is_solution x y ∧
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
    is_solution (x + a * I) (y + b * I) ∧
    is_solution (x + c * I) (y + d * I) ∧
    (x + a * I ≠ x + c * I) ∧
    (∀ (u v : ℂ), is_solution u v → (u = x ∧ v = y) ∨ 
                                    (u = x + a * I ∧ v = y + b * I) ∨ 
                                    (u = x + c * I ∧ v = y + d * I)) :=
by sorry

end NUMINAMATH_CALUDE_unique_real_solution_and_two_imaginary_l1412_141202


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1412_141284

theorem negative_square_cubed (m : ℝ) : (-m^2)^3 = -m^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1412_141284


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_eight_l1412_141203

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem inverse_sum_equals_negative_eight :
  ∃ (a b : ℝ), f a = 4 ∧ f b = -100 ∧ a + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_eight_l1412_141203


namespace NUMINAMATH_CALUDE_yellow_not_more_than_green_l1412_141285

/-- Represents the three types of parrots -/
inductive ParrotType
  | Green
  | Yellow
  | Mottled

/-- Represents whether a parrot tells the truth or lies -/
inductive ParrotResponse
  | Truth
  | Lie

/-- The total number of parrots -/
def totalParrots : Nat := 100

/-- The number of parrots that agreed with each statement -/
def agreeingParrots : Nat := 50

/-- Function that determines how a parrot responds based on its type -/
def parrotBehavior (t : ParrotType) (statement : Nat) : ParrotResponse :=
  match t with
  | ParrotType.Green => ParrotResponse.Truth
  | ParrotType.Yellow => ParrotResponse.Lie
  | ParrotType.Mottled => if statement == 1 then ParrotResponse.Truth else ParrotResponse.Lie

/-- Theorem stating that the number of yellow parrots cannot exceed the number of green parrots -/
theorem yellow_not_more_than_green 
  (G Y M : Nat) 
  (h_total : G + Y + M = totalParrots)
  (h_first_statement : G + M / 2 = agreeingParrots)
  (h_second_statement : M / 2 + Y = agreeingParrots) :
  Y ≤ G :=
sorry

end NUMINAMATH_CALUDE_yellow_not_more_than_green_l1412_141285


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1412_141252

theorem inequality_solution_set (t : ℝ) (a : ℝ) : 
  (∀ x : ℝ, (tx^2 - 6*x + t^2 < 0) ↔ (x < a ∨ x > 1)) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1412_141252


namespace NUMINAMATH_CALUDE_base_representation_of_200_l1412_141276

theorem base_representation_of_200 :
  ∃! b : ℕ, b > 1 ∧ b^5 ≤ 200 ∧ 200 < b^6 := by sorry

end NUMINAMATH_CALUDE_base_representation_of_200_l1412_141276


namespace NUMINAMATH_CALUDE_initial_marbles_calculation_l1412_141215

/-- The number of marbles Connie initially had -/
def initial_marbles : ℝ := 972.1

/-- The number of marbles Connie gave to Juan -/
def marbles_to_juan : ℝ := 183.5

/-- The number of marbles Connie gave to Maria -/
def marbles_to_maria : ℝ := 245.7

/-- The number of marbles Connie received from Mike -/
def marbles_from_mike : ℝ := 50.3

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.2

/-- Theorem stating that the initial number of marbles is equal to the sum of
    the current marbles, marbles given away, minus marbles received -/
theorem initial_marbles_calculation :
  initial_marbles = marbles_left + marbles_to_juan + marbles_to_maria - marbles_from_mike :=
by sorry

end NUMINAMATH_CALUDE_initial_marbles_calculation_l1412_141215


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l1412_141297

-- Define a pentagon
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

-- Define the theorem
theorem pentagon_angle_measure (PQRST : Pentagon) 
  (h1 : PQRST.P = PQRST.R ∧ PQRST.R = PQRST.T)  -- ∠P ≅ ∠R ≅ ∠T
  (h2 : PQRST.Q + PQRST.S = 180)  -- ∠Q is supplementary to ∠S
  (h3 : PQRST.P + PQRST.Q + PQRST.R + PQRST.S + PQRST.T = 540)  -- Sum of angles in a pentagon
  : PQRST.T = 120 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l1412_141297


namespace NUMINAMATH_CALUDE_hogwarts_total_students_l1412_141226

-- Define the given conditions
def total_participants : ℕ := 246
def total_boys : ℕ := 255

-- Define the relationship between participating boys and non-participating girls
def boys_participating_girls_not (total_students : ℕ) : Prop :=
  ∃ (boys_participating : ℕ) (girls_not_participating : ℕ),
    boys_participating = girls_not_participating + 11 ∧
    boys_participating ≤ total_boys ∧
    girls_not_participating ≤ total_students - total_boys

-- Theorem statement
theorem hogwarts_total_students : 
  ∃ (total_students : ℕ),
    total_students = 490 ∧
    boys_participating_girls_not total_students :=
by
  sorry


end NUMINAMATH_CALUDE_hogwarts_total_students_l1412_141226


namespace NUMINAMATH_CALUDE_dalmatians_with_right_ear_spot_l1412_141286

theorem dalmatians_with_right_ear_spot (total : ℕ) (left_only : ℕ) (right_only : ℕ) (no_spots : ℕ) :
  total = 101 →
  left_only = 29 →
  right_only = 17 →
  no_spots = 22 →
  total - no_spots - left_only = 50 :=
by sorry

end NUMINAMATH_CALUDE_dalmatians_with_right_ear_spot_l1412_141286


namespace NUMINAMATH_CALUDE_faucet_filling_time_l1412_141246

/-- Given that five faucets can fill a 150-gallon tub in 10 minutes,
    prove that ten faucets will fill a 50-gallon tub in 100 seconds. -/
theorem faucet_filling_time 
  (fill_rate : ℝ)  -- Rate at which one faucet fills in gallons per minute
  (h1 : 5 * fill_rate * 10 = 150)  -- Five faucets fill 150 gallons in 10 minutes
  : 10 * fill_rate * (100 / 60) = 50  -- Ten faucets fill 50 gallons in 100 seconds
  := by sorry

end NUMINAMATH_CALUDE_faucet_filling_time_l1412_141246


namespace NUMINAMATH_CALUDE_elaines_earnings_increase_l1412_141247

-- Define Elaine's earnings last year
variable (E : ℝ)

-- Define the percentage increase in earnings
variable (P : ℝ)

-- Theorem statement
theorem elaines_earnings_increase :
  -- Last year's rent spending
  (0.20 * E) > 0 →
  -- This year's rent spending is 143.75% of last year's
  (0.25 * (E * (1 + P / 100))) = (1.4375 * (0.20 * E)) →
  -- Conclusion: Earnings increased by 15%
  P = 15 := by
sorry

end NUMINAMATH_CALUDE_elaines_earnings_increase_l1412_141247


namespace NUMINAMATH_CALUDE_digit_sum_property_l1412_141224

/-- A function that checks if a natural number has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop := sorry

/-- A function that generates all digit permutations of a natural number -/
def digit_permutations (n : ℕ) : Finset ℕ := sorry

/-- A function that checks if a natural number is composed solely of ones -/
def all_ones (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has at least one digit 5 or greater -/
def has_digit_ge_5 (n : ℕ) : Prop := sorry

theorem digit_sum_property (n : ℕ) :
  has_no_zero_digits n →
  ∃ (p₁ p₂ p₃ : ℕ), p₁ ∈ digit_permutations n ∧ 
                    p₂ ∈ digit_permutations n ∧ 
                    p₃ ∈ digit_permutations n ∧
                    all_ones (n + p₁ + p₂ + p₃) →
  has_digit_ge_5 n :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l1412_141224


namespace NUMINAMATH_CALUDE_square_fraction_count_l1412_141245

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, ∃ m : ℤ, n / (25 - n) = m^2) ∧ 
    (∀ n : ℤ, n ∉ S → ¬∃ m : ℤ, n / (25 - n) = m^2) ∧ 
    S.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l1412_141245


namespace NUMINAMATH_CALUDE_alpha_square_greater_beta_square_l1412_141273

theorem alpha_square_greater_beta_square 
  (α β : ℝ) 
  (h1 : α ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h2 : β ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
sorry

end NUMINAMATH_CALUDE_alpha_square_greater_beta_square_l1412_141273


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_3087_l1412_141213

theorem smallest_prime_factor_of_3087 : Nat.minFac 3087 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_3087_l1412_141213


namespace NUMINAMATH_CALUDE_gwen_spent_eight_dollars_l1412_141295

/-- The amount of money Gwen received for her birthday. -/
def initial_amount : ℕ := 14

/-- The amount of money Gwen has left. -/
def remaining_amount : ℕ := 6

/-- The amount of money Gwen spent. -/
def spent_amount : ℕ := initial_amount - remaining_amount

theorem gwen_spent_eight_dollars : spent_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_gwen_spent_eight_dollars_l1412_141295


namespace NUMINAMATH_CALUDE_lawn_area_is_40_l1412_141281

/-- Represents a rectangular lawn with a path --/
structure LawnWithPath where
  length : ℝ
  width : ℝ
  pathWidth : ℝ

/-- Calculates the remaining lawn area after subtracting the path --/
def remainingLawnArea (lawn : LawnWithPath) : ℝ :=
  lawn.length * lawn.width - lawn.length * lawn.pathWidth

/-- Theorem stating that for the given dimensions, the remaining lawn area is 40 square meters --/
theorem lawn_area_is_40 (lawn : LawnWithPath) 
  (h1 : lawn.length = 10)
  (h2 : lawn.width = 5)
  (h3 : lawn.pathWidth = 1) : 
  remainingLawnArea lawn = 40 := by
  sorry

#check lawn_area_is_40

end NUMINAMATH_CALUDE_lawn_area_is_40_l1412_141281


namespace NUMINAMATH_CALUDE_bell_ringing_fraction_l1412_141238

theorem bell_ringing_fraction :
  let big_bell_rings : ℕ := 36
  let total_rings : ℕ := 52
  let small_bell_rings (f : ℚ) : ℚ := f * big_bell_rings + 4

  ∃ f : ℚ, f = 1/3 ∧ (↑big_bell_rings : ℚ) + small_bell_rings f = total_rings := by
  sorry

end NUMINAMATH_CALUDE_bell_ringing_fraction_l1412_141238


namespace NUMINAMATH_CALUDE_combined_travel_time_l1412_141254

/-- Given a car that takes 4.5 hours to reach station B, and a train that takes 2 hours longer
    than the car to cover the same distance, the combined time for both to reach station B
    is 11 hours. -/
theorem combined_travel_time (car_time train_time : ℝ) : 
  car_time = 4.5 →
  train_time = car_time + 2 →
  car_time + train_time = 11 := by
sorry

end NUMINAMATH_CALUDE_combined_travel_time_l1412_141254


namespace NUMINAMATH_CALUDE_distance_between_points_l1412_141260

/-- The distance between two points (-3, -4) and (5, 6) is 2√41 -/
theorem distance_between_points : 
  let a : ℝ × ℝ := (-3, -4)
  let b : ℝ × ℝ := (5, 6)
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2) = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1412_141260


namespace NUMINAMATH_CALUDE_isosceles_60_similar_l1412_141271

/-- An isosceles triangle with a 60° interior angle -/
structure IsoscelesTriangle60 where
  -- We represent the triangle by its three angles
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  -- The triangle is isosceles
  isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)
  -- One of the angles is 60°
  has60Degree : angle1 = 60 ∨ angle2 = 60 ∨ angle3 = 60
  -- The sum of angles in a triangle is 180°
  sumIs180 : angle1 + angle2 + angle3 = 180

/-- Two isosceles triangles with a 60° interior angle are similar -/
theorem isosceles_60_similar (t1 t2 : IsoscelesTriangle60) : 
  t1.angle1 = t2.angle1 ∧ t1.angle2 = t2.angle2 ∧ t1.angle3 = t2.angle3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_60_similar_l1412_141271


namespace NUMINAMATH_CALUDE_playground_count_l1412_141293

/-- The total number of people on the playground after late arrivals -/
def total_people (initial_boys initial_girls teachers late_boys late_girls : ℕ) : ℕ :=
  initial_boys + initial_girls + teachers + late_boys + late_girls

/-- Theorem stating the total number of people on the playground after late arrivals -/
theorem playground_count : total_people 44 53 5 3 2 = 107 := by
  sorry

end NUMINAMATH_CALUDE_playground_count_l1412_141293


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1412_141269

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let original_area := L * B
  let new_length := 0.8 * L
  let new_breadth := 0.8 * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1412_141269


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l1412_141219

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1_bounds : 1 < a 1 ∧ a 1 < 3)
  (h_a3 : a 3 = 4)
  (b : ℕ → ℝ)
  (h_b_def : ∀ n, b n = 2^(a n)) :
  (∃ r, ∀ n, b (n + 1) = r * b n) ∧
  (b 1 < b 2) ∧
  (b 2 > 4) ∧
  (b 2 * b 4 = 256) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l1412_141219


namespace NUMINAMATH_CALUDE_difference_largest_third_smallest_l1412_141282

def digits : List Nat := [1, 6, 8]

def largest_number : Nat := 861

def third_smallest_number : Nat := 618

theorem difference_largest_third_smallest :
  largest_number - third_smallest_number = 243 := by
  sorry

end NUMINAMATH_CALUDE_difference_largest_third_smallest_l1412_141282


namespace NUMINAMATH_CALUDE_nested_radical_value_l1412_141232

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (20 + x) ∧ x > 0 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1412_141232


namespace NUMINAMATH_CALUDE_divisibility_by_53_l1412_141218

theorem divisibility_by_53 (n : ℕ) : 53 ∣ (10^(n+3) + 17) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_53_l1412_141218


namespace NUMINAMATH_CALUDE_geometric_sequence_max_point_l1412_141277

/-- Given real numbers a, b, c, and d forming a geometric sequence,
    and (b, c) being the coordinates of the maximum point of the curve y = 3x - x^3,
    prove that ad = 2. -/
theorem geometric_sequence_max_point (a b c d : ℝ) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, 3 * b - b^3 ≥ 3 * x - x^3) →  -- maximum point condition
  c = 3 * b - b^3 →  -- y-coordinate of maximum point
  a * d = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_point_l1412_141277


namespace NUMINAMATH_CALUDE_right_triangle_legs_product_divisible_by_12_l1412_141229

theorem right_triangle_legs_product_divisible_by_12 
  (a b c : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  12 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_product_divisible_by_12_l1412_141229


namespace NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l1412_141212

theorem sum_of_19th_powers_zero (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_zero : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l1412_141212


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l1412_141200

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l1412_141200


namespace NUMINAMATH_CALUDE_wall_height_proof_l1412_141274

/-- The height of a wall built with a specific number of bricks of given dimensions. -/
theorem wall_height_proof (brick_length brick_width brick_height : ℝ)
  (wall_length wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.08 →
  wall_length = 10 →
  wall_width = 24.5 →
  num_bricks = 12250 →
  ∃ (h : ℝ), h = 0.08 ∧ num_bricks * (brick_length * brick_width * brick_height) = wall_length * h * wall_width :=
by sorry

end NUMINAMATH_CALUDE_wall_height_proof_l1412_141274


namespace NUMINAMATH_CALUDE_package_servings_l1412_141227

/-- The number of servings in a package of candy. -/
def servings_in_package (calories_per_serving : ℕ) (calories_in_half : ℕ) : ℕ :=
  (2 * calories_in_half) / calories_per_serving

/-- Theorem: Given a package where each serving has 120 calories and half the package contains 180 calories, 
    prove that there are 3 servings in the package. -/
theorem package_servings : servings_in_package 120 180 = 3 := by
  sorry

end NUMINAMATH_CALUDE_package_servings_l1412_141227


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1412_141250

theorem inequality_implies_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 4^x - 2^(x+1) - a ≤ 0) → a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1412_141250


namespace NUMINAMATH_CALUDE_zoo_field_trip_remaining_individuals_l1412_141251

/-- Represents the number of individuals from a school -/
structure SchoolGroup :=
  (students : ℕ)
  (parents : ℕ)
  (teachers : ℕ)

/-- Calculates the total number of individuals in a school group -/
def SchoolGroup.total (sg : SchoolGroup) : ℕ :=
  sg.students + sg.parents + sg.teachers

theorem zoo_field_trip_remaining_individuals
  (school_a : SchoolGroup)
  (school_b : SchoolGroup)
  (school_c : SchoolGroup)
  (school_d : SchoolGroup)
  (h1 : school_a = ⟨10, 5, 2⟩)
  (h2 : school_b = ⟨12, 3, 2⟩)
  (h3 : school_c = ⟨15, 3, 0⟩)
  (h4 : school_d = ⟨20, 4, 0⟩)
  (left_students_ab : ℕ)
  (left_students_c : ℕ)
  (left_students_d : ℕ)
  (left_parents_a : ℕ)
  (left_parents_c : ℕ)
  (h5 : left_students_ab = 10)
  (h6 : left_students_c = 6)
  (h7 : left_students_d = 9)
  (h8 : left_parents_a = 2)
  (h9 : left_parents_c = 1)
  : (school_a.total + school_b.total + school_c.total + school_d.total) -
    (left_students_ab + left_students_c + left_students_d + left_parents_a + left_parents_c) = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_zoo_field_trip_remaining_individuals_l1412_141251


namespace NUMINAMATH_CALUDE_solution_set_satisfies_inequalities_l1412_141233

def S : Set ℝ := {x | 0 < x ∧ x < 1}

theorem solution_set_satisfies_inequalities :
  ∀ x ∈ S, x * (x + 2) > 0 ∧ |x| < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_satisfies_inequalities_l1412_141233


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1412_141221

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem circle_intersection_theorem :
  ∃ (x₁ y₁ x₂ y₂ m : ℝ),
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ →
    m = 8/5 ∧
    ∀ (x y : ℝ), (x - 4/5)^2 + (y - 8/5)^2 = 16/5 ↔
      circle_equation x y (8/5) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1412_141221


namespace NUMINAMATH_CALUDE_triangle_area_l1412_141231

theorem triangle_area (P Q R : ℝ) (r R : ℝ) (h1 : r = 3) (h2 : R = 15) 
  (h3 : 2 * Real.cos Q = Real.cos P + Real.cos R) : 
  ∃ (area : ℝ), area = 27 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1412_141231


namespace NUMINAMATH_CALUDE_fill_measuring_cup_l1412_141236

/-- The capacity of a spoon in milliliters -/
def spoon_capacity : ℝ := 5

/-- The volume of a measuring cup in liters -/
def cup_volume : ℝ := 1

/-- The conversion factor from liters to milliliters -/
def liter_to_ml : ℝ := 1000

/-- The number of spoons needed to fill the measuring cup -/
def spoons_needed : ℕ := 200

theorem fill_measuring_cup : 
  ⌊(cup_volume * liter_to_ml) / spoon_capacity⌋ = spoons_needed := by
  sorry

end NUMINAMATH_CALUDE_fill_measuring_cup_l1412_141236


namespace NUMINAMATH_CALUDE_harolds_remaining_money_l1412_141209

/-- Represents Harold's financial situation and calculates his remaining money --/
def harolds_finances (income rent car_payment groceries : ℚ) : ℚ := 
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement_savings := remaining / 2
  remaining - retirement_savings

/-- Theorem stating that Harold will have $650.00 left after expenses and retirement savings --/
theorem harolds_remaining_money :
  harolds_finances 2500 700 300 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_harolds_remaining_money_l1412_141209


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1412_141241

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ+) : ℚ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ+) : ℚ := n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

-- Define b_n
def b (n : ℕ+) : ℚ := 1 / (arithmetic_sequence (n + 1) * arithmetic_sequence (n + 2))

-- Define T_n
def T (n : ℕ+) : ℚ := (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

-- Theorem statement
theorem arithmetic_sequence_properties :
  (arithmetic_sequence 1 + arithmetic_sequence 13 = 26) ∧
  (S 9 = 81) →
  (∀ n : ℕ+, arithmetic_sequence n = 2 * n - 1) ∧
  (∀ n : ℕ+, T n = n / (3 * (2 * n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1412_141241


namespace NUMINAMATH_CALUDE_salary_spending_percentage_l1412_141294

theorem salary_spending_percentage 
  (total_salary : ℝ) 
  (a_salary : ℝ) 
  (b_spending_rate : ℝ) 
  (h1 : total_salary = 5000)
  (h2 : a_salary = 3750)
  (h3 : b_spending_rate = 0.85)
  (h4 : a_salary * (1 - a_spending_rate) = (total_salary - a_salary) * (1 - b_spending_rate)) :
  a_spending_rate = 0.95 := by
sorry

end NUMINAMATH_CALUDE_salary_spending_percentage_l1412_141294


namespace NUMINAMATH_CALUDE_cube_split_with_39_l1412_141210

/-- Given a natural number m > 1, if m³ can be split into a sum of consecutive odd numbers 
    starting from (m+1)² and one of these odd numbers is 39, then m = 6 -/
theorem cube_split_with_39 (m : ℕ) (h1 : m > 1) :
  (∃ k : ℕ, (m + 1)^2 + 2*k = 39) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_with_39_l1412_141210


namespace NUMINAMATH_CALUDE_time_to_get_ahead_l1412_141211

/-- Proves that the time for a faster traveler to get 1/3 mile ahead of a slower traveler is 2 minutes -/
theorem time_to_get_ahead (man_speed woman_speed : ℝ) (catch_up_time : ℝ) : 
  man_speed = 5 →
  woman_speed = 15 →
  catch_up_time = 4 →
  (woman_speed - man_speed) * 2 / 60 = 1 / 3 :=
by
  sorry

#check time_to_get_ahead

end NUMINAMATH_CALUDE_time_to_get_ahead_l1412_141211


namespace NUMINAMATH_CALUDE_star_example_l1412_141216

/-- The ⬥ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + 2*y) * (x - y)

/-- Theorem stating that 5 ⬥ (2 ⬥ 3) = -143 -/
theorem star_example : star 5 (star 2 3) = -143 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l1412_141216


namespace NUMINAMATH_CALUDE_volleyball_ticket_sales_l1412_141279

theorem volleyball_ticket_sales (total_tickets : ℕ) (jude_tickets : ℕ) (left_tickets : ℕ) 
  (h1 : total_tickets = 100)
  (h2 : jude_tickets = 16)
  (h3 : left_tickets = 40)
  : total_tickets - left_tickets - 2 * jude_tickets - jude_tickets = jude_tickets - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_ticket_sales_l1412_141279


namespace NUMINAMATH_CALUDE_inequality_range_l1412_141290

theorem inequality_range (a : ℝ) : 
  (∃ x : ℝ, x ≥ 1 ∧ (1 + 1/x)^(x + a) ≥ Real.exp 1) → 
  a ≥ 1/Real.log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1412_141290


namespace NUMINAMATH_CALUDE_odd_function_negative_l1412_141278

/-- An odd function f with a specific definition for non-negative x -/
def odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x ≥ 0, f x = x * (1 - x))

/-- Theorem stating the form of f(x) for non-positive x -/
theorem odd_function_negative (f : ℝ → ℝ) (h : odd_function f) :
  ∀ x ≤ 0, f x = x * (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_negative_l1412_141278


namespace NUMINAMATH_CALUDE_square_root_equation_l1412_141288

theorem square_root_equation (n : ℝ) : 
  Real.sqrt (8 + n) = 9 → n = 73 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_l1412_141288


namespace NUMINAMATH_CALUDE_unique_valid_assignment_l1412_141206

/-- Represents the possible arithmetic operations --/
inductive Operation
  | Plus
  | Minus
  | Multiply
  | Divide
  | Equal

/-- Represents the assignment of operations to letters --/
structure Assignment :=
  (A B C D E : Operation)

/-- Checks if an assignment is valid according to the problem conditions --/
def is_valid_assignment (a : Assignment) : Prop :=
  a.A ≠ a.B ∧ a.A ≠ a.C ∧ a.A ≠ a.D ∧ a.A ≠ a.E ∧
  a.B ≠ a.C ∧ a.B ≠ a.D ∧ a.B ≠ a.E ∧
  a.C ≠ a.D ∧ a.C ≠ a.E ∧
  a.D ≠ a.E ∧
  (a.A = Operation.Plus ∨ a.B = Operation.Plus ∨ a.C = Operation.Plus ∨ a.D = Operation.Plus ∨ a.E = Operation.Plus) ∧
  (a.A = Operation.Minus ∨ a.B = Operation.Minus ∨ a.C = Operation.Minus ∨ a.D = Operation.Minus ∨ a.E = Operation.Minus) ∧
  (a.A = Operation.Multiply ∨ a.B = Operation.Multiply ∨ a.C = Operation.Multiply ∨ a.D = Operation.Multiply ∨ a.E = Operation.Multiply) ∧
  (a.A = Operation.Divide ∨ a.B = Operation.Divide ∨ a.C = Operation.Divide ∨ a.D = Operation.Divide ∨ a.E = Operation.Divide) ∧
  (a.A = Operation.Equal ∨ a.B = Operation.Equal ∨ a.C = Operation.Equal ∨ a.D = Operation.Equal ∨ a.E = Operation.Equal)

/-- Checks if an assignment satisfies the equations --/
def satisfies_equations (a : Assignment) : Prop :=
  (a.A = Operation.Divide ∧ 4 / 2 = 2) ∧
  (a.B = Operation.Equal) ∧
  (a.C = Operation.Multiply ∧ 4 * 2 = 8) ∧
  (a.D = Operation.Plus ∧ 2 + 3 = 5) ∧
  (a.E = Operation.Minus ∧ 5 - 1 = 4)

/-- The main theorem: there is a unique valid assignment that satisfies the equations --/
theorem unique_valid_assignment :
  ∃! (a : Assignment), is_valid_assignment a ∧ satisfies_equations a :=
sorry

end NUMINAMATH_CALUDE_unique_valid_assignment_l1412_141206


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l1412_141237

theorem cycle_gain_percent (cost_price selling_price : ℚ) (h1 : cost_price = 900) (h2 : selling_price = 1100) :
  (selling_price - cost_price) / cost_price * 100 = (2 : ℚ) / 9 * 100 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l1412_141237


namespace NUMINAMATH_CALUDE_triangle_count_specific_l1412_141291

/-- The number of triangles formed by points on two sides of a triangle -/
def triangles_from_points (n m : ℕ) : ℕ :=
  Nat.choose (n + m + 1) 3 - Nat.choose (n + 1) 3 - Nat.choose (m + 1) 3

/-- Theorem: The number of triangles formed by 5 points on one side,
    6 points on another side, and 1 shared vertex is 165 -/
theorem triangle_count_specific : triangles_from_points 5 6 = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_specific_l1412_141291


namespace NUMINAMATH_CALUDE_b_25_mod_55_l1412_141253

/-- Definition of b_n as a function that concatenates integers from 5 to n+4 -/
def b (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that b_25 mod 55 = 39 -/
theorem b_25_mod_55 : b 25 % 55 = 39 := by
  sorry

end NUMINAMATH_CALUDE_b_25_mod_55_l1412_141253


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1412_141220

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_inverse : ∃ k : ℝ, ∀ x y, x * y = k) 
    (h_nonzero_x : x₁ ≠ 0 ∧ x₂ ≠ 0) (h_nonzero_y : y₁ ≠ 0 ∧ y₂ ≠ 0) (h_ratio_x : x₁ / x₂ = 4 / 5) 
    (h_correspond : x₁ * y₁ = x₂ * y₂) : y₁ / y₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1412_141220


namespace NUMINAMATH_CALUDE_circle_radius_implies_a_value_l1412_141289

theorem circle_radius_implies_a_value (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*a*x + 4*a*y = 0) → 
  (∃ (c_x c_y : ℝ), ∀ (x y : ℝ), (x - c_x)^2 + (y - c_y)^2 = 5) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_implies_a_value_l1412_141289


namespace NUMINAMATH_CALUDE_keith_pears_count_l1412_141267

def total_pears : Nat := 5
def jason_pears : Nat := 2

theorem keith_pears_count : total_pears - jason_pears = 3 := by
  sorry

end NUMINAMATH_CALUDE_keith_pears_count_l1412_141267


namespace NUMINAMATH_CALUDE_simplify_fraction_l1412_141268

theorem simplify_fraction : (2^6 + 2^4) / (2^5 - 2^2) = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1412_141268


namespace NUMINAMATH_CALUDE_total_age_problem_l1412_141228

theorem total_age_problem (a b c : ℕ) : 
  b = 10 →
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_total_age_problem_l1412_141228


namespace NUMINAMATH_CALUDE_parabola_shift_l1412_141270

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 7

/-- The reference parabola function -/
def g (x : ℝ) : ℝ := x^2

/-- The shifted reference parabola function -/
def h (x : ℝ) : ℝ := g (x + 3) - 2

theorem parabola_shift :
  ∀ x : ℝ, f x = h x :=
sorry

end NUMINAMATH_CALUDE_parabola_shift_l1412_141270


namespace NUMINAMATH_CALUDE_unique_number_l1412_141298

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000) ∧ (n < 10000) ∧
  (n % 10 = (n / 100) % 10) ∧
  (n - (n % 10 * 1000 + (n / 10) % 10 * 100 + (n / 100) % 10 * 10 + n / 1000) = 7812)

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 1979 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l1412_141298


namespace NUMINAMATH_CALUDE_distance_to_xy_plane_l1412_141259

/-- The distance from a point (3, 2, -5) to the xy-plane is 5. -/
theorem distance_to_xy_plane : 
  let p : ℝ × ℝ × ℝ := (3, 2, -5)
  abs (p.2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_xy_plane_l1412_141259


namespace NUMINAMATH_CALUDE_circle_properties_l1412_141234

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equations
def line1_equation (x y : ℝ) : Prop :=
  3*x + 4*y - 6 = 0

def line2_equation (x y : ℝ) : Prop :=
  x - y - 1 = 0

-- Define the theorem
theorem circle_properties :
  -- Part 1: The circle exists when m < 5
  (∀ m : ℝ, m < 5 → ∃ x y : ℝ, circle_equation x y m) ∧
  -- Part 2: When the circle intersects line1 at M and N with |MN| = 2√3, m = 1
  (∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line1_equation x₁ y₁ ∧
    line1_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12 ∧
    m = 1) ∧
  -- Part 3: When the circle intersects line2 at A and B, there exists m = -2
  -- such that the circle with diameter AB passes through the origin
  (∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line2_equation x₁ y₁ ∧
    line2_equation x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    m = -2) := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l1412_141234


namespace NUMINAMATH_CALUDE_subset_empty_range_superset_range_l1412_141242

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

-- Theorem for part I
theorem subset_empty_range : ¬∃ a : ℝ, M ⊆ N a := by sorry

-- Theorem for part II
theorem superset_range : {a : ℝ | M ⊇ N a} = {a : ℝ | a ≤ 3} := by sorry

end NUMINAMATH_CALUDE_subset_empty_range_superset_range_l1412_141242


namespace NUMINAMATH_CALUDE_min_value_of_f_l1412_141248

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 6*x - 8

-- Theorem stating that f(x) achieves its minimum when x = -3
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1412_141248


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l1412_141255

theorem shirt_price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.8 * first_sale_price
  final_price / original_price = 0.64 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l1412_141255


namespace NUMINAMATH_CALUDE_festival_attendance_l1412_141261

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1500 →
  festival_attendees = 800 →
  girls + boys = total_students →
  (3 * girls) / 5 + (2 * boys) / 5 = festival_attendees →
  (3 * girls) / 5 = 600 :=
by sorry

end NUMINAMATH_CALUDE_festival_attendance_l1412_141261


namespace NUMINAMATH_CALUDE_jake_snake_revenue_l1412_141263

/-- Calculates the total revenue from selling baby snakes --/
def total_revenue (num_snakes : ℕ) (eggs_per_snake : ℕ) (regular_price : ℕ) (rare_price_multiplier : ℕ) : ℕ :=
  let total_babies := num_snakes * eggs_per_snake
  let num_regular_babies := total_babies - 1
  let rare_price := regular_price * rare_price_multiplier
  num_regular_babies * regular_price + rare_price

/-- The revenue from Jake's snake business --/
theorem jake_snake_revenue :
  total_revenue 3 2 250 4 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_jake_snake_revenue_l1412_141263


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l1412_141258

theorem positive_integer_solutions_count :
  let n : ℕ := 30
  let k : ℕ := 3
  (Nat.choose (n - 1) (k - 1) : ℕ) = 406 := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l1412_141258


namespace NUMINAMATH_CALUDE_lcm_of_210_and_913_l1412_141230

theorem lcm_of_210_and_913 :
  let a : ℕ := 210
  let b : ℕ := 913
  let hcf : ℕ := 83
  Nat.lcm a b = 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_210_and_913_l1412_141230


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1412_141240

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, 8 - 5*x > 25 → x ≤ -4 ∧ 8 - 5*(-4) > 25 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1412_141240


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l1412_141243

theorem quiz_competition_participants :
  let initial_participants : ℕ := 300
  let first_round_ratio : ℚ := 2/5
  let second_round_ratio : ℚ := 1/4
  let final_participants : ℕ := 30
  (initial_participants : ℚ) * first_round_ratio * second_round_ratio = final_participants := by
sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l1412_141243


namespace NUMINAMATH_CALUDE_train_speed_l1412_141239

/-- Calculate the speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 500) (h2 : time = 20) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1412_141239


namespace NUMINAMATH_CALUDE_f_properties_l1412_141265

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem f_properties :
  (∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f 2 ∧ deriv f 2 = f' 2 ∧ f' 2 < 0) ∧
  (∃ (x_max : ℝ), x_max = 1 ∧ ∀ x, f x ≤ f x_max ∧ f x_max = Real.exp 1) ∧
  (∀ x > 1, ∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f x ∧ deriv f x = f' x ∧ f' x < 0) ∧
  (∀ a : ℝ, (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) → 0 < a ∧ a < Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1412_141265


namespace NUMINAMATH_CALUDE_henrys_game_purchase_l1412_141266

/-- Henry's money problem -/
theorem henrys_game_purchase (initial : ℕ) (birthday_gift : ℕ) (final : ℕ) 
  (h1 : initial = 11)
  (h2 : birthday_gift = 18)
  (h3 : final = 19) :
  initial + birthday_gift - final = 10 := by
  sorry

end NUMINAMATH_CALUDE_henrys_game_purchase_l1412_141266


namespace NUMINAMATH_CALUDE_ratio_sum_equation_solver_l1412_141264

theorem ratio_sum_equation_solver (x y z a : ℚ) : 
  (∃ k : ℚ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * a - 5 →
  x + y + z = 70 →
  a = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equation_solver_l1412_141264


namespace NUMINAMATH_CALUDE_cube_zero_of_fourth_power_zero_l1412_141280

theorem cube_zero_of_fourth_power_zero (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : A ^ 3 = 0 := by sorry

end NUMINAMATH_CALUDE_cube_zero_of_fourth_power_zero_l1412_141280


namespace NUMINAMATH_CALUDE_exam_results_l1412_141235

theorem exam_results (total_students : ℕ) (second_division_percent : ℚ) 
  (just_passed : ℕ) (h1 : total_students = 300) 
  (h2 : second_division_percent = 54/100) (h3 : just_passed = 57) : 
  (1 - second_division_percent - (just_passed : ℚ) / total_students) = 27/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l1412_141235


namespace NUMINAMATH_CALUDE_existence_of_special_n_l1412_141207

theorem existence_of_special_n (t : ℕ) : ∃ n : ℕ, n > 1 ∧ 
  (Nat.gcd n t = 1) ∧ 
  (∀ k x m : ℕ, k ≥ 1 → m > 1 → n^k + t ≠ x^m) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_n_l1412_141207


namespace NUMINAMATH_CALUDE_heptagonal_prism_faces_and_vertices_l1412_141292

/-- A heptagonal prism is a three-dimensional shape with two heptagonal bases and rectangular lateral faces. -/
structure HeptagonalPrism where
  baseFaces : Nat
  lateralFaces : Nat
  baseVertices : Nat

/-- Properties of a heptagonal prism -/
def heptagonalPrismProperties : HeptagonalPrism where
  baseFaces := 2
  lateralFaces := 7
  baseVertices := 7

/-- Theorem: A heptagonal prism has 9 faces and 14 vertices -/
theorem heptagonal_prism_faces_and_vertices :
  let h := heptagonalPrismProperties
  (h.baseFaces + h.lateralFaces = 9) ∧ (h.baseVertices * h.baseFaces = 14) := by
  sorry

end NUMINAMATH_CALUDE_heptagonal_prism_faces_and_vertices_l1412_141292


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l1412_141217

def M (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_sum_theorem (a b c d : ℝ) :
  ¬(IsUnit (M a b c d).det) →
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c) = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l1412_141217


namespace NUMINAMATH_CALUDE_colored_paper_usage_l1412_141225

theorem colored_paper_usage (initial_sheets : ℕ) (sheets_used : ℕ) : 
  initial_sheets = 82 →
  initial_sheets - sheets_used = sheets_used - 6 →
  sheets_used = 44 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_usage_l1412_141225


namespace NUMINAMATH_CALUDE_black_cells_remain_even_one_black_cell_impossible_l1412_141201

/-- Represents a chessboard -/
structure Chessboard :=
  (black_cells : ℕ)

/-- Represents a repainting operation on a 2x2 square -/
def repaint (board : Chessboard) : Chessboard :=
  { black_cells := board.black_cells + (4 - 2 * (board.black_cells % 4)) }

/-- Initial chessboard state -/
def initial_board : Chessboard :=
  { black_cells := 32 }

/-- Theorem stating that the number of black cells remains even after any number of repainting operations -/
theorem black_cells_remain_even (n : ℕ) :
  ∀ (board : Chessboard),
  (board.black_cells % 2 = 0) →
  ((repaint^[n] board).black_cells % 2 = 0) :=
sorry

/-- Main theorem: It's impossible to have exactly one black cell after repainting operations -/
theorem one_black_cell_impossible :
  ¬ ∃ (n : ℕ), (repaint^[n] initial_board).black_cells = 1 :=
sorry

end NUMINAMATH_CALUDE_black_cells_remain_even_one_black_cell_impossible_l1412_141201


namespace NUMINAMATH_CALUDE_complex_roots_unity_l1412_141256

theorem complex_roots_unity (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs z₃ = 1)
  (h4 : z₁ + z₂ + z₃ = 1) 
  (h5 : z₁ * z₂ * z₃ = 1) :
  ({z₁, z₂, z₃} : Finset ℂ) = {1, Complex.I, -Complex.I} := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_unity_l1412_141256


namespace NUMINAMATH_CALUDE_polygon_150_diagonals_l1412_141283

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_150_diagonals :
  (num_diagonals 150 = 11025) ∧
  (9900 ≠ num_diagonals 150 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polygon_150_diagonals_l1412_141283


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1412_141299

theorem bernoulli_inequality (x r : ℝ) (hx : x > 0) (hr : r > 1) :
  (1 + x)^r > 1 + r * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1412_141299


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l1412_141244

theorem solution_exists_in_interval : ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x + x = 2 := by
  sorry


end NUMINAMATH_CALUDE_solution_exists_in_interval_l1412_141244


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l1412_141287

theorem cos_2alpha_plus_3pi_over_5 (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) :
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l1412_141287


namespace NUMINAMATH_CALUDE_base7_25_to_binary_l1412_141223

/-- Converts a number from base 7 to base 10 -/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 2 -/
def decimalToBinary (n : ℕ) : List ℕ := sorry

theorem base7_25_to_binary :
  decimalToBinary (base7ToDecimal 25) = [1, 0, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base7_25_to_binary_l1412_141223


namespace NUMINAMATH_CALUDE_money_problem_l1412_141262

theorem money_problem (a b : ℚ) (h1 : 7 * a + b = 89) (h2 : 4 * a - b = 38) :
  a = 127 / 11 ∧ b = 90 / 11 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1412_141262


namespace NUMINAMATH_CALUDE_scale_division_l1412_141296

theorem scale_division (total_length : ℝ) (num_parts : ℕ) (part_length : ℝ) : 
  total_length = 90 → num_parts = 5 → part_length * num_parts = total_length → part_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l1412_141296


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1412_141214

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sixth term of a geometric sequence satisfying given conditions. -/
theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : IsGeometric a)
  (h_sum : a 1 + a 2 = -1)
  (h_diff : a 1 - a 3 = -3) :
  a 6 = -32 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1412_141214


namespace NUMINAMATH_CALUDE_system_solution_l1412_141222

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 5) (h2 : x + 2 * y = 6) : x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1412_141222
