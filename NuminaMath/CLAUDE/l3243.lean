import Mathlib

namespace NUMINAMATH_CALUDE_pizza_fraction_l3243_324352

theorem pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) :
  total_slices = 12 →
  whole_slice = 1 →
  shared_slice = 1 / 2 →
  (whole_slice + shared_slice) / total_slices = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l3243_324352


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_square_midpoint_triangle_l3243_324357

/-- Given a square with side length 12, this theorem proves that the radius of the circle
    inscribed in the triangle formed by connecting midpoints of adjacent sides to each other
    and to the opposite side is 2√5 - √2. -/
theorem inscribed_circle_radius_in_square_midpoint_triangle :
  let square_side : ℝ := 12
  let midpoint_triangle_area : ℝ := 54
  let midpoint_triangle_semiperimeter : ℝ := 6 * Real.sqrt 5 + 3 * Real.sqrt 2
  let inscribed_circle_radius : ℝ := midpoint_triangle_area / midpoint_triangle_semiperimeter
  inscribed_circle_radius = 2 * Real.sqrt 5 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_in_square_midpoint_triangle_l3243_324357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3243_324337

/-- An arithmetic sequence with given first term, second term, and last term -/
structure ArithmeticSequence where
  first_term : ℕ
  second_term : ℕ
  last_term : ℕ

/-- The number of terms in an arithmetic sequence -/
def num_terms (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℕ :=
  seq.second_term - seq.first_term

theorem arithmetic_sequence_length :
  let seq := ArithmeticSequence.mk 13 19 127
  num_terms seq = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3243_324337


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3243_324310

theorem quadratic_equation_properties (m : ℝ) :
  let equation := fun x => x^2 - 2*m*x + m^2 - 4*m - 1
  (∃ x : ℝ, equation x = 0) ↔ m ≥ -1/4
  ∧
  equation 1 = 0 → m = 0 ∨ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3243_324310


namespace NUMINAMATH_CALUDE_family_reunion_food_l3243_324344

/-- The amount of food Peter buys for the family reunion -/
def total_food (chicken hamburger hotdog side : ℝ) : ℝ :=
  chicken + hamburger + hotdog + side

theorem family_reunion_food : ∃ (chicken hamburger hotdog side : ℝ),
  chicken = 16 ∧
  hamburger = chicken / 2 ∧
  hotdog = hamburger + 2 ∧
  side = hotdog / 2 ∧
  total_food chicken hamburger hotdog side = 39 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_food_l3243_324344


namespace NUMINAMATH_CALUDE_line_translation_theorem_l3243_324333

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - l.slope * dx + dy }

theorem line_translation_theorem :
  let original_line : Line := { slope := 2, intercept := -3 }
  let translated_line := translate original_line 2 3
  translated_line = { slope := 2, intercept := -4 } := by sorry

end NUMINAMATH_CALUDE_line_translation_theorem_l3243_324333


namespace NUMINAMATH_CALUDE_hostel_expenditure_hostel_expenditure_result_l3243_324313

/-- Calculates the new total expenditure of a hostel after accommodating more students -/
theorem hostel_expenditure (initial_students : ℕ) (additional_students : ℕ) 
  (average_decrease : ℕ) (total_increase : ℕ) : ℕ :=
  let new_students := initial_students + additional_students
  let original_average := (total_increase + new_students * average_decrease) / (new_students - initial_students)
  new_students * (original_average - average_decrease)

/-- The total expenditure of the hostel after accommodating more students is 7500 rupees -/
theorem hostel_expenditure_result : 
  hostel_expenditure 100 25 10 500 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_hostel_expenditure_result_l3243_324313


namespace NUMINAMATH_CALUDE_inequality_theorem_l3243_324347

theorem inequality_theorem (p q r : ℝ) 
  (h_order : p < q)
  (h_inequality : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) :
  p + 2*q + 3*r = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3243_324347


namespace NUMINAMATH_CALUDE_average_weight_section_A_l3243_324301

theorem average_weight_section_A (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_B : ℝ) (avg_weight_total : ℝ) :
  students_A = 24 →
  students_B = 16 →
  avg_weight_B = 35 →
  avg_weight_total = 38 →
  (students_A * avg_weight_section_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total →
  avg_weight_section_A = 40 := by
  sorry

#check average_weight_section_A

end NUMINAMATH_CALUDE_average_weight_section_A_l3243_324301


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l3243_324365

theorem reciprocal_equals_self (x : ℝ) : x ≠ 0 ∧ x = 1 / x → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l3243_324365


namespace NUMINAMATH_CALUDE_tenth_term_is_37_l3243_324311

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 3 + a 5 = 26) ∧
  (a 1 + a 2 + a 3 + a 4 = 28)

/-- The 10th term of the arithmetic sequence is 37 -/
theorem tenth_term_is_37 (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 10 = 37 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_37_l3243_324311


namespace NUMINAMATH_CALUDE_sum_of_x₁_and_x₂_l3243_324331

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def B : Set ℝ := {x | ∃ (x₁ x₂ : ℝ), x₁ ≤ x ∧ x ≤ x₂}

-- Define the conditions for union and intersection
axiom union_condition : A ∪ B = {x | x > -2}
axiom intersection_condition : A ∩ B = {x | 1 < x ∧ x ≤ 3}

-- The theorem to prove
theorem sum_of_x₁_and_x₂ : 
  ∃ (x₁ x₂ : ℝ), (∀ x, x ∈ B ↔ x₁ ≤ x ∧ x ≤ x₂) ∧ x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x₁_and_x₂_l3243_324331


namespace NUMINAMATH_CALUDE_right_angle_clackers_l3243_324306

/-- The number of clackers in a full circle -/
def clackers_in_full_circle : ℕ := 600

/-- The fraction of a full circle that a right angle represents -/
def right_angle_fraction : ℚ := 1/4

/-- The number of clackers in a right angle -/
def clackers_in_right_angle : ℕ := 150

/-- Theorem: The number of clackers in a right angle is 150 -/
theorem right_angle_clackers :
  clackers_in_right_angle = (clackers_in_full_circle : ℚ) * right_angle_fraction := by
  sorry

end NUMINAMATH_CALUDE_right_angle_clackers_l3243_324306


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3243_324320

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3243_324320


namespace NUMINAMATH_CALUDE_rectangle_area_l3243_324336

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 120,
    prove that its area is 675. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 120 → l * b = 675 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3243_324336


namespace NUMINAMATH_CALUDE_cleaning_time_with_help_l3243_324315

-- Define the grove dimensions
def trees_width : ℕ := 4
def trees_height : ℕ := 5

-- Define the initial cleaning time per tree
def initial_cleaning_time : ℕ := 6

-- Define the helper effect (halves the cleaning time)
def helper_effect : ℚ := 1/2

-- Theorem to prove
theorem cleaning_time_with_help :
  let total_trees := trees_width * trees_height
  let cleaning_time_with_help := initial_cleaning_time * helper_effect
  let total_cleaning_time := (total_trees : ℚ) * cleaning_time_with_help
  total_cleaning_time / 60 = 1 := by sorry

end NUMINAMATH_CALUDE_cleaning_time_with_help_l3243_324315


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l3243_324380

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The point symmetric to P(1, -2) with respect to the x-axis is (1, 2) -/
theorem symmetric_point_x_axis :
  let P : Point := { x := 1, y := -2 }
  symmetricXAxis P = { x := 1, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l3243_324380


namespace NUMINAMATH_CALUDE_sticker_difference_l3243_324308

theorem sticker_difference (initial_stickers : ℕ) : 
  initial_stickers > 0 →
  (initial_stickers - 15 : ℤ) / (initial_stickers + 18 : ℤ) = 2 / 5 →
  (initial_stickers + 18) - (initial_stickers - 15) = 33 := by
  sorry

#check sticker_difference

end NUMINAMATH_CALUDE_sticker_difference_l3243_324308


namespace NUMINAMATH_CALUDE_range_of_a_l3243_324334

/-- The function g(x) = ax + 2 -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, g a x₁ = f x₀) →
  a ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3243_324334


namespace NUMINAMATH_CALUDE_pancake_problem_l3243_324361

theorem pancake_problem (pancakes_made : ℕ) (family_size : ℕ) : pancakes_made = 12 → family_size = 8 → 
  (pancakes_made - family_size) + (family_size - (pancakes_made - family_size)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pancake_problem_l3243_324361


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3243_324318

/-- Given a block with houses and junk mail to distribute, calculate the number of pieces per house -/
def junk_mail_per_house (num_houses : ℕ) (total_junk_mail : ℕ) : ℕ :=
  total_junk_mail / num_houses

/-- Theorem: For a block with 6 houses and 24 pieces of junk mail, each house receives 4 pieces -/
theorem junk_mail_distribution :
  junk_mail_per_house 6 24 = 4 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3243_324318


namespace NUMINAMATH_CALUDE_fraction_equals_98_when_x_is_3_l3243_324304

theorem fraction_equals_98_when_x_is_3 :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64 + 2*x^2) / (x^4 + 8 + x^2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_98_when_x_is_3_l3243_324304


namespace NUMINAMATH_CALUDE_line_intersects_circle_intersection_point_polar_coordinates_l3243_324369

-- Define the line l
def line_l (x y : ℝ) : Prop := y - 1 = 2 * (x + 1)

-- Define the circle C₁
def circle_C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 4

-- Define the curve C₂
def curve_C₂ (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Theorem 1: Line l intersects circle C₁
theorem line_intersects_circle : ∃ (x y : ℝ), line_l x y ∧ circle_C₁ x y := by sorry

-- Theorem 2: The intersection point of C₁ and C₂ is (2, 2) in Cartesian coordinates
theorem intersection_point : ∃! (x y : ℝ), circle_C₁ x y ∧ curve_C₂ x y ∧ x = 2 ∧ y = 2 := by sorry

-- Theorem 3: The polar coordinates of the intersection point are (2√2, π/4)
theorem polar_coordinates : 
  let (x, y) := (2, 2)
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  ρ = 2 * Real.sqrt 2 ∧ θ = π / 4 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_intersection_point_polar_coordinates_l3243_324369


namespace NUMINAMATH_CALUDE_train_length_l3243_324349

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (cross_time : ℝ) : 
  speed_kmph = 72 → cross_time = 7 → speed_kmph * (1000 / 3600) * cross_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3243_324349


namespace NUMINAMATH_CALUDE_turquoise_tile_cost_l3243_324379

/-- Proves that the cost of each turquoise tile is $13 given the problem conditions -/
theorem turquoise_tile_cost :
  ∀ (total_area : ℝ) (tiles_per_sqft : ℝ) (purple_cost : ℝ) (savings : ℝ),
    total_area = 96 →
    tiles_per_sqft = 4 →
    purple_cost = 11 →
    savings = 768 →
    ∃ (turquoise_cost : ℝ),
      turquoise_cost = 13 ∧
      (total_area * tiles_per_sqft) * turquoise_cost - (total_area * tiles_per_sqft) * purple_cost = savings :=
by
  sorry


end NUMINAMATH_CALUDE_turquoise_tile_cost_l3243_324379


namespace NUMINAMATH_CALUDE_probability_same_gender_is_four_ninths_l3243_324335

/-- Represents a school with a specific number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- Calculates the total number of teachers in a school -/
def School.total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- Calculates the number of ways to select two teachers of the same gender -/
def same_gender_selections (s1 s2 : School) : ℕ :=
  s1.male_teachers * s2.male_teachers + s1.female_teachers * s2.female_teachers

/-- Calculates the total number of ways to select one teacher from each school -/
def total_selections (s1 s2 : School) : ℕ :=
  s1.total_teachers * s2.total_teachers

/-- The probability of selecting two teachers of the same gender -/
def probability_same_gender (s1 s2 : School) : ℚ :=
  (same_gender_selections s1 s2 : ℚ) / (total_selections s1 s2 : ℚ)

theorem probability_same_gender_is_four_ninths :
  let school_A : School := { male_teachers := 2, female_teachers := 1 }
  let school_B : School := { male_teachers := 1, female_teachers := 2 }
  probability_same_gender school_A school_B = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_gender_is_four_ninths_l3243_324335


namespace NUMINAMATH_CALUDE_log_relationship_l3243_324339

theorem log_relationship (a b : ℝ) : 
  a = Real.log 243 / Real.log 5 → b = Real.log 27 / Real.log 3 → a = 5 * b / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relationship_l3243_324339


namespace NUMINAMATH_CALUDE_roof_area_l3243_324362

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 4 * width →
  length - width = 45 →
  width * length = 900 := by
sorry

end NUMINAMATH_CALUDE_roof_area_l3243_324362


namespace NUMINAMATH_CALUDE_bikes_in_parking_lot_l3243_324371

theorem bikes_in_parking_lot :
  let num_cars : ℕ := 10
  let total_wheels : ℕ := 44
  let wheels_per_car : ℕ := 4
  let wheels_per_bike : ℕ := 2
  let num_bikes : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_bike
  num_bikes = 2 := by sorry

end NUMINAMATH_CALUDE_bikes_in_parking_lot_l3243_324371


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l3243_324319

/-- 
Given a canoe that rows downstream at 12 km/hr and a stream with a speed of 2 km/hr,
this theorem proves that the speed of the canoe when rowing upstream is 8 km/hr.
-/
theorem canoe_upstream_speed :
  let downstream_speed : ℝ := 12
  let stream_speed : ℝ := 2
  let canoe_speed : ℝ := downstream_speed - stream_speed
  let upstream_speed : ℝ := canoe_speed - stream_speed
  upstream_speed = 8 := by sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l3243_324319


namespace NUMINAMATH_CALUDE_a_months_is_seven_l3243_324350

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  a_oxen : ℕ
  b_oxen : ℕ
  c_oxen : ℕ
  b_months : ℕ
  c_months : ℕ
  total_rent : ℕ
  c_share : ℕ

/-- Calculates the number of months a put his oxen for grazing -/
def calculate_a_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, a put his oxen for 7 months -/
theorem a_months_is_seven (rental : PastureRental)
  (h1 : rental.a_oxen = 10)
  (h2 : rental.b_oxen = 12)
  (h3 : rental.c_oxen = 15)
  (h4 : rental.b_months = 5)
  (h5 : rental.c_months = 3)
  (h6 : rental.total_rent = 140)
  (h7 : rental.c_share = 36) :
  calculate_a_months rental = 7 :=
sorry

end NUMINAMATH_CALUDE_a_months_is_seven_l3243_324350


namespace NUMINAMATH_CALUDE_z_is_negative_intercept_l3243_324381

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Converts an objective function z = ax - y to a linear equation y = ax - z -/
def objectiveFunctionToLinearEquation (a : ℝ) (z : ℝ) : LinearEquation :=
  { slope := a, intercept := -z }

/-- Theorem: In the equation z = 3x - y, z represents the negative of the vertical intercept -/
theorem z_is_negative_intercept (z : ℝ) :
  let eq := objectiveFunctionToLinearEquation 3 z
  eq.intercept = -z := by sorry

end NUMINAMATH_CALUDE_z_is_negative_intercept_l3243_324381


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_time_proof_l3243_324382

/-- Calculates the time taken for two trains to pass each other --/
theorem train_passing_time (train_length : ℝ) (speed_fast : ℝ) (speed_slow : ℝ) : ℝ :=
  let speed_fast_ms := speed_fast * 1000 / 3600
  let speed_slow_ms := speed_slow * 1000 / 3600
  let relative_speed := speed_fast_ms + speed_slow_ms
  train_length / relative_speed

/-- Proves that the time taken for the slower train to pass the driver of the faster train is approximately 18 seconds --/
theorem train_passing_time_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_passing_time 475 55 40 - 18| < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_time_proof_l3243_324382


namespace NUMINAMATH_CALUDE_function_characterization_l3243_324353

def is_valid_function (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, (n : ℕ)^3 - (n : ℕ)^2 ≤ (f n : ℕ) * (f (f n) : ℕ)^2 ∧ 
             (f n : ℕ) * (f (f n) : ℕ)^2 ≤ (n : ℕ)^3 + (n : ℕ)^2

theorem function_characterization (f : ℕ+ → ℕ+) (h : is_valid_function f) : 
  ∀ n : ℕ+, f n = n - 1 ∨ f n = n ∨ f n = n + 1 :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3243_324353


namespace NUMINAMATH_CALUDE_approximate_cost_price_of_toy_l3243_324376

/-- The cost price of a toy given the selling conditions --/
def cost_price_of_toy (num_toys : ℕ) (total_selling_price : ℚ) (gain_in_toys : ℕ) : ℚ :=
  let selling_price_per_toy := total_selling_price / num_toys
  let x := selling_price_per_toy * num_toys / (num_toys + gain_in_toys)
  x

/-- Theorem stating the approximate cost price of a toy under given conditions --/
theorem approximate_cost_price_of_toy :
  let calculated_price := cost_price_of_toy 18 27300 3
  ⌊calculated_price⌋ = 1300 := by sorry

end NUMINAMATH_CALUDE_approximate_cost_price_of_toy_l3243_324376


namespace NUMINAMATH_CALUDE_f_g_inequality_l3243_324322

/-- The function f(x) = -x³ + x² + x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + x + a

/-- The function g(x) = 2a - x³ -/
def g (a : ℝ) (x : ℝ) : ℝ := 2*a - x^3

/-- Theorem: If g(x) ≥ f(x) for all x ∈ [0, 1], then a ≥ 2 -/
theorem f_g_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g a x ≥ f a x) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_g_inequality_l3243_324322


namespace NUMINAMATH_CALUDE_range_of_m_l3243_324389

-- Define the sets A and B
def A : Set ℝ := {a | a < -1}
def B (m : ℝ) : Set ℝ := {x | 3*m < x ∧ x < m + 2}

-- Define the proposition P
def P (a : ℝ) : Prop := ∃ x : ℝ, a*x^2 + 2*x - 1 = 0

-- State the theorem
theorem range_of_m :
  (∀ a : ℝ, ¬(P a)) →
  (∀ m : ℝ, ∀ x : ℝ, x ∈ B m → x ∉ A) →
  {m : ℝ | -1/3 ≤ m} = {m : ℝ | ∃ x : ℝ, x ∈ B m} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3243_324389


namespace NUMINAMATH_CALUDE_simultaneous_equations_solutions_l3243_324324

theorem simultaneous_equations_solutions :
  let eq1 (x y : ℝ) := x^2 + 3*y = 10
  let eq2 (x y : ℝ) := 3 + y = 10/x
  (eq1 (-5) (-5) ∧ eq2 (-5) (-5)) ∧
  (eq1 2 2 ∧ eq2 2 2) ∧
  (eq1 3 (1/3) ∧ eq2 3 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solutions_l3243_324324


namespace NUMINAMATH_CALUDE_row_swap_matrix_l3243_324374

theorem row_swap_matrix : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] := by
  sorry

end NUMINAMATH_CALUDE_row_swap_matrix_l3243_324374


namespace NUMINAMATH_CALUDE_sum_components_eq_46_l3243_324327

/-- Represents a trapezoid with four sides --/
structure Trapezoid :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)

/-- Represents the sum of areas in the form r₄√n₄ + r₅√n₅ + r₆ --/
structure AreaSum :=
  (r₄ : ℚ) (r₅ : ℚ) (r₆ : ℚ) (n₄ : ℕ) (n₅ : ℕ)

/-- Function to calculate the sum of all possible areas of a trapezoid --/
def sumAreas (t : Trapezoid) : AreaSum :=
  sorry

/-- Theorem stating that the sum of components equals 46 for the given trapezoid --/
theorem sum_components_eq_46 (t : Trapezoid) (a : AreaSum) :
  t.side1 = 4 ∧ t.side2 = 6 ∧ t.side3 = 8 ∧ t.side4 = 10 ∧
  a = sumAreas t →
  a.r₄ + a.r₅ + a.r₆ + a.n₄ + a.n₅ = 46 :=
sorry

end NUMINAMATH_CALUDE_sum_components_eq_46_l3243_324327


namespace NUMINAMATH_CALUDE_zeros_of_quadratic_l3243_324366

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that -1 and 3 are the zeros of the quadratic function f -/
theorem zeros_of_quadratic :
  (f (-1) = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_quadratic_l3243_324366


namespace NUMINAMATH_CALUDE_g_composition_of_three_l3243_324332

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 5*n - 3

theorem g_composition_of_three : g (g (g 3)) = 232 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l3243_324332


namespace NUMINAMATH_CALUDE_max_value_implies_a_range_l3243_324358

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (a + 2) * x

theorem max_value_implies_a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, f a x ≤ f a (1/2)) : 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_range_l3243_324358


namespace NUMINAMATH_CALUDE_ursulas_purchases_l3243_324309

theorem ursulas_purchases (tea_price : ℝ) 
  (h1 : tea_price = 10)
  (h2 : tea_price > 0) :
  let cheese_price := tea_price / 2
  let butter_price := 0.8 * cheese_price
  let bread_price := butter_price / 2
  tea_price + cheese_price + butter_price + bread_price = 21 :=
by sorry

end NUMINAMATH_CALUDE_ursulas_purchases_l3243_324309


namespace NUMINAMATH_CALUDE_cubic_function_property_l3243_324359

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 3
  f (-2) = 7 → f 2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3243_324359


namespace NUMINAMATH_CALUDE_prime_power_digit_repetition_l3243_324399

theorem prime_power_digit_repetition (p n : ℕ) : 
  Prime p → p > 3 → (10^19 ≤ p^n ∧ p^n < 10^20) → 
  ∃ (d : ℕ) (i j k : ℕ), i < j ∧ j < k ∧ i < 20 ∧ j < 20 ∧ k < 20 ∧
  d < 10 ∧ (p^n / 10^i) % 10 = d ∧ (p^n / 10^j) % 10 = d ∧ (p^n / 10^k) % 10 = d :=
by sorry

end NUMINAMATH_CALUDE_prime_power_digit_repetition_l3243_324399


namespace NUMINAMATH_CALUDE_prob_diff_games_l3243_324312

/-- The probability of getting heads on a single coin flip -/
def p_heads : ℚ := 3/5

/-- The probability of getting tails on a single coin flip -/
def p_tails : ℚ := 2/5

/-- The probability of winning Game A -/
def p_win_game_a : ℚ := p_heads^4 + p_tails^4

/-- The probability of winning Game B -/
def p_win_game_b : ℚ := (p_heads^2 + p_tails^2) * (p_heads^3 + p_tails^3)

theorem prob_diff_games : p_win_game_a - p_win_game_b = 6/625 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_games_l3243_324312


namespace NUMINAMATH_CALUDE_line_intercept_at_10_l3243_324307

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

theorem line_intercept_at_10 : 
  let l : Line := { x₁ := 7, y₁ := 3, x₂ := 3, y₂ := 7 }
  xIntercept l = 10 := by sorry

end NUMINAMATH_CALUDE_line_intercept_at_10_l3243_324307


namespace NUMINAMATH_CALUDE_K_idempotent_l3243_324340

/-- The set of all 2013 × 2013 arrays with entries 0 and 1 -/
def F : Type := Fin 2013 → Fin 2013 → Fin 2

/-- The sum of all entries sharing a row or column with a[i,j] -/
def S (A : F) (i j : Fin 2013) : ℕ :=
  (Finset.sum (Finset.range 2013) (fun k => A i k)) +
  (Finset.sum (Finset.range 2013) (fun k => A k j)) -
  A i j

/-- The transformation K -/
def K (A : F) : F :=
  fun i j => (S A i j) % 2

/-- The main theorem: K(K(A)) = K(A) for all A in F -/
theorem K_idempotent (A : F) : K (K A) = K A := by sorry

end NUMINAMATH_CALUDE_K_idempotent_l3243_324340


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3243_324383

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the removal of a smaller prism from a larger solid -/
structure PrismRemoval where
  original : RectangularSolid
  removed : RectangularSolid
  flushFaces : ℕ

/-- Theorem stating that the surface area remains unchanged after removal -/
theorem surface_area_unchanged (removal : PrismRemoval) :
  removal.original = RectangularSolid.mk 4 3 2 →
  removal.removed = RectangularSolid.mk 1 1 2 →
  removal.flushFaces = 2 →
  surfaceArea removal.original = surfaceArea removal.original - surfaceArea removal.removed + 2 * removal.removed.length * removal.removed.width :=
by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l3243_324383


namespace NUMINAMATH_CALUDE_smallest_four_digit_all_different_l3243_324348

/-- A function that checks if a natural number has all digits different --/
def allDigitsDifferent (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

/-- The smallest four-digit number with all digits different --/
def smallestFourDigitAllDifferent : ℕ := 1023

/-- Theorem: 1023 is the smallest four-digit number with all digits different --/
theorem smallest_four_digit_all_different :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ allDigitsDifferent n → smallestFourDigitAllDifferent ≤ n) ∧
  1000 ≤ smallestFourDigitAllDifferent ∧
  smallestFourDigitAllDifferent < 10000 ∧
  allDigitsDifferent smallestFourDigitAllDifferent :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_all_different_l3243_324348


namespace NUMINAMATH_CALUDE_min_value_xyz_l3243_324342

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  x + 3 * y + 6 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 6 * z₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l3243_324342


namespace NUMINAMATH_CALUDE_marble_problem_l3243_324387

theorem marble_problem (M : ℕ) 
  (h1 : M > 0)
  (h2 : (M - M / 3) / 4 > 0)
  (h3 : M - M / 3 - (M - M / 3) / 4 - 2 * ((M - M / 3) / 4) = 7) : 
  M = 42 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l3243_324387


namespace NUMINAMATH_CALUDE_chord_longer_than_arc_l3243_324375

theorem chord_longer_than_arc (R : ℝ) (h : R > 0) :
  let angle := 60 * π / 180
  let arc_length := angle * R
  let new_radius := 1.05 * R
  let chord_length := 2 * new_radius * Real.sin (angle / 2)
  chord_length > arc_length := by sorry

end NUMINAMATH_CALUDE_chord_longer_than_arc_l3243_324375


namespace NUMINAMATH_CALUDE_additive_inverse_solution_l3243_324373

theorem additive_inverse_solution (x : ℝ) : (2*x - 12) + (x + 3) = 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_solution_l3243_324373


namespace NUMINAMATH_CALUDE_union_equals_universal_l3243_324343

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_equals_universal : M ∪ N = U := by
  sorry

end NUMINAMATH_CALUDE_union_equals_universal_l3243_324343


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3243_324368

theorem jelly_bean_probability (p_red p_orange p_green p_yellow : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_green = 0.25 →
  p_red + p_orange + p_green + p_yellow = 1 →
  p_yellow = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3243_324368


namespace NUMINAMATH_CALUDE_fourth_month_sales_l3243_324354

/-- Calculates the missing sales amount for a month given the sales of other months and the average --/
def calculate_missing_sales (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

theorem fourth_month_sales :
  let sale1 : ℕ := 6400
  let sale2 : ℕ := 7000
  let sale3 : ℕ := 6800
  let sale5 : ℕ := 6500
  let sale6 : ℕ := 5100
  let average : ℕ := 6500
  calculate_missing_sales sale1 sale2 sale3 sale5 sale6 average = 7200 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l3243_324354


namespace NUMINAMATH_CALUDE_binomial_factorial_product_l3243_324360

theorem binomial_factorial_product : (Nat.choose 60 3) * (Nat.factorial 10) = 124467072000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_factorial_product_l3243_324360


namespace NUMINAMATH_CALUDE_new_bucket_capacity_l3243_324346

/-- Represents the capacity of a water tank in liters. -/
def TankCapacity : ℝ := 22 * 13.5

/-- Proves that given a tank that can be filled by either 22 buckets of 13.5 liters each
    or 33 buckets of equal capacity, the capacity of each of the 33 buckets is 9 liters. -/
theorem new_bucket_capacity : 
  ∀ (new_capacity : ℝ), 
  (33 * new_capacity = TankCapacity) → 
  new_capacity = 9 := by
sorry

end NUMINAMATH_CALUDE_new_bucket_capacity_l3243_324346


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3243_324321

/-- A parabola defined by y = ax² where a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- A line with slope 1 -/
structure Line where
  b : ℝ

/-- Intersection points of a parabola and a line -/
structure Intersection (p : Parabola) (l : Line) where
  x₁ : ℝ
  x₂ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = p.a * x₁^2
  eq₂ : y₂ = p.a * x₂^2
  eq₃ : y₁ = x₁ + l.b
  eq₄ : y₂ = x₂ + l.b

/-- The theorem to be proved -/
theorem parabola_focus_directrix_distance 
  (p : Parabola) (l : Line) (i : Intersection p l) :
  (i.x₁ + i.x₂) / 2 = 1 → 1 / (4 * p.a) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3243_324321


namespace NUMINAMATH_CALUDE_catman_do_whisker_count_l3243_324356

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers (princess_puff_whiskers : ℕ) : ℕ :=
  2 * princess_puff_whiskers - 6

/-- Theorem stating the number of whiskers Catman Do has -/
theorem catman_do_whisker_count :
  catman_do_whiskers 14 = 22 := by
  sorry

end NUMINAMATH_CALUDE_catman_do_whisker_count_l3243_324356


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3243_324314

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x - 3| :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3243_324314


namespace NUMINAMATH_CALUDE_pyramid_volume_l3243_324355

theorem pyramid_volume (base_length base_width height : ℝ) 
  (h1 : base_length = 2/3)
  (h2 : base_width = 1/2)
  (h3 : height = 1) : 
  (1/3 : ℝ) * base_length * base_width * height = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3243_324355


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3243_324300

theorem isosceles_triangle_side_lengths 
  (perimeter : ℝ) 
  (height : ℝ) 
  (is_isosceles : Bool) 
  (h1 : perimeter = 16) 
  (h2 : height = 4) 
  (h3 : is_isosceles = true) : 
  ∃ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 ∧ a + b + c = perimeter := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3243_324300


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l3243_324377

theorem tan_theta_in_terms_of_x (θ x : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.cos (θ/2) = Real.sqrt ((x - 2)/(2*x))) : 
  Real.tan θ = -1/2 * Real.sqrt (x^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l3243_324377


namespace NUMINAMATH_CALUDE_abs_neg_five_l3243_324328

theorem abs_neg_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_l3243_324328


namespace NUMINAMATH_CALUDE_cube_sum_equals_sum_l3243_324384

theorem cube_sum_equals_sum (a b : ℝ) : 
  (a / (1 + b) + b / (1 + a) = 1) → a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_sum_l3243_324384


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3243_324378

open Real

theorem indefinite_integral_proof (x : ℝ) (C : ℝ) (h : x ≠ -2 ∧ x ≠ -1) :
  deriv (λ y => 2 * log (abs (y + 2)) - 1 / (2 * (y + 1)^2) + C) x =
  (2 * x^3 + 6 * x^2 + 7 * x + 4) / ((x + 2) * (x + 1)^3) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3243_324378


namespace NUMINAMATH_CALUDE_sequence_property_l3243_324390

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a n)
  (h2 : ∀ n, a n < a (n + 1))
  (h3 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l3243_324390


namespace NUMINAMATH_CALUDE_walter_age_2005_conditions_hold_l3243_324316

-- Define Walter's age in 2000
def walter_age_2000 : ℚ := 4 / 3

-- Define grandmother's age in 2000
def grandmother_age_2000 : ℚ := 2 * walter_age_2000

-- Define the current year
def current_year : ℕ := 2000

-- Define the target year
def target_year : ℕ := 2005

-- Define the sum of birth years
def sum_birth_years : ℕ := 4004

-- Theorem statement
theorem walter_age_2005 :
  (walter_age_2000 + (target_year - current_year : ℚ)) = 19 / 3 :=
by
  sorry

-- Verify the conditions
theorem conditions_hold :
  (walter_age_2000 = grandmother_age_2000 / 2) ∧
  (current_year - walter_age_2000 + current_year - grandmother_age_2000 = sum_birth_years) :=
by
  sorry

end NUMINAMATH_CALUDE_walter_age_2005_conditions_hold_l3243_324316


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3243_324392

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 + 4*x + 3 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + 4*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3243_324392


namespace NUMINAMATH_CALUDE_work_completion_time_B_l3243_324345

theorem work_completion_time_B (a b : ℝ) : 
  (a + b = 1/6) →  -- A and B together complete 1/6 of the work in one day
  (a = 1/11) →     -- A alone completes 1/11 of the work in one day
  (b = 5/66) →     -- B alone completes 5/66 of the work in one day
  (1/b = 66/5) :=  -- The time B takes to complete the work alone is 66/5 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_B_l3243_324345


namespace NUMINAMATH_CALUDE_initial_men_count_initial_men_count_is_seven_l3243_324386

/-- Proves that the initial number of men in a group is 7 given specific conditions about age changes. -/
theorem initial_men_count : ℕ :=
  let initial_average : ℝ := sorry
  let final_average : ℝ := initial_average + 4
  let replaced_men_ages : Fin 2 → ℕ := ![26, 30]
  let women_average_age : ℝ := 42
  let men_count : ℕ := sorry
  have h1 : final_average * men_count = initial_average * men_count + 4 * men_count := sorry
  have h2 : (men_count - 2) * initial_average + 2 * women_average_age = men_count * final_average := sorry
  have h3 : 2 * women_average_age - (replaced_men_ages 0 + replaced_men_ages 1) = 4 * men_count := sorry
  7

theorem initial_men_count_is_seven : initial_men_count = 7 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_initial_men_count_is_seven_l3243_324386


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3243_324326

theorem smallest_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 5 = 1 ∧
  b % 4 = 2 ∧
  b % 7 = 3 ∧
  ∀ c : ℕ, c > 0 → c % 5 = 1 → c % 4 = 2 → c % 7 = 3 → b ≤ c :=
by
  use 86
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3243_324326


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_8_l3243_324395

theorem largest_four_digit_divisible_by_8 : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → n % 8 = 0 → n ≤ 9992 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_8_l3243_324395


namespace NUMINAMATH_CALUDE_f_plus_g_is_linear_l3243_324317

/-- Represents a cubic function ax³ + bx² + cx + d -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The function resulting from reflecting and translating a cubic function -/
def reflected_translated (cf : CubicFunction) (x : ℝ) : ℝ :=
  cf.a * (x - 10)^3 + cf.b * (x - 10)^2 + cf.c * (x - 10) + cf.d

/-- The function resulting from reflecting about x-axis, then translating a cubic function -/
def reflected_translated_negative (cf : CubicFunction) (x : ℝ) : ℝ :=
  -cf.a * (x + 10)^3 - cf.b * (x + 10)^2 - cf.c * (x + 10) - cf.d

/-- The sum of the two reflected and translated functions -/
def f_plus_g (cf : CubicFunction) (x : ℝ) : ℝ :=
  reflected_translated cf x + reflected_translated_negative cf x

/-- Theorem stating that f_plus_g is a non-horizontal linear function -/
theorem f_plus_g_is_linear (cf : CubicFunction) :
  ∃ m k, m ≠ 0 ∧ ∀ x, f_plus_g cf x = m * x + k :=
sorry

end NUMINAMATH_CALUDE_f_plus_g_is_linear_l3243_324317


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l3243_324341

/-- Given vectors a and b, prove that the magnitude of 2a + b is 5√2 -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  a = (3, 2) → b = (-1, 1) → ‖(2 • a) + b‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l3243_324341


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l3243_324397

theorem cistern_emptying_time (fill_time : ℝ) (combined_time : ℝ) : 
  fill_time = 7 → combined_time = 31.5 → 
  (fill_time⁻¹ - (fill_time⁻¹ - combined_time⁻¹)⁻¹) = 9⁻¹ :=
by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l3243_324397


namespace NUMINAMATH_CALUDE_road_repair_workers_l3243_324303

/-- Represents the work done by a group of workers -/
structure Work where
  persons : ℕ
  days : ℕ
  hours_per_day : ℕ

/-- Calculates the total work units -/
def total_work (w : Work) : ℕ := w.persons * w.days * w.hours_per_day

theorem road_repair_workers (first_group : Work) (second_group : Work) :
  first_group.days = 12 ∧
  first_group.hours_per_day = 5 ∧
  second_group.persons = 30 ∧
  second_group.days = 17 ∧
  second_group.hours_per_day = 6 ∧
  total_work first_group = total_work second_group →
  first_group.persons = 51 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_workers_l3243_324303


namespace NUMINAMATH_CALUDE_water_bottle_consumption_l3243_324388

theorem water_bottle_consumption (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : 
  total = 24 → 
  first_day_fraction = 1/3 → 
  remaining = 8 → 
  (total - (first_day_fraction * total).num - remaining : ℚ) / (total - (first_day_fraction * total).num) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_water_bottle_consumption_l3243_324388


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3243_324385

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 16 + a 30 = 60 →
  a 10 + a 22 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3243_324385


namespace NUMINAMATH_CALUDE_angle_A_measure_l3243_324302

/-- Given a geometric figure with the following properties:
  - Angle B measures 120°
  - A line divides the space opposite angle B on a straight line into two angles
  - One of these angles measures 50°
  - Angle A is vertically opposite to the angle that is not 50°
  Prove that angle A measures 130° -/
theorem angle_A_measure (B : ℝ) (angle1 : ℝ) (angle2 : ℝ) (A : ℝ) 
  (h1 : B = 120)
  (h2 : angle1 + angle2 = 180 - B)
  (h3 : angle1 = 50)
  (h4 : A = 180 - angle2) :
  A = 130 := by sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3243_324302


namespace NUMINAMATH_CALUDE_max_m_value_l3243_324370

theorem max_m_value (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + m*x₁ + 6 = 0 ∧ x₂^2 + m*x₂ + 6 = 0 ∧ |x₁ - x₂| = Real.sqrt 85) →
  m ≤ Real.sqrt 109 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3243_324370


namespace NUMINAMATH_CALUDE_positive_less_than_one_inequality_l3243_324393

theorem positive_less_than_one_inequality (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_positive_less_than_one_inequality_l3243_324393


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3243_324394

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3243_324394


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l3243_324351

/-- Given a curve f(x) = ax - b/x, prove that if its tangent line at (2, f(2)) 
    is 7x - 4y - 12 = 0, then a = 1 and b = 3 -/
theorem tangent_line_implies_a_and_b (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x - b / x
  let f' : ℝ → ℝ := λ x => a + b / (x^2)
  let tangent_slope : ℝ := f' 2
  let point_on_curve : ℝ := f 2
  (7 * 2 - 4 * point_on_curve - 12 = 0 ∧ 
   7 - 4 * tangent_slope = 0) →
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l3243_324351


namespace NUMINAMATH_CALUDE_expression_simplification_l3243_324367

theorem expression_simplification (x y m n : ℝ) : 
  (2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2) ∧
  (9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3243_324367


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3243_324363

theorem quadratic_factorization (x : ℝ) : x^2 + 6*x = 1 ↔ (x + 3)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3243_324363


namespace NUMINAMATH_CALUDE_ryan_recruitment_count_l3243_324398

def total_funding_required : ℕ := 1000
def ryan_initial_funds : ℕ := 200
def average_funding_per_person : ℕ := 10

theorem ryan_recruitment_count :
  (total_funding_required - ryan_initial_funds) / average_funding_per_person = 80 := by
  sorry

end NUMINAMATH_CALUDE_ryan_recruitment_count_l3243_324398


namespace NUMINAMATH_CALUDE_shower_tiles_width_l3243_324364

/-- Given a 3-walled shower with 20 tiles running the height of each wall and 480 tiles in total,
    the number of tiles running the width of each wall is 8. -/
theorem shower_tiles_width (num_walls : Nat) (height_tiles : Nat) (total_tiles : Nat) :
  num_walls = 3 → height_tiles = 20 → total_tiles = 480 →
  ∃ width_tiles : Nat, width_tiles = 8 ∧ num_walls * height_tiles * width_tiles = total_tiles :=
by sorry

end NUMINAMATH_CALUDE_shower_tiles_width_l3243_324364


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l3243_324305

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + 1

theorem circumcircle_radius_of_triangle (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x = Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (f C = 2) →
  (a + b = 4) →
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 3) →
  (∃ R : ℝ, R = c / (2 * Real.sin C) ∧ R = 2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l3243_324305


namespace NUMINAMATH_CALUDE_rowing_speed_problem_l3243_324391

/-- The rowing speed problem -/
theorem rowing_speed_problem (v c : ℝ) (h1 : c = 1.1)
  (h2 : (v + c) * t = (v - c) * (2 * t) → t ≠ 0) : v = 3.3 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_problem_l3243_324391


namespace NUMINAMATH_CALUDE_orange_balls_count_l3243_324372

theorem orange_balls_count :
  let total_balls : ℕ := 100
  let red_balls : ℕ := 30
  let blue_balls : ℕ := 20
  let yellow_balls : ℕ := 10
  let green_balls : ℕ := 5
  let pink_balls : ℕ := 2 * green_balls
  let orange_balls : ℕ := 3 * pink_balls
  red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls →
  orange_balls = 30 :=
by sorry

end NUMINAMATH_CALUDE_orange_balls_count_l3243_324372


namespace NUMINAMATH_CALUDE_f_value_at_107_5_l3243_324396

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_value_at_107_5 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : ∀ x, f (x + 3) = -1 / f x)
  (h_interval : ∀ x ∈ Set.Icc (-3) (-2), f x = 4 * x) :
  f 107.5 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_107_5_l3243_324396


namespace NUMINAMATH_CALUDE_three_lines_divide_plane_l3243_324329

/-- A line in the plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at different points -/
def intersect_differently (l1 l2 l3 : Line) : Prop :=
  ¬ parallel l1 l2 ∧ ¬ parallel l1 l3 ∧ ¬ parallel l2 l3 ∧
  (l1.a * l2.b - l1.b * l2.a ≠ 0) ∧
  (l1.a * l3.b - l1.b * l3.a ≠ 0) ∧
  (l2.a * l3.b - l2.b * l3.a ≠ 0)

/-- The three given lines -/
def line1 : Line := ⟨1, 2, -1⟩
def line2 : Line := ⟨1, 0, 1⟩
def line3 (k : ℝ) : Line := ⟨1, k, 0⟩

theorem three_lines_divide_plane (k : ℝ) : 
  intersect_differently line1 line2 (line3 k) ↔ (k = 0 ∨ k = 1 ∨ k = 2) :=
sorry

end NUMINAMATH_CALUDE_three_lines_divide_plane_l3243_324329


namespace NUMINAMATH_CALUDE_weight_difference_l3243_324338

/-- Given weights of five individuals A, B, C, D, and E, prove that E weighs 6 kg more than D
    under specific average weight conditions. -/
theorem weight_difference (W_A W_B W_C W_D W_E : ℝ) : 
  (W_A + W_B + W_C) / 3 = 84 →
  (W_A + W_B + W_C + W_D) / 4 = 80 →
  (W_B + W_C + W_D + W_E) / 4 = 79 →
  W_A = 78 →
  W_E - W_D = 6 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l3243_324338


namespace NUMINAMATH_CALUDE_complex_circle_extrema_l3243_324325

theorem complex_circle_extrema (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 1) :
  (∃ w : ℂ, Complex.abs (w - (1 + 2*I)) = 1 ∧ Complex.abs (w - (3 + I)) = Real.sqrt 5 + 1) ∧
  (∃ v : ℂ, Complex.abs (v - (1 + 2*I)) = 1 ∧ Complex.abs (v - (3 + I)) = Real.sqrt 5 - 1) ∧
  (∀ u : ℂ, Complex.abs (u - (1 + 2*I)) = 1 →
    Real.sqrt 5 - 1 ≤ Complex.abs (u - (3 + I)) ∧ Complex.abs (u - (3 + I)) ≤ Real.sqrt 5 + 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_circle_extrema_l3243_324325


namespace NUMINAMATH_CALUDE_prism_volume_l3243_324323

/-- 
  Given a right prism with an isosceles triangle base ABC, where:
  - AB = AC
  - ∠BAC = α
  - A line segment of length l from the upper vertex A₁ to the center of 
    the circumscribed circle of ABC makes an angle β with the base plane
  
  The volume of the prism is l³ sin(2β) cos(β) sin(α) cos²(α/2)
-/
theorem prism_volume 
  (α β l : ℝ) 
  (h_α : 0 < α ∧ α < π) 
  (h_β : 0 < β ∧ β < π/2) 
  (h_l : l > 0) : 
  ∃ (V : ℝ), V = l^3 * Real.sin (2*β) * Real.cos β * Real.sin α * (Real.cos (α/2))^2 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3243_324323


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3243_324330

/-- Given a boat that travels 13 km/hr downstream and 4 km/hr upstream,
    prove that its speed in still water is 8.5 km/hr. -/
theorem boat_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : downstream_speed = 13)
  (h2 : upstream_speed = 4) :
  (downstream_speed + upstream_speed) / 2 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3243_324330
