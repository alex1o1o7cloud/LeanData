import Mathlib

namespace NUMINAMATH_CALUDE_girls_boys_difference_l663_66329

theorem girls_boys_difference (girls boys : ℝ) (h1 : girls = 542.0) (h2 : boys = 387.0) :
  girls - boys = 155.0 := by
  sorry

end NUMINAMATH_CALUDE_girls_boys_difference_l663_66329


namespace NUMINAMATH_CALUDE_A_equals_B_l663_66319

-- Define set A
def A : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 2*y^2}

-- Define set B
def B : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 6*x*y + 11*y^2}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l663_66319


namespace NUMINAMATH_CALUDE_election_votes_calculation_l663_66337

theorem election_votes_calculation (V : ℝ) 
  (h1 : 0.30 * V + 0.25 * V + 0.20 * V + 0.25 * V = V)  -- Condition 2
  (h2 : (0.30 * V + 0.0225 * V) - (0.25 * V + 0.0225 * V) = 1350)  -- Conditions 3, 4, and 5
  : V = 27000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l663_66337


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l663_66310

/-- An isosceles triangle with sides of 3cm and 6cm has a perimeter of 15cm. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 3 ∧ b = 6 ∧ 
  (c = a ∨ c = b) ∧  -- Isosceles condition
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l663_66310


namespace NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l663_66330

theorem no_three_digit_perfect_square_sum :
  ∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  ¬∃ m : ℕ, m^2 = 111 * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_perfect_square_sum_l663_66330


namespace NUMINAMATH_CALUDE_population_increase_rate_l663_66343

/-- If a population increases by 220 persons in 55 minutes at a constant rate,
    then the rate of population increase is 15 seconds per person. -/
theorem population_increase_rate 
  (total_increase : ℕ) 
  (time_minutes : ℕ) 
  (h1 : total_increase = 220)
  (h2 : time_minutes = 55) :
  (time_minutes * 60) / total_increase = 15 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_l663_66343


namespace NUMINAMATH_CALUDE_juan_running_time_l663_66344

/-- Given that Juan ran at a speed of 10.0 miles per hour and covered a distance of 800 miles,
    prove that the time he ran equals 80 hours. -/
theorem juan_running_time (speed : ℝ) (distance : ℝ) (h1 : speed = 10.0) (h2 : distance = 800) :
  distance / speed = 80 :=
by sorry

end NUMINAMATH_CALUDE_juan_running_time_l663_66344


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l663_66374

theorem more_girls_than_boys (total_pupils : ℕ) (girls : ℕ) 
  (h1 : total_pupils = 926)
  (h2 : girls = 692)
  (h3 : girls > total_pupils - girls) :
  girls - (total_pupils - girls) = 458 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l663_66374


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l663_66332

-- Part 1
theorem simplify_expression_1 (a b : ℝ) : 2*a - (-3*b - 3*(3*a - b)) = 11*a := by sorry

-- Part 2
theorem simplify_expression_2 (a b : ℝ) : 12*a*b^2 - (7*a^2*b - (a*b^2 - 3*a^2*b)) = 13*a*b^2 - 10*a^2*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l663_66332


namespace NUMINAMATH_CALUDE_inequality_proof_l663_66348

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 
  Real.sqrt 3 + (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l663_66348


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l663_66365

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 24 * x + c = 0) →  -- exactly one solution
  (a + c = 31) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 9 ∧ c = 22) :=                 -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l663_66365


namespace NUMINAMATH_CALUDE_smartphone_price_l663_66340

theorem smartphone_price (x : ℝ) : (0.90 * x - 100) = (0.80 * x - 20) → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_l663_66340


namespace NUMINAMATH_CALUDE_expression_equals_four_l663_66325

theorem expression_equals_four :
  let a := 7 + Real.sqrt 48
  let b := 7 - Real.sqrt 48
  (a^2023 + b^2023)^2 - (a^2023 - b^2023)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l663_66325


namespace NUMINAMATH_CALUDE_largest_geometric_three_digit_number_l663_66376

/-- Checks if three digits form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Checks if three digits are distinct -/
def are_distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Represents a three-digit number -/
def three_digit_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem largest_geometric_three_digit_number :
  ∀ n : ℕ,
  (∃ a b c : ℕ, n = three_digit_number a b c ∧
                a ≤ 8 ∧
                a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
                a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
                is_geometric_sequence a b c ∧
                are_distinct a b c) →
  n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_geometric_three_digit_number_l663_66376


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l663_66320

theorem simplify_trig_expression :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l663_66320


namespace NUMINAMATH_CALUDE_middle_circle_radius_l663_66335

/-- A configuration of five consecutively tangent circles between two parallel lines -/
structure CircleConfiguration where
  /-- The radii of the five circles, from smallest to largest -/
  radii : Fin 5 → ℝ
  /-- The radii are positive -/
  radii_pos : ∀ i, 0 < radii i
  /-- The radii are in ascending order -/
  radii_ascending : ∀ i j, i < j → radii i ≤ radii j
  /-- The circles are tangent to each other -/
  tangent_circles : ∀ i : Fin 4, radii i + radii (i + 1) = radii (i + 1) - radii i

/-- The theorem stating that the middle circle's radius is 10 cm -/
theorem middle_circle_radius
  (config : CircleConfiguration)
  (h_smallest : config.radii 0 = 5)
  (h_largest : config.radii 4 = 15) :
  config.radii 2 = 10 :=
sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l663_66335


namespace NUMINAMATH_CALUDE_difference_of_squares_l663_66327

theorem difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l663_66327


namespace NUMINAMATH_CALUDE_double_counted_is_eight_l663_66375

/-- The number of double-counted toddlers in Bill's count -/
def double_counted : ℕ := 26 - 21 + 3

/-- Proof that the number of double-counted toddlers is 8 -/
theorem double_counted_is_eight : double_counted = 8 := by
  sorry

#eval double_counted

end NUMINAMATH_CALUDE_double_counted_is_eight_l663_66375


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l663_66366

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_16 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → digit_sum n = 16 → n ≤ 880 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_16_l663_66366


namespace NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_to_product_l663_66363

theorem no_triangle_with_cube_sum_equal_to_product (x y z : ℝ) :
  (0 < x ∧ 0 < y ∧ 0 < z) →
  (x + y > z ∧ y + z > x ∧ z + x > y) →
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_to_product_l663_66363


namespace NUMINAMATH_CALUDE_equation_solution_l663_66385

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (2 / x + (3 / x) / (6 / x) = 1.5) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l663_66385


namespace NUMINAMATH_CALUDE_fraction_zero_iff_x_zero_l663_66341

theorem fraction_zero_iff_x_zero (x : ℝ) (h : x ≠ -2) :
  2 * x / (x + 2) = 0 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_fraction_zero_iff_x_zero_l663_66341


namespace NUMINAMATH_CALUDE_roots_have_unit_modulus_l663_66373

theorem roots_have_unit_modulus (z : ℂ) :
  (11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_have_unit_modulus_l663_66373


namespace NUMINAMATH_CALUDE_ship_grain_calculation_l663_66336

/-- The amount of grain spilled into the water, in tons -/
def grain_spilled : ℕ := 49952

/-- The amount of grain remaining onboard, in tons -/
def grain_remaining : ℕ := 918

/-- The original amount of grain on the ship, in tons -/
def original_grain : ℕ := grain_spilled + grain_remaining

theorem ship_grain_calculation :
  original_grain = 50870 :=
sorry

end NUMINAMATH_CALUDE_ship_grain_calculation_l663_66336


namespace NUMINAMATH_CALUDE_inequality_theorem_l663_66360

theorem inequality_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x * (x - z)^2 + y * (y - z)^2 ≥ (x - z) * (y - z) * (x + y - z) ∧
  (x * (x - z)^2 + y * (y - z)^2 = (x - z) * (y - z) * (x + y - z) ↔ 
    (x = y ∧ y = z) ∨ (x = y ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l663_66360


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l663_66355

/-- Given a cube with a face perimeter of 24 cm, prove its volume is 216 cubic cm. -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 24) :
  let side_length := face_perimeter / 4
  side_length ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l663_66355


namespace NUMINAMATH_CALUDE_max_speed_theorem_l663_66398

/-- Represents a set of observations for machine speed and defective items produced. -/
structure Observation where
  speed : ℝ
  defects : ℝ

/-- Calculates the slope of the linear regression line. -/
def calculateSlope (observations : List Observation) : ℝ :=
  sorry

/-- Calculates the y-intercept of the linear regression line. -/
def calculateIntercept (observations : List Observation) (slope : ℝ) : ℝ :=
  sorry

/-- Theorem: The maximum speed at which the machine can operate while producing
    no more than 10 defective items per hour is 15 revolutions per second. -/
theorem max_speed_theorem (observations : List Observation)
    (h1 : observations = [⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩])
    (h2 : ∀ obs ∈ observations, obs.speed > 0 ∧ obs.defects > 0)
    (h3 : calculateSlope observations > 0) : 
    let slope := calculateSlope observations
    let intercept := calculateIntercept observations slope
    Int.floor ((10 - intercept) / slope) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_speed_theorem_l663_66398


namespace NUMINAMATH_CALUDE_percentage_of_300_l663_66388

/-- Calculates the percentage of a given amount -/
def percentage (percent : ℚ) (amount : ℚ) : ℚ :=
  (percent / 100) * amount

/-- Proves that 25% of Rs. 300 is equal to Rs. 75 -/
theorem percentage_of_300 : percentage 25 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_300_l663_66388


namespace NUMINAMATH_CALUDE_circles_intersection_l663_66353

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 10*y + 36 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 5)

-- Theorem statement
theorem circles_intersection :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y ↔ (x, y) = intersection_point) ∧
  (intersection_point.1 * intersection_point.2 = 15) :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_l663_66353


namespace NUMINAMATH_CALUDE_cubic_with_repeated_root_l663_66356

/-- Given a cubic polynomial 2x^3 + 8x^2 - 120x + k = 0 with a repeated root and positive k,
    prove that k = 6400/27 -/
theorem cubic_with_repeated_root (k : ℝ) : 
  (∃ x y : ℝ, (2 * x^3 + 8 * x^2 - 120 * x + k = 0) ∧ 
               (2 * y^3 + 8 * y^2 - 120 * y + k = 0) ∧ 
               (x ≠ y)) ∧
  (∃ z : ℝ, (2 * z^3 + 8 * z^2 - 120 * z + k = 0) ∧ 
            (∀ w : ℝ, 2 * w^3 + 8 * w^2 - 120 * w + k = 0 → w = z ∨ w = x ∨ w = y)) ∧
  (k > 0) →
  k = 6400 / 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_with_repeated_root_l663_66356


namespace NUMINAMATH_CALUDE_unique_prime_mersenne_sequence_l663_66352

theorem unique_prime_mersenne_sequence (n : ℕ+) : 
  (Nat.Prime (2^n.val - 1) ∧ 
   Nat.Prime (2^(n.val + 2) - 1) ∧ 
   ¬(7 ∣ (2^(n.val + 1) - 1))) ↔ 
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_mersenne_sequence_l663_66352


namespace NUMINAMATH_CALUDE_min_shading_for_symmetry_l663_66357

/-- Represents a triangular figure with some shaded triangles -/
structure TriangularFigure where
  total_triangles : Nat
  shaded_triangles : Nat
  h_shaded_le_total : shaded_triangles ≤ total_triangles

/-- Calculates the minimum number of additional triangles to shade for axial symmetry -/
def min_additional_shading (figure : TriangularFigure) : Nat :=
  sorry

/-- Theorem stating the minimum additional shading for the given problem -/
theorem min_shading_for_symmetry (figure : TriangularFigure) 
  (h_total : figure.total_triangles = 54)
  (h_some_shaded : figure.shaded_triangles > 0)
  (h_not_all_shaded : figure.shaded_triangles < 54) :
  min_additional_shading figure = 6 :=
sorry

end NUMINAMATH_CALUDE_min_shading_for_symmetry_l663_66357


namespace NUMINAMATH_CALUDE_multiply_by_seven_l663_66384

theorem multiply_by_seven (x : ℝ) (h : 8 * x = 64) : 7 * x = 56 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_l663_66384


namespace NUMINAMATH_CALUDE_third_digit_even_l663_66315

theorem third_digit_even (n : ℤ) : ∃ k : ℤ, (10*n + 5)^2 = 1000*k + 200*m + 25 ∧ m % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_third_digit_even_l663_66315


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l663_66317

/-- The eccentricity of a hyperbola with given properties is between 1 and 2√3/3 -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | b * x = a * y ∨ b * x = -a * y}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ (p : ℝ × ℝ), p ∈ asymptotes ∩ circle) →
  1 < e ∧ e < 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l663_66317


namespace NUMINAMATH_CALUDE_v_2023_equals_1_l663_66303

-- Define the function g
def g : ℕ → ℕ
| 1 => 3
| 2 => 4
| 3 => 2
| 4 => 1
| 5 => 5
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

-- Theorem statement
theorem v_2023_equals_1 : v 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_v_2023_equals_1_l663_66303


namespace NUMINAMATH_CALUDE_words_per_page_l663_66358

theorem words_per_page (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 154 →
  max_words_per_page = 120 →
  total_words_mod = 250 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 227 = total_words_mod % 227 ∧
    words_per_page = 49 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l663_66358


namespace NUMINAMATH_CALUDE_jelly_bean_count_l663_66395

/-- The number of red jelly beans in one bag -/
def red_in_bag : ℕ := 24

/-- The number of white jelly beans in one bag -/
def white_in_bag : ℕ := 18

/-- The number of bags needed to fill the fishbowl -/
def bags_to_fill : ℕ := 3

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white : ℕ := (red_in_bag + white_in_bag) * bags_to_fill

theorem jelly_bean_count : total_red_white = 126 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_count_l663_66395


namespace NUMINAMATH_CALUDE_expression_value_l663_66349

theorem expression_value (x y : ℝ) (h : x - 2*y + 3 = 0) : 1 - 2*x + 4*y = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l663_66349


namespace NUMINAMATH_CALUDE_equation_solution_l663_66391

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l663_66391


namespace NUMINAMATH_CALUDE_distribute_five_into_four_l663_66328

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else if k = 1 then 1
  else k

theorem distribute_five_into_four :
  distribute_objects 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_four_l663_66328


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l663_66318

theorem cement_mixture_weight (sand_ratio : ℚ) (water_ratio : ℚ) (gravel_weight : ℚ) 
  (h1 : sand_ratio = 1 / 3)
  (h2 : water_ratio = 1 / 2)
  (h3 : gravel_weight = 8) :
  ∃ (total_weight : ℚ), 
    sand_ratio * total_weight + water_ratio * total_weight + gravel_weight = total_weight ∧ 
    total_weight = 48 := by
sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l663_66318


namespace NUMINAMATH_CALUDE_ant_walk_length_l663_66334

theorem ant_walk_length (r₁ r₂ : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 15) : 
  let quarter_large := (1/4) * 2 * Real.pi * r₂
  let half_small := (1/2) * 2 * Real.pi * r₁
  let radial := r₂ - r₁
  quarter_large + half_small + 2 * radial = 12.5 * Real.pi + 20 := by
sorry

end NUMINAMATH_CALUDE_ant_walk_length_l663_66334


namespace NUMINAMATH_CALUDE_alloy_mixture_l663_66323

/-- Given two alloys with metal ratios m:n and p:q respectively, 
    this theorem proves the amounts of each alloy needed to create 1 kg 
    of a new alloy with equal parts of both metals. -/
theorem alloy_mixture (m n p q : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) :
  let x := (1 : ℝ) / 2 + (m * p - n * q) / (2 * (n * p - m * q))
  x * (n / (m + n)) + (1 - x) * (p / (p + q)) = 
  x * (m / (m + n)) + (1 - x) * (q / (p + q)) :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_l663_66323


namespace NUMINAMATH_CALUDE_abs_plus_square_zero_implies_product_l663_66326

theorem abs_plus_square_zero_implies_product (a b : ℝ) : 
  |a - 1| + (b + 2)^2 = 0 → a * b^a = -2 := by
sorry

end NUMINAMATH_CALUDE_abs_plus_square_zero_implies_product_l663_66326


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l663_66306

/-- Theorem: Volume ratio of water in a cone filled to 2/3 of its height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height : ℝ := 2 / 3 * h
  let water_radius : ℝ := 2 / 3 * r
  let cone_volume : ℝ := (1 / 3) * π * r^2 * h
  let water_volume : ℝ := (1 / 3) * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l663_66306


namespace NUMINAMATH_CALUDE_intersection_equals_subset_implies_a_values_l663_66308

def A (a : ℝ) : Set ℝ := {x | x - a = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equals_subset_implies_a_values (a : ℝ) 
  (h : A a ∩ B a = B a) : 
  a = 1 ∨ a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_subset_implies_a_values_l663_66308


namespace NUMINAMATH_CALUDE_binomial_18_choose_6_l663_66339

theorem binomial_18_choose_6 : Nat.choose 18 6 = 4765 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_6_l663_66339


namespace NUMINAMATH_CALUDE_son_age_l663_66369

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_l663_66369


namespace NUMINAMATH_CALUDE_x_value_proof_l663_66333

theorem x_value_proof (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l663_66333


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l663_66397

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Represents 1111 in any base -/
def digits1111 : List Nat := [1, 1, 1, 1]

/-- The main theorem -/
theorem smallest_base_perfect_square :
  (∀ b : Nat, b > 0 → b < 7 → ¬isPerfectSquare (toBase10 digits1111 b)) ∧
  isPerfectSquare (toBase10 digits1111 7) := by
  sorry

#check smallest_base_perfect_square

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l663_66397


namespace NUMINAMATH_CALUDE_complex_modulus_product_l663_66377

/-- Given a complex number with modulus √2018, prove that its product with its conjugate is 2018 -/
theorem complex_modulus_product (a b : ℝ) : 
  (Complex.abs (Complex.mk a b))^2 = 2018 → (a + Complex.I * b) * (a - Complex.I * b) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l663_66377


namespace NUMINAMATH_CALUDE_polynomial_root_mean_l663_66342

theorem polynomial_root_mean (a b c d k : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (k - a) * (k - b) * (k - c) * (k - d) = 4 →
  k = (a + b + c + d) / 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_mean_l663_66342


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l663_66399

theorem min_value_trig_expression (x : ℝ) :
  (Real.sin x)^8 + 16 * (Real.cos x)^8 + 1 ≥ 
  4.7692 * ((Real.sin x)^6 + 4 * (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l663_66399


namespace NUMINAMATH_CALUDE_right_triangle_trig_inequality_l663_66379

theorem right_triangle_trig_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : A < π/4) 
  (h3 : A + B + C = π) (h4 : C = π/2) : Real.cos B < Real.sin B := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_inequality_l663_66379


namespace NUMINAMATH_CALUDE_correct_operation_l663_66316

theorem correct_operation (a : ℝ) : 3 * a^2 - 4 * a^2 = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l663_66316


namespace NUMINAMATH_CALUDE_combined_shape_perimeter_l663_66378

/-- The perimeter of a combined shape of a right triangle and rectangle -/
theorem combined_shape_perimeter (a b c d : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 10) 
  (h4 : d^2 = a^2 + b^2) : a + b + c + d = 22 := by
  sorry

end NUMINAMATH_CALUDE_combined_shape_perimeter_l663_66378


namespace NUMINAMATH_CALUDE_quadratic_to_alternative_form_integer_values_iff_integer_coefficients_l663_66359

/-- Represents a quadratic expression Ax² + Bx + C -/
structure QuadraticExpression (α : Type) [Ring α] where
  A : α
  B : α
  C : α

/-- Represents the alternative form k(x(x-1)/2) + lx + m -/
structure AlternativeForm (α : Type) [Ring α] where
  k : α
  l : α
  m : α

/-- States that a quadratic expression can be written in the alternative form -/
theorem quadratic_to_alternative_form {α : Type} [Ring α] (q : QuadraticExpression α) :
  ∃ (a : AlternativeForm α), a.k = 2 * q.A ∧ a.l = q.A + q.B ∧ a.m = q.C :=
sorry

/-- States that the quadratic expression takes integer values for all integer x
    if and only if k, l, m in the alternative form are integers -/
theorem integer_values_iff_integer_coefficients (q : QuadraticExpression ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, q.A * x^2 + q.B * x + q.C = y) ↔
  (∃ (a : AlternativeForm ℤ), a.k = 2 * q.A ∧ a.l = q.A + q.B ∧ a.m = q.C) :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_alternative_form_integer_values_iff_integer_coefficients_l663_66359


namespace NUMINAMATH_CALUDE_outfits_count_l663_66368

/-- The number of possible outfits given the number of shirts, pants, and shoes. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (shoes : ℕ) : ℕ :=
  shirts * pants * shoes

/-- Theorem stating that the number of outfits from 4 shirts, 5 pants, and 2 shoes is 40. -/
theorem outfits_count : number_of_outfits 4 5 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l663_66368


namespace NUMINAMATH_CALUDE_water_problem_solution_l663_66361

def water_problem (total_water : ℕ) (car_water : ℕ) (num_cars : ℕ) (plant_water_diff : ℕ) : ℕ :=
  let car_total := car_water * num_cars
  let plant_water := car_total - plant_water_diff
  let used_water := car_total + plant_water
  let remaining_water := total_water - used_water
  remaining_water / 2

theorem water_problem_solution :
  water_problem 65 7 2 11 = 24 := by
  sorry

end NUMINAMATH_CALUDE_water_problem_solution_l663_66361


namespace NUMINAMATH_CALUDE_patches_in_unit_l663_66324

/-- The number of patches in a unit given cost price, selling price, and net profit -/
theorem patches_in_unit (cost_price selling_price net_profit : ℚ) : 
  cost_price = 1.25 → 
  selling_price = 12 → 
  net_profit = 1075 → 
  (net_profit / (selling_price - cost_price) : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_patches_in_unit_l663_66324


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l663_66387

theorem triangle_angle_sum (A B C : Real) (h1 : A = 30) (h2 : B = 50) :
  C = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l663_66387


namespace NUMINAMATH_CALUDE_smallest_scalene_perimeter_l663_66345

-- Define a scalene triangle with integer side lengths
def ScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b

-- Theorem statement
theorem smallest_scalene_perimeter :
  ∀ a b c : ℕ, ScaleneTriangle a b c → a + b + c ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_scalene_perimeter_l663_66345


namespace NUMINAMATH_CALUDE_chord_length_l663_66362

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l663_66362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l663_66307

/-- Proves that 1, 3, and 5 form a monotonically increasing arithmetic sequence with -1 and 7 -/
theorem arithmetic_sequence_proof : 
  let sequence := [-1, 1, 3, 5, 7]
  (∀ i : Fin 4, sequence[i] < sequence[i+1]) ∧ 
  (∃ d : ℤ, ∀ i : Fin 4, sequence[i+1] - sequence[i] = d) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l663_66307


namespace NUMINAMATH_CALUDE_fraction_comparison_l663_66331

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / (a - c) < a / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l663_66331


namespace NUMINAMATH_CALUDE_range_of_f_l663_66381

def f (x : ℤ) : ℤ := (x - 1)^2 + 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l663_66381


namespace NUMINAMATH_CALUDE_energy_bar_difference_l663_66302

theorem energy_bar_difference (older younger : ℕ) 
  (h1 : older = younger + 17) : 
  (older - 3) = (younger + 3) + 11 := by
  sorry

end NUMINAMATH_CALUDE_energy_bar_difference_l663_66302


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l663_66301

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℝ),
    x^4 + 5 = (x^2 - 4*x + 7) * q + r ∧
    r.degree < (x^2 - 4*x + 7).degree ∧
    r = 8*x - 58 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l663_66301


namespace NUMINAMATH_CALUDE_stating_botanical_garden_visitors_l663_66386

/-- Represents the growth rate of visitors in a botanical garden -/
def growth_rate_equation (x : ℝ) : Prop :=
  (1 + x)^2 = 3

/-- 
Theorem stating that the growth rate equation holds given the conditions:
- The number of visitors in March is three times that of January
- x is the average growth rate of visitors in February and March
-/
theorem botanical_garden_visitors (x : ℝ) 
  (h_march : ∃ (a : ℝ), a > 0 ∧ a * (1 + x)^2 = 3 * a) : 
  growth_rate_equation x := by
  sorry

end NUMINAMATH_CALUDE_stating_botanical_garden_visitors_l663_66386


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l663_66338

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 2 - 3*x - 7*x^3 + 12*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- Theorem: The value of c that makes h(x) a polynomial of degree 3 is -5/12 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -5/12 ∧ 
  (∀ (x : ℝ), h c x = 1 + (-12 - 3*c)*x + (3 - 0*c)*x^2 + (-4 - 7*c)*x^3) :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l663_66338


namespace NUMINAMATH_CALUDE_kim_exam_average_l663_66309

/-- Given Kim's five exam scores, prove that the average is 89.6 -/
theorem kim_exam_average :
  let scores : List ℝ := [92, 89, 90, 92, 85]
  (scores.sum / scores.length : ℝ) = 89.6 := by
  sorry

end NUMINAMATH_CALUDE_kim_exam_average_l663_66309


namespace NUMINAMATH_CALUDE_einstein_fundraising_l663_66383

/-- Einstein's fundraising problem -/
theorem einstein_fundraising 
  (goal : ℕ)
  (pizza_price potato_price soda_price : ℚ)
  (pizza_sold potato_sold soda_sold : ℕ) :
  goal = 500 ∧ 
  pizza_price = 12 ∧ 
  potato_price = 3/10 ∧ 
  soda_price = 2 ∧
  pizza_sold = 15 ∧ 
  potato_sold = 40 ∧ 
  soda_sold = 25 →
  (goal : ℚ) - (pizza_price * pizza_sold + potato_price * potato_sold + soda_price * soda_sold) = 258 :=
by sorry


end NUMINAMATH_CALUDE_einstein_fundraising_l663_66383


namespace NUMINAMATH_CALUDE_circle_area_increase_l663_66382

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l663_66382


namespace NUMINAMATH_CALUDE_kelly_games_to_give_away_l663_66392

/-- The number of games Kelly needs to give away to reach her desired number of games -/
def games_to_give_away (initial_games : ℕ) (desired_games : ℕ) : ℕ :=
  initial_games - desired_games

/-- Proof that Kelly needs to give away 15 games -/
theorem kelly_games_to_give_away :
  games_to_give_away 50 35 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_to_give_away_l663_66392


namespace NUMINAMATH_CALUDE_partnership_profit_l663_66311

/-- Calculates the total profit of a partnership given the investments, time periods, and one partner's profit. -/
def total_profit (a_investment : ℕ) (b_investment : ℕ) (a_period : ℕ) (b_period : ℕ) (b_profit : ℕ) : ℕ :=
  let profit_ratio := (a_investment * a_period) / (b_investment * b_period)
  let total_parts := profit_ratio + 1
  total_parts * b_profit

/-- Theorem stating that under the given conditions, the total profit is 42000. -/
theorem partnership_profit : 
  ∀ (b_investment : ℕ) (b_period : ℕ),
    b_investment > 0 → b_period > 0 →
    total_profit (3 * b_investment) b_investment (2 * b_period) b_period 6000 = 42000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l663_66311


namespace NUMINAMATH_CALUDE_total_crayons_l663_66370

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l663_66370


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l663_66364

/-- Given a line x = a (a > 0) tangent to the circle (x-1)^2 + y^2 = 4, prove a = 3 -/
theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 → 
  (∀ y : ℝ, (a - 1)^2 + y^2 ≥ 4) ∧ 
  (∃ y : ℝ, (a - 1)^2 + y^2 = 4) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l663_66364


namespace NUMINAMATH_CALUDE_last_match_wickets_specific_case_l663_66354

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  initialWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken in the last match -/
def lastMatchWickets (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the number of wickets in the last match is 8 -/
theorem last_match_wickets_specific_case :
  let stats : BowlerStats := {
    initialAverage := 12.4,
    initialWickets := 175,
    lastMatchRuns := 26,
    averageDecrease := 0.4
  }
  lastMatchWickets stats = 8 := by sorry

end NUMINAMATH_CALUDE_last_match_wickets_specific_case_l663_66354


namespace NUMINAMATH_CALUDE_price_reduction_proof_l663_66304

/-- Proves that the average percentage reduction for each of two consecutive price reductions from $25 to $16 is 20% -/
theorem price_reduction_proof (initial_price final_price : ℝ) 
  (h_initial : initial_price = 25)
  (h_final : final_price = 16) :
  ∃ (reduction_percent : ℝ), 
    reduction_percent = 0.2 ∧ 
    initial_price * (1 - reduction_percent)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l663_66304


namespace NUMINAMATH_CALUDE_infinite_sum_floor_floor_2x_l663_66372

/-- For any real number x, the sum of floor((x + 2^k) / 2^(k+1)) from k=0 to infinity is equal to floor(x). -/
theorem infinite_sum_floor (x : ℝ) : 
  (∑' k, ⌊(x + 2^k) / 2^(k+1)⌋) = ⌊x⌋ :=
by sorry

/-- For any real number x, floor(2x) = floor(x) + floor(x + 1/2). -/
theorem floor_2x (x : ℝ) : 
  ⌊2*x⌋ = ⌊x⌋ + ⌊x + 1/2⌋ :=
by sorry

end NUMINAMATH_CALUDE_infinite_sum_floor_floor_2x_l663_66372


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l663_66350

theorem floor_ceil_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l663_66350


namespace NUMINAMATH_CALUDE_cube_surface_area_l663_66347

theorem cube_surface_area (volume : ℝ) (surface_area : ℝ) : 
  volume = 343 → surface_area = 294 → 
  (∃ (side : ℝ), volume = side^3 ∧ surface_area = 6 * side^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l663_66347


namespace NUMINAMATH_CALUDE_not_enough_money_l663_66393

/-- The cost of a new smartwatch in rubles -/
def smartwatch_cost : ℕ := 2019

/-- The amount of money Namzhil has in rubles -/
def namzhil_money : ℕ := (500^2 + 4 * 500 + 3) * 498^2 - 500^2 * 503 * 497

/-- Theorem stating that Namzhil does not have enough money to buy the smartwatch -/
theorem not_enough_money : namzhil_money < smartwatch_cost := by
  sorry

end NUMINAMATH_CALUDE_not_enough_money_l663_66393


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l663_66390

theorem mod_equivalence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 27483 % 17 = n := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l663_66390


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l663_66322

theorem trigonometric_expression_equality :
  let sin30 := 1 / 2
  let cos30 := Real.sqrt 3 / 2
  let tan60 := Real.sqrt 3
  2 * sin30 + cos30 * tan60 = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l663_66322


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l663_66367

theorem cosine_sum_simplification :
  Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + 
  Real.cos (10 * Real.pi / 15) + Real.cos (14 * Real.pi / 15) = 
  (Real.sqrt 17 - 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l663_66367


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l663_66314

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line2D.passesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line2D.isPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line2D) 
  (point : Point2D) 
  (perp_line : Line2D) : 
  given_line.a = 1 → 
  given_line.b = 1 → 
  given_line.c = -3 →
  point.x = 2 →
  point.y = -1 →
  perp_line.a = 1 →
  perp_line.b = -1 →
  perp_line.c = -3 →
  perp_line.passesThroughPoint point ∧ 
  perp_line.isPerpendicular given_line := by
  sorry

#check perpendicular_line_through_point

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l663_66314


namespace NUMINAMATH_CALUDE_real_y_condition_l663_66305

theorem real_y_condition (x y : ℝ) : 
  (9 * y^2 - 6 * x * y + 2 * x + 7 = 0) → 
  (∃ (y : ℝ), 9 * y^2 - 6 * x * y + 2 * x + 7 = 0) ↔ (x ≤ -2 ∨ x ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l663_66305


namespace NUMINAMATH_CALUDE_when_you_rescind_price_is_85_l663_66371

/-- The price of a CD of "The Life Journey" -/
def life_journey_price : ℕ := 100

/-- The price of a CD of "A Day a Life" -/
def day_life_price : ℕ := 50

/-- The number of each CD type bought -/
def quantity : ℕ := 3

/-- The total amount spent -/
def total_spent : ℕ := 705

/-- The price of a CD of "When You Rescind" -/
def when_you_rescind_price : ℕ := 85

/-- Theorem stating that the price of "When You Rescind" CD is 85 -/
theorem when_you_rescind_price_is_85 :
  quantity * life_journey_price + quantity * day_life_price + quantity * when_you_rescind_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_when_you_rescind_price_is_85_l663_66371


namespace NUMINAMATH_CALUDE_cos_120_degrees_l663_66396

theorem cos_120_degrees : Real.cos (2 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l663_66396


namespace NUMINAMATH_CALUDE_area_of_curve_l663_66389

theorem area_of_curve (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 16 ∧ 
   A = Real.pi * (Real.sqrt ((x - 2)^2 + (y + 3)^2))^2 ∧
   x^2 + y^2 - 4*x + 6*y - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_area_of_curve_l663_66389


namespace NUMINAMATH_CALUDE_book_arrangement_and_selection_l663_66346

/-- Given 3 math books, 4 physics books, and 2 chemistry books, prove:
    1. The number of arrangements keeping books of the same subject together
    2. The number of ways to select exactly 2 math books, 2 physics books, and 1 chemistry book
    3. The number of ways to select 5 books with at least 1 math book -/
theorem book_arrangement_and_selection 
  (math_books : ℕ) (physics_books : ℕ) (chemistry_books : ℕ) 
  (h_math : math_books = 3) 
  (h_physics : physics_books = 4) 
  (h_chemistry : chemistry_books = 2) :
  (-- 1. Number of arrangements
   (Nat.factorial math_books) * (Nat.factorial physics_books) * 
   (Nat.factorial chemistry_books) * (Nat.factorial 3) = 1728) ∧ 
  (-- 2. Number of ways to select 2 math, 2 physics, 1 chemistry
   (Nat.choose math_books 2) * (Nat.choose physics_books 2) * 
   (Nat.choose chemistry_books 1) = 36) ∧
  (-- 3. Number of ways to select 5 books with at least 1 math
   (Nat.choose (math_books + physics_books + chemistry_books) 5) - 
   (Nat.choose (physics_books + chemistry_books) 5) = 120) := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_and_selection_l663_66346


namespace NUMINAMATH_CALUDE_tickets_to_buy_l663_66300

def ferris_wheel_cost : ℕ := 6
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 7
def antonieta_tickets : ℕ := 2

theorem tickets_to_buy : 
  ferris_wheel_cost + roller_coaster_cost + log_ride_cost - antonieta_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_tickets_to_buy_l663_66300


namespace NUMINAMATH_CALUDE_solve_for_b_l663_66312

theorem solve_for_b (b : ℝ) : 
  4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → b = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l663_66312


namespace NUMINAMATH_CALUDE_total_sum_calculation_l663_66380

/-- Given a sum to be divided among four parts in the ratio 5 : 9 : 6 : 5,
    if the sum of the first and third parts is $7022.222222222222,
    then the total sum is $15959.59595959596. -/
theorem total_sum_calculation (a b c d : ℝ) : 
  a / 5 = b / 9 ∧ a / 5 = c / 6 ∧ a / 5 = d / 5 →
  a + c = 7022.222222222222 →
  a + b + c + d = 15959.59595959596 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l663_66380


namespace NUMINAMATH_CALUDE_actual_height_is_236_l663_66394

/-- The actual height of a boy in a class, given the following conditions:
  * There are 35 boys in the class
  * The initial average height was calculated as 185 cm
  * One boy's height was wrongly written as 166 cm
  * The actual average height is 183 cm
-/
def actual_height : ℕ :=
  let num_boys : ℕ := 35
  let initial_avg : ℕ := 185
  let wrong_height : ℕ := 166
  let actual_avg : ℕ := 183
  let initial_total : ℕ := num_boys * initial_avg
  let actual_total : ℕ := num_boys * actual_avg
  let height_difference : ℕ := initial_total - actual_total
  wrong_height + height_difference

theorem actual_height_is_236 : actual_height = 236 := by
  sorry

end NUMINAMATH_CALUDE_actual_height_is_236_l663_66394


namespace NUMINAMATH_CALUDE_function_properties_l663_66351

/-- Given functions f and g with parameter a, proves properties about their extrema and monotonicity -/
theorem function_properties (a : ℝ) (h : a ≤ 0) :
  let f := fun x : ℝ ↦ Real.exp x + a * x
  let g := fun x : ℝ ↦ a * x - Real.log x
  -- The minimum of f occurs at ln(-a)
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = Real.log (-a)) ∧
  -- The minimum value of f is -a + a * ln(-a)
  (∃ (y_min : ℝ), ∀ (x : ℝ), f x ≥ y_min ∧ y_min = -a + a * Real.log (-a)) ∧
  -- f has no maximum value
  (¬∃ (y_max : ℝ), ∀ (x : ℝ), f x ≤ y_max) ∧
  -- f and g have the same monotonicity on some interval iff a ∈ (-∞, -1)
  (∃ (M : Set ℝ), (∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → (f x < f y ↔ g x < g y)) ↔ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l663_66351


namespace NUMINAMATH_CALUDE_right_angle_in_triangle_l663_66313

theorem right_angle_in_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = Real.pi) →
  -- Side lengths are positive
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Given conditions
  (Real.sin B = Real.sin (2 * A)) →
  (c = 2 * a) →
  -- Conclusion
  C = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_right_angle_in_triangle_l663_66313


namespace NUMINAMATH_CALUDE_octagon_square_ratio_l663_66321

theorem octagon_square_ratio :
  let octagons_per_row : ℕ := 5
  let octagon_rows : ℕ := 4
  let squares_per_row : ℕ := 4
  let square_rows : ℕ := 3
  let total_octagons : ℕ := octagons_per_row * octagon_rows
  let total_squares : ℕ := squares_per_row * square_rows
  (total_octagons : ℚ) / total_squares = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_octagon_square_ratio_l663_66321
