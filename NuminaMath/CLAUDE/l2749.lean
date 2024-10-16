import Mathlib

namespace NUMINAMATH_CALUDE_power_of_product_l2749_274945

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l2749_274945


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l2749_274979

theorem buffet_meal_combinations : ℕ := by
  -- Define the number of options for each food category
  let num_meats : ℕ := 4
  let num_vegetables : ℕ := 4
  let num_desserts : ℕ := 4
  let num_drinks : ℕ := 2

  -- Define the number of items Tyler chooses from each category
  let chosen_meats : ℕ := 2
  let chosen_vegetables : ℕ := 2
  let chosen_desserts : ℕ := 1
  let chosen_drinks : ℕ := 1

  -- Calculate the total number of meal combinations
  have h : (Nat.choose num_meats chosen_meats) * 
           (Nat.choose num_vegetables chosen_vegetables) * 
           num_desserts * num_drinks = 288 := by sorry

  exact 288

end NUMINAMATH_CALUDE_buffet_meal_combinations_l2749_274979


namespace NUMINAMATH_CALUDE_binomial_100_97_l2749_274995

theorem binomial_100_97 : Nat.choose 100 97 = 161700 := by
  sorry

end NUMINAMATH_CALUDE_binomial_100_97_l2749_274995


namespace NUMINAMATH_CALUDE_f_properties_l2749_274978

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x * Real.exp x else x^2 - 2*x + 1/2

theorem f_properties :
  (∀ x, x ≤ 0 → (deriv f) x = 2 * (1 + x) * Real.exp x) ∧
  (∀ x, x > 0 → (deriv f) x = 2*x - 2) ∧
  ((deriv f) (-2) = -2 / Real.exp 2) ∧
  (∀ x, f x ≥ -2 / Real.exp 1) ∧
  (∃ x, f x = -2 / Real.exp 1) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2749_274978


namespace NUMINAMATH_CALUDE_inscribed_squares_side_length_l2749_274997

/-- Right triangle ABC with two inscribed squares -/
structure RightTriangleWithSquares where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side AC (hypotenuse) -/
  ac : ℝ
  /-- Side length of the inscribed squares -/
  s : ℝ
  /-- AB = 6 -/
  ab_eq : ab = 6
  /-- BC = 8 -/
  bc_eq : bc = 8
  /-- AC = 10 -/
  ac_eq : ac = 10
  /-- Pythagorean theorem holds -/
  pythagorean : ab ^ 2 + bc ^ 2 = ac ^ 2
  /-- The two squares do not overlap -/
  non_overlapping : 2 * s ≤ (ab * bc) / ac

/-- The side length of each inscribed square is 2.4 -/
theorem inscribed_squares_side_length (t : RightTriangleWithSquares) : t.s = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_side_length_l2749_274997


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2749_274990

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |f a b c x| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |2 * a * x + b| ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2749_274990


namespace NUMINAMATH_CALUDE_original_recipe_juice_l2749_274917

/-- The amount of key lime juice in tablespoons that Audrey uses -/
def audrey_juice : ℕ := 8

/-- The factor by which Audrey increases the amount of key lime juice -/
def increase_factor : ℕ := 2

/-- The amount of juice in tablespoons that one key lime yields -/
def juice_per_lime : ℕ := 1

theorem original_recipe_juice : 
  audrey_juice / increase_factor = 4 := by sorry

end NUMINAMATH_CALUDE_original_recipe_juice_l2749_274917


namespace NUMINAMATH_CALUDE_probability_is_three_fiftieths_l2749_274940

/-- Represents a 5x5x5 cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)

/-- Represents the count of different types of unit cubes -/
structure CubeCounts :=
  (two_blue : ℕ)
  (unpainted : ℕ)

/-- Calculates the probability of selecting specific cube types -/
def probability_two_blue_and_unpainted (cube : PaintedCube) (counts : CubeCounts) : ℚ :=
  let total_combinations := (cube.total_cubes.choose 2 : ℚ)
  let favorable_outcomes := (counts.two_blue * counts.unpainted : ℚ)
  favorable_outcomes / total_combinations

/-- The main theorem to be proved -/
theorem probability_is_three_fiftieths (cube : PaintedCube) (counts : CubeCounts) : 
  cube.size = 5 ∧ 
  cube.total_cubes = 125 ∧ 
  cube.blue_faces = 2 ∧ 
  cube.red_faces = 1 ∧
  counts.two_blue = 9 ∧
  counts.unpainted = 51 →
  probability_two_blue_and_unpainted cube counts = 3 / 50 :=
sorry

end NUMINAMATH_CALUDE_probability_is_three_fiftieths_l2749_274940


namespace NUMINAMATH_CALUDE_power_of_power_l2749_274929

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2749_274929


namespace NUMINAMATH_CALUDE_bernardo_wins_smallest_number_l2749_274935

theorem bernardo_wins_smallest_number : ∃ M : ℕ, 
  M ≤ 799 ∧ 
  (∀ k : ℕ, k < M → 
    (2 * k ≤ 800 ∧ 
     2 * k + 70 ≤ 800 ∧ 
     4 * k + 140 ≤ 800 ∧ 
     4 * k + 210 ≤ 800 ∧ 
     8 * k + 420 ≤ 800) → 
    8 * k + 490 ≤ 800) ∧
  2 * M ≤ 800 ∧ 
  2 * M + 70 ≤ 800 ∧ 
  4 * M + 140 ≤ 800 ∧ 
  4 * M + 210 ≤ 800 ∧ 
  8 * M + 420 ≤ 800 ∧ 
  8 * M + 490 > 800 ∧
  M = 37 :=
by sorry

end NUMINAMATH_CALUDE_bernardo_wins_smallest_number_l2749_274935


namespace NUMINAMATH_CALUDE_greatest_non_sum_of_complex_l2749_274943

/-- A natural number is complex if it has at least two different prime divisors. -/
def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

/-- A natural number is representable as the sum of two complex numbers. -/
def is_sum_of_complex (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ n = a + b

/-- 23 is the greatest natural number that cannot be represented as the sum of two complex numbers. -/
theorem greatest_non_sum_of_complex : ∀ n : ℕ, n > 23 → is_sum_of_complex n ∧ ¬is_sum_of_complex 23 :=
sorry

end NUMINAMATH_CALUDE_greatest_non_sum_of_complex_l2749_274943


namespace NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l2749_274952

theorem multiply_three_six_and_quarter : 3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_six_and_quarter_l2749_274952


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l2749_274909

/-- Given a segment AB with endpoints A(2, -2) and B(14, 4), extended through B to point C
    such that BC = 1/3 * AB, prove that the coordinates of point C are (18, 6). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (2, -2) →
  B = (14, 4) →
  C.1 - B.1 = (1/3) * (B.1 - A.1) →
  C.2 - B.2 = (1/3) * (B.2 - A.2) →
  C = (18, 6) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l2749_274909


namespace NUMINAMATH_CALUDE_jim_toads_difference_l2749_274950

theorem jim_toads_difference (tim_toads sarah_toads : ℕ) 
  (h1 : tim_toads = 30)
  (h2 : sarah_toads = 100)
  (h3 : sarah_toads = 2 * jim_toads)
  (h4 : jim_toads > tim_toads) : 
  jim_toads - tim_toads = 20 := by
sorry

end NUMINAMATH_CALUDE_jim_toads_difference_l2749_274950


namespace NUMINAMATH_CALUDE_M_values_l2749_274946

theorem M_values (a b : ℚ) (hab : a * b ≠ 0) :
  let M := (2 * abs a) / a + (3 * b) / abs b
  M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 := by sorry

end NUMINAMATH_CALUDE_M_values_l2749_274946


namespace NUMINAMATH_CALUDE_pushup_difference_l2749_274900

theorem pushup_difference (zachary_pushups john_pushups : ℕ) 
  (h1 : zachary_pushups = 51)
  (h2 : john_pushups = 69)
  (h3 : ∃ david_pushups : ℕ, david_pushups = john_pushups + 4) :
  ∃ david_pushups : ℕ, david_pushups - zachary_pushups = 22 :=
by sorry

end NUMINAMATH_CALUDE_pushup_difference_l2749_274900


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2749_274901

theorem inequality_solution_set (x : ℝ) : 
  (Set.Ioo (-3 : ℝ) 1 : Set ℝ) = {x | (1 - x) * (3 + x) > 0} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2749_274901


namespace NUMINAMATH_CALUDE_gcd_problem_l2749_274956

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1061) :
  Int.gcd (3 * b^2 + 41 * b + 96) (b + 17) = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2749_274956


namespace NUMINAMATH_CALUDE_box_2_neg1_3_equals_1_l2749_274936

def box (a b c : ℤ) : ℚ :=
  let k : ℤ := 2
  (k : ℚ) * (a : ℚ) ^ b - (b : ℚ) ^ c + (c : ℚ) ^ (a - k)

theorem box_2_neg1_3_equals_1 :
  box 2 (-1) 3 = 1 := by sorry

end NUMINAMATH_CALUDE_box_2_neg1_3_equals_1_l2749_274936


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l2749_274923

theorem cubic_root_equation_solution :
  ∃ x : ℝ, (((3 - x) ^ (1/3 : ℝ)) + ((x - 1) ^ (1/2 : ℝ)) = 2) ∧ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l2749_274923


namespace NUMINAMATH_CALUDE_diameter_is_65_l2749_274932

/-- Represents a circle with a diameter and a perpendicular chord --/
structure Circle where
  diameter : ℕ
  chord : ℕ
  is_two_digit : 10 ≤ diameter ∧ diameter < 100
  is_reversed : chord = (diameter % 10) * 10 + (diameter / 10)

/-- The distance from the center to the intersection of the chord and diameter --/
def center_to_intersection (c : Circle) : ℚ :=
  let r := c.diameter / 2
  let h := c.chord / 2
  ((r * r - h * h : ℚ) / (r * r)).sqrt

theorem diameter_is_65 (c : Circle) 
  (h_rational : ∃ (q : ℚ), center_to_intersection c = q) :
  c.diameter = 65 := by
  sorry

#check diameter_is_65

end NUMINAMATH_CALUDE_diameter_is_65_l2749_274932


namespace NUMINAMATH_CALUDE_bridget_apples_l2749_274918

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 4 + 6 = x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l2749_274918


namespace NUMINAMATH_CALUDE_cyclist_journey_l2749_274960

theorem cyclist_journey 
  (v : ℝ) -- original speed in mph
  (t : ℝ) -- original time in hours
  (d : ℝ) -- distance in miles
  (h₁ : d = v * t) -- distance = speed * time
  (h₂ : d = (v + 1/3) * (3/4 * t)) -- increased speed condition
  (h₃ : d = (v - 1/3) * (t + 3/2)) -- decreased speed condition
  : v = 1 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_journey_l2749_274960


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l2749_274986

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l2749_274986


namespace NUMINAMATH_CALUDE_power_of_product_l2749_274905

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l2749_274905


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l2749_274903

theorem cubic_equation_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a = 12 →
  b^3 - 6*b^2 + 11*b = 12 →
  c^3 - 6*c^2 + 11*c = 12 →
  a*b/c + b*c/a + c*a/b = -23/12 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l2749_274903


namespace NUMINAMATH_CALUDE_average_weight_of_three_l2749_274965

/-- Given the weights of three people with specific relationships, prove their average weight. -/
theorem average_weight_of_three (ishmael ponce jalen : ℝ) : 
  ishmael = ponce + 20 →
  ponce = jalen - 10 →
  jalen = 160 →
  (ishmael + ponce + jalen) / 3 = 160 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_three_l2749_274965


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2749_274938

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 10 → x * y = 200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2749_274938


namespace NUMINAMATH_CALUDE_system_solution_l2749_274999

theorem system_solution : ∃ (X Y : ℝ), 
  (X + (X + 2*Y) / (X^2 + Y^2) = 2 ∧ 
   Y + (2*X - Y) / (X^2 + Y^2) = 0) ↔ 
  ((X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2749_274999


namespace NUMINAMATH_CALUDE_exists_number_not_divisible_by_3_with_digit_product_3_l2749_274941

def numbers : List Nat := [4621, 4631, 4641, 4651, 4661]

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def is_divisible_by_3 (n : Nat) : Prop :=
  n % 3 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem exists_number_not_divisible_by_3_with_digit_product_3 :
  ∃ n ∈ numbers, ¬(is_divisible_by_3 n) ∧ (units_digit n) * (tens_digit n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_not_divisible_by_3_with_digit_product_3_l2749_274941


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l2749_274989

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 : 
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l2749_274989


namespace NUMINAMATH_CALUDE_line_parallel_plane_not_all_lines_l2749_274993

/-- A plane in 3D space -/
structure Plane3D where
  -- Define plane properties here
  mk :: -- Constructor

/-- A line in 3D space -/
structure Line3D where
  -- Define line properties here
  mk :: -- Constructor

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when a line is in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem line_parallel_plane_not_all_lines 
  (p : Plane3D) : 
  ∃ (l : Line3D), parallel_line_plane l p ∧ 
  ∃ (m : Line3D), line_in_plane m p ∧ ¬parallel_lines l m :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_not_all_lines_l2749_274993


namespace NUMINAMATH_CALUDE_explosion_hyperbola_eccentricity_l2749_274919

/-- The eccentricity of a hyperbola formed by an explosion point, given two sentry posts
    1400m apart, a time difference of 3s in hearing the explosion, and a speed of sound of 340m/s -/
theorem explosion_hyperbola_eccentricity
  (distance_between_posts : ℝ)
  (time_difference : ℝ)
  (speed_of_sound : ℝ)
  (h_distance : distance_between_posts = 1400)
  (h_time : time_difference = 3)
  (h_speed : speed_of_sound = 340) :
  let c : ℝ := distance_between_posts / 2
  let a : ℝ := time_difference * speed_of_sound / 2
  c / a = 70 / 51 :=
by sorry

end NUMINAMATH_CALUDE_explosion_hyperbola_eccentricity_l2749_274919


namespace NUMINAMATH_CALUDE_score_difference_l2749_274914

def score_60_percent : ℝ := 0.15
def score_75_percent : ℝ := 0.20
def score_85_percent : ℝ := 0.40
def score_95_percent : ℝ := 1 - (score_60_percent + score_75_percent + score_85_percent)

def mean_score : ℝ :=
  60 * score_60_percent + 75 * score_75_percent + 85 * score_85_percent + 95 * score_95_percent

def median_score : ℝ := 85

theorem score_difference : median_score - mean_score = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l2749_274914


namespace NUMINAMATH_CALUDE_fraction_equality_l2749_274996

theorem fraction_equality : (45 : ℚ) / (8 - 3 / 7) = 315 / 53 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2749_274996


namespace NUMINAMATH_CALUDE_triangle_problem_l2749_274955

theorem triangle_problem (a b c A B C : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A) →
  (a = 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (A = π/3 ∧ b = 2 ∧ c = 2) := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2749_274955


namespace NUMINAMATH_CALUDE_problem_statement_l2749_274992

theorem problem_statement (p q r u v w : ℝ) 
  (eq1 : 17 * u + q * v + r * w = 0)
  (eq2 : p * u + 29 * v + r * w = 0)
  (eq3 : p * u + q * v + 56 * w = 0)
  (h1 : p ≠ 17)
  (h2 : u ≠ 0) :
  p / (p - 17) + q / (q - 29) + r / (r - 56) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2749_274992


namespace NUMINAMATH_CALUDE_contest_prize_distribution_l2749_274963

theorem contest_prize_distribution (total_prize : ℕ) (total_winners : ℕ) 
  (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ) 
  (h1 : total_prize = 800) (h2 : total_winners = 18) 
  (h3 : first_prize = 200) (h4 : second_prize = 150) (h5 : third_prize = 120) :
  let remaining_prize := total_prize - (first_prize + second_prize + third_prize)
  let remaining_winners := total_winners - 3
  remaining_prize / remaining_winners = 22 := by
sorry

end NUMINAMATH_CALUDE_contest_prize_distribution_l2749_274963


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2749_274906

/-- A three-digit number composed of distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones
  valid_range : hundreds ∈ Finset.range 10 ∧ tens ∈ Finset.range 10 ∧ ones ∈ Finset.range 10

/-- Check if one digit is the average of the other two -/
def has_average_digit (n : ThreeDigitNumber) : Prop :=
  2 * n.hundreds = n.tens + n.ones ∨
  2 * n.tens = n.hundreds + n.ones ∨
  2 * n.ones = n.hundreds + n.tens

/-- Check if the sum of digits is divisible by 3 -/
def sum_divisible_by_three (n : ThreeDigitNumber) : Prop :=
  (n.hundreds + n.tens + n.ones) % 3 = 0

/-- The set of all valid three-digit numbers satisfying the conditions -/
def valid_numbers : Finset ThreeDigitNumber :=
  sorry

theorem count_valid_numbers : valid_numbers.card = 160 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2749_274906


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l2749_274980

theorem price_ratio_theorem (cost : ℝ) (price1 price2 : ℝ) 
  (h1 : price1 = cost * (1 + 0.35))
  (h2 : price2 = cost * (1 - 0.10)) :
  price2 / price1 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l2749_274980


namespace NUMINAMATH_CALUDE_exact_time_solution_l2749_274920

def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def hour_hand_start : ℝ := 270

def clock_problem (t : ℝ) : Prop :=
  0 ≤ t ∧ t < 60 ∧
  |minute_hand_speed * (t + 5) - (hour_hand_start + hour_hand_speed * (t - 2))| = 180

theorem exact_time_solution :
  ∃ t : ℝ, clock_problem t ∧ (21 : ℝ) < t ∧ t < 22 :=
sorry

end NUMINAMATH_CALUDE_exact_time_solution_l2749_274920


namespace NUMINAMATH_CALUDE_logarithm_equations_solutions_l2749_274933

theorem logarithm_equations_solutions :
  (∀ x : ℝ, x^2 - 1 > 0 ∧ x^2 - 1 ≠ 1 ∧ x^3 + 6 > 0 ∧ x^3 + 6 = 4*x^2 - x →
    x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 - 4 > 0 ∧ x^3 + x > 0 ∧ x^3 + x ≠ 1 ∧ x^3 + x = 4*x^2 - 6 →
    x = 3) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equations_solutions_l2749_274933


namespace NUMINAMATH_CALUDE_student_contribution_l2749_274908

theorem student_contribution 
  (total_raised : ℕ) 
  (num_students : ℕ) 
  (cost_per_student : ℕ) 
  (remaining_funds : ℕ) : 
  total_raised = 50 → 
  num_students = 20 → 
  cost_per_student = 7 → 
  remaining_funds = 10 → 
  (total_raised - remaining_funds) / num_students = 5 :=
by sorry

end NUMINAMATH_CALUDE_student_contribution_l2749_274908


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l2749_274937

/-- The ratio of a regular pentagon's side length to a rectangle's width, 
    given that they have the same perimeter and the rectangle's length is twice its width -/
theorem pentagon_rectangle_ratio (perimeter : ℝ) : 
  perimeter > 0 → 
  (5 : ℝ) * (perimeter / 5) / (perimeter / 6) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l2749_274937


namespace NUMINAMATH_CALUDE_five_items_four_bags_l2749_274967

/-- The number of ways to distribute n distinct objects into k indistinguishable containers -/
def distributeObjects (n k : ℕ) : ℕ := sorry

/-- There are 5 distinct items and 4 identical bags -/
theorem five_items_four_bags : distributeObjects 5 4 = 36 := by sorry

end NUMINAMATH_CALUDE_five_items_four_bags_l2749_274967


namespace NUMINAMATH_CALUDE_max_ratio_three_digit_number_to_digit_sum_l2749_274961

theorem max_ratio_three_digit_number_to_digit_sum :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c : ℚ) / (a + b + c) ≤ 100 ∧
    ∃ (a₀ b₀ c₀ : ℕ),
      1 ≤ a₀ ∧ a₀ ≤ 9 ∧
      0 ≤ b₀ ∧ b₀ ≤ 9 ∧
      0 ≤ c₀ ∧ c₀ ≤ 9 ∧
      (100 * a₀ + 10 * b₀ + c₀ : ℚ) / (a₀ + b₀ + c₀) = 100 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_three_digit_number_to_digit_sum_l2749_274961


namespace NUMINAMATH_CALUDE_no_real_solutions_l2749_274964

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2749_274964


namespace NUMINAMATH_CALUDE_binomial_sum_odd_terms_l2749_274971

theorem binomial_sum_odd_terms (n : ℕ) (h : n > 0) (h_equal : Nat.choose n 4 = Nat.choose n 6) :
  (Finset.range ((n + 1) / 2)).sum (fun k => Nat.choose n (2 * k)) = 2^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_odd_terms_l2749_274971


namespace NUMINAMATH_CALUDE_edge_stop_probability_l2749_274928

-- Define the grid size
def gridSize : Nat := 4

-- Define a position on the grid
structure Position where
  x : Nat
  y : Nat
  deriving Repr

-- Define the possible directions
inductive Direction
  | Up
  | Down
  | Left
  | Right

-- Define whether a position is on the edge
def isEdge (pos : Position) : Bool :=
  pos.x == 1 || pos.x == gridSize || pos.y == 1 || pos.y == gridSize

-- Define the next position after a move, with wrap-around
def nextPosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Up => ⟨pos.x, if pos.y == gridSize then 1 else pos.y + 1⟩
  | Direction.Down => ⟨pos.x, if pos.y == 1 then gridSize else pos.y - 1⟩
  | Direction.Left => ⟨if pos.x == 1 then gridSize else pos.x - 1, pos.y⟩
  | Direction.Right => ⟨if pos.x == gridSize then 1 else pos.x + 1, pos.y⟩

-- Define the probability of stopping at an edge within n hops
def probStopAtEdge (start : Position) (n : Nat) : Real :=
  sorry

-- Theorem statement
theorem edge_stop_probability :
  probStopAtEdge ⟨2, 1⟩ 5 =
    probStopAtEdge ⟨2, 1⟩ 1 +
    probStopAtEdge ⟨2, 1⟩ 2 +
    probStopAtEdge ⟨2, 1⟩ 3 +
    probStopAtEdge ⟨2, 1⟩ 4 +
    probStopAtEdge ⟨2, 1⟩ 5 :=
  sorry

end NUMINAMATH_CALUDE_edge_stop_probability_l2749_274928


namespace NUMINAMATH_CALUDE_sherry_banana_bread_l2749_274973

/-- Calculates the number of bananas needed for a given number of loaves -/
def bananas_needed (total_loaves : ℕ) (loaves_per_batch : ℕ) (bananas_per_batch : ℕ) : ℕ :=
  (total_loaves / loaves_per_batch) * bananas_per_batch

theorem sherry_banana_bread (total_loaves : ℕ) (loaves_per_batch : ℕ) (bananas_per_batch : ℕ) 
  (h1 : total_loaves = 99)
  (h2 : loaves_per_batch = 3)
  (h3 : bananas_per_batch = 1) :
  bananas_needed total_loaves loaves_per_batch bananas_per_batch = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_sherry_banana_bread_l2749_274973


namespace NUMINAMATH_CALUDE_square_root_equals_seven_l2749_274957

theorem square_root_equals_seven (m : ℝ) : (∀ x : ℝ, x ^ 2 = m ↔ x = 7 ∨ x = -7) → m = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equals_seven_l2749_274957


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2749_274954

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 ∧
  sin α * cos α + cos α ^ 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2749_274954


namespace NUMINAMATH_CALUDE_sin_double_plus_sin_squared_l2749_274972

theorem sin_double_plus_sin_squared (α : Real) (h : Real.tan α = 1/2) :
  Real.sin (2 * α) + Real.sin α ^ 2 = 1 := by sorry

end NUMINAMATH_CALUDE_sin_double_plus_sin_squared_l2749_274972


namespace NUMINAMATH_CALUDE_cartons_packed_l2749_274970

theorem cartons_packed (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_carton : ℕ) 
  (h1 : total_cups = 768) 
  (h2 : cups_per_box = 12) 
  (h3 : boxes_per_carton = 8) : 
  total_cups / (cups_per_box * boxes_per_carton) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cartons_packed_l2749_274970


namespace NUMINAMATH_CALUDE_fraction_expression_equality_l2749_274942

theorem fraction_expression_equality : 
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 9) = 1548 / 805 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_equality_l2749_274942


namespace NUMINAMATH_CALUDE_scientific_notation_of_218000_l2749_274987

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_218000 :
  toScientificNotation 218000 = ScientificNotation.mk 2.18 5 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_218000_l2749_274987


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2749_274915

theorem quadratic_one_solution (n : ℝ) : 
  (n > 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2749_274915


namespace NUMINAMATH_CALUDE_pentagon_vertex_c_y_coordinate_l2749_274934

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def has_vertical_symmetry (p : Pentagon) : Prop := sorry

/-- The main theorem -/
theorem pentagon_vertex_c_y_coordinate
  (p : Pentagon)
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 5))
  (h3 : ∃ y, p.C = (2.5, y))
  (h4 : p.D = (5, 5))
  (h5 : p.E = (5, 0))
  (h6 : has_vertical_symmetry p)
  (h7 : area p = 50)
  : p.C.2 = 15 := by sorry


end NUMINAMATH_CALUDE_pentagon_vertex_c_y_coordinate_l2749_274934


namespace NUMINAMATH_CALUDE_not_sum_to_seven_l2749_274962

def pairs : List (Int × Int) := [(4, 3), (-1, 8), (10, -2), (2, 5), (3, 5)]

def sum_to_seven (pair : Int × Int) : Bool :=
  pair.1 + pair.2 = 7

theorem not_sum_to_seven : 
  ∀ (pair : Int × Int), 
    pair ∈ pairs → 
      (¬(sum_to_seven pair) ↔ (pair = (10, -2) ∨ pair = (3, 5))) := by
  sorry

#eval pairs.filter (λ pair => ¬(sum_to_seven pair))

end NUMINAMATH_CALUDE_not_sum_to_seven_l2749_274962


namespace NUMINAMATH_CALUDE_nonnegative_integer_solutions_x_squared_eq_6x_l2749_274913

theorem nonnegative_integer_solutions_x_squared_eq_6x :
  ∃! n : ℕ, (∃ s : Finset ℕ, s.card = n ∧
    ∀ x : ℕ, x ∈ s ↔ x^2 = 6*x) ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_nonnegative_integer_solutions_x_squared_eq_6x_l2749_274913


namespace NUMINAMATH_CALUDE_minimum_guests_l2749_274983

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 411) (h2 : max_per_guest = 2.5) :
  ⌈total_food / max_per_guest⌉ = 165 := by
  sorry

end NUMINAMATH_CALUDE_minimum_guests_l2749_274983


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2749_274930

/-- The units digit of m^2 + 2^m is 7, where m = 2021^2 + 3^2021 -/
theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : m = 2021^2 + 3^2021 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l2749_274930


namespace NUMINAMATH_CALUDE_sum_of_series_l2749_274922

/-- The sum of the infinite series ∑(n=1 to ∞) (4n+1)/3^n is equal to 7/2 -/
theorem sum_of_series : ∑' n : ℕ, (4 * n + 1 : ℝ) / 3^n = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_l2749_274922


namespace NUMINAMATH_CALUDE_statement_II_must_be_true_l2749_274921

-- Define the possible digits
inductive Digit
| two
| three
| five
| six
| other

-- Define the statements
def statement_I (d : Digit) : Prop := d = Digit.two
def statement_II (d : Digit) : Prop := d ≠ Digit.three
def statement_III (d : Digit) : Prop := d = Digit.five
def statement_IV (d : Digit) : Prop := d ≠ Digit.six

-- Define the problem conditions
def conditions (d : Digit) : Prop :=
  ∃ (s1 s2 s3 : Prop),
    (s1 ∧ s2 ∧ s3) ∧
    (s1 = statement_I d ∨ s1 = statement_II d ∨ s1 = statement_III d ∨ s1 = statement_IV d) ∧
    (s2 = statement_I d ∨ s2 = statement_II d ∨ s2 = statement_III d ∨ s2 = statement_IV d) ∧
    (s3 = statement_I d ∨ s3 = statement_II d ∨ s3 = statement_III d ∨ s3 = statement_IV d) ∧
    (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3)

-- Theorem: Given the conditions, Statement II must be true
theorem statement_II_must_be_true :
  ∀ d : Digit, conditions d → statement_II d :=
by
  sorry

end NUMINAMATH_CALUDE_statement_II_must_be_true_l2749_274921


namespace NUMINAMATH_CALUDE_complex_power_difference_l2749_274944

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : 
  (1 + i)^18 - (1 - i)^18 = 1024 * i :=
sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2749_274944


namespace NUMINAMATH_CALUDE_greatest_number_l2749_274994

theorem greatest_number : 
  let a := 1000 + 0.01
  let b := 1000 * 0.01
  let c := 1000 / 0.01
  let d := 0.01 / 1000
  let e := 1000 - 0.01
  (c > a) ∧ (c > b) ∧ (c > d) ∧ (c > e) := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l2749_274994


namespace NUMINAMATH_CALUDE_digit_2015_is_zero_l2749_274926

/-- A sequence formed by arranging all positive integers in increasing order -/
def integer_sequence : ℕ → ℕ := sorry

/-- The nth digit in the integer sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem: If the 11th digit in the integer sequence is 0, then the 2015th digit is also 0 -/
theorem digit_2015_is_zero (h : nth_digit 11 = 0) : nth_digit 2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_2015_is_zero_l2749_274926


namespace NUMINAMATH_CALUDE_charity_book_donation_l2749_274976

theorem charity_book_donation (initial_books : ℕ) (books_per_donation : ℕ) 
  (borrowed_books : ℕ) (final_books : ℕ) : 
  initial_books = 300 →
  books_per_donation = 5 →
  borrowed_books = 140 →
  final_books = 210 →
  (final_books + borrowed_books - initial_books) / books_per_donation = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_book_donation_l2749_274976


namespace NUMINAMATH_CALUDE_total_fruits_eaten_l2749_274985

theorem total_fruits_eaten (sophie_oranges_per_day : ℕ) (hannah_grapes_per_day : ℕ) (days : ℕ) :
  sophie_oranges_per_day = 20 →
  hannah_grapes_per_day = 40 →
  days = 30 →
  sophie_oranges_per_day * days + hannah_grapes_per_day * days = 1800 :=
by sorry

end NUMINAMATH_CALUDE_total_fruits_eaten_l2749_274985


namespace NUMINAMATH_CALUDE_interesting_iff_prime_power_l2749_274969

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧
  ∀ x y : ℕ, (Nat.gcd x n ≠ 1 ∧ Nat.gcd y n ≠ 1) → Nat.gcd (x + y) n ≠ 1

theorem interesting_iff_prime_power (n : ℕ) :
  is_interesting n ↔ ∃ (p : ℕ) (s : ℕ), Nat.Prime p ∧ s > 0 ∧ n = p^s :=
sorry

end NUMINAMATH_CALUDE_interesting_iff_prime_power_l2749_274969


namespace NUMINAMATH_CALUDE_sqrt_180_simplified_l2749_274947

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplified_l2749_274947


namespace NUMINAMATH_CALUDE_range_of_a_l2749_274998

theorem range_of_a (a : ℝ) : a > 0 →
  (((∀ x y : ℝ, x < y → a^x > a^y) ↔ ¬(∀ x : ℝ, x^2 - 3*a*x + 1 > 0)) ↔
   (2/3 ≤ a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2749_274998


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l2749_274931

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - (shoes_lost + 1) / 2

/-- Theorem stating that with 25 initial pairs and 9 shoes lost,
    the maximum number of complete pairs remaining is 20. -/
theorem max_pairs_after_loss : max_remaining_pairs 25 9 = 20 := by
  sorry

#eval max_remaining_pairs 25 9

end NUMINAMATH_CALUDE_max_pairs_after_loss_l2749_274931


namespace NUMINAMATH_CALUDE_perfect_squares_solution_l2749_274968

theorem perfect_squares_solution (x y : ℤ) :
  (∃ a : ℤ, x + y = a^2) →
  (∃ b : ℤ, 2*x + 3*y = b^2) →
  (∃ c : ℤ, 3*x + y = c^2) →
  x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_solution_l2749_274968


namespace NUMINAMATH_CALUDE_keaton_orange_harvest_frequency_l2749_274911

/-- Represents Keaton's farm earnings and harvest information -/
structure FarmData where
  yearly_earnings : ℕ
  apple_harvest_interval : ℕ
  apple_harvest_value : ℕ
  orange_harvest_value : ℕ

/-- Calculates the frequency of orange harvests in months -/
def orange_harvest_frequency (data : FarmData) : ℕ :=
  12 / (data.yearly_earnings - (12 / data.apple_harvest_interval * data.apple_harvest_value)) / data.orange_harvest_value

/-- Theorem stating that Keaton's orange harvest frequency is 2 months -/
theorem keaton_orange_harvest_frequency :
  orange_harvest_frequency ⟨420, 3, 30, 50⟩ = 2 := by
  sorry

end NUMINAMATH_CALUDE_keaton_orange_harvest_frequency_l2749_274911


namespace NUMINAMATH_CALUDE_john_lewis_meeting_point_l2749_274977

/-- Represents the journey between two cities --/
structure Journey where
  distance : ℝ
  johnSpeed : ℝ
  lewisOutboundSpeed : ℝ
  lewisReturnSpeed : ℝ
  johnBreakFrequency : ℝ
  johnBreakDuration : ℝ
  lewisBreakFrequency : ℝ
  lewisBreakDuration : ℝ

/-- Calculates the meeting point of John and Lewis --/
def meetingPoint (j : Journey) : ℝ :=
  sorry

/-- Theorem stating the meeting point of John and Lewis --/
theorem john_lewis_meeting_point :
  let j : Journey := {
    distance := 240,
    johnSpeed := 40,
    lewisOutboundSpeed := 60,
    lewisReturnSpeed := 50,
    johnBreakFrequency := 2,
    johnBreakDuration := 0.25,
    lewisBreakFrequency := 2.5,
    lewisBreakDuration := 1/3
  }
  ∃ (ε : ℝ), ε > 0 ∧ |meetingPoint j - 23.33| < ε :=
sorry

end NUMINAMATH_CALUDE_john_lewis_meeting_point_l2749_274977


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2749_274939

/-- Given r is the positive real solution to x³ - x² + ¼x - 1 = 0,
    prove that the infinite sum r³ + 2r⁶ + 3r⁹ + 4r¹² + ... equals 16r -/
theorem cubic_root_sum (r : ℝ) (hr : r > 0) (hroot : r^3 - r^2 + (1/4)*r - 1 = 0) :
  (∑' n, (n : ℝ) * r^(3*n)) = 16*r := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2749_274939


namespace NUMINAMATH_CALUDE_nested_sqrt_twelve_l2749_274910

theorem nested_sqrt_twelve (x : ℝ) : x > 0 ∧ x = Real.sqrt (12 + x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_twelve_l2749_274910


namespace NUMINAMATH_CALUDE_grid_sum_puzzle_l2749_274927

theorem grid_sum_puzzle :
  ∃ (a b c d e f g : ℤ),
    a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 ∧
    a + (-1) + 2 = 4 ∧
    2 + 1 + b = 3 ∧
    c + (-4) + (-3) = -2 ∧
    b - 5 - 4 = e ∧
    f = d - 3 ∧
    g = d + 3 ∧
    -8 = 4 + 3 - 9 - 2 + f + g :=
by sorry

end NUMINAMATH_CALUDE_grid_sum_puzzle_l2749_274927


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2749_274924

theorem sum_of_roots_cubic_equation :
  let f (x : ℝ) := 3 * x^3 - 6 * x^2 - 9 * x
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ + r₂ + r₃ = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2749_274924


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2749_274981

open Complex

theorem modulus_of_complex_fraction : 
  let z : ℂ := exp (π / 3 * I)
  ∀ (euler_formula : ∀ x : ℝ, exp (x * I) = cos x + I * sin x),
  abs (z / (1 - I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2749_274981


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2749_274974

/-- Theorem about a triangle ABC with specific properties -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < C → C < Real.pi / 2 →
  a * c * Real.cos B - b * c * Real.cos A = 3 * b^2 →
  c = Real.sqrt 11 →
  Real.sin C = 2 * Real.sqrt 2 / 3 →
  (a / b = 2) ∧ (1/2 * a * b * Real.sin C = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2749_274974


namespace NUMINAMATH_CALUDE_shirt_final_price_l2749_274966

/-- The final price of a shirt after two successive discounts --/
theorem shirt_final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  list_price = 150 → 
  discount1 = 19.954259576901087 →
  discount2 = 12.55 →
  list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = 105 := by
sorry

end NUMINAMATH_CALUDE_shirt_final_price_l2749_274966


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2749_274907

theorem sum_seven_consecutive_integers (n : ℤ) :
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2749_274907


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l2749_274916

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Calculates the probability of drawing a specific color from a bag -/
def Bag.prob (b : Bag) (color : ℕ) : ℚ :=
  (color : ℚ) / (b.total : ℚ)

/-- The configuration of Bag A -/
def bagA : Bag := { white := 5, black := 6 }

/-- The configuration of Bag B -/
def bagB : Bag := { yellow := 8, blue := 6 }

/-- The configuration of Bag C -/
def bagC : Bag := { yellow := 3, blue := 9 }

/-- The probability of drawing a yellow marble as the second marble -/
def yellowProbability : ℚ :=
  bagA.prob bagA.white * bagB.prob bagB.yellow +
  bagA.prob bagA.black * bagC.prob bagC.yellow

theorem yellow_marble_probability :
  yellowProbability = 61 / 154 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l2749_274916


namespace NUMINAMATH_CALUDE_divide_meter_into_hundred_parts_l2749_274991

theorem divide_meter_into_hundred_parts : 
  ∀ (total_length : ℝ) (num_parts : ℕ),
    total_length = 1 →
    num_parts = 100 →
    (total_length / num_parts : ℝ) = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_divide_meter_into_hundred_parts_l2749_274991


namespace NUMINAMATH_CALUDE_restaurant_group_adults_l2749_274925

/-- Calculates the number of adults in a restaurant group given the total bill, 
    number of children, and cost per meal. -/
theorem restaurant_group_adults 
  (total_bill : ℕ) 
  (num_children : ℕ) 
  (cost_per_meal : ℕ) : 
  total_bill = 56 → 
  num_children = 5 → 
  cost_per_meal = 8 → 
  (total_bill - num_children * cost_per_meal) / cost_per_meal = 2 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_adults_l2749_274925


namespace NUMINAMATH_CALUDE_ellipse_equation_l2749_274912

theorem ellipse_equation (a b : ℝ) (M : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    A.1 = 0 ∧ B.1 = 0 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (2 * Real.sqrt 6 / 3)^2 ∧
    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (2 * Real.sqrt 6 / 3)^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 6 / 3)^2) →
  a^2 = 6 ∧ b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2749_274912


namespace NUMINAMATH_CALUDE_suit_price_calculation_l2749_274904

theorem suit_price_calculation (original_price : ℝ) : 
  (original_price * 1.2 * 0.8 = 144) → original_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l2749_274904


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2749_274902

/-- The locus of points (x, y) in the complex plane satisfying 
    |z-2+i| + |z+3-i| = 6 is an ellipse -/
theorem locus_is_ellipse (z : ℂ) :
  let x := z.re
  let y := z.im
  (Complex.abs (z - (2 - Complex.I)) + Complex.abs (z - (-3 + Complex.I)) = 6) ↔
  ∃ (a b : ℝ) (h : 0 < b ∧ b < a),
    (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l2749_274902


namespace NUMINAMATH_CALUDE_wendy_ribbon_calculation_l2749_274982

/-- The amount of ribbon Wendy used to wrap presents, in inches. -/
def ribbon_used : ℕ := 46

/-- The amount of ribbon Wendy had left, in inches. -/
def ribbon_left : ℕ := 38

/-- The total amount of ribbon Wendy bought, in inches. -/
def total_ribbon : ℕ := ribbon_used + ribbon_left

theorem wendy_ribbon_calculation :
  total_ribbon = 84 := by
  sorry

end NUMINAMATH_CALUDE_wendy_ribbon_calculation_l2749_274982


namespace NUMINAMATH_CALUDE_quarter_piles_count_l2749_274958

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of dimes -/
def dime_piles : ℕ := 6

/-- Represents the number of piles of nickels -/
def nickel_piles : ℕ := 9

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value of all coins in cents -/
def total_value : ℕ := 2100

/-- Theorem stating that the number of piles of quarters is 4 -/
theorem quarter_piles_count : 
  ∃ (quarter_piles : ℕ), 
    quarter_piles * coins_per_pile * quarter_value + 
    dime_piles * coins_per_pile * dime_value + 
    nickel_piles * coins_per_pile * nickel_value + 
    penny_piles * coins_per_pile * penny_value = total_value ∧ 
    quarter_piles = 4 := by
  sorry

end NUMINAMATH_CALUDE_quarter_piles_count_l2749_274958


namespace NUMINAMATH_CALUDE_value_of_b_l2749_274948

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2749_274948


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l2749_274975

theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ ∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = x := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l2749_274975


namespace NUMINAMATH_CALUDE_sum_odd_plus_even_l2749_274951

def sum_odd_integers (n : ℕ) : ℕ :=
  (n + 1) * n

def sum_even_integers (n : ℕ) : ℕ :=
  n * (n + 1)

def m : ℕ := sum_odd_integers 56

def t : ℕ := sum_even_integers 25

theorem sum_odd_plus_even : m + t = 3786 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_plus_even_l2749_274951


namespace NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eleven_l2749_274953

theorem thirteen_pow_seven_mod_eleven : 13^7 % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eleven_l2749_274953


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2749_274959

theorem infinite_series_sum (a b : ℝ) 
  (h : (a / (b + 1)) / (1 - 1 / (b + 1)) = 3) : 
  (a / (a + 2*b)) / (1 - 1 / (a + 2*b)) = 3*(b + 1) / (5*b + 2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2749_274959


namespace NUMINAMATH_CALUDE_square_sum_equals_25_l2749_274949

theorem square_sum_equals_25 (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2) 
  (h2 : x + 6 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_25_l2749_274949


namespace NUMINAMATH_CALUDE_marge_garden_weeds_l2749_274984

def garden_problem (total_seeds planted_seeds non_growing_seeds eaten_fraction 
                    strangled_fraction kept_weeds final_plants : ℕ) : Prop :=
  let grown_plants := planted_seeds - non_growing_seeds
  let eaten_plants := (grown_plants / 3 : ℕ)
  let uneaten_plants := grown_plants - eaten_plants
  let strangled_plants := (uneaten_plants / 3 : ℕ)
  let healthy_plants := uneaten_plants - strangled_plants
  healthy_plants + kept_weeds = final_plants

theorem marge_garden_weeds : 
  ∃ (pulled_weeds : ℕ), 
    garden_problem 23 23 5 (1/3) (1/3) 1 9 ∧ 
    pulled_weeds = 3 := by sorry

end NUMINAMATH_CALUDE_marge_garden_weeds_l2749_274984


namespace NUMINAMATH_CALUDE_optimal_fence_placement_l2749_274988

/-- Represents the state of a tree (healthy or dead) --/
inductive TreeState
  | Healthy
  | Dead

/-- Represents a 6x6 grid of trees --/
def TreeGrid := Fin 6 → Fin 6 → TreeState

/-- Represents a fence placement --/
structure FencePlacement where
  vertical : Fin 3
  horizontal : Fin 3

/-- Checks if a tree is isolated by the given fence placement --/
def isIsolated (grid : TreeGrid) (fences : FencePlacement) (row col : Fin 6) : Prop :=
  sorry

/-- Counts the number of healthy trees in the grid --/
def countHealthyTrees (grid : TreeGrid) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem optimal_fence_placement
  (grid : TreeGrid)
  (healthy_count : countHealthyTrees grid = 20) :
  ∃ (fences : FencePlacement),
    ∀ (row col : Fin 6),
      grid row col = TreeState.Healthy →
      isIsolated grid fences row col :=
sorry

end NUMINAMATH_CALUDE_optimal_fence_placement_l2749_274988
