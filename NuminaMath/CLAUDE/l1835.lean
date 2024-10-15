import Mathlib

namespace NUMINAMATH_CALUDE_not_surjective_product_of_injective_exists_surjective_factors_l1835_183540

-- Part a
theorem not_surjective_product_of_injective (f g : ℤ → ℤ) 
  (hf : Function.Injective f) (hg : Function.Injective g) :
  ¬ Function.Surjective (fun x ↦ f x * g x) := by
  sorry

-- Part b
theorem exists_surjective_factors (f : ℤ → ℤ) (hf : Function.Surjective f) :
  ∃ g h : ℤ → ℤ, Function.Surjective g ∧ Function.Surjective h ∧
    ∀ x, f x = g x * h x := by
  sorry

end NUMINAMATH_CALUDE_not_surjective_product_of_injective_exists_surjective_factors_l1835_183540


namespace NUMINAMATH_CALUDE_pet_food_sale_discount_l1835_183586

def msrp : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def min_sale_price : ℝ := 25.20

theorem pet_food_sale_discount : ∃ (additional_discount : ℝ),
  additional_discount = 0.20 ∧
  min_sale_price = msrp * (1 - max_regular_discount) * (1 - additional_discount) :=
sorry

end NUMINAMATH_CALUDE_pet_food_sale_discount_l1835_183586


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1835_183539

theorem inequality_solution_set (x : ℝ) :
  (x / (x - 1) + (x + 1) / (2 * x) ≥ 5 / 2) ↔ (x ≥ 1 / 2 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1835_183539


namespace NUMINAMATH_CALUDE_sunflower_rose_height_difference_l1835_183557

/-- The height difference between a sunflower and a rose bush -/
theorem sunflower_rose_height_difference :
  let sunflower_height : ℚ := 9 + 3/5
  let rose_height : ℚ := 5 + 4/5
  sunflower_height - rose_height = 3 + 4/5 := by sorry

end NUMINAMATH_CALUDE_sunflower_rose_height_difference_l1835_183557


namespace NUMINAMATH_CALUDE_pythagorean_triples_example_l1835_183568

-- Define a Pythagorean triple
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the two sets of triples
def triple1 : (ℕ × ℕ × ℕ) := (3, 4, 5)
def triple2 : (ℕ × ℕ × ℕ) := (6, 8, 10)

-- Theorem stating that both triples are Pythagorean triples
theorem pythagorean_triples_example :
  (is_pythagorean_triple triple1.1 triple1.2.1 triple1.2.2) ∧
  (is_pythagorean_triple triple2.1 triple2.2.1 triple2.2.2) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triples_example_l1835_183568


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1835_183506

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1835_183506


namespace NUMINAMATH_CALUDE_range_of_linear_function_l1835_183517

def g (c d x : ℝ) : ℝ := c * x + d

theorem range_of_linear_function (c d : ℝ) (hc : c > 0) :
  Set.range (fun x => g c d x) = Set.Icc (-c + d) (2*c + d) :=
sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l1835_183517


namespace NUMINAMATH_CALUDE_solution_value_l1835_183554

theorem solution_value (a b : ℝ) : 
  (2 : ℝ) * a + (-1 : ℝ) * b = -1 → 2 * a - b + 2017 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1835_183554


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l1835_183547

/-- The length of rope Mr. Sanchez bought last week in feet -/
def rope_last_week : ℕ := 6

/-- The difference in feet between last week's and this week's rope purchase -/
def rope_difference : ℕ := 4

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The total length of rope Mr. Sanchez bought in inches -/
def total_rope_inches : ℕ := (rope_last_week + (rope_last_week - rope_difference)) * inches_per_foot

theorem sanchez_rope_theorem : total_rope_inches = 96 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l1835_183547


namespace NUMINAMATH_CALUDE_multiply_binomials_l1835_183567

theorem multiply_binomials (a b : ℝ) : (3*a + 2*b) * (a - 2*b) = 3*a^2 - 4*a*b - 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_binomials_l1835_183567


namespace NUMINAMATH_CALUDE_perpendicular_vector_solution_l1835_183571

def direction_vector : ℝ × ℝ := (2, 1)

theorem perpendicular_vector_solution :
  ∃! v : ℝ × ℝ, v.1 + v.2 = 1 ∧ v.1 * direction_vector.1 + v.2 * direction_vector.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_solution_l1835_183571


namespace NUMINAMATH_CALUDE_movie_duration_l1835_183562

theorem movie_duration (tuesday_time : ℕ) (max_movies : ℕ) 
  (h1 : tuesday_time = 270)
  (h2 : max_movies = 9) : 
  ∃ (movie_length : ℕ), movie_length = 90 ∧ 
  ∃ (tuesday_movies : ℕ), 
    tuesday_movies * movie_length = tuesday_time ∧
    3 * tuesday_movies = max_movies :=
by
  sorry

end NUMINAMATH_CALUDE_movie_duration_l1835_183562


namespace NUMINAMATH_CALUDE_exists_unvisited_planet_l1835_183507

/-- A type representing a planet in the solar system -/
structure Planet where
  id : ℕ

/-- A function that returns the closest planet to a given planet -/
def closest_planet (planets : Finset Planet) : Planet → Planet :=
  sorry

theorem exists_unvisited_planet (n : ℕ) (h : n ≥ 1) :
  ∀ (planets : Finset Planet),
    Finset.card planets = 2 * n + 1 →
    (∀ p q : Planet, p ∈ planets → q ∈ planets → p ≠ q → 
      closest_planet planets p ≠ closest_planet planets q) →
    ∃ p : Planet, p ∈ planets ∧ 
      ∀ q : Planet, q ∈ planets → closest_planet planets q ≠ p :=
sorry

end NUMINAMATH_CALUDE_exists_unvisited_planet_l1835_183507


namespace NUMINAMATH_CALUDE_distance_between_red_lights_l1835_183581

/-- The distance between lights in inches -/
def light_spacing : ℕ := 8

/-- The number of lights in a complete color pattern cycle -/
def pattern_length : ℕ := 2 + 3 + 1

/-- The position of the nth red light in the sequence -/
def red_light_position (n : ℕ) : ℕ :=
  (n - 1) / 2 * pattern_length + (n - 1) % 2 + 1

/-- Convert inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem distance_between_red_lights :
  inches_to_feet (light_spacing * (red_light_position 15 - red_light_position 4)) = 19.3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_red_lights_l1835_183581


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1835_183575

/-- Given that the solution set of ax^2 + 5x - 2 > 0 is {x | 1/2 < x < 2},
    prove that a = -2 and the solution set of ax^2 - 5x + a^2 - 1 > 0 is {x | -3 < x < 1/2} -/
theorem quadratic_inequality_problem (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (a = -2 ∧ 
   ∀ x : ℝ, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1835_183575


namespace NUMINAMATH_CALUDE_hannah_dessert_cost_l1835_183509

def county_fair_problem (initial_amount : ℝ) (amount_left : ℝ) : Prop :=
  let total_spent := initial_amount - amount_left
  let rides_cost := initial_amount / 2
  let dessert_cost := total_spent - rides_cost
  dessert_cost = 5

theorem hannah_dessert_cost :
  county_fair_problem 30 10 := by
  sorry

end NUMINAMATH_CALUDE_hannah_dessert_cost_l1835_183509


namespace NUMINAMATH_CALUDE_fraction_integer_values_fraction_values_l1835_183558

theorem fraction_integer_values (n : ℕ) : 
  (∃ k : ℤ, (8 * n + 157 : ℤ) / (4 * n + 7) = k) ↔ (n = 1 ∨ n = 34) :=
by sorry

theorem fraction_values (n : ℕ) :
  n = 1 → (8 * n + 157 : ℤ) / (4 * n + 7) = 15 ∧
  n = 34 → (8 * n + 157 : ℤ) / (4 * n + 7) = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_integer_values_fraction_values_l1835_183558


namespace NUMINAMATH_CALUDE_combination_equality_l1835_183578

theorem combination_equality (n : ℕ) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l1835_183578


namespace NUMINAMATH_CALUDE_division_relation_l1835_183576

theorem division_relation : 
  (29.94 / 1.45 = 17.9) → (2994 / 14.5 = 1790) := by
  sorry

end NUMINAMATH_CALUDE_division_relation_l1835_183576


namespace NUMINAMATH_CALUDE_dividend_calculation_l1835_183504

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18)
  (h2 : quotient = 9)
  (h3 : remainder = 4) :
  divisor * quotient + remainder = 166 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1835_183504


namespace NUMINAMATH_CALUDE_solution_system_1_solution_system_2_l1835_183521

-- System (1)
theorem solution_system_1 (x y : ℝ) : 
  (4*x + 8*y = 12 ∧ 3*x - 2*y = 5) → (x = 2 ∧ y = 1/2) := by sorry

-- System (2)
theorem solution_system_2 (x y : ℝ) : 
  ((1/2)*x - (y+1)/3 = 1 ∧ 6*x + 2*y = 10) → (x = 2 ∧ y = -1) := by sorry

end NUMINAMATH_CALUDE_solution_system_1_solution_system_2_l1835_183521


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l1835_183584

theorem floor_plus_self_eq_fifteen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 15/4 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l1835_183584


namespace NUMINAMATH_CALUDE_point_placement_on_line_l1835_183537

theorem point_placement_on_line : ∃ (a b c d : ℝ),
  |b - a| = 10 ∧
  |c - a| = 3 ∧
  |d - b| = 5 ∧
  |d - c| = 8 ∧
  a = 0 ∧ b = 10 ∧ c = -3 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_placement_on_line_l1835_183537


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1835_183583

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : Nat
  numGroups : Nat
  studentsPerGroup : Nat
  selectedNumber : Nat
  selectedGroup : Nat

/-- Theorem: In a systematic sampling of 50 students into 10 groups of 5,
    if the student numbered 12 is selected from the third group,
    then the student numbered 37 will be selected from the eighth group. -/
theorem systematic_sampling_theorem (s : SystematicSampling)
    (h1 : s.totalStudents = 50)
    (h2 : s.numGroups = 10)
    (h3 : s.studentsPerGroup = 5)
    (h4 : s.selectedNumber = 12)
    (h5 : s.selectedGroup = 3) :
    s.selectedNumber + (8 - s.selectedGroup) * s.studentsPerGroup = 37 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1835_183583


namespace NUMINAMATH_CALUDE_heather_initial_blocks_l1835_183508

/-- The number of blocks Heather shared with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ended up with -/
def remaining_blocks : ℕ := 45

/-- The initial number of blocks Heather had -/
def initial_blocks : ℕ := shared_blocks + remaining_blocks

theorem heather_initial_blocks : initial_blocks = 86 := by
  sorry

end NUMINAMATH_CALUDE_heather_initial_blocks_l1835_183508


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l1835_183563

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

-- Define the theorem
theorem quadratic_point_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : f c (-1) = y₁)
  (h2 : f c 2 = y₂)
  (h3 : f c (-3) = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l1835_183563


namespace NUMINAMATH_CALUDE_no_square_143_b_l1835_183538

theorem no_square_143_b : ¬ ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), b^2 + 4*b + 3 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_square_143_b_l1835_183538


namespace NUMINAMATH_CALUDE_complex_sum_powers_l1835_183526

theorem complex_sum_powers (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l1835_183526


namespace NUMINAMATH_CALUDE_nickel_count_l1835_183552

/-- Given a purchase of 150 cents paid with 50 coins consisting of only pennies and nickels,
    prove that the number of nickels used is 25. -/
theorem nickel_count (p n : ℕ) : 
  p + n = 50 →  -- Total number of coins
  p + 5 * n = 150 →  -- Total value in cents
  n = 25 := by sorry

end NUMINAMATH_CALUDE_nickel_count_l1835_183552


namespace NUMINAMATH_CALUDE_different_arrangements_count_l1835_183597

def num_red_balls : ℕ := 6
def num_green_balls : ℕ := 3
def num_selected_balls : ℕ := 4

def num_arrangements : ℕ := 15

theorem different_arrangements_count :
  (num_red_balls = 6) →
  (num_green_balls = 3) →
  (num_selected_balls = 4) →
  num_arrangements = 15 := by
  sorry

end NUMINAMATH_CALUDE_different_arrangements_count_l1835_183597


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l1835_183598

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4)
  (h6 : x = 1) :
  ∀ y : ℝ, (∀ z : ℤ, (0 < z ∧ z < y ∧ 
                       0 < z ∧ z < 15 ∧
                       -1 < z ∧ z < 5 ∧
                       0 < z ∧ z < 3 ∧
                       z + 2 < 4 ∧
                       z = 1) → z ≤ x) → 
  y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l1835_183598


namespace NUMINAMATH_CALUDE_stating_ball_338_position_l1835_183519

/-- 
Given a circular arrangement of 1000 cups where balls are placed in every 7th cup 
starting from cup 1, this function calculates the cup number for the nth ball.
-/
def ball_position (n : ℕ) : ℕ := 
  (1 + (n - 1) * 7) % 1000

/-- 
Theorem stating that the 338th ball will be placed in cup 359 
in the described arrangement.
-/
theorem ball_338_position : ball_position 338 = 359 := by
  sorry

#eval ball_position 338  -- This line is for verification purposes

end NUMINAMATH_CALUDE_stating_ball_338_position_l1835_183519


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l1835_183528

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_equals_one :
  ∀ a : ℝ,
  perpendicular (-a/3) 3 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l1835_183528


namespace NUMINAMATH_CALUDE_altitude_inscribed_radius_relation_l1835_183582

-- Define a triangle type
structure Triangle where
  -- Three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the triangle inequality holds
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitudes of the triangle
def altitude (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define the inscribed circle radius
def inscribed_radius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem altitude_inscribed_radius_relation (t : Triangle) :
  let (h₁, h₂, h₃) := altitude t
  let r := inscribed_radius t
  1 / h₁ + 1 / h₂ + 1 / h₃ = 1 / r := by sorry

end NUMINAMATH_CALUDE_altitude_inscribed_radius_relation_l1835_183582


namespace NUMINAMATH_CALUDE_min_value_approx_l1835_183599

-- Define the function to be minimized
def f (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / (a + b + c)

-- Define the conditions
def valid_digits (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ b > 3

-- Theorem statement
theorem min_value_approx (a b c : ℕ) (h : valid_digits a b c) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ f a b c ≥ 19.62 - ε :=
sorry

end NUMINAMATH_CALUDE_min_value_approx_l1835_183599


namespace NUMINAMATH_CALUDE_systematic_sampling_l1835_183588

theorem systematic_sampling 
  (total_students : Nat) 
  (num_segments : Nat) 
  (segment_size : Nat) 
  (sixteenth_segment_num : Nat) :
  total_students = 160 →
  num_segments = 20 →
  segment_size = 8 →
  sixteenth_segment_num = 125 →
  ∃ (first_segment_num : Nat),
    first_segment_num = 5 ∧
    sixteenth_segment_num = first_segment_num + segment_size * (16 - 1) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1835_183588


namespace NUMINAMATH_CALUDE_grading_orders_mod_100_l1835_183551

/-- The number of students --/
def num_students : ℕ := 40

/-- The number of problems per student --/
def problems_per_student : ℕ := 3

/-- The number of different grading orders --/
def N : ℕ := 2 * 3^(num_students - 2)

/-- Theorem stating the result of N modulo 100 --/
theorem grading_orders_mod_100 : N % 100 = 78 := by
  sorry

end NUMINAMATH_CALUDE_grading_orders_mod_100_l1835_183551


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1835_183590

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  (n / 1000 ≠ n / 100 % 10) ∧ 
  (n / 1000 ≠ n / 10 % 10) ∧ 
  (n / 1000 ≠ n % 10) ∧ 
  (n / 100 % 10 ≠ n / 10 % 10) ∧ 
  (n / 100 % 10 ≠ n % 10) ∧ 
  (n / 10 % 10 ≠ n % 10) ∧  -- All digits are different
  (n / 1000 = 5 ∨ n / 100 % 10 = 5 ∨ n / 10 % 10 = 5 ∨ n % 10 = 5) ∧  -- Includes the digit 5
  (n % (n / 1000) = 0) ∧ 
  (n % (n / 100 % 10) = 0) ∧ 
  (n % (n / 10 % 10) = 0) ∧ 
  (n % (n % 10) = 0)  -- Divisible by each of its digits

theorem smallest_valid_number : 
  is_valid_number 5124 ∧ 
  ∀ m : ℕ, is_valid_number m → m ≥ 5124 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1835_183590


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1835_183574

/-- Given a rhombus with area 24 and one diagonal of length 6, its perimeter is 20. -/
theorem rhombus_perimeter (area : ℝ) (diagonal1 : ℝ) (perimeter : ℝ) : 
  area = 24 → diagonal1 = 6 → perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1835_183574


namespace NUMINAMATH_CALUDE_roots_relation_l1835_183522

-- Define the original quadratic equation
def original_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the original equation
def root1 (a b c : ℝ) : ℝ := sorry
def root2 (a b c : ℝ) : ℝ := sorry

-- Define the new quadratic equation
def new_equation (a b c y : ℝ) : Prop := a^2 * y^2 + a * (b - c) * y - b * c = 0

-- State the theorem
theorem roots_relation (a b c : ℝ) (ha : a ≠ 0) :
  (∃ y1 y2 : ℝ, new_equation a b c y1 ∧ new_equation a b c y2 ∧
    y1 = root1 a b c + root2 a b c ∧
    y2 = root1 a b c * root2 a b c) :=
sorry

end NUMINAMATH_CALUDE_roots_relation_l1835_183522


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l1835_183593

/-- 
Represents the candy distribution function.
For a given number of students n and a position i,
it returns the number of candies given to the student at position i.
-/
def candy_distribution (n : ℕ) (i : ℕ) : ℕ :=
  sorry

/-- 
Checks if every student receives at least one candy
for a given number of students n.
-/
def every_student_gets_candy (n : ℕ) : Prop :=
  sorry

/-- 
Checks if a given natural number is a power of 2.
-/
def is_power_of_two (n : ℕ) : Prop :=
  sorry

/-- 
Theorem: For n ≥ 2, every student receives at least one candy
if and only if n is a power of 2.
-/
theorem candy_distribution_theorem (n : ℕ) (h : n ≥ 2) :
  every_student_gets_candy n ↔ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l1835_183593


namespace NUMINAMATH_CALUDE_zero_bounds_l1835_183580

theorem zero_bounds (a : ℝ) (x₀ : ℝ) (h_a : a > 0) 
  (h_zero : Real.exp (2 * x₀) + (a + 2) * Real.exp x₀ + a * x₀ = 0) : 
  Real.log (2 * a / (4 * a + 5)) < x₀ ∧ x₀ < -1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_bounds_l1835_183580


namespace NUMINAMATH_CALUDE_quadratic_with_one_solution_l1835_183573

theorem quadratic_with_one_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →
  a + c = 11 →
  a < c →
  (a = (11 - Real.sqrt 21) / 2 ∧ c = (11 + Real.sqrt 21) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_with_one_solution_l1835_183573


namespace NUMINAMATH_CALUDE_product_pqr_l1835_183500

theorem product_pqr (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
sorry

end NUMINAMATH_CALUDE_product_pqr_l1835_183500


namespace NUMINAMATH_CALUDE_cube_sum_product_l1835_183570

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 := by sorry

end NUMINAMATH_CALUDE_cube_sum_product_l1835_183570


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l1835_183556

theorem sum_remainder_mod_11 : 
  (101234 + 101235 + 101236 + 101237 + 101238 + 101239 + 101240) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l1835_183556


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1835_183560

-- Define the quadratic polynomial
def p (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_max_value (a b c : ℝ) (ha : a > 0) (h1 : p a b c 1 = 4) (h2 : p a b c 2 = 15) :
  (∃ (x : ℝ), ∀ (y : ℝ), p a b c y ≤ p a b c x) ∧
  (∀ (x : ℝ), p a b c x ≤ 4) ∧
  p a b c 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1835_183560


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1835_183520

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 5th term of an arithmetic sequence equals 8, given a₃ + a₇ = 16 -/
theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) (sum_eq : a 3 + a 7 = 16) : 
  a 5 = 8 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1835_183520


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1835_183587

theorem repeating_decimal_sum : 
  let x : ℚ := 2 / 9
  let y : ℚ := 1 / 33
  x + y = 25 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1835_183587


namespace NUMINAMATH_CALUDE_ratio_equality_l1835_183564

theorem ratio_equality (x y : ℝ) (h : 1.5 * x = 0.04 * y) :
  (y - x) / (y + x) = 73 / 77 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1835_183564


namespace NUMINAMATH_CALUDE_two_digit_quadratic_equation_l1835_183544

theorem two_digit_quadratic_equation :
  ∃ (P : ℕ), 
    (P ≥ 10 ∧ P < 100) ∧ 
    (∀ x : ℝ, x^2 + P*x + 2001 = (x + 29) * (x + 69)) :=
sorry

end NUMINAMATH_CALUDE_two_digit_quadratic_equation_l1835_183544


namespace NUMINAMATH_CALUDE_three_digit_number_difference_l1835_183501

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  ones_range : ones ≥ 0 ∧ ones ≤ 9

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the product of digits of a three-digit number -/
def ThreeDigitNumber.digitProduct (n : ThreeDigitNumber) : Nat :=
  n.hundreds * n.tens * n.ones

theorem three_digit_number_difference (a b c : ThreeDigitNumber) :
  a.digitProduct = 64 →
  b.digitProduct = 35 →
  c.digitProduct = 81 →
  a.hundreds + b.hundreds + c.hundreds = 24 →
  a.tens + b.tens + c.tens = 12 →
  a.ones + b.ones + c.ones = 6 →
  max (max a.value b.value) c.value - min (min a.value b.value) c.value = 182 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_difference_l1835_183501


namespace NUMINAMATH_CALUDE_same_sign_range_l1835_183503

theorem same_sign_range (m : ℝ) : (2 - m) * (|m| - 3) > 0 ↔ m ∈ Set.Ioo 2 3 ∪ Set.Iio (-3) := by
  sorry

end NUMINAMATH_CALUDE_same_sign_range_l1835_183503


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1835_183589

/-- Given a rectangle with length 8 and diagonal 17, its perimeter is 46 -/
theorem rectangle_perimeter (length width diagonal : ℝ) : 
  length = 8 → 
  diagonal = 17 → 
  length^2 + width^2 = diagonal^2 → 
  2 * (length + width) = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1835_183589


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l1835_183512

/-- Given points A and B in 2D space, and a point C such that BC = 1/2 * AB,
    prove that C has specific coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (-3, 5) → 
  B = (9, -1) → 
  C - B = (1/2 : ℝ) • (B - A) → 
  C = (15, -4) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l1835_183512


namespace NUMINAMATH_CALUDE_equation_roots_range_l1835_183532

theorem equation_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    3^(2*x + 1) + (m-1)*(3^(x+1) - 1) - (m-3)*3^x = 0 ∧
    3^(2*y + 1) + (m-1)*(3^(y+1) - 1) - (m-3)*3^y = 0) →
  m < (-3 - Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l1835_183532


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1835_183516

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 16 → x * y = 162 → x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1835_183516


namespace NUMINAMATH_CALUDE_fred_balloon_count_l1835_183514

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 6

/-- The factor by which Fred has more balloons than Sally -/
def fred_factor : ℕ := 3

/-- The number of blue balloons Fred has -/
def fred_balloons : ℕ := sally_balloons * fred_factor

theorem fred_balloon_count : fred_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloon_count_l1835_183514


namespace NUMINAMATH_CALUDE_multiple_of_nine_problem_l1835_183594

theorem multiple_of_nine_problem (N : ℕ) : 
  (∃ k : ℕ, N = 9 * k) →
  (∃ Q : ℕ, N = 9 * Q ∧ Q = 9 * 25 + 7) →
  N = 2088 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_problem_l1835_183594


namespace NUMINAMATH_CALUDE_average_of_numbers_l1835_183569

def numbers : List ℝ := [3, 16, 33, 28]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1835_183569


namespace NUMINAMATH_CALUDE_vet_count_l1835_183502

theorem vet_count (total : ℕ) 
  (puppy_kibble : ℕ → ℕ) (yummy_kibble : ℕ → ℕ)
  (h1 : puppy_kibble total = (20 * total) / 100)
  (h2 : yummy_kibble total = (30 * total) / 100)
  (h3 : yummy_kibble total - puppy_kibble total = 100) :
  total = 1000 := by
sorry

end NUMINAMATH_CALUDE_vet_count_l1835_183502


namespace NUMINAMATH_CALUDE_count_equal_S_consecutive_l1835_183565

def S (n : ℕ) : ℕ := (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8)

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem count_equal_S_consecutive : 
  ∃ (A B : ℕ), A ≠ B ∧ 
    is_three_digit A ∧ is_three_digit B ∧
    S A = S (A + 1) ∧ S B = S (B + 1) ∧
    ∀ (n : ℕ), is_three_digit n ∧ S n = S (n + 1) → n = A ∨ n = B :=
by sorry

end NUMINAMATH_CALUDE_count_equal_S_consecutive_l1835_183565


namespace NUMINAMATH_CALUDE_translation_proof_l1835_183553

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_proof :
  let P : Point := { x := -3, y := 2 }
  let translated_P := translate (translate P 2 0) 0 (-2)
  translated_P = { x := -1, y := 0 } := by sorry

end NUMINAMATH_CALUDE_translation_proof_l1835_183553


namespace NUMINAMATH_CALUDE_binomial_probability_two_l1835_183535

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem binomial_probability_two (X : ℕ → ℝ) :
  (∀ k, X k = binomial_pmf 6 (1/3) k) →
  X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_l1835_183535


namespace NUMINAMATH_CALUDE_min_product_equal_sum_l1835_183561

theorem min_product_equal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ ∧ a₀ * b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_product_equal_sum_l1835_183561


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l1835_183513

/-- Given a man's rowing rate in still water and his speed against the stream,
    prove that his speed with the stream is equal to twice his rate in still water
    minus his speed against the stream. -/
theorem mans_speed_with_stream
  (rate_still_water : ℝ)
  (speed_against_stream : ℝ)
  (h1 : rate_still_water = 7)
  (h2 : speed_against_stream = 4) :
  rate_still_water + (rate_still_water - speed_against_stream) = 2 * rate_still_water - speed_against_stream :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l1835_183513


namespace NUMINAMATH_CALUDE_rectangle_width_l1835_183549

theorem rectangle_width (perimeter : ℝ) (length_difference : ℝ) (width : ℝ) : 
  perimeter = 48 →
  length_difference = 2 →
  perimeter = 2 * (width + length_difference) + 2 * width →
  width = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1835_183549


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1835_183591

theorem number_exceeding_percentage (x : ℝ) : x = 0.2 * x + 40 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1835_183591


namespace NUMINAMATH_CALUDE_tom_sleep_hours_l1835_183536

/-- Proves that Tom was getting 6 hours of sleep before increasing it by 1/3 to 8 hours --/
theorem tom_sleep_hours : 
  ∀ (x : ℝ), 
  (x + (1/3) * x = 8) → 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_tom_sleep_hours_l1835_183536


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1835_183545

-- Define the variables x and y as real numbers
variable (x y : ℝ)

-- State the theorem
theorem express_y_in_terms_of_x (h : 2 * x + y = 1) : y = -2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1835_183545


namespace NUMINAMATH_CALUDE_quadratic_common_point_l1835_183533

theorem quadratic_common_point (a b c : ℝ) : 
  let f₁ := fun x => a * x^2 - b * x + c
  let f₂ := fun x => b * x^2 - c * x + a
  let f₃ := fun x => c * x^2 - a * x + b
  f₁ (-1) = f₂ (-1) ∧ f₂ (-1) = f₃ (-1) ∧ f₃ (-1) = a + b + c := by
sorry

end NUMINAMATH_CALUDE_quadratic_common_point_l1835_183533


namespace NUMINAMATH_CALUDE_sum_of_integer_solutions_is_zero_l1835_183510

theorem sum_of_integer_solutions_is_zero : 
  ∃ (S : Finset Int), 
    (∀ x ∈ S, x^4 - 49*x^2 + 576 = 0) ∧ 
    (∀ x : Int, x^4 - 49*x^2 + 576 = 0 → x ∈ S) ∧ 
    (S.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integer_solutions_is_zero_l1835_183510


namespace NUMINAMATH_CALUDE_ellipse_with_same_foci_and_eccentricity_l1835_183515

/-- The standard equation of an ellipse with the same foci as another ellipse and a given eccentricity -/
theorem ellipse_with_same_foci_and_eccentricity 
  (a₁ b₁ : ℝ) 
  (h₁ : 0 < a₁ ∧ 0 < b₁) 
  (h₂ : a₁ > b₁) 
  (e : ℝ) 
  (he : e = Real.sqrt 5 / 5) :
  let c₁ := Real.sqrt (a₁^2 - b₁^2)
  let a := 5
  let b := Real.sqrt 20
  ∀ x y : ℝ, 
    (x^2 / a₁^2 + y^2 / b₁^2 = 1) → 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
     c₁ = Real.sqrt 5 ∧ 
     e = c₁ / a) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_with_same_foci_and_eccentricity_l1835_183515


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1835_183572

theorem sum_of_fractions : (1 : ℚ) / 3 + (5 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1835_183572


namespace NUMINAMATH_CALUDE_george_second_day_hours_l1835_183579

/-- Calculates the hours worked on the second day given the hourly rate, 
    hours worked on the first day, and total earnings for two days. -/
def hoursWorkedSecondDay (hourlyRate : ℚ) (hoursFirstDay : ℚ) (totalEarnings : ℚ) : ℚ :=
  (totalEarnings - hourlyRate * hoursFirstDay) / hourlyRate

/-- Proves that given the specific conditions of the problem, 
    the hours worked on the second day is 2. -/
theorem george_second_day_hours : 
  hoursWorkedSecondDay 5 7 45 = 2 := by
  sorry

end NUMINAMATH_CALUDE_george_second_day_hours_l1835_183579


namespace NUMINAMATH_CALUDE_tan_half_sum_l1835_183577

theorem tan_half_sum (p q : Real) 
  (h1 : Real.cos p + Real.cos q = 1/3) 
  (h2 : Real.sin p + Real.sin q = 8/17) : 
  Real.tan ((p + q) / 2) = 24/17 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_l1835_183577


namespace NUMINAMATH_CALUDE_paper_reams_for_haley_l1835_183546

theorem paper_reams_for_haley (total_reams sister_reams : ℕ) 
  (h1 : total_reams = 5)
  (h2 : sister_reams = 3) :
  total_reams - sister_reams = 2 := by
  sorry

end NUMINAMATH_CALUDE_paper_reams_for_haley_l1835_183546


namespace NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l1835_183585

theorem minimum_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geo_mean : Real.sqrt 5 = Real.sqrt (5^a * 5^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l1835_183585


namespace NUMINAMATH_CALUDE_intersection_M_N_l1835_183543

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x : ℕ | x - 1 ≥ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1835_183543


namespace NUMINAMATH_CALUDE_spider_web_paths_l1835_183542

/-- The number of paths from (0, 0) to (m, n) on a grid, moving only right and up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The coordinates of the fly -/
def flyPosition : ℕ × ℕ := (5, 3)

theorem spider_web_paths :
  gridPaths flyPosition.1 flyPosition.2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_l1835_183542


namespace NUMINAMATH_CALUDE_divisible_by_10101_l1835_183518

/-- Given a two-digit number, returns the six-digit number formed by repeating it three times -/
def f (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + 10 * n

/-- Theorem: For any two-digit number n, f(n) is divisible by 10101 -/
theorem divisible_by_10101 (n : ℕ) (h : 10 ≤ n ∧ n < 100) : 
  ∃ k : ℕ, f n = 10101 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_10101_l1835_183518


namespace NUMINAMATH_CALUDE_find_a_l1835_183595

def U (a : ℝ) : Set ℝ := {3, a, a^2 + 2*a - 3}
def A : Set ℝ := {2, 3}

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, x ∈ U a → x ∈ A ∨ x = 5) ∧
  (∀ x : ℝ, x ∈ A → x ∈ U a) ∧
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_find_a_l1835_183595


namespace NUMINAMATH_CALUDE_gas_cost_equation_l1835_183592

theorem gas_cost_equation (x : ℚ) : x > 0 →
  (∃ (n m : ℕ), n = 4 ∧ m = 7 ∧ x / n - x / m = 10) ↔ x = 280 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_equation_l1835_183592


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1835_183530

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1) :
  Real.sqrt (((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1835_183530


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1835_183596

theorem polynomial_factorization (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = (x - 3) * (x - 2)) →
  (a = 1 ∧ b = -5 ∧ c = 6) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1835_183596


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1835_183505

/-- The area of an equilateral triangle with altitude √15 is 5√3 square units. -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 15) :
  let side : ℝ := 2 * Real.sqrt 5
  let area : ℝ := (side * h) / 2
  area = 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1835_183505


namespace NUMINAMATH_CALUDE_perfect_linear_relationship_l1835_183534

-- Define a scatter plot as a list of points
def ScatterPlot := List (ℝ × ℝ)

-- Define a function to check if all points lie on a straight line
def allPointsOnLine (plot : ScatterPlot) : Prop := sorry

-- Define residuals
def residuals (plot : ScatterPlot) : List ℝ := sorry

-- Define sum of squares of residuals
def sumSquaresResiduals (plot : ScatterPlot) : ℝ := sorry

-- Define correlation coefficient
def correlationCoefficient (plot : ScatterPlot) : ℝ := sorry

-- Theorem statement
theorem perfect_linear_relationship (plot : ScatterPlot) :
  allPointsOnLine plot →
  (∀ r ∈ residuals plot, r = 0) ∧
  sumSquaresResiduals plot = 0 ∧
  |correlationCoefficient plot| = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_linear_relationship_l1835_183534


namespace NUMINAMATH_CALUDE_total_handshakes_l1835_183566

/-- Represents the number of people in the meeting -/
def total_people : ℕ := 40

/-- Represents the number of people who mostly know each other -/
def group1_size : ℕ := 25

/-- Represents the number of strangers within group1 -/
def strangers_in_group1 : ℕ := 5

/-- Represents the number of people who know no one -/
def group2_size : ℕ := 15

/-- Calculates the number of handshakes between strangers in group1 -/
def handshakes_in_group1 : ℕ := strangers_in_group1 * (strangers_in_group1 - 1) / 2

/-- Calculates the number of handshakes involving group2 -/
def handshakes_involving_group2 : ℕ := group2_size * (total_people - 1)

/-- The main theorem stating the total number of handshakes -/
theorem total_handshakes : 
  handshakes_in_group1 + handshakes_involving_group2 = 595 := by sorry

end NUMINAMATH_CALUDE_total_handshakes_l1835_183566


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l1835_183529

def is_representable (m : ℕ) : Prop :=
  ∃ (n p : ℕ), p.Prime ∧ m = n^2 + p

theorem infinitely_many_non_representable :
  ∀ k : ℕ, ∃ m : ℕ, m > k ∧ ¬ is_representable m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l1835_183529


namespace NUMINAMATH_CALUDE_correct_swap_l1835_183524

def swap_values (a b : ℕ) : ℕ × ℕ :=
  let c := b
  let b' := a
  let a' := c
  (a', b')

theorem correct_swap :
  swap_values 6 5 = (5, 6) := by
sorry

end NUMINAMATH_CALUDE_correct_swap_l1835_183524


namespace NUMINAMATH_CALUDE_degree_of_g_l1835_183523

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9 * x^5 + 4 * x^3 + 2 * x - 6

-- Define a proposition for the degree of a polynomial
def hasDegree (p : ℝ → ℝ) (n : ℕ) : Prop := sorry

-- State the theorem
theorem degree_of_g 
  (g : ℝ → ℝ) 
  (h : hasDegree (fun x => f x + g x) 2) : 
  hasDegree g 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_g_l1835_183523


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1835_183548

theorem chess_tournament_games (n : ℕ) (h : n = 5) : 
  (n * (n - 1)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1835_183548


namespace NUMINAMATH_CALUDE_solution_set_of_f_x_plus_one_gt_zero_l1835_183531

def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + (1 - x)) = f x

def monotone_decreasing_from_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f y < f x

theorem solution_set_of_f_x_plus_one_gt_zero
  (f : ℝ → ℝ)
  (h_sym : symmetric_about_one f)
  (h_mono : monotone_decreasing_from_one f)
  (h_f_zero : f 0 = 0) :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_f_x_plus_one_gt_zero_l1835_183531


namespace NUMINAMATH_CALUDE_staircase_extension_l1835_183559

/-- Calculates the number of additional toothpicks needed to extend a staircase -/
def additional_toothpicks (initial_steps : ℕ) (final_steps : ℕ) (initial_toothpicks : ℕ) (increase_rate : ℕ) : ℕ :=
  sorry

/-- Theorem: Given a 4-step staircase with 28 toothpicks and an increase rate of 3,
    33 additional toothpicks are needed to build a 6-step staircase -/
theorem staircase_extension :
  additional_toothpicks 4 6 28 3 = 33 :=
sorry

end NUMINAMATH_CALUDE_staircase_extension_l1835_183559


namespace NUMINAMATH_CALUDE_number_of_boxes_l1835_183525

def total_oranges : ℕ := 45
def oranges_per_box : ℕ := 5

theorem number_of_boxes : total_oranges / oranges_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boxes_l1835_183525


namespace NUMINAMATH_CALUDE_intersection_circles_angle_relation_l1835_183511

/-- Given two intersecting circles with radius R and centers separated by a distance greater than R,
    prove that the angle β formed at one intersection point is three times the angle α formed at the other intersection point. -/
theorem intersection_circles_angle_relation (R : ℝ) (center_distance : ℝ) (α β : ℝ) :
  R > 0 →
  center_distance > R →
  α > 0 →
  β > 0 →
  β = 3 * α :=
by sorry

end NUMINAMATH_CALUDE_intersection_circles_angle_relation_l1835_183511


namespace NUMINAMATH_CALUDE_product_of_numbers_with_hcf_l1835_183527

theorem product_of_numbers_with_hcf (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 11 ∧ 
  a = 33 ∧ 
  ∀ c, c > 0 ∧ Nat.gcd a c = 11 → b ≤ c →
  a * b = 363 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_hcf_l1835_183527


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_plus_square_root_l1835_183550

theorem unique_solution_cube_root_plus_square_root (x : ℝ) :
  (((x - 3) ^ (1/3 : ℝ)) + ((5 - x) ^ (1/2 : ℝ)) = 2) ↔ (x = 4) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_plus_square_root_l1835_183550


namespace NUMINAMATH_CALUDE_wood_cutting_problem_l1835_183541

theorem wood_cutting_problem : Nat.gcd 90 72 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wood_cutting_problem_l1835_183541


namespace NUMINAMATH_CALUDE_inequality_proof_l1835_183555

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1835_183555
