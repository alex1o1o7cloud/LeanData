import Mathlib

namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2249_224909

/-- The area of a shape composed of a right triangle and 12 congruent squares -/
theorem shaded_area_theorem (hypotenuse : ℝ) (num_squares : ℕ) :
  hypotenuse = 10 →
  num_squares = 12 →
  let leg := hypotenuse / Real.sqrt 2
  let triangle_area := leg * leg / 2
  let square_side := leg / 3
  let square_area := square_side * square_side
  let total_squares_area := num_squares * square_area
  triangle_area + total_squares_area = 275 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2249_224909


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2249_224945

theorem min_value_quadratic :
  ∃ (min_y : ℝ), min_y = -44 ∧ ∀ (x y : ℝ), y = x^2 + 16*x + 20 → y ≥ min_y :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2249_224945


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2249_224911

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2249_224911


namespace NUMINAMATH_CALUDE_smallest_x_l2249_224985

theorem smallest_x (x a b : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (h3 : x > 0) : x ≥ 200000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_l2249_224985


namespace NUMINAMATH_CALUDE_probability_zeros_not_adjacent_l2249_224937

-- Define the total number of elements
def total_elements : ℕ := 5

-- Define the number of ones
def num_ones : ℕ := 3

-- Define the number of zeros
def num_zeros : ℕ := 2

-- Define the total number of arrangements
def total_arrangements : ℕ := Nat.factorial total_elements

-- Define the number of arrangements where zeros are adjacent
def adjacent_zero_arrangements : ℕ := 2 * Nat.factorial (total_elements - 1)

-- Statement to prove
theorem probability_zeros_not_adjacent :
  (1 : ℚ) - (adjacent_zero_arrangements : ℚ) / total_arrangements = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_probability_zeros_not_adjacent_l2249_224937


namespace NUMINAMATH_CALUDE_intersection_set_exists_l2249_224919

/-- A structure representing a collection of subsets with specific intersection properties -/
structure IntersectionSet (k : ℕ) where
  A : Set (Set ℕ)
  infinite : Set.Infinite A
  k_intersection : ∀ (S : Finset (Set ℕ)), S.card = k → S.toSet ⊆ A → ∃! x, ∀ s ∈ S, x ∈ s
  k_plus_one_empty : ∀ (S : Finset (Set ℕ)), S.card = k + 1 → S.toSet ⊆ A → ∀ x, ∃ s ∈ S, x ∉ s

/-- Theorem stating the existence of an IntersectionSet for any k > 1 -/
theorem intersection_set_exists (k : ℕ) (h : k > 1) : ∃ I : IntersectionSet k, True := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_exists_l2249_224919


namespace NUMINAMATH_CALUDE_total_profit_is_29_20_l2249_224954

/-- Represents the profit calculation for candied fruits --/
def candied_fruit_profit (num_apples num_grapes num_oranges : ℕ)
  (apple_price apple_cost grape_price grape_cost orange_price orange_cost : ℚ) : ℚ :=
  let apple_profit := num_apples * (apple_price - apple_cost)
  let grape_profit := num_grapes * (grape_price - grape_cost)
  let orange_profit := num_oranges * (orange_price - orange_cost)
  apple_profit + grape_profit + orange_profit

/-- Theorem stating that the total profit is $29.20 given the problem conditions --/
theorem total_profit_is_29_20 :
  candied_fruit_profit 15 12 10 2 1.2 1.5 0.9 2.5 1.5 = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_29_20_l2249_224954


namespace NUMINAMATH_CALUDE_expression_evaluation_l2249_224955

theorem expression_evaluation : (8^5) / (4 * 2^5 + 16) = (2^11) / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2249_224955


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l2249_224940

/-- Given a line of 1.5 meters with 15.25 centimeters erased, the resulting length is 134.75 centimeters. -/
theorem line_length_after_erasing (original_length : Real) (erased_length : Real) :
  original_length = 1.5 ∧ erased_length = 15.25 →
  original_length * 100 - erased_length = 134.75 :=
by sorry

end NUMINAMATH_CALUDE_line_length_after_erasing_l2249_224940


namespace NUMINAMATH_CALUDE_find_sets_A_and_B_l2249_224977

def I : Set ℕ := {x | x ≤ 8 ∧ x > 0}

theorem find_sets_A_and_B 
  (h1 : A ∪ (I \ B) = {1, 3, 4, 5, 6, 7})
  (h2 : (I \ A) ∪ B = {1, 2, 4, 5, 6, 8})
  (h3 : (I \ A) ∩ (I \ B) = {1, 5, 6}) :
  A = {3, 4, 7} ∧ B = {2, 4, 8} := by
sorry

end NUMINAMATH_CALUDE_find_sets_A_and_B_l2249_224977


namespace NUMINAMATH_CALUDE_calculation_proof_l2249_224965

theorem calculation_proof : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2249_224965


namespace NUMINAMATH_CALUDE_percent_of_percent_l2249_224950

theorem percent_of_percent (y : ℝ) : (0.3 * 0.6 * y) = (0.18 * y) := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l2249_224950


namespace NUMINAMATH_CALUDE_power_of_five_equality_l2249_224962

theorem power_of_five_equality (n : ℕ) : 5^n = 5 * 25^3 * 625^2 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_equality_l2249_224962


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_17_l2249_224976

theorem consecutive_integers_around_sqrt_17 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 17) → (Real.sqrt 17 < b) → (a + b = 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_17_l2249_224976


namespace NUMINAMATH_CALUDE_asymptote_equation_correct_l2249_224998

/-- Represents a hyperbola with equation x^2 - y^2/b^2 = 1 and one focus at (2, 0) -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- The equation of the asymptotes of the hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (Real.sqrt 3 * x = y) ∨ (Real.sqrt 3 * x = -y)

/-- Theorem stating that the equation of the asymptotes is correct -/
theorem asymptote_equation_correct (h : Hyperbola) :
  asymptote_equation h = λ x y => (Real.sqrt 3 * x = y) ∨ (Real.sqrt 3 * x = -y) :=
by sorry

end NUMINAMATH_CALUDE_asymptote_equation_correct_l2249_224998


namespace NUMINAMATH_CALUDE_unique_divisible_number_l2249_224924

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem unique_divisible_number :
  ∃! D : ℕ, D < 10 ∧ 
    is_divisible_by_3 (sum_of_digits (1000 + D * 10 + 4)) ∧ 
    is_divisible_by_4 (last_two_digits (1000 + D * 10 + 4)) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l2249_224924


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l2249_224956

theorem min_a_for_quadratic_inequality :
  (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → ∀ a : ℝ, x^2 + 2*a*x + 1 ≥ 0) →
  (∃ a_min : ℝ, a_min = -5/4 ∧
    (∀ a : ℝ, (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → x^2 + 2*a*x + 1 ≥ 0) → a ≥ a_min) ∧
    (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → x^2 + 2*a_min*x + 1 ≥ 0)) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l2249_224956


namespace NUMINAMATH_CALUDE_fraction_comparison_l2249_224957

theorem fraction_comparison : 
  (100 : ℚ) / 101 > 199 / 201 ∧ 199 / 201 > 99 / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2249_224957


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2249_224908

theorem trigonometric_equation_solution (x : ℝ) : 
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2249_224908


namespace NUMINAMATH_CALUDE_factor_expression_l2249_224913

theorem factor_expression (c : ℝ) : 210 * c^2 + 35 * c = 35 * c * (6 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2249_224913


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2249_224927

/-- The line on which the circle's center lies -/
def center_line (x y : ℝ) : Prop := y = -4 * x

/-- The line tangent to the circle -/
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The point of tangency -/
def tangent_point : ℝ × ℝ := (3, -2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 4)^2 = 8

theorem circle_equation_correct :
  ∃ (c : ℝ × ℝ), 
    center_line c.1 c.2 ∧
    (∃ (r : ℝ), r > 0 ∧
      ∀ (p : ℝ × ℝ), 
        circle_equation p.1 p.2 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    tangent_line tangent_point.1 tangent_point.2 ∧
    circle_equation tangent_point.1 tangent_point.2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2249_224927


namespace NUMINAMATH_CALUDE_negation_of_exp_positive_forall_l2249_224915

theorem negation_of_exp_positive_forall :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exp_positive_forall_l2249_224915


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l2249_224975

theorem consecutive_negative_integers_product_sum (n : ℤ) : 
  n < 0 ∧ n > -50 ∧ n * (n + 1) = 2400 → n + (n + 1) = -97 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l2249_224975


namespace NUMINAMATH_CALUDE_ali_seashells_l2249_224968

/-- Proves that Ali started with 180 seashells given the conditions of the problem -/
theorem ali_seashells : 
  ∀ S : ℕ, 
  (S - 40 - 30) / 2 = 55 → 
  S = 180 := by
sorry

end NUMINAMATH_CALUDE_ali_seashells_l2249_224968


namespace NUMINAMATH_CALUDE_number_division_property_l2249_224944

theorem number_division_property : ∃ x : ℝ, x / 5 = 80 + x / 6 := by
  sorry

end NUMINAMATH_CALUDE_number_division_property_l2249_224944


namespace NUMINAMATH_CALUDE_probability_of_red_is_half_l2249_224991

/-- A cube with a specific color distribution -/
structure ColoredCube where
  total_faces : ℕ
  red_faces : ℕ
  yellow_faces : ℕ
  green_faces : ℕ
  tricolor_faces : ℕ

/-- The probability of a specific color facing up when throwing the cube -/
def probability_of_color (cube : ColoredCube) (color_faces : ℕ) : ℚ :=
  color_faces / cube.total_faces

/-- Our specific cube with the given color distribution -/
def our_cube : ColoredCube :=
  { total_faces := 6
  , red_faces := 2
  , yellow_faces := 2
  , green_faces := 1
  , tricolor_faces := 1 }

/-- Theorem stating that the probability of red facing up is 1/2 -/
theorem probability_of_red_is_half :
  probability_of_color our_cube (our_cube.red_faces + our_cube.tricolor_faces) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_is_half_l2249_224991


namespace NUMINAMATH_CALUDE_negation_equivalence_l2249_224961

theorem negation_equivalence :
  ¬(∃ x : ℝ, x > 1 ∧ x^2 - x > 0) ↔ (∀ x : ℝ, x > 1 → x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2249_224961


namespace NUMINAMATH_CALUDE_part1_part2_l2249_224947

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part1 (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.C = 5 * Real.pi / 6) 
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) : 
  t.c = Real.sqrt 13 := by
sorry

-- Part 2
theorem part2 (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.B = Real.pi / 3) : 
  ∃ (x y : ℝ), x = -Real.sqrt 3 ∧ y = 2 * Real.sqrt 3 ∧ 
  ∀ z, (2 * t.c - t.a = z) → (x < z ∧ z < y) := by
sorry

end NUMINAMATH_CALUDE_part1_part2_l2249_224947


namespace NUMINAMATH_CALUDE_tom_mileage_l2249_224943

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of days Tom drives 100 miles per week -/
def fixed_mileage_days : ℕ := 4

/-- The fixed mileage Tom drives on certain days -/
def fixed_mileage : ℕ := 100

/-- The weekly fee Tom pays -/
def weekly_fee : ℚ := 100

/-- The cost per mile driven -/
def cost_per_mile : ℚ := 1 / 10

/-- The total amount Tom pays in a year -/
def total_yearly_cost : ℚ := 7800

/-- The number of miles Tom drives on Monday, Wednesday, and Friday in a year -/
def miles_on_mwf : ℕ := 10400

theorem tom_mileage :
  let total_miles : ℚ := (total_yearly_cost - weeks_per_year * weekly_fee) / cost_per_mile
  let fixed_miles : ℚ := weeks_per_year * fixed_mileage_days * fixed_mileage
  ↑miles_on_mwf = total_miles - fixed_miles :=
sorry

end NUMINAMATH_CALUDE_tom_mileage_l2249_224943


namespace NUMINAMATH_CALUDE_unit_complex_rational_power_minus_one_is_rational_l2249_224967

/-- A complex number with rational real and imaginary parts and modulus 1 -/
structure UnitComplexRational where
  re : ℚ
  im : ℚ
  norm_sq : re^2 + im^2 = 1

/-- The main theorem: z^(2n) - 1 is rational for any integer n -/
theorem unit_complex_rational_power_minus_one_is_rational
  (z : UnitComplexRational) (n : ℤ) :
  ∃ (q : ℚ), (z.re + z.im * Complex.I) ^ (2 * n) - 1 = q := by
  sorry

end NUMINAMATH_CALUDE_unit_complex_rational_power_minus_one_is_rational_l2249_224967


namespace NUMINAMATH_CALUDE_alternating_color_probability_l2249_224948

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of successful alternating sequences -/
def successful_sequences : ℕ := 2

/-- The probability of drawing all balls with alternating colors -/
def alternating_probability : ℚ := successful_sequences / (total_balls.choose white_balls)

theorem alternating_color_probability :
  alternating_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l2249_224948


namespace NUMINAMATH_CALUDE_different_rhetorical_device_l2249_224999

-- Define the rhetorical devices
inductive RhetoricalDevice
| Metaphor
| Personification

-- Define a function to assign rhetorical devices to options
def assignRhetoricalDevice (option : Char) : RhetoricalDevice :=
  match option with
  | 'A' => RhetoricalDevice.Metaphor
  | _ => RhetoricalDevice.Personification

-- Theorem statement
theorem different_rhetorical_device :
  ∀ (x : Char), x ≠ 'A' →
  assignRhetoricalDevice 'A' ≠ assignRhetoricalDevice x :=
by
  sorry

#check different_rhetorical_device

end NUMINAMATH_CALUDE_different_rhetorical_device_l2249_224999


namespace NUMINAMATH_CALUDE_circular_seating_l2249_224981

theorem circular_seating (total_people : Nat) (seated_people : Nat) (arrangements : Nat) :
  total_people = 6 →
  seated_people ≤ total_people →
  arrangements = 144 →
  arrangements = Nat.factorial (seated_people - 1) →
  seated_people = 5 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_l2249_224981


namespace NUMINAMATH_CALUDE_initial_gummy_worms_l2249_224932

def gummy_worms (n : ℕ) : ℕ → ℕ
  | 0 => n  -- Initial number of gummy worms
  | d + 1 => (gummy_worms n d) / 2  -- Number of gummy worms after d + 1 days

theorem initial_gummy_worms :
  ∀ n : ℕ, gummy_worms n 4 = 4 → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_gummy_worms_l2249_224932


namespace NUMINAMATH_CALUDE_frank_pepe_height_difference_l2249_224936

-- Define the players
structure Player where
  name : String
  height : Float

-- Define the team
def team : List Player :=
  [
    { name := "Big Joe", height := 8 },
    { name := "Ben", height := 7 },
    { name := "Larry", height := 6 },
    { name := "Frank", height := 5.5 },
    { name := "Pepe", height := 4.5 }
  ]

-- Define the height difference function
def heightDifference (p1 p2 : Player) : Float :=
  p1.height - p2.height

-- Theorem statement
theorem frank_pepe_height_difference :
  let frank := team.find? (fun p => p.name = "Frank")
  let pepe := team.find? (fun p => p.name = "Pepe")
  ∀ (f p : Player), frank = some f → pepe = some p →
    heightDifference f p = 1 := by
  sorry

end NUMINAMATH_CALUDE_frank_pepe_height_difference_l2249_224936


namespace NUMINAMATH_CALUDE_infinite_special_integers_l2249_224970

theorem infinite_special_integers (m : ℕ) :
  let n : ℕ := (m^2 + m + 2)^2 + (m^2 + m + 2) + 3
  ∀ p : ℕ, Prime p → p ∣ (n^2 + 3) →
    ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3) :=
by
  sorry

#check infinite_special_integers

end NUMINAMATH_CALUDE_infinite_special_integers_l2249_224970


namespace NUMINAMATH_CALUDE_number_problem_l2249_224912

theorem number_problem : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  n / sum = 2 * diff ∧ n % sum = 50 ∧ n = 220050 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2249_224912


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l2249_224964

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the squared distance between two points in 2D space -/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between intersection points of two specific circles -/
theorem intersection_distance_squared (c1 c2 : Circle)
  (h1 : c1 = ⟨(1, -2), 5⟩)
  (h2 : c2 = ⟨(1, 4), 3⟩) :
  ∃ (p1 p2 : ℝ × ℝ),
    squaredDistance p1 c1.center = c1.radius^2 ∧
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧
    squaredDistance p2 c2.center = c2.radius^2 ∧
    squaredDistance p1 p2 = 224/9 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l2249_224964


namespace NUMINAMATH_CALUDE_airport_exchange_rate_fraction_l2249_224963

def official_rate : ℚ := 5 / 1
def willie_euros : ℚ := 70
def airport_dollars : ℚ := 10

theorem airport_exchange_rate_fraction : 
  (airport_dollars / (willie_euros / official_rate)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_airport_exchange_rate_fraction_l2249_224963


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2249_224983

/-- Given an integer n, returns its last two digits as a pair of natural numbers -/
def lastTwoDigits (n : ℤ) : ℕ × ℕ :=
  let tens := (n % 100 / 10).toNat
  let units := (n % 10).toNat
  (tens, units)

/-- Theorem: For any integer divisible by 4 with the sum of its last two digits equal to 17,
    the product of its last two digits is 72 -/
theorem last_two_digits_product (n : ℤ) 
  (div_by_4 : 4 ∣ n) 
  (sum_17 : (lastTwoDigits n).1 + (lastTwoDigits n).2 = 17) : 
  (lastTwoDigits n).1 * (lastTwoDigits n).2 = 72 :=
by
  sorry

#check last_two_digits_product

end NUMINAMATH_CALUDE_last_two_digits_product_l2249_224983


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2249_224960

theorem quadratic_factorization_sum : ∃ (a b c d : ℝ),
  (∀ x, x^2 + 23*x + 132 = (x + a) * (x + b)) ∧
  (∀ x, x^2 - 25*x + 168 = (x - c) * (x - d)) ∧
  (a + c + d = 42) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2249_224960


namespace NUMINAMATH_CALUDE_p_plus_q_equals_twenty_one_halves_l2249_224901

theorem p_plus_q_equals_twenty_one_halves 
  (p q : ℝ) 
  (hp : p^3 - 21*p^2 + 35*p - 105 = 0) 
  (hq : 5*q^3 - 35*q^2 - 175*q + 1225 = 0) : 
  p + q = 21/2 := by sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_twenty_one_halves_l2249_224901


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2249_224921

theorem solution_set_inequality (x : ℝ) : 
  1 / x < 1 / 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2249_224921


namespace NUMINAMATH_CALUDE_unique_k_solution_l2249_224934

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem unique_k_solution (k : ℤ) : 
  k % 2 = 1 ∧ f (f (f k)) = 35 → k = 55 :=
by sorry

end NUMINAMATH_CALUDE_unique_k_solution_l2249_224934


namespace NUMINAMATH_CALUDE_arithmetic_progression_bound_l2249_224952

theorem arithmetic_progression_bound :
  ∃ (C : ℝ), C > 1 ∧
  ∀ (n : ℕ) (a : ℕ → ℕ),
    n > 1 →
    (∀ i j, i < j ∧ j ≤ n → a i < a j) →
    (∃ (d : ℚ), ∀ i j, i ≤ n ∧ j ≤ n → (1 : ℚ) / a i - (1 : ℚ) / a j = d * (i - j)) →
    (a 0 : ℝ) > C^n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_bound_l2249_224952


namespace NUMINAMATH_CALUDE_triangle_game_probability_l2249_224931

/-- A game board constructed from an equilateral triangle -/
structure GameBoard :=
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (h_positive : 0 < total_sections)
  (h_shaded_le_total : shaded_sections ≤ total_sections)

/-- The probability of the spinner landing in a shaded region -/
def landing_probability (board : GameBoard) : ℚ :=
  board.shaded_sections / board.total_sections

/-- Theorem stating that for a game board with 6 total sections and 2 shaded sections,
    the probability of landing in a shaded region is 1/3 -/
theorem triangle_game_probability :
  ∀ (board : GameBoard),
    board.total_sections = 6 →
    board.shaded_sections = 2 →
    landing_probability board = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_game_probability_l2249_224931


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2249_224917

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2249_224917


namespace NUMINAMATH_CALUDE_collinearity_iff_harmonic_l2249_224973

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (incident : Point → Line → Prop)

-- Define the collinearity relation
variable (collinear : Point → Point → Point → Prop)

-- Define the harmonic relation
variable (harmonic : Point → Point → Point → Point → Prop)

-- Define the points and lines
variable (A B C D E F H P X Y : Point)
variable (hA hB gA gB : Line)

-- Define the geometric conditions
variable (h1 : incident A hA)
variable (h2 : incident A gA)
variable (h3 : incident B hB)
variable (h4 : incident B gB)
variable (h5 : incident C hA ∧ incident C gB)
variable (h6 : incident D hB ∧ incident D gA)
variable (h7 : incident E gA ∧ incident E gB)
variable (h8 : incident F hA ∧ incident F hB)
variable (h9 : incident P hB)
variable (h10 : incident H gA)
variable (h11 : ∃ CP EF, incident X CP ∧ incident X EF ∧ incident C CP ∧ incident P CP ∧ incident E EF ∧ incident F EF)
variable (h12 : ∃ EP HF, incident Y EP ∧ incident Y HF ∧ incident E EP ∧ incident P EP ∧ incident H HF ∧ incident F HF)

-- State the theorem
theorem collinearity_iff_harmonic :
  collinear X Y B ↔ harmonic A H E D :=
sorry

end NUMINAMATH_CALUDE_collinearity_iff_harmonic_l2249_224973


namespace NUMINAMATH_CALUDE_function_identity_l2249_224933

theorem function_identity (f : ℕ → ℕ) :
  (∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l2249_224933


namespace NUMINAMATH_CALUDE_max_payment_is_31_l2249_224916

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := {n : ℕ | 2000 ≤ n ∧ n ≤ 2099}

/-- Payments for divisibility -/
def payments : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Divisors to check -/
def divisors : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Calculate the payment for a given number -/
def calculatePayment (n : FourDigitNumber) : ℕ :=
  (List.zip divisors payments).foldl
    (fun acc (d, p) => if n % d = 0 then acc + p else acc)
    0

/-- The maximum payment possible -/
def maxPayment : ℕ := 31

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), calculatePayment n = maxPayment ∧
  ∀ (m : FourDigitNumber), calculatePayment m ≤ maxPayment :=
sorry

end NUMINAMATH_CALUDE_max_payment_is_31_l2249_224916


namespace NUMINAMATH_CALUDE_problem_statement_l2249_224946

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2249_224946


namespace NUMINAMATH_CALUDE_ferry_problem_l2249_224923

/-- Represents the ferry problem and proves the speed of the current and distance between docks. -/
theorem ferry_problem (still_water_speed time_against time_with : ℝ) 
  (h1 : still_water_speed = 12)
  (h2 : time_against = 10)
  (h3 : time_with = 6) :
  ∃ (current_speed distance : ℝ),
    current_speed = 3 ∧
    distance = 90 ∧
    time_with * (still_water_speed + current_speed) = time_against * (still_water_speed - current_speed) ∧
    distance = (still_water_speed + current_speed) * time_with :=
by
  sorry


end NUMINAMATH_CALUDE_ferry_problem_l2249_224923


namespace NUMINAMATH_CALUDE_points_per_vegetable_l2249_224971

/-- Proves that the number of points given for each vegetable eaten is 2 --/
theorem points_per_vegetable (total_points : ℕ) (num_students : ℕ) (num_weeks : ℕ) (veggies_per_week : ℕ)
  (h1 : total_points = 200)
  (h2 : num_students = 25)
  (h3 : num_weeks = 2)
  (h4 : veggies_per_week = 2) :
  total_points / (num_students * num_weeks * veggies_per_week) = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_per_vegetable_l2249_224971


namespace NUMINAMATH_CALUDE_max_value_of_f_l2249_224972

def f (x y : ℝ) : ℝ := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 951625 / 256 ∧
  (∀ (x y : ℝ), x + y = 5 → f x y ≤ M) ∧
  (∃ (x y : ℝ), x + y = 5 ∧ f x y = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2249_224972


namespace NUMINAMATH_CALUDE_sqrt_square_12321_l2249_224966

theorem sqrt_square_12321 : (Real.sqrt 12321)^2 = 12321 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_12321_l2249_224966


namespace NUMINAMATH_CALUDE_dalmatians_right_spot_count_l2249_224974

/-- The number of Dalmatians with a spot on the right ear -/
def dalmatians_with_right_spot (total : ℕ) (left_only : ℕ) (right_only : ℕ) (no_spots : ℕ) : ℕ :=
  total - left_only - no_spots

/-- Theorem stating the number of Dalmatians with a spot on the right ear -/
theorem dalmatians_right_spot_count :
  dalmatians_with_right_spot 101 29 17 22 = 50 := by
  sorry

#eval dalmatians_with_right_spot 101 29 17 22

end NUMINAMATH_CALUDE_dalmatians_right_spot_count_l2249_224974


namespace NUMINAMATH_CALUDE_point_p_coordinates_l2249_224987

/-- Given points A(2, 3) and B(4, -3), if a point P satisfies |AP| = 3/2 |PB|, 
    then P has coordinates (16/5, 0). -/
theorem point_p_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  ‖A - P‖ = (3/2) * ‖P - B‖ → 
  P = (16/5, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l2249_224987


namespace NUMINAMATH_CALUDE_doubled_parallelepiped_volume_l2249_224982

/-- The volume of a rectangular parallelepiped with doubled dimensions -/
theorem doubled_parallelepiped_volume 
  (original_length original_width original_height : ℝ) 
  (h_length : original_length = 75) 
  (h_width : original_width = 80) 
  (h_height : original_height = 120) : 
  (2 * original_length) * (2 * original_width) * (2 * original_height) / 1000000 = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_doubled_parallelepiped_volume_l2249_224982


namespace NUMINAMATH_CALUDE_pythagorean_triple_9_12_15_l2249_224995

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_9_12_15_l2249_224995


namespace NUMINAMATH_CALUDE_emily_gardens_l2249_224949

theorem emily_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 :=
by sorry

end NUMINAMATH_CALUDE_emily_gardens_l2249_224949


namespace NUMINAMATH_CALUDE_simplified_expression_equals_22_5_l2249_224935

theorem simplified_expression_equals_22_5 : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_22_5_l2249_224935


namespace NUMINAMATH_CALUDE_domain_of_g_l2249_224980

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-1) 4

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 0 (5/2) := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l2249_224980


namespace NUMINAMATH_CALUDE_casino_chip_loss_difference_l2249_224984

theorem casino_chip_loss_difference : 
  ∀ (x y : ℕ), 
    x + y = 16 →  -- Total number of chips lost
    20 * x + 100 * y = 880 →  -- Value of lost chips
    x - y = 2 :=  -- Difference in number of chips lost
by
  sorry

end NUMINAMATH_CALUDE_casino_chip_loss_difference_l2249_224984


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l2249_224907

/-- The function f(x) = 3 sin x + 4 cos x has a maximum value of 5 -/
theorem max_value_of_sin_cos_combination :
  ∃ (M : ℝ), M = 5 ∧ ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l2249_224907


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l2249_224902

theorem butterflies_in_garden (initial : ℕ) (remaining : ℕ) : 
  remaining = 6 ∧ 3 * remaining = 2 * initial → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l2249_224902


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2249_224953

theorem fraction_to_decimal : (7 : ℚ) / 50 = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2249_224953


namespace NUMINAMATH_CALUDE_vitamin_a_daily_serving_l2249_224969

/-- The amount of Vitamin A in each pill (in mg) -/
def vitamin_a_per_pill : ℕ := 50

/-- The number of pills needed for the weekly recommended amount -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The recommended daily serving of Vitamin A (in mg) -/
def recommended_daily_serving : ℕ := (vitamin_a_per_pill * pills_per_week) / days_per_week

theorem vitamin_a_daily_serving :
  recommended_daily_serving = 200 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_daily_serving_l2249_224969


namespace NUMINAMATH_CALUDE_power_of_product_l2249_224997

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2249_224997


namespace NUMINAMATH_CALUDE_jessica_milk_problem_l2249_224920

theorem jessica_milk_problem (initial_milk : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial_milk = 5 →
  given_away = 16 / 3 →
  remaining = initial_milk - given_away →
  remaining = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_jessica_milk_problem_l2249_224920


namespace NUMINAMATH_CALUDE_tan_sin_expression_simplification_l2249_224903

theorem tan_sin_expression_simplification :
  Real.tan (70 * π / 180) * Real.sin (80 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_expression_simplification_l2249_224903


namespace NUMINAMATH_CALUDE_countMultiplesIs943_l2249_224942

/-- The number of integers between 1 and 3000 (inclusive) that are multiples of 5 or 7 but not multiples of 35 -/
def countMultiples : ℕ := sorry

theorem countMultiplesIs943 : countMultiples = 943 := by sorry

end NUMINAMATH_CALUDE_countMultiplesIs943_l2249_224942


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2249_224906

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + y * f x

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) (h2 : f 1 = 3) : f 501 = 503 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2249_224906


namespace NUMINAMATH_CALUDE_division_theorem_l2249_224978

theorem division_theorem (A : ℕ) : 14 = 3 * A + 2 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l2249_224978


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2249_224922

theorem quadratic_equation_solution (k : ℝ) (x : ℝ) :
  k * x^2 - (3 * k + 3) * x + 2 * k + 6 = 0 →
  (k = 0 → x = 2) ∧
  (k ≠ 0 → (x = 2 ∨ x = 1 + 3 / k)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2249_224922


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l2249_224994

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The statement of the problem -/
theorem coin_flip_probability_difference : 
  prob_k_heads 4 3 - prob_k_heads 4 4 = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l2249_224994


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2249_224992

theorem last_two_digits_sum (n : ℕ) : n = 30 → (7^n + 13^n) % 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2249_224992


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2249_224905

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which point C lies
def line_C (x : ℝ) : ℝ := 3 * x + 3

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Theorem statement
theorem point_C_coordinates :
  ∃ (C : ℝ × ℝ), 
    (C.2 = line_C C.1) ∧ 
    (abs ((C.1 - A.1) * (B.2 - A.2) - (B.1 - A.1) * (C.2 - A.2)) / 2 = triangle_area) ∧
    ((C = (-1, 0)) ∨ (C = (5/3, 8))) :=
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2249_224905


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2249_224939

theorem sum_of_roots_equation (x : ℝ) : 
  (10 = (x^3 - 5*x^2 - 10*x) / (x + 2)) → 
  (∃ (y z : ℝ), x + y + z = 5 ∧ 
    10 = (y^3 - 5*y^2 - 10*y) / (y + 2) ∧
    10 = (z^3 - 5*z^2 - 10*z) / (z + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2249_224939


namespace NUMINAMATH_CALUDE_max_protesters_l2249_224986

theorem max_protesters (population : ℕ) (reforms : ℕ) (dislike_per_reform : ℕ) :
  population = 96 →
  reforms = 5 →
  dislike_per_reform = population / 2 →
  (∀ r : ℕ, r ≤ reforms → dislike_per_reform = population / 2) →
  (∃ max_protesters : ℕ,
    max_protesters ≤ population ∧
    max_protesters * (reforms / 2 + 1) ≤ reforms * dislike_per_reform ∧
    ∀ n : ℕ, n ≤ population →
      n * (reforms / 2 + 1) ≤ reforms * dislike_per_reform →
      n ≤ max_protesters) →
  (∃ max_protesters : ℕ, max_protesters = 80) :=
by sorry

end NUMINAMATH_CALUDE_max_protesters_l2249_224986


namespace NUMINAMATH_CALUDE_problem_solution_l2249_224918

theorem problem_solution (r s : ℝ) 
  (h1 : 1 < r) 
  (h2 : r < s) 
  (h3 : 1/r + 1/s = 3/4) 
  (h4 : r*s = 8) : 
  s = 4 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2249_224918


namespace NUMINAMATH_CALUDE_gunther_tractor_payment_l2249_224993

/-- Calculates the monthly payment for a loan given the total amount and loan term in years -/
def monthly_payment (total_amount : ℕ) (years : ℕ) : ℚ :=
  (total_amount : ℚ) / (years * 12 : ℚ)

/-- Proves that for a $9000 loan over 5 years, the monthly payment is $150 -/
theorem gunther_tractor_payment :
  monthly_payment 9000 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_gunther_tractor_payment_l2249_224993


namespace NUMINAMATH_CALUDE_variance_estimation_l2249_224951

/-- Represents the data for a group of students -/
structure GroupData where
  count : ℕ
  average_score : ℝ
  variance : ℝ

/-- Calculates the estimated variance of test scores given two groups of students -/
def estimated_variance (male : GroupData) (female : GroupData) : ℝ :=
  let total_count := male.count + female.count
  let male_weight := male.count / total_count
  let female_weight := female.count / total_count
  let overall_average := male_weight * male.average_score + female_weight * female.average_score
  male_weight * (male.variance + (overall_average - male.average_score)^2) +
  female_weight * (female.variance + (female.average_score - overall_average)^2)

theorem variance_estimation (male : GroupData) (female : GroupData) :
  male.count = 400 →
  female.count = 600 →
  male.average_score = 80 →
  male.variance = 10 →
  female.average_score = 60 →
  female.variance = 20 →
  estimated_variance male female = 112 := by
  sorry

end NUMINAMATH_CALUDE_variance_estimation_l2249_224951


namespace NUMINAMATH_CALUDE_library_loan_availability_l2249_224914

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for books available for loan
variable (available_for_loan : Book → Prop)

-- Theorem statement
theorem library_loan_availability (h : ¬∀ (b : Book), available_for_loan b) :
  (∃ (b : Book), ¬available_for_loan b) ∧ (¬∀ (b : Book), available_for_loan b) :=
by sorry

end NUMINAMATH_CALUDE_library_loan_availability_l2249_224914


namespace NUMINAMATH_CALUDE_logarithm_inconsistency_l2249_224988

-- Define a custom logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given logarithmic values
def lg3 : ℝ := 0.47712
def lg1_5 : ℝ := 0.17609
def lg5 : ℝ := 0.69897
def lg2 : ℝ := 0.30103
def lg7_incorrect : ℝ := 0.84519

-- Theorem statement
theorem logarithm_inconsistency :
  lg 3 = lg3 ∧
  lg 1.5 = lg1_5 ∧
  lg 5 = lg5 ∧
  lg 2 = lg2 ∧
  lg 7 ≠ lg7_incorrect :=
by sorry

end NUMINAMATH_CALUDE_logarithm_inconsistency_l2249_224988


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2249_224979

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + a*b + a*c + b^2 + b*c + c^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2249_224979


namespace NUMINAMATH_CALUDE_power_of_eleven_l2249_224929

def total_prime_factors (x : ℕ) : ℕ := 26 + 5 + x

theorem power_of_eleven (x : ℕ) (h : total_prime_factors x = 33) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eleven_l2249_224929


namespace NUMINAMATH_CALUDE_white_space_area_is_31_l2249_224938

/-- Represents the dimensions of a rectangular board -/
structure Board :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a board -/
def boardArea (b : Board) : ℕ := b.width * b.height

/-- Represents the area covered by each letter -/
structure LetterAreas :=
  (C : ℕ)
  (O : ℕ)
  (D : ℕ)
  (E : ℕ)

/-- Calculates the total area covered by all letters -/
def totalLetterArea (l : LetterAreas) : ℕ := l.C + l.O + l.D + l.E

/-- The main theorem stating the white space area -/
theorem white_space_area_is_31 (board : Board) (letters : LetterAreas) : 
  board.width = 4 ∧ board.height = 18 ∧ 
  letters.C = 8 ∧ letters.O = 10 ∧ letters.D = 10 ∧ letters.E = 13 →
  boardArea board - totalLetterArea letters = 31 := by
  sorry


end NUMINAMATH_CALUDE_white_space_area_is_31_l2249_224938


namespace NUMINAMATH_CALUDE_distracted_scientist_waiting_time_l2249_224941

/-- The average waiting time for the first bite given the conditions of the distracted scientist problem -/
theorem distracted_scientist_waiting_time 
  (first_rod_bites : ℝ) 
  (second_rod_bites : ℝ) 
  (total_bites : ℝ) 
  (time_interval : ℝ) 
  (h1 : first_rod_bites = 3) 
  (h2 : second_rod_bites = 2) 
  (h3 : total_bites = first_rod_bites + second_rod_bites) 
  (h4 : time_interval = 6) : 
  (time_interval / total_bites) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_distracted_scientist_waiting_time_l2249_224941


namespace NUMINAMATH_CALUDE_vector_equality_transitive_l2249_224930

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equality_transitive (a b c : V) :
  a = b → b = c → a = c := by sorry

end NUMINAMATH_CALUDE_vector_equality_transitive_l2249_224930


namespace NUMINAMATH_CALUDE_sally_seashell_theorem_l2249_224928

/-- The amount of money Sally can make by selling seashells -/
def sally_seashell_money (monday_shells : ℕ) (price_per_shell : ℚ) : ℚ :=
  (monday_shells + monday_shells / 2) * price_per_shell

/-- Theorem: Sally can make $54 by selling all her seashells -/
theorem sally_seashell_theorem :
  sally_seashell_money 30 (120/100) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sally_seashell_theorem_l2249_224928


namespace NUMINAMATH_CALUDE_smallest_class_size_l2249_224989

theorem smallest_class_size (N : ℕ) (G : ℕ) : N = 7 ↔ 
  (N > 0 ∧ G > 0 ∧ (25 : ℚ) / 100 < (G : ℚ) / N ∧ (G : ℚ) / N < (30 : ℚ) / 100) ∧
  ∀ (M : ℕ) (H : ℕ), M < N → ¬(M > 0 ∧ H > 0 ∧ (25 : ℚ) / 100 < (H : ℚ) / M ∧ (H : ℚ) / M < (30 : ℚ) / 100) :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2249_224989


namespace NUMINAMATH_CALUDE_school_boys_count_l2249_224958

theorem school_boys_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 128 →
  boys = 80 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l2249_224958


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2249_224910

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x ≤ 1}
def Q : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2249_224910


namespace NUMINAMATH_CALUDE_journey_time_proof_l2249_224925

theorem journey_time_proof (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed1 = 40)
  (h4 : speed2 = 60) :
  ∃ (t1 : ℝ), t1 = 3 ∧ 
  ∃ (t2 : ℝ), t1 + t2 = total_time ∧ 
  speed1 * t1 + speed2 * t2 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_journey_time_proof_l2249_224925


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l2249_224926

/-- The average price of books Sandy bought given the conditions -/
def average_price_per_book (books_shop1 books_shop2 : ℕ) (price_shop1 price_shop2 : ℚ) : ℚ :=
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2)

/-- Theorem stating that the average price Sandy paid per book is $18 -/
theorem sandy_average_book_price :
  let books_shop1 : ℕ := 65
  let books_shop2 : ℕ := 55
  let price_shop1 : ℚ := 1280
  let price_shop2 : ℚ := 880
  average_price_per_book books_shop1 books_shop2 price_shop1 price_shop2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_sandy_average_book_price_l2249_224926


namespace NUMINAMATH_CALUDE_product_in_base7_l2249_224904

/-- Converts a base-7 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 7) ((m % 7) :: acc)
    go n []

/-- The product of 325₇ and 6₇ in base 7 is 2624₇ --/
theorem product_in_base7 :
  toBase7 (toBase10 [5, 2, 3] * toBase10 [6]) = [4, 2, 6, 2] := by
  sorry

end NUMINAMATH_CALUDE_product_in_base7_l2249_224904


namespace NUMINAMATH_CALUDE_robin_gum_packages_l2249_224990

theorem robin_gum_packages (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 18) (h2 : total_pieces = 486) :
  total_pieces / pieces_per_package = 27 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l2249_224990


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_quadratic_equation_constant_geometric_sequence_ratio_exponential_expression_l2249_224959

-- Problem 1
theorem consecutive_odd_integers_sum (k : ℤ) : 
  k + (k + 2) + (k + 4) = 51 → k = 15 := by sorry

-- Problem 2
theorem quadratic_equation_constant (x k a C : ℝ) :
  x^2 + 6*x + k = (x + a)^2 + C → C = 6 := by sorry

-- Problem 3
theorem geometric_sequence_ratio (p q r s R : ℝ) :
  p/q = 2 ∧ q/r = 2 ∧ r/s = 2 ∧ R = p/s → R = 8 := by sorry

-- Problem 4
theorem exponential_expression (n : ℕ) (A : ℝ) :
  A = (3^n * 9^(n+1)) / 27^(n-1) → A = 729 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_quadratic_equation_constant_geometric_sequence_ratio_exponential_expression_l2249_224959


namespace NUMINAMATH_CALUDE_project_hours_calculation_l2249_224900

theorem project_hours_calculation (kate : ℕ) (pat : ℕ) (mark : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 85) :
  kate + pat + mark = 153 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_calculation_l2249_224900


namespace NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l2249_224996

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b, if q satisfies ‖q - b‖ = 3 ‖q - a‖, 
    then q is at a fixed distance from (9/8)a + (-1/8)b. -/
theorem fixed_distance_from_linear_combination (a b q : E) 
  (h : ‖q - b‖ = 3 * ‖q - a‖) :
  ∃ (c : ℝ), ∀ (q : E), ‖q - b‖ = 3 * ‖q - a‖ → 
    ‖q - ((9/8 : ℝ) • a + (-1/8 : ℝ) • b)‖ = c :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_from_linear_combination_l2249_224996
