import Mathlib

namespace NUMINAMATH_CALUDE_marathon_first_hour_distance_l995_99512

/-- Represents a marathon runner's performance -/
structure MarathonRunner where
  initialPace : ℝ  -- Initial pace in miles per hour
  totalDistance : ℝ -- Total marathon distance in miles
  totalTime : ℝ     -- Total race time in hours
  remainingPaceFactor : ℝ -- Factor for remaining pace (e.g., 0.8 for 80%)

/-- Calculates the distance covered in the first hour -/
def distanceInFirstHour (runner : MarathonRunner) : ℝ :=
  runner.initialPace

/-- Calculates the remaining distance after the first hour -/
def remainingDistance (runner : MarathonRunner) : ℝ :=
  runner.totalDistance - distanceInFirstHour runner

/-- Calculates the time spent running the remaining distance -/
def remainingTime (runner : MarathonRunner) : ℝ :=
  runner.totalTime - 1

/-- Theorem: The distance covered in the first hour of a 26-mile marathon is 10 miles -/
theorem marathon_first_hour_distance
  (runner : MarathonRunner)
  (h1 : runner.totalDistance = 26)
  (h2 : runner.totalTime = 3)
  (h3 : runner.remainingPaceFactor = 0.8)
  (h4 : remainingTime runner = 2)
  (h5 : remainingDistance runner / (runner.initialPace * runner.remainingPaceFactor) = remainingTime runner) :
  distanceInFirstHour runner = 10 := by
  sorry


end NUMINAMATH_CALUDE_marathon_first_hour_distance_l995_99512


namespace NUMINAMATH_CALUDE_remainder_sum_l995_99541

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 47) 
  (hd : d % 45 = 14) : 
  (c + d) % 15 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l995_99541


namespace NUMINAMATH_CALUDE_crayon_distribution_l995_99571

theorem crayon_distribution (initial_boxes : Nat) (crayons_per_box : Nat) 
  (to_mae : Nat) (to_rey : Nat) (left : Nat) :
  initial_boxes = 7 →
  crayons_per_box = 15 →
  to_mae = 12 →
  to_rey = 20 →
  left = 25 →
  (initial_boxes * crayons_per_box - to_mae - to_rey - left) - to_mae = 36 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_l995_99571


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_five_l995_99578

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: If in a triangle ABC, b*cos(A) + a*cos(B) = c^2 and a = b = 2, 
    then the perimeter of the triangle is 5 -/
theorem triangle_perimeter_is_five (t : Triangle) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = t.c^2)
  (h2 : t.a = 2)
  (h3 : t.b = 2) : 
  t.a + t.b + t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_is_five_l995_99578


namespace NUMINAMATH_CALUDE_sets_and_conditions_l995_99502

def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2*a}

theorem sets_and_conditions :
  (∀ x, x ∈ (A ∪ B 3) ↔ -2 < x ∧ x < 7) ∧
  (∀ x, x ∈ (A ∩ B 3) ↔ -1 < x ∧ x < 5) ∧
  (∀ a, (∀ x, x ∈ B a → x ∈ A) ↔ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sets_and_conditions_l995_99502


namespace NUMINAMATH_CALUDE_equal_share_of_sweets_l995_99561

/-- Represents the number of sweets Jennifer has of each color -/
structure Sweets where
  green : Nat
  blue : Nat
  yellow : Nat

/-- The total number of people sharing the sweets -/
def totalPeople : Nat := 4

/-- Jennifer's sweets -/
def jenniferSweets : Sweets := { green := 212, blue := 310, yellow := 502 }

/-- Theorem stating that each person gets 256 sweets when Jennifer shares equally -/
theorem equal_share_of_sweets (s : Sweets) (h : s = jenniferSweets) :
  (s.green + s.blue + s.yellow) / totalPeople = 256 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_of_sweets_l995_99561


namespace NUMINAMATH_CALUDE_lunchroom_students_l995_99545

theorem lunchroom_students (students_per_table : ℕ) (num_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : num_tables = 34) : 
  students_per_table * num_tables = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l995_99545


namespace NUMINAMATH_CALUDE_ice_cream_cost_theorem_l995_99520

/-- Ice cream order details and prices -/
structure IceCreamOrder where
  kiddie_price : ℚ
  regular_price : ℚ
  double_price : ℚ
  sprinkles_price : ℚ
  nuts_price : ℚ
  discount_rate : ℚ
  regular_with_nuts : ℕ
  kiddie_with_sprinkles : ℕ
  double_with_both : ℕ
  regular_with_sprinkles : ℕ
  regular_with_nuts_only : ℕ

/-- Calculate the total cost of an ice cream order after applying the discount -/
def total_cost_after_discount (order : IceCreamOrder) : ℚ :=
  let subtotal := 
    order.regular_with_nuts * (order.regular_price + order.nuts_price) +
    order.kiddie_with_sprinkles * (order.kiddie_price + order.sprinkles_price) +
    order.double_with_both * (order.double_price + order.nuts_price + order.sprinkles_price) +
    order.regular_with_sprinkles * (order.regular_price + order.sprinkles_price) +
    order.regular_with_nuts_only * (order.regular_price + order.nuts_price)
  subtotal * (1 - order.discount_rate)

/-- Theorem stating that the given ice cream order costs $49.50 after discount -/
theorem ice_cream_cost_theorem (order : IceCreamOrder) 
  (h1 : order.kiddie_price = 3)
  (h2 : order.regular_price = 4)
  (h3 : order.double_price = 6)
  (h4 : order.sprinkles_price = 1)
  (h5 : order.nuts_price = 3/2)
  (h6 : order.discount_rate = 1/10)
  (h7 : order.regular_with_nuts = 2)
  (h8 : order.kiddie_with_sprinkles = 2)
  (h9 : order.double_with_both = 3)
  (h10 : order.regular_with_sprinkles = 1)
  (h11 : order.regular_with_nuts_only = 1) :
  total_cost_after_discount order = 99/2 :=
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_theorem_l995_99520


namespace NUMINAMATH_CALUDE_grapefruit_orchards_l995_99589

theorem grapefruit_orchards (total : ℕ) (lemon : ℕ) (orange : ℕ) (lime_grapefruit : ℕ) :
  total = 16 →
  lemon = 8 →
  orange = lemon / 2 →
  lime_grapefruit = total - lemon - orange →
  lime_grapefruit / 2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_grapefruit_orchards_l995_99589


namespace NUMINAMATH_CALUDE_straw_length_theorem_l995_99509

/-- The total length of overlapping straws -/
def total_length (straw_length : ℕ) (overlap : ℕ) (num_straws : ℕ) : ℕ :=
  straw_length + (straw_length - overlap) * (num_straws - 1)

/-- Theorem: The total length of 30 straws is 576 cm -/
theorem straw_length_theorem :
  total_length 25 6 30 = 576 := by
  sorry

end NUMINAMATH_CALUDE_straw_length_theorem_l995_99509


namespace NUMINAMATH_CALUDE_variation_relationship_l995_99564

theorem variation_relationship (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_variation_relationship_l995_99564


namespace NUMINAMATH_CALUDE_function_inequality_l995_99588

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h_sym : ∀ x, f (2 - x) = f x)
  (h_deriv : ∀ x, x ≠ 1 → (deriv f x) / (x - 1) < 0)
  (x₁ x₂ : ℝ) (h_sum : x₁ + x₂ > 2) (h_order : x₁ < x₂) :
  f x₁ > f x₂ :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l995_99588


namespace NUMINAMATH_CALUDE_units_digit_of_difference_is_seven_l995_99584

-- Define a three-digit number
def ThreeDigitNumber (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

-- Define the relationship between hundreds and units digits
def HundredsUnitsRelation (a c : ℕ) : Prop :=
  a = c - 3

-- Define the original number
def OriginalNumber (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

-- Define the reversed number
def ReversedNumber (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

-- Theorem: The units digit of the difference is 7
theorem units_digit_of_difference_is_seven 
  (a b c : ℕ) 
  (h1 : ThreeDigitNumber a b c) 
  (h2 : HundredsUnitsRelation a c) : 
  (OriginalNumber a b c - ReversedNumber a b c) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_difference_is_seven_l995_99584


namespace NUMINAMATH_CALUDE_rectangle_length_ratio_l995_99577

theorem rectangle_length_ratio (L B : ℝ) (L' : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (L' * (3 * B) = (3/2) * (L * B)) → L' / L = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_ratio_l995_99577


namespace NUMINAMATH_CALUDE_tan_negative_55_6_pi_l995_99511

theorem tan_negative_55_6_pi : Real.tan (-55/6 * Real.pi) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_55_6_pi_l995_99511


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l995_99594

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero :
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l995_99594


namespace NUMINAMATH_CALUDE_max_rounds_le_three_l995_99544

/-- The number of students for a given n -/
def num_students (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The game described in the problem -/
structure ClassroomGame (n : ℕ) where
  n_gt_one : n > 1
  students : ℕ := num_students n
  classrooms : ℕ := n
  capacities : List ℕ := List.range n

/-- The maximum number of rounds possible in the game -/
def max_rounds (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of rounds is at most 3 -/
theorem max_rounds_le_three (n : ℕ) (game : ClassroomGame n) : max_rounds n ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_max_rounds_le_three_l995_99544


namespace NUMINAMATH_CALUDE_longest_rod_in_cube_l995_99510

theorem longest_rod_in_cube (side_length : ℝ) (h : side_length = 4) :
  Real.sqrt (3 * side_length^2) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_rod_in_cube_l995_99510


namespace NUMINAMATH_CALUDE_perfect_rectangle_theorem_l995_99570

/-- Represents a perfect rectangle divided into squares -/
structure PerfectRectangle where
  squares : List ℕ
  is_perfect : squares.length > 0

/-- The specific perfect rectangle from the problem -/
def given_rectangle : PerfectRectangle where
  squares := [9, 16, 2, 5, 7, 25, 28, 33]
  is_perfect := by simp

/-- Checks if the list is sorted in ascending order -/
def is_sorted (l : List ℕ) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≤ l[j]!

/-- The main theorem to prove -/
theorem perfect_rectangle_theorem (rect : PerfectRectangle) :
  rect = given_rectangle →
  is_sorted (rect.squares.filter (λ x => x ≠ 9 ∧ x ≠ 16)) ∧
  (rect.squares.filter (λ x => x ≠ 9 ∧ x ≠ 16)).length = 6 :=
by sorry

end NUMINAMATH_CALUDE_perfect_rectangle_theorem_l995_99570


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l995_99508

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 3 + 4 * Complex.I)) = 6 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l995_99508


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l995_99599

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l995_99599


namespace NUMINAMATH_CALUDE_minimum_additional_marbles_lisa_additional_marbles_l995_99598

theorem minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_additional_marbles :
  minimum_additional_marbles 12 40 = 38 := by
  sorry

end NUMINAMATH_CALUDE_minimum_additional_marbles_lisa_additional_marbles_l995_99598


namespace NUMINAMATH_CALUDE_intersection_with_complement_l995_99565

-- Define the sets
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ

-- State the theorem
theorem intersection_with_complement :
  P ∩ (U \ Q) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l995_99565


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l995_99525

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_subsequence (a : ℕ → ℝ) (k : ℕ → ℕ) (q : ℝ) : Prop :=
  ∀ n, a (k (n + 1)) = a (k n) * q

def strictly_increasing (k : ℕ → ℕ) : Prop :=
  ∀ n, k n < k (n + 1)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ) (d q : ℝ) (k : ℕ → ℕ) 
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_subsequence a k q)
  (h_incr : strictly_increasing k)
  (h_d_neq_0 : d ≠ 0)
  (h_k1 : k 1 = 1)
  (h_k2 : k 2 = 3)
  (h_k3 : k 3 = 8) :
  (a 1 / d = 4 / 3) ∧ 
  ((∀ n, k (n + 1) = k n * q) ↔ a 1 / d = 1) ∧
  ((∀ n, k (n + 1) = k n * q) → 
   (∀ n : ℕ, 0 < n → a n + a (k n) > 2 * k n) → 
   a 1 ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l995_99525


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l995_99553

theorem unique_solution_factorial_equation :
  ∃! (n : ℕ), n > 0 ∧ (n + 2).factorial - (n + 1).factorial - n.factorial = n^2 + n^4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l995_99553


namespace NUMINAMATH_CALUDE_bucket_calculation_reduced_capacity_buckets_l995_99595

/-- Given a tank that requires a certain number of buckets to fill and a reduction in bucket capacity,
    calculate the new number of buckets required to fill the tank. -/
theorem bucket_calculation (original_buckets : ℕ) (capacity_reduction : ℚ) : 
  original_buckets / capacity_reduction = original_buckets * (1 / capacity_reduction) :=
by sorry

/-- Prove that 105 buckets are required when the original number of buckets is 42
    and the capacity is reduced to two-fifths. -/
theorem reduced_capacity_buckets : 
  let original_buckets : ℕ := 42
  let capacity_reduction : ℚ := 2 / 5
  original_buckets / capacity_reduction = 105 :=
by sorry

end NUMINAMATH_CALUDE_bucket_calculation_reduced_capacity_buckets_l995_99595


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l995_99593

theorem infinite_solutions_exist :
  ∃ m : ℕ+, ∀ n : ℕ, ∃ a b c : ℕ+,
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = m / (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l995_99593


namespace NUMINAMATH_CALUDE_ellipse_sum_l995_99501

-- Define the foci
def F₁ : ℝ × ℝ := (0, 1)
def F₂ : ℝ × ℝ := (6, 1)

-- Define the distance sum constant
def distance_sum : ℝ := 8

-- Define the ellipse properties
def ellipse_properties (h k a b : ℝ) : Prop :=
  ∀ (x y : ℝ),
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔
    Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
    Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = distance_sum

-- Theorem statement
theorem ellipse_sum (h k a b : ℝ) :
  ellipse_properties h k a b →
  h + k + a + b = 8 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l995_99501


namespace NUMINAMATH_CALUDE_equation_solution_l995_99523

theorem equation_solution : ∃! x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ (x / (x - 1) = 2 / (x^2 - 1)) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l995_99523


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l995_99521

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let angle_BAC := 2 * Real.pi / 3
  let AB := 3
  true  -- We don't need to specify all properties of the triangle

-- Define point D on BC
def point_D_on_BC (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • B + t • C

-- Define the condition BD = 2DC
def BD_equals_2DC (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = 2/3 ∧ D = (1 - t) • B + t • C

-- Main theorem
theorem triangle_ABC_properties 
  (A B C D : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_D_on_BC : point_D_on_BC B C D)
  (h_BD_2DC : BD_equals_2DC B C D) :
  (∀ (area_ABC : ℝ), area_ABC = 3 * Real.sqrt 3 → 
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = Real.sqrt 37) ∧
  (∀ (AD : ℝ), AD = 1 → 
    (let area_ABD := Real.sqrt 3 / 4 * 3; true)) :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l995_99521


namespace NUMINAMATH_CALUDE_number_problem_l995_99515

theorem number_problem (N : ℝ) : 
  (3/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N * (1/2 : ℝ) = 45 → 
  (60/100 : ℝ) * N = 540 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l995_99515


namespace NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l995_99524

theorem smallest_area_of_2020th_square (n : ℕ) (A : ℕ) : 
  n > 0 → 
  n^2 = 2019 + A → 
  A ≠ 1 → 
  (∀ m : ℕ, m > 0 ∧ m^2 = 2019 + A → n ≤ m) → 
  A ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l995_99524


namespace NUMINAMATH_CALUDE_number_division_remainder_l995_99583

theorem number_division_remainder (N : ℕ) : 
  N % 5 = 0 ∧ N / 5 = 2 → N % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainder_l995_99583


namespace NUMINAMATH_CALUDE_new_year_fireworks_display_l995_99560

def fireworks_per_number : ℕ := 6
def fireworks_per_letter : ℕ := 5
def additional_boxes : ℕ := 50
def fireworks_per_box : ℕ := 8

def year_numbers : ℕ := 4
def phrase_letters : ℕ := 12

theorem new_year_fireworks_display :
  let year_fireworks := year_numbers * fireworks_per_number
  let phrase_fireworks := phrase_letters * fireworks_per_letter
  let additional_fireworks := additional_boxes * fireworks_per_box
  year_fireworks + phrase_fireworks + additional_fireworks = 476 := by
sorry

end NUMINAMATH_CALUDE_new_year_fireworks_display_l995_99560


namespace NUMINAMATH_CALUDE_fraction_simplification_l995_99519

theorem fraction_simplification :
  let x := (1/2 - 1/3) / (3/7 + 1/9)
  x * (1/4) = 21/272 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l995_99519


namespace NUMINAMATH_CALUDE_snackles_remainder_l995_99586

theorem snackles_remainder (m : ℕ) (h : m % 11 = 4) : (3 * m) % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_snackles_remainder_l995_99586


namespace NUMINAMATH_CALUDE_broccoli_production_increase_l995_99581

def broccoli_production_difference (this_year_production : ℕ) 
  (last_year_side_length : ℕ) : Prop :=
  this_year_production = 1600 ∧
  last_year_side_length * last_year_side_length < this_year_production ∧
  (last_year_side_length + 1) * (last_year_side_length + 1) = this_year_production ∧
  this_year_production - (last_year_side_length * last_year_side_length) = 79

theorem broccoli_production_increase : 
  ∃ (last_year_side_length : ℕ), broccoli_production_difference 1600 last_year_side_length :=
sorry

end NUMINAMATH_CALUDE_broccoli_production_increase_l995_99581


namespace NUMINAMATH_CALUDE_nested_root_simplification_l995_99573

theorem nested_root_simplification (b : ℝ) (h : b > 0) :
  (((b^16)^(1/3))^(1/4))^3 * (((b^16)^(1/4))^(1/3))^3 = b^8 := by
sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l995_99573


namespace NUMINAMATH_CALUDE_specific_sculpture_surface_area_l995_99538

/-- Represents a cube sculpture with a 3x3 bottom layer and a cross-shaped top layer --/
structure CubeSculpture where
  cubeEdgeLength : ℝ
  bottomLayerSize : ℕ
  topLayerSize : ℕ

/-- Calculates the exposed surface area of the cube sculpture --/
def exposedSurfaceArea (sculpture : CubeSculpture) : ℝ :=
  sorry

/-- Theorem stating that the exposed surface area of the specific sculpture is 46 square meters --/
theorem specific_sculpture_surface_area :
  let sculpture : CubeSculpture := {
    cubeEdgeLength := 1,
    bottomLayerSize := 3,
    topLayerSize := 5
  }
  exposedSurfaceArea sculpture = 46 := by sorry

end NUMINAMATH_CALUDE_specific_sculpture_surface_area_l995_99538


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l995_99587

theorem tan_theta_minus_pi_over_four (θ : ℝ) :
  (Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)).re = 0 →
  Real.tan (θ - π/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l995_99587


namespace NUMINAMATH_CALUDE_sum_of_digits_less_than_1000_is_13500_l995_99551

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def sum_of_digits_less_than_1000 : ℕ :=
  (List.range 1000).map digit_sum |> List.sum

theorem sum_of_digits_less_than_1000_is_13500 :
  sum_of_digits_less_than_1000 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_less_than_1000_is_13500_l995_99551


namespace NUMINAMATH_CALUDE_small_prob_event_cannot_occur_is_false_l995_99516

-- Define a probability space
variable (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

-- Define an event as a measurable set
def Event (Ω : Type) [MeasurableSpace Ω] := {A : Set Ω // MeasurableSet A}

-- Define a very small probability
def VerySmallProbability (ε : ℝ) : Prop := 0 < ε ∧ ε < 1/1000000

-- Statement: An event with a very small probability cannot occur
theorem small_prob_event_cannot_occur_is_false :
  ∃ (A : Event Ω) (ε : ℝ), VerySmallProbability ε ∧ P A < ε ∧ ¬(P A = 0) :=
sorry

end NUMINAMATH_CALUDE_small_prob_event_cannot_occur_is_false_l995_99516


namespace NUMINAMATH_CALUDE_water_tank_capacity_l995_99548

theorem water_tank_capacity (c : ℝ) : 
  (c / 3 + 10) / c = 2 / 5 → c = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l995_99548


namespace NUMINAMATH_CALUDE_carly_running_ratio_l995_99503

def week1_distance : ℝ := 2
def week2_distance : ℝ := 2 * week1_distance + 3
def week4_distance : ℝ := 4
def week3_distance : ℝ := week4_distance + 5

theorem carly_running_ratio :
  week3_distance / week2_distance = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_carly_running_ratio_l995_99503


namespace NUMINAMATH_CALUDE_triangle_height_decrease_l995_99591

theorem triangle_height_decrease (b h : ℝ) (b_new h_new : ℝ) :
  b > 0 → h > 0 →
  b_new = 1.1 * b →
  (1/2) * b_new * h_new = 1.045 * ((1/2) * b * h) →
  h_new = 0.5 * h := by
sorry

end NUMINAMATH_CALUDE_triangle_height_decrease_l995_99591


namespace NUMINAMATH_CALUDE_cuboid_area_example_l995_99531

/-- The surface area of a cuboid -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 3 cm, length 4 cm, and height 5 cm is 94 cm² -/
theorem cuboid_area_example : cuboid_surface_area 3 4 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l995_99531


namespace NUMINAMATH_CALUDE_min_value_theorem_l995_99568

theorem min_value_theorem (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x_0 : ℝ, a * x_0^2 + 2 * x_0 + b = 0) :
  (∀ a b : ℝ, 2 * a^2 + b^2 ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, 2 * a^2 + b^2 = 2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l995_99568


namespace NUMINAMATH_CALUDE_problem_2021_l995_99554

theorem problem_2021 : (2021^2 - 2020) / 2021 + 7 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_problem_2021_l995_99554


namespace NUMINAMATH_CALUDE_remaining_travel_distance_l995_99592

theorem remaining_travel_distance 
  (total_distance : ℕ)
  (amoli_speed : ℕ)
  (amoli_time : ℕ)
  (anayet_speed : ℕ)
  (anayet_time : ℕ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_speed = 61)
  (h5 : anayet_time = 2) :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time) = 121 :=
by sorry

end NUMINAMATH_CALUDE_remaining_travel_distance_l995_99592


namespace NUMINAMATH_CALUDE_k_range_for_not_in_second_quadrant_l995_99563

/-- A linear function that does not pass through the second quadrant -/
structure LinearFunctionNotInSecondQuadrant where
  k : ℝ
  not_in_second_quadrant : ∀ x y : ℝ, y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)

/-- The range of k for a linear function not passing through the second quadrant -/
theorem k_range_for_not_in_second_quadrant (f : LinearFunctionNotInSecondQuadrant) : f.k ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_k_range_for_not_in_second_quadrant_l995_99563


namespace NUMINAMATH_CALUDE_passengers_after_first_stop_l995_99526

/-- 
Given a train with an initial number of passengers and some passengers getting off at the first stop,
this theorem proves the number of passengers remaining after the first stop.
-/
theorem passengers_after_first_stop 
  (initial_passengers : ℕ) 
  (passengers_left : ℕ) 
  (h1 : initial_passengers = 48)
  (h2 : passengers_left = initial_passengers - 17) : 
  passengers_left = 31 := by
  sorry

end NUMINAMATH_CALUDE_passengers_after_first_stop_l995_99526


namespace NUMINAMATH_CALUDE_right_triangle_area_l995_99572

/-- The area of a right triangle with hypotenuse 9 inches and one angle 30° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 9 →  -- hypotenuse is 9 inches
  α = 30 * π / 180 →  -- one angle is 30°
  area = (9^2 * Real.sin (30 * π / 180) * Real.sin (60 * π / 180)) / 4 →  -- area formula for right triangle
  area = 10.125 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l995_99572


namespace NUMINAMATH_CALUDE_perfect_squares_l995_99585

theorem perfect_squares (m n a : ℝ) (h : a = m * n) : 
  ((m - n) / 2)^2 + a = ((m + n) / 2)^2 ∧ 
  ((m + n) / 2)^2 - a = ((m - n) / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_l995_99585


namespace NUMINAMATH_CALUDE_bales_stored_l995_99507

/-- Given the initial number of bales and the final number of bales,
    prove that Jason stored 23 bales in the barn. -/
theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73)
  (h2 : final_bales = 96) :
  final_bales - initial_bales = 23 := by
  sorry

end NUMINAMATH_CALUDE_bales_stored_l995_99507


namespace NUMINAMATH_CALUDE_riverside_academy_statistics_l995_99517

/-- The number of students taking statistics at Riverside Academy -/
def students_taking_statistics (total_students : ℕ) (physics_students : ℕ) (both_subjects : ℕ) : ℕ :=
  total_students - (physics_students - both_subjects)

/-- Theorem: The number of students taking statistics is 21 -/
theorem riverside_academy_statistics :
  let total_students : ℕ := 25
  let physics_students : ℕ := 10
  let both_subjects : ℕ := 6
  students_taking_statistics total_students physics_students both_subjects = 21 := by
  sorry

end NUMINAMATH_CALUDE_riverside_academy_statistics_l995_99517


namespace NUMINAMATH_CALUDE_total_earnings_is_4440_l995_99533

/-- Represents a car type with its rental rate -/
structure CarType where
  name : String
  rate : Nat

/-- Represents a rental record -/
structure Rental where
  carType : CarType
  duration : Nat  -- in minutes

def redCar : CarType := ⟨"red", 3⟩
def whiteCar : CarType := ⟨"white", 2⟩
def blueCar : CarType := ⟨"blue", 4⟩
def greenCar : CarType := ⟨"green", 5⟩

def rentals : List Rental := [
  ⟨redCar, 240⟩,
  ⟨redCar, 180⟩,
  ⟨redCar, 180⟩,
  ⟨whiteCar, 150⟩,
  ⟨whiteCar, 210⟩,
  ⟨blueCar, 90⟩,
  ⟨blueCar, 240⟩,
  ⟨greenCar, 120⟩
]

def rentalEarnings (r : Rental) : Nat :=
  r.carType.rate * r.duration

def totalEarnings : Nat :=
  (rentals.map rentalEarnings).sum

theorem total_earnings_is_4440 : totalEarnings = 4440 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_4440_l995_99533


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l995_99504

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ k : ℕ, k > 0 ∧ k ∣ n → k ≤ w ∧ w = 12 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l995_99504


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_y_value_l995_99569

theorem arithmetic_geometric_progression_y_value
  (x y z : ℝ)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (arithmetic_prog : 2 * y = x + z)
  (geometric_prog1 : ∃ r : ℝ, r ≠ 0 ∧ -y = r * (x + 1) ∧ z = r * (-y))
  (geometric_prog2 : ∃ s : ℝ, s ≠ 0 ∧ y = s * x ∧ z + 2 = s * y) :
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_y_value_l995_99569


namespace NUMINAMATH_CALUDE_tank_capacity_l995_99582

theorem tank_capacity (x : ℝ) 
  (h1 : (5/6 : ℝ) * x - 15 = (2/3 : ℝ) * x) : x = 90 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l995_99582


namespace NUMINAMATH_CALUDE_monomial_exponents_l995_99514

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℤ) : Prop :=
  ∀ x y, ∃ k₁ k₂ : ℤ, k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ a x y = k₁ ∧ b x y = k₂

theorem monomial_exponents (m n : ℕ) :
  like_terms (fun x y => 6 * x^5 * y^(2*n)) (fun x y => -2 * x^m * y^4) →
  m + 2*n = 9 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_l995_99514


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l995_99590

/-- Given 4 siblings where 3 are 4, 5, and 7 years older than the youngest,
    and their average age is 21, prove that the age of the youngest sibling is 17. -/
theorem youngest_sibling_age (y : ℕ) : 
  (y + (y + 4) + (y + 5) + (y + 7)) / 4 = 21 → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l995_99590


namespace NUMINAMATH_CALUDE_equation_solutions_l995_99579

def solution_set : Set (ℤ × ℤ) :=
  {(-15, -3), (-1, -1), (2, 14), (3, -21), (5, -7), (6, -6), (20, -4)}

def satisfies_equation (pair : ℤ × ℤ) : Prop :=
  let (x, y) := pair
  x ≠ 0 ∧ y ≠ 0 ∧ (5 : ℚ) / x - (7 : ℚ) / y = 2

theorem equation_solutions :
  ∀ (x y : ℤ), satisfies_equation (x, y) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l995_99579


namespace NUMINAMATH_CALUDE_smallest_integer_lower_bound_l995_99539

theorem smallest_integer_lower_bound 
  (a b c d : ℤ) 
  (different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (average : (a + b + c + d) / 4 = 76) 
  (largest : d = 90) 
  (ordered : a ≤ b ∧ b ≤ c ∧ c ≤ d) : 
  a ≥ 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_lower_bound_l995_99539


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l995_99549

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 52) :
  (perimeter / 4) ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l995_99549


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_squares_l995_99557

theorem rectangle_perimeter_from_squares (square_area : Real) (h : square_area = 25) : 
  let side_length := Real.sqrt square_area
  let rectangle_length := 2 * side_length
  let rectangle_width := side_length
  2 * (rectangle_length + rectangle_width) = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_from_squares_l995_99557


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l995_99550

theorem square_minus_product_plus_square : 5^2 - 3*4 + 3^2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l995_99550


namespace NUMINAMATH_CALUDE_complex_inequality_condition_l995_99558

theorem complex_inequality_condition (z : ℂ) :
  (∀ z, Complex.abs z ≤ 1 → Complex.abs (Complex.re z) ≤ 1 ∧ Complex.abs (Complex.im z) ≤ 1) ∧
  (∃ z, Complex.abs (Complex.re z) ≤ 1 ∧ Complex.abs (Complex.im z) ≤ 1 ∧ Complex.abs z > 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_condition_l995_99558


namespace NUMINAMATH_CALUDE_work_days_calculation_l995_99513

/-- Proves that A and B worked together for 10 days given the conditions -/
theorem work_days_calculation (a_rate : ℚ) (b_rate : ℚ) (remaining_work : ℚ) : 
  a_rate = 1 / 30 →
  b_rate = 1 / 40 →
  remaining_work = 5 / 12 →
  ∃ d : ℚ, d = 10 ∧ (a_rate + b_rate) * d = 1 - remaining_work :=
by sorry

end NUMINAMATH_CALUDE_work_days_calculation_l995_99513


namespace NUMINAMATH_CALUDE_complex_calculation_l995_99540

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Define w as a function of z
def w (z : ℂ) : ℂ := z^2 + 3 - 4

-- Theorem statement
theorem complex_calculation :
  w z = 2 * Complex.I - 1 := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l995_99540


namespace NUMINAMATH_CALUDE_parking_arrangement_equality_parking_spaces_count_l995_99537

/-- Number of arrangements of k elements from n elements -/
def A (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of parking spaces -/
def n : ℕ := sorry

/-- Theorem stating the equality of probabilities for different parking arrangements -/
theorem parking_arrangement_equality : A (n - 2) 3 = A 3 2 * A (n - 2) 2 := by sorry

/-- Theorem proving that n equals 10 -/
theorem parking_spaces_count : n = 10 := by sorry

end NUMINAMATH_CALUDE_parking_arrangement_equality_parking_spaces_count_l995_99537


namespace NUMINAMATH_CALUDE_marbles_exchange_l995_99534

/-- Represents the number of marbles each person has -/
structure Marbles where
  tyrone : ℕ
  eric : ℕ

/-- The initial state of marbles -/
def initial : Marbles := ⟨150, 30⟩

/-- The number of marbles Tyrone gives to Eric -/
def marbles_given : ℕ := sorry

/-- The final state of marbles after the exchange -/
def final : Marbles := ⟨initial.tyrone - marbles_given, initial.eric + marbles_given⟩

theorem marbles_exchange :
  (final.tyrone = 3 * initial.eric) ∧ (marbles_given = 60) := by sorry

end NUMINAMATH_CALUDE_marbles_exchange_l995_99534


namespace NUMINAMATH_CALUDE_solve_system_with_equal_xy_l995_99543

theorem solve_system_with_equal_xy (x y n : ℝ) 
  (eq1 : 5 * x - 4 * y = n)
  (eq2 : 3 * x + 5 * y = 8)
  (eq3 : x = y) :
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_with_equal_xy_l995_99543


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l995_99518

theorem ten_thousandths_place_of_5_32 :
  (5 : ℚ) / 32 = 0.15625 := by sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l995_99518


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l995_99500

theorem quadratic_completion_of_square (b : ℝ) (p : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/6 = (x+p)^2 + 1/18) → 
  b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l995_99500


namespace NUMINAMATH_CALUDE_sphere_in_cube_volume_l995_99567

/-- The volume of a sphere inscribed in a cube of edge length 2 -/
theorem sphere_in_cube_volume :
  let cube_edge : ℝ := 2
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_volume_l995_99567


namespace NUMINAMATH_CALUDE_abs_two_x_minus_one_lt_one_l995_99527

theorem abs_two_x_minus_one_lt_one (x y : ℝ) 
  (h1 : |x - y - 1| ≤ 1/3) 
  (h2 : |2*y + 1| ≤ 1/6) : 
  |2*x - 1| < 1 := by
sorry

end NUMINAMATH_CALUDE_abs_two_x_minus_one_lt_one_l995_99527


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l995_99547

theorem sum_of_four_primes_divisible_by_60 
  (p q r s : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hs : Nat.Prime s) 
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) : 
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l995_99547


namespace NUMINAMATH_CALUDE_optimal_discount_order_l995_99559

def book_price : ℚ := 30
def flat_discount : ℚ := 5
def percentage_discount : ℚ := 0.25

def price_flat_then_percent : ℚ := (book_price - flat_discount) * (1 - percentage_discount)
def price_percent_then_flat : ℚ := book_price * (1 - percentage_discount) - flat_discount

theorem optimal_discount_order :
  price_percent_then_flat < price_flat_then_percent ∧
  price_flat_then_percent - price_percent_then_flat = 125 / 100 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_order_l995_99559


namespace NUMINAMATH_CALUDE_correct_simplification_l995_99566

theorem correct_simplification (a b : ℝ) : 5*a - (b - 1) = 5*a - b + 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_simplification_l995_99566


namespace NUMINAMATH_CALUDE_dave_candy_pieces_l995_99530

/-- Calculates the number of candy pieces Dave has left after giving some boxes away. -/
def candyPiecesLeft (initialBoxes : ℕ) (boxesGivenAway : ℕ) (piecesPerBox : ℕ) : ℕ :=
  (initialBoxes - boxesGivenAway) * piecesPerBox

/-- Proves that Dave has 21 pieces of candy left. -/
theorem dave_candy_pieces : 
  candyPiecesLeft 12 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_dave_candy_pieces_l995_99530


namespace NUMINAMATH_CALUDE_dot_product_specific_value_l995_99580

/-- Dot product of two 3D vectors -/
def dot_product (a b c p q r : ℝ) : ℝ := a * p + b * q + c * r

theorem dot_product_specific_value :
  let y : ℝ := 12.5
  let n : ℝ := dot_product 3 4 5 y (-2) 1
  n = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_specific_value_l995_99580


namespace NUMINAMATH_CALUDE_club_assignment_count_l995_99597

/-- Represents the four clubs --/
inductive Club
| Literature
| Drama
| Anime
| Love

/-- Represents the five students --/
inductive Student
| A
| B
| C
| D
| E

/-- A valid club assignment is a function from Student to Club --/
def ClubAssignment := Student → Club

/-- Checks if a club assignment is valid according to the problem conditions --/
def is_valid_assignment (assignment : ClubAssignment) : Prop :=
  (∀ c : Club, ∃ s : Student, assignment s = c) ∧ 
  (assignment Student.A ≠ Club.Anime)

/-- The number of valid club assignments --/
def num_valid_assignments : ℕ := sorry

theorem club_assignment_count : num_valid_assignments = 180 := by sorry

end NUMINAMATH_CALUDE_club_assignment_count_l995_99597


namespace NUMINAMATH_CALUDE_complement_of_union_l995_99535

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define set M
def M : Set Nat := {2, 4}

-- Define set N
def N : Set Nat := {0, 4}

-- Theorem statement
theorem complement_of_union (U M N : Set Nat) : 
  U = {0, 1, 2, 3, 4} → M = {2, 4} → N = {0, 4} → 
  (U \ (M ∪ N)) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l995_99535


namespace NUMINAMATH_CALUDE_inequality_proof_l995_99555

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l995_99555


namespace NUMINAMATH_CALUDE_distribution_methods_eq_240_l995_99529

/-- The number of ways to distribute 5 volunteers into 4 groups and assign them to intersections -/
def distributionMethods : ℕ := 
  (Nat.choose 5 2) * (Nat.factorial 4)

/-- Theorem stating that the number of distribution methods is 240 -/
theorem distribution_methods_eq_240 : distributionMethods = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribution_methods_eq_240_l995_99529


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l995_99576

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l995_99576


namespace NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l995_99574

/-- Represents the ratios in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation of the drink -/
def sport_ratio : DrinkRatio :=
  ⟨1, 
   3 * standard_ratio.corn_syrup / standard_ratio.flavoring,
   standard_ratio.water / (2 * standard_ratio.flavoring)⟩

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def corn_syrup_amount : ℚ := 8

/-- Theorem stating the amount of water in the sport formulation -/
theorem water_amount_in_sport_formulation :
  (corn_syrup_amount * sport_ratio.water) / sport_ratio.corn_syrup = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l995_99574


namespace NUMINAMATH_CALUDE_yard_length_with_26_trees_l995_99505

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * tree_distance

theorem yard_length_with_26_trees :
  yard_length 26 11 = 275 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_26_trees_l995_99505


namespace NUMINAMATH_CALUDE_simplify_expression_l995_99532

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) :
  let M := a - b
  (2 * a) / (a^2 - b^2) - 1 / M = 1 / (a + b) := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l995_99532


namespace NUMINAMATH_CALUDE_smallest_chocolate_beverage_volume_l995_99575

/-- Represents the ratio of milk to syrup in the chocolate beverage -/
def milk_syrup_ratio : ℚ := 5 / 2

/-- Volume of milk in each bottle (in liters) -/
def milk_bottle_volume : ℚ := 2

/-- Volume of syrup in each bottle (in liters) -/
def syrup_bottle_volume : ℚ := 14 / 10

/-- Finds the smallest number of whole bottles of milk and syrup that satisfy the ratio -/
def find_smallest_bottles : ℕ × ℕ := (7, 4)

/-- Calculates the total volume of the chocolate beverage -/
def total_volume (bottles : ℕ × ℕ) : ℚ :=
  milk_bottle_volume * bottles.1 + syrup_bottle_volume * bottles.2

/-- Theorem stating that the smallest volume of chocolate beverage that can be made
    using only whole bottles of milk and syrup is 19.6 L -/
theorem smallest_chocolate_beverage_volume :
  total_volume (find_smallest_bottles) = 196 / 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_chocolate_beverage_volume_l995_99575


namespace NUMINAMATH_CALUDE_calculate_expression_l995_99536

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l995_99536


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l995_99556

theorem arctan_sum_special_case (a b : ℝ) : 
  a = 1/3 → (a + 1) * (b + 1) = 5/2 → Real.arctan a + Real.arctan b = Real.arctan (29/17) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l995_99556


namespace NUMINAMATH_CALUDE_line_through_points_l995_99596

/-- Theorem: Line passing through specific points with given conditions -/
theorem line_through_points (k x y : ℚ) : 
  (k + 4) / 4 = k →  -- slope condition
  x - y = 2 →        -- condition on x and y
  k - x = 3 →        -- condition on k and x
  k = 4/3 ∧ x = -5/3 ∧ y = -11/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l995_99596


namespace NUMINAMATH_CALUDE_three_n_squared_plus_nine_composite_l995_99562

theorem three_n_squared_plus_nine_composite (n : ℕ) : ∃ (k : ℕ), k > 1 ∧ k ∣ (3 * n^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_three_n_squared_plus_nine_composite_l995_99562


namespace NUMINAMATH_CALUDE_sequence_length_correct_l995_99506

/-- The number of terms in the arithmetic sequence from 5 to 2n-1 with a common difference of 2 -/
def sequence_length (n : ℕ) : ℕ :=
  n - 2

/-- The nth term of the sequence -/
def sequence_term (n : ℕ) : ℕ :=
  2 * n + 3

theorem sequence_length_correct (n : ℕ) :
  sequence_term (sequence_length n) = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_correct_l995_99506


namespace NUMINAMATH_CALUDE_chord_passes_through_fixed_point_l995_99546

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Parabola C with equation x^2 = 4y -/
def parabolaC (p : Point) : Prop :=
  p.x^2 = 4 * p.y

/-- Dot product of two vectors represented by points -/
def dotProduct (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- Condition that the dot product of OA and OB is -4 -/
def dotProductCondition (a b : Point) : Prop :=
  dotProduct a b = -4

/-- Line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem stating that if a chord AB of parabola C satisfies the dot product condition,
    then the line AB always passes through the point (0, 2) -/
theorem chord_passes_through_fixed_point 
  (a b : Point) (l : Line) 
  (h1 : parabolaC a) 
  (h2 : parabolaC b) 
  (h3 : dotProductCondition a b) 
  (h4 : linePassesThrough l a) 
  (h5 : linePassesThrough l b) : 
  linePassesThrough l (Point.mk 0 2) :=
sorry

end NUMINAMATH_CALUDE_chord_passes_through_fixed_point_l995_99546


namespace NUMINAMATH_CALUDE_larger_number_four_times_smaller_l995_99522

theorem larger_number_four_times_smaller
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_distinct : a ≠ b)
  (h_equation : a^3 - b^3 = 3*(2*a^2*b - 3*a*b^2 + b^3)) :
  a = 4*b :=
sorry

end NUMINAMATH_CALUDE_larger_number_four_times_smaller_l995_99522


namespace NUMINAMATH_CALUDE_total_shells_l995_99552

theorem total_shells (morning_shells afternoon_shells : ℕ) 
  (h1 : morning_shells = 292)
  (h2 : afternoon_shells = 324) :
  morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l995_99552


namespace NUMINAMATH_CALUDE_arithmetic_computation_l995_99542

theorem arithmetic_computation : 5 + 4 * (2 - 7)^2 = 105 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l995_99542


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l995_99528

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SpecificSum (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SpecificSum a) : 
  a 9 - (1/2) * a 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l995_99528
