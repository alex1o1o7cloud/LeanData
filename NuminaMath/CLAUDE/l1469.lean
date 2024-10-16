import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l1469_146954

theorem min_value_theorem (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  ∃ (min : ℝ), min = -9/8 ∧ ∀ (z : ℝ), z = x + y + x * y → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1469_146954


namespace NUMINAMATH_CALUDE_sum_of_first_eight_multiples_of_eleven_l1469_146941

/-- The sum of the first n distinct positive integer multiples of m -/
def sum_of_multiples (n m : ℕ) : ℕ := 
  m * n * (n + 1) / 2

/-- Theorem: The sum of the first 8 distinct positive integer multiples of 11 is 396 -/
theorem sum_of_first_eight_multiples_of_eleven : 
  sum_of_multiples 8 11 = 396 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_eight_multiples_of_eleven_l1469_146941


namespace NUMINAMATH_CALUDE_no_three_numbers_with_special_property_l1469_146947

theorem no_three_numbers_with_special_property : 
  ¬ (∃ (a b c : ℕ), 
    a > 1 ∧ b > 1 ∧ c > 1 ∧
    b ∣ (a^2 - 1) ∧ c ∣ (a^2 - 1) ∧
    a ∣ (b^2 - 1) ∧ c ∣ (b^2 - 1) ∧
    a ∣ (c^2 - 1) ∧ b ∣ (c^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_three_numbers_with_special_property_l1469_146947


namespace NUMINAMATH_CALUDE_function_monotonicity_l1469_146945

theorem function_monotonicity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x^2 - 3*x + 2) * (deriv (deriv f) x) < 0) :
  ∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 := by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l1469_146945


namespace NUMINAMATH_CALUDE_more_than_three_solutions_l1469_146930

/-- Represents a trapezoid with bases b₁ and b₂, and height h -/
structure Trapezoid where
  b₁ : ℕ
  b₂ : ℕ
  h : ℕ

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℕ :=
  (t.b₁ + t.b₂) * t.h / 2

/-- Predicate for valid trapezoid solutions -/
def isValidSolution (m n : ℕ) : Prop :=
  m + n = 6 ∧
  10 ∣ (10 * m) ∧
  10 ∣ (10 * n) ∧
  area { b₁ := 10 * m, b₂ := 10 * n, h := 60 } = 1800

theorem more_than_three_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card > 3 ∧ ∀ (p : ℕ × ℕ), p ∈ S → isValidSolution p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_more_than_three_solutions_l1469_146930


namespace NUMINAMATH_CALUDE_locus_of_M_l1469_146905

/-- Circle with center at the origin and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Line parallel to y-axis at distance a from origin -/
def Line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = a}

/-- Polar of point A with respect to circle -/
def Polar (r a β : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + β * p.2 = r^2}

/-- Perpendicular line from A to e -/
def Perpendicular (β : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = β}

/-- Locus of point M -/
def Locus (r a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = r^2 - a * p.1}

theorem locus_of_M (r a : ℝ) (hr : r > 0) (ha : a ≠ 0) :
  ∀ β : ℝ, ∃ M : ℝ × ℝ,
    M ∈ Polar r a β ∩ Perpendicular β →
    M ∈ Locus r a :=
  sorry

end NUMINAMATH_CALUDE_locus_of_M_l1469_146905


namespace NUMINAMATH_CALUDE_guesthouse_fixed_rate_l1469_146970

/-- A guesthouse charging system with a fixed rate for the first night and an additional fee for subsequent nights. -/
structure Guesthouse where
  first_night : ℕ  -- Fixed rate for the first night
  subsequent : ℕ  -- Fee for each subsequent night

/-- The total cost for a stay at the guesthouse. -/
def total_cost (g : Guesthouse) (nights : ℕ) : ℕ :=
  g.first_night + g.subsequent * (nights - 1)

theorem guesthouse_fixed_rate :
  ∃ (g : Guesthouse),
    total_cost g 5 = 220 ∧
    total_cost g 8 = 370 ∧
    g.first_night = 20 := by
  sorry

end NUMINAMATH_CALUDE_guesthouse_fixed_rate_l1469_146970


namespace NUMINAMATH_CALUDE_units_digit_of_17_times_24_l1469_146955

theorem units_digit_of_17_times_24 : (17 * 24) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_times_24_l1469_146955


namespace NUMINAMATH_CALUDE_correct_operation_l1469_146953

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1469_146953


namespace NUMINAMATH_CALUDE_fabulous_iff_not_power_of_two_l1469_146916

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (n ∣ a^n - a)

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem fabulous_iff_not_power_of_two (n : ℕ) :
  is_fabulous n ↔ (¬ is_power_of_two n) :=
sorry

end NUMINAMATH_CALUDE_fabulous_iff_not_power_of_two_l1469_146916


namespace NUMINAMATH_CALUDE_only_first_is_prime_one_prime_in_sequence_l1469_146917

/-- Generates the nth number in the sequence starting with 47 and repeating 47 sequentially -/
def sequenceNumber (n : ℕ) : ℕ :=
  if n = 0 then 47 else
  (sequenceNumber (n - 1)) * 100 + 47

/-- Theorem stating that only the first number in the sequence is prime -/
theorem only_first_is_prime :
  ∀ n : ℕ, n > 0 → ¬ Nat.Prime (sequenceNumber n) :=
by
  sorry

/-- Corollary stating that there is exactly one prime number in the sequence -/
theorem one_prime_in_sequence :
  (∃! k : ℕ, Nat.Prime (sequenceNumber k)) :=
by
  sorry

end NUMINAMATH_CALUDE_only_first_is_prime_one_prime_in_sequence_l1469_146917


namespace NUMINAMATH_CALUDE_difference_of_squares_problem_1_problem_2_l1469_146904

-- Difference of squares formula
theorem difference_of_squares (a b : ℤ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

-- Problem 1
theorem problem_1 : 3001 * 2999 = 3000^2 - 1^2 := by sorry

-- Problem 2
theorem problem_2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) = 2^64 - 1 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_problem_1_problem_2_l1469_146904


namespace NUMINAMATH_CALUDE_dice_sides_proof_l1469_146952

theorem dice_sides_proof (n : ℕ) (h : n ≥ 3) :
  (3 / n^2 : ℚ)^2 = 1/9 → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_dice_sides_proof_l1469_146952


namespace NUMINAMATH_CALUDE_laptop_price_theorem_l1469_146975

/-- The sticker price of the laptop. -/
def sticker_price : ℝ := 250

/-- The price at store A after discount and rebate. -/
def price_A (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at store B after discount and rebate. -/
def price_B (x : ℝ) : ℝ := 0.7 * x - 50

/-- Theorem stating that the sticker price satisfies the given conditions. -/
theorem laptop_price_theorem : 
  price_A sticker_price = price_B sticker_price - 25 := by
  sorry

#check laptop_price_theorem

end NUMINAMATH_CALUDE_laptop_price_theorem_l1469_146975


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_x_plus_2y_equals_5_l1469_146981

theorem positive_integer_solutions_of_x_plus_2y_equals_5 :
  {(x, y) : ℕ × ℕ | x + 2 * y = 5 ∧ x > 0 ∧ y > 0} = {(1, 2), (3, 1)} := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_x_plus_2y_equals_5_l1469_146981


namespace NUMINAMATH_CALUDE_line_direction_vector_c_l1469_146982

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (c : ℝ) : Prop :=
  let direction := (p2.1 - p1.1, p2.2 - p1.2)
  direction.1 = 3 ∧ direction.2 = c

/-- Theorem stating that for a line passing through (-6, 1) and (-3, 4) with direction vector (3, c), c must equal 3 -/
theorem line_direction_vector_c (c : ℝ) :
  Line (-6, 1) (-3, 4) c → c = 3 := by
  sorry


end NUMINAMATH_CALUDE_line_direction_vector_c_l1469_146982


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1469_146918

theorem solution_set_equivalence (x : ℝ) : 
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1469_146918


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1469_146907

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1469_146907


namespace NUMINAMATH_CALUDE_newberg_total_landed_l1469_146984

/-- Represents the passenger data for an airport -/
structure AirportData where
  onTime : ℕ
  late : ℕ
  cancelled : ℕ

/-- Calculates the total number of landed passengers, excluding cancelled flights -/
def totalLanded (data : AirportData) : ℕ :=
  data.onTime + data.late

/-- Theorem: The total number of passengers who landed in Newberg last year is 28,690 -/
theorem newberg_total_landed :
  let airportA : AirportData := ⟨16507, 256, 198⟩
  let airportB : AirportData := ⟨11792, 135, 151⟩
  totalLanded airportA + totalLanded airportB = 28690 := by
  sorry


end NUMINAMATH_CALUDE_newberg_total_landed_l1469_146984


namespace NUMINAMATH_CALUDE_equation_solution_l1469_146915

theorem equation_solution : 
  ∃! x : ℚ, 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) + 2 * x ∧ x = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1469_146915


namespace NUMINAMATH_CALUDE_intersection_distance_l1469_146967

theorem intersection_distance : ∃ (p₁ p₂ : ℝ × ℝ),
  (p₁.1^2 + p₁.2 = 10 ∧ p₁.1 + p₁.2 = 10) ∧
  (p₂.1^2 + p₂.2 = 10 ∧ p₂.1 + p₂.2 = 10) ∧
  p₁ ≠ p₂ ∧
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1469_146967


namespace NUMINAMATH_CALUDE_triangle_formation_l1469_146913

/-- Determines if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given groups of numbers --/
def group_A : (ℝ × ℝ × ℝ) := (5, 7, 12)
def group_B : (ℝ × ℝ × ℝ) := (7, 7, 15)
def group_C : (ℝ × ℝ × ℝ) := (6, 9, 16)
def group_D : (ℝ × ℝ × ℝ) := (6, 8, 12)

theorem triangle_formation :
  ¬(can_form_triangle group_A.1 group_A.2.1 group_A.2.2) ∧
  ¬(can_form_triangle group_B.1 group_B.2.1 group_B.2.2) ∧
  ¬(can_form_triangle group_C.1 group_C.2.1 group_C.2.2) ∧
  can_form_triangle group_D.1 group_D.2.1 group_D.2.2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1469_146913


namespace NUMINAMATH_CALUDE_max_rooms_less_than_55_l1469_146946

/-- Represents the number of rooms with different combinations of bouquets -/
structure RoomCounts where
  chrysOnly : ℕ
  carnOnly : ℕ
  roseOnly : ℕ
  chrysCarn : ℕ
  chrysRose : ℕ
  carnRose : ℕ
  allThree : ℕ

/-- The conditions of the mansion and its bouquets -/
def MansionConditions (r : RoomCounts) : Prop :=
  r.chrysCarn = 2 ∧
  r.chrysRose = 3 ∧
  r.carnRose = 4 ∧
  r.chrysOnly + r.chrysCarn + r.chrysRose + r.allThree = 10 ∧
  r.carnOnly + r.chrysCarn + r.carnRose + r.allThree = 20 ∧
  r.roseOnly + r.chrysRose + r.carnRose + r.allThree = 30

/-- The total number of rooms in the mansion -/
def totalRooms (r : RoomCounts) : ℕ :=
  r.chrysOnly + r.carnOnly + r.roseOnly + r.chrysCarn + r.chrysRose + r.carnRose + r.allThree

/-- Theorem stating that the maximum number of rooms is less than 55 -/
theorem max_rooms_less_than_55 (r : RoomCounts) (h : MansionConditions r) : 
  totalRooms r < 55 := by
  sorry


end NUMINAMATH_CALUDE_max_rooms_less_than_55_l1469_146946


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l1469_146968

theorem three_digit_cube_divisible_by_16 : 
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 16 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l1469_146968


namespace NUMINAMATH_CALUDE_oliver_birthday_gift_l1469_146956

/-- The amount of money Oliver's friend gave him on his birthday --/
def friend_gift (initial_amount savings frisbee_cost puzzle_cost final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount + savings - frisbee_cost - puzzle_cost)

theorem oliver_birthday_gift :
  friend_gift 9 5 4 3 15 = 8 :=
by sorry

end NUMINAMATH_CALUDE_oliver_birthday_gift_l1469_146956


namespace NUMINAMATH_CALUDE_f_min_value_l1469_146919

/-- The function f(x) = x^2 + 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 15

/-- Theorem: The minimum value of f(x) = x^2 + 8x + 15 is -1 -/
theorem f_min_value : ∃ (a : ℝ), f a = -1 ∧ ∀ (x : ℝ), f x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l1469_146919


namespace NUMINAMATH_CALUDE_sine_shift_left_l1469_146999

/-- Shifting a sine function to the left -/
theorem sine_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let shift : ℝ := π / 6
  let g (t : ℝ) := f (t + shift)
  g x = Real.sin (x + π / 6) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_left_l1469_146999


namespace NUMINAMATH_CALUDE_x_squared_eq_y_squared_necessary_not_sufficient_l1469_146976

theorem x_squared_eq_y_squared_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → x^2 = y^2) ∧
  (∃ x y : ℝ, x^2 = y^2 ∧ x ≠ y) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_y_squared_necessary_not_sufficient_l1469_146976


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l1469_146971

theorem sum_of_odd_numbers (N : ℕ) : 
  991 + 993 + 995 + 997 + 999 = 5000 - N → N = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l1469_146971


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l1469_146901

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l1469_146901


namespace NUMINAMATH_CALUDE_negation_of_both_even_l1469_146908

theorem negation_of_both_even (a b : ℤ) : 
  ¬(Even a ∧ Even b) ↔ ¬(Even a) ∨ ¬(Even b) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_both_even_l1469_146908


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1469_146993

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1469_146993


namespace NUMINAMATH_CALUDE_anna_transportation_tax_l1469_146902

/-- Calculates the transportation tax for a vehicle -/
def calculate_tax (engine_power : ℕ) (tax_rate : ℕ) (months_owned : ℕ) (months_in_year : ℕ) : ℕ :=
  (engine_power * tax_rate * months_owned) / months_in_year

/-- Represents the transportation tax problem for Anna Ivanovna -/
theorem anna_transportation_tax :
  let engine_power : ℕ := 250
  let tax_rate : ℕ := 75
  let months_owned : ℕ := 2
  let months_in_year : ℕ := 12
  calculate_tax engine_power tax_rate months_owned months_in_year = 3125 := by
  sorry


end NUMINAMATH_CALUDE_anna_transportation_tax_l1469_146902


namespace NUMINAMATH_CALUDE_megan_markers_theorem_l1469_146964

/-- The number of markers Megan had initially -/
def initial_markers : ℕ := sorry

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := 109

/-- The total number of markers Megan has now -/
def total_markers : ℕ := 326

/-- Theorem stating that the initial number of markers plus the markers given by Robert equals the total number of markers Megan has now -/
theorem megan_markers_theorem : initial_markers + roberts_markers = total_markers := by sorry

end NUMINAMATH_CALUDE_megan_markers_theorem_l1469_146964


namespace NUMINAMATH_CALUDE_number_of_daughters_l1469_146988

theorem number_of_daughters (a : ℕ) : 
  (a.Prime) → 
  (64 + a^2 = 16*a + 1) → 
  a = 7 := by sorry

end NUMINAMATH_CALUDE_number_of_daughters_l1469_146988


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l1469_146996

theorem max_value_x_sqrt_1_minus_4x_squared (x : ℝ) :
  0 < x → x < 1/2 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l1469_146996


namespace NUMINAMATH_CALUDE_inequality_proof_l1469_146962

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a * b) ≤ (1 / 3) * Real.sqrt ((a^2 + b^2) / 2) + (2 / 3) * (2 / (1 / a + 1 / b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1469_146962


namespace NUMINAMATH_CALUDE_f_two_roots_implies_a_gt_three_l1469_146931

/-- The function f(x) = x³ - ax² + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- f has two distinct positive roots -/
def has_two_distinct_positive_roots (a : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

theorem f_two_roots_implies_a_gt_three (a : ℝ) :
  has_two_distinct_positive_roots a → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_f_two_roots_implies_a_gt_three_l1469_146931


namespace NUMINAMATH_CALUDE_diver_min_trips_l1469_146929

/-- Calculates the minimum number of trips required to transport objects --/
def min_trips (objects_per_trip : ℕ) (total_objects : ℕ) : ℕ :=
  (total_objects + objects_per_trip - 1) / objects_per_trip

/-- Theorem: Given a diver who can carry 3 objects at a time and has found 17 objects,
    the minimum number of trips required to transport all objects is 6 --/
theorem diver_min_trips :
  min_trips 3 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_diver_min_trips_l1469_146929


namespace NUMINAMATH_CALUDE_postage_for_420g_book_l1469_146927

/-- Calculates the postage cost for mailing a book in China. -/
def postage_cost (weight : ℕ) : ℚ :=
  let base_rate : ℚ := 7/10
  let additional_rate : ℚ := 4/10
  let base_weight : ℕ := 100
  let additional_weight := (weight - 1) / base_weight + 1
  base_rate + additional_rate * additional_weight

/-- Theorem stating that the postage cost for a 420g book is 2.3 yuan. -/
theorem postage_for_420g_book :
  postage_cost 420 = 23/10 := by sorry

end NUMINAMATH_CALUDE_postage_for_420g_book_l1469_146927


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1469_146900

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x - f y) = f (f x) - f y - 1) : 
  (∀ x : ℤ, f x = -1) ∨ (∀ x : ℤ, f x = x + 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1469_146900


namespace NUMINAMATH_CALUDE_ketchup_mustard_arrangement_l1469_146949

/-- Represents the number of ways to arrange ketchup and mustard bottles. -/
def arrange_bottles (k m : ℕ) : ℕ := sorry

/-- The property that no ketchup bottle is between two mustard bottles. -/
def valid_arrangement (k m : ℕ) (arrangement : List Bool) : Prop := sorry

/-- The main theorem stating the number of valid arrangements. -/
theorem ketchup_mustard_arrangement :
  ∃ (n : ℕ), 
    (n = arrange_bottles 3 7) ∧ 
    (∀ arrangement : List Bool, 
      (arrangement.length = 10) →
      (arrangement.count true = 3) →
      (arrangement.count false = 7) →
      valid_arrangement 3 7 arrangement) ∧
    n = 22 := by sorry

end NUMINAMATH_CALUDE_ketchup_mustard_arrangement_l1469_146949


namespace NUMINAMATH_CALUDE_goldfish_count_l1469_146914

/-- The number of goldfish in the aquarium -/
def total_goldfish : ℕ := 100

/-- The number of goldfish Maggie was allowed to take home -/
def allowed_goldfish : ℕ := total_goldfish / 2

/-- The number of goldfish Maggie caught -/
def caught_goldfish : ℕ := (3 * allowed_goldfish) / 5

/-- The number of goldfish Maggie still needs to catch -/
def remaining_goldfish : ℕ := 20

theorem goldfish_count : 
  total_goldfish = 100 ∧
  allowed_goldfish = total_goldfish / 2 ∧
  caught_goldfish = (3 * allowed_goldfish) / 5 ∧
  remaining_goldfish = allowed_goldfish - caught_goldfish ∧
  remaining_goldfish = 20 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l1469_146914


namespace NUMINAMATH_CALUDE_trees_space_theorem_l1469_146909

/-- Calculates the total space required for fruit trees in Quinton's yard -/
def total_space_for_trees (apple_width peach_width apple_space peach_space : ℕ) : ℕ :=
  (2 * apple_width + apple_space) + (2 * peach_width + peach_space)

/-- Theorem stating that the total space required for the given tree configuration is 71 feet -/
theorem trees_space_theorem : 
  total_space_for_trees 10 12 12 15 = 71 := by
  sorry

#eval total_space_for_trees 10 12 12 15

end NUMINAMATH_CALUDE_trees_space_theorem_l1469_146909


namespace NUMINAMATH_CALUDE_spinner_final_direction_l1469_146910

-- Define the four cardinal directions
inductive Direction
| North
| East
| South
| West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- State the theorem
theorem spinner_final_direction :
  let initial_direction := Direction.North
  let clockwise_turns := 5 + 1/2
  let counterclockwise_turns := 2 + 3/4
  rotate (rotate initial_direction clockwise_turns) (-counterclockwise_turns) = Direction.West :=
by sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l1469_146910


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l1469_146937

theorem stamp_collection_problem (current_stamps : ℕ) : 
  (current_stamps : ℚ) * (1 + 20 / 100) = 48 → current_stamps = 40 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l1469_146937


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1469_146957

theorem complex_equation_solution (i : ℂ) (x : ℂ) (h1 : i * i = -1) (h2 : i * x = 1 + i) : x = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1469_146957


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1469_146994

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m b : ℝ) : ℝ := m

/-- The slope of a line in the form ax + y + c = 0 is -a -/
def slope_of_general_line (a c : ℝ) : ℝ := -a

theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope_of_general_line a (-5)) (slope_of_line 7 (-2)) → a = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1469_146994


namespace NUMINAMATH_CALUDE_first_course_cost_proof_l1469_146998

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

end NUMINAMATH_CALUDE_first_course_cost_proof_l1469_146998


namespace NUMINAMATH_CALUDE_sqrt_three_solution_l1469_146997

theorem sqrt_three_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 3*a) : a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_solution_l1469_146997


namespace NUMINAMATH_CALUDE_power_of_two_100_l1469_146979

theorem power_of_two_100 :
  (10^30 : ℕ) ≤ 2^100 ∧ 2^100 < (10^31 : ℕ) ∧ 2^100 % 1000 = 376 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_100_l1469_146979


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1469_146987

/-- The number of people in the line -/
def total_people : ℕ := 6

/-- The number of people in the adjacent group (Xiao Kai and 2 elderly) -/
def adjacent_group : ℕ := 3

/-- The number of volunteers -/
def volunteers : ℕ := 3

/-- The number of ways to arrange the adjacent group internally -/
def adjacent_group_arrangements : ℕ := 2

/-- The number of possible positions for the adjacent group in the line -/
def adjacent_group_positions : ℕ := total_people - adjacent_group - 1

/-- The number of arrangements for the volunteers -/
def volunteer_arrangements : ℕ := 6

theorem number_of_arrangements :
  adjacent_group_arrangements * adjacent_group_positions * volunteer_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1469_146987


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1469_146923

-- Define the sample space
def Ω : Type := Fin 3 → Fin 3

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A: sum of numbers drawn is 6
def A : Set Ω := {ω : Ω | ω 0 + ω 1 + ω 2 = 5}

-- Define event B: number 2 is drawn three times
def B : Set Ω := {ω : Ω | ∀ i, ω i = 1}

-- State the theorem
theorem conditional_probability_B_given_A :
  P B / P A = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1469_146923


namespace NUMINAMATH_CALUDE_smallest_block_size_block_with_336_cubes_exists_l1469_146948

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of invisible cubes when viewed from a corner -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Calculates the total number of cubes in the block -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem stating the smallest possible number of cubes in the block -/
theorem smallest_block_size (d : BlockDimensions) :
  invisibleCubes d = 143 → totalCubes d ≥ 336 := by
  sorry

/-- Theorem proving the existence of a block with 336 cubes and 143 invisible cubes -/
theorem block_with_336_cubes_exists :
  ∃ d : BlockDimensions, invisibleCubes d = 143 ∧ totalCubes d = 336 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_block_with_336_cubes_exists_l1469_146948


namespace NUMINAMATH_CALUDE_trig_inequality_l1469_146932

theorem trig_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (h_eq : Real.sin x = x * Real.cos y) : 
  x/2 < y ∧ y < x :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1469_146932


namespace NUMINAMATH_CALUDE_gasoline_reduction_l1469_146983

theorem gasoline_reduction (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.2 * P
  let new_spending := 1.08 * (P * Q)
  let new_quantity := new_spending / new_price
  (Q - new_quantity) / Q = 0.1 := by
sorry

end NUMINAMATH_CALUDE_gasoline_reduction_l1469_146983


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1469_146990

theorem cubic_expression_value (x : ℝ) (h : x^2 - 2*x - 1 = 0) :
  x^3 - x^2 - 3*x + 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1469_146990


namespace NUMINAMATH_CALUDE_triangle_side_length_l1469_146944

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → B = π/4 → b = Real.sqrt 6 - Real.sqrt 2 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b / Real.sin B = c / Real.sin C →
  c = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1469_146944


namespace NUMINAMATH_CALUDE_workers_savings_l1469_146977

/-- Proves that if a worker saves 1/3 of their constant monthly take-home pay for 12 months, 
    the total amount saved is 6 times the amount not saved in one month. -/
theorem workers_savings (monthly_pay : ℝ) : 
  monthly_pay > 0 →
  let monthly_savings := (1/3) * monthly_pay
  let monthly_not_saved := monthly_pay - monthly_savings
  let total_savings := 12 * monthly_savings
  total_savings = 6 * monthly_not_saved := by
sorry


end NUMINAMATH_CALUDE_workers_savings_l1469_146977


namespace NUMINAMATH_CALUDE_expression_evaluation_l1469_146961

theorem expression_evaluation :
  let x : ℝ := -3
  let numerator := 5 + x * (5 + x) - 5^2
  let denominator := x - 5 + x^2
  numerator / denominator = -26 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1469_146961


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l1469_146989

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 3}

-- Define the range of m
def m_range : Set ℝ := {m | m < -4 ∨ m > 2}

-- Theorem statement
theorem subset_implies_m_range :
  ∀ m : ℝ, B m ⊆ A → m ∈ m_range :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l1469_146989


namespace NUMINAMATH_CALUDE_min_distance_PQ_l1469_146972

def f (x : ℝ) : ℝ := x^2 - 2*x

def distance_squared (x : ℝ) : ℝ := (x - 4)^2 + (f x + 1)^2

theorem min_distance_PQ :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x : ℝ), Real.sqrt (distance_squared x) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_PQ_l1469_146972


namespace NUMINAMATH_CALUDE_penny_throwing_ratio_l1469_146978

/-- Given the conditions of the penny-throwing problem, prove that the ratio of Rocky's pennies to Gretchen's is 1:3 -/
theorem penny_throwing_ratio (rachelle gretchen rocky : ℕ) : 
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rachelle + gretchen + rocky = 300 →
  rocky / gretchen = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_penny_throwing_ratio_l1469_146978


namespace NUMINAMATH_CALUDE_max_distinct_dance_counts_29_15_l1469_146963

/-- Represents the maximum number of distinct dance counts that can be reported
    given a number of boys and girls at a ball, where each boy can dance with
    each girl at most once. -/
def max_distinct_dance_counts (num_boys num_girls : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 29 boys and 15 girls, the maximum number of
    distinct dance counts is 29. -/
theorem max_distinct_dance_counts_29_15 :
  max_distinct_dance_counts 29 15 = 29 := by sorry

end NUMINAMATH_CALUDE_max_distinct_dance_counts_29_15_l1469_146963


namespace NUMINAMATH_CALUDE_tan_sin_cos_relation_l1469_146951

theorem tan_sin_cos_relation (α : Real) (h : Real.tan α = -3) :
  (Real.sin α = 3 * Real.sqrt 10 / 10 ∨ Real.sin α = -3 * Real.sqrt 10 / 10) ∧
  (Real.cos α = Real.sqrt 10 / 10 ∨ Real.cos α = -Real.sqrt 10 / 10) :=
by sorry

end NUMINAMATH_CALUDE_tan_sin_cos_relation_l1469_146951


namespace NUMINAMATH_CALUDE_manufacturing_plant_optimization_l1469_146920

noncomputable def f (x : ℝ) : ℝ := 4 * (1 - x) * x^2

def domain (t : ℝ) (x : ℝ) : Prop := 0 < x ∧ x ≤ 2*t/(2*t+1)

theorem manufacturing_plant_optimization (t : ℝ) 
  (h1 : 0 < t) (h2 : t ≤ 2) :
  (f 0.5 = 0.5) ∧
  (∀ x, domain t x →
    (1 ≤ t → f x ≤ 16/27 ∧ (f x = 16/27 → x = 2/3)) ∧
    (t < 1 → f x ≤ 16*t^2/(2*t+1)^3 ∧ (f x = 16*t^2/(2*t+1)^3 → x = 2*t/(2*t+1)))) :=
by sorry

end NUMINAMATH_CALUDE_manufacturing_plant_optimization_l1469_146920


namespace NUMINAMATH_CALUDE_dihedral_angle_adjacent_faces_l1469_146958

/-- Given a regular n-sided pyramid with base dihedral angle α, 
    the dihedral angle φ between adjacent lateral faces satisfies:
    cos(φ/2) = sin(α) * sin(π/n) -/
theorem dihedral_angle_adjacent_faces 
  (n : ℕ) 
  (α : ℝ) 
  (h_n : n ≥ 3) 
  (h_α : 0 < α ∧ α < π / 2) : 
  ∃ φ : ℝ, 
    0 < φ ∧ 
    φ < π ∧ 
    Real.cos (φ / 2) = Real.sin α * Real.sin (π / n) := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_adjacent_faces_l1469_146958


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1469_146936

theorem smallest_common_factor (n : ℕ) : 
  (∀ k < 60, ¬ ∃ m > 1, m ∣ (11 * k - 6) ∧ m ∣ (8 * k + 5)) ∧
  (∃ m > 1, m ∣ (11 * 60 - 6) ∧ m ∣ (8 * 60 + 5)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1469_146936


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_k_range_l1469_146973

theorem ellipse_eccentricity_k_range (k : ℝ) (e : ℝ) :
  (∃ x y : ℝ, x^2 / k + y^2 / 4 = 1) →
  (1/2 < e ∧ e < 1) →
  (0 < k ∧ k < 3) ∨ (16/3 < k) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_k_range_l1469_146973


namespace NUMINAMATH_CALUDE_divisibility_by_two_in_odd_base_system_l1469_146943

theorem divisibility_by_two_in_odd_base_system (d : ℕ) (h_odd : Odd d) :
  ∀ (x : ℕ) (digits : List ℕ),
    (x = digits.foldr (λ a acc => a + d * acc) 0) →
    (x % 2 = 0 ↔ digits.sum % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_two_in_odd_base_system_l1469_146943


namespace NUMINAMATH_CALUDE_proposition_and_variants_l1469_146926

theorem proposition_and_variants (x y : ℝ) :
  -- Original proposition
  (x^2 + y^2 = 0 → x * y = 0) ∧
  -- Converse (false)
  ¬(x * y = 0 → x^2 + y^2 = 0) ∧
  -- Inverse (false)
  ¬(x^2 + y^2 ≠ 0 → x * y ≠ 0) ∧
  -- Contrapositive (true)
  (x * y ≠ 0 → x^2 + y^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_and_variants_l1469_146926


namespace NUMINAMATH_CALUDE_stating_race_outcomes_count_l1469_146974

/-- Represents the number of participants in the race -/
def total_participants : ℕ := 6

/-- Represents the number of top positions we're considering -/
def top_positions : ℕ := 3

/-- Represents the number of participants eligible for top positions -/
def eligible_participants : ℕ := total_participants - 1

/-- 
Calculates the number of different outcomes for top positions in a race
given the number of eligible participants and the number of top positions,
assuming no ties.
-/
def race_outcomes (eligible : ℕ) (positions : ℕ) : ℕ :=
  (eligible - positions + 1).factorial / (eligible - positions).factorial

/-- 
Theorem stating that the number of different 1st-2nd-3rd place outcomes
in a race with 6 participants, where one participant cannot finish 
in the top three and there are no ties, is equal to 60.
-/
theorem race_outcomes_count : 
  race_outcomes eligible_participants top_positions = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_race_outcomes_count_l1469_146974


namespace NUMINAMATH_CALUDE_intersection_point_on_x_equals_4_l1469_146942

/-- An ellipse with center at origin and foci on coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.equation (l : Line) (p : Point) : Prop :=
  p.y = l.k * (p.x - l.m)

/-- The main theorem -/
theorem intersection_point_on_x_equals_4 
  (e : Ellipse)
  (h_passes_through_A : e.equation ⟨-2, 0⟩)
  (h_passes_through_B : e.equation ⟨2, 0⟩)
  (h_passes_through_C : e.equation ⟨1, 3/2⟩)
  (l : Line)
  (h_k_nonzero : l.k ≠ 0)
  (M N : Point)
  (h_M_on_E : e.equation M)
  (h_N_on_E : e.equation N)
  (h_M_on_l : l.equation M)
  (h_N_on_l : l.equation N) :
  ∃ (P : Point), P.x = 4 ∧ 
    (∃ (t : ℝ), P = ⟨4, t * (M.y + 2) + (1 - t) * M.y⟩) ∧
    (∃ (s : ℝ), P = ⟨4, s * (N.y - 2) + (1 - s) * N.y⟩) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_x_equals_4_l1469_146942


namespace NUMINAMATH_CALUDE_total_rainfall_is_23_inches_l1469_146912

/-- Calculates the total rainfall over three days given specific conditions --/
def totalRainfall (mondayHours : ℝ) (mondayRate : ℝ) 
                  (tuesdayHours : ℝ) (tuesdayRate : ℝ)
                  (wednesdayHours : ℝ) : ℝ :=
  mondayHours * mondayRate + 
  tuesdayHours * tuesdayRate + 
  wednesdayHours * (2 * tuesdayRate)

/-- Proves that the total rainfall over the three days is 23 inches --/
theorem total_rainfall_is_23_inches : 
  totalRainfall 7 1 4 2 2 = 23 := by
  sorry


end NUMINAMATH_CALUDE_total_rainfall_is_23_inches_l1469_146912


namespace NUMINAMATH_CALUDE_tank_capacity_l1469_146938

/-- Represents the capacity of a tank and its inlet/outlet properties. -/
structure Tank where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- The tank satisfies the given conditions. -/
def satisfies_conditions (t : Tank) : Prop :=
  t.outlet_time = 5 ∧
  t.inlet_rate = 8 * 60 ∧
  t.combined_time = t.outlet_time + 3

/-- The theorem stating that a tank satisfying the given conditions has a capacity of 6400 litres. -/
theorem tank_capacity (t : Tank) (h : satisfies_conditions t) : t.capacity = 6400 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1469_146938


namespace NUMINAMATH_CALUDE_number_of_roses_rose_count_proof_l1469_146906

theorem number_of_roses (vase_capacity : ℕ) (num_carnations : ℕ) (num_vases : ℕ) : ℕ :=
  let total_flowers := vase_capacity * num_vases
  total_flowers - num_carnations

theorem rose_count_proof 
  (vase_capacity : ℕ) 
  (num_carnations : ℕ) 
  (num_vases : ℕ) 
  (h1 : vase_capacity = 6) 
  (h2 : num_carnations = 7) 
  (h3 : num_vases = 9) : 
  number_of_roses vase_capacity num_carnations num_vases = 47 := by
  sorry

end NUMINAMATH_CALUDE_number_of_roses_rose_count_proof_l1469_146906


namespace NUMINAMATH_CALUDE_monic_quartic_value_l1469_146985

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value (p : ℝ → ℝ) :
  is_monic_quartic p →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 5 = 40 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_value_l1469_146985


namespace NUMINAMATH_CALUDE_omega_value_l1469_146928

theorem omega_value (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x + π / 6)) →
  ω > 0 →
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 3 → f x < f y) →
  f (π / 4) = f (π / 2) →
  ω = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_omega_value_l1469_146928


namespace NUMINAMATH_CALUDE_equation_system_solution_l1469_146960

theorem equation_system_solution (a b x y : ℝ) 
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : Real.log (Real.sqrt a) / Real.log (a^(1/x)) + Real.log (Real.sqrt b) / Real.log (b^(1/y)) = a / Real.sqrt 3) :
  x = a * Real.sqrt 3 / 3 ∧ y = a * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1469_146960


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1469_146924

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_at_neg_two : (fun x => a * x^2 + b * x + c) (-2) = a^2
  passes_through_point : a * (-1)^2 + b * (-1) + c = 6

/-- Theorem stating that for a quadratic function with given properties, (a+c)/b = 1/2 -/
theorem quadratic_function_property (f : QuadraticFunction) : (f.a + f.c) / f.b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1469_146924


namespace NUMINAMATH_CALUDE_l_shaped_grid_squares_l1469_146995

/-- Represents a modified L-shaped grid -/
structure LShapedGrid :=
  (size : Nat)
  (missing_size : Nat)
  (missing_row : Nat)
  (missing_col : Nat)

/-- Counts the number of squares in the L-shaped grid -/
def count_squares (grid : LShapedGrid) : Nat :=
  sorry

/-- The main theorem stating that the number of squares in the specific L-shaped grid is 61 -/
theorem l_shaped_grid_squares :
  let grid : LShapedGrid := {
    size := 6,
    missing_size := 2,
    missing_row := 5,
    missing_col := 1
  }
  count_squares grid = 61 := by sorry

end NUMINAMATH_CALUDE_l_shaped_grid_squares_l1469_146995


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1469_146965

/-- Given an isosceles right triangle with a square inscribed as described in Figure 1
    with an area of 256 cm², prove that the area of the square inscribed as described
    in Figure 2 is 576 - 256√2 cm². -/
theorem inscribed_square_area (s : ℝ) (h1 : s^2 = 256) : ∃ S : ℝ,
  S^2 = 576 - 256 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1469_146965


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l1469_146933

def lapTime1 : ℕ := 5
def lapTime2 : ℕ := 8
def lapTime3 : ℕ := 9
def startTime : ℕ := 7 * 60  -- 7:00 AM in minutes since midnight

def meetingTime : ℕ := startTime + Nat.lcm (Nat.lcm lapTime1 lapTime2) lapTime3

theorem earliest_meeting_time :
  meetingTime = 13 * 60  -- 1:00 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l1469_146933


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l1469_146939

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : elec = 45)
  (h4 : both = 32) :
  total - cs - elec + both = 15 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l1469_146939


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1469_146940

def A : Set Int := {-1, 0, 1, 2}
def B : Set Int := {-2, 0, 2, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1469_146940


namespace NUMINAMATH_CALUDE_select_team_count_l1469_146925

/-- The number of players in the basketball team -/
def total_players : ℕ := 12

/-- The number of players to be selected for the team -/
def team_size : ℕ := 5

/-- The number of twins in the team -/
def num_twins : ℕ := 2

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of ways to select the team with the given conditions -/
def select_team : ℕ := binomial total_players team_size - binomial (total_players - num_twins) (team_size - num_twins)

theorem select_team_count : select_team = 672 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l1469_146925


namespace NUMINAMATH_CALUDE_ellipse_iff_m_range_l1469_146903

-- Define the equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (2 + m) - y^2 / (m + 1) = 1

-- Define the condition for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    ellipse_equation x y m ↔ x^2 / a^2 + y^2 / b^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

-- State the theorem
theorem ellipse_iff_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_iff_m_range_l1469_146903


namespace NUMINAMATH_CALUDE_min_sum_abs_l1469_146959

theorem min_sum_abs (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abs_l1469_146959


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l1469_146922

/-- Represents the peg game with n holes and k pegs. -/
structure PegGame where
  n : ℕ
  k : ℕ
  h1 : 1 ≤ k
  h2 : k < n

/-- Predicate that determines if Alice has a winning strategy. -/
def alice_wins (game : PegGame) : Prop :=
  ¬(Even game.n ∧ Even game.k)

/-- The main theorem about Alice's winning strategy in the peg game. -/
theorem alice_winning_strategy (game : PegGame) :
  alice_wins game ↔
  (∃ (strategy : Unit), 
    (∀ (bob_move : Unit), ∃ (alice_move : Unit), 
      -- Alice can always make a move that leads to a winning position
      true)) := by sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l1469_146922


namespace NUMINAMATH_CALUDE_quadratic_polynomials_theorem_l1469_146935

theorem quadratic_polynomials_theorem (a b c d : ℝ) : 
  let f (x : ℝ) := x^2 + a*x + b
  let g (x : ℝ) := x^2 + c*x + d
  (∀ x, f x ≠ g x) →  -- f and g are distinct
  g (-a/2) = 0 →  -- x-coordinate of vertex of f is a root of g
  f (-c/2) = 0 →  -- x-coordinate of vertex of g is a root of f
  f 50 = -50 ∧ g 50 = -50 →  -- f and g intersect at (50, -50)
  (∃ x₁ x₂, ∀ x, f x ≥ f x₁ ∧ g x ≥ g x₂ ∧ f x₁ = g x₂) →  -- minimum value of f is the same as g
  a + c = -200 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_theorem_l1469_146935


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l1469_146969

theorem power_of_two_plus_one (b m n : ℕ) : 
  b > 1 → 
  m > n → 
  (∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) → 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l1469_146969


namespace NUMINAMATH_CALUDE_stating_nth_smallest_d₀_is_correct_l1469_146921

/-- 
Given a non-negative integer d₀ and a positive integer v,
this function returns true if v² = 8d₀, false otherwise.
-/
def is_valid_pair (d₀ v : ℕ) : Prop :=
  v^2 = 8 * d₀

/-- 
This function returns the nth smallest non-negative integer d₀
such that there exists a positive integer v where v² = 8d₀.
-/
def nth_smallest_d₀ (n : ℕ) : ℕ :=
  4^(n-1)

/-- 
Theorem stating that nth_smallest_d₀ correctly computes
the nth smallest d₀ satisfying the required property.
-/
theorem nth_smallest_d₀_is_correct (n : ℕ) :
  n > 0 →
  (∃ v : ℕ, is_valid_pair (nth_smallest_d₀ n) v) ∧
  (∀ d : ℕ, d < nth_smallest_d₀ n →
    (∃ v : ℕ, is_valid_pair d v) →
    (∃ k < n, d = nth_smallest_d₀ k)) :=
by sorry

end NUMINAMATH_CALUDE_stating_nth_smallest_d₀_is_correct_l1469_146921


namespace NUMINAMATH_CALUDE_cougar_sleep_duration_l1469_146980

/-- Given a cougar's nightly sleep duration C and a zebra's nightly sleep duration Z,
    where Z = C + 2 and C + Z = 70, prove that C = 34. -/
theorem cougar_sleep_duration (C Z : ℕ) (h1 : Z = C + 2) (h2 : C + Z = 70) : C = 34 := by
  sorry

end NUMINAMATH_CALUDE_cougar_sleep_duration_l1469_146980


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1469_146986

/-- Given that a, 6, and b form an arithmetic sequence in that order, prove that a + b = 12 -/
theorem arithmetic_sequence_sum (a b : ℝ) 
  (h : ∃ d : ℝ, a + d = 6 ∧ b = a + 2*d) : 
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1469_146986


namespace NUMINAMATH_CALUDE_total_squares_is_86_l1469_146991

/-- The number of squares of a given size in a 6x6 grid -/
def count_squares (size : Nat) : Nat :=
  (7 - size) ^ 2

/-- The total number of squares of sizes 1x1, 2x2, 3x3, and 4x4 in a 6x6 grid -/
def total_squares : Nat :=
  count_squares 1 + count_squares 2 + count_squares 3 + count_squares 4

theorem total_squares_is_86 : total_squares = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_is_86_l1469_146991


namespace NUMINAMATH_CALUDE_max_distance_complex_unit_circle_l1469_146911

theorem max_distance_complex_unit_circle (z : ℂ) :
  Complex.abs z = 1 → Complex.abs (z - (3 - 4 * Complex.I)) ≤ 6 ∧ 
  ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - (3 - 4 * Complex.I)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_unit_circle_l1469_146911


namespace NUMINAMATH_CALUDE_power_equation_solution_l1469_146950

theorem power_equation_solution :
  ∃ (x : ℕ), (12 : ℝ)^x * 6^4 / 432 = 5184 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1469_146950


namespace NUMINAMATH_CALUDE_taylor_family_reunion_l1469_146992

theorem taylor_family_reunion (kids : ℕ) (adults : ℕ) (tables : ℕ) 
  (h1 : kids = 45) 
  (h2 : adults = 123) 
  (h3 : tables = 14) : 
  (kids + adults) / tables = 12 := by
sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_l1469_146992


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_l1469_146934

theorem cube_sum_minus_product (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a * b + a * c + b * c = 40) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1575 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_l1469_146934


namespace NUMINAMATH_CALUDE_factorization_a_squared_plus_2a_l1469_146966

theorem factorization_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a*(a+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_plus_2a_l1469_146966
