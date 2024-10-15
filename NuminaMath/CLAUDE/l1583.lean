import Mathlib

namespace NUMINAMATH_CALUDE_speed_to_arrive_on_time_l1583_158314

/-- The speed required to arrive on time given late and early arrival conditions -/
theorem speed_to_arrive_on_time (d : ℝ) (t : ℝ) (h1 : d = 50 * (t + 1/12)) (h2 : d = 70 * (t - 1/12)) : 
  d / t = 58 := by
  sorry

end NUMINAMATH_CALUDE_speed_to_arrive_on_time_l1583_158314


namespace NUMINAMATH_CALUDE_geometric_ratio_from_arithmetic_l1583_158305

/-- An arithmetic sequence with a non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b (n + 1) = r * b n

/-- The theorem statement -/
theorem geometric_ratio_from_arithmetic (a : ℕ → ℝ) (d : ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a d →
  (∃ k, b k = a 1 ∧ b (k + 1) = a 3 ∧ b (k + 2) = a 7) →
  ∃ r, geometric_sequence b r ∧ r = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_ratio_from_arithmetic_l1583_158305


namespace NUMINAMATH_CALUDE_canada_population_1998_l1583_158392

/-- Proves that 30.3 million is equal to 30,300,000 --/
theorem canada_population_1998 : (30.3 : ℝ) * 1000000 = 30300000 := by
  sorry

end NUMINAMATH_CALUDE_canada_population_1998_l1583_158392


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1583_158366

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1583_158366


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1583_158339

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (12 * q) * Real.sqrt (8 * q^2) * Real.sqrt (9 * q^5) = 12 * q^4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1583_158339


namespace NUMINAMATH_CALUDE_total_books_l1583_158336

-- Define the number of books for each person
def sam_books : ℕ := 110

-- Joan has twice as many books as Sam
def joan_books : ℕ := 2 * sam_books

-- Tom has half the number of books as Joan
def tom_books : ℕ := joan_books / 2

-- Alice has 3 times the number of books Tom has
def alice_books : ℕ := 3 * tom_books

-- Theorem statement
theorem total_books : sam_books + joan_books + tom_books + alice_books = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1583_158336


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1583_158321

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π -/
theorem cylinder_surface_area :
  ∀ (h : ℝ) (c : ℝ),
  h = 2 →
  c = 2 * Real.pi →
  2 * Real.pi * (c / (2 * Real.pi))^2 + c * h = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1583_158321


namespace NUMINAMATH_CALUDE_complex_equation_solution_product_l1583_158380

theorem complex_equation_solution_product (x : ℂ) :
  x^3 + x^2 + 3*x = 2 + 2*Complex.I →
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    x₁^3 + x₁^2 + 3*x₁ = 2 + 2*Complex.I ∧
    x₂^3 + x₂^2 + 3*x₂ = 2 + 2*Complex.I ∧
    (x₁.re * x₂.re = 1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_product_l1583_158380


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1583_158365

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 8, 10} : Set ℤ) →
  (b^2 * (3*b - 2) % 5 ≠ 0 ↔ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1583_158365


namespace NUMINAMATH_CALUDE_original_number_l1583_158310

theorem original_number : ∃ x : ℕ, 100 * x - x = 1980 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1583_158310


namespace NUMINAMATH_CALUDE_square_difference_204_202_l1583_158334

theorem square_difference_204_202 : 204^2 - 202^2 = 812 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_204_202_l1583_158334


namespace NUMINAMATH_CALUDE_solution_set_a_eq_1_min_value_range_l1583_158309

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 3

-- Theorem 1: Solution set for a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x ≤ 4} = Set.Icc 0 (1/2) := by sorry

-- Theorem 2: Range of a for which f(x) has a minimum value
theorem min_value_range :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) ↔ a ∈ Set.Icc (-3) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_1_min_value_range_l1583_158309


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_l1583_158316

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

-- Theorem for the solution set of f(x) > 1
theorem solution_set_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∃ x, f x + 4 ≥ |1 - 2*m|} = {m : ℝ | -6 ≤ m ∧ m ≤ 8} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_l1583_158316


namespace NUMINAMATH_CALUDE_third_difference_zero_implies_quadratic_l1583_158379

/-- A function from integers to real numbers -/
def IntFunction := ℤ → ℝ

/-- The third difference of a function -/
def thirdDifference (f : IntFunction) : IntFunction :=
  fun n => f (n + 3) - 3 * f (n + 2) + 3 * f (n + 1) - f n

/-- A function is quadratic if it can be expressed as a*n^2 + b*n + c for some real a, b, c -/
def isQuadratic (f : IntFunction) : Prop :=
  ∃ a b c : ℝ, ∀ n : ℤ, f n = a * n^2 + b * n + c

theorem third_difference_zero_implies_quadratic (f : IntFunction) 
  (h : ∀ n : ℤ, thirdDifference f n = 0) : 
  isQuadratic f := by
  sorry

end NUMINAMATH_CALUDE_third_difference_zero_implies_quadratic_l1583_158379


namespace NUMINAMATH_CALUDE_distribute_balls_result_l1583_158351

/-- The number of ways to distribute balls to students -/
def distribute_balls (red black white : ℕ) : ℕ :=
  let min_boys := 2
  let min_girl := 3
  let remaining_red := red - (2 * min_boys + min_girl)
  let remaining_black := black - (2 * min_boys + min_girl)
  let remaining_white := white - (2 * min_boys + min_girl)
  (Nat.choose (remaining_red + 2) 2) *
  (Nat.choose (remaining_black + 2) 2) *
  (Nat.choose (remaining_white + 2) 2)

/-- Theorem stating the number of ways to distribute the balls -/
theorem distribute_balls_result : distribute_balls 10 15 20 = 47250 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_result_l1583_158351


namespace NUMINAMATH_CALUDE_set_equality_solution_l1583_158360

theorem set_equality_solution (x y : ℝ) : 
  ({x, y, x + y} : Set ℝ) = ({0, x^2, x*y} : Set ℝ) →
  ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_solution_l1583_158360


namespace NUMINAMATH_CALUDE_cos_equality_angle_l1583_158373

theorem cos_equality_angle (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (370 * π / 180) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_angle_l1583_158373


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1583_158391

/-- Given a point M(x₀, y₀) outside the circle x² + y² = 2,
    prove that the line x₀x + y₀y = 2 intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 > 2) :
  ∃ (x y : ℝ), x^2 + y^2 = 2 ∧ x₀*x + y₀*y = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1583_158391


namespace NUMINAMATH_CALUDE_school_supplies_cost_l1583_158330

theorem school_supplies_cost :
  let pencil_cartons : ℕ := 20
  let pencil_boxes_per_carton : ℕ := 10
  let pencil_box_cost : ℕ := 2
  let marker_cartons : ℕ := 10
  let marker_boxes_per_carton : ℕ := 5
  let marker_box_cost : ℕ := 4
  
  pencil_cartons * pencil_boxes_per_carton * pencil_box_cost +
  marker_cartons * marker_boxes_per_carton * marker_box_cost = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l1583_158330


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l1583_158377

theorem product_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (h3 : z + 1/x = 2) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l1583_158377


namespace NUMINAMATH_CALUDE_cube_sum_equals_negative_27_l1583_158381

theorem cube_sum_equals_negative_27 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) : 
  a^3 + b^3 + c^3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_negative_27_l1583_158381


namespace NUMINAMATH_CALUDE_quadratic_properties_l1583_158304

/-- A quadratic function passing through (-3, 0) with axis of symmetry x = -1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  passes_through_minus_three : a * (-3)^2 + b * (-3) + c = 0
  axis_of_symmetry : -b / (2 * a) = -1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a + f.b + f.c = 0) ∧
  (2 * f.c + 3 * f.b = 0) ∧
  (∀ k : ℝ, k > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f.a * x₁^2 + f.b * x₁ + f.c = k * (x₁ + 1) ∧
    f.a * x₂^2 + f.b * x₂ + f.c = k * (x₂ + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1583_158304


namespace NUMINAMATH_CALUDE_water_price_l1583_158342

/-- Given that six bottles of 2 liters of water cost $12, prove that the price of 1 liter of water is $1. -/
theorem water_price (bottles : ℕ) (liters_per_bottle : ℝ) (total_cost : ℝ) :
  bottles = 6 →
  liters_per_bottle = 2 →
  total_cost = 12 →
  total_cost / (bottles * liters_per_bottle) = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_price_l1583_158342


namespace NUMINAMATH_CALUDE_selection_ways_l1583_158345

def total_students : ℕ := 10
def selected_students : ℕ := 4
def specific_students : ℕ := 2

theorem selection_ways : 
  (Nat.choose total_students selected_students - 
   Nat.choose (total_students - specific_students) selected_students) = 140 :=
by sorry

end NUMINAMATH_CALUDE_selection_ways_l1583_158345


namespace NUMINAMATH_CALUDE_eggs_eaten_in_morning_l1583_158315

theorem eggs_eaten_in_morning (initial_eggs : ℕ) (afternoon_eggs : ℕ) (remaining_eggs : ℕ) :
  initial_eggs = 20 →
  afternoon_eggs = 3 →
  remaining_eggs = 13 →
  initial_eggs - remaining_eggs - afternoon_eggs = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_eaten_in_morning_l1583_158315


namespace NUMINAMATH_CALUDE_specific_garage_full_spots_l1583_158302

/-- Represents a parking garage with given specifications -/
structure ParkingGarage where
  stories : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsThirdLevel : Nat
  openSpotsFourthLevel : Nat

/-- Calculates the number of full parking spots in the garage -/
def fullParkingSpots (garage : ParkingGarage) : Nat :=
  garage.stories * garage.spotsPerLevel - 
  (garage.openSpotsFirstLevel + garage.openSpotsSecondLevel + 
   garage.openSpotsThirdLevel + garage.openSpotsFourthLevel)

/-- Theorem stating the number of full parking spots in the specific garage -/
theorem specific_garage_full_spots :
  ∃ (garage : ParkingGarage),
    garage.stories = 4 ∧
    garage.spotsPerLevel = 100 ∧
    garage.openSpotsFirstLevel = 58 ∧
    garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2 ∧
    garage.openSpotsThirdLevel = garage.openSpotsSecondLevel + 5 ∧
    garage.openSpotsFourthLevel = 31 ∧
    fullParkingSpots garage = 186 := by
  sorry

end NUMINAMATH_CALUDE_specific_garage_full_spots_l1583_158302


namespace NUMINAMATH_CALUDE_smallest_n_with_divisibility_l1583_158375

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_n_with_divisibility : ∃! N : ℕ, 
  N > 0 ∧ 
  (is_divisible N (2^2) ∨ is_divisible (N+1) (2^2) ∨ is_divisible (N+2) (2^2) ∨ is_divisible (N+3) (2^2)) ∧
  (is_divisible N (3^2) ∨ is_divisible (N+1) (3^2) ∨ is_divisible (N+2) (3^2) ∨ is_divisible (N+3) (3^2)) ∧
  (is_divisible N (5^2) ∨ is_divisible (N+1) (5^2) ∨ is_divisible (N+2) (5^2) ∨ is_divisible (N+3) (5^2)) ∧
  (is_divisible N (11^2) ∨ is_divisible (N+1) (11^2) ∨ is_divisible (N+2) (11^2) ∨ is_divisible (N+3) (11^2)) ∧
  (∀ M : ℕ, M < N → 
    ¬((is_divisible M (2^2) ∨ is_divisible (M+1) (2^2) ∨ is_divisible (M+2) (2^2) ∨ is_divisible (M+3) (2^2)) ∧
      (is_divisible M (3^2) ∨ is_divisible (M+1) (3^2) ∨ is_divisible (M+2) (3^2) ∨ is_divisible (M+3) (3^2)) ∧
      (is_divisible M (5^2) ∨ is_divisible (M+1) (5^2) ∨ is_divisible (M+2) (5^2) ∨ is_divisible (M+3) (5^2)) ∧
      (is_divisible M (11^2) ∨ is_divisible (M+1) (11^2) ∨ is_divisible (M+2) (11^2) ∨ is_divisible (M+3) (11^2)))) ∧
  N = 484 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_divisibility_l1583_158375


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1583_158355

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.6666666666666666)) :
  x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1583_158355


namespace NUMINAMATH_CALUDE_smallest_divisible_page_number_l1583_158354

theorem smallest_divisible_page_number : 
  (∀ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧ 17 ∣ n → n ≥ 68068) ∧ 
  (4 ∣ 68068) ∧ (13 ∣ 68068) ∧ (7 ∣ 68068) ∧ (11 ∣ 68068) ∧ (17 ∣ 68068) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_page_number_l1583_158354


namespace NUMINAMATH_CALUDE_volunteer_count_l1583_158320

/-- Represents the number of volunteers selected from each school -/
structure Volunteers where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- The ratio of students in schools A, B, and C -/
def schoolRatio : Fin 3 → ℕ
  | 0 => 2  -- School A
  | 1 => 3  -- School B
  | 2 => 5  -- School C

/-- The total ratio sum -/
def totalRatio : ℕ := (schoolRatio 0) + (schoolRatio 1) + (schoolRatio 2)

/-- Stratified sampling condition -/
def isStratifiedSample (v : Volunteers) : Prop :=
  (v.schoolA * schoolRatio 1 = v.schoolB * schoolRatio 0) ∧
  (v.schoolA * schoolRatio 2 = v.schoolC * schoolRatio 0)

/-- The main theorem -/
theorem volunteer_count (v : Volunteers) 
  (h_stratified : isStratifiedSample v) 
  (h_schoolA : v.schoolA = 6) : 
  v.schoolA + v.schoolB + v.schoolC = 30 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_count_l1583_158320


namespace NUMINAMATH_CALUDE_parabola_c_value_l1583_158371

/-- A parabola with equation y = x^2 + bx + c passes through points (2,3) and (5,6) -/
def parabola_through_points (b c : ℝ) : Prop :=
  3 = 2^2 + 2*b + c ∧ 6 = 5^2 + 5*b + c

/-- The theorem stating that c = -13 for the given parabola -/
theorem parabola_c_value : ∃ b : ℝ, parabola_through_points b (-13) := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1583_158371


namespace NUMINAMATH_CALUDE_base_product_sum_theorem_l1583_158328

/-- Represents a number in a given base --/
structure BaseNumber (base : ℕ) where
  value : ℕ

/-- Converts a BaseNumber to its decimal representation --/
def toDecimal {base : ℕ} (n : BaseNumber base) : ℕ := sorry

/-- Converts a decimal number to a BaseNumber --/
def fromDecimal (base : ℕ) (n : ℕ) : BaseNumber base := sorry

/-- Multiplies two BaseNumbers --/
def mult {base : ℕ} (a b : BaseNumber base) : BaseNumber base := sorry

/-- Adds two BaseNumbers --/
def add {base : ℕ} (a b : BaseNumber base) : BaseNumber base := sorry

theorem base_product_sum_theorem :
  ∀ c : ℕ,
    c > 1 →
    let thirteen := fromDecimal c 13
    let seventeen := fromDecimal c 17
    let nineteen := fromDecimal c 19
    let product := mult thirteen (mult seventeen nineteen)
    let sum := add thirteen (add seventeen nineteen)
    toDecimal product = toDecimal (fromDecimal c 4375) →
    toDecimal sum = toDecimal (fromDecimal 8 53) := by
  sorry

end NUMINAMATH_CALUDE_base_product_sum_theorem_l1583_158328


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1583_158347

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x + y > 1 ∧ x^2 + y^2 ≤ 1) ∧ 
  (∃ u v : ℝ, u^2 + v^2 > 1 ∧ u + v ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1583_158347


namespace NUMINAMATH_CALUDE_f_monotone_implies_a_range_l1583_158398

/-- A piecewise function f depending on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*(a-1)*x else (8-a)*x + 4

/-- f is monotonically increasing on ℝ -/
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem stating that if f is monotonically increasing, then 2 ≤ a ≤ 5 -/
theorem f_monotone_implies_a_range (a : ℝ) :
  monotone_increasing (f a) → 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

#check f_monotone_implies_a_range

end NUMINAMATH_CALUDE_f_monotone_implies_a_range_l1583_158398


namespace NUMINAMATH_CALUDE_prob_at_least_two_equals_result_l1583_158389

-- Define the probabilities for each person hitting the target
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.6

-- Define the probability of at least two people hitting the target
def prob_at_least_two : ℝ :=
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) -
  (prob_A * (1 - prob_B) * (1 - prob_C) +
   (1 - prob_A) * prob_B * (1 - prob_C) +
   (1 - prob_A) * (1 - prob_B) * prob_C)

-- Theorem statement
theorem prob_at_least_two_equals_result :
  prob_at_least_two = 0.832 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_equals_result_l1583_158389


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l1583_158340

theorem opposite_of_negative_three : -((-3) : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l1583_158340


namespace NUMINAMATH_CALUDE_sin_equal_implies_isosceles_exists_isosceles_with_unequal_sines_l1583_158307

/-- A triangle ABC is isosceles if at least two of its sides are equal. -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = b ∨ b = c ∨ a = c

/-- The sine of an angle in a triangle. -/
noncomputable def sinAngle (A B C : ℝ × ℝ) (vertex : ℝ × ℝ) : ℝ :=
  sorry -- Definition of sine for an angle in a triangle

theorem sin_equal_implies_isosceles (A B C : ℝ × ℝ) :
  sinAngle A B C A = sinAngle A B C B → IsIsosceles A B C :=
sorry

theorem exists_isosceles_with_unequal_sines :
  ∃ (A B C : ℝ × ℝ), IsIsosceles A B C ∧ sinAngle A B C A ≠ sinAngle A B C B :=
sorry

end NUMINAMATH_CALUDE_sin_equal_implies_isosceles_exists_isosceles_with_unequal_sines_l1583_158307


namespace NUMINAMATH_CALUDE_expected_points_is_seventeen_thirds_l1583_158368

/-- Represents the outcomes of the biased die -/
inductive Outcome
| Odd
| EvenNotSix
| Six

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Odd => 1/2
  | Outcome.EvenNotSix => 1/3
  | Outcome.Six => 1/6

/-- The points gained for each outcome -/
def points (o : Outcome) : ℚ :=
  match o with
  | Outcome.Odd => 9/2  -- Average of 1, 3, and 5
  | Outcome.EvenNotSix => 3  -- Average of 2 and 4
  | Outcome.Six => -5

/-- The expected value of points gained -/
def expected_value : ℚ :=
  (probability Outcome.Odd * points Outcome.Odd) +
  (probability Outcome.EvenNotSix * points Outcome.EvenNotSix) +
  (probability Outcome.Six * points Outcome.Six)

theorem expected_points_is_seventeen_thirds :
  expected_value = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_points_is_seventeen_thirds_l1583_158368


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1583_158394

theorem imaginary_part_of_complex_expression :
  let z : ℂ := 1 - I
  (z^2 + 2/z).im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1583_158394


namespace NUMINAMATH_CALUDE_convention_handshakes_specific_l1583_158332

/-- The number of handshakes in a convention with specified conditions -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

theorem convention_handshakes_specific : convention_handshakes 5 5 = 250 := by
  sorry

#eval convention_handshakes 5 5

end NUMINAMATH_CALUDE_convention_handshakes_specific_l1583_158332


namespace NUMINAMATH_CALUDE_acute_triangle_exists_l1583_158343

/-- Given 5 real numbers representing lengths of line segments,
    if any three of these numbers can form a triangle,
    then there exists a combination of three numbers that forms a triangle with all acute angles. -/
theorem acute_triangle_exists (a b c d e : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_triangle : ∀ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) →
                               (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) →
                               (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) →
                               x ≠ y ∧ y ≠ z ∧ x ≠ z →
                               x + y > z ∧ y + z > x ∧ x + z > y) :
  ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                 (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                 (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ x^2 + z^2 > y^2 :=
by sorry


end NUMINAMATH_CALUDE_acute_triangle_exists_l1583_158343


namespace NUMINAMATH_CALUDE_least_total_cost_equal_quantity_l1583_158333

def strawberry_pack_size : ℕ := 6
def strawberry_pack_price : ℕ := 2
def blueberry_pack_size : ℕ := 5
def blueberry_pack_price : ℕ := 3
def cherry_pack_size : ℕ := 8
def cherry_pack_price : ℕ := 4

def least_common_multiple (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

def total_cost (lcm : ℕ) : ℕ :=
  (lcm / strawberry_pack_size) * strawberry_pack_price +
  (lcm / blueberry_pack_size) * blueberry_pack_price +
  (lcm / cherry_pack_size) * cherry_pack_price

theorem least_total_cost_equal_quantity :
  total_cost (least_common_multiple strawberry_pack_size blueberry_pack_size cherry_pack_size) = 172 := by
  sorry

end NUMINAMATH_CALUDE_least_total_cost_equal_quantity_l1583_158333


namespace NUMINAMATH_CALUDE_sugar_salt_diff_is_one_l1583_158353

/-- A baking recipe with specified ingredient amounts -/
structure Recipe where
  flour : ℕ
  sugar : ℕ
  salt : ℕ

/-- The difference in cups between sugar and salt in a recipe -/
def sugar_salt_difference (r : Recipe) : ℤ :=
  r.sugar - r.salt

/-- Theorem: The difference between sugar and salt in the given recipe is 1 cup -/
theorem sugar_salt_diff_is_one (r : Recipe) (h : r.flour = 6 ∧ r.sugar = 8 ∧ r.salt = 7) : 
  sugar_salt_difference r = 1 := by
  sorry

#eval sugar_salt_difference {flour := 6, sugar := 8, salt := 7}

end NUMINAMATH_CALUDE_sugar_salt_diff_is_one_l1583_158353


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1583_158363

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1583_158363


namespace NUMINAMATH_CALUDE_f_2017_value_l1583_158390

theorem f_2017_value (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x * (deriv f 0) - 1) :
  f 2017 = 2016 * 2018 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_value_l1583_158390


namespace NUMINAMATH_CALUDE_three_digit_prime_with_special_property_l1583_158370

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def last_digit_is_sum_of_first_two (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones = hundreds + tens

theorem three_digit_prime_with_special_property (p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_three_digit : is_three_digit p)
  (h_different : all_digits_different p)
  (h_sum : last_digit_is_sum_of_first_two p) :
  p % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_prime_with_special_property_l1583_158370


namespace NUMINAMATH_CALUDE_base_sum_equals_55_base_7_l1583_158325

/-- Represents a number in a given base --/
def BaseNumber (base : ℕ) := ℕ

/-- Converts a base number to its decimal representation --/
def to_decimal (base : ℕ) (n : BaseNumber base) : ℕ := sorry

/-- Converts a decimal number to its representation in a given base --/
def from_decimal (base : ℕ) (n : ℕ) : BaseNumber base := sorry

/-- Multiplies two numbers in a given base --/
def base_mul (base : ℕ) (a b : BaseNumber base) : BaseNumber base := sorry

/-- Adds two numbers in a given base --/
def base_add (base : ℕ) (a b : BaseNumber base) : BaseNumber base := sorry

theorem base_sum_equals_55_base_7 (c : ℕ) 
  (h : base_mul c (base_mul c (from_decimal c 14) (from_decimal c 18)) (from_decimal c 17) = from_decimal c 4185) :
  base_add c (base_add c (from_decimal c 14) (from_decimal c 18)) (from_decimal c 17) = from_decimal 7 55 := 
sorry

end NUMINAMATH_CALUDE_base_sum_equals_55_base_7_l1583_158325


namespace NUMINAMATH_CALUDE_cosine_derivative_at_pi_over_two_l1583_158385

theorem cosine_derivative_at_pi_over_two :
  deriv (fun x => Real.cos x) (π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_derivative_at_pi_over_two_l1583_158385


namespace NUMINAMATH_CALUDE_expression_evaluation_l1583_158362

theorem expression_evaluation : 
  4 * Real.sin (60 * π / 180) - abs (-2) - Real.sqrt 12 + (-1) ^ 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1583_158362


namespace NUMINAMATH_CALUDE_euro_calculation_l1583_158384

-- Define the € operation
def euro (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem euro_calculation : euro 7 (euro 4 5) = 560 := by
  sorry

end NUMINAMATH_CALUDE_euro_calculation_l1583_158384


namespace NUMINAMATH_CALUDE_regular_polygon_side_length_l1583_158338

theorem regular_polygon_side_length 
  (n : ℕ) 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 48) 
  (h₂ : a₂ = 55) 
  (h₃ : n > 2) 
  (h₄ : (n * a₃^2) / (4 * Real.tan (π / n)) = 
        (n * a₁^2) / (4 * Real.tan (π / n)) + 
        (n * a₂^2) / (4 * Real.tan (π / n))) : 
  a₃ = 73 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_side_length_l1583_158338


namespace NUMINAMATH_CALUDE_yogurt_production_cost_l1583_158313

/-- The cost of producing three batches of yogurt given the following conditions:
  - Milk costs $1.5 per liter
  - Fruit costs $2 per kilogram
  - One batch of yogurt requires 10 liters of milk and 3 kilograms of fruit
-/
theorem yogurt_production_cost :
  let milk_cost_per_liter : ℚ := 3/2
  let fruit_cost_per_kg : ℚ := 2
  let milk_per_batch : ℚ := 10
  let fruit_per_batch : ℚ := 3
  let num_batches : ℕ := 3
  (milk_cost_per_liter * milk_per_batch + fruit_cost_per_kg * fruit_per_batch) * num_batches = 63 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_production_cost_l1583_158313


namespace NUMINAMATH_CALUDE_equation_solution_l1583_158388

theorem equation_solution : 
  ∃ x : ℝ, 
    (2.5 * ((3.6 * x * 2.50) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002) ∧ 
    (abs (x - 0.48) < 0.00000000000001) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1583_158388


namespace NUMINAMATH_CALUDE_inequality_condition_l1583_158383

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ a ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1583_158383


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1583_158378

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 15 * S →
  (S - C) / C * 100 = 233.33 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1583_158378


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1583_158317

theorem unknown_number_proof (x : ℝ) : x - (1002 / 200.4) = 3029 → x = 3034 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1583_158317


namespace NUMINAMATH_CALUDE_completing_square_sum_l1583_158382

theorem completing_square_sum (d e f : ℤ) : 
  d > 0 ∧ 
  (∀ x : ℝ, 25 * x^2 + 30 * x - 24 = 0 ↔ (d * x + e)^2 = f) → 
  d + e + f = 41 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l1583_158382


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l1583_158308

/-- Systematic sampling function that returns the number drawn from the nth group -/
def systematicSample (firstNumber : ℕ) (groupNumber : ℕ) (interval : ℕ) : ℕ :=
  firstNumber + interval * (groupNumber - 1)

theorem systematic_sampling_first_number :
  ∀ (totalStudents : ℕ) (numGroups : ℕ) (firstNumber : ℕ),
    totalStudents = 160 →
    numGroups = 20 →
    systematicSample firstNumber 15 8 = 116 →
    firstNumber = 4 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l1583_158308


namespace NUMINAMATH_CALUDE_ticket_revenue_calculation_l1583_158356

/-- Calculates the total revenue from ticket sales given the specified conditions -/
theorem ticket_revenue_calculation (total_tickets : ℕ) (student_price nonstudent_price : ℚ)
  (student_tickets : ℕ) (h1 : total_tickets = 821) (h2 : student_price = 2)
  (h3 : nonstudent_price = 3) (h4 : student_tickets = 530) :
  (student_tickets : ℚ) * student_price +
  ((total_tickets - student_tickets) : ℚ) * nonstudent_price = 1933 := by
  sorry

end NUMINAMATH_CALUDE_ticket_revenue_calculation_l1583_158356


namespace NUMINAMATH_CALUDE_fixed_fee_is_7_42_l1583_158346

/-- Represents the monthly bill structure for an online service provider -/
structure Bill where
  fixed_fee : ℝ
  connect_time_charge : ℝ
  data_usage_charge_per_gb : ℝ

/-- The December bill without data usage -/
def december_bill (b : Bill) : ℝ :=
  b.fixed_fee + b.connect_time_charge

/-- The January bill with 3 GB data usage -/
def january_bill (b : Bill) : ℝ :=
  b.fixed_fee + b.connect_time_charge + 3 * b.data_usage_charge_per_gb

/-- Theorem stating that the fixed monthly fee is $7.42 -/
theorem fixed_fee_is_7_42 (b : Bill) : b.fixed_fee = 7.42 :=
  by
  have h1 : december_bill b = 18.50 := by sorry
  have h2 : january_bill b = 23.45 := by sorry
  have h3 : january_bill b - december_bill b = 3 * b.data_usage_charge_per_gb := by sorry
  sorry

end NUMINAMATH_CALUDE_fixed_fee_is_7_42_l1583_158346


namespace NUMINAMATH_CALUDE_angle_sum_from_tangent_roots_l1583_158359

theorem angle_sum_from_tangent_roots (α β : Real) :
  (∃ x y : Real, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ 
                 y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
                 x = Real.tan α ∧ 
                 y = Real.tan β) →
  -π/2 < α ∧ α < π/2 →
  -π/2 < β ∧ β < π/2 →
  α + β = -2*π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_from_tangent_roots_l1583_158359


namespace NUMINAMATH_CALUDE_conjunction_is_false_l1583_158397

theorem conjunction_is_false :
  let p := ∀ x : ℝ, x < 1 → x < 2
  let q := ∃ x : ℝ, x^2 + 1 = 0
  ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_conjunction_is_false_l1583_158397


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l1583_158364

def scores : List ℕ := [86, 73, 55, 98, 76, 93, 88, 72, 77, 62, 81, 79, 68, 82, 91]

def is_grade_B (score : ℕ) : Bool :=
  87 ≤ score ∧ score ≤ 93

def count_grade_B (scores : List ℕ) : ℕ :=
  (scores.filter is_grade_B).length

theorem percentage_of_B_grades :
  (count_grade_B scores : ℚ) / (scores.length : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l1583_158364


namespace NUMINAMATH_CALUDE_inequality_problem_l1583_158303

theorem inequality_problem (x y : ℝ) 
  (h1 : 2 * x - 3 * y > 2 * x) 
  (h2 : 2 * x + 3 * y < 3 * y) : 
  x < 0 ∧ y < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l1583_158303


namespace NUMINAMATH_CALUDE_polynomial_properties_l1583_158300

/-- A polynomial of the form f(x) = ax^5 + bx^3 + 4x + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem polynomial_properties :
  ∀ (a b c : ℝ),
    (f a b c 0 = 6 → c = 6) ∧
    (f a b c 0 = -2 ∧ f a b c 1 = 5 → f a b c (-1) = -9) ∧
    (f a b c 5 + f a b c (-5) = 6 ∧ f a b c 2 = 8 → f a b c (-2) = -2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1583_158300


namespace NUMINAMATH_CALUDE_new_person_weight_bus_weight_problem_l1583_158399

theorem new_person_weight (initial_count : ℕ) (initial_average : ℝ) (weight_decrease : ℝ) : ℝ :=
  let total_weight := initial_count * initial_average
  let new_count := initial_count + 1
  let new_average := initial_average - weight_decrease
  let new_total_weight := new_count * new_average
  new_total_weight - total_weight

theorem bus_weight_problem :
  new_person_weight 30 102 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_bus_weight_problem_l1583_158399


namespace NUMINAMATH_CALUDE_distance_is_7920_meters_l1583_158329

/-- The distance traveled by a man driving at constant speed from the site of a blast -/
def distance_traveled (speed_of_sound : ℝ) (time_between_blasts : ℝ) (time_heard_second_blast : ℝ) : ℝ :=
  speed_of_sound * (time_heard_second_blast - time_between_blasts)

/-- Theorem stating that the distance traveled is 7920 meters -/
theorem distance_is_7920_meters :
  let speed_of_sound : ℝ := 330
  let time_between_blasts : ℝ := 30 * 60  -- 30 minutes in seconds
  let time_heard_second_blast : ℝ := 30 * 60 + 24  -- 30 minutes and 24 seconds in seconds
  distance_traveled speed_of_sound time_between_blasts time_heard_second_blast = 7920 := by
  sorry


end NUMINAMATH_CALUDE_distance_is_7920_meters_l1583_158329


namespace NUMINAMATH_CALUDE_function_properties_l1583_158341

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - 5 * x^2 - b * x

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 - 10 * x - b

theorem function_properties (a b : ℝ) :
  (f_derivative a b 3 = 0) →  -- x = 3 is an extreme point
  (f a b 1 = -1) →           -- f(1) = -1
  (a = 1 ∧ b = -3) ∧         -- Part 1: a = 1 and b = -3
  (∀ x ∈ Set.Icc 2 4, f 1 (-3) x ≥ -9) ∧  -- Part 2: Minimum value on [2, 4] is -9
  (∀ x ∈ Set.Icc 2 4, f 1 (-3) x ≤ 0) ∧   -- Part 3: Maximum value on [2, 4] is 0
  (f 1 (-3) 3 = -9) ∧        -- Minimum occurs at x = 3
  (f 1 (-3) 4 = 0)           -- Maximum occurs at x = 4
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l1583_158341


namespace NUMINAMATH_CALUDE_correct_num_bedrooms_l1583_158323

/-- The number of bedrooms to clean -/
def num_bedrooms : ℕ := sorry

/-- Time in minutes to clean one bedroom -/
def bedroom_time : ℕ := 20

/-- Time in minutes to clean the living room -/
def living_room_time : ℕ := num_bedrooms * bedroom_time

/-- Time in minutes to clean one bathroom -/
def bathroom_time : ℕ := 2 * living_room_time

/-- Time in minutes to clean the house (bedrooms, living room, and bathrooms) -/
def house_time : ℕ := num_bedrooms * bedroom_time + living_room_time + 2 * bathroom_time

/-- Time in minutes to clean outside -/
def outside_time : ℕ := 2 * house_time

/-- Total time in minutes for all three siblings to work -/
def total_work_time : ℕ := 3 * 4 * 60

theorem correct_num_bedrooms : num_bedrooms = 3 := by sorry

end NUMINAMATH_CALUDE_correct_num_bedrooms_l1583_158323


namespace NUMINAMATH_CALUDE_common_root_condition_rational_roots_if_common_root_l1583_158395

structure QuadraticEquation (α : Type) [Field α] where
  p : α
  q : α

def hasCommonRoot {α : Type} [Field α] (eq1 eq2 : QuadraticEquation α) : Prop :=
  ∃ x : α, x^2 + eq1.p * x + eq1.q = 0 ∧ x^2 + eq2.p * x + eq2.q = 0

theorem common_root_condition {α : Type} [Field α] (eq1 eq2 : QuadraticEquation α) :
  hasCommonRoot eq1 eq2 ↔ (eq1.p - eq2.p) * (eq1.p * eq2.q - eq2.p * eq1.q) + (eq1.q - eq2.q)^2 = 0 :=
sorry

theorem rational_roots_if_common_root (eq1 eq2 : QuadraticEquation ℚ) 
  (h1 : hasCommonRoot eq1 eq2) (h2 : eq1 ≠ eq2) :
  ∃ (x y : ℚ), (x^2 + eq1.p * x + eq1.q = 0 ∧ y^2 + eq1.p * y + eq1.q = 0) ∧
                (x^2 + eq2.p * x + eq2.q = 0 ∧ y^2 + eq2.p * y + eq2.q = 0) :=
sorry

end NUMINAMATH_CALUDE_common_root_condition_rational_roots_if_common_root_l1583_158395


namespace NUMINAMATH_CALUDE_number_difference_l1583_158319

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1583_158319


namespace NUMINAMATH_CALUDE_hospital_staff_count_l1583_158312

theorem hospital_staff_count (total : ℕ) (doc_ratio nurse_ratio : ℕ) (nurse_count : ℕ) : 
  total = 280 → 
  doc_ratio = 5 →
  nurse_ratio = 9 →
  doc_ratio + nurse_ratio = (total / nurse_count) →
  nurse_count = 180 := by
sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l1583_158312


namespace NUMINAMATH_CALUDE_purchase_cost_l1583_158396

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 4

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 3/2

/-- The discount applied when purchasing at least 10 sandwiches -/
def bulk_discount : ℚ := 5

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 10

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The total cost of the purchase -/
def total_cost : ℚ := 
  (num_sandwiches * sandwich_cost - bulk_discount) + (num_sodas * soda_cost)

theorem purchase_cost : total_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l1583_158396


namespace NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_one_third_l1583_158376

/-- Two vectors in R² -/
def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b : Fin 2 → ℝ := ![3, 1]

/-- Dot product of two vectors in R² -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Theorem: Vectors a and b are perpendicular if and only if x = -1/3 -/
theorem perpendicular_iff_x_eq_neg_one_third (x : ℝ) : 
  dot_product (a x) b = 0 ↔ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_x_eq_neg_one_third_l1583_158376


namespace NUMINAMATH_CALUDE_sachin_age_l1583_158350

theorem sachin_age (sachin rahul : ℝ) 
  (h1 : sachin = rahul - 9)
  (h2 : sachin / rahul = 7 / 9) : 
  sachin = 31.5 := by
sorry

end NUMINAMATH_CALUDE_sachin_age_l1583_158350


namespace NUMINAMATH_CALUDE_triangle_centroid_distances_l1583_158348

/-- Given a triangle DEF with centroid G, prove that if the sum of squared distances
    from G to the vertices is 72, then the sum of squared side lengths is 216. -/
theorem triangle_centroid_distances (D E F G : ℝ × ℝ) : 
  G = ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3) →  -- G is the centroid
  (G.1 - D.1)^2 + (G.2 - D.2)^2 +    -- GD^2
  (G.1 - E.1)^2 + (G.2 - E.2)^2 +    -- GE^2
  (G.1 - F.1)^2 + (G.2 - F.2)^2 = 72 →  -- GF^2
  (D.1 - E.1)^2 + (D.2 - E.2)^2 +    -- DE^2
  (D.1 - F.1)^2 + (D.2 - F.2)^2 +    -- DF^2
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 216  -- EF^2
:= by sorry

end NUMINAMATH_CALUDE_triangle_centroid_distances_l1583_158348


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1583_158331

theorem simplify_fraction_product : 
  (4 : ℚ) * (18 / 5) * (35 / -63) * (8 / 14) = -32 / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1583_158331


namespace NUMINAMATH_CALUDE_remainder_cube_l1583_158326

theorem remainder_cube (n : ℤ) : n % 13 = 5 → n^3 % 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_cube_l1583_158326


namespace NUMINAMATH_CALUDE_line_slope_135_degrees_l1583_158393

/-- The slope of a line in degrees -/
def Slope : Type := ℝ

/-- The equation of a line in the form mx + y + c = 0 -/
structure Line where
  m : ℝ
  c : ℝ

/-- The tangent of an angle in degrees -/
noncomputable def tan_degrees (θ : ℝ) : ℝ := sorry

theorem line_slope_135_degrees (l : Line) (h : l.c = 2) : 
  (tan_degrees 135 = -l.m) → l.m = 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_135_degrees_l1583_158393


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1583_158372

theorem tangent_line_to_logarithmic_curve (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * x = 1 + Real.log x ∧ 
    ∀ y : ℝ, y > 0 → a * y ≤ 1 + Real.log y) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1583_158372


namespace NUMINAMATH_CALUDE_photograph_perimeter_l1583_158349

theorem photograph_perimeter (w h m : ℝ) 
  (area_with_2inch_border : (w + 4) * (h + 4) = m)
  (area_with_4inch_border : (w + 8) * (h + 8) = m + 94)
  : 2 * (w + h) = 23 := by
  sorry

end NUMINAMATH_CALUDE_photograph_perimeter_l1583_158349


namespace NUMINAMATH_CALUDE_min_weighings_to_identify_defective_l1583_158322

/-- Represents a piece that can be either standard or defective -/
inductive Piece
| Standard
| Defective

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal
| LeftHeavier
| RightHeavier

/-- A function that simulates a weighing on a balance scale -/
def weigh (left right : List Piece) : WeighingResult := sorry

/-- The set of all possible pieces -/
def allPieces : Finset Piece := sorry

/-- The number of pieces -/
def numPieces : Nat := 5

/-- The number of standard pieces -/
def numStandard : Nat := 4

/-- The number of defective pieces -/
def numDefective : Nat := 1

/-- A strategy for identifying the defective piece -/
def identifyDefective : Nat → Option Piece := sorry

theorem min_weighings_to_identify_defective :
  ∃ (strategy : Nat → Option Piece),
    (∀ defective : Piece, 
      defective ∈ allPieces → 
      ∃ n : Nat, n ≤ 3 ∧ strategy n = some defective) ∧
    (∀ m : Nat, 
      (∀ defective : Piece, 
        defective ∈ allPieces → 
        (∃ n : Nat, n ≤ m ∧ strategy n = some defective)) → 
      m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_min_weighings_to_identify_defective_l1583_158322


namespace NUMINAMATH_CALUDE_lending_interest_rate_l1583_158387

/-- Proves that the lending interest rate is 6% given the specified conditions --/
theorem lending_interest_rate (borrowed_amount : ℕ) (borrowing_period : ℕ) 
  (borrowing_rate : ℚ) (gain_per_year : ℕ) (lending_rate : ℚ) : 
  borrowed_amount = 6000 →
  borrowing_period = 2 →
  borrowing_rate = 4 / 100 →
  gain_per_year = 120 →
  (borrowed_amount * borrowing_rate * borrowing_period + 
   borrowing_period * gain_per_year) / (borrowed_amount * borrowing_period) * 100 = lending_rate →
  lending_rate = 6 / 100 := by
sorry


end NUMINAMATH_CALUDE_lending_interest_rate_l1583_158387


namespace NUMINAMATH_CALUDE_system_solution_l1583_158386

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, 4*x - 3*y = k ∧ 2*x + 3*y = 5 ∧ x = y) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1583_158386


namespace NUMINAMATH_CALUDE_speed_ratio_of_perpendicular_paths_l1583_158374

/-- The ratio of speeds of two objects moving along perpendicular paths -/
theorem speed_ratio_of_perpendicular_paths
  (vA vB : ℝ) -- Speeds of objects A and B
  (h1 : vA > 0 ∧ vB > 0) -- Both speeds are positive
  (h2 : ∃ t1 : ℝ, t1 > 0 ∧ t1 * vA = |700 - t1 * vB|) -- Equidistant at time t1
  (h3 : ∃ t2 : ℝ, t2 > t1 ∧ t2 * vA = |700 - t2 * vB|) -- Equidistant at time t2 > t1
  : vA / vB = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_speed_ratio_of_perpendicular_paths_l1583_158374


namespace NUMINAMATH_CALUDE_sara_quarters_theorem_l1583_158318

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := 783

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad : ℕ := 271

/-- The total number of quarters Sara has now -/
def total_quarters : ℕ := 1054

/-- Theorem stating that the initial number of quarters plus the quarters from dad equals the total quarters -/
theorem sara_quarters_theorem : initial_quarters + quarters_from_dad = total_quarters := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_theorem_l1583_158318


namespace NUMINAMATH_CALUDE_dorokhov_vacation_cost_l1583_158324

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  price_young : ℕ
  price_old : ℕ
  age_threshold : ℕ
  discount_rate : ℚ
  is_commission : Bool

/-- Calculates the total cost for a family's vacation package -/
def calculate_total_cost (agency : TravelAgency) (num_adults num_children : ℕ) (child_age : ℕ) : ℚ :=
  sorry

/-- The Dorokhov family's vacation cost theorem -/
theorem dorokhov_vacation_cost :
  let globus : TravelAgency := {
    name := "Globus",
    price_young := 11200,
    price_old := 25400,
    age_threshold := 5,
    discount_rate := -2/100,
    is_commission := false
  }
  let around_the_world : TravelAgency := {
    name := "Around the World",
    price_young := 11400,
    price_old := 23500,
    age_threshold := 6,
    discount_rate := 1/100,
    is_commission := true
  }
  let num_adults : ℕ := 2
  let num_children : ℕ := 1
  let child_age : ℕ := 5
  
  min (calculate_total_cost globus num_adults num_children child_age)
      (calculate_total_cost around_the_world num_adults num_children child_age) = 58984 := by
  sorry

end NUMINAMATH_CALUDE_dorokhov_vacation_cost_l1583_158324


namespace NUMINAMATH_CALUDE_supervisors_per_bus_l1583_158367

theorem supervisors_per_bus (total_buses : ℕ) (total_supervisors : ℕ) 
  (h1 : total_buses = 7) 
  (h2 : total_supervisors = 21) : 
  total_supervisors / total_buses = 3 := by
  sorry

end NUMINAMATH_CALUDE_supervisors_per_bus_l1583_158367


namespace NUMINAMATH_CALUDE_carl_reaches_goal_in_53_days_l1583_158337

/-- Represents Carl's earnings and candy bar goal --/
structure CarlsEarnings where
  candy_bar_cost : ℚ
  weekly_trash_pay : ℚ
  biweekly_dog_pay : ℚ
  aunt_payment : ℚ
  candy_bar_goal : ℕ

/-- Calculates the number of days needed for Carl to reach his candy bar goal --/
def days_to_reach_goal (e : CarlsEarnings) : ℕ :=
  sorry

/-- Theorem stating that given Carl's specific earnings and goal, it takes 53 days to reach the goal --/
theorem carl_reaches_goal_in_53_days :
  let e : CarlsEarnings := {
    candy_bar_cost := 1/2,
    weekly_trash_pay := 3/4,
    biweekly_dog_pay := 5/4,
    aunt_payment := 5,
    candy_bar_goal := 30
  }
  days_to_reach_goal e = 53 := by
  sorry

end NUMINAMATH_CALUDE_carl_reaches_goal_in_53_days_l1583_158337


namespace NUMINAMATH_CALUDE_largest_seventh_term_coefficient_l1583_158327

/-- 
Given that in the expansion of (x + y)^n the coefficient of the seventh term is the largest,
this theorem states that n must be either 11, 12, or 13.
-/
theorem largest_seventh_term_coefficient (n : ℕ) : 
  (∀ k : ℕ, k ≠ 6 → (n.choose k) ≤ (n.choose 6)) → 
  n = 11 ∨ n = 12 ∨ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_seventh_term_coefficient_l1583_158327


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l1583_158335

theorem solution_implies_k_value (x k : ℚ) : 
  (x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l1583_158335


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l1583_158361

theorem smallest_proportional_part (total : ℕ) (parts : List ℕ) : 
  total = 360 → 
  parts = [5, 7, 4, 8] → 
  (parts.sum : ℚ) > 0 → 
  let proportional_parts := parts.map (λ p => (p : ℚ) * total / parts.sum)
  List.minimum proportional_parts = some 60 := by
sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l1583_158361


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l1583_158311

-- Define the binary number
def binary_number : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_number : Nat := 56

-- Theorem statement
theorem binary_to_octal_conversion :
  (binary_number.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0) = octal_number := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l1583_158311


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1583_158357

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The property that f(x^2) = f(f(x)) = (f(x))^2 for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x^2) = f (f x) ∧ f (x^2) = (f x)^2

theorem unique_quadratic_function :
  ∃! f : ℝ → ℝ, QuadraticFunction f ∧ SatisfiesCondition f ∧ ∀ x, f x = x^2 :=
by
  sorry

#check unique_quadratic_function

end NUMINAMATH_CALUDE_unique_quadratic_function_l1583_158357


namespace NUMINAMATH_CALUDE_range_of_m_l1583_158358

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 2 / x + 1 / y = 1 / 3) (h_ineq : ∀ m : ℝ, x + 2 * y > m^2 - 2 * m) :
  -4 < m ∧ m < 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1583_158358


namespace NUMINAMATH_CALUDE_lcm_problem_l1583_158344

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l1583_158344


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1583_158306

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^(m^2 - 2) - 3*x + 1 = a*x^2 + b*x + c) → 
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1583_158306


namespace NUMINAMATH_CALUDE_mabel_marbles_l1583_158369

/-- Given information about marbles of Amanda, Katrina, and Mabel -/
def marble_problem (amanda katrina mabel : ℕ) : Prop :=
  (amanda + 12 = 2 * katrina) ∧
  (mabel = 5 * katrina) ∧
  (mabel = amanda + 63)

/-- Theorem stating that under the given conditions, Mabel has 85 marbles -/
theorem mabel_marbles :
  ∀ amanda katrina mabel : ℕ,
  marble_problem amanda katrina mabel →
  mabel = 85 := by
  sorry

end NUMINAMATH_CALUDE_mabel_marbles_l1583_158369


namespace NUMINAMATH_CALUDE_min_score_jack_l1583_158301

-- Define the parameters of the normal distribution
def mean : ℝ := 60
def std_dev : ℝ := 10

-- Define the z-score for the 90th percentile (top 10%)
def z_score_90th_percentile : ℝ := 1.28

-- Define the function to calculate the score from z-score
def score_from_z (z : ℝ) : ℝ := z * std_dev + mean

-- Define the 90th percentile score
def score_90th_percentile : ℝ := score_from_z z_score_90th_percentile

-- Define the upper bound of 2 standard deviations above the mean
def two_std_dev_above_mean : ℝ := mean + 2 * std_dev

-- State the theorem
theorem min_score_jack : 
  ∀ (score : ℕ), 
    (score ≥ ⌈score_90th_percentile⌉) ∧ 
    (↑score ≤ two_std_dev_above_mean) → 
    score ≥ 73 :=
sorry

end NUMINAMATH_CALUDE_min_score_jack_l1583_158301


namespace NUMINAMATH_CALUDE_max_cab_value_l1583_158352

/-- Represents a two-digit number AB --/
def TwoDigitNumber (a b : Nat) : Prop :=
  10 ≤ 10 * a + b ∧ 10 * a + b < 100

/-- Represents a three-digit number CAB --/
def ThreeDigitNumber (c a b : Nat) : Prop :=
  100 ≤ 100 * c + 10 * a + b ∧ 100 * c + 10 * a + b < 1000

/-- The main theorem statement --/
theorem max_cab_value :
  ∀ a b c : Nat,
  a < 10 → b < 10 → c < 10 →
  TwoDigitNumber a b →
  ThreeDigitNumber c a b →
  (10 * a + b) * a = 100 * c + 10 * a + b →
  100 * c + 10 * a + b ≤ 895 :=
by sorry

end NUMINAMATH_CALUDE_max_cab_value_l1583_158352
