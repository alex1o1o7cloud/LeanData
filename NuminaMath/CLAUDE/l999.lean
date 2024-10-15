import Mathlib

namespace NUMINAMATH_CALUDE_additional_capacity_l999_99936

/-- Represents the number of cars used by the swimming club -/
def num_cars : Nat := 2

/-- Represents the number of vans used by the swimming club -/
def num_vans : Nat := 3

/-- Represents the number of people in each car -/
def people_per_car : Nat := 5

/-- Represents the number of people in each van -/
def people_per_van : Nat := 3

/-- Represents the maximum capacity of each car -/
def max_car_capacity : Nat := 6

/-- Represents the maximum capacity of each van -/
def max_van_capacity : Nat := 8

/-- Theorem stating the number of additional people that could have ridden with the swim team -/
theorem additional_capacity : 
  (num_cars * max_car_capacity + num_vans * max_van_capacity) - 
  (num_cars * people_per_car + num_vans * people_per_van) = 17 := by
  sorry

end NUMINAMATH_CALUDE_additional_capacity_l999_99936


namespace NUMINAMATH_CALUDE_circle_tangent_theorem_l999_99954

-- Define the circle
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*a*x + 2*y - 1 = 0}

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-5, a)

-- Define the condition for the tangent lines
def TangentCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0

theorem circle_tangent_theorem :
  ∀ a : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
    P a ∈ Circle a →
    (x₁, y₁) ∈ Circle a →
    (x₂, y₂) ∈ Circle a →
    TangentCondition x₁ y₁ x₂ y₂ →
    a = 3 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_theorem_l999_99954


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l999_99966

/-- 
Given a quadratic expression of the form 3x^2 + nx + 108, 
this theorem states that the largest value of n for which 
the expression can be factored as the product of two linear 
factors with integer coefficients is 325.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (A B : ℤ), 
    (3 * A = 1 ∧ B = 108) → 
    (∀ (x : ℤ), 3 * x^2 + n * x + 108 = (3 * x + A) * (x + B)) ∧
    (∀ (m : ℤ), (∃ (C D : ℤ), ∀ (x : ℤ), 
      3 * x^2 + m * x + 108 = (3 * x + C) * (x + D)) → 
      m ≤ n)) ∧
  (∀ (n : ℤ), (∀ (A B : ℤ), 
    (3 * A = 1 ∧ B = 108) → 
    (∀ (x : ℤ), 3 * x^2 + n * x + 108 = (3 * x + A) * (x + B)) ∧
    (∀ (m : ℤ), (∃ (C D : ℤ), ∀ (x : ℤ), 
      3 * x^2 + m * x + 108 = (3 * x + C) * (x + D)) → 
      m ≤ n)) → 
  n = 325) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l999_99966


namespace NUMINAMATH_CALUDE_white_tiger_number_count_l999_99915

/-- A function that returns true if a number is a multiple of 6 -/
def isMultipleOf6 (n : ℕ) : Bool :=
  n % 6 = 0

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A function that returns true if a number is a "White Tiger number" -/
def isWhiteTigerNumber (n : ℕ) : Bool :=
  isMultipleOf6 n ∧ sumOfDigits n = 6

/-- The count of "White Tiger numbers" up to 2022 -/
def whiteTigerNumberCount : ℕ :=
  (List.range 2023).filter isWhiteTigerNumber |>.length

theorem white_tiger_number_count : whiteTigerNumberCount = 30 := by
  sorry

end NUMINAMATH_CALUDE_white_tiger_number_count_l999_99915


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l999_99914

theorem triangle_angle_sum (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8) :
  let θ := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  (π - θ) * (180 / π) = 120 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l999_99914


namespace NUMINAMATH_CALUDE_initial_position_proof_l999_99946

def moves : List Int := [-5, 4, 2, -3, 1]
def final_position : Int := 6

theorem initial_position_proof :
  (moves.foldl (· + ·) final_position) = 7 := by sorry

end NUMINAMATH_CALUDE_initial_position_proof_l999_99946


namespace NUMINAMATH_CALUDE_one_fourth_of_12_8_l999_99911

theorem one_fourth_of_12_8 :
  let x : ℚ := 12.8 / 4
  x = 16 / 5 ∧ x = 3 + 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_one_fourth_of_12_8_l999_99911


namespace NUMINAMATH_CALUDE_train_length_l999_99905

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 25 → speed * time * (1000 / 3600) = 833.25 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l999_99905


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l999_99925

theorem student_average_greater_than_true_average 
  (a b c : ℝ) (h : a < b ∧ b < c) : (a + b + c) / 2 > (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l999_99925


namespace NUMINAMATH_CALUDE_base5_addition_puzzle_l999_99927

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : ℕ) : ℕ := sorry

/-- Represents a digit in base 5 --/
structure Digit5 where
  value : ℕ
  property : value < 5

theorem base5_addition_puzzle :
  ∀ (S H E : Digit5),
    S.value ≠ 0 ∧ H.value ≠ 0 ∧ E.value ≠ 0 →
    S.value ≠ H.value ∧ S.value ≠ E.value ∧ H.value ≠ E.value →
    (S.value * 25 + H.value * 5 + E.value) + (H.value * 5 + E.value) = 
    (S.value * 25 + E.value * 5 + S.value) →
    S.value = 4 ∧ H.value = 1 ∧ E.value = 2 ∧ 
    toBase5 (S.value + H.value + E.value) = 12 :=
by sorry

end NUMINAMATH_CALUDE_base5_addition_puzzle_l999_99927


namespace NUMINAMATH_CALUDE_edge_sum_is_144_l999_99976

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- The three dimensions of the solid
  a : ℝ
  b : ℝ
  c : ℝ
  -- Volume is 432 cm³
  volume_eq : a * b * c = 432
  -- Surface area is 432 cm²
  surface_area_eq : 2 * (a * b + b * c + a * c) = 432
  -- Dimensions are in geometric progression
  geometric_progression : b * b = a * c

/-- The sum of the lengths of all edges of the rectangular solid is 144 cm -/
theorem edge_sum_is_144 (solid : RectangularSolid) :
  4 * (solid.a + solid.b + solid.c) = 144 := by
  sorry

end NUMINAMATH_CALUDE_edge_sum_is_144_l999_99976


namespace NUMINAMATH_CALUDE_shirts_total_cost_l999_99991

/-- Calculates the total cost of shirts with given prices, quantities, discounts, and taxes -/
def totalCost (price1 price2 : ℝ) (quantity1 quantity2 : ℕ) (discount tax : ℝ) : ℝ :=
  quantity1 * (price1 * (1 - discount)) + quantity2 * (price2 * (1 + tax))

/-- Theorem stating that the total cost of the shirts is $82.50 -/
theorem shirts_total_cost :
  totalCost 15 20 3 2 0.1 0.05 = 82.5 := by
  sorry

#eval totalCost 15 20 3 2 0.1 0.05

end NUMINAMATH_CALUDE_shirts_total_cost_l999_99991


namespace NUMINAMATH_CALUDE_system_solution_l999_99941

theorem system_solution :
  ∃! (x y : ℚ), 2 * x - 3 * y = 1 ∧ (y + 1) / 4 + 1 = (x + 2) / 3 ∧ x = 3 ∧ y = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l999_99941


namespace NUMINAMATH_CALUDE_complex_equation_implication_l999_99938

theorem complex_equation_implication (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + b * i) * i = 1 + 2 * i →
  a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implication_l999_99938


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l999_99939

/-- 
Given an infinite geometric series with first term a and common ratio r,
if the sum of the original series is 81 times the sum of the series
that results when the first four terms are removed, then r = 1/3.
-/
theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) →
  r = 1/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l999_99939


namespace NUMINAMATH_CALUDE_intersection_M_N_l999_99958

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = Real.sin x}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l999_99958


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l999_99993

/-- Represents the odometer reading as a triple of natural numbers -/
structure OdometerReading where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : b ≥ 1
  h2 : a + b + c ≤ 9

/-- Represents Liam's car trip -/
structure CarTrip where
  speed : ℕ
  hours : ℕ
  initial : OdometerReading
  final : OdometerReading
  h1 : speed = 60
  h2 : final.a = initial.b
  h3 : final.b = initial.c
  h4 : final.c = initial.a
  h5 : 100 * final.b + 10 * final.c + final.a - (100 * initial.a + 10 * initial.b + initial.c) = speed * hours

theorem odometer_sum_squares (trip : CarTrip) : 
  trip.initial.a^2 + trip.initial.b^2 + trip.initial.c^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l999_99993


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l999_99930

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem stating that 132₄ × 21₄ ÷ 3₄ = 1122₄ --/
theorem base4_multiplication_division :
  let a := base4ToBase10 [2, 3, 1]  -- 132₄
  let b := base4ToBase10 [1, 2]     -- 21₄
  let c := base4ToBase10 [3]        -- 3₄
  let result := base10ToBase4 ((a * b) / c)
  result = [2, 2, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l999_99930


namespace NUMINAMATH_CALUDE_machine_speed_ratio_l999_99908

def machine_a_rate (parts_a : ℕ) (time_a : ℕ) : ℚ := parts_a / time_a
def machine_b_rate (parts_b : ℕ) (time_b : ℕ) : ℚ := parts_b / time_b

theorem machine_speed_ratio :
  let parts_a_100 : ℕ := 100
  let time_a_100 : ℕ := 40
  let parts_a_50 : ℕ := 50
  let time_a_50 : ℕ := 10
  let parts_b : ℕ := 100
  let time_b : ℕ := 40
  machine_a_rate parts_a_100 time_a_100 = machine_b_rate parts_b time_b →
  machine_a_rate parts_a_50 time_a_50 / machine_b_rate parts_b time_b = 2 := by
sorry

end NUMINAMATH_CALUDE_machine_speed_ratio_l999_99908


namespace NUMINAMATH_CALUDE_three_by_five_rectangle_triangles_l999_99948

/-- Represents a rectangle divided into a grid with diagonal lines. -/
structure GridRectangle where
  horizontal_divisions : Nat
  vertical_divisions : Nat

/-- Counts the number of triangles in a GridRectangle. -/
def count_triangles (rect : GridRectangle) : Nat :=
  sorry

/-- Theorem stating that a 3x5 GridRectangle contains 76 triangles. -/
theorem three_by_five_rectangle_triangles :
  count_triangles ⟨3, 5⟩ = 76 := by
  sorry

end NUMINAMATH_CALUDE_three_by_five_rectangle_triangles_l999_99948


namespace NUMINAMATH_CALUDE_melanie_total_dimes_l999_99924

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

theorem melanie_total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_total_dimes_l999_99924


namespace NUMINAMATH_CALUDE_short_trees_count_l999_99961

/-- The number of short trees in the park after planting -/
def short_trees_after_planting (initial_short_trees planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + planted_short_trees

/-- Theorem: The number of short trees after planting is 95 -/
theorem short_trees_count : short_trees_after_planting 31 64 = 95 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_count_l999_99961


namespace NUMINAMATH_CALUDE_count_flippy_divisible_by_18_l999_99969

def is_flippy (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ 
    n = a * 100000 + b * 10000 + a * 1000 + b * 100 + a * 10 + b

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

theorem count_flippy_divisible_by_18 :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_flippy n ∧ is_six_digit n ∧ n % 18 = 0) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_flippy_divisible_by_18_l999_99969


namespace NUMINAMATH_CALUDE_total_marbles_l999_99904

/-- The total number of marbles given the conditions of red, blue, and green marbles -/
theorem total_marbles (r : ℝ) (b : ℝ) (g : ℝ) : 
  r > 0 → 
  r = 1.5 * b → 
  g = 1.8 * r → 
  r + b + g = 3.467 * r := by
sorry


end NUMINAMATH_CALUDE_total_marbles_l999_99904


namespace NUMINAMATH_CALUDE_friend_gcd_l999_99975

theorem friend_gcd (a b : ℕ) (h : ∃ k : ℕ, a * b = k^2) :
  ∃ m : ℕ, a * Nat.gcd a b = m^2 := by
sorry

end NUMINAMATH_CALUDE_friend_gcd_l999_99975


namespace NUMINAMATH_CALUDE_halfway_between_one_fifth_and_one_third_l999_99963

theorem halfway_between_one_fifth_and_one_third :
  (1 / 5 : ℚ) / 2 + (1 / 3 : ℚ) / 2 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_fifth_and_one_third_l999_99963


namespace NUMINAMATH_CALUDE_product_constant_percentage_change_l999_99995

theorem product_constant_percentage_change (x1 y1 x2 y2 : ℝ) :
  x1 * y1 = x2 * y2 ∧ 
  y2 = y1 * (1 - 44.44444444444444 / 100) →
  x2 = x1 * (1 + 80 / 100) :=
by sorry

end NUMINAMATH_CALUDE_product_constant_percentage_change_l999_99995


namespace NUMINAMATH_CALUDE_monkey_peaches_l999_99912

theorem monkey_peaches (x : ℕ) : 
  (x / 2 - 12 + (x / 2 + 12) / 2 + 12 = x - 19) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_monkey_peaches_l999_99912


namespace NUMINAMATH_CALUDE_total_wait_days_l999_99902

/-- The number of days Mark waits for his first vaccine appointment -/
def first_appointment_wait : ℕ := 4

/-- The number of days Mark waits for his second vaccine appointment -/
def second_appointment_wait : ℕ := 20

/-- The number of weeks Mark waits for the vaccine to be fully effective -/
def full_effectiveness_wait_weeks : ℕ := 2

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: The total number of days Mark waits is 38 -/
theorem total_wait_days : 
  first_appointment_wait + second_appointment_wait + (full_effectiveness_wait_weeks * days_per_week) = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_wait_days_l999_99902


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l999_99935

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- State the theorem
theorem quadratic_function_properties (a : ℝ) :
  (∀ x : ℝ, f a x = f a (4 - x)) →
  (a = 4) ∧
  (Set.Icc 0 3).image (f a) = Set.Icc (-1) 3 ∧
  ∃ (g : ℝ → ℝ), (∀ x : ℝ, f a x = (x - 2)^2 + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l999_99935


namespace NUMINAMATH_CALUDE_set_B_equals_l999_99945

def A : Set Int := {-2, -1, 1, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem set_B_equals : B = {1, 4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_set_B_equals_l999_99945


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l999_99989

/-- The number of ways to distribute students among communities -/
def distribute_students (n_students : ℕ) (n_communities : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the correct number of arrangements -/
theorem student_distribution_theorem :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l999_99989


namespace NUMINAMATH_CALUDE_trig_expression_value_l999_99928

theorem trig_expression_value (α : Real) 
  (h1 : π/2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  2 / (Real.cos α - Real.sin α) = -10/7 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l999_99928


namespace NUMINAMATH_CALUDE_cubic_solution_sum_l999_99921

theorem cubic_solution_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a = 12 →
  b^3 - 6*b^2 + 11*b = 12 →
  c^3 - 6*c^2 + 11*c = 12 →
  a * b / c + b * c / a + c * a / b = -23 / 12 :=
by sorry

end NUMINAMATH_CALUDE_cubic_solution_sum_l999_99921


namespace NUMINAMATH_CALUDE_compute_expression_l999_99996

theorem compute_expression : 3 * 3^4 - 9^60 / 9^57 = -486 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l999_99996


namespace NUMINAMATH_CALUDE_joe_initial_cars_l999_99962

/-- The number of cars Joe will have after getting more cars -/
def total_cars : ℕ := 62

/-- The number of additional cars Joe will get -/
def additional_cars : ℕ := 12

/-- Theorem: Joe's initial number of cars is 50 -/
theorem joe_initial_cars : 
  total_cars - additional_cars = 50 := by
  sorry

end NUMINAMATH_CALUDE_joe_initial_cars_l999_99962


namespace NUMINAMATH_CALUDE_two_digit_divisible_number_exists_l999_99903

theorem two_digit_divisible_number_exists : ∃ n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  n % 8 = 0 ∧ n % 12 = 0 ∧ n % 18 = 0 ∧
  60 ≤ n ∧ n ≤ 79 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_divisible_number_exists_l999_99903


namespace NUMINAMATH_CALUDE_max_sum_of_complex_product_l999_99979

/-- The maximum sum of real and imaginary parts of the product of two specific complex functions of θ -/
theorem max_sum_of_complex_product :
  let z1 (θ : ℝ) := (8 + Complex.I) * Real.sin θ + (7 + 4 * Complex.I) * Real.cos θ
  let z2 (θ : ℝ) := (1 + 8 * Complex.I) * Real.sin θ + (4 + 7 * Complex.I) * Real.cos θ
  ∃ (θ : ℝ), ∀ (φ : ℝ), (z1 θ * z2 θ).re + (z1 θ * z2 θ).im ≥ (z1 φ * z2 φ).re + (z1 φ * z2 φ).im ∧
  (z1 θ * z2 θ).re + (z1 θ * z2 θ).im = 125 :=
by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_complex_product_l999_99979


namespace NUMINAMATH_CALUDE_smallest_equal_packages_l999_99949

/-- The number of pencils in each pack -/
def pencils_per_pack : ℕ := 10

/-- The number of pencil sharpeners in each pack -/
def sharpeners_per_pack : ℕ := 14

/-- The smallest number of pencil sharpener packages needed -/
def min_sharpener_packages : ℕ := 5

theorem smallest_equal_packages :
  ∃ (pencil_packs : ℕ),
    pencil_packs * pencils_per_pack = min_sharpener_packages * sharpeners_per_pack ∧
    ∀ (k : ℕ), k < min_sharpener_packages →
      ¬∃ (m : ℕ), m * pencils_per_pack = k * sharpeners_per_pack :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_packages_l999_99949


namespace NUMINAMATH_CALUDE_simplify_expression_l999_99913

theorem simplify_expression (a : ℝ) : (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) = 120 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l999_99913


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l999_99940

/-- Reflects a point (x, y) about the line y = -x --/
def reflectAboutNegativeX (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  reflectAboutNegativeX original_center = (3, -8) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l999_99940


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l999_99916

def M : ℕ := 42 * 43 * 75 * 196

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l999_99916


namespace NUMINAMATH_CALUDE_older_rabbit_catch_up_steps_l999_99923

/-- Represents the rabbits in the race -/
inductive Rabbit
| Younger
| Older

/-- Properties of the rabbit race -/
structure RaceProperties where
  initial_lead : ℕ
  younger_steps_per_time : ℕ
  older_steps_per_time : ℕ
  younger_distance_steps : ℕ
  older_distance_steps : ℕ
  younger_distance : ℕ
  older_distance : ℕ

/-- The race between the two rabbits -/
def rabbit_race (props : RaceProperties) : Prop :=
  props.initial_lead = 10 ∧
  props.younger_steps_per_time = 4 ∧
  props.older_steps_per_time = 3 ∧
  props.younger_distance_steps = 7 ∧
  props.older_distance_steps = 5 ∧
  props.younger_distance = props.older_distance

/-- Theorem stating the number of steps for the older rabbit to catch up -/
theorem older_rabbit_catch_up_steps (props : RaceProperties) 
  (h : rabbit_race props) : ∃ (steps : ℕ), steps = 150 := by
  sorry


end NUMINAMATH_CALUDE_older_rabbit_catch_up_steps_l999_99923


namespace NUMINAMATH_CALUDE_t_of_f_6_l999_99932

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)

noncomputable def f (x : ℝ) : ℝ := 6 - t x

theorem t_of_f_6 : t (f 6) = Real.sqrt 26 - 2 := by
  sorry

end NUMINAMATH_CALUDE_t_of_f_6_l999_99932


namespace NUMINAMATH_CALUDE_percentage_problem_l999_99999

theorem percentage_problem (x : ℝ) : (23 / 100) * x = 150 → x = 15000 / 23 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l999_99999


namespace NUMINAMATH_CALUDE_inequality_proof_l999_99957

theorem inequality_proof (a b : ℝ) (h1 : a < 0) (h2 : b > 0) (h3 : a + b < 0) :
  -a > b ∧ b > -b ∧ -b > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l999_99957


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l999_99984

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l999_99984


namespace NUMINAMATH_CALUDE_circle_and_sphere_sum_l999_99964

theorem circle_and_sphere_sum (c : ℝ) (h : c = 18 * Real.pi) :
  let r := c / (2 * Real.pi)
  (Real.pi * r^2) + (4/3 * Real.pi * r^3) = 1053 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_and_sphere_sum_l999_99964


namespace NUMINAMATH_CALUDE_odd_function_theorem_l999_99978

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of function g in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

/-- Theorem: If f is odd, g(x) = f(x) + 2, and g(1) = 1, then g(-1) = 3 -/
theorem odd_function_theorem (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : g f 1 = 1) : g f (-1) = 3 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_theorem_l999_99978


namespace NUMINAMATH_CALUDE_catch_up_theorem_l999_99967

/-- The number of days after which the second student catches up with the first student -/
def catch_up_day : ℕ := 13

/-- The distance walked by the first student each day -/
def first_student_daily_distance : ℕ := 7

/-- The distance walked by the second student on the nth day -/
def second_student_daily_distance (n : ℕ) : ℕ := n

/-- The total distance walked by the first student after n days -/
def first_student_total_distance (n : ℕ) : ℕ :=
  n * first_student_daily_distance

/-- The total distance walked by the second student after n days -/
def second_student_total_distance (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem catch_up_theorem :
  first_student_total_distance catch_up_day = second_student_total_distance catch_up_day :=
by sorry

end NUMINAMATH_CALUDE_catch_up_theorem_l999_99967


namespace NUMINAMATH_CALUDE_cubic_root_sum_l999_99918

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a*b/c + b*c/a + c*a/b = 49/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l999_99918


namespace NUMINAMATH_CALUDE_glove_selection_ways_l999_99937

/-- The number of different pairs of gloves -/
def num_pairs : ℕ := 6

/-- The number of gloves to be selected -/
def num_selected : ℕ := 4

/-- The number of matching pairs in the selection -/
def num_matching_pairs : ℕ := 1

/-- The total number of ways to select the gloves -/
def total_ways : ℕ := 240

theorem glove_selection_ways :
  (num_pairs : ℕ) = 6 →
  (num_selected : ℕ) = 4 →
  (num_matching_pairs : ℕ) = 1 →
  (total_ways : ℕ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_glove_selection_ways_l999_99937


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l999_99952

-- Define the conditions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is sufficient but not necessary for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l999_99952


namespace NUMINAMATH_CALUDE_journey_distance_l999_99981

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 5)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 112 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l999_99981


namespace NUMINAMATH_CALUDE_expected_draws_for_specific_box_l999_99988

/-- A box containing red and white balls -/
structure Box where
  red : ℕ
  white : ℕ

/-- The expected number of draws needed to pick a white ball -/
def expectedDraws (b : Box) : ℚ :=
  -- Definition to be proved
  11/9

/-- Theorem stating the expected number of draws for a specific box configuration -/
theorem expected_draws_for_specific_box :
  let b : Box := ⟨2, 8⟩
  expectedDraws b = 11/9 := by
  sorry


end NUMINAMATH_CALUDE_expected_draws_for_specific_box_l999_99988


namespace NUMINAMATH_CALUDE_set_relationship_theorem_l999_99985

def A : Set ℝ := {-1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 2}

def whale_swallowing (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def moth_eating (X Y : Set ℝ) : Prop := 
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem set_relationship_theorem : 
  {a : ℝ | a ≥ 0 ∧ (whale_swallowing A (B a) ∨ moth_eating A (B a))} = {0, 1/2, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_theorem_l999_99985


namespace NUMINAMATH_CALUDE_sum_E_equals_1600_l999_99917

-- Define E(n) as the sum of even digits in n
def E (n : ℕ) : ℕ := sorry

-- Define the sum of E(n) from 1 to 200
def sum_E : ℕ := (Finset.range 200).sum (fun i => E (i + 1))

-- Theorem to prove
theorem sum_E_equals_1600 : sum_E = 1600 := by sorry

end NUMINAMATH_CALUDE_sum_E_equals_1600_l999_99917


namespace NUMINAMATH_CALUDE_new_girl_weight_l999_99973

theorem new_girl_weight (W : ℝ) (new_weight : ℝ) :
  (W - 40 + new_weight) / 20 = W / 20 + 2 →
  new_weight = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l999_99973


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l999_99986

theorem curve_is_hyperbola (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) →
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), b * x^2 + a * y^2 = a * b ↔ x^2 / A - y^2 / B = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l999_99986


namespace NUMINAMATH_CALUDE_plane_speed_calculation_l999_99909

theorem plane_speed_calculation (D : ℝ) (V : ℝ) (h1 : D = V * 5) (h2 : D = 720 * (5/3)) :
  V = 240 := by
sorry

end NUMINAMATH_CALUDE_plane_speed_calculation_l999_99909


namespace NUMINAMATH_CALUDE_shopping_mall_purchase_l999_99907

/-- Represents the shopping mall's purchase of products A and B -/
structure ProductPurchase where
  cost_price_A : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ
  profit_margin_B : ℝ
  total_units : ℕ
  total_cost : ℝ

/-- Theorem stating the correct number of units purchased for each product -/
theorem shopping_mall_purchase (p : ProductPurchase)
  (h1 : p.cost_price_A = 40)
  (h2 : p.selling_price_A = 60)
  (h3 : p.selling_price_B = 80)
  (h4 : p.profit_margin_B = 0.6)
  (h5 : p.total_units = 50)
  (h6 : p.total_cost = 2200) :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = p.total_units ∧
    units_A * p.cost_price_A + units_B * (p.selling_price_B / (1 + p.profit_margin_B)) = p.total_cost ∧
    units_A = 30 ∧
    units_B = 20 := by
  sorry


end NUMINAMATH_CALUDE_shopping_mall_purchase_l999_99907


namespace NUMINAMATH_CALUDE_fedya_can_keep_below_1000_l999_99998

/-- Represents the state of the number on the screen -/
structure ScreenNumber where
  value : ℕ
  minutes : ℕ

/-- Increases the number by 102 -/
def increment (n : ScreenNumber) : ScreenNumber :=
  { value := n.value + 102, minutes := n.minutes + 1 }

/-- Rearranges the digits of a number -/
def rearrange (n : ℕ) : ℕ := sorry

/-- Fedya's strategy to keep the number below 1000 -/
def fedya_strategy (n : ScreenNumber) : ScreenNumber :=
  if n.value < 1000 then n else { n with value := rearrange n.value }

/-- Theorem stating that Fedya can always keep the number below 1000 -/
theorem fedya_can_keep_below_1000 :
  ∀ (n : ℕ), n < 1000 →
  ∃ (strategy : ℕ → ScreenNumber),
    (∀ (k : ℕ), (strategy k).value < 1000) ∧
    strategy 0 = { value := 123, minutes := 0 } ∧
    (∀ (k : ℕ), strategy (k + 1) = fedya_strategy (increment (strategy k))) :=
sorry

end NUMINAMATH_CALUDE_fedya_can_keep_below_1000_l999_99998


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l999_99934

/-- For any constant a > 0 and a ≠ 1, the function f(x) = a^(x-1) - 1 passes through the point (1, 0) -/
theorem fixed_point_of_exponential_shift (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) - 1
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l999_99934


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l999_99968

/-- Represents the maximum distance a car can travel by switching tires -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  let switch_point := front_tire_life / 2
  switch_point + min (front_tire_life - switch_point) (rear_tire_life - switch_point)

/-- Theorem stating the maximum distance a car can travel with given tire lives -/
theorem max_distance_for_given_tires :
  max_distance 21000 28000 = 24000 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l999_99968


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l999_99956

/-- The length of a rectangular solid with given dimensions and surface area -/
theorem rectangular_solid_length
  (width : ℝ) (depth : ℝ) (surface_area : ℝ)
  (h_width : width = 9)
  (h_depth : depth = 6)
  (h_surface_area : surface_area = 408)
  (h_formula : surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth) :
  length = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l999_99956


namespace NUMINAMATH_CALUDE_tammy_climbing_speed_l999_99931

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing_speed :
  ∀ (v : ℝ), -- v represents the speed on the first day
  v > 0 →
  v * 7 + (v + 0.5) * 5 + (v + 1.5) * 8 = 85 →
  7 + 5 + 8 = 20 →
  (v + 0.5) = 4.025 :=
by
  sorry

end NUMINAMATH_CALUDE_tammy_climbing_speed_l999_99931


namespace NUMINAMATH_CALUDE_prob_even_after_removal_l999_99983

/-- Probability of selecting a dot from a face with n dots -/
def probSelectDot (n : ℕ) : ℚ := n / 21

/-- Probability that a face with n dots remains even after removing two dots -/
def probRemainsEven (n : ℕ) : ℚ :=
  if n % 2 = 0
  then 1 - probSelectDot n * ((n - 1) / 20)
  else probSelectDot n * ((n - 1) / 20)

/-- The probability of rolling an even number of dots after removing two random dots -/
def probEvenAfterRemoval : ℚ :=
  (1 / 6) * (probRemainsEven 1 + probRemainsEven 2 + probRemainsEven 3 +
             probRemainsEven 4 + probRemainsEven 5 + probRemainsEven 6)

theorem prob_even_after_removal :
  probEvenAfterRemoval = 167 / 630 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_after_removal_l999_99983


namespace NUMINAMATH_CALUDE_original_number_exists_l999_99987

theorem original_number_exists : ∃ x : ℝ, 3 * (2 * x + 5) = 117 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_l999_99987


namespace NUMINAMATH_CALUDE_equation_solution_l999_99919

theorem equation_solution : ∃! x : ℝ, (x + 1) / 2 = x - 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l999_99919


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l999_99947

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d ≤ 10000 ∧ 
  ∃ a b c d, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
             a + b + c + d = 200 ∧ 
             a * b + b * c + c * d = 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l999_99947


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l999_99900

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 14 :=
by
  -- The unique solution is x = 4.5
  use 4.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l999_99900


namespace NUMINAMATH_CALUDE_max_value_is_27_l999_99943

/-- Represents the crop types -/
inductive Crop
| Melon
| Fruit
| Vegetable

/-- Represents the problem parameters -/
structure ProblemParams where
  totalLaborers : ℕ
  totalLand : ℕ
  laborRequirement : Crop → ℚ
  cropValue : Crop → ℚ

/-- Represents the allocation of crops to land -/
structure Allocation where
  melonAcres : ℚ
  fruitAcres : ℚ
  vegetableAcres : ℚ

/-- Calculates the total value for a given allocation -/
def totalValue (p : ProblemParams) (a : Allocation) : ℚ :=
  a.melonAcres * p.cropValue Crop.Melon +
  a.fruitAcres * p.cropValue Crop.Fruit +
  a.vegetableAcres * p.cropValue Crop.Vegetable

/-- Checks if an allocation is valid according to the problem constraints -/
def isValidAllocation (p : ProblemParams) (a : Allocation) : Prop :=
  a.melonAcres + a.fruitAcres + a.vegetableAcres = p.totalLand ∧
  a.melonAcres * p.laborRequirement Crop.Melon +
  a.fruitAcres * p.laborRequirement Crop.Fruit +
  a.vegetableAcres * p.laborRequirement Crop.Vegetable = p.totalLaborers

/-- The main theorem stating that the maximum value is 27 million yuan -/
theorem max_value_is_27 (p : ProblemParams)
  (h1 : p.totalLaborers = 20)
  (h2 : p.totalLand = 50)
  (h3 : p.laborRequirement Crop.Melon = 1/2)
  (h4 : p.laborRequirement Crop.Fruit = 1/3)
  (h5 : p.laborRequirement Crop.Vegetable = 1/4)
  (h6 : p.cropValue Crop.Melon = 6/10)
  (h7 : p.cropValue Crop.Fruit = 1/2)
  (h8 : p.cropValue Crop.Vegetable = 3/10) :
  ∃ (a : Allocation), isValidAllocation p a ∧
    ∀ (a' : Allocation), isValidAllocation p a' → totalValue p a' ≤ 27 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_27_l999_99943


namespace NUMINAMATH_CALUDE_hot_tea_sales_average_l999_99970

/-- Represents the linear relationship between temperature and cups of hot tea sold -/
structure HotDrinkSales where
  slope : ℝ
  intercept : ℝ

/-- Calculates the average cups of hot tea sold given average temperature -/
def average_sales (model : HotDrinkSales) (avg_temp : ℝ) : ℝ :=
  model.slope * avg_temp + model.intercept

theorem hot_tea_sales_average (model : HotDrinkSales) (avg_temp : ℝ) 
    (h1 : model.slope = -2)
    (h2 : model.intercept = 58)
    (h3 : avg_temp = 12) :
    average_sales model avg_temp = 34 := by
  sorry

#check hot_tea_sales_average

end NUMINAMATH_CALUDE_hot_tea_sales_average_l999_99970


namespace NUMINAMATH_CALUDE_third_side_length_l999_99980

theorem third_side_length (a b : ℝ) (h1 : a = 3.14) (h2 : b = 0.67) : 
  ∃ m : ℤ, (m : ℝ) > |a - b| ∧ (m : ℝ) < a + b ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_side_length_l999_99980


namespace NUMINAMATH_CALUDE_negation_of_cubic_greater_than_square_l999_99901

theorem negation_of_cubic_greater_than_square :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) := by sorry

end NUMINAMATH_CALUDE_negation_of_cubic_greater_than_square_l999_99901


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l999_99910

theorem max_value_of_exponential_difference (x : ℝ) :
  ∃ (max : ℝ), max = 1/4 ∧ ∀ (y : ℝ), 5^y - 25^y ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l999_99910


namespace NUMINAMATH_CALUDE_problem_statement_l999_99960

-- Define proposition p
def p : Prop := ∀ x : ℝ, (|x| = x ↔ x > 0)

-- Define proposition q
def q : Prop := (¬∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem to prove
theorem problem_statement : ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l999_99960


namespace NUMINAMATH_CALUDE_min_y_value_l999_99951

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 10*x + 36*y) : 
  ∀ z : ℝ, (∃ w : ℝ, w^2 + z^2 = 10*w + 36*z) → y ≤ z → -7 ≤ y :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l999_99951


namespace NUMINAMATH_CALUDE_age_ratio_problem_l999_99965

theorem age_ratio_problem (sam drew : ℕ) : 
  sam + drew = 54 → sam = 18 → sam * 2 = drew :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l999_99965


namespace NUMINAMATH_CALUDE_matrix_equation_implies_even_dimension_l999_99926

theorem matrix_equation_implies_even_dimension (n : ℕ+) :
  (∃ (A B : Matrix (Fin n) (Fin n) ℝ), 
    Matrix.det A ≠ 0 ∧ 
    Matrix.det B ≠ 0 ∧ 
    A * B - B * A = B ^ 2 * A) → 
  Even n := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_implies_even_dimension_l999_99926


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l999_99922

/-- 
Given a boat with speed 48 kmph in still water and a stream with speed 16 kmph,
prove that the ratio of time taken to row upstream to the time taken to row downstream is 2:1.
-/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 48) 
  (h2 : stream_speed = 16) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l999_99922


namespace NUMINAMATH_CALUDE_yuna_candies_l999_99906

theorem yuna_candies (initial_candies remaining_candies : ℕ) 
  (h1 : initial_candies = 23)
  (h2 : remaining_candies = 7) :
  initial_candies - remaining_candies = 16 := by
  sorry

end NUMINAMATH_CALUDE_yuna_candies_l999_99906


namespace NUMINAMATH_CALUDE_inequality_proof_l999_99953

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l999_99953


namespace NUMINAMATH_CALUDE_derivative_x_sin_x_l999_99944

theorem derivative_x_sin_x (x : ℝ) :
  let f : ℝ → ℝ := λ x => x * Real.sin x
  (deriv f) x = Real.sin x + x * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_sin_x_l999_99944


namespace NUMINAMATH_CALUDE_subtract_from_square_l999_99950

theorem subtract_from_square (n : ℕ) (h : n = 17) : n^2 - n = 272 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_square_l999_99950


namespace NUMINAMATH_CALUDE_negation_equivalence_l999_99990

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l999_99990


namespace NUMINAMATH_CALUDE_sum_remainder_by_eight_l999_99974

theorem sum_remainder_by_eight (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_eight_l999_99974


namespace NUMINAMATH_CALUDE_parabola_coefficients_l999_99972

/-- A parabola with vertex (h, k), vertical axis of symmetry, passing through point (x₀, y₀) -/
structure Parabola where
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The quadratic function representing the parabola -/
def quadratic_function (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem parabola_coefficients (p : Parabola) 
  (h_vertex : p.h = 2 ∧ p.k = -3)
  (h_point : p.x₀ = 0 ∧ p.y₀ = 1)
  (h_passes : quadratic_function p p.x₀ = p.y₀) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -4 ∧ c = 1 ∧
  ∀ x, quadratic_function p x = a * x^2 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l999_99972


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l999_99997

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 2 = 8 →                                 -- Given condition
  a 5 = 64 →                                -- Given condition
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l999_99997


namespace NUMINAMATH_CALUDE_billing_method_comparison_l999_99992

/-- Cost calculation for Method A -/
def cost_A (x : ℝ) : ℝ := 8 + 0.2 * x

/-- Cost calculation for Method B -/
def cost_B (x : ℝ) : ℝ := 0.3 * x

/-- Theorem comparing billing methods based on call duration -/
theorem billing_method_comparison (x : ℝ) :
  (x < 80 → cost_B x < cost_A x) ∧
  (x = 80 → cost_A x = cost_B x) ∧
  (x > 80 → cost_A x < cost_B x) := by
  sorry

end NUMINAMATH_CALUDE_billing_method_comparison_l999_99992


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l999_99920

theorem algebraic_expression_value (x : ℝ) :
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l999_99920


namespace NUMINAMATH_CALUDE_intersection_M_N_l999_99994

def M : Set ℝ := {0, 2}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l999_99994


namespace NUMINAMATH_CALUDE_train_length_l999_99942

/-- Calculates the length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh / 3.6 * time_s = 225 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l999_99942


namespace NUMINAMATH_CALUDE_division_problem_l999_99955

theorem division_problem (dividend quotient divisor remainder : ℕ) 
  (h1 : remainder = 8)
  (h2 : divisor = 3 * remainder + 3)
  (h3 : dividend = 251)
  (h4 : dividend = divisor * quotient + remainder) :
  ∃ (m : ℕ), divisor = m * quotient ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l999_99955


namespace NUMINAMATH_CALUDE_digit_sum_property_l999_99971

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : sum_of_digits n = 100)
  (h2 : sum_of_digits (44 * n) = 800) : 
  sum_of_digits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l999_99971


namespace NUMINAMATH_CALUDE_tips_fraction_of_income_l999_99929

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the total income of a waitress -/
def totalIncome (w : WaitressIncome) : ℚ :=
  w.salary + w.tips

/-- Theorem: If a waitress's tips are 7/4 of her salary, then 7/11 of her income comes from tips -/
theorem tips_fraction_of_income (w : WaitressIncome) 
  (h : w.tips = (7 : ℚ) / 4 * w.salary) : 
  w.tips / totalIncome w = (7 : ℚ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_of_income_l999_99929


namespace NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_mod_5_l999_99977

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square_mod_5 (n : ℕ) : Prop :=
  ∃ m : ℕ, n ≡ m^2 [ZMOD 5]

theorem sum_of_factorials_perfect_square_mod_5 (n : ℕ+) :
  is_perfect_square_mod_5 (sum_of_factorials n) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_mod_5_l999_99977


namespace NUMINAMATH_CALUDE_smallest_s_for_arithmetic_progression_l999_99982

open Real

theorem smallest_s_for_arithmetic_progression (β : ℝ) (s : ℝ) :
  0 < β ∧ β < π / 2 →
  (∃ d : ℝ, arcsin (sin (3 * β)) + d = arcsin (sin (5 * β)) ∧
            arcsin (sin (5 * β)) + d = arcsin (sin (10 * β)) ∧
            arcsin (sin (10 * β)) + d = arcsin (sin (s * β))) →
  s ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_s_for_arithmetic_progression_l999_99982


namespace NUMINAMATH_CALUDE_square_circle_union_area_l999_99959

/-- The area of the union of a square with side length 10 and a circle with radius 10 
    centered at one of the square's vertices is equal to 100 + 75π. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 10
  let circle_radius : ℝ := 10
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1 / 4 : ℝ) * circle_area
  square_area + circle_area - overlap_area = 100 + 75 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l999_99959


namespace NUMINAMATH_CALUDE_candy_sales_average_l999_99933

/-- The average of candy sales for five months -/
def average_candy_sales (jan feb mar apr may : ℕ) : ℚ :=
  (jan + feb + mar + apr + may) / 5

/-- Theorem stating that the average candy sales is 96 dollars -/
theorem candy_sales_average :
  average_candy_sales 110 80 70 130 90 = 96 := by sorry

end NUMINAMATH_CALUDE_candy_sales_average_l999_99933
