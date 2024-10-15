import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2770_277097

open Set

def A : Set ℝ := {x | |x - 1| ≥ 2}
def B : Set ℕ := {x | x < 4}

theorem complement_A_intersect_B :
  (𝒰 \ A) ∩ (coe '' B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2770_277097


namespace NUMINAMATH_CALUDE_parallel_vectors_mn_value_l2770_277073

def vector_a (m n : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 2
  | 1 => 2*m - 3
  | 2 => n + 2

def vector_b (m n : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 4
  | 1 => 2*m + 1
  | 2 => 3*n - 2

theorem parallel_vectors_mn_value (m n : ℝ) :
  (∃ (k : ℝ), ∀ (i : Fin 3), vector_a m n i = k * vector_b m n i) →
  m * n = 21 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_mn_value_l2770_277073


namespace NUMINAMATH_CALUDE_homework_pages_proof_l2770_277028

theorem homework_pages_proof (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 8 → 
  math_pages = reading_pages + 3 → 
  total_pages = math_pages + reading_pages → 
  total_pages = 13 := by
sorry

end NUMINAMATH_CALUDE_homework_pages_proof_l2770_277028


namespace NUMINAMATH_CALUDE_even_function_sum_l2770_277010

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-2*a) (3*a - 1), f b x = f b (-x)) →
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_even_function_sum_l2770_277010


namespace NUMINAMATH_CALUDE_absolute_value_problem_l2770_277061

theorem absolute_value_problem (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : 
  y - 2*q = 3 - 3*q := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l2770_277061


namespace NUMINAMATH_CALUDE_range_of_m_l2770_277004

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x - 2 - m < 0) → m > -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2770_277004


namespace NUMINAMATH_CALUDE_equal_cost_at_20_minutes_unique_solution_l2770_277091

/-- The base rate for United Telephone service -/
def united_base_rate : ℝ := 11

/-- The per-minute charge for United Telephone -/
def united_per_minute : ℝ := 0.25

/-- The base rate for Atlantic Call service -/
def atlantic_base_rate : ℝ := 12

/-- The per-minute charge for Atlantic Call -/
def atlantic_per_minute : ℝ := 0.20

/-- The total cost for United Telephone service for m minutes -/
def united_cost (m : ℝ) : ℝ := united_base_rate + united_per_minute * m

/-- The total cost for Atlantic Call service for m minutes -/
def atlantic_cost (m : ℝ) : ℝ := atlantic_base_rate + atlantic_per_minute * m

/-- Theorem stating that the costs are equal at 20 minutes -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 :=
by sorry

/-- Theorem stating that 20 minutes is the unique solution -/
theorem unique_solution (m : ℝ) :
  united_cost m = atlantic_cost m ↔ m = 20 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_20_minutes_unique_solution_l2770_277091


namespace NUMINAMATH_CALUDE_second_train_length_proof_l2770_277078

/-- The length of the second train given the conditions of the problem -/
def second_train_length : ℝ := 199.9760019198464

/-- The length of the first train in meters -/
def first_train_length : ℝ := 100

/-- The speed of the first train in kilometers per hour -/
def first_train_speed : ℝ := 42

/-- The speed of the second train in kilometers per hour -/
def second_train_speed : ℝ := 30

/-- The time it takes for the trains to clear each other in seconds -/
def clearing_time : ℝ := 14.998800095992321

theorem second_train_length_proof :
  second_train_length = 
    (first_train_speed + second_train_speed) * (1000 / 3600) * clearing_time - first_train_length := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_proof_l2770_277078


namespace NUMINAMATH_CALUDE_heating_pad_cost_per_use_l2770_277059

/-- The cost per use of a heating pad -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * weeks)

/-- Theorem: The cost per use of a $30 heating pad used 3 times a week for 2 weeks is $5 -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by sorry

end NUMINAMATH_CALUDE_heating_pad_cost_per_use_l2770_277059


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2770_277043

theorem polynomial_factorization (x y z : ℝ) :
  2 * x^3 - x^2 * z - 4 * x^2 * y + 2 * x * y * z + 2 * x * y^2 - y^2 * z = (2 * x - z) * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2770_277043


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l2770_277041

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 8) : 
  a^2 + b^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l2770_277041


namespace NUMINAMATH_CALUDE_nara_height_l2770_277008

/-- Given the heights of Sangheon, Chiho, and Nara, prove Nara's height -/
theorem nara_height (sangheon_height : ℝ) (chiho_diff : ℝ) (nara_diff : ℝ) :
  sangheon_height = 1.56 →
  chiho_diff = 0.14 →
  nara_diff = 0.27 →
  sangheon_height - chiho_diff + nara_diff = 1.69 := by
sorry

end NUMINAMATH_CALUDE_nara_height_l2770_277008


namespace NUMINAMATH_CALUDE_selling_price_a_is_1600_l2770_277051

/-- Represents the sales and pricing information for bicycle types A and B --/
structure BikeData where
  lastYearTotalSalesA : ℕ
  priceDecreaseA : ℕ
  salesDecreasePercentage : ℚ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ

/-- Calculates the selling price of type A bikes this year --/
def calculateSellingPriceA (data : BikeData) : ℕ :=
  sorry

/-- Theorem stating that the selling price of type A bikes this year is 1600 yuan --/
theorem selling_price_a_is_1600 (data : BikeData) 
  (h1 : data.lastYearTotalSalesA = 50000)
  (h2 : data.priceDecreaseA = 400)
  (h3 : data.salesDecreasePercentage = 1/5)
  (h4 : data.purchasePriceA = 1100)
  (h5 : data.purchasePriceB = 1400)
  (h6 : data.sellingPriceB = 2000) :
  calculateSellingPriceA data = 1600 :=
sorry

end NUMINAMATH_CALUDE_selling_price_a_is_1600_l2770_277051


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2770_277055

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age : 
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℚ),
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 3 →
  (team_size : ℚ) * A = 
    ((team_size - 2) : ℚ) * (A - 1) + 
    (captain_age : ℚ) + 
    ((captain_age + wicket_keeper_age_diff) : ℚ) →
  A = 31 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2770_277055


namespace NUMINAMATH_CALUDE_total_marbles_count_l2770_277076

/-- The total number of marbles Mary and Joan have -/
def total_marbles : ℕ :=
  let mary_yellow := 9
  let mary_blue := 7
  let mary_green := 6
  let joan_yellow := 3
  let joan_blue := 5
  let joan_green := 4
  mary_yellow + mary_blue + mary_green + joan_yellow + joan_blue + joan_green

theorem total_marbles_count : total_marbles = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l2770_277076


namespace NUMINAMATH_CALUDE_servant_cash_compensation_l2770_277019

/-- Calculates the cash compensation for a servant given the annual salary, work period, and value of a non-cash item received. -/
def servant_compensation (annual_salary : ℚ) (work_months : ℕ) (item_value : ℚ) : ℚ :=
  annual_salary * (work_months / 12 : ℚ) - item_value

/-- Proves that the cash compensation for the servant is 57.5 given the problem conditions. -/
theorem servant_cash_compensation : 
  servant_compensation 90 9 10 = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_servant_cash_compensation_l2770_277019


namespace NUMINAMATH_CALUDE_teachers_combined_age_l2770_277034

theorem teachers_combined_age
  (num_students : ℕ)
  (student_avg_age : ℚ)
  (num_teachers : ℕ)
  (total_avg_age : ℚ)
  (h1 : num_students = 30)
  (h2 : student_avg_age = 18)
  (h3 : num_teachers = 2)
  (h4 : total_avg_age = 19) :
  (num_students + num_teachers) * total_avg_age -
  (num_students * student_avg_age) = 68 := by
sorry

end NUMINAMATH_CALUDE_teachers_combined_age_l2770_277034


namespace NUMINAMATH_CALUDE_fourth_root_of_polynomial_l2770_277081

theorem fourth_root_of_polynomial (a b : ℝ) : 
  (∀ x : ℝ, b * x^3 + (3*b + a) * x^2 + (a - 2*b) * x + (5 - b) = 0 ↔ 
    x = -1 ∨ x = 2 ∨ x = 4 ∨ x = -8) → 
  ∃ x : ℝ, x = -8 ∧ b * x^3 + (3*b + a) * x^2 + (a - 2*b) * x + (5 - b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_of_polynomial_l2770_277081


namespace NUMINAMATH_CALUDE_quiz_probability_l2770_277035

theorem quiz_probability (n : ℕ) (m : ℕ) (p : ℚ) : 
  n = 6 → 
  m = 4 → 
  p = 1 - (3/4)^6 → 
  p = 3367/4096 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l2770_277035


namespace NUMINAMATH_CALUDE_jaco_total_payment_l2770_277037

/-- Calculates the total amount a customer pays given item prices and a discount policy. -/
def calculateTotalWithDiscount (shoePrice sockPrice bagPrice : ℚ) : ℚ :=
  let totalBeforeDiscount := shoePrice + 2 * sockPrice + bagPrice
  let discountableAmount := max (totalBeforeDiscount - 100) 0
  let discount := discountableAmount * (1 / 10)
  totalBeforeDiscount - discount

/-- Theorem stating that Jaco will pay $118 for his purchases. -/
theorem jaco_total_payment :
  calculateTotalWithDiscount 74 2 42 = 118 := by
  sorry

#eval calculateTotalWithDiscount 74 2 42

end NUMINAMATH_CALUDE_jaco_total_payment_l2770_277037


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2770_277070

theorem intersection_chord_length (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → (x - 3)^2 + (y - 2)^2 = 4 → 
    ∃ M N : ℝ × ℝ, 
      (M.1 - 3)^2 + (M.2 - 2)^2 = 4 ∧ 
      (N.1 - 3)^2 + (N.2 - 2)^2 = 4 ∧ 
      M.2 = k * M.1 + 3 ∧ 
      N.2 = k * N.1 + 3 ∧ 
      (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) → 
  -3/4 ≤ k ∧ k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2770_277070


namespace NUMINAMATH_CALUDE_students_over_capacity_l2770_277016

/-- Calculates the number of students over capacity given the initial conditions --/
theorem students_over_capacity
  (ratio : ℚ)
  (teachers : ℕ)
  (increase_percent : ℚ)
  (capacity : ℕ)
  (h_ratio : ratio = 27.5)
  (h_teachers : teachers = 42)
  (h_increase : increase_percent = 0.15)
  (h_capacity : capacity = 1300) :
  ⌊(ratio * teachers) * (1 + increase_percent)⌋ - capacity = 28 :=
by sorry

end NUMINAMATH_CALUDE_students_over_capacity_l2770_277016


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l2770_277065

theorem max_value_theorem (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 17 :=
sorry

theorem max_value_achievable : 
  ∃ (x y : ℝ), (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) = Real.sqrt 17 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l2770_277065


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_59_l2770_277025

theorem least_positive_integer_multiple_59 : 
  ∃ (x : ℕ+), (∀ (y : ℕ+), y < x → ¬(59 ∣ (2 * y + 51)^2)) ∧ (59 ∣ (2 * x + 51)^2) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_59_l2770_277025


namespace NUMINAMATH_CALUDE_max_product_value_l2770_277013

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition1 : a 1 + a 3 = 30
  sum_condition2 : a 2 + a 4 = 10

/-- The product of the first n terms of the sequence -/
def product (seq : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (Finset.range n).prod (fun i => seq.a ⟨i + 1, Nat.succ_pos i⟩)

/-- The theorem stating the maximum value of the product -/
theorem max_product_value (seq : ArithmeticSequence) :
  ∃ max_val : ℝ, max_val = 729 ∧ ∀ n : ℕ+, product seq n ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_product_value_l2770_277013


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_15_l2770_277044

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |> List.sum

theorem units_digit_factorial_sum_15 :
  units_digit (factorial_sum 15) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_15_l2770_277044


namespace NUMINAMATH_CALUDE_point_difference_on_plane_l2770_277015

/-- Given two points on a plane, prove that the difference in their x and z coordinates are 3 and 0 respectively. -/
theorem point_difference_on_plane (m n z p q : ℝ) (k : ℝ) (hk : k ≠ 0) :
  (m = n / 6 - 2 / 5 + z / k) →
  (m + p = (n + 18) / 6 - 2 / 5 + (z + q) / k) →
  p = 3 ∧ q = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_difference_on_plane_l2770_277015


namespace NUMINAMATH_CALUDE_cos_squared_sum_range_l2770_277064

theorem cos_squared_sum_range (α β : ℝ) (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  ∃ (x : ℝ), x ∈ Set.Icc (14/9 : ℝ) 2 ∧
  x = (Real.cos α)^2 + (Real.cos β)^2 ∧
  ∀ (y : ℝ), y = (Real.cos α)^2 + (Real.cos β)^2 → y ∈ Set.Icc (14/9 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_cos_squared_sum_range_l2770_277064


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l2770_277039

theorem square_sum_nonzero_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l2770_277039


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2770_277063

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r > 0, ∀ n, a (n + 1) = r * a n)

/-- The fourth term of a positive geometric sequence is 2 if the product of the third and fifth terms is 4 -/
theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : PositiveGeometricSequence a)
  (h_prod : a 3 * a 5 = 4) :
  a 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2770_277063


namespace NUMINAMATH_CALUDE_difference_of_squares_l2770_277089

theorem difference_of_squares (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) :
  a^2 - b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2770_277089


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l2770_277099

-- Define the universal set U
def U : Set Nat := {1, 3, 5}

-- Define the set A
def A : Set Nat := {1, 5}

-- State the theorem
theorem complement_of_A_wrt_U :
  {x ∈ U | x ∉ A} = {3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l2770_277099


namespace NUMINAMATH_CALUDE_average_price_reduction_l2770_277027

theorem average_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x ≥ 0 ∧ x ≤ 1) : 
  x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_average_price_reduction_l2770_277027


namespace NUMINAMATH_CALUDE_dogsled_race_speed_difference_l2770_277079

theorem dogsled_race_speed_difference 
  (course_length : ℝ) 
  (team_w_speed : ℝ) 
  (time_difference : ℝ) :
  course_length = 300 →
  team_w_speed = 20 →
  time_difference = 3 →
  let team_w_time := course_length / team_w_speed
  let team_a_time := team_w_time - time_difference
  let team_a_speed := course_length / team_a_time
  team_a_speed - team_w_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_difference_l2770_277079


namespace NUMINAMATH_CALUDE_consecutive_roots_quadratic_l2770_277088

theorem consecutive_roots_quadratic (n : ℕ) (hn : n > 1) :
  let f : ℝ → ℝ := λ x => x^2 - (2*n - 1)*x + n*(n-1)
  (f (n - 1) = 0) ∧ (f n = 0) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_roots_quadratic_l2770_277088


namespace NUMINAMATH_CALUDE_johns_roommates_multiple_l2770_277036

/-- Given that Bob has 10 roommates and John has 25 roommates, 
    prove that the multiple of Bob's roommates that John has five more than is 2. -/
theorem johns_roommates_multiple (bob_roommates john_roommates : ℕ) : 
  bob_roommates = 10 → john_roommates = 25 → 
  ∃ (x : ℕ), john_roommates = x * bob_roommates + 5 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_johns_roommates_multiple_l2770_277036


namespace NUMINAMATH_CALUDE_square_points_probability_l2770_277067

/-- The number of points around the square -/
def num_points : ℕ := 8

/-- The number of pairs of points that are one unit apart -/
def favorable_pairs : ℕ := 8

/-- The total number of ways to choose two points from the available points -/
def total_pairs : ℕ := num_points.choose 2

/-- The probability of choosing two points that are one unit apart -/
def probability : ℚ := favorable_pairs / total_pairs

theorem square_points_probability : probability = 2/7 := by sorry

end NUMINAMATH_CALUDE_square_points_probability_l2770_277067


namespace NUMINAMATH_CALUDE_tetrahedron_with_two_square_intersections_l2770_277023

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a tetrahedron and a plane -/
def intersection (t : Tetrahedron) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a square -/
def is_square (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- The side length of a square -/
def side_length (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the existence of a tetrahedron with the desired properties -/
theorem tetrahedron_with_two_square_intersections :
  ∃ (t : Tetrahedron) (p1 p2 : Plane),
    p1 ≠ p2 ∧
    is_square (intersection t p1) ∧
    is_square (intersection t p2) ∧
    side_length (intersection t p1) ≤ 1 ∧
    side_length (intersection t p2) ≥ 100 :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_with_two_square_intersections_l2770_277023


namespace NUMINAMATH_CALUDE_max_min_on_interval_l2770_277080

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = min) ∧
    max = 5 ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l2770_277080


namespace NUMINAMATH_CALUDE_toy_purchase_with_discount_l2770_277093

theorem toy_purchase_with_discount (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  discount_percent = 20 →
  (num_toys * cost_per_toy) * (1 - discount_percent / 100) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_with_discount_l2770_277093


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2770_277011

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area : 
  let h : ℝ := 8
  let r : ℝ := 3
  let lateral_area := 2 * π * r * h
  let base_area := π * r^2
  let total_surface_area := lateral_area + 2 * base_area
  total_surface_area = 66 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2770_277011


namespace NUMINAMATH_CALUDE_point_A_on_circle_point_B_on_circle_point_C_on_circle_circle_equation_unique_l2770_277092

/-- A circle passes through points A(2, 0), B(4, 0), and C(0, 2) -/
def circle_through_points (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 3)^2 = 10

/-- Point A lies on the circle -/
theorem point_A_on_circle : circle_through_points 2 0 := by sorry

/-- Point B lies on the circle -/
theorem point_B_on_circle : circle_through_points 4 0 := by sorry

/-- Point C lies on the circle -/
theorem point_C_on_circle : circle_through_points 0 2 := by sorry

/-- The equation (x - 3)² + (y - 3)² = 10 represents the unique circle 
    passing through points A(2, 0), B(4, 0), and C(0, 2) -/
theorem circle_equation_unique : 
  ∀ x y : ℝ, circle_through_points x y ↔ (x - 3)^2 + (y - 3)^2 = 10 := by sorry

end NUMINAMATH_CALUDE_point_A_on_circle_point_B_on_circle_point_C_on_circle_circle_equation_unique_l2770_277092


namespace NUMINAMATH_CALUDE_value_of_2a_plus_3b_l2770_277031

theorem value_of_2a_plus_3b (a b : ℚ) 
  (eq1 : 3 * a + 6 * b = 48) 
  (eq2 : 8 * a + 4 * b = 84) : 
  2 * a + 3 * b = 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_plus_3b_l2770_277031


namespace NUMINAMATH_CALUDE_present_difference_l2770_277071

/-- The number of presents Santana buys for her siblings in a year --/
def presents_count : ℕ → ℕ
| 1 => 4  -- March
| 2 => 1  -- May
| 3 => 1  -- June
| 4 => 1  -- October
| 5 => 1  -- November
| 6 => 2  -- December
| _ => 0

/-- The total number of siblings Santana has --/
def total_siblings : ℕ := 10

/-- The number of presents bought in the first half of the year --/
def first_half_presents : ℕ := presents_count 1 + presents_count 2 + presents_count 3

/-- The number of presents bought in the second half of the year --/
def second_half_presents : ℕ := 
  presents_count 4 + presents_count 5 + presents_count 6 + total_siblings + total_siblings

theorem present_difference : second_half_presents - first_half_presents = 18 := by
  sorry

#eval second_half_presents - first_half_presents

end NUMINAMATH_CALUDE_present_difference_l2770_277071


namespace NUMINAMATH_CALUDE_metal_waste_problem_l2770_277014

/-- Given a rectangle with length twice its width, prove that the area wasted
    when cutting out a maximum circular piece and then a maximum square piece
    from that circle is 3/2 of the original rectangle's area. -/
theorem metal_waste_problem (w : ℝ) (hw : w > 0) :
  let rectangle_area := 2 * w^2
  let circle_area := π * (w/2)^2
  let square_area := (w * Real.sqrt 2 / 2)^2
  let waste_area := rectangle_area - square_area
  waste_area = (3/2) * rectangle_area := by
  sorry

end NUMINAMATH_CALUDE_metal_waste_problem_l2770_277014


namespace NUMINAMATH_CALUDE_smallest_prime_minister_l2770_277082

/-- A positive integer is primer if it has a prime number of distinct prime factors. -/
def isPrimer (n : ℕ+) : Prop := sorry

/-- A positive integer is primest if it has a primer number of distinct primer factors. -/
def isPrimest (n : ℕ+) : Prop := sorry

/-- A positive integer is prime-minister if it has a primest number of distinct primest factors. -/
def isPrimeMinister (n : ℕ+) : Prop := sorry

/-- The smallest prime-minister number -/
def smallestPrimeMinister : ℕ+ := 378000

theorem smallest_prime_minister :
  isPrimeMinister smallestPrimeMinister ∧
  ∀ n : ℕ+, n < smallestPrimeMinister → ¬isPrimeMinister n := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_minister_l2770_277082


namespace NUMINAMATH_CALUDE_digit_addition_subtraction_problem_l2770_277021

/- Define digits as natural numbers from 0 to 9 -/
def Digit := {n : ℕ // n ≤ 9}

/- Define a function to convert a two-digit number to its value -/
def twoDigitValue (tens : Digit) (ones : Digit) : ℕ := 10 * tens.val + ones.val

theorem digit_addition_subtraction_problem (A B C D : Digit) :
  (twoDigitValue A B + twoDigitValue C A = twoDigitValue D A) ∧
  (twoDigitValue A B - twoDigitValue C A = A.val) →
  D.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_addition_subtraction_problem_l2770_277021


namespace NUMINAMATH_CALUDE_carl_typing_words_l2770_277090

/-- Calculates the total number of words typed given a typing speed, daily typing duration, and number of days. -/
def total_words_typed (typing_speed : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  typing_speed * 60 * hours_per_day * days

/-- Proves that given the specified conditions, Carl types 84000 words in 7 days. -/
theorem carl_typing_words :
  total_words_typed 50 4 7 = 84000 := by
  sorry

end NUMINAMATH_CALUDE_carl_typing_words_l2770_277090


namespace NUMINAMATH_CALUDE_average_income_l2770_277077

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    calculate the average monthly income of the remaining pair. -/
theorem average_income (p q r : ℕ) : 
  (p + q) / 2 = 2050 →
  (p + r) / 2 = 6200 →
  p = 3000 →
  (q + r) / 2 = 5250 := by
  sorry


end NUMINAMATH_CALUDE_average_income_l2770_277077


namespace NUMINAMATH_CALUDE_pencil_arrangement_theorem_l2770_277022

def yellow_pencils : ℕ := 6
def red_pencils : ℕ := 3
def blue_pencils : ℕ := 4

def total_pencils : ℕ := yellow_pencils + red_pencils + blue_pencils

def total_arrangements : ℕ := Nat.factorial total_pencils / (Nat.factorial yellow_pencils * Nat.factorial red_pencils * Nat.factorial blue_pencils)

def arrangements_with_adjacent_blue : ℕ := Nat.factorial (total_pencils - blue_pencils + 1) / (Nat.factorial yellow_pencils * Nat.factorial red_pencils)

theorem pencil_arrangement_theorem :
  total_arrangements - arrangements_with_adjacent_blue = 274400 := by
  sorry

end NUMINAMATH_CALUDE_pencil_arrangement_theorem_l2770_277022


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_powers_of_half_l2770_277003

theorem simplify_fraction_sum_powers_of_half :
  1 / (1 / ((1/2)^0) + 1 / ((1/2)^1) + 1 / ((1/2)^2) + 1 / ((1/2)^3)) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_powers_of_half_l2770_277003


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_range_l2770_277045

/-- Given a line y = kx + 3 intersecting a circle (x - 2)² + (y - 3)² = 4 at points M and N,
    if |MN| ≥ 2√3, then -√3/3 ≤ k ≤ √3/3 -/
theorem line_circle_intersection_k_range (k : ℝ) (M N : ℝ × ℝ) :
  (∀ x y, y = k * x + 3 → (x - 2)^2 + (y - 3)^2 = 4 → (x, y) = M ∨ (x, y) = N) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12 →
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_k_range_l2770_277045


namespace NUMINAMATH_CALUDE_problem_solution_l2770_277032

noncomputable section

def f (a : ℝ) (x : ℝ) := a * (3 : ℝ)^x + (3 : ℝ)^(-x)

def g (m : ℝ) (x : ℝ) := (Real.log x / Real.log 2)^2 + 2 * (Real.log x / Real.log 2) + m

theorem problem_solution :
  (∀ x, f a x = f a (-x)) →
  a = 1 ∧
  (∀ x y, 0 < x → x < y → f 1 x < f 1 y) ∧
  (∃ α β, α ≠ β ∧ 1/8 ≤ α ∧ α ≤ 4 ∧ 1/8 ≤ β ∧ β ≤ 4 ∧ g m α = 0 ∧ g m β = 0) →
  -3 ≤ m ∧ m < 1 ∧ α * β = 1/4 :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2770_277032


namespace NUMINAMATH_CALUDE_current_speed_l2770_277098

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 21)
  (h2 : speed_against_current = 16) :
  ∃ (man_speed current_speed : ℝ),
    man_speed + current_speed = speed_with_current ∧
    man_speed - current_speed = speed_against_current ∧
    current_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l2770_277098


namespace NUMINAMATH_CALUDE_balloon_arrangements_l2770_277085

def word_length : Nat := 7
def repeated_letters : Nat := 2
def repetitions_per_letter : Nat := 2

theorem balloon_arrangements :
  (word_length.factorial) / (repeated_letters.factorial * repetitions_per_letter.factorial) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l2770_277085


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l2770_277075

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Extractable in p-arithmetic -/
def extractable_in_p_arithmetic (x : ℝ) (p : ℕ) : Prop := sorry

theorem fibonacci_divisibility (p k : ℕ) (h_prime : Nat.Prime p) 
  (h_sqrt5 : extractable_in_p_arithmetic (Real.sqrt 5) p) :
  p^k ∣ fib (p^(k-1) * (p-1)) :=
sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l2770_277075


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2770_277083

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 + 4*x^2*Real.sqrt 3 + 12*x + 8*Real.sqrt 3) + (x + 2*Real.sqrt 3)
  ∃ (z₁ z₂ z₃ : ℂ),
    z₁ = -2 * Real.sqrt 3 ∧
    z₂ = -2 * Real.sqrt 3 + Complex.I ∧
    z₃ = -2 * Real.sqrt 3 - Complex.I ∧
    (∀ z : ℂ, f z = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2770_277083


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_9_starting_with_7_l2770_277048

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def starts_with_7 (n : ℕ) : Prop := ∃ m : ℕ, n = 70000 + m ∧ m < 30000

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∀ n : ℕ, is_five_digit n → starts_with_7 n → is_multiple_of_9 n → n ≥ 70002 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_9_starting_with_7_l2770_277048


namespace NUMINAMATH_CALUDE_binary_rep_156_ones_minus_zeros_eq_zero_l2770_277018

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Counts the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Counts the number of false values in a list of booleans -/
def countZeros (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_rep_156_ones_minus_zeros_eq_zero :
  let binary := toBinary 156
  let y := countOnes binary
  let x := countZeros binary
  y - x = 0 := by sorry

end NUMINAMATH_CALUDE_binary_rep_156_ones_minus_zeros_eq_zero_l2770_277018


namespace NUMINAMATH_CALUDE_intersection_area_l2770_277074

/-- The area of intersection between two boards of widths 5 inches and 7 inches,
    crossing at a 45-degree angle. -/
theorem intersection_area (board1_width board2_width : ℝ) (angle : ℝ) :
  board1_width = 5 →
  board2_width = 7 →
  angle = π / 4 →
  (board1_width * board2_width * Real.sin angle) = (35 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_area_l2770_277074


namespace NUMINAMATH_CALUDE_horner_method_operations_l2770_277049

def f (x : ℝ) : ℝ := x^6 + 1

def horner_eval (x : ℝ) : ℝ := ((((((x * x + 0) * x + 0) * x + 0) * x + 0) * x + 0) * x + 1)

theorem horner_method_operations (x : ℝ) :
  (∃ (exp_count mult_count add_count : ℕ),
    horner_eval x = f x ∧
    exp_count = 0 ∧
    mult_count = 6 ∧
    add_count = 6) :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2770_277049


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_l2770_277000

/-- Given a right triangle with legs of lengths 3 and 4, 
    the height on the hypotenuse is 12/5 -/
theorem height_on_hypotenuse (a b c h : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → h * c = 2 * (a * b / 2) → h = 12/5 := by sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_l2770_277000


namespace NUMINAMATH_CALUDE_sqrt_equation_difference_l2770_277006

theorem sqrt_equation_difference (a b : ℕ+) 
  (h1 : Real.sqrt 18 = (a : ℝ) * Real.sqrt 2) 
  (h2 : Real.sqrt 8 = 2 * Real.sqrt (b : ℝ)) : 
  (a : ℤ) - (b : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_difference_l2770_277006


namespace NUMINAMATH_CALUDE_possible_a3_values_l2770_277060

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Theorem: Possible values of a_3 in the arithmetic sequence -/
theorem possible_a3_values (seq : ArithmeticSequence) 
  (h_a5 : seq.a 5 = 6)
  (h_a3_gt_1 : seq.a 3 > 1)
  (h_geometric : ∃ (m : ℕ → ℕ), 
    (∀ t, 5 < m t ∧ (t > 0 → m (t-1) < m t)) ∧ 
    (∀ t, ∃ r, seq.a (m t) = seq.a 3 * r^(t+1) ∧ seq.a 5 = seq.a 3 * r^2)) :
  seq.a 3 = 3 ∨ seq.a 3 = 2 ∨ seq.a 3 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_possible_a3_values_l2770_277060


namespace NUMINAMATH_CALUDE_y_derivative_l2770_277029

open Real

noncomputable def y (x : ℝ) : ℝ := 2 * (cos x / sin x ^ 4) + 3 * (cos x / sin x ^ 2)

theorem y_derivative (x : ℝ) (h : sin x ≠ 0) : 
  deriv y x = 3 * (1 / sin x) - 8 * (1 / sin x) ^ 5 := by
sorry

end NUMINAMATH_CALUDE_y_derivative_l2770_277029


namespace NUMINAMATH_CALUDE_union_M_N_equals_reals_l2770_277005

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x^2 ≥ x}

-- State the theorem
theorem union_M_N_equals_reals : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_reals_l2770_277005


namespace NUMINAMATH_CALUDE_closest_point_l2770_277096

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 5*t
  | 1 => -2 + 4*t
  | 2 => 1 + 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -1
  | 1 => 1
  | 2 => -3

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 4
  | 2 => 2

theorem closest_point (t : ℝ) :
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -16/45 := by sorry

end NUMINAMATH_CALUDE_closest_point_l2770_277096


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2770_277054

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with households of different income levels -/
structure Community where
  totalHouseholds : Nat
  highIncomeHouseholds : Nat
  middleIncomeHouseholds : Nat
  lowIncomeHouseholds : Nat

/-- Represents a group of senior soccer players -/
structure SoccerTeam where
  totalPlayers : Nat

/-- Determines the best sampling method for a given community and sample size -/
def bestSamplingMethodForCommunity (c : Community) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Determines the best sampling method for a given soccer team and sample size -/
def bestSamplingMethodForSoccerTeam (t : SoccerTeam) (sampleSize : Nat) : SamplingMethod :=
  sorry

theorem correct_sampling_methods 
  (community : Community)
  (soccerTeam : SoccerTeam)
  (communitySampleSize : Nat)
  (soccerSampleSize : Nat)
  (h1 : community.totalHouseholds = 500)
  (h2 : community.highIncomeHouseholds = 125)
  (h3 : community.middleIncomeHouseholds = 280)
  (h4 : community.lowIncomeHouseholds = 95)
  (h5 : communitySampleSize = 100)
  (h6 : soccerTeam.totalPlayers = 12)
  (h7 : soccerSampleSize = 3) :
  bestSamplingMethodForCommunity community communitySampleSize = SamplingMethod.Stratified ∧
  bestSamplingMethodForSoccerTeam soccerTeam soccerSampleSize = SamplingMethod.Random :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2770_277054


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2770_277050

/-- The length of the major axis of an ellipse C, given specific conditions -/
theorem ellipse_major_axis_length : 
  ∀ (m : ℝ) (x y : ℝ → ℝ),
    (m > 0) →
    (∀ t, 2 * (x t) - (y t) + 4 = 0) →
    (∀ t, (x t)^2 / m + (y t)^2 / 2 = 1) →
    (∃ t₀, (x t₀, y t₀) = (-2, 0) ∨ (x t₀, y t₀) = (0, 4)) →
    ∃ a b : ℝ, a^2 = m ∧ b^2 = 2 ∧ 2 * max a b = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2770_277050


namespace NUMINAMATH_CALUDE_smallest_prime_eight_less_than_square_l2770_277057

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_eight_less_than_square : 
  (∀ n : ℕ, n > 0 ∧ is_prime n ∧ (∃ m : ℕ, n = m * m - 8) → n ≥ 17) ∧ 
  (17 > 0 ∧ is_prime 17 ∧ ∃ m : ℕ, 17 = m * m - 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_eight_less_than_square_l2770_277057


namespace NUMINAMATH_CALUDE_product_plus_one_equals_square_l2770_277056

theorem product_plus_one_equals_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_equals_square_l2770_277056


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2770_277040

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of real number m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m^2 + 3*m + 2) (m^2 - m - 6)

theorem pure_imaginary_condition (m : ℝ) :
  IsPureImaginary (z m) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2770_277040


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l2770_277012

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 5 * x + b * y + c * z = 0)
  (eq2 : a * x + 7 * y + c * z = 0)
  (eq3 : a * x + b * y + 9 * z = 0)
  (ha : a ≠ 5)
  (hx : x ≠ 0) :
  a / (a - 5) + b / (b - 7) + c / (c - 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l2770_277012


namespace NUMINAMATH_CALUDE_spectators_count_l2770_277009

/-- The number of wristbands distributed -/
def total_wristbands : ℕ := 250

/-- The number of wristbands each person received -/
def wristbands_per_person : ℕ := 2

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 125 := by
  sorry

end NUMINAMATH_CALUDE_spectators_count_l2770_277009


namespace NUMINAMATH_CALUDE_binary_representation_of_500_l2770_277086

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_representation_of_500 :
  to_binary 500 = [true, false, false, true, true, true, true, true, true] :=
by sorry

end NUMINAMATH_CALUDE_binary_representation_of_500_l2770_277086


namespace NUMINAMATH_CALUDE_jump_rope_solution_l2770_277072

/-- The cost of jump ropes A and B satisfy the given conditions -/
def jump_rope_cost (cost_A cost_B : ℝ) : Prop :=
  10 * cost_A + 5 * cost_B = 175 ∧ 15 * cost_A + 10 * cost_B = 300

/-- The solution to the jump rope cost problem -/
theorem jump_rope_solution :
  ∃ (cost_A cost_B : ℝ), jump_rope_cost cost_A cost_B ∧ cost_A = 10 ∧ cost_B = 15 := by
  sorry

#check jump_rope_solution

end NUMINAMATH_CALUDE_jump_rope_solution_l2770_277072


namespace NUMINAMATH_CALUDE_business_card_exchanges_count_l2770_277095

/-- Represents a business conference with two groups of people -/
structure BusinessConference where
  total_people : ℕ
  group1_size : ℕ
  group2_size : ℕ
  h_total : total_people = group1_size + group2_size
  h_group1 : group1_size = 25
  h_group2 : group2_size = 15

/-- Calculates the number of business card exchanges in a business conference -/
def business_card_exchanges (conf : BusinessConference) : ℕ :=
  conf.group1_size * conf.group2_size

/-- Theorem stating that the number of business card exchanges is 375 -/
theorem business_card_exchanges_count (conf : BusinessConference) :
  business_card_exchanges conf = 375 := by
  sorry

#eval business_card_exchanges ⟨40, 25, 15, rfl, rfl, rfl⟩

end NUMINAMATH_CALUDE_business_card_exchanges_count_l2770_277095


namespace NUMINAMATH_CALUDE_c1_c2_not_collinear_l2770_277038

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem c1_c2_not_collinear (a b : ℝ × ℝ × ℝ) 
  (h1 : a = ⟨-9, 5, 3⟩) 
  (h2 : b = ⟨7, 1, -2⟩) : 
  ¬ ∃ (k : ℝ), 2 • a - b = k • (3 • a + 5 • b) :=
sorry

end NUMINAMATH_CALUDE_c1_c2_not_collinear_l2770_277038


namespace NUMINAMATH_CALUDE_thousandth_coprime_to_105_l2770_277053

/-- The sequence of positive integers coprime to 105, arranged in ascending order -/
def coprimeSeq : ℕ → ℕ := sorry

/-- The 1000th term of the sequence is 2186 -/
theorem thousandth_coprime_to_105 : coprimeSeq 1000 = 2186 := by sorry

end NUMINAMATH_CALUDE_thousandth_coprime_to_105_l2770_277053


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2770_277020

theorem triangle_abc_properties (a b : ℝ) (A B C : ℝ) :
  -- Conditions
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) →
  (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = 1) →
  -- Conclusions
  (C = 120 * π / 180) ∧
  (Real.sqrt ((a^2 + b^2 + a*b) : ℝ) = Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2770_277020


namespace NUMINAMATH_CALUDE_exactly_two_valid_multiplications_l2770_277052

def is_valid_multiplication (a b : ℕ) : Prop :=
  100 ≤ a ∧ a < 1000 ∧  -- a is a three-digit number
  a / 100 = 1 ∧  -- a starts with 1
  1 ≤ b ∧ b < 10 ∧  -- b is a single-digit number
  1000 ≤ a * b ∧ a * b < 10000 ∧  -- product is four digits
  (a * (b % 10) / 100 = 1)  -- third row starts with '100'

theorem exactly_two_valid_multiplications :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ a ∈ s, ∃ b, is_valid_multiplication a b :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_multiplications_l2770_277052


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2770_277001

theorem right_triangle_side_length 
  (A B C : Real) 
  (BC : Real) 
  (h1 : A = Real.pi / 2) 
  (h2 : BC = 10) 
  (h3 : Real.tan C = 3 * Real.cos B) : 
  ∃ AB : Real, AB = 20 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2770_277001


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2770_277069

theorem no_integer_solutions (m s : ℤ) (h : m * s = 2000^2001) :
  ¬∃ (x y : ℤ), m * x^2 - s * y^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2770_277069


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l2770_277042

theorem equidistant_point_on_y_axis : 
  ∃ y : ℚ, y = 13/6 ∧ 
  (∀ (x : ℚ), x = 0 → 
    (x^2 + y^2) = ((x - 2)^2 + (y - 3)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l2770_277042


namespace NUMINAMATH_CALUDE_piglet_gave_two_balloons_l2770_277026

/-- The number of balloons Piglet eventually gave to Eeyore -/
def piglet_balloons : ℕ := 2

/-- The number of balloons Winnie-the-Pooh prepared -/
def pooh_balloons (n : ℕ) : ℕ := 2 * n

/-- The number of balloons Owl prepared -/
def owl_balloons (n : ℕ) : ℕ := 4 * n

/-- The total number of balloons Eeyore received -/
def total_balloons : ℕ := 44

/-- Theorem stating that Piglet gave 2 balloons to Eeyore -/
theorem piglet_gave_two_balloons :
  ∃ (n : ℕ), 
    piglet_balloons + pooh_balloons n + owl_balloons n = total_balloons ∧
    n > piglet_balloons ∧
    piglet_balloons = 2 := by
  sorry


end NUMINAMATH_CALUDE_piglet_gave_two_balloons_l2770_277026


namespace NUMINAMATH_CALUDE_toaster_sales_l2770_277058

/-- Represents the inverse proportionality between number of customers and toaster cost -/
def inverse_proportional (customers : ℕ) (cost : ℝ) (k : ℝ) : Prop :=
  (customers : ℝ) * cost = k

/-- Proves that if 12 customers buy a $500 toaster, then 8 customers will buy a $750 toaster,
    given the inverse proportionality relationship -/
theorem toaster_sales (k : ℝ) :
  inverse_proportional 12 500 k →
  inverse_proportional 8 750 k :=
by
  sorry

end NUMINAMATH_CALUDE_toaster_sales_l2770_277058


namespace NUMINAMATH_CALUDE_mork_and_mindy_tax_rate_l2770_277068

/-- Calculates the combined tax rate for Mork and Mindy given their individual tax rates and income ratio. -/
theorem mork_and_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.1) 
  (h2 : mindy_rate = 0.2) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.175 := by
sorry

#eval (0.1 + 0.2 * 3) / (1 + 3)

end NUMINAMATH_CALUDE_mork_and_mindy_tax_rate_l2770_277068


namespace NUMINAMATH_CALUDE_mirror_area_l2770_277033

/-- The area of a rectangular mirror fitting exactly inside a frame -/
theorem mirror_area (frame_length frame_width frame_thickness : ℕ) 
  (h1 : frame_length = 100)
  (h2 : frame_width = 130)
  (h3 : frame_thickness = 15) : 
  (frame_length - 2 * frame_thickness) * (frame_width - 2 * frame_thickness) = 7000 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l2770_277033


namespace NUMINAMATH_CALUDE_transformation_matrix_correct_l2770_277030

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_factor : ℝ := 2

theorem transformation_matrix_correct :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]
  ∀ (v : Fin 2 → ℝ),
    M.mulVec v = scaling_factor • (rotation_matrix.mulVec v) :=
by sorry

end NUMINAMATH_CALUDE_transformation_matrix_correct_l2770_277030


namespace NUMINAMATH_CALUDE_square_difference_plus_fifty_l2770_277084

theorem square_difference_plus_fifty : (312^2 - 288^2) / 24 + 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_fifty_l2770_277084


namespace NUMINAMATH_CALUDE_happy_street_traffic_happy_street_traffic_proof_l2770_277024

theorem happy_street_traffic (tuesday : ℕ) (thursday friday weekend_day : ℕ) 
  (total : ℕ) : ℕ :=
  let monday := tuesday - tuesday / 5
  let thursday_to_sunday := thursday + friday + 2 * weekend_day
  let monday_to_wednesday := total - thursday_to_sunday
  let wednesday := monday_to_wednesday - (monday + tuesday)
  wednesday - monday

#check happy_street_traffic 25 10 10 5 97 = 2

theorem happy_street_traffic_proof : 
  happy_street_traffic 25 10 10 5 97 = 2 := by
sorry

end NUMINAMATH_CALUDE_happy_street_traffic_happy_street_traffic_proof_l2770_277024


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_is_zero_l2770_277066

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 - 3*z = 8 - 6*i

-- Theorem statement
theorem sum_of_imaginary_parts_is_zero :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧ 
  z₁ ≠ z₂ ∧ (z₁.im + z₂.im = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_is_zero_l2770_277066


namespace NUMINAMATH_CALUDE_min_value_implies_a_equals_9_l2770_277087

theorem min_value_implies_a_equals_9 (t a : ℝ) (h1 : 0 < t) (h2 : t < π / 2) (h3 : a > 0) :
  (∀ s, 0 < s ∧ s < π / 2 → (1 / Real.cos s + a / (1 - Real.cos s)) ≥ 16) ∧
  (∃ s, 0 < s ∧ s < π / 2 ∧ 1 / Real.cos s + a / (1 - Real.cos s) = 16) →
  a = 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_equals_9_l2770_277087


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l2770_277062

theorem unique_root_quadratic (k : ℝ) :
  (∃! x : ℝ, k * x^2 - 3 * x + 2 = 0) → k = 0 ∨ k = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l2770_277062


namespace NUMINAMATH_CALUDE_sum_of_twenty_and_ten_l2770_277046

theorem sum_of_twenty_and_ten : 20 + 10 = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_twenty_and_ten_l2770_277046


namespace NUMINAMATH_CALUDE_brinley_zoo_count_l2770_277094

/-- The number of animals Brinley counted at the San Diego Zoo --/
def total_animals (snakes arctic_foxes leopards bee_eaters cheetahs alligators : ℕ) : ℕ :=
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

/-- Theorem stating the total number of animals Brinley counted at the zoo --/
theorem brinley_zoo_count : ∃ (snakes arctic_foxes leopards bee_eaters cheetahs alligators : ℕ),
  snakes = 100 ∧
  arctic_foxes = 80 ∧
  leopards = 20 ∧
  bee_eaters = 10 * leopards ∧
  cheetahs = snakes / 2 ∧
  alligators = 2 * (arctic_foxes + leopards) ∧
  total_animals snakes arctic_foxes leopards bee_eaters cheetahs alligators = 650 :=
by
  sorry


end NUMINAMATH_CALUDE_brinley_zoo_count_l2770_277094


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2770_277047

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1/3) :
  Real.cos (α - π/4)^2 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2770_277047


namespace NUMINAMATH_CALUDE_parallelogram_bisector_l2770_277002

-- Define the parallelogram
def parallelogram : List (ℝ × ℝ) := [(5, 25), (5, 50), (14, 58), (14, 33)]

-- Define the property of the line
def divides_equally (m n : ℕ) : Prop :=
  let slope := m / n
  ∃ (b : ℝ), 
    (25 + b) / 5 = (58 - b) / 14 ∧ 
    (25 + b) / 5 = slope ∧
    (b > -25 ∧ b < 33)  -- Ensure the line intersects the parallelogram

-- Main theorem
theorem parallelogram_bisector :
  ∃ (m n : ℕ), 
    m.Coprime n ∧
    divides_equally m n ∧
    m = 71 ∧ n = 19 ∧
    m + n = 90 := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_l2770_277002


namespace NUMINAMATH_CALUDE_no_sum_of_three_different_squares_128_l2770_277007

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_three_different_squares (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    is_perfect_square a ∧ 
    is_perfect_square b ∧ 
    is_perfect_square c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = n

theorem no_sum_of_three_different_squares_128 : 
  ¬(sum_of_three_different_squares 128) := by
  sorry

end NUMINAMATH_CALUDE_no_sum_of_three_different_squares_128_l2770_277007


namespace NUMINAMATH_CALUDE_roots_of_equation_l2770_277017

theorem roots_of_equation (x : ℝ) : 
  (x + 2)^2 = 8 ↔ x = 2 * Real.sqrt 2 - 2 ∨ x = -2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2770_277017
