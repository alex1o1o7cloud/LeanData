import Mathlib

namespace NUMINAMATH_CALUDE_dara_waiting_time_l3307_330718

/-- Represents the company's employment requirements and employee information --/
structure CompanyData where
  initial_min_age : ℕ
  age_increase_rate : ℕ
  age_increase_period : ℕ
  jane_age : ℕ
  tom_age_diff : ℕ
  tom_join_min_age : ℕ
  dara_internship_age : ℕ
  dara_internship_duration : ℕ
  dara_training_age : ℕ
  dara_training_duration : ℕ

/-- Calculates the waiting time for Dara to be eligible for employment --/
def calculate_waiting_time (data : CompanyData) : ℕ :=
  sorry

/-- Theorem stating that Dara has to wait 19 years before she can be employed --/
theorem dara_waiting_time (data : CompanyData) :
  data.initial_min_age = 25 ∧
  data.age_increase_rate = 1 ∧
  data.age_increase_period = 5 ∧
  data.jane_age = 28 ∧
  data.tom_age_diff = 10 ∧
  data.tom_join_min_age = 24 ∧
  data.dara_internship_age = 22 ∧
  data.dara_internship_duration = 3 ∧
  data.dara_training_age = 24 ∧
  data.dara_training_duration = 2 →
  calculate_waiting_time data = 19 :=
by sorry

end NUMINAMATH_CALUDE_dara_waiting_time_l3307_330718


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3307_330799

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_theorem : 
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3307_330799


namespace NUMINAMATH_CALUDE_min_value_is_four_l3307_330785

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  d : ℚ
  hd : d ≠ 0
  ha1 : a 1 = 1
  hGeometric : (a 3) ^ 2 = (a 1) * (a 13)
  hArithmetic : ∀ n : ℕ+, a n = a 1 + (n - 1) * d

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The expression to be minimized -/
def f (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (2 * S seq n + 16) / (seq.a n + 3)

/-- Theorem stating the minimum value of the expression -/
theorem min_value_is_four (seq : ArithmeticSequence) :
  ∃ n₀ : ℕ+, ∀ n : ℕ+, f seq n ≥ f seq n₀ ∧ f seq n₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_four_l3307_330785


namespace NUMINAMATH_CALUDE_cylinder_reciprocal_sum_l3307_330796

theorem cylinder_reciprocal_sum (r h : ℝ) (volume_eq : π * r^2 * h = 2) (surface_area_eq : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_reciprocal_sum_l3307_330796


namespace NUMINAMATH_CALUDE_area_of_specific_region_l3307_330784

/-- The area of a specific region in a circle with an inscribed regular hexagon -/
theorem area_of_specific_region (r : ℝ) (s : ℝ) (h_r : r = 3) (h_s : s = 2) :
  let circle_area := π * r^2
  let hexagon_side := s
  let sector_angle := 120
  let sector_area := (sector_angle / 360) * circle_area
  let triangle_area := (1/2) * r^2 * Real.sin (sector_angle * π / 180)
  sector_area - triangle_area = 3 * π - (9 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_region_l3307_330784


namespace NUMINAMATH_CALUDE_painted_cubes_count_l3307_330771

/-- Represents a cube with given dimensions -/
structure Cube where
  size : Nat

/-- Represents a painted cube -/
structure PaintedCube extends Cube where
  painted : Bool

/-- Calculates the number of 1-inch cubes with at least one painted face -/
def paintedCubes (c : PaintedCube) : Nat :=
  c.size ^ 3 - (c.size - 2) ^ 3

/-- Theorem: In a 10×10×10 painted cube, 488 small cubes have at least one painted face -/
theorem painted_cubes_count :
  let c : PaintedCube := { size := 10, painted := true }
  paintedCubes c = 488 := by sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l3307_330771


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3307_330742

theorem inequality_system_solution_set :
  {x : ℝ | -2*x ≤ 6 ∧ x + 1 < 0} = {x : ℝ | -3 ≤ x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3307_330742


namespace NUMINAMATH_CALUDE_tim_score_theorem_l3307_330795

/-- Sum of the first n even numbers -/
def sumFirstNEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- A number is recognizable if it's 90 (for this specific problem) -/
def isRecognizable (x : ℕ) : Prop := x = 90

/-- A number is a square number if it's the square of some integer -/
def isSquareNumber (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem tim_score_theorem :
  ∃ n : ℕ, isSquareNumber n ∧ isRecognizable (sumFirstNEvenNumbers n) ∧
  ∀ m : ℕ, m < n → ¬(isSquareNumber m ∧ isRecognizable (sumFirstNEvenNumbers m)) :=
by sorry

end NUMINAMATH_CALUDE_tim_score_theorem_l3307_330795


namespace NUMINAMATH_CALUDE_matchbox_cars_percentage_l3307_330754

theorem matchbox_cars_percentage (total : ℕ) (truck_percent : ℚ) (convertibles : ℕ) : 
  total = 125 →
  truck_percent = 8 / 100 →
  convertibles = 35 →
  (((total : ℚ) - (truck_percent * total) - (convertibles : ℚ)) / total) * 100 = 64 := by
sorry

end NUMINAMATH_CALUDE_matchbox_cars_percentage_l3307_330754


namespace NUMINAMATH_CALUDE_birds_on_fence_l3307_330755

/-- Given that there are initially 12 birds on a fence and after more birds land
    there are a total of 20 birds, prove that 8 birds landed on the fence. -/
theorem birds_on_fence (initial_birds : ℕ) (total_birds : ℕ) (h1 : initial_birds = 12) (h2 : total_birds = 20) :
  total_birds - initial_birds = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3307_330755


namespace NUMINAMATH_CALUDE_f_not_monotonic_range_l3307_330783

/-- The function f(x) = x³ - 12x -/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

/-- A function is not monotonic on an interval if its derivative has a zero in that interval -/
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f' x = 0

/-- The theorem stating the range of k for which f is not monotonic on (k, k+2) -/
theorem f_not_monotonic_range :
  ∀ k : ℝ, not_monotonic f k (k+2) ↔ (k > -4 ∧ k < -2) ∨ (k > 0 ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_range_l3307_330783


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3307_330791

theorem min_value_quadratic (a b : ℝ) :
  2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044 ≥ 1976 ∧
  (2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044 = 1976 ↔ a = 8 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3307_330791


namespace NUMINAMATH_CALUDE_distribute_five_four_l3307_330706

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of partitions of n into at most k parts -/
def partitions (n k : ℕ) : ℕ := sorry

theorem distribute_five_four : distribute 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_five_four_l3307_330706


namespace NUMINAMATH_CALUDE_museum_ticket_cost_class_trip_cost_l3307_330738

/-- Calculates the total cost of museum tickets for a class, including a group discount -/
theorem museum_ticket_cost (num_students num_teachers : ℕ) 
  (student_price teacher_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let regular_cost := num_students * student_price + num_teachers * teacher_price
  let discount := if total_people ≥ 25 then discount_rate * regular_cost else 0
  regular_cost - discount

/-- Proves that the total cost for the class trip is $230.40 -/
theorem class_trip_cost : 
  museum_ticket_cost 30 4 8 12 (20/100) = 230.4 := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_class_trip_cost_l3307_330738


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3307_330710

theorem polynomial_factorization (x y : ℝ) : 
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2)*(x^2 - 3*y + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3307_330710


namespace NUMINAMATH_CALUDE_exists_prime_pair_solution_l3307_330762

/-- A pair of prime numbers (p, q) is a solution if the quadratic equation
    px^2 - qx + p = 0 has rational roots. -/
def is_solution (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧
  ∃ (x y : ℚ), p * x^2 - q * x + p = 0 ∧ p * y^2 - q * y + p = 0 ∧ x ≠ y

/-- There exists a pair of prime numbers (p, q) that is a solution. -/
theorem exists_prime_pair_solution : ∃ (p q : ℕ), is_solution p q :=
sorry

end NUMINAMATH_CALUDE_exists_prime_pair_solution_l3307_330762


namespace NUMINAMATH_CALUDE_joan_eggs_count_l3307_330737

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- Theorem: Joan bought 72 eggs -/
theorem joan_eggs_count : dozens_bought * eggs_per_dozen = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_eggs_count_l3307_330737


namespace NUMINAMATH_CALUDE_range_of_difference_l3307_330713

theorem range_of_difference (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x = x^2 - 2*x) →
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a b, f x = y) →
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_difference_l3307_330713


namespace NUMINAMATH_CALUDE_horner_method_for_f_at_3_l3307_330793

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + x^2 + x + 1

-- Theorem statement
theorem horner_method_for_f_at_3 : f 3 = 283 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_at_3_l3307_330793


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3307_330736

def M : Set ℝ := {x | (x + 1) * (x - 3) ≤ 0}
def N : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3307_330736


namespace NUMINAMATH_CALUDE_sum_digits_first_1998_even_l3307_330704

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits used to write all even integers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ := sorry

/-- The 1998th positive even integer -/
def n_1998 : ℕ := 3996

theorem sum_digits_first_1998_even : sum_digits_even n_1998 = 7440 := by sorry

end NUMINAMATH_CALUDE_sum_digits_first_1998_even_l3307_330704


namespace NUMINAMATH_CALUDE_factorial_products_squares_l3307_330708

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (fun i => i + 1)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem factorial_products_squares :
  (is_perfect_square (factorial 7 * factorial 8)) ∧
  (¬ is_perfect_square (factorial 5 * factorial 6)) ∧
  (¬ is_perfect_square (factorial 5 * factorial 7)) ∧
  (¬ is_perfect_square (factorial 6 * factorial 7)) ∧
  (¬ is_perfect_square (factorial 6 * factorial 8)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_products_squares_l3307_330708


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_153_l3307_330752

theorem largest_of_three_consecutive_integers_sum_153 :
  ∀ (x y z : ℤ), 
    (y = x + 1) → 
    (z = y + 1) → 
    (x + y + z = 153) → 
    (max x (max y z) = 52) :=
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_153_l3307_330752


namespace NUMINAMATH_CALUDE_prism_volume_l3307_330734

-- Define a right rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the volume of a rectangular prism
def volume (p : RectangularPrism) : ℝ := p.length * p.width * p.height

-- Define the areas of the faces
def faceArea1 (p : RectangularPrism) : ℝ := p.length * p.width
def faceArea2 (p : RectangularPrism) : ℝ := p.width * p.height
def faceArea3 (p : RectangularPrism) : ℝ := p.length * p.height

-- State the theorem
theorem prism_volume (p : RectangularPrism)
  (h1 : faceArea1 p = 60)
  (h2 : faceArea2 p = 72)
  (h3 : faceArea3 p = 90) :
  volume p = 4320 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3307_330734


namespace NUMINAMATH_CALUDE_product_1_to_30_trailing_zeros_l3307_330724

/-- The number of trailing zeros in the product of integers from 1 to n -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The product of integers from 1 to 30 has 7 trailing zeros -/
theorem product_1_to_30_trailing_zeros :
  trailingZeros 30 = 7 := by
sorry


end NUMINAMATH_CALUDE_product_1_to_30_trailing_zeros_l3307_330724


namespace NUMINAMATH_CALUDE_min_real_roots_l3307_330727

/-- A polynomial of degree 10 with real coefficients -/
structure Polynomial10 where
  coeffs : Fin 11 → ℝ
  lead_coeff_nonzero : coeffs 10 ≠ 0

/-- The roots of a polynomial -/
def roots (p : Polynomial10) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinct_abs_values (p : Polynomial10) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def num_real_roots (p : Polynomial10) : ℕ := sorry

/-- If a polynomial of degree 10 with real coefficients has exactly 6 distinct absolute values
    among its roots, then it has at least 3 real roots -/
theorem min_real_roots (p : Polynomial10) :
  distinct_abs_values p = 6 → num_real_roots p ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_real_roots_l3307_330727


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3307_330728

/-- Given the definitions of p, q, r, and s, prove that (1/p + 1/q + 1/r + 1/s)² = 560/151321 -/
theorem sum_of_reciprocals_squared (p q r s : ℝ) 
  (hp : p = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35)
  (hq : q = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35)
  (hr : r = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35)
  (hs : s = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35) :
  (1/p + 1/q + 1/r + 1/s)^2 = 560/151321 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3307_330728


namespace NUMINAMATH_CALUDE_consecutive_sequence_unique_l3307_330775

/-- Three consecutive natural numbers forming an arithmetic and geometric sequence -/
def ConsecutiveSequence (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧
  (b + 2)^2 = (a + 1) * (c + 5)

theorem consecutive_sequence_unique :
  ∀ a b c : ℕ, ConsecutiveSequence a b c → a = 1 ∧ b = 2 ∧ c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_sequence_unique_l3307_330775


namespace NUMINAMATH_CALUDE_fraction_invariance_l3307_330702

theorem fraction_invariance (x y : ℝ) (h : x ≠ y) : 
  (3 * x) / (3 * x - 3 * y) = x / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l3307_330702


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3307_330774

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8) (h2 : x - y = 3/8) : x^2 - y^2 = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3307_330774


namespace NUMINAMATH_CALUDE_sum_areas_halving_circles_l3307_330703

/-- The sum of areas of an infinite series of circles with halving radii -/
theorem sum_areas_halving_circles (π : ℝ) (h : π > 0) : 
  let r₀ : ℝ := 2  -- radius of the first circle
  let seriesSum : ℝ := ∑' n, π * (r₀ * (1/2)^n)^2  -- sum of areas
  seriesSum = 16 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_areas_halving_circles_l3307_330703


namespace NUMINAMATH_CALUDE_cat_toy_cost_l3307_330761

theorem cat_toy_cost (total_payment change cage_cost : ℚ) 
  (h1 : total_payment = 20)
  (h2 : change = 0.26)
  (h3 : cage_cost = 10.97) :
  total_payment - change - cage_cost = 8.77 := by
  sorry

end NUMINAMATH_CALUDE_cat_toy_cost_l3307_330761


namespace NUMINAMATH_CALUDE_second_train_length_correct_l3307_330746

/-- The length of the second train given the conditions of the problem -/
def second_train_length : ℝ := 119.98240140788738

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 42

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 30

/-- The length of the first train in meters -/
def first_train_length : ℝ := 100

/-- The time taken for the trains to clear each other in seconds -/
def clearing_time : ℝ := 10.999120070394369

/-- Theorem stating that the calculated length of the second train is correct given the problem conditions -/
theorem second_train_length_correct :
  second_train_length = 
    (first_train_speed + second_train_speed) * (1000 / 3600) * clearing_time - first_train_length :=
by
  sorry


end NUMINAMATH_CALUDE_second_train_length_correct_l3307_330746


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3307_330760

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) :
  (2 * x₁^2 + 5 * x₁ - 12 = 0) →
  (2 * x₂^2 + 5 * x₂ - 12 = 0) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 73/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3307_330760


namespace NUMINAMATH_CALUDE_orange_bags_weight_l3307_330726

/-- If 12 bags of oranges weigh 24 pounds, then 8 bags of oranges weigh 16 pounds. -/
theorem orange_bags_weight (total_weight : ℝ) (total_bags : ℕ) (target_bags : ℕ) :
  total_weight = 24 ∧ total_bags = 12 ∧ target_bags = 8 →
  (target_bags : ℝ) * (total_weight / total_bags) = 16 :=
by sorry

end NUMINAMATH_CALUDE_orange_bags_weight_l3307_330726


namespace NUMINAMATH_CALUDE_alyssa_cherries_cost_l3307_330744

/-- The amount Alyssa paid for cherries -/
def cherries_cost (total_spent grapes_cost : ℚ) : ℚ :=
  total_spent - grapes_cost

/-- Proof that Alyssa paid $9.85 for cherries -/
theorem alyssa_cherries_cost :
  let total_spent : ℚ := 21.93
  let grapes_cost : ℚ := 12.08
  cherries_cost total_spent grapes_cost = 9.85 := by
  sorry

#eval cherries_cost 21.93 12.08

end NUMINAMATH_CALUDE_alyssa_cherries_cost_l3307_330744


namespace NUMINAMATH_CALUDE_quadratic_equation_a_value_l3307_330790

theorem quadratic_equation_a_value (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c →
    (x = -2 ∧ y = -3) ∨ (x = 1 ∧ y = 0)) →
  (∀ x y : ℝ, y = a * (x + 2)^2 - 3) →
  a = 1/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_a_value_l3307_330790


namespace NUMINAMATH_CALUDE_perfect_square_consecutive_integers_l3307_330759

theorem perfect_square_consecutive_integers (n : ℤ) : 
  (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_consecutive_integers_l3307_330759


namespace NUMINAMATH_CALUDE_triangle_area_l3307_330709

/-- Given a triangle with perimeter 20 and inradius 3, prove its area is 30 -/
theorem triangle_area (T : Set ℝ) (perimeter inradius : ℝ) : 
  perimeter = 20 →
  inradius = 3 →
  (∃ (area : ℝ), area = inradius * (perimeter / 2) ∧ area = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3307_330709


namespace NUMINAMATH_CALUDE_greatest_equal_distribution_l3307_330716

theorem greatest_equal_distribution (a b c : ℕ) (ha : a = 1050) (hb : b = 1260) (hc : c = 210) :
  Nat.gcd a (Nat.gcd b c) = 210 := by
  sorry

end NUMINAMATH_CALUDE_greatest_equal_distribution_l3307_330716


namespace NUMINAMATH_CALUDE_sqrt_10_between_3_and_4_l3307_330756

theorem sqrt_10_between_3_and_4 : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_between_3_and_4_l3307_330756


namespace NUMINAMATH_CALUDE_scientific_notation_103000000_l3307_330714

theorem scientific_notation_103000000 : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 103000000 = a * (10 : ℝ) ^ n ∧ a = 1.03 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_103000000_l3307_330714


namespace NUMINAMATH_CALUDE_work_completion_time_l3307_330701

/-- Represents the number of days it takes for a worker to complete the work alone -/
structure Worker where
  days : ℝ

/-- Represents the work scenario -/
structure WorkScenario where
  a : Worker
  b : Worker
  c : Worker
  cLeaveDays : ℝ

/-- Calculates the time taken to complete the work given a work scenario -/
def completionTime (scenario : WorkScenario) : ℝ :=
  sorry

/-- The specific work scenario from the problem -/
def problemScenario : WorkScenario :=
  { a := ⟨30⟩
  , b := ⟨30⟩
  , c := ⟨40⟩
  , cLeaveDays := 4 }

/-- Theorem stating that the work is completed in approximately 15 days -/
theorem work_completion_time :
  ⌈completionTime problemScenario⌉ = 15 :=
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3307_330701


namespace NUMINAMATH_CALUDE_m_minus_n_equals_l3307_330725

def M : Set Nat := {1, 3, 5, 7, 9}
def N : Set Nat := {2, 3, 5}

def setDifference (A B : Set Nat) : Set Nat :=
  {x | x ∈ A ∧ x ∉ B}

theorem m_minus_n_equals : setDifference M N = {1, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_l3307_330725


namespace NUMINAMATH_CALUDE_solution_set_of_trigonometric_system_l3307_330740

theorem solution_set_of_trigonometric_system :
  let S := {(x, y) | 
    2 * (Real.cos x)^2 + 2 * Real.sqrt 2 * Real.cos x * (Real.cos (4*x))^2 + (Real.cos (4*x))^2 = 0 ∧
    Real.sin x = Real.cos y}
  S = {(x, y) | 
    (∃ k n : ℤ, x = 3 * Real.pi / 4 + 2 * Real.pi * ↑k ∧ (y = Real.pi / 4 + 2 * Real.pi * ↑n ∨ y = -Real.pi / 4 + 2 * Real.pi * ↑n)) ∨
    (∃ k n : ℤ, x = -3 * Real.pi / 4 + 2 * Real.pi * ↑k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * ↑n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * ↑n))} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_trigonometric_system_l3307_330740


namespace NUMINAMATH_CALUDE_ben_pea_picking_time_l3307_330745

/-- Given Ben's rate of picking sugar snap peas, calculate the time needed to pick a different amount -/
theorem ben_pea_picking_time (initial_peas initial_time target_peas : ℕ) : 
  initial_peas > 0 → initial_time > 0 → target_peas > 0 →
  (target_peas * initial_time) / initial_peas = 9 :=
by
  sorry

#check ben_pea_picking_time 56 7 72

end NUMINAMATH_CALUDE_ben_pea_picking_time_l3307_330745


namespace NUMINAMATH_CALUDE_symmetric_lines_l3307_330797

/-- Given two lines L and K symmetric to each other with respect to y=x,
    where L has equation y = ax + b (a ≠ 0, b ≠ 0),
    prove that K has equation y = (1/a)x - (b/a) -/
theorem symmetric_lines (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let L : ℝ → ℝ := fun x => a * x + b
  let K : ℝ → ℝ := fun x => (1 / a) * x - (b / a)
  (∀ x y, y = L x ↔ x = L y) →
  (∀ x, K x = (1 / a) * x - (b / a)) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_l3307_330797


namespace NUMINAMATH_CALUDE_casper_enter_exit_ways_l3307_330711

/-- The number of windows in the castle -/
def num_windows : ℕ := 8

/-- The number of ways Casper can enter and exit the castle -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that the number of ways Casper can enter and exit is 56 -/
theorem casper_enter_exit_ways : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_casper_enter_exit_ways_l3307_330711


namespace NUMINAMATH_CALUDE_factorization_theorem_l3307_330788

theorem factorization_theorem (a : ℝ) : 4 * a^2 - 4 = 4 * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3307_330788


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_60_45045_l3307_330792

theorem gcd_lcm_sum_60_45045 : Nat.gcd 60 45045 + Nat.lcm 60 45045 = 180195 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_60_45045_l3307_330792


namespace NUMINAMATH_CALUDE_number_of_divisors_of_45_l3307_330781

theorem number_of_divisors_of_45 : Nat.card {d : ℕ | d ∣ 45} = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_45_l3307_330781


namespace NUMINAMATH_CALUDE_dog_catches_rabbit_l3307_330769

/-- Proves that a dog chasing a rabbit catches up in 4 minutes under given conditions -/
theorem dog_catches_rabbit (dog_speed rabbit_speed : ℝ) (head_start : ℝ) :
  dog_speed = 24 ∧ rabbit_speed = 15 ∧ head_start = 0.6 →
  (head_start / (dog_speed - rabbit_speed)) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_rabbit_l3307_330769


namespace NUMINAMATH_CALUDE_divisibility_of_repeating_digits_l3307_330719

theorem divisibility_of_repeating_digits : ∃ (k m : ℕ), k > 0 ∧ (1989 * (10^(4*k) - 1) / 9) * 10^m % 1988 = 0 ∧
                                          ∃ (n : ℕ), n > 0 ∧ (1988 * (10^(4*n) - 1) / 9) % 1989 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_repeating_digits_l3307_330719


namespace NUMINAMATH_CALUDE_oyster_consumption_l3307_330765

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- The number of oysters Crabby eats -/
def crabby_oysters : ℕ := 2 * squido_oysters

/-- The total number of oysters eaten by Crabby and Squido -/
def total_oysters : ℕ := squido_oysters + crabby_oysters

theorem oyster_consumption :
  total_oysters = 600 :=
sorry

end NUMINAMATH_CALUDE_oyster_consumption_l3307_330765


namespace NUMINAMATH_CALUDE_oliver_bath_water_usage_l3307_330780

/-- Calculates the weekly water usage for baths given the bucket capacity, 
    number of buckets to fill the tub, number of buckets removed, and days per week -/
def weekly_water_usage (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (buckets_removed : ℕ) (days_per_week : ℕ) : ℕ :=
  (buckets_to_fill - buckets_removed) * bucket_capacity * days_per_week

/-- Theorem stating that given the specific conditions, the weekly water usage is 9240 ounces -/
theorem oliver_bath_water_usage :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_oliver_bath_water_usage_l3307_330780


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l3307_330741

theorem no_linear_term_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ∃ b c d : ℝ, (x^2 + a*x - 2)*(x - 1) = x^3 + b*x^2 + d) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l3307_330741


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3307_330720

/-- Given a triangle ABC with sides a = 7, b = 5, and c = 3, 
    the measure of angle A is 120 degrees. -/
theorem triangle_angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) :
  let a := ‖B - C‖
  let b := ‖A - C‖
  let c := ‖A - B‖
  a = 7 ∧ b = 5 ∧ c = 3 →
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) * (180 / Real.pi) = 120 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_measure_l3307_330720


namespace NUMINAMATH_CALUDE_total_goals_after_five_matches_l3307_330758

/-- A football player's goal scoring record -/
structure FootballPlayer where
  goals_before_fifth : ℕ  -- Total goals before the fifth match
  matches_before_fifth : ℕ -- Number of matches before the fifth match (should be 4)

/-- The problem statement -/
theorem total_goals_after_five_matches (player : FootballPlayer) 
  (h1 : player.matches_before_fifth = 4)
  (h2 : (player.goals_before_fifth : ℚ) / 4 + 0.2 = 
        ((player.goals_before_fifth + 4) : ℚ) / 5) : 
  player.goals_before_fifth + 4 = 16 := by
  sorry

#check total_goals_after_five_matches

end NUMINAMATH_CALUDE_total_goals_after_five_matches_l3307_330758


namespace NUMINAMATH_CALUDE_derivative_cos_2x_plus_1_l3307_330772

theorem derivative_cos_2x_plus_1 (x : ℝ) :
  deriv (fun x => Real.cos (2 * x + 1)) x = -2 * Real.sin (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_2x_plus_1_l3307_330772


namespace NUMINAMATH_CALUDE_system_solution_l3307_330789

theorem system_solution : 
  let x : ℚ := 8 / 47
  let y : ℚ := 138 / 47
  (7 * x = 10 - 3 * y) ∧ (4 * x = 5 * y - 14) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3307_330789


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3307_330712

-- Problem 1
theorem problem_one : 2 * Real.cos (π / 4) + |1 - Real.sqrt 2| + (-2) ^ 0 = 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two (a : ℝ) : 3 * a + 2 * a * (a - 1) = 2 * a ^ 2 + a := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3307_330712


namespace NUMINAMATH_CALUDE_rod_mass_is_one_fourth_l3307_330751

/-- The linear density function of the rod -/
def ρ : ℝ → ℝ := fun x ↦ x^3

/-- The length of the rod -/
def rod_length : ℝ := 1

/-- The mass of the rod -/
noncomputable def rod_mass : ℝ := ∫ x in (0)..(rod_length), ρ x

/-- Theorem: The mass of the rod is equal to 1/4 -/
theorem rod_mass_is_one_fourth : rod_mass = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rod_mass_is_one_fourth_l3307_330751


namespace NUMINAMATH_CALUDE_tangent_circle_value_l3307_330794

/-- A line in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 1 = 0

/-- A circle in polar coordinates -/
def polar_circle (a ρ θ : ℝ) : Prop :=
  ρ = 2 * a * Real.cos θ ∧ a > 0

/-- Tangency condition between a line and a circle -/
def is_tangent (a : ℝ) : Prop :=
  ∃ ρ θ, polar_line ρ θ ∧ polar_circle a ρ θ

theorem tangent_circle_value :
  ∃ a, is_tangent a ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_value_l3307_330794


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3307_330717

theorem product_of_sums_equals_difference_of_powers : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * 
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3307_330717


namespace NUMINAMATH_CALUDE_probability_sum_20_l3307_330732

def total_balls : ℕ := 5
def balls_labeled_5 : ℕ := 3
def balls_labeled_10 : ℕ := 2
def balls_drawn : ℕ := 3
def target_sum : ℕ := 20

theorem probability_sum_20 : 
  (Nat.choose balls_labeled_5 2 * Nat.choose balls_labeled_10 1) / 
  Nat.choose total_balls balls_drawn = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_sum_20_l3307_330732


namespace NUMINAMATH_CALUDE_equation_solution_l3307_330768

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 16))) = 55 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3307_330768


namespace NUMINAMATH_CALUDE_age_determination_l3307_330747

/-- Represents a triple of positive integers -/
structure AgeTriple where
  a : Nat
  b : Nat
  c : Nat
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- The product of the three ages is 2450 -/
def product_is_2450 (t : AgeTriple) : Prop :=
  t.a * t.b * t.c = 2450

/-- The sum of the three ages is even -/
def sum_is_even (t : AgeTriple) : Prop :=
  ∃ k : Nat, t.a + t.b + t.c = 2 * k

/-- The smallest age is unique -/
def smallest_is_unique (t : AgeTriple) : Prop :=
  (t.a < t.b ∧ t.a < t.c) ∨ (t.b < t.a ∧ t.b < t.c) ∨ (t.c < t.a ∧ t.c < t.b)

theorem age_determination :
  ∃! (t1 t2 : AgeTriple),
    product_is_2450 t1 ∧
    product_is_2450 t2 ∧
    sum_is_even t1 ∧
    sum_is_even t2 ∧
    t1 ≠ t2 ∧
    (∀ t : AgeTriple, product_is_2450 t ∧ sum_is_even t → t = t1 ∨ t = t2) ∧
    ∃! (t : AgeTriple),
      product_is_2450 t ∧
      sum_is_even t ∧
      smallest_is_unique t ∧
      (t = t1 ∨ t = t2) :=
by
  sorry

#check age_determination

end NUMINAMATH_CALUDE_age_determination_l3307_330747


namespace NUMINAMATH_CALUDE_set_operations_l3307_330773

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ B = {1, 2, 3}) ∧
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3307_330773


namespace NUMINAMATH_CALUDE_lunch_group_probability_l3307_330715

theorem lunch_group_probability (total_students : ℕ) (num_groups : ℕ) (friends : ℕ) 
  (h1 : total_students = 800)
  (h2 : num_groups = 4)
  (h3 : friends = 4)
  (h4 : total_students % num_groups = 0) :
  (1 : ℚ) / (num_groups ^ (friends - 1)) = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_lunch_group_probability_l3307_330715


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l3307_330770

/-- 
Given an isosceles triangle with base l and height h, and a rectangle with length l and width w,
if their areas are equal, then the height of the triangle is twice the width of the rectangle.
-/
theorem isosceles_triangle_rectangle_equal_area 
  (l w h : ℝ) (l_pos : l > 0) (w_pos : w > 0) (h_pos : h > 0) : 
  (1 / 2 : ℝ) * l * h = l * w → h = 2 * w := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l3307_330770


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3307_330757

theorem simplify_and_evaluate_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) (h4 : a ≠ -1) (h5 : a = 1) :
  1 - (a - 2) / a / ((a^2 - 4) / (a^2 + a)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3307_330757


namespace NUMINAMATH_CALUDE_factor_condition_l3307_330722

theorem factor_condition (a b c m l : ℝ) : 
  ((b + c) * (c + a) * (a + b) + a * b * c = 
   (m * (a^2 + b^2 + c^2) + l * (a * b + a * c + b * c)) * k) →
  (m = 0 ∧ l = a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_factor_condition_l3307_330722


namespace NUMINAMATH_CALUDE_sams_weight_l3307_330779

/-- Given the weights of Tyler, Sam, Peter, and Alex, prove Sam's weight --/
theorem sams_weight (tyler sam peter alex : ℝ) : 
  tyler = sam + 25 →
  peter = tyler / 2 →
  alex = 2 * (sam + peter) →
  peter = 65 →
  sam = 105 := by
  sorry

end NUMINAMATH_CALUDE_sams_weight_l3307_330779


namespace NUMINAMATH_CALUDE_pool_filling_time_l3307_330735

/-- Represents the volume of the pool -/
def pool_volume : ℝ := 1

/-- Represents the rate at which pipe X fills the pool -/
def rate_X : ℝ := sorry

/-- Represents the rate at which pipe Y fills the pool -/
def rate_Y : ℝ := sorry

/-- Represents the rate at which pipe Z fills the pool -/
def rate_Z : ℝ := sorry

/-- Time taken by pipes X and Y together to fill the pool -/
def time_XY : ℝ := 3

/-- Time taken by pipes X and Z together to fill the pool -/
def time_XZ : ℝ := 6

/-- Time taken by pipes Y and Z together to fill the pool -/
def time_YZ : ℝ := 4.5

theorem pool_filling_time :
  let time_XYZ := pool_volume / (rate_X + rate_Y + rate_Z)
  pool_volume / (rate_X + rate_Y) = time_XY ∧
  pool_volume / (rate_X + rate_Z) = time_XZ ∧
  pool_volume / (rate_Y + rate_Z) = time_YZ →
  (time_XYZ ≥ 2.76 ∧ time_XYZ ≤ 2.78) := by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3307_330735


namespace NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_l3307_330729

theorem x_squared_minus_four_y_squared (x y : ℝ) 
  (eq1 : x + 2*y = 4) 
  (eq2 : x - 2*y = -1) : 
  x^2 - 4*y^2 = -4 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_l3307_330729


namespace NUMINAMATH_CALUDE_larger_number_of_two_l3307_330730

theorem larger_number_of_two (x y : ℝ) : 
  x - y = 7 → x + y = 41 → max x y = 24 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_two_l3307_330730


namespace NUMINAMATH_CALUDE_largest_c_for_negative_four_in_range_l3307_330750

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_four_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), 
    (∃ (x : ℝ), f c' x = -4) → c' ≤ c) ∧
  (∃ (x : ℝ), f (9/4) x = -4) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_four_in_range_l3307_330750


namespace NUMINAMATH_CALUDE_empty_set_equality_l3307_330753

theorem empty_set_equality : 
  {x : ℝ | x^2 + 2 = 0} = {y : ℝ | y^2 + 1 < 0} := by sorry

end NUMINAMATH_CALUDE_empty_set_equality_l3307_330753


namespace NUMINAMATH_CALUDE_vending_machine_probability_l3307_330787

/-- Represents a vending machine with toys and their prices -/
structure VendingMachine :=
  (num_toys : ℕ)
  (min_price : ℚ)
  (price_increment : ℚ)

/-- Represents Peter's initial money -/
structure InitialMoney :=
  (quarters : ℕ)
  (bill : ℚ)

/-- The main theorem statement -/
theorem vending_machine_probability
  (vm : VendingMachine)
  (money : InitialMoney)
  (favorite_toy_price : ℚ) :
  vm.num_toys = 10 →
  vm.min_price = 25/100 →
  vm.price_increment = 25/100 →
  money.quarters = 10 →
  money.bill = 20 →
  favorite_toy_price = 2 →
  (probability_need_break_bill : ℚ) →
  probability_need_break_bill = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l3307_330787


namespace NUMINAMATH_CALUDE_pen_count_is_39_l3307_330763

/-- Calculate the final number of pens after a series of operations -/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * 2) - sharon_takes

/-- Theorem stating that given the initial conditions, the final number of pens is 39 -/
theorem pen_count_is_39 :
  final_pen_count 7 22 19 = 39 := by
  sorry

#eval final_pen_count 7 22 19

end NUMINAMATH_CALUDE_pen_count_is_39_l3307_330763


namespace NUMINAMATH_CALUDE_probability_sum_eleven_l3307_330743

def seven_sided_die : Finset Nat := Finset.range 7
def five_sided_die : Finset Nat := Finset.range 5

def total_outcomes : Nat := seven_sided_die.card * five_sided_die.card

def successful_outcomes : Finset (Nat × Nat) :=
  {(4, 4), (5, 3), (6, 2)}

theorem probability_sum_eleven :
  (successful_outcomes.card : ℚ) / total_outcomes = 3 / 35 := by
sorry

end NUMINAMATH_CALUDE_probability_sum_eleven_l3307_330743


namespace NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_digit_l3307_330764

def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0

def units_digit (n : ℕ) : ℕ := n % 10

def is_digit (d : ℕ) : Prop := d < 10

theorem smallest_non_five_divisible_unit_digit : 
  ∀ d : ℕ, is_digit d → 
  (∀ n : ℕ, is_divisible_by_five n → units_digit n ≠ d) → 
  d ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_digit_l3307_330764


namespace NUMINAMATH_CALUDE_cos_2theta_value_l3307_330777

theorem cos_2theta_value (θ : Real) 
  (h : Real.sin (2 * θ) - 4 * Real.sin (θ + Real.pi / 3) * Real.sin (θ - Real.pi / 6) = Real.sqrt 3 / 3) : 
  Real.cos (2 * θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l3307_330777


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3307_330705

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3307_330705


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3307_330723

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (4 - m, 2)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant (m : ℝ) :
  in_second_quadrant (P m) → m = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_point_in_second_quadrant_l3307_330723


namespace NUMINAMATH_CALUDE_f_13_equals_neg_2_l3307_330707

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_13_equals_neg_2 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period : has_period f 4) 
  (h_f_neg_1 : f (-1) = 2) : 
  f 13 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_13_equals_neg_2_l3307_330707


namespace NUMINAMATH_CALUDE_min_value_theorem_l3307_330778

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : m * 2 + n * 2 = 2) : 
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧ 
    ∀ (x y : ℝ), x > 0 → y > 0 → x * 2 + y * 2 = 2 → 1 / x + 2 / y ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3307_330778


namespace NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l3307_330782

theorem sum_geq_three_cube_root_three
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (h : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) :
  a + b + c ≥ 3 * (3 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_geq_three_cube_root_three_l3307_330782


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_800_l3307_330798

theorem greatest_multiple_of_four_cubed_less_than_800 :
  ∃ (x : ℕ), x = 8 ∧ 
  (∀ (y : ℕ), y > 0 ∧ 4 ∣ y ∧ y^3 < 800 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_800_l3307_330798


namespace NUMINAMATH_CALUDE_tan_squared_fixed_point_l3307_330739

noncomputable def f (x : ℝ) : ℝ := 1 / ((x + 1) / x)

theorem tan_squared_fixed_point (t : ℝ) (h : 0 ≤ t ∧ t ≤ π / 2) :
  f (Real.tan t ^ 2) = Real.tan t ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_fixed_point_l3307_330739


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l3307_330776

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ m n : ℤ, x^2 + b*x + 2023 = (x + m) * (x + n)) → b ≥ 136 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l3307_330776


namespace NUMINAMATH_CALUDE_cube_of_cube_root_fourth_smallest_prime_l3307_330748

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- State the theorem
theorem cube_of_cube_root_fourth_smallest_prime :
  (fourth_smallest_prime : ℝ) = ((fourth_smallest_prime : ℝ) ^ (1/3 : ℝ)) ^ 3 :=
sorry

end NUMINAMATH_CALUDE_cube_of_cube_root_fourth_smallest_prime_l3307_330748


namespace NUMINAMATH_CALUDE_zero_at_neg_one_one_zero_in_interval_l3307_330786

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 2 - a

-- Theorem 1: When a = -1, the function has a zero at x = 1
theorem zero_at_neg_one :
  f (-1) 1 = 0 := by sorry

-- Theorem 2: The function has exactly one zero in (0, 1] iff -1 ≤ a ≤ 0 or a ≤ -2
theorem one_zero_in_interval (a : ℝ) :
  (∃! x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ↔ (-1 ≤ a ∧ a ≤ 0) ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_zero_at_neg_one_one_zero_in_interval_l3307_330786


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_one_two_l3307_330767

-- Define the universal set U
def U : Set ℤ := {x : ℤ | |x - 1| < 3}

-- Define set A
def A : Set ℤ := {1, 2, 3}

-- Define the complement of B in U
def C_U_B : Set ℤ := {-1, 3}

-- Theorem to prove
theorem A_intersect_B_equals_one_two : A ∩ (U \ C_U_B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_one_two_l3307_330767


namespace NUMINAMATH_CALUDE_probability_square_or_triangle_l3307_330721

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of triangles
def num_triangles : ℕ := 3

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- Theorem statement
theorem probability_square_or_triangle :
  (num_triangles + num_squares : ℚ) / total_figures = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_triangle_l3307_330721


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3307_330733

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area_ratio : ℝ
  sum_of_parallel_sides : ℝ
  area_ratio_condition : area_ratio = 5 / 3
  sum_condition : AB + CD = sum_of_parallel_sides

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC
    is 5:3, and AB + CD = 160 cm, then AB = 100 cm -/
theorem trapezoid_segment_length (t : Trapezoid) (h : t.sum_of_parallel_sides = 160) : t.AB = 100 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_segment_length_l3307_330733


namespace NUMINAMATH_CALUDE_max_value_on_unit_circle_l3307_330731

/-- The maximum value of f(z) = |z^3 - z + 2| on the unit circle -/
theorem max_value_on_unit_circle :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧
  (∀ z : ℂ, Complex.abs z = 1 →
    Complex.abs (z^3 - z + 2) ≤ M) ∧
  (∃ z : ℂ, Complex.abs z = 1 ∧
    Complex.abs (z^3 - z + 2) = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_unit_circle_l3307_330731


namespace NUMINAMATH_CALUDE_factorization_equality_l3307_330766

theorem factorization_equality (x y : ℝ) : x^2*y - 6*x*y + 9*y = y*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3307_330766


namespace NUMINAMATH_CALUDE_common_roots_sum_l3307_330700

theorem common_roots_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + b*x + c = 0) →
  (∃ y : ℝ, y^2 + y + a = 0 ∧ y^2 + c*y + b = 0) →
  a + b + c = -3 := by
sorry

end NUMINAMATH_CALUDE_common_roots_sum_l3307_330700


namespace NUMINAMATH_CALUDE_f_positive_iff_f_plus_3abs_min_f_plus_3abs_min_value_l3307_330749

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part (1)
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > 1 ∨ x < -5 := by sorry

-- Theorem for part (2)
theorem f_plus_3abs_min (x : ℝ) : f x + 3 * |x - 4| ≥ 9 := by sorry

-- Theorem for the minimum value
theorem f_plus_3abs_min_value : ∃ x : ℝ, f x + 3 * |x - 4| = 9 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_f_plus_3abs_min_f_plus_3abs_min_value_l3307_330749
