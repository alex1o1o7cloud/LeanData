import Mathlib

namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l1825_182542

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l1825_182542


namespace NUMINAMATH_CALUDE_coordinate_transform_sum_l1825_182538

/-- Definition of the original coordinate system -/
structure OriginalCoord where
  x : ℝ
  y : ℝ

/-- Definition of the new coordinate system -/
structure NewCoord where
  x : ℝ
  y : ℝ

/-- Definition of a line -/
structure Line where
  slope : ℝ
  point : OriginalCoord

/-- Function to transform coordinates from original to new system -/
def transform (p : OriginalCoord) (L M : Line) : NewCoord :=
  sorry

/-- Theorem statement -/
theorem coordinate_transform_sum :
  let A : OriginalCoord := ⟨24, -1⟩
  let B : OriginalCoord := ⟨5, 6⟩
  let P : OriginalCoord := ⟨-14, 27⟩
  let L : Line := ⟨5/12, A⟩
  let M : Line := ⟨-12/5, B⟩  -- Perpendicular slope
  let new_P : NewCoord := transform P L M
  new_P.x + new_P.y = 31 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_transform_sum_l1825_182538


namespace NUMINAMATH_CALUDE_final_value_l1825_182500

/-- The value of A based on bundles --/
def A : ℕ := 6 * 1000 + 36 * 100

/-- The value of B based on jumping twice --/
def B : ℕ := 876 - 197 - 197

/-- Theorem stating the final result --/
theorem final_value : A - B = 9118 := by
  sorry

end NUMINAMATH_CALUDE_final_value_l1825_182500


namespace NUMINAMATH_CALUDE_smaller_sphere_radius_l1825_182501

-- Define the type for a sphere
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the function to check if two spheres are externally tangent
def are_externally_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (s1.radius + s2.radius)^2

-- Define the theorem
theorem smaller_sphere_radius 
  (s1 s2 s3 s4 : Sphere)
  (h1 : s1.radius = 2)
  (h2 : s2.radius = 2)
  (h3 : s3.radius = 3)
  (h4 : s4.radius = 3)
  (h5 : are_externally_tangent s1 s2)
  (h6 : are_externally_tangent s1 s3)
  (h7 : are_externally_tangent s1 s4)
  (h8 : are_externally_tangent s2 s3)
  (h9 : are_externally_tangent s2 s4)
  (h10 : are_externally_tangent s3 s4)
  (s5 : Sphere)
  (h11 : are_externally_tangent s1 s5)
  (h12 : are_externally_tangent s2 s5)
  (h13 : are_externally_tangent s3 s5)
  (h14 : are_externally_tangent s4 s5) :
  s5.radius = 6/11 :=
by sorry

end NUMINAMATH_CALUDE_smaller_sphere_radius_l1825_182501


namespace NUMINAMATH_CALUDE_unreachable_141_l1825_182582

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  (n % 10) * digit_product (n / 10)

def next_number (n : ℕ) : Set ℕ :=
  {n + digit_product n, n - digit_product n}

def reachable (start : ℕ) : Set ℕ :=
  sorry

theorem unreachable_141 :
  141 ∉ reachable 141 \ {141} :=
sorry

end NUMINAMATH_CALUDE_unreachable_141_l1825_182582


namespace NUMINAMATH_CALUDE_probability_three_blue_pens_l1825_182548

def total_pens : ℕ := 15
def blue_pens : ℕ := 8
def red_pens : ℕ := 7
def num_trials : ℕ := 7
def num_blue_picks : ℕ := 3

def prob_blue : ℚ := blue_pens / total_pens
def prob_red : ℚ := red_pens / total_pens

def binomial_coefficient (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem probability_three_blue_pens :
  (binomial_coefficient num_trials num_blue_picks : ℚ) *
  (prob_blue ^ num_blue_picks) *
  (prob_red ^ (num_trials - num_blue_picks)) =
  43025920 / 170859375 := by sorry

end NUMINAMATH_CALUDE_probability_three_blue_pens_l1825_182548


namespace NUMINAMATH_CALUDE_no_natural_solution_l1825_182590

theorem no_natural_solution : ¬ ∃ (m n : ℕ), (1 : ℚ) / m + (1 : ℚ) / n = (7 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1825_182590


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l1825_182514

/-- A function that returns the set of digits of a natural number -/
def digits (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that checks if a natural number is a six-digit number -/
def isSixDigit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

/-- The theorem stating that 142857 is the unique six-digit number satisfying the given conditions -/
theorem unique_six_digit_number :
  ∃! p : ℕ, isSixDigit p ∧
    (∀ i : Fin 6, isSixDigit ((i.val + 1) * p)) ∧
    (∀ i : Fin 6, digits ((i.val + 1) * p) = digits p) ∧
    p = 142857 :=
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l1825_182514


namespace NUMINAMATH_CALUDE_tan_pi_four_minus_theta_l1825_182561

theorem tan_pi_four_minus_theta (θ : Real) (h : (Real.tan θ) = -2) :
  Real.tan (π / 4 - θ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_four_minus_theta_l1825_182561


namespace NUMINAMATH_CALUDE_max_sum_three_consecutive_l1825_182534

/-- A circular arrangement of numbers from 1 to 10 -/
def CircularArrangement := Fin 10 → Fin 10

/-- The sum of three consecutive numbers in a circular arrangement -/
def sumThreeConsecutive (arr : CircularArrangement) (i : Fin 10) : Nat :=
  arr i + arr ((i + 1) % 10) + arr ((i + 2) % 10)

/-- The theorem stating the maximum sum of three consecutive numbers -/
theorem max_sum_three_consecutive :
  (∀ arr : CircularArrangement, ∃ i : Fin 10, sumThreeConsecutive arr i ≥ 18) ∧
  ¬(∀ arr : CircularArrangement, ∃ i : Fin 10, sumThreeConsecutive arr i ≥ 19) :=
sorry

end NUMINAMATH_CALUDE_max_sum_three_consecutive_l1825_182534


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l1825_182563

/-- Given points A, B, C in a plane rectangular coordinate system,
    where AB is parallel to the x-axis and AC is parallel to the y-axis,
    prove that a + b = -1 -/
theorem point_coordinates_sum (a b : ℝ) : 
  (∃ (A B C : ℝ × ℝ),
    A = (a, -1) ∧
    B = (2, 3 - b) ∧
    C = (-5, 4) ∧
    A.2 = B.2 ∧  -- AB is parallel to x-axis
    A.1 = C.1    -- AC is parallel to y-axis
  ) →
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l1825_182563


namespace NUMINAMATH_CALUDE_bridge_length_l1825_182580

/-- The length of a bridge given specific train conditions -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  train_speed * crossing_time - train_length = 205 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1825_182580


namespace NUMINAMATH_CALUDE_equality_sum_l1825_182552

theorem equality_sum (M N : ℚ) : 
  (3 : ℚ) / 5 = M / 30 ∧ (3 : ℚ) / 5 = 90 / N → M + N = 168 := by
  sorry

end NUMINAMATH_CALUDE_equality_sum_l1825_182552


namespace NUMINAMATH_CALUDE_magnitude_ratio_not_sufficient_for_parallel_l1825_182573

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem magnitude_ratio_not_sufficient_for_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ (a b : V), ‖a‖ = 2 * ‖b‖ → parallel a b) := by
  sorry


end NUMINAMATH_CALUDE_magnitude_ratio_not_sufficient_for_parallel_l1825_182573


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1825_182522

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 7) → x = 24.25 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1825_182522


namespace NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l1825_182503

theorem fraction_sum_equals_percentage (y : ℝ) (h : y > 0) :
  (7 * y) / 20 + (3 * y) / 10 = 0.65 * y := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l1825_182503


namespace NUMINAMATH_CALUDE_range_of_p_exists_point_C_l1825_182562

-- Define the parabola L: x^2 = 2py
def L (p : ℝ) := {(x, y) : ℝ × ℝ | x^2 = 2*p*y ∧ p > 0}

-- Define point M
def M : ℝ × ℝ := (2, 2)

-- Define the condition for points A and B
def satisfies_condition (A B : ℝ × ℝ) (p : ℝ) :=
  A ∈ L p ∧ B ∈ L p ∧ A ≠ B ∧ 
  (A.1 - M.1, A.2 - M.2) = (-B.1 + M.1, -B.2 + M.2)

-- Theorem 1: Range of p
theorem range_of_p (p : ℝ) :
  (∃ A B, satisfies_condition A B p) → p > 1 :=
sorry

-- Define the circle through three points
def circle_through (A B C : ℝ × ℝ) := 
  {(x, y) : ℝ × ℝ | (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
                    (x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2}

-- Define the tangent line to the parabola at a point
def tangent_line (p : ℝ) (C : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | y - C.2 = (C.1 / (2*p)) * (x - C.1)}

-- Theorem 2: Existence of point C when p = 2
theorem exists_point_C :
  ∃ C, C ∈ L 2 ∧ C ≠ (0, 0) ∧ C ≠ (4, 4) ∧
       C.1 = -2 ∧ C.2 = 1 ∧
       (∀ x y, (x, y) ∈ circle_through (0, 0) (4, 4) C →
               (x, y) ∈ tangent_line 2 C) :=
sorry

end NUMINAMATH_CALUDE_range_of_p_exists_point_C_l1825_182562


namespace NUMINAMATH_CALUDE_parallel_planes_and_line_l1825_182511

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the "not contained in" relation for lines and planes
variable (line_not_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_planes_and_line 
  (α β : Plane) (a : Line) 
  (h1 : plane_parallel α β)
  (h2 : line_not_in_plane a α)
  (h3 : line_not_in_plane a β)
  (h4 : line_parallel_plane a α) :
  line_parallel_plane a β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_and_line_l1825_182511


namespace NUMINAMATH_CALUDE_exists_m_f_even_l1825_182585

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^2 + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- There exists an m ∈ ℝ such that f(x) = x^2 + mx is an even function -/
theorem exists_m_f_even : ∃ m : ℝ, IsEven (f m) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_f_even_l1825_182585


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1825_182592

theorem largest_n_divisibility (n : ℕ) : (n + 1) ∣ (n^3 + 10) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1825_182592


namespace NUMINAMATH_CALUDE_ratio_equation_solution_product_l1825_182523

theorem ratio_equation_solution_product (x : ℝ) :
  (3 * x + 5) / (4 * x + 4) = (5 * x + 4) / (10 * x + 5) →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (3 * x₁ + 5) / (4 * x₁ + 4) = (5 * x₁ + 4) / (10 * x₁ + 5) ∧
    (3 * x₂ + 5) / (4 * x₂ + 4) = (5 * x₂ + 4) / (10 * x₂ + 5) ∧
    x₁ * x₂ = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_product_l1825_182523


namespace NUMINAMATH_CALUDE_uniform_pickup_ways_l1825_182560

def number_of_students : ℕ := 5
def correct_picks : ℕ := 2

theorem uniform_pickup_ways :
  (number_of_students.choose correct_picks) * 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_uniform_pickup_ways_l1825_182560


namespace NUMINAMATH_CALUDE_lawn_chair_price_calculation_l1825_182504

/-- Calculates the final price and overall percent decrease of a lawn chair after discounts and tax --/
theorem lawn_chair_price_calculation (original_price : ℝ) 
  (first_discount_rate second_discount_rate tax_rate : ℝ) :
  original_price = 72.95 ∧ 
  first_discount_rate = 0.10 ∧ 
  second_discount_rate = 0.15 ∧ 
  tax_rate = 0.07 →
  ∃ (final_price percent_decrease : ℝ),
    (abs (final_price - 59.71) < 0.01) ∧ 
    (abs (percent_decrease - 23.5) < 0.1) ∧
    final_price = (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) * (1 + tax_rate) ∧
    percent_decrease = (1 - (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) / original_price) * 100 := by
  sorry

end NUMINAMATH_CALUDE_lawn_chair_price_calculation_l1825_182504


namespace NUMINAMATH_CALUDE_brandy_excess_caffeine_l1825_182528

/-- Represents the caffeine consumption and tolerance of a person named Brandy --/
structure BrandyCaffeine where
  weight : ℝ
  baseLimit : ℝ
  additionalTolerance : ℝ
  coffeeConsumption : ℝ
  energyDrinkConsumption : ℝ
  medicationEffect : ℝ

/-- Calculates the excess caffeine consumed by Brandy --/
def excessCaffeineConsumed (b : BrandyCaffeine) : ℝ :=
  let maxSafe := b.weight * b.baseLimit + b.additionalTolerance - b.medicationEffect
  let consumed := b.coffeeConsumption + b.energyDrinkConsumption
  consumed - maxSafe

/-- Theorem stating that Brandy has consumed 495 mg more caffeine than her adjusted maximum safe amount --/
theorem brandy_excess_caffeine :
  let b : BrandyCaffeine := {
    weight := 60,
    baseLimit := 2.5,
    additionalTolerance := 50,
    coffeeConsumption := 2 * 95,
    energyDrinkConsumption := 4 * 120,
    medicationEffect := 25
  }
  excessCaffeineConsumed b = 495 := by sorry

end NUMINAMATH_CALUDE_brandy_excess_caffeine_l1825_182528


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l1825_182516

theorem quadratic_roots_average (c : ℝ) 
  (h : ∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - 6 * x + c = 0 ∧ 2 * y^2 - 6 * y + c = 0) :
  ∃ x y : ℝ, x ≠ y ∧ 
    2 * x^2 - 6 * x + c = 0 ∧ 
    2 * y^2 - 6 * y + c = 0 ∧ 
    (x + y) / 2 = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l1825_182516


namespace NUMINAMATH_CALUDE_less_expensive_coat_cost_l1825_182553

/-- Represents the cost of a coat and its lifespan in years -/
structure Coat where
  cost : ℕ
  lifespan : ℕ

/-- Calculates the total cost of a coat over a given period -/
def totalCost (coat : Coat) (period : ℕ) : ℕ :=
  (period / coat.lifespan) * coat.cost

theorem less_expensive_coat_cost (expensive_coat less_expensive_coat : Coat) : 
  expensive_coat.cost = 300 →
  expensive_coat.lifespan = 15 →
  less_expensive_coat.lifespan = 5 →
  totalCost expensive_coat 30 + 120 = totalCost less_expensive_coat 30 →
  less_expensive_coat.cost = 120 := by
sorry

end NUMINAMATH_CALUDE_less_expensive_coat_cost_l1825_182553


namespace NUMINAMATH_CALUDE_volume_of_three_cubes_cuboid_l1825_182509

/-- The volume of a cuboid formed by attaching three identical cubes -/
def cuboid_volume (cube_side_length : ℝ) (num_cubes : ℕ) : ℝ :=
  (cube_side_length ^ 3) * num_cubes

/-- Theorem: The volume of a cuboid formed by three 6cm cubes is 648 cm³ -/
theorem volume_of_three_cubes_cuboid : 
  cuboid_volume 6 3 = 648 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_three_cubes_cuboid_l1825_182509


namespace NUMINAMATH_CALUDE_e_pi_plus_pi_e_approx_l1825_182555

/-- Approximate value of e -/
def e_approx : ℝ := 2.718

/-- Approximate value of π -/
def π_approx : ℝ := 3.14159

/-- Theorem stating that e^π + π^e is approximately equal to 45.5999 -/
theorem e_pi_plus_pi_e_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |Real.exp π_approx + Real.exp e_approx - 45.5999| < ε :=
sorry

end NUMINAMATH_CALUDE_e_pi_plus_pi_e_approx_l1825_182555


namespace NUMINAMATH_CALUDE_max_value_sin_squared_minus_two_sin_minus_two_l1825_182576

theorem max_value_sin_squared_minus_two_sin_minus_two :
  ∀ x : ℝ, 
    -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 →
    ∀ y : ℝ, 
      y = Real.sin x ^ 2 - 2 * Real.sin x - 2 →
      y ≤ 1 ∧ ∃ x₀ : ℝ, Real.sin x₀ ^ 2 - 2 * Real.sin x₀ - 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_sin_squared_minus_two_sin_minus_two_l1825_182576


namespace NUMINAMATH_CALUDE_consecutive_root_count_l1825_182541

/-- A function that checks if a number is divisible by 5 -/
def divisible_by_five (m : ℤ) : Prop := ∃ k : ℤ, m = 5 * k

/-- A function that checks if two integers are consecutive -/
def consecutive (a b : ℤ) : Prop := b = a + 1

/-- A function that checks if a number is a positive integer -/
def is_positive_integer (x : ℤ) : Prop := x > 0

/-- The main theorem -/
theorem consecutive_root_count :
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, n < 50 ∧ is_positive_integer n) ∧ 
    (∀ n ∈ S, ∃ m : ℤ, 
      divisible_by_five m ∧
      ∃ a b : ℤ, is_positive_integer a ∧ is_positive_integer b ∧ consecutive a b ∧
      a * b = m ∧ a + b = n) ∧
    Finset.card S = 5 := by sorry

end NUMINAMATH_CALUDE_consecutive_root_count_l1825_182541


namespace NUMINAMATH_CALUDE_paul_bought_two_pants_l1825_182530

def shirtPrice : ℝ := 15
def pantPrice : ℝ := 40
def suitPrice : ℝ := 150
def sweaterPrice : ℝ := 30
def storeDiscount : ℝ := 0.2
def couponDiscount : ℝ := 0.1
def finalSpent : ℝ := 252

def totalBeforeDiscount (numPants : ℝ) : ℝ :=
  4 * shirtPrice + numPants * pantPrice + suitPrice + 2 * sweaterPrice

def discountedTotal (numPants : ℝ) : ℝ :=
  (1 - storeDiscount) * totalBeforeDiscount numPants

def finalTotal (numPants : ℝ) : ℝ :=
  (1 - couponDiscount) * discountedTotal numPants

theorem paul_bought_two_pants :
  ∃ (numPants : ℝ), numPants = 2 ∧ finalTotal numPants = finalSpent :=
sorry

end NUMINAMATH_CALUDE_paul_bought_two_pants_l1825_182530


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1825_182525

theorem arithmetic_mean_problem (a b c : ℝ) :
  let numbers := [a, b, c, 108]
  (numbers.sum / numbers.length = 92) →
  ((a + b + c) / 3 = 260 / 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1825_182525


namespace NUMINAMATH_CALUDE_monotonicity_and_minimum_l1825_182586

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + a + 1)

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp (-x) * (-a * x^2 + 2 * a * x - a - 1)

theorem monotonicity_and_minimum :
  (∀ x, a ≥ 0 → f_derivative a x < 0) ∧
  (a < 0 → ∃ r₁ r₂, r₁ < r₂ ∧ r₂ < 0 ∧
    (∀ x, x < r₁ → f_derivative a x > 0) ∧
    (∀ x, r₁ < x ∧ x < r₂ → f_derivative a x < 0) ∧
    (∀ x, x > r₂ → f_derivative a x > 0)) ∧
  (-1 < a ∧ a < 0 → ∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≥ f a 2) :=
by sorry

end

end NUMINAMATH_CALUDE_monotonicity_and_minimum_l1825_182586


namespace NUMINAMATH_CALUDE_log_expression_simplification_l1825_182520

theorem log_expression_simplification 
  (p q r s t z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + 2 * Real.log (r / s) - Real.log (p * t / (s * z)) = 
  Real.log (r * z / (s * t)) := by
sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l1825_182520


namespace NUMINAMATH_CALUDE_gray_sections_total_seeds_l1825_182566

theorem gray_sections_total_seeds (circle1_total : ℕ) (circle2_total : ℕ) (white_section : ℕ)
  (h1 : circle1_total = 87)
  (h2 : circle2_total = 110)
  (h3 : white_section = 68) :
  (circle1_total - white_section) + (circle2_total - white_section) = 61 := by
  sorry

end NUMINAMATH_CALUDE_gray_sections_total_seeds_l1825_182566


namespace NUMINAMATH_CALUDE_average_of_first_16_even_divisible_by_5_l1825_182506

def first_16_even_divisible_by_5 : List Nat :=
  List.range 16 |> List.map (fun n => 10 * (n + 1))

theorem average_of_first_16_even_divisible_by_5 :
  (List.sum first_16_even_divisible_by_5) / first_16_even_divisible_by_5.length = 85 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_16_even_divisible_by_5_l1825_182506


namespace NUMINAMATH_CALUDE_xy_min_max_l1825_182521

theorem xy_min_max (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) (h3 : -2 ≤ a ∧ a ≤ 2) :
  (∃ (x y : ℝ), x * y = -1 ∧ 
    ∀ (x' y' : ℝ), x' + y' = a → x'^2 + y'^2 = -a^2 + 2 → x' * y' ≥ -1) ∧
  (∃ (x y : ℝ), x * y = 1/3 ∧ 
    ∀ (x' y' : ℝ), x' + y' = a → x'^2 + y'^2 = -a^2 + 2 → x' * y' ≤ 1/3) :=
by sorry

end NUMINAMATH_CALUDE_xy_min_max_l1825_182521


namespace NUMINAMATH_CALUDE_distance_product_on_curve_l1825_182595

/-- The curve C defined by xy = 2 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 2}

/-- The theorem stating that the product of distances from any point on C to the axes is 2 -/
theorem distance_product_on_curve (p : ℝ × ℝ) (h : p ∈ C) :
  |p.1| * |p.2| = 2 := by
  sorry


end NUMINAMATH_CALUDE_distance_product_on_curve_l1825_182595


namespace NUMINAMATH_CALUDE_score_difference_theorem_l1825_182547

def score_distribution : List (Float × Float) := [
  (75, 0.15),
  (85, 0.30),
  (90, 0.25),
  (95, 0.10),
  (100, 0.20)
]

def mean (dist : List (Float × Float)) : Float :=
  (dist.map (fun (score, freq) => score * freq)).sum

def median (dist : List (Float × Float)) : Float :=
  90  -- The median is 90 based on the given distribution

theorem score_difference_theorem :
  mean score_distribution - median score_distribution = -1.25 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_theorem_l1825_182547


namespace NUMINAMATH_CALUDE_prob_one_common_correct_l1825_182589

/-- The number of numbers in the lottery -/
def total_numbers : ℕ := 45

/-- The number of numbers each participant chooses -/
def chosen_numbers : ℕ := 6

/-- Calculates the probability of exactly one common number between two independently chosen combinations -/
def prob_one_common : ℚ :=
  (chosen_numbers : ℚ) * (Nat.choose (total_numbers - chosen_numbers) (chosen_numbers - 1) : ℚ) /
  (Nat.choose total_numbers chosen_numbers : ℚ)

/-- Theorem stating that the probability of exactly one common number is correct -/
theorem prob_one_common_correct :
  prob_one_common = (6 : ℚ) * (Nat.choose 39 5 : ℚ) / (Nat.choose 45 6 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_prob_one_common_correct_l1825_182589


namespace NUMINAMATH_CALUDE_product_of_specific_difference_and_cube_difference_l1825_182519

theorem product_of_specific_difference_and_cube_difference
  (x y : ℝ) (h1 : x - y = 4) (h2 : x^3 - y^3 = 28) : x * y = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_difference_and_cube_difference_l1825_182519


namespace NUMINAMATH_CALUDE_base5_1234_equals_194_l1825_182537

def base5_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem base5_1234_equals_194 :
  base5_to_decimal [4, 3, 2, 1] = 194 := by
  sorry

end NUMINAMATH_CALUDE_base5_1234_equals_194_l1825_182537


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1825_182557

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, (abs x > a → x^2 - x - 2 > 0) ∧ 
  (∃ y : ℝ, y^2 - y - 2 > 0 ∧ abs y ≤ a)) ↔ 
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1825_182557


namespace NUMINAMATH_CALUDE_expression_equality_l1825_182535

theorem expression_equality : 2 * (2^7 + 2^7 + 2^8)^(1/4) = 8 * 2^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1825_182535


namespace NUMINAMATH_CALUDE_chastity_gummy_packs_l1825_182526

/-- Given Chastity's candy purchase scenario, prove the number of gummy packs bought. -/
theorem chastity_gummy_packs :
  ∀ (initial_money : ℚ) 
    (remaining_money : ℚ) 
    (lollipop_count : ℕ) 
    (lollipop_price : ℚ) 
    (gummy_pack_price : ℚ),
  initial_money = 15 →
  remaining_money = 5 →
  lollipop_count = 4 →
  lollipop_price = 3/2 →
  gummy_pack_price = 2 →
  ∃ (gummy_pack_count : ℕ),
    gummy_pack_count = 2 ∧
    initial_money - remaining_money = 
      (lollipop_count : ℚ) * lollipop_price + (gummy_pack_count : ℚ) * gummy_pack_price :=
by sorry

end NUMINAMATH_CALUDE_chastity_gummy_packs_l1825_182526


namespace NUMINAMATH_CALUDE_soap_cost_for_year_l1825_182524

/-- The cost of soap for a year given the duration and price of a single bar -/
theorem soap_cost_for_year (months_per_bar : ℕ) (price_per_bar : ℕ) : 
  months_per_bar = 2 → price_per_bar = 8 → (12 / months_per_bar) * price_per_bar = 48 := by
  sorry

#check soap_cost_for_year

end NUMINAMATH_CALUDE_soap_cost_for_year_l1825_182524


namespace NUMINAMATH_CALUDE_expression_always_zero_l1825_182577

theorem expression_always_zero (x y : ℝ) : 
  5 * (x^3 - 3*x^2*y - 2*x*y^2) - 3 * (x^3 - 5*x^2*y + 2*y^3) + 2 * (-x^3 + 5*x*y^2 + 3*y^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_zero_l1825_182577


namespace NUMINAMATH_CALUDE_farm_ratio_change_l1825_182579

/-- Represents the farm's livestock inventory --/
structure Farm where
  horses : ℕ
  cows : ℕ

/-- Calculates the ratio of horses to cows as a pair of natural numbers --/
def ratio (f : Farm) : ℕ × ℕ :=
  let gcd := Nat.gcd f.horses f.cows
  (f.horses / gcd, f.cows / gcd)

theorem farm_ratio_change (initial : Farm) (final : Farm) : 
  (ratio initial = (3, 1)) →
  (final.horses = initial.horses - 15) →
  (final.cows = initial.cows + 15) →
  (final.horses = final.cows + 30) →
  (ratio final = (5, 3)) := by
  sorry


end NUMINAMATH_CALUDE_farm_ratio_change_l1825_182579


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1825_182565

theorem complex_number_quadrant (z : ℂ) : z = (2 + Complex.I) / 3 → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1825_182565


namespace NUMINAMATH_CALUDE_count_integer_lengths_problem_triangle_l1825_182527

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of distinct integer lengths of line segments 
    from a vertex to points on the hypotenuse -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { leg1 := 15, leg2 := 36 }

/-- The theorem stating that the number of distinct integer lengths 
    for the given triangle is 24 -/
theorem count_integer_lengths_problem_triangle : 
  countIntegerLengths problemTriangle = 24 :=
sorry

end NUMINAMATH_CALUDE_count_integer_lengths_problem_triangle_l1825_182527


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1825_182518

theorem cube_root_simplification :
  (72^3 + 108^3 + 144^3 : ℝ)^(1/3) = 36 * 99^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1825_182518


namespace NUMINAMATH_CALUDE_quadratic_extrema_l1825_182510

theorem quadratic_extrema :
  (∀ x : ℝ, 2 * x^2 - 1 ≥ -1) ∧
  (∀ x : ℝ, 2 * x^2 - 1 = -1 ↔ x = 0) ∧
  (∀ x : ℝ, -2 * (x + 1)^2 + 1 ≤ 1) ∧
  (∀ x : ℝ, -2 * (x + 1)^2 + 1 = 1 ↔ x = -1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 ≥ -1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 = -1 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l1825_182510


namespace NUMINAMATH_CALUDE_john_weight_lifting_l1825_182508

/-- John's weight lifting problem -/
theorem john_weight_lifting 
  (weight_per_rep : ℕ) 
  (reps_per_set : ℕ) 
  (num_sets : ℕ) 
  (h1 : weight_per_rep = 15)
  (h2 : reps_per_set = 10)
  (h3 : num_sets = 3) :
  weight_per_rep * reps_per_set * num_sets = 450 := by
  sorry

#check john_weight_lifting

end NUMINAMATH_CALUDE_john_weight_lifting_l1825_182508


namespace NUMINAMATH_CALUDE_one_km_equals_500_chains_l1825_182581

-- Define the units
def kilometer : ℕ → ℕ := id
def hectometer : ℕ → ℕ := id
def chain : ℕ → ℕ := id

-- Define the conversion factors
axiom km_to_hm : ∀ x : ℕ, kilometer x = hectometer (10 * x)
axiom hm_to_chain : ∀ x : ℕ, hectometer x = chain (50 * x)

-- Theorem to prove
theorem one_km_equals_500_chains : kilometer 1 = chain 500 := by
  sorry

end NUMINAMATH_CALUDE_one_km_equals_500_chains_l1825_182581


namespace NUMINAMATH_CALUDE_sequence_a_property_l1825_182539

def sequence_a (n : ℕ+) : ℚ := 1 / ((n + 1) * (n + 2))

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := (n * (n + 1) : ℚ) / 2 * a n

theorem sequence_a_property (a : ℕ+ → ℚ) : 
  a 1 = 1/6 → 
  (∀ n : ℕ+, S n a = (n * (n + 1) : ℚ) / 2 * a n) → 
  ∀ n : ℕ+, a n = sequence_a n :=
sorry

end NUMINAMATH_CALUDE_sequence_a_property_l1825_182539


namespace NUMINAMATH_CALUDE_eighteen_is_seventyfive_percent_of_twentyfour_l1825_182505

theorem eighteen_is_seventyfive_percent_of_twentyfour (x : ℝ) : 
  18 = 0.75 * x → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_is_seventyfive_percent_of_twentyfour_l1825_182505


namespace NUMINAMATH_CALUDE_xyz_inequality_and_sum_l1825_182588

theorem xyz_inequality_and_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 8) :
  ((x + y < 7) → (x / (1 + x) + y / (1 + y) > 2 * Real.sqrt ((x * y) / (x * y + 8)))) ∧
  (⌈(1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) + 1 / Real.sqrt (1 + z))⌉ = 2) := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_and_sum_l1825_182588


namespace NUMINAMATH_CALUDE_expression_evaluation_l1825_182597

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -1/4
  ((3*x + 2*y) * (3*x - 2*y) - (3*x - 2*y)^2) / (4*y) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1825_182597


namespace NUMINAMATH_CALUDE_nancys_payment_is_384_l1825_182517

/-- Nancy's annual payment for her daughter's car insurance -/
def nancys_annual_payment (total_monthly_cost : ℝ) (nancy_share_percent : ℝ) : ℝ :=
  total_monthly_cost * nancy_share_percent * 12

/-- Proof that Nancy's annual payment is $384 -/
theorem nancys_payment_is_384 :
  nancys_annual_payment 80 0.4 = 384 := by
  sorry

end NUMINAMATH_CALUDE_nancys_payment_is_384_l1825_182517


namespace NUMINAMATH_CALUDE_box_packing_l1825_182575

theorem box_packing (total_items : Nat) (items_per_small_box : Nat) (small_boxes_per_big_box : Nat)
  (h1 : total_items = 8640)
  (h2 : items_per_small_box = 12)
  (h3 : small_boxes_per_big_box = 6)
  (h4 : items_per_small_box > 0)
  (h5 : small_boxes_per_big_box > 0) :
  total_items / (items_per_small_box * small_boxes_per_big_box) = 120 := by
sorry

end NUMINAMATH_CALUDE_box_packing_l1825_182575


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1825_182572

theorem arithmetic_calculation : 4 * (8 - 3) / 2 - 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1825_182572


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l1825_182570

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2

/-- If S_2 / S_4 = 1/3 for an arithmetic sequence, then S_4 / S_8 = 3/10 -/
theorem arithmetic_sequence_ratio_property (seq : ArithmeticSequence) 
    (h : seq.S 2 / seq.S 4 = 1/3) : 
    seq.S 4 / seq.S 8 = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l1825_182570


namespace NUMINAMATH_CALUDE_area_formula_l1825_182568

/-- Triangle with sides a, b, c and angle A -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Theorem: Area formula for triangles with angle A = 60° or 120° -/
theorem area_formula (t : Triangle) :
  (t.angleA = 60 → area t = (Real.sqrt 3 / 4) * (t.a^2 - (t.b - t.c)^2)) ∧
  (t.angleA = 120 → area t = (Real.sqrt 3 / 12) * (t.a^2 - (t.b - t.c)^2)) := by
  sorry

end NUMINAMATH_CALUDE_area_formula_l1825_182568


namespace NUMINAMATH_CALUDE_fruit_boxes_problem_l1825_182591

theorem fruit_boxes_problem (total_pears : ℕ) : 
  (∃ (fruits_per_box : ℕ), 
    fruits_per_box = 12 + total_pears / 9 ∧ 
    fruits_per_box = (12 + total_pears) / 3 ∧
    fruits_per_box = 16) := by
  sorry

end NUMINAMATH_CALUDE_fruit_boxes_problem_l1825_182591


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_two_l1825_182549

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_four_units_from_negative_two : 
  {x : ℝ | distance x (-2) = 4} = {2, -6} := by
  sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_two_l1825_182549


namespace NUMINAMATH_CALUDE_midpoint_chain_l1825_182502

/-- Given a line segment AB with multiple midpoints, prove its length --/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 5) →        -- AG = 5
  (B - A = 160) :=     -- AB = 160
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1825_182502


namespace NUMINAMATH_CALUDE_greatest_value_problem_l1825_182533

theorem greatest_value_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hy : 0 < y₁ ∧ y₁ < y₂) 
  (hsum : x₁ + x₂ = 1 ∧ y₁ + y₂ = 1) : 
  max (x₁*y₁ + x₂*y₂) (max (x₁*x₂ + y₁*y₂) (max (x₁*y₂ + x₂*y₁) (1/2))) = x₁*y₁ + x₂*y₂ := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_problem_l1825_182533


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l1825_182593

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 + Real.log x else 3 * x + 4

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l1825_182593


namespace NUMINAMATH_CALUDE_final_price_is_correct_l1825_182556

def electronic_discount_rate : ℚ := 0.20
def clothing_discount_rate : ℚ := 0.15
def voucher_threshold : ℚ := 200
def voucher_value : ℚ := 20
def electronic_item_price : ℚ := 150
def clothing_item_price : ℚ := 80
def clothing_item_count : ℕ := 2

def calculate_final_price : ℚ := by
  -- Define the calculation here
  sorry

theorem final_price_is_correct :
  calculate_final_price = 236 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_correct_l1825_182556


namespace NUMINAMATH_CALUDE_xy_sum_problem_l1825_182544

theorem xy_sum_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (hx_bound : x < 15) (hy_bound : y < 15) (h_eq : x + y + x * y = 119) : 
  x + y = 21 ∨ x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l1825_182544


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l1825_182587

def balloon_permutations : ℕ := 1260

theorem balloon_arrangement_count :
  let total_letters : ℕ := 7
  let repeated_l : ℕ := 2
  let repeated_o : ℕ := 2
  balloon_permutations = Nat.factorial total_letters / (Nat.factorial repeated_l * Nat.factorial repeated_o) := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l1825_182587


namespace NUMINAMATH_CALUDE_painting_time_ratio_l1825_182583

-- Define the painting times for each person
def matt_time : ℝ := 12
def rachel_time : ℝ := 13

-- Define Patty's time in terms of a variable
def patty_time : ℝ → ℝ := λ p => p

-- Define Rachel's time in terms of Patty's time
def rachel_time_calc : ℝ → ℝ := λ p => 2 * p + 5

-- Theorem statement
theorem painting_time_ratio :
  ∃ p : ℝ, 
    rachel_time_calc p = rachel_time ∧ 
    (patty_time p) / matt_time = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_ratio_l1825_182583


namespace NUMINAMATH_CALUDE_total_distance_l1825_182532

def road_trip (tracy_miles michelle_miles katie_miles : ℕ) : Prop :=
  tracy_miles = 2 * michelle_miles + 20 ∧
  michelle_miles = 3 * katie_miles ∧
  michelle_miles = 294

theorem total_distance (tracy_miles michelle_miles katie_miles : ℕ) 
  (h : road_trip tracy_miles michelle_miles katie_miles) : 
  tracy_miles + michelle_miles + katie_miles = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_l1825_182532


namespace NUMINAMATH_CALUDE_max_area_difference_l1825_182554

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left vertex A and left focus F
def A : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := x = k*y - 1

-- Define the intersection points C and D
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | E p.1 p.2 ∧ line_through_F k p.1 p.2}

-- Define the area difference function
def area_difference (C D : ℝ × ℝ) : ℝ :=
  |C.2 + D.2|

-- Theorem statement
theorem max_area_difference :
  ∃ (max_diff : ℝ), max_diff = Real.sqrt 3 / 2 ∧
  ∀ (k : ℝ) (C D : ℝ × ℝ),
    C ∈ intersection_points k → D ∈ intersection_points k →
    area_difference C D ≤ max_diff :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l1825_182554


namespace NUMINAMATH_CALUDE_three_sum_exists_l1825_182545

theorem three_sum_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_increasing : ∀ i j, i < j → a i < a j) 
  (h_bound : a (Fin.last n) < 2 * n) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j = a k :=
sorry

end NUMINAMATH_CALUDE_three_sum_exists_l1825_182545


namespace NUMINAMATH_CALUDE_ampersand_example_l1825_182512

-- Define the & operation
def ampersand (a b : ℚ) : ℚ := (a + 1) / b

-- State the theorem
theorem ampersand_example : ampersand 2 (ampersand 3 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_example_l1825_182512


namespace NUMINAMATH_CALUDE_gcd_459_357_l1825_182584

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1825_182584


namespace NUMINAMATH_CALUDE_min_value_expression_l1825_182531

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (3 * r) / (p + 2 * q) + (3 * p) / (2 * r + q) + (2 * q) / (p + r) ≥ 29 / 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1825_182531


namespace NUMINAMATH_CALUDE_max_value_theorem_l1825_182596

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (max : ℝ), max = -9/2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1825_182596


namespace NUMINAMATH_CALUDE_john_rachel_toy_difference_l1825_182550

theorem john_rachel_toy_difference (jason_toys : ℕ) (rachel_toys : ℕ) :
  jason_toys = 21 →
  rachel_toys = 1 →
  ∃ (john_toys : ℕ),
    jason_toys = 3 * john_toys ∧
    john_toys > rachel_toys ∧
    john_toys - rachel_toys = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_rachel_toy_difference_l1825_182550


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1825_182567

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℚ) (interest : ℚ) (time : ℕ) :
  rate = 4 / 100 →
  interest = 128 →
  time = 4 →
  ∃ (principal : ℚ), principal * rate * time = interest ∧ principal = 800 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1825_182567


namespace NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l1825_182559

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) +
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l1825_182559


namespace NUMINAMATH_CALUDE_rational_sum_power_l1825_182551

theorem rational_sum_power (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : 
  (n + m)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_power_l1825_182551


namespace NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l1825_182558

theorem difference_of_odd_squares_divisible_by_eight (n p : ℤ) :
  ∃ k : ℤ, (2 * n + 1)^2 - (2 * p + 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l1825_182558


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l1825_182546

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6*I) : 
  Complex.abs z^2 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l1825_182546


namespace NUMINAMATH_CALUDE_intersection_A_B_l1825_182529

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1825_182529


namespace NUMINAMATH_CALUDE_pasta_bins_l1825_182599

theorem pasta_bins (total_bins soup_bins vegetable_bins : ℝ) 
  (h_total : total_bins = 0.75)
  (h_soup : soup_bins = 0.12)
  (h_vegetable : vegetable_bins = 0.12) :
  total_bins - soup_bins - vegetable_bins = 0.51 := by
sorry

end NUMINAMATH_CALUDE_pasta_bins_l1825_182599


namespace NUMINAMATH_CALUDE_first_meeting_cd_l1825_182543

-- Define the cars and their properties
structure Car where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

-- Define the race scenario
def race_scenario (a b c d : Car) : Prop :=
  a.direction ∧ b.direction ∧ ¬c.direction ∧ ¬d.direction ∧
  a.speed ≠ b.speed ∧ a.speed ≠ c.speed ∧ a.speed ≠ d.speed ∧
  b.speed ≠ c.speed ∧ b.speed ≠ d.speed ∧ c.speed ≠ d.speed ∧
  a.speed + c.speed = b.speed + d.speed ∧
  a.speed - b.speed = d.speed - c.speed

-- Define the meeting times
def first_meeting_ac_bd : ℝ := 7
def first_meeting_ab : ℝ := 53

-- Theorem statement
theorem first_meeting_cd 
  (a b c d : Car) 
  (h : race_scenario a b c d) :
  ∃ t : ℝ, t = first_meeting_ab :=
sorry

end NUMINAMATH_CALUDE_first_meeting_cd_l1825_182543


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1825_182594

theorem correct_average_after_error_correction (n : ℕ) (initial_avg : ℚ) (wrong_value correct_value : ℚ) :
  n = 10 →
  initial_avg = 23 →
  wrong_value = 26 →
  correct_value = 36 →
  (n : ℚ) * initial_avg + (correct_value - wrong_value) = n * 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1825_182594


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1825_182515

/-- Calculates the missing sale amount given the other sales and desired average -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 desired_average : ℕ) : ℕ :=
  5 * desired_average - (sale1 + sale2 + sale3 + sale5)

/-- Theorem: Given the sales for 5 consecutive months, where 4 of the 5 sales are known,
    and the desired average sale, the sale in the fourth month must be 7720. -/
theorem fourth_month_sale 
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale5 : ℕ) (desired_average : ℕ)
  (h1 : sale1 = 5420)
  (h2 : sale2 = 5660)
  (h3 : sale3 = 6200)
  (h4 : sale5 = 6500)
  (h5 : desired_average = 6300) :
  calculate_missing_sale sale1 sale2 sale3 sale5 desired_average = 7720 := by
  sorry

#eval calculate_missing_sale 5420 5660 6200 6500 6300

end NUMINAMATH_CALUDE_fourth_month_sale_l1825_182515


namespace NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l1825_182507

theorem quadratic_roots_real_distinct (d : ℝ) : 
  let a : ℝ := 3
  let b : ℝ := -4 * Real.sqrt 3
  let c : ℝ := d
  let discriminant : ℝ := b^2 - 4*a*c
  discriminant = 12 →
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l1825_182507


namespace NUMINAMATH_CALUDE_company_workers_count_l1825_182574

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  supervisors : ℕ
  teamLeadsPerSupervisor : ℕ
  workersPerTeamLead : ℕ

/-- Calculates the total number of workers in a company given its hierarchy -/
def totalWorkers (c : CompanyHierarchy) : ℕ :=
  c.supervisors * c.teamLeadsPerSupervisor * c.workersPerTeamLead

/-- Theorem stating that a company with 13 supervisors, 3 team leads per supervisor,
    and 10 workers per team lead has 390 workers in total -/
theorem company_workers_count :
  let c : CompanyHierarchy := {
    supervisors := 13,
    teamLeadsPerSupervisor := 3,
    workersPerTeamLead := 10
  }
  totalWorkers c = 390 := by
  sorry


end NUMINAMATH_CALUDE_company_workers_count_l1825_182574


namespace NUMINAMATH_CALUDE_gina_netflix_minutes_l1825_182540

/-- Represents the number of times Gina chooses what to watch compared to her sister -/
def gina_choice_ratio : ℕ := 3

/-- Represents the number of times Gina's sister chooses what to watch -/
def sister_choice_ratio : ℕ := 1

/-- The number of shows Gina's sister watches per week -/
def sister_shows_per_week : ℕ := 24

/-- The length of each show in minutes -/
def show_length : ℕ := 50

/-- Theorem stating that Gina chooses 3600 minutes of Netflix per week -/
theorem gina_netflix_minutes :
  (sister_shows_per_week * gina_choice_ratio * show_length) / (gina_choice_ratio + sister_choice_ratio) = 3600 :=
sorry

end NUMINAMATH_CALUDE_gina_netflix_minutes_l1825_182540


namespace NUMINAMATH_CALUDE_journey_time_difference_l1825_182598

/-- Proves that the difference in arrival times is 15 minutes for a 70 km journey 
    when comparing speeds of 40 km/hr (on time) and 35 km/hr (late). -/
theorem journey_time_difference (distance : ℝ) (speed_on_time speed_late : ℝ) : 
  distance = 70 ∧ speed_on_time = 40 ∧ speed_late = 35 →
  (distance / speed_late - distance / speed_on_time) * 60 = 15 := by
sorry

end NUMINAMATH_CALUDE_journey_time_difference_l1825_182598


namespace NUMINAMATH_CALUDE_song_count_difference_l1825_182564

/- Define the problem parameters -/
def total_days_in_june : ℕ := 30
def weekend_days : ℕ := 8
def vivian_daily_songs : ℕ := 10
def total_monthly_songs : ℕ := 396

/- Calculate the number of days they played songs -/
def playing_days : ℕ := total_days_in_june - weekend_days

/- Calculate Vivian's total songs for the month -/
def vivian_monthly_songs : ℕ := vivian_daily_songs * playing_days

/- Calculate Clara's total songs for the month -/
def clara_monthly_songs : ℕ := total_monthly_songs - vivian_monthly_songs

/- Calculate Clara's daily song count -/
def clara_daily_songs : ℕ := clara_monthly_songs / playing_days

/- Theorem to prove -/
theorem song_count_difference : vivian_daily_songs - clara_daily_songs = 2 := by
  sorry

end NUMINAMATH_CALUDE_song_count_difference_l1825_182564


namespace NUMINAMATH_CALUDE_domain_subset_iff_a_range_l1825_182578

theorem domain_subset_iff_a_range (a : ℝ) (h : a < 1) :
  (∀ x, (x - a - 1) * (2 * a - x) > 0 → x ∈ (Set.Iic (-1) ∪ Set.Ici 1)) ↔
  a ∈ (Set.Iic (-2) ∪ Set.Icc (1/2) 1) :=
sorry

end NUMINAMATH_CALUDE_domain_subset_iff_a_range_l1825_182578


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1825_182569

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : S ∩ (U \ T) = {1, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1825_182569


namespace NUMINAMATH_CALUDE_b_age_is_27_l1825_182571

/-- The ages of four people A, B, C, and D. -/
structure Ages where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The conditions of the problem. -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.a + ages.b + ages.c + ages.d) / 4 = 28 ∧
  (ages.a + ages.c) / 2 = 29 ∧
  (2 * ages.b + 3 * ages.d) / 5 = 27 ∧
  ages.a = 1.1 * (ages.a / 1.1) ∧
  ages.c = 1.1 * (ages.c / 1.1) ∧
  ages.b = 1.15 * (ages.b / 1.15) ∧
  ages.d = 1.15 * (ages.d / 1.15)

/-- The theorem stating that given the problem conditions, B's age is 27. -/
theorem b_age_is_27 (ages : Ages) (h : problem_conditions ages) : ages.b = 27 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_27_l1825_182571


namespace NUMINAMATH_CALUDE_distance_from_origin_l1825_182536

theorem distance_from_origin (x y : ℝ) (h1 : y = 20) 
  (h2 : Real.sqrt ((x - 2)^2 + (y - 15)^2) = 15) (h3 : x > 2) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (604 + 40 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1825_182536


namespace NUMINAMATH_CALUDE_max_intersections_seven_segments_l1825_182513

/-- A closed polyline with a given number of segments. -/
structure ClosedPolyline :=
  (segments : ℕ)

/-- The maximum number of self-intersection points for a closed polyline. -/
def max_self_intersections (p : ClosedPolyline) : ℕ :=
  (p.segments * (p.segments - 3)) / 2

/-- Theorem: The maximum number of self-intersection points in a closed polyline with 7 segments is 14. -/
theorem max_intersections_seven_segments :
  ∃ (p : ClosedPolyline), p.segments = 7 ∧ max_self_intersections p = 14 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_seven_segments_l1825_182513
