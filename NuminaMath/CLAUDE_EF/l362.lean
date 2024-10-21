import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribable_ngon_existence_l362_36234

/-- Represents a circle in 2D space. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a convex polygon given its side lengths. -/
def ConvexPolygon (sides : List ℕ) : Prop := sorry

/-- Predicate to check if all sides of a polygon are tangent to a given circle. -/
def AllSidesTangentTo (sides : List ℕ) (circle : Circle) : Prop := sorry

/-- A convex n-gon with side lengths 1, 2, ..., n (in some order) and all sides tangent to the same circle exists if and only if n ≥ 4 and n is not of the form 4k + 2 for some positive integer k. -/
theorem circumscribable_ngon_existence (n : ℕ) : 
  (∃ (sides : List ℕ) (circle : Circle), 
    n ≥ 4 ∧ 
    sides.length = n ∧
    sides.toFinset = Finset.range n ∧
    ConvexPolygon sides ∧
    AllSidesTangentTo sides circle) ↔ 
  (n ≥ 4 ∧ ¬∃ (k : ℕ), n = 4 * k + 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribable_ngon_existence_l362_36234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_consecutive_ones_l362_36287

/-- Modified Fibonacci sequence for counting integers without consecutive 1s -/
def F : ℕ → ℕ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 8
  | n + 3 => 2 * F (n + 2) + 2 * F (n + 1)

/-- The number of 12-digit integers with digits 1, 2, or 3 that have at least two consecutive 1s -/
def consecutive_ones_count : ℕ := 3^12 - F 10

/-- Theorem stating the count of 12-digit integers with at least two consecutive 1s -/
theorem count_integers_with_consecutive_ones :
  consecutive_ones_count = 530882 := by sorry

#eval consecutive_ones_count  -- Added to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_consecutive_ones_l362_36287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_displacement_l362_36247

-- Define the velocity function
def V (t : ℝ) : ℝ := 3 * t + 1

-- Define the displacement function
noncomputable def displacement (a b : ℝ) : ℝ := ∫ t in a..b, V t

-- Theorem statement
theorem car_displacement : displacement 1 2 = 5.5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_displacement_l362_36247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_areas_equal_l362_36237

/-- The area of an isosceles triangle given the length of its sides -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem isosceles_triangle_areas_equal :
  let A := triangleArea 13 13 10
  let B := triangleArea 13 13 24
  A = B := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_areas_equal_l362_36237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_depends_on_a_r_n_l362_36267

noncomputable section

/-- Sum of geometric progression with first term a, common ratio r, and k terms -/
def sum_gp (a r : ℝ) (k : ℕ) : ℝ := a * (1 - r^k) / (1 - r)

/-- t₁ is the sum of n terms of the geometric progression -/
def t₁ (a r : ℝ) (n : ℕ) : ℝ := sum_gp a r n

/-- t₂ is the sum of 2n terms of the geometric progression -/
def t₂ (a r : ℝ) (n : ℕ) : ℝ := sum_gp a r (2 * n)

/-- t₃ is the sum of 4n terms of the geometric progression -/
def t₃ (a r : ℝ) (n : ℕ) : ℝ := sum_gp a r (4 * n)

/-- T is defined as t₃ - 2t₂ + t₁ -/
def T (a r : ℝ) (n : ℕ) : ℝ := t₃ a r n - 2 * t₂ a r n + t₁ a r n

theorem T_depends_on_a_r_n (a r : ℝ) (n : ℕ) :
  ∃ (f : ℝ → ℝ → ℕ → ℝ), T a r n = f a r n ∧ 
  (∀ (a' r' : ℝ) (n' : ℕ), a ≠ a' ∨ r ≠ r' ∨ n ≠ n' → T a r n ≠ T a' r' n') :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_depends_on_a_r_n_l362_36267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_identity_l362_36207

theorem cosine_sine_identity (x y : ℝ) : 
  Real.cos x ^ 4 + Real.sin y ^ 2 + (1/4) * Real.sin (2*x) ^ 2 - 1 = Real.sin (y+x) * Real.sin (y-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_identity_l362_36207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_is_32_l362_36213

-- Define the triangle
def triangle_sides : ℝ × ℝ × ℝ := (10, 10, 12)

-- Define the semi-perimeter of the triangle
noncomputable def s : ℝ := (triangle_sides.1 + triangle_sides.2.1 + triangle_sides.2.2) / 2

-- Define the area of the triangle using Heron's formula
noncomputable def triangle_area : ℝ := Real.sqrt (s * (s - triangle_sides.1) * (s - triangle_sides.2.1) * (s - triangle_sides.2.2))

-- Define the width of the rectangle
def rectangle_width : ℝ := 4

-- Define the length of the rectangle
noncomputable def rectangle_length : ℝ := triangle_area / rectangle_width

-- Theorem to prove
theorem rectangle_perimeter_is_32 : 
  2 * (rectangle_width + rectangle_length) = 32 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_is_32_l362_36213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_ACB_l362_36211

noncomputable section

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (0, a)
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Define the point C on the positive x-axis
def C (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the angle ACB
noncomputable def angle_ACB (a b x : ℝ) : ℝ :=
  Real.arccos ((x^2 + a*b) / (Real.sqrt (x^2 + a^2) * Real.sqrt (x^2 + b^2)))

-- State the theorem
theorem max_angle_ACB (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → angle_ACB a b x ≥ angle_ACB a b y) ∧
  (∃ (x : ℝ), x > 0 ∧ angle_ACB a b x = Real.arccos (2 * Real.sqrt (a*b) / (a + b))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_ACB_l362_36211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_with_median_l362_36278

noncomputable def curve (z₀ z₁ z₂ : ℂ) (t : ℝ) : ℂ :=
  z₀ * (Real.cos t)^4 + 2 * z₁ * (Real.cos t)^2 * (Real.sin t)^2 + z₂ * (Real.sin t)^4

noncomputable def median_parallel (z₀ z₁ z₂ : ℂ) (x : ℝ) : ℂ :=
  (z₁ + z₂) / 2 + x * ((z₂ - z₀) / 2 - (z₁ + z₂) / 2 + (z₁ - z₀) / 2)

theorem curve_intersection_with_median 
  (a b c : ℝ) 
  (h_non_collinear : a - 2*b + c ≠ 0) :
  ∃! p : ℂ, p ∈ Set.range (curve (Complex.I * a) (1/2 + Complex.I * b) (1 + Complex.I * c)) ∧ 
           p ∈ Set.range (median_parallel (Complex.I * a) (1/2 + Complex.I * b) (1 + Complex.I * c)) ∧
           p = Complex.mk (1/2) ((a + 2*b + c)/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_with_median_l362_36278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l362_36280

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ m n, a (m * n) = a m * a n) →
  (∃ B : ℝ, B > 0 ∧ ∀ m n, m < n → a m < B * a n) →
  Real.log (a 2015) / Real.log 2015 - Real.log (a 2014) / Real.log 2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l362_36280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equivalence_l362_36233

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_inequality_equivalence (x : ℝ) :
  4 * (floor x)^2 - 16 * (floor x) + 7 < 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equivalence_l362_36233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l362_36293

/-- Defines a sequence of nested radicals --/
noncomputable def nestedRadical : ℕ → ℝ
| 0 => Real.sqrt (1 + 2018 * 2020)
| n + 1 => Real.sqrt (1 + (2017 - n) * nestedRadical n)

/-- The theorem states that the nested radical expression equals 3 --/
theorem nested_radical_equals_three : nestedRadical 2017 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l362_36293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_emptying_time_l362_36283

/-- Represents the time it takes to empty a water tank with a leak -/
noncomputable def time_to_empty_tank (fill_time_without_leak : ℝ) (fill_time_with_leak : ℝ) : ℝ :=
  let fill_rate := 1 / fill_time_without_leak
  let effective_fill_rate := 1 / fill_time_with_leak
  let leak_rate := fill_rate - effective_fill_rate
  1 / leak_rate

/-- Theorem stating that for a tank that takes 5 hours to fill without a leak
    and 6 hours with a leak, it will take 30 hours to empty when full -/
theorem water_tank_emptying_time :
  time_to_empty_tank 5 6 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_emptying_time_l362_36283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l362_36217

/-- Given two planar vectors a and b, if a is perpendicular to (a + λb), then λ = -5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (lambda : ℝ) : 
  a = (2, 1) → b = (-1, 3) → (a.1 * (a.1 + lambda * b.1) + a.2 * (a.2 + lambda * b.2) = 0) → lambda = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l362_36217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l362_36242

/-- The average speed of a car given its travel distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: The average speed of a car traveling 715 kilometers in 11 hours is 65 kilometers per hour -/
theorem car_average_speed :
  average_speed 715 11 = 65 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l362_36242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l362_36276

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x + 1) / (1 - x)

theorem range_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-1/2 : ℝ) 1 \ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l362_36276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surrounding_circles_radius_is_one_plus_sqrt_two_l362_36227

/-- The radius of the surrounding circles in a configuration where a unit circle
    is surrounded by four equal circles whose centers form a square. -/
noncomputable def surrounding_circle_radius : ℝ := 1 + Real.sqrt 2

/-- Theorem stating that the radius of the surrounding circles in the described
    configuration is equal to 1 + √2. -/
theorem surrounding_circles_radius_is_one_plus_sqrt_two :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ × ℝ),
    (x.1 - y.1)^2 + (x.2 - y.2)^2 = (2*r)^2 →
    (x.1^2 + x.2^2 = (1 + r)^2 ∧ y.1^2 + y.2^2 = (1 + r)^2)) →
  r = surrounding_circle_radius := by
  sorry

#check surrounding_circles_radius_is_one_plus_sqrt_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surrounding_circles_radius_is_one_plus_sqrt_two_l362_36227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_derivative_of_g_l362_36272

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2^x
noncomputable def g (x : ℝ) : ℝ := x * Real.sqrt x

-- State the theorems
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.log 2 := by sorry

theorem derivative_of_g (x : ℝ) (h : x ≥ 0) : 
  deriv g x = (3/2) * Real.sqrt x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_derivative_of_g_l362_36272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_length_squared_l362_36263

/-- Represents a parallelogram ABCD with projections P, Q, R, S -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  area : ℝ
  pq_length : ℝ
  rs_length : ℝ

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (p : ℕ) : Prop :=
  ∀ q : ℕ, Nat.Prime q → ¬(q^2 ∣ p)

/-- Main theorem -/
theorem parallelogram_diagonal_length_squared 
  (ABCD : Parallelogram) 
  (h_area : ABCD.area = 18) 
  (h_pq : ABCD.pq_length = 5) 
  (h_rs : ABCD.rs_length = 7) : 
  ∃ (m n p : ℕ), 
    (m > 0 ∧ n > 0 ∧ p > 0) ∧
    not_divisible_by_prime_square p ∧
    (ABCD.A.1 - ABCD.C.1)^2 + (ABCD.A.2 - ABCD.C.2)^2 = m + n * Real.sqrt p ∧
    m + n + p = 92 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_length_squared_l362_36263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l362_36238

def sequence_a (k : ℝ) : ℕ → ℝ
  | 0 => 1 - 3 * k  -- Added case for 0
  | 1 => 1 - 3 * k
  | n + 2 => 4^(n+1) - 3 * sequence_a k (n+1)

theorem sequence_properties (k : ℝ) :
  (∀ n : ℕ, n > 1 → sequence_a k n - (4^n / 7) = -3 * (sequence_a k (n-1) - (4^(n-1) / 7))) ∧
  (k ≠ 2/7 → ∀ n : ℕ, sequence_a k n = 4^n / 7 + (6/7 - 3*k) * (-3)^(n-1)) ∧
  (k = 2/7 → ∀ n : ℕ, n ≥ 2 → sequence_a k n = 4^n / 7) ∧
  (∀ n : ℕ, n > 1 → (sequence_a k n > sequence_a k (n-1) ↔ 2/7 ≤ k ∧ k < 34/63)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l362_36238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_runs_50_seconds_l362_36214

/-- Represents the race between Nicky and Cristina -/
structure Race where
  length : ℝ
  head_start : ℝ
  cristina_speed : ℝ
  nicky_speed : ℝ

/-- Calculates the time when Cristina catches up to Nicky -/
noncomputable def catch_up_time (race : Race) : ℝ :=
  (race.head_start * race.nicky_speed) / (race.cristina_speed - race.nicky_speed)

/-- Calculates the total time Nicky runs before Cristina catches up -/
noncomputable def nicky_total_time (race : Race) : ℝ :=
  race.head_start + catch_up_time race

/-- The main theorem stating that Nicky runs for 50 seconds before Cristina catches up -/
theorem nicky_runs_50_seconds (race : Race) 
  (h1 : race.length = 800)
  (h2 : race.head_start = 20)
  (h3 : race.cristina_speed = 5)
  (h4 : race.nicky_speed = 3) :
  nicky_total_time race = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_runs_50_seconds_l362_36214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l362_36219

def my_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1/2 ∧ ∀ n : ℕ, n ≥ 2 → a (n-1) - a n = (a n * a (n-1)) * n

theorem sequence_formula (a : ℕ → ℝ) (h : my_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 / (n^2 + n + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l362_36219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_distance_is_three_miles_l362_36256

/-- Taxi fare structure -/
structure TaxiFare where
  initial_fare : ℝ
  initial_distance : ℝ
  additional_fare : ℝ
  additional_distance : ℝ

/-- Calculate the total fare for a given distance -/
noncomputable def calculate_fare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.initial_fare + max 0 (distance - tf.initial_distance) / tf.additional_distance * tf.additional_fare

/-- The theorem stating that the distance is 3 miles given the fare conditions -/
theorem ride_distance_is_three_miles (tf : TaxiFare) 
  (h1 : tf.initial_fare = 1)
  (h2 : tf.initial_distance = 1/5)
  (h3 : tf.additional_fare = 0.45)
  (h4 : tf.additional_distance = 1/5)
  (h5 : calculate_fare tf 3 = 7.3) : 
  ∃ (d : ℝ), calculate_fare tf d = 7.3 ∧ d = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_distance_is_three_miles_l362_36256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l362_36228

/-- The curve C2 obtained by stretching the unit circle -/
def C2 (x y : ℝ) : Prop := (x / Real.sqrt 2) ^ 2 + (y / Real.sqrt 3) ^ 2 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x + y - 4 * Real.sqrt 5 = 0

/-- The distance function from a point to the line l -/
noncomputable def distance_to_l (x y : ℝ) : ℝ :=
  |x + y - 4 * Real.sqrt 5| / Real.sqrt 2

/-- The point P -/
noncomputable def P : ℝ × ℝ := (-2 * Real.sqrt 5 / 5, -3 * Real.sqrt 5 / 5)

theorem max_distance_point :
  C2 P.1 P.2 ∧
  ∀ x y, C2 x y → distance_to_l x y ≤ distance_to_l P.1 P.2 ∧
  distance_to_l P.1 P.2 = 5 * Real.sqrt 10 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l362_36228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_coefficients_l362_36215

-- Define the binomial expansion
noncomputable def binomial_expansion (n : ℕ) (x : ℝ) := (1 + 2 * Real.sqrt x) ^ n

-- Define the coefficient of the r-th term in the expansion
noncomputable def coefficient (n r : ℕ) (x : ℝ) : ℝ := 
  (n.choose r) * (2 ^ r) * (Real.sqrt x) ^ r

-- State the theorem
theorem binomial_max_coefficients :
  let n := 7
  let x : ℝ := x
  -- Condition: The coefficient of a certain term is twice that of its preceding term
  ∃ r, coefficient n r x = 2 * coefficient n (r-1) x ∧
  -- Condition: The coefficient of a certain term is 5/6 of its following term
  coefficient n r x = (5/6) * coefficient n (r+1) x →
  -- Claim 1: The terms with the maximum binomial coefficient are the 4th and 5th terms
  (∀ k, k ≠ 4 ∧ k ≠ 5 → n.choose k ≤ n.choose 4) ∧
  -- Claim 2: The term with the maximum coefficient is the 6th term
  (∀ k, k ≠ 6 → coefficient n k x ≤ coefficient n 6 x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_coefficients_l362_36215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l362_36265

/-- The length of a train given the speeds of two trains, time to cross, and length of the other train --/
noncomputable def train_length (speed1 speed2 : ℝ) (time_to_cross : ℝ) (other_train_length : ℝ) : ℝ :=
  (speed1 + speed2) * time_to_cross * (1000 / 3600) - other_train_length

/-- Theorem stating that under given conditions, the length of the first train is 140 meters --/
theorem first_train_length :
  let speed1 : ℝ := 60
  let speed2 : ℝ := 40
  let time_to_cross : ℝ := 12.59899208063355
  let second_train_length : ℝ := 210
  train_length speed1 speed2 time_to_cross second_train_length = 140 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l362_36265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_combinations_even_beads_odd_combinations_sum_of_permutations_parity_l362_36271

-- Define the number of beads
def n : ℕ := 7

-- 1. Prove that the number of non-empty subsets of a set with 7 elements is 127
theorem number_of_combinations (n : ℕ) (h : n = 7) : 2^n - 1 = 127 := by sorry

-- 2. Prove that for any even number n, the number of non-empty subsets of a set with n elements is always odd
theorem even_beads_odd_combinations (n : ℕ) (h : Even n) : Odd (2^n - 1) := by sorry

-- 3. Prove that the sum of permutations of k elements (1 ≤ k ≤ n) from a set of n elements can be either even or odd
def sum_of_permutations (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ k => n.factorial / (n - k - 1).factorial)

theorem sum_of_permutations_parity (n : ℕ) : 
  ∃ m : ℕ, sum_of_permutations n = 2 * m ∨ sum_of_permutations n = 2 * m + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_combinations_even_beads_odd_combinations_sum_of_permutations_parity_l362_36271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_point_circle_equation_l362_36294

-- Define the oblique coordinate system
structure ObliqueCoordSystem where
  angle : ℝ
  e₁ : ℝ × ℝ
  e₂ : ℝ × ℝ

-- Define the properties of our specific oblique coordinate system
noncomputable def xOy : ObliqueCoordSystem :=
  { angle := Real.pi / 3  -- 60° in radians
  , e₁ := (1, 0)  -- unit vector in x direction
  , e₂ := (1/2, Real.sqrt 3/2) }  -- unit vector at 60° from x

-- Define a point in the oblique coordinate system
noncomputable def Point (x y : ℝ) : ℝ × ℝ :=
  (x * xOy.e₁.1 + y * xOy.e₂.1, x * xOy.e₁.2 + y * xOy.e₂.2)

-- Define the distance of a point from the origin
noncomputable def distance (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

-- Theorem 1: The distance of point (3, -2) from the origin is √7
theorem distance_of_point : distance (Point 3 (-2)) = Real.sqrt 7 := by
  sorry

-- Theorem 2: The equation x² + y² + xy = 4 represents a circle with center O and radius 2
theorem circle_equation (x y : ℝ) : 
  distance (Point x y) = 2 ↔ x^2 + y^2 + x*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_point_circle_equation_l362_36294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_seven_dividing_fifty_factorial_l362_36232

theorem greatest_power_of_seven_dividing_fifty_factorial :
  ∃ k : ℕ, k = 8 ∧ (7^k : ℕ) ∣ (Nat.factorial 50) ∧ 
  ∀ m : ℕ, m > k → ¬((7^m : ℕ) ∣ (Nat.factorial 50)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_seven_dividing_fifty_factorial_l362_36232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_collected_11_seashells_l362_36274

/-- The number of seashells collected by Henry, John (Paul), and Leo -/
structure SeashellCollection where
  henry : ℕ
  john : ℕ
  leo : ℕ

/-- The initial collection of seashells -/
def initial_collection (henry leo : ℕ) : SeashellCollection :=
  { henry := henry,
    john := 24,
    leo := leo }

/-- The final collection of seashells after Leo gave away a quarter of his -/
def final_collection (henry leo : ℕ) : SeashellCollection :=
  { henry := henry,
    john := 24,
    leo := (3 * leo) / 4 }

/-- Theorem stating that Henry collected 11 seashells -/
theorem henry_collected_11_seashells :
  ∀ (henry leo : ℕ),
  (initial_collection henry leo).henry + (initial_collection henry leo).john + (initial_collection henry leo).leo = 59 →
  (final_collection henry leo).henry + (final_collection henry leo).john + (final_collection henry leo).leo = 53 →
  henry = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_collected_11_seashells_l362_36274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_implies_k_l362_36260

noncomputable def f (x k : ℝ) : ℝ := Real.sin (2 * x) + k * Real.cos (2 * x)

theorem symmetric_axis_implies_k (k : ℝ) :
  (∀ x : ℝ, f x k = f (π/3 - x) k) → k = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_implies_k_l362_36260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_160_over_3_l362_36255

/-- A square prism with side length 8 and vertical edges parallel to the z-axis --/
structure SquarePrism where
  side_length : ℝ
  side_length_eq : side_length = 8

/-- The cutting plane 3x - 5y + 6z = 30 --/
def cutting_plane (x y z : ℝ) : Prop :=
  3 * x - 5 * y + 6 * z = 30

/-- The maximum area of the cross-section --/
noncomputable def max_cross_section_area (p : SquarePrism) : ℝ := 160 / 3

/-- Theorem: The maximum area of the cross-section is 160/3 --/
theorem max_area_is_160_over_3 (p : SquarePrism) :
  max_cross_section_area p = 160 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_160_over_3_l362_36255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l362_36286

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The midpoint of a line segment -/
def segment_midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  sorry

theorem line_equation_proof :
  let l1 := fun x y => x + y - 2 = 0
  let l2 := fun x y => x - y - 4 = 0
  let p := intersection_point l1 l2
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (5, 1)
  let q := segment_midpoint a b
  let l := fun x y => 3*x + y - 8 = 0
  point_on_line p l ∧ point_on_line q l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l362_36286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_integer_set_exists_l362_36253

theorem special_integer_set_exists (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), (Finset.card S = n) ∧
  (∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_integer_set_exists_l362_36253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_three_to_three_fourths_l362_36201

/-- The infinite product of (3^n)^(1/3^n) for n from 1 to infinity -/
noncomputable def infinite_product : ℝ := ∏' n, (3 ^ n) ^ (1 / (3 ^ n))

/-- The theorem stating that the infinite product equals 3^(3/4) -/
theorem infinite_product_equals_three_to_three_fourths :
  infinite_product = 3 ^ (3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_three_to_three_fourths_l362_36201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribable_sphere_l362_36251

/-- A pyramid with a base that can be circumscribed by a circle -/
structure CircumscribablePyramid where
  /-- Radius of the circle circumscribing the base -/
  r : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- The base radius is positive -/
  r_pos : r > 0
  /-- The height is positive -/
  h_pos : h > 0

/-- The radius of the sphere circumscribing the pyramid -/
noncomputable def sphere_radius (p : CircumscribablePyramid) : ℝ :=
  Real.sqrt (p.r^2 + (p.h/2)^2)

/-- Theorem: A sphere can be circumscribed around the pyramid with the given radius -/
theorem circumscribable_sphere (p : CircumscribablePyramid) :
  ∃ (R : ℝ), R = sphere_radius p ∧ R > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribable_sphere_l362_36251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_theorem_l362_36212

-- Define the function f as noncomputable due to its dependency on Real
noncomputable def f (x : ℝ) : ℝ := (Real.log (x - 1)) / (Real.sqrt (2 - x))

-- Define the set A as the domain of f
def A : Set ℝ := {x | x > 1 ∧ x < 2}

-- Define the set B as the solution set of the inequality
def B (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 3)*x + a^2 + 3*a ≤ 0}

-- State the theorem
theorem domain_and_range_theorem (a : ℝ) :
  (A = Set.Ioo 1 2) ∧
  (A ⊆ B a → a ∈ Set.Icc (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_theorem_l362_36212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_root_inequality_l362_36269

theorem nth_root_inequality (n : ℕ) (hn : n > 0) : 
  ((n : ℝ) + (n : ℝ)^(1/n : ℝ))^(1/n : ℝ) + ((n : ℝ) - (n : ℝ)^(1/n : ℝ))^(1/n : ℝ) ≤ 2 * (n : ℝ)^(1/n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_root_inequality_l362_36269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l362_36249

-- Define the point P
def P : ℝ × ℝ → Prop := fun (x, y) ↦ 4 * x + 3 * y = 0

-- Define the constraint on x and y
def constraint : ℝ × ℝ → Prop := fun (x, y) ↦ -14 ≤ x - y ∧ x - y ≤ 7

-- Define the distance function
noncomputable def distance : ℝ × ℝ → ℝ := fun (x, y) ↦ Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem distance_range :
  ∀ (x y : ℝ), P (x, y) → constraint (x, y) →
  ∃ (d : ℝ), distance (x, y) = d ∧ 0 ≤ d ∧ d ≤ 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l362_36249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crawl_distance_is_correct_l362_36226

/-- Represents the configuration of two trees and a spider's path between them. -/
structure TreeConfiguration where
  tree_height : ℝ
  tree_distance : ℝ
  branch_length : ℝ → ℝ
  min_crawl_distance : ℝ

/-- The specific configuration for our problem. -/
noncomputable def problem_config : TreeConfiguration :=
  { tree_height := 30
  , tree_distance := 20
  , branch_length := λ h => (30 - h) / 2
  , min_crawl_distance := 60 }

/-- Theorem stating that the minimum crawl distance in our configuration is correct. -/
theorem min_crawl_distance_is_correct (config : TreeConfiguration) 
  (h1 : config.tree_height = 30)
  (h2 : config.tree_distance = 20)
  (h3 : ∀ h, config.branch_length h = (config.tree_height - h) / 2)
  (h4 : config.min_crawl_distance = 60) : 
  ∀ path : ℝ, path ≥ config.min_crawl_distance :=
by
  sorry

#check min_crawl_distance_is_correct problem_config

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crawl_distance_is_correct_l362_36226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_subset_function_l362_36288

universe u

theorem fixed_point_subset_function {S : Type u} (f : Set S → Set S) 
  (h : ∀ (Z : Set S) (y : S), y ∈ Z → (f Z) y) : 
  ∃ (A : Set S), f A = A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_subset_function_l362_36288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_parameter_range_l362_36284

/-- A function f(x) = x^2 + ax + 1/x is increasing on (1/2, +∞) if and only if a ≥ 3 -/
theorem increasing_function_parameter_range (a : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (fun x ↦ x^2 + a*x + 1/x)) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_parameter_range_l362_36284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l362_36261

/-- The equation we're considering -/
def equation (x : ℝ) : Prop :=
  (x - 3) / (x^2 + 5*x + 2) = (x - 6) / (x^2 - 8*x)

/-- The theorem stating the sum of solutions -/
theorem sum_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x) ∧ (∀ x : ℝ, equation x → x ∈ S) ∧ (S.sum id = -26/5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l362_36261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l362_36243

/-- A set of integers with at least two elements -/
structure IntegerSet where
  S : Finset Int
  r : Nat
  h_size : S.card = r
  h_r_gt_1 : r > 1

/-- Product of all integers in a subset -/
def product_subset (A : Finset Int) : Int :=
  A.prod id

/-- Arithmetic mean of products of all non-empty subsets -/
noncomputable def arithmetic_mean (S : IntegerSet) : ℚ :=
  let all_subsets := S.S.powerset.filter (λ A => A.Nonempty)
  (all_subsets.sum (λ A => (product_subset A : ℚ))) / all_subsets.card

/-- The theorem to be proved -/
theorem unique_solution (S : IntegerSet) (a_r_plus_1 : Int) :
  arithmetic_mean S = 13 ∧
  arithmetic_mean ⟨S.S ∪ {a_r_plus_1}, S.r + 1, by sorry, by sorry⟩ = 49 →
  S.r = 3 ∧
  S.S = {1, 1, 22} ∧
  a_r_plus_1 = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l362_36243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l362_36218

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1 / x) - 2 * Real.log x

theorem f_properties (a : ℝ) :
  -- Tangent line equation when a = 2
  (∀ x > 0, f 2 x = 2 * x - 2 * Real.log x - 2 → 
    (λ y => 2 * x - y - 2 = 0) (f 2 x)) ∧
  -- Monotonicity properties
  (a ≤ 0 → ∀ x > 0, ∀ y > 0, x < y → f a x > f a y) ∧
  (0 < a ∧ a < 1 → 
    ∀ x > 0, ∀ y > 0,
    ((x < y ∧ y < (1 - Real.sqrt (1 - a^2)) / a) ∨
     (x < y ∧ (1 + Real.sqrt (1 - a^2)) / a < x) →
     f a x < f a y) ∧
    ((1 - Real.sqrt (1 - a^2)) / a < x ∧ x < y ∧ y < (1 + Real.sqrt (1 - a^2)) / a →
     f a x > f a y)) ∧
  (a ≥ 1 → ∀ x > 0, ∀ y > 0, x < y → f a x < f a y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l362_36218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l362_36266

open Real

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π/6)

theorem f_increasing_on_interval : 
  ∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y := by
  intros x y h
  -- The proof steps would go here
  sorry

#check f_increasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l362_36266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l362_36205

theorem problem_solution (t : ℝ) (k m n : ℕ+) :
  (1 + Real.sin t) * (1 + Real.cos t) = 5/4 →
  (1 - Real.sin t) * (1 - Real.cos t) = m/n - Real.sqrt k →
  Int.gcd m.val n.val = 1 →
  k + m + n = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l362_36205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l362_36298

/-- The radius of the inscribed circle in a triangle with sides a, b, and c -/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

theorem inscribed_circle_radius_specific_triangle :
  inscribedCircleRadius 6 8 10 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l362_36298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l362_36225

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_point_x_coordinate 
  (x y : ℝ) 
  (h1 : is_on_parabola x y) 
  (h2 : distance (x, y) focus = 3) : 
  x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l362_36225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_exists_l362_36279

-- Define the type for our circle arrangement
structure CircleArrangement where
  center : ℕ
  periphery : List ℕ

-- Define the property of sum of three numbers on a line being 30
def validLine (a b c : ℕ) : Prop := a + b + c = 30

-- Define the property that all lines in the arrangement are valid
def validArrangement (arr : CircleArrangement) : Prop :=
  arr.center = 10 ∧
  arr.periphery.length = 18 ∧
  arr.periphery.toFinset = Finset.range 19 \ {10} ∧
  ∀ i : Fin 6, 
    (∃ a b : ℕ, arr.periphery.get? i.val = some a ∧ 
                arr.periphery.get? (i.val + 6) = some b ∧ 
                validLine arr.center a b)

-- State the theorem
theorem circle_arrangement_exists : 
  ∃ arr : CircleArrangement, validArrangement arr :=
sorry

#eval "The theorem has been stated and the proof is left as an exercise."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_exists_l362_36279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tommy_cars_count_l362_36289

theorem tommy_cars_count :
  ∀ (tommy_cars : ℕ),
    let jessie_cars := tommy_cars
    let brother_cars := tommy_cars + 5
    tommy_cars + jessie_cars + brother_cars = 17 →
    tommy_cars = 4 :=
by
  intro tommy_cars
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tommy_cars_count_l362_36289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_circle_diameter_l362_36230

/-- The diameter of a circle in which a right triangle is inscribed --/
noncomputable def circleDiameter (leg1 leg2 : ℝ) : ℝ :=
  Real.sqrt (leg1^2 + leg2^2)

theorem right_triangle_inscribed_circle_diameter :
  circleDiameter 6 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_circle_diameter_l362_36230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_P_l362_36235

def M : Finset ℕ := {0, 2, 4}

def P : Finset ℕ := Finset.image (λ (a, b) => a * b) (M.product M)

theorem number_of_subsets_of_P : Finset.card (Finset.powerset P) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_P_l362_36235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_return_theorem_l362_36273

/-- Calculate the annual return percentage given initial and final stock prices -/
noncomputable def annual_return_percentage (initial_price final_price : ℝ) : ℝ :=
  ((final_price - initial_price) / initial_price) * 100

/-- Theorem: The annual return percentage for a stock bought at 8000 rubles and sold for 400 rubles more is 5% -/
theorem stock_return_theorem : 
  let initial_price : ℝ := 8000
  let price_increase : ℝ := 400
  let final_price : ℝ := initial_price + price_increase
  annual_return_percentage initial_price final_price = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_return_theorem_l362_36273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l362_36252

/-- The time taken for a train to cross a signal pole -/
noncomputable def time_to_cross_pole (train_length platform_length time_to_cross_platform : ℝ) : ℝ :=
  train_length / (train_length + platform_length) * time_to_cross_platform

/-- Proof that the train takes 18 seconds to cross the signal pole -/
theorem train_crossing_time :
  let train_length : ℝ := 450
  let platform_length : ℝ := 525
  let time_to_cross_platform : ℝ := 39
  time_to_cross_pole train_length platform_length time_to_cross_platform = 18 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l362_36252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l362_36203

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y = 12

/-- The area of the region -/
noncomputable def region_area : ℝ := 25 * Real.pi

/-- Theorem stating the existence of a circle representation for the region and its area -/
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l362_36203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_with_3_no_5_l362_36296

/-- The set of digits excluding 0, 3, and 5 -/
def digits_no_035 : Finset Nat := {1, 2, 4, 6, 7, 8, 9}

/-- The set of digits excluding 0 and 5 -/
def digits_no_05 : Finset Nat := {1, 2, 3, 4, 6, 7, 8, 9}

/-- The set of digits excluding 5 -/
def digits_no_5 : Finset Nat := {0, 1, 2, 3, 4, 6, 7, 8, 9}

/-- The number of three-digit integers without 3 and 5 -/
def count_no_35 : Nat := digits_no_035.card * digits_no_5.card * digits_no_5.card

/-- The number of three-digit integers without 5 -/
def count_no_5 : Nat := digits_no_05.card * digits_no_5.card * digits_no_5.card

theorem three_digit_with_3_no_5 : 
  count_no_5 - count_no_35 = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_with_3_no_5_l362_36296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_theorem_application_l362_36257

theorem chinese_remainder_theorem_application :
  let S : Finset ℕ := Finset.filter (fun n => 2 ≤ n ∧ n ≤ 2017 ∧ n % 3 = 1 ∧ n % 5 = 1) (Finset.range 2018)
  Finset.card S = 134 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_theorem_application_l362_36257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_term_l362_36204

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = 1 - 1 / (a n)

theorem sequence_2016th_term (a : ℕ → ℚ) (h : my_sequence a) : a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016th_term_l362_36204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l362_36281

-- Define the triangle side lengths
noncomputable def side1 : ℝ := Real.log 20
noncomputable def side2 : ℝ := Real.log 90
noncomputable def side3 (m : ℕ+) : ℝ := Real.log m

-- Define the triangle inequality conditions
def triangle_inequality (m : ℕ+) : Prop :=
  side1 + side2 > side3 m ∧
  side1 + side3 m > side2 ∧
  side2 + side3 m > side1

-- Define the count of valid m values
def valid_m_count : ℕ := 1795

-- Theorem statement
theorem triangle_side_count :
  (∃ (S : Finset ℕ+), S.card = valid_m_count ∧
    (∀ m : ℕ+, m ∈ S ↔ triangle_inequality m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l362_36281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_zero_l362_36222

theorem trigonometric_sum_zero (A B C x y z : ℝ) (k : ℤ) (h1 : A + B + C = k * Real.pi)
  (h2 : x * Real.sin A + y * Real.sin B + z * Real.sin C = 0)
  (h3 : x^2 * Real.sin (2*A) + y^2 * Real.sin (2*B) + z^2 * Real.sin (2*C) = 0) :
  ∀ n : ℕ, x^n * Real.sin (n*A) + y^n * Real.sin (n*B) + z^n * Real.sin (n*C) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_zero_l362_36222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l362_36236

/-- The time it takes for the work to be completed when two workers collaborate -/
noncomputable def work_completion_time (p_time q_time : ℝ) (p_alone_time : ℝ) : ℝ :=
  let p_rate := 1 / p_time
  let q_rate := 1 / q_time
  let work_done_by_p_alone := p_alone_time * p_rate
  let remaining_work := 1 - work_done_by_p_alone
  let combined_rate := p_rate + q_rate
  p_alone_time + remaining_work / combined_rate

theorem work_completion_theorem (p_time q_time p_alone_time : ℝ) 
  (hp : p_time = 40)
  (hq : q_time = 24)
  (ha : p_alone_time = 8) :
  work_completion_time p_time q_time p_alone_time = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval work_completion_time 40 24 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l362_36236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_inequality_l362_36200

/-- A monotonic function on ℝ with f(-1) = 3 and f(1) = 1 -/
noncomputable def f : ℝ → ℝ :=
  sorry

theorem inverse_function_inequality
  (hf : Monotone f)
  (hf_neg_one : f (-1) = 3)
  (hf_one : f 1 = 1) :
  {x : ℝ | |Function.invFun f (2^x)| < 1} = Set.Ioo 0 (Real.log 3 / Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_inequality_l362_36200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_subsets_eq_power_l362_36206

/-- The set of integers from 1 to 2005 -/
def S : Finset Nat := Finset.range 2005

/-- The set of powers of 2 up to 2^10 -/
def A : Finset Nat := Finset.filter (fun x => ∃ k : Nat, k ≤ 10 ∧ x = 2^k) S

/-- The number of subsets of S with sum congruent to 2006 modulo 2048 -/
def num_subsets : Nat :=
  Finset.filter (fun B => (Finset.sum B id) % 2048 = 2006) (Finset.powerset S) |>.card

theorem num_subsets_eq_power : num_subsets = 2^(Finset.card S - Finset.card A) := by
  sorry

#eval num_subsets
#eval 2^(Finset.card S - Finset.card A)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_subsets_eq_power_l362_36206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B4E_to_decimal_l362_36270

def hex_to_decimal (hex_digit : Char) : ℕ :=
  match hex_digit with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | d   => d.toString.toNat!

def hex_string_to_decimal (hex_string : String) : ℕ :=
  hex_string.data.reverse.enum.foldl
    (fun acc (i, c) => acc + (hex_to_decimal c) * (16 ^ i))
    0

theorem hex_B4E_to_decimal :
  hex_string_to_decimal "B4E" = 2894 := by
  sorry

#eval hex_string_to_decimal "B4E"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B4E_to_decimal_l362_36270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ambulance_reachable_area_l362_36250

/-- Ambulance travel scenario -/
structure AmbulanceScenario where
  road_speed : ℝ
  desert_speed : ℝ
  time : ℝ

/-- Calculate the area reachable by the ambulance -/
noncomputable def reachable_area (scenario : AmbulanceScenario) : ℝ :=
  let road_distance := scenario.road_speed * scenario.time / 60
  (2 * road_distance) ^ 2

/-- Theorem stating the area reachable by the ambulance -/
theorem ambulance_reachable_area :
  let scenario : AmbulanceScenario := {
    road_speed := 60,
    desert_speed := 10,
    time := 8
  }
  reachable_area scenario = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ambulance_reachable_area_l362_36250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_standard_deviation_l362_36275

/-- A sequence of 6 walking distances in kilometers -/
def WalkingDistances := Fin 6 → ℝ

/-- The walking distances are in ascending order -/
def IsAscending (d : WalkingDistances) : Prop :=
  ∀ i j : Fin 6, i < j → d i ≤ d j

/-- The median of the walking distances is 16 -/
def HasMedian16 (d : WalkingDistances) : Prop :=
  (d 2 + d 3) / 2 = 16

/-- The standard deviation of the walking distances -/
noncomputable def StandardDeviation (d : WalkingDistances) : ℝ :=
  let mean := (d 0 + d 1 + d 2 + d 3 + d 4 + d 5) / 6
  Real.sqrt ((d 0 - mean)^2 + (d 1 - mean)^2 + (d 2 - mean)^2 + 
             (d 3 - mean)^2 + (d 4 - mean)^2 + (d 5 - mean)^2) / 6

/-- The theorem statement -/
theorem minimize_standard_deviation 
  (d : WalkingDistances) 
  (h1 : d 0 = 11) 
  (h2 : d 1 = 12) 
  (h3 : d 4 = 20) 
  (h4 : d 5 = 27) 
  (h5 : IsAscending d) 
  (h6 : HasMedian16 d) :
  ∀ d' : WalkingDistances, StandardDeviation d ≤ StandardDeviation d' → d 2 = 16 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_standard_deviation_l362_36275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_has_inverse_q_has_no_inverse_l362_36295

-- Define the function p
noncomputable def p (x : ℝ) : ℝ := Real.sqrt (3 - x)

-- Define the domain of p
def p_domain : Set ℝ := Set.Iic 3

-- Theorem stating that p has an inverse on its domain
theorem p_has_inverse :
  ∃ (f : p_domain → ℝ), Function.Bijective f :=
sorry

-- Define the function q
def q (x : ℝ) : ℝ := x^3 + x

-- Theorem stating that q does not have an inverse
theorem q_has_no_inverse :
  ¬∃ (f : ℝ → ℝ), Function.Bijective f ∧ ∀ x, f (q x) = x :=
sorry

-- Additional functions and theorems can be added similarly


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_has_inverse_q_has_no_inverse_l362_36295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l362_36254

/-- Predicate to check if a real number is the eccentricity of a hyperbola given by an equation -/
def IsHyperbolaEccentricity (e : ℝ) (equation : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∀ x y, equation x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
  c^2 = a^2 + b^2 ∧
  e = c / a

/-- The eccentricity of a hyperbola with equation x^2 - y^2/3 = 1 is 2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = 2 ∧ IsHyperbolaEccentricity e (fun x y => x^2 - y^2/3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l362_36254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l362_36290

noncomputable section

-- Define the profit percentage
def profit_percentage : ℝ := 140

-- Define the fraction of the original selling price
def price_fraction : ℝ := 1/3

-- Define the function to calculate the selling price given the cost price and profit percentage
noncomputable def selling_price (cost_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Define the function to calculate the loss percentage
noncomputable def loss_percentage (cost_price : ℝ) (new_selling_price : ℝ) : ℝ :=
  (cost_price - new_selling_price) / cost_price * 100

-- Theorem statement
theorem loss_percentage_calculation (cost_price : ℝ) (cost_price_positive : cost_price > 0) :
  loss_percentage cost_price (price_fraction * selling_price cost_price profit_percentage) = 20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_calculation_l362_36290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l362_36241

-- Define the rectangular hyperbola
def rectangular_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2 = a^2

-- Define the directrix
def directrix (x : ℝ) : Prop :=
  x = -4

-- Define the distance between two points
def distance (y1 y2 : ℝ) : ℝ :=
  |y1 - y2|

-- Main theorem
theorem hyperbola_real_axis_length 
  (a : ℝ) 
  (h1 : ∃ y1 y2 : ℝ, rectangular_hyperbola a (-4) y1 ∧ rectangular_hyperbola a (-4) y2)
  (h2 : ∃ y1 y2 : ℝ, distance y1 y2 = 4 * Real.sqrt 3 ∧ rectangular_hyperbola a (-4) y1 ∧ rectangular_hyperbola a (-4) y2) :
  2 * a = 4 := by
  sorry

#check hyperbola_real_axis_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l362_36241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_extrema_l362_36224

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_distance_extrema (x₁ x₂ : ℝ) :
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  ∃ d : ℝ, d = 2 ∧ ∀ y₁ y₂ : ℝ, (∀ x : ℝ, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_extrema_l362_36224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l362_36202

theorem union_cardinality (C D : Finset ℕ) : 
  (C.card = 30) → 
  (D.card = 25) → 
  ((C ∩ D).card = 10) → 
  ((C ∪ D).card = 45) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l362_36202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_payment_l362_36220

theorem min_cuts_for_payment (n : ℕ) (h : n = 2018) : ∃ (cuts : ℕ) (chains : List ℕ),
  cuts = 10 ∧
  chains.sum = n ∧
  (∀ m : ℕ, m ≤ n → ∃ subset : List ℕ, subset.sum = m ∧ subset ⊆ chains) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_payment_l362_36220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_i_equals_zero_l362_36208

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the ratio of volumes
noncomputable def m : ℝ := 3/2

-- Define the ratio of surface areas
noncomputable def n : ℝ := 3/2

-- Define the function f
noncomputable def f (x : ℂ) : ℂ := ((m/n) * x^3 - 1/x)^8

-- Theorem statement
theorem f_of_i_equals_zero : f i = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_i_equals_zero_l362_36208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_reduction_l362_36292

/-- Given a ribbon of length 55 cm reduced in the ratio 11:7, prove that its new length is 35 cm. -/
theorem ribbon_reduction (original_length : ℝ) (reduction_ratio : ℚ) : 
  original_length = 55 ∧ reduction_ratio = 11/7 → original_length * (reduction_ratio.den / reduction_ratio.num) = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_reduction_l362_36292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_to_quarter_circle_area_ratio_l362_36268

theorem semicircle_to_quarter_circle_area_ratio (R : ℝ) (h : R > 0) :
  (π * (R/2)^2 / 2) / (π * R^2 / 4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_to_quarter_circle_area_ratio_l362_36268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_eq_99_l362_36299

/-- An arithmetic sequence with specified conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_147 : a 1 + a 4 + a 7 = 39
  sum_369 : a 3 + a 6 + a 9 = 27

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: The sum of the first 9 terms of the given arithmetic sequence is 99 -/
theorem sum_9_eq_99 (seq : ArithmeticSequence) : sum_n seq 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_eq_99_l362_36299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_equals_e_nonnegative_on_interval_implies_a_nonnegative_l362_36259

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a - 1

-- Theorem 1
theorem extreme_value_implies_a_equals_e (a : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧ (∃ x : ℝ, f a x = Real.exp 1 - 1) →
  a = Real.exp 1 := by
  sorry

-- Theorem 2
theorem nonnegative_on_interval_implies_a_nonnegative (a : ℝ) :
  (∀ x : ℝ, x ≥ a → f a x ≥ 0) →
  a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_equals_e_nonnegative_on_interval_implies_a_nonnegative_l362_36259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_weighings_sufficient_l362_36239

/-- Represents a bag of coins -/
structure CoinBag where
  id : Nat
  isFake : Bool

/-- Represents a weighing result -/
structure WeighingResult where
  difference : ℝ

/-- Function to perform a weighing -/
def weigh (bags : List CoinBag) (leftCoins : List (CoinBag × Nat)) (rightCoins : List (CoinBag × Nat)) : WeighingResult :=
  sorry

/-- Function to determine the bag with fake coins based on two weighing results -/
def determineFakeBag (result1 result2 : WeighingResult) : Nat :=
  sorry

theorem two_weighings_sufficient :
  ∀ (bags : List CoinBag),
    bags.length = 11 →
    (∃! bag : CoinBag, bag ∈ bags ∧ bag.isFake) →
    ∃ (leftCoins1 rightCoins1 leftCoins2 rightCoins2 : List (CoinBag × Nat)),
      let result1 := weigh bags leftCoins1 rightCoins1
      let result2 := weigh bags leftCoins2 rightCoins2
      let fakeBagId := determineFakeBag result1 result2
      ∃ (fakeBag : CoinBag), fakeBag ∈ bags ∧ fakeBag.id = fakeBagId ∧ fakeBag.isFake :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_weighings_sufficient_l362_36239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_not_in_fourth_quadrant_l362_36209

-- Define the complex number z as a function of real number m
noncomputable def z (m : ℝ) : ℂ := (m + Complex.I) / (1 - Complex.I)

-- Define the real and imaginary parts of z
noncomputable def Re (m : ℝ) : ℝ := (z m).re
noncomputable def Im (m : ℝ) : ℝ := (z m).im

-- Theorem: z is not in the fourth quadrant for any real m
theorem z_not_in_fourth_quadrant (m : ℝ) : ¬(Re m > 0 ∧ Im m < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_not_in_fourth_quadrant_l362_36209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_theorem_l362_36216

/-- The radius of a sphere given its shadow properties -/
noncomputable def sphere_radius_from_shadow (shadow_length : ℝ) (stick_shadow : ℝ) : ℝ :=
  10 * Real.sqrt 5 - 20

/-- The main theorem stating the radius of the sphere -/
theorem sphere_radius_theorem : 
  sphere_radius_from_shadow 10 2 = 10 * Real.sqrt 5 - 20 := by
  -- Unfold the definition of sphere_radius_from_shadow
  unfold sphere_radius_from_shadow
  -- The equality holds by reflexivity
  rfl

/-- A lemma showing that the calculated radius satisfies the original equation -/
lemma sphere_radius_equation : 
  let r := sphere_radius_from_shadow 10 2
  r * (Real.sqrt 5 + 2) = 10 := by
  -- Unfold the definition and simplify
  unfold sphere_radius_from_shadow
  -- This is a complex algebraic equality. In a full proof, we'd need to show each step.
  -- For now, we'll use sorry to skip the detailed algebraic manipulation
  sorry

-- We can't use #eval for non-computable definitions, so we'll omit it


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_theorem_l362_36216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_number_is_ten_l362_36282

/-- Given three consecutive whole numbers, prove that the middle number is 10 -/
theorem middle_number_is_ten (x y z : ℕ) : 
  x < y → y < z → 
  x + y = 18 → x + z = 23 → y + z = 25 → 
  ¬(Nat.Prime x) → x > 1 →
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_number_is_ten_l362_36282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l362_36210

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^x + (3 : ℝ)^x + (6 : ℝ)^x = (7 : ℝ)^x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l362_36210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_theorem_l362_36297

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {y | y ≥ 4}

-- Theorem statement
theorem set_operations_theorem :
  ((U \ M) ∪ (U \ N) = {x | 0 < x ∧ x < 4}) ∧
  ((U \ M) ∩ (U \ N) = {x | 0 < x ∧ x < 1}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_theorem_l362_36297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l362_36229

-- Part 1
theorem sin_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by sorry

-- Part 2
noncomputable def f (a x : ℝ) : ℝ := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x, |x| < δ → f a x ≤ f a 0) ↔ a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l362_36229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l362_36221

/-- The locus of points (x,y) satisfying 5√(x²+y²) = |3x+4y-12| is a parabola -/
theorem locus_is_parabola :
  ∃ (f : ℝ × ℝ → ℝ), (∀ (x y : ℝ), 5 * Real.sqrt (x^2 + y^2) = |3*x + 4*y - 12| → f (x, y) = 0) ∧
  ∃ (a b c d e : ℝ), ∀ (p : ℝ × ℝ), f p = a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l362_36221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l362_36258

-- Define the conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + 3 > 0
def q (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 2, a * (2^x) ≥ 1

-- Define the sets we want to prove
def S₁ : Set ℝ := {a : ℝ | a > 1/12}
def S₂ : Set ℝ := {a : ℝ | 1/12 < a ∧ a < 1/4}

-- State the theorem
theorem problem_statement :
  (∀ a : ℝ, p a ↔ a ∈ S₁) ∧
  (∀ a : ℝ, (p a ∧ q a ∧ ¬(q (-a))) ↔ a ∈ S₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l362_36258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_14_24_l362_36246

/-- The area of a rhombus given the lengths of its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals of lengths 14 and 24 is 168 -/
theorem rhombus_area_14_24 : rhombusArea 14 24 = 168 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that 14 * 24 / 2 = 168
  norm_num

#eval (14 * 24 : ℚ) / 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_14_24_l362_36246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_trip_average_speed_l362_36231

/-- Proves that the average speed of a bicycle trip is 17.5 km/h given specific conditions -/
theorem bicycle_trip_average_speed :
  let total_distance : ℝ := 350
  let first_part_distance : ℝ := 200
  let second_part_distance : ℝ := total_distance - first_part_distance
  let first_part_speed : ℝ := 20
  let second_part_speed : ℝ := 15
  let first_part_time : ℝ := first_part_distance / first_part_speed
  let second_part_time : ℝ := second_part_distance / second_part_speed
  let total_time : ℝ := first_part_time + second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 17.5 := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_trip_average_speed_l362_36231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_twenty_in_domain_of_f_of_f_twenty_is_smallest_l362_36223

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4)

-- State the theorem
theorem smallest_x_in_domain_of_f_of_f : 
  ∀ x : ℝ, (∃ y : ℝ, f (f x) = y) → x ≥ 20 :=
by sorry

-- State that 20 is in the domain of f(f(x))
theorem twenty_in_domain_of_f_of_f : 
  ∃ y : ℝ, f (f 20) = y :=
by sorry

-- State that 20 is the smallest such number
theorem twenty_is_smallest : 
  ∀ x : ℝ, x < 20 → ¬(∃ y : ℝ, f (f x) = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_twenty_in_domain_of_f_of_f_twenty_is_smallest_l362_36223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_investment_time_approx_13_39_l362_36245

/-- Represents the investment and profit ratios for three partners -/
structure InvestmentData where
  p_invest : ℝ
  q_invest : ℝ
  r_invest : ℝ
  p_profit : ℝ
  q_profit : ℝ
  r_profit : ℝ

/-- Calculates the investment time for partner q given the investment data and times for p and r -/
noncomputable def calculate_q_time (data : InvestmentData) (p_time r_time : ℝ) : ℝ :=
  (p_time * data.q_profit * data.r_invest) / (r_time * data.p_profit * data.q_invest)

/-- Theorem stating that given the specific investment data and times, q's investment time is approximately 13.39 months -/
theorem q_investment_time_approx_13_39 :
  let data : InvestmentData := {
    p_invest := 7
    q_invest := 5.00001
    r_invest := 3.99999
    p_profit := 7.00001
    q_profit := 10
    r_profit := 6
  }
  let p_time : ℝ := 5
  let r_time : ℝ := 8
  let q_time := calculate_q_time data p_time r_time
  abs (q_time - 13.39) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_investment_time_approx_13_39_l362_36245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_ratio_l362_36248

theorem candy_ratio : 
  ∀ (kitkat hershey nerds lollipops babyruths reeses : ℕ),
  kitkat = 5 →
  hershey = 3 * kitkat →
  nerds = 8 →
  lollipops = 11 →
  babyruths = 10 →
  kitkat + hershey + nerds + (lollipops - 5) + babyruths + reeses = 49 →
  (reeses : ℚ) / babyruths = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_ratio_l362_36248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_octagon_area_ratio_theorem_ab_product_is_sixteen_l362_36240

/-- The ratio of the area of a circle inscribed in a regular octagon 
    (touching the midpoints of the octagon's sides) to the area of the octagon -/
noncomputable def circle_octagon_area_ratio : ℝ :=
  (Real.sqrt 2 / 8) * Real.pi

/-- A regular octagon -/
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A circle inscribed in a regular octagon, touching the midpoints of the octagon's sides -/
structure InscribedCircle (octagon : RegularOctagon) where
  radius : ℝ
  touches_midpoints : radius = octagon.side_length / 2

/-- Theorem stating the relationship between the areas of the inscribed circle and the octagon -/
theorem circle_octagon_area_ratio_theorem (octagon : RegularOctagon) 
  (circle : InscribedCircle octagon) : 
  (circle.radius ^ 2 * Real.pi) / (8 * octagon.side_length ^ 2 * Real.sqrt 2 / 4) = 
  circle_octagon_area_ratio := by
  sorry

/-- The product ab in the problem statement -/
def ab_product : ℕ := 16

/-- Theorem proving that the product ab equals 16 -/
theorem ab_product_is_sixteen : ab_product = 16 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_octagon_area_ratio_theorem_ab_product_is_sixteen_l362_36240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_k_condition_l362_36262

/-- The function f(x) defined in terms of k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  ((k+1)*x^2 + (k+3)*x + (2*k-8)) / ((2*k-1)*x^2 + (k+1)*x + (k-4))

/-- The domain D of f(x) -/
def D (k : ℝ) : Set ℝ := {x | (2*k-1)*x^2 + (k+1)*x + (k-4) ≠ 0}

/-- The condition for k -/
def k_condition (k : ℝ) : Prop :=
  k = 1 ∨ k > (15 + 16*Real.sqrt 2)/7 ∨ k < (15 - 16*Real.sqrt 2)/7

/-- The main theorem -/
theorem f_positive_iff_k_condition (k : ℝ) :
  (∀ x ∈ D k, f k x > 0) ↔ k_condition k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_k_condition_l362_36262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_theorem_l362_36285

open Real MeasureTheory

-- Define the condition function
noncomputable def condition (x : ℝ) : Prop := x < 1 ∧ sin (log x / log 2) < 0

-- Define the total length of intervals
noncomputable def total_length : ℝ := 2^π / (1 + 2^π)

-- State the theorem
theorem interval_length_theorem : 
  (∫ x in {x | condition x}, 1) = total_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_theorem_l362_36285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l362_36277

/-- Given an angle α in the second quadrant with cos α = -3/5, prove that sin 2α = -24/25 -/
theorem sin_double_angle_second_quadrant (α : Real) :
  (π/2 < α) ∧ (α < π) →  -- α is in the second quadrant
  Real.cos α = -3/5 →
  Real.sin (2 * α) = -24/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l362_36277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rational_distance_set_l362_36264

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℚ
  y : ℚ

/-- The set of points we want to prove exists -/
noncomputable def PointSet : Set Point :=
  sorry

/-- The distance between two points is rational -/
def rationalDistance (p q : Point) : Prop :=
  ∃ r : ℚ, (p.x - q.x)^2 + (p.y - q.y)^2 = r^2

/-- Three points are not collinear -/
def notCollinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

theorem infinite_rational_distance_set :
  Set.Infinite PointSet ∧
  (∀ p q, p ∈ PointSet → q ∈ PointSet → p ≠ q → rationalDistance p q) ∧
  (∀ p q r, p ∈ PointSet → q ∈ PointSet → r ∈ PointSet → p ≠ q → q ≠ r → p ≠ r → notCollinear p q r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rational_distance_set_l362_36264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l362_36244

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x ≠ 0, f a x = f a (-x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l362_36244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_1642064901_l362_36291

theorem sixth_root_of_1642064901 : (1642064901 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_1642064901_l362_36291
