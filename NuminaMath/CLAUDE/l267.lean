import Mathlib

namespace NUMINAMATH_CALUDE_probability_point_between_C_and_D_l267_26727

/-- Given points A, B, C, and D on a line segment AB where AB = 4AD and AB = 3BC,
    the probability that a randomly selected point on AB is between C and D is 5/12. -/
theorem probability_point_between_C_and_D 
  (A B C D : ℝ) 
  (h_order : A ≤ C ∧ C ≤ D ∧ D ≤ B) 
  (h_AB_4AD : B - A = 4 * (D - A))
  (h_AB_3BC : B - A = 3 * (B - C)) : 
  (D - C) / (B - A) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_between_C_and_D_l267_26727


namespace NUMINAMATH_CALUDE_sequence_bound_l267_26719

theorem sequence_bound (x : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → |x i - x j| ≥ 1 / (i + j))
  (h2 : ∀ (i : ℕ), 0 ≤ x i ∧ x i ≤ c) : 
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l267_26719


namespace NUMINAMATH_CALUDE_exists_polygon_with_equal_area_division_l267_26798

/-- A polygon in the plane --/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_closed : ∀ (p : ℝ × ℝ), p ∈ vertices → ∃ (q : ℝ × ℝ), q ∈ vertices ∧ q ≠ p

/-- A point is on the boundary of a polygon --/
def OnBoundary (p : ℝ × ℝ) (poly : Polygon) : Prop :=
  p ∈ poly.vertices

/-- A line divides a polygon into two parts --/
def DividesPolygon (l : Set (ℝ × ℝ)) (poly : Polygon) : Prop :=
  ∃ (A B : Set (ℝ × ℝ)), A ∪ B = poly.vertices ∧ A ∩ B ⊆ l

/-- The area of a set of points in the plane --/
def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- A line divides a polygon into two equal parts --/
def DividesEquallyByArea (l : Set (ℝ × ℝ)) (poly : Polygon) : Prop :=
  ∃ (A B : Set (ℝ × ℝ)), 
    DividesPolygon l poly ∧
    Area A = Area B

/-- Main theorem: There exists a polygon and a point on its boundary such that 
    any line passing through this point divides the area of the polygon into two equal parts --/
theorem exists_polygon_with_equal_area_division :
  ∃ (poly : Polygon) (p : ℝ × ℝ), 
    OnBoundary p poly ∧
    ∀ (l : Set (ℝ × ℝ)), p ∈ l → DividesEquallyByArea l poly := by
  sorry

end NUMINAMATH_CALUDE_exists_polygon_with_equal_area_division_l267_26798


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l267_26758

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 2 < 0 ↔ 1/3 < x ∧ x < 1/2) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l267_26758


namespace NUMINAMATH_CALUDE_range_of_g_l267_26735

theorem range_of_g (x : ℝ) : 
  -(1/4) ≤ Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6 ∧ 
  Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6 ≤ 3/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_g_l267_26735


namespace NUMINAMATH_CALUDE_complex_modulus_l267_26795

theorem complex_modulus (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l267_26795


namespace NUMINAMATH_CALUDE_polynomial_simplification_l267_26710

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x^6 + x^5 + 3*x^4 + 2*x^2 + 15
def q (x : ℝ) : ℝ := x^6 + x^5 + 4*x^4 - x^3 + x^2 + 18
def r (x : ℝ) : ℝ := x^6 - x^4 + x^3 + x^2 - 3

-- State the theorem
theorem polynomial_simplification (x : ℝ) : p x - q x = r x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l267_26710


namespace NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l267_26767

/-- Proves that a man's rowing speed in still water is 25 km/hr given the conditions of downstream speed and time. -/
theorem mans_rowing_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 3) -- Current speed in km/hr
  (h2 : distance = 90) -- Distance in meters
  (h3 : time = 17.998560115190784) -- Time in seconds
  : ∃ (still_water_speed : ℝ), still_water_speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l267_26767


namespace NUMINAMATH_CALUDE_initial_test_count_l267_26711

theorem initial_test_count (initial_avg : ℝ) (improved_avg : ℝ) (lowest_score : ℝ) :
  initial_avg = 35 →
  improved_avg = 40 →
  lowest_score = 20 →
  ∃ n : ℕ,
    n > 1 ∧
    (n : ℝ) * initial_avg = ((n : ℝ) - 1) * improved_avg + lowest_score ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_initial_test_count_l267_26711


namespace NUMINAMATH_CALUDE_triangle_4_4_7_l267_26783

/-- A triangle can be formed from three line segments if the sum of any two sides
    is greater than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 4, 4, and 7 can form a triangle. -/
theorem triangle_4_4_7 :
  can_form_triangle 4 4 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_4_4_7_l267_26783


namespace NUMINAMATH_CALUDE_first_digit_853_base8_l267_26749

/-- The first digit of the base 8 representation of a natural number -/
def firstDigitBase8 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 8).succ
    (n / 8^(k-1)) % 8

theorem first_digit_853_base8 :
  firstDigitBase8 853 = 1 := by
sorry

end NUMINAMATH_CALUDE_first_digit_853_base8_l267_26749


namespace NUMINAMATH_CALUDE_sector_area_of_ring_l267_26784

/-- The area of a 60° sector of the ring between two concentric circles with radii 12 and 8 -/
theorem sector_area_of_ring (π : ℝ) : 
  let r₁ : ℝ := 12  -- radius of larger circle
  let r₂ : ℝ := 8   -- radius of smaller circle
  let ring_area : ℝ := π * (r₁^2 - r₂^2)
  let sector_angle : ℝ := 60
  let full_angle : ℝ := 360
  let sector_area : ℝ := (sector_angle / full_angle) * ring_area
  sector_area = (40 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_of_ring_l267_26784


namespace NUMINAMATH_CALUDE_ticket_cost_difference_l267_26734

theorem ticket_cost_difference : 
  let num_adults : ℕ := 9
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  (num_adults * adult_ticket_price) - (num_children * child_ticket_price) = 50 := by
sorry

end NUMINAMATH_CALUDE_ticket_cost_difference_l267_26734


namespace NUMINAMATH_CALUDE_total_pupils_l267_26752

theorem total_pupils (pizza : ℕ) (burgers : ℕ) (both : ℕ) 
  (h1 : pizza = 125) 
  (h2 : burgers = 115) 
  (h3 : both = 40) : 
  pizza + burgers - both = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_l267_26752


namespace NUMINAMATH_CALUDE_probability_adjacent_points_hexagon_l267_26754

/-- The number of points on the regular hexagon -/
def num_points : ℕ := 6

/-- The number of adjacent pairs on the regular hexagon -/
def num_adjacent_pairs : ℕ := 6

/-- The probability of selecting two adjacent points on a regular hexagon -/
theorem probability_adjacent_points_hexagon : 
  (num_adjacent_pairs : ℚ) / (num_points.choose 2) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_points_hexagon_l267_26754


namespace NUMINAMATH_CALUDE_min_period_tan_2x_l267_26733

/-- The minimum positive period of the function y = tan 2x is π/2 -/
theorem min_period_tan_2x : 
  let f : ℝ → ℝ := λ x => Real.tan (2 * x)
  ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧ 
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
    p = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_period_tan_2x_l267_26733


namespace NUMINAMATH_CALUDE_min_dot_product_l267_26768

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2

-- Define the points P and Q
def P (x : ℝ) : ℝ × ℝ := (2 - x, 0)
def Q (x : ℝ) : ℝ × ℝ := (2, 1 + x)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem min_dot_product :
  ∀ (A B C D : ℝ × ℝ) (x : ℝ),
    Rectangle A B C D →
    A = (0, 1) →
    B = (2, 1) →
    C = (2, 0) →
    D = (0, 0) →
    0 ≤ x →
    x ≤ 2 →
    (∀ y : ℝ, 0 ≤ y → y ≤ 2 →
      dot_product ((-2 + y, 1)) (y, 1 + y) ≥ 3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_l267_26768


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l267_26794

theorem min_value_quadratic_function :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  ∃ m : ℝ, m = -3 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 1 → f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l267_26794


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l267_26750

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → zeros i ≠ zeros j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, zeros i = b + d * i.val
  is_zero : ∀ x, x^4 + j*x^2 + k*x + 256 = (x - zeros 0) * (x - zeros 1) * (x - zeros 2) * (x - zeros 3)

/-- The value of j in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_value (p : ArithmeticProgressionPolynomial) : p.j = -40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l267_26750


namespace NUMINAMATH_CALUDE_percentage_reduction_optimal_price_increase_l267_26746

-- Define the original price
def original_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the initial profit per kilogram
def initial_profit : ℝ := 10

-- Define the initial daily sales
def initial_sales : ℝ := 500

-- Define the sales decrease per yuan of price increase
def sales_decrease_rate : ℝ := 20

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Theorem for the percentage reduction
theorem percentage_reduction :
  ∃ (r : ℝ), r > 0 ∧ r < 1 ∧ original_price * (1 - r)^2 = final_price ∧ r = 0.2 := by sorry

-- Theorem for the optimal price increase
theorem optimal_price_increase :
  ∃ (x : ℝ), x > 0 ∧
    (initial_profit + x) * (initial_sales - sales_decrease_rate * x) = target_profit ∧
    x = 5 := by sorry

end NUMINAMATH_CALUDE_percentage_reduction_optimal_price_increase_l267_26746


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l267_26716

theorem modulo_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [ZMOD 15] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l267_26716


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l267_26745

/-- 
Given an arithmetic series of consecutive integers with first term (k^2 - k + 1),
prove that the sum of the first (k + 2) terms is equal to k^3 + (3k^2)/2 + k/2 + 2.
-/
theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℚ := k^2 - k + 1
  let n : ℕ := k + 2
  let S := (n : ℚ) / 2 * (a₁ + (a₁ + (n - 1)))
  S = k^3 + (3 * k^2) / 2 + k / 2 + 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_series_sum_l267_26745


namespace NUMINAMATH_CALUDE_greatest_number_in_set_l267_26730

/-- A set of consecutive multiples of 2 -/
def ConsecutiveMultiplesOf2 (n : ℕ) (start : ℕ) : Set ℕ :=
  {x : ℕ | ∃ k : ℕ, k < n ∧ x = start + 2 * k}

theorem greatest_number_in_set (s : Set ℕ) :
  s = ConsecutiveMultiplesOf2 50 56 →
  ∃ m : ℕ, m ∈ s ∧ ∀ x ∈ s, x ≤ m ∧ m = 154 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_in_set_l267_26730


namespace NUMINAMATH_CALUDE_largest_number_with_properties_l267_26771

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that extracts a two-digit number from adjacent digits in a larger number -/
def twoDigitNumber (n : ℕ) (i : ℕ) : ℕ :=
  (n / 10^i % 100)

/-- A function that checks if all two-digit numbers formed by adjacent digits are prime -/
def allTwoDigitPrime (n : ℕ) : Prop :=
  ∀ i : ℕ, i < (Nat.digits 10 n).length - 1 → isPrime (twoDigitNumber n i)

/-- A function that checks if all two-digit prime numbers formed are distinct -/
def allTwoDigitPrimeDistinct (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j < (Nat.digits 10 n).length - 1 → 
    twoDigitNumber n i ≠ twoDigitNumber n j

/-- The main theorem stating that 617371311979 is the largest number satisfying the conditions -/
theorem largest_number_with_properties :
  (∀ m : ℕ, m > 617371311979 → 
    ¬(allTwoDigitPrime m ∧ allTwoDigitPrimeDistinct m)) ∧
  (allTwoDigitPrime 617371311979 ∧ allTwoDigitPrimeDistinct 617371311979) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_properties_l267_26771


namespace NUMINAMATH_CALUDE_bonus_calculation_l267_26760

/-- A quadratic function f(x) = kx^2 + b that satisfies certain conditions -/
def f (k b x : ℝ) : ℝ := k * x^2 + b

theorem bonus_calculation (k b : ℝ) (h1 : k > 0) 
  (h2 : f k b 10 = 0) (h3 : f k b 20 = 2) : f k b 200 = 266 := by
  sorry

end NUMINAMATH_CALUDE_bonus_calculation_l267_26760


namespace NUMINAMATH_CALUDE_exists_infinite_periodic_sequence_l267_26766

/-- A sequence of natural numbers -/
def InfiniteSequence := ℕ → ℕ

/-- Property: every natural number appears infinitely many times in the sequence -/
def AppearsInfinitelyOften (s : InfiniteSequence) : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, ∃ i ≥ k, s i = n

/-- Property: the sequence is periodic modulo m for every positive integer m -/
def PeriodicModulo (s : InfiniteSequence) : Prop :=
  ∀ m : ℕ+, ∃ p : ℕ+, ∀ i : ℕ, s (i + p) ≡ s i [MOD m]

/-- Theorem: There exists a sequence of natural numbers that appears infinitely often
    and is periodic modulo every positive integer -/
theorem exists_infinite_periodic_sequence :
  ∃ s : InfiniteSequence, AppearsInfinitelyOften s ∧ PeriodicModulo s := by
  sorry

end NUMINAMATH_CALUDE_exists_infinite_periodic_sequence_l267_26766


namespace NUMINAMATH_CALUDE_symmetric_points_nm_value_l267_26777

/-- Given two points P and Q symmetric with respect to the y-axis, prove that n^m = 1/2 -/
theorem symmetric_points_nm_value (m n : ℝ) : 
  (m - 1 = -2) → (4 = n + 2) → n^m = 1/2 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_nm_value_l267_26777


namespace NUMINAMATH_CALUDE_solve_for_a_l267_26741

theorem solve_for_a (y : ℝ) (h1 : y > 0) (h2 : (a * y) / 20 + (3 * y) / 10 = 0.7 * y) : a = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l267_26741


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l267_26737

theorem quadratic_equation_solution (x₁ : ℚ) (h₁ : x₁ = 3/4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) : 
  ∃ x₂ : ℚ, x₂ = -31/6 ∧ 72 * x₂^2 + 39 * x₂ - 18 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l267_26737


namespace NUMINAMATH_CALUDE_frustum_volume_l267_26707

/-- The volume of a frustum formed by cutting a square pyramid parallel to its base -/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) :
  base_edge = 16 →
  altitude = 10 →
  small_base_edge = 8 →
  small_altitude = 5 →
  let original_volume := (1 / 3) * base_edge^2 * altitude
  let small_volume := (1 / 3) * small_base_edge^2 * small_altitude
  original_volume - small_volume = 2240 / 3 :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l267_26707


namespace NUMINAMATH_CALUDE_sugar_profit_problem_l267_26713

theorem sugar_profit_problem (total_sugar : ℝ) (sugar_at_18_percent : ℝ) 
  (overall_profit_percent : ℝ) (profit_18_percent : ℝ) :
  total_sugar = 1000 →
  sugar_at_18_percent = 600 →
  overall_profit_percent = 14 →
  profit_18_percent = 18 →
  ∃ (remaining_profit_percent : ℝ),
    remaining_profit_percent = 8 ∧
    (sugar_at_18_percent * (1 + profit_18_percent / 100) + 
     (total_sugar - sugar_at_18_percent) * (1 + remaining_profit_percent / 100)) / total_sugar
    = 1 + overall_profit_percent / 100 :=
by sorry

end NUMINAMATH_CALUDE_sugar_profit_problem_l267_26713


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l267_26744

-- Define the types of sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Define a situation
structure Situation where
  description : String
  populationSize : Nat
  sampleSize : Nat

-- Define a function to determine the appropriate sampling method
def appropriateSamplingMethod (s : Situation) : SamplingMethod :=
  sorry

-- Define the three situations
def situation1 : Situation :=
  { description := "Selecting 2 students from each class"
  , populationSize := 0  -- We don't know the exact population size
  , sampleSize := 2 }

def situation2 : Situation :=
  { description := "Selecting 12 students from a class with different score ranges"
  , populationSize := 62  -- 10 + 40 + 12
  , sampleSize := 12 }

def situation3 : Situation :=
  { description := "Arranging tracks for 6 students in a 400m final"
  , populationSize := 6
  , sampleSize := 6 }

-- Theorem stating the correct sampling methods for each situation
theorem correct_sampling_methods :
  (appropriateSamplingMethod situation1 = SamplingMethod.Systematic) ∧
  (appropriateSamplingMethod situation2 = SamplingMethod.Stratified) ∧
  (appropriateSamplingMethod situation3 = SamplingMethod.SimpleRandom) :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l267_26744


namespace NUMINAMATH_CALUDE_tetrahedron_volume_not_unique_l267_26742

/-- Represents a tetrahedron with face areas and circumradius -/
structure Tetrahedron where
  face_areas : Fin 4 → ℝ
  circumradius : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of a tetrahedron is not uniquely determined by its face areas and circumradius -/
theorem tetrahedron_volume_not_unique : ∃ (t1 t2 : Tetrahedron), 
  (∀ i : Fin 4, t1.face_areas i = t2.face_areas i) ∧ 
  t1.circumradius = t2.circumradius ∧ 
  volume t1 ≠ volume t2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_not_unique_l267_26742


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l267_26702

theorem stratified_sampling_male_count :
  ∀ (total_employees : ℕ) 
    (female_employees : ℕ) 
    (sample_size : ℕ),
  total_employees = 120 →
  female_employees = 72 →
  sample_size = 15 →
  (total_employees - female_employees) * sample_size / total_employees = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l267_26702


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l267_26791

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  ∃ (min : ℝ), min = 80 ∧ 
  ∀ (x : ℝ), x = (p*t)^2 + (q*u)^2 + (r*v)^2 + (s*w)^2 → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l267_26791


namespace NUMINAMATH_CALUDE_population_growth_l267_26797

/-- The initial population of the town -/
def initial_population : ℝ := 1000

/-- The growth rate in the first year -/
def first_year_growth : ℝ := 0.10

/-- The growth rate in the second year -/
def second_year_growth : ℝ := 0.20

/-- The final population after two years -/
def final_population : ℝ := 1320

/-- Theorem stating the relationship between initial and final population -/
theorem population_growth :
  initial_population * (1 + first_year_growth) * (1 + second_year_growth) = final_population := by
  sorry

#check population_growth

end NUMINAMATH_CALUDE_population_growth_l267_26797


namespace NUMINAMATH_CALUDE_smallest_representable_66_88_l267_26799

/-- Represents a base-b digit --/
def IsDigitBase (d : ℕ) (b : ℕ) : Prop := d < b

/-- Converts a two-digit number in base b to base 10 --/
def BaseToDecimal (d₁ d₂ : ℕ) (b : ℕ) : ℕ := d₁ * b + d₂

/-- States that a number n can be represented as CC₆ and DD₈ --/
def RepresentableAs66And88 (n : ℕ) : Prop :=
  ∃ (c d : ℕ), IsDigitBase c 6 ∧ IsDigitBase d 8 ∧
    n = BaseToDecimal c c 6 ∧ n = BaseToDecimal d d 8

theorem smallest_representable_66_88 :
  (∀ m, RepresentableAs66And88 m → m ≥ 63) ∧ RepresentableAs66And88 63 := by sorry

end NUMINAMATH_CALUDE_smallest_representable_66_88_l267_26799


namespace NUMINAMATH_CALUDE_quintons_fruit_trees_l267_26769

/-- Represents the width of an apple tree in feet -/
def apple_width : ℕ := 10

/-- Represents the space needed between apple trees in feet -/
def apple_space : ℕ := 12

/-- Represents the width of a peach tree in feet -/
def peach_width : ℕ := 12

/-- Represents the space needed between peach trees in feet -/
def peach_space : ℕ := 15

/-- Represents the total space available for all trees in feet -/
def total_space : ℕ := 71

/-- Calculates the total number of fruit trees Quinton can plant -/
def total_fruit_trees : ℕ :=
  let apple_trees := 2
  let apple_total_space := apple_trees * apple_width + (apple_trees - 1) * apple_space
  let peach_space_left := total_space - apple_total_space
  let peach_trees := 1 + (peach_space_left - peach_width) / (peach_width + peach_space)
  apple_trees + peach_trees

theorem quintons_fruit_trees :
  total_fruit_trees = 4 := by
  sorry

end NUMINAMATH_CALUDE_quintons_fruit_trees_l267_26769


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l267_26715

theorem arithmetic_sequence_sum (a₁ a₄ a₁₀ : ℚ) (n : ℕ) : 
  a₁ = -3 → a₄ = 4 → a₁₀ = 40 → n = 10 → 
  (n : ℚ) / 2 * (a₁ + a₁₀) = 285 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l267_26715


namespace NUMINAMATH_CALUDE_art_class_earnings_l267_26706

def art_class_problem (price_per_class : ℚ) (saturday_attendance : ℕ) : ℚ :=
  let sunday_attendance := saturday_attendance / 2
  let total_attendance := saturday_attendance + sunday_attendance
  price_per_class * total_attendance

theorem art_class_earnings :
  art_class_problem 10 20 = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_art_class_earnings_l267_26706


namespace NUMINAMATH_CALUDE_three_liters_to_pints_l267_26792

-- Define the conversion rate from liters to pints
def liters_to_pints (liters : ℝ) : ℝ := 2.16 * liters

-- Theorem statement
theorem three_liters_to_pints : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, 
  |x - 3| < δ → |liters_to_pints x - 6.5| < ε :=
sorry

end NUMINAMATH_CALUDE_three_liters_to_pints_l267_26792


namespace NUMINAMATH_CALUDE_actual_weight_of_three_bags_l267_26700

/-- The actual weight of three bags of food given their labeled weight and deviations -/
theorem actual_weight_of_three_bags 
  (labeled_weight : ℕ) 
  (num_bags : ℕ) 
  (deviation1 deviation2 deviation3 : ℤ) : 
  labeled_weight = 200 → 
  num_bags = 3 → 
  deviation1 = 10 → 
  deviation2 = -16 → 
  deviation3 = -11 → 
  (labeled_weight * num_bags : ℤ) + deviation1 + deviation2 + deviation3 = 583 := by
  sorry

end NUMINAMATH_CALUDE_actual_weight_of_three_bags_l267_26700


namespace NUMINAMATH_CALUDE_three_planes_intersection_count_l267_26743

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- Represents the intersection of two planes -/
inductive PlanesIntersection
  | Line
  | Empty

/-- Represents the number of intersection lines between three planes -/
inductive IntersectionCount
  | One
  | Three

/-- Function to determine how two planes intersect -/
def planesIntersect (p1 p2 : Plane3D) : PlanesIntersection :=
  sorry

/-- 
Given three planes in 3D space that intersect each other pairwise,
prove that the number of their intersection lines is either 1 or 3.
-/
theorem three_planes_intersection_count 
  (p1 p2 p3 : Plane3D)
  (h12 : planesIntersect p1 p2 = PlanesIntersection.Line)
  (h23 : planesIntersect p2 p3 = PlanesIntersection.Line)
  (h31 : planesIntersect p3 p1 = PlanesIntersection.Line) :
  ∃ (count : IntersectionCount), 
    (count = IntersectionCount.One ∨ count = IntersectionCount.Three) :=
by
  sorry

end NUMINAMATH_CALUDE_three_planes_intersection_count_l267_26743


namespace NUMINAMATH_CALUDE_princes_wish_fulfilled_l267_26712

/-- Represents a knight at the round table -/
structure Knight where
  city : Nat
  hasGoldGoblet : Bool

/-- Represents the state of the round table -/
def RoundTable := List Knight

/-- Checks if two knights from the same city have gold goblets -/
def sameCity_haveGold (table : RoundTable) : Bool := sorry

/-- Rotates the goblets one position to the right -/
def rotateGoblets (table : RoundTable) : RoundTable := sorry

theorem princes_wish_fulfilled 
  (initial_table : RoundTable)
  (h1 : initial_table.length = 13)
  (h2 : ∃ k : Nat, 1 < k ∧ k < 13 ∧ (initial_table.filter Knight.hasGoldGoblet).length = k)
  (h3 : ∃ k : Nat, 1 < k ∧ k < 13 ∧ (initial_table.map Knight.city).toFinset.card = k) :
  ∃ n : Nat, sameCity_haveGold (n.iterate rotateGoblets initial_table) := by
  sorry

end NUMINAMATH_CALUDE_princes_wish_fulfilled_l267_26712


namespace NUMINAMATH_CALUDE_regular_polygon_60_properties_l267_26762

/-- A regular polygon with 60 sides -/
structure RegularPolygon60 where
  -- No additional fields needed as the number of sides is fixed

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The measure of an exterior angle in a regular polygon -/
def exterior_angle (n : ℕ) : ℚ := 360 / n

theorem regular_polygon_60_properties (p : RegularPolygon60) :
  (num_diagonals 60 = 1710) ∧ (exterior_angle 60 = 6) := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_60_properties_l267_26762


namespace NUMINAMATH_CALUDE_T_equals_eleven_l267_26731

/-- Given a natural number S, we define F as the sum of powers of 2 from 0 to S -/
def F (S : ℕ) : ℝ := (2^(S+1) - 1)

/-- T is defined as the square root of the ratio of logarithms -/
noncomputable def T (S : ℕ) : ℝ := Real.sqrt (Real.log (1 + F S) / Real.log 2)

/-- The theorem states that for S = 120, T equals 11 -/
theorem T_equals_eleven : T 120 = 11 := by sorry

end NUMINAMATH_CALUDE_T_equals_eleven_l267_26731


namespace NUMINAMATH_CALUDE_melted_to_spending_value_ratio_l267_26780

-- Define the weight of a quarter in ounces
def quarter_weight : ℚ := 1/5

-- Define the value of melted gold per ounce in dollars
def melted_gold_value_per_ounce : ℚ := 100

-- Define the spending value of a quarter in dollars
def quarter_spending_value : ℚ := 1/4

-- Theorem statement
theorem melted_to_spending_value_ratio : 
  (melted_gold_value_per_ounce / quarter_weight) / (1 / quarter_spending_value) = 80 := by
  sorry

end NUMINAMATH_CALUDE_melted_to_spending_value_ratio_l267_26780


namespace NUMINAMATH_CALUDE_fraction_equality_l267_26726

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l267_26726


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l267_26773

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) :
  -1 < a - b ∧ a - b < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l267_26773


namespace NUMINAMATH_CALUDE_larger_integer_is_fifteen_l267_26747

theorem larger_integer_is_fifteen (a b : ℤ) : 
  (a : ℚ) / b = 1 / 3 → 
  (a + 10 : ℚ) / b = 1 → 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_is_fifteen_l267_26747


namespace NUMINAMATH_CALUDE_isosceles_if_root_one_equilateral_roots_l267_26709

/-- Triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.b) * x^2 - 2 * t.c * x + (t.a - t.b)

theorem isosceles_if_root_one (t : Triangle) :
  quadratic t 1 = 0 → t.a = t.c :=
sorry

theorem equilateral_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (quadratic t 0 = 0 ∧ quadratic t 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_if_root_one_equilateral_roots_l267_26709


namespace NUMINAMATH_CALUDE_gcd_g_x_l267_26739

def g (x : ℤ) : ℤ := (3*x+8)*(5*x+1)*(11*x+6)*(2*x+3)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 12096 * k) : 
  Int.gcd (g x) x = 144 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l267_26739


namespace NUMINAMATH_CALUDE_petya_cannot_win_l267_26788

/-- Represents a chess tournament --/
structure ChessTournament where
  players : ℕ
  games_per_player : ℕ
  total_games : ℕ
  last_place_max_points : ℕ

/-- Creates a chess tournament with the given number of players --/
def create_tournament (n : ℕ) : ChessTournament :=
  { players := n
  , games_per_player := n - 1
  , total_games := n * (n - 1) / 2
  , last_place_max_points := (n * (n - 1) / 2) / n }

/-- Theorem: Petya cannot become the winner after disqualification --/
theorem petya_cannot_win (t : ChessTournament) 
  (h1 : t.players = 10) 
  (h2 : t = create_tournament 10) 
  (h3 : t.last_place_max_points ≤ 4) :
  ∃ (remaining_players : ℕ) (remaining_games : ℕ),
    remaining_players = t.players - 1 ∧
    remaining_games = remaining_players * (remaining_players - 1) / 2 ∧
    remaining_games / remaining_players ≥ t.last_place_max_points :=
by sorry

end NUMINAMATH_CALUDE_petya_cannot_win_l267_26788


namespace NUMINAMATH_CALUDE_max_m_value_l267_26748

theorem max_m_value (A B C D : ℝ × ℝ) (m : ℝ) : 
  A = (1, 0) →
  B = (0, 1) →
  C = (a, b) →
  D = (c, d) →
  (∀ a b c d : ℝ, 
    (c - a)^2 + (d - b)^2 ≥ 
    (m - 2) * (a * c + b * d) + 
    m * (a * 0 + b * 1) * (c * 1 + d * 0)) →
  ∃ m_max : ℝ, m_max = Real.sqrt 5 - 1 ∧ 
    (∀ m' : ℝ, (∀ a b c d : ℝ, 
      (c - a)^2 + (d - b)^2 ≥ 
      (m' - 2) * (a * c + b * d) + 
      m' * (a * 0 + b * 1) * (c * 1 + d * 0)) → 
    m' ≤ m_max) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l267_26748


namespace NUMINAMATH_CALUDE_base_2_representation_315_l267_26774

/-- Given a natural number n, returns the number of zeros in its binary representation -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- Given a natural number n, returns the number of ones in its binary representation -/
def count_ones (n : ℕ) : ℕ := sorry

theorem base_2_representation_315 : 
  let x := count_zeros 315
  let y := count_ones 315
  y - x = 5 := by sorry

end NUMINAMATH_CALUDE_base_2_representation_315_l267_26774


namespace NUMINAMATH_CALUDE_money_distribution_l267_26714

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem statement -/
theorem money_distribution (s : Share) : 
  s.b = 0.65 * s.a → 
  s.c = 0.4 * s.a → 
  s.c = 56 → 
  s.a + s.b + s.c = 287 := by
  sorry


end NUMINAMATH_CALUDE_money_distribution_l267_26714


namespace NUMINAMATH_CALUDE_average_weight_of_all_boys_l267_26776

/-- Given two groups of boys with known average weights, 
    calculate the average weight of all boys. -/
theorem average_weight_of_all_boys 
  (group1_count : ℕ) 
  (group1_avg_weight : ℝ) 
  (group2_count : ℕ) 
  (group2_avg_weight : ℝ) 
  (h1 : group1_count = 24)
  (h2 : group1_avg_weight = 50.25)
  (h3 : group2_count = 8)
  (h4 : group2_avg_weight = 45.15) :
  (group1_count * group1_avg_weight + group2_count * group2_avg_weight) / 
  (group1_count + group2_count) = 48.975 := by
sorry

#eval (24 * 50.25 + 8 * 45.15) / (24 + 8)

end NUMINAMATH_CALUDE_average_weight_of_all_boys_l267_26776


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l267_26765

/-- Given the total number of doughnuts made, the number of boxes sold, and the number of doughnuts
given away, prove that the number of doughnuts in each box is equal to
(total doughnuts made - doughnuts given away) divided by the number of boxes sold. -/
theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (h1 : total_doughnuts ≥ doughnuts_given_away)
  (h2 : boxes_sold > 0)
  (h3 : total_doughnuts - doughnuts_given_away = boxes_sold * (total_doughnuts - doughnuts_given_away) / boxes_sold) :
  (total_doughnuts - doughnuts_given_away) / boxes_sold =
  (total_doughnuts - doughnuts_given_away) / boxes_sold :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l267_26765


namespace NUMINAMATH_CALUDE_bee_legs_count_l267_26724

theorem bee_legs_count (legs_per_bee : ℕ) (num_bees : ℕ) (h : legs_per_bee = 6) :
  legs_per_bee * num_bees = 48 ↔ num_bees = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_legs_count_l267_26724


namespace NUMINAMATH_CALUDE_parabola_x_axis_intersections_l267_26772

/-- The number of intersection points between y = 3x^2 + 2x + 1 and the x-axis is 0 -/
theorem parabola_x_axis_intersections :
  let f (x : ℝ) := 3 * x^2 + 2 * x + 1
  (∃ x : ℝ, f x = 0) = False :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_axis_intersections_l267_26772


namespace NUMINAMATH_CALUDE_two_polygons_exist_l267_26721

/-- Represents a polygon with a given number of sides. -/
structure Polygon where
  sides : ℕ

/-- Calculates the sum of interior angles of a polygon. -/
def sumInteriorAngles (p : Polygon) : ℕ :=
  (p.sides - 2) * 180

/-- Calculates the number of diagonals in a polygon. -/
def numDiagonals (p : Polygon) : ℕ :=
  p.sides * (p.sides - 3) / 2

/-- Theorem stating the existence of two polygons satisfying the given conditions. -/
theorem two_polygons_exist : ∃ (p1 p2 : Polygon),
  (sumInteriorAngles p1 + sumInteriorAngles p2 = 1260) ∧
  (numDiagonals p1 + numDiagonals p2 = 14) ∧
  ((p1.sides = 6 ∧ p2.sides = 5) ∨ (p1.sides = 5 ∧ p2.sides = 6)) := by
  sorry

end NUMINAMATH_CALUDE_two_polygons_exist_l267_26721


namespace NUMINAMATH_CALUDE_probability_theorem_l267_26786

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the number of each roll type prepared -/
def rolls_per_type : ℕ := 3

/-- Represents the total number of rolls -/
def total_rolls : ℕ := num_roll_types * rolls_per_type

/-- Represents the number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_roll_types

/-- Calculates the probability of each guest receiving one roll of each type -/
def probability_one_of_each : ℚ :=
  (rolls_per_type ^ num_roll_types * (rolls_per_type - 1) ^ num_roll_types * (rolls_per_type - 2) ^ num_roll_types) /
  (Nat.choose total_rolls rolls_per_guest * Nat.choose (total_rolls - rolls_per_guest) rolls_per_guest * Nat.choose (total_rolls - 2*rolls_per_guest) rolls_per_guest)

theorem probability_theorem :
  probability_one_of_each = 12 / 321 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l267_26786


namespace NUMINAMATH_CALUDE_ellipse_max_min_sum_absolute_values_l267_26717

theorem ellipse_max_min_sum_absolute_values :
  ∀ x y : ℝ, x^2/4 + y^2/9 = 1 →
  (∃ a b : ℝ, a^2/4 + b^2/9 = 1 ∧ |a| + |b| = 3) ∧
  (∃ c d : ℝ, c^2/4 + d^2/9 = 1 ∧ |c| + |d| = 2) ∧
  (∀ z w : ℝ, z^2/4 + w^2/9 = 1 → |z| + |w| ≤ 3 ∧ |z| + |w| ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_min_sum_absolute_values_l267_26717


namespace NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l267_26785

theorem min_value_xy_plus_two_over_xy (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 1 → x * y + 2 / (x * y) ≤ z * w + 2 / (z * w)) ∧ 
  x * y + 2 / (x * y) = 33 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l267_26785


namespace NUMINAMATH_CALUDE_tangent_sum_given_ratio_l267_26705

theorem tangent_sum_given_ratio (α : Real) :
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/3 →
  Real.tan (α + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_given_ratio_l267_26705


namespace NUMINAMATH_CALUDE_solution_set_inequality_l267_26781

theorem solution_set_inequality (x : ℝ) : 
  (x - 1)^2 > 4 ↔ x < -1 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l267_26781


namespace NUMINAMATH_CALUDE_cube_parabola_locus_l267_26761

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  origin : Point3D
  sideLength : ℝ

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Calculate the distance from a point to a plane -/
def distanceToPlane (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry

/-- Check if a point is on a face of the cube -/
def isOnFace (p : Point3D) (cube : Cube) (face : Plane3D) : Prop :=
  sorry

/-- Define a parabola as a set of points -/
def isParabola (points : Set Point3D) : Prop :=
  sorry

theorem cube_parabola_locus (cube : Cube) (B : Point3D) (faceBCC₁B₁ planeCDD₁C₁ : Plane3D) :
  let locus := {M : Point3D | isOnFace M cube faceBCC₁B₁ ∧ 
                               distance M B = distanceToPlane M planeCDD₁C₁}
  isParabola locus := by
  sorry

end NUMINAMATH_CALUDE_cube_parabola_locus_l267_26761


namespace NUMINAMATH_CALUDE_value_of_c_l267_26796

theorem value_of_c (a b c : ℝ) : 
  8 = (4 / 100) * a →
  4 = (8 / 100) * b →
  c = b / a →
  c = 0.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l267_26796


namespace NUMINAMATH_CALUDE_cube_surface_area_l267_26704

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1000 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 600 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l267_26704


namespace NUMINAMATH_CALUDE_mike_fish_per_hour_l267_26703

/-- Represents the number of fish Mike can catch in one hour -/
def M : ℕ := sorry

/-- The number of fish Jim can catch in one hour -/
def jim_catch : ℕ := 2 * M

/-- The number of fish Bob can catch in one hour -/
def bob_catch : ℕ := 3 * M

/-- The total number of fish caught by all three in 40 minutes -/
def total_40min : ℕ := (2 * M / 3) + (4 * M / 3) + (2 * M)

/-- The number of fish Jim catches in the remaining 20 minutes -/
def jim_20min : ℕ := 2 * M / 3

/-- The total number of fish caught in one hour -/
def total_catch : ℕ := total_40min + jim_20min

theorem mike_fish_per_hour : 
  (total_catch = 140) → (M = 30) := by sorry

end NUMINAMATH_CALUDE_mike_fish_per_hour_l267_26703


namespace NUMINAMATH_CALUDE_prob_two_odd_chips_l267_26778

-- Define the set of numbers on the chips
def ChipNumbers : Set ℕ := {1, 2, 3, 4}

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop := n % 2 = 1

-- Define the probability of drawing an odd-numbered chip from one box
def probOddFromOneBox : ℚ := (2 : ℚ) / 4

-- Theorem statement
theorem prob_two_odd_chips :
  (probOddFromOneBox * probOddFromOneBox) = (1 : ℚ) / 4 :=
sorry

end NUMINAMATH_CALUDE_prob_two_odd_chips_l267_26778


namespace NUMINAMATH_CALUDE_kyles_speed_l267_26725

theorem kyles_speed (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (distance_difference : ℝ) :
  joseph_speed = 50 →
  joseph_time = 2.5 →
  kyle_time = 2 →
  distance_difference = 1 →
  joseph_speed * joseph_time = kyle_time * (joseph_speed * joseph_time - distance_difference) / kyle_time →
  (joseph_speed * joseph_time - distance_difference) / kyle_time = 62 :=
by
  sorry

#check kyles_speed

end NUMINAMATH_CALUDE_kyles_speed_l267_26725


namespace NUMINAMATH_CALUDE_stating_rabbit_distribution_count_l267_26756

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 2

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 4

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
  Calculates the number of ways to distribute rabbits to pet stores
  such that no store has both a parent and a child
--/
def distribute_rabbits : ℕ :=
  -- Definition of the function to calculate the number of ways
  -- This is left undefined as the actual implementation is not provided
  sorry

/-- 
  Theorem stating that the number of ways to distribute the rabbits
  is equal to 560
--/
theorem rabbit_distribution_count : distribute_rabbits = 560 := by
  sorry

end NUMINAMATH_CALUDE_stating_rabbit_distribution_count_l267_26756


namespace NUMINAMATH_CALUDE_exactly_one_B_divisible_by_7_l267_26782

def is_multiple_of_7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

def number_47B (B : ℕ) : ℕ :=
  400 + 70 + B

theorem exactly_one_B_divisible_by_7 :
  ∃! B : ℕ, B ≤ 9 ∧ is_multiple_of_7 (number_47B B) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_B_divisible_by_7_l267_26782


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l267_26757

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A nonagon (nine-sided polygon) has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l267_26757


namespace NUMINAMATH_CALUDE_librarian_took_books_l267_26740

theorem librarian_took_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : 
  total_books = 34 →
  books_per_shelf = 3 →
  shelves_needed = 9 →
  total_books - (books_per_shelf * shelves_needed) = 7 := by
sorry

end NUMINAMATH_CALUDE_librarian_took_books_l267_26740


namespace NUMINAMATH_CALUDE_books_added_by_marta_l267_26728

def initial_books : ℕ := 38
def final_books : ℕ := 48

theorem books_added_by_marta : 
  final_books - initial_books = 10 := by sorry

end NUMINAMATH_CALUDE_books_added_by_marta_l267_26728


namespace NUMINAMATH_CALUDE_set_intersection_problem_l267_26718

theorem set_intersection_problem (p q : ℝ) : 
  let M := {x : ℝ | x^2 - 5*x ≤ 0}
  let N := {x : ℝ | p < x ∧ x < 6}
  ({x : ℝ | 2 < x ∧ x ≤ q} = M ∩ N) → p + q = 7 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l267_26718


namespace NUMINAMATH_CALUDE_natural_number_representation_with_distinct_powers_l267_26732

theorem natural_number_representation_with_distinct_powers : ∃ (N : ℕ) 
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
  (∃ (x₁ x₂ : ℕ), a₁ = x₁^2 ∧ a₂ = x₂^2) ∧
  (∃ (y₁ y₂ : ℕ), b₁ = y₁^3 ∧ b₂ = y₂^3) ∧
  (∃ (z₁ z₂ : ℕ), c₁ = z₁^5 ∧ c₂ = z₂^5) ∧
  (∃ (w₁ w₂ : ℕ), d₁ = w₁^7 ∧ d₂ = w₂^7) ∧
  N = a₁ - a₂ ∧ N = b₁ - b₂ ∧ N = c₁ - c₂ ∧ N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by
  sorry


end NUMINAMATH_CALUDE_natural_number_representation_with_distinct_powers_l267_26732


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l267_26775

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  let d_on_ab := ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.D = k • t.A + (1 - k) • t.B
  let ad_bisects_bac := ∃ k : ℝ, k > 0 ∧ k • (t.C - t.A) = t.D - t.A
  let bd_length := dist t.B t.D = 36
  let bc_length := dist t.B t.C = 45
  let ac_length := dist t.A t.C = 40
  d_on_ab ∧ ad_bisects_bac ∧ bd_length ∧ bc_length ∧ ac_length

-- State the theorem
theorem angle_bisector_theorem (t : Triangle) :
  satisfies_conditions t → dist t.A t.D = 68 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l267_26775


namespace NUMINAMATH_CALUDE_prime_product_theorem_l267_26787

def largest_one_digit_prime : ℕ := 7
def second_largest_one_digit_prime : ℕ := 5
def second_largest_two_digit_prime : ℕ := 89

theorem prime_product_theorem :
  largest_one_digit_prime * second_largest_one_digit_prime * second_largest_two_digit_prime = 3115 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_theorem_l267_26787


namespace NUMINAMATH_CALUDE_power_sum_difference_l267_26764

theorem power_sum_difference : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l267_26764


namespace NUMINAMATH_CALUDE_quadratic_one_root_l267_26701

/-- A quadratic function with coefficients a = 1, b = 4, and c = n -/
def f (n : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + n

/-- The discriminant of the quadratic function f -/
def discriminant (n : ℝ) : ℝ := 4^2 - 4*1*n

theorem quadratic_one_root (n : ℝ) :
  (∃! x, f n x = 0) ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l267_26701


namespace NUMINAMATH_CALUDE_mia_excess_over_double_darwin_l267_26755

def darwin_money : ℕ := 45
def mia_money : ℕ := 110

theorem mia_excess_over_double_darwin : mia_money - 2 * darwin_money = 20 := by
  sorry

end NUMINAMATH_CALUDE_mia_excess_over_double_darwin_l267_26755


namespace NUMINAMATH_CALUDE_sin_inequality_equivalence_l267_26793

theorem sin_inequality_equivalence (a b : ℝ) :
  (∀ x : ℝ, Real.sin x + Real.sin a ≥ b * Real.cos x) ↔
  (∃ n : ℤ, a = (4 * n + 1) * Real.pi / 2) ∧ (b = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_equivalence_l267_26793


namespace NUMINAMATH_CALUDE_maple_taller_than_pine_l267_26736

-- Define the heights of the trees
def pine_height : ℚ := 15 + 1/4
def maple_height : ℚ := 20 + 2/3

-- Define the height difference
def height_difference : ℚ := maple_height - pine_height

-- Theorem to prove
theorem maple_taller_than_pine :
  height_difference = 5 + 5/12 := by sorry

end NUMINAMATH_CALUDE_maple_taller_than_pine_l267_26736


namespace NUMINAMATH_CALUDE_time_to_see_again_value_l267_26770

/-- The time (in seconds) before Jenny and Kenny can see each other again -/
def time_to_see_again (jenny_speed : ℝ) (kenny_speed : ℝ) (path_distance : ℝ) (building_diameter : ℝ) (initial_distance : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time before Jenny and Kenny can see each other again -/
theorem time_to_see_again_value :
  time_to_see_again 2 4 300 150 300 = 48 := by
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_value_l267_26770


namespace NUMINAMATH_CALUDE_expression_simplification_l267_26723

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.sqrt 2) 
  (hy : y = 2 * Real.sqrt 2) : 
  (4 * y^2 - x^2) / (x^2 + 2*x*y + y^2) / ((x - 2*y) / (2*x^2 + 2*x*y)) = -10 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l267_26723


namespace NUMINAMATH_CALUDE_brothers_additional_lambs_l267_26763

/-- The number of lambs Merry takes care of -/
def merrys_lambs : ℕ := 10

/-- The total number of lambs -/
def total_lambs : ℕ := 23

/-- The number of lambs Merry's brother takes care of -/
def brothers_lambs : ℕ := total_lambs - merrys_lambs

/-- The additional number of lambs Merry's brother takes care of compared to Merry -/
def additional_lambs : ℕ := brothers_lambs - merrys_lambs

theorem brothers_additional_lambs :
  additional_lambs = 3 ∧ brothers_lambs > merrys_lambs := by
  sorry

end NUMINAMATH_CALUDE_brothers_additional_lambs_l267_26763


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l267_26738

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l267_26738


namespace NUMINAMATH_CALUDE_min_dot_product_vectors_l267_26720

/-- The dot product of vectors (1, x) and (x, x+1) has a minimum value of -1 -/
theorem min_dot_product_vectors : 
  ∃ (min : ℝ), min = -1 ∧ 
  ∀ (x : ℝ), (1 * x + x * (x + 1)) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_vectors_l267_26720


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l267_26722

-- Define the propositions p and q
def p (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (t + 2) + y^2 / (t - 10) = 1

def q (t m : ℝ) : Prop := -m < t ∧ t < m + 1 ∧ m > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- Theorem statement
theorem hyperbola_m_range :
  (∀ t, sufficient_not_necessary (p t) (∃ m, q t m)) →
  ∀ m, (m > 0 ∧ m ≤ 2) ↔ (∃ t, q t m ∧ p t) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l267_26722


namespace NUMINAMATH_CALUDE_total_stickers_l267_26790

def initial_stickers : Float := 20.0
def bought_stickers : Float := 26.0
def birthday_stickers : Float := 20.0
def sister_gift : Float := 6.0
def mother_gift : Float := 58.0

theorem total_stickers : 
  initial_stickers + bought_stickers + birthday_stickers + sister_gift + mother_gift = 130.0 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_l267_26790


namespace NUMINAMATH_CALUDE_rectangle_area_change_l267_26708

/-- Given a rectangle with area 540 square centimeters, if its length is increased by 15% and
    its width is decreased by 15%, the new area will be 527.55 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h : l * w = 540) :
  (1.15 * l) * (0.85 * w) = 527.55 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l267_26708


namespace NUMINAMATH_CALUDE_inequality_proof_l267_26753

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l267_26753


namespace NUMINAMATH_CALUDE_impossible_score_53_l267_26789

/-- Represents the score of a quiz -/
structure QuizScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_questions : ℕ
  score : ℤ

/-- Calculates the score based on the number of correct, incorrect, and unanswered questions -/
def calculate_score (c i u : ℕ) : ℤ :=
  (4 : ℤ) * c - i

/-- Checks if a QuizScore is valid according to the quiz rules -/
def is_valid_score (qs : QuizScore) : Prop :=
  qs.correct + qs.incorrect + qs.unanswered = qs.total_questions ∧
  qs.score = calculate_score qs.correct qs.incorrect qs.unanswered

/-- Theorem: It's impossible to achieve a score of 53 in the given quiz -/
theorem impossible_score_53 :
  ¬ ∃ (qs : QuizScore), qs.total_questions = 15 ∧ is_valid_score qs ∧ qs.score = 53 :=
by sorry

end NUMINAMATH_CALUDE_impossible_score_53_l267_26789


namespace NUMINAMATH_CALUDE_prob_win_is_four_sevenths_l267_26779

/-- The probability of Lola losing a match -/
def prob_lose : ℚ := 3/7

/-- The theorem stating that the probability of Lola winning a match is 4/7 -/
theorem prob_win_is_four_sevenths :
  let prob_win := 1 - prob_lose
  prob_win = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_win_is_four_sevenths_l267_26779


namespace NUMINAMATH_CALUDE_celia_savings_l267_26751

def weekly_food_budget : ℕ := 100
def num_weeks : ℕ := 4
def monthly_rent : ℕ := 1500
def monthly_streaming : ℕ := 30
def monthly_cell_phone : ℕ := 50
def savings_rate : ℚ := 1 / 10

def total_spending : ℕ := weekly_food_budget * num_weeks + monthly_rent + monthly_streaming + monthly_cell_phone

def savings : ℚ := (total_spending : ℚ) * savings_rate

theorem celia_savings : savings = 198 := by sorry

end NUMINAMATH_CALUDE_celia_savings_l267_26751


namespace NUMINAMATH_CALUDE_rachel_painting_time_l267_26759

def minutes_per_day_first_6 : ℕ := 100
def days_first_period : ℕ := 6
def minutes_per_day_next_2 : ℕ := 120
def days_second_period : ℕ := 2
def target_average : ℕ := 110
def total_days : ℕ := 10

theorem rachel_painting_time :
  (minutes_per_day_first_6 * days_first_period +
   minutes_per_day_next_2 * days_second_period +
   (target_average * total_days - 
    (minutes_per_day_first_6 * days_first_period +
     minutes_per_day_next_2 * days_second_period))) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_rachel_painting_time_l267_26759


namespace NUMINAMATH_CALUDE_expression_evaluation_l267_26729

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 2
  (x + 2 * y)^2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l267_26729
