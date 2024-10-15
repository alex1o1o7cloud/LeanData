import Mathlib

namespace NUMINAMATH_CALUDE_lee_earnings_theorem_l151_15118

/-- Represents the lawn care services and their charges -/
structure LawnCareService where
  mowing : Nat
  trimming : Nat
  weedRemoval : Nat
  leafBlowing : Nat
  fertilizing : Nat

/-- Represents the number of services provided -/
structure ServicesProvided where
  mowing : Nat
  trimming : Nat
  weedRemoval : Nat
  leafBlowing : Nat
  fertilizing : Nat

/-- Represents the tips received for each service -/
structure TipsReceived where
  mowing : List Nat
  trimming : List Nat
  weedRemoval : List Nat
  leafBlowing : List Nat

/-- Calculates the total earnings from services and tips -/
def calculateTotalEarnings (charges : LawnCareService) (services : ServicesProvided) (tips : TipsReceived) : Nat :=
  let serviceEarnings := 
    charges.mowing * services.mowing +
    charges.trimming * services.trimming +
    charges.weedRemoval * services.weedRemoval +
    charges.leafBlowing * services.leafBlowing +
    charges.fertilizing * services.fertilizing
  let tipEarnings :=
    tips.mowing.sum + tips.trimming.sum + tips.weedRemoval.sum + tips.leafBlowing.sum
  serviceEarnings + tipEarnings

/-- Theorem stating that Lee's total earnings are $923 -/
theorem lee_earnings_theorem (charges : LawnCareService) (services : ServicesProvided) (tips : TipsReceived)
    (h1 : charges = { mowing := 33, trimming := 15, weedRemoval := 10, leafBlowing := 20, fertilizing := 25 })
    (h2 : services = { mowing := 16, trimming := 8, weedRemoval := 5, leafBlowing := 4, fertilizing := 3 })
    (h3 : tips = { mowing := [10, 10, 12, 15], trimming := [5, 7], weedRemoval := [5], leafBlowing := [6] }) :
    calculateTotalEarnings charges services tips = 923 := by
  sorry


end NUMINAMATH_CALUDE_lee_earnings_theorem_l151_15118


namespace NUMINAMATH_CALUDE_m_range_l151_15191

-- Define the function f on the interval [-2, 2]
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_domain : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ 0
axiom f_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = f x
axiom f_decreasing : ∀ a b, 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → a ≠ b → (f a - f b) / (a - b) < 0

-- Define the theorem
theorem m_range (m : ℝ) (h : f (1 - m) < f m) : -1 ≤ m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l151_15191


namespace NUMINAMATH_CALUDE_festival_attendance_l151_15111

/-- Proves the attendance on the second day of a three-day festival -/
theorem festival_attendance (total : ℕ) (day1 day2 day3 : ℕ) : 
  total = 2700 →
  day2 = day1 / 2 →
  day3 = 3 * day1 →
  total = day1 + day2 + day3 →
  day2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l151_15111


namespace NUMINAMATH_CALUDE_repeating_digits_divisible_by_11_l151_15163

/-- A function that generates a 9-digit number by repeating the first three digits three times -/
def repeatingDigits (a b c : ℕ) : ℕ :=
  100000000 * a + 10000000 * b + 1000000 * c +
  100000 * a + 10000 * b + 1000 * c +
  100 * a + 10 * b + c

/-- Theorem stating that any 9-digit number formed by repeating the first three digits three times is divisible by 11 -/
theorem repeating_digits_divisible_by_11 (a b c : ℕ) (h : 0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10) :
  11 ∣ repeatingDigits a b c := by
  sorry


end NUMINAMATH_CALUDE_repeating_digits_divisible_by_11_l151_15163


namespace NUMINAMATH_CALUDE_hyperbola_equation_l151_15188

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ k : ℝ, k * x + y = k * x + 2 → a = b) →
  (∃ c : ℝ, c^2 = 24 - 16 ∧ c^2 = a^2 + b^2) →
  a^2 = 4 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l151_15188


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l151_15164

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2210 → n + (n + 1) = -95 := by sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l151_15164


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l151_15162

theorem min_translation_for_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ φ : ℝ, φ > 0 ∧
    (∀ x, f (x - φ) = -f (-x + φ)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, f (x - ψ) = -f (-x + ψ)) → φ ≤ ψ) ∧
    φ = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l151_15162


namespace NUMINAMATH_CALUDE_flower_beds_count_l151_15185

/-- Calculates the total number of flower beds in a garden with three sections. -/
def totalFlowerBeds (seeds1 seeds2 seeds3 : ℕ) (seedsPerBed1 seedsPerBed2 seedsPerBed3 : ℕ) : ℕ :=
  (seeds1 / seedsPerBed1) + (seeds2 / seedsPerBed2) + (seeds3 / seedsPerBed3)

/-- Proves that the total number of flower beds is 105 given the specific conditions. -/
theorem flower_beds_count :
  totalFlowerBeds 470 320 210 10 10 8 = 105 := by
  sorry

#eval totalFlowerBeds 470 320 210 10 10 8

end NUMINAMATH_CALUDE_flower_beds_count_l151_15185


namespace NUMINAMATH_CALUDE_max_a_is_maximum_l151_15103

/-- The polynomial function f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The condition that |f(x)| ≤ 1 for all x in [0, 1] -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1

/-- The maximum value of a that satisfies the condition -/
def max_a : ℝ := 8

/-- Theorem stating that max_a is the maximum value satisfying the condition -/
theorem max_a_is_maximum :
  (condition max_a) ∧ (∀ a : ℝ, a > max_a → ¬(condition a)) :=
sorry

end NUMINAMATH_CALUDE_max_a_is_maximum_l151_15103


namespace NUMINAMATH_CALUDE_safe_mountain_climb_l151_15142

theorem safe_mountain_climb : ∃ t : ℕ,
  t ≥ 0 ∧
  t % 26 ≠ 0 ∧ t % 26 ≠ 1 ∧
  t % 14 ≠ 0 ∧ t % 14 ≠ 1 ∧
  (t + 6) % 26 ≠ 0 ∧ (t + 6) % 26 ≠ 1 ∧
  (t + 6) % 14 ≠ 0 ∧ (t + 6) % 14 ≠ 1 ∧
  t + 24 < 26 * 14 := by
  sorry

end NUMINAMATH_CALUDE_safe_mountain_climb_l151_15142


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l151_15134

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * cross_time_s

/-- Proof that a train's length is approximately 100.02 meters -/
theorem train_length_proof (speed_kmh : ℝ) (cross_time_s : ℝ) 
  (h1 : speed_kmh = 60) 
  (h2 : cross_time_s = 6) : 
  ∃ ε > 0, |train_length speed_kmh cross_time_s - 100.02| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l151_15134


namespace NUMINAMATH_CALUDE_books_given_difference_l151_15136

theorem books_given_difference (mike_books_tuesday : ℕ) (mike_gave : ℕ) (lily_total : ℕ)
  (h1 : mike_books_tuesday = 45)
  (h2 : mike_gave = 10)
  (h3 : lily_total = 35) :
  lily_total - mike_gave - (mike_gave) = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_given_difference_l151_15136


namespace NUMINAMATH_CALUDE_smallest_xy_value_smallest_xy_is_172_min_xy_value_l151_15161

theorem smallest_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) :
  ∀ (a b : ℕ+), 7 * a + 4 * b = 200 → x * y ≤ a * b :=
by sorry

theorem smallest_xy_is_172 :
  ∃ (x y : ℕ+), 7 * x + 4 * y = 200 ∧ x * y = 172 :=
by sorry

theorem min_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) :
  x * y ≥ 172 :=
by sorry

end NUMINAMATH_CALUDE_smallest_xy_value_smallest_xy_is_172_min_xy_value_l151_15161


namespace NUMINAMATH_CALUDE_round_trip_distance_l151_15179

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The distance to the star in light-years -/
def star_distance : ℝ := 25

/-- The duration of the round trip in years -/
def trip_duration : ℝ := 50

/-- The total distance traveled by light in a round trip to the star over the given duration -/
def total_distance : ℝ := 2 * star_distance * light_year_distance

theorem round_trip_distance : total_distance = 5.87e14 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_l151_15179


namespace NUMINAMATH_CALUDE_fathers_savings_l151_15102

theorem fathers_savings (total : ℝ) : 
  (total / 2 - (total / 2) * 0.6) = 2000 → total = 10000 := by
  sorry

end NUMINAMATH_CALUDE_fathers_savings_l151_15102


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l151_15195

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l151_15195


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l151_15189

theorem no_natural_square_diff_2014 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l151_15189


namespace NUMINAMATH_CALUDE_subset_of_sqrt_two_in_sqrt_three_set_l151_15160

theorem subset_of_sqrt_two_in_sqrt_three_set :
  {Real.sqrt 2} ⊆ {x : ℝ | x ≤ Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_subset_of_sqrt_two_in_sqrt_three_set_l151_15160


namespace NUMINAMATH_CALUDE_right_triangle_angles_l151_15105

/-- Represents a right triangle with external angles on the hypotenuse in the ratio 9:11 -/
structure RightTriangle where
  -- First acute angle in degrees
  α : ℝ
  -- Second acute angle in degrees
  β : ℝ
  -- The triangle is right-angled
  right_angle : α + β = 90
  -- The external angles on the hypotenuse are in the ratio 9:11
  external_angle_ratio : (180 - α) / (90 + α) = 9 / 11

/-- Theorem stating the acute angles of the specified right triangle -/
theorem right_triangle_angles (t : RightTriangle) : t.α = 58.5 ∧ t.β = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l151_15105


namespace NUMINAMATH_CALUDE_tv_price_proof_l151_15176

theorem tv_price_proof (X : ℝ) : 
  X * (1 + 0.4) * 0.8 - X = 270 → X = 2250 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_proof_l151_15176


namespace NUMINAMATH_CALUDE_binomial_variance_four_half_l151_15183

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ :=
  ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem binomial_variance_four_half :
  ∀ ξ : BinomialDistribution, ξ.n = 4 ∧ ξ.p = 1/2 → variance ξ = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_four_half_l151_15183


namespace NUMINAMATH_CALUDE_quadratic_sum_l151_15121

/-- A quadratic function f(x) = px^2 + qx + r with vertex (-3, 4) passing through (0, 1) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := fun x ↦ p * x^2 + q * x + r

/-- The vertex of the quadratic function -/
def vertex (p q r : ℝ) : ℝ × ℝ := (-3, 4)

/-- The function passes through the point (0, 1) -/
def passes_through_origin (p q r : ℝ) : Prop :=
  QuadraticFunction p q r 0 = 1

theorem quadratic_sum (p q r : ℝ) :
  vertex p q r = (-3, 4) →
  passes_through_origin p q r →
  p + q + r = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l151_15121


namespace NUMINAMATH_CALUDE_quadratic_sum_l151_15100

/-- Given a quadratic expression 4x^2 - 8x + 1, when expressed in the form a(x-h)^2 + k,
    the sum of a, h, and k equals 2. -/
theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) → a + h + k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l151_15100


namespace NUMINAMATH_CALUDE_village_population_l151_15137

theorem village_population (P : ℝ) : 0.85 * (0.95 * P) = 3294 → P = 4080 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l151_15137


namespace NUMINAMATH_CALUDE_total_borrowed_by_lunchtime_l151_15129

/-- Represents the number of books on a shelf at different times of the day -/
structure ShelfState where
  initial : ℕ
  added : ℕ
  borrowed_morning : ℕ
  borrowed_afternoon : ℕ
  remaining : ℕ

/-- Calculates the number of books borrowed by lunchtime for a given shelf -/
def borrowed_by_lunchtime (shelf : ShelfState) : ℕ :=
  shelf.initial + shelf.added - (shelf.remaining + shelf.borrowed_afternoon)

/-- The state of shelf A -/
def shelf_a : ShelfState := {
  initial := 100,
  added := 40,
  borrowed_morning := 0,  -- Unknown, to be calculated
  borrowed_afternoon := 30,
  remaining := 60
}

/-- The state of shelf B -/
def shelf_b : ShelfState := {
  initial := 150,
  added := 20,
  borrowed_morning := 50,
  borrowed_afternoon := 0,  -- Not needed for the calculation
  remaining := 80
}

/-- The state of shelf C -/
def shelf_c : ShelfState := {
  initial := 200,
  added := 10,
  borrowed_morning := 0,  -- Unknown, to be calculated
  borrowed_afternoon := 45,
  remaining := 200 + 10 - 130  -- 130 is total borrowed throughout the day
}

/-- Theorem stating that the total number of books borrowed by lunchtime across all shelves is 165 -/
theorem total_borrowed_by_lunchtime :
  borrowed_by_lunchtime shelf_a + borrowed_by_lunchtime shelf_b + borrowed_by_lunchtime shelf_c = 165 := by
  sorry

end NUMINAMATH_CALUDE_total_borrowed_by_lunchtime_l151_15129


namespace NUMINAMATH_CALUDE_fourth_degree_equation_roots_l151_15114

theorem fourth_degree_equation_roots :
  ∃ (r₁ r₂ r₃ r₄ : ℂ),
    (∀ x : ℂ, 3 * x^4 + 2 * x^3 - 7 * x^2 + 2 * x + 3 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) :=
by sorry

end NUMINAMATH_CALUDE_fourth_degree_equation_roots_l151_15114


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l151_15107

theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : son_age = 26 → age_difference = 28 → 
  ∃ y : ℕ, (son_age + y + age_difference) = 2 * (son_age + y) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l151_15107


namespace NUMINAMATH_CALUDE_spring_problem_l151_15159

/-- Represents the length of a spring as a function of mass -/
def spring_length (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem spring_problem (k : ℝ) :
  spring_length k 6 0 = 6 →
  spring_length k 6 4 = 7.2 →
  spring_length k 6 5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_spring_problem_l151_15159


namespace NUMINAMATH_CALUDE_simplified_expression_approximation_l151_15170

theorem simplified_expression_approximation :
  let expr := Real.sqrt 5 * 5^(1/3) + 18 / (2^2) * 3 - 8^(3/2)
  ∃ ε > 0, |expr + 1.8| < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_approximation_l151_15170


namespace NUMINAMATH_CALUDE_tub_ratio_is_one_third_l151_15140

/-- Represents the number of tubs in various categories -/
structure TubCounts where
  total : ℕ
  storage : ℕ
  usual_vendor : ℕ

/-- Calculates the ratio of tubs bought from new vendor to usual vendor -/
def tub_ratio (t : TubCounts) : Rat :=
  let new_vendor := t.total - t.storage - t.usual_vendor
  (new_vendor : Rat) / t.usual_vendor

/-- Theorem stating the ratio of tubs bought from new vendor to usual vendor -/
theorem tub_ratio_is_one_third (t : TubCounts) 
  (h_total : t.total = 100)
  (h_storage : t.storage = 20)
  (h_usual : t.usual_vendor = 60) :
  tub_ratio t = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tub_ratio_is_one_third_l151_15140


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l151_15112

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a * x^2 : ℝ) + (b * x : ℝ) + (c : ℝ)

theorem quadratic_function_k_value
  (a b c k : ℤ)
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : 60 < QuadraticFunction a b c 7 ∧ QuadraticFunction a b c 7 < 70)
  (h3 : 80 < QuadraticFunction a b c 8 ∧ QuadraticFunction a b c 8 < 90)
  (h4 : (2000 : ℝ) * (k : ℝ) < QuadraticFunction a b c 50 ∧
        QuadraticFunction a b c 50 < (2000 : ℝ) * ((k + 1) : ℝ)) :
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l151_15112


namespace NUMINAMATH_CALUDE_product_of_cosines_l151_15158

theorem product_of_cosines : 
  (1 + Real.cos (π/8)) * (1 + Real.cos (3*π/8)) * (1 + Real.cos (5*π/8)) * (1 + Real.cos (7*π/8)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l151_15158


namespace NUMINAMATH_CALUDE_largest_n_sin_cos_inequality_l151_15117

theorem largest_n_sin_cos_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 2/m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_sin_cos_inequality_l151_15117


namespace NUMINAMATH_CALUDE_find_a_l151_15141

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 4) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l151_15141


namespace NUMINAMATH_CALUDE_greatest_gcd_4Tn_n_minus_1_l151_15152

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- The statement to be proved -/
theorem greatest_gcd_4Tn_n_minus_1 :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (4 * T n) (n - 1) ≤ 4 ∧
  Nat.gcd (4 * T k) (k - 1) = 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_4Tn_n_minus_1_l151_15152


namespace NUMINAMATH_CALUDE_carton_height_calculation_l151_15148

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of items that can fit along one dimension -/
def maxItemsAlongDimension (containerSize itemSize : ℕ) : ℕ :=
  containerSize / itemSize

/-- Calculates the total number of items that can fit on the base of the container -/
def itemsOnBase (containerBase itemBase : Dimensions) : ℕ :=
  (maxItemsAlongDimension containerBase.length itemBase.length) *
  (maxItemsAlongDimension containerBase.width itemBase.width)

/-- Calculates the number of layers of items that can be stacked in the container -/
def numberOfLayers (maxItems itemsPerLayer : ℕ) : ℕ :=
  maxItems / itemsPerLayer

/-- Calculates the height of the container based on the number of layers and item height -/
def containerHeight (layers itemHeight : ℕ) : ℕ :=
  layers * itemHeight

theorem carton_height_calculation (cartonBase : Dimensions) (soapBox : Dimensions) (maxSoapBoxes : ℕ) :
  cartonBase.length = 25 →
  cartonBase.width = 42 →
  soapBox.length = 7 →
  soapBox.width = 12 →
  soapBox.height = 5 →
  maxSoapBoxes = 150 →
  containerHeight (numberOfLayers maxSoapBoxes (itemsOnBase cartonBase soapBox)) soapBox.height = 80 := by
  sorry

#check carton_height_calculation

end NUMINAMATH_CALUDE_carton_height_calculation_l151_15148


namespace NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l151_15181

theorem divisible_by_18_sqrt_between_30_and_30_5 : 
  ∀ n : ℕ, 
    n > 0 ∧ 
    n % 18 = 0 ∧ 
    30 < Real.sqrt n ∧ 
    Real.sqrt n < 30.5 → 
    n = 900 ∨ n = 918 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l151_15181


namespace NUMINAMATH_CALUDE_scaled_equation_l151_15110

theorem scaled_equation (h : 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_scaled_equation_l151_15110


namespace NUMINAMATH_CALUDE_bill_initial_money_l151_15116

theorem bill_initial_money (ann_initial bill_initial : ℕ) (transfer : ℕ) : 
  ann_initial = 777 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer →
  bill_initial = 1111 := by
sorry

end NUMINAMATH_CALUDE_bill_initial_money_l151_15116


namespace NUMINAMATH_CALUDE_condition_relationship_l151_15143

theorem condition_relationship (x : ℝ) : 
  (∀ x, x^2 - 2*x + 1 ≤ 0 → x > 0) ∧ 
  (∃ x, x > 0 ∧ x^2 - 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l151_15143


namespace NUMINAMATH_CALUDE_beetle_probability_theorem_l151_15186

/-- Represents the probability of a beetle touching a horizontal edge first -/
def beetle_horizontal_edge_probability (start_x start_y : ℕ) (grid_size : ℕ) : ℝ :=
  sorry

/-- The grid is 10x10 -/
def grid_size : ℕ := 10

/-- The beetle starts at (3, 4) -/
def start_x : ℕ := 3
def start_y : ℕ := 4

/-- Theorem stating the probability of the beetle touching a horizontal edge first -/
theorem beetle_probability_theorem :
  beetle_horizontal_edge_probability start_x start_y grid_size = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_beetle_probability_theorem_l151_15186


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l151_15198

theorem quadratic_equivalence :
  ∀ x : ℝ, (x^2 - 6*x + 4 = 0) ↔ ((x - 3)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l151_15198


namespace NUMINAMATH_CALUDE_log_equation_solution_l151_15156

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  (Real.log y^3 / Real.log 3) + (Real.log y / Real.log (1/3)) = 6 → y = 27 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l151_15156


namespace NUMINAMATH_CALUDE_area_of_right_trapezoid_l151_15175

/-- 
Given a horizontally placed right trapezoid whose oblique axonometric projection
is an isosceles trapezoid with a bottom angle of 45°, legs of length 1, and 
top base of length 1, the area of the original right trapezoid is 2 + √2.
-/
theorem area_of_right_trapezoid (h : ℝ) (w : ℝ) :
  h = 2 →
  w = 1 + Real.sqrt 2 →
  (1 / 2 : ℝ) * (w + 1) * h = 2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_area_of_right_trapezoid_l151_15175


namespace NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l151_15187

theorem cube_edge_length_from_circumscribed_sphere_volume :
  ∀ (edge_length : ℝ) (sphere_volume : ℝ),
    sphere_volume = 4 * Real.pi / 3 →
    (∃ (sphere_radius : ℝ),
      sphere_volume = 4 / 3 * Real.pi * sphere_radius ^ 3 ∧
      edge_length ^ 2 * 3 = (2 * sphere_radius) ^ 2) →
    edge_length = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l151_15187


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l151_15178

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 9th term of the geometric sequence with first term 4 and common ratio 1 is 4 -/
theorem ninth_term_of_sequence : geometric_sequence 4 1 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l151_15178


namespace NUMINAMATH_CALUDE_x_pos_sufficient_not_necessary_for_abs_x_pos_l151_15192

theorem x_pos_sufficient_not_necessary_for_abs_x_pos :
  (∃ (x : ℝ), |x| > 0 ∧ x ≤ 0) ∧
  (∀ (x : ℝ), x > 0 → |x| > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_pos_sufficient_not_necessary_for_abs_x_pos_l151_15192


namespace NUMINAMATH_CALUDE_negation_equivalence_l151_15169

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l151_15169


namespace NUMINAMATH_CALUDE_coefficient_m3n5_in_binomial_expansion_l151_15147

theorem coefficient_m3n5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (3 : ℕ)^(8 - k) * (5 : ℕ)^k) = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_m3n5_in_binomial_expansion_l151_15147


namespace NUMINAMATH_CALUDE_paint_cost_most_cost_effective_l151_15104

/-- Represents the payment options for the house painting job -/
inductive PaymentOption
  | WorkerDay
  | PaintCost
  | PaintedArea
  | HourlyRate

/-- Calculates the cost of a payment option given the job parameters -/
def calculate_cost (option : PaymentOption) (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (paint_cost : ℕ) (painted_area : ℕ) : ℕ :=
  match option with
  | PaymentOption.WorkerDay => workers * days * 30
  | PaymentOption.PaintCost => (paint_cost * 30) / 100
  | PaymentOption.PaintedArea => painted_area * 12
  | PaymentOption.HourlyRate => workers * hours_per_day * days * 4

/-- Theorem stating that the PaintCost option is the most cost-effective -/
theorem paint_cost_most_cost_effective (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (paint_cost : ℕ) (painted_area : ℕ) 
  (h1 : workers = 5)
  (h2 : hours_per_day = 8)
  (h3 : days = 10)
  (h4 : paint_cost = 4800)
  (h5 : painted_area = 150) :
  ∀ option, option ≠ PaymentOption.PaintCost → 
    calculate_cost PaymentOption.PaintCost workers hours_per_day days paint_cost painted_area ≤ 
    calculate_cost option workers hours_per_day days paint_cost painted_area :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_most_cost_effective_l151_15104


namespace NUMINAMATH_CALUDE_translated_point_coordinates_l151_15101

-- Define the points in the 2D plane
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 3)
def A' : ℝ × ℝ := (2, 1)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Define the translated point B'
def B' : ℝ × ℝ := (B.1 + translation_vector.1, B.2 + translation_vector.2)

-- Theorem statement
theorem translated_point_coordinates :
  B' = (4, 4) := by sorry

end NUMINAMATH_CALUDE_translated_point_coordinates_l151_15101


namespace NUMINAMATH_CALUDE_fraction_value_l151_15126

theorem fraction_value (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l151_15126


namespace NUMINAMATH_CALUDE_bisection_sqrt2_approximation_l151_15193

theorem bisection_sqrt2_approximation :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2| ≤ 0.1 := by
  sorry

end NUMINAMATH_CALUDE_bisection_sqrt2_approximation_l151_15193


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_value_l151_15113

theorem function_inequality_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 3*x + m ≥ 2*x^2 - 4*x) ↔ (-1 ≤ x ∧ x ≤ 2)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_value_l151_15113


namespace NUMINAMATH_CALUDE_wendy_running_distance_l151_15130

/-- The distance Wendy walked in miles -/
def distance_walked : ℝ := 9.17

/-- The additional distance Wendy ran compared to what she walked in miles -/
def additional_distance_ran : ℝ := 10.67

/-- The total distance Wendy ran in miles -/
def distance_ran : ℝ := distance_walked + additional_distance_ran

theorem wendy_running_distance : distance_ran = 19.84 := by
  sorry

end NUMINAMATH_CALUDE_wendy_running_distance_l151_15130


namespace NUMINAMATH_CALUDE_shaded_area_value_l151_15149

theorem shaded_area_value (d : ℝ) : 
  (3 * (2 - (1/2 * π * 1^2))) = 6 + d * π → d = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_value_l151_15149


namespace NUMINAMATH_CALUDE_cannot_reach_2000_l151_15177

theorem cannot_reach_2000 (a b : ℕ) : a * 12 + b * 17 ≠ 2000 := by
  sorry

end NUMINAMATH_CALUDE_cannot_reach_2000_l151_15177


namespace NUMINAMATH_CALUDE_second_grade_years_l151_15174

/-- Given information about Mrs. Randall's teaching career -/
def total_teaching_years : ℕ := 26
def third_grade_years : ℕ := 18

/-- Theorem stating the number of years Mrs. Randall taught second grade -/
theorem second_grade_years : total_teaching_years - third_grade_years = 8 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_years_l151_15174


namespace NUMINAMATH_CALUDE_pond_length_l151_15153

/-- Given a rectangular pond with width 10 meters, depth 8 meters, and volume 1600 cubic meters,
    prove that the length of the pond is 20 meters. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) (length : ℝ) :
  width = 10 →
  depth = 8 →
  volume = 1600 →
  volume = length * width * depth →
  length = 20 := by
sorry

end NUMINAMATH_CALUDE_pond_length_l151_15153


namespace NUMINAMATH_CALUDE_f_strictly_increasing_and_odd_l151_15150

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_strictly_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_and_odd_l151_15150


namespace NUMINAMATH_CALUDE_plaza_design_properties_l151_15144

/-- Represents the plaza design and cost structure -/
structure PlazaDesign where
  sideLength : ℝ
  lightTileCost : ℝ
  darkTileCost : ℝ
  borderWidth : ℝ

/-- Calculates the total cost of materials for the plaza design -/
def totalCost (design : PlazaDesign) : ℝ :=
  sorry

/-- Calculates the side length of the central light square -/
def centralSquareSideLength (design : PlazaDesign) : ℝ :=
  sorry

/-- Theorem stating the properties of the plaza design -/
theorem plaza_design_properties (design : PlazaDesign) 
  (h1 : design.sideLength = 20)
  (h2 : design.lightTileCost = 100000)
  (h3 : design.darkTileCost = 300000)
  (h4 : design.borderWidth = 2)
  (h5 : totalCost design = 2 * (design.darkTileCost / 4)) :
  totalCost design = 150000 ∧ centralSquareSideLength design = 10.5 :=
sorry

end NUMINAMATH_CALUDE_plaza_design_properties_l151_15144


namespace NUMINAMATH_CALUDE_circumcircle_radius_from_centroid_distance_l151_15173

/-- Given a triangle ABC with sides a, b, c, where c = AB, prove that if
    (b - c) / (a + c) = (c - a) / (b + c), then the radius R of the circumcircle
    satisfies R² = d² + c²/3, where d is the distance from the circumcircle
    center to the centroid of the triangle. -/
theorem circumcircle_radius_from_centroid_distance (a b c d : ℝ) :
  (b - c) / (a + c) = (c - a) / (b + c) →
  ∃ (R : ℝ), R > 0 ∧ R^2 = d^2 + c^2 / 3 :=
sorry

end NUMINAMATH_CALUDE_circumcircle_radius_from_centroid_distance_l151_15173


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l151_15180

theorem no_solution_to_equation :
  ¬∃ x : ℝ, (1 / (x + 8) + 1 / (x + 5) + 1 / (x + 1) = 1 / (x + 11) + 1 / (x + 2) + 1 / (x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l151_15180


namespace NUMINAMATH_CALUDE_floor_times_self_eq_90_l151_15154

theorem floor_times_self_eq_90 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 90 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_90_l151_15154


namespace NUMINAMATH_CALUDE_triangle_special_angle_l151_15109

/-- Given a triangle with side lengths a, b, and c satisfying the equation
    (c^2)/(a+b) + (a^2)/(b+c) = b, the angle opposite side b is 60°. -/
theorem triangle_special_angle (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
    (eq : c^2/(a+b) + a^2/(b+c) = b) : 
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))
  B = π/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l151_15109


namespace NUMINAMATH_CALUDE_george_hourly_rate_l151_15133

/-- Calculates the hourly rate given total income and hours worked -/
def hourly_rate (total_income : ℚ) (total_hours : ℚ) : ℚ :=
  total_income / total_hours

theorem george_hourly_rate :
  let monday_hours : ℚ := 7
  let tuesday_hours : ℚ := 2
  let total_hours : ℚ := monday_hours + tuesday_hours
  let total_income : ℚ := 45
  hourly_rate total_income total_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_george_hourly_rate_l151_15133


namespace NUMINAMATH_CALUDE_product_of_fractions_l151_15196

theorem product_of_fractions : 
  (1/2 : ℚ) * (9/1 : ℚ) * (1/8 : ℚ) * (64/1 : ℚ) * (1/128 : ℚ) * (729/1 : ℚ) * (1/2187 : ℚ) * (19683/1 : ℚ) = 59049/32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l151_15196


namespace NUMINAMATH_CALUDE_system_solution_l151_15145

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3*x + Real.sqrt (3*x - y) + y = 6
def equation2 (x y : ℝ) : Prop := 9*x^2 + 3*x - y - y^2 = 36

-- Define the solution set
def solutions : Set (ℝ × ℝ) := {(2, -3), (6, -18)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solution_l151_15145


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l151_15165

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius : ℝ := r / 2
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l151_15165


namespace NUMINAMATH_CALUDE_mission_duration_l151_15167

theorem mission_duration (planned_duration : ℝ) : 
  (1.6 * planned_duration + 3 = 11) → planned_duration = 5 := by
  sorry

end NUMINAMATH_CALUDE_mission_duration_l151_15167


namespace NUMINAMATH_CALUDE_total_spent_is_88_70_l151_15124

-- Define the constants
def pizza_price : ℝ := 10
def pizza_quantity : ℕ := 5
def pizza_discount_threshold : ℕ := 3
def pizza_discount_rate : ℝ := 0.15

def soft_drink_price : ℝ := 1.5
def soft_drink_quantity : ℕ := 10

def hamburger_price : ℝ := 3
def hamburger_quantity : ℕ := 6
def hamburger_discount_threshold : ℕ := 5
def hamburger_discount_rate : ℝ := 0.1

-- Define the function to calculate the total spent
def total_spent : ℝ :=
  let robert_pizza_cost := 
    if pizza_quantity > pizza_discount_threshold
    then pizza_price * pizza_quantity * (1 - pizza_discount_rate)
    else pizza_price * pizza_quantity
  let robert_drinks_cost := soft_drink_price * soft_drink_quantity
  let teddy_hamburger_cost := 
    if hamburger_quantity > hamburger_discount_threshold
    then hamburger_price * hamburger_quantity * (1 - hamburger_discount_rate)
    else hamburger_price * hamburger_quantity
  let teddy_drinks_cost := soft_drink_price * soft_drink_quantity
  robert_pizza_cost + robert_drinks_cost + teddy_hamburger_cost + teddy_drinks_cost

-- Theorem statement
theorem total_spent_is_88_70 : total_spent = 88.70 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_88_70_l151_15124


namespace NUMINAMATH_CALUDE_find_r_l151_15155

theorem find_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 3^r) 
  (h2 : 45 = k * 9^r) : 
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_find_r_l151_15155


namespace NUMINAMATH_CALUDE_negation_of_exists_is_forall_l151_15122

theorem negation_of_exists_is_forall :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_is_forall_l151_15122


namespace NUMINAMATH_CALUDE_log_inequality_l151_15172

theorem log_inequality : 
  let x := Real.log 2 / Real.log 5
  let y := Real.log 2
  let z := Real.sqrt 2
  x < y ∧ y < z := by sorry

end NUMINAMATH_CALUDE_log_inequality_l151_15172


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l151_15135

/-- Given a group of children with various emotional states and genders, 
    prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad 
  (total_children : ℕ) 
  (happy_children sad_children confused_children excited_children neither_happy_nor_sad : ℕ)
  (total_boys total_girls : ℕ)
  (happy_boys sad_girls confused_boys excited_girls : ℕ)
  (h1 : total_children = 80)
  (h2 : happy_children = 35)
  (h3 : sad_children = 15)
  (h4 : confused_children = 10)
  (h5 : excited_children = 5)
  (h6 : neither_happy_nor_sad = 15)
  (h7 : total_boys = 45)
  (h8 : total_girls = 35)
  (h9 : happy_boys = 8)
  (h10 : sad_girls = 7)
  (h11 : confused_boys = 4)
  (h12 : excited_girls = 3)
  (h13 : total_children = happy_children + sad_children + confused_children + excited_children + neither_happy_nor_sad)
  (h14 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls) + confused_boys + (excited_children - excited_girls)) = 23 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l151_15135


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l151_15125

def P : ℕ := 45 * 45 * 98 * 480

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry

def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_of_odd_divisors P) * 126 = sum_of_even_divisors P := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l151_15125


namespace NUMINAMATH_CALUDE_nine_times_s_on_half_l151_15123

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem nine_times_s_on_half : s (s (s (s (s (s (s (s (s (1/2)))))))))  = 13/15 := by
  sorry

end NUMINAMATH_CALUDE_nine_times_s_on_half_l151_15123


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l151_15139

/-- A line segment parameterized by t, connecting two points in 2D space. -/
structure LineSegment where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The point on the line segment at a given parameter t. -/
def LineSegment.point_at (l : LineSegment) (t : ℝ) : ℝ × ℝ :=
  (l.a * t + l.b, l.c * t + l.d)

theorem line_segment_parameter_sum_of_squares :
  ∀ l : LineSegment,
  (l.point_at 0 = (-3, 5)) →
  (l.point_at 0.5 = (0.5, 7.5)) →
  (l.point_at 1 = (4, 10)) →
  l.a^2 + l.b^2 + l.c^2 + l.d^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l151_15139


namespace NUMINAMATH_CALUDE_faster_train_speed_l151_15194

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  crossing_time = 10 →
  (2 * train_length) / crossing_time = 40 / 3 :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l151_15194


namespace NUMINAMATH_CALUDE_intercept_sum_l151_15132

/-- A line is described by the equation y + 3 = -3(x + 2) -/
def line_equation (x y : ℝ) : Prop := y + 3 = -3 * (x + 2)

/-- The x-intercept of the line -/
def x_intercept : ℝ := -3

/-- The y-intercept of the line -/
def y_intercept : ℝ := -9

theorem intercept_sum :
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept ∧ x_intercept + y_intercept = -12 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l151_15132


namespace NUMINAMATH_CALUDE_prove_a_value_l151_15151

theorem prove_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {0, 1} → 
  B = {-1, 0, a+3} → 
  A ⊆ B → 
  a = -2 := by sorry

end NUMINAMATH_CALUDE_prove_a_value_l151_15151


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_cricket_bat_profit_is_twenty_percent_l151_15127

/-- Calculates the profit percentage for seller A given the conditions of the cricket bat sale --/
theorem cricket_bat_profit_percentage 
  (cost_price_A : ℝ) 
  (profit_percentage_B : ℝ) 
  (selling_price_C : ℝ) : ℝ :=
  let selling_price_B := selling_price_C / (1 + profit_percentage_B)
  let profit_A := selling_price_B - cost_price_A
  let profit_percentage_A := (profit_A / cost_price_A) * 100
  
  profit_percentage_A

/-- The profit percentage for A when selling the cricket bat to B is 20% --/
theorem cricket_bat_profit_is_twenty_percent : 
  cricket_bat_profit_percentage 152 0.25 228 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_cricket_bat_profit_is_twenty_percent_l151_15127


namespace NUMINAMATH_CALUDE_set_c_forms_triangle_l151_15128

/-- Triangle inequality theorem for a set of three line segments -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The set of line segments (4, 5, 6) can form a triangle -/
theorem set_c_forms_triangle : satisfies_triangle_inequality 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_set_c_forms_triangle_l151_15128


namespace NUMINAMATH_CALUDE_largest_whole_number_inequality_l151_15131

theorem largest_whole_number_inequality (x : ℕ) : x ≤ 3 ↔ (1 / 4 : ℚ) + (x : ℚ) / 5 < 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_inequality_l151_15131


namespace NUMINAMATH_CALUDE_sum_in_range_l151_15199

theorem sum_in_range : 
  let sum := (17/4 : ℚ) + (11/4 : ℚ) + (57/8 : ℚ)
  14 < sum ∧ sum < 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l151_15199


namespace NUMINAMATH_CALUDE_power_seven_145_mod_12_l151_15171

theorem power_seven_145_mod_12 : 7^145 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_145_mod_12_l151_15171


namespace NUMINAMATH_CALUDE_probability_multiple_of_seven_l151_15184

/-- The probability of selecting a page number that is a multiple of 7 from a book with 500 pages -/
theorem probability_multiple_of_seven (total_pages : ℕ) (h : total_pages = 500) :
  (Finset.filter (fun n => n % 7 = 0) (Finset.range total_pages)).card / total_pages = 71 / 500 :=
by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_seven_l151_15184


namespace NUMINAMATH_CALUDE_builder_cost_l151_15115

/-- The cost of hiring builders to construct houses -/
theorem builder_cost (builders_per_floor : ℕ) (days_per_floor : ℕ) (daily_wage : ℕ)
  (num_builders : ℕ) (num_houses : ℕ) (floors_per_house : ℕ) :
  builders_per_floor = 3 →
  days_per_floor = 30 →
  daily_wage = 100 →
  num_builders = 6 →
  num_houses = 5 →
  floors_per_house = 6 →
  (num_houses * floors_per_house * days_per_floor * daily_wage * num_builders) / builders_per_floor = 270000 :=
by sorry

end NUMINAMATH_CALUDE_builder_cost_l151_15115


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l151_15168

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence with positive terms -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

theorem arithmetic_geometric_inequality (a b : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_geom : GeometricSequence b)
    (h_eq1 : a 1 = b 1) (h_eq2 : a 2 = b 2) (h_neq : a 1 ≠ a 2) :
    ∀ n : ℕ, n ≥ 3 → a n < b n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l151_15168


namespace NUMINAMATH_CALUDE_cylinder_min_surface_area_l151_15197

/-- For a right circular cylinder with fixed volume, the surface area is minimized when the diameter equals the height -/
theorem cylinder_min_surface_area (V : ℝ) (h V_pos : V > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  V = π * r^2 * h ∧
  (∀ (r' h' : ℝ), r' > 0 → h' > 0 → V = π * r'^2 * h' →
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') ∧
  h = 2 * r := by
  sorry

#check cylinder_min_surface_area

end NUMINAMATH_CALUDE_cylinder_min_surface_area_l151_15197


namespace NUMINAMATH_CALUDE_snowman_volume_snowman_volume_calculation_l151_15166

theorem snowman_volume (π : ℝ) : ℝ → ℝ → ℝ → ℝ :=
  fun r₁ r₂ r₃ =>
    let sphere_volume := fun r : ℝ => (4 / 3) * π * r^3
    sphere_volume r₁ + sphere_volume r₂ + sphere_volume r₃

theorem snowman_volume_calculation (π : ℝ) :
  snowman_volume π 4 5 6 = (1620 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_snowman_volume_snowman_volume_calculation_l151_15166


namespace NUMINAMATH_CALUDE_range_of_a_l151_15106

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
  (¬ ∃ x : ℝ, x^2 - x + a = 0) ∧
  ((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a > 1/4 ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l151_15106


namespace NUMINAMATH_CALUDE_extra_cat_food_l151_15190

theorem extra_cat_food (food_one_cat food_two_cats : ℝ)
  (h1 : food_one_cat = 0.5)
  (h2 : food_two_cats = 0.9) :
  food_two_cats - food_one_cat = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_extra_cat_food_l151_15190


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l151_15146

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 203000 = ScientificNotation.mk 2.03 5 sorry := by
  sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l151_15146


namespace NUMINAMATH_CALUDE_percentage_of_C_grades_l151_15119

/-- Represents a grade with its lower and upper bounds -/
structure Grade where
  letter : String
  lower : Nat
  upper : Nat

/-- Checks if a score falls within a grade range -/
def isInGradeRange (score : Nat) (grade : Grade) : Bool :=
  score >= grade.lower ∧ score <= grade.upper

/-- The grading scale -/
def gradingScale : List Grade := [
  ⟨"A", 95, 100⟩,
  ⟨"A-", 90, 94⟩,
  ⟨"B+", 85, 89⟩,
  ⟨"B", 80, 84⟩,
  ⟨"C+", 77, 79⟩,
  ⟨"C", 73, 76⟩,
  ⟨"D", 70, 72⟩,
  ⟨"F", 0, 69⟩
]

/-- The list of student scores -/
def scores : List Nat := [98, 75, 86, 77, 60, 94, 72, 79, 69, 82, 70, 93, 74, 87, 78, 84, 95, 73]

/-- Theorem stating that the percentage of students who received a grade of C is 16.67% -/
theorem percentage_of_C_grades (ε : Real) (h : ε > 0) : 
  ∃ (p : Real), abs (p - 16.67) < ε ∧ 
  p = (100 : Real) * (scores.filter (fun score => 
    ∃ (g : Grade), g ∈ gradingScale ∧ g.letter = "C" ∧ isInGradeRange score g
  )).length / scores.length :=
sorry

end NUMINAMATH_CALUDE_percentage_of_C_grades_l151_15119


namespace NUMINAMATH_CALUDE_friday_earnings_calculation_l151_15138

/-- Represents the earnings of Johannes' vegetable shop over three days -/
structure VegetableShopEarnings where
  wednesday : ℝ
  friday : ℝ
  today : ℝ

/-- Calculates the total earnings over three days -/
def total_earnings (e : VegetableShopEarnings) : ℝ :=
  e.wednesday + e.friday + e.today

theorem friday_earnings_calculation (e : VegetableShopEarnings) 
  (h1 : e.wednesday = 30)
  (h2 : e.today = 42)
  (h3 : total_earnings e = 48 * 2) : 
  e.friday = 24 := by
  sorry

end NUMINAMATH_CALUDE_friday_earnings_calculation_l151_15138


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l151_15108

/-- Given two vectors a and b in ℝ³, prove that k = -2 when (k * a + b) is perpendicular to a. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ × ℝ) (k : ℝ) : 
  a = (1, 1, 1) → 
  b = (1, 2, 3) → 
  (k * a.1 + b.1, k * a.2.1 + b.2.1, k * a.2.2 + b.2.2) • a = 0 → 
  k = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l151_15108


namespace NUMINAMATH_CALUDE_carolyns_silverware_knives_percentage_l151_15157

/-- The percentage of knives in Carolyn's silverware after a trade --/
theorem carolyns_silverware_knives_percentage 
  (initial_knives : ℕ) 
  (initial_forks : ℕ) 
  (initial_spoons_multiplier : ℕ) 
  (traded_knives : ℕ) 
  (traded_spoons : ℕ) 
  (h1 : initial_knives = 6)
  (h2 : initial_forks = 12)
  (h3 : initial_spoons_multiplier = 3)
  (h4 : traded_knives = 10)
  (h5 : traded_spoons = 6) :
  let initial_spoons := initial_knives * initial_spoons_multiplier
  let final_knives := initial_knives + traded_knives
  let final_spoons := initial_spoons - traded_spoons
  let total_silverware := final_knives + initial_forks + final_spoons
  (final_knives : ℚ) / (total_silverware : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_carolyns_silverware_knives_percentage_l151_15157


namespace NUMINAMATH_CALUDE_certain_number_equation_l151_15182

theorem certain_number_equation (x : ℝ) : 5100 - (x / 20.4) = 5095 ↔ x = 102 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l151_15182


namespace NUMINAMATH_CALUDE_ellipse_focal_distances_l151_15120

theorem ellipse_focal_distances (x y : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  x^2 / 25 + y^2 = 1 →  -- P is on the ellipse
  P = (x, y) →  -- P's coordinates
  (∃ d : ℝ, d = 2 ∧ (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = d ∨
                     Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = d)) →
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) +
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 10 →
  (∃ d : ℝ, d = 8 ∧ (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = d ∨
                     Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = d)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_distances_l151_15120
