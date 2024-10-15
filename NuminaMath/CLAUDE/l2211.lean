import Mathlib

namespace NUMINAMATH_CALUDE_binomial_20_4_l2211_221188

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l2211_221188


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2211_221143

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 8) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2211_221143


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2211_221137

theorem complex_equation_solution (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 56 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2211_221137


namespace NUMINAMATH_CALUDE_factorial_calculation_l2211_221152

theorem factorial_calculation : (4 * Nat.factorial 6 + 36 * Nat.factorial 5) / Nat.factorial 7 = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l2211_221152


namespace NUMINAMATH_CALUDE_number_above_265_l2211_221179

/-- Represents the pyramid-like array of numbers -/
def pyramid_array (n : ℕ) : List ℕ :=
  List.range (n * n + 1) -- This generates a list of numbers from 0 to n^2

/-- The number of elements in the nth row of the pyramid -/
def row_length (n : ℕ) : ℕ := 2 * n - 1

/-- The starting number of the nth row -/
def row_start (n : ℕ) : ℕ := (n - 1) ^ 2 + 1

/-- The position of a number in its row -/
def position_in_row (x : ℕ) : ℕ :=
  x - row_start (Nat.sqrt x) + 1

/-- The number directly above a given number in the pyramid -/
def number_above (x : ℕ) : ℕ :=
  row_start (Nat.sqrt x - 1) + position_in_row x - 1

theorem number_above_265 :
  number_above 265 = 234 := by sorry

end NUMINAMATH_CALUDE_number_above_265_l2211_221179


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l2211_221199

def is_single_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n a b : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    a ≠ b ∧
    is_prime a ∧
    is_prime b ∧
    is_prime (a + b) ∧
    n = a * b * (a + b) ∧
    (∀ (m : ℕ), 
      (∃ (x y : ℕ), 
        is_single_digit x ∧ 
        is_single_digit y ∧ 
        x ≠ y ∧ 
        is_prime x ∧ 
        is_prime y ∧ 
        is_prime (x + y) ∧ 
        m = x * y * (x + y)) → m ≤ n) ∧
    sum_of_digits n = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l2211_221199


namespace NUMINAMATH_CALUDE_power_sum_integer_l2211_221138

theorem power_sum_integer (x : ℝ) (h : ∃ (a : ℤ), x + 1/x = a) :
  ∀ (n : ℕ), ∃ (b : ℤ), x^n + 1/(x^n) = b :=
by sorry

end NUMINAMATH_CALUDE_power_sum_integer_l2211_221138


namespace NUMINAMATH_CALUDE_shorter_pipe_length_l2211_221117

/-- Given a pipe of 177 inches cut into two pieces, where one piece is twice the length of the other,
    prove that the length of the shorter piece is 59 inches. -/
theorem shorter_pipe_length (total_length : ℝ) (short_length : ℝ) :
  total_length = 177 →
  total_length = short_length + 2 * short_length →
  short_length = 59 := by
  sorry

end NUMINAMATH_CALUDE_shorter_pipe_length_l2211_221117


namespace NUMINAMATH_CALUDE_rock_splash_width_l2211_221127

theorem rock_splash_width 
  (num_pebbles num_rocks num_boulders : ℕ)
  (total_width pebble_splash_width boulder_splash_width : ℝ)
  (h1 : num_pebbles = 6)
  (h2 : num_rocks = 3)
  (h3 : num_boulders = 2)
  (h4 : total_width = 7)
  (h5 : pebble_splash_width = 1/4)
  (h6 : boulder_splash_width = 2)
  : (total_width - num_pebbles * pebble_splash_width - num_boulders * boulder_splash_width) / num_rocks = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rock_splash_width_l2211_221127


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2211_221112

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ m) → n ≤ m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2211_221112


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2211_221100

def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem correct_quadratic_equation (a b c : ℝ) :
  (∃ b₁ c₁, is_root 1 b₁ c₁ 7 ∧ is_root 1 b₁ c₁ 3) →
  (∃ b₂ c₂, is_root 1 b₂ c₂ 11 ∧ is_root 1 b₂ c₂ (-1)) →
  (a = 1 ∧ b = -10 ∧ c = 32) :=
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2211_221100


namespace NUMINAMATH_CALUDE_train_crossing_time_l2211_221166

/-- Proves that a train of length 500 m, traveling at 180 km/h, takes 10 seconds to cross an electric pole. -/
theorem train_crossing_time :
  let train_length : ℝ := 500  -- Length of the train in meters
  let train_speed_kmh : ℝ := 180  -- Speed of the train in km/h
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600  -- Speed in m/s
  let crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross the pole
  crossing_time = 10 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2211_221166


namespace NUMINAMATH_CALUDE_salary_increase_proof_l2211_221139

/-- Proves that given an employee's new annual salary of $90,000 after a 38.46153846153846% increase, the amount of the salary increase is $25,000. -/
theorem salary_increase_proof (new_salary : ℝ) (percent_increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : percent_increase = 38.46153846153846) : 
  new_salary - (new_salary / (1 + percent_increase / 100)) = 25000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l2211_221139


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2211_221150

theorem min_value_quadratic :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + 6 * x + 1487
  ∃ (m : ℝ), m = 1484 ∧ ∀ x, f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2211_221150


namespace NUMINAMATH_CALUDE_S_seven_two_l2211_221123

def S (a b : ℕ) : ℕ := 3 * a + 5 * b

theorem S_seven_two : S 7 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_S_seven_two_l2211_221123


namespace NUMINAMATH_CALUDE_square_trinomial_equality_l2211_221178

theorem square_trinomial_equality : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_trinomial_equality_l2211_221178


namespace NUMINAMATH_CALUDE_cos_symmetry_l2211_221182

theorem cos_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.cos (2 * x + π / 3)
  let symmetry_point := π / 3
  f (symmetry_point + x) = f (symmetry_point - x) := by
  sorry

#check cos_symmetry

end NUMINAMATH_CALUDE_cos_symmetry_l2211_221182


namespace NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l2211_221162

/-- Represents the systematic sampling method for a population --/
structure SystematicSampling where
  populationSize : Nat
  numGroups : Nat
  sampleSize : Nat
  firstDrawn : Nat

/-- Calculates the number drawn for a given group --/
def SystematicSampling.numberDrawn (s : SystematicSampling) (group : Nat) : Nat :=
  let offset := (s.firstDrawn + 33 * group) % 100
  let baseNumber := (group - 1) * (s.populationSize / s.numGroups)
  baseNumber + offset

/-- The main theorem to prove --/
theorem systematic_sampling_seventh_group 
  (s : SystematicSampling)
  (h1 : s.populationSize = 1000)
  (h2 : s.numGroups = 10)
  (h3 : s.sampleSize = 10)
  (h4 : s.firstDrawn = 57) :
  s.numberDrawn 7 = 688 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l2211_221162


namespace NUMINAMATH_CALUDE_circle_centers_locus_l2211_221192

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16

-- Define the property of being externally tangent to C₁
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

-- Define the property of being internally tangent to C₂
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (4 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := 84 * a^2 + 100 * b^2 - 168 * a - 441 = 0

-- State the theorem
theorem circle_centers_locus (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) ↔ locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_circle_centers_locus_l2211_221192


namespace NUMINAMATH_CALUDE_triangle_problem_l2211_221186

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : abc.c = 13)
  (h2 : Real.cos abc.A = 5/13) :
  (abc.a = 36 → Real.sin abc.C = 1/3) ∧ 
  (abc.a * abc.b * Real.sin abc.C / 2 = 6 → abc.a = 4 * Real.sqrt 10 ∧ abc.b = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2211_221186


namespace NUMINAMATH_CALUDE_boa_constrictor_length_is_70_l2211_221164

/-- The length of the garden snake in inches -/
def garden_snake_length : ℕ := 10

/-- The factor by which the boa constrictor is longer than the garden snake -/
def boa_length_factor : ℕ := 7

/-- The length of the boa constrictor in inches -/
def boa_constrictor_length : ℕ := garden_snake_length * boa_length_factor

theorem boa_constrictor_length_is_70 : boa_constrictor_length = 70 := by
  sorry

end NUMINAMATH_CALUDE_boa_constrictor_length_is_70_l2211_221164


namespace NUMINAMATH_CALUDE_unique_divisible_by_13_l2211_221169

def base_7_to_10 (d : Nat) : Nat :=
  3 * 7^3 + d * 7^2 + d * 7 + 6

theorem unique_divisible_by_13 : 
  ∃! d : Nat, d < 7 ∧ (base_7_to_10 d) % 13 = 0 ∧ base_7_to_10 d = 1035 + 56 * d :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_13_l2211_221169


namespace NUMINAMATH_CALUDE_increasing_function_sum_inequality_l2211_221125

/-- An increasing function on the real line. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: For an increasing function f and real numbers a and b,
    if a + b ≥ 0, then f(a) + f(b) ≥ f(-a) + f(-b). -/
theorem increasing_function_sum_inequality
  (f : ℝ → ℝ) (hf : IncreasingFunction f) (a b : ℝ) :
  a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_sum_inequality_l2211_221125


namespace NUMINAMATH_CALUDE_max_d_value_l2211_221135

def a (n : ℕ+) : ℕ := 99 + 2 * n ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 11) ∧ (∀ (n : ℕ+), d n ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2211_221135


namespace NUMINAMATH_CALUDE_line_plane_intersection_l2211_221191

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (intersects : Line → Plane → Prop)

-- State the theorem
theorem line_plane_intersection
  (a b : Line) (α : Plane)
  (h1 : parallel a α)
  (h2 : perpendicular b a) :
  intersects b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l2211_221191


namespace NUMINAMATH_CALUDE_triangle_side_length_l2211_221111

/-- Represents a triangle with sides a, b, c and heights ha, hb, hc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ

/-- Theorem: In a triangle with sides AC = 6 cm and BC = 3 cm, 
    if the half-sum of heights to AC and BC equals the height to AB, 
    then AB = 4 cm -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.b = 6)
  (h2 : t.c = 3)
  (h3 : (t.ha + t.hb) / 2 = t.hc) : 
  t.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2211_221111


namespace NUMINAMATH_CALUDE_additional_charging_time_l2211_221198

/-- Represents the charging characteristics of a mobile battery -/
structure BatteryCharging where
  initial_charge_time : ℕ  -- Time to reach 20% charge in minutes
  initial_charge_percent : ℕ  -- Initial charge percentage
  total_charge_time : ℕ  -- Total time to reach P% charge in minutes

/-- Theorem stating the additional charging time -/
theorem additional_charging_time (b : BatteryCharging) 
  (h1 : b.initial_charge_time = 60)  -- 1 hour = 60 minutes
  (h2 : b.initial_charge_percent = 20)
  (h3 : b.total_charge_time = b.initial_charge_time + 150) :
  b.total_charge_time - b.initial_charge_time = 150 := by
  sorry

#check additional_charging_time

end NUMINAMATH_CALUDE_additional_charging_time_l2211_221198


namespace NUMINAMATH_CALUDE_money_distribution_l2211_221134

theorem money_distribution (a b c d : ℕ) : 
  a + b + c + d = 2000 →
  a + c = 900 →
  b + c = 1100 →
  a + d = 700 →
  c = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2211_221134


namespace NUMINAMATH_CALUDE_sample_capacity_l2211_221167

/-- Given a sample divided into groups, prove that the total sample capacity is 144
    when one group has a frequency of 36 and a frequency rate of 0.25. -/
theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) : 
  frequency = 36 → frequency_rate = 1/4 → n = frequency / frequency_rate → n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l2211_221167


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_l2211_221132

theorem power_of_two_plus_one_square (k z : ℕ) :
  2^k + 1 = z^2 → k = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_l2211_221132


namespace NUMINAMATH_CALUDE_rectangle_width_l2211_221110

/-- A rectangle with a perimeter of 20 cm and length 2 cm more than its width has a width of 4 cm. -/
theorem rectangle_width (w : ℝ) (h1 : 2 * (w + 2) + 2 * w = 20) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2211_221110


namespace NUMINAMATH_CALUDE_special_right_triangle_pair_theorem_l2211_221176

/-- Two right triangles with specific properties -/
structure SpecialRightTrianglePair where
  /-- The length of the common leg -/
  x : ℝ
  /-- The length of the other leg of T₁ -/
  y : ℝ
  /-- The length of the hypotenuse of T₁ -/
  w : ℝ
  /-- The length of the other leg of T₂ -/
  v : ℝ
  /-- The length of the hypotenuse of T₂ -/
  z : ℝ
  /-- Area of T₁ is 3 -/
  area_t1 : x * y / 2 = 3
  /-- Area of T₂ is 4 -/
  area_t2 : x * v / 2 = 4
  /-- Hypotenuse of T₁ is twice the length of the hypotenuse of T₂ -/
  hypotenuse_relation : w = 2 * z
  /-- Pythagorean theorem for T₁ -/
  pythagorean_t1 : x^2 + y^2 = w^2
  /-- Pythagorean theorem for T₂ -/
  pythagorean_t2 : x^2 + v^2 = z^2

/-- The square of the product of the third sides is 2304/25 -/
theorem special_right_triangle_pair_theorem (t : SpecialRightTrianglePair) :
  (t.y * t.v)^2 = 2304/25 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_pair_theorem_l2211_221176


namespace NUMINAMATH_CALUDE_cookie_bags_count_l2211_221159

theorem cookie_bags_count (cookies_per_bag : ℕ) (total_cookies : ℕ) (h1 : cookies_per_bag = 19) (h2 : total_cookies = 703) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_count_l2211_221159


namespace NUMINAMATH_CALUDE_families_with_car_or_ebike_l2211_221101

theorem families_with_car_or_ebike (total_car : ℕ) (total_ebike : ℕ) (both : ℕ) :
  total_car = 35 → total_ebike = 65 → both = 20 →
  total_car + total_ebike - both = 80 := by
  sorry

end NUMINAMATH_CALUDE_families_with_car_or_ebike_l2211_221101


namespace NUMINAMATH_CALUDE_expression_value_l2211_221174

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2211_221174


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2211_221185

theorem difference_of_numbers (x y : ℝ) (h_sum : x + y = 36) (h_product : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2211_221185


namespace NUMINAMATH_CALUDE_inequality_proof_l2211_221187

theorem inequality_proof (h : Real.log (1/2) < 0) : (1/2)^3 < (1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2211_221187


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2211_221130

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (·*·) 1

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  (∀ m : ℕ, m > 0 → a m * a (m + 2) = 2 * a (m + 1)) →
  SequenceProduct a (2 * m + 1) = 128 →
  m = 3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2211_221130


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l2211_221154

theorem mystery_book_shelves :
  let books_per_shelf : ℕ := 4
  let picture_book_shelves : ℕ := 3
  let total_books : ℕ := 32
  let mystery_book_shelves : ℕ := (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf
  mystery_book_shelves = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l2211_221154


namespace NUMINAMATH_CALUDE_equal_area_floors_width_l2211_221175

/-- Represents the dimensions of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the area of a rectangular floor -/
def area (f : Floor) : ℝ := f.length * f.width

theorem equal_area_floors_width :
  ∀ (X Y : Floor),
  area X = area Y →
  X.length = 18 →
  X.width = 10 →
  Y.length = 20 →
  Y.width = 9 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_floors_width_l2211_221175


namespace NUMINAMATH_CALUDE_square_roots_problem_l2211_221171

theorem square_roots_problem (x m : ℝ) : 
  x > 0 ∧ 
  (m + 3)^2 = x ∧ 
  (2*m - 15)^2 = x ∧ 
  m + 3 ≠ 2*m - 15 → 
  x = 49 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2211_221171


namespace NUMINAMATH_CALUDE_factorial_sum_equals_power_of_two_l2211_221170

theorem factorial_sum_equals_power_of_two (a b c : ℕ+) : 
  (Nat.factorial a.val + Nat.factorial b.val = 2^(Nat.factorial c.val)) ↔ 
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) := by
sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_power_of_two_l2211_221170


namespace NUMINAMATH_CALUDE_grandma_inheritance_l2211_221113

theorem grandma_inheritance (total : ℕ) (shelby_share : ℕ) (remaining_grandchildren : ℕ) :
  total = 124600 →
  shelby_share = total / 2 →
  remaining_grandchildren = 10 →
  (total - shelby_share) / remaining_grandchildren = 6230 :=
by sorry

end NUMINAMATH_CALUDE_grandma_inheritance_l2211_221113


namespace NUMINAMATH_CALUDE_m_range_l2211_221168

theorem m_range (m : ℝ) : 
  (2 * 3 - m > 4) ∧ (2 * 2 - m ≤ 4) → 0 ≤ m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2211_221168


namespace NUMINAMATH_CALUDE_average_weight_increase_l2211_221115

theorem average_weight_increase (initial_count : ℕ) (original_weight replaced_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  original_weight = 101 →
  (original_weight - replaced_weight) / initial_count = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2211_221115


namespace NUMINAMATH_CALUDE_largest_consecutive_non_prime_under_50_l2211_221147

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_consecutive_non_prime_under_50 (a b c d e f : ℕ) :
  a < 100 ∧ b < 100 ∧ c < 100 ∧ d < 100 ∧ e < 100 ∧ f < 100 →  -- two-digit integers
  a < 50 ∧ b < 50 ∧ c < 50 ∧ d < 50 ∧ e < 50 ∧ f < 50 →  -- less than 50
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧ f = e + 1 →  -- consecutive
  ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ 
  ¬(is_prime d) ∧ ¬(is_prime e) ∧ ¬(is_prime f) →  -- not prime
  f = 37 :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_non_prime_under_50_l2211_221147


namespace NUMINAMATH_CALUDE_odd_prime_divisor_condition_l2211_221148

theorem odd_prime_divisor_condition (n : ℕ) :
  (n > 0 ∧ ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) ↔ (Nat.Prime n ∧ n % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_odd_prime_divisor_condition_l2211_221148


namespace NUMINAMATH_CALUDE_prove_arrangements_l2211_221177

def num_students : ℕ := 7

def adjacent_pair : ℕ := 1
def non_adjacent_pair : ℕ := 1
def remaining_students : ℕ := num_students - 4

def arrangements_theorem : Prop :=
  (num_students = 7) →
  (adjacent_pair = 1) →
  (non_adjacent_pair = 1) →
  (remaining_students = num_students - 4) →
  (Nat.factorial 2 * Nat.factorial 4 * (Nat.factorial 5 / Nat.factorial 3) =
   Nat.factorial 2 * Nat.factorial 4 * Nat.factorial 5 / Nat.factorial 3)

theorem prove_arrangements : arrangements_theorem := by sorry

end NUMINAMATH_CALUDE_prove_arrangements_l2211_221177


namespace NUMINAMATH_CALUDE_colored_plane_triangles_l2211_221158

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  sideLength : ℝ
  isEquilateral : 
    (a.x - b.x)^2 + (a.y - b.y)^2 = sideLength^2 ∧
    (b.x - c.x)^2 + (b.y - c.y)^2 = sideLength^2 ∧
    (c.x - a.x)^2 + (c.y - a.y)^2 = sideLength^2

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  a : Point
  b : Point
  c : Point
  legLength : ℝ
  isIsoscelesRight :
    (a.x - b.x)^2 + (a.y - b.y)^2 = 2 * legLength^2 ∧
    (b.x - c.x)^2 + (b.y - c.y)^2 = legLength^2 ∧
    (c.x - a.x)^2 + (c.y - a.y)^2 = legLength^2

-- State the theorem
theorem colored_plane_triangles (coloring : Coloring) :
  (∃ t : EquilateralTriangle, 
    (t.sideLength = 673 * Real.sqrt 3 ∨ t.sideLength = 2019) ∧
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) ∧
  (∃ t : IsoscelesRightTriangle,
    (t.legLength = 1010 * Real.sqrt 2 ∨ t.legLength = 2020) ∧
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) :=
by sorry

end NUMINAMATH_CALUDE_colored_plane_triangles_l2211_221158


namespace NUMINAMATH_CALUDE_gcd_six_digit_repeated_is_1001_l2211_221161

/-- A function that generates a six-digit number by repeating a three-digit number -/
def repeat_three_digit (n : ℕ) : ℕ :=
  1001 * n

/-- The set of all six-digit numbers formed by repeating a three-digit number -/
def six_digit_repeated_set : Set ℕ :=
  {m | ∃ n, 100 ≤ n ∧ n < 1000 ∧ m = repeat_three_digit n}

/-- Theorem stating that the greatest common divisor of all numbers in the set is 1001 -/
theorem gcd_six_digit_repeated_is_1001 :
  ∃ d, d > 0 ∧ (∀ m ∈ six_digit_repeated_set, d ∣ m) ∧
  (∀ k, k > 0 → (∀ m ∈ six_digit_repeated_set, k ∣ m) → k ≤ d) ∧ d = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_six_digit_repeated_is_1001_l2211_221161


namespace NUMINAMATH_CALUDE_first_digit_891_base8_l2211_221153

/-- Represents a positive integer in a given base --/
def BaseRepresentation (n : ℕ+) (base : ℕ) : List ℕ :=
  sorry

/-- Returns the first (leftmost) digit of a number's representation in a given base --/
def firstDigit (n : ℕ+) (base : ℕ) : ℕ :=
  match BaseRepresentation n base with
  | [] => 0  -- This case should never occur for positive integers
  | d::_ => d

theorem first_digit_891_base8 :
  firstDigit 891 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_891_base8_l2211_221153


namespace NUMINAMATH_CALUDE_ninth_term_is_18_l2211_221193

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  first_term : a 1 = 1 / 2
  condition : a 2 * a 8 = 2 * a 5 + 3

/-- The 9th term of the geometric sequence is 18 -/
theorem ninth_term_is_18 (seq : GeometricSequence) : seq.a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_18_l2211_221193


namespace NUMINAMATH_CALUDE_inequality_proof_l2211_221141

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 1/2) : 
  (1 - a + c) / (Real.sqrt c * (Real.sqrt a + 2 * Real.sqrt b)) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2211_221141


namespace NUMINAMATH_CALUDE_duct_tape_cutting_time_l2211_221124

/-- The time required to cut all strands of duct tape -/
def cutting_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  total_strands / (hannah_rate + son_rate)

/-- Theorem stating the time required to cut all strands -/
theorem duct_tape_cutting_time :
  cutting_time 22 8 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_duct_tape_cutting_time_l2211_221124


namespace NUMINAMATH_CALUDE_inverse_of_7_mod_45_l2211_221196

theorem inverse_of_7_mod_45 : ∃ x : ℤ, 0 ≤ x ∧ x < 45 ∧ (7 * x) % 45 = 1 :=
  by
  use 32
  sorry

end NUMINAMATH_CALUDE_inverse_of_7_mod_45_l2211_221196


namespace NUMINAMATH_CALUDE_range_of_a_l2211_221102

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a : 
  (∀ x, p x → (∀ a, q x a)) ∧ 
  (∃ x a, ¬(p x) ∧ q x a) → 
  ∀ a, 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2211_221102


namespace NUMINAMATH_CALUDE_rectangle_area_l2211_221118

/-- Given a rectangle where the length is 3 times the width and the width is 6 inches,
    prove that the area is 108 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 6 →
  length = 3 * width →
  area = length * width →
  area = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2211_221118


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2211_221181

theorem simplify_radical_product (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (40 * x) * Real.sqrt (45 * x) * Real.sqrt (56 * x) = 120 * Real.sqrt (7 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2211_221181


namespace NUMINAMATH_CALUDE_remaining_three_digit_numbers_l2211_221103

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical adjacent digits and a different third digit -/
def excluded_numbers : ℕ := 162

/-- The remaining count of three-digit numbers after exclusion -/
def remaining_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem remaining_three_digit_numbers : remaining_numbers = 738 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_digit_numbers_l2211_221103


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l2211_221105

/-- The average birth rate in people per two seconds -/
def average_birth_rate : ℝ := 5

/-- The death rate in people per two seconds -/
def death_rate : ℝ := 3

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℝ := 43200

/-- The net population increase in one day -/
def net_increase_per_day : ℝ := 86400

theorem birth_rate_calculation :
  (average_birth_rate - death_rate) * intervals_per_day = net_increase_per_day :=
by sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l2211_221105


namespace NUMINAMATH_CALUDE_complement_A_union_B_equals_interval_intersection_A_B_empty_t_range_l2211_221136

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * 3 * x + 10

-- Define set A
def A : Set ℝ := {x | f x > 0}

-- Define set B
def B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1}

-- Theorem for part (I)
theorem complement_A_union_B_equals_interval :
  (Aᶜ ∪ B 1) = {x | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Theorem for part (II)
theorem intersection_A_B_empty_t_range (t : ℝ) :
  (A ∩ B t = ∅) ↔ (-2 ≤ t ∧ t ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_complement_A_union_B_equals_interval_intersection_A_B_empty_t_range_l2211_221136


namespace NUMINAMATH_CALUDE_bucket_weight_l2211_221133

theorem bucket_weight (c d : ℝ) : 
  (∃ (x y : ℝ), x + 3/4 * y = c ∧ x + 1/3 * y = d) → 
  (∃ (full_weight : ℝ), full_weight = (8*c - 3*d) / 5) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l2211_221133


namespace NUMINAMATH_CALUDE_function_non_negative_implies_bounds_l2211_221119

theorem function_non_negative_implies_bounds 
  (a b A B : ℝ) 
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_bounds_l2211_221119


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2211_221140

theorem polynomial_division_remainder :
  ∀ (R : Polynomial ℤ) (Q : Polynomial ℤ),
    (Polynomial.degree R < 2) →
    (x^101 : Polynomial ℤ) = (x^2 - 3*x + 2) * Q + R →
    R = 2^101 * (x - 1) - (x - 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2211_221140


namespace NUMINAMATH_CALUDE_smallest_class_size_class_size_satisfies_conditions_l2211_221149

theorem smallest_class_size (n : ℕ) : (n ≡ 1 [ZMOD 6] ∧ n ≡ 2 [ZMOD 8] ∧ n ≡ 4 [ZMOD 10]) → n ≥ 274 :=
by sorry

theorem class_size_satisfies_conditions : 274 ≡ 1 [ZMOD 6] ∧ 274 ≡ 2 [ZMOD 8] ∧ 274 ≡ 4 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_class_size_satisfies_conditions_l2211_221149


namespace NUMINAMATH_CALUDE_sequence_matches_given_terms_l2211_221194

/-- Definition of the sequence -/
def a (n : ℕ) : ℚ := (n + 2 : ℚ) / (2 * n + 3 : ℚ)

/-- The theorem stating that the first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  (a 1 = 3 / 5) ∧ (a 2 = 4 / 7) ∧ (a 3 = 5 / 9) ∧ (a 4 = 6 / 11) :=
by sorry

end NUMINAMATH_CALUDE_sequence_matches_given_terms_l2211_221194


namespace NUMINAMATH_CALUDE_no_nonzero_integer_solution_l2211_221104

theorem no_nonzero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_integer_solution_l2211_221104


namespace NUMINAMATH_CALUDE_cosine_set_product_l2211_221172

open Real Set

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = cos (arithmeticSequence a₁ (2 * π / 3) n)}

theorem cosine_set_product (a₁ : ℝ) :
  (∃ a b : ℝ, S a₁ = {a, b} ∧ a ≠ b) → 
  ∀ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_set_product_l2211_221172


namespace NUMINAMATH_CALUDE_three_points_determine_plane_l2211_221121

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space using the general equation ax + by + cz + d = 0
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if two planes are perpendicular
def perpendicularPlanes (p1 p2 : Plane) : Prop :=
  p1.a * p2.a + p1.b * p2.b + p1.c * p2.c = 0

-- Function to check if a point lies on a plane
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Theorem statement
theorem three_points_determine_plane 
  (p1 p2 p3 : Plane) 
  (point1 point2 point3 : Point3D) : 
  perpendicularPlanes p1 p2 ∧ 
  perpendicularPlanes p2 p3 ∧ 
  perpendicularPlanes p3 p1 ∧ 
  pointOnPlane point1 p1 ∧ 
  pointOnPlane point2 p2 ∧ 
  pointOnPlane point3 p3 → 
  ∃! (resultPlane : Plane), 
    pointOnPlane point1 resultPlane ∧ 
    pointOnPlane point2 resultPlane ∧ 
    pointOnPlane point3 resultPlane :=
by
  sorry

end NUMINAMATH_CALUDE_three_points_determine_plane_l2211_221121


namespace NUMINAMATH_CALUDE_jake_present_weight_l2211_221156

/-- Jake's present weight in pounds -/
def jake_weight : ℕ := sorry

/-- Jake's sister's weight in pounds -/
def sister_weight : ℕ := sorry

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℕ := 224

theorem jake_present_weight : 
  (jake_weight - 20 = 2 * sister_weight) ∧ 
  (jake_weight + sister_weight = combined_weight) → 
  jake_weight = 156 := by
sorry

end NUMINAMATH_CALUDE_jake_present_weight_l2211_221156


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2211_221107

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ (∃ x : ℝ, Real.log (x^2 + 1) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2211_221107


namespace NUMINAMATH_CALUDE_line_and_segment_properties_l2211_221114

-- Define the points and lines
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (1, -1)
def C : ℝ × ℝ := (0, 2)

def line_l (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line_m (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the theorem
theorem line_and_segment_properties :
  -- Given conditions
  (∃ a : ℝ, A = (2, a) ∧ B = (a, -1)) →
  (∀ x y : ℝ, line_l x y ↔ ∃ t : ℝ, x = 2 * (1 - t) + t ∧ y = 1 * (1 - t) + (-1) * t) →
  (∀ x y : ℝ, line_l x y → line_m (x + 1) (y + 1)) →
  -- Conclusions
  (∀ x y : ℝ, line_l x y ↔ 2 * x - y - 3 = 0) ∧
  Real.sqrt 10 = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_line_and_segment_properties_l2211_221114


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_l2211_221109

/-- Given two points P and Q, prove that if their midpoint has x-coordinate 18, then the x-coordinate of P is 6. -/
theorem midpoint_x_coordinate (a : ℝ) : 
  let P : ℝ × ℝ := (a, 2)
  let Q : ℝ × ℝ := (30, -6)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  midpoint.1 = 18 → a = 6 := by
  sorry

#check midpoint_x_coordinate

end NUMINAMATH_CALUDE_midpoint_x_coordinate_l2211_221109


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2211_221151

/-- Given a quadratic equation 2x^2 - 1 = 6x, prove its coefficients -/
theorem quadratic_equation_coefficients :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - 1 - 6 * x
  ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ a = 2 ∧ b = -6 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2211_221151


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l2211_221189

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ (n / 100000 = 2)

def move_first_to_last (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

theorem unique_six_digit_number : 
  ∀ n : ℕ, is_valid_number n → (move_first_to_last n = 3 * n) → n = 285714 :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l2211_221189


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2211_221122

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≥ 1}

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- Theorem statement
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -1 ∨ x = 1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2211_221122


namespace NUMINAMATH_CALUDE_expected_value_8_sided_die_l2211_221160

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  let outcomes := standard_8_sided_die
  let prob (n : ℕ) := if n ∈ outcomes then (1 : ℚ) / 8 else 0
  let value (n : ℕ) := n + 1
  Finset.sum outcomes (λ n ↦ prob n * value n) = (9 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_8_sided_die_l2211_221160


namespace NUMINAMATH_CALUDE_angle_equivalence_l2211_221173

/-- Given α = 2022°, if β has the same terminal side as α and β ∈ (0, 2π), then β = 37π/30 radians. -/
theorem angle_equivalence (α β : Real) : 
  α = 2022 * (π / 180) →  -- Convert 2022° to radians
  (∃ k : ℤ, β = α + 2 * π * k) →  -- Same terminal side
  0 < β ∧ β < 2 * π →  -- β ∈ (0, 2π)
  β = 37 * π / 30 := by
sorry

end NUMINAMATH_CALUDE_angle_equivalence_l2211_221173


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2211_221163

/-- The polynomial f(x) = x^5 - 5x^4 + 8x^3 + 25x^2 - 14x - 40 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 8*x^3 + 25*x^2 - 14*x - 40

/-- The remainder when f(x) is divided by (x-2) -/
def remainder : ℝ := f 2

theorem polynomial_remainder : remainder = 48 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2211_221163


namespace NUMINAMATH_CALUDE_max_four_digit_sum_l2211_221142

def A (s n k : ℕ) : ℕ :=
  if n = 1 then
    if 1 ≤ s ∧ s ≤ k then 1 else 0
  else if s < n then 0
  else if k = 0 then 0
  else A (s - k) (n - 1) (k - 1) + A s n (k - 1)

theorem max_four_digit_sum :
  (∀ s, s ≠ 20 → A s 4 9 ≤ A 20 4 9) ∧
  A 20 4 9 = 12 := by sorry

end NUMINAMATH_CALUDE_max_four_digit_sum_l2211_221142


namespace NUMINAMATH_CALUDE_gate_buyers_count_l2211_221157

/-- The number of people who pre-bought tickets -/
def preBuyers : ℕ := 20

/-- The price of a pre-bought ticket -/
def prePrice : ℕ := 155

/-- The price of a ticket bought at the gate -/
def gatePrice : ℕ := 200

/-- The additional amount paid by gate buyers compared to pre-buyers -/
def additionalAmount : ℕ := 2900

/-- The number of people who bought tickets at the gate -/
def gateBuyers : ℕ := 30

theorem gate_buyers_count :
  gateBuyers * gatePrice = preBuyers * prePrice + additionalAmount := by
  sorry

end NUMINAMATH_CALUDE_gate_buyers_count_l2211_221157


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l2211_221180

theorem ordered_pairs_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 36) (Finset.range 37 ×ˢ Finset.range 37)).card ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l2211_221180


namespace NUMINAMATH_CALUDE_inscribed_circle_segment_ratios_l2211_221165

/-- Given a triangle with sides in ratio 5:4:3, prove the ratios of segments divided by tangent points of inscribed circle -/
theorem inscribed_circle_segment_ratios (a b c : ℝ) (h : a / b = 5 / 4 ∧ b / c = 4 / 3) :
  let r := (a + b - c) / 2
  let s := (a + b + c) / 2
  (r / (s - b), r / (s - c), (s - c) / (s - b)) = (1 / 3, 1 / 2, 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_segment_ratios_l2211_221165


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_l2211_221195

/-- Given points A, B, C, D, E, and F in a coordinate plane, where:
    A is at (0,8), B at (0,0), C at (10,0)
    D is the midpoint of AB
    E is the midpoint of BC
    F is the intersection of lines AE and CD
    Prove that the sum of F's coordinates is 6 -/
theorem intersection_coordinate_sum :
  let A : ℝ × ℝ := (0, 8)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (10, 0)
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let m_AE : ℝ := (E.2 - A.2) / (E.1 - A.1)
  let b_AE : ℝ := A.2 - m_AE * A.1
  let m_CD : ℝ := (D.2 - C.2) / (D.1 - C.1)
  let b_CD : ℝ := C.2 - m_CD * C.1
  let F : ℝ × ℝ := ((b_CD - b_AE) / (m_AE - m_CD), m_AE * ((b_CD - b_AE) / (m_AE - m_CD)) + b_AE)
  F.1 + F.2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_intersection_coordinate_sum_l2211_221195


namespace NUMINAMATH_CALUDE_reciprocal_equal_self_l2211_221126

theorem reciprocal_equal_self (x : ℝ) : x ≠ 0 ∧ x = 1 / x ↔ x = 1 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_equal_self_l2211_221126


namespace NUMINAMATH_CALUDE_all_red_raise_hands_eventually_l2211_221146

/-- Represents the color of a stamp -/
inductive StampColor
| Red
| Green

/-- Represents a faculty member -/
structure FacultyMember where
  stamp : StampColor

/-- Represents the state of the game on a given day -/
structure GameState where
  day : ℕ
  faculty : List FacultyMember
  handsRaised : List FacultyMember

/-- Predicate to check if a faculty member raises their hand -/
def raisesHand (member : FacultyMember) (state : GameState) : Prop :=
  member ∈ state.handsRaised

/-- The main theorem to be proved -/
theorem all_red_raise_hands_eventually 
  (n : ℕ) 
  (faculty : List FacultyMember) 
  (h1 : faculty.length = n) 
  (h2 : ∃ m, m ∈ faculty ∧ m.stamp = StampColor.Red) :
  ∃ (finalState : GameState), 
    finalState.day = n ∧ 
    ∀ m, m ∈ faculty → m.stamp = StampColor.Red → raisesHand m finalState :=
  sorry


end NUMINAMATH_CALUDE_all_red_raise_hands_eventually_l2211_221146


namespace NUMINAMATH_CALUDE_brooke_jumping_jacks_l2211_221190

def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50

def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

def brooke_multiplier : ℕ := 3

theorem brooke_jumping_jacks : sidney_total * brooke_multiplier = 438 := by
  sorry

end NUMINAMATH_CALUDE_brooke_jumping_jacks_l2211_221190


namespace NUMINAMATH_CALUDE_largest_C_gap_l2211_221106

/-- Represents a square on the chessboard -/
structure Square :=
  (row : Fin 8)
  (col : Fin 8)

/-- The chessboard is an 8x8 grid of squares -/
def Chessboard := Fin 8 → Fin 8 → Square

/-- Two squares are adjacent if they share a side or vertex -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row = s2.row ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col = s2.col) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col = s2.col) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val = s2.col.val + 1)

/-- A numbering of the chessboard is a function assigning each square a unique number from 1 to 64 -/
def Numbering := Square → Fin 64

/-- A C-gap is a number g such that for every numbering, there exist two adjacent squares whose numbers differ by at least g -/
def is_C_gap (g : ℕ) : Prop :=
  ∀ (n : Numbering), ∃ (s1 s2 : Square), 
    adjacent s1 s2 ∧ |n s1 - n s2| ≥ g

/-- The theorem stating that the largest C-gap for an 8x8 chessboard is 9 -/
theorem largest_C_gap : 
  (is_C_gap 9) ∧ (∀ g : ℕ, g > 9 → ¬(is_C_gap g)) :=
sorry

end NUMINAMATH_CALUDE_largest_C_gap_l2211_221106


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2211_221129

theorem reciprocal_problem (x : ℝ) (h : 6 * x = 12) : 150 * (1 / x) = 75 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2211_221129


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2211_221197

theorem hemisphere_surface_area (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  3 * Real.pi * r^2 = 972 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2211_221197


namespace NUMINAMATH_CALUDE_expression_evaluation_l2211_221145

theorem expression_evaluation : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2211_221145


namespace NUMINAMATH_CALUDE_earliest_retirement_year_l2211_221116

/-- Rule of 70 retirement eligibility function -/
def eligible_to_retire (current_year : ℕ) (hire_year : ℕ) (hire_age : ℕ) : Prop :=
  (current_year - hire_year) + (hire_age + (current_year - hire_year)) ≥ 70

/-- Theorem: The earliest retirement year for an employee hired in 1989 at age 32 is 2008 -/
theorem earliest_retirement_year :
  ∀ year : ℕ, year ≥ 1989 →
  (eligible_to_retire year 1989 32 ↔ year ≥ 2008) :=
by sorry

end NUMINAMATH_CALUDE_earliest_retirement_year_l2211_221116


namespace NUMINAMATH_CALUDE_tangent_line_at_1_0_l2211_221128

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_1_0 :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := f' p.1
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  (∀ x, tangent_line x = x - 1) ∧ f p.1 = p.2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_at_1_0_l2211_221128


namespace NUMINAMATH_CALUDE_both_balls_prob_at_least_one_ball_prob_l2211_221183

/-- The probability space for the ball experiment -/
structure BallProbSpace where
  /-- The probability of ball A falling into the box -/
  prob_A : ℝ
  /-- The probability of ball B falling into the box -/
  prob_B : ℝ
  /-- The probability of ball A falling into the box is 1/2 -/
  hA : prob_A = 1/2
  /-- The probability of ball B falling into the box is 1/3 -/
  hB : prob_B = 1/3
  /-- The events A and B are independent -/
  indep : ∀ {p : ℝ} {q : ℝ}, p = prob_A → q = prob_B → p * q = prob_A * prob_B

/-- The probability that both ball A and ball B fall into the box is 1/6 -/
theorem both_balls_prob (space : BallProbSpace) : space.prob_A * space.prob_B = 1/6 := by
  sorry

/-- The probability that at least one of ball A and ball B falls into the box is 2/3 -/
theorem at_least_one_ball_prob (space : BallProbSpace) :
  1 - (1 - space.prob_A) * (1 - space.prob_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_both_balls_prob_at_least_one_ball_prob_l2211_221183


namespace NUMINAMATH_CALUDE_factorization_difference_l2211_221131

theorem factorization_difference (c d : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d)) → c - d = 8 := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_l2211_221131


namespace NUMINAMATH_CALUDE_quadratic_increasing_after_vertex_l2211_221108

def f (x : ℝ) : ℝ := (x - 1)^2 + 5

theorem quadratic_increasing_after_vertex (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > x1) : 
  f x2 > f x1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_after_vertex_l2211_221108


namespace NUMINAMATH_CALUDE_floor_abs_negative_34_1_l2211_221155

theorem floor_abs_negative_34_1 : ⌊|(-34.1 : ℝ)|⌋ = 34 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_34_1_l2211_221155


namespace NUMINAMATH_CALUDE_pet_shop_inventory_l2211_221184

/-- Given a pet shop with dogs, cats, and bunnies in stock, prove the total number of dogs and bunnies. -/
theorem pet_shop_inventory (dogs cats bunnies : ℕ) : 
  dogs = 112 →
  dogs / bunnies = 4 / 9 →
  dogs / cats = 4 / 7 →
  dogs + bunnies = 364 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_inventory_l2211_221184


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2211_221144

def initial_price : ℝ := 50
def first_year_increase : ℝ := 2  -- 200% increase
def second_year_decrease : ℝ := 0.5  -- 50% decrease

def final_price : ℝ :=
  initial_price * (1 + first_year_increase) * second_year_decrease

theorem stock_price_calculation :
  final_price = 75 := by sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2211_221144


namespace NUMINAMATH_CALUDE_stating_election_cases_l2211_221120

/-- Represents the number of candidates for the election -/
def num_candidates : ℕ := 3

/-- Represents the number of positions to be filled -/
def num_positions : ℕ := 2

/-- 
Theorem stating that the number of ways to select a president and vice president 
from a group of three people, where one person cannot hold both positions, is equal to 6.
-/
theorem election_cases : 
  num_candidates * (num_candidates - 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_stating_election_cases_l2211_221120
