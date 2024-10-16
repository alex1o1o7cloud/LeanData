import Mathlib

namespace NUMINAMATH_CALUDE_exists_positive_c_less_than_sum_l2931_293169

theorem exists_positive_c_less_than_sum (a b : ℝ) (h : a < b) :
  ∃ c : ℝ, c > 0 ∧ a < b + c := by
sorry

end NUMINAMATH_CALUDE_exists_positive_c_less_than_sum_l2931_293169


namespace NUMINAMATH_CALUDE_twentieth_sample_number_l2931_293110

/-- Calculates the nth number in a systematic sample. -/
def systematicSample (totalItems : Nat) (sampleSize : Nat) (firstNumber : Nat) (n : Nat) : Nat :=
  let k := totalItems / sampleSize
  firstNumber + (n - 1) * k

/-- Proves that the 20th number in the systematic sample is 395. -/
theorem twentieth_sample_number 
  (totalItems : Nat) 
  (sampleSize : Nat) 
  (firstNumber : Nat) 
  (h1 : totalItems = 1000) 
  (h2 : sampleSize = 50) 
  (h3 : firstNumber = 15) :
  systematicSample totalItems sampleSize firstNumber 20 = 395 := by
  sorry

#eval systematicSample 1000 50 15 20

end NUMINAMATH_CALUDE_twentieth_sample_number_l2931_293110


namespace NUMINAMATH_CALUDE_school_population_l2931_293162

/-- Given the initial number of girls and boys in a school, and the number of additional girls who joined,
    calculate the total number of pupils after the new girls joined. -/
theorem school_population (initial_girls initial_boys additional_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  additional_girls = 418 →
  initial_girls + initial_boys + additional_girls = 1346 := by
sorry

end NUMINAMATH_CALUDE_school_population_l2931_293162


namespace NUMINAMATH_CALUDE_function_bound_l2931_293190

def ContinuousFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| < |x - y|

theorem function_bound (f : ℝ → ℝ) (h1 : ContinuousFunction f) (h2 : f 0 = f 1) :
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l2931_293190


namespace NUMINAMATH_CALUDE_fish_pond_population_l2931_293123

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  (initial_tagged : ℚ) / (second_catch : ℚ) = (tagged_in_second : ℚ) / (second_catch : ℚ) →
  (initial_tagged * second_catch : ℚ) / (tagged_in_second : ℚ) = 3200 :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l2931_293123


namespace NUMINAMATH_CALUDE_main_triangle_area_l2931_293101

/-- A triangle with a point inside it -/
structure TriangleWithInnerPoint where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The point inside the triangle -/
  inner_point : ℝ × ℝ
  /-- The point is inside the triangle -/
  point_inside : inner_point ∈ triangle

/-- The areas of smaller triangles formed by lines parallel to the sides -/
structure SmallerTriangleAreas where
  /-- The first smaller triangle area -/
  area1 : ℝ
  /-- The second smaller triangle area -/
  area2 : ℝ
  /-- The third smaller triangle area -/
  area3 : ℝ

/-- Calculate the area of the main triangle given the areas of smaller triangles -/
def calculateMainTriangleArea (smaller_areas : SmallerTriangleAreas) : ℝ :=
  sorry

/-- The theorem stating the relationship between smaller triangle areas and the main triangle area -/
theorem main_triangle_area 
  (t : TriangleWithInnerPoint) 
  (areas : SmallerTriangleAreas)
  (h1 : areas.area1 = 16)
  (h2 : areas.area2 = 25)
  (h3 : areas.area3 = 36) :
  calculateMainTriangleArea areas = 225 :=
sorry

end NUMINAMATH_CALUDE_main_triangle_area_l2931_293101


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2931_293157

theorem floor_ceil_sum : ⌊(1.002 : ℝ)⌋ + ⌈(3.998 : ℝ)⌉ + ⌈(-0.999 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2931_293157


namespace NUMINAMATH_CALUDE_paired_with_32_l2931_293114

def numbers : List ℕ := [36, 27, 42, 32, 28, 31, 23, 17]

theorem paired_with_32 (pair_sum : ℕ) 
  (h1 : pair_sum = (numbers.sum / 4))
  (h2 : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a + b = pair_sum → a ≠ b)
  (h3 : ∀ (n : ℕ), n ∈ numbers → ∃ (m : ℕ), m ∈ numbers ∧ m ≠ n ∧ n + m = pair_sum) :
  ∃ (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧ 
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 = pair_sum) ∧
    (32, 27) ∈ pairs :=
sorry

end NUMINAMATH_CALUDE_paired_with_32_l2931_293114


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2931_293120

/-- The repeating decimal 0.overline{43} -/
def repeating_decimal : ℚ := 43 / 99

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = 43 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2931_293120


namespace NUMINAMATH_CALUDE_dans_remaining_money_l2931_293136

/-- Calculates the remaining money after purchases. -/
def remaining_money (initial : ℕ) (candy_price : ℕ) (chocolate_price : ℕ) : ℕ :=
  initial - (candy_price + chocolate_price)

/-- Proves that Dan has $2 left after his purchases. -/
theorem dans_remaining_money :
  remaining_money 7 2 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l2931_293136


namespace NUMINAMATH_CALUDE_students_liking_both_l2931_293117

theorem students_liking_both (total : ℕ) (fries : ℕ) (burgers : ℕ) (neither : ℕ) :
  total = 25 →
  fries = 15 →
  burgers = 10 →
  neither = 6 →
  ∃ (both : ℕ), both = 12 ∧ total = fries + burgers - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_l2931_293117


namespace NUMINAMATH_CALUDE_tangent_line_at_2_2_tangent_lines_through_origin_l2931_293107

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 1

-- Theorem for part (I)
theorem tangent_line_at_2_2 :
  ∃ (m b : ℝ), (∀ x y, y = m*x + b ↔ 5*x - y - 8 = 0) ∧
               f 2 = 2 ∧
               f' 2 = m :=
sorry

-- Theorem for part (II)
theorem tangent_lines_through_origin :
  ∃ (x₁ x₂ : ℝ),
    (f x₁ = 0 ∧ f' x₁ = 1 ∧ x₁ ≠ x₂) ∧
    (f x₂ = 0 ∧ f' x₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_2_tangent_lines_through_origin_l2931_293107


namespace NUMINAMATH_CALUDE_angle_bisector_vector_l2931_293108

/-- Given points A and B in a Cartesian coordinate system, 
    and a point C on the angle bisector of ∠AOB with |OC| = 2, 
    prove that OC has specific coordinates. -/
theorem angle_bisector_vector (A B C : ℝ × ℝ) : 
  A = (0, 1) →
  B = (-3, 4) →
  (C.1 * A.2 = C.2 * A.1 ∧ C.1 * B.2 = C.2 * B.1) → -- C is on angle bisector
  C.1^2 + C.2^2 = 4 → -- |OC| = 2
  C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_vector_l2931_293108


namespace NUMINAMATH_CALUDE_carnival_participants_l2931_293191

theorem carnival_participants (n : ℕ) (masks costumes both : ℕ) : 
  n ≥ 42 →
  masks = (3 * n) / 7 →
  costumes = (5 * n) / 6 →
  both = masks + costumes - n →
  both ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_carnival_participants_l2931_293191


namespace NUMINAMATH_CALUDE_barbara_candies_l2931_293103

/-- Calculates the remaining number of candies Barbara has -/
def remaining_candies (initial : ℝ) (used : ℝ) (received : ℝ) (eaten : ℝ) : ℝ :=
  initial - used + received - eaten

/-- Proves that Barbara has 18.4 candies left -/
theorem barbara_candies : remaining_candies 18.5 4.2 6.8 2.7 = 18.4 := by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_l2931_293103


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2931_293116

theorem cost_price_percentage (cost_price selling_price : ℝ) 
  (h : selling_price = 4 * cost_price) : 
  cost_price / selling_price = 1 / 4 := by
  sorry

#check cost_price_percentage

end NUMINAMATH_CALUDE_cost_price_percentage_l2931_293116


namespace NUMINAMATH_CALUDE_fruit_basket_count_l2931_293187

/-- The number of ways to choose k items from n identical items -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := (n + k - 1).choose k

/-- The number of possible fruit baskets -/
def fruit_baskets (apples oranges : ℕ) : ℕ :=
  (apples) * (choose_with_repetition (oranges + 1) 1)

theorem fruit_basket_count :
  fruit_baskets 7 12 = 91 :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l2931_293187


namespace NUMINAMATH_CALUDE_min_redistributions_correct_l2931_293177

/-- Represents the redistribution process for a deck of 8 cards -/
def redistribute (deck : Vector ℕ 8) : Vector ℕ 8 :=
  sorry

/-- Checks if the deck is in its original order -/
def is_original_order (deck original : Vector ℕ 8) : Prop :=
  deck = original

/-- The minimum number of redistributions needed to restore the original order -/
def min_redistributions : ℕ := 3

/-- Theorem stating that the minimum number of redistributions to restore the original order is 3 -/
theorem min_redistributions_correct (original : Vector ℕ 8) :
  ∃ (n : ℕ), n = min_redistributions ∧
  ∀ (m : ℕ), m < n → ¬(is_original_order ((redistribute^[m]) original) original) ∧
  is_original_order ((redistribute^[n]) original) original :=
sorry

end NUMINAMATH_CALUDE_min_redistributions_correct_l2931_293177


namespace NUMINAMATH_CALUDE_solution_range_l2931_293171

theorem solution_range (x m : ℝ) : 
  (x + m) / 3 - (2 * x - 1) / 2 = m ∧ x ≤ 0 → m ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2931_293171


namespace NUMINAMATH_CALUDE_tangent_segment_length_l2931_293148

-- Define the circle and points
variable (circle : Type) (A B C P Q R : ℝ × ℝ)

-- Define the properties of tangents and points
def is_tangent (point : ℝ × ℝ) (touch_point : ℝ × ℝ) : Prop := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem tangent_segment_length 
  (h1 : is_tangent A B)
  (h2 : is_tangent A C)
  (h3 : is_tangent P Q)
  (h4 : is_tangent R Q)
  (h5 : distance P B = distance P R)
  (h6 : distance A B = 24) :
  distance P Q = 12 := by sorry

end NUMINAMATH_CALUDE_tangent_segment_length_l2931_293148


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l2931_293127

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 < 0
def inequality2 (x : ℝ) : Prop := 2 * x / (x + 1) ≥ 1

-- Define the solution sets
def solution_set1 : Set ℝ := {x | 1/2 < x ∧ x < 1}
def solution_set2 : Set ℝ := {x | x < -1 ∨ x ≥ 1}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 :=
sorry

theorem inequality2_solution : 
  ∀ x : ℝ, x ≠ -1 → (inequality2 x ↔ x ∈ solution_set2) :=
sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l2931_293127


namespace NUMINAMATH_CALUDE_circle_sectors_and_square_area_l2931_293183

/-- Given a circle with radius 6 and two perpendicular diameters, 
    prove that the sum of the areas of two 120° sectors and 
    the square formed by connecting the diameter endpoints 
    is equal to 24π + 144. -/
theorem circle_sectors_and_square_area :
  let r : ℝ := 6
  let sector_angle : ℝ := 120
  let sector_area := (sector_angle / 360) * π * r^2
  let square_side := 2 * r
  let square_area := square_side^2
  2 * sector_area + square_area = 24 * π + 144 := by
sorry

end NUMINAMATH_CALUDE_circle_sectors_and_square_area_l2931_293183


namespace NUMINAMATH_CALUDE_total_count_is_1500_l2931_293154

/-- The number of people counted on the second day -/
def second_day_count : ℕ := 500

/-- The number of people counted on the first day -/
def first_day_count : ℕ := 2 * second_day_count

/-- The total number of people counted over two days -/
def total_count : ℕ := first_day_count + second_day_count

theorem total_count_is_1500 : total_count = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_count_is_1500_l2931_293154


namespace NUMINAMATH_CALUDE_smallest_representable_integer_l2931_293130

theorem smallest_representable_integer :
  ∃ (m n : ℕ+), 11 = 36 * m - 5 * n ∧
  ∀ (k : ℕ+) (m' n' : ℕ+), k < 11 → k ≠ 36 * m' - 5 * n' :=
sorry

end NUMINAMATH_CALUDE_smallest_representable_integer_l2931_293130


namespace NUMINAMATH_CALUDE_joes_test_count_l2931_293118

/-- Given Joe's test scores, prove the number of initial tests --/
theorem joes_test_count (initial_avg : ℚ) (lowest_score : ℚ) (new_avg : ℚ) :
  initial_avg = 40 →
  lowest_score = 25 →
  new_avg = 45 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg = ((n : ℚ) - 1) * new_avg + lowest_score ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_joes_test_count_l2931_293118


namespace NUMINAMATH_CALUDE_inequality_theorem_l2931_293145

theorem inequality_theorem (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9*x*y*z ≥ 9*(x*y + y*z + z*x) ∧
  ((x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9*x*y*z = 9*(x*y + y*z + z*x) ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2931_293145


namespace NUMINAMATH_CALUDE_summer_salutations_l2931_293192

/-- The number of sun salutation yoga poses Summer performs on weekdays -/
def poses_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The total number of sun salutations Summer performs in a year -/
def total_salutations : ℕ := poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem stating that Summer performs 1300 sun salutations throughout an entire year -/
theorem summer_salutations : total_salutations = 1300 := by
  sorry

end NUMINAMATH_CALUDE_summer_salutations_l2931_293192


namespace NUMINAMATH_CALUDE_unique_integer_root_l2931_293176

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 8*x + 24

theorem unique_integer_root : 
  (∀ x : ℤ, polynomial x = 0 ↔ x = 2) := by sorry

end NUMINAMATH_CALUDE_unique_integer_root_l2931_293176


namespace NUMINAMATH_CALUDE_f_equals_cos_2x_l2931_293164

theorem f_equals_cos_2x (x : ℝ) : 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) = 
  Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_f_equals_cos_2x_l2931_293164


namespace NUMINAMATH_CALUDE_g_zero_l2931_293139

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_zero : g (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_l2931_293139


namespace NUMINAMATH_CALUDE_sam_poured_buckets_l2931_293146

/-- The number of buckets Sam initially poured into the pool -/
def initial_buckets : ℝ := 1

/-- The number of buckets Sam added later -/
def additional_buckets : ℝ := 8.8

/-- The total number of buckets Sam poured into the pool -/
def total_buckets : ℝ := initial_buckets + additional_buckets

theorem sam_poured_buckets : total_buckets = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_sam_poured_buckets_l2931_293146


namespace NUMINAMATH_CALUDE_inverse_f_at_5_l2931_293147

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem inverse_f_at_5 :
  ∃ (f_inv : ℝ → ℝ), (∀ x ≥ 0, f_inv (f x) = x) ∧ f_inv 5 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_5_l2931_293147


namespace NUMINAMATH_CALUDE_girls_in_college_l2931_293125

theorem girls_in_college (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 1040 →
  boy_ratio = 8 →
  girl_ratio = 5 →
  (boy_ratio + girl_ratio) * (total_students / (boy_ratio + girl_ratio)) = total_students →
  girl_ratio * (total_students / (boy_ratio + girl_ratio)) = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_girls_in_college_l2931_293125


namespace NUMINAMATH_CALUDE_geese_duck_difference_l2931_293112

/-- The number of more geese than ducks remaining at the duck park after a series of events --/
theorem geese_duck_difference : ℕ := by
  -- Define initial numbers
  let initial_ducks : ℕ := 25
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let initial_swans : ℕ := 3 * initial_ducks + 8

  -- Define changes in population
  let arriving_ducks : ℕ := 4
  let arriving_geese : ℕ := 7
  let leaving_swans : ℕ := 9
  let leaving_geese : ℕ := 5
  let returning_geese : ℕ := 15
  let returning_swans : ℕ := 11

  -- Calculate intermediate populations
  let ducks_after_arrival : ℕ := initial_ducks + arriving_ducks
  let geese_after_arrival : ℕ := initial_geese + arriving_geese
  let swans_after_leaving : ℕ := initial_swans - leaving_swans
  let geese_after_leaving : ℕ := geese_after_arrival - leaving_geese
  let final_geese : ℕ := geese_after_leaving + returning_geese
  let final_swans : ℕ := swans_after_leaving + returning_swans

  -- Calculate birds leaving
  let leaving_ducks : ℕ := 2 * ducks_after_arrival
  let leaving_swans : ℕ := final_swans / 2

  -- Calculate final populations
  let remaining_ducks : ℕ := ducks_after_arrival - leaving_ducks
  let remaining_geese : ℕ := final_geese

  -- Prove the difference
  have h : remaining_geese - remaining_ducks = 57 := by sorry

  exact 57

end NUMINAMATH_CALUDE_geese_duck_difference_l2931_293112


namespace NUMINAMATH_CALUDE_original_bananas_count_l2931_293129

/-- The number of bananas originally in the jar. -/
def original_bananas : ℕ := sorry

/-- The number of bananas removed from the jar. -/
def removed_bananas : ℕ := 5

/-- The number of bananas left in the jar after removal. -/
def remaining_bananas : ℕ := 41

/-- Theorem: The original number of bananas is equal to 46. -/
theorem original_bananas_count : original_bananas = 46 := by
  sorry

end NUMINAMATH_CALUDE_original_bananas_count_l2931_293129


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_zero_product_property_l2931_293109

-- Proposition 1
theorem quadratic_equation_real_roots (k : ℝ) (h : k > 0) :
  ∃ x : ℝ, x^2 + 2*x - k = 0 :=
sorry

-- Proposition 4
theorem zero_product_property (x y : ℝ) :
  x * y = 0 → x = 0 ∨ y = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_zero_product_property_l2931_293109


namespace NUMINAMATH_CALUDE_largest_root_is_two_l2931_293105

/-- The polynomial representing the difference between the curve and the line -/
def P (a b c : ℝ) (x : ℝ) : ℝ := x^6 - 6*x^5 + 17*x^4 + 6*x^3 + a*x^2 - b*x - c

/-- Theorem stating that the largest root of the polynomial is 2 -/
theorem largest_root_is_two (a b c : ℝ) : 
  (∃ p q r : ℝ, ∀ x : ℝ, P a b c x = (x - p)^2 * (x - q)^2 * (x - r)^2) → 
  (∃ x : ℝ, P a b c x = 0 ∧ ∀ y : ℝ, P a b c y = 0 → y ≤ x) →
  (∃ x : ℝ, P a b c x = 0 ∧ x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_two_l2931_293105


namespace NUMINAMATH_CALUDE_expansion_properties_l2931_293111

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x+2)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The coefficient of x^k in the expansion of (x+2)^n -/
def coeff (n k : ℕ) : ℕ := sorry

theorem expansion_properties :
  let n : ℕ := 8
  let a₀ : ℕ := coeff n 0
  let a₁ : ℕ := coeff n 1
  let a₂ : ℕ := coeff n 2
  -- a₀, a₁, a₂ form an arithmetic sequence
  (a₁ - a₀ = a₂ - a₁) →
  -- The middle (5th) term is 1120x⁴
  (coeff n 4 = 1120) ∧
  -- The sum of coefficients of odd powers is 3280
  (coeff n 1 + coeff n 3 + coeff n 5 + coeff n 7 = 3280) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l2931_293111


namespace NUMINAMATH_CALUDE_download_speed_scientific_notation_l2931_293175

/-- The network download speed of 5G in KB per second -/
def download_speed : ℕ := 1300000

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation := sorry

theorem download_speed_scientific_notation :
  to_scientific_notation download_speed = ScientificNotation.mk 1.3 6 := by sorry

end NUMINAMATH_CALUDE_download_speed_scientific_notation_l2931_293175


namespace NUMINAMATH_CALUDE_five_cube_grid_toothpicks_l2931_293141

/-- Calculates the number of toothpicks needed for a cube-shaped grid --/
def toothpicks_for_cube_grid (n : ℕ) : ℕ :=
  let vertical_toothpicks := (n + 1)^2 * n
  let horizontal_toothpicks := 2 * (n + 1) * (n + 1) * n
  vertical_toothpicks + horizontal_toothpicks

/-- Theorem stating that a 5x5x5 cube grid requires 2340 toothpicks --/
theorem five_cube_grid_toothpicks :
  toothpicks_for_cube_grid 5 = 2340 := by
  sorry


end NUMINAMATH_CALUDE_five_cube_grid_toothpicks_l2931_293141


namespace NUMINAMATH_CALUDE_complex_magnitude_l2931_293152

theorem complex_magnitude (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : 
  Complex.abs z = 5 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2931_293152


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2931_293165

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℕ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  b = c - 1575 →
  a < 1991 →
  c = 1800 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2931_293165


namespace NUMINAMATH_CALUDE_negation_of_neither_even_l2931_293173

theorem negation_of_neither_even (a b : ℤ) :
  ¬(¬(Even a) ∧ ¬(Even b)) ↔ (Even a ∨ Even b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_neither_even_l2931_293173


namespace NUMINAMATH_CALUDE_parabola_c_value_l2931_293135

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 3 = 5 →  -- vertex condition
  p.x_coord 6 = 0 →  -- point condition
  p.c = 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2931_293135


namespace NUMINAMATH_CALUDE_intersection_count_l2931_293150

/-- The set A as defined in the problem -/
def A : Set (ℤ × ℤ) := {p | ∃ m : ℤ, m > 0 ∧ p.1 = m ∧ p.2 = -3*m + 2}

/-- The set B as defined in the problem -/
def B (a : ℤ) : Set (ℤ × ℤ) := {p | ∃ n : ℤ, n > 0 ∧ p.1 = n ∧ p.2 = a*(a^2 - n + 1)}

/-- The theorem stating that there are exactly 10 integer values of a for which A ∩ B ≠ ∅ -/
theorem intersection_count :
  ∃! (s : Finset ℤ), s.card = 10 ∧ ∀ a : ℤ, a ∈ s ↔ (A ∩ B a).Nonempty :=
by sorry

end NUMINAMATH_CALUDE_intersection_count_l2931_293150


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2931_293121

theorem isosceles_triangle_area (h : ℝ) (p : ℝ) :
  h = 8 →
  p = 32 →
  ∃ (base : ℝ) (leg : ℝ),
    leg + leg + base = p ∧
    h^2 + (base/2)^2 = leg^2 ∧
    (1/2) * base * h = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2931_293121


namespace NUMINAMATH_CALUDE_max_value_of_d_l2931_293193

def a (n : ℕ) : ℕ := n^3 + 4

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_value_of_d : ∃ (k : ℕ), d k = 433 ∧ ∀ (n : ℕ), d n ≤ 433 := by sorry

end NUMINAMATH_CALUDE_max_value_of_d_l2931_293193


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2931_293153

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  b : ℕ → ℝ
  positive : ∀ n, b n > 0
  q : ℝ
  q_gt_one : q > 1
  geometric : ∀ n, b (n + 1) = q * b n

/-- The inequality holds for the 4th, 5th, 7th, and 8th terms of the geometric sequence -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.b 4 * seq.b 8 > seq.b 5 * seq.b 7 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2931_293153


namespace NUMINAMATH_CALUDE_series_convergence_l2931_293151

/-- The series ∑(n=1 to ∞) [x^(2n-1) / ((n^2 + 1) * 3^n)] converges absolutely if and only if -√3 ≤ x ≤ √3 -/
theorem series_convergence (x : ℝ) : 
  (∑' n, (x^(2*n-1) / ((n^2 + 1) * 3^n))) ≠ 0 ↔ -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_l2931_293151


namespace NUMINAMATH_CALUDE_tan_sum_with_product_l2931_293198

theorem tan_sum_with_product (x y : Real) (h1 : x + y = π / 3) 
  (h2 : Real.sqrt 3 = Real.tan (π / 3)) : 
  Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_with_product_l2931_293198


namespace NUMINAMATH_CALUDE_star_polygon_n_is_24_l2931_293113

/-- A n-pointed regular star polygon -/
structure StarPolygon (n : ℕ) where
  edges : Fin (2 * n) → ℝ
  angleA : Fin n → ℝ
  angleB : Fin n → ℝ
  edges_congruent : ∀ i j, edges i = edges j
  angleA_congruent : ∀ i j, angleA i = angleA j
  angleB_congruent : ∀ i j, angleB i = angleB j
  angle_difference : ∀ i, angleA i = angleB i - 15

/-- The theorem stating that n = 24 for the given star polygon -/
theorem star_polygon_n_is_24 (n : ℕ) (star : StarPolygon n) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_n_is_24_l2931_293113


namespace NUMINAMATH_CALUDE_petya_running_time_l2931_293156

theorem petya_running_time (a V : ℝ) (h1 : a > 0) (h2 : V > 0) :
  (a / (2.5 * V) + a / (1.6 * V)) > (a / V) := by
  sorry

end NUMINAMATH_CALUDE_petya_running_time_l2931_293156


namespace NUMINAMATH_CALUDE_log_equation_solution_l2931_293166

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 9 = 2.4 → x = (81 ^ (1/5)) ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2931_293166


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l2931_293189

-- Define the base-7 number as a list of digits
def base7Number : List Nat := [4, 5, 3, 6]

-- Define the function to convert from base 7 to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 1644 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l2931_293189


namespace NUMINAMATH_CALUDE_inequality_proof_l2931_293167

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  (a + b + c) / 4 ≥ (Real.sqrt (a * b - 1)) / (b + c) + 
                    (Real.sqrt (b * c - 1)) / (c + a) + 
                    (Real.sqrt (c * a - 1)) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2931_293167


namespace NUMINAMATH_CALUDE_product_digit_sum_equals_999_l2931_293126

/-- The number of digits in the second factor -/
def n : ℕ := 111

/-- The second factor in the product -/
def second_factor (n : ℕ) : ℕ := (10^n - 1) / 9

/-- The product of 9 and the second factor -/
def product (n : ℕ) : ℕ := 9 * second_factor n

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

theorem product_digit_sum_equals_999 :
  sum_of_digits (product n) = 999 :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_equals_999_l2931_293126


namespace NUMINAMATH_CALUDE_chess_tournament_l2931_293149

theorem chess_tournament (n : ℕ) (k : ℚ) : n > 2 →
  (8 : ℚ) + n * k = (n + 2) * (n + 1) / 2 →
  (∀ m : ℕ, m > 2 → (8 : ℚ) + m * k ≠ (m + 2) * (m + 1) / 2 → m ≠ n) →
  n = 7 ∨ n = 14 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_l2931_293149


namespace NUMINAMATH_CALUDE_simplify_expression_l2931_293184

theorem simplify_expression (c : ℝ) : ((3 * c + 6) - 6 * c) / 3 = -c + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2931_293184


namespace NUMINAMATH_CALUDE_sum_of_first_20_lucky_numbers_mod_1000_l2931_293163

def isLucky (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

def luckyNumbers : List ℕ :=
  (List.range 20).map (λ i => 7 * (10^i - 1) / 9)

theorem sum_of_first_20_lucky_numbers_mod_1000 :
  (luckyNumbers.sum) % 1000 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_20_lucky_numbers_mod_1000_l2931_293163


namespace NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l2931_293196

/-- The correlation coefficient is a measure related to the relationship between two variables. -/
def correlation_coefficient : Type := sorry

/-- The strength of the linear relationship between two variables. -/
def linear_relationship_strength : Type := sorry

/-- The correlation coefficient measures the strength of the linear relationship between two variables. -/
theorem correlation_coefficient_measures_linear_relationship :
  correlation_coefficient = linear_relationship_strength := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_measures_linear_relationship_l2931_293196


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l2931_293100

theorem magnitude_of_complex_power : 
  Complex.abs ((2 : ℂ) + (2 : ℂ) * Complex.I) ^ 8 = (4096 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l2931_293100


namespace NUMINAMATH_CALUDE_discount_profit_theorem_l2931_293138

/-- Given a discount percentage and a profit percentage without discount,
    calculate the profit percentage with the discount. -/
def profit_with_discount (discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  (1 + profit_without_discount) * (1 - discount) - 1

/-- Theorem stating that with a 4% discount and 25% profit without discount,
    the profit percentage with discount is 20%. -/
theorem discount_profit_theorem :
  profit_with_discount 0.04 0.25 = 0.20 := by
  sorry

#eval profit_with_discount 0.04 0.25

end NUMINAMATH_CALUDE_discount_profit_theorem_l2931_293138


namespace NUMINAMATH_CALUDE_valid_rental_plans_l2931_293137

/-- Represents a bus rental plan --/
structure RentalPlan where
  typeA : Nat  -- Number of Type A buses
  typeB : Nat  -- Number of Type B buses

/-- Checks if a rental plan can accommodate exactly the given number of students --/
def isValidPlan (plan : RentalPlan) (totalStudents : Nat) (typeACapacity : Nat) (typeBCapacity : Nat) : Prop :=
  plan.typeA * typeACapacity + plan.typeB * typeBCapacity = totalStudents

/-- Theorem stating that the three given rental plans are valid for 37 students --/
theorem valid_rental_plans :
  let totalStudents := 37
  let typeACapacity := 8
  let typeBCapacity := 4
  let plan1 : RentalPlan := ⟨2, 6⟩
  let plan2 : RentalPlan := ⟨3, 4⟩
  let plan3 : RentalPlan := ⟨4, 2⟩
  isValidPlan plan1 totalStudents typeACapacity typeBCapacity ∧
  isValidPlan plan2 totalStudents typeACapacity typeBCapacity ∧
  isValidPlan plan3 totalStudents typeACapacity typeBCapacity :=
by sorry


end NUMINAMATH_CALUDE_valid_rental_plans_l2931_293137


namespace NUMINAMATH_CALUDE_inequality_holds_l2931_293195

theorem inequality_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2931_293195


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l2931_293132

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena) ∧
  ∀ (A B V G D : Bool),
    (A → (B ∧ ¬V)) →
    (B → (G ∨ D)) →
    (¬V → (¬B ∧ ¬D)) →
    (¬A → (B ∧ ¬G)) →
    (A, B, V, G, D) = (false, true, true, false, true) :=
by sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l2931_293132


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2931_293170

/-- The slope of the given line y = 2x -/
def slope_given : ℚ := 2

/-- The point through which the perpendicular line passes -/
def point : ℚ × ℚ := (1, 1)

/-- The equation of the line to be proved -/
def line_equation (x y : ℚ) : Prop := x + 2 * y - 3 = 0

/-- Theorem stating that the line equation represents the perpendicular line -/
theorem perpendicular_line_equation :
  (∀ x y, line_equation x y ↔ y - point.2 = (-1 / slope_given) * (x - point.1)) ∧
  line_equation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2931_293170


namespace NUMINAMATH_CALUDE_gcd_of_256_180_720_l2931_293178

theorem gcd_of_256_180_720 : Nat.gcd 256 (Nat.gcd 180 720) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_720_l2931_293178


namespace NUMINAMATH_CALUDE_salary_problem_l2931_293106

/-- Salary problem -/
theorem salary_problem 
  (jan feb mar apr may : ℕ)  -- Salaries for each month
  (h1 : (jan + feb + mar + apr) / 4 = 8000)  -- Average for Jan-Apr
  (h2 : (feb + mar + apr + may) / 4 = 8800)  -- Average for Feb-May
  (h3 : may = 6500)  -- May's salary
  : jan = 3300 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l2931_293106


namespace NUMINAMATH_CALUDE_grants_total_earnings_l2931_293172

def first_month : ℕ := 350
def second_month : ℕ := 2 * first_month + 50
def third_month : ℕ := 4 * (first_month + second_month)

theorem grants_total_earnings :
  first_month + second_month + third_month = 5500 := by
  sorry

end NUMINAMATH_CALUDE_grants_total_earnings_l2931_293172


namespace NUMINAMATH_CALUDE_manoj_transaction_gain_l2931_293182

/-- Calculate simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℚ := 3900

/-- The interest rate Manoj pays to Anwar -/
def borrowing_rate : ℚ := 6

/-- The amount Manoj lent to Ramu -/
def lent_amount : ℚ := 5655

/-- The interest rate Manoj charges Ramu -/
def lending_rate : ℚ := 9

/-- The time period for both transactions in years -/
def time_period : ℚ := 3

/-- Manoj's gain from the transaction -/
def manoj_gain : ℚ :=
  simple_interest lent_amount lending_rate time_period -
  simple_interest borrowed_amount borrowing_rate time_period

theorem manoj_transaction_gain :
  manoj_gain = 824.85 := by sorry

end NUMINAMATH_CALUDE_manoj_transaction_gain_l2931_293182


namespace NUMINAMATH_CALUDE_unsupported_attendees_l2931_293161

-- Define the total number of attendees
def total_attendance : ℕ := 500

-- Define the percentage of supporters for each team
def team_a_percentage : ℚ := 35 / 100
def team_b_percentage : ℚ := 25 / 100
def team_c_percentage : ℚ := 20 / 100
def team_d_percentage : ℚ := 15 / 100

-- Define the overlap percentages
def team_ab_overlap_percentage : ℚ := 10 / 100
def team_bc_overlap_percentage : ℚ := 5 / 100
def team_cd_overlap_percentage : ℚ := 7 / 100

-- Define the number of people attending for atmosphere
def atmosphere_attendees : ℕ := 30

-- Theorem to prove
theorem unsupported_attendees :
  ∃ (unsupported : ℕ),
    unsupported = total_attendance -
      (((team_a_percentage + team_b_percentage + team_c_percentage + team_d_percentage) * total_attendance).floor -
       ((team_ab_overlap_percentage * team_a_percentage * total_attendance +
         team_bc_overlap_percentage * team_b_percentage * total_attendance +
         team_cd_overlap_percentage * team_c_percentage * total_attendance).floor) +
       atmosphere_attendees) ∧
    unsupported = 26 := by
  sorry

end NUMINAMATH_CALUDE_unsupported_attendees_l2931_293161


namespace NUMINAMATH_CALUDE_tangent_segments_area_l2931_293115

/-- The area of the region formed by all line segments of length 6 that are tangent to a circle with radius 4 at their midpoints -/
theorem tangent_segments_area (r : ℝ) (l : ℝ) (h_r : r = 4) (h_l : l = 6) :
  let outer_radius := Real.sqrt (r^2 + (l/2)^2)
  (π * outer_radius^2 - π * r^2) = 9 * π :=
sorry

end NUMINAMATH_CALUDE_tangent_segments_area_l2931_293115


namespace NUMINAMATH_CALUDE_cuboid_volume_example_l2931_293199

/-- A cuboid with given base area and height -/
structure Cuboid where
  base_area : ℝ
  height : ℝ

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.base_area * c.height

/-- Theorem: The volume of a cuboid with base area 14 cm² and height 13 cm is 182 cm³ -/
theorem cuboid_volume_example : 
  let c : Cuboid := { base_area := 14, height := 13 }
  volume c = 182 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_example_l2931_293199


namespace NUMINAMATH_CALUDE_square_minus_a_nonpositive_iff_a_geq_four_l2931_293124

theorem square_minus_a_nonpositive_iff_a_geq_four :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_square_minus_a_nonpositive_iff_a_geq_four_l2931_293124


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_square_extensions_l2931_293160

/-- The radius of a circle tangent to the extensions of two sides of a square,
    where two tangents from the opposite corner form a specific angle. -/
theorem circle_radius_tangent_to_square_extensions 
  (side_length : ℝ) 
  (tangent_angle : ℝ) 
  (sin_half_angle : ℝ) :
  side_length = 6 + 2 * Real.sqrt 5 →
  tangent_angle = 36 →
  sin_half_angle = (Real.sqrt 5 - 1) / 4 →
  ∃ (radius : ℝ), 
    radius = 2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) ∧
    radius = side_length * Real.sqrt 2 / 
      ((4 / (Real.sqrt 5 - 1)) - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_square_extensions_l2931_293160


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2931_293181

theorem quadratic_function_range (a : ℝ) : 
  (∃ y₁ y₂ y₃ y₄ : ℝ, 
    (y₁ = a * (-4)^2 + 4 * a * (-4) - 6) ∧
    (y₂ = a * (-3)^2 + 4 * a * (-3) - 6) ∧
    (y₃ = a * 0^2 + 4 * a * 0 - 6) ∧
    (y₄ = a * 2^2 + 4 * a * 2 - 6) ∧
    ((y₁ > 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ > 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ > 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ > 0))) →
  (a < -2 ∨ a > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2931_293181


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l2931_293122

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Main theorem: If a line is perpendicular to two different planes, then the planes are parallel -/
theorem line_perp_two_planes_implies_parallel (l : Line3D) (α β : Plane3D) :
  α ≠ β → perpendicular l α → perpendicular l β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l2931_293122


namespace NUMINAMATH_CALUDE_cube_sum_geq_mixed_terms_l2931_293158

theorem cube_sum_geq_mixed_terms (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 ≥ x^2*y + x*y^2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_geq_mixed_terms_l2931_293158


namespace NUMINAMATH_CALUDE_array_sum_theorem_l2931_293104

-- Define the array structure
def array_sum (p : ℕ) : ℚ := 3 * p^2 / ((3*p - 1) * (p - 1))

-- Define the result of (m+n) mod 2009
def result_mod_2009 (p : ℕ) : ℕ :=
  let m : ℕ := 3 * p^2
  let n : ℕ := (3*p - 1) * (p - 1)
  (m + n) % 2009

-- The main theorem
theorem array_sum_theorem :
  array_sum 2008 = 3 * 2008^2 / ((3*2008 - 1) * (2008 - 1)) ∧
  result_mod_2009 2008 = 1 := by sorry

end NUMINAMATH_CALUDE_array_sum_theorem_l2931_293104


namespace NUMINAMATH_CALUDE_monomial_2015_coeff_l2931_293133

/-- The coefficient of the nth monomial in the sequence -/
def monomial_coeff (n : ℕ) : ℤ := (-1)^n * (2*n - 1)

/-- The theorem stating that the 2015th monomial coefficient is -4029 -/
theorem monomial_2015_coeff : monomial_coeff 2015 = -4029 := by
  sorry

end NUMINAMATH_CALUDE_monomial_2015_coeff_l2931_293133


namespace NUMINAMATH_CALUDE_february_messages_l2931_293119

def text_messages (month : ℕ) : ℕ :=
  2^month

theorem february_messages :
  text_messages 3 = 8 ∧ text_messages 4 = 16 :=
by sorry

end NUMINAMATH_CALUDE_february_messages_l2931_293119


namespace NUMINAMATH_CALUDE_triangle_side_length_l2931_293179

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.α > 0 ∧ t.β > 0 ∧ t.γ > 0 ∧
  t.α + t.β + t.γ = Real.pi ∧
  3 * t.α + 2 * t.β = Real.pi

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h : TriangleProperties t) 
  (ha : t.a = 2) 
  (hb : t.b = 3) : 
  t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2931_293179


namespace NUMINAMATH_CALUDE_sector_to_cone_area_ratio_l2931_293143

/-- Given a sector with central angle 135° and area S₁, when formed into a cone
    with total surface area S₂, prove that S₁/S₂ = 8/11 -/
theorem sector_to_cone_area_ratio :
  ∀ (S₁ S₂ : ℝ),
  S₁ > 0 → S₂ > 0 →
  (∃ (r : ℝ), r > 0 ∧
    S₁ = (135 / 360) * π * r^2 ∧
    S₂ = S₁ + π * ((3/8) * r)^2) →
  S₁ / S₂ = 8 / 11 := by
sorry


end NUMINAMATH_CALUDE_sector_to_cone_area_ratio_l2931_293143


namespace NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l2931_293197

/-- A parabola with vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The locus of midpoints of right-angled chords of a parabola -/
def midpoint_locus (P : Parabola) : Parabola :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem parabola_midpoint_locus_ratio (P : Parabola) :
  let Q := midpoint_locus P
  (distance P.focus Q.focus) / (distance P.vertex Q.vertex) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l2931_293197


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2931_293144

theorem complex_modulus_problem (m n : ℝ) : 
  (m / (1 + Complex.I)) = (1 - n * Complex.I) → 
  Complex.abs (m + n * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2931_293144


namespace NUMINAMATH_CALUDE_fraction_equality_l2931_293102

theorem fraction_equality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2931_293102


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l2931_293128

theorem max_rabbits_with_traits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both_traits : ℕ) :
  long_ears = 13 →
  jump_far = 17 →
  both_traits ≥ 3 →
  N ≤ 27 :=
by
  sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l2931_293128


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2931_293140

/-- Convert a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Convert a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

/-- The binary representation of 110111001₂ -/
def binary_number : List Bool := [true, true, false, true, true, true, false, false, true]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_number) = [1, 3, 2, 2, 1] := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2931_293140


namespace NUMINAMATH_CALUDE_bananas_bought_l2931_293185

/-- The number of times Ronald went to the store last month -/
def store_visits : ℕ := 2

/-- The number of bananas Ronald buys each time he goes to the store -/
def bananas_per_visit : ℕ := 10

/-- The total number of bananas Ronald bought last month -/
def total_bananas : ℕ := store_visits * bananas_per_visit

theorem bananas_bought : total_bananas = 20 := by
  sorry

end NUMINAMATH_CALUDE_bananas_bought_l2931_293185


namespace NUMINAMATH_CALUDE_angies_taxes_paid_l2931_293180

/-- Represents the weekly expenses for necessities, taxes, and utilities -/
structure WeeklyExpenses where
  necessities : ℕ
  taxes : ℕ
  utilities : ℕ

/-- Represents Angie's monthly finances -/
structure MonthlyFinances where
  salary : ℕ
  week1 : WeeklyExpenses
  week2 : WeeklyExpenses
  week3 : WeeklyExpenses
  week4 : WeeklyExpenses
  leftover : ℕ

/-- Calculates the total taxes paid in a month -/
def totalTaxesPaid (finances : MonthlyFinances) : ℕ :=
  finances.week1.taxes + finances.week2.taxes + finances.week3.taxes + finances.week4.taxes

/-- Theorem stating that Angie's total taxes paid for the month is $30 -/
theorem angies_taxes_paid (finances : MonthlyFinances) 
    (h1 : finances.salary = 80)
    (h2 : finances.week1 = ⟨12, 8, 5⟩)
    (h3 : finances.week2 = ⟨15, 6, 7⟩)
    (h4 : finances.week3 = ⟨10, 9, 6⟩)
    (h5 : finances.week4 = ⟨14, 7, 4⟩)
    (h6 : finances.leftover = 18) :
    totalTaxesPaid finances = 30 := by
  sorry

#eval totalTaxesPaid ⟨80, ⟨12, 8, 5⟩, ⟨15, 6, 7⟩, ⟨10, 9, 6⟩, ⟨14, 7, 4⟩, 18⟩

end NUMINAMATH_CALUDE_angies_taxes_paid_l2931_293180


namespace NUMINAMATH_CALUDE_inequality_proof_l2931_293188

theorem inequality_proof (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 1/2) :
  (k + a/(b+c)) * (k + b/(c+a)) * (k + c/(a+b)) ≥ (k + 1/2)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2931_293188


namespace NUMINAMATH_CALUDE_circle_radius_l2931_293134

/-- A circle with equation x^2 + y^2 - 2x + my - 4 = 0 that is symmetric about the line 2x + y = 0 has a radius of 3 -/
theorem circle_radius (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + m*y - 4 = 0 → (∃ x' y' : ℝ, x'^2 + y'^2 - 2*x' + m*y' - 4 = 0 ∧ 
    2*x + y = 0 ∧ 2*x' + y' = 0 ∧ x + x' = 2*x ∧ y + y' = 2*y)) → 
  (∃ c_x c_y : ℝ, ∀ x y : ℝ, (x - c_x)^2 + (y - c_y)^2 = 3^2 ↔ x^2 + y^2 - 2*x + m*y - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2931_293134


namespace NUMINAMATH_CALUDE_function_property_l2931_293168

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  is_even_function f :=
sorry

end NUMINAMATH_CALUDE_function_property_l2931_293168


namespace NUMINAMATH_CALUDE_accidental_division_correction_l2931_293155

theorem accidental_division_correction (x : ℝ) : 
  x / 15 = 6 → x * 15 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_accidental_division_correction_l2931_293155


namespace NUMINAMATH_CALUDE_sum_parity_when_sum_of_squares_even_l2931_293159

theorem sum_parity_when_sum_of_squares_even (m n : ℤ) : 
  Even (m^2 + n^2) → Even (m + n) :=
by sorry

end NUMINAMATH_CALUDE_sum_parity_when_sum_of_squares_even_l2931_293159


namespace NUMINAMATH_CALUDE_banana_cost_is_three_l2931_293142

/-- The cost of a bunch of bananas -/
def banana_cost : ℝ := 3

/-- The cost of a dozen apples -/
def apple_dozen_cost : ℝ := 2

/-- Arnold's purchase: 1 dozen apples and 1 bunch of bananas -/
def arnold_purchase : ℝ := apple_dozen_cost + banana_cost

/-- Tony's purchase: 2 dozen apples and 1 bunch of bananas -/
def tony_purchase : ℝ := 2 * apple_dozen_cost + banana_cost

theorem banana_cost_is_three :
  arnold_purchase = 5 ∧ tony_purchase = 7 → banana_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_is_three_l2931_293142


namespace NUMINAMATH_CALUDE_bianca_candy_eaten_l2931_293174

theorem bianca_candy_eaten (total : ℕ) (piles : ℕ) (per_pile : ℕ) 
  (h1 : total = 78)
  (h2 : piles = 6)
  (h3 : per_pile = 8) :
  total - (piles * per_pile) = 30 := by
  sorry

end NUMINAMATH_CALUDE_bianca_candy_eaten_l2931_293174


namespace NUMINAMATH_CALUDE_day_15_net_income_l2931_293131

/-- Calculate the net income on a given day of business -/
def net_income (initial_income : ℝ) (daily_multiplier : ℝ) (daily_expenses : ℝ) (tax_rate : ℝ) (day : ℕ) : ℝ :=
  let gross_income := initial_income * daily_multiplier^(day - 1)
  let tax := tax_rate * gross_income
  let after_tax := gross_income - tax
  after_tax - daily_expenses

/-- The net income on the 15th day of business is $12,913,916.3 -/
theorem day_15_net_income :
  net_income 3 3 100 0.1 15 = 12913916.3 := by
  sorry

end NUMINAMATH_CALUDE_day_15_net_income_l2931_293131


namespace NUMINAMATH_CALUDE_jeans_final_price_is_correct_l2931_293194

def socks_price : ℝ := 5
def tshirt_price : ℝ := socks_price + 10
def jeans_price : ℝ := 2 * tshirt_price
def jeans_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08

def jeans_final_price : ℝ :=
  let discounted_price := jeans_price * (1 - jeans_discount_rate)
  discounted_price * (1 + sales_tax_rate)

theorem jeans_final_price_is_correct :
  jeans_final_price = 27.54 := by sorry

end NUMINAMATH_CALUDE_jeans_final_price_is_correct_l2931_293194


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2931_293186

/-- 
For a quadratic equation x^2 + 8x + q = 0 to have two distinct real roots,
q must be less than 16.
-/
theorem quadratic_roots_condition (q : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 8*x + q = 0 ∧ y^2 + 8*y + q = 0) ↔ q < 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2931_293186
