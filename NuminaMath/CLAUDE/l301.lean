import Mathlib

namespace NUMINAMATH_CALUDE_angle_bisector_ratio_not_unique_l301_30188

/-- Represents a triangle --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

/-- Represents the ratio of an angle bisector to its corresponding side --/
def angle_bisector_ratio (t : Triangle) : ℝ := 
  sorry -- Definition of angle bisector ratio

/-- Two triangles are similar if their corresponding angles are equal --/
def similar (t1 t2 : Triangle) : Prop :=
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

theorem angle_bisector_ratio_not_unique :
  ∃ (t1 t2 : Triangle) (r : ℝ), 
    angle_bisector_ratio t1 = r ∧ 
    angle_bisector_ratio t2 = r ∧ 
    ¬(similar t1 t2) :=
  sorry


end NUMINAMATH_CALUDE_angle_bisector_ratio_not_unique_l301_30188


namespace NUMINAMATH_CALUDE_factorization_identity_l301_30150

theorem factorization_identity (x y : ℝ) : (x - y)^2 + 2*y*(x - y) = (x - y)*(x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l301_30150


namespace NUMINAMATH_CALUDE_fraction_sqrt_cube_root_equals_power_l301_30162

theorem fraction_sqrt_cube_root_equals_power (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b)^(1/2) / (a * b)^(1/3) = a^(7/6) * b^(1/6) := by sorry

end NUMINAMATH_CALUDE_fraction_sqrt_cube_root_equals_power_l301_30162


namespace NUMINAMATH_CALUDE_max_value_constraint_l301_30173

theorem max_value_constraint (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 173 ∧ 
  (∀ a b c : ℝ, 9*a^2 + 4*b^2 + 25*c^2 = 1 → 8*a + 3*b + 10*c ≤ max) ∧
  (8*x + 3*y + 10*z = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l301_30173


namespace NUMINAMATH_CALUDE_paint_for_similar_statues_l301_30184

theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_statues : ℝ)
  (h1 : original_height = 6)
  (h2 : original_paint = 1)
  (h3 : new_height = 1)
  (h4 : num_statues = 540)
  : (num_statues * new_height^2 * original_paint) / original_height^2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_similar_statues_l301_30184


namespace NUMINAMATH_CALUDE_walkway_area_is_416_l301_30164

/-- Represents the garden layout and calculates the walkway area -/
def garden_walkway_area (rows : Nat) (cols : Nat) (bed_length : Nat) (bed_width : Nat) (walkway_width : Nat) : Nat :=
  let total_width := cols * bed_length + (cols + 1) * walkway_width
  let total_length := rows * bed_width + (rows + 1) * walkway_width
  let total_area := total_width * total_length
  let bed_area := rows * cols * bed_length * bed_width
  total_area - bed_area

/-- Theorem stating that the walkway area for the given garden configuration is 416 square feet -/
theorem walkway_area_is_416 :
  garden_walkway_area 4 3 8 3 2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_416_l301_30164


namespace NUMINAMATH_CALUDE_triangle_similarity_equality_equivalence_l301_30195

/-- Two triangles are similar if their corresponding sides are proportional -/
def SimilarTriangles (a b c a₁ b₁ c₁ : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁

/-- The theorem stating the equivalence between triangle similarity and the given equation -/
theorem triangle_similarity_equality_equivalence
  (a b c a₁ b₁ c₁ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  SimilarTriangles a b c a₁ b₁ c₁ ↔
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) =
  Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) := by
    sorry

#check triangle_similarity_equality_equivalence

end NUMINAMATH_CALUDE_triangle_similarity_equality_equivalence_l301_30195


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l301_30174

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2

theorem f_monotone_increasing :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l301_30174


namespace NUMINAMATH_CALUDE_one_positive_integer_solution_l301_30102

theorem one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ (25 : ℝ) - 5 * n > 15 :=
by sorry

end NUMINAMATH_CALUDE_one_positive_integer_solution_l301_30102


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l301_30157

theorem arithmetic_sequence_length (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) (n : ℕ) 
  (h₁ : a₁ = 3.25)
  (h₂ : aₙ = 55.25)
  (h₃ : d = 4)
  (h₄ : aₙ = a₁ + (n - 1) * d) :
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l301_30157


namespace NUMINAMATH_CALUDE_sqrt_a_plus_one_real_l301_30199

theorem sqrt_a_plus_one_real (a : ℝ) : (∃ (x : ℝ), x ^ 2 = a + 1) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_one_real_l301_30199


namespace NUMINAMATH_CALUDE_truck_length_l301_30109

/-- The length of a truck given its speed and tunnel transit time -/
theorem truck_length (tunnel_length : ℝ) (transit_time : ℝ) (speed_mph : ℝ) :
  tunnel_length = 330 →
  transit_time = 6 →
  speed_mph = 45 →
  (speed_mph * 5280 / 3600) * transit_time - tunnel_length = 66 :=
by sorry

end NUMINAMATH_CALUDE_truck_length_l301_30109


namespace NUMINAMATH_CALUDE_equation_solutions_l301_30175

theorem equation_solutions : 
  ∀ x : ℝ, x^4 + (3 - x)^4 = 146 ↔ 
  x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l301_30175


namespace NUMINAMATH_CALUDE_sophie_total_spend_l301_30114

-- Define the quantities and prices
def cupcakes : ℕ := 5
def cupcake_price : ℚ := 2

def doughnuts : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_slices : ℕ := 4
def apple_pie_price : ℚ := 2

def cookies : ℕ := 15
def cookie_price : ℚ := 0.6

-- Define the total cost function
def total_cost : ℚ :=
  cupcakes * cupcake_price +
  doughnuts * doughnut_price +
  apple_pie_slices * apple_pie_price +
  cookies * cookie_price

-- Theorem statement
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end NUMINAMATH_CALUDE_sophie_total_spend_l301_30114


namespace NUMINAMATH_CALUDE_bottle_capacity_ratio_l301_30160

theorem bottle_capacity_ratio (c1 c2 : ℝ) : 
  c1 > 0 ∧ c2 > 0 →  -- Capacities are positive
  c1 / 2 + c2 / 4 = (c1 + c2) / 3 →  -- Oil is 1/3 of total mixture
  c2 / c1 = 2 := by sorry

end NUMINAMATH_CALUDE_bottle_capacity_ratio_l301_30160


namespace NUMINAMATH_CALUDE_unique_arrangement_l301_30107

-- Define the types for containers and liquids
inductive Container : Type
  | Bottle
  | Glass
  | Jug
  | Jar

inductive Liquid : Type
  | Milk
  | Lemonade
  | Kvass
  | Water

-- Define the arrangement as a function from Container to Liquid
def Arrangement := Container → Liquid

-- Define the conditions
def water_milk_not_in_bottle (arr : Arrangement) : Prop :=
  arr Container.Bottle ≠ Liquid.Water ∧ arr Container.Bottle ≠ Liquid.Milk

def lemonade_between_jug_and_kvass (arr : Arrangement) : Prop :=
  (arr Container.Bottle = Liquid.Lemonade ∧ arr Container.Jug = Liquid.Milk ∧ arr Container.Jar = Liquid.Kvass) ∨
  (arr Container.Glass = Liquid.Lemonade ∧ arr Container.Bottle = Liquid.Kvass ∧ arr Container.Jar = Liquid.Milk) ∨
  (arr Container.Bottle = Liquid.Milk ∧ arr Container.Glass = Liquid.Lemonade ∧ arr Container.Jug = Liquid.Kvass)

def jar_not_lemonade_or_water (arr : Arrangement) : Prop :=
  arr Container.Jar ≠ Liquid.Lemonade ∧ arr Container.Jar ≠ Liquid.Water

def glass_next_to_jar_and_milk (arr : Arrangement) : Prop :=
  (arr Container.Glass = Liquid.Water ∧ arr Container.Jug = Liquid.Milk) ∨
  (arr Container.Glass = Liquid.Kvass ∧ arr Container.Bottle = Liquid.Milk)

-- Define the correct arrangement
def correct_arrangement : Arrangement :=
  fun c => match c with
  | Container.Bottle => Liquid.Lemonade
  | Container.Glass => Liquid.Water
  | Container.Jug => Liquid.Milk
  | Container.Jar => Liquid.Kvass

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement),
    water_milk_not_in_bottle arr ∧
    lemonade_between_jug_and_kvass arr ∧
    jar_not_lemonade_or_water arr ∧
    glass_next_to_jar_and_milk arr →
    arr = correct_arrangement :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l301_30107


namespace NUMINAMATH_CALUDE_product_mod_seven_l301_30152

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l301_30152


namespace NUMINAMATH_CALUDE_no_number_with_2011_quotient_and_remainder_l301_30118

-- Function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_number_with_2011_quotient_and_remainder :
  ¬ ∃ (n : ℕ), 
    let s := sumOfDigits n
    n / s = 2011 ∧ n % s = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_number_with_2011_quotient_and_remainder_l301_30118


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l301_30128

theorem geometric_progression_ratio (x y z r : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → x ≠ y → y ≠ z → x ≠ z →
  (∃ a : ℝ, a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2) →
  x * (y - z) * y * (z - x) * z * (x - y) = (y * (z - x))^2 →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l301_30128


namespace NUMINAMATH_CALUDE_total_squares_5x6_l301_30125

/-- The number of squares of a given size in a grid --/
def count_squares (rows : ℕ) (cols : ℕ) (size : ℕ) : ℕ :=
  (rows - size) * (cols - size)

/-- The total number of squares in a 5x6 grid --/
def total_squares : ℕ :=
  count_squares 5 6 1 + count_squares 5 6 2 + count_squares 5 6 3 + count_squares 5 6 4

/-- Theorem: The total number of squares in a 5x6 grid is 40 --/
theorem total_squares_5x6 : total_squares = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_5x6_l301_30125


namespace NUMINAMATH_CALUDE_green_eyed_students_l301_30155

theorem green_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) :
  total = 50 →
  both = 10 →
  neither = 5 →
  ∃ (green : ℕ),
    green * 2 = (total - both - neither) - green ∧
    green = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_eyed_students_l301_30155


namespace NUMINAMATH_CALUDE_mode_of_dataset_l301_30105

def dataset : List ℕ := [2, 2, 2, 3, 3, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_dataset : mode dataset = 2 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_dataset_l301_30105


namespace NUMINAMATH_CALUDE_egg_distribution_l301_30185

theorem egg_distribution (num_boxes : ℝ) (eggs_per_box : ℝ) (h1 : num_boxes = 2.0) (h2 : eggs_per_box = 1.5) :
  num_boxes * eggs_per_box = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l301_30185


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l301_30187

theorem cookie_boxes_problem (n : ℕ) : 
  (n ≥ 1) →
  (n - 7 ≥ 1) →
  (n - 2 ≥ 1) →
  ((n - 7) + (n - 2) < n) →
  (n = 8) := by
  sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l301_30187


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l301_30151

theorem absolute_value_equation_solution_count : 
  ∃! x : ℝ, |x - 5| = |x + 3| := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l301_30151


namespace NUMINAMATH_CALUDE_average_donation_l301_30197

theorem average_donation (total_people : ℝ) (h_total_positive : total_people > 0) : 
  let group1_fraction : ℝ := 1 / 10
  let group2_fraction : ℝ := 3 / 4
  let group3_fraction : ℝ := 1 - group1_fraction - group2_fraction
  let donation1 : ℝ := 200
  let donation2 : ℝ := 100
  let donation3 : ℝ := 50
  let total_donation : ℝ := 
    group1_fraction * donation1 * total_people + 
    group2_fraction * donation2 * total_people + 
    group3_fraction * donation3 * total_people
  total_donation / total_people = 102.5 := by
sorry

end NUMINAMATH_CALUDE_average_donation_l301_30197


namespace NUMINAMATH_CALUDE_triangle_base_height_difference_l301_30178

theorem triangle_base_height_difference (base height : ℚ) : 
  base = 5/6 → height = 4/6 → base - height = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_height_difference_l301_30178


namespace NUMINAMATH_CALUDE_circle_properties_l301_30176

/-- Given a circle with equation x^2 + y^2 = 10x - 8y + 4, prove its properties --/
theorem circle_properties :
  let equation := fun (x y : ℝ) => x^2 + y^2 = 10*x - 8*y + 4
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 = 5 ∧ center.2 = -4) ∧  -- Center is (5, -4)
    (radius = 3 * Real.sqrt 5) ∧     -- Radius is 3√5
    (center.1 + center.2 = 1) ∧      -- Sum of center coordinates is 1
    ∀ (x y : ℝ), equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l301_30176


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l301_30149

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (4 - a) * x + 7 else a^x

-- Define what it means for f to be increasing on ℝ
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (is_increasing (f a)) ↔ (3 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l301_30149


namespace NUMINAMATH_CALUDE_original_price_calculation_l301_30141

theorem original_price_calculation (price paid : ℝ) (h1 : paid = 18) (h2 : paid = (1/4) * price) : price = 72 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l301_30141


namespace NUMINAMATH_CALUDE_compound_interest_rate_l301_30146

/-- Proves that given the compound interest conditions, the rate of interest is 5% -/
theorem compound_interest_rate (P R : ℝ) 
  (h1 : P * (1 + R / 100) ^ 2 = 17640)
  (h2 : P * (1 + R / 100) ^ 3 = 18522) : 
  R = 5 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l301_30146


namespace NUMINAMATH_CALUDE_sum_mod_twelve_l301_30193

theorem sum_mod_twelve : (2101 + 2103 + 2105 + 2107 + 2109) % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_twelve_l301_30193


namespace NUMINAMATH_CALUDE_expand_and_simplify_l301_30161

theorem expand_and_simplify (x : ℝ) :
  5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l301_30161


namespace NUMINAMATH_CALUDE_fraction_sum_to_decimal_l301_30127

theorem fraction_sum_to_decimal : (9 : ℚ) / 10 + (8 : ℚ) / 100 = (98 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_to_decimal_l301_30127


namespace NUMINAMATH_CALUDE_a_range_l301_30147

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- State the theorem
theorem a_range (a : ℝ) : A ∪ B a = B a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l301_30147


namespace NUMINAMATH_CALUDE_intersection_locus_l301_30123

/-- The locus of intersection points of two lines passing through fixed points on the x-axis and intersecting a parabola at four concyclic points. -/
theorem intersection_locus (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∀ (l m : ℝ → ℝ → Prop) (P : ℝ × ℝ),
    (∀ y, l a y ↔ y = 0) →  -- Line l passes through (a, 0)
    (∀ y, m b y ↔ y = 0) →  -- Line m passes through (b, 0)
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,  -- Four distinct intersection points
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧ m x₃ y₃ ∧ m x₄ y₄ ∧
      y₁^2 = x₁ ∧ y₂^2 = x₂ ∧ y₃^2 = x₃ ∧ y₄^2 = x₄ ∧
      ∃ (r : ℝ) (c : ℝ × ℝ), -- Points are concyclic
        (x₁ - c.1)^2 + (y₁ - c.2)^2 = r^2 ∧
        (x₂ - c.1)^2 + (y₂ - c.2)^2 = r^2 ∧
        (x₃ - c.1)^2 + (y₃ - c.2)^2 = r^2 ∧
        (x₄ - c.1)^2 + (y₄ - c.2)^2 = r^2) →
    (∀ x y, l x y ∧ m x y → P = (x, y)) →  -- P is the intersection of l and m
    P.1 = (a + b) / 2  -- The x-coordinate of P satisfies 2x - (a + b) = 0
  := by sorry

end NUMINAMATH_CALUDE_intersection_locus_l301_30123


namespace NUMINAMATH_CALUDE_min_disks_is_fifteen_l301_30116

/-- Represents the storage problem with given file sizes and disk capacity. -/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  files_09mb : ℕ
  files_08mb : ℕ
  files_05mb : ℕ
  h_total : total_files = files_09mb + files_08mb + files_05mb

/-- Calculates the minimum number of disks required for the given storage problem. -/
def min_disks_required (p : StorageProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of disks required for the given problem is 15. -/
theorem min_disks_is_fifteen :
  let p : StorageProblem := {
    total_files := 35,
    disk_capacity := 8/5,
    files_09mb := 5,
    files_08mb := 10,
    files_05mb := 20,
    h_total := by rfl
  }
  min_disks_required p = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_is_fifteen_l301_30116


namespace NUMINAMATH_CALUDE_taxi_fare_problem_l301_30137

/-- Represents the fare for a taxi ride. -/
structure TaxiFare where
  distance : ℝ  -- Distance traveled in kilometers
  cost : ℝ      -- Cost in dollars
  h_positive : distance > 0

/-- States that taxi fares are directly proportional to the distance traveled. -/
def DirectlyProportional (f₁ f₂ : TaxiFare) : Prop :=
  f₁.cost / f₁.distance = f₂.cost / f₂.distance

theorem taxi_fare_problem (f₁ : TaxiFare) 
    (h₁ : f₁.distance = 80 ∧ f₁.cost = 200) :
    ∃ (f₂ : TaxiFare), 
      f₂.distance = 120 ∧ 
      DirectlyProportional f₁ f₂ ∧ 
      f₂.cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_problem_l301_30137


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l301_30131

/-- Represents a factorization of x^2 + bx + 2016 into (x + r)(x + s) -/
structure Factorization where
  r : ℤ
  s : ℤ
  sum_eq : r + s = b
  product_eq : r * s = 2016

/-- Returns true if the quadratic x^2 + bx + 2016 can be factored with integer coefficients -/
def has_integer_factorization (b : ℤ) : Prop :=
  ∃ f : Factorization, f.r + f.s = b ∧ f.r * f.s = 2016

theorem smallest_factorizable_b :
  (has_integer_factorization 90) ∧
  (∀ b : ℤ, 0 < b → b < 90 → ¬(has_integer_factorization b)) :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l301_30131


namespace NUMINAMATH_CALUDE_grass_field_width_l301_30103

/-- Given a rectangular grass field with length 85 m, surrounded by a 2.5 m wide path 
    with an area of 1450 sq m, the width of the grass field is 200 m. -/
theorem grass_field_width (field_length : ℝ) (path_width : ℝ) (path_area : ℝ) :
  field_length = 85 →
  path_width = 2.5 →
  path_area = 1450 →
  ∃ field_width : ℝ,
    (field_length + 2 * path_width) * (field_width + 2 * path_width) -
    field_length * field_width = path_area ∧
    field_width = 200 :=
by sorry

end NUMINAMATH_CALUDE_grass_field_width_l301_30103


namespace NUMINAMATH_CALUDE_g_equals_zero_l301_30130

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_equals_zero : g (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_zero_l301_30130


namespace NUMINAMATH_CALUDE_four_distinct_positive_roots_l301_30133

/-- The polynomial f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - x^3 + 8*a*x^2 - a*x + a^2

/-- Theorem stating the condition for f(x) to have four distinct positive roots -/
theorem four_distinct_positive_roots (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) ↔
  (1/25 < a ∧ a < 1/24) :=
sorry

end NUMINAMATH_CALUDE_four_distinct_positive_roots_l301_30133


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l301_30154

theorem boys_to_girls_ratio (S G : ℚ) (h : S > 0) (h1 : G > 0) (h2 : (1/2) * G = (1/3) * S) :
  (S - G) / G = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l301_30154


namespace NUMINAMATH_CALUDE_floor_counterexamples_l301_30190

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Statement of the theorem
theorem floor_counterexamples : ∃ (x y : ℝ),
  (floor (2^x) ≠ floor (2^(floor x))) ∧
  (floor (y^2) ≠ (floor y)^2) := by
  sorry

end NUMINAMATH_CALUDE_floor_counterexamples_l301_30190


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l301_30192

theorem binomial_expansion_example : 
  57^4 + 4*(57^3 * 2) + 6*(57^2 * 2^2) + 4*(57 * 2^3) + 2^4 = 12117361 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l301_30192


namespace NUMINAMATH_CALUDE_sum_and_round_l301_30158

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  100 * ((x + 50) / 100)

theorem sum_and_round : round_to_nearest_hundred (128 + 264) = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_round_l301_30158


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l301_30139

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → a * x^2 - 3 * x - a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l301_30139


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l301_30196

/-- The area of the triangle formed by points (0, 0), (4, 2), and (4, -4) is 4√5 square units -/
theorem triangle_area : ℝ :=
let A : ℝ × ℝ := (0, 0)
let B : ℝ × ℝ := (4, 2)
let C : ℝ × ℝ := (4, -4)
let triangle_area := Real.sqrt 5 * 4
triangle_area

/-- Proof that the area of the triangle formed by points (0, 0), (4, 2), and (4, -4) is 4√5 square units -/
theorem triangle_area_proof : triangle_area = Real.sqrt 5 * 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l301_30196


namespace NUMINAMATH_CALUDE_circle_max_min_distances_l301_30135

/-- Circle C with center (3,4) and radius 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

/-- Point A -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B -/
def B : ℝ × ℝ := (1, 0)

/-- Distance squared between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The expression to be maximized and minimized -/
def d (P : ℝ × ℝ) : ℝ :=
  distanceSquared P A + distanceSquared P B

theorem circle_max_min_distances :
  (∀ P ∈ C, d P ≤ 74) ∧ (∀ P ∈ C, d P ≥ 34) ∧ (∃ P ∈ C, d P = 74) ∧ (∃ P ∈ C, d P = 34) := by
  sorry

end NUMINAMATH_CALUDE_circle_max_min_distances_l301_30135


namespace NUMINAMATH_CALUDE_solution_characterization_l301_30191

def f (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.2, 5 * p.2 - p.1)

def h (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.2, p.1)

def solution_set : Set (ℕ × ℕ) :=
  {(1, 2), (1, 3), (2, 1), (3, 1)} ∪
  {p | ∃ n : ℕ, p = Nat.iterate f n (1, 2) ∨ p = Nat.iterate f n (1, 3)} ∪
  {p | ∃ n : ℕ, p = h (Nat.iterate f n (1, 2)) ∨ p = h (Nat.iterate f n (1, 3))}

theorem solution_characterization :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (x^2 + y^2 - 5*x*y + 5 = 0 ↔ (x, y) ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l301_30191


namespace NUMINAMATH_CALUDE_exists_similar_package_with_ten_boxes_l301_30112

/-- Represents a rectangular box with dimensions a, b, and c -/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Represents a package containing boxes -/
structure Package where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

/-- Defines geometric similarity between a package and a box -/
def geometricallySimilar (p : Package) (b : Box) : Prop :=
  ∃ k : ℝ, k > 0 ∧ p.x = k * b.a ∧ p.y = k * b.b ∧ p.z = k * b.c

/-- Defines if a package can contain exactly 10 boxes -/
def canContainTenBoxes (p : Package) (b : Box) : Prop :=
  (p.x = 10 * b.a ∧ p.y = b.b ∧ p.z = b.c) ∨
  (p.x = 5 * b.a ∧ p.y = 2 * b.b ∧ p.z = b.c)

/-- Theorem stating that there exists a package geometrically similar to a box and containing 10 boxes -/
theorem exists_similar_package_with_ten_boxes (b : Box) :
  ∃ p : Package, geometricallySimilar p b ∧ canContainTenBoxes p b := by
  sorry

end NUMINAMATH_CALUDE_exists_similar_package_with_ten_boxes_l301_30112


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l301_30145

/-- Given a cube with edge length 3 cm, if we cut out 6 smaller cubes each with edge length 1 cm,
    the remaining volume is 21 cm³. -/
theorem remaining_cube_volume :
  let large_cube_edge : ℝ := 3
  let small_cube_edge : ℝ := 1
  let num_faces : ℕ := 6
  let original_volume := large_cube_edge ^ 3
  let cut_out_volume := num_faces * small_cube_edge ^ 3
  original_volume - cut_out_volume = 21 := by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l301_30145


namespace NUMINAMATH_CALUDE_clock_overlaps_in_24_hours_l301_30163

/-- Represents a clock with an hour hand and a minute hand -/
structure Clock :=
  (hour_revolutions : ℕ)
  (minute_revolutions : ℕ)

/-- The number of overlaps between the hour and minute hands -/
def overlaps (c : Clock) : ℕ := c.minute_revolutions - c.hour_revolutions

theorem clock_overlaps_in_24_hours :
  ∃ (c : Clock), c.hour_revolutions = 2 ∧ c.minute_revolutions = 24 ∧ overlaps c = 22 :=
sorry

end NUMINAMATH_CALUDE_clock_overlaps_in_24_hours_l301_30163


namespace NUMINAMATH_CALUDE_strictly_increasing_f_implies_a_nonneg_l301_30132

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- State the theorem
theorem strictly_increasing_f_implies_a_nonneg 
  (h : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_f_implies_a_nonneg_l301_30132


namespace NUMINAMATH_CALUDE_ali_monday_flowers_l301_30119

/-- The number of flowers Ali sold on Monday -/
def monday_flowers : ℕ := sorry

/-- The number of flowers Ali sold on Tuesday -/
def tuesday_flowers : ℕ := 8

/-- The number of flowers Ali sold on Friday -/
def friday_flowers : ℕ := 2 * monday_flowers

/-- The total number of flowers Ali sold -/
def total_flowers : ℕ := 20

theorem ali_monday_flowers : 
  monday_flowers + tuesday_flowers + friday_flowers = total_flowers → monday_flowers = 4 := by
sorry

end NUMINAMATH_CALUDE_ali_monday_flowers_l301_30119


namespace NUMINAMATH_CALUDE_sqrt_of_four_l301_30170

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l301_30170


namespace NUMINAMATH_CALUDE_pharmaceutical_royalties_l301_30148

theorem pharmaceutical_royalties (first_royalties second_royalties second_sales : ℝ)
  (ratio_decrease : ℝ) (h1 : first_royalties = 8)
  (h2 : second_royalties = 9) (h3 : second_sales = 108)
  (h4 : ratio_decrease = 0.7916666666666667) :
  ∃ first_sales : ℝ,
    first_sales = 20 ∧
    (first_royalties / first_sales) - (second_royalties / second_sales) =
      ratio_decrease * (first_royalties / first_sales) :=
by sorry

end NUMINAMATH_CALUDE_pharmaceutical_royalties_l301_30148


namespace NUMINAMATH_CALUDE_profit_share_difference_l301_30134

/-- Given the investments and profit share of B, calculate the difference between profit shares of A and C -/
theorem profit_share_difference (investment_A investment_B investment_C profit_B : ℕ) : 
  investment_A = 8000 →
  investment_B = 10000 →
  investment_C = 12000 →
  profit_B = 1900 →
  ∃ (profit_A profit_C : ℕ),
    profit_A * investment_B = profit_B * investment_A ∧
    profit_C * investment_B = profit_B * investment_C ∧
    profit_C - profit_A = 760 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_difference_l301_30134


namespace NUMINAMATH_CALUDE_determinant_of_roots_l301_30136

theorem determinant_of_roots (s p q : ℝ) (a b c : ℝ) : 
  a^3 + s*a^2 + p*a + q = 0 → 
  b^3 + s*b^2 + p*b + q = 0 → 
  c^3 + s*c^2 + p*c + q = 0 → 
  Matrix.det !![a, b, c; b, c, a; c, a, b] = -s*(s^2 - 3*p) := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_roots_l301_30136


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l301_30189

theorem triangle_sine_inequality (A B C : Real) (h_triangle : A + B + C = Real.pi) :
  -2 < Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ (3 / 2) * Real.sqrt 3 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = (3 / 2) * Real.sqrt 3 ↔
   A = 7 * Real.pi / 9 ∧ B = Real.pi / 9 ∧ C = Real.pi / 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l301_30189


namespace NUMINAMATH_CALUDE_two_car_garage_count_l301_30166

theorem two_car_garage_count (total : ℕ) (pool : ℕ) (both : ℕ) (neither : ℕ) :
  total = 70 →
  pool = 40 →
  both = 35 →
  neither = 15 →
  ∃ garage : ℕ, garage = 50 ∧ garage + pool - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_two_car_garage_count_l301_30166


namespace NUMINAMATH_CALUDE_mona_unique_players_l301_30180

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (groups : ℕ) (players_per_group : ℕ) (repeated_players : ℕ) : ℕ :=
  groups * players_per_group - repeated_players

/-- Theorem stating the number of unique players Mona grouped with --/
theorem mona_unique_players :
  let groups : ℕ := 9
  let players_per_group : ℕ := 4
  let repeated_players : ℕ := 3
  unique_players groups players_per_group repeated_players = 33 := by
  sorry

#eval unique_players 9 4 3

end NUMINAMATH_CALUDE_mona_unique_players_l301_30180


namespace NUMINAMATH_CALUDE_abs_diff_plus_smaller_l301_30167

theorem abs_diff_plus_smaller (a b : ℝ) (h : a > b) : |a - b| + b = a := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_plus_smaller_l301_30167


namespace NUMINAMATH_CALUDE_bingley_bracelets_l301_30121

theorem bingley_bracelets (initial : ℕ) : 
  let kellys_bracelets : ℕ := 16
  let received : ℕ := kellys_bracelets / 4
  let total : ℕ := initial + received
  let given_away : ℕ := total / 3
  let remaining : ℕ := total - given_away
  remaining = 6 → initial = 5 := by sorry

end NUMINAMATH_CALUDE_bingley_bracelets_l301_30121


namespace NUMINAMATH_CALUDE_symmetry_axis_of_f_l301_30186

/-- The quadratic function f(x) = -2(x-1)^2 + 3 -/
def f (x : ℝ) : ℝ := -2 * (x - 1)^2 + 3

/-- The axis of symmetry for the quadratic function f -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of f(x) = -2(x-1)^2 + 3 is x = 1 -/
theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by
  sorry


end NUMINAMATH_CALUDE_symmetry_axis_of_f_l301_30186


namespace NUMINAMATH_CALUDE_target_compound_has_one_iodine_l301_30129

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  nitrogen : ℕ
  hydrogen : ℕ
  iodine : ℕ

/-- Atomic weights of elements -/
def atomic_weight : Fin 3 → ℝ
| 0 => 14.01  -- Nitrogen
| 1 => 1.01   -- Hydrogen
| 2 => 126.90 -- Iodine

/-- Calculate the molecular weight of a compound -/
def molecular_weight (c : Compound) : ℝ :=
  c.nitrogen * atomic_weight 0 + c.hydrogen * atomic_weight 1 + c.iodine * atomic_weight 2

/-- The compound in question -/
def target_compound : Compound := { nitrogen := 1, hydrogen := 4, iodine := 1 }

/-- Theorem stating that the target compound has exactly one iodine atom -/
theorem target_compound_has_one_iodine :
  molecular_weight target_compound = 145 ∧ target_compound.iodine = 1 := by
  sorry

end NUMINAMATH_CALUDE_target_compound_has_one_iodine_l301_30129


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_l301_30110

/-- Given a rectangle ACDE, a point B on AC, a point F on AE, and an equilateral triangle CEF,
    prove that the area of ACDE + CEF - ABF is 1100 + (225 * Real.sqrt 3) / 4 -/
theorem rectangle_triangle_area (A B C D E F : ℝ × ℝ) : 
  let AC : ℝ := 40
  let AE : ℝ := 30
  let AB : ℝ := AC / 3
  let AF : ℝ := AE / 2
  let area_ACDE : ℝ := AC * AE
  let area_CEF : ℝ := (Real.sqrt 3 / 4) * AF^2
  let area_ABF : ℝ := (1 / 2) * AB * AF
  area_ACDE + area_CEF - area_ABF = 1100 + (225 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_l301_30110


namespace NUMINAMATH_CALUDE_triangle_area_l301_30198

/-- The area of a triangle with base 7 units and height 3 units is 10.5 square units. -/
theorem triangle_area : 
  let base : ℝ := 7
  let height : ℝ := 3
  let area : ℝ := (1/2) * base * height
  area = 10.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l301_30198


namespace NUMINAMATH_CALUDE_solution_set_correct_l301_30104

/-- The solution set of the equation 3sin(x) = 1 + cos(2x) -/
def SolutionSet : Set ℝ :=
  {x | ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)}

/-- The original equation -/
def OriginalEquation (x : ℝ) : Prop :=
  3 * Real.sin x = 1 + Real.cos (2 * x)

theorem solution_set_correct :
  ∀ x : ℝ, x ∈ SolutionSet ↔ OriginalEquation x := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l301_30104


namespace NUMINAMATH_CALUDE_two_roots_implies_c_value_l301_30194

/-- A cubic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

/-- The number of roots of f for a given c -/
def num_roots (c : ℝ) : ℕ := sorry

/-- Theorem stating that if f has exactly two roots, then c is either -2 or 2 -/
theorem two_roots_implies_c_value (c : ℝ) :
  num_roots c = 2 → c = -2 ∨ c = 2 := by sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_value_l301_30194


namespace NUMINAMATH_CALUDE_max_contribution_l301_30177

theorem max_contribution 
  (total : ℝ) 
  (num_people : ℕ) 
  (min_contribution : ℝ) 
  (h1 : total = 20) 
  (h2 : num_people = 12) 
  (h3 : min_contribution = 1) 
  (h4 : ∀ p, p ≤ num_people → p • min_contribution ≤ total) : 
  ∃ max_contrib : ℝ, max_contrib = 9 ∧ 
    ∀ individual_contrib, 
      individual_contrib ≤ max_contrib ∧ 
      (num_people - 1) • min_contribution + individual_contrib = total :=
sorry

end NUMINAMATH_CALUDE_max_contribution_l301_30177


namespace NUMINAMATH_CALUDE_balloon_count_l301_30171

/-- Represents the number of balloons of each color and their arrangement --/
structure BalloonArrangement where
  red : Nat
  yellow : Nat
  blue : Nat
  yellow_spaces : Nat
  yellow_unfilled : Nat

/-- Calculates the total number of balloons --/
def total_balloons (arrangement : BalloonArrangement) : Nat :=
  arrangement.red + arrangement.yellow + arrangement.blue

/-- Theorem stating the correct number of yellow and blue balloons --/
theorem balloon_count (arrangement : BalloonArrangement) 
  (h1 : arrangement.red = 40)
  (h2 : arrangement.yellow_spaces = arrangement.red - 1)
  (h3 : arrangement.yellow_unfilled = 3)
  (h4 : arrangement.yellow = arrangement.yellow_spaces + arrangement.yellow_unfilled)
  (h5 : arrangement.blue = total_balloons arrangement - 1) :
  arrangement.yellow = 42 ∧ arrangement.blue = 81 := by
  sorry

#check balloon_count

end NUMINAMATH_CALUDE_balloon_count_l301_30171


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_2004_l301_30115

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2004 -/
def sum_of_last_two_digits : ℕ :=
  let n : ℕ := 8^2004
  let tens_digit : ℕ := (n / 10) % 10
  let units_digit : ℕ := n % 10
  tens_digit + units_digit

theorem sum_of_last_two_digits_of_8_pow_2004 :
  sum_of_last_two_digits = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_2004_l301_30115


namespace NUMINAMATH_CALUDE_rotation_of_D_around_E_l301_30124

-- Define the points
def D : ℝ × ℝ := (3, 2)
def E : ℝ × ℝ := (6, 5)
def F : ℝ × ℝ := (6, 2)

-- Define the rotation function
def rotate180AroundPoint (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

-- Theorem statement
theorem rotation_of_D_around_E :
  rotate180AroundPoint E D = (9, 8) := by sorry

end NUMINAMATH_CALUDE_rotation_of_D_around_E_l301_30124


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l301_30138

/-- A function to check if a number is a palindrome in a given base -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function to convert a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome : 
  ∀ k : ℕ, k > 15 → isPalindromeInBase k 3 → isPalindromeInBase k 5 → k ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l301_30138


namespace NUMINAMATH_CALUDE_square_area_l301_30153

theorem square_area (x : ℝ) : 
  (6 * x - 18 = 3 * x + 9) → 
  (6 * x - 18)^2 = 1296 := by
sorry

end NUMINAMATH_CALUDE_square_area_l301_30153


namespace NUMINAMATH_CALUDE_fermat_for_small_exponents_l301_30159

theorem fermat_for_small_exponents (x y z n : ℕ) (h : n ≥ z) :
  x^n + y^n ≠ z^n := by
  sorry

end NUMINAMATH_CALUDE_fermat_for_small_exponents_l301_30159


namespace NUMINAMATH_CALUDE_linear_equation_exponent_sum_l301_30143

theorem linear_equation_exponent_sum (a b : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, 4*x^(a+b) - 3*y^(3*a+2*b-4) = k*x + m*y + 2) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_sum_l301_30143


namespace NUMINAMATH_CALUDE_fraction_equality_l301_30181

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l301_30181


namespace NUMINAMATH_CALUDE_sum_unchanged_l301_30182

theorem sum_unchanged (a b c : ℤ) (h : a + b + c = 1281) :
  (a - 329) + (b + 401) + (c - 72) = 1281 := by
sorry

end NUMINAMATH_CALUDE_sum_unchanged_l301_30182


namespace NUMINAMATH_CALUDE_complex_fourth_power_equality_l301_30142

theorem complex_fourth_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4) : b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_equality_l301_30142


namespace NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l301_30117

theorem smallest_y_in_arithmetic_series (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- all terms are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- arithmetic series condition
  x * y * z = 125 →  -- product condition
  y ≥ 5 ∧ ∀ y' : ℝ, (∃ x' z' : ℝ, x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
    (∃ d' : ℝ, x' = y' - d' ∧ z' = y' + d') ∧ 
    x' * y' * z' = 125) → y' ≥ 5 := by
  sorry

#check smallest_y_in_arithmetic_series

end NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l301_30117


namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l301_30120

/-- Cost of Plan X in cents for m megabytes -/
def cost_x (m : ℕ) : ℕ := 15 * m

/-- Cost of Plan Y in cents for m megabytes -/
def cost_y (m : ℕ) : ℕ := 2500 + 7 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective than Plan X -/
def min_megabytes : ℕ := 313

theorem plan_y_more_cost_effective :
  ∀ m : ℕ, m ≥ min_megabytes → cost_y m < cost_x m ∧
  ∀ n : ℕ, n < min_megabytes → cost_y n ≥ cost_x n :=
by sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l301_30120


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_integers_l301_30144

theorem largest_of_four_consecutive_integers (a b c d : ℕ) : 
  a > 0 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ a * b * c * d = 840 → d = 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_integers_l301_30144


namespace NUMINAMATH_CALUDE_n_sum_of_squares_l301_30183

theorem n_sum_of_squares (n : ℕ) (h1 : n > 2) 
  (h2 : ∃ (x : ℕ), n^2 = (x + 1)^3 - x^3) : 
  (∃ (a b : ℕ), n = a^2 + b^2) ∧ 
  (∃ (m : ℕ), m > 2 ∧ (∃ (y : ℕ), m^2 = (y + 1)^3 - y^3) ∧ (∃ (c d : ℕ), m = c^2 + d^2)) :=
by sorry

end NUMINAMATH_CALUDE_n_sum_of_squares_l301_30183


namespace NUMINAMATH_CALUDE_quarterback_no_throw_percentage_l301_30156

/-- Given a quarterback's statistics in a game, calculate the percentage of time he doesn't throw a pass. -/
theorem quarterback_no_throw_percentage 
  (total_attempts : ℕ) 
  (sacks : ℕ) 
  (h1 : total_attempts = 80) 
  (h2 : sacks = 12) 
  (h3 : 2 * sacks = total_attempts - (total_attempts - 2 * sacks)) : 
  (2 * sacks : ℚ) / total_attempts = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_quarterback_no_throw_percentage_l301_30156


namespace NUMINAMATH_CALUDE_movie_screening_attendance_l301_30169

theorem movie_screening_attendance (total_guests : ℕ) 
  (h1 : total_guests = 50)
  (h2 : ∃ women : ℕ, women = total_guests / 2)
  (h3 : ∃ men : ℕ, men = 15)
  (h4 : ∃ children : ℕ, children = total_guests - (total_guests / 2 + 15))
  (h5 : ∃ men_left : ℕ, men_left = 15 / 5)
  (h6 : ∃ children_left : ℕ, children_left = 4) :
  total_guests - (15 / 5 + 4) = 43 := by
sorry


end NUMINAMATH_CALUDE_movie_screening_attendance_l301_30169


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l301_30168

/-- Given Isabella's initial and final hair lengths, prove that her hair growth is 6 inches. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l301_30168


namespace NUMINAMATH_CALUDE_annie_pays_36_for_12kg_l301_30172

-- Define the price function for oranges
def price (mass : ℝ) : ℝ := sorry

-- Define the given conditions
axiom price_proportional : ∃ k : ℝ, ∀ m : ℝ, price m = k * m
axiom paid_36_for_12kg : price 12 = 36

-- Theorem to prove
theorem annie_pays_36_for_12kg : price 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_annie_pays_36_for_12kg_l301_30172


namespace NUMINAMATH_CALUDE_system_solution_l301_30111

theorem system_solution :
  ∀ x y : ℕ+,
  (x.val * y.val + x.val + y.val = 71 ∧
   x.val^2 * y.val + x.val * y.val^2 = 880) →
  ((x.val = 11 ∧ y.val = 5) ∨ (x.val = 5 ∧ y.val = 11)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l301_30111


namespace NUMINAMATH_CALUDE_mike_seed_count_l301_30140

theorem mike_seed_count (seeds_left : ℕ) (seeds_to_left : ℕ) (seeds_to_new : ℕ) 
  (h1 : seeds_to_left = 20)
  (h2 : seeds_left = 30)
  (h3 : seeds_to_new = 30) :
  seeds_left + seeds_to_left + 2 * seeds_to_left + seeds_to_new = 120 := by
  sorry

#check mike_seed_count

end NUMINAMATH_CALUDE_mike_seed_count_l301_30140


namespace NUMINAMATH_CALUDE_division_problem_l301_30108

theorem division_problem (dividend quotient divisor remainder x : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  dividend = 113 →
  divisor = 3 * remainder + x →
  dividend = divisor * quotient + remainder →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l301_30108


namespace NUMINAMATH_CALUDE_greatest_third_term_in_arithmetic_sequence_l301_30179

theorem greatest_third_term_in_arithmetic_sequence :
  ∀ (a d : ℕ),
  a > 0 →
  d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 65 →
  (a + 2*d) = 13 ∧ ∀ (b e : ℕ), b > 0 → e > 0 → 
  b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 65 →
  (b + 2*e) ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_term_in_arithmetic_sequence_l301_30179


namespace NUMINAMATH_CALUDE_ditch_length_greater_than_70_l301_30101

/-- Represents a square field with irrigation ditches -/
structure IrrigatedField where
  side_length : ℝ
  ditch_length : ℝ
  max_distance_to_ditch : ℝ

/-- Theorem stating that the total length of ditches in the irrigated field is greater than 70 units -/
theorem ditch_length_greater_than_70 (field : IrrigatedField) 
  (h1 : field.side_length = 12)
  (h2 : field.max_distance_to_ditch ≤ 1) :
  field.ditch_length > 70 := by
  sorry

end NUMINAMATH_CALUDE_ditch_length_greater_than_70_l301_30101


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l301_30122

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Calculates the numeric value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

theorem correct_subtraction_result : ∀ (minuend subtrahend : TwoDigitNumber),
  minuend.units = 3 →
  (minuend.value - 3 + 5) - 25 = 60 →
  subtrahend.value = 52 →
  minuend.value - subtrahend.value = 31 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l301_30122


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l301_30165

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(14 ∣ (427398 - y))) ∧ 
  (14 ∣ (427398 - x)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l301_30165


namespace NUMINAMATH_CALUDE_circle_probability_theorem_l301_30113

theorem circle_probability_theorem (R : ℝ) (h : R = 4) :
  let outer_circle_area := π * R^2
  let inner_circle_radius := R - 3
  let inner_circle_area := π * inner_circle_radius^2
  (inner_circle_area / outer_circle_area) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_circle_probability_theorem_l301_30113


namespace NUMINAMATH_CALUDE_wang_gang_seat_location_l301_30100

/-- Represents a seat in a classroom -/
structure Seat where
  row : Nat
  column : Nat

/-- Represents a classroom -/
structure Classroom where
  rows : Nat
  columns : Nat

/-- Checks if a seat is valid for a given classroom -/
def is_valid_seat (c : Classroom) (s : Seat) : Prop :=
  s.row ≤ c.rows ∧ s.column ≤ c.columns

theorem wang_gang_seat_location (c : Classroom) (s : Seat) :
  c.rows = 7 ∧ c.columns = 8 ∧ s = Seat.mk 5 8 ∧ is_valid_seat c s →
  s.row = 5 ∧ s.column = 8 := by
  sorry

end NUMINAMATH_CALUDE_wang_gang_seat_location_l301_30100


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l301_30106

theorem opposite_of_negative_2023 : 
  (-((-2023 : ℝ)) = (2023 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l301_30106


namespace NUMINAMATH_CALUDE_oplus_comm_l301_30126

def oplus (a b : ℕ+) : ℕ+ := a ^ b.val + b ^ a.val

theorem oplus_comm (a b : ℕ+) : oplus a b = oplus b a := by
  sorry

end NUMINAMATH_CALUDE_oplus_comm_l301_30126
